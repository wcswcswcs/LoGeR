# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
import os
import warnings

from torch import Tensor
from torch import nn
import torch

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        # warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")


# Cache for block masks to avoid recreation
_BLOCK_MASK_CACHE = {}


def _compact_kv_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    source_keep_mask: Tensor,
    attention_mass_stats: list | None = None,
    attention_mass_max_queries: int = 512,
) -> Tensor:
    """Run SDPA with per-sample compacted K/V source tokens.

    This keeps all query rows and only removes selected key/value source
    columns.  Different samples may keep different source-token counts, so we
    loop over the batch dimension instead of padding back to a dense mask.
    """

    if source_keep_mask.ndim != 2:
        raise ValueError(f"compact_kv source_keep_mask must be [B,N], got {tuple(source_keep_mask.shape)}")
    if int(source_keep_mask.shape[0]) != int(q.shape[0]) or int(source_keep_mask.shape[1]) != int(k.shape[2]):
        raise ValueError(
            "compact_kv source_keep_mask shape mismatch: "
            f"mask={tuple(source_keep_mask.shape)} q={tuple(q.shape)} k={tuple(k.shape)}"
        )
    source_keep_mask = source_keep_mask.to(device=q.device, dtype=torch.bool)
    if attention_mass_stats is not None:
        with torch.no_grad():
            removed_before_vals = []
            retained_before_vals = []
            removed_tokens = []
            kept_tokens = []
            query_samples = []
            head_dim = max(1, int(q.shape[-1]))
            scale = 1.0 / math.sqrt(float(head_dim))
            max_q = max(1, int(attention_mass_max_queries))
            for b in range(int(q.shape[0])):
                keep_b = source_keep_mask[b]
                removed_b = ~keep_b
                n_tokens = int(keep_b.numel())
                if n_tokens <= 0:
                    continue
                if int(removed_b.sum().item()) <= 0:
                    removed_before_vals.append(0.0)
                    retained_before_vals.append(1.0)
                    removed_tokens.append(0)
                    kept_tokens.append(int(keep_b.sum().item()))
                    query_samples.append(0)
                    continue
                if n_tokens > max_q:
                    q_idx = torch.linspace(0, n_tokens - 1, steps=max_q, device=q.device).round().long().unique()
                else:
                    q_idx = torch.arange(n_tokens, device=q.device)
                qb = q[b : b + 1, :, q_idx, :].float()
                kb = k[b : b + 1].float()
                scores = torch.matmul(qb, kb.transpose(-2, -1)) * scale
                attn_full = torch.softmax(scores, dim=-1)
                removed_mass = attn_full[..., removed_b].sum(dim=-1)
                retained_mass = attn_full[..., keep_b].sum(dim=-1)
                removed_before_vals.append(float(removed_mass.mean().item()))
                retained_before_vals.append(float(retained_mass.mean().item()))
                removed_tokens.append(int(removed_b.sum().item()))
                kept_tokens.append(int(keep_b.sum().item()))
                query_samples.append(int(q_idx.numel()))
            if removed_before_vals:
                attention_mass_stats.append({
                    "attention_mass_removed_before": float(torch.tensor(removed_before_vals).mean().item()),
                    "attention_mass_removed_after": 0.0,
                    "attention_mass_retained_before": float(torch.tensor(retained_before_vals).mean().item()),
                    "attention_mass_retained_after": 1.0,
                    "attention_mass_removed_tokens_mean": float(torch.tensor(removed_tokens, dtype=torch.float32).mean().item()),
                    "attention_mass_retained_tokens_mean": float(torch.tensor(kept_tokens, dtype=torch.float32).mean().item()),
                    "attention_mass_query_sample_tokens_mean": float(torch.tensor(query_samples, dtype=torch.float32).mean().item()),
                    "attention_mass_sampled": bool(max(removed_tokens or [0]) > 0),
                })
    outs = []
    for b in range(int(q.shape[0])):
        idx = torch.nonzero(source_keep_mask[b], as_tuple=False).reshape(-1)
        if idx.numel() == 0:
            idx = torch.arange(int(k.shape[2]), device=q.device)
        qb = q[b : b + 1]
        kb = k[b : b + 1, :, idx, :]
        vb = v[b : b + 1, :, idx, :]
        with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            outs.append(scaled_dot_product_attention(qb, kb, vb))
    return torch.cat(outs, dim=0)


def _sample_query_indices(n_tokens: int, max_queries: int, device: torch.device) -> Tensor:
    max_q = max(1, int(max_queries))
    if int(n_tokens) > max_q:
        return torch.linspace(0, int(n_tokens) - 1, steps=max_q, device=device).round().long().unique()
    return torch.arange(int(n_tokens), device=device)


def _append_source_soft_mass_stats(
    q: Tensor,
    k: Tensor,
    *,
    affected_mask: Tensor,
    base_attn_mask: Tensor | None,
    source_bias_values: Tensor | None,
    source_weights: Tensor | None,
    attention_mass_stats: list | None,
    attention_mass_max_queries: int,
    metric_type: str,
) -> None:
    """Record affected-source attention/effective-mass before and after soft control."""

    if attention_mass_stats is None:
        return
    if affected_mask.ndim != 2:
        raise ValueError(f"source soft affected_mask must be [B,N], got {tuple(affected_mask.shape)}")
    if int(affected_mask.shape[0]) != int(q.shape[0]) or int(affected_mask.shape[1]) != int(k.shape[2]):
        raise ValueError(
            "source soft affected_mask shape mismatch: "
            f"mask={tuple(affected_mask.shape)} q={tuple(q.shape)} k={tuple(k.shape)}"
        )
    affected_mask = affected_mask.to(device=q.device, dtype=torch.bool)
    base_bias = base_attn_mask.to(device=q.device, dtype=torch.float32) if base_attn_mask is not None else None
    bias = source_bias_values.to(device=q.device, dtype=torch.float32) if source_bias_values is not None else None
    weights = source_weights.to(device=q.device, dtype=torch.float32) if source_weights is not None else None
    if bias is not None and tuple(bias.shape) != tuple(affected_mask.shape):
        raise ValueError(f"source soft bias shape mismatch: bias={tuple(bias.shape)} mask={tuple(affected_mask.shape)}")
    if weights is not None and tuple(weights.shape) != tuple(affected_mask.shape):
        raise ValueError(f"source soft weight shape mismatch: weights={tuple(weights.shape)} mask={tuple(affected_mask.shape)}")

    with torch.no_grad():
        before_vals = []
        after_vals = []
        actual_after_vals = []
        retained_before_vals = []
        retained_after_vals = []
        affected_tokens = []
        source_weight_vals = []
        query_samples = []
        head_dim = max(1, int(q.shape[-1]))
        scale = 1.0 / math.sqrt(float(head_dim))
        for b in range(int(q.shape[0])):
            affected_b = affected_mask[b]
            n_tokens = int(affected_b.numel())
            if n_tokens <= 0:
                continue
            q_idx = _sample_query_indices(n_tokens, attention_mass_max_queries, q.device)
            qb = q[b : b + 1, :, q_idx, :].float()
            kb = k[b : b + 1].float()
            scores = torch.matmul(qb, kb.transpose(-2, -1)) * scale
            if base_bias is not None:
                scores = scores + base_bias[b : b + 1, :, q_idx, :]
            attn_full = torch.softmax(scores, dim=-1)
            affected_mass_before = attn_full[..., affected_b].sum(dim=-1)
            retained_mass_before = attn_full[..., ~affected_b].sum(dim=-1)

            if bias is not None:
                scores_after = scores + bias[b].reshape(1, 1, 1, n_tokens)
                attn_after = torch.softmax(scores_after, dim=-1)
                affected_mass_actual_after = attn_after[..., affected_b].sum(dim=-1)
                affected_mass_after = affected_mass_actual_after
                retained_mass_after = attn_after[..., ~affected_b].sum(dim=-1)
            elif weights is not None:
                weight_b = weights[b].reshape(1, 1, 1, n_tokens)
                affected_mass_actual_after = affected_mass_before
                affected_mass_after = (attn_full[..., affected_b] * weight_b[..., affected_b]).sum(dim=-1)
                retained_mass_after = retained_mass_before
            else:
                affected_mass_actual_after = affected_mass_before
                affected_mass_after = affected_mass_before
                retained_mass_after = retained_mass_before

            before_vals.append(float(affected_mass_before.mean().item()))
            after_vals.append(float(affected_mass_after.mean().item()))
            actual_after_vals.append(float(affected_mass_actual_after.mean().item()))
            retained_before_vals.append(float(retained_mass_before.mean().item()))
            retained_after_vals.append(float(retained_mass_after.mean().item()))
            affected_tokens.append(int(affected_b.sum().item()))
            query_samples.append(int(q_idx.numel()))
            if weights is not None and bool(affected_b.any()):
                source_weight_vals.append(float(weights[b][affected_b].mean().item()))

        if before_vals:
            attention_mass_stats.append({
                "attention_mass_metric": str(metric_type),
                "attention_mass_removed_before": float(torch.tensor(before_vals).mean().item()),
                "attention_mass_removed_after": float(torch.tensor(after_vals).mean().item()),
                "attention_mass_actual_after": float(torch.tensor(actual_after_vals).mean().item()),
                "attention_mass_retained_before": float(torch.tensor(retained_before_vals).mean().item()),
                "attention_mass_retained_after": float(torch.tensor(retained_after_vals).mean().item()),
                "attention_mass_removed_tokens_mean": float(torch.tensor(affected_tokens, dtype=torch.float32).mean().item()),
                "attention_mass_query_sample_tokens_mean": float(torch.tensor(query_samples, dtype=torch.float32).mean().item()),
                "source_value_weight_mean": (
                    float(torch.tensor(source_weight_vals).mean().item()) if source_weight_vals else None
                ),
                "attention_mass_sampled": bool(max(affected_tokens or [0]) > 0),
            })


def _source_soft_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    control: dict,
) -> Tensor:
    """Run SDPA with source-column soft bias or V-only attenuation."""

    affected_mask = control["affected_mask"].to(device=q.device, dtype=torch.bool)
    base_attn_mask = control.get("base_attn_mask")
    source_bias_values = control.get("source_bias_values")
    source_weights = control.get("source_weights")
    if base_attn_mask is not None:
        base_attn_mask = base_attn_mask.to(device=q.device, dtype=q.dtype)
    if source_bias_values is not None:
        source_bias_values = source_bias_values.to(device=q.device, dtype=q.dtype)
    if source_weights is not None:
        source_weights = source_weights.to(device=q.device, dtype=v.dtype)
    _append_source_soft_mass_stats(
        q,
        k,
        affected_mask=affected_mask,
        base_attn_mask=base_attn_mask,
        source_bias_values=source_bias_values,
        source_weights=source_weights,
        attention_mass_stats=control.get("attention_mass_stats"),
        attention_mass_max_queries=int(control.get("attention_mass_max_queries", 512) or 512),
        metric_type=str(control.get("attention_mass_metric", "source_soft")),
    )
    v_eff = v
    if source_weights is not None:
        v_eff = v * source_weights[:, None, :, None]
    attn_mask = base_attn_mask
    if source_bias_values is not None:
        source_bias = source_bias_values[:, None, None, :]
        attn_mask = source_bias if attn_mask is None else attn_mask + source_bias
    if attn_mask is None and q.dtype == torch.bfloat16:
        with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return scaled_dot_product_attention(q, k, v_eff)
    with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        return scaled_dot_product_attention(q, k, v_eff, attn_mask=attn_mask)


def get_causal_block_mask(P, B, H, M, N, device="cuda", _compile=True):
    """
    Get causal block mask with efficient caching based on logical parameters.
    
    Args:
        P: tokens per frame (image)
        B: batch size (not used in cache key since mask can be reused across batch sizes)
        H: number of heads
        M: query sequence length (num_frames * P)
        N: key sequence length (num_frames * P) 
        device: target device
        _compile: whether to compile
    
    Returns:
        Block mask where tokens within the same image can see each other,
        but tokens from different images can only see previous images.
    """
    if not FLEX_ATTENTION_AVAILABLE:
        return None
    
    # Create cache key based on logical parameters
    device_idx = device.index if hasattr(device, 'index') else 0
    cache_key = (P, H, M, N, device_idx, _compile)
    
    if cache_key in _BLOCK_MASK_CACHE:
        cached_mask = _BLOCK_MASK_CACHE[cache_key]
        return cached_mask
    
    # Create the score function
    # Tokens within the same frame can see each other
    # Tokens from frame i can see all tokens from frames 0 to i
    def causal_mask(b, h, q_idx, kv_idx):
        q_frame = q_idx // P
        kv_frame = kv_idx // P
        return q_frame >= kv_frame
    
    # Create new block mask
    block_mask = create_block_mask(causal_mask, B, H, M, N, device=device, _compile=_compile)
    
    # Cache it
    _BLOCK_MASK_CACHE[cache_key] = block_mask
    
    return block_mask


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    
class FlashAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]

        if q.dtype == torch.bfloat16:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(q, k, v)
        else:
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


"""
Following is written by GPT-4o
"""
class CrossAttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Separate projection layers for query, key, and value
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        B, N, C = query.shape
        _, M, _ = key.shape

        # Project query, key, and value
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        # Scale query
        q = q * self.scale

        # Compute attention scores
        attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, M)
        if attn_bias is not None:
            attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttentionRope(CrossAttentionRope):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(query, key, value, attn_bias)

        B, N, C = query.shape
        _, M, _ = key.shape

        # Project query, key, and value
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Compute memory-efficient attention
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.rope = rope

    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttentionRope(AttentionRope):
    def forward(self, x: Tensor, attn_bias=None, xpos=None, attn_mask=None) -> Tensor:
        compact_kv = attn_mask if isinstance(attn_mask, dict) and attn_mask.get("type") == "compact_kv" else None
        source_soft = attn_mask if isinstance(attn_mask, dict) and attn_mask.get("type") == "source_soft" else None
        if compact_kv is not None or source_soft is not None:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
            q, k, v = [qkv[:, :, i] for i in range(3)]
            q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

            if self.rope is not None:
                q = self.rope(q, xpos)
                k = self.rope(k, xpos)

            target_dtype = v.dtype
            if q.dtype != target_dtype:
                q = q.to(target_dtype)
            if k.dtype != target_dtype:
                k = k.to(target_dtype)

            if compact_kv is not None:
                x = _compact_kv_sdpa(
                    q,
                    k,
                    v,
                    compact_kv["source_keep_mask"],
                    compact_kv.get("attention_mass_stats"),
                    int(compact_kv.get("attention_mass_max_queries", 512) or 512),
                )
            else:
                x = _source_soft_sdpa(q, k, v, source_soft)
            x = x.transpose(1, 2).reshape([B, N, C])
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # If attn_mask is provided and flex_attention is available, use flex_attention
        if attn_mask is not None and FLEX_ATTENTION_AVAILABLE:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
            q, k, v = [qkv[:,:,i] for i in range(3)]
            q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

            if self.rope is not None:
                q = self.rope(q, xpos)
                k = self.rope(k, xpos)

            # Ensure all tensors have the same dtype
            target_dtype = v.dtype
            if q.dtype != target_dtype:
                q = q.to(target_dtype)
            if k.dtype != target_dtype:
                k = k.to(target_dtype)
            
            x = flex_attention(
                q, k, v,
                block_mask=attn_mask,
                scale=None,
                enable_gqa=False,
                return_lse=False
            )
            x = x.transpose(1, 2).reshape([B, N, C])
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        
        # Otherwise use xformers memory_efficient_attention
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, attn_bias=attn_bias, xpos=xpos, attn_mask=attn_mask)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        qkv = qkv.transpose(1, 3)
        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1).reshape(frame_num, 261, frame_num, 261).mean(dim=[1, 3]).sum(1)         # for frame attention matrix
        # global_valid_id = torch.where(score_matrix > 0)
        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class FlashAttentionRope(AttentionRope):
    def compute_kv(self, x: Tensor, xpos=None) -> tuple[Tensor, Tensor]:
        """Compute K, V for caching. Returns (K, V) after norm and RoPE."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        
        if self.rope is not None:
            k = self.rope(k, xpos)
        
        return k, v
    
    def forward_with_kv_cache(
        self, 
        x: Tensor, 
        k_cache: Tensor, 
        v_cache: Tensor,
        xpos=None, 
        xpos_cache=None,
        attn_mask=None
    ) -> Tensor:
        """Forward with pre-computed KV cache for history tokens.
        
        Args:
            x: Current tokens [B, N_curr, C]
            k_cache: Cached K from history [B, num_heads, N_hist, head_dim]
            v_cache: Cached V from history [B, num_heads, N_hist, head_dim]
            xpos: Position info for current tokens
            xpos_cache: Position info for cached tokens (unused, positions already applied)
            attn_mask: Optional attention mask
        
        Returns:
            Output for current tokens only [B, N_curr, C]
        """
        B, N_curr, C = x.shape
        
        # Compute Q, K, V for current tokens
        qkv = self.qkv(x).reshape(B, N_curr, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        
        # Concatenate cached KV with current KV
        # k_cache, v_cache: [B, num_heads, N_hist, head_dim]
        # k, v: [B, num_heads, N_curr, head_dim]
        k_full = torch.cat([k_cache, k], dim=2)
        v_full = torch.cat([v_cache, v], dim=2)
        
        # Compute attention.  SWA overlap control can pass a compact descriptor
        # instead of a dense mask; in that case we run the native full attention
        # first and then recompute only the affected overlap query rows.
        overlap_bias = attn_mask if isinstance(attn_mask, dict) and attn_mask.get("type") == "overlap_bias" else None
        dense_attn_mask = None if overlap_bias is not None else attn_mask
        is_float_mask = (
            dense_attn_mask is not None
            and torch.is_tensor(dense_attn_mask)
            and torch.is_floating_point(dense_attn_mask)
        )
        
        if dense_attn_mask is not None and FLEX_ATTENTION_AVAILABLE and not is_float_mask:
            target_dtype = v_full.dtype
            if q.dtype != target_dtype:
                q = q.to(target_dtype)
            if k_full.dtype != target_dtype:
                k_full = k_full.to(target_dtype)
            
            x = flex_attention(
                q, k_full, v_full,
                block_mask=dense_attn_mask,
                scale=None,
                enable_gqa=False,
                return_lse=False
            )
        else:
            if q.dtype == torch.bfloat16 and not is_float_mask:
                with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    x = scaled_dot_product_attention(q, k_full, v_full)
            else:
                with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                    x = scaled_dot_product_attention(q, k_full, v_full, attn_mask=dense_attn_mask)

        if overlap_bias is not None:
            qn = min(int(overlap_bias.get("query_tokens", 0)), int(q.shape[2]))
            source_start = max(0, int(overlap_bias.get("source_start", 0)))
            source_end = min(int(overlap_bias.get("source_end", source_start)), int(k_full.shape[2]))
            bias_values = overlap_bias.get("bias_values")
            block_size = max(1, int(overlap_bias.get("query_block_size", 256)))
            if qn > 0 and source_end > source_start and torch.is_tensor(bias_values):
                sn = source_end - source_start
                bias_values = bias_values.to(device=q.device, dtype=q.dtype)
                bias_values = bias_values[:, :qn, :sn]
                overlap_out = []
                for q0 in range(0, qn, block_size):
                    q1 = min(q0 + block_size, qn)
                    q_chunk = q[:, :, q0:q1, :]
                    local_bias = torch.zeros(
                        q.shape[0],
                        1,
                        q1 - q0,
                        k_full.shape[2],
                        device=q.device,
                        dtype=q.dtype,
                    )
                    local_bias[:, :, :, source_start:source_end] = bias_values[:, q0:q1, :].unsqueeze(1)
                    with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                        overlap_out.append(
                            scaled_dot_product_attention(q_chunk, k_full, v_full, attn_mask=local_bias)
                        )
                if overlap_out:
                    x = x.clone()
                    x[:, :, :qn, :] = torch.cat(overlap_out, dim=2)
        
        x = x.transpose(1, 2).reshape([B, N_curr, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: Tensor, attn_bias=None, xpos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        compact_kv = attn_mask if isinstance(attn_mask, dict) and attn_mask.get("type") == "compact_kv" else None
        source_soft = attn_mask if isinstance(attn_mask, dict) and attn_mask.get("type") == "source_soft" else None
        if compact_kv is not None or source_soft is not None:
            if compact_kv is not None:
                x = _compact_kv_sdpa(
                    q,
                    k,
                    v,
                    compact_kv["source_keep_mask"],
                    compact_kv.get("attention_mass_stats"),
                    int(compact_kv.get("attention_mass_max_queries", 512) or 512),
                )
            else:
                x = _source_soft_sdpa(q, k, v, source_soft)
            x = x.transpose(1, 2).reshape([B, N, C])
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # If attn_mask (block_mask) is provided and flex_attention is available, use it
        # If attn_mask (block_mask) is provided and flex_attention is available, use it
        # [MODIFIED] Check if attn_mask is a float tensor (bias). If so, skip flex_attention
        # because flex_attention typically expects a BlockMask or boolean mask.
        is_float_mask = (attn_mask is not None and torch.is_floating_point(attn_mask))
        
        if attn_mask is not None and FLEX_ATTENTION_AVAILABLE and not is_float_mask:
            # Ensure all tensors have the same dtype for flex_attention
            target_dtype = v.dtype
            if q.dtype != target_dtype:
                q = q.to(target_dtype)
            if k.dtype != target_dtype:
                k = k.to(target_dtype)
            
            x = flex_attention(
                q, k, v,
                block_mask=attn_mask,
                scale=None,  # flex_attention applies 1/sqrt(d) automatically
                enable_gqa=False,
                return_lse=False
            )
        else:
            # Use standard scaled_dot_product_attention
            if q.dtype == torch.bfloat16 and not is_float_mask:
                with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    x = scaled_dot_product_attention(q, k, v)
            else:
                # Fallback to MATH/EFFICIENT if using float mask or other dtypes
                with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                    x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def get_attn_score(blk_class, x, frame_num, token_length, xpos=None):
    x = blk_class.norm1(x)
    
    B, N, C = x.shape
    qkv = blk_class.attn.qkv(x).reshape(B, N, 3, blk_class.attn.num_heads, C // blk_class.attn.num_heads)
    
    qkv = qkv.transpose(1, 3)
    # q, k, v = unbind(qkv, 2)
    q, k, v = [qkv[:,:,i] for i in range(3)]
    q, k = blk_class.attn.q_norm(q).to(v.dtype), blk_class.attn.k_norm(k).to(v.dtype)

    if blk_class.attn.rope is not None:
        q = blk_class.attn.rope(q, xpos)
        k = blk_class.attn.rope(k, xpos)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    score = (q.permute(0, 2, 1, 3) * blk_class.attn.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1).reshape(B, frame_num, token_length, frame_num, token_length).mean(dim=[2, 4]).sum(-1)

    return score


from .prope import _prepare_apply_fns, _prepare_apply_fns_query
class PRopeFlashAttention(AttentionRope):
    def forward(self, x: Tensor, extrinsics, H, W, patch_h, patch_w, K=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=extrinsics,
            Ks=K,
            patches_x=patch_w,
            patches_y=patch_h,
            image_width=H,
            image_height=W,
        )
        q = apply_fn_q(q)
        k = apply_fn_kv(k)
        v = apply_fn_kv(v)

        if q.dtype == torch.bfloat16 and attn_mask is None:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(q, k, v)
        else:
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        x = apply_fn_o(x)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashCrossAttentionRope(CrossAttentionRope):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C)
            key: Tensor of shape (B, M, C)
            value: Tensor of shape (B, M, C),
        Returns:
            Tensor of shape (B, N, C),
        """
        B, N, C = query.shape
        _, M, _ = key.shape

        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)
        
        dropout_p = self.attn_drop.p if self.training else 0.0
        
        if q.dtype == torch.bfloat16:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=dropout_p
                )
        else:
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias, dropout_p=dropout_p
                )

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
