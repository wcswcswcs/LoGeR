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
from pathlib import Path

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
                source_tokens = int(keep_b.numel())
                query_tokens = int(q.shape[2])
                if source_tokens <= 0 or query_tokens <= 0:
                    continue
                if int(removed_b.sum().item()) <= 0:
                    removed_before_vals.append(0.0)
                    retained_before_vals.append(1.0)
                    removed_tokens.append(0)
                    kept_tokens.append(int(keep_b.sum().item()))
                    query_samples.append(0)
                    continue
                if query_tokens > max_q:
                    q_idx = torch.linspace(0, query_tokens - 1, steps=max_q, device=q.device).round().long().unique()
                else:
                    q_idx = torch.arange(query_tokens, device=q.device)
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


def _source_attention_head_mask(control: dict, head_count: int, device: torch.device, dtype: torch.dtype) -> Tensor | None:
    raw = control.get("source_attention_head_indices")
    if raw is None or str(raw).strip() == "":
        return None
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        items = [raw]
    indices: list[int] = []
    for item in items:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < int(head_count):
            indices.append(idx)
    if not indices:
        return None
    mask = torch.zeros((int(head_count),), device=device, dtype=dtype)
    mask[torch.tensor(sorted(set(indices)), device=device, dtype=torch.long)] = 1.0
    return mask.reshape(1, int(head_count), 1, 1)


def _append_source_soft_mass_stats(
    q: Tensor,
    k: Tensor,
    *,
    affected_mask: Tensor,
    stable_anchor_mask: Tensor | None,
    query_mask: Tensor | None,
    base_attn_mask: Tensor | None,
    source_bias_values: Tensor | None,
    source_weights: Tensor | None,
    source_head_mask: Tensor | None,
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
    qmask = query_mask.to(device=q.device, dtype=torch.bool) if query_mask is not None else None
    if bias is not None and tuple(bias.shape) != tuple(affected_mask.shape):
        raise ValueError(f"source soft bias shape mismatch: bias={tuple(bias.shape)} mask={tuple(affected_mask.shape)}")
    if weights is not None and tuple(weights.shape) != tuple(affected_mask.shape):
        raise ValueError(f"source soft weight shape mismatch: weights={tuple(weights.shape)} mask={tuple(affected_mask.shape)}")
    if qmask is not None and (int(qmask.shape[0]) != int(q.shape[0]) or int(qmask.shape[1]) != int(q.shape[2])):
        raise ValueError(f"source soft query mask shape mismatch: query_mask={tuple(qmask.shape)} q={tuple(q.shape)}")
    stable_mask = None
    if stable_anchor_mask is not None:
        stable_mask = stable_anchor_mask.to(device=q.device, dtype=torch.bool)
        if tuple(stable_mask.shape) != tuple(affected_mask.shape):
            raise ValueError(
                "source soft stable_anchor_mask shape mismatch: "
                f"stable={tuple(stable_mask.shape)} mask={tuple(affected_mask.shape)}"
            )

    with torch.no_grad():
        before_vals = []
        after_vals = []
        actual_after_vals = []
        retained_before_vals = []
        retained_after_vals = []
        affected_tokens = []
        source_weight_vals = []
        query_mask_vals = []
        query_samples = []
        stable_before_vals = []
        stable_after_vals = []
        stable_actual_after_vals = []
        stable_tokens = []
        head_dim = max(1, int(q.shape[-1]))
        scale = 1.0 / math.sqrt(float(head_dim))
        query_tokens = int(q.shape[2])
        for b in range(int(q.shape[0])):
            affected_b = affected_mask[b]
            source_tokens = int(affected_b.numel())
            if source_tokens <= 0 or query_tokens <= 0:
                continue
            q_idx = _sample_query_indices(query_tokens, attention_mass_max_queries, q.device)
            qb = q[b : b + 1, :, q_idx, :].float()
            kb = k[b : b + 1].float()
            q_selected = qmask[b, q_idx] if qmask is not None else None
            scores = torch.matmul(qb, kb.transpose(-2, -1)) * scale
            if base_bias is not None:
                scores = scores + base_bias[b : b + 1, :, q_idx, :]
            attn_full = torch.softmax(scores, dim=-1)
            affected_mass_before = attn_full[..., affected_b].sum(dim=-1)
            retained_mass_before = attn_full[..., ~affected_b].sum(dim=-1)
            stable_b = stable_mask[b] if stable_mask is not None else None
            stable_mass_before = None
            if stable_b is not None and bool(stable_b.any()):
                stable_mass_before = attn_full[..., stable_b].sum(dim=-1)

            if bias is not None:
                bias_after = bias[b].reshape(1, 1, 1, source_tokens)
                if source_head_mask is not None:
                    bias_after = bias_after * source_head_mask.to(device=q.device, dtype=bias_after.dtype)
                if q_selected is not None:
                    bias_after = bias_after * q_selected.reshape(1, 1, -1, 1).to(dtype=bias_after.dtype)
                scores_after = scores + bias_after
                attn_after = torch.softmax(scores_after, dim=-1)
                affected_mass_actual_after = attn_after[..., affected_b].sum(dim=-1)
                affected_mass_after = affected_mass_actual_after
                retained_mass_after = attn_after[..., ~affected_b].sum(dim=-1)
                stable_mass_actual_after = attn_after[..., stable_b].sum(dim=-1) if stable_mass_before is not None else None
                stable_mass_after = stable_mass_actual_after
            elif weights is not None:
                weight_b = weights[b].reshape(1, 1, 1, source_tokens)
                affected_mass_actual_after = affected_mass_before
                affected_mass_after = (attn_full[..., affected_b] * weight_b[..., affected_b]).sum(dim=-1)
                retained_mass_after = retained_mass_before
                stable_mass_actual_after = stable_mass_before
                stable_mass_after = (
                    (attn_full[..., stable_b] * weight_b[..., stable_b]).sum(dim=-1)
                    if stable_mass_before is not None else None
                )
            else:
                affected_mass_actual_after = affected_mass_before
                affected_mass_after = affected_mass_before
                retained_mass_after = retained_mass_before
                stable_mass_actual_after = stable_mass_before
                stable_mass_after = stable_mass_before

            before_vals.append(float(affected_mass_before.mean().item()))
            after_vals.append(float(affected_mass_after.mean().item()))
            actual_after_vals.append(float(affected_mass_actual_after.mean().item()))
            retained_before_vals.append(float(retained_mass_before.mean().item()))
            retained_after_vals.append(float(retained_mass_after.mean().item()))
            affected_tokens.append(int(affected_b.sum().item()))
            query_samples.append(int(q_idx.numel()))
            if weights is not None and bool(affected_b.any()):
                source_weight_vals.append(float(weights[b][affected_b].mean().item()))
            if q_selected is not None:
                query_mask_vals.append(float(q_selected.float().mean().item()))
            if stable_mass_before is not None and stable_mass_after is not None:
                stable_before_vals.append(float(stable_mass_before.mean().item()))
                stable_after_vals.append(float(stable_mass_after.mean().item()))
                stable_actual_after_vals.append(float(stable_mass_actual_after.mean().item()))
                stable_tokens.append(int(stable_b.sum().item()))

        if before_vals:
            stable_before = float(torch.tensor(stable_before_vals).mean().item()) if stable_before_vals else None
            stable_after = float(torch.tensor(stable_after_vals).mean().item()) if stable_after_vals else None
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
                "attention_mass_query_mask_selected_ratio": (
                    float(torch.tensor(query_mask_vals).mean().item()) if query_mask_vals else None
                ),
                "attention_mass_stable_anchor_before": stable_before,
                "attention_mass_stable_anchor_after": stable_after,
                "attention_mass_stable_anchor_actual_after": (
                    float(torch.tensor(stable_actual_after_vals).mean().item()) if stable_actual_after_vals else None
                ),
                "attention_mass_stable_anchor_tokens_mean": (
                    float(torch.tensor(stable_tokens, dtype=torch.float32).mean().item()) if stable_tokens else None
                ),
                "attention_mass_stable_anchor_preservation_ratio": (
                    float(stable_after / max(abs(stable_before), 1e-12))
                    if stable_before is not None and stable_after is not None else None
                ),
                "attention_mass_sampled": bool(max(affected_tokens or [0]) > 0),
            })


def _dump_source_soft_attention_sample(
    q: Tensor,
    k: Tensor,
    *,
    affected_mask: Tensor,
    query_mask: Tensor | None,
    base_attn_mask: Tensor | None,
    source_bias_values: Tensor | None,
    source_weights: Tensor | None,
    control: dict,
) -> dict:
    """Persist a sampled raw QK attention map for audit-only cue construction."""

    dump_dir_text = str(control.get("source_attention_map_dump_dir", "") or "").strip()
    if not dump_dir_text:
        return {}
    if affected_mask.ndim != 2:
        return {"source_attention_map_dump_error": f"affected_mask_ndim={affected_mask.ndim}"}
    if int(affected_mask.shape[0]) != int(q.shape[0]) or int(affected_mask.shape[1]) != int(k.shape[2]):
        return {
            "source_attention_map_dump_error": (
                f"shape_mismatch mask={tuple(affected_mask.shape)} q={tuple(q.shape)} k={tuple(k.shape)}"
            )
        }

    try:
        max_queries = int(control.get("source_attention_map_dump_max_queries", 64) or 64)
        dump_dtype_name = str(control.get("source_attention_map_dump_dtype", "float16") or "float16").lower()
        dump_dtype = torch.float32 if dump_dtype_name == "float32" else torch.float16
        hook_path = str(control.get("source_attention_map_dump_hook_path", "unknown") or "unknown")
        layer = int(control.get("source_attention_map_dump_layer", -1))
        chunk = int(control.get("source_attention_map_dump_chunk_idx", -1))
        n_tokens = int(k.shape[2])
        full_query_marginal = bool(control.get("source_attention_map_dump_full_query_marginal", False))
        head_marginal = bool(control.get("source_attention_map_dump_head_marginal", False))
        source_head_mask = _source_attention_head_mask(control, int(q.shape[1]), q.device, torch.float32)
        query_block = max(1, int(control.get("source_attention_map_dump_query_block", 32) or 32))
        if full_query_marginal:
            head_dim = max(1, int(q.shape[-1]))
            scale = 1.0 / math.sqrt(float(head_dim))
            with torch.no_grad():
                source_sum_before = torch.zeros(
                    (int(q.shape[0]), int(k.shape[2])),
                    device=q.device,
                    dtype=torch.float32,
                )
                source_sum_after = (
                    torch.zeros_like(source_sum_before) if source_bias_values is not None else None
                )
                source_sum_before_head = (
                    torch.zeros(
                        (int(q.shape[0]), int(q.shape[1]), int(k.shape[2])),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    if head_marginal else None
                )
                source_sum_after_head = (
                    torch.zeros_like(source_sum_before_head)
                    if head_marginal and source_bias_values is not None else None
                )
                base_bias = (
                    base_attn_mask.to(device=q.device, dtype=torch.float32)
                    if base_attn_mask is not None else None
                )
                bias = (
                    source_bias_values.to(device=q.device, dtype=torch.float32)
                    if source_bias_values is not None else None
                )
                qmask = query_mask.to(device=q.device, dtype=torch.bool) if query_mask is not None else None
                kb = k.float()
                query_count = int(q.shape[2])
                denom = max(1, int(q.shape[1]) * query_count)
                for q0 in range(0, query_count, query_block):
                    q1 = min(query_count, q0 + query_block)
                    qb = q[:, :, q0:q1, :].float()
                    scores = torch.matmul(qb, kb.transpose(-2, -1)) * scale
                    if base_bias is not None:
                        scores = scores + base_bias[:, :, q0:q1, :]
                    attn_before = torch.softmax(scores, dim=-1)
                    source_sum_before += attn_before.sum(dim=(1, 2))
                    if source_sum_before_head is not None:
                        source_sum_before_head += attn_before.sum(dim=2)
                    if bias is not None and source_sum_after is not None:
                        bias_after = bias[:, None, None, :]
                        if source_head_mask is not None:
                            bias_after = bias_after * source_head_mask
                        if qmask is not None:
                            bias_after = bias_after * qmask[:, None, q0:q1, None].to(dtype=bias_after.dtype)
                        attn_after = torch.softmax(scores + bias_after, dim=-1)
                        source_sum_after += attn_after.sum(dim=(1, 2))
                        if source_sum_after_head is not None:
                            source_sum_after_head += attn_after.sum(dim=2)
                before_marginal = (source_sum_before / float(denom)).detach().cpu().to(dtype=dump_dtype)
                before_head_marginal = (
                    (source_sum_before_head / float(max(1, query_count))).detach().cpu().to(dtype=dump_dtype)
                    if source_sum_before_head is not None else None
                )
                affected_cpu_float = affected_mask.detach().cpu().float()

                payload = {
                    "schema": "acl2_v68_fullquery_source_attention_marginal_v1",
                    "chunk_idx": chunk,
                    "layer": layer,
                    "hook_path": hook_path,
                    "sampled_not_full_attention_map": False,
                    "full_query_source_marginal": True,
                    "pairwise_attention_matrix_stored": False,
                    "per_head_source_marginal": bool(before_head_marginal is not None),
                    "source_attention_head_limited": bool(source_head_mask is not None),
                    "source_attention_head_indices": str(control.get("source_attention_head_indices", "") or ""),
                    "query_axis_fully_covered": True,
                    "attention_source": "raw_qk_softmax_inside_source_soft_sdpa",
                    "source_attention_before_marginal": before_marginal,
                    "affected_mask": affected_mask.detach().cpu().to(dtype=torch.bool),
                    "query_mask": qmask.detach().cpu().to(dtype=torch.bool) if qmask is not None else None,
                    "source_attention_top_quantile": control.get("source_attention_top_quantile"),
                    "attention_mass_metric": control.get("attention_mass_metric"),
                    "dump_dtype": str(dump_dtype),
                    "q_shape": [int(v) for v in q.shape],
                    "k_shape": [int(v) for v in k.shape],
                    "query_block_size": int(query_block),
                    "full_query_count": query_count,
                    "source_token_count": int(k.shape[2]),
                    "source_attention_before_affected_mass_mean": float(
                        (before_marginal.float() * affected_cpu_float).sum(dim=-1).mean().item()
                    ),
                }
                if before_head_marginal is not None:
                    payload["source_attention_before_head_marginal"] = before_head_marginal
                    payload["source_attention_before_head_affected_mass_mean"] = float(
                        (before_head_marginal.float() * affected_cpu_float[:, None, :]).sum(dim=-1).mean().item()
                    )
                if source_sum_after is not None:
                    after_marginal = (source_sum_after / float(denom)).detach().cpu().to(dtype=dump_dtype)
                    payload["source_attention_after_bias_marginal"] = after_marginal
                    payload["source_attention_after_affected_mass_mean"] = float(
                        (after_marginal.float() * affected_cpu_float).sum(dim=-1).mean().item()
                    )
                    if source_sum_after_head is not None:
                        after_head_marginal = (
                            source_sum_after_head / float(max(1, query_count))
                        ).detach().cpu().to(dtype=dump_dtype)
                        payload["source_attention_after_bias_head_marginal"] = after_head_marginal
                        payload["source_attention_after_head_affected_mass_mean"] = float(
                            (after_head_marginal.float() * affected_cpu_float[:, None, :]).sum(dim=-1).mean().item()
                        )
                if source_weights is not None:
                    payload["source_weights"] = source_weights.detach().cpu().to(dtype=dump_dtype)

            out_dir = Path(dump_dir_text)
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_hook = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in hook_path)
            chunk_name = f"chunk_{chunk:03d}" if chunk >= 0 else "chunk_unknown"
            layer_name = f"layer_{layer:03d}" if layer >= 0 else "layer_unknown"
            out_path = out_dir / f"{chunk_name}_{layer_name}_{safe_hook}_source_attention_fullquery_marginal.pt"
            torch.save(payload, out_path)
            return {
                "source_attention_map_dump_path": str(out_path),
                "source_attention_map_dump_full_query_marginal": True,
                "source_attention_map_dump_head_marginal": bool(before_head_marginal is not None),
                "source_attention_map_dump_pairwise_matrix_stored": False,
                "source_attention_map_dump_shape": [int(v) for v in before_marginal.shape],
                "source_attention_map_dump_head_shape": (
                    [int(v) for v in before_head_marginal.shape] if before_head_marginal is not None else None
                ),
            }

        q_idx = _sample_query_indices(n_tokens, max_queries, q.device)
        head_dim = max(1, int(q.shape[-1]))
        scale = 1.0 / math.sqrt(float(head_dim))

        with torch.no_grad():
            qb = q[:, :, q_idx, :].float()
            kb = k.float()
            scores = torch.matmul(qb, kb.transpose(-2, -1)) * scale
            if base_attn_mask is not None:
                base_bias = base_attn_mask.to(device=q.device, dtype=torch.float32)
                scores = scores + base_bias[:, :, q_idx, :]
            attention_before = torch.softmax(scores, dim=-1)

            payload = {
                "schema": "acl2_v68_sampled_source_attention_map_v1",
                "chunk_idx": chunk,
                "layer": layer,
                "hook_path": hook_path,
                "sampled_not_full_attention_map": True,
                "pairwise_attention_matrix_stored": False,
                "sampled_pairwise_attention_matrix_stored": True,
                "attention_source": "raw_qk_softmax_inside_source_soft_sdpa",
                "query_indices": q_idx.detach().cpu().long(),
                "attention_before_control": attention_before.detach().cpu().to(dtype=dump_dtype),
                "affected_mask": affected_mask.detach().cpu().to(dtype=torch.bool),
                "source_attention_top_quantile": control.get("source_attention_top_quantile"),
                "attention_mass_metric": control.get("attention_mass_metric"),
                "dump_dtype": str(dump_dtype),
                "q_shape": [int(v) for v in q.shape],
                "k_shape": [int(v) for v in k.shape],
                "source_token_count": int(k.shape[2]),
            }
            if source_bias_values is not None:
                bias = source_bias_values.to(device=q.device, dtype=torch.float32)
                bias_after = bias[:, None, None, :]
                if source_head_mask is not None:
                    bias_after = bias_after * source_head_mask
                qmask = query_mask.to(device=q.device, dtype=torch.bool) if query_mask is not None else None
                if qmask is not None:
                    bias_after = bias_after * qmask[:, None, q_idx, None].to(dtype=bias_after.dtype)
                    payload["query_mask"] = qmask.detach().cpu().to(dtype=torch.bool)
                scores_after = scores + bias_after
                payload["attention_after_bias_control"] = torch.softmax(scores_after, dim=-1).detach().cpu().to(dtype=dump_dtype)
                payload["source_attention_head_limited"] = bool(source_head_mask is not None)
                payload["source_attention_head_indices"] = str(control.get("source_attention_head_indices", "") or "")
            if source_weights is not None:
                payload["source_weights"] = source_weights.detach().cpu().to(dtype=dump_dtype)

        out_dir = Path(dump_dir_text)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_hook = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in hook_path)
        chunk_name = f"chunk_{chunk:03d}" if chunk >= 0 else "chunk_unknown"
        layer_name = f"layer_{layer:03d}" if layer >= 0 else "layer_unknown"
        out_path = out_dir / f"{chunk_name}_{layer_name}_{safe_hook}_source_attention_sample.pt"
        torch.save(payload, out_path)
        return {
            "source_attention_map_dump_path": str(out_path),
            "source_attention_map_dump_sampled": True,
            "source_attention_map_dump_queries": int(q_idx.numel()),
            "source_attention_map_dump_shape": [int(v) for v in attention_before.shape],
        }
    except Exception as exc:  # pragma: no cover - diagnostic path must not break inference.
        return {"source_attention_map_dump_error": f"{type(exc).__name__}: {exc}"}


def _select_source_attention_top_mask(
    q: Tensor,
    k: Tensor,
    *,
    eligible_mask: Tensor,
    query_mask: Tensor | None,
    base_attn_mask: Tensor | None,
    attention_mass_max_queries: int,
    quantile: float,
    random_same_mass: bool,
    random_salt: int,
) -> Tensor:
    """Select source columns with the largest pre-control attention mass."""

    if eligible_mask.ndim != 2:
        raise ValueError(f"source attention eligible_mask must be [B,N], got {tuple(eligible_mask.shape)}")
    if int(eligible_mask.shape[0]) != int(q.shape[0]) or int(eligible_mask.shape[1]) != int(k.shape[2]):
        raise ValueError(
            "source attention eligible_mask shape mismatch: "
            f"mask={tuple(eligible_mask.shape)} q={tuple(q.shape)} k={tuple(k.shape)}"
        )
    eligible_mask = eligible_mask.to(device=q.device, dtype=torch.bool)
    qmask = query_mask.to(device=q.device, dtype=torch.bool) if query_mask is not None else None
    if qmask is not None and (int(qmask.shape[0]) != int(q.shape[0]) or int(qmask.shape[1]) != int(q.shape[2])):
        raise ValueError(f"source attention query_mask shape mismatch: query_mask={tuple(qmask.shape)} q={tuple(q.shape)}")
    base_bias = base_attn_mask.to(device=q.device, dtype=torch.float32) if base_attn_mask is not None else None
    out = torch.zeros_like(eligible_mask, dtype=torch.bool)
    head_dim = max(1, int(q.shape[-1]))
    scale = 1.0 / math.sqrt(float(head_dim))
    qv = float(quantile)
    qv = min(max(qv, 0.0), 1.0)
    with torch.no_grad():
        for b in range(int(q.shape[0])):
            valid = eligible_mask[b]
            if int(valid.sum().item()) <= 0:
                continue
            n_tokens = int(valid.numel())
            if qmask is not None:
                query_valid = torch.nonzero(qmask[b], as_tuple=False).reshape(-1)
                if int(query_valid.numel()) <= 0:
                    continue
                sampled_local = _sample_query_indices(int(query_valid.numel()), attention_mass_max_queries, q.device)
                q_idx = query_valid[sampled_local]
            else:
                q_idx = _sample_query_indices(n_tokens, attention_mass_max_queries, q.device)
            qb = q[b : b + 1, :, q_idx, :].float()
            kb = k[b : b + 1].float()
            scores = torch.matmul(qb, kb.transpose(-2, -1)) * scale
            if base_bias is not None:
                scores = scores + base_bias[b : b + 1, :, q_idx, :]
            attn_full = torch.softmax(scores, dim=-1)
            source_mass = attn_full.mean(dim=(0, 1, 2))
            valid_mass = source_mass[valid]
            if valid_mass.numel() <= 0:
                continue
            thr = torch.quantile(valid_mass.float(), qv)
            selected = valid & (source_mass >= thr)
            if random_same_mass:
                k_count = int(selected.sum().item())
                idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
                if k_count > 0 and idx.numel() > 0:
                    token_idx = torch.arange(n_tokens, device=q.device, dtype=torch.float32)
                    scores_rand = torch.frac(
                        torch.sin((token_idx + 1.0 + float(b + random_salt) * 97.0) * 12.9898)
                        * 43758.5453
                    )
                    top = torch.topk(scores_rand[idx], min(k_count, int(idx.numel()))).indices
                    selected = torch.zeros_like(valid, dtype=torch.bool)
                    selected[idx[top]] = True
            out[b] = selected
    return out


def _source_soft_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    control: dict,
) -> Tensor:
    """Run SDPA with source-column soft bias or V-only attenuation."""

    affected_mask = control["affected_mask"].to(device=q.device, dtype=torch.bool)
    query_mask = control.get("query_mask")
    if query_mask is not None:
        query_mask = query_mask.to(device=q.device, dtype=torch.bool)
        if int(query_mask.shape[0]) != int(q.shape[0]) or int(query_mask.shape[1]) != int(q.shape[2]):
            raise ValueError(f"source soft query_mask shape mismatch: query_mask={tuple(query_mask.shape)} q={tuple(q.shape)}")
    base_attn_mask = control.get("base_attn_mask")
    source_bias_values = control.get("source_bias_values")
    source_weights = control.get("source_weights")
    stable_anchor_mask = control.get("stable_anchor_mask")
    source_head_mask = _source_attention_head_mask(control, int(q.shape[1]), q.device, torch.float32)
    if base_attn_mask is not None:
        base_attn_mask = base_attn_mask.to(device=q.device, dtype=q.dtype)
    if source_bias_values is not None:
        source_bias_values = source_bias_values.to(device=q.device, dtype=q.dtype)
    if source_weights is not None:
        source_weights = source_weights.to(device=q.device, dtype=v.dtype)
    if stable_anchor_mask is not None:
        stable_anchor_mask = stable_anchor_mask.to(device=q.device, dtype=torch.bool)
    top_quantile = control.get("source_attention_top_quantile")
    if top_quantile is not None:
        eligible_mask = control.get("source_attention_top_eligible_mask", affected_mask)
        affected_mask = _select_source_attention_top_mask(
            q,
            k,
            eligible_mask=eligible_mask,
            query_mask=query_mask,
            base_attn_mask=base_attn_mask,
            attention_mass_max_queries=int(control.get("attention_mass_max_queries", 512) or 512),
            quantile=float(top_quantile),
            random_same_mass=bool(control.get("source_attention_top_random_same_mass", False)),
            random_salt=int(control.get("source_attention_top_random_salt", 0) or 0),
        )
        rho = float(control.get("source_attention_top_rho", 0.5) or 0.5)
        if bool(control.get("source_attention_top_boost_action", False)):
            source_bias_values = torch.log1p((rho * affected_mask.to(dtype=torch.float32)).clamp_min(0.0)).to(
                device=q.device, dtype=q.dtype
            )
        else:
            min_keep = float(control.get("source_attention_top_min_keep", 0.5) or 0.5)
            keep = (1.0 - rho * affected_mask.to(dtype=torch.float32)).clamp(min_keep, 1.0)
            source_bias_values = torch.log(keep.clamp_min(1e-4)).to(device=q.device, dtype=q.dtype)
        source_weights = None
    anchor_mask = control.get("source_attention_top_anchor_boost_mask")
    if anchor_mask is not None:
        anchor_mask = anchor_mask.to(device=q.device, dtype=torch.bool)
        anchor_score = control.get("source_attention_top_anchor_boost_score")
        if anchor_score is None:
            anchor_score = anchor_mask.to(dtype=torch.float32)
        else:
            anchor_score = anchor_score.to(device=q.device, dtype=torch.float32).clamp_min(0.0)
        anchor_rho = float(control.get("source_attention_top_anchor_boost_rho", 0.5) or 0.5)
        anchor_bias = torch.log1p((anchor_rho * anchor_score).clamp_min(0.0)).to(device=q.device, dtype=q.dtype)
        if source_bias_values is None:
            source_bias_values = torch.zeros_like(anchor_bias, device=q.device, dtype=q.dtype)
        source_bias_values = source_bias_values + torch.where(
            anchor_mask,
            anchor_bias,
            torch.zeros_like(source_bias_values),
        )
        stable_anchor_mask = anchor_mask if stable_anchor_mask is None else (stable_anchor_mask | anchor_mask)
        source_weights = None
    _append_source_soft_mass_stats(
        q,
        k,
        affected_mask=affected_mask,
        stable_anchor_mask=stable_anchor_mask,
        query_mask=query_mask,
        base_attn_mask=base_attn_mask,
        source_bias_values=source_bias_values,
        source_weights=source_weights,
        source_head_mask=source_head_mask,
        attention_mass_stats=control.get("attention_mass_stats"),
        attention_mass_max_queries=int(control.get("attention_mass_max_queries", 512) or 512),
        metric_type=str(control.get("attention_mass_metric", "source_soft")),
    )
    dump_info = _dump_source_soft_attention_sample(
        q,
        k,
        affected_mask=affected_mask,
        query_mask=query_mask,
        base_attn_mask=base_attn_mask,
        source_bias_values=source_bias_values,
        source_weights=source_weights,
        control=control,
    )
    stats = control.get("attention_mass_stats")
    if dump_info and isinstance(stats, list):
        if stats:
            stats[-1].update(dump_info)
        else:
            stats.append(dump_info)
    v_eff = v
    if source_weights is not None:
        v_eff = v * source_weights[:, None, :, None]
    attn_mask = base_attn_mask
    if source_bias_values is not None:
        source_bias = source_bias_values[:, None, None, :]
        if source_head_mask is not None:
            source_bias = source_bias * source_head_mask.to(device=q.device, dtype=source_bias.dtype)
        if query_mask is not None:
            source_bias = source_bias * query_mask[:, None, :, None].to(dtype=source_bias.dtype)
        attn_mask = source_bias if attn_mask is None else attn_mask + source_bias
    if attn_mask is None and q.dtype == torch.bfloat16:
        with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return scaled_dot_product_attention(q, k, v_eff)
    with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        return scaled_dot_product_attention(q, k, v_eff, attn_mask=attn_mask)


def _plain_sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    if q.dtype == torch.bfloat16:
        with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return scaled_dot_product_attention(q, k, v)
    with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        return scaled_dot_product_attention(q, k, v)


def _prev_ttt_anchor_query_soft_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    control: dict,
) -> Tensor:
    """Downweight or trace previous-anchor sources only for risky query rows."""

    stats = control.get("attention_mass_stats")
    stat_prefix = str(control.get("stat_prefix", "prev_ttt_anchor_query_soft") or "prev_ttt_anchor_query_soft")
    metric_type = str(control.get("attention_mass_metric", stat_prefix) or stat_prefix)
    trace_only = (
        str(control.get("mode", "") or "").strip().lower() == "trace_only"
        or bool(control.get("trace_only", False))
    )

    def stat_key(suffix: str) -> str:
        return f"{stat_prefix}_{suffix}"

    def plain() -> Tensor:
        return _plain_sdpa(q, k, v)

    history_tokens = min(max(0, int(control.get("history_tokens", 0) or 0)), int(k.shape[2]))
    stable_anchor_mask = control.get("stable_anchor_mask")
    if stable_anchor_mask is None or history_tokens <= 0:
        if isinstance(stats, list):
            stats.append({
                "attention_mass_metric": metric_type,
                stat_key("applied"): False,
                stat_key("trace_only"): bool(trace_only),
                stat_key("reason"): "missing_stable_anchor_mask_or_history",
            })
        return plain()

    stable_hist = stable_anchor_mask.to(device=q.device, dtype=torch.bool)
    if stable_hist.ndim == 1:
        stable_hist = stable_hist.reshape(1, -1).expand(int(q.shape[0]), -1)
    if stable_hist.ndim != 2 or int(stable_hist.shape[0]) != int(q.shape[0]):
        raise ValueError(
            f"{stat_prefix} stable_anchor_mask must be [B,N_hist], "
            f"got {tuple(stable_hist.shape)} q={tuple(q.shape)}"
        )
    if int(stable_hist.shape[1]) < history_tokens:
        pad = torch.zeros(
            int(stable_hist.shape[0]),
            history_tokens - int(stable_hist.shape[1]),
            device=q.device,
            dtype=torch.bool,
        )
        stable_hist = torch.cat([pad, stable_hist], dim=1)
    elif int(stable_hist.shape[1]) > history_tokens:
        stable_hist = stable_hist[:, -history_tokens:]

    source_seed_ids = control.get("source_seed_ids")
    seed_hist = None
    if source_seed_ids is not None:
        seed_hist = source_seed_ids.to(device=q.device, dtype=torch.long)
        if seed_hist.ndim == 1:
            seed_hist = seed_hist.reshape(1, -1).expand(int(q.shape[0]), -1)
        if seed_hist.ndim != 2 or int(seed_hist.shape[0]) != int(q.shape[0]):
            raise ValueError(
                f"{stat_prefix} source_seed_ids must be [B,N_hist], "
                f"got {tuple(seed_hist.shape)} q={tuple(q.shape)}"
            )
        if int(seed_hist.shape[1]) < history_tokens:
            seed_pad = torch.full(
                (int(seed_hist.shape[0]), history_tokens - int(seed_hist.shape[1])),
                -1,
                device=q.device,
                dtype=torch.long,
            )
            seed_hist = torch.cat([seed_pad, seed_hist], dim=1)
        elif int(seed_hist.shape[1]) > history_tokens:
            seed_hist = seed_hist[:, -history_tokens:]

    def _normalize_history_ids(raw: Tensor | None, name: str) -> Tensor | None:
        if raw is None:
            return None
        ids = raw.to(device=q.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1).expand(int(q.shape[0]), -1)
        if ids.ndim != 2 or int(ids.shape[0]) != int(q.shape[0]):
            raise ValueError(
                f"{stat_prefix} {name} must be [B,N_hist], "
                f"got {tuple(ids.shape)} q={tuple(q.shape)}"
            )
        if int(ids.shape[1]) < history_tokens:
            pad = torch.full(
                (int(ids.shape[0]), history_tokens - int(ids.shape[1])),
                -1,
                device=q.device,
                dtype=torch.long,
            )
            ids = torch.cat([pad, ids], dim=1)
        elif int(ids.shape[1]) > history_tokens:
            ids = ids[:, -history_tokens:]
        return ids

    def _normalize_query_ids(raw: Tensor | None, name: str) -> Tensor | None:
        if raw is None:
            return None
        ids = raw.to(device=q.device, dtype=torch.long)
        query_tokens = int(q.shape[2])
        if ids.ndim == 1:
            ids = ids.reshape(1, -1).expand(int(q.shape[0]), -1)
        if ids.ndim != 2 or int(ids.shape[0]) != int(q.shape[0]):
            raise ValueError(
                f"{stat_prefix} {name} must be [B,N_query], "
                f"got {tuple(ids.shape)} q={tuple(q.shape)}"
            )
        if int(ids.shape[1]) < query_tokens:
            pad = torch.full(
                (int(ids.shape[0]), query_tokens - int(ids.shape[1])),
                -1,
                device=q.device,
                dtype=torch.long,
            )
            ids = torch.cat([ids, pad], dim=1)
        elif int(ids.shape[1]) > query_tokens:
            ids = ids[:, :query_tokens]
        return ids

    source_instance_hist = _normalize_history_ids(control.get("source_instance_ids"), "source_instance_ids")
    query_seed_ids = _normalize_query_ids(control.get("query_seed_ids"), "query_seed_ids")
    query_instance_ids = _normalize_query_ids(control.get("query_instance_ids"), "query_instance_ids")
    direct_match_mode = str(control.get("direct_match_mode", "any") or "any").strip().lower()
    if direct_match_mode not in {"any", "same_seed", "same_masklet"}:
        direct_match_mode = "any"
    direct_match_available = (
        direct_match_mode == "any"
        or (direct_match_mode == "same_seed" and seed_hist is not None and query_seed_ids is not None)
        or (
            direct_match_mode == "same_masklet"
            and source_instance_hist is not None
            and query_instance_ids is not None
        )
    )
    direct_match_missing_reason = ""
    if not direct_match_available:
        direct_match_missing_reason = (
            "missing_seed_ids_for_same_seed"
            if direct_match_mode == "same_seed"
            else "missing_instance_ids_for_same_masklet"
        )

    affected_full = torch.zeros((int(q.shape[0]), int(k.shape[2])), device=q.device, dtype=torch.bool)
    affected_full[:, :history_tokens] = stable_hist
    topk = min(max(1, int(control.get("topk", 8) or 8)), history_tokens)
    threshold_raw = control.get("query_head_frac_threshold", 0.75)
    threshold = min(max(float(0.75 if threshold_raw is None else threshold_raw), 0.0), 1.0)
    query_block_size = max(1, int(control.get("query_block_size", 64) or 64))
    min_direct_witness_seeds = max(0, int(control.get("min_direct_witness_seeds", 0) or 0))
    with torch.no_grad():
        k_hist = k[:, :, :history_tokens, :].detach()
        scale = 1.0 / math.sqrt(float(max(1, int(q.shape[-1]))))
        query_head_frac = torch.zeros(
            int(q.shape[0]),
            int(q.shape[2]),
            device=q.device,
            dtype=torch.float32,
        )
        direct_witness_hit_count = 0
        witness_seed_parts = []
        expanded = stable_hist[:, None, None, :].expand(
            int(q.shape[0]),
            int(q.shape[1]),
            query_block_size,
            history_tokens,
        )
        expanded_seed = None
        if seed_hist is not None:
            expanded_seed = seed_hist[:, None, None, :].expand(
                int(q.shape[0]),
                int(q.shape[1]),
                query_block_size,
                history_tokens,
            )
        expanded_instance = None
        if source_instance_hist is not None:
            expanded_instance = source_instance_hist[:, None, None, :].expand(
                int(q.shape[0]),
                int(q.shape[1]),
                query_block_size,
                history_tokens,
            )
        for q0 in range(0, int(q.shape[2]), query_block_size):
            q1 = min(q0 + query_block_size, int(q.shape[2]))
            qb = q[:, :, q0:q1, :].detach()
            scores_hist = torch.matmul(qb, k_hist.transpose(-2, -1)) * scale
            _, topk_indices = torch.topk(scores_hist, k=topk, dim=-1)
            expanded_block = expanded[:, :, : q1 - q0, :]
            hits = torch.gather(expanded_block, -1, topk_indices.long())
            if direct_match_mode == "same_seed":
                if expanded_seed is not None and query_seed_ids is not None:
                    seed_block = expanded_seed[:, :, : q1 - q0, :]
                    seed_topk = torch.gather(seed_block, -1, topk_indices.long())
                    query_seed = query_seed_ids[:, None, q0:q1, None].expand_as(seed_topk)
                    hits = hits & (seed_topk == query_seed) & (seed_topk >= 0)
                else:
                    hits = torch.zeros_like(hits)
            elif direct_match_mode == "same_masklet":
                if expanded_instance is not None and query_instance_ids is not None:
                    inst_block = expanded_instance[:, :, : q1 - q0, :]
                    inst_topk = torch.gather(inst_block, -1, topk_indices.long())
                    query_inst = query_instance_ids[:, None, q0:q1, None].expand_as(inst_topk)
                    hits = hits & (inst_topk == query_inst) & (inst_topk >= 0)
                else:
                    hits = torch.zeros_like(hits)
            query_head_frac[:, q0:q1] = hits.any(dim=-1).float().mean(dim=1)
            direct_witness_hit_count += int(hits.sum().item())
            if expanded_seed is not None:
                seed_block = expanded_seed[:, :, : q1 - q0, :]
                seed_hits = torch.gather(seed_block, -1, topk_indices.long())
                seed_hits = torch.where(hits, seed_hits, torch.full_like(seed_hits, -1))
                witness_seed_parts.append(seed_hits.detach().reshape(-1).cpu())
        query_mask = query_head_frac >= threshold
        source_count = int(stable_hist.sum().item())
        selected_query_count = int(query_mask.sum().item())
        direct_witness_seed_count = 0
        if witness_seed_parts:
            witness_seeds = torch.cat(witness_seed_parts, dim=0)
            witness_seeds = witness_seeds[witness_seeds >= 0]
            if int(witness_seeds.numel()) > 0:
                direct_witness_seed_count = int(torch.unique(witness_seeds).numel())
        carrier_rule_pass = bool(
            selected_query_count > 0
            and (min_direct_witness_seeds <= 0 or direct_witness_seed_count >= min_direct_witness_seeds)
        )

    if source_count <= 0 or selected_query_count <= 0:
        if isinstance(stats, list):
            stats.append({
                "attention_mass_metric": metric_type,
                stat_key("applied"): False,
                stat_key("trace_only"): bool(trace_only),
                stat_key("reason"): "empty_sources_or_queries",
                stat_key("source_tokens"): source_count,
                stat_key("selected_query_count"): selected_query_count,
                stat_key("query_selected_frac"): float(query_mask.float().mean().item())
                if query_mask.numel() else 0.0,
                stat_key("topk"): int(topk),
                stat_key("head_frac_threshold"): float(threshold),
                stat_key("query_block_size"): int(query_block_size),
                stat_key("direct_witness_hit_count"): int(direct_witness_hit_count),
                stat_key("direct_witness_seed_count"): int(direct_witness_seed_count),
                stat_key("min_direct_witness_seeds"): int(min_direct_witness_seeds),
                stat_key("carrier_rule_pass"): bool(carrier_rule_pass),
                stat_key("direct_match_mode"): str(direct_match_mode),
                stat_key("direct_match_available"): bool(direct_match_available),
                stat_key("direct_match_missing_reason"): str(direct_match_missing_reason),
            })
        return plain()

    rho_raw = control.get("rho", 0.5)
    min_keep_raw = control.get("min_keep", 0.5)
    rho = min(max(float(0.5 if rho_raw is None else rho_raw), 0.0), 1.0)
    min_keep = min(max(float(0.5 if min_keep_raw is None else min_keep_raw), 1e-4), 1.0)
    keep = max(min_keep, 1.0 - rho)
    source_bias_values = torch.zeros((int(q.shape[0]), int(k.shape[2])), device=q.device, dtype=q.dtype)
    source_bias_values[:, :history_tokens] = torch.where(
        stable_hist,
        torch.full_like(stable_hist, math.log(max(keep, 1e-4)), dtype=q.dtype),
        torch.zeros_like(stable_hist, dtype=q.dtype),
    )
    _append_source_soft_mass_stats(
        q,
        k,
        affected_mask=affected_full,
        stable_anchor_mask=affected_full,
        query_mask=query_mask,
        base_attn_mask=None,
        source_bias_values=source_bias_values,
        source_weights=None,
        source_head_mask=None,
        attention_mass_stats=stats,
        attention_mass_max_queries=int(control.get("attention_mass_max_queries", 512) or 512),
        metric_type=metric_type,
    )
    if trace_only:
        if isinstance(stats, list) and stats:
            stats[-1].update({
                stat_key("applied"): False,
                stat_key("trace_only"): True,
                stat_key("reason"): "trace_only_ok",
                stat_key("source_tokens"): source_count,
                stat_key("selected_query_count"): selected_query_count,
                stat_key("query_selected_frac"): float(query_mask.float().mean().item())
                if query_mask.numel() else 0.0,
                stat_key("topk"): int(topk),
                stat_key("head_frac_threshold"): float(threshold),
                stat_key("query_block_size"): int(query_block_size),
                stat_key("rho"): float(rho),
                stat_key("min_keep"): float(min_keep),
                stat_key("keep"): float(keep),
                stat_key("direct_witness_hit_count"): int(direct_witness_hit_count),
                stat_key("direct_witness_seed_count"): int(direct_witness_seed_count),
                stat_key("min_direct_witness_seeds"): int(min_direct_witness_seeds),
                stat_key("carrier_rule_pass"): bool(carrier_rule_pass),
                stat_key("direct_match_mode"): str(direct_match_mode),
                stat_key("direct_match_available"): bool(direct_match_available),
                stat_key("direct_match_missing_reason"): str(direct_match_missing_reason),
            })
        return plain()

    out = plain()
    out = out.clone()
    local_bias_base = torch.zeros(
        1,
        1,
        1,
        int(k.shape[2]),
        device=q.device,
        dtype=q.dtype,
    )
    local_bias_base[..., :history_tokens] = torch.where(
        stable_hist[:1, None, None, :],
        torch.full_like(stable_hist[:1, None, None, :], math.log(max(keep, 1e-4)), dtype=q.dtype),
        torch.zeros_like(stable_hist[:1, None, None, :], dtype=q.dtype),
    )
    for b_idx in range(int(q.shape[0])):
        query_idx = torch.nonzero(query_mask[b_idx], as_tuple=False).reshape(-1)
        if int(query_idx.numel()) <= 0:
            continue
        bias_b = local_bias_base
        if int(q.shape[0]) != 1:
            bias_b = torch.zeros(
                1,
                1,
                1,
                int(k.shape[2]),
                device=q.device,
                dtype=q.dtype,
            )
            bias_b[..., :history_tokens] = torch.where(
                stable_hist[b_idx : b_idx + 1, None, None, :],
                torch.full_like(stable_hist[b_idx : b_idx + 1, None, None, :], math.log(max(keep, 1e-4)), dtype=q.dtype),
                torch.zeros_like(stable_hist[b_idx : b_idx + 1, None, None, :], dtype=q.dtype),
            )
        for q_start in range(0, int(query_idx.numel()), query_block_size):
            q_sel = query_idx[q_start : q_start + query_block_size]
            local_bias = bias_b.expand(1, 1, int(q_sel.numel()), int(k.shape[2]))
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                out[b_idx : b_idx + 1, :, q_sel, :] = scaled_dot_product_attention(
                    q[b_idx : b_idx + 1, :, q_sel, :],
                    k[b_idx : b_idx + 1],
                    v[b_idx : b_idx + 1],
                    attn_mask=local_bias,
                )
    if isinstance(stats, list) and stats:
        stats[-1].update({
            stat_key("applied"): True,
            stat_key("trace_only"): False,
            stat_key("reason"): "ok",
            stat_key("source_tokens"): source_count,
            stat_key("selected_query_count"): selected_query_count,
            stat_key("query_selected_frac"): float(query_mask.float().mean().item())
            if query_mask.numel() else 0.0,
            stat_key("topk"): int(topk),
            stat_key("head_frac_threshold"): float(threshold),
            stat_key("query_block_size"): int(query_block_size),
            stat_key("rho"): float(rho),
            stat_key("min_keep"): float(min_keep),
            stat_key("keep"): float(keep),
            stat_key("direct_witness_hit_count"): int(direct_witness_hit_count),
            stat_key("direct_witness_seed_count"): int(direct_witness_seed_count),
            stat_key("min_direct_witness_seeds"): int(min_direct_witness_seeds),
            stat_key("carrier_rule_pass"): bool(carrier_rule_pass),
        })
    return out


def _record_source_soft_trace_only(q: Tensor, k: Tensor, control: dict) -> None:
    """Record raw Q/K source-attention diagnostics without changing attention output."""

    affected_mask = control["affected_mask"].to(device=q.device, dtype=torch.bool)
    query_mask = control.get("query_mask")
    if query_mask is not None:
        query_mask = query_mask.to(device=q.device, dtype=torch.bool)
    base_attn_mask = control.get("base_attn_mask")
    source_bias_values = control.get("source_bias_values")
    source_weights = control.get("source_weights")
    stable_anchor_mask = control.get("stable_anchor_mask")
    source_head_mask = _source_attention_head_mask(control, int(q.shape[1]), q.device, torch.float32)
    if base_attn_mask is not None:
        base_attn_mask = base_attn_mask.to(device=q.device, dtype=q.dtype)
    if source_bias_values is not None:
        source_bias_values = source_bias_values.to(device=q.device, dtype=q.dtype)
    if source_weights is not None:
        source_weights = source_weights.to(device=q.device, dtype=q.dtype)
    if stable_anchor_mask is not None:
        stable_anchor_mask = stable_anchor_mask.to(device=q.device, dtype=torch.bool)
    _append_source_soft_mass_stats(
        q,
        k,
        affected_mask=affected_mask,
        stable_anchor_mask=stable_anchor_mask,
        query_mask=query_mask,
        base_attn_mask=base_attn_mask,
        source_bias_values=source_bias_values,
        source_weights=source_weights,
        source_head_mask=source_head_mask,
        attention_mass_stats=control.get("attention_mass_stats"),
        attention_mass_max_queries=int(control.get("attention_mass_max_queries", 512) or 512),
        metric_type=str(control.get("attention_mass_metric", "source_soft_trace_only")),
    )
    dump_info = _dump_source_soft_attention_sample(
        q,
        k,
        affected_mask=affected_mask,
        query_mask=query_mask,
        base_attn_mask=base_attn_mask,
        source_bias_values=source_bias_values,
        source_weights=source_weights,
        control=control,
    )
    stats = control.get("attention_mass_stats")
    if dump_info and isinstance(stats, list):
        if stats:
            stats[-1].update(dump_info)
        else:
            stats.append(dump_info)


def _append_overlap_bias_mass_stats(
    q: Tensor,
    k: Tensor,
    *,
    source_start: int,
    source_end: int,
    bias_values: Tensor,
    head_bias_scale: Tensor | None = None,
    attention_mass_stats: list | None,
    attention_mass_max_queries: int,
    metric_type: str,
) -> None:
    """Record sampled attention mass on overlap-bias source tokens."""

    if attention_mass_stats is None:
        return
    if not torch.is_tensor(bias_values) or bias_values.ndim != 3:
        return
    qn = min(int(bias_values.shape[1]), int(q.shape[2]))
    sn = min(int(bias_values.shape[2]), max(0, int(source_end) - int(source_start)))
    source_start = max(0, int(source_start))
    source_end = min(int(source_end), int(k.shape[2]))
    sn = min(sn, max(0, source_end - source_start))
    if qn <= 0 or sn <= 0:
        return

    with torch.no_grad():
        bias = bias_values[:, :qn, :sn].to(device=q.device, dtype=torch.float32)
        head_dim = max(1, int(q.shape[-1]))
        scale = 1.0 / math.sqrt(float(head_dim))
        source_before_vals = []
        source_after_vals = []
        selected_before_vals = []
        selected_after_vals = []
        selected_head_max_before_vals = []
        selected_head_max_after_vals = []
        source_head_before_vals = []
        source_head_after_vals = []
        selected_head_before_vals = []
        selected_head_after_vals = []
        selected_tokens = []
        query_samples = []
        for b in range(int(q.shape[0])):
            q_idx = _sample_query_indices(qn, int(attention_mass_max_queries), q.device)
            if int(q_idx.numel()) <= 0:
                continue
            qb = q[b : b + 1, :, q_idx, :].float()
            kb = k[b : b + 1].float()
            logits = torch.matmul(qb, kb.transpose(-2, -1)) * scale
            local_bias = torch.zeros_like(logits)
            source_bias = bias[b : b + 1, q_idx, :].unsqueeze(1)
            if torch.is_tensor(head_bias_scale):
                source_bias = source_bias * head_bias_scale.to(device=q.device, dtype=torch.float32).view(1, -1, 1, 1)
            local_bias[:, :, :, source_start:source_end] = source_bias
            attn_before = torch.softmax(logits, dim=-1)
            attn_after = torch.softmax(logits + local_bias, dim=-1)

            source_before = attn_before[..., source_start:source_end].sum(dim=-1)
            source_after = attn_after[..., source_start:source_end].sum(dim=-1)
            selected_source = bias[b].abs().amax(dim=0) > 1e-12
            if int(selected_source.sum().item()) > 0:
                full_selected = torch.zeros(int(k.shape[2]), device=q.device, dtype=torch.bool)
                full_selected[source_start:source_end] = selected_source[:sn]
                selected_before = attn_before[..., full_selected].sum(dim=-1)
                selected_after = attn_after[..., full_selected].sum(dim=-1)
                selected_head_before = selected_before.mean(dim=2).reshape(-1)
                selected_head_after = selected_after.mean(dim=2).reshape(-1)
            else:
                selected_before = torch.zeros_like(source_before)
                selected_after = torch.zeros_like(source_after)
                selected_head_before = torch.zeros(int(q.shape[1]), device=q.device, dtype=torch.float32)
                selected_head_after = torch.zeros(int(q.shape[1]), device=q.device, dtype=torch.float32)

            source_before_vals.append(float(source_before.mean().item()))
            source_after_vals.append(float(source_after.mean().item()))
            selected_before_vals.append(float(selected_before.mean().item()))
            selected_after_vals.append(float(selected_after.mean().item()))
            source_head_before_vals.append(source_before.mean(dim=2).reshape(-1).detach().cpu())
            source_head_after_vals.append(source_after.mean(dim=2).reshape(-1).detach().cpu())
            selected_head_before_vals.append(selected_head_before.detach().cpu())
            selected_head_after_vals.append(selected_head_after.detach().cpu())
            selected_head_max_before_vals.append(float(selected_head_before.max().item()))
            selected_head_max_after_vals.append(float(selected_head_after.max().item()))
            selected_tokens.append(int(selected_source.sum().item()))
            query_samples.append(int(q_idx.numel()))

        if selected_before_vals:
            selected_before_mean = float(torch.tensor(selected_before_vals).mean().item())
            selected_after_mean = float(torch.tensor(selected_after_vals).mean().item())
            source_before_mean = float(torch.tensor(source_before_vals).mean().item())
            source_after_mean = float(torch.tensor(source_after_vals).mean().item())
            head_before_mean = float(torch.tensor(selected_head_max_before_vals).mean().item())
            head_after_mean = float(torch.tensor(selected_head_max_after_vals).mean().item())
            source_head_before = torch.stack(source_head_before_vals, dim=0).float().mean(dim=0)
            source_head_after = torch.stack(source_head_after_vals, dim=0).float().mean(dim=0)
            selected_head_before = torch.stack(selected_head_before_vals, dim=0).float().mean(dim=0)
            selected_head_after = torch.stack(selected_head_after_vals, dim=0).float().mean(dim=0)
            source_head_lift = source_head_after - source_head_before
            selected_head_lift = selected_head_after - selected_head_before
            selected_top_head = int(torch.argmax(selected_head_lift).item()) if selected_head_lift.numel() else -1
            source_top_head = int(torch.argmax(source_head_lift).item()) if source_head_lift.numel() else -1
            attention_mass_stats.append({
                "attention_mass_metric": str(metric_type),
                "attention_mass_removed_before": selected_before_mean,
                "attention_mass_removed_after": selected_after_mean,
                "attention_mass_actual_after": selected_after_mean,
                "attention_mass_retained_before": source_before_mean,
                "attention_mass_retained_after": source_after_mean,
                "attention_mass_removed_tokens_mean": float(torch.tensor(selected_tokens, dtype=torch.float32).mean().item()),
                "attention_mass_retained_tokens_mean": float(sn),
                "attention_mass_query_sample_tokens_mean": float(torch.tensor(query_samples, dtype=torch.float32).mean().item()),
                "attention_mass_sampled": bool(max(selected_tokens or [0]) > 0),
                "swa_overlap_attention_mass_source_before": source_before_mean,
                "swa_overlap_attention_mass_source_after": source_after_mean,
                "swa_overlap_attention_mass_selected_before": selected_before_mean,
                "swa_overlap_attention_mass_selected_after": selected_after_mean,
                "swa_overlap_attention_mass_selected_lift": selected_after_mean - selected_before_mean,
                "swa_overlap_attention_mass_source_lift": source_after_mean - source_before_mean,
                "swa_overlap_attention_mass_selected_head_max_before": head_before_mean,
                "swa_overlap_attention_mass_selected_head_max_after": head_after_mean,
                "swa_overlap_attention_mass_selected_head_max_lift": head_after_mean - head_before_mean,
                "swa_overlap_attention_mass_source_before_by_head": [
                    float(v) for v in source_head_before.tolist()
                ],
                "swa_overlap_attention_mass_source_after_by_head": [
                    float(v) for v in source_head_after.tolist()
                ],
                "swa_overlap_attention_mass_source_lift_by_head": [
                    float(v) for v in source_head_lift.tolist()
                ],
                "swa_overlap_attention_mass_selected_before_by_head": [
                    float(v) for v in selected_head_before.tolist()
                ],
                "swa_overlap_attention_mass_selected_after_by_head": [
                    float(v) for v in selected_head_after.tolist()
                ],
                "swa_overlap_attention_mass_selected_lift_by_head": [
                    float(v) for v in selected_head_lift.tolist()
                ],
                "swa_overlap_attention_mass_selected_top_head_by_lift": selected_top_head,
                "swa_overlap_attention_mass_selected_top_head_lift": (
                    float(selected_head_lift[selected_top_head].item()) if selected_top_head >= 0 else None
                ),
                "swa_overlap_attention_mass_source_top_head_by_lift": source_top_head,
                "swa_overlap_attention_mass_source_top_head_lift": (
                    float(source_head_lift[source_top_head].item()) if source_top_head >= 0 else None
                ),
            })


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
        source_trace_only = (
            source_soft is not None and str(source_soft.get("mode", "")).lower() == "trace_only"
        )
        if source_trace_only:
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

            _record_source_soft_trace_only(q, k, source_soft)
            source_soft = None
            attn_mask = None
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
    def compute_qkv(self, x: Tensor, xpos=None) -> tuple[Tensor, Tensor, Tensor]:
        """Compute Q, K, V after norm/RoPE for cache diagnostics."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        return q, k, v

    def compute_kv(self, x: Tensor, xpos=None) -> tuple[Tensor, Tensor]:
        """Compute K, V for caching. Returns (K, V) after norm and RoPE."""
        _, k, v = self.compute_qkv(x, xpos=xpos)
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
        compact_kv = attn_mask if isinstance(attn_mask, dict) and attn_mask.get("type") == "compact_kv" else None
        source_soft = attn_mask if isinstance(attn_mask, dict) and attn_mask.get("type") == "source_soft" else None
        overlap_bias = attn_mask if isinstance(attn_mask, dict) and attn_mask.get("type") == "overlap_bias" else None
        prev_ttt_anchor_query_soft = (
            attn_mask
            if isinstance(attn_mask, dict) and attn_mask.get("type") == "prev_ttt_anchor_query_soft"
            else None
        )
        dense_attn_mask = None if (
            compact_kv is not None
            or source_soft is not None
            or overlap_bias is not None
            or prev_ttt_anchor_query_soft is not None
        ) else attn_mask
        is_float_mask = (
            dense_attn_mask is not None
            and torch.is_tensor(dense_attn_mask)
            and torch.is_floating_point(dense_attn_mask)
        )
        
        if compact_kv is not None:
            x = _compact_kv_sdpa(
                q,
                k_full,
                v_full,
                compact_kv["source_keep_mask"],
                compact_kv.get("attention_mass_stats"),
                int(compact_kv.get("attention_mass_max_queries", 512) or 512),
            )
        elif source_soft is not None:
            x = _source_soft_sdpa(q, k_full, v_full, source_soft)
        elif prev_ttt_anchor_query_soft is not None:
            x = _prev_ttt_anchor_query_soft_sdpa(q, k_full, v_full, prev_ttt_anchor_query_soft)
        elif dense_attn_mask is not None and FLEX_ATTENTION_AVAILABLE and not is_float_mask:
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
                head_bias_scale = None
                head_indices = overlap_bias.get("head_indices")
                if isinstance(head_indices, (list, tuple)) and len(head_indices) > 0:
                    head_bias_scale = torch.zeros(int(q.shape[1]), device=q.device, dtype=q.dtype)
                    for raw_head in head_indices:
                        try:
                            head_idx = int(raw_head)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= head_idx < int(q.shape[1]):
                            head_bias_scale[head_idx] = 1.0
                _append_overlap_bias_mass_stats(
                    q,
                    k_full,
                    source_start=source_start,
                    source_end=source_end,
                    bias_values=bias_values,
                    head_bias_scale=head_bias_scale,
                    attention_mass_stats=overlap_bias.get("attention_mass_stats"),
                    attention_mass_max_queries=int(overlap_bias.get("attention_mass_max_queries", 64) or 64),
                    metric_type=str(overlap_bias.get("attention_mass_metric", "swa_overlap_bias")),
                )
                overlap_out = []
                for q0 in range(0, qn, block_size):
                    q1 = min(q0 + block_size, qn)
                    q_chunk = q[:, :, q0:q1, :]
                    local_bias_heads = int(q.shape[1]) if torch.is_tensor(head_bias_scale) else 1
                    local_bias = torch.zeros(
                        q.shape[0],
                        local_bias_heads,
                        q1 - q0,
                        k_full.shape[2],
                        device=q.device,
                        dtype=q.dtype,
                    )
                    source_bias = bias_values[:, q0:q1, :].unsqueeze(1)
                    if torch.is_tensor(head_bias_scale):
                        source_bias = source_bias * head_bias_scale.view(1, -1, 1, 1)
                    local_bias[:, :, :, source_start:source_end] = source_bias
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
        if source_soft is not None and str(source_soft.get("mode", "")).lower() == "trace_only":
            _record_source_soft_trace_only(q, k, source_soft)
            source_soft = None
            attn_mask = None
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
