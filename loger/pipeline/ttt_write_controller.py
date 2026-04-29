"""
Stage E: TTT Write Controller

Takes the token-level write prior ``A_tok`` (from Stage D) and the
``WriteCacheOutput`` (from Stage A) to perform a **delayed write-back**
of TTT fast weights:

    W_m  →  W_{m+1}

The controller replays the TTT update loop with modified per-token
learning rates (lr_new = A_tok · lr_original), so that low-prior tokens
contribute less to the fast-weight update.  It also applies an optional
block-level write gain that scales the entire update direction based on
the chunk-level suppression ratio.

Phase 1: deterministic replay with token-level prior weighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from loger.models.ttt import fast_weight_replay_update
from .geometry_backbone import WriteCacheOutput, TTTLayerCache
from .geometry_backbone import TOKEN_TYPE_PATCH


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class WriteResult:
    """Output of the TTT Write Controller — the committed W_{m+1}."""

    w0: List[Optional[torch.Tensor]]    # per-layer branch-0 weights
    w1: List[Optional[torch.Tensor]]    # per-layer branch-1 weights
    w2: List[Optional[torch.Tensor]]    # per-layer branch-2 weights
    history: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None

    debug: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core controller
# ---------------------------------------------------------------------------
class TTTWriteController:
    """Stage E of the Semantic Prior Pipeline.

    Usage::

        ctrl = TTTWriteController()
        result = ctrl.run(write_cache, A_tok, device="cuda")
        # result.w0/w1/w2 is W_{m+1} — pass to next chunk's Stage A
    """

    def __init__(
        self,
        *,
        lambda_min: float = 0.0,
        lambda_max: float = 1.0,
        device: str = "cuda",
        write_mode: str = "semantic",
        eta_mean_preserve: bool = False,
        eta_norm_eps: float = 1e-8,
        prior_branch_mask: str = "0,1,2",
        prior_layer_mode: str = "all",
        prior_single_layer: int = -1,
    ):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.device = device
        self.write_mode = write_mode
        self.eta_mean_preserve = bool(eta_mean_preserve)
        self.eta_norm_eps = float(eta_norm_eps)
        self.prior_branch_mask = self._parse_branch_mask(prior_branch_mask)
        self.prior_layer_mode = str(prior_layer_mode)
        self.prior_single_layer = int(prior_single_layer)

    # -- public API --------------------------------------------------------

    def run(
        self,
        write_cache: WriteCacheOutput,
        A_tok: Optional[torch.Tensor],
        B_chunk_geo: Optional[float] = None,
        device: Optional[str] = None,
        token_type: Optional[torch.Tensor] = None,
    ) -> WriteResult:
        """Perform delayed write-back: W_m → W_{m+1}.

        Parameters
        ----------
        write_cache :
            ``WriteCacheOutput`` from Stage A containing per-layer
            cached primitives and the old weights W_m.
        A_tok :
            Token-level write prior from Stage D, shape ``[L_tok]``.
            Values in [0, 1]; higher means more write-allowed.
        device :
            Device to run the replay on.  Defaults to ``self.device``.

        Returns
        -------
        WriteResult
            ``w0``, ``w1``, ``w2`` lists ready to be fed as
            ``ttt_state_input`` to the next chunk's Geometry Backbone.
        """
        mode = self.write_mode
        dev = device or self.device
        n_layers = write_cache.num_ttt_layers

        history = write_cache.history_provisional

        if mode == "native":
            debug_info = {
                "mode": mode,
                "native_write_through": True,
            }
            return WriteResult(
                w0=list(write_cache.w0_provisional),
                w1=list(write_cache.w1_provisional),
                w2=list(write_cache.w2_provisional),
                history=history,
                debug=debug_info,
            )

        if mode not in {"semantic", "unity_replay"}:
            raise ValueError(f"Unsupported write_mode: {mode}")

        w0_new: List[Optional[torch.Tensor]] = [None] * n_layers
        w1_new: List[Optional[torch.Tensor]] = [None] * n_layers
        w2_new: List[Optional[torch.Tensor]] = [None] * n_layers

        debug_info: Dict[str, Any] = {"mode": mode}

        for li, lc in enumerate(write_cache.layer_caches):
            layer_prior_enabled = self._layer_prior_enabled(li, n_layers)
            if mode == "unity_replay" or not layer_prior_enabled:
                effective_prior = None
                effective_budget = 1.0
                active_branch_mask: Tuple[int, ...] = ()
            else:
                effective_prior = A_tok
                effective_budget = B_chunk_geo
                active_branch_mask = self.prior_branch_mask
            w0_li, w1_li, w2_li, layer_debug = self._replay_layer(
                lc, effective_prior, effective_budget, dev,
                token_type=token_type,
                active_branch_mask=active_branch_mask,
                layer_prior_enabled=bool(layer_prior_enabled),
            )
            w0_new[li] = w0_li
            w1_new[li] = w1_li
            w2_new[li] = w2_li
            debug_info[f"layer_{li}"] = layer_debug

        return WriteResult(
            w0=w0_new,
            w1=w1_new,
            w2=w2_new,
            history=history,
            debug=debug_info,
        )

    # -- per-layer replay --------------------------------------------------

    def _replay_layer(
        self,
        lc: TTTLayerCache,
        A_tok: Optional[torch.Tensor],
        B_chunk_geo: Optional[float],
        device: str,
        *,
        token_type: Optional[torch.Tensor] = None,
        active_branch_mask: Tuple[int, ...] = (0, 1, 2),
        layer_prior_enabled: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Replay one TTT layer's update with prior-weighted lr."""
        # Move cached tensors to device.  The original forward mixes
        # bfloat16 (q/k/v inside autocast) and float32 (weights, lr).
        # We cast everything to bfloat16 for the matmul-heavy replay
        # (matching the original autocast context), except lr which
        # stays float32 as in the original code.
        compute_dtype = torch.bfloat16
        k = lc.k.to(device=device, dtype=compute_dtype)
        v = lc.v.to(device=device, dtype=compute_dtype)
        lr0 = lc.lr0.to(device=device, dtype=torch.float32)
        lr1 = lc.lr1.to(device=device, dtype=torch.float32)
        lr2 = lc.lr2.to(device=device, dtype=torch.float32)
        w0_old = lc.w0_old.to(device=device, dtype=compute_dtype)
        w1_old = lc.w1_old.to(device=device, dtype=compute_dtype)
        w2_old = lc.w2_old.to(device=device, dtype=compute_dtype)
        momentum = lc.momentum.to(device=device, dtype=torch.float32) if lc.momentum is not None else None

        # Build token_prior: shape [1, l, 1] to broadcast over batch*heads
        # A_tok is [L_tok] but cached k is [b*h, l, d] where l is only the
        # tokens that went through this TTT layer (patch tokens from the
        # window processed by decode).  We need to align.
        l = k.shape[1]
        if A_tok is None:
            prior_flat = torch.ones(l)
        elif A_tok.shape[0] >= l:
            prior_flat = A_tok[:l]
        else:
            prior_flat = torch.ones(l)
            prior_flat[:A_tok.shape[0]] = A_tok
        token_prior = prior_flat.to(device).unsqueeze(0).unsqueeze(-1)  # [1, l, 1]
        unity_prior = torch.ones_like(token_prior)
        branch_enabled = tuple(bool(A_tok is not None and i in active_branch_mask) for i in range(3))
        token_prior0 = token_prior if branch_enabled[0] else unity_prior
        token_prior1 = token_prior if branch_enabled[1] else unity_prior
        token_prior2 = token_prior if branch_enabled[2] else unity_prior

        mean_prior = prior_flat.mean().item()
        budget_geo = float(B_chunk_geo) if B_chunk_geo is not None else mean_prior
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * budget_geo

        debug = self._build_prior_debug(
            prior_flat=prior_flat,
            A_tok=A_tok,
            token_type=token_type,
            cache_l=int(l),
            lr0=lr0,
            lr1=lr1,
            lr2=lr2,
            branch_prior_flat=(
                token_prior0.squeeze(0).squeeze(-1).detach().cpu(),
                token_prior1.squeeze(0).squeeze(-1).detach().cpu(),
                token_prior2.squeeze(0).squeeze(-1).detach().cpu(),
            ),
        )
        debug.update({
            "mean_prior": mean_prior,
            "budget_geo": budget_geo,
            "lambda_write": lam,
            "eta_mean_preserve": self.eta_mean_preserve,
            "prior_layer_mode": self.prior_layer_mode,
            "layer_prior_enabled": bool(layer_prior_enabled),
            "prior_branch_mask": list(active_branch_mask),
            "branch0_prior_enabled": branch_enabled[0],
            "branch1_prior_enabled": branch_enabled[1],
            "branch2_prior_enabled": branch_enabled[2],
        })

        lam0 = lam if branch_enabled[0] else 1.0
        lam1 = lam if branch_enabled[1] else 1.0
        lam2 = lam if branch_enabled[2] else 1.0

        if self.eta_mean_preserve and A_tok is not None:
            if branch_enabled[0]:
                lr0, scale0, post0 = self._eta_normalize_lr(lr0, token_prior0)
            else:
                scale0, post0 = 1.0, 1.0
            if branch_enabled[1]:
                lr1, scale1, post1 = self._eta_normalize_lr(lr1, token_prior1)
            else:
                scale1, post1 = 1.0, 1.0
            if branch_enabled[2]:
                lr2, scale2, post2 = self._eta_normalize_lr(lr2, token_prior2)
            else:
                scale2, post2 = 1.0, 1.0
            debug.update({
                "eta_norm_scale_lr0": scale0,
                "eta_norm_scale_lr1": scale1,
                "eta_norm_scale_lr2": scale2,
                "m_eta_after_lr0": post0,
                "m_eta_after_lr1": post1,
                "m_eta_after_lr2": post2,
            })

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            w0_new, w1_new, w2_new = fast_weight_replay_update(
                w0_old, w1_old, w2_old,
                k, v,
                lr0 * lam0, lr1 * lam1, lr2 * lam2,
                token_prior,
                lc.ttt_op_order,
                muon_update_steps=lc.muon_update_steps,
                momentum=momentum,
                ttt_update_steps=lc.ttt_update_steps,
                token_prior0=token_prior0,
                token_prior1=token_prior1,
                token_prior2=token_prior2,
            )

        return w0_new.cpu(), w1_new.cpu(), w2_new.cpu(), debug

    def _eta_normalize_lr(
        self,
        lr: torch.Tensor,
        token_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        """Scale lr so lr-weighted prior mass stays near native replay."""
        denom = lr.sum().clamp_min(self.eta_norm_eps)
        m_eta = (lr * token_prior).sum() / denom
        scale = torch.reciprocal(m_eta.clamp_min(self.eta_norm_eps))
        lr_new = lr * scale
        post = ((lr_new * token_prior).sum() / denom).detach().float().item()
        return lr_new, float(scale.detach().float().item()), float(post)

    @staticmethod
    def _parse_branch_mask(mask: str) -> Tuple[int, ...]:
        if mask is None:
            return (0, 1, 2)
        text = str(mask).strip().lower()
        if text in {"", "all", "0,1,2"}:
            return (0, 1, 2)
        if text in {"none", "off"}:
            return ()
        branches = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            idx = int(part)
            if idx not in (0, 1, 2):
                raise ValueError(f"prior branch must be 0, 1, or 2, got {idx}")
            if idx not in branches:
                branches.append(idx)
        return tuple(branches)

    def _layer_prior_enabled(self, layer_idx: int, n_layers: int) -> bool:
        mode = self.prior_layer_mode
        if mode == "all":
            return True
        if mode == "early":
            return layer_idx < max(1, n_layers // 2)
        if mode == "late":
            return layer_idx >= n_layers // 2
        if mode == "middle":
            lo = n_layers // 3
            hi = max(lo + 1, (2 * n_layers) // 3)
            return lo <= layer_idx < hi
        if mode == "single":
            return layer_idx == self.prior_single_layer
        raise ValueError(f"Unsupported prior_layer_mode: {mode}")

    def _build_prior_debug(
        self,
        *,
        prior_flat: torch.Tensor,
        A_tok: Optional[torch.Tensor],
        token_type: Optional[torch.Tensor],
        cache_l: int,
        lr0: torch.Tensor,
        lr1: torch.Tensor,
        lr2: torch.Tensor,
        branch_prior_flat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        prior_cpu = prior_flat.detach().cpu().float()
        debug: Dict[str, Any] = {
            "cache_l": int(cache_l),
            "mean_prior_flat": float(prior_cpu.mean().item()) if prior_cpu.numel() else 0.0,
            "min_prior_flat": float(prior_cpu.min().item()) if prior_cpu.numel() else 0.0,
            "max_prior_flat": float(prior_cpu.max().item()) if prior_cpu.numel() else 0.0,
            "std_prior_flat": float(prior_cpu.std(unbiased=False).item()) if prior_cpu.numel() else 0.0,
            "first_20_prior_values": [
                float(x) for x in prior_cpu[:20].tolist()
            ],
        }

        branch_prior_cpu = branch_prior_flat or (prior_cpu, prior_cpu, prior_cpu)

        def _eta_ratio(lr: torch.Tensor, branch_prior: torch.Tensor) -> float:
            lr_cpu = lr.detach().cpu().float()
            if lr_cpu.numel() == 0:
                return 1.0
            prior = branch_prior.detach().cpu().float().to(dtype=lr_cpu.dtype).view(1, -1, 1)
            denom = lr_cpu.sum().clamp_min(1e-8)
            return float(((lr_cpu * prior).sum() / denom).item())

        debug["m_eta_lr0"] = _eta_ratio(lr0, branch_prior_cpu[0])
        debug["m_eta_lr1"] = _eta_ratio(lr1, branch_prior_cpu[1])
        debug["m_eta_lr2"] = _eta_ratio(lr2, branch_prior_cpu[2])
        debug["mean_prior_branch0"] = float(branch_prior_cpu[0].float().mean().item())
        debug["mean_prior_branch1"] = float(branch_prior_cpu[1].float().mean().item())
        debug["mean_prior_branch2"] = float(branch_prior_cpu[2].float().mean().item())

        if A_tok is not None:
            a_cpu = A_tok.detach().cpu().float()
            debug["L_tok"] = int(a_cpu.shape[0])
        else:
            a_cpu = None
            debug["L_tok"] = int(cache_l)

        if token_type is None:
            return debug

        tt = token_type.detach().cpu().long()
        debug["token_type_L"] = int(tt.shape[0])
        prefix = tt[: min(cache_l, int(tt.shape[0]))]
        debug["first_20_token_type_if_available"] = [
            int(x) for x in prefix[:20].tolist()
        ]
        if prefix.numel() > 0:
            prefix_patch = prefix == TOKEN_TYPE_PATCH
            debug["prefix_patch_tokens"] = int(prefix_patch.sum().item())
            debug["prefix_special_tokens"] = int((~prefix_patch).sum().item())
            if prefix_patch.any():
                debug["mean_prior_prefix_patch"] = float(
                    prior_cpu[: prefix.numel()][prefix_patch].mean().item()
                )
            if (~prefix_patch).any():
                debug["mean_prior_prefix_special"] = float(
                    prior_cpu[: prefix.numel()][~prefix_patch].mean().item()
                )

        if a_cpu is not None and a_cpu.shape[0] == tt.shape[0]:
            patch_mask = tt == TOKEN_TYPE_PATCH
            special_mask = ~patch_mask
            debug["num_patch_tokens_in_A_tok"] = int(patch_mask.sum().item())
            debug["num_special_tokens_in_A_tok"] = int(special_mask.sum().item())
            if patch_mask.any():
                debug["mean_prior_patch_expected"] = float(a_cpu[patch_mask].mean().item())
            if special_mask.any():
                debug["mean_prior_special_expected"] = float(a_cpu[special_mask].mean().item())

        return debug
