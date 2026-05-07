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
    transient_delta: Optional[Dict[str, List[Optional[torch.Tensor]]]] = None

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
        prior_layer_branch_policy: Optional[str] = None,
        update_delta_scale: float = 1.0,
        update_delta_scales: Optional[str] = None,
        update_native_mix_scales: Optional[str] = None,
        prior_transform_mode: str = "none",
        prior_anti_scale: float = 0.0,
        prior_gamma: float = 1.0,
        update_token_scope: str = "all",
        update_token_scope_floor: float = 0.0,
        replay_feature_gate_mode: str = "none",
        replay_feature_gate_rho: float = 0.0,
        replay_feature_gate_min: float = 0.5,
        replay_feature_gate_branch_mask: str = "all",
        replay_token_filter_mode: str = "none",
        replay_token_filter_ratio: float = 1.0,
        replay_token_filter_threshold: float = 1.0,
        replay_token_filter_scope: str = "all",
        replay_token_filter_branch_mask: str = "all",
        replay_token_filter_blend: float = 1.0,
        replay_token_filter_blend_mode: str = "linear",
        transient_delta_subtract_scale: float = 0.0,
        transient_delta_branch_mask: str = "0",
        commit_ema_alpha: float = 1.0,
        commit_ema_branch_mask: str = "all",
        native_delta_gate_mode: str = "none",
        native_delta_gate_min_cos: float = 0.0,
        native_delta_gate_fallback: float = 0.0,
        native_delta_gate_cap_ratio: float = 1.0,
        native_delta_gate_branch_mask: str = "all",
        commit_filter_mode: str = "none",
        commit_filter_risk_source: str = "d_tok",
        commit_filter_scope: str = "tail_overlap",
        commit_filter_stat: str = "mean",
        commit_filter_base: float = 0.0,
        commit_filter_gain: float = 1.0,
        commit_filter_min: float = 0.0,
        commit_filter_max: float = 1.0,
        commit_filter_branch_mask: str = "0",
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
        self.prior_layer_branch_policy_text = str(prior_layer_branch_policy or "").strip()
        self.prior_layer_branch_policy = self._parse_layer_branch_policy(self.prior_layer_branch_policy_text)
        self.update_delta_scale = float(update_delta_scale)
        self.update_delta_scales = self._parse_delta_scales(
            update_delta_scales,
            default=self.update_delta_scale,
        )
        self.update_native_mix_scales = self._parse_delta_scales(
            update_native_mix_scales,
            default=1.0,
        )
        self.prior_transform_mode = str(prior_transform_mode or "none").strip().lower()
        self.prior_anti_scale = float(prior_anti_scale)
        self.prior_gamma = float(prior_gamma)
        self.update_token_scope = str(update_token_scope or "all")
        self.update_token_scope_floor = float(update_token_scope_floor)
        self.replay_feature_gate_mode = str(replay_feature_gate_mode or "none").strip().lower()
        self.replay_feature_gate_rho = float(replay_feature_gate_rho)
        self.replay_feature_gate_min = float(replay_feature_gate_min)
        self.replay_feature_gate_branch_mask = self._parse_branch_mask(replay_feature_gate_branch_mask)
        self.replay_token_filter_mode = str(replay_token_filter_mode or "none").strip().lower()
        self.replay_token_filter_ratio = float(replay_token_filter_ratio)
        self.replay_token_filter_threshold = float(replay_token_filter_threshold)
        self.replay_token_filter_scope = str(replay_token_filter_scope or "all").strip().lower()
        self.replay_token_filter_branch_mask = self._parse_branch_mask(replay_token_filter_branch_mask)
        self.replay_token_filter_blend = float(replay_token_filter_blend)
        self.replay_token_filter_blend_mode = str(replay_token_filter_blend_mode or "linear").strip().lower()
        self.transient_delta_subtract_scale = float(transient_delta_subtract_scale)
        self.transient_delta_branch_mask = self._parse_branch_mask(transient_delta_branch_mask)
        self.commit_ema_alpha = float(commit_ema_alpha)
        self.commit_ema_branch_mask = self._parse_branch_mask(commit_ema_branch_mask)
        self.native_delta_gate_mode = str(native_delta_gate_mode or "none").strip().lower()
        self.native_delta_gate_min_cos = float(native_delta_gate_min_cos)
        self.native_delta_gate_fallback = float(native_delta_gate_fallback)
        self.native_delta_gate_cap_ratio = float(native_delta_gate_cap_ratio)
        self.native_delta_gate_branch_mask = self._parse_branch_mask(native_delta_gate_branch_mask)
        self.commit_filter_mode = str(commit_filter_mode or "none").strip().lower()
        self.commit_filter_risk_source = str(commit_filter_risk_source or "d_tok").strip().lower()
        self.commit_filter_scope = str(commit_filter_scope or "tail_overlap").strip().lower()
        self.commit_filter_stat = str(commit_filter_stat or "mean").strip().lower()
        self.commit_filter_base = float(commit_filter_base)
        self.commit_filter_gain = float(commit_filter_gain)
        self.commit_filter_min = float(commit_filter_min)
        self.commit_filter_max = float(commit_filter_max)
        self.commit_filter_branch_mask = self._parse_branch_mask(commit_filter_branch_mask)

    # -- public API --------------------------------------------------------

    def run(
        self,
        write_cache: WriteCacheOutput,
        A_tok: Optional[torch.Tensor],
        B_chunk_geo: Optional[float] = None,
        device: Optional[str] = None,
        token_type: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        overlap_frames: int = 0,
        risk_tok: Optional[torch.Tensor] = None,
        prev_transient_delta: Optional[Dict[str, List[Optional[torch.Tensor]]]] = None,
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
        transient_delta: Dict[str, List[Optional[torch.Tensor]]] = {
            "w0": [None] * n_layers,
            "w1": [None] * n_layers,
            "w2": [None] * n_layers,
        }

        debug_info: Dict[str, Any] = {"mode": mode}

        for li, lc in enumerate(write_cache.layer_caches):
            layer_prior_enabled = self._layer_prior_enabled(li, n_layers)
            active_branch_mask = self._layer_branch_mask(li, n_layers) if layer_prior_enabled else ()
            if mode == "unity_replay" or not layer_prior_enabled or len(active_branch_mask) == 0:
                effective_prior = None
                effective_budget = 1.0
            else:
                effective_prior = A_tok
                effective_budget = B_chunk_geo
            w0_li, w1_li, w2_li, layer_debug, layer_transient_delta = self._replay_layer(
                lc, effective_prior, effective_budget, dev,
                token_type=token_type,
                active_branch_mask=active_branch_mask,
                layer_prior_enabled=bool(layer_prior_enabled),
                num_frames=num_frames,
                overlap_frames=overlap_frames,
            )
            w0_new[li] = w0_li
            w1_new[li] = w1_li
            w2_new[li] = w2_li
            if layer_transient_delta is not None:
                for branch_name in ("w0", "w1", "w2"):
                    value = layer_transient_delta.get(branch_name)
                    if value is not None:
                        transient_delta[branch_name][li] = value
            debug_info[f"layer_{li}"] = layer_debug

        self._apply_native_delta_gate(write_cache, w0_new, w1_new, w2_new, debug_info)
        self._mix_with_native_provisional(write_cache, w0_new, w1_new, w2_new, debug_info)
        self._apply_commit_risk_filter(
            write_cache,
            w0_new,
            w1_new,
            w2_new,
            debug_info,
            risk_tok=risk_tok,
            A_tok=A_tok,
            token_type=token_type,
            num_frames=num_frames,
            overlap_frames=overlap_frames,
        )
        self._apply_commit_ema(write_cache, w0_new, w1_new, w2_new, debug_info)
        self._apply_previous_transient_delta(
            prev_transient_delta,
            w0_new,
            w1_new,
            w2_new,
            debug_info,
        )
        transient_delta_out = transient_delta if self._has_transient_delta(transient_delta) else None
        debug_info.update({
            "ttt_transient_delta_stored": transient_delta_out is not None,
            "ttt_transient_delta_subtract_scale": float(self.transient_delta_subtract_scale),
            "ttt_transient_delta_branch_mask": list(self.transient_delta_branch_mask),
        })

        return WriteResult(
            w0=w0_new,
            w1=w1_new,
            w2=w2_new,
            history=history,
            transient_delta=transient_delta_out,
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
        num_frames: Optional[int] = None,
        overlap_frames: int = 0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
        Optional[Dict[str, Optional[torch.Tensor]]],
    ]:
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

        # Build token_prior: shape [1, l, 1] to broadcast over batch*heads.
        # The LoGeR TTT cache normally uses the full decoder token layout
        # [register, role, patches] per frame.  Some diagnostic paths may cache
        # only patch tokens, so alignment first honors exact length, then falls
        # back to token_type-based patch extraction only when the replay length
        # matches the patch-token count.
        l = k.shape[1]
        prior_flat, align_debug = self._align_prior_to_replay_tokens(
            A_tok,
            token_type=token_type,
            cache_l=int(l),
        )
        prior_flat, scope_debug = self._apply_token_scope(
            prior_flat,
            cache_l=int(l),
            num_frames=num_frames,
            overlap_frames=overlap_frames,
        )
        prior_flat, transform_debug = self._apply_prior_transform(prior_flat)
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
            "prior_layer_branch_policy": self.prior_layer_branch_policy_text,
            "layer_prior_enabled": bool(layer_prior_enabled),
            "prior_branch_mask": list(active_branch_mask),
            "branch0_prior_enabled": branch_enabled[0],
            "branch1_prior_enabled": branch_enabled[1],
            "branch2_prior_enabled": branch_enabled[2],
            "ttt_write_delta_scale": self.update_delta_scale,
            "ttt_write_delta_scales": list(self.update_delta_scales),
            "ttt_write_delta_applied": bool(layer_prior_enabled and A_tok is not None),
        })
        debug.update(align_debug)
        debug.update(scope_debug)
        debug.update(transform_debug)

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

        k_native_full, v_native_full = k, v
        k_gate_full, v_gate_full, replay_feature_debug = self._apply_replay_feature_gate(
            k, v, prior_flat,
            token_type=token_type,
            num_frames=num_frames,
        )
        debug.update(replay_feature_debug)
        replay_order_full = lc.ttt_op_order
        filter_idx, replay_filter_debug = self._select_replay_token_indices(
            prior_flat,
            cache_l=int(l),
            num_frames=num_frames,
            overlap_frames=overlap_frames,
        )
        if not (layer_prior_enabled and A_tok is not None and len(active_branch_mask) > 0):
            filter_idx = None
            replay_filter_debug["ttt_replay_token_filter_applied"] = False
            replay_filter_debug["ttt_replay_token_filter_tokens_after"] = int(l)
            replay_filter_debug["ttt_replay_token_filter_layer_disabled"] = True
        token_filter_branch_mask = tuple(self.replay_token_filter_branch_mask)
        token_filter_branch_isolated = (
            filter_idx is not None
            and token_filter_branch_mask != (0, 1, 2)
            and len(token_filter_branch_mask) > 0
        )
        if filter_idx is not None:
            idx_dev = filter_idx.to(device=k.device, dtype=torch.long)
            k_native_filt = k_native_full.index_select(1, idx_dev)
            v_native_filt = v_native_full.index_select(1, idx_dev)
            k_gate_filt = k_gate_full.index_select(1, idx_dev)
            v_gate_filt = v_gate_full.index_select(1, idx_dev)
            lr0_filt = lr0.index_select(1, idx_dev)
            lr1_filt = lr1.index_select(1, idx_dev)
            lr2_filt = lr2.index_select(1, idx_dev)
            token_prior_filt = token_prior.index_select(1, idx_dev)
            token_prior0_filt = token_prior0.index_select(1, idx_dev)
            token_prior1_filt = token_prior1.index_select(1, idx_dev)
            token_prior2_filt = token_prior2.index_select(1, idx_dev)
            momentum_filt = momentum.index_select(1, idx_dev) if momentum is not None else None
            replay_order_filt = [(0, int(idx_dev.numel()), True, False)]
        else:
            k_native_filt = v_native_filt = k_gate_filt = v_gate_filt = None
            lr0_filt = lr1_filt = lr2_filt = None
            token_prior_filt = token_prior0_filt = token_prior1_filt = token_prior2_filt = None
            momentum_filt = None
            replay_order_filt = None
        debug.update(replay_filter_debug)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            feature_branch_mask = tuple(self.replay_feature_gate_branch_mask)
            branch_isolated = (
                bool(replay_feature_debug.get("ttt_replay_feature_gate_applied", False))
                and feature_branch_mask != (0, 1, 2)
            )
            def replay_once(
                k_in: torch.Tensor,
                v_in: torch.Tensor,
                lr0_in: torch.Tensor,
                lr1_in: torch.Tensor,
                lr2_in: torch.Tensor,
                token_prior_in: torch.Tensor,
                token_prior0_in: torch.Tensor,
                token_prior1_in: torch.Tensor,
                token_prior2_in: torch.Tensor,
                replay_order_in: Any,
                momentum_in: Optional[torch.Tensor],
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return fast_weight_replay_update(
                    w0_old, w1_old, w2_old,
                    k_in, v_in,
                    lr0_in * lam0, lr1_in * lam1, lr2_in * lam2,
                    token_prior_in,
                    replay_order_in,
                    muon_update_steps=lc.muon_update_steps,
                    momentum=momentum_in,
                    ttt_update_steps=lc.ttt_update_steps,
                    token_prior0=token_prior0_in,
                    token_prior1=token_prior1_in,
                    token_prior2=token_prior2_in,
                )

            def replay_with_feature_select(
                k_native_in: torch.Tensor,
                v_native_in: torch.Tensor,
                k_gate_in: torch.Tensor,
                v_gate_in: torch.Tensor,
                lr0_in: torch.Tensor,
                lr1_in: torch.Tensor,
                lr2_in: torch.Tensor,
                token_prior_in: torch.Tensor,
                token_prior0_in: torch.Tensor,
                token_prior1_in: torch.Tensor,
                token_prior2_in: torch.Tensor,
                replay_order_in: Any,
                momentum_in: Optional[torch.Tensor],
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if branch_isolated:
                    w0_base, w1_base, w2_base = replay_once(
                        k_native_in, v_native_in,
                        lr0_in, lr1_in, lr2_in,
                        token_prior_in, token_prior0_in, token_prior1_in, token_prior2_in,
                        replay_order_in, momentum_in,
                    )
                    w0_gate, w1_gate, w2_gate = replay_once(
                        k_gate_in, v_gate_in,
                        lr0_in, lr1_in, lr2_in,
                        token_prior_in, token_prior0_in, token_prior1_in, token_prior2_in,
                        replay_order_in, momentum_in,
                    )
                    return (
                        w0_gate if 0 in feature_branch_mask else w0_base,
                        w1_gate if 1 in feature_branch_mask else w1_base,
                        w2_gate if 2 in feature_branch_mask else w2_base,
                    )
                return replay_once(
                    k_gate_in, v_gate_in,
                    lr0_in, lr1_in, lr2_in,
                    token_prior_in, token_prior0_in, token_prior1_in, token_prior2_in,
                    replay_order_in, momentum_in,
                )

            token_filter_blend = min(max(float(self.replay_token_filter_blend), 0.0), 1.0)
            token_filter_blend_mode = str(self.replay_token_filter_blend_mode or "linear").strip().lower()
            token_filter_blend_debug: Dict[str, Any] = {}
            transient_delta: Dict[str, Optional[torch.Tensor]] = {"w0": None, "w1": None, "w2": None}

            def renorm_like(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
                ref_norm = reference.detach().float().norm(dim=1, keepdim=True)
                out = candidate.float()
                out = out / (out.norm(dim=1, keepdim=True) + 1e-5) * ref_norm
                return out.to(reference.dtype)

            def store_transient_delta(
                branch_name: str,
                candidate: torch.Tensor,
                filt: torch.Tensor,
            ) -> None:
                delta = candidate.float() - filt.float()
                transient_delta[branch_name] = delta.detach().cpu().to(dtype=candidate.dtype)
                delta_norm = delta.detach().float().norm(dim=1)
                filt_delta_norm = (filt.float() - candidate.float()).detach().float().norm(dim=1)
                token_filter_blend_debug[f"ttt_transient_delta_{branch_name}_stored"] = True
                token_filter_blend_debug[f"ttt_transient_delta_{branch_name}_norm_mean"] = float(
                    delta_norm.mean().item()
                ) if delta_norm.numel() else 0.0
                token_filter_blend_debug[f"ttt_transient_delta_{branch_name}_norm_max"] = float(
                    delta_norm.max().item()
                ) if delta_norm.numel() else 0.0
                token_filter_blend_debug[f"ttt_transient_delta_{branch_name}_check_norm_mean"] = float(
                    filt_delta_norm.mean().item()
                ) if filt_delta_norm.numel() else 0.0

            def maybe_blend_token_filter(
                base: torch.Tensor,
                filt: torch.Tensor,
                old: torch.Tensor,
                branch_name: str,
            ) -> torch.Tensor:
                if token_filter_blend_mode in {"ttl_dynamic", "transient_dynamic", "dynamic_ttl"}:
                    store_transient_delta(branch_name, base, filt)
                    token_filter_blend_debug[f"ttt_replay_token_filter_{branch_name}_ttl_mode"] = "dynamic"
                    return base
                if token_filter_blend_mode in {
                    "project_anti_dynamic",
                    "proj_anti_dynamic",
                    "anti_dynamic_project",
                    "project_dynamic_residual",
                }:
                    if token_filter_blend <= 0.0:
                        return base
                    static_delta = filt.float() - old.float()
                    dynamic_delta = base.float() - filt.float()
                    denom = (static_delta * static_delta).sum(dim=1, keepdim=True) + 1e-6
                    coeff = (dynamic_delta * static_delta).sum(dim=1, keepdim=True) / denom
                    aligned_dynamic = coeff.clamp(min=0.0) * static_delta
                    anti_dynamic = dynamic_delta - aligned_dynamic
                    candidate = base.float() - token_filter_blend * anti_dynamic
                    dynamic_norm = dynamic_delta.detach().float().norm(dim=1)
                    anti_norm = anti_dynamic.detach().float().norm(dim=1)
                    token_filter_blend_debug[f"ttt_replay_token_filter_{branch_name}_proj_coeff_mean"] = float(
                        coeff.detach().float().mean().item()
                    )
                    token_filter_blend_debug[f"ttt_replay_token_filter_{branch_name}_anti_dyn_norm_mean"] = float(
                        anti_norm.mean().item()
                    ) if anti_norm.numel() else 0.0
                    token_filter_blend_debug[f"ttt_replay_token_filter_{branch_name}_anti_dyn_fraction_mean"] = float(
                        (anti_norm / (dynamic_norm + 1e-6)).mean().item()
                    ) if anti_norm.numel() else 0.0
                    return renorm_like(base, candidate)
                if token_filter_blend_mode in {"aligned_dynamic", "align_dynamic", "aligned_dyn", "align_dyn"}:
                    if token_filter_blend <= 0.0:
                        return base
                    if token_filter_blend >= 1.0:
                        return filt
                    static_delta = filt.float() - old.float()
                    dynamic_delta = base.float() - filt.float()
                    denom = (
                        static_delta.norm(dim=1, keepdim=True)
                        * dynamic_delta.norm(dim=1, keepdim=True)
                        + 1e-6
                    )
                    align_cos = (static_delta * dynamic_delta).sum(dim=1, keepdim=True) / denom
                    dyn_keep = (1.0 - token_filter_blend) * align_cos.clamp(min=0.0, max=1.0)
                    candidate = filt.float() + dyn_keep * dynamic_delta
                    token_filter_blend_debug[f"ttt_replay_token_filter_{branch_name}_align_cos_mean"] = float(
                        align_cos.detach().float().mean().item()
                    )
                    token_filter_blend_debug[f"ttt_replay_token_filter_{branch_name}_dyn_keep_mean"] = float(
                        dyn_keep.detach().float().mean().item()
                    )
                    token_filter_blend_debug[f"ttt_replay_token_filter_{branch_name}_dyn_keep_max"] = float(
                        dyn_keep.detach().float().max().item()
                    )
                    return renorm_like(base, candidate)
                if token_filter_blend_mode in {
                    "ttl_aligned_dynamic",
                    "transient_aligned_dynamic",
                    "aligned_dynamic_ttl",
                    "align_dynamic_ttl",
                }:
                    static_delta = filt.float() - old.float()
                    dynamic_delta = base.float() - filt.float()
                    denom = (
                        static_delta.norm(dim=1, keepdim=True)
                        * dynamic_delta.norm(dim=1, keepdim=True)
                        + 1e-6
                    )
                    align_cos = (static_delta * dynamic_delta).sum(dim=1, keepdim=True) / denom
                    dyn_keep = (1.0 - token_filter_blend) * align_cos.clamp(min=0.0, max=1.0)
                    candidate = renorm_like(base, filt.float() + dyn_keep * dynamic_delta)
                    store_transient_delta(branch_name, candidate, filt)
                    token_filter_blend_debug[f"ttt_replay_token_filter_{branch_name}_align_cos_mean"] = float(
                        align_cos.detach().float().mean().item()
                    )
                    token_filter_blend_debug[f"ttt_replay_token_filter_{branch_name}_dyn_keep_mean"] = float(
                        dyn_keep.detach().float().mean().item()
                    )
                    token_filter_blend_debug[f"ttt_replay_token_filter_{branch_name}_dyn_keep_max"] = float(
                        dyn_keep.detach().float().max().item()
                    )
                    token_filter_blend_debug[f"ttt_replay_token_filter_{branch_name}_ttl_mode"] = "aligned_dynamic"
                    return candidate
                if token_filter_blend <= 0.0:
                    return base
                if token_filter_blend >= 1.0:
                    return filt
                return self._scale_delta_and_renorm(base, filt, token_filter_blend)

            if token_filter_branch_isolated:
                w0_base, w1_base, w2_base = replay_with_feature_select(
                    k_native_full, v_native_full, k_gate_full, v_gate_full,
                    lr0, lr1, lr2,
                    token_prior, token_prior0, token_prior1, token_prior2,
                    replay_order_full, momentum,
                )
                w0_filt, w1_filt, w2_filt = replay_with_feature_select(
                    k_native_filt, v_native_filt, k_gate_filt, v_gate_filt,
                    lr0_filt, lr1_filt, lr2_filt,
                        token_prior_filt, token_prior0_filt, token_prior1_filt, token_prior2_filt,
                        replay_order_filt, momentum_filt,
                    )
                w0_new = maybe_blend_token_filter(w0_base, w0_filt, w0_old, "w0") if 0 in token_filter_branch_mask else w0_base
                w1_new = maybe_blend_token_filter(w1_base, w1_filt, w1_old, "w1") if 1 in token_filter_branch_mask else w1_base
                w2_new = maybe_blend_token_filter(w2_base, w2_filt, w2_old, "w2") if 2 in token_filter_branch_mask else w2_base
            else:
                if filter_idx is not None and token_filter_branch_mask == (0, 1, 2):
                    w0_base, w1_base, w2_base = replay_with_feature_select(
                        k_native_full, v_native_full, k_gate_full, v_gate_full,
                        lr0, lr1, lr2,
                        token_prior, token_prior0, token_prior1, token_prior2,
                        replay_order_full, momentum,
                    )
                    w0_filt, w1_filt, w2_filt = replay_with_feature_select(
                        k_native_filt, v_native_filt, k_gate_filt, v_gate_filt,
                        lr0_filt, lr1_filt, lr2_filt,
                        token_prior_filt, token_prior0_filt, token_prior1_filt, token_prior2_filt,
                        replay_order_filt, momentum_filt,
                    )
                    w0_new = maybe_blend_token_filter(w0_base, w0_filt, w0_old, "w0")
                    w1_new = maybe_blend_token_filter(w1_base, w1_filt, w1_old, "w1")
                    w2_new = maybe_blend_token_filter(w2_base, w2_filt, w2_old, "w2")
                else:
                    w0_new, w1_new, w2_new = replay_with_feature_select(
                        k_native_full, v_native_full, k_gate_full, v_gate_full,
                        lr0, lr1, lr2,
                        token_prior, token_prior0, token_prior1, token_prior2,
                        replay_order_full, momentum,
                    )

        debug.update({
            "ttt_replay_feature_branch_mask": list(self.replay_feature_gate_branch_mask),
            "ttt_replay_feature_branch_isolated": bool(
                replay_feature_debug.get("ttt_replay_feature_gate_applied", False)
                and tuple(self.replay_feature_gate_branch_mask) != (0, 1, 2)
            ),
            "ttt_replay_token_filter_branch_mask": list(self.replay_token_filter_branch_mask),
            "ttt_replay_token_filter_branch_isolated": bool(token_filter_branch_isolated),
            "ttt_replay_token_filter_blend": min(max(float(self.replay_token_filter_blend), 0.0), 1.0),
            "ttt_replay_token_filter_blend_mode": str(self.replay_token_filter_blend_mode or "linear").strip().lower(),
        })
        debug.update(token_filter_blend_debug)

        if layer_prior_enabled and A_tok is not None:
            s0, s1, s2 = self.update_delta_scales
            if s0 != 1.0:
                w0_new = self._scale_delta_and_renorm(w0_old, w0_new, s0)
            if s1 != 1.0:
                w1_new = self._scale_delta_and_renorm(w1_old, w1_new, s1)
            if s2 != 1.0:
                w2_new = self._scale_delta_and_renorm(w2_old, w2_new, s2)

        transient_out = transient_delta if any(v is not None for v in transient_delta.values()) else None
        return w0_new.cpu(), w1_new.cpu(), w2_new.cpu(), debug, transient_out

    def _align_prior_to_replay_tokens(
        self,
        A_tok: Optional[torch.Tensor],
        *,
        token_type: Optional[torch.Tensor],
        cache_l: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Align HMC priors to the TTT replay cache token layout."""
        debug: Dict[str, Any] = {
            "ttt_prior_alignment_mode": "unity" if A_tok is None else "legacy",
            "ttt_prior_alignment_cache_tokens": int(cache_l),
            "ttt_prior_alignment_full_tokens": int(A_tok.numel()) if A_tok is not None else 0,
            "ttt_prior_alignment_patch_tokens": 0,
            "ttt_prior_alignment_special_tokens": 0,
        }
        if A_tok is None:
            return torch.ones(int(cache_l), dtype=torch.float32), debug

        prior = A_tok.detach().cpu().float().reshape(-1)
        if prior.numel() == cache_l:
            debug["ttt_prior_alignment_mode"] = "direct_length"
            return prior.clone(), debug

        if token_type is not None:
            tt = token_type.detach().cpu().long().reshape(-1)
            if tt.numel() == prior.numel():
                patch_mask = tt == TOKEN_TYPE_PATCH
                patch_prior = prior[patch_mask]
                debug.update({
                    "ttt_prior_alignment_patch_tokens": int(patch_mask.sum().item()),
                    "ttt_prior_alignment_special_tokens": int((~patch_mask).sum().item()),
                })
                if patch_prior.numel() == cache_l:
                    debug["ttt_prior_alignment_mode"] = "patch_token_type"
                    return patch_prior.clone(), debug
            debug.update({
                "ttt_prior_alignment_token_type_mismatch": True,
                "ttt_prior_alignment_token_type_tokens": int(tt.numel()),
            })

        if prior.numel() >= cache_l:
            debug["ttt_prior_alignment_mode"] = "legacy_prefix"
            return prior[:cache_l].clone(), debug
        out = torch.ones(int(cache_l), dtype=torch.float32)
        if prior.numel() > 0:
            out[: int(prior.numel())] = prior
        debug["ttt_prior_alignment_padded"] = True
        return out, debug

    def _apply_prior_transform(self, prior_flat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Optionally reshape write eligibility before it multiplies TTT lr.

        The normal semantic write path uses ``prior`` in ``[0, 1]`` as a
        learning-rate multiplier.  Because the replay update is subsequently
        orthogonalized and weight-normalized, merely reducing lr may not remove
        a harmful dynamic direction.  The anti-dynamic modes deliberately allow
        low-prior tokens to contribute a small negative update, testing whether
        they should be *unlearned* rather than only not written.
        """
        mode = str(self.prior_transform_mode or "none").strip().lower()
        gamma = max(float(self.prior_gamma), 1e-6)
        anti = max(float(self.prior_anti_scale), 0.0)
        p = prior_flat.detach().float().reshape(-1)
        debug: Dict[str, Any] = {
            "ttt_write_prior_transform_mode": mode,
            "ttt_write_prior_transform_gamma": gamma,
            "ttt_write_prior_anti_scale": anti,
            "ttt_write_prior_transform_applied": False,
        }
        if mode in {"", "none", "off"} or p.numel() == 0:
            return prior_flat, debug

        p_min = p.min()
        p_max = p.max()
        denom = (p_max - p_min).clamp_min(1e-6)
        p_norm = ((p - p_min) / denom).clamp(0.0, 1.0)

        if mode in {"focal_static", "static_focal", "pow"}:
            out = p.clamp_min(0.0).pow(gamma)
        elif mode in {"anti_dynamic", "dynamic_anti"}:
            out = p - anti * (1.0 - p)
        elif mode in {"signed_center", "center_signed"}:
            out = 2.0 * p - 1.0
        elif mode in {"signed_focal", "focal_signed"}:
            static = p.clamp_min(0.0).pow(gamma)
            dynamic = (1.0 - p).clamp_min(0.0).pow(gamma)
            out = static - anti * dynamic
        elif mode in {"anti_dynamic_norm", "dynamic_anti_norm"}:
            out = p_norm - anti * (1.0 - p_norm)
        elif mode in {"signed_center_norm", "center_signed_norm"}:
            out = 2.0 * p_norm - 1.0
        elif mode in {"signed_focal_norm", "focal_signed_norm"}:
            static = p_norm.pow(gamma)
            dynamic = (1.0 - p_norm).pow(gamma)
            out = static - anti * dynamic
        else:
            raise ValueError(f"Unsupported TTT write prior transform mode: {self.prior_transform_mode}")

        out = out.to(device=prior_flat.device, dtype=prior_flat.dtype)
        out_cpu = out.detach().float()
        debug.update({
            "ttt_write_prior_transform_applied": True,
            "ttt_write_prior_mean_before": float(p.mean().item()),
            "ttt_write_prior_mean_after": float(out_cpu.mean().item()),
            "ttt_write_prior_min_after": float(out_cpu.min().item()),
            "ttt_write_prior_p10_after": float(torch.quantile(out_cpu, 0.10).item()),
            "ttt_write_prior_p50_after": float(torch.quantile(out_cpu, 0.50).item()),
            "ttt_write_prior_p90_after": float(torch.quantile(out_cpu, 0.90).item()),
            "ttt_write_prior_negative_mass": float((out_cpu < 0).float().mean().item()),
        })
        return out, debug

    def _select_replay_token_indices(
        self,
        prior_flat: torch.Tensor,
        *,
        cache_l: int,
        num_frames: Optional[int],
        overlap_frames: int,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Select a hard replay token subset before Muon/zeropower aggregation.

        Soft lr priors can be largely folded by zeropower normalization and
        fast-weight norm restoration.  This hook changes the update objective
        more directly by removing low-prior tokens from the replay aggregate.
        """
        mode = str(self.replay_token_filter_mode or "none").strip().lower()
        ratio = min(max(float(self.replay_token_filter_ratio), 0.0), 1.0)
        threshold = float(self.replay_token_filter_threshold)
        scope = str(self.replay_token_filter_scope or "all").strip().lower()
        debug: Dict[str, Any] = {
            "ttt_replay_token_filter_mode": mode,
            "ttt_replay_token_filter_ratio": ratio,
            "ttt_replay_token_filter_threshold": threshold,
            "ttt_replay_token_filter_scope": scope,
            "ttt_replay_token_filter_applied": False,
            "ttt_replay_token_filter_tokens_before": int(cache_l),
            "ttt_replay_token_filter_tokens_after": int(cache_l),
        }
        if mode in {"", "none", "off"} or cache_l <= 1 or prior_flat.numel() == 0:
            return None, debug
        prior = prior_flat.detach().float().reshape(-1)
        if prior.numel() != cache_l:
            n = min(int(prior.numel()), int(cache_l))
            tmp = torch.ones(int(cache_l), dtype=torch.float32)
            if n > 0:
                tmp[:n] = prior[:n]
            prior = tmp
        if mode in {"static_topk", "top_static", "topk"}:
            k_keep = max(1, int(round(cache_l * ratio)))
            idx = torch.topk(prior, k=min(k_keep, cache_l), largest=True).indices
        elif mode in {"dynamic_veto", "veto", "threshold"}:
            idx = torch.nonzero(prior >= threshold, as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                idx = torch.topk(prior, k=1, largest=True).indices
        elif mode in {"per_frame_static_topk", "frame_static_topk", "frame_topk"}:
            n_frames = int(num_frames or 0)
            if n_frames <= 0 or cache_l % n_frames != 0:
                debug["ttt_replay_token_filter_invalid_frame_layout"] = True
                return None, debug
            per_frame = cache_l // n_frames
            k_pf = max(1, int(round(per_frame * ratio)))
            chunks: List[torch.Tensor] = []
            for fi in range(n_frames):
                lo = fi * per_frame
                hi = lo + per_frame
                local = torch.topk(prior[lo:hi], k=min(k_pf, per_frame), largest=True).indices + lo
                chunks.append(local)
            idx = torch.cat(chunks, dim=0) if chunks else torch.empty(0, dtype=torch.long)
        elif mode in {
            "scoped_dynamic_veto",
            "overlap_dynamic_veto",
            "scope_dynamic_veto",
            "scoped_veto",
        }:
            scope_mask, scope_debug = self._replay_token_filter_scope_mask(
                cache_l=int(cache_l),
                num_frames=num_frames,
                overlap_frames=overlap_frames,
                scope=scope,
            )
            debug.update(scope_debug)
            if not bool(scope_debug.get("ttt_replay_token_filter_scope_valid", True)):
                return None, debug
            outside = torch.nonzero(~scope_mask, as_tuple=False).reshape(-1)
            scoped_prior = prior[scope_mask]
            scoped_idx = torch.nonzero(scope_mask, as_tuple=False).reshape(-1)
            kept_scoped = scoped_idx[scoped_prior >= threshold]
            if kept_scoped.numel() == 0 and scoped_idx.numel() > 0:
                best = torch.topk(scoped_prior, k=1, largest=True).indices
                kept_scoped = scoped_idx.index_select(0, best)
            idx = torch.cat([outside, kept_scoped], dim=0)
        elif mode in {
            "scoped_static_topk",
            "overlap_static_topk",
            "scope_static_topk",
            "scoped_topk",
        }:
            scope_mask, scope_debug = self._replay_token_filter_scope_mask(
                cache_l=int(cache_l),
                num_frames=num_frames,
                overlap_frames=overlap_frames,
                scope=scope,
            )
            debug.update(scope_debug)
            if not bool(scope_debug.get("ttt_replay_token_filter_scope_valid", True)):
                return None, debug
            outside = torch.nonzero(~scope_mask, as_tuple=False).reshape(-1)
            scoped_prior = prior[scope_mask]
            scoped_idx = torch.nonzero(scope_mask, as_tuple=False).reshape(-1)
            if scoped_idx.numel() == 0:
                return None, debug
            k_keep = max(1, int(round(int(scoped_idx.numel()) * ratio)))
            local = torch.topk(scoped_prior, k=min(k_keep, int(scoped_idx.numel())), largest=True).indices
            kept_scoped = scoped_idx.index_select(0, local)
            idx = torch.cat([outside, kept_scoped], dim=0)
        else:
            raise ValueError(f"Unsupported TTT replay token filter mode: {self.replay_token_filter_mode}")
        idx = torch.sort(idx.to(dtype=torch.long)).values
        if idx.numel() >= cache_l:
            return None, debug
        kept_prior = prior.index_select(0, idx)
        debug.update({
            "ttt_replay_token_filter_applied": True,
            "ttt_replay_token_filter_tokens_after": int(idx.numel()),
            "ttt_replay_token_filter_keep_mass": float(idx.numel() / max(cache_l, 1)),
            "ttt_replay_token_filter_prior_mean_before": float(prior.mean().item()),
            "ttt_replay_token_filter_prior_mean_after": float(kept_prior.mean().item()) if kept_prior.numel() else 1.0,
            "ttt_replay_token_filter_prior_min_after": float(kept_prior.min().item()) if kept_prior.numel() else 1.0,
            "ttt_replay_token_filter_prior_q10_after": float(torch.quantile(kept_prior, 0.10).item()) if kept_prior.numel() else 1.0,
        })
        return idx, debug

    def _replay_token_filter_scope_mask(
        self,
        *,
        cache_l: int,
        num_frames: Optional[int],
        overlap_frames: int,
        scope: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        scope_text = str(scope or "all").strip().lower()
        mask = torch.ones(int(cache_l), dtype=torch.bool)
        debug: Dict[str, Any] = {
            "ttt_replay_token_filter_scope_valid": True,
            "ttt_replay_token_filter_scope_tokens": int(cache_l),
            "ttt_replay_token_filter_scope_mass": 1.0,
            "ttt_replay_token_filter_overlap_frames": int(max(overlap_frames, 0)),
        }
        if scope_text in {"", "all", "full"}:
            return mask, debug
        n_frames = int(num_frames or 0)
        ov = max(int(overlap_frames), 0)
        if n_frames <= 0 or ov <= 0 or cache_l <= 0 or cache_l % n_frames != 0:
            debug["ttt_replay_token_filter_scope_valid"] = False
            return mask, debug
        tokens_per_frame = cache_l // n_frames
        n = min(cache_l, ov * tokens_per_frame)
        mask = torch.zeros(int(cache_l), dtype=torch.bool)
        if scope_text in {"tail_overlap", "overlap_tail", "tail"}:
            mask[-n:] = True
        elif scope_text in {"head_overlap", "overlap_head", "head"}:
            mask[:n] = True
        elif scope_text in {"both_overlap", "overlap_both"}:
            mask[:n] = True
            mask[-n:] = True
        else:
            raise ValueError(f"Unsupported TTT replay token filter scope: {self.replay_token_filter_scope}")
        debug.update({
            "ttt_replay_token_filter_scope_tokens": int(mask.sum().item()),
            "ttt_replay_token_filter_scope_mass": float(mask.float().mean().item()) if mask.numel() else 1.0,
            "ttt_replay_token_filter_tokens_per_frame": int(tokens_per_frame),
        })
        return mask, debug

    def _apply_replay_feature_gate(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        prior_flat: torch.Tensor,
        *,
        token_type: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Dampen low-prior token feature residuals before TTT replay.

        The ordinary write prior scales token learning rates, but the TTT
        update then passes through zeropower normalization.  Centering K/V
        residuals changes the replay gradient direction itself, which is a
        stronger diagnostic for dynamic-region write contamination.  The
        frame-static variants are more local: patch tokens are blended toward
        a static-token centroid from the same frame, avoiding a global centroid
        that can erase chunk-level geometry.
        """
        mode = str(self.replay_feature_gate_mode or "none").strip().lower()
        rho = max(float(self.replay_feature_gate_rho), 0.0)
        min_gate = min(max(float(self.replay_feature_gate_min), 0.0), 1.0)
        debug: Dict[str, Any] = {
            "ttt_replay_feature_gate_mode": mode,
            "ttt_replay_feature_gate_rho": rho,
            "ttt_replay_feature_gate_min": min_gate,
            "ttt_replay_feature_gate_applied": False,
            "ttt_replay_feature_gate_frame_static": False,
        }
        if mode in {"", "none", "off"} or rho <= 0.0 or prior_flat.numel() == 0:
            return k, v, debug
        frame_static_modes = {
            "k_frame_static_center", "key_frame_static_center",
            "v_frame_static_center", "value_frame_static_center",
            "kv_frame_static_center", "both_frame_static_center",
            "frame_static_center",
        }
        global_center_modes = {
            "k_centered", "key_centered",
            "v_centered", "value_centered",
            "kv_centered", "both_centered",
        }
        if mode not in (global_center_modes | frame_static_modes):
            raise ValueError(f"Unsupported TTT replay feature gate mode: {self.replay_feature_gate_mode}")

        prior = prior_flat.detach().float().view(-1)
        if prior.numel() != k.shape[1]:
            n = min(int(prior.numel()), int(k.shape[1]))
            tmp = torch.ones(int(k.shape[1]), dtype=torch.float32)
            if n > 0:
                tmp[:n] = prior[:n]
            prior = tmp
        p_min = prior.min()
        p_max = prior.max()
        denom = (p_max - p_min).clamp_min(1e-6)
        risk = ((p_max - prior) / denom).clamp(0.0, 1.0)
        gate = (1.0 - rho * risk).clamp(min=min_gate, max=1.0)
        static_weight = (1.0 - risk).clamp_min(0.0)
        if float(static_weight.sum().item()) <= 1e-6:
            static_weight = torch.ones_like(static_weight)
        w = static_weight.to(device=k.device, dtype=torch.float32).view(1, -1, 1)
        gate_t = gate.to(device=k.device, dtype=torch.float32).view(1, -1, 1)

        def _center_gate(x: torch.Tensor) -> torch.Tensor:
            x_f = x.float()
            center = (x_f * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            gated = center + gate_t * (x_f - center)
            return gated.to(dtype=x.dtype)

        def _patch_mask_for_replay() -> torch.Tensor:
            mask = torch.ones(int(k.shape[1]), dtype=torch.bool)
            if token_type is None:
                return mask
            tt = token_type.detach().cpu().long().reshape(-1)
            if tt.numel() == mask.numel():
                return tt == TOKEN_TYPE_PATCH
            if tt.numel() == prior.numel():
                # In the normal full-token path this branch is not needed, but
                # it keeps diagnostics sane if prior was prefix/pad aligned.
                n = min(int(tt.numel()), int(mask.numel()))
                out = torch.zeros_like(mask)
                out[:n] = tt[:n] == TOKEN_TYPE_PATCH
                return out if bool(out.any().item()) else mask
            return mask

        def _frame_static_gate(x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
            n_frames = int(num_frames or 0)
            n_tokens = int(x.shape[1])
            if n_frames <= 0 or n_tokens <= 0 or n_tokens % n_frames != 0:
                return _center_gate(x), {
                    "ttt_replay_feature_gate_frame_static_invalid": True,
                    "ttt_replay_feature_gate_frame_static_fallback": "global_center",
                }
            tokens_per_frame = n_tokens // n_frames
            patch_mask = _patch_mask_for_replay()
            if patch_mask.numel() != n_tokens:
                patch_mask = torch.ones(n_tokens, dtype=torch.bool)
            if not bool(patch_mask.any().item()):
                patch_mask = torch.ones(n_tokens, dtype=torch.bool)

            x_f = x.float()
            out = x_f.clone()
            static_w = static_weight.detach().float().reshape(-1)
            gate_vec = gate.detach().float().reshape(-1)
            apply_tokens = 0
            pool_tokens = 0
            for fi in range(n_frames):
                lo = fi * tokens_per_frame
                hi = lo + tokens_per_frame
                frame_patch = patch_mask[lo:hi]
                if bool(frame_patch.any().item()):
                    frame_apply = frame_patch
                else:
                    frame_apply = torch.ones(tokens_per_frame, dtype=torch.bool)
                local_static = static_w[lo:hi].clone()
                local_pool = frame_apply & (local_static > 0)
                if not bool(local_pool.any().item()):
                    local_pool = frame_apply
                    local_static = torch.ones_like(local_static)
                pool_idx = torch.nonzero(local_pool, as_tuple=False).reshape(-1) + lo
                apply_idx = torch.nonzero(frame_apply, as_tuple=False).reshape(-1) + lo
                if pool_idx.numel() == 0 or apply_idx.numel() == 0:
                    continue
                weights = static_w.index_select(0, pool_idx).to(device=x.device, dtype=torch.float32)
                if float(weights.sum().item()) <= 1e-6:
                    weights = torch.ones_like(weights)
                denom = weights.sum().clamp_min(1e-6)
                center = (
                    x_f.index_select(1, pool_idx.to(device=x.device))
                    * weights.view(1, -1, 1)
                ).sum(dim=1, keepdim=True) / denom
                apply_dev = apply_idx.to(device=x.device)
                g_local = gate_vec.index_select(0, apply_idx).to(device=x.device, dtype=torch.float32).view(1, -1, 1)
                vals = x_f.index_select(1, apply_dev)
                out[:, apply_dev, :] = center + g_local * (vals - center)
                apply_tokens += int(apply_idx.numel())
                pool_tokens += int(pool_idx.numel())
            return out.to(dtype=x.dtype), {
                "ttt_replay_feature_gate_frame_static_invalid": False,
                "ttt_replay_feature_gate_tokens_per_frame": int(tokens_per_frame),
                "ttt_replay_feature_gate_patch_tokens": int(patch_mask.sum().item()),
                "ttt_replay_feature_gate_frame_static_apply_tokens": int(apply_tokens),
                "ttt_replay_feature_gate_frame_static_pool_tokens": int(pool_tokens),
            }

        targets: List[str] = []
        frame_extra: Dict[str, Any] = {}
        use_frame_static = mode in frame_static_modes
        if mode in {
            "k_centered", "key_centered", "kv_centered", "both_centered",
            "k_frame_static_center", "key_frame_static_center",
            "kv_frame_static_center", "both_frame_static_center",
            "frame_static_center",
        }:
            if use_frame_static:
                k, frame_extra = _frame_static_gate(k)
            else:
                k = _center_gate(k)
            targets.append("k")
        if mode in {
            "v_centered", "value_centered", "kv_centered", "both_centered",
            "v_frame_static_center", "value_frame_static_center",
            "kv_frame_static_center", "both_frame_static_center",
            "frame_static_center",
        }:
            if use_frame_static:
                v, frame_extra_v = _frame_static_gate(v)
                frame_extra.update(frame_extra_v)
            else:
                v = _center_gate(v)
            targets.append("v")

        gate_cpu = gate.detach().float()
        risk_cpu = risk.detach().float()
        debug.update({
            "ttt_replay_feature_gate_applied": True,
            "ttt_replay_feature_gate_targets": targets,
            "ttt_replay_feature_gate_mean": float(gate_cpu.mean().item()),
            "ttt_replay_feature_gate_p10": float(torch.quantile(gate_cpu, 0.10).item()),
            "ttt_replay_feature_gate_p50": float(torch.quantile(gate_cpu, 0.50).item()),
            "ttt_replay_feature_gate_p90": float(torch.quantile(gate_cpu, 0.90).item()),
            "ttt_replay_feature_gate_mean_abs_delta": float((1.0 - gate_cpu).mean().item()),
            "ttt_replay_feature_gate_max_abs_delta": float((1.0 - gate_cpu).max().item()),
            "ttt_replay_feature_risk_mean": float(risk_cpu.mean().item()),
            "ttt_replay_feature_risk_q90": float(torch.quantile(risk_cpu, 0.90).item()),
            "ttt_replay_feature_prior_min": float(p_min.item()),
            "ttt_replay_feature_prior_max": float(p_max.item()),
            "ttt_replay_feature_gate_frame_static": bool(use_frame_static),
        })
        debug.update(frame_extra)
        return k, v, debug

    def _mix_with_native_provisional(
        self,
        write_cache: WriteCacheOutput,
        w0_new: List[Optional[torch.Tensor]],
        w1_new: List[Optional[torch.Tensor]],
        w2_new: List[Optional[torch.Tensor]],
        debug_info: Dict[str, Any],
    ) -> None:
        """Interpolate semantic replay against the native replay result.

        ``update_delta_scales`` scales semantic replay relative to the old fast
        weights.  This mixer is different: it treats the native replay as the
        continuity-preserving anchor and applies only a fraction of the semantic
        correction.  A scale of 1.0 is current behavior; 0.0 is native write.
        """
        s0, s1, s2 = self.update_native_mix_scales
        if s0 == 1.0 and s1 == 1.0 and s2 == 1.0:
            debug_info["ttt_write_native_mix_applied"] = False
            debug_info["ttt_write_native_mix_scales"] = [s0, s1, s2]
            return

        branches = (
            ("w0", w0_new, write_cache.w0_provisional, s0),
            ("w1", w1_new, write_cache.w1_provisional, s1),
            ("w2", w2_new, write_cache.w2_provisional, s2),
        )
        applied = 0
        for name, semantic_list, native_list, scale in branches:
            for li, semantic in enumerate(semantic_list):
                if semantic is None or li >= len(native_list):
                    continue
                native = native_list[li]
                if native is None:
                    continue
                semantic_list[li] = self._scale_delta_and_renorm(native, semantic, scale)
                layer_debug = debug_info.get(f"layer_{li}")
                if isinstance(layer_debug, dict):
                    layer_debug[f"ttt_write_native_mix_{name}"] = float(scale)
                applied += 1

        debug_info["ttt_write_native_mix_applied"] = True
        debug_info["ttt_write_native_mix_scales"] = [s0, s1, s2]
        debug_info["ttt_write_native_mix_num_tensors"] = int(applied)

    def _apply_commit_risk_filter(
        self,
        write_cache: WriteCacheOutput,
        w0_new: List[Optional[torch.Tensor]],
        w1_new: List[Optional[torch.Tensor]],
        w2_new: List[Optional[torch.Tensor]],
        debug_info: Dict[str, Any],
        *,
        risk_tok: Optional[torch.Tensor],
        A_tok: Optional[torch.Tensor],
        token_type: Optional[torch.Tensor],
        num_frames: Optional[int],
        overlap_frames: int,
    ) -> None:
        """Filter only the committed TTT state that propagates to next chunk.

        Replay-time gates can remove information the current chunk still needs.
        This post-filter leaves the replay objective intact, then shortens or
        redirects the final fast-weight commit based on dynamic risk in the
        overlap region that will be handed to the next chunk.
        """
        mode = str(self.commit_filter_mode or "none").strip().lower()
        debug_info["ttt_write_commit_filter_mode"] = mode
        if mode in {"", "none", "off"}:
            debug_info["ttt_write_commit_filter_applied"] = False
            return
        if mode not in {
            "native_to_candidate_by_risk",
            "native2candidate_by_risk",
            "native_to_semantic_by_risk",
            "old_decay_by_risk",
        }:
            raise ValueError(f"Unsupported TTT commit filter mode: {self.commit_filter_mode}")

        risk_source = str(self.commit_filter_risk_source or "d_tok").strip().lower()
        if risk_source in {"d", "dyn", "dynamic"}:
            risk_source = "d_tok"
        if risk_source in {"prior", "write", "write_prior"}:
            risk_source = "write_prior"
        if risk_source not in {"d_tok", "write_prior"}:
            raise ValueError(f"Unsupported TTT commit filter risk source: {self.commit_filter_risk_source}")

        source_tok = risk_tok if risk_source == "d_tok" else A_tok
        if source_tok is None:
            debug_info.update({
                "ttt_write_commit_filter_applied": False,
                "ttt_write_commit_filter_missing_risk": True,
                "ttt_write_commit_filter_risk_source": risk_source,
            })
            return

        branch_mask = tuple(self.commit_filter_branch_mask)
        if len(branch_mask) == 0:
            debug_info.update({
                "ttt_write_commit_filter_applied": False,
                "ttt_write_commit_filter_branch_mask": [],
            })
            return

        base = float(self.commit_filter_base)
        gain = float(self.commit_filter_gain)
        lo = min(float(self.commit_filter_min), float(self.commit_filter_max))
        hi = max(float(self.commit_filter_min), float(self.commit_filter_max))
        scope = str(self.commit_filter_scope or "tail_overlap").strip().lower()
        stat_name = str(self.commit_filter_stat or "mean").strip().lower()

        branches = (
            ("w0", 0, w0_new, write_cache.w0_provisional, "w0_old"),
            ("w1", 1, w1_new, write_cache.w1_provisional, "w1_old"),
            ("w2", 2, w2_new, write_cache.w2_provisional, "w2_old"),
        )

        n_layers = len(write_cache.layer_caches)
        applied = 0
        risk_values: List[float] = []
        scale_values: List[float] = []
        invalid_scope = False

        for li, lc in enumerate(write_cache.layer_caches):
            if not self._layer_prior_enabled(li, n_layers):
                continue
            cache_l = int(lc.k.shape[1]) if lc.k is not None and lc.k.ndim >= 2 else 0
            if cache_l <= 0:
                continue
            risk_flat, align_debug = self._align_prior_to_replay_tokens(
                source_tok,
                token_type=token_type,
                cache_l=cache_l,
            )
            risk_flat = risk_flat.detach().float().reshape(-1).clamp(0.0, 1.0)
            if risk_source == "write_prior":
                risk_flat = (1.0 - risk_flat).clamp(0.0, 1.0)
            scope_mask, scope_debug = self._commit_filter_scope_mask(
                cache_l=cache_l,
                num_frames=num_frames,
                overlap_frames=overlap_frames,
                scope=scope,
            )
            if not bool(scope_debug.get("ttt_write_commit_filter_scope_valid", True)):
                invalid_scope = True
            if scope_mask.numel() != risk_flat.numel() or not bool(scope_mask.any().item()):
                selected = risk_flat
            else:
                selected = risk_flat[scope_mask]
            risk_stat = self._commit_filter_stat(selected, stat_name)
            risk_values.append(risk_stat)
            scale = base - gain * risk_stat if mode == "old_decay_by_risk" else base + gain * risk_stat
            scale = max(lo, min(hi, float(scale)))
            scale_values.append(scale)

            layer_debug = debug_info.get(f"layer_{li}")
            if isinstance(layer_debug, dict):
                layer_debug.update({
                    "ttt_write_commit_filter_risk": risk_stat,
                    "ttt_write_commit_filter_scale": scale,
                    "ttt_write_commit_filter_scope": scope,
                    "ttt_write_commit_filter_stat": stat_name,
                    "ttt_write_commit_filter_risk_source": risk_source,
                })
                layer_debug.update({
                    f"commit_filter_{k}": v
                    for k, v in align_debug.items()
                    if k.startswith("ttt_prior_alignment_")
                })
                layer_debug.update(scope_debug)

            for name, branch_idx, values, native_list, old_attr in branches:
                if branch_idx not in branch_mask:
                    continue
                if li >= len(values) or li >= len(native_list):
                    continue
                candidate = values[li]
                native = native_list[li]
                old = getattr(lc, old_attr, None)
                if candidate is None:
                    continue
                if mode in {
                    "native_to_candidate_by_risk",
                    "native2candidate_by_risk",
                    "native_to_semantic_by_risk",
                }:
                    anchor = native
                else:
                    anchor = old
                if anchor is None:
                    continue
                anchor_t = anchor.to(device=candidate.device, dtype=candidate.dtype)
                values[li] = self._scale_delta_and_renorm(anchor_t, candidate, scale).cpu()
                if isinstance(layer_debug, dict):
                    layer_debug[f"ttt_write_commit_filter_{name}_scale"] = scale
                applied += 1

        debug_info.update({
            "ttt_write_commit_filter_applied": bool(applied),
            "ttt_write_commit_filter_mode": mode,
            "ttt_write_commit_filter_risk_source": risk_source,
            "ttt_write_commit_filter_scope": scope,
            "ttt_write_commit_filter_stat": stat_name,
            "ttt_write_commit_filter_base": base,
            "ttt_write_commit_filter_gain": gain,
            "ttt_write_commit_filter_min": lo,
            "ttt_write_commit_filter_max": hi,
            "ttt_write_commit_filter_branch_mask": list(branch_mask),
            "ttt_write_commit_filter_num_tensors": int(applied),
            "ttt_write_commit_filter_scope_invalid": bool(invalid_scope),
            "ttt_write_commit_filter_risk_mean": (
                float(sum(risk_values) / len(risk_values)) if risk_values else 0.0
            ),
            "ttt_write_commit_filter_scale_mean": (
                float(sum(scale_values) / len(scale_values)) if scale_values else 1.0
            ),
            "ttt_write_commit_filter_scale_min": float(min(scale_values)) if scale_values else 1.0,
            "ttt_write_commit_filter_scale_max": float(max(scale_values)) if scale_values else 1.0,
        })

    def _commit_filter_scope_mask(
        self,
        *,
        cache_l: int,
        num_frames: Optional[int],
        overlap_frames: int,
        scope: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        scope_text = str(scope or "all").strip().lower()
        mask = torch.ones(int(cache_l), dtype=torch.bool)
        debug: Dict[str, Any] = {
            "ttt_write_commit_filter_scope_valid": True,
            "ttt_write_commit_filter_scope_tokens": int(cache_l),
            "ttt_write_commit_filter_scope_mass": 1.0,
            "ttt_write_commit_filter_overlap_frames": int(max(overlap_frames, 0)),
        }
        if scope_text in {"", "all", "full"}:
            return mask, debug
        n_frames = int(num_frames or 0)
        ov = max(int(overlap_frames), 0)
        if n_frames <= 0 or ov <= 0 or cache_l <= 0 or cache_l % n_frames != 0:
            debug["ttt_write_commit_filter_scope_valid"] = False
            return mask, debug
        tokens_per_frame = cache_l // n_frames
        n = min(cache_l, ov * tokens_per_frame)
        mask = torch.zeros(int(cache_l), dtype=torch.bool)
        if scope_text in {"tail_overlap", "overlap_tail", "tail"}:
            mask[-n:] = True
        elif scope_text in {"head_overlap", "overlap_head", "head"}:
            mask[:n] = True
        elif scope_text in {"both_overlap", "overlap_both"}:
            mask[:n] = True
            mask[-n:] = True
        else:
            raise ValueError(f"Unsupported TTT commit filter scope: {self.commit_filter_scope}")
        debug.update({
            "ttt_write_commit_filter_scope_tokens": int(mask.sum().item()),
            "ttt_write_commit_filter_scope_mass": float(mask.float().mean().item()) if mask.numel() else 1.0,
            "ttt_write_commit_filter_tokens_per_frame": int(tokens_per_frame),
        })
        return mask, debug

    @staticmethod
    def _commit_filter_stat(values: torch.Tensor, stat_name: str) -> float:
        if values.numel() == 0:
            return 0.0
        stat = str(stat_name or "mean").strip().lower()
        vals = values.detach().float().reshape(-1)
        if stat in {"mean", "avg"}:
            return float(vals.mean().item())
        if stat in {"q90", "p90", "quantile90"}:
            return float(torch.quantile(vals, 0.90).item())
        if stat in {"q75", "p75", "quantile75"}:
            return float(torch.quantile(vals, 0.75).item())
        if stat in {"max", "peak"}:
            return float(vals.max().item())
        if stat in {"mass_gt_05", "mass>0.5", "gt05", "gt_05"}:
            return float((vals > 0.5).float().mean().item())
        raise ValueError(f"Unsupported TTT commit filter stat: {stat_name}")

    def _apply_commit_ema(
        self,
        write_cache: WriteCacheOutput,
        w0_new: List[Optional[torch.Tensor]],
        w1_new: List[Optional[torch.Tensor]],
        w2_new: List[Optional[torch.Tensor]],
        debug_info: Dict[str, Any],
    ) -> None:
        """EMA-smooth the final committed fast weights against W_m.

        This is deliberately placed after semantic replay, native-delta gate,
        and native-mix.  It tests the final write lifetime/objective directly:
        W_commit = W_old + alpha * (W_candidate - W_old).
        """
        alpha = float(self.commit_ema_alpha)
        branch_mask = tuple(self.commit_ema_branch_mask)
        debug_info["ttt_write_commit_ema_branch_mask"] = list(branch_mask)
        if alpha == 1.0:
            debug_info["ttt_write_commit_ema_applied"] = False
            debug_info["ttt_write_commit_ema_alpha"] = alpha
            return
        alpha = max(0.0, alpha)
        branches = (
            ("w0", 0, w0_new, "w0_old"),
            ("w1", 1, w1_new, "w1_old"),
            ("w2", 2, w2_new, "w2_old"),
        )
        applied = 0
        n_layers = len(write_cache.layer_caches)
        for name, branch_idx, values, old_attr in branches:
            for li, candidate in enumerate(values):
                layer_debug = debug_info.get(f"layer_{li}")
                if isinstance(layer_debug, dict):
                    layer_debug[f"ttt_write_commit_ema_{name}_alpha"] = (
                        alpha if branch_idx in branch_mask else 1.0
                    )
                if not self._layer_prior_enabled(li, n_layers):
                    if isinstance(layer_debug, dict):
                        layer_debug[f"ttt_write_commit_ema_{name}_alpha"] = 1.0
                    continue
                if branch_idx not in branch_mask:
                    continue
                if candidate is None or li >= len(write_cache.layer_caches):
                    continue
                old = getattr(write_cache.layer_caches[li], old_attr, None)
                if old is None:
                    continue
                old_t = old.to(device=candidate.device, dtype=candidate.dtype)
                values[li] = self._scale_delta_and_renorm(old_t, candidate, alpha).cpu()
                applied += 1
        debug_info["ttt_write_commit_ema_applied"] = bool(applied)
        debug_info["ttt_write_commit_ema_alpha"] = alpha
        debug_info["ttt_write_commit_ema_num_tensors"] = int(applied)

    @staticmethod
    def _has_transient_delta(
        transient_delta: Optional[Dict[str, List[Optional[torch.Tensor]]]],
    ) -> bool:
        if not isinstance(transient_delta, dict):
            return False
        for values in transient_delta.values():
            if isinstance(values, list) and any(v is not None for v in values):
                return True
        return False

    @staticmethod
    def _renorm_to_reference(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        ref_norm = reference.detach().float().norm(dim=1, keepdim=True)
        out = candidate.float()
        out = out / (out.norm(dim=1, keepdim=True) + 1e-5) * ref_norm
        return out.to(dtype=reference.dtype)

    def _apply_previous_transient_delta(
        self,
        prev_transient_delta: Optional[Dict[str, List[Optional[torch.Tensor]]]],
        w0_new: List[Optional[torch.Tensor]],
        w1_new: List[Optional[torch.Tensor]],
        w2_new: List[Optional[torch.Tensor]],
        debug_info: Dict[str, Any],
    ) -> None:
        """Remove one-hop dynamic residuals before they become long-term TTT memory."""
        scale = float(self.transient_delta_subtract_scale)
        branch_mask = tuple(self.transient_delta_branch_mask)
        debug_info["ttt_transient_delta_prev_present"] = self._has_transient_delta(prev_transient_delta)
        debug_info["ttt_transient_delta_prev_subtract_scale"] = scale
        debug_info["ttt_transient_delta_prev_branch_mask"] = list(branch_mask)
        if scale <= 0.0 or not self._has_transient_delta(prev_transient_delta):
            debug_info["ttt_transient_delta_prev_subtract_applied"] = False
            debug_info["ttt_transient_delta_prev_subtract_tensors"] = 0
            return

        branches = (
            ("w0", 0, w0_new),
            ("w1", 1, w1_new),
            ("w2", 2, w2_new),
        )
        applied = 0
        norm_vals: List[float] = []
        for branch_name, branch_idx, values in branches:
            if branch_idx not in branch_mask:
                continue
            prev_values = prev_transient_delta.get(branch_name) if isinstance(prev_transient_delta, dict) else None
            if not isinstance(prev_values, list):
                continue
            for li, candidate in enumerate(values):
                if candidate is None or li >= len(prev_values):
                    continue
                prev_delta = prev_values[li]
                if prev_delta is None or tuple(prev_delta.shape) != tuple(candidate.shape):
                    continue
                prev_t = prev_delta.to(device=candidate.device, dtype=candidate.dtype)
                raw = candidate.float() - scale * prev_t.float()
                values[li] = self._renorm_to_reference(candidate, raw).cpu()
                delta_norm = prev_t.detach().float().norm(dim=1)
                if delta_norm.numel():
                    norm_vals.append(float(delta_norm.mean().item()))
                layer_debug = debug_info.get(f"layer_{li}")
                if isinstance(layer_debug, dict):
                    layer_debug[f"ttt_transient_delta_prev_subtract_{branch_name}_scale"] = scale
                    layer_debug[f"ttt_transient_delta_prev_subtract_{branch_name}_applied"] = True
                applied += 1

        debug_info["ttt_transient_delta_prev_subtract_applied"] = bool(applied)
        debug_info["ttt_transient_delta_prev_subtract_tensors"] = int(applied)
        debug_info["ttt_transient_delta_prev_norm_mean"] = (
            float(torch.tensor(norm_vals).mean().item()) if norm_vals else 0.0
        )

    def _apply_native_delta_gate(
        self,
        write_cache: WriteCacheOutput,
        w0_new: List[Optional[torch.Tensor]],
        w1_new: List[Optional[torch.Tensor]],
        w2_new: List[Optional[torch.Tensor]],
        debug_info: Dict[str, Any],
    ) -> None:
        """Gate semantic replay correction against native replay continuity.

        Token priors enter the TTT update before zeropower normalization, so a
        low prior is not a reliable post-update magnitude control.  This gate
        works after replay: it treats the native provisional update as the
        continuity anchor and only keeps the semantic correction when its
        branch/head delta is compatible with the native update direction, and
        optionally caps the correction magnitude.
        """
        mode = str(self.native_delta_gate_mode or "none").strip().lower()
        if mode in {"", "none", "off"}:
            debug_info["ttt_write_native_delta_gate_applied"] = False
            debug_info["ttt_write_native_delta_gate_mode"] = mode
            return
        if mode not in {"cosine", "cosine_soft", "cap", "cosine_cap"}:
            raise ValueError(f"Unsupported native delta gate mode: {self.native_delta_gate_mode}")

        min_cos = float(self.native_delta_gate_min_cos)
        fallback = min(max(float(self.native_delta_gate_fallback), 0.0), 1.0)
        cap_ratio = max(float(self.native_delta_gate_cap_ratio), 0.0)
        branch_mask = tuple(self.native_delta_gate_branch_mask)
        branches = (
            ("w0", 0, w0_new, write_cache.w0_provisional, "w0_old"),
            ("w1", 1, w1_new, write_cache.w1_provisional, "w1_old"),
            ("w2", 2, w2_new, write_cache.w2_provisional, "w2_old"),
        )
        eps = 1e-8
        applied = 0
        scale_means: List[float] = []
        cosine_means: List[float] = []

        for name, branch_idx, semantic_list, native_list, old_attr in branches:
            if branch_idx not in branch_mask:
                continue
            for li, semantic in enumerate(semantic_list):
                if semantic is None or li >= len(native_list) or li >= len(write_cache.layer_caches):
                    continue
                native = native_list[li]
                if native is None:
                    continue
                old = getattr(write_cache.layer_caches[li], old_attr, None)
                if old is None:
                    continue

                semantic_f = semantic.detach().float()
                native_f = native.detach().float()
                old_f = old.detach().float()
                if old_f.device != semantic_f.device:
                    old_f = old_f.to(device=semantic_f.device)
                if native_f.device != semantic_f.device:
                    native_f = native_f.to(device=semantic_f.device)

                semantic_delta = semantic_f - old_f
                native_delta = native_f - old_f
                correction = semantic_f - native_f
                reduce_dims = tuple(range(1, semantic_f.ndim))
                semantic_norm = torch.linalg.vector_norm(semantic_delta, dim=reduce_dims).clamp_min(eps)
                native_norm = torch.linalg.vector_norm(native_delta, dim=reduce_dims).clamp_min(eps)
                correction_norm = torch.linalg.vector_norm(correction, dim=reduce_dims).clamp_min(eps)
                dot = (semantic_delta * native_delta).sum(dim=reduce_dims)
                cosine = (dot / (semantic_norm * native_norm)).clamp(-1.0, 1.0)
                scale = torch.ones_like(cosine)

                if mode in {"cosine", "cosine_cap"}:
                    scale = torch.where(cosine >= min_cos, scale, torch.full_like(scale, fallback))
                elif mode == "cosine_soft":
                    denom = max(1.0 - min_cos, eps)
                    soft = ((cosine - min_cos) / denom).clamp(0.0, 1.0)
                    scale = fallback + (1.0 - fallback) * soft

                if mode in {"cap", "cosine_cap"} and cap_ratio > 0.0:
                    cap = cap_ratio * native_norm
                    cap_scale = (cap / correction_norm).clamp(max=1.0)
                    scale = torch.minimum(scale, cap_scale)
                elif mode == "cap" and cap_ratio <= 0.0:
                    scale = torch.zeros_like(scale)

                view_shape = [scale.shape[0]] + [1] * (semantic_f.ndim - 1)
                gated = native_f + scale.view(*view_shape) * correction
                old_norm = old_f.norm(dim=1, keepdim=True)
                gated = gated / (gated.norm(dim=1, keepdim=True) + 1e-5) * old_norm
                semantic_list[li] = gated.to(dtype=semantic.dtype).cpu()

                layer_debug = debug_info.get(f"layer_{li}")
                if isinstance(layer_debug, dict):
                    layer_debug[f"ttt_write_native_delta_gate_{name}_mode"] = mode
                    layer_debug[f"ttt_write_native_delta_gate_{name}_scale_mean"] = float(scale.mean().item())
                    layer_debug[f"ttt_write_native_delta_gate_{name}_scale_min"] = float(scale.min().item())
                    layer_debug[f"ttt_write_native_delta_gate_{name}_cos_mean"] = float(cosine.mean().item())
                    layer_debug[f"ttt_write_native_delta_gate_{name}_cos_min"] = float(cosine.min().item())
                    layer_debug[f"ttt_write_native_delta_gate_{name}_cap_ratio"] = float(cap_ratio)
                applied += 1
                scale_means.append(float(scale.mean().item()))
                cosine_means.append(float(cosine.mean().item()))

        debug_info["ttt_write_native_delta_gate_applied"] = bool(applied)
        debug_info["ttt_write_native_delta_gate_mode"] = mode
        debug_info["ttt_write_native_delta_gate_branch_mask"] = list(branch_mask)
        debug_info["ttt_write_native_delta_gate_min_cos"] = min_cos
        debug_info["ttt_write_native_delta_gate_fallback"] = fallback
        debug_info["ttt_write_native_delta_gate_cap_ratio"] = cap_ratio
        debug_info["ttt_write_native_delta_gate_num_tensors"] = int(applied)
        debug_info["ttt_write_native_delta_gate_scale_mean"] = (
            float(sum(scale_means) / len(scale_means)) if scale_means else 1.0
        )
        debug_info["ttt_write_native_delta_gate_cos_mean"] = (
            float(sum(cosine_means) / len(cosine_means)) if cosine_means else 1.0
        )

    def _apply_token_scope(
        self,
        prior_flat: torch.Tensor,
        *,
        cache_l: int,
        num_frames: Optional[int],
        overlap_frames: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Restrict or protect replay contribution around overlap-frame tokens.

        The ``*_native`` and ``*_no_boost`` variants are intentionally not
        hard scopes: they keep the full-chunk semantic prior and only alter
        the selected overlap seam.  This lets us test whether the seam should
        be protected without removing the non-overlap TTT continuity signal.
        """
        scope = str(self.update_token_scope or "all").strip().lower()
        debug: Dict[str, Any] = {
            "ttt_write_token_scope": scope,
            "ttt_write_token_scope_floor": float(self.update_token_scope_floor),
            "ttt_write_scope_applied": False,
            "ttt_write_scope_mass": 1.0,
            "ttt_write_scope_tokens": int(cache_l),
            "ttt_write_scope_overlap_frames": int(max(overlap_frames, 0)),
        }
        if scope in {"", "all", "full"}:
            return prior_flat, debug
        n_frames = int(num_frames or 0)
        ov = max(int(overlap_frames), 0)
        if n_frames <= 0 or ov <= 0 or cache_l <= 0 or cache_l % n_frames != 0:
            debug["ttt_write_scope_invalid"] = True
            return prior_flat, debug
        tokens_per_frame = cache_l // n_frames
        n = min(cache_l, ov * tokens_per_frame)
        mask = torch.zeros(cache_l, dtype=torch.bool)
        veto_scope = scope in {
            "tail_overlap_veto",
            "overlap_tail_veto",
            "tail_veto",
            "head_overlap_veto",
            "overlap_head_veto",
            "head_veto",
            "both_overlap_veto",
            "overlap_both_veto",
        }
        drop_scope = scope in {
            "tail_overlap_drop",
            "overlap_tail_drop",
            "tail_drop",
            "head_overlap_drop",
            "overlap_head_drop",
            "head_drop",
            "both_overlap_drop",
            "overlap_both_drop",
        }
        native_scope = scope in {
            "tail_overlap_native",
            "overlap_tail_native",
            "tail_native",
            "head_overlap_native",
            "overlap_head_native",
            "head_native",
            "both_overlap_native",
            "overlap_both_native",
        }
        no_boost_scope = scope in {
            "tail_overlap_no_boost",
            "overlap_tail_no_boost",
            "tail_no_boost",
            "head_overlap_no_boost",
            "overlap_head_no_boost",
            "head_no_boost",
            "both_overlap_no_boost",
            "overlap_both_no_boost",
        }
        scope_base = (
            scope
            .replace("_veto", "")
            .replace("_drop", "")
            .replace("_native", "")
            .replace("_no_boost", "")
        )
        if scope_base in {"tail_overlap", "overlap_tail", "tail"}:
            mask[-n:] = True
        elif scope_base in {"head_overlap", "overlap_head", "head"}:
            mask[:n] = True
        elif scope_base in {"both_overlap", "overlap_both"}:
            mask[:n] = True
            mask[-n:] = True
        else:
            raise ValueError(f"Unsupported TTT write token scope: {self.update_token_scope}")
        scoped = prior_flat.clone()
        floor = min(max(float(self.update_token_scope_floor), 0.0), 1.0)
        if veto_scope:
            # Veto mode keeps the non-overlap replay native and only applies
            # the semantic prior inside the overlap seam. This is the inverse
            # of the hard/floor scope used for diagnostic tail-only replay.
            scoped[~mask] = 1.0
        elif drop_scope:
            # Drop mode is stronger than veto: preserve the non-overlap replay
            # exactly and suppress the overlap seam itself. This tests whether
            # duplicated overlap frames are the harmful part of the next-chunk
            # fast-weight update.
            scoped[~mask] = 1.0
            scoped[mask] = scoped[mask] * floor
        elif native_scope:
            # Protect the selected seam by blending its semantic prior back
            # toward native replay, while leaving the rest of the chunk's
            # semantic write prior untouched.
            scoped[mask] = 1.0 + floor * (scoped[mask] - 1.0)
        elif no_boost_scope:
            # Keep dynamic suppression in the seam but remove semantic boosts
            # above native.  This tests "write less in risky overlap areas"
            # without deleting continuity-critical value/background updates.
            scoped[mask] = torch.minimum(scoped[mask], torch.ones_like(scoped[mask]))
        else:
            scoped[~mask] = scoped[~mask] * floor
        debug.update({
            "ttt_write_scope_applied": True,
            "ttt_write_scope_veto_mode": bool(veto_scope),
            "ttt_write_scope_drop_mode": bool(drop_scope),
            "ttt_write_scope_native_mode": bool(native_scope),
            "ttt_write_scope_no_boost_mode": bool(no_boost_scope),
            "ttt_write_scope_mass": float(mask.float().mean().item()) if mask.numel() else 1.0,
            "ttt_write_scope_floor": floor,
            "ttt_write_scope_tokens": int(mask.sum().item()),
            "ttt_write_scope_tokens_per_frame": int(tokens_per_frame),
            "ttt_write_scope_prior_mean_before": float(prior_flat.float().mean().item()) if prior_flat.numel() else 1.0,
            "ttt_write_scope_prior_mean_after": float(scoped.float().mean().item()) if scoped.numel() else 1.0,
        })
        return scoped, debug

    @staticmethod
    def _scale_delta_and_renorm(
        old: torch.Tensor,
        new: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scale the replayed fast-weight delta after zeropower normalization."""
        out = old + float(scale) * (new - old)
        old_norm = old.detach().norm(dim=1, keepdim=True)
        return out / (out.norm(dim=1, keepdim=True) + 1e-5) * old_norm

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

    @classmethod
    def _parse_layer_branch_policy(cls, policy: str) -> Tuple[Tuple[str, Tuple[int, int], Tuple[int, ...]], ...]:
        """Parse layer-range branch overrides such as ``0-5:all;6-11:0;12-17:none``.

        Ranges are inclusive.  Named ranges are resolved later because they depend
        on the number of TTT layers: ``early``, ``middle``, and ``late``.
        """
        text = str(policy or "").strip().lower()
        if text in {"", "none", "off", "default"}:
            return ()
        rules: List[Tuple[str, Tuple[int, int], Tuple[int, ...]]] = []
        for raw_rule in text.replace("|", ";").split(";"):
            rule = raw_rule.strip()
            if not rule:
                continue
            if ":" not in rule:
                raise ValueError(
                    "ttt layer-branch policy rules must be formatted as "
                    f"layer-range:branch-mask, got {raw_rule!r}"
                )
            raw_selector, raw_mask = rule.split(":", 1)
            selector = raw_selector.strip().lower()
            branches = cls._parse_branch_mask(raw_mask.strip())
            if selector in {"early", "middle", "late", "all"}:
                rules.append((selector, (-1, -1), branches))
                continue
            if "-" in selector:
                start_s, end_s = selector.split("-", 1)
                start = int(start_s.strip())
                end = int(end_s.strip())
            else:
                start = end = int(selector)
            if start < 0 or end < start:
                raise ValueError(f"Invalid layer range in TTT layer-branch policy: {selector!r}")
            rules.append(("range", (start, end), branches))
        return tuple(rules)

    @staticmethod
    def _parse_delta_scales(scales: Optional[str], *, default: float) -> Tuple[float, float, float]:
        if scales is None:
            return (float(default), float(default), float(default))
        text = str(scales).strip().lower()
        if text in {"", "none", "default"}:
            return (float(default), float(default), float(default))
        if text in {"all", "*"}:
            return (float(default), float(default), float(default))
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) == 1:
            val = float(parts[0])
            return (val, val, val)
        if len(parts) != 3:
            raise ValueError(
                "ttt write delta scales must be one scalar or three comma-separated "
                f"values for w0,w1,w2; got {scales!r}"
            )
        return (float(parts[0]), float(parts[1]), float(parts[2]))

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

    def _layer_branch_mask(self, layer_idx: int, n_layers: int) -> Tuple[int, ...]:
        if not self.prior_layer_branch_policy:
            return self.prior_branch_mask
        selected: Optional[Tuple[int, ...]] = None
        for selector, bounds, branches in self.prior_layer_branch_policy:
            if selector == "all":
                match = True
            elif selector == "early":
                match = layer_idx < max(1, n_layers // 2)
            elif selector == "late":
                match = layer_idx >= n_layers // 2
            elif selector == "middle":
                lo = n_layers // 3
                hi = max(lo + 1, (2 * n_layers) // 3)
                match = lo <= layer_idx < hi
            else:
                start, end = bounds
                match = start <= layer_idx <= end
            if match:
                selected = branches
        return self.prior_branch_mask if selected is None else selected

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
