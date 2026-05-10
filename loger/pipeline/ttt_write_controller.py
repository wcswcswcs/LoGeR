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
    transient_delta: Optional[Dict[str, Any]] = None

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
        special_token_policy: str = "none",
        special_token_floor: float = 0.0,
        special_token_ceiling: float = 1.0,
        gradient_reversal_mode: str = "none",
        gradient_reversal_gamma: float = 0.0,
        gradient_reversal_branch_mask: str = "0",
        gradient_reversal_branch_gammas: Optional[str] = None,
        gradient_reversal_layer_gammas: Optional[str] = None,
        gradient_reversal_head_routes: Optional[str] = None,
        gradient_reversal_negative_frac: float = 0.0,
        gradient_reversal_risk_source: str = "prior",
        tri_replay_positive_frac: float = 0.35,
        tri_replay_negative_frac: float = 0.15,
        tri_replay_neutral_lambda: float = 1.0,
        gradient_reversal_transient_mode: str = "none",
        gradient_reversal_transient_branch_mask: str = "",
        gradient_reversal_transient_long_scale: float = 0.0,
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
        transient_delta_ttl: int = 1,
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
        self.special_token_policy = str(special_token_policy or "none").strip().lower()
        self.special_token_floor = float(special_token_floor)
        self.special_token_ceiling = float(special_token_ceiling)
        self.gradient_reversal_mode = str(gradient_reversal_mode or "none").strip().lower()
        self.gradient_reversal_gamma = float(gradient_reversal_gamma)
        self.gradient_reversal_branch_mask = self._parse_branch_mask(gradient_reversal_branch_mask)
        self.gradient_reversal_branch_gammas = self._parse_branch_gamma_map(gradient_reversal_branch_gammas)
        self.gradient_reversal_layer_gammas = self._parse_layer_gamma_map(gradient_reversal_layer_gammas)
        self.gradient_reversal_head_routes = self._parse_layer_head_routes(gradient_reversal_head_routes)
        self.gradient_reversal_negative_frac = float(gradient_reversal_negative_frac)
        self.gradient_reversal_risk_source = str(gradient_reversal_risk_source or "prior").strip().lower()
        self.tri_replay_positive_frac = float(tri_replay_positive_frac)
        self.tri_replay_negative_frac = float(tri_replay_negative_frac)
        self.tri_replay_neutral_lambda = float(tri_replay_neutral_lambda)
        self.gradient_reversal_transient_mode = str(gradient_reversal_transient_mode or "none").strip().lower()
        self.gradient_reversal_transient_long_scale = float(gradient_reversal_transient_long_scale)
        gr_transient_mask_text = str(gradient_reversal_transient_branch_mask or "").strip().lower()
        self.gradient_reversal_transient_branch_mask = (
            ()
            if gr_transient_mask_text in {"", "same", "active"}
            else self._parse_branch_mask(gradient_reversal_transient_branch_mask)
        )
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
        self.transient_delta_ttl = max(int(transient_delta_ttl), 1)
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
        prev_transient_delta: Optional[Dict[str, Any]] = None,
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
        transient_delta: Dict[str, Any] = {
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
                layer_idx=int(li),
                token_type=token_type,
                risk_tok=risk_tok,
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

        self._summarize_ttt_self_cues(debug_info, n_layers)
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
        carry_transient_delta = self._apply_previous_transient_delta(
            prev_transient_delta,
            w0_new,
            w1_new,
            w2_new,
            debug_info,
        )
        transient_delta_out = transient_delta if self._has_transient_delta(transient_delta) else carry_transient_delta
        if self._has_transient_delta(transient_delta):
            transient_delta_out["_ttl_remaining"] = int(self.transient_delta_ttl)
            mode_tag = str(self.gradient_reversal_transient_mode or "none").strip().lower()
            if mode_tag in {"dual_lifetime", "dual_fast_weight", "apply_short_delta", "short_apply_delta"}:
                transient_delta_out["_mode"] = mode_tag
                transient_delta_out["_apply_scale"] = 1.0
                transient_delta_out["_long_scale"] = float(self.gradient_reversal_transient_long_scale)
            else:
                transient_delta_out["_mode"] = "subtract_delta"
        debug_info.update({
            "ttt_transient_delta_stored": transient_delta_out is not None,
            "ttt_transient_delta_mode_out": str(transient_delta_out.get("_mode", "")) if isinstance(transient_delta_out, dict) else "",
            "ttt_transient_delta_subtract_scale": float(self.transient_delta_subtract_scale),
            "ttt_transient_delta_branch_mask": list(self.transient_delta_branch_mask),
            "ttt_transient_delta_ttl": int(self.transient_delta_ttl),
            "ttt_transient_delta_ttl_out": int(transient_delta_out.get("_ttl_remaining", 0)) if isinstance(transient_delta_out, dict) else 0,
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
        layer_idx: int = -1,
        token_type: Optional[torch.Tensor] = None,
        risk_tok: Optional[torch.Tensor] = None,
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
        prior_flat, special_debug = self._apply_special_token_policy(
            prior_flat,
            token_type=token_type,
            cache_l=int(l),
            align_mode=str(align_debug.get("ttt_prior_alignment_mode", "")),
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
        debug.update(special_debug)
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

        token_prior0_pre_gr = token_prior0
        token_prior1_pre_gr = token_prior1
        token_prior2_pre_gr = token_prior2
        layer_branch_gammas = self._effective_gradient_reversal_branch_gammas(layer_idx)
        gradient_reversal_risk_flat, gradient_reversal_risk_debug = self._build_gradient_reversal_risk_flat(
            lc,
            prior_flat=prior_flat,
            risk_tok=risk_tok,
            token_type=token_type,
            cache_l=int(l),
            effective_branch_gammas=layer_branch_gammas,
        )
        debug.update(gradient_reversal_risk_debug)
        (
            token_prior0,
            token_prior1,
            token_prior2,
        ), gradient_reversal_debug = self._apply_gradient_reversal_prior(
            prior_flat,
            token_prior0,
            token_prior1,
            token_prior2,
            branch_enabled=branch_enabled,
            device=device,
            risk_flat=gradient_reversal_risk_flat,
            effective_branch_gammas=layer_branch_gammas,
            layer_idx=layer_idx,
        )
        debug.update(gradient_reversal_debug)

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
                *,
                scale: float = 1.0,
            ) -> None:
                delta = (candidate.float() - filt.float()) * float(scale)
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

            def maybe_store_gradient_reversal_transient(
                candidate_w0: torch.Tensor,
                candidate_w1: torch.Tensor,
                candidate_w2: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                gr_transient_mode = str(self.gradient_reversal_transient_mode or "none").strip().lower()
                token_filter_blend_debug["ttt_gradient_reversal_transient_mode"] = gr_transient_mode
                token_filter_blend_debug["ttt_gradient_reversal_transient_applied"] = False
                out_w0, out_w1, out_w2 = candidate_w0, candidate_w1, candidate_w2
                if gr_transient_mode in {"", "none", "off"}:
                    return out_w0, out_w1, out_w2
                if gr_transient_mode not in {
                    "one_hop_delta",
                    "onehop_delta",
                    "transient_delta",
                    "ttl_delta",
                    "short_delta",
                    "dual_lifetime",
                    "dual_fast_weight",
                    "apply_short_delta",
                    "short_apply_delta",
                }:
                    raise ValueError(
                        f"Unsupported TTT gradient reversal transient mode: {self.gradient_reversal_transient_mode}"
                    )
                if not bool(gradient_reversal_debug.get("ttt_gradient_reversal_applied", False)):
                    token_filter_blend_debug["ttt_gradient_reversal_transient_skip"] = "no_gradient_reversal"
                    return out_w0, out_w1, out_w2
                if filter_idx is not None:
                    token_filter_blend_debug["ttt_gradient_reversal_transient_skip"] = "token_filter_active"
                    return out_w0, out_w1, out_w2
                active_gr = tuple(
                    int(i)
                    for i in gradient_reversal_debug.get("ttt_gradient_reversal_active_branches", [])
                    if 0 <= int(i) <= 2
                )
                if len(active_gr) == 0:
                    token_filter_blend_debug["ttt_gradient_reversal_transient_skip"] = "no_active_branch"
                    return out_w0, out_w1, out_w2
                transient_mask = (
                    tuple(self.gradient_reversal_transient_branch_mask)
                    if len(self.gradient_reversal_transient_branch_mask) > 0
                    else active_gr
                )
                transient_mask = tuple(int(i) for i in transient_mask if int(i) in active_gr)
                if len(transient_mask) == 0:
                    token_filter_blend_debug["ttt_gradient_reversal_transient_skip"] = "empty_branch_mask"
                    return out_w0, out_w1, out_w2

                ref_w0, ref_w1, ref_w2 = replay_with_feature_select(
                    k_native_full, v_native_full, k_gate_full, v_gate_full,
                    lr0, lr1, lr2,
                    token_prior,
                    token_prior0_pre_gr,
                    token_prior1_pre_gr,
                    token_prior2_pre_gr,
                    replay_order_full,
                    momentum,
                )
                dual_lifetime = gr_transient_mode in {
                    "dual_lifetime",
                    "dual_fast_weight",
                    "apply_short_delta",
                    "short_apply_delta",
                }
                long_scale = min(max(float(self.gradient_reversal_transient_long_scale), 0.0), 1.0)

                def split_long_candidate(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
                    if not dual_lifetime or long_scale <= 0.0:
                        return reference
                    if long_scale >= 1.0:
                        return candidate
                    raw = reference.float() + long_scale * (candidate.float() - reference.float())
                    return renorm_like(reference, raw)

                mix_scales = self.update_native_mix_scales
                if 0 in transient_mask:
                    residual_scale = float(mix_scales[0]) * (1.0 - long_scale if dual_lifetime else 1.0)
                    store_transient_delta("w0", candidate_w0, ref_w0, scale=residual_scale)
                    if dual_lifetime:
                        out_w0 = split_long_candidate(candidate_w0, ref_w0)
                if 1 in transient_mask:
                    residual_scale = float(mix_scales[1]) * (1.0 - long_scale if dual_lifetime else 1.0)
                    store_transient_delta("w1", candidate_w1, ref_w1, scale=residual_scale)
                    if dual_lifetime:
                        out_w1 = split_long_candidate(candidate_w1, ref_w1)
                if 2 in transient_mask:
                    residual_scale = float(mix_scales[2]) * (1.0 - long_scale if dual_lifetime else 1.0)
                    store_transient_delta("w2", candidate_w2, ref_w2, scale=residual_scale)
                    if dual_lifetime:
                        out_w2 = split_long_candidate(candidate_w2, ref_w2)
                token_filter_blend_debug["ttt_gradient_reversal_transient_applied"] = True
                token_filter_blend_debug["ttt_gradient_reversal_transient_dual_lifetime"] = bool(dual_lifetime)
                token_filter_blend_debug["ttt_gradient_reversal_transient_long_scale"] = float(long_scale)
                token_filter_blend_debug["ttt_gradient_reversal_transient_branch_mask"] = list(transient_mask)
                token_filter_blend_debug["ttt_gradient_reversal_transient_native_mix_scales"] = [
                    float(x) for x in mix_scales
                ]
                return out_w0, out_w1, out_w2

            def maybe_apply_two_replay_negative(
                candidate_w0: torch.Tensor,
                candidate_w1: torch.Tensor,
                candidate_w2: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                gr_mode = str(self.gradient_reversal_mode or "none").strip().lower()
                token_filter_blend_debug["ttt_two_replay_mode"] = gr_mode
                token_filter_blend_debug["ttt_two_replay_applied"] = False
                tri_modes = {"tri_replay", "three_replay", "pos_neu_neg_replay", "pos_neg_neu_replay"}
                if gr_mode not in {"two_replay", "separate_replay", "pos_neg_replay", *tri_modes}:
                    return candidate_w0, candidate_w1, candidate_w2
                if not bool(gradient_reversal_debug.get("ttt_gradient_reversal_applied", False)):
                    token_filter_blend_debug["ttt_two_replay_skip"] = "gradient_reversal_inactive"
                    return candidate_w0, candidate_w1, candidate_w2
                if filter_idx is not None:
                    token_filter_blend_debug["ttt_two_replay_skip"] = "token_filter_active"
                    return candidate_w0, candidate_w1, candidate_w2

                active = tuple(
                    int(i)
                    for i in gradient_reversal_debug.get("ttt_gradient_reversal_active_branches", [])
                    if 0 <= int(i) <= 2
                )
                if len(active) == 0:
                    token_filter_blend_debug["ttt_two_replay_skip"] = "no_active_branch"
                    return candidate_w0, candidate_w1, candidate_w2

                risk = gradient_reversal_risk_flat
                if risk is None:
                    p = prior_flat.detach().float().reshape(-1)
                    p_min = p.min()
                    p_max = p.max()
                    risk = ((p_max - p) / (p_max - p_min).clamp_min(1e-6)).clamp(0.0, 1.0)
                else:
                    risk = risk.detach().float().reshape(-1).clamp(0.0, 1.0)
                    if risk.numel() != int(l):
                        aligned = torch.zeros(int(l), dtype=torch.float32, device=risk.device)
                        n = min(int(risk.numel()), int(l))
                        if n > 0:
                            aligned[:n] = risk[:n]
                        risk = aligned

                if gr_mode in tri_modes:
                    p = prior_flat.detach().float().reshape(-1)
                    if p.numel() != int(l):
                        p_aligned = torch.ones(int(l), dtype=torch.float32, device=p.device)
                        n = min(int(p.numel()), int(l))
                        if n > 0:
                            p_aligned[:n] = p[:n]
                        p = p_aligned
                    pos_frac = min(max(float(self.tri_replay_positive_frac), 0.0), 1.0)
                    neg_frac_cfg = float(self.tri_replay_negative_frac)
                    if neg_frac_cfg <= 0.0:
                        neg_frac_cfg = float(self.gradient_reversal_negative_frac)
                    neg_frac = min(max(float(neg_frac_cfg), 0.0), 1.0)
                    if pos_frac <= 0.0:
                        pos_mask = torch.zeros_like(risk, dtype=torch.bool)
                        pos_thr = torch.tensor(0.0, dtype=risk.dtype, device=risk.device)
                    elif pos_frac >= 1.0:
                        pos_mask = torch.ones_like(risk, dtype=torch.bool)
                        pos_thr = torch.tensor(1.0, dtype=risk.dtype, device=risk.device)
                    else:
                        pos_thr = torch.quantile(risk, pos_frac)
                        pos_mask = risk <= pos_thr
                    if neg_frac <= 0.0:
                        neg_mask = torch.zeros_like(risk, dtype=torch.bool)
                        neg_thr = torch.tensor(1.0, dtype=risk.dtype, device=risk.device)
                    elif neg_frac >= 1.0:
                        neg_mask = torch.ones_like(risk, dtype=torch.bool)
                        neg_thr = torch.tensor(0.0, dtype=risk.dtype, device=risk.device)
                    else:
                        neg_thr = torch.quantile(risk, 1.0 - neg_frac)
                        neg_mask = risk >= neg_thr
                    pos_mask = pos_mask & (~neg_mask)
                    neu_mask = ~(pos_mask | neg_mask)

                    def replay_group(group_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                        group_prior = group_vec.to(device=device, dtype=token_prior0.dtype).view(1, -1, 1)
                        gp0 = group_prior if 0 in active else token_prior0_pre_gr
                        gp1 = group_prior if 1 in active else token_prior1_pre_gr
                        gp2 = group_prior if 2 in active else token_prior2_pre_gr
                        return replay_with_feature_select(
                            k_native_full, v_native_full, k_gate_full, v_gate_full,
                            lr0, lr1, lr2,
                            group_prior,
                            gp0,
                            gp1,
                            gp2,
                            replay_order_full,
                            momentum,
                        )

                    pos_vec = (p * pos_mask.float()).clamp_min(0.0)
                    neu_vec = (p * neu_mask.float()).clamp_min(0.0)
                    neg_vec = (risk * neg_mask.float()).clamp_min(0.0)
                    pos_w0, pos_w1, pos_w2 = replay_group(pos_vec)
                    neu_w0, neu_w1, neu_w2 = replay_group(neu_vec)
                    neg_w0, neg_w1, neg_w2 = replay_group(neg_vec)
                branch_gammas = {
                    str(k): float(v)
                    for k, v in gradient_reversal_debug.get(
                        "ttt_gradient_reversal_active_branch_gammas", {}
                    ).items()
                }
                neu_lambda = float(self.tri_replay_neutral_lambda)

                def maybe_route_heads(
                    branch_idx: int,
                    name: str,
                    base: torch.Tensor,
                    controlled: torch.Tensor,
                ) -> torch.Tensor:
                    routed_heads = self._gradient_reversal_head_indices_for_layer(
                        layer_idx=int(layer_idx),
                        head_count=int(base.shape[0]) if base.ndim > 0 else 0,
                    )
                    if routed_heads is None:
                        return controlled
                    if branch_idx not in active:
                        return base
                    if len(routed_heads) == 0:
                        token_filter_blend_debug[f"ttt_head_routed_{name}_skip"] = "empty"
                        return base
                    idx = torch.tensor(routed_heads, dtype=torch.long, device=base.device)
                    out = base.clone()
                    out.index_copy_(0, idx, controlled.index_select(0, idx))
                    token_filter_blend_debug[f"ttt_head_routed_{name}_applied"] = True
                    token_filter_blend_debug[f"ttt_head_routed_{name}_layer"] = int(layer_idx)
                    token_filter_blend_debug[f"ttt_head_routed_{name}_heads"] = [int(x) for x in routed_heads]
                    token_filter_blend_debug[f"ttt_head_routed_{name}_head_count"] = int(base.shape[0])
                    return out

                def apply_tri_branch(
                    branch_idx: int,
                    name: str,
                    full_pos: torch.Tensor,
                    pos: torch.Tensor,
                    neu: torch.Tensor,
                    neg: torch.Tensor,
                    old: torch.Tensor,
                ) -> torch.Tensor:
                    if branch_idx not in active:
                        return full_pos
                    gamma = float(branch_gammas.get(str(branch_idx), self.gradient_reversal_gamma))
                    raw = (
                        old.float()
                        + (pos.float() - old.float())
                        + neu_lambda * (neu.float() - old.float())
                        - gamma * (neg.float() - old.float())
                    )
                    token_filter_blend_debug[f"ttt_tri_replay_{name}_gamma"] = float(gamma)
                    token_filter_blend_debug[f"ttt_tri_replay_{name}_pos_delta_norm_mean"] = float(
                        (pos.float() - old.float()).detach().norm(dim=1).mean().item()
                    )
                    token_filter_blend_debug[f"ttt_tri_replay_{name}_neu_delta_norm_mean"] = float(
                        (neu.float() - old.float()).detach().norm(dim=1).mean().item()
                    )
                    token_filter_blend_debug[f"ttt_tri_replay_{name}_neg_delta_norm_mean"] = float(
                        (neg.float() - old.float()).detach().norm(dim=1).mean().item()
                    )
                    controlled = renorm_like(full_pos, raw)
                    route_base = full_pos
                    base_gamma = float(
                        self.gradient_reversal_branch_gammas.get(
                            int(branch_idx),
                            self.gradient_reversal_gamma,
                        )
                    )
                    if self.gradient_reversal_head_routes and base_gamma > 0.0:
                        base_raw = (
                            old.float()
                            + (pos.float() - old.float())
                            + neu_lambda * (neu.float() - old.float())
                            - base_gamma * (neg.float() - old.float())
                        )
                        route_base = renorm_like(full_pos, base_raw)
                        token_filter_blend_debug[f"ttt_head_routed_{name}_base_gamma"] = float(base_gamma)
                    return maybe_route_heads(branch_idx, name, route_base, controlled)

                out_w0 = apply_tri_branch(0, "w0", candidate_w0, pos_w0, neu_w0, neg_w0, w0_old)
                out_w1 = apply_tri_branch(1, "w1", candidate_w1, pos_w1, neu_w1, neg_w1, w1_old)
                out_w2 = apply_tri_branch(2, "w2", candidate_w2, pos_w2, neu_w2, neg_w2, w2_old)
                risk_cpu = risk.detach().float().cpu()
                token_filter_blend_debug["ttt_two_replay_applied"] = True
                token_filter_blend_debug["ttt_tri_replay_applied"] = True
                token_filter_blend_debug["ttt_tri_replay_active_branches"] = list(active)
                token_filter_blend_debug["ttt_tri_replay_positive_frac"] = float(pos_frac)
                token_filter_blend_debug["ttt_tri_replay_negative_frac"] = float(neg_frac)
                token_filter_blend_debug["ttt_tri_replay_neutral_lambda"] = float(neu_lambda)
                token_filter_blend_debug["ttt_tri_replay_pos_threshold"] = float(pos_thr.item())
                token_filter_blend_debug["ttt_tri_replay_neg_threshold"] = float(neg_thr.item())
                token_filter_blend_debug["ttt_tri_replay_pos_mass"] = float(pos_mask.float().mean().item())
                token_filter_blend_debug["ttt_tri_replay_neu_mass"] = float(neu_mask.float().mean().item())
                token_filter_blend_debug["ttt_tri_replay_neg_mass"] = float(neg_mask.float().mean().item())
                token_filter_blend_debug["ttt_two_replay_active_branches"] = list(active)
                token_filter_blend_debug["ttt_two_replay_risk_mean"] = float(risk_cpu.mean().item())
                token_filter_blend_debug["ttt_two_replay_risk_p90"] = float(torch.quantile(risk_cpu, 0.90).item())
                return out_w0, out_w1, out_w2

                risk_prior = risk.to(device=device, dtype=token_prior0.dtype).view(1, -1, 1)
                neg_prior0 = risk_prior if 0 in active else token_prior0_pre_gr
                neg_prior1 = risk_prior if 1 in active else token_prior1_pre_gr
                neg_prior2 = risk_prior if 2 in active else token_prior2_pre_gr
                neg_w0, neg_w1, neg_w2 = replay_with_feature_select(
                    k_native_full, v_native_full, k_gate_full, v_gate_full,
                    lr0, lr1, lr2,
                    risk_prior,
                    neg_prior0,
                    neg_prior1,
                    neg_prior2,
                    replay_order_full,
                    momentum,
                )

                branch_gammas = {
                    str(k): float(v)
                    for k, v in gradient_reversal_debug.get(
                        "ttt_gradient_reversal_active_branch_gammas", {}
                    ).items()
                }

                def apply_branch(
                    branch_idx: int,
                    name: str,
                    pos: torch.Tensor,
                    neg: torch.Tensor,
                    old: torch.Tensor,
                ) -> torch.Tensor:
                    if branch_idx not in active:
                        return pos
                    gamma = float(branch_gammas.get(str(branch_idx), self.gradient_reversal_gamma))
                    candidate = pos.float() - gamma * (neg.float() - old.float())
                    delta_norm = (neg.float() - old.float()).detach().norm(dim=1)
                    token_filter_blend_debug[f"ttt_two_replay_{name}_gamma"] = float(gamma)
                    token_filter_blend_debug[f"ttt_two_replay_{name}_neg_delta_norm_mean"] = float(
                        delta_norm.mean().item()
                    ) if delta_norm.numel() else 0.0
                    return renorm_like(pos, candidate)

                out_w0 = apply_branch(0, "w0", candidate_w0, neg_w0, w0_old)
                out_w1 = apply_branch(1, "w1", candidate_w1, neg_w1, w1_old)
                out_w2 = apply_branch(2, "w2", candidate_w2, neg_w2, w2_old)
                risk_cpu = risk.detach().float().cpu()
                token_filter_blend_debug["ttt_two_replay_applied"] = True
                token_filter_blend_debug["ttt_two_replay_active_branches"] = list(active)
                token_filter_blend_debug["ttt_two_replay_risk_mean"] = float(risk_cpu.mean().item())
                token_filter_blend_debug["ttt_two_replay_risk_p90"] = float(torch.quantile(risk_cpu, 0.90).item())
                return out_w0, out_w1, out_w2

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

            w0_new, w1_new, w2_new = maybe_apply_two_replay_negative(w0_new, w1_new, w2_new)
            w0_new, w1_new, w2_new = maybe_store_gradient_reversal_transient(w0_new, w1_new, w2_new)

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

    def _apply_special_token_policy(
        self,
        prior_flat: torch.Tensor,
        *,
        token_type: Optional[torch.Tensor],
        cache_l: int,
        align_mode: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Optionally tie register/role token write prior to patch-token risk.

        LoGeR's replay layout usually repeats ``[registers, role, patches]``
        per frame.  If special tokens are left at 1.0 while patch tokens are
        suppressed, dynamic context can still enter the TTT fast weights
        through those special tokens.  The policy is disabled by default and
        only mutates exact/full token layouts, not patch-only diagnostic
        replays.
        """
        mode = str(self.special_token_policy or "none").strip().lower()
        debug: Dict[str, Any] = {
            "ttt_special_token_policy": mode,
            "ttt_special_token_policy_applied": False,
            "ttt_special_token_floor": float(self.special_token_floor),
            "ttt_special_token_ceiling": float(self.special_token_ceiling),
        }
        if mode in {"", "none", "off"} or prior_flat.numel() == 0 or token_type is None:
            return prior_flat, debug
        if str(align_mode) == "patch_token_type":
            debug["ttt_special_token_policy_skipped"] = "patch_only_alignment"
            return prior_flat, debug

        tt = token_type.detach().cpu().long().reshape(-1)
        if tt.numel() < int(cache_l):
            debug["ttt_special_token_policy_skipped"] = "token_type_short"
            return prior_flat, debug
        tt = tt[: int(cache_l)]
        if tt.numel() != prior_flat.numel():
            debug["ttt_special_token_policy_skipped"] = "length_mismatch"
            return prior_flat, debug

        special_mask = tt != TOKEN_TYPE_PATCH
        patch_mask = tt == TOKEN_TYPE_PATCH
        if not bool(special_mask.any()) or not bool(patch_mask.any()):
            debug["ttt_special_token_policy_skipped"] = "missing_patch_or_special"
            return prior_flat, debug

        out = prior_flat.detach().float().clone()
        before = out[special_mask].clone()
        patch_vals = out[patch_mask].float()
        lo = max(0.0, min(float(self.special_token_floor), float(self.special_token_ceiling)))
        hi = min(2.0, max(float(self.special_token_ceiling), lo))

        def _stat(vals: torch.Tensor, stat_mode: str) -> float:
            vals = vals.detach().float().reshape(-1)
            if vals.numel() == 0:
                return 1.0
            if stat_mode in {"mean", "patch_mean", "global_mean"}:
                v = vals.mean()
            elif stat_mode in {"q10", "patch_q10", "global_q10"}:
                v = torch.quantile(vals, 0.10)
            elif stat_mode in {"q25", "patch_q25", "global_q25"}:
                v = torch.quantile(vals, 0.25)
            elif stat_mode in {"q50", "median", "patch_median", "global_median"}:
                v = torch.quantile(vals, 0.50)
            elif stat_mode in {"min", "patch_min", "global_min"}:
                v = vals.min()
            else:
                v = vals.mean()
            return float(v.clamp(lo, hi).item())

        frame_modes = {
            "frame_mean", "per_frame_mean",
            "frame_q10", "per_frame_q10",
            "frame_q25", "per_frame_q25",
            "frame_min", "per_frame_min",
        }
        if mode in frame_modes:
            stat_mode = mode.replace("per_", "").replace("frame_", "")
            n = int(tt.numel())
            i = 0
            frames = 0
            while i < n:
                s0 = i
                while i < n and int(tt[i].item()) != TOKEN_TYPE_PATCH:
                    i += 1
                p0 = i
                while i < n and int(tt[i].item()) == TOKEN_TYPE_PATCH:
                    i += 1
                if s0 < p0 and p0 < i:
                    val = _stat(out[p0:i], stat_mode)
                    out[s0:p0] = val
                    frames += 1
                elif i == s0:
                    i += 1
            debug["ttt_special_token_policy_frames"] = int(frames)
        else:
            val = _stat(patch_vals, mode)
            out[special_mask] = val
            debug["ttt_special_token_policy_global_value"] = float(val)

        after = out[special_mask].float()
        debug.update({
            "ttt_special_token_policy_applied": True,
            "ttt_special_token_count": int(special_mask.sum().item()),
            "ttt_special_token_patch_count": int(patch_mask.sum().item()),
            "ttt_special_token_patch_mean": float(patch_vals.mean().item()),
            "ttt_special_token_patch_q10": float(torch.quantile(patch_vals, 0.10).item()),
            "ttt_special_token_patch_q25": float(torch.quantile(patch_vals, 0.25).item()),
            "ttt_special_token_mean_before": float(before.float().mean().item()),
            "ttt_special_token_mean_after": float(after.mean().item()),
            "ttt_special_token_min_after": float(after.min().item()),
            "ttt_special_token_max_after": float(after.max().item()),
        })
        return out.to(dtype=prior_flat.dtype), debug

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

    @staticmethod
    def _normalize01_vec(x: torch.Tensor) -> torch.Tensor:
        y = torch.nan_to_num(x.detach().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if y.numel() == 0:
            return y
        lo = y.min()
        hi = y.max()
        denom = (hi - lo).clamp_min(1e-6)
        return ((y - lo) / denom).clamp(0.0, 1.0)

    def _ttt_layer_residual_risk(self, lc: TTTLayerCache, cache_l: int) -> Optional[torch.Tensor]:
        y = getattr(lc, "apply_output_raw", None)
        v = getattr(lc, "v", None)
        if y is None or v is None:
            return None
        if y.shape != v.shape or y.ndim < 2:
            return None
        y_cpu = y.detach().cpu().float()
        v_cpu = v.detach().cpu().float()
        res = (y_cpu - v_cpu).norm(dim=-1) / v_cpu.norm(dim=-1).clamp_min(1e-6)
        if res.ndim == 1:
            per_tok = res
        else:
            per_tok = res.reshape(-1, res.shape[-1]).mean(dim=0)
        out = torch.zeros(int(cache_l), dtype=torch.float32)
        n = min(int(per_tok.numel()), int(cache_l))
        if n <= 0:
            return None
        out[:n] = per_tok[:n].detach().float()
        return self._normalize01_vec(out)

    def _build_gradient_reversal_risk_flat(
        self,
        lc: TTTLayerCache,
        *,
        prior_flat: torch.Tensor,
        risk_tok: Optional[torch.Tensor],
        token_type: Optional[torch.Tensor],
        cache_l: int,
        effective_branch_gammas: Optional[Dict[int, float]] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Build an optional TTT-internal risk map for gradient reversal.

        The default TTGR risk is derived from low write prior.  These alternate
        sources let the TTT replay cache define what is actually harmful, while
        leaving the positive write prior unchanged.
        """
        source = str(self.gradient_reversal_risk_source or "prior").strip().lower()
        debug: Dict[str, Any] = {
            "ttt_gradient_reversal_risk_source": source,
            "ttt_gradient_reversal_risk_source_applied": False,
        }
        gr_mode = str(self.gradient_reversal_mode or "none").strip().lower()
        branch_gamma_map = (
            {int(k): max(float(v), 0.0) for k, v in effective_branch_gammas.items() if 0 <= int(k) <= 2}
            if effective_branch_gammas is not None
            else {
                int(k): max(float(v), 0.0)
                for k, v in self.gradient_reversal_branch_gammas.items()
                if 0 <= int(k) <= 2
            }
        )
        max_gamma = max(branch_gamma_map.values(), default=max(float(self.gradient_reversal_gamma), 0.0))
        if gr_mode in {"", "none", "off"} or max_gamma <= 0.0:
            debug["ttt_gradient_reversal_risk_source_skip"] = "gradient_reversal_off"
            return None, debug
        if source in {"", "prior", "write_prior", "low_prior", "none", "off"}:
            return None, debug

        residual_risk: Optional[torch.Tensor] = None
        if source in {
            "ttt_residual",
            "residual",
            "ttt_self_residual",
            "self_residual",
            "ttt_residual_x_dg",
            "residual_x_dg",
            "ttt_residual_times_dg",
        }:
            residual_risk = self._ttt_layer_residual_risk(lc, cache_l)
            if residual_risk is None:
                debug["ttt_gradient_reversal_risk_source_missing_residual"] = True
                return None, debug

        if source in {"ttt_residual", "residual", "ttt_self_residual", "self_residual"}:
            risk = residual_risk
        elif source in {
            "ttt_w0_conflict",
            "w0_conflict",
            "ttt_update_conflict",
            "update_conflict",
            "ttt_w0_anti",
            "w0_anti",
            "ttt_update_anti",
            "update_anti",
            "ttt_w0_energy",
            "w0_energy",
            "ttt_update_energy",
            "update_energy",
            "ttt_w0_conflict_energy",
            "w0_conflict_energy",
            "ttt_update_conflict_energy",
            "update_conflict_energy",
        }:
            risk, conflict_debug = self._ttt_layer_w0_update_risk(
                lc,
                cache_l=cache_l,
                prior_flat=prior_flat,
                mode=source,
            )
            debug.update(conflict_debug)
            if risk is None:
                debug["ttt_gradient_reversal_risk_source_missing_update_conflict"] = True
                return None, debug
        elif source in {"d_tok", "control", "control_prior", "external_d", "dg", "d_g"}:
            if risk_tok is None:
                debug["ttt_gradient_reversal_risk_source_missing_external"] = True
                return None, debug
            risk, align_debug = self._align_prior_to_replay_tokens(
                risk_tok,
                token_type=token_type,
                cache_l=cache_l,
            )
            debug.update({
                "ttt_gradient_reversal_risk_alignment_mode": align_debug.get("ttt_prior_alignment_mode"),
                "ttt_gradient_reversal_risk_alignment_full_tokens": align_debug.get("ttt_prior_alignment_full_tokens"),
            })
            risk = risk.detach().float().reshape(-1).clamp(0.0, 1.0)
        elif source in {"ttt_residual_x_dg", "residual_x_dg", "ttt_residual_times_dg"}:
            if risk_tok is None:
                debug["ttt_gradient_reversal_risk_source_missing_external"] = True
                return None, debug
            ext, align_debug = self._align_prior_to_replay_tokens(
                risk_tok,
                token_type=token_type,
                cache_l=cache_l,
            )
            debug.update({
                "ttt_gradient_reversal_risk_alignment_mode": align_debug.get("ttt_prior_alignment_mode"),
                "ttt_gradient_reversal_risk_alignment_full_tokens": align_debug.get("ttt_prior_alignment_full_tokens"),
            })
            risk = self._normalize01_vec(residual_risk * ext.detach().float().reshape(-1).clamp(0.0, 1.0))
        else:
            raise ValueError(f"Unsupported TTT gradient reversal risk source: {self.gradient_reversal_risk_source}")

        if risk is None or risk.numel() == 0:
            return None, debug
        risk = risk.detach().float().reshape(-1)
        if risk.numel() != prior_flat.numel():
            out = torch.zeros_like(prior_flat.detach().float().reshape(-1))
            n = min(int(risk.numel()), int(out.numel()))
            out[:n] = risk[:n]
            risk = out
        prior_cpu = prior_flat.detach().float().reshape(-1)
        debug.update({
            "ttt_gradient_reversal_risk_source_applied": True,
            "ttt_gradient_reversal_risk_source_mean": float(risk.mean().item()),
            "ttt_gradient_reversal_risk_source_p90": float(torch.quantile(risk, 0.90).item()),
            "ttt_gradient_reversal_risk_source_corr_prior": self._corr_1d(risk, prior_cpu),
        })
        return risk, debug

    def _ttt_layer_w0_update_risk(
        self,
        lc: TTTLayerCache,
        *,
        cache_l: int,
        prior_flat: torch.Tensor,
        mode: str,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Estimate token risk from the TTT w0 update geometry itself.

        This is a lightweight pre-zeropower diagnostic.  For each token it
        builds the raw w0 contribution direction and compares it with the
        layer's aggregate w0 update direction.  Tokens with poor/negative
        alignment are plausible negative evidence because they fight the
        chunk's own continuity update, not merely because an external cue says
        they look dynamic.
        """
        debug: Dict[str, Any] = {}
        try:
            k = getattr(lc, "k", None)
            v = getattr(lc, "v", None)
            lr0 = getattr(lc, "lr0", None)
            w0 = getattr(lc, "w0_old", None)
            w1 = getattr(lc, "w1_old", None)
            w2 = getattr(lc, "w2_old", None)
            if any(x is None for x in (k, v, lr0, w0, w1, w2)):
                return None, debug
            if k.ndim != 3 or v.ndim != 3 or lr0.ndim != 3:
                return None, debug
            if int(k.shape[1]) <= 0:
                return None, debug

            kf = k.detach().cpu().float()
            vf = v.detach().cpu().float()
            lr = lr0.detach().cpu().float()
            w0f = w0.detach().cpu().float()
            w1f = w1.detach().cpu().float()
            w2f = w2.detach().cpu().float()
            l = min(int(kf.shape[1]), int(cache_l), int(prior_flat.numel()))
            if l <= 0:
                return None, debug
            kf = kf[:, :l, :]
            vf = vf[:, :l, :]
            lr = lr[:, :l, :]
            p = prior_flat.detach().cpu().float().reshape(-1)[:l].view(1, l, 1)
            lr_eff = lr * p

            gate = torch.bmm(kf, w0f)
            hidden_before_mul = torch.bmm(kf, w2f)
            dhidden = torch.bmm(vf, w1f.transpose(1, 2))
            dgate = dhidden * hidden_before_mul
            sigma = torch.sigmoid(gate)
            dgate_before_act = dgate * sigma * (1.0 + gate * (1.0 - sigma))

            aggregate = torch.bmm((kf * lr_eff).transpose(1, 2), dgate_before_act)
            agg_norm = aggregate.flatten(1).norm(dim=1).clamp_min(1e-6)
            token_dot = (torch.bmm(kf, aggregate) * dgate_before_act).sum(dim=-1)
            k_norm = kf.norm(dim=-1)
            d_norm = dgate_before_act.norm(dim=-1)
            denom = k_norm * d_norm * agg_norm.view(-1, 1) + 1e-6
            cos = (token_dot / denom).clamp(-1.0, 1.0)
            energy = (lr_eff.squeeze(-1).abs() * k_norm * d_norm).detach().float()
            energy_risk = self._normalize01_vec(energy.reshape(-1)).view_as(energy)

            mode_text = str(mode or "").strip().lower()
            if mode_text in {"ttt_w0_anti", "w0_anti", "ttt_update_anti", "update_anti"}:
                risk_b_l = (-cos).clamp_min(0.0)
            elif mode_text in {"ttt_w0_energy", "w0_energy", "ttt_update_energy", "update_energy"}:
                risk_b_l = energy_risk
            elif mode_text in {
                "ttt_w0_conflict_energy",
                "w0_conflict_energy",
                "ttt_update_conflict_energy",
                "update_conflict_energy",
            }:
                risk_b_l = ((1.0 - cos) * 0.5).clamp(0.0, 1.0) * energy_risk
            else:
                risk_b_l = ((1.0 - cos) * 0.5).clamp(0.0, 1.0)

            per_tok = risk_b_l.mean(dim=0)
            out = torch.zeros(int(cache_l), dtype=torch.float32)
            out[:l] = per_tok.detach().float()
            cos_flat = cos.detach().float().reshape(-1)
            energy_flat = energy.detach().float().reshape(-1)
            risk_head = risk_b_l.detach().float().mean(dim=1)
            energy_head = energy.detach().float().mean(dim=1)
            cos_head = cos.detach().float().mean(dim=1)
            head_count = int(risk_head.numel())
            top_k = min(5, head_count)
            if top_k > 0:
                top_vals, top_idx = torch.topk(risk_head, k=top_k, largest=True)
                top_energy = energy_head.index_select(0, top_idx)
                top_cos = cos_head.index_select(0, top_idx)
            else:
                top_vals = top_idx = top_energy = top_cos = torch.empty(0)
            debug.update({
                "ttt_update_conflict_mode": mode_text,
                "ttt_update_conflict_cos_mean": float(cos_flat.mean().item()),
                "ttt_update_conflict_cos_p10": float(torch.quantile(cos_flat, 0.10).item()),
                "ttt_update_conflict_cos_p90": float(torch.quantile(cos_flat, 0.90).item()),
                "ttt_update_conflict_negative_cos_mass": float((cos_flat < 0).float().mean().item()),
                "ttt_update_conflict_energy_mean": float(energy_flat.mean().item()),
                "ttt_update_conflict_energy_p90": float(torch.quantile(energy_flat, 0.90).item()),
                "ttt_update_conflict_risk_mean": float(risk_b_l.detach().float().mean().item()),
                "ttt_update_conflict_risk_p90": float(torch.quantile(risk_b_l.detach().float().reshape(-1), 0.90).item()),
                "ttt_update_conflict_head_count": head_count,
                "ttt_update_conflict_risk_head_mean": [float(x) for x in risk_head.tolist()],
                "ttt_update_conflict_energy_head_mean": [float(x) for x in energy_head.tolist()],
                "ttt_update_conflict_cos_head_mean": [float(x) for x in cos_head.tolist()],
                "ttt_update_conflict_top_head_indices_by_risk": [int(x) for x in top_idx.tolist()],
                "ttt_update_conflict_top_head_risk_mean": [float(x) for x in top_vals.tolist()],
                "ttt_update_conflict_top_head_energy_mean": [float(x) for x in top_energy.tolist()],
                "ttt_update_conflict_top_head_cos_mean": [float(x) for x in top_cos.tolist()],
            })
            return self._normalize01_vec(out), debug
        except RuntimeError as exc:
            debug["ttt_update_conflict_error"] = str(exc)
            return None, debug

    def _summarize_ttt_self_cues(self, debug_info: Dict[str, Any], n_layers: int) -> None:
        """Add a compact run-level summary for TTT-internal cue diagnostics."""
        layers: List[int] = []
        energy_mean: List[float] = []
        energy_p90: List[float] = []
        risk_mean: List[float] = []
        risk_p90: List[float] = []
        cos_mean: List[float] = []
        neg_cos_mass: List[float] = []
        top_head_by_layer: List[int] = []
        top_head_risk_by_layer: List[float] = []
        for li in range(int(n_layers)):
            layer_debug = debug_info.get(f"layer_{li}")
            if not isinstance(layer_debug, dict):
                continue
            if "ttt_update_conflict_energy_mean" not in layer_debug:
                continue
            layers.append(int(li))
            energy_mean.append(float(layer_debug.get("ttt_update_conflict_energy_mean", 0.0)))
            energy_p90.append(float(layer_debug.get("ttt_update_conflict_energy_p90", 0.0)))
            risk_mean.append(float(layer_debug.get("ttt_update_conflict_risk_mean", 0.0)))
            risk_p90.append(float(layer_debug.get("ttt_update_conflict_risk_p90", 0.0)))
            cos_mean.append(float(layer_debug.get("ttt_update_conflict_cos_mean", 0.0)))
            neg_cos_mass.append(float(layer_debug.get("ttt_update_conflict_negative_cos_mass", 0.0)))
            top_heads = layer_debug.get("ttt_update_conflict_top_head_indices_by_risk") or []
            top_risks = layer_debug.get("ttt_update_conflict_top_head_risk_mean") or []
            top_head_by_layer.append(int(top_heads[0]) if len(top_heads) > 0 else -1)
            top_head_risk_by_layer.append(float(top_risks[0]) if len(top_risks) > 0 else 0.0)
        if not layers:
            return
        best_layer = layers[int(torch.tensor(risk_mean).argmax().item())]
        debug_info.update({
            "ttt_self_cue_update_conflict_present": True,
            "ttt_self_cue_update_conflict_layers": layers,
            "ttt_self_cue_update_conflict_energy_mean_by_layer": energy_mean,
            "ttt_self_cue_update_conflict_energy_p90_by_layer": energy_p90,
            "ttt_self_cue_update_conflict_risk_mean_by_layer": risk_mean,
            "ttt_self_cue_update_conflict_risk_p90_by_layer": risk_p90,
            "ttt_self_cue_update_conflict_cos_mean_by_layer": cos_mean,
            "ttt_self_cue_update_conflict_negative_cos_mass_by_layer": neg_cos_mass,
            "ttt_self_cue_update_conflict_top_head_by_layer": top_head_by_layer,
            "ttt_self_cue_update_conflict_top_head_risk_by_layer": top_head_risk_by_layer,
            "ttt_self_cue_update_conflict_peak_layer": int(best_layer),
            "ttt_self_cue_update_conflict_peak_layer_risk_mean": float(max(risk_mean)),
        })

    @staticmethod
    def _corr_1d(a: torch.Tensor, b: torch.Tensor) -> float:
        aa = a.detach().float().reshape(-1)
        bb = b.detach().float().reshape(-1)
        n = min(int(aa.numel()), int(bb.numel()))
        if n < 2:
            return 0.0
        aa = aa[:n]
        bb = bb[:n]
        aa = aa - aa.mean()
        bb = bb - bb.mean()
        den = aa.norm() * bb.norm()
        if float(den.item()) <= 1e-8:
            return 0.0
        return float((aa @ bb / den).item())

    def _apply_gradient_reversal_prior(
        self,
        prior_flat: torch.Tensor,
        token_prior0: torch.Tensor,
        token_prior1: torch.Tensor,
        token_prior2: torch.Tensor,
        *,
        branch_enabled: Tuple[bool, bool, bool],
        device: str,
        risk_flat: Optional[torch.Tensor] = None,
        effective_branch_gammas: Optional[Dict[int, float]] = None,
        layer_idx: int = -1,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """Convert low-prior tokens into small negative replay evidence.

        This hook runs after optional eta mean-preservation.  That ordering is
        intentional: eta normalization still sees the normal write prior, while
        selected branches can receive a signed multiplier in the actual replay.
        A high-risk token therefore changes update direction instead of merely
        reducing the positive learning-rate mass.
        """
        mode = str(self.gradient_reversal_mode or "none").strip().lower()
        gamma = max(float(self.gradient_reversal_gamma), 0.0)
        branch_mask = tuple(self.gradient_reversal_branch_mask)
        branch_gamma_map = (
            {int(k): max(float(v), 0.0) for k, v in effective_branch_gammas.items() if 0 <= int(k) <= 2}
            if effective_branch_gammas is not None
            else {
                int(k): max(float(v), 0.0)
                for k, v in self.gradient_reversal_branch_gammas.items()
                if 0 <= int(k) <= 2
            }
        )
        if branch_gamma_map:
            branch_gammas = branch_gamma_map
        else:
            branch_gammas = {
                int(i): gamma
                for i in branch_mask
                if 0 <= int(i) <= 2
            }
        max_gamma = max(branch_gammas.values(), default=0.0)
        debug: Dict[str, Any] = {
            "ttt_gradient_reversal_mode": mode,
            "ttt_gradient_reversal_gamma": gamma,
            "ttt_gradient_reversal_branch_mask": list(branch_mask),
            "ttt_gradient_reversal_branch_gammas": {
                str(int(k)): float(v)
                for k, v in sorted(branch_gamma_map.items())
            },
            "ttt_gradient_reversal_layer_idx": int(layer_idx),
            "ttt_gradient_reversal_layer_gammas": {
                str(int(k)): float(v)
                for k, v in sorted(self.gradient_reversal_layer_gammas.items())
            },
            "ttt_gradient_reversal_layer_routed": bool(self.gradient_reversal_layer_gammas),
            "ttt_gradient_reversal_head_routes": {
                str(int(k)): [int(x) for x in v]
                for k, v in sorted(self.gradient_reversal_head_routes.items())
            },
            "ttt_gradient_reversal_negative_frac": float(self.gradient_reversal_negative_frac),
            "ttt_gradient_reversal_applied": False,
        }
        if mode in {"", "none", "off"} or max_gamma <= 0.0 or prior_flat.numel() == 0:
            return (token_prior0, token_prior1, token_prior2), debug
        active = tuple(
            int(i)
            for i, g in sorted(branch_gammas.items())
            if g > 0.0 and 0 <= int(i) <= 2 and branch_enabled[int(i)]
        )
        if len(active) == 0:
            debug["ttt_gradient_reversal_no_active_branch"] = True
            return (token_prior0, token_prior1, token_prior2), debug

        p = prior_flat.detach().float().reshape(-1)
        p_min = p.min()
        p_max = p.max()
        denom = (p_max - p_min).clamp_min(1e-6)
        if risk_flat is not None:
            r = risk_flat.detach().float().reshape(-1)
            if r.numel() != p.numel():
                r_aligned = torch.zeros_like(p)
                n = min(int(r.numel()), int(p.numel()))
                if n > 0:
                    r_aligned[:n] = r[:n]
                r = r_aligned
            risk = r.clamp(0.0, 1.0)
            risk_source_effective = str(self.gradient_reversal_risk_source or "prior").strip().lower()
        else:
            risk = ((p_max - p) / denom).clamp(0.0, 1.0)
            risk_source_effective = "prior_low"

        neg_mask: Optional[torch.Tensor] = None
        if mode in {"negative_tail", "tail", "bottom_frac", "tail_low_prior"}:
            neg_frac = max(min(float(self.gradient_reversal_negative_frac), 1.0), 0.0)
            if neg_frac <= 0.0:
                neg_mask = risk > 0.5
                threshold = torch.tensor(0.5, device=risk.device, dtype=risk.dtype)
            elif neg_frac >= 1.0:
                neg_mask = torch.ones_like(risk, dtype=torch.bool)
                threshold = torch.tensor(0.0, device=risk.device, dtype=risk.dtype)
            else:
                threshold = torch.quantile(risk, 1.0 - neg_frac)
                neg_mask = risk >= threshold
            debug["ttt_gradient_reversal_tail_threshold"] = float(threshold.item())
        elif mode in {
            "two_replay",
            "separate_replay",
            "pos_neg_replay",
            "tri_replay",
            "three_replay",
            "pos_neu_neg_replay",
            "pos_neg_neu_replay",
        }:
            neg_mask = None
        elif mode not in {
            "low_prior",
            "dynamic",
            "risk",
            "signed_low_prior",
            "hard",
            "hard_low_prior",
            "hard_dynamic",
        }:
            raise ValueError(f"Unsupported TTT gradient reversal mode: {self.gradient_reversal_mode}")

        priors = [token_prior0, token_prior1, token_prior2]
        signed_by_branch: Dict[int, torch.Tensor] = {}
        for branch_idx in active:
            branch_gamma = max(float(branch_gammas[int(branch_idx)]), 0.0)
            if mode in {
                "two_replay",
                "separate_replay",
                "pos_neg_replay",
                "tri_replay",
                "three_replay",
                "pos_neu_neg_replay",
                "pos_neg_neu_replay",
            }:
                signed = p
            elif mode in {"low_prior", "dynamic", "risk", "signed_low_prior"}:
                signed = p * (1.0 - risk) - branch_gamma * risk
            elif mode in {"negative_tail", "tail", "bottom_frac", "tail_low_prior"}:
                assert neg_mask is not None
                signed = torch.where(
                    neg_mask,
                    -torch.full_like(p, branch_gamma),
                    p,
                )
            else:
                signed = torch.where(
                    risk > 0.5,
                    -torch.full_like(p, branch_gamma),
                    p,
                )
            signed_by_branch[int(branch_idx)] = signed
            signed_token = signed.to(device=device, dtype=token_prior0.dtype).view(1, -1, 1)
            priors[int(branch_idx)] = signed_token

        signed_stack = torch.stack([signed_by_branch[int(i)].detach().float() for i in active], dim=0)
        signed_cpu = signed_stack.reshape(-1)
        risk_cpu = risk.detach().float()
        debug.update({
            "ttt_gradient_reversal_applied": True,
            "ttt_gradient_reversal_active_branches": list(active),
            "ttt_gradient_reversal_active_branch_gammas": {
                str(int(i)): float(branch_gammas[int(i)])
                for i in active
            },
            "ttt_gradient_reversal_prior_min": float(p_min.item()),
            "ttt_gradient_reversal_prior_max": float(p_max.item()),
            "ttt_gradient_reversal_risk_source_effective": risk_source_effective,
            "ttt_gradient_reversal_risk_mean": float(risk_cpu.mean().item()),
            "ttt_gradient_reversal_risk_p90": float(torch.quantile(risk_cpu, 0.90).item()),
            "ttt_gradient_reversal_signed_mean": float(signed_cpu.mean().item()),
            "ttt_gradient_reversal_signed_min": float(signed_cpu.min().item()),
            "ttt_gradient_reversal_signed_p10": float(torch.quantile(signed_cpu, 0.10).item()),
            "ttt_gradient_reversal_signed_p50": float(torch.quantile(signed_cpu, 0.50).item()),
            "ttt_gradient_reversal_signed_p90": float(torch.quantile(signed_cpu, 0.90).item()),
            "ttt_gradient_reversal_negative_mass": float((signed_cpu < 0).float().mean().item()),
            "ttt_gradient_reversal_branch_signed_mean": {
                str(int(i)): float(signed_by_branch[int(i)].detach().float().mean().item())
                for i in active
            },
            "ttt_gradient_reversal_branch_negative_mass": {
                str(int(i)): float((signed_by_branch[int(i)].detach().float() < 0).float().mean().item())
                for i in active
            },
        })
        return (priors[0], priors[1], priors[2]), debug

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
        if risk_source in {"ttt_residual", "residual", "ttt_self_residual", "self_residual"}:
            risk_source = "ttt_residual"
        if risk_source in {"ttt_residual_x_dg", "residual_x_dg", "ttt_residual_times_dg"}:
            risk_source = "ttt_residual_x_dg"
        if risk_source in {
            "ttt_w0_conflict",
            "w0_conflict",
            "ttt_update_conflict",
            "update_conflict",
            "ttt_w0_anti",
            "w0_anti",
            "ttt_update_anti",
            "update_anti",
            "ttt_w0_energy",
            "w0_energy",
            "ttt_update_energy",
            "update_energy",
            "ttt_w0_conflict_energy",
            "w0_conflict_energy",
            "ttt_update_conflict_energy",
            "update_conflict_energy",
        }:
            risk_source = {
                "ttt_w0_conflict": "update_conflict",
                "w0_conflict": "update_conflict",
                "ttt_update_conflict": "update_conflict",
                "ttt_w0_anti": "update_anti",
                "w0_anti": "update_anti",
                "ttt_update_anti": "update_anti",
                "ttt_w0_energy": "update_energy",
                "w0_energy": "update_energy",
                "ttt_update_energy": "update_energy",
                "ttt_w0_conflict_energy": "update_conflict_energy",
                "w0_conflict_energy": "update_conflict_energy",
                "ttt_update_conflict_energy": "update_conflict_energy",
            }.get(risk_source, risk_source)
        if risk_source not in {
            "d_tok",
            "write_prior",
            "ttt_residual",
            "ttt_residual_x_dg",
            "update_conflict",
            "update_anti",
            "update_energy",
            "update_conflict_energy",
        }:
            raise ValueError(f"Unsupported TTT commit filter risk source: {self.commit_filter_risk_source}")

        source_tok = risk_tok if risk_source == "d_tok" else A_tok
        if risk_source in {"d_tok", "write_prior"} and source_tok is None:
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
            layer_debug = debug_info.get(f"layer_{li}")
            cache_l = int(lc.k.shape[1]) if lc.k is not None and lc.k.ndim >= 2 else 0
            if cache_l <= 0:
                continue
            align_debug: Dict[str, Any] = {}
            risk_stat_override: Optional[float] = None
            if risk_source in {"ttt_residual", "ttt_residual_x_dg"}:
                residual = self._ttt_layer_residual_risk(lc, cache_l)
                if residual is None:
                    invalid_scope = True
                    layer_debug = debug_info.get(f"layer_{li}")
                    if isinstance(layer_debug, dict):
                        layer_debug["ttt_write_commit_filter_missing_residual"] = True
                    continue
                risk_flat = residual.detach().float().reshape(-1).clamp(0.0, 1.0)
                if risk_source == "ttt_residual_x_dg":
                    if risk_tok is None:
                        debug_info.update({
                            "ttt_write_commit_filter_applied": False,
                            "ttt_write_commit_filter_missing_external": True,
                            "ttt_write_commit_filter_risk_source": risk_source,
                        })
                        return
                    ext_flat, align_debug = self._align_prior_to_replay_tokens(
                        risk_tok,
                        token_type=token_type,
                        cache_l=cache_l,
                    )
                    risk_flat = self._normalize01_vec(
                        risk_flat * ext_flat.detach().float().reshape(-1).clamp(0.0, 1.0)
                    )
            elif risk_source in {
                "update_conflict",
                "update_anti",
                "update_energy",
                "update_conflict_energy",
            }:
                if scope == "all" and isinstance(layer_debug, dict):
                    cached_key = None
                    if stat_name == "mean":
                        cached_key = "ttt_gradient_reversal_risk_source_mean"
                    elif stat_name == "q90":
                        cached_key = "ttt_gradient_reversal_risk_source_p90"
                    if cached_key and cached_key in layer_debug:
                        risk_stat_override = float(layer_debug[cached_key])
                        align_debug["ttt_write_commit_filter_reused_gradient_risk_stat"] = True
                        align_debug["ttt_write_commit_filter_reused_gradient_risk_key"] = cached_key
                if risk_stat_override is None:
                    if A_tok is None:
                        prior_flat = torch.ones(int(cache_l), dtype=torch.float32)
                    else:
                        prior_flat, align_debug = self._align_prior_to_replay_tokens(
                            A_tok,
                            token_type=token_type,
                            cache_l=cache_l,
                        )
                        prior_flat = prior_flat.detach().float().reshape(-1).clamp(0.0, 1.0)
                    risk_vec, conflict_debug = self._ttt_layer_w0_update_risk(
                        lc,
                        cache_l=cache_l,
                        prior_flat=prior_flat,
                        mode=risk_source,
                    )
                    if risk_vec is None:
                        invalid_scope = True
                        if isinstance(layer_debug, dict):
                            layer_debug["ttt_write_commit_filter_missing_update_conflict"] = True
                        continue
                    risk_flat = risk_vec.detach().float().reshape(-1).clamp(0.0, 1.0)
                    align_debug.update(conflict_debug)
                else:
                    risk_flat = torch.empty(0, dtype=torch.float32)
            else:
                risk_flat, align_debug = self._align_prior_to_replay_tokens(
                    source_tok,
                    token_type=token_type,
                    cache_l=cache_l,
                )
                risk_flat = risk_flat.detach().float().reshape(-1).clamp(0.0, 1.0)
                if risk_source == "write_prior":
                    risk_flat = (1.0 - risk_flat).clamp(0.0, 1.0)
            if risk_stat_override is not None:
                scope_debug = {
                    "ttt_write_commit_filter_scope_valid": True,
                    "ttt_write_commit_filter_scope_tokens": int(cache_l),
                    "ttt_write_commit_filter_scope_mass": 1.0,
                    "ttt_write_commit_filter_overlap_frames": int(overlap_frames),
                }
                risk_stat = risk_stat_override
            else:
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
        transient_delta: Optional[Dict[str, Any]],
    ) -> bool:
        if not isinstance(transient_delta, dict):
            return False
        for branch_name in ("w0", "w1", "w2"):
            values = transient_delta.get(branch_name)
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
        prev_transient_delta: Optional[Dict[str, Any]],
        w0_new: List[Optional[torch.Tensor]],
        w1_new: List[Optional[torch.Tensor]],
        w2_new: List[Optional[torch.Tensor]],
        debug_info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Remove one-hop dynamic residuals before they become long-term TTT memory."""
        scale = float(self.transient_delta_subtract_scale)
        branch_mask = tuple(self.transient_delta_branch_mask)
        prev_present = self._has_transient_delta(prev_transient_delta)
        prev_ttl = int(prev_transient_delta.get("_ttl_remaining", 1)) if isinstance(prev_transient_delta, dict) else 0
        debug_info["ttt_transient_delta_prev_present"] = prev_present
        debug_info["ttt_transient_delta_prev_ttl_in"] = int(prev_ttl)
        debug_info["ttt_transient_delta_prev_subtract_scale"] = scale
        debug_info["ttt_transient_delta_prev_branch_mask"] = list(branch_mask)
        if scale <= 0.0 or not prev_present:
            debug_info["ttt_transient_delta_prev_subtract_applied"] = False
            debug_info["ttt_transient_delta_prev_subtract_tensors"] = 0
            prev_mode = str(prev_transient_delta.get("_mode", "")) if isinstance(prev_transient_delta, dict) else ""
            if prev_present and prev_mode in {"dual_lifetime", "dual_fast_weight", "apply_short_delta", "short_apply_delta"}:
                if prev_ttl > 1:
                    carry = dict(prev_transient_delta)
                    carry["_ttl_remaining"] = int(prev_ttl - 1)
                    debug_info["ttt_transient_delta_prev_carry"] = True
                    debug_info["ttt_transient_delta_prev_ttl_out"] = int(prev_ttl - 1)
                    debug_info["ttt_transient_delta_prev_carry_reason"] = "dual_lifetime_no_subtract"
                    return carry
                debug_info["ttt_transient_delta_prev_carry"] = False
                debug_info["ttt_transient_delta_prev_ttl_out"] = 0
                debug_info["ttt_transient_delta_prev_carry_reason"] = "dual_lifetime_expired"
            return None
        if prev_ttl > 1:
            carry = dict(prev_transient_delta) if isinstance(prev_transient_delta, dict) else None
            if isinstance(carry, dict):
                carry["_ttl_remaining"] = int(prev_ttl - 1)
            debug_info["ttt_transient_delta_prev_subtract_applied"] = False
            debug_info["ttt_transient_delta_prev_subtract_tensors"] = 0
            debug_info["ttt_transient_delta_prev_carry"] = True
            debug_info["ttt_transient_delta_prev_ttl_out"] = int(prev_ttl - 1)
            return carry

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
        debug_info["ttt_transient_delta_prev_carry"] = False
        debug_info["ttt_transient_delta_prev_ttl_out"] = 0
        return None

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

    @staticmethod
    def _parse_branch_gamma_map(text: Optional[str]) -> Dict[int, float]:
        if text is None:
            return {}
        raw = str(text).strip()
        if raw == "" or raw.lower() in {"none", "off"}:
            return {}
        aliases = {
            "w0": 0,
            "b0": 0,
            "branch0": 0,
            "w1": 1,
            "b1": 1,
            "branch1": 1,
            "w2": 2,
            "b2": 2,
            "branch2": 2,
        }
        out: Dict[int, float] = {}
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(
                    f"Invalid branch gamma entry '{part}', expected BRANCH:GAMMA"
                )
            key, value = part.split(":", 1)
            key = key.strip().lower()
            branch = aliases[key] if key in aliases else int(key)
            if branch not in (0, 1, 2):
                raise ValueError(f"gradient reversal branch must be 0, 1, or 2, got {branch}")
            out[branch] = max(float(value), 0.0)
        return dict(sorted(out.items()))

    @staticmethod
    def _parse_layer_gamma_map(text: Optional[str]) -> Dict[int, float]:
        if text is None:
            return {}
        raw = str(text).strip()
        if raw == "" or raw.lower() in {"none", "off"}:
            return {}
        out: Dict[int, float] = {}
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(
                    f"Invalid layer gamma entry '{part}', expected LAYER:GAMMA"
                )
            key, value = part.split(":", 1)
            layer = int(key.strip())
            if layer < 0:
                raise ValueError(f"gradient reversal layer must be non-negative, got {layer}")
            out[layer] = max(float(value), 0.0)
        return dict(sorted(out.items()))

    @staticmethod
    def _parse_layer_head_routes(text: Optional[str]) -> Dict[int, Tuple[int, ...]]:
        if text is None:
            return {}
        raw = str(text).strip()
        if raw == "" or raw.lower() in {"none", "off"}:
            return {}
        out: Dict[int, Tuple[int, ...]] = {}
        for part in raw.replace("|", ";").split(";"):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(
                    f"Invalid layer head route entry '{part}', expected LAYER:HEADS"
                )
            key, value = part.split(":", 1)
            layer = int(key.strip())
            if layer < 0:
                raise ValueError(f"gradient reversal head-route layer must be non-negative, got {layer}")
            head_text = value.strip().lower()
            if head_text in {"", "none", "off"}:
                out[layer] = tuple()
                continue
            if head_text in {"all", "*"}:
                out[layer] = (-1,)
                continue
            heads: List[int] = []
            for head_part in head_text.replace("+", ",").replace("/", ",").split(","):
                head_part = head_part.strip()
                if not head_part:
                    continue
                head = int(head_part)
                if head < 0:
                    raise ValueError(f"gradient reversal head index must be non-negative, got {head}")
                if head not in heads:
                    heads.append(head)
            out[layer] = tuple(heads)
        return dict(sorted(out.items()))

    def _effective_gradient_reversal_branch_gammas(self, layer_idx: int) -> Optional[Dict[int, float]]:
        """Return branch gammas after optional layer routing.

        When no layer map is configured, callers use the historical branch/global
        gamma behavior by receiving ``None``.  When a layer map is configured,
        listed layers get that gamma on the active branch mask; unlisted layers
        fall back to the historical branch/global gamma behavior.  This allows a
        layer map to act as a conflict-cue boost over an all-layer base; setting
        the global gamma to zero still gives layer-only routing.
        """
        if not self.gradient_reversal_layer_gammas:
            return None
        if int(layer_idx) not in self.gradient_reversal_layer_gammas:
            return None
        layer_gamma = max(float(self.gradient_reversal_layer_gammas.get(int(layer_idx), 0.0)), 0.0)
        if self.gradient_reversal_branch_gammas:
            branches = tuple(
                int(k)
                for k in sorted(self.gradient_reversal_branch_gammas.keys())
                if 0 <= int(k) <= 2
            )
        else:
            branches = tuple(int(i) for i in self.gradient_reversal_branch_mask if 0 <= int(i) <= 2)
        return {int(i): layer_gamma for i in branches}

    def _gradient_reversal_head_indices_for_layer(
        self,
        *,
        layer_idx: int,
        head_count: int,
    ) -> Optional[List[int]]:
        if not self.gradient_reversal_head_routes:
            return None
        route = self.gradient_reversal_head_routes.get(int(layer_idx))
        if route is None:
            return None
        if int(head_count) <= 0:
            return []
        if any(int(h) < 0 for h in route):
            return list(range(int(head_count)))
        return [int(h) for h in route if 0 <= int(h) < int(head_count)]

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
