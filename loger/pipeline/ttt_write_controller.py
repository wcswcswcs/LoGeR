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

import os
from pathlib import Path
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
        update_native_mix_chunks: Optional[str] = None,
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
        tri_replay_role_mode: str = "fixed",
        gradient_reversal_transient_mode: str = "none",
        gradient_reversal_transient_branch_mask: str = "",
        gradient_reversal_transient_long_scale: float = 0.0,
        gradient_reversal_transient_apply_scale: float = 1.0,
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
        commit_ema_chunks: Optional[str] = None,
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
        scale_state_mode: str = "none",
        scale_state_proxy: str = "pose_step_ema",
        scale_state_carrier: str = "all",
        scale_state_alpha: float = 0.0,
        scale_state_branch_mask: str = "0",
        scale_state_chunks: Optional[str] = None,
        scale_state_sample_tokens: int = 0,
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
        self.update_native_mix_chunks = self._parse_chunk_mask(update_native_mix_chunks)
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
        self.tri_replay_role_mode = str(tri_replay_role_mode or "fixed").strip().lower()
        self.gradient_reversal_transient_mode = str(gradient_reversal_transient_mode or "none").strip().lower()
        self.gradient_reversal_transient_long_scale = float(gradient_reversal_transient_long_scale)
        self.gradient_reversal_transient_apply_scale = float(gradient_reversal_transient_apply_scale)
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
        self.commit_ema_chunks = self._parse_chunk_mask(commit_ema_chunks)
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
        self.scale_state_mode = str(scale_state_mode or "none").strip().lower()
        self.scale_state_proxy = str(scale_state_proxy or "pose_step_ema").strip().lower()
        self.scale_state_carrier = str(scale_state_carrier or "all").strip().lower()
        self.scale_state_alpha = float(scale_state_alpha)
        self.scale_state_branch_mask = self._parse_branch_mask(scale_state_branch_mask)
        self.scale_state_chunks = self._parse_chunk_mask(scale_state_chunks)
        self.scale_state_sample_tokens = max(int(scale_state_sample_tokens), 0)
        self.scale_state_active = False
        self.scale_state_chunk_idx = -1
        self.scale_state_log_ratio = 0.0
        self.scale_state_reason = "not_configured"
        self.scale_state_payload: Dict[str, Any] = {}
        self.v11_projection_action_mode = "none"
        self.v11_projection_action_active = False
        self.v11_projection_chunk_idx = -1
        self.current_chunk_idx = -1
        self.v11_projection_scale_log_ratio = 0.0
        self.v11_projection_action_strength = 1.0
        self.v11_projection_action_deadband = 0.0
        self.v11_projection_action_reason = ""
        self.state_energy_ema_alpha = 0.25
        self.state_energy_gamma_gain_min = 0.25
        self.state_energy_gamma_gain_max = 4.0
        self.state_energy_gamma_min = 1e-4
        self.state_energy_gamma_max = 2e-2
        self.state_energy_role_k = 0.5
        self.state_energy_commit_tau_c = 0.0
        self.state_energy_commit_u_max = 2.0
        self.state_energy_target_ema: Dict[str, float] = {}
        self.tail_state_energy_ema: Dict[str, float] = {}
        self.tail_commit_energy_ema: Dict[str, float] = {}
        self.tail_commit_risk_ema: Dict[str, float] = {}

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
        self._summarize_commit_against_native(write_cache, w0_new, w1_new, w2_new, debug_info)
        transient_delta_out = transient_delta if self._has_transient_delta(transient_delta) else carry_transient_delta
        if self._has_transient_delta(transient_delta):
            transient_delta_out["_ttl_remaining"] = int(self.transient_delta_ttl)
            mode_tag = str(self.gradient_reversal_transient_mode or "none").strip().lower()
            if mode_tag in {"dual_lifetime", "dual_fast_weight", "apply_short_delta", "short_apply_delta"}:
                transient_delta_out["_mode"] = mode_tag
                transient_delta_out["_apply_scale"] = float(self.gradient_reversal_transient_apply_scale)
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
            num_frames=num_frames,
            overlap_frames=overlap_frames,
            layer_idx=layer_idx,
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
            overlap_frames=overlap_frames,
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
                token_filter_blend_debug["ttt_gradient_reversal_transient_apply_scale"] = float(
                    self.gradient_reversal_transient_apply_scale
                )
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

                def quantile_role_masks(
                    pos_frac_in: float,
                    neg_frac_in: float,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    pos_frac_out = min(max(float(pos_frac_in), 0.0), 1.0)
                    neg_frac_out = min(max(float(neg_frac_in), 0.0), 1.0)
                    if pos_frac_out <= 0.0:
                        pos = torch.zeros_like(risk, dtype=torch.bool)
                        pos_thr_out = torch.tensor(0.0, dtype=risk.dtype, device=risk.device)
                    elif pos_frac_out >= 1.0:
                        pos = torch.ones_like(risk, dtype=torch.bool)
                        pos_thr_out = torch.tensor(1.0, dtype=risk.dtype, device=risk.device)
                    else:
                        pos_thr_out = torch.quantile(risk, pos_frac_out)
                        pos = risk <= pos_thr_out
                    if neg_frac_out <= 0.0:
                        neg = torch.zeros_like(risk, dtype=torch.bool)
                        neg_thr_out = torch.tensor(1.0, dtype=risk.dtype, device=risk.device)
                    elif neg_frac_out >= 1.0:
                        neg = torch.ones_like(risk, dtype=torch.bool)
                        neg_thr_out = torch.tensor(0.0, dtype=risk.dtype, device=risk.device)
                    else:
                        neg_thr_out = torch.quantile(risk, 1.0 - neg_frac_out)
                        neg = risk >= neg_thr_out
                    pos = pos & (~neg)
                    neu = ~(pos | neg)
                    return pos, neu, neg, pos_thr_out, neg_thr_out, pos_frac_out, neg_frac_out

                def capped_role_masks(
                    pos: torch.Tensor,
                    neg: torch.Tensor,
                    *,
                    source: str,
                    min_pos: float = 0.20,
                    max_pos: float = 0.60,
                    min_neg: float = 0.03,
                    max_neg: float = 0.25,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    pos_mass_raw = float(pos.float().mean().item()) if pos.numel() else 0.0
                    neg_mass_raw = float(neg.float().mean().item()) if neg.numel() else 0.0
                    if pos_mass_raw <= 0.0 or neg_mass_raw <= 0.0:
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "empty_role"
                        token_filter_blend_debug["ttt_tri_replay_role_fallback_source"] = source
                        return quantile_role_masks(self.tri_replay_positive_frac, self.tri_replay_negative_frac)
                    pos_frac_cap = min(max(pos_mass_raw, min_pos), max_pos)
                    neg_frac_cap = min(max(neg_mass_raw, min_neg), max_neg)
                    if abs(pos_frac_cap - pos_mass_raw) > 1e-6 or abs(neg_frac_cap - neg_mass_raw) > 1e-6:
                        token_filter_blend_debug["ttt_tri_replay_role_mass_capped"] = True
                        token_filter_blend_debug["ttt_tri_replay_role_raw_pos_mass"] = pos_mass_raw
                        token_filter_blend_debug["ttt_tri_replay_role_raw_neg_mass"] = neg_mass_raw
                        return quantile_role_masks(pos_frac_cap, neg_frac_cap)
                    pos = pos & (~neg)
                    neu = ~(pos | neg)
                    if not bool(pos.any()) or not bool(neg.any()):
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "role_overlap_collapse"
                        token_filter_blend_debug["ttt_tri_replay_role_fallback_source"] = source
                        return quantile_role_masks(self.tri_replay_positive_frac, self.tri_replay_negative_frac)
                    pos_thr_out = risk[pos].max()
                    neg_thr_out = risk[neg].min()
                    return pos, neu, neg, pos_thr_out, neg_thr_out, pos_mass_raw, neg_mass_raw

                def uncapped_role_masks(
                    pos: torch.Tensor,
                    neg: torch.Tensor,
                    *,
                    source: str,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    """Return role masks without converting them to fixed top percentages."""
                    pos = pos.bool()
                    neg = neg.bool()
                    if pos.numel() != risk.numel() or neg.numel() != risk.numel():
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "shape_mismatch"
                        token_filter_blend_debug["ttt_tri_replay_role_fallback_source"] = source
                        pos = torch.zeros_like(risk, dtype=torch.bool)
                        neg = torch.zeros_like(risk, dtype=torch.bool)
                    pos = pos & (~neg)
                    if not bool(pos.any()) or not bool(neg.any()):
                        vals = risk.detach().float()
                        mean = vals.mean()
                        std = vals.std(unbiased=False)
                        if float(std.item()) > 1e-8:
                            pos = vals <= (mean - 0.25 * std)
                            neg = vals >= (mean + 0.25 * std)
                    if not bool(pos.any()) and risk.numel() > 0:
                        pos = risk <= risk.min()
                    if not bool(neg.any()) and risk.numel() > 0:
                        neg = risk >= risk.max()
                    pos = pos & (~neg)
                    neu = ~(pos | neg)
                    if bool(pos.any()):
                        pos_thr_out = risk[pos].max()
                    else:
                        pos_thr_out = risk.new_tensor(0.0)
                    if bool(neg.any()):
                        neg_thr_out = risk[neg].min()
                    else:
                        neg_thr_out = risk.new_tensor(1.0)
                    pos_mass = float(pos.float().mean().item()) if pos.numel() else 0.0
                    neg_mass = float(neg.float().mean().item()) if neg.numel() else 0.0
                    token_filter_blend_debug["ttt_tri_replay_role_uncapped"] = True
                    token_filter_blend_debug["ttt_tri_replay_role_source"] = source
                    token_filter_blend_debug["ttt_tri_replay_role_uncapped_pos_mass"] = pos_mass
                    token_filter_blend_debug["ttt_tri_replay_role_uncapped_neg_mass"] = neg_mass
                    return pos, neu, neg, pos_thr_out, neg_thr_out, pos_mass, neg_mass

                def kmeans3_role_masks() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    vals = risk.detach().float()
                    if vals.numel() < 3 or float(vals.max().item() - vals.min().item()) < 1e-8:
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "kmeans_degenerate_risk"
                        return quantile_role_masks(self.tri_replay_positive_frac, self.tri_replay_negative_frac)
                    centers = torch.quantile(vals, torch.tensor([0.2, 0.5, 0.8], device=vals.device, dtype=vals.dtype))
                    labels = torch.zeros_like(vals, dtype=torch.long)
                    for _ in range(12):
                        dist = (vals[:, None] - centers[None, :]).abs()
                        labels = torch.argmin(dist, dim=1)
                        new_centers = centers.clone()
                        for idx in range(3):
                            mask = labels == idx
                            if bool(mask.any()):
                                new_centers[idx] = vals[mask].mean()
                        if torch.allclose(new_centers, centers, atol=1e-6, rtol=0.0):
                            centers = new_centers
                            break
                        centers = new_centers
                    counts = [int((labels == idx).sum().item()) for idx in range(3)]
                    if sum(1 for c in counts if c > 0) < 3:
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "kmeans_cluster_collapse"
                        token_filter_blend_debug["ttt_tri_replay_kmeans_counts"] = counts
                        return quantile_role_masks(self.tri_replay_positive_frac, self.tri_replay_negative_frac)
                    order = torch.argsort(centers)
                    pos = labels == int(order[0].item())
                    neg = labels == int(order[-1].item())
                    token_filter_blend_debug["ttt_tri_replay_kmeans_centers"] = [
                        float(x) for x in centers.detach().cpu().tolist()
                    ]
                    token_filter_blend_debug["ttt_tri_replay_kmeans_counts"] = counts
                    return capped_role_masks(pos, neg, source="kmeans3")

                def otsu3_role_masks() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    vals = risk.detach().float()
                    if vals.numel() < 3 or float(vals.max().item() - vals.min().item()) < 1e-8:
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "otsu_degenerate_risk"
                        return quantile_role_masks(self.tri_replay_positive_frac, self.tri_replay_negative_frac)
                    hist = torch.histc(vals, bins=64, min=0.0, max=1.0).float()
                    prob = hist / hist.sum().clamp_min(1e-6)
                    bins = (torch.arange(64, device=vals.device, dtype=vals.dtype) + 0.5) / 64.0
                    total_mean = (prob * bins).sum()
                    best_score = torch.tensor(-1.0, device=vals.device)
                    best = (20, 44)
                    for i in range(1, 63):
                        w0 = prob[:i].sum()
                        if float(w0.item()) <= 0.0:
                            continue
                        m0 = (prob[:i] * bins[:i]).sum() / w0
                        for j in range(i + 1, 64):
                            w1 = prob[i:j].sum()
                            w2 = prob[j:].sum()
                            if float(w1.item()) <= 0.0 or float(w2.item()) <= 0.0:
                                continue
                            m1 = (prob[i:j] * bins[i:j]).sum() / w1
                            m2 = (prob[j:] * bins[j:]).sum() / w2
                            score = w0 * (m0 - total_mean) ** 2 + w1 * (m1 - total_mean) ** 2 + w2 * (m2 - total_mean) ** 2
                            if bool(score > best_score):
                                best_score = score
                                best = (i, j)
                    thr0 = torch.tensor(float(best[0]) / 64.0, device=vals.device, dtype=vals.dtype)
                    thr1 = torch.tensor(float(best[1]) / 64.0, device=vals.device, dtype=vals.dtype)
                    pos = vals <= thr0
                    neg = vals >= thr1
                    token_filter_blend_debug["ttt_tri_replay_otsu_thresholds"] = [float(thr0.item()), float(thr1.item())]
                    token_filter_blend_debug["ttt_tri_replay_otsu_score"] = float(best_score.item())
                    return capped_role_masks(pos, neg, source="otsu3")

                def mad_role_masks() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    vals = risk.detach().float()
                    med = torch.median(vals)
                    mad = torch.median((vals - med).abs())
                    if float(mad.item()) < 1e-8:
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "mad_zero"
                        return quantile_role_masks(self.tri_replay_positive_frac, self.tri_replay_negative_frac)
                    z = (vals - med) / mad.clamp_min(1e-6)
                    pos = z <= -0.5
                    neg = z >= 1.5
                    token_filter_blend_debug["ttt_tri_replay_mad_median"] = float(med.item())
                    token_filter_blend_debug["ttt_tri_replay_mad"] = float(mad.item())
                    return capped_role_masks(pos, neg, source="mad", min_pos=0.0, max_pos=1.0, min_neg=0.0, max_neg=1.0)

                def adaptive_quantile_role_masks() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    vals = risk.detach().float()
                    q10 = torch.quantile(vals, 0.10)
                    q90 = torch.quantile(vals, 0.90)
                    spread = float((q90 - q10).item())
                    if spread < 0.15:
                        neg_frac_adapt = 0.05
                    elif spread < 0.30:
                        neg_frac_adapt = 0.12
                    else:
                        neg_frac_adapt = 0.18
                    token_filter_blend_debug["ttt_tri_replay_adaptive_spread"] = spread
                    token_filter_blend_debug["ttt_tri_replay_adaptive_neg_frac"] = float(neg_frac_adapt)
                    return quantile_role_masks(0.35, neg_frac_adapt)

                def adaptive_writer_role_masks() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    vals = risk.detach().float()
                    if vals.numel() < 3 or float(vals.max().item() - vals.min().item()) < 1e-8:
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "adaptive_writer_degenerate_risk"
                        return uncapped_role_masks(vals <= vals.mean(), vals >= vals.mean(), source="adaptive_writer_mean")
                    hist = torch.histc(vals, bins=64, min=0.0, max=1.0).float()
                    prob = hist / hist.sum().clamp_min(1e-6)
                    bins = (torch.arange(64, device=vals.device, dtype=vals.dtype) + 0.5) / 64.0
                    total_mean = (prob * bins).sum()
                    best_score = torch.tensor(-1.0, device=vals.device)
                    best = (16, 48)
                    for i in range(1, 63):
                        w0 = prob[:i].sum()
                        if float(w0.item()) <= 0.0:
                            continue
                        m0 = (prob[:i] * bins[:i]).sum() / w0
                        for j in range(i + 1, 64):
                            w1 = prob[i:j].sum()
                            w2 = prob[j:].sum()
                            if float(w1.item()) <= 0.0 or float(w2.item()) <= 0.0:
                                continue
                            m1 = (prob[i:j] * bins[i:j]).sum() / w1
                            m2 = (prob[j:] * bins[j:]).sum() / w2
                            score = w0 * (m0 - total_mean) ** 2 + w1 * (m1 - total_mean) ** 2 + w2 * (m2 - total_mean) ** 2
                            if bool(score > best_score):
                                best_score = score
                                best = (i, j)
                    thr0 = torch.tensor(float(best[0]) / 64.0, device=vals.device, dtype=vals.dtype)
                    thr1 = torch.tensor(float(best[1]) / 64.0, device=vals.device, dtype=vals.dtype)
                    token_filter_blend_debug["ttt_tri_replay_adaptive_writer_thresholds"] = [
                        float(thr0.item()),
                        float(thr1.item()),
                    ]
                    token_filter_blend_debug["ttt_tri_replay_adaptive_writer_otsu_score"] = float(best_score.item())
                    return uncapped_role_masks(vals <= thr0, vals >= thr1, source="adaptive_writer_otsu3")

                def adaptive_writer_sc_gamma_role_masks() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    """State-conditioned split roles without percentage fallback.

                    v53 replaces fixed positive/negative percentages with a
                    per-chunk robust margin over write-prior safety and risk
                    danger.  If the current chunk collapses to an empty role,
                    keep the collapse visible in debug and let the branch replay
                    use zero mass instead of silently substituting a quantile.
                    """
                    vals = risk.detach().float()
                    pp = p.detach().float()
                    if vals.numel() < 3 or pp.numel() != vals.numel():
                        token_filter_blend_debug["ttt_tri_replay_role_collapsed"] = True
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "sc_gamma_shape"
                        pos = torch.zeros_like(vals, dtype=torch.bool)
                        neg = torch.zeros_like(vals, dtype=torch.bool)
                        neu = torch.ones_like(vals, dtype=torch.bool)
                        zero = vals.new_tensor(0.0)
                        one = vals.new_tensor(1.0)
                        return pos, neu, neg, zero, one, 0.0, 0.0

                    p_min_local = pp.min()
                    p_max_local = pp.max()
                    p_norm = ((pp - p_min_local) / (p_max_local - p_min_local).clamp_min(1e-6)).clamp(0.0, 1.0)
                    safety = (p_norm * (1.0 - vals)).clamp(0.0, 1.0)
                    danger = (vals * (1.0 - p_norm)).clamp(0.0, 1.0)

                    def robust_z(x: torch.Tensor, name: str) -> torch.Tensor:
                        med = torch.median(x)
                        mad = torch.median((x - med).abs())
                        token_filter_blend_debug[f"ttt_tri_replay_sc_gamma_{name}_median"] = float(med.item())
                        token_filter_blend_debug[f"ttt_tri_replay_sc_gamma_{name}_mad"] = float(mad.item())
                        return (x - med) / mad.clamp_min(1e-6)

                    margin = robust_z(safety, "safety") - robust_z(danger, "danger")
                    margin_med = torch.median(margin)
                    margin_mad = torch.median((margin - margin_med).abs())
                    if float(margin_mad.item()) <= 1e-8:
                        token_filter_blend_debug["ttt_tri_replay_role_collapsed"] = True
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "sc_gamma_margin_mad_zero"
                        pos = torch.zeros_like(vals, dtype=torch.bool)
                        neg = torch.zeros_like(vals, dtype=torch.bool)
                    else:
                        pos_thr = margin_med + 0.5 * margin_mad
                        neg_thr = margin_med - 0.5 * margin_mad
                        pos = margin > pos_thr
                        neg = margin < neg_thr
                        pos = pos & (~neg)
                        token_filter_blend_debug["ttt_tri_replay_sc_gamma_margin_median"] = float(margin_med.item())
                        token_filter_blend_debug["ttt_tri_replay_sc_gamma_margin_mad"] = float(margin_mad.item())
                        token_filter_blend_debug["ttt_tri_replay_sc_gamma_pos_margin_threshold"] = float(pos_thr.item())
                        token_filter_blend_debug["ttt_tri_replay_sc_gamma_neg_margin_threshold"] = float(neg_thr.item())

                    pos_mass = float(pos.float().mean().item()) if pos.numel() else 0.0
                    neg_mass = float(neg.float().mean().item()) if neg.numel() else 0.0
                    collapsed = pos_mass <= 0.0 or neg_mass <= 0.0
                    token_filter_blend_debug["ttt_tri_replay_role_collapsed"] = bool(collapsed)
                    token_filter_blend_debug["ttt_tri_replay_role_source"] = "adaptive_writer_sc_gamma"
                    token_filter_blend_debug["ttt_tri_replay_role_uncapped"] = True
                    token_filter_blend_debug["ttt_tri_replay_sc_gamma_prior_std"] = float(pp.std(unbiased=False).item())
                    token_filter_blend_debug["ttt_tri_replay_sc_gamma_risk_std"] = float(vals.std(unbiased=False).item())
                    neu = ~(pos | neg)
                    pos_thr_out = vals[pos].max() if bool(pos.any()) else vals.new_tensor(0.0)
                    neg_thr_out = vals[neg].min() if bool(neg.any()) else vals.new_tensor(1.0)
                    return pos, neu, neg, pos_thr_out, neg_thr_out, pos_mass, neg_mass

                def adaptive_writer_state_energy_role_masks() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    """v54 state-conditioned split roles without mass targets.

                    Roles are derived only from current write-prior/risk state:
                    safety = norm(p) * (1 - risk), danger = risk * (2 - norm(p)).
                    Median + k*MAD thresholds keep role mass emergent rather
                    than configured as a fixed percentage.
                    """
                    vals = risk.detach().float().clamp(0.0, 1.0)
                    pp = p.detach().float()
                    if vals.numel() < 3 or pp.numel() != vals.numel():
                        token_filter_blend_debug["ttt_tri_replay_role_collapsed"] = True
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "state_energy_shape"
                        pos = torch.zeros_like(vals, dtype=torch.bool)
                        neg = torch.zeros_like(vals, dtype=torch.bool)
                        neu = torch.ones_like(vals, dtype=torch.bool)
                        zero = vals.new_tensor(0.0)
                        one = vals.new_tensor(1.0)
                        return pos, neu, neg, zero, one, 0.0, 0.0

                    p_min_local = pp.min()
                    p_max_local = pp.max()
                    p_norm = ((pp - p_min_local) / (p_max_local - p_min_local).clamp_min(1e-6)).clamp(0.0, 1.0)
                    safety = (p_norm * (1.0 - vals)).clamp(0.0, 1.0)
                    danger = (vals * (2.0 - p_norm)).clamp(0.0, 2.0)
                    k = float(self.state_energy_role_k)

                    def high_threshold(x: torch.Tensor, prefix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                        med = torch.median(x)
                        mad = torch.median((x - med).abs())
                        thr = med + k * mad
                        token_filter_blend_debug[f"ttt_tri_replay_state_energy_{prefix}_median"] = float(med.item())
                        token_filter_blend_debug[f"ttt_tri_replay_state_energy_{prefix}_mad"] = float(mad.item())
                        token_filter_blend_debug[f"ttt_tri_replay_state_energy_{prefix}_threshold"] = float(thr.item())
                        return thr, med, mad

                    safety_thr, _safety_med, safety_mad = high_threshold(safety, "safety")
                    danger_thr, _danger_med, danger_mad = high_threshold(danger, "danger")
                    pos = safety > safety_thr
                    neg = danger > danger_thr
                    both = pos & neg
                    if bool(both.any()):
                        prefer_pos = safety >= danger
                        pos = pos & ((~both) | prefer_pos)
                        neg = neg & ((~both) | (~prefer_pos))

                    pos_mass = float(pos.float().mean().item()) if pos.numel() else 0.0
                    neg_mass = float(neg.float().mean().item()) if neg.numel() else 0.0
                    collapsed = pos_mass <= 0.0 or neg_mass <= 0.0
                    token_filter_blend_debug["ttt_tri_replay_role_collapsed"] = bool(collapsed)
                    if float(safety_mad.item()) <= 1e-8 or float(danger_mad.item()) <= 1e-8:
                        token_filter_blend_debug["ttt_tri_replay_state_energy_mad_zero"] = True
                    token_filter_blend_debug["ttt_tri_replay_role_source"] = "adaptive_writer_state_energy"
                    token_filter_blend_debug["ttt_tri_replay_role_uncapped"] = True
                    token_filter_blend_debug["ttt_tri_replay_state_energy_k"] = float(k)
                    token_filter_blend_debug["ttt_tri_replay_state_energy_prior_std"] = float(pp.std(unbiased=False).item())
                    token_filter_blend_debug["ttt_tri_replay_state_energy_risk_std"] = float(vals.std(unbiased=False).item())
                    token_filter_blend_debug["ttt_tri_replay_state_energy_overlap_mass"] = float(both.float().mean().item())
                    neu = ~(pos | neg)
                    return pos, neu, neg, safety_thr, danger_thr, pos_mass, neg_mass

                def adaptive_writer_binary_write_role_masks(
                    variant: str,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    """v56 binary write roles without top percentage targets.

                    `binary_anchor` tests stable-anchor-only long writes.  Tokens
                    that are not stable anchors are left out of the long write by
                    setting their neutral long-write lambda to zero later in the
                    split branch.  `risk_veto` lets non-risk tokens commit while
                    risk tokens remain current-chunk evidence only.
                    """
                    vals = risk.detach().float().clamp(0.0, 1.0)
                    pp = p.detach().float()
                    zero = vals.new_tensor(0.0)
                    one = vals.new_tensor(1.0)
                    if vals.numel() < 3 or pp.numel() != vals.numel():
                        token_filter_blend_debug["ttt_tri_replay_role_collapsed"] = True
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = f"{variant}_shape"
                        pos = torch.zeros_like(vals, dtype=torch.bool)
                        neg = torch.zeros_like(vals, dtype=torch.bool)
                        neu = torch.ones_like(vals, dtype=torch.bool)
                        return pos, neu, neg, zero, one, 0.0, 0.0

                    p_min_local = pp.min()
                    p_max_local = pp.max()
                    p_norm = ((pp - p_min_local) / (p_max_local - p_min_local).clamp_min(1e-6)).clamp(0.0, 1.0)

                    def otsu2_threshold(score: torch.Tensor, prefix: str) -> torch.Tensor:
                        s = score.detach().float().clamp(0.0, 1.0)
                        if s.numel() < 3 or float((s.max() - s.min()).item()) <= 1e-8:
                            med = torch.median(s) if s.numel() else s.new_tensor(0.5)
                            token_filter_blend_debug[f"ttt_tri_replay_{prefix}_threshold_mode"] = "median_degenerate"
                            token_filter_blend_debug[f"ttt_tri_replay_{prefix}_threshold"] = float(med.item())
                            return med
                        hist = torch.histc(s, bins=64, min=0.0, max=1.0).float()
                        prob = hist / hist.sum().clamp_min(1e-6)
                        bins = (torch.arange(64, device=s.device, dtype=s.dtype) + 0.5) / 64.0
                        total_mean = (prob * bins).sum()
                        best_score = s.new_tensor(-1.0)
                        best = 32
                        for idx in range(1, 64):
                            w0 = prob[:idx].sum()
                            w1 = prob[idx:].sum()
                            if float(w0.item()) <= 0.0 or float(w1.item()) <= 0.0:
                                continue
                            m0 = (prob[:idx] * bins[:idx]).sum() / w0
                            m1 = (prob[idx:] * bins[idx:]).sum() / w1
                            between = w0 * (m0 - total_mean) ** 2 + w1 * (m1 - total_mean) ** 2
                            if bool(between > best_score):
                                best_score = between
                                best = idx
                        thr = s.new_tensor(float(best) / 64.0)
                        token_filter_blend_debug[f"ttt_tri_replay_{prefix}_threshold_mode"] = "otsu2"
                        token_filter_blend_debug[f"ttt_tri_replay_{prefix}_threshold"] = float(thr.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{prefix}_otsu_score"] = float(best_score.item())
                        return thr

                    if variant == "binary_anchor":
                        # risk_source normally folds C23 D_g into residual risk
                        # for v56 (`ttt_residual_x_dg`).  There is no separate
                        # D token in this controller path, so record the boundary
                        # explicitly instead of pretending an independent D_i was
                        # available here.
                        stable_score = (p_norm * (1.0 - vals)).clamp(0.0, 1.0)
                        thr = otsu2_threshold(stable_score, "binary_anchor")
                        pos = stable_score >= thr
                        neu = ~pos
                        token_filter_blend_debug["ttt_tri_replay_binary_anchor_score_mean"] = float(stable_score.mean().item())
                        token_filter_blend_debug["ttt_tri_replay_binary_anchor_score_p90"] = float(torch.quantile(stable_score, 0.90).item())
                        token_filter_blend_debug["ttt_tri_replay_binary_anchor_d_folded_into_risk"] = True
                    elif variant == "risk_veto":
                        eps = vals.new_tensor(1e-3)
                        veto_score = (vals * (1.0 - p_norm + eps)).clamp(0.0, 1.0)
                        thr = otsu2_threshold(veto_score, "risk_veto")
                        risk_tokens = veto_score >= thr
                        pos = ~risk_tokens
                        neu = risk_tokens
                        token_filter_blend_debug["ttt_tri_replay_risk_veto_score_mean"] = float(veto_score.mean().item())
                        token_filter_blend_debug["ttt_tri_replay_risk_veto_score_p90"] = float(torch.quantile(veto_score, 0.90).item())
                    else:
                        raise ValueError(f"Unsupported v56 binary write variant: {variant}")

                    neg = torch.zeros_like(vals, dtype=torch.bool)
                    pos_mass = float(pos.float().mean().item()) if pos.numel() else 0.0
                    neu_mass = float(neu.float().mean().item()) if neu.numel() else 0.0
                    token_filter_blend_debug["ttt_tri_replay_role_source"] = f"v56_{variant}"
                    token_filter_blend_debug["ttt_tri_replay_role_uncapped"] = True
                    token_filter_blend_debug["ttt_tri_replay_no_negative_branch"] = True
                    token_filter_blend_debug["ttt_tri_replay_stable_anchor_token_mass"] = float(pos_mass)
                    token_filter_blend_debug["ttt_tri_replay_no_long_write_token_mass"] = float(neu_mass)
                    token_filter_blend_debug["ttt_tri_replay_risk_token_mass"] = float(neu_mass if variant == "risk_veto" else 0.0)
                    token_filter_blend_debug["ttt_tri_replay_role_collapsed"] = bool(pos_mass <= 0.0 or neu_mass <= 0.0)
                    return pos, neu, neg, thr, one, pos_mass, 0.0

                def adaptive_writer_robust_role_masks() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    """Risk/prior-shaped adaptive roles without fixed top percentages.

                    The Otsu-only writer tended to classify the large zero-risk
                    plateau as positive evidence.  This variant asks for both
                    low risk and high write prior for positive replay, while
                    negative replay is driven by high risk and low write prior.
                    Thresholds are derived from robust per-chunk distribution
                    scale, not from a configured mass.
                    """
                    vals = risk.detach().float()
                    pp = p.detach().float()
                    if vals.numel() < 3 or pp.numel() != vals.numel():
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "adaptive_writer_robust_shape"
                        return adaptive_writer_role_masks()
                    p_min_local = pp.min()
                    p_max_local = pp.max()
                    p_norm = ((pp - p_min_local) / (p_max_local - p_min_local).clamp_min(1e-6)).clamp(0.0, 1.0)
                    safety = (p_norm * (1.0 - vals)).clamp(0.0, 1.0)
                    danger = (vals * (1.0 + (1.0 - p_norm))).clamp(0.0, 2.0)

                    def high_threshold(x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
                        x = x.detach().float().reshape(-1)
                        med = torch.median(x)
                        mad = torch.median((x - med).abs())
                        mean = x.mean()
                        std = x.std(unbiased=False)
                        if float(mad.item()) > 1e-8:
                            thr = med + mad
                            mode = "median_plus_mad"
                        elif float(std.item()) > 1e-8:
                            thr = mean + std
                            mode = "mean_plus_std"
                        else:
                            thr = x.max()
                            mode = "max_degenerate"
                        return thr, {
                            "median": float(med.item()),
                            "mad": float(mad.item()),
                            "mean": float(mean.item()),
                            "std": float(std.item()),
                            "threshold": float(thr.item()),
                            "mode": mode,
                        }

                    safety_thr, safety_debug = high_threshold(safety)
                    danger_thr, danger_debug = high_threshold(danger)
                    pos = safety >= safety_thr
                    neg = danger >= danger_thr
                    pos = pos & (~neg)

                    if not bool(pos.any()) or not bool(neg.any()):
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "adaptive_writer_robust_empty"
                        token_filter_blend_debug["ttt_tri_replay_role_fallback_pos_any"] = bool(pos.any())
                        token_filter_blend_debug["ttt_tri_replay_role_fallback_neg_any"] = bool(neg.any())
                        return adaptive_writer_role_masks()

                    token_filter_blend_debug["ttt_tri_replay_robust_safety"] = safety_debug
                    token_filter_blend_debug["ttt_tri_replay_robust_danger"] = danger_debug
                    token_filter_blend_debug["ttt_tri_replay_robust_prior_std"] = float(pp.std(unbiased=False).item())
                    token_filter_blend_debug["ttt_tri_replay_robust_risk_std"] = float(vals.std(unbiased=False).item())
                    return uncapped_role_masks(pos, neg, source="adaptive_writer_robust")

                def adaptive_writer_cluster3d_role_masks() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
                    vals = risk.detach().float().reshape(-1)
                    pp = p.detach().float().reshape(-1)
                    if vals.numel() < 3 or pp.numel() != vals.numel():
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "adaptive_writer_cluster3d_shape"
                        return adaptive_writer_robust_role_masks()
                    p_min_local = pp.min()
                    p_max_local = pp.max()
                    p_norm = ((pp - p_min_local) / (p_max_local - p_min_local).clamp_min(1e-6)).clamp(0.0, 1.0)

                    update_norm = vals.new_zeros(vals.numel())
                    try:
                        kf = k_native_full.detach().float()
                        vf = v_native_full.detach().float()
                        lr = lr0.detach().float().abs().squeeze(-1)
                        if lr.ndim == 2:
                            lr_tok = lr.mean(dim=0)
                        else:
                            lr_tok = lr.reshape(-1)
                        n = min(int(vals.numel()), int(kf.shape[1]), int(vf.shape[1]), int(lr_tok.numel()))
                        if n > 0:
                            k_norm = kf[:, :n, :].norm(dim=-1).mean(dim=0)
                            v_norm = vf[:, :n, :].norm(dim=-1).mean(dim=0)
                            upd = (k_norm * v_norm * lr_tok[:n]).detach().float()
                            update_norm[:n] = self._normalize01_vec(upd)
                    except RuntimeError as exc:
                        token_filter_blend_debug["ttt_tri_replay_cluster3d_update_proxy_error"] = str(exc)
                        update_norm = vals.clone()

                    x = torch.stack([vals, (1.0 - p_norm).clamp(0.0, 1.0), update_norm.clamp(0.0, 1.0)], dim=1)
                    pos_score = x[:, 0] + x[:, 1] + 0.5 * x[:, 2]
                    neg_score = x[:, 0] + x[:, 1] + x[:, 2]
                    init_idx = [
                        int(torch.argmin(pos_score).item()),
                        int(torch.argmin((neg_score - torch.median(neg_score)).abs()).item()),
                        int(torch.argmax(neg_score).item()),
                    ]
                    centers = x[torch.tensor(init_idx, device=x.device, dtype=torch.long)].clone()
                    labels = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
                    for _ in range(8):
                        dist = torch.cdist(x, centers, p=2.0)
                        labels = torch.argmin(dist, dim=1)
                        new_centers = centers.clone()
                        for ci in range(3):
                            mask = labels == ci
                            if bool(mask.any()):
                                new_centers[ci] = x[mask].mean(dim=0)
                        if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
                            centers = new_centers
                            break
                        centers = new_centers

                    cluster_pos_score = centers[:, 0] + centers[:, 1] + 0.5 * centers[:, 2]
                    cluster_neg_score = centers[:, 0] + centers[:, 1] + centers[:, 2]
                    pos_cluster = int(torch.argmin(cluster_pos_score).item())
                    neg_cluster = int(torch.argmax(cluster_neg_score).item())
                    if pos_cluster == neg_cluster:
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "adaptive_writer_cluster3d_degenerate"
                        return adaptive_writer_robust_role_masks()
                    pos = labels == pos_cluster
                    neg = labels == neg_cluster
                    pos = pos & (~neg)
                    if not bool(pos.any()) or not bool(neg.any()):
                        token_filter_blend_debug["ttt_tri_replay_role_fallback"] = "adaptive_writer_cluster3d_empty"
                        return adaptive_writer_robust_role_masks()
                    token_filter_blend_debug["ttt_tri_replay_cluster3d_centers"] = [
                        [float(v) for v in row]
                        for row in centers.detach().cpu().tolist()
                    ]
                    token_filter_blend_debug["ttt_tri_replay_cluster3d_pos_cluster"] = int(pos_cluster)
                    token_filter_blend_debug["ttt_tri_replay_cluster3d_neg_cluster"] = int(neg_cluster)
                    return uncapped_role_masks(pos, neg, source="adaptive_writer_cluster3d")

                if gr_mode in tri_modes:
                    p = prior_flat.detach().float().reshape(-1)
                    if p.numel() != int(l):
                        p_aligned = torch.ones(int(l), dtype=torch.float32, device=p.device)
                        n = min(int(p.numel()), int(l))
                        if n > 0:
                            p_aligned[:n] = p[:n]
                        p = p_aligned
                    neg_frac_cfg = float(self.tri_replay_negative_frac)
                    if neg_frac_cfg <= 0.0:
                        neg_frac_cfg = float(self.gradient_reversal_negative_frac)
                    role_mode = str(self.tri_replay_role_mode or "fixed").strip().lower()
                    token_filter_blend_debug["ttt_tri_replay_role_mode"] = role_mode
                    if role_mode in {"", "fixed", "quantile", "fixed_quantile"}:
                        pos_mask, neu_mask, neg_mask, pos_thr, neg_thr, pos_frac, neg_frac = quantile_role_masks(
                            self.tri_replay_positive_frac,
                            neg_frac_cfg,
                        )
                    elif role_mode in {"kmeans3", "kmeans", "1d_kmeans3"}:
                        pos_mask, neu_mask, neg_mask, pos_thr, neg_thr, pos_frac, neg_frac = kmeans3_role_masks()
                    elif role_mode in {"otsu3", "otsu", "otsu_hist"}:
                        pos_mask, neu_mask, neg_mask, pos_thr, neg_thr, pos_frac, neg_frac = otsu3_role_masks()
                    elif role_mode in {"mad", "robust_mad"}:
                        pos_mask, neu_mask, neg_mask, pos_thr, neg_thr, pos_frac, neg_frac = mad_role_masks()
                    elif role_mode in {"adaptive_quantile", "adaptive", "spread_quantile"}:
                        pos_mask, neu_mask, neg_mask, pos_thr, neg_thr, pos_frac, neg_frac = adaptive_quantile_role_masks()
                    elif role_mode in {
                        "adaptive_writer",
                        "adaptive_writer_fused",
                        "adaptive_otsu_writer",
                        "adaptive_otsu_fused",
                        "no_percentage",
                        "no_percentage_fused",
                    }:
                        pos_mask, neu_mask, neg_mask, pos_thr, neg_thr, pos_frac, neg_frac = adaptive_writer_role_masks()
                    elif role_mode in {
                        "adaptive_writer_robust",
                        "adaptive_writer_robust_fused",
                        "adaptive_writer_robust_split",
                        "robust_adaptive_writer",
                        "robust_adaptive_writer_fused",
                        "robust_adaptive_writer_split",
                        "no_percentage_robust",
                        "no_percentage_robust_fused",
                        "no_percentage_robust_split",
                        "adaptive_writer_conflictlite_split",
                        "conflictlite_adaptive_writer_split",
                        "no_percentage_conflictlite_split",
                        "adaptive_writer_energy_matched_split",
                        "energy_matched_adaptive_writer_split",
                        "no_percentage_energy_matched_split",
                    }:
                        pos_mask, neu_mask, neg_mask, pos_thr, neg_thr, pos_frac, neg_frac = adaptive_writer_robust_role_masks()
                    elif role_mode in {
                        "adaptive_writer_sc_gamma_split",
                        "sc_gamma_split",
                        "no_percentage_sc_gamma_split",
                        "adaptive_writer_sc_gamma_commit_split",
                        "sc_gamma_commit_split",
                        "no_percentage_sc_gamma_commit_split",
                    }:
                        pos_mask, neu_mask, neg_mask, pos_thr, neg_thr, pos_frac, neg_frac = adaptive_writer_sc_gamma_role_masks()
                    elif role_mode in {
                        "adaptive_writer_state_energy_matched_split",
                        "state_energy_matched_split",
                        "no_percentage_state_energy_matched_split",
                        "adaptive_writer_state_energy_commit_split",
                        "adaptive_writer_state_energy_directional_commit_split",
                        "state_energy_directional_commit_split",
                        "no_percentage_state_energy_directional_commit_split",
                        "adaptive_writer_tail_state_continuity_guard",
                        "tail_state_continuity_guard",
                        "adaptive_writer_tail_state_continuity_guard_selective_commit",
                        "tail_state_continuity_guard_selective_commit",
                    }:
                        pos_mask, neu_mask, neg_mask, pos_thr, neg_thr, pos_frac, neg_frac = adaptive_writer_state_energy_role_masks()
                    elif role_mode in {
                        "adaptive_writer_binary_anchor_split",
                        "binary_stable_anchor_split",
                        "stable_anchor_binary_split",
                    }:
                        pos_mask, neu_mask, neg_mask, pos_thr, neg_thr, pos_frac, neg_frac = adaptive_writer_binary_write_role_masks(
                            "binary_anchor"
                        )
                    elif role_mode in {
                        "adaptive_writer_risk_veto_split",
                        "risk_veto_binary_split",
                        "no_long_write_risk_veto_split",
                    }:
                        pos_mask, neu_mask, neg_mask, pos_thr, neg_thr, pos_frac, neg_frac = adaptive_writer_binary_write_role_masks(
                            "risk_veto"
                        )
                    elif role_mode in {
                        "adaptive_writer_cluster3d_split",
                        "cluster3d_adaptive_writer_split",
                        "no_percentage_cluster3d_split",
                    }:
                        pos_mask, neu_mask, neg_mask, pos_thr, neg_thr, pos_frac, neg_frac = adaptive_writer_cluster3d_role_masks()
                    else:
                        raise ValueError(f"Unsupported tri replay role mode: {self.tri_replay_role_mode}")

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

                    adaptive_writer_fused = role_mode in {
                        "adaptive_writer",
                        "adaptive_writer_fused",
                        "adaptive_otsu_writer",
                        "adaptive_otsu_fused",
                        "no_percentage",
                        "no_percentage_fused",
                        "adaptive_writer_robust",
                        "adaptive_writer_robust_fused",
                        "robust_adaptive_writer",
                        "robust_adaptive_writer_fused",
                        "no_percentage_robust",
                        "no_percentage_robust_fused",
                    }
                    if adaptive_writer_fused:
                        token_count = max(float(risk.numel()), 1.0)
                        pos_risk_mean = risk[pos_mask].mean() if bool(pos_mask.any()) else risk.mean()
                        neg_risk_mean = risk[neg_mask].mean() if bool(neg_mask.any()) else risk.mean()
                        neg_mass = neg_mask.float().mean()
                        risk_gap = (neg_risk_mean - pos_risk_mean).clamp_min(0.0)
                        prior_scale = p.std(unbiased=False).to(device=risk.device, dtype=risk.dtype).clamp_min(1e-4)
                        if role_mode in {
                            "adaptive_writer_robust",
                            "adaptive_writer_robust_fused",
                            "robust_adaptive_writer",
                            "robust_adaptive_writer_fused",
                            "no_percentage_robust",
                            "no_percentage_robust_fused",
                        }:
                            gamma_eff_t = (risk_gap * neg_mass * prior_scale).clamp(1e-4, 2e-2)
                        else:
                            gamma_eff_t = (risk_gap * neg_mass / torch.sqrt(risk.new_tensor(token_count))).clamp(1e-4, 2e-2)
                        if bool(neu_mask.any()):
                            neu_lambda_t = (1.0 - risk[neu_mask].mean()).clamp(0.20, 1.0)
                        else:
                            neu_lambda_t = risk.new_tensor(0.0)
                        signed_vec = (
                            p * pos_mask.float()
                            + neu_lambda_t * p * neu_mask.float()
                            - gamma_eff_t * risk * neg_mask.float()
                        )
                        fused_w0, fused_w1, fused_w2 = replay_group(signed_vec)
                        token_filter_blend_debug["ttt_tri_replay_adaptive_writer_fused"] = True
                        token_filter_blend_debug["ttt_tri_replay_adaptive_gamma"] = float(gamma_eff_t.item())
                        token_filter_blend_debug["ttt_tri_replay_adaptive_neutral_lambda"] = float(neu_lambda_t.item())
                        token_filter_blend_debug["ttt_tri_replay_adaptive_risk_gap"] = float(risk_gap.item())
                        token_filter_blend_debug["ttt_tri_replay_adaptive_prior_scale"] = float(prior_scale.item())
                        token_filter_blend_debug["ttt_tri_replay_adaptive_token_count"] = int(risk.numel())
                        token_filter_blend_debug["ttt_two_replay_applied"] = False
                        token_filter_blend_debug["ttt_two_replay_debug_note"] = "tri_replay_path; use ttt_tri_replay_applied"
                        token_filter_blend_debug["ttt_tri_replay_applied"] = True
                        token_filter_blend_debug["ttt_tri_replay_active_branches"] = list(active)
                        token_filter_blend_debug["ttt_tri_replay_positive_frac"] = float(pos_frac)
                        token_filter_blend_debug["ttt_tri_replay_negative_frac"] = float(neg_frac)
                        token_filter_blend_debug["ttt_tri_replay_neutral_lambda"] = float(neu_lambda_t.item())
                        token_filter_blend_debug["ttt_tri_replay_pos_threshold"] = float(pos_thr.item())
                        token_filter_blend_debug["ttt_tri_replay_neg_threshold"] = float(neg_thr.item())
                        token_filter_blend_debug["ttt_tri_replay_pos_mass"] = float(pos_mask.float().mean().item())
                        token_filter_blend_debug["ttt_tri_replay_neu_mass"] = float(neu_mask.float().mean().item())
                        token_filter_blend_debug["ttt_tri_replay_neg_mass"] = float(neg_mask.float().mean().item())
                        token_filter_blend_debug["ttt_two_replay_active_branches"] = list(active)
                        token_filter_blend_debug["ttt_two_replay_risk_mean"] = float(risk.detach().float().mean().item())
                        token_filter_blend_debug["ttt_two_replay_risk_p90"] = float(torch.quantile(risk.detach().float(), 0.90).item())
                        return fused_w0, fused_w1, fused_w2

                    pos_vec = (p * pos_mask.float()).clamp_min(0.0)
                    neu_vec = (p * neu_mask.float()).clamp_min(0.0)
                    neg_vec = (risk * neg_mask.float()).clamp_min(0.0)
                    pos_w0, pos_w1, pos_w2 = replay_group(pos_vec)
                    neu_w0, neu_w1, neu_w2 = replay_group(neu_vec)
                    neg_w0, neg_w1, neg_w2 = replay_group(neg_vec)
                else:
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
                branch_gammas = {
                    str(k): float(v)
                    for k, v in gradient_reversal_debug.get(
                        "ttt_gradient_reversal_active_branch_gammas", {}
                    ).items()
                }
                adaptive_writer_split = role_mode in {
                    "adaptive_writer_robust_split",
                    "robust_adaptive_writer_split",
                    "no_percentage_robust_split",
                    "adaptive_writer_conflictlite_split",
                    "conflictlite_adaptive_writer_split",
                    "no_percentage_conflictlite_split",
                    "adaptive_writer_energy_matched_split",
                    "energy_matched_adaptive_writer_split",
                    "no_percentage_energy_matched_split",
                    "adaptive_writer_sc_gamma_split",
                    "sc_gamma_split",
                    "no_percentage_sc_gamma_split",
                    "adaptive_writer_sc_gamma_commit_split",
                    "sc_gamma_commit_split",
                    "no_percentage_sc_gamma_commit_split",
                    "adaptive_writer_state_energy_matched_split",
                    "state_energy_matched_split",
                    "no_percentage_state_energy_matched_split",
                    "adaptive_writer_state_energy_commit_split",
                    "adaptive_writer_state_energy_directional_commit_split",
                    "state_energy_directional_commit_split",
                    "no_percentage_state_energy_directional_commit_split",
                    "adaptive_writer_tail_state_continuity_guard",
                    "tail_state_continuity_guard",
                    "adaptive_writer_tail_state_continuity_guard_selective_commit",
                    "tail_state_continuity_guard_selective_commit",
                    "adaptive_writer_binary_anchor_split",
                    "binary_stable_anchor_split",
                    "stable_anchor_binary_split",
                    "adaptive_writer_risk_veto_split",
                    "risk_veto_binary_split",
                    "no_long_write_risk_veto_split",
                    "adaptive_writer_cluster3d_split",
                    "cluster3d_adaptive_writer_split",
                    "no_percentage_cluster3d_split",
                }
                adaptive_writer_energy_matched_split = role_mode in {
                    "adaptive_writer_energy_matched_split",
                    "energy_matched_adaptive_writer_split",
                    "no_percentage_energy_matched_split",
                }
                adaptive_writer_sc_gamma_split = role_mode in {
                    "adaptive_writer_sc_gamma_split",
                    "sc_gamma_split",
                    "no_percentage_sc_gamma_split",
                    "adaptive_writer_sc_gamma_commit_split",
                    "sc_gamma_commit_split",
                    "no_percentage_sc_gamma_commit_split",
                }
                adaptive_writer_state_energy_matched_split = role_mode in {
                    "adaptive_writer_state_energy_matched_split",
                    "state_energy_matched_split",
                    "no_percentage_state_energy_matched_split",
                    "adaptive_writer_state_energy_commit_split",
                    "adaptive_writer_state_energy_directional_commit_split",
                    "state_energy_directional_commit_split",
                    "no_percentage_state_energy_directional_commit_split",
                }
                adaptive_writer_tail_state_continuity_guard = role_mode in {
                    "adaptive_writer_tail_state_continuity_guard",
                    "tail_state_continuity_guard",
                    "adaptive_writer_tail_state_continuity_guard_selective_commit",
                    "tail_state_continuity_guard_selective_commit",
                }
                adaptive_writer_binary_no_long_write = role_mode in {
                    "adaptive_writer_binary_anchor_split",
                    "binary_stable_anchor_split",
                    "stable_anchor_binary_split",
                    "adaptive_writer_risk_veto_split",
                    "risk_veto_binary_split",
                    "no_long_write_risk_veto_split",
                }
                adaptive_split_gamma: Optional[float] = None
                if adaptive_writer_split:
                    pos_risk_mean = risk[pos_mask].mean() if bool(pos_mask.any()) else risk.mean()
                    neg_risk_mean = risk[neg_mask].mean() if bool(neg_mask.any()) else risk.mean()
                    neg_mass = neg_mask.float().mean()
                    risk_gap = (neg_risk_mean - pos_risk_mean).clamp_min(0.0)
                    prior_scale = p.std(unbiased=False).to(device=risk.device, dtype=risk.dtype).clamp_min(1e-4)
                    gamma_eff_t = (risk_gap * neg_mass * prior_scale).clamp(1e-4, 2e-2)
                    if bool(neu_mask.any()):
                        neu_lambda_t = (1.0 - risk[neu_mask].mean()).clamp(0.20, 1.0)
                    else:
                        neu_lambda_t = risk.new_tensor(0.0)
                    adaptive_split_gamma = float(gamma_eff_t.item())
                    neu_lambda = float(neu_lambda_t.item())
                    token_filter_blend_debug["ttt_tri_replay_adaptive_writer_split"] = True
                    token_filter_blend_debug["ttt_tri_replay_adaptive_writer_energy_matched_split"] = bool(
                        adaptive_writer_energy_matched_split
                    )
                    token_filter_blend_debug["ttt_tri_replay_adaptive_writer_state_energy_matched_split"] = bool(
                        adaptive_writer_state_energy_matched_split
                    )
                    token_filter_blend_debug["ttt_tri_replay_adaptive_gamma"] = float(gamma_eff_t.item())
                    token_filter_blend_debug["ttt_tri_replay_adaptive_neutral_lambda"] = float(neu_lambda_t.item())
                    token_filter_blend_debug["ttt_tri_replay_adaptive_risk_gap"] = float(risk_gap.item())
                    token_filter_blend_debug["ttt_tri_replay_adaptive_prior_scale"] = float(prior_scale.item())
                    token_filter_blend_debug["ttt_tri_replay_adaptive_token_count"] = int(risk.numel())
                else:
                    neu_lambda = float(self.tri_replay_neutral_lambda)
                energy_matched_gammas: List[float] = []
                energy_matched_lambdas: List[float] = []

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
                    pos_delta = pos.float() - old.float()
                    neu_delta = neu.float() - old.float()
                    neg_delta = neg.float() - old.float()
                    pos_norm = pos_delta.detach().norm(dim=1)
                    neu_norm = neu_delta.detach().norm(dim=1)
                    neg_norm = neg_delta.detach().norm(dim=1)
                    if adaptive_writer_binary_no_long_write:
                        gamma = 0.0
                        neu_lambda_branch = 0.0
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_v56_binary_no_long_write"] = True
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_v56_binary_gamma"] = float(gamma)
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_v56_binary_neutral_lambda"] = float(
                            neu_lambda_branch
                        )
                    elif adaptive_writer_energy_matched_split:
                        pos_rms = pos_norm.square().mean().sqrt()
                        neu_rms = neu_norm.square().mean().sqrt()
                        neg_rms = neg_norm.square().mean().sqrt()
                        pos_risk_mean = risk[pos_mask].mean() if bool(pos_mask.any()) else risk.mean()
                        neg_risk_mean = risk[neg_mask].mean() if bool(neg_mask.any()) else risk.mean()
                        sep = (neg_risk_mean - pos_risk_mean).clamp(0.0, 1.0)
                        gamma_t = (pos_rms / neg_rms.clamp_min(1e-6) * sep).clamp(1e-4, 2e-2)
                        if bool(neu_mask.any()):
                            neu_risk_mean = risk[neu_mask].mean()
                            neu_lambda_t = (pos_rms / neu_rms.clamp_min(1e-6) * (1.0 - neu_risk_mean)).clamp(0.20, 1.0)
                        else:
                            neu_lambda_t = risk.new_tensor(0.0)
                        gamma = float(gamma_t.item())
                        neu_lambda_branch = float(neu_lambda_t.item())
                        energy_matched_gammas.append(float(gamma))
                        energy_matched_lambdas.append(float(neu_lambda_branch))
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_energy_matched_pos_rms"] = float(pos_rms.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_energy_matched_neu_rms"] = float(neu_rms.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_energy_matched_neg_rms"] = float(neg_rms.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_energy_matched_risk_sep"] = float(sep.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_energy_matched_neutral_lambda"] = float(neu_lambda_branch)
                    elif adaptive_writer_state_energy_matched_split or adaptive_writer_tail_state_continuity_guard:
                        native_delta = full_pos.float() - old.float()
                        native_norm = native_delta.detach().norm(dim=1)
                        native_rms = native_norm.square().mean().sqrt()
                        configured_gamma0 = float(branch_gammas.get(str(branch_idx), 0.0))
                        gamma0 = configured_gamma0
                        gamma0_source = "branch_or_layer_gamma"
                        if gamma0 <= 0.0:
                            gamma0 = float(self.gradient_reversal_gamma)
                            gamma0_source = "global_gradient_reversal_gamma"
                        if gamma0 <= 0.0:
                            gamma0 = 0.0075
                            gamma0_source = "v54_default_gamma0"
                        if bool(neu_mask.any()):
                            neu_lambda_t = (1.0 - risk[neu_mask].mean()).clamp(0.20, 1.0)
                        else:
                            neu_lambda_t = risk.new_tensor(0.0)
                        pre_raw = (
                            old.float()
                            + pos_delta
                            + float(neu_lambda_t.item()) * neu_delta
                            - float(gamma0) * neg_delta
                        )
                        pre_controlled = renorm_like(full_pos, pre_raw)
                        candidate_delta = pre_controlled.float() - old.float()
                        candidate_norm = candidate_delta.detach().norm(dim=1)
                        candidate_rms = candidate_norm.square().mean().sqrt()
                        ema_key = f"layer_{int(layer_idx)}_branch_{int(branch_idx)}"
                        ema_prev = self.state_energy_target_ema.get(ema_key)
                        if ema_prev is None:
                            target_energy = native_rms.detach()
                            ema_source = "bootstrap_current_native"
                        else:
                            target_energy = native_rms.new_tensor(float(ema_prev))
                            ema_source = "causal_ema_previous"
                        g_energy = (
                            target_energy / candidate_rms.clamp_min(1e-6)
                        ).clamp(self.state_energy_gamma_gain_min, self.state_energy_gamma_gain_max)
                        risk_med = torch.median(risk.detach().float())
                        risk_mad = torch.median((risk.detach().float() - risk_med).abs())
                        neg_mass = neg_mask.float().mean()
                        g_risk = (risk_mad * neg_mass).clamp(0.0, 1.0)
                        gamma_t = (
                            float(gamma0) * g_energy * (1.0 + g_risk)
                        ).clamp(self.state_energy_gamma_min, self.state_energy_gamma_max)
                        neu_lambda_branch = float(neu_lambda_t.item())
                        if adaptive_writer_tail_state_continuity_guard:
                            flat_c = candidate_delta.detach().float().reshape(-1)
                            flat_n = native_delta.detach().float().reshape(-1)
                            cos_den = (flat_c.norm() * flat_n.norm()).clamp_min(1e-12)
                            cand_native_cos = ((flat_c @ flat_n) / cos_den).clamp(-1.0, 1.0)
                            energy_ref = target_energy.detach().clamp_min(1e-6)
                            overshoot = (candidate_rms.detach() / energy_ref).clamp(0.0, 10.0)
                            neutral_risk = risk[neu_mask].mean() if bool(neu_mask.any()) else risk.mean()
                            p_med = torch.median(p.detach().float())
                            p_mad = torch.median((p.detach().float() - p_med).abs())
                            static_anchor = ((p.detach().float() >= p_med + p_mad) & (risk.detach().float() <= risk_med)).float().mean()
                            chunk_idx = int(getattr(self, "current_chunk_idx", getattr(self, "v11_projection_chunk_idx", -1)))
                            reset_age = float(chunk_idx % 5) if chunk_idx >= 0 else 0.0
                            reset_age_factor = risk.new_tensor(reset_age / 4.0)
                            cos_risk = ((risk.new_tensor(0.88) - cand_native_cos) / 0.38).clamp(0.0, 1.0)
                            overshoot_risk = ((overshoot - 1.05) / 0.80).clamp(0.0, 1.0)
                            neutral_risk_term = neutral_risk.clamp(0.0, 1.0)
                            static_risk = (1.0 - static_anchor).clamp(0.0, 1.0)
                            tail_risk = (
                                0.35 * overshoot_risk
                                + 0.30 * cos_risk
                                + 0.15 * neutral_risk_term
                                + 0.10 * static_risk
                                + 0.10 * reset_age_factor
                            ).clamp(0.0, 1.0)
                            gamma_shrink = (1.0 - 0.65 * tail_risk).clamp(0.25, 1.0)
                            lambda_boost = (0.45 + 0.50 * tail_risk).clamp(0.45, 0.95)
                            gamma_t = (gamma_t * gamma_shrink).clamp(self.state_energy_gamma_min, self.state_energy_gamma_max)
                            neu_lambda_branch = float(max(float(neu_lambda_branch), float(lambda_boost.item())))
                            tail_key = f"layer_{int(layer_idx)}_branch_{int(branch_idx)}"
                            prev_tail = self.tail_state_energy_ema.get(tail_key)
                            ema_in = float(candidate_rms.detach().item())
                            self.tail_state_energy_ema[tail_key] = (
                                ema_in if prev_tail is None else 0.75 * float(prev_tail) + 0.25 * ema_in
                            )
                            token_filter_blend_debug[f"ttt_tri_replay_{name}_tail_guard_reset_age_mod5"] = float(reset_age)
                            token_filter_blend_debug[f"ttt_tri_replay_{name}_tail_guard_candidate_native_cos"] = float(cand_native_cos.item())
                            token_filter_blend_debug[f"ttt_tri_replay_{name}_tail_guard_overshoot"] = float(overshoot.item())
                            token_filter_blend_debug[f"ttt_tri_replay_{name}_tail_guard_neutral_risk"] = float(neutral_risk.item())
                            token_filter_blend_debug[f"ttt_tri_replay_{name}_tail_guard_static_anchor_mass"] = float(static_anchor.item())
                            token_filter_blend_debug[f"ttt_tri_replay_{name}_tail_guard_risk"] = float(tail_risk.item())
                            token_filter_blend_debug[f"ttt_tri_replay_{name}_tail_guard_gamma_shrink"] = float(gamma_shrink.item())
                            token_filter_blend_debug[f"ttt_tri_replay_{name}_tail_guard_lambda_boost"] = float(lambda_boost.item())
                            token_filter_blend_debug[f"ttt_tri_replay_{name}_tail_guard_energy_ema_next"] = float(self.tail_state_energy_ema[tail_key])
                        gamma = float(gamma_t.item())
                        updated_target = (
                            (1.0 - float(self.state_energy_ema_alpha)) * float(target_energy.item())
                            + float(self.state_energy_ema_alpha) * float(native_rms.item())
                        )
                        self.state_energy_target_ema[ema_key] = float(updated_target)
                        energy_matched_gammas.append(float(gamma))
                        energy_matched_lambdas.append(float(neu_lambda_branch))
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_state_energy_gamma0"] = float(gamma0)
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_state_energy_gamma0_source"] = gamma0_source
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_state_energy_native_rms"] = float(native_rms.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_state_energy_candidate_rms"] = float(candidate_rms.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_state_energy_target_rms"] = float(target_energy.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_state_energy_target_source"] = ema_source
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_state_energy_target_ema_next"] = float(updated_target)
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_state_energy_g_energy"] = float(g_energy.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_state_energy_risk_mad"] = float(risk_mad.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_state_energy_neg_mass"] = float(neg_mass.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_state_energy_g_risk"] = float(g_risk.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_state_energy_neutral_lambda"] = float(neu_lambda_branch)
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_tail_state_continuity_guard"] = bool(
                            adaptive_writer_tail_state_continuity_guard
                        )
                    elif adaptive_writer_sc_gamma_split:
                        native_delta = full_pos.float() - old.float()
                        native_norm = native_delta.detach().norm(dim=1)
                        native_rms = native_norm.square().mean().sqrt()
                        neg_rms = neg_norm.square().mean().sqrt()
                        risk_var = risk.detach().float().var(unbiased=False)
                        p_var = p.detach().float().var(unbiased=False)
                        var_ratio = torch.sqrt(risk_var / p_var.clamp_min(1e-6))
                        configured_rho = float(branch_gammas.get(str(branch_idx), 0.0))
                        rho = configured_rho if configured_rho > 0.0 else 0.005
                        gamma_t = (rho * native_rms / neg_rms.clamp_min(1e-6) * var_ratio).clamp(1e-4, 2e-2)
                        if bool(neu_mask.any()):
                            neu_lambda_t = (1.0 - risk[neu_mask].mean()).clamp(0.20, 1.0)
                        else:
                            neu_lambda_t = risk.new_tensor(0.0)
                        gamma = float(gamma_t.item())
                        neu_lambda_branch = float(neu_lambda_t.item())
                        energy_matched_gammas.append(float(gamma))
                        energy_matched_lambdas.append(float(neu_lambda_branch))
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_sc_gamma_rho"] = float(rho)
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_sc_gamma_configured_rho"] = float(configured_rho)
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_sc_gamma_rho_source"] = (
                            "layer_or_branch_gamma" if configured_rho > 0.0 else "default"
                        )
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_sc_gamma_native_rms"] = float(native_rms.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_sc_gamma_neg_rms"] = float(neg_rms.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_sc_gamma_risk_var"] = float(risk_var.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_sc_gamma_prior_var"] = float(p_var.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_sc_gamma_var_ratio"] = float(var_ratio.item())
                        token_filter_blend_debug[f"ttt_tri_replay_{name}_sc_gamma_neutral_lambda"] = float(neu_lambda_branch)
                    else:
                        gamma = (
                            float(adaptive_split_gamma)
                            if adaptive_writer_split and adaptive_split_gamma is not None
                            else float(branch_gammas.get(str(branch_idx), self.gradient_reversal_gamma))
                        )
                        neu_lambda_branch = float(neu_lambda)
                    raw = (
                        old.float()
                        + pos_delta
                        + neu_lambda_branch * neu_delta
                        - gamma * neg_delta
                    )
                    token_filter_blend_debug[f"ttt_tri_replay_{name}_gamma"] = float(gamma)
                    token_filter_blend_debug[f"ttt_tri_replay_{name}_pos_delta_norm_mean"] = float(
                        pos_norm.mean().item()
                    )
                    token_filter_blend_debug[f"ttt_tri_replay_{name}_neu_delta_norm_mean"] = float(
                        neu_norm.mean().item()
                    )
                    token_filter_blend_debug[f"ttt_tri_replay_{name}_neg_delta_norm_mean"] = float(
                        neg_norm.mean().item()
                    )
                    controlled = renorm_like(full_pos, raw)
                    route_base = full_pos
                    base_gamma = float(
                        adaptive_split_gamma
                        if adaptive_writer_split and adaptive_split_gamma is not None
                        else self.gradient_reversal_branch_gammas.get(
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
                if (
                    adaptive_writer_energy_matched_split
                    or adaptive_writer_sc_gamma_split
                    or adaptive_writer_state_energy_matched_split
                    or adaptive_writer_tail_state_continuity_guard
                    or adaptive_writer_binary_no_long_write
                ) and energy_matched_gammas:
                    gamma_mean = float(sum(energy_matched_gammas) / len(energy_matched_gammas))
                    lambda_mean = float(sum(energy_matched_lambdas) / len(energy_matched_lambdas))
                    token_filter_blend_debug["ttt_tri_replay_adaptive_gamma"] = gamma_mean
                    token_filter_blend_debug["ttt_tri_replay_adaptive_neutral_lambda"] = lambda_mean
                    if adaptive_writer_energy_matched_split:
                        token_filter_blend_debug["ttt_tri_replay_energy_matched_gamma_mean"] = gamma_mean
                        token_filter_blend_debug["ttt_tri_replay_energy_matched_neutral_lambda_mean"] = lambda_mean
                    if adaptive_writer_sc_gamma_split:
                        token_filter_blend_debug["ttt_tri_replay_sc_gamma_gamma_mean"] = gamma_mean
                        token_filter_blend_debug["ttt_tri_replay_sc_gamma_neutral_lambda_mean"] = lambda_mean
                    if adaptive_writer_state_energy_matched_split:
                        token_filter_blend_debug["ttt_tri_replay_state_energy_gamma_mean"] = gamma_mean
                        token_filter_blend_debug["ttt_tri_replay_state_energy_neutral_lambda_mean"] = lambda_mean
                    if adaptive_writer_tail_state_continuity_guard:
                        token_filter_blend_debug["ttt_tri_replay_tail_state_gamma_mean"] = gamma_mean
                        token_filter_blend_debug["ttt_tri_replay_tail_state_neutral_lambda_mean"] = lambda_mean
                risk_cpu = risk.detach().float().cpu()
                token_filter_blend_debug["ttt_two_replay_applied"] = False
                token_filter_blend_debug["ttt_two_replay_debug_note"] = "tri_replay_path; use ttt_tri_replay_applied"
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
        y_dev = y.detach().float()
        v_dev = v.detach().float()
        res = (y_dev - v_dev).norm(dim=-1) / v_dev.norm(dim=-1).clamp_min(1e-6)
        if res.ndim == 1:
            per_tok = res
        else:
            per_tok = res.reshape(-1, res.shape[-1]).mean(dim=0)
        out = torch.zeros(int(cache_l), dtype=torch.float32, device=per_tok.device)
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
        num_frames: Optional[int] = None,
        overlap_frames: int = 0,
        layer_idx: int = -1,
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
        adaptive_writer_role = str(self.tri_replay_role_mode or "").strip().lower() in {
            "adaptive_writer",
            "adaptive_writer_fused",
            "adaptive_otsu_writer",
            "adaptive_otsu_fused",
            "no_percentage",
            "no_percentage_fused",
            "adaptive_writer_robust",
            "adaptive_writer_robust_fused",
            "adaptive_writer_robust_split",
            "robust_adaptive_writer",
            "robust_adaptive_writer_fused",
            "robust_adaptive_writer_split",
            "no_percentage_robust",
            "no_percentage_robust_fused",
            "no_percentage_robust_split",
            "adaptive_writer_conflictlite_split",
            "conflictlite_adaptive_writer_split",
            "no_percentage_conflictlite_split",
            "adaptive_writer_energy_matched_split",
            "energy_matched_adaptive_writer_split",
            "no_percentage_energy_matched_split",
            "adaptive_writer_sc_gamma_split",
            "sc_gamma_split",
            "no_percentage_sc_gamma_split",
            "adaptive_writer_sc_gamma_commit_split",
            "sc_gamma_commit_split",
            "no_percentage_sc_gamma_commit_split",
            "adaptive_writer_state_energy_matched_split",
            "state_energy_matched_split",
            "no_percentage_state_energy_matched_split",
            "adaptive_writer_state_energy_commit_split",
            "adaptive_writer_state_energy_directional_commit_split",
            "state_energy_directional_commit_split",
            "no_percentage_state_energy_directional_commit_split",
            "adaptive_writer_tail_state_continuity_guard",
            "tail_state_continuity_guard",
            "adaptive_writer_tail_state_continuity_guard_selective_commit",
            "tail_state_continuity_guard_selective_commit",
            "adaptive_writer_binary_anchor_split",
            "binary_stable_anchor_split",
            "stable_anchor_binary_split",
            "adaptive_writer_risk_veto_split",
            "risk_veto_binary_split",
            "no_long_write_risk_veto_split",
            "adaptive_writer_cluster3d_split",
            "cluster3d_adaptive_writer_split",
            "no_percentage_cluster3d_split",
        }
        tri_mode_active = gr_mode in {
            "tri_replay",
            "three_replay",
            "pos_neu_neg_replay",
            "pos_neg_neu_replay",
        }
        if gr_mode in {"", "none", "off"} or (max_gamma <= 0.0 and not (tri_mode_active and adaptive_writer_role)):
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
            "conflict_lite_selected_layers",
            "conflictlite_selected_layers",
            "selected_layer_conflict_lite",
            "conflict_lite_layer0",
            "conflictlite_layer0",
            "conflict_lite_layer8",
            "conflictlite_layer8",
            "conflict_lite_layer17",
            "conflictlite_layer17",
            "conflict_lite_layer0_sample2048",
            "conflictlite_layer0_sample2048",
            "conflict_lite_layer17_sample2048",
            "conflictlite_layer17_sample2048",
        }:
            residual_risk = self._ttt_layer_residual_risk(lc, cache_l)
            if residual_risk is None:
                debug["ttt_gradient_reversal_risk_source_missing_residual"] = True
                return None, debug

        if source in {"ttt_residual", "residual", "ttt_self_residual", "self_residual"}:
            risk = residual_risk
        elif source in {
            "v11_projection",
            "projection",
            "v11_gt_projection",
            "gt_projection",
            "v11_gt_scale_projection",
            "gt_scale_projection",
            "scale_projection",
            "oracle_scale_projection",
        }:
            risk, projection_debug = self._ttt_layer_v11_gt_scale_projection_risk(
                lc,
                cache_l=cache_l,
                prior_flat=prior_flat,
            )
            debug.update(projection_debug)
            if risk is None:
                debug["ttt_gradient_reversal_risk_source_missing_v11_projection"] = True
                return None, debug
        elif source in {
            "v19_scale_state",
            "scale_state",
            "online_scale_state",
            "nogt_scale_state",
            "trajectory_scale_state",
        }:
            risk, scale_debug = self._ttt_layer_v19_scale_state_projection_risk(
                lc,
                cache_l=cache_l,
                prior_flat=prior_flat,
                risk_tok=risk_tok,
                token_type=token_type,
                num_frames=num_frames,
                overlap_frames=overlap_frames,
            )
            debug.update(scale_debug)
            if risk is None:
                debug["ttt_gradient_reversal_risk_source_missing_v19_scale_state"] = True
                return None, debug
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
        elif source in {
            "conflict_lite_selected_layers",
            "conflictlite_selected_layers",
            "selected_layer_conflict_lite",
            "conflict_lite_layer0",
            "conflictlite_layer0",
            "conflict_lite_layer8",
            "conflictlite_layer8",
            "conflict_lite_layer17",
            "conflictlite_layer17",
            "conflict_lite_layer0_sample2048",
            "conflictlite_layer0_sample2048",
            "conflict_lite_layer17_sample2048",
            "conflictlite_layer17_sample2048",
        }:
            sample_tokens = 0
            if source in {"conflict_lite_layer0", "conflictlite_layer0"}:
                selected_layers = {0}
            elif source in {"conflict_lite_layer0_sample2048", "conflictlite_layer0_sample2048"}:
                selected_layers = {0}
                sample_tokens = 2048
            elif source in {"conflict_lite_layer8", "conflictlite_layer8"}:
                selected_layers = {8}
            elif source in {"conflict_lite_layer17", "conflictlite_layer17"}:
                selected_layers = {17}
            elif source in {"conflict_lite_layer17_sample2048", "conflictlite_layer17_sample2048"}:
                selected_layers = {17}
                sample_tokens = 2048
            else:
                selected_layers = {0, 8, 17}
            selected = int(layer_idx) in selected_layers
            debug["ttt_conflictlite_selected_layers"] = sorted(selected_layers)
            debug["ttt_conflictlite_layer_idx"] = int(layer_idx)
            debug["ttt_conflictlite_selected_layer"] = bool(selected)
            debug["ttt_conflictlite_sample_tokens"] = int(sample_tokens)
            if selected:
                conflict_risk, conflict_debug = self._ttt_layer_w0_update_risk(
                    lc,
                    cache_l=cache_l,
                    prior_flat=prior_flat,
                    mode="update_conflict_energy",
                    sample_tokens=sample_tokens,
                )
                debug.update(conflict_debug)
                if conflict_risk is not None:
                    risk = conflict_risk
                else:
                    risk = residual_risk
                    debug["ttt_conflictlite_fallback"] = "residual"
            elif risk_tok is not None:
                ext, align_debug = self._align_prior_to_replay_tokens(
                    risk_tok,
                    token_type=token_type,
                    cache_l=cache_l,
                )
                debug.update({
                    "ttt_gradient_reversal_risk_alignment_mode": align_debug.get("ttt_prior_alignment_mode"),
                    "ttt_gradient_reversal_risk_alignment_full_tokens": align_debug.get("ttt_prior_alignment_full_tokens"),
                    "ttt_conflictlite_fallback": "residual_x_dg_nonselected_layer",
                })
                risk = self._normalize01_vec(residual_risk * ext.detach().float().reshape(-1).clamp(0.0, 1.0))
            else:
                risk = residual_risk
                debug["ttt_conflictlite_fallback"] = "residual_nonselected_layer"
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
        prior_vec = prior_flat.detach().to(device=risk.device, dtype=torch.float32).reshape(-1)
        debug.update({
            "ttt_gradient_reversal_risk_source_applied": True,
            "ttt_gradient_reversal_risk_source_mean": float(risk.mean().item()),
            "ttt_gradient_reversal_risk_source_p90": float(torch.quantile(risk, 0.90).item()),
            "ttt_gradient_reversal_risk_source_corr_prior": self._corr_1d(risk, prior_vec),
            "_ttt_gradient_reversal_risk_source_vector_count": int(risk.numel()),
        })
        if not adaptive_writer_role:
            # Legacy diagnostics can consume this vector for token-level
            # causal conditions.  Adaptive writer full runs skip it because
            # the writer already receives `risk` directly and the CPU clone +
            # log printing dominates runtime for 40k-token chunks.
            debug["_ttt_gradient_reversal_risk_source_vector"] = risk.detach().cpu().float().clone()
        return risk, debug

    def _ttt_layer_w0_update_risk(
        self,
        lc: TTTLayerCache,
        *,
        cache_l: int,
        prior_flat: torch.Tensor,
        mode: str,
        sample_tokens: int = 0,
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

            device = k.device
            kf = k.detach().to(device=device, dtype=torch.float32)
            vf = v.detach().to(device=device, dtype=torch.float32)
            lr = lr0.detach().to(device=device, dtype=torch.float32)
            w0f = w0.detach().to(device=device, dtype=torch.float32)
            w1f = w1.detach().to(device=device, dtype=torch.float32)
            w2f = w2.detach().to(device=device, dtype=torch.float32)
            l = min(int(kf.shape[1]), int(cache_l), int(prior_flat.numel()))
            if l <= 0:
                return None, debug
            sample_count = int(sample_tokens or 0)
            sample_idx: Optional[torch.Tensor] = None
            if sample_count > 0 and l > sample_count:
                sample_idx = torch.linspace(0, l - 1, steps=sample_count, device=device).round().long().unique()
                l_eff = int(sample_idx.numel())
                kf = kf.index_select(1, sample_idx)
                vf = vf.index_select(1, sample_idx)
                lr = lr.index_select(1, sample_idx)
                p_base = prior_flat.detach().to(device=device, dtype=torch.float32).reshape(-1)[:l]
                p = p_base.index_select(0, sample_idx).view(1, l_eff, 1)
            else:
                kf = kf[:, :l, :]
                vf = vf[:, :l, :]
                lr = lr[:, :l, :]
                p = prior_flat.detach().to(device=device, dtype=torch.float32).reshape(-1)[:l].view(1, l, 1)
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
            out = torch.zeros(int(cache_l), dtype=torch.float32, device=device)
            if sample_idx is not None:
                full_pos = torch.arange(l, dtype=torch.float32, device=device)
                nearest = torch.clamp(
                    (full_pos / max(float(l - 1), 1.0) * max(int(per_tok.numel()) - 1, 0)).round().long(),
                    min=0,
                    max=max(int(per_tok.numel()) - 1, 0),
                )
                out[:l] = per_tok.detach().float().index_select(0, nearest)
            else:
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
                "ttt_update_conflict_sample_tokens_requested": int(sample_tokens or 0),
                "ttt_update_conflict_sample_tokens_used": int(per_tok.numel()),
                "ttt_update_conflict_sampled": bool(sample_idx is not None),
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

    def _v19_scale_state_carrier_mask(
        self,
        *,
        cache_l: int,
        prior_flat: torch.Tensor,
        risk_tok: Optional[torch.Tensor],
        token_type: Optional[torch.Tensor],
        num_frames: Optional[int],
        overlap_frames: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        carrier = str(self.scale_state_carrier or "all").strip().lower()
        mask = torch.ones(int(cache_l), dtype=torch.bool)
        debug: Dict[str, Any] = {
            "v19_scale_state_carrier": carrier,
            "v19_scale_state_carrier_valid": True,
        }
        if carrier in {"", "all", "full", "token"}:
            debug["v19_scale_state_carrier_mass"] = 1.0
            return mask, debug

        if carrier in {"special", "special_token", "special_tokens", "register", "registers"}:
            if token_type is None:
                debug["v19_scale_state_carrier_valid"] = False
                debug["v19_scale_state_carrier_skip"] = "missing_token_type"
                return torch.zeros(int(cache_l), dtype=torch.bool), debug
            tt = token_type.detach().cpu().long().reshape(-1)
            if tt.numel() == cache_l:
                mask = tt != TOKEN_TYPE_PATCH
            else:
                debug["v19_scale_state_carrier_valid"] = False
                debug["v19_scale_state_carrier_skip"] = "token_type_not_replay_aligned"
                return torch.zeros(int(cache_l), dtype=torch.bool), debug
        elif carrier in {"structure_lowdg", "lowdg", "low_dg", "static_lowdg", "structure"}:
            if risk_tok is not None:
                aligned, align_debug = self._align_prior_to_replay_tokens(
                    risk_tok,
                    token_type=token_type,
                    cache_l=cache_l,
                )
                debug["v19_scale_state_carrier_risk_alignment"] = align_debug.get("ttt_prior_alignment_mode")
                risk = aligned.detach().float().reshape(-1)
            else:
                risk = (1.0 - prior_flat.detach().float().reshape(-1)).clamp(0.0, 1.0)
                debug["v19_scale_state_carrier_risk_alignment"] = "fallback_1_minus_prior"
            if risk.numel() != cache_l:
                tmp = torch.ones(int(cache_l), dtype=torch.float32)
                n = min(int(risk.numel()), int(cache_l))
                if n > 0:
                    tmp[:n] = risk[:n]
                risk = tmp
            thr = torch.quantile(risk, 0.20)
            mask = risk <= thr
            debug["v19_scale_state_carrier_risk_q20"] = float(thr.item())
        elif carrier in {
            "overlap_static",
            "overlap_static_anchor",
            "static_anchor_overlap",
            "overlap_anchor",
        }:
            scope_mask, scope_debug = self._replay_token_filter_scope_mask(
                cache_l=int(cache_l),
                num_frames=num_frames,
                overlap_frames=overlap_frames,
                scope="both_overlap",
            )
            debug.update({
                "v19_scale_state_carrier_scope_valid": scope_debug.get("ttt_replay_token_filter_scope_valid"),
                "v19_scale_state_carrier_scope_mass": scope_debug.get("ttt_replay_token_filter_scope_mass"),
            })
            if not bool(scope_debug.get("ttt_replay_token_filter_scope_valid", True)):
                debug["v19_scale_state_carrier_valid"] = False
                debug["v19_scale_state_carrier_skip"] = "invalid_overlap_scope"
                return torch.zeros(int(cache_l), dtype=torch.bool), debug
            if risk_tok is not None:
                aligned, align_debug = self._align_prior_to_replay_tokens(
                    risk_tok,
                    token_type=token_type,
                    cache_l=cache_l,
                )
                debug["v19_scale_state_carrier_risk_alignment"] = align_debug.get("ttt_prior_alignment_mode")
                risk = aligned.detach().float().reshape(-1)
            else:
                risk = (1.0 - prior_flat.detach().float().reshape(-1)).clamp(0.0, 1.0)
                debug["v19_scale_state_carrier_risk_alignment"] = "fallback_1_minus_prior"
            if risk.numel() != cache_l:
                tmp = torch.ones(int(cache_l), dtype=torch.float32)
                n = min(int(risk.numel()), int(cache_l))
                if n > 0:
                    tmp[:n] = risk[:n]
                risk = tmp
            scoped_risk = risk[scope_mask]
            if scoped_risk.numel() == 0:
                debug["v19_scale_state_carrier_valid"] = False
                debug["v19_scale_state_carrier_skip"] = "empty_overlap_scope"
                return torch.zeros(int(cache_l), dtype=torch.bool), debug
            thr = torch.quantile(scoped_risk, 0.35)
            mask = scope_mask & (risk <= thr)
            debug["v19_scale_state_carrier_risk_q35_scoped"] = float(thr.item())
        else:
            raise ValueError(f"Unsupported v19 scale-state carrier: {self.scale_state_carrier}")

        debug["v19_scale_state_carrier_tokens"] = int(mask.sum().item())
        debug["v19_scale_state_carrier_mass"] = float(mask.float().mean().item()) if mask.numel() else 0.0
        return mask, debug

    def _ttt_layer_v19_scale_state_projection_risk(
        self,
        lc: TTTLayerCache,
        *,
        cache_l: int,
        prior_flat: torch.Tensor,
        risk_tok: Optional[torch.Tensor],
        token_type: Optional[torch.Tensor],
        num_frames: Optional[int],
        overlap_frames: int,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Route replay roles from a no-GT online trajectory-scale proxy."""
        mode_text = str(getattr(self, "scale_state_mode", "none") or "none").strip().lower()
        active = bool(getattr(self, "scale_state_active", False))
        scale_log = float(getattr(self, "scale_state_log_ratio", 0.0) or 0.0)
        alpha = max(float(getattr(self, "scale_state_alpha", 0.0) or 0.0), 0.0)
        debug: Dict[str, Any] = {
            "v19_scale_state_mode": mode_text,
            "v19_scale_state_proxy": str(getattr(self, "scale_state_proxy", "")),
            "v19_scale_state_active": active,
            "v19_scale_state_chunk_idx": int(getattr(self, "scale_state_chunk_idx", -1)),
            "v19_scale_state_log_ratio": scale_log,
            "v19_scale_state_alpha": alpha,
            "v19_scale_state_reason": str(getattr(self, "scale_state_reason", "")),
            "v19_scale_state_risk_method": "raw_w0_token_outer_mean_x_nogt_scale_proxy",
            "v19_scale_state_risk_applied": False,
            "v19_scale_state_branch_mask": list(self.scale_state_branch_mask),
            "v19_scale_state_chunks": [int(x) for x in self.scale_state_chunks],
            "v19_scale_state_sample_tokens_requested": int(getattr(self, "scale_state_sample_tokens", 0) or 0),
            "v19_scale_state_sampled": False,
            "v19_scale_state_sample_tokens_used": 0,
        }
        if mode_text in {"", "none", "off"}:
            debug["v19_scale_state_risk_skip"] = "mode_off"
            return None, debug
        if not active:
            debug["v19_scale_state_risk_skip"] = "inactive"
            return None, debug
        if alpha <= 0.0:
            debug["v19_scale_state_risk_skip"] = "alpha_zero"
            return None, debug
        if abs(scale_log) <= 1e-8:
            debug["v19_scale_state_risk_skip"] = "proxy_zero"
            return None, debug

        carrier_mask, carrier_debug = self._v19_scale_state_carrier_mask(
            cache_l=cache_l,
            prior_flat=prior_flat,
            risk_tok=risk_tok,
            token_type=token_type,
            num_frames=num_frames,
            overlap_frames=overlap_frames,
        )
        debug.update(carrier_debug)
        if carrier_mask.numel() != int(cache_l) or int(carrier_mask.sum().item()) == 0:
            debug["v19_scale_state_risk_skip"] = "empty_carrier"
            return None, debug

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
            l = min(int(k.shape[1]), int(cache_l), int(prior_flat.numel()))
            if l <= 0:
                return None, debug

            device = k.device
            sample_tokens = int(getattr(self, "scale_state_sample_tokens", 0) or 0)
            sample_idx: Optional[torch.Tensor] = None
            if sample_tokens > 0 and l > sample_tokens:
                sample_idx = torch.linspace(
                    0,
                    l - 1,
                    steps=sample_tokens,
                    device=device,
                    dtype=torch.float32,
                ).round().long().unique(sorted=True)

            if sample_idx is not None:
                kf = k.detach().to(device=device, dtype=torch.float32).index_select(1, sample_idx)
                vf = v.detach().to(device=device, dtype=torch.float32).index_select(1, sample_idx)
                lr = lr0.detach().to(device=device, dtype=torch.float32).index_select(1, sample_idx)
                p = prior_flat.detach().to(device=device, dtype=torch.float32).reshape(-1)[:l].index_select(0, sample_idx).view(1, -1, 1)
            else:
                kf = k.detach().to(device=device, dtype=torch.float32)[:, :l, :]
                vf = v.detach().to(device=device, dtype=torch.float32)[:, :l, :]
                lr = lr0.detach().to(device=device, dtype=torch.float32)[:, :l, :]
                p = prior_flat.detach().to(device=device, dtype=torch.float32).reshape(-1)[:l].view(1, l, 1)
            w0f = w0.detach().to(device=device, dtype=torch.float32)
            w1f = w1.detach().to(device=device, dtype=torch.float32)
            w2f = w2.detach().to(device=device, dtype=torch.float32)
            lr_eff = lr * p

            gate = torch.bmm(kf, w0f)
            hidden_before_mul = torch.bmm(kf, w2f)
            dhidden = torch.bmm(vf, w1f.transpose(1, 2))
            dgate = dhidden * hidden_before_mul
            sigma = torch.sigmoid(gate)
            dgate_before_act = dgate * sigma * (1.0 + gate * (1.0 - sigma))

            token_outer_mean = (
                lr_eff.squeeze(-1)
                * kf.mean(dim=-1)
                * dgate_before_act.mean(dim=-1)
            )
            scale_sign = 1.0 if scale_log >= 0.0 else -1.0
            orientation = -1.0 if any(tag in mode_text for tag in ("inverse", "flip")) else 1.0
            signed_projection = token_outer_mean * float(scale_sign * orientation)
            per_tok_projection = signed_projection.mean(dim=0)

            k_norm = kf.norm(dim=-1)
            d_norm = dgate_before_act.norm(dim=-1)
            energy = (lr_eff.squeeze(-1).abs() * k_norm * d_norm).detach().float()
            per_tok_energy = energy.mean(dim=0)
            energy_risk = self._normalize01_vec(per_tok_energy)

            proj_abs = signed_projection.detach().float().reshape(-1).abs()
            proj_scale = torch.quantile(proj_abs, 0.90).clamp_min(1e-12) if proj_abs.numel() else torch.tensor(1.0)
            projection_norm = (per_tok_projection / proj_scale).clamp(-1.0, 1.0)
            directional_risk = ((projection_norm + 1.0) * 0.5).clamp(0.0, 1.0)
            raw_risk = (directional_risk * (0.25 + 0.75 * energy_risk)).clamp(0.0, 1.0)
            strength = max(alpha * abs(scale_log), 0.0)
            risk = (0.5 + strength * (raw_risk - 0.5)).clamp(0.0, 1.0)
            carrier_full = carrier_mask[:l].detach().to(device=device).bool()
            carrier = carrier_full.index_select(0, sample_idx) if sample_idx is not None else carrier_full
            risk = torch.where(carrier, risk, torch.full_like(risk, 0.5))

            out = torch.full((int(cache_l),), 0.5, dtype=torch.float32, device=device)
            if sample_idx is not None:
                full_positions = torch.arange(l, device=device)
                nearest = torch.bucketize(full_positions, sample_idx)
                nearest = nearest.clamp(max=sample_idx.numel() - 1)
                prev_nearest = (nearest - 1).clamp(min=0)
                dist_next = (sample_idx.index_select(0, nearest) - full_positions).abs()
                dist_prev = (full_positions - sample_idx.index_select(0, prev_nearest)).abs()
                chosen = torch.where(dist_prev <= dist_next, prev_nearest, nearest)
                out[:l] = risk.detach().float().index_select(0, chosen)
            else:
                out[:l] = risk.detach().float()
            proj_flat = signed_projection.detach().float().reshape(-1)
            risk_flat = risk.detach().float().reshape(-1)
            energy_flat = energy.detach().float().reshape(-1)
            debug.update({
                "v19_scale_state_risk_applied": True,
                "v19_scale_state_scale_sign": float(scale_sign),
                "v19_scale_state_orientation": float(orientation),
                "v19_scale_state_token_count": int(l),
                "v19_scale_state_carrier_count_aligned": int(carrier.sum().item()),
                "v19_scale_state_sampled": bool(sample_idx is not None),
                "v19_scale_state_sample_tokens_used": int(sample_idx.numel()) if sample_idx is not None else int(l),
                "v19_scale_state_projection_mean": float(proj_flat.mean().item()) if proj_flat.numel() else 0.0,
                "v19_scale_state_projection_p10": float(torch.quantile(proj_flat, 0.10).item()) if proj_flat.numel() else 0.0,
                "v19_scale_state_projection_p90": float(torch.quantile(proj_flat, 0.90).item()) if proj_flat.numel() else 0.0,
                "v19_scale_state_projection_scale_p90abs": float(proj_scale.item()),
                "v19_scale_state_helpful_token_mass": float((per_tok_projection < 0.0).float().mean().item()),
                "v19_scale_state_harmful_token_mass": float((per_tok_projection > 0.0).float().mean().item()),
                "v19_scale_state_energy_mean": float(energy_flat.mean().item()) if energy_flat.numel() else 0.0,
                "v19_scale_state_energy_p90": float(torch.quantile(energy_flat, 0.90).item()) if energy_flat.numel() else 0.0,
                "v19_scale_state_risk_mean": float(risk_flat.mean().item()) if risk_flat.numel() else 0.0,
                "v19_scale_state_risk_p90": float(torch.quantile(risk_flat, 0.90).item()) if risk_flat.numel() else 0.0,
            })
            return out, debug
        except RuntimeError as exc:
            debug["v19_scale_state_risk_error"] = str(exc)
            return None, debug

    def _ttt_layer_v11_gt_scale_projection_risk(
        self,
        lc: TTTLayerCache,
        *,
        cache_l: int,
        prior_flat: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Route tri-replay roles by an oracle window-scale projection proxy.

        v11's oracle action keeps the trajectory untouched, but lets the TTT
        write controller use the current chunk's GT scale residual to decide
        which replay-token contributions look corrective vs harmful.  The
        first implementation uses a cheap raw-w0 contribution scalar as the
        TTT-space projection proxy: the signed mean of each token's pre-muon
        outer-product update, oriented by the GT step-scale residual.
        """
        mode_text = str(getattr(self, "v11_projection_action_mode", "none") or "none").strip().lower()
        active = bool(getattr(self, "v11_projection_action_active", False))
        scale_log = float(getattr(self, "v11_projection_scale_log_ratio", 0.0) or 0.0)
        deadband = max(float(getattr(self, "v11_projection_action_deadband", 0.0) or 0.0), 0.0)
        debug: Dict[str, Any] = {
            "v11_projection_action_mode": mode_text,
            "v11_projection_action_active": active,
            "v11_projection_action_chunk_idx": int(getattr(self, "v11_projection_chunk_idx", -1)),
            "v11_projection_scale_log_ratio": scale_log,
            "v11_projection_action_strength": float(
                getattr(self, "v11_projection_action_strength", 1.0) or 1.0
            ),
            "v11_projection_action_deadband": deadband,
            "v11_projection_action_reason": str(getattr(self, "v11_projection_action_reason", "")),
            "v11_projection_risk_method": "raw_w0_token_outer_mean_x_gt_scale_residual",
            "v11_projection_risk_applied": False,
        }
        if not active or mode_text in {"", "none", "off"}:
            debug["v11_projection_risk_skip"] = "action_inactive"
            return None, debug
        if abs(scale_log) <= deadband:
            debug["v11_projection_risk_skip"] = "scale_deadband"
            return None, debug

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
            l = min(int(k.shape[1]), int(cache_l), int(prior_flat.numel()))
            if l <= 0:
                return None, debug

            kf = k.detach().cpu().float()[:, :l, :]
            vf = v.detach().cpu().float()[:, :l, :]
            lr = lr0.detach().cpu().float()[:, :l, :]
            w0f = w0.detach().cpu().float()
            w1f = w1.detach().cpu().float()
            w2f = w2.detach().cpu().float()
            p = prior_flat.detach().cpu().float().reshape(-1)[:l].view(1, l, 1)
            lr_eff = lr * p

            gate = torch.bmm(kf, w0f)
            hidden_before_mul = torch.bmm(kf, w2f)
            dhidden = torch.bmm(vf, w1f.transpose(1, 2))
            dgate = dhidden * hidden_before_mul
            sigma = torch.sigmoid(gate)
            dgate_before_act = dgate * sigma * (1.0 + gate * (1.0 - sigma))

            token_outer_mean = (
                lr_eff.squeeze(-1)
                * kf.mean(dim=-1)
                * dgate_before_act.mean(dim=-1)
            )
            scale_sign = 1.0 if scale_log >= 0.0 else -1.0
            orientation = -1.0 if any(tag in mode_text for tag in ("inverse", "flip")) else 1.0
            signed_projection = token_outer_mean * float(scale_sign * orientation)
            per_tok_projection = signed_projection.mean(dim=0)

            k_norm = kf.norm(dim=-1)
            d_norm = dgate_before_act.norm(dim=-1)
            energy = (lr_eff.squeeze(-1).abs() * k_norm * d_norm).detach().float()
            per_tok_energy = energy.mean(dim=0)
            energy_risk = self._normalize01_vec(per_tok_energy)

            proj_abs = signed_projection.detach().float().reshape(-1).abs()
            proj_scale = torch.quantile(proj_abs, 0.90).clamp_min(1e-12) if proj_abs.numel() else torch.tensor(1.0)
            projection_norm = (per_tok_projection / proj_scale).clamp(-1.0, 1.0)
            # Positive means same-sign with the drift residual proxy and is
            # routed as harmful; negative is routed as corrective.
            directional_risk = ((projection_norm + 1.0) * 0.5).clamp(0.0, 1.0)
            risk = (directional_risk * (0.25 + 0.75 * energy_risk)).clamp(0.0, 1.0)
            strength = max(float(getattr(self, "v11_projection_action_strength", 1.0) or 1.0), 0.0)
            if strength != 1.0:
                risk = (0.5 + strength * (risk - 0.5)).clamp(0.0, 1.0)

            out = torch.zeros(int(cache_l), dtype=torch.float32)
            out[:l] = risk.detach().float()
            proj_flat = signed_projection.detach().float().reshape(-1)
            risk_flat = risk.detach().float().reshape(-1)
            energy_flat = energy.detach().float().reshape(-1)
            debug.update({
                "v11_projection_risk_applied": True,
                "v11_projection_scale_sign": float(scale_sign),
                "v11_projection_orientation": float(orientation),
                "v11_projection_token_count": int(l),
                "v11_projection_projection_mean": float(proj_flat.mean().item()) if proj_flat.numel() else 0.0,
                "v11_projection_projection_p10": float(torch.quantile(proj_flat, 0.10).item()) if proj_flat.numel() else 0.0,
                "v11_projection_projection_p90": float(torch.quantile(proj_flat, 0.90).item()) if proj_flat.numel() else 0.0,
                "v11_projection_projection_scale_p90abs": float(proj_scale.item()),
                "v11_projection_helpful_token_mass": float((per_tok_projection < 0.0).float().mean().item()),
                "v11_projection_harmful_token_mass": float((per_tok_projection > 0.0).float().mean().item()),
                "v11_projection_energy_mean": float(energy_flat.mean().item()) if energy_flat.numel() else 0.0,
                "v11_projection_energy_p90": float(torch.quantile(energy_flat, 0.90).item()) if energy_flat.numel() else 0.0,
                "v11_projection_risk_mean": float(risk_flat.mean().item()) if risk_flat.numel() else 0.0,
                "v11_projection_risk_p90": float(torch.quantile(risk_flat, 0.90).item()) if risk_flat.numel() else 0.0,
            })
            return out, debug
        except RuntimeError as exc:
            debug["v11_projection_risk_error"] = str(exc)
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
        adaptive_writer_role = str(self.tri_replay_role_mode or "").strip().lower() in {
            "adaptive_writer",
            "adaptive_writer_fused",
            "adaptive_otsu_writer",
            "adaptive_otsu_fused",
            "no_percentage",
            "no_percentage_fused",
            "adaptive_writer_robust",
            "adaptive_writer_robust_fused",
            "adaptive_writer_robust_split",
            "robust_adaptive_writer",
            "robust_adaptive_writer_fused",
            "robust_adaptive_writer_split",
            "no_percentage_robust",
            "no_percentage_robust_fused",
            "no_percentage_robust_split",
            "adaptive_writer_conflictlite_split",
            "conflictlite_adaptive_writer_split",
            "no_percentage_conflictlite_split",
            "adaptive_writer_energy_matched_split",
            "energy_matched_adaptive_writer_split",
            "no_percentage_energy_matched_split",
            "adaptive_writer_sc_gamma_split",
            "sc_gamma_split",
            "no_percentage_sc_gamma_split",
            "adaptive_writer_sc_gamma_commit_split",
            "sc_gamma_commit_split",
            "no_percentage_sc_gamma_commit_split",
            "adaptive_writer_state_energy_matched_split",
            "state_energy_matched_split",
            "no_percentage_state_energy_matched_split",
            "adaptive_writer_state_energy_commit_split",
            "adaptive_writer_state_energy_directional_commit_split",
            "state_energy_directional_commit_split",
            "no_percentage_state_energy_directional_commit_split",
            "adaptive_writer_tail_state_continuity_guard",
            "tail_state_continuity_guard",
            "adaptive_writer_tail_state_continuity_guard_selective_commit",
            "tail_state_continuity_guard_selective_commit",
            "adaptive_writer_binary_anchor_split",
            "binary_stable_anchor_split",
            "stable_anchor_binary_split",
            "adaptive_writer_risk_veto_split",
            "risk_veto_binary_split",
            "no_long_write_risk_veto_split",
            "adaptive_writer_cluster3d_split",
            "cluster3d_adaptive_writer_split",
            "no_percentage_cluster3d_split",
        }
        tri_mode_active = mode in {
            "tri_replay",
            "three_replay",
            "pos_neu_neg_replay",
            "pos_neg_neu_replay",
        }
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
        if (
            mode in {"", "none", "off"}
            or prior_flat.numel() == 0
            or (max_gamma <= 0.0 and not (tri_mode_active and adaptive_writer_role))
        ):
            return (token_prior0, token_prior1, token_prior2), debug
        active = tuple(
            int(i)
            for i, g in sorted(branch_gammas.items())
            if (g > 0.0 or (tri_mode_active and adaptive_writer_role))
            and 0 <= int(i) <= 2
            and branch_enabled[int(i)]
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
            risk_source_text = str(self.gradient_reversal_risk_source or "prior").strip().lower()
            if risk_source_text in {
                "v11_projection",
                "projection",
                "v11_gt_projection",
                "gt_projection",
                "v11_gt_scale_projection",
                "gt_scale_projection",
                "scale_projection",
                "oracle_scale_projection",
                "v19_scale_state",
                "scale_state",
                "online_scale_state",
                "nogt_scale_state",
                "trajectory_scale_state",
            }:
                debug["ttt_gradient_reversal_projection_skip"] = "missing_projection_risk"
                return (token_prior0, token_prior1, token_prior2), debug
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
        overlap_frames: int = 0,
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
        overlap_pseudo_modes = {
            "overlap_pseudo_replay_v",
            "overlap_pseudo_v",
            "tail_head_overlap_v",
            "overlap_pseudo_replay_kv",
            "overlap_pseudo_kv",
            "tail_head_overlap_kv",
        }
        if mode not in (global_center_modes | frame_static_modes | overlap_pseudo_modes):
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

        def _overlap_pseudo_replay(
            x: torch.Tensor,
            *,
            target_name: str,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            n_frames = int(num_frames or 0)
            ov = max(int(overlap_frames), 0)
            n_tokens = int(x.shape[1])
            extra: Dict[str, Any] = {
                "ttt_overlap_pseudo_replay_target": target_name,
                "ttt_overlap_pseudo_replay_weight": float(min(max(rho, 0.0), 1.0)),
                "ttt_overlap_pseudo_replay_overlap_frames": int(ov),
                "ttt_overlap_pseudo_replay_applied": False,
            }
            if n_frames <= 0 or ov <= 0 or n_tokens <= 0 or n_tokens % n_frames != 0:
                extra["ttt_overlap_pseudo_replay_invalid"] = True
                return x, extra
            if n_frames <= ov:
                extra["ttt_overlap_pseudo_replay_invalid"] = True
                extra["ttt_overlap_pseudo_replay_skip"] = "not_enough_frames"
                return x, extra
            tokens_per_frame = n_tokens // n_frames
            patch_mask = _patch_mask_for_replay()
            if patch_mask.numel() != n_tokens:
                patch_mask = torch.ones(n_tokens, dtype=torch.bool)
            local_patch = patch_mask[:tokens_per_frame].clone()
            if not bool(local_patch.any().item()):
                local_patch = torch.ones(tokens_per_frame, dtype=torch.bool)
            local_idx = torch.nonzero(local_patch, as_tuple=False).reshape(-1)
            if local_idx.numel() == 0:
                extra["ttt_overlap_pseudo_replay_skip"] = "empty_local_tokens"
                return x, extra
            head_indices: List[torch.Tensor] = []
            tail_indices: List[torch.Tensor] = []
            for fi in range(ov):
                head_base = fi * tokens_per_frame
                tail_base = (n_frames - ov + fi) * tokens_per_frame
                head_indices.append(local_idx + head_base)
                tail_indices.append(local_idx + tail_base)
            head_idx = torch.cat(head_indices, dim=0).to(device=x.device, dtype=torch.long)
            tail_idx = torch.cat(tail_indices, dim=0).to(device=x.device, dtype=torch.long)
            if head_idx.numel() == 0 or tail_idx.numel() == 0:
                extra["ttt_overlap_pseudo_replay_skip"] = "empty_pair_tokens"
                return x, extra
            weight = min(max(float(rho), 0.0), 1.0)
            x_f = x.float().clone()
            head_vals = x_f.index_select(1, head_idx)
            tail_vals = x_f.index_select(1, tail_idx)
            mixed = tail_vals + weight * (head_vals - tail_vals)
            x_f[:, tail_idx, :] = mixed
            delta = (mixed - tail_vals).detach().float()
            extra.update({
                "ttt_overlap_pseudo_replay_applied": True,
                "ttt_overlap_pseudo_replay_invalid": False,
                "ttt_overlap_pseudo_replay_tokens": int(tail_idx.numel()),
                "ttt_overlap_pseudo_replay_tokens_per_frame": int(tokens_per_frame),
                "ttt_overlap_pseudo_replay_patch_tokens_per_frame": int(local_idx.numel()),
                "ttt_overlap_pseudo_replay_weight_mean": float(weight),
                "ttt_overlap_pseudo_replay_delta_norm_mean": float(delta.norm(dim=-1).mean().item())
                if delta.numel() else 0.0,
            })
            return x_f.to(dtype=x.dtype), extra

        if mode in overlap_pseudo_modes:
            targets: List[str] = []
            extra: Dict[str, Any] = {}
            if mode in {"overlap_pseudo_replay_kv", "overlap_pseudo_kv", "tail_head_overlap_kv"}:
                k, extra_k = _overlap_pseudo_replay(k, target_name="k")
                v, extra_v = _overlap_pseudo_replay(v, target_name="v")
                targets.extend(["k", "v"])
                extra.update({f"k_{key}": val for key, val in extra_k.items()})
                extra.update({f"v_{key}": val for key, val in extra_v.items()})
                applied = bool(extra_k.get("ttt_overlap_pseudo_replay_applied", False)) or bool(
                    extra_v.get("ttt_overlap_pseudo_replay_applied", False)
                )
                tokens = int(extra_v.get("ttt_overlap_pseudo_replay_tokens", 0) or 0)
                weight_mean = float(extra_v.get("ttt_overlap_pseudo_replay_weight_mean", min(max(rho, 0.0), 1.0)) or 0.0)
            else:
                v, extra_v = _overlap_pseudo_replay(v, target_name="v")
                targets.append("v")
                extra.update(extra_v)
                applied = bool(extra_v.get("ttt_overlap_pseudo_replay_applied", False))
                tokens = int(extra_v.get("ttt_overlap_pseudo_replay_tokens", 0) or 0)
                weight_mean = float(extra_v.get("ttt_overlap_pseudo_replay_weight_mean", min(max(rho, 0.0), 1.0)) or 0.0)
            debug.update({
                "ttt_replay_feature_gate_applied": applied,
                "ttt_replay_feature_gate_targets": targets,
                "ttt_replay_feature_gate_overlap_pseudo_replay": True,
                "overlap_replay_token_count": tokens,
                "overlap_replay_weight_mean": weight_mean,
                "ttt_replay_feature_gate_frame_static": False,
            })
            debug.update(extra)
            return k, v, debug

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
        mix_chunks = tuple(self.update_native_mix_chunks)
        current_chunk = int(getattr(self, "current_chunk_idx", getattr(self, "v11_projection_chunk_idx", -1)))
        debug_info["ttt_write_native_mix_chunks"] = list(mix_chunks)
        debug_info["ttt_write_native_mix_current_chunk"] = current_chunk
        if mix_chunks and current_chunk not in mix_chunks:
            debug_info["ttt_write_native_mix_applied"] = False
            debug_info["ttt_write_native_mix_scales"] = [s0, s1, s2]
            debug_info["ttt_write_native_mix_chunk_gate_active"] = False
            return
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
        debug_info["ttt_write_native_mix_chunk_gate_active"] = True
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
            "native_distance_adaptive_ema",
            "candidate_native_distance_ema",
            "state_conditioned_commit",
            "state_energy_directional_commit",
            "directional_commit_guard",
            "state_energy_directional_commit_guard",
            "tail_state_selective_commit",
            "selective_commit_ema",
            "tail_state_continuity_selective_commit",
        }:
            raise ValueError(f"Unsupported TTT commit filter mode: {self.commit_filter_mode}")

        branch_mask = tuple(self.commit_filter_branch_mask)
        if len(branch_mask) == 0:
            debug_info.update({
                "ttt_write_commit_filter_applied": False,
                "ttt_write_commit_filter_branch_mask": [],
            })
            return

        lo = min(float(self.commit_filter_min), float(self.commit_filter_max))
        hi = max(float(self.commit_filter_min), float(self.commit_filter_max))

        if mode in {
            "state_energy_directional_commit",
            "directional_commit_guard",
            "state_energy_directional_commit_guard",
            "tail_state_selective_commit",
            "selective_commit_ema",
            "tail_state_continuity_selective_commit",
        }:
            branches = (
                ("w0", 0, w0_new, write_cache.w0_provisional, "w0_old"),
                ("w1", 1, w1_new, write_cache.w1_provisional, "w1_old"),
                ("w2", 2, w2_new, write_cache.w2_provisional, "w2_old"),
            )
            records: List[
                Tuple[
                    int,
                    str,
                    int,
                    List[Optional[torch.Tensor]],
                    torch.Tensor,
                    torch.Tensor,
                    float,
                    float,
                    float,
                    float,
                ]
            ] = []
            n_layers = len(write_cache.layer_caches)
            for li, lc in enumerate(write_cache.layer_caches):
                if not self._layer_prior_enabled(li, n_layers):
                    continue
                for name, branch_idx, values, native_list, old_attr in branches:
                    if branch_idx not in branch_mask or li >= len(values) or li >= len(native_list):
                        continue
                    candidate = values[li]
                    native = native_list[li]
                    old = getattr(lc, old_attr, None)
                    if candidate is None or native is None or old is None:
                        continue
                    native_t = native.to(device=candidate.device, dtype=candidate.dtype)
                    old_t = old.to(device=candidate.device, dtype=candidate.dtype)
                    candidate_delta = candidate.float() - old_t.float()
                    native_delta = native_t.float() - old_t.float()
                    candidate_norm = candidate_delta.norm().clamp_min(1e-12)
                    native_norm = native_delta.norm().clamp_min(1e-12)
                    den = (candidate_norm * native_norm).clamp_min(1e-12)
                    cos = float(((candidate_delta.reshape(-1) @ native_delta.reshape(-1)) / den).clamp(-1.0, 1.0).item())
                    energy_ratio = float((candidate_norm / native_norm).item())
                    records.append((
                        li,
                        name,
                        branch_idx,
                        values,
                        native_t,
                        candidate,
                        float(cos),
                        float(energy_ratio),
                        float(candidate_norm.item()),
                        float(native_norm.item()),
                    ))

            tau_c = float(self.state_energy_commit_tau_c)
            tau_c = max(min(tau_c, 0.999), -1.0)
            u_max = max(float(self.state_energy_commit_u_max), 1.000001)
            tail_selective = mode in {
                "tail_state_selective_commit",
                "selective_commit_ema",
                "tail_state_continuity_selective_commit",
            }
            risk_mean = 0.0
            risk_mad = 0.0
            risk_high_mass = 0.0
            risk_high = False
            anchor_mass = 1.0
            if tail_selective:
                if risk_tok is not None:
                    risk_flat = risk_tok.detach().float().reshape(-1)
                    risk_flat = risk_flat[torch.isfinite(risk_flat)]
                    if risk_flat.numel() > 0:
                        risk_mean = float(risk_flat.mean().item())
                        risk_med_t = torch.median(risk_flat)
                        risk_mad_t = torch.median((risk_flat - risk_med_t).abs())
                        risk_mad = float(risk_mad_t.item())
                        risk_thr = risk_med_t + risk_mad_t
                        risk_high_mass = float((risk_flat > risk_thr).float().mean().item())
                prev_risk = self.tail_commit_risk_ema.get("global")
                risk_high = bool(
                    risk_high_mass >= 0.15
                    or (prev_risk is not None and risk_mean > float(prev_risk) + 0.25 * max(risk_mad, 1e-6))
                )
                self.tail_commit_risk_ema["global"] = (
                    risk_mean if prev_risk is None else 0.75 * float(prev_risk) + 0.25 * risk_mean
                )
                if A_tok is not None:
                    a_flat = A_tok.detach().float().reshape(-1)
                    a_flat = a_flat[torch.isfinite(a_flat)]
                    if a_flat.numel() > 0:
                        a_med = torch.median(a_flat)
                        a_mad = torch.median((a_flat - a_med).abs())
                        anchor_mass = float((a_flat >= a_med + a_mad).float().mean().item())
            applied = 0
            alpha_values: List[float] = []
            cos_values: List[float] = []
            energy_ratios: List[float] = []
            for li, name, _branch_idx, values, native_t, candidate, cos, energy_ratio, cand_norm, nat_norm in records:
                if tail_selective:
                    ema_key = f"layer_{int(li)}_{name}"
                    prev_energy = self.tail_commit_energy_ema.get(ema_key)
                    envelope = float(prev_energy) if prev_energy is not None else float(energy_ratio)
                    overshoot = bool(energy_ratio > max(1.05, 1.10 * envelope))
                    low_cos = bool(cos < 0.88)
                    anchor_ok = bool(anchor_mass > 0.02)
                    risk_norm = max(0.0, min(1.0, risk_mean + risk_high_mass))
                    overshoot_norm = max(0.0, min(1.0, (energy_ratio - 1.0) / 1.25))
                    cos_norm = max(0.0, min(1.0, (0.88 - cos) / 0.88))
                    r_risk = max(0.0, min(1.0, 0.45 * overshoot_norm + 0.35 * cos_norm + 0.20 * risk_norm))
                    triggered = bool(low_cos and overshoot and risk_high and anchor_ok)
                    alpha = max(lo, min(hi, 1.0 - r_risk)) if triggered else 1.0
                    self.tail_commit_energy_ema[ema_key] = (
                        float(energy_ratio)
                        if prev_energy is None
                        else 0.75 * float(prev_energy) + 0.25 * float(energy_ratio)
                    )
                    a_cos = 1.0 - cos_norm
                    a_energy = 1.0 - overshoot_norm
                else:
                    a_cos = (cos - tau_c) / max(1.0 - tau_c, 1e-6)
                    a_cos = max(0.0, min(1.0, float(a_cos)))
                    a_energy = (u_max - energy_ratio) / max(u_max - 1.0, 1e-6)
                    a_energy = max(0.0, min(1.0, float(a_energy)))
                    alpha = max(lo, min(hi, float(a_cos * a_energy)))
                    triggered = alpha < 0.999999
                triggered = alpha < 0.999999
                if triggered:
                    values[li] = self._scale_delta_and_renorm(native_t, candidate, alpha).cpu()
                    applied += 1
                alpha_values.append(float(alpha))
                cos_values.append(float(cos))
                energy_ratios.append(float(energy_ratio))
                layer_debug = debug_info.get(f"layer_{li}")
                if isinstance(layer_debug, dict):
                    layer_debug[f"ttt_write_commit_filter_{name}_cosine"] = float(cos)
                    layer_debug[f"ttt_write_commit_filter_{name}_energy_ratio"] = float(energy_ratio)
                    layer_debug[f"ttt_write_commit_filter_{name}_a_cos"] = float(a_cos)
                    layer_debug[f"ttt_write_commit_filter_{name}_a_energy"] = float(a_energy)
                    layer_debug[f"ttt_write_commit_filter_{name}_alpha"] = float(alpha)
                    layer_debug[f"ttt_write_commit_filter_{name}_candidate_delta_norm"] = float(cand_norm)
                    layer_debug[f"ttt_write_commit_filter_{name}_native_delta_norm"] = float(nat_norm)
                    layer_debug[f"ttt_write_commit_filter_{name}_triggered"] = bool(triggered)
                    if tail_selective:
                        layer_debug[f"ttt_write_commit_filter_{name}_tail_selective"] = True
                        layer_debug[f"ttt_write_commit_filter_{name}_tail_risk_mean"] = float(risk_mean)
                        layer_debug[f"ttt_write_commit_filter_{name}_tail_risk_high_mass"] = float(risk_high_mass)
                        layer_debug[f"ttt_write_commit_filter_{name}_tail_risk_high"] = bool(risk_high)
                        layer_debug[f"ttt_write_commit_filter_{name}_tail_anchor_mass"] = float(anchor_mass)

            debug_info.update({
                "ttt_write_commit_filter_applied": bool(applied),
                "ttt_write_commit_filter_mode": mode,
                "ttt_write_commit_filter_branch_mask": list(branch_mask),
                "ttt_write_commit_filter_num_tensors": int(applied),
                "ttt_write_commit_filter_min": lo,
                "ttt_write_commit_filter_max": hi,
                "ttt_write_commit_filter_state_energy_tau_c": float(tau_c),
                "ttt_write_commit_filter_state_energy_u_max": float(u_max),
                "ttt_write_commit_filter_state_energy_cos_mean": (
                    float(sum(cos_values) / len(cos_values)) if cos_values else 0.0
                ),
                "ttt_write_commit_filter_state_energy_ratio_mean": (
                    float(sum(energy_ratios) / len(energy_ratios)) if energy_ratios else 0.0
                ),
                "ttt_write_commit_filter_scale_mean": (
                    float(sum(alpha_values) / len(alpha_values)) if alpha_values else 1.0
                ),
                "ttt_write_commit_filter_activation_rate": (
                    float(sum(1 for x in alpha_values if x < 0.999999) / len(alpha_values)) if alpha_values else 0.0
                ),
            })
            if tail_selective:
                debug_info.update({
                    "ttt_write_commit_filter_tail_selective": True,
                    "ttt_write_commit_filter_tail_risk_mean": float(risk_mean),
                    "ttt_write_commit_filter_tail_risk_mad": float(risk_mad),
                    "ttt_write_commit_filter_tail_risk_high_mass": float(risk_high_mass),
                    "ttt_write_commit_filter_tail_risk_high": bool(risk_high),
                    "ttt_write_commit_filter_tail_anchor_mass": float(anchor_mass),
                })
            return

        if mode in {
            "native_distance_adaptive_ema",
            "candidate_native_distance_ema",
            "state_conditioned_commit",
        }:
            branches = (
                ("w0", 0, w0_new, write_cache.w0_provisional, "w0_old"),
                ("w1", 1, w1_new, write_cache.w1_provisional, "w1_old"),
                ("w2", 2, w2_new, write_cache.w2_provisional, "w2_old"),
            )
            records: List[Tuple[int, str, int, List[Optional[torch.Tensor]], torch.Tensor, torch.Tensor, float, float]] = []
            q_values: List[float] = []
            c_values: List[float] = []
            n_layers = len(write_cache.layer_caches)
            for li, lc in enumerate(write_cache.layer_caches):
                if not self._layer_prior_enabled(li, n_layers):
                    continue
                for name, branch_idx, values, native_list, old_attr in branches:
                    if branch_idx not in branch_mask or li >= len(values) or li >= len(native_list):
                        continue
                    candidate = values[li]
                    native = native_list[li]
                    old = getattr(lc, old_attr, None)
                    if candidate is None or native is None or old is None:
                        continue
                    native_t = native.to(device=candidate.device, dtype=candidate.dtype)
                    old_t = old.to(device=candidate.device, dtype=candidate.dtype)
                    candidate_delta = candidate.float() - old_t.float()
                    native_delta = native_t.float() - old_t.float()
                    diff = candidate.float() - native_t.float()
                    native_norm = native_delta.norm().clamp_min(1e-12)
                    q = float((diff.norm() / native_norm).item())
                    den = (candidate_delta.norm() * native_delta.norm()).clamp_min(1e-12)
                    cos = float(((candidate_delta.reshape(-1) @ native_delta.reshape(-1)) / den).clamp(-1.0, 1.0).item())
                    c = float(max(0.0, 1.0 - cos))
                    records.append((li, name, branch_idx, values, native_t, candidate, q, c))
                    q_values.append(q)
                    c_values.append(c)

            def robust_threshold(values: List[float]) -> float:
                if not values:
                    return float("inf")
                vals = torch.tensor(values, dtype=torch.float32)
                med = torch.median(vals)
                mad = torch.median((vals - med).abs())
                if float(mad.item()) <= 1e-8:
                    return float((med + vals.std(unbiased=False)).item())
                return float((med + mad).item())

            q_threshold = robust_threshold(q_values)
            c_threshold = robust_threshold(c_values)
            applied = 0
            alpha_values: List[float] = []
            trigger_values: List[bool] = []
            for li, name, _branch_idx, values, native_t, candidate, q, c in records:
                triggered = bool(q > q_threshold or c > c_threshold)
                alpha = 1.0
                if triggered:
                    alpha = 1.0 - (q / (q + 1.0))
                    alpha = max(lo, min(hi, float(alpha)))
                    values[li] = self._scale_delta_and_renorm(native_t, candidate, alpha).cpu()
                    applied += 1
                alpha_values.append(float(alpha))
                trigger_values.append(triggered)
                layer_debug = debug_info.get(f"layer_{li}")
                if isinstance(layer_debug, dict):
                    layer_debug[f"ttt_write_commit_filter_{name}_q"] = float(q)
                    layer_debug[f"ttt_write_commit_filter_{name}_cosine_distance"] = float(c)
                    layer_debug[f"ttt_write_commit_filter_{name}_alpha"] = float(alpha)
                    layer_debug[f"ttt_write_commit_filter_{name}_triggered"] = bool(triggered)

            debug_info.update({
                "ttt_write_commit_filter_applied": bool(applied),
                "ttt_write_commit_filter_mode": mode,
                "ttt_write_commit_filter_branch_mask": list(branch_mask),
                "ttt_write_commit_filter_num_tensors": int(applied),
                "ttt_write_commit_filter_min": lo,
                "ttt_write_commit_filter_max": hi,
                "ttt_write_commit_filter_state_q_threshold": float(q_threshold),
                "ttt_write_commit_filter_state_c_threshold": float(c_threshold),
                "ttt_write_commit_filter_state_q_mean": (
                    float(sum(q_values) / len(q_values)) if q_values else 0.0
                ),
                "ttt_write_commit_filter_state_c_mean": (
                    float(sum(c_values) / len(c_values)) if c_values else 0.0
                ),
                "ttt_write_commit_filter_scale_mean": (
                    float(sum(alpha_values) / len(alpha_values)) if alpha_values else 1.0
                ),
                "ttt_write_commit_filter_activation_rate": (
                    float(sum(1 for x in trigger_values if x) / len(trigger_values)) if trigger_values else 0.0
                ),
            })
            return

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

        base = float(self.commit_filter_base)
        gain = float(self.commit_filter_gain)
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
        ema_chunks = tuple(self.commit_ema_chunks)
        current_chunk = int(getattr(self, "current_chunk_idx", getattr(self, "v11_projection_chunk_idx", -1)))
        debug_info["ttt_write_commit_ema_branch_mask"] = list(branch_mask)
        debug_info["ttt_write_commit_ema_chunks"] = list(ema_chunks)
        debug_info["ttt_write_commit_ema_current_chunk"] = current_chunk
        if ema_chunks and current_chunk not in ema_chunks:
            debug_info["ttt_write_commit_ema_applied"] = False
            debug_info["ttt_write_commit_ema_alpha"] = alpha
            debug_info["ttt_write_commit_ema_chunk_gate_active"] = False
            return
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
        debug_info["ttt_write_commit_ema_chunk_gate_active"] = True
        debug_info["ttt_write_commit_ema_num_tensors"] = int(applied)

    def _summarize_commit_against_native(
        self,
        write_cache: WriteCacheOutput,
        w0_new: List[Optional[torch.Tensor]],
        w1_new: List[Optional[torch.Tensor]],
        w2_new: List[Optional[torch.Tensor]],
        debug_info: Dict[str, Any],
    ) -> None:
        """Record detached post-write delta summaries without changing state."""
        enabled = str(os.environ.get("TTT_WRITE_POST_ZP_SUMMARY", "1")).strip().lower()
        if enabled in {"0", "false", "off", "no"}:
            debug_info["ttt_post_zp_delta_summary_available"] = False
            debug_info["ttt_post_zp_delta_summary_disabled"] = True
            return

        branches = (
            ("w0", w0_new, write_cache.w0_provisional, "w0_old"),
            ("w1", w1_new, write_cache.w1_provisional, "w1_old"),
            ("w2", w2_new, write_cache.w2_provisional, "w2_old"),
        )
        eps = 1e-12
        committed_norms: List[float] = []
        native_norms: List[float] = []
        action_norms: List[float] = []
        committed_native_cos: List[float] = []
        action_native_cos: List[float] = []
        row_count = 0
        dump_dir_text = str(os.environ.get("TTT_WRITE_POST_ZP_DUMP_DIR", "") or "").strip()
        dump_dtype_name = str(os.environ.get("TTT_WRITE_POST_ZP_DUMP_DTYPE", "float16") or "float16").strip().lower()
        dump_dtype = torch.float32 if dump_dtype_name == "float32" else torch.float16
        dump_payload: Optional[Dict[str, Any]] = None
        if dump_dir_text:
            current_chunk = int(getattr(self, "current_chunk_idx", getattr(self, "v11_projection_chunk_idx", -1)))
            dump_payload = {
                "schema": "acl2_v68_ttt_post_zp_delta_dump_v1",
                "chunk_idx": current_chunk,
                "dump_dtype": str(dump_dtype),
                "tensors_are_fast_weight_deltas_not_spatial_token_maps": True,
                "rows": [],
                "deltas": {},
            }

        for name, committed_list, native_list, old_attr in branches:
            for li, committed in enumerate(committed_list):
                if committed is None or li >= len(native_list) or li >= len(write_cache.layer_caches):
                    continue
                native = native_list[li]
                old = getattr(write_cache.layer_caches[li], old_attr, None)
                if native is None or old is None:
                    continue
                committed_f = committed.detach().float()
                native_f = native.detach().float().to(device=committed_f.device)
                old_f = old.detach().float().to(device=committed_f.device)
                if committed_f.shape != native_f.shape or committed_f.shape != old_f.shape:
                    layer_debug = debug_info.get(f"layer_{li}")
                    if isinstance(layer_debug, dict):
                        layer_debug[f"ttt_post_zp_{name}_summary_skip"] = "shape_mismatch"
                    continue

                committed_delta = committed_f - old_f
                native_delta = native_f - old_f
                action_delta = committed_f - native_f
                committed_norm = torch.linalg.vector_norm(committed_delta.reshape(-1)).clamp_min(eps)
                native_norm = torch.linalg.vector_norm(native_delta.reshape(-1)).clamp_min(eps)
                action_norm = torch.linalg.vector_norm(action_delta.reshape(-1)).clamp_min(eps)
                cos_committed = (
                    committed_delta.reshape(-1) @ native_delta.reshape(-1)
                ) / (committed_norm * native_norm)
                cos_action = (
                    action_delta.reshape(-1) @ native_delta.reshape(-1)
                ) / (action_norm * native_norm)
                cos_committed = cos_committed.clamp(-1.0, 1.0)
                cos_action = cos_action.clamp(-1.0, 1.0)

                c_norm = float(committed_norm.item())
                n_norm = float(native_norm.item())
                a_norm = float(action_norm.item())
                c_cos = float(cos_committed.item())
                a_cos = float(cos_action.item())
                committed_norms.append(c_norm)
                native_norms.append(n_norm)
                action_norms.append(a_norm)
                committed_native_cos.append(c_cos)
                action_native_cos.append(a_cos)
                row_count += 1

                layer_debug = debug_info.get(f"layer_{li}")
                if isinstance(layer_debug, dict):
                    layer_debug[f"ttt_post_zp_{name}_committed_delta_norm"] = c_norm
                    layer_debug[f"ttt_post_zp_{name}_native_delta_norm"] = n_norm
                    layer_debug[f"ttt_post_zp_{name}_action_delta_norm"] = a_norm
                    layer_debug[f"candidate_native_cos_{name}"] = c_cos
                    layer_debug[f"candidate_action_native_cos_{name}"] = a_cos

                if dump_payload is not None:
                    key = f"layer_{int(li):02d}_{name}"
                    dump_payload["rows"].append({
                        "layer": int(li),
                        "branch": str(name),
                        "shape": [int(v) for v in committed_delta.shape],
                        "committed_delta_norm": c_norm,
                        "native_delta_norm": n_norm,
                        "action_delta_norm": a_norm,
                        "candidate_native_cos": c_cos,
                        "candidate_action_native_cos": a_cos,
                    })
                    dump_payload["deltas"][key] = {
                        "committed_delta": committed_delta.detach().cpu().to(dtype=dump_dtype),
                        "native_delta": native_delta.detach().cpu().to(dtype=dump_dtype),
                        "action_delta": action_delta.detach().cpu().to(dtype=dump_dtype),
                    }

        def mean(values: List[float]) -> Optional[float]:
            return float(sum(values) / len(values)) if values else None

        debug_info["ttt_post_zp_delta_summary_available"] = bool(row_count)
        debug_info["ttt_post_zp_delta_summary_count"] = int(row_count)
        debug_info["ttt_post_zp_committed_delta_norm_mean"] = mean(committed_norms)
        debug_info["ttt_post_zp_native_delta_norm_mean"] = mean(native_norms)
        debug_info["ttt_post_zp_action_delta_norm_mean"] = mean(action_norms)
        debug_info["candidate_native_cosine_mean"] = mean(committed_native_cos)
        debug_info["candidate_action_native_cosine_mean"] = mean(action_native_cos)
        if dump_payload is not None:
            out_dir = Path(dump_dir_text)
            out_dir.mkdir(parents=True, exist_ok=True)
            chunk_idx = int(dump_payload.get("chunk_idx", -1))
            if chunk_idx >= 0:
                out_path = out_dir / f"chunk_{chunk_idx:03d}_ttt_post_zp_delta.pt"
            else:
                out_path = out_dir / "chunk_unknown_ttt_post_zp_delta.pt"
            torch.save(dump_payload, out_path)
            debug_info["ttt_post_zp_delta_dump_path"] = str(out_path)
            debug_info["ttt_post_zp_delta_dump_tensor_groups"] = int(len(dump_payload["deltas"]))
            debug_info["ttt_post_zp_delta_dump_rows"] = int(len(dump_payload["rows"]))
            debug_info["ttt_post_zp_delta_dump_dtype"] = str(dump_dtype)

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
        if mode not in {"cosine", "cosine_soft", "cap", "cosine_cap", "orthogonal_suppress"}:
            raise ValueError(f"Unsupported native delta gate mode: {self.native_delta_gate_mode}")

        min_cos = float(self.native_delta_gate_min_cos)
        fallback = min(max(float(self.native_delta_gate_fallback), 0.0), 1.0)
        orthogonal_rho = fallback
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

                if mode == "orthogonal_suppress":
                    coeff = dot / native_norm.square().clamp_min(eps)
                    view_shape = [coeff.shape[0]] + [1] * (semantic_f.ndim - 1)
                    parallel = coeff.view(*view_shape) * native_delta
                    perpendicular = semantic_delta - parallel
                    routed_delta = parallel + orthogonal_rho * perpendicular
                    gated = old_f + routed_delta
                    scale = torch.full_like(cosine, orthogonal_rho)
                elif mode in {"cosine", "cosine_cap"}:
                    scale = torch.where(cosine >= min_cos, scale, torch.full_like(scale, fallback))
                elif mode == "cosine_soft":
                    denom = max(1.0 - min_cos, eps)
                    soft = ((cosine - min_cos) / denom).clamp(0.0, 1.0)
                    scale = fallback + (1.0 - fallback) * soft

                if mode != "orthogonal_suppress" and mode in {"cap", "cosine_cap"} and cap_ratio > 0.0:
                    cap = cap_ratio * native_norm
                    cap_scale = (cap / correction_norm).clamp(max=1.0)
                    scale = torch.minimum(scale, cap_scale)
                elif mode == "cap" and cap_ratio <= 0.0:
                    scale = torch.zeros_like(scale)

                if mode != "orthogonal_suppress":
                    view_shape = [scale.shape[0]] + [1] * (semantic_f.ndim - 1)
                    gated = native_f + scale.view(*view_shape) * correction
                pre_delta_norm = semantic_norm.detach()
                old_norm = old_f.norm(dim=1, keepdim=True)
                gated = gated / (gated.norm(dim=1, keepdim=True) + 1e-5) * old_norm
                post_delta = gated - old_f
                post_delta_norm = torch.linalg.vector_norm(post_delta, dim=reduce_dims).clamp_min(eps)
                pre_post_dot = (semantic_delta * post_delta).sum(dim=reduce_dims)
                pre_post_cos = (pre_post_dot / (pre_delta_norm * post_delta_norm)).clamp(-1.0, 1.0)
                semantic_list[li] = gated.to(dtype=semantic.dtype).cpu()

                layer_debug = debug_info.get(f"layer_{li}")
                if isinstance(layer_debug, dict):
                    layer_debug[f"ttt_write_native_delta_gate_{name}_mode"] = mode
                    layer_debug[f"ttt_write_native_delta_gate_{name}_scale_mean"] = float(scale.mean().item())
                    layer_debug[f"ttt_write_native_delta_gate_{name}_scale_min"] = float(scale.min().item())
                    layer_debug[f"ttt_write_native_delta_gate_{name}_cos_mean"] = float(cosine.mean().item())
                    layer_debug[f"ttt_write_native_delta_gate_{name}_cos_min"] = float(cosine.min().item())
                    layer_debug[f"ttt_write_native_delta_gate_{name}_cap_ratio"] = float(cap_ratio)
                    layer_debug[f"ttt_write_native_delta_gate_{name}_orthogonal_rho"] = float(orthogonal_rho)
                    layer_debug[f"ttt_write_native_delta_gate_{name}_pre_delta_norm_mean"] = float(pre_delta_norm.mean().item())
                    layer_debug[f"ttt_write_native_delta_gate_{name}_post_delta_norm_mean"] = float(post_delta_norm.mean().item())
                    layer_debug[f"ttt_write_native_delta_gate_{name}_pre_post_cos_mean"] = float(pre_post_cos.mean().item())
                    layer_debug[f"ttt_write_native_delta_gate_{name}_norm_restore_ratio_mean"] = float(
                        (post_delta_norm / pre_delta_norm).mean().item()
                    )
                applied += 1
                scale_means.append(float(scale.mean().item()))
                cosine_means.append(float(cosine.mean().item()))

        debug_info["ttt_write_native_delta_gate_applied"] = bool(applied)
        debug_info["ttt_write_native_delta_gate_mode"] = mode
        debug_info["ttt_write_native_delta_gate_branch_mask"] = list(branch_mask)
        debug_info["ttt_write_native_delta_gate_min_cos"] = min_cos
        debug_info["ttt_write_native_delta_gate_fallback"] = fallback
        debug_info["ttt_write_native_delta_gate_orthogonal_rho"] = orthogonal_rho
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
    def _parse_chunk_mask(text: Optional[str]) -> Tuple[int, ...]:
        if text is None:
            return ()
        raw = str(text).strip().lower()
        if raw in {"", "all", "none", "off"}:
            return ()
        chunks: List[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s), int(end_s)
                lo, hi = min(start, end), max(start, end)
                for idx in range(lo, hi + 1):
                    if idx not in chunks:
                        chunks.append(idx)
            else:
                idx = int(part)
                if idx not in chunks:
                    chunks.append(idx)
        return tuple(chunks)

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
