"""
Stage B: Dynamic Cue Extractor

Extracts five-channel geometric cues and a geometry-driven write-allow map
from LoGeR's structured geometry outputs.  The cues quantify, for every
pixel in the current chunk, how likely the region is to be statically
consistent, dynamically violated, occluded, uncertain, or suitable as a
long-term memory write anchor.

Phase 1 implements:
  - chunk-internal pairwise point residuals (same-pixel world-space
    comparison + depth-ordering occlusion check)
  - five cue channels: C_stat, C_dyn, C_occ, C_unc, C_anchor
  - geometry-driven write-allow map G_write_geo
  - exact patch-level pooling to token grid
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .geometry_backbone import GeometryOutput, PATCH_SIZE

# ---- Cue channel indices ---------------------------------------------------
CUE_STAT = 0
CUE_DYN = 1
CUE_OCC = 2
CUE_UNC = 3
CUE_ANCHOR = 4
NUM_CUE_CHANNELS = 5


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class CueOutput:
    """Structured output of Stage B for a single chunk.

    All tensors are on CPU, float32.

    Attributes
    ----------
    E_cue : [T, H_p, W_p, 5]
        Five-channel cue tensor (channels: stat, dyn, occ, unc, anchor).
    G_write_geo : [T, H_p, W_p]
        Geometry-driven write-allow map in [0, 1].
    E_cue_patch : [T, H_tok, W_tok, 5]  (optional)
        Patch-pooled cue tensor.
    G_write_geo_patch : [T, H_tok, W_tok]  (optional)
        Patch-pooled write-allow map.
    """

    E_cue: torch.Tensor
    G_write_geo: torch.Tensor
    C_dyn_explicit: Optional[torch.Tensor] = None
    C_dyn_implicit: Optional[torch.Tensor] = None
    C_dyn_fusion_max: Optional[torch.Tensor] = None
    C_dyn_fusion_soft_or: Optional[torch.Tensor] = None
    C_dyn_fusion_avg: Optional[torch.Tensor] = None
    C_dyn_fusion_addclip: Optional[torch.Tensor] = None
    E_cue_patch: Optional[torch.Tensor] = None
    G_write_geo_patch: Optional[torch.Tensor] = None

    num_frames: int = 0
    spatial_resolution: Tuple[int, int] = (0, 0)
    patch_grid: Tuple[int, int] = (0, 0)
    debug: Dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors -----------------------------------------------
    @property
    def C_stat(self) -> torch.Tensor:
        return self.E_cue[..., CUE_STAT]

    @property
    def C_dyn(self) -> torch.Tensor:
        return self.E_cue[..., CUE_DYN]

    @property
    def C_occ(self) -> torch.Tensor:
        return self.E_cue[..., CUE_OCC]

    @property
    def C_unc(self) -> torch.Tensor:
        return self.E_cue[..., CUE_UNC]

    @property
    def C_anchor(self) -> torch.Tensor:
        return self.E_cue[..., CUE_ANCHOR]


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------
class DynamicCueExtractor:
    """Stage B of the Semantic Prior Pipeline.

    Thin, stateless, CPU-only module that converts a
    :class:`GeometryOutput` into geometric evidence cues for downstream
    write-control modules.

    Usage::

        extractor = DynamicCueExtractor(k_intra=3, sigma_pt=0.03)
        cue_output = extractor.run(geometry_output)
    """

    def __init__(
        self,
        *,
        # Support set
        k_intra: int = 10,
        use_attention_prior: bool = True,
        support_time_decay: float = 2.0,
        support_temporal_weight: float = 0.35,
        support_affinity_weight: float = 0.45,
        support_static_weight: float = 0.20,
        # Point residual scale.  Same-pixel world-space comparison has
        # an inherent parallax baseline even on static surfaces (~0.03
        # median, ~0.08 mean for indoor).  sigma_pt must be well above
        # this baseline so that static regions achieve high C_stat.
        # sigma_pt = 0.25 puts the static P90 (≈0.08) at exp(-0.32)≈0.73,
        # while a walking person (r_pt≈0.3) drops to exp(-1.2)≈0.30.
        sigma_pt: float = 0.25,
        proxy_mode: str = "same_pixel",
        # Occlusion depth threshold — fraction of scene scale.
        # tau_occ = 0.05 yields ~1.8% occlusion on small-motion indoor.
        tau_occ: float = 0.05,
        # C_dyn combination weights
        alpha_1: float = 0.8,
        alpha_2: float = 0.0,
        alpha_3: float = 0.5,
        attn_stat_fusion_weight: float = 0.35,
        attn_dyn_weight: float = 0.30,
        attn_gate_power: float = 1.0,
        attn_debias_kernel: int = 7,
        dyn_fusion_mode: str = "max",
        implicit_weight: float = 1.0,
        implicit_gate_floor: float = 0.25,
        implicit_calib_min_range: float = 1e-3,
        nonocc_dynamic: bool = False,
        # C_unc parameters
        conf_floor: float = 0.1,
        unc_conf_weight: float = 0.3,
        # G_write_geo linear combination
        lambda_s: float = 1.0,
        lambda_a: float = 0.5,
        lambda_d: float = 0.8,
        lambda_o: float = 0.3,
        lambda_u: float = 0.5,
        # Trimmed-mean fraction to trim from each tail
        trim_ratio: float = 0.2,
        # Whether to produce patch-level outputs
        compute_patch_cues: bool = True,
    ):
        self.k_intra = k_intra
        self.max_support_views = 4
        self.use_attention_prior = use_attention_prior
        self.support_time_decay = support_time_decay
        self.support_temporal_weight = support_temporal_weight
        self.support_affinity_weight = support_affinity_weight
        self.support_static_weight = support_static_weight
        self.sigma_pt = sigma_pt
        valid_proxy_modes = {"same_pixel", "reprojection"}
        if proxy_mode not in valid_proxy_modes:
            raise ValueError(f"Unsupported Stage-B proxy_mode: {proxy_mode}")
        self.proxy_mode = proxy_mode
        self.tau_occ = tau_occ

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.attn_stat_fusion_weight = attn_stat_fusion_weight
        self.attn_dyn_weight = attn_dyn_weight  # kept for CLI/API compatibility
        self.attn_gate_power = attn_gate_power
        self.attn_debias_kernel = max(int(attn_debias_kernel), 1)
        valid_fusion = {
            "explicit",
            "implicit",
            "max",
            "soft_or",
            "avg",
            "addclip",
            "calibrated_soft_or",
        }
        if dyn_fusion_mode not in valid_fusion:
            raise ValueError(f"Unsupported dyn_fusion_mode: {dyn_fusion_mode}")
        self.dyn_fusion_mode = dyn_fusion_mode
        self.implicit_weight = float(implicit_weight)
        self.implicit_gate_floor = float(implicit_gate_floor)
        self.implicit_calib_min_range = float(implicit_calib_min_range)
        self.nonocc_dynamic = bool(nonocc_dynamic)

        self.conf_floor = conf_floor
        self.unc_conf_weight = unc_conf_weight

        self.lambda_s = lambda_s
        self.lambda_a = lambda_a
        self.lambda_d = lambda_d
        self.lambda_o = lambda_o
        self.lambda_u = lambda_u

        self.trim_ratio = trim_ratio
        self.compute_patch_cues = compute_patch_cues

    # ---- public API --------------------------------------------------------

    def run(self, geo: GeometryOutput) -> CueOutput:
        """Compute cues from a :class:`GeometryOutput`.

        Parameters
        ----------
        geo : GeometryOutput
            Structured geometry output from Stage A.

        Returns
        -------
        CueOutput
            Five-channel cues, write-allow map, and optional patch-level
            pooled versions.
        """
        world_pts = geo.world_points   # [T, H, W, 3]
        local_pts = geo.local_points   # [T, H, W, 3]
        cam_poses = geo.camera_poses   # [T, 4, 4]
        conf = geo.confidence          # [T, H, W]
        frame_attention_prior = geo.frame_attention_prior  # [T, T] or None
        attn_dynamic_patch = geo.attn_dynamic_patch        # [T, H_tok, W_tok] or None

        T, H, W = conf.shape

        # -- single-frame fallback -------------------------------------------
        if T <= 1:
            return self._single_frame_fallback(conf, geo.patch_grid)

        # -- multi-frame cue extraction --------------------------------------
        attn_dynamic = self._upsample_attn_dynamic(
            attn_dynamic_patch, target_hw=(H, W),
        )
        support_idx, support_valid, support_score = self._build_support_tensor(
            T,
            frame_attention_prior=frame_attention_prior,
            attn_dynamic_patch=attn_dynamic_patch,
        )
        T_cw = torch.inverse(cam_poses)  # [T, 4, 4]

        C_stat, C_stat_geom, C_occ, C_unc, debug_info = self._compute_pairwise_cues(
            world_pts,
            local_pts,
            T_cw,
            conf,
            support_idx,
            support_valid,
            support_score=support_score,
            attn_dynamic=attn_dynamic,
        )

        # Explicit geometry and implicit key-cosine evidence are fused as
        # independent dynamic cues. We preserve the stronger branch at each
        # location so the implicit cue can keep its own shape instead of only
        # acting as a small correction on top of geometry.
        D_exp = torch.clamp(
            self.alpha_1 * (1.0 - C_stat_geom)
            + self.alpha_2 * 0.0  # boundary-break term reserved for Phase 2
            - self.alpha_3 * C_occ,
            0.0, 1.0,
        )
        if attn_dynamic is not None and self.use_attention_prior:
            # Keep the implicit cue close to the MUT3R-style Stage-A feature:
            # selected frame-attention key-cosine maps are averaged in Stage A,
            # then used directly before max-fusion. We intentionally do not
            # apply local debiasing, extra geometric gating, or an additional
            # scalar attenuation in this path.
            attn_dyn_evidence = attn_dynamic.clamp(0, 1)
            attn_dyn_support = attn_dyn_evidence
            C_dyn_fusion_max = torch.maximum(D_exp, attn_dyn_support).clamp(0.0, 1.0)
            C_dyn_fusion_soft_or = (
                1.0 - (1.0 - D_exp) * (1.0 - attn_dyn_support)
            ).clamp(0.0, 1.0)
            C_dyn_fusion_avg = (0.5 * (D_exp + attn_dyn_support)).clamp(0.0, 1.0)
            C_dyn_fusion_addclip = (D_exp + attn_dyn_support).clamp(0.0, 1.0)
            C_dyn = self._select_dynamic_fusion(
                D_exp=D_exp,
                D_imp=attn_dyn_support,
                C_stat=C_stat,
                C_unc=C_unc,
                fusion_max=C_dyn_fusion_max,
                fusion_soft_or=C_dyn_fusion_soft_or,
                fusion_avg=C_dyn_fusion_avg,
                fusion_addclip=C_dyn_fusion_addclip,
                debug_info=debug_info,
            )
            debug_info["attention_dynamic_mean"] = attn_dynamic.mean().item()
            debug_info["attention_dynamic_used_mean"] = attn_dyn_evidence.mean().item()
            debug_info["attention_support_mean"] = attn_dyn_support.mean().item()
        else:
            attn_dyn_support = torch.zeros_like(D_exp)
            C_dyn_fusion_max = D_exp
            C_dyn_fusion_soft_or = D_exp
            C_dyn_fusion_avg = D_exp
            C_dyn_fusion_addclip = D_exp
            C_dyn = D_exp
            debug_info["attention_fusion_mode"] = "explicit_no_attention"
        debug_info["explicit_dynamic_mean"] = D_exp.mean().item()
        debug_info["selected_dynamic_mean"] = C_dyn.mean().item()

        debug_info["attention_prior_used"] = bool(
            self.use_attention_prior
            and (frame_attention_prior is not None or attn_dynamic is not None)
        )
        if frame_attention_prior is not None:
            debug_info["frame_attention_mean"] = frame_attention_prior.mean().item()
        debug_info["support_score_per_frame"] = support_score.sum(dim=1)

        if self.nonocc_dynamic:
            C_dyn_raw = C_dyn
            C_dyn = (C_dyn * (1.0 - C_occ) * (1.0 - C_unc)).clamp(0.0, 1.0)
            debug_info["nonocc_dynamic_enabled"] = True
            debug_info["selected_dynamic_raw_mean"] = C_dyn_raw.mean().item()
            debug_info["selected_dynamic_nonocc_mean"] = C_dyn.mean().item()
        else:
            debug_info["nonocc_dynamic_enabled"] = False

        # C_anchor = C_stat * (1 - C_dyn) * (1 - C_unc)
        C_anchor = C_stat * (1.0 - C_dyn) * (1.0 - C_unc)

        # -- stack into E_cue [T, H, W, 5] ----------------------------------
        E_cue = torch.stack([C_stat, C_dyn, C_occ, C_unc, C_anchor], dim=-1)

        # -- G_write_geo -----------------------------------------------------
        z_geo = (
            self.lambda_s * C_stat
            + self.lambda_a * C_anchor
            - self.lambda_d * C_dyn
            - self.lambda_o * C_occ
            - self.lambda_u * C_unc
        )
        G_write_geo = torch.sigmoid(z_geo)

        # -- patch pooling ---------------------------------------------------
        E_cue_patch = None
        G_write_geo_patch = None
        if self.compute_patch_cues and geo.patch_grid != (0, 0):
            E_cue_patch, G_write_geo_patch = self._patch_pool(
                E_cue, G_write_geo, geo.patch_grid,
            )

        return CueOutput(
            E_cue=E_cue,
            G_write_geo=G_write_geo,
            C_dyn_explicit=D_exp,
            C_dyn_implicit=attn_dyn_support,
            C_dyn_fusion_max=C_dyn_fusion_max,
            C_dyn_fusion_soft_or=C_dyn_fusion_soft_or,
            C_dyn_fusion_avg=C_dyn_fusion_avg,
            C_dyn_fusion_addclip=C_dyn_fusion_addclip,
            E_cue_patch=E_cue_patch,
            G_write_geo_patch=G_write_geo_patch,
            num_frames=T,
            spatial_resolution=(H, W),
            patch_grid=geo.patch_grid,
            debug=debug_info,
        )

    def _select_dynamic_fusion(
        self,
        *,
        D_exp: torch.Tensor,
        D_imp: torch.Tensor,
        C_stat: torch.Tensor,
        C_unc: torch.Tensor,
        fusion_max: torch.Tensor,
        fusion_soft_or: torch.Tensor,
        fusion_avg: torch.Tensor,
        fusion_addclip: torch.Tensor,
        debug_info: Dict[str, Any],
    ) -> torch.Tensor:
        mode = self.dyn_fusion_mode
        debug_info["attention_fusion_mode"] = mode
        debug_info["implicit_weight"] = self.implicit_weight

        if mode == "explicit":
            return D_exp.clamp(0.0, 1.0)
        if mode == "implicit":
            return (self.implicit_weight * D_imp).clamp(0.0, 1.0)
        if mode == "max":
            return fusion_max
        if mode == "soft_or":
            return fusion_soft_or
        if mode == "avg":
            return fusion_avg
        if mode == "addclip":
            return fusion_addclip

        # Calibrated weighted soft-or from the KITTI01 GSL-WC plan.
        q50 = torch.quantile(D_imp.flatten().float(), 0.50)
        q95 = torch.quantile(D_imp.flatten().float(), 0.95)
        span = (q95 - q50).clamp_min(0.0)
        if float(span.item()) < self.implicit_calib_min_range:
            D_imp_cal = torch.zeros_like(D_imp)
            effective_weight = 0.0
        else:
            D_imp_cal = ((D_imp - q50) / (span + 1e-6)).clamp(0.0, 1.0)
            effective_weight = self.implicit_weight

        C_anchor_exp = (C_stat * (1.0 - D_exp) * (1.0 - C_unc)).clamp(0.0, 1.0)
        floor = float(self.implicit_gate_floor)
        g_imp = (floor + (1.0 - floor) * (1.0 - C_anchor_exp)).clamp(floor, 1.0)
        D_imp_weighted = (effective_weight * g_imp * D_imp_cal).clamp(0.0, 1.0)
        C_dyn = (1.0 - (1.0 - D_exp) * (1.0 - D_imp_weighted)).clamp(0.0, 1.0)

        debug_info["implicit_q50"] = float(q50.item())
        debug_info["implicit_q95"] = float(q95.item())
        debug_info["implicit_calib_span"] = float(span.item())
        debug_info["implicit_effective_weight"] = float(effective_weight)
        debug_info["implicit_calibrated_mean"] = float(D_imp_cal.mean().item())
        debug_info["implicit_gate_mean"] = float(g_imp.mean().item())
        return C_dyn

    # ---- support set -------------------------------------------------------

    def _build_support_tensor(
        self,
        T: int,
        frame_attention_prior: Optional[torch.Tensor] = None,
        attn_dynamic_patch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.Tensor]:
        """Build a ``[T, K]`` index tensor and matching validity mask.

        For each frame *t*, first restrict candidates to the local temporal
        window ``[t-k_intra//2, t+k_intra//2]`` (excluding ``t`` itself). It
        then samples up to four support views uniformly across that window.
        When attention priors are available, each temporal bin chooses the
        highest-scoring view inside that bin.
        """
        K = self.max_support_views
        idx = torch.zeros(T, K, dtype=torch.long)
        valid = torch.zeros(T, K, dtype=torch.bool)
        score = torch.zeros(T, K, dtype=torch.float32)

        has_attention_prior = (
            self.use_attention_prior
            and (frame_attention_prior is not None or attn_dynamic_patch is not None)
        )

        if K <= 0:
            return idx, valid, score

        time_ids = torch.arange(T, dtype=torch.float32)
        time_dist = (time_ids[:, None] - time_ids[None, :]).abs()
        time_score = torch.exp(-time_dist / max(self.support_time_decay, 1e-6))

        combined = self.support_temporal_weight * time_score

        if frame_attention_prior is not None:
            affinity = frame_attention_prior.float().clamp(0, 1)
            affinity = 0.5 * (affinity + affinity.transpose(0, 1))
            combined = combined + self.support_affinity_weight * affinity

        if attn_dynamic_patch is not None:
            static_patch = (1.0 - attn_dynamic_patch.float().clamp(0, 1)).reshape(T, -1)
            static_overlap = torch.einsum("td,sd->ts", static_patch, static_patch)
            static_overlap = static_overlap / max(static_patch.shape[1], 1)
            combined = combined + self.support_static_weight * static_overlap

        combined.fill_diagonal_(0.0)
        combined = combined / combined.amax(dim=-1, keepdim=True).clamp_min(1e-6)

        for t in range(T):
            window_radius = max(int(self.k_intra) // 2, 0)
            left = max(0, t - window_radius)
            right = min(T, t + window_radius + 1)
            candidate_idx = torch.arange(left, right, dtype=torch.long)
            candidate_idx = candidate_idx[candidate_idx != t]
            num_candidates = int(candidate_idx.numel())
            if num_candidates <= 0:
                continue

            num_take = min(K, num_candidates)
            if num_candidates <= num_take:
                selected = candidate_idx
                if has_attention_prior:
                    selected_scores = combined[t, selected]
                else:
                    selected_scores = torch.ones(num_take, dtype=torch.float32)
            else:
                selected_parts = []
                selected_score_parts = []
                row_scores = combined[t, candidate_idx] if has_attention_prior else None
                for bin_idx in range(num_take):
                    start = (bin_idx * num_candidates) // num_take
                    end = ((bin_idx + 1) * num_candidates) // num_take
                    if end <= start:
                        end = start + 1
                    bin_candidates = candidate_idx[start:end]
                    if bin_candidates.numel() == 0:
                        continue
                    if has_attention_prior:
                        bin_scores = row_scores[start:end]
                        best_rel = torch.argmax(bin_scores)
                        selected_parts.append(bin_candidates[best_rel:best_rel + 1])
                        selected_score_parts.append(bin_scores[best_rel:best_rel + 1])
                    else:
                        center_rel = bin_candidates.numel() // 2
                        selected_parts.append(bin_candidates[center_rel:center_rel + 1])
                        selected_score_parts.append(torch.ones(1, dtype=torch.float32))

                if not selected_parts:
                    continue
                selected = torch.cat(selected_parts, dim=0)
                selected_scores = torch.cat(selected_score_parts, dim=0)

            take = int(selected.numel())
            idx[t, :take] = selected
            valid[t, :take] = True
            score[t, :take] = selected_scores

        return idx, valid, score

    @staticmethod
    def _upsample_attn_dynamic(
        attn_dynamic_patch: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if attn_dynamic_patch is None:
            return None
        target_h, target_w = target_hw
        return F.interpolate(
            attn_dynamic_patch.unsqueeze(1),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).clamp(0, 1)

    def _debias_attn_dynamic(self, attn_dynamic: torch.Tensor) -> torch.Tensor:
        """Suppress large low-frequency scene-layout responses in attention features."""
        _, h, w = attn_dynamic.shape
        max_kernel = max(min(h, w), 1)
        if max_kernel % 2 == 0:
            max_kernel = max(max_kernel - 1, 1)
        kernel = min(max(int(self.attn_debias_kernel), 1), max_kernel)
        if kernel % 2 == 0:
            kernel += 1
        if kernel <= 1:
            return attn_dynamic.clamp(0, 1)

        pad = kernel // 2
        x = attn_dynamic.unsqueeze(1)
        pad_mode = "reflect" if h > pad and w > pad else "replicate"
        x_pad = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)
        trend = F.avg_pool2d(x_pad, kernel_size=kernel, stride=1)
        residual = torch.relu(x - trend)
        scale = residual.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (residual / scale).squeeze(1).clamp(0, 1)

    # ---- pairwise evidence -------------------------------------------------

    def _compute_pairwise_cues(
        self,
        world_pts: torch.Tensor,    # [T, H, W, 3]
        local_pts: torch.Tensor,    # [T, H, W, 3]
        T_cw: torch.Tensor,         # [T, 4, 4]
        conf: torch.Tensor,         # [T, H, W]
        support_idx: torch.Tensor,  # [T, K]
        support_valid: torch.Tensor, # [T, K]
        support_score: Optional[torch.Tensor] = None,  # [T, K]
        attn_dynamic: Optional[torch.Tensor] = None,   # [T, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compute C_stat, C_stat_geom, C_occ, C_unc from pairwise residuals.

        Returns (C_stat_fused, C_stat_geom, C_occ, C_unc, debug_dict).
        """
        T, H, W, _ = world_pts.shape
        K = support_idx.shape[1]
        eps = 1e-7

        # Pre-compute world-point norms for the relative residual denominator
        norm_t = world_pts.norm(dim=-1)  # [T, H, W]
        intrinsics = None
        if self.proxy_mode == "reprojection":
            intrinsics = self._estimate_pinhole_intrinsics(local_pts, conf)

        # -- Per-pair accumulators -----------------------------------------------
        # C_stat: pure geometric consistency (confidence goes to C_unc).
        # Only pixels where both frames have conf > conf_floor participate.
        all_consistency = torch.zeros(K, T, H, W)
        all_consistency_valid = torch.zeros(K, T, H, W, dtype=torch.bool)
        weighted_consistency_sum = torch.zeros(T, H, W)
        weighted_consistency_weight = torch.zeros(T, H, W)

        # C_occ: confidence-weighted fraction of depth-ordering violations
        occ_weighted_sum = torch.zeros(T, H, W)
        weight_sum = torch.zeros(T, H, W)

        # C_unc: weighted valid-projection ratio
        valid_weighted_sum = torch.zeros(T, H, W)

        # Debug: mean point residual
        r_pt_sum = torch.zeros(T, H, W)
        r_pt_count = torch.zeros(T, H, W)

        for k in range(K):
            s_idx = support_idx[:, k]         # [T]
            valid_k = support_valid[:, k]     # [T] bool
            if not valid_k.any():
                continue

            # Expand frame-level validity to spatial dims: [T, H, W]
            vk = valid_k.float()[:, None, None].expand(T, H, W)
            support_w = valid_k.float()
            if support_score is not None:
                support_w = support_score[:, k].float()
            support_w_map = support_w[:, None, None].expand(T, H, W)

            # ---- gather support-frame data ---------------------------------
            R_s = T_cw[s_idx][:, :3, :3]     # [T, 3, 3]
            t_s = T_cw[s_idx][:, :3, 3]      # [T, 3]
            X_in_s = (
                torch.einsum("tij, thwj -> thwi", R_s, world_pts)
                + t_s[:, None, None, :]
            )  # [T, H, W, 3]
            z_proj = X_in_s[..., 2]           # projected depth in frame s

            if self.proxy_mode == "reprojection":
                assert intrinsics is not None
                X_s, conf_s, depth_s, proj_valid = self._sample_support_reprojection(
                    world_pts=world_pts,
                    local_pts=local_pts,
                    conf=conf,
                    support_index=s_idx,
                    X_in_support=X_in_s,
                    intrinsics=intrinsics,
                )
            else:
                X_s = world_pts[s_idx]              # [T, H, W, 3]
                conf_s = conf[s_idx]                # [T, H, W]
                depth_s = local_pts[s_idx, ..., 2]  # [T, H, W]
                proj_valid = z_proj > 0

            # ---- point residual --------------------------------------------
            diff = (world_pts - X_s).norm(dim=-1)  # [T, H, W]
            r_pt = diff / (eps + norm_t)           # [T, H, W]

            # Confidence-based pixel validity mask
            conf_ok = (conf > self.conf_floor) & (conf_s > self.conf_floor)

            # Pure geometric consistency (no conf multiplication).
            # Confidence-unreliable pixels are excluded via the validity
            # mask so they don't bias C_stat.
            c = torch.exp(-r_pt / self.sigma_pt)    # [T, H, W] in (0, 1]
            all_consistency[k] = c
            pair_valid = valid_k[:, None, None].expand(T, H, W) & conf_ok
            all_consistency_valid[k] = pair_valid

            pair_weight = support_w_map
            if attn_dynamic is not None and self.use_attention_prior:
                static_t = (1.0 - attn_dynamic).clamp(0, 1)
                static_s = (1.0 - attn_dynamic[s_idx]).clamp(0, 1)
                pair_weight = pair_weight * torch.sqrt((static_t * static_s).clamp_min(0.0))

            weighted_valid = pair_weight * pair_valid.float()
            weighted_consistency_sum += c * weighted_valid
            weighted_consistency_weight += weighted_valid

            # Confidence weight (used only for C_occ / C_unc)
            w = conf * conf_s  # [T, H, W]

            # ---- occlusion check (depth ordering) --------------------------
            occ_flag = (z_proj - depth_s > self.tau_occ).float()
            occ_weighted_sum += occ_flag * w * support_w_map * vk
            weight_sum += w * support_w_map * vk

            # ---- validity (for C_unc) --------------------------------------
            pair_proj_valid = (proj_valid & conf_ok).float()
            valid_weighted_sum += w * pair_proj_valid * support_w_map * vk

            # ---- debug accumulator -----------------------------------------
            r_pt_sum += r_pt * vk
            r_pt_count += vk

        # -- aggregate -------------------------------------------------------

        # C_stat: trimmed mean of *pure geometric* consistency.
        # Pixels with low confidence are excluded from the mean, not penalised.
        C_stat_geom = self._trimmed_mean(
            all_consistency, all_consistency_valid, trim=self.trim_ratio,
        )
        weighted_consistency = torch.where(
            weighted_consistency_weight > 0,
            weighted_consistency_sum / (weighted_consistency_weight + eps),
            C_stat_geom,
        )
        if attn_dynamic is not None and self.use_attention_prior:
            C_stat = torch.clamp(
                (1.0 - self.attn_stat_fusion_weight) * C_stat_geom
                + self.attn_stat_fusion_weight * weighted_consistency,
                0.0,
                1.0,
            )
        else:
            C_stat = C_stat_geom

        # C_occ: confidence-weighted occlusion fraction
        C_occ = (occ_weighted_sum / (weight_sum + eps)).clamp(0, 1)

        # C_unc: two terms blended — support coverage and raw confidence.
        n_support = support_valid.sum(dim=1).float()  # [T]
        if support_score is not None:
            support_mass = support_score.sum(dim=1).float().clamp_min(1e-6)
        else:
            support_mass = n_support.clamp_min(1.0)
        unc_support = 1.0 - valid_weighted_sum / (support_mass[:, None, None] + eps)
        unc_conf = 1.0 - conf.clamp(0, 1)
        C_unc = (
            (1.0 - self.unc_conf_weight) * unc_support
            + self.unc_conf_weight * unc_conf
        ).clamp(0, 1)

        # -- debug dict ------------------------------------------------------
        mean_rpt = r_pt_sum / (r_pt_count + eps)
        debug = {
            "support_count_per_frame": n_support,
            "support_mass_per_frame": support_mass,
            "mean_point_residual": mean_rpt.mean().item(),
            "mean_point_residual_map": mean_rpt,
            "geometry_consistency_mean": C_stat_geom.mean().item(),
            "weighted_consistency_mean": weighted_consistency.mean().item(),
            "stageb_proxy_mode": self.proxy_mode,
        }

        return C_stat, C_stat_geom, C_occ, C_unc, debug

    def _estimate_pinhole_intrinsics(
        self,
        local_pts: torch.Tensor,
        conf: torch.Tensor,
    ) -> torch.Tensor:
        """Fit per-frame pinhole parameters from the predicted pointmap."""
        T, H, W, _ = local_pts.shape
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        intr = torch.zeros(T, 4, dtype=torch.float32)
        default_fx = max(float(W - 1), 1.0)
        default_fy = max(float(H - 1), 1.0)
        default_cx = 0.5 * float(W - 1)
        default_cy = 0.5 * float(H - 1)

        for t in range(T):
            z = local_pts[t, ..., 2]
            valid = (z.abs() > 1e-6) & (conf[t] > self.conf_floor)
            if int(valid.sum().item()) < 16:
                intr[t] = torch.tensor([default_fx, default_fy, default_cx, default_cy])
                continue
            x_norm = (local_pts[t, ..., 0] / z).reshape(-1)
            y_norm = (local_pts[t, ..., 1] / z).reshape(-1)
            u = xx.reshape(-1)
            v = yy.reshape(-1)
            w = conf[t].reshape(-1).clamp_min(0.0)
            m = valid.reshape(-1)
            fx, cx = self._weighted_line_fit(x_norm[m], u[m], w[m], default_fx, default_cx)
            fy, cy = self._weighted_line_fit(y_norm[m], v[m], w[m], default_fy, default_cy)
            intr[t] = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
        return intr

    @staticmethod
    def _weighted_line_fit(
        x: torch.Tensor,
        y: torch.Tensor,
        weight: torch.Tensor,
        default_slope: float,
        default_intercept: float,
    ) -> Tuple[float, float]:
        w = weight.float().clamp_min(0.0)
        w_sum = w.sum()
        if float(w_sum.item()) <= 1e-6:
            return float(default_slope), float(default_intercept)
        mx = (w * x).sum() / w_sum
        my = (w * y).sum() / w_sum
        vx = (w * (x - mx).square()).sum() / w_sum
        if float(vx.item()) <= 1e-8:
            return float(default_slope), float(default_intercept)
        cov = (w * (x - mx) * (y - my)).sum() / w_sum
        slope = cov / vx
        intercept = my - slope * mx
        if not bool(torch.isfinite(slope)) or abs(float(slope.item())) <= 1e-6:
            return float(default_slope), float(default_intercept)
        if not bool(torch.isfinite(intercept)):
            return float(default_slope), float(default_intercept)
        return float(slope.item()), float(intercept.item())

    def _sample_support_reprojection(
        self,
        *,
        world_pts: torch.Tensor,
        local_pts: torch.Tensor,
        conf: torch.Tensor,
        support_index: torch.Tensor,
        X_in_support: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample support-frame maps at projected coordinates."""
        T, H, W, _ = world_pts.shape
        fx = intrinsics[support_index, 0][:, None, None]
        fy = intrinsics[support_index, 1][:, None, None]
        cx = intrinsics[support_index, 2][:, None, None]
        cy = intrinsics[support_index, 3][:, None, None]
        z = X_in_support[..., 2].clamp_min(1e-6)
        u = fx * (X_in_support[..., 0] / z) + cx
        v = fy * (X_in_support[..., 1] / z) + cy
        valid = (
            (X_in_support[..., 2] > 1e-6)
            & (u >= 0.0) & (u <= float(W - 1))
            & (v >= 0.0) & (v <= float(H - 1))
        )
        u_norm = 2.0 * (u / max(float(W - 1), 1.0)) - 1.0
        v_norm = 2.0 * (v / max(float(H - 1), 1.0)) - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1)

        support_world = world_pts[support_index].permute(0, 3, 1, 2)
        support_conf = conf[support_index].unsqueeze(1)
        support_depth = local_pts[support_index, ..., 2].unsqueeze(1)

        sampled_world = F.grid_sample(
            support_world, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        ).permute(0, 2, 3, 1)
        sampled_conf = F.grid_sample(
            support_conf, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(1)
        sampled_depth = F.grid_sample(
            support_depth, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(1)

        return sampled_world, sampled_conf, sampled_depth, valid

    # ---- trimmed mean ------------------------------------------------------

    @staticmethod
    def _trimmed_mean(
        values: torch.Tensor,
        valid: torch.BoolTensor,
        trim: float = 0.2,
    ) -> torch.Tensor:
        """Trimmed mean over axis 0 of ``values`` respecting ``valid``.

        Parameters
        ----------
        values : [K, T, H, W]
        valid  : [K, T, H, W] bool
        trim   : fraction to trim from each tail
        """
        K = values.shape[0]

        # Replace invalid entries with +inf so they sort to the end
        large = values[valid].max().item() + 1.0 if valid.any() else 1.0
        sortable = torch.where(valid, values, torch.tensor(large))
        sorted_vals, _ = sortable.sort(dim=0)  # [K, T, H, W]

        n_valid = valid.sum(dim=0)  # [T, H, W]

        # Trim bounds per pixel
        trim_lo = (n_valid.float() * trim).long()
        trim_hi = (n_valid - trim_lo).clamp(min=1)

        # Index mask: keep entries in [trim_lo, trim_hi)
        k_range = torch.arange(K).reshape(K, 1, 1, 1).expand_as(sorted_vals)
        keep = (
            (k_range >= trim_lo.unsqueeze(0))
            & (k_range < trim_hi.unsqueeze(0))
            & (k_range < n_valid.unsqueeze(0))
        )

        result = (sorted_vals * keep.float()).sum(dim=0) / (keep.sum(dim=0).float() + 1e-7)
        return result

    # ---- single-frame fallback ---------------------------------------------

    def _single_frame_fallback(
        self,
        conf: torch.Tensor,
        patch_grid: Tuple[int, int],
    ) -> CueOutput:
        """Return neutral cues when only one frame is available."""
        T, H, W = conf.shape
        C_stat = conf.clone()
        C_dyn = torch.zeros_like(conf)
        C_occ = torch.zeros_like(conf)
        C_unc = 1.0 - conf.clamp(0, 1)
        C_anchor = C_stat * (1.0 - C_unc)

        E_cue = torch.stack([C_stat, C_dyn, C_occ, C_unc, C_anchor], dim=-1)
        z_geo = (
            self.lambda_s * C_stat
            + self.lambda_a * C_anchor
            - self.lambda_d * C_dyn
            - self.lambda_o * C_occ
            - self.lambda_u * C_unc
        )
        G_write_geo = torch.sigmoid(z_geo)

        E_cue_patch = None
        G_write_geo_patch = None
        if self.compute_patch_cues and patch_grid != (0, 0):
            E_cue_patch, G_write_geo_patch = self._patch_pool(
                E_cue, G_write_geo, patch_grid,
            )

        return CueOutput(
            E_cue=E_cue,
            G_write_geo=G_write_geo,
            C_dyn_explicit=torch.zeros_like(C_dyn),
            C_dyn_implicit=torch.zeros_like(C_dyn),
            C_dyn_fusion_max=C_dyn,
            C_dyn_fusion_soft_or=C_dyn,
            C_dyn_fusion_avg=C_dyn,
            C_dyn_fusion_addclip=C_dyn,
            E_cue_patch=E_cue_patch,
            G_write_geo_patch=G_write_geo_patch,
            num_frames=T,
            spatial_resolution=(H, W),
            patch_grid=patch_grid,
            debug={"support_count_per_frame": torch.zeros(T), "mean_point_residual": 0.0},
        )

    # ---- patch pooling -----------------------------------------------------

    @staticmethod
    def _patch_pool(
        E_cue: torch.Tensor,
        G_write_geo: torch.Tensor,
        patch_grid: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact mean-pooling over PATCH_SIZE × PATCH_SIZE blocks.

        Returns ``(E_cue_patch, G_write_geo_patch)``.
        """
        T, H, W, C = E_cue.shape
        H_tok, W_tok = patch_grid
        pH = H // H_tok
        pW = W // W_tok

        E_patch = E_cue[:, :H_tok * pH, :W_tok * pW, :].reshape(
            T, H_tok, pH, W_tok, pW, C,
        ).mean(dim=(2, 4))  # [T, H_tok, W_tok, C]

        G_patch = G_write_geo[:, :H_tok * pH, :W_tok * pW].reshape(
            T, H_tok, pH, W_tok, pW,
        ).mean(dim=(2, 4))  # [T, H_tok, W_tok]

        return E_patch, G_patch
