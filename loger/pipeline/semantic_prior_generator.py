"""
Stage D: Semantic Prior Generator

Fuses geometric cues (Stage B) and semantic masklets (Stage C) into a
unified token-level write prior ``A_tok`` that the TTT Write Controller
(Stage E) uses to modulate per-token contributions during fast-weight
update.

Three representation layers are produced:
  1. masklet-level  ``A_mask``  [J, T]
  2. pixel-level    ``A_pix``   [T, H_p, W_p]
  3. token-level    ``A_tok``   [L_tok]

Phase 1 uses a **rule-based** (no trainable parameters) formulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .geometry_backbone import (
    GeometryOutput,
    PATCH_SIZE,
    TOKEN_TYPE_PATCH,
    TOKEN_TYPE_REGISTER,
    TOKEN_TYPE_ROLE,
)
from .dynamic_cue_extractor import CueOutput
from .video_masklet_frontend import MaskletOutput, DEFAULT_SEMANTIC_WEIGHTS


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class PriorOutput:
    """Structured output of the Semantic Prior Generator."""

    A_mask: torch.Tensor          # [J, T]   masklet-level write allow
    A_pix: torch.Tensor           # [T, H_p, W_p]  pixel-level write allow
    A_tok: torch.Tensor           # [L_tok]  token-level write prior
    A_patch_flat: torch.Tensor    # [L_patch]

    A_special: float = 1.0        # scalar for special tokens

    debug: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------
class SemanticPriorGenerator:
    """Stage D — rule-based semantic prior (Phase 1, no trainable params).

    Usage::

        gen = SemanticPriorGenerator()
        prior = gen.run(cue, masklet, geo)
    """

    def __init__(
        self,
        *,
        # A_mask linear combination (§7.2 of design doc)
        b_stat: float = 1.0,
        b_anchor: float = 0.8,
        b_dyn: float = 1.0,
        b_occ: float = 0.5,
        b_unc: float = 0.5,
        b_quality: float = 0.3,
        b_sem: float = 0.5,
        b_bias: float = 0.0,
        # Cue-only prior (§8.1)
        c_stat: float = 1.0,
        c_anchor: float = 0.5,
        c_dyn: float = 1.0,
        c_occ: float = 0.3,
        c_unc: float = 0.5,
        c_bias: float = 0.0,
        # Special token prior (§9.2)
        kappa_special: float = 0.5,
        a_min_special: float = 0.3,
        # Layer schedule
        omega_layer: float = 1.0,
        a_min_token: float = 0.05,
    ):
        self.b_stat = b_stat
        self.b_anchor = b_anchor
        self.b_dyn = b_dyn
        self.b_occ = b_occ
        self.b_unc = b_unc
        self.b_quality = b_quality
        self.b_sem = b_sem
        self.b_bias = b_bias

        self.c_stat = c_stat
        self.c_anchor = c_anchor
        self.c_dyn = c_dyn
        self.c_occ = c_occ
        self.c_unc = c_unc
        self.c_bias = c_bias

        self.kappa_special = kappa_special
        self.a_min_special = a_min_special
        self.omega_layer = omega_layer
        self.a_min_token = a_min_token

    # -- public API --------------------------------------------------------

    def run(
        self,
        cue: CueOutput,
        masklet: MaskletOutput,
        geo: GeometryOutput,
    ) -> PriorOutput:
        T = cue.num_frames
        H_p, W_p = cue.spatial_resolution
        H_tok, W_tok = geo.patch_grid

        # =================================================================
        # 1. masklet-level prior  A_mask [J, T]
        # =================================================================
        A_mask = self._compute_masklet_prior(cue, masklet, T, H_p, W_p)

        # =================================================================
        # 2. pixel-level prior  A_pix [T, H_p, W_p]
        # =================================================================
        A_pix = self._compute_pixel_prior(cue, masklet, A_mask, T, H_p, W_p)

        # =================================================================
        # 3. token-level prior  A_tok [L_tok]
        # =================================================================
        A_patch_flat, A_special_val, A_tok = self._compute_token_prior(
            A_pix, geo, T, H_tok, W_tok,
        )

        return PriorOutput(
            A_mask=A_mask,
            A_pix=A_pix,
            A_tok=A_tok,
            A_patch_flat=A_patch_flat,
            A_special=A_special_val,
            debug={
                "rho_suppr_chunk": (1 - A_patch_flat).mean().item(),
            },
        )

    # -- masklet-level prior -----------------------------------------------

    def _compute_masklet_prior(
        self, cue: CueOutput, mo: MaskletOutput,
        T: int, H_p: int, W_p: int,
    ) -> torch.Tensor:
        J = mo.num_masklets
        if J == 0:
            return torch.zeros(0, T)

        C_stat = cue.E_cue[..., 0]   # [T, H_p, W_p]
        C_dyn = cue.E_cue[..., 1]
        C_occ = cue.E_cue[..., 2]
        C_unc = cue.E_cue[..., 3]
        C_anchor = cue.E_cue[..., 4]
        cue_maps = [C_stat, C_dyn, C_occ, C_unc, C_anchor]

        H_mask, W_mask = mo.frame_height, mo.frame_width
        if (H_p, W_p) != (H_mask, W_mask):
            cue_maps = [
                F.interpolate(
                    c.unsqueeze(1), size=(H_mask, W_mask),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
                for c in cue_maps
            ]
        C_stat, C_dyn, C_occ, C_unc, C_anchor = cue_maps

        T_min = min(T, mo.num_frames)
        A_mask = torch.zeros(J, T)

        for j in range(J):
            for t in range(T_min):
                if not mo.V_mask[j, t]:
                    continue
                mask_t = mo.M_mask[j, t].bool()
                n = mask_t.sum().item()
                if n == 0:
                    continue
                bar_stat = C_stat[t][mask_t].mean().item()
                bar_dyn = C_dyn[t][mask_t].mean().item()
                bar_occ = C_occ[t][mask_t].mean().item()
                bar_unc = C_unc[t][mask_t].mean().item()
                bar_anchor = C_anchor[t][mask_t].mean().item()

                g = mo.G_sem[j].item()
                w_sem = DEFAULT_SEMANTIC_WEIGHTS.get(g, 0.15)
                q = mo.Q_mask[j, t].item()

                z = (
                    self.b_bias
                    + self.b_stat * bar_stat
                    + self.b_anchor * bar_anchor
                    - self.b_dyn * bar_dyn
                    - self.b_occ * bar_occ
                    - self.b_unc * bar_unc
                    + self.b_quality * q
                    + self.b_sem * w_sem
                )
                A_mask[j, t] = torch.sigmoid(torch.tensor(z)).item()

        return A_mask

    # -- pixel-level prior -------------------------------------------------

    def _compute_pixel_prior(
        self, cue: CueOutput, mo: MaskletOutput,
        A_mask: torch.Tensor, T: int, H_p: int, W_p: int,
    ) -> torch.Tensor:
        C_stat = cue.E_cue[..., 0]
        C_dyn = cue.E_cue[..., 1]
        C_occ = cue.E_cue[..., 2]
        C_unc = cue.E_cue[..., 3]
        C_anchor = cue.E_cue[..., 4]

        z_cue = (
            self.c_bias
            + self.c_stat * C_stat
            + self.c_anchor * C_anchor
            - self.c_dyn * C_dyn
            - self.c_occ * C_occ
            - self.c_unc * C_unc
        )
        A_pix_cue = torch.sigmoid(z_cue)   # [T, H_p, W_p]

        J = mo.num_masklets
        if J == 0:
            return A_pix_cue

        H_mask, W_mask = mo.frame_height, mo.frame_width
        T_min = min(T, mo.num_frames)

        A_pix_mask = torch.zeros(T, H_p, W_p)
        m_cov = torch.zeros(T, H_p, W_p)

        for j in range(J):
            for t in range(T_min):
                if not mo.V_mask[j, t]:
                    continue
                mask_full = mo.M_mask[j, t]   # [H_mask, W_mask]
                if (H_p, W_p) != (H_mask, W_mask):
                    mask_r = F.interpolate(
                        mask_full.unsqueeze(0).unsqueeze(0),
                        size=(H_p, W_p), mode="bilinear", align_corners=False,
                    ).squeeze(0).squeeze(0)
                else:
                    mask_r = mask_full

                a = A_mask[j, t].item()
                contribution = mask_r * a
                A_pix_mask[t] = torch.max(A_pix_mask[t], contribution)
                m_cov[t] = torch.max(m_cov[t], mask_r)

        A_pix = m_cov * A_pix_mask + (1 - m_cov) * A_pix_cue
        return A_pix.clamp(0, 1)

    # -- token-level prior -------------------------------------------------

    def _compute_token_prior(
        self,
        A_pix: torch.Tensor,
        geo: GeometryOutput,
        T: int,
        H_tok: int,
        W_tok: int,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        pH = A_pix.shape[1] // H_tok
        pW = A_pix.shape[2] // W_tok

        A_patch = A_pix[
            :, :H_tok * pH, :W_tok * pW
        ].reshape(T, H_tok, pH, W_tok, pW).mean(dim=(2, 4))  # [T, H_tok, W_tok]

        A_patch_flat = A_patch.reshape(-1)   # [L_patch]

        # Apply layer schedule + floor
        A_patch_flat = (
            self.omega_layer * A_patch_flat + (1 - self.omega_layer)
        ).clamp(self.a_min_token, 1.0)

        # Special token prior
        rho_suppr = (1 - A_patch_flat).mean().item()
        A_special_val = max(1 - self.kappa_special * rho_suppr, self.a_min_special)

        # Build full A_tok: iterate token_type to assign patch or special prior
        token_type = geo.token_type   # [L_tok]
        L_tok = token_type.shape[0]
        A_tok = torch.full((L_tok,), A_special_val)

        patch_idx = 0
        for i in range(L_tok):
            tt = token_type[i].item()
            if tt == TOKEN_TYPE_PATCH:
                if patch_idx < A_patch_flat.shape[0]:
                    A_tok[i] = A_patch_flat[patch_idx]
                patch_idx += 1

        return A_patch_flat, A_special_val, A_tok
