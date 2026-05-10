"""
Stage D: Semantic Prior Generator

v2 redesign: decoupled write policy

  - Geometry Eligibility Branch -> ``Elig_pix``
  - Semantic Value Branch       -> ``v_sem``
  - Mask Trust Branch           -> ``r_mask``

The final absolute write gate is produced via:

  ``A_mask`` -> ``A_pix`` -> ``A_patch_flat`` / ``A_special`` -> ``A_tok``

Chunk-level write budget is carried separately as ``B_chunk_geo`` and is
consumed by Stage E.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .geometry_backbone import GeometryOutput, TOKEN_TYPE_PATCH
from .dynamic_cue_extractor import CueOutput
from .video_masklet_frontend import (
    MaskletOutput,
    SEMANTIC_GROUP_LOW_VALUE_STUFF,
    SEMANTIC_GROUP_MOVABLE_THING,
    SEMANTIC_GROUP_STATIC_THING,
    SEMANTIC_GROUP_STRUCTURE_ANCHOR,
    SEMANTIC_GROUP_UNCERTAIN_REGION,
)


@dataclass
class PriorOutput:
    """Structured output of the Semantic Prior Generator."""

    A_mask: torch.Tensor
    A_pix: torch.Tensor
    A_tok: torch.Tensor
    A_patch_flat: torch.Tensor

    Elig_pix: torch.Tensor
    r_mask: torch.Tensor
    E_patch_flat: torch.Tensor
    V_sem_patch_flat: Optional[torch.Tensor] = None
    R_mask_patch_flat: Optional[torch.Tensor] = None

    B_chunk_geo: float = 0.0
    A_special: float = 1.0

    debug: Dict[str, Any] = field(default_factory=dict)


class SemanticPriorGenerator:
    """Stage D — v2 decoupled semantic prior (rule-based, no trainable params)."""

    def __init__(
        self,
        *,
        use_g_write_geo: bool = True,
        k_pos: float = 1.5,
        k_risk: float = 3.0,
        b_elig: float = 0.0,
        rho_sem: float = 0.6,
        a_min_special: float = 0.3,
        a_token_floor: float = 0.0,
        value_structure: float = 1.0,
        value_background: float = 0.7,
        value_distractor: float = 0.4,
        value_movable: float = 0.1,
        value_uncertain: float = 0.4,
    ):
        self.use_g_write_geo = use_g_write_geo
        self.k_pos = float(k_pos)
        self.k_risk = float(k_risk)
        self.b_elig = float(b_elig)
        self.rho_sem = float(rho_sem)
        self.a_min_special = float(a_min_special)
        self.a_token_floor = float(a_token_floor)

        self.value_structure = float(value_structure)
        self.value_background = float(value_background)
        self.value_distractor = float(value_distractor)
        self.value_movable = float(value_movable)
        self.value_uncertain = float(value_uncertain)

    # -- public API ----------------------------------------------------

    def run(
        self,
        cue: CueOutput,
        masklet: MaskletOutput,
        geo: GeometryOutput,
    ) -> PriorOutput:
        T = cue.num_frames
        H_p, W_p = cue.spatial_resolution
        H_tok, W_tok = geo.patch_grid

        Elig_pix = self._compute_geometry_eligibility(cue)
        v_sem = self._compute_semantic_value(masklet)
        r_mask = self._compute_mask_trust(masklet, T)
        A_mask = self._compute_masklet_gate(
            Elig_pix=Elig_pix,
            mo=masklet,
            v_sem=v_sem,
            T=T,
            H_p=H_p,
            W_p=W_p,
        )
        A_pix, V_sem_pix, R_mask_pix = self._compute_pixel_prior(
            Elig_pix=Elig_pix,
            mo=masklet,
            A_mask=A_mask,
            v_sem=v_sem,
            r_mask=r_mask,
            T=T,
            H_p=H_p,
            W_p=W_p,
        )
        A_patch_flat, E_patch_flat, A_special, B_chunk_geo, A_tok = self._compute_token_prior(
            A_pix=A_pix,
            Elig_pix=Elig_pix,
            cue=cue,
            geo=geo,
            H_tok=H_tok,
            W_tok=W_tok,
        )
        V_sem_patch_flat = self._pool_to_patch(V_sem_pix, H_tok, W_tok).reshape(-1).float().clamp(0.0, 1.0)
        R_mask_patch_flat = self._pool_to_patch(R_mask_pix, H_tok, W_tok).reshape(-1).float().clamp(0.0, 1.0)

        return PriorOutput(
            A_mask=A_mask,
            A_pix=A_pix,
            A_tok=A_tok,
            A_patch_flat=A_patch_flat,
            Elig_pix=Elig_pix,
            r_mask=r_mask,
            E_patch_flat=E_patch_flat,
            V_sem_patch_flat=V_sem_patch_flat,
            R_mask_patch_flat=R_mask_patch_flat,
            B_chunk_geo=B_chunk_geo,
            A_special=A_special,
            debug={
                "v_sem": v_sem,
                "rho_sem": self.rho_sem,
                "rho_suppr_chunk": float(1.0 - A_patch_flat.mean().item()) if A_patch_flat.numel() > 0 else 0.0,
                "mean_elig": float(Elig_pix.mean().item()) if Elig_pix.numel() > 0 else 0.0,
                "mean_a_pix": float(A_pix.mean().item()) if A_pix.numel() > 0 else 0.0,
                "mean_r_mask": float(r_mask.mean().item()) if r_mask.numel() > 0 else 0.0,
                "mean_v_sem_patch": float(V_sem_patch_flat.mean().item()) if V_sem_patch_flat.numel() > 0 else 1.0,
                "mean_r_mask_patch": float(R_mask_patch_flat.mean().item()) if R_mask_patch_flat.numel() > 0 else 0.0,
                "a_token_floor": self.a_token_floor,
            },
        )

    # -- branch 1: geometry eligibility --------------------------------

    def _compute_geometry_eligibility(self, cue: CueOutput) -> torch.Tensor:
        if self.use_g_write_geo and cue.G_write_geo is not None:
            return cue.G_write_geo.float().clamp(0.0, 1.0)

        c_stat = cue.E_cue[..., 0]
        c_dyn = cue.E_cue[..., 1]
        c_occ = cue.E_cue[..., 2]
        c_unc = cue.E_cue[..., 3]
        c_anchor = cue.E_cue[..., 4]

        p_pos = 0.5 * c_stat + 0.5 * c_anchor
        p_risk = 0.5 * c_dyn + 0.25 * c_occ + 0.25 * c_unc
        return torch.sigmoid(self.k_pos * p_pos - self.k_risk * p_risk + self.b_elig)

    # -- branch 2: semantic value --------------------------------------

    def _compute_semantic_value(self, mo: MaskletOutput) -> torch.Tensor:
        J = mo.num_masklets
        if J == 0:
            return torch.zeros(0, dtype=torch.float32)

        v_sem = torch.full((J,), self.value_uncertain, dtype=torch.float32)
        groups = mo.G_sem.to(dtype=torch.long)
        v_sem[groups == SEMANTIC_GROUP_STRUCTURE_ANCHOR] = self.value_structure
        v_sem[groups == SEMANTIC_GROUP_STATIC_THING] = self.value_background
        v_sem[groups == SEMANTIC_GROUP_LOW_VALUE_STUFF] = self.value_distractor
        v_sem[groups == SEMANTIC_GROUP_MOVABLE_THING] = self.value_movable
        v_sem[groups == SEMANTIC_GROUP_UNCERTAIN_REGION] = self.value_uncertain
        return v_sem

    # -- branch 3: mask trust ------------------------------------------

    def _compute_mask_trust(self, mo: MaskletOutput, T: int) -> torch.Tensor:
        J = mo.num_masklets
        if J == 0:
            return torch.zeros(0, T, dtype=torch.float32)

        T_use = min(T, mo.num_frames)
        r_mask = torch.zeros(J, T, dtype=torch.float32)
        r_mask[:, :T_use] = mo.V_mask[:, :T_use].float() * mo.Q_mask[:, :T_use].float()
        return r_mask.clamp(0.0, 1.0)

    # -- masklet gate ---------------------------------------------------

    def _compute_masklet_gate(
        self,
        *,
        Elig_pix: torch.Tensor,
        mo: MaskletOutput,
        v_sem: torch.Tensor,
        T: int,
        H_p: int,
        W_p: int,
    ) -> torch.Tensor:
        J = mo.num_masklets
        if J == 0:
            return torch.zeros(0, T, dtype=torch.float32)

        H_mask, W_mask = mo.frame_height, mo.frame_width
        elig_for_mean = Elig_pix
        if (H_p, W_p) != (H_mask, W_mask):
            elig_for_mean = F.interpolate(
                Elig_pix.unsqueeze(1),
                size=(H_mask, W_mask),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        T_use = min(T, mo.num_frames)
        A_mask = torch.zeros(J, T, dtype=torch.float32)
        sem_mod = ((1.0 - self.rho_sem) + self.rho_sem * v_sem).clamp(0.0, 1.0)

        for j in range(J):
            for t in range(T_use):
                if not bool(mo.V_mask[j, t]):
                    continue
                mask_t = mo.M_mask[j, t].bool()
                if not bool(mask_t.any()):
                    continue
                e_bar = float(elig_for_mean[t][mask_t].mean().item())
                A_mask[j, t] = torch.clamp(
                    torch.tensor(e_bar * float(sem_mod[j].item()), dtype=torch.float32),
                    0.0,
                    1.0,
                )

        return A_mask

    # -- pixel fusion ---------------------------------------------------

    def _compute_pixel_prior(
        self,
        *,
        Elig_pix: torch.Tensor,
        mo: MaskletOutput,
        A_mask: torch.Tensor,
        v_sem: torch.Tensor,
        r_mask: torch.Tensor,
        T: int,
        H_p: int,
        W_p: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        J = mo.num_masklets
        if J == 0:
            return Elig_pix.clone(), torch.ones_like(Elig_pix), torch.zeros_like(Elig_pix)

        T_use = min(T, mo.num_frames)
        H_mask, W_mask = mo.frame_height, mo.frame_width

        best_score = torch.zeros(T, H_p, W_p, dtype=torch.float32)
        A_sem_pix = torch.zeros(T, H_p, W_p, dtype=torch.float32)
        V_sem_pix = torch.ones(T, H_p, W_p, dtype=torch.float32)
        R_mask_pix = torch.zeros(T, H_p, W_p, dtype=torch.float32)

        for j in range(J):
            for t in range(T_use):
                if not bool(mo.V_mask[j, t]):
                    continue
                mask_t = mo.M_mask[j, t].float()
                if not bool(mask_t.bool().any()):
                    continue
                if (H_p, W_p) != (H_mask, W_mask):
                    mask_r = F.interpolate(
                        mask_t.unsqueeze(0).unsqueeze(0),
                        size=(H_p, W_p),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                else:
                    mask_r = mask_t

                score = mask_r * float(r_mask[j, t].item()) * float(A_mask[j, t].item())
                update = score > best_score[t]
                best_score[t] = torch.where(update, score, best_score[t])
                A_sem_pix[t] = torch.where(
                    update,
                    torch.full_like(A_sem_pix[t], float(A_mask[j, t].item())),
                    A_sem_pix[t],
                )
                V_sem_pix[t] = torch.where(
                    update,
                    torch.full_like(V_sem_pix[t], float(v_sem[j].item())),
                    V_sem_pix[t],
                )
                R_mask_pix[t] = torch.where(
                    update,
                    torch.full_like(R_mask_pix[t], float(r_mask[j, t].item())),
                    R_mask_pix[t],
                )

        A_pix = R_mask_pix * A_sem_pix + (1.0 - R_mask_pix) * Elig_pix
        return A_pix.clamp(0.0, 1.0), V_sem_pix.clamp(0.0, 1.0), R_mask_pix.clamp(0.0, 1.0)

    # -- token projection ------------------------------------------------

    def _compute_token_prior(
        self,
        *,
        A_pix: torch.Tensor,
        Elig_pix: torch.Tensor,
        cue: CueOutput,
        geo: GeometryOutput,
        H_tok: int,
        W_tok: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float, torch.Tensor]:
        A_patch = self._pool_to_patch(A_pix, H_tok, W_tok)
        A_patch_flat = A_patch.reshape(-1).float().clamp(0.0, 1.0)

        if cue.G_write_geo_patch is not None and tuple(cue.patch_grid) == (H_tok, W_tok):
            E_patch = cue.G_write_geo_patch.float()
        else:
            E_patch = self._pool_to_patch(Elig_pix, H_tok, W_tok)
        E_patch_flat = E_patch.reshape(-1).float().clamp(0.0, 1.0)

        B_chunk_geo = float(E_patch_flat.mean().item()) if E_patch_flat.numel() > 0 else 0.0
        A_special = float(
            torch.clamp(
                torch.tensor(
                    self.a_min_special + (1.0 - self.a_min_special) * B_chunk_geo,
                    dtype=torch.float32,
                ),
                self.a_min_special,
                1.0,
            ).item()
        )

        token_type = geo.token_type
        L_tok = int(token_type.shape[0])
        A_tok = torch.full((L_tok,), A_special, dtype=torch.float32)

        patch_idx = 0
        for i in range(L_tok):
            if int(token_type[i].item()) == TOKEN_TYPE_PATCH:
                if patch_idx < A_patch_flat.numel():
                    A_tok[i] = A_patch_flat[patch_idx]
                patch_idx += 1

        if self.a_token_floor > 0.0:
            floor = float(max(0.0, min(1.0, self.a_token_floor)))
            A_tok = A_tok.clamp_min(floor)

        return A_patch_flat, E_patch_flat, A_special, B_chunk_geo, A_tok

    # -- utilities ------------------------------------------------------

    def _pool_to_patch(
        self,
        pix_map: torch.Tensor,
        H_tok: int,
        W_tok: int,
    ) -> torch.Tensor:
        T, H_p, W_p = pix_map.shape
        if H_tok <= 0 or W_tok <= 0:
            raise ValueError(f"Invalid patch grid: {(H_tok, W_tok)}")

        pH = max(H_p // H_tok, 1)
        pW = max(W_p // W_tok, 1)
        H_trim = H_tok * pH
        W_trim = W_tok * pW

        pooled = pix_map[:, :H_trim, :W_trim].reshape(
            T, H_tok, pH, W_tok, pW,
        ).mean(dim=(2, 4))
        return pooled
