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
    canonicalize_label,
)

SEMANTIC_ROLE_FALLBACK = 0
SEMANTIC_ROLE_POSITIVE_LONG = 1
SEMANTIC_ROLE_NEUTRAL_KEEP = 2
SEMANTIC_ROLE_NEGATIVE_SHORT = 3
SEMANTIC_ROLE_PROTECT_NEUTRAL = 4

SEMANTIC_ROLE_NAMES = {
    SEMANTIC_ROLE_FALLBACK: "FALLBACK",
    SEMANTIC_ROLE_POSITIVE_LONG: "POSITIVE_LONG",
    SEMANTIC_ROLE_NEUTRAL_KEEP: "NEUTRAL_KEEP",
    SEMANTIC_ROLE_NEGATIVE_SHORT: "NEGATIVE_SHORT",
    SEMANTIC_ROLE_PROTECT_NEUTRAL: "PROTECT_NEUTRAL",
}

SEMANTIC_FINE_LABEL_UNKNOWN = 0
SEMANTIC_FINE_LABEL_TO_ID = {
    "unknown": SEMANTIC_FINE_LABEL_UNKNOWN,
    "road": 1,
    "sidewalk": 2,
    "building": 3,
    "wall": 4,
    "fence": 5,
    "bridge": 6,
    "railing": 7,
    "floor": 8,
    "stair": 9,
    "sky": 20,
    "vegetation": 21,
    "grass": 22,
    "tree": 23,
    "plant": 24,
    "water": 25,
    "cloud": 26,
    "person": 40,
    "people": 40,
    "rider": 41,
    "car": 42,
    "bus": 43,
    "truck": 44,
    "bicycle": 45,
    "motorcycle": 46,
    "animal": 47,
}
SEMANTIC_FINE_ID_TO_LABEL = {
    int(v): str(k)
    for k, v in SEMANTIC_FINE_LABEL_TO_ID.items()
    if int(v) not in {40} or str(k) == "person"
}

SEMANTIC_FINE_STRUCTURE_IDS = {
    SEMANTIC_FINE_LABEL_TO_ID[k]
    for k in ("road", "sidewalk", "building", "wall", "fence", "bridge", "railing", "floor", "stair")
}
SEMANTIC_FINE_SKY_IDS = {SEMANTIC_FINE_LABEL_TO_ID["sky"], SEMANTIC_FINE_LABEL_TO_ID["cloud"]}
SEMANTIC_FINE_VEGETATION_IDS = {
    SEMANTIC_FINE_LABEL_TO_ID[k]
    for k in ("vegetation", "grass", "tree", "plant")
}
SEMANTIC_FINE_MOVABLE_IDS = {
    SEMANTIC_FINE_LABEL_TO_ID[k]
    for k in ("person", "rider", "car", "bus", "truck", "bicycle", "motorcycle", "animal")
}


def semantic_fine_label_id(label: str) -> int:
    """Return a stable integer id for a Stage-C canonical fine label."""

    try:
        canonical = canonicalize_label(str(label))
    except Exception:
        canonical = str(label).strip().lower()
    return int(SEMANTIC_FINE_LABEL_TO_ID.get(canonical, SEMANTIC_FINE_LABEL_UNKNOWN))


def semantic_fine_label_name(label_id: int) -> str:
    return SEMANTIC_FINE_ID_TO_LABEL.get(int(label_id), "unknown")


def semantic_fine_label_ids_from_masklets(mo: MaskletOutput) -> torch.Tensor:
    labels = list(getattr(mo, "L_sem", []) or [])
    ids = [semantic_fine_label_id(labels[j]) if j < len(labels) else SEMANTIC_FINE_LABEL_UNKNOWN for j in range(int(mo.num_masklets))]
    return torch.tensor(ids, dtype=torch.long)


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
    G_sem_patch_flat: Optional[torch.Tensor] = None
    Q_sem_patch_flat: Optional[torch.Tensor] = None
    L_sem_patch_flat: Optional[torch.Tensor] = None
    R_sem_patch_flat: Optional[torch.Tensor] = None
    R_frame_patch_flat: Optional[torch.Tensor] = None
    R_global_patch_flat: Optional[torch.Tensor] = None
    R_swa_patch_flat: Optional[torch.Tensor] = None
    R_ttt_patch_flat: Optional[torch.Tensor] = None
    G_sem_tok: Optional[torch.Tensor] = None
    Q_sem_tok: Optional[torch.Tensor] = None
    L_sem_tok: Optional[torch.Tensor] = None
    V_sem_tok: Optional[torch.Tensor] = None
    R_sem_tok: Optional[torch.Tensor] = None
    R_frame_tok: Optional[torch.Tensor] = None
    R_global_tok: Optional[torch.Tensor] = None
    R_swa_tok: Optional[torch.Tensor] = None
    R_ttt_tok: Optional[torch.Tensor] = None

    B_chunk_geo: float = 0.0
    A_special: float = 1.0

    debug: Dict[str, Any] = field(default_factory=dict)


def project_masklet_semantic_groups(
    mo: MaskletOutput,
    geo: GeometryOutput,
    *,
    num_frames: int,
    pixel_resolution: Tuple[int, int],
    patch_grid: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    """Project exact Stage-C masklet semantic IDs to patch and token layouts.

    This is intentionally discrete: it carries ``MaskletOutput.G_sem`` through
    as group IDs and ``MaskletOutput.L_sem`` as stable fine label IDs instead
    of inferring semantic roles from the scalar semantic value prior.
    """

    T = int(num_frames)
    H_p, W_p = int(pixel_resolution[0]), int(pixel_resolution[1])
    H_tok, W_tok = int(patch_grid[0]), int(patch_grid[1])
    if H_tok <= 0 or W_tok <= 0:
        raise ValueError(f"Invalid patch grid: {(H_tok, W_tok)}")

    group_patch = torch.full(
        (T, H_tok, W_tok),
        int(SEMANTIC_GROUP_UNCERTAIN_REGION),
        dtype=torch.long,
    )
    quality_patch = torch.zeros((T, H_tok, W_tok), dtype=torch.float32)
    label_patch = torch.full((T, H_tok, W_tok), -1, dtype=torch.long)
    best_score = torch.zeros((T, H_tok, W_tok), dtype=torch.float32)

    J = int(mo.num_masklets)
    T_use = min(T, int(mo.num_frames))
    if J > 0 and T_use > 0:
        H_mask, W_mask = int(mo.frame_height), int(mo.frame_width)
        groups = mo.G_sem.detach().cpu().long().reshape(-1)
        label_ids = semantic_fine_label_ids_from_masklets(mo)
        trust = (mo.V_mask.detach().cpu().float() * mo.Q_mask.detach().cpu().float()).clamp(0.0, 1.0)
        for j in range(J):
            group_id = int(groups[j].item()) if j < int(groups.numel()) else int(SEMANTIC_GROUP_UNCERTAIN_REGION)
            label_id = int(label_ids[j].item()) if j < int(label_ids.numel()) else int(SEMANTIC_FINE_LABEL_UNKNOWN)
            for t in range(T_use):
                if not bool(mo.V_mask[j, t]):
                    continue
                mask_t = mo.M_mask[j, t].detach().cpu().float()
                if not bool(mask_t.bool().any()):
                    continue
                if (H_mask, W_mask) != (H_p, W_p):
                    mask_t = F.interpolate(
                        mask_t.unsqueeze(0).unsqueeze(0),
                        size=(H_p, W_p),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                mask_patch = F.interpolate(
                    mask_t.unsqueeze(0).unsqueeze(0),
                    size=(H_tok, W_tok),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0).clamp(0.0, 1.0)
                q = float(trust[j, t].item()) if j < trust.shape[0] and t < trust.shape[1] else 0.0
                score = mask_patch * q
                update = score > best_score[t]
                best_score[t] = torch.where(update, score, best_score[t])
                group_patch[t] = torch.where(update, torch.full_like(group_patch[t], group_id), group_patch[t])
                quality_patch[t] = torch.where(update, torch.full_like(quality_patch[t], q), quality_patch[t])
                label_patch[t] = torch.where(update, torch.full_like(label_patch[t], label_id), label_patch[t])

    G_patch_flat = group_patch.reshape(-1).long()
    Q_patch_flat = quality_patch.reshape(-1).float().clamp(0.0, 1.0)
    L_patch_flat = label_patch.reshape(-1).long()
    R_patch_flat = semantic_roles_from_groups(G_patch_flat, Q_patch_flat)
    path_roles = semantic_path_role_priors_from_fine_labels(L_patch_flat, G_patch_flat, Q_patch_flat)

    token_type = geo.token_type.detach().cpu().long()
    L_tok = int(token_type.numel())
    G_tok = torch.full((L_tok,), int(SEMANTIC_GROUP_UNCERTAIN_REGION), dtype=torch.long)
    Q_tok = torch.zeros((L_tok,), dtype=torch.float32)
    L_label_tok = torch.full((L_tok,), -1, dtype=torch.long)
    R_tok = torch.full((L_tok,), int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
    R_frame_tok = torch.full((L_tok,), int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
    R_global_tok = torch.full((L_tok,), int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
    R_swa_tok = torch.full((L_tok,), int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
    R_ttt_tok = torch.full((L_tok,), int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
    patch_idx = 0
    for i in range(L_tok):
        if int(token_type[i].item()) == TOKEN_TYPE_PATCH:
            if patch_idx < int(G_patch_flat.numel()):
                G_tok[i] = G_patch_flat[patch_idx]
                Q_tok[i] = Q_patch_flat[patch_idx]
                L_label_tok[i] = L_patch_flat[patch_idx]
                R_tok[i] = R_patch_flat[patch_idx]
                R_frame_tok[i] = path_roles["R_frame_patch_flat"][patch_idx]
                R_global_tok[i] = path_roles["R_global_patch_flat"][patch_idx]
                R_swa_tok[i] = path_roles["R_swa_patch_flat"][patch_idx]
                R_ttt_tok[i] = path_roles["R_ttt_patch_flat"][patch_idx]
            patch_idx += 1

    return {
        "G_sem_patch_flat": G_patch_flat,
        "Q_sem_patch_flat": Q_patch_flat,
        "L_sem_patch_flat": L_patch_flat,
        "R_sem_patch_flat": R_patch_flat,
        "R_frame_patch_flat": path_roles["R_frame_patch_flat"],
        "R_global_patch_flat": path_roles["R_global_patch_flat"],
        "R_swa_patch_flat": path_roles["R_swa_patch_flat"],
        "R_ttt_patch_flat": path_roles["R_ttt_patch_flat"],
        "G_sem_tok": G_tok,
        "Q_sem_tok": Q_tok,
        "L_sem_tok": L_label_tok,
        "R_sem_tok": R_tok,
        "R_frame_tok": R_frame_tok,
        "R_global_tok": R_global_tok,
        "R_swa_tok": R_swa_tok,
        "R_ttt_tok": R_ttt_tok,
    }


def semantic_values_from_groups(groups: torch.Tensor) -> torch.Tensor:
    """Default coarse semantic value used for role diagnostics.

    Runtime experiments may override the scalar value branch, but the fixed
    map here keeps v23 role logging auditable even in pass-through smoke runs.
    """

    G = groups.detach().cpu().long()
    out = torch.full_like(G, 0.4, dtype=torch.float32)
    out[G == SEMANTIC_GROUP_STRUCTURE_ANCHOR] = 1.0
    out[G == SEMANTIC_GROUP_STATIC_THING] = 0.7
    out[G == SEMANTIC_GROUP_LOW_VALUE_STUFF] = 0.4
    out[G == SEMANTIC_GROUP_MOVABLE_THING] = 0.1
    out[G == SEMANTIC_GROUP_UNCERTAIN_REGION] = 0.4
    return out.clamp(0.0, 1.0)


def semantic_roles_from_groups(groups: torch.Tensor, trust: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Coarse fallback role table before geometry-conditioned refinement.

    HMC refines these roles with dynamic/attention risk.  This helper provides
    a non-empty, token-aligned role stream for no-op and debug-only v23 audits.
    """

    G = groups.detach().cpu().long()
    R = torch.full_like(G, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
    R[G == SEMANTIC_GROUP_STRUCTURE_ANCHOR] = int(SEMANTIC_ROLE_PROTECT_NEUTRAL)
    R[G == SEMANTIC_GROUP_STATIC_THING] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
    R[G == SEMANTIC_GROUP_LOW_VALUE_STUFF] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
    R[G == SEMANTIC_GROUP_MOVABLE_THING] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
    R[G == SEMANTIC_GROUP_UNCERTAIN_REGION] = int(SEMANTIC_ROLE_FALLBACK)
    if trust is not None:
        Q = trust.detach().cpu().float().reshape(-1)
        if Q.numel() == R.numel():
            R[Q < 0.05] = int(SEMANTIC_ROLE_FALLBACK)
    return R


def semantic_path_role_priors_from_fine_labels(
    labels: torch.Tensor,
    groups: torch.Tensor,
    trust: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Fine-label role priors before HMC adds D/conflict/scale conditions."""

    L = labels.detach().cpu().long().reshape(-1)
    G = groups.detach().cpu().long().reshape(-1)
    R_frame = torch.full_like(L, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
    R_global = torch.full_like(L, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
    R_swa = torch.full_like(L, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
    R_ttt = torch.full_like(L, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)

    known = L >= 0
    structure = torch.zeros_like(known, dtype=torch.bool)
    sky = torch.zeros_like(known, dtype=torch.bool)
    vegetation = torch.zeros_like(known, dtype=torch.bool)
    movable = torch.zeros_like(known, dtype=torch.bool)
    for label_id in SEMANTIC_FINE_STRUCTURE_IDS:
        structure |= L == int(label_id)
    for label_id in SEMANTIC_FINE_SKY_IDS:
        sky |= L == int(label_id)
    for label_id in SEMANTIC_FINE_VEGETATION_IDS:
        vegetation |= L == int(label_id)
    for label_id in SEMANTIC_FINE_MOVABLE_IDS:
        movable |= L == int(label_id)

    # Coarse fallback keeps legacy behavior auditable for labels not in the
    # explicit fine taxonomy.
    coarse_structure = G == int(SEMANTIC_GROUP_STRUCTURE_ANCHOR)
    coarse_lowstuff = G == int(SEMANTIC_GROUP_LOW_VALUE_STUFF)
    coarse_movable = G == int(SEMANTIC_GROUP_MOVABLE_THING)
    structure |= coarse_structure & ~sky & ~vegetation
    vegetation |= coarse_lowstuff & ~sky
    movable |= coarse_movable

    R_frame[structure] = int(SEMANTIC_ROLE_PROTECT_NEUTRAL)
    R_global[structure] = int(SEMANTIC_ROLE_PROTECT_NEUTRAL)
    R_swa[structure] = int(SEMANTIC_ROLE_PROTECT_NEUTRAL)
    R_ttt[structure] = int(SEMANTIC_ROLE_POSITIVE_LONG)

    R_frame[sky] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
    R_global[sky] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
    R_swa[sky] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
    R_ttt[sky] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)

    weak_context = vegetation & ~sky
    R_frame[weak_context] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
    R_global[weak_context] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
    R_swa[weak_context] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
    R_ttt[weak_context] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)

    R_frame[movable] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
    R_global[movable] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
    R_swa[movable] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
    R_ttt[movable] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)

    if trust is not None:
        Q = trust.detach().cpu().float().reshape(-1)
        if Q.numel() == L.numel():
            low_trust = Q < 0.05
            R_frame[low_trust] = int(SEMANTIC_ROLE_FALLBACK)
            R_global[low_trust] = int(SEMANTIC_ROLE_FALLBACK)
            R_swa[low_trust] = int(SEMANTIC_ROLE_FALLBACK)
            R_ttt[low_trust] = int(SEMANTIC_ROLE_FALLBACK)
    unknown = (~known) | (L == int(SEMANTIC_FINE_LABEL_UNKNOWN))
    R_frame[unknown] = int(SEMANTIC_ROLE_FALLBACK)
    R_global[unknown] = int(SEMANTIC_ROLE_FALLBACK)
    R_swa[unknown] = int(SEMANTIC_ROLE_FALLBACK)
    R_ttt[unknown] = int(SEMANTIC_ROLE_FALLBACK)
    return {
        "R_frame_patch_flat": R_frame,
        "R_global_patch_flat": R_global,
        "R_swa_patch_flat": R_swa,
        "R_ttt_patch_flat": R_ttt,
    }


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
        group_projection = project_masklet_semantic_groups(
            masklet,
            geo,
            num_frames=T,
            pixel_resolution=(H_p, W_p),
            patch_grid=(H_tok, W_tok),
        )
        V_sem_tok = torch.full((int(geo.token_type.numel()),), 1.0, dtype=torch.float32)
        token_type = geo.token_type.detach().cpu().long()
        patch_mask = token_type == TOKEN_TYPE_PATCH
        n_patch = int(patch_mask.sum().item())
        if n_patch > 0:
            vals = V_sem_patch_flat.detach().cpu().float().reshape(-1)
            padded = torch.ones((n_patch,), dtype=torch.float32)
            padded[: min(n_patch, int(vals.numel()))] = vals[:n_patch]
            V_sem_tok[patch_mask] = padded

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
            G_sem_patch_flat=group_projection["G_sem_patch_flat"],
            Q_sem_patch_flat=group_projection["Q_sem_patch_flat"],
            L_sem_patch_flat=group_projection["L_sem_patch_flat"],
            R_sem_patch_flat=group_projection["R_sem_patch_flat"],
            R_frame_patch_flat=group_projection["R_frame_patch_flat"],
            R_global_patch_flat=group_projection["R_global_patch_flat"],
            R_swa_patch_flat=group_projection["R_swa_patch_flat"],
            R_ttt_patch_flat=group_projection["R_ttt_patch_flat"],
            G_sem_tok=group_projection["G_sem_tok"],
            Q_sem_tok=group_projection["Q_sem_tok"],
            L_sem_tok=group_projection["L_sem_tok"],
            V_sem_tok=V_sem_tok.clamp(0.0, 1.0),
            R_sem_tok=group_projection["R_sem_tok"],
            R_frame_tok=group_projection["R_frame_tok"],
            R_global_tok=group_projection["R_global_tok"],
            R_swa_tok=group_projection["R_swa_tok"],
            R_ttt_tok=group_projection["R_ttt_tok"],
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
                "semantic_group_taxonomy": "stage_c_coarse_5_groups",
                "semantic_fine_label_available": True,
                "semantic_fine_label_exact_source": "MaskletOutput.L_sem",
                "semantic_group_exact_source": "MaskletOutput.G_sem",
                "semantic_group_token_count": int(group_projection["G_sem_tok"].numel()),
                "semantic_role_token_count": int(group_projection["R_sem_tok"].numel()),
                "semantic_role_policy": "fine_label_path_prior",
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
