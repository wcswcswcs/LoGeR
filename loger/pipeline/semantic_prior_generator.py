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
    label_to_group,
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
    "ground": 10,
    "crosswalk": 11,
    "pole": 12,
    "traffic sign": 13,
    "billboard": 14,
    "house": 15,
    "sky": 20,
    "vegetation": 21,
    "grass": 22,
    "tree": 23,
    "plant": 24,
    "water": 25,
    "cloud": 26,
    "terrain": 27,
    "mountain": 28,
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
    for k in ("road", "sidewalk", "building", "wall", "fence", "bridge", "railing", "floor", "stair", "ground", "crosswalk", "pole", "traffic sign", "house")
}
SEMANTIC_FINE_SKY_IDS = {SEMANTIC_FINE_LABEL_TO_ID["sky"], SEMANTIC_FINE_LABEL_TO_ID["cloud"]}
SEMANTIC_FINE_VEGETATION_IDS = {
    SEMANTIC_FINE_LABEL_TO_ID[k]
    for k in ("vegetation", "grass", "tree", "plant", "terrain", "mountain")
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
    stage_c_seed_global_track_idx_patch_flat: Optional[torch.Tensor] = None
    stage_c_seed_global_track_idx_tok: Optional[torch.Tensor] = None
    stage_c_masklet_instance_idx_patch_flat: Optional[torch.Tensor] = None
    stage_c_masklet_instance_idx_tok: Optional[torch.Tensor] = None

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
    seed_patch = torch.full((T, H_tok, W_tok), -1, dtype=torch.long)
    masklet_instance_patch = torch.full((T, H_tok, W_tok), -1, dtype=torch.long)
    best_score = torch.zeros((T, H_tok, W_tok), dtype=torch.float32)

    J = int(mo.num_masklets)
    T_use = min(T, int(mo.num_frames))
    if J > 0 and T_use > 0:
        H_mask, W_mask = int(mo.frame_height), int(mo.frame_width)
        groups = mo.G_sem.detach().cpu().long().reshape(-1)
        label_ids = semantic_fine_label_ids_from_masklets(mo)
        seed_ids = list(getattr(mo, "seed_global_track_idx", []) or [])
        trust = (mo.V_mask.detach().cpu().float() * mo.Q_mask.detach().cpu().float()).clamp(0.0, 1.0)
        for j in range(J):
            group_id = int(groups[j].item()) if j < int(groups.numel()) else int(SEMANTIC_GROUP_UNCERTAIN_REGION)
            label_id = int(label_ids[j].item()) if j < int(label_ids.numel()) else int(SEMANTIC_FINE_LABEL_UNKNOWN)
            try:
                seed_id = int(seed_ids[j]) if j < len(seed_ids) and seed_ids[j] is not None else -1
            except (TypeError, ValueError):
                seed_id = -1
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
                seed_patch[t] = torch.where(update, torch.full_like(seed_patch[t], seed_id), seed_patch[t])
                masklet_instance_patch[t] = torch.where(
                    update,
                    torch.full_like(masklet_instance_patch[t], int(j)),
                    masklet_instance_patch[t],
                )

    G_patch_flat = group_patch.reshape(-1).long()
    Q_patch_flat = quality_patch.reshape(-1).float().clamp(0.0, 1.0)
    L_patch_flat = label_patch.reshape(-1).long()
    seed_patch_flat = seed_patch.reshape(-1).long()
    masklet_instance_patch_flat = masklet_instance_patch.reshape(-1).long()
    R_patch_flat = semantic_roles_from_groups(G_patch_flat, Q_patch_flat)
    path_roles = semantic_path_role_priors_from_fine_labels(L_patch_flat, G_patch_flat, Q_patch_flat)

    token_type = geo.token_type.detach().cpu().long()
    L_tok = int(token_type.numel())
    G_tok = torch.full((L_tok,), int(SEMANTIC_GROUP_UNCERTAIN_REGION), dtype=torch.long)
    Q_tok = torch.zeros((L_tok,), dtype=torch.float32)
    L_label_tok = torch.full((L_tok,), -1, dtype=torch.long)
    seed_tok = torch.full((L_tok,), -1, dtype=torch.long)
    masklet_instance_tok = torch.full((L_tok,), -1, dtype=torch.long)
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
                seed_tok[i] = seed_patch_flat[patch_idx]
                masklet_instance_tok[i] = masklet_instance_patch_flat[patch_idx]
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
        "stage_c_seed_global_track_idx_patch_flat": seed_patch_flat,
        "stage_c_seed_global_track_idx_tok": seed_tok,
        "stage_c_masklet_instance_idx_patch_flat": masklet_instance_patch_flat,
        "stage_c_masklet_instance_idx_tok": masklet_instance_tok,
        "R_sem_tok": R_tok,
        "R_frame_tok": R_frame_tok,
        "R_global_tok": R_global_tok,
        "R_swa_tok": R_swa_tok,
        "R_ttt_tok": R_ttt_tok,
    }


def _normalise_dense_label_name(label: str) -> str:
    return str(label).strip().lower().replace("_", " ").replace("/", " ")


def _dense_label_metadata(label_names: Any) -> Dict[str, Any]:
    names = [str(x) for x in (list(label_names) if label_names is not None else [])]
    group_ids = []
    fine_ids = []
    canonical_names = []
    void_ids = []
    dynamic_ids = []
    sky_ids = []
    vegetation_ids = []
    vertical_static_ids = []
    ground_static_ids = []

    dynamic = {"car", "truck", "bus", "van", "person", "people", "rider", "cyclist", "bicycle", "motorcycle", "animal"}
    sky = {"sky", "cloud"}
    vegetation = {"vegetation", "tree", "grass", "plant", "terrain", "mountain"}
    vertical_static = {
        "building",
        "wall",
        "fence",
        "railing",
        "pole",
        "traffic sign",
        "traffic light",
        "bridge",
        "billboard",
        "house",
    }
    ground_static = {"road", "ground", "sidewalk", "floor", "crosswalk"}
    void = {"void", "unknown", "unlabeled"}

    for idx, raw_name in enumerate(names):
        norm = _normalise_dense_label_name(raw_name)
        try:
            canonical = canonicalize_label(norm)
        except Exception:
            canonical = norm
        if "fence" in norm and canonical not in {"fence", "railing"}:
            canonical = "fence"
        elif "billboard" in norm:
            canonical = "billboard"
        elif canonical == "unknown" and norm in {"ground", "crosswalk", "house", "mountain"}:
            canonical = norm

        if norm in void or canonical in void or idx == 0:
            group = int(SEMANTIC_GROUP_UNCERTAIN_REGION)
            fine = int(SEMANTIC_FINE_LABEL_UNKNOWN)
            void_ids.append(idx)
        elif canonical in dynamic:
            group = int(SEMANTIC_GROUP_MOVABLE_THING)
            fine = semantic_fine_label_id(canonical)
            dynamic_ids.append(idx)
        elif canonical in sky:
            group = int(SEMANTIC_GROUP_LOW_VALUE_STUFF)
            fine = semantic_fine_label_id(canonical)
            sky_ids.append(idx)
        elif canonical in vegetation:
            group = int(SEMANTIC_GROUP_LOW_VALUE_STUFF)
            fine = semantic_fine_label_id(canonical)
            vegetation_ids.append(idx)
        elif canonical in vertical_static:
            group = int(SEMANTIC_GROUP_STRUCTURE_ANCHOR)
            fine = semantic_fine_label_id(canonical)
            vertical_static_ids.append(idx)
        elif canonical in ground_static:
            group = int(SEMANTIC_GROUP_STRUCTURE_ANCHOR)
            fine = semantic_fine_label_id(canonical)
            ground_static_ids.append(idx)
        else:
            group = int(label_to_group(canonical))
            fine = semantic_fine_label_id(canonical)

        group_ids.append(group)
        fine_ids.append(fine)
        canonical_names.append(canonical)

    return {
        "label_names": names,
        "canonical_names": canonical_names,
        "group_ids": torch.tensor(group_ids, dtype=torch.long),
        "fine_ids": torch.tensor(fine_ids, dtype=torch.long),
        "void_ids": void_ids,
        "dynamic_ids": dynamic_ids,
        "sky_ids": sky_ids,
        "vegetation_ids": vegetation_ids,
        "vertical_static_ids": vertical_static_ids,
        "ground_static_ids": ground_static_ids,
    }


def _normalize_dense_semantic_confidence(
    confidence_maps: Any,
    *,
    target_shape: Tuple[int, int, int],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    T, H, W = (int(target_shape[0]), int(target_shape[1]), int(target_shape[2]))
    default = torch.ones((T, H, W), dtype=torch.float32)
    debug: Dict[str, Any] = {
        "semantic_confidence_available": False,
        "semantic_confidence_shape": None,
        "semantic_confidence_raw_min": None,
        "semantic_confidence_raw_max": None,
        "semantic_confidence_normalized_min": None,
        "semantic_confidence_normalized_max": None,
        "semantic_confidence_normalization_applied": False,
        "semantic_confidence_projection_note": "unavailable_default_ones",
    }
    if confidence_maps is None:
        return default, debug
    if not isinstance(confidence_maps, torch.Tensor):
        confidence_maps = torch.as_tensor(confidence_maps)
    debug["semantic_confidence_available"] = True
    debug["semantic_confidence_shape"] = [int(x) for x in confidence_maps.shape]
    was_uint8 = confidence_maps.dtype == torch.uint8
    conf = confidence_maps.detach().cpu().float()
    if conf.ndim == 4 and int(conf.shape[-1]) == 1:
        conf = conf[..., 0]
        debug["semantic_confidence_projection_note"] = "squeezed_last_singleton"
    if conf.ndim != 3:
        debug["semantic_confidence_available"] = False
        debug["semantic_confidence_projection_note"] = f"invalid_ndim_{int(conf.ndim)}_default_ones"
        return default, debug

    raw_finite = torch.isfinite(conf)
    if bool(raw_finite.any().item()):
        raw_vals = conf[raw_finite]
        raw_min = float(raw_vals.min().item())
        raw_max = float(raw_vals.max().item())
    else:
        raw_min = 0.0
        raw_max = 0.0
    debug["semantic_confidence_raw_min"] = raw_min
    debug["semantic_confidence_raw_max"] = raw_max
    normalization_applied = bool(was_uint8 or raw_max > 2.0)
    if normalization_applied:
        conf = conf / 255.0
    conf = torch.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    if int(conf.shape[0]) < T:
        pad = torch.zeros((T - int(conf.shape[0]), int(conf.shape[-2]), int(conf.shape[-1])), dtype=torch.float32)
        conf = torch.cat([conf, pad], dim=0)
        debug["semantic_confidence_projection_note"] = "temporal_zero_pad"
    conf = conf[:T]
    if (int(conf.shape[1]), int(conf.shape[2])) != (H, W):
        conf = F.interpolate(conf[:, None], size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        prev_note = str(debug.get("semantic_confidence_projection_note") or "as_is")
        debug["semantic_confidence_projection_note"] = f"{prev_note}+bilinear_resize"
    conf = conf.clamp(0.0, 1.0)
    finite = torch.isfinite(conf)
    if bool(finite.any().item()):
        vals = conf[finite]
        debug["semantic_confidence_normalized_min"] = float(vals.min().item())
        debug["semantic_confidence_normalized_max"] = float(vals.max().item())
    else:
        debug["semantic_confidence_normalized_min"] = 0.0
        debug["semantic_confidence_normalized_max"] = 0.0
    debug["semantic_confidence_normalization_applied"] = normalization_applied
    if debug["semantic_confidence_projection_note"] == "unavailable_default_ones":
        debug["semantic_confidence_projection_note"] = "as_is"
    return conf.to(torch.float32), debug


def _mode_pool_dense_semantic_patches(
    maps: torch.Tensor,
    confidence: torch.Tensor,
    *,
    patch_grid: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T, H, W = int(maps.shape[0]), int(maps.shape[1]), int(maps.shape[2])
    H_tok, W_tok = int(patch_grid[0]), int(patch_grid[1])
    label_patch = torch.zeros((T, H_tok, W_tok), dtype=torch.long)
    purity_patch = torch.zeros((T, H_tok, W_tok), dtype=torch.float32)
    confidence_patch = torch.zeros((T, H_tok, W_tok), dtype=torch.float32)
    y_edges = [int(round(i * H / max(H_tok, 1))) for i in range(H_tok + 1)]
    x_edges = [int(round(i * W / max(W_tok, 1))) for i in range(W_tok + 1)]
    y_edges[0], y_edges[-1] = 0, H
    x_edges[0], x_edges[-1] = 0, W
    for t in range(T):
        frame = maps[t]
        conf = confidence[t]
        for yi in range(H_tok):
            y0 = max(0, min(H, y_edges[yi]))
            y1 = max(y0 + 1, min(H, y_edges[yi + 1]))
            for xi in range(W_tok):
                x0 = max(0, min(W, x_edges[xi]))
                x1 = max(x0 + 1, min(W, x_edges[xi + 1]))
                region = frame[y0:y1, x0:x1].reshape(-1).long()
                if region.numel() <= 0:
                    continue
                values, counts = torch.unique(region, return_counts=True)
                best = int(torch.argmax(counts).item())
                mode_label = values[best]
                mode_count = counts[best]
                label_patch[t, yi, xi] = mode_label
                purity_patch[t, yi, xi] = float(mode_count.item()) / float(max(int(region.numel()), 1))
                mode_mask = region == mode_label
                conf_region = conf[y0:y1, x0:x1].reshape(-1).float()
                if bool(mode_mask.any().item()):
                    confidence_patch[t, yi, xi] = conf_region[mode_mask].mean()
    return label_patch, purity_patch.clamp(0.0, 1.0), confidence_patch.clamp(0.0, 1.0)


def project_dense_semantic_label_maps(
    masklet: MaskletOutput,
    geo: GeometryOutput,
    *,
    num_frames: int,
    pixel_resolution: Tuple[int, int],
    patch_grid: Tuple[int, int],
) -> Optional[Dict[str, torch.Tensor]]:
    """Project dense semantic label maps to the existing patch/token streams."""

    sem = getattr(masklet, "semantic_segmentation", {}) or {}
    label_maps = sem.get("label_maps")
    if label_maps is None:
        return None
    if not isinstance(label_maps, torch.Tensor):
        label_maps = torch.as_tensor(label_maps)
    if int(label_maps.ndim) != 3:
        return None

    T = int(num_frames)
    H_tok, W_tok = int(patch_grid[0]), int(patch_grid[1])
    if T <= 0 or H_tok <= 0 or W_tok <= 0:
        return None
    T_use = min(T, int(label_maps.shape[0]))
    maps = label_maps[:T_use].detach().cpu().long()
    if T_use < T:
        pad = torch.zeros((T - T_use, int(maps.shape[-2]), int(maps.shape[-1])), dtype=torch.long)
        maps = torch.cat([maps, pad], dim=0)

    confidence_maps = sem.get("confidence_maps")
    confidence, confidence_debug = _normalize_dense_semantic_confidence(
        confidence_maps,
        target_shape=(T, int(maps.shape[-2]), int(maps.shape[-1])),
    )
    label_patch, patch_purity, patch_confidence = _mode_pool_dense_semantic_patches(
        maps,
        confidence,
        patch_grid=(H_tok, W_tok),
    )

    meta = _dense_label_metadata(sem.get("label_names", []))
    max_label_id = int(label_patch.max().item()) if label_patch.numel() > 0 else 0
    lut_len = max(max_label_id + 1, int(meta["group_ids"].numel()), 1)
    group_lut = torch.full((lut_len,), int(SEMANTIC_GROUP_UNCERTAIN_REGION), dtype=torch.long)
    fine_lut = torch.full((lut_len,), int(SEMANTIC_FINE_LABEL_UNKNOWN), dtype=torch.long)
    n_meta = int(meta["group_ids"].numel())
    if n_meta > 0:
        group_lut[:n_meta] = meta["group_ids"][: min(n_meta, lut_len)]
        fine_lut[:n_meta] = meta["fine_ids"][: min(n_meta, lut_len)]

    safe_label_patch = label_patch.clamp(min=0, max=lut_len - 1)
    group_patch = group_lut[safe_label_patch]
    fine_patch = fine_lut[safe_label_patch]
    void_ids = set(int(x) for x in meta["void_ids"])
    if void_ids:
        nonvoid_patch = torch.ones_like(label_patch, dtype=torch.float32)
        for vid in void_ids:
            nonvoid_patch[label_patch == int(vid)] = 0.0
    else:
        nonvoid_patch = (label_patch != 0).float()
    semantic_trust_patch = (patch_confidence * patch_purity.square() * nonvoid_patch).clamp(0.0, 1.0)

    G_patch_flat = group_patch.reshape(-1).long()
    Q_patch_flat = semantic_trust_patch.reshape(-1).float().clamp(0.0, 1.0)
    L_patch_flat = fine_patch.reshape(-1).long()
    R_patch_flat = semantic_roles_from_groups(G_patch_flat, Q_patch_flat)
    path_roles = semantic_path_role_priors_from_fine_labels(L_patch_flat, G_patch_flat, Q_patch_flat)

    token_type = geo.token_type.detach().cpu().long()
    L_tok = int(token_type.numel())
    G_tok = torch.full((L_tok,), int(SEMANTIC_GROUP_UNCERTAIN_REGION), dtype=torch.long)
    Q_tok = torch.zeros((L_tok,), dtype=torch.float32)
    L_label_tok = torch.full((L_tok,), int(SEMANTIC_FINE_LABEL_UNKNOWN), dtype=torch.long)
    seed_tok = torch.full((L_tok,), -1, dtype=torch.long)
    seed_patch_flat = torch.full_like(L_patch_flat, -1, dtype=torch.long)
    masklet_instance_tok = torch.full((L_tok,), -1, dtype=torch.long)
    masklet_instance_patch_flat = torch.full_like(L_patch_flat, -1, dtype=torch.long)
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

    nonvoid_ratio = float(nonvoid_patch.reshape(-1).mean().item()) if nonvoid_patch.numel() > 0 else 0.0
    patch_purity_mean = float(patch_purity.mean().item()) if patch_purity.numel() > 0 else 0.0
    patch_confidence_mean = float(patch_confidence.mean().item()) if patch_confidence.numel() > 0 else 0.0
    semantic_trust_mean = float(Q_patch_flat.mean().item()) if Q_patch_flat.numel() > 0 else 0.0
    semantic_source = (
        "dense_label_maps_and_confidence_maps"
        if bool(confidence_debug.get("semantic_confidence_available"))
        else "dense_label_maps_without_confidence"
    )
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
        "stage_c_seed_global_track_idx_patch_flat": seed_patch_flat,
        "stage_c_seed_global_track_idx_tok": seed_tok,
        "stage_c_masklet_instance_idx_patch_flat": masklet_instance_patch_flat,
        "stage_c_masklet_instance_idx_tok": masklet_instance_tok,
        "R_sem_tok": R_tok,
        "R_frame_tok": R_frame_tok,
        "R_global_tok": R_global_tok,
        "R_swa_tok": R_swa_tok,
        "R_ttt_tok": R_ttt_tok,
        "debug": {
            "semantic_source": semantic_source,
            "dense_semantic_available": True,
            "semantic_confidence_available": bool(confidence_debug.get("semantic_confidence_available")),
            "semantic_confidence_shape": confidence_debug.get("semantic_confidence_shape"),
            "semantic_confidence_raw_min": confidence_debug.get("semantic_confidence_raw_min"),
            "semantic_confidence_raw_max": confidence_debug.get("semantic_confidence_raw_max"),
            "semantic_confidence_normalized_min": confidence_debug.get("semantic_confidence_normalized_min"),
            "semantic_confidence_normalized_max": confidence_debug.get("semantic_confidence_normalized_max"),
            "semantic_confidence_normalization_applied": confidence_debug.get("semantic_confidence_normalization_applied"),
            "semantic_confidence_projection_note": confidence_debug.get("semantic_confidence_projection_note"),
            "dense_semantic_patch_nonvoid_ratio": nonvoid_ratio,
            "dense_semantic_token_projection_nonempty": bool(Q_patch_flat.numel() > 0 and nonvoid_ratio > 0.0),
            "dense_semantic_label_names": meta["label_names"],
            "dense_semantic_canonical_names": meta["canonical_names"],
            "dense_semantic_dynamic_label_ids": meta["dynamic_ids"],
            "dense_semantic_sky_label_ids": meta["sky_ids"],
            "dense_semantic_vegetation_label_ids": meta["vegetation_ids"],
            "dense_semantic_vertical_static_label_ids": meta["vertical_static_ids"],
            "dense_semantic_ground_static_label_ids": meta["ground_static_ids"],
            "dense_semantic_projection_mode": "mode_pool_confidence_trust",
            "dense_semantic_patch_purity": patch_purity_mean,
            "dense_semantic_patch_confidence_mean": patch_confidence_mean,
            "semantic_trust_mean": semantic_trust_mean,
        },
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
        group_projection = project_dense_semantic_label_maps(
            masklet,
            geo,
            num_frames=T,
            pixel_resolution=(H_p, W_p),
            patch_grid=(H_tok, W_tok),
        )
        semantic_projection_source = "dense_label_maps_without_confidence"
        if group_projection is None:
            group_projection = project_masklet_semantic_groups(
                masklet,
                geo,
                num_frames=T,
                pixel_resolution=(H_p, W_p),
                patch_grid=(H_tok, W_tok),
            )
            semantic_projection_source = "masklet_sparse_projection"
        else:
            semantic_projection_source = str(
                group_projection.get("debug", {}).get("semantic_source") or semantic_projection_source
            )
            sparse_seed_projection = project_masklet_semantic_groups(
                masklet,
                geo,
                num_frames=T,
                pixel_resolution=(H_p, W_p),
                patch_grid=(H_tok, W_tok),
            )
            group_projection["stage_c_seed_global_track_idx_patch_flat"] = sparse_seed_projection[
                "stage_c_seed_global_track_idx_patch_flat"
            ]
            group_projection["stage_c_seed_global_track_idx_tok"] = sparse_seed_projection[
                "stage_c_seed_global_track_idx_tok"
            ]
            group_projection["stage_c_masklet_instance_idx_patch_flat"] = sparse_seed_projection[
                "stage_c_masklet_instance_idx_patch_flat"
            ]
            group_projection["stage_c_masklet_instance_idx_tok"] = sparse_seed_projection[
                "stage_c_masklet_instance_idx_tok"
            ]
            group_projection.setdefault("debug", {})["stage_c_seed_projection_source"] = (
                "masklet_sparse_projection_merged_with_dense_semantic_labels"
            )
            seed_tok = sparse_seed_projection["stage_c_seed_global_track_idx_tok"]
            masklet_instance_tok = sparse_seed_projection["stage_c_masklet_instance_idx_tok"]
            group_projection["debug"]["stage_c_seed_token_nonnegative_count"] = int(
                (seed_tok.detach().cpu().long().reshape(-1) >= 0).sum().item()
            )
            group_projection["debug"]["stage_c_masklet_instance_token_nonnegative_count"] = int(
                (masklet_instance_tok.detach().cpu().long().reshape(-1) >= 0).sum().item()
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
            stage_c_seed_global_track_idx_patch_flat=group_projection.get("stage_c_seed_global_track_idx_patch_flat"),
            stage_c_seed_global_track_idx_tok=group_projection.get("stage_c_seed_global_track_idx_tok"),
            stage_c_masklet_instance_idx_patch_flat=group_projection.get("stage_c_masklet_instance_idx_patch_flat"),
            stage_c_masklet_instance_idx_tok=group_projection.get("stage_c_masklet_instance_idx_tok"),
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
                "semantic_source": semantic_projection_source,
                "dense_semantic_available": bool(semantic_projection_source.startswith("dense_label_maps")),
                "semantic_confidence_available": group_projection.get("debug", {}).get("semantic_confidence_available"),
                "semantic_confidence_shape": group_projection.get("debug", {}).get("semantic_confidence_shape"),
                "semantic_confidence_raw_min": group_projection.get("debug", {}).get("semantic_confidence_raw_min"),
                "semantic_confidence_raw_max": group_projection.get("debug", {}).get("semantic_confidence_raw_max"),
                "semantic_confidence_normalized_min": group_projection.get("debug", {}).get("semantic_confidence_normalized_min"),
                "semantic_confidence_normalized_max": group_projection.get("debug", {}).get("semantic_confidence_normalized_max"),
                "semantic_confidence_normalization_applied": group_projection.get("debug", {}).get("semantic_confidence_normalization_applied"),
                "semantic_confidence_projection_note": group_projection.get("debug", {}).get("semantic_confidence_projection_note"),
                "dense_semantic_patch_nonvoid_ratio": group_projection.get("debug", {}).get("dense_semantic_patch_nonvoid_ratio"),
                "dense_semantic_token_projection_nonempty": group_projection.get("debug", {}).get("dense_semantic_token_projection_nonempty"),
                "dense_semantic_projection_mode": group_projection.get("debug", {}).get("dense_semantic_projection_mode"),
                "dense_semantic_patch_purity": group_projection.get("debug", {}).get("dense_semantic_patch_purity"),
                "stage_c_seed_projection_source": group_projection.get("debug", {}).get("stage_c_seed_projection_source"),
                "stage_c_seed_token_nonnegative_count": group_projection.get("debug", {}).get("stage_c_seed_token_nonnegative_count"),
                "stage_c_masklet_instance_token_nonnegative_count": group_projection.get("debug", {}).get("stage_c_masklet_instance_token_nonnegative_count"),
                "dense_semantic_patch_confidence_mean": group_projection.get("debug", {}).get("dense_semantic_patch_confidence_mean"),
                "semantic_trust_mean": group_projection.get("debug", {}).get("semantic_trust_mean"),
                "dense_semantic_label_names": group_projection.get("debug", {}).get("dense_semantic_label_names"),
                "dense_semantic_canonical_names": group_projection.get("debug", {}).get("dense_semantic_canonical_names"),
                "semantic_group_taxonomy": "stage_c_coarse_5_groups",
                "semantic_fine_label_available": True,
                "semantic_fine_label_exact_source": "semantic_segmentation.label_maps" if semantic_projection_source.startswith("dense_label_maps") else "MaskletOutput.L_sem",
                "semantic_group_exact_source": "semantic_segmentation.label_maps" if semantic_projection_source.startswith("dense_label_maps") else "MaskletOutput.G_sem",
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
