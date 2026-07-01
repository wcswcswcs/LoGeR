"""
Hybrid Memory Controller for LoGeR pipeline v2.

This module is the v2 replacement shell for the old TTT-only write
controller.  The first implementation intentionally keeps the working
control surface conservative:

* Pass 1: probe forward from committed state, collecting geometry and TTT
  write primitives without committing state.
* Pass 2: controlled forward from the same committed state.  The frame
  attention, SWA read, TTT apply, and chunk/global-attention identity hooks are
  wired through real Pi3 model sites; non-identity read controllers are still
  intentionally conservative.

That gives us the correct two-pass protocol and preserves the old BL-01 style
TTT branch/layer experiments while leaving clear extension points for stronger
read-path controllers.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .dynamic_cue_extractor import CUE_ANCHOR, CUE_DYN, CUE_OCC, CUE_UNC, CueOutput
from .geometry_backbone import (
    GeometryOutput,
    LoGeRGeometryBackbone,
    TOKEN_TYPE_PATCH,
    WriteCacheOutput,
)
from .semantic_prior_generator import (
    PriorOutput,
    SEMANTIC_FINE_LABEL_TO_ID,
    SEMANTIC_FINE_LABEL_UNKNOWN,
    SEMANTIC_FINE_MOVABLE_IDS,
    SEMANTIC_FINE_SKY_IDS,
    SEMANTIC_FINE_STRUCTURE_IDS,
    SEMANTIC_FINE_VEGETATION_IDS,
    SEMANTIC_ROLE_FALLBACK,
    SEMANTIC_ROLE_NEGATIVE_SHORT,
    SEMANTIC_ROLE_NEUTRAL_KEEP,
    SEMANTIC_ROLE_POSITIVE_LONG,
    SEMANTIC_ROLE_PROTECT_NEUTRAL,
)
from .ttt_write_controller import TTTWriteController, WriteResult
from .video_masklet_frontend import (
    SEMANTIC_GROUP_LOW_VALUE_STUFF,
    SEMANTIC_GROUP_MOVABLE_THING,
    SEMANTIC_GROUP_STATIC_THING,
    SEMANTIC_GROUP_STRUCTURE_ANCHOR,
    SEMANTIC_GROUP_UNCERTAIN_REGION,
)


_DUAL_LIFETIME_TRANSIENT_MODES = {
    "dual_lifetime",
    "dual_fast_weight",
    "apply_short_delta",
    "short_apply_delta",
}

_V67_SEMCUE_VERTICAL_STATIC_FINE_IDS = {
    int(SEMANTIC_FINE_LABEL_TO_ID[name])
    for name in ("building", "wall", "fence", "bridge", "railing", "pole", "traffic sign", "billboard", "house")
    if name in SEMANTIC_FINE_LABEL_TO_ID
}
_V67_SEMCUE_GROUND_STATIC_FINE_IDS = {
    int(SEMANTIC_FINE_LABEL_TO_ID[name])
    for name in ("road", "sidewalk", "floor", "stair", "ground", "crosswalk")
    if name in SEMANTIC_FINE_LABEL_TO_ID
}
_V68_LOWOBS_FINE_IDS = {
    int(SEMANTIC_FINE_LABEL_TO_ID[name])
    for name in ("tree", "vegetation", "grass", "mountain")
    if name in SEMANTIC_FINE_LABEL_TO_ID
}


def _is_dual_lifetime_transient_delta(transient_delta: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(transient_delta, dict):
        return False
    mode = str(transient_delta.get("_mode", "")).strip().lower()
    if mode not in _DUAL_LIFETIME_TRANSIENT_MODES:
        return False
    ttl = int(transient_delta.get("_ttl_remaining", 0) or 0)
    if ttl <= 0:
        return False
    for branch_name in ("w0", "w1", "w2"):
        values = transient_delta.get(branch_name)
        if isinstance(values, list) and any(v is not None for v in values):
            return True
    return False


def _renorm_like(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if candidate.ndim < 2:
        return candidate.to(dtype=reference.dtype)
    ref = reference.detach().float()
    out = candidate.float()
    ref_norm = ref.norm(dim=1, keepdim=True).clamp_min(1e-6)
    out_norm = out.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return (out / out_norm * ref_norm).to(dtype=reference.dtype)


def _optional_1d_float_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        out = value.detach().cpu().float().reshape(-1)
    else:
        try:
            out = torch.as_tensor(value, dtype=torch.float32).detach().cpu().reshape(-1)
        except (TypeError, ValueError):
            return None
    return out if int(out.numel()) > 0 else None


def _parse_chunk_index_set(value: Any) -> set:
    out = set()
    for item in str(value or "").split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    return out


def _apply_dual_lifetime_delta_to_branch(
    branch_values: Any,
    delta_values: Any,
    *,
    scale: float,
) -> Any:
    if not isinstance(branch_values, list) or not isinstance(delta_values, list):
        return branch_values
    out: List[Any] = list(branch_values)
    for li, base in enumerate(branch_values):
        if base is None or li >= len(delta_values):
            continue
        delta = delta_values[li]
        if delta is None or not torch.is_tensor(base) or not torch.is_tensor(delta):
            continue
        if tuple(base.shape) != tuple(delta.shape):
            continue
        delta_t = delta.to(device=base.device, dtype=base.dtype)
        candidate = base.float() + float(scale) * delta_t.float()
        out[li] = _renorm_like(base, candidate)
    return out


def _ttt_state_with_dual_lifetime_apply(ttt_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(ttt_state, dict):
        return ttt_state
    transient_delta = ttt_state.get("transient_delta")
    if not _is_dual_lifetime_transient_delta(transient_delta):
        return ttt_state
    scale = float(transient_delta.get("_apply_scale", 1.0))
    out = dict(ttt_state)
    for branch_name in ("w0", "w1", "w2"):
        out[branch_name] = _apply_dual_lifetime_delta_to_branch(
            ttt_state.get(branch_name),
            transient_delta.get(branch_name),
            scale=scale,
        )
    out["_dual_lifetime_apply"] = {
        "mode": str(transient_delta.get("_mode", "")),
        "ttl": int(transient_delta.get("_ttl_remaining", 0) or 0),
        "scale": float(scale),
    }
    return out


def _write_cache_with_long_old_weights_for_dual_lifetime(
    write_cache: WriteCacheOutput,
    long_ttt_state: Optional[Dict[str, Any]],
) -> Tuple[WriteCacheOutput, Dict[str, Any]]:
    debug = {
        "ttt_dual_lifetime_long_old_override": False,
        "ttt_dual_lifetime_long_old_override_tensors": 0,
    }
    if not isinstance(long_ttt_state, dict):
        return write_cache, debug
    if not _is_dual_lifetime_transient_delta(long_ttt_state.get("transient_delta")):
        return write_cache, debug
    branch_fields = (
        ("w0", "w0_old"),
        ("w1", "w1_old"),
        ("w2", "w2_old"),
    )
    new_layer_caches = []
    replaced = 0
    for li, layer_cache in enumerate(write_cache.layer_caches):
        updates: Dict[str, torch.Tensor] = {}
        for branch_name, field_name in branch_fields:
            branch_values = long_ttt_state.get(branch_name)
            if not isinstance(branch_values, list) or li >= len(branch_values):
                continue
            long_old = branch_values[li]
            current_old = getattr(layer_cache, field_name, None)
            if long_old is None or current_old is None:
                continue
            if not torch.is_tensor(long_old) or not torch.is_tensor(current_old):
                continue
            if tuple(long_old.shape) != tuple(current_old.shape):
                continue
            updates[field_name] = long_old.to(device=current_old.device, dtype=current_old.dtype)
        if updates:
            new_layer_caches.append(replace(layer_cache, **updates))
            replaced += len(updates)
        else:
            new_layer_caches.append(layer_cache)
    if replaced <= 0:
        return write_cache, debug
    debug["ttt_dual_lifetime_long_old_override"] = True
    debug["ttt_dual_lifetime_long_old_override_tensors"] = int(replaced)
    return replace(write_cache, layer_caches=new_layer_caches), debug


@dataclass
class HybridMemoryState:
    """Committed hybrid memory state read by a chunk.

    Phase-1 stores the existing LoGeR adaptive state in ``ttt_state``.  The
    ``history`` entry inside that dict is the current SWA/local-memory carrier
    used by the model.  Explicit ``swa_state`` / ``ref_state`` fields are kept
    so callers can migrate without changing the public protocol again.
    """

    ttt_state: Optional[Dict[str, Any]] = None
    swa_state: Optional[Dict[str, Any]] = None
    ref_state: Optional[Dict[str, Any]] = None
    prev_control_summary: Optional[Dict[str, Any]] = None
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_ttt_input(self) -> Optional[Dict[str, Any]]:
        return _ttt_state_with_dual_lifetime_apply(self.ttt_state)

    @classmethod
    def from_ttt_state(cls, ttt_state: Optional[Dict[str, Any]]) -> "HybridMemoryState":
        return cls(ttt_state=ttt_state)


@dataclass
class HybridMemoryCacheOutput:
    """Probe/controlled memory cache.

    ``ttt_cache`` stores replay primitives, while the read-path fields hold
    compact trace records collected from real identity hook sites.
    """

    token_meta: Dict[str, Any]
    ttt_cache: Optional[WriteCacheOutput] = None
    swa_cache: Optional[Dict[str, Any]] = None
    frame_attn_cache: Optional[Dict[str, Any]] = None
    chunk_attn_cache: Optional[Dict[str, Any]] = None
    state_refs: Dict[str, Any] = field(default_factory=dict)
    update_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeOutput:
    geometry: GeometryOutput
    token_meta: Dict[str, Any]
    hybrid_cache: HybridMemoryCacheOutput
    probe_trace: Dict[str, Any]
    native_provisional_state: HybridMemoryState
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridMemoryControlPrior:
    """Control bundle produced by Stage D for v2.

    ``P_ttt_write`` is the mature controlled path today.  Read-path fields are
    passed to real hook sites, but only identity/small single-path controllers
    should be trusted until Phase C validates them.
    """

    D_tok: Optional[torch.Tensor] = None
    R_tok: Optional[torch.Tensor] = None
    U_tok: Optional[torch.Tensor] = None
    P_ref: Optional[torch.Tensor] = None
    S_tok: Optional[torch.Tensor] = None
    G_sem_tok: Optional[torch.Tensor] = None
    Q_sem_tok: Optional[torch.Tensor] = None
    L_sem_tok: Optional[torch.Tensor] = None
    V_sem_tok: Optional[torch.Tensor] = None
    R_sem_tok: Optional[torch.Tensor] = None
    R_frame_tok: Optional[torch.Tensor] = None
    R_global_tok: Optional[torch.Tensor] = None
    R_swa_tok: Optional[torch.Tensor] = None
    R_ttt_tok: Optional[torch.Tensor] = None
    stage_c_seed_global_track_idx_tok: Optional[torch.Tensor] = None
    stage_c_masklet_instance_idx_tok: Optional[torch.Tensor] = None
    K_stable_tok: Optional[torch.Tensor] = None
    swa_redirection_source_mask_tok: Optional[torch.Tensor] = None
    C_ttt_conflict_tok: Optional[torch.Tensor] = None
    S_scale_risk_tok: Optional[torch.Tensor] = None
    A_anchor_tok: Optional[torch.Tensor] = None
    A_anchor_mask_tok: Optional[torch.Tensor] = None
    P_ttt_write: Optional[torch.Tensor] = None
    P_ttt_read: Optional[torch.Tensor] = None
    P_swa_read_prev: Optional[torch.Tensor] = None
    D_swa_write_tok: Optional[torch.Tensor] = None
    frame_bias_spec: Optional[Dict[str, Any]] = None
    chunk_bias_spec: Optional[Dict[str, Any]] = None
    sparse_route_mask: Optional[torch.Tensor] = None
    B_chunk_geo: Optional[float] = None
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridMemoryResult:
    geometry_output: GeometryOutput
    state_next: HybridMemoryState
    ttt_state_next: Optional[Dict[str, Any]]
    swa_state_next: Optional[Dict[str, Any]]
    write_result: Optional[WriteResult] = None
    control_trace: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)


def _tensor_fingerprint(t: torch.Tensor) -> str:
    t_cpu = t.detach().cpu()
    sample = t_cpu.reshape(-1)[:16].float()
    payload = {
        "shape": list(t_cpu.shape),
        "dtype": str(t_cpu.dtype),
        "sum16": float(sample.sum().item()) if sample.numel() else 0.0,
        "mean16": float(sample.mean().item()) if sample.numel() else 0.0,
    }
    return json.dumps(payload, sort_keys=True)


def hybrid_state_fingerprint(state: Optional[HybridMemoryState]) -> str:
    """Small debug fingerprint for double-write checks.

    This is not a cryptographic tensor hash.  It is intentionally cheap and
    stable enough to catch "controlled pass accidentally consumed probe state"
    type bugs in logs.
    """

    if state is None or state.ttt_state is None:
        return "empty"
    h = hashlib.sha1()
    ttt = state.ttt_state
    for key in ("w0", "w1", "w2"):
        values = ttt.get(key, []) or []
        h.update(key.encode("utf-8"))
        h.update(str(len(values)).encode("utf-8"))
        for value in values:
            if value is None:
                h.update(b"none")
            elif torch.is_tensor(value):
                h.update(_tensor_fingerprint(value).encode("utf-8"))
            else:
                h.update(str(type(value)).encode("utf-8"))
    history = ttt.get("history")
    h.update(b"history")
    h.update(str(0 if history is None else len(history)).encode("utf-8"))
    return h.hexdigest()[:16]


def _build_token_meta(geo: GeometryOutput) -> Dict[str, Any]:
    token_type = geo.token_type.detach().cpu().long()
    L_tok = int(token_type.shape[0])
    T = max(int(geo.num_frames), 1)
    tokens_per_frame = max(L_tok // T, 1)
    token_frame_id = (torch.arange(L_tok, dtype=torch.long) // tokens_per_frame).clamp_max(T - 1)
    token_local_index = torch.arange(L_tok, dtype=torch.long) % tokens_per_frame
    token_order_id = torch.arange(L_tok, dtype=torch.long)
    token_valid = torch.ones(L_tok, dtype=torch.bool)
    protected_default_mask = token_type != TOKEN_TYPE_PATCH
    return {
        "patch_meta": geo.patch_meta.detach().cpu().long(),
        "token_type": token_type,
        "token_frame_id": token_frame_id,
        "token_local_index": token_local_index,
        "token_order_id": token_order_id,
        "token_valid": token_valid,
        "special_token_meta": {
            "num_special_tokens": int((token_type != TOKEN_TYPE_PATCH).sum().item()),
            "num_patch_tokens": int((token_type == TOKEN_TYPE_PATCH).sum().item()),
        },
        "protected_default_mask": protected_default_mask,
    }


def _token_from_patch_values(
    geo: GeometryOutput,
    patch_values: torch.Tensor,
    *,
    special_value: float,
) -> torch.Tensor:
    token_type = geo.token_type.detach().cpu().long()
    out = torch.full((int(token_type.shape[0]),), float(special_value), dtype=torch.float32)
    patch_mask = token_type == TOKEN_TYPE_PATCH
    n_patch = int(patch_mask.sum().item())
    vals = patch_values.detach().cpu().float().reshape(-1)
    if vals.numel() < n_patch:
        padded = torch.full((n_patch,), float(special_value), dtype=torch.float32)
        padded[: vals.numel()] = vals
        vals = padded
    out[patch_mask] = vals[:n_patch]
    return out


def _normalize01(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().float()
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return torch.zeros_like(values, dtype=torch.float32)
    safe = torch.where(finite, values, torch.zeros_like(values))
    vmin = safe[finite].min()
    vmax = safe[finite].max()
    if float((vmax - vmin).abs().item()) < 1e-8:
        return torch.zeros_like(safe, dtype=torch.float32)
    return ((safe - vmin) / (vmax - vmin)).clamp(0.0, 1.0)


def _stable_semantic_score(
    labels: Optional[torch.Tensor],
    groups: Optional[torch.Tensor],
    trust: Optional[torch.Tensor],
    n_patch: int,
    *,
    missing_trust_policy: str = "zero",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    score = torch.zeros((n_patch,), dtype=torch.float32)
    debug: Dict[str, Any] = {
        "semantic_anchor_label_missing": labels is None,
        "semantic_anchor_group_missing": groups is None,
        "semantic_anchor_trust_missing": trust is None,
    }
    if labels is not None and int(labels.numel()) == n_patch:
        L = labels.detach().cpu().long().reshape(-1)
        known = (L >= 0) & (L != int(SEMANTIC_FINE_LABEL_UNKNOWN))
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
        score[structure & known] = 1.0
        score[(sky | vegetation) & known] = 0.15
        score[movable & known] = 0.0
        debug.update({
            "semantic_anchor_fine_structure_tokens": int((structure & known).sum().item()),
            "semantic_anchor_fine_context_tokens": int(((sky | vegetation) & known).sum().item()),
            "semantic_anchor_fine_dynamic_tokens": int((movable & known).sum().item()),
        })
    if groups is not None and int(groups.numel()) == n_patch:
        G = groups.detach().cpu().long().reshape(-1)
        coarse = torch.zeros_like(score)
        coarse[G == int(SEMANTIC_GROUP_STRUCTURE_ANCHOR)] = 0.90
        coarse[G == int(SEMANTIC_GROUP_STATIC_THING)] = 0.70
        coarse[G == int(SEMANTIC_GROUP_LOW_VALUE_STUFF)] = 0.15
        coarse[G == int(SEMANTIC_GROUP_MOVABLE_THING)] = 0.0
        coarse[G == int(SEMANTIC_GROUP_UNCERTAIN_REGION)] = 0.0
        score = torch.maximum(score, coarse)
        debug.update({
            "semantic_anchor_group_structure_tokens": int((G == int(SEMANTIC_GROUP_STRUCTURE_ANCHOR)).sum().item()),
            "semantic_anchor_group_static_tokens": int((G == int(SEMANTIC_GROUP_STATIC_THING)).sum().item()),
            "semantic_anchor_group_dynamic_tokens": int((G == int(SEMANTIC_GROUP_MOVABLE_THING)).sum().item()),
        })
    if trust is not None and int(trust.numel()) == n_patch:
        score = score * trust.detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
        debug["semantic_anchor_trust_missing_policy"] = "provided"
    elif str(missing_trust_policy or "zero").strip().lower() in {"neutral", "one", "ones", "unit"}:
        debug["semantic_anchor_trust_missing_policy"] = "neutral"
    else:
        score = torch.zeros_like(score)
        debug["semantic_anchor_trust_missing_policy"] = "zero"
    return score.clamp(0.0, 1.0), debug


def _deterministic_patch_noise(n_patch: int, *, seed: float) -> torch.Tensor:
    idx = torch.arange(n_patch, dtype=torch.float32)
    return torch.frac(torch.sin((idx + 1.0 + float(seed)) * 12.9898) * 43758.5453)


def _select_spatially_diverse_anchors(
    raw_score: torch.Tensor,
    *,
    num_frames: int,
    patch_grid: Tuple[int, int],
    target_ratio: float,
    min_ratio: float,
    max_ratio: float,
    min_score: float,
    grid_rows: int,
    grid_cols: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    score = torch.nan_to_num(
        raw_score.detach().cpu().float().reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    n_patch = int(score.numel())
    mask = torch.zeros((n_patch,), dtype=torch.bool)
    if n_patch <= 0:
        return mask, {"semantic_anchor_select_reason": "empty_score"}
    H, W = int(patch_grid[0]), int(patch_grid[1])
    T = max(int(num_frames), 1)
    if H <= 0 or W <= 0 or H * W * T != n_patch:
        H, W, T = 1, n_patch, 1

    eligible = torch.nonzero(score > float(min_score), as_tuple=False).reshape(-1)
    if int(eligible.numel()) <= 0:
        return mask, {
            "semantic_anchor_select_reason": "no_score_above_min",
            "semantic_anchor_min_score": float(min_score),
        }

    min_k = max(1, int(round(max(0.0, float(min_ratio)) * n_patch)))
    max_k = max(min_k, int(round(max(float(max_ratio), float(min_ratio)) * n_patch)))
    target_k = int(round(max(0.0, float(target_ratio)) * n_patch))
    target_k = min(max(target_k, min_k), max_k, int(eligible.numel()))

    patch_linear = torch.arange(n_patch) % max(H * W, 1)
    rows = (patch_linear // max(W, 1)).long()
    cols = (patch_linear % max(W, 1)).long()
    gr = max(int(grid_rows), 1)
    gc = max(int(grid_cols), 1)
    rr = torch.clamp(rows * gr // max(H, 1), 0, gr - 1)
    cc = torch.clamp(cols * gc // max(W, 1), 0, gc - 1)
    cells = rr * gc + cc
    num_cells = gr * gc
    cell_cap = max(1, int(math.ceil(target_k / max(num_cells, 1))) * 2)
    cell_counts = torch.zeros((num_cells,), dtype=torch.long)

    order = eligible[torch.argsort(score[eligible], descending=True)]
    for idx in order.tolist():
        cell = int(cells[idx].item())
        if int(cell_counts[cell].item()) >= cell_cap:
            continue
        mask[idx] = True
        cell_counts[cell] += 1
        if int(mask.sum().item()) >= target_k:
            break
    if int(mask.sum().item()) < target_k:
        for idx in order.tolist():
            if mask[idx]:
                continue
            mask[idx] = True
            if int(mask.sum().item()) >= target_k:
                break

    hit_counts = cell_counts.float()
    hit_counts = hit_counts[hit_counts > 0]
    if hit_counts.numel() > 0:
        probs = hit_counts / hit_counts.sum().clamp_min(1.0)
        entropy = float((-(probs * probs.log()).sum() / math.log(max(num_cells, 2))).item())
    else:
        entropy = 0.0
    return mask, {
        "semantic_anchor_select_reason": "ok" if int(mask.sum().item()) > 0 else "empty_after_selection",
        "semantic_anchor_target_ratio": float(target_ratio),
        "semantic_anchor_min_ratio": float(min_ratio),
        "semantic_anchor_max_ratio": float(max_ratio),
        "semantic_anchor_min_score": float(min_score),
        "semantic_anchor_target_tokens": int(target_k),
        "semantic_anchor_eligible_tokens": int(eligible.numel()),
        "semantic_anchor_cell_cap": int(cell_cap),
        "semantic_anchor_spatial_entropy": entropy,
        "semantic_anchor_spatial_cells_hit": int((cell_counts > 0).sum().item()),
        "semantic_anchor_spatial_cells_total": int(num_cells),
    }


def _patch_labels_from_token_labels(
    token_labels: Optional[torch.Tensor],
    token_type: torch.Tensor,
) -> Optional[torch.Tensor]:
    if token_labels is None:
        return None
    labels = token_labels.detach().cpu().long().reshape(-1)
    token_type_cpu = token_type.detach().cpu().long().reshape(-1)
    if int(labels.numel()) != int(token_type_cpu.numel()):
        return None
    patch_mask = token_type_cpu == TOKEN_TYPE_PATCH
    if not bool(patch_mask.any()):
        return None
    return labels[patch_mask]


def _semantic_z_recondition_patch(
    base_patch: torch.Tensor,
    patch_labels: Optional[torch.Tensor],
    *,
    min_count: int = 32,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    base = torch.nan_to_num(base_patch.detach().cpu().float().reshape(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    debug: Dict[str, Any] = {
        "v31_semantic_recondition_applied": False,
        "v31_semantic_recondition_mode": "semantic_z",
        "v31_semantic_label_count": 0,
        "v31_semantic_label_fallback_ratio": 1.0,
        "v31_semantic_min_count": int(min_count),
    }
    if patch_labels is None:
        debug["v31_semantic_recondition_reason"] = "missing_patch_labels"
        return base, debug
    labels = patch_labels.detach().cpu().long().reshape(-1)
    if int(labels.numel()) != int(base.numel()) or int(base.numel()) <= 0:
        debug["v31_semantic_recondition_reason"] = "empty_patch_labels"
        return base, debug
    n = int(base.numel())
    out = base.clone()
    fallback = torch.zeros((n,), dtype=torch.bool)
    label_stats: Dict[str, Dict[str, float]] = {}
    applied_labels = 0
    for label in labels.unique(sorted=True):
        mask = labels == label
        count = int(mask.sum().item())
        if count < int(min_count):
            fallback |= mask
            continue
        vals = base[mask]
        std = vals.std(unbiased=False)
        if not torch.isfinite(std) or float(std.item()) < 1e-6:
            fallback |= mask
            continue
        mean = vals.mean()
        z = ((vals - mean) / std.clamp_min(1e-6)).clamp(-6.0, 6.0)
        out[mask] = torch.sigmoid(z)
        label_stats[str(int(label.item()))] = {
            "count": float(count),
            "mean": float(mean.item()),
            "std": float(std.item()),
        }
        applied_labels += 1
    debug.update(
        {
            "v31_semantic_recondition_applied": bool(applied_labels > 0),
            "v31_semantic_label_count": int(applied_labels),
            "v31_semantic_label_fallback_ratio": float(fallback.float().mean().item()) if fallback.numel() else 1.0,
            "v31_semantic_label_stats": label_stats,
        }
    )
    if applied_labels <= 0:
        debug["v31_semantic_recondition_reason"] = "no_label_with_nonzero_variance"
    return out.clamp(0.0, 1.0), debug


def _parse_v67_semz_read_cue(read_cue_source: str) -> Optional[Dict[str, Any]]:
    src = str(read_cue_source or "")
    if not (src.startswith("v67.semz_") and src.endswith(".c23past")):
        return None
    body = src[len("v67.semz_") : -len(".c23past")]
    parts = [p for p in body.split(".") if p]
    if not parts:
        return None
    head = parts[0]
    control_mode = parts[1] if len(parts) > 1 else "semantic"
    if control_mode in {"control", "sem"}:
        control_mode = "semantic"
    if "_beta" not in head:
        return None
    label_mode, beta_token = head.split("_beta", 1)
    if label_mode not in {"coarse", "fine"}:
        return None
    try:
        beta_int = int(beta_token)
    except ValueError:
        return None
    # Historical names use beta025/beta040/beta525/beta650 for 0.25/0.40/0.525/0.65.
    beta = float(beta_int) / (1000.0 if beta_int >= 100 else 100.0)
    return {
        "label_mode": label_mode,
        "beta": beta,
        "beta_token": beta_token,
        "control_mode": control_mode,
    }


def _v67_semantic_mad_recondition_patch(
    base_patch: torch.Tensor,
    patch_labels: Optional[torch.Tensor],
    *,
    beta: float,
    min_count: int = 256,
    control_mode: str = "semantic",
    chunk_idx: int = -1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    base_shape = tuple(base_patch.shape)
    base = torch.nan_to_num(
        base_patch.detach().cpu().float().reshape(-1),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    debug: Dict[str, Any] = {
        "v67_semz_available": False,
        "v67_semz_mode": "median_mad_beta",
        "v67_semz_beta": float(beta),
        "v67_semz_min_count": int(min_count),
        "v67_semz_control_mode": str(control_mode or "semantic"),
        "v67_semz_chunk_idx": int(chunk_idx),
        "v67_semz_base_mean": float(base.mean().item()) if base.numel() else 0.0,
    }
    if patch_labels is None:
        debug["v67_semz_reason"] = "missing_patch_labels"
        return base.reshape(base_shape), debug
    labels = patch_labels.detach().cpu().long().reshape(-1)
    if int(labels.numel()) != int(base.numel()) or int(base.numel()) <= 0:
        debug["v67_semz_reason"] = "empty_or_mismatched_patch_labels"
        return base.reshape(base_shape), debug

    mode = str(control_mode or "semantic").strip().lower()
    seed = int(6700003 + max(int(chunk_idx), 0) * 1009 + int(base.numel()))
    if mode in {"random_hist", "random_same_histogram", "random"}:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        labels = labels[torch.randperm(int(labels.numel()), generator=generator)]
    elif mode in {"shuffled", "shuffled_spatial", "spatial_shuffle", "shuffle"}:
        if len(base_shape) >= 3 and int(labels.numel()) == int(torch.tensor(base_shape).prod().item()):
            label_grid = labels.reshape(base_shape)
            flat_frames = label_grid.reshape(int(base_shape[0]), -1).clone()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            for frame_idx in range(int(flat_frames.shape[0])):
                flat_frames[frame_idx] = flat_frames[frame_idx, torch.randperm(int(flat_frames.shape[1]), generator=generator)]
            labels = flat_frames.reshape(-1)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            labels = labels[torch.randperm(int(labels.numel()), generator=generator)]
    elif mode in {"geometry_only_z", "geometry", "global_z"}:
        labels = torch.zeros_like(labels)
    elif mode in {"semantic_only", "semantic_prior_only"}:
        out = torch.full_like(base, 0.5, dtype=torch.float32)
        out[labels == int(SEMANTIC_GROUP_MOVABLE_THING)] = 0.85
        out[labels == int(SEMANTIC_GROUP_LOW_VALUE_STUFF)] = 0.65
        out[labels == int(SEMANTIC_GROUP_UNCERTAIN_REGION)] = 0.55
        out[labels == int(SEMANTIC_GROUP_STATIC_THING)] = 0.35
        out[labels == int(SEMANTIC_GROUP_STRUCTURE_ANCHOR)] = 0.20
        debug.update({
            "v67_semz_available": True,
            "v67_semz_reason": "semantic_only_prior",
            "v67_semz_label_count": int(torch.unique(labels).numel()),
            "v67_semz_fallback_ratio": 0.0,
            "v67_semz_read_mean_after": float(out.mean().item()) if out.numel() else 0.0,
            "v67_semz_random_seed": seed,
        })
        return out.clamp(0.0, 1.0).reshape(base_shape), debug
    elif mode not in {"semantic", "none", ""}:
        debug["v67_semz_reason"] = f"unknown_control_mode:{mode}"
        return base.reshape(base_shape), debug

    out = base.clone()
    fallback = torch.zeros_like(base, dtype=torch.bool)
    label_stats: Dict[str, Dict[str, float]] = {}
    applied_labels = 0
    beta = float(beta)
    eps = 1e-6
    for label in labels.unique(sorted=True):
        mask = labels == label
        count = int(mask.sum().item())
        if count < int(min_count):
            fallback |= mask
            label_stats[str(int(label.item()))] = {"count": float(count), "fallback": 1.0}
            continue
        vals = base[mask]
        med = torch.median(vals)
        mad = torch.median(torch.abs(vals - med))
        if not torch.isfinite(mad):
            fallback |= mask
            label_stats[str(int(label.item()))] = {
                "count": float(count),
                "median": float(med.item()) if torch.isfinite(med) else 0.0,
                "mad": 0.0,
                "fallback": 1.0,
                "fallback_reason": "nonfinite_mad",
            }
            continue
        mad_value = float(mad.item())
        denom = mad.clamp_min(eps)
        z = ((vals - med) / denom).clamp(-6.0, 6.0)
        out[mask] = (0.5 + beta * z).clamp(0.0, 1.0)
        label_stats[str(int(label.item()))] = {
            "count": float(count),
            "median": float(med.item()),
            "mad": mad_value,
            "mad_epsilon_used": bool(mad_value < eps),
            "mean_before": float(vals.mean().item()),
            "mean_after": float(out[mask].mean().item()),
            "fallback": 0.0,
        }
        applied_labels += 1

    debug.update({
        "v67_semz_available": bool(applied_labels > 0),
        "v67_semz_reason": "ok" if applied_labels > 0 else "no_label_with_sufficient_mad_support",
        "v67_semz_label_count": int(applied_labels),
        "v67_semz_fallback_ratio": float(fallback.float().mean().item()) if fallback.numel() else 1.0,
        "v67_semz_label_stats": label_stats,
        "v67_semz_read_mean_after": float(out.mean().item()) if out.numel() else 0.0,
        "v67_semz_random_seed": seed,
    })
    return out.clamp(0.0, 1.0).reshape(base_shape), debug


def _semantic_anchor_rescue_patch(
    base_patch: torch.Tensor,
    *,
    token_type: torch.Tensor,
    prior_output: Optional[PriorOutput],
    gg_deep_static_patch: torch.Tensor,
    conf_patch: torch.Tensor,
    key_avg_patch: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    base = torch.nan_to_num(
        base_patch.detach().cpu().float().reshape(-1),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    n_patch = int(base.numel())
    debug: Dict[str, Any] = {
        "semantic_anchor_rescue_applied": False,
        "semantic_anchor_rescue_alpha": float(alpha),
        "semantic_anchor_rescue_reason": "ok",
    }
    if prior_output is None or n_patch <= 0:
        debug["semantic_anchor_rescue_reason"] = "missing_prior_or_empty_patch"
        return base, debug
    G_sem_tok = getattr(prior_output, "G_sem_tok", None)
    Q_sem_tok = getattr(prior_output, "Q_sem_tok", None)
    L_sem_tok = getattr(prior_output, "L_sem_tok", None)
    label_patch = _patch_labels_from_token_labels(L_sem_tok, token_type)
    group_patch = _patch_labels_from_token_labels(G_sem_tok, token_type)
    trust_patch = None
    if Q_sem_tok is not None:
        q = Q_sem_tok.detach().cpu().float().reshape(-1)
        token_type_cpu = token_type.detach().cpu().long().reshape(-1)
        if int(q.numel()) == int(token_type_cpu.numel()):
            trust_patch = q[token_type_cpu == TOKEN_TYPE_PATCH].clamp(0.0, 1.0)
    sem_score, sem_debug = _stable_semantic_score(label_patch, group_patch, trust_patch, n_patch)
    debug.update({f"rescue_{k}": v for k, v in sem_debug.items()})
    if int(sem_score.numel()) != n_patch or not bool((sem_score > 0.0).any()):
        debug["semantic_anchor_rescue_reason"] = "empty_semantic_stable_score"
        return base, debug
    static_score = _flatten_or_default(gg_deep_static_patch, base).clamp(0.0, 1.0)
    conf_score = _flatten_or_default(conf_patch, base).clamp(0.0, 1.0)
    source_score = 0.25 + 0.75 * _normalize01(_flatten_or_default(key_avg_patch, base))
    rescue = (sem_score * (0.5 + 0.5 * static_score) * conf_score * source_score).clamp(0.0, 1.0)
    out = (base * (1.0 - float(alpha) * rescue)).clamp(0.0, 1.0)
    debug.update({
        "semantic_anchor_rescue_applied": bool((out - base).abs().max().item() > 1e-8),
        "semantic_anchor_rescue_score_mean": float(rescue.mean().item()) if rescue.numel() else 0.0,
        "semantic_anchor_rescue_score_q90": float(torch.quantile(rescue, 0.9).item()) if rescue.numel() else 0.0,
        "semantic_anchor_rescue_base_mean": float(base.mean().item()) if base.numel() else 0.0,
        "semantic_anchor_rescue_after_mean": float(out.mean().item()) if out.numel() else 0.0,
        "semantic_anchor_rescue_delta_mean": float((out - base).mean().item()) if out.numel() else 0.0,
        "semantic_anchor_rescue_static_score_mean": float(static_score.mean().item()) if static_score.numel() else 0.0,
    })
    return out, debug


def _parse_v67_semantic_cue(read_cue_source: str) -> Optional[Dict[str, Any]]:
    src = str(read_cue_source or "").strip().lower()
    if not src.startswith("v67.semcue_"):
        return None
    body = src[len("v67.semcue_"):]
    random_same_distribution = False
    for suffix in ("_random_same_mass", ".random_same_mass", "_random_same_distribution", ".random_same_distribution"):
        if body.endswith(suffix):
            random_same_distribution = True
            body = body[: -len(suffix)]
            break
    aliases = {
        "anchor_support": "anchor_support",
        "support": "anchor_support",
        "stable_support": "anchor_support",
        "anchor_risk": "anchor_risk",
        "risk": "anchor_risk",
        "low_observability": "anchor_risk",
    }
    kind = aliases.get(body)
    if kind is None:
        return None
    return {
        "kind": kind,
        "random_same_distribution": bool(random_same_distribution),
    }


def _strip_v68_control_suffix(body: str) -> Tuple[str, str]:
    control = "none"
    suffixes = (
        ("confidence_neutral_anchor_comp_same_attention_mass_random", "confidence_neutral_anchor_comp_same_attention_mass_random"),
        ("confidence_neutral_anchor_comp_group_stratified_random", "confidence_neutral_anchor_comp_group_stratified_random"),
        ("confidence_neutral_anchor_comp_label_shuffled", "confidence_neutral_anchor_comp_label_shuffled"),
        ("confidence_neutral_anchor_comp", "confidence_neutral_anchor_comp"),
        ("confidence_neutral_same_attention_mass_random", "confidence_neutral_same_attention_mass_random"),
        ("confidence_neutral_group_stratified_random", "confidence_neutral_group_stratified_random"),
        ("confidence_neutral_label_shuffled", "confidence_neutral_label_shuffled"),
        ("confidence_neutral", "confidence_neutral"),
        ("same_composition_and_dg_random", "same_composition_and_dg_random"),
        ("same_composition_dg_random", "same_composition_and_dg_random"),
        ("random_same_composition_and_dg", "same_composition_and_dg_random"),
        ("same_attention_mass_random", "same_attention_mass_random"),
        ("random_same_attention_mass", "same_attention_mass_random"),
        ("group_stratified_random", "group_stratified_random"),
        ("random_group_stratified", "group_stratified_random"),
        ("lowstuff_within_group_random", "lowstuff_within_group_random"),
        ("random_lowstuff_within_group", "lowstuff_within_group_random"),
        ("high_d_within_group_random", "high_d_within_group_random"),
        ("highd_within_group_random", "high_d_within_group_random"),
        ("random_high_d_within_group", "high_d_within_group_random"),
        ("random_highd_within_group", "high_d_within_group_random"),
        ("same_cue_distribution_random", "same_cue_distribution_random"),
        ("random_same_distribution", "same_cue_distribution_random"),
        ("random_same_mass", "same_cue_distribution_random"),
        ("label_confidence_shuffled", "joint_label_confidence_shuffled"),
        ("joint_label_confidence_shuffled", "joint_label_confidence_shuffled"),
        ("joint_shuffled", "joint_label_confidence_shuffled"),
        ("label_shuffled", "label_shuffled"),
        ("confidence_shuffled", "confidence_shuffled"),
        ("geometry_only", "geometry_only"),
        ("geo_only", "geometry_only"),
        ("semantic_only", "semantic_only"),
        ("sem_only", "semantic_only"),
    )
    changed = True
    while changed:
        changed = False
        for suffix, name in suffixes:
            for sep in (".", "_"):
                token = f"{sep}{suffix}"
                if body.endswith(token):
                    control = name
                    body = body[: -len(token)]
                    changed = True
                    break
            if changed:
                break
    return body, control


def _deterministic_permutation(indices: torch.Tensor, *, seed: float) -> torch.Tensor:
    idx = indices.detach().cpu().long().reshape(-1)
    if int(idx.numel()) <= 1:
        return idx
    noise = _deterministic_patch_noise(int(idx.numel()), seed=float(seed))
    return idx[torch.argsort(noise, stable=True)]


def _shuffle_values_within_strata(values: torch.Tensor, strata: Optional[torch.Tensor], *, seed: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
    vals = values.detach().cpu().float().reshape(-1)
    n = int(vals.numel())
    out = vals.clone()
    debug: Dict[str, Any] = {
        "v68_read_control_shuffle_mode": "strata",
        "v68_read_control_shuffle_tokens": int(n),
        "v68_read_control_shuffle_groups": 0,
        "v68_read_control_shuffle_fallback": False,
    }
    if n <= 1:
        debug["v68_read_control_shuffle_reason"] = "too_few_tokens"
        return out, debug
    if strata is None or int(strata.numel()) != n:
        perm = _deterministic_permutation(torch.arange(n), seed=seed)
        debug.update({
            "v68_read_control_shuffle_mode": "global_fallback",
            "v68_read_control_shuffle_fallback": True,
            "v68_read_control_shuffle_reason": "missing_or_mismatched_strata",
        })
        return vals[perm], debug
    labels = strata.detach().cpu().long().reshape(-1)
    groups = torch.unique(labels)
    shuffled_groups = 0
    for group in groups.tolist():
        idx = torch.nonzero(labels == int(group), as_tuple=False).reshape(-1)
        if int(idx.numel()) <= 1:
            continue
        perm = _deterministic_permutation(idx, seed=seed + float(int(group) + 1009))
        out[idx] = vals[perm]
        shuffled_groups += 1
    debug.update({
        "v68_read_control_shuffle_groups": int(shuffled_groups),
        "v68_read_control_shuffle_reason": "ok" if shuffled_groups > 0 else "all_strata_singleton",
    })
    return out, debug


def _shuffle_values_in_eligible_strata(
    values: torch.Tensor,
    eligible: torch.Tensor,
    strata: Optional[torch.Tensor],
    *,
    seed: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    vals = values.detach().cpu().float().reshape(-1)
    n = int(vals.numel())
    mask = eligible.detach().cpu().bool().reshape(-1) if eligible is not None else torch.zeros((n,), dtype=torch.bool)
    if int(mask.numel()) != n:
        mask = torch.zeros((n,), dtype=torch.bool)
    out = vals.clone()
    debug: Dict[str, Any] = {
        "v68_read_control_shuffle_mode": "eligible_strata",
        "v68_read_control_eligible_tokens": int(mask.sum().item()),
        "v68_read_control_shuffle_fallback": False,
    }
    if n <= 1 or int(mask.sum().item()) <= 1:
        debug["v68_read_control_shuffle_reason"] = "too_few_eligible_tokens"
        return out, debug
    labels = strata.detach().cpu().long().reshape(-1) if strata is not None and int(strata.numel()) == n else torch.zeros((n,), dtype=torch.long)
    shuffled_groups = 0
    for group in torch.unique(labels[mask]).tolist():
        idx = torch.nonzero(mask & (labels == int(group)), as_tuple=False).reshape(-1)
        if int(idx.numel()) <= 1:
            continue
        perm = _deterministic_permutation(idx, seed=seed + float(int(group) + 2027))
        out[idx] = vals[perm]
        shuffled_groups += 1
    debug.update({
        "v68_read_control_shuffle_groups": int(shuffled_groups),
        "v68_read_control_shuffle_reason": "ok" if shuffled_groups > 0 else "all_eligible_strata_singleton",
    })
    return out, debug


def _v68_control_patch_metadata(
    *,
    token_type: torch.Tensor,
    prior_output: Optional[PriorOutput],
    motion: torch.Tensor,
) -> Dict[str, Optional[torch.Tensor]]:
    token_type_cpu = token_type.detach().cpu().long().reshape(-1)
    patch_mask = token_type_cpu == TOKEN_TYPE_PATCH
    n_patch = int(patch_mask.sum().item())
    groups = None
    labels = None
    if prior_output is not None:
        groups = _patch_labels_from_token_labels(getattr(prior_output, "G_sem_tok", None), token_type_cpu)
        labels = _patch_labels_from_token_labels(getattr(prior_output, "L_sem_tok", None), token_type_cpu)
    groups = groups.detach().cpu().long().reshape(-1) if groups is not None and int(groups.numel()) == n_patch else None
    labels = labels.detach().cpu().long().reshape(-1) if labels is not None and int(labels.numel()) == n_patch else None
    lowstuff = torch.zeros((n_patch,), dtype=torch.bool)
    if groups is not None:
        lowstuff |= groups == int(SEMANTIC_GROUP_LOW_VALUE_STUFF)
    if labels is not None:
        for label_id in SEMANTIC_FINE_VEGETATION_IDS:
            lowstuff |= labels == int(label_id)
        for label_id in _V68_LOWOBS_FINE_IDS:
            lowstuff |= labels == int(label_id)
    mot = torch.nan_to_num(motion.detach().cpu().float().reshape(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if int(mot.numel()) != n_patch:
        mot = torch.zeros((n_patch,), dtype=torch.float32)
    high_d = torch.zeros((n_patch,), dtype=torch.bool)
    if int(mot.numel()) > 0:
        high_d = mot >= torch.quantile(mot, 0.75)
    return {
        "groups": groups,
        "labels": labels,
        "lowstuff": lowstuff,
        "high_d": high_d,
        "motion": mot,
    }


def _apply_v68_read_output_control(
    values: torch.Tensor,
    *,
    control: str,
    token_type: torch.Tensor,
    prior_output: Optional[PriorOutput],
    motion: torch.Tensor,
    num_frames: int,
    chunk_idx: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    vals = values.detach().cpu().float().reshape(-1)
    control_l = str(control or "none").strip().lower()
    post_control_l = control_l
    if control_l == "confidence_neutral_same_attention_mass_random":
        post_control_l = "same_attention_mass_random"
    elif control_l == "confidence_neutral_group_stratified_random":
        post_control_l = "group_stratified_random"
    elif control_l == "confidence_neutral_anchor_comp_same_attention_mass_random":
        post_control_l = "same_attention_mass_random"
    elif control_l == "confidence_neutral_anchor_comp_group_stratified_random":
        post_control_l = "group_stratified_random"
    elif control_l in {"confidence_neutral", "confidence_neutral_label_shuffled"}:
        post_control_l = "none"
    elif control_l in {"confidence_neutral_anchor_comp", "confidence_neutral_anchor_comp_label_shuffled"}:
        post_control_l = "none"
    seed = 6817000.0 + max(int(chunk_idx), 0) * 101.0
    debug: Dict[str, Any] = {
        "v68_read_control_applied": False,
        "v68_read_control_effective": control_l,
        "v68_read_control_post_effective": post_control_l,
    }
    if post_control_l == "geometry_only":
        mot = torch.nan_to_num(motion.detach().cpu().float().reshape(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if int(mot.numel()) != int(vals.numel()):
            debug.update({
                "v68_read_control_shuffle_reason": "geometry_motion_size_mismatch",
                "v68_read_control_motion_tokens": int(mot.numel()),
                "v68_read_control_value_tokens": int(vals.numel()),
            })
            return vals, debug
        debug.update({
            "v68_read_control_applied": True,
            "v68_read_control_shuffle_mode": "geometry_only_motion_replace",
            "v68_read_control_eligible_tokens": int(vals.numel()),
        })
        return mot, debug
    if post_control_l not in {
        "same_cue_distribution_random",
        "same_attention_mass_random",
        "group_stratified_random",
        "lowstuff_within_group_random",
        "high_d_within_group_random",
        "same_composition_and_dg_random",
    }:
        return vals, debug

    meta = _v68_control_patch_metadata(token_type=token_type, prior_output=prior_output, motion=motion)
    groups = meta.get("groups")
    lowstuff = meta.get("lowstuff")
    high_d = meta.get("high_d")
    mot = meta.get("motion")
    if post_control_l == "same_cue_distribution_random":
        perm = _deterministic_permutation(torch.arange(int(vals.numel())), seed=seed)
        out = vals[perm]
        debug.update({
            "v68_read_control_applied": True,
            "v68_read_control_shuffle_mode": "global_same_distribution",
            "v68_read_control_eligible_tokens": int(vals.numel()),
        })
        return out, debug
    if post_control_l == "same_attention_mass_random":
        n = int(vals.numel())
        T = max(int(num_frames), 1)
        if n > 1 and n % T == 0:
            per_frame = n // T
            out = vals.clone()
            for frame_idx in range(T):
                idx = torch.arange(frame_idx * per_frame, (frame_idx + 1) * per_frame, dtype=torch.long)
                perm = _deterministic_permutation(idx, seed=seed + 31.0 * frame_idx)
                out[idx] = vals[perm]
            debug.update({
                "v68_read_control_applied": True,
                "v68_read_control_shuffle_mode": "per_frame_same_distribution",
                "v68_read_control_frames": int(T),
                "v68_read_control_tokens_per_frame": int(per_frame),
            })
            return out, debug
        out, sub_debug = _shuffle_values_within_strata(vals, None, seed=seed)
        debug.update(sub_debug)
        debug["v68_read_control_applied"] = True
        return out, debug
    if post_control_l == "group_stratified_random":
        out, sub_debug = _shuffle_values_within_strata(vals, groups, seed=seed)
        debug.update(sub_debug)
        debug["v68_read_control_applied"] = True
        return out, debug
    if post_control_l == "lowstuff_within_group_random":
        eligible = lowstuff if lowstuff is not None else torch.zeros_like(vals, dtype=torch.bool)
        out, sub_debug = _shuffle_values_in_eligible_strata(vals, eligible, groups, seed=seed)
        debug.update(sub_debug)
        debug["v68_read_control_applied"] = True
        debug["v68_read_control_lowstuff_tokens"] = int(eligible.sum().item()) if eligible is not None else 0
        return out, debug
    if post_control_l == "high_d_within_group_random":
        eligible = high_d if high_d is not None else torch.zeros_like(vals, dtype=torch.bool)
        out, sub_debug = _shuffle_values_in_eligible_strata(vals, eligible, groups, seed=seed)
        debug.update(sub_debug)
        debug["v68_read_control_applied"] = True
        debug["v68_read_control_high_d_tokens"] = int(eligible.sum().item()) if eligible is not None else 0
        return out, debug
    if post_control_l == "same_composition_and_dg_random":
        n = int(vals.numel())
        if mot is not None and int(mot.numel()) == n:
            q1 = torch.quantile(mot, 0.25)
            q2 = torch.quantile(mot, 0.50)
            q3 = torch.quantile(mot, 0.75)
            d_bin = torch.zeros((n,), dtype=torch.long)
            d_bin += (mot > q1).long()
            d_bin += (mot > q2).long()
            d_bin += (mot > q3).long()
        else:
            d_bin = torch.zeros((n,), dtype=torch.long)
        base_group = groups if groups is not None and int(groups.numel()) == n else torch.zeros((n,), dtype=torch.long)
        strata = (base_group + 10).long() * 4 + d_bin.long()
        out, sub_debug = _shuffle_values_within_strata(vals, strata, seed=seed)
        debug.update(sub_debug)
        debug["v68_read_control_applied"] = True
        debug["v68_read_control_shuffle_mode"] = "semantic_group_x_dquartile"
        return out, debug
    return vals, debug


def _parse_v68_layer_list(body: str) -> List[int]:
    layers: List[int] = []
    parts = [p for p in str(body or "").replace("-", "_").replace(".", "_").split("_") if p]
    i = 0
    while i < len(parts):
        token = parts[i].strip().lower()
        parsed: List[int] = []
        if token.startswith("layers") and token[len("layers"):].isdigit():
            parsed.append(int(token[len("layers"):]))
        elif token.startswith("l") and token[1:].isdigit():
            parsed.append(int(token[1:]))
        if parsed:
            j = i + 1
            while j < len(parts) and parts[j].isdigit():
                parsed.append(int(parts[j]))
                j += 1
            layers.extend(parsed)
            i = j
            continue
        i += 1
    out: List[int] = []
    seen = set()
    for layer in layers or [5, 7]:
        li = int(layer)
        if li not in seen:
            out.append(li)
            seen.add(li)
    return out


def _parse_v68_read_cue(read_cue_source: str) -> Optional[Dict[str, Any]]:
    src = str(read_cue_source or "").strip().lower()
    if not src.startswith("v68.read."):
        return None
    body = src[len("v68.read."):]
    body, control = _strip_v68_control_suffix(body)
    tap = "global_k_raw_patchvec_layers"
    if ".v" in body or "_v" in body or "global_v" in body or "value" in body:
        tap = "global_v_raw_patchvec_layers"
    elif ".q" in body or "_q" in body or "global_q" in body:
        tap = "global_q_raw_patchvec_layers"
    if "semantic_only" in body or "sem_only" in body:
        fusion = "semantic_only"
    elif "geometry_only" in body or "geo_only" in body:
        fusion = "geometry_only"
    elif "support_gate" in body or "semsupport" in body:
        fusion = "support_gate"
    else:
        fusion = "motion_semrisk"
    layers = _parse_v68_layer_list(body)
    return {
        "path": "read",
        "body": body,
        "tap": tap,
        "layers": layers,
        "fusion": fusion,
        "control": control,
    }


def _parse_v78_l07_l13_read_cue(read_cue_source: str) -> Optional[Dict[str, Any]]:
    src = str(read_cue_source or "").strip().lower()
    if not src.startswith("v78.l07_l13."):
        return None
    body = src[len("v78.l07_l13."):]
    body, control = _strip_v68_control_suffix(body)
    aliases = {
        "l13_only": "l13_only",
        "g1_l13_only": "l13_only",
        "l07_action_only": "l07_action_only",
        "g2_l07_action_only": "l07_action_only",
        "mask_neg": "l07_mask_l13_neg",
        "l07_mask_l13_neg": "l07_mask_l13_neg",
        "g3_l07_mask_l13_neg": "l07_mask_l13_neg",
        "mask_stable": "l07_mask_l13_stable",
        "l07_mask_l13_stable": "l07_mask_l13_stable",
        "g4_l07_mask_l13_stable": "l07_mask_l13_stable",
        "mask_neg_plus_stable": "l07_mask_l13_neg_plus_stable",
        "l07_mask_l13_neg_plus_stable": "l07_mask_l13_neg_plus_stable",
        "g5_l07_mask_l13_neg_plus_stable": "l07_mask_l13_neg_plus_stable",
        "contrast": "l07_mask_l13_contrast",
        "l07_mask_l13_contrast": "l07_mask_l13_contrast",
        "g6_l07_mask_l13_contrast": "l07_mask_l13_contrast",
    }
    mode = aliases.get(body)
    if mode is None:
        return None
    return {
        "body": body,
        "mode": mode,
        "control": control,
        "l07_layer": 7,
        "l13_layer": 13,
    }


_V95_TRACKH_GATED_L07_SOURCES = {
    "gate.fa_key_all_mean_ge_q60.then_v78_l07",
    "gate.fa_key_middle_mean_ge_q60.then_v78_l07",
    "gate.gg_qk_shallow_mean_le_q40.then_v78_l07",
    "gate.gg_qk_shallow_mean_le_q40.chunk_ge_6.then_v78_l07",
    "gate.gg_qk_shallow_mean_le_q40.chunk_ge_6.rtok_ge_0p005.then_v78_l07",
    "gate.gg_qk_shallow_mean_le_q40.chunk_ge_33.rtok_ge_0p005.then_v78_l07",
    "gate.gg_qk_shallow_mean_le_q40.chunk_eq_33.rtok_ge_0p005.then_v78_l07",
    "gate.gg_qk_shallow_mean_le_q40.chunk_eq_36.rtok_ge_0p005.then_v78_l07",
}

_V95_TRACKH_GATED_L07_CONTROL_SUFFIX_TO_CUE = {
    ".geometry_only": "v78.l07_l13.l07_action_only.geometry_only",
    ".label_shuffled": "v78.l07_l13.l07_action_only.label_shuffled",
    ".confidence_shuffled": "v78.l07_l13.l07_action_only.confidence_shuffled",
    ".same_attention_mass_random": "v78.l07_l13.l07_action_only.same_attention_mass_random",
    ".group_stratified_random": "v78.l07_l13.l07_action_only.group_stratified_random",
    ".confidence_neutral": "v78.l07_l13.l07_action_only.confidence_neutral",
    ".confidence_neutral_label_shuffled": "v78.l07_l13.l07_action_only.confidence_neutral_label_shuffled",
    ".confidence_neutral_same_attention_mass_random": "v78.l07_l13.l07_action_only.confidence_neutral_same_attention_mass_random",
    ".confidence_neutral_group_stratified_random": "v78.l07_l13.l07_action_only.confidence_neutral_group_stratified_random",
    ".confidence_neutral_anchor_comp": "v78.l07_l13.l07_action_only.confidence_neutral_anchor_comp",
    ".confidence_neutral_anchor_comp_label_shuffled": "v78.l07_l13.l07_action_only.confidence_neutral_anchor_comp_label_shuffled",
    ".confidence_neutral_anchor_comp_same_attention_mass_random": "v78.l07_l13.l07_action_only.confidence_neutral_anchor_comp_same_attention_mass_random",
    ".confidence_neutral_anchor_comp_group_stratified_random": "v78.l07_l13.l07_action_only.confidence_neutral_anchor_comp_group_stratified_random",
}

_V95_TRACKH_GATED_L07_SOURCES = _V95_TRACKH_GATED_L07_SOURCES | {
    f"gate.gg_qk_shallow_mean_le_q40.chunk_ge_{min_chunk}.rtok_ge_0p005.then_v78_l07{suffix}"
    for min_chunk in (6, 33)
    for suffix in _V95_TRACKH_GATED_L07_CONTROL_SUFFIX_TO_CUE
} | {
    f"gate.gg_qk_shallow_mean_le_q40.chunk_eq_{chunk_idx}.rtok_ge_0p005.then_v78_l07{suffix}"
    for chunk_idx in (33, 36)
    for suffix in _V95_TRACKH_GATED_L07_CONTROL_SUFFIX_TO_CUE
}


def _is_v95_trackh_gated_l07_source(read_cue_source: str) -> bool:
    return str(read_cue_source or "").strip() in _V95_TRACKH_GATED_L07_SOURCES


def _v95_trackh_gated_l07_gate_key(read_cue_source: str) -> str:
    src = str(read_cue_source or "").strip()
    for suffix in _V95_TRACKH_GATED_L07_CONTROL_SUFFIX_TO_CUE:
        if src.endswith(suffix):
            return src[: -len(suffix)]
    return src


def _v95_trackh_gated_l07_active_read_cue(read_cue_source: str) -> str:
    src = str(read_cue_source or "").strip()
    for suffix, cue in _V95_TRACKH_GATED_L07_CONTROL_SUFFIX_TO_CUE.items():
        if src.endswith(suffix):
            return cue
    return "v78.l07_l13.l07_action_only"


def _read_cue_requires_semantic_prior(read_cue_source: str) -> bool:
    src = str(read_cue_source or "").strip()
    lowered = src.lower()
    if lowered.startswith("v68.read."):
        return not any(token in lowered for token in (".geometry_only", "_geometry_only", ".geo_only", "_geo_only"))
    if lowered.startswith("v78.l07_l13."):
        return True
    if _is_v95_trackh_gated_l07_source(lowered):
        return True
    if lowered.startswith("v67.semgeo_"):
        return True
    if lowered.startswith("v67.semcue_"):
        return True
    if lowered.startswith("v67.semz_") and lowered.endswith(".c23past"):
        return True
    if lowered in {"v31.sem_z_fine.c23past", "v31.sem_z_coarse.c23past"}:
        return True
    if lowered.startswith(("v31.sem_resid_fine_l", "v31.sem_resid_coarse_l")) and lowered.endswith(".c23past"):
        return True
    if lowered.startswith("semantic_anchor_rescue."):
        return True
    return False


def _strip_v70_radio_control_suffix(body: str) -> Tuple[str, str]:
    control = "none"
    suffixes = (
        ("same_cue_distribution_random", "same_cue_distribution_random"),
        ("random_same_distribution", "same_cue_distribution_random"),
        ("random_same_mass", "same_cue_distribution_random"),
        ("radio_feature_shuffle", "radio_feature_shuffle"),
        ("feature_shuffle", "radio_feature_shuffle"),
        ("radio_risk_shuffle", "radio_risk_shuffle"),
        ("risk_shuffle", "radio_risk_shuffle"),
        ("radio_component_shuffle", "radio_component_shuffle"),
        ("component_shuffle", "radio_component_shuffle"),
        ("geometry_only", "geometry_only"),
        ("geo_only", "geometry_only"),
    )
    changed = True
    while changed:
        changed = False
        for suffix, name in suffixes:
            for sep in (".", "_"):
                token = f"{sep}{suffix}"
                if body.endswith(token):
                    control = name
                    body = body[: -len(token)]
                    changed = True
                    break
            if changed:
                break
    return body, control


def _parse_v70_radio_read_cue(read_cue_source: str) -> Optional[Dict[str, Any]]:
    src = str(read_cue_source or "").strip().lower()
    if not src.startswith("v70.radio."):
        return None
    body = src[len("v70.radio."):]
    body, control = _strip_v70_radio_control_suffix(body)
    if body in {"object_interior_floor", "object_interior", "interior_floor", "read_r3"}:
        mode = "object_interior_floor"
    elif body in {"cross_object_risk_veto", "risk_veto", "cross_object_veto", "read_r4"}:
        mode = "cross_object_risk_veto"
    else:
        return {
            "body": body,
            "mode": "unknown",
            "control": control,
        }
    return {
        "body": body,
        "mode": mode,
        "control": control,
    }


def _v70_radio_sidecar_path(sidecar_dir: str, chunk_idx: int) -> Optional[Path]:
    root = Path(str(sidecar_dir or "")).expanduser()
    if not root.is_dir():
        return None
    hits = sorted(root.glob(f"chunk_{int(chunk_idx):03d}_*/radio_sidecar.pt"))
    return hits[0] if hits else None


def _semantic_overlap_support_path(support_dir: str, chunk_idx: int, kind: str) -> Optional[Path]:
    root = Path(str(support_dir or "")).expanduser()
    if not root.is_dir():
        return None
    hits = sorted(root.glob(f"**/chunk_{int(chunk_idx):03d}_swa_overlap_*_layer_*.pt"))
    preferred_kind = str(kind or "").strip()
    if preferred_kind:
        preferred = [p for p in hits if f"_swa_overlap_{preferred_kind}_" in p.name]
        if preferred:
            hits = preferred
    return hits[0] if hits else None


def _v70_radio_resize_map(value: Any, *, num_frames: int, patch_grid: Tuple[int, int]) -> torch.Tensor:
    tensor = value.detach().cpu().float() if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
    tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=1.0, neginf=0.0)
    if tensor.ndim != 3:
        return torch.zeros((max(int(num_frames), 0) * max(int(patch_grid[0]), 0) * max(int(patch_grid[1]), 0),), dtype=torch.float32)
    target_t = max(int(num_frames), 1)
    target_h, target_w = max(int(patch_grid[0]), 1), max(int(patch_grid[1]), 1)
    if tuple(int(x) for x in tensor.shape) != (target_t, target_h, target_w):
        tensor = F.interpolate(
            tensor[None, None],
            size=(target_t, target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )[0, 0]
    return tensor.reshape(-1).clamp(0.0, 1.0)


def _v70_radio_read_cue_patch(
    *,
    sidecar_dir: str,
    cfg: Dict[str, Any],
    default: torch.Tensor,
    num_frames: int,
    patch_grid: Tuple[int, int],
    chunk_idx: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    control = str(cfg.get("control", "none"))
    mode = str(cfg.get("mode", "unknown"))
    debug: Dict[str, Any] = {
        "v70_radio_read_available": False,
        "v70_radio_read_reason": "not_run",
        "v70_radio_read_cue_source": f"v70.radio.{cfg.get('body', '')}",
        "v70_radio_read_mode": mode,
        "v70_radio_read_control": control,
        "v70_radio_sidecar_dir": str(sidecar_dir or ""),
        "v70_radio_chunk_idx": int(chunk_idx),
    }
    if mode == "unknown":
        debug["v70_radio_read_reason"] = f"unknown_mode:{cfg.get('body', '')}"
        return torch.zeros_like(default), debug
    sidecar_path = _v70_radio_sidecar_path(sidecar_dir, chunk_idx)
    if sidecar_path is None:
        debug["v70_radio_read_reason"] = "missing_sidecar"
        return torch.zeros_like(default), debug
    try:
        try:
            payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(sidecar_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        debug["v70_radio_read_reason"] = f"sidecar_load_error:{type(exc).__name__}:{exc}"
        debug["v70_radio_sidecar_path"] = str(sidecar_path)
        return torch.zeros_like(default), debug
    required = (
        "object_interior_score",
        "object_boundary_score",
        "radio_dynamic_score",
        "radio_sky_context_score",
        "radio_lowtrust_score",
        "temporal_stability",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        debug["v70_radio_read_reason"] = f"missing_fields:{','.join(missing)}"
        debug["v70_radio_sidecar_path"] = str(sidecar_path)
        return torch.zeros_like(default), debug

    interior = _v70_radio_resize_map(payload["object_interior_score"], num_frames=num_frames, patch_grid=patch_grid)
    boundary = _v70_radio_resize_map(payload["object_boundary_score"], num_frames=num_frames, patch_grid=patch_grid)
    dynamic = _v70_radio_resize_map(payload["radio_dynamic_score"], num_frames=num_frames, patch_grid=patch_grid)
    sky = _v70_radio_resize_map(payload["radio_sky_context_score"], num_frames=num_frames, patch_grid=patch_grid)
    lowtrust = _v70_radio_resize_map(payload["radio_lowtrust_score"], num_frames=num_frames, patch_grid=patch_grid)
    stability = _v70_radio_resize_map(payload["temporal_stability"], num_frames=num_frames, patch_grid=patch_grid)
    geom_default = default.detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
    risk = torch.maximum(boundary, torch.maximum(dynamic, torch.maximum(sky, lowtrust))).clamp(0.0, 1.0)
    interior_static = (interior * stability).clamp(0.0, 1.0)
    if control in {"radio_feature_shuffle", "same_cue_distribution_random"}:
        noise = _deterministic_patch_noise(int(interior_static.numel()), seed=7003000 + max(int(chunk_idx), 0) * 101.0)
        perm = torch.argsort(noise, stable=True)
        interior_static = interior_static[perm]
        risk = risk[perm]
    elif control == "radio_risk_shuffle":
        noise = _deterministic_patch_noise(int(risk.numel()), seed=7003100 + max(int(chunk_idx), 0) * 101.0)
        risk = risk[torch.argsort(noise, stable=True)]
    elif control == "radio_component_shuffle":
        noise = _deterministic_patch_noise(int(interior_static.numel()), seed=7003200 + max(int(chunk_idx), 0) * 101.0)
        interior_static = interior_static[torch.argsort(noise, stable=True)]
    if mode == "cross_object_risk_veto":
        out = risk
    elif control == "geometry_only":
        out = geom_default
    else:
        # Geometry-first READ treats D as source risk.  Keep the stage-C
        # geometry risk as the base signal, then lower risk only on stable
        # RADIO interiors while retaining boundary/dynamic/sky/low-trust risk.
        stable_interior = (interior_static * (1.0 - risk)).clamp(0.0, 1.0)
        risk_boost = (risk * (1.0 - interior_static)).clamp(0.0, 1.0)
        out = (geom_default * (1.0 - 0.55 * stable_interior) + 0.25 * risk_boost).clamp(0.0, 1.0)
    out_before_norm = out.detach().cpu().float().reshape(-1)
    out = _robust_quantile01(out_before_norm, num_frames=max(int(num_frames), 1))
    out = _flatten_or_default(out, default).clamp(0.0, 1.0)
    sidecar_grid = payload.get("patch_grid", None)
    debug.update({
        "v70_radio_read_available": True,
        "v70_radio_read_reason": "ok",
        "v70_radio_sidecar_path": str(sidecar_path),
        "v70_radio_sidecar_patch_grid": list(sidecar_grid) if isinstance(sidecar_grid, (list, tuple)) else None,
        "v70_radio_target_patch_grid": [int(patch_grid[0]), int(patch_grid[1])],
        "v70_radio_sidecar_frames": int(payload["object_interior_score"].shape[0]),
        "v70_radio_target_frames": int(num_frames),
        "v70_radio_interior_mean": float(interior.mean().item()) if interior.numel() else 0.0,
        "v70_radio_interior_static_mean": float(interior_static.mean().item()) if interior_static.numel() else 0.0,
        "v70_radio_risk_mean": float(risk.mean().item()) if risk.numel() else 0.0,
        "v70_radio_risk_q90": float(torch.quantile(risk, 0.90).item()) if risk.numel() else 0.0,
        "v70_radio_geom_default_mean": float(geom_default.mean().item()) if geom_default.numel() else 0.0,
        "v70_radio_geom_default_q90": float(torch.quantile(geom_default, 0.90).item()) if geom_default.numel() else 0.0,
        "v70_radio_output_mean_before_norm": float(out_before_norm.mean().item()) if out_before_norm.numel() else 0.0,
        "v70_radio_output_q90_before_norm": float(torch.quantile(out_before_norm, 0.90).item()) if out_before_norm.numel() else 0.0,
        "v70_radio_output_mean": float(out.mean().item()) if out.numel() else 0.0,
        "v70_radio_output_q90": float(torch.quantile(out, 0.90).item()) if out.numel() else 0.0,
        "v70_radio_output_gt050_mass": float((out > 0.50).float().mean().item()) if out.numel() else 0.0,
        "v70_radio_corr_output_risk": _pearson_corr(out, risk),
        "v70_radio_corr_output_interior_static": _pearson_corr(out, interior_static),
    })
    return out, debug


def _v68_patch_semantic_support_risk(
    *,
    token_type: torch.Tensor,
    prior_output: Optional[PriorOutput],
    control: str,
    chunk_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    token_type_cpu = token_type.detach().cpu().long().reshape(-1)
    patch_mask = token_type_cpu == TOKEN_TYPE_PATCH
    n_patch = int(patch_mask.sum().item())
    zeros = torch.zeros((n_patch,), dtype=torch.float32)
    debug: Dict[str, Any] = {
        "v68_read_semantic_available": False,
        "v68_read_semantic_control": str(control),
        "v68_read_patch_tokens": int(n_patch),
    }
    if prior_output is None or n_patch <= 0:
        debug["v68_read_semantic_reason"] = "missing_prior_or_empty_patch"
        return zeros, torch.ones((n_patch,), dtype=torch.float32), zeros, debug

    L_patch = _patch_labels_from_token_labels(getattr(prior_output, "L_sem_tok", None), token_type_cpu)
    G_patch = _patch_labels_from_token_labels(getattr(prior_output, "G_sem_tok", None), token_type_cpu)
    Q_patch = None
    Q_sem_tok = getattr(prior_output, "Q_sem_tok", None)
    if Q_sem_tok is not None:
        q = Q_sem_tok.detach().cpu().float().reshape(-1)
        if int(q.numel()) == int(token_type_cpu.numel()):
            Q_patch = q[patch_mask].clamp(0.0, 1.0)
    if Q_patch is None or int(Q_patch.numel()) != n_patch:
        debug["v68_read_semantic_reason"] = "missing_trust"
        return zeros, torch.ones((n_patch,), dtype=torch.float32), zeros, debug

    control_l = str(control).strip().lower()
    trust = Q_patch.detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
    labels = L_patch.detach().cpu().long().reshape(-1) if L_patch is not None and int(L_patch.numel()) == n_patch else None
    groups = G_patch.detach().cpu().long().reshape(-1) if G_patch is not None and int(G_patch.numel()) == n_patch else None

    seed = 6809000 + max(int(chunk_idx), 0) * 101.0
    if control_l in {
        "label_shuffled",
        "joint_label_confidence_shuffled",
        "confidence_neutral_label_shuffled",
        "confidence_neutral_anchor_comp_label_shuffled",
    }:
        noise = _deterministic_patch_noise(n_patch, seed=seed + 11.0)
        perm = torch.argsort(noise, stable=True)
        if labels is not None:
            labels = labels[perm]
        if groups is not None:
            groups = groups[perm]
    if control_l in {"confidence_shuffled", "joint_label_confidence_shuffled"}:
        noise = _deterministic_patch_noise(n_patch, seed=seed + 23.0)
        perm = torch.argsort(noise, stable=True)
        trust = trust[perm]
    if control_l in {
        "confidence_neutral",
        "confidence_neutral_label_shuffled",
        "confidence_neutral_same_attention_mass_random",
        "confidence_neutral_group_stratified_random",
        "confidence_neutral_anchor_comp",
        "confidence_neutral_anchor_comp_label_shuffled",
        "confidence_neutral_anchor_comp_same_attention_mass_random",
        "confidence_neutral_anchor_comp_group_stratified_random",
    }:
        trust = torch.ones_like(trust)

    vertical = torch.zeros((n_patch,), dtype=torch.bool)
    road = torch.zeros((n_patch,), dtype=torch.bool)
    dynamic = torch.zeros((n_patch,), dtype=torch.bool)
    sky = torch.zeros((n_patch,), dtype=torch.bool)
    lowstuff = torch.zeros((n_patch,), dtype=torch.bool)
    known = torch.zeros((n_patch,), dtype=torch.bool)
    if labels is not None:
        known = (labels >= 0) & (labels != int(SEMANTIC_FINE_LABEL_UNKNOWN))
        for label_id in _V67_SEMCUE_VERTICAL_STATIC_FINE_IDS:
            vertical |= labels == int(label_id)
        for label_id in _V67_SEMCUE_GROUND_STATIC_FINE_IDS:
            road |= labels == int(label_id)
        for label_id in SEMANTIC_FINE_MOVABLE_IDS:
            dynamic |= labels == int(label_id)
        for label_id in SEMANTIC_FINE_SKY_IDS:
            sky |= labels == int(label_id)
        for label_id in SEMANTIC_FINE_VEGETATION_IDS:
            lowstuff |= labels == int(label_id)
        for label_id in _V68_LOWOBS_FINE_IDS:
            lowstuff |= labels == int(label_id)
    if groups is not None:
        vertical |= groups == int(SEMANTIC_GROUP_STRUCTURE_ANCHOR)
        road |= groups == int(SEMANTIC_GROUP_STATIC_THING)
        dynamic |= groups == int(SEMANTIC_GROUP_MOVABLE_THING)
        lowstuff |= groups == int(SEMANTIC_GROUP_LOW_VALUE_STUFF)
        known |= groups >= 0

    support = trust * (vertical.float() + 0.25 * road.float()).clamp(0.0, 1.0)
    label_risk = torch.maximum(dynamic.float(), torch.maximum(0.9 * sky.float(), 0.7 * lowstuff.float()))
    void_lowtrust = (~known).float()
    label_risk = torch.maximum(label_risk, void_lowtrust)
    risk = (trust * label_risk + (1.0 - trust)).clamp(0.0, 1.0)
    lowobs = (
        0.5 * sky.float()
        + 0.4 * road.float()
        + 0.4 * lowstuff.float()
        - 0.8 * vertical.float()
    ).clamp(0.0, 1.0) * trust

    debug.update({
        "v68_read_semantic_available": True,
        "v68_read_semantic_reason": "ok",
        "v68_read_semantic_confidence_neutralized": control_l.startswith("confidence_neutral"),
        "v68_read_semantic_trust_mean": float(trust.mean().item()) if trust.numel() else 0.0,
        "v68_read_semantic_support_mean": float(support.mean().item()) if support.numel() else 0.0,
        "v68_read_semantic_support_q90": float(torch.quantile(support, 0.90).item()) if support.numel() else 0.0,
        "v68_read_semantic_risk_mean": float(risk.mean().item()) if risk.numel() else 0.0,
        "v68_read_semantic_risk_q90": float(torch.quantile(risk, 0.90).item()) if risk.numel() else 0.0,
        "v68_read_semantic_lowobs_mean": float(lowobs.mean().item()) if lowobs.numel() else 0.0,
        "v68_read_semantic_vertical_tokens": int(vertical.sum().item()),
        "v68_read_semantic_road_tokens": int(road.sum().item()),
        "v68_read_semantic_dynamic_tokens": int(dynamic.sum().item()),
        "v68_read_semantic_sky_tokens": int(sky.sum().item()),
        "v68_read_semantic_lowstuff_tokens": int(lowstuff.sum().item()),
        "v68_read_semantic_known_tokens": int(known.sum().item()),
    })
    return support.clamp(0.0, 1.0), risk.clamp(0.0, 1.0), lowobs.clamp(0.0, 1.0), debug


def _v68_grammotion_patch(
    q_layers: Optional[torch.Tensor],
    k_layers: Optional[torch.Tensor],
    v_layers: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    tap: str,
    layers: List[int],
    num_frames: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if str(tap) == "global_v_raw_patchvec_layers":
        tensor = v_layers
    elif str(tap) == "global_q_raw_patchvec_layers":
        tensor = q_layers
    else:
        tensor = k_layers
    debug: Dict[str, Any] = {
        "v68_read_gram_available": False,
        "v68_read_gram_tap": str(tap),
        "v68_read_gram_layers_requested": list(int(x) for x in layers),
    }
    if tensor is None:
        debug["v68_read_gram_reason"] = "missing_tap_tensor"
        return torch.zeros_like(default), debug
    raw = tensor.detach().cpu().float()
    if raw.ndim != 5:
        debug["v68_read_gram_reason"] = f"bad_tensor_ndim:{raw.ndim}"
        return torch.zeros_like(default), debug
    T, L, H, W, D = [int(x) for x in raw.shape]
    if T < 2 or L <= 0 or H <= 0 or W <= 0 or D <= 0:
        debug["v68_read_gram_reason"] = "insufficient_shape"
        return torch.zeros_like(default), debug
    selected = [li for li in layers if 0 <= int(li) < L]
    if not selected:
        debug["v68_read_gram_reason"] = "no_valid_selected_layers"
        return torch.zeros_like(default), debug

    layer_scores: List[torch.Tensor] = []
    for layer in selected:
        feat = F.normalize(raw[:, int(layer)].reshape(T, H * W, D), dim=-1, eps=1e-6)
        mean: Optional[torch.Tensor] = None
        m2: Optional[torch.Tensor] = None
        count = 0
        for t in range(T):
            gram = feat[t] @ feat[t].T
            count += 1
            if mean is None:
                mean = gram.clone()
                m2 = torch.zeros_like(gram)
                continue
            delta = gram - mean
            mean = mean + delta / float(count)
            assert m2 is not None
            m2 = m2 + delta * (gram - mean)
        assert m2 is not None
        row_var = (m2 / float(max(count - 1, 1))).mean(dim=1).reshape(T * 0 + H * W)
        layer_scores.append(row_var)
    raw_score = torch.stack(layer_scores, dim=0).mean(dim=0)
    repeated = raw_score.reshape(1, H * W).repeat(T, 1).reshape(-1)
    motion = _robust_quantile01(repeated, num_frames=max(int(num_frames), 1))
    motion = _flatten_or_default(motion, default).clamp(0.0, 1.0)
    debug.update({
        "v68_read_gram_available": True,
        "v68_read_gram_reason": "ok",
        "v68_read_gram_layers": list(int(x) for x in selected),
        "v68_read_gram_num_frames": int(T),
        "v68_read_gram_patch_grid": [int(H), int(W)],
        "v68_read_gram_feature_dim": int(D),
        "v68_read_gram_mean": float(motion.mean().item()) if motion.numel() else 0.0,
        "v68_read_gram_q90": float(torch.quantile(motion, 0.90).item()) if motion.numel() else 0.0,
        "v68_read_gram_gt050_mass": float((motion > 0.50).float().mean().item()) if motion.numel() else 0.0,
    })
    return motion, debug


def _v68_read_cue_patch(
    *,
    token_type: torch.Tensor,
    prior_output: Optional[PriorOutput],
    cfg: Dict[str, Any],
    q_layers: Optional[torch.Tensor],
    k_layers: Optional[torch.Tensor],
    v_layers: Optional[torch.Tensor],
    default: torch.Tensor,
    num_frames: int,
    chunk_idx: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    fusion = str(cfg.get("fusion", "motion_semrisk"))
    control = str(cfg.get("control", "none"))
    layers = [int(x) for x in cfg.get("layers", [5, 7])]
    tap = str(cfg.get("tap", "global_k_raw_patchvec_layers"))
    motion, gram_debug = _v68_grammotion_patch(
        q_layers,
        k_layers,
        v_layers,
        default,
        tap=tap,
        layers=layers,
        num_frames=num_frames,
    )
    support, risk, lowobs, sem_debug = _v68_patch_semantic_support_risk(
        token_type=token_type,
        prior_output=prior_output,
        control=control,
        chunk_idx=chunk_idx,
    )
    debug: Dict[str, Any] = {
        "v68_read_available": False,
        "v68_read_cue_source": f"v68.read.{cfg.get('body', '')}",
        "v68_read_path": "read",
        "v68_read_fusion": fusion,
        "v68_read_control": control,
        "v68_read_tap": tap,
        "v68_read_layers": layers,
    }
    debug.update(gram_debug)
    debug.update(sem_debug)
    if fusion == "geometry_only" or control == "geometry_only":
        out = motion
        semantic_required = False
    elif fusion == "semantic_only" or control == "semantic_only":
        out = risk
        semantic_required = True
    elif fusion == "support_gate":
        out = _robust_quantile01((motion * (1.0 - support)).clamp(0.0, 1.0), num_frames=max(int(num_frames), 1))
        semantic_required = True
    else:
        out = _robust_quantile01((motion * risk).clamp(0.0, 1.0), num_frames=max(int(num_frames), 1))
        semantic_required = True
    out_before_control = out.clone()
    if semantic_required and not bool(sem_debug.get("v68_read_semantic_available", False)):
        debug["v68_read_reason"] = "semantic_required_but_unavailable"
        return torch.zeros_like(default), debug
    if not bool(gram_debug.get("v68_read_gram_available", False)) and fusion not in {"semantic_only"} and control != "semantic_only":
        debug["v68_read_reason"] = "grammotion_unavailable"
        return torch.zeros_like(default), debug
    out, control_debug = _apply_v68_read_output_control(
        out,
        control=control,
        token_type=token_type,
        prior_output=prior_output,
        motion=motion,
        num_frames=num_frames,
        chunk_idx=chunk_idx,
    )
    debug.update(control_debug)

    out = torch.nan_to_num(out.detach().cpu().float().reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    out = _flatten_or_default(out, default).clamp(0.0, 1.0)
    debug.update({
        "v68_read_available": True,
        "v68_read_reason": "ok",
        "v68_read_output_mean_before_control": float(out_before_control.mean().item()) if out_before_control.numel() else 0.0,
        "v68_read_output_q90_before_control": float(torch.quantile(out_before_control, 0.90).item()) if out_before_control.numel() else 0.0,
        "v68_read_output_gt050_mass_before_control": float((out_before_control > 0.50).float().mean().item()) if out_before_control.numel() else 0.0,
        "v68_read_output_mean": float(out.mean().item()) if out.numel() else 0.0,
        "v68_read_output_q90": float(torch.quantile(out, 0.90).item()) if out.numel() else 0.0,
        "v68_read_output_gt050_mass": float((out > 0.50).float().mean().item()) if out.numel() else 0.0,
        "v68_read_corr_motion_risk": _pearson_corr(motion, risk),
        "v68_read_corr_output_motion": _pearson_corr(out, motion),
        "v68_read_corr_output_sem_support": _pearson_corr(out, support),
        "v68_read_corr_output_sem_risk": _pearson_corr(out, risk),
        "v68_read_corr_output_sem_lowobs": _pearson_corr(out, lowobs),
    })
    return out, debug


def _v78_l07_l13_read_cue_patch(
    *,
    token_type: torch.Tensor,
    prior_output: Optional[PriorOutput],
    cfg: Dict[str, Any],
    q_layers: Optional[torch.Tensor],
    k_layers: Optional[torch.Tensor],
    v_layers: Optional[torch.Tensor],
    default: torch.Tensor,
    num_frames: int,
    chunk_idx: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    mode = str(cfg.get("mode", "l07_mask_l13_neg"))
    control = str(cfg.get("control", "none"))
    l07_layer = int(cfg.get("l07_layer", 7))
    l13_layer = int(cfg.get("l13_layer", 13))
    debug: Dict[str, Any] = {
        "v78_l07_l13_available": False,
        "v78_l07_l13_body": str(cfg.get("body", "")),
        "v78_l07_l13_mode": mode,
        "v78_l07_l13_control": control,
        "v78_l07_l13_l07_layer": int(l07_layer),
        "v78_l07_l13_l13_layer": int(l13_layer),
        "v78_l07_l13_l07_tap": "global_k_raw_patchvec_layers",
        "v78_l07_l13_l13_tap": "global_v_raw_patchvec_layers",
    }
    l07_motion, l07_debug = _v68_grammotion_patch(
        q_layers,
        k_layers,
        v_layers,
        default,
        tap="global_k_raw_patchvec_layers",
        layers=[l07_layer],
        num_frames=num_frames,
    )
    l13_motion, l13_debug = _v68_grammotion_patch(
        q_layers,
        k_layers,
        v_layers,
        default,
        tap="global_v_raw_patchvec_layers",
        layers=[l13_layer],
        num_frames=num_frames,
    )
    debug.update({f"v78_l07_{k}": v for k, v in l07_debug.items()})
    debug.update({f"v78_l13_{k}": v for k, v in l13_debug.items()})
    support, risk, lowobs, sem_debug = _v68_patch_semantic_support_risk(
        token_type=token_type,
        prior_output=prior_output,
        control=control,
        chunk_idx=chunk_idx,
    )
    debug.update({f"v78_l07_l13_{k}": v for k, v in sem_debug.items()})
    if not bool(l07_debug.get("v68_read_gram_available", False)):
        debug["v78_l07_l13_reason"] = "l07_layout_motion_unavailable"
        return torch.zeros_like(default), debug
    if not bool(l13_debug.get("v68_read_gram_available", False)):
        debug["v78_l07_l13_reason"] = "l13_value_motion_unavailable"
        return torch.zeros_like(default), debug
    if not bool(sem_debug.get("v68_read_semantic_available", False)):
        debug["v78_l07_l13_reason"] = "semantic_required_but_unavailable"
        return torch.zeros_like(default), debug

    l07_motion = _flatten_or_default(l07_motion, default).clamp(0.0, 1.0)
    l13_motion = _flatten_or_default(l13_motion, default).clamp(0.0, 1.0)
    support = _flatten_or_default(support, default).clamp(0.0, 1.0)
    risk = _flatten_or_default(risk, default).clamp(0.0, 1.0)
    lowobs = _flatten_or_default(lowobs, default).clamp(0.0, 1.0)

    l13_neg = _robust_quantile01((l13_motion * risk).clamp(0.0, 1.0), num_frames=max(int(num_frames), 1))
    l07_layout = _robust_quantile01(
        (l07_motion * (0.30 + 0.50 * support + 0.20 * (1.0 - lowobs))).clamp(0.0, 1.0),
        num_frames=max(int(num_frames), 1),
    )
    l07_nonlayout = (1.0 - l07_layout).clamp(0.0, 1.0)
    if mode == "l13_only":
        out = l13_neg
    elif mode == "l07_action_only":
        out = _robust_quantile01((l07_motion * risk).clamp(0.0, 1.0), num_frames=max(int(num_frames), 1))
    elif mode == "l07_mask_l13_stable":
        out = (l13_neg * l07_nonlayout).clamp(0.0, 1.0)
    elif mode == "l07_mask_l13_neg_plus_stable":
        suppress_stable = (1.0 - 0.50 * support * l07_layout).clamp(0.0, 1.0)
        out = (l13_neg * (0.50 + 0.50 * l07_layout) * suppress_stable).clamp(0.0, 1.0)
    elif mode == "l07_mask_l13_contrast":
        contrast = (l13_neg - l07_layout).abs().clamp(0.0, 1.0)
        out = _robust_quantile01((l13_neg * contrast).clamp(0.0, 1.0), num_frames=max(int(num_frames), 1))
    else:
        out = (l13_neg * l07_layout).clamp(0.0, 1.0)
    anchor_comp_requested = "anchor_comp" in control.lower()
    anchor_comp_applied = False
    anchor_comp_strength = 0.0
    anchor_comp_gate = (support * l07_layout).clamp(0.0, 1.0)
    anchor_comp_before = out.clone()
    out_before_control = out.clone()
    out, control_debug = _apply_v68_read_output_control(
        out,
        control=control,
        token_type=token_type,
        prior_output=prior_output,
        motion=l13_motion,
        num_frames=num_frames,
        chunk_idx=chunk_idx,
    )
    debug.update({f"v78_l07_l13_{k}": v for k, v in control_debug.items()})
    out = torch.nan_to_num(out.detach().cpu().float().reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
    out = _flatten_or_default(out, default).clamp(0.0, 1.0)
    debug.update({
        "v78_l07_l13_available": True,
        "v78_l07_l13_reason": "ok",
        "v78_l07_l13_l07_layout_mean": float(l07_layout.mean().item()) if l07_layout.numel() else 0.0,
        "v78_l07_l13_l07_layout_q90": float(torch.quantile(l07_layout, 0.90).item()) if l07_layout.numel() else 0.0,
        "v78_l07_l13_l07_layout_gt050_mass": float((l07_layout > 0.50).float().mean().item()) if l07_layout.numel() else 0.0,
        "v78_l07_l13_l13_neg_mean": float(l13_neg.mean().item()) if l13_neg.numel() else 0.0,
        "v78_l07_l13_l13_neg_q90": float(torch.quantile(l13_neg, 0.90).item()) if l13_neg.numel() else 0.0,
        "v78_l07_l13_anchor_comp_requested": bool(anchor_comp_requested),
        "v78_l07_l13_anchor_comp_applied": bool(anchor_comp_applied),
        "v78_l07_l13_anchor_comp_strength": float(anchor_comp_strength) if anchor_comp_applied else 0.0,
        "v78_l07_l13_anchor_comp_gate_mean": float(anchor_comp_gate.mean().item()) if anchor_comp_gate.numel() else 0.0,
        "v78_l07_l13_anchor_comp_gate_q90": float(torch.quantile(anchor_comp_gate, 0.90).item()) if anchor_comp_gate.numel() else 0.0,
        "v78_l07_l13_anchor_comp_output_mean_before": float(anchor_comp_before.mean().item()) if anchor_comp_before.numel() else 0.0,
        "v78_l07_l13_anchor_comp_output_mean_after": float(out_before_control.mean().item()) if out_before_control.numel() else 0.0,
        "v78_l07_l13_anchor_comp_removed_mean": (
            float((anchor_comp_before - out_before_control).mean().item())
            if anchor_comp_before.numel() and out_before_control.numel()
            else 0.0
        ),
        "v78_l07_l13_output_mean_before_control": float(out_before_control.mean().item()) if out_before_control.numel() else 0.0,
        "v78_l07_l13_output_q90_before_control": float(torch.quantile(out_before_control, 0.90).item()) if out_before_control.numel() else 0.0,
        "v78_l07_l13_output_gt050_mass_before_control": float((out_before_control > 0.50).float().mean().item()) if out_before_control.numel() else 0.0,
        "v78_l07_l13_output_mean": float(out.mean().item()) if out.numel() else 0.0,
        "v78_l07_l13_output_q90": float(torch.quantile(out, 0.90).item()) if out.numel() else 0.0,
        "v78_l07_l13_output_gt050_mass": float((out > 0.50).float().mean().item()) if out.numel() else 0.0,
        "v78_l07_l13_corr_output_l07_layout": _pearson_corr(out, l07_layout),
        "v78_l07_l13_corr_output_l13_neg": _pearson_corr(out, l13_neg),
        "v78_l07_l13_corr_l07_l13": _pearson_corr(l07_layout, l13_neg),
    })
    return out, debug


def _strip_v67_random_suffix(body: str) -> Tuple[str, bool]:
    random_same_distribution = False
    for suffix in ("_random_same_mass", ".random_same_mass", "_random_same_distribution", ".random_same_distribution"):
        if body.endswith(suffix):
            random_same_distribution = True
            body = body[: -len(suffix)]
            break
    return body, random_same_distribution


def _parse_v67_semgeo_read_cue(read_cue_source: str) -> Optional[Dict[str, Any]]:
    src = str(read_cue_source or "").strip().lower()
    if src.startswith("v67.geocue_"):
        body, random_same_distribution = _strip_v67_random_suffix(src[len("v67.geocue_"):])
        if not body:
            return None
        return {
            "mode": "geo_only",
            "geo_kind": body,
            "sem_kind": None,
            "fusion": "geo",
            "random_same_distribution": bool(random_same_distribution),
        }
    if not src.startswith("v67.semgeo_"):
        return None
    body, random_same_distribution = _strip_v67_random_suffix(src[len("v67.semgeo_"):])
    parts = [p for p in body.split("_") if p]
    if len(parts) < 2:
        return None
    sem_aliases = {
        "support": "anchor_support",
        "anchor_support": "anchor_support",
        "stable": "anchor_support",
        "risk": "anchor_risk",
        "anchor_risk": "anchor_risk",
    }
    sem_kind = sem_aliases.get(parts[0])
    if sem_kind is None:
        return None
    fusion = "gate"
    if parts[-1] in {"mul", "gate", "blend", "min", "geo"}:
        fusion = parts[-1]
        geo_kind = "_".join(parts[1:-1])
    else:
        geo_kind = "_".join(parts[1:])
    if not geo_kind:
        return None
    return {
        "mode": "sem_geo",
        "geo_kind": geo_kind,
        "sem_kind": sem_kind,
        "fusion": fusion,
        "random_same_distribution": bool(random_same_distribution),
    }


def _v67_geo_candidate_patch(
    geo_kind: str,
    *,
    dyn_patch: torch.Tensor,
    occ_patch: torch.Tensor,
    unc_patch: torch.Tensor,
    conf_patch: torch.Tensor,
    anchor_patch: torch.Tensor,
    stage_geo_patch: torch.Tensor,
    qkstable_patch: torch.Tensor,
    deepstatic_patch: torch.Tensor,
    smd_patch: torch.Tensor,
    fa_decay_patch: torch.Tensor,
    key_avg_patch: torch.Tensor,
    chunk_idx: int,
    num_frames: int = 1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    kind = str(geo_kind or "").strip().lower()
    n_patch = int(dyn_patch.numel())
    zeros = torch.zeros((n_patch,), dtype=torch.float32)
    dyn = dyn_patch.detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
    occ = _flatten_or_default(occ_patch, dyn).clamp(0.0, 1.0)
    unc = _flatten_or_default(unc_patch, dyn).clamp(0.0, 1.0)
    conf = _flatten_or_default(conf_patch, dyn).clamp(0.0, 1.0)
    anchor = _flatten_or_default(anchor_patch, dyn).clamp(0.0, 1.0)
    stage_geo = _flatten_or_default(stage_geo_patch, dyn).clamp(0.0, 1.0)
    qkstable = _flatten_or_default(qkstable_patch, dyn).clamp(0.0, 1.0)
    deepstatic = _flatten_or_default(deepstatic_patch, dyn).clamp(0.0, 1.0)
    smd = _flatten_or_default(smd_patch, dyn).clamp(0.0, 1.0)
    fa_decay = _flatten_or_default(fa_decay_patch, dyn).clamp(0.0, 1.0)
    key_avg = _flatten_or_default(key_avg_patch, dyn).clamp(0.0, 1.0)
    reliable_static = torch.sqrt(((1.0 - dyn) * (1.0 - occ) * (1.0 - unc) * conf).clamp(0.0, 1.0))
    loose_static = torch.sqrt(((1.0 - dyn) * (1.0 - occ) * (1.0 - 0.5 * unc).clamp(0.0, 1.0) * conf).clamp(0.0, 1.0))
    aliases = {
        "conf": "confstatic",
        "static": "confstatic",
        "conf_static": "confstatic",
        "conf_loose": "confloose",
        "loose": "confloose",
        "conf_q": "confq",
        "conf_quantile": "confq",
        "reliable": "confstatic",
        "reliable_static": "confstatic",
        "stageb": "stagebgeo",
        "stageb_geo": "stagebgeo",
        "gwrite": "stagebgeo",
        "qk": "qkstable",
        "qk_stable": "qkstable",
        "deep": "deepstatic",
        "deep_static": "deepstatic",
        "attn": "attndecay",
        "attn_decay": "attndecay",
        "keydecay": "attndecay",
        "loger": "logermix",
        "mix": "logermix",
    }
    kind = aliases.get(kind, kind)
    if kind == "confstatic":
        out = reliable_static
    elif kind == "confloose":
        out = loose_static
    elif kind == "confq":
        out = _robust_quantile01(reliable_static, num_frames=max(int(num_frames), 1))
    elif kind == "stagebgeo":
        out = stage_geo
    elif kind == "qkstable":
        out = qkstable
    elif kind == "deepstatic":
        out = deepstatic
    elif kind == "smd":
        out = smd
    elif kind == "attndecay":
        out = fa_decay
    elif kind == "keyavg":
        out = key_avg
    elif kind == "anchorconf":
        out = torch.sqrt((anchor * conf * (1.0 - unc).clamp(0.0, 1.0)).clamp(0.0, 1.0))
    elif kind == "logermix":
        out = (
            0.30 * loose_static
            + 0.25 * qkstable
            + 0.20 * deepstatic
            + 0.10 * stage_geo
            + 0.15 * key_avg
        ).clamp(0.0, 1.0)
    elif kind == "strictmix":
        out = torch.sqrt((reliable_static * qkstable * (0.25 + 0.75 * deepstatic)).clamp(0.0, 1.0))
    else:
        debug = {
            "v67_semgeo_available": False,
            "v67_semgeo_reason": f"unknown_geo_kind:{geo_kind}",
            "v67_semgeo_geo_kind": str(geo_kind),
            "v67_semgeo_chunk_idx": int(chunk_idx),
        }
        return zeros, debug

    out = torch.nan_to_num(out.float().reshape(-1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    debug = {
        "v67_semgeo_geo_kind": kind,
        "v67_semgeo_chunk_idx": int(chunk_idx),
        "v67_semgeo_geo_mean": float(out.mean().item()) if out.numel() else 0.0,
        "v67_semgeo_geo_q90": float(torch.quantile(out, 0.90).item()) if out.numel() else 0.0,
        "v67_semgeo_geo_gt050_mass": float((out > 0.50).float().mean().item()) if out.numel() else 0.0,
        "v67_semgeo_geo_gt075_mass": float((out > 0.75).float().mean().item()) if out.numel() else 0.0,
        "v67_semgeo_geo_corr_dyn": _pearson_corr(out, dyn),
        "v67_semgeo_geo_corr_unc": _pearson_corr(out, unc),
        "v67_semgeo_geo_corr_occ": _pearson_corr(out, occ),
        "v67_semgeo_geo_corr_conf": _pearson_corr(out, conf),
        "v67_semgeo_geo_corr_anchor": _pearson_corr(out, anchor),
    }
    return out, debug


def _v67_semgeo_cue_patch(
    *,
    token_type: torch.Tensor,
    prior_output: Optional[PriorOutput],
    cfg: Dict[str, Any],
    dyn_patch: torch.Tensor,
    occ_patch: torch.Tensor,
    unc_patch: torch.Tensor,
    conf_patch: torch.Tensor,
    anchor_patch: torch.Tensor,
    stage_geo_patch: torch.Tensor,
    qkstable_patch: torch.Tensor,
    deepstatic_patch: torch.Tensor,
    smd_patch: torch.Tensor,
    fa_decay_patch: torch.Tensor,
    key_avg_patch: torch.Tensor,
    chunk_idx: int,
    num_frames: int = 1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    geo_kind = str(cfg.get("geo_kind", "confstatic"))
    fusion = str(cfg.get("fusion", "gate")).strip().lower()
    sem_kind = cfg.get("sem_kind")
    random_same_distribution = bool(cfg.get("random_same_distribution", False))
    geo_score, geo_debug = _v67_geo_candidate_patch(
        geo_kind,
        dyn_patch=dyn_patch,
        occ_patch=occ_patch,
        unc_patch=unc_patch,
        conf_patch=conf_patch,
        anchor_patch=anchor_patch,
        stage_geo_patch=stage_geo_patch,
        qkstable_patch=qkstable_patch,
        deepstatic_patch=deepstatic_patch,
        smd_patch=smd_patch,
        fa_decay_patch=fa_decay_patch,
        key_avg_patch=key_avg_patch,
        num_frames=max(int(num_frames), 1),
        chunk_idx=chunk_idx,
    )
    debug: Dict[str, Any] = {
        "v67_semgeo_available": bool(geo_score.numel() > 0),
        "v67_semgeo_mode": str(cfg.get("mode", "sem_geo")),
        "v67_semgeo_geo_kind": str(geo_debug.get("v67_semgeo_geo_kind", geo_kind)),
        "v67_semgeo_sem_kind": sem_kind,
        "v67_semgeo_fusion": fusion,
        "v67_semgeo_random_same_distribution": random_same_distribution,
        "v67_semgeo_chunk_idx": int(chunk_idx),
        "v67_semgeo_patch_tokens": int(geo_score.numel()),
    }
    debug.update(geo_debug)
    if not bool(geo_debug.get("v67_semgeo_available", True)):
        debug["v67_semgeo_available"] = False
        debug["v67_semgeo_reason"] = geo_debug.get("v67_semgeo_reason", "geo_unavailable")
        return geo_score, debug

    sem_score = None
    sem_debug: Dict[str, Any] = {}
    if sem_kind is not None:
        sem_score, sem_debug = _v67_semantic_anchor_cue_patch(
            token_type=token_type,
            prior_output=prior_output,
            kind=str(sem_kind),
            random_same_distribution=False,
            chunk_idx=chunk_idx,
        )
        debug.update({
            "v67_semgeo_sem_available": bool(sem_debug.get("v67_semcue_available", False)),
            "v67_semgeo_sem_support_mean": sem_debug.get("v67_semcue_support_mean"),
            "v67_semgeo_sem_support_q90": sem_debug.get("v67_semcue_support_q90"),
            "v67_semgeo_sem_risk_mean": sem_debug.get("v67_semcue_risk_mean"),
            "v67_semgeo_sem_risk_q90": sem_debug.get("v67_semcue_risk_q90"),
            "v67_semgeo_sem_reason": sem_debug.get("v67_semcue_reason"),
        })
        if int(sem_score.numel()) != int(geo_score.numel()):
            debug["v67_semgeo_available"] = False
            debug["v67_semgeo_reason"] = "sem_geo_size_mismatch"
            return torch.zeros_like(geo_score), debug

    if sem_score is None:
        out = geo_score
    elif fusion == "mul":
        out = (sem_score * geo_score).clamp(0.0, 1.0)
    elif fusion == "blend":
        out = (0.5 * sem_score + 0.5 * geo_score).clamp(0.0, 1.0)
    elif fusion == "min":
        out = torch.minimum(sem_score, geo_score).clamp(0.0, 1.0)
    elif fusion == "geo":
        out = geo_score
    else:
        out = (geo_score * (0.25 + 0.75 * sem_score)).clamp(0.0, 1.0)
        fusion = "gate"

    before_random = out.clone()
    if random_same_distribution and int(out.numel()) > 1:
        noise = _deterministic_patch_noise(int(out.numel()), seed=6713009 + max(int(chunk_idx), 0) * 101.0)
        out = out[torch.argsort(noise, stable=True)]

    debug.update({
        "v67_semgeo_available": True,
        "v67_semgeo_reason": "ok",
        "v67_semgeo_fusion": fusion,
        "v67_semgeo_output_mean_before_random": float(before_random.mean().item()) if before_random.numel() else 0.0,
        "v67_semgeo_output_q90_before_random": float(torch.quantile(before_random, 0.90).item()) if before_random.numel() else 0.0,
        "v67_semgeo_output_mean": float(out.mean().item()) if out.numel() else 0.0,
        "v67_semgeo_output_q90": float(torch.quantile(out, 0.90).item()) if out.numel() else 0.0,
        "v67_semgeo_output_gt050_mass": float((out > 0.50).float().mean().item()) if out.numel() else 0.0,
        "v67_semgeo_output_gt075_mass": float((out > 0.75).float().mean().item()) if out.numel() else 0.0,
        "v67_semgeo_output_corr_geo": _pearson_corr(out, geo_score),
        "v67_semgeo_output_corr_dyn": _pearson_corr(out, dyn_patch),
        "v67_semgeo_output_corr_conf": _pearson_corr(out, conf_patch),
        "v67_semgeo_output_corr_unc": _pearson_corr(out, unc_patch),
    })
    if sem_score is not None:
        debug["v67_semgeo_output_corr_sem"] = _pearson_corr(out, sem_score)
    return out.clamp(0.0, 1.0), debug


def _v67_semantic_anchor_cue_patch(
    *,
    token_type: torch.Tensor,
    prior_output: Optional[PriorOutput],
    kind: str,
    random_same_distribution: bool,
    chunk_idx: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    token_type_cpu = token_type.detach().cpu().long().reshape(-1)
    n_patch = int((token_type_cpu == TOKEN_TYPE_PATCH).sum().item())
    out = torch.zeros((n_patch,), dtype=torch.float32)
    debug: Dict[str, Any] = {
        "v67_semcue_available": False,
        "v67_semcue_kind": str(kind),
        "v67_semcue_random_same_distribution": bool(random_same_distribution),
        "v67_semcue_chunk_idx": int(chunk_idx),
        "v67_semcue_patch_tokens": int(n_patch),
    }
    if prior_output is None or n_patch <= 0:
        debug["v67_semcue_reason"] = "missing_prior_or_empty_patch"
        return out, debug

    G_patch = _patch_labels_from_token_labels(getattr(prior_output, "G_sem_tok", None), token_type_cpu)
    L_patch = _patch_labels_from_token_labels(getattr(prior_output, "L_sem_tok", None), token_type_cpu)
    Q_patch = None
    Q_sem_tok = getattr(prior_output, "Q_sem_tok", None)
    if Q_sem_tok is not None:
        q = Q_sem_tok.detach().cpu().float().reshape(-1)
        if int(q.numel()) == int(token_type_cpu.numel()):
            Q_patch = q[token_type_cpu == TOKEN_TYPE_PATCH].clamp(0.0, 1.0)
    trust = Q_patch if Q_patch is not None and int(Q_patch.numel()) == n_patch else torch.ones((n_patch,), dtype=torch.float32)

    support = torch.zeros((n_patch,), dtype=torch.float32)
    risk = torch.full((n_patch,), 0.75, dtype=torch.float32)
    known = torch.zeros((n_patch,), dtype=torch.bool)

    if L_patch is not None and int(L_patch.numel()) == n_patch:
        L = L_patch.detach().cpu().long().reshape(-1)
        known = (L >= 0) & (L != int(SEMANTIC_FINE_LABEL_UNKNOWN))
        vertical = torch.zeros_like(known)
        ground = torch.zeros_like(known)
        sky = torch.zeros_like(known)
        vegetation = torch.zeros_like(known)
        movable = torch.zeros_like(known)
        for label_id in _V67_SEMCUE_VERTICAL_STATIC_FINE_IDS:
            vertical |= L == int(label_id)
        for label_id in _V67_SEMCUE_GROUND_STATIC_FINE_IDS:
            ground |= L == int(label_id)
        for label_id in SEMANTIC_FINE_SKY_IDS:
            sky |= L == int(label_id)
        for label_id in SEMANTIC_FINE_VEGETATION_IDS:
            vegetation |= L == int(label_id)
        for label_id in SEMANTIC_FINE_MOVABLE_IDS:
            movable |= L == int(label_id)
        support[vertical & known] = 1.00
        support[ground & known] = torch.maximum(support[ground & known], torch.full_like(support[ground & known], 0.55))
        support[(sky | vegetation) & known] = torch.maximum(
            support[(sky | vegetation) & known],
            torch.full_like(support[(sky | vegetation) & known], 0.10),
        )
        support[movable & known] = 0.0
        risk[vertical & known] = 0.05
        risk[ground & known] = 0.35
        risk[sky & known] = 1.00
        risk[vegetation & known] = torch.maximum(risk[vegetation & known], torch.full_like(risk[vegetation & known], 0.90))
        risk[movable & known] = torch.maximum(risk[movable & known], torch.full_like(risk[movable & known], 0.85))
        debug.update({
            "v67_semcue_vertical_static_tokens": int((vertical & known).sum().item()),
            "v67_semcue_ground_static_tokens": int((ground & known).sum().item()),
            "v67_semcue_sky_tokens": int((sky & known).sum().item()),
            "v67_semcue_vegetation_tokens": int((vegetation & known).sum().item()),
            "v67_semcue_movable_tokens": int((movable & known).sum().item()),
            "v67_semcue_known_fine_tokens": int(known.sum().item()),
        })

    if G_patch is not None and int(G_patch.numel()) == n_patch:
        G = G_patch.detach().cpu().long().reshape(-1)
        coarse_support = torch.zeros_like(support)
        coarse_risk = torch.full_like(risk, 0.75)
        coarse_support[G == int(SEMANTIC_GROUP_STRUCTURE_ANCHOR)] = 0.75
        coarse_support[G == int(SEMANTIC_GROUP_STATIC_THING)] = 0.65
        coarse_support[G == int(SEMANTIC_GROUP_LOW_VALUE_STUFF)] = 0.10
        coarse_support[G == int(SEMANTIC_GROUP_MOVABLE_THING)] = 0.0
        coarse_support[G == int(SEMANTIC_GROUP_UNCERTAIN_REGION)] = 0.0
        coarse_risk[G == int(SEMANTIC_GROUP_STRUCTURE_ANCHOR)] = 0.25
        coarse_risk[G == int(SEMANTIC_GROUP_STATIC_THING)] = 0.35
        coarse_risk[G == int(SEMANTIC_GROUP_LOW_VALUE_STUFF)] = 0.90
        coarse_risk[G == int(SEMANTIC_GROUP_MOVABLE_THING)] = 0.85
        coarse_risk[G == int(SEMANTIC_GROUP_UNCERTAIN_REGION)] = 0.75
        support = torch.maximum(support, coarse_support)
        risk = torch.maximum(risk, coarse_risk)
        debug.update({
            "v67_semcue_group_structure_tokens": int((G == int(SEMANTIC_GROUP_STRUCTURE_ANCHOR)).sum().item()),
            "v67_semcue_group_static_tokens": int((G == int(SEMANTIC_GROUP_STATIC_THING)).sum().item()),
            "v67_semcue_group_lowstuff_tokens": int((G == int(SEMANTIC_GROUP_LOW_VALUE_STUFF)).sum().item()),
            "v67_semcue_group_movable_tokens": int((G == int(SEMANTIC_GROUP_MOVABLE_THING)).sum().item()),
            "v67_semcue_group_uncertain_tokens": int((G == int(SEMANTIC_GROUP_UNCERTAIN_REGION)).sum().item()),
        })

    support = (support * trust).clamp(0.0, 1.0)
    risk = torch.maximum(risk, (1.0 - trust).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    if kind == "anchor_support":
        out = support
    elif kind == "anchor_risk":
        out = risk
    else:
        debug["v67_semcue_reason"] = f"unknown_kind:{kind}"
        return torch.zeros((n_patch,), dtype=torch.float32), debug

    before_random = out.clone()
    if random_same_distribution and int(out.numel()) > 1:
        noise = _deterministic_patch_noise(int(out.numel()), seed=6705007 + max(int(chunk_idx), 0) * 101.0)
        out = out[torch.argsort(noise, stable=True)]

    debug.update({
        "v67_semcue_available": True,
        "v67_semcue_reason": "ok",
        "v67_semcue_trust_missing": Q_patch is None,
        "v67_semcue_trust_mean": float(trust.mean().item()) if trust.numel() else 0.0,
        "v67_semcue_support_mean": float(support.mean().item()) if support.numel() else 0.0,
        "v67_semcue_support_q90": float(torch.quantile(support, 0.90).item()) if support.numel() else 0.0,
        "v67_semcue_risk_mean": float(risk.mean().item()) if risk.numel() else 0.0,
        "v67_semcue_risk_q90": float(torch.quantile(risk, 0.90).item()) if risk.numel() else 0.0,
        "v67_semcue_output_mean_before_random": float(before_random.mean().item()) if before_random.numel() else 0.0,
        "v67_semcue_output_q90_before_random": float(torch.quantile(before_random, 0.90).item()) if before_random.numel() else 0.0,
        "v67_semcue_output_mean": float(out.mean().item()) if out.numel() else 0.0,
        "v67_semcue_output_q90": float(torch.quantile(out, 0.90).item()) if out.numel() else 0.0,
        "v67_semcue_output_gt050_mass": float((out > 0.50).float().mean().item()) if out.numel() else 0.0,
        "v67_semcue_output_gt075_mass": float((out > 0.75).float().mean().item()) if out.numel() else 0.0,
    })
    return out.clamp(0.0, 1.0), debug


def _robust_quantile01(
    values: torch.Tensor,
    *,
    num_frames: int,
    qlo: float = 0.50,
    qhi: float = 0.95,
) -> torch.Tensor:
    vals = torch.nan_to_num(values.detach().cpu().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if vals.numel() == 0:
        return vals
    qlo = min(max(float(qlo), 0.0), 1.0)
    qhi = min(max(float(qhi), qlo + 1e-4), 1.0)
    if num_frames <= 0 or vals.numel() % num_frames != 0:
        lo = torch.quantile(vals, qlo)
        hi = torch.quantile(vals, qhi)
        return ((vals - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    per_frame = vals.reshape(num_frames, -1)
    lo = torch.quantile(per_frame, qlo, dim=1, keepdim=True)
    hi = torch.quantile(per_frame, qhi, dim=1, keepdim=True)
    return ((per_frame - lo) / (hi - lo).clamp_min(1e-6)).reshape(-1).clamp(0.0, 1.0)


def _centered_percentile_rank(values: torch.Tensor) -> torch.Tensor:
    vals = torch.nan_to_num(values.detach().cpu().float(), nan=0.0, posinf=1.0, neginf=0.0)
    n = int(vals.numel())
    if n <= 1:
        return torch.zeros_like(vals)
    if float((vals.max() - vals.min()).abs().item()) < 1e-8:
        return torch.zeros_like(vals)
    order = torch.argsort(vals, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(n, dtype=torch.float32)
    return 2.0 * (ranks / float(n - 1)) - 1.0


def _frame_patch_mask(num_patch: int, num_frames: int, *, first: int = 0, last: int = 0) -> torch.Tensor:
    if num_patch <= 0 or num_frames <= 0 or num_patch % num_frames != 0:
        return torch.zeros(num_patch, dtype=torch.bool)
    patches_per_frame = num_patch // num_frames
    frame_id = (torch.arange(num_patch, dtype=torch.long) // patches_per_frame).clamp_max(num_frames - 1)
    mask = torch.zeros(num_patch, dtype=torch.bool)
    if first > 0:
        mask |= frame_id < int(first)
    if last > 0:
        mask |= frame_id >= max(num_frames - int(last), 0)
    return mask


def _high_quantile_mask(values: torch.Tensor, q: float) -> torch.Tensor:
    vals = values.detach().cpu().float().reshape(-1)
    if vals.numel() == 0:
        return torch.zeros_like(vals, dtype=torch.bool)
    q = min(max(float(q), 0.0), 1.0)
    thr = torch.quantile(vals, q)
    return vals >= thr


def _topk_per_frame(values: torch.Tensor, *, num_frames: int, frac: float) -> torch.Tensor:
    vals = values.detach().cpu().float().reshape(-1)
    if frac <= 0.0 or frac >= 1.0 or num_frames <= 0 or vals.numel() == 0:
        return vals
    if vals.numel() % num_frames != 0:
        k = max(1, int(round(vals.numel() * frac)))
        if k >= vals.numel():
            return vals
        idx = torch.topk(vals, k=k, largest=True).indices
        out = torch.zeros_like(vals)
        out[idx] = vals[idx]
        return out
    per_frame = vals.reshape(num_frames, -1)
    k = max(1, int(round(per_frame.shape[1] * frac)))
    if k >= per_frame.shape[1]:
        return vals
    out = torch.zeros_like(per_frame)
    idx = torch.topk(per_frame, k=k, dim=1, largest=True).indices
    out.scatter_(1, idx, per_frame.gather(1, idx))
    return out.reshape(-1)


def _adaptive_quantile_calibrate(
    values: torch.Tensor,
    *,
    num_frames: int,
    target_mass: float,
    tau: float,
) -> torch.Tensor:
    vals = values.detach().cpu().float().reshape(-1)
    q = float(target_mass)
    if q <= 0.0 or q >= 1.0 or vals.numel() == 0:
        return _normalize01(vals)
    tau = max(float(tau), 1e-4)
    if num_frames <= 0 or vals.numel() % num_frames != 0:
        threshold = torch.quantile(vals, max(0.0, min(1.0, 1.0 - q)))
        return torch.sigmoid((vals - threshold) / tau).clamp(0.0, 1.0)
    per_frame = vals.reshape(num_frames, -1)
    threshold = torch.quantile(per_frame, max(0.0, min(1.0, 1.0 - q)), dim=1, keepdim=True)
    return torch.sigmoid((per_frame - threshold) / tau).reshape(-1).clamp(0.0, 1.0)


def _connected_components_count(mask_2d: torch.Tensor) -> int:
    mask = mask_2d.detach().cpu().bool()
    if mask.ndim != 2 or not bool(mask.any()):
        return 0
    h, w = int(mask.shape[0]), int(mask.shape[1])
    seen = torch.zeros((h, w), dtype=torch.bool)
    count = 0
    for y in range(h):
        for x in range(w):
            if not bool(mask[y, x]) or bool(seen[y, x]):
                continue
            count += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and bool(mask[ny, nx]) and not bool(seen[ny, nx]):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return count


def _fragmentation(values: torch.Tensor, *, num_frames: int, patch_grid: Tuple[int, int], thr: float = 0.5) -> float:
    vals = values.detach().cpu().float().reshape(-1)
    h, w = int(patch_grid[0]), int(patch_grid[1])
    if num_frames <= 0 or h <= 0 or w <= 0 or vals.numel() < num_frames * h * w:
        return 0.0
    cube = vals[: num_frames * h * w].reshape(num_frames, h, w)
    frags: List[float] = []
    for frame in cube:
        mask = frame > float(thr)
        active = int(mask.sum().item())
        if active <= 0:
            frags.append(0.0)
            continue
        frags.append(float(_connected_components_count(mask)) / float(active))
    return float(torch.tensor(frags, dtype=torch.float32).mean().item()) if frags else 0.0


def _frame_bias_energy(
    D_tok: torch.Tensor,
    P_ref: torch.Tensor,
    *,
    num_frames: int,
    mode: str,
    K_stable_tok: Optional[torch.Tensor] = None,
) -> float:
    D = D_tok.detach().cpu().float().reshape(-1)
    ref = P_ref.detach().cpu().float().reshape(-1) if P_ref is not None else torch.zeros_like(D)
    if num_frames <= 0 or D.numel() == 0 or D.numel() % num_frames != 0:
        return 0.0
    tokens_per_frame = D.numel() // num_frames
    D = (D * (1.0 - ref.clamp(0.0, 1.0))).reshape(num_frames, tokens_per_frame)
    Dq = D[:, :, None]
    Dk = D[:, None, :]
    if mode == "key":
        keep = (1.0 - Dk).expand(-1, tokens_per_frame, -1)
    elif mode == "query":
        return 0.0
    elif mode in {
        "qk_pair_stable_harm",
        "qk_pair_random_same_mass",
        "read_qk_pair_bias",
        "qk_pair_key_stability",
        "qk_pair_key_stability_random_same_mass",
    }:
        q_risk = Dq.clamp(0.0, 1.0)
        if mode in {"qk_pair_key_stability", "qk_pair_key_stability_random_same_mass"} and K_stable_tok is not None:
            K = K_stable_tok.detach().cpu().float().reshape(-1)
            if K.numel() == D_tok.detach().cpu().reshape(-1).numel():
                K = K.reshape(num_frames, tokens_per_frame)
                if P_ref is not None:
                    K = K * (1.0 - ref.clamp(0.0, 1.0).reshape(num_frames, tokens_per_frame))
                k_stable = K.clamp(0.0, 1.0)[:, None, :]
                k_harm = (1.0 - k_stable).clamp(0.0, 1.0)
            else:
                k_harm = Dk.clamp(0.0, 1.0)
                k_stable = (1.0 - k_harm).clamp(0.0, 1.0)
        else:
            k_harm = Dk.clamp(0.0, 1.0)
            k_stable = (1.0 - k_harm).clamp(0.0, 1.0)
        pair_score = q_risk * (k_stable - k_harm)
        return float(pair_score.abs().mean().item()) if pair_score.numel() else 0.0
    else:
        keep = 1.0 - (1.0 - Dq) * Dk
    bias = torch.log(keep.clamp_min(1e-4))
    return float(bias.abs().mean().item()) if bias.numel() else 0.0


def _flatten_optional_patch_map(map_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if map_tensor is None:
        return None
    return map_tensor.detach().cpu().float().reshape(-1)


def _layer_indices(layer_ids: Optional[torch.Tensor], n_layers: int, group: str) -> torch.Tensor:
    if n_layers <= 0:
        return torch.zeros((0,), dtype=torch.long)
    idx_all = torch.arange(n_layers, dtype=torch.long)
    if layer_ids is not None and int(layer_ids.numel()) == n_layers:
        ids = layer_ids.detach().cpu().long().reshape(-1)
    else:
        ids = idx_all

    if group == "l0":
        hit = torch.nonzero(ids == 0, as_tuple=False).reshape(-1)
        return hit[:1] if hit.numel() else idx_all[:1]
    if group == "l4":
        hit = torch.nonzero(ids == 4, as_tuple=False).reshape(-1)
        return hit[:1] if hit.numel() else idx_all[min(1, n_layers - 1) : min(2, n_layers)]
    if group == "all":
        return idx_all

    third = max(n_layers // 3, 1)
    if group == "shallow":
        return idx_all[:third]
    if group == "middle":
        start = third
        end = max(2 * third, start + 1)
        return idx_all[start:min(end, n_layers)]
    if group == "deep":
        return idx_all[min(2 * third, n_layers - 1):]
    return idx_all


def _flatten_layer_group_map(
    layer_maps: Optional[torch.Tensor],
    layer_ids: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    group: str,
    stat: str = "mean",
) -> torch.Tensor:
    if layer_maps is None:
        return torch.zeros_like(default)
    maps = layer_maps.detach().cpu().float()
    if maps.ndim != 4:
        return _flatten_or_default(maps, default)
    # Expected shape: [T, L, H_tok, W_tok].
    n_layers = int(maps.shape[1])
    idx = _layer_indices(layer_ids, n_layers, group)
    if idx.numel() == 0:
        return torch.zeros_like(default)
    selected = maps.index_select(1, idx)
    if stat == "var":
        reduced = selected.var(dim=1, unbiased=False)
    else:
        reduced = selected.mean(dim=1)
    return _flatten_or_default(reduced, default)


def _valid_patch_signal(values: torch.Tensor) -> bool:
    vals = torch.nan_to_num(values.detach().cpu().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    return bool(vals.numel() > 0 and float(vals.max().item()) > 1e-6)


def _patch_signal_debug(prefix: str, values: torch.Tensor) -> Dict[str, Any]:
    vals = torch.nan_to_num(values.detach().cpu().float().reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if int(vals.numel()) <= 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_q90": None,
            f"{prefix}_max": None,
            f"{prefix}_gt050_mass": 0.0,
        }
    return {
        f"{prefix}_mean": float(vals.mean().item()),
        f"{prefix}_q90": float(torch.quantile(vals, 0.90).item()),
        f"{prefix}_max": float(vals.max().item()),
        f"{prefix}_gt050_mass": float((vals > 0.50).float().mean().item()),
    }


def _select_frame_key_stability_patch(
    geo: GeometryOutput,
    default: torch.Tensor,
    *,
    num_frames: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Select a non-GT key-stability source without silently zeroing missing avg maps."""
    zero = torch.zeros_like(default.detach().cpu().float().reshape(-1))
    candidates: List[Tuple[str, torch.Tensor]] = []

    avg_raw = _flatten_optional_patch_map(geo.frame_attn_key_cosine_avg)
    if avg_raw is not None and int(avg_raw.numel()) > 0:
        candidates.append((
            "frame_attn_key_cosine_avg",
            _normalize01(_flatten_or_default(geo.frame_attn_key_cosine_avg, default)),
        ))

    if geo.frame_attn_cosine_key_layers is not None:
        candidates.append((
            "frame_attn_cosine_key_layers_all_robustq",
            _robust_quantile01(
                _flatten_layer_group_map(
                    geo.frame_attn_cosine_key_layers,
                    geo.frame_attn_cosine_layer_ids,
                    default,
                    group="all",
                ),
                num_frames=num_frames,
            ),
        ))

    endpoint_parts: List[torch.Tensor] = []
    for field_name in ("frame_attn_key_cosine_l0", "frame_attn_key_cosine_l4"):
        vals = _flatten_optional_patch_map(getattr(geo, field_name, None))
        if vals is None or int(vals.numel()) <= 0:
            continue
        endpoint_parts.append(_flatten_or_default(getattr(geo, field_name), default))
    if endpoint_parts:
        endpoint_raw = torch.stack(endpoint_parts, dim=0).mean(dim=0)
        candidates.append((
            "frame_attn_key_cosine_l0_l4_mean_robustq",
            _robust_quantile01(endpoint_raw, num_frames=num_frames),
        ))

    selected_name = "missing_zero"
    selected = zero
    for name, patch in candidates:
        if _valid_patch_signal(patch):
            selected_name = name
            selected = patch.detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
            break

    debug: Dict[str, Any] = {
        "key_stability_source": selected_name,
        "key_stability_source_fallback_used": selected_name not in {
            "frame_attn_key_cosine_avg",
            "missing_zero",
        },
        "key_stability_source_candidate_count": int(len(candidates)),
    }
    debug.update(_patch_signal_debug("key_stability_base_patch", selected))
    return selected, debug


def _flatten_or_default(map_tensor: Optional[torch.Tensor], default: torch.Tensor) -> torch.Tensor:
    vals = _flatten_optional_patch_map(map_tensor)
    if vals is None or vals.numel() == 0:
        return torch.zeros_like(default)
    if vals.numel() == default.numel():
        return vals
    out = torch.zeros_like(default)
    n = min(int(vals.numel()), int(default.numel()))
    out[:n] = vals[:n]
    return out


def _flatten_cue_field_or_default(
    map_tensor: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    num_frames: int,
    patch_grid: Tuple[int, int],
) -> torch.Tensor:
    """Flatten a CueOutput pixel/grid field to the patch-token layout."""
    if map_tensor is None:
        return torch.zeros_like(default)
    vals = map_tensor.detach().cpu().float()
    if vals.numel() == default.numel():
        return vals.reshape(-1)
    h_tok, w_tok = int(patch_grid[0]), int(patch_grid[1])
    if vals.ndim == 3 and num_frames > 0 and h_tok > 0 and w_tok > 0:
        pooled = F.interpolate(
            vals.unsqueeze(1),
            size=(h_tok, w_tok),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        if pooled.numel() == default.numel():
            return pooled.reshape(-1).clamp(0.0, 1.0)
    return _flatten_or_default(vals, default).clamp(0.0, 1.0)


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu().float().reshape(-1)
    b = b.detach().cpu().float().reshape(-1)
    n = min(int(a.numel()), int(b.numel()))
    if n < 2:
        return 0.0
    a = a[:n]
    b = b[:n]
    mask = torch.isfinite(a) & torch.isfinite(b)
    if int(mask.sum().item()) < 2:
        return 0.0
    a = a[mask] - a[mask].mean()
    b = b[mask] - b[mask].mean()
    denom = (a.square().sum().sqrt() * b.square().sum().sqrt()).clamp_min(1e-8)
    return float((a * b).sum().div(denom).clamp(-1.0, 1.0).item())


def _global_centroid_metric(
    q_layers: Optional[torch.Tensor],
    k_layers: Optional[torch.Tensor],
    layer_ids: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    group: str,
    kind: str,
) -> torch.Tensor:
    if q_layers is None or k_layers is None:
        return torch.zeros_like(default)
    q_raw = q_layers.detach().cpu().float()
    k_raw = k_layers.detach().cpu().float()
    if q_raw.ndim != 5 or k_raw.ndim != 5:
        return torch.zeros_like(default)
    # Expected shape: [T, L, H_tok, W_tok, D].
    T, L, H, W, D = q_raw.shape
    if T <= 0 or L <= 0 or H <= 0 or W <= 0 or D <= 0:
        return torch.zeros_like(default)
    idx = _layer_indices(layer_ids, int(L), group)
    if idx.numel() == 0:
        return torch.zeros_like(default)
    q = F.normalize(q_raw.index_select(1, idx).reshape(T, idx.numel(), H * W, D), dim=-1)
    k = F.normalize(k_raw.index_select(1, idx).reshape(T, idx.numel(), H * W, D), dim=-1)
    q_cent = F.normalize(q.mean(dim=2), dim=-1)  # [T, Lg, D]
    k_cent = F.normalize(k.mean(dim=2), dim=-1)

    per_frame: List[torch.Tensor] = []
    for t in range(T):
        support = [s for s in range(T) if s != t]
        if not support:
            per_frame.append(torch.zeros((H * W,), dtype=torch.float32))
            continue
        if kind == "qq_low":
            cent = q_cent[support].permute(1, 0, 2)  # [Lg, S, D]
            sims = torch.einsum("lpd,lsd->lps", q[t], cent)
            raw = 1.0 - ((sims.mean(dim=(0, 2)) + 1.0) * 0.5)
        elif kind == "kk_low":
            cent = k_cent[support].permute(1, 0, 2)
            sims = torch.einsum("lpd,lsd->lps", k[t], cent)
            raw = 1.0 - ((sims.mean(dim=(0, 2)) + 1.0) * 0.5)
        elif kind == "deep_static":
            q_sim = torch.einsum("lpd,lsd->lps", q[t], q_cent[support].permute(1, 0, 2))
            k_sim = torch.einsum("lpd,lsd->lps", k[t], k_cent[support].permute(1, 0, 2))
            raw = (((q_sim.mean(dim=(0, 2)) + 1.0) * 0.5) + ((k_sim.mean(dim=(0, 2)) + 1.0) * 0.5)) * 0.5
        else:
            cent = k_cent[support].permute(1, 0, 2)
            sims = torch.einsum("lpd,lsd->lps", q[t], cent)
            raw = sims.var(dim=2, unbiased=False).mean(dim=0)
        per_frame.append(raw.reshape(-1))
    out = torch.stack(per_frame, dim=0).reshape(-1)
    return _flatten_or_default(out, default)


def _acl2_support_indices(
    num_frames: int,
    t: int,
    support: str,
    *,
    overlap_frames: int = 0,
) -> List[int]:
    if num_frames <= 1:
        return []
    support = support.lower()
    ov = min(max(int(overlap_frames), 0), max(int(num_frames) // 2, 0))
    seam = set(range(0, ov)) | set(range(max(num_frames - ov, 0), num_frames)) if ov > 0 else set()
    if support in {"full_chunk", "chunk_full", "full_chunk_true", "chunk_full_true"}:
        return [s for s in range(num_frames) if s != t]
    if support in {"full_chunk_no_overlap", "full_no_overlap", "no_overlap", "no_overlap_true"}:
        return [s for s in range(num_frames) if s != t and s not in seam]
    if support == "off246":
        offsets = (-6, -4, -2, 2, 4, 6)
        return [t + d for d in offsets if 0 <= t + d < num_frames]
    if support == "near12":
        offsets = (-2, -1, 1, 2)
        return [t + d for d in offsets if 0 <= t + d < num_frames]
    if support == "near24":
        offsets = (-4, -2, 2, 4)
        return [t + d for d in offsets if 0 <= t + d < num_frames]
    if support == "far612":
        offsets = (-12, -8, -6, 6, 8, 12)
        return [t + d for d in offsets if 0 <= t + d < num_frames]
    if support in {"past", "past_only"}:
        return list(range(0, t))
    if support in {"future", "future_only"}:
        return list(range(t + 1, num_frames))
    if support in {"past_plus_near_future12", "past_near_future12", "causal_past_plus_near_future"}:
        return list(range(0, t)) + [s for s in (t + 1, t + 2) if 0 <= s < num_frames]
    if support in {"past_plus_future_light", "past_future_light", "past075_future025", "past_plus_future_light_real"}:
        return [s for s in range(num_frames) if s != t]
    if support in {"full", "noovlp", "ovlp_only", "overlap_excluded"}:
        # The current HMC cue builder does not receive chunk-boundary metadata,
        # so noovlp/ovlp_only fall back to full support for this first ACL2 pass.
        return [s for s in range(num_frames) if s != t]
    return [s for s in range(num_frames) if s != t]


_ACL2_PAST_FUTURE_LIGHT_SUPPORTS = {
    "past_plus_future_light",
    "past_future_light",
    "past075_future025",
    "past_plus_future_light_real",
}


def _acl2_weighted_past_future_centroid(
    centroids: torch.Tensor,
    t: int,
    support_indices: List[int],
) -> torch.Tensor:
    past = [s for s in support_indices if s < t]
    future = [s for s in support_indices if s > t]
    pieces: List[torch.Tensor] = []
    weights: List[float] = []
    if past:
        pieces.append(centroids[past].mean(dim=0))
        weights.append(0.75)
    if future:
        pieces.append(centroids[future].mean(dim=0))
        weights.append(0.25)
    if not pieces:
        return centroids[support_indices].permute(1, 0, 2)
    w = torch.tensor(weights, dtype=centroids.dtype, device=centroids.device)
    w = w / w.sum().clamp_min(1e-12)
    return (torch.stack(pieces, dim=0) * w[:, None, None]).sum(dim=0).unsqueeze(1)


def _acl2_layer_indices(n_layers: int, layerwin: str) -> torch.Tensor:
    if n_layers <= 0:
        return torch.zeros((0,), dtype=torch.long)
    aliases = {
        "middle": "g6_11",
        "currentmiddle": "g6_11",
        "v4d_shallow": "g0",
        "v4d_mid": "g2_6",
        "v4d_deep": "g13_17",
        "v4d_deepvar": "g13_15",
    }
    spec = aliases.get(layerwin.lower(), layerwin.lower())
    if spec.startswith("g"):
        body = spec[1:]
        try:
            if "_" in body:
                a_str, b_str = body.split("_", 1)
                start = int(a_str)
                end = int(b_str)
            else:
                start = end = int(body)
        except ValueError:
            return torch.arange(n_layers, dtype=torch.long)
        start = max(0, min(start, n_layers - 1))
        end = max(0, min(end, n_layers - 1))
        if end < start:
            start, end = end, start
        return torch.arange(start, end + 1, dtype=torch.long)
    return torch.arange(n_layers, dtype=torch.long)


def _global_acl2_centroid_metric(
    q_layers: Optional[torch.Tensor],
    k_layers: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    basis: str,
    stat: str,
    layerwin: str,
    support: str,
    overlap_frames: int = 0,
) -> torch.Tensor:
    """Head-mean ACL2 global q/k cue.

    This first ACL2 implementation keeps the memory footprint of the existing
    head-mean export path.  It compares patch vectors with support-frame
    centroids for arbitrary layer windows and temporal supports; strict
    per-head/full attention-map statistics are left to the heavier cache path.
    """
    if q_layers is None or k_layers is None:
        return torch.zeros_like(default)
    q_raw = q_layers.detach().float()
    k_raw = k_layers.detach().float()
    if q_raw.ndim != 5 or k_raw.ndim != 5:
        return torch.zeros_like(default)
    T, L, H, W, D = q_raw.shape
    if T <= 0 or L <= 0 or H <= 0 or W <= 0 or D <= 0:
        return torch.zeros_like(default)
    idx = _acl2_layer_indices(int(L), layerwin)
    if idx.numel() == 0:
        return torch.zeros_like(default)

    q = F.normalize(q_raw.index_select(1, idx).reshape(T, idx.numel(), H * W, D), dim=-1)
    k = F.normalize(k_raw.index_select(1, idx).reshape(T, idx.numel(), H * W, D), dim=-1)
    q_cent = q.mean(dim=2)  # [T, Lg, D], intentionally not renormalized.
    k_cent = k.mean(dim=2)

    basis = basis.lower()
    stat = stat.lower()
    support_key = support.lower()
    full_support = support_key in {
        "full",
        "full_chunk",
        "chunk_full",
        "full_chunk_true",
        "chunk_full_true",
        "noovlp",
        "ovlp_only",
        "overlap_excluded",
    }
    no_overlap_support = support_key in {"full_chunk_no_overlap", "full_no_overlap", "no_overlap", "no_overlap_true"}

    if full_support and not no_overlap_support and stat != "var" and T > 1:
        if basis == "kk":
            tgt_all = k
            cent_all = k_cent
        elif basis == "qk":
            tgt_all = q
            cent_all = k_cent
        else:
            tgt_all = q
            cent_all = q_cent
        sum_cent = cent_all.sum(dim=0, keepdim=True)
        supp_cent = (sum_cent - cent_all) / float(T - 1)  # [T, Lg, D]
        sim = (tgt_all * supp_cent.unsqueeze(2)).sum(dim=-1)  # [T, Lg, P]
        sim01 = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)
        raw = sim01.mean(dim=1)
        if stat not in {"high", "mean"}:
            raw = 1.0 - raw
        return _flatten_or_default(raw.reshape(-1), default)

    per_frame: List[torch.Tensor] = []
    for t in range(T):
        supp = _acl2_support_indices(T, t, support, overlap_frames=overlap_frames)
        if not supp:
            per_frame.append(torch.zeros((H * W,), dtype=torch.float32))
            continue
        if basis == "kk":
            tgt = k[t]
            if support_key in _ACL2_PAST_FUTURE_LIGHT_SUPPORTS:
                cent = _acl2_weighted_past_future_centroid(k_cent, t, supp)
            else:
                cent = k_cent[supp].permute(1, 0, 2)
        elif basis == "qk":
            tgt = q[t]
            if support_key in _ACL2_PAST_FUTURE_LIGHT_SUPPORTS:
                cent = _acl2_weighted_past_future_centroid(k_cent, t, supp)
            else:
                cent = k_cent[supp].permute(1, 0, 2)
        else:
            tgt = q[t]
            if support_key in _ACL2_PAST_FUTURE_LIGHT_SUPPORTS:
                cent = _acl2_weighted_past_future_centroid(q_cent, t, supp)
            else:
                cent = q_cent[supp].permute(1, 0, 2)
        sim = torch.einsum("lpd,lsd->lps", tgt, cent)
        sim01 = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)
        if stat == "var":
            raw = sim01.var(dim=2, unbiased=False).mean(dim=0)
        elif stat in {"high", "mean"}:
            raw = sim01.mean(dim=(0, 2))
        else:
            raw = 1.0 - sim01.mean(dim=(0, 2))
        per_frame.append(raw.reshape(-1))
    out = torch.stack(per_frame, dim=0).reshape(-1)
    return _flatten_or_default(out, default)


def _acl2_global_patch(
    q_layers: Optional[torch.Tensor],
    k_layers: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    num_frames: int,
    basis: str,
    stat: str,
    layerwin: str,
    support: str,
    overlap_frames: int = 0,
) -> torch.Tensor:
    raw = _global_acl2_centroid_metric(
        q_layers,
        k_layers,
        default,
        basis=basis,
        stat=stat,
        layerwin=layerwin,
        support=support,
        overlap_frames=overlap_frames,
    )
    return _robust_quantile01(raw, num_frames=num_frames)


def _acl2_read_patch_from_source(
    source: str,
    q_layers: Optional[torch.Tensor],
    k_layers: Optional[torch.Tensor],
    default: torch.Tensor,
    *,
    num_frames: int,
    overlap_frames: int = 0,
) -> Optional[torch.Tensor]:
    src = source
    if src.startswith("acl2."):
        src = src[len("acl2."):]

    if src.startswith("gg."):
        parts = src.split(".")
        # gg.qq.low.g13_15.off246.headmean.robustq
        if len(parts) >= 7 and parts[0] == "gg" and parts[5].startswith("head") and parts[6] == "robustq":
            basis, stat, layerwin, support = parts[1], parts[2], parts[3], parts[4]
            return _acl2_global_patch(
                q_layers,
                k_layers,
                default,
                num_frames=num_frames,
                basis=basis,
                stat=stat,
                layerwin=layerwin,
                support=support,
                overlap_frames=overlap_frames,
            )

    if not src.startswith("v4d."):
        return None

    mean1 = _acl2_global_patch(
        q_layers, k_layers, default,
        num_frames=num_frames, basis="qq", stat="high", layerwin="g0", support="off246",
        overlap_frames=overlap_frames,
    )
    var1 = _acl2_global_patch(
        q_layers, k_layers, default,
        num_frames=num_frames, basis="qq", stat="var", layerwin="g0_2", support="off246",
        overlap_frames=overlap_frames,
    )
    mean2_low = _acl2_global_patch(
        q_layers, k_layers, default,
        num_frames=num_frames, basis="qq", stat="low", layerwin="g2_6", support="off246",
        overlap_frames=overlap_frames,
    )
    mean3_high = _acl2_global_patch(
        q_layers, k_layers, default,
        num_frames=num_frames, basis="kk", stat="high", layerwin="g13_17", support="off246",
        overlap_frames=overlap_frames,
    )
    var3_high = _acl2_global_patch(
        q_layers, k_layers, default,
        num_frames=num_frames, basis="qk", stat="var", layerwin="g13_15", support="off246",
        overlap_frames=overlap_frames,
    )
    var3_low = (1.0 - var3_high).clamp(0.0, 1.0)

    if src == "v4d.mean1":
        return mean1
    if src == "v4d.var1":
        return var1
    if src in {"v4d.mean2", "v4d.mean2.low"}:
        return mean2_low
    if src in {"v4d.mean3", "v4d.mean3.high", "v4d.mean3.high_as_rescue"}:
        return mean3_high
    if src in {"v4d.var3", "v4d.var3.low", "v4d.var3.low_as_rescue_conf"}:
        return var3_low
    if src == "v4d.product.m1_m2low_m3":
        raw = mean1 * mean2_low * mean3_high
    elif src == "v4d.product.v1_m2low_m3":
        raw = var1 * mean2_low * mean3_high
    elif src == "v4d.product.m2low_m3":
        raw = mean2_low * mean3_high
    elif src == "v4d.product.m1_m2low_m3_v3low":
        raw = mean1 * mean2_low * mean3_high * var3_low
    elif src == "v4d.product.m2low_rescue":
        rescue = (mean3_high * var3_low).clamp(0.0, 1.0)
        raw = mean2_low * (1.0 - 0.5 * rescue).clamp(0.0, 1.0)
    else:
        return None
    return _robust_quantile01(raw, num_frames=num_frames)


class HybridMemoryController:
    """v2 controller coordinating probe and controlled passes."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        lambda_min: float = 1.0,
        lambda_max: float = 1.0,
        eta_mean_preserve: bool = False,
        eta_norm_eps: float = 1e-8,
        prior_branch_mask: str = "0",
        prior_layer_mode: str = "all",
        prior_single_layer: int = -1,
        prior_layer_branch_policy: Optional[str] = None,
        ttt_write_delta_scale: float = 1.0,
        ttt_write_delta_scales: Optional[str] = None,
        ttt_write_native_mix_scales: Optional[str] = None,
        ttt_write_native_mix_chunks: Optional[str] = None,
        ttt_write_prior_transform_mode: str = "none",
        ttt_write_prior_anti_scale: float = 0.0,
        ttt_write_prior_gamma: float = 1.0,
        ttt_write_special_token_policy: str = "none",
        ttt_write_special_token_floor: float = 0.0,
        ttt_write_special_token_ceiling: float = 1.0,
        ttt_write_gradient_reversal_mode: str = "none",
        ttt_write_gradient_reversal_gamma: float = 0.0,
        ttt_write_gradient_reversal_branch_mask: str = "0",
        ttt_write_gradient_reversal_branch_gammas: Optional[str] = None,
        ttt_write_gradient_reversal_layer_gammas: Optional[str] = None,
        ttt_write_gradient_reversal_head_routes: Optional[str] = None,
        ttt_write_gradient_reversal_negative_frac: float = 0.0,
        ttt_write_gradient_reversal_risk_source: str = "prior",
        ttt_write_tri_replay_positive_frac: float = 0.35,
        ttt_write_tri_replay_negative_frac: float = 0.15,
        ttt_write_tri_replay_neutral_lambda: float = 1.0,
        ttt_write_tri_replay_role_mode: str = "fixed",
        ttt_write_gradient_reversal_transient_mode: str = "none",
        ttt_write_gradient_reversal_transient_branch_mask: str = "",
        ttt_write_gradient_reversal_transient_long_scale: float = 0.0,
        ttt_write_gradient_reversal_transient_apply_scale: float = 1.0,
        ttt_write_replay_feature_gate_mode: str = "none",
        ttt_write_replay_feature_gate_rho: float = 0.0,
        ttt_write_replay_feature_gate_min: float = 0.5,
        ttt_write_replay_feature_gate_branch_mask: str = "all",
        ttt_write_replay_token_filter_mode: str = "none",
        ttt_write_replay_token_filter_ratio: float = 1.0,
        ttt_write_replay_token_filter_threshold: float = 1.0,
        ttt_write_replay_token_filter_scope: str = "all",
        ttt_write_replay_token_filter_branch_mask: str = "all",
        ttt_write_replay_token_filter_blend: float = 1.0,
        ttt_write_replay_token_filter_blend_mode: str = "linear",
        ttt_write_transient_delta_subtract_scale: float = 0.0,
        ttt_write_transient_delta_branch_mask: str = "0",
        ttt_write_transient_delta_ttl: int = 1,
        ttt_write_commit_ema_alpha: float = 1.0,
        ttt_write_commit_ema_branch_mask: str = "all",
        ttt_write_commit_ema_chunks: Optional[str] = None,
        ttt_write_native_delta_gate_mode: str = "none",
        ttt_write_native_delta_gate_min_cos: float = 0.0,
        ttt_write_native_delta_gate_fallback: float = 0.0,
        ttt_write_native_delta_gate_cap_ratio: float = 1.0,
        ttt_write_native_delta_gate_branch_mask: str = "all",
        ttt_write_commit_filter_mode: str = "none",
        ttt_write_commit_filter_risk_source: str = "d_tok",
        ttt_write_commit_filter_scope: str = "tail_overlap",
        ttt_write_commit_filter_stat: str = "mean",
        ttt_write_commit_filter_base: float = 0.0,
        ttt_write_commit_filter_gain: float = 1.0,
        ttt_write_commit_filter_min: float = 0.0,
        ttt_write_commit_filter_max: float = 1.0,
        ttt_write_commit_filter_branch_mask: str = "0",
        ttt_write_scale_state_mode: str = "none",
        ttt_write_scale_state_proxy: str = "pose_step_ema",
        ttt_write_scale_state_carrier: str = "all",
        ttt_write_scale_state_alpha: float = 0.0,
        ttt_write_scale_state_branch_mask: str = "0",
        ttt_write_scale_state_chunks: str = "",
        ttt_write_scale_state_sample_tokens: int = 0,
        enable_frame_read_control: bool = False,
        enable_swa_read_control: bool = False,
        enable_ttt_apply_control: bool = False,
        enable_chunk_read_control: bool = False,
        beta_frame: float = 0.0,
        beta_swa: float = 0.0,
        beta_chunk: float = 0.0,
        swa_gate_min: float = 0.85,
        enable_swa_overlap_bias: bool = False,
        swa_overlap_bias_beta: float = 0.0,
        swa_overlap_bias_min_keep: float = 1e-4,
        swa_overlap_bias_mode: str = "pair",
        swa_overlap_bias_layer_mode: str = "last",
        swa_overlap_bias_single_layer: int = -1,
        swa_overlap_bias_head_indices: str = "",
        swa_overlap_bias_record_attention_mass: bool = False,
        swa_overlap_bias_attention_mass_max_queries: int = 64,
        swa_overlap_external_mask_csv: str = "",
        swa_overlap_external_mask_variant: str = "current_role_anchor",
        swa_overlap_external_mask_seq: str = "",
        enable_swa_overlap_source_gate: bool = False,
        swa_overlap_source_gate_rho: float = 0.0,
        swa_overlap_source_gate_min: float = 0.85,
        swa_overlap_source_gate_mode: str = "source",
        swa_overlap_source_gate_target: str = "v",
        swa_overlap_source_gate_layer_mode: str = "last",
        swa_overlap_source_gate_single_layer: int = -1,
        enable_swa_prev_ttt_stable_anchor_gate: bool = False,
        swa_prev_ttt_stable_anchor_gate_rho: float = 0.0,
        swa_prev_ttt_stable_anchor_gate_min: float = 0.85,
        swa_prev_ttt_stable_anchor_gate_target: str = "v",
        swa_prev_ttt_stable_anchor_gate_layer_mode: str = "last",
        swa_prev_ttt_stable_anchor_gate_single_layer: int = -1,
        enable_swa_prev_ttt_anchor_query_soft: bool = False,
        swa_prev_ttt_anchor_query_soft_rho: float = 0.0,
        swa_prev_ttt_anchor_query_soft_min_keep: float = 0.5,
        swa_prev_ttt_anchor_query_soft_query_head_frac_threshold: float = 0.75,
        swa_prev_ttt_anchor_query_soft_topk: int = 8,
        swa_prev_ttt_anchor_query_soft_query_block_size: int = 64,
        swa_prev_ttt_anchor_query_soft_layer_mode: str = "last",
        swa_prev_ttt_anchor_query_soft_single_layer: int = -1,
        swa_prev_ttt_anchor_query_soft_attention_mass_max_queries: int = 64,
        enable_swa_prev_ttt_tracked_instance_query_soft_trace: bool = False,
        swa_prev_ttt_tracked_instance_query_soft_rho: float = 0.0,
        swa_prev_ttt_tracked_instance_query_soft_min_keep: float = 0.5,
        swa_prev_ttt_tracked_instance_query_soft_query_head_frac_threshold: float = 0.75,
        swa_prev_ttt_tracked_instance_query_soft_topk: int = 4,
        swa_prev_ttt_tracked_instance_query_soft_query_block_size: int = 64,
        swa_prev_ttt_tracked_instance_query_soft_layer_mode: str = "last",
        swa_prev_ttt_tracked_instance_query_soft_single_layer: int = -1,
        swa_prev_ttt_tracked_instance_query_soft_attention_mass_max_queries: int = 64,
        swa_prev_ttt_tracked_instance_query_soft_min_direct_witness_seeds: int = 4,
        swa_prev_ttt_tracked_instance_query_soft_direct_match_mode: str = "any",
        enable_swa_prev_ttt_tracked_instance_query_soft_action: bool = False,
        swa_prev_ttt_tracked_instance_query_soft_action_runtime_authorized: bool = False,
        enable_swa_overlap_source_replace: bool = False,
        swa_overlap_source_replace_alpha: float = 0.0,
        swa_overlap_source_replace_mode: str = "union",
        swa_overlap_source_replace_target: str = "kv",
        swa_overlap_source_replace_layer_mode: str = "last",
        swa_overlap_source_replace_single_layer: int = -1,
        enable_v102_state_machine_trace: bool = False,
        v102_state_machine_action: str = "TRANSMIT_SUPPORTED_ANCHORS",
        v102_state_machine_layer_mode: str = "last",
        v102_state_machine_single_layer: int = -1,
        v102_state_machine_strict_gate_pass: bool = False,
        v102_state_machine_true_l3_gate_pass: bool = False,
        enable_v102_state_machine_action_probe: bool = False,
        v102_state_machine_probe_impl: str = "compact_kv_reject_unreliable",
        v102_state_machine_unreliable_d_min: float = 0.50,
        v102_state_machine_unreliable_g_min: float = 0.50,
        v102_state_machine_supported_d_max: float = 0.25,
        v102_state_machine_supported_k_min: float = 0.0,
        v102_state_machine_supported_require_static_semantic: bool = True,
        v102_state_machine_soft_unsupported_min_keep: float = 0.50,
        v102_state_machine_hold_prev_frames: int = 1,
        v102_state_machine_hold_soft_min_keep: float = 0.50,
        v102_state_machine_delay_current_soft_min_keep: float = 0.50,
        v102_state_machine_context_soft_min_keep: float = 0.50,
        v102_state_machine_min_history_keep_frac: float = 0.05,
        v102_state_machine_attention_mass_max_queries: int = 64,
        swa_overlap_feature_dump_dir: str = "",
        swa_overlap_feature_dump_dtype: str = "float16",
        swa_raw_transport_trace_dir: str = "",
        swa_raw_transport_trace_layer_mode: str = "all",
        swa_raw_transport_trace_single_layer: int = -1,
        swa_raw_transport_trace_max_queries: int = 128,
        swa_raw_transport_trace_topk: int = 8,
        swa_raw_transport_trace_direct_match_only: bool = False,
        swa_raw_transport_trace_query_block_size: int = 128,
        enable_context_source_skip: bool = False,
        context_source_skip_impl: str = "bias",
        context_source_skip_scope: str = "frame",
        context_source_skip_mode: str = "hard",
        context_source_skip_mask: str = "dg_q90",
        context_source_skip_frame_region: str = "all",
        context_source_skip_query_region: str = "all",
        context_source_skip_layer_mode: str = "early",
        context_source_skip_single_layer: int = -1,
        context_source_skip_head_indices: str = "",
        context_source_skip_soft_rho: float = 0.5,
        context_source_skip_soft_min_keep: float = 0.5,
        context_source_skip_source_attention_top_quantile: float = -1.0,
        context_source_skip_source_attention_top_random_same_mass: bool = False,
        context_source_skip_record_attention_mass: bool = False,
        context_source_skip_attention_mass_max_queries: int = 512,
        context_source_skip_attention_map_dump_dir: str = "",
        context_source_skip_attention_map_dump_max_queries: int = 64,
        context_source_skip_attention_map_dump_dtype: str = "float16",
        context_source_skip_attention_map_dump_full_query_marginal: bool = False,
        context_source_skip_attention_map_dump_head_marginal: bool = False,
        context_source_skip_attention_map_dump_query_block: int = 32,
        semantic_role_policy: str = "none",
        semantic_memory_paths: str = "",
        semantic_role_highd_quantile: float = 0.80,
        semantic_role_low_trust: float = 0.20,
        semantic_role_positive_scale: float = 1.05,
        semantic_role_neutral_scale: float = 0.85,
        semantic_role_negative_scale: float = 0.65,
        semantic_role_ttt_read_harm_veto_mode: str = "none",
        semantic_role_ttt_read_harm_veto_scale: float = 0.0,
        semantic_role_ttt_read_harm_veto_min_harm: int = 1,
        semantic_role_ttt_read_harm_veto_max_stable_positive: int = 0,
        semantic_role_swa_negative_scale: float = 1.0,
        semantic_role_swa_protect_scale: float = 1.0,
        semantic_role_control_mode: str = "none",
        semantic_role_control_seed: int = 12345,
        semantic_swa_redirection_phase3_root: str = "",
        semantic_swa_redirection_stable_case: str = "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST",
        semantic_swa_redirection_random_stable_case: str = "P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST",
        semantic_swa_redirection_layer: int = 3,
        semantic_swa_redirection_top_quantile: float = 0.75,
        semantic_swa_redirection_overlap_frames: int = 3,
        semantic_swa_redirection_random_control: bool = False,
        semantic_anchor_mode: str = "semantic",
        semantic_anchor_target_ratio: float = 0.12,
        semantic_anchor_min_ratio: float = 0.03,
        semantic_anchor_max_ratio: float = 0.30,
        semantic_anchor_min_score: float = 0.02,
        semantic_anchor_grid_rows: int = 4,
        semantic_anchor_grid_cols: int = 4,
        semantic_anchor_missing_trust_policy: str = "zero",
        semantic_anchor_value_fallback: str = "off",
        enable_semantic_anchor_ttt_floor: bool = False,
        rho_ttt_apply: float = 0.0,
        read_layer_mode: str = "all",
        read_single_layer: int = -1,
        read_cue_source: str = "dyn",
        v70_radio_sidecar_dir: str = "",
        enable_v70_radio_ttt_write_prior: bool = False,
        v70_radio_ttt_mode: str = "static_object_write_floor",
        v70_radio_ttt_control: str = "none",
        v70_radio_ttt_static_boost: float = 0.35,
        v70_radio_ttt_suppress: float = 0.50,
        v70_radio_ttt_min_confidence: float = 0.45,
        v70_radio_ttt_min_stability: float = 0.35,
        v70_radio_ttt_min_interior: float = 0.25,
        v70_radio_ttt_max_activity_risk: float = 0.85,
        v70_radio_ttt_boundary_threshold: float = 0.85,
        v70_radio_ttt_renorm_mean: bool = True,
        semantic_ttt_overlap_support_dir: str = "",
        semantic_ttt_overlap_support_kind: str = "source_gate",
        semantic_ttt_overlap_support_score_key: str = "score_overlap",
        semantic_ttt_overlap_support_scope: str = "head_overlap",
        semantic_ttt_overlap_support_floor: float = 0.0,
        semantic_ttt_overlap_support_renorm_mean: bool = False,
        read_topk_frac: float = 0.0,
        frame_bias_mode: str = "pair",
        frame_bias_query_region: str = "all",
        frame_bias_head_indices: str = "",
        frame_attention_record_bias_mass: bool = False,
        frame_attention_bias_mass_max_queries: int = 64,
        chunk_bias_mode: str = "key",
        swa_bias_mode: str = "prev_key",
        flow_model: str = "none",
        flow_pair_stride: int = 1,
        flow_fb_thr: float = 1.5,
        flow_residual_thr: float = 0.15,
        gram_layer_groups: str = "shallow,middle,deep",
        read_calib_mode: str = "none",
        read_target_mass: float = 0.06,
        read_calib_tau: float = 0.05,
        read_blend_lambda: float = 0.25,
        read_quality_mass_min: float = 0.03,
        read_quality_mass_max: float = 0.20,
        read_quality_anchor_max: float = 0.35,
        read_quality_frag_max: float = 0.15,
        beta_policy: str = "fixed",
        beta_energy_target: float = 0.0,
        beta_min: float = 0.5,
        beta_max: float = 1.5,
        read_protect_ref: bool = True,
        read_protect_static: bool = False,
        read_protection_mode: str = "none",
        read_ref_strength: float = 1.0,
        read_overlap_frames: int = 0,
        read_reset_frames: int = 1,
        read_attention_q: float = 0.90,
        read_static_anchor_thr: float = 0.6,
        read_static_dyn_thr: float = 0.3,
        hmc_write_score_source: str = "stage_d",
        hmc_write_sparse_ratio: float = 1.0,
        hmc_write_sparse_mode: str = "none",
        hmc_write_alpha: float = 0.1,
        hmc_write_min: float = 0.8,
        hmc_write_max: float = 1.2,
        ttt_apply_min_gate: float = 0.0,
        enable_swa_write_control: bool = False,
        swa_write_mode: str = "none",
        swa_write_rho: float = 0.0,
        swa_write_min_gate: float = 0.0,
        swa_write_sparse_ratio: float = 1.0,
        swa_write_layer_mode: str = "all",
        swa_write_single_layer: int = -1,
        swa_write_scope: str = "all",
        swa_write_keep_scope: str = "all",
        swa_write_score_source: str = "read",
        swa_write_cache_blend_alpha: float = 0.0,
        swa_write_cache_blend_mode: str = "dynamic",
        swa_write_cache_blend_target: str = "v",
        ttt_write_token_scope: str = "all",
        ttt_write_token_scope_floor: float = 0.0,
        semantic_action_active_chunks: str = "",
        semantic_action_inactive_read_cue_source: str = "",
        v32_semantic_cue_active_chunks: str = "",
        v32_semantic_cue_trigger_mode: str = "off",
        v32_semantic_cue_trigger_high_threshold: float = 0.80,
        v32_semantic_cue_trigger_k: float = 1.5,
        v32_semantic_cue_trigger_min_mass: float = 0.05,
        v32_semantic_cue_trigger_warmup: int = 3,
        v32_semantic_cue_trigger_ema_alpha: float = 0.25,
        read_cue_patch_dump_dir: str = "",
        read_cue_patch_dump_dtype: str = "float16",
        fast_cue_eval: bool = False,
    ):
        self.device = device
        self.enable_frame_read_control = bool(enable_frame_read_control)
        self.enable_swa_read_control = bool(enable_swa_read_control)
        self.enable_ttt_apply_control = bool(enable_ttt_apply_control)
        self.enable_chunk_read_control = bool(enable_chunk_read_control)
        self.beta_frame = float(beta_frame)
        self.beta_swa = float(beta_swa)
        self.beta_chunk = float(beta_chunk)
        self.swa_gate_min = float(swa_gate_min)
        self.enable_swa_overlap_bias = bool(enable_swa_overlap_bias)
        self.swa_overlap_bias_beta = float(swa_overlap_bias_beta)
        self.swa_overlap_bias_min_keep = float(swa_overlap_bias_min_keep)
        self.swa_overlap_bias_mode = str(swa_overlap_bias_mode)
        self.swa_overlap_bias_layer_mode = str(swa_overlap_bias_layer_mode)
        self.swa_overlap_bias_single_layer = int(swa_overlap_bias_single_layer)
        self.swa_overlap_bias_head_indices = str(swa_overlap_bias_head_indices or "")
        self.swa_overlap_bias_record_attention_mass = bool(swa_overlap_bias_record_attention_mass)
        self.swa_overlap_bias_attention_mass_max_queries = int(swa_overlap_bias_attention_mass_max_queries)
        self.swa_overlap_external_mask_csv = str(swa_overlap_external_mask_csv or "")
        self.swa_overlap_external_mask_variant = str(swa_overlap_external_mask_variant or "current_role_anchor")
        self.swa_overlap_external_mask_seq = str(swa_overlap_external_mask_seq or "")
        self.enable_swa_overlap_source_gate = bool(enable_swa_overlap_source_gate)
        self.swa_overlap_source_gate_rho = float(swa_overlap_source_gate_rho)
        self.swa_overlap_source_gate_min = float(swa_overlap_source_gate_min)
        self.swa_overlap_source_gate_mode = str(swa_overlap_source_gate_mode)
        self.swa_overlap_source_gate_target = str(swa_overlap_source_gate_target)
        self.swa_overlap_source_gate_layer_mode = str(swa_overlap_source_gate_layer_mode)
        self.swa_overlap_source_gate_single_layer = int(swa_overlap_source_gate_single_layer)
        self.enable_swa_prev_ttt_stable_anchor_gate = bool(enable_swa_prev_ttt_stable_anchor_gate)
        self.swa_prev_ttt_stable_anchor_gate_rho = float(swa_prev_ttt_stable_anchor_gate_rho)
        self.swa_prev_ttt_stable_anchor_gate_min = float(swa_prev_ttt_stable_anchor_gate_min)
        self.swa_prev_ttt_stable_anchor_gate_target = str(swa_prev_ttt_stable_anchor_gate_target)
        self.swa_prev_ttt_stable_anchor_gate_layer_mode = str(swa_prev_ttt_stable_anchor_gate_layer_mode)
        self.swa_prev_ttt_stable_anchor_gate_single_layer = int(swa_prev_ttt_stable_anchor_gate_single_layer)
        self.enable_swa_prev_ttt_anchor_query_soft = bool(enable_swa_prev_ttt_anchor_query_soft)
        self.swa_prev_ttt_anchor_query_soft_rho = float(swa_prev_ttt_anchor_query_soft_rho)
        self.swa_prev_ttt_anchor_query_soft_min_keep = float(swa_prev_ttt_anchor_query_soft_min_keep)
        self.swa_prev_ttt_anchor_query_soft_query_head_frac_threshold = float(
            swa_prev_ttt_anchor_query_soft_query_head_frac_threshold
        )
        self.swa_prev_ttt_anchor_query_soft_topk = int(swa_prev_ttt_anchor_query_soft_topk)
        self.swa_prev_ttt_anchor_query_soft_query_block_size = int(swa_prev_ttt_anchor_query_soft_query_block_size)
        self.swa_prev_ttt_anchor_query_soft_layer_mode = str(swa_prev_ttt_anchor_query_soft_layer_mode)
        self.swa_prev_ttt_anchor_query_soft_single_layer = int(swa_prev_ttt_anchor_query_soft_single_layer)
        self.swa_prev_ttt_anchor_query_soft_attention_mass_max_queries = int(
            swa_prev_ttt_anchor_query_soft_attention_mass_max_queries
        )
        self.enable_swa_prev_ttt_tracked_instance_query_soft_trace = bool(
            enable_swa_prev_ttt_tracked_instance_query_soft_trace
        )
        self.swa_prev_ttt_tracked_instance_query_soft_rho = float(
            swa_prev_ttt_tracked_instance_query_soft_rho
        )
        self.swa_prev_ttt_tracked_instance_query_soft_min_keep = float(
            swa_prev_ttt_tracked_instance_query_soft_min_keep
        )
        self.swa_prev_ttt_tracked_instance_query_soft_query_head_frac_threshold = float(
            swa_prev_ttt_tracked_instance_query_soft_query_head_frac_threshold
        )
        self.swa_prev_ttt_tracked_instance_query_soft_topk = int(
            swa_prev_ttt_tracked_instance_query_soft_topk
        )
        self.swa_prev_ttt_tracked_instance_query_soft_query_block_size = int(
            swa_prev_ttt_tracked_instance_query_soft_query_block_size
        )
        self.swa_prev_ttt_tracked_instance_query_soft_layer_mode = str(
            swa_prev_ttt_tracked_instance_query_soft_layer_mode
        )
        self.swa_prev_ttt_tracked_instance_query_soft_single_layer = int(
            swa_prev_ttt_tracked_instance_query_soft_single_layer
        )
        self.swa_prev_ttt_tracked_instance_query_soft_attention_mass_max_queries = int(
            swa_prev_ttt_tracked_instance_query_soft_attention_mass_max_queries
        )
        self.swa_prev_ttt_tracked_instance_query_soft_min_direct_witness_seeds = int(
            swa_prev_ttt_tracked_instance_query_soft_min_direct_witness_seeds
        )
        self.swa_prev_ttt_tracked_instance_query_soft_direct_match_mode = str(
            swa_prev_ttt_tracked_instance_query_soft_direct_match_mode or "any"
        )
        self.enable_swa_prev_ttt_tracked_instance_query_soft_action = bool(
            enable_swa_prev_ttt_tracked_instance_query_soft_action
        )
        self.swa_prev_ttt_tracked_instance_query_soft_action_runtime_authorized = bool(
            swa_prev_ttt_tracked_instance_query_soft_action_runtime_authorized
        )
        self.enable_swa_overlap_source_replace = bool(enable_swa_overlap_source_replace)
        self.swa_overlap_source_replace_alpha = float(swa_overlap_source_replace_alpha)
        self.swa_overlap_source_replace_mode = str(swa_overlap_source_replace_mode)
        self.swa_overlap_source_replace_target = str(swa_overlap_source_replace_target)
        self.swa_overlap_source_replace_layer_mode = str(swa_overlap_source_replace_layer_mode)
        self.swa_overlap_source_replace_single_layer = int(swa_overlap_source_replace_single_layer)
        self.enable_v102_state_machine_trace = bool(enable_v102_state_machine_trace)
        self.v102_state_machine_action = str(v102_state_machine_action or "TRANSMIT_SUPPORTED_ANCHORS")
        self.v102_state_machine_layer_mode = str(v102_state_machine_layer_mode or "last")
        self.v102_state_machine_single_layer = int(v102_state_machine_single_layer)
        self.v102_state_machine_strict_gate_pass = bool(v102_state_machine_strict_gate_pass)
        self.v102_state_machine_true_l3_gate_pass = bool(v102_state_machine_true_l3_gate_pass)
        self.enable_v102_state_machine_action_probe = bool(enable_v102_state_machine_action_probe)
        self.v102_state_machine_probe_impl = str(v102_state_machine_probe_impl or "compact_kv_reject_unreliable")
        self.v102_state_machine_unreliable_d_min = float(v102_state_machine_unreliable_d_min)
        self.v102_state_machine_unreliable_g_min = float(v102_state_machine_unreliable_g_min)
        self.v102_state_machine_supported_d_max = float(v102_state_machine_supported_d_max)
        self.v102_state_machine_supported_k_min = float(v102_state_machine_supported_k_min)
        self.v102_state_machine_supported_require_static_semantic = bool(
            v102_state_machine_supported_require_static_semantic
        )
        self.v102_state_machine_soft_unsupported_min_keep = float(v102_state_machine_soft_unsupported_min_keep)
        self.v102_state_machine_hold_prev_frames = int(v102_state_machine_hold_prev_frames)
        self.v102_state_machine_hold_soft_min_keep = float(v102_state_machine_hold_soft_min_keep)
        self.v102_state_machine_delay_current_soft_min_keep = float(v102_state_machine_delay_current_soft_min_keep)
        self.v102_state_machine_context_soft_min_keep = float(v102_state_machine_context_soft_min_keep)
        self.v102_state_machine_min_history_keep_frac = float(v102_state_machine_min_history_keep_frac)
        self.v102_state_machine_attention_mass_max_queries = int(v102_state_machine_attention_mass_max_queries)
        self.swa_overlap_feature_dump_dir = str(swa_overlap_feature_dump_dir)
        self.swa_overlap_feature_dump_dtype = str(swa_overlap_feature_dump_dtype)
        self.swa_raw_transport_trace_dir = str(swa_raw_transport_trace_dir or "")
        self.swa_raw_transport_trace_layer_mode = str(swa_raw_transport_trace_layer_mode or "all")
        self.swa_raw_transport_trace_single_layer = int(swa_raw_transport_trace_single_layer)
        self.swa_raw_transport_trace_max_queries = int(swa_raw_transport_trace_max_queries)
        self.swa_raw_transport_trace_topk = int(swa_raw_transport_trace_topk)
        self.swa_raw_transport_trace_direct_match_only = bool(swa_raw_transport_trace_direct_match_only)
        self.swa_raw_transport_trace_query_block_size = int(swa_raw_transport_trace_query_block_size)
        self.enable_context_source_skip = bool(enable_context_source_skip)
        self.context_source_skip_impl = str(context_source_skip_impl)
        self.context_source_skip_scope = str(context_source_skip_scope)
        self.context_source_skip_mode = str(context_source_skip_mode)
        self.context_source_skip_mask = str(context_source_skip_mask)
        self.context_source_skip_frame_region = str(context_source_skip_frame_region)
        self.context_source_skip_query_region = str(context_source_skip_query_region)
        self.context_source_skip_layer_mode = str(context_source_skip_layer_mode)
        self.context_source_skip_single_layer = int(context_source_skip_single_layer)
        self.context_source_skip_head_indices = str(context_source_skip_head_indices)
        self.context_source_skip_soft_rho = float(context_source_skip_soft_rho)
        self.context_source_skip_soft_min_keep = float(context_source_skip_soft_min_keep)
        self.context_source_skip_source_attention_top_quantile = float(context_source_skip_source_attention_top_quantile)
        self.context_source_skip_source_attention_top_random_same_mass = bool(
            context_source_skip_source_attention_top_random_same_mass
        )
        self.context_source_skip_record_attention_mass = bool(context_source_skip_record_attention_mass)
        self.context_source_skip_attention_mass_max_queries = int(context_source_skip_attention_mass_max_queries)
        self.context_source_skip_attention_map_dump_dir = str(context_source_skip_attention_map_dump_dir)
        self.context_source_skip_attention_map_dump_max_queries = int(context_source_skip_attention_map_dump_max_queries)
        self.context_source_skip_attention_map_dump_dtype = str(context_source_skip_attention_map_dump_dtype)
        self.context_source_skip_attention_map_dump_full_query_marginal = bool(
            context_source_skip_attention_map_dump_full_query_marginal
        )
        self.context_source_skip_attention_map_dump_head_marginal = bool(
            context_source_skip_attention_map_dump_head_marginal
        )
        self.context_source_skip_attention_map_dump_query_block = int(
            context_source_skip_attention_map_dump_query_block
        )
        self.semantic_role_policy = str(semantic_role_policy)
        self.semantic_memory_paths = str(semantic_memory_paths)
        self.semantic_role_highd_quantile = float(semantic_role_highd_quantile)
        self.semantic_role_low_trust = float(semantic_role_low_trust)
        self.semantic_role_positive_scale = float(semantic_role_positive_scale)
        self.semantic_role_neutral_scale = float(semantic_role_neutral_scale)
        self.semantic_role_negative_scale = float(semantic_role_negative_scale)
        self.semantic_role_ttt_read_harm_veto_mode = str(
            semantic_role_ttt_read_harm_veto_mode or "none"
        ).strip().lower()
        self.semantic_role_ttt_read_harm_veto_scale = max(
            0.0,
            min(1.5, float(semantic_role_ttt_read_harm_veto_scale)),
        )
        self.semantic_role_ttt_read_harm_veto_min_harm = max(
            0,
            int(semantic_role_ttt_read_harm_veto_min_harm),
        )
        self.semantic_role_ttt_read_harm_veto_max_stable_positive = max(
            0,
            int(semantic_role_ttt_read_harm_veto_max_stable_positive),
        )
        self.semantic_role_swa_negative_scale = float(semantic_role_swa_negative_scale)
        self.semantic_role_swa_protect_scale = float(semantic_role_swa_protect_scale)
        self.semantic_role_control_mode = str(semantic_role_control_mode or "none").strip().lower()
        self.semantic_role_control_seed = int(semantic_role_control_seed)
        self.semantic_swa_redirection_phase3_root = str(semantic_swa_redirection_phase3_root or "")
        self.semantic_swa_redirection_stable_case = str(semantic_swa_redirection_stable_case or "")
        self.semantic_swa_redirection_random_stable_case = str(semantic_swa_redirection_random_stable_case or "")
        self.semantic_swa_redirection_layer = int(semantic_swa_redirection_layer)
        self.semantic_swa_redirection_top_quantile = float(semantic_swa_redirection_top_quantile)
        self.semantic_swa_redirection_overlap_frames = max(int(semantic_swa_redirection_overlap_frames), 0)
        self.semantic_swa_redirection_random_control = bool(semantic_swa_redirection_random_control)
        self.semantic_anchor_mode = str(semantic_anchor_mode or "semantic").strip().lower()
        self.semantic_anchor_target_ratio = float(semantic_anchor_target_ratio)
        self.semantic_anchor_min_ratio = float(semantic_anchor_min_ratio)
        self.semantic_anchor_max_ratio = float(semantic_anchor_max_ratio)
        self.semantic_anchor_min_score = float(semantic_anchor_min_score)
        self.semantic_anchor_grid_rows = max(int(semantic_anchor_grid_rows), 1)
        self.semantic_anchor_grid_cols = max(int(semantic_anchor_grid_cols), 1)
        self.semantic_anchor_missing_trust_policy = str(semantic_anchor_missing_trust_policy or "zero").strip().lower()
        self.semantic_anchor_value_fallback = str(semantic_anchor_value_fallback or "off").strip().lower()
        self.enable_semantic_anchor_ttt_floor = bool(enable_semantic_anchor_ttt_floor)
        self.semantic_condition_conflict_level = "unavailable"
        self.semantic_condition_conflict_source = "none"
        self.semantic_condition_conflict_value = 0.0
        self.semantic_condition_conflict_summary: Dict[str, Any] = {}
        self.semantic_condition_conflict_tok: Optional[torch.Tensor] = None
        self.semantic_condition_conflict_token_exact = False
        self.semantic_condition_scale_level = "unavailable"
        self.semantic_condition_scale_source = "none"
        self.semantic_condition_scale_value = 0.0
        self.semantic_condition_scale_summary: Dict[str, Any] = {}
        self.semantic_condition_scale_tok: Optional[torch.Tensor] = None
        self.semantic_condition_scale_token_exact = False
        self.rho_ttt_apply = float(rho_ttt_apply)
        self.read_layer_mode = str(read_layer_mode)
        self.read_single_layer = int(read_single_layer)
        self.read_cue_source = str(read_cue_source)
        self.v70_radio_sidecar_dir = str(v70_radio_sidecar_dir or "")
        self.enable_v70_radio_ttt_write_prior = bool(enable_v70_radio_ttt_write_prior)
        self.v70_radio_ttt_mode = str(v70_radio_ttt_mode or "static_object_write_floor")
        self.v70_radio_ttt_control = str(v70_radio_ttt_control or "none")
        self.v70_radio_ttt_static_boost = float(v70_radio_ttt_static_boost)
        self.v70_radio_ttt_suppress = float(v70_radio_ttt_suppress)
        self.v70_radio_ttt_min_confidence = float(v70_radio_ttt_min_confidence)
        self.v70_radio_ttt_min_stability = float(v70_radio_ttt_min_stability)
        self.v70_radio_ttt_min_interior = float(v70_radio_ttt_min_interior)
        self.v70_radio_ttt_max_activity_risk = float(v70_radio_ttt_max_activity_risk)
        self.v70_radio_ttt_boundary_threshold = float(v70_radio_ttt_boundary_threshold)
        self.v70_radio_ttt_renorm_mean = bool(v70_radio_ttt_renorm_mean)
        self.semantic_ttt_overlap_support_dir = str(semantic_ttt_overlap_support_dir or "")
        self.semantic_ttt_overlap_support_kind = str(semantic_ttt_overlap_support_kind or "source_gate")
        self.semantic_ttt_overlap_support_score_key = str(semantic_ttt_overlap_support_score_key or "score_overlap")
        self.semantic_ttt_overlap_support_scope = str(semantic_ttt_overlap_support_scope or "head_overlap")
        self.semantic_ttt_overlap_support_floor = float(semantic_ttt_overlap_support_floor)
        self.semantic_ttt_overlap_support_renorm_mean = bool(semantic_ttt_overlap_support_renorm_mean)
        self.read_topk_frac = float(read_topk_frac)
        self.frame_bias_mode = str(frame_bias_mode)
        self.frame_bias_query_region = str(frame_bias_query_region or "all")
        self.frame_bias_head_indices = str(frame_bias_head_indices or "")
        self.frame_attention_record_bias_mass = bool(frame_attention_record_bias_mass)
        self.frame_attention_bias_mass_max_queries = int(frame_attention_bias_mass_max_queries)
        self.chunk_bias_mode = str(chunk_bias_mode)
        self.swa_bias_mode = str(swa_bias_mode)
        self.flow_model = str(flow_model)
        self.flow_pair_stride = max(int(flow_pair_stride), 1)
        self.flow_fb_thr = float(flow_fb_thr)
        self.flow_residual_thr = float(flow_residual_thr)
        self.gram_layer_groups = str(gram_layer_groups)
        self.read_calib_mode = str(read_calib_mode)
        self.read_target_mass = float(read_target_mass)
        self.read_calib_tau = float(read_calib_tau)
        self.read_blend_lambda = float(read_blend_lambda)
        self.read_quality_mass_min = float(read_quality_mass_min)
        self.read_quality_mass_max = float(read_quality_mass_max)
        self.read_quality_anchor_max = float(read_quality_anchor_max)
        self.read_quality_frag_max = float(read_quality_frag_max)
        self.beta_policy = str(beta_policy)
        self.beta_energy_target = float(beta_energy_target)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.read_protect_ref = bool(read_protect_ref)
        self.read_protect_static = bool(read_protect_static)
        self.read_protection_mode = str(read_protection_mode)
        self.read_ref_strength = float(read_ref_strength)
        self.read_overlap_frames = max(int(read_overlap_frames), 0)
        self.read_reset_frames = max(int(read_reset_frames), 0)
        self.read_attention_q = float(read_attention_q)
        self.read_static_anchor_thr = float(read_static_anchor_thr)
        self.read_static_dyn_thr = float(read_static_dyn_thr)
        self.hmc_write_score_source = str(hmc_write_score_source)
        self.hmc_write_sparse_ratio = float(hmc_write_sparse_ratio)
        self.hmc_write_sparse_mode = str(hmc_write_sparse_mode)
        self.hmc_write_alpha = float(hmc_write_alpha)
        self.hmc_write_min = float(hmc_write_min)
        self.hmc_write_max = float(hmc_write_max)
        self.ttt_apply_min_gate = float(ttt_apply_min_gate)
        self.enable_swa_write_control = bool(enable_swa_write_control)
        self.swa_write_mode = str(swa_write_mode)
        self.swa_write_rho = float(swa_write_rho)
        self.swa_write_min_gate = float(swa_write_min_gate)
        self.swa_write_sparse_ratio = float(swa_write_sparse_ratio)
        self.swa_write_layer_mode = str(swa_write_layer_mode)
        self.swa_write_single_layer = int(swa_write_single_layer)
        self.swa_write_scope = str(swa_write_scope)
        self.swa_write_keep_scope = str(swa_write_keep_scope)
        self.swa_write_score_source = str(swa_write_score_source)
        self.swa_write_cache_blend_alpha = float(swa_write_cache_blend_alpha)
        self.swa_write_cache_blend_mode = str(swa_write_cache_blend_mode)
        self.swa_write_cache_blend_target = str(swa_write_cache_blend_target)
        self.ttt_write_token_scope = str(ttt_write_token_scope)
        self.ttt_write_token_scope_floor = float(ttt_write_token_scope_floor)
        self.semantic_action_active_chunks = _parse_chunk_index_set(semantic_action_active_chunks)
        self.semantic_action_inactive_read_cue_source = str(semantic_action_inactive_read_cue_source or "")
        self.v32_semantic_cue_active_chunks = _parse_chunk_index_set(v32_semantic_cue_active_chunks)
        self.v32_semantic_cue_trigger_mode = str(v32_semantic_cue_trigger_mode or "off")
        self.v32_semantic_cue_trigger_high_threshold = float(v32_semantic_cue_trigger_high_threshold)
        self.v32_semantic_cue_trigger_k = float(v32_semantic_cue_trigger_k)
        self.v32_semantic_cue_trigger_min_mass = float(v32_semantic_cue_trigger_min_mass)
        self.v32_semantic_cue_trigger_warmup = max(int(v32_semantic_cue_trigger_warmup), 0)
        self.v32_semantic_cue_trigger_ema_alpha = float(v32_semantic_cue_trigger_ema_alpha)
        self._v32_semantic_cue_ema: Optional[float] = None
        self._v32_semantic_cue_mad: Optional[float] = None
        self._v32_semantic_cue_count = 0
        self.read_cue_patch_dump_dir = str(read_cue_patch_dump_dir or "")
        self.read_cue_patch_dump_dtype = str(read_cue_patch_dump_dtype or "float16")
        self.fast_cue_eval = bool(fast_cue_eval)
        self.ttt_update_controller = TTTWriteController(
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            device=device,
            write_mode="semantic",
            eta_mean_preserve=eta_mean_preserve,
            eta_norm_eps=eta_norm_eps,
            prior_branch_mask=prior_branch_mask,
            prior_layer_mode=prior_layer_mode,
            prior_single_layer=prior_single_layer,
            prior_layer_branch_policy=prior_layer_branch_policy,
            update_delta_scale=ttt_write_delta_scale,
            update_delta_scales=ttt_write_delta_scales,
            update_native_mix_scales=ttt_write_native_mix_scales,
            update_native_mix_chunks=ttt_write_native_mix_chunks,
            prior_transform_mode=ttt_write_prior_transform_mode,
            prior_anti_scale=ttt_write_prior_anti_scale,
            prior_gamma=ttt_write_prior_gamma,
            special_token_policy=ttt_write_special_token_policy,
            special_token_floor=ttt_write_special_token_floor,
            special_token_ceiling=ttt_write_special_token_ceiling,
            gradient_reversal_mode=ttt_write_gradient_reversal_mode,
            gradient_reversal_gamma=ttt_write_gradient_reversal_gamma,
            gradient_reversal_branch_mask=ttt_write_gradient_reversal_branch_mask,
            gradient_reversal_branch_gammas=ttt_write_gradient_reversal_branch_gammas,
            gradient_reversal_layer_gammas=ttt_write_gradient_reversal_layer_gammas,
            gradient_reversal_head_routes=ttt_write_gradient_reversal_head_routes,
            gradient_reversal_negative_frac=ttt_write_gradient_reversal_negative_frac,
            gradient_reversal_risk_source=ttt_write_gradient_reversal_risk_source,
            tri_replay_positive_frac=ttt_write_tri_replay_positive_frac,
            tri_replay_negative_frac=ttt_write_tri_replay_negative_frac,
            tri_replay_neutral_lambda=ttt_write_tri_replay_neutral_lambda,
            tri_replay_role_mode=ttt_write_tri_replay_role_mode,
            gradient_reversal_transient_mode=ttt_write_gradient_reversal_transient_mode,
            gradient_reversal_transient_branch_mask=ttt_write_gradient_reversal_transient_branch_mask,
            gradient_reversal_transient_long_scale=ttt_write_gradient_reversal_transient_long_scale,
            gradient_reversal_transient_apply_scale=ttt_write_gradient_reversal_transient_apply_scale,
            update_token_scope=ttt_write_token_scope,
            update_token_scope_floor=ttt_write_token_scope_floor,
            replay_feature_gate_mode=ttt_write_replay_feature_gate_mode,
            replay_feature_gate_rho=ttt_write_replay_feature_gate_rho,
            replay_feature_gate_min=ttt_write_replay_feature_gate_min,
            replay_feature_gate_branch_mask=ttt_write_replay_feature_gate_branch_mask,
            replay_token_filter_mode=ttt_write_replay_token_filter_mode,
            replay_token_filter_ratio=ttt_write_replay_token_filter_ratio,
            replay_token_filter_threshold=ttt_write_replay_token_filter_threshold,
            replay_token_filter_scope=ttt_write_replay_token_filter_scope,
            replay_token_filter_branch_mask=ttt_write_replay_token_filter_branch_mask,
            replay_token_filter_blend=ttt_write_replay_token_filter_blend,
            replay_token_filter_blend_mode=ttt_write_replay_token_filter_blend_mode,
            transient_delta_subtract_scale=ttt_write_transient_delta_subtract_scale,
            transient_delta_branch_mask=ttt_write_transient_delta_branch_mask,
            transient_delta_ttl=ttt_write_transient_delta_ttl,
            commit_ema_alpha=ttt_write_commit_ema_alpha,
            commit_ema_branch_mask=ttt_write_commit_ema_branch_mask,
            commit_ema_chunks=ttt_write_commit_ema_chunks,
            native_delta_gate_mode=ttt_write_native_delta_gate_mode,
            native_delta_gate_min_cos=ttt_write_native_delta_gate_min_cos,
            native_delta_gate_fallback=ttt_write_native_delta_gate_fallback,
            native_delta_gate_cap_ratio=ttt_write_native_delta_gate_cap_ratio,
            native_delta_gate_branch_mask=ttt_write_native_delta_gate_branch_mask,
            commit_filter_mode=ttt_write_commit_filter_mode,
            commit_filter_risk_source=ttt_write_commit_filter_risk_source,
            commit_filter_scope=ttt_write_commit_filter_scope,
            commit_filter_stat=ttt_write_commit_filter_stat,
            commit_filter_base=ttt_write_commit_filter_base,
            commit_filter_gain=ttt_write_commit_filter_gain,
            commit_filter_min=ttt_write_commit_filter_min,
            commit_filter_max=ttt_write_commit_filter_max,
            commit_filter_branch_mask=ttt_write_commit_filter_branch_mask,
            scale_state_mode=ttt_write_scale_state_mode,
            scale_state_proxy=ttt_write_scale_state_proxy,
            scale_state_carrier=ttt_write_scale_state_carrier,
            scale_state_alpha=ttt_write_scale_state_alpha,
            scale_state_branch_mask=ttt_write_scale_state_branch_mask,
            scale_state_chunks=ttt_write_scale_state_chunks,
            scale_state_sample_tokens=ttt_write_scale_state_sample_tokens,
        )

    def run_probe(
        self,
        backbone: LoGeRGeometryBackbone,
        images: torch.Tensor,
        state_m: Optional[HybridMemoryState],
        *,
        collect_hybrid_trace: bool = True,
    ) -> ProbeOutput:
        committed_hash = hybrid_state_fingerprint(state_m)
        geo, write_cache = backbone.run(
            images,
            # Probe/write-cache collection must stay on the long-lived TTT state.
            # Dual-lifetime short deltas are apply-time evidence for the
            # controlled pass; letting them enter probe would contaminate the
            # long commit primitives we are trying to keep separate.
            ttt_state=state_m.ttt_state if state_m is not None else None,
            cache_ttt_primitives=True,
            hmc_control=self._build_model_hmc_control(None, mode="probe", identity_hooks=True),
        )
        token_meta = _build_token_meta(geo)
        provisional_ttt = self._state_from_write_cache(write_cache)
        provisional = HybridMemoryState.from_ttt_state(provisional_ttt)
        cache = HybridMemoryCacheOutput(
            token_meta=token_meta,
            ttt_cache=write_cache,
            state_refs={
                "committed_state_hash": committed_hash,
                "native_provisional_state_hash": hybrid_state_fingerprint(provisional),
            },
            update_meta={
                "collect_hybrid_trace": bool(collect_hybrid_trace),
                "implemented_trace_paths": [
                    "frame_attn_hook",
                    "swa_read_hook",
                    "ttt_apply_hook",
                    "chunk_attn_hook",
                    "ttt_cache",
                ],
                "missing_trace_paths": ["swa_dense_importance_map", "chunk_dense_dynamic_mass_map"],
            },
        )
        trace_dict = geo.hmc_trace or {}
        cache.frame_attn_cache = {"records": trace_dict.get("frame_attention", [])}
        cache.swa_cache = {"records": trace_dict.get("swa_read", [])}
        cache.chunk_attn_cache = {"records": trace_dict.get("chunk_attention", [])}
        trace = self._build_probe_trace(geo, write_cache)
        return ProbeOutput(
            geometry=geo,
            token_meta=token_meta,
            hybrid_cache=cache,
            probe_trace=trace,
            native_provisional_state=provisional,
            debug={
                "mode": "probe",
                "committed_state_hash": committed_hash,
                "native_provisional_state_hash": hybrid_state_fingerprint(provisional),
                "ttt_cache_layers": int(write_cache.num_ttt_layers),
                "token_count": int(geo.token_type.numel()),
                "read_path_traces": "real_identity_hook_sites",
                "hmc_hook_trace_counts": {
                    "frame_attention": len(trace_dict.get("frame_attention", [])),
                    "swa_read": len(trace_dict.get("swa_read", [])),
                    "ttt_apply": len(trace_dict.get("ttt_apply", [])),
                    "chunk_attention": len(trace_dict.get("chunk_attention", [])),
                },
            },
        )

    def _build_phase_e_protection_patch(
        self,
        *,
        mode: str,
        num_patch: int,
        num_frames: int,
        safe_patch: torch.Tensor,
        key_avg_patch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        mode = str(mode or "none")
        zero = torch.zeros(num_patch, dtype=torch.bool)
        if num_patch <= 0:
            return zero.float(), {"read_protection_mode": mode}

        overlap_mask = _frame_patch_mask(
            num_patch,
            num_frames,
            first=self.read_overlap_frames,
            last=self.read_overlap_frames,
        )
        reset_mask = _frame_patch_mask(num_patch, num_frames, first=self.read_reset_frames, last=0)
        anchor_mask = safe_patch.detach().cpu().bool().reshape(-1)
        attn_mask = _high_quantile_mask(key_avg_patch, self.read_attention_q)

        if mode in {"none", "off"}:
            protect = zero
        elif mode in {"static", "anchor", "high_anchor"}:
            protect = anchor_mask
        elif mode == "overlap":
            protect = overlap_mask
        elif mode in {"reset", "ref"}:
            protect = reset_mask
        elif mode in {"attention", "attn"}:
            protect = attn_mask
        elif mode == "combined_light":
            protect = overlap_mask | anchor_mask | attn_mask
        elif mode == "combined_strong":
            protect = overlap_mask | reset_mask | anchor_mask | attn_mask
        else:
            protect = zero

        debug = {
            "read_protection_mode": mode,
            "read_ref_strength": float(self.read_ref_strength),
            "read_overlap_frames": int(self.read_overlap_frames),
            "read_reset_frames": int(self.read_reset_frames),
            "read_attention_q": float(self.read_attention_q),
            "protect_overlap_count": int(overlap_mask.sum().item()),
            "protect_reset_count": int(reset_mask.sum().item()),
            "protect_anchor_count": int(anchor_mask.sum().item()),
            "protect_attention_count": int(attn_mask.sum().item()),
            "protect_patch_count": int(protect.sum().item()),
            "protect_patch_mass": float(protect.float().mean().item()) if protect.numel() else 0.0,
        }
        return protect.float(), debug

    def _ttt_residual_token_score(self, probe: ProbeOutput, L_tok: int) -> torch.Tensor:
        write_cache = probe.hybrid_cache.ttt_cache
        if write_cache is None or not write_cache.layer_caches:
            return torch.zeros(L_tok, dtype=torch.float32)

        accum = torch.zeros(L_tok, dtype=torch.float32)
        count = torch.zeros(L_tok, dtype=torch.float32)
        for layer_cache in write_cache.layer_caches:
            y = getattr(layer_cache, "apply_output_raw", None)
            v = getattr(layer_cache, "v", None)
            if y is None or v is None:
                continue
            y_cpu = y.detach().cpu().float()
            v_cpu = v.detach().cpu().float()
            if y_cpu.shape != v_cpu.shape or y_cpu.ndim < 2:
                continue
            res = (y_cpu - v_cpu).norm(dim=-1) / v_cpu.norm(dim=-1).clamp_min(1e-6)
            if res.ndim == 2:
                per_tok = res.mean(dim=0)
            else:
                per_tok = res.reshape(-1, res.shape[-1]).mean(dim=0)
            n = min(int(per_tok.numel()), L_tok)
            if n <= 0:
                continue
            accum[:n] += per_tok[:n]
            count[:n] += 1.0
        return torch.where(count > 0, accum / count.clamp_min(1.0), torch.zeros_like(accum))

    def _read_cue_dump_dtype(self) -> torch.dtype:
        name = str(self.read_cue_patch_dump_dtype or "float16").strip().lower()
        if name in {"fp32", "float32"}:
            return torch.float32
        if name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        return torch.float16

    @staticmethod
    def _patch_vector_to_grid_or_flat(
        values: Optional[torch.Tensor],
        *,
        num_frames: int,
        patch_grid: Tuple[int, int],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if values is None:
            return None
        x = values.detach().cpu().float().reshape(-1)
        t = max(int(num_frames), 1)
        h = max(int(patch_grid[0]), 1)
        w = max(int(patch_grid[1]), 1)
        if int(x.numel()) == t * h * w:
            x = x.reshape(t, h, w)
        return x.to(dtype=dtype)

    @staticmethod
    def _read_active_q90_mask(read_patch: torch.Tensor, *, num_frames: int, patch_grid: Tuple[int, int]) -> torch.Tensor:
        x = read_patch.detach().cpu().float().reshape(-1)
        t = max(int(num_frames), 1)
        h = max(int(patch_grid[0]), 1)
        w = max(int(patch_grid[1]), 1)
        if int(x.numel()) != t * h * w:
            if x.numel() <= 0:
                return torch.zeros((0,), dtype=torch.bool)
            k = max(1, int(math.ceil(0.10 * int(x.numel()))))
            idx = torch.topk(x, k=min(k, int(x.numel())), largest=True).indices
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[idx] = True
            return mask
        grid = x.reshape(t, h * w)
        k = max(1, int(math.ceil(0.10 * h * w)))
        idx = torch.topk(grid, k=min(k, h * w), dim=1, largest=True).indices
        mask = torch.zeros_like(grid, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask.reshape(t, h, w)

    @staticmethod
    def _dump_stats(values: torch.Tensor) -> Dict[str, Any]:
        x = values.detach().cpu().float().reshape(-1)
        x = x[torch.isfinite(x)]
        if int(x.numel()) <= 0:
            return {"mean": None, "q90": None, "max": None, "finite_count": 0}
        return {
            "mean": float(x.mean().item()),
            "q90": float(torch.quantile(x, 0.90).item()),
            "max": float(x.max().item()),
            "finite_count": int(x.numel()),
        }

    def _dump_read_cue_patch(
        self,
        *,
        read_patch: torch.Tensor,
        num_frames: int,
        patch_grid: Tuple[int, int],
        chunk_idx: int,
        cue_source_effective: str,
        patch_debug: Dict[str, Any],
        extra_patches: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, Any]:
        dump_dir = str(self.read_cue_patch_dump_dir or "")
        if not dump_dir:
            return {"read_cue_patch_dump_status": "disabled"}
        try:
            dtype = self._read_cue_dump_dtype()
            out_dir = Path(dump_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            read_grid = self._patch_vector_to_grid_or_flat(
                read_patch,
                num_frames=num_frames,
                patch_grid=patch_grid,
                dtype=dtype,
            )
            if read_grid is None:
                return {"read_cue_patch_dump_status": "skipped_no_read_patch"}
            q90_mask = self._read_active_q90_mask(read_patch, num_frames=num_frames, patch_grid=patch_grid)
            gt050_mask = (read_grid.float() > 0.50)
            tensors: Dict[str, Any] = {
                "read_patch_final": read_grid,
                "read_active_gt050_patch": gt050_mask.cpu(),
                "read_active_q90_patch": q90_mask.cpu(),
            }
            for key, value in sorted(extra_patches.items()):
                tensor = self._patch_vector_to_grid_or_flat(
                    value,
                    num_frames=num_frames,
                    patch_grid=patch_grid,
                    dtype=dtype,
                )
                if tensor is not None:
                    tensors[key] = tensor
            payload = {
                "schema": "acl2_v79_read_cue_patch_dump_v1",
                "chunk_idx": int(chunk_idx),
                "num_frames": int(num_frames),
                "patch_grid": [int(patch_grid[0]), int(patch_grid[1])],
                "read_cue_source": str(self.read_cue_source),
                "cue_source_effective": str(cue_source_effective),
                "read_calib_mode": str(self.read_calib_mode),
                "read_target_mass": float(self.read_target_mass),
                "read_calib_tau": float(self.read_calib_tau),
                "read_blend_lambda": float(self.read_blend_lambda),
                "read_topk_frac": float(self.read_topk_frac),
                "dtype": str(dtype).replace("torch.", ""),
                "debug": dict(patch_debug),
                "stats": {
                    "read_patch_final": self._dump_stats(read_grid.float()),
                    "read_active_gt050_mass": float(gt050_mask.float().mean().item()) if gt050_mask.numel() else 0.0,
                    "read_active_q90_mass": float(q90_mask.float().mean().item()) if q90_mask.numel() else 0.0,
                },
                "tensors": tensors,
            }
            out_path = out_dir / f"chunk_{int(chunk_idx):03d}_read_cue_patch.pt"
            torch.save(payload, out_path)
            return {
                "read_cue_patch_dump_status": "saved",
                "read_cue_patch_dump_path": str(out_path),
                "read_cue_patch_dump_read_mean": payload["stats"]["read_patch_final"]["mean"],
                "read_cue_patch_dump_q90_mass": payload["stats"]["read_active_q90_mass"],
            }
        except Exception as exc:
            return {
                "read_cue_patch_dump_status": "failed",
                "read_cue_patch_dump_error": f"{type(exc).__name__}:{exc}",
            }

    def _build_semantic_anchor_bank(
        self,
        *,
        token_type: torch.Tensor,
        D_tok: torch.Tensor,
        P_ttt_write: Optional[torch.Tensor],
        G_sem_tok: Optional[torch.Tensor],
        Q_sem_tok: Optional[torch.Tensor],
        L_sem_tok: Optional[torch.Tensor],
        S_tok: Optional[torch.Tensor],
        V_sem_tok: Optional[torch.Tensor],
        conf_patch: Optional[torch.Tensor],
        key_avg_patch: Optional[torch.Tensor],
        num_frames: int,
        patch_grid: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        token_type_cpu = token_type.detach().cpu().long().reshape(-1)
        patch_mask = token_type_cpu == TOKEN_TYPE_PATCH
        L_tok = int(token_type_cpu.numel())
        n_patch = int(patch_mask.sum().item())
        A_tok = torch.zeros((L_tok,), dtype=torch.float32)
        M_tok = torch.zeros((L_tok,), dtype=torch.float32)
        mode = str(self.semantic_anchor_mode or "none").strip().lower()
        debug: Dict[str, Any] = {
            "semantic_anchor_mode": mode,
            "semantic_anchor_bank_available": False,
            "semantic_anchor_token_count": 0,
            "semantic_anchor_token_ratio": 0.0,
            "semantic_anchor_patch_tokens": int(n_patch),
        }
        if mode in {"", "none", "off", "noop"}:
            debug["semantic_anchor_reason"] = "disabled"
            return A_tok, M_tok, debug
        if n_patch <= 0 or D_tok is None:
            debug["semantic_anchor_reason"] = "missing_patch_or_D"
            return A_tok, M_tok, debug

        D = D_tok.detach().cpu().float().reshape(-1)
        if int(D.numel()) != L_tok:
            debug["semantic_anchor_reason"] = "D_shape_mismatch"
            return A_tok, M_tok, debug
        D_patch = D[patch_mask].clamp(0.0, 1.0)

        label_patch = _patch_labels_from_token_labels(L_sem_tok, token_type_cpu)
        group_patch = _patch_labels_from_token_labels(G_sem_tok, token_type_cpu)
        trust_patch = None
        if Q_sem_tok is not None:
            q = Q_sem_tok.detach().cpu().float().reshape(-1)
            if int(q.numel()) == L_tok:
                trust_patch = q[patch_mask].clamp(0.0, 1.0)
        sem_score, sem_debug = _stable_semantic_score(
            label_patch,
            group_patch,
            trust_patch,
            n_patch,
            missing_trust_policy=self.semantic_anchor_missing_trust_policy,
        )
        debug.update(sem_debug)
        value_fallback_used = False
        value_patch = None
        if V_sem_tok is not None:
            v = V_sem_tok.detach().cpu().float().reshape(-1)
            if int(v.numel()) == L_tok:
                value_patch = v[patch_mask].clamp(0.0, 1.0)
        if value_patch is None and S_tok is not None:
            s = S_tok.detach().cpu().float().reshape(-1)
            if int(s.numel()) == L_tok:
                value_patch = s[patch_mask].clamp(0.0, 1.0)
        if (
            str(self.semantic_anchor_value_fallback).lower() in {"semantic_value", "v_sem", "value"}
            and value_patch is not None
            and int(value_patch.numel()) == n_patch
            and (not bool((sem_score > 0.0).any()))
        ):
            sem_score = value_patch.clamp(0.0, 1.0)
            value_fallback_used = True
        debug.update({
            "semantic_anchor_value_fallback": self.semantic_anchor_value_fallback,
            "semantic_anchor_value_fallback_used": bool(value_fallback_used),
            "semantic_anchor_value_fallback_available": value_patch is not None,
            "semantic_anchor_sem_score_max": float(sem_score.max().item()) if sem_score.numel() else 0.0,
            "semantic_anchor_sem_score_mean_all": float(sem_score.mean().item()) if sem_score.numel() else 0.0,
        })

        stage_missing = P_ttt_write is None or int(P_ttt_write.numel()) != L_tok
        if stage_missing:
            stage_patch = torch.ones((n_patch,), dtype=torch.float32)
        else:
            stage_patch = P_ttt_write.detach().cpu().float().reshape(-1)[patch_mask].clamp(0.0, 2.0)
        stage_score = 0.25 + 0.75 * _normalize01(stage_patch)

        if conf_patch is None or int(conf_patch.numel()) < n_patch:
            conf_score = torch.ones((n_patch,), dtype=torch.float32)
            conf_missing = True
        else:
            conf_score = conf_patch.detach().cpu().float().reshape(-1)[:n_patch].clamp(0.0, 1.0)
            conf_missing = False
        if key_avg_patch is None or int(key_avg_patch.numel()) < n_patch:
            source_score = torch.ones((n_patch,), dtype=torch.float32)
            source_missing = True
        else:
            source_score = 0.25 + 0.75 * _normalize01(key_avg_patch.detach().cpu().float().reshape(-1)[:n_patch])
            source_missing = False

        geo_score = (torch.sqrt((1.0 - D_patch).clamp(0.0, 1.0)) * stage_score * conf_score).clamp(0.0, 1.0)
        raw_score = (sem_score * geo_score * source_score).clamp(0.0, 1.0)
        debug.update({
            "semantic_anchor_raw_score_max": float(raw_score.max().item()) if raw_score.numel() else 0.0,
            "semantic_anchor_raw_score_mean_all": float(raw_score.mean().item()) if raw_score.numel() else 0.0,
            "semantic_anchor_geo_score_max": float(geo_score.max().item()) if geo_score.numel() else 0.0,
            "semantic_anchor_geo_score_mean_all": float(geo_score.mean().item()) if geo_score.numel() else 0.0,
            "semantic_anchor_source_score_max": float(source_score.max().item()) if source_score.numel() else 0.0,
            "semantic_anchor_source_score_mean_all": float(source_score.mean().item()) if source_score.numel() else 0.0,
        })
        base_mask, select_debug = _select_spatially_diverse_anchors(
            raw_score,
            num_frames=num_frames,
            patch_grid=patch_grid,
            target_ratio=self.semantic_anchor_target_ratio,
            min_ratio=self.semantic_anchor_min_ratio,
            max_ratio=self.semantic_anchor_max_ratio,
            min_score=self.semantic_anchor_min_score,
            grid_rows=self.semantic_anchor_grid_rows,
            grid_cols=self.semantic_anchor_grid_cols,
        )
        score_for_selection = raw_score
        selected = base_mask
        if mode in {"random", "random_same_mass", "random_same_mass_anchor"}:
            k = int(base_mask.sum().item())
            eligible = torch.nonzero(raw_score > float(self.semantic_anchor_min_score), as_tuple=False).reshape(-1)
            selected = torch.zeros_like(base_mask)
            if k > 0 and int(eligible.numel()) > 0:
                noise = _deterministic_patch_noise(n_patch, seed=1729.0 + float(getattr(self, "current_chunk_idx", 0)))
                chosen = eligible[torch.argsort(noise[eligible], descending=True)[: min(k, int(eligible.numel()))]]
                selected[chosen] = True
                score_for_selection = torch.where(selected, raw_score[base_mask].mean() if bool(base_mask.any()) else raw_score, raw_score * 0.0)
            select_debug["semantic_anchor_control"] = "random_same_mass"
        elif mode in {"shuffled", "shuffled_semantic", "shuffled_semantic_label"}:
            noise = _deterministic_patch_noise(n_patch, seed=31337.0 + float(getattr(self, "current_chunk_idx", 0)))
            perm = torch.argsort(noise, descending=True)
            shuffled_raw = (sem_score[perm] * geo_score * source_score).clamp(0.0, 1.0)
            selected, select_debug = _select_spatially_diverse_anchors(
                shuffled_raw,
                num_frames=num_frames,
                patch_grid=patch_grid,
                target_ratio=self.semantic_anchor_target_ratio,
                min_ratio=self.semantic_anchor_min_ratio,
                max_ratio=self.semantic_anchor_max_ratio,
                min_score=self.semantic_anchor_min_score,
                grid_rows=self.semantic_anchor_grid_rows,
                grid_cols=self.semantic_anchor_grid_cols,
            )
            score_for_selection = shuffled_raw
            select_debug["semantic_anchor_control"] = "shuffled_semantic_label"

        anchor_score = torch.zeros_like(raw_score)
        if bool(selected.any()):
            selected_score = score_for_selection[selected].clamp_min(self.semantic_anchor_min_score)
            max_score = selected_score.max().clamp_min(1e-6)
            anchor_score[selected] = (selected_score / max_score).clamp(0.0, 1.0)

        A_tok[patch_mask] = anchor_score
        M_tok[patch_mask] = selected.float()
        anchor_count = int(selected.sum().item())
        static_count = 0
        dynamic_count = 0
        if group_patch is not None and int(group_patch.numel()) == n_patch and anchor_count > 0:
            G = group_patch.detach().cpu().long().reshape(-1)
            static_mask = (
                (G == int(SEMANTIC_GROUP_STRUCTURE_ANCHOR))
                | (G == int(SEMANTIC_GROUP_STATIC_THING))
            )
            dynamic_mask = G == int(SEMANTIC_GROUP_MOVABLE_THING)
            static_count = int((selected & static_mask).sum().item())
            dynamic_count = int((selected & dynamic_mask).sum().item())
        debug.update(select_debug)
        debug.update({
            "semantic_anchor_bank_available": bool(anchor_count > 0),
            "semantic_anchor_reason": select_debug.get("semantic_anchor_select_reason"),
            "semantic_anchor_token_count": anchor_count,
            "semantic_anchor_token_ratio": float(anchor_count / max(n_patch, 1)),
            "semantic_anchor_score_mean": float(anchor_score[selected].mean().item()) if anchor_count > 0 else 0.0,
            "semantic_anchor_raw_score_mean": float(raw_score[selected].mean().item()) if anchor_count > 0 else 0.0,
            "semantic_anchor_static_semantic_ratio": float(static_count / max(anchor_count, 1)),
            "semantic_anchor_dynamic_semantic_ratio": float(dynamic_count / max(anchor_count, 1)),
            "semantic_anchor_stage_proxy_missing": bool(stage_missing),
            "semantic_anchor_conf_missing": bool(conf_missing),
            "semantic_anchor_source_attention_proxy_missing": bool(source_missing),
            "semantic_anchor_geo_score_mean": float(geo_score[selected].mean().item()) if anchor_count > 0 else 0.0,
            "semantic_anchor_sem_score_mean": float(sem_score[selected].mean().item()) if anchor_count > 0 else 0.0,
            "semantic_anchor_source_score_mean": float(source_score[selected].mean().item()) if anchor_count > 0 else 0.0,
            "semantic_anchor_missing_trust_policy": self.semantic_anchor_missing_trust_policy,
            "semantic_anchor_value_fallback": self.semantic_anchor_value_fallback,
            "semantic_anchor_value_fallback_used": bool(value_fallback_used),
        })
        return A_tok.clamp(0.0, 1.0), M_tok.clamp(0.0, 1.0), debug

    def _apply_semantic_anchor_write_floor(
        self,
        *,
        token_type: torch.Tensor,
        P_ttt_write: Optional[torch.Tensor],
        A_anchor_tok: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        debug: Dict[str, Any] = {
            "semantic_anchor_write_floor_enabled": bool(self.enable_semantic_anchor_ttt_floor),
            "semantic_anchor_write_floor_applied": False,
        }
        if not self.enable_semantic_anchor_ttt_floor:
            return P_ttt_write, debug
        if P_ttt_write is None or A_anchor_tok is None:
            debug["semantic_anchor_write_floor_reason"] = "missing_write_prior_or_anchor"
            return P_ttt_write, debug
        P = P_ttt_write.detach().cpu().float().reshape(-1).clone()
        A = A_anchor_tok.detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
        token_type_cpu = token_type.detach().cpu().long().reshape(-1)
        if int(P.numel()) != int(A.numel()) or int(P.numel()) != int(token_type_cpu.numel()):
            debug["semantic_anchor_write_floor_reason"] = "shape_mismatch"
            return P_ttt_write, debug
        patch_mask = token_type_cpu == TOKEN_TYPE_PATCH
        anchor = (A > 0.0) & patch_mask
        non_anchor = (~anchor) & patch_mask
        if not bool(anchor.any()):
            debug["semantic_anchor_write_floor_reason"] = "empty_anchor"
            return P_ttt_write, debug
        mean_w = float(P[patch_mask].mean().item()) if bool(patch_mask.any()) else 1.0
        mean_a = float(A[anchor].mean().item())
        lam = min(max(mean_w / max(mean_a, 1e-6), 0.2), 1.0)
        floor = (float(lam) * A).clamp(0.0, 1.0)
        P_new = torch.maximum(P, floor)
        changed = (P_new - P).abs() > 1e-8
        repair_mode = "scalar_anchor_floor"
        if not bool(changed[anchor].any()):
            hard_floor = mean_w * (1.0 + 0.05 * A)
            hard_floor = hard_floor.clamp_min(0.0)
            P_new = torch.where(anchor, torch.maximum(P, hard_floor), P)
            changed = (P_new - P).abs() > 1e-8
            repair_mode = "hard_eligibility_floor_after_scalar_noop"
        debug.update({
            "semantic_anchor_write_floor_applied": bool(changed[anchor].any()),
            "semantic_anchor_write_floor_reason": "ok",
            "semantic_anchor_write_floor_lambda": float(lam),
            "semantic_anchor_write_floor_repair_mode": repair_mode,
            "anchor_write_floor_applied_count": int((changed & anchor).sum().item()),
            "anchor_write_mass_before": float(P[anchor].mean().item()),
            "anchor_write_mass": float(P_new[anchor].mean().item()),
            "anchor_write_mass_delta": float((P_new[anchor] - P[anchor]).mean().item()),
            "non_anchor_write_mass_before": float(P[non_anchor].mean().item()) if bool(non_anchor.any()) else None,
            "non_anchor_write_mass": float(P_new[non_anchor].mean().item()) if bool(non_anchor.any()) else None,
            "non_anchor_write_mass_delta": float((P_new[non_anchor] - P[non_anchor]).mean().item()) if bool(non_anchor.any()) else None,
        })
        return P_new.to(dtype=P_ttt_write.dtype), debug

    def _apply_v70_radio_ttt_write_prior(
        self,
        *,
        token_type: torch.Tensor,
        P_ttt_write: Optional[torch.Tensor],
        num_frames: int,
        patch_grid: Tuple[int, int],
        chunk_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        debug: Dict[str, Any] = {
            "v70_radio_ttt_write_prior_enabled": bool(self.enable_v70_radio_ttt_write_prior),
            "v70_radio_ttt_write_prior_applied": False,
            "v70_radio_ttt_mode": str(self.v70_radio_ttt_mode),
            "v70_radio_ttt_control": str(self.v70_radio_ttt_control),
            "v70_radio_chunk_idx": int(chunk_idx),
        }
        if not self.enable_v70_radio_ttt_write_prior:
            return P_ttt_write, debug
        if P_ttt_write is None:
            debug["v70_radio_ttt_reason"] = "missing_write_prior"
            return P_ttt_write, debug
        sidecar_path = _v70_radio_sidecar_path(self.v70_radio_sidecar_dir, int(chunk_idx))
        if sidecar_path is None:
            debug["v70_radio_ttt_reason"] = "missing_sidecar"
            debug["v70_radio_sidecar_dir"] = str(self.v70_radio_sidecar_dir or "")
            return P_ttt_write, debug
        try:
            try:
                payload = torch.load(sidecar_path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(sidecar_path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            debug["v70_radio_ttt_reason"] = f"sidecar_load_error:{type(exc).__name__}:{exc}"
            debug["v70_radio_sidecar_path"] = str(sidecar_path)
            return P_ttt_write, debug
        required = (
            "radio_confidence",
            "temporal_stability",
            "object_interior_score",
            "object_boundary_score",
            "radio_dynamic_score",
            "radio_sky_context_score",
            "radio_lowtrust_score",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            debug["v70_radio_ttt_reason"] = f"missing_fields:{','.join(missing)}"
            debug["v70_radio_sidecar_path"] = str(sidecar_path)
            return P_ttt_write, debug

        token_type_cpu = token_type.detach().cpu().long().reshape(-1)
        P = P_ttt_write.detach().cpu().float().reshape(-1).clone()
        if int(P.numel()) != int(token_type_cpu.numel()):
            debug["v70_radio_ttt_reason"] = "shape_mismatch"
            debug["v70_radio_ttt_prior_tokens"] = int(P.numel())
            debug["v70_radio_ttt_token_type_tokens"] = int(token_type_cpu.numel())
            return P_ttt_write, debug
        patch_mask = token_type_cpu == TOKEN_TYPE_PATCH
        n_patch = int(patch_mask.sum().item())
        if n_patch <= 0:
            debug["v70_radio_ttt_reason"] = "empty_patch_tokens"
            return P_ttt_write, debug

        conf = _v70_radio_resize_map(payload["radio_confidence"], num_frames=num_frames, patch_grid=patch_grid)
        stability = _v70_radio_resize_map(payload["temporal_stability"], num_frames=num_frames, patch_grid=patch_grid)
        interior = _v70_radio_resize_map(payload["object_interior_score"], num_frames=num_frames, patch_grid=patch_grid)
        boundary = _v70_radio_resize_map(payload["object_boundary_score"], num_frames=num_frames, patch_grid=patch_grid)
        dynamic = _v70_radio_resize_map(payload["radio_dynamic_score"], num_frames=num_frames, patch_grid=patch_grid)
        sky = _v70_radio_resize_map(payload["radio_sky_context_score"], num_frames=num_frames, patch_grid=patch_grid)
        lowtrust = _v70_radio_resize_map(payload["radio_lowtrust_score"], num_frames=num_frames, patch_grid=patch_grid)
        maps = [conf, stability, interior, boundary, dynamic, sky, lowtrust]
        if any(int(x.numel()) != n_patch for x in maps):
            debug["v70_radio_ttt_reason"] = "radio_patch_count_mismatch"
            debug["v70_radio_ttt_patch_tokens"] = int(n_patch)
            debug["v70_radio_ttt_radio_tokens"] = [int(x.numel()) for x in maps]
            return P_ttt_write, debug

        activity = torch.maximum(dynamic, torch.maximum(sky, lowtrust)).clamp(0.0, 1.0)
        static = (
            (conf >= float(self.v70_radio_ttt_min_confidence))
            & (stability >= float(self.v70_radio_ttt_min_stability))
            & (interior >= float(self.v70_radio_ttt_min_interior))
            & (activity <= float(self.v70_radio_ttt_max_activity_risk))
        ).float()
        dynamic_lowstable = (
            (activity > float(self.v70_radio_ttt_max_activity_risk))
            | (stability < float(self.v70_radio_ttt_min_stability))
            | (conf < float(self.v70_radio_ttt_min_confidence))
        ).float()
        cross_object = (
            (boundary >= float(self.v70_radio_ttt_boundary_threshold))
            & (interior < float(self.v70_radio_ttt_min_interior))
        ).float()
        control = str(self.v70_radio_ttt_control).strip().lower()
        if control == "spatial_shuffle":
            seed = 7017000 + max(int(chunk_idx), 0) * 101 + int(n_patch)
            perm = torch.argsort(_deterministic_patch_noise(n_patch, seed=seed), stable=True)
            static = static[perm]
            dynamic_lowstable = dynamic_lowstable[perm]
            cross_object = cross_object[perm]
        elif control == "geometry_only":
            static = torch.zeros_like(static)
            dynamic_lowstable = torch.zeros_like(dynamic_lowstable)
            cross_object = torch.zeros_like(cross_object)

        mode = str(self.v70_radio_ttt_mode).strip().lower()
        scale = torch.ones(n_patch, dtype=torch.float32)
        group = torch.zeros(n_patch, dtype=torch.float32)
        if mode == "static_object_write_floor":
            scale = scale + float(self.v70_radio_ttt_static_boost) * static
            group = static
        elif mode == "dynamic_lowstable_no_persistent":
            scale = torch.clamp(scale - float(self.v70_radio_ttt_suppress) * dynamic_lowstable, min=0.0)
            group = dynamic_lowstable
        elif mode == "cross_object_conflict_veto":
            scale = torch.clamp(scale - float(self.v70_radio_ttt_suppress) * cross_object, min=0.0)
            group = cross_object
        elif mode == "combined_static_dynamic_cross":
            scale = scale + float(self.v70_radio_ttt_static_boost) * static
            scale = scale * torch.clamp(torch.ones_like(scale) - float(self.v70_radio_ttt_suppress) * dynamic_lowstable, min=0.0)
            scale = scale * torch.clamp(torch.ones_like(scale) - float(self.v70_radio_ttt_suppress) * cross_object, min=0.0)
            group = torch.maximum(dynamic_lowstable, torch.maximum(cross_object, static)).clamp(0.0, 1.0)
        else:
            debug["v70_radio_ttt_reason"] = f"unknown_mode:{mode}"
            return P_ttt_write, debug

        P_patch = P[patch_mask].clone()
        P_new_patch = (P_patch * scale).clamp(0.0, 2.0)
        if self.v70_radio_ttt_renorm_mean and P_new_patch.numel():
            old_mean = float(P_patch.mean().item())
            new_mean = float(P_new_patch.mean().item())
            if math.isfinite(old_mean) and math.isfinite(new_mean) and new_mean > 1e-12:
                P_new_patch = (P_new_patch * (old_mean / new_mean)).clamp(0.0, 2.0)
        P_new = P.clone()
        P_new[patch_mask] = P_new_patch
        group_mass = group > 0.5
        if bool(group_mass.any()):
            before = float(P_patch[group_mass].mean().item())
            after = float(P_new_patch[group_mass].mean().item())
            rel = float((after - before) / max(abs(before), 1e-12))
        else:
            before = None
            after = None
            rel = None
        changed = (P_new_patch - P_patch).abs() > 1e-8
        debug.update({
            "v70_radio_ttt_write_prior_applied": bool(changed.any()),
            "v70_radio_ttt_reason": "ok",
            "v70_radio_sidecar_path": str(sidecar_path),
            "v70_radio_sidecar_patch_grid": list(payload.get("patch_grid", [])) if isinstance(payload.get("patch_grid", None), (list, tuple)) else None,
            "v70_radio_target_patch_grid": [int(patch_grid[0]), int(patch_grid[1])],
            "v70_radio_ttt_patch_tokens": int(n_patch),
            "v70_radio_ttt_changed_patch_frac": float(changed.float().mean().item()) if changed.numel() else 0.0,
            "v70_radio_ttt_static_mask_mean": float(static.mean().item()) if static.numel() else 0.0,
            "v70_radio_ttt_dynamic_lowstable_mask_mean": float(dynamic_lowstable.mean().item()) if dynamic_lowstable.numel() else 0.0,
            "v70_radio_ttt_cross_object_mask_mean": float(cross_object.mean().item()) if cross_object.numel() else 0.0,
            "v70_radio_ttt_activity_risk_mean": float(activity.mean().item()) if activity.numel() else 0.0,
            "v70_radio_ttt_group_prior_before": before,
            "v70_radio_ttt_group_prior_after": after,
            "v70_radio_ttt_group_prior_relative_change": rel,
            "v70_radio_ttt_prior_mean_before": float(P_patch.mean().item()) if P_patch.numel() else 1.0,
            "v70_radio_ttt_prior_mean_after": float(P_new_patch.mean().item()) if P_new_patch.numel() else 1.0,
            "v70_radio_ttt_prior_abs_change_ratio": float((P_new_patch - P_patch).abs().mean().item() / max(float(P_patch.mean().item()), 1e-12)) if P_patch.numel() else 0.0,
        })
        return P_new.to(device=P_ttt_write.device, dtype=P_ttt_write.dtype), debug

    def _apply_semantic_ttt_overlap_support_prior(
        self,
        *,
        token_type: torch.Tensor,
        P_ttt_write: Optional[torch.Tensor],
        num_frames: int,
        patch_grid: Tuple[int, int],
        chunk_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        debug: Dict[str, Any] = {
            "semantic_ttt_overlap_support_enabled": bool(str(self.semantic_ttt_overlap_support_dir or "").strip()),
            "semantic_ttt_overlap_support_applied": False,
            "semantic_ttt_overlap_support_dir": str(self.semantic_ttt_overlap_support_dir or ""),
            "semantic_ttt_overlap_support_kind": str(self.semantic_ttt_overlap_support_kind or ""),
            "semantic_ttt_overlap_support_score_key": str(self.semantic_ttt_overlap_support_score_key or ""),
            "semantic_ttt_overlap_support_scope": str(self.semantic_ttt_overlap_support_scope or ""),
            "semantic_ttt_overlap_support_floor": float(self.semantic_ttt_overlap_support_floor),
        }
        if not bool(debug["semantic_ttt_overlap_support_enabled"]):
            debug["semantic_ttt_overlap_support_reason"] = "dir_not_configured"
            return P_ttt_write, debug
        if P_ttt_write is None:
            debug["semantic_ttt_overlap_support_reason"] = "missing_ttt_write_prior"
            return P_ttt_write, debug

        path = _semantic_overlap_support_path(
            self.semantic_ttt_overlap_support_dir,
            int(chunk_idx),
            self.semantic_ttt_overlap_support_kind,
        )
        if path is None:
            debug["semantic_ttt_overlap_support_reason"] = "map_missing"
            return P_ttt_write, debug
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            debug["semantic_ttt_overlap_support_reason"] = f"load_error:{type(exc).__name__}:{exc}"
            debug["semantic_ttt_overlap_support_path"] = str(path)
            return P_ttt_write, debug
        if not isinstance(payload, dict):
            debug["semantic_ttt_overlap_support_reason"] = "payload_not_dict"
            debug["semantic_ttt_overlap_support_path"] = str(path)
            return P_ttt_write, debug

        key = str(self.semantic_ttt_overlap_support_score_key or "score_overlap").strip()
        score = payload.get(key)
        if not torch.is_tensor(score):
            debug["semantic_ttt_overlap_support_reason"] = f"missing_score_key:{key}"
            debug["semantic_ttt_overlap_support_path"] = str(path)
            return P_ttt_write, debug
        score = torch.nan_to_num(score.detach().cpu().float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if score.ndim == 3:
            score = score[0]
        if score.ndim == 1:
            score = score.unsqueeze(0)
        if score.ndim != 2:
            debug["semantic_ttt_overlap_support_reason"] = f"unsupported_score_shape:{tuple(score.shape)}"
            debug["semantic_ttt_overlap_support_path"] = str(path)
            return P_ttt_write, debug

        token_type_cpu = token_type.detach().cpu().long().reshape(-1)
        patch_mask = token_type_cpu == TOKEN_TYPE_PATCH
        n_patch = int(patch_mask.sum().item())
        T = max(int(num_frames), 1)
        if n_patch <= 0 or n_patch % T != 0:
            debug["semantic_ttt_overlap_support_reason"] = "invalid_patch_frame_layout"
            debug["semantic_ttt_overlap_support_patch_tokens"] = int(n_patch)
            debug["semantic_ttt_overlap_support_num_frames"] = int(T)
            return P_ttt_write, debug
        patches_per_frame = n_patch // T
        target_h, target_w = int(patch_grid[0]), int(patch_grid[1])
        if target_h > 0 and target_w > 0 and target_h * target_w != patches_per_frame:
            debug["semantic_ttt_overlap_support_reason"] = "patch_grid_mismatch"
            debug["semantic_ttt_overlap_support_patch_tokens_per_frame"] = int(patches_per_frame)
            debug["semantic_ttt_overlap_support_target_patch_grid"] = [target_h, target_w]
            return P_ttt_write, debug

        tokens_per_frame = int(payload.get("tokens_per_frame", int(score.shape[-1])) or int(score.shape[-1]))
        token_count = min(int(tokens_per_frame), int(score.shape[-1]))
        patch_start = max(0, token_count - patches_per_frame) if token_count >= patches_per_frame else 0
        patch_count = min(patches_per_frame, int(score.shape[-1]) - patch_start)
        if patch_count != patches_per_frame:
            debug["semantic_ttt_overlap_support_reason"] = "support_patch_count_mismatch"
            debug["semantic_ttt_overlap_support_tokens_per_frame"] = int(tokens_per_frame)
            debug["semantic_ttt_overlap_support_patch_tokens_per_frame"] = int(patches_per_frame)
            debug["semantic_ttt_overlap_support_path"] = str(path)
            return P_ttt_write, debug

        ov = min(int(score.shape[0]), T)
        if ov <= 0:
            debug["semantic_ttt_overlap_support_reason"] = "empty_overlap"
            debug["semantic_ttt_overlap_support_path"] = str(path)
            return P_ttt_write, debug
        support_frames = score[:ov, patch_start : patch_start + patches_per_frame].reshape(ov, patches_per_frame)
        support_slice = support_frames.reshape(-1)
        full_support = torch.ones(n_patch, dtype=torch.float32)
        scope = str(self.semantic_ttt_overlap_support_scope or "head_overlap").strip().lower()
        frame_slots: List[Tuple[int, int]] = []
        if scope in {"head", "head_overlap", "overlap_head"}:
            frame_slots = [(i, i) for i in range(0, ov)]
        elif scope in {"tail", "tail_overlap", "overlap_tail"}:
            frame_slots = [(T - ov + i, i) for i in range(0, ov)]
        elif scope in {"both", "both_overlap", "overlap_both"}:
            frame_slots = [(i, i) for i in range(0, ov)] + [(T - ov + i, i) for i in range(0, ov)]
        else:
            debug["semantic_ttt_overlap_support_reason"] = f"unsupported_scope:{scope}"
            debug["semantic_ttt_overlap_support_path"] = str(path)
            return P_ttt_write, debug
        for slot, source_i in frame_slots:
            lo = int(slot) * patches_per_frame
            hi = lo + patches_per_frame
            full_support[lo:hi] = support_frames[int(source_i)]

        floor = min(max(float(self.semantic_ttt_overlap_support_floor), 0.0), 1.0)
        scale = (floor + (1.0 - floor) * full_support).clamp(0.0, 1.0)
        P = P_ttt_write.detach().cpu().float().reshape(-1).clone()
        patch_only_prior = False
        if P.numel() == token_type_cpu.numel():
            P_patch = P[patch_mask].clone()
        elif P.numel() == n_patch:
            patch_only_prior = True
            P_patch = P.clone()
        else:
            debug["semantic_ttt_overlap_support_reason"] = "prior_token_type_length_mismatch"
            debug["semantic_ttt_overlap_support_prior_tokens"] = int(P.numel())
            debug["semantic_ttt_overlap_support_token_type_tokens"] = int(token_type_cpu.numel())
            debug["semantic_ttt_overlap_support_patch_tokens"] = int(n_patch)
            debug["semantic_ttt_overlap_support_path"] = str(path)
            return P_ttt_write, debug
        P_new_patch = (P_patch * scale).clamp(0.0, 2.0)
        if self.semantic_ttt_overlap_support_renorm_mean and P_new_patch.numel():
            old_mean = float(P_patch.mean().item())
            new_mean = float(P_new_patch.mean().item())
            if math.isfinite(old_mean) and math.isfinite(new_mean) and new_mean > 1e-12:
                P_new_patch = (P_new_patch * (old_mean / new_mean)).clamp(0.0, 2.0)
        if patch_only_prior:
            P_out = P_new_patch
        else:
            P[patch_mask] = P_new_patch
            P_out = P
        changed = (P_new_patch - P_patch).abs() > 1e-8
        active_mask = scale < 0.999999
        debug.update({
            "semantic_ttt_overlap_support_applied": bool(changed.any()),
            "semantic_ttt_overlap_support_reason": "ok",
            "semantic_ttt_overlap_support_path": str(path),
            "semantic_ttt_overlap_support_artifact": str(payload.get("artifact", "")),
            "semantic_ttt_overlap_support_payload_kind": str(payload.get("kind", "")),
            "semantic_ttt_overlap_support_payload_mode": str(payload.get("mode", "")),
            "semantic_ttt_overlap_support_payload_schema": str(payload.get("schema", "")),
            "semantic_ttt_overlap_support_chunk_idx": int(chunk_idx),
            "semantic_ttt_overlap_support_overlap_frames": int(ov),
            "semantic_ttt_overlap_support_patch_tokens": int(n_patch),
            "semantic_ttt_overlap_support_patch_only_prior": bool(patch_only_prior),
            "semantic_ttt_overlap_support_patch_tokens_per_frame": int(patches_per_frame),
            "semantic_ttt_overlap_support_tokens_per_frame": int(tokens_per_frame),
            "semantic_ttt_overlap_support_patch_start": int(patch_start),
            "semantic_ttt_overlap_support_target_patch_grid": [target_h, target_w],
            "semantic_ttt_overlap_support_frame_slots": [int(x[0]) for x in frame_slots],
            "semantic_ttt_overlap_support_changed_patch_frac": float(changed.float().mean().item()) if changed.numel() else 0.0,
            "semantic_ttt_overlap_support_active_patch_frac": float(active_mask.float().mean().item()) if active_mask.numel() else 0.0,
            "semantic_ttt_overlap_support_score_mean": float(support_slice.mean().item()) if support_slice.numel() else 1.0,
            "semantic_ttt_overlap_support_score_q10": float(torch.quantile(support_slice, 0.10).item()) if support_slice.numel() else 1.0,
            "semantic_ttt_overlap_support_score_q50": float(torch.quantile(support_slice, 0.50).item()) if support_slice.numel() else 1.0,
            "semantic_ttt_overlap_support_score_q90": float(torch.quantile(support_slice, 0.90).item()) if support_slice.numel() else 1.0,
            "semantic_ttt_overlap_support_prior_mean_before": float(P_patch.mean().item()) if P_patch.numel() else 1.0,
            "semantic_ttt_overlap_support_prior_mean_after": float(P_new_patch.mean().item()) if P_new_patch.numel() else 1.0,
        })
        return P_out.to(device=P_ttt_write.device, dtype=P_ttt_write.dtype), debug

    def _phase_e_write_prior(
        self,
        *,
        probe: ProbeOutput,
        token_type: torch.Tensor,
        dyn_patch: Optional[torch.Tensor],
        explicit_dyn_patch: Optional[torch.Tensor],
        occ_patch: Optional[torch.Tensor],
        unc_patch: Optional[torch.Tensor],
        read_patch: Optional[torch.Tensor],
        P_ref: torch.Tensor,
        base_write_prior: Optional[torch.Tensor] = None,
        semantic_value_patch: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        source = str(self.hmc_write_score_source)
        if source in {"stage_d", "prior", "bl01"}:
            return None, {"hmc_write_score_source": source, "hmc_write_override": False}

        L_tok = int(token_type.numel())
        patch_mask = token_type == TOKEN_TYPE_PATCH
        num_patch = int(patch_mask.sum().item())
        if num_patch <= 0:
            return torch.ones(L_tok, dtype=torch.float32), {"hmc_write_score_source": source, "hmc_write_override": True}

        needs_residual = source in {
            "ttt_residual",
            "residual",
            "ttt_residual_reliable",
            "residual_reliable",
            "residual_reliability",
            "alignment_confidence",
            "alignment",
        }
        if needs_residual:
            residual_tok = self._ttt_residual_token_score(probe, L_tok)
            residual_patch = residual_tok[patch_mask].detach().cpu().float()
        else:
            residual_patch = torch.zeros(num_patch, dtype=torch.float32)

        def _patch_or(default: float, tensor: Optional[torch.Tensor]) -> torch.Tensor:
            if tensor is None:
                return torch.full((num_patch,), float(default), dtype=torch.float32)
            vals = tensor.detach().cpu().float().reshape(-1)
            if vals.numel() < num_patch:
                padded = torch.full((num_patch,), float(default), dtype=torch.float32)
                padded[: vals.numel()] = vals
                return padded
            return vals[:num_patch]

        dyn = _patch_or(0.0, dyn_patch).clamp(0.0, 1.0)
        exp_dyn = _patch_or(0.0, explicit_dyn_patch).clamp(0.0, 1.0)
        occ = _patch_or(0.0, occ_patch).clamp(0.0, 1.0)
        unc = _patch_or(0.0, unc_patch).clamp(0.0, 1.0)
        read = _patch_or(0.0, read_patch).clamp(0.0, 1.0)
        sem_value = _patch_or(1.0, semantic_value_patch).clamp(0.0, 1.0)
        pref = P_ref[patch_mask].detach().cpu().float().clamp(0.0, 1.0)
        reliability = ((1.0 - read) * (1.0 - occ) * (1.0 - unc) * (1.0 - pref)).clamp(0.0, 1.0)
        lo = min(float(self.hmc_write_min), float(self.hmc_write_max))
        hi = max(float(self.hmc_write_min), float(self.hmc_write_max))
        stage_d_composed_source = source.startswith("stage_d_x_") or source.startswith("v5_stage_d_x_")
        semantic_write_sources = {
            "semantic_value",
            "sem_value",
            "sem_x_dg_inv_sqrt",
            "semantic_x_dg_inv_sqrt",
            "stage_d_x_sem",
            "stage_d_x_semantic",
            "stage_d_x_sem_x_dg_inv_sqrt",
            "stage_d_x_semantic_x_dg_inv_sqrt",
        }
        if source in semantic_write_sources:
            # Semantic-composition sources should use the historical v5 stage_d
            # dynamic-rank base, not the Stage D A_tok as the base prior.
            # Otherwise the semantic prior is effectively used twice and the
            # old stage_d_x_* override may erase semantic effects.
            base_patch = (
                1.0 + float(self.hmc_write_alpha) * (-_centered_percentile_rank(dyn))
            ).clamp(lo, hi)
        elif base_write_prior is not None:
            base_write = base_write_prior.detach().cpu().float().reshape(-1)
            if base_write.numel() >= L_tok:
                base_patch = base_write[:L_tok][patch_mask].clamp(0.0, 2.0)
            else:
                padded = torch.ones(L_tok, dtype=torch.float32)
                padded[: base_write.numel()] = base_write
                base_patch = padded[patch_mask].clamp(0.0, 2.0)
            if stage_d_composed_source and base_patch.numel() > 1:
                spread = float((base_patch.max() - base_patch.min()).abs().item())
                if spread < 1e-8:
                    base_patch = (
                        1.0 + float(self.hmc_write_alpha) * (-_centered_percentile_rank(dyn))
                    ).clamp(lo, hi)
        else:
            if stage_d_composed_source:
                # Historical v5 "stage_d" is the eta-mean-preserving dynamic-rank
                # write prior. Rebuild it here when semantic prior is intentionally
                # ignored, so D_g write-score variants remain true no-ops w.r.t.
                # Stage C/D wiring.
                base_patch = (
                    1.0 + float(self.hmc_write_alpha) * (-_centered_percentile_rank(dyn))
                ).clamp(lo, hi)
            else:
                base_patch = torch.ones(num_patch, dtype=torch.float32)
        base_score = _normalize01(base_patch)

        if source in {"dyn", "old_dyn"}:
            raw_score = (1.0 - dyn).clamp(0.0, 1.0)
        elif source in {"explicit_dyn_inv", "exp_dyn_inv", "v5_exp_dyn_inv"}:
            raw_score = (1.0 - exp_dyn).clamp(0.0, 1.0)
        elif source in {"dg_inv", "dg_locked_inv", "read_inv", "read_inverse", "v5_dg_inv"}:
            raw_score = (1.0 - read).clamp(0.0, 1.0)
        elif source in {"dg_inv_sqrt", "v5_dg_inv_sqrt"}:
            raw_score = torch.sqrt((1.0 - read).clamp(0.0, 1.0))
        elif source in {"dg_inv_sq", "v5_dg_inv_sq"}:
            raw_score = (1.0 - read).clamp(0.0, 1.0).square()
        elif source in {"stage_d_x_dg_inv", "v5_stage_d_x_dg_inv"}:
            raw_score = (base_score * (1.0 - read).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_inv_sqrt", "v5_stage_d_x_dg_inv_sqrt"}:
            raw_score = (base_score * torch.sqrt((1.0 - read).clamp(0.0, 1.0))).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_high_inv", "v5_stage_d_x_dg_high_inv"}:
            high_read = ((read - 0.5) * 2.0).clamp(0.0, 1.0)
            raw_score = (base_score * (1.0 - high_read)).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_high_inv_sqrt", "v5_stage_d_x_dg_high_inv_sqrt"}:
            high_read = ((read - 0.5) * 2.0).clamp(0.0, 1.0)
            raw_score = (base_score * torch.sqrt((1.0 - high_read).clamp(0.0, 1.0))).clamp(0.0, 1.0)
        elif source in {"stage_d_x_exp_inv", "v5_stage_d_x_exp_inv"}:
            raw_score = (base_score * (1.0 - exp_dyn).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        elif source in {"stage_d_x_exp_inv_sqrt", "v5_stage_d_x_exp_inv_sqrt"}:
            raw_score = (base_score * torch.sqrt((1.0 - exp_dyn).clamp(0.0, 1.0))).clamp(0.0, 1.0)
        elif source in {"stage_d_x_exp_inv_sq", "v5_stage_d_x_exp_inv_sq", "stage_d_x_exp_focal2"}:
            exp_static = (1.0 - exp_dyn).clamp(0.0, 1.0)
            raw_score = (base_score * exp_static.square()).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_exp_inv_sqrt", "v5_stage_d_x_dg_exp_inv_sqrt"}:
            dg_static = torch.sqrt((1.0 - read).clamp(0.0, 1.0))
            exp_static = torch.sqrt((1.0 - exp_dyn).clamp(0.0, 1.0))
            raw_score = (base_score * dg_static * exp_static).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_exp_inter_inv", "v5_stage_d_x_dg_exp_inter_inv"}:
            consensus_dyn = (read * exp_dyn).clamp(0.0, 1.0)
            raw_score = (base_score * (1.0 - consensus_dyn)).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_exp_inter_inv_sqrt", "v5_stage_d_x_dg_exp_inter_inv_sqrt"}:
            consensus_dyn = (read * exp_dyn).clamp(0.0, 1.0)
            raw_score = (base_score * torch.sqrt((1.0 - consensus_dyn).clamp(0.0, 1.0))).clamp(0.0, 1.0)
        elif source in {"stage_d_x_union_dyn_inv", "v5_stage_d_x_union_dyn_inv"}:
            union_dyn = torch.maximum(read, exp_dyn).clamp(0.0, 1.0)
            raw_score = (base_score * (1.0 - union_dyn)).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_inv_sq", "v5_stage_d_x_dg_inv_sq", "stage_d_x_static_focal2"}:
            static = (1.0 - read).clamp(0.0, 1.0)
            raw_score = (base_score * static.square()).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_inv_pow4", "v5_stage_d_x_dg_inv_pow4", "stage_d_x_static_focal4"}:
            static = (1.0 - read).clamp(0.0, 1.0)
            raw_score = (base_score * static.square().square()).clamp(0.0, 1.0)
        elif source in {"stage_d_x_dg_boundary_focal2", "v5_stage_d_x_dg_boundary_focal2"}:
            static = (1.0 - read).clamp(0.0, 1.0)
            dynamic = read.clamp(0.0, 1.0)
            raw_score = (base_score * static * dynamic.square()).clamp(0.0, 1.0)
        elif source in {"semantic_value", "sem_value"}:
            raw_score = sem_value.clamp(0.0, 1.0)
        elif source in {"sem_x_dg_inv_sqrt", "semantic_x_dg_inv_sqrt"}:
            raw_score = (sem_value * torch.sqrt((1.0 - read).clamp(0.0, 1.0))).clamp(0.0, 1.0)
        elif source in {"stage_d_x_sem", "stage_d_x_semantic"}:
            raw_score = (base_score * sem_value).clamp(0.0, 1.0)
        elif source in {"stage_d_x_sem_x_dg_inv_sqrt", "stage_d_x_semantic_x_dg_inv_sqrt"}:
            dg_static = torch.sqrt((1.0 - read).clamp(0.0, 1.0))
            raw_score = (base_score * sem_value * dg_static).clamp(0.0, 1.0)
        elif source in {"ttt_residual", "residual"}:
            raw_score = _normalize01(residual_patch)
        elif source in {"ttt_residual_reliable", "residual_reliable", "residual_reliability"}:
            raw_score = _normalize01(residual_patch) * reliability
        elif source in {"alignment_confidence", "alignment"}:
            raw_score = _normalize01(1.0 - _normalize01(residual_patch)) * reliability
        else:
            raw_score = (1.0 - dyn).clamp(0.0, 1.0)

        raw_score = torch.nan_to_num(raw_score.float(), nan=0.0, posinf=1.0, neginf=0.0)
        centered = _centered_percentile_rank(raw_score)
        p_patch = (1.0 + float(self.hmc_write_alpha) * centered).clamp(lo, hi)

        sparse_ratio = min(max(float(self.hmc_write_sparse_ratio), 0.0), 1.0)
        sparse_selected = torch.ones(num_patch, dtype=torch.bool)
        if sparse_ratio < 1.0:
            k = max(1, int(round(num_patch * sparse_ratio)))
            k = min(k, num_patch)
            sparse_selected = torch.zeros(num_patch, dtype=torch.bool)
            if k > 0:
                idx = torch.topk(raw_score, k=k, largest=True).indices
                sparse_selected[idx] = True
            if self.hmc_write_sparse_mode in {"hard", "exact", "exact_preserve", "none"}:
                p_patch = sparse_selected.float()
            else:
                p_patch = p_patch * sparse_selected.float()

        A_tok = torch.ones(L_tok, dtype=torch.float32)
        A_tok[patch_mask] = p_patch

        debug = {
            "hmc_write_score_source": source,
            "hmc_write_override": True,
            "hmc_write_alpha": float(self.hmc_write_alpha),
            "hmc_write_min": lo,
            "hmc_write_max": hi,
            "hmc_write_sparse_ratio": sparse_ratio,
            "hmc_write_sparse_mode": self.hmc_write_sparse_mode,
            "hmc_write_selected_mass": float(sparse_selected.float().mean().item()) if sparse_selected.numel() else 1.0,
            "hmc_write_score_mean": float(raw_score.mean().item()) if raw_score.numel() else 0.0,
            "hmc_write_score_q90": float(torch.quantile(raw_score, 0.9).item()) if raw_score.numel() else 0.0,
            "hmc_write_prior_mean_patch": float(p_patch.mean().item()) if p_patch.numel() else 1.0,
            "hmc_write_residual_mean_patch": float(residual_patch.mean().item()) if residual_patch.numel() else 0.0,
            "hmc_write_residual_q90_patch": float(torch.quantile(residual_patch, 0.9).item()) if residual_patch.numel() else 0.0,
            "hmc_write_reliability_mean_patch": float(reliability.mean().item()) if reliability.numel() else 1.0,
            "hmc_write_corr_score_dyn": _pearson_corr(raw_score, dyn),
            "hmc_write_corr_score_exp_dyn": _pearson_corr(raw_score, exp_dyn),
            "hmc_write_corr_score_sem_value": _pearson_corr(raw_score, sem_value),
            "hmc_write_sem_value_mean": float(sem_value.mean().item()) if sem_value.numel() else 1.0,
            "hmc_write_sem_value_q10": float(torch.quantile(sem_value, 0.1).item()) if sem_value.numel() else 1.0,
            "hmc_write_sem_value_q90": float(torch.quantile(sem_value, 0.9).item()) if sem_value.numel() else 1.0,
            "hmc_write_corr_score_unc": _pearson_corr(raw_score, unc),
            "hmc_write_corr_score_residual": _pearson_corr(raw_score, residual_patch),
        }
        return A_tok, debug

    def _semantic_path_enabled(self, path: str) -> bool:
        paths = {
            p.strip().lower()
            for p in str(self.semantic_memory_paths or "").replace(";", ",").split(",")
            if p.strip()
        }
        policy = str(self.semantic_role_policy or "none").strip().lower()
        if policy in {"", "none", "off", "noop", "debug_only"}:
            return False
        return "all" in paths or path.lower() in paths

    @staticmethod
    def _role_counts(values: Optional[torch.Tensor]) -> Dict[str, int]:
        if values is None:
            return {}
        v = values.detach().cpu().long().reshape(-1)
        if v.numel() == 0:
            return {}
        unique, counts = torch.unique(v, return_counts=True)
        return {str(int(k.item())): int(c.item()) for k, c in zip(unique, counts)}

    @staticmethod
    def _semantic_role_control_offset(
        *,
        G: torch.Tensor,
        L: torch.Tensor,
        D: torch.Tensor,
        Q: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> int:
        """Build a deterministic per-chunk seed offset from semantic/risk tokens."""

        accum = 0x345678
        specs = (
            (G.detach().cpu().long().reshape(-1), 1),
            (L.detach().cpu().long().reshape(-1), 17),
            ((D.detach().cpu().float().reshape(-1) * 1000.0).long(), 31),
            ((Q.detach().cpu().float().reshape(-1) * 1000.0).long(), 47),
        )
        mask = patch_mask.detach().cpu().bool().reshape(-1)
        for values, mul in specs:
            if values.numel() != mask.numel() or not bool(mask.any()):
                continue
            selected = values[mask]
            if selected.numel() == 0:
                continue
            stride = max(int(selected.numel()) // 128, 1)
            sample = selected[::stride]
            total = int(sample.sum().item())
            accum = (accum * 1000003 + total * mul + int(selected.numel()) * (mul + 97)) & 0x7FFFFFFF
        return int(accum)

    def _apply_ttt_semantic_role_control(
        self,
        roles: torch.Tensor,
        *,
        G: torch.Tensor,
        L: torch.Tensor,
        D: torch.Tensor,
        Q: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply same-path TTT role controls while preserving role mass."""

        mode = str(getattr(self, "semantic_role_control_mode", "none") or "none").strip().lower()
        seed = int(getattr(self, "semantic_role_control_seed", 12345))
        idx = torch.nonzero(patch_mask.detach().cpu().bool().reshape(-1), as_tuple=False).reshape(-1)
        before_counts = self._role_counts(roles[idx]) if idx.numel() else {}
        debug: Dict[str, Any] = {
            "semantic_role_control_mode": mode or "none",
            "semantic_role_control_seed": seed,
            "semantic_role_control_scope": "ttt_patch_roles",
            "semantic_role_control_applied": False,
            "semantic_role_control_reason": "disabled",
            "R_ttt_role_counts_before_control": before_counts,
            "R_ttt_role_counts_after_control": before_counts,
            "semantic_role_control_changed_fraction": 0.0,
        }
        if mode in {"", "none", "off", "noop"}:
            return roles, debug
        if idx.numel() <= 1:
            debug["semantic_role_control_reason"] = "insufficient_patch_tokens"
            return roles, debug

        offset = self._semantic_role_control_offset(G=G, L=L, D=D, Q=Q, patch_mask=patch_mask)
        effective_seed = int((seed + offset) % 2147483647)
        if effective_seed <= 0:
            effective_seed += 1
        patch_roles = roles.detach().cpu().long().reshape(-1)[idx].clone()
        controlled_roles = patch_roles.clone()

        if mode in {"random_same_mass", "random", "random_role_same_mass"}:
            unique, counts = torch.unique(patch_roles, return_counts=True)
            multiset = torch.repeat_interleave(unique, counts)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(effective_seed)
            controlled_roles = multiset[torch.randperm(int(multiset.numel()), generator=generator)]
            debug["semantic_role_control_reason"] = "random_same_mass"
        elif mode in {"shuffled_semantic", "semantic_shuffle", "cyclic_shuffle"}:
            shift = int(effective_seed % int(patch_roles.numel()))
            if shift == 0:
                shift = max(int(patch_roles.numel()) // 2, 1)
            controlled_roles = torch.roll(patch_roles, shifts=shift)
            debug["semantic_role_control_reason"] = "cyclic_shuffled_semantic_roles"
            debug["semantic_role_control_shift"] = shift
        else:
            debug["semantic_role_control_reason"] = f"unknown_mode:{mode}"
            return roles, debug

        out = roles.detach().cpu().long().reshape(-1).clone()
        out[idx] = controlled_roles
        changed = (controlled_roles != patch_roles).float()
        debug.update({
            "semantic_role_control_applied": True,
            "semantic_role_control_effective_seed": effective_seed,
            "R_ttt_role_counts_after_control": self._role_counts(out[idx]),
            "semantic_role_control_changed_fraction": float(changed.mean().item()) if changed.numel() else 0.0,
            "semantic_role_control_note": "Role-level TTT control preserves role mass and perturbs patch-token role positions.",
        })
        return out.reshape_as(roles), debug

    def _apply_swa_semantic_role_control(
        self,
        roles: torch.Tensor,
        *,
        G: torch.Tensor,
        L: torch.Tensor,
        D: torch.Tensor,
        Q: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply same-mass SWA role controls without touching TTT role fields."""

        mode = str(getattr(self, "semantic_role_control_mode", "none") or "none").strip().lower()
        seed = int(getattr(self, "semantic_role_control_seed", 12345))
        idx = torch.nonzero(patch_mask.detach().cpu().bool().reshape(-1), as_tuple=False).reshape(-1)
        before_counts = self._role_counts(roles[idx]) if idx.numel() else {}
        debug: Dict[str, Any] = {
            "semantic_swa_role_control_mode": mode or "none",
            "semantic_swa_role_control_seed": seed,
            "semantic_swa_role_control_scope": "swa_patch_roles",
            "semantic_swa_role_control_applied": False,
            "semantic_swa_role_control_reason": "disabled",
            "R_swa_role_counts_before_control": before_counts,
            "R_swa_role_counts_after_control": before_counts,
            "semantic_swa_role_control_changed_fraction": 0.0,
        }
        if mode in {"", "none", "off", "noop"}:
            return roles, debug
        if idx.numel() <= 1:
            debug["semantic_swa_role_control_reason"] = "insufficient_patch_tokens"
            return roles, debug

        offset = self._semantic_role_control_offset(G=G, L=L, D=D, Q=Q, patch_mask=patch_mask)
        effective_seed = int((seed + offset) % 2147483647)
        if effective_seed <= 0:
            effective_seed += 1
        patch_roles = roles.detach().cpu().long().reshape(-1)[idx].clone()
        controlled_roles = patch_roles.clone()

        if mode in {"random_same_mass", "random", "random_role_same_mass"}:
            unique, counts = torch.unique(patch_roles, return_counts=True)
            multiset = torch.repeat_interleave(unique, counts)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(effective_seed)
            controlled_roles = multiset[torch.randperm(int(multiset.numel()), generator=generator)]
            debug["semantic_swa_role_control_reason"] = "random_same_mass"
        elif mode in {"shuffled_semantic", "semantic_shuffle", "cyclic_shuffle"}:
            shift = int(effective_seed % int(patch_roles.numel()))
            if shift == 0:
                shift = max(int(patch_roles.numel()) // 2, 1)
            controlled_roles = torch.roll(patch_roles, shifts=shift)
            debug["semantic_swa_role_control_reason"] = "cyclic_shuffled_semantic_roles"
            debug["semantic_swa_role_control_shift"] = shift
        else:
            debug["semantic_swa_role_control_reason"] = f"unknown_mode:{mode}"
            return roles, debug

        out = roles.detach().cpu().long().reshape(-1).clone()
        out[idx] = controlled_roles
        changed = (controlled_roles != patch_roles).float()
        debug.update({
            "semantic_swa_role_control_applied": True,
            "semantic_swa_role_control_effective_seed": effective_seed,
            "R_swa_role_counts_after_control": self._role_counts(out[idx]),
            "semantic_swa_role_control_changed_fraction": float(changed.mean().item()) if changed.numel() else 0.0,
            "semantic_swa_role_control_note": "SWA control preserves role mass and perturbs patch-token source roles.",
        })
        return out.reshape_as(roles), debug

    @staticmethod
    def _label_counts(values: Optional[torch.Tensor]) -> Dict[str, int]:
        if values is None:
            return {}
        v = values.detach().cpu().long().reshape(-1)
        if v.numel() == 0:
            return {}
        unique, counts = torch.unique(v, return_counts=True)
        return {str(int(k.item())): int(c.item()) for k, c in zip(unique, counts)}

    @classmethod
    def _label_role_counts(
        cls,
        labels: Optional[torch.Tensor],
        roles: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Dict[str, Dict[str, int]]:
        if labels is None or roles is None:
            return {}
        l = labels.detach().cpu().long().reshape(-1)
        r = roles.detach().cpu().long().reshape(-1)
        if int(l.numel()) != int(r.numel()) or int(l.numel()) == 0:
            return {}
        if mask is not None:
            m = mask.detach().cpu().bool().reshape(-1)
            if int(m.numel()) == int(l.numel()):
                l = l[m]
                r = r[m]
        out: Dict[str, Dict[str, int]] = {}
        for label_id in torch.unique(l):
            label_mask = l == label_id
            out[str(int(label_id.item()))] = cls._role_counts(r[label_mask])
        return out

    @staticmethod
    def _label_condition_metrics(
        labels: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        *,
        D: torch.Tensor,
        Q: torch.Tensor,
        C: Optional[torch.Tensor] = None,
        S: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, float]]:
        if labels is None:
            return {}
        l = labels.detach().cpu().long().reshape(-1)
        d = D.detach().cpu().float().reshape(-1)
        q = Q.detach().cpu().float().reshape(-1)
        c = C.detach().cpu().float().reshape(-1) if C is not None else None
        s = S.detach().cpu().float().reshape(-1) if S is not None else None
        if int(l.numel()) != int(d.numel()) or int(l.numel()) != int(q.numel()) or int(l.numel()) == 0:
            return {}
        if c is not None and int(c.numel()) != int(l.numel()):
            c = None
        if s is not None and int(s.numel()) != int(l.numel()):
            s = None
        if mask is not None:
            m = mask.detach().cpu().bool().reshape(-1)
            if int(m.numel()) == int(l.numel()):
                l = l[m]
                d = d[m]
                q = q[m]
                if c is not None:
                    c = c[m]
                if s is not None:
                    s = s[m]
        out: Dict[str, Dict[str, float]] = {}
        for label_id in torch.unique(l):
            label_mask = l == label_id
            if not bool(label_mask.any()):
                continue
            d_vals = d[label_mask]
            q_vals = q[label_mask]
            out[str(int(label_id.item()))] = {
                "token_count": float(label_mask.sum().item()),
                "D_mean": float(d_vals.mean().item()),
                "D_p90": float(torch.quantile(d_vals, 0.90).item()) if int(d_vals.numel()) > 1 else float(d_vals.mean().item()),
                "Q_mean": float(q_vals.mean().item()),
                "Q_p10": float(torch.quantile(q_vals, 0.10).item()) if int(q_vals.numel()) > 1 else float(q_vals.mean().item()),
            }
            if c is not None:
                c_vals = c[label_mask]
                out[str(int(label_id.item()))].update({
                    "conflict_mean": float(c_vals.mean().item()),
                    "conflict_p90": float(torch.quantile(c_vals, 0.90).item()) if int(c_vals.numel()) > 1 else float(c_vals.mean().item()),
                })
            if s is not None:
                s_vals = s[label_mask]
                out[str(int(label_id.item()))].update({
                    "scale_risk_mean": float(s_vals.mean().item()),
                    "scale_risk_p90": float(torch.quantile(s_vals, 0.90).item()) if int(s_vals.numel()) > 1 else float(s_vals.mean().item()),
                })
        return out

    def set_semantic_condition_signals(
        self,
        *,
        conflict_level: str = "unavailable",
        conflict_source: str = "none",
        conflict_value: float = 0.0,
        conflict_summary: Optional[Dict[str, Any]] = None,
        conflict_tok: Optional[Any] = None,
        conflict_token_exact: bool = False,
        scale_level: str = "unavailable",
        scale_source: str = "none",
        scale_value: float = 0.0,
        scale_summary: Optional[Dict[str, Any]] = None,
        scale_tok: Optional[Any] = None,
        scale_token_exact: bool = False,
    ) -> None:
        """Set provenance-tagged semantic risk conditions for role routing.

        v27 used audited chunk-level broadcasts for conflict/scale risk. v28
        can pass token-level tensors; the provenance fields prevent token
        proxies from being mistaken for stronger evidence than they are.
        """

        self.semantic_condition_conflict_level = str(conflict_level or "unavailable")
        self.semantic_condition_conflict_source = str(conflict_source or "none")
        self.semantic_condition_conflict_value = float(conflict_value or 0.0)
        self.semantic_condition_conflict_summary = dict(conflict_summary or {})
        self.semantic_condition_conflict_tok = _optional_1d_float_tensor(conflict_tok)
        self.semantic_condition_conflict_token_exact = bool(conflict_token_exact and self.semantic_condition_conflict_tok is not None)
        self.semantic_condition_scale_level = str(scale_level or "unavailable")
        self.semantic_condition_scale_source = str(scale_source or "none")
        self.semantic_condition_scale_value = float(scale_value or 0.0)
        self.semantic_condition_scale_summary = dict(scale_summary or {})
        self.semantic_condition_scale_tok = _optional_1d_float_tensor(scale_tok)
        self.semantic_condition_scale_token_exact = bool(scale_token_exact and self.semantic_condition_scale_tok is not None)

    def _swa_stable_redirection_token_mask(
        self,
        *,
        token_type: torch.Tensor,
        patch_mask: torch.Tensor,
        num_frames: int,
        patch_grid: Tuple[int, int],
        chunk_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load a Phase3 SWA stable overlap map as a token-aligned role mask."""

        out = torch.zeros_like(patch_mask, dtype=torch.bool)
        root = Path(str(getattr(self, "semantic_swa_redirection_phase3_root", "") or "")).expanduser()
        use_random = bool(getattr(self, "semantic_swa_redirection_random_control", False))
        case = (
            str(getattr(self, "semantic_swa_redirection_random_stable_case", "") or "")
            if use_random
            else str(getattr(self, "semantic_swa_redirection_stable_case", "") or "")
        )
        layer = int(getattr(self, "semantic_swa_redirection_layer", 3))
        q = min(max(float(getattr(self, "semantic_swa_redirection_top_quantile", 0.75)), 0.0), 1.0)
        overlap_frames = max(int(getattr(self, "semantic_swa_redirection_overlap_frames", 3)), 0)
        debug: Dict[str, Any] = {
            "swa_redirection_available": False,
            "swa_redirection_reason": "disabled",
            "swa_redirection_phase3_root": str(root) if str(root) else "",
            "swa_redirection_case": case,
            "swa_redirection_random_control": bool(use_random),
            "swa_redirection_layer": int(layer),
            "swa_redirection_top_quantile": float(q),
            "swa_redirection_overlap_frames_requested": int(overlap_frames),
            "swa_redirection_chunk_idx": int(chunk_idx),
            "swa_redirection_selected_tokens": 0,
        }
        if not str(root) or not case:
            return out, debug
        feature_dir = root / f"chunk{int(chunk_idx):02d}" / case / "swa_overlap_feature_maps"
        feature_path = feature_dir / f"chunk_{int(chunk_idx):03d}_swa_overlap_source_replace_layer_{int(layer):02d}.pt"
        if not feature_path.is_file():
            candidates = sorted(feature_dir.glob(f"chunk_{int(chunk_idx):03d}_swa_overlap_source_replace_layer_*.pt"))
            if candidates:
                feature_path = candidates[-1]
            else:
                debug.update({
                    "swa_redirection_reason": "missing_feature_map",
                    "swa_redirection_feature_path": str(feature_path),
                })
                return out, debug
        try:
            payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            debug.update({
                "swa_redirection_reason": f"feature_load_error:{type(exc).__name__}:{exc}",
                "swa_redirection_feature_path": str(feature_path),
            })
            return out, debug
        score = payload.get("score_overlap") if isinstance(payload, dict) else None
        if not torch.is_tensor(score):
            debug.update({
                "swa_redirection_reason": "missing_score_overlap",
                "swa_redirection_feature_path": str(feature_path),
            })
            return out, debug
        x = score.detach().cpu().float()
        if x.ndim == 3:
            x = x[0]
        if x.ndim != 2:
            debug.update({
                "swa_redirection_reason": f"bad_score_shape:{tuple(score.shape)}",
                "swa_redirection_feature_path": str(feature_path),
            })
            return out, debug
        H, W = int(patch_grid[0]), int(patch_grid[1])
        T = int(num_frames)
        patch_per_frame = max(H * W, 0)
        tokens_per_frame = int(payload.get("tokens_per_frame", int(x.shape[-1])) or int(x.shape[-1])) if isinstance(payload, dict) else int(x.shape[-1])
        patch_start = max(0, int(tokens_per_frame) - patch_per_frame)
        patch_end = patch_start + patch_per_frame
        if patch_per_frame <= 0 or T <= 0 or int(x.shape[-1]) < patch_end:
            debug.update({
                "swa_redirection_reason": "patch_grid_mismatch",
                "swa_redirection_feature_path": str(feature_path),
                "swa_redirection_score_shape": list(int(v) for v in x.shape),
                "swa_redirection_patch_grid": [H, W],
            })
            return out, debug
        ov = min(int(overlap_frames or payload.get("overlap_frames_effective", 0) or x.shape[0]), int(x.shape[0]), T)
        patch_score = x[:ov, patch_start:patch_end].reshape(ov, H, W).clamp_min(0.0)
        positive = patch_score[patch_score > 0.0]
        if int(positive.numel()) <= 0:
            debug.update({
                "swa_redirection_reason": "empty_positive_support",
                "swa_redirection_feature_path": str(feature_path),
                "swa_redirection_overlap_frames_effective": int(ov),
            })
            return out, debug
        threshold = float(torch.quantile(positive, q).item())
        selected_cube = (patch_score >= threshold) & (patch_score > 0.0)
        patch_flat = torch.zeros((T, H, W), dtype=torch.bool)
        patch_flat[:ov] = selected_cube
        patch_values = patch_flat.reshape(-1)
        patch_indices = torch.nonzero(patch_mask.detach().cpu().bool().reshape(-1), as_tuple=False).reshape(-1)
        if int(patch_indices.numel()) != int(patch_values.numel()):
            debug.update({
                "swa_redirection_reason": "token_patch_count_mismatch",
                "swa_redirection_feature_path": str(feature_path),
                "swa_redirection_patch_token_count": int(patch_indices.numel()),
                "swa_redirection_mask_patch_count": int(patch_values.numel()),
            })
            return out, debug
        out = out.detach().cpu().bool().reshape(-1)
        out[patch_indices] = patch_values
        out = out.reshape_as(patch_mask)
        frame_counts = [int(v) for v in selected_cube.reshape(ov, -1).sum(dim=1).tolist()]
        debug.update({
            "swa_redirection_available": True,
            "swa_redirection_reason": "ok",
            "swa_redirection_feature_path": str(feature_path),
            "swa_redirection_schema": payload.get("schema") if isinstance(payload, dict) else None,
            "swa_redirection_overlap_frames_effective": int(ov),
            "swa_redirection_patch_grid": [H, W],
            "swa_redirection_positive_support": int(positive.numel()),
            "swa_redirection_threshold": float(threshold),
            "swa_redirection_selected_tokens": int(out.sum().item()),
            "swa_redirection_selected_patch_mass": float(selected_cube.float().mean().item()) if selected_cube.numel() else 0.0,
            "swa_redirection_frame_counts": frame_counts,
            "swa_redirection_score_positive_mean": float(positive.mean().item()),
            "swa_redirection_score_positive_q75": float(torch.quantile(positive, 0.75).item()),
            "swa_redirection_score_positive_q90": float(torch.quantile(positive, 0.90).item()),
        })
        return out, debug

    def _fine_path_roles(
        self,
        *,
        G: torch.Tensor,
        L: torch.Tensor,
        D: torch.Tensor,
        Q: torch.Tensor,
        patch_mask: torch.Tensor,
        high_d: torch.Tensor,
        low_d: torch.Tensor,
        high_trust: torch.Tensor,
        read_patch_tok: Optional[torch.Tensor] = None,
        key_stability_tok: Optional[torch.Tensor] = None,
        num_frames: int = 0,
        patch_grid: Tuple[int, int] = (0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Build path-specific fine-label roles with D/Q and v27 risk signals."""

        R_frame = torch.full_like(G, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
        R_global = torch.full_like(G, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
        R_swa = torch.full_like(G, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
        R_ttt = torch.full_like(G, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
        policy = str(self.semantic_role_policy or "").strip().lower()
        causal_policy = policy.startswith("causal_")

        conflict_level = str(getattr(self, "semantic_condition_conflict_level", "unavailable") or "unavailable")
        scale_level = str(getattr(self, "semantic_condition_scale_level", "unavailable") or "unavailable")
        conflict_available = conflict_level not in {"", "none", "off", "unavailable"}
        scale_available = scale_level not in {"", "none", "off", "unavailable"}
        conflict_value = float(getattr(self, "semantic_condition_conflict_value", 0.0) or 0.0)
        scale_value = float(getattr(self, "semantic_condition_scale_value", 0.0) or 0.0)
        conflict_tok = _optional_1d_float_tensor(getattr(self, "semantic_condition_conflict_tok", None))
        scale_tok = _optional_1d_float_tensor(getattr(self, "semantic_condition_scale_tok", None))
        conflict_token_exact = bool(
            getattr(self, "semantic_condition_conflict_token_exact", False)
            and conflict_tok is not None
            and int(conflict_tok.numel()) == int(D.numel())
        )
        scale_token_exact = bool(
            getattr(self, "semantic_condition_scale_token_exact", False)
            and scale_tok is not None
            and int(scale_tok.numel()) == int(D.numel())
        )
        if conflict_available and conflict_token_exact and conflict_tok is not None:
            C = conflict_tok.to(dtype=D.dtype).reshape_as(D).clamp(0.0, 1.0)
        else:
            C = torch.full_like(D, max(0.0, min(1.0, conflict_value))) if conflict_available else torch.zeros_like(D)
        if scale_available and scale_token_exact and scale_tok is not None:
            S = scale_tok.to(dtype=D.dtype).reshape_as(D).clamp(0.0, 1.0)
        else:
            S = torch.full_like(D, max(0.0, min(1.0, scale_value))) if scale_available else torch.zeros_like(D)
        high_conflict = C >= 0.50 if conflict_available else torch.zeros_like(patch_mask, dtype=torch.bool)
        high_scale = S >= 0.50 if scale_available else torch.zeros_like(patch_mask, dtype=torch.bool)

        known = (L != int(SEMANTIC_FINE_LABEL_UNKNOWN)) & (L >= 0) & patch_mask
        structure = torch.zeros_like(patch_mask, dtype=torch.bool)
        sky = torch.zeros_like(patch_mask, dtype=torch.bool)
        vegetation = torch.zeros_like(patch_mask, dtype=torch.bool)
        movable = torch.zeros_like(patch_mask, dtype=torch.bool)
        for label_id in SEMANTIC_FINE_STRUCTURE_IDS:
            structure |= L == int(label_id)
        for label_id in SEMANTIC_FINE_SKY_IDS:
            sky |= L == int(label_id)
        for label_id in SEMANTIC_FINE_VEGETATION_IDS:
            vegetation |= L == int(label_id)
        for label_id in SEMANTIC_FINE_MOVABLE_IDS:
            movable |= L == int(label_id)

        structure = (structure | ((G == int(SEMANTIC_GROUP_STRUCTURE_ANCHOR)) & known)) & patch_mask
        sky = sky & patch_mask
        vegetation = (vegetation | ((G == int(SEMANTIC_GROUP_LOW_VALUE_STUFF)) & known & ~sky)) & patch_mask
        movable = (movable | ((G == int(SEMANTIC_GROUP_MOVABLE_THING)) & patch_mask)) & patch_mask
        unknown = patch_mask & (~known)

        if causal_policy:
            stable_structure = structure & low_d & high_trust & (~high_conflict) & (~high_scale)
            risky_structure = structure & high_trust & (high_d | high_conflict | high_scale)
        else:
            stable_structure = structure & low_d & high_trust
            risky_structure = structure & (~stable_structure)
        sky_highd = sky & high_d & high_trust
        lowstuff_highd = (sky | vegetation | movable) & high_d & high_trust
        risky_nonstructure = (vegetation | movable) & high_d & high_trust
        if causal_policy:
            risky_nonstructure = (vegetation | movable) & high_d & high_trust & (high_conflict | high_scale | high_d)
        weak_nonstructure = (vegetation | movable) & (~risky_nonstructure) & high_trust

        read_active = torch.zeros_like(patch_mask, dtype=torch.bool)
        if read_patch_tok is not None:
            read_values = read_patch_tok.detach().cpu().float().reshape(-1)
            if int(read_values.numel()) == int(D.numel()):
                read_active = (read_values > 0.50) & patch_mask
        key_stability_available = False
        key_stability_thr: Optional[float] = None
        key_stable = torch.zeros_like(patch_mask, dtype=torch.bool)
        if key_stability_tok is not None:
            key_stability = key_stability_tok.detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
            if int(key_stability.numel()) == int(D.numel()) and bool(patch_mask.any()):
                key_stability_available = True
                q = min(max(float(self.semantic_role_highd_quantile), 0.0), 1.0)
                key_stability_thr = float(torch.quantile(key_stability[patch_mask], q).item())
                key_stable = (key_stability >= float(key_stability_thr)) & patch_mask

        # Frame/global source roles: source protection for stable structure,
        # weak source for sky/vegetation, source skip only when semantic risk is
        # also high-D. Query tokens remain untouched in the model hook.
        R_frame[stable_structure] = int(SEMANTIC_ROLE_POSITIVE_LONG)
        R_global[stable_structure] = int(SEMANTIC_ROLE_POSITIVE_LONG)
        R_frame[risky_structure] = int(SEMANTIC_ROLE_PROTECT_NEUTRAL)
        R_global[risky_structure] = int(SEMANTIC_ROLE_PROTECT_NEUTRAL)
        R_frame[sky & high_trust] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
        R_global[sky & high_trust] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
        R_frame[weak_nonstructure] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
        R_global[weak_nonstructure] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
        R_frame[risky_nonstructure] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
        R_global[risky_nonstructure] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
        if policy in {"fine_fg_lowstuff_highd_skip", "causal_fg_semantic_risk_skip", "causal_fg_soft_skip"}:
            R_frame[sky_highd] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
            R_global[sky_highd] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
        if causal_policy and policy in {"causal_fg_structure_conflict_protect", "causal_fg_risk_only"}:
            R_frame[structure & (high_conflict | high_scale) & high_trust] = int(SEMANTIC_ROLE_PROTECT_NEUTRAL)
            R_global[structure & (high_conflict | high_scale) & high_trust] = int(SEMANTIC_ROLE_PROTECT_NEUTRAL)

        # SWA is local continuity: structure anchors are protected; sky is
        # neutral context; vegetation/movable drops only under high-D risk.
        R_swa[stable_structure | risky_structure] = int(SEMANTIC_ROLE_PROTECT_NEUTRAL)
        R_swa[sky & high_trust] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
        R_swa[weak_nonstructure] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
        R_swa[risky_nonstructure] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)

        # TTT is long memory: stable structure is positive; sky is never
        # positive/negative; vegetation/movable can only be short negative when
        # high-D. Conflict/scale gates are explicitly unavailable here.
        R_ttt[stable_structure] = int(SEMANTIC_ROLE_POSITIVE_LONG)
        R_ttt[risky_structure] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
        R_ttt[sky & high_trust] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
        R_ttt[weak_nonstructure] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
        R_ttt[risky_nonstructure] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
        if policy in {"fine_ttt_lowstuff_highd_short", "causal_ttt_lowstuff_highd_conflict_shortneg", "causal_ttt_full_role_tree"}:
            R_ttt[lowstuff_highd] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
        swa_redirection_policy = policy in {
            "fine_ttt_swa_stable_top25_positive_context_neutral",
            "fine_ttt_swa_stable_redirection_positive_context_neutral",
            "fine_ttt_swa_stable_top25_source_boost_context_neutral",
        }
        swa_source_boost_policy = policy in {
            "fine_ttt_swa_stable_top25_source_boost_context_neutral",
        }
        swa_redirection_mask = torch.zeros_like(patch_mask, dtype=torch.bool)
        swa_redirection_debug: Dict[str, Any] = {
            "swa_redirection_policy_applied": False,
            "swa_redirection_policy": policy,
        }
        read_aligned_ttt_policy = policy in {
            "fine_ttt_read_active_all_positive",
            "fine_ttt_read_active_key_stable_positive_context_neutral",
            "fine_ttt_read_active_key_stable_lowd_positive_context_neutral",
            "fine_ttt_read_active_regime_guarded_pos_neutral",
            "fine_ttt_read_active_structure_positive_highd_short",
            "fine_ttt_read_active_structure_positive_context_neutral",
            "fine_ttt_read_aligned_structure_positive",
        }
        read_active_key_stable_lowd = torch.zeros_like(patch_mask, dtype=torch.bool)
        if read_aligned_ttt_policy:
            read_active_structure = read_active & structure & high_trust
            if policy in {
                "fine_ttt_read_active_key_stable_positive_context_neutral",
                "fine_ttt_read_active_key_stable_lowd_positive_context_neutral",
            }:
                read_active_key_stable = read_active & key_stable & high_trust
                if policy == "fine_ttt_read_active_key_stable_lowd_positive_context_neutral":
                    read_active_key_stable_lowd = read_active_key_stable & low_d
                    read_active_key_positive = read_active_key_stable_lowd
                else:
                    read_active_key_positive = read_active_key_stable
                read_active_key_context = read_active & high_trust & (~read_active_key_positive)
                R_ttt[read_active_key_positive] = int(SEMANTIC_ROLE_POSITIVE_LONG)
                R_ttt[read_active_key_context] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            elif policy == "fine_ttt_read_active_regime_guarded_pos_neutral":
                read_active_stable_structure = read_active & structure & low_d & high_trust
                read_active_guarded_context = read_active & high_trust & (~read_active_stable_structure)
                R_ttt[read_active_stable_structure] = int(SEMANTIC_ROLE_POSITIVE_LONG)
                R_ttt[read_active_guarded_context] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            elif policy == "fine_ttt_read_active_all_positive":
                R_ttt[read_active & high_trust] = int(SEMANTIC_ROLE_POSITIVE_LONG)
            else:
                R_ttt[read_active_structure] = int(SEMANTIC_ROLE_POSITIVE_LONG)
            if policy == "fine_ttt_read_active_structure_positive_context_neutral":
                read_active_context = read_active & high_trust & (~read_active_structure)
                R_ttt[read_active_context] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            elif policy not in {
                "fine_ttt_read_active_all_positive",
                "fine_ttt_read_active_key_stable_positive_context_neutral",
                "fine_ttt_read_active_key_stable_lowd_positive_context_neutral",
                "fine_ttt_read_active_regime_guarded_pos_neutral",
            }:
                read_active_highd_context = read_active & high_d & high_trust & (sky | vegetation | movable)
                R_ttt[read_active_highd_context] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
        if causal_policy:
            high_risk = high_conflict | high_scale
            R_ttt[structure & high_trust & high_risk] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            R_ttt[structure & low_d & high_trust & (~high_risk)] = int(SEMANTIC_ROLE_POSITIVE_LONG)
            R_ttt[(vegetation | movable) & high_d & high_trust & high_risk] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
            R_ttt[sky & high_trust] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            if policy in {"causal_ttt_block_highconflict_structure_longwrite"}:
                R_ttt[structure & high_conflict & high_trust] = int(SEMANTIC_ROLE_PROTECT_NEUTRAL)
            if policy in {"causal_ttt_risk_only"}:
                R_ttt[patch_mask & high_trust & high_risk] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
                R_ttt[structure & high_trust & (~high_risk)] = int(SEMANTIC_ROLE_POSITIVE_LONG)

        if policy in {"v76_ttt_sky_neutral_only", "v76_ttt_sky_only"}:
            R_ttt = torch.full_like(R_ttt, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
            R_ttt[(stable_structure | risky_structure) & high_trust] = int(SEMANTIC_ROLE_POSITIVE_LONG)
            R_ttt[sky & high_trust] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            R_ttt[(vegetation | movable) & high_d & high_trust] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
        elif policy in {
            "v76_ttt_vegetation_neutral_only",
            "v76_ttt_tree_neutral_only",
            "v76_ttt_vegetation_only",
            "v76_ttt_tree_only",
        }:
            R_ttt = torch.full_like(R_ttt, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
            R_ttt[(stable_structure | risky_structure) & high_trust] = int(SEMANTIC_ROLE_POSITIVE_LONG)
            R_ttt[vegetation & high_trust] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            R_ttt[movable & high_d & high_trust] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
        elif policy in {
            "v76_ttt_structure_neutral_only",
            "v76_ttt_layout_neutral_only",
            "v76_ttt_structure_only",
        }:
            R_ttt = torch.full_like(R_ttt, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
            R_ttt[(stable_structure | risky_structure) & high_trust] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            R_ttt[(vegetation | movable) & high_d & high_trust] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)

        v57_group_level_source = torch.zeros_like(patch_mask, dtype=torch.bool)
        if policy in {
            "v57_forced_any_semantic_source_skip",
            "v57_force_any_source_skip",
            "forced_semantic_source_skip",
        }:
            semantic_any = (known | ((G != int(SEMANTIC_GROUP_UNCERTAIN_REGION)) & patch_mask)) & high_trust
            v57_group_level_source |= semantic_any
            R_frame[semantic_any] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
            R_global[semantic_any] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
            R_swa[semantic_any] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            R_ttt[semantic_any] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)

        if policy in {
            "v57_sky_lowstuff_source_skip",
            "v57_sky_lowstuff_vegetation_source_skip",
            "sky_lowstuff_source_skip",
        }:
            weak_context_source = (sky | vegetation | ((G == int(SEMANTIC_GROUP_LOW_VALUE_STUFF)) & patch_mask)) & high_trust
            v57_group_level_source |= weak_context_source
            R_frame[weak_context_source] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
            R_global[weak_context_source] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
            R_swa[weak_context_source] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            R_ttt[weak_context_source] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)

        for R in (R_frame, R_global, R_swa, R_ttt):
            R[(unknown & ~v57_group_level_source) | (~high_trust & patch_mask)] = int(SEMANTIC_ROLE_FALLBACK)

        if swa_redirection_policy:
            swa_redirection_mask, swa_redirection_debug = self._swa_stable_redirection_token_mask(
                token_type=G,
                patch_mask=patch_mask,
                num_frames=max(int(num_frames), 1),
                patch_grid=(int(patch_grid[0]), int(patch_grid[1])),
                chunk_idx=int(getattr(self, "current_chunk_idx", -1)),
            )
            swa_redirection_mask = swa_redirection_mask & patch_mask
            if bool(swa_redirection_debug.get("swa_redirection_available", False)) and bool(swa_redirection_mask.any()):
                base_positive_outside = (R_ttt == int(SEMANTIC_ROLE_POSITIVE_LONG)) & patch_mask & (~swa_redirection_mask)
                read_context_outside = read_active & high_trust & (~swa_redirection_mask)
                frame_positive_outside = (R_frame == int(SEMANTIC_ROLE_POSITIVE_LONG)) & patch_mask & (~swa_redirection_mask)
                global_positive_outside = (R_global == int(SEMANTIC_ROLE_POSITIVE_LONG)) & patch_mask & (~swa_redirection_mask)
                R_ttt[base_positive_outside | read_context_outside] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
                if swa_source_boost_policy:
                    R_frame[frame_positive_outside] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
                    R_global[global_positive_outside] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
                R_ttt[swa_redirection_mask] = int(SEMANTIC_ROLE_POSITIVE_LONG)
                R_frame[swa_redirection_mask] = int(SEMANTIC_ROLE_POSITIVE_LONG)
                R_global[swa_redirection_mask] = int(SEMANTIC_ROLE_POSITIVE_LONG)
                R_swa[swa_redirection_mask] = int(SEMANTIC_ROLE_PROTECT_NEUTRAL)
                swa_redirection_debug.update({
                    "swa_redirection_policy_applied": True,
                    "swa_redirection_source_boost_policy": bool(swa_source_boost_policy),
                    "swa_redirection_positive_count": int((swa_redirection_mask & (R_ttt == int(SEMANTIC_ROLE_POSITIVE_LONG))).sum().item()),
                    "swa_redirection_neutralized_base_positive_count": int(base_positive_outside.sum().item()),
                    "swa_redirection_neutralized_read_context_count": int(read_context_outside.sum().item()),
                    "swa_redirection_neutralized_frame_positive_count": int(frame_positive_outside.sum().item()) if swa_source_boost_policy else 0,
                    "swa_redirection_neutralized_global_positive_count": int(global_positive_outside.sum().item()) if swa_source_boost_policy else 0,
                })

        read_active_mask = read_active & patch_mask
        if swa_redirection_policy:
            read_stable_proxy = swa_redirection_mask & patch_mask
        elif policy == "fine_ttt_read_active_key_stable_positive_context_neutral":
            read_stable_proxy = read_active_mask & key_stable & high_trust
        elif policy == "fine_ttt_read_active_key_stable_lowd_positive_context_neutral":
            read_stable_proxy = read_active_mask & key_stable & low_d & high_trust
        else:
            read_stable_proxy = read_active_mask & structure & low_d & high_trust
        read_harm_proxy = read_active_mask & high_d & high_trust
        read_active_positive = read_active_mask & (R_ttt == int(SEMANTIC_ROLE_POSITIVE_LONG))
        read_active_neutral = read_active_mask & (
            (R_ttt == int(SEMANTIC_ROLE_NEUTRAL_KEEP))
            | (R_ttt == int(SEMANTIC_ROLE_PROTECT_NEUTRAL))
        )
        read_active_negative = read_active_mask & (R_ttt == int(SEMANTIC_ROLE_NEGATIVE_SHORT))
        read_stable_positive = read_stable_proxy & (R_ttt == int(SEMANTIC_ROLE_POSITIVE_LONG))
        read_harm_negative = read_harm_proxy & (R_ttt == int(SEMANTIC_ROLE_NEGATIVE_SHORT))
        read_harm_no_persistent = read_harm_proxy & (
            (R_ttt == int(SEMANTIC_ROLE_NEGATIVE_SHORT))
            | (R_ttt == int(SEMANTIC_ROLE_NEUTRAL_KEEP))
            | (R_ttt == int(SEMANTIC_ROLE_PROTECT_NEUTRAL))
        )
        read_active_regime_guard_neutral = (
            read_active_mask & high_trust & (R_ttt == int(SEMANTIC_ROLE_NEUTRAL_KEEP))
            if policy == "fine_ttt_read_active_regime_guarded_pos_neutral"
            else torch.zeros_like(patch_mask, dtype=torch.bool)
        )
        read_active_key_context_neutral = (
            read_active_mask & high_trust & (R_ttt == int(SEMANTIC_ROLE_NEUTRAL_KEEP))
            if policy in {
                "fine_ttt_read_active_key_stable_positive_context_neutral",
                "fine_ttt_read_active_key_stable_lowd_positive_context_neutral",
            }
            else torch.zeros_like(patch_mask, dtype=torch.bool)
        )
        if policy == "fine_ttt_read_active_structure_positive_context_neutral":
            read_active_context_neutral = read_active_mask & high_trust & (~(read_active & structure & high_trust))
        elif swa_redirection_policy:
            read_active_context_neutral = (
                read_active_mask
                & high_trust
                & (~swa_redirection_mask)
                & (R_ttt == int(SEMANTIC_ROLE_NEUTRAL_KEEP))
            )
        elif policy in {
            "fine_ttt_read_active_key_stable_positive_context_neutral",
            "fine_ttt_read_active_key_stable_lowd_positive_context_neutral",
        }:
            read_active_context_neutral = read_active_key_context_neutral
        else:
            read_active_context_neutral = read_active_regime_guard_neutral

        def _count(mask: torch.Tensor) -> int:
            return int(mask.sum().item())

        def _rate(num: torch.Tensor, den: torch.Tensor) -> Optional[float]:
            den_count = _count(den)
            if den_count <= 0:
                return None
            return float(_count(num) / max(den_count, 1))

        read_ttt_alignment_terms = [
            value
            for value in (
                _rate(read_stable_positive, read_stable_proxy),
                _rate(read_harm_no_persistent, read_harm_proxy),
            )
            if value is not None
        ]
        read_ttt_alignment_score = (
            float(sum(read_ttt_alignment_terms) / len(read_ttt_alignment_terms))
            if read_ttt_alignment_terms
            else None
        )
        read_ttt_role_alignment = {
            "schema": "acl2_v79_read_ttt_role_alignment_log_v1",
            "source": "in_controller_runtime_role_intersection",
            "note": (
                (
                    "SWA-redirection policy uses external Phase3 SWA stable-positive top-quantile support "
                    "as the stable proxy; harm remains READ active high-D high-trust."
                )
                if swa_redirection_policy
                else (
                    "READ stable/harm are runtime proxies: stable=READ active structure low-D high-trust; "
                    "harm=READ active high-D high-trust."
                )
            ),
            "policy": policy,
            "read_active_count": _count(read_active_mask),
            "read_stable_proxy_count": _count(read_stable_proxy),
            "read_harm_proxy_count": _count(read_harm_proxy),
            "read_active_positive_count": _count(read_active_positive),
            "read_active_neutral_count": _count(read_active_neutral),
            "read_active_negative_count": _count(read_active_negative),
            "read_active_positive_rate": _rate(read_active_positive, read_active_mask),
            "read_active_neutral_rate": _rate(read_active_neutral, read_active_mask),
            "read_active_negative_rate": _rate(read_active_negative, read_active_mask),
            "read_stable_positive_count": _count(read_stable_positive),
            "read_harm_negative_count": _count(read_harm_negative),
            "read_harm_no_persistent_count": _count(read_harm_no_persistent),
            "read_stable_positive_rate": _rate(read_stable_positive, read_stable_proxy),
            "read_harm_negative_rate": _rate(read_harm_negative, read_harm_proxy),
            "read_harm_no_persistent_rate": _rate(read_harm_no_persistent, read_harm_proxy),
            "alignment_score": read_ttt_alignment_score,
        }

        debug = {
            "runtime_fine_role_policy_available": True,
            "fine_label_token_count": int(known.sum().item()),
            "fine_label_counts": self._label_counts(L[patch_mask]),
            "condition_signal_D_available": True,
            "condition_signal_Q_available": True,
            "condition_signal_conflict_available": bool(conflict_available),
            "condition_signal_scale_risk_available": bool(scale_available),
            "condition_signal_conflict_level": conflict_level,
            "condition_signal_scale_risk_level": scale_level,
            "condition_signal_conflict_token_exact": bool(conflict_token_exact),
            "condition_signal_scale_risk_token_exact": bool(scale_token_exact),
            "condition_signal_conflict_token_nonempty": bool(conflict_tok is not None and int(conflict_tok.numel()) > 0),
            "condition_signal_scale_risk_token_nonempty": bool(scale_tok is not None and int(scale_tok.numel()) > 0),
            "condition_signal_conflict_token_count": int(conflict_tok.numel()) if conflict_tok is not None else 0,
            "condition_signal_scale_risk_token_count": int(scale_tok.numel()) if scale_tok is not None else 0,
            "condition_signal_conflict_token_mean": float(C[patch_mask].mean().item()) if conflict_available and bool(patch_mask.any()) else 0.0,
            "condition_signal_conflict_token_p90": float(torch.quantile(C[patch_mask], 0.90).item()) if conflict_available and bool(patch_mask.any()) else 0.0,
            "condition_signal_scale_risk_token_mean": float(S[patch_mask].mean().item()) if scale_available and bool(patch_mask.any()) else 0.0,
            "condition_signal_scale_risk_token_p90": float(torch.quantile(S[patch_mask], 0.90).item()) if scale_available and bool(patch_mask.any()) else 0.0,
            "condition_signal_conflict_source": str(getattr(self, "semantic_condition_conflict_source", "none")),
            "condition_signal_scale_risk_source": str(getattr(self, "semantic_condition_scale_source", "none")),
            "condition_signal_conflict_value": float(conflict_value),
            "condition_signal_scale_risk_value": float(scale_value),
            "condition_signal_conflict_summary": dict(getattr(self, "semantic_condition_conflict_summary", {}) or {}),
            "condition_signal_scale_risk_summary": dict(getattr(self, "semantic_condition_scale_summary", {}) or {}),
            "condition_fallback_note": (
                "Risk condition levels and token_exact flags are recorded so "
                "broadcast/proxy signals are not counted as stronger evidence."
            ),
            "fine_role_policy_variant": policy,
            "read_aligned_ttt_policy_applied": bool(read_aligned_ttt_policy),
            "read_aligned_ttt_read_active_count": int((read_active & patch_mask).sum().item()),
            "read_aligned_ttt_read_active_structure_positive_count": int(
                (
                    read_active_positive
                    if read_aligned_ttt_policy
                    else (
                        (read_active & high_trust)
                        if policy == "fine_ttt_read_active_all_positive"
                        else (read_active & structure & high_trust)
                    )
                ).sum().item()
            ),
            "read_aligned_ttt_read_active_stable_structure_positive_count": int(
                (read_stable_proxy & (R_ttt == int(SEMANTIC_ROLE_POSITIVE_LONG))).sum().item()
            ),
            "read_aligned_ttt_read_active_regime_guard_neutral_count": int(
                read_active_regime_guard_neutral.sum().item()
            ),
            "read_aligned_ttt_read_active_regime_highd_count": int(
                (read_active_mask & high_d & high_trust).sum().item()
            ),
            "read_aligned_ttt_key_stability_available": bool(key_stability_available),
            "read_aligned_ttt_key_stability_threshold": key_stability_thr,
            "read_aligned_ttt_read_active_key_stable_count": int(
                (read_active_mask & key_stable & high_trust).sum().item()
            ),
            "read_aligned_ttt_read_active_key_stable_positive_count": int(
                (read_active_mask & key_stable & high_trust & (R_ttt == int(SEMANTIC_ROLE_POSITIVE_LONG))).sum().item()
            ),
            "read_aligned_ttt_read_active_key_stable_lowd_count": int(
                (read_active_mask & key_stable & low_d & high_trust).sum().item()
            ),
            "read_aligned_ttt_read_active_key_stable_lowd_positive_count": int(
                (
                    read_active_mask
                    & key_stable
                    & low_d
                    & high_trust
                    & (R_ttt == int(SEMANTIC_ROLE_POSITIVE_LONG))
                ).sum().item()
            ),
            "read_aligned_ttt_read_active_key_context_neutral_count": int(
                read_active_key_context_neutral.sum().item()
            ),
            "READ_TTT_ROLE_ALIGNMENT_LOG": read_ttt_role_alignment,
            "read_ttt_role_alignment": read_ttt_role_alignment,
            "read_ttt_alignment_score": read_ttt_alignment_score,
            "prior_read_ttt_role_alignment": read_ttt_role_alignment,
            "prior_read_ttt_alignment_score": read_ttt_alignment_score,
            "read_aligned_ttt_read_active_structure_positive_count_legacy": int(
                (
                    (read_active & high_trust)
                    if policy == "fine_ttt_read_active_all_positive"
                    else (read_active & structure & high_trust)
                ).sum().item()
            ),
            "read_aligned_ttt_read_active_context_neutral_count": int(
                read_active_context_neutral.sum().item()
            ),
            "read_aligned_ttt_read_active_highd_context_negative_count": int(
                (read_active & high_d & high_trust & (sky | vegetation | movable)).sum().item()
                if policy
                not in {
                    "fine_ttt_read_active_structure_positive_context_neutral",
                    "fine_ttt_read_active_all_positive",
                }
                else 0
            ),
            "R_frame_role_counts": self._role_counts(R_frame[patch_mask]),
            "R_global_role_counts": self._role_counts(R_global[patch_mask]),
            "R_swa_role_counts": self._role_counts(R_swa[patch_mask]),
            "R_ttt_role_counts": self._role_counts(R_ttt[patch_mask]),
            "fine_label_condition_metrics": self._label_condition_metrics(
                L,
                patch_mask,
                D=D,
                Q=Q,
                C=C if conflict_available else None,
                S=S if scale_available else None,
            ),
            "fine_label_path_role_counts": {
                "frame": self._label_role_counts(L, R_frame, patch_mask),
                "global": self._label_role_counts(L, R_global, patch_mask),
                "swa": self._label_role_counts(L, R_swa, patch_mask),
                "ttt": self._label_role_counts(L, R_ttt, patch_mask),
            },
        }
        debug.update(swa_redirection_debug)
        return R_frame, R_global, R_swa, R_ttt, debug

    @staticmethod
    def _swa_redirection_source_mask_from_roles(
        R_frame_tok: Optional[torch.Tensor],
        R_global_tok: Optional[torch.Tensor],
        role_debug: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        if not (
            bool(role_debug.get("swa_redirection_policy_applied", False))
            and bool(role_debug.get("swa_redirection_source_boost_policy", False))
        ):
            return None
        if R_frame_tok is None:
            role_debug["swa_redirection_source_mask_reason"] = "missing_R_frame_tok"
            return None
        frame_pos = (
            R_frame_tok.detach().cpu().long().reshape(-1)
            == int(SEMANTIC_ROLE_POSITIVE_LONG)
        )
        if R_global_tok is not None:
            global_pos = (
                R_global_tok.detach().cpu().long().reshape(-1)
                == int(SEMANTIC_ROLE_POSITIVE_LONG)
            )
            if int(global_pos.numel()) == int(frame_pos.numel()):
                source_mask = frame_pos & global_pos
                role_debug["swa_redirection_source_mask_global_positive_count"] = int(global_pos.sum().item())
                role_debug["swa_redirection_source_mask_frame_global_match"] = bool(torch.equal(frame_pos, global_pos))
            else:
                source_mask = frame_pos
                role_debug["swa_redirection_source_mask_frame_global_match"] = False
                role_debug["swa_redirection_source_mask_reason"] = "R_global_tok_shape_mismatch"
        else:
            source_mask = frame_pos
            role_debug["swa_redirection_source_mask_frame_global_match"] = None
        count = int(source_mask.sum().item())
        expected = role_debug.get("swa_redirection_selected_tokens")
        expected_int = int(expected) if expected is not None else None
        role_debug.update({
            "swa_redirection_source_mask_available": True,
            "swa_redirection_source_mask_token_count": count,
            "swa_redirection_source_mask_matches_selected": (
                bool(expected_int == count) if expected_int is not None else None
            ),
        })
        return source_mask

    def _apply_semantic_role_policy(
        self,
        *,
        token_type: torch.Tensor,
        D_tok: torch.Tensor,
        P_ref: torch.Tensor,
        V_sem_tok: Optional[torch.Tensor],
        G_sem_tok: Optional[torch.Tensor],
        Q_sem_tok: Optional[torch.Tensor],
        L_sem_tok: Optional[torch.Tensor],
        R_sem_tok: Optional[torch.Tensor],
        R_frame_tok: Optional[torch.Tensor],
        R_global_tok: Optional[torch.Tensor],
        R_swa_tok: Optional[torch.Tensor],
        R_ttt_tok: Optional[torch.Tensor],
        P_ttt_write: Optional[torch.Tensor],
        D_swa_write_tok: Optional[torch.Tensor],
        read_patch_tok: Optional[torch.Tensor] = None,
        key_stability_tok: Optional[torch.Tensor] = None,
        num_frames: int = 0,
        patch_grid: Tuple[int, int] = (0, 0),
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, Any],
    ]:
        """Refine semantic roles with dynamic risk and route them to memory paths."""

        policy = str(self.semantic_role_policy or "none").strip().lower()
        paths = {
            p.strip().lower()
            for p in str(self.semantic_memory_paths or "").replace(";", ",").split(",")
            if p.strip()
        }
        paths_requested = sorted(paths)
        if policy in {"", "none", "off", "noop"}:
            debug = {
                "semantic_role_policy": policy or "none",
                "semantic_memory_paths": paths_requested,
                "semantic_role_consumed_any": False,
                "semantic_role_reason": "disabled",
            }
            return R_sem_tok, R_frame_tok, R_global_tok, R_swa_tok, R_ttt_tok, P_ttt_write, D_swa_write_tok, debug

        patch_mask = token_type.detach().cpu().long() == TOKEN_TYPE_PATCH
        if G_sem_tok is None or D_tok is None or not bool(patch_mask.any()):
            debug = {
                "semantic_role_policy": policy,
                "semantic_memory_paths": paths_requested,
                "semantic_role_consumed_any": False,
                "semantic_role_reason": "missing_semantic_or_dynamic_tokens",
            }
            return R_sem_tok, R_frame_tok, R_global_tok, R_swa_tok, R_ttt_tok, P_ttt_write, D_swa_write_tok, debug

        G = G_sem_tok.detach().cpu().long().reshape(-1)
        D = D_tok.detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
        L = (
            L_sem_tok.detach().cpu().long().reshape(-1)
            if L_sem_tok is not None
            else torch.full_like(G, int(SEMANTIC_FINE_LABEL_UNKNOWN))
        )
        Q = (
            Q_sem_tok.detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
            if Q_sem_tok is not None
            else torch.ones_like(D)
        )
        if G.numel() != D.numel() or L.numel() != D.numel():
            debug = {
                "semantic_role_policy": policy,
                "semantic_memory_paths": paths_requested,
                "semantic_role_consumed_any": False,
                "semantic_role_reason": "shape_mismatch",
                "semantic_group_tokens": int(G.numel()),
                "semantic_label_tokens": int(L.numel()),
                "dynamic_tokens": int(D.numel()),
            }
            return R_sem_tok, R_frame_tok, R_global_tok, R_swa_tok, R_ttt_tok, P_ttt_write, D_swa_write_tok, debug

        patch_scores = D[patch_mask]
        if patch_scores.numel() > 0:
            q = min(max(float(self.semantic_role_highd_quantile), 0.0), 1.0)
            high_thr = float(torch.quantile(patch_scores, q).item())
        else:
            high_thr = 1.0
        high_d = D >= high_thr
        low_d = ~high_d
        high_trust = Q >= float(self.semantic_role_low_trust)

        path_role_override_policy = policy in {"v29c_masklet_override", "v36_synthetic_override"}
        fine_policy = (
            policy.startswith("fine_")
            or policy.startswith("causal_")
            or policy.startswith("v76_")
            or path_role_override_policy
        )
        fine_debug: Dict[str, Any] = {}
        if path_role_override_policy:
            def _valid_or_fallback(stream: Optional[torch.Tensor]) -> torch.Tensor:
                if stream is None:
                    return torch.full_like(G, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
                r = stream.detach().cpu().long().reshape(-1)
                if int(r.numel()) != int(G.numel()):
                    return torch.full_like(G, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
                return r.clone()

            R_frame = _valid_or_fallback(R_frame_tok)
            R_global = _valid_or_fallback(R_global_tok)
            R_swa = _valid_or_fallback(R_swa_tok)
            R_ttt = _valid_or_fallback(R_ttt_tok)
            R = R_ttt.clone()
            fine_debug = {
                "runtime_fine_role_policy_available": True,
                "fine_role_policy_variant": policy,
                "v29c_path_role_override_consumed": True,
                "fine_label_token_count": int(((L != int(SEMANTIC_FINE_LABEL_UNKNOWN)) & (L >= 0) & patch_mask).sum().item()),
                "fine_label_counts": self._label_counts(L[patch_mask]),
                "fine_label_path_role_counts": {
                    "frame": self._label_role_counts(L, R_frame, patch_mask),
                    "global": self._label_role_counts(L, R_global, patch_mask),
                    "swa": self._label_role_counts(L, R_swa, patch_mask),
                    "ttt": self._label_role_counts(L, R_ttt, patch_mask),
                },
            }
        elif fine_policy:
            R_frame, R_global, R_swa, R_ttt, fine_debug = self._fine_path_roles(
                G=G,
                L=L,
                D=D,
                Q=Q,
                patch_mask=patch_mask,
                high_d=high_d,
                low_d=low_d,
                high_trust=high_trust,
                read_patch_tok=read_patch_tok,
                key_stability_tok=key_stability_tok,
                num_frames=num_frames,
                patch_grid=patch_grid,
            )
            # Shared role is retained for legacy summaries. Model hook sites
            # consume the path-specific roles where available.
            R = R_ttt.clone()
        else:
            R = torch.full_like(G, int(SEMANTIC_ROLE_FALLBACK), dtype=torch.long)
            structure = (G == SEMANTIC_GROUP_STRUCTURE_ANCHOR) & patch_mask
            static = (G == SEMANTIC_GROUP_STATIC_THING) & patch_mask
            movable = (G == SEMANTIC_GROUP_MOVABLE_THING) & patch_mask
            lowstuff = (G == SEMANTIC_GROUP_LOW_VALUE_STUFF) & patch_mask
            uncertain = (G == SEMANTIC_GROUP_UNCERTAIN_REGION) & patch_mask

            R[structure & low_d & high_trust] = int(SEMANTIC_ROLE_POSITIVE_LONG)
            R[structure & ~(low_d & high_trust)] = int(SEMANTIC_ROLE_PROTECT_NEUTRAL)
            R[static & low_d & high_trust] = int(SEMANTIC_ROLE_POSITIVE_LONG)
            R[static & ~(low_d & high_trust)] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            R[lowstuff & low_d & high_trust] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            R[lowstuff & high_d & high_trust] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
            R[movable & high_d & high_trust] = int(SEMANTIC_ROLE_NEGATIVE_SHORT)
            R[movable & ~(high_d & high_trust)] = int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            R[uncertain | (~high_trust & patch_mask)] = int(SEMANTIC_ROLE_FALLBACK)
            R_frame = R.clone()
            R_global = R.clone()
            R_swa = R.clone()
            R_ttt = R.clone()

        frame_on = self._semantic_path_enabled("frame")
        global_on = self._semantic_path_enabled("global") or self._semantic_path_enabled("chunk")
        swa_on = self._semantic_path_enabled("swa")
        ttt_on = self._semantic_path_enabled("ttt")
        lifecycle_on = self._semantic_path_enabled("lifecycle")

        role_control_debug: Dict[str, Any] = {
            "semantic_role_control_mode": str(getattr(self, "semantic_role_control_mode", "none") or "none").strip().lower(),
            "semantic_role_control_seed": int(getattr(self, "semantic_role_control_seed", 12345)),
            "semantic_role_control_scope": "ttt_patch_roles",
            "semantic_role_control_applied": False,
            "semantic_role_control_reason": "ttt_path_disabled_or_missing_write_prior",
            "R_ttt_role_counts_before_control": self._role_counts(R_ttt[patch_mask]),
            "R_ttt_role_counts_after_control": self._role_counts(R_ttt[patch_mask]),
            "semantic_role_control_changed_fraction": 0.0,
        }
        swa_role_control_debug: Dict[str, Any] = {
            "semantic_swa_role_control_mode": str(getattr(self, "semantic_role_control_mode", "none") or "none").strip().lower(),
            "semantic_swa_role_control_seed": int(getattr(self, "semantic_role_control_seed", 12345)),
            "semantic_swa_role_control_scope": "swa_patch_roles",
            "semantic_swa_role_control_applied": False,
            "semantic_swa_role_control_reason": "swa_path_disabled",
            "R_swa_role_counts_before_control": self._role_counts(R_swa[patch_mask]),
            "R_swa_role_counts_after_control": self._role_counts(R_swa[patch_mask]),
            "semantic_swa_role_control_changed_fraction": 0.0,
        }
        if (ttt_on or lifecycle_on) and P_ttt_write is not None:
            R_ttt, role_control_debug = self._apply_ttt_semantic_role_control(
                R_ttt,
                G=G,
                L=L,
                D=D,
                Q=Q,
                patch_mask=patch_mask,
            )
            R = R_ttt.clone()
        if swa_on:
            R_swa, swa_role_control_debug = self._apply_swa_semantic_role_control(
                R_swa,
                G=G,
                L=L,
                D=D,
                Q=Q,
                patch_mask=patch_mask,
            )

        ttt_write_adjust_debug: Dict[str, Any] = {
            "ttt_semantic_write_role_stats": {},
            "ttt_semantic_write_role_max_abs_rel_change": 0.0,
            "ttt_semantic_write_role_intended_change_ge20": False,
            "semantic_role_ttt_read_harm_veto_mode": str(
                getattr(self, "semantic_role_ttt_read_harm_veto_mode", "none") or "none"
            ),
            "semantic_role_ttt_read_harm_veto_applied": False,
            "semantic_role_ttt_read_harm_veto_reason": "ttt_path_disabled_or_missing_write_prior",
        }
        P_out = P_ttt_write
        if (ttt_on or lifecycle_on) and P_ttt_write is not None:
            P_before = P_ttt_write.detach().cpu().float().reshape(-1).clone()
            P = P_before.clone()
            pos = R_ttt == int(SEMANTIC_ROLE_POSITIVE_LONG)
            neu = R_ttt == int(SEMANTIC_ROLE_NEUTRAL_KEEP)
            neg = R_ttt == int(SEMANTIC_ROLE_NEGATIVE_SHORT)
            P[pos] = (P[pos] * float(self.semantic_role_positive_scale)).clamp(0.0, 1.5)
            P[neu] = (P[neu] * float(self.semantic_role_neutral_scale)).clamp(0.0, 1.5)
            P[neg] = (P[neg] * float(self.semantic_role_negative_scale)).clamp(0.0, 1.5)
            P_after_role = P.clone()
            veto_mode = str(getattr(self, "semantic_role_ttt_read_harm_veto_mode", "none") or "none").strip().lower()
            veto_scale = max(0.0, min(1.5, float(getattr(self, "semantic_role_ttt_read_harm_veto_scale", 0.0))))
            veto_min_harm = max(0, int(getattr(self, "semantic_role_ttt_read_harm_veto_min_harm", 1)))
            veto_max_stable_positive = max(
                0,
                int(getattr(self, "semantic_role_ttt_read_harm_veto_max_stable_positive", 0)),
            )
            read_active = torch.zeros_like(patch_mask, dtype=torch.bool)
            if read_patch_tok is not None:
                read_values = read_patch_tok.detach().cpu().float().reshape(-1)
                if int(read_values.numel()) == int(D.numel()):
                    read_active = (read_values > 0.50) & patch_mask
            read_harm_proxy = read_active & high_d & high_trust
            read_stable_positive = read_active & low_d & high_trust & pos
            read_harm_no_persistent = read_harm_proxy & (neg | neu | (R_ttt == int(SEMANTIC_ROLE_PROTECT_NEUTRAL)))
            veto_mask = read_harm_no_persistent
            if veto_mode == "read_harm_negative_no_stable":
                veto_mask = read_harm_proxy & neg
            elif veto_mode in {"read_active_no_stable", "read_active_harm_no_stable"}:
                veto_mask = read_active & high_trust
            read_harm_count = int(read_harm_no_persistent.sum().item())
            read_stable_positive_count = int(read_stable_positive.sum().item())
            veto_candidate_count = int((veto_mask & patch_mask).sum().item())
            veto_applied = False
            veto_reason = "disabled"
            veto_before_mean = None
            veto_after_mean = None
            if veto_mode not in {"", "none", "off", "noop"}:
                if veto_mode not in {
                    "read_harm_no_stable",
                    "read_harm_negative_no_stable",
                    "read_active_no_stable",
                    "read_active_harm_no_stable",
                }:
                    veto_reason = f"unknown_mode:{veto_mode}"
                elif read_harm_count < veto_min_harm:
                    veto_reason = "below_min_read_harm"
                elif read_stable_positive_count > veto_max_stable_positive:
                    veto_reason = "stable_positive_available"
                elif veto_candidate_count <= 0:
                    veto_reason = "empty_veto_mask"
                else:
                    active_veto = veto_mask & patch_mask
                    veto_before_mean = float(P_after_role[active_veto].mean().item())
                    P[active_veto] = (P[active_veto] * veto_scale).clamp(0.0, 1.5)
                    veto_after_mean = float(P[active_veto].mean().item())
                    veto_applied = True
                    veto_reason = "read_harm_without_stable_positive"
            P_out = P
            role_specs = (
                ("positive", pos, float(self.semantic_role_positive_scale)),
                ("neutral", neu, float(self.semantic_role_neutral_scale)),
                ("negative", neg, float(self.semantic_role_negative_scale)),
            )
            role_stats: Dict[str, Any] = {}
            max_abs_rel = 0.0
            for name, mask, scale_value in role_specs:
                mask = mask & patch_mask
                count = int(mask.sum().item())
                before_mean = float(P_before[mask].mean().item()) if count > 0 else None
                after_mean = float(P[mask].mean().item()) if count > 0 else None
                rel_change = (
                    float((after_mean - before_mean) / max(abs(before_mean), 1e-8))
                    if before_mean is not None and after_mean is not None
                    else None
                )
                if rel_change is not None:
                    max_abs_rel = max(max_abs_rel, abs(float(rel_change)))
                role_stats[name] = {
                    "count": count,
                    "scale": scale_value,
                    "before_mean": before_mean,
                    "after_mean": after_mean,
                    "relative_change": rel_change,
                }
            ttt_write_adjust_debug = {
                "ttt_semantic_write_role_stats": role_stats,
                "ttt_semantic_write_role_max_abs_rel_change": float(max_abs_rel),
                "ttt_semantic_write_role_intended_change_ge20": bool(max_abs_rel >= 0.20),
                "semantic_role_ttt_read_harm_veto_mode": veto_mode or "none",
                "semantic_role_ttt_read_harm_veto_scale": float(veto_scale),
                "semantic_role_ttt_read_harm_veto_min_harm": int(veto_min_harm),
                "semantic_role_ttt_read_harm_veto_max_stable_positive": int(veto_max_stable_positive),
                "semantic_role_ttt_read_harm_veto_applied": bool(veto_applied),
                "semantic_role_ttt_read_harm_veto_reason": veto_reason,
                "semantic_role_ttt_read_harm_veto_read_active_count": int(read_active.sum().item()),
                "semantic_role_ttt_read_harm_veto_read_harm_no_persistent_count": int(read_harm_count),
                "semantic_role_ttt_read_harm_veto_read_stable_positive_count": int(read_stable_positive_count),
                "semantic_role_ttt_read_harm_veto_candidate_count": int(veto_candidate_count),
                "semantic_role_ttt_read_harm_veto_before_mean": veto_before_mean,
                "semantic_role_ttt_read_harm_veto_after_mean": veto_after_mean,
            }

        D_swa_out = D_swa_write_tok
        swa_adjust_debug: Dict[str, Any] = {
            "semantic_role_swa_negative_scale": float(self.semantic_role_swa_negative_scale),
            "semantic_role_swa_protect_scale": float(getattr(self, "semantic_role_swa_protect_scale", 1.0)),
            "semantic_role_swa_negative_count": int(((R_swa == int(SEMANTIC_ROLE_NEGATIVE_SHORT)) & patch_mask).sum().item()),
            "semantic_role_swa_protect_count": int(((R_swa == int(SEMANTIC_ROLE_PROTECT_NEUTRAL)) & patch_mask).sum().item()),
            "semantic_role_swa_protect_adjusted": False,
            "semantic_role_swa_score_before_mean": None,
            "semantic_role_swa_score_after_mean": None,
            "semantic_role_swa_score_protect_before_mean": None,
            "semantic_role_swa_score_protect_after_mean": None,
            "semantic_role_swa_score_negative_after_mean": None,
        }
        if swa_on:
            base = (
                D_swa_write_tok.detach().cpu().float().reshape(-1).clone()
                if D_swa_write_tok is not None
                else torch.zeros_like(D)
            )
            neg_score = (R_swa == int(SEMANTIC_ROLE_NEGATIVE_SHORT)).float() * float(self.semantic_role_swa_negative_scale)
            D_swa_out = torch.maximum(base, neg_score).clamp(0.0, 1.0)
            protect = (R_swa == int(SEMANTIC_ROLE_PROTECT_NEUTRAL)) & patch_mask
            protect_scale = min(max(float(getattr(self, "semantic_role_swa_protect_scale", 1.0)), 0.0), 1.0)
            before = D_swa_out.clone()
            if protect_scale < 1.0 and bool(protect.any()):
                D_swa_out[protect] = (D_swa_out[protect] * protect_scale).clamp(0.0, 1.0)
            neg = (R_swa == int(SEMANTIC_ROLE_NEGATIVE_SHORT)) & patch_mask
            swa_adjust_debug.update({
                "semantic_role_swa_negative_scale": float(self.semantic_role_swa_negative_scale),
                "semantic_role_swa_protect_scale": float(protect_scale),
                "semantic_role_swa_protect_adjusted": bool(protect_scale < 1.0 and bool(protect.any())),
                "semantic_role_swa_score_before_mean": float(before[patch_mask].mean().item()) if bool(patch_mask.any()) else 0.0,
                "semantic_role_swa_score_after_mean": float(D_swa_out[patch_mask].mean().item()) if bool(patch_mask.any()) else 0.0,
                "semantic_role_swa_score_protect_before_mean": float(before[protect].mean().item()) if bool(protect.any()) else None,
                "semantic_role_swa_score_protect_after_mean": float(D_swa_out[protect].mean().item()) if bool(protect.any()) else None,
                "semantic_role_swa_score_negative_after_mean": float(D_swa_out[neg].mean().item()) if bool(neg.any()) else None,
            })

        V = (
            V_sem_tok.detach().cpu().float().reshape(-1).clamp(0.0, 1.0)
            if V_sem_tok is not None and int(V_sem_tok.numel()) == int(G.numel())
            else torch.ones_like(D)
        )
        groups = sorted({int(x.item()) for x in torch.unique(G[patch_mask])}) if bool(patch_mask.any()) else []
        per_group: Dict[str, Dict[str, Any]] = {}
        for group_id in groups:
            mask = (G == group_id) & patch_mask
            role_counts = self._role_counts(R[mask])
            per_group[str(group_id)] = {
                "token_count": int(mask.sum().item()),
                "D_mean": float(D[mask].mean().item()) if bool(mask.any()) else 0.0,
                "D_p90": float(torch.quantile(D[mask], 0.90).item()) if int(mask.sum().item()) > 1 else 0.0,
                "Q_mean": float(Q[mask].mean().item()) if bool(mask.any()) else 0.0,
                "V_mean": float(V[mask].mean().item()) if bool(mask.any()) else 0.0,
                "role_counts": role_counts,
            }

        debug = {
            "semantic_role_policy": policy,
            "semantic_memory_paths": paths_requested,
            "semantic_role_consumed_any": bool(frame_on or global_on or swa_on or ttt_on or lifecycle_on),
            "path_specific_role_streams_available": bool(fine_policy),
            "runtime_fine_role_policy_available": bool(fine_policy),
            "frame_semantic_source_consumed": bool(frame_on),
            "chunk_global_semantic_source_consumed": bool(global_on),
            "swa_semantic_source_consumed": bool(swa_on),
            "ttt_semantic_role_consumed": bool(ttt_on),
            "lifecycle_semantic_role_consumed": bool(lifecycle_on),
            "semantic_role_highd_quantile": float(self.semantic_role_highd_quantile),
            "semantic_role_highd_threshold": float(high_thr),
            "semantic_role_low_trust": float(self.semantic_role_low_trust),
            "semantic_role_counts": self._role_counts(R[patch_mask]),
            "R_frame_role_counts": self._role_counts(R_frame[patch_mask]),
            "R_global_role_counts": self._role_counts(R_global[patch_mask]),
            "R_swa_role_counts": self._role_counts(R_swa[patch_mask]),
            "R_ttt_role_counts": self._role_counts(R_ttt[patch_mask]),
            "semantic_group_role_metrics": per_group,
            "semantic_role_ttt_adjusted": bool((ttt_on or lifecycle_on) and P_ttt_write is not None),
            "semantic_role_swa_adjusted": bool(swa_on),
        }
        debug.update(fine_debug)
        debug.update(role_control_debug)
        debug.update(swa_role_control_debug)
        debug.update(swa_adjust_debug)
        debug.update(ttt_write_adjust_debug)
        return R, R_frame, R_global, R_swa, R_ttt, P_out, D_swa_out, debug

    def _apply_swa_history_write_gate(
        self,
        history: Optional[List[Optional[Dict[str, torch.Tensor]]]],
        control_prior: Optional[HybridMemoryControlPrior],
        geo: GeometryOutput,
    ) -> Tuple[Optional[List[Optional[Dict[str, torch.Tensor]]]], Dict[str, Any]]:
        """Gate SWA KV history before it is committed for the next chunk."""
        mode = str(self.swa_write_mode)
        keep_scope = str(self.swa_write_keep_scope or "all")
        wants_gate = mode not in {"", "none", "off"}
        wants_keep = keep_scope not in {"", "all", "full"}
        cache_blend_alpha = max(float(self.swa_write_cache_blend_alpha), 0.0)
        wants_cache_blend = cache_blend_alpha > 0.0
        debug: Dict[str, Any] = {
            "swa_write_enabled": bool(self.enable_swa_write_control),
            "swa_write_mode": mode,
            "swa_write_score_source": str(self.swa_write_score_source),
            "swa_write_rho": float(self.swa_write_rho),
            "swa_write_min_gate": float(self.swa_write_min_gate),
            "swa_write_sparse_ratio": float(self.swa_write_sparse_ratio),
            "swa_write_layer_mode": str(self.swa_write_layer_mode),
            "swa_write_single_layer": int(self.swa_write_single_layer),
            "swa_write_scope": str(self.swa_write_scope),
            "swa_write_keep_scope": keep_scope,
            "swa_write_cache_blend_alpha": cache_blend_alpha,
            "swa_write_cache_blend_mode": str(self.swa_write_cache_blend_mode),
            "swa_write_cache_blend_target": str(self.swa_write_cache_blend_target),
            "swa_write_cache_blend_applied_layers": 0,
            "swa_write_applied_layers": 0,
            "swa_write_history_tokens_before": 0,
            "swa_write_history_tokens_after": 0,
        }
        if (
            not self.enable_swa_write_control
            or (not wants_gate and not wants_keep and not wants_cache_blend)
            or history is None
            or control_prior is None
            or control_prior.D_tok is None
        ):
            return history, debug

        token_type = geo.token_type.detach().cpu().long()
        patch_mask = token_type == TOKEN_TYPE_PATCH
        source_tok = control_prior.D_swa_write_tok if control_prior.D_swa_write_tok is not None else control_prior.D_tok
        D_patch = source_tok.detach().cpu().float().reshape(-1)[patch_mask].clamp(0.0, 1.0)
        if D_patch.numel() == 0:
            return history, debug

        rho = max(float(self.swa_write_rho), 0.0)
        min_gate = min(max(float(self.swa_write_min_gate), 0.0), 1.0)
        per_frame = 0
        overlap_frames = max(int(self.read_overlap_frames), 0)
        if int(geo.num_frames) > 0 and D_patch.numel() % int(geo.num_frames) == 0:
            per_frame = int(D_patch.numel() // int(geo.num_frames))

        def _scope_mask(scope_name: str) -> torch.Tensor:
            scope_norm = str(scope_name or "all")
            exclude = False
            if scope_norm.startswith("exclude_"):
                exclude = True
                scope_norm = scope_norm[len("exclude_"):]
            mask = torch.ones_like(D_patch, dtype=torch.bool)
            if scope_norm in {"all", "full", ""}:
                return ~mask if exclude else mask
            if per_frame <= 0 or overlap_frames <= 0:
                empty = torch.zeros_like(D_patch, dtype=torch.bool)
                return ~empty if exclude else empty
            n = min(int(D_patch.numel()), overlap_frames * per_frame)
            mask = torch.zeros_like(D_patch, dtype=torch.bool)
            if scope_norm in {"tail_overlap", "overlap_tail", "tail"}:
                mask[-n:] = True
            elif scope_norm in {"head_overlap", "overlap_head", "head"}:
                mask[:n] = True
            elif scope_norm in {"both_overlap", "overlap_both"}:
                mask[:n] = True
                mask[-n:] = True
            else:
                raise ValueError(f"Unsupported SWA write scope: {scope_name}")
            return ~mask if exclude else mask

        scope_mask = _scope_mask(str(self.swa_write_scope or "all"))
        keep_mask = _scope_mask(keep_scope)
        D_scope = D_patch[scope_mask]
        if wants_gate and D_scope.numel() == 0:
            return history, debug
        if wants_gate:
            if mode in {"v_centered", "kv_centered", "v_resid_centered", "kv_resid_centered"}:
                static_rank = _centered_percentile_rank((1.0 - D_scope).clamp(0.0, 1.0))
                max_gate = max(1.0, 2.0 - min_gate)
                gate_scope = (1.0 + rho * static_rank).clamp(min_gate, max_gate)
            else:
                gate_scope = (1.0 - rho * D_scope).clamp(min_gate, 1.0)
        else:
            gate_scope = torch.ones_like(D_scope)
        blend_mode = str(self.swa_write_cache_blend_mode or "dynamic").strip().lower()
        if blend_mode in {"dynamic", "dyn", "score", "d"}:
            blend_score_scope = D_scope.clamp(0.0, 1.0)
        elif blend_mode in {"static", "inv", "inverse"}:
            blend_score_scope = (1.0 - D_scope).clamp(0.0, 1.0)
        elif blend_mode in {"all", "flat", "constant"}:
            blend_score_scope = torch.ones_like(D_scope)
        else:
            raise ValueError(f"Unsupported SWA cache blend mode: {self.swa_write_cache_blend_mode}")
        sparse_ratio = min(max(float(self.swa_write_sparse_ratio), 0.0), 1.0)
        selected_mass = 1.0
        if wants_gate and sparse_ratio < 1.0:
            k = max(1, int(round(D_scope.numel() * sparse_ratio)))
            k = min(k, int(D_scope.numel()))
            static_score = (1.0 - D_scope).clamp(0.0, 1.0)
            selected = torch.zeros_like(static_score, dtype=torch.bool)
            selected[torch.topk(static_score, k=k, largest=True).indices] = True
            selected_mass = float(selected.float().mean().item())
            gate_scope = torch.where(selected, gate_scope, torch.full_like(gate_scope, min_gate))
        base_gate = torch.ones_like(D_patch, dtype=torch.float32)
        base_gate[scope_mask] = gate_scope
        base_blend = torch.zeros_like(D_patch, dtype=torch.float32)
        if wants_cache_blend:
            base_blend[scope_mask] = (cache_blend_alpha * blend_score_scope).clamp(0.0, min(cache_blend_alpha, 1.0))

        def _aligned_gate(num_tokens: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            gate = base_gate
            if gate.numel() < num_tokens:
                prefix = torch.ones(num_tokens - gate.numel(), dtype=torch.float32)
                gate = torch.cat([prefix, gate], dim=0)
            elif gate.numel() > num_tokens:
                gate = gate[-num_tokens:]
            return gate.reshape(1, 1, num_tokens, 1).to(device=device, dtype=dtype)

        def _aligned_scope_mask(num_tokens: int, *, device: torch.device) -> torch.Tensor:
            mask = scope_mask
            if mask.numel() < num_tokens:
                prefix = torch.zeros(num_tokens - mask.numel(), dtype=torch.bool)
                mask = torch.cat([prefix, mask], dim=0)
            elif mask.numel() > num_tokens:
                mask = mask[-num_tokens:]
            return mask.to(device=device, dtype=torch.bool)

        def _aligned_blend(num_tokens: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            blend = base_blend
            if blend.numel() < num_tokens:
                prefix = torch.zeros(num_tokens - blend.numel(), dtype=torch.float32)
                blend = torch.cat([prefix, blend], dim=0)
            elif blend.numel() > num_tokens:
                blend = blend[-num_tokens:]
            return blend.reshape(1, 1, num_tokens, 1).to(device=device, dtype=dtype)

        def _aligned_keep_indices(num_tokens: int, *, device: torch.device) -> Optional[torch.Tensor]:
            if not wants_keep:
                return None
            mask = keep_mask
            if mask.numel() < num_tokens:
                prefix = torch.ones(num_tokens - mask.numel(), dtype=torch.bool)
                mask = torch.cat([prefix, mask], dim=0)
            elif mask.numel() > num_tokens:
                mask = mask[-num_tokens:]
            idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                idx = torch.arange(num_tokens, dtype=torch.long)
            return idx.to(device=device, dtype=torch.long)

        gated_history: List[Optional[Dict[str, torch.Tensor]]] = []
        gate_means: List[float] = []
        gate_p10: List[float] = []
        gate_p50: List[float] = []
        gate_p90: List[float] = []
        applied = 0
        cache_blend_applied = 0
        tokens_before: List[int] = []
        tokens_after: List[int] = []
        n_layers = len(history)
        for layer_idx, item in enumerate(history):
            if not self._swa_write_layer_enabled(layer_idx, n_layers):
                gated_history.append(item)
                continue
            if not isinstance(item, dict) or "v" not in item:
                gated_history.append(item)
                continue
            v = item["v"]
            if not torch.is_tensor(v) or v.ndim < 3:
                gated_history.append(item)
                continue
            num_tokens = int(v.shape[2])
            tokens_before.append(num_tokens)
            new_item = dict(item)
            keep_idx = _aligned_keep_indices(num_tokens, device=v.device)
            if keep_idx is not None:
                new_item["v"] = v.index_select(2, keep_idx)
                if torch.is_tensor(item.get("k")):
                    new_item["k"] = item["k"].index_select(2, keep_idx.to(device=item["k"].device))
                if torch.is_tensor(item.get("v_post")):
                    new_item["v_post"] = item["v_post"].index_select(2, keep_idx.to(device=item["v_post"].device))
                if torch.is_tensor(item.get("k_post")):
                    new_item["k_post"] = item["k_post"].index_select(2, keep_idx.to(device=item["k_post"].device))
                if torch.is_tensor(item.get("hidden_pre")):
                    new_item["hidden_pre"] = item["hidden_pre"].index_select(
                        1,
                        keep_idx.to(device=item["hidden_pre"].device),
                    )
                v = new_item["v"]
                num_tokens = int(v.shape[2])
            gate = _aligned_gate(num_tokens, device=v.device, dtype=v.dtype)
            scope_aligned = _aligned_scope_mask(num_tokens, device=v.device)
            blend = _aligned_blend(num_tokens, device=v.device, dtype=v.dtype)

            def _residual_center(tensor: torch.Tensor, gate_tensor: torch.Tensor) -> torch.Tensor:
                idx = torch.nonzero(scope_aligned, as_tuple=False).reshape(-1)
                if idx.numel() == 0:
                    return tensor
                src = tensor.index_select(2, idx)
                center = src.mean(dim=2, keepdim=True)
                g = gate_tensor.index_select(2, idx).to(device=tensor.device, dtype=tensor.dtype)
                updated = center + g * (src - center)
                out = tensor.clone()
                out.index_copy_(2, idx.to(device=tensor.device), updated)
                return out

            if mode in {"v_resid", "kv_resid", "v_resid_centered", "kv_resid_centered"}:
                new_item["v"] = _residual_center(v, gate)
            elif mode in {"v", "value", "kv", "both", "v_centered", "kv_centered"}:
                new_item["v"] = v * gate
            if (
                mode in {"k", "key", "kv", "both", "kv_centered", "kv_resid", "kv_resid_centered"}
                and torch.is_tensor(item.get("k"))
            ):
                k_tensor = new_item["k"] if keep_idx is not None else item["k"]
                if mode in {"kv_resid", "kv_resid_centered"}:
                    new_item["k"] = _residual_center(
                        k_tensor,
                        gate.to(device=k_tensor.device, dtype=k_tensor.dtype),
                    )
                else:
                    new_item["k"] = k_tensor * gate.to(device=k_tensor.device, dtype=k_tensor.dtype)
            if wants_cache_blend:
                target = str(self.swa_write_cache_blend_target or "v").strip().lower()
                if target in {"v", "value", "kv", "both"} and torch.is_tensor(new_item.get("v_post")):
                    post = new_item["v_post"].to(device=new_item["v"].device, dtype=new_item["v"].dtype)
                    new_item["v"] = new_item["v"] * (1.0 - blend) + post * blend
                    cache_blend_applied += 1
                if target in {"k", "key", "kv", "both"} and torch.is_tensor(new_item.get("k_post")) and torch.is_tensor(new_item.get("k")):
                    blend_k = blend.to(device=new_item["k"].device, dtype=new_item["k"].dtype)
                    post = new_item["k_post"].to(device=new_item["k"].device, dtype=new_item["k"].dtype)
                    new_item["k"] = new_item["k"] * (1.0 - blend_k) + post * blend_k
                    cache_blend_applied += 1
                new_item.pop("v_post", None)
                new_item.pop("k_post", None)
            gated_history.append(new_item)
            gate_f = gate.detach().cpu().float().reshape(-1)
            gate_means.append(float(gate_f.mean().item()))
            gate_p10.append(float(torch.quantile(gate_f, 0.10).item()))
            gate_p50.append(float(torch.quantile(gate_f, 0.50).item()))
            gate_p90.append(float(torch.quantile(gate_f, 0.90).item()))
            applied += 1
            tokens_after.append(num_tokens)

        debug.update({
            "swa_write_applied_layers": int(applied),
            "swa_write_cache_blend_applied_layers": int(cache_blend_applied),
            "swa_write_gate_mean": float(torch.tensor(gate_means).mean().item()) if gate_means else 1.0,
            "swa_write_gate_p10": float(torch.tensor(gate_p10).mean().item()) if gate_p10 else 1.0,
            "swa_write_gate_p50": float(torch.tensor(gate_p50).mean().item()) if gate_p50 else 1.0,
            "swa_write_gate_p90": float(torch.tensor(gate_p90).mean().item()) if gate_p90 else 1.0,
            "swa_write_selected_mass": float(selected_mass),
            "swa_write_scope_mass": float(scope_mask.float().mean().item()) if scope_mask.numel() else 1.0,
            "swa_write_scope_tokens": int(scope_mask.sum().item()),
            "swa_write_keep_mass": float(keep_mask.float().mean().item()) if keep_mask.numel() else 1.0,
            "swa_write_keep_tokens": int(keep_mask.sum().item()),
            "swa_write_overlap_frames": int(overlap_frames),
            "swa_write_history_tokens_before": max(tokens_before) if tokens_before else 0,
            "swa_write_history_tokens_after": max(tokens_after) if tokens_after else 0,
            "swa_write_D_patch_mean": float(D_patch.mean().item()),
            "swa_write_D_patch_q90": float(torch.quantile(D_patch, 0.90).item()),
            "swa_write_D_scope_mean": float(D_scope.mean().item()) if D_scope.numel() else 0.0,
            "swa_write_D_scope_q90": float(torch.quantile(D_scope, 0.90).item()) if D_scope.numel() else 0.0,
            "swa_write_D_keep_mean": float(D_patch[keep_mask].mean().item()) if keep_mask.any() else 0.0,
            "swa_write_D_keep_q90": float(torch.quantile(D_patch[keep_mask], 0.90).item()) if keep_mask.any() else 0.0,
            "swa_write_cache_blend_scope_mean": float(base_blend[scope_mask].mean().item()) if scope_mask.any() else 0.0,
            "swa_write_cache_blend_scope_q90": float(torch.quantile(base_blend[scope_mask], 0.90).item()) if scope_mask.any() else 0.0,
        })
        return gated_history, debug

    def _swa_write_layer_enabled(self, layer_idx: int, n_layers: int) -> bool:
        mode = str(self.swa_write_layer_mode or "all")
        if mode == "all":
            return True
        if mode == "first":
            return layer_idx == 0
        if mode == "last":
            return layer_idx == max(0, n_layers - 1)
        if mode == "early":
            return layer_idx < max(1, n_layers // 2)
        if mode == "late":
            return layer_idx >= n_layers // 2
        if mode == "middle":
            lo = n_layers // 3
            hi = max(lo + 1, (2 * n_layers) // 3)
            return lo <= layer_idx < hi
        if mode == "single":
            return layer_idx == int(self.swa_write_single_layer)
        raise ValueError(f"Unsupported swa_write_layer_mode: {mode}")

    def build_control_prior(
        self,
        *,
        probe: ProbeOutput,
        cue: Optional[CueOutput],
        prior_output: Optional[PriorOutput],
        mode: str,
    ) -> HybridMemoryControlPrior:
        saved_read_cue_source = self.read_cue_source
        saved_semantic_role_policy = self.semantic_role_policy
        saved_semantic_memory_paths = self.semantic_memory_paths
        saved_semantic_role_control_mode = self.semantic_role_control_mode
        saved_semantic_anchor_mode = self.semantic_anchor_mode
        saved_enable_semantic_anchor_ttt_floor = self.enable_semantic_anchor_ttt_floor
        gate_mode = "fixed_chunks" if self.semantic_action_active_chunks else "ungated"
        gate_active = self._semantic_action_chunk_active()
        inactive_source = self.semantic_action_inactive_read_cue_source.strip()
        use_inactive_source = bool(
            inactive_source
            and self.semantic_action_active_chunks
            and not gate_active
        )
        inactive_chunk = bool(self.semantic_action_active_chunks and not gate_active)
        read_cue_semantic_prior_requested = _read_cue_requires_semantic_prior(self.read_cue_source)
        keep_inactive_read_cue_prior = bool(
            inactive_chunk
            and not use_inactive_source
            and read_cue_semantic_prior_requested
            and prior_output is not None
        )
        prior_output_for_impl = prior_output if ((not inactive_chunk) or keep_inactive_read_cue_prior) else None
        if use_inactive_source:
            self.read_cue_source = inactive_source
        if inactive_chunk:
            self.semantic_role_policy = "none"
            self.semantic_memory_paths = ""
            self.semantic_role_control_mode = "none"
            self.semantic_anchor_mode = "none"
            self.enable_semantic_anchor_ttt_floor = False
        try:
            out = self._build_control_prior_impl(
                probe=probe,
                cue=cue,
                prior_output=prior_output_for_impl,
                mode=mode,
            )
            out.debug["semantic_action_chunk_gate_mode"] = gate_mode
            out.debug["semantic_action_chunk_gate_active"] = bool(gate_active)
            out.debug["semantic_action_prior_gate_active"] = bool(not inactive_chunk)
            out.debug["semantic_action_prior_consumed"] = bool(prior_output_for_impl is not None)
            out.debug["semantic_action_prior_consumed_for_action"] = bool((not inactive_chunk) and prior_output is not None)
            out.debug["semantic_action_read_cue_requires_semantic_prior"] = bool(read_cue_semantic_prior_requested)
            out.debug["semantic_action_read_cue_prior_consumed"] = bool(keep_inactive_read_cue_prior)
            if inactive_chunk and keep_inactive_read_cue_prior:
                inactive_policy = "disable_semantic_action_keep_read_cue_prior"
            elif inactive_chunk:
                inactive_policy = "disable_semantic_prior_role_control_anchor"
            else:
                inactive_policy = "none"
            out.debug["semantic_action_inactive_policy"] = inactive_policy
            out.debug["semantic_action_read_cue_gate_mode"] = gate_mode
            out.debug["semantic_action_read_cue_gate_active"] = bool(not use_inactive_source)
            out.debug["semantic_action_inactive_read_cue_source"] = inactive_source or None
            out.debug["semantic_action_active_chunks"] = sorted(int(v) for v in self.semantic_action_active_chunks)
            out.debug["semantic_action_chunk_idx"] = int(getattr(self, "current_chunk_idx", -1))
            return out
        finally:
            self.read_cue_source = saved_read_cue_source
            self.semantic_role_policy = saved_semantic_role_policy
            self.semantic_memory_paths = saved_semantic_memory_paths
            self.semantic_role_control_mode = saved_semantic_role_control_mode
            self.semantic_anchor_mode = saved_semantic_anchor_mode
            self.enable_semantic_anchor_ttt_floor = saved_enable_semantic_anchor_ttt_floor

    def _build_control_prior_impl(
        self,
        *,
        probe: ProbeOutput,
        cue: Optional[CueOutput],
        prior_output: Optional[PriorOutput],
        mode: str,
    ) -> HybridMemoryControlPrior:
        geo = probe.geometry
        token_type = geo.token_type.detach().cpu().long()
        L_tok = int(token_type.shape[0])
        patch_mask = token_type == TOKEN_TYPE_PATCH
        num_frames = max(int(geo.num_frames), 1)

        if cue is not None and cue.E_cue_patch is not None:
            patch_cube = cue.E_cue_patch.detach().cpu().float()
            patch = patch_cube.reshape(-1, patch_cube.shape[-1])
            dyn_patch = patch[:, CUE_DYN].clamp(0.0, 1.0)
            occ_patch = patch[:, CUE_OCC].clamp(0.0, 1.0)
            unc_patch = patch[:, CUE_UNC].clamp(0.0, 1.0)
            anchor_patch = patch[:, CUE_ANCHOR].clamp(0.0, 1.0)
            conf_patch = _flatten_or_default(geo.confidence, dyn_patch).clamp(0.0, 1.0)
            fast_write_sources = {
                "stage_d",
                "prior",
                "bl01",
                "dg_inv",
                "dg_locked_inv",
                "read_inv",
                "read_inverse",
                "v5_dg_inv",
                "explicit_dyn_inv",
                "exp_dyn_inv",
                "v5_exp_dyn_inv",
                "dg_inv_sqrt",
                "v5_dg_inv_sqrt",
                "dg_inv_sq",
                "v5_dg_inv_sq",
                "stage_d_x_dg_inv",
                "v5_stage_d_x_dg_inv",
                "stage_d_x_dg_inv_sqrt",
                "v5_stage_d_x_dg_inv_sqrt",
                "stage_d_x_dg_high_inv",
                "v5_stage_d_x_dg_high_inv",
                "stage_d_x_dg_high_inv_sqrt",
                "v5_stage_d_x_dg_high_inv_sqrt",
                "stage_d_x_exp_inv",
                "v5_stage_d_x_exp_inv",
                "stage_d_x_exp_inv_sqrt",
                "v5_stage_d_x_exp_inv_sqrt",
                "stage_d_x_exp_inv_sq",
                "v5_stage_d_x_exp_inv_sq",
                "stage_d_x_exp_focal2",
                "stage_d_x_dg_exp_inv_sqrt",
                "v5_stage_d_x_dg_exp_inv_sqrt",
                "stage_d_x_dg_exp_inter_inv",
                "v5_stage_d_x_dg_exp_inter_inv",
                "stage_d_x_dg_exp_inter_inv_sqrt",
                "v5_stage_d_x_dg_exp_inter_inv_sqrt",
                "stage_d_x_union_dyn_inv",
                "v5_stage_d_x_union_dyn_inv",
                "stage_d_x_dg_inv_sq",
                "v5_stage_d_x_dg_inv_sq",
                "stage_d_x_static_focal2",
                "stage_d_x_dg_inv_pow4",
                "v5_stage_d_x_dg_inv_pow4",
                "stage_d_x_static_focal4",
                "stage_d_x_dg_boundary_focal2",
                "v5_stage_d_x_dg_boundary_focal2",
                "semantic_value",
                "sem_value",
                "sem_x_dg_inv_sqrt",
                "semantic_x_dg_inv_sqrt",
                "stage_d_x_sem",
                "stage_d_x_semantic",
                "stage_d_x_sem_x_dg_inv_sqrt",
                "stage_d_x_semantic_x_dg_inv_sqrt",
                "residual_reliability",
            }
            fast_acl2_only = (
                self.fast_cue_eval
                and self.read_cue_source.startswith("acl2.gg.")
                and self.read_cue_source.endswith(".headmean.robustq")
                and self.read_protection_mode == "none"
                and not self.read_protect_static
                and str(self.hmc_write_score_source) in fast_write_sources
            )
            if fast_acl2_only:
                explicit_patch = _flatten_cue_field_or_default(
                    cue.C_dyn_explicit,
                    dyn_patch,
                    num_frames=num_frames,
                    patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
                ).clamp(0.0, 1.0)
                read_patch = _acl2_read_patch_from_source(
                    self.read_cue_source,
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    dyn_patch,
                    num_frames=num_frames,
                    overlap_frames=self.read_overlap_frames,
                )
                if read_patch is None:
                    read_patch = dyn_patch
                read_patch = read_patch.clamp(0.0, 1.0)
                if self.read_calib_mode == "per_frame_quantile":
                    calibrated = _adaptive_quantile_calibrate(
                        read_patch,
                        num_frames=num_frames,
                        target_mass=self.read_target_mass,
                        tau=self.read_calib_tau,
                    ).to(read_patch.device)
                    blend = min(max(float(self.read_blend_lambda), 0.0), 1.0)
                    read_patch = ((1.0 - blend) * read_patch + blend * calibrated).clamp(0.0, 1.0)
                if self.read_topk_frac > 0.0:
                    read_patch = _topk_per_frame(read_patch, num_frames=num_frames, frac=self.read_topk_frac)

                swa_source = str(self.swa_write_score_source or "read").strip().lower()
                if swa_source in {"", "read", "dg", "dg_locked", "d_g"}:
                    swa_write_patch = read_patch
                elif swa_source in {"explicit", "explicit_dyn", "exp_dyn"}:
                    swa_write_patch = explicit_patch
                elif swa_source in {"old", "old_dyn", "dyn"}:
                    swa_write_patch = dyn_patch
                elif swa_source in {"union", "union_dyn", "dg_exp_union", "max_dg_exp"}:
                    swa_write_patch = torch.maximum(read_patch, explicit_patch).clamp(0.0, 1.0)
                elif swa_source in {"intersection", "inter", "dg_exp_inter", "min_dg_exp"}:
                    swa_write_patch = torch.minimum(read_patch, explicit_patch).clamp(0.0, 1.0)
                else:
                    raise ValueError(f"Unsupported SWA write score source: {self.swa_write_score_source}")

                phase_e_protect_patch = torch.zeros_like(read_patch)
                D_tok = _token_from_patch_values(geo, read_patch, special_value=0.0).clamp(0.0, 1.0)
                D_swa_write_tok = _token_from_patch_values(geo, swa_write_patch, special_value=0.0).clamp(0.0, 1.0)
                R_tok = _token_from_patch_values(
                    geo,
                    ((1.0 - occ_patch) * (1.0 - unc_patch)).clamp(0.0, 1.0),
                    special_value=1.0,
                )
                P_safe = _token_from_patch_values(geo, phase_e_protect_patch, special_value=0.0)

                old_mask = dyn_patch > 0.5
                new_mask = read_patch > 0.5
                inter = (old_mask & new_mask).float().sum()
                union = (old_mask | new_mask).float().sum().clamp_min(1.0)
                new_count = new_mask.float().sum().clamp_min(1.0)
                old_count = old_mask.float().sum().clamp_min(1.0)
                anchor_collision = (
                    float(((read_patch * anchor_patch).sum() / read_patch.sum().clamp_min(1e-8)).item())
                    if read_patch.numel()
                    else 0.0
                )
                fragmentation = _fragmentation(
                    read_patch,
                    num_frames=num_frames,
                    patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
                    thr=0.5,
                )
                dynamic_mass_001 = float((read_patch > 0.01).float().mean().item()) if read_patch.numel() else 0.0
                dynamic_mass_050 = float((read_patch > 0.5).float().mean().item()) if read_patch.numel() else 0.0
                quality_mass_pass = self.read_quality_mass_min <= dynamic_mass_050 <= self.read_quality_mass_max
                quality_anchor_pass = anchor_collision <= self.read_quality_anchor_max
                quality_frag_pass = fragmentation <= self.read_quality_frag_max
                patch_debug = {
                    "fast_cue_eval": True,
                    "mean_D_patch": float(read_patch.mean().item()) if read_patch.numel() else 0.0,
                    "q10_D_patch": float(torch.quantile(read_patch, 0.1).item()) if read_patch.numel() else 0.0,
                    "q50_D_patch": float(torch.quantile(read_patch, 0.5).item()) if read_patch.numel() else 0.0,
                    "q90_D_patch": float(torch.quantile(read_patch, 0.9).item()) if read_patch.numel() else 0.0,
                    "dynamic_mass_D_gt_001": dynamic_mass_001,
                    "dynamic_mass_D_gt_050": dynamic_mass_050,
                    "dynamic_mass_D_gt_075": float((read_patch > 0.75).float().mean().item()) if read_patch.numel() else 0.0,
                    "anchor_collision": anchor_collision,
                    "fragmentation": fragmentation,
                    "old_dyn_iou": float((inter / union).item()),
                    "old_dyn_coverage": float((inter / new_count).item()),
                    "old_dyn_recall": float((inter / old_count).item()),
                    "cue_quality_mass_pass": bool(quality_mass_pass),
                    "cue_quality_anchor_pass": bool(quality_anchor_pass),
                    "cue_quality_frag_pass": bool(quality_frag_pass),
                    "cue_quality_pass": bool(quality_mass_pass and quality_anchor_pass and quality_frag_pass),
                    "cue_gate": 1.0,
                    "fallback_rate": 0.0,
                    "cue_source_effective": self.read_cue_source,
                    "corr_D_unc": _pearson_corr(read_patch, unc_patch),
                    "corr_D_occ": _pearson_corr(read_patch, occ_patch),
                    "corr_D_conf": _pearson_corr(read_patch, conf_patch),
                    "corr_D_inv_conf": _pearson_corr(read_patch, 1.0 - conf_patch),
                    "corr_D_old_dyn": _pearson_corr(read_patch, dyn_patch),
                    "swa_write_score_source": swa_source,
                    "swa_write_score_mean_patch": float(swa_write_patch.mean().item()) if swa_write_patch.numel() else 0.0,
                    "swa_write_score_q90_patch": float(torch.quantile(swa_write_patch, 0.9).item()) if swa_write_patch.numel() else 0.0,
                    "swa_write_corr_score_D": _pearson_corr(swa_write_patch, read_patch),
                    "swa_write_corr_score_exp_dyn": _pearson_corr(swa_write_patch, explicit_patch),
                    "swa_write_corr_score_old_dyn": _pearson_corr(swa_write_patch, dyn_patch),
                }

                U_tok = torch.ones(L_tok, dtype=torch.float32)
                P_ref = (token_type != TOKEN_TYPE_PATCH).float() if self.read_protect_ref else torch.zeros(L_tok, dtype=torch.float32)
                key_stability_patch, key_stability_debug = _select_frame_key_stability_patch(
                    geo,
                    dyn_patch,
                    num_frames=num_frames,
                )
                key_stability_tok = _token_from_patch_values(
                    geo,
                    key_stability_patch,
                    special_value=0.0,
                ).clamp(0.0, 1.0)
                key_stability_debug.update({
                    "key_stability_modulated_by_qk_var": False,
                    "key_stability_tok_gt050_mass": (
                        float((key_stability_tok > 0.50).float().mean().item())
                        if key_stability_tok.numel()
                        else 0.0
                    ),
                })
                patch_debug.update(key_stability_debug)
                raw_frame_bias_energy = _frame_bias_energy(
                    D_tok,
                    P_ref,
                    num_frames=num_frames,
                    mode=self.frame_bias_mode,
                    K_stable_tok=key_stability_tok,
                )
                beta_frame_effective = self.beta_frame
                beta_was_clipped = False
                if self.beta_policy == "bias_energy_norm" and raw_frame_bias_energy > 0.0 and self.beta_energy_target > 0.0:
                    unclipped = self.beta_frame * self.beta_energy_target / max(raw_frame_bias_energy, 1e-8)
                    beta_frame_effective = min(max(unclipped, self.beta_min), self.beta_max)
                    beta_was_clipped = abs(beta_frame_effective - unclipped) > 1e-8

                P_ttt_write = None
                B_chunk_geo = None
                if prior_output is not None:
                    P_ttt_write = prior_output.A_tok.detach().cpu().float()
                    B_chunk_geo = float(prior_output.B_chunk_geo)
                    S_tok = _token_from_patch_values(
                        geo,
                        getattr(prior_output, "V_sem_patch_flat", torch.ones(0)),
                        special_value=1.0,
                    ) if getattr(prior_output, "V_sem_patch_flat", None) is not None else None
                    G_sem_tok = getattr(prior_output, "G_sem_tok", None)
                    if G_sem_tok is not None:
                        G_sem_tok = G_sem_tok.detach().cpu().long()
                    Q_sem_tok = getattr(prior_output, "Q_sem_tok", None)
                    if Q_sem_tok is not None:
                        Q_sem_tok = Q_sem_tok.detach().cpu().float()
                    L_sem_tok = getattr(prior_output, "L_sem_tok", None)
                    if L_sem_tok is not None:
                        L_sem_tok = L_sem_tok.detach().cpu().long()
                    V_sem_tok = getattr(prior_output, "V_sem_tok", None)
                    if V_sem_tok is not None:
                        V_sem_tok = V_sem_tok.detach().cpu().float()
                    R_sem_tok = getattr(prior_output, "R_sem_tok", None)
                    if R_sem_tok is not None:
                        R_sem_tok = R_sem_tok.detach().cpu().long()
                    R_frame_tok = getattr(prior_output, "R_frame_tok", None)
                    if R_frame_tok is not None:
                        R_frame_tok = R_frame_tok.detach().cpu().long()
                    R_global_tok = getattr(prior_output, "R_global_tok", None)
                    if R_global_tok is not None:
                        R_global_tok = R_global_tok.detach().cpu().long()
                    R_swa_tok = getattr(prior_output, "R_swa_tok", None)
                    if R_swa_tok is not None:
                        R_swa_tok = R_swa_tok.detach().cpu().long()
                    R_ttt_tok = getattr(prior_output, "R_ttt_tok", None)
                    if R_ttt_tok is not None:
                        R_ttt_tok = R_ttt_tok.detach().cpu().long()
                    stage_c_seed_global_track_idx_tok = getattr(
                        prior_output,
                        "stage_c_seed_global_track_idx_tok",
                        None,
                    )
                    if stage_c_seed_global_track_idx_tok is not None:
                        stage_c_seed_global_track_idx_tok = (
                            stage_c_seed_global_track_idx_tok.detach().cpu().long()
                        )
                    stage_c_masklet_instance_idx_tok = getattr(
                        prior_output,
                        "stage_c_masklet_instance_idx_tok",
                        None,
                    )
                    if stage_c_masklet_instance_idx_tok is not None:
                        stage_c_masklet_instance_idx_tok = (
                            stage_c_masklet_instance_idx_tok.detach().cpu().long()
                        )
                    override_write, write_debug = self._phase_e_write_prior(
                        probe=probe,
                        token_type=token_type,
                        dyn_patch=dyn_patch,
                        explicit_dyn_patch=explicit_patch,
                        occ_patch=occ_patch,
                        unc_patch=unc_patch,
                        read_patch=read_patch,
                        P_ref=P_ref,
                        base_write_prior=P_ttt_write,
                        semantic_value_patch=getattr(prior_output, "V_sem_patch_flat", None),
                    )
                    if override_write is not None:
                        P_ttt_write = override_write
                        B_chunk_geo = 1.0
                    patch_debug.update(write_debug)
                else:
                    S_tok = None
                    G_sem_tok = None
                    Q_sem_tok = None
                    L_sem_tok = None
                    V_sem_tok = None
                    R_sem_tok = None
                    R_frame_tok = None
                    R_global_tok = None
                    R_swa_tok = None
                    R_ttt_tok = None
                    stage_c_seed_global_track_idx_tok = None
                    stage_c_masklet_instance_idx_tok = None
                    override_write, write_debug = self._phase_e_write_prior(
                        probe=probe,
                        token_type=token_type,
                        dyn_patch=dyn_patch,
                        explicit_dyn_patch=explicit_patch,
                        occ_patch=occ_patch,
                        unc_patch=unc_patch,
                        read_patch=read_patch,
                        P_ref=P_ref,
                        base_write_prior=None,
                        semantic_value_patch=None,
                    )
                    if override_write is not None:
                        P_ttt_write = override_write
                        B_chunk_geo = 1.0
                    patch_debug.update(write_debug)

                if mode in {"unity_replay", "native", "identity_hooks", "read_path_only", "probe_only"}:
                    P_ttt_write = None
                    B_chunk_geo = None
                (
                    R_sem_tok,
                    R_frame_tok,
                    R_global_tok,
                    R_swa_tok,
                    R_ttt_tok,
                    P_ttt_write,
                    D_swa_write_tok,
                    role_debug,
                ) = self._apply_semantic_role_policy(
                    token_type=token_type,
                    D_tok=D_tok,
                    P_ref=P_ref,
                    V_sem_tok=V_sem_tok,
                    G_sem_tok=G_sem_tok,
                    Q_sem_tok=Q_sem_tok,
                    L_sem_tok=L_sem_tok,
                    R_sem_tok=R_sem_tok,
                    R_frame_tok=R_frame_tok,
                    R_global_tok=R_global_tok,
                    R_swa_tok=R_swa_tok,
                    R_ttt_tok=R_ttt_tok,
                    P_ttt_write=P_ttt_write,
                    D_swa_write_tok=D_swa_write_tok,
                    read_patch_tok=_token_from_patch_values(geo, read_patch, special_value=0.0),
                    key_stability_tok=key_stability_tok,
                    num_frames=num_frames,
                    patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
                )
                swa_redirection_source_mask_tok = self._swa_redirection_source_mask_from_roles(
                    R_frame_tok,
                    R_global_tok,
                    role_debug,
                )
                patch_debug.update(role_debug)

                A_anchor_tok, A_anchor_mask_tok, anchor_debug = self._build_semantic_anchor_bank(
                    token_type=token_type,
                    D_tok=D_tok,
                    P_ttt_write=P_ttt_write,
                    G_sem_tok=G_sem_tok,
                    Q_sem_tok=Q_sem_tok,
                    L_sem_tok=L_sem_tok,
                    S_tok=S_tok,
                    V_sem_tok=V_sem_tok,
                    conf_patch=conf_patch,
                    key_avg_patch=key_stability_patch,
                    num_frames=num_frames,
                    patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
                )
                P_ttt_write, anchor_floor_debug = self._apply_semantic_anchor_write_floor(
                    token_type=token_type,
                    P_ttt_write=P_ttt_write,
                    A_anchor_tok=A_anchor_tok,
                )
                patch_debug.update(anchor_debug)
                patch_debug.update(anchor_floor_debug)
                P_ttt_write, v70_ttt_debug = self._apply_v70_radio_ttt_write_prior(
                    token_type=token_type,
                    P_ttt_write=P_ttt_write,
                    num_frames=num_frames,
                    patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
                    chunk_idx=int(getattr(self, "current_chunk_idx", -1)),
                )
                patch_debug.update(v70_ttt_debug)
                P_ttt_write, semantic_ttt_support_debug = self._apply_semantic_ttt_overlap_support_prior(
                    token_type=token_type,
                    P_ttt_write=P_ttt_write,
                    num_frames=num_frames,
                    patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
                    chunk_idx=int(getattr(self, "current_chunk_idx", -1)),
                )
                patch_debug.update(semantic_ttt_support_debug)

                frame_read_effective_enabled = bool(self.enable_frame_read_control)
                if (
                    _is_v95_trackh_gated_l07_source(self.read_cue_source)
                    and patch_debug.get("v95_trackH_gate_active") is False
                ):
                    frame_read_effective_enabled = False
                    patch_debug["v95_trackH_gate_strict_noop"] = True

                return HybridMemoryControlPrior(
                    D_tok=D_tok,
                    R_tok=R_tok,
                    U_tok=U_tok,
                    P_ref=P_ref,
                    S_tok=S_tok,
                    G_sem_tok=G_sem_tok,
                    Q_sem_tok=Q_sem_tok,
                    L_sem_tok=L_sem_tok,
                    V_sem_tok=V_sem_tok,
                    R_sem_tok=R_sem_tok,
                    R_frame_tok=R_frame_tok,
                    R_global_tok=R_global_tok,
                    R_swa_tok=R_swa_tok,
                    R_ttt_tok=R_ttt_tok,
                    stage_c_seed_global_track_idx_tok=stage_c_seed_global_track_idx_tok,
                    stage_c_masklet_instance_idx_tok=stage_c_masklet_instance_idx_tok,
                    K_stable_tok=key_stability_tok,
                    swa_redirection_source_mask_tok=swa_redirection_source_mask_tok,
                    A_anchor_tok=A_anchor_tok,
                    A_anchor_mask_tok=A_anchor_mask_tok,
                    C_ttt_conflict_tok=(
                        _optional_1d_float_tensor(getattr(self, "semantic_condition_conflict_tok", None))
                        if bool(role_debug.get("condition_signal_conflict_token_exact", False))
                        else None
                    ),
                    S_scale_risk_tok=(
                        _optional_1d_float_tensor(getattr(self, "semantic_condition_scale_tok", None))
                        if bool(role_debug.get("condition_signal_scale_risk_token_exact", False))
                        else None
                    ),
                    P_ttt_write=P_ttt_write,
                    P_ttt_read=None,
                    P_swa_read_prev=None,
                    D_swa_write_tok=D_swa_write_tok,
                    frame_bias_spec={
                        "enabled": frame_read_effective_enabled,
                        "beta": self.beta_frame,
                        "beta_effective": beta_frame_effective,
                        "mode": self.frame_bias_mode,
                        "query_region": self.frame_bias_query_region,
                        "head_indices": self.frame_bias_head_indices,
                        "cue_source": self.read_cue_source,
                        "topk_frac": self.read_topk_frac,
                        "calib_mode": self.read_calib_mode,
                        "target_mass": self.read_target_mass,
                        "calib_tau": self.read_calib_tau,
                        "blend_lambda": self.read_blend_lambda,
                        "protection_mode": self.read_protection_mode,
                        "ref_strength": self.read_ref_strength,
                    },
                    chunk_bias_spec={
                        "enabled": self.enable_chunk_read_control,
                        "beta": self.beta_chunk,
                        "mode": self.chunk_bias_mode,
                        "cue_source": self.read_cue_source,
                    },
                    B_chunk_geo=B_chunk_geo,
                    debug={
                        "mode": mode,
                        "mean_D_tok": float(D_tok.mean().item()) if D_tok.numel() else 0.0,
                        "max_D_tok": float(D_tok.max().item()) if D_tok.numel() else 0.0,
                        "q90_D_tok": float(torch.quantile(D_tok, 0.9).item()) if D_tok.numel() else 0.0,
                        "mean_R_tok": float(R_tok.mean().item()) if R_tok.numel() else 1.0,
                        "protected_token_count": int((P_ref > 0.5).sum().item()),
                        "safe_patch_token_count": int((P_safe > 0.5).sum().item()),
                        "patch_token_count": int(patch_mask.sum().item()),
                        "read_cue_source": self.read_cue_source,
                        "read_topk_frac": self.read_topk_frac,
                        "read_calib_mode": self.read_calib_mode,
                        "read_target_mass": self.read_target_mass,
                        "read_calib_tau": self.read_calib_tau,
                        "read_blend_lambda": self.read_blend_lambda,
                        "read_quality_mass_min": self.read_quality_mass_min,
                        "read_quality_mass_max": self.read_quality_mass_max,
                        "read_quality_anchor_max": self.read_quality_anchor_max,
                        "read_quality_frag_max": self.read_quality_frag_max,
                        "beta_policy": self.beta_policy,
                        "beta_frame_effective": beta_frame_effective,
                        "beta_energy_target": self.beta_energy_target,
                        "beta_raw_frame_bias_energy": raw_frame_bias_energy,
                        "beta_was_clipped": bool(beta_was_clipped),
                        "frame_bias_mode": self.frame_bias_mode,
                        "frame_bias_query_region": self.frame_bias_query_region,
                        "frame_bias_head_indices": self.frame_bias_head_indices,
                        "chunk_bias_mode": self.chunk_bias_mode,
                        "swa_bias_mode": self.swa_bias_mode,
                        "read_protect_ref": self.read_protect_ref,
                        "read_protect_static": self.read_protect_static,
                        "read_protection_mode": self.read_protection_mode,
                        "read_ref_strength": self.read_ref_strength,
                        "read_overlap_frames": self.read_overlap_frames,
                        "read_reset_frames": self.read_reset_frames,
                        "read_attention_q": self.read_attention_q,
                        "hmc_write_score_source": self.hmc_write_score_source,
                        "hmc_write_sparse_ratio": self.hmc_write_sparse_ratio,
                        "hmc_write_sparse_mode": self.hmc_write_sparse_mode,
                        "ttt_write_prior_present": P_ttt_write is not None,
                        "ttt_write_prior_mean": float(P_ttt_write.mean().item()) if P_ttt_write is not None else 1.0,
                        "read_path_control": "real_hook_identity_or_requested",
                        **patch_debug,
                    },
                )
            kk_patch = _normalize01(_flatten_or_default(geo.dyn4d_kk_mean_patch, dyn_patch))
            qq_patch = _normalize01(_flatten_or_default(geo.dyn4d_qq_mean_patch, dyn_patch))
            qk_var_patch = _normalize01(_flatten_or_default(geo.dyn4d_qk_var_patch, dyn_patch))
            dyn4d_patch = _normalize01(_flatten_or_default(geo.dyn4d_patch, dyn_patch))
            key_avg_patch = _normalize01(_flatten_or_default(geo.frame_attn_key_cosine_avg, dyn_patch))
            query_avg_patch = _normalize01(_flatten_or_default(geo.frame_attn_cosine_avg, dyn_patch))
            query_shallow_patch = _normalize01(_flatten_or_default(geo.frame_attn_cosine_shallow, dyn_patch))
            query_deep_patch = _normalize01(_flatten_or_default(geo.frame_attn_cosine_deep, dyn_patch))
            attn_patch = _normalize01(_flatten_or_default(geo.attn_dynamic_patch, dyn_patch))
            frame_layer_ids = geo.frame_attn_cosine_layer_ids
            global_layer_ids = geo.dyn4d_global_layer_ids
            fa_key_l0_raw = _flatten_or_default(geo.frame_attn_key_cosine_l0, dyn_patch)
            fa_key_l4_raw = _flatten_or_default(geo.frame_attn_key_cosine_l4, dyn_patch)
            if geo.frame_attn_cosine_key_layers is None:
                # Current LoGeR trace exports explicit l0/l4 maps but not the full
                # frame-attention layer stack; keep the plan's group names runnable
                # by mapping them onto the available shallow/deeper endpoints.
                fa_key_shallow_raw = fa_key_l0_raw
                fa_key_middle_raw = 0.5 * (fa_key_l0_raw + fa_key_l4_raw)
                fa_key_deep_raw = fa_key_l4_raw
                fa_key_all_raw = fa_key_middle_raw
                fa_key_layer_var_raw = torch.stack([fa_key_l0_raw, fa_key_l4_raw], dim=0).var(dim=0, unbiased=False)
            else:
                fa_key_shallow_raw = _flatten_layer_group_map(
                    geo.frame_attn_cosine_key_layers,
                    frame_layer_ids,
                    dyn_patch,
                    group="shallow",
                )
                fa_key_middle_raw = _flatten_layer_group_map(
                    geo.frame_attn_cosine_key_layers,
                    frame_layer_ids,
                    dyn_patch,
                    group="middle",
                )
                fa_key_deep_raw = _flatten_layer_group_map(
                    geo.frame_attn_cosine_key_layers,
                    frame_layer_ids,
                    dyn_patch,
                    group="deep",
                )
                fa_key_all_raw = _flatten_layer_group_map(
                    geo.frame_attn_cosine_key_layers,
                    frame_layer_ids,
                    dyn_patch,
                    group="all",
                )
                fa_key_layer_var_raw = _flatten_layer_group_map(
                    geo.frame_attn_cosine_key_layers,
                    frame_layer_ids,
                    dyn_patch,
                    group="all",
                    stat="var",
                )
            fa_query_shallow_raw = _flatten_layer_group_map(
                geo.frame_attn_cosine_query_layers,
                frame_layer_ids,
                dyn_patch,
                group="shallow",
            )
            fa_key_l0_patch = _robust_quantile01(fa_key_l0_raw, num_frames=num_frames)
            fa_key_l4_patch = _robust_quantile01(fa_key_l4_raw, num_frames=num_frames)
            fa_key_shallow_patch = _robust_quantile01(fa_key_shallow_raw, num_frames=num_frames)
            fa_key_middle_patch = _robust_quantile01(fa_key_middle_raw, num_frames=num_frames)
            fa_key_deep_patch = _robust_quantile01(fa_key_deep_raw, num_frames=num_frames)
            fa_key_all_patch = _robust_quantile01(fa_key_all_raw, num_frames=num_frames)
            fa_key_layer_var_patch = _robust_quantile01(
                fa_key_layer_var_raw,
                num_frames=num_frames,
            )
            fa_key_decay_patch = _robust_quantile01(
                fa_key_shallow_raw * (fa_key_shallow_raw - fa_key_deep_raw).clamp_min(0.0),
                num_frames=num_frames,
            )
            fa_key_l0_deep_decay_patch = _robust_quantile01(
                fa_key_l0_raw * (fa_key_l0_raw - fa_key_deep_raw).clamp_min(0.0),
                num_frames=num_frames,
            )
            fa_key_l4_deep_decay_patch = _robust_quantile01(
                fa_key_l4_raw * (fa_key_l4_raw - fa_key_deep_raw).clamp_min(0.0),
                num_frames=num_frames,
            )
            fa_key_deep_low_patch = _robust_quantile01(1.0 - fa_key_deep_raw, num_frames=num_frames)
            fa_query_shallow_patch = _robust_quantile01(fa_query_shallow_raw, num_frames=num_frames)
            key_stability_base_patch, key_stability_debug = _select_frame_key_stability_patch(
                geo,
                dyn_patch,
                num_frames=num_frames,
            )
            key_stability_patch = (
                key_stability_base_patch * (1.0 - qk_var_patch).clamp(0.0, 1.0)
            ).clamp(0.0, 1.0)
            key_stability_debug.update({
                "key_stability_modulated_by_qk_var": True,
                **_patch_signal_debug("key_stability_patch", key_stability_patch),
            })
            gg_qk_middle_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="middle",
                    kind="qk_var",
                ),
                num_frames=num_frames,
            )
            gg_qk_shallow_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="shallow",
                    kind="qk_var",
                ),
                num_frames=num_frames,
            )
            gg_qk_deep_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="deep",
                    kind="qk_var",
                ),
                num_frames=num_frames,
            )
            gg_qq_middle_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="middle",
                    kind="qq_low",
                ),
                num_frames=num_frames,
            )
            gg_kk_middle_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="middle",
                    kind="kk_low",
                ),
                num_frames=num_frames,
            )
            gg_qq_shallow_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="shallow",
                    kind="qq_low",
                ),
                num_frames=num_frames,
            )
            gg_deep_static_patch = _robust_quantile01(
                _global_centroid_metric(
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    global_layer_ids,
                    dyn_patch,
                    group="deep",
                    kind="deep_static",
                ),
                num_frames=num_frames,
            )
            gg_smd_a1b1g1_patch = _robust_quantile01(
                gg_qq_shallow_patch * gg_qk_middle_patch * gg_deep_static_patch,
                num_frames=num_frames,
            )
            gg_smd_a0b1g1_patch = _robust_quantile01(
                gg_qk_middle_patch * gg_deep_static_patch,
                num_frames=num_frames,
            )
            explicit_patch = _flatten_cue_field_or_default(
                cue.C_dyn_explicit,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            implicit_patch = _flatten_cue_field_or_default(
                cue.C_dyn_implicit,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            fusion_max_patch = _flatten_cue_field_or_default(
                cue.C_dyn_fusion_max,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            fusion_soft_or_patch = _flatten_cue_field_or_default(
                cue.C_dyn_fusion_soft_or,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            fusion_avg_patch = _flatten_cue_field_or_default(
                cue.C_dyn_fusion_avg,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            fusion_addclip_patch = _flatten_cue_field_or_default(
                cue.C_dyn_fusion_addclip,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            stage_geo_patch = _flatten_cue_field_or_default(
                cue.G_write_geo_patch,
                dyn_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            ).clamp(0.0, 1.0)
            flow_proxy_raw = (dyn_patch * (1.0 - occ_patch).clamp(0.0, 1.0) * conf_patch).clamp(0.0, 1.0)
            structure_like = (anchor_patch >= self.read_static_anchor_thr) & (dyn_patch <= self.read_static_dyn_thr)
            flow_sem_veto_raw = torch.where(structure_like, torch.zeros_like(flow_proxy_raw), flow_proxy_raw)
            flow_proxy_calib = _adaptive_quantile_calibrate(
                flow_proxy_raw,
                num_frames=num_frames,
                target_mass=self.read_target_mass,
                tau=self.read_calib_tau,
            )
            flow_sem_veto_calib = _adaptive_quantile_calibrate(
                flow_sem_veto_raw,
                num_frames=num_frames,
                target_mass=self.read_target_mass,
                tau=self.read_calib_tau,
            )
            cue_gate = 1.0
            fallback_rate = 0.0
            cue_source_effective = self.read_cue_source
            cue_extra_debug: Dict[str, Any] = {}
            read_skip_calibration = False
            acl2_patch = _acl2_read_patch_from_source(
                self.read_cue_source,
                geo.global_q_raw_patchvec_layers,
                geo.global_k_raw_patchvec_layers,
                dyn_patch,
                num_frames=num_frames,
                overlap_frames=self.read_overlap_frames,
            )
            v68_cfg = _parse_v68_read_cue(self.read_cue_source)
            v68_patch = None
            if v68_cfg is not None:
                current_chunk_idx = int(getattr(self, "current_chunk_idx", -1))
                v68_patch, v68_debug = _v68_read_cue_patch(
                    token_type=token_type,
                    prior_output=prior_output,
                    cfg=v68_cfg,
                    q_layers=geo.global_q_raw_patchvec_layers,
                    k_layers=geo.global_k_raw_patchvec_layers,
                    v_layers=getattr(geo, "global_v_raw_patchvec_layers", None),
                    default=dyn_patch,
                    num_frames=num_frames,
                    chunk_idx=current_chunk_idx,
                )
                cue_extra_debug.update(v68_debug)
            v78_l07_l13_cfg = _parse_v78_l07_l13_read_cue(self.read_cue_source)
            v78_l07_l13_patch = None
            if v78_l07_l13_cfg is not None:
                current_chunk_idx = int(getattr(self, "current_chunk_idx", -1))
                v78_l07_l13_patch, v78_debug = _v78_l07_l13_read_cue_patch(
                    token_type=token_type,
                    prior_output=prior_output,
                    cfg=v78_l07_l13_cfg,
                    q_layers=geo.global_q_raw_patchvec_layers,
                    k_layers=geo.global_k_raw_patchvec_layers,
                    v_layers=getattr(geo, "global_v_raw_patchvec_layers", None),
                    default=dyn_patch,
                    num_frames=num_frames,
                    chunk_idx=current_chunk_idx,
                )
                cue_extra_debug.update(v78_debug)
            v70_radio_cfg = _parse_v70_radio_read_cue(self.read_cue_source)
            v70_radio_patch = None
            if v70_radio_cfg is not None:
                current_chunk_idx = int(getattr(self, "current_chunk_idx", -1))
                v70_radio_patch, v70_radio_debug = _v70_radio_read_cue_patch(
                    sidecar_dir=self.v70_radio_sidecar_dir,
                    cfg=v70_radio_cfg,
                    default=dyn_patch,
                    num_frames=num_frames,
                    patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
                    chunk_idx=current_chunk_idx,
                )
                cue_extra_debug.update(v70_radio_debug)

            if v78_l07_l13_patch is not None:
                read_patch = v78_l07_l13_patch
                cue_source_effective = self.read_cue_source
            elif v70_radio_patch is not None:
                read_patch = v70_radio_patch
                cue_source_effective = self.read_cue_source
            elif v68_patch is not None:
                read_patch = v68_patch
                cue_source_effective = self.read_cue_source
            elif acl2_patch is not None:
                read_patch = acl2_patch
            elif self.read_cue_source in {"dyn", "old_dyn", "old_dyn_calibrated_soft_or"}:
                read_patch = dyn_patch
            elif self.read_cue_source == "explicit_dyn_only":
                read_patch = explicit_patch
            elif self.read_cue_source in {"implicit_dyn_only", "manual_implicit_dyn"}:
                read_patch = implicit_patch
            elif self.read_cue_source == "old_dyn_max":
                read_patch = fusion_max_patch
            elif self.read_cue_source == "old_dyn_soft_or":
                read_patch = fusion_soft_or_patch
            elif self.read_cue_source == "old_dyn_avg":
                read_patch = fusion_avg_patch
            elif self.read_cue_source == "old_dyn_addclip":
                read_patch = fusion_addclip_patch
            elif self.read_cue_source == "inverted_dyn":
                read_patch = (1.0 - dyn_patch).clamp(0.0, 1.0)
            elif self.read_cue_source == "random":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(1729 + int(dyn_patch.numel()))
                read_patch = torch.rand(dyn_patch.shape, generator=generator, dtype=torch.float32)
            elif self.read_cue_source == "dyn_reliable":
                read_patch = (dyn_patch * (1.0 - unc_patch) * (1.0 - occ_patch)).clamp(0.0, 1.0)
            elif self.read_cue_source == "gram_lite":
                raw = (1.0 - kk_patch).clamp(0.0, 1.0) * (1.0 - qq_patch).clamp(0.0, 1.0) * (0.5 + dyn4d_patch)
                read_patch = _normalize01(raw)
            elif self.read_cue_source == "gram4d":
                raw = (1.0 - kk_patch).clamp(0.0, 1.0) * qk_var_patch * (0.25 + qq_patch) * (0.5 + dyn4d_patch)
                read_patch = _normalize01(raw)
            elif self.read_cue_source == "dyn4d_patch":
                read_patch = dyn4d_patch
            elif self.read_cue_source == "qk_var":
                read_patch = qk_var_patch
            elif self.read_cue_source == "qqkk_disagree":
                raw = (1.0 - qq_patch).clamp(0.0, 1.0) * (1.0 - kk_patch).clamp(0.0, 1.0)
                read_patch = _normalize01(raw)
            elif self.read_cue_source == "entropy":
                # Dense attention rows are not exported; this is a conservative
                # dispersion proxy from available attention/dyn4d statistics.
                raw = 0.45 * qk_var_patch + 0.35 * attn_patch + 0.20 * (1.0 - key_avg_patch).clamp(0.0, 1.0)
                read_patch = _normalize01(raw)
            elif self.read_cue_source in {"flow", "flow_proxy"}:
                read_patch = _normalize01(flow_proxy_raw)
            elif self.read_cue_source in {"flow_sem_veto", "flow_sem_veto_proxy"}:
                # v3 patch-match flow proxy: use reprojection/dynamic residual if
                # Stage-B was configured for it, then suppress occluded/uncertain
                # low-confidence patches. RAFT/GMFlow integration is left explicit
                # via flow_model for later runs.
                read_patch = _normalize01(flow_sem_veto_raw)
            elif self.read_cue_source == "flow_proxy_calib":
                read_patch = flow_proxy_calib
            elif self.read_cue_source == "flow_sem_veto_calib":
                read_patch = flow_sem_veto_calib
            elif self.read_cue_source == "old_dyn_plus_flow_proxy":
                lam = min(max(self.read_blend_lambda, 0.0), 1.0)
                read_patch = _normalize01((1.0 - lam) * dyn_patch + lam * flow_proxy_calib)
            elif self.read_cue_source == "old_dyn_plus_flow_sem_veto":
                lam = min(max(self.read_blend_lambda, 0.0), 1.0)
                read_patch = _normalize01((1.0 - lam) * dyn_patch + lam * flow_sem_veto_calib)
            elif self.read_cue_source == "old_dyn_gram_lite_agree":
                lam = min(max(self.read_blend_lambda, 0.0), 1.0)
                gram = _normalize01((1.0 - kk_patch).clamp(0.0, 1.0) * (1.0 - qq_patch).clamp(0.0, 1.0) * (0.5 + dyn4d_patch))
                agree = (1.0 + lam * (gram - 0.5)).clamp(0.25, 1.75)
                read_patch = (dyn_patch * agree).clamp(0.0, 1.0)
            elif self.read_cue_source == "old_dyn_gram4d_agree":
                lam = min(max(self.read_blend_lambda, 0.0), 1.0)
                gram = _normalize01((1.0 - kk_patch).clamp(0.0, 1.0) * qk_var_patch * (0.25 + qq_patch) * (0.5 + dyn4d_patch))
                agree = (1.0 + lam * (gram - 0.5)).clamp(0.25, 1.75)
                read_patch = (dyn_patch * agree).clamp(0.0, 1.0)
            elif self.read_cue_source == "old_dyn_key_static_rescue":
                lam = min(max(self.read_blend_lambda, 0.0), 1.0)
                rescue = (key_avg_patch * anchor_patch * (1.0 - unc_patch).clamp(0.0, 1.0)).clamp(0.0, 1.0)
                read_patch = (dyn_patch * (1.0 - lam * rescue)).clamp(0.0, 1.0)
            elif self.read_cue_source in {
                "mix.c23_g4_soft_l010",
                "mix.c23_g4_soft_l025",
                "mix.c23_g4_soft_l050",
                "mix.c23_g4_soft_l075",
            }:
                D23 = _acl2_read_patch_from_source(
                    "acl2.gg.qq.low.g2_3.full.headmean.robustq",
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    dyn_patch,
                    num_frames=num_frames,
                    overlap_frames=self.read_overlap_frames,
                )
                D4 = _acl2_read_patch_from_source(
                    "acl2.gg.qq.low.g4.full.headmean.robustq",
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    dyn_patch,
                    num_frames=num_frames,
                    overlap_frames=self.read_overlap_frames,
                )
                if D23 is None:
                    D23 = dyn_patch
                if D4 is None:
                    D4 = dyn_patch
                lam_g4 = 0.10
                if self.read_cue_source.endswith("_l025"):
                    lam_g4 = 0.25
                elif self.read_cue_source.endswith("_l050"):
                    lam_g4 = 0.50
                elif self.read_cue_source.endswith("_l075"):
                    lam_g4 = 0.75
                read_patch = ((1.0 - lam_g4) * D23 + lam_g4 * D4).clamp(0.0, 1.0)
                cue_source_effective = self.read_cue_source
            elif self.read_cue_source in {
                "mix.c24past_old_route_lg100_lo025",
                "mix.c24past_old_route_lg100_lo050",
                "mix.c24past_exp_route_lg100_le050",
                "mix.c23past_old_route_lg100_lo025",
                "mix.c24past_static_rescue_a025",
                "mix.c23past_static_rescue_a025",
            }:
                if "c23past" in self.read_cue_source:
                    dg_source = "acl2.gg.qq.low.g2_3.past_only.headmean.robustq"
                else:
                    dg_source = "acl2.gg.qq.low.g2_4.past_only.headmean.robustq"
                D_g = _acl2_read_patch_from_source(
                    dg_source,
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    dyn_patch,
                    num_frames=num_frames,
                    overlap_frames=self.read_overlap_frames,
                )
                if D_g is None:
                    D_g = dyn_patch
                D_g = D_g.clamp(0.0, 1.0)
                if "static_rescue" in self.read_cue_source:
                    alpha = 0.25
                    read_patch = (D_g * (1.0 - alpha * gg_deep_static_patch)).clamp(0.0, 1.0)
                else:
                    aux = explicit_patch if "_exp_" in self.read_cue_source else fusion_addclip_patch
                    lam_g = 1.0
                    lam_aux = 0.50 if ("lo050" in self.read_cue_source or "le050" in self.read_cue_source) else 0.25
                    g_on = D_g > 0.5
                    aux_on = aux > 0.5
                    read_patch = torch.zeros_like(D_g)
                    read_patch = torch.where(g_on & aux_on, torch.maximum(D_g, aux), read_patch)
                    read_patch = torch.where(g_on & (~aux_on), (lam_g * D_g).clamp(0.0, 1.0), read_patch)
                    read_patch = torch.where((~g_on) & aux_on, (lam_aux * aux).clamp(0.0, 1.0), read_patch)
                cue_source_effective = self.read_cue_source
            elif self.read_cue_source in {
                "semantic_anchor_rescue.c23past_l025",
                "semantic_anchor_rescue.c23past_l050",
            }:
                D_g = _acl2_read_patch_from_source(
                    "acl2.gg.qq.low.g2_3.past_only.headmean.robustq",
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    dyn_patch,
                    num_frames=num_frames,
                    overlap_frames=self.read_overlap_frames,
                )
                if D_g is None:
                    D_g = dyn_patch
                alpha = 0.25 if self.read_cue_source.endswith("_l025") else 0.50
                read_patch, rescue_debug = _semantic_anchor_rescue_patch(
                    D_g.clamp(0.0, 1.0),
                    token_type=token_type,
                    prior_output=prior_output,
                    gg_deep_static_patch=gg_deep_static_patch,
                    conf_patch=conf_patch,
                    key_avg_patch=key_avg_patch,
                    alpha=alpha,
                )
                cue_extra_debug.update(rescue_debug)
                cue_source_effective = self.read_cue_source
            elif _parse_v67_semgeo_read_cue(self.read_cue_source) is not None:
                semgeo_cfg = _parse_v67_semgeo_read_cue(self.read_cue_source) or {}
                current_chunk_idx = int(getattr(self, "current_chunk_idx", -1))
                read_patch, semgeo_debug = _v67_semgeo_cue_patch(
                    token_type=token_type,
                    prior_output=prior_output,
                    cfg=semgeo_cfg,
                    dyn_patch=dyn_patch,
                    occ_patch=occ_patch,
                    unc_patch=unc_patch,
                    conf_patch=conf_patch,
                    anchor_patch=anchor_patch,
                    stage_geo_patch=stage_geo_patch,
                    qkstable_patch=(1.0 - gg_qk_middle_patch).clamp(0.0, 1.0),
                    deepstatic_patch=gg_deep_static_patch,
                    smd_patch=gg_smd_a1b1g1_patch,
                    fa_decay_patch=fa_key_decay_patch,
                    key_avg_patch=key_avg_patch,
                    num_frames=num_frames,
                    chunk_idx=current_chunk_idx,
                )
                semgeo_debug["v67_semgeo_read_cue_source"] = self.read_cue_source
                cue_extra_debug.update(semgeo_debug)
                cue_source_effective = self.read_cue_source
            elif _parse_v67_semantic_cue(self.read_cue_source) is not None:
                semcue_cfg = _parse_v67_semantic_cue(self.read_cue_source) or {}
                current_chunk_idx = int(getattr(self, "current_chunk_idx", -1))
                read_patch, semcue_debug = _v67_semantic_anchor_cue_patch(
                    token_type=token_type,
                    prior_output=prior_output,
                    kind=str(semcue_cfg.get("kind", "anchor_risk")),
                    random_same_distribution=bool(semcue_cfg.get("random_same_distribution", False)),
                    chunk_idx=current_chunk_idx,
                )
                semcue_debug["v67_semcue_read_cue_source"] = self.read_cue_source
                cue_extra_debug.update(semcue_debug)
                cue_source_effective = self.read_cue_source
            elif _parse_v67_semz_read_cue(self.read_cue_source) is not None:
                v67_cfg = _parse_v67_semz_read_cue(self.read_cue_source) or {}
                D_g = _acl2_read_patch_from_source(
                    "acl2.gg.qq.low.g2_3.past_only.headmean.robustq",
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    dyn_patch,
                    num_frames=num_frames,
                    overlap_frames=self.read_overlap_frames,
                )
                if D_g is None:
                    D_g = dyn_patch
                D_g = D_g.clamp(0.0, 1.0)
                label_mode = str(v67_cfg.get("label_mode", "coarse"))
                label_field = "G_sem_tok" if label_mode == "coarse" else "L_sem_tok"
                label_tok = getattr(prior_output, label_field, None) if prior_output is not None else None
                patch_labels = _patch_labels_from_token_labels(label_tok, token_type)
                current_chunk_idx = int(getattr(self, "current_chunk_idx", -1))
                read_patch, v67_debug = _v67_semantic_mad_recondition_patch(
                    D_g,
                    patch_labels,
                    beta=float(v67_cfg.get("beta", 0.525)),
                    min_count=256,
                    control_mode=str(v67_cfg.get("control_mode", "semantic")),
                    chunk_idx=current_chunk_idx,
                )
                v67_debug["v67_semz_label_field"] = label_field
                v67_debug["v67_semz_read_cue_source"] = self.read_cue_source
                cue_extra_debug.update(v67_debug)
                cue_source_effective = self.read_cue_source
            elif self.read_cue_source in {
                "v31.sem_z_fine.c23past",
                "v31.sem_z_coarse.c23past",
            } or (
                self.read_cue_source.startswith(("v31.sem_resid_fine_l", "v31.sem_resid_coarse_l"))
                and self.read_cue_source.endswith(".c23past")
            ):
                D_g = _acl2_read_patch_from_source(
                    "acl2.gg.qq.low.g2_3.past_only.headmean.robustq",
                    geo.global_q_raw_patchvec_layers,
                    geo.global_k_raw_patchvec_layers,
                    dyn_patch,
                    num_frames=num_frames,
                    overlap_frames=self.read_overlap_frames,
                )
                if D_g is None:
                    D_g = dyn_patch
                D_g = D_g.clamp(0.0, 1.0)
                label_field = "G_sem_tok" if "_coarse" in self.read_cue_source else "L_sem_tok"
                label_tok = getattr(prior_output, label_field, None) if prior_output is not None else None
                patch_labels = _patch_labels_from_token_labels(label_tok, token_type)
                sem_z_patch, sem_z_debug = _semantic_z_recondition_patch(
                    D_g,
                    patch_labels,
                    min_count=32,
                )
                sem_z_high_mass = float(
                    (sem_z_patch >= self.v32_semantic_cue_trigger_high_threshold).float().mean().item()
                ) if sem_z_patch.numel() else 0.0
                sem_d_mean = float(sem_z_patch.mean().item()) if sem_z_patch.numel() else 0.0
                sem_d_q90 = float(torch.quantile(sem_z_patch.float().flatten(), 0.9).item()) if sem_z_patch.numel() else 0.0
                current_chunk_idx = int(getattr(self, "current_chunk_idx", -1))
                v32_gate_mode = "ungated"
                v32_gate_reason = "ungated_all_chunks"
                v32_semantic_cue_active = True
                trigger_threshold = None
                trigger_ema_before = self._v32_semantic_cue_ema
                trigger_mad_before = self._v32_semantic_cue_mad
                if self.v32_semantic_cue_active_chunks:
                    v32_gate_mode = "fixed_chunks"
                    v32_semantic_cue_active = current_chunk_idx in self.v32_semantic_cue_active_chunks
                    v32_gate_reason = "fixed_chunk_active" if v32_semantic_cue_active else "fixed_chunk_inactive"
                elif self.v32_semantic_cue_trigger_mode in {
                    "semantic_z_ema_mad",
                    "semantic_d_mean_ema_mad",
                    "semantic_d_q90_ema_mad",
                }:
                    v32_gate_mode = self.v32_semantic_cue_trigger_mode
                    if self.v32_semantic_cue_trigger_mode == "semantic_d_mean_ema_mad":
                        trigger_metric = sem_d_mean
                    elif self.v32_semantic_cue_trigger_mode == "semantic_d_q90_ema_mad":
                        trigger_metric = sem_d_q90
                    else:
                        trigger_metric = sem_z_high_mass
                    if self._v32_semantic_cue_ema is None:
                        trigger_threshold = self.v32_semantic_cue_trigger_min_mass
                    else:
                        trigger_threshold = max(
                            self.v32_semantic_cue_trigger_min_mass,
                            self._v32_semantic_cue_ema
                            + self.v32_semantic_cue_trigger_k * float(self._v32_semantic_cue_mad or 0.0),
                        )
                    warmup_done = self._v32_semantic_cue_count >= self.v32_semantic_cue_trigger_warmup
                    v32_semantic_cue_active = bool(warmup_done and trigger_metric > float(trigger_threshold))
                    v32_gate_reason = "runtime_trigger_active" if v32_semantic_cue_active else "runtime_trigger_inactive"
                    alpha = min(max(self.v32_semantic_cue_trigger_ema_alpha, 0.0), 1.0)
                    if self._v32_semantic_cue_ema is None:
                        self._v32_semantic_cue_ema = trigger_metric
                        self._v32_semantic_cue_mad = 0.0
                    else:
                        prev_ema = self._v32_semantic_cue_ema
                        self._v32_semantic_cue_ema = (1.0 - alpha) * self._v32_semantic_cue_ema + alpha * trigger_metric
                        abs_dev = abs(trigger_metric - prev_ema)
                        self._v32_semantic_cue_mad = (1.0 - alpha) * float(self._v32_semantic_cue_mad or 0.0) + alpha * abs_dev
                    self._v32_semantic_cue_count += 1
                if not v32_semantic_cue_active:
                    read_patch = D_g
                    sem_z_debug["v31_semantic_recondition_applied"] = False
                    sem_z_debug["v31_semantic_recondition_reason"] = v32_gate_reason
                elif ".sem_resid_" in self.read_cue_source:
                    lam_sem = 0.25
                    try:
                        lam_token = self.read_cue_source.split("_l", 1)[1].split(".", 1)[0]
                        if lam_token.isdigit():
                            lam_sem = min(max(float(int(lam_token)) / 100.0, 0.0), 1.0)
                    except IndexError:
                        lam_sem = 0.25
                    read_patch = ((1.0 - lam_sem) * D_g + lam_sem * sem_z_patch).clamp(0.0, 1.0)
                    sem_z_debug["v31_semantic_recondition_mode"] = f"semantic_residual_l{int(round(lam_sem * 100)):03d}"
                    sem_z_debug["v31_semantic_residual_lambda"] = float(lam_sem)
                else:
                    read_patch = sem_z_patch.clamp(0.0, 1.0)
                sem_z_debug["v31_semantic_label_field"] = label_field
                sem_z_debug["v32_semantic_cue_gate_mode"] = v32_gate_mode
                sem_z_debug["v32_semantic_cue_active"] = bool(v32_semantic_cue_active)
                sem_z_debug["v32_semantic_cue_gate_reason"] = v32_gate_reason
                sem_z_debug["v32_semantic_cue_chunk_idx"] = int(current_chunk_idx)
                sem_z_debug["v32_semantic_z_high_mass"] = float(sem_z_high_mass)
                sem_z_debug["v32_semantic_d_mean"] = float(sem_d_mean)
                sem_z_debug["v32_semantic_d_q90"] = float(sem_d_q90)
                sem_z_debug["v32_semantic_trigger_metric"] = (
                    float(trigger_metric) if "trigger_metric" in locals() else None
                )
                sem_z_debug["v32_semantic_z_high_threshold"] = float(self.v32_semantic_cue_trigger_high_threshold)
                sem_z_debug["v32_semantic_trigger_threshold"] = (
                    float(trigger_threshold) if trigger_threshold is not None else None
                )
                sem_z_debug["v32_semantic_trigger_ema_before"] = (
                    float(trigger_ema_before) if trigger_ema_before is not None else None
                )
                sem_z_debug["v32_semantic_trigger_mad_before"] = (
                    float(trigger_mad_before) if trigger_mad_before is not None else None
                )
                sem_z_debug["v32_semantic_trigger_ema_after"] = (
                    float(self._v32_semantic_cue_ema) if self._v32_semantic_cue_ema is not None else None
                )
                sem_z_debug["v32_semantic_trigger_mad_after"] = (
                    float(self._v32_semantic_cue_mad) if self._v32_semantic_cue_mad is not None else None
                )
                sem_z_debug["v32_semantic_trigger_count"] = int(self._v32_semantic_cue_count)
                cue_source_effective = self.read_cue_source
                fallback_rate = float(sem_z_debug.get("v31_semantic_label_fallback_ratio", fallback_rate))
                cue_extra_debug.update(sem_z_debug)
            elif self.read_cue_source == "old_dyn_switch_flow_proxy":
                dyn_mass_flow = float((flow_proxy_calib > 0.5).float().mean().item()) if flow_proxy_calib.numel() else 0.0
                anchor_flow = float(
                    ((flow_proxy_calib * anchor_patch).sum() / flow_proxy_calib.sum().clamp_min(1e-8)).item()
                ) if flow_proxy_calib.numel() else 0.0
                mass_gate = min(max((dyn_mass_flow - 0.02) / 0.04, 0.0), 1.0)
                anchor_gate = min(max((self.read_quality_anchor_max - anchor_flow) / 0.20, 0.0), 1.0)
                cue_gate = float(mass_gate * anchor_gate)
                fallback_rate = 1.0 - cue_gate
                read_patch = _normalize01((1.0 - cue_gate) * dyn_patch + cue_gate * flow_proxy_calib)
            elif self.read_cue_source == "old_dyn_switch_flow_sem_veto":
                dyn_mass_flow = float((flow_sem_veto_calib > 0.5).float().mean().item()) if flow_sem_veto_calib.numel() else 0.0
                anchor_flow = float(
                    ((flow_sem_veto_calib * anchor_patch).sum() / flow_sem_veto_calib.sum().clamp_min(1e-8)).item()
                ) if flow_sem_veto_calib.numel() else 0.0
                mass_gate = min(max((dyn_mass_flow - 0.02) / 0.04, 0.0), 1.0)
                anchor_gate = min(max((self.read_quality_anchor_max - anchor_flow) / 0.20, 0.0), 1.0)
                cue_gate = float(mass_gate * anchor_gate)
                fallback_rate = 1.0 - cue_gate
                read_patch = _normalize01((1.0 - cue_gate) * dyn_patch + cue_gate * flow_sem_veto_calib)
            elif self.read_cue_source == "internal_attn":
                internal = _flatten_optional_patch_map(geo.attn_dynamic_patch)
                read_patch = internal.clamp(0.0, 1.0) if internal is not None and internal.numel() else dyn_patch
            elif self.read_cue_source == "key_cosine_avg":
                internal = _flatten_optional_patch_map(geo.frame_attn_key_cosine_avg)
                read_patch = _normalize01(internal) if internal is not None and internal.numel() else dyn_patch
            elif self.read_cue_source == "key_cosine_shallow":
                internal = _flatten_optional_patch_map(geo.frame_attn_key_cosine_shallow)
                read_patch = _normalize01(internal) if internal is not None and internal.numel() else dyn_patch
            elif self.read_cue_source == "key_cosine_deep":
                internal = _flatten_optional_patch_map(geo.frame_attn_key_cosine_deep)
                read_patch = _normalize01(internal) if internal is not None and internal.numel() else dyn_patch
            elif self.read_cue_source == "query_cosine_avg":
                read_patch = query_avg_patch
            elif self.read_cue_source == "query_cosine_shallow":
                read_patch = query_shallow_patch
            elif self.read_cue_source == "query_cosine_deep":
                read_patch = query_deep_patch
            elif self.read_cue_source == "shallow_deep_disagree":
                read_patch = _normalize01((query_shallow_patch - query_deep_patch).abs())
            elif self.read_cue_source == "fa.key.l0.high.robustq":
                read_patch = fa_key_l0_patch
            elif self.read_cue_source == "fa.key.l4.high.robustq":
                read_patch = fa_key_l4_patch
            elif self.read_cue_source == "fa.key.shallow.high.robustq":
                read_patch = fa_key_shallow_patch
            elif self.read_cue_source == "fa.key.middle.high.robustq":
                read_patch = fa_key_middle_patch
            elif self.read_cue_source == "fa.key.deep.high.robustq":
                read_patch = fa_key_deep_patch
            elif self.read_cue_source == "fa.key.all.high.robustq":
                read_patch = fa_key_all_patch
            elif _is_v95_trackh_gated_l07_source(self.read_cue_source):
                gate_specs = {
                    "gate.fa_key_all_mean_ge_q60.then_v78_l07": {
                        "patch": fa_key_all_patch,
                        "direction": "ge",
                        "threshold": 0.235332936049,
                        "source": "fa_key_all_patch_mean",
                    },
                    "gate.fa_key_middle_mean_ge_q60.then_v78_l07": {
                        "patch": fa_key_middle_patch,
                        "direction": "ge",
                        "threshold": 0.235332936049,
                        "source": "fa_key_middle_patch_mean",
                    },
                    "gate.gg_qk_shallow_mean_le_q40.then_v78_l07": {
                        "patch": gg_qk_shallow_patch,
                        "direction": "le",
                        "threshold": 0.225390625,
                        "source": "gg_qk_shallow_patch_mean",
                    },
                    "gate.gg_qk_shallow_mean_le_q40.chunk_ge_6.then_v78_l07": {
                        "patch": gg_qk_shallow_patch,
                        "direction": "le",
                        "threshold": 0.225390625,
                        "source": "gg_qk_shallow_patch_mean",
                        "min_chunk_idx": 6,
                    },
                    "gate.gg_qk_shallow_mean_le_q40.chunk_ge_6.rtok_ge_0p005.then_v78_l07": {
                        "patch": gg_qk_shallow_patch,
                        "direction": "le",
                        "threshold": 0.225390625,
                        "source": "gg_qk_shallow_patch_mean",
                        "min_chunk_idx": 6,
                        "min_mean_R_tok": 0.005,
                    },
                    "gate.gg_qk_shallow_mean_le_q40.chunk_ge_33.rtok_ge_0p005.then_v78_l07": {
                        "patch": gg_qk_shallow_patch,
                        "direction": "le",
                        "threshold": 0.225390625,
                        "source": "gg_qk_shallow_patch_mean",
                        "min_chunk_idx": 33,
                        "min_mean_R_tok": 0.005,
                    },
                    "gate.gg_qk_shallow_mean_le_q40.chunk_eq_33.rtok_ge_0p005.then_v78_l07": {
                        "patch": gg_qk_shallow_patch,
                        "direction": "le",
                        "threshold": 0.225390625,
                        "source": "gg_qk_shallow_patch_mean",
                        "min_chunk_idx": 33,
                        "max_chunk_idx": 33,
                        "min_mean_R_tok": 0.005,
                    },
                    "gate.gg_qk_shallow_mean_le_q40.chunk_eq_36.rtok_ge_0p005.then_v78_l07": {
                        "patch": gg_qk_shallow_patch,
                        "direction": "le",
                        "threshold": 0.225390625,
                        "source": "gg_qk_shallow_patch_mean",
                        "min_chunk_idx": 36,
                        "max_chunk_idx": 36,
                        "min_mean_R_tok": 0.005,
                    },
                }
                gate_source = str(self.read_cue_source)
                gate_key = _v95_trackh_gated_l07_gate_key(gate_source)
                gate_spec = gate_specs[gate_key]
                active_read_cue = _v95_trackh_gated_l07_active_read_cue(gate_source)
                gate_patch = gate_spec["patch"]
                gate_threshold = float(gate_spec["threshold"])
                gate_direction = str(gate_spec["direction"])
                gate_min_chunk_idx = int(gate_spec.get("min_chunk_idx", -1))
                gate_max_chunk_idx = int(gate_spec.get("max_chunk_idx", -1))
                gate_min_mean_R_tok = gate_spec.get("min_mean_R_tok")
                gate_score = float(gate_patch.mean().item()) if gate_patch.numel() else 0.0
                gate_R_tok = _token_from_patch_values(
                    geo,
                    ((1.0 - occ_patch) * (1.0 - unc_patch)).clamp(0.0, 1.0),
                    special_value=1.0,
                )
                gate_mean_R_tok = float(gate_R_tok.float().mean().item()) if gate_R_tok.numel() else 1.0
                if gate_direction == "le":
                    gate_active = bool(gate_score <= gate_threshold)
                else:
                    gate_active = bool(gate_score >= gate_threshold)
                current_chunk_idx = int(getattr(self, "current_chunk_idx", -1))
                if gate_min_chunk_idx >= 0 and current_chunk_idx < gate_min_chunk_idx:
                    gate_active = False
                if gate_max_chunk_idx >= 0 and current_chunk_idx > gate_max_chunk_idx:
                    gate_active = False
                gate_mean_R_tok_pass = True
                if gate_min_mean_R_tok is not None:
                    gate_mean_R_tok_pass = bool(gate_mean_R_tok >= float(gate_min_mean_R_tok))
                    if not gate_mean_R_tok_pass:
                        gate_active = False
                l07_patch = None
                l07_debug: Dict[str, Any] = {}
                if gate_active:
                    l07_cfg = _parse_v78_l07_l13_read_cue(active_read_cue)
                    if l07_cfg is not None:
                        l07_patch, l07_debug = _v78_l07_l13_read_cue_patch(
                            token_type=token_type,
                            prior_output=prior_output,
                            cfg=l07_cfg,
                            q_layers=geo.global_q_raw_patchvec_layers,
                            k_layers=geo.global_k_raw_patchvec_layers,
                            v_layers=getattr(geo, "global_v_raw_patchvec_layers", None),
                            default=dyn_patch,
                            num_frames=num_frames,
                            chunk_idx=current_chunk_idx,
                        )
                if gate_active and l07_patch is not None:
                    read_patch = l07_patch
                    cue_extra_debug.update(l07_debug)
                    cue_gate = 1.0
                    fallback_rate = 0.0
                    cue_source_effective = f"{self.read_cue_source}:active"
                else:
                    read_patch = torch.zeros_like(dyn_patch)
                    read_skip_calibration = True
                    cue_gate = 0.0
                    fallback_rate = 1.0
                    cue_source_effective = f"{self.read_cue_source}:inactive"
                cue_extra_debug.update(
                    {
                        "v95_trackH_gate_source": str(gate_spec["source"]),
                        "v95_trackH_gate_direction": gate_direction,
                        "v95_trackH_gate_min_chunk_idx": gate_min_chunk_idx if gate_min_chunk_idx >= 0 else None,
                        "v95_trackH_gate_max_chunk_idx": gate_max_chunk_idx if gate_max_chunk_idx >= 0 else None,
                        "v95_trackH_gate_threshold": float(gate_threshold),
                        "v95_trackH_gate_score": float(gate_score),
                        "v95_trackH_gate_min_mean_R_tok": (
                            float(gate_min_mean_R_tok) if gate_min_mean_R_tok is not None else None
                        ),
                        "v95_trackH_gate_mean_R_tok": float(gate_mean_R_tok),
                        "v95_trackH_gate_mean_R_tok_pass": bool(gate_mean_R_tok_pass),
                        "v95_trackH_gate_active": bool(gate_active),
                        "v95_trackH_gate_chunk_idx": int(current_chunk_idx),
                        "v95_trackH_gate_active_read_cue": active_read_cue,
                    }
                )
            elif self.read_cue_source == "fa.key.shallow_deep.decay.robustq":
                read_patch = fa_key_decay_patch
            elif self.read_cue_source == "fa.key.l0_deep.decay.robustq":
                read_patch = fa_key_l0_deep_decay_patch
            elif self.read_cue_source == "fa.key.l4_deep.decay.robustq":
                read_patch = fa_key_l4_deep_decay_patch
            elif self.read_cue_source == "fa.key.layerVar.robustq":
                read_patch = fa_key_layer_var_patch
            elif self.read_cue_source == "fa.key.deep.low.robustq":
                read_patch = fa_key_deep_low_patch
            elif self.read_cue_source == "fa.query.shallow.high.robustq":
                read_patch = fa_query_shallow_patch
            elif self.read_cue_source == "gg.qk.middle.var.robustq":
                read_patch = gg_qk_middle_patch
            elif self.read_cue_source == "gg.qk.shallow.var.robustq":
                read_patch = gg_qk_shallow_patch
            elif self.read_cue_source == "gg.qk.deep.var.robustq":
                read_patch = gg_qk_deep_patch
            elif self.read_cue_source == "gg.qq.middle.low.robustq":
                read_patch = gg_qq_middle_patch
            elif self.read_cue_source == "gg.kk.middle.low.robustq":
                read_patch = gg_kk_middle_patch
            elif self.read_cue_source == "gg.smd.product.a1b1g1.robustq":
                read_patch = gg_smd_a1b1g1_patch
            elif self.read_cue_source == "gg.smd.product.a0b1g1.robustq":
                read_patch = gg_smd_a0b1g1_patch
            elif self.read_cue_source == "mix.exp_add_fa_decay_l05":
                read_patch = (explicit_patch + 0.5 * fa_key_decay_patch).clamp(0.0, 1.0)
            elif self.read_cue_source == "mix.old_agree_fa_decay_l05":
                agree = (1.0 + 0.5 * (2.0 * fa_key_decay_patch - 1.0)).clamp(0.25, 1.75)
                read_patch = (fusion_addclip_patch * agree).clamp(0.0, 1.0)
            elif self.read_cue_source == "mix.old_veto_gg_deep_static_l05":
                read_patch = (fusion_addclip_patch * (1.0 - 0.5 * gg_deep_static_patch)).clamp(0.0, 1.0)
            elif self.read_cue_source == "mix.old_route_gg_smd":
                reliability = (1.0 - (gg_smd_a1b1g1_patch - fusion_addclip_patch).abs()).clamp(0.0, 1.0)
                read_patch = (reliability * fusion_addclip_patch + (1.0 - reliability) * explicit_patch).clamp(0.0, 1.0)
            elif self.read_cue_source == "mix.exp_add_gg_qkvar_l05":
                read_patch = (explicit_patch + 0.5 * gg_qk_middle_patch).clamp(0.0, 1.0)
            else:
                read_patch = dyn_patch

            if (not read_skip_calibration) and self.read_calib_mode == "per_frame_quantile":
                calibrated = _adaptive_quantile_calibrate(
                    read_patch,
                    num_frames=num_frames,
                    target_mass=self.read_target_mass,
                    tau=self.read_calib_tau,
                ).to(read_patch.device)
                blend = min(max(float(self.read_blend_lambda), 0.0), 1.0)
                read_patch = ((1.0 - blend) * read_patch + blend * calibrated).clamp(0.0, 1.0)

            if (not read_skip_calibration) and self.read_topk_frac > 0.0:
                read_patch = _topk_per_frame(read_patch, num_frames=num_frames, frac=self.read_topk_frac)

            safe_patch = (anchor_patch >= self.read_static_anchor_thr) & (dyn_patch <= self.read_static_dyn_thr)
            phase_e_protect_patch, protect_debug = self._build_phase_e_protection_patch(
                mode=self.read_protection_mode,
                num_patch=int(read_patch.numel()),
                num_frames=num_frames,
                safe_patch=safe_patch,
                key_avg_patch=key_stability_base_patch,
            )
            if self.read_protect_static:
                phase_e_protect_patch = torch.maximum(phase_e_protect_patch, safe_patch.float())
            if bool((phase_e_protect_patch > 0.0).any()):
                strength = min(max(float(self.read_ref_strength), 0.0), 1.0)
                read_patch = (read_patch * (1.0 - strength * phase_e_protect_patch)).clamp(0.0, 1.0)
                cue_source_effective = f"{cue_source_effective}+protect:{self.read_protection_mode}"

            D_tok = _token_from_patch_values(geo, read_patch, special_value=0.0).clamp(0.0, 1.0)
            R_tok = _token_from_patch_values(
                geo,
                ((1.0 - occ_patch) * (1.0 - unc_patch)).clamp(0.0, 1.0),
                special_value=1.0,
            )
            P_safe = _token_from_patch_values(geo, phase_e_protect_patch.float(), special_value=0.0)
            old_mask = dyn_patch > 0.5
            new_mask = read_patch > 0.5
            inter = (old_mask & new_mask).float().sum()
            union = (old_mask | new_mask).float().sum().clamp_min(1.0)
            new_count = new_mask.float().sum().clamp_min(1.0)
            old_count = old_mask.float().sum().clamp_min(1.0)
            anchor_collision = (
                float(((read_patch * anchor_patch).sum() / read_patch.sum().clamp_min(1e-8)).item())
                if read_patch.numel()
                else 0.0
            )
            fragmentation = _fragmentation(
                read_patch,
                num_frames=num_frames,
                patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
                thr=0.5,
            )
            dynamic_mass_001 = float((read_patch > 0.01).float().mean().item()) if read_patch.numel() else 0.0
            dynamic_mass_050 = float((read_patch > 0.5).float().mean().item()) if read_patch.numel() else 0.0
            quality_mass_pass = self.read_quality_mass_min <= dynamic_mass_050 <= self.read_quality_mass_max
            quality_anchor_pass = anchor_collision <= self.read_quality_anchor_max
            quality_frag_pass = fragmentation <= self.read_quality_frag_max
            patch_debug = {
                "mean_D_patch": float(read_patch.mean().item()) if read_patch.numel() else 0.0,
                "q10_D_patch": float(torch.quantile(read_patch, 0.1).item()) if read_patch.numel() else 0.0,
                "q50_D_patch": float(torch.quantile(read_patch, 0.5).item()) if read_patch.numel() else 0.0,
                "q90_D_patch": float(torch.quantile(read_patch, 0.9).item()) if read_patch.numel() else 0.0,
                "dynamic_mass_D_gt_001": dynamic_mass_001,
                "dynamic_mass_D_gt_050": dynamic_mass_050,
                "dynamic_mass_D_gt_075": float((read_patch > 0.75).float().mean().item()) if read_patch.numel() else 0.0,
                "anchor_collision": anchor_collision,
                "fragmentation": fragmentation,
                "old_dyn_iou": float((inter / union).item()),
                "old_dyn_coverage": float((inter / new_count).item()),
                "old_dyn_recall": float((inter / old_count).item()),
                "cue_quality_mass_pass": bool(quality_mass_pass),
                "cue_quality_anchor_pass": bool(quality_anchor_pass),
                "cue_quality_frag_pass": bool(quality_frag_pass),
                "cue_quality_pass": bool(quality_mass_pass and quality_anchor_pass and quality_frag_pass),
                "cue_gate": float(cue_gate),
                "fallback_rate": float(fallback_rate),
                "cue_source_effective": cue_source_effective,
                "read_calib_mode": self.read_calib_mode,
                "read_target_mass": self.read_target_mass,
                "read_calib_tau": self.read_calib_tau,
                "read_blend_lambda": self.read_blend_lambda,
                "read_quality_mass_min": self.read_quality_mass_min,
                "read_quality_mass_max": self.read_quality_mass_max,
                "read_quality_anchor_max": self.read_quality_anchor_max,
                "read_quality_frag_max": self.read_quality_frag_max,
                "corr_D_unc": _pearson_corr(read_patch, unc_patch),
                "corr_D_occ": _pearson_corr(read_patch, occ_patch),
                "corr_D_conf": _pearson_corr(read_patch, conf_patch),
                "corr_D_inv_conf": _pearson_corr(read_patch, 1.0 - conf_patch),
                "corr_D_old_dyn": _pearson_corr(read_patch, dyn_patch),
                "flow_model": self.flow_model,
                "flow_pair_stride": self.flow_pair_stride,
                "flow_fb_thr": self.flow_fb_thr,
                "flow_residual_thr": self.flow_residual_thr,
                "gram_layer_groups": self.gram_layer_groups,
                **protect_debug,
                **cue_extra_debug,
                **key_stability_debug,
            }
            patch_debug.update(
                self._dump_read_cue_patch(
                    read_patch=read_patch,
                    num_frames=num_frames,
                    patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
                    chunk_idx=int(getattr(self, "current_chunk_idx", -1)),
                    cue_source_effective=cue_source_effective,
                    patch_debug=patch_debug,
                    extra_patches={
                        "dyn_patch": dyn_patch,
                        "confidence_patch": conf_patch,
                        "uncertainty_patch": unc_patch,
                        "occlusion_patch": occ_patch,
                        "anchor_patch": anchor_patch,
                        "key_avg_patch": key_avg_patch,
                        "key_stability_base_patch": key_stability_base_patch,
                        "key_stability_patch": key_stability_patch,
                        "qk_var_patch": qk_var_patch,
                        "qq_patch": qq_patch,
                        "kk_patch": kk_patch,
                        "dyn4d_patch": dyn4d_patch,
                        "query_avg_patch": query_avg_patch,
                        "query_shallow_patch": query_shallow_patch,
                        "query_deep_patch": query_deep_patch,
                        "fa_query_shallow_patch": fa_query_shallow_patch,
                        "fa_key_l0_patch": fa_key_l0_patch,
                        "fa_key_l4_patch": fa_key_l4_patch,
                        "fa_key_shallow_patch": fa_key_shallow_patch,
                        "fa_key_middle_patch": fa_key_middle_patch,
                        "fa_key_deep_patch": fa_key_deep_patch,
                        "fa_key_all_patch": fa_key_all_patch,
                        "fa_key_layer_var_patch": fa_key_layer_var_patch,
                        "fa_key_decay_patch": fa_key_decay_patch,
                        "fa_key_l0_deep_decay_patch": fa_key_l0_deep_decay_patch,
                        "fa_key_l4_deep_decay_patch": fa_key_l4_deep_decay_patch,
                        "fa_key_deep_low_patch": fa_key_deep_low_patch,
                        "gg_qk_middle_patch": gg_qk_middle_patch,
                        "gg_qk_shallow_patch": gg_qk_shallow_patch,
                        "gg_qk_deep_patch": gg_qk_deep_patch,
                        "gg_qq_middle_patch": gg_qq_middle_patch,
                        "gg_kk_middle_patch": gg_kk_middle_patch,
                        "gg_qq_shallow_patch": gg_qq_shallow_patch,
                        "gg_deep_static_patch": gg_deep_static_patch,
                        "gg_smd_a1b1g1_patch": gg_smd_a1b1g1_patch,
                        "gg_smd_a0b1g1_patch": gg_smd_a0b1g1_patch,
                    },
                )
            )
        else:
            D_tok = torch.zeros(L_tok, dtype=torch.float32)
            R_tok = torch.ones(L_tok, dtype=torch.float32)
            P_safe = torch.zeros(L_tok, dtype=torch.float32)
            patch_debug = {}

        U_tok = torch.ones(L_tok, dtype=torch.float32)
        P_ref = (token_type != TOKEN_TYPE_PATCH).float() if self.read_protect_ref else torch.zeros(L_tok, dtype=torch.float32)
        if bool((P_safe > 0.0).any()):
            P_ref = torch.maximum(P_ref, P_safe)
        D_swa_write_tok = D_tok.detach().cpu().float().clone()
        if cue is not None and cue.E_cue_patch is not None:
            key_stability_tok = _token_from_patch_values(
                geo,
                key_stability_patch,
                special_value=0.0,
            ).clamp(0.0, 1.0)
            patch_debug.update({
                "key_stability_tok_gt050_mass": (
                    float((key_stability_tok > 0.50).float().mean().item())
                    if key_stability_tok.numel()
                    else 0.0
                ),
            })
        else:
            key_stability_tok = torch.zeros_like(D_tok)
        raw_frame_bias_energy = _frame_bias_energy(
            D_tok,
            P_ref,
            num_frames=num_frames,
            mode=self.frame_bias_mode,
            K_stable_tok=key_stability_tok,
        )
        beta_frame_effective = self.beta_frame
        beta_was_clipped = False
        if self.beta_policy == "bias_energy_norm" and raw_frame_bias_energy > 0.0 and self.beta_energy_target > 0.0:
            unclipped = self.beta_frame * self.beta_energy_target / max(raw_frame_bias_energy, 1e-8)
            beta_frame_effective = min(max(unclipped, self.beta_min), self.beta_max)
            beta_was_clipped = abs(beta_frame_effective - unclipped) > 1e-8
        P_ttt_write = None
        B_chunk_geo = None
        if prior_output is not None:
            P_ttt_write = prior_output.A_tok.detach().cpu().float()
            B_chunk_geo = float(prior_output.B_chunk_geo)
            S_tok = _token_from_patch_values(
                geo,
                getattr(prior_output, "V_sem_patch_flat", torch.ones(0)),
                special_value=1.0,
            ) if getattr(prior_output, "V_sem_patch_flat", None) is not None else None
            G_sem_tok = getattr(prior_output, "G_sem_tok", None)
            if G_sem_tok is not None:
                G_sem_tok = G_sem_tok.detach().cpu().long()
            Q_sem_tok = getattr(prior_output, "Q_sem_tok", None)
            if Q_sem_tok is not None:
                Q_sem_tok = Q_sem_tok.detach().cpu().float()
            L_sem_tok = getattr(prior_output, "L_sem_tok", None)
            if L_sem_tok is not None:
                L_sem_tok = L_sem_tok.detach().cpu().long()
            V_sem_tok = getattr(prior_output, "V_sem_tok", None)
            if V_sem_tok is not None:
                V_sem_tok = V_sem_tok.detach().cpu().float()
            R_sem_tok = getattr(prior_output, "R_sem_tok", None)
            if R_sem_tok is not None:
                R_sem_tok = R_sem_tok.detach().cpu().long()
            R_frame_tok = getattr(prior_output, "R_frame_tok", None)
            if R_frame_tok is not None:
                R_frame_tok = R_frame_tok.detach().cpu().long()
            R_global_tok = getattr(prior_output, "R_global_tok", None)
            if R_global_tok is not None:
                R_global_tok = R_global_tok.detach().cpu().long()
            R_swa_tok = getattr(prior_output, "R_swa_tok", None)
            if R_swa_tok is not None:
                R_swa_tok = R_swa_tok.detach().cpu().long()
            R_ttt_tok = getattr(prior_output, "R_ttt_tok", None)
            if R_ttt_tok is not None:
                R_ttt_tok = R_ttt_tok.detach().cpu().long()
            stage_c_seed_global_track_idx_tok = getattr(
                prior_output,
                "stage_c_seed_global_track_idx_tok",
                None,
            )
            if stage_c_seed_global_track_idx_tok is not None:
                stage_c_seed_global_track_idx_tok = stage_c_seed_global_track_idx_tok.detach().cpu().long()
            stage_c_masklet_instance_idx_tok = getattr(
                prior_output,
                "stage_c_masklet_instance_idx_tok",
                None,
            )
            if stage_c_masklet_instance_idx_tok is not None:
                stage_c_masklet_instance_idx_tok = stage_c_masklet_instance_idx_tok.detach().cpu().long()
            override_write, write_debug = self._phase_e_write_prior(
                probe=probe,
                token_type=token_type,
                dyn_patch=dyn_patch if cue is not None and cue.E_cue_patch is not None else None,
                explicit_dyn_patch=explicit_patch if cue is not None and cue.E_cue_patch is not None else None,
                occ_patch=occ_patch if cue is not None and cue.E_cue_patch is not None else None,
                unc_patch=unc_patch if cue is not None and cue.E_cue_patch is not None else None,
                read_patch=read_patch if cue is not None and cue.E_cue_patch is not None else None,
                P_ref=P_ref,
                base_write_prior=P_ttt_write,
                semantic_value_patch=getattr(prior_output, "V_sem_patch_flat", None),
            )
            if override_write is not None:
                P_ttt_write = override_write
                B_chunk_geo = 1.0
            patch_debug.update(write_debug)
        else:
            S_tok = None
            G_sem_tok = None
            Q_sem_tok = None
            L_sem_tok = None
            V_sem_tok = None
            R_sem_tok = None
            R_frame_tok = None
            R_global_tok = None
            R_swa_tok = None
            R_ttt_tok = None
            stage_c_seed_global_track_idx_tok = None
            stage_c_masklet_instance_idx_tok = None
            patch_debug.update({"hmc_write_score_source": self.hmc_write_score_source, "hmc_write_override": False})

        if mode in {"unity_replay", "native", "identity_hooks", "read_path_only", "probe_only"}:
            P_ttt_write = None
            B_chunk_geo = None
        (
            R_sem_tok,
            R_frame_tok,
            R_global_tok,
            R_swa_tok,
            R_ttt_tok,
            P_ttt_write,
            D_swa_write_tok,
            role_debug,
        ) = self._apply_semantic_role_policy(
            token_type=token_type,
            D_tok=D_tok,
            P_ref=P_ref,
            V_sem_tok=V_sem_tok,
            G_sem_tok=G_sem_tok,
            Q_sem_tok=Q_sem_tok,
            L_sem_tok=L_sem_tok,
            R_sem_tok=R_sem_tok,
            R_frame_tok=R_frame_tok,
            R_global_tok=R_global_tok,
            R_swa_tok=R_swa_tok,
            R_ttt_tok=R_ttt_tok,
            P_ttt_write=P_ttt_write,
            D_swa_write_tok=D_swa_write_tok,
            read_patch_tok=_token_from_patch_values(geo, read_patch, special_value=0.0),
            key_stability_tok=key_stability_tok,
            num_frames=num_frames,
            patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
        )
        swa_redirection_source_mask_tok = self._swa_redirection_source_mask_from_roles(
            R_frame_tok,
            R_global_tok,
            role_debug,
        )
        patch_debug.update(role_debug)

        A_anchor_tok, A_anchor_mask_tok, anchor_debug = self._build_semantic_anchor_bank(
            token_type=token_type,
            D_tok=D_tok,
            P_ttt_write=P_ttt_write,
            G_sem_tok=G_sem_tok,
            Q_sem_tok=Q_sem_tok,
            L_sem_tok=L_sem_tok,
            S_tok=S_tok,
            V_sem_tok=V_sem_tok,
            conf_patch=conf_patch if cue is not None and cue.E_cue_patch is not None else None,
            key_avg_patch=key_stability_base_patch if cue is not None and cue.E_cue_patch is not None else None,
            num_frames=num_frames,
            patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
        )
        P_ttt_write, anchor_floor_debug = self._apply_semantic_anchor_write_floor(
            token_type=token_type,
            P_ttt_write=P_ttt_write,
            A_anchor_tok=A_anchor_tok,
        )
        patch_debug.update(anchor_debug)
        patch_debug.update(anchor_floor_debug)
        P_ttt_write, v70_ttt_debug = self._apply_v70_radio_ttt_write_prior(
            token_type=token_type,
            P_ttt_write=P_ttt_write,
            num_frames=num_frames,
            patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            chunk_idx=int(getattr(self, "current_chunk_idx", -1)),
        )
        patch_debug.update(v70_ttt_debug)
        P_ttt_write, semantic_ttt_support_debug = self._apply_semantic_ttt_overlap_support_prior(
            token_type=token_type,
            P_ttt_write=P_ttt_write,
            num_frames=num_frames,
            patch_grid=(int(geo.patch_grid[0]), int(geo.patch_grid[1])),
            chunk_idx=int(getattr(self, "current_chunk_idx", -1)),
        )
        patch_debug.update(semantic_ttt_support_debug)

        frame_read_effective_enabled = bool(self.enable_frame_read_control)
        if (
            _is_v95_trackh_gated_l07_source(self.read_cue_source)
            and cue_extra_debug.get("v95_trackH_gate_active") is False
        ):
            frame_read_effective_enabled = False
            cue_extra_debug["v95_trackH_gate_strict_noop"] = True
        elif _is_v95_trackh_gated_l07_source(self.read_cue_source):
            cue_extra_debug["v95_trackH_gate_strict_noop"] = False

        return HybridMemoryControlPrior(
            D_tok=D_tok,
            R_tok=R_tok,
            U_tok=U_tok,
            P_ref=P_ref,
            S_tok=S_tok,
            G_sem_tok=G_sem_tok,
            Q_sem_tok=Q_sem_tok,
            L_sem_tok=L_sem_tok,
            V_sem_tok=V_sem_tok,
            R_sem_tok=R_sem_tok,
            R_frame_tok=R_frame_tok,
            R_global_tok=R_global_tok,
            R_swa_tok=R_swa_tok,
            R_ttt_tok=R_ttt_tok,
            stage_c_seed_global_track_idx_tok=stage_c_seed_global_track_idx_tok,
            stage_c_masklet_instance_idx_tok=stage_c_masklet_instance_idx_tok,
            K_stable_tok=key_stability_tok,
            swa_redirection_source_mask_tok=swa_redirection_source_mask_tok,
            A_anchor_tok=A_anchor_tok,
            A_anchor_mask_tok=A_anchor_mask_tok,
            C_ttt_conflict_tok=(
                _optional_1d_float_tensor(getattr(self, "semantic_condition_conflict_tok", None))
                if bool(role_debug.get("condition_signal_conflict_token_exact", False))
                else None
            ),
            S_scale_risk_tok=(
                _optional_1d_float_tensor(getattr(self, "semantic_condition_scale_tok", None))
                if bool(role_debug.get("condition_signal_scale_risk_token_exact", False))
                else None
            ),
            P_ttt_write=P_ttt_write,
            P_ttt_read=None,
            P_swa_read_prev=None,
            D_swa_write_tok=D_swa_write_tok,
            frame_bias_spec={
                "enabled": frame_read_effective_enabled,
                "beta": self.beta_frame,
                "beta_effective": beta_frame_effective,
                "mode": self.frame_bias_mode,
                "cue_source": self.read_cue_source,
                "topk_frac": self.read_topk_frac,
                "calib_mode": self.read_calib_mode,
                "target_mass": self.read_target_mass,
                "calib_tau": self.read_calib_tau,
                "blend_lambda": self.read_blend_lambda,
                "protection_mode": self.read_protection_mode,
                "ref_strength": self.read_ref_strength,
            },
            chunk_bias_spec={
                "enabled": self.enable_chunk_read_control,
                "beta": self.beta_chunk,
                "mode": self.chunk_bias_mode,
                "cue_source": self.read_cue_source,
            },
            B_chunk_geo=B_chunk_geo,
            debug={
                "mode": mode,
                "mean_D_tok": float(D_tok.mean().item()) if D_tok.numel() else 0.0,
                "max_D_tok": float(D_tok.max().item()) if D_tok.numel() else 0.0,
                "q90_D_tok": float(torch.quantile(D_tok, 0.9).item()) if D_tok.numel() else 0.0,
                "mean_R_tok": float(R_tok.mean().item()) if R_tok.numel() else 1.0,
                "protected_token_count": int((P_ref > 0.5).sum().item()),
                "safe_patch_token_count": int((P_safe > 0.5).sum().item()),
                "patch_token_count": int(patch_mask.sum().item()),
                "read_cue_source": self.read_cue_source,
                "v70_radio_sidecar_dir": self.v70_radio_sidecar_dir,
                "read_topk_frac": self.read_topk_frac,
                "read_calib_mode": self.read_calib_mode,
                "read_target_mass": self.read_target_mass,
                "read_calib_tau": self.read_calib_tau,
                "read_blend_lambda": self.read_blend_lambda,
                "read_quality_mass_min": self.read_quality_mass_min,
                "read_quality_mass_max": self.read_quality_mass_max,
                "read_quality_anchor_max": self.read_quality_anchor_max,
                "read_quality_frag_max": self.read_quality_frag_max,
                "beta_policy": self.beta_policy,
                "beta_frame_effective": beta_frame_effective,
                "beta_energy_target": self.beta_energy_target,
                "beta_raw_frame_bias_energy": raw_frame_bias_energy,
                "beta_was_clipped": bool(beta_was_clipped),
                "frame_bias_mode": self.frame_bias_mode,
                "chunk_bias_mode": self.chunk_bias_mode,
                "swa_bias_mode": self.swa_bias_mode,
                "read_protect_ref": self.read_protect_ref,
                "read_protect_static": self.read_protect_static,
                "read_protection_mode": self.read_protection_mode,
                "read_ref_strength": self.read_ref_strength,
                "read_overlap_frames": self.read_overlap_frames,
                "read_reset_frames": self.read_reset_frames,
                "read_attention_q": self.read_attention_q,
                "hmc_write_score_source": self.hmc_write_score_source,
                "hmc_write_sparse_ratio": self.hmc_write_sparse_ratio,
                "hmc_write_sparse_mode": self.hmc_write_sparse_mode,
                "ttt_write_prior_present": P_ttt_write is not None,
                "ttt_write_prior_mean": float(P_ttt_write.mean().item()) if P_ttt_write is not None else 1.0,
                "read_path_control": "real_hook_identity_or_requested",
                **patch_debug,
            },
        )

    def run_controlled(
        self,
        backbone: LoGeRGeometryBackbone,
        images: torch.Tensor,
        state_m: Optional[HybridMemoryState],
        control_prior: Optional[HybridMemoryControlPrior],
        *,
        mode: str,
        token_type: Optional[torch.Tensor] = None,
        skip_ttt_write_replay: bool = False,
    ) -> HybridMemoryResult:
        if mode == "probe_only":
            raise ValueError("probe_only should be handled by the caller after run_probe().")

        input_hash = hybrid_state_fingerprint(state_m)
        model_hmc_control = self._build_model_hmc_control(control_prior, mode=mode)
        if state_m is not None and state_m.prev_control_summary is not None:
            prev_D_patch = state_m.prev_control_summary.get("D_patch")
            if prev_D_patch is not None:
                model_hmc_control["D_prev_patch"] = prev_D_patch
            prev_G_patch = state_m.prev_control_summary.get("G_patch")
            if prev_G_patch is not None:
                model_hmc_control["G_prev_patch"] = prev_G_patch
            prev_L_patch = state_m.prev_control_summary.get("L_patch")
            if prev_L_patch is not None:
                model_hmc_control["L_prev_patch"] = prev_L_patch
            prev_stage_c_seed_patch = state_m.prev_control_summary.get("stage_c_seed_global_track_idx_patch")
            if prev_stage_c_seed_patch is not None:
                model_hmc_control["stage_c_seed_global_track_idx_prev_patch"] = prev_stage_c_seed_patch
            prev_stage_c_masklet_instance_patch = state_m.prev_control_summary.get("stage_c_masklet_instance_idx_patch")
            if prev_stage_c_masklet_instance_patch is not None:
                model_hmc_control["stage_c_masklet_instance_idx_prev_patch"] = prev_stage_c_masklet_instance_patch
            for prev_key in (
                "ttt_stable_anchor_mask_patch",
                "ttt_stable_anchor_id_patch",
                "ttt_stable_anchor_retention_patch",
                "ttt_stable_anchor_residual_patch",
                "ttt_stable_anchor_z_write_key_norm_patch",
                "ttt_stable_anchor_z_write_key_vec_patch",
                "ttt_stable_anchor_z_write_key_sketch_patch",
                "ttt_stable_anchor_z_write_hidden_vec_patch",
                "ttt_tracked_instance_anchor_mask_patch",
                "ttt_tracked_instance_anchor_id_patch",
                "ttt_tracked_instance_anchor_seed_patch",
            ):
                prev_value = state_m.prev_control_summary.get(prev_key)
                if prev_value is not None:
                    model_hmc_control[f"prev_{prev_key}"] = prev_value
            if "ttt_stable_anchor_source_chunk_idx" in state_m.prev_control_summary:
                model_hmc_control["prev_ttt_stable_anchor_source_chunk_idx"] = int(
                    state_m.prev_control_summary.get("ttt_stable_anchor_source_chunk_idx", -1)
                )
            if "ttt_stable_anchor_token_count" in state_m.prev_control_summary:
                model_hmc_control["prev_ttt_stable_anchor_token_count"] = int(
                    state_m.prev_control_summary.get("ttt_stable_anchor_token_count", 0) or 0
                )
            if "ttt_tracked_instance_anchor_source_chunk_idx" in state_m.prev_control_summary:
                model_hmc_control["prev_ttt_tracked_instance_anchor_source_chunk_idx"] = int(
                    state_m.prev_control_summary.get("ttt_tracked_instance_anchor_source_chunk_idx", -1)
                )
            if "ttt_tracked_instance_anchor_token_count" in state_m.prev_control_summary:
                model_hmc_control["prev_ttt_tracked_instance_anchor_token_count"] = int(
                    state_m.prev_control_summary.get("ttt_tracked_instance_anchor_token_count", 0) or 0
                )
        geo, write_cache = backbone.run(
            images,
            ttt_state=state_m.to_ttt_input() if state_m is not None else None,
            cache_ttt_primitives=True,
            hmc_control=model_hmc_control,
        )

        write_result: Optional[WriteResult] = None
        if mode in {"native", "identity_hooks"}:
            ttt_next = self._state_from_write_cache(write_cache)
            write_debug = {
                "mode": mode,
                "native_write_through": True,
                "ttt_layers_committed": int(write_cache.num_ttt_layers),
            }
        else:
            if mode in {"unity_replay", "read_path_only"}:
                write_mode = "unity_replay"
                A_tok = None
                B_chunk_geo = None
            elif mode in {"ttt_write_only", "hybrid"}:
                write_mode = "semantic"
                A_tok = control_prior.P_ttt_write if control_prior is not None else None
                B_chunk_geo = control_prior.B_chunk_geo if control_prior is not None else None
            else:
                raise ValueError(f"Unsupported hybrid memory mode: {mode}")

            if skip_ttt_write_replay:
                ttt_next = self._state_from_write_cache(write_cache)
                write_debug = {
                    "mode": write_mode,
                    "semantic_write_skipped": True,
                    "skip_reason": "probe_ttt_write_commit_replays_probe_cache",
                    "ttt_layers_committed": int(write_cache.num_ttt_layers),
                }
            else:
                old_mode = self.ttt_update_controller.write_mode
                self.ttt_update_controller.write_mode = write_mode
                self.ttt_update_controller.current_chunk_idx = int(getattr(self, "current_chunk_idx", -1))
                scale_state_debug = self._configure_ttt_write_scale_state(
                    geo,
                    state_m.prev_control_summary if state_m is not None else None,
                )
                write_cache_for_commit, dual_lifetime_debug = _write_cache_with_long_old_weights_for_dual_lifetime(
                    write_cache,
                    state_m.ttt_state if state_m is not None else None,
                )
                write_result = self.ttt_update_controller.run(
                    write_cache_for_commit,
                    A_tok,
                    B_chunk_geo=B_chunk_geo,
                    device=self.device,
                    token_type=token_type,
                    num_frames=int(geo.num_frames),
                    overlap_frames=int(self.read_overlap_frames),
                    risk_tok=control_prior.D_tok if control_prior is not None else None,
                    role_tok=control_prior.R_ttt_tok if control_prior is not None else None,
                    prev_transient_delta=(
                        state_m.ttt_state.get("transient_delta")
                        if state_m is not None and isinstance(state_m.ttt_state, dict)
                        else None
                    ),
                )
                self.ttt_update_controller.write_mode = old_mode
                write_result.debug.update(dual_lifetime_debug)
                write_result.debug["ttt_scale_state_diagnostic"] = scale_state_debug
                gated_history, swa_write_debug = self._apply_swa_history_write_gate(
                    write_result.history,
                    control_prior,
                    geo,
                )
                write_result.history = gated_history
                write_result.debug.update(swa_write_debug)
                ttt_next = {"w0": write_result.w0, "w1": write_result.w1, "w2": write_result.w2}
                if write_result.history is not None:
                    ttt_next["history"] = write_result.history
                if write_result.transient_delta is not None:
                    ttt_next["transient_delta"] = write_result.transient_delta
                write_debug = write_result.debug

        state_next = HybridMemoryState(
            ttt_state=ttt_next,
            swa_state=None,
            ref_state=None,
            prev_control_summary=self._build_prev_control_summary(
                control_prior,
                geo,
                prev_control_summary=state_m.prev_control_summary if state_m is not None else None,
                ttt_token_contribution_diagnostic=(
                    write_result.token_contribution_diagnostic if write_result is not None else None
                ),
                chunk_idx=int(getattr(self, "current_chunk_idx", -1)),
            ),
        )
        output_hash = hybrid_state_fingerprint(state_next)
        trace_dict = geo.hmc_trace or {}
        hook_effect_summary: Dict[str, Dict[str, Any]] = {}
        for path_name, records in trace_dict.items():
            if not isinstance(records, list):
                continue
            mean_vals = [
                float(rec.get("mean_abs_bias", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("mean_abs_bias") is not None
            ]
            mean_vals.extend([
                float(rec.get("swa_overlap_bias_mean_abs", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_bias_mean_abs") is not None
            ])
            max_vals = [
                float(rec.get("max_abs_bias", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("max_abs_bias") is not None
            ]
            max_vals.extend([
                float(rec.get("swa_overlap_bias_max_abs", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_bias_max_abs") is not None
            ])
            gate_mean_vals = [
                float(rec.get("swa_gate_mean", 1.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_gate_mean") is not None
            ]
            overlap_source_gate_mean_vals = [
                float(rec.get("swa_overlap_source_gate_mean", 1.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_gate_mean") is not None
            ]
            gate_delta_vals = [
                float(rec.get("mean_abs_gate_delta", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("mean_abs_gate_delta") is not None
            ]
            overlap_source_gate_delta_vals = [
                float(rec.get("swa_overlap_source_gate_mean_abs_delta", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_gate_mean_abs_delta") is not None
            ]
            max_gate_delta_vals = [
                float(rec.get("max_abs_gate_delta", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("max_abs_gate_delta") is not None
            ]
            overlap_source_max_gate_delta_vals = [
                float(rec.get("swa_overlap_source_gate_max_abs_delta", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_gate_max_abs_delta") is not None
            ]
            overlap_source_score_vals = [
                float(rec.get("swa_overlap_source_gate_score_mean", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_gate_score_mean") is not None
            ]
            overlap_source_score_q90_vals = [
                float(rec.get("swa_overlap_source_gate_score_q90", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_gate_score_q90") is not None
            ]
            overlap_source_replace_alpha_vals = [
                float(rec.get("swa_overlap_source_replace_alpha_mean", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_replace_alpha_mean") is not None
            ]
            overlap_source_replace_alpha_p90_vals = [
                float(rec.get("swa_overlap_source_replace_alpha_p90", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_replace_alpha_p90") is not None
            ]
            overlap_source_replace_score_vals = [
                float(rec.get("swa_overlap_source_replace_score_mean", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_overlap_source_replace_score_mean") is not None
            ]
            prev_ttt_anchor_gate_mean_vals = [
                float(rec.get("swa_prev_ttt_stable_anchor_gate_mean", 1.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_prev_ttt_stable_anchor_gate_mean") is not None
            ]
            prev_ttt_anchor_gate_delta_vals = [
                float(rec.get("swa_prev_ttt_stable_anchor_gate_mean_abs_delta", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_prev_ttt_stable_anchor_gate_mean_abs_delta") is not None
            ]
            prev_ttt_anchor_max_gate_delta_vals = [
                float(rec.get("swa_prev_ttt_stable_anchor_gate_max_abs_delta", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_prev_ttt_stable_anchor_gate_max_abs_delta") is not None
            ]
            prev_ttt_anchor_gate_token_vals = [
                int(rec.get("swa_prev_ttt_stable_anchor_gate_tokens", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            prev_ttt_anchor_gate_frac_vals = [
                float(rec.get("swa_prev_ttt_stable_anchor_gate_selected_frac", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("swa_prev_ttt_stable_anchor_gate_selected_frac") is not None
            ]
            prev_ttt_anchor_query_soft_source_token_vals = [
                int(rec.get("prev_ttt_anchor_query_soft_source_tokens", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            prev_ttt_anchor_query_soft_selected_query_vals = [
                int(rec.get("prev_ttt_anchor_query_soft_selected_query_count", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            prev_ttt_anchor_query_soft_selected_frac_vals = [
                float(rec.get("prev_ttt_anchor_query_soft_query_selected_frac", 0.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("prev_ttt_anchor_query_soft_query_selected_frac") is not None
            ]
            prev_ttt_tracked_instance_query_soft_source_token_vals = [
                int(rec.get("prev_ttt_tracked_instance_query_soft_source_tokens", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            prev_ttt_tracked_instance_query_soft_selected_query_vals = [
                int(rec.get("prev_ttt_tracked_instance_query_soft_selected_query_count", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            prev_ttt_tracked_instance_query_soft_selected_frac_vals = [
                float(rec.get("prev_ttt_tracked_instance_query_soft_query_selected_frac", 0.0))
                for rec in records
                if isinstance(rec, dict)
                and rec.get("prev_ttt_tracked_instance_query_soft_query_selected_frac") is not None
            ]
            prev_ttt_tracked_instance_query_soft_seed_count_vals = [
                int(rec.get("prev_ttt_tracked_instance_query_soft_direct_witness_seed_count", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            prev_ttt_tracked_instance_query_soft_rule_pass_vals = [
                bool(rec.get("prev_ttt_tracked_instance_query_soft_carrier_rule_pass", False))
                for rec in records
                if isinstance(rec, dict)
                and rec.get("prev_ttt_tracked_instance_query_soft_carrier_rule_pass") is not None
            ]
            source_keep_ratio_vals = [
                float(rec.get("source_keep_ratio", 1.0))
                for rec in records
                if isinstance(rec, dict) and rec.get("source_keep_ratio") is not None
            ]
            source_skip_token_vals = [
                int(rec.get("source_skip_tokens", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            source_control_token_vals = [
                int(rec.get("source_control_tokens", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            source_boost_token_vals = [
                int(rec.get("source_boost_tokens", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            source_boost_applied_vals = [
                bool(rec.get("semantic_anchor_boost_applied", False))
                for rec in records
                if isinstance(rec, dict)
            ]
            empty_source_event_vals = [
                int(rec.get("empty_source_events", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            history_token_vals = [
                int(rec.get("history_tokens", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            d_prev_token_vals = [
                int(rec.get("d_prev_tokens", 0) or 0)
                for rec in records
                if isinstance(rec, dict)
            ]
            attention_mass_removed_before_vals = [
                float(rec.get("attention_mass_removed_before"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_removed_before") is not None
            ]
            attention_mass_removed_after_vals = [
                float(rec.get("attention_mass_removed_after"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_removed_after") is not None
            ]
            attention_mass_retained_before_vals = [
                float(rec.get("attention_mass_retained_before"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_retained_before") is not None
            ]
            attention_mass_retained_after_vals = [
                float(rec.get("attention_mass_retained_after"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_retained_after") is not None
            ]
            attention_mass_actual_after_vals = [
                float(rec.get("attention_mass_actual_after"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_actual_after") is not None
            ]
            attention_mass_query_sample_vals = [
                float(rec.get("attention_mass_query_sample_tokens_mean"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_query_sample_tokens_mean") is not None
            ]
            attention_mass_removed_token_vals = [
                float(rec.get("attention_mass_removed_tokens_mean"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_removed_tokens_mean") is not None
            ]
            attention_mass_retained_token_vals = [
                float(rec.get("attention_mass_retained_tokens_mean"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_retained_tokens_mean") is not None
            ]
            attention_mass_stable_anchor_before_vals = [
                float(rec.get("attention_mass_stable_anchor_before"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_stable_anchor_before") is not None
            ]
            attention_mass_stable_anchor_after_vals = [
                float(rec.get("attention_mass_stable_anchor_after"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_stable_anchor_after") is not None
            ]
            attention_mass_stable_anchor_actual_after_vals = [
                float(rec.get("attention_mass_stable_anchor_actual_after"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_stable_anchor_actual_after") is not None
            ]
            attention_mass_stable_anchor_token_vals = [
                float(rec.get("attention_mass_stable_anchor_tokens_mean"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_stable_anchor_tokens_mean") is not None
            ]
            attention_mass_stable_anchor_preservation_vals = [
                float(rec.get("attention_mass_stable_anchor_preservation_ratio"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_stable_anchor_preservation_ratio") is not None
            ]
            attention_mass_metrics = sorted({
                str(rec.get("attention_mass_metric"))
                for rec in records
                if isinstance(rec, dict) and rec.get("attention_mass_metric") is not None
            })
            extra_numeric_summary: Dict[str, float] = {}
            for extra_key in (
                "context_source_selected_token_count",
                "context_source_selected_token_ratio",
                "context_source_selected_fine_sky_frac",
                "context_source_selected_group_structure_frac",
                "context_source_selected_group_static_frac",
                "context_source_selected_group_movable_frac",
                "context_source_selected_group_lowstuff_frac",
                "context_source_selected_group_uncertain_frac",
                "v40_r2_sky_token_count",
                "v40_r2_sky_token_ratio",
                "v40_r2_sky_highd_token_ratio",
                "v40_r2_source_mass_proxy_ratio",
                "v40_r2_sky_highd_threshold",
                "v40_r2_global_keep_proxy_after",
                "v40_r3_highd_quantile",
                "v40_r3_highd_threshold",
                "v67_phase2_highd_quantile",
                "v67_phase2_highd_threshold",
                "v67_source_attention_top_quantile",
                "v67_source_attention_semantic_group_id",
                "v67_source_attention_group_eligible_tokens",
                "swa_overlap_source_semantic_highd_quantile",
                "swa_overlap_source_semantic_tokens_before_random",
                "swa_overlap_source_semantic_selected_tokens",
                "swa_overlap_source_semantic_selected_ratio",
                "swa_overlap_source_semantic_selected_index_mean",
                "swa_overlap_source_semantic_selected_D_mean",
                "swa_overlap_attention_mass_source_before",
                "swa_overlap_attention_mass_source_after",
                "swa_overlap_attention_mass_source_lift",
                "swa_overlap_attention_mass_selected_before",
                "swa_overlap_attention_mass_selected_after",
                "swa_overlap_attention_mass_selected_lift",
                "swa_overlap_attention_mass_selected_head_max_before",
                "swa_overlap_attention_mass_selected_head_max_after",
                "swa_overlap_attention_mass_selected_head_max_lift",
                "source_control_score_mean",
                "source_control_score_max",
                "source_weight_mean",
                "source_weight_min",
                "semantic_anchor_rescue_source_tokens",
                "semantic_anchor_rescue_source_ratio",
                "semantic_anchor_rescue_source_score_mean",
                "semantic_anchor_rescue_source_score_max",
                "frame_bias_positive_pair_mass_before",
                "frame_bias_positive_pair_mass_after",
                "frame_bias_positive_pair_mass_lift",
                "frame_bias_negative_pair_mass_before",
                "frame_bias_negative_pair_mass_after",
                "frame_bias_negative_pair_mass_lift",
                "frame_bias_positive_pair_count_mean",
                "frame_bias_negative_pair_count_mean",
                "frame_bias_positive_pair_fraction",
                "frame_bias_negative_pair_fraction",
                "frame_bias_positive_bias_mean",
                "frame_bias_negative_bias_mean",
                "frame_bias_attention_mass_query_count",
                "swa_raw_transport_trace_sampled_query_count",
                "swa_raw_transport_trace_head_count",
                "swa_raw_transport_current_tokens",
                "swa_raw_transport_history_tokens",
                "swa_raw_transport_d_prev_low_pair_tokens",
                "swa_raw_transport_d_prev_high_pair_tokens",
                "swa_raw_transport_g_prev_high_pair_tokens",
                "swa_raw_transport_k_stable_pair_tokens",
                "swa_raw_transport_label_static_structure_pair_tokens",
                "swa_raw_transport_stable_pair_strict_tokens",
                "swa_raw_transport_stable_pair_semantic_lowd_tokens",
                "swa_raw_transport_stable_pair_lowd_nonunreliable_tokens",
                "swa_raw_transport_stable_pair_tokens",
                "swa_raw_transport_unreliable_pair_tokens",
                "swa_raw_transport_qk_similarity_mean",
                "swa_raw_transport_qk_similarity_max_mean",
                "swa_raw_transport_route_entropy_mean",
                "swa_raw_transport_feature_residual_mean",
                "swa_raw_transport_cache_k_stability_mean",
                "swa_raw_transport_cache_v_stability_mean",
                "swa_raw_transport_stable_pair_mass_mean",
                "swa_raw_transport_unreliable_pair_mass_mean",
                "swa_raw_transport_stable_actual_minus_random_mean",
                "swa_raw_transport_unreliable_actual_minus_random_mean",
                "swa_raw_transport_topk_identity_available",
                "swa_raw_transport_topk_identity_topk",
                "swa_raw_transport_top1_cache_index_unique_frac_mean",
                "swa_raw_transport_top1_cache_frame_unique_frac_mean",
                "swa_raw_transport_top1_cache_index_switch_rate_mean",
                "swa_raw_transport_top1_cache_frame_switch_rate_mean",
                "swa_raw_transport_top1_same_frame_frac_mean",
                "swa_raw_transport_topk_query_frame_hit_frac_mean",
                "swa_raw_transport_topk_same_frame_frac_mean",
                "swa_raw_transport_top1_abs_frame_delta_mean",
                "prev_ttt_tracked_instance_query_soft_source_tokens",
                "prev_ttt_tracked_instance_query_soft_selected_query_count",
                "prev_ttt_tracked_instance_query_soft_query_selected_frac",
                "prev_ttt_tracked_instance_query_soft_topk",
                "prev_ttt_tracked_instance_query_soft_head_frac_threshold",
                "prev_ttt_tracked_instance_query_soft_query_block_size",
                "prev_ttt_tracked_instance_query_soft_direct_witness_hit_count",
                "prev_ttt_tracked_instance_query_soft_direct_witness_seed_count",
                "prev_ttt_tracked_instance_query_soft_min_direct_witness_seeds",
                "v102_swa_state_machine_history_tokens",
                "v102_swa_state_machine_current_tokens",
                "v102_swa_state_machine_swa_layer",
                "v102_swa_state_machine_unreliable_d_min",
                "v102_swa_state_machine_unreliable_g_min",
                "v102_swa_state_machine_supported_d_max",
                "v102_swa_state_machine_supported_k_min",
                "v102_swa_state_machine_soft_unsupported_min_keep",
                "v102_swa_state_machine_hold_prev_frames",
                "v102_swa_state_machine_hold_history_frames",
                "v102_swa_state_machine_hold_reference_tokens",
                "v102_swa_state_machine_hold_d_low_tokens",
                "v102_swa_state_machine_hold_semantic_static_tokens",
                "v102_swa_state_machine_hold_k_stable_tokens",
                "v102_swa_state_machine_hold_soft_min_keep",
                "v102_swa_state_machine_delay_current_tokens",
                "v102_swa_state_machine_delay_current_frac",
                "v102_swa_state_machine_delay_current_soft_min_keep",
                "v102_swa_state_machine_context_semantic_tokens",
                "v102_swa_state_machine_context_scale_observable_tokens",
                "v102_swa_state_machine_context_d_low_tokens",
                "v102_swa_state_machine_context_demoted_tokens",
                "v102_swa_state_machine_context_soft_min_keep",
                "v102_swa_state_machine_min_history_keep_frac",
                "v102_swa_state_machine_unreliable_d_high_tokens",
                "v102_swa_state_machine_unreliable_g_high_tokens",
                "v102_swa_state_machine_supported_d_low_tokens",
                "v102_swa_state_machine_supported_semantic_static_tokens",
                "v102_swa_state_machine_supported_k_stable_tokens",
                "v102_swa_state_machine_supported_history_tokens",
                "v102_swa_state_machine_rejected_history_tokens",
                "v102_swa_state_machine_kept_history_tokens",
                "v102_swa_state_machine_source_keep_tokens",
                "v102_swa_state_machine_source_total_tokens",
                "v102_swa_state_machine_rejected_history_frac",
                "attention_mass_removed_before",
                "attention_mass_removed_after",
                "attention_mass_retained_before",
                "attention_mass_retained_after",
                "attention_mass_removed_tokens_mean",
                "attention_mass_retained_tokens_mean",
                "attention_mass_query_sample_tokens_mean",
            ):
                vals = [
                    float(rec.get(extra_key))
                    for rec in records
                    if isinstance(rec, dict)
                    and rec.get(extra_key) is not None
                    and torch.isfinite(torch.tensor(float(rec.get(extra_key))))
                ]
                if vals:
                    extra_numeric_summary[f"mean_{extra_key}"] = float(torch.tensor(vals).mean().item())
                    extra_numeric_summary[f"max_{extra_key}"] = max(vals)
            extra_array_summary: Dict[str, Any] = {}
            for extra_key in (
                "swa_overlap_attention_mass_source_before_by_head",
                "swa_overlap_attention_mass_source_after_by_head",
                "swa_overlap_attention_mass_source_lift_by_head",
                "swa_overlap_attention_mass_selected_before_by_head",
                "swa_overlap_attention_mass_selected_after_by_head",
                "swa_overlap_attention_mass_selected_lift_by_head",
            ):
                arrays: List[List[float]] = []
                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    raw = rec.get(extra_key)
                    if not isinstance(raw, (list, tuple)):
                        continue
                    vals: List[float] = []
                    valid = True
                    for item in raw:
                        try:
                            value = float(item)
                        except (TypeError, ValueError):
                            valid = False
                            break
                        if not math.isfinite(value):
                            valid = False
                            break
                        vals.append(value)
                    if valid and vals:
                        arrays.append(vals)
                if arrays:
                    head_count = min(len(arr) for arr in arrays)
                    if head_count > 0:
                        tensor = torch.tensor([arr[:head_count] for arr in arrays], dtype=torch.float32)
                        mean_by_head = tensor.mean(dim=0)
                        max_by_head = tensor.max(dim=0).values
                        extra_array_summary[extra_key] = [float(v) for v in mean_by_head.tolist()]
                        extra_array_summary[f"max_{extra_key}"] = [float(v) for v in max_by_head.tolist()]
            selected_lift_by_head = extra_array_summary.get("swa_overlap_attention_mass_selected_lift_by_head")
            if isinstance(selected_lift_by_head, list) and selected_lift_by_head:
                selected_top_head = int(max(range(len(selected_lift_by_head)), key=lambda idx: selected_lift_by_head[idx]))
                extra_array_summary["swa_overlap_attention_mass_selected_top_head_by_lift"] = selected_top_head
                extra_array_summary["swa_overlap_attention_mass_selected_top_head_lift"] = float(
                    selected_lift_by_head[selected_top_head]
                )
            source_lift_by_head = extra_array_summary.get("swa_overlap_attention_mass_source_lift_by_head")
            if isinstance(source_lift_by_head, list) and source_lift_by_head:
                source_top_head = int(max(range(len(source_lift_by_head)), key=lambda idx: source_lift_by_head[idx]))
                extra_array_summary["swa_overlap_attention_mass_source_top_head_by_lift"] = source_top_head
                extra_array_summary["swa_overlap_attention_mass_source_top_head_lift"] = float(
                    source_lift_by_head[source_top_head]
                )
            extra_bool_summary: Dict[str, float] = {}
            for extra_key in (
                "v40_r2_source_mass_proxy_gate_pass",
                "v40_r2_no_source_mass_control",
                "v40_r3_residual_available",
                "v40_r3_source_attention_mass_available_before_action",
                "v67_source_attention_random_same_mass",
                "v67_source_attention_group_missing_semantic",
                "semantic_anchor_rescue_source_available",
                "swa_overlap_source_semantic_missing_labels",
                "swa_overlap_source_semantic_random_same_mass",
                "swa_raw_transport_trace_available",
                "swa_raw_transport_stable_fallback_used",
                "swa_raw_transport_stable_groups_nonempty",
                "swa_raw_transport_unreliable_groups_nonempty",
                "v102_swa_state_machine_trace_available",
                "v102_swa_state_machine_trace_applied",
                "v102_swa_state_machine_scaffold_only",
                "v102_swa_state_machine_runtime_action_allowed",
                "v102_swa_state_machine_strict_gate_pass",
                "v102_swa_state_machine_true_l3_gate_pass",
                "v102_swa_state_machine_action_probe_enabled",
                "v102_swa_state_machine_supported_require_static_semantic",
                "v102_swa_state_machine_supported_fallback_used",
                "prev_ttt_tracked_instance_query_soft_trace_only",
                "prev_ttt_tracked_instance_query_soft_carrier_rule_pass",
                "prev_ttt_tracked_instance_query_soft_direct_match_available",
            ):
                vals = [
                    bool(rec.get(extra_key))
                    for rec in records
                    if isinstance(rec, dict) and rec.get(extra_key) is not None
                ]
                if vals:
                    extra_bool_summary[f"frac_{extra_key}"] = float(torch.tensor(vals, dtype=torch.float32).mean().item())
            extra_categorical_summary: Dict[str, Any] = {}
            for extra_key in (
                "v102_swa_state_machine_action",
                "v102_swa_state_machine_reason",
                "v102_swa_state_machine_required_terms",
                "v102_swa_state_machine_probe_impl",
                "prev_ttt_tracked_instance_query_soft_direct_match_mode",
                "prev_ttt_tracked_instance_query_soft_direct_match_missing_reason",
            ):
                vals = sorted({
                    str(rec.get(extra_key))
                    for rec in records
                    if isinstance(rec, dict) and rec.get(extra_key) not in (None, "")
                })
                if vals:
                    extra_categorical_summary[f"values_{extra_key}"] = vals
            hook_effect_summary[path_name] = {
                "num_calls": len(records),
                "num_enabled_layers": sum(
                    1 for rec in records
                    if isinstance(rec, dict)
                    and (
                        bool(rec.get("layer_enabled", True))
                        or bool(rec.get("swa_overlap_bias_applied", False))
                        or bool(rec.get("swa_overlap_source_gate_applied", False))
                        or bool(rec.get("swa_prev_ttt_stable_anchor_gate_applied", False))
                        or bool(rec.get("prev_ttt_anchor_query_soft_applied", False))
                        or rec.get("prev_ttt_tracked_instance_query_soft_reason") is not None
                        or bool(rec.get("swa_overlap_source_replace_applied", False))
                    )
                ),
                "num_swa_overlap_bias_applied": sum(
                    1 for rec in records if isinstance(rec, dict) and bool(rec.get("swa_overlap_bias_applied", False))
                ),
                "num_swa_overlap_source_gate_applied": sum(
                    1 for rec in records if isinstance(rec, dict) and bool(rec.get("swa_overlap_source_gate_applied", False))
                ),
                "num_swa_overlap_source_replace_applied": sum(
                    1 for rec in records
                    if isinstance(rec, dict) and bool(rec.get("swa_overlap_source_replace_applied", False))
                ),
                "num_swa_prev_ttt_stable_anchor_gate_applied": sum(
                    1 for rec in records
                    if isinstance(rec, dict) and bool(rec.get("swa_prev_ttt_stable_anchor_gate_applied", False))
                ),
                "num_swa_prev_ttt_anchor_query_soft_applied": sum(
                    1 for rec in records
                    if isinstance(rec, dict) and bool(rec.get("prev_ttt_anchor_query_soft_applied", False))
                ),
                "num_swa_prev_ttt_tracked_instance_query_soft_trace": sum(
                    1 for rec in records
                    if isinstance(rec, dict)
                    and rec.get("prev_ttt_tracked_instance_query_soft_reason") is not None
                ),
                "num_swa_prev_ttt_tracked_instance_query_soft_carrier_rule_pass": sum(
                    1 for rec in records
                    if isinstance(rec, dict)
                    and bool(rec.get("prev_ttt_tracked_instance_query_soft_carrier_rule_pass", False))
                ),
                "num_source_gate_applied": sum(
                    1 for rec in records if isinstance(rec, dict) and bool(rec.get("source_gate_applied", False))
                ),
                "num_context_source_skip_applied": sum(
                    1 for rec in records if isinstance(rec, dict) and bool(rec.get("context_source_skip_applied", False))
                ),
                "num_v102_swa_state_machine_trace_available": sum(
                    1 for rec in records
                    if isinstance(rec, dict) and bool(rec.get("v102_swa_state_machine_trace_available", False))
                ),
                "num_v102_swa_state_machine_trace_applied": sum(
                    1 for rec in records
                    if isinstance(rec, dict) and bool(rec.get("v102_swa_state_machine_trace_applied", False))
                ),
                "num_v102_swa_state_machine_runtime_action_allowed": sum(
                    1 for rec in records
                    if isinstance(rec, dict) and bool(rec.get("v102_swa_state_machine_runtime_action_allowed", False))
                ),
                "num_v102_swa_state_machine_scaffold_only": sum(
                    1 for rec in records
                    if isinstance(rec, dict) and bool(rec.get("v102_swa_state_machine_scaffold_only", False))
                ),
                "mean_abs_bias": float(torch.tensor(mean_vals).mean().item()) if mean_vals else 0.0,
                "max_abs_bias": max(max_vals) if max_vals else 0.0,
                "mean_swa_gate": float(torch.tensor(gate_mean_vals).mean().item()) if gate_mean_vals else 1.0,
                "mean_abs_gate_delta": float(torch.tensor(gate_delta_vals).mean().item()) if gate_delta_vals else 0.0,
                "max_abs_gate_delta": max(max_gate_delta_vals) if max_gate_delta_vals else 0.0,
                "mean_swa_overlap_source_gate": float(torch.tensor(overlap_source_gate_mean_vals).mean().item())
                if overlap_source_gate_mean_vals else 1.0,
                "mean_swa_overlap_source_gate_delta": float(torch.tensor(overlap_source_gate_delta_vals).mean().item())
                if overlap_source_gate_delta_vals else 0.0,
                "max_swa_overlap_source_gate_delta": max(overlap_source_max_gate_delta_vals)
                if overlap_source_max_gate_delta_vals else 0.0,
                "mean_swa_overlap_source_score": float(torch.tensor(overlap_source_score_vals).mean().item())
                if overlap_source_score_vals else 0.0,
                "mean_swa_overlap_source_score_q90": float(torch.tensor(overlap_source_score_q90_vals).mean().item())
                if overlap_source_score_q90_vals else 0.0,
                "mean_swa_overlap_source_replace_alpha": float(torch.tensor(overlap_source_replace_alpha_vals).mean().item())
                if overlap_source_replace_alpha_vals else 0.0,
                "mean_swa_overlap_source_replace_alpha_p90": float(torch.tensor(overlap_source_replace_alpha_p90_vals).mean().item())
                if overlap_source_replace_alpha_p90_vals else 0.0,
                "mean_swa_overlap_source_replace_score": float(torch.tensor(overlap_source_replace_score_vals).mean().item())
                if overlap_source_replace_score_vals else 0.0,
                "mean_swa_prev_ttt_stable_anchor_gate": float(torch.tensor(prev_ttt_anchor_gate_mean_vals).mean().item())
                if prev_ttt_anchor_gate_mean_vals else 1.0,
                "mean_swa_prev_ttt_stable_anchor_gate_delta": float(torch.tensor(prev_ttt_anchor_gate_delta_vals).mean().item())
                if prev_ttt_anchor_gate_delta_vals else 0.0,
                "max_swa_prev_ttt_stable_anchor_gate_delta": max(prev_ttt_anchor_max_gate_delta_vals)
                if prev_ttt_anchor_max_gate_delta_vals else 0.0,
                "max_swa_prev_ttt_stable_anchor_gate_tokens": max(prev_ttt_anchor_gate_token_vals)
                if prev_ttt_anchor_gate_token_vals else 0,
                "mean_swa_prev_ttt_stable_anchor_gate_selected_frac": float(
                    torch.tensor(prev_ttt_anchor_gate_frac_vals).mean().item()
                ) if prev_ttt_anchor_gate_frac_vals else 0.0,
                "max_swa_prev_ttt_anchor_query_soft_source_tokens": max(prev_ttt_anchor_query_soft_source_token_vals)
                if prev_ttt_anchor_query_soft_source_token_vals else 0,
                "max_swa_prev_ttt_anchor_query_soft_selected_queries": max(prev_ttt_anchor_query_soft_selected_query_vals)
                if prev_ttt_anchor_query_soft_selected_query_vals else 0,
                "mean_swa_prev_ttt_anchor_query_soft_selected_frac": float(
                    torch.tensor(prev_ttt_anchor_query_soft_selected_frac_vals).mean().item()
                ) if prev_ttt_anchor_query_soft_selected_frac_vals else 0.0,
                "max_swa_prev_ttt_tracked_instance_query_soft_source_tokens": (
                    max(prev_ttt_tracked_instance_query_soft_source_token_vals)
                    if prev_ttt_tracked_instance_query_soft_source_token_vals else 0
                ),
                "max_swa_prev_ttt_tracked_instance_query_soft_selected_queries": (
                    max(prev_ttt_tracked_instance_query_soft_selected_query_vals)
                    if prev_ttt_tracked_instance_query_soft_selected_query_vals else 0
                ),
                "mean_swa_prev_ttt_tracked_instance_query_soft_selected_frac": float(
                    torch.tensor(prev_ttt_tracked_instance_query_soft_selected_frac_vals).mean().item()
                ) if prev_ttt_tracked_instance_query_soft_selected_frac_vals else 0.0,
                "max_swa_prev_ttt_tracked_instance_query_soft_direct_witness_seed_count": (
                    max(prev_ttt_tracked_instance_query_soft_seed_count_vals)
                    if prev_ttt_tracked_instance_query_soft_seed_count_vals else 0
                ),
                "frac_swa_prev_ttt_tracked_instance_query_soft_carrier_rule_pass": float(
                    torch.tensor(prev_ttt_tracked_instance_query_soft_rule_pass_vals, dtype=torch.float32).mean().item()
                ) if prev_ttt_tracked_instance_query_soft_rule_pass_vals else 0.0,
                "mean_context_source_keep_ratio": float(torch.tensor(source_keep_ratio_vals).mean().item())
                if source_keep_ratio_vals else 1.0,
                "max_context_source_skip_tokens": max(source_skip_token_vals) if source_skip_token_vals else 0,
                "max_context_source_control_tokens": max(source_control_token_vals) if source_control_token_vals else 0,
                "max_context_source_boost_tokens": max(source_boost_token_vals) if source_boost_token_vals else 0,
                "num_semantic_anchor_boost_applied": int(sum(1 for v in source_boost_applied_vals if v)),
                "num_context_empty_source_events": int(sum(empty_source_event_vals)) if empty_source_event_vals else 0,
                "attention_mass_available": bool(attention_mass_removed_before_vals),
                "attention_mass_metrics_seen": attention_mass_metrics,
                "mean_attention_mass_removed_before": float(torch.tensor(attention_mass_removed_before_vals).mean().item())
                if attention_mass_removed_before_vals else None,
                "mean_attention_mass_removed_after": float(torch.tensor(attention_mass_removed_after_vals).mean().item())
                if attention_mass_removed_after_vals else None,
                "mean_attention_mass_actual_after": float(torch.tensor(attention_mass_actual_after_vals).mean().item())
                if attention_mass_actual_after_vals else None,
                "mean_attention_mass_retained_before": float(torch.tensor(attention_mass_retained_before_vals).mean().item())
                if attention_mass_retained_before_vals else None,
                "mean_attention_mass_retained_after": float(torch.tensor(attention_mass_retained_after_vals).mean().item())
                if attention_mass_retained_after_vals else None,
                "mean_attention_mass_query_sample_tokens": float(torch.tensor(attention_mass_query_sample_vals).mean().item())
                if attention_mass_query_sample_vals else None,
                "mean_attention_mass_removed_tokens": float(torch.tensor(attention_mass_removed_token_vals).mean().item())
                if attention_mass_removed_token_vals else None,
                "mean_attention_mass_retained_tokens": float(torch.tensor(attention_mass_retained_token_vals).mean().item())
                if attention_mass_retained_token_vals else None,
                "mean_attention_mass_stable_anchor_before": float(torch.tensor(attention_mass_stable_anchor_before_vals).mean().item())
                if attention_mass_stable_anchor_before_vals else None,
                "mean_attention_mass_stable_anchor_after": float(torch.tensor(attention_mass_stable_anchor_after_vals).mean().item())
                if attention_mass_stable_anchor_after_vals else None,
                "mean_attention_mass_stable_anchor_actual_after": float(torch.tensor(attention_mass_stable_anchor_actual_after_vals).mean().item())
                if attention_mass_stable_anchor_actual_after_vals else None,
                "mean_attention_mass_stable_anchor_tokens": float(torch.tensor(attention_mass_stable_anchor_token_vals).mean().item())
                if attention_mass_stable_anchor_token_vals else None,
                "mean_attention_mass_stable_anchor_preservation_ratio": float(torch.tensor(attention_mass_stable_anchor_preservation_vals).mean().item())
                if attention_mass_stable_anchor_preservation_vals else None,
                "max_history_tokens": max(history_token_vals) if history_token_vals else 0,
                "max_d_prev_tokens": max(d_prev_token_vals) if d_prev_token_vals else 0,
            }
            hook_effect_summary[path_name].update(extra_numeric_summary)
            hook_effect_summary[path_name].update(extra_array_summary)
            hook_effect_summary[path_name].update(extra_bool_summary)
            hook_effect_summary[path_name].update(extra_categorical_summary)
        frame_bias_spec = (control_prior.frame_bias_spec or {}) if control_prior is not None else {}
        frame_read_effective_enabled = bool(self.enable_frame_read_control)
        frame_read_effective_enabled = frame_read_effective_enabled and bool(
            frame_bias_spec.get("enabled", frame_read_effective_enabled)
        )
        implemented_paths: List[str] = []
        if mode == "identity_hooks":
            implemented_paths = ["frame_attention", "swa_read", "ttt_apply", "chunk_attention"]
        elif mode == "ttt_write_only":
            implemented_paths = ["ttt_update"]
            if self.enable_swa_read_control:
                implemented_paths.append("swa_read")
            elif self.enable_swa_overlap_bias:
                implemented_paths.append("swa_overlap_bias")
            elif self.enable_swa_overlap_source_gate:
                implemented_paths.append("swa_overlap_source_gate")
            elif self.enable_swa_prev_ttt_stable_anchor_gate:
                implemented_paths.append("swa_prev_ttt_stable_anchor_gate")
            elif self.enable_swa_prev_ttt_anchor_query_soft:
                implemented_paths.append("swa_prev_ttt_anchor_query_soft")
            elif self.enable_swa_prev_ttt_tracked_instance_query_soft_action:
                implemented_paths.append(
                    "swa_prev_ttt_tracked_instance_query_soft_action"
                    if self.swa_prev_ttt_tracked_instance_query_soft_action_runtime_authorized
                    else "swa_prev_ttt_tracked_instance_query_soft_action_requested_trace_only"
                )
            elif self.enable_swa_prev_ttt_tracked_instance_query_soft_trace:
                implemented_paths.append("swa_prev_ttt_tracked_instance_query_soft_trace")
            elif self.enable_swa_overlap_source_replace:
                implemented_paths.append("swa_overlap_source_replace")
            if self.swa_raw_transport_trace_dir:
                implemented_paths.append("swa_raw_transport_trace")
        elif mode == "hybrid":
            implemented_paths = ["ttt_update"]
            if frame_read_effective_enabled:
                implemented_paths.append("frame_attention")
            if self.enable_swa_read_control:
                implemented_paths.append("swa_read")
            elif self.enable_swa_overlap_bias:
                implemented_paths.append("swa_overlap_bias")
            elif self.enable_swa_overlap_source_gate:
                implemented_paths.append("swa_overlap_source_gate")
            elif self.enable_swa_prev_ttt_stable_anchor_gate:
                implemented_paths.append("swa_prev_ttt_stable_anchor_gate")
            elif self.enable_swa_prev_ttt_anchor_query_soft:
                implemented_paths.append("swa_prev_ttt_anchor_query_soft")
            elif self.enable_swa_prev_ttt_tracked_instance_query_soft_action:
                implemented_paths.append(
                    "swa_prev_ttt_tracked_instance_query_soft_action"
                    if self.swa_prev_ttt_tracked_instance_query_soft_action_runtime_authorized
                    else "swa_prev_ttt_tracked_instance_query_soft_action_requested_trace_only"
                )
            elif self.enable_swa_prev_ttt_tracked_instance_query_soft_trace:
                implemented_paths.append("swa_prev_ttt_tracked_instance_query_soft_trace")
            elif self.enable_swa_overlap_source_replace:
                implemented_paths.append("swa_overlap_source_replace")
            if self.swa_raw_transport_trace_dir:
                implemented_paths.append("swa_raw_transport_trace")
            if self.enable_context_source_skip:
                implemented_paths.append("context_source_skip")
            if self.enable_ttt_apply_control:
                implemented_paths.append("ttt_apply")
            if self.enable_chunk_read_control:
                implemented_paths.append("chunk_attention")
        elif mode == "read_path_only":
            if frame_read_effective_enabled:
                implemented_paths.append("frame_attention")
            if self.enable_swa_read_control:
                implemented_paths.append("swa_read")
            elif self.enable_swa_overlap_bias:
                implemented_paths.append("swa_overlap_bias")
            elif self.enable_swa_overlap_source_gate:
                implemented_paths.append("swa_overlap_source_gate")
            elif self.enable_swa_prev_ttt_stable_anchor_gate:
                implemented_paths.append("swa_prev_ttt_stable_anchor_gate")
            elif self.enable_swa_prev_ttt_anchor_query_soft:
                implemented_paths.append("swa_prev_ttt_anchor_query_soft")
            elif self.enable_swa_prev_ttt_tracked_instance_query_soft_action:
                implemented_paths.append(
                    "swa_prev_ttt_tracked_instance_query_soft_action"
                    if self.swa_prev_ttt_tracked_instance_query_soft_action_runtime_authorized
                    else "swa_prev_ttt_tracked_instance_query_soft_action_requested_trace_only"
                )
            elif self.enable_swa_prev_ttt_tracked_instance_query_soft_trace:
                implemented_paths.append("swa_prev_ttt_tracked_instance_query_soft_trace")
            elif self.enable_swa_overlap_source_replace:
                implemented_paths.append("swa_overlap_source_replace")
            if self.swa_raw_transport_trace_dir:
                implemented_paths.append("swa_raw_transport_trace")
            if self.enable_context_source_skip:
                implemented_paths.append("context_source_skip")
            if self.enable_ttt_apply_control:
                implemented_paths.append("ttt_apply")
            if self.enable_chunk_read_control:
                implemented_paths.append("chunk_attention")
        return HybridMemoryResult(
            geometry_output=geo,
            state_next=state_next,
            ttt_state_next=ttt_next,
            swa_state_next=None,
            write_result=write_result,
            control_trace={
                "implemented_paths": implemented_paths,
                "identity_hook_paths": ["frame_attention", "swa_read", "ttt_apply", "chunk_attention"]
                if mode == "identity_hooks" else [],
                "hook_trace_counts": {
                    "frame_attention": len(trace_dict.get("frame_attention", [])),
                    "swa_read": len(trace_dict.get("swa_read", [])),
                    "ttt_apply": len(trace_dict.get("ttt_apply", [])),
                    "chunk_attention": len(trace_dict.get("chunk_attention", [])),
                },
                "hook_effect_summary": hook_effect_summary,
                "mode": mode,
            },
            debug={
                "mode": mode,
                "controlled_input_state_hash": input_hash,
                "controlled_output_state_hash": output_hash,
                "ttt_cache_layers": int(write_cache.num_ttt_layers),
                "write_debug": write_debug,
                "read_path_control": "real_identity_hook_sites" if mode == "identity_hooks" else "available_hook_sites",
            },
        )

    def build_probe_ttt_write_state(
        self,
        probe: ProbeOutput,
        control_prior: Optional[HybridMemoryControlPrior],
        *,
        token_type: Optional[torch.Tensor] = None,
        prev_ttt_state: Optional[Dict[str, Any]] = None,
        prev_control_summary: Optional[Dict[str, Any]] = None,
    ) -> Tuple[HybridMemoryState, Optional[WriteResult]]:
        """Commit TTT writes from the native probe cache, not controlled tokens.

        Phase D uses this to combine read-path controlled *outputs* with a
        separately decided TTT write memory update.  The controlled pass may
        improve the current chunk, but its altered tokens should not implicitly
        become the source of the next chunk's fast weights.
        """

        write_cache = probe.hybrid_cache.ttt_cache
        if write_cache is None:
            return probe.native_provisional_state, None
        write_cache, dual_lifetime_debug = _write_cache_with_long_old_weights_for_dual_lifetime(
            write_cache,
            prev_ttt_state,
        )
        A_tok = control_prior.P_ttt_write if control_prior is not None else None
        B_chunk_geo = control_prior.B_chunk_geo if control_prior is not None else None
        old_mode = self.ttt_update_controller.write_mode
        self.ttt_update_controller.write_mode = "semantic"
        self.ttt_update_controller.current_chunk_idx = int(getattr(self, "current_chunk_idx", -1))
        scale_state_debug = self._configure_ttt_write_scale_state(
            probe.geometry,
            prev_control_summary,
        )
        write_result = self.ttt_update_controller.run(
            write_cache,
            A_tok,
            B_chunk_geo=B_chunk_geo,
            device=self.device,
            token_type=token_type,
            num_frames=int(probe.geometry.num_frames),
            overlap_frames=int(self.read_overlap_frames),
            risk_tok=control_prior.D_tok if control_prior is not None else None,
            role_tok=control_prior.R_ttt_tok if control_prior is not None else None,
            prev_transient_delta=(
                prev_ttt_state.get("transient_delta")
                if isinstance(prev_ttt_state, dict)
                else None
            ),
        )
        self.ttt_update_controller.write_mode = old_mode
        write_result.debug.update(dual_lifetime_debug)
        write_result.debug["ttt_scale_state_diagnostic"] = scale_state_debug
        gated_history, swa_write_debug = self._apply_swa_history_write_gate(
            write_result.history,
            control_prior,
            probe.geometry,
        )
        write_result.history = gated_history
        write_result.debug.update(swa_write_debug)
        ttt_next = {"w0": write_result.w0, "w1": write_result.w1, "w2": write_result.w2}
        if write_result.history is not None:
            ttt_next["history"] = write_result.history
        if write_result.transient_delta is not None:
            ttt_next["transient_delta"] = write_result.transient_delta
        state_next = HybridMemoryState(
            ttt_state=ttt_next,
            swa_state=None,
            ref_state=None,
            prev_control_summary=self._build_prev_control_summary(
                control_prior,
                probe.geometry,
                prev_control_summary=prev_control_summary,
                ttt_token_contribution_diagnostic=write_result.token_contribution_diagnostic,
                chunk_idx=int(getattr(self, "current_chunk_idx", -1)),
            ),
            debug={
                "commit_source": "probe_ttt_write",
                "write_debug": write_result.debug,
                "probe_native_state_hash": hybrid_state_fingerprint(probe.native_provisional_state),
            },
        )
        return state_next, write_result

    def _semantic_action_chunk_active(self) -> bool:
        if not self.semantic_action_active_chunks:
            return True
        try:
            chunk_idx = int(getattr(self, "current_chunk_idx", -1))
        except (TypeError, ValueError):
            return False
        return chunk_idx in self.semantic_action_active_chunks

    def _build_model_hmc_control(
        self,
        control_prior: Optional[HybridMemoryControlPrior],
        *,
        mode: str,
        identity_hooks: bool = False,
    ) -> Dict[str, Any]:
        is_identity = bool(identity_hooks or mode == "identity_hooks")
        semantic_action_active = (not is_identity) and self._semantic_action_chunk_active()
        semantic_gate_mode = "fixed_chunks" if self.semantic_action_active_chunks else "ungated"
        semantic_paths = self.semantic_memory_paths if semantic_action_active else ""
        semantic_policy = self.semantic_role_policy if semantic_action_active else "none"
        frame_bias_spec = (control_prior.frame_bias_spec or {}) if control_prior is not None else {}
        context_source_skip_enabled = bool(self.enable_context_source_skip and semantic_action_active)
        if (
            context_source_skip_enabled
            and _is_v95_trackh_gated_l07_source(self.read_cue_source)
            and "anchor_comp" in str(self.read_cue_source).lower()
        ):
            context_source_skip_enabled = bool(frame_bias_spec.get("enabled", False))
        frame_read_effective_enabled = bool(self.enable_frame_read_control)
        frame_read_effective_enabled = frame_read_effective_enabled and bool(
            frame_bias_spec.get("enabled", frame_read_effective_enabled)
        )
        return {
            "identity_hooks": is_identity,
            "collect_trace": True,
            "D_tok": control_prior.D_tok if control_prior is not None else None,
            "P_ref": control_prior.P_ref if control_prior is not None else None,
            "S_tok": control_prior.S_tok if control_prior is not None else None,
            "G_sem_tok": control_prior.G_sem_tok if control_prior is not None else None,
            "Q_sem_tok": control_prior.Q_sem_tok if control_prior is not None else None,
            "L_sem_tok": control_prior.L_sem_tok if control_prior is not None else None,
            "stage_c_seed_global_track_idx_tok": (
                control_prior.stage_c_seed_global_track_idx_tok if control_prior is not None else None
            ),
            "stage_c_masklet_instance_idx_tok": (
                control_prior.stage_c_masklet_instance_idx_tok if control_prior is not None else None
            ),
            "V_sem_tok": control_prior.V_sem_tok if control_prior is not None else None,
            "R_sem_tok": control_prior.R_sem_tok if control_prior is not None else None,
            "R_frame_tok": control_prior.R_frame_tok if control_prior is not None else None,
            "R_global_tok": control_prior.R_global_tok if control_prior is not None else None,
            "R_swa_tok": control_prior.R_swa_tok if control_prior is not None else None,
            "R_ttt_tok": control_prior.R_ttt_tok if control_prior is not None else None,
            "K_stable_tok": control_prior.K_stable_tok if control_prior is not None else None,
            "swa_redirection_source_mask_tok": (
                control_prior.swa_redirection_source_mask_tok if control_prior is not None else None
            ),
            "A_anchor_tok": control_prior.A_anchor_tok if control_prior is not None else None,
            "A_anchor_mask_tok": control_prior.A_anchor_mask_tok if control_prior is not None else None,
            "semantic_anchor_mode": self.semantic_anchor_mode,
            "semantic_action_chunk_gate_mode": semantic_gate_mode,
            "semantic_action_chunk_gate_active": bool(semantic_action_active),
            "semantic_action_active_chunks": sorted(int(v) for v in self.semantic_action_active_chunks),
            "semantic_action_chunk_idx": int(getattr(self, "current_chunk_idx", -1)),
            "semantic_role_policy": semantic_policy,
            "semantic_memory_paths": semantic_paths,
            "enable_frame_read_control": False if is_identity else frame_read_effective_enabled,
            "enable_swa_read_control": False if is_identity else self.enable_swa_read_control,
            "enable_ttt_apply_control": False if is_identity else self.enable_ttt_apply_control,
            "enable_chunk_read_control": False if is_identity else self.enable_chunk_read_control,
            "beta_frame": 0.0 if (is_identity or not frame_read_effective_enabled) else (
                float((control_prior.frame_bias_spec or {}).get("beta_effective", self.beta_frame))
                if control_prior is not None else self.beta_frame
            ),
            "beta_swa": 0.0 if is_identity else self.beta_swa,
            "swa_gate_min": 1.0 if is_identity else self.swa_gate_min,
            "swa_layer_mode": "first_swa_only",
            "enable_swa_overlap_bias": False if is_identity else self.enable_swa_overlap_bias,
            "swa_overlap_bias_beta": 0.0 if is_identity else self.swa_overlap_bias_beta,
            "swa_overlap_bias_min_keep": self.swa_overlap_bias_min_keep,
            "swa_overlap_bias_mode": self.swa_overlap_bias_mode,
            "swa_overlap_bias_layer_mode": self.swa_overlap_bias_layer_mode,
            "swa_overlap_bias_single_layer": self.swa_overlap_bias_single_layer,
            "swa_overlap_bias_head_indices": "" if is_identity else self.swa_overlap_bias_head_indices,
            "swa_overlap_bias_record_attention_mass": False if is_identity else self.swa_overlap_bias_record_attention_mass,
            "swa_overlap_bias_attention_mass_max_queries": self.swa_overlap_bias_attention_mass_max_queries,
            "swa_overlap_external_mask_csv": "" if is_identity else self.swa_overlap_external_mask_csv,
            "swa_overlap_external_mask_variant": self.swa_overlap_external_mask_variant,
            "swa_overlap_external_mask_seq": self.swa_overlap_external_mask_seq,
            "enable_swa_overlap_source_gate": False if is_identity else self.enable_swa_overlap_source_gate,
            "swa_overlap_source_gate_rho": 0.0 if is_identity else self.swa_overlap_source_gate_rho,
            "swa_overlap_source_gate_min": self.swa_overlap_source_gate_min,
            "swa_overlap_source_gate_mode": self.swa_overlap_source_gate_mode,
            "swa_overlap_source_gate_target": self.swa_overlap_source_gate_target,
            "swa_overlap_source_gate_layer_mode": self.swa_overlap_source_gate_layer_mode,
            "swa_overlap_source_gate_single_layer": self.swa_overlap_source_gate_single_layer,
            "enable_swa_prev_ttt_stable_anchor_gate": (
                False if is_identity else self.enable_swa_prev_ttt_stable_anchor_gate
            ),
            "swa_prev_ttt_stable_anchor_gate_rho": (
                0.0 if is_identity else self.swa_prev_ttt_stable_anchor_gate_rho
            ),
            "swa_prev_ttt_stable_anchor_gate_min": self.swa_prev_ttt_stable_anchor_gate_min,
            "swa_prev_ttt_stable_anchor_gate_target": self.swa_prev_ttt_stable_anchor_gate_target,
            "swa_prev_ttt_stable_anchor_gate_layer_mode": self.swa_prev_ttt_stable_anchor_gate_layer_mode,
            "swa_prev_ttt_stable_anchor_gate_single_layer": self.swa_prev_ttt_stable_anchor_gate_single_layer,
            "enable_swa_prev_ttt_anchor_query_soft": (
                False if is_identity else self.enable_swa_prev_ttt_anchor_query_soft
            ),
            "swa_prev_ttt_anchor_query_soft_rho": (
                0.0 if is_identity else self.swa_prev_ttt_anchor_query_soft_rho
            ),
            "swa_prev_ttt_anchor_query_soft_min_keep": self.swa_prev_ttt_anchor_query_soft_min_keep,
            "swa_prev_ttt_anchor_query_soft_query_head_frac_threshold": (
                self.swa_prev_ttt_anchor_query_soft_query_head_frac_threshold
            ),
            "swa_prev_ttt_anchor_query_soft_topk": self.swa_prev_ttt_anchor_query_soft_topk,
            "swa_prev_ttt_anchor_query_soft_query_block_size": (
                self.swa_prev_ttt_anchor_query_soft_query_block_size
            ),
            "swa_prev_ttt_anchor_query_soft_layer_mode": self.swa_prev_ttt_anchor_query_soft_layer_mode,
            "swa_prev_ttt_anchor_query_soft_single_layer": self.swa_prev_ttt_anchor_query_soft_single_layer,
            "swa_prev_ttt_anchor_query_soft_attention_mass_max_queries": (
                self.swa_prev_ttt_anchor_query_soft_attention_mass_max_queries
            ),
            "enable_swa_prev_ttt_tracked_instance_query_soft_trace": (
                False if is_identity else self.enable_swa_prev_ttt_tracked_instance_query_soft_trace
            ),
            "enable_swa_prev_ttt_tracked_instance_query_soft_action": (
                False if is_identity else self.enable_swa_prev_ttt_tracked_instance_query_soft_action
            ),
            "swa_prev_ttt_tracked_instance_query_soft_action_runtime_authorized": (
                False
                if is_identity
                else self.swa_prev_ttt_tracked_instance_query_soft_action_runtime_authorized
            ),
            "swa_prev_ttt_tracked_instance_query_soft_rho": (
                0.0 if is_identity else self.swa_prev_ttt_tracked_instance_query_soft_rho
            ),
            "swa_prev_ttt_tracked_instance_query_soft_min_keep": (
                self.swa_prev_ttt_tracked_instance_query_soft_min_keep
            ),
            "swa_prev_ttt_tracked_instance_query_soft_query_head_frac_threshold": (
                self.swa_prev_ttt_tracked_instance_query_soft_query_head_frac_threshold
            ),
            "swa_prev_ttt_tracked_instance_query_soft_topk": self.swa_prev_ttt_tracked_instance_query_soft_topk,
            "swa_prev_ttt_tracked_instance_query_soft_query_block_size": (
                self.swa_prev_ttt_tracked_instance_query_soft_query_block_size
            ),
            "swa_prev_ttt_tracked_instance_query_soft_layer_mode": (
                self.swa_prev_ttt_tracked_instance_query_soft_layer_mode
            ),
            "swa_prev_ttt_tracked_instance_query_soft_single_layer": (
                self.swa_prev_ttt_tracked_instance_query_soft_single_layer
            ),
            "swa_prev_ttt_tracked_instance_query_soft_attention_mass_max_queries": (
                self.swa_prev_ttt_tracked_instance_query_soft_attention_mass_max_queries
            ),
            "swa_prev_ttt_tracked_instance_query_soft_min_direct_witness_seeds": (
                self.swa_prev_ttt_tracked_instance_query_soft_min_direct_witness_seeds
            ),
            "swa_prev_ttt_tracked_instance_query_soft_direct_match_mode": (
                self.swa_prev_ttt_tracked_instance_query_soft_direct_match_mode
            ),
            "enable_swa_overlap_source_replace": False if is_identity else self.enable_swa_overlap_source_replace,
            "swa_overlap_source_replace_alpha": 0.0 if is_identity else self.swa_overlap_source_replace_alpha,
            "swa_overlap_source_replace_mode": self.swa_overlap_source_replace_mode,
            "swa_overlap_source_replace_target": self.swa_overlap_source_replace_target,
            "swa_overlap_source_replace_layer_mode": self.swa_overlap_source_replace_layer_mode,
            "swa_overlap_source_replace_single_layer": self.swa_overlap_source_replace_single_layer,
            "enable_v102_state_machine_trace": False if is_identity else self.enable_v102_state_machine_trace,
            "v102_state_machine_action": self.v102_state_machine_action,
            "v102_state_machine_layer_mode": self.v102_state_machine_layer_mode,
            "v102_state_machine_single_layer": self.v102_state_machine_single_layer,
            "v102_state_machine_strict_gate_pass": False if is_identity else self.v102_state_machine_strict_gate_pass,
            "v102_state_machine_true_l3_gate_pass": (
                False if is_identity else self.v102_state_machine_true_l3_gate_pass
            ),
            "enable_v102_state_machine_action_probe": (
                False if is_identity else self.enable_v102_state_machine_action_probe
            ),
            "v102_state_machine_probe_impl": self.v102_state_machine_probe_impl,
            "v102_state_machine_unreliable_d_min": self.v102_state_machine_unreliable_d_min,
            "v102_state_machine_unreliable_g_min": self.v102_state_machine_unreliable_g_min,
            "v102_state_machine_supported_d_max": self.v102_state_machine_supported_d_max,
            "v102_state_machine_supported_k_min": self.v102_state_machine_supported_k_min,
            "v102_state_machine_supported_require_static_semantic": (
                False if is_identity else self.v102_state_machine_supported_require_static_semantic
            ),
            "v102_state_machine_soft_unsupported_min_keep": self.v102_state_machine_soft_unsupported_min_keep,
            "v102_state_machine_hold_prev_frames": self.v102_state_machine_hold_prev_frames,
            "v102_state_machine_hold_soft_min_keep": self.v102_state_machine_hold_soft_min_keep,
            "v102_state_machine_delay_current_soft_min_keep": self.v102_state_machine_delay_current_soft_min_keep,
            "v102_state_machine_context_soft_min_keep": self.v102_state_machine_context_soft_min_keep,
            "v102_state_machine_min_history_keep_frac": self.v102_state_machine_min_history_keep_frac,
            "v102_state_machine_attention_mass_max_queries": self.v102_state_machine_attention_mass_max_queries,
            "swa_overlap_feature_dump_dir": "" if is_identity else self.swa_overlap_feature_dump_dir,
            "swa_overlap_feature_dump_dtype": self.swa_overlap_feature_dump_dtype,
            "swa_raw_transport_trace_dir": "" if is_identity else self.swa_raw_transport_trace_dir,
            "swa_raw_transport_trace_layer_mode": self.swa_raw_transport_trace_layer_mode,
            "swa_raw_transport_trace_single_layer": self.swa_raw_transport_trace_single_layer,
            "swa_raw_transport_trace_max_queries": self.swa_raw_transport_trace_max_queries,
            "swa_raw_transport_trace_topk": self.swa_raw_transport_trace_topk,
            "swa_raw_transport_trace_direct_match_only": (
                False if is_identity else self.swa_raw_transport_trace_direct_match_only
            ),
            "swa_raw_transport_trace_query_block_size": self.swa_raw_transport_trace_query_block_size,
            "enable_context_source_skip": context_source_skip_enabled,
            "context_source_skip_impl": "bias" if is_identity else self.context_source_skip_impl,
            "context_source_skip_scope": self.context_source_skip_scope,
            "context_source_skip_mode": self.context_source_skip_mode,
            "context_source_skip_mask": self.context_source_skip_mask,
            "context_source_skip_frame_region": self.context_source_skip_frame_region,
            "context_source_skip_query_region": self.context_source_skip_query_region,
            "context_source_skip_layer_mode": self.context_source_skip_layer_mode,
            "context_source_skip_single_layer": self.context_source_skip_single_layer,
            "context_source_skip_head_indices": "" if is_identity else self.context_source_skip_head_indices,
            "context_source_skip_soft_rho": 0.0 if is_identity else self.context_source_skip_soft_rho,
            "context_source_skip_soft_min_keep": self.context_source_skip_soft_min_keep,
            "context_source_skip_source_attention_top_quantile": (
                -1.0 if is_identity else self.context_source_skip_source_attention_top_quantile
            ),
            "context_source_skip_source_attention_top_random_same_mass": (
                False if is_identity else self.context_source_skip_source_attention_top_random_same_mass
            ),
            "context_source_skip_record_attention_mass": False if is_identity else self.context_source_skip_record_attention_mass,
            "context_source_skip_attention_mass_max_queries": self.context_source_skip_attention_mass_max_queries,
            "context_source_skip_attention_map_dump_dir": "" if is_identity else self.context_source_skip_attention_map_dump_dir,
            "context_source_skip_attention_map_dump_max_queries": self.context_source_skip_attention_map_dump_max_queries,
            "context_source_skip_attention_map_dump_dtype": self.context_source_skip_attention_map_dump_dtype,
            "context_source_skip_attention_map_dump_full_query_marginal": (
                False if is_identity else self.context_source_skip_attention_map_dump_full_query_marginal
            ),
            "context_source_skip_attention_map_dump_head_marginal": (
                False if is_identity else self.context_source_skip_attention_map_dump_head_marginal
            ),
            "context_source_skip_attention_map_dump_query_block": self.context_source_skip_attention_map_dump_query_block,
            "swa_write_cache_store_post": self.enable_swa_write_control and self.swa_write_cache_blend_alpha > 0.0,
            "swa_overlap_frames": int(self.read_overlap_frames),
            "beta_chunk": 0.0 if is_identity else self.beta_chunk,
            "rho_ttt_apply": 0.0 if is_identity else self.rho_ttt_apply,
            "read_layer_mode": "all" if is_identity else self.read_layer_mode,
            "read_single_layer": -1 if is_identity else self.read_single_layer,
            "read_cue_source": self.read_cue_source,
            "v70_radio_sidecar_dir": self.v70_radio_sidecar_dir,
            "read_topk_frac": self.read_topk_frac,
            "beta_policy": self.beta_policy,
            "beta_energy_target": self.beta_energy_target,
            "frame_bias_mode": self.frame_bias_mode,
            "frame_bias_query_region": self.frame_bias_query_region,
            "frame_bias_head_indices": self.frame_bias_head_indices,
            "frame_attention_record_bias_mass": False if is_identity else self.frame_attention_record_bias_mass,
            "frame_attention_bias_mass_max_queries": self.frame_attention_bias_mass_max_queries,
            "chunk_bias_mode": self.chunk_bias_mode,
            "swa_bias_mode": self.swa_bias_mode,
            "ttt_apply_min_gate": 0.0 if is_identity else self.ttt_apply_min_gate,
        }

    def _build_probe_trace(
        self,
        geo: GeometryOutput,
        write_cache: WriteCacheOutput,
    ) -> Dict[str, Any]:
        trace: Dict[str, Any] = {
            "internal_dynamic_patch": geo.attn_dynamic_patch,
            "ttt_residual_patch": None,
            "swa_prev_importance": None,
            "frame_attn_entropy_patch": None,
            "chunk_attn_dynamic_mass": None,
            "reference_importance": None,
            "debug": {
                "ttt_layers": int(write_cache.num_ttt_layers),
                "implemented": [
                    "internal_dynamic_patch_if_available",
                    "ttt_write_cache",
                    "real_frame_attn_hook_sites",
                    "real_swa_hook_sites",
                    "real_ttt_apply_hook_sites",
                    "real_chunk_attn_hook_sites",
                    "ttt_residual_summary",
                ],
                "missing": ["swa_dense_importance_map", "chunk_dense_dynamic_mass_map"],
            },
        }
        residual_means: List[float] = []
        residual_maxes: List[float] = []
        for lc in write_cache.layer_caches:
            if lc.apply_output_raw is None:
                continue
            pred = lc.apply_output_raw.float()
            target = lc.v.float()
            denom = torch.linalg.norm(target, dim=-1).clamp_min(1e-6)
            residual = torch.linalg.norm(pred - target, dim=-1) / denom
            residual_means.append(float(residual.mean().item()))
            residual_maxes.append(float(residual.max().item()))
        trace["ttt_residual_summary"] = {
            "layers": len(residual_means),
            "mean": float(sum(residual_means) / len(residual_means)) if residual_means else None,
            "max": float(max(residual_maxes)) if residual_maxes else None,
        }
        if geo.hmc_trace is not None:
            trace["hook_trace_counts"] = {
                "frame_attention": len(geo.hmc_trace.get("frame_attention", [])),
                "swa_read": len(geo.hmc_trace.get("swa_read", [])),
                "ttt_apply": len(geo.hmc_trace.get("ttt_apply", [])),
                "chunk_attention": len(geo.hmc_trace.get("chunk_attention", [])),
            }
        return trace

    @staticmethod
    def _state_from_write_cache(write_cache: WriteCacheOutput) -> Dict[str, Any]:
        state = {
            "w0": list(write_cache.w0_provisional),
            "w1": list(write_cache.w1_provisional),
            "w2": list(write_cache.w2_provisional),
        }
        if write_cache.history_provisional is not None:
            state["history"] = write_cache.history_provisional
        return state

    @staticmethod
    def _pose_step_scale_summary(geo: GeometryOutput) -> Dict[str, Any]:
        poses = getattr(geo, "camera_poses", None)
        if not isinstance(poses, torch.Tensor):
            return {"available": False, "reason": "missing_camera_poses"}
        try:
            p = poses.detach().cpu().float()
            if p.ndim != 3 or p.shape[-2:] != (4, 4) or p.shape[0] < 2:
                return {"available": False, "reason": "invalid_camera_pose_shape"}
            t = p[:, :3, 3]
            step = torch.linalg.norm(t[1:] - t[:-1], dim=-1)
            finite = step[torch.isfinite(step) & (step > 1e-12)]
            if int(finite.numel()) <= 0:
                return {"available": False, "reason": "empty_finite_pose_steps"}
            return {
                "available": True,
                "step_count": int(finite.numel()),
                "pose_step_median": float(torch.median(finite).item()),
                "pose_step_mean": float(finite.mean().item()),
                "pose_step_p90": float(torch.quantile(finite, 0.90).item()) if int(finite.numel()) > 1 else float(finite.mean().item()),
            }
        except RuntimeError as exc:
            return {"available": False, "reason": "pose_step_runtime_error", "error": str(exc)}

    @staticmethod
    def _positive_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out) or out <= 1e-12:
            return None
        return out

    def _configure_ttt_write_scale_state(
        self,
        geo: GeometryOutput,
        prev_control_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ctrl = self.ttt_update_controller
        mode_text = str(getattr(ctrl, "scale_state_mode", "none") or "none").strip().lower()
        proxy_text = str(getattr(ctrl, "scale_state_proxy", "pose_step_ema") or "pose_step_ema").strip().lower()
        alpha = max(float(getattr(ctrl, "scale_state_alpha", 0.0) or 0.0), 0.0)
        chunk_idx = int(getattr(self, "current_chunk_idx", -1))
        ctrl.scale_state_active = False
        ctrl.scale_state_chunk_idx = chunk_idx
        ctrl.scale_state_log_ratio = 0.0
        ctrl.scale_state_reason = "not_configured"
        ctrl.scale_state_payload = {}

        pose_summary = self._pose_step_scale_summary(geo)
        debug: Dict[str, Any] = {
            "schema": "acl2_v96_ttt_write_scale_state_pose_step_proxy_v1",
            "mode": mode_text,
            "proxy": proxy_text,
            "alpha": float(alpha),
            "chunk_idx": int(chunk_idx),
            "active": False,
            "reason": "not_configured",
            "pose_step_summary": pose_summary,
            "runtime_action_allowed": False,
        }

        def finish(reason: str, *, active: bool = False, log_ratio: float = 0.0) -> Dict[str, Any]:
            ctrl.scale_state_active = bool(active)
            ctrl.scale_state_log_ratio = float(log_ratio)
            ctrl.scale_state_reason = reason
            debug["active"] = bool(active)
            debug["reason"] = reason
            debug["log_ratio"] = float(log_ratio)
            debug["scale_weight_estimate"] = float(min(1.0, abs(float(log_ratio)) * alpha))
            ctrl.scale_state_payload = dict(debug)
            return debug

        if mode_text in {"", "none", "off"}:
            return finish("mode_off")
        if getattr(ctrl, "scale_state_chunks", None) and chunk_idx not in ctrl.scale_state_chunks:
            debug["configured_chunks"] = sorted(int(x) for x in ctrl.scale_state_chunks)
            return finish("chunk_not_selected")
        if alpha <= 0.0:
            return finish("alpha_zero")
        if proxy_text not in {"pose_step_ema", "pose_step", "pose_step_median", "trajectory_pose_step"}:
            return finish("unsupported_proxy")
        if not bool(pose_summary.get("available", False)):
            return finish(str(pose_summary.get("reason", "missing_pose_step")))

        current = self._positive_float(pose_summary.get("pose_step_median"))
        prev_summary = prev_control_summary if isinstance(prev_control_summary, dict) else {}
        reference = (
            self._positive_float(prev_summary.get("ttt_scale_state_pose_step_ema"))
            if proxy_text == "pose_step_ema"
            else None
        )
        if reference is None:
            reference = self._positive_float(prev_summary.get("ttt_pose_step_median"))
        debug["current_pose_step_median"] = current
        debug["reference_pose_step"] = reference
        debug["reference_source"] = (
            "prev_pose_step_ema"
            if self._positive_float(prev_summary.get("ttt_scale_state_pose_step_ema")) is not None and proxy_text == "pose_step_ema"
            else "prev_pose_step_median"
            if reference is not None
            else "missing"
        )
        if current is None:
            return finish("current_pose_step_unavailable")
        if reference is None:
            return finish("previous_pose_step_unavailable")
        log_ratio = math.log(max(current, 1e-12) / max(reference, 1e-12))
        if abs(log_ratio) <= 1e-8:
            return finish("proxy_zero", active=False, log_ratio=log_ratio)
        return finish("active", active=True, log_ratio=log_ratio)

    @staticmethod
    def _build_prev_control_summary(
        control_prior: Optional[HybridMemoryControlPrior],
        geo: GeometryOutput,
        *,
        prev_control_summary: Optional[Dict[str, Any]] = None,
        ttt_token_contribution_diagnostic: Optional[Dict[str, Any]] = None,
        chunk_idx: int = -1,
    ) -> Optional[Dict[str, Any]]:
        if control_prior is None or control_prior.D_tok is None:
            return None
        token_type = geo.token_type.detach().cpu().long()
        patch_mask = token_type == TOKEN_TYPE_PATCH
        source_tok = (
            control_prior.D_swa_write_tok
            if control_prior.D_swa_write_tok is not None
            else control_prior.D_tok
        )
        out: Dict[str, Any] = {
            "mean_D_tok": float(control_prior.D_tok.float().mean().item()),
            "mean_D_patch": float(control_prior.D_tok[patch_mask].float().mean().item()) if patch_mask.any() else 0.0,
            "mean_D_swa_source_tok": float(source_tok.float().mean().item()),
            "mean_D_swa_source_patch": float(source_tok[patch_mask].float().mean().item()) if patch_mask.any() else 0.0,
            "D_patch_source": "D_swa_write_tok" if control_prior.D_swa_write_tok is not None else "D_tok",
            "mean_R_tok": float(control_prior.R_tok.float().mean().item()) if control_prior.R_tok is not None else 1.0,
            "num_frames": int(geo.num_frames),
            "patch_grid": tuple(int(x) for x in geo.patch_grid),
            "D_patch": source_tok[patch_mask].detach().cpu().float() if patch_mask.any() else None,
        }
        pose_summary = HybridMemoryController._pose_step_scale_summary(geo)
        out["ttt_pose_step_summary"] = pose_summary
        if bool(pose_summary.get("available", False)):
            current = HybridMemoryController._positive_float(pose_summary.get("pose_step_median"))
            if current is not None:
                out["ttt_pose_step_median"] = float(current)
                out["ttt_pose_step_mean"] = float(pose_summary.get("pose_step_mean", current))
                prev_summary = prev_control_summary if isinstance(prev_control_summary, dict) else {}
                prev_ema = HybridMemoryController._positive_float(
                    prev_summary.get("ttt_scale_state_pose_step_ema")
                )
                ema_alpha = 0.5
                out["ttt_scale_state_pose_step_ema"] = (
                    float(current)
                    if prev_ema is None
                    else float((1.0 - ema_alpha) * prev_ema + ema_alpha * current)
                )
                out["ttt_scale_state_pose_step_ema_alpha"] = float(ema_alpha)
        for field_name, tensor, dtype in (
            ("G_patch", control_prior.G_sem_tok, torch.long),
            ("L_patch", control_prior.L_sem_tok, torch.long),
            (
                "stage_c_seed_global_track_idx_patch",
                control_prior.stage_c_seed_global_track_idx_tok,
                torch.long,
            ),
            (
                "stage_c_masklet_instance_idx_patch",
                control_prior.stage_c_masklet_instance_idx_tok,
                torch.long,
            ),
        ):
            if tensor is None:
                continue
            flat = tensor.detach().cpu().to(dtype=dtype).reshape(-1)
            if int(flat.numel()) == int(token_type.numel()):
                out[field_name] = flat[patch_mask]
            elif int(flat.numel()) == int(patch_mask.sum().item()):
                out[field_name] = flat
        diag = ttt_token_contribution_diagnostic if isinstance(ttt_token_contribution_diagnostic, dict) else {}
        stable_mask_tok = diag.get("stable_anchor_mask_tok")
        if torch.is_tensor(stable_mask_tok):
            stable_flat = stable_mask_tok.detach().cpu().bool().reshape(-1)
            patch_count = int(patch_mask.sum().item())
            stable_patch: Optional[torch.Tensor] = None
            if int(stable_flat.numel()) == int(token_type.numel()):
                stable_patch = stable_flat[patch_mask]
            elif int(stable_flat.numel()) == patch_count:
                stable_patch = stable_flat
            if stable_patch is not None:
                out["ttt_stable_anchor_mask_patch"] = stable_patch.cpu()
                out["ttt_stable_anchor_token_count"] = int(stable_patch.sum().item())
                out["ttt_stable_anchor_source"] = "probe_ttt_write_token_contribution_diagnostic"
                out["ttt_stable_anchor_source_chunk_idx"] = int(chunk_idx)
                patch_ids = torch.arange(patch_count, dtype=torch.int64)
                if int(chunk_idx) >= 0:
                    patch_ids = patch_ids + int(chunk_idx) * 1_000_000
                out["ttt_stable_anchor_id_patch"] = torch.where(
                    stable_patch,
                    patch_ids,
                    torch.full_like(patch_ids, -1),
                )
                for diag_key, out_key, dtype in (
                    ("stable_anchor_retention_tok", "ttt_stable_anchor_retention_patch", torch.float32),
                    ("stable_anchor_residual_tok", "ttt_stable_anchor_residual_patch", torch.float32),
                    ("z_write_key_norm_tok", "ttt_stable_anchor_z_write_key_norm_patch", torch.float32),
                ):
                    diag_tensor = diag.get(diag_key)
                    if not torch.is_tensor(diag_tensor):
                        continue
                    diag_flat = diag_tensor.detach().cpu().to(dtype=dtype).reshape(-1)
                    if int(diag_flat.numel()) == int(token_type.numel()):
                        out[out_key] = diag_flat[patch_mask]
                    elif int(diag_flat.numel()) == patch_count:
                        out[out_key] = diag_flat
                z_write_key_vec = diag.get("z_write_key_vec_tok")
                if torch.is_tensor(z_write_key_vec):
                    vec_flat = z_write_key_vec.detach().cpu().to(dtype=torch.float32).reshape(
                        int(z_write_key_vec.shape[0]),
                        -1,
                    )
                    if int(vec_flat.shape[0]) == int(token_type.numel()):
                        out["ttt_stable_anchor_z_write_key_vec_patch"] = vec_flat[patch_mask]
                    elif int(vec_flat.shape[0]) == patch_count:
                        out["ttt_stable_anchor_z_write_key_vec_patch"] = vec_flat
                z_write_key_sketch = diag.get("z_write_key_sketch_tok")
                if torch.is_tensor(z_write_key_sketch):
                    sketch_flat = z_write_key_sketch.detach().cpu().to(dtype=torch.float32).reshape(
                        int(z_write_key_sketch.shape[0]),
                        -1,
                    )
                    if int(sketch_flat.shape[0]) == int(token_type.numel()):
                        out["ttt_stable_anchor_z_write_key_sketch_patch"] = sketch_flat[patch_mask]
                    elif int(sketch_flat.shape[0]) == patch_count:
                        out["ttt_stable_anchor_z_write_key_sketch_patch"] = sketch_flat
                z_write_hidden_vec = diag.get("z_write_hidden_vec_tok")
                if torch.is_tensor(z_write_hidden_vec):
                    hidden_flat = z_write_hidden_vec.detach().cpu().to(dtype=torch.float32).reshape(
                        int(z_write_hidden_vec.shape[0]),
                        -1,
                    )
                    if int(hidden_flat.shape[0]) == int(token_type.numel()):
                        out["ttt_stable_anchor_z_write_hidden_vec_patch"] = hidden_flat[patch_mask]
                    elif int(hidden_flat.shape[0]) == patch_count:
                        out["ttt_stable_anchor_z_write_hidden_vec_patch"] = hidden_flat
        seed_patch = out.get("stage_c_seed_global_track_idx_patch")
        masklet_instance_patch = out.get("stage_c_masklet_instance_idx_patch")
        if torch.is_tensor(masklet_instance_patch):
            inst_flat = masklet_instance_patch.detach().cpu().long().reshape(-1)
            patch_count = int(patch_mask.sum().item())
            if int(inst_flat.numel()) == patch_count:
                tracked_mask = inst_flat >= 0
                seed_flat: Optional[torch.Tensor] = None
                if torch.is_tensor(seed_patch):
                    seed_flat = seed_patch.detach().cpu().long().reshape(-1)
                    if int(seed_flat.numel()) == patch_count:
                        tracked_mask = tracked_mask & (seed_flat >= 0)
                    else:
                        seed_flat = None
                out["ttt_tracked_instance_anchor_mask_patch"] = tracked_mask.cpu()
                out["ttt_tracked_instance_anchor_id_patch"] = torch.where(
                    tracked_mask,
                    inst_flat,
                    torch.full_like(inst_flat, -1),
                ).cpu()
                if seed_flat is not None:
                    out["ttt_tracked_instance_anchor_seed_patch"] = torch.where(
                        tracked_mask,
                        seed_flat,
                        torch.full_like(seed_flat, -1),
                    ).cpu()
                out["ttt_tracked_instance_anchor_token_count"] = int(tracked_mask.sum().item())
                out["ttt_tracked_instance_anchor_source"] = (
                    "diagnostic_stage_c_masklet_instance_patch_counterfactual_no_runtime"
                )
                out["ttt_tracked_instance_anchor_source_chunk_idx"] = int(chunk_idx)
        return out


__all__ = [
    "HybridMemoryState",
    "HybridMemoryCacheOutput",
    "ProbeOutput",
    "HybridMemoryControlPrior",
    "HybridMemoryResult",
    "HybridMemoryController",
    "hybrid_state_fingerprint",
]
