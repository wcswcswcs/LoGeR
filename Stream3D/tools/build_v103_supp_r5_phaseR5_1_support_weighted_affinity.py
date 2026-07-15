#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_v103_phase3_fast_carrier_reliability_filter import SCENE_INPUTS  # noqa: E402
from build_v103_phase4_primitive_affinity_feature import (  # noqa: E402
    _build_incidence,
    _compute_scene_arrays,
    _countsketch,
    _ensure_mmap_cache,
    _exact_dense_subset,
    _hash_mask,
    _load_cached,
    _mask_observations,
    _mask_weights,
    _pair_error,
)
from build_v103_phase5_mask_level_affinity_pooling import (  # noqa: E402
    _build_raw_sketch,
    _pair_values_strict_leave_two_out_bucket_zeroed,
    _pool_features,
    _sample_mask_pairs,
)


PHASE_ID = "v103_supp_r5_phaseR5_1_support_weighted_affinity"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r5_support_weighted_affinity"
DEFAULT_FACT_LOCK_ROOT = AUDIT_ROOT / "v103_supp_r5_fact_lock"
DEFAULT_PHASES1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers"
DEFAULT_D4RT_ROOT_BY_SCENE = {
    "scene0011_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "scene0050_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
}

VARIANTS = [
    {
        "variant_id": "F0_anchor_only",
        "support_lambda": 0.0,
        "semantic_filter": False,
        "veto_attenuation": False,
        "description": "z_i = Normalize(z_i^A)",
    },
    {
        "variant_id": "F1_anchor_plus_support_010",
        "support_lambda": 0.10,
        "semantic_filter": False,
        "veto_attenuation": False,
        "description": "z_i = Normalize(z_i^A + 0.10 z_i^S)",
    },
    {
        "variant_id": "F2_anchor_plus_support_020",
        "support_lambda": 0.20,
        "semantic_filter": False,
        "veto_attenuation": False,
        "description": "z_i = Normalize(z_i^A + 0.20 z_i^S)",
    },
    {
        "variant_id": "F3_anchor_plus_support_035",
        "support_lambda": 0.35,
        "semantic_filter": False,
        "veto_attenuation": False,
        "description": "z_i = Normalize(z_i^A + 0.35 z_i^S)",
    },
    {
        "variant_id": "F4_anchor_plus_semantic_filtered_support_020",
        "support_lambda": 0.20,
        "semantic_filter": True,
        "veto_attenuation": False,
        "description": "z_i = Normalize(z_i^A + 0.20 z_i^{S,sem})",
    },
    {
        "variant_id": "F5_anchor_plus_veto_attenuated_support_020",
        "support_lambda": 0.20,
        "semantic_filter": False,
        "veto_attenuation": True,
        "description": "z_i = Normalize(z_i^A + 0.20 z_i^{S,veto})",
    },
    {
        "variant_id": "F6_anchor_plus_semantic_filtered_veto_attenuated_support_020",
        "support_lambda": 0.20,
        "semantic_filter": True,
        "veto_attenuation": True,
        "description": "z_i = Normalize(z_i^A + 0.20 z_i^{S,sem+veto})",
    },
    {
        "variant_id": "F7_support_only_diagnostic",
        "anchor_scale": 0.0,
        "support_lambda": 1.0,
        "semantic_filter": False,
        "veto_attenuation": False,
        "diagnostic_only": True,
        "description": "diagnostic only: z_i = Normalize(z_i^S), used for R5-2 minus-A attribution",
    },
]


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _torch_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _safe_p(values: np.ndarray, pct: float, mask: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, float(pct)))


def _valid_rate(feature: np.ndarray, mask: np.ndarray | None = None) -> float:
    if feature.size == 0:
        return 0.0
    valid = np.linalg.norm(feature, axis=1) > 0.0
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if not np.any(m):
            return 0.0
        valid = valid[m]
    return float(np.mean(valid))


def _bucket_stats(mask_count: int, sketch_dim: int) -> dict[str, float]:
    bucket, _sign = _hash_mask(np.arange(int(mask_count), dtype=np.int64), int(sketch_dim))
    load_all = np.bincount(bucket, minlength=int(sketch_dim)).astype(np.float64)
    load = load_all[load_all > 0]
    collision = float(np.sum(np.maximum(load_all - 1.0, 0.0)) / max(int(mask_count), 1))
    return {
        "bucket_load_mean": float(np.mean(load)) if load.size else 0.0,
        "bucket_load_p95": float(np.percentile(load, 95)) if load.size else 0.0,
        "collision_mass_ratio": collision,
    }


def _role_flags_for_scene(role_df: pd.DataFrame, scene: str, carrier_id: np.ndarray) -> tuple[dict[str, np.ndarray], str]:
    sdf = role_df[role_df["scene_id"].astype(str) == scene].copy()
    if sdf.empty:
        raise RuntimeError(f"no S1 role rows found for scene={scene}")
    required = {"carrier_id", "is_A_anchor", "is_S_support", "is_V_veto"}
    missing = sorted(required.difference(sdf.columns))
    if missing:
        raise RuntimeError(f"S1 carrier_role_rows missing columns: {missing}")

    ids = sdf["carrier_id"].to_numpy(dtype=np.int64, copy=False)
    order = np.argsort(ids, kind="mergesort")
    ids_sorted = ids[order]
    target = np.asarray(carrier_id, dtype=np.int64)
    pos = np.searchsorted(ids_sorted, target)
    found = (pos < ids_sorted.shape[0]) & (ids_sorted[np.minimum(pos, ids_sorted.shape[0] - 1)] == target)
    if not np.all(found):
        raise RuntimeError(f"S1 role rows do not cover all carrier ids for {scene}; missing={int(np.count_nonzero(~found))}")

    def col(name: str) -> np.ndarray:
        return sdf[name].to_numpy(dtype=bool, copy=False)[order][pos]

    variant = ""
    if "selected_variant_id" in sdf.columns:
        values = [str(v) for v in sdf["selected_variant_id"].dropna().unique().tolist()]
        variant = values[0] if values else ""
    return {"A_anchor": col("is_A_anchor"), "S_support": col("is_S_support"), "V_veto": col("is_V_veto")}, variant


def _visible_by_frame(diag: dict[str, Any], carrier_indices: np.ndarray) -> np.ndarray:
    frame_count = len(diag["frame_ids"])
    visible = np.zeros((frame_count,), dtype=np.int64)
    in_image = np.asarray(diag["in_image"], dtype=bool)
    for fi in range(frame_count):
        visible[fi] = int(np.count_nonzero(in_image[fi, carrier_indices])) if carrier_indices.size else 0
    return visible


def _remap_incidence_to_union(incidence: np.ndarray, role_indices: np.ndarray, union_inverse: np.ndarray) -> np.ndarray:
    if incidence.size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    out = incidence.copy()
    role_local = out[:, 0].astype(np.int64, copy=False)
    global_idx = np.asarray(role_indices, dtype=np.int64)[role_local]
    union_local = union_inverse[global_idx]
    if np.any(union_local < 0):
        raise RuntimeError("role incidence carrier not present in union carrier set")
    out[:, 0] = union_local.astype(np.float64)
    return out


def _build_role_incidence(
    *,
    diag: dict[str, Any],
    arrays: dict[str, np.ndarray],
    batch: dict[str, np.ndarray],
    carrier_indices: np.ndarray,
    union_inverse: np.ndarray,
    obs_lookup: dict[tuple[int, int], int],
    affinity_risk_mode: str,
) -> np.ndarray:
    local_inc = _build_incidence(
        diag=diag,
        arrays=arrays,
        batch=batch,
        carrier_indices=carrier_indices,
        obs_lookup=obs_lookup,
        variant={},
        affinity_risk_mode=affinity_risk_mode,
    )
    return _remap_incidence_to_union(local_inc, carrier_indices, union_inverse)


def _carrier_semantic_weight(arrays: dict[str, np.ndarray]) -> np.ndarray:
    stability = np.asarray(arrays.get("semantic_short_range_stability", np.ones_like(arrays["carrier_id"], dtype=np.float32)), dtype=np.float32)
    contradiction = np.asarray(arrays.get("semantic_contradiction_rate", np.zeros_like(stability)), dtype=np.float32)
    return np.clip(stability * (1.0 - contradiction), 0.0, 1.0).astype(np.float32, copy=False)


def _carrier_veto_score(arrays: dict[str, np.ndarray], v_role: np.ndarray) -> np.ndarray:
    broad = np.asarray(arrays.get("broad_mask_participation_rate", np.zeros_like(v_role, dtype=np.float32)), dtype=np.float32)
    competing = np.asarray(arrays.get("competing_mask_conflict_rate", np.zeros_like(broad)), dtype=np.float32)
    contradiction = np.asarray(arrays.get("semantic_contradiction_rate", np.zeros_like(broad)), dtype=np.float32)
    source = np.asarray(arrays.get("source_risk_score", np.zeros_like(broad)), dtype=np.float32)
    score = np.maximum.reduce([broad, competing, contradiction, source])
    score = np.maximum(score, np.asarray(v_role, dtype=np.float32))
    return np.clip(score, 0.0, 1.0).astype(np.float32, copy=False)


def _scale_support_incidence(
    support_incidence: np.ndarray,
    union_carrier_indices: np.ndarray,
    carrier_sem_weight: np.ndarray,
    carrier_veto_score: np.ndarray,
    variant: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if support_incidence.size == 0 or float(variant["support_lambda"]) <= 0.0:
        return np.zeros((0, 5), dtype=np.float64), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    out = support_incidence.copy()
    local = out[:, 0].astype(np.int64, copy=False)
    global_idx = np.asarray(union_carrier_indices, dtype=np.int64)[local]
    sem = carrier_sem_weight[global_idx].astype(np.float32, copy=False) if bool(variant["semantic_filter"]) else np.ones(local.shape[0], dtype=np.float32)
    semantic_gate_min = variant.get("semantic_gate_min", None)
    if semantic_gate_min is not None:
        sem = sem * (sem >= float(semantic_gate_min)).astype(np.float32, copy=False)
    veto_score = carrier_veto_score[global_idx].astype(np.float32, copy=False)
    veto = (1.0 - veto_score).astype(np.float32, copy=False) if bool(variant["veto_attenuation"]) else np.ones(local.shape[0], dtype=np.float32)
    factor = float(variant["support_lambda"]) * sem * veto
    out[:, 4] = out[:, 4].astype(np.float64) * factor.astype(np.float64)
    keep = np.isfinite(out[:, 4]) & (out[:, 4] > 0.0)
    return out[keep].astype(np.float64, copy=False), sem[keep], veto_score[keep]


def _support_counts(mask_idx: np.ndarray, carrier_idx: np.ndarray, mask_count: int) -> np.ndarray:
    if mask_idx.size == 0:
        return np.zeros((mask_count,), dtype=np.int64)
    pair = np.stack([mask_idx.astype(np.int64, copy=False), carrier_idx.astype(np.int64, copy=False)], axis=1)
    uniq = np.unique(pair, axis=0)
    return np.bincount(uniq[:, 0], minlength=int(mask_count)).astype(np.int64, copy=False)


def _pair_separation(
    feature: np.ndarray,
    incidence_by_mask: list[np.ndarray],
    carrier_idx: np.ndarray,
    mask_frame: np.ndarray,
    mask_is_object: np.ndarray,
    mask_is_broad: np.ndarray,
    max_pairs: int,
    device: torch.device,
) -> dict[str, Any]:
    pseudo, hard, same_frame, broad = _sample_mask_pairs(
        incidence_by_mask,
        carrier_idx,
        mask_frame,
        mask_is_object,
        mask_is_broad,
        int(max_pairs),
    )
    pos = _pair_values_strict_leave_two_out_bucket_zeroed(feature, pseudo, device=device)
    neg = _pair_values_strict_leave_two_out_bucket_zeroed(feature, hard, device=device)
    same_vals = _pair_values_strict_leave_two_out_bucket_zeroed(feature, same_frame, device=device)
    broad_vals = _pair_values_strict_leave_two_out_bucket_zeroed(feature, broad, device=device)
    return {
        "pseudo_positive_affinity_mean": float(np.mean(pos)) if pos.size else 0.0,
        "hard_negative_affinity_mean": float(np.mean(neg)) if neg.size else 0.0,
        "hard_negative_separation": (float(np.mean(pos)) if pos.size else 0.0) - (float(np.mean(neg)) if neg.size else 0.0),
        "pseudo_positive_pair_count": int(pos.size),
        "hard_negative_pair_count": int(neg.size),
        "same_frame_pair_count": int(same_vals.size),
        "broad_pair_count": int(broad_vals.size),
        "same_frame_competing_mask_affinity_p95": float(np.percentile(same_vals, 95)) if same_vals.size else 0.0,
        "broad_mask_affinity_p95": float(np.percentile(broad_vals, 95)) if broad_vals.size else 0.0,
    }


def _gate_row(scene: str, variant_id: str, gate_id: str, passed: bool, observed: Any, required: str, blocking: bool) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r5_phaseR5_1_gate_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "variant_id": variant_id,
        "gate_id": gate_id,
        "pass": bool(passed),
        "observed": json.dumps(_jsonable(observed), sort_keys=True) if isinstance(observed, (dict, list, tuple)) else observed,
        "required": required,
        "blocking_for_variant": bool(blocking),
        "uses_gt": False,
        "uses_future": False,
    }


def _run_scene(
    scene: str,
    phase2_root: Path,
    phaseS1_root: Path,
    role_df: pd.DataFrame,
    output_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    t0 = time.time()
    scene_out = output_root / scene
    scene_out.mkdir(parents=True, exist_ok=True)
    spec = dict(SCENE_INPUTS[scene])
    spec["phase2_root"] = phase2_root
    diag, _unused_a, _unused_b, arrays = _compute_scene_arrays(scene, spec, scene_out, int(args.cupy_device_id))
    cache_dir, _manifest = _ensure_mmap_cache(phase2_root)
    batch = _load_cached(cache_dir)
    carrier_id = np.asarray(arrays["carrier_id"], dtype=np.int64)
    role_flags, selected_s1_variant = _role_flags_for_scene(role_df, scene, carrier_id)
    mask_frame, mask_label, mask_is_object, mask_is_broad, obs_lookup = _mask_observations(diag)
    mask_count = int(mask_frame.shape[0])

    a_mask = role_flags["A_anchor"] & ~role_flags["V_veto"]
    s_mask = role_flags["S_support"]
    union_mask = a_mask | s_mask
    union_indices = np.flatnonzero(union_mask).astype(np.int64)
    if union_indices.size == 0:
        raise RuntimeError(f"{scene} has no A/S carrier union")
    union_inverse = np.full(carrier_id.shape[0], -1, dtype=np.int64)
    union_inverse[union_indices] = np.arange(union_indices.shape[0], dtype=np.int64)

    a_indices = np.flatnonzero(a_mask).astype(np.int64)
    s_indices = np.flatnonzero(s_mask).astype(np.int64)
    a_inc = _build_role_incidence(
        diag=diag,
        arrays=arrays,
        batch=batch,
        carrier_indices=a_indices,
        union_inverse=union_inverse,
        obs_lookup=obs_lookup,
        affinity_risk_mode=str(args.affinity_risk_mode),
    )
    s_inc = _build_role_incidence(
        diag=diag,
        arrays=arrays,
        batch=batch,
        carrier_indices=s_indices,
        union_inverse=union_inverse,
        obs_lookup=obs_lookup,
        affinity_risk_mode=str(args.affinity_risk_mode),
    )
    base_inc = np.concatenate([a_inc, s_inc], axis=0) if a_inc.size and s_inc.size else (a_inc if a_inc.size else s_inc)
    visible = _visible_by_frame(diag, union_indices)
    weights, weight_meta = _mask_weights(
        incidence=base_inc,
        mask_count=mask_count,
        mask_frame=mask_frame,
        mask_is_object=mask_is_object,
        mask_is_broad=mask_is_broad,
        visible_reliable_by_frame=visible,
        specificity_mode=str(args.specificity_mode),
        specificity_alpha=float(args.specificity_alpha),
        no_idf=False,
    )
    sem_weight = _carrier_semantic_weight(arrays)
    veto_score = _carrier_veto_score(arrays, role_flags["V_veto"])
    carrier_broad = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float32)[union_indices]

    feature_scenes: dict[str, Any] = {
        "selected_phaseS1_variant_id": selected_s1_variant,
        "carrier_count": int(union_indices.shape[0]),
        "carrier_id": torch.as_tensor(carrier_id[union_indices], dtype=torch.int64),
        "is_A_anchor": torch.as_tensor(a_mask[union_indices], dtype=torch.bool),
        "is_S_support": torch.as_tensor(s_mask[union_indices], dtype=torch.bool),
        "is_V_veto": torch.as_tensor(role_flags["V_veto"][union_indices], dtype=torch.bool),
        "mask_observation_index": torch.arange(mask_count, dtype=torch.int64),
        "mask_frame": torch.as_tensor(mask_frame.astype(np.int64), dtype=torch.int64),
        "mask_label": torch.as_tensor(mask_label.astype(np.int64), dtype=torch.int64),
        "mask_is_object_like": torch.as_tensor(mask_is_object.astype(bool), dtype=torch.bool),
        "mask_is_broad": torch.as_tensor(mask_is_broad.astype(bool), dtype=torch.bool),
        "mask_weight": torch.as_tensor(weights.astype(np.float32), dtype=torch.float32),
        "variants": {},
    }

    summary_rows: list[dict[str, Any]] = []
    affinity_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    f0_broad_ratio: float | None = None

    for variant in VARIANTS:
        variant_t0 = time.time()
        variant_id = str(variant["variant_id"])
        s_scaled, s_sem_rows, s_veto_rows = _scale_support_incidence(s_inc, union_indices, sem_weight, veto_score, variant)
        anchor_scale = float(variant.get("anchor_scale", 1.0))
        if anchor_scale > 0.0 and a_inc.size:
            a_scaled = a_inc.copy()
            a_scaled[:, 4] = a_scaled[:, 4].astype(np.float64) * anchor_scale
        else:
            a_scaled = np.zeros((0, 5), dtype=np.float64)
        if s_scaled.size:
            incidence = np.concatenate([a_scaled, s_scaled], axis=0) if a_scaled.size else s_scaled
            role_is_support = np.concatenate(
                [np.zeros(a_scaled.shape[0], dtype=bool), np.ones(s_scaled.shape[0], dtype=bool)]
            )
        else:
            incidence = a_scaled.copy()
            role_is_support = np.zeros(a_scaled.shape[0], dtype=bool)
        if incidence.size == 0:
            incidence = np.zeros((0, 5), dtype=np.float64)
            role_is_support = np.zeros((0,), dtype=bool)

        primitive, primitive_runtime = _countsketch(incidence, weights, int(union_indices.shape[0]), int(args.sketch_dim), device)
        subset, exact = _exact_dense_subset(incidence, weights, int(union_indices.shape[0]), mask_count, int(args.exact_subset_size))
        p95_error, max_error = _pair_error(exact, primitive, subset)

        if incidence.size:
            carrier_idx = incidence[:, 0].astype(np.int64, copy=False)
            mask_idx = incidence[:, 1].astype(np.int64, copy=False)
            b_ia = incidence[:, 4].astype(np.float32, copy=False)
        else:
            carrier_idx = np.zeros((0,), dtype=np.int64)
            mask_idx = np.zeros((0,), dtype=np.int64)
            b_ia = np.zeros((0,), dtype=np.float32)
        incidence_by_mask = [np.flatnonzero(mask_idx == m).astype(np.int64) for m in range(mask_count)]
        raw = _build_raw_sketch(carrier_idx, mask_idx, b_ia, weights, int(union_indices.shape[0]), int(args.sketch_dim), device)
        alpha = b_ia.astype(np.float32, copy=False)
        mask_feature, mask_runtime = _pool_features(
            variant_id="P0_mean_reliability_weighted",
            raw=raw,
            incidence_by_mask=incidence_by_mask,
            carrier_idx=carrier_idx,
            mask_idx=mask_idx,
            b_ia=b_ia,
            alpha=alpha,
            carrier_broad=carrier_broad,
            mask_weight=weights,
            topk=int(args.topk_carriers),
            trim_quantile=float(args.trim_quantile),
            device=device,
        )
        pair_meta = _pair_separation(
            mask_feature,
            incidence_by_mask,
            carrier_idx,
            mask_frame,
            mask_is_object,
            mask_is_broad,
            int(args.max_pair_rows),
            device,
        )

        mass = np.sqrt(weights[mask_idx].astype(np.float64)) * np.abs(b_ia.astype(np.float64)) if mask_idx.size else np.zeros((0,), dtype=np.float64)
        total_mass = float(np.sum(mass))
        support_mass = float(np.sum(mass[role_is_support])) if mass.size else 0.0
        broad_mass = float(np.sum(mass[mask_is_broad[mask_idx]])) if mass.size else 0.0
        object_mass = float(np.sum(mass[mask_is_object[mask_idx]])) if mass.size else 0.0
        if role_is_support.any():
            local_s = carrier_idx[role_is_support]
            global_s = union_indices[local_s]
            veto_overlap_support = role_flags["V_veto"][global_s]
            veto_overlap_mass = float(np.sum(mass[role_is_support][veto_overlap_support]))
        else:
            veto_overlap_mass = 0.0
        support_count = _support_counts(mask_idx[role_is_support], carrier_idx[role_is_support], mask_count) if role_is_support.any() else np.zeros((mask_count,), dtype=np.int64)
        anchor_count = _support_counts(mask_idx[~role_is_support], carrier_idx[~role_is_support], mask_count) if (~role_is_support).any() else np.zeros((mask_count,), dtype=np.int64)
        all_count = _support_counts(mask_idx, carrier_idx, mask_count)
        norms = np.linalg.norm(mask_feature, axis=1) if mask_feature.size else np.zeros((mask_count,), dtype=np.float32)
        support_ratio = float(support_mass / max(total_mass, 1e-12))
        broad_ratio = float(broad_mass / max(total_mass, 1e-12))
        object_ratio = float(object_mass / max(total_mass, 1e-12))
        veto_ratio = float(veto_overlap_mass / max(total_mass, 1e-12))
        if variant_id == "F0_anchor_only":
            f0_broad_ratio = broad_ratio
        broad_required = (float(f0_broad_ratio) + float(args.broad_plus_budget)) if f0_broad_ratio is not None else float("inf")
        bucket = _bucket_stats(mask_count, int(args.sketch_dim))
        row = {
            "schema_version": "stream4d_v103_supp_r5_phaseR5_1_role_feature_summary_row_v1",
            "phase_id": PHASE_ID,
            "variant_id": variant_id,
            "scene_id": scene,
            "selected_phaseS1_variant_id": selected_s1_variant,
            "variant_description": str(variant["description"]),
            "carrier_or_segment_count": int(union_indices.shape[0]),
            "A_anchor_incidence_count": int(a_inc.shape[0]),
            "S_support_incidence_count": int(s_inc.shape[0]),
            "S_support_weighted_incidence_count": int(np.count_nonzero(role_is_support)),
            "support_contribution_ratio": support_ratio,
            "broad_contribution_ratio": broad_ratio,
            "object_like_contribution_ratio": object_ratio,
            "veto_overlap_contribution_ratio": veto_ratio,
            "countsketch_dim": int(args.sketch_dim),
            "bucket_load_mean": bucket["bucket_load_mean"],
            "bucket_load_p95": bucket["bucket_load_p95"],
            "collision_mass_ratio": bucket["collision_mass_ratio"],
            "exact_vs_sketch_cosine_p95_error": float(p95_error),
            "exact_vs_sketch_cosine_max_error": float(max_error),
            "mask_feature_valid_rate": _valid_rate(mask_feature, mask_is_object),
            "mask_feature_valid_rate_all": _valid_rate(mask_feature),
            "mask_feature_norm_p50": _safe_p(norms, 50),
            "mask_feature_norm_p05": _safe_p(norms, 5),
            "object_like_anchor_count_p10": _safe_p(anchor_count, 10, mask_is_object),
            "object_like_support_count_p10": _safe_p(support_count, 10, mask_is_object),
            "object_like_all_count_p10": _safe_p(all_count, 10, mask_is_object),
            "semantic_filter_enabled": bool(variant["semantic_filter"]),
            "semantic_gate_min": variant.get("semantic_gate_min", ""),
            "veto_attenuation_enabled": bool(variant["veto_attenuation"]),
            "anchor_scale": anchor_scale,
            "diagnostic_only": bool(variant.get("diagnostic_only", False)),
            "support_lambda": float(variant["support_lambda"]),
            "support_semantic_weight_mean": float(np.mean(s_sem_rows)) if s_sem_rows.size else "",
            "support_veto_score_mean": float(np.mean(s_veto_rows)) if s_veto_rows.size else "",
            "primitive_runtime_sec": float(primitive_runtime),
            "mask_pool_runtime_sec": float(mask_runtime),
            "variant_runtime_sec": time.time() - variant_t0,
            "uses_gt": False,
            "uses_future": False,
            **pair_meta,
            **{f"mask_weight_{k}": v for k, v in weight_meta.items()},
        }
        summary_rows.append(row)
        affinity_rows.extend(
            [
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_1_role_affinity_feature_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
            "role_family": "A_anchor",
            "carrier_or_segment_count": int(np.count_nonzero(a_mask)),
            "raw_incidence_count": int(a_inc.shape[0]),
                    "weighted_incidence_count": int(a_scaled.shape[0]),
                    "weighted_mass": float(total_mass - support_mass),
                    "contribution_ratio": float((total_mass - support_mass) / max(total_mass, 1e-12)),
                    "uses_gt": False,
                    "uses_future": False,
                },
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_1_role_affinity_feature_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "role_family": "S_support",
                    "carrier_or_segment_count": int(np.count_nonzero(s_mask)),
                    "raw_incidence_count": int(s_inc.shape[0]),
                    "weighted_incidence_count": int(np.count_nonzero(role_is_support)),
                    "weighted_mass": support_mass,
                    "contribution_ratio": support_ratio,
                    "veto_overlap_weighted_mass": veto_overlap_mass,
                    "veto_overlap_contribution_ratio": veto_ratio,
                    "uses_gt": False,
                    "uses_future": False,
                },
            ]
        )

        is_f0 = variant_id == "F0_anchor_only"
        support_gate_applicable = not is_f0
        diagnostic_only = bool(variant.get("diagnostic_only", False))
        feature_gate_blocking = (not is_f0) and (not diagnostic_only)
        support_gate_blocking = support_gate_applicable and (not diagnostic_only)
        gates = [
            _gate_row(scene, variant_id, "exact_vs_sketch_cosine_p95_error_le_0p005", float(p95_error) <= float(args.exact_p95_threshold), p95_error, f"<={args.exact_p95_threshold}", True),
            _gate_row(scene, variant_id, "mask_feature_valid_rate_ge_0p95", row["mask_feature_valid_rate"] >= float(args.valid_rate_threshold), row["mask_feature_valid_rate"], f">={args.valid_rate_threshold}", feature_gate_blocking),
            _gate_row(scene, variant_id, "support_contribution_ratio_in_range", (not support_gate_applicable) or (float(args.support_ratio_min) <= support_ratio <= float(args.support_ratio_max)), support_ratio, f"[{args.support_ratio_min},{args.support_ratio_max}] for support variants; n/a for F0; diagnostic-only variants are nonblocking", support_gate_blocking),
            _gate_row(scene, variant_id, "broad_contribution_ratio_le_anchor_plus_0p10", (f0_broad_ratio is None) or (broad_ratio <= broad_required + 1e-12), {"current": broad_ratio, "anchor_only": f0_broad_ratio}, f"<= anchor-only + {args.broad_plus_budget}; diagnostic-only variants are nonblocking", support_gate_blocking),
            _gate_row(scene, variant_id, "veto_overlap_contribution_ratio_le_half_support", (not support_gate_applicable) or (veto_ratio <= support_ratio * float(args.veto_overlap_fraction_max) + 1e-12), {"veto_overlap": veto_ratio, "support": support_ratio}, f"<= support_contribution_ratio * {args.veto_overlap_fraction_max}; diagnostic-only variants are nonblocking", support_gate_blocking),
        ]
        gate_rows.extend(gates)
        feature_scenes["variants"][variant_id] = {
            "description": str(variant["description"]),
            "support_lambda": float(variant["support_lambda"]),
            "anchor_scale": anchor_scale,
            "diagnostic_only": diagnostic_only,
            "semantic_filter_enabled": bool(variant["semantic_filter"]),
            "semantic_gate_min": variant.get("semantic_gate_min", ""),
            "veto_attenuation_enabled": bool(variant["veto_attenuation"]),
            "mask_feature": torch.as_tensor(mask_feature.astype(np.float16, copy=False), dtype=torch.float16),
            "anchor_support_count": torch.as_tensor(anchor_count.astype(np.int64, copy=False), dtype=torch.int64),
            "support_support_count": torch.as_tensor(support_count.astype(np.int64, copy=False), dtype=torch.int64),
            "all_support_count": torch.as_tensor(all_count.astype(np.int64, copy=False), dtype=torch.int64),
            "summary": row,
        }
        if bool(args.save_primitive_features):
            feature_scenes["variants"][variant_id]["primitive_feature"] = torch.as_tensor(primitive.astype(np.float16, copy=False), dtype=torch.float16)
        del primitive, raw
        if device.type == "cuda":
            torch.cuda.empty_cache()

    performance_rows.append(
        {
            "schema_version": "stream4d_v103_supp_r5_phaseR5_1_performance_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "runtime_sec": time.time() - t0,
            "torch_device": str(device),
            "cupy_device_id": int(args.cupy_device_id),
            "projection_backend": diag["performance"].get("projection_backend", ""),
            "semantic_backend": diag["performance"].get("semantic_backend", ""),
            "uses_gt": False,
            "uses_future": False,
        }
    )
    return summary_rows, affinity_rows, gate_rows, feature_scenes, performance_rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    fact_lock_root = _project(args.fact_lock_root)
    fact = _read_json(fact_lock_root / "summary.json")
    if not bool(fact.get("phase_r5_0_pass", False)):
        raise RuntimeError(f"R5-0 fact lock has not passed: {fact_lock_root / 'summary.json'}")
    phaseS1_root = _project(args.phaseS1_root)
    phaseS1_summary = _read_json(phaseS1_root / "summary.json")
    if phaseS1_summary.get("decision") != "PASS_ENTER_PHASES2_ROLE_AWARE_AFFINITY":
        raise RuntimeError(f"Phase S1 has not passed: {phaseS1_root / 'summary.json'}")
    role_df = pd.read_parquet(
        phaseS1_root / "carrier_role_rows.parquet",
        columns=["scene_id", "selected_variant_id", "carrier_id", "is_A_anchor", "is_S_support", "is_V_veto"],
    )
    phase2_roots = {
        "scene0011_00": _project(args.scene0011_d4rt_root),
        "scene0050_00": _project(args.scene0050_d4rt_root),
    }
    device = _torch_device(str(args.torch_device))

    all_summary_rows: list[dict[str, Any]] = []
    all_affinity_rows: list[dict[str, Any]] = []
    all_gate_rows: list[dict[str, Any]] = []
    all_performance_rows: list[dict[str, Any]] = []
    feature_payload = {
        "schema_version": "stream4d_v103_supp_r5_phaseR5_1_role_mask_level_feature_v1",
        "phase_id": PHASE_ID,
        "sketch_seed": 10317,
        "sketch_dim": int(args.sketch_dim),
        "specificity_mode": str(args.specificity_mode),
        "affinity_risk_mode": str(args.affinity_risk_mode),
        "clip_backfill_policy": {
            "clip_used_in_R5_1": False,
            "low_resolution_compact_feature_map_allowed_by_plan": True,
            "high_resolution_high_dim_dense_pixel_map_allowed": False,
        },
        "uses_gt": False,
        "uses_future": False,
        "scenes": {},
    }
    for scene in ["scene0011_00", "scene0050_00"]:
        rows, affinity_rows, gates, scene_payload, perf = _run_scene(
            scene=scene,
            phase2_root=phase2_roots[scene],
            phaseS1_root=phaseS1_root,
            role_df=role_df,
            output_root=out,
            args=args,
            device=device,
        )
        all_summary_rows.extend(rows)
        all_affinity_rows.extend(affinity_rows)
        all_gate_rows.extend(gates)
        all_performance_rows.extend(perf)
        feature_payload["scenes"][scene] = scene_payload

    summary_df = pd.DataFrame(all_summary_rows)
    gate_df = pd.DataFrame(all_gate_rows)
    support_variants = [
        v["variant_id"]
        for v in VARIANTS
        if v["variant_id"] != "F0_anchor_only" and not bool(v.get("diagnostic_only", False))
    ]
    passing_support_variants: list[str] = []
    for variant_id in support_variants:
        sub = gate_df[(gate_df["variant_id"].astype(str) == variant_id) & (gate_df["blocking_for_variant"].astype(bool))]
        if not sub.empty and bool(sub["pass"].all()):
            passing_support_variants.append(str(variant_id))
    f0_sub = gate_df[(gate_df["variant_id"].astype(str) == "F0_anchor_only") & (gate_df["blocking_for_variant"].astype(bool))]
    f0_pass = (not f0_sub.empty) and bool(f0_sub["pass"].all())
    phase_pass = bool(passing_support_variants)
    failure_rows = []
    if not passing_support_variants:
        failure_rows.append(
            {
                "schema_version": "stream4d_v103_supp_r5_phaseR5_1_failure_row_v1",
                "phase_id": PHASE_ID,
                "blocker": "SUPPORT_WEIGHTED_FEATURE_NO_PASSING_VARIANT",
                "detail": "No support-weighted F1-F6 variant passed exact/valid/support-ratio/broad/veto-overlap gates across both scenes.",
                "repair_direction": "Follow R5 repair ladder: strengthen IDF, lower lambda_S, enable semantic-filtered/veto-attenuated support; if still failing, move support to veto/score branch.",
            }
        )
    if not phase_pass:
        bad_broad = summary_df[
            (summary_df["variant_id"].astype(str) != "F0_anchor_only")
            & (summary_df["broad_contribution_ratio"].astype(float) > summary_df.groupby("scene_id")["broad_contribution_ratio"].transform("min").astype(float) + float(args.broad_plus_budget))
        ]
        if not bad_broad.empty:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_1_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "blocker": "SUPPORT_WEIGHTED_FEATURE_BROAD_LEAKAGE",
                    "detail": f"broad_leak_variant_rows={len(bad_broad)}",
                    "repair_direction": "Lower lambda_S or increase support veto/broad attenuation before AP phases.",
                }
            )

    pd.DataFrame(all_affinity_rows).to_parquet(out / "role_affinity_feature_rows.parquet", index=False)
    _write_csv(out / "role_feature_summary_rows.csv", all_summary_rows)
    _write_csv(out / "role_feature_gate_rows.csv", all_gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "performance_rows.csv", all_performance_rows)
    feature_path = out / "role_mask_level_feature.pt"
    torch.save(feature_payload, feature_path)

    artifact_rows = [
        {
            "schema_version": "stream4d_v103_supp_r5_phaseR5_1_artifact_row_v1",
            "phase_id": PHASE_ID,
            "artifact_role": role,
            "path": _rel(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "uses_gt": False,
            "uses_future": False,
        }
        for role, path in [
            ("role_affinity_feature_rows", out / "role_affinity_feature_rows.parquet"),
            ("role_feature_summary_rows", out / "role_feature_summary_rows.csv"),
            ("role_feature_gate_rows", out / "role_feature_gate_rows.csv"),
            ("role_mask_level_feature", feature_path),
            ("failure_rows", out / "failure_rows.csv"),
            ("performance_rows", out / "performance_rows.csv"),
            ("last_command", out / "last_command.txt"),
        ]
    ]
    _write_csv(out / "artifact_rows.csv", artifact_rows)

    summary = {
        "schema_version": "stream4d_v103_supp_r5_phaseR5_1_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "decision": "PASS_ENTER_PHASE_R5_2_SUPPORT_EDGE_ATTRIBUTION" if phase_pass else "NO_GO_REPAIR_PHASE_R5_1_SUPPORT_WEIGHTED_FEATURE",
        "phase_r5_1_pass": bool(phase_pass),
        "failure_count": int(len(failure_rows)),
        "fact_lock_root": _rel(fact_lock_root),
        "phaseS1_root": _rel(phaseS1_root),
        "scene_ids": ["scene0011_00", "scene0050_00"],
        "variant_ids": [str(v["variant_id"]) for v in VARIANTS],
        "passing_support_variants": passing_support_variants,
        "f0_anchor_only_pass": bool(f0_pass),
        "sketch_dim": int(args.sketch_dim),
        "exact_p95_threshold": float(args.exact_p95_threshold),
        "valid_rate_threshold": float(args.valid_rate_threshold),
        "support_ratio_range": [float(args.support_ratio_min), float(args.support_ratio_max)],
        "broad_plus_budget": float(args.broad_plus_budget),
        "veto_overlap_fraction_max": float(args.veto_overlap_fraction_max),
        "runs_AP": False,
        "uses_gt_for_gate": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - t0,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "role_affinity_feature_rows": _rel(out / "role_affinity_feature_rows.parquet"),
            "role_feature_summary_rows": _rel(out / "role_feature_summary_rows.csv"),
            "role_feature_gate_rows": _rel(out / "role_feature_gate_rows.csv"),
            "role_mask_level_feature": _rel(feature_path),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "performance_rows": _rel(out / "performance_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
        "truthfulness_note": (
            "R5-1 constructs and audits support-weighted primitive/mask-level affinity features. "
            "It does not run local AP, does not use GT, and does not construct object predictions. "
            "CLIP crop backfill is not used in this phase; low-resolution compact CLIP maps remain allowed by the plan but are not claimed here."
        ),
    }
    _write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream4D v103 R5 Phase R5-1 support-weighted affinity feature audit.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--fact-lock-root", default=str(DEFAULT_FACT_LOCK_ROOT))
    parser.add_argument("--phaseS1-root", default=str(DEFAULT_PHASES1_ROOT))
    parser.add_argument("--scene0011-d4rt-root", default=str(DEFAULT_D4RT_ROOT_BY_SCENE["scene0011_00"]))
    parser.add_argument("--scene0050-d4rt-root", default=str(DEFAULT_D4RT_ROOT_BY_SCENE["scene0050_00"]))
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--sketch-dim", type=int, default=256)
    parser.add_argument("--exact-subset-size", type=int, default=4096)
    parser.add_argument("--max-pair-rows", type=int, default=4096)
    parser.add_argument("--topk-carriers", type=int, default=64)
    parser.add_argument("--trim-quantile", type=float, default=0.10)
    parser.add_argument("--specificity-mode", default="idf_object_preserve_downweight")
    parser.add_argument("--specificity-alpha", type=float, default=1.0)
    parser.add_argument("--affinity-risk-mode", default="source_and_competing_penalty")
    parser.add_argument("--exact-p95-threshold", type=float, default=0.005)
    parser.add_argument("--valid-rate-threshold", type=float, default=0.95)
    parser.add_argument("--support-ratio-min", type=float, default=0.05)
    parser.add_argument("--support-ratio-max", type=float, default=0.45)
    parser.add_argument("--broad-plus-budget", type=float, default=0.10)
    parser.add_argument("--veto-overlap-fraction-max", type=float, default=0.50)
    parser.add_argument("--save-primitive-features", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["phase_r5_1_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
