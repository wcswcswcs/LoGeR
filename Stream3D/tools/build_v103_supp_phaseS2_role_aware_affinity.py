#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from itertools import combinations
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
    _countsketch,
    _ensure_mmap_cache,
    _exact_dense_subset,
    _hash_mask,
    _load_cached,
    _mask_observations,
    _mask_weights,
    _pair_error,
    _compute_scene_arrays,
)
from build_v103_phase5_mask_level_affinity_pooling import (  # noqa: E402
    _build_raw_sketch,
    _pair_values_static,
    _pool_features,
    _sample_mask_pairs,
)


PHASE_ID = "v103_supp_phaseS2_role_aware_affinity"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_phaseS2_role_aware_affinity"
DEFAULT_PHASES0_ROOT = AUDIT_ROOT / "v103_supp_phaseS0_fact_lock"
DEFAULT_PHASES1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers"
SUPPLEMENT_PLAN_DOC = REPO_ROOT / "docs/stream4d_v103_supplement_multirole_carrier_affinity_field_plan.md"

DEFAULT_PHASE2_ROOT_BY_SCENE = {
    "scene0011_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "scene0050_00": AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
}


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


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


def _role_flags_for_scene(role_df: pd.DataFrame, scene: str, carrier_id: np.ndarray) -> tuple[dict[str, np.ndarray], str]:
    sdf = role_df[role_df["scene_id"] == scene].copy()
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
        missing_count = int(np.count_nonzero(~found))
        raise RuntimeError(f"S1 carrier rows do not cover scene={scene} carrier ids; missing_count={missing_count}")

    def col(name: str) -> np.ndarray:
        return sdf[name].to_numpy(dtype=bool, copy=False)[order][pos]

    variant = ""
    if "selected_variant_id" in sdf.columns:
        values = [str(v) for v in sdf["selected_variant_id"].dropna().unique().tolist()]
        variant = values[0] if values else ""
    return {
        "A_anchor": col("is_A_anchor"),
        "S_support_raw": col("is_S_support"),
        "V_veto": col("is_V_veto"),
    }, variant


def _visible_by_frame(diag: dict[str, Any], carrier_indices: np.ndarray) -> np.ndarray:
    frame_count = len(diag["frame_ids"])
    visible = np.zeros((frame_count,), dtype=np.int64)
    in_image = np.asarray(diag["in_image"], dtype=bool)
    for fi in range(frame_count):
        visible[fi] = int(np.count_nonzero(in_image[fi, carrier_indices])) if carrier_indices.size else 0
    return visible


def _safe_p10(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, 10))


def _valid_rate(feature: np.ndarray, mask: np.ndarray | None = None) -> float:
    if feature.size == 0:
        return 0.0
    valid = np.linalg.norm(feature, axis=1) > 0.0
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return 0.0
        valid = valid[mask]
    return float(np.mean(valid))


def _norm_mean(feature: np.ndarray) -> float:
    if feature.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(feature, axis=1)))


def _loo_drop(loo_feature: np.ndarray, no_loo_feature: np.ndarray) -> float:
    if loo_feature.size == 0 or no_loo_feature.size == 0:
        return 0.0
    loo_norm = np.linalg.norm(loo_feature, axis=1)
    no_norm = np.linalg.norm(no_loo_feature, axis=1)
    valid = (loo_norm > 0.0) & (no_norm > 0.0)
    if not np.any(valid):
        return 0.0
    cos = np.sum(loo_feature[valid] * no_loo_feature[valid], axis=1)
    return float(np.mean(np.maximum(0.0, 1.0 - cos)))


def _bucket_stats(mask_count: int, sketch_dim: int) -> dict[str, float]:
    bucket, _sign = _hash_mask(np.arange(int(mask_count), dtype=np.int64), int(sketch_dim))
    load_all = np.bincount(bucket, minlength=int(sketch_dim)).astype(np.float64)
    load = load_all[load_all > 0]
    return {
        "bucket_load_p90": float(np.percentile(load, 90)) if load.size else 0.0,
        "bucket_load_p95": float(np.percentile(load, 95)) if load.size else 0.0,
        "bucket_collision_mass_mean": float(np.sum(np.maximum(load_all - 1.0, 0.0)) / max(int(mask_count), 1)),
    }


def _hard_negative_separation(
    feature: np.ndarray,
    incidence: np.ndarray,
    mask_frame: np.ndarray,
    mask_is_object: np.ndarray,
    mask_is_broad: np.ndarray,
    max_pairs: int,
) -> tuple[float, dict[str, int]]:
    if incidence.size == 0 or feature.size == 0:
        return 0.0, {
            "pseudo_positive_pair_count": 0,
            "hard_negative_pair_count": 0,
            "same_frame_pair_count": 0,
            "broad_pair_count": 0,
        }
    mask_idx = incidence[:, 1].astype(np.int64)
    carrier_idx = incidence[:, 0].astype(np.int64)
    incidence_by_mask = [np.flatnonzero(mask_idx == m).astype(np.int64) for m in range(mask_frame.shape[0])]
    pseudo, hard, same_frame, broad = _sample_mask_pairs(
        incidence_by_mask,
        carrier_idx,
        mask_frame,
        mask_is_object,
        mask_is_broad,
        int(max_pairs),
    )
    pos = _pair_values_static(feature, pseudo)
    neg = _pair_values_static(feature, hard)
    return (float(np.mean(pos)) if pos.size else 0.0) - (float(np.mean(neg)) if neg.size else 0.0), {
        "pseudo_positive_pair_count": int(pos.size),
        "hard_negative_pair_count": int(neg.size),
        "same_frame_pair_count": int(same_frame.shape[0]),
        "broad_pair_count": int(broad.shape[0]),
    }


def _build_role_incidence(
    *,
    diag: dict[str, Any],
    arrays: dict[str, np.ndarray],
    batch: dict[str, np.ndarray],
    role_mask: np.ndarray,
    obs_lookup: dict[tuple[int, int], int],
    affinity_risk_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    carrier_indices = np.flatnonzero(np.asarray(role_mask, dtype=bool)).astype(np.int64)
    incidence = _build_incidence(
        diag=diag,
        arrays=arrays,
        batch=batch,
        carrier_indices=carrier_indices,
        obs_lookup=obs_lookup,
        variant={},
        affinity_risk_mode=str(affinity_risk_mode),
    )
    return carrier_indices, incidence


def _role_features(
    *,
    role_id: str,
    diag: dict[str, Any],
    arrays: dict[str, np.ndarray],
    batch: dict[str, np.ndarray],
    role_mask: np.ndarray,
    obs_lookup: dict[tuple[int, int], int],
    mask_frame: np.ndarray,
    mask_label: np.ndarray,
    mask_is_object: np.ndarray,
    mask_is_broad: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    carrier_indices, incidence = _build_role_incidence(
        diag=diag,
        arrays=arrays,
        batch=batch,
        role_mask=role_mask,
        obs_lookup=obs_lookup,
        affinity_risk_mode=str(args.affinity_risk_mode),
    )
    visible = _visible_by_frame(diag, carrier_indices)
    weights, weight_meta = _mask_weights(
        incidence=incidence,
        mask_count=int(mask_frame.shape[0]),
        mask_frame=mask_frame,
        mask_is_object=mask_is_object,
        mask_is_broad=mask_is_broad,
        visible_reliable_by_frame=visible,
        specificity_mode=str(args.specificity_mode),
        specificity_alpha=float(args.specificity_alpha),
        no_idf=False,
    )
    primitive, primitive_runtime = _countsketch(
        incidence,
        weights,
        int(carrier_indices.shape[0]),
        int(args.sketch_dim),
        device,
    )
    subset, exact = _exact_dense_subset(
        incidence,
        weights,
        int(carrier_indices.shape[0]),
        int(mask_frame.shape[0]),
        int(args.exact_subset_size),
    )
    p95_error, max_error = _pair_error(exact, primitive, subset)

    if incidence.size:
        carrier_idx = incidence[:, 0].astype(np.int64)
        mask_idx = incidence[:, 1].astype(np.int64)
        b_ia = incidence[:, 4].astype(np.float32)
    else:
        carrier_idx = np.zeros((0,), dtype=np.int64)
        mask_idx = np.zeros((0,), dtype=np.int64)
        b_ia = np.zeros((0,), dtype=np.float32)
    support_count = np.bincount(mask_idx, minlength=int(mask_frame.shape[0])).astype(np.int64, copy=False)
    incidence_by_mask = [np.flatnonzero(mask_idx == m).astype(np.int64) for m in range(mask_frame.shape[0])]
    reliability = np.asarray(arrays["reliability_s2"], dtype=np.float32)[carrier_indices] if carrier_indices.size else np.zeros((0,), dtype=np.float32)
    carrier_broad = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float32)[carrier_indices] if carrier_indices.size else np.zeros((0,), dtype=np.float32)
    alpha = (reliability[carrier_idx] * b_ia).astype(np.float32, copy=False) if carrier_idx.size else np.zeros((0,), dtype=np.float32)
    raw = _build_raw_sketch(
        carrier_idx,
        mask_idx,
        b_ia,
        weights,
        int(carrier_indices.shape[0]),
        int(args.sketch_dim),
        device,
    )
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
    no_loo_feature, no_loo_runtime = _pool_features(
        variant_id="P4_no_leave_one_out_control",
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
    separation, pair_counts = _hard_negative_separation(
        mask_feature,
        incidence,
        mask_frame,
        mask_is_object,
        mask_is_broad,
        int(args.max_pair_rows),
    )
    bucket = _bucket_stats(int(mask_frame.shape[0]), int(args.sketch_dim))
    intervention_mask = mask_is_object | mask_is_broad
    metrics = {
        "role_id": role_id,
        "carrier_count": int(carrier_indices.shape[0]),
        "incidence_row_count": int(incidence.shape[0]),
        "valid_mask_rate": _valid_rate(mask_feature, mask_is_object),
        "valid_mask_rate_definition": "mask_is_object_like",
        "valid_mask_rate_all": _valid_rate(mask_feature),
        "object_like_valid_mask_rate": _valid_rate(mask_feature, mask_is_object),
        "broad_valid_mask_rate": _valid_rate(mask_feature, mask_is_broad),
        "intervention_valid_mask_rate": _valid_rate(mask_feature, intervention_mask),
        "intervention_mask_count": int(np.count_nonzero(intervention_mask)),
        "mask_carrier_count_p10": _safe_p10(support_count),
        "intervention_mask_carrier_count_p10": _safe_p10(support_count, intervention_mask),
        "object_like_mask_carrier_count_p10": _safe_p10(support_count, mask_is_object),
        "mask_level_feature_norm_mean": _norm_mean(mask_feature),
        "leave_one_out_self_similarity_drop_mean": _loo_drop(mask_feature, no_loo_feature),
        "exact_vs_sketch_cosine_p95_error": float(p95_error),
        "exact_vs_sketch_cosine_max_error": float(max_error),
        "exact_subset_count": int(subset.shape[0]),
        "hard_negative_separation": float(separation),
        "primitive_runtime_sec": float(primitive_runtime),
        "mask_pool_runtime_sec": float(mask_runtime),
        "no_leave_one_out_pool_runtime_sec": float(no_loo_runtime),
        **bucket,
        **pair_counts,
        **{f"mask_weight_{k}": v for k, v in weight_meta.items()},
    }
    payload = {
        "carrier_indices": carrier_indices,
        "carrier_id": np.asarray(batch["carrier_id"], dtype=np.int64)[carrier_indices],
        "primitive_feature": primitive,
        "mask_feature": mask_feature,
        "support_count": support_count,
        "mask_frame": mask_frame,
        "mask_label": mask_label,
        "mask_is_object_like": mask_is_object,
        "mask_is_broad": mask_is_broad,
        "metrics": metrics,
    }
    return payload, metrics


def _mask_carriers_from_incidence(incidence: np.ndarray, mask_count: int) -> list[np.ndarray]:
    if incidence.size == 0:
        return [np.zeros((0,), dtype=np.int64) for _ in range(int(mask_count))]
    mask_idx = incidence[:, 1].astype(np.int64)
    carrier_idx = incidence[:, 0].astype(np.int64)
    out: list[np.ndarray] = []
    for mask_id in range(int(mask_count)):
        rows = np.flatnonzero(mask_idx == int(mask_id))
        if rows.size:
            out.append(np.unique(carrier_idx[rows]).astype(np.int64, copy=False))
        else:
            out.append(np.zeros((0,), dtype=np.int64))
    return out


def _veto_pair_rows(
    *,
    scene: str,
    diag: dict[str, Any],
    arrays: dict[str, np.ndarray],
    batch: dict[str, np.ndarray],
    veto_mask: np.ndarray,
    obs_lookup: dict[tuple[int, int], int],
    mask_frame: np.ndarray,
    mask_label: np.ndarray,
    mask_is_object: np.ndarray,
    mask_is_broad: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int]:
    carrier_indices, incidence = _build_role_incidence(
        diag=diag,
        arrays=arrays,
        batch=batch,
        role_mask=veto_mask,
        obs_lookup=obs_lookup,
        affinity_risk_mode="base",
    )
    carriers_by_mask = _mask_carriers_from_incidence(incidence, int(mask_frame.shape[0]))
    rows: list[dict[str, Any]] = []
    hard_count = 0
    broad = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float32)[carrier_indices] if carrier_indices.size else np.zeros((0,), dtype=np.float32)
    sem = np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float32)[carrier_indices] if carrier_indices.size else np.zeros((0,), dtype=np.float32)
    competing = np.asarray(arrays["competing_mask_conflict_rate"], dtype=np.float32)[carrier_indices] if carrier_indices.size else np.zeros((0,), dtype=np.float32)
    jitter = np.asarray(arrays["normalized_jitter"], dtype=np.float32)[carrier_indices] if carrier_indices.size else np.zeros((0,), dtype=np.float32)

    by_frame: dict[int, list[int]] = {}
    broad_by_frame: dict[int, list[int]] = {}
    for mask_id, frame in enumerate(mask_frame.astype(np.int64).tolist()):
        if bool(mask_is_object[mask_id]) and carriers_by_mask[mask_id].size:
            by_frame.setdefault(int(frame), []).append(mask_id)
        if bool(mask_is_broad[mask_id]) and carriers_by_mask[mask_id].size:
            broad_by_frame.setdefault(int(frame), []).append(mask_id)

    candidates: list[tuple[int, int, str, bool]] = []
    max_rows = int(args.max_veto_pair_rows)
    for frame, masks in by_frame.items():
        for a, b in combinations(masks, 2):
            candidates.append((int(a), int(b), "same_frame_object_competing", True))
            if len(candidates) >= max_rows:
                break
        if len(candidates) >= max_rows:
            break
        for a in masks:
            for b in broad_by_frame.get(int(frame), []):
                candidates.append((int(a), int(b), "same_frame_object_broad", True))
                if len(candidates) >= max_rows:
                    break
            if len(candidates) >= max_rows:
                break
        if len(candidates) >= max_rows:
            break

    for pair_id, (a, b, pair_type, hard) in enumerate(candidates):
        shared = np.intersect1d(carriers_by_mask[a], carriers_by_mask[b], assume_unique=True)
        veto_union = np.union1d(carriers_by_mask[a], carriers_by_mask[b])
        if veto_union.size == 0:
            continue
        broad_mass = float(np.sum(broad[veto_union]))
        sem_mass = float(np.sum(sem[veto_union]))
        competing_mass = float(np.sum(competing[veto_union]))
        boundary_mass = float(np.sum(jitter[veto_union]))
        denom = float(max(veto_union.size, 1))
        score = 1.0 - math.exp(-float(broad_mass + sem_mass + competing_mass + boundary_mass) / denom)
        hard_active = bool(hard)
        hard_count += int(hard_active)
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_phaseS2_veto_pair_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "mask_pair_id": f"{scene}:{a}:{b}",
                "pair_type": pair_type,
                "mask_a": int(a),
                "mask_b": int(b),
                "frame_a": int(mask_frame[a]),
                "frame_b": int(mask_frame[b]),
                "mask_label_a": int(mask_label[a]),
                "mask_label_b": int(mask_label[b]),
                "veto_carrier_count": int(veto_union.size),
                "shared_veto_carrier_count": int(shared.size),
                "boundary_crossing_mass": boundary_mass,
                "boundary_crossing_proxy": "normalized_jitter_sum",
                "broad_risk_mass": broad_mass,
                "semantic_contradiction_mass": sem_mass,
                "competing_mask_conflict_mass": competing_mass,
                "veto_score": float(score),
                "hard_cannot_link_active": hard_active,
                "uses_gt": False,
                "uses_future": False,
            }
        )
        if len(rows) >= max_rows:
            break
    return rows, hard_count


def _feature_artifact(scene_payloads: dict[str, dict[str, Any]], role_key: str, artifact_kind: str) -> dict[str, Any]:
    scenes: dict[str, Any] = {}
    for scene, payloads in scene_payloads.items():
        payload = payloads[role_key]
        if artifact_kind == "primitive":
            scenes[scene] = {
                "carrier_id": torch.as_tensor(payload["carrier_id"], dtype=torch.int64),
                "feature": torch.as_tensor(payload["primitive_feature"], dtype=torch.float16),
                "feature_norm_source_dtype": "float32",
                "carrier_count": int(payload["metrics"]["carrier_count"]),
                "incidence_row_count": int(payload["metrics"]["incidence_row_count"]),
            }
        else:
            scenes[scene] = {
                "mask_observation_index": torch.arange(int(payload["mask_frame"].shape[0]), dtype=torch.int64),
                "mask_frame": torch.as_tensor(payload["mask_frame"], dtype=torch.int64),
                "mask_label": torch.as_tensor(payload["mask_label"], dtype=torch.int64),
                "mask_is_object_like": torch.as_tensor(payload["mask_is_object_like"], dtype=torch.bool),
                "mask_is_broad": torch.as_tensor(payload["mask_is_broad"], dtype=torch.bool),
                "support_count": torch.as_tensor(payload["support_count"], dtype=torch.int64),
                "feature": torch.as_tensor(payload["mask_feature"], dtype=torch.float16),
                "feature_norm_source_dtype": "float32",
            }
    return {
        "schema_version": f"stream4d_v103_supp_phaseS2_{artifact_kind}_feature_v1",
        "phase_id": PHASE_ID,
        "role_key": role_key,
        "artifact_kind": artifact_kind,
        "sketch_seed": 10317,
        "role_feature_policy": (
            "A_anchor produces positive precision witness features. S_support produces compatibility/coverage support features and is not allowed to trigger merges alone. "
            "S_support_positive_nonveto is recorded diagnostically. V_veto evidence is written to veto_pair_rows and must be applied as risk/cannot-link in S3."
        ),
        "uses_gt": False,
        "uses_future": False,
        "scenes": scenes,
    }


def _scene_metric_row(
    scene: str,
    selected_variant_id: str,
    payloads: dict[str, dict[str, Any]],
    veto_rows: list[dict[str, Any]],
    hard_cannot_link_count: int,
) -> dict[str, Any]:
    a = payloads["A_anchor"]["metrics"]
    s = payloads["S_support_raw"]["metrics"]
    positive = payloads["S_support_positive"]["metrics"]
    as_ = payloads["A_anchor_support"]["metrics"]
    exact_max = max(float(a["exact_vs_sketch_cosine_p95_error"]), float(s["exact_vs_sketch_cosine_p95_error"]), float(as_["exact_vs_sketch_cosine_p95_error"]))
    bucket_p90 = max(float(a["bucket_load_p90"]), float(s["bucket_load_p90"]), float(as_["bucket_load_p90"]))
    collision = max(float(a["bucket_collision_mass_mean"]), float(s["bucket_collision_mass_mean"]), float(as_["bucket_collision_mass_mean"]))
    return {
        "schema_version": "stream4d_v103_supp_phaseS2_feature_metric_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "selected_phaseS1_variant_id": selected_variant_id,
        "mask_count": int(payloads["A_anchor_support"]["mask_frame"].shape[0]),
        "valid_mask_rate_definition": as_["valid_mask_rate_definition"],
        "intervention_mask_count": int(as_["intervention_mask_count"]),
        "A_anchor_valid_mask_rate": a["valid_mask_rate"],
        "S_support_valid_mask_rate": s["valid_mask_rate"],
        "S_support_positive_nonveto_valid_mask_rate": positive["valid_mask_rate"],
        "AS_valid_mask_rate": as_["valid_mask_rate"],
        "A_anchor_valid_mask_rate_all": a["valid_mask_rate_all"],
        "S_support_valid_mask_rate_all": s["valid_mask_rate_all"],
        "S_support_positive_nonveto_valid_mask_rate_all": positive["valid_mask_rate_all"],
        "AS_valid_mask_rate_all": as_["valid_mask_rate_all"],
        "A_anchor_intervention_valid_mask_rate": a["intervention_valid_mask_rate"],
        "S_support_intervention_valid_mask_rate": s["intervention_valid_mask_rate"],
        "S_support_positive_nonveto_intervention_valid_mask_rate": positive["intervention_valid_mask_rate"],
        "AS_intervention_valid_mask_rate": as_["intervention_valid_mask_rate"],
        "A_anchor_object_like_valid_mask_rate": a["object_like_valid_mask_rate"],
        "S_support_object_like_valid_mask_rate": s["object_like_valid_mask_rate"],
        "S_support_positive_nonveto_object_like_valid_mask_rate": positive["object_like_valid_mask_rate"],
        "AS_object_like_valid_mask_rate": as_["object_like_valid_mask_rate"],
        "A_anchor_broad_valid_mask_rate": a["broad_valid_mask_rate"],
        "S_support_broad_valid_mask_rate": s["broad_valid_mask_rate"],
        "S_support_positive_nonveto_broad_valid_mask_rate": positive["broad_valid_mask_rate"],
        "AS_broad_valid_mask_rate": as_["broad_valid_mask_rate"],
        "A_anchor_mask_carrier_count_p10": a["mask_carrier_count_p10"],
        "S_support_mask_carrier_count_p10": s["mask_carrier_count_p10"],
        "S_support_positive_nonveto_mask_carrier_count_p10": positive["mask_carrier_count_p10"],
        "AS_mask_carrier_count_p10": as_["mask_carrier_count_p10"],
        "A_anchor_intervention_mask_carrier_count_p10": a["intervention_mask_carrier_count_p10"],
        "S_support_intervention_mask_carrier_count_p10": s["intervention_mask_carrier_count_p10"],
        "S_support_positive_nonveto_intervention_mask_carrier_count_p10": positive["intervention_mask_carrier_count_p10"],
        "AS_intervention_mask_carrier_count_p10": as_["intervention_mask_carrier_count_p10"],
        "A_anchor_object_like_mask_carrier_count_p10": a["object_like_mask_carrier_count_p10"],
        "S_support_object_like_mask_carrier_count_p10": s["object_like_mask_carrier_count_p10"],
        "S_support_positive_nonveto_object_like_mask_carrier_count_p10": positive["object_like_mask_carrier_count_p10"],
        "AS_object_like_mask_carrier_count_p10": as_["object_like_mask_carrier_count_p10"],
        "mask_level_feature_norm_mean_A": a["mask_level_feature_norm_mean"],
        "mask_level_feature_norm_mean_S": s["mask_level_feature_norm_mean"],
        "mask_level_feature_norm_mean_AS": as_["mask_level_feature_norm_mean"],
        "leave_one_out_self_similarity_drop_mean": min(
            float(a["leave_one_out_self_similarity_drop_mean"]),
            float(s["leave_one_out_self_similarity_drop_mean"]),
            float(as_["leave_one_out_self_similarity_drop_mean"]),
        ),
        "leave_one_out_self_similarity_drop_mean_A": a["leave_one_out_self_similarity_drop_mean"],
        "leave_one_out_self_similarity_drop_mean_S": s["leave_one_out_self_similarity_drop_mean"],
        "leave_one_out_self_similarity_drop_mean_AS": as_["leave_one_out_self_similarity_drop_mean"],
        "exact_vs_sketch_cosine_p95_error": exact_max,
        "exact_vs_sketch_cosine_p95_error_A": a["exact_vs_sketch_cosine_p95_error"],
        "exact_vs_sketch_cosine_p95_error_S": s["exact_vs_sketch_cosine_p95_error"],
        "exact_vs_sketch_cosine_p95_error_AS": as_["exact_vs_sketch_cosine_p95_error"],
        "bucket_load_p90": bucket_p90,
        "bucket_collision_mass_mean": collision,
        "hard_negative_separation_A": a["hard_negative_separation"],
        "hard_negative_separation_S": s["hard_negative_separation"],
        "hard_negative_separation_AS": as_["hard_negative_separation"],
        "veto_mask_pair_count": int(len(veto_rows)),
        "hard_cannot_link_pair_count": int(hard_cannot_link_count),
        "A_anchor_carrier_count": a["carrier_count"],
        "S_support_carrier_count": s["carrier_count"],
        "S_support_positive_nonveto_carrier_count": positive["carrier_count"],
        "AS_carrier_count": as_["carrier_count"],
        "uses_gt": False,
        "uses_future": False,
    }


def _gate_rows_for_scene(scene: str, metric: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    gate_specs = [
        (
            "exact_vs_sketch_cosine_p95_error_le_threshold",
            float(metric["exact_vs_sketch_cosine_p95_error"]) <= float(args.exact_p95_threshold),
            metric["exact_vs_sketch_cosine_p95_error"],
            f"<={args.exact_p95_threshold}",
        ),
        ("AS_valid_mask_rate_ge_0p95", float(metric["AS_valid_mask_rate"]) >= 0.95, metric["AS_valid_mask_rate"], ">=0.95"),
        ("S_support_valid_mask_rate_ge_0p90", float(metric["S_support_valid_mask_rate"]) >= 0.90, metric["S_support_valid_mask_rate"], ">=0.90"),
        (
            "leave_one_out_self_similarity_drop_mean_gt_0",
            float(metric["leave_one_out_self_similarity_drop_mean"]) > 0.0,
            metric["leave_one_out_self_similarity_drop_mean"],
            ">0",
        ),
        (
            "bucket_load_p90_le_budget",
            float(metric["bucket_load_p90"]) <= float(args.bucket_load_p90_budget),
            metric["bucket_load_p90"],
            f"<={args.bucket_load_p90_budget}",
        ),
        (
            "hard_cannot_link_pair_count_gt_0",
            int(metric["hard_cannot_link_pair_count"]) > 0,
            metric["hard_cannot_link_pair_count"],
            ">0",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for gate_name, ok, observed, required in gate_specs:
        rows.append(
            {
                "schema_version": "stream4d_v103_supp_phaseS2_gate_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "gate_name": gate_name,
                "pass": bool(ok),
                "observed": observed,
                "required": required,
                "uses_gt": False,
            }
        )
    return rows


def _gt_diagnostic_rows(scenes: list[str]) -> list[dict[str, Any]]:
    metrics = [
        "same_object_mask_pair_Kgeo_A_mean",
        "same_object_mask_pair_Kgeo_S_mean",
        "same_object_mask_pair_Kgeo_AS_mean",
        "same_semantic_diff_object_Kgeo_A_mean",
        "same_semantic_diff_object_Kgeo_S_mean",
        "same_semantic_diff_object_Kgeo_AS_mean",
        "AUC_A",
        "AUC_S",
        "AUC_AS",
    ]
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        for metric in metrics:
            rows.append(
                {
                    "schema_version": "stream4d_v103_supp_phaseS2_gt_diagnostic_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "metric_name": metric,
                    "status": "not_run",
                    "value": "",
                    "uses_gt_for_gate": False,
                    "note": "GT diagnostic intentionally not used for Phase S2 gates; S2 checks arithmetic, coverage, leave-one-out effect, and veto row construction.",
                }
            )
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    phaseS0_root = _project(args.phaseS0_root)
    phaseS1_root = _project(args.phaseS1_root)
    phaseS0_summary = _read_json(phaseS0_root / "summary.json")
    phaseS1_summary = _read_json(phaseS1_root / "summary.json")
    if phaseS0_summary.get("decision") != "PASS_ENTER_PHASES1_MULTIROLE_CARRIER_SETS":
        raise RuntimeError(f"Phase S0 has not passed: {phaseS0_root / 'summary.json'}")
    if phaseS1_summary.get("decision") != "PASS_ENTER_PHASES2_ROLE_AWARE_AFFINITY":
        raise RuntimeError(f"Phase S1 has not passed: {phaseS1_root / 'summary.json'}")

    output_root.mkdir(parents=True, exist_ok=True)
    role_df = pd.read_parquet(
        phaseS1_root / "carrier_role_rows.parquet",
        columns=["scene_id", "selected_variant_id", "carrier_id", "is_A_anchor", "is_S_support", "is_V_veto"],
    )
    device = _torch_device(str(args.torch_device))

    scene_payloads: dict[str, dict[str, Any]] = {}
    feature_rows: list[dict[str, Any]] = []
    all_veto_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    selected_by_scene: dict[str, str] = {}

    for scene in ["scene0011_00", "scene0050_00"]:
        scene_t0 = time.time()
        scene_out = output_root / scene
        scene_out.mkdir(parents=True, exist_ok=True)
        spec = dict(SCENE_INPUTS[scene])
        spec["phase2_root"] = DEFAULT_PHASE2_ROOT_BY_SCENE[scene]
        diag, _unused_a, _unused_b, arrays = _compute_scene_arrays(scene, spec, scene_out, int(args.cupy_device_id))
        cache_dir, _manifest = _ensure_mmap_cache(spec["phase2_root"])
        batch = _load_cached(cache_dir)
        carrier_id = np.asarray(arrays["carrier_id"], dtype=np.int64)
        role_flags, selected_variant = _role_flags_for_scene(role_df, scene, carrier_id)
        selected_by_scene[scene] = selected_variant

        mask_frame, mask_label, mask_is_object, mask_is_broad, obs_lookup = _mask_observations(diag)
        a_mask = role_flags["A_anchor"] & ~role_flags["V_veto"]
        s_raw_mask = role_flags["S_support_raw"]
        s_positive_mask = role_flags["S_support_raw"] & ~role_flags["V_veto"] & ~a_mask
        as_mask = a_mask | s_raw_mask
        veto_mask = role_flags["V_veto"]

        payloads: dict[str, dict[str, Any]] = {}
        for role_id, role_mask in [
            ("A_anchor", a_mask),
            ("S_support_raw", s_raw_mask),
            ("S_support_positive", s_positive_mask),
            ("A_anchor_support", as_mask),
        ]:
            payload, _metrics = _role_features(
                role_id=role_id,
                diag=diag,
                arrays=arrays,
                batch=batch,
                role_mask=role_mask,
                obs_lookup=obs_lookup,
                mask_frame=mask_frame,
                mask_label=mask_label,
                mask_is_object=mask_is_object,
                mask_is_broad=mask_is_broad,
                args=args,
                device=device,
            )
            payloads[role_id] = payload

        veto_rows, hard_cannot_link_count = _veto_pair_rows(
            scene=scene,
            diag=diag,
            arrays=arrays,
            batch=batch,
            veto_mask=veto_mask,
            obs_lookup=obs_lookup,
            mask_frame=mask_frame,
            mask_label=mask_label,
            mask_is_object=mask_is_object,
            mask_is_broad=mask_is_broad,
            args=args,
        )
        all_veto_rows.extend(veto_rows)
        metric = _scene_metric_row(scene, selected_variant, payloads, veto_rows, hard_cannot_link_count)
        feature_rows.append(metric)
        scene_gate_rows = _gate_rows_for_scene(scene, metric, args)
        gate_rows.extend(scene_gate_rows)
        for gate in scene_gate_rows:
            if not bool(gate["pass"]):
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v103_supp_phaseS2_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "scene_id": scene,
                        "failure_id": gate["gate_name"],
                        "severity": "blocking",
                        "evidence": f"observed={gate['observed']} required={gate['required']}",
                        "repair_direction": (
                            "Follow Phase S2 repair ladder: if exact-vs-sketch error is high, increase sketch dimension or repair incidence; "
                            "if support coverage is low, return to S1 query/role construction; if leave-one-out drop is zero, repair pooling; "
                            "if veto/cannot-link is absent, strengthen V_veto construction before S3."
                        ),
                    }
                )
        scene_payloads[scene] = payloads
        performance_rows.append(
            {
                "schema_version": "stream4d_v103_supp_phaseS2_performance_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "selected_phaseS1_variant_id": selected_variant,
                "runtime_sec": time.time() - scene_t0,
                "projection_backend": diag["performance"].get("projection_backend", ""),
                "semantic_backend": diag["performance"].get("semantic_backend", ""),
                "torch_device": str(device),
                "cupy_device_id": int(args.cupy_device_id),
                "uses_gt": False,
            }
        )

    artifact_paths = {
        "primitive_feature_anchor": output_root / "primitive_feature_anchor.pt",
        "primitive_feature_support": output_root / "primitive_feature_support.pt",
        "mask_feature_anchor": output_root / "mask_feature_anchor.pt",
        "mask_feature_support": output_root / "mask_feature_support.pt",
        "mask_feature_anchor_support": output_root / "mask_feature_anchor_support.pt",
    }
    torch.save(_feature_artifact(scene_payloads, "A_anchor", "primitive"), artifact_paths["primitive_feature_anchor"])
    torch.save(_feature_artifact(scene_payloads, "S_support_raw", "primitive"), artifact_paths["primitive_feature_support"])
    torch.save(_feature_artifact(scene_payloads, "A_anchor", "mask"), artifact_paths["mask_feature_anchor"])
    torch.save(_feature_artifact(scene_payloads, "S_support_raw", "mask"), artifact_paths["mask_feature_support"])
    torch.save(_feature_artifact(scene_payloads, "A_anchor_support", "mask"), artifact_paths["mask_feature_anchor_support"])

    veto_path = output_root / "veto_pair_rows.parquet"
    pd.DataFrame(all_veto_rows).to_parquet(veto_path, index=False)
    feature_metric_path = output_root / "feature_metric_rows.csv"
    gt_diag_path = output_root / "gt_diagnostic_rows.csv"
    gate_path = output_root / "gate_rows.csv"
    failure_path = output_root / "failure_rows.csv"
    performance_path = output_root / "performance_rows.csv"
    artifact_path = output_root / "artifact_rows.csv"
    summary_path = output_root / "summary.json"

    artifact_rows = [
        {
            "schema_version": "stream4d_v103_supp_phaseS2_artifact_row_v1",
            "phase_id": PHASE_ID,
            "artifact_role": name,
            "path": _rel(path),
            "exists": path.exists(),
            "uses_gt": False,
        }
        for name, path in {**artifact_paths, "veto_pair_rows": veto_path}.items()
    ]
    _write_csv(feature_metric_path, feature_rows)
    _write_csv(gt_diag_path, _gt_diagnostic_rows(["scene0011_00", "scene0050_00"]))
    _write_csv(gate_path, gate_rows)
    _write_csv(failure_path, failure_rows)
    _write_csv(performance_path, performance_rows)
    _write_csv(artifact_path, artifact_rows)

    phaseS2_pass = not failure_rows
    decision = "PASS_ENTER_PHASES3_SCAFFOLDED_MASK_GRAPH" if phaseS2_pass else "NO_GO_REPAIR_PHASES2_ROLE_AWARE_AFFINITY"
    summary = {
        "schema_version": "stream4d_v103_supp_phaseS2_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "decision": decision,
        "phaseS2_pass": bool(phaseS2_pass),
        "failure_count": len(failure_rows),
        "selected_phaseS1_variant_by_scene": selected_by_scene,
        "phaseS0_root": _rel(phaseS0_root),
        "phaseS1_root": _rel(phaseS1_root),
        "sketch_dim": int(args.sketch_dim),
        "exact_p95_threshold": float(args.exact_p95_threshold),
        "bucket_load_p90_budget": float(args.bucket_load_p90_budget),
        "feature_policy": {
            "A_anchor": "positive precision witness",
            "S_support": "compatibility coverage feature only; cannot trigger merge without anchor/skeleton support",
            "S_support_positive_nonveto": "diagnostic S_support subset with V_veto carriers removed",
            "V_veto": "risk/cannot-link evidence in veto_pair_rows; applied as veto rather than positive merge evidence",
            "valid_mask_rate_gate_universe": "mask_is_object_like; all/intervention/broad coverage rates are reported as diagnostics",
        },
        "uses_gt_for_gate": False,
        "uses_future": False,
        "runs_AP": False,
        "outputs": {
            "summary": _rel(summary_path),
            "primitive_feature_anchor": _rel(artifact_paths["primitive_feature_anchor"]),
            "primitive_feature_support": _rel(artifact_paths["primitive_feature_support"]),
            "mask_feature_anchor": _rel(artifact_paths["mask_feature_anchor"]),
            "mask_feature_support": _rel(artifact_paths["mask_feature_support"]),
            "mask_feature_anchor_support": _rel(artifact_paths["mask_feature_anchor_support"]),
            "veto_pair_rows": _rel(veto_path),
            "feature_metric_rows": _rel(feature_metric_path),
            "gt_diagnostic_rows": _rel(gt_diag_path),
            "gate_rows": _rel(gate_path),
            "failure_rows": _rel(failure_path),
            "performance_rows": _rel(performance_path),
            "artifact_rows": _rel(artifact_path),
        },
        "truthfulness_note": (
            "Phase S2 does not run AP and does not use GT. It audits role-aware primitive/mask feature arithmetic, "
            "leave-one-out effect, CountSketch approximation, and V_veto cannot-link row construction. "
            "S_support is retained as compatibility/coverage support and cannot trigger merges alone; the non-veto support subset is reported separately. "
            "Hard support/AS coverage gates use object-like masks; broad/risk masks are audited by intervention coverage and veto/cannot-link rows."
        ),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phaseS0-root", default=str(DEFAULT_PHASES0_ROOT))
    parser.add_argument("--phaseS1-root", default=str(DEFAULT_PHASES1_ROOT))
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--sketch-dim", type=int, default=256)
    parser.add_argument("--exact-subset-size", type=int, default=4096)
    parser.add_argument("--exact-p95-threshold", type=float, default=1e-3)
    parser.add_argument("--bucket-load-p90-budget", type=float, default=64.0)
    parser.add_argument("--specificity-mode", default="idf_object_preserve_downweight")
    parser.add_argument("--specificity-alpha", type=float, default=1.0)
    parser.add_argument("--affinity-risk-mode", default="source_and_competing_penalty")
    parser.add_argument("--topk-carriers", type=int, default=64)
    parser.add_argument("--trim-quantile", type=float, default=0.10)
    parser.add_argument("--max-pair-rows", type=int, default=2000)
    parser.add_argument("--max-veto-pair-rows", type=int, default=200000)
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    raise SystemExit(0 if summary["phaseS2_pass"] else 2)


if __name__ == "__main__":
    main()
