#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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

PHASE_ID = "v103_r2_phase3_da3_semsoft_feature_gate"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
DEFAULT_PHASE1_ROOT = AUDIT_ROOT / "v103_r2_phase1_semantic_soft_candidate_universe"
DEFAULT_PHASE2_ROOT = AUDIT_ROOT / "v103_r2_phase2_da3_semsoft_support_alpha_density_topk_reliable_veto_r4_variantid"
SKETCH_SEED = 10323

VARIANTS = [
    {
        "variant_id": "r2p3_v1_da3w050_idf100_broad005",
        "da3_weight": 0.50,
        "baseline_weight": 0.25,
        "idf_alpha": 1.00,
        "broad_quality": 0.05,
    },
    {
        "variant_id": "r2p3_v2_da3w035_idf150_broad003",
        "da3_weight": 0.35,
        "baseline_weight": 0.25,
        "idf_alpha": 1.50,
        "broad_quality": 0.03,
    },
    {
        "variant_id": "r2p3_v3_da3w025_idf200_broad002",
        "da3_weight": 0.25,
        "baseline_weight": 0.25,
        "idf_alpha": 2.00,
        "broad_quality": 0.02,
    },
    {
        "variant_id": "r2p3_v4_da3w010_idf250_broad001_risk030",
        "da3_weight": 0.10,
        "baseline_weight": 0.25,
        "idf_alpha": 2.50,
        "broad_quality": 0.01,
        "max_incidence_risk": 0.30,
    },
    {
        "variant_id": "r2p3_v5_da3w010_idf250_broad001_anchor_or_lowrisk",
        "da3_weight": 0.10,
        "baseline_weight": 0.25,
        "idf_alpha": 2.50,
        "broad_quality": 0.01,
        "max_incidence_risk": 0.35,
        "anchor_or_support_or_lowrisk_only": True,
    },
]


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norm, eps)).astype(np.float32, copy=False)


def _standardize01(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.nanpercentile(arr[finite], 1))
    hi = float(np.nanpercentile(arr[finite], 99))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def _hash(mask_idx: np.ndarray, sketch_dim: int) -> tuple[np.ndarray, np.ndarray]:
    idx = mask_idx.astype(np.int64, copy=False)
    bucket = ((idx * 2654435761 + SKETCH_SEED) % int(sketch_dim)).astype(np.int64)
    sign = np.where(((idx * 1103515245 + SKETCH_SEED) % 2) == 0, 1.0, -1.0).astype(np.float32)
    return bucket, sign


def _rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return 0.5
    values = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size, dtype=np.int8), np.zeros(neg.size, dtype=np.int8)])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    pos_rank_sum = float(np.sum(ranks[labels == 1]))
    return float((pos_rank_sum - pos.size * (pos.size + 1) / 2.0) / max(pos.size * neg.size, 1))


def _baseline_feature(candidates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = candidates["A_anchor_support_count"].to_numpy(dtype=np.float32)
    s = candidates["S_support_count"].to_numpy(dtype=np.float32)
    cols = [
        _standardize01(np.log1p(np.maximum(a, 0.0))),
        _standardize01(np.log1p(np.maximum(s, 0.0))),
        (a > 0).astype(np.float32),
        (s > 0).astype(np.float32),
    ]
    feat = np.stack(cols, axis=1).astype(np.float32)
    valid = (a > 0) | (s > 0)
    feat[~valid] = 0.0
    mass = np.log1p(np.maximum(a, 0.0)) + 0.5 * np.log1p(np.maximum(s, 0.0))
    return _normalize_rows(feat), valid.astype(bool), mass.astype(np.float32)


def _mask_quality(candidates: pd.DataFrame, variant: dict[str, Any]) -> np.ndarray:
    risk = candidates["risk_score"].to_numpy(dtype=np.float32)
    broad = candidates["semantic_broad_flag"].map(_as_bool).to_numpy(dtype=bool)
    extra = candidates["candidate_delta_type"].astype(str).eq("extra_over_semhard").to_numpy(dtype=bool)
    quality = np.clip(1.0 - risk, 0.02, 1.0).astype(np.float32)
    quality[broad] *= float(variant["broad_quality"])
    quality[extra] *= 0.85
    return quality.astype(np.float32, copy=False)


def _component_matrix(
    candidates: pd.DataFrame,
    incidence: pd.DataFrame,
    variant: dict[str, Any],
    *,
    shuffle_masks: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    obs_to_idx = {str(obs): i for i, obs in enumerate(candidates["mask_observation_id"].astype(str).tolist())}
    component_ids = sorted({int(v) for v in incidence["component_id"].tolist()})
    comp_to_idx = {cid: i for i, cid in enumerate(component_ids)}
    matrix = np.zeros((len(component_ids), len(candidates)), dtype=np.float32)
    quality = _mask_quality(candidates, variant)
    rng = np.random.default_rng(SKETCH_SEED)
    mask_perm = rng.permutation(len(candidates)) if shuffle_masks else np.arange(len(candidates), dtype=np.int64)
    for row in incidence.to_dict("records"):
        obs_idx = obs_to_idx.get(str(row["mask_observation_id"]))
        if obs_idx is None:
            continue
        target_idx = int(mask_perm[obs_idx])
        comp_idx = comp_to_idx[int(row["component_id"])]
        count = float(row.get("component_mask_gaussian_count", 0.0) or 0.0)
        risk = float(row.get("risk_score", candidates.iloc[obs_idx].get("risk_score", 0.0)) or 0.0)
        value = float(variant["da3_weight"]) * math.sqrt(max(count, 1.0)) * max(1.0 - risk, 0.02) * float(quality[obs_idx])
        if np.isfinite(value) and value > 0.0:
            matrix[comp_idx, target_idx] += value
    return matrix, np.asarray(component_ids, dtype=np.int64), comp_to_idx


def _primitive_sketch_raw(matrix: np.ndarray, sketch_dim: int) -> np.ndarray:
    comp_count, mask_count = matrix.shape
    out = np.zeros((comp_count, int(sketch_dim)), dtype=np.float32)
    mask_idx = np.arange(mask_count, dtype=np.int64)
    bucket, sign = _hash(mask_idx, int(sketch_dim))
    for mi in range(mask_count):
        vals = matrix[:, mi]
        nz = vals > 0
        if np.any(nz):
            out[nz, bucket[mi]] += sign[mi] * vals[nz]
    return out.astype(np.float32, copy=False)


def _primitive_sketch(matrix: np.ndarray, sketch_dim: int) -> np.ndarray:
    return _normalize_rows(_primitive_sketch_raw(matrix, sketch_dim))


def _mask_feature_from_exact_primitives(matrix: np.ndarray) -> np.ndarray:
    comp_count, mask_count = matrix.shape
    out = np.zeros((mask_count, mask_count), dtype=np.float32)
    for comp_idx in range(comp_count):
        raw = matrix[comp_idx]
        masks = np.flatnonzero(raw > 0)
        if masks.size == 0:
            continue
        for mi in masks.tolist():
            contrib = raw.copy()
            contrib[mi] = 0.0
            norm = float(np.linalg.norm(contrib))
            if norm <= 1e-12:
                continue
            out[mi] += raw[mi] * (contrib / norm)
    return _normalize_rows(out)


def _mask_feature_from_sketch_primitives(matrix: np.ndarray, primitive_sketch_raw: np.ndarray, *, sketch_dim: int) -> np.ndarray:
    comp_count, mask_count = matrix.shape
    out_dim = primitive_sketch_raw.shape[1]
    out = np.zeros((mask_count, out_dim), dtype=np.float32)
    mask_idx = np.arange(mask_count, dtype=np.int64)
    bucket, sign = _hash(mask_idx, int(sketch_dim))
    for comp_idx in range(comp_count):
        masks = np.flatnonzero(matrix[comp_idx] > 0)
        if masks.size == 0:
            continue
        for mi in masks.tolist():
            contrib = primitive_sketch_raw[comp_idx].copy()
            contrib[bucket[mi]] -= sign[mi] * matrix[comp_idx, mi]
            norm = float(np.linalg.norm(contrib))
            if norm <= 1e-12:
                continue
            out[mi] += matrix[comp_idx, mi] * (contrib / norm)
    return _normalize_rows(out)


def _pair_sets(candidates: pd.DataFrame, incidence: pd.DataFrame, max_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    obs_to_idx = {str(obs): i for i, obs in enumerate(candidates["mask_observation_id"].astype(str).tolist())}
    positives: list[tuple[int, int]] = []
    for _component, rows in incidence.groupby("component_id"):
        idx = sorted({obs_to_idx[str(obs)] for obs in rows["mask_observation_id"].astype(str).tolist() if str(obs) in obs_to_idx})
        if len(idx) < 2:
            continue
        limit = min(len(idx), 32)
        for i in range(limit):
            for j in range(i + 1, limit):
                positives.append((idx[i], idx[j]))
                if len(positives) >= max_pairs:
                    break
            if len(positives) >= max_pairs:
                break
        if len(positives) >= max_pairs:
            break

    component_sets: dict[int, set[int]] = {}
    for _component, rows in incidence.groupby("component_id"):
        idx = {obs_to_idx[str(obs)] for obs in rows["mask_observation_id"].astype(str).tolist() if str(obs) in obs_to_idx}
        for mi in idx:
            component_sets.setdefault(mi, set()).update(idx)

    negatives: list[tuple[int, int]] = []
    for _frame, rows in candidates.groupby("frame_local_index"):
        idx = rows.index.to_numpy(dtype=np.int64)
        if idx.size < 2:
            continue
        for apos in range(idx.size):
            a = int(idx[apos])
            for bpos in range(apos + 1, idx.size):
                b = int(idx[bpos])
                if b in component_sets.get(a, set()):
                    continue
                negatives.append((a, b))
                if len(negatives) >= max_pairs:
                    break
            if len(negatives) >= max_pairs:
                break
        if len(negatives) >= max_pairs:
            break
    return np.asarray(positives, dtype=np.int64), np.asarray(negatives, dtype=np.int64)


def _pair_stats(feature: np.ndarray, pos_pairs: np.ndarray, neg_pairs: np.ndarray) -> dict[str, float]:
    def sims(pairs: np.ndarray) -> np.ndarray:
        if pairs.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.sum(feature[pairs[:, 0]] * feature[pairs[:, 1]], axis=1).astype(np.float32)

    pos = sims(pos_pairs)
    neg = sims(neg_pairs)
    pos_mean = float(np.mean(pos)) if pos.size else 0.0
    neg_mean = float(np.mean(neg)) if neg.size else 0.0
    return {
        "object_like_pair_similarity_mean": pos_mean,
        "same_frame_competing_similarity_mean": neg_mean,
        "hard_negative_separation_margin": pos_mean - neg_mean,
        "hard_negative_separation_auc": _rank_auc(pos, neg),
        "positive_pair_count": int(pos.size),
        "hard_negative_pair_count": int(neg.size),
    }


def _scene_variant(
    scene_id: str,
    candidates_all: pd.DataFrame,
    incidence_all: pd.DataFrame,
    best_variant_id: str,
    variant: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidates = candidates_all[candidates_all["scene_id"].astype(str) == scene_id].copy().reset_index(drop=True)
    incidence = incidence_all[
        (incidence_all["scene_id"].astype(str) == scene_id)
        & (incidence_all["variant_id"].astype(str) == best_variant_id)
    ].copy()
    if "max_incidence_risk" in variant:
        incidence = incidence[incidence["risk_score"].astype(float) <= float(variant["max_incidence_risk"])].copy()
    if bool(variant.get("anchor_or_support_or_lowrisk_only", False)) and len(incidence):
        keep = (
            incidence["A_anchor_hit"].map(_as_bool)
            | (incidence["S_support_count"].astype(float) > 0)
            | (incidence["risk_score"].astype(float) <= float(variant.get("max_incidence_risk", 0.35)))
        )
        incidence = incidence[keep].copy()
    baseline, baseline_valid, baseline_mass = _baseline_feature(candidates)
    matrix, component_ids, _comp_to_idx = _component_matrix(candidates, incidence, variant, shuffle_masks=False)
    shuffled_matrix, _unused_component_ids, _unused = _component_matrix(candidates, incidence, variant, shuffle_masks=True)

    primitive_exact = _normalize_rows(matrix)
    primitive_sketch_raw = _primitive_sketch_raw(matrix, int(args.sketch_dim))
    primitive_sketch = _normalize_rows(primitive_sketch_raw)
    mask_da3_exact = _mask_feature_from_exact_primitives(matrix)
    mask_da3_sketch = _mask_feature_from_sketch_primitives(matrix, primitive_sketch_raw, sketch_dim=int(args.sketch_dim))
    shuffled_primitive_raw = _primitive_sketch_raw(shuffled_matrix, int(args.sketch_dim))
    shuffled_primitive = _normalize_rows(shuffled_primitive_raw)
    mask_da3_shuffled = _mask_feature_from_sketch_primitives(shuffled_matrix, shuffled_primitive_raw, sketch_dim=int(args.sketch_dim))

    base_weight = float(variant["baseline_weight"])
    mask_exact = _normalize_rows(np.concatenate([baseline * base_weight, mask_da3_exact], axis=1))
    mask_sketch = _normalize_rows(np.concatenate([baseline * base_weight, mask_da3_sketch], axis=1))
    mask_shuffled = _normalize_rows(np.concatenate([baseline * base_weight, mask_da3_shuffled], axis=1))

    diff = np.abs(mask_exact @ mask_exact.T - mask_sketch @ mask_sketch.T)
    exact_vs_sketch_p95 = float(np.percentile(diff.reshape(-1), 95)) if diff.size else 0.0
    exact_vs_sketch_max = float(np.max(diff)) if diff.size else 0.0

    valid = np.linalg.norm(mask_sketch, axis=1) > 0.0
    semantic_broad = candidates["semantic_broad_flag"].map(_as_bool).to_numpy(dtype=bool)
    extra = candidates["candidate_delta_type"].astype(str).eq("extra_over_semhard").to_numpy(dtype=bool)
    boundary_values = candidates["semantic_boundary_variance"].to_numpy(dtype=np.float32)
    boundary = boundary_values >= float(np.nanmedian(boundary_values)) if boundary_values.size else np.zeros((0,), dtype=bool)
    da3_mass_by_mask = np.sum(matrix, axis=0).astype(np.float32)
    total_da3_mass = float(np.sum(da3_mass_by_mask))
    total_baseline_mass = float(np.sum(baseline_mass))
    baseline_broad_ratio = float(np.sum(baseline_mass[semantic_broad]) / max(total_baseline_mass, 1e-12))
    da3_broad_ratio = float(np.sum(da3_mass_by_mask[semantic_broad]) / max(total_da3_mass, 1e-12))
    extended_broad_ratio = float((np.sum(baseline_mass[semantic_broad]) + np.sum(da3_mass_by_mask[semantic_broad])) / max(total_baseline_mass + total_da3_mass, 1e-12))
    semantic_soft_ratio = float(np.sum(da3_mass_by_mask[extra]) / max(total_da3_mass, 1e-12))
    baseline_object_cov = float(np.mean(baseline_valid)) if baseline_valid.size else 0.0
    extended_object_cov = float(np.mean(valid)) if valid.size else 0.0
    boundary_delta = float(np.mean(valid[boundary]) - np.mean(baseline_valid[boundary])) if np.any(boundary) else 0.0

    pos_pairs, neg_pairs = _pair_sets(candidates, incidence, int(args.max_pair_count))
    real_stats = _pair_stats(mask_sketch, pos_pairs, neg_pairs)
    baseline_stats = _pair_stats(_normalize_rows(baseline), pos_pairs, neg_pairs)
    shuffled_stats = _pair_stats(mask_shuffled, pos_pairs, neg_pairs)
    real_minus_shuffled = float(real_stats["hard_negative_separation_margin"] - shuffled_stats["hard_negative_separation_margin"])

    row = {
        "schema_version": "stream4d_v103_r2_phase3_metric_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "variant_id": variant["variant_id"],
        "phase2_best_variant_id": best_variant_id,
        "mask_feature_valid_rate": float(np.mean(valid)) if valid.size else 0.0,
        "baseline_mask_feature_valid_rate": float(np.mean(baseline_valid)) if baseline_valid.size else 0.0,
        "object_like_mask_coverage_delta": extended_object_cov - baseline_object_cov,
        "boundary_band_coverage_delta": boundary_delta,
        "broad_contribution_ratio": extended_broad_ratio,
        "previous_broad_contribution_ratio": baseline_broad_ratio,
        "da3_broad_contribution_ratio": da3_broad_ratio,
        "semantic_soft_contribution_ratio": semantic_soft_ratio,
        "exact_vs_sketch_cosine_p95_error": exact_vs_sketch_p95,
        "exact_vs_sketch_cosine_max_error": exact_vs_sketch_max,
        "hard_negative_separation_auc": real_stats["hard_negative_separation_auc"],
        "previous_hard_negative_separation_auc": baseline_stats["hard_negative_separation_auc"],
        "hard_negative_separation_auc_delta": real_stats["hard_negative_separation_auc"] - baseline_stats["hard_negative_separation_auc"],
        "same_frame_competing_similarity_mean": real_stats["same_frame_competing_similarity_mean"],
        "object_like_pair_similarity_mean": real_stats["object_like_pair_similarity_mean"],
        "real_hard_negative_separation_margin": real_stats["hard_negative_separation_margin"],
        "shuffled_DA3_support_control_similarity": shuffled_stats["hard_negative_separation_margin"],
        "real_minus_shuffled_DA3_support_separation": real_minus_shuffled,
        "positive_pair_count": real_stats["positive_pair_count"],
        "hard_negative_pair_count": real_stats["hard_negative_pair_count"],
        "component_count": int(matrix.shape[0]),
        "candidate_mask_count": int(matrix.shape[1]),
        "da3_supported_mask_count": int(np.count_nonzero(da3_mass_by_mask > 0)),
        "uses_gt_for_selection": False,
        "uses_future": False,
    }
    tensors = {
        "component_ids": component_ids,
        "mask_observation_id": candidates["mask_observation_id"].astype(str).tolist(),
        "primitive_feature": primitive_sketch.astype(np.float32),
        "mask_feature": mask_sketch.astype(np.float32),
        "baseline_feature": baseline.astype(np.float32),
    }
    return row, tensors, {"real": real_stats, "baseline": baseline_stats, "shuffled": shuffled_stats}


def _gate(gate_id: str, passed: bool, observed: Any, required: Any, scene_id: str = "ALL", repair: str = "") -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_r2_phase3_gate_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "gate_id": gate_id,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "repair_direction": repair,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    phase1_root = _project(args.phase1_root)
    phase2_root = _project(args.phase2_root)
    candidates = pd.read_csv(phase1_root / "candidate_universe_rows.csv")
    incidence = pd.read_parquet(phase2_root / "da3_semsoft_primitive_incidence_rows.parquet")
    casebook = pd.read_csv(phase2_root / "da3_semsoft_casebook_rows.csv")
    best_by_scene = {str(row["scene_id"]): str(row["variant_id"]) for row in casebook.to_dict("records")}

    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    tensor_payload = {
        "schema_version": "stream4d_v103_r2_phase3_role_extended_feature_v1",
        "phase_id": PHASE_ID,
        "phase1_root": _rel(phase1_root),
        "phase2_root": _rel(phase2_root),
        "sketch_dim": int(args.sketch_dim),
        "sketch_seed": SKETCH_SEED,
        "scenes": {},
        "uses_gt_for_selection": False,
        "uses_future": False,
    }
    for variant in VARIANTS:
        for scene in ["scene0011_00", "scene0050_00"]:
            metric, tensors, pair_stats = _scene_variant(scene, candidates, incidence, best_by_scene[scene], variant, args)
            metric_rows.append(metric)
            if str(variant["variant_id"]) == str(VARIANTS[0]["variant_id"]):
                tensor_payload["scenes"][scene] = tensors
            for control_id, stats in pair_stats.items():
                pair_rows.append({
                    "schema_version": "stream4d_v103_r2_phase3_hard_negative_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "variant_id": variant["variant_id"],
                    "control_id": control_id,
                    **stats,
                })

    best_rows: list[dict[str, Any]] = []
    for scene in ["scene0011_00", "scene0050_00"]:
        rows = [row for row in metric_rows if row["scene_id"] == scene]
        best_rows.append(max(rows, key=lambda row: (
            float(row["hard_negative_separation_auc_delta"]),
            float(row["real_minus_shuffled_DA3_support_separation"]),
            float(row["mask_feature_valid_rate"]),
        )))

    gate_rows: list[dict[str, Any]] = []
    for row in best_rows:
        scene = str(row["scene_id"])
        gate_rows.append(_gate(f"{scene}_mask_feature_valid_rate_ge_0p95", row["mask_feature_valid_rate"] >= 0.95, row["mask_feature_valid_rate"], ">= 0.95", scene, "Use current Phase4/Phase5 baseline mask feature as base, or restrict R2-3 universe to supported masks."))
        gate_rows.append(_gate(f"{scene}_object_like_mask_coverage_delta_gt_0", row["object_like_mask_coverage_delta"] > 0.0, row["object_like_mask_coverage_delta"], "> 0", scene, "Increase DA3 support only through high-quality per-mask top-K, not broad relaxation."))
        gate_rows.append(_gate(f"{scene}_boundary_band_coverage_delta_ge_0", row["boundary_band_coverage_delta"] >= 0.0, row["boundary_band_coverage_delta"], ">= 0", scene, "Add stronger IDF or anchor-near-only support if boundary coverage regresses."))
        gate_rows.append(_gate(f"{scene}_broad_contribution_ratio_le_previous_plus_0p03", row["broad_contribution_ratio"] <= row["previous_broad_contribution_ratio"] + 0.03, row["broad_contribution_ratio"], f"<= {row['previous_broad_contribution_ratio'] + 0.03}", scene, "Lower DA3 weight or increase semantic-soft broad downweight."))
        gate_rows.append(_gate(f"{scene}_exact_vs_sketch_cosine_p95_error_le_0p002", row["exact_vs_sketch_cosine_p95_error"] <= 0.002, row["exact_vs_sketch_cosine_p95_error"], "<= 0.002", scene, "Increase sketch_dim before trusting feature separation."))
        gate_rows.append(_gate(f"{scene}_hard_negative_separation_auc_ge_previous_plus_0p02", row["hard_negative_separation_auc"] >= row["previous_hard_negative_separation_auc"] + 0.02, row["hard_negative_separation_auc"], f">= {row['previous_hard_negative_separation_auc'] + 0.02}", scene, "Reduce DA3 weight or use anchor-near-only support if separation is control-like."))
        gate_rows.append(_gate(f"{scene}_real_minus_shuffled_DA3_support_separation_ge_0p01", row["real_minus_shuffled_DA3_support_separation"] >= 0.01, row["real_minus_shuffled_DA3_support_separation"], ">= 0.01", scene, "Mark DA3_SEMSOFT_FEATURE_NOT_OBJECT_SPECIFIC if shuffled control is too close."))
    gate_rows.append(_gate("uses_gt_for_selection_false", True, False, "False"))

    failure_rows = [
        {
            "schema_version": "stream4d_v103_r2_phase3_failure_row_v1",
            "phase_id": PHASE_ID,
            "failure_id": row["gate_id"],
            "scene_id": row["scene_id"],
            "severity": "blocker",
            "observed": row["observed"],
            "expected": row["required"],
            "repair_direction": row["repair_direction"],
        }
        for row in gate_rows
        if not row["pass"]
    ]

    primitive_path = out / "role_extended_primitive_feature.pt"
    mask_path = out / "role_extended_mask_feature.pt"
    torch.save({**tensor_payload, "feature_kind": "primitive"}, primitive_path)
    torch.save({**tensor_payload, "feature_kind": "mask"}, mask_path)
    _write_csv(out / "mask_feature_metric_rows.csv", metric_rows)
    _write_csv(out / "hard_negative_separation_rows.csv", pair_rows)
    _write_csv(out / "coverage_delta_rows.csv", best_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    (out / "last_command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")

    summary = {
        "schema_version": "stream4d_v103_r2_phase3_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_R2_3_DA3_SEMSOFT_FEATURE_GATE" if not failure_rows else "NO_GO_R2_3_DA3_SEMSOFT_FEATURE_GATE",
        "phase1_root": _rel(phase1_root),
        "phase2_root": _rel(phase2_root),
        "variant_count": len(VARIANTS),
        "best_by_scene": {row["scene_id"]: row for row in best_rows},
        "failure_count": len(failure_rows),
        "uses_gt_for_selection": False,
        "uses_future": False,
        "truthfulness_note": "R2-3 builds a GT-free support-only DA3-semsoft feature gate. It does not run AP and does not treat semantic features as primitive support.",
        "outputs": {
            "role_extended_primitive_feature": _rel(primitive_path),
            "role_extended_mask_feature": _rel(mask_path),
            "mask_feature_metric_rows": _rel(out / "mask_feature_metric_rows.csv"),
            "hard_negative_separation_rows": _rel(out / "hard_negative_separation_rows.csv"),
            "coverage_delta_rows": _rel(out / "coverage_delta_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "summary": _rel(out / "summary.json"),
        },
    }
    _write_json(out / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v103 R2-3 DA3-semsoft feature/control gate.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase1-root", default=str(DEFAULT_PHASE1_ROOT))
    parser.add_argument("--phase2-root", default=str(DEFAULT_PHASE2_ROOT))
    parser.add_argument("--sketch-dim", type=int, default=8192)
    parser.add_argument("--max-pair-count", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    summary = build(parse_args())
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["decision"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
