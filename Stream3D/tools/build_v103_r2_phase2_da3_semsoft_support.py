#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.spatial import cKDTree


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_r2_phase2_da3_semsoft_support"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
DEFAULT_PHASE1_ROOT = AUDIT_ROOT / "v103_r2_phase1_semantic_soft_candidate_universe"
DEFAULT_PHASE9B_CACHE_ROOT = AUDIT_ROOT / "v103_phase9b_da3_c0001_provider_readiness_all_r2_cache"
DEFAULT_PHASE9A_ROOT = AUDIT_ROOT / "v103_phase9a_da3_c0001_provider_export"

VARIANTS = [
    {
        "variant_id": "r2v1_topk4096_knn8_q60_seed1_veto025_risk080_minc4",
        "knn_k": 8,
        "edge_radius_quantile": 0.60,
        "edge_radius_factor": 1.5,
        "max_universe_gaussians": 250000,
        "per_mask_topk_high_quality": 4096,
        "min_alpha_hit_quantile": 0.60,
        "min_density_hit_quantile": 0.60,
        "min_component_size": 4,
        "min_seed_gaussians": 1,
        "min_object_like_footprint_count": 3,
        "max_veto_footprint_rate": 0.25,
        "reliable_veto_support_quantile": 0.95,
        "reliable_veto_support_ratio": 6.0,
        "max_candidate_risk_mean": 0.80,
        "max_area_broad_rate": 0.10,
    },
    {
        "variant_id": "r2v2_topk2048_knn8_q50_seed1_veto025_risk080_minc4",
        "knn_k": 8,
        "edge_radius_quantile": 0.50,
        "edge_radius_factor": 1.3,
        "max_universe_gaussians": 250000,
        "per_mask_topk_high_quality": 2048,
        "min_alpha_hit_quantile": 0.60,
        "min_density_hit_quantile": 0.60,
        "min_component_size": 4,
        "min_seed_gaussians": 1,
        "min_object_like_footprint_count": 3,
        "max_veto_footprint_rate": 0.25,
        "reliable_veto_support_quantile": 0.95,
        "reliable_veto_support_ratio": 6.0,
        "max_candidate_risk_mean": 0.80,
        "max_area_broad_rate": 0.10,
    },
    {
        "variant_id": "r2v3_topk8192_knn12_q60_seed1_veto025_risk080_minc4",
        "knn_k": 12,
        "edge_radius_quantile": 0.60,
        "edge_radius_factor": 1.3,
        "max_universe_gaussians": 250000,
        "per_mask_topk_high_quality": 8192,
        "min_alpha_hit_quantile": 0.60,
        "min_density_hit_quantile": 0.60,
        "min_component_size": 4,
        "min_seed_gaussians": 1,
        "min_object_like_footprint_count": 3,
        "max_veto_footprint_rate": 0.25,
        "reliable_veto_support_quantile": 0.95,
        "reliable_veto_support_ratio": 6.0,
        "max_candidate_risk_mean": 0.80,
        "max_area_broad_rate": 0.10,
    },
    {
        "variant_id": "r2v4_topk150k_knn12_q75_veto010_risk075",
        "knn_k": 12,
        "edge_radius_quantile": 0.75,
        "edge_radius_factor": 2.0,
        "max_universe_gaussians": 150000,
        "per_mask_topk_high_quality": 0,
        "min_alpha_hit_quantile": 0.70,
        "min_density_hit_quantile": 0.80,
        "min_component_size": 8,
        "min_seed_gaussians": 1,
        "min_object_like_footprint_count": 3,
        "max_veto_footprint_rate": 0.10,
        "reliable_veto_support_quantile": 0.90,
        "reliable_veto_support_ratio": 4.0,
        "max_candidate_risk_mean": 0.75,
        "max_area_broad_rate": 0.10,
    },
    {
        "variant_id": "r2v5_emit_nonbroad_risk055_topk2048_knn8_q50_seed1_veto025",
        "knn_k": 8,
        "edge_radius_quantile": 0.50,
        "edge_radius_factor": 1.3,
        "max_universe_gaussians": 250000,
        "per_mask_topk_high_quality": 2048,
        "min_alpha_hit_quantile": 0.60,
        "min_density_hit_quantile": 0.60,
        "min_component_size": 4,
        "min_seed_gaussians": 1,
        "min_object_like_footprint_count": 3,
        "max_veto_footprint_rate": 0.25,
        "reliable_veto_support_quantile": 0.95,
        "reliable_veto_support_ratio": 6.0,
        "max_candidate_risk_mean": 0.80,
        "max_area_broad_rate": 0.10,
        "emit_support_policy_id": "emit_nonbroad_lowrisk_footprint_only",
        "emit_max_area_ratio": 0.12,
        "emit_max_risk": 0.55,
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x.astype(np.float32, copy=False), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _robust01(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.nanquantile(arr[finite], 0.01))
    hi = float(np.nanquantile(arr[finite], 0.99))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def _scene_phase9a_dir(phase9a_root: Path, scene_id: str) -> Path:
    matches = sorted(phase9a_root.glob(f"{scene_id}_chunk32_start029_process252"))
    if not matches:
        matches = sorted(phase9a_root.glob(f"{scene_id}_chunk*"))
    if not matches:
        raise FileNotFoundError(f"Missing DA3 phase9a export directory for {scene_id} under {phase9a_root}")
    return matches[0]


@lru_cache(maxsize=8)
def _load_gaussian_quality_cached(phase9a_root_str: str, scene_id: str) -> dict[str, Any]:
    phase9a_root = _project(phase9a_root_str)
    ply_path = _scene_phase9a_dir(phase9a_root, scene_id) / "gs_ply" / "0000.ply"
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or [])
    if "alpha" in names:
        alpha = np.asarray(vertex["alpha"], dtype=np.float32)
        alpha_source = "ply_alpha"
    elif "opacity" in names:
        alpha = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
        alpha_source = "sigmoid_ply_opacity"
    else:
        raise ValueError(f"{ply_path} has no alpha or opacity property; cannot apply required 3DGS alpha gate")

    if "density" in names:
        density = np.asarray(vertex["density"], dtype=np.float32)
        density_log = np.log(np.maximum(density.astype(np.float64), 1e-30)).astype(np.float32)
        density_source = "ply_density"
    elif {"scale_0", "scale_1", "scale_2"}.issubset(names):
        scale_sum = (
            np.asarray(vertex["scale_0"], dtype=np.float32)
            + np.asarray(vertex["scale_1"], dtype=np.float32)
            + np.asarray(vertex["scale_2"], dtype=np.float32)
        )
        density_log = (np.log(np.maximum(alpha.astype(np.float64), 1e-30)) - scale_sum.astype(np.float64)).astype(np.float32)
        density = np.exp(np.clip(density_log.astype(np.float64), -60.0, 60.0)).astype(np.float32)
        density_source = "alpha_div_exp_scale_sum_proxy"
    else:
        raise ValueError(f"{ply_path} has no density or scale_0/1/2 properties; cannot apply required 3DGS density gate")

    quality_score = (0.5 * _robust01(alpha) + 0.5 * _robust01(density_log)).astype(np.float32)
    return {
        "ply_path": ply_path,
        "alpha": alpha.astype(np.float32, copy=False),
        "density": density.astype(np.float32, copy=False),
        "density_log": density_log.astype(np.float32, copy=False),
        "quality_score": quality_score,
        "alpha_source": alpha_source,
        "density_source": density_source,
        "property_names": sorted(names),
    }


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = int(self.parent[x])
        while p != int(self.parent[p]):
            self.parent[p] = self.parent[self.parent[p]]
            p = int(self.parent[p])
        return p

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _candidate_tables(phase1_root: Path, scene_id: str) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    df = pd.read_csv(phase1_root / "candidate_universe_rows.csv")
    df = df[df["scene_id"].astype(str) == scene_id].copy()
    meta = {str(row["mask_observation_id"]): row for row in df.to_dict("records")}
    return df, meta


@lru_cache(maxsize=8)
def _support_counts_cached(mask_by_frame_path: str, phase1_root_str: str, scene_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase1_root = _project(phase1_root_str)
    candidates, _ = _candidate_tables(phase1_root, scene_id)
    mask_by_frame = np.load(mask_by_frame_path, mmap_mode="r")
    candidate_by_frame: dict[int, set[int]] = defaultdict(set)
    seed_by_frame: dict[int, set[int]] = defaultdict(set)
    veto_by_frame: dict[int, set[int]] = defaultdict(set)
    for row in candidates.to_dict("records"):
        fi = int(row["frame_local_index"])
        mid = int(row["mask_id"])
        candidate_by_frame[fi].add(mid)
        if _as_bool(row.get("A_anchor_hit", False)):
            seed_by_frame[fi].add(mid)
        if _as_bool(row.get("V_veto_hit", False)):
            veto_by_frame[fi].add(mid)

    hit_count = np.zeros(mask_by_frame.shape[1], dtype=np.uint8)
    seed_count = np.zeros(mask_by_frame.shape[1], dtype=np.uint8)
    veto_count = np.zeros(mask_by_frame.shape[1], dtype=np.uint8)
    for fi in range(mask_by_frame.shape[0]):
        labels = np.asarray(mask_by_frame[fi], dtype=np.uint16)
        if candidate_by_frame.get(fi):
            hit_count += np.isin(labels, list(candidate_by_frame[fi])).astype(np.uint8)
        if seed_by_frame.get(fi):
            seed_count += np.isin(labels, list(seed_by_frame[fi])).astype(np.uint8)
        if veto_by_frame.get(fi):
            veto_count += np.isin(labels, list(veto_by_frame[fi])).astype(np.uint8)
    return hit_count, seed_count, veto_count


def _select_universe(
    mask_by_frame: np.ndarray,
    candidates: pd.DataFrame,
    hit_count: np.ndarray,
    seed_count: np.ndarray,
    veto_count: np.ndarray,
    quality: dict[str, Any],
    variant: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    idx = np.flatnonzero(hit_count > 0).astype(np.int64)
    raw_count = int(idx.shape[0])
    if idx.size == 0:
        topk = int(variant.get("per_mask_topk_high_quality", 0))
        return idx, seed_count[idx], veto_count[idx], {
            "raw_hit_gaussian_count": 0,
            "quality_filter_kept_count": 0,
            "quality_filter_after_cap_count": 0,
            "per_mask_topk_high_quality": topk,
            "per_mask_topk_input_count": 0,
            "per_mask_topk_output_count": 0,
            "per_mask_topk_mask_observation_count": 0,
            "per_mask_topk_truncated_mask_observation_count": 0,
            "alpha_threshold": None,
            "density_log_threshold": None,
            "density_threshold": None,
            "alpha_quantile": float(variant["min_alpha_hit_quantile"]),
            "density_quantile": float(variant["min_density_hit_quantile"]),
        }

    alpha = np.asarray(quality["alpha"], dtype=np.float32)
    density_log = np.asarray(quality["density_log"], dtype=np.float32)
    quality_score = np.asarray(quality["quality_score"], dtype=np.float32)
    finite = np.isfinite(alpha[idx]) & np.isfinite(density_log[idx]) & np.isfinite(quality_score[idx])
    if not np.any(finite):
        idx = np.zeros((0,), dtype=np.int64)
        topk = int(variant.get("per_mask_topk_high_quality", 0))
        return idx, seed_count[idx], veto_count[idx], {
            "raw_hit_gaussian_count": raw_count,
            "quality_filter_kept_count": 0,
            "quality_filter_after_cap_count": 0,
            "per_mask_topk_high_quality": topk,
            "per_mask_topk_input_count": 0,
            "per_mask_topk_output_count": 0,
            "per_mask_topk_mask_observation_count": 0,
            "per_mask_topk_truncated_mask_observation_count": 0,
            "alpha_threshold": None,
            "density_log_threshold": None,
            "density_threshold": None,
            "alpha_quantile": float(variant["min_alpha_hit_quantile"]),
            "density_quantile": float(variant["min_density_hit_quantile"]),
        }

    finite_idx = idx[finite]
    alpha_threshold = float(np.quantile(alpha[finite_idx], float(variant["min_alpha_hit_quantile"])))
    density_log_threshold = float(np.quantile(density_log[finite_idx], float(variant["min_density_hit_quantile"])))
    keep = (alpha[finite_idx] >= alpha_threshold) & (density_log[finite_idx] >= density_log_threshold)
    idx = finite_idx[keep].astype(np.int64, copy=False)
    kept_count = int(idx.shape[0])
    per_mask_topk = int(variant.get("per_mask_topk_high_quality", 0))
    topk_meta: dict[str, Any] = {
        "per_mask_topk_high_quality": per_mask_topk,
        "per_mask_topk_input_count": int(idx.shape[0]),
        "per_mask_topk_output_count": int(idx.shape[0]),
    }
    if per_mask_topk > 0 and idx.size:
        idx, topk_meta = _apply_per_mask_topk(mask_by_frame, candidates, idx, quality_score, per_mask_topk)

    if idx.shape[0] > int(variant["max_universe_gaussians"]):
        order = np.lexsort((
            idx,
            -seed_count[idx].astype(np.int16),
            -hit_count[idx].astype(np.int16),
            -quality_score[idx].astype(np.float32),
        ))
        idx = idx[order[: int(variant["max_universe_gaussians"])]]
    selection_meta = {
        "raw_hit_gaussian_count": raw_count,
        "quality_filter_kept_count": kept_count,
        "quality_filter_after_cap_count": int(idx.shape[0]),
        "alpha_threshold": alpha_threshold,
        "density_log_threshold": density_log_threshold,
        "density_threshold": float(np.exp(np.clip(density_log_threshold, -60.0, 60.0))),
        "alpha_quantile": float(variant["min_alpha_hit_quantile"]),
        "density_quantile": float(variant["min_density_hit_quantile"]),
        "quality_score_mean_after_cap": float(np.mean(quality_score[idx])) if idx.size else 0.0,
        "quality_score_p50_after_cap": float(np.quantile(quality_score[idx], 0.50)) if idx.size else 0.0,
        **topk_meta,
    }
    return idx, seed_count[idx], veto_count[idx], selection_meta


def _apply_per_mask_topk(
    mask_by_frame: np.ndarray,
    candidates: pd.DataFrame,
    idx: np.ndarray,
    quality_score: np.ndarray,
    topk: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_by_frame: dict[int, set[int]] = defaultdict(set)
    for row in candidates.to_dict("records"):
        candidate_by_frame[int(row["frame_local_index"])].add(int(row["mask_id"]))

    selected: list[np.ndarray] = []
    selected_pair_count = 0
    truncated_pair_count = 0
    for fi in range(mask_by_frame.shape[0]):
        wanted = candidate_by_frame.get(fi)
        if not wanted:
            continue
        labels = np.asarray(mask_by_frame[fi, idx], dtype=np.uint16)
        keep = np.isin(labels, list(wanted))
        if not np.any(keep):
            continue
        frame_idx = idx[keep]
        frame_labels = labels[keep].astype(np.int64, copy=False)
        frame_scores = quality_score[frame_idx]
        order = np.lexsort((-frame_scores, frame_labels))
        sorted_idx = frame_idx[order]
        sorted_labels = frame_labels[order]
        label_values, starts, counts = np.unique(sorted_labels, return_index=True, return_counts=True)
        for start, count in zip(starts.tolist(), counts.tolist()):
            take = min(int(count), topk)
            selected.append(sorted_idx[start : start + take])
            selected_pair_count += 1
            if int(count) > topk:
                truncated_pair_count += 1
    if selected:
        out = np.unique(np.concatenate(selected).astype(np.int64, copy=False))
    else:
        out = np.zeros((0,), dtype=np.int64)
    return out, {
        "per_mask_topk_high_quality": topk,
        "per_mask_topk_input_count": int(idx.shape[0]),
        "per_mask_topk_output_count": int(out.shape[0]),
        "per_mask_topk_mask_observation_count": int(selected_pair_count),
        "per_mask_topk_truncated_mask_observation_count": int(truncated_pair_count),
    }


def _veto_policy(candidates: pd.DataFrame, variant: dict[str, Any]) -> dict[str, Any]:
    veto_counts = candidates["V_veto_support_count"].to_numpy(dtype=np.float64)
    positive = veto_counts[np.isfinite(veto_counts) & (veto_counts > 0)]
    quantile = float(variant["reliable_veto_support_quantile"])
    threshold = float(np.quantile(positive, quantile)) if positive.size else float("inf")
    return {
        "veto_policy_id": "reliable_veto_high_support_and_opposes_anchor_support_v1",
        "reliable_veto_support_quantile": quantile,
        "reliable_veto_support_threshold": threshold,
        "reliable_veto_support_ratio": float(variant["reliable_veto_support_ratio"]),
    }


def _veto_observation_features(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    v = float(row.get("V_veto_support_count", 0.0) or 0.0)
    a = float(row.get("A_anchor_support_count", 0.0) or 0.0)
    s = float(row.get("S_support_count", 0.0) or 0.0)
    support_ref = max(a, s, 1.0)
    strength = float(v / support_ref) if v > 0 else 0.0
    reliable_conflict = bool(
        _as_bool(row.get("V_veto_hit", False))
        and v >= float(policy["reliable_veto_support_threshold"])
        and strength >= float(policy["reliable_veto_support_ratio"])
    )
    return {
        "raw_veto_hit": _as_bool(row.get("V_veto_hit", False)),
        "V_veto_support_count": int(v),
        "A_anchor_support_count": int(a),
        "S_support_count": int(s),
        "veto_support_strength_vs_anchor_support": strength,
        "reliable_veto_conflict": reliable_conflict,
    }


def _emit_support_row(row: dict[str, Any], variant: dict[str, Any]) -> bool:
    if "emit_max_area_ratio" in variant and float(row.get("area_ratio", 0.0) or 0.0) >= float(variant["emit_max_area_ratio"]):
        return False
    if "emit_max_risk" in variant and float(row.get("risk_score", 0.0) or 0.0) > float(variant["emit_max_risk"]):
        return False
    return True


def _component_labels(xyz: np.ndarray, variant: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if xyz.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), {"edge_radius": 0.0, "edge_count": 0}
    tree = cKDTree(xyz)
    k = min(int(variant["knn_k"]) + 1, xyz.shape[0])
    dist, nn = tree.query(xyz, k=k, workers=-1)
    if dist.ndim == 1:
        dist = dist[:, None]
        nn = nn[:, None]
    nn_dist = dist[:, 1:].reshape(-1) if dist.shape[1] > 1 else np.asarray([], dtype=np.float32)
    positive = nn_dist[np.isfinite(nn_dist) & (nn_dist > 0)]
    base_radius = float(np.quantile(positive, float(variant["edge_radius_quantile"]))) if positive.size else 0.0
    edge_radius = base_radius * float(variant["edge_radius_factor"])
    dsu = DSU(xyz.shape[0])
    edge_count = 0
    for i in range(xyz.shape[0]):
        for jpos in range(1, nn.shape[1]):
            j = int(nn[i, jpos])
            if j < 0 or j == i:
                continue
            if float(dist[i, jpos]) <= edge_radius:
                dsu.union(i, j)
                edge_count += 1
    roots = np.asarray([dsu.find(i) for i in range(xyz.shape[0])], dtype=np.int64)
    unique, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64), {
        "edge_radius": edge_radius,
        "edge_count": int(edge_count),
        "component_count_total": int(unique.shape[0]),
        "nn_distance_p50": float(np.quantile(positive, 0.50)) if positive.size else 0.0,
        "nn_distance_p90": float(np.quantile(positive, 0.90)) if positive.size else 0.0,
        "nn_distance_p95": float(np.quantile(positive, 0.95)) if positive.size else 0.0,
    }


def _component_footprints(
    *,
    scene_id: str,
    mask_by_frame: np.ndarray,
    universe_idx: np.ndarray,
    component_labels: np.ndarray,
    candidates: pd.DataFrame,
    meta: dict[str, dict[str, Any]],
    veto_policy: dict[str, Any],
    variant: dict[str, Any],
    variant_id: str,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    candidate_labels_by_frame: dict[int, set[int]] = defaultdict(set)
    for row in candidates.to_dict("records"):
        candidate_labels_by_frame[int(row["frame_local_index"])].add(int(row["mask_id"]))

    comp: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "obs": set(),
            "obs_counts": defaultdict(int),
            "anchor_obs": set(),
            "extra_obs": set(),
            "area_broad_obs": set(),
            "raw_veto_obs": set(),
            "veto_obs": set(),
            "risk_values": [],
        }
    )
    mask_rows: list[dict[str, Any]] = []
    for fi in range(mask_by_frame.shape[0]):
        labels = np.asarray(mask_by_frame[fi, universe_idx], dtype=np.uint16)
        keep_labels = candidate_labels_by_frame.get(fi)
        if not keep_labels:
            continue
        keep = np.isin(labels, list(keep_labels))
        if not np.any(keep):
            continue
        kept_labels = labels[keep].astype(np.int64, copy=False)
        kept_components = component_labels[keep]
        pairs = np.stack([kept_components, kept_labels], axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        frame_id = int(candidates[candidates["frame_local_index"].astype(int) == fi]["frame_id"].iloc[0])
        for (cid, mask_id), count in zip(unique_pairs.tolist(), counts.tolist()):
            obs = f"{scene_id}:{frame_id}:{int(mask_id)}"
            row = meta.get(obs)
            if row is None:
                continue
            cid = int(cid)
            veto_features = _veto_observation_features(row, veto_policy)
            emitted_to_support = _emit_support_row(row, variant)
            if emitted_to_support:
                comp[cid]["obs"].add(obs)
                comp[cid]["obs_counts"][obs] += int(count)
                comp[cid]["risk_values"].append(float(row.get("risk_score", 0.0)))
                if _as_bool(row.get("A_anchor_hit", False)):
                    comp[cid]["anchor_obs"].add(obs)
                if str(row.get("candidate_delta_type", "")) == "extra_over_semhard":
                    comp[cid]["extra_obs"].add(obs)
                if float(row.get("area_ratio", 0.0)) >= 0.12:
                    comp[cid]["area_broad_obs"].add(obs)
                if veto_features["raw_veto_hit"]:
                    comp[cid]["raw_veto_obs"].add(obs)
                if veto_features["reliable_veto_conflict"]:
                    comp[cid]["veto_obs"].add(obs)
            mask_rows.append(
                {
                    "schema_version": "stream4d_v103_r2_phase2_component_mask_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_id,
                    "variant_id": variant_id,
                    "component_id": cid,
                    "frame_id": frame_id,
                    "frame_local_index": fi,
                    "mask_id": int(mask_id),
                    "mask_observation_id": obs,
                    "component_mask_gaussian_count": int(count),
                    "candidate_delta_type": row.get("candidate_delta_type", ""),
                    "A_anchor_hit": _as_bool(row.get("A_anchor_hit", False)),
                    "V_veto_hit": veto_features["raw_veto_hit"],
                    "V_veto_reliable_conflict": veto_features["reliable_veto_conflict"],
                    "V_veto_support_count": veto_features["V_veto_support_count"],
                    "A_anchor_support_count": veto_features["A_anchor_support_count"],
                    "S_support_count": veto_features["S_support_count"],
                    "veto_support_strength_vs_anchor_support": veto_features["veto_support_strength_vs_anchor_support"],
                    "veto_policy_id": veto_policy["veto_policy_id"],
                    "reliable_veto_support_threshold": veto_policy["reliable_veto_support_threshold"],
                    "reliable_veto_support_ratio": veto_policy["reliable_veto_support_ratio"],
                    "emitted_to_support": emitted_to_support,
                    "emit_support_policy_id": variant.get("emit_support_policy_id", "emit_all_candidate_footprint"),
                    "emit_max_area_ratio": variant.get("emit_max_area_ratio", None),
                    "emit_max_risk": variant.get("emit_max_risk", None),
                    "area_ratio": float(row.get("area_ratio", 0.0)),
                    "risk_score": float(row.get("risk_score", 0.0)),
                    "uses_gt_for_selection": False,
                }
            )
    return comp, mask_rows


def _scene_variant(
    scene_id: str,
    phase1_root: Path,
    phase9b_root: Path,
    phase9a_root: Path,
    variant: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, meta = _candidate_tables(phase1_root, scene_id)
    veto_policy = _veto_policy(candidates, variant)
    scene_dir = phase9b_root / scene_id
    mask_by_frame_path = scene_dir / "mask_by_frame.npy"
    mask_by_frame = np.load(mask_by_frame_path, mmap_mode="r")
    xyz_all = np.load(scene_dir / "xyz.npy", mmap_mode="r")
    quality = _load_gaussian_quality_cached(str(phase9a_root), scene_id)
    if len(quality["alpha"]) != mask_by_frame.shape[1] or len(quality["alpha"]) != xyz_all.shape[0]:
        raise ValueError(
            f"DA3 quality/mask/xyz length mismatch for {scene_id}: "
            f"quality={len(quality['alpha'])} mask={mask_by_frame.shape[1]} xyz={xyz_all.shape[0]}"
        )
    hit_count_all, seed_count_all, veto_count_all = _support_counts_cached(str(mask_by_frame_path), str(phase1_root), scene_id)
    universe_idx, seed_count, veto_count, selection_meta = _select_universe(
        mask_by_frame,
        candidates,
        hit_count_all,
        seed_count_all,
        veto_count_all,
        quality,
        variant,
    )
    xyz = np.asarray(xyz_all[universe_idx], dtype=np.float32)
    labels, graph_meta = _component_labels(xyz, variant)
    comp, mask_rows = _component_footprints(
        scene_id=scene_id,
        mask_by_frame=mask_by_frame,
        universe_idx=universe_idx,
        component_labels=labels,
        candidates=candidates,
        meta=meta,
        veto_policy=veto_policy,
        variant=variant,
        variant_id=str(variant["variant_id"]),
    )
    component_size = np.bincount(labels, minlength=(int(labels.max()) + 1) if labels.size else 0)
    seed_by_comp = np.bincount(labels, weights=(seed_count > 0).astype(np.float32), minlength=component_size.shape[0])
    veto_by_comp = np.bincount(labels, weights=(veto_count > 0).astype(np.float32), minlength=component_size.shape[0])
    alpha_u = np.asarray(quality["alpha"], dtype=np.float32)[universe_idx]
    density_log_u = np.asarray(quality["density_log"], dtype=np.float32)[universe_idx]
    quality_score_u = np.asarray(quality["quality_score"], dtype=np.float32)[universe_idx]
    alpha_by_comp = np.divide(
        np.bincount(labels, weights=alpha_u, minlength=component_size.shape[0]),
        np.maximum(component_size, 1),
    ) if component_size.size else np.asarray([], dtype=np.float32)
    density_log_by_comp = np.divide(
        np.bincount(labels, weights=density_log_u, minlength=component_size.shape[0]),
        np.maximum(component_size, 1),
    ) if component_size.size else np.asarray([], dtype=np.float32)
    quality_by_comp = np.divide(
        np.bincount(labels, weights=quality_score_u, minlength=component_size.shape[0]),
        np.maximum(component_size, 1),
    ) if component_size.size else np.asarray([], dtype=np.float32)

    component_rows: list[dict[str, Any]] = []
    clean_ids: set[int] = set()
    for cid in range(component_size.shape[0]):
        data = comp.get(cid, {})
        obs = set(data.get("obs", set()))
        extra = set(data.get("extra_obs", set()))
        area_broad = set(data.get("area_broad_obs", set()))
        raw_veto_obs = set(data.get("raw_veto_obs", set()))
        veto_obs = set(data.get("veto_obs", set()))
        risk_values = list(data.get("risk_values", []))
        object_like_count = len(obs)
        induced_count = len(extra)
        induced_broad_count = len(area_broad & extra)
        raw_veto_rate = float(len(raw_veto_obs) / max(len(obs), 1))
        veto_rate = float(len(veto_obs) / max(len(obs), 1))
        risk_mean = float(np.mean(risk_values)) if risk_values else 0.0
        area_broad_rate = float(induced_broad_count / max(induced_count, 1))
        is_clean = bool(
            seed_by_comp[cid] >= int(variant["min_seed_gaussians"])
            and object_like_count >= int(variant["min_object_like_footprint_count"])
            and int(component_size[cid]) >= int(variant["min_component_size"])
            and veto_rate <= float(variant["max_veto_footprint_rate"])
            and risk_mean <= float(variant["max_candidate_risk_mean"])
            and area_broad_rate <= float(variant["max_area_broad_rate"])
        )
        if is_clean:
            clean_ids.add(cid)
        component_rows.append(
            {
                "schema_version": "stream4d_v103_r2_phase2_component_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene_id,
                "variant_id": variant["variant_id"],
                "component_id": cid,
                "component_gaussian_count": int(component_size[cid]),
                "component_seed_gaussian_count": int(seed_by_comp[cid]),
                "component_veto_gaussian_count": int(veto_by_comp[cid]),
                "component_alpha_mean": float(alpha_by_comp[cid]),
                "component_density_log_mean": float(density_log_by_comp[cid]),
                "component_quality_score_mean": float(quality_by_comp[cid]),
                "object_like_footprint_count": object_like_count,
                "induced_obs_count": induced_count,
                "induced_object_like_obs_count": induced_count,
                "induced_broad_obs_count": induced_broad_count,
                "induced_broad_obs_rate": area_broad_rate,
                "component_raw_veto_footprint_rate": raw_veto_rate,
                "component_reliable_veto_obs_count": int(len(veto_obs)),
                "component_veto_footprint_rate": veto_rate,
                "component_semantic_risk_mean": risk_mean,
                "component_semantic_risk_p90": float(np.quantile(risk_values, 0.90)) if risk_values else 0.0,
                "is_clean_component": is_clean,
                "uses_gt_for_selection": False,
            }
        )
    clean_components = [row for row in component_rows if bool(row["is_clean_component"])]
    clean_component_sum_induced_obs = int(sum(int(row["induced_obs_count"]) for row in clean_components))
    clean_component_sum_induced_obj = int(sum(int(row["induced_object_like_obs_count"]) for row in clean_components))
    clean_component_sum_induced_broad = int(sum(int(row["induced_broad_obs_count"]) for row in clean_components))
    clean_ids = {int(row["component_id"]) for row in clean_components}
    clean_mask_rows = [row for row in mask_rows if int(row["component_id"]) in clean_ids and bool(row.get("emitted_to_support", True))]
    clean_unique_obs = {str(row["mask_observation_id"]) for row in clean_mask_rows}
    clean_unique_induced_obs = {
        str(row["mask_observation_id"])
        for row in clean_mask_rows
        if str(row.get("candidate_delta_type", "")) == "extra_over_semhard"
    }
    clean_unique_induced_broad_obs = {
        str(row["mask_observation_id"])
        for row in clean_mask_rows
        if str(row.get("candidate_delta_type", "")) == "extra_over_semhard" and float(row.get("area_ratio", 0.0)) >= 0.12
    }
    clean_induced_obs = int(len(clean_unique_induced_obs))
    clean_induced_obj = clean_induced_obs
    clean_induced_broad = int(len(clean_unique_induced_broad_obs))
    summary = {
        "schema_version": "stream4d_v103_r2_phase2_scene_variant_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "variant_id": variant["variant_id"],
        "DA3_primitive_count": int(mask_by_frame.shape[1]),
        "universe_gaussian_count_raw": int(selection_meta["raw_hit_gaussian_count"]),
        "universe_gaussian_count_after_quality": int(selection_meta["quality_filter_kept_count"]),
        "universe_gaussian_count_after_per_mask_topk": int(selection_meta["per_mask_topk_output_count"]),
        "universe_gaussian_count_after_cap": int(universe_idx.shape[0]),
        "per_mask_topk_high_quality": int(selection_meta["per_mask_topk_high_quality"]),
        "per_mask_topk_input_count": int(selection_meta["per_mask_topk_input_count"]),
        "per_mask_topk_output_count": int(selection_meta["per_mask_topk_output_count"]),
        "per_mask_topk_mask_observation_count": int(selection_meta.get("per_mask_topk_mask_observation_count", 0)),
        "per_mask_topk_truncated_mask_observation_count": int(selection_meta.get("per_mask_topk_truncated_mask_observation_count", 0)),
        "gaussian_quality_gate": "hit_gaussian_alpha_and_density_proxy_quantile_filter",
        "gaussian_quality_alpha_source": quality["alpha_source"],
        "gaussian_quality_density_source": quality["density_source"],
        "da3_ply_path": _rel(quality["ply_path"]),
        "emit_support_policy_id": variant.get("emit_support_policy_id", "emit_all_candidate_footprint"),
        "emit_max_area_ratio": variant.get("emit_max_area_ratio", ""),
        "emit_max_risk": variant.get("emit_max_risk", ""),
        "veto_policy_id": veto_policy["veto_policy_id"],
        "reliable_veto_support_quantile": veto_policy["reliable_veto_support_quantile"],
        "reliable_veto_support_threshold": veto_policy["reliable_veto_support_threshold"],
        "reliable_veto_support_ratio": veto_policy["reliable_veto_support_ratio"],
        "alpha_hit_quantile": float(selection_meta["alpha_quantile"]),
        "density_hit_quantile": float(selection_meta["density_quantile"]),
        "alpha_threshold": selection_meta["alpha_threshold"],
        "density_log_threshold": selection_meta["density_log_threshold"],
        "density_threshold": selection_meta["density_threshold"],
        "quality_score_mean_after_cap": float(selection_meta["quality_score_mean_after_cap"]),
        "quality_score_p50_after_cap": float(selection_meta["quality_score_p50_after_cap"]),
        "clean_component_count": int(len(clean_components)),
        "clean_with_induced_component_count": int(sum(int(row["induced_obs_count"]) > 0 for row in clean_components)),
        "clean_unique_mask_observation_count": int(len(clean_unique_obs)),
        "clean_induced_obs_count_component_sum": clean_component_sum_induced_obs,
        "clean_induced_object_like_obs_count_component_sum": clean_component_sum_induced_obj,
        "clean_induced_broad_obs_count_component_sum": clean_component_sum_induced_broad,
        "clean_induced_obs_count": clean_induced_obs,
        "clean_induced_object_like_obs_count": clean_induced_obj,
        "clean_induced_broad_obs_count": clean_induced_broad,
        "clean_induced_broad_rate": float(clean_induced_broad / max(clean_induced_obs, 1)),
        "component_anchor_hit_rate": float(np.mean([int(row["component_seed_gaussian_count"]) > 0 for row in component_rows])) if component_rows else 0.0,
        "component_raw_veto_hit_rate": float(np.mean([float(row["component_raw_veto_footprint_rate"]) > 0.0 for row in clean_components])) if clean_components else 1.0,
        "component_veto_conflict_rate": float(np.mean([float(row["component_veto_footprint_rate"]) > 0.0 for row in clean_components])) if clean_components else 1.0,
        "component_semantic_risk_mean": float(np.mean([float(row["component_semantic_risk_mean"]) for row in clean_components])) if clean_components else 0.0,
        "component_semantic_risk_p90": float(np.quantile([float(row["component_semantic_risk_mean"]) for row in clean_components], 0.90)) if clean_components else 0.0,
        "component_backend": "scipy_ckdtree_c0001_da3_mask_by_frame_components",
        **graph_meta,
        "uses_gt_for_selection": False,
    }
    return summary, component_rows, mask_rows, clean_mask_rows


def _gate(gate_id: str, passed: bool, observed: Any, required: Any, repair_direction: str = "") -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_r2_phase2_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_id": gate_id,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "repair_direction": repair_direction,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    phase1_root = _project(args.phase1_root)
    phase9b_root = _project(args.phase9b_cache_root)
    phase9a_root = _project(args.phase9a_root)
    all_scene_variant_rows: list[dict[str, Any]] = []
    all_components: list[dict[str, Any]] = []
    all_component_masks: list[dict[str, Any]] = []
    all_clean_masks: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for scene in ["scene0011_00", "scene0050_00"]:
            scene_summary, component_rows, mask_rows, clean_mask_rows = _scene_variant(scene, phase1_root, phase9b_root, phase9a_root, variant)
            all_scene_variant_rows.append(scene_summary)
            all_components.extend(component_rows)
            all_component_masks.extend(mask_rows)
            all_clean_masks.extend(clean_mask_rows)

    best_by_scene: dict[str, dict[str, Any]] = {}
    for scene in ["scene0011_00", "scene0050_00"]:
        rows = [row for row in all_scene_variant_rows if row["scene_id"] == scene]
        best_by_scene[scene] = max(
            rows,
            key=lambda row: (
                int(row["clean_component_count"]),
                int(row["clean_induced_object_like_obs_count"]),
                -float(row["clean_induced_broad_rate"]),
            ),
        )

    gate_rows: list[dict[str, Any]] = []
    for scene, row in best_by_scene.items():
        gate_rows.append(_gate(f"{scene}_clean_component_count_ge_5", int(row["clean_component_count"]) >= 5, row["clean_component_count"], ">= 5", "Lower component footprint support without relaxing object-like/non-broad guard."))
        gate_rows.append(_gate(f"{scene}_clean_induced_object_like_obs_count_ge_30", int(row["clean_induced_object_like_obs_count"]) >= 30, row["clean_induced_object_like_obs_count"], ">= 30", "Try per-mask top-k support or return to R2-1 risk modeling."))
        gate_rows.append(_gate(f"{scene}_clean_induced_broad_rate_le_0p10", float(row["clean_induced_broad_rate"]) <= 0.10, row["clean_induced_broad_rate"], "<= 0.10", "Stop broad-dominated family and tighten semantic risk."))
        gate_rows.append(_gate(f"{scene}_component_veto_conflict_rate_le_0p05", float(row["component_veto_conflict_rate"]) <= 0.05, row["component_veto_conflict_rate"], "<= 0.05", "Do not promote DA3-soft support if V_veto conflicts dominate."))
    gate_rows.append(_gate("uses_gt_for_selection_false", True, False, "False"))

    failure_rows = [
        {
            "schema_version": "stream4d_v103_r2_phase2_failure_row_v1",
            "phase_id": PHASE_ID,
            "failure_id": row["gate_id"],
            "severity": "blocker",
            "observed": row["observed"],
            "expected": row["required"],
            "repair_direction": row["repair_direction"],
        }
        for row in gate_rows
        if not row["pass"]
    ]

    component_path = out / "da3_semsoft_component_rows.csv"
    mask_path = out / "da3_semsoft_component_mask_rows.csv"
    incidence_path = out / "da3_semsoft_primitive_incidence_rows.parquet"
    casebook_path = out / "da3_semsoft_casebook_rows.csv"
    policy_path = out / "component_policy_rows.csv"
    _write_csv(component_path, all_components)
    _write_csv(mask_path, all_component_masks)
    incidence_df = pd.DataFrame(all_clean_masks)
    if incidence_df.empty:
        incidence_df = pd.DataFrame(
            columns=[
                "schema_version",
                "phase_id",
                "scene_id",
                "variant_id",
                "component_id",
                "frame_id",
                "frame_local_index",
                "mask_id",
                "mask_observation_id",
                "component_mask_gaussian_count",
                "candidate_delta_type",
                "A_anchor_hit",
                "V_veto_hit",
                "V_veto_reliable_conflict",
                "V_veto_support_count",
                "A_anchor_support_count",
                "S_support_count",
                "veto_support_strength_vs_anchor_support",
                "veto_policy_id",
                "reliable_veto_support_threshold",
                "reliable_veto_support_ratio",
                "emitted_to_support",
                "emit_support_policy_id",
                "emit_max_area_ratio",
                "emit_max_risk",
                "area_ratio",
                "risk_score",
                "uses_gt_for_selection",
            ]
        )
    incidence_df.to_parquet(incidence_path, index=False)
    _write_csv(casebook_path, [row for row in all_scene_variant_rows if row["variant_id"] == best_by_scene[row["scene_id"]]["variant_id"]])
    _write_csv(policy_path, [dict(schema_version="stream4d_v103_r2_phase2_component_policy_row_v1", phase_id=PHASE_ID, **variant) for variant in VARIANTS])
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    (out / "last_command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")

    pass_all = not failure_rows
    summary = {
        "schema_version": "stream4d_v103_r2_phase2_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_R2_2_DA3_SEMSOFT_PROVIDER_SUPPORT_READY" if pass_all else "NO_GO_DA3_SEMSOFT_PROVIDER_BLOCKER",
        "phase1_root": _rel(phase1_root),
        "phase9b_cache_root": _rel(phase9b_root),
        "phase9a_root": _rel(phase9a_root),
        "variant_count": len(VARIANTS),
        "best_by_scene": best_by_scene,
        "clean_component_count_min": min(int(row["clean_component_count"]) for row in best_by_scene.values()),
        "clean_induced_object_like_obs_count_min": min(int(row["clean_induced_object_like_obs_count"]) for row in best_by_scene.values()),
        "clean_induced_broad_rate_max": max(float(row["clean_induced_broad_rate"]) for row in best_by_scene.values()),
        "component_veto_conflict_rate_max": max(float(row["component_veto_conflict_rate"]) for row in best_by_scene.values()),
        "uses_gt_for_selection": False,
        "truthfulness_note": "R2-2 uses c0001 DA3 mask_by_frame/xyz caches and filters DA3 Gaussian primitives by high alpha/density-proxy quantiles before component growth. Raw V_veto hits remain risk/audit fields; hard veto conflict requires high support and opposition to A/S support. The incidence parquet is limited to clean component mask footprints.",
        "outputs": {
            "da3_semsoft_component_rows": _rel(component_path),
            "da3_semsoft_component_mask_rows": _rel(mask_path),
            "da3_semsoft_primitive_incidence_rows": _rel(incidence_path),
            "da3_semsoft_casebook_rows": _rel(casebook_path),
            "component_policy_rows": _rel(policy_path),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "summary": _rel(out / "summary.json"),
        },
    }
    _write_json(out / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v103 R2-2 DA3 semantic-soft support components.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase1-root", default=str(DEFAULT_PHASE1_ROOT))
    parser.add_argument("--phase9b-cache-root", default=str(DEFAULT_PHASE9B_CACHE_ROOT))
    parser.add_argument("--phase9a-root", default=str(DEFAULT_PHASE9A_ROOT))
    return parser.parse_args()


def main() -> int:
    summary = build(parse_args())
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["decision"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
