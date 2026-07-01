#!/usr/bin/env python3
"""Build ACL2 v87 Phase1 scale-conditioned pair universe."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from v86_soft_latent_utils import (
    clamp01,
    effective_sample_size,
    finite_quantile,
    pair_key,
    parse_bool,
    safe_float,
    safe_int,
    seq_norm,
    weighted_rank,
    write_csv,
    write_json,
)


DEFAULT_ANCHOR_ROWS = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase1_anchor_pair_universe/anchor_pair_rows.csv"
)
DEFAULT_V86_SOFT_ROWS = Path(
    "results/acl2_v86tf_robust_soft_latent_gauge_transport/phase1_soft_pair_universe/soft_pair_rows.csv"
)
DEFAULT_FEATURE_PT = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase2_qk_feature_bank/qk_anchor_features.pt"
)
DEFAULT_SCALE_LABELS = Path(
    "results/acl2_v86tf_robust_soft_latent_gauge_transport/phase4_offline_scale_labels/offline_scale_jump_rows.csv"
)
DEFAULT_OUT = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase1_scale_conditioned_pair_universe"
)

PATCH_COLS = 66
PATCH_ROWS = 19
PATCH_CELL = 14.0


ROW_FIELDS = [
    "seq",
    "prev_chunk",
    "curr_chunk",
    "pair_id",
    "base_case_type",
    "quality_type",
    "stress_type",
    "prev_frame_id",
    "curr_frame_id",
    "prev_patch_id",
    "curr_patch_id",
    "prev_pixel_x",
    "prev_pixel_y",
    "curr_pixel_x",
    "curr_pixel_y",
    "raw_overlap_residual",
    "confidence_weighted_residual",
    "prev_confidence",
    "curr_confidence",
    "either_zero_conf",
    "both_zero_conf",
    "semantic_projection_valid",
    "same_label",
    "same_role",
    "cross_boundary_flag",
    "dynamic_flag",
    "local_neighbor_count",
    "local_shape_log_ratio_median",
    "local_shape_log_ratio_iqr",
    "local_shape_log_ratio_signed_median",
    "pairwise_distance_ratio_mean",
    "pairwise_distance_ratio_std",
    "local_spread_prev",
    "local_spread_curr",
    "local_depth_median",
    "parallax_proxy",
    "observability_score",
    "support_weight",
    "conflict_weight",
    "absence_weight",
    "state_label",
    "state_reason",
    "support_sufficient_dim8",
    "support_sufficient_dim4",
    "conflict_high_flag",
    "absence_high_flag",
    "q_feature_available",
    "k_feature_available",
    "q_feature_source",
    "k_feature_source",
    "feature_source_path",
    "layer_id",
    "head_id",
    "feature_dim",
    "q_norm",
    "k_norm",
    "qk_identity_residual",
    "qk_cosine",
    "feature_rank_group",
    "prev_chunk_sim3_scale_gt",
    "curr_chunk_sim3_scale_gt",
    "adjacent_log_scale_jump_gt",
    "abs_log_scale_jump_gt",
    "scale_jump_high_label",
    "full_ATE_pair_context",
    "per_chunk_oracle_gap_context",
    "true_route_mass_support",
    "true_route_mass_conflict",
    "same_mass_random_route_mass",
    "semantic_shuffle_route_mass",
    "route_entropy_before",
    "route_entropy_after",
    "merge_weight_support",
    "merge_weight_conflict",
    "merge_counterfactual_improvement",
    "runtime_action_fidelity",
    "runtime_geometry_delta",
    "local_shape_proxy_source",
    "local_shape_proxy_status",
    "anchor_row_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-rows", type=Path, default=DEFAULT_ANCHOR_ROWS)
    parser.add_argument("--v86-soft-rows", type=Path, default=DEFAULT_V86_SOFT_ROWS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_PT)
    parser.add_argument("--scale-labels", type=Path, default=DEFAULT_SCALE_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--patch-radius", type=int, default=0)
    parser.add_argument("--shape-aggregator", choices=["median", "trimmed_mean"], default="median")
    parser.add_argument("--ratio-mode", choices=["abs", "signed"], default="abs")
    parser.add_argument("--sample-max", type=int, default=384)
    parser.add_argument("--feature-dim", type=int, default=8)
    return parser.parse_args()


def _load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["q_features"].detach().cpu().float().numpy(), payload["k_features"].detach().cpu().float().numpy()


def _bool(value: Any) -> bool:
    return parse_bool(value)


def _same_label(row: Mapping[str, Any]) -> bool:
    text = str(row.get("same_label") or "").strip()
    if text:
        return parse_bool(text)
    return str(row.get("prev_sem_label")) == str(row.get("curr_sem_label"))


def _same_role(row: Mapping[str, Any]) -> bool:
    value = row.get("same_role")
    if safe_float(value) is not None:
        return bool(float(value) >= 0.5)
    return parse_bool(value)


def _stress_type(row: Mapping[str, Any]) -> str:
    seq = seq_norm(row.get("seq"))
    quality = str(row.get("quality_label") or "")
    source = str(row.get("source_path") or "")
    zero = parse_bool(row.get("zero_conf_flag"))
    prev_conf = safe_float(row.get("prev_sem_conf"))
    curr_conf = safe_float(row.get("curr_sem_conf"))
    if "low_conf_stress" in quality:
        return "quality_low_conf_stress"
    if zero:
        return "zero_confidence"
    if seq == "01" and "minconf0" in source:
        return "seq01_minconf0_overlap_source"
    if seq == "01" and (prev_conf == 0.0 or curr_conf == 0.0):
        return "seq01_zero_semantic_confidence"
    return ""


def _semantic_projection_valid(row: Mapping[str, Any]) -> bool:
    prev_label = safe_float(row.get("prev_sem_label"))
    curr_label = safe_float(row.get("curr_sem_label"))
    prev_conf = safe_float(row.get("prev_sem_conf"))
    curr_conf = safe_float(row.get("curr_sem_conf"))
    return prev_label is not None and curr_label is not None and (prev_conf or 0.0) > 0 and (curr_conf or 0.0) > 0


def _patch_id_from_yx(y: int, x: int) -> int:
    yy = max(0, min(PATCH_ROWS - 1, int(y)))
    xx = max(0, min(PATCH_COLS - 1, int(x)))
    return yy * PATCH_COLS + xx


def _patch_yx_from_id(patch_id: Any) -> tuple[int, int] | None:
    pid = safe_int(patch_id)
    if pid is None or pid < 0:
        return None
    return int(pid // PATCH_COLS), int(pid % PATCH_COLS)


def _tensor_np(obj: Mapping[str, Any], key: str) -> np.ndarray | None:
    value = obj.get(key)
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _aggregate(values: np.ndarray, mode: str) -> float:
    arr = values[np.isfinite(values)]
    if arr.size == 0:
        return float("nan")
    if mode == "trimmed_mean" and arr.size >= 5:
        lo, hi = np.quantile(arr, [0.10, 0.90])
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        if trimmed.size:
            return float(np.mean(trimmed))
    return float(np.median(arr))


def _knn_patch_stats(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    prev_pixels: np.ndarray,
    target_y: int,
    target_x: int,
    *,
    k_neighbors: int,
    patch_radius: int,
    sample_max: int,
    shape_aggregator: str,
) -> dict[str, Any]:
    if prev_pts is None or curr_pts is None or prev_pixels is None:
        return {"available": False, "status": "missing_raw_arrays"}
    if prev_pts.ndim != 2 or curr_pts.ndim != 2 or prev_pixels.ndim != 2:
        return {"available": False, "status": "bad_array_shape"}
    patch_y = np.floor(prev_pixels[:, 0].astype(np.float64) / PATCH_CELL).astype(np.int64)
    patch_x = np.floor(prev_pixels[:, 1].astype(np.float64) / PATCH_CELL).astype(np.int64)
    mask = (
        np.isfinite(prev_pts).all(axis=1)
        & np.isfinite(curr_pts).all(axis=1)
        & np.isfinite(prev_pixels).all(axis=1)
        & (patch_y >= target_y - patch_radius)
        & (patch_y <= target_y + patch_radius)
        & (patch_x >= target_x - patch_radius)
        & (patch_x <= target_x + patch_radius)
    )
    idx = np.flatnonzero(mask)
    if idx.size <= max(2, k_neighbors):
        return {"available": False, "status": f"too_few_neighbors:{idx.size}", "local_neighbor_count": int(idx.size)}
    if idx.size > sample_max:
        pick = np.linspace(0, idx.size - 1, sample_max).round().astype(np.int64)
        idx = idx[pick]
    p = prev_pts[idx].astype(np.float64)
    c = curr_pts[idx].astype(np.float64)
    n = p.shape[0]
    dp = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    dc = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    mean_d = 0.5 * (dp + dc)
    np.fill_diagonal(mean_d, np.inf)
    kk = min(max(1, int(k_neighbors)), n - 1)
    neigh = np.argpartition(mean_d, kth=kk - 1, axis=1)[:, :kk]
    rows = np.arange(n)[:, None]
    prev_d = dp[rows, neigh]
    curr_d = dc[rows, neigh]
    eps = 1e-8
    signed_log = np.log((curr_d + eps) / (prev_d + eps))
    abs_log = np.abs(signed_log)
    per_point_abs = np.median(abs_log, axis=1)
    per_point_signed = np.median(signed_log, axis=1)
    pair_ratio = (curr_d + eps) / (prev_d + eps)
    spread_prev = np.std(prev_d, axis=1)
    spread_curr = np.std(curr_d, axis=1)
    return {
        "available": True,
        "status": "raw_overlap_knn",
        "local_neighbor_count": int(n),
        "local_shape_log_ratio_median": _aggregate(per_point_abs, shape_aggregator),
        "local_shape_log_ratio_iqr": float(np.quantile(per_point_abs, 0.75) - np.quantile(per_point_abs, 0.25)),
        "local_shape_log_ratio_signed_median": _aggregate(per_point_signed, shape_aggregator),
        "pairwise_distance_ratio_mean": float(np.mean(pair_ratio[np.isfinite(pair_ratio)])),
        "pairwise_distance_ratio_std": float(np.std(pair_ratio[np.isfinite(pair_ratio)])),
        "local_spread_prev": float(np.median(spread_prev[np.isfinite(spread_prev)])),
        "local_spread_curr": float(np.median(spread_curr[np.isfinite(spread_curr)])),
        "local_depth_median": float(np.median(0.5 * (p[:, 2] + c[:, 2]))),
    }


class RawShapeCache:
    def __init__(self, *, k_neighbors: int, patch_radius: int, sample_max: int, shape_aggregator: str) -> None:
        self.k_neighbors = k_neighbors
        self.patch_radius = patch_radius
        self.sample_max = sample_max
        self.shape_aggregator = shape_aggregator
        self._objects: dict[str, dict[str, Any]] = {}
        self._stats: dict[tuple[str, int, int], dict[str, Any]] = {}
        self.missing: Counter[str] = Counter()

    def _load(self, path_text: Any) -> dict[str, Any] | None:
        path = str(path_text or "").strip()
        if not path:
            self.missing["empty_source_path"] += 1
            return None
        if path not in self._objects:
            try:
                obj = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                self.missing[f"load_error:{type(exc).__name__}"] += 1
                self._objects[path] = {"_load_error": str(exc)}
            else:
                self._objects[path] = obj if isinstance(obj, dict) else {"_bad_payload_type": str(type(obj))}
        obj = self._objects[path]
        if "_load_error" in obj or "_bad_payload_type" in obj:
            return None
        return obj

    def stats(self, path_text: Any, patch_id: Any) -> dict[str, Any]:
        path = str(path_text or "").strip()
        yx = _patch_yx_from_id(patch_id)
        if not path or yx is None:
            return {"available": False, "status": "missing_source_or_patch"}
        target_y, target_x = yx
        key = (path, int(target_y), int(target_x))
        if key in self._stats:
            return self._stats[key]
        obj = self._load(path)
        if obj is None:
            out = {"available": False, "status": "raw_pt_load_failed"}
        else:
            prev_pts = _tensor_np(obj, "prev_overlap_local_points")
            if prev_pts is None:
                prev_pts = _tensor_np(obj, "prev_overlap_points")
            curr_pts = _tensor_np(obj, "curr_overlap_local_points")
            if curr_pts is None:
                curr_pts = _tensor_np(obj, "curr_overlap_points")
            prev_pixels = _tensor_np(obj, "prev_pixel_coords")
            if prev_pixels is None:
                prev_pixels = _tensor_np(obj, "curr_pixel_coords")
            out = _knn_patch_stats(
                prev_pts,
                curr_pts,
                prev_pixels,
                target_y,
                target_x,
                k_neighbors=self.k_neighbors,
                patch_radius=self.patch_radius,
                sample_max=self.sample_max,
                shape_aggregator=self.shape_aggregator,
            )
        self._stats[key] = out
        return out


def _exp_score(value: float | None, scale: float | None, missing_value: float) -> float:
    if value is None or scale is None or scale <= 0:
        return missing_value
    return clamp01(math.exp(-max(0.0, value) / scale))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return None
    v = v[mask]
    w = w[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / max(float(w.sum()), 1e-12)
    return float(v[min(int(np.searchsorted(cdf, 0.5)), v.size - 1)])


def _state_from_weights(
    *,
    stress_type: str,
    obs: float,
    support: float,
    conflict: float,
    absence: float,
    same_label: bool,
    same_role: bool,
) -> tuple[str, str]:
    if stress_type:
        return "STRESS", stress_type
    if obs < 0.05:
        return "ABSENCE", "low_observability"
    if support >= 0.05 and conflict < max(0.05, support * 1.10):
        if not same_label:
            return "ABSENCE", "semantic_mismatch_prevents_positive_support"
        if same_role is False:
            return "ABSENCE", "role_mismatch_prevents_positive_support"
        return "SUPPORT", "shape_overlap_observable_consistent"
    if conflict >= 0.05 and conflict >= support * 1.10:
        return "CONFLICT", "shape_overlap_observable_conflict"
    if absence >= 0.50:
        return "ABSENCE", "support_conflict_mass_too_low"
    return "ABSENCE", "ambiguous_low_margin"


def _scale_label_map(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    if "abs_log_scale_jump" in df.columns:
        valid = pd.to_numeric(df["abs_log_scale_jump"], errors="coerce").dropna()
        high_threshold = float(valid.quantile(0.75)) if len(valid) else float("nan")
    else:
        high_threshold = float("nan")
    for _, row in df.iterrows():
        key = (seq_norm(row.get("seq")), int(row["prev_chunk"]), int(row["curr_chunk"]))
        abs_jump = safe_float(row.get("abs_log_scale_jump"))
        out[key] = {
            "prev_chunk_sim3_scale_gt": safe_float(row.get("chunk_scale_prev")),
            "curr_chunk_sim3_scale_gt": safe_float(row.get("chunk_scale_curr")),
            "adjacent_log_scale_jump_gt": safe_float(row.get("adjacent_log_scale_jump")),
            "abs_log_scale_jump_gt": abs_jump,
            "scale_jump_high_label": bool(abs_jump is not None and math.isfinite(high_threshold) and abs_jump >= high_threshold),
            "full_ATE_pair_context": safe_float(row.get("full_ATE_contribution_proxy")),
            "per_chunk_oracle_gap_context": "",
        }
    return out


def _norm_qk(q: np.ndarray, k: np.ndarray) -> tuple[float, float, float, float]:
    qn = float(np.linalg.norm(q))
    kn = float(np.linalg.norm(k))
    resid = float(np.linalg.norm(q - k))
    cos = float(np.dot(q, k) / max(qn * kn, 1e-12))
    return qn, kn, resid, cos


def _feature_rank_group(rank: int, feature_dim: int) -> str:
    if rank >= int(0.75 * feature_dim):
        return "rank_sufficient"
    if rank > 0:
        return "rank_low"
    return "rank_unavailable"


def main() -> None:
    args = parse_args()
    anchor = pd.read_csv(args.anchor_rows)
    q_features, k_features = _load_features(args.features)
    if len(anchor) != q_features.shape[0] or len(anchor) != k_features.shape[0]:
        raise ValueError(f"row/features mismatch rows={len(anchor)} q={q_features.shape} k={k_features.shape}")

    # The v86 soft table is optional context. Reading it also records that v87 is not rebuilding blind.
    v86_soft_exists = args.v86_soft_rows.exists()
    scale_by_pair = _scale_label_map(args.scale_labels)
    shape_cache = RawShapeCache(
        k_neighbors=args.k_neighbors,
        patch_radius=args.patch_radius,
        sample_max=args.sample_max,
        shape_aggregator=args.shape_aggregator,
    )

    initial: list[dict[str, Any]] = []
    for idx, row_s in anchor.iterrows():
        row = row_s.to_dict()
        raw_stats = shape_cache.stats(row.get("source_path"), row.get("prev_patch_id"))
        shape_value = safe_float(raw_stats.get("local_shape_log_ratio_signed_median" if args.ratio_mode == "signed" else "local_shape_log_ratio_median"))
        proxy_source = "raw_overlap_knn" if raw_stats.get("available") else "csv_fallback"
        proxy_status = str(raw_stats.get("status") or "unknown")
        if shape_value is None:
            shape_value = safe_float(row.get("pairwise_distance_ratio_residual")) or safe_float(row.get("local_shape_residual"))
        spread_prev = safe_float(raw_stats.get("local_spread_prev")) or safe_float(row.get("local_3d_spread_prev"))
        spread_curr = safe_float(raw_stats.get("local_spread_curr")) or safe_float(row.get("local_3d_spread_curr"))
        scale_row = scale_by_pair.get(pair_key(row), {})
        q = q_features[idx, : args.feature_dim].astype(np.float64)
        k = k_features[idx, : args.feature_dim].astype(np.float64)
        qn, kn, qk_resid, qk_cos = _norm_qk(q, k)
        prev_conf = safe_float(row.get("prev_sem_conf")) or 0.0
        curr_conf = safe_float(row.get("curr_sem_conf")) or 0.0
        stress_type = _stress_type(row)
        same_label = _same_label(row)
        same_role = _same_role(row)
        q_avail = parse_bool(row.get("feature_q_available"))
        k_avail = parse_bool(row.get("feature_k_available"))
        out = {
            "seq": seq_norm(row.get("seq")),
            "prev_chunk": safe_int(row.get("prev_chunk")),
            "curr_chunk": safe_int(row.get("curr_chunk")),
            "pair_id": row.get("pair_id"),
            "base_case_type": row.get("case_label"),
            "quality_type": row.get("quality_label"),
            "stress_type": stress_type,
            "prev_frame_id": safe_int(row.get("prev_frame_id")),
            "curr_frame_id": safe_int(row.get("curr_frame_id")),
            "prev_patch_id": safe_int(row.get("prev_patch_id")),
            "curr_patch_id": safe_int(row.get("curr_patch_id")),
            "prev_pixel_x": safe_float(row.get("prev_pixel_x")),
            "prev_pixel_y": safe_float(row.get("prev_pixel_y")),
            "curr_pixel_x": safe_float(row.get("curr_pixel_x")),
            "curr_pixel_y": safe_float(row.get("curr_pixel_y")),
            "raw_overlap_residual": safe_float(row.get("raw_overlap_residual")),
            "confidence_weighted_residual": safe_float(row.get("confidence_weighted_residual")),
            "prev_confidence": prev_conf,
            "curr_confidence": curr_conf,
            "either_zero_conf": bool(prev_conf <= 0.0 or curr_conf <= 0.0 or parse_bool(row.get("zero_conf_flag"))),
            "both_zero_conf": bool(prev_conf <= 0.0 and curr_conf <= 0.0),
            "semantic_projection_valid": _semantic_projection_valid(row),
            "same_label": same_label,
            "same_role": same_role,
            "cross_boundary_flag": parse_bool(row.get("cross_boundary_flag")),
            "dynamic_flag": parse_bool(row.get("dynamic_risk_flag")),
            "local_neighbor_count": safe_int(raw_stats.get("local_neighbor_count")),
            "local_shape_log_ratio_median": shape_value,
            "local_shape_log_ratio_iqr": safe_float(raw_stats.get("local_shape_log_ratio_iqr")),
            "local_shape_log_ratio_signed_median": safe_float(raw_stats.get("local_shape_log_ratio_signed_median")),
            "pairwise_distance_ratio_mean": safe_float(raw_stats.get("pairwise_distance_ratio_mean")) or safe_float(row.get("pairwise_distance_ratio_residual")),
            "pairwise_distance_ratio_std": safe_float(raw_stats.get("pairwise_distance_ratio_std")),
            "local_spread_prev": spread_prev,
            "local_spread_curr": spread_curr,
            "local_depth_median": safe_float(raw_stats.get("local_depth_median")),
            "parallax_proxy": safe_float(row.get("parallax_score")),
            "q_feature_available": q_avail,
            "k_feature_available": k_avail,
            "q_feature_source": "v85_direct_pca_swa_current_q",
            "k_feature_source": "v85_direct_pca_swa_cache_k",
            "feature_source_path": row.get("feature_source_path"),
            "layer_id": "pooled_diagnostic",
            "head_id": "pooled_diagnostic",
            "feature_dim": args.feature_dim,
            "q_norm": qn,
            "k_norm": kn,
            "qk_identity_residual": qk_resid,
            "qk_cosine": qk_cos,
            "prev_chunk_sim3_scale_gt": scale_row.get("prev_chunk_sim3_scale_gt"),
            "curr_chunk_sim3_scale_gt": scale_row.get("curr_chunk_sim3_scale_gt"),
            "adjacent_log_scale_jump_gt": scale_row.get("adjacent_log_scale_jump_gt"),
            "abs_log_scale_jump_gt": scale_row.get("abs_log_scale_jump_gt"),
            "scale_jump_high_label": scale_row.get("scale_jump_high_label", ""),
            "full_ATE_pair_context": scale_row.get("full_ATE_pair_context"),
            "per_chunk_oracle_gap_context": scale_row.get("per_chunk_oracle_gap_context", ""),
            "true_route_mass_support": "",
            "true_route_mass_conflict": "",
            "same_mass_random_route_mass": "",
            "semantic_shuffle_route_mass": "",
            "route_entropy_before": "",
            "route_entropy_after": "",
            "merge_weight_support": "",
            "merge_weight_conflict": "",
            "merge_counterfactual_improvement": "",
            "runtime_action_fidelity": "",
            "runtime_geometry_delta": "",
            "local_shape_proxy_source": proxy_source,
            "local_shape_proxy_status": proxy_status,
            "anchor_row_index": int(idx),
        }
        initial.append(out)

    spread_values = [min(v for v in [safe_float(row.get("local_spread_prev")), safe_float(row.get("local_spread_curr"))] if v is not None) for row in initial if safe_float(row.get("local_spread_prev")) is not None and safe_float(row.get("local_spread_curr")) is not None]
    spread_q75 = float(np.quantile(np.asarray(spread_values, dtype=np.float64), 0.75)) if spread_values else 1.0
    shape_q75 = finite_quantile([row.get("local_shape_log_ratio_median") for row in initial], 0.75) or 0.20
    shape_q50 = finite_quantile([row.get("local_shape_log_ratio_median") for row in initial], 0.50) or shape_q75
    overlap_q75 = finite_quantile([row.get("confidence_weighted_residual") or row.get("raw_overlap_residual") for row in initial], 0.75) or 0.20
    overlap_loose = max(overlap_q75 * 2.0, overlap_q75 + 1e-6)

    rows: list[dict[str, Any]] = []
    for row in initial:
        conf = math.sqrt(clamp01(row["prev_confidence"]) * clamp01(row["curr_confidence"]))
        spread = min(safe_float(row.get("local_spread_prev")) or 0.0, safe_float(row.get("local_spread_curr")) or 0.0)
        spread_score = clamp01(spread / max(spread_q75, 1e-12))
        risk_gate = 0.0 if (row["either_zero_conf"] or row["cross_boundary_flag"] or row["dynamic_flag"]) else 1.0
        obs = conf * spread_score * risk_gate
        shape = safe_float(row.get("local_shape_log_ratio_median"))
        overlap = safe_float(row.get("confidence_weighted_residual")) or safe_float(row.get("raw_overlap_residual"))
        sem_support = 1.0 if row["same_label"] and row["same_role"] else (0.5 if row["same_label"] else 0.25)
        support = obs * _exp_score(shape, shape_q75, 0.0) * _exp_score(overlap, overlap_q75, 0.0) * sem_support
        conflict = obs * (1.0 - _exp_score(shape, max(shape_q50, 1e-9), 0.0)) * _exp_score(overlap, overlap_loose, 0.0)
        if row["stress_type"]:
            support = 0.0
            conflict = 0.0
        absence = 1.0 - clamp01((support + conflict) / 0.10)
        state, reason = _state_from_weights(
            stress_type=str(row["stress_type"] or ""),
            obs=obs,
            support=support,
            conflict=conflict,
            absence=absence,
            same_label=bool(row["same_label"]),
            same_role=bool(row["same_role"]),
        )
        out = dict(row)
        out.update(
            {
                "observability_score": obs,
                "support_weight": support,
                "conflict_weight": conflict,
                "absence_weight": absence,
                "state_label": state,
                "state_reason": reason,
                "support_sufficient_dim8": bool(support >= 0.05 and row["q_feature_available"] and row["k_feature_available"]),
                "support_sufficient_dim4": bool(support >= 0.03 and row["q_feature_available"] and row["k_feature_available"]),
                "conflict_high_flag": bool(conflict >= 0.05),
                "absence_high_flag": bool(absence >= 0.50),
                "feature_rank_group": "row_feature_present" if row["q_feature_available"] and row["k_feature_available"] else "row_feature_missing",
            }
        )
        rows.append(out)

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["seq"], int(row["prev_chunk"]), int(row["curr_chunk"]))].append(row)

    by_pair: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        idxs = np.asarray([int(item["anchor_row_index"]) for item in items], dtype=np.int64)
        support_w = np.asarray([float(item["support_weight"] or 0.0) for item in items], dtype=np.float64)
        conflict_w = np.asarray([float(item["conflict_weight"] or 0.0) for item in items], dtype=np.float64)
        obs_w = np.asarray([float(item["observability_score"] or 0.0) for item in items], dtype=np.float64)
        row_w = np.maximum(support_w + conflict_w, 1e-12)
        states = Counter(str(item["state_label"]) for item in items)
        case = Counter(str(item["base_case_type"]) for item in items).most_common(1)[0][0]
        quality = Counter(str(item["quality_type"]) for item in items).most_common(1)[0][0]
        shape_vals = np.asarray([safe_float(item.get("local_shape_log_ratio_median")) or np.nan for item in items], dtype=np.float64)
        signed_shape = np.asarray([safe_float(item.get("local_shape_log_ratio_signed_median")) or np.nan for item in items], dtype=np.float64)
        overlap_vals = np.asarray([safe_float(item.get("confidence_weighted_residual")) or safe_float(item.get("raw_overlap_residual")) or np.nan for item in items], dtype=np.float64)
        depth_vals = np.asarray([safe_float(item.get("local_depth_median")) or np.nan for item in items], dtype=np.float64)
        support_ess = effective_sample_size(support_w)
        conflict_ess = effective_sample_size(conflict_w)
        combined_ess = effective_sample_size(support_w + conflict_w)
        support_rank_q = weighted_rank(q_features[idxs, : args.feature_dim], support_w)
        support_rank_k = weighted_rank(k_features[idxs, : args.feature_dim], support_w)
        support_dim8 = bool(support_ess >= 3 * args.feature_dim and support_rank_q >= int(0.75 * args.feature_dim) and support_rank_k >= int(0.75 * args.feature_dim))
        support_dim4 = bool(support_ess >= 12 and support_rank_q >= 3 and support_rank_k >= 3)
        absence_score = 1.0 - clamp01(combined_ess / 10.0)
        if states.get("STRESS", 0) == len(items):
            pair_state = "STRESS"
        elif combined_ess < 10:
            pair_state = "ABSENCE"
        elif float(np.nansum(conflict_w)) > float(np.nansum(support_w)) * 1.10:
            pair_state = "CONFLICT"
        else:
            pair_state = "SUPPORT"
        by_pair.append(
            {
                "seq": key[0],
                "prev_chunk": key[1],
                "curr_chunk": key[2],
                "base_case_type": case,
                "quality_type": quality,
                "pair_row_count": len(items),
                "state_label": pair_state,
                "support_row_count": states.get("SUPPORT", 0),
                "conflict_row_count": states.get("CONFLICT", 0),
                "absence_row_count": states.get("ABSENCE", 0),
                "stress_row_count": states.get("STRESS", 0),
                "support_mass": float(np.nansum(support_w)),
                "conflict_mass": float(np.nansum(conflict_w)),
                "support_effective_sample_size": support_ess,
                "conflict_effective_sample_size": conflict_ess,
                "support_or_conflict_effective_sample_size": combined_ess,
                "absence_score": absence_score,
                "observability_mean": float(np.nanmean(obs_w)) if obs_w.size else float("nan"),
                "weighted_median_local_shape_log_ratio": _weighted_median(shape_vals, row_w),
                "weighted_median_signed_shape_log_ratio": _weighted_median(signed_shape, row_w),
                "mean_confidence_weighted_overlap_residual": float(np.nanmean(overlap_vals)),
                "median_depth": float(np.nanmedian(depth_vals)) if np.isfinite(depth_vals).any() else float("nan"),
                "q_feature_availability": float(np.mean([bool(item["q_feature_available"]) for item in items])),
                "k_feature_availability": float(np.mean([bool(item["k_feature_available"]) for item in items])),
                "raw_shape_proxy_available_ratio": float(np.mean([item["local_shape_proxy_source"] == "raw_overlap_knn" for item in items])),
                "support_sufficient_dim8": support_dim8,
                "support_sufficient_dim4": support_dim4,
                "support_rank_q": support_rank_q,
                "support_rank_k": support_rank_k,
                "feature_rank_group": _feature_rank_group(min(support_rank_q, support_rank_k), args.feature_dim),
                "prev_chunk_sim3_scale_gt": items[0].get("prev_chunk_sim3_scale_gt"),
                "curr_chunk_sim3_scale_gt": items[0].get("curr_chunk_sim3_scale_gt"),
                "adjacent_log_scale_jump_gt": items[0].get("adjacent_log_scale_jump_gt"),
                "abs_log_scale_jump_gt": items[0].get("abs_log_scale_jump_gt"),
                "scale_jump_high_label": items[0].get("scale_jump_high_label"),
            }
        )

    seq_rows: list[dict[str, Any]] = []
    for seq, items in sorted(defaultdict(list, {seq: [r for r in by_pair if r["seq"] == seq] for seq in {r["seq"] for r in by_pair}}).items()):
        counts = Counter(str(item["state_label"]) for item in items)
        seq_rows.append(
            {
                "seq": seq,
                "adjacent_pair_count": len(items),
                "support_pairs": counts.get("SUPPORT", 0),
                "conflict_pairs": counts.get("CONFLICT", 0),
                "absence_pairs": counts.get("ABSENCE", 0),
                "stress_pairs": counts.get("STRESS", 0),
            }
        )

    hist_rows: list[dict[str, Any]] = []
    for state, sub in sorted(defaultdict(list, {state: [r for r in rows if r["state_label"] == state] for state in {r["state_label"] for r in rows}}).items()):
        vals = np.asarray([safe_float(row.get("local_shape_log_ratio_median")) or np.nan for row in sub], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        hist_rows.append(
            {
                "state_label": state,
                "row_count": len(sub),
                "shape_q25": float(np.quantile(vals, 0.25)) if vals.size else "",
                "shape_q50": float(np.quantile(vals, 0.50)) if vals.size else "",
                "shape_q75": float(np.quantile(vals, 0.75)) if vals.size else "",
                "shape_q90": float(np.quantile(vals, 0.90)) if vals.size else "",
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "scale_conditioned_pair_rows.csv", rows, fields=ROW_FIELDS)
    write_csv(args.out_dir / "scale_conditioned_pair_by_adjacent.csv", by_pair)
    write_csv(args.out_dir / "state_distribution_by_seq.csv", seq_rows)
    write_csv(args.out_dir / "local_shape_histograms.csv", hist_rows)

    missing_report = [
        "# Missing Artifact Report",
        "",
        f"- anchor_rows: `{args.anchor_rows}` exists={args.anchor_rows.exists()}",
        f"- v86_soft_rows: `{args.v86_soft_rows}` exists={v86_soft_exists}",
        f"- features: `{args.features}` exists={args.features.exists()}",
        f"- scale_labels: `{args.scale_labels}` exists={args.scale_labels.exists()}",
        f"- raw_pt_load_issue_counts: `{dict(shape_cache.missing)}`",
        "",
        "Rows with unavailable raw local shape fall back to CSV residual fields and are marked `local_shape_proxy_source=csv_fallback`.",
    ]
    (args.out_dir / "missing_artifact_report.md").write_text("\n".join(missing_report) + "\n", encoding="utf-8")

    summary = {
        "phase": "Phase1_scale_conditioned_pair_universe_build",
        "row_count": len(rows),
        "pair_count": len(by_pair),
        "k_neighbors": args.k_neighbors,
        "patch_radius": args.patch_radius,
        "shape_aggregator": args.shape_aggregator,
        "ratio_mode": args.ratio_mode,
        "feature_dim": args.feature_dim,
        "spread_q75": spread_q75,
        "shape_q50": shape_q50,
        "shape_q75": shape_q75,
        "overlap_q75": overlap_q75,
        "row_state_counts": dict(Counter(row["state_label"] for row in rows)),
        "pair_state_counts": dict(Counter(row["state_label"] for row in by_pair)),
        "raw_shape_proxy_available_ratio": float(np.mean([row["local_shape_proxy_source"] == "raw_overlap_knn" for row in rows])) if rows else 0.0,
        "notes": [
            "Local shape uses raw overlap point neighborhoods from .pt files when available.",
            "Offline scale labels are copied into rows for audit context only and are not used by the state classifier.",
            "STRESS rows are retained but never positive SUPPORT.",
        ],
    }
    write_json(args.out_dir / "support_conflict_absence_summary.json", summary)

    print(f"row_count={len(rows)}")
    print(f"pair_count={len(by_pair)}")
    print(f"raw_shape_proxy_available_ratio={summary['raw_shape_proxy_available_ratio']:.6g}")
    print(f"pair_state_counts={summary['pair_state_counts']}")


if __name__ == "__main__":
    main()
