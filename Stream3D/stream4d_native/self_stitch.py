from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

@dataclass
class OverlapMatchResult:
    prev_xyz: np.ndarray
    curr_xyz: np.ndarray
    stats: dict[str, Any]


def scene_scale_from_points(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 2:
        return 1.0
    lo = np.percentile(pts, 5, axis=0)
    hi = np.percentile(pts, 95, axis=0)
    scale = float(np.linalg.norm(hi - lo))
    return max(scale, 1e-6)


def residual_diagnostics(residual: np.ndarray, *, scene_scale: float | None = None, mad_k: float = 3.0) -> dict[str, Any]:
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "residual_median": None,
            "residual_p90": None,
            "residual_p95": None,
            "residual_mad": None,
            "inlier_ratio_abs005": None,
            "inlier_ratio_abs010": None,
            "inlier_ratio_rel001": None,
            "inlier_ratio_rel002": None,
            "inlier_ratio_mad": None,
            "mad_threshold": None,
            "scene_scale": scene_scale,
        }
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    tau_mad = float(median + float(mad_k) * mad)
    scale = max(float(scene_scale if scene_scale is not None else 1.0), 1e-6)
    return {
        "residual_median": median,
        "residual_p90": float(np.percentile(values, 90)),
        "residual_p95": float(np.percentile(values, 95)),
        "residual_mad": mad,
        "inlier_ratio_abs005": float(np.mean(values <= 0.05)),
        "inlier_ratio_abs010": float(np.mean(values <= 0.10)),
        "inlier_ratio_rel001": float(np.mean(values <= 0.01 * scale)),
        "inlier_ratio_rel002": float(np.mean(values <= 0.02 * scale)),
        "inlier_ratio_mad": float(np.mean(values <= tau_mad)),
        "mad_threshold": tau_mad,
        "scene_scale": scale,
    }


def fit_sim3_with_diagnostics(source: np.ndarray, target: np.ndarray, *, scene_scale: float | None = None) -> dict[str, Any]:
    from .sim3 import fit_sim3_umeyama

    fit = fit_sim3_umeyama(source, target)
    residual = np.asarray(fit["residual"], dtype=np.float64)
    metrics = residual_diagnostics(
        residual,
        scene_scale=scene_scale if scene_scale is not None else scene_scale_from_points(target),
    )
    fit.update(metrics)
    # Backward-compatible key, now a true absolute-threshold inlier ratio.
    fit["inlier_ratio"] = fit["inlier_ratio_abs010"]
    return fit


def _optional_array(data: dict[str, np.ndarray], key: str, length: int, default: int = -1) -> np.ndarray:
    if key in data:
        return np.asarray(data[key], dtype=np.int64).reshape(-1)
    return np.full((length,), int(default), dtype=np.int64)


def _match_by_id(
    prev_values: np.ndarray,
    curr_values: np.ndarray,
    prev_positions: np.ndarray,
    curr_positions: np.ndarray,
    used_prev: set[int],
    used_curr: set[int],
    *,
    require_nonnegative: bool,
) -> list[tuple[int, int]]:
    prev_map: dict[int, list[int]] = {}
    for pos, value in zip(prev_positions.tolist(), prev_values.tolist()):
        if int(pos) in used_prev:
            continue
        if require_nonnegative and int(value) < 0:
            continue
        prev_map.setdefault(int(value), []).append(int(pos))
    pairs: list[tuple[int, int]] = []
    for pos, value in zip(curr_positions.tolist(), curr_values.tolist()):
        if int(pos) in used_curr:
            continue
        if require_nonnegative and int(value) < 0:
            continue
        candidates = prev_map.get(int(value), [])
        while candidates and candidates[0] in used_prev:
            candidates.pop(0)
        if not candidates:
            continue
        prev_pos = candidates.pop(0)
        used_prev.add(prev_pos)
        used_curr.add(int(pos))
        pairs.append((prev_pos, int(pos)))
    return pairs


def _source_pixel_keys(data: dict[str, np.ndarray], positions: np.ndarray) -> list[tuple[int, int, int]]:
    src_global = _optional_array(data, "src_frame_global", int(data["xyz"].shape[1]), default=-1)
    src_xy = np.asarray(data.get("src_xy", np.full((data["xyz"].shape[1], 2), -1)), dtype=np.int64).reshape(-1, 2)
    keys: list[tuple[int, int, int]] = []
    for pos in positions.tolist():
        keys.append((int(src_global[int(pos)]), int(src_xy[int(pos), 0]), int(src_xy[int(pos), 1])))
    return keys


def _match_by_source_pixel(
    prev: dict[str, np.ndarray],
    curr: dict[str, np.ndarray],
    prev_positions: np.ndarray,
    curr_positions: np.ndarray,
    used_prev: set[int],
    used_curr: set[int],
) -> list[tuple[int, int]]:
    prev_keys = _source_pixel_keys(prev, prev_positions)
    curr_keys = _source_pixel_keys(curr, curr_positions)
    prev_map: dict[tuple[int, int, int], list[int]] = {}
    for pos, key in zip(prev_positions.tolist(), prev_keys):
        if int(pos) in used_prev or key[0] < 0 or key[1] < 0 or key[2] < 0:
            continue
        prev_map.setdefault(key, []).append(int(pos))
    pairs: list[tuple[int, int]] = []
    for pos, key in zip(curr_positions.tolist(), curr_keys):
        if int(pos) in used_curr or key[0] < 0 or key[1] < 0 or key[2] < 0:
            continue
        candidates = prev_map.get(key, [])
        while candidates and candidates[0] in used_prev:
            candidates.pop(0)
        if not candidates:
            continue
        prev_pos = candidates.pop(0)
        used_prev.add(prev_pos)
        used_curr.add(int(pos))
        pairs.append((prev_pos, int(pos)))
    return pairs


def _match_by_mutual_uv(
    prev_uv: np.ndarray,
    curr_uv: np.ndarray,
    prev_positions: np.ndarray,
    curr_positions: np.ndarray,
    used_prev: set[int],
    used_curr: set[int],
    *,
    uv_radius: float,
) -> tuple[list[tuple[int, int]], int]:
    prev_available = np.asarray([pos for pos in prev_positions.tolist() if int(pos) not in used_prev], dtype=np.int64)
    curr_available = np.asarray([pos for pos in curr_positions.tolist() if int(pos) not in used_curr], dtype=np.int64)
    if prev_available.size == 0 or curr_available.size == 0:
        return [], 0
    p_uv = prev_uv[prev_available]
    c_uv = curr_uv[curr_available]
    c_tree = cKDTree(c_uv)
    p_to_c_dist, p_to_c_idx = c_tree.query(p_uv, k=1, distance_upper_bound=float(uv_radius))
    p_tree = cKDTree(p_uv)
    c_to_p_dist, c_to_p_idx = p_tree.query(c_uv, k=1, distance_upper_bound=float(uv_radius))
    pairs: list[tuple[int, int]] = []
    cycle_candidates = 0
    for p_local, (dist, c_local) in enumerate(zip(p_to_c_dist.tolist(), p_to_c_idx.tolist())):
        if not np.isfinite(dist) or int(c_local) >= curr_available.size:
            continue
        cycle_candidates += 1
        if int(c_to_p_idx[int(c_local)]) != int(p_local) or not np.isfinite(c_to_p_dist[int(c_local)]):
            continue
        p_pos = int(prev_available[int(p_local)])
        c_pos = int(curr_available[int(c_local)])
        if p_pos in used_prev or c_pos in used_curr:
            continue
        used_prev.add(p_pos)
        used_curr.add(c_pos)
        pairs.append((p_pos, c_pos))
    return pairs, cycle_candidates


def _uv_inbounds(uv: np.ndarray) -> np.ndarray:
    values = np.asarray(uv, dtype=np.float64)
    return np.isfinite(values).all(axis=1) & (values[:, 0] >= 0.0) & (values[:, 0] <= 1.0) & (values[:, 1] >= 0.0) & (values[:, 1] <= 1.0)


def _carrier_ids(data: dict[str, np.ndarray], length: int) -> np.ndarray:
    if "carrier_id" in data:
        return np.asarray(data["carrier_id"], dtype=np.int64).reshape(-1)
    return np.full((int(length),), -1, dtype=np.int64)


def match_overlap_carriers(
    prev: dict[str, np.ndarray],
    curr: dict[str, np.ndarray],
    *,
    min_visibility: float = 0.5,
    min_confidence: float = 0.5,
    uv_radius: float = 0.01,
    max_matches_per_frame: int = 512,
) -> OverlapMatchResult:
    prev_xyz_parts: list[np.ndarray] = []
    curr_xyz_parts: list[np.ndarray] = []
    prev_frame_ids = [int(v) for v in prev["frame_ids"]]
    curr_frame_ids = [int(v) for v in curr["frame_ids"]]
    prev_by_frame = {int(frame_id): idx for idx, frame_id in enumerate(prev_frame_ids)}
    curr_by_frame = {int(frame_id): idx for idx, frame_id in enumerate(curr_frame_ids)}
    overlap_frames = sorted(set(prev_by_frame) & set(curr_by_frame))
    stats = {
        "overlap_frame_count": int(len(overlap_frames)),
        "raw_persistent_match_count": 0,
        "raw_carrier_id_match_count": 0,
        "raw_source_pixel_match_count": 0,
        "raw_mutual_uv_match_count": 0,
        "used_persistent_match_count": 0,
        "used_carrier_id_match_count": 0,
        "used_source_pixel_match_count": 0,
        "used_mutual_uv_match_count": 0,
        "raw_total_match_count": 0,
        "used_total_match_count": 0,
        "used_anchor_count": 0,
        "match_source_stable_id_count": 0,
        "match_source_persistent_id_count": 0,
        "match_source_carrier_id_count": 0,
        "match_source_same_source_pixel_count": 0,
        "match_source_mutual_uv_count": 0,
        "match_source_rejected_count": 0,
        "mutual_uv_cycle_candidate_count": 0,
        "candidate_point_count": 0,
        "uv_inbounds_point_count": 0,
        "visibility_confidence_pass_point_count": 0,
        "appearance_consistency_available": False,
        "appearance_consistency_pass_ratio": None,
        "missing_carrier_id_in_prev": bool("carrier_id" not in prev),
        "missing_carrier_id_in_curr": bool("carrier_id" not in curr),
        "default_range_id_detected": False,
    }
    for frame_id in overlap_frames:
        prev_local = prev_by_frame[int(frame_id)]
        curr_local = curr_by_frame[int(frame_id)]
        prev_uv_ok = _uv_inbounds(prev["uv"][prev_local])
        curr_uv_ok = _uv_inbounds(curr["uv"][curr_local])
        stats["candidate_point_count"] += int(prev["xyz"].shape[1] + curr["xyz"].shape[1])
        stats["uv_inbounds_point_count"] += int(np.count_nonzero(prev_uv_ok) + np.count_nonzero(curr_uv_ok))
        p_ok = (
            prev["valid"][prev_local]
            & (prev["visibility"][prev_local] >= float(min_visibility))
            & (prev["confidence"][prev_local] >= float(min_confidence))
            & prev_uv_ok
            & np.isfinite(prev["xyz"][prev_local]).all(axis=1)
        )
        c_ok = (
            curr["valid"][curr_local]
            & (curr["visibility"][curr_local] >= float(min_visibility))
            & (curr["confidence"][curr_local] >= float(min_confidence))
            & curr_uv_ok
            & np.isfinite(curr["xyz"][curr_local]).all(axis=1)
        )
        prev_positions = np.flatnonzero(p_ok)
        curr_positions = np.flatnonzero(c_ok)
        stats["visibility_confidence_pass_point_count"] += int(prev_positions.size + curr_positions.size)
        if prev_positions.size == 0 or curr_positions.size == 0:
            continue
        used_prev: set[int] = set()
        used_curr: set[int] = set()
        prev_persistent = _optional_array(prev, "persistent_tube_id", int(prev["xyz"].shape[1]), default=-1)
        curr_persistent = _optional_array(curr, "persistent_tube_id", int(curr["xyz"].shape[1]), default=-1)
        persistent_pairs = _match_by_id(
            prev_persistent[prev_positions],
            curr_persistent[curr_positions],
            prev_positions,
            curr_positions,
            used_prev,
            used_curr,
            require_nonnegative=True,
        )
        prev_carrier = _carrier_ids(prev, int(prev["xyz"].shape[1]))
        curr_carrier = _carrier_ids(curr, int(curr["xyz"].shape[1]))
        carrier_pairs = _match_by_id(
            prev_carrier[prev_positions],
            curr_carrier[curr_positions],
            prev_positions,
            curr_positions,
            used_prev,
            used_curr,
            require_nonnegative=True,
        )
        source_pairs = _match_by_source_pixel(prev, curr, prev_positions, curr_positions, used_prev, used_curr)
        uv_pairs, uv_cycle_candidates = _match_by_mutual_uv(
            prev["uv"][prev_local],
            curr["uv"][curr_local],
            prev_positions,
            curr_positions,
            used_prev,
            used_curr,
            uv_radius=float(uv_radius),
        )
        tagged_pairs: list[tuple[int, int, str]] = (
            [(p, c, "persistent") for p, c in persistent_pairs]
            + [(p, c, "carrier_id") for p, c in carrier_pairs]
            + [(p, c, "source_pixel") for p, c in source_pairs]
            + [(p, c, "mutual_uv") for p, c in uv_pairs]
        )
        stats["raw_persistent_match_count"] += int(len(persistent_pairs))
        stats["raw_carrier_id_match_count"] += int(len(carrier_pairs))
        stats["raw_source_pixel_match_count"] += int(len(source_pairs))
        stats["raw_mutual_uv_match_count"] += int(len(uv_pairs))
        pairs = tagged_pairs
        if len(pairs) > int(max_matches_per_frame):
            keep = np.linspace(0, len(pairs) - 1, num=int(max_matches_per_frame), dtype=np.int64)
            pairs = [pairs[int(idx)] for idx in keep.tolist()]
        used_counts = {
            "persistent": sum(1 for _, _, kind in pairs if kind == "persistent"),
            "carrier_id": sum(1 for _, _, kind in pairs if kind == "carrier_id"),
            "source_pixel": sum(1 for _, _, kind in pairs if kind == "source_pixel"),
            "mutual_uv": sum(1 for _, _, kind in pairs if kind == "mutual_uv"),
        }
        stats["used_persistent_match_count"] += int(used_counts["persistent"])
        stats["used_carrier_id_match_count"] += int(used_counts["carrier_id"])
        stats["used_source_pixel_match_count"] += int(used_counts["source_pixel"])
        stats["used_mutual_uv_match_count"] += int(used_counts["mutual_uv"])
        stats["match_source_persistent_id_count"] += int(used_counts["persistent"])
        stats["match_source_carrier_id_count"] += int(used_counts["carrier_id"])
        stats["match_source_stable_id_count"] += int(used_counts["persistent"] + used_counts["carrier_id"])
        stats["match_source_same_source_pixel_count"] += int(used_counts["source_pixel"])
        stats["match_source_mutual_uv_count"] += int(used_counts["mutual_uv"])
        stats["mutual_uv_cycle_candidate_count"] += int(uv_cycle_candidates)
        stats["match_source_rejected_count"] += int(max(len(tagged_pairs) - len(pairs), 0))
        if pairs:
            p_idx = np.asarray([item[0] for item in pairs], dtype=np.int64)
            c_idx = np.asarray([item[1] for item in pairs], dtype=np.int64)
            prev_xyz_parts.append(prev["xyz"][prev_local, p_idx])
            curr_xyz_parts.append(curr["xyz"][curr_local, c_idx])
    stats["raw_total_match_count"] = int(
        stats["raw_persistent_match_count"]
        + stats["raw_carrier_id_match_count"]
        + stats["raw_source_pixel_match_count"]
        + stats["raw_mutual_uv_match_count"]
    )
    matched = (
        int(stats["match_source_stable_id_count"])
        + int(stats["match_source_same_source_pixel_count"])
        + int(stats["match_source_mutual_uv_count"])
    )
    stats["used_total_match_count"] = int(matched)
    stats["used_anchor_count"] = int(matched)
    stats["overlap_anchor_count"] = int(matched)
    stats["subsample_rate"] = float(matched / max(int(stats["raw_total_match_count"]), 1))
    stats["used_persistent_ratio"] = float(stats["used_persistent_match_count"] / max(matched, 1))
    stats["used_carrier_id_ratio"] = float(stats["used_carrier_id_match_count"] / max(matched, 1))
    stats["used_source_pixel_ratio"] = float(stats["used_source_pixel_match_count"] / max(matched, 1))
    stats["used_mutual_uv_ratio"] = float(stats["used_mutual_uv_match_count"] / max(matched, 1))
    stats["used_persistent_or_carrier_id_ratio"] = float(stats["match_source_stable_id_count"] / max(matched, 1))
    stats["stable_id_match_ratio"] = float(stats["match_source_stable_id_count"] / max(matched, 1))
    stats["same_source_pixel_match_ratio"] = float(stats["match_source_same_source_pixel_count"] / max(matched, 1))
    stats["mutual_uv_match_ratio"] = float(stats["match_source_mutual_uv_count"] / max(matched, 1))
    stats["cycle_consistency_pass_ratio"] = float(
        stats["match_source_mutual_uv_count"] / max(stats["mutual_uv_cycle_candidate_count"], 1)
    )
    stats["uv_inbounds_ratio"] = float(stats["uv_inbounds_point_count"] / max(stats["candidate_point_count"], 1))
    stats["visibility_confidence_pass_ratio"] = float(
        stats["visibility_confidence_pass_point_count"] / max(stats["candidate_point_count"], 1)
    )
    if not prev_xyz_parts:
        return OverlapMatchResult(
            prev_xyz=np.empty((0, 1, 3), dtype=np.float32),
            curr_xyz=np.empty((0, 1, 3), dtype=np.float32),
            stats=stats,
        )
    return OverlapMatchResult(
        prev_xyz=np.concatenate(prev_xyz_parts, axis=0).reshape(-1, 1, 3).astype(np.float32),
        curr_xyz=np.concatenate(curr_xyz_parts, axis=0).reshape(-1, 1, 3).astype(np.float32),
        stats=stats,
    )
