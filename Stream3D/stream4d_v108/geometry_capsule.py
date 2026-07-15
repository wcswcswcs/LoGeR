from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class GeometryAnchor:
    global_object_id: int
    source_frame_id: int
    point_xyz: tuple[float, float, float]
    source_uv: tuple[float, float]
    depth_m: float
    distance_to_mask_edge_px: float
    visible: bool = True
    occluded: bool = False

    @property
    def is_safe_prompt_anchor(self) -> bool:
        return self.visible and not self.occluded and self.distance_to_mask_edge_px > 0


@dataclass(frozen=True)
class ProjectedAnchor:
    anchor: GeometryAnchor
    target_frame_id: int
    target_uv: tuple[float, float]
    target_depth_m: float
    depth_residual_m: float
    in_frame: bool
    occluded: bool

    @property
    def usable_for_prompt(self) -> bool:
        return self.anchor.is_safe_prompt_anchor and self.in_frame and not self.occluded


@dataclass(frozen=True)
class GeometrySupport:
    depth_valid_fraction: float
    depth_compactness: float | None
    plane_like_support: float | None
    anchor_conflict_count: int

    @property
    def has_basic_support(self) -> bool:
        return self.depth_valid_fraction > 0 and self.anchor_conflict_count == 0


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    mask_b = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask_b)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0, max(bx0 - ax1, ax0 - bx1))
    dy = max(0, max(by0 - ay1, ay0 - by1))
    return float(math.sqrt(dx * dx + dy * dy))


def mask_edge_distance(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(np.asarray(mask).astype(np.uint8), cv2.DIST_L2, 3)


def interior_candidate_mask(
    mask: np.ndarray,
    *,
    min_distance_px: float,
    fallback_quantile: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return an interior mask; fall back to deepest pixels for thin objects."""

    mask_b = np.asarray(mask).astype(bool)
    dist = mask_edge_distance(mask_b)
    if not np.any(mask_b):
        return np.zeros_like(mask_b, dtype=bool), dist, "empty_mask"
    if float(min_distance_px) <= 0.0:
        return mask_b, dist, "full_mask"
    candidate = mask_b & (dist >= float(min_distance_px))
    if np.any(candidate):
        return candidate, dist, "distance_threshold"
    values = dist[mask_b]
    if values.size == 0:
        return np.zeros_like(mask_b, dtype=bool), dist, "empty_distance_values"
    threshold = float(np.quantile(values, min(max(float(fallback_quantile), 0.0), 1.0)))
    fallback = mask_b & (dist >= threshold)
    if np.any(fallback):
        return fallback, dist, "deepest_quantile_fallback"
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    one = np.zeros_like(mask_b, dtype=bool)
    one[int(y), int(x)] = True
    return one, dist, "deepest_pixel_fallback"


def sample_interior_points(
    mask: np.ndarray,
    *,
    count: int,
    min_distance_px: float,
    seed: int,
    max_candidates: int = 2500,
) -> tuple[list[tuple[int, int, float]], dict[str, Any]]:
    candidate, dist, source = interior_candidate_mask(mask, min_distance_px=float(min_distance_px))
    ys, xs = np.where(candidate)
    stats = {
        "requested_count": int(count),
        "candidate_source": source,
        "candidate_pixel_count": int(ys.size),
        "min_distance_px": float(min_distance_px),
        "sampled_count": 0,
    }
    if ys.size == 0 or int(count) <= 0:
        return [], stats
    rng = np.random.default_rng(int(seed))
    coords = np.stack([ys.astype(np.float32), xs.astype(np.float32)], axis=1)
    if coords.shape[0] > int(max_candidates):
        keep = rng.choice(coords.shape[0], size=int(max_candidates), replace=False)
        coords = coords[keep]
    if coords.shape[0] <= int(count):
        selected = coords.astype(np.int64)
    else:
        chosen: list[int] = [int(rng.integers(0, coords.shape[0]))]
        min_dist2 = np.sum((coords - coords[chosen[0]]) ** 2, axis=1)
        while len(chosen) < int(count):
            idx = int(np.argmax(min_dist2))
            chosen.append(idx)
            dist2 = np.sum((coords - coords[idx]) ** 2, axis=1)
            min_dist2 = np.minimum(min_dist2, dist2)
        selected = coords[np.asarray(chosen, dtype=np.int64)].astype(np.int64)
    points = [(int(y), int(x), float(dist[int(y), int(x)])) for y, x in selected.tolist()]
    stats["sampled_count"] = int(len(points))
    return points, stats


def mask_depth_support(
    mask: np.ndarray,
    *,
    depth: np.ndarray,
    depth_conf: np.ndarray | None = None,
    min_depth_conf: float = 0.0,
    core_min_distance_px: float = 0.0,
) -> dict[str, Any]:
    mask_b = np.asarray(mask).astype(bool)
    area = int(np.count_nonzero(mask_b))
    if area <= 0:
        return {
            "mask_area_px": 0,
            "depth_valid_fraction": 0.0,
            "depth_conf_valid_fraction": 0.0,
            "depth_median_m": -1.0,
            "depth_p10_m": -1.0,
            "depth_p90_m": -1.0,
            "depth_range_p10_p90_m": -1.0,
            "core_area_px": 0,
            "core_depth_valid_fraction": 0.0,
            "core_depth_median_m": -1.0,
            "core_depth_range_p10_p90_m": -1.0,
        }
    depth_arr = np.asarray(depth)
    if depth_arr.shape[:2] != mask_b.shape[:2]:
        raise ValueError({"mask_shape": list(mask_b.shape), "depth_shape": list(depth_arr.shape)})
    valid_depth = np.isfinite(depth_arr) & (depth_arr > 0.0)
    if depth_conf is not None and np.asarray(depth_conf).shape[:2] == depth_arr.shape[:2]:
        conf_arr = np.asarray(depth_conf)
        conf_valid = np.isfinite(conf_arr) & (conf_arr >= float(min_depth_conf))
    else:
        conf_valid = np.ones_like(mask_b, dtype=bool)
    valid = mask_b & valid_depth & conf_valid
    values = depth_arr[valid]

    def robust_stats(vals: np.ndarray) -> tuple[float, float, float, float]:
        if vals.size == 0:
            return -1.0, -1.0, -1.0, -1.0
        p10 = float(np.percentile(vals, 10))
        p90 = float(np.percentile(vals, 90))
        return float(np.median(vals)), p10, p90, float(p90 - p10)

    median, p10, p90, rng = robust_stats(values)
    core_mask, _dist, _source = interior_candidate_mask(mask_b, min_distance_px=float(core_min_distance_px))
    core_area = int(np.count_nonzero(core_mask))
    core_valid = core_mask & valid_depth & conf_valid
    core_values = depth_arr[core_valid]
    core_median, _cp10, _cp90, core_rng = robust_stats(core_values)
    return {
        "mask_area_px": int(area),
        "depth_valid_fraction": float(np.count_nonzero(mask_b & valid_depth) / max(area, 1)),
        "depth_conf_valid_fraction": float(np.count_nonzero(mask_b & conf_valid) / max(area, 1)),
        "depth_median_m": float(median),
        "depth_p10_m": float(p10),
        "depth_p90_m": float(p90),
        "depth_range_p10_p90_m": float(rng),
        "core_area_px": int(core_area),
        "core_depth_valid_fraction": float(np.count_nonzero(core_valid) / max(core_area, 1)),
        "core_depth_median_m": float(core_median),
        "core_depth_range_p10_p90_m": float(core_rng),
    }


def _largest_radius_component(coords: np.ndarray, radius_px: float) -> tuple[set[int], int]:
    if coords.shape[0] == 0:
        return set(), 0
    radius2 = float(radius_px) * float(radius_px)
    visited: set[int] = set()
    best: set[int] = set()
    component_count = 0
    for start in range(int(coords.shape[0])):
        if start in visited:
            continue
        component_count += 1
        current: set[int] = set()
        queue: deque[int] = deque([int(start)])
        visited.add(int(start))
        while queue:
            idx = int(queue.popleft())
            current.add(idx)
            delta = coords - coords[idx]
            neighbors = np.where(np.sum(delta * delta, axis=1) <= radius2)[0]
            for neighbor in neighbors.tolist():
                ni = int(neighbor)
                if ni not in visited:
                    visited.add(ni)
                    queue.append(ni)
        if len(current) > len(best):
            best = current
    return best, int(component_count)


def point_conflict_diagnostics(
    positive_xy: list[tuple[float, float]],
    negative_xy: list[tuple[float, float]],
    *,
    negative_radius_px: float,
    positive_cluster_radius_px: float,
    min_positive_points: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "positive_count": int(len(positive_xy)),
        "negative_count": int(len(negative_xy)),
        "negative_radius_px": float(negative_radius_px),
        "positive_cluster_radius_px": float(positive_cluster_radius_px),
        "positive_negative_conflict_count": 0,
        "min_positive_to_negative_distance_px": -1.0,
        "positive_cluster_component_count": 0,
        "positive_cluster_largest_size": 0,
        "positive_cluster_outlier_count": 0,
        "min_positive_points": int(min_positive_points),
    }
    if positive_xy and negative_xy:
        pos = np.asarray(positive_xy, dtype=np.float32).reshape(-1, 2)
        neg = np.asarray(negative_xy, dtype=np.float32).reshape(-1, 2)
        distances = np.sqrt(np.sum((pos[:, None, :] - neg[None, :, :]) ** 2, axis=2))
        nearest = distances.min(axis=1)
        out["positive_negative_conflict_count"] = int(np.count_nonzero(nearest <= float(negative_radius_px)))
        out["min_positive_to_negative_distance_px"] = float(nearest.min()) if nearest.size else -1.0
    if positive_xy and float(positive_cluster_radius_px) > 0.0:
        coords = np.asarray(positive_xy, dtype=np.float32).reshape(-1, 2)
        largest, component_count = _largest_radius_component(coords, float(positive_cluster_radius_px))
        out["positive_cluster_component_count"] = int(component_count)
        out["positive_cluster_largest_size"] = int(len(largest))
        if len(largest) >= max(1, int(min_positive_points)):
            out["positive_cluster_outlier_count"] = int(len(positive_xy) - len(largest))
    return out
