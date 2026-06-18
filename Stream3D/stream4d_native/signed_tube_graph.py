from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import json
import numpy as np

from .measurement_bank import MaskMeasurement
from .object_tube_io import MergeGeometryError, TubeRecord


@dataclass
class TubeGraphEdge:
    tube_i: int
    tube_j: int
    sign: int
    score: float
    distance: float
    threshold: float
    measurement_id: str
    guard_event: dict[str, Any]


@dataclass
class TubeGraphResult:
    edges: list[TubeGraphEdge]
    blocked_events: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def _finite_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return pts[np.isfinite(pts).all(axis=1)]


def _representative_point(points: np.ndarray) -> np.ndarray | None:
    pts = _finite_points(points)
    if pts.size == 0:
        return None
    return np.median(pts, axis=0).astype(np.float32)


def _spacing_scale(tubes: list[TubeRecord]) -> float:
    reps = []
    for tube in tubes:
        pts = tube.xyz_canonical
        if pts is None:
            continue
        rep = _representative_point(pts)
        if rep is not None:
            reps.append(rep)
    if len(reps) < 3:
        return 1.0
    arr = np.asarray(reps, dtype=np.float32)
    if arr.shape[0] > 512:
        idx = np.linspace(0, arr.shape[0] - 1, 512, dtype=np.int64)
        arr = arr[idx]
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist[dist == 0.0] = np.nan
    nearest = np.nanmin(dist, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 1.0
    return float(max(np.median(nearest), 1e-6))


def _emit(event_logger: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    if event_logger is not None:
        event_logger(dict(event))


def _parse_guard_error(exc: MergeGeometryError) -> dict[str, Any]:
    try:
        return json.loads(str(exc))
    except json.JSONDecodeError:
        return {"guard_pass": False, "guard_reason": str(exc)}


def build_signed_tube_graph(
    tubes: list[TubeRecord],
    measurements: list[MaskMeasurement],
    *,
    context: str = "v25_native_signed_tube_graph",
    same_chunk_only: bool = False,
    threshold_alpha: float = 2.0,
    event_logger: Callable[[dict[str, Any]], None] | None = None,
) -> TubeGraphResult:
    """Build a signed tube graph from guarded metric reads."""

    by_id = {int(t.tube_id): t for t in tubes}
    spacing = _spacing_scale(tubes)
    threshold = float(threshold_alpha) * float(spacing)
    edges: list[TubeGraphEdge] = []
    blocked: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for meas in measurements:
        for tube_i, tube_j in meas.same_mask_merge_pairs:
            pair = tuple(sorted((int(tube_i), int(tube_j))))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            left = by_id.get(pair[0])
            right = by_id.get(pair[1])
            if left is None or right is None:
                continue
            if same_chunk_only and int(left.chunk_id) != int(right.chunk_id):
                event = {
                    "event_type": "metric_merge_blocked",
                    "guard_reason": "same_chunk_only_mode",
                    "tube_i": int(left.tube_id),
                    "tube_j": int(right.tube_id),
                    "chunk_i": int(left.chunk_id),
                    "chunk_j": int(right.chunk_id),
                    "measurement_id": meas.measurement_id,
                    "distance_threshold_type": "spacing_normalized",
                    "distance_threshold": threshold,
                }
                blocked.append(event)
                _emit(event_logger, event)
                continue
            try:
                geom_i, geom_j, guard = left.get_geometry_for_merge(
                    right,
                    context,
                    merge_type="metric_edge",
                )
            except MergeGeometryError as exc:
                event = _parse_guard_error(exc)
                event.update(
                    {
                        "event_type": "metric_merge_blocked",
                        "measurement_id": meas.measurement_id,
                        "distance_threshold_type": "spacing_normalized",
                        "distance_threshold": threshold,
                    }
                )
                blocked.append(event)
                _emit(event_logger, event)
                continue
            rep_i = _representative_point(geom_i)
            rep_j = _representative_point(geom_j)
            if rep_i is None or rep_j is None:
                continue
            distance = float(np.linalg.norm(rep_i - rep_j))
            score = max(0.0, 1.0 - distance / max(threshold, 1e-6))
            guard.update(
                {
                    "event_type": "metric_merge_read",
                    "measurement_id": meas.measurement_id,
                    "distance": distance,
                    "distance_threshold": threshold,
                    "distance_threshold_type": "spacing_normalized",
                    "spacing_median": spacing,
                    "threshold_alpha": float(threshold_alpha),
                }
            )
            _emit(event_logger, guard)
            if distance <= threshold:
                edges.append(
                    TubeGraphEdge(
                        tube_i=int(pair[0]),
                        tube_j=int(pair[1]),
                        sign=1,
                        score=float(score),
                        distance=distance,
                        threshold=threshold,
                        measurement_id=meas.measurement_id,
                        guard_event=dict(guard),
                    )
                )
    diagnostics = {
        "tube_count": int(len(tubes)),
        "candidate_pair_count": int(len(seen_pairs)),
        "positive_edge_count": int(len(edges)),
        "blocked_event_count": int(len(blocked)),
        "spacing_median": float(spacing),
        "threshold_alpha": float(threshold_alpha),
        "distance_threshold": float(threshold),
        "distance_threshold_type": "spacing_normalized",
        "metric_read_event_count": int(len(seen_pairs) - len(blocked)),
        "cross_chunk_edge_count": int(
            sum(1 for edge in edges if int(by_id[edge.tube_i].chunk_id) != int(by_id[edge.tube_j].chunk_id))
        ),
    }
    return TubeGraphResult(edges=edges, blocked_events=blocked, diagnostics=diagnostics)
