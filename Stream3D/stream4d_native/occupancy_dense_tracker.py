from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .occupancy_state import OccupancyCoverageTargets, SpatioTemporalOccupancyState


@dataclass(frozen=True)
class QueryBudget:
    max_source_points: int = 4096
    source_points_per_round: int = 256


DecodeFn = Callable[[np.ndarray], list[dict[str, Any]]]


def filter_tracks_before_marking_occupancy(
    tracks: list[dict[str, Any]],
    *,
    min_visibility: float,
    min_confidence: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for track in tracks:
        visibility = np.asarray(track.get("visibility", []), dtype=np.float32)
        confidence = np.asarray(track.get("confidence", []), dtype=np.float32)
        valid = np.asarray(track.get("valid", np.ones_like(visibility, dtype=bool)), dtype=bool)
        if visibility.size == 0 or confidence.size == 0:
            continue
        ok = valid & (visibility >= float(min_visibility)) & (confidence >= float(min_confidence))
        if np.count_nonzero(ok) == 0:
            continue
        item = dict(track)
        item["valid"] = ok
        out.append(item)
    return out


def query_d4rt_tubes_with_spatiotemporal_occupancy(
    *,
    frames: np.ndarray,
    masks: np.ndarray | None,
    decode_source_points: DecodeFn,
    coverage_targets: OccupancyCoverageTargets | None = None,
    query_budget: QueryBudget | None = None,
    warmstart_tracks: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adaptive dense tube extraction driver.

    ``decode_source_points`` is the only model-specific hook. In production it
    should call OpenD4RT encode/decode helpers once per clip; tests can pass a
    deterministic fake decoder.
    """

    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"frames must have shape [T,H,W,C], got {frames.shape}")
    targets = coverage_targets or OccupancyCoverageTargets()
    budget = query_budget or QueryBudget()
    state = SpatioTemporalOccupancyState(
        num_frames=int(frames.shape[0]),
        image_height=int(frames.shape[1]),
        image_width=int(frames.shape[2]),
        masks=masks,
    )
    for track in warmstart_tracks or []:
        state.mark_visible_track_as_visited(
            track=track,
            tube_id=-1,
            mark_radius_px=int(targets.mark_radius_px),
        )
    state.output_tubes = 0

    tubes: list[dict[str, Any]] = []
    t0 = time.time()
    while not state.coverage_satisfied(targets):
        if state.source_queries >= int(budget.max_source_points):
            break
        remaining = max(0, int(budget.max_source_points) - state.source_queries)
        source_points = state.sample_unvisited_source_points(
            batch_size=min(int(budget.source_points_per_round), remaining),
            priority_order=[
                "overlap_anchor_unvisited",
                "large_mask_interior_uncovered",
                "mask_boundary_uncovered",
                "uncertain_region_uncovered",
                "uniform_unvisited",
            ],
        )
        if source_points.shape[0] == 0:
            break
        decoded = decode_source_points(source_points)
        filtered = filter_tracks_before_marking_occupancy(
            decoded,
            min_visibility=float(targets.min_visibility),
            min_confidence=float(targets.min_confidence),
        )
        if not filtered:
            break
        for track in filtered:
            tube_id = len(tubes)
            tubes.append(track)
            state.mark_visible_track_as_visited(
                track=track,
                tube_id=tube_id,
                mark_radius_px=int(targets.mark_radius_px),
            )
        state.round_index += 1

    diagnostics = state.summarize(
        query_budget_hit=state.source_queries >= int(budget.max_source_points),
        total_time_sec=float(time.time() - t0),
    )
    diagnostics["warmstart_track_count"] = int(len(warmstart_tracks or []))
    return tubes, diagnostics
