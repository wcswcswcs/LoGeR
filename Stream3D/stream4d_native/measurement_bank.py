from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .object_tube_io import TubeRecord


@dataclass
class MaskMeasurement:
    measurement_id: str
    frame_global: int
    mask_id: int
    tube_ids: list[int]
    inside_tube_ids: list[int]
    mask_area: int = 0
    boundary_tube_ids: list[int] = field(default_factory=list)
    outside_visible_tube_ids: list[int] = field(default_factory=list)
    same_mask_merge_pairs: list[tuple[int, int]] = field(default_factory=list)
    boundary_safe_merge_pairs: list[tuple[int, int]] = field(default_factory=list)
    boundary_crossing_cut_pairs: list[tuple[int, int]] = field(default_factory=list)
    same_frame_different_mask_cannot_link_pairs: list[tuple[int, int]] = field(default_factory=list)
    visible_outside_negative_pairs: list[tuple[int, int]] = field(default_factory=list)
    mask_distance_to_boundary_per_tube: dict[int, float] = field(default_factory=dict)
    mask_eroded_interior_flag_per_tube: dict[int, bool] = field(default_factory=dict)
    mask_boundary_band_flag_per_tube: dict[int, bool] = field(default_factory=dict)
    appearance_feature_per_tube: dict[int, list[float]] = field(default_factory=dict)
    appearance_similarity_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    motion_consistency_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _tube_visible_at(tube: TubeRecord, local_idx: int, *, min_visibility: float, min_confidence: float) -> bool:
    visibility = tube.get_geometry_for_measurement(field="visibility")
    confidence = tube.get_geometry_for_measurement(field="confidence")
    return bool(visibility[local_idx] >= float(min_visibility) and confidence[local_idx] >= float(min_confidence))


def _mask_id_for_tube(
    tube: TubeRecord,
    local_idx: int,
    masks_by_frame: dict[int, np.ndarray] | None,
) -> int:
    if masks_by_frame is None:
        return 1
    frame = int(tube.target_frames_global[local_idx])
    mask = masks_by_frame.get(frame)
    if mask is None:
        return 0
    uv = tube.get_geometry_for_measurement(field="uv")[local_idx]
    height, width = mask.shape[:2]
    x = int(np.clip(np.rint(float(uv[0]) * (width - 1)), 0, width - 1))
    y = int(np.clip(np.rint(float(uv[1]) * (height - 1)), 0, height - 1))
    return int(mask[y, x])


def _limited_pairs(unique_ids: list[int], max_pairs: int, chunk_by_id: dict[int, int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for idx, left in enumerate(unique_ids):
        for right in unique_ids[idx + 1 :]:
            if chunk_by_id.get(int(left)) == chunk_by_id.get(int(right)):
                continue
            pairs.append((int(left), int(right)))
            if len(pairs) >= int(max_pairs):
                return pairs
    for idx, left in enumerate(unique_ids):
        for right in unique_ids[idx + 1 :]:
            if chunk_by_id.get(int(left)) != chunk_by_id.get(int(right)):
                continue
            pairs.append((int(left), int(right)))
            if len(pairs) >= int(max_pairs):
                return pairs
    return pairs


def _limited_bipartite_pairs(left_ids: list[int], right_ids: list[int], max_pairs: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for left in left_ids:
        for right in right_ids:
            pair = tuple(sorted((int(left), int(right))))
            if pair[0] == pair[1] or pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) >= int(max_pairs):
                return pairs
    return pairs


def count_pair_measurement_evidence(
    measurements: list[MaskMeasurement],
    candidate_pairs: set[tuple[int, int]] | list[tuple[int, int]],
) -> dict[tuple[int, int], dict[str, int]]:
    """Count image-space positive/negative evidence for candidate tube pairs."""

    evidence = {
        tuple(sorted((int(left), int(right)))): {
            "same_mask_count": 0,
            "boundary_safe_count": 0,
            "boundary_cross_count": 0,
            "same_frame_cannot_link_count": 0,
            "visible_outside_negative_count": 0,
        }
        for left, right in candidate_pairs
    }
    if not evidence:
        return evidence
    for meas in measurements:
        inside = {int(v) for v in meas.inside_tube_ids}
        outside = {int(v) for v in meas.outside_visible_tube_ids}
        same_mask = {tuple(sorted((int(left), int(right)))) for left, right in meas.same_mask_merge_pairs}
        boundary_safe = {tuple(sorted((int(left), int(right)))) for left, right in meas.boundary_safe_merge_pairs}
        boundary_cross = {tuple(sorted((int(left), int(right)))) for left, right in meas.boundary_crossing_cut_pairs}
        cannot_link = {
            tuple(sorted((int(left), int(right)))) for left, right in meas.same_frame_different_mask_cannot_link_pairs
        }
        visible_outside_pairs = {tuple(sorted((int(left), int(right)))) for left, right in meas.visible_outside_negative_pairs}
        if not inside:
            continue
        for pair, counts in evidence.items():
            left, right = pair
            left_inside = left in inside
            right_inside = right in inside
            if pair in same_mask or (left_inside and right_inside):
                counts["same_mask_count"] += 1
            if pair in boundary_safe:
                counts["boundary_safe_count"] += 1
            if pair in boundary_cross:
                counts["boundary_cross_count"] += 1
            if pair in cannot_link:
                counts["same_frame_cannot_link_count"] += 1
            if ((left_inside and right in outside) or (right_inside and left in outside)) or pair in visible_outside_pairs:
                counts["visible_outside_negative_count"] += 1
    return evidence


def build_measurement_bank(
    tubes: list[TubeRecord],
    *,
    masks_by_frame: dict[int, np.ndarray] | None = None,
    min_visibility: float = 0.5,
    min_confidence: float = 0.5,
    max_pairs_per_measurement: int = 256,
    interior_distance_px: float = 3.0,
    boundary_distance_px: float = 2.0,
    boundary_cross_radius_px: float = 12.0,
) -> tuple[list[MaskMeasurement], dict[str, Any]]:
    """Build image-space measurements without reading any 3D metric geometry."""

    grouped: dict[tuple[int, int], dict[int, tuple[float, int, int]]] = {}
    visible_by_frame: dict[int, set[int]] = {}
    xy_by_frame_tube: dict[tuple[int, int], tuple[int, int]] = {}
    mask_by_frame_tube: dict[tuple[int, int], int] = {}
    distance_cache: dict[tuple[int, int], np.ndarray] = {}
    chunk_by_id = {int(tube.tube_id): int(tube.chunk_id) for tube in tubes}
    for tube in tubes:
        frames = np.asarray(tube.target_frames_global, dtype=np.int64).reshape(-1)
        for local_idx, frame in enumerate(frames.tolist()):
            frame_id = int(frame)
            if not _tube_visible_at(tube, local_idx, min_visibility=min_visibility, min_confidence=min_confidence):
                continue
            uv = tube.get_geometry_for_measurement(field="uv")[local_idx]
            if not (np.isfinite(uv).all() and 0.0 <= float(uv[0]) <= 1.0 and 0.0 <= float(uv[1]) <= 1.0):
                continue
            mask = masks_by_frame.get(frame_id) if masks_by_frame is not None else None
            if mask is not None:
                height, width = mask.shape[:2]
            else:
                height, width = 1, 1
            x = int(np.clip(np.rint(float(uv[0]) * (width - 1)), 0, width - 1))
            y = int(np.clip(np.rint(float(uv[1]) * (height - 1)), 0, height - 1))
            visible_by_frame.setdefault(frame_id, set()).add(int(tube.tube_id))
            xy_by_frame_tube[(frame_id, int(tube.tube_id))] = (x, y)
            mask_id = _mask_id_for_tube(tube, local_idx, masks_by_frame)
            mask_by_frame_tube[(frame_id, int(tube.tube_id))] = int(mask_id)
            if mask_id <= 0 and masks_by_frame is not None:
                continue
            dist_to_boundary = float("inf")
            if mask is not None and int(mask_id) > 0:
                cache_key = (frame_id, int(mask_id))
                if cache_key not in distance_cache:
                    binary = (mask == int(mask_id)).astype(np.uint8)
                    distance_cache[cache_key] = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
                dist_to_boundary = float(distance_cache[cache_key][y, x])
            grouped.setdefault((frame_id, int(mask_id)), {})[int(tube.tube_id)] = (dist_to_boundary, x, y)

    measurements: list[MaskMeasurement] = []
    for (frame, mask_id), tube_entries in sorted(grouped.items()):
        unique_ids = sorted(set(int(v) for v in tube_entries))
        pairs = _limited_pairs(unique_ids, int(max_pairs_per_measurement), chunk_by_id)
        interior_ids = [
            int(tube_id)
            for tube_id in unique_ids
            if float(tube_entries[int(tube_id)][0]) >= float(interior_distance_px)
        ]
        boundary_ids = [
            int(tube_id)
            for tube_id in unique_ids
            if float(tube_entries[int(tube_id)][0]) < float(boundary_distance_px)
        ]
        boundary_safe_pairs = _limited_pairs(interior_ids, int(max_pairs_per_measurement), chunk_by_id)
        outside = sorted(visible_by_frame.get(int(frame), set()) - set(unique_ids))
        visible_negative_pairs = _limited_bipartite_pairs(unique_ids, outside, int(max_pairs_per_measurement))
        different_mask_ids = [
            int(tube_id)
            for tube_id in outside
            if int(mask_by_frame_tube.get((int(frame), int(tube_id)), 0)) > 0
            and int(mask_by_frame_tube.get((int(frame), int(tube_id)), 0)) != int(mask_id)
        ]
        cannot_link_pairs = _limited_bipartite_pairs(unique_ids, different_mask_ids, int(max_pairs_per_measurement))
        boundary_cross_pairs: list[tuple[int, int]] = []
        for inside_id in boundary_ids:
            ix, iy = xy_by_frame_tube.get((int(frame), int(inside_id)), (0, 0))
            close_outside: list[int] = []
            for outside_id in outside:
                ox, oy = xy_by_frame_tube.get((int(frame), int(outside_id)), (10**9, 10**9))
                if float(np.hypot(float(ix - ox), float(iy - oy))) <= float(boundary_cross_radius_px):
                    close_outside.append(int(outside_id))
            boundary_cross_pairs.extend(
                _limited_bipartite_pairs([inside_id], close_outside, int(max_pairs_per_measurement) - len(boundary_cross_pairs))
            )
            if len(boundary_cross_pairs) >= int(max_pairs_per_measurement):
                break
        mask_area = 0
        if masks_by_frame is not None and int(frame) in masks_by_frame:
            mask_area = int(np.count_nonzero(masks_by_frame[int(frame)] == int(mask_id)))
        measurements.append(
            MaskMeasurement(
                measurement_id=f"f{int(frame):06d}_m{int(mask_id):04d}",
                frame_global=int(frame),
                mask_id=int(mask_id),
                tube_ids=unique_ids,
                inside_tube_ids=unique_ids,
                mask_area=mask_area,
                boundary_tube_ids=boundary_ids,
                outside_visible_tube_ids=outside,
                same_mask_merge_pairs=pairs,
                boundary_safe_merge_pairs=boundary_safe_pairs,
                boundary_crossing_cut_pairs=boundary_cross_pairs,
                same_frame_different_mask_cannot_link_pairs=cannot_link_pairs,
                visible_outside_negative_pairs=visible_negative_pairs,
                mask_distance_to_boundary_per_tube={int(tube_id): float(tube_entries[int(tube_id)][0]) for tube_id in unique_ids},
                mask_eroded_interior_flag_per_tube={int(tube_id): int(tube_id) in set(interior_ids) for tube_id in unique_ids},
                mask_boundary_band_flag_per_tube={int(tube_id): int(tube_id) in set(boundary_ids) for tube_id in unique_ids},
                metadata={
                    "pair_count": int(len(pairs)),
                    "boundary_safe_pair_count": int(len(boundary_safe_pairs)),
                    "boundary_cross_pair_count": int(len(boundary_cross_pairs)),
                    "cannot_link_pair_count": int(len(cannot_link_pairs)),
                    "visible_outside_negative_pair_count": int(len(visible_negative_pairs)),
                },
            )
        )

    diagnostics = {
        "tube_count": int(len(tubes)),
        "measurement_count": int(len(measurements)),
        "same_mask_pair_count": int(sum(len(m.same_mask_merge_pairs) for m in measurements)),
        "num_raw_same_mask_pairs": int(sum(len(m.same_mask_merge_pairs) for m in measurements)),
        "num_boundary_safe_merge_pairs": int(sum(len(m.boundary_safe_merge_pairs) for m in measurements)),
        "num_boundary_cross_cut_pairs": int(sum(len(m.boundary_crossing_cut_pairs) for m in measurements)),
        "num_same_frame_cannot_link_pairs": int(sum(len(m.same_frame_different_mask_cannot_link_pairs) for m in measurements)),
        "num_visible_outside_negative_pairs": int(sum(len(m.visible_outside_negative_pairs) for m in measurements)),
        "num_appearance_pairs": int(sum(len(m.appearance_similarity_pairs) for m in measurements)),
        "num_motion_pairs": int(sum(len(m.motion_consistency_pairs) for m in measurements)),
        "measurement_uses_metric_geometry": False,
        "measurement_geometry_fields": ["uv", "visibility", "confidence"],
        "interior_distance_px": float(interior_distance_px),
        "boundary_distance_px": float(boundary_distance_px),
        "boundary_cross_radius_px": float(boundary_cross_radius_px),
    }
    return measurements, diagnostics
