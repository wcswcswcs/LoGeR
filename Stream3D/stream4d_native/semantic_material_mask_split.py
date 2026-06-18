from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MaterialMaskSplitDiagnostic:
    frame_id: int
    mask_id: int
    input_area: int
    visible_tube_count: int
    split_applied: bool
    output_fragment_count: int
    reason: str


@dataclass(frozen=True)
class MaterialBackfillDiagnostic:
    frame_id: int
    candidate_area: int
    visible_tube_count: int
    selected: bool
    reason: str


def split_masks_by_material_uv(
    masks_by_frame: dict[int, list[tuple[int, np.ndarray]]],
    tubes: list[Any],
    *,
    min_area: int = 32,
    max_splits: int = 3,
    min_tubes: int = 2,
    min_cluster_distance_px: float = 16.0,
    max_mask_area_ratio: float = 0.50,
    min_visibility: float = 0.5,
    min_confidence: float = 0.5,
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict[str, Any]]:
    """Split masks by D4RT material tube UV anchors.

    The splitter uses only image-space tube UV, visibility, and confidence. GT
    labels remain diagnostic-only in the caller.
    """

    anchors_by_frame = _tube_anchors_by_frame(
        tubes,
        min_visibility=float(min_visibility),
        min_confidence=float(min_confidence),
    )
    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    diagnostics: list[MaterialMaskSplitDiagnostic] = []
    for frame_id in sorted(masks_by_frame):
        masks = masks_by_frame.get(int(frame_id), [])
        used_ids = {int(mask_id) for mask_id, _mask in masks}
        next_id = max(used_ids, default=0) + 1
        frame_out: list[tuple[int, np.ndarray]] = []
        frame_anchors = anchors_by_frame.get(int(frame_id), [])
        for mask_id, mask in masks:
            mask_bool = np.asarray(mask, dtype=bool)
            anchors = _anchors_inside_mask(frame_anchors, mask_bool)
            fragments, reason = _split_one_mask(
                mask_bool,
                anchors,
                min_area=int(min_area),
                max_splits=int(max_splits),
                min_tubes=int(min_tubes),
                min_cluster_distance_px=float(min_cluster_distance_px),
                max_mask_area_ratio=float(max_mask_area_ratio),
            )
            if len(fragments) >= 2:
                for fragment in fragments:
                    while next_id in used_ids:
                        next_id += 1
                    frame_out.append((int(next_id), fragment))
                    used_ids.add(int(next_id))
                    next_id += 1
                split_applied = True
            else:
                frame_out.append((int(mask_id), mask_bool))
                split_applied = False
            diagnostics.append(
                MaterialMaskSplitDiagnostic(
                    frame_id=int(frame_id),
                    mask_id=int(mask_id),
                    input_area=int(mask_bool.sum()),
                    visible_tube_count=int(len(anchors)),
                    split_applied=bool(split_applied),
                    output_fragment_count=int(len(fragments) if split_applied else 1),
                    reason=str(reason),
                )
            )
        out[int(frame_id)] = frame_out
    diag_rows = [d.__dict__ for d in diagnostics]
    return out, {
        "input_frame_count": int(len(masks_by_frame)),
        "input_mask_count": int(sum(len(v) for v in masks_by_frame.values())),
        "output_mask_count": int(sum(len(v) for v in out.values())),
        "split_mask_count": int(sum(1 for d in diagnostics if d.split_applied)),
        "created_fragment_count": int(sum(d.output_fragment_count for d in diagnostics if d.split_applied)),
        "total_visible_tube_anchors_inside_masks": int(sum(d.visible_tube_count for d in diagnostics)),
        "min_area": int(min_area),
        "max_splits": int(max_splits),
        "min_tubes": int(min_tubes),
        "min_cluster_distance_px": float(min_cluster_distance_px),
        "max_mask_area_ratio": float(max_mask_area_ratio),
        "min_visibility": float(min_visibility),
        "min_confidence": float(min_confidence),
        "diagnostics": diag_rows,
    }


def backfill_masks_by_material_support(
    primary: dict[int, list[tuple[int, np.ndarray]]],
    supplements: list[dict[int, list[tuple[int, np.ndarray]]]],
    tubes: list[Any],
    *,
    overlap_iou: float = 0.10,
    max_backfill_per_frame: int = 8,
    min_tubes: int = 1,
    max_candidate_area_fraction: float = 1.0,
    min_visibility: float = 0.5,
    min_confidence: float = 0.5,
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict[str, Any]]:
    """Backfill low-overlap masks only when visible D4RT anchors support them."""

    if not tubes:
        raise ValueError("material backfill source requires loadable D4RT records")
    anchors_by_frame = _tube_anchors_by_frame(
        tubes,
        min_visibility=float(min_visibility),
        min_confidence=float(min_confidence),
    )
    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    diagnostics: list[MaterialBackfillDiagnostic] = []
    frame_ids = sorted({int(frame_id) for frame_id in primary} | {int(frame_id) for group in supplements for frame_id in group})
    for frame_id in frame_ids:
        selected: list[tuple[int, np.ndarray]] = [
            (int(mask_id), np.asarray(mask, dtype=bool)) for mask_id, mask in primary.get(int(frame_id), [])
        ]
        next_id = max([int(mask_id) for mask_id, _mask in selected], default=0) + 1
        candidates: list[tuple[int, int, np.ndarray]] = []
        frame_anchors = anchors_by_frame.get(int(frame_id), [])
        for group in supplements:
            for _mask_id, mask in group.get(int(frame_id), []):
                candidate = np.asarray(mask, dtype=bool)
                area = int(candidate.sum())
                area_fraction = float(area / max(int(candidate.size), 1))
                if (
                    float(max_candidate_area_fraction) > 0.0
                    and float(max_candidate_area_fraction) < 1.0
                    and area_fraction > float(max_candidate_area_fraction)
                ):
                    diagnostics.append(
                        MaterialBackfillDiagnostic(
                            frame_id=int(frame_id),
                            candidate_area=area,
                            visible_tube_count=0,
                            selected=False,
                            reason="candidate_area_too_large",
                        )
                    )
                    continue
                if any(_mask_iou(candidate, existing) >= float(overlap_iou) for _existing_id, existing in selected):
                    diagnostics.append(
                        MaterialBackfillDiagnostic(
                            frame_id=int(frame_id),
                            candidate_area=area,
                            visible_tube_count=0,
                            selected=False,
                            reason="overlaps_primary_or_selected",
                        )
                    )
                    continue
                anchors = _anchors_inside_mask(frame_anchors, candidate)
                tube_count = len({int(tube_id) for _x, _y, tube_id in anchors})
                if tube_count < int(min_tubes):
                    diagnostics.append(
                        MaterialBackfillDiagnostic(
                            frame_id=int(frame_id),
                            candidate_area=area,
                            visible_tube_count=int(tube_count),
                            selected=False,
                            reason="insufficient_material_tube_anchors",
                        )
                    )
                    continue
                candidates.append((int(tube_count), area, candidate))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        added = 0
        for tube_count, area, mask in candidates:
            if added >= int(max_backfill_per_frame):
                diagnostics.append(
                    MaterialBackfillDiagnostic(
                        frame_id=int(frame_id),
                        candidate_area=int(area),
                        visible_tube_count=int(tube_count),
                        selected=False,
                        reason="max_backfill_per_frame_reached",
                    )
                )
                continue
            if any(_mask_iou(mask, existing) >= float(overlap_iou) for _existing_id, existing in selected):
                diagnostics.append(
                    MaterialBackfillDiagnostic(
                        frame_id=int(frame_id),
                        candidate_area=int(area),
                        visible_tube_count=int(tube_count),
                        selected=False,
                        reason="overlaps_selected_backfill",
                    )
                )
                continue
            selected.append((int(next_id), mask))
            diagnostics.append(
                MaterialBackfillDiagnostic(
                    frame_id=int(frame_id),
                    candidate_area=int(area),
                    visible_tube_count=int(tube_count),
                    selected=True,
                    reason="selected_by_material_support",
                )
            )
            next_id += 1
            added += 1
        out[int(frame_id)] = selected
    diag_rows = [d.__dict__ for d in diagnostics]
    return out, {
        "primary_mask_count": int(sum(len(v) for v in primary.values())),
        "supplement_group_count": int(len(supplements)),
        "input_supplement_mask_count": int(sum(len(v) for group in supplements for v in group.values())),
        "output_mask_count": int(sum(len(v) for v in out.values())),
        "selected_backfill_count": int(sum(1 for d in diagnostics if d.selected)),
        "candidate_count": int(len(diagnostics)),
        "rejected_overlap_count": int(sum(1 for d in diagnostics if d.reason in {"overlaps_primary_or_selected", "overlaps_selected_backfill"})),
        "rejected_oversize_count": int(sum(1 for d in diagnostics if d.reason == "candidate_area_too_large")),
        "rejected_no_material_support_count": int(
            sum(1 for d in diagnostics if d.reason == "insufficient_material_tube_anchors")
        ),
        "selected_visible_tube_anchor_count": int(sum(d.visible_tube_count for d in diagnostics if d.selected)),
        "total_visible_tube_anchors_inside_candidates": int(sum(d.visible_tube_count for d in diagnostics)),
        "max_backfill_per_frame": int(max_backfill_per_frame),
        "min_tubes": int(min_tubes),
        "max_candidate_area_fraction": float(max_candidate_area_fraction),
        "min_visibility": float(min_visibility),
        "min_confidence": float(min_confidence),
        "diagnostics": diag_rows,
    }


def _tube_anchors_by_frame(
    tubes: list[Any],
    *,
    min_visibility: float,
    min_confidence: float,
) -> dict[int, list[tuple[float, float, int]]]:
    anchors: dict[int, list[tuple[float, float, int]]] = {}
    for tube in tubes:
        frames = np.asarray(getattr(tube, "target_frames_global"), dtype=np.int64).reshape(-1)
        uv = np.asarray(tube.get_geometry_for_measurement(field="uv"), dtype=np.float32)
        visibility = np.asarray(tube.get_geometry_for_measurement(field="visibility"), dtype=np.float32)
        confidence = np.asarray(tube.get_geometry_for_measurement(field="confidence"), dtype=np.float32)
        tube_id = int(getattr(tube, "tube_id", -1))
        for local_idx, frame_id in enumerate(frames.tolist()):
            if local_idx >= uv.shape[0] or local_idx >= visibility.shape[0] or local_idx >= confidence.shape[0]:
                continue
            if float(visibility[local_idx]) < float(min_visibility) or float(confidence[local_idx]) < float(min_confidence):
                continue
            xy = uv[local_idx]
            if not (np.isfinite(xy).all() and 0.0 <= float(xy[0]) <= 1.0 and 0.0 <= float(xy[1]) <= 1.0):
                continue
            anchors.setdefault(int(frame_id), []).append((float(xy[0]), float(xy[1]), tube_id))
    return anchors


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(inter / max(union, 1))


def _anchors_inside_mask(
    anchors: list[tuple[float, float, int]],
    mask: np.ndarray,
) -> list[tuple[float, float, int]]:
    if not anchors:
        return []
    height, width = mask.shape[:2]
    inside: list[tuple[float, float, int]] = []
    for u, v, tube_id in anchors:
        x = int(np.clip(np.rint(float(u) * (width - 1)), 0, width - 1))
        y = int(np.clip(np.rint(float(v) * (height - 1)), 0, height - 1))
        if bool(mask[y, x]):
            inside.append((float(x), float(y), int(tube_id)))
    return inside


def _split_one_mask(
    mask: np.ndarray,
    anchors: list[tuple[float, float, int]],
    *,
    min_area: int,
    max_splits: int,
    min_tubes: int,
    min_cluster_distance_px: float,
    max_mask_area_ratio: float,
) -> tuple[list[np.ndarray], str]:
    mask_bool = np.asarray(mask, dtype=bool)
    mask_area = int(mask_bool.sum())
    if mask_area < int(min_area) * 2:
        return [], "mask_too_small"
    if float(mask_area / max(mask_bool.size, 1)) > float(max_mask_area_ratio):
        return [], "mask_area_ratio_too_large"
    if len(anchors) < int(min_tubes):
        return [], "insufficient_tube_anchors"
    points = np.asarray([[float(x), float(y)] for x, y, _tube_id in anchors], dtype=np.float32)
    unique_points = np.unique(np.rint(points).astype(np.int32), axis=0)
    if unique_points.shape[0] < 2:
        return [], "tube_anchors_not_spatially_distinct"
    k = min(int(max_splits), int(unique_points.shape[0]), max(2, int(len(anchors) // max(int(min_tubes), 1))))
    if k < 2:
        k = 2
    labels = _deterministic_kmeans(points, k=k, iters=16)
    centers: list[np.ndarray] = []
    for label in range(k):
        selected = points[labels == int(label)]
        if selected.shape[0] == 0:
            continue
        centers.append(selected.mean(axis=0))
    if len(centers) < 2:
        return [], "single_material_cluster"
    center_array = np.asarray(centers, dtype=np.float32)
    center_dist = np.sqrt(((center_array[:, None, :] - center_array[None, :, :]) ** 2).sum(axis=2))
    if float(np.max(center_dist)) < float(min_cluster_distance_px):
        return [], "material_clusters_too_close"

    yy, xx = np.nonzero(mask_bool)
    pixels = np.stack([xx.astype(np.float32), yy.astype(np.float32)], axis=1)
    dist = ((pixels[:, None, :] - center_array[None, :, :]) ** 2).sum(axis=2)
    pixel_labels = np.argmin(dist, axis=1)
    label_map = np.full(mask_bool.shape, -1, dtype=np.int16)
    label_map[yy, xx] = pixel_labels.astype(np.int16)
    fragments: list[np.ndarray] = []
    covered = np.zeros_like(mask_bool, dtype=bool)
    for label in range(len(centers)):
        candidate = (label_map == int(label)) & mask_bool
        if int(candidate.sum()) < int(min_area):
            continue
        fragments.append(candidate)
        covered |= candidate
    residual = mask_bool & ~covered
    if int(residual.sum()) >= int(min_area):
        fragments.append(residual)
    if len(fragments) < 2:
        return [], "split_fragments_below_min_area"
    covered_area = int(np.logical_or.reduce(fragments).sum()) if fragments else 0
    if covered_area < int(mask_bool.sum()) * 0.90:
        return [], "split_covered_area_too_low"
    return fragments, "split_by_material_uv"


def _deterministic_kmeans(vectors: np.ndarray, *, k: int, iters: int) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if int(k) <= 1 or x.shape[0] == 1:
        return np.zeros((x.shape[0],), dtype=np.int64)
    centered = x - x.mean(axis=0, keepdims=True)
    axis = int(np.argmax(np.var(centered, axis=0)))
    proj = centered[:, axis]
    seeds: list[int] = []
    for q in np.linspace(0.0, 1.0, num=int(k), dtype=np.float32):
        target = float(np.quantile(proj, float(q)))
        index = int(np.argmin(np.abs(proj - target)))
        if index not in seeds:
            seeds.append(index)
    while len(seeds) < int(k):
        remaining = [idx for idx in range(x.shape[0]) if idx not in seeds]
        if not remaining:
            break
        dist = ((x[remaining, None, :] - x[seeds][None, :, :]) ** 2).sum(axis=2).min(axis=1)
        seeds.append(int(remaining[int(np.argmax(dist))]))
    centers = x[seeds[: int(k)]].copy()
    labels = np.zeros((x.shape[0],), dtype=np.int64)
    for _ in range(int(iters)):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dist, axis=1).astype(np.int64)
        for label in range(centers.shape[0]):
            selected = x[labels == label]
            if selected.shape[0] > 0:
                centers[label] = selected.mean(axis=0)
    return labels
