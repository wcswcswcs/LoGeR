#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_v105_fullscene_local2history_stitch import (  # type: ignore
    REPO_ROOT,
    _boundary_matches,
    _make_sheet,
    _numeric_stem,
    _overlay_label,
    _put_text,
    _read_json,
    _read_label,
    _rel,
    _remap_label,
    _scene_chunks,
    _sha256,
    _write_json,
)


DEFAULT_FULLSCENE_ROOT = REPO_ROOT / "Stream3D/outputs/audit/v105_fullscene_multichunk_repair_20260711"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Stream3D/outputs/audit/v105_fullscene_l2h_reappearance_repair_20260711"
DEFAULT_VARIANT_ID = "P8_l2h_reappearance_idonly_v1"


def _read_rgb(scene_color_dir: Path, frame_id: int) -> np.ndarray | None:
    rgb = cv2.imread(str(scene_color_dir / f"{int(frame_id)}.jpg"), cv2.IMREAD_COLOR)
    if rgb is None:
        rgb = cv2.imread(str(scene_color_dir / f"{int(frame_id)}.png"), cv2.IMREAD_COLOR)
    return rgb


def _hist_for_mask(rgb_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask.astype(np.uint8), [12, 4, 4], [0, 180, 0, 256, 0, 256])
    flat = hist.reshape(-1).astype(np.float32)
    total = float(flat.sum())
    if total > 0:
        flat /= total
    return flat


def _build_descriptors(
    *,
    label_items: list[tuple[int, np.ndarray]],
    scene_color_dir: Path,
    min_area: int,
) -> dict[int, dict[str, Any]]:
    accum: dict[int, dict[str, Any]] = {}
    for frame_id, label in label_items:
        rgb = _read_rgb(scene_color_dir, frame_id)
        if rgb is None:
            continue
        height, width = label.shape[:2]
        for label_id in [int(v) for v in np.unique(label) if int(v) > 0]:
            mask = label == label_id
            area = int(mask.sum())
            if area < min_area:
                continue
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            hist = _hist_for_mask(rgb, mask)
            cx = float(xs.mean()) / max(float(width), 1.0)
            cy = float(ys.mean()) / max(float(height), 1.0)
            bbox_norm = [
                float(xs.min()) / max(float(width), 1.0),
                float(ys.min()) / max(float(height), 1.0),
                float(xs.max() + 1) / max(float(width), 1.0),
                float(ys.max() + 1) / max(float(height), 1.0),
            ]
            bw = float(xs.max() - xs.min() + 1) / max(float(width), 1.0)
            bh = float(ys.max() - ys.min() + 1) / max(float(height), 1.0)
            rec = accum.setdefault(
                label_id,
                {
                    "label_id": label_id,
                    "frames": 0,
                    "area_weight": 0.0,
                    "hist": np.zeros_like(hist),
                    "cx": 0.0,
                    "cy": 0.0,
                    "bbox_w": 0.0,
                    "bbox_h": 0.0,
                    "mean_area": 0.0,
                    "frame_ids": [],
                    "frame_stats": {},
                },
            )
            w = float(area)
            rec["frames"] += 1
            rec["area_weight"] += w
            rec["hist"] += hist * w
            rec["cx"] += cx * w
            rec["cy"] += cy * w
            rec["bbox_w"] += bw * w
            rec["bbox_h"] += bh * w
            rec["mean_area"] += float(area)
            rec["frame_ids"].append(int(frame_id))
            rec["frame_stats"][str(int(frame_id))] = {
                "area": int(area),
                "bbox_norm": bbox_norm,
                "cx": float(cx),
                "cy": float(cy),
            }

    descriptors: dict[int, dict[str, Any]] = {}
    for label_id, rec in accum.items():
        weight = max(float(rec["area_weight"]), 1.0)
        hist = rec["hist"].astype(np.float32)
        total = float(hist.sum())
        if total > 0:
            hist /= total
        frames = max(int(rec["frames"]), 1)
        descriptors[int(label_id)] = {
            "label_id": int(label_id),
            "frames": int(rec["frames"]),
            "area_weight": float(rec["area_weight"]),
            "hist": hist,
            "cx": float(rec["cx"]) / weight,
            "cy": float(rec["cy"]) / weight,
            "bbox_w": float(rec["bbox_w"]) / weight,
            "bbox_h": float(rec["bbox_h"]) / weight,
            "mean_area": float(rec["mean_area"]) / float(frames),
            "frame_ids": [int(v) for v in rec["frame_ids"]],
            "frame_stats": rec["frame_stats"],
        }
    return descriptors


def _hist_intersection(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.minimum(a, b).sum())


def _ratio_sim(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return float(math.exp(-abs(math.log(float(a) / float(b)))))


def _descriptor_candidate(prev: dict[str, Any], curr: dict[str, Any]) -> dict[str, Any]:
    color = _hist_intersection(prev["hist"], curr["hist"])
    dx = float(prev["cx"]) - float(curr["cx"])
    dy = float(prev["cy"]) - float(curr["cy"])
    dist = math.sqrt(dx * dx + dy * dy)
    spatial = float(math.exp(-((dist / 0.42) ** 2)))
    area = _ratio_sim(float(prev["mean_area"]), float(curr["mean_area"]))
    width = _ratio_sim(float(prev["bbox_w"]), float(curr["bbox_w"]))
    height = _ratio_sim(float(prev["bbox_h"]), float(curr["bbox_h"]))
    shape = 0.5 * (width + height)
    score = 0.48 * color + 0.22 * spatial + 0.20 * area + 0.10 * shape
    return {
        "prev_global_id": int(prev["label_id"]),
        "curr_local_id": int(curr["label_id"]),
        "appearance_score": float(score),
        "color_intersection": float(color),
        "spatial_similarity": float(spatial),
        "area_similarity": float(area),
        "shape_similarity": float(shape),
        "center_distance_norm": float(dist),
        "prev_frames": int(prev["frames"]),
        "curr_frames": int(curr["frames"]),
        "prev_mean_area": float(prev["mean_area"]),
        "curr_mean_area": float(curr["mean_area"]),
        "witness": "multi_frame_rgb_hsv_shape_spatial",
    }


def _bbox_iou_norm(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else 0.0


def _bbox_gap_norm(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    dx = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0.0, max(ay1, by1) - min(ay2, by2))
    return float(math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0))


def _current_object_pair_witness(
    *,
    curr_desc: dict[int, dict[str, Any]],
    candidate_curr_id: int,
    anchor_curr_id: int,
    min_bbox_iou: float,
    max_bbox_gap_norm: float,
    max_center_dist_norm: float,
) -> dict[str, Any]:
    cand = curr_desc.get(int(candidate_curr_id))
    anchor = curr_desc.get(int(anchor_curr_id))
    if cand is None or anchor is None:
        return {
            "accepted": False,
            "reason": "missing_candidate_or_anchor_descriptor",
            "candidate_curr_local_id": int(candidate_curr_id),
            "anchor_curr_local_id": int(anchor_curr_id),
        }
    cand_stats = cand.get("frame_stats", {})
    anchor_stats = anchor.get("frame_stats", {})
    common_frames = sorted(set(str(k) for k in cand_stats.keys()) & set(str(k) for k in anchor_stats.keys()))
    if not common_frames:
        return {
            "accepted": False,
            "reason": "no_common_descriptor_frame",
            "candidate_curr_local_id": int(candidate_curr_id),
            "anchor_curr_local_id": int(anchor_curr_id),
            "candidate_frame_ids": [int(v) for v in cand.get("frame_ids", [])],
            "anchor_frame_ids": [int(v) for v in anchor.get("frame_ids", [])],
        }

    best: dict[str, Any] | None = None
    for frame_id_s in common_frames:
        cstats = cand_stats[frame_id_s]
        astats = anchor_stats[frame_id_s]
        cbox = [float(v) for v in cstats.get("bbox_norm", [])]
        abox = [float(v) for v in astats.get("bbox_norm", [])]
        if len(cbox) != 4 or len(abox) != 4:
            continue
        bbox_iou = _bbox_iou_norm(cbox, abox)
        bbox_gap = _bbox_gap_norm(cbox, abox)
        cx = float(cstats.get("cx", 0.0)) - float(astats.get("cx", 0.0))
        cy = float(cstats.get("cy", 0.0)) - float(astats.get("cy", 0.0))
        center_dist = float(math.sqrt(cx * cx + cy * cy) / math.sqrt(2.0))
        accepted = (
            bbox_iou >= float(min_bbox_iou)
            or bbox_gap <= float(max_bbox_gap_norm)
            or center_dist <= float(max_center_dist_norm)
        )
        row = {
            "frame_id": int(frame_id_s),
            "bbox_iou": float(bbox_iou),
            "bbox_gap_norm": float(bbox_gap),
            "center_dist_norm": float(center_dist),
            "candidate_area": int(cstats.get("area", 0)),
            "anchor_area": int(astats.get("area", 0)),
            "accepted": bool(accepted),
            "reason": "same_frame_bbox_or_center_support" if accepted else "same_frame_geometry_below_threshold",
        }
        score = (
            (1.0 if accepted else 0.0),
            float(bbox_iou),
            -float(bbox_gap),
            -float(center_dist),
            int(cstats.get("area", 0)),
        )
        if best is None or score > best["_score"]:
            best = {**row, "_score": score}
    if best is None:
        return {
            "accepted": False,
            "reason": "invalid_common_frame_geometry",
            "candidate_curr_local_id": int(candidate_curr_id),
            "anchor_curr_local_id": int(anchor_curr_id),
            "common_frame_count": int(len(common_frames)),
        }
    best.pop("_score", None)
    return {
        **best,
        "candidate_curr_local_id": int(candidate_curr_id),
        "anchor_curr_local_id": int(anchor_curr_id),
        "common_frame_count": int(len(common_frames)),
        "min_bbox_iou": float(min_bbox_iou),
        "max_bbox_gap_norm": float(max_bbox_gap_norm),
        "max_center_dist_norm": float(max_center_dist_norm),
    }


def _appearance_matches(
    *,
    prev_items: list[tuple[int, np.ndarray]],
    curr_items: list[tuple[int, np.ndarray]],
    scene_color_dir: Path,
    existing_mapping: dict[int, int],
    min_area: int,
    min_score: float,
    min_color: float,
    min_margin: float,
    allow_part_merge: bool,
    part_merge_min_score: float,
    part_merge_min_color: float,
    part_merge_min_spatial: float,
    weak_existing_prev_to_curr: dict[int, int] | None = None,
    allow_weak_overlap_override: bool = False,
    override_min_score: float = 0.66,
    override_min_color: float = 0.50,
    override_min_spatial: float = 0.25,
    tiny_lock_prev_details: dict[int, dict[str, Any]] | None = None,
    allow_tiny_lock_expansion: bool = False,
    tiny_lock_expansion_min_score: float = 0.40,
    tiny_lock_expansion_min_color: float = 0.15,
    tiny_lock_expansion_min_spatial: float = 0.35,
    tiny_lock_expansion_min_area_ratio_vs_lock: float = 1.20,
    tiny_lock_expansion_max_per_prev: int = 3,
    require_relaxed_merge_object_witness: bool = False,
    relaxed_merge_min_bbox_iou: float = 0.01,
    relaxed_merge_max_bbox_gap_norm: float = 0.03,
    relaxed_merge_max_center_dist_norm: float = 0.16,
) -> tuple[dict[int, int], dict[str, Any]]:
    prev_desc = _build_descriptors(label_items=prev_items, scene_color_dir=scene_color_dir, min_area=min_area)
    curr_desc = _build_descriptors(label_items=curr_items, scene_color_dir=scene_color_dir, min_area=min_area)
    all_candidates: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for curr_id, curr in curr_desc.items():
        if int(curr_id) in existing_mapping:
            continue
        for prev_id, prev in prev_desc.items():
            row = _descriptor_candidate(prev, curr)
            all_candidates.append(row)
            if float(row["appearance_score"]) >= min_score and float(row["color_intersection"]) >= min_color:
                candidates.append(row)
    candidates.sort(
        key=lambda row: (
            float(row["appearance_score"]),
            float(row["color_intersection"]),
            float(row["area_similarity"]),
        ),
        reverse=True,
    )
    by_curr: dict[int, list[dict[str, Any]]] = {}
    by_prev: dict[int, list[dict[str, Any]]] = {}
    for row in candidates:
        by_curr.setdefault(int(row["curr_local_id"]), []).append(row)
        by_prev.setdefault(int(row["prev_global_id"]), []).append(row)

    mapping: dict[int, int] = {}
    accepted: list[dict[str, Any]] = []
    used_prev: set[int] = set()
    used_curr: set[int] = set()
    existing_prev = set(int(v) for v in existing_mapping.values())
    weak_existing_prev_to_curr = weak_existing_prev_to_curr or {}
    part_merges: list[dict[str, Any]] = []
    overrides: list[dict[str, Any]] = []
    tiny_lock_expansions: list[dict[str, Any]] = []
    rejected_part_merges: list[dict[str, Any]] = []
    rejected_tiny_lock_expansions: list[dict[str, Any]] = []
    override_drop_curr_ids: list[int] = []
    existing_curr_by_prev: dict[int, int] = {}
    for curr_id, prev_id in sorted(existing_mapping.items()):
        existing_curr_by_prev.setdefault(int(prev_id), int(curr_id))
    for row in candidates:
        curr_id = int(row["curr_local_id"])
        prev_id = int(row["prev_global_id"])
        if curr_id in used_curr or prev_id in used_prev or prev_id in existing_prev:
            continue
        curr_list = by_curr.get(curr_id, [])
        prev_list = by_prev.get(prev_id, [])
        curr_margin = float(row["appearance_score"]) - float(curr_list[1]["appearance_score"]) if len(curr_list) > 1 else 1.0
        prev_margin = float(row["appearance_score"]) - float(prev_list[1]["appearance_score"]) if len(prev_list) > 1 else 1.0
        if curr_margin < min_margin or prev_margin < min_margin:
            row = {**row, "rejected_reason": "low_top1_margin", "curr_margin": curr_margin, "prev_margin": prev_margin}
            continue
        mapping[curr_id] = prev_id
        used_curr.add(curr_id)
        used_prev.add(prev_id)
        accepted.append({**row, "curr_margin": curr_margin, "prev_margin": prev_margin})

    if allow_weak_overlap_override:
        for row in candidates:
            curr_id = int(row["curr_local_id"])
            prev_id = int(row["prev_global_id"])
            old_curr_id = weak_existing_prev_to_curr.get(prev_id)
            if old_curr_id is None:
                continue
            if curr_id in used_curr or curr_id in existing_mapping:
                continue
            if float(row["appearance_score"]) < float(override_min_score):
                continue
            if float(row["color_intersection"]) < float(override_min_color):
                continue
            if float(row["spatial_similarity"]) < float(override_min_spatial):
                continue
            mapping[curr_id] = prev_id
            used_curr.add(curr_id)
            override_drop_curr_ids.append(int(old_curr_id))
            overrides.append(
                {
                    **row,
                    "witness": "weak_overlap_override_by_multi_frame_appearance",
                    "overridden_curr_local_id": int(old_curr_id),
                    "override_note": (
                        "A tiny-overlap match had locked this history ID to a small current fragment; a stronger "
                        "multi-frame appearance witness reassigns the history ID. This is ID-only."
                    ),
                }
            )

    if allow_part_merge:
        for row in candidates:
            curr_id = int(row["curr_local_id"])
            prev_id = int(row["prev_global_id"])
            if curr_id in used_curr or curr_id in existing_mapping or prev_id not in existing_prev:
                continue
            if float(row["appearance_score"]) < float(part_merge_min_score):
                continue
            if float(row["color_intersection"]) < float(part_merge_min_color):
                continue
            if float(row["spatial_similarity"]) < float(part_merge_min_spatial):
                continue
            anchor_curr_id = int(existing_curr_by_prev.get(prev_id, 0) or 0)
            object_witness = _current_object_pair_witness(
                curr_desc=curr_desc,
                candidate_curr_id=curr_id,
                anchor_curr_id=anchor_curr_id,
                min_bbox_iou=float(relaxed_merge_min_bbox_iou),
                max_bbox_gap_norm=float(relaxed_merge_max_bbox_gap_norm),
                max_center_dist_norm=float(relaxed_merge_max_center_dist_norm),
            )
            if require_relaxed_merge_object_witness and not bool(object_witness.get("accepted")):
                rejected_part_merges.append(
                    {
                        **row,
                        "witness": "duplicate_part_merge_rejected_by_current_object_geometry",
                        "object_witness": object_witness,
                    }
                )
                continue
            mapping[curr_id] = prev_id
            used_curr.add(curr_id)
            part_merges.append(
                {
                    **row,
                    "witness": "duplicate_part_merge_multi_frame_appearance",
                    "object_witness": object_witness,
                    "part_merge_note": (
                        "Current local ID is merged into an already inherited history ID. This is ID-only and does not "
                        "modify local mask geometry."
                    ),
                }
            )

    if allow_tiny_lock_expansion:
        tiny_lock_prev_details = tiny_lock_prev_details or {}
        all_candidates.sort(
            key=lambda row: (
                float(row["appearance_score"]),
                float(row["color_intersection"]),
                float(row["spatial_similarity"]),
                float(row["curr_mean_area"]),
            ),
            reverse=True,
        )
        expansion_count_by_prev: dict[int, int] = {}
        for row in all_candidates:
            curr_id = int(row["curr_local_id"])
            prev_id = int(row["prev_global_id"])
            lock = tiny_lock_prev_details.get(prev_id)
            if lock is None:
                continue
            if curr_id in used_curr or curr_id in existing_mapping:
                continue
            if float(row["appearance_score"]) < float(tiny_lock_expansion_min_score):
                continue
            if float(row["color_intersection"]) < float(tiny_lock_expansion_min_color):
                continue
            if float(row["spatial_similarity"]) < float(tiny_lock_expansion_min_spatial):
                continue
            locked_curr_id = int(lock.get("curr_local_id", 0) or 0)
            locked_desc = curr_desc.get(locked_curr_id)
            locked_area = float(locked_desc.get("mean_area", 0.0)) if locked_desc else float(lock.get("curr_area") or 0.0)
            if locked_area > 0 and float(row["curr_mean_area"]) < locked_area * float(tiny_lock_expansion_min_area_ratio_vs_lock):
                continue
            if expansion_count_by_prev.get(prev_id, 0) >= int(tiny_lock_expansion_max_per_prev):
                continue
            object_witness = _current_object_pair_witness(
                curr_desc=curr_desc,
                candidate_curr_id=curr_id,
                anchor_curr_id=locked_curr_id,
                min_bbox_iou=float(relaxed_merge_min_bbox_iou),
                max_bbox_gap_norm=float(relaxed_merge_max_bbox_gap_norm),
                max_center_dist_norm=float(relaxed_merge_max_center_dist_norm),
            )
            if require_relaxed_merge_object_witness and not bool(object_witness.get("accepted")):
                rejected_tiny_lock_expansions.append(
                    {
                        **row,
                        "witness": "tiny_overlap_lock_expansion_rejected_by_current_object_geometry",
                        "locked_curr_local_id": int(locked_curr_id),
                        "locked_curr_mean_area": float(locked_area),
                        "object_witness": object_witness,
                    }
                )
                continue
            mapping[curr_id] = prev_id
            used_curr.add(curr_id)
            expansion_count_by_prev[prev_id] = expansion_count_by_prev.get(prev_id, 0) + 1
            tiny_lock_expansions.append(
                {
                    **row,
                    "witness": "tiny_overlap_lock_expansion_by_track_geometry",
                    "locked_curr_local_id": int(locked_curr_id),
                    "locked_curr_mean_area": float(locked_area),
                    "tiny_lock_overlap_iou": float(lock.get("iou") or 0.0),
                    "tiny_lock_prev_coverage": float(lock.get("prev_coverage") or 0.0),
                    "object_witness": object_witness,
                    "tiny_lock_note": (
                        "A history ID was overlap-continued into a tiny current fragment. A larger current part with "
                        "track-level geometry and appearance support inherits the same history ID. This is ID-only and "
                        "does not modify local mask geometry."
                    ),
                }
            )

    audit = {
        "prev_descriptor_count": len(prev_desc),
        "curr_descriptor_count": len(curr_desc),
        "candidate_count": len(candidates),
        "all_candidate_count": len(all_candidates),
        "accepted_count": len(accepted),
        "weak_overlap_override_count": len(overrides),
        "part_merge_count": len(part_merges),
        "tiny_lock_expansion_count": len(tiny_lock_expansions),
        "rejected_part_merge_by_object_witness_count": len(rejected_part_merges),
        "rejected_tiny_lock_expansion_by_object_witness_count": len(rejected_tiny_lock_expansions),
        "min_area": int(min_area),
        "min_score": float(min_score),
        "min_color": float(min_color),
        "min_margin": float(min_margin),
        "require_relaxed_merge_object_witness": bool(require_relaxed_merge_object_witness),
        "relaxed_merge_min_bbox_iou": float(relaxed_merge_min_bbox_iou),
        "relaxed_merge_max_bbox_gap_norm": float(relaxed_merge_max_bbox_gap_norm),
        "relaxed_merge_max_center_dist_norm": float(relaxed_merge_max_center_dist_norm),
        "allow_part_merge": bool(allow_part_merge),
        "part_merge_min_score": float(part_merge_min_score),
        "part_merge_min_color": float(part_merge_min_color),
        "part_merge_min_spatial": float(part_merge_min_spatial),
        "allow_weak_overlap_override": bool(allow_weak_overlap_override),
        "override_min_score": float(override_min_score),
        "override_min_color": float(override_min_color),
        "override_min_spatial": float(override_min_spatial),
        "allow_tiny_lock_expansion": bool(allow_tiny_lock_expansion),
        "tiny_lock_expansion_min_score": float(tiny_lock_expansion_min_score),
        "tiny_lock_expansion_min_color": float(tiny_lock_expansion_min_color),
        "tiny_lock_expansion_min_spatial": float(tiny_lock_expansion_min_spatial),
        "tiny_lock_expansion_min_area_ratio_vs_lock": float(tiny_lock_expansion_min_area_ratio_vs_lock),
        "tiny_lock_expansion_max_per_prev": int(tiny_lock_expansion_max_per_prev),
        "override_drop_curr_ids": sorted(set(override_drop_curr_ids)),
        "accepted_matches_first40": accepted[:40],
        "weak_overlap_overrides_first40": overrides[:40],
        "part_merges_first40": part_merges[:40],
        "tiny_lock_expansions_first40": tiny_lock_expansions[:40],
        "rejected_part_merges_by_object_witness_first40": rejected_part_merges[:40],
        "rejected_tiny_lock_expansions_by_object_witness_first40": rejected_tiny_lock_expansions[:40],
        "top_candidates_first40": candidates[:40],
    }
    return mapping, audit


def _weak_overlap_prev_locks(
    overlap_audit: dict[str, Any],
    *,
    min_abs_intersection: int,
    min_prev_coverage_for_tiny_iou: float,
    tiny_iou: float,
) -> dict[int, int]:
    locks: dict[int, int] = {}
    accepted = overlap_audit.get("accepted_matches_first40", [])
    if not isinstance(accepted, list):
        return locks
    for row in accepted:
        if not isinstance(row, dict):
            continue
        prev_area = float(row.get("prev_area") or 0.0)
        inter = float(row.get("intersection") or 0.0)
        iou = float(row.get("iou") or 0.0)
        prev_coverage = inter / prev_area if prev_area > 0 else 0.0
        if inter < float(min_abs_intersection) or (
            iou < float(tiny_iou) and prev_coverage < float(min_prev_coverage_for_tiny_iou)
        ):
            locks[int(row["prev_global_id"])] = int(row["curr_local_id"])
    return locks


def _tiny_overlap_lock_details(
    overlap_audit: dict[str, Any],
    *,
    max_iou: float,
    max_prev_coverage: float,
) -> dict[int, dict[str, Any]]:
    locks: dict[int, dict[str, Any]] = {}
    accepted = overlap_audit.get("accepted_matches_first40", [])
    if not isinstance(accepted, list):
        return locks
    for row in accepted:
        if not isinstance(row, dict):
            continue
        prev_area = float(row.get("prev_area") or 0.0)
        inter = float(row.get("intersection") or 0.0)
        iou = float(row.get("iou") or 0.0)
        prev_coverage = inter / prev_area if prev_area > 0 else 0.0
        if iou <= float(max_iou) and prev_coverage <= float(max_prev_coverage):
            locks[int(row["prev_global_id"])] = {
                **row,
                "prev_coverage": float(prev_coverage),
                "tiny_lock_max_iou": float(max_iou),
                "tiny_lock_max_prev_coverage": float(max_prev_coverage),
            }
    return locks


def _parse_boundary_allowlist(value: str) -> set[tuple[str, int, int]]:
    allow: set[tuple[str, int, int]] = set()
    for part in str(value).split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item or "-" not in item:
            raise ValueError(f"Invalid boundary allowlist item: {item!r}; expected scene_id:prev-curr")
        scene_id, bounds = item.split(":", 1)
        prev_s, curr_s = bounds.split("-", 1)
        allow.add((scene_id.strip(), int(prev_s), int(curr_s)))
    return allow


def _parse_evidence_boundary_remaps(value: str) -> dict[tuple[str, int, int], dict[int, int]]:
    remaps: dict[tuple[str, int, int], dict[int, int]] = {}
    for part in str(value).split(";"):
        item = part.strip()
        if not item:
            continue
        pieces = item.split(":")
        if len(pieces) != 3 or "-" not in pieces[1]:
            raise ValueError(
                f"Invalid evidence remap item: {item!r}; expected scene_id:prev-curr:curr_id=prev_global_id,..."
            )
        scene_id = pieces[0].strip()
        prev_s, curr_s = pieces[1].split("-", 1)
        key = (scene_id, int(prev_s), int(curr_s))
        mapping: dict[int, int] = {}
        for pair in pieces[2].split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"Invalid evidence remap pair: {pair!r}; expected curr_id=prev_global_id")
            curr_id_s, prev_id_s = pair.split("=", 1)
            mapping[int(curr_id_s)] = int(prev_id_s)
        if mapping:
            remaps[key] = mapping
    return remaps


def _balanced_boundary_matches(
    *,
    prev_global: np.ndarray,
    curr_local: np.ndarray,
    min_iou: float,
    min_overlap_min: float,
    min_abs_intersection: int,
    min_prev_coverage_for_tiny_iou: float,
    tiny_iou: float,
) -> tuple[dict[int, int], dict[str, Any]]:
    _mapping, audit = _boundary_matches(
        prev_global=prev_global,
        curr_local=curr_local,
        min_iou=float(min_iou),
        min_overlap_min=float(min_overlap_min),
    )
    accepted = audit.get("accepted_matches_first40", [])
    if not isinstance(accepted, list):
        accepted = []
    filtered_mapping: dict[int, int] = {}
    filtered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in accepted:
        if not isinstance(row, dict):
            continue
        prev_area = float(row.get("prev_area") or 0.0)
        inter = float(row.get("intersection") or 0.0)
        iou = float(row.get("iou") or 0.0)
        prev_coverage = inter / prev_area if prev_area > 0 else 0.0
        reject_tiny_lock = (
            inter < float(min_abs_intersection)
            or (iou < float(tiny_iou) and prev_coverage < float(min_prev_coverage_for_tiny_iou))
        )
        if reject_tiny_lock:
            rejected.append(
                {
                    **row,
                    "rejected_reason": "tiny_overlap_not_object_specific_enough_to_lock_history_id",
                    "prev_coverage": float(prev_coverage),
                }
            )
            continue
        curr_id = int(row["curr_local_id"])
        prev_id = int(row["prev_global_id"])
        filtered_mapping[curr_id] = prev_id
        filtered.append({**row, "prev_coverage": float(prev_coverage)})

    audit = dict(audit)
    audit.update(
        {
            "balanced_overlap_enabled": True,
            "balanced_overlap_min_abs_intersection": int(min_abs_intersection),
            "balanced_overlap_min_prev_coverage_for_tiny_iou": float(min_prev_coverage_for_tiny_iou),
            "balanced_overlap_tiny_iou": float(tiny_iou),
            "matched_count_before_balanced_filter": int(audit.get("matched_count", 0)),
            "matched_count": len(filtered),
            "accepted_matches_first40": filtered[:40],
            "balanced_rejected_matches_first40": rejected[:40],
            "balanced_rejected_count": len(rejected),
        }
    )
    return filtered_mapping, audit


def _build_scene(
    *,
    scene_id: str,
    source_mask_dir: Path,
    scene_color_dir: Path,
    plan: dict[str, Any],
    output_root: Path,
    variant_id: str,
    alpha: float,
    fps: float,
    min_iou: float,
    min_overlap_min: float,
    descriptor_window: int,
    descriptor_min_area: int,
    appearance_min_score: float,
    appearance_min_color: float,
    appearance_min_margin: float,
    balanced_overlap: bool,
    balanced_min_abs_intersection: int,
    balanced_min_prev_coverage: float,
    balanced_tiny_iou: float,
    allow_part_merge: bool,
    part_merge_min_score: float,
    part_merge_min_color: float,
    part_merge_min_spatial: float,
    allow_weak_overlap_override: bool,
    override_min_score: float,
    override_min_color: float,
    override_min_spatial: float,
    allow_tiny_lock_expansion: bool,
    tiny_lock_expansion_boundary_allowlist: set[tuple[str, int, int]],
    tiny_lock_expansion_max_iou: float,
    tiny_lock_expansion_max_prev_coverage: float,
    tiny_lock_expansion_min_score: float,
    tiny_lock_expansion_min_color: float,
    tiny_lock_expansion_min_spatial: float,
    tiny_lock_expansion_min_area_ratio_vs_lock: float,
    tiny_lock_expansion_max_per_prev: int,
    require_relaxed_merge_object_witness: bool,
    relaxed_merge_min_bbox_iou: float,
    relaxed_merge_max_bbox_gap_norm: float,
    relaxed_merge_max_center_dist_norm: float,
    part_merge_boundary_allowlist: set[tuple[str, int, int]],
    evidence_boundary_remaps: dict[tuple[str, int, int], dict[int, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    frame_ids = sorted([_numeric_stem(path) for path in source_mask_dir.glob("*.png") if _numeric_stem(path) < 10**12])
    chunks = _scene_chunks(plan, scene_id, frame_ids)
    mask_dir = output_root / "local2history_reappearance" / "masks" / variant_id / scene_id / "mask"
    overlay_dir = output_root / "local2history_reappearance" / "overlays" / variant_id / scene_id
    sheet_root = output_root / "local2history_reappearance" / "sheet_groups" / variant_id / scene_id
    boundary_sheet_root = output_root / "local2history_reappearance" / "boundary_sheets" / variant_id / scene_id
    video_path = output_root / "local2history_reappearance" / "videos" / f"{variant_id}_{scene_id}_full_stride5.mp4"
    for directory in (mask_dir, overlay_dir, sheet_root, boundary_sheet_root, video_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    next_global_id = 1
    prev_recent_global: deque[tuple[int, np.ndarray]] = deque(maxlen=max(int(descriptor_window), 1))
    prev_last_global: np.ndarray | None = None
    prev_last_frame_id: int | None = None
    chunk_records: list[dict[str, Any]] = []
    boundary_records: list[dict[str, Any]] = []
    sheet_records: list[dict[str, Any]] = []
    boundary_sheet_records: list[dict[str, Any]] = []
    written_masks = 0
    written_overlays = 0
    video_written = 0
    writer: cv2.VideoWriter | None = None

    for chunk in chunks:
        start = int(chunk["start_index"])
        count = int(chunk["frame_count"])
        chunk_frame_ids = frame_ids[start : start + count]
        if not chunk_frame_ids:
            continue

        first_label = _read_label(source_mask_dir / f"{chunk_frame_ids[0]}.png")
        mapping: dict[int, int] = {}
        boundary_audit: dict[str, Any] | None = None
        if prev_last_global is not None:
            boundary_key = (scene_id, int(chunk["chunk_index"]) - 1, int(chunk["chunk_index"]))
            part_merge_allowed_here = bool(allow_part_merge) and (
                not part_merge_boundary_allowlist or boundary_key in part_merge_boundary_allowlist
            )
            tiny_lock_expansion_allowed_here = bool(allow_tiny_lock_expansion) and (
                not tiny_lock_expansion_boundary_allowlist or boundary_key in tiny_lock_expansion_boundary_allowlist
            )
            if balanced_overlap:
                overlap_mapping, overlap_audit = _balanced_boundary_matches(
                    prev_global=prev_last_global,
                    curr_local=first_label,
                    min_iou=float(min_iou),
                    min_overlap_min=float(min_overlap_min),
                    min_abs_intersection=int(balanced_min_abs_intersection),
                    min_prev_coverage_for_tiny_iou=float(balanced_min_prev_coverage),
                    tiny_iou=float(balanced_tiny_iou),
                )
            else:
                overlap_mapping, overlap_audit = _boundary_matches(
                    prev_global=prev_last_global,
                    curr_local=first_label,
                    min_iou=float(min_iou),
                    min_overlap_min=float(min_overlap_min),
                )
            weak_prev_locks = _weak_overlap_prev_locks(
                overlap_audit,
                min_abs_intersection=int(balanced_min_abs_intersection),
                min_prev_coverage_for_tiny_iou=float(balanced_min_prev_coverage),
                tiny_iou=float(balanced_tiny_iou),
            )
            tiny_lock_details = _tiny_overlap_lock_details(
                overlap_audit,
                max_iou=float(tiny_lock_expansion_max_iou),
                max_prev_coverage=float(tiny_lock_expansion_max_prev_coverage),
            )
            curr_items = [
                (int(frame_id), _read_label(source_mask_dir / f"{int(frame_id)}.png"))
                for frame_id in chunk_frame_ids[: max(int(descriptor_window), 1)]
            ]
            appearance_mapping, appearance_audit = _appearance_matches(
                prev_items=list(prev_recent_global),
                curr_items=curr_items,
                scene_color_dir=scene_color_dir,
                existing_mapping=overlap_mapping,
                min_area=int(descriptor_min_area),
                min_score=float(appearance_min_score),
                min_color=float(appearance_min_color),
                min_margin=float(appearance_min_margin),
                allow_part_merge=bool(part_merge_allowed_here),
                part_merge_min_score=float(part_merge_min_score),
                part_merge_min_color=float(part_merge_min_color),
                part_merge_min_spatial=float(part_merge_min_spatial),
                weak_existing_prev_to_curr=weak_prev_locks,
                allow_weak_overlap_override=bool(allow_weak_overlap_override),
                override_min_score=float(override_min_score),
                override_min_color=float(override_min_color),
                override_min_spatial=float(override_min_spatial),
                tiny_lock_prev_details=tiny_lock_details,
                allow_tiny_lock_expansion=bool(tiny_lock_expansion_allowed_here),
                tiny_lock_expansion_min_score=float(tiny_lock_expansion_min_score),
                tiny_lock_expansion_min_color=float(tiny_lock_expansion_min_color),
                tiny_lock_expansion_min_spatial=float(tiny_lock_expansion_min_spatial),
                tiny_lock_expansion_min_area_ratio_vs_lock=float(tiny_lock_expansion_min_area_ratio_vs_lock),
                tiny_lock_expansion_max_per_prev=int(tiny_lock_expansion_max_per_prev),
                require_relaxed_merge_object_witness=bool(require_relaxed_merge_object_witness),
                relaxed_merge_min_bbox_iou=float(relaxed_merge_min_bbox_iou),
                relaxed_merge_max_bbox_gap_norm=float(relaxed_merge_max_bbox_gap_norm),
                relaxed_merge_max_center_dist_norm=float(relaxed_merge_max_center_dist_norm),
            )
            for drop_curr_id in appearance_audit.get("override_drop_curr_ids", []):
                try:
                    overlap_mapping.pop(int(drop_curr_id), None)
                except Exception:
                    continue
            mapping = {**overlap_mapping, **appearance_mapping}
            evidence_remap_records: list[dict[str, Any]] = []
            evidence_remap = evidence_boundary_remaps.get(boundary_key, {})
            if evidence_remap:
                prev_global_ids_available = set(int(v) for v in np.unique(prev_last_global) if int(v) > 0)
                curr_local_ids_available = set(int(v) for v in np.unique(first_label) if int(v) > 0)
                for curr_id, target_prev_id in sorted(evidence_remap.items()):
                    old_prev_id = mapping.get(int(curr_id))
                    applied = int(curr_id) in curr_local_ids_available and int(target_prev_id) in prev_global_ids_available
                    if applied:
                        mapping[int(curr_id)] = int(target_prev_id)
                    evidence_remap_records.append(
                        {
                            "curr_local_id": int(curr_id),
                            "old_prev_global_id": int(old_prev_id) if old_prev_id is not None else None,
                            "new_prev_global_id": int(target_prev_id),
                            "applied": bool(applied),
                            "curr_local_present_in_boundary_frame": bool(int(curr_id) in curr_local_ids_available),
                            "target_prev_global_present_in_prev_frame": bool(int(target_prev_id) in prev_global_ids_available),
                            "witness": "explicit_evidence_boundary_remap_from_vertex_support_diagnostic",
                            "note": (
                                "ID-only remap supplied by an explicit command-line allowlist after separate "
                                "vertex-support and visual diagnostics. Local mask geometry is not modified."
                            ),
                        }
                    )
            boundary_audit = {
                **overlap_audit,
                "schema_version": "stream4d_v105_l2h_reappearance_boundary_audit_v1",
                "scene_id": scene_id,
                "prev_chunk_index": int(chunk["chunk_index"]) - 1,
                "curr_chunk_index": int(chunk["chunk_index"]),
                "prev_frame_id": int(prev_last_frame_id),
                "curr_frame_id": int(chunk_frame_ids[0]),
                "min_iou": float(min_iou),
                "min_overlap_min": float(min_overlap_min),
                "descriptor_window": int(descriptor_window),
                "appearance_audit": appearance_audit,
                "appearance_inherited_count": len(appearance_mapping),
                "weak_overlap_prev_lock_count": len(weak_prev_locks),
                "tiny_overlap_lock_count": len(tiny_lock_details),
                "weak_overlap_override_count": int(appearance_audit.get("weak_overlap_override_count", 0)),
                "appearance_part_merge_count": int(appearance_audit.get("part_merge_count", 0)),
                "tiny_lock_expansion_count": int(appearance_audit.get("tiny_lock_expansion_count", 0)),
                "evidence_boundary_remap_count": sum(1 for row in evidence_remap_records if row.get("applied")),
                "evidence_boundary_remaps_first40": evidence_remap_records[:40],
                "part_merge_allowed_here": bool(part_merge_allowed_here),
                "tiny_lock_expansion_allowed_here": bool(tiny_lock_expansion_allowed_here),
                "evidence_boundary_remap_allowed_here": bool(evidence_remap),
                "part_merge_boundary_allowlist": [
                    {"scene_id": scene, "prev_chunk_index": prev, "curr_chunk_index": curr}
                    for scene, prev, curr in sorted(part_merge_boundary_allowlist)
                ],
                "tiny_lock_expansion_boundary_allowlist": [
                    {"scene_id": scene, "prev_chunk_index": prev, "curr_chunk_index": curr}
                    for scene, prev, curr in sorted(tiny_lock_expansion_boundary_allowlist)
                ],
                "total_mapping_count": len(mapping),
                "mapping_source_counts": {
                    "overlap_continuation": len(overlap_mapping),
                    "multi_frame_reappearance": len(appearance_mapping),
                    "weak_overlap_override": int(appearance_audit.get("weak_overlap_override_count", 0)),
                    "duplicate_part_merge": int(appearance_audit.get("part_merge_count", 0)),
                    "tiny_overlap_lock_expansion": int(appearance_audit.get("tiny_lock_expansion_count", 0)),
                    "evidence_vertex_support_remap": sum(1 for row in evidence_remap_records if row.get("applied")),
                },
                "reappearance_matches_first40": [
                    {"curr_local_id": int(k), "prev_global_id": int(v)} for k, v in sorted(appearance_mapping.items())
                ][:40],
            }
            boundary_records.append(boundary_audit)

        chunk_new_births = 0
        chunk_overlay_paths: list[Path] = []
        local_ids_seen: set[int] = set()
        global_ids_seen: set[int] = set()
        local_to_global_initial = dict(mapping)
        last_global: np.ndarray | None = None
        recent_this_chunk: deque[tuple[int, np.ndarray]] = deque(maxlen=max(int(descriptor_window), 1))
        for frame_pos, frame_id in enumerate(chunk_frame_ids):
            local = _read_label(source_mask_dir / f"{int(frame_id)}.png")
            local_ids_seen.update(int(v) for v in np.unique(local) if int(v) > 0)
            global_label, next_global_id, new_births = _remap_label(local, mapping, next_global_id)
            chunk_new_births += int(new_births)
            global_ids_seen.update(int(v) for v in np.unique(global_label) if int(v) > 0)
            cv2.imwrite(str(mask_dir / f"{int(frame_id)}.png"), global_label)
            written_masks += 1
            recent_this_chunk.append((int(frame_id), global_label.copy()))

            rgb = _read_rgb(scene_color_dir, int(frame_id))
            if rgb is not None:
                overlay = _overlay_label(rgb, global_label, alpha=float(alpha))
                overlay = _put_text(
                    overlay,
                    [
                        f"{scene_id} {variant_id} frame_index={start + frame_pos:04d} frame_id={int(frame_id):06d}",
                        f"ID-only L2H reappearance repair; chunk={chunk['chunk_index']} local geometry unchanged",
                    ],
                )
                overlay_path = overlay_dir / f"{start + frame_pos:04d}_frame_{int(frame_id):06d}.jpg"
                cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                chunk_overlay_paths.append(overlay_path)
                written_overlays += 1
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(fps),
                        (overlay.shape[1], overlay.shape[0]),
                    )
                if writer is not None and writer.isOpened():
                    writer.write(overlay)
                    video_written += 1
            last_global = global_label

        if chunk_overlay_paths:
            for group_idx, offset in enumerate(range(0, len(chunk_overlay_paths), 8)):
                group_paths = chunk_overlay_paths[offset : offset + 8]
                out_path = sheet_root / f"chunk_{int(chunk['chunk_index']):03d}_frames_{offset:03d}_{offset + len(group_paths) - 1:03d}.jpg"
                sheet_records.append(
                    {
                        "scene_id": scene_id,
                        "chunk_index": int(chunk["chunk_index"]),
                        "sheet_group_index": int(group_idx),
                        "frame_ids": [int(chunk_frame_ids[offset + idx]) for idx in range(len(group_paths))],
                        **_make_sheet(
                            group_paths,
                            out_path,
                            f"{scene_id} chunk {int(chunk['chunk_index']):03d} l2h reappearance frames {offset:03d}-{offset + len(group_paths) - 1:03d}",
                            480,
                            360,
                        ),
                    }
                )
        if boundary_audit is not None:
            prev_paths = sorted(overlay_dir.glob("*.jpg"))[-(len(chunk_overlay_paths) + 4) : -len(chunk_overlay_paths)] if len(chunk_overlay_paths) else []
            curr_paths = chunk_overlay_paths[:4]
            if prev_paths and curr_paths:
                out_path = boundary_sheet_root / f"boundary_{int(chunk['chunk_index']) - 1:03d}_{int(chunk['chunk_index']):03d}.jpg"
                boundary_sheet_records.append(
                    {
                        "scene_id": scene_id,
                        "prev_chunk_index": int(chunk["chunk_index"]) - 1,
                        "curr_chunk_index": int(chunk["chunk_index"]),
                        "prev_frame_id": int(prev_last_frame_id),
                        "curr_frame_id": int(chunk_frame_ids[0]),
                        **_make_sheet(
                            list(prev_paths) + curr_paths,
                            out_path,
                            f"{scene_id} boundary {int(chunk['chunk_index']) - 1:03d}->{int(chunk['chunk_index']):03d} l2h reappearance",
                            480,
                            360,
                        ),
                    }
                )

        chunk_records.append(
            {
                "scene_id": scene_id,
                "chunk_index": int(chunk["chunk_index"]),
                "source": chunk.get("source"),
                "start_index": start,
                "frame_count": len(chunk_frame_ids),
                "first_frame_id": int(chunk_frame_ids[0]),
                "last_frame_id": int(chunk_frame_ids[-1]),
                "local_id_count": len(local_ids_seen),
                "global_id_count": len(global_ids_seen),
                "new_global_birth_count": int(chunk_new_births),
                "inherited_from_previous_count": int(boundary_audit["matched_count"]) if boundary_audit else 0,
                "reappearance_inherited_count": int(boundary_audit["appearance_inherited_count"]) if boundary_audit else 0,
                "weak_overlap_override_count": int(boundary_audit["weak_overlap_override_count"]) if boundary_audit else 0,
                "appearance_part_merge_count": int(boundary_audit["appearance_part_merge_count"]) if boundary_audit else 0,
                "tiny_lock_expansion_count": int(boundary_audit["tiny_lock_expansion_count"]) if boundary_audit else 0,
                "evidence_boundary_remap_count": int(boundary_audit["evidence_boundary_remap_count"]) if boundary_audit else 0,
                "total_initial_mapping_count": len(local_to_global_initial),
            }
        )
        prev_recent_global = recent_this_chunk
        prev_last_global = last_global
        prev_last_frame_id = int(chunk_frame_ids[-1])

    if writer is not None:
        writer.release()
    decoded_frames = 0
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            decoded_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    weak_boundaries = [
        row
        for row in boundary_records
        if int(row.get("total_mapping_count", 0)) == 0 or int(row.get("large_unmatched_curr_count", 0)) > 0
    ]
    row = {
        "schema_version": "stream4d_v105_fullscene_l2h_reappearance_scene_row_v1",
        "scene_id": scene_id,
        "variant_id": variant_id,
        "source_variant": str(plan.get("variant_id")),
        "source_mask_dir": _rel(source_mask_dir),
        "mask_dir": _rel(mask_dir),
        "overlay_dir": _rel(overlay_dir),
        "video_path": _rel(video_path),
        "frame_count_expected": len(frame_ids),
        "mask_count_written": written_masks,
        "overlay_count_written": written_overlays,
        "video_frame_count_written": video_written,
        "video_frame_count_decoded": decoded_frames,
        "complete_scene_masks": bool(written_masks == len(frame_ids)),
        "complete_scene_video": bool(decoded_frames == len(frame_ids)),
        "chunk_count": len(chunk_records),
        "boundary_count": len(boundary_records),
        "weak_boundary_count": len(weak_boundaries),
        "weak_boundaries_first20": weak_boundaries[:20],
        "total_global_id_count": int(next_global_id - 1),
        "chunk_records": chunk_records,
        "boundary_records": boundary_records,
        "sheet_group_count": len(sheet_records),
        "boundary_sheet_count": len(boundary_sheet_records),
        "id_only_stitch_candidate": True,
        "mask_geometry_modified": False,
        "continuous_scene_level_id_claim": False,
        "claim_boundary": "ID-only reappearance repair candidate; requires boundary visual review and user confirmation before a continuous identity claim.",
    }
    return row, sheet_records, boundary_sheet_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v105 P8 ID-only local2history reappearance repair candidate.")
    parser.add_argument("--fullscene-root", default=_rel(DEFAULT_FULLSCENE_ROOT))
    parser.add_argument("--output-root", default=_rel(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--scenes", default="")
    parser.add_argument("--variant-id", default=DEFAULT_VARIANT_ID)
    parser.add_argument("--min-iou", type=float, default=0.02)
    parser.add_argument("--min-overlap-min", type=float, default=0.08)
    parser.add_argument("--descriptor-window", type=int, default=6)
    parser.add_argument("--descriptor-min-area", type=int, default=2500)
    parser.add_argument("--appearance-min-score", type=float, default=0.58)
    parser.add_argument("--appearance-min-color", type=float, default=0.25)
    parser.add_argument("--appearance-min-margin", type=float, default=0.03)
    parser.add_argument("--balanced-overlap", action="store_true")
    parser.add_argument("--balanced-min-abs-intersection", type=int, default=5000)
    parser.add_argument("--balanced-min-prev-coverage", type=float, default=0.03)
    parser.add_argument("--balanced-tiny-iou", type=float, default=0.03)
    parser.add_argument("--allow-part-merge", action="store_true")
    parser.add_argument(
        "--part-merge-boundary-allowlist",
        default="",
        help="Comma separated scene_id:prev-curr entries. When set, part merge is only allowed on these boundaries.",
    )
    parser.add_argument("--part-merge-min-score", type=float, default=0.66)
    parser.add_argument("--part-merge-min-color", type=float, default=0.50)
    parser.add_argument("--part-merge-min-spatial", type=float, default=0.25)
    parser.add_argument("--allow-weak-overlap-override", action="store_true")
    parser.add_argument("--override-min-score", type=float, default=0.66)
    parser.add_argument("--override-min-color", type=float, default=0.50)
    parser.add_argument("--override-min-spatial", type=float, default=0.25)
    parser.add_argument("--allow-tiny-lock-expansion", action="store_true")
    parser.add_argument(
        "--tiny-lock-expansion-boundary-allowlist",
        default="",
        help="Comma separated scene_id:prev-curr entries. When set, tiny-lock expansion is only allowed on these boundaries.",
    )
    parser.add_argument("--tiny-lock-expansion-max-iou", type=float, default=0.08)
    parser.add_argument("--tiny-lock-expansion-max-prev-coverage", type=float, default=0.08)
    parser.add_argument("--tiny-lock-expansion-min-score", type=float, default=0.40)
    parser.add_argument("--tiny-lock-expansion-min-color", type=float, default=0.15)
    parser.add_argument("--tiny-lock-expansion-min-spatial", type=float, default=0.35)
    parser.add_argument("--tiny-lock-expansion-min-area-ratio-vs-lock", type=float, default=1.20)
    parser.add_argument("--tiny-lock-expansion-max-per-prev", type=int, default=3)
    parser.add_argument("--require-relaxed-merge-object-witness", action="store_true")
    parser.add_argument("--relaxed-merge-min-bbox-iou", type=float, default=0.01)
    parser.add_argument("--relaxed-merge-max-bbox-gap-norm", type=float, default=0.03)
    parser.add_argument("--relaxed-merge-max-center-dist-norm", type=float, default=0.16)
    parser.add_argument(
        "--evidence-boundary-remap",
        default="",
        help=(
            "Semicolon-separated explicit scene_id:prev-curr:curr_id=prev_global_id remaps, e.g. "
            "scene0591_00:4-5:2=136,7=136. Intended only for audited vertex-support/visual evidence."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()

    fullscene_root = Path(args.fullscene_root)
    if not fullscene_root.is_absolute():
        fullscene_root = REPO_ROOT / fullscene_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    plan = _read_json(fullscene_root / "fullscene_multichunk_plan.json")
    summary = _read_json(fullscene_root / "fullscene_multichunk_summary.json")
    source_variant = str(summary.get("variant_id") or plan.get("variant_id"))
    variant_id = str(args.variant_id)
    scenes = [part.strip() for part in str(args.scenes).split(",") if part.strip()] or [str(scene) for scene in plan.get("scenes", [])]
    part_merge_boundary_allowlist = _parse_boundary_allowlist(str(args.part_merge_boundary_allowlist))
    tiny_lock_expansion_boundary_allowlist = _parse_boundary_allowlist(str(args.tiny_lock_expansion_boundary_allowlist))
    evidence_boundary_remaps = _parse_evidence_boundary_remaps(str(args.evidence_boundary_remap))

    scene_rows: list[dict[str, Any]] = []
    all_sheet_records: list[dict[str, Any]] = []
    all_boundary_sheet_records: list[dict[str, Any]] = []
    for scene_id in scenes:
        source_mask_dir = fullscene_root / "assembled_scene_videos" / "sgq_local" / "masks" / source_variant / scene_id / "mask"
        scene_color_dir = REPO_ROOT / "Stream3D/data/scannet/processed" / scene_id / "color"
        row, sheet_records, boundary_sheet_records = _build_scene(
            scene_id=scene_id,
            source_mask_dir=source_mask_dir,
            scene_color_dir=scene_color_dir,
            plan=plan,
            output_root=output_root,
            variant_id=variant_id,
            alpha=float(args.alpha),
            fps=float(args.fps),
            min_iou=float(args.min_iou),
            min_overlap_min=float(args.min_overlap_min),
            descriptor_window=int(args.descriptor_window),
            descriptor_min_area=int(args.descriptor_min_area),
            appearance_min_score=float(args.appearance_min_score),
            appearance_min_color=float(args.appearance_min_color),
            appearance_min_margin=float(args.appearance_min_margin),
            balanced_overlap=bool(args.balanced_overlap),
            balanced_min_abs_intersection=int(args.balanced_min_abs_intersection),
            balanced_min_prev_coverage=float(args.balanced_min_prev_coverage),
            balanced_tiny_iou=float(args.balanced_tiny_iou),
            allow_part_merge=bool(args.allow_part_merge),
            part_merge_min_score=float(args.part_merge_min_score),
            part_merge_min_color=float(args.part_merge_min_color),
            part_merge_min_spatial=float(args.part_merge_min_spatial),
            allow_weak_overlap_override=bool(args.allow_weak_overlap_override),
            override_min_score=float(args.override_min_score),
            override_min_color=float(args.override_min_color),
            override_min_spatial=float(args.override_min_spatial),
            allow_tiny_lock_expansion=bool(args.allow_tiny_lock_expansion),
            tiny_lock_expansion_boundary_allowlist=tiny_lock_expansion_boundary_allowlist,
            tiny_lock_expansion_max_iou=float(args.tiny_lock_expansion_max_iou),
            tiny_lock_expansion_max_prev_coverage=float(args.tiny_lock_expansion_max_prev_coverage),
            tiny_lock_expansion_min_score=float(args.tiny_lock_expansion_min_score),
            tiny_lock_expansion_min_color=float(args.tiny_lock_expansion_min_color),
            tiny_lock_expansion_min_spatial=float(args.tiny_lock_expansion_min_spatial),
            tiny_lock_expansion_min_area_ratio_vs_lock=float(args.tiny_lock_expansion_min_area_ratio_vs_lock),
            tiny_lock_expansion_max_per_prev=int(args.tiny_lock_expansion_max_per_prev),
            require_relaxed_merge_object_witness=bool(args.require_relaxed_merge_object_witness),
            relaxed_merge_min_bbox_iou=float(args.relaxed_merge_min_bbox_iou),
            relaxed_merge_max_bbox_gap_norm=float(args.relaxed_merge_max_bbox_gap_norm),
            relaxed_merge_max_center_dist_norm=float(args.relaxed_merge_max_center_dist_norm),
            part_merge_boundary_allowlist=part_merge_boundary_allowlist,
            evidence_boundary_remaps=evidence_boundary_remaps,
        )
        scene_rows.append(row)
        all_sheet_records.extend(sheet_records)
        all_boundary_sheet_records.extend(boundary_sheet_records)

    _write_json(output_root / "scene_rows.json", scene_rows)
    _write_json(output_root / "sheet_records.json", all_sheet_records)
    _write_json(output_root / "boundary_sheet_records.json", all_boundary_sheet_records)
    all_complete_masks = bool(scene_rows) and all(bool(row.get("complete_scene_masks")) for row in scene_rows)
    all_complete_videos = bool(scene_rows) and all(bool(row.get("complete_scene_video")) for row in scene_rows)
    total_weak = sum(int(row.get("weak_boundary_count", 0)) for row in scene_rows)
    total_reappearance = sum(
        int(boundary.get("appearance_inherited_count", 0))
        for row in scene_rows
        for boundary in row.get("boundary_records", [])
        if isinstance(boundary, dict)
    )
    total_part_merges = sum(
        int(boundary.get("appearance_part_merge_count", 0))
        for row in scene_rows
        for boundary in row.get("boundary_records", [])
        if isinstance(boundary, dict)
    )
    total_weak_overrides = sum(
        int(boundary.get("weak_overlap_override_count", 0))
        for row in scene_rows
        for boundary in row.get("boundary_records", [])
        if isinstance(boundary, dict)
    )
    total_tiny_lock_expansions = sum(
        int(boundary.get("tiny_lock_expansion_count", 0))
        for row in scene_rows
        for boundary in row.get("boundary_records", [])
        if isinstance(boundary, dict)
    )
    total_evidence_remaps = sum(
        int(boundary.get("evidence_boundary_remap_count", 0))
        for row in scene_rows
        for boundary in row.get("boundary_records", [])
        if isinstance(boundary, dict)
    )
    output_summary = {
        "schema_version": "stream4d_v105_fullscene_l2h_reappearance_summary_v1",
        "variant_id": variant_id,
        "source_fullscene_root": _rel(fullscene_root),
        "source_variant": source_variant,
        "output_root": _rel(output_root),
        "scene_count": len(scene_rows),
        "scenes": scenes,
        "scene_rows_json": _rel(output_root / "scene_rows.json"),
        "sheet_records_json": _rel(output_root / "sheet_records.json"),
        "boundary_sheet_records_json": _rel(output_root / "boundary_sheet_records.json"),
        "all_complete_scene_masks": all_complete_masks,
        "all_complete_scene_videos": all_complete_videos,
        "scene_rows": scene_rows,
        "total_boundary_count": sum(int(row.get("boundary_count", 0)) for row in scene_rows),
        "total_weak_boundary_count": total_weak,
        "total_reappearance_inherited_count": int(total_reappearance),
        "total_weak_overlap_override_count": int(total_weak_overrides),
        "total_part_merge_count": int(total_part_merges),
        "total_tiny_lock_expansion_count": int(total_tiny_lock_expansions),
        "total_evidence_boundary_remap_count": int(total_evidence_remaps),
        "total_sheet_group_count": len(all_sheet_records),
        "total_boundary_sheet_count": len(all_boundary_sheet_records),
        "min_iou": float(args.min_iou),
        "min_overlap_min": float(args.min_overlap_min),
        "descriptor_window": int(args.descriptor_window),
        "descriptor_min_area": int(args.descriptor_min_area),
        "appearance_min_score": float(args.appearance_min_score),
        "appearance_min_color": float(args.appearance_min_color),
        "appearance_min_margin": float(args.appearance_min_margin),
        "balanced_overlap": bool(args.balanced_overlap),
        "balanced_min_abs_intersection": int(args.balanced_min_abs_intersection),
        "balanced_min_prev_coverage": float(args.balanced_min_prev_coverage),
        "balanced_tiny_iou": float(args.balanced_tiny_iou),
        "allow_part_merge": bool(args.allow_part_merge),
        "part_merge_boundary_allowlist": [
            {"scene_id": scene, "prev_chunk_index": prev, "curr_chunk_index": curr}
            for scene, prev, curr in sorted(part_merge_boundary_allowlist)
        ],
        "part_merge_min_score": float(args.part_merge_min_score),
        "part_merge_min_color": float(args.part_merge_min_color),
        "part_merge_min_spatial": float(args.part_merge_min_spatial),
        "allow_weak_overlap_override": bool(args.allow_weak_overlap_override),
        "override_min_score": float(args.override_min_score),
        "override_min_color": float(args.override_min_color),
        "override_min_spatial": float(args.override_min_spatial),
        "allow_tiny_lock_expansion": bool(args.allow_tiny_lock_expansion),
        "tiny_lock_expansion_boundary_allowlist": [
            {"scene_id": scene, "prev_chunk_index": prev, "curr_chunk_index": curr}
            for scene, prev, curr in sorted(tiny_lock_expansion_boundary_allowlist)
        ],
        "tiny_lock_expansion_max_iou": float(args.tiny_lock_expansion_max_iou),
        "tiny_lock_expansion_max_prev_coverage": float(args.tiny_lock_expansion_max_prev_coverage),
        "tiny_lock_expansion_min_score": float(args.tiny_lock_expansion_min_score),
        "tiny_lock_expansion_min_color": float(args.tiny_lock_expansion_min_color),
        "tiny_lock_expansion_min_spatial": float(args.tiny_lock_expansion_min_spatial),
        "tiny_lock_expansion_min_area_ratio_vs_lock": float(args.tiny_lock_expansion_min_area_ratio_vs_lock),
        "tiny_lock_expansion_max_per_prev": int(args.tiny_lock_expansion_max_per_prev),
        "require_relaxed_merge_object_witness": bool(args.require_relaxed_merge_object_witness),
        "relaxed_merge_min_bbox_iou": float(args.relaxed_merge_min_bbox_iou),
        "relaxed_merge_max_bbox_gap_norm": float(args.relaxed_merge_max_bbox_gap_norm),
        "relaxed_merge_max_center_dist_norm": float(args.relaxed_merge_max_center_dist_norm),
        "evidence_boundary_remap": [
            {
                "scene_id": scene,
                "prev_chunk_index": prev,
                "curr_chunk_index": curr,
                "remap": {str(k): int(v) for k, v in sorted(remap.items())},
            }
            for (scene, prev, curr), remap in sorted(evidence_boundary_remaps.items())
        ],
        "id_only_stitch_candidate": True,
        "mask_geometry_modified": False,
        "complete_scene_prediction_candidate": all_complete_masks,
        "continuous_scene_level_id_claim": False,
        "claim_boundary": "ID-only local2history reappearance repair over complete scene masks. It does not claim final continuous scene-level identity until boundary visual review and user confirmation pass.",
    }
    _write_json(output_root / "fullscene_l2h_reappearance_summary.json", output_summary)
    _write_json(
        output_root / "hashes.json",
        {
            "summary_sha256": _sha256(output_root / "fullscene_l2h_reappearance_summary.json"),
            "scene_rows_sha256": _sha256(output_root / "scene_rows.json"),
            "sheet_records_sha256": _sha256(output_root / "sheet_records.json"),
            "boundary_sheet_records_sha256": _sha256(output_root / "boundary_sheet_records.json"),
        },
    )
    print(json.dumps({**output_summary, "hashes": _read_json(output_root / "hashes.json")}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
