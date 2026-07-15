#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_v105_fullscene_l2h_reappearance_repair import (  # type: ignore
    REPO_ROOT,
    _balanced_boundary_matches,
    _boundary_matches,
    _build_descriptors,
    _current_object_pair_witness,
    _descriptor_candidate,
    _numeric_stem,
    _read_json,
    _read_label,
    _scene_chunks,
    _sha256,
    _tiny_overlap_lock_details,
    _weak_overlap_prev_locks,
    _write_json,
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def _sort_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("appearance_score", 0.0)),
            float(row.get("color_intersection", 0.0)),
            float(row.get("area_similarity", 0.0)),
            float(row.get("spatial_similarity", 0.0)),
        ),
        reverse=True,
    )


def _small_candidate(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    keys = [
        "prev_global_id",
        "curr_local_id",
        "appearance_score",
        "color_intersection",
        "spatial_similarity",
        "area_similarity",
        "shape_similarity",
        "center_distance_norm",
        "prev_frames",
        "curr_frames",
        "prev_mean_area",
        "curr_mean_area",
        "curr_margin",
        "prev_margin",
    ]
    return {key: row.get(key) for key in keys if key in row}


def _margin(row: dict[str, Any], by_curr: dict[int, list[dict[str, Any]]], by_prev: dict[int, list[dict[str, Any]]]) -> dict[str, float]:
    curr_id = int(row["curr_local_id"])
    prev_id = int(row["prev_global_id"])
    curr_list = by_curr.get(curr_id, [])
    prev_list = by_prev.get(prev_id, [])
    curr_margin = float(row["appearance_score"]) - float(curr_list[1]["appearance_score"]) if len(curr_list) > 1 else 1.0
    prev_margin = float(row["appearance_score"]) - float(prev_list[1]["appearance_score"]) if len(prev_list) > 1 else 1.0
    return {"curr_margin": float(curr_margin), "prev_margin": float(prev_margin)}


def _build_candidate_tables(
    prev_desc: dict[int, dict[str, Any]],
    curr_desc: dict[int, dict[str, Any]],
    existing_mapping: dict[int, int],
    *,
    min_score: float,
    min_color: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    all_candidates: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for curr_id, curr in curr_desc.items():
        if int(curr_id) in existing_mapping:
            continue
        for prev_id, prev in prev_desc.items():
            row = _descriptor_candidate(prev, curr)
            all_candidates.append(row)
            if float(row["appearance_score"]) >= float(min_score) and float(row["color_intersection"]) >= float(min_color):
                candidates.append(row)
    candidates = _sort_candidates(candidates)
    all_candidates = _sort_candidates(all_candidates)
    by_curr: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_prev: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_curr[int(row["curr_local_id"])].append(row)
        by_prev[int(row["prev_global_id"])].append(row)
    return all_candidates, candidates, by_curr, by_prev


def _simulate_matching(
    *,
    candidates: list[dict[str, Any]],
    all_candidates: list[dict[str, Any]],
    by_curr: dict[int, list[dict[str, Any]]],
    by_prev: dict[int, list[dict[str, Any]]],
    curr_desc: dict[int, dict[str, Any]],
    existing_mapping: dict[int, int],
    weak_existing_prev_to_curr: dict[int, int],
    tiny_lock_prev_details: dict[int, dict[str, Any]],
    min_margin: float,
    allow_weak_overlap_override: bool,
    override_min_score: float,
    override_min_color: float,
    override_min_spatial: float,
    allow_part_merge: bool,
    part_merge_min_score: float,
    part_merge_min_color: float,
    part_merge_min_spatial: float,
    allow_tiny_lock_expansion: bool,
    tiny_lock_expansion_min_score: float,
    tiny_lock_expansion_min_color: float,
    tiny_lock_expansion_min_spatial: float,
    tiny_lock_expansion_min_area_ratio_vs_lock: float,
    tiny_lock_expansion_max_per_prev: int,
    require_relaxed_merge_object_witness: bool,
    relaxed_merge_min_bbox_iou: float,
    relaxed_merge_max_bbox_gap_norm: float,
    relaxed_merge_max_center_dist_norm: float,
) -> tuple[dict[int, int], dict[int, list[dict[str, Any]]]]:
    mapping: dict[int, int] = {}
    events: dict[int, list[dict[str, Any]]] = defaultdict(list)
    used_prev: set[int] = set()
    used_curr: set[int] = set()
    existing_prev = set(int(v) for v in existing_mapping.values())
    existing_curr_by_prev: dict[int, int] = {}
    for curr_id, prev_id in sorted(existing_mapping.items()):
        existing_curr_by_prev.setdefault(int(prev_id), int(curr_id))

    for row in candidates:
        curr_id = int(row["curr_local_id"])
        prev_id = int(row["prev_global_id"])
        if curr_id in used_curr:
            events[curr_id].append({"phase": "direct_one_to_one", "reason": "curr_already_used", "candidate": _small_candidate(row)})
            continue
        if prev_id in used_prev:
            events[curr_id].append({"phase": "direct_one_to_one", "reason": "prev_already_used_by_competition", "candidate": _small_candidate(row)})
            continue
        if prev_id in existing_prev:
            events[curr_id].append({"phase": "direct_one_to_one", "reason": "prev_already_overlap_existing", "candidate": _small_candidate(row)})
            continue
        margins = _margin(row, by_curr, by_prev)
        row_with_margin = {**row, **margins}
        if margins["curr_margin"] < float(min_margin) or margins["prev_margin"] < float(min_margin):
            events[curr_id].append({"phase": "direct_one_to_one", "reason": "low_top1_margin", "candidate": _small_candidate(row_with_margin)})
            continue
        mapping[curr_id] = prev_id
        used_curr.add(curr_id)
        used_prev.add(prev_id)
        events[curr_id].append({"phase": "direct_one_to_one", "reason": "accepted", "candidate": _small_candidate(row_with_margin)})

    if allow_weak_overlap_override:
        for row in candidates:
            curr_id = int(row["curr_local_id"])
            prev_id = int(row["prev_global_id"])
            old_curr_id = weak_existing_prev_to_curr.get(prev_id)
            if old_curr_id is None:
                continue
            if curr_id in used_curr or curr_id in existing_mapping:
                events[curr_id].append({"phase": "weak_overlap_override", "reason": "curr_unavailable", "candidate": _small_candidate(row)})
                continue
            if float(row["appearance_score"]) < float(override_min_score):
                events[curr_id].append({"phase": "weak_overlap_override", "reason": "score_below_override", "candidate": _small_candidate(row)})
                continue
            if float(row["color_intersection"]) < float(override_min_color):
                events[curr_id].append({"phase": "weak_overlap_override", "reason": "color_below_override", "candidate": _small_candidate(row)})
                continue
            if float(row["spatial_similarity"]) < float(override_min_spatial):
                events[curr_id].append({"phase": "weak_overlap_override", "reason": "spatial_below_override", "candidate": _small_candidate(row)})
                continue
            mapping[curr_id] = prev_id
            used_curr.add(curr_id)
            events[curr_id].append(
                {
                    "phase": "weak_overlap_override",
                    "reason": "accepted",
                    "overridden_curr_local_id": int(old_curr_id),
                    "candidate": _small_candidate(row),
                }
            )

    if allow_part_merge:
        for row in candidates:
            curr_id = int(row["curr_local_id"])
            prev_id = int(row["prev_global_id"])
            if curr_id in used_curr:
                events[curr_id].append({"phase": "part_merge", "reason": "curr_already_used", "candidate": _small_candidate(row)})
                continue
            if curr_id in existing_mapping:
                events[curr_id].append({"phase": "part_merge", "reason": "curr_already_overlap_existing", "candidate": _small_candidate(row)})
                continue
            if prev_id not in existing_prev:
                continue
            if float(row["appearance_score"]) < float(part_merge_min_score):
                events[curr_id].append({"phase": "part_merge", "reason": "score_below_part_merge", "candidate": _small_candidate(row)})
                continue
            if float(row["color_intersection"]) < float(part_merge_min_color):
                events[curr_id].append({"phase": "part_merge", "reason": "color_below_part_merge", "candidate": _small_candidate(row)})
                continue
            if float(row["spatial_similarity"]) < float(part_merge_min_spatial):
                events[curr_id].append({"phase": "part_merge", "reason": "spatial_below_part_merge", "candidate": _small_candidate(row)})
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
                events[curr_id].append(
                    {
                        "phase": "part_merge",
                        "reason": "object_witness_rejected",
                        "anchor_curr_local_id": int(anchor_curr_id),
                        "candidate": _small_candidate(row),
                        "object_witness": object_witness,
                    }
                )
                continue
            mapping[curr_id] = prev_id
            used_curr.add(curr_id)
            events[curr_id].append(
                {
                    "phase": "part_merge",
                    "reason": "accepted",
                    "anchor_curr_local_id": int(anchor_curr_id),
                    "candidate": _small_candidate(row),
                    "object_witness": object_witness,
                }
            )

    if allow_tiny_lock_expansion:
        expansion_count_by_prev: dict[int, int] = {}
        for row in all_candidates:
            curr_id = int(row["curr_local_id"])
            prev_id = int(row["prev_global_id"])
            lock = tiny_lock_prev_details.get(prev_id)
            if lock is None:
                continue
            if curr_id in used_curr:
                events[curr_id].append({"phase": "tiny_lock_expansion", "reason": "curr_already_used", "candidate": _small_candidate(row)})
                continue
            if curr_id in existing_mapping:
                events[curr_id].append({"phase": "tiny_lock_expansion", "reason": "curr_already_overlap_existing", "candidate": _small_candidate(row)})
                continue
            if float(row["appearance_score"]) < float(tiny_lock_expansion_min_score):
                events[curr_id].append({"phase": "tiny_lock_expansion", "reason": "score_below_tiny_lock", "candidate": _small_candidate(row)})
                continue
            if float(row["color_intersection"]) < float(tiny_lock_expansion_min_color):
                events[curr_id].append({"phase": "tiny_lock_expansion", "reason": "color_below_tiny_lock", "candidate": _small_candidate(row)})
                continue
            if float(row["spatial_similarity"]) < float(tiny_lock_expansion_min_spatial):
                events[curr_id].append({"phase": "tiny_lock_expansion", "reason": "spatial_below_tiny_lock", "candidate": _small_candidate(row)})
                continue
            locked_curr_id = int(lock.get("curr_local_id", 0) or 0)
            locked_desc = curr_desc.get(locked_curr_id)
            locked_area = float(locked_desc.get("mean_area", 0.0)) if locked_desc else float(lock.get("curr_area") or 0.0)
            if locked_area > 0 and float(row["curr_mean_area"]) < locked_area * float(tiny_lock_expansion_min_area_ratio_vs_lock):
                events[curr_id].append({"phase": "tiny_lock_expansion", "reason": "area_ratio_below_tiny_lock", "candidate": _small_candidate(row)})
                continue
            if expansion_count_by_prev.get(prev_id, 0) >= int(tiny_lock_expansion_max_per_prev):
                events[curr_id].append({"phase": "tiny_lock_expansion", "reason": "max_per_prev_reached", "candidate": _small_candidate(row)})
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
                events[curr_id].append(
                    {
                        "phase": "tiny_lock_expansion",
                        "reason": "object_witness_rejected",
                        "locked_curr_local_id": int(locked_curr_id),
                        "candidate": _small_candidate(row),
                        "object_witness": object_witness,
                    }
                )
                continue
            mapping[curr_id] = prev_id
            used_curr.add(curr_id)
            expansion_count_by_prev[prev_id] = expansion_count_by_prev.get(prev_id, 0) + 1
            events[curr_id].append(
                {
                    "phase": "tiny_lock_expansion",
                    "reason": "accepted",
                    "locked_curr_local_id": int(locked_curr_id),
                    "candidate": _small_candidate(row),
                    "object_witness": object_witness,
                }
            )

    return mapping, events


def _category_for_row(
    *,
    curr_id: int,
    curr_desc: dict[int, dict[str, Any]],
    prev_desc: dict[int, dict[str, Any]],
    existing_mapping: dict[int, int],
    all_candidates: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    simulated_mapping: dict[int, int],
    events: dict[int, list[dict[str, Any]]],
    part_merge_min_score: float,
    part_merge_min_color: float,
    part_merge_min_spatial: float,
) -> tuple[str, dict[str, Any]]:
    curr_events = events.get(int(curr_id), [])
    if curr_id in simulated_mapping:
        return "diagnostic_mismatch_would_have_mapped", {
            "simulated_prev_global_id": int(simulated_mapping[curr_id]),
            "events_first10": curr_events[:10],
        }
    if curr_id not in curr_desc:
        return "curr_descriptor_missing_or_area_below_min", {"events_first10": curr_events[:10]}
    if not prev_desc:
        return "prev_descriptor_missing", {"events_first10": curr_events[:10]}
    curr_all = [row for row in all_candidates if int(row["curr_local_id"]) == int(curr_id)]
    curr_pass = [row for row in candidates if int(row["curr_local_id"]) == int(curr_id)]
    top_any = curr_all[0] if curr_all else None
    top_pass = curr_pass[0] if curr_pass else None
    object_rejects = [ev for ev in curr_events if ev.get("reason") == "object_witness_rejected"]
    if object_rejects:
        return "object_witness_rejected", {
            "top_any": _small_candidate(top_any),
            "top_pass": _small_candidate(top_pass),
            "object_rejects_first5": object_rejects[:5],
            "events_first10": curr_events[:10],
        }
    part_existing = [
        row
        for row in curr_pass
        if int(row["prev_global_id"]) in set(int(v) for v in existing_mapping.values())
        and float(row["appearance_score"]) >= float(part_merge_min_score)
        and float(row["color_intersection"]) >= float(part_merge_min_color)
        and float(row["spatial_similarity"]) >= float(part_merge_min_spatial)
    ]
    if part_existing:
        return "part_merge_threshold_pass_without_object_witness_event", {
            "top_any": _small_candidate(top_any),
            "top_pass": _small_candidate(top_pass),
            "part_existing_first5": [_small_candidate(row) for row in part_existing[:5]],
            "events_first10": curr_events[:10],
        }
    low_margin = [ev for ev in curr_events if ev.get("reason") == "low_top1_margin"]
    if low_margin:
        return "direct_candidate_low_top1_margin", {
            "top_any": _small_candidate(top_any),
            "top_pass": _small_candidate(top_pass),
            "low_margin_first5": low_margin[:5],
            "events_first10": curr_events[:10],
        }
    prev_comp = [ev for ev in curr_events if ev.get("reason") == "prev_already_used_by_competition"]
    if prev_comp:
        return "direct_candidate_prev_used_by_competition", {
            "top_any": _small_candidate(top_any),
            "top_pass": _small_candidate(top_pass),
            "prev_competition_first5": prev_comp[:5],
            "events_first10": curr_events[:10],
        }
    existing_prev_events = [ev for ev in curr_events if ev.get("reason") == "prev_already_overlap_existing"]
    if existing_prev_events:
        return "candidate_points_to_existing_prev_but_part_merge_threshold_not_met", {
            "top_any": _small_candidate(top_any),
            "top_pass": _small_candidate(top_pass),
            "existing_prev_events_first5": existing_prev_events[:5],
            "events_first10": curr_events[:10],
        }
    if not curr_pass:
        if top_any is None:
            return "no_candidate_rows", {"events_first10": curr_events[:10]}
        if float(top_any["appearance_score"]) < 0.55:
            return "top_candidate_score_low", {"top_any": _small_candidate(top_any), "events_first10": curr_events[:10]}
        if float(top_any["color_intersection"]) < 0.25:
            return "top_candidate_color_low", {"top_any": _small_candidate(top_any), "events_first10": curr_events[:10]}
        return "no_candidate_passed_base_thresholds", {"top_any": _small_candidate(top_any), "events_first10": curr_events[:10]}
    return "unclassified_candidate_not_mapped", {
        "top_any": _small_candidate(top_any),
        "top_pass": _small_candidate(top_pass),
        "events_first10": curr_events[:10],
    }


def build(l2h_summary_path: Path, post_l2h_records_path: Path, output_root: Path) -> dict[str, Any]:
    l2h = json.loads(l2h_summary_path.read_text(encoding="utf-8"))
    post_records = json.loads(post_l2h_records_path.read_text(encoding="utf-8"))
    post_by_boundary = {
        (
            str(row.get("scene_id")),
            int(row.get("prev_chunk_index")),
            int(row.get("curr_chunk_index")),
        ): row
        for row in post_records
        if isinstance(row, dict)
    }

    source_root = _resolve(str(l2h["source_fullscene_root"]))
    plan = _read_json(source_root / "fullscene_multichunk_plan.json")
    descriptor_window = int(l2h.get("descriptor_window", 6))
    descriptor_min_area = int(l2h.get("descriptor_min_area", 2500))
    min_iou = float(l2h.get("min_iou", 0.02))
    min_overlap_min = float(l2h.get("min_overlap_min", 0.08))
    balanced_overlap = bool(l2h.get("balanced_overlap", False))
    balanced_min_abs_intersection = int(l2h.get("balanced_min_abs_intersection", 5000))
    balanced_min_prev_coverage = float(l2h.get("balanced_min_prev_coverage", 0.03))
    balanced_tiny_iou = float(l2h.get("balanced_tiny_iou", 0.03))

    records: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    boundary_counts: Counter[str] = Counter()

    for scene in l2h.get("scene_rows", []) or []:
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene["scene_id"])
        source_mask_dir = _resolve(str(scene["source_mask_dir"]))
        output_mask_dir = _resolve(str(scene["mask_dir"]))
        scene_color_dir = REPO_ROOT / "Stream3D/data/scannet/processed" / scene_id / "color"
        frame_ids = sorted([_numeric_stem(path) for path in source_mask_dir.glob("*.png") if _numeric_stem(path) < 10**12])
        chunks = {int(chunk["chunk_index"]): chunk for chunk in _scene_chunks(plan, scene_id, frame_ids)}

        for boundary in scene.get("boundary_records", []) or []:
            if not isinstance(boundary, dict):
                continue
            boundary_key = (scene_id, int(boundary["prev_chunk_index"]), int(boundary["curr_chunk_index"]))
            post = post_by_boundary.get(boundary_key)
            if not post:
                continue
            new_rows = post.get("new_large_first20", []) or []
            if not new_rows:
                continue
            prev_chunk = chunks.get(int(boundary["prev_chunk_index"]))
            curr_chunk = chunks.get(int(boundary["curr_chunk_index"]))
            if not prev_chunk or not curr_chunk:
                continue
            prev_frame_ids = frame_ids[int(prev_chunk["start_index"]) : int(prev_chunk["start_index"]) + int(prev_chunk["frame_count"])]
            curr_frame_ids = frame_ids[int(curr_chunk["start_index"]) : int(curr_chunk["start_index"]) + int(curr_chunk["frame_count"])]
            prev_items = [
                (int(frame_id), _read_label(output_mask_dir / f"{int(frame_id)}.png"))
                for frame_id in prev_frame_ids[-max(descriptor_window, 1) :]
            ]
            curr_items = [
                (int(frame_id), _read_label(source_mask_dir / f"{int(frame_id)}.png"))
                for frame_id in curr_frame_ids[: max(descriptor_window, 1)]
            ]
            prev_global = _read_label(output_mask_dir / f"{int(boundary['prev_frame_id'])}.png")
            curr_local = _read_label(source_mask_dir / f"{int(boundary['curr_frame_id'])}.png")
            if balanced_overlap:
                existing_mapping, overlap_audit = _balanced_boundary_matches(
                    prev_global=prev_global,
                    curr_local=curr_local,
                    min_iou=min_iou,
                    min_overlap_min=min_overlap_min,
                    min_abs_intersection=balanced_min_abs_intersection,
                    min_prev_coverage_for_tiny_iou=balanced_min_prev_coverage,
                    tiny_iou=balanced_tiny_iou,
                )
            else:
                existing_mapping, overlap_audit = _boundary_matches(
                    prev_global=prev_global,
                    curr_local=curr_local,
                    min_iou=min_iou,
                    min_overlap_min=min_overlap_min,
                )
            weak_locks = _weak_overlap_prev_locks(
                overlap_audit,
                min_abs_intersection=balanced_min_abs_intersection,
                min_prev_coverage_for_tiny_iou=balanced_min_prev_coverage,
                tiny_iou=balanced_tiny_iou,
            )
            tiny_details = _tiny_overlap_lock_details(
                overlap_audit,
                max_iou=float(l2h.get("tiny_lock_expansion_max_iou", 0.08)),
                max_prev_coverage=float(l2h.get("tiny_lock_expansion_max_prev_coverage", 0.08)),
            )
            prev_desc = _build_descriptors(label_items=prev_items, scene_color_dir=scene_color_dir, min_area=descriptor_min_area)
            curr_desc = _build_descriptors(label_items=curr_items, scene_color_dir=scene_color_dir, min_area=descriptor_min_area)
            all_candidates, candidates, by_curr, by_prev = _build_candidate_tables(
                prev_desc,
                curr_desc,
                existing_mapping,
                min_score=float(l2h.get("appearance_min_score", 0.58)),
                min_color=float(l2h.get("appearance_min_color", 0.25)),
            )
            simulated_mapping, events = _simulate_matching(
                candidates=candidates,
                all_candidates=all_candidates,
                by_curr=by_curr,
                by_prev=by_prev,
                curr_desc=curr_desc,
                existing_mapping=existing_mapping,
                weak_existing_prev_to_curr=weak_locks,
                tiny_lock_prev_details=tiny_details,
                min_margin=float(l2h.get("appearance_min_margin", 0.03)),
                allow_weak_overlap_override=bool(l2h.get("allow_weak_overlap_override", False)),
                override_min_score=float(l2h.get("override_min_score", 0.66)),
                override_min_color=float(l2h.get("override_min_color", 0.50)),
                override_min_spatial=float(l2h.get("override_min_spatial", 0.25)),
                allow_part_merge=bool(l2h.get("allow_part_merge", False)),
                part_merge_min_score=float(l2h.get("part_merge_min_score", 0.66)),
                part_merge_min_color=float(l2h.get("part_merge_min_color", 0.50)),
                part_merge_min_spatial=float(l2h.get("part_merge_min_spatial", 0.25)),
                allow_tiny_lock_expansion=bool(l2h.get("allow_tiny_lock_expansion", False)),
                tiny_lock_expansion_min_score=float(l2h.get("tiny_lock_expansion_min_score", 0.40)),
                tiny_lock_expansion_min_color=float(l2h.get("tiny_lock_expansion_min_color", 0.15)),
                tiny_lock_expansion_min_spatial=float(l2h.get("tiny_lock_expansion_min_spatial", 0.35)),
                tiny_lock_expansion_min_area_ratio_vs_lock=float(l2h.get("tiny_lock_expansion_min_area_ratio_vs_lock", 1.20)),
                tiny_lock_expansion_max_per_prev=int(l2h.get("tiny_lock_expansion_max_per_prev", 3)),
                require_relaxed_merge_object_witness=bool(l2h.get("require_relaxed_merge_object_witness", False)),
                relaxed_merge_min_bbox_iou=float(l2h.get("relaxed_merge_min_bbox_iou", 0.01)),
                relaxed_merge_max_bbox_gap_norm=float(l2h.get("relaxed_merge_max_bbox_gap_norm", 0.03)),
                relaxed_merge_max_center_dist_norm=float(l2h.get("relaxed_merge_max_center_dist_norm", 0.16)),
            )

            boundary_counts[f"{scene_id}:{boundary['prev_chunk_index']}-{boundary['curr_chunk_index']}"] += len(new_rows)
            for row in new_rows:
                if not isinstance(row, dict):
                    continue
                curr_id = int(row["curr_local_id"])
                category, evidence = _category_for_row(
                    curr_id=curr_id,
                    curr_desc=curr_desc,
                    prev_desc=prev_desc,
                    existing_mapping=existing_mapping,
                    all_candidates=all_candidates,
                    candidates=candidates,
                    simulated_mapping=simulated_mapping,
                    events=events,
                    part_merge_min_score=float(l2h.get("part_merge_min_score", 0.66)),
                    part_merge_min_color=float(l2h.get("part_merge_min_color", 0.50)),
                    part_merge_min_spatial=float(l2h.get("part_merge_min_spatial", 0.25)),
                )
                category_counts[category] += 1
                scene_counts[scene_id][category] += 1
                records.append(
                    {
                        "scene_id": scene_id,
                        "prev_chunk_index": int(boundary["prev_chunk_index"]),
                        "curr_chunk_index": int(boundary["curr_chunk_index"]),
                        "prev_frame_id": int(boundary["prev_frame_id"]),
                        "curr_frame_id": int(boundary["curr_frame_id"]),
                        "curr_local_id": curr_id,
                        "output_global_id": int(row.get("output_global_id", 0) or 0),
                        "local_area": int(row.get("area", 0) or 0),
                        "category": category,
                        "prev_descriptor_count": len(prev_desc),
                        "curr_descriptor_count": len(curr_desc),
                        "existing_overlap_mapping_count": len(existing_mapping),
                        "weak_lock_count": len(weak_locks),
                        "tiny_lock_count": len(tiny_details),
                        "base_candidate_count_for_curr": sum(1 for cand in candidates if int(cand["curr_local_id"]) == curr_id),
                        "all_candidate_count_for_curr": sum(1 for cand in all_candidates if int(cand["curr_local_id"]) == curr_id),
                        "evidence": evidence,
                    }
                )

    records_path = output_root / "remaining_new_birth_records.json"
    _write_json(records_path, records)
    summary = {
        "schema_version": "stream4d_v105_l2h_remaining_new_birth_diagnostic_v1",
        "l2h_summary": _rel(l2h_summary_path),
        "l2h_summary_sha256": _sha256(l2h_summary_path),
        "post_l2h_records": _rel(post_l2h_records_path),
        "post_l2h_records_sha256": _sha256(post_l2h_records_path),
        "variant_id": l2h.get("variant_id"),
        "record_count": len(records),
        "category_counts": dict(sorted(category_counts.items())),
        "scene_category_counts": {scene: dict(sorted(counter.items())) for scene, counter in sorted(scene_counts.items())},
        "boundary_new_large_counts": dict(sorted(boundary_counts.items())),
        "records_json": _rel(records_path),
        "records_sha256": _sha256(records_path),
        "claim_boundary": (
            "Diagnostic only. Categories are recomputed from local2history descriptors, overlap mappings, "
            "competition rules, and same-frame object-witness checks. This does not prove visual identity."
        ),
    }
    _write_json(output_root / "remaining_new_birth_summary.json", summary)
    _write_json(
        output_root / "hashes.json",
        {
            "summary_sha256": _sha256(output_root / "remaining_new_birth_summary.json"),
            "records_sha256": _sha256(records_path),
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose remaining post-L2H large new births for a v105 L2H variant.")
    parser.add_argument("--l2h-summary", required=True)
    parser.add_argument("--post-l2h-records", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    summary = build(_resolve(args.l2h_summary), _resolve(args.post_l2h_records), _resolve(args.output_root))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
