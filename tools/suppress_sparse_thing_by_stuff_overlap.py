#!/usr/bin/env python3
"""Suppress likely static-roadside false thing tracks using auxiliary stuff masks.

This is an auditable offline post-process: it does not create manual merge
groups and it does not edit per-track labels.  It removes whole thing tracks
whose masks repeatedly overlap static/risky stuff masks such as pole or
billboard.  The intended use is to keep sign/pole prompts as a suppression
signal instead of emitting them as extra thing IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import (  # noqa: E402
    _unpack_mask_np,
    clone_sparse,
    coverage_stats,
    load_sparse,
    make_contact_sheet,
    parse_contact_frames,
    refresh_track_frame,
    track_stats,
)
from run_video_masklet_front_end import collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402
from loger.pipeline.video_masklet_frontend import canonicalize_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suppress thing tracks that overlap static/risky stuff masks.")
    parser.add_argument("--thing_pt", required=True)
    parser.add_argument("--stuff_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--thing_labels", default="person,car")
    parser.add_argument(
        "--suppress_stuff_labels",
        default="billboard_or_bulletin_board,pole,handrail_or_fence,traffic_sign,traffic sign,sign,signboard",
    )
    parser.add_argument("--containment_threshold", type=float, default=0.50)
    parser.add_argument("--person_mean_containment", type=float, default=0.55)
    parser.add_argument("--person_spike_containment", type=float, default=0.85)
    parser.add_argument("--person_spike_min_mean", type=float, default=0.30)
    parser.add_argument("--person_spike_min_frames", type=int, default=2)
    parser.add_argument("--drop_short_person_max_frames", type=int, default=2)
    parser.add_argument("--car_mean_containment", type=float, default=0.60)
    parser.add_argument("--car_spike_containment", type=float, default=0.70)
    parser.add_argument("--car_spike_min_frames", type=int, default=2)
    parser.add_argument("--car_spike_max_visible_frames", type=int, default=25)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--contact_frames", default="auto")
    parser.add_argument("--prune_tiny_duplicate_fragments", type=int, default=0)
    parser.add_argument("--prune_duplicate_labels", default="car")
    parser.add_argument("--prune_max_area_px", type=int, default=24)
    parser.add_argument("--prune_min_containment", type=float, default=0.50)
    parser.add_argument("--prune_min_box_iou", type=float, default=0.50)
    return parser.parse_args()


def _parse_csv(value: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in str(value or "").split(","):
        label = canonicalize_label(item.strip())
        if not label:
            continue
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def _shape_check(thing: Any, stuff: Any) -> None:
    if int(thing.num_frames) != int(stuff.num_frames):
        raise RuntimeError(f"Frame-count mismatch: thing={thing.num_frames}, stuff={stuff.num_frames}")
    if (int(thing.frame_height), int(thing.frame_width)) != (int(stuff.frame_height), int(stuff.frame_width)):
        raise RuntimeError(
            "Frame-shape mismatch: "
            f"thing={(thing.frame_height, thing.frame_width)}, stuff={(stuff.frame_height, stuff.frame_width)}"
        )


def _load_processing_frames(
    input_video: str,
    processing_max_side: int,
    frames_limit: int,
    expected_h: int,
    expected_w: int,
    num_frames: int,
) -> Tuple[List[str], List[str]]:
    image_paths, temp_dir = collect_image_paths(input_video, 0, -1, 1)
    temp_dirs = [temp_dir] if temp_dir else []
    if int(frames_limit) > 0:
        image_paths = image_paths[: int(frames_limit)]
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(
        image_paths,
        int(processing_max_side),
    )
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    if len(image_paths) < int(num_frames):
        raise RuntimeError(f"Need at least {num_frames} frames, got {len(image_paths)}")
    image_paths = image_paths[: int(num_frames)]
    if tuple(proc_shape) != (int(expected_h), int(expected_w)):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    return list(image_paths), temp_dirs


def _stuff_union_by_frame(stuff: Any, labels: Sequence[str]) -> Dict[int, Dict[str, np.ndarray]]:
    wanted = set(labels)
    H, W = int(stuff.frame_height), int(stuff.frame_width)
    by_frame: Dict[int, Dict[str, np.ndarray]] = {}
    for track in stuff.tracks:
        if str(track.get("source_type")) != "stuff_static":
            continue
        label = canonicalize_label(str(track.get("L_sem", "")))
        if label not in wanted:
            continue
        for frame_idx, packed in track.get("mask_by_frame", {}).items():
            mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W)
            frame_dict = by_frame.setdefault(int(frame_idx), {})
            if label in frame_dict:
                frame_dict[label] |= mask
            else:
                frame_dict[label] = mask.astype(bool)
    return by_frame


def _track_overlap_stats(
    track: Dict[str, Any],
    by_frame: Dict[int, Dict[str, np.ndarray]],
    H: int,
    W: int,
    containment_threshold: float,
) -> Dict[str, Any]:
    frames = sorted(int(f) for f in track.get("mask_by_frame", {}).keys())
    containments: List[float] = []
    ious: List[float] = []
    best_labels: List[str] = []
    per_label: Dict[str, List[float]] = {}
    for frame_idx in frames:
        packed = track.get("mask_by_frame", {}).get(frame_idx)
        if packed is None:
            continue
        thing_mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W)
        thing_area = int(thing_mask.sum())
        if thing_area <= 0:
            continue
        best_containment = 0.0
        best_iou = 0.0
        best_label = ""
        for stuff_label, stuff_mask in by_frame.get(frame_idx, {}).items():
            inter = int(np.logical_and(thing_mask, stuff_mask).sum())
            if inter <= 0:
                continue
            containment = float(inter) / float(max(thing_area, 1))
            union = int(np.logical_or(thing_mask, stuff_mask).sum())
            iou = float(inter) / float(max(union, 1))
            per_label.setdefault(stuff_label, []).append(containment)
            if containment > best_containment:
                best_containment = containment
                best_iou = iou
                best_label = stuff_label
        containments.append(best_containment)
        ious.append(best_iou)
        best_labels.append(best_label)

    if not containments:
        return {
            "visible_frames": len(frames),
            "mean_containment": 0.0,
            "max_containment": 0.0,
            "frames_over_threshold": 0,
            "over_threshold_ratio": 0.0,
            "mean_iou": 0.0,
            "max_iou": 0.0,
            "dominant_stuff_label": "",
            "per_label_mean_containment": {},
        }

    over = [value for value in containments if value >= float(containment_threshold)]
    per_label_mean = {
        label: float(np.mean(values))
        for label, values in sorted(per_label.items())
        if values
    }
    dominant_label = ""
    if per_label_mean:
        dominant_label = max(per_label_mean, key=lambda key: per_label_mean[key])

    return {
        "visible_frames": len(frames),
        "mean_containment": float(np.mean(containments)),
        "max_containment": float(np.max(containments)),
        "frames_over_threshold": int(len(over)),
        "over_threshold_ratio": float(len(over)) / float(max(len(containments), 1)),
        "mean_iou": float(np.mean(ious)),
        "max_iou": float(np.max(ious)),
        "dominant_stuff_label": dominant_label,
        "per_label_mean_containment": per_label_mean,
    }


def _should_drop(label: str, stats: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    label = canonicalize_label(label)
    visible = int(stats["visible_frames"])
    mean_containment = float(stats["mean_containment"])
    max_containment = float(stats["max_containment"])
    over_frames = int(stats["frames_over_threshold"])

    if label == "person":
        if visible <= int(args.drop_short_person_max_frames):
            return True, "short_person_track"
        if mean_containment >= float(args.person_mean_containment):
            return True, "person_mean_static_stuff_containment"
        if (
            max_containment >= float(args.person_spike_containment)
            and mean_containment >= float(args.person_spike_min_mean)
            and over_frames >= int(args.person_spike_min_frames)
        ):
            return True, "person_static_stuff_spike"
        return False, ""

    if label == "car":
        if mean_containment >= float(args.car_mean_containment):
            return True, "car_mean_static_stuff_containment"
        if (
            visible <= int(args.car_spike_max_visible_frames)
            and max_containment >= float(args.car_spike_containment)
            and over_frames >= int(args.car_spike_min_frames)
        ):
            return True, "short_car_static_stuff_spike"
        return False, ""

    return False, ""


def _auto_contact_frames(rows: Sequence[Dict[str, Any]], limit: int) -> List[int]:
    frames: List[int] = []
    for row in rows:
        if not bool(row.get("dropped")):
            continue
        for key in ("birth_frame", "end_frame"):
            frame = int(row.get(key, -1))
            for delta in (-2, -1, 0, 1, 2):
                idx = frame + delta
                if 0 <= idx < limit:
                    frames.append(idx)
    deduped = sorted(set(frames))
    return deduped[:40]


def _box_iou(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)[:4]
    bb = np.asarray(b, dtype=np.float32).reshape(-1)[:4]
    if aa.size < 4 or bb.size < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in aa]
    bx1, by1, bx2, by2 = [float(v) for v in bb]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _prune_tiny_duplicate_fragments(
    sparse: Any,
    labels: Sequence[str],
    max_area_px: int,
    min_containment: float,
    min_box_iou: float,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    wanted = set(labels)
    H, W = int(sparse.frame_height), int(sparse.frame_width)
    rows: List[Dict[str, Any]] = []
    touched_frames: set[int] = set()
    for frame_idx in range(int(sparse.num_frames)):
        entries: List[Dict[str, Any]] = []
        for track_index, track in enumerate(sparse.tracks):
            label = canonicalize_label(str(track.get("L_sem", "")))
            if label not in wanted:
                continue
            packed = track.get("mask_by_frame", {}).get(frame_idx)
            if packed is None:
                continue
            mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W)
            area = int(mask.sum())
            if area <= 0:
                continue
            box = track.get("box_by_frame", {}).get(frame_idx)
            entries.append(
                {
                    "track_index": int(track_index),
                    "track": track,
                    "label": label,
                    "mask": mask,
                    "area": int(area),
                    "box": box,
                }
            )
        if len(entries) <= 1:
            continue
        for small in entries:
            if int(small["area"]) > int(max_area_px):
                continue
            best: Optional[Dict[str, Any]] = None
            best_reason = ""
            best_containment = 0.0
            best_box_iou = 0.0
            for other in entries:
                if int(other["track_index"]) == int(small["track_index"]):
                    continue
                if str(other["label"]) != str(small["label"]):
                    continue
                if int(other["area"]) <= int(small["area"]):
                    continue
                inter = int(np.logical_and(small["mask"], other["mask"]).sum())
                containment = float(inter) / float(max(int(small["area"]), 1))
                box_iou = _box_iou(small["box"], other["box"])
                reason = ""
                if containment >= float(min_containment):
                    reason = "tiny_mask_contained_by_same_label"
                elif box_iou >= float(min_box_iou):
                    reason = "tiny_box_overlaps_same_label"
                if not reason:
                    continue
                score = max(containment, box_iou)
                prev_score = max(best_containment, best_box_iou)
                if best is None or score > prev_score:
                    best = other
                    best_reason = reason
                    best_containment = containment
                    best_box_iou = box_iou
            if best is None:
                continue
            refresh_track_frame(small["track"], frame_idx, np.zeros((H, W), dtype=bool), H, W)
            touched_frames.add(int(frame_idx))
            rows.append(
                {
                    "frame": int(frame_idx),
                    "track_index": int(small["track_index"]),
                    "label": str(small["label"]),
                    "area_px": int(small["area"]),
                    "matched_track_index": int(best["track_index"]),
                    "matched_area_px": int(best["area"]),
                    "containment": float(best_containment),
                    "box_iou": float(best_box_iou),
                    "reason": best_reason,
                }
            )
    return rows, sorted(touched_frames)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thing = load_sparse(Path(args.thing_pt))
    stuff = load_sparse(Path(args.stuff_pt))
    _shape_check(thing, stuff)

    output = clone_sparse(thing)
    thing_labels = set(_parse_csv(args.thing_labels))
    stuff_labels = _parse_csv(args.suppress_stuff_labels)
    H, W = int(output.frame_height), int(output.frame_width)
    by_frame = _stuff_union_by_frame(stuff, stuff_labels)

    kept_tracks: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for track_index, track in enumerate(output.tracks):
        label = canonicalize_label(str(track.get("L_sem", "")))
        frames = sorted(int(f) for f in track.get("mask_by_frame", {}).keys())
        row: Dict[str, Any] = {
            "track_index": int(track_index),
            "label": label,
            "source_type": str(track.get("source_type", "")),
            "birth_frame": int(min(frames)) if frames else -1,
            "end_frame": int(max(frames)) if frames else -1,
            "visible_frames": int(len(frames)),
            "dropped": False,
            "drop_reason": "",
        }
        if label in thing_labels and str(track.get("source_type")) != "stuff_static":
            stats = _track_overlap_stats(track, by_frame, H, W, float(args.containment_threshold))
            drop, reason = _should_drop(label, stats, args)
            row.update(
                {
                    "mean_containment": float(stats["mean_containment"]),
                    "max_containment": float(stats["max_containment"]),
                    "frames_over_threshold": int(stats["frames_over_threshold"]),
                    "over_threshold_ratio": float(stats["over_threshold_ratio"]),
                    "mean_iou": float(stats["mean_iou"]),
                    "max_iou": float(stats["max_iou"]),
                    "dominant_stuff_label": str(stats["dominant_stuff_label"]),
                    "per_label_mean_containment": json.dumps(stats["per_label_mean_containment"], sort_keys=True),
                }
            )
            if drop:
                row["dropped"] = True
                row["drop_reason"] = reason
            else:
                kept_tracks.append(track)
        else:
            kept_tracks.append(track)
            row.update(
                {
                    "mean_containment": 0.0,
                    "max_containment": 0.0,
                    "frames_over_threshold": 0,
                    "over_threshold_ratio": 0.0,
                    "mean_iou": 0.0,
                    "max_iou": 0.0,
                    "dominant_stuff_label": "",
                    "per_label_mean_containment": "{}",
                }
            )
        rows.append(row)

    output.tracks = kept_tracks
    output.num_masklets = len(kept_tracks)
    fragment_rows: List[Dict[str, Any]] = []
    fragment_frames: List[int] = []
    if int(args.prune_tiny_duplicate_fragments):
        fragment_rows, fragment_frames = _prune_tiny_duplicate_fragments(
            output,
            _parse_csv(args.prune_duplicate_labels),
            int(args.prune_max_area_px),
            float(args.prune_min_containment),
            float(args.prune_min_box_iou),
        )
        output.tracks = [track for track in output.tracks if track.get("mask_by_frame")]
        output.num_masklets = len(output.tracks)
    debug = {
        "format": "suppress_sparse_thing_by_stuff_overlap_v1",
        "thing_pt": str(args.thing_pt),
        "stuff_pt": str(args.stuff_pt),
        "thing_labels": sorted(thing_labels),
        "suppress_stuff_labels": stuff_labels,
        "thresholds": {
            "containment_threshold": float(args.containment_threshold),
            "person_mean_containment": float(args.person_mean_containment),
            "person_spike_containment": float(args.person_spike_containment),
            "person_spike_min_mean": float(args.person_spike_min_mean),
            "person_spike_min_frames": int(args.person_spike_min_frames),
            "drop_short_person_max_frames": int(args.drop_short_person_max_frames),
            "car_mean_containment": float(args.car_mean_containment),
            "car_spike_containment": float(args.car_spike_containment),
            "car_spike_min_frames": int(args.car_spike_min_frames),
            "car_spike_max_visible_frames": int(args.car_spike_max_visible_frames),
        },
        "input_tracks": int(len(thing.tracks)),
        "output_tracks": int(len(output.tracks)),
        "dropped_tracks": int(sum(1 for row in rows if bool(row["dropped"]))),
        "dropped_by_label": {},
        "dropped_by_reason": {},
        "tiny_duplicate_fragment_prune": {
            "enabled": bool(int(args.prune_tiny_duplicate_fragments)),
            "labels": _parse_csv(args.prune_duplicate_labels),
            "max_area_px": int(args.prune_max_area_px),
            "min_containment": float(args.prune_min_containment),
            "min_box_iou": float(args.prune_min_box_iou),
            "removed_masks": int(len(fragment_rows)),
            "touched_frames": fragment_frames,
        },
    }
    for row in rows:
        if not bool(row["dropped"]):
            continue
        label = str(row["label"])
        reason = str(row["drop_reason"])
        debug["dropped_by_label"][label] = int(debug["dropped_by_label"].get(label, 0)) + 1
        debug["dropped_by_reason"][reason] = int(debug["dropped_by_reason"].get(reason, 0)) + 1
    output.debug["suppress_sparse_thing_by_stuff_overlap"] = debug

    output_pt = output_dir / "sparse_masklets.pt"
    save_sparse_output(output_pt, output)

    csv_path = output_dir / "suppressed_tracks.csv"
    fieldnames = [
        "track_index",
        "label",
        "source_type",
        "birth_frame",
        "end_frame",
        "visible_frames",
        "dropped",
        "drop_reason",
        "dominant_stuff_label",
        "mean_containment",
        "max_containment",
        "frames_over_threshold",
        "over_threshold_ratio",
        "mean_iou",
        "max_iou",
        "per_label_mean_containment",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    fragment_csv_path = output_dir / "pruned_tiny_duplicate_fragments.csv"
    fragment_fieldnames = [
        "frame",
        "track_index",
        "label",
        "area_px",
        "matched_track_index",
        "matched_area_px",
        "containment",
        "box_iou",
        "reason",
    ]
    with fragment_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fragment_fieldnames)
        writer.writeheader()
        for row in fragment_rows:
            writer.writerow({key: row.get(key, "") for key in fragment_fieldnames})

    summary = {
        **debug,
        "coverage_before": coverage_stats(thing),
        "coverage_after": coverage_stats(output),
        "track_stats_before": track_stats(thing),
        "track_stats_after": track_stats(output),
        "output_pt": str(output_pt),
        "output_video": str(output_dir / "overlay_final.mp4"),
        "suppressed_tracks_csv": str(csv_path),
        "pruned_tiny_duplicate_fragments_csv": str(fragment_csv_path),
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    image_paths, temp_dirs = _load_processing_frames(
        str(args.input_video),
        int(args.processing_max_side),
        int(args.frames_limit),
        H,
        W,
        int(output.num_frames),
    )
    try:
        create_tracking_video_v2(
            image_paths,
            output,
            str(output_dir / "overlay_final.mp4"),
            fps=int(args.fps),
            mask_alpha=float(args.mask_alpha),
        )
        if str(args.contact_frames).strip().lower() == "auto":
            contact_frames = _auto_contact_frames(rows, int(output.num_frames))
        else:
            contact_frames = parse_contact_frames(str(args.contact_frames), int(output.num_frames))
        if contact_frames:
            make_contact_sheet(
                image_paths,
                thing,
                output,
                contact_frames,
                output_dir / "contact_before_after.jpg",
                float(args.mask_alpha),
            )
    finally:
        for temp_dir in temp_dirs:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
