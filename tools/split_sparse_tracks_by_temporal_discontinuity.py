#!/usr/bin/env python3
"""Split sparse tracks at temporal discontinuities.

This is an auditable post-process, not a manual merge/split list. It splits a
track when consecutive visible components are separated by a long gap or by a
large center jump after a gap. The goal is to prevent one output ID from being
reused for multiple physical objects.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import clone_sparse, load_sparse, track_stats  # noqa: E402
from run_video_masklet_front_end import collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402
from loger.pipeline.video_masklet_frontend import canonicalize_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split sparse tracks by temporal discontinuity.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--labels", default="car")
    parser.add_argument("--max_gap_keep", type=int, default=12)
    parser.add_argument("--center_jump_px", type=float, default=80.0)
    parser.add_argument("--center_scale_floor", type=float, default=24.0)
    parser.add_argument("--center_jump_norm", type=float, default=2.5)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--render_style", choices=["debug", "clean"], default="debug")
    return parser.parse_args()


def _parse_labels(raw: str) -> Optional[set[str]]:
    text = str(raw or "").strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    return {canonicalize_label(part.strip()) for part in text.split(",") if canonicalize_label(part.strip())}


def _box_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return None
    return arr[:4]


def _center(box: np.ndarray) -> np.ndarray:
    return np.asarray([(float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5])


def _box_scale(box: np.ndarray, floor: float) -> float:
    return max(
        float(floor),
        1.0,
        float(box[2] - box[0]),
        float(box[3] - box[1]),
    )


def _frame_components(frames: Sequence[int]) -> List[List[int]]:
    components: List[List[int]] = []
    current: List[int] = []
    prev: Optional[int] = None
    for frame_idx in sorted(int(f) for f in frames):
        if prev is None or frame_idx == prev + 1:
            current.append(frame_idx)
        else:
            if current:
                components.append(current)
            current = [frame_idx]
        prev = frame_idx
    if current:
        components.append(current)
    return components


def _copy_track_slice(track: Dict[str, Any], frames: Sequence[int], parent_idx: int, split_idx: int) -> Dict[str, Any]:
    selected = [int(f) for f in sorted(frames)]
    copied = {
        "mask_by_frame": {f: track.get("mask_by_frame", {})[f] for f in selected},
        "box_by_frame": {f: track.get("box_by_frame", {})[f] for f in selected},
        "q_by_frame": {f: track.get("q_by_frame", {})[f] for f in selected},
        "area_by_frame": {f: track.get("area_by_frame", {})[f] for f in selected},
        "L_sem": track.get("L_sem"),
        "G_sem": int(track.get("G_sem", 0)),
        "W_sem": float(track.get("W_sem", 0.0)),
        "source_type": track.get("source_type"),
        "birth_frame": int(selected[0]) if selected else int(track.get("birth_frame", 0)),
        "frame_height": int(track.get("frame_height", 0)),
        "frame_width": int(track.get("frame_width", 0)),
        "_split_parent_track_index": int(parent_idx),
        "_split_child_index": int(split_idx),
    }
    return copied


def _split_track(
    track: Dict[str, Any],
    parent_idx: int,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    frames = sorted(int(f) for f in track.get("mask_by_frame", {}).keys())
    if not frames:
        return [], []
    components = _frame_components(frames)
    if len(components) <= 1:
        return [_copy_track_slice(track, frames, parent_idx, 0)], []

    segments: List[List[int]] = []
    current: List[int] = list(components[0])
    events: List[Dict[str, Any]] = []
    segment_idx = 0
    for comp_idx, comp in enumerate(components[1:], start=1):
        prev_end = int(current[-1])
        next_start = int(comp[0])
        gap = int(next_start - prev_end - 1)
        prev_box = _box_array(track.get("box_by_frame", {}).get(prev_end))
        next_box = _box_array(track.get("box_by_frame", {}).get(next_start))
        center_dist = 0.0
        center_norm = 0.0
        if prev_box is not None and next_box is not None:
            center_dist = float(np.linalg.norm(_center(prev_box) - _center(next_box)))
            scale = max(_box_scale(prev_box, float(args.center_scale_floor)), _box_scale(next_box, float(args.center_scale_floor)))
            center_norm = float(center_dist / max(scale, 1e-6))

        reasons: List[str] = []
        if gap > int(args.max_gap_keep):
            reasons.append("long_gap")
        if center_dist > float(args.center_jump_px):
            reasons.append("center_jump_px")
        if center_norm > float(args.center_jump_norm):
            reasons.append("center_jump_norm")

        if reasons:
            segments.append(current)
            events.append(
                {
                    "parent_track_index": int(parent_idx),
                    "label": str(track.get("L_sem", "")),
                    "split_after_frame": int(prev_end),
                    "next_frame": int(next_start),
                    "gap": int(gap),
                    "center_jump_px": float(center_dist),
                    "center_jump_norm": float(center_norm),
                    "reason": "+".join(reasons),
                    "left_child_index": int(segment_idx),
                    "right_child_index": int(segment_idx + 1),
                }
            )
            segment_idx += 1
            current = list(comp)
        else:
            current.extend(comp)

    if current:
        segments.append(current)

    children = [_copy_track_slice(track, segment, parent_idx, idx) for idx, segment in enumerate(segments)]
    return children, events


def _load_processing_frames(input_video: str, processing_max_side: int, expected_h: int, expected_w: int, num_frames: int) -> Tuple[List[str], List[str]]:
    image_paths, temp_dir = collect_image_paths(input_video, 0, -1, 1)
    temp_dirs = [temp_dir] if temp_dir else []
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(image_paths, int(processing_max_side))
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    if len(image_paths) < int(num_frames):
        raise RuntimeError(f"Need at least {num_frames} frames, got {len(image_paths)}")
    image_paths = image_paths[: int(num_frames)]
    if tuple(proc_shape) != (int(expected_h), int(expected_w)):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    return list(image_paths), temp_dirs


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "parent_track_index",
        "label",
        "split_after_frame",
        "next_frame",
        "gap",
        "center_jump_px",
        "center_jump_norm",
        "reason",
        "left_child_index",
        "right_child_index",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    sparse = load_sparse(Path(args.input_pt))
    labels = _parse_labels(str(args.labels))
    out = clone_sparse(sparse)
    out_tracks: List[Dict[str, Any]] = []
    split_events: List[Dict[str, Any]] = []
    split_parent_count = 0

    for parent_idx, track in enumerate(sparse.tracks):
        label = canonicalize_label(str(track.get("L_sem", "")))
        if labels is not None and label not in labels:
            out_tracks.append(track)
            continue
        children, events = _split_track(track, parent_idx, args)
        if len(children) > 1:
            split_parent_count += 1
        out_tracks.extend(children)
        split_events.extend(events)

    out.tracks = out_tracks
    out.num_masklets = len(out_tracks)
    out.debug = dict(out.debug)
    out.debug["temporal_discontinuity_split"] = {
        "format": "split_sparse_tracks_by_temporal_discontinuity_v1",
        "labels": sorted(labels) if labels is not None else "all",
        "input_tracks": int(len(sparse.tracks)),
        "output_tracks": int(len(out_tracks)),
        "split_parent_tracks": int(split_parent_count),
        "split_events": int(len(split_events)),
        "max_gap_keep": int(args.max_gap_keep),
        "center_jump_px": float(args.center_jump_px),
        "center_scale_floor": float(args.center_scale_floor),
        "center_jump_norm": float(args.center_jump_norm),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_pt = out_dir / "sparse_masklets.pt"
    save_sparse_output(output_pt, out)
    _write_csv(out_dir / "split_tracks.csv", split_events)

    image_paths, temp_dirs = _load_processing_frames(
        str(args.input_video),
        int(args.processing_max_side),
        int(out.frame_height),
        int(out.frame_width),
        int(out.num_frames),
    )
    try:
        output_video = out_dir / "overlay_final.mp4"
        create_tracking_video_v2(
            image_paths,
            out,
            str(output_video),
            fps=int(args.fps),
            mask_alpha=float(args.mask_alpha),
            render_style=str(args.render_style),
        )
    finally:
        for temp_dir in temp_dirs:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    metrics = {
        "input_pt": str(args.input_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "split_tracks_csv": str(out_dir / "split_tracks.csv"),
        "input_tracks": int(len(sparse.tracks)),
        "output_tracks": int(len(out.tracks)),
        "split_parent_tracks": int(split_parent_count),
        "split_events": int(len(split_events)),
        "thresholds": out.debug["temporal_discontinuity_split"],
        "track_stats_before": track_stats(sparse),
        "track_stats_after": track_stats(out),
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
