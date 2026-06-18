#!/usr/bin/env python3
"""Export a frame slice from a compact sparse_masklets_v1 file.

This is an audit helper for older or long sparse outputs. It does not run any
model; it clips existing sparse masklets to a frame window, writes a compact
slice, an overlay video, metrics JSON, and a single-column contact sheet.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import (  # noqa: E402
    clone_sparse,
    coverage_stats,
    load_sparse,
    parse_contact_frames,
    track_stats,
)
from run_video_masklet_front_end import SparseMaskletOutput, collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import create_tracking_video_v2, render_clean_frame, save_sparse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a sparse masklet slice for review.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument(
        "--video_start_frame",
        type=int,
        default=None,
        help="Frame offset in the source video. Defaults to --start_frame. Use this when the sparse file is already a local slice.",
    )
    parser.add_argument("--frames_limit", type=int, default=300)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--keep_source_types", default="all")
    parser.add_argument("--contact_frames", default="0,30,60,90,120,150,180,210,240,270,299")
    return parser.parse_args()


def _source_set(spec: str) -> set[str] | None:
    text = str(spec or "").strip()
    if not text or text.lower() == "all":
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def _clip_sparse(sparse: SparseMaskletOutput, start_frame: int, frames_limit: int, keep_sources: set[str] | None) -> SparseMaskletOutput:
    start = max(int(start_frame), 0)
    limit = int(frames_limit)
    end = int(sparse.num_frames) if limit <= 0 else min(int(sparse.num_frames), start + limit)
    if start >= end:
        raise ValueError(f"Invalid frame slice [{start}, {end}) for {sparse.num_frames} frames")

    clipped = clone_sparse(sparse)
    kept_tracks: List[Dict[str, Any]] = []
    for raw_track in clipped.tracks:
        if keep_sources is not None and str(raw_track.get("source_type", "")) not in keep_sources:
            continue
        track = dict(raw_track)
        for key in ("mask_by_frame", "box_by_frame", "q_by_frame", "area_by_frame"):
            values = raw_track.get(key, {})
            track[key] = {
                int(frame_idx) - start: value
                for frame_idx, value in values.items()
                if start <= int(frame_idx) < end
            }
        if not track.get("mask_by_frame"):
            continue
        track["birth_frame"] = max(0, int(track.get("birth_frame", start)) - start)
        kept_tracks.append(track)

    clipped.tracks = kept_tracks
    clipped.num_masklets = len(kept_tracks)
    clipped.num_frames = end - start
    clipped.debug["export_sparse_masklet_slice"] = {
        "input_num_frames": int(sparse.num_frames),
        "start_frame": start,
        "end_frame": end,
        "frames_limit": int(frames_limit),
        "keep_source_types": "all" if keep_sources is None else sorted(keep_sources),
        "output_tracks": int(len(kept_tracks)),
    }
    return clipped


def _load_processing_frames(args: argparse.Namespace, expected_h: int, expected_w: int, start_frame: int, num_frames: int) -> tuple[List[str], List[str]]:
    image_paths, temp_dir = collect_image_paths(args.input_video, start_frame, start_frame + max(int(num_frames), 0), 1)
    temp_dirs = [temp_dir] if temp_dir else []
    if len(image_paths) < int(num_frames):
        raise RuntimeError(f"Need at least {num_frames} frames from source offset {start_frame}, got {len(image_paths)}")
    image_paths = image_paths[: int(num_frames)]
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(
        image_paths,
        int(args.processing_max_side),
    )
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    if tuple(proc_shape) != (expected_h, expected_w):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    return list(image_paths), temp_dirs


def _make_single_contact(
    image_paths: Sequence[str],
    sparse: SparseMaskletOutput,
    frame_indices: Sequence[int],
    output_path: Path,
    mask_alpha: float,
) -> None:
    cells: List[np.ndarray] = []
    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= len(image_paths):
            continue
        bgr = cv2.imread(image_paths[frame_idx], cv2.IMREAD_COLOR)
        if bgr is None:
            rgb = np.zeros((sparse.frame_height, sparse.frame_width, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (sparse.frame_height, sparse.frame_width):
                rgb = cv2.resize(rgb, (sparse.frame_width, sparse.frame_height))
        rendered = render_clean_frame(rgb, sparse, frame_idx, mask_alpha=mask_alpha)
        cv2.putText(rendered, f"f={frame_idx}", (9, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(rendered, f"f={frame_idx}", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(rendered)
    if not cells:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contact = np.concatenate(cells, axis=0)
    contact_bgr = cv2.cvtColor(contact, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), contact_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse = load_sparse(Path(args.input_pt))
    keep_sources = _source_set(args.keep_source_types)
    clipped = _clip_sparse(sparse, int(args.start_frame), int(args.frames_limit), keep_sources)
    video_start_frame = int(args.start_frame) if args.video_start_frame is None else int(args.video_start_frame)
    image_paths, temp_dirs = _load_processing_frames(
        args,
        int(clipped.frame_height),
        int(clipped.frame_width),
        max(video_start_frame, 0),
        int(clipped.num_frames),
    )

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    metrics_path = output_dir / "metrics_summary.json"
    contact_path = output_dir / "contact_sheet.jpg"

    save_sparse_output(output_pt, clipped)
    create_tracking_video_v2(
        image_paths,
        clipped,
        str(output_video),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style="clean",
    )
    _make_single_contact(
        image_paths,
        clipped,
        parse_contact_frames(args.contact_frames, int(clipped.num_frames)),
        contact_path,
        float(args.mask_alpha),
    )

    summary = {
        "input_pt": str(args.input_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "coverage": coverage_stats(clipped),
        "track_stats": track_stats(clipped),
        "export_debug": clipped.debug.get("export_sparse_masklet_slice", {}),
        "video_start_frame": int(video_start_frame),
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
