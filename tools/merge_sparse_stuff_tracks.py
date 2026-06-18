#!/usr/bin/env python3
"""Merge thing tracks from one sparse file with stuff tracks from another."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import (  # noqa: E402
    clone_sparse,
    coverage_stats,
    load_sparse,
    make_contact_sheet,
    parse_contact_frames,
    track_stats,
)
from run_video_masklet_front_end import collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge base things with replacement stuff tracks.")
    parser.add_argument("--base_pt", required=True)
    parser.add_argument("--stuff_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument(
        "--video_start_frame",
        type=int,
        default=0,
        help="Frame offset in the source video used only for rendering windowed sparse outputs.",
    )
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--contact_frames", default="0,30,60,90,120,150,180,210,240,270,299")
    return parser.parse_args()


def _load_processing_frames(args: argparse.Namespace, expected_h: int, expected_w: int, num_frames: int) -> tuple[List[str], List[str]]:
    start_frame = max(int(getattr(args, "video_start_frame", 0)), 0)
    image_paths, temp_dir = collect_image_paths(args.input_video, start_frame, -1, 1)
    temp_dirs = [temp_dir] if temp_dir else []
    if args.frames_limit and int(args.frames_limit) > 0:
        image_paths = image_paths[: int(args.frames_limit)]
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(
        image_paths,
        int(args.processing_max_side),
    )
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    if len(image_paths) < num_frames:
        raise RuntimeError(f"Need at least {num_frames} frames, got {len(image_paths)}")
    image_paths = image_paths[:num_frames]
    if tuple(proc_shape) != (expected_h, expected_w):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    return list(image_paths), temp_dirs


def _merge(base: Any, stuff: Any, base_pt: str, stuff_pt: str) -> Any:
    if int(base.num_frames) != int(stuff.num_frames):
        raise ValueError(f"Frame count mismatch: base={base.num_frames}, stuff={stuff.num_frames}")
    if (int(base.frame_height), int(base.frame_width)) != (int(stuff.frame_height), int(stuff.frame_width)):
        raise ValueError(
            "Frame shape mismatch: "
            f"base={(base.frame_height, base.frame_width)}, stuff={(stuff.frame_height, stuff.frame_width)}"
        )
    merged = clone_sparse(base)
    thing_tracks: List[Dict[str, Any]] = [
        track for track in merged.tracks if str(track.get("source_type", "")) != "stuff_static"
    ]
    stuff_tracks: List[Dict[str, Any]] = [
        track for track in clone_sparse(stuff).tracks if str(track.get("source_type", "")) == "stuff_static"
    ]
    merged.tracks = thing_tracks + stuff_tracks
    merged.num_masklets = len(merged.tracks)
    merged.debug["merge_sparse_stuff_tracks"] = {
        "base_pt": str(base_pt),
        "stuff_pt": str(stuff_pt),
        "base_tracks": int(len(base.tracks)),
        "base_thing_tracks_kept": int(len(thing_tracks)),
        "replacement_stuff_tracks": int(len(stuff_tracks)),
        "output_tracks": int(len(merged.tracks)),
    }
    return merged


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_sparse(Path(args.base_pt))
    stuff = load_sparse(Path(args.stuff_pt))
    merged = _merge(base, stuff, args.base_pt, args.stuff_pt)
    image_paths, temp_dirs = _load_processing_frames(args, merged.frame_height, merged.frame_width, merged.num_frames)

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    metrics_path = output_dir / "metrics_summary.json"
    contact_path = output_dir / "contact_before_after.jpg"

    save_sparse_output(output_pt, merged)
    create_tracking_video_v2(
        image_paths,
        merged,
        str(output_video),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style="clean",
    )
    make_contact_sheet(
        image_paths,
        base,
        merged,
        parse_contact_frames(args.contact_frames, merged.num_frames),
        contact_path,
        float(args.mask_alpha),
    )

    summary = {
        "base_pt": str(args.base_pt),
        "stuff_pt": str(args.stuff_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "before": coverage_stats(base),
        "after": coverage_stats(merged),
        "delta": {
            key: float(coverage_stats(merged)[key]) - float(coverage_stats(base)[key])
            for key in coverage_stats(base).keys()
            if key in coverage_stats(merged)
        },
        "track_stats_after": track_stats(merged),
        "merge_debug": merged.debug.get("merge_sparse_stuff_tracks", {}),
        "video_start_frame": int(args.video_start_frame),
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
