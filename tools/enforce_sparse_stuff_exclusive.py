#!/usr/bin/env python3
"""Make stuff labels mutually exclusive in a sparse masklet output."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

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
    parser = argparse.ArgumentParser(description="Enforce mutually exclusive stuff masks.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--priority", default="curtain,ceiling,floor,wall")
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--min_visible_frames", type=int, default=1)
    parser.add_argument("--contact_frames", default="0,30,60,90,120,150,180,210,240,270,299")
    return parser.parse_args()


def _parse_csv(value: str) -> List[str]:
    return [canonicalize_label(item.strip()) for item in str(value or "").split(",") if item.strip()]


def _load_processing_frames(args: argparse.Namespace, expected_h: int, expected_w: int, num_frames: int) -> tuple[List[str], List[str]]:
    image_paths, temp_dir = collect_image_paths(args.input_video, 0, -1, 1)
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


def _stuff_tracks_by_label(sparse: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for track in sparse.tracks:
        if str(track.get("source_type")) != "stuff_static":
            continue
        label = canonicalize_label(str(track.get("L_sem", "")))
        out.setdefault(label, track)
    return out


def _overlap_summary(sparse: Any) -> Dict[str, Dict[str, float]]:
    tracks = _stuff_tracks_by_label(sparse)
    labels = sorted(tracks)
    summary: Dict[str, Dict[str, float]] = {}
    H, W = int(sparse.frame_height), int(sparse.frame_width)
    for idx, a in enumerate(labels):
        for b in labels[idx + 1 :]:
            inter = 0
            union = 0
            area_a = 0
            area_b = 0
            overlap_frames = 0
            for frame_idx in range(int(sparse.num_frames)):
                pa = tracks[a].get("mask_by_frame", {}).get(frame_idx)
                pb = tracks[b].get("mask_by_frame", {}).get(frame_idx)
                if pa is None or pb is None:
                    continue
                ma = _unpack_mask_np(np.asarray(pa, dtype=np.uint8), H, W)
                mb = _unpack_mask_np(np.asarray(pb, dtype=np.uint8), H, W)
                local_inter = int((ma & mb).sum())
                if local_inter:
                    overlap_frames += 1
                inter += local_inter
                union += int((ma | mb).sum())
                area_a += int(ma.sum())
                area_b += int(mb.sum())
            key = f"{a}|{b}"
            summary[key] = {
                "overlap_frames": int(overlap_frames),
                "inter_pixels": int(inter),
                "iou": float(inter) / float(max(union, 1)),
                "cover_a": float(inter) / float(max(area_a, 1)),
                "cover_b": float(inter) / float(max(area_b, 1)),
            }
    return summary


def _enforce(sparse: Any, priority: List[str], min_visible_frames: int) -> tuple[Any, Dict[str, Any]]:
    out = clone_sparse(sparse)
    tracks = _stuff_tracks_by_label(out)
    ordered = [label for label in priority if label in tracks]
    ordered.extend(label for label in sorted(tracks) if label not in set(ordered))
    H, W = int(out.frame_height), int(out.frame_width)
    debug: Dict[str, Any] = {
        "priority": ordered,
        "removed_pixels": {label: 0 for label in ordered},
        "removed_frames": {label: 0 for label in ordered},
        "dropped_short": {},
    }

    for frame_idx in range(int(out.num_frames)):
        occupied = np.zeros((H, W), dtype=bool)
        cleaned: Dict[str, np.ndarray] = {}
        for label in ordered:
            track = tracks[label]
            packed = track.get("mask_by_frame", {}).get(frame_idx)
            if packed is None:
                cleaned[label] = np.zeros((H, W), dtype=bool)
                continue
            mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
            before = int(mask.sum())
            mask = mask & ~occupied
            removed = before - int(mask.sum())
            if removed > 0:
                debug["removed_pixels"][label] += int(removed)
                debug["removed_frames"][label] += 1
            cleaned[label] = mask
            occupied |= mask
        for label, mask in cleaned.items():
            refresh_track_frame(tracks[label], frame_idx, mask, H, W)

    keep_tracks: List[Dict[str, Any]] = []
    for track in out.tracks:
        if str(track.get("source_type")) != "stuff_static":
            keep_tracks.append(track)
            continue
        label = canonicalize_label(str(track.get("L_sem", "")))
        visible = len(track.get("mask_by_frame", {}))
        if visible < int(min_visible_frames):
            debug["dropped_short"][label] = int(visible)
            continue
        keep_tracks.append(track)
    out.tracks = keep_tracks
    out.num_masklets = len(keep_tracks)
    out.debug["enforce_sparse_stuff_exclusive"] = debug
    return out, debug


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse = load_sparse(Path(args.input_pt))
    before_overlap = _overlap_summary(sparse)
    exclusive, debug = _enforce(sparse, _parse_csv(args.priority), int(args.min_visible_frames))
    after_overlap = _overlap_summary(exclusive)
    image_paths, temp_dirs = _load_processing_frames(args, exclusive.frame_height, exclusive.frame_width, exclusive.num_frames)

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    contact_path = output_dir / "contact_before_after.jpg"
    metrics_path = output_dir / "metrics_summary.json"

    save_sparse_output(output_pt, exclusive)
    create_tracking_video_v2(
        image_paths,
        exclusive,
        str(output_video),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style="clean",
    )
    make_contact_sheet(
        image_paths,
        sparse,
        exclusive,
        parse_contact_frames(args.contact_frames, exclusive.num_frames),
        contact_path,
        float(args.mask_alpha),
    )

    summary = {
        "input_pt": str(args.input_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "before": coverage_stats(sparse),
        "after": coverage_stats(exclusive),
        "delta": {
            key: float(coverage_stats(exclusive)[key]) - float(coverage_stats(sparse)[key])
            for key in coverage_stats(sparse)
            if key in coverage_stats(exclusive)
        },
        "track_stats_after": track_stats(exclusive),
        "exclusive_debug": debug,
        "overlap_before": before_overlap,
        "overlap_after": after_overlap,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
