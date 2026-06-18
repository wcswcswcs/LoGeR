#!/usr/bin/env python3
"""Fuse VidEoMT semantic sparse tracks with v3 SAM3.1/MOT thing tracks."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_sparse_masklet_slice import _make_single_contact  # noqa: E402
from refine_sparse_stuff_masks import (  # noqa: E402
    _unpack_mask_np,
    clone_sparse,
    coverage_stats,
    load_sparse,
    parse_contact_frames,
    refresh_track_frame,
    track_stats,
)
from run_video_masklet_front_end import SparseMaskletOutput, collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse VidEoMT VSPW sparse output with v3 SAM3.1 thing tracks.")
    parser.add_argument("--videomt_pt", required=True)
    parser.add_argument("--sam31_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--fusion_role",
        default="diagnostic_thingstuff_plus_sam31",
        choices=["diagnostic_thingstuff_plus_sam31", "final_stuff_plus_sam31"],
        help="Recorded intent. Diagnostic may keep VidEoMT semantic thing residuals; final should use VidEoMT only as stuff.",
    )
    parser.add_argument(
        "--drop_videomt_labels",
        default="",
        help="Comma-separated VidEoMT labels to drop before fusion, e.g. car,person. Empty keeps input labels.",
    )
    parser.add_argument(
        "--min_videomt_frames_after_subtract",
        type=int,
        default=1,
        help="Drop VidEoMT semantic tracks with fewer remaining mask frames after subtraction.",
    )
    parser.add_argument("--subtract_sam31_from_videomt", type=int, default=1)
    parser.add_argument("--sam31_dilate_px", type=int, default=0)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--render_video", type=int, default=1)
    parser.add_argument("--render_contact_sheet", type=int, default=1)
    parser.add_argument("--contact_frames", default="0,64,128,256,384,512,640,768,896,1024,1100")
    return parser.parse_args()


def _split_labels(raw: str) -> Set[str]:
    return {part.strip().lower().replace(" ", "_") for part in str(raw or "").split(",") if part.strip()}


def _label_of(track: Dict[str, Any]) -> str:
    return str(track.get("label", track.get("L_sem", "")))


def _norm_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def _track_frame_count(track: Dict[str, Any]) -> int:
    return int(len(track.get("mask_by_frame", {}) or {}))


def _label_counts(tracks: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(_label_of(track) for track in tracks).items()))


def _source_counts(tracks: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(track.get("source_type", "")) for track in tracks).items()))


def _shape_check(left: SparseMaskletOutput, right: SparseMaskletOutput, name: str) -> None:
    if int(left.num_frames) != int(right.num_frames):
        raise RuntimeError(f"{name} frame count mismatch: {right.num_frames} vs {left.num_frames}")
    if (int(left.frame_height), int(left.frame_width)) != (int(right.frame_height), int(right.frame_width)):
        raise RuntimeError(
            f"{name} shape mismatch: {(right.frame_height, right.frame_width)} "
            f"vs {(left.frame_height, left.frame_width)}"
        )


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
    if len(image_paths) < int(num_frames):
        raise RuntimeError(f"Need at least {num_frames} frames, got {len(image_paths)}")
    image_paths = image_paths[: int(num_frames)]
    if tuple(proc_shape) != (int(expected_h), int(expected_w)):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    return list(image_paths), temp_dirs


def _sam31_thing_tracks(sam31: SparseMaskletOutput) -> List[Dict[str, Any]]:
    return [
        copy.deepcopy(track)
        for track in sam31.tracks
        if str(track.get("source_type", "")) != "stuff_static"
    ]


def _build_union_by_frame(
    tracks: List[Dict[str, Any]],
    H: int,
    W: int,
    num_frames: int,
    dilate_px: int,
) -> List[np.ndarray]:
    unions = [np.zeros((H, W), dtype=bool) for _ in range(int(num_frames))]
    for track in tracks:
        for frame_idx, packed in track.get("mask_by_frame", {}).items():
            idx = int(frame_idx)
            if idx < 0 or idx >= int(num_frames):
                continue
            unions[idx] |= _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
    if int(dilate_px) > 0:
        size = int(dilate_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        unions = [
            cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) if mask.any() else mask
            for mask in unions
        ]
    return unions


def _subtract_union_from_videomt(
    sparse: SparseMaskletOutput,
    unions: List[np.ndarray],
) -> Dict[str, Any]:
    H = int(sparse.frame_height)
    W = int(sparse.frame_width)
    debug: Dict[str, Any] = {
        "tracks_touched": 0,
        "frames_touched": 0,
        "removed_pixels": 0,
        "emptied_track_frames": 0,
    }
    for track in sparse.tracks:
        touched_track = False
        for frame_idx in sorted(int(idx) for idx in list(track.get("mask_by_frame", {}).keys())):
            if frame_idx < 0 or frame_idx >= len(unions):
                continue
            union = unions[frame_idx]
            if not union.any():
                continue
            packed = track.get("mask_by_frame", {}).get(frame_idx)
            if packed is None:
                continue
            before = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
            overlap = before & union
            if not overlap.any():
                continue
            after = before & ~union
            debug["removed_pixels"] = int(debug["removed_pixels"]) + int(overlap.sum())
            debug["frames_touched"] = int(debug["frames_touched"]) + 1
            if not after.any():
                debug["emptied_track_frames"] = int(debug["emptied_track_frames"]) + 1
            refresh_track_frame(track, frame_idx, after, H, W)
            touched_track = True
        if touched_track:
            debug["tracks_touched"] = int(debug["tracks_touched"]) + 1
    return debug


def _filter_videomt_tracks(
    tracks: Sequence[Dict[str, Any]],
    drop_labels: Set[str],
    min_frames: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    dropped_by_label: Counter[str] = Counter()
    dropped_empty_by_label: Counter[str] = Counter()
    dropped_examples: List[Dict[str, Any]] = []
    min_frames = max(int(min_frames), 1)
    for idx, track in enumerate(tracks):
        label = _label_of(track)
        norm = _norm_label(label)
        frame_count = _track_frame_count(track)
        reason = ""
        if norm in drop_labels:
            reason = "drop_videomt_label"
            dropped_by_label[label] += 1
        elif frame_count < min_frames:
            reason = "too_few_remaining_frames"
            dropped_empty_by_label[label] += 1
        if reason:
            if len(dropped_examples) < 50:
                dropped_examples.append(
                    {
                        "old_index": int(idx),
                        "label": str(label),
                        "source_type": str(track.get("source_type", "")),
                        "remaining_frames": int(frame_count),
                        "reason": reason,
                    }
                )
            continue
        kept.append(track)
    return kept, {
        "drop_labels": sorted(drop_labels),
        "min_frames": int(min_frames),
        "input_tracks": int(len(tracks)),
        "kept_tracks": int(len(kept)),
        "dropped_by_label": dict(sorted(dropped_by_label.items())),
        "dropped_too_short_by_label": dict(sorted(dropped_empty_by_label.items())),
        "dropped_examples": dropped_examples,
    }


def _videomt_sam31_overlap_pixels(
    tracks: Sequence[Dict[str, Any]],
    unions: List[np.ndarray],
    H: int,
    W: int,
) -> int:
    overlap_pixels = 0
    for track in tracks:
        for frame_idx, packed in track.get("mask_by_frame", {}).items():
            idx = int(frame_idx)
            if idx < 0 or idx >= len(unions):
                continue
            union = unions[idx]
            if not union.any():
                continue
            mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
            if not mask.any():
                continue
            overlap_pixels += int(np.logical_and(mask, union).sum())
    return int(overlap_pixels)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videomt = load_sparse(Path(args.videomt_pt))
    sam31 = load_sparse(Path(args.sam31_pt))
    _shape_check(videomt, sam31, "sam31")

    output = clone_sparse(videomt)
    input_videomt_tracks = copy.deepcopy(output.tracks)
    sam31_tracks = _sam31_thing_tracks(sam31)
    unions = _build_union_by_frame(
        sam31_tracks,
        int(output.frame_height),
        int(output.frame_width),
        int(output.num_frames),
        int(args.sam31_dilate_px),
    )
    subtract_debug: Dict[str, Any] = {}
    if int(args.subtract_sam31_from_videomt):
        subtract_debug = _subtract_union_from_videomt(output, unions)

    drop_labels = _split_labels(args.drop_videomt_labels)
    filtered_videomt_tracks, filter_debug = _filter_videomt_tracks(
        output.tracks,
        drop_labels,
        int(args.min_videomt_frames_after_subtract),
    )
    kept_videomt_label_counts = _label_counts(filtered_videomt_tracks)
    kept_videomt_source_counts = _source_counts(filtered_videomt_tracks)
    residual_overlap_pixels = _videomt_sam31_overlap_pixels(
        filtered_videomt_tracks,
        unions,
        int(output.frame_height),
        int(output.frame_width),
    )

    output.tracks = list(filtered_videomt_tracks)
    output.tracks.extend(sam31_tracks)
    output.num_masklets = len(output.tracks)
    output.debug = dict(output.debug)
    output.debug["fuse_videomt_sam31_sparse"] = {
        "format": "fuse_videomt_sam31_sparse_v2",
        "fusion_role": str(args.fusion_role),
        "videomt_pt": str(args.videomt_pt),
        "sam31_pt": str(args.sam31_pt),
        "input_videomt_label_counts": _label_counts(input_videomt_tracks),
        "input_videomt_source_counts": _source_counts(input_videomt_tracks),
        "kept_videomt_label_counts": kept_videomt_label_counts,
        "kept_videomt_source_counts": kept_videomt_source_counts,
        "sam31_thing_label_counts": _label_counts(sam31_tracks),
        "sam31_thing_source_counts": _source_counts(sam31_tracks),
        "output_label_counts": _label_counts(output.tracks),
        "output_source_counts": _source_counts(output.tracks),
        "sam31_thing_tracks_appended": int(len(sam31_tracks)),
        "subtract_sam31_from_videomt": int(args.subtract_sam31_from_videomt),
        "sam31_dilate_px": int(args.sam31_dilate_px),
        "drop_videomt_labels": sorted(drop_labels),
        "subtract_debug": subtract_debug,
        "filter_debug": filter_debug,
        "residual_videomt_sam31_overlap_pixels_after_subtract": int(residual_overlap_pixels),
    }

    image_paths: List[str] = []
    temp_dirs: List[str] = []
    if int(args.render_video) or int(args.render_contact_sheet):
        image_paths, temp_dirs = _load_processing_frames(
            args,
            int(output.frame_height),
            int(output.frame_width),
            int(output.num_frames),
        )
    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    contact_path = output_dir / "contact_sheet.jpg"
    metrics_path = output_dir / "metrics_summary.json"

    save_sparse_output(output_pt, output)
    if int(args.render_video):
        create_tracking_video_v2(
            image_paths,
            output,
            str(output_video),
            fps=int(args.fps),
            mask_alpha=float(args.mask_alpha),
            render_style="clean",
        )
    if int(args.render_contact_sheet):
        _make_single_contact(
            image_paths,
            output,
            parse_contact_frames(str(args.contact_frames), int(output.num_frames)),
            contact_path,
            float(args.mask_alpha),
        )

    summary = {
        "videomt_pt": str(args.videomt_pt),
        "sam31_pt": str(args.sam31_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video) if int(args.render_video) else "",
        "contact_sheet": str(contact_path) if int(args.render_contact_sheet) else "",
        "render_video": bool(int(args.render_video)),
        "render_contact_sheet": bool(int(args.render_contact_sheet)),
        "num_frames": int(output.num_frames),
        "num_tracks": int(output.num_masklets),
        "num_videomt_tracks_input": int(len(videomt.tracks)),
        "num_sam31_thing_tracks_appended": int(len(sam31_tracks)),
        "coverage": coverage_stats(output),
        "track_stats": track_stats(output),
        "fusion_debug": output.debug["fuse_videomt_sam31_sparse"],
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
