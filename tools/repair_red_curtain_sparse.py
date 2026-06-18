#!/usr/bin/env python3
"""Repair a curtain stuff track from color/geometry evidence in an existing sparse output.

This is an audit tool for Taylor-like stage footage. It deliberately does not
pretend to be a general semantic backend: it starts from an existing sparse
result and only edits one stuff label inside a specified frame range.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import (  # noqa: E402
    _unpack_mask_np,
    build_thing_union,
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
    parser = argparse.ArgumentParser(description="Repair red curtain masks in a sparse output.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="curtain")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=-1)
    parser.add_argument("--mode", choices=["union", "replace", "intersect"], default="union")
    parser.add_argument("--hue_ranges", default="0-24,160-179")
    parser.add_argument("--sat_min", type=int, default=55)
    parser.add_argument("--value_min", type=int, default=25)
    parser.add_argument("--value_max", type=int, default=245)
    parser.add_argument("--thing_dilate_px", type=int, default=5)
    parser.add_argument("--morph_open_px", type=int, default=1)
    parser.add_argument("--morph_close_px", type=int, default=5)
    parser.add_argument("--dilate_px", type=int, default=0)
    parser.add_argument("--keep_components", type=int, default=3)
    parser.add_argument("--min_component_area", type=float, default=0.006)
    parser.add_argument("--min_area_ratio", type=float, default=0.015)
    parser.add_argument("--max_area_ratio", type=float, default=0.88)
    parser.add_argument("--subtract_thing", type=int, default=1)
    parser.add_argument("--protect_labels", default="", help="Comma-separated stuff labels to subtract from the repair candidate.")
    parser.add_argument("--protect_dilate_px", type=int, default=2)
    parser.add_argument(
        "--extra_protect_pt",
        default="",
        help="Optional auxiliary sparse file whose selected stuff labels are also subtracted from the repair candidate.",
    )
    parser.add_argument(
        "--extra_protect_labels",
        default="",
        help="Comma-separated stuff labels from --extra_protect_pt to subtract from the repair candidate.",
    )
    parser.add_argument("--extra_protect_dilate_px", type=int, default=1)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--contact_frames", default="0,30,60,90,120,150,179,210,240,270,299")
    return parser.parse_args()


def _parse_hue_ranges(spec: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for raw in str(spec or "").replace(";", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" not in item:
            value = int(item)
            ranges.append((value, value))
            continue
        lo_text, hi_text = item.split("-", 1)
        lo = max(0, min(179, int(lo_text)))
        hi = max(0, min(179, int(hi_text)))
        if lo > hi:
            raise ValueError(f"Invalid hue range: {item!r}")
        ranges.append((lo, hi))
    if not ranges:
        raise ValueError("--hue_ranges produced no ranges")
    return ranges


def _parse_csv(value: str) -> List[str]:
    return [canonicalize_label(item.strip()) for item in str(value or "").split(",") if item.strip()]


def _kernel(px: int) -> np.ndarray:
    size = max(int(px), 0)
    if size <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    size = size * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


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
    if len(image_paths) < num_frames:
        raise RuntimeError(f"Need at least {num_frames} frames, got {len(image_paths)}")
    image_paths = image_paths[:num_frames]
    if tuple(proc_shape) != (expected_h, expected_w):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    return list(image_paths), temp_dirs


def _find_or_create_stuff_track(sparse: Any, label: str) -> Dict[str, Any]:
    wanted = canonicalize_label(label)
    for track in sparse.tracks:
        if str(track.get("source_type")) != "stuff_static":
            continue
        if canonicalize_label(str(track.get("L_sem", ""))) == wanted:
            return track
    from run_video_masklet_front_end import _make_sparse_stuff_track  # local import avoids broad dependency at import time

    track = _make_sparse_stuff_track(wanted, int(sparse.frame_height), int(sparse.frame_width))
    sparse.tracks.append(track)
    sparse.num_masklets = len(sparse.tracks)
    return track


def _existing_mask(track: Dict[str, Any], frame_idx: int, H: int, W: int) -> np.ndarray:
    packed = track.get("mask_by_frame", {}).get(int(frame_idx))
    if packed is None:
        return np.zeros((H, W), dtype=bool)
    return _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)


def _stuff_mask_union(sparse: Any, labels: List[str], frame_idx: int, H: int, W: int) -> np.ndarray:
    wanted = set(labels)
    if not wanted:
        return np.zeros((H, W), dtype=bool)
    out = np.zeros((H, W), dtype=bool)
    for track in sparse.tracks:
        if str(track.get("source_type")) != "stuff_static":
            continue
        label = canonicalize_label(str(track.get("L_sem", "")))
        if label not in wanted:
            continue
        packed = track.get("mask_by_frame", {}).get(int(frame_idx))
        if packed is None:
            continue
        out |= _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
    return out


def _shape_check(base: Any, other: Any, name: str) -> None:
    if int(base.num_frames) != int(other.num_frames):
        raise RuntimeError(f"{name} frame count mismatch: {other.num_frames} vs {base.num_frames}")
    base_shape = (int(base.frame_height), int(base.frame_width))
    other_shape = (int(other.frame_height), int(other.frame_width))
    if other_shape != base_shape:
        raise RuntimeError(f"{name} shape mismatch: {other_shape} vs {base_shape}")


def _red_orange_mask(
    image_path: str,
    H: int,
    W: int,
    hue_ranges: List[Tuple[int, int]],
    sat_min: int,
    value_min: int,
    value_max: int,
) -> np.ndarray:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read frame: {image_path}")
    if bgr.shape[:2] != (H, W):
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hue_mask = np.zeros((H, W), dtype=bool)
    for lo, hi in hue_ranges:
        hue_mask |= (h >= lo) & (h <= hi)
    return hue_mask & (s >= int(sat_min)) & (v >= int(value_min)) & (v <= int(value_max))


def _keep_components(mask: np.ndarray, keep_count: int, min_component_area: float) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    H, W = mask.shape
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    min_area = int(round(float(min_component_area) * float(H * W)))
    components: List[Tuple[int, int]] = []
    for component_id in range(1, n):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, component_id))
    if not components:
        return np.zeros_like(mask, dtype=bool)
    components.sort(reverse=True)
    keep_ids = [component_id for _area, component_id in components[: max(int(keep_count), 1)]]
    return np.isin(labels, keep_ids)


def _clean_candidate(mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    out = mask.astype(bool)
    if int(args.morph_open_px) > 0 and out.any():
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, _kernel(args.morph_open_px)).astype(bool)
    if int(args.morph_close_px) > 0 and out.any():
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(args.morph_close_px)).astype(bool)
    if int(args.dilate_px) > 0 and out.any():
        out = cv2.dilate(out.astype(np.uint8), _kernel(args.dilate_px), iterations=1).astype(bool)
    out = _keep_components(out, int(args.keep_components), float(args.min_component_area))
    return out.astype(bool)


def _combine(existing: np.ndarray, candidate: np.ndarray, mode: str) -> np.ndarray:
    if mode == "union":
        return existing | candidate
    if mode == "replace":
        return candidate
    if mode == "intersect":
        return existing & candidate
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_sparse(Path(args.input_pt))
    extra_protect = None
    if str(args.extra_protect_pt or "").strip():
        extra_protect = load_sparse(Path(args.extra_protect_pt))
        _shape_check(base, extra_protect, "extra_protect")
    output = clone_sparse(base)
    H, W = int(output.frame_height), int(output.frame_width)
    image_paths, temp_dirs = _load_processing_frames(
        args.input_video,
        int(args.processing_max_side),
        int(args.frames_limit),
        H,
        W,
        int(output.num_frames),
    )

    label = canonicalize_label(args.label)
    track = _find_or_create_stuff_track(output, label)
    hue_ranges = _parse_hue_ranges(args.hue_ranges)
    protect_labels = _parse_csv(args.protect_labels)
    extra_protect_labels = _parse_csv(args.extra_protect_labels)
    frame_start = max(0, int(args.frame_start))
    frame_end = int(args.frame_end)
    if frame_end < 0:
        frame_end = int(output.num_frames) - 1
    frame_end = min(frame_end, int(output.num_frames) - 1)

    debug: Dict[str, Any] = {
        "format": "repair_red_curtain_sparse_v1",
        "input_pt": str(args.input_pt),
        "label": label,
        "mode": str(args.mode),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "hue_ranges": hue_ranges,
        "sat_min": int(args.sat_min),
        "value_min": int(args.value_min),
        "value_max": int(args.value_max),
        "subtract_thing": int(args.subtract_thing),
        "thing_dilate_px": int(args.thing_dilate_px),
        "protect_labels": protect_labels,
        "protect_dilate_px": int(args.protect_dilate_px),
        "extra_protect_pt": str(args.extra_protect_pt or ""),
        "extra_protect_labels": extra_protect_labels,
        "extra_protect_dilate_px": int(args.extra_protect_dilate_px),
        "frames_considered": 0,
        "frames_modified": 0,
        "frames_rejected_area": 0,
        "pixels_added": 0,
        "pixels_removed": 0,
        "protected_pixels_base": 0,
        "protected_pixels_extra": 0,
        "candidate_area_mean": 0.0,
        "output_area_mean": 0.0,
        "per_frame": [],
    }
    candidate_areas: List[float] = []
    output_areas: List[float] = []
    thing_kernel = _kernel(int(args.thing_dilate_px))
    protect_kernel = _kernel(int(args.protect_dilate_px))
    extra_protect_kernel = _kernel(int(args.extra_protect_dilate_px))

    for frame_idx in range(frame_start, frame_end + 1):
        debug["frames_considered"] += 1
        existing = _existing_mask(track, frame_idx, H, W)
        candidate = _red_orange_mask(
            image_paths[frame_idx],
            H,
            W,
            hue_ranges,
            int(args.sat_min),
            int(args.value_min),
            int(args.value_max),
        )
        candidate = _clean_candidate(candidate, args)
        if int(args.subtract_thing):
            thing = build_thing_union(output, frame_idx)
            if thing.any():
                thing = cv2.dilate(thing.astype(np.uint8), thing_kernel, iterations=1).astype(bool)
                candidate &= ~thing
        protected = _stuff_mask_union(base, protect_labels, frame_idx, H, W)
        if protected.any():
            if int(args.protect_dilate_px) > 0:
                protected = cv2.dilate(protected.astype(np.uint8), protect_kernel, iterations=1).astype(bool)
            debug["protected_pixels_base"] += int((candidate & protected).sum())
            candidate &= ~protected
        extra_protected = np.zeros((H, W), dtype=bool)
        if extra_protect is not None and extra_protect_labels:
            extra_protected = _stuff_mask_union(extra_protect, extra_protect_labels, frame_idx, H, W)
            if extra_protected.any():
                if int(args.extra_protect_dilate_px) > 0:
                    extra_protected = cv2.dilate(
                        extra_protected.astype(np.uint8),
                        extra_protect_kernel,
                        iterations=1,
                    ).astype(bool)
                debug["protected_pixels_extra"] += int((candidate & extra_protected).sum())
                candidate &= ~extra_protected
        candidate_area = float(candidate.sum()) / float(max(H * W, 1))
        candidate_areas.append(candidate_area)
        if candidate_area < float(args.min_area_ratio) or candidate_area > float(args.max_area_ratio):
            debug["frames_rejected_area"] += 1
            debug["per_frame"].append(
                {
                    "frame_idx": int(frame_idx),
                    "candidate_area": candidate_area,
                    "status": "rejected_area",
                }
            )
            continue
        combined = _combine(existing, candidate, str(args.mode))
        output_area = float(combined.sum()) / float(max(H * W, 1))
        output_areas.append(output_area)
        added = int((combined & ~existing).sum())
        removed = int((existing & ~combined).sum())
        if added or removed:
            debug["frames_modified"] += 1
            debug["pixels_added"] += added
            debug["pixels_removed"] += removed
        refresh_track_frame(track, frame_idx, combined, H, W)
        debug["per_frame"].append(
            {
                "frame_idx": int(frame_idx),
                "candidate_area": candidate_area,
                "output_area": output_area,
                "pixels_added": added,
                "pixels_removed": removed,
                "status": "updated",
            }
        )

    if candidate_areas:
        debug["candidate_area_mean"] = float(np.mean(candidate_areas))
    if output_areas:
        debug["output_area_mean"] = float(np.mean(output_areas))
    output.debug["repair_red_curtain_sparse"] = debug
    output.num_masklets = len(output.tracks)

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    contact_path = output_dir / "contact_before_after.jpg"
    metrics_path = output_dir / "metrics_summary.json"
    save_sparse_output(output_pt, output)
    create_tracking_video_v2(
        image_paths,
        output,
        str(output_video),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style="clean",
    )
    make_contact_sheet(
        image_paths,
        base,
        output,
        parse_contact_frames(args.contact_frames, int(output.num_frames)),
        contact_path,
        float(args.mask_alpha),
    )

    before_stats = coverage_stats(base)
    after_stats = coverage_stats(output)
    summary = {
        "input_pt": str(args.input_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "before": before_stats,
        "after": after_stats,
        "delta": {
            key: float(after_stats[key]) - float(before_stats[key])
            for key in before_stats
            if key in after_stats
        },
        "track_stats_after": track_stats(output),
        "repair_debug": debug,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
