#!/usr/bin/env python3
"""Create a cleaner review variant from an existing sparse_masklets.pt.

This tool does not run detector, semantic, or video segmentation models.
It only applies auditable output-policy filters to an existing v2 compact
sparse_masklets_v1 file, then writes a new sparse file, overlay video, metrics,
and a before/after contact sheet.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter sparse masklets for a clean review video.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--keep_stuff_labels", default="all")
    parser.add_argument("--drop_stuff_labels", default="")
    parser.add_argument(
        "--stuff_area_max",
        default="",
        help="Per-label max area ratio, e.g. 'wall:0.45;floor:0.28;curtain:0.45'.",
    )
    parser.add_argument(
        "--stuff_vertical_keep",
        default="",
        help="Per-label vertical keep range as y0-y1 fractions, e.g. 'floor:0.52-1.0;ceiling:0.0-0.28'.",
    )
    parser.add_argument(
        "--stuff_keep_components",
        default="",
        help="Per-label number of largest connected components to keep, e.g. 'wall:2;floor:1'.",
    )
    parser.add_argument(
        "--stuff_min_component_area",
        default="",
        help="Per-label min connected-component area ratio, e.g. 'wall:0.004;floor:0.003'.",
    )
    parser.add_argument("--subtract_thing_dilate_px", type=int, default=0)
    parser.add_argument("--morph_open_px", type=int, default=0)
    parser.add_argument("--morph_close_px", type=int, default=0)
    parser.add_argument("--min_visible_frames_after", type=int, default=1)
    parser.add_argument("--contact_frames", default="0,30,60,90,120,150,180,210,240,270,299")
    return parser.parse_args()


def _label_set(spec: str) -> Optional[set[str]]:
    text = str(spec or "").strip()
    if not text or text.lower() == "all":
        return None
    return {item.strip().lower() for item in text.split(",") if item.strip()}


def _parse_area_caps(spec: str) -> Dict[str, float]:
    caps: Dict[str, float] = {}
    for raw_item in str(spec or "").replace(",", ";").split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --stuff_area_max item: {item!r}")
        label, value = item.split(":", 1)
        label = label.strip().lower()
        if not label:
            raise ValueError(f"Invalid empty label in --stuff_area_max item: {item!r}")
        cap = float(value)
        if cap < 0.0 or cap > 1.0:
            raise ValueError(f"Area cap for {label!r} must be in [0,1], got {cap}")
        caps[label] = cap
    return caps


def _parse_vertical_keep(spec: str) -> Dict[str, tuple[float, float]]:
    ranges: Dict[str, tuple[float, float]] = {}
    for raw_item in str(spec or "").replace(",", ";").split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item or "-" not in item:
            raise ValueError(f"Invalid --stuff_vertical_keep item: {item!r}")
        label, value = item.split(":", 1)
        y0_text, y1_text = value.split("-", 1)
        label = label.strip().lower()
        y0 = float(y0_text)
        y1 = float(y1_text)
        if not label or y0 < 0.0 or y1 > 1.0 or y0 >= y1:
            raise ValueError(f"Invalid vertical keep range for {label!r}: {value!r}")
        ranges[label] = (y0, y1)
    return ranges


def _parse_int_map(spec: str) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for raw_item in str(spec or "").replace(",", ";").split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid integer map item: {item!r}")
        label, value = item.split(":", 1)
        label = label.strip().lower()
        parsed = int(value)
        if not label or parsed < 1:
            raise ValueError(f"Invalid integer map value for {label!r}: {value!r}")
        values[label] = parsed
    return values


def _parse_float_map(spec: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for raw_item in str(spec or "").replace(",", ";").split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid float map item: {item!r}")
        label, value = item.split(":", 1)
        label = label.strip().lower()
        parsed = float(value)
        if not label or parsed < 0.0 or parsed > 1.0:
            raise ValueError(f"Invalid float map value for {label!r}: {value!r}")
        values[label] = parsed
    return values


def _kernel(px: int) -> np.ndarray:
    size = max(int(px), 0)
    if size <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    size = size * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _keep_components(mask: np.ndarray, keep_count: int, min_area_ratio: float) -> np.ndarray:
    if not mask.any():
        return mask
    H, W = mask.shape
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components: List[tuple[int, int]] = []
    min_area = int(round(float(min_area_ratio) * float(H * W)))
    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, component_id))
    if not components:
        return np.zeros_like(mask, dtype=bool)
    components.sort(reverse=True)
    keep_ids = {component_id for _area, component_id in components[: int(keep_count)]}
    return np.isin(labels, list(keep_ids))


def _filter_sparse(
    sparse: Any,
    keep_stuff: Optional[set[str]],
    drop_stuff: set[str],
    area_caps: Dict[str, float],
    vertical_keep: Dict[str, tuple[float, float]],
    keep_components: Dict[str, int],
    min_component_area: Dict[str, float],
    subtract_thing_dilate_px: int,
    morph_open_px: int,
    morph_close_px: int,
    min_visible_frames_after: int,
) -> tuple[Any, Dict[str, Any]]:
    filtered = clone_sparse(sparse)
    kept_tracks: List[Dict[str, Any]] = []
    thing_union_cache: Dict[int, np.ndarray] = {}
    debug: Dict[str, Any] = {
        "input_tracks": int(len(filtered.tracks)),
        "output_tracks": 0,
        "keep_stuff_labels": "all" if keep_stuff is None else sorted(keep_stuff),
        "drop_stuff_labels": sorted(drop_stuff),
        "stuff_area_max": dict(sorted(area_caps.items())),
        "stuff_vertical_keep": {label: list(value) for label, value in sorted(vertical_keep.items())},
        "stuff_keep_components": dict(sorted(keep_components.items())),
        "stuff_min_component_area": dict(sorted(min_component_area.items())),
        "subtract_thing_dilate_px": int(subtract_thing_dilate_px),
        "morph_open_px": int(morph_open_px),
        "morph_close_px": int(morph_close_px),
        "min_visible_frames_after": int(min_visible_frames_after),
        "dropped_tracks": {},
        "dropped_masks": {},
        "modified_masks": {},
        "kept_stuff_tracks": {},
        "kept_stuff_masks": {},
    }
    min_visible = max(int(min_visible_frames_after), 1)
    H, W = int(filtered.frame_height), int(filtered.frame_width)

    for track in filtered.tracks:
        source = str(track.get("source_type", ""))
        label = str(track.get("L_sem", "")).lower()
        if source != "stuff_static":
            kept_tracks.append(track)
            continue

        drop_reason = ""
        if keep_stuff is not None and label not in keep_stuff:
            drop_reason = "label_not_in_keep_set"
        if label in drop_stuff:
            drop_reason = "label_in_drop_set"
        if drop_reason:
            debug["dropped_tracks"][label] = int(debug["dropped_tracks"].get(label, 0)) + 1
            continue

        for frame_idx in sorted(int(frame_idx) for frame_idx in list(track.get("mask_by_frame", {}).keys())):
            packed = track.get("mask_by_frame", {}).get(frame_idx)
            if packed is None:
                continue
            mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W)
            before_area = int(mask.sum())

            y_range = vertical_keep.get(label)
            if y_range is not None:
                y0, y1 = y_range
                y0_px = int(round(y0 * H))
                y1_px = int(round(y1 * H))
                vertical = np.zeros_like(mask, dtype=bool)
                vertical[y0_px:y1_px, :] = True
                mask &= vertical

            if int(subtract_thing_dilate_px) > 0:
                if frame_idx not in thing_union_cache:
                    thing = build_thing_union(sparse, frame_idx).astype(np.uint8)
                    thing = cv2.dilate(thing, _kernel(int(subtract_thing_dilate_px)), iterations=1)
                    thing_union_cache[frame_idx] = thing.astype(bool)
                mask &= ~thing_union_cache[frame_idx]

            if int(morph_open_px) > 0 and mask.any():
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _kernel(int(morph_open_px))).astype(bool)
            if int(morph_close_px) > 0 and mask.any():
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(int(morph_close_px))).astype(bool)

            component_count = keep_components.get(label)
            component_min_area = float(min_component_area.get(label, 0.0))
            if component_count is not None or component_min_area > 0.0:
                mask = _keep_components(mask, int(component_count or 999999), component_min_area)

            after_area = int(mask.sum())
            if after_area != before_area:
                debug["modified_masks"][label] = int(debug["modified_masks"].get(label, 0)) + 1
            refresh_track_frame(track, frame_idx, mask, H, W)

        cap = area_caps.get(label)
        if cap is not None:
            removed_frames: List[int] = []
            for frame_idx, area in list(track.get("area_by_frame", {}).items()):
                if float(area) > float(cap):
                    removed_frames.append(int(frame_idx))
            for frame_idx in removed_frames:
                track.get("mask_by_frame", {}).pop(frame_idx, None)
                track.get("box_by_frame", {}).pop(frame_idx, None)
                track.get("q_by_frame", {}).pop(frame_idx, None)
                track.get("area_by_frame", {}).pop(frame_idx, None)
            if removed_frames:
                debug["dropped_masks"][label] = int(debug["dropped_masks"].get(label, 0)) + len(removed_frames)

        visible = len(track.get("mask_by_frame", {}))
        if visible < min_visible:
            debug["dropped_tracks"][label] = int(debug["dropped_tracks"].get(label, 0)) + 1
            continue

        debug["kept_stuff_tracks"][label] = int(debug["kept_stuff_tracks"].get(label, 0)) + 1
        debug["kept_stuff_masks"][label] = int(debug["kept_stuff_masks"].get(label, 0)) + visible
        kept_tracks.append(track)

    filtered.tracks = kept_tracks
    filtered.num_masklets = len(kept_tracks)
    debug["output_tracks"] = int(len(kept_tracks))
    filtered.debug["offline_stuff_output_filter"] = debug
    return filtered, debug


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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse = load_sparse(Path(args.input_pt))
    keep_stuff = _label_set(args.keep_stuff_labels)
    drop_stuff = _label_set(args.drop_stuff_labels) or set()
    area_caps = _parse_area_caps(args.stuff_area_max)
    vertical_keep = _parse_vertical_keep(args.stuff_vertical_keep)
    keep_components = _parse_int_map(args.stuff_keep_components)
    min_component_area = _parse_float_map(args.stuff_min_component_area)

    image_paths, temp_dirs = _load_processing_frames(
        args,
        sparse.frame_height,
        sparse.frame_width,
        sparse.num_frames,
    )
    before_stats = coverage_stats(sparse)
    filtered, filter_debug = _filter_sparse(
        sparse,
        keep_stuff,
        drop_stuff,
        area_caps,
        vertical_keep,
        keep_components,
        min_component_area,
        int(args.subtract_thing_dilate_px),
        int(args.morph_open_px),
        int(args.morph_close_px),
        int(args.min_visible_frames_after),
    )
    after_stats = coverage_stats(filtered)

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    metrics_path = output_dir / "metrics_summary.json"
    contact_path = output_dir / "contact_before_after.jpg"

    save_sparse_output(output_pt, filtered)
    create_tracking_video_v2(
        image_paths,
        filtered,
        str(output_video),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style="clean",
    )
    make_contact_sheet(
        image_paths,
        sparse,
        filtered,
        parse_contact_frames(args.contact_frames, sparse.num_frames),
        contact_path,
        float(args.mask_alpha),
    )

    summary = {
        "input_pt": str(args.input_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "before": before_stats,
        "after": after_stats,
        "delta": {
            key: float(after_stats[key]) - float(before_stats[key])
            for key in before_stats.keys()
            if key in after_stats
        },
        "track_stats_after": track_stats(filtered),
        "filter_debug": filter_debug,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
