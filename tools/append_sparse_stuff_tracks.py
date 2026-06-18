#!/usr/bin/env python3
"""Append selected stuff tracks from an auxiliary sparse output to a base output."""

from __future__ import annotations

import argparse
import copy
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
    parser = argparse.ArgumentParser(description="Append selected auxiliary stuff tracks to a base sparse output.")
    parser.add_argument("--base_pt", required=True)
    parser.add_argument("--aux_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--append_labels", required=True)
    parser.add_argument("--thing_mask_pt", default="", help="Sparse file whose non-stuff tracks are subtracted from appended masks.")
    parser.add_argument("--thing_dilate_px", type=int, default=5)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--morph_open_px", type=int, default=0)
    parser.add_argument("--morph_close_px", type=int, default=1)
    parser.add_argument("--keep_components", default="mirror:1;door:1;screen:2;cabinet:2;table:2")
    parser.add_argument("--min_component_area", default="mirror:0.002;door:0.004;screen:0.002;cabinet:0.003;table:0.002")
    parser.add_argument("--area_max", default="mirror:0.20;door:0.38;screen:0.22;cabinet:0.25;table:0.18")
    parser.add_argument("--horizontal_keep", default="")
    parser.add_argument("--vertical_keep", default="")
    parser.add_argument("--bbox_center_x_keep", default="")
    parser.add_argument("--min_visible_frames", type=int, default=6)
    parser.add_argument("--contact_frames", default="0,30,60,90,120,150,180,210,240,270,299")
    return parser.parse_args()


def _parse_csv(value: str) -> List[str]:
    return [canonicalize_label(item.strip()) for item in str(value or "").split(",") if item.strip()]


def _parse_float_map(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw in str(spec or "").replace(",", ";").split(";"):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid float map item: {item!r}")
        key, value = item.split(":", 1)
        out[canonicalize_label(key.strip())] = float(value)
    return out


def _parse_int_map(spec: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for raw in str(spec or "").replace(",", ";").split(";"):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid int map item: {item!r}")
        key, value = item.split(":", 1)
        out[canonicalize_label(key.strip())] = int(value)
    return out


def _parse_range_map(spec: str, name: str) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for raw in str(spec or "").replace(",", ";").split(";"):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item or "-" not in item:
            raise ValueError(f"Invalid {name} range item: {item!r}")
        key, value = item.split(":", 1)
        start_text, end_text = value.split("-", 1)
        start = float(start_text)
        end = float(end_text)
        if start < 0.0 or end > 1.0 or start >= end:
            raise ValueError(f"Invalid {name} range {value!r} for {key!r}")
        out[canonicalize_label(key.strip())] = (start, end)
    return out


def _kernel(px: int) -> np.ndarray:
    size = max(int(px), 0)
    if size <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    size = size * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _load_processing_frames(args: argparse.Namespace, expected_h: int, expected_w: int, num_frames: int) -> Tuple[List[str], List[str]]:
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


def _apply_vertical(mask: np.ndarray, label: str, vertical: Dict[str, Tuple[float, float]]) -> np.ndarray:
    if label not in vertical or not mask.any():
        return mask
    H, _W = mask.shape
    y0, y1 = vertical[label]
    keep = np.zeros_like(mask, dtype=bool)
    keep[int(round(y0 * H)) : int(round(y1 * H)), :] = True
    return mask & keep


def _apply_horizontal(mask: np.ndarray, label: str, horizontal: Dict[str, Tuple[float, float]]) -> np.ndarray:
    if label not in horizontal or not mask.any():
        return mask
    _H, W = mask.shape
    x0, x1 = horizontal[label]
    keep = np.zeros_like(mask, dtype=bool)
    keep[:, int(round(x0 * W)) : int(round(x1 * W))] = True
    return mask & keep


def _keep_components(
    mask: np.ndarray,
    keep_count: int,
    min_area_ratio: float,
    center_x_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    if not mask.any():
        return mask
    H, W = mask.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    min_area = int(round(float(min_area_ratio) * float(H * W)))
    components: List[Tuple[int, int]] = []
    for component_id in range(1, n):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            if center_x_range is not None:
                cx = float(centroids[component_id][0]) / float(max(W, 1))
                x0, x1 = center_x_range
                if cx < x0 or cx > x1:
                    continue
            components.append((area, component_id))
    if not components:
        return np.zeros_like(mask, dtype=bool)
    components.sort(reverse=True)
    keep_ids = [component_id for _area, component_id in components[: max(int(keep_count), 1)]]
    return np.isin(labels, keep_ids)


def _clean_mask(
    mask: np.ndarray,
    label: str,
    thing_mask: np.ndarray,
    thing_kernel: np.ndarray,
    thing_dilate_px: int,
    horizontal: Dict[str, Tuple[float, float]],
    vertical: Dict[str, Tuple[float, float]],
    center_x_keep: Dict[str, Tuple[float, float]],
    keep_components: Dict[str, int],
    min_component_area: Dict[str, float],
    open_px: int,
    close_px: int,
) -> np.ndarray:
    out = mask.astype(bool)
    if int(thing_dilate_px) > 0 and thing_mask.any():
        thing = cv2.dilate(thing_mask.astype(np.uint8), thing_kernel, iterations=1).astype(bool)
        out &= ~thing
    out = _apply_horizontal(out, label, horizontal)
    out = _apply_vertical(out, label, vertical)
    if int(open_px) > 0 and out.any():
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, _kernel(open_px)).astype(bool)
    if int(close_px) > 0 and out.any():
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(close_px)).astype(bool)
    out = _keep_components(
        out,
        keep_components.get(label, 1),
        min_component_area.get(label, 0.0),
        center_x_keep.get(label),
    )
    return out.astype(bool)


def _selected_aux_tracks(aux: Any, labels: List[str]) -> List[Dict[str, Any]]:
    wanted = set(labels)
    tracks: List[Dict[str, Any]] = []
    for track in aux.tracks:
        if str(track.get("source_type")) != "stuff_static":
            continue
        label = canonicalize_label(str(track.get("L_sem", "")))
        if label in wanted:
            tracks.append(copy.deepcopy(track))
    return tracks


def _shape_check(base: Any, other: Any, name: str) -> None:
    if int(base.num_frames) != int(other.num_frames):
        raise RuntimeError(f"{name} frame count mismatch: {other.num_frames} vs base {base.num_frames}")
    if (int(base.frame_height), int(base.frame_width)) != (int(other.frame_height), int(other.frame_width)):
        raise RuntimeError(f"{name} shape mismatch: {(other.frame_height, other.frame_width)} vs base {(base.frame_height, base.frame_width)}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_sparse(Path(args.base_pt))
    aux = load_sparse(Path(args.aux_pt))
    _shape_check(base, aux, "aux")
    thing_sparse = load_sparse(Path(args.thing_mask_pt)) if str(args.thing_mask_pt or "").strip() else base
    _shape_check(base, thing_sparse, "thing_mask")

    labels = _parse_csv(args.append_labels)
    area_max = _parse_float_map(args.area_max)
    horizontal = _parse_range_map(args.horizontal_keep, "horizontal_keep")
    vertical = _parse_range_map(args.vertical_keep, "vertical_keep")
    center_x_keep = _parse_range_map(args.bbox_center_x_keep, "bbox_center_x_keep")
    keep_components = _parse_int_map(args.keep_components)
    min_component_area = _parse_float_map(args.min_component_area)
    H, W = int(base.frame_height), int(base.frame_width)
    thing_kernel = _kernel(int(args.thing_dilate_px))

    output = clone_sparse(base)
    debug: Dict[str, Any] = {
        "format": "append_sparse_stuff_tracks_v1",
        "base_pt": str(args.base_pt),
        "aux_pt": str(args.aux_pt),
        "thing_mask_pt": str(args.thing_mask_pt or args.base_pt),
        "append_labels": labels,
        "area_max": area_max,
        "horizontal_keep": horizontal,
        "thing_dilate_px": int(args.thing_dilate_px),
        "bbox_center_x_keep": center_x_keep,
        "input_aux_tracks": 0,
        "appended_tracks": {},
        "dropped_short": {},
        "dropped_area_frames": {},
        "modified_frames": {},
        "thing_removed_pixels": {},
    }

    appended: List[Dict[str, Any]] = []
    for track in _selected_aux_tracks(aux, labels):
        label = canonicalize_label(str(track.get("L_sem", "")))
        debug["input_aux_tracks"] += 1
        for frame_idx in sorted(int(idx) for idx in list(track.get("mask_by_frame", {}).keys())):
            packed = track.get("mask_by_frame", {}).get(frame_idx)
            if packed is None:
                continue
            before = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
            thing = build_thing_union(thing_sparse, frame_idx)
            after = _clean_mask(
                before,
                label,
                thing,
                thing_kernel,
                int(args.thing_dilate_px),
                horizontal,
                vertical,
                center_x_keep,
                keep_components,
                min_component_area,
                int(args.morph_open_px),
                int(args.morph_close_px),
            )
            removed_by_thing = int((before & thing).sum()) if thing.any() else 0
            if removed_by_thing:
                debug["thing_removed_pixels"][label] = int(debug["thing_removed_pixels"].get(label, 0)) + removed_by_thing
            if int(after.sum()) != int(before.sum()):
                debug["modified_frames"][label] = int(debug["modified_frames"].get(label, 0)) + 1
            area_ratio = float(after.sum()) / float(max(H * W, 1))
            if area_ratio > float(area_max.get(label, 1.0)):
                refresh_track_frame(track, frame_idx, np.zeros((H, W), dtype=bool), H, W)
                debug["dropped_area_frames"][label] = int(debug["dropped_area_frames"].get(label, 0)) + 1
                continue
            refresh_track_frame(track, frame_idx, after, H, W)
        visible = len(track.get("mask_by_frame", {}))
        if visible < int(args.min_visible_frames):
            debug["dropped_short"][label] = int(visible)
            continue
        appended.append(track)
        debug["appended_tracks"][label] = int(visible)

    output.tracks.extend(appended)
    output.num_masklets = len(output.tracks)
    output.debug["append_sparse_stuff_tracks"] = debug

    image_paths, temp_dirs = _load_processing_frames(args, H, W, int(output.num_frames))
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
        "base_pt": str(args.base_pt),
        "aux_pt": str(args.aux_pt),
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
        "append_debug": debug,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
