#!/usr/bin/env python3
"""Subtract sparse mask tracks from selected stuff tracks.

This audit tool applies a precomputed negative sparse mask, such as a SAM2
prompt/VOS mirror-opening track, to one or more stuff labels in a base sparse
masklet file. It does not classify or generate new semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    make_contact_sheet,
    parse_contact_frames,
    refresh_track_frame,
    track_stats,
)
from run_video_masklet_front_end import collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402
from loger.pipeline.video_masklet_frontend import canonicalize_label  # noqa: E402

Point = Tuple[float, float]
Polygon = Tuple[Point, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subtract sparse negative masks from selected stuff tracks.")
    parser.add_argument("--base_pt", required=True)
    parser.add_argument("--negative_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--edit_labels", default="curtain")
    parser.add_argument("--negative_labels", default="mirror")
    parser.add_argument("--negative_dilate_px", type=int, default=0)
    parser.add_argument("--protect_red", type=int, default=0, help="Do not subtract negative pixels that look like red curtain.")
    parser.add_argument("--protect_red_hue_ranges", default="0-24,160-179")
    parser.add_argument("--protect_red_sat_min", type=int, default=55)
    parser.add_argument("--protect_red_value_min", type=int, default=25)
    parser.add_argument("--protect_red_value_max", type=int, default=245)
    parser.add_argument("--protect_red_dilate_px", type=int, default=0)
    parser.add_argument("--protect_red_keep_components", type=int, default=0, help="If >0, keep only the largest N red protect components.")
    parser.add_argument("--protect_red_min_component_area", type=float, default=0.0, help="Minimum red protect component area ratio.")
    parser.add_argument(
        "--restrict_polygon_keyframes",
        default="",
        help="Optional semicolon-separated frame:x0,y0,x1,y1,... polygons. Negative mask is intersected with the interpolated polygon.",
    )
    parser.add_argument("--restrict_polygon_pad_px", type=int, default=0)
    parser.add_argument("--restrict_apply_before_first", type=int, default=0)
    parser.add_argument("--restrict_apply_after_last", type=int, default=0)
    parser.add_argument("--apply_only_frames_with_negative", type=int, default=1)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--contact_frames", default="120,150,179,210,240,270,299")
    return parser.parse_args()


def _parse_csv(value: str) -> List[str]:
    labels = [canonicalize_label(item.strip()) for item in str(value or "").split(",") if item.strip()]
    if not labels:
        raise ValueError("empty label list")
    return labels


def _kernel(px: int) -> np.ndarray:
    size = max(int(px), 0)
    if size <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    size = size * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _parse_hue_ranges(spec: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for raw in str(spec or "").replace(";", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" not in item:
            value = max(0, min(179, int(item)))
            ranges.append((value, value))
            continue
        lo_text, hi_text = item.split("-", 1)
        lo = max(0, min(179, int(lo_text)))
        hi = max(0, min(179, int(hi_text)))
        if lo > hi:
            raise ValueError(f"Invalid hue range: {item!r}")
        ranges.append((lo, hi))
    if not ranges:
        raise ValueError("--protect_red_hue_ranges produced no ranges")
    return ranges


def _parse_polygon_keyframes(spec: str) -> List[Tuple[int, Polygon]]:
    keyframes: List[Tuple[int, Polygon]] = []
    expected_vertices: int | None = None
    for raw in str(spec or "").replace("|", ";").split(";"):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid polygon keyframe item: {item!r}")
        frame_text, points_text = item.split(":", 1)
        values = [float(x.strip()) for x in points_text.split(",") if x.strip()]
        if len(values) < 6 or len(values) % 2 != 0:
            raise ValueError(f"Invalid polygon values for {item!r}; expected x0,y0,x1,y1,...")
        points = tuple((values[i], values[i + 1]) for i in range(0, len(values), 2))
        if expected_vertices is None:
            expected_vertices = len(points)
        elif len(points) != expected_vertices:
            raise ValueError(
                f"Polygon keyframe {item!r} has {len(points)} vertices; expected {expected_vertices}"
            )
        keyframes.append((int(frame_text.strip()), points))
    if not keyframes:
        raise ValueError("--restrict_polygon_keyframes produced no entries")
    keyframes.sort(key=lambda pair: pair[0])
    return keyframes


def _interp_polygon(
    frame_idx: int,
    keyframes: Sequence[Tuple[int, Polygon]],
    apply_before_first: bool,
    apply_after_last: bool,
) -> Polygon | None:
    if frame_idx < keyframes[0][0]:
        return keyframes[0][1] if apply_before_first else None
    if frame_idx > keyframes[-1][0]:
        return keyframes[-1][1] if apply_after_last else None
    for idx, (frame, polygon) in enumerate(keyframes):
        if frame_idx == frame:
            return polygon
        if frame_idx < frame:
            prev_frame, prev_polygon = keyframes[idx - 1]
            denom = max(frame - prev_frame, 1)
            alpha = float(frame_idx - prev_frame) / float(denom)
            return tuple(
                (
                    float(prev_polygon[i][0]) * (1.0 - alpha) + float(polygon[i][0]) * alpha,
                    float(prev_polygon[i][1]) * (1.0 - alpha) + float(polygon[i][1]) * alpha,
                )
                for i in range(len(polygon))
            )
    return None


def _polygon_mask(shape: Tuple[int, int], polygon: Polygon, pad: int) -> np.ndarray:
    H, W = shape
    pts = np.asarray(
        [
            [
                max(0, min(W - 1, int(round(x)))),
                max(0, min(H - 1, int(round(y)))),
            ]
            for x, y in polygon
        ],
        dtype=np.int32,
    )
    mask = np.zeros((H, W), dtype=np.uint8)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], 1)
    if int(pad) > 0 and mask.any():
        size = int(pad) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def _red_protect_mask(
    image_path: str,
    H: int,
    W: int,
    hue_ranges: Sequence[Tuple[int, int]],
    sat_min: int,
    value_min: int,
    value_max: int,
    dilate_kernel: np.ndarray,
    dilate_px: int,
    keep_components: int,
    min_component_area: float,
) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((H, W), dtype=bool)
    if bgr.shape[:2] != (H, W):
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = np.zeros((H, W), dtype=bool)
    for lo, hi in hue_ranges:
        mask |= (hue >= int(lo)) & (hue <= int(hi))
    mask &= sat >= int(sat_min)
    mask &= value >= int(value_min)
    mask &= value <= int(value_max)
    if int(dilate_px) > 0 and mask.any():
        mask = cv2.dilate(mask.astype(np.uint8), dilate_kernel, iterations=1).astype(bool)
    if mask.any() and (int(keep_components) > 0 or float(min_component_area) > 0.0):
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        components: List[Tuple[int, int]] = []
        min_pixels = int(round(float(min_component_area) * float(H * W)))
        for comp_id in range(1, int(num_labels)):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < min_pixels:
                continue
            components.append((area, comp_id))
        components.sort(reverse=True)
        if int(keep_components) > 0:
            components = components[: int(keep_components)]
        kept_ids = {comp_id for _area, comp_id in components}
        mask = np.isin(labels, list(kept_ids)) if kept_ids else np.zeros((H, W), dtype=bool)
    return mask


def _shape_check(base: Any, other: Any, name: str) -> None:
    if int(base.num_frames) != int(other.num_frames):
        raise RuntimeError(f"{name} frame count mismatch: {other.num_frames} vs {base.num_frames}")
    base_shape = (int(base.frame_height), int(base.frame_width))
    other_shape = (int(other.frame_height), int(other.frame_width))
    if other_shape != base_shape:
        raise RuntimeError(f"{name} shape mismatch: {other_shape} vs {base_shape}")


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
    image_paths = list(image_paths[: int(num_frames)])
    if tuple(proc_shape) != (int(expected_h), int(expected_w)):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    return image_paths, temp_dirs


def _negative_union(
    sparse: Any,
    labels: Sequence[str],
    frame_idx: int,
    H: int,
    W: int,
    dilate_kernel: np.ndarray,
    dilate_px: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    wanted = set(labels)
    union = np.zeros((H, W), dtype=bool)
    per_label: Dict[str, int] = {}
    for track in sparse.tracks:
        if str(track.get("source_type")) != "stuff_static":
            continue
        label = canonicalize_label(str(track.get("L_sem", "")))
        if label not in wanted:
            continue
        packed = track.get("mask_by_frame", {}).get(int(frame_idx))
        if packed is None:
            continue
        mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
        if int(dilate_px) > 0 and mask.any():
            mask = cv2.dilate(mask.astype(np.uint8), dilate_kernel, iterations=1).astype(bool)
        per_label[label] = per_label.get(label, 0) + int(mask.sum())
        union |= mask
    return union, per_label


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_sparse(Path(args.base_pt))
    negative = load_sparse(Path(args.negative_pt))
    _shape_check(base, negative, "negative")
    output = clone_sparse(base)
    H, W = int(output.frame_height), int(output.frame_width)
    edit_labels = set(_parse_csv(args.edit_labels))
    negative_labels = _parse_csv(args.negative_labels)
    dilate_kernel = _kernel(int(args.negative_dilate_px))
    red_hue_ranges = _parse_hue_ranges(args.protect_red_hue_ranges)
    red_kernel = _kernel(int(args.protect_red_dilate_px))
    restrict_polygon_keyframes = (
        _parse_polygon_keyframes(args.restrict_polygon_keyframes)
        if str(args.restrict_polygon_keyframes or "").strip()
        else []
    )

    image_paths, temp_dirs = _load_processing_frames(
        args.input_video,
        int(args.processing_max_side),
        int(args.frames_limit),
        H,
        W,
        int(output.num_frames),
    )

    debug: Dict[str, Any] = {
        "format": "subtract_sparse_stuff_masks_v1",
        "base_pt": str(args.base_pt),
        "negative_pt": str(args.negative_pt),
        "edit_labels": sorted(edit_labels),
        "negative_labels": list(negative_labels),
        "negative_dilate_px": int(args.negative_dilate_px),
        "protect_red": int(args.protect_red),
        "protect_red_hue_ranges": [[int(lo), int(hi)] for lo, hi in red_hue_ranges],
        "protect_red_sat_min": int(args.protect_red_sat_min),
        "protect_red_value_min": int(args.protect_red_value_min),
        "protect_red_value_max": int(args.protect_red_value_max),
        "protect_red_dilate_px": int(args.protect_red_dilate_px),
        "protect_red_keep_components": int(args.protect_red_keep_components),
        "protect_red_min_component_area": float(args.protect_red_min_component_area),
        "restrict_polygon_keyframes": [
            {"frame": int(frame), "points_xy": [[float(x), float(y)] for x, y in polygon]}
            for frame, polygon in restrict_polygon_keyframes
        ],
        "restrict_polygon_pad_px": int(args.restrict_polygon_pad_px),
        "restrict_apply_before_first": int(args.restrict_apply_before_first),
        "restrict_apply_after_last": int(args.restrict_apply_after_last),
        "apply_only_frames_with_negative": int(args.apply_only_frames_with_negative),
        "frames_with_negative": 0,
        "frames_modified": 0,
        "negative_pixels_union": 0,
        "negative_pixels_after_polygon_restrict": 0,
        "red_protect_pixels_union": 0,
        "negative_pixels_after_red_protect": 0,
        "pixels_removed": 0,
        "per_frame": [],
    }

    for frame_idx in range(int(output.num_frames)):
        negative_mask, negative_per_label = _negative_union(
            negative,
            negative_labels,
            frame_idx,
            H,
            W,
            dilate_kernel,
            int(args.negative_dilate_px),
        )
        if negative_mask.any():
            debug["frames_with_negative"] += 1
            debug["negative_pixels_union"] += int(negative_mask.sum())
        elif int(args.apply_only_frames_with_negative):
            continue
        polygon_restricted_pixels = 0
        if restrict_polygon_keyframes and negative_mask.any():
            polygon = _interp_polygon(
                frame_idx,
                restrict_polygon_keyframes,
                bool(int(args.restrict_apply_before_first)),
                bool(int(args.restrict_apply_after_last)),
            )
            if polygon is None:
                negative_mask &= False
            else:
                region = _polygon_mask((H, W), polygon, int(args.restrict_polygon_pad_px))
                before_restrict = int(negative_mask.sum())
                negative_mask &= region
                polygon_restricted_pixels = before_restrict - int(negative_mask.sum())
        debug["negative_pixels_after_polygon_restrict"] += int(negative_mask.sum())
        red_protect = np.zeros((H, W), dtype=bool)
        red_protected_pixels = 0
        if int(args.protect_red) and negative_mask.any():
            red_protect = _red_protect_mask(
                image_paths[frame_idx],
                H,
                W,
                red_hue_ranges,
                int(args.protect_red_sat_min),
                int(args.protect_red_value_min),
                int(args.protect_red_value_max),
                red_kernel,
                int(args.protect_red_dilate_px),
                int(args.protect_red_keep_components),
                float(args.protect_red_min_component_area),
            )
            red_protected_pixels = int((negative_mask & red_protect).sum())
            debug["red_protect_pixels_union"] += red_protected_pixels
            negative_mask &= ~red_protect
        debug["negative_pixels_after_red_protect"] += int(negative_mask.sum())
        if not negative_mask.any() and int(args.apply_only_frames_with_negative):
            continue

        frame_removed = 0
        labels_removed: Dict[str, int] = {}
        for track in output.tracks:
            if str(track.get("source_type")) != "stuff_static":
                continue
            label = canonicalize_label(str(track.get("L_sem", "")))
            if label not in edit_labels:
                continue
            packed = track.get("mask_by_frame", {}).get(int(frame_idx))
            if packed is None:
                continue
            mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
            removed = int((mask & negative_mask).sum())
            if removed <= 0:
                continue
            mask &= ~negative_mask
            refresh_track_frame(track, int(frame_idx), mask, H, W)
            frame_removed += removed
            labels_removed[label] = labels_removed.get(label, 0) + removed
        if frame_removed:
            debug["frames_modified"] += 1
            debug["pixels_removed"] += int(frame_removed)
            debug["per_frame"].append(
                {
                    "frame_idx": int(frame_idx),
                    "negative_pixels": int(negative_mask.sum()),
                    "negative_per_label": negative_per_label,
                    "polygon_restricted_pixels": int(polygon_restricted_pixels),
                    "red_protected_pixels": int(red_protected_pixels),
                    "pixels_removed": int(frame_removed),
                    "labels_removed": labels_removed,
                }
            )

    output.debug["subtract_sparse_stuff_masks"] = debug
    output.num_masklets = len(output.tracks)

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    contact_path = output_dir / "contact_before_after.jpg"
    negative_contact_path = output_dir / "negative_mask_contact.jpg"
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
    contact_frames = parse_contact_frames(args.contact_frames, int(output.num_frames))
    make_contact_sheet(image_paths, base, output, contact_frames, contact_path, float(args.mask_alpha))
    _make_single_contact(image_paths, negative, contact_frames, negative_contact_path, float(args.mask_alpha))

    before = coverage_stats(base)
    after = coverage_stats(output)
    summary = {
        "base_pt": str(args.base_pt),
        "negative_pt": str(args.negative_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "negative_contact": str(negative_contact_path),
        "before": before,
        "after": after,
        "delta": {key: float(after[key]) - float(before[key]) for key in before.keys()},
        "track_stats_after": track_stats(output),
        "subtract_debug": debug,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
