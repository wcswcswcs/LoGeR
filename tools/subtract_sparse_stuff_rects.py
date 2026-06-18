#!/usr/bin/env python3
"""Subtract audited regions from selected sparse stuff tracks.

This is a scene-audit tool, not a general semantic backend. It is useful when a
known false-positive region, such as a mirror/reflection opening, can be
documented as frame keyframes and linearly interpolated through a short video
range. The region can be either a rectangle or a polygon with a fixed number of
vertices across keyframes.
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


Rect = Tuple[float, float, float, float]
Point = Tuple[float, float]
Polygon = Tuple[Point, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subtract interpolated audited regions from selected stuff masks.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--labels", default="curtain")
    parser.add_argument(
        "--rect_keyframes",
        default="",
        help="Semicolon-separated frame:x0,y0,x1,y1 entries in processed-frame pixels.",
    )
    parser.add_argument(
        "--polygon_keyframes",
        default="",
        help="Semicolon-separated frame:x0,y0,x1,y1,... entries. Each keyframe must have the same vertex count.",
    )
    parser.add_argument("--rect_pad_px", type=int, default=0)
    parser.add_argument("--apply_before_first", type=int, default=0)
    parser.add_argument("--apply_after_last", type=int, default=0)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--contact_frames", default="120,150,179,210,240,270,299")
    return parser.parse_args()


def _parse_csv(value: str) -> List[str]:
    labels = [canonicalize_label(item.strip()) for item in str(value or "").split(",") if item.strip()]
    if not labels:
        raise ValueError("--labels resolved to an empty label set")
    return labels


def _parse_rect_keyframes(spec: str) -> List[Tuple[int, Rect]]:
    keyframes: List[Tuple[int, Rect]] = []
    for raw in str(spec or "").replace("|", ";").split(";"):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid rect keyframe item: {item!r}")
        frame_text, rect_text = item.split(":", 1)
        values = [float(x.strip()) for x in rect_text.split(",") if x.strip()]
        if len(values) != 4:
            raise ValueError(f"Invalid rect values for {item!r}; expected x0,y0,x1,y1")
        x0, y0, x1, y1 = values
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid rect with non-positive size: {item!r}")
        keyframes.append((int(frame_text.strip()), (x0, y0, x1, y1)))
    if not keyframes:
        raise ValueError("--rect_keyframes produced no entries")
    keyframes.sort(key=lambda pair: pair[0])
    return keyframes


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
        raise ValueError("--polygon_keyframes produced no entries")
    keyframes.sort(key=lambda pair: pair[0])
    return keyframes


def _interp_rect(
    frame_idx: int,
    keyframes: Sequence[Tuple[int, Rect]],
    apply_before_first: bool,
    apply_after_last: bool,
) -> Rect | None:
    if frame_idx < keyframes[0][0]:
        return keyframes[0][1] if apply_before_first else None
    if frame_idx > keyframes[-1][0]:
        return keyframes[-1][1] if apply_after_last else None
    for idx, (frame, rect) in enumerate(keyframes):
        if frame_idx == frame:
            return rect
        if frame_idx < frame:
            prev_frame, prev_rect = keyframes[idx - 1]
            denom = max(frame - prev_frame, 1)
            alpha = float(frame_idx - prev_frame) / float(denom)
            return tuple(
                float(prev_rect[i]) * (1.0 - alpha) + float(rect[i]) * alpha
                for i in range(4)
            )  # type: ignore[return-value]
    return None


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


def _rect_mask(shape: Tuple[int, int], rect: Rect, pad: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    H, W = shape
    x0, y0, x1, y1 = rect
    x0i = max(0, min(W, int(round(x0)) - int(pad)))
    y0i = max(0, min(H, int(round(y0)) - int(pad)))
    x1i = max(0, min(W, int(round(x1)) + int(pad)))
    y1i = max(0, min(H, int(round(y1)) + int(pad)))
    mask = np.zeros((H, W), dtype=bool)
    if x1i > x0i and y1i > y0i:
        mask[y0i:y1i, x0i:x1i] = True
    return mask, (x0i, y0i, x1i, y1i)


def _polygon_mask(shape: Tuple[int, int], polygon: Polygon, pad: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
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
    return mask.astype(bool), [(int(x), int(y)) for x, y in pts.tolist()]


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
    image_paths = list(image_paths[:num_frames])
    if tuple(proc_shape) != (expected_h, expected_w):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    return image_paths, temp_dirs


def _write_region_contact(
    image_paths: Sequence[str],
    region_kind: str,
    rect_keyframes: Sequence[Tuple[int, Rect]],
    polygon_keyframes: Sequence[Tuple[int, Polygon]],
    frames: Sequence[int],
    output_path: Path,
    H: int,
    W: int,
    pad: int,
    apply_before_first: bool,
    apply_after_last: bool,
) -> None:
    cells: List[np.ndarray] = []
    for frame_idx in frames:
        if frame_idx < 0 or frame_idx >= len(image_paths):
            continue
        bgr = cv2.imread(image_paths[frame_idx], cv2.IMREAD_COLOR)
        if bgr is None:
            rgb = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H))
        if region_kind == "polygon":
            polygon = _interp_polygon(frame_idx, polygon_keyframes, apply_before_first, apply_after_last)
            if polygon is not None:
                _mask, pts = _polygon_mask((H, W), polygon, pad)
                cv2.polylines(rgb, [np.asarray(pts, dtype=np.int32)], True, (255, 40, 40), 3)
        else:
            rect = _interp_rect(frame_idx, rect_keyframes, apply_before_first, apply_after_last)
            if rect is not None:
                _mask, (x0, y0, x1, y1) = _rect_mask((H, W), rect, pad)
                cv2.rectangle(rgb, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (255, 40, 40), 3)
        cv2.putText(
            rgb,
            f"frame {frame_idx}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cells.append(rgb)
    if not cells:
        return
    rows = []
    for start in range(0, len(cells), 2):
        row_cells = cells[start : start + 2]
        if len(row_cells) == 1:
            row_cells.append(np.zeros_like(row_cells[0]))
        rows.append(np.concatenate(row_cells, axis=1))
    contact = np.concatenate(rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = set(_parse_csv(args.labels))
    has_polygon = bool(str(args.polygon_keyframes or "").strip())
    has_rect = bool(str(args.rect_keyframes or "").strip())
    if has_polygon and has_rect:
        raise ValueError("Use only one of --polygon_keyframes or --rect_keyframes.")
    if not has_polygon and not has_rect:
        raise ValueError("One of --polygon_keyframes or --rect_keyframes is required.")
    region_kind = "polygon" if has_polygon else "rect"
    rect_keyframes = _parse_rect_keyframes(args.rect_keyframes) if has_rect else []
    polygon_keyframes = _parse_polygon_keyframes(args.polygon_keyframes) if has_polygon else []
    apply_before_first = bool(int(args.apply_before_first))
    apply_after_last = bool(int(args.apply_after_last))

    base = load_sparse(Path(args.input_pt))
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

    debug: Dict[str, Any] = {
        "format": "subtract_sparse_stuff_rects_v1",
        "input_pt": str(args.input_pt),
        "labels": sorted(labels),
        "region_kind": region_kind,
        "rect_keyframes": [
            {"frame": int(frame), "rect_xyxy": [float(v) for v in rect]}
            for frame, rect in rect_keyframes
        ],
        "polygon_keyframes": [
            {
                "frame": int(frame),
                "points_xy": [[float(x), float(y)] for x, y in polygon],
            }
            for frame, polygon in polygon_keyframes
        ],
        "rect_pad_px": int(args.rect_pad_px),
        "apply_before_first": int(apply_before_first),
        "apply_after_last": int(apply_after_last),
        "frames_modified": 0,
        "pixels_removed": 0,
        "per_frame": [],
    }

    for frame_idx in range(int(output.num_frames)):
        if region_kind == "polygon":
            polygon = _interp_polygon(frame_idx, polygon_keyframes, apply_before_first, apply_after_last)
            if polygon is None:
                continue
            region_mask, region_points = _polygon_mask((H, W), polygon, int(args.rect_pad_px))
            region_record: Dict[str, Any] = {"polygon_xy": [[int(x), int(y)] for x, y in region_points]}
        else:
            rect = _interp_rect(frame_idx, rect_keyframes, apply_before_first, apply_after_last)
            if rect is None:
                continue
            region_mask, rect_xyxy = _rect_mask((H, W), rect, int(args.rect_pad_px))
            region_record = {"rect_xyxy": [int(v) for v in rect_xyxy]}
        if not region_mask.any():
            continue
        frame_removed = 0
        labels_removed: Dict[str, int] = {}
        for track in output.tracks:
            if str(track.get("source_type")) != "stuff_static":
                continue
            label = canonicalize_label(str(track.get("L_sem", "")))
            if label not in labels:
                continue
            packed = track.get("mask_by_frame", {}).get(int(frame_idx))
            if packed is None:
                continue
            mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
            removed = int((mask & region_mask).sum())
            if removed <= 0:
                continue
            mask &= ~region_mask
            refresh_track_frame(track, int(frame_idx), mask, H, W)
            frame_removed += removed
            labels_removed[label] = labels_removed.get(label, 0) + removed
        if frame_removed:
            debug["frames_modified"] += 1
            debug["pixels_removed"] += int(frame_removed)
            row = {
                "frame_idx": int(frame_idx),
                "pixels_removed": int(frame_removed),
                "labels_removed": labels_removed,
            }
            row.update(region_record)
            debug["per_frame"].append(row)

    output.debug["subtract_sparse_stuff_rects"] = debug
    output.num_masklets = len(output.tracks)

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    contact_path = output_dir / "contact_before_after.jpg"
    rect_contact_path = output_dir / "manual_region_contact.jpg"
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
    make_contact_sheet(
        image_paths,
        base,
        output,
        contact_frames,
        contact_path,
        float(args.mask_alpha),
    )
    _write_region_contact(
        image_paths,
        region_kind,
        rect_keyframes,
        polygon_keyframes,
        contact_frames,
        rect_contact_path,
        H,
        W,
        int(args.rect_pad_px),
        apply_before_first,
        apply_after_last,
    )

    before_stats = coverage_stats(base)
    after_stats = coverage_stats(output)
    summary = {
        "input_pt": str(args.input_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "manual_region_contact": str(rect_contact_path),
        "before": before_stats,
        "after": after_stats,
        "delta": {
            key: float(after_stats[key]) - float(before_stats[key])
            for key in before_stats
            if key in after_stats
        },
        "track_stats_after": track_stats(output),
        "rect_debug": debug,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
