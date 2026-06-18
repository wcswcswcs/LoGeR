#!/usr/bin/env python3
"""Fuse two sparse stuff candidates with auditable indoor-scene constraints."""

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
import torch

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
    track_stats,
)
from run_video_masklet_front_end import (  # noqa: E402
    _make_sparse_stuff_track,
    _mask_to_box_np,
    _pack_mask_np,
    collect_image_paths,
    prepare_processing_image_paths,
)
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402
from loger.pipeline.video_masklet_frontend import canonicalize_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse sparse stuff candidate outputs.")
    parser.add_argument("--base_pt", required=True, help="Sparse file whose non-stuff tracks are kept.")
    parser.add_argument("--primary_stuff_pt", required=True, help="Conservative stuff source, e.g. LSeg scenegeom.")
    parser.add_argument("--secondary_stuff_pt", required=True, help="Supplemental stuff source, e.g. GroundedSAM2 negative.")
    parser.add_argument("--negative_mask_pt", default="", help="Optional sparse file with labels to subtract, e.g. curtain/person masks.")
    parser.add_argument("--negative_mask_labels", default="", help="Comma-separated labels from --negative_mask_pt to subtract.")
    parser.add_argument("--negative_mask_dilate_px", type=int, default=3)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--labels", default="wall,floor,ceiling")
    parser.add_argument("--red_exclusion_labels", default="wall,ceiling")
    parser.add_argument("--red_sat_min", type=int, default=80)
    parser.add_argument("--red_value_min", type=int, default=30)
    parser.add_argument("--red_value_max", type=int, default=190)
    parser.add_argument("--red_dilate_px", type=int, default=3)
    parser.add_argument("--red_exclude_primary", type=int, default=1)
    parser.add_argument("--thing_dilate_px", type=int, default=7)
    parser.add_argument("--morph_open_px", type=int, default=1)
    parser.add_argument("--morph_close_px", type=int, default=2)
    parser.add_argument("--keep_components", default="wall:2;floor:1;ceiling:1")
    parser.add_argument("--min_component_area", default="wall:0.006;floor:0.003;ceiling:0.002")
    parser.add_argument("--area_max", default="wall:0.55;floor:0.30;ceiling:0.12")
    parser.add_argument("--secondary_area_max", default="wall:0.42;floor:0.26;ceiling:0.10")
    parser.add_argument("--vertical_keep", default="floor:0.45-1.0;ceiling:0.0-0.38")
    parser.add_argument("--min_visible_frames", type=int, default=8)
    parser.add_argument("--contact_frames", default="0,30,60,90,120,150,180,210,240,270,299")
    return parser.parse_args()


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


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


def _parse_vertical_map(spec: str) -> Dict[str, tuple[float, float]]:
    out: Dict[str, tuple[float, float]] = {}
    for raw in str(spec or "").replace(",", ";").split(";"):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item or "-" not in item:
            raise ValueError(f"Invalid vertical range item: {item!r}")
        key, value = item.split(":", 1)
        y0_text, y1_text = value.split("-", 1)
        y0 = float(y0_text)
        y1 = float(y1_text)
        if y0 < 0.0 or y1 > 1.0 or y0 >= y1:
            raise ValueError(f"Invalid vertical range {value!r} for {key!r}")
        out[canonicalize_label(key.strip())] = (y0, y1)
    return out


def _kernel(px: int) -> np.ndarray:
    size = max(int(px), 0)
    if size <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    size = size * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _unpack_track_mask(track: Optional[Dict[str, Any]], frame_idx: int, H: int, W: int) -> Optional[np.ndarray]:
    if track is None:
        return None
    packed = track.get("mask_by_frame", {}).get(int(frame_idx))
    if packed is None:
        return None
    return _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)


def _label_tracks(sparse: Any, labels: Sequence[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    wanted = {canonicalize_label(label) for label in labels}
    out: Dict[str, Optional[Dict[str, Any]]] = {label: None for label in wanted}
    for track in sparse.tracks:
        if str(track.get("source_type")) != "stuff_static":
            continue
        label = canonicalize_label(str(track.get("L_sem", "")))
        if label in wanted and out.get(label) is None:
            out[label] = track
    return out


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


def _red_risk_mask(
    image_path: str,
    H: int,
    W: int,
    sat_min: int,
    value_min: int,
    value_max: int,
    dilate_px: int,
) -> np.ndarray:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((H, W), dtype=bool)
    if bgr.shape[:2] != (H, W):
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red_hue = (h <= 10) | (h >= 165)
    mask = red_hue & (s >= int(sat_min)) & (v >= int(value_min)) & (v <= int(value_max))
    if int(dilate_px) > 0:
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(dilate_px)).astype(bool)
        mask = cv2.dilate(mask.astype(np.uint8), _kernel(dilate_px), iterations=1).astype(bool)
    return mask.astype(bool)


def _apply_vertical(mask: np.ndarray, label: str, vertical: Dict[str, tuple[float, float]]) -> np.ndarray:
    if label not in vertical or not mask.any():
        return mask
    H, _W = mask.shape
    y0, y1 = vertical[label]
    keep = np.zeros_like(mask, dtype=bool)
    keep[int(round(y0 * H)) : int(round(y1 * H)), :] = True
    return mask & keep


def _keep_components(mask: np.ndarray, keep_count: int, min_area_ratio: float) -> np.ndarray:
    if not mask.any():
        return mask
    H, W = mask.shape
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components: List[tuple[int, int]] = []
    min_area = int(round(float(min_area_ratio) * float(H * W)))
    for component_id in range(1, n):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, component_id))
    if not components:
        return np.zeros_like(mask, dtype=bool)
    components.sort(reverse=True)
    keep_ids = [component_id for _area, component_id in components[: int(keep_count)]]
    return np.isin(labels, keep_ids)


def _clean_mask(
    mask: np.ndarray,
    label: str,
    vertical: Dict[str, tuple[float, float]],
    keep_components: Dict[str, int],
    min_component_area: Dict[str, float],
    open_px: int,
    close_px: int,
) -> np.ndarray:
    out = mask.astype(bool)
    out = _apply_vertical(out, label, vertical)
    if open_px > 0:
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, _kernel(open_px)).astype(bool)
    if close_px > 0:
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(close_px)).astype(bool)
    out = _keep_components(
        out,
        int(keep_components.get(label, 2)),
        float(min_component_area.get(label, 0.002)),
    )
    return out.astype(bool)


def _write_mask(track: Dict[str, Any], frame_idx: int, mask: np.ndarray, score: float, H: int, W: int) -> None:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return
    track["mask_by_frame"][int(frame_idx)] = _pack_mask_np(mask_bool)
    track["box_by_frame"][int(frame_idx)] = torch.from_numpy(_mask_to_box_np(mask_bool))
    track["q_by_frame"][int(frame_idx)] = float(score)
    track["area_by_frame"][int(frame_idx)] = float(mask_bool.sum()) / float(max(H * W, 1))


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask.astype(bool)
    return cv2.dilate(mask.astype(np.uint8), _kernel(px), iterations=1).astype(bool)


def _fuse(
    base: Any,
    primary: Any,
    secondary: Any,
    negative_sparse: Optional[Any],
    image_paths: List[str],
    args: argparse.Namespace,
) -> tuple[Any, Dict[str, Any]]:
    labels = [canonicalize_label(label) for label in _parse_csv(args.labels)]
    red_labels = {canonicalize_label(label) for label in _parse_csv(args.red_exclusion_labels)}
    area_max = _parse_float_map(args.area_max)
    secondary_area_max = _parse_float_map(args.secondary_area_max)
    vertical = _parse_vertical_map(args.vertical_keep)
    keep_components = _parse_int_map(args.keep_components)
    min_component_area = _parse_float_map(args.min_component_area)

    output = clone_sparse(base)
    thing_tracks = [track for track in output.tracks if str(track.get("source_type")) != "stuff_static"]
    H, W = int(output.frame_height), int(output.frame_width)
    T = int(output.num_frames)
    primary_tracks = _label_tracks(primary, labels)
    secondary_tracks = _label_tracks(secondary, labels)
    negative_labels = [canonicalize_label(label) for label in _parse_csv(args.negative_mask_labels)]
    negative_tracks = _label_tracks(negative_sparse, negative_labels) if negative_sparse is not None else {}
    fused_tracks = {label: _make_sparse_stuff_track(label, H, W) for label in labels}

    debug: Dict[str, Any] = {
        "base_pt": str(args.base_pt),
        "primary_stuff_pt": str(args.primary_stuff_pt),
        "secondary_stuff_pt": str(args.secondary_stuff_pt),
        "negative_mask_pt": str(args.negative_mask_pt or ""),
        "negative_mask_labels": negative_labels,
        "negative_mask_dilate_px": int(args.negative_mask_dilate_px),
        "labels": labels,
        "red_exclusion_labels": sorted(red_labels),
        "red_sat_min": int(args.red_sat_min),
        "red_value_min": int(args.red_value_min),
        "red_value_max": int(args.red_value_max),
        "red_dilate_px": int(args.red_dilate_px),
        "red_exclude_primary": int(args.red_exclude_primary),
        "thing_dilate_px": int(args.thing_dilate_px),
        "area_max": area_max,
        "secondary_area_max": secondary_area_max,
        "vertical_keep": {label: list(value) for label, value in vertical.items()},
        "keep_components": keep_components,
        "min_component_area": min_component_area,
        "label_counts": {
            label: {
                "primary_only": 0,
                "secondary_only": 0,
                "union": 0,
                "primary_after_area_guard": 0,
                "dropped_area": 0,
                "empty": 0,
                "red_removed_pixels": 0,
                "thing_removed_pixels": 0,
                "negative_mask_removed_pixels": 0,
            }
            for label in labels
        },
        "frame_debug": [],
    }

    for frame_idx in range(T):
        thing = _dilate(build_thing_union(base, frame_idx), int(args.thing_dilate_px))
        red = _red_risk_mask(
            image_paths[frame_idx],
            H,
            W,
            int(args.red_sat_min),
            int(args.red_value_min),
            int(args.red_value_max),
            int(args.red_dilate_px),
        )
        negative_union = np.zeros((H, W), dtype=bool)
        for negative_label in negative_labels:
            negative_mask = _unpack_track_mask(negative_tracks.get(negative_label), frame_idx, H, W)
            if negative_mask is not None:
                negative_union |= negative_mask.astype(bool)
        negative_union = _dilate(negative_union, int(args.negative_mask_dilate_px))
        frame_item: Dict[str, Any] = {"frame_idx": int(frame_idx), "labels": {}}
        for label in labels:
            primary_mask = _unpack_track_mask(primary_tracks.get(label), frame_idx, H, W)
            secondary_mask = _unpack_track_mask(secondary_tracks.get(label), frame_idx, H, W)
            primary_mask = np.zeros((H, W), dtype=bool) if primary_mask is None else primary_mask.astype(bool)
            secondary_mask = np.zeros((H, W), dtype=bool) if secondary_mask is None else secondary_mask.astype(bool)

            if label in red_labels and (primary_mask.any() or secondary_mask.any()):
                removed = 0
                if int(args.red_exclude_primary) and primary_mask.any():
                    before = int(primary_mask.sum())
                    primary_mask = primary_mask & ~red
                    removed += before - int(primary_mask.sum())
                if secondary_mask.any():
                    before = int(secondary_mask.sum())
                    secondary_mask = secondary_mask & ~red
                    removed += before - int(secondary_mask.sum())
                debug["label_counts"][label]["red_removed_pixels"] += int(removed)
            if negative_union.any():
                before_primary = int(primary_mask.sum())
                before_secondary = int(secondary_mask.sum())
                primary_mask = primary_mask & ~negative_union
                secondary_mask = secondary_mask & ~negative_union
                debug["label_counts"][label]["negative_mask_removed_pixels"] += (
                    before_primary
                    + before_secondary
                    - int(primary_mask.sum())
                    - int(secondary_mask.sum())
                )
            if thing.any():
                before_primary = int(primary_mask.sum())
                before_secondary = int(secondary_mask.sum())
                primary_mask = primary_mask & ~thing
                secondary_mask = secondary_mask & ~thing
                debug["label_counts"][label]["thing_removed_pixels"] += (
                    before_primary
                    + before_secondary
                    - int(primary_mask.sum())
                    - int(secondary_mask.sum())
                )

            primary_clean = _clean_mask(
                primary_mask,
                label,
                vertical,
                keep_components,
                min_component_area,
                int(args.morph_open_px),
                int(args.morph_close_px),
            )
            secondary_clean = _clean_mask(
                secondary_mask,
                label,
                vertical,
                keep_components,
                min_component_area,
                int(args.morph_open_px),
                int(args.morph_close_px),
            )
            secondary_area = float(secondary_clean.sum()) / float(max(H * W, 1))
            if secondary_area > float(secondary_area_max.get(label, 1.0)):
                secondary_clean = np.zeros_like(secondary_clean, dtype=bool)

            if primary_clean.any() and secondary_clean.any():
                fused = primary_clean | secondary_clean
                source = "union"
            elif primary_clean.any():
                fused = primary_clean
                source = "primary_only"
            elif secondary_clean.any():
                fused = secondary_clean
                source = "secondary_only"
            else:
                fused = np.zeros((H, W), dtype=bool)
                source = "empty"

            fused = _clean_mask(
                fused,
                label,
                vertical,
                keep_components,
                min_component_area,
                int(args.morph_open_px),
                int(args.morph_close_px),
            )
            area = float(fused.sum()) / float(max(H * W, 1))
            if area > float(area_max.get(label, 1.0)):
                fused = primary_clean if primary_clean.any() else np.zeros_like(fused, dtype=bool)
                source = "primary_after_area_guard" if primary_clean.any() else "dropped_area"
                area = float(fused.sum()) / float(max(H * W, 1))
                debug["label_counts"][label]["dropped_area"] += 1

            if fused.any():
                _write_mask(fused_tracks[label], frame_idx, fused, 1.0, H, W)
            debug["label_counts"][label][source if source in debug["label_counts"][label] else "empty"] += 1
            frame_item["labels"][label] = {
                "source": source,
                "area": area,
                "primary_area": float(primary_clean.sum()) / float(max(H * W, 1)),
                "secondary_area": secondary_area,
            }
        debug["frame_debug"].append(frame_item)

    kept_stuff: List[Dict[str, Any]] = []
    dropped_short: Dict[str, int] = {}
    for label, track in fused_tracks.items():
        visible = len(track.get("mask_by_frame", {}))
        if visible < int(args.min_visible_frames):
            dropped_short[label] = visible
            continue
        kept_stuff.append(track)
    output.tracks = thing_tracks + kept_stuff
    output.num_masklets = len(output.tracks)
    debug["thing_tracks_kept"] = int(len(thing_tracks))
    debug["stuff_tracks_kept"] = int(len(kept_stuff))
    debug["dropped_short"] = dropped_short
    output.debug["fuse_sparse_stuff_candidates"] = {k: v for k, v in debug.items() if k != "frame_debug"}
    return output, debug


def _shape_check(base: Any, other: Any, name: str) -> None:
    if int(base.num_frames) != int(other.num_frames):
        raise ValueError(f"{name} frame count mismatch: base={base.num_frames}, other={other.num_frames}")
    if (int(base.frame_height), int(base.frame_width)) != (int(other.frame_height), int(other.frame_width)):
        raise ValueError(
            f"{name} shape mismatch: base={(base.frame_height, base.frame_width)}, "
            f"other={(other.frame_height, other.frame_width)}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_sparse(Path(args.base_pt))
    primary = load_sparse(Path(args.primary_stuff_pt))
    secondary = load_sparse(Path(args.secondary_stuff_pt))
    negative_sparse = load_sparse(Path(args.negative_mask_pt)) if str(args.negative_mask_pt or "").strip() else None
    _shape_check(base, primary, "primary")
    _shape_check(base, secondary, "secondary")
    if negative_sparse is not None:
        _shape_check(base, negative_sparse, "negative")
    image_paths, temp_dirs = _load_processing_frames(args, base.frame_height, base.frame_width, base.num_frames)

    fused, debug = _fuse(base, primary, secondary, negative_sparse, image_paths, args)
    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    contact_path = output_dir / "contact_before_after.jpg"
    metrics_path = output_dir / "metrics_summary.json"
    frame_debug_path = output_dir / "frame_debug.json"

    save_sparse_output(output_pt, fused)
    create_tracking_video_v2(
        image_paths,
        fused,
        str(output_video),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style="clean",
    )
    make_contact_sheet(
        image_paths,
        base,
        fused,
        parse_contact_frames(args.contact_frames, fused.num_frames),
        contact_path,
        float(args.mask_alpha),
    )

    summary = {
        "base_pt": str(args.base_pt),
        "primary_stuff_pt": str(args.primary_stuff_pt),
        "secondary_stuff_pt": str(args.secondary_stuff_pt),
        "negative_mask_pt": str(args.negative_mask_pt or ""),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "frame_debug_path": str(frame_debug_path),
        "before_base": coverage_stats(base),
        "after_fused": coverage_stats(fused),
        "delta_vs_base": {
            key: float(coverage_stats(fused)[key]) - float(coverage_stats(base)[key])
            for key in coverage_stats(base).keys()
            if key in coverage_stats(fused)
        },
        "track_stats_after": track_stats(fused),
        "fusion_debug": fused.debug.get("fuse_sparse_stuff_candidates", {}),
    }
    frame_debug_path.write_text(json.dumps(debug["frame_debug"], ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
