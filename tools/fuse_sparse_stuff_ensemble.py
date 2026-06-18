#!/usr/bin/env python3
"""Fuse multiple sparse stuff sources by per-label weighted consensus."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    parser = argparse.ArgumentParser(description="Fuse sparse stuff tracks by weighted ensemble consensus.")
    parser.add_argument("--base_pt", required=True, help="Sparse file whose non-stuff tracks are kept.")
    parser.add_argument(
        "--source_pts",
        required=True,
        help="Comma-separated source specs. Each item is path or name=path.",
    )
    parser.add_argument(
        "--source_weights",
        default="",
        help="Comma-separated weights aligned with --source_pts. Empty means all 1.0.",
    )
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--labels", default="wall,floor,ceiling,curtain")
    parser.add_argument(
        "--vote_thresholds",
        default="wall:1.5;floor:1.0;ceiling:1.0;curtain:1.5",
        help="Per-label weighted vote threshold.",
    )
    parser.add_argument("--thing_mask_pt", default="", help="Optional sparse file with thing masks to subtract.")
    parser.add_argument("--thing_dilate_px", type=int, default=5)
    parser.add_argument("--morph_open_px", type=int, default=1)
    parser.add_argument("--morph_close_px", type=int, default=2)
    parser.add_argument("--keep_components", default="wall:2;floor:1;ceiling:1;curtain:2")
    parser.add_argument("--min_component_area", default="wall:0.004;floor:0.003;ceiling:0.002;curtain:0.006")
    parser.add_argument("--area_max", default="wall:0.60;floor:0.34;ceiling:0.14;curtain:0.86")
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


def _parse_vertical_map(spec: str) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
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


def _parse_sources(spec: str, weight_spec: str) -> List[Tuple[str, Path, float]]:
    raw_sources = _parse_csv(spec)
    if not raw_sources:
        raise ValueError("--source_pts resolved to an empty source list")
    raw_weights = [float(item) for item in _parse_csv(weight_spec)]
    if raw_weights and len(raw_weights) != len(raw_sources):
        raise ValueError("--source_weights length must match --source_pts")
    sources: List[Tuple[str, Path, float]] = []
    for idx, item in enumerate(raw_sources):
        if "=" in item:
            name, path_text = item.split("=", 1)
            name = name.strip() or f"source{idx}"
        else:
            path_text = item
            name = Path(path_text).parent.name or f"source{idx}"
        weight = raw_weights[idx] if raw_weights else 1.0
        sources.append((name, Path(path_text), float(weight)))
    return sources


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


def _track_mask(track: Optional[Dict[str, Any]], frame_idx: int, H: int, W: int) -> np.ndarray:
    if track is None:
        return np.zeros((H, W), dtype=bool)
    packed = track.get("mask_by_frame", {}).get(int(frame_idx))
    if packed is None:
        return np.zeros((H, W), dtype=bool)
    return _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)


def _stuff_track_map(sparse: Any, labels: Sequence[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    wanted = {canonicalize_label(label) for label in labels}
    out: Dict[str, Optional[Dict[str, Any]]] = {label: None for label in wanted}
    for track in sparse.tracks:
        if str(track.get("source_type")) != "stuff_static":
            continue
        label = canonicalize_label(str(track.get("L_sem", "")))
        if label in wanted and out[label] is None:
            out[label] = track
    return out


def _apply_vertical(mask: np.ndarray, label: str, vertical: Dict[str, Tuple[float, float]]) -> np.ndarray:
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
    min_area = int(round(float(min_area_ratio) * float(H * W)))
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


def _clean_mask(
    mask: np.ndarray,
    label: str,
    vertical: Dict[str, Tuple[float, float]],
    keep_components: Dict[str, int],
    min_component_area: Dict[str, float],
    open_px: int,
    close_px: int,
) -> np.ndarray:
    out = mask.astype(bool)
    out = _apply_vertical(out, label, vertical)
    if int(open_px) > 0:
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, _kernel(open_px)).astype(bool)
    if int(close_px) > 0:
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(close_px)).astype(bool)
    out = _keep_components(
        out,
        keep_components.get(label, 1),
        min_component_area.get(label, 0.0),
    )
    return out.astype(bool)


def _write_track_frame(track: Dict[str, Any], frame_idx: int, mask: np.ndarray, score: float, H: int, W: int) -> None:
    mask_bool = np.asarray(mask).astype(bool)
    if not mask_bool.any():
        return
    track["mask_by_frame"][int(frame_idx)] = _pack_mask_np(mask_bool)
    track["box_by_frame"][int(frame_idx)] = torch.from_numpy(_mask_to_box_np(mask_bool))
    track["q_by_frame"][int(frame_idx)] = float(score)
    track["area_by_frame"][int(frame_idx)] = float(mask_bool.sum()) / float(max(H * W, 1))
    if len(track["mask_by_frame"]) == 1:
        track["birth_frame"] = int(frame_idx)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [canonicalize_label(label) for label in _parse_csv(args.labels)]
    labels = [label for idx, label in enumerate(labels) if label and label not in labels[:idx]]
    if not labels:
        raise ValueError("--labels resolved to an empty label list")

    base = load_sparse(Path(args.base_pt))
    sources = _parse_sources(args.source_pts, args.source_weights)
    loaded_sources = [(name, load_sparse(path), weight, str(path)) for name, path, weight in sources]
    H, W = int(base.frame_height), int(base.frame_width)
    num_frames = int(base.num_frames)
    for name, sparse, _weight, path in loaded_sources:
        if int(sparse.frame_height) != H or int(sparse.frame_width) != W:
            raise RuntimeError(f"Source {name} shape mismatch: {path}")
        if int(sparse.num_frames) < num_frames:
            raise RuntimeError(f"Source {name} has only {sparse.num_frames} frames, need {num_frames}: {path}")

    output = clone_sparse(base)
    output.tracks = [track for track in output.tracks if str(track.get("source_type")) != "stuff_static"]
    fused_tracks = {label: _make_sparse_stuff_track(label, H, W) for label in labels}
    source_maps = [(name, _stuff_track_map(sparse, labels), weight) for name, sparse, weight, _path in loaded_sources]
    thresholds = _parse_float_map(args.vote_thresholds)
    vertical = _parse_vertical_map(args.vertical_keep)
    keep_components = _parse_int_map(args.keep_components)
    min_component_area = _parse_float_map(args.min_component_area)
    area_max = _parse_float_map(args.area_max)
    thing_sparse = load_sparse(Path(args.thing_mask_pt)) if str(args.thing_mask_pt or "").strip() else base

    image_paths, temp_dirs = _load_processing_frames(args, H, W, num_frames)
    dilate_kernel = _kernel(int(args.thing_dilate_px))
    debug: Dict[str, Any] = {
        "format": "sparse_stuff_ensemble_v1",
        "base_pt": str(args.base_pt),
        "sources": [
            {"name": name, "path": path, "weight": float(weight)}
            for name, _sparse, weight, path in loaded_sources
        ],
        "labels": labels,
        "vote_thresholds": thresholds,
        "label_counts": {label: {"kept": 0, "empty": 0, "area_dropped": 0} for label in labels},
        "support_hist": {
            label: {name: 0 for name, _maps, _weight in source_maps}
            for label in labels
        },
        "thing_removed_pixels": {label: 0 for label in labels},
    }

    for frame_idx in range(num_frames):
        thing_mask = build_thing_union(thing_sparse, frame_idx)
        if int(args.thing_dilate_px) > 0 and thing_mask.any():
            thing_mask = cv2.dilate(thing_mask.astype(np.uint8), dilate_kernel, iterations=1).astype(bool)
        for label in labels:
            vote = np.zeros((H, W), dtype=np.float32)
            supporting_sources = 0
            for source_name, tracks_by_label, weight in source_maps:
                mask = _track_mask(tracks_by_label.get(label), frame_idx, H, W)
                if mask.any():
                    debug["support_hist"][label][source_name] += 1
                    supporting_sources += 1
                    vote[mask] += float(weight)
            threshold = float(thresholds.get(label, 1.0))
            mask = vote >= threshold
            if not mask.any():
                debug["label_counts"][label]["empty"] += 1
                continue
            before_thing = int(mask.sum())
            mask = mask & ~thing_mask
            debug["thing_removed_pixels"][label] += int(before_thing - int(mask.sum()))
            mask = _clean_mask(
                mask,
                label,
                vertical,
                keep_components,
                min_component_area,
                int(args.morph_open_px),
                int(args.morph_close_px),
            )
            if not mask.any():
                debug["label_counts"][label]["empty"] += 1
                continue
            area_ratio = float(mask.sum()) / float(max(H * W, 1))
            if area_ratio > float(area_max.get(label, 1.0)):
                debug["label_counts"][label]["area_dropped"] += 1
                continue
            score = float(vote[mask].mean()) if mask.any() else float(supporting_sources)
            _write_track_frame(fused_tracks[label], frame_idx, mask, score, H, W)
            debug["label_counts"][label]["kept"] += 1

    kept_stuff_tracks: List[Dict[str, Any]] = []
    dropped_short: Dict[str, int] = {}
    for label, track in fused_tracks.items():
        visible = len(track.get("mask_by_frame", {}))
        if visible < int(args.min_visible_frames):
            dropped_short[label] = int(visible)
            continue
        kept_stuff_tracks.append(track)
    output.tracks.extend(kept_stuff_tracks)
    output.num_masklets = len(output.tracks)
    output.debug["sparse_stuff_ensemble"] = debug
    debug["dropped_short"] = dropped_short
    debug["stuff_tracks_kept"] = int(len(kept_stuff_tracks))

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
        parse_contact_frames(args.contact_frames, num_frames),
        contact_path,
        float(args.mask_alpha),
    )

    before_stats = coverage_stats(base)
    after_stats = coverage_stats(output)
    summary = {
        "base_pt": str(args.base_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "before_base": before_stats,
        "after_ensemble": after_stats,
        "delta_vs_base": {
            key: float(after_stats[key]) - float(before_stats[key])
            for key in before_stats
            if key in after_stats
        },
        "track_stats_after": track_stats(output),
        "ensemble_debug": debug,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for temp_dir in reversed(temp_dirs):
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
