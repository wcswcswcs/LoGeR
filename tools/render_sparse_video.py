#!/usr/bin/env python3
"""Render sparse masklet outputs as instance or semantic overlay videos."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import _unpack_mask_np, coverage_stats, load_sparse, track_stats  # noqa: E402
from run_video_masklet_front_end import collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import create_tracking_video_v2  # noqa: E402


PALETTE: Dict[str, tuple[int, int, int]] = {
    "road": (104, 104, 104),
    "sky": (96, 180, 238),
    "grass": (74, 162, 74),
    "tree": (32, 120, 76),
    "handrail_or_fence": (56, 100, 176),
    "guardrail": (56, 100, 176),
    "pole": (220, 188, 74),
    "traffic sign": (245, 214, 58),
    "car": (220, 74, 74),
    "person": (235, 120, 60),
    "building": (160, 126, 192),
    "house": (156, 110, 168),
    "bridge": (136, 100, 84),
    "wall": (156, 156, 156),
    "ground": (130, 108, 72),
    "mountain": (104, 144, 132),
    "billboard_or_bulletin_board": (235, 122, 192),
    "other_construction": (170, 136, 86),
    "crosswalk": (240, 240, 220),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a sparse masklet file to video.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--metrics_json", default="")
    parser.add_argument("--mode", choices=["instance", "semantic"], default="instance")
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.40)
    parser.add_argument("--semantic_background", choices=["image", "black"], default="image")
    parser.add_argument("--draw_contours", type=int, default=1)
    parser.add_argument("--draw_info", type=int, default=0)
    parser.add_argument("--source_types", default="all", help="Comma-separated source_type filter, or all.")
    parser.add_argument("--labels", default="all", help="Comma-separated L_sem label filter, or all.")
    return parser.parse_args()


def _split_filter(raw: str) -> set[str] | None:
    text = str(raw or "").strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    return {part.strip() for part in text.split(",") if part.strip()}


def _filter_sparse(sparse: Any, source_types: set[str] | None, labels: set[str] | None) -> Any:
    if source_types is None and labels is None:
        return sparse
    tracks = []
    for track in sparse.tracks:
        if source_types is not None and str(track.get("source_type", "")) not in source_types:
            continue
        if labels is not None and str(track.get("L_sem", "")) not in labels:
            continue
        tracks.append(track)
    debug = dict(sparse.debug)
    debug["render_sparse_video_filter"] = {
        "input_tracks": int(len(sparse.tracks)),
        "output_tracks": int(len(tracks)),
        "source_types": sorted(source_types) if source_types is not None else "all",
        "labels": sorted(labels) if labels is not None else "all",
    }
    return type(sparse)(
        tracks=tracks,
        num_masklets=len(tracks),
        num_frames=int(sparse.num_frames),
        frame_height=int(sparse.frame_height),
        frame_width=int(sparse.frame_width),
        debug=debug,
    )


def _stable_colour(label: str) -> tuple[int, int, int]:
    label = str(label)
    if label in PALETTE:
        return PALETTE[label]
    value = 2166136261
    for byte in label.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    hue = value % 180
    hsv = np.array([[[hue, 150, 230]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _load_frames(input_video: str, processing_max_side: int, expected_h: int, expected_w: int, num_frames: int) -> tuple[List[str], List[str]]:
    image_paths, temp_dir = collect_image_paths(input_video, 0, -1, 1)
    temp_dirs = [temp_dir] if temp_dir else []
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(image_paths, int(processing_max_side))
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    if len(image_paths) < int(num_frames):
        raise RuntimeError(f"Need at least {num_frames} frames, got {len(image_paths)}")
    if tuple(proc_shape) != (int(expected_h), int(expected_w)):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    return image_paths[: int(num_frames)], temp_dirs


def _active_mask(track: Dict[str, Any], frame_idx: int, H: int, W: int) -> np.ndarray | None:
    packed = track.get("mask_by_frame", {}).get(int(frame_idx))
    if packed is None:
        packed = track.get("mask_by_frame", {}).get(str(int(frame_idx)))
    if packed is None:
        return None
    mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
    if not mask.any():
        return None
    return mask


def _render_semantic_frame(rgb: np.ndarray, sparse: Any, frame_idx: int, mask_alpha: float, background: str, draw_contours: bool, draw_info: bool) -> np.ndarray:
    H, W = int(sparse.frame_height), int(sparse.frame_width)
    out = np.zeros((H, W, 3), dtype=np.uint8) if str(background) == "black" else rgb.copy()
    order = sorted(
        range(int(sparse.num_masklets)),
        key=lambda idx: 0 if str(sparse.tracks[idx].get("source_type", "")) == "stuff_static" else 1,
    )
    for track_idx in order:
        track = sparse.tracks[track_idx]
        mask = _active_mask(track, frame_idx, H, W)
        if mask is None:
            continue
        label = str(track.get("L_sem", "unknown"))
        colour = np.asarray(_stable_colour(label), dtype=np.uint8)
        out[mask] = (
            out[mask].astype(np.float32) * (1.0 - float(mask_alpha))
            + colour.astype(np.float32) * float(mask_alpha)
        ).astype(np.uint8)
        if draw_contours:
            contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, tuple(int(x) for x in colour.tolist()), 1)
    if draw_info:
        info = f"Frame {frame_idx}/{sparse.num_frames} semantic"
        cv2.putText(out, info, (9, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(out, info, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_semantic_video(image_paths: List[str], sparse: Any, output_video: Path, fps: int, mask_alpha: float, background: str, draw_contours: bool, draw_info: bool) -> None:
    H, W = int(sparse.frame_height), int(sparse.frame_width)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (W, H))
    for frame_idx, image_path in enumerate(image_paths):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            rgb = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H))
        rendered = _render_semantic_frame(rgb, sparse, int(frame_idx), float(mask_alpha), background, draw_contours, draw_info)
        writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved semantic video to {output_video}  ({len(image_paths)} frames, {fps} FPS)")


def main() -> None:
    args = parse_args()
    sparse = load_sparse(Path(args.input_pt))
    sparse = _filter_sparse(sparse, _split_filter(str(args.source_types)), _split_filter(str(args.labels)))
    image_paths, temp_dirs = _load_frames(
        str(args.input_video),
        int(args.processing_max_side),
        int(sparse.frame_height),
        int(sparse.frame_width),
        int(sparse.num_frames),
    )
    try:
        output_video = Path(args.output_video)
        if str(args.mode) == "instance":
            create_tracking_video_v2(
                image_paths,
                sparse,
                str(output_video),
                fps=int(args.fps),
                mask_alpha=float(args.mask_alpha),
                render_style="clean",
            )
        else:
            render_semantic_video(
                image_paths,
                sparse,
                output_video,
                int(args.fps),
                float(args.mask_alpha),
                str(args.semantic_background),
                bool(int(args.draw_contours)),
                bool(int(args.draw_info)),
            )

        labels = sorted({str(track.get("L_sem", "unknown")) for track in sparse.tracks})
        metrics_path = Path(args.metrics_json) if str(args.metrics_json).strip() else output_video.with_suffix(".metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "input_pt": str(args.input_pt),
            "input_video": str(args.input_video),
            "output_video": str(output_video),
            "mode": str(args.mode),
            "num_frames": int(sparse.num_frames),
            "num_tracks": int(sparse.num_masklets),
            "labels": labels,
            "label_colours_rgb": {label: list(_stable_colour(label)) for label in labels},
            "coverage": coverage_stats(sparse),
            "track_stats": track_stats(sparse),
            "source_debug": dict(sparse.debug),
        }
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output_video": str(output_video), "metrics_json": str(metrics_path)}, ensure_ascii=False, indent=2))
    finally:
        for path in reversed(temp_dirs):
            if path:
                shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
