#!/usr/bin/env python3
"""Render selected sparse masklet frames with visible track ids."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import _unpack_mask_np, load_sparse  # noqa: E402
from run_video_masklet_front_end import collect_image_paths, is_video, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import get_colour  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render sparse masklet ID contact sheet.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_jpg", required=True)
    parser.add_argument("--frames", required=True, help="Comma-separated frame indices.")
    parser.add_argument("--start_frame", type=int, default=0, help="Source-frame offset for local sparse frames.")
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--only_things", type=int, default=1)
    parser.add_argument("--labels", default="all", help="Comma-separated labels to render, or all.")
    return parser.parse_args()


def parse_frames(raw: str, num_frames: int, start_frame: int = 0) -> List[int]:
    frames: List[int] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        frame_idx = int(item)
        local_idx = frame_idx
        if int(start_frame) <= frame_idx < int(start_frame) + int(num_frames):
            local_idx = frame_idx - int(start_frame)
        if 0 <= local_idx < num_frames:
            frames.append(int(local_idx))
    return frames


def as_box(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return None
    return arr[:4]


def label_filter(raw: str) -> set[str] | None:
    text = str(raw or "").strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def draw_frame(
    rgb: np.ndarray,
    sparse: Any,
    frame_idx: int,
    mask_alpha: float,
    only_things: bool,
    labels: set[str] | None,
    display_frame_idx: int | None = None,
) -> np.ndarray:
    H, W = int(sparse.frame_height), int(sparse.frame_width)
    out = rgb.copy()
    active = []
    for track_idx, track in enumerate(sparse.tracks):
        source_type = str(track.get("source_type", ""))
        if only_things and source_type != "thing_tracked":
            continue
        if labels is not None and str(track.get("L_sem", "")) not in labels:
            continue
        packed = track.get("mask_by_frame", {}).get(int(frame_idx))
        if packed is None:
            continue
        mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
        if not mask.any():
            continue
        active.append((track_idx, track, mask))

    for track_idx, track, mask in active:
        colour = np.asarray(get_colour(track_idx), dtype=np.uint8)
        out[mask] = (out[mask].astype(np.float32) * (1.0 - mask_alpha) + colour.astype(np.float32) * mask_alpha).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(x) for x in colour.tolist()), 2)

    for track_idx, track, _mask in active:
        box = as_box(track.get("box_by_frame", {}).get(int(frame_idx)))
        if box is None:
            ys, xs = np.where(_mask)
            if len(xs) == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        else:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        source = str(track.get("tracking_source", track.get("mask_source", "")))
        source_tag = "D" if "detector_only" in source else "V" if "vos" in source else "?"
        label = f"{track.get('L_sem', '')}#{track_idx}{source_tag}"
        text_x = max(2, min(W - 80, x1))
        text_y = max(12, min(H - 4, y1 - 4))
        cv2.rectangle(out, (text_x - 1, text_y - 11), (min(W - 1, text_x + 78), min(H - 1, text_y + 3)), (0, 0, 0), -1)
        cv2.putText(out, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    if display_frame_idx is None or int(display_frame_idx) == int(frame_idx):
        info = f"Frame {frame_idx}  active things: {len(active)}"
    else:
        info = f"Frame {int(display_frame_idx)} (local {frame_idx})  active things: {len(active)}"
    cv2.putText(out, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    sparse = load_sparse(Path(args.input_pt))
    labels = label_filter(str(args.labels))
    start_frame = int(args.start_frame)
    sparse_num_frames = max(0, int(sparse.num_frames))
    end_frame = -1
    if sparse_num_frames > 0:
        # collect_image_paths uses an inclusive end for videos but Python-slice
        # exclusive end for image directories. Request exactly the sparse span.
        end_frame = start_frame + sparse_num_frames - 1 if is_video(str(args.input_video)) else start_frame + sparse_num_frames
    image_paths, temp_dir = collect_image_paths(args.input_video, start_frame, end_frame, 1)
    image_paths, resize_tmp, _orig_shape, _proc_shape = prepare_processing_image_paths(image_paths, int(args.processing_max_side))
    frames = parse_frames(args.frames, min(int(sparse.num_frames), len(image_paths)), start_frame)
    if not frames:
        raise ValueError(
            f"no valid frames requested; sparse_num_frames={int(sparse.num_frames)} available_images={len(image_paths)}"
        )
    try:
        rendered = []
        for frame_idx in frames:
            bgr = cv2.imread(str(image_paths[frame_idx]), cv2.IMREAD_COLOR)
            if bgr is None:
                rgb = np.zeros((int(sparse.frame_height), int(sparse.frame_width), 3), dtype=np.uint8)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (int(sparse.frame_height), int(sparse.frame_width)):
                    rgb = cv2.resize(rgb, (int(sparse.frame_width), int(sparse.frame_height)))
            rendered.append(
                draw_frame(
                    rgb,
                    sparse,
                    frame_idx,
                    float(args.mask_alpha),
                    bool(int(args.only_things)),
                    labels,
                    display_frame_idx=start_frame + int(frame_idx),
                )
            )

        cols = max(1, int(args.cols))
        rows = int(np.ceil(len(rendered) / cols))
        H, W = rendered[0].shape[:2]
        sheet = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
        for idx, frame in enumerate(rendered):
            r, c = divmod(idx, cols)
            sheet[r * H : (r + 1) * H, c * W : (c + 1) * W] = frame
        output = Path(args.output_jpg)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"Saved {output} ({len(frames)} frames)")
    finally:
        for path in [temp_dir, resize_tmp]:
            if path:
                import shutil

                shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
