#!/usr/bin/env python3
"""Render consecutive 16-view sparse ID audit sheets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import _unpack_mask_np, load_sparse  # noqa: E402
from run_video_masklet_front_end import collect_image_paths, get_colour, prepare_processing_image_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 16-view sparse track-id audit sheets.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--labels", default="car")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1, help="Inclusive global end frame; -1 means sparse end.")
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--step", type=int, default=16)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--mask_alpha", type=float, default=0.42)
    parser.add_argument("--only_things", type=int, default=1)
    return parser.parse_args()


def _label_set(raw: str) -> set[str] | None:
    text = str(raw or "").strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def _as_box(value: Any, mask: np.ndarray) -> tuple[int, int, int, int]:
    if value is not None:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= 4:
            return tuple(int(round(float(x))) for x in arr[:4])
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _source_tag(track: Dict[str, Any]) -> str:
    source = str(track.get("tracking_source", track.get("mask_source", "")))
    if "detector_only" in source:
        return "D"
    if "vos" in source:
        return "V"
    if str(track.get("source_type", "")) == "stuff_static":
        return "S"
    return "?"


def _active_tracks(sparse: Any, frame_idx: int, labels: set[str] | None, only_things: bool) -> List[tuple[int, Dict[str, Any], np.ndarray]]:
    H, W = int(sparse.frame_height), int(sparse.frame_width)
    out: List[tuple[int, Dict[str, Any], np.ndarray]] = []
    for track_idx, track in enumerate(sparse.tracks):
        if only_things and str(track.get("source_type", "")) == "stuff_static":
            continue
        if labels is not None and str(track.get("L_sem", "")) not in labels:
            continue
        packed = track.get("mask_by_frame", {}).get(int(frame_idx))
        if packed is None:
            packed = track.get("mask_by_frame", {}).get(str(int(frame_idx)))
        if packed is None:
            continue
        mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
        if mask.any():
            out.append((track_idx, track, mask))
    return out


def _draw_frame(
    rgb: np.ndarray,
    sparse: Any,
    frame_idx: int,
    labels: set[str] | None,
    only_things: bool,
    mask_alpha: float,
) -> np.ndarray:
    out = rgb.copy()
    H, W = out.shape[:2]
    active = _active_tracks(sparse, frame_idx, labels, only_things)
    for track_idx, track, mask in active:
        colour = np.asarray(get_colour(track_idx), dtype=np.uint8)
        out[mask] = (
            out[mask].astype(np.float32) * (1.0 - float(mask_alpha))
            + colour.astype(np.float32) * float(mask_alpha)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(x) for x in colour.tolist()), 2)

    for track_idx, track, mask in active:
        x1, y1, x2, y2 = _as_box(track.get("box_by_frame", {}).get(int(frame_idx)), mask)
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        colour = tuple(int(x) for x in np.asarray(get_colour(track_idx), dtype=np.uint8).tolist())
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)
        label = f"{track_idx}{_source_tag(track)}"
        tx = max(1, min(W - 34, x1 + 1))
        ty = max(9, min(H - 2, y1 - 2))
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

    info = f"f={frame_idx} active={len(active)}"
    cv2.rectangle(out, (0, 0), (min(W - 1, 116), 14), (0, 0, 0), -1)
    cv2.putText(out, info, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _write_sheet(frames: Sequence[np.ndarray], cols: int, output_path: Path) -> None:
    cols = max(1, int(cols))
    rows = int(np.ceil(len(frames) / float(cols)))
    H, W = frames[0].shape[:2]
    sheet = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r, c = divmod(idx, cols)
        sheet[r * H : (r + 1) * H, c * W : (c + 1) * W] = frame
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    args = parse_args()
    sparse = load_sparse(Path(args.input_pt))
    labels = _label_set(str(args.labels))
    image_paths, temp_dir = collect_image_paths(args.input_video, 0, -1, 1)
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(
        image_paths,
        int(args.processing_max_side),
    )
    try:
        if tuple(proc_shape) != (int(sparse.frame_height), int(sparse.frame_width)):
            raise RuntimeError(f"Frame shape {proc_shape} != sparse shape {(sparse.frame_height, sparse.frame_width)}")
        max_frame = min(int(sparse.num_frames), len(image_paths)) - 1
        start = max(0, int(args.start_frame))
        end = max_frame if int(args.end_frame) < 0 else min(max_frame, int(args.end_frame))
        group_size = max(1, int(args.group_size))
        step = max(1, int(args.step))
        output_dir = Path(args.output_dir)
        rows: List[Dict[str, Any]] = []
        for group_start in range(start, end + 1, step):
            group_end = min(end, group_start + group_size - 1)
            rendered: List[np.ndarray] = []
            active_union: set[int] = set()
            for frame_idx in range(group_start, group_end + 1):
                bgr = cv2.imread(str(image_paths[frame_idx]), cv2.IMREAD_COLOR)
                if bgr is None:
                    rgb = np.zeros((int(sparse.frame_height), int(sparse.frame_width), 3), dtype=np.uint8)
                else:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rendered_frame = _draw_frame(
                    rgb,
                    sparse,
                    int(frame_idx),
                    labels,
                    bool(int(args.only_things)),
                    float(args.mask_alpha),
                )
                rendered.append(rendered_frame)
                for track_idx, _track, _mask in _active_tracks(sparse, frame_idx, labels, bool(int(args.only_things))):
                    active_union.add(int(track_idx))
            if rendered:
                name = f"car_id_16view_{group_start:04d}_{group_end:04d}.jpg"
                _write_sheet(rendered, int(args.cols), output_dir / name)
                rows.append({"file": name, "start": group_start, "end": group_end, "active_track_indices": sorted(active_union)})
        (output_dir / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir), "sheets": len(rows)}, indent=2))
    finally:
        for path in [temp_dir, resize_tmp]:
            if path:
                shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
