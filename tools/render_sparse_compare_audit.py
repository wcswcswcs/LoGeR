#!/usr/bin/env python3
"""Render v2/v3 sparse-masklet comparison sheets for visual audit.

The standard review renderer is useful for quick videos, but its large labels
cover small KITTI objects. This audit renderer keeps labels optional and tiny,
and also writes per-frame compare images so reviewers can inspect exact frames.
"""

from __future__ import annotations

import argparse
import json
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

from refine_sparse_stuff_masks import _unpack_mask_np, load_sparse  # noqa: E402
from run_video_masklet_front_end import collect_image_paths, get_colour, prepare_processing_image_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render low-occlusion sparse-masklet comparison audit images.")
    parser.add_argument("--input_video", required=True, help="Image directory or video used to create both sparse files.")
    parser.add_argument("--left_pt", required=True)
    parser.add_argument("--left_name", default="left")
    parser.add_argument("--right_pt", required=True)
    parser.add_argument("--right_name", default="right")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument(
        "--windows",
        required=True,
        help="Comma-separated name:start-end specs, e.g. win640:640-645,tail:1090-1100. Ends are inclusive.",
    )
    parser.add_argument("--modes", default="outline,tiny_id", help="Comma-separated: outline,tiny_id.")
    parser.add_argument("--include_stuff", type=int, default=1)
    parser.add_argument("--thing_alpha", type=float, default=0.24)
    parser.add_argument("--stuff_alpha", type=float, default=0.08)
    parser.add_argument("--contour_thickness", type=int, default=1)
    parser.add_argument("--tiny_label_scale", type=float, default=0.26)
    parser.add_argument("--max_contact_frames", type=int, default=14)
    parser.add_argument("--contact_cols", type=int, default=7)
    parser.add_argument("--write_all_frame_pairs", type=int, default=1)
    parser.add_argument("--write_zoom_contacts", type=int, default=1)
    parser.add_argument("--zoom_scale", type=float, default=2.0)
    parser.add_argument("--zoom_pad", type=int, default=28)
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def parse_windows(raw: str) -> List[Tuple[str, List[int]]]:
    windows: List[Tuple[str, List[int]]] = []
    for spec in _parse_csv(raw):
        if ":" not in spec or "-" not in spec:
            raise ValueError(f"Bad window spec: {spec!r}")
        name, span = spec.split(":", 1)
        start_s, end_s = span.split("-", 1)
        start, end = int(start_s), int(end_s)
        if end < start:
            raise ValueError(f"Bad descending window spec: {spec!r}")
        windows.append((name.strip(), list(range(start, end + 1))))
    return windows


def _as_box(value: Any, mask: np.ndarray) -> Tuple[int, int, int, int]:
    if value is not None:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in arr[:4]]
            return x1, y1, x2, y2
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _active_tracks(sparse: Any, frame_idx: int, include_stuff: bool) -> List[Tuple[int, Dict[str, Any], np.ndarray]]:
    H, W = int(sparse.frame_height), int(sparse.frame_width)
    active: List[Tuple[int, Dict[str, Any], np.ndarray]] = []
    for track_idx, track in enumerate(sparse.tracks):
        source_type = str(track.get("source_type", ""))
        if source_type == "stuff_static" and not include_stuff:
            continue
        packed = track.get("mask_by_frame", {}).get(int(frame_idx))
        if packed is None:
            packed = track.get("mask_by_frame", {}).get(str(int(frame_idx)))
        if packed is None:
            continue
        mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
        if mask.any():
            active.append((track_idx, track, mask))
    return active


def draw_frame(
    rgb: np.ndarray,
    sparse: Any,
    frame_idx: int,
    mode: str,
    include_stuff: bool,
    thing_alpha: float,
    stuff_alpha: float,
    contour_thickness: int,
    tiny_label_scale: float,
) -> np.ndarray:
    out = rgb.copy()
    active = _active_tracks(sparse, frame_idx, include_stuff)

    def rank(item: Tuple[int, Dict[str, Any], np.ndarray]) -> int:
        return 0 if str(item[1].get("source_type", "")) == "stuff_static" else 1

    for track_idx, track, mask in sorted(active, key=rank):
        source_type = str(track.get("source_type", ""))
        colour = np.asarray(get_colour(int(track_idx)), dtype=np.uint8)
        alpha = float(stuff_alpha if source_type == "stuff_static" else thing_alpha)
        if alpha > 0.0:
            out[mask] = (
                out[mask].astype(np.float32) * (1.0 - alpha)
                + colour.astype(np.float32) * alpha
            ).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = max(1, int(contour_thickness))
        cv2.drawContours(out, contours, -1, tuple(int(x) for x in colour.tolist()), thickness)

    if mode == "tiny_id":
        H, W = out.shape[:2]
        for track_idx, track, mask in active:
            if str(track.get("source_type", "")) == "stuff_static":
                continue
            x1, y1, x2, y2 = _as_box(track.get("box_by_frame", {}).get(int(frame_idx)), mask)
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))
            colour = tuple(int(x) for x in np.asarray(get_colour(int(track_idx)), dtype=np.uint8).tolist())
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)
            text = f"{track_idx}"
            text_y = max(8, y1 - 2)
            cv2.putText(out, text, (x1 + 1, text_y), cv2.FONT_HERSHEY_SIMPLEX, tiny_label_scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, text, (x1 + 1, text_y), cv2.FONT_HERSHEY_SIMPLEX, tiny_label_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def add_header(rgb: np.ndarray, title: str) -> np.ndarray:
    H, W = rgb.shape[:2]
    header = np.zeros((24, W, 3), dtype=np.uint8)
    header[:] = (18, 18, 18)
    cv2.putText(header, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([header, rgb])


def make_pair(left: np.ndarray, right: np.ndarray, left_title: str, right_title: str) -> np.ndarray:
    return np.vstack([add_header(left, left_title), add_header(right, right_title)])


def sample_frames(frames: Sequence[int], max_count: int) -> List[int]:
    frames = [int(x) for x in frames]
    if len(frames) <= max_count:
        return frames
    idxs = np.linspace(0, len(frames) - 1, int(max_count)).round().astype(int).tolist()
    out: List[int] = []
    for idx in idxs:
        frame = frames[int(idx)]
        if frame not in out:
            out.append(frame)
    return out


def tile_images(images: Sequence[np.ndarray], cols: int) -> np.ndarray:
    if not images:
        raise ValueError("No images to tile")
    cols = max(1, int(cols))
    rows = int(np.ceil(len(images) / cols))
    H, W = images[0].shape[:2]
    sheet = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
    for idx, image in enumerate(images):
        if image.shape[:2] != (H, W):
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        r, c = divmod(idx, cols)
        sheet[r * H : (r + 1) * H, c * W : (c + 1) * W] = image
    return sheet


def union_thing_box(left_sparse: Any, right_sparse: Any, frame_idx: int, include_stuff: bool, pad: int) -> Tuple[int, int, int, int]:
    boxes: List[Tuple[int, int, int, int]] = []
    H, W = int(left_sparse.frame_height), int(left_sparse.frame_width)
    for sparse in (left_sparse, right_sparse):
        for _track_idx, track, mask in _active_tracks(sparse, frame_idx, include_stuff=False):
            boxes.append(_as_box(track.get("box_by_frame", {}).get(int(frame_idx)), mask))
    if not boxes:
        return 0, 0, W - 1, H - 1
    x1 = max(0, min(box[0] for box in boxes) - int(pad))
    y1 = max(0, min(box[1] for box in boxes) - int(pad))
    x2 = min(W - 1, max(box[2] for box in boxes) + int(pad))
    y2 = min(H - 1, max(box[3] for box in boxes) + int(pad))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, W - 1, H - 1
    return x1, y1, x2, y2


def crop_zoom(pair: np.ndarray, frame_h_with_header: int, crop: Tuple[int, int, int, int], scale: float) -> np.ndarray:
    x1, y1, x2, y2 = crop
    header_h = 24
    chunks = []
    for row_start in (0, frame_h_with_header):
        title = pair[row_start : row_start + header_h]
        image = pair[row_start + header_h : row_start + frame_h_with_header]
        crop_img = image[y1 : y2 + 1, x1 : x2 + 1]
        if scale != 1.0:
            crop_img = cv2.resize(crop_img, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_NEAREST)
        title = cv2.resize(title, (crop_img.shape[1], header_h), interpolation=cv2.INTER_NEAREST)
        chunks.append(np.vstack([title, crop_img]))
    return np.vstack(chunks)


def main() -> None:
    args = parse_args()
    left = load_sparse(Path(args.left_pt))
    right = load_sparse(Path(args.right_pt))
    if (left.frame_height, left.frame_width, left.num_frames) != (right.frame_height, right.frame_width, right.num_frames):
        raise RuntimeError(
            "Sparse shape mismatch: "
            f"left={(left.frame_height, left.frame_width, left.num_frames)} "
            f"right={(right.frame_height, right.frame_width, right.num_frames)}"
        )
    image_paths, temp_dir = collect_image_paths(args.input_video, 0, -1, 1)
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(image_paths, int(args.processing_max_side))
    if tuple(proc_shape) != (int(left.frame_height), int(left.frame_width)):
        raise RuntimeError(f"Processed frame shape {proc_shape} != sparse shape {(left.frame_height, left.frame_width)}")
    image_paths = image_paths[: int(left.num_frames)]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = parse_windows(args.windows)
    modes = _parse_csv(args.modes)
    try:
        manifest: Dict[str, Any] = {
            "input_video": str(args.input_video),
            "left_pt": str(args.left_pt),
            "left_name": str(args.left_name),
            "right_pt": str(args.right_pt),
            "right_name": str(args.right_name),
            "processing_max_side": int(args.processing_max_side),
            "include_stuff": bool(int(args.include_stuff)),
            "thing_alpha": float(args.thing_alpha),
            "stuff_alpha": float(args.stuff_alpha),
            "windows": {name: [int(frames[0]), int(frames[-1])] for name, frames in windows},
            "modes": modes,
            "outputs": [],
        }
        for mode in modes:
            if mode not in {"outline", "tiny_id"}:
                raise ValueError(f"Unsupported mode: {mode}")
            mode_dir = out_dir / mode
            (mode_dir / "frames").mkdir(parents=True, exist_ok=True)
            for window_name, requested_frames in windows:
                frames = [f for f in requested_frames if 0 <= int(f) < len(image_paths)]
                if not frames:
                    continue
                pair_by_frame: Dict[int, np.ndarray] = {}
                frame_dir = mode_dir / "frames" / window_name
                frame_dir.mkdir(parents=True, exist_ok=True)
                for frame_idx in frames:
                    bgr = cv2.imread(str(image_paths[frame_idx]), cv2.IMREAD_COLOR)
                    if bgr is None:
                        rgb = np.zeros((int(left.frame_height), int(left.frame_width), 3), dtype=np.uint8)
                    else:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        if rgb.shape[:2] != (int(left.frame_height), int(left.frame_width)):
                            rgb = cv2.resize(rgb, (int(left.frame_width), int(left.frame_height)))
                    left_img = draw_frame(
                        rgb,
                        left,
                        int(frame_idx),
                        mode,
                        bool(int(args.include_stuff)),
                        float(args.thing_alpha),
                        float(args.stuff_alpha),
                        int(args.contour_thickness),
                        float(args.tiny_label_scale),
                    )
                    right_img = draw_frame(
                        rgb,
                        right,
                        int(frame_idx),
                        mode,
                        bool(int(args.include_stuff)),
                        float(args.thing_alpha),
                        float(args.stuff_alpha),
                        int(args.contour_thickness),
                        float(args.tiny_label_scale),
                    )
                    pair = make_pair(
                        left_img,
                        right_img,
                        f"{args.left_name} frame {frame_idx}",
                        f"{args.right_name} frame {frame_idx}",
                    )
                    pair_by_frame[int(frame_idx)] = pair
                    if bool(int(args.write_all_frame_pairs)):
                        frame_path = frame_dir / f"frame_{frame_idx:06d}.jpg"
                        cv2.imwrite(str(frame_path), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
                sampled = sample_frames(frames, int(args.max_contact_frames))
                contact = tile_images([pair_by_frame[int(f)] for f in sampled], int(args.contact_cols))
                contact_path = mode_dir / f"{window_name}_contact.jpg"
                cv2.imwrite(str(contact_path), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))
                manifest["outputs"].append(str(contact_path))
                if bool(int(args.write_zoom_contacts)):
                    zooms = []
                    frame_h_with_header = int(left.frame_height) + 24
                    for frame_idx in sampled:
                        crop = union_thing_box(left, right, int(frame_idx), bool(int(args.include_stuff)), int(args.zoom_pad))
                        zooms.append(crop_zoom(pair_by_frame[int(frame_idx)], frame_h_with_header, crop, float(args.zoom_scale)))
                    zoom_contact = tile_images(zooms, int(args.contact_cols))
                    zoom_path = mode_dir / f"{window_name}_zoom_contact.jpg"
                    cv2.imwrite(str(zoom_path), cv2.cvtColor(zoom_contact, cv2.COLOR_RGB2BGR))
                    manifest["outputs"].append(str(zoom_path))
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        for path in [temp_dir, resize_tmp]:
            if path:
                shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
