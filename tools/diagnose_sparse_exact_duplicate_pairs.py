#!/usr/bin/env python3
"""Diagnose exact same-label duplicate thing pairs in sparse masklets.

The duplicate criterion matches the Video Masklet Front-end v2 audit logs:
same-frame, same-label thing tracks where mask IoU is high, or bbox IoU is high
enough to indicate a possible split identity. For crowded videos, callers can
require bbox-only hits to also have mask containment evidence so occluded but
distinct people are not counted as actionable duplicates.
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import load_sparse  # noqa: E402
from run_video_masklet_front_end import _unpack_mask_np  # noqa: E402


Residual = Tuple[float, int, str, int, int, float, float, float, int, int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find exact duplicate thing pairs in sparse masklets.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--output_txt", required=True)
    parser.add_argument("--overlay_video", default="")
    parser.add_argument("--output_jpg", default="")
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--mask_iou_threshold", type=float, default=0.35)
    parser.add_argument("--box_iou_threshold", type=float, default=0.75)
    parser.add_argument("--box_min_containment", type=float, default=0.0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    return parser.parse_args()


def _as_box(track: Dict[str, Any], frame_idx: int) -> Optional[np.ndarray]:
    box = track.get("box_by_frame", {}).get(int(frame_idx))
    if box is None:
        return None
    if hasattr(box, "detach"):
        box = box.detach().cpu().numpy()
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4,):
        return None
    return box


def _box_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return float(inter / (area_a + area_b - inter + 1e-6))


def _unpack(track: Dict[str, Any], frame_idx: int, height: int, width: int) -> Optional[np.ndarray]:
    packed = track.get("mask_by_frame", {}).get(int(frame_idx))
    if packed is None:
        return None
    return _unpack_mask_np(np.asarray(packed, dtype=np.uint8), height, width).astype(bool)


def find_residuals(
    tracks: List[Dict[str, Any]],
    height: int,
    width: int,
    mask_iou_threshold: float,
    box_iou_threshold: float,
    box_min_containment: float,
) -> List[Residual]:
    by_frame_label: Dict[int, Dict[str, List[int]]] = {}
    for track_idx, track in enumerate(tracks):
        if "thing" not in str(track.get("source_type", "")).lower():
            continue
        label = str(track.get("L_sem", "")).lower()
        for frame_idx in track.get("mask_by_frame", {}).keys():
            by_frame_label.setdefault(int(frame_idx), {}).setdefault(label, []).append(track_idx)

    residuals: List[Residual] = []
    for frame_idx, label_map in by_frame_label.items():
        for label, track_indices in label_map.items():
            if len(track_indices) < 2:
                continue
            for left_idx, right_idx in combinations(sorted(track_indices), 2):
                left = tracks[left_idx]
                right = tracks[right_idx]
                left_mask = _unpack(left, frame_idx, height, width)
                right_mask = _unpack(right, frame_idx, height, width)
                if left_mask is None or right_mask is None:
                    continue
                left_area = int(left_mask.sum())
                right_area = int(right_mask.sum())
                if left_area <= 0 or right_area <= 0:
                    continue
                inter = int(np.logical_and(left_mask, right_mask).sum())
                union = int(np.logical_or(left_mask, right_mask).sum())
                mask_iou = 0.0 if union <= 0 else float(inter / float(union))
                containment = float(inter / float(min(left_area, right_area) + 1e-6))
                bbox_iou = _box_iou(_as_box(left, frame_idx), _as_box(right, frame_idx))
                mask_duplicate = mask_iou >= mask_iou_threshold
                box_duplicate = bbox_iou >= box_iou_threshold and containment >= box_min_containment
                if mask_duplicate or box_duplicate:
                    residuals.append(
                        (
                            max(mask_iou, bbox_iou),
                            int(frame_idx),
                            label,
                            int(left_idx),
                            int(right_idx),
                            mask_iou,
                            bbox_iou,
                            containment,
                            left_area,
                            right_area,
                            union,
                            inter,
                            min(left_area, right_area),
                        )
                    )

    residuals.sort(reverse=True)
    return residuals


def _draw_box(img: np.ndarray, box: Optional[np.ndarray], sx: float, sy: float, color: tuple[int, int, int], text: str) -> None:
    if box is None:
        return
    x1, y1, x2, y2 = box.astype(float)
    p1 = (int(round(x1 * sx)), int(round(y1 * sy)))
    p2 = (int(round(x2 * sx)), int(round(y2 * sy)))
    cv2.rectangle(img, p1, p2, color, 2)
    cv2.putText(img, text, (p1[0], max(18, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def write_diagnostic_image(
    residuals: List[Residual],
    tracks: List[Dict[str, Any]],
    overlay_video: Path,
    output_jpg: Path,
    sparse_height: int,
    sparse_width: int,
    top_k: int,
) -> None:
    cap = cv2.VideoCapture(str(overlay_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open overlay video: {overlay_video}")

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or sparse_width
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or sparse_height
    sx = video_width / float(sparse_width)
    sy = video_height / float(sparse_height)
    tiles: List[np.ndarray] = []

    for rank, row in enumerate(residuals[:top_k], 1):
        _score, frame_idx, label, left_idx, right_idx, mask_iou, bbox_iou, containment, *_ = row
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        _draw_box(frame, _as_box(tracks[left_idx], frame_idx), sx, sy, (0, 255, 255), str(left_idx))
        _draw_box(frame, _as_box(tracks[right_idx], frame_idx), sx, sy, (255, 0, 255), str(right_idx))
        cv2.rectangle(frame, (0, 0), (video_width, 52), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"#{rank} f={frame_idx} {label} tracks={left_idx}/{right_idx}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"mask_iou={mask_iou:.3f} box_iou={bbox_iou:.3f} contain={containment:.3f}",
            (8, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(frame)

    cap.release()
    if not tiles:
        grid = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        cv2.putText(grid, "no exact residuals", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cols = min(3, len(tiles))
        rows = int(math.ceil(len(tiles) / cols))
        blank = np.zeros_like(tiles[0])
        while len(tiles) < rows * cols:
            tiles.append(blank.copy())
        grid = np.concatenate(
            [np.concatenate(tiles[row * cols : (row + 1) * cols], axis=1) for row in range(rows)],
            axis=0,
        )

    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_jpg), grid)


def main() -> None:
    args = parse_args()
    sparse = load_sparse(Path(args.input_pt))
    residuals = find_residuals(
        sparse.tracks,
        int(sparse.frame_height),
        int(sparse.frame_width),
        float(args.mask_iou_threshold),
        float(args.box_iou_threshold),
        float(args.box_min_containment),
    )

    output_txt = Path(args.output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as fp:
        for row in residuals[: int(args.top_k)]:
            fp.write(repr(row) + "\n")

    if args.overlay_video and args.output_jpg:
        write_diagnostic_image(
            residuals=residuals,
            tracks=sparse.tracks,
            overlay_video=Path(args.overlay_video),
            output_jpg=Path(args.output_jpg),
            sparse_height=int(sparse.frame_height),
            sparse_width=int(sparse.frame_width),
            top_k=int(args.top_k),
        )

    print(f"input_pt={args.input_pt}")
    print(f"tracks={len(sparse.tracks)}")
    print(f"duplicate_count={len(residuals)}")
    print(f"output_txt={output_txt}")
    if args.output_jpg:
        print(f"output_jpg={Path(args.output_jpg)}")
    for row in residuals[: int(args.top_k)]:
        print(row)


if __name__ == "__main__":
    main()
