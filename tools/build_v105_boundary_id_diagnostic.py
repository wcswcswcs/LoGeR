#!/usr/bin/env python3
"""Build a four-panel ID diagnostic sheet for v105 boundary audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _read_label(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(str(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int32)


def _parse_ids(value: str) -> list[int]:
    ids: list[int] = []
    for part in str(value).split(","):
        item = part.strip()
        if item:
            ids.append(int(item))
    return ids


def _color_for_id(label_id: int) -> tuple[int, int, int]:
    if label_id == 0:
        return (28, 28, 28)
    x = (int(label_id) * 2654435761) & 0xFFFFFFFF
    return (
        int(70 + (x & 127)),
        int(70 + ((x >> 8) & 127)),
        int(70 + ((x >> 16) & 127)),
    )


def _special_color(label_id: int) -> tuple[int, int, int] | None:
    palette = [
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 255),
        (255, 180, 0),
        (0, 180, 255),
        (180, 255, 0),
        (255, 255, 255),
    ]
    if label_id == 0:
        return None
    return palette[abs(int(label_id)) % len(palette)]


def _render_label(label: np.ndarray, title: str, highlight_ids: set[int]) -> np.ndarray:
    h, w = label.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = (28, 28, 28)
    for raw_id in np.unique(label):
        label_id = int(raw_id)
        if label_id == 0:
            continue
        color = _color_for_id(label_id)
        if label_id in highlight_ids:
            color = _special_color(label_id) or color
        image[label == label_id] = color

    out = image.copy()
    cv2.putText(out, title, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    for raw_id in sorted(int(x) for x in np.unique(label) if int(x) != 0):
        if raw_id not in highlight_ids:
            continue
        ys, xs = np.where(label == raw_id)
        if len(xs) == 0:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 2)
        pos = (max(5, cx - 15), max(55, cy))
        cv2.putText(out, str(raw_id), pos, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, str(raw_id), pos, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _label_stats(local_label: np.ndarray, global_label: np.ndarray, local_ids: list[int]) -> list[dict[str, object]]:
    stats: list[dict[str, object]] = []
    for local_id in local_ids:
        ys, xs = np.where(local_label == int(local_id))
        if len(xs) == 0:
            stats.append({"local_id": int(local_id), "present": False})
            continue
        global_vals, counts = np.unique(global_label[local_label == int(local_id)], return_counts=True)
        overlap = sorted(
            [{"global_id": int(v), "pixel_count": int(c)} for v, c in zip(global_vals, counts)],
            key=lambda item: -int(item["pixel_count"]),
        )
        stats.append(
            {
                "local_id": int(local_id),
                "present": True,
                "area": int(len(xs)),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "global_overlap_top10": overlap[:10],
            }
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-mask-dir", required=True, type=Path)
    parser.add_argument("--global-mask-dir", required=True, type=Path)
    parser.add_argument("--prev-frame", required=True, type=int)
    parser.add_argument("--curr-frame", required=True, type=int)
    parser.add_argument("--candidate-local-ids", required=True)
    parser.add_argument("--highlight-global-ids", required=True)
    parser.add_argument("--focus-global-id", required=True, type=int)
    parser.add_argument("--output-image", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    candidate_ids = _parse_ids(args.candidate_local_ids)
    highlight_global_ids = set(_parse_ids(args.highlight_global_ids))
    prev_global = _read_label(args.global_mask_dir / f"{args.prev_frame}.png")
    curr_local = _read_label(args.source_mask_dir / f"{args.curr_frame}.png")
    curr_global = _read_label(args.global_mask_dir / f"{args.curr_frame}.png")

    focus = np.zeros_like(curr_global)
    focus[curr_global == int(args.focus_global_id)] = int(args.focus_global_id)

    panels = [
        _render_label(prev_global, f"prev global f{args.prev_frame:04d}", highlight_global_ids),
        _render_label(curr_local, f"curr source local f{args.curr_frame:04d}", set(candidate_ids)),
        _render_label(curr_global, f"curr global f{args.curr_frame:04d}", highlight_global_ids | {int(args.focus_global_id)}),
        _render_label(focus, f"curr global{args.focus_global_id} mask only", {int(args.focus_global_id)}),
    ]
    canvas = np.concatenate(
        [np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)],
        axis=0,
    )
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_image), canvas)

    record = {
        "schema_version": "stream4d_v105_boundary_id_diagnostic_v1",
        "scene_id": args.scene_id,
        "source_mask_dir": str(args.source_mask_dir),
        "global_mask_dir": str(args.global_mask_dir),
        "prev_frame": int(args.prev_frame),
        "curr_frame": int(args.curr_frame),
        "candidate_local_ids": candidate_ids,
        "highlight_global_ids": sorted(highlight_global_ids),
        "focus_global_id": int(args.focus_global_id),
        "candidate_local_stats": _label_stats(curr_local, curr_global, candidate_ids),
        "focus_global_area_curr": int((curr_global == int(args.focus_global_id)).sum()),
        "output_image": str(args.output_image),
    }
    args.output_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"output_image_sha256={hashlib.sha256(args.output_image.read_bytes()).hexdigest()}")
    print(f"output_json_sha256={hashlib.sha256(args.output_json.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
