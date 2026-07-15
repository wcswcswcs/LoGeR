#!/usr/bin/env python3
"""Measure cross-boundary 3D vertex support overlap for v105 local2history audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _read_label(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(str(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int32)


def _parse_ids(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    out: list[int] = []
    for part in str(value).split(","):
        item = part.strip()
        if item:
            out.append(int(item))
    return out


def _frame_window(start: int, *, count: int, stride: int, direction: str) -> list[int]:
    if direction == "prev":
        frames = [int(start) - int(stride) * i for i in range(max(int(count), 1))]
        return sorted([frame for frame in frames if frame >= 0])
    if direction == "curr":
        return [int(start) + int(stride) * i for i in range(max(int(count), 1))]
    raise ValueError(f"Unsupported direction: {direction}")


def _label_at_vertex_shape(label_path: Path, vertex_shape: tuple[int, int]) -> np.ndarray:
    label = _read_label(label_path)
    if tuple(label.shape[:2]) == tuple(vertex_shape):
        return label
    height, width = vertex_shape
    resized = cv2.resize(label, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.int32)


def _support_sets(
    *,
    label_dir: Path,
    vertex_map_dir: Path,
    frames: Iterable[int],
    selected_ids: list[int] | None,
) -> tuple[dict[int, set[int]], list[dict[str, object]]]:
    supports: dict[int, set[int]] = {}
    frame_records: list[dict[str, object]] = []
    selected = set(int(v) for v in selected_ids) if selected_ids is not None else None
    for frame_id in frames:
        vertex_path = vertex_map_dir / f"{int(frame_id)}.npz"
        label_path = label_dir / f"{int(frame_id)}.png"
        if not vertex_path.exists() or not label_path.exists():
            frame_records.append(
                {
                    "frame_id": int(frame_id),
                    "label_path": str(label_path),
                    "vertex_path": str(vertex_path),
                    "exists": False,
                }
            )
            continue
        npz = np.load(vertex_path)
        vertex_idx = npz["vertex_idx"].astype(np.int64)
        label = _label_at_vertex_shape(label_path, tuple(vertex_idx.shape[:2]))
        valid = vertex_idx >= 0
        ids = [int(v) for v in np.unique(label[valid]) if int(v) > 0]
        if selected is not None:
            ids = [label_id for label_id in ids if label_id in selected]
        for label_id in ids:
            mask = valid & (label == int(label_id))
            if not bool(mask.any()):
                continue
            verts = np.unique(vertex_idx[mask]).astype(np.int64)
            bucket = supports.setdefault(int(label_id), set())
            bucket.update(int(v) for v in verts.tolist())
        frame_records.append(
            {
                "frame_id": int(frame_id),
                "label_path": str(label_path),
                "vertex_path": str(vertex_path),
                "exists": True,
                "label_shape": [int(v) for v in label.shape[:2]],
                "vertex_shape": [int(v) for v in vertex_idx.shape[:2]],
                "valid_vertex_pixels": int(valid.sum()),
                "selected_ids": sorted(ids),
            }
        )
    return supports, frame_records


def _pair_rows(
    prev_supports: dict[int, set[int]],
    curr_supports: dict[int, set[int]],
    *,
    min_support_points: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for curr_id, curr_set in sorted(curr_supports.items()):
        if len(curr_set) < int(min_support_points):
            continue
        for prev_id, prev_set in sorted(prev_supports.items()):
            if len(prev_set) < int(min_support_points):
                continue
            inter = len(prev_set & curr_set)
            if inter <= 0:
                continue
            union = len(prev_set | curr_set)
            prev_cov = inter / max(len(prev_set), 1)
            curr_cov = inter / max(len(curr_set), 1)
            rows.append(
                {
                    "prev_global_id": int(prev_id),
                    "curr_local_id": int(curr_id),
                    "prev_support_points": int(len(prev_set)),
                    "curr_support_points": int(len(curr_set)),
                    "intersection_points": int(inter),
                    "union_points": int(union),
                    "point_iou": float(inter / max(union, 1)),
                    "prev_coverage": float(prev_cov),
                    "curr_coverage": float(curr_cov),
                    "min_coverage": float(min(prev_cov, curr_cov)),
                    "max_coverage": float(max(prev_cov, curr_cov)),
                    "object_to_part_score": float(curr_cov * min(1.0, inter / max(int(min_support_points), 1))),
                }
            )
    rows.sort(
        key=lambda row: (
            float(row["object_to_part_score"]),
            float(row["curr_coverage"]),
            float(row["point_iou"]),
            int(row["intersection_points"]),
        ),
        reverse=True,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-mask-dir", required=True, type=Path)
    parser.add_argument("--global-mask-dir", required=True, type=Path)
    parser.add_argument("--vertex-map-dir", required=True, type=Path)
    parser.add_argument("--prev-frame", required=True, type=int)
    parser.add_argument("--curr-frame", required=True, type=int)
    parser.add_argument("--prev-global-ids", default="")
    parser.add_argument("--curr-local-ids", default="")
    parser.add_argument("--window-frames", type=int, default=6)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--min-support-points", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    prev_ids = _parse_ids(args.prev_global_ids)
    curr_ids = _parse_ids(args.curr_local_ids)
    prev_frames = _frame_window(args.prev_frame, count=args.window_frames, stride=args.stride, direction="prev")
    curr_frames = _frame_window(args.curr_frame, count=args.window_frames, stride=args.stride, direction="curr")
    prev_supports, prev_frame_records = _support_sets(
        label_dir=args.global_mask_dir,
        vertex_map_dir=args.vertex_map_dir,
        frames=prev_frames,
        selected_ids=prev_ids,
    )
    curr_supports, curr_frame_records = _support_sets(
        label_dir=args.source_mask_dir,
        vertex_map_dir=args.vertex_map_dir,
        frames=curr_frames,
        selected_ids=curr_ids,
    )
    pair_rows = _pair_rows(prev_supports, curr_supports, min_support_points=args.min_support_points)
    record = {
        "schema_version": "stream4d_v105_boundary_vertex_support_diagnostic_v1",
        "scene_id": args.scene_id,
        "source_mask_dir": str(args.source_mask_dir),
        "global_mask_dir": str(args.global_mask_dir),
        "vertex_map_dir": str(args.vertex_map_dir),
        "prev_frame": int(args.prev_frame),
        "curr_frame": int(args.curr_frame),
        "prev_frames": prev_frames,
        "curr_frames": curr_frames,
        "prev_global_ids_filter": prev_ids,
        "curr_local_ids_filter": curr_ids,
        "window_frames": int(args.window_frames),
        "stride": int(args.stride),
        "min_support_points": int(args.min_support_points),
        "prev_support_counts": {str(k): int(len(v)) for k, v in sorted(prev_supports.items())},
        "curr_support_counts": {str(k): int(len(v)) for k, v in sorted(curr_supports.items())},
        "pair_count": int(len(pair_rows)),
        "top_pairs": pair_rows[: max(int(args.top_k), 0)],
        "prev_frame_records": prev_frame_records,
        "curr_frame_records": curr_frame_records,
        "diagnostic_note": (
            "Masks are resized with nearest-neighbor to the vertex_idx map resolution when needed. "
            "These rows are geometry support evidence only; they do not by themselves prove object identity."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"output_json_sha256={hashlib.sha256(args.output_json.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
