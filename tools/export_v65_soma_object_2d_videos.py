#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from stream4d_native.v65_common import sha256_file
from stream4d_native.v65_visualization_export import _id_colors


def _read_support_rows(path: Path, scene: str) -> tuple[dict[int, list[tuple[str, int]]], dict[str, int]]:
    by_frame: dict[int, set[tuple[str, int]]] = defaultdict(set)
    object_to_idx: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene_id") != scene:
                continue
            if row.get("support_kind") != "object_declared_mask_observation_support":
                continue
            frame_text = str(row.get("frame_id") or "").strip()
            mask_text = str(row.get("observed_mask_id") or "").strip()
            object_id = str(row.get("object_id") or "").strip()
            if not frame_text or not mask_text or not object_id:
                continue
            if object_id not in object_to_idx:
                object_to_idx[object_id] = len(object_to_idx) + 1
            by_frame[int(frame_text)].add((object_id, int(mask_text)))
    return {frame: sorted(items) for frame, items in sorted(by_frame.items())}, object_to_idx


def _overlay(rgb: np.ndarray, labels: np.ndarray, *, alpha: float) -> np.ndarray:
    if labels.shape[:2] != rgb.shape[:2]:
        labels = cv2.resize(labels, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    positive = labels > 0
    colors = np.zeros((*labels.shape[:2], 3), dtype=np.uint8)
    ids = np.unique(labels[positive])
    for value in ids:
        colors[labels == value] = _id_colors(np.asarray([int(value)], dtype=np.int64))[0]
    out = rgb.copy()
    out[positive] = (
        (1.0 - float(alpha)) * out[positive].astype(np.float32)
        + float(alpha) * colors[positive].astype(np.float32)
    ).astype(np.uint8)
    return out


def _put_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _open_writer(path: Path, shape: tuple[int, int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(shape[0]), int(shape[1])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


def export(args: argparse.Namespace) -> dict[str, Any]:
    scene_root = STREAM3D_ROOT / "data" / "scannet" / "processed" / args.scene
    rgb_dir = scene_root / "color"
    soma_mask_dir = scene_root / f"output_{args.backbone}" / "mask"
    gt_instance_dir = scene_root / "instance" / "instance"
    gt_sem_dir = scene_root / "label-filt"
    support_by_frame, object_to_idx = _read_support_rows(Path(args.object_support_rows), args.scene)
    if not support_by_frame:
        raise RuntimeError(f"no SOMA object support rows for scene={args.scene}")
    frames = sorted(support_by_frame)
    if args.max_frames > 0:
        frames = frames[: int(args.max_frames)]

    first_rgb = cv2.imread(str(rgb_dir / f"{frames[0]}.jpg"), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise FileNotFoundError(rgb_dir / f"{frames[0]}.jpg")
    if args.resize_width > 0 and first_rgb.shape[1] != int(args.resize_width):
        scale = float(args.resize_width) / float(first_rgb.shape[1])
        frame_shape = (int(round(first_rgb.shape[0] * scale)), int(args.resize_width), 3)
    else:
        frame_shape = first_rgb.shape

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    soma_video = output_root / f"{args.scene}_soma_objects_2d.mp4"
    gt_video = output_root / f"{args.scene}_gt_instances_2d.mp4"
    gt_sem_video = output_root / f"{args.scene}_gt_sem_2d.mp4"
    compare_video = output_root / f"{args.scene}_soma_vs_gt_2d.mp4"
    compare_sem_video = output_root / f"{args.scene}_soma_vs_gt_sem_2d.mp4"
    soma_writer = _open_writer(soma_video, frame_shape, args.fps)
    gt_writer = _open_writer(gt_video, frame_shape, args.fps)
    gt_sem_writer = _open_writer(gt_sem_video, frame_shape, args.fps)
    compare_writer = _open_writer(compare_video, (frame_shape[0], frame_shape[1] * 2, 3), args.fps)
    compare_sem_writer = _open_writer(compare_sem_video, (frame_shape[0], frame_shape[1] * 2, 3), args.fps)

    object_ids: set[str] = set()
    total_soma_pixels = 0
    total_gt_pixels = 0
    total_gt_sem_pixels = 0
    frame_rows: list[dict[str, Any]] = []
    try:
        for frame in frames:
            rgb_bgr = cv2.imread(str(rgb_dir / f"{frame}.jpg"), cv2.IMREAD_COLOR)
            crop_mask = cv2.imread(str(soma_mask_dir / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
            gt_instance = cv2.imread(str(gt_instance_dir / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
            gt_sem = cv2.imread(str(gt_sem_dir / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
            if rgb_bgr is None or crop_mask is None or gt_instance is None or gt_sem is None:
                frame_rows.append(
                    {
                        "frame_id": frame,
                        "ok": False,
                        "missing_rgb": rgb_bgr is None,
                        "missing_soma_mask": crop_mask is None,
                        "missing_gt_instance": gt_instance is None,
                        "missing_gt_sem": gt_sem is None,
                    }
                )
                continue
            if crop_mask.ndim == 3:
                crop_mask = crop_mask[..., 0]
            if gt_instance.ndim == 3:
                gt_instance = gt_instance[..., 0]
            if gt_sem.ndim == 3:
                gt_sem = gt_sem[..., 0]
            soma_labels = np.zeros(crop_mask.shape[:2], dtype=np.int32)
            for object_id, mask_id in support_by_frame.get(frame, []):
                object_ids.add(object_id)
                soma_labels[crop_mask == int(mask_id)] = int(object_to_idx[object_id])
            soma_rgb = _overlay(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB), soma_labels, alpha=args.alpha)
            gt_rgb = _overlay(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB), gt_instance.astype(np.int32), alpha=args.alpha)
            gt_sem_rgb = _overlay(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB), gt_sem.astype(np.int32), alpha=args.alpha)
            soma_rgb = _put_label(soma_rgb, f"SOMA object support | {args.scene} frame {frame}")
            gt_rgb = _put_label(gt_rgb, f"GT instance | {args.scene} frame {frame}")
            gt_sem_rgb = _put_label(gt_sem_rgb, f"GT semantic | {args.scene} frame {frame}")
            if (soma_rgb.shape[0], soma_rgb.shape[1]) != frame_shape[:2]:
                soma_rgb = cv2.resize(soma_rgb, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_AREA)
                gt_rgb = cv2.resize(gt_rgb, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_AREA)
                gt_sem_rgb = cv2.resize(gt_sem_rgb, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_AREA)
            soma_writer.write(cv2.cvtColor(soma_rgb, cv2.COLOR_RGB2BGR))
            gt_writer.write(cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))
            gt_sem_writer.write(cv2.cvtColor(gt_sem_rgb, cv2.COLOR_RGB2BGR))
            compare_writer.write(cv2.cvtColor(np.concatenate([soma_rgb, gt_rgb], axis=1), cv2.COLOR_RGB2BGR))
            compare_sem_writer.write(cv2.cvtColor(np.concatenate([soma_rgb, gt_sem_rgb], axis=1), cv2.COLOR_RGB2BGR))
            soma_pixels = int(np.count_nonzero(soma_labels))
            gt_pixels = int(np.count_nonzero(gt_instance))
            gt_sem_pixels = int(np.count_nonzero(gt_sem))
            total_soma_pixels += soma_pixels
            total_gt_pixels += gt_pixels
            total_gt_sem_pixels += gt_sem_pixels
            frame_rows.append(
                {
                    "frame_id": int(frame),
                    "ok": True,
                    "soma_support_pair_count": int(len(support_by_frame.get(frame, []))),
                    "soma_overlay_pixel_count": soma_pixels,
                    "gt_overlay_pixel_count": gt_pixels,
                    "gt_sem_overlay_pixel_count": gt_sem_pixels,
                    "gt_instance_count": int(np.unique(gt_instance[gt_instance > 0]).shape[0]),
                    "gt_sem_label_count": int(np.unique(gt_sem[gt_sem > 0]).shape[0]),
                }
            )
    finally:
        soma_writer.release()
        gt_writer.release()
        gt_sem_writer.release()
        compare_writer.release()
        compare_sem_writer.release()

    status = {
        "phase": "v65_soma_object_2d_videos",
        "scene": args.scene,
        "frame_count": int(sum(1 for row in frame_rows if row.get("ok"))),
        "requested_frame_count": int(len(frames)),
        "frame_ids": [int(v) for v in frames],
        "object_count": int(len(object_ids)),
        "support_pair_count": int(sum(len(v) for v in support_by_frame.values())),
        "total_soma_overlay_pixels": int(total_soma_pixels),
        "total_gt_overlay_pixels": int(total_gt_pixels),
        "total_gt_sem_overlay_pixels": int(total_gt_sem_pixels),
        "soma_video": str(soma_video),
        "gt_video": str(gt_video),
        "gt_sem_video": str(gt_sem_video),
        "compare_video": str(compare_video),
        "compare_sem_video": str(compare_sem_video),
        "soma_video_sha256": sha256_file(soma_video.resolve()),
        "gt_video_sha256": sha256_file(gt_video.resolve()),
        "gt_sem_video_sha256": sha256_file(gt_sem_video.resolve()),
        "compare_video_sha256": sha256_file(compare_video.resolve()),
        "compare_sem_video_sha256": sha256_file(compare_sem_video.resolve()),
        "gt_sem_source": str(gt_sem_dir),
        "source_support_rows": str(args.object_support_rows),
        "backbone": args.backbone,
        "frame_rows": frame_rows,
    }
    status_path = output_root / "video_export_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one-scene SOMA object and GT instance 2D videos.")
    parser.add_argument("--scene", default="scene0050_00")
    parser.add_argument("--object-support-rows", default="Stream3D/outputs/audit/v65_soma_object_bank/soma_object_support_rows.csv")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--max-frames", type=int, default=0)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
