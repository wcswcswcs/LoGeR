#!/usr/bin/env python3
"""Run a local MMSeg Mask2Former checkpoint as STUFF-only sparse output.

This audit tool is intended for indoor scene parsing checkpoints such as the
local ScanNet/Replica Mask2Former weights. It does not run detector, SAM, or
fusion stages.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_sparse_masklet_slice import _make_single_contact  # noqa: E402
from refine_sparse_stuff_masks import coverage_stats, parse_contact_frames, track_stats  # noqa: E402
from run_video_masklet_front_end import SparseMaskletOutput, collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import _make_track, _write_mask, create_tracking_video_v2, save_sparse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMSeg Mask2Former STUFF-only sparse smoke.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="/home/tmp_datasets/weights/m2f/mask2former_r50_scannet_2d_240x320_pretrain.py")
    parser.add_argument("--checkpoint", default="/home/tmp_datasets/weights/m2f/best_mIoU_iter_50000.pth")
    parser.add_argument("--mmseg_root", default="/mnt/data/users/chengshun.wang/mmsegmentation")
    parser.add_argument("--frames_limit", type=int, default=64)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--labels", default="wall,floor,door,window,curtain,table,desk,cabinet,bookshelf")
    parser.add_argument(
        "--class_id_labels",
        default="",
        help="Optional numeric-class override, e.g. '47:wall,23:floor'. Useful when checkpoint meta only has '0'..'N'.",
    )
    parser.add_argument("--ignore_labels", default="invalid,otherfurn,picture,counter")
    parser.add_argument("--min_area_ratio", type=float, default=0.004)
    parser.add_argument("--max_area_ratio", type=float, default=0.90)
    parser.add_argument("--morph_kernel", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contact_frames", default="0,10,20,30,40,50,63")
    return parser.parse_args()


def _split_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _parse_class_id_labels(text: str, num_classes: int) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for item in _split_csv(text):
        if ":" not in item:
            raise ValueError(f"Bad --class_id_labels item {item!r}; expected '<class_id>:<label>'")
        left, right = item.split(":", 1)
        class_id = int(left.strip())
        label = right.strip()
        if not label:
            raise ValueError(f"Bad --class_id_labels item {item!r}; empty label")
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"Class id {class_id} out of range for num_classes={num_classes}")
        out[class_id] = label
    return out


def _load_processing_frames(args: argparse.Namespace) -> tuple[List[str], List[str], tuple[int, int]]:
    image_paths, temp_dir = collect_image_paths(
        args.input_video,
        int(args.start_frame),
        int(args.end_frame),
        int(args.frame_stride),
    )
    temp_dirs = [temp_dir] if temp_dir else []
    if args.frames_limit and int(args.frames_limit) > 0:
        image_paths = image_paths[: int(args.frames_limit)]
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(
        image_paths,
        int(args.processing_max_side),
    )
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    return list(image_paths), temp_dirs, tuple(int(x) for x in proc_shape)


def _kernel(size: int) -> np.ndarray | None:
    k = int(size)
    if k <= 1:
        return None
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _postprocess_mask(mask: np.ndarray, kernel: np.ndarray | None) -> np.ndarray:
    out = mask.astype(np.uint8)
    if kernel is not None:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out.astype(bool)


def _class_names(model: Any) -> List[str]:
    meta = getattr(model, "dataset_meta", None) or {}
    classes = list(meta.get("classes", []))
    if not classes:
        raise RuntimeError("MMSeg model has no dataset_meta['classes']; cannot map class ids to labels")
    return [str(x) for x in classes]


def _pred_array(result: Any) -> np.ndarray:
    pred = result.pred_sem_seg.data
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu()
    pred = np.asarray(pred).squeeze()
    if pred.ndim != 2:
        raise RuntimeError(f"Unexpected pred_sem_seg shape: {pred.shape}")
    return pred.astype(np.int64, copy=False)


def main() -> None:
    args = parse_args()
    mmseg_root = Path(args.mmseg_root).resolve()
    if str(mmseg_root) not in sys.path:
        sys.path.insert(0, str(mmseg_root))

    from mmseg.apis import init_model, inference_model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = _split_csv(args.labels)
    ignore_labels = {x.lower() for x in _split_csv(args.ignore_labels)}
    if not labels:
        raise ValueError("--labels cannot be empty")

    image_paths, temp_dirs, proc_shape = _load_processing_frames(args)
    H, W = proc_shape
    kernel = _kernel(int(args.morph_kernel))
    min_area = float(args.min_area_ratio)
    max_area = float(args.max_area_ratio)

    t0 = time.time()
    print(f"MMSeg init config={args.config} checkpoint={args.checkpoint} device={args.device}", flush=True)
    model = init_model(args.config, args.checkpoint, device=str(args.device))
    classes = _class_names(model)
    class_id_labels = _parse_class_id_labels(args.class_id_labels, len(classes))
    label_to_id = {name.lower(): idx for idx, name in enumerate(classes)}
    for class_id, label in class_id_labels.items():
        label_to_id[label.lower()] = int(class_id)
    selected = {label: label_to_id[label.lower()] for label in labels if label.lower() in label_to_id}
    missing = [label for label in labels if label.lower() not in label_to_id]
    if not selected:
        raise RuntimeError(f"None of requested labels exist in model classes={classes}")

    tracks: Dict[str, Dict[str, Any]] = {
        label: _make_track(label, "stuff_static", 0, H, W, "mmseg_m2f", None)
        for label in selected
        if label.lower() not in ignore_labels
    }
    observed_pixels = {class_id_labels.get(idx, name): 0 for idx, name in enumerate(classes)}
    masks_added = {label: 0 for label in tracks}
    frames_with_any = 0

    try:
        for frame_idx, path in enumerate(image_paths):
            result = inference_model(model, path)
            pred = _pred_array(result)
            if pred.shape != (H, W):
                pred = cv2.resize(pred.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            vals, counts = np.unique(pred, return_counts=True)
            for val, count in zip(vals.tolist(), counts.tolist()):
                if 0 <= int(val) < len(classes):
                    observed_pixels[class_id_labels.get(int(val), classes[int(val)])] += int(count)

            any_mask = False
            for label, class_id in selected.items():
                if label not in tracks:
                    continue
                mask = pred == int(class_id)
                mask = _postprocess_mask(mask, kernel)
                area_ratio = float(mask.sum()) / float(max(H * W, 1))
                if area_ratio < min_area or area_ratio > max_area:
                    continue
                _write_mask(tracks[label], frame_idx, mask, 1.0, H, W)
                masks_added[label] += 1
                any_mask = True
            frames_with_any += int(any_mask)
            if len(image_paths) >= 16 and (frame_idx + 1) % 16 == 0:
                print(f"  MMSeg M2F processed {frame_idx + 1}/{len(image_paths)} frames", flush=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sparse_tracks = [track for track in tracks.values() if track.get("mask_by_frame")]
    sparse = SparseMaskletOutput(
        tracks=sparse_tracks,
        num_masklets=len(sparse_tracks),
        num_frames=len(image_paths),
        frame_height=H,
        frame_width=W,
        debug={
            "mmseg_m2f_stuff_standalone": {
                "config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "mmseg_root": str(mmseg_root),
                "classes": classes,
                "class_id_labels": {str(k): v for k, v in sorted(class_id_labels.items())},
                "requested_labels": labels,
                "selected_labels": selected,
                "missing_labels": missing,
                "ignore_labels": sorted(ignore_labels),
                "frames": int(len(image_paths)),
                "min_area_ratio": min_area,
                "max_area_ratio": max_area,
                "morph_kernel": int(args.morph_kernel),
                "observed_pixels": observed_pixels,
                "masks_added": masks_added,
                "frames_with_any": int(frames_with_any),
                "elapsed_seconds": float(time.time() - t0),
            }
        },
    )

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    metrics_path = output_dir / "metrics_summary.json"
    contact_path = output_dir / "contact_sheet.jpg"

    save_sparse_output(output_pt, sparse)
    create_tracking_video_v2(
        image_paths,
        sparse,
        str(output_video),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style="clean",
    )
    _make_single_contact(
        image_paths,
        sparse,
        parse_contact_frames(args.contact_frames, sparse.num_frames),
        contact_path,
        float(args.mask_alpha),
    )

    summary = {
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "coverage": coverage_stats(sparse),
        "track_stats": track_stats(sparse),
        "mmseg_m2f_debug": sparse.debug["mmseg_m2f_stuff_standalone"],
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
