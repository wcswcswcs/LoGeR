#!/usr/bin/env python3
"""Run GroupViT as an open-vocabulary STUFF-only sparse backend.

This is an audit smoke tool. It does not run detector, SAM, or fusion stages.
It converts GroupViT per-frame segmentation logits into v2-compatible sparse
stuff tracks so the same renderer and metrics helpers can be used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, GroupViTModel

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
    parser = argparse.ArgumentParser(description="Run GroupViT STUFF-only sparse smoke.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="nvidia/groupvit-gcc-redcaps")
    parser.add_argument("--frames_limit", type=int, default=64)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--labels", default="wall,floor,ceiling,curtain")
    parser.add_argument(
        "--negative_labels",
        default="person,people,clothing,hair,chair,table,desk,furniture,screen,monitor,object,background",
    )
    parser.add_argument("--prompt_template", default="a photo of {label}")
    parser.add_argument("--confidence_threshold", type=float, default=0.35)
    parser.add_argument("--min_area_ratio", type=float, default=0.004)
    parser.add_argument("--max_area_ratio", type=float, default=0.90)
    parser.add_argument("--morph_kernel", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local_files_only", type=int, default=1)
    parser.add_argument("--do_center_crop", type=int, default=1)
    parser.add_argument(
        "--pre_resize_square",
        type=int,
        default=0,
        help="Resize each full frame to model square input before the processor, avoiding center-crop loss.",
    )
    parser.add_argument("--square_size", type=int, default=224)
    parser.add_argument("--contact_frames", default="0,10,20,30,40,50,63")
    return parser.parse_args()


def _split_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _resolve_device(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(requested)


def _load_processing_frames(args: argparse.Namespace) -> tuple[List[str], List[str], tuple[int, int]]:
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


def _read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_logits_to_frame(logits: torch.Tensor, H: int, W: int) -> np.ndarray:
    logits = logits.detach().float().cpu().numpy()
    resized = np.empty((logits.shape[0], H, W), dtype=np.float32)
    for idx in range(logits.shape[0]):
        resized[idx] = cv2.resize(logits[idx], (W, H), interpolation=cv2.INTER_LINEAR)
    return resized


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = _split_csv(args.labels)
    negative_labels = _split_csv(args.negative_labels)
    if not labels:
        raise ValueError("--labels cannot be empty")
    all_labels = labels + [label for label in negative_labels if label not in labels]
    prompts = [str(args.prompt_template).format(label=label) for label in all_labels]
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    device = _resolve_device(args.device)
    local_files_only = bool(int(args.local_files_only))
    image_paths, temp_dirs, proc_shape = _load_processing_frames(args)
    H, W = proc_shape
    kernel = _kernel(int(args.morph_kernel))
    threshold = float(args.confidence_threshold)
    min_area = float(args.min_area_ratio)
    max_area = float(args.max_area_ratio)

    tracks: Dict[str, Dict[str, Any]] = {
        label: _make_track(label, "stuff_static", 0, H, W, "groupvit", None)
        for label in labels
    }

    t0 = time.time()
    print(f"GroupViT loading model={args.model} device={device}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=local_files_only)
    model = GroupViTModel.from_pretrained(args.model, local_files_only=local_files_only)
    model.eval().to(device)

    observed_pixels = {label: 0 for label in all_labels}
    masks_added = {label: 0 for label in labels}
    frames_with_any = 0

    try:
        for start in range(0, len(image_paths), max(1, int(args.batch_size))):
            batch_paths = image_paths[start : start + max(1, int(args.batch_size))]
            images = []
            for path in batch_paths:
                rgb = _read_rgb(path)
                if bool(int(args.pre_resize_square)):
                    side = max(1, int(args.square_size))
                    rgb = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_LINEAR)
                images.append(Image.fromarray(rgb))
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                do_center_crop=bool(int(args.do_center_crop)),
            )
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_segmentation=True)
            logits_batch = outputs.segmentation_logits
            if logits_batch.ndim != 4:
                raise RuntimeError(f"Unexpected segmentation logits shape: {tuple(logits_batch.shape)}")

            for offset, frame_path in enumerate(batch_paths):
                frame_idx = start + offset
                logits = _resize_logits_to_frame(logits_batch[offset], H, W)
                prob = torch.from_numpy(logits).softmax(0).numpy()
                pred = prob.argmax(axis=0)
                conf = prob.max(axis=0)

                any_mask = False
                for label in all_labels:
                    label_idx = label_to_idx[label]
                    observed_pixels[label] += int((pred == label_idx).sum())

                for label in labels:
                    label_idx = label_to_idx[label]
                    mask = (pred == label_idx) & (conf >= threshold)
                    mask = _postprocess_mask(mask, kernel)
                    area_ratio = float(mask.sum()) / float(max(H * W, 1))
                    if area_ratio < min_area or area_ratio > max_area:
                        continue
                    _write_mask(tracks[label], frame_idx, mask, float(conf[mask].mean()) if mask.any() else 0.0, H, W)
                    masks_added[label] += 1
                    any_mask = True
                frames_with_any += int(any_mask)

            if len(image_paths) >= 16 and (start + len(batch_paths)) % 16 == 0:
                print(f"  GroupViT processed {start + len(batch_paths)}/{len(image_paths)} frames", flush=True)
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
            "groupvit_stuff_standalone": {
                "model": str(args.model),
                "labels": labels,
                "negative_labels": negative_labels,
                "prompts": prompts,
                "frames": int(len(image_paths)),
                "threshold": threshold,
                "min_area_ratio": min_area,
                "max_area_ratio": max_area,
                "morph_kernel": int(args.morph_kernel),
                "do_center_crop": bool(int(args.do_center_crop)),
                "pre_resize_square": bool(int(args.pre_resize_square)),
                "square_size": int(args.square_size),
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
        "groupvit_debug": sparse.debug["groupvit_stuff_standalone"],
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
