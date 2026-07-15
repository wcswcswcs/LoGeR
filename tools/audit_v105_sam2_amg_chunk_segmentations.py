#!/usr/bin/env python3
"""Visualize SAM2 AMG segmentation independently on every frame of one chunk.

This is intentionally not a tracker path: each RGB frame is segmented with
SAM2AutomaticMaskGenerator, then visualized independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RGB_ROOT = REPO_ROOT / "Stream3D" / "data" / "scannet" / "processed"
DEFAULT_OUT = REPO_ROOT / "Stream3D" / "outputs" / "audit" / "v105_sam2_amg_per_frame_scene0011_r1"
PALETTE = np.array(
    [
        [230, 57, 70],
        [29, 53, 87],
        [69, 123, 157],
        [42, 157, 143],
        [233, 196, 106],
        [244, 162, 97],
        [131, 56, 236],
        [255, 0, 110],
        [58, 134, 255],
        [6, 214, 160],
        [255, 209, 102],
        [17, 138, 178],
        [239, 71, 111],
        [7, 59, 76],
        [118, 200, 147],
        [251, 86, 7],
        [0, 166, 251],
        [255, 190, 11],
        [158, 42, 43],
        [90, 24, 154],
        [64, 145, 108],
        [204, 213, 174],
        [188, 71, 73],
        [53, 80, 112],
    ],
    dtype=np.uint8,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def label_from_masks(masks: list[np.ndarray], h: int, w: int, max_masks: int) -> np.ndarray:
    if not masks:
        return np.zeros((h, w), dtype=np.uint16)
    areas = np.asarray([int(np.count_nonzero(m)) for m in masks], dtype=np.int64)
    order = np.argsort(-areas)[: max(int(max_masks), 1)]
    label = np.zeros((h, w), dtype=np.uint16)
    for out_id, idx in enumerate(order, start=1):
        mask = np.asarray(masks[int(idx)]).astype(bool)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        label[mask] = int(out_id)
    return label


def overlay_label(rgb: np.ndarray, label: np.ndarray, alpha: float = 0.52) -> np.ndarray:
    h, w = rgb.shape[:2]
    if label.shape[:2] != (h, w):
        label = cv2.resize(label.astype(np.uint16), (w, h), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(rgb)
    ids = [int(v) for v in np.unique(label) if int(v) > 0]
    for obj_id in ids:
        color[label == obj_id] = PALETTE[(obj_id - 1) % len(PALETTE)]
    a = (label > 0).astype(np.float32)[..., None] * float(alpha)
    blended = (rgb.astype(np.float32) * (1.0 - a) + color.astype(np.float32) * a).astype(np.uint8)
    boundaries = np.zeros((h, w), dtype=np.uint8)
    for obj_id in ids:
        m = (label == obj_id).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundaries, contours, -1, 255, 2)
    blended[boundaries > 0] = np.array([255, 255, 255], dtype=np.uint8)
    return blended


def annotate_frame(rgb: np.ndarray, text: str) -> Image.Image:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    pad = 10
    draw.rectangle((0, 0, bbox[2] + 2 * pad, bbox[3] + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return image


def mask_stats(label: np.ndarray) -> dict[str, Any]:
    ids = [int(v) for v in np.unique(label) if int(v) > 0]
    rows: list[dict[str, Any]] = []
    for obj_id in ids:
        mask = label == int(obj_id)
        area = int(np.count_nonzero(mask))
        bbox = bbox_xyxy(mask)
        rows.append(
            {
                "id": int(obj_id),
                "area": int(area),
                "area_ratio": float(area) / float(max(label.size, 1)),
                "bbox_xyxy": bbox or [],
            }
        )
    rows.sort(key=lambda row: int(row["area"]), reverse=True)
    return {
        "visible_id_count": int(len(ids)),
        "foreground_pixels": int(np.count_nonzero(label > 0)),
        "foreground_ratio": float(np.count_nonzero(label > 0)) / float(max(label.size, 1)),
        "top_masks": rows[:12],
    }


def parse_frame_ids(value: str, start: int, stride: int, count: int) -> list[int]:
    if value.strip():
        return [int(part.strip()) for part in value.replace(";", ",").split(",") if part.strip()]
    return [int(start + i * stride) for i in range(int(count))]


def make_sheet(frame_paths: list[Path], out_path: Path, cell_width: int) -> None:
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    if not images:
        raise RuntimeError("no frames for sheet")
    resized: list[Image.Image] = []
    for image in images:
        ratio = float(cell_width) / float(max(image.width, 1))
        resized.append(image.resize((int(cell_width), int(round(image.height * ratio))), Image.Resampling.LANCZOS))
    cell_h = max(img.height for img in resized)
    canvas = Image.new("RGB", (4 * int(cell_width), 4 * int(cell_h)), (0, 0, 0))
    for idx, image in enumerate(resized):
        r, c = divmod(idx, 4)
        canvas.paste(image, (c * int(cell_width), r * int(cell_h)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default="scene0011_00")
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--chunk-index-offset", type=int, default=0)
    parser.add_argument("--rgb-root", default=str(DEFAULT_RGB_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--repo-path", default=str(REPO_ROOT / "Grounded-SAM-2"))
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "Grounded-SAM-2" / "checkpoints" / "sam2.1_hiera_large.pt"))
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--points-per-batch", type=int, default=128)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.7)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--stability-score-offset", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=2)
    parser.add_argument("--min-mask-region-area", type=int, default=25)
    parser.add_argument("--max-masks-for-label", type=int, default=96)
    parser.add_argument("--sheet-cell-width", type=int, default=520)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 AMG chunk audit requires CUDA")
    frame_ids = parse_frame_ids(args.frame_ids, args.frame_start, args.frame_stride, args.frame_count)
    output_root = Path(args.output_root).resolve()
    labels_dir = output_root / "labels"
    overlays_dir = output_root / "overlays"
    sheets_dir = output_root / "sheets"
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    for name in list(sys.modules):
        if name == "sam2" or name.startswith("sam2."):
            del sys.modules[name]
    repo_path = Path(args.repo_path).resolve()
    sys.path.insert(0, str(repo_path))

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    t0 = time.time()
    model = build_sam2(args.model_cfg, str(checkpoint), device="cuda", apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=int(args.points_per_side),
        points_per_batch=int(args.points_per_batch),
        pred_iou_thresh=float(args.pred_iou_thresh),
        stability_score_thresh=float(args.stability_score_thresh),
        stability_score_offset=float(args.stability_score_offset),
        crop_n_layers=int(args.crop_n_layers),
        box_nms_thresh=float(args.box_nms_thresh),
        crop_n_points_downscale_factor=int(args.crop_n_points_downscale_factor),
        min_mask_region_area=int(args.min_mask_region_area),
        use_m2m=True,
        output_mode="binary_mask",
    )

    records: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    rgb_root = Path(args.rgb_root).resolve() / args.scene_id / "color"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for chunk_idx, frame_id in enumerate(frame_ids):
            global_chunk_idx = int(args.chunk_index_offset) + int(chunk_idx)
            frame_t0 = time.time()
            rgb_path = rgb_root / f"{int(frame_id)}.jpg"
            rgb = read_rgb(rgb_path)
            h, w = rgb.shape[:2]
            masks = list(generator.generate(rgb))
            mask_rows: list[tuple[float, np.ndarray]] = []
            for row in masks:
                mask = np.asarray(row.get("segmentation")).astype(bool)
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                score = float(row.get("predicted_iou", row.get("stability_score", 0.0)) or 0.0)
                mask_rows.append((score, mask))
            mask_rows.sort(key=lambda item: item[0], reverse=True)
            selected = [mask for _, mask in mask_rows[: int(args.max_masks_for_label)]]
            label = label_from_masks(selected, h, w, int(args.max_masks_for_label))
            label_path = labels_dir / f"frame_{int(frame_id):06d}.png"
            overlay_path = overlays_dir / f"frame_{global_chunk_idx:02d}_id_{int(frame_id):06d}.jpg"
            cv2.imwrite(str(label_path), label)
            overlay = overlay_label(rgb, label)
            annotated = annotate_frame(overlay, f"frame {global_chunk_idx:02d} / id {int(frame_id)}")
            annotated.save(overlay_path, quality=95)
            overlay_paths.append(overlay_path)
            records.append(
                {
                    "schema_version": "stream4d_v105_sam2_amg_per_frame_row_v1",
                    "scene_id": args.scene_id,
                    "chunk_frame_index": int(global_chunk_idx),
                    "shard_local_frame_index": int(chunk_idx),
                    "frame_id": int(frame_id),
                    "rgb_path": str(rgb_path),
                    "rgb_sha256": sha256_file(rgb_path),
                    "label_path": str(label_path),
                    "overlay_path": str(overlay_path),
                    "native_rgb_height": int(h),
                    "native_rgb_width": int(w),
                    "raw_mask_count": int(len(masks)),
                    "visualized_mask_count": int(min(len(mask_rows), int(args.max_masks_for_label))),
                    "runtime_sec": float(time.time() - frame_t0),
                    "stats": mask_stats(label),
                }
            )
            print(
                json.dumps(
                    {
                        "frame_index": int(chunk_idx),
                        "chunk_frame_index": int(global_chunk_idx),
                        "frame_id": int(frame_id),
                        "raw_mask_count": int(len(masks)),
                        "visualized_mask_count": int(min(len(mask_rows), int(args.max_masks_for_label))),
                        "overlay_path": str(overlay_path),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    sheet_paths: list[Path] = []
    for start in range(0, len(overlay_paths), 16):
        part = overlay_paths[start : start + 16]
        if not part:
            continue
        end = start + len(part) - 1
        sheet_path = sheets_dir / f"sam2_amg_{args.scene_id}_frames_{start:02d}_{end:02d}_4x4.jpg"
        make_sheet(part, sheet_path, int(args.sheet_cell_width))
        sheet_paths.append(sheet_path)

    peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    summary = {
        "schema_version": "stream4d_v105_sam2_amg_per_frame_chunk_summary_v1",
        "scene_id": args.scene_id,
        "frame_ids": [int(v) for v in frame_ids],
        "input_policy": "native_scannet_rgb_no_pipeline_resize",
        "tracking_used": False,
        "segmentor": "sam2.1_hiera_large",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_cfg": str(args.model_cfg),
        "call": "SAM2AutomaticMaskGenerator independently per frame; no SAM2 video predictor, no tracker, no temporal ID stabilization",
        "amg_params": {
            "points_per_side": int(args.points_per_side),
            "base_grid_point_count": int(args.points_per_side) * int(args.points_per_side),
            "points_per_batch": int(args.points_per_batch),
            "pred_iou_thresh": float(args.pred_iou_thresh),
            "stability_score_thresh": float(args.stability_score_thresh),
            "stability_score_offset": float(args.stability_score_offset),
            "crop_n_layers": int(args.crop_n_layers),
            "box_nms_thresh": float(args.box_nms_thresh),
            "crop_n_points_downscale_factor": int(args.crop_n_points_downscale_factor),
            "min_mask_region_area": int(args.min_mask_region_area),
            "use_m2m": True,
            "max_masks_for_label": int(args.max_masks_for_label),
        },
        "frame_count": int(len(frame_ids)),
        "runtime_sec": float(time.time() - t0),
        "peak_gpu_memory_mb": float(peak_mb),
        "records": records,
        "sheet_paths": [str(path) for path in sheet_paths],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "sheet_paths": [str(p) for p in sheet_paths]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
