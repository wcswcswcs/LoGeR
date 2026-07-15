#!/usr/bin/env python3
"""Per-frame 4D_PM-style point-prompt segmentation audit for v105.

This is a diagnostic segmentor-only runner: every input frame is segmented
independently, with no video propagation, no tracker, and no temporal ID reuse.
Run one provider per process so SAM2 and EdgeTAM cannot share the same ``sam2``
module namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
DEFAULT_RGB_ROOT = STREAM3D_ROOT / "data" / "scannet" / "processed"
DEFAULT_OUT = STREAM3D_ROOT / "outputs" / "audit" / "v105_4dpm_style_per_frame_numpts800_scene0011_r1"

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


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**31 - 1)


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_frame_ids(value: str, start: int, stride: int, count: int) -> list[int]:
    if value.strip():
        return [int(part.strip()) for part in value.replace(";", ",").split(",") if part.strip()]
    return [int(start + i * stride) for i in range(int(count))]


def remove_small_regions(mask: np.ndarray, area_thresh: int, mode: str) -> tuple[np.ndarray, bool]:
    if mode not in {"holes", "islands"}:
        raise ValueError(f"unsupported region mode: {mode}")
    correct_holes = mode == "holes"
    working = (correct_holes ^ mask.astype(bool)).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working, 8)
    sizes = stats[:, -1][1:]
    small = [idx + 1 for idx, size in enumerate(sizes) if int(size) < int(area_thresh)]
    if not small:
        return mask.astype(bool), False
    fill_labels = [0] + small
    if not correct_holes:
        fill_labels = [idx for idx in range(n_labels) if idx not in fill_labels]
        if not fill_labels and sizes.size:
            fill_labels = [int(np.argmax(sizes)) + 1]
    return np.isin(regions, fill_labels), True


def fix_mask_regions(mask: np.ndarray, area_thresh: int = 64) -> np.ndarray:
    fixed, _ = remove_small_regions(mask.astype(bool), area_thresh=area_thresh, mode="holes")
    fixed, _ = remove_small_regions(fixed.astype(bool), area_thresh=area_thresh, mode="islands")
    return fixed.astype(bool)


def disjoin_smallest_first(masks: np.ndarray, h: int, w: int, empty_ratio: float, fix_small_regions: bool) -> np.ndarray:
    if masks.size == 0:
        return np.zeros((0, h, w), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    rows = []
    for idx in np.argsort(np.count_nonzero(masks.reshape(masks.shape[0], -1), axis=1)):
        mask = masks[int(idx)].astype(bool)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if fix_small_regions:
            mask = fix_mask_regions(mask, area_thresh=64)
        rows.append(mask)
    claimed = np.zeros((h, w), dtype=bool)
    kept = []
    min_pixels = int(h * w * float(empty_ratio))
    for mask in rows:
        residual = mask & ~claimed
        if int(np.count_nonzero(residual)) > min_pixels:
            kept.append(residual)
        claimed |= mask
    if kept:
        return np.stack(kept, axis=0).astype(bool)
    return np.zeros((0, h, w), dtype=bool)


def label_from_masks(masks: np.ndarray, h: int, w: int) -> np.ndarray:
    if masks.size == 0:
        return np.zeros((h, w), dtype=np.uint16)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    label = np.zeros((h, w), dtype=np.uint16)
    for out_id, mask in enumerate(masks.astype(bool), start=1):
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


def annotate_frame(rgb: np.ndarray, title: str, lines: list[str]) -> Image.Image:
    pad = 62
    image = Image.fromarray(rgb)
    panel = Image.new("RGB", (image.width, image.height + pad), (18, 20, 24))
    panel.paste(image, (0, pad))
    draw = ImageDraw.Draw(panel)
    try:
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font_b = ImageFont.load_default()
        font = ImageFont.load_default()
    draw.text((8, 6), title, fill=(245, 245, 245), font=font_b)
    y = 31
    for line in lines[:2]:
        draw.text((8, y), line, fill=(215, 218, 225), font=font)
        y += 15
    return panel


def make_sheet(frame_paths: list[Path], out_path: Path, cell_width: int) -> None:
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    if not images:
        raise RuntimeError("no frames for sheet")
    resized = []
    for image in images:
        ratio = float(cell_width) / float(max(image.width, 1))
        resized.append(image.resize((int(cell_width), int(round(image.height * ratio))), Image.Resampling.LANCZOS))
    cell_h = max(img.height for img in resized)
    canvas = Image.new("RGB", (4 * int(cell_width), 4 * cell_h), (0, 0, 0))
    for idx, image in enumerate(resized):
        rr, cc = divmod(idx, 4)
        canvas.paste(image, (cc * int(cell_width), rr * cell_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def mask_stats(label: np.ndarray) -> dict[str, Any]:
    ids = [int(v) for v in np.unique(label) if int(v) > 0]
    top = []
    for obj_id in ids:
        ys, xs = np.where(label == obj_id)
        if xs.size == 0:
            continue
        top.append(
            {
                "id": int(obj_id),
                "area": int(xs.size),
                "area_ratio": float(xs.size) / float(label.size),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            }
        )
    top.sort(key=lambda row: int(row["area"]), reverse=True)
    return {
        "visible_id_count": int(len(ids)),
        "foreground_pixels": int(np.count_nonzero(label > 0)),
        "foreground_ratio": float(np.count_nonzero(label > 0)) / float(label.size),
        "top_masks": top[:12],
    }


def make_points_yx_torch(num_pts: int, seed: int, mode: str, device: str = "cuda") -> tuple[Any, dict[str, Any]]:
    import torch

    mode = str(mode).strip().lower()
    if mode in {"random", "random_seeded", "4dpm_seeded_random"}:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        points = torch.rand((int(num_pts), 2), device=device, generator=generator) * 2.0 - 1.0
        return points, {
            "point_mode": "4dpm_seeded_random",
            "seed": int(seed),
            "coordinate_format": "normalized_yx_minus1_1",
            "source": "third_party/4D_PM/frontend/segment/infer.py random fallback with audit seed",
        }
    if mode in {"grid", "4dpm_grid", "deterministic_grid"}:
        side = int(torch.ceil(torch.sqrt(torch.tensor(float(max(num_pts, 1)), device=device))).item())
        coords = torch.linspace(-0.95, 0.95, side, device=device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)[: int(num_pts)], {
            "point_mode": "4dpm_deterministic_point_grid",
            "grid_side": int(side),
            "coordinate_format": "normalized_yx_minus1_1",
            "source": "third_party/4D_PM/frontend/segment/infer.py::_deterministic_keypoint_grid",
        }
    raise ValueError(f"unsupported point mode: {mode}")


def points_yx_to_xy_pixels_np(points_yx: Any, h: int, w: int) -> np.ndarray:
    arr = points_yx.detach().float().cpu().numpy().astype(np.float32)
    y = np.clip((arr[:, 0] + 1.0) * 0.5, 0.0, 1.0) * float(max(h - 1, 1))
    x = np.clip((arr[:, 1] + 1.0) * 0.5, 0.0, 1.0) * float(max(w - 1, 1))
    return np.stack([x, y], axis=1).astype(np.float32)


def points_yx_to_xy_rel_np(points_yx: Any) -> np.ndarray:
    arr = points_yx.detach().float().cpu().numpy().astype(np.float32)
    x = np.clip((arr[:, 1] + 1.0) * 0.5, 0.0, 1.0)
    y = np.clip((arr[:, 0] + 1.0) * 0.5, 0.0, 1.0)
    return np.stack([x, y], axis=1).astype(np.float32)


def masks_to_boxes_np(masks: np.ndarray) -> np.ndarray:
    boxes = np.zeros((int(masks.shape[0]), 4), dtype=np.float32)
    for idx, mask in enumerate(masks.astype(bool)):
        ys, xs = np.where(mask)
        if xs.size:
            boxes[idx] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return boxes


def run_sam2_like(args: argparse.Namespace, *, source_key: str) -> dict[str, Any]:
    import torch
    from torchvision.ops import nms

    base_output_root = Path(args.output_root).resolve()
    rgb_root = Path(args.rgb_root).resolve() / args.scene_id / "color"
    if source_key == "sam2":
        provider = "sam2_4dpm_points"
        repo_path = REPO_ROOT / "Grounded-SAM-2"
        checkpoint = repo_path / "checkpoints" / "sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif source_key == "edgetam":
        provider = "edgetam_4dpm_points"
        repo_path = REPO_ROOT / "third_party" / "EdgeTAM"
        checkpoint = repo_path / "checkpoints" / "edgetam.pt"
        model_cfg = "edgetam.yaml"
    else:
        raise ValueError(source_key)

    for name in list(sys.modules):
        if name == "sam2" or name.startswith("sam2."):
            del sys.modules[name]
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    sys.path.insert(0, str(repo_path))
    if source_key == "edgetam":
        os.chdir(repo_path)
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.utils.amg import batched_mask_to_box, calculate_stability_score

        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        t_init = time.time()
        image_model = build_sam2(model_cfg, str(checkpoint), device="cuda")
        predictor = SAM2ImagePredictor(image_model)
        init_sec = time.time() - t_init
        records = []
        overlay_paths = []
        output_root = base_output_root / provider
        label_dir = output_root / "labels"
        overlay_dir = output_root / "overlays"
        sheet_dir = output_root / "sheets"
        for directory in (label_dir, overlay_dir, sheet_dir):
            directory.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for chunk_idx, frame_id in enumerate(parse_frame_ids(args.frame_ids, args.frame_start, args.frame_stride, args.frame_count)):
                frame_t0 = time.time()
                rgb_path = rgb_root / f"{int(frame_id)}.jpg"
                rgb = read_rgb(rgb_path)
                h, w = rgb.shape[:2]
                frame_seed = stable_seed(args.seed, args.scene_id, int(frame_id), int(args.num_pts), args.point_mode)
                points_yx, point_meta = make_points_yx_torch(int(args.num_pts), frame_seed, args.point_mode)
                predictor.reset_predictor()
                predictor.set_image(rgb)
                selected_all = []
                selected_scores_all = []
                prompt_with_good = 0
                raw_option_count = 0
                batch_size = max(int(args.points_per_batch), 1)
                for start in range(0, int(args.num_pts), batch_size):
                    points_batch = points_yx[start : start + batch_size]
                    pts_px = 0.5 * torch.tensor([h - 1, w - 1], device="cuda", dtype=torch.float32) * (points_batch + 1.0)
                    pts_px = pts_px.round().long().flip(-1).float()
                    coords = predictor._transforms.transform_coords(pts_px.unsqueeze(1), normalize=True, orig_hw=(h, w))
                    labels = torch.ones((points_batch.shape[0], 1), dtype=torch.int, device="cuda")
                    masks, iou_predictions, _ = predictor._predict(coords, labels, multimask_output=True, return_logits=True)
                    stability = calculate_stability_score(masks, 0.0, float(args.stability_score_offset))
                    good = (iou_predictions > float(args.iou_threshold)) & (stability >= float(args.stability_threshold))
                    raw_option_count += int(good.numel())
                    areas = (masks > 0.0).sum(dim=(-1, -2), dtype=torch.int64)
                    masked_areas = areas.clone()
                    masked_areas[~good] = torch.iinfo(torch.int64).max
                    has_good = good.any(dim=1)
                    prompt_with_good += int(has_good.sum().item())
                    chosen_idx = masked_areas.argmin(dim=1)
                    prompt_indices = torch.nonzero(has_good, as_tuple=False).flatten()
                    if prompt_indices.numel() > 0:
                        selected = masks[prompt_indices, chosen_idx[prompt_indices]] > 0.0
                        selected_scores = stability[prompt_indices, chosen_idx[prompt_indices]].float()
                        selected_all.append(selected)
                        selected_scores_all.append(selected_scores)
                if selected_all:
                    selected_t = torch.cat(selected_all, dim=0)
                    selected_scores_t = torch.cat(selected_scores_all, dim=0)
                    boxes = batched_mask_to_box(selected_t).float()
                    keep = nms(boxes, selected_scores_t, iou_threshold=float(args.box_nms_thresh))
                    selected_t = selected_t[keep]
                    selected_np = selected_t.detach().cpu().numpy().astype(bool)
                else:
                    selected_np = np.zeros((0, h, w), dtype=bool)
                disjoint_np = disjoin_smallest_first(
                    selected_np,
                    h,
                    w,
                    empty_ratio=float(args.empty_ratio),
                    fix_small_regions=bool(args.fix_small_regions),
                )
                label = label_from_masks(disjoint_np, h, w)
                label_path = label_dir / f"frame_{int(frame_id):06d}.png"
                cv2.imwrite(str(label_path), label)
                overlay = overlay_label(rgb, label)
                stats = mask_stats(label)
                runtime_sec = time.time() - frame_t0
                annotated = annotate_frame(
                    overlay,
                    f"{provider} frame {chunk_idx:02d} / id {int(frame_id)}",
                    [f"masks={stats['visible_id_count']} fg={stats['foreground_ratio']:.3f}", f"time={runtime_sec:.2f}s points={int(args.num_pts)}"],
                )
                overlay_path = overlay_dir / f"frame_{chunk_idx:02d}_id_{int(frame_id):06d}.jpg"
                annotated.save(overlay_path, quality=95)
                overlay_paths.append(overlay_path)
                record = {
                    "schema_version": "stream4d_v105_4dpm_style_per_frame_row_v1",
                    "provider": provider,
                    "scene_id": args.scene_id,
                    "chunk_frame_index": int(chunk_idx),
                    "frame_id": int(frame_id),
                    "rgb_path": str(rgb_path),
                    "rgb_sha256": sha256_file(rgb_path),
                    "label_path": str(label_path),
                    "overlay_path": str(overlay_path),
                    "native_rgb_height": int(h),
                    "native_rgb_width": int(w),
                    "num_pts": int(args.num_pts),
                    "points_per_batch": int(args.points_per_batch),
                    "point_sampling": point_meta,
                    "iou_threshold": float(args.iou_threshold),
                    "stability_threshold": float(args.stability_threshold),
                    "stability_score_offset": float(args.stability_score_offset),
                    "box_nms_thresh": float(args.box_nms_thresh),
                    "raw_multimask_option_count": int(raw_option_count),
                    "prompt_with_good_mask_count": int(prompt_with_good),
                    "post_nms_mask_count": int(selected_np.shape[0]),
                    "post_disjoint_mask_count": int(disjoint_np.shape[0]),
                    "runtime_sec": float(runtime_sec),
                    "stats": stats,
                }
                records.append(record)
                print(json.dumps({k: record[k] for k in ("provider", "frame_id", "prompt_with_good_mask_count", "post_disjoint_mask_count", "runtime_sec")}), flush=True)
        sheet_paths = []
        for start, end in [(0, 16), (16, 32)]:
            part = overlay_paths[start:end]
            if not part:
                continue
            sheet_path = sheet_dir / f"{provider}_{args.scene_id}_chunk_frames_{start:02d}_{start + len(part) - 1:02d}_4x4.jpg"
            make_sheet(part, sheet_path, int(args.sheet_cell_width))
            sheet_paths.append(str(sheet_path))
        runtimes = [float(row["runtime_sec"]) for row in records]
        summary = {
            "schema_version": "stream4d_v105_4dpm_style_per_frame_summary_v1",
            "provider": provider,
            "scene_id": args.scene_id,
            "frame_count": int(len(records)),
            "frame_ids": [int(row["frame_id"]) for row in records],
            "tracking_used": False,
            "video_predictor_used": False,
            "call": "4D_PM-style independent per-frame SAM2ImagePredictor point prompts: seeded random normalized yx points, multimask_output=True, iou/stability filter, select-smallest per point, box NMS, small-region fix, smallest-first disjoin",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_cfg": model_cfg,
            "num_pts": int(args.num_pts),
            "points_per_batch": int(args.points_per_batch),
            "iou_threshold": float(args.iou_threshold),
            "stability_threshold": float(args.stability_threshold),
            "stability_score_offset": float(args.stability_score_offset),
            "box_nms_thresh": float(args.box_nms_thresh),
            "empty_ratio": float(args.empty_ratio),
            "model_init_runtime_sec": float(init_sec),
            "mean_runtime_sec_per_frame": float(np.mean(runtimes)) if runtimes else 0.0,
            "median_runtime_sec_per_frame": float(np.median(runtimes)) if runtimes else 0.0,
            "min_runtime_sec_per_frame": float(np.min(runtimes)) if runtimes else 0.0,
            "max_runtime_sec_per_frame": float(np.max(runtimes)) if runtimes else 0.0,
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0),
            "sheet_paths": sheet_paths,
            "records": records,
        }
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"summary": str(summary_path), "sheets": sheet_paths, "mean_runtime_sec_per_frame": summary["mean_runtime_sec_per_frame"]}, indent=2), flush=True)
        return summary
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path


def run_sam31(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torchvision.ops import nms

    provider = "sam31_4dpm_points"
    root = REPO_ROOT / "third_party" / "sam3"
    checkpoint = REPO_ROOT / "ckpts" / "SAM3" / "sam3.1_multiplex.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    old_sys_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        t_init = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model = build_sam3_image_model(
                checkpoint_path=str(checkpoint),
                load_from_HF=False,
                device="cuda",
                compile=False,
                enable_inst_interactivity=True,
            )
            processor = Sam3Processor(model, confidence_threshold=float(args.sam31_confidence_threshold), device="cuda")
        init_sec = time.time() - t_init
        output_root = Path(args.output_root).resolve() / provider
        label_dir = output_root / "labels"
        overlay_dir = output_root / "overlays"
        sheet_dir = output_root / "sheets"
        for directory in (label_dir, overlay_dir, sheet_dir):
            directory.mkdir(parents=True, exist_ok=True)
        rgb_root = Path(args.rgb_root).resolve() / args.scene_id / "color"
        records = []
        overlay_paths = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for chunk_idx, frame_id in enumerate(parse_frame_ids(args.frame_ids, args.frame_start, args.frame_stride, args.frame_count)):
                frame_t0 = time.time()
                rgb_path = rgb_root / f"{int(frame_id)}.jpg"
                rgb = read_rgb(rgb_path)
                h, w = rgb.shape[:2]
                frame_seed = stable_seed(args.seed, args.scene_id, int(frame_id), int(args.num_pts), args.point_mode)
                points_yx, point_meta = make_points_yx_torch(int(args.num_pts), frame_seed, args.point_mode)
                point_xy = points_yx_to_xy_pixels_np(points_yx, h=h, w=w)
                state = processor.set_image(Image.fromarray(np.ascontiguousarray(rgb).astype(np.uint8)))
                masks_for_frame = []
                scores_for_frame = []
                raw_option_count = 0
                batch_failure_count = 0
                batch_size = max(int(args.sam31_points_per_batch), 1)
                for start in range(0, point_xy.shape[0], batch_size):
                    coords = point_xy[start : start + batch_size].reshape(-1, 1, 2).astype(np.float32)
                    labels = np.ones((coords.shape[0], 1), dtype=np.int32)
                    try:
                        mask_options, score_values, _ = model.predict_inst(
                            state,
                            point_coords=coords,
                            point_labels=labels,
                            multimask_output=True,
                        )
                    except Exception:
                        batch_failure_count += 1
                        continue
                    mask_options_np = np.asarray(mask_options)
                    score_values_np = np.asarray(score_values, dtype=np.float32)
                    if mask_options_np.ndim == 2:
                        mask_options_np = mask_options_np[None, None, :, :]
                    elif mask_options_np.ndim == 3:
                        if score_values_np.ndim == 1 and score_values_np.size == mask_options_np.shape[0]:
                            mask_options_np = mask_options_np[None, :, :, :]
                        else:
                            mask_options_np = mask_options_np[:, None, :, :]
                    if score_values_np.ndim == 0:
                        score_values_np = score_values_np.reshape(1, 1)
                    elif score_values_np.ndim == 1:
                        if mask_options_np.shape[0] == 1 and score_values_np.size == mask_options_np.shape[1]:
                            score_values_np = score_values_np[None, :]
                        else:
                            score_values_np = score_values_np[:, None]
                    raw_option_count += int(mask_options_np.shape[0] * mask_options_np.shape[1])
                    for prompt_idx in range(mask_options_np.shape[0]):
                        scores = score_values_np[prompt_idx]
                        valid = np.where(scores >= float(args.sam31_quality_threshold))[0]
                        if valid.size == 0:
                            continue
                        if str(args.sam31_per_point_selection_policy).strip().lower() in {"smallest", "small"}:
                            areas = []
                            for option_idx in valid:
                                mask_tmp = np.asarray(mask_options_np[prompt_idx, int(option_idx)]).astype(bool)
                                areas.append(int(np.count_nonzero(mask_tmp)))
                            option_idx = int(valid[int(np.argmin(np.asarray(areas)))])
                        else:
                            option_idx = int(valid[int(np.argmax(scores[valid]))])
                        mask = np.asarray(mask_options_np[prompt_idx, option_idx]).astype(bool)
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        if np.count_nonzero(mask) <= 0:
                            continue
                        masks_for_frame.append(mask)
                        scores_for_frame.append(float(scores[option_idx]))
                if masks_for_frame:
                    masks_np = np.stack(masks_for_frame, axis=0)
                    boxes_t = torch.from_numpy(masks_to_boxes_np(masks_np)).float()
                    scores_t = torch.as_tensor(scores_for_frame, dtype=torch.float32)
                    keep = nms(boxes_t, scores_t, iou_threshold=float(args.box_nms_thresh)).cpu().numpy()
                    masks_np = masks_np[keep.astype(np.int64)]
                else:
                    masks_np = np.zeros((0, h, w), dtype=bool)
                disjoint_np = disjoin_smallest_first(
                    masks_np,
                    h,
                    w,
                    empty_ratio=float(args.empty_ratio),
                    fix_small_regions=bool(args.fix_small_regions),
                )
                label = label_from_masks(disjoint_np, h, w)
                label_path = label_dir / f"frame_{int(frame_id):06d}.png"
                cv2.imwrite(str(label_path), label)
                overlay = overlay_label(rgb, label)
                stats = mask_stats(label)
                runtime_sec = time.time() - frame_t0
                annotated = annotate_frame(
                    overlay,
                    f"{provider} frame {chunk_idx:02d} / id {int(frame_id)}",
                    [f"masks={stats['visible_id_count']} fg={stats['foreground_ratio']:.3f}", f"time={runtime_sec:.2f}s points={int(args.num_pts)}"],
                )
                overlay_path = overlay_dir / f"frame_{chunk_idx:02d}_id_{int(frame_id):06d}.jpg"
                annotated.save(overlay_path, quality=95)
                overlay_paths.append(overlay_path)
                record = {
                    "schema_version": "stream4d_v105_4dpm_style_per_frame_row_v1",
                    "provider": provider,
                    "scene_id": args.scene_id,
                    "chunk_frame_index": int(chunk_idx),
                    "frame_id": int(frame_id),
                    "rgb_path": str(rgb_path),
                    "rgb_sha256": sha256_file(rgb_path),
                    "label_path": str(label_path),
                    "overlay_path": str(overlay_path),
                    "native_rgb_height": int(h),
                    "native_rgb_width": int(w),
                    "num_pts": int(args.num_pts),
                    "points_per_batch": int(args.sam31_points_per_batch),
                    "point_sampling": point_meta,
                    "sam31_confidence_threshold": float(args.sam31_confidence_threshold),
                    "sam31_quality_threshold": float(args.sam31_quality_threshold),
                    "sam31_per_point_selection_policy": str(args.sam31_per_point_selection_policy),
                    "box_nms_thresh": float(args.box_nms_thresh),
                    "raw_multimask_option_count": int(raw_option_count),
                    "prompt_with_selected_mask_count": int(len(masks_for_frame)),
                    "prompt_batch_failure_count": int(batch_failure_count),
                    "post_nms_mask_count": int(masks_np.shape[0]),
                    "post_disjoint_mask_count": int(disjoint_np.shape[0]),
                    "runtime_sec": float(runtime_sec),
                    "stats": stats,
                }
                records.append(record)
                print(json.dumps({k: record[k] for k in ("provider", "frame_id", "prompt_with_selected_mask_count", "post_disjoint_mask_count", "runtime_sec")}), flush=True)
        sheet_paths = []
        for start, end in [(0, 16), (16, 32)]:
            part = overlay_paths[start:end]
            if not part:
                continue
            sheet_path = sheet_dir / f"{provider}_{args.scene_id}_chunk_frames_{start:02d}_{start + len(part) - 1:02d}_4x4.jpg"
            make_sheet(part, sheet_path, int(args.sheet_cell_width))
            sheet_paths.append(str(sheet_path))
        runtimes = [float(row["runtime_sec"]) for row in records]
        summary = {
            "schema_version": "stream4d_v105_4dpm_style_per_frame_summary_v1",
            "provider": provider,
            "scene_id": args.scene_id,
            "frame_count": int(len(records)),
            "frame_ids": [int(row["frame_id"]) for row in records],
            "tracking_used": False,
            "video_predictor_used": False,
            "call": "SAM3.1 image model predict_inst per frame using the same 4D_PM-style seeded random point prompts; multimask_output=True, best-score per point by default, box NMS, small-region fix, smallest-first disjoin",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "num_pts": int(args.num_pts),
            "points_per_batch": int(args.sam31_points_per_batch),
            "sam31_confidence_threshold": float(args.sam31_confidence_threshold),
            "sam31_quality_threshold": float(args.sam31_quality_threshold),
            "sam31_per_point_selection_policy": str(args.sam31_per_point_selection_policy),
            "box_nms_thresh": float(args.box_nms_thresh),
            "empty_ratio": float(args.empty_ratio),
            "model_init_runtime_sec": float(init_sec),
            "mean_runtime_sec_per_frame": float(np.mean(runtimes)) if runtimes else 0.0,
            "median_runtime_sec_per_frame": float(np.median(runtimes)) if runtimes else 0.0,
            "min_runtime_sec_per_frame": float(np.min(runtimes)) if runtimes else 0.0,
            "max_runtime_sec_per_frame": float(np.max(runtimes)) if runtimes else 0.0,
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0),
            "sheet_paths": sheet_paths,
            "records": records,
        }
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"summary": str(summary_path), "sheets": sheet_paths, "mean_runtime_sec_per_frame": summary["mean_runtime_sec_per_frame"]}, indent=2), flush=True)
        return summary
    finally:
        sys.path[:] = old_sys_path


def run_sam31_official(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torchvision.ops import nms

    provider = "sam31_official_4dpm_points"
    root = REPO_ROOT / "third_party" / "sam3"
    checkpoint = REPO_ROOT / "ckpts" / "SAM3" / "sam3.1_multiplex.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    old_sys_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        os.environ.setdefault("LOG_LEVEL", "WARNING")
        from sam3 import build_sam3_predictor

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        t_init = time.time()
        predictor = build_sam3_predictor(
            version="sam3.1",
            checkpoint_path=str(checkpoint),
            compile=False,
            async_loading_frames=False,
            max_num_objects=int(args.sam31_official_max_objects),
            multiplex_count=int(args.sam31_official_multiplex_count),
            use_fa3=False,
        )
        init_sec = time.time() - t_init
        output_root = Path(args.output_root).resolve() / provider
        label_dir = output_root / "labels"
        overlay_dir = output_root / "overlays"
        sheet_dir = output_root / "sheets"
        for directory in (label_dir, overlay_dir, sheet_dir):
            directory.mkdir(parents=True, exist_ok=True)
        rgb_root = Path(args.rgb_root).resolve() / args.scene_id / "color"
        records = []
        overlay_paths = []
        with torch.inference_mode():
            for chunk_idx, frame_id in enumerate(parse_frame_ids(args.frame_ids, args.frame_start, args.frame_stride, args.frame_count)):
                frame_t0 = time.time()
                rgb_path = rgb_root / f"{int(frame_id)}.jpg"
                rgb = read_rgb(rgb_path)
                h, w = rgb.shape[:2]
                frame_seed = stable_seed(args.seed, args.scene_id, int(frame_id), int(args.num_pts), args.point_mode)
                points_yx, point_meta = make_points_yx_torch(int(args.num_pts), frame_seed, args.point_mode)
                points_xy_rel = points_yx_to_xy_rel_np(points_yx)
                prompt_rows = []
                collected_masks = []
                collected_scores = []
                collected_obj_ids = []
                session_batch_size = max(1, int(args.sam31_official_session_batch_size))
                for batch_start in range(0, int(args.num_pts), session_batch_size):
                    batch_points = points_xy_rel[batch_start : batch_start + session_batch_size]
                    out = None
                    with tempfile.TemporaryDirectory(prefix=f"sam31_official_{args.scene_id}_{int(frame_id):06d}_{batch_start:04d}_") as tmp:
                        frame_dir = Path(tmp)
                        os.symlink(rgb_path, frame_dir / "000000.jpg")
                        response = predictor.handle_request({"type": "start_session", "resource_path": str(frame_dir)})
                        session_id = response["session_id"]
                        try:
                            for local_point_idx, point_xy in enumerate(batch_points):
                                point_idx = int(batch_start + local_point_idx)
                                prompt_t0 = time.time()
                                try:
                                    out = predictor.handle_request(
                                        {
                                            "type": "add_prompt",
                                            "session_id": session_id,
                                            "frame_index": 0,
                                            "points": torch.tensor(point_xy.reshape(1, 2), dtype=torch.float32),
                                            "point_labels": torch.tensor([1], dtype=torch.int32),
                                            "obj_id": int(local_point_idx + 1),
                                        }
                                    )["outputs"]
                                    masks_tmp = np.asarray(out.get("out_binary_masks", []))
                                    prompt_rows.append(
                                        {
                                            "point_idx": point_idx,
                                            "session_batch_start": int(batch_start),
                                            "runtime_sec": float(time.time() - prompt_t0),
                                            "returned_mask_count": int(masks_tmp.shape[0]) if masks_tmp.ndim >= 3 else 0,
                                        }
                                    )
                                except Exception as exc:
                                    prompt_rows.append(
                                        {
                                            "point_idx": point_idx,
                                            "session_batch_start": int(batch_start),
                                            "runtime_sec": float(time.time() - prompt_t0),
                                            "error": repr(exc),
                                        }
                                    )
                        finally:
                            try:
                                predictor.handle_request({"type": "close_session", "session_id": session_id})
                            except Exception:
                                pass
                            torch.cuda.empty_cache()
                    if out is None:
                        continue
                    batch_masks = np.asarray(out.get("out_binary_masks", []))
                    if batch_masks.ndim == 2:
                        batch_masks = batch_masks[None, :, :]
                    if batch_masks.ndim != 3 or batch_masks.shape[0] == 0:
                        continue
                    batch_masks = batch_masks.astype(bool)
                    batch_scores = np.asarray(out.get("out_probs", np.ones((batch_masks.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
                    if batch_scores.size != batch_masks.shape[0]:
                        batch_scores = np.ones((batch_masks.shape[0],), dtype=np.float32)
                    batch_obj_ids = np.asarray(out.get("out_obj_ids", np.arange(1, batch_masks.shape[0] + 1)), dtype=np.int64).reshape(-1)
                    if batch_obj_ids.size != batch_masks.shape[0]:
                        batch_obj_ids = np.arange(1, batch_masks.shape[0] + 1, dtype=np.int64)
                    collected_masks.append(batch_masks)
                    collected_scores.append(batch_scores)
                    collected_obj_ids.append(batch_obj_ids + int(batch_start))
                if collected_masks:
                    masks_np = np.concatenate(collected_masks, axis=0).astype(bool)
                    scores_np = np.concatenate(collected_scores, axis=0).astype(np.float32)
                    obj_ids_np = np.concatenate(collected_obj_ids, axis=0).astype(np.int64)
                else:
                    masks_np = np.zeros((0, h, w), dtype=bool)
                    scores_np = np.zeros((0,), dtype=np.float32)
                    obj_ids_np = np.zeros((0,), dtype=np.int64)
                collected_mask_count_pre_nms = int(masks_np.shape[0])
                if masks_np.shape[0] > 0:
                    boxes_t = torch.from_numpy(masks_to_boxes_np(masks_np)).float()
                    scores_t = torch.from_numpy(scores_np.astype(np.float32)).float()
                    keep = nms(boxes_t, scores_t, iou_threshold=float(args.box_nms_thresh)).cpu().numpy().astype(np.int64)
                    masks_np = masks_np[keep]
                    scores_np = scores_np[keep]
                    obj_ids_np = obj_ids_np[keep]
                disjoint_np = disjoin_smallest_first(
                    masks_np,
                    h,
                    w,
                    empty_ratio=float(args.empty_ratio),
                    fix_small_regions=bool(args.fix_small_regions),
                )
                label = label_from_masks(disjoint_np, h, w)
                label_path = label_dir / f"frame_{int(frame_id):06d}.png"
                cv2.imwrite(str(label_path), label)
                overlay = overlay_label(rgb, label)
                stats = mask_stats(label)
                runtime_sec = time.time() - frame_t0
                annotated = annotate_frame(
                    overlay,
                    f"{provider} frame {chunk_idx:02d} / id {int(frame_id)}",
                    [f"masks={stats['visible_id_count']} fg={stats['foreground_ratio']:.3f}", f"time={runtime_sec:.2f}s points={int(args.num_pts)}"],
                )
                overlay_path = overlay_dir / f"frame_{chunk_idx:02d}_id_{int(frame_id):06d}.jpg"
                annotated.save(overlay_path, quality=95)
                overlay_paths.append(overlay_path)
                record = {
                    "schema_version": "stream4d_v105_4dpm_style_per_frame_row_v1",
                    "provider": provider,
                    "scene_id": args.scene_id,
                    "chunk_frame_index": int(chunk_idx),
                    "frame_id": int(frame_id),
                    "rgb_path": str(rgb_path),
                    "rgb_sha256": sha256_file(rgb_path),
                    "label_path": str(label_path),
                    "overlay_path": str(overlay_path),
                    "native_rgb_height": int(h),
                    "native_rgb_width": int(w),
                    "num_pts": int(args.num_pts),
                    "point_sampling": point_meta,
                    "sam31_official_max_objects": int(args.sam31_official_max_objects),
                    "sam31_official_multiplex_count": int(args.sam31_official_multiplex_count),
                    "sam31_official_session_batch_size": int(args.sam31_official_session_batch_size),
                    "use_fa3": False,
                    "box_nms_thresh": float(args.box_nms_thresh),
                    "raw_prompt_count": int(args.num_pts),
                    "prompt_success_count": int(sum("error" not in row for row in prompt_rows)),
                    "prompt_error_count": int(sum("error" in row for row in prompt_rows)),
                    "official_collected_mask_count_pre_nms": int(collected_mask_count_pre_nms),
                    "official_returned_mask_count": int(len(obj_ids_np)),
                    "official_returned_obj_ids_after_nms": [int(v) for v in obj_ids_np[:200]],
                    "prompt_error_examples": [row for row in prompt_rows if "error" in row][:5],
                    "post_disjoint_mask_count": int(disjoint_np.shape[0]),
                    "runtime_sec": float(runtime_sec),
                    "mean_prompt_runtime_sec": float(np.mean([row["runtime_sec"] for row in prompt_rows])) if prompt_rows else 0.0,
                    "stats": stats,
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            k: record[k]
                            for k in (
                                "provider",
                                "frame_id",
                                "prompt_success_count",
                                "prompt_error_count",
                                "official_returned_mask_count",
                                "post_disjoint_mask_count",
                                "runtime_sec",
                            )
                        }
                    ),
                    flush=True,
                )
        sheet_paths = []
        for start, end in [(0, 16), (16, 32)]:
            part = overlay_paths[start:end]
            if not part:
                continue
            sheet_path = sheet_dir / f"{provider}_{args.scene_id}_chunk_frames_{start:02d}_{start + len(part) - 1:02d}_4x4.jpg"
            make_sheet(part, sheet_path, int(args.sheet_cell_width))
            sheet_paths.append(str(sheet_path))
        runtimes = [float(row["runtime_sec"]) for row in records]
        summary = {
            "schema_version": "stream4d_v105_4dpm_style_per_frame_summary_v1",
            "provider": provider,
            "scene_id": args.scene_id,
            "frame_count": int(len(records)),
            "frame_ids": [int(row["frame_id"]) for row in records],
            "tracking_used": False,
            "video_predictor_used": True,
            "call": "Official SAM3.1 multiplex predictor point interactivity per frame: batched single-frame sessions per RGB, 4D_PM-style seeded random positive point prompts, one obj_id per point inside each session batch, no propagation, concatenate proposals, box NMS, small-region fix, smallest-first disjoin",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "num_pts": int(args.num_pts),
            "sam31_official_max_objects": int(args.sam31_official_max_objects),
            "sam31_official_multiplex_count": int(args.sam31_official_multiplex_count),
            "sam31_official_session_batch_size": int(args.sam31_official_session_batch_size),
            "use_fa3": False,
            "box_nms_thresh": float(args.box_nms_thresh),
            "empty_ratio": float(args.empty_ratio),
            "model_init_runtime_sec": float(init_sec),
            "mean_runtime_sec_per_frame": float(np.mean(runtimes)) if runtimes else 0.0,
            "median_runtime_sec_per_frame": float(np.median(runtimes)) if runtimes else 0.0,
            "min_runtime_sec_per_frame": float(np.min(runtimes)) if runtimes else 0.0,
            "max_runtime_sec_per_frame": float(np.max(runtimes)) if runtimes else 0.0,
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0),
            "sheet_paths": sheet_paths,
            "records": records,
        }
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"summary": str(summary_path), "sheets": sheet_paths, "mean_runtime_sec_per_frame": summary["mean_runtime_sec_per_frame"]}, indent=2), flush=True)
        return summary
    finally:
        sys.path[:] = old_sys_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["sam2", "edgetam", "sam31", "sam31_official"])
    parser.add_argument("--scene-id", default="scene0011_00")
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--rgb-root", default=str(DEFAULT_RGB_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--num-pts", type=int, default=800)
    parser.add_argument("--point-mode", default="random_seeded", choices=["random_seeded", "4dpm_grid"])
    parser.add_argument("--seed", type=int, default=105)
    parser.add_argument("--points-per-batch", type=int, default=128)
    parser.add_argument("--sam31-points-per-batch", type=int, default=64)
    parser.add_argument("--iou-threshold", type=float, default=0.8)
    parser.add_argument("--stability-threshold", type=float, default=0.8)
    parser.add_argument("--stability-score-offset", type=float, default=1.0)
    parser.add_argument("--box-nms-thresh", type=float, default=0.8)
    parser.add_argument("--empty-ratio", type=float, default=0.001)
    parser.add_argument("--fix-small-regions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam31-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--sam31-quality-threshold", type=float, default=0.0)
    parser.add_argument("--sam31-per-point-selection-policy", default="best_score", choices=["best_score", "smallest"])
    parser.add_argument("--sam31-official-max-objects", type=int, default=800)
    parser.add_argument("--sam31-official-multiplex-count", type=int, default=16)
    parser.add_argument("--sam31-official-session-batch-size", type=int, default=128)
    parser.add_argument("--sheet-cell-width", type=int, default=520)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this audit")
    if args.provider in {"sam2", "edgetam"}:
        run_sam2_like(args, source_key=args.provider)
    elif args.provider == "sam31":
        run_sam31(args)
    elif args.provider == "sam31_official":
        run_sam31_official(args)
    else:
        raise ValueError(args.provider)


if __name__ == "__main__":
    main()
