#!/usr/bin/env python3
"""Audit v105 first-frame segmentor outputs before pipeline filtering.

This script intentionally runs one provider per process. SAM2 and EdgeTAM both
use the module name ``sam2`` from different source trees, so separate processes
avoid silently mixing implementations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
DEFAULT_OUT = STREAM3D_ROOT / "outputs" / "audit" / "v105_first_frame_segmentor_audit_r3_raw_native"


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


def label_from_masks(masks: np.ndarray, h: int, w: int, order: str = "large_to_small") -> np.ndarray:
    if masks.ndim == 2:
        masks = masks[None, :, :]
    bool_masks = masks.astype(bool)
    areas = np.count_nonzero(bool_masks.reshape(bool_masks.shape[0], -1), axis=1)
    if order == "small_to_large":
        indices = np.argsort(areas)
    elif order == "score":
        indices = np.arange(bool_masks.shape[0])
    else:
        indices = np.argsort(-areas)
    label = np.zeros((h, w), dtype=np.uint16)
    for out_id, idx in enumerate(indices, start=1):
        mask = bool_masks[int(idx)]
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


def save_overlay(rgb: np.ndarray, label: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_label(rgb, label)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def mask_stats(label: np.ndarray) -> dict[str, Any]:
    ids = [int(v) for v in np.unique(label) if int(v) > 0]
    rows = []
    for obj_id in ids:
        ys, xs = np.where(label == obj_id)
        if xs.size == 0:
            continue
        rows.append(
            {
                "id": int(obj_id),
                "area": int(xs.size),
                "area_ratio": float(xs.size) / float(label.size),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            }
        )
    rows.sort(key=lambda row: int(row["area"]), reverse=True)
    return {
        "visible_id_count": int(len(ids)),
        "foreground_pixels": int(np.count_nonzero(label > 0)),
        "foreground_ratio": float(np.count_nonzero(label > 0)) / float(label.size),
        "top_masks": rows[:12],
    }


def write_record(out_dir: Path, provider: str, payload: dict[str, Any]) -> None:
    path = out_dir / f"{provider}_record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def deterministic_points(num_pts: int, h: int, w: int, seed: int) -> "Any":
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    return torch.rand((num_pts, 2), device="cuda", generator=generator) * 2.0 - 1.0


def four_dpm_grid_points(num_pts: int) -> "Any":
    import torch

    side = int(torch.ceil(torch.sqrt(torch.tensor(float(max(num_pts, 1)), device="cuda"))).item())
    coords = torch.linspace(-0.95, 0.95, side, device="cuda")
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)[:num_pts]


def make_4dpm_points(num_pts: int, h: int, w: int, seed: int, point_mode: str) -> tuple["Any", dict[str, Any]]:
    mode = str(point_mode).strip().lower()
    if mode == "4dpm_grid":
        points = four_dpm_grid_points(num_pts)
        side = int(np.ceil(np.sqrt(float(max(num_pts, 1)))))
        return points, {
            "point_mode": "4dpm_grid",
            "grid_side": int(side),
            "coordinate_range": [-0.95, 0.95],
            "source": "third_party/4D_PM/frontend/segment/infer.py::_deterministic_keypoint_grid",
        }
    if mode == "random_seeded":
        return deterministic_points(num_pts, h, w, seed), {
            "point_mode": "random_seeded",
            "seed": int(seed),
            "coordinate_range": [-1.0, 1.0],
            "source": "third_party/4D_PM/frontend/segment/infer.py random fallback",
        }
    raise ValueError(f"unsupported 4DPM point mode: {point_mode}")


def run_fastsam(args: argparse.Namespace) -> None:
    import torch
    from ultralytics import FastSAM

    out_dir = Path(args.output_root).resolve()
    imgsz = int(args.fastsam_imgsz)
    conf = float(args.fastsam_conf)
    iou = float(args.fastsam_iou)
    if imgsz == 1024 and abs(conf - 0.4) < 1e-9 and abs(iou - 0.9) < 1e-9:
        provider = "fastsam_native_official"
    else:
        provider = f"fastsam_native_official_i{imgsz}_c{int(round(conf * 100)):03d}_nms{int(round(iou * 100)):03d}"
    rgb_path = Path(args.rgb_path)
    rgb = read_rgb(rgb_path)
    h, w = rgb.shape[:2]
    t0 = time.time()
    checkpoint = REPO_ROOT / "FastSAM-x.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = FastSAM(str(checkpoint))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    result = model(str(rgb_path), device=0 if torch.cuda.is_available() else "cpu", imgsz=imgsz, conf=conf, iou=iou, retina_masks=True, verbose=False)[0]
    if result.masks is None:
        masks = np.zeros((0, h, w), dtype=bool)
        scores = np.zeros((0,), dtype=np.float32)
    else:
        masks = result.masks.data.detach().cpu().numpy() > 0.5
        if masks.ndim == 2:
            masks = masks[None, :, :]
        if result.boxes is not None and getattr(result.boxes, "conf", None) is not None:
            scores = result.boxes.conf.detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            scores = np.ones((masks.shape[0],), dtype=np.float32)
    label = label_from_masks(masks, h, w, order="large_to_small")
    mask_path = out_dir / f"{provider}_label.png"
    cv2.imwrite(str(mask_path), label)
    overlay_path = out_dir / f"{provider}_overlay.jpg"
    save_overlay(rgb, label, overlay_path)
    peak_mb: float | str = ""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    write_record(
        out_dir,
        provider,
        {
            "provider": provider,
            "status": "completed",
            "input_policy": "native_scannet_rgb_no_pipeline_resize",
            "rgb_path": str(rgb_path),
            "rgb_sha256": sha256_file(rgb_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "call": f"ultralytics.FastSAM(FastSAM-x.pt)(native_rgb, imgsz={imgsz}, conf={conf}, iou={iou}, retina_masks=True)",
            "fastsam_imgsz": imgsz,
            "fastsam_conf": conf,
            "fastsam_iou": iou,
            "raw_mask_count": int(masks.shape[0]),
            "score_min": float(scores.min()) if scores.size else 0.0,
            "score_max": float(scores.max()) if scores.size else 0.0,
            "label_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "runtime_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_mb,
            "stats": mask_stats(label),
        },
    )


def _parse_bool4(value: str) -> list[bool]:
    parts = [part.strip().lower() for part in str(value).replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError(f"expected four comma-separated booleans for T,B,L,R edges, got {value!r}")
    truthy = {"1", "true", "yes", "y", "allow", "t"}
    falsy = {"0", "false", "no", "n", "block", "f"}
    parsed = []
    for part in parts:
        if part in truthy:
            parsed.append(True)
        elif part in falsy:
            parsed.append(False)
        else:
            raise ValueError(f"invalid boolean token in edge policy: {part!r}")
    return parsed


def _load_binary_mask(path_value: str, h: int, w: int) -> np.ndarray | None:
    if not str(path_value).strip():
        return None
    path = _resolve_existing_path(path_value)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.int8)


def _fastsam_mask_rows(masks: np.ndarray, edge_width: int) -> list[dict[str, Any]]:
    rows = []
    if masks.size == 0:
        return rows
    for idx in range(int(masks.shape[0])):
        mask = masks[idx].astype(bool)
        area = int(np.count_nonzero(mask))
        if area <= 0:
            rows.append(
                {
                    "raw_index": int(idx),
                    "area": 0,
                    "area_ratio": 0.0,
                    "bbox_xyxy": [],
                    "touches_tblr": [False, False, False, False],
                }
            )
            continue
        ys, xs = np.where(mask)
        ew = max(1, int(edge_width))
        rows.append(
            {
                "raw_index": int(idx),
                "area": area,
                "area_ratio": float(area) / float(mask.size),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "touches_tblr": [
                    bool(np.any(mask[:ew, :])),
                    bool(np.any(mask[-ew:, :])),
                    bool(np.any(mask[:, :ew])),
                    bool(np.any(mask[:, -ew:])),
                ],
            }
        )
    rows.sort(key=lambda row: int(row["area"]), reverse=True)
    return rows


def _filter_fastsam_roman_masks(
    masks: np.ndarray,
    *,
    h: int,
    w: int,
    min_area: float,
    max_area: float,
    allow_tblr_edges: list[bool],
    edge_width: int,
    ignore_mask: np.ndarray | None,
    keep_mask: np.ndarray | None,
    keep_minimal_intersection: float,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    kept: list[np.ndarray] = []
    decisions: list[dict[str, Any]] = []
    removal_counts = {
        "empty": 0,
        "edge": 0,
        "ignore_mask_intersection": 0,
        "keep_mask_intersection": 0,
        "area_too_small": 0,
        "area_too_large": 0,
        "kept": 0,
    }
    if masks.ndim == 2:
        masks = masks[None, :, :]
    for idx in range(int(masks.shape[0])):
        mask = masks[idx].astype(bool)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(np.count_nonzero(mask))
        ew = max(1, int(edge_width))
        touches = [
            bool(np.any(mask[:ew, :])),
            bool(np.any(mask[-ew:, :])),
            bool(np.any(mask[:, :ew])),
            bool(np.any(mask[:, -ew:])),
        ]
        reason = "kept"
        keep_intersection_ratio = None
        if area <= 0:
            reason = "empty"
        elif any(touch and not allow for touch, allow in zip(touches, allow_tblr_edges)):
            reason = "edge"
        elif ignore_mask is not None and bool(np.any(mask & ignore_mask.astype(bool))):
            reason = "ignore_mask_intersection"
        elif keep_mask is not None:
            keep_intersection_ratio = float(np.count_nonzero(mask & keep_mask.astype(bool))) / float(max(area, 1))
            if keep_intersection_ratio < float(keep_minimal_intersection):
                reason = "keep_mask_intersection"
        if reason == "kept" and area < float(min_area):
            reason = "area_too_small"
        if reason == "kept" and area > float(max_area):
            reason = "area_too_large"
        if reason == "kept":
            kept.append(mask)
        removal_counts[reason] += 1
        decisions.append(
            {
                "raw_index": int(idx),
                "decision": reason,
                "area": area,
                "area_ratio": float(area) / float(h * w),
                "touches_tblr": touches,
                "keep_intersection_ratio": keep_intersection_ratio,
            }
        )
    if kept:
        return np.stack(kept, axis=0), decisions, removal_counts
    return np.zeros((0, h, w), dtype=bool), decisions, removal_counts


def run_fastsam_roman(args: argparse.Namespace) -> None:
    import torch
    from ultralytics import FastSAM

    out_dir = Path(args.output_root).resolve()
    imgsz = int(args.fastsam_imgsz)
    conf = float(args.fastsam_conf)
    iou = float(args.fastsam_iou)
    min_div = float(args.fastsam_roman_min_mask_len_div)
    max_div = float(args.fastsam_roman_max_mask_len_div)
    edge_width = int(args.fastsam_roman_edge_width)
    allow_tblr_edges = _parse_bool4(args.fastsam_roman_allow_tblr_edges)
    edge_token = "".join("1" if value else "0" for value in allow_tblr_edges)
    provider = (
        f"fastsam_roman_i{imgsz}_c{int(round(conf * 100)):03d}_nms{int(round(iou * 100)):03d}"
        f"_mind{int(round(min_div))}_maxd{int(round(max_div))}_e{edge_token}"
    )
    rgb_path = Path(args.rgb_path)
    rgb = read_rgb(rgb_path)
    h, w = rgb.shape[:2]
    image_area = float(h * w)
    min_area = image_area / (min_div**2) if min_div > 0 else 0.0
    max_area = image_area / (max_div**2) if max_div > 0 else float("inf")
    if float(args.fastsam_roman_min_area) >= 0:
        min_area = float(args.fastsam_roman_min_area)
    if float(args.fastsam_roman_max_area) >= 0:
        max_area = float(args.fastsam_roman_max_area)
    ignore_mask = _load_binary_mask(args.fastsam_roman_ignore_mask, h, w)
    keep_mask = _load_binary_mask(args.fastsam_roman_keep_mask, h, w)
    t0 = time.time()
    checkpoint = REPO_ROOT / "FastSAM-x.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = FastSAM(str(checkpoint))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    result = model(str(rgb_path), device=0 if torch.cuda.is_available() else "cpu", imgsz=imgsz, conf=conf, iou=iou, retina_masks=True, verbose=False)[0]
    if result.masks is None:
        raw_masks = np.zeros((0, h, w), dtype=bool)
        scores = np.zeros((0,), dtype=np.float32)
    else:
        raw_masks = result.masks.data.detach().cpu().numpy() > 0.5
        if raw_masks.ndim == 2:
            raw_masks = raw_masks[None, :, :]
        if result.boxes is not None and getattr(result.boxes, "conf", None) is not None:
            scores = result.boxes.conf.detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            scores = np.ones((raw_masks.shape[0],), dtype=np.float32)
    filtered_masks, filter_decisions, removal_counts = _filter_fastsam_roman_masks(
        raw_masks,
        h=h,
        w=w,
        min_area=min_area,
        max_area=max_area,
        allow_tblr_edges=allow_tblr_edges,
        edge_width=edge_width,
        ignore_mask=ignore_mask,
        keep_mask=keep_mask,
        keep_minimal_intersection=float(args.fastsam_roman_keep_minimal_intersection),
    )
    label = label_from_masks(filtered_masks, h, w, order=str(args.fastsam_roman_label_order))
    mask_path = out_dir / f"{provider}_label.png"
    cv2.imwrite(str(mask_path), label)
    overlay_path = out_dir / f"{provider}_overlay.jpg"
    save_overlay(rgb, label, overlay_path)
    peak_mb: float | str = ""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    write_record(
        out_dir,
        provider,
        {
            "provider": provider,
            "status": "completed",
            "input_policy": "native_scannet_rgb_no_pipeline_resize",
            "rgb_path": str(rgb_path),
            "rgb_sha256": sha256_file(rgb_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "call": (
                "ROMAN-style FastSAM-x everything route: ultralytics.FastSAM raw masks equivalent to "
                "FastSAMPrompt.everything_prompt(), then area/edge/optional ignore-keep postprocess"
            ),
            "reference": {
                "roman_wrapper": "https://github.com/mit-acl/ROMAN/blob/main/roman/map/fastsam_wrapper.py",
                "official_fastsam_prompt": "https://github.com/CASIA-IVA-Lab/FastSAM/blob/main/fastsam/prompt.py",
            },
            "fastsam_imgsz": imgsz,
            "fastsam_conf": conf,
            "fastsam_iou": iou,
            "raw_mask_count": int(raw_masks.shape[0]),
            "filtered_mask_count": int(filtered_masks.shape[0]),
            "score_min": float(scores.min()) if scores.size else 0.0,
            "score_max": float(scores.max()) if scores.size else 0.0,
            "roman_filter": {
                "min_mask_len_div": min_div,
                "max_mask_len_div": max_div,
                "min_area": float(min_area),
                "max_area": float(max_area),
                "edge_width": edge_width,
                "allow_tblr_edges": allow_tblr_edges,
                "ignore_mask_path": str(args.fastsam_roman_ignore_mask),
                "keep_mask_path": str(args.fastsam_roman_keep_mask),
                "keep_mask_minimal_intersection": float(args.fastsam_roman_keep_minimal_intersection),
                "label_order": str(args.fastsam_roman_label_order),
                "removal_counts": removal_counts,
            },
            "raw_mask_rows": _fastsam_mask_rows(raw_masks, edge_width=edge_width),
            "filter_decisions": filter_decisions,
            "label_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "runtime_sec": time.time() - t0,
            "peak_gpu_memory_mb": peak_mb,
            "stats": mask_stats(label),
        },
    )


def _run_sam2_like_4dpm(
    *,
    args: argparse.Namespace,
    provider: str,
    repo_path: Path,
    checkpoint: Path,
    model_cfg: str,
    clear_sam2_modules: bool,
) -> None:
    import torch
    from torchvision.ops import nms

    out_dir = Path(args.output_root).resolve()
    if clear_sam2_modules:
        for name in list(sys.modules):
            if name == "sam2" or name.startswith("sam2."):
                del sys.modules[name]
    old_cwd = Path.cwd()
    sys.path.insert(0, str(repo_path))
    if provider.startswith("edgetam"):
        os.chdir(repo_path)
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.utils.amg import batched_mask_to_box, calculate_stability_score

        rgb_path = Path(args.rgb_path)
        rgb = read_rgb(rgb_path)
        h, w = rgb.shape[:2]
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        image_model = build_sam2(model_cfg, str(checkpoint), device="cuda")
        predictor = SAM2ImagePredictor(image_model)
        num_pts = int(args.num_pts)
        points, point_record = make_4dpm_points(num_pts, h, w, int(args.seed), str(args.point_mode))
        provider = (
            f"{provider}_{point_record['point_mode']}_n{num_pts}"
            f"_iou{int(round(float(args.iou_threshold) * 100)):03d}"
            f"_stab{int(round(float(args.stability_threshold) * 100)):03d}"
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.reset_predictor()
            predictor.set_image(rgb)
            pts_px = 0.5 * (torch.tensor([h - 1, w - 1], device="cuda", dtype=torch.float32)) * (points + 1.0)
            pts_px = pts_px.round().long().flip(-1).float()
            coords = predictor._transforms.transform_coords(pts_px.unsqueeze(1), normalize=True, orig_hw=(h, w))
            labels = torch.ones((num_pts, 1), dtype=torch.int, device="cuda")
            masks, iou_predictions, _ = predictor._predict(coords, labels, multimask_output=True, return_logits=True)
            stability = calculate_stability_score(masks, 0.0, 1.0)
            good = (iou_predictions > float(args.iou_threshold)) & (stability >= float(args.stability_threshold))
            areas = (masks > 0.0).sum(dim=(-1, -2), dtype=torch.int64)
            areas_masked = areas.clone()
            areas_masked[~good] = torch.iinfo(torch.int64).max
            per_prompt_has_good = good.any(dim=1)
            chosen_idx = areas_masked.argmin(dim=1)
            prompt_indices = torch.nonzero(per_prompt_has_good, as_tuple=False).flatten()
            selected = masks[prompt_indices, chosen_idx[prompt_indices]] > 0.0
            selected_scores = stability[prompt_indices, chosen_idx[prompt_indices]]
            if selected.shape[0] > 0:
                boxes = batched_mask_to_box(selected).float()
                keep = nms(boxes, selected_scores.float(), iou_threshold=float(args.box_nms_thresh))
                selected = selected[keep]
                selected_scores = selected_scores[keep]
                order = selected.sum(dim=(-1, -2)).argsort()
                selected = selected[order]
                selected_scores = selected_scores[order]
                claimed = torch.zeros((h, w), dtype=torch.bool, device="cuda")
                disjoint = []
                min_pixels = int(h * w * float(args.empty_ratio))
                for mask in selected:
                    residual = mask & ~claimed
                    if int(residual.sum().item()) > min_pixels:
                        disjoint.append(residual.detach().cpu().numpy().astype(bool))
                        claimed |= mask
                masks_np = np.stack(disjoint, axis=0) if disjoint else np.zeros((0, h, w), dtype=bool)
            else:
                masks_np = np.zeros((0, h, w), dtype=bool)
            peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        label = label_from_masks(masks_np, h, w, order="score")
        mask_path = out_dir / f"{provider}_label.png"
        cv2.imwrite(str(mask_path), label)
        overlay_path = out_dir / f"{provider}_overlay.jpg"
        save_overlay(rgb, label, overlay_path)
        write_record(
            out_dir,
            provider,
            {
                "provider": provider,
                "status": "completed",
                "input_policy": "native_scannet_rgb_no_pipeline_resize",
                "rgb_path": str(rgb_path),
                "rgb_sha256": sha256_file(rgb_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "model_cfg": model_cfg,
                "call": "4D_PM-style SAM point prompts: point grid/random in normalized yx coordinates, multimask, iou/stability filter, select-smallest per point, box NMS, smallest-first disjoint write",
                "num_pts": num_pts,
                "seed": int(args.seed),
                "point_sampling": point_record,
                "native_rgb_height": int(h),
                "native_rgb_width": int(w),
                "iou_threshold": float(args.iou_threshold),
                "stability_threshold": float(args.stability_threshold),
                "box_nms_thresh": float(args.box_nms_thresh),
                "raw_prompt_count": int(num_pts),
                "prompt_with_good_mask_count": int(per_prompt_has_good.sum().item()),
                "post_disjoint_mask_count": int(masks_np.shape[0]),
                "label_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "runtime_sec": time.time() - t0,
                "peak_gpu_memory_mb": peak_mb,
                "stats": mask_stats(label),
            },
        )
    finally:
        os.chdir(old_cwd)


def run_sam2_4dpm(args: argparse.Namespace) -> None:
    _run_sam2_like_4dpm(
        args=args,
        provider="sam2_native_4dpm_points",
        repo_path=REPO_ROOT / "Grounded-SAM-2",
        checkpoint=REPO_ROOT / "Grounded-SAM-2" / "checkpoints" / "sam2.1_hiera_large.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        clear_sam2_modules=True,
    )


def run_edgetam_4dpm(args: argparse.Namespace) -> None:
    _run_sam2_like_4dpm(
        args=args,
        provider="edgetam_native_4dpm_points",
        repo_path=REPO_ROOT / "third_party" / "EdgeTAM",
        checkpoint=REPO_ROOT / "third_party" / "EdgeTAM" / "checkpoints" / "edgetam.pt",
        model_cfg="edgetam.yaml",
        clear_sam2_modules=True,
    )


def _run_sam2_like_amg(
    *,
    args: argparse.Namespace,
    provider: str,
    repo_path: Path,
    checkpoint: Path,
    model_cfg: str,
) -> None:
    import torch

    out_dir = Path(args.output_root).resolve()
    for name in list(sys.modules):
        if name == "sam2" or name.startswith("sam2."):
            del sys.modules[name]
    old_cwd = Path.cwd()
    sys.path.insert(0, str(repo_path))
    if provider.startswith("edgetam"):
        os.chdir(repo_path)
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        rgb_path = Path(args.rgb_path)
        rgb = read_rgb(rgb_path)
        h, w = rgb.shape[:2]
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        model = build_sam2(model_cfg, str(checkpoint), device="cuda", apply_postprocessing=False)
        generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=64,
            points_per_batch=128,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25,
            use_m2m=True,
            output_mode="binary_mask",
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = list(generator.generate(rgb))
        masks_np = []
        scores = []
        for row in masks:
            mask = np.asarray(row.get("segmentation")).astype(bool)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            masks_np.append(mask)
            scores.append(float(row.get("predicted_iou", row.get("stability_score", 0.0)) or 0.0))
        if masks_np:
            order = np.argsort(-np.asarray(scores, dtype=np.float32))
            max_masks = min(len(order), int(args.max_masks_for_label))
            masks_arr = np.stack([masks_np[int(i)] for i in order[:max_masks]], axis=0)
        else:
            masks_arr = np.zeros((0, h, w), dtype=bool)
        label = label_from_masks(masks_arr, h, w, order="large_to_small")
        mask_path = out_dir / f"{provider}_label.png"
        cv2.imwrite(str(mask_path), label)
        overlay_path = out_dir / f"{provider}_overlay.jpg"
        save_overlay(rgb, label, overlay_path)
        peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        write_record(
            out_dir,
            provider,
            {
                "provider": provider,
                "status": "completed",
                "input_policy": "native_scannet_rgb_no_pipeline_resize",
                "rgb_path": str(rgb_path),
                "rgb_sha256": sha256_file(rgb_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "model_cfg": model_cfg,
                "call": "official notebook-style SAM2AutomaticMaskGenerator on native RGB: points_per_side=64, pred_iou=0.7, stability=0.92, crop_n_layers=1, use_m2m=True",
                "raw_mask_count": int(len(masks)),
                "max_masks_for_label": int(args.max_masks_for_label),
                "native_rgb_height": int(h),
                "native_rgb_width": int(w),
                "points_per_side": 64,
                "base_grid_point_count": 64 * 64,
                "points_per_batch": 128,
                "pred_iou_thresh": 0.7,
                "stability_score_thresh": 0.92,
                "stability_score_offset": 0.7,
                "crop_n_layers": 1,
                "box_nms_thresh": 0.7,
                "crop_n_points_downscale_factor": 2,
                "min_mask_region_area": 25,
                "use_m2m": True,
                "label_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "runtime_sec": time.time() - t0,
                "peak_gpu_memory_mb": peak_mb,
                "stats": mask_stats(label),
            },
        )
    finally:
        os.chdir(old_cwd)


def run_sam2_amg_tuned(args: argparse.Namespace) -> None:
    _run_sam2_like_amg(
        args=args,
        provider="sam2_native_official_amg_tuned",
        repo_path=REPO_ROOT / "Grounded-SAM-2",
        checkpoint=REPO_ROOT / "Grounded-SAM-2" / "checkpoints" / "sam2.1_hiera_large.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    )


def run_edgetam_amg_tuned(args: argparse.Namespace) -> None:
    _run_sam2_like_amg(
        args=args,
        provider="edgetam_native_official_amg_tuned",
        repo_path=REPO_ROOT / "third_party" / "EdgeTAM",
        checkpoint=REPO_ROOT / "third_party" / "EdgeTAM" / "checkpoints" / "edgetam.pt",
        model_cfg="edgetam.yaml",
    )


def run_sam3_points(args: argparse.Namespace) -> None:
    import torch

    provider = "sam3_native_predict_inst_points"
    rgb_path = Path(args.rgb_path)
    rgb = read_rgb(rgb_path)
    h, w = rgb.shape[:2]
    out_dir = Path(args.output_root).resolve()
    checkpoint = REPO_ROOT / "ckpts" / "SAM3" / "sam3.pt"
    root = REPO_ROOT / "third_party" / "sam3"
    old_sys_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        points = deterministic_points(int(args.num_pts), h, w, int(args.seed))
        points_px = points.detach().cpu().numpy().astype(np.float32)
        points_px[:, 0] = ((points_px[:, 0] + 1.0) * 0.5) * float(h - 1)
        points_px[:, 1] = ((points_px[:, 1] + 1.0) * 0.5) * float(w - 1)
        point_xy = points_px[:, ::-1].copy()
        masks_for_frame: list[np.ndarray] = []
        scores_for_frame: list[float] = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model = build_sam3_image_model(
                checkpoint_path=str(checkpoint),
                load_from_HF=False,
                device="cuda",
                compile=False,
                enable_inst_interactivity=True,
            )
            processor = Sam3Processor(model, confidence_threshold=0.0, device="cuda")
            state = processor.set_image(Image.fromarray(np.ascontiguousarray(rgb).astype(np.uint8)))
            batch = max(int(args.points_per_batch), 1)
            for start in range(0, point_xy.shape[0], batch):
                coords = point_xy[start : start + batch].reshape(-1, 1, 2).astype(np.float32)
                labels = np.ones((coords.shape[0], 1), dtype=np.int32)
                mask_options, score_values, _ = model.predict_inst(
                    state,
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=True,
                )
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
                for prompt_idx in range(mask_options_np.shape[0]):
                    option_scores = score_values_np[prompt_idx]
                    option_idx = int(np.argmax(option_scores))
                    mask = np.asarray(mask_options_np[prompt_idx, option_idx]).astype(bool)
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    masks_for_frame.append(mask)
                    scores_for_frame.append(float(option_scores[option_idx]))
        if masks_for_frame:
            masks_np = np.stack(masks_for_frame, axis=0)
            order = np.argsort(-np.asarray(scores_for_frame, dtype=np.float32))
            masks_np = masks_np[order[: min(len(order), int(args.max_masks_for_label))]]
        else:
            masks_np = np.zeros((0, h, w), dtype=bool)
        label = label_from_masks(masks_np, h, w, order="large_to_small")
        mask_path = out_dir / f"{provider}_label.png"
        cv2.imwrite(str(mask_path), label)
        overlay_path = out_dir / f"{provider}_overlay.jpg"
        save_overlay(rgb, label, overlay_path)
        peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        write_record(
            out_dir,
            provider,
            {
                "provider": provider,
                "status": "completed",
                "input_policy": "native_scannet_rgb_no_pipeline_resize",
                "rgb_path": str(rgb_path),
                "rgb_sha256": sha256_file(rgb_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "call": "SAM3 image model predict_inst on native RGB, 300 seeded positive points, multimask_output=True, best score per point",
                "num_pts": int(args.num_pts),
                "points_per_batch": int(args.points_per_batch),
                "raw_selected_prompt_mask_count": int(len(masks_for_frame)),
                "max_masks_for_label": int(args.max_masks_for_label),
                "score_min": float(min(scores_for_frame)) if scores_for_frame else 0.0,
                "score_max": float(max(scores_for_frame)) if scores_for_frame else 0.0,
                "label_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "runtime_sec": time.time() - t0,
                "peak_gpu_memory_mb": peak_mb,
                "stats": mask_stats(label),
            },
        )
    finally:
        sys.path[:] = old_sys_path


def _resolve_existing_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        (REPO_ROOT / path).resolve(),
        (STREAM3D_ROOT / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path_value)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return token or "vocab"


def load_sam3_text_prompts(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]], str]:
    prompts: list[str] = []
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []

    inline_prompts = [item.strip() for item in str(args.sam3_text_prompts).split(";") if item.strip()]
    if inline_prompts:
        added = 0
        for prompt in inline_prompts:
            key = prompt.casefold()
            if key in seen:
                continue
            seen.add(key)
            prompts.append(prompt)
            added += 1
        sources.append({"type": "inline_semicolon_list", "added_prompt_count": int(added)})

    provider_vocab_token = "text_concepts"
    prompts_file = str(args.sam3_text_prompts_file).strip()
    if prompts_file:
        path = _resolve_existing_path(prompts_file)
        column = str(args.sam3_vocab_column).strip()
        max_prompts = int(args.sam3_max_prompts)
        file_prompts: list[str] = []
        if path.suffix.lower() == ".tsv":
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                if reader.fieldnames is None or column not in reader.fieldnames:
                    raise ValueError(f"{path} does not contain SAM3 vocab column {column!r}; columns={reader.fieldnames}")
                for row in reader:
                    value = str(row.get(column, "")).strip()
                    if value:
                        file_prompts.append(value)
        else:
            with path.open("r", encoding="utf-8") as f:
                file_prompts = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]

        added = 0
        raw_count = len(file_prompts)
        for prompt in file_prompts:
            key = prompt.casefold()
            if key in seen:
                continue
            seen.add(key)
            prompts.append(prompt)
            added += 1
            if max_prompts > 0 and added >= max_prompts:
                break
        sources.append(
            {
                "type": "file",
                "path": str(path),
                "sha256": sha256_file(path),
                "column": column if path.suffix.lower() == ".tsv" else "",
                "raw_prompt_count": int(raw_count),
                "added_prompt_count": int(added),
                "max_prompts": int(max_prompts),
            }
        )
        stem = "scannet" if "scannet" in path.name.lower() or "scannet" in str(path).lower() else path.stem
        provider_vocab_token = f"{_safe_token(stem)}_{_safe_token(column)}" if path.suffix.lower() == ".tsv" else _safe_token(path.stem)

    if not prompts:
        raise ValueError("sam3_text_concepts requires --sam3-text-prompts or --sam3-text-prompts-file")
    return prompts, sources, provider_vocab_token


def run_sam3_text_concepts(args: argparse.Namespace) -> None:
    import torch

    prompts, prompt_sources, provider_vocab_token = load_sam3_text_prompts(args)
    confidence = float(args.sam3_confidence)
    provider = f"sam3_native_{provider_vocab_token}_t{int(round(confidence * 100)):03d}"
    rgb_path = Path(args.rgb_path)
    rgb = read_rgb(rgb_path)
    h, w = rgb.shape[:2]
    out_dir = Path(args.output_root).resolve()
    checkpoint = REPO_ROOT / "ckpts" / "SAM3" / "sam3.pt"
    root = REPO_ROOT / "third_party" / "sam3"
    old_sys_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mask_rows: list[dict[str, Any]] = []
        per_prompt_rows: list[dict[str, Any]] = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model = build_sam3_image_model(
                checkpoint_path=str(checkpoint),
                load_from_HF=False,
                device="cuda",
                compile=False,
                enable_inst_interactivity=False,
            )
            processor = Sam3Processor(model, confidence_threshold=confidence, device="cuda")
            state = processor.set_image(Image.fromarray(np.ascontiguousarray(rgb).astype(np.uint8)))
            for prompt in prompts:
                processor.reset_all_prompts(state)
                output = processor.set_text_prompt(prompt=prompt, state=state)
                masks_t = output.get("masks")
                scores_t = output.get("scores")
                boxes_t = output.get("boxes")
                if masks_t is None:
                    per_prompt_rows.append({"prompt": prompt, "raw_mask_count": 0})
                    continue
                masks_np = masks_t.detach().cpu().numpy().astype(bool)
                while masks_np.ndim >= 4 and 1 in masks_np.shape:
                    axis = [idx for idx, size in enumerate(masks_np.shape) if size == 1][0]
                    masks_np = np.squeeze(masks_np, axis=axis)
                if masks_np.ndim == 2:
                    masks_np = masks_np[None, :, :]
                scores_np = (
                    scores_t.detach().float().cpu().numpy().reshape(-1)
                    if scores_t is not None
                    else np.ones((masks_np.shape[0],), dtype=np.float32)
                )
                boxes_np = (
                    boxes_t.detach().float().cpu().numpy().reshape(-1, 4)
                    if boxes_t is not None
                    else np.zeros((masks_np.shape[0], 4), dtype=np.float32)
                )
                prompt_kept = 0
                for idx in range(int(masks_np.shape[0])):
                    mask = masks_np[int(idx)]
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    area = int(np.count_nonzero(mask))
                    if area <= 0:
                        continue
                    prompt_kept += 1
                    mask_rows.append(
                        {
                            "mask": mask,
                            "score": float(scores_np[int(idx)]) if int(idx) < scores_np.size else 0.0,
                            "prompt": prompt,
                            "source_index": int(idx),
                            "area": area,
                            "box_xyxy": boxes_np[int(idx)].astype(float).tolist() if int(idx) < boxes_np.shape[0] else [],
                        }
                    )
                per_prompt_rows.append(
                    {
                        "prompt": prompt,
                        "raw_mask_count": int(masks_np.shape[0]),
                        "nonempty_mask_count": int(prompt_kept),
                        "score_min": float(scores_np.min()) if scores_np.size else 0.0,
                        "score_max": float(scores_np.max()) if scores_np.size else 0.0,
                    }
                )
        selected: list[dict[str, Any]] = []
        duplicate_skip_count = 0
        for row in sorted(mask_rows, key=lambda item: (-float(item["score"]), -int(item["area"]), str(item["prompt"]), int(item["source_index"]))):
            mask = np.asarray(row["mask"]).astype(bool)
            duplicate = False
            for kept in selected:
                kept_mask = np.asarray(kept["mask"]).astype(bool)
                inter = float(np.count_nonzero(mask & kept_mask))
                if inter <= 0.0:
                    continue
                union = float(np.count_nonzero(mask | kept_mask))
                smaller = float(min(np.count_nonzero(mask), np.count_nonzero(kept_mask)))
                if union > 0.0 and (inter / union >= float(args.sam3_text_dedup_iou) or inter / max(smaller, 1.0) >= 0.95):
                    duplicate = True
                    break
            if duplicate:
                duplicate_skip_count += 1
                continue
            selected.append(row)
            if len(selected) >= int(args.max_masks_for_label):
                break
        if selected:
            masks_np = np.stack([np.asarray(row["mask"]).astype(bool) for row in selected], axis=0)
        else:
            masks_np = np.zeros((0, h, w), dtype=bool)
        label = label_from_masks(masks_np, h, w, order="large_to_small")
        mask_path = out_dir / f"{provider}_label.png"
        cv2.imwrite(str(mask_path), label)
        overlay_path = out_dir / f"{provider}_overlay.jpg"
        save_overlay(rgb, label, overlay_path)
        peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        write_record(
            out_dir,
            provider,
            {
                "provider": provider,
                "status": "completed",
                "input_policy": "native_scannet_rgb_no_pipeline_resize",
                "rgb_path": str(rgb_path),
                "rgb_sha256": sha256_file(rgb_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "call": "official SAM3 image model Sam3Processor.set_text_prompt on native RGB; text prompts loaded from explicit inline/file vocabulary",
                "sam3_text_prompt_sources": prompt_sources,
                "sam3_text_prompt_count": int(len(prompts)),
                "sam3_text_prompts": prompts,
                "sam3_confidence_threshold": confidence,
                "sam3_text_dedup_iou": float(args.sam3_text_dedup_iou),
                "raw_mask_count": int(len(mask_rows)),
                "selected_mask_count": int(len(selected)),
                "duplicate_skip_count": int(duplicate_skip_count),
                "max_masks_for_label": int(args.max_masks_for_label),
                "per_prompt_rows": per_prompt_rows,
                "selected_prompt_rows": [
                    {
                        "prompt": str(row["prompt"]),
                        "score": float(row["score"]),
                        "area": int(row["area"]),
                        "box_xyxy": row.get("box_xyxy", []),
                    }
                    for row in selected
                ],
                "label_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "runtime_sec": time.time() - t0,
                "peak_gpu_memory_mb": peak_mb,
                "stats": mask_stats(label),
            },
        )
    finally:
        sys.path[:] = old_sys_path


def run_cropformer_existing(args: argparse.Namespace) -> None:
    provider = "cropformer_native_existing"
    rgb_path = Path(args.rgb_path)
    rgb = read_rgb(rgb_path)
    mask_path_src = STREAM3D_ROOT / "data" / "scannet" / "processed" / args.scene_id / "output_Cropformer" / "mask" / f"{int(args.frame_id)}.png"
    label = cv2.imread(str(mask_path_src), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(mask_path_src)
    if label.ndim == 3:
        label = label[:, :, 0]
    out_dir = Path(args.output_root).resolve()
    mask_path = out_dir / f"{provider}_label.png"
    cv2.imwrite(str(mask_path), label)
    overlay_path = out_dir / f"{provider}_overlay.jpg"
    save_overlay(rgb, label.astype(np.uint16), overlay_path)
    write_record(
        out_dir,
        provider,
        {
            "provider": provider,
            "status": "completed",
            "input_policy": "native_scannet_processed_existing_cropformer_mask",
            "rgb_path": str(rgb_path),
            "rgb_sha256": sha256_file(rgb_path),
            "source_mask_path": str(mask_path_src),
            "source_mask_sha256": sha256_file(mask_path_src),
            "call": "existing ScanNet processed CropFormer entity mask, native resolution",
            "label_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "stats": mask_stats(label.astype(np.uint16)),
        },
    )


def run_cropformer_live(args: argparse.Namespace) -> None:
    provider = f"cropformer_native_live_c{int(round(float(args.cropformer_confidence) * 100)):03d}"
    rgb_path = Path(args.rgb_path)
    rgb = read_rgb(rgb_path)
    out_dir = Path(args.output_root).resolve()
    shadow_root = out_dir / "_cropformer_live_input"
    shadow_scene = shadow_root / args.scene_id
    shadow_color = shadow_scene / "color"
    if shadow_scene.exists():
        shutil.rmtree(shadow_scene)
    shadow_color.mkdir(parents=True, exist_ok=True)
    shadow_rgb = shadow_color / f"{int(args.frame_id)}.jpg"
    cv2.imwrite(str(shadow_rgb), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    demo_script = STREAM3D_ROOT / "third_party" / "detectron2" / "projects" / "CropFormer" / "demo_cropformer" / "Cropformer.py"
    config_path = STREAM3D_ROOT / "third_party" / "detectron2" / "projects" / "CropFormer" / "configs" / "entityv2" / "entity_segmentation" / "mask2former_hornet_3x.yaml"
    weight_path = STREAM3D_ROOT / "third_party" / "seg_models" / "Mask2Former_hornet_3x_576d0b.pth"
    for required in (demo_script, config_path, weight_path):
        if not required.exists():
            raise FileNotFoundError(required)
    command = [
        sys.executable,
        "third_party/detectron2/projects/CropFormer/demo_cropformer/Cropformer.py",
        "--config-file",
        "third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml",
        "--root",
        str(shadow_root),
        "--image_path_pattern",
        "color/*.jpg",
        "--dataset",
        "scannet",
        "--seq_name_list",
        args.scene_id,
        "--confidence-threshold",
        str(float(args.cropformer_confidence)),
        "--opts",
        "MODEL.WEIGHTS",
        "third_party/seg_models/Mask2Former_hornet_3x_576d0b.pth",
    ]
    log_path = out_dir / f"{provider}_run.log"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=STREAM3D_ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    source_mask = shadow_scene / "output_Cropformer" / "mask" / f"{int(args.frame_id)}.png"
    if proc.returncode != 0:
        raise RuntimeError(f"CropFormer live command failed rc={proc.returncode}; see {log_path}")
    label = cv2.imread(str(source_mask), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(source_mask)
    if label.ndim == 3:
        label = label[:, :, 0]
    label = label.astype(np.uint16)
    mask_path = out_dir / f"{provider}_label.png"
    cv2.imwrite(str(mask_path), label)
    overlay_path = out_dir / f"{provider}_overlay.jpg"
    save_overlay(rgb, label, overlay_path)
    write_record(
        out_dir,
        provider,
        {
            "provider": provider,
            "status": "completed",
            "input_policy": "native_scannet_rgb_live_model_no_pipeline_resize",
            "rgb_path": str(rgb_path),
            "rgb_sha256": sha256_file(rgb_path),
            "shadow_rgb_path": str(shadow_rgb),
            "shadow_rgb_sha256": sha256_file(shadow_rgb),
            "source_mask_path": str(source_mask),
            "source_mask_sha256": sha256_file(source_mask),
            "checkpoint": str(weight_path),
            "checkpoint_sha256": sha256_file(weight_path),
            "config": str(config_path),
            "call": "CropFormer demo_cropformer live model inference on native RGB shadow input",
            "command": command,
            "log_path": str(log_path),
            "cropformer_confidence_threshold": float(args.cropformer_confidence),
            "label_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "runtime_sec": time.time() - t0,
            "stats": mask_stats(label),
        },
    )


def annotate_panel(img: np.ndarray, title: str, lines: list[str]) -> Image.Image:
    pad = 74
    panel = Image.new("RGB", (img.shape[1], img.shape[0] + pad), (18, 20, 24))
    panel.paste(Image.fromarray(img), (0, pad))
    draw = ImageDraw.Draw(panel)
    try:
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font_b = ImageFont.load_default()
        font = ImageFont.load_default()
    draw.text((8, 7), title, fill=(245, 245, 245), font=font_b)
    y = 33
    for line in lines[:3]:
        draw.text((8, y), line, fill=(215, 218, 225), font=font)
        y += 15
    return panel


def run_summary(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_root).resolve()
    rgb = read_rgb(Path(args.rgb_path))
    records = []
    for path in sorted(out_dir.glob("*_record.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    panels: list[Image.Image] = []
    base_small = cv2.resize(rgb, (432, 323), interpolation=cv2.INTER_AREA)
    panels.append(annotate_panel(base_small, "Native RGB", [f"{rgb.shape[1]}x{rgb.shape[0]}", "no pipeline resize"]))
    for rec in records:
        overlay = read_rgb(Path(rec["overlay_path"]))
        overlay_small = cv2.resize(overlay, (432, 323), interpolation=cv2.INTER_AREA)
        stats = rec.get("stats", {})
        lines = [
            f"status={rec.get('status','')}, masks={stats.get('visible_id_count','')}",
            f"fg={float(stats.get('foreground_ratio',0.0)):.3f}, runtime={float(rec.get('runtime_sec',0.0)):.1f}s",
            str(rec.get("input_policy", ""))[:70],
        ]
        panels.append(annotate_panel(overlay_small, rec["provider"], lines))
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    panel_w = max(p.width for p in panels)
    panel_h = max(p.height for p in panels)
    sheet = Image.new("RGB", (cols * panel_w, rows * panel_h), (18, 20, 24))
    for i, panel in enumerate(panels):
        canvas = Image.new("RGB", (panel_w, panel_h), (18, 20, 24))
        canvas.paste(panel, (0, 0))
        sheet.paste(canvas, ((i % cols) * panel_w, (i // cols) * panel_h))
    sheet_path = out_dir / "native_raw_first_frame_segmentors_sheet.jpg"
    sheet.save(sheet_path, quality=95)
    summary = {
        "schema_version": "stream4d_v105_first_frame_native_raw_segmentor_audit_v1",
        "scene_id": args.scene_id,
        "frame_id": int(args.frame_id),
        "rgb_path": args.rgb_path,
        "rgb_sha256": sha256_file(Path(args.rgb_path)),
        "sheet_path": str(sheet_path),
        "records": records,
    }
    summary_path = out_dir / "native_raw_first_frame_segmentors_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"sheet_path": str(sheet_path), "summary_path": str(summary_path), "providers": [r["provider"] for r in records]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        required=True,
        choices=[
            "fastsam",
            "fastsam_roman",
            "sam2_4dpm",
            "edgetam_4dpm",
            "sam2_amg_tuned",
            "edgetam_amg_tuned",
            "sam3_points",
            "sam3_text",
            "cropformer",
            "cropformer_live",
            "summary",
        ],
    )
    parser.add_argument("--scene-id", default="scene0011_00")
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--rgb-path", default=str(STREAM3D_ROOT / "data" / "scannet" / "processed" / "scene0011_00" / "color" / "0.jpg"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--num-pts", type=int, default=300)
    parser.add_argument("--point-mode", choices=["4dpm_grid", "random_seeded"], default="4dpm_grid")
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--max-masks-for-label", type=int, default=96)
    parser.add_argument("--sam3-text-prompts", default="")
    parser.add_argument("--sam3-text-prompts-file", default="")
    parser.add_argument("--sam3-vocab-column", default="nyu40class")
    parser.add_argument("--sam3-max-prompts", type=int, default=0)
    parser.add_argument("--sam3-confidence", type=float, default=0.3)
    parser.add_argument("--sam3-text-dedup-iou", type=float, default=0.85)
    parser.add_argument("--fastsam-imgsz", type=int, default=1024)
    parser.add_argument("--fastsam-conf", type=float, default=0.3)
    parser.add_argument("--fastsam-iou", type=float, default=0.5)
    parser.add_argument("--fastsam-roman-min-mask-len-div", type=float, default=30.0)
    parser.add_argument("--fastsam-roman-max-mask-len-div", type=float, default=3.0)
    parser.add_argument("--fastsam-roman-min-area", type=float, default=-1.0)
    parser.add_argument("--fastsam-roman-max-area", type=float, default=-1.0)
    parser.add_argument("--fastsam-roman-edge-width", type=int, default=5)
    parser.add_argument("--fastsam-roman-allow-tblr-edges", default="1,1,1,1")
    parser.add_argument("--fastsam-roman-ignore-mask", default="")
    parser.add_argument("--fastsam-roman-keep-mask", default="")
    parser.add_argument("--fastsam-roman-keep-minimal-intersection", type=float, default=0.3)
    parser.add_argument("--fastsam-roman-label-order", choices=["large_to_small", "small_to_large", "score"], default="large_to_small")
    parser.add_argument("--cropformer-confidence", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=105031)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--stability-threshold", type=float, default=0.6)
    parser.add_argument("--box-nms-thresh", type=float, default=0.8)
    parser.add_argument("--empty-ratio", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    if args.provider == "fastsam":
        run_fastsam(args)
    elif args.provider == "fastsam_roman":
        run_fastsam_roman(args)
    elif args.provider == "sam2_4dpm":
        run_sam2_4dpm(args)
    elif args.provider == "edgetam_4dpm":
        run_edgetam_4dpm(args)
    elif args.provider == "sam2_amg_tuned":
        run_sam2_amg_tuned(args)
    elif args.provider == "edgetam_amg_tuned":
        run_edgetam_amg_tuned(args)
    elif args.provider == "sam3_points":
        run_sam3_points(args)
    elif args.provider == "sam3_text":
        run_sam3_text_concepts(args)
    elif args.provider == "cropformer":
        run_cropformer_existing(args)
    elif args.provider == "cropformer_live":
        run_cropformer_live(args)
    elif args.provider == "summary":
        run_summary(args)
    else:
        raise ValueError(args.provider)


if __name__ == "__main__":
    main()
