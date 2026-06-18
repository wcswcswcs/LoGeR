from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.scannet_stream import ScanNetStream
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v37_temporal_curriculum import (
    _aggregate_seed_rows,
    _load_instance_map,
    _mean,
    _quantile,
    _safe_div,
    _write_csv,
    _write_json,
)

DINO_CHECKPOINTS = [
    "/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth",
    "/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
]


def _raw_frame_paths(mask_root: Path, scene: str, source: str, mode: str) -> list[Path]:
    root = mask_root / scene / source / mode
    return sorted(root.glob(f"{source}_frame*_masks.npz"))


def _frame_from_path(path: Path) -> int:
    return int(path.stem.split("_frame", 1)[1].split("_", 1)[0])


def _kmeans(features: np.ndarray, k: int, *, iterations: int = 8) -> np.ndarray:
    centers = [features[0].astype(np.float32)]
    while len(centers) < int(k):
        stacked = np.stack(centers, axis=0)
        d2 = np.sum((features[:, None, :] - stacked[None, :, :]) ** 2, axis=2)
        centers.append(features[int(np.argmax(np.min(d2, axis=1)))].astype(np.float32))
    centers_arr = np.stack(centers, axis=0)
    for _ in range(int(iterations)):
        d2 = np.sum((features[:, None, :] - centers_arr[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        new_centers = centers_arr.copy()
        for idx in range(int(k)):
            members = features[labels == idx]
            if members.size:
                new_centers[idx] = np.mean(members, axis=0)
        if np.allclose(new_centers, centers_arr):
            break
        centers_arr = new_centers
    return centers_arr


def _parts_from_label_map(label_map: np.ndarray, mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    out = []
    for label in sorted(int(v) for v in np.unique(label_map) if int(v) >= 0):
        binary = np.asarray((label_map == int(label)) & mask, dtype=np.uint8)
        count, cc = cv2.connectedComponents(binary, connectivity=8)
        for comp_id in range(1, int(count)):
            part = cc == comp_id
            if int(part.sum()) >= int(min_area):
                out.append(part)
    return out


def _locate_dino_checkpoint(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "dino_checkpoint", "") or "")
    candidates = [explicit] if explicit else []
    candidates.extend(DINO_CHECKPOINTS)
    for item in candidates:
        if item and Path(item).exists():
            return str(item)
    raise FileNotFoundError(f"no DINO checkpoint found; checked={candidates}")


def _make_dino_context(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    import timm

    checkpoint = _locate_dino_checkpoint(args)
    device = str(getattr(args, "device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model = timm.create_model(str(args.dino_backbone), pretrained=False, num_classes=0)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return {
        "checkpoint": checkpoint,
        "device": device,
        "model": model,
        "torch": torch,
        "F": F,
        "cache": {},
    }


def _dino_grid_for_frame(context: dict[str, Any], scene: str, frame: int, rgb: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    key = (str(scene), int(frame))
    cache = context["cache"]
    if key in cache:
        return cache[key]
    torch = context["torch"]
    F = context["F"]
    image_size = int(args.dino_image_size)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy(((resized - mean) / std).transpose(2, 0, 1)).float()[None].to(context["device"])
    with torch.inference_mode():
        out = context["model"].forward_features(tensor)
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            tokens = out["x_norm_patchtokens"]
        elif isinstance(out, dict) and "x" in out:
            tokens = out["x"][:, 1:, :]
        else:
            tokens = out[:, 1:, :] if getattr(out, "ndim", 0) == 3 else out.reshape(out.shape[0], -1, out.shape[-1])
        tokens = F.normalize(tokens.float(), dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    grid = int(round(np.sqrt(tokens.shape[0])))
    if grid * grid != int(tokens.shape[0]):
        raise ValueError(f"non-square DINO token grid: token_count={tokens.shape[0]}")
    token_grid = tokens.reshape(grid, grid, tokens.shape[-1])
    cache[key] = token_grid
    return token_grid


def _dino_split(
    mask: np.ndarray,
    dino_grid: np.ndarray,
    args: argparse.Namespace,
    *,
    force_k: int | None = None,
    guarded: bool = True,
) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area < int(args.dino_min_split_area):
        return [mask]
    grid_h, grid_w, dim = dino_grid.shape
    mask_grid = cv2.resize(mask.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA) >= float(args.dino_min_patch_fraction)
    yy, xx = np.nonzero(mask_grid)
    if yy.size < int(args.dino_min_token_count):
        return [mask]
    token_features = dino_grid[yy, xx].astype(np.float32)
    if int(args.dino_max_kmeans_tokens) > 0 and yy.size > int(args.dino_max_kmeans_tokens):
        stride = max(1, int(np.ceil(yy.size / int(args.dino_max_kmeans_tokens))))
        sample = np.arange(0, yy.size, stride, dtype=np.int64)
    else:
        sample = np.arange(yy.size, dtype=np.int64)
    if force_k is not None:
        k = int(force_k)
    else:
        k = 3 if area >= int(args.dino_large_area) else 2
    k = max(2, min(k, int(args.dino_max_splits), int(len(sample) // max(int(args.dino_min_tokens_per_child), 1))))
    if k < 2:
        return [mask]
    spatial = np.stack(
        [
            xx.astype(np.float32) / max(grid_w - 1, 1),
            yy.astype(np.float32) / max(grid_h - 1, 1),
        ],
        axis=1,
    )
    sample_features = np.concatenate(
        [token_features[sample], spatial[sample] * float(args.dino_spatial_weight)],
        axis=1,
    )
    centers = _kmeans(sample_features, k, iterations=int(args.dino_kmeans_iterations))
    full_features = np.concatenate([token_features, spatial * float(args.dino_spatial_weight)], axis=1)
    d2 = np.sum((full_features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1).astype(np.int32)
    if guarded:
        dino_centers = centers[:, :dim]
        norms = np.linalg.norm(dino_centers, axis=1, keepdims=True)
        dino_centers = dino_centers / np.maximum(norms, 1e-6)
        cosine = np.clip(dino_centers @ dino_centers.T, -1.0, 1.0)
        distance = 1.0 - cosine
        positive = distance[distance > 1e-6]
        min_distance = float(np.min(positive)) if positive.size else 0.0
        if min_distance < float(args.dino_min_center_distance):
            return [mask]
    grid_labels = np.full((grid_h, grid_w), -1, dtype=np.int32)
    grid_labels[yy, xx] = labels
    full_labels = cv2.resize(grid_labels, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    parts = _parts_from_label_map(full_labels, mask, int(args.min_child_area))
    if len(parts) < 2:
        return [mask]
    if guarded:
        if len(parts) > int(args.dino_guarded_max_child_count):
            return [mask]
        fractions = [float(part.sum() / max(area, 1)) for part in parts]
        if min(fractions) < float(args.dino_guarded_min_child_fraction):
            return [mask]
    return parts


def _dino_part_compactness(mask: np.ndarray, dino_grid: np.ndarray, args: argparse.Namespace) -> float:
    mask = np.asarray(mask, dtype=bool)
    grid_h, grid_w, _dim = dino_grid.shape
    mask_grid = cv2.resize(mask.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA) >= float(args.dino_min_patch_fraction)
    yy, xx = np.nonzero(mask_grid)
    if yy.size == 0:
        return 0.0
    tokens = dino_grid[yy, xx].astype(np.float32)
    center = np.mean(tokens, axis=0)
    center = center / max(float(np.linalg.norm(center)), 1e-6)
    return float(np.mean(tokens @ center))


def _compact_threshold_from_variant(variant: str) -> float:
    if "compact050" in variant:
        return 0.50
    if "compact055" in variant:
        return 0.55
    if "compact070" in variant:
        return 0.70
    if "compact065" in variant:
        return 0.65
    if "compact060" in variant:
        return 0.60
    raise ValueError(f"compact threshold not encoded in variant: {variant}")


def _gain_margin_from_variant(variant: str) -> float:
    if "gain020" in variant:
        return 0.020
    if "gain050" in variant:
        return 0.050
    raise ValueError(f"gain margin not encoded in variant: {variant}")


def _boundary_quantile_from_variant(variant: str, default: float) -> float:
    if "q95" in variant:
        return 0.95
    if "q90" in variant:
        return 0.90
    if "q85" in variant:
        return 0.85
    return float(default)


def _rgb_split(
    mask: np.ndarray,
    rgb: np.ndarray,
    args: argparse.Namespace,
    *,
    force_k: int | None = None,
    guarded: bool = False,
) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area < int(args.rgb_min_split_area):
        return [mask]
    yy, xx = np.nonzero(mask)
    if yy.size < int(args.min_child_area) * 2:
        return [mask]
    pix = rgb[yy, xx].astype(np.float32) / 255.0
    color_std = float(np.mean(np.std(pix, axis=0))) if pix.size else 0.0
    if color_std < float(args.min_color_std):
        return [mask]
    k = int(force_k or (3 if area >= int(args.rgb_large_area) else 2))
    k = max(2, min(k, int(args.max_rgb_splits), int(yy.size // max(int(args.min_child_area), 1))))
    if k < 2:
        return [mask]
    stride = max(1, int(np.ceil(yy.size / max(int(args.max_kmeans_samples), 1))))
    sample_idx = np.arange(0, yy.size, stride, dtype=np.int64)
    sample_pix = pix[sample_idx]
    sample_xy = np.stack(
        [
            xx[sample_idx].astype(np.float32) / max(mask.shape[1] - 1, 1),
            yy[sample_idx].astype(np.float32) / max(mask.shape[0] - 1, 1),
        ],
        axis=1,
    )
    features_sample = np.concatenate([sample_pix, sample_xy * float(args.spatial_weight)], axis=1)
    centers = _kmeans(features_sample, k)
    if guarded and k > 1:
        color_centers = centers[:, :3]
        color_d2 = np.sum((color_centers[:, None, :] - color_centers[None, :, :]) ** 2, axis=2)
        positive = color_d2[color_d2 > 0]
        min_color_dist = float(np.sqrt(np.min(positive))) if positive.size else 0.0
        if min_color_dist < float(args.guarded_min_color_center_distance):
            return [mask]
    xy = np.stack(
        [
            xx.astype(np.float32) / max(mask.shape[1] - 1, 1),
            yy.astype(np.float32) / max(mask.shape[0] - 1, 1),
        ],
        axis=1,
    )
    features_full = np.concatenate([pix, xy * float(args.spatial_weight)], axis=1)
    d2 = np.sum((features_full[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1).astype(np.int32)
    label_map = np.full(mask.shape, -1, dtype=np.int32)
    label_map[yy, xx] = labels
    parts = _parts_from_label_map(label_map, mask, int(args.min_child_area))
    if len(parts) < 2:
        return [mask]
    if guarded:
        if len(parts) > int(args.guarded_max_child_count):
            return [mask]
        fractions = [float(part.sum() / max(area, 1)) for part in parts]
        if min(fractions) < float(args.guarded_min_child_fraction):
            return [mask]
    return parts


def _boundary_gradient(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    gray = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    scale = float(np.max(grad))
    return grad / scale if scale > 1e-6 else grad


def _boundary_split(
    mask: np.ndarray,
    rgb: np.ndarray,
    args: argparse.Namespace,
    *,
    variant: str,
) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area < int(args.boundary_min_split_area):
        return [mask]
    grad = _boundary_gradient(rgb)
    if grad.shape != mask.shape:
        grad = cv2.resize(grad.astype(np.float32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    inside = grad[mask]
    if inside.size < int(args.min_child_area) * 2:
        return [mask]
    quantile = _boundary_quantile_from_variant(variant, float(args.boundary_gradient_quantile))
    threshold = max(float(np.quantile(inside, quantile)), float(args.boundary_min_gradient))
    edge = (grad >= threshold) & mask
    dilate = int(args.boundary_edge_dilate)
    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), kernel, iterations=1).astype(bool) & mask
    core = mask & ~edge
    count, core_cc = cv2.connectedComponents(core.astype(np.uint8), connectivity=8)
    seed_labels = []
    for comp_id in range(1, int(count)):
        part = core_cc == comp_id
        part_area = int(part.sum())
        if part_area >= int(args.min_child_area) and float(part_area / max(area, 1)) >= float(args.boundary_min_child_fraction):
            seed_labels.append(int(comp_id))
    if len(seed_labels) < 2 or len(seed_labels) > int(args.boundary_max_child_count):
        return [mask]
    if float(sum(int(np.count_nonzero(core_cc == comp_id)) for comp_id in seed_labels) / max(area, 1)) < float(args.boundary_min_core_coverage):
        return [mask]
    if variant.startswith("boundary_edgecut_"):
        parts = [core_cc == comp_id for comp_id in seed_labels]
    elif variant.startswith("boundary_watershed_"):
        markers = np.zeros(mask.shape, dtype=np.int32)
        markers[~mask] = 1
        for out_id, comp_id in enumerate(seed_labels, start=2):
            markers[core_cc == comp_id] = int(out_id)
        ws_image = cv2.cvtColor((np.clip(grad, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        markers = cv2.watershed(ws_image, markers)
        parts = [(markers == out_id) & mask for out_id in range(2, 2 + len(seed_labels))]
    else:
        raise ValueError(f"unknown boundary split variant: {variant}")
    parts = [np.asarray(part, dtype=bool) for part in parts if int(np.asarray(part, dtype=bool).sum()) >= int(args.min_child_area)]
    return parts if len(parts) >= 2 else [mask]


def _oracle_split(mask: np.ndarray, gt: np.ndarray | None, args: argparse.Namespace) -> list[np.ndarray]:
    if gt is None:
        return [mask]
    if gt.shape != mask.shape:
        gt = cv2.resize(gt.astype(np.int32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    parts = []
    for gt_id in sorted(int(v) for v in np.unique(gt[mask]) if int(v) > 0):
        part = np.asarray(mask & (gt == int(gt_id)), dtype=bool)
        if int(part.sum()) >= int(args.min_child_area):
            parts.append(part)
    return parts if parts else [mask]


def _select_frame_paths(paths: list[Path], args: argparse.Namespace) -> list[Path]:
    stride = max(1, int(args.frame_stride))
    selected = paths[::stride]
    if int(args.max_frames) > 0:
        selected = selected[: int(args.max_frames)]
    return selected


def _update_gt_area(frame: int, gt: np.ndarray | None, gt_area: Counter[tuple[int, int]]) -> None:
    if gt is None:
        return
    vals, counts = np.unique(gt, return_counts=True)
    for value, count in zip(vals.tolist(), counts.tolist()):
        if int(value) > 0:
            gt_area[(int(frame), int(value))] = int(count)


def _mask_gt_counts(mask: np.ndarray, gt: np.ndarray | None) -> Counter[int]:
    counts: Counter[int] = Counter()
    if gt is None:
        return counts
    if gt.shape != mask.shape:
        gt = cv2.resize(gt.astype(np.int32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    vals, overlaps = np.unique(gt[mask], return_counts=True)
    for value, count in zip(vals.tolist(), overlaps.tolist()):
        if int(value) > 0:
            counts[int(value)] += int(count)
    return counts


def _seed_row_from_stream(
    *,
    scene: str,
    variant: str,
    status: str,
    ratios: list[float],
    mixed_count: int,
    best_iou: dict[tuple[int, int], float],
    gt_area: Counter[tuple[int, int]],
    stats: dict[str, int],
) -> dict[str, Any]:
    row = {
        "scene": scene,
        "variant": f"C11_{variant}",
        "status": status,
        "same_frame_seed_count": int(len(ratios)),
        "same_frame_mixed_seed_rate": _safe_div(int(mixed_count), len(ratios)),
        "seed_purity_mean": _mean(ratios),
        "seed_purity_p10": _quantile(ratios, 0.10),
        "seed_GT_coverage@0.05": _safe_div(sum(1 for key in gt_area if best_iou.get(key, 0.0) >= 0.05), len(gt_area)),
        "seed_GT_coverage@0.10": _safe_div(sum(1 for key in gt_area if best_iou.get(key, 0.0) >= 0.10), len(gt_area)),
    }
    row.update({key: int(value) for key, value in stats.items()})
    return row


def _stream_variant(
    scene: str,
    variant: str,
    raw_paths: list[Path],
    args: argparse.Namespace,
    dino_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    stream = ScanNetStream(seq_name=scene)
    gt_area: Counter[tuple[int, int]] = Counter()
    best_iou: dict[tuple[int, int], float] = defaultdict(float)
    ratios: list[float] = []
    mixed_count = 0
    stats: dict[str, int] = defaultdict(int)
    processed_paths = _select_frame_paths(raw_paths, args)
    save_root = Path(args.save_source_root) if str(args.save_source_root) else None
    save_source = str(args.save_source_name)
    save_mode = variant
    for path in processed_paths:
        frame = _frame_from_path(path)
        masks = [np.asarray(mask, dtype=bool) for mask in np.asarray(np.load(path)["masks"], dtype=bool)]
        if int(args.max_parent_masks_per_frame) > 0:
            masks = sorted(masks, key=lambda item: int(item.sum()), reverse=True)[: int(args.max_parent_masks_per_frame)]
        gt = _load_instance_map(stream, frame)
        if masks and gt is not None and gt.shape != masks[0].shape:
            gt = cv2.resize(gt.astype(np.int32), (masks[0].shape[1], masks[0].shape[0]), interpolation=cv2.INTER_NEAREST)
        _update_gt_area(frame, gt, gt_area)
        rgb = None
        if (variant.startswith("rgb_") or variant.startswith("dino_") or variant.startswith("boundary_")) and masks:
            rgb = stream.load_rgb(frame)
            if rgb.shape[:2] != masks[0].shape:
                rgb = cv2.resize(rgb, (masks[0].shape[1], masks[0].shape[0]), interpolation=cv2.INTER_AREA)
        dino_grid = None
        if variant.startswith("dino_") and masks:
            if dino_context is None or rgb is None:
                raise RuntimeError("DINO split requested without DINO context")
            dino_grid = _dino_grid_for_frame(dino_context, scene, frame, rgb, args)
        stats["processed_frames"] += 1
        frame_parts: list[np.ndarray] = []
        for mask in masks:
            stats["parent_masks"] += 1
            if variant == "raw_passthrough":
                parts = [mask]
            elif variant == "oracle_gt_split":
                parts = _oracle_split(mask, gt, args)
            elif variant == "rgb_k2_split":
                assert rgb is not None
                parts = _rgb_split(mask, rgb, args, force_k=2)
            elif variant == "rgb_k3_large_split":
                assert rgb is not None
                parts = _rgb_split(mask, rgb, args, force_k=None)
            elif variant == "rgb_k2_guarded_split":
                assert rgb is not None
                parts = _rgb_split(mask, rgb, args, force_k=2, guarded=True)
            elif variant == "rgb_k3_guarded_split":
                assert rgb is not None
                parts = _rgb_split(mask, rgb, args, force_k=None, guarded=True)
            elif variant.startswith("boundary_edgecut_") or variant.startswith("boundary_watershed_"):
                assert rgb is not None
                parts = _boundary_split(mask, rgb, args, variant=variant)
                if len(parts) > 1:
                    stats["boundary_accepted_parent_masks"] += 1
                else:
                    stats["boundary_rejected_parent_masks"] += 1
            elif variant == "dino_k2_guarded_split":
                assert dino_grid is not None
                parts = _dino_split(mask, dino_grid, args, force_k=2, guarded=True)
            elif variant == "dino_k3_large_guarded_split":
                assert dino_grid is not None
                parts = _dino_split(mask, dino_grid, args, force_k=None, guarded=True)
            elif variant in {
                "dino_k2_compact050_filter",
                "dino_k2_compact055_filter",
                "dino_k2_compact060_filter",
                "dino_k2_compact065_filter",
                "dino_k2_compact070_filter",
            }:
                assert dino_grid is not None
                raw_parts = _dino_split(mask, dino_grid, args, force_k=2, guarded=True)
                threshold = _compact_threshold_from_variant(variant)
                parts = [
                    part
                    for part in raw_parts
                    if _dino_part_compactness(np.asarray(part, dtype=bool), dino_grid, args) >= threshold
                ]
                stats["dino_compact_filter_candidates"] += len(raw_parts)
                stats["dino_compact_filter_kept"] += len(parts)
                stats["dino_compact_filter_dropped"] += len(raw_parts) - len(parts)
            elif variant in {"dino_k2_gain020_split", "dino_k2_gain050_split"}:
                assert dino_grid is not None
                raw_parts = _dino_split(mask, dino_grid, args, force_k=2, guarded=True)
                parent_compact = _dino_part_compactness(np.asarray(mask, dtype=bool), dino_grid, args)
                margin = _gain_margin_from_variant(variant)
                if len(raw_parts) > 1:
                    child_compact = [_dino_part_compactness(np.asarray(part, dtype=bool), dino_grid, args) for part in raw_parts]
                    gain = float(np.mean(child_compact) - parent_compact)
                    accept = bool(gain >= margin and min(child_compact) >= parent_compact)
                else:
                    child_compact = []
                    gain = 0.0
                    accept = False
                parts = raw_parts if accept else [mask]
                stats["dino_gain_candidates"] += len(raw_parts)
                stats["dino_gain_accepted_parent_masks"] += int(accept)
                stats["dino_gain_rejected_parent_masks"] += int((len(raw_parts) > 1) and not accept)
                stats["dino_gain_margin_milli_sum"] += int(round(margin * 1000.0))
                stats["dino_gain_observed_milli_sum"] += int(round(gain * 1000.0))
            else:
                raise ValueError(variant)
            stats["child_masks"] += len(parts)
            if len(parts) > 1:
                stats["split_parent_masks"] += 1
            for part in parts:
                area = int(np.asarray(part, dtype=bool).sum())
                if area < int(args.min_region_area):
                    stats["dropped_small_child_masks"] += 1
                    continue
                if save_root is not None:
                    frame_parts.append(np.asarray(part, dtype=bool))
                counts = _mask_gt_counts(np.asarray(part, dtype=bool), gt)
                labeled = int(sum(counts.values()))
                dominant = int(max(counts.values())) if counts else 0
                ratio = _safe_div(dominant, labeled)
                ratios.append(float(ratio or 0.0))
                if len(counts) > 1 and float(ratio or 0.0) < 0.95:
                    mixed_count += 1
                for gt_id, overlap in counts.items():
                    denom = area + int(gt_area.get((int(frame), int(gt_id)), 0)) - int(overlap)
                    if denom > 0:
                        best_iou[(int(frame), int(gt_id))] = max(
                            best_iou[(int(frame), int(gt_id))],
                            float(int(overlap) / denom),
                        )
        if save_root is not None:
            out_dir = save_root / scene / save_source / save_mode
            out_dir.mkdir(parents=True, exist_ok=True)
            if frame_parts:
                out_masks = np.stack(frame_parts, axis=0).astype(bool)
            elif masks:
                out_masks = np.zeros((0, *masks[0].shape), dtype=bool)
            else:
                out_masks = np.zeros((0, 0, 0), dtype=bool)
            np.savez_compressed(
                out_dir / f"{save_source}_frame{int(frame):06d}_masks.npz",
                masks=out_masks,
            )
    stats["raw_frame_count"] = int(len(raw_paths))
    stats["selected_frame_count"] = int(len(processed_paths))
    if variant == "oracle_gt_split":
        status = "oracle_upper_bound"
    elif variant == "raw_passthrough":
        status = "raw_exact_mask_baseline"
    elif variant.startswith("dino_"):
        status = "dino_method_split"
    elif variant.startswith("boundary_"):
        status = "boundary_method_split"
    else:
        status = "rgb_method_split"
    return (
        _seed_row_from_stream(
            scene=scene,
            variant=variant,
            status=status,
            ratios=ratios,
            mixed_count=mixed_count,
            best_iou=best_iou,
            gt_area=gt_area,
            stats=stats,
        ),
        {key: int(value) for key, value in stats.items()},
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _read_split(Path(args.split))
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    needs_dino = any(variant.startswith("dino_") for variant in variants)
    dino_context = _make_dino_context(args) if needs_dino else None
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_rows = []
    manifests = {}
    for scene in scenes:
        raw_paths = _raw_frame_paths(Path(args.mask_root), scene, args.source, args.mode)
        stats = {}
        for variant in variants:
            row, variant_stats = _stream_variant(scene, variant, raw_paths, args, dino_context=dino_context)
            scene_rows.append(row)
            stats[variant] = variant_stats
        manifests[scene] = {
            "raw_frame_count": int(len(raw_paths)),
            "selected_frame_count": int(len(_select_frame_paths(raw_paths, args))),
            "frame_stride": int(args.frame_stride),
            "max_frames": int(args.max_frames),
            "max_parent_masks_per_frame": int(args.max_parent_masks_per_frame),
            "save_source_root": str(args.save_source_root),
            "save_source_name": str(args.save_source_name),
            "uses_gt_for_prediction": "oracle_gt_split only",
            "rgb_variants_use_gt_for_prediction": False,
            "dino_variants_use_gt_for_prediction": False,
            "boundary_variants_use_gt_for_prediction": False,
            "dino_checkpoint": None if dino_context is None else str(dino_context["checkpoint"]),
            "dino_backbone": str(args.dino_backbone) if needs_dino else "",
            "uses_gt_for_diagnostic_labels": True,
            "overlapping_input_masks_preserved": True,
            "diagnostic_metric_path": "streaming exact-mask GT overlap; no single-label map overwrite",
            "stats": {variant: dict(stats[variant]) for variant in variants},
        }
    summary = _aggregate_seed_rows(scene_rows)
    _write_csv(out_dir / "same_frame_oracle_rgb_split_scene_rows.csv", scene_rows)
    _write_csv(out_dir / "same_frame_oracle_rgb_split_summary.csv", summary)
    _write_json(out_dir / "same_frame_oracle_rgb_split_summary.json", summary)
    payload = {
        "plan": "docs/stream4d_v37_temporal_curriculum_masklet_plan.md",
        "phase": "Phase C oracle/RGB split diagnostic",
        "summary": summary,
        "manifests": manifests,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "uses_gt_for_prediction": "only oracle_gt_split branch",
        "uses_gt_for_diagnostic_labels": True,
    }
    _write_json(out_dir / "same_frame_oracle_rgb_split_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v37 Phase C oracle upper-bound and RGB split diagnostics.")
    parser.add_argument("--mask-root", default="outputs/audit/v36_external_mask_source_all")
    parser.add_argument("--source", default="watershed")
    parser.add_argument("--mode", default="all_masks")
    parser.add_argument("--split", default="splits/scannet_scene0081.txt")
    parser.add_argument("--output-root", default="outputs/audit/v37_same_frame_objectlets/oracle_rgb_split")
    parser.add_argument("--variants", default="raw_passthrough,oracle_gt_split,rgb_k2_split,rgb_k3_large_split")
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--min-child-area", type=int, default=64)
    parser.add_argument("--rgb-min-split-area", type=int, default=512)
    parser.add_argument("--rgb-large-area", type=int, default=24000)
    parser.add_argument("--max-rgb-splits", type=int, default=3)
    parser.add_argument("--max-kmeans-samples", type=int, default=4096)
    parser.add_argument("--spatial-weight", type=float, default=0.35)
    parser.add_argument("--min-color-std", type=float, default=0.025)
    parser.add_argument("--guarded-min-color-center-distance", type=float, default=0.18)
    parser.add_argument("--guarded-min-child-fraction", type=float, default=0.08)
    parser.add_argument("--guarded-max-child-count", type=int, default=4)
    parser.add_argument("--boundary-min-split-area", type=int, default=1024)
    parser.add_argument("--boundary-gradient-quantile", type=float, default=0.90)
    parser.add_argument("--boundary-min-gradient", type=float, default=0.08)
    parser.add_argument("--boundary-edge-dilate", type=int, default=1)
    parser.add_argument("--boundary-min-child-fraction", type=float, default=0.05)
    parser.add_argument("--boundary-min-core-coverage", type=float, default=0.35)
    parser.add_argument("--boundary-max-child-count", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dino-backbone", default="vit_small_patch14_dinov2")
    parser.add_argument("--dino-checkpoint", default="")
    parser.add_argument("--dino-image-size", type=int, default=518)
    parser.add_argument("--dino-min-split-area", type=int, default=1024)
    parser.add_argument("--dino-large-area", type=int, default=24000)
    parser.add_argument("--dino-max-splits", type=int, default=3)
    parser.add_argument("--dino-min-token-count", type=int, default=6)
    parser.add_argument("--dino-min-tokens-per-child", type=int, default=3)
    parser.add_argument("--dino-min-patch-fraction", type=float, default=0.20)
    parser.add_argument("--dino-spatial-weight", type=float, default=0.20)
    parser.add_argument("--dino-kmeans-iterations", type=int, default=8)
    parser.add_argument("--dino-max-kmeans-tokens", type=int, default=2048)
    parser.add_argument("--dino-min-center-distance", type=float, default=0.035)
    parser.add_argument("--dino-guarded-min-child-fraction", type=float, default=0.05)
    parser.add_argument("--dino-guarded-max-child-count", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-parent-masks-per-frame", type=int, default=0)
    parser.add_argument("--save-source-root", default="")
    parser.add_argument("--save-source-name", default="v37_oracle_rgb_split")
    args = parser.parse_args()
    print(json.dumps(_json_safe(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
