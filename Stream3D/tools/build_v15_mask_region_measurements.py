"""Build v15 mask-region measurement candidates and diagnostics.

The prediction artifact produced by this tool is diagnostic-only: each exported
mask-region is still a measurement candidate, not a claimed object method. GT is
not read for candidate generation. If --gt-diagnostic is set, GT is read only to
attribute region purity/completeness after the prediction has been written.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluation.constants import SCANNET_IDS
from stream4d.export_scannet import ScanNetExporter
from stream4d.measurement_bank import MeasurementBank, json_safe, load_mask, load_rgb, read_seq_list
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


MIN_GT_REGION_SIZE = 100
THRESHOLDS = (0.10, 0.25, 0.50, 0.75)


@dataclass
class RegionCandidate:
    region_id: int
    frame_id: int
    mask_id: int
    component_id: int
    surfels: np.ndarray
    pixels_xy: np.ndarray
    pixel_count: int
    bbox_xyxy: tuple[int, int, int, int]
    boundary_safe_ratio: float
    visible_outside_negative_ratio: float
    parent_area_ratio: float
    split_mode: str


def _class_agnostic_gt(gt_ids: np.ndarray) -> np.ndarray:
    return gt_ids % 1000 + int(SCANNET_IDS[0]) * 1000


def _gt_instance_masks(gt_ids: np.ndarray) -> tuple[np.ndarray, list[int], np.ndarray]:
    masks: list[np.ndarray] = []
    ids: list[int] = []
    areas: list[int] = []
    for instance_id in np.unique(gt_ids):
        instance_id = int(instance_id)
        if instance_id < 1000:
            continue
        mask = gt_ids == instance_id
        area = int(np.count_nonzero(mask))
        if area < MIN_GT_REGION_SIZE:
            continue
        masks.append(mask)
        ids.append(instance_id)
        areas.append(area)
    if not masks:
        return np.zeros((0, gt_ids.shape[0]), dtype=bool), ids, np.zeros((0,), dtype=np.int64)
    return np.stack(masks, axis=0), ids, np.asarray(areas, dtype=np.int64)


def _xy_from_uv(uv: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x = np.rint(uv[:, 0] * float(max(width - 1, 1))).astype(np.int64)
    y = np.rint(uv[:, 1] * float(max(height - 1, 1))).astype(np.int64)
    x = np.clip(x, 0, max(width - 1, 0))
    y = np.clip(y, 0, max(height - 1, 0))
    return np.stack([x, y], axis=1)


def _sample_pixels(xy: np.ndarray, stride: int, max_pixels: int) -> np.ndarray:
    if xy.size == 0:
        return xy.reshape(0, 2).astype(np.int64)
    stride = max(1, int(stride))
    if stride > 1:
        keep = ((xy[:, 0] % stride) == 0) & ((xy[:, 1] % stride) == 0)
        xy = xy[keep]
    if xy.shape[0] == 0:
        return xy.astype(np.int64)
    max_pixels = int(max_pixels)
    if max_pixels > 0 and xy.shape[0] > max_pixels:
        keep = np.linspace(0, xy.shape[0] - 1, num=max_pixels, dtype=np.int64)
        xy = xy[keep]
    return xy.astype(np.int64, copy=False)


def _bbox_from_pixels(xy: np.ndarray) -> tuple[int, int, int, int]:
    if xy.size == 0:
        return (0, 0, 0, 0)
    return (int(xy[:, 0].min()), int(xy[:, 1].min()), int(xy[:, 0].max()), int(xy[:, 1].max()))


def _surfel_stats(bank: MeasurementBank, frame_idx: int, surfels: np.ndarray, boundary_safe_px: float) -> tuple[float, float]:
    if surfels.size == 0:
        return 0.0, 0.0
    boundary = np.asarray(bank.boundary_distance[frame_idx, surfels], dtype=np.float32)
    negative = np.asarray(bank.negative_observation[:, surfels], dtype=bool)
    visible = np.asarray(bank.visible_ok[:, surfels], dtype=bool)
    boundary_safe_ratio = float(np.mean(boundary >= float(boundary_safe_px))) if boundary.size else 0.0
    negative_ratio = float(np.count_nonzero(negative) / max(np.count_nonzero(visible), 1))
    return boundary_safe_ratio, negative_ratio


def _cluster_surfels_by_grid(
    xy: np.ndarray,
    surfels: np.ndarray,
    bbox: tuple[int, int, int, int],
    grid: int,
) -> dict[tuple[int, int], np.ndarray]:
    if surfels.size == 0:
        return {}
    x0, y0, x1, y1 = bbox
    width = max(int(x1 - x0 + 1), 1)
    height = max(int(y1 - y0 + 1), 1)
    gx = np.floor((xy[:, 0] - x0) / max(width, 1) * int(grid)).astype(np.int64)
    gy = np.floor((xy[:, 1] - y0) / max(height, 1) * int(grid)).astype(np.int64)
    gx = np.clip(gx, 0, int(grid) - 1)
    gy = np.clip(gy, 0, int(grid) - 1)
    out: dict[tuple[int, int], list[int]] = {}
    for key, surfel in zip(zip(gx.tolist(), gy.tolist()), surfels.tolist()):
        out.setdefault((int(key[0]), int(key[1])), []).append(int(surfel))
    return {key: np.asarray(vals, dtype=np.int64) for key, vals in out.items()}


def _voronoi_pixels_for_clusters(
    component_pixels: np.ndarray,
    cluster_surfels: dict[tuple[int, int], np.ndarray],
    surfel_xy: dict[int, tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    if not cluster_surfels:
        return {}
    keys = sorted(cluster_surfels.keys())
    centroids: list[np.ndarray] = []
    for key in keys:
        xy = np.asarray([surfel_xy[int(idx)] for idx in cluster_surfels[key].tolist()], dtype=np.float32)
        centroids.append(np.mean(xy, axis=0))
    centroid_arr = np.stack(centroids, axis=0)
    pixels = component_pixels.astype(np.float32)
    # Component masks are per-frame 2D objects; the number of clusters is kept
    # small by grid splitting, so a dense distance matrix is manageable.
    distances = np.sum((pixels[:, None, :] - centroid_arr[None, :, :]) ** 2, axis=2)
    owner = np.argmin(distances, axis=1)
    out: dict[tuple[int, int], np.ndarray] = {}
    for local_idx, key in enumerate(keys):
        out[key] = component_pixels[owner == local_idx]
    return out


def _regions_for_frame_mask(
    *,
    bank: MeasurementBank,
    frame_idx: int,
    frame_id: int,
    mask: np.ndarray,
    mask_id: int,
    next_region_id: int,
    mode: str,
    split_grid: int,
    min_surfels: int,
    min_region_pixels: int,
    boundary_safe_px: float,
    erode_px: int,
    pixel_stride: int,
    max_pixels_per_region: int,
) -> list[RegionCandidate]:
    parent_binary = mask == int(mask_id)
    if int(np.count_nonzero(parent_binary)) < int(min_region_pixels):
        return []
    if mode == "boundary_core" and int(erode_px) > 0:
        kernel = np.ones((int(erode_px) * 2 + 1, int(erode_px) * 2 + 1), dtype=np.uint8)
        binary = cv2.erode(parent_binary.astype(np.uint8), kernel, iterations=1).astype(bool)
        if int(np.count_nonzero(binary)) < int(min_region_pixels):
            binary = parent_binary
    else:
        binary = parent_binary

    target_ids = np.asarray(bank.target_mask_id[frame_idx], dtype=np.int64)
    positive = np.asarray(bank.positive_observation[frame_idx], dtype=bool)
    surfels_all = np.flatnonzero(positive & (target_ids == int(mask_id))).astype(np.int64)
    if surfels_all.size == 0:
        return []
    surfel_xy_all = _xy_from_uv(np.asarray(bank.uv_pred[frame_idx, surfels_all], dtype=np.float32), mask.shape[:2])

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    regions: list[RegionCandidate] = []
    region_id = int(next_region_id)
    surfel_xy_lookup = {int(s): (int(x), int(y)) for s, (x, y) in zip(surfels_all.tolist(), surfel_xy_all.tolist())}
    for component_id in range(1, int(num_labels)):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < int(min_region_pixels):
            continue
        x0 = int(stats[component_id, cv2.CC_STAT_LEFT])
        y0 = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        bbox = (x0, y0, x0 + max(w - 1, 0), y0 + max(h - 1, 0))
        in_component = labels[surfel_xy_all[:, 1], surfel_xy_all[:, 0]] == int(component_id)
        component_surfels = surfels_all[in_component]
        component_xy = surfel_xy_all[in_component]
        if component_surfels.size < int(min_surfels):
            continue
        ys, xs = np.where(labels == int(component_id))
        component_pixels = np.stack([xs.astype(np.int64), ys.astype(np.int64)], axis=1)
        if mode in {"component", "boundary_core"}:
            pieces = {("component", 0): component_surfels}
            pixels_by_piece = {("component", 0): component_pixels}
        elif mode == "seed_voronoi":
            pieces = _cluster_surfels_by_grid(component_xy, component_surfels, bbox, grid=int(split_grid))
            if len(pieces) <= 1:
                pieces = {("component", 0): component_surfels}
                pixels_by_piece = {("component", 0): component_pixels}
            else:
                pixels_by_piece = _voronoi_pixels_for_clusters(component_pixels, pieces, surfel_xy_lookup)
        else:
            raise ValueError(f"Unsupported mask-region mode: {mode}")

        for _, surfels in sorted(pieces.items(), key=lambda item: (int(item[0][0] != "component"), str(item[0]))):
            pixels = pixels_by_piece.get(_, np.empty((0, 2), dtype=np.int64))
            if surfels.size < int(min_surfels) or pixels.shape[0] < int(min_region_pixels):
                continue
            sampled_pixels = _sample_pixels(pixels, stride=int(pixel_stride), max_pixels=int(max_pixels_per_region))
            if sampled_pixels.shape[0] == 0:
                continue
            boundary_ratio, negative_ratio = _surfel_stats(bank, frame_idx, surfels, boundary_safe_px)
            regions.append(
                RegionCandidate(
                    region_id=region_id,
                    frame_id=int(frame_id),
                    mask_id=int(mask_id),
                    component_id=int(component_id),
                    surfels=np.asarray(sorted(int(v) for v in surfels.tolist()), dtype=np.int64),
                    pixels_xy=sampled_pixels,
                    pixel_count=int(pixels.shape[0]),
                    bbox_xyxy=_bbox_from_pixels(pixels),
                    boundary_safe_ratio=float(boundary_ratio),
                    visible_outside_negative_ratio=float(negative_ratio),
                    parent_area_ratio=float(pixels.shape[0] / max(np.count_nonzero(parent_binary), 1)),
                    split_mode=mode,
                )
            )
            region_id += 1
    return regions


def _build_regions_for_scene(args: argparse.Namespace, scene: str) -> tuple[MeasurementBank, list[RegionCandidate], dict[str, Any]]:
    bank = MeasurementBank.load(Path(args.bank_root) / scene / "measurement_bank.npz")
    scene_root = Path(args.scannet_root) / scene
    mask_dir = scene_root / f"output_{args.backbone}" / "mask"
    regions: list[RegionCandidate] = []
    skipped_masks = 0
    for frame_idx, frame_id in enumerate(np.asarray(bank.frame_ids, dtype=np.int64).tolist()):
        if not bool(bank.mask_frame_available[frame_idx]):
            continue
        mask = load_mask(mask_dir / f"{int(frame_id)}.png")
        if mask is None:
            continue
        positive_ids = np.asarray(bank.target_mask_id[frame_idx][bank.positive_observation[frame_idx]], dtype=np.int64)
        mask_ids = [int(v) for v in np.unique(positive_ids).tolist() if int(v) > 0]
        for mask_id in mask_ids:
            new_regions = _regions_for_frame_mask(
                bank=bank,
                frame_idx=int(frame_idx),
                frame_id=int(frame_id),
                mask=mask,
                mask_id=int(mask_id),
                next_region_id=len(regions),
                mode=args.mode,
                split_grid=int(args.split_grid),
                min_surfels=int(args.min_surfels),
                min_region_pixels=int(args.min_region_pixels),
                boundary_safe_px=float(args.boundary_safe_px),
                erode_px=int(args.erode_px),
                pixel_stride=int(args.pixel_stride),
                max_pixels_per_region=int(args.max_pixels_per_region),
            )
            if new_regions:
                regions.extend(new_regions)
            else:
                skipped_masks += 1
    num_regions_before_cap = int(len(regions))
    if len(regions) > int(args.max_regions_per_scene):
        regions.sort(
            key=lambda r: (
                -float(r.boundary_safe_ratio),
                float(r.visible_outside_negative_ratio),
                -int(r.surfels.shape[0]),
                -int(r.pixel_count),
                int(r.frame_id),
                int(r.mask_id),
            )
        )
        regions = regions[: int(args.max_regions_per_scene)]
        regions.sort(key=lambda r: int(r.region_id))
    summary = {
        "scene": scene,
        "num_regions_before_cap": int(num_regions_before_cap),
        "num_regions_after_cap": int(len(regions)),
        "skipped_masks_without_region": int(skipped_masks),
    }
    return bank, regions, summary


def _export_regions(
    *,
    args: argparse.Namespace,
    scene: str,
    bank: MeasurementBank,
    regions: list[RegionCandidate],
    output_config: str,
) -> tuple[dict[str, Any], list[RegionCandidate], list[np.ndarray]]:
    stream = ScanNetStream(seq_name=scene, backbone=args.backbone, root=args.scannet_root)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=output_config,
        export_nn_radius=float(args.export_nn_radius),
        export_support_mode="reuse_point_ids",
        export_score_mode="area",
        export_min_points_per_object=0,
    )
    pred_masks: list[np.ndarray] = []
    pred_scores: list[float] = []
    kept_regions: list[RegionCandidate] = []
    point_sets: list[np.ndarray] = []
    backproject_queries = 0
    backproject_hits = 0
    for region in regions:
        point_ids, _ = exporter._backproject_xy(region.frame_id, region.pixels_xy, nn_radius=float(args.export_nn_radius))
        point_ids = np.unique(point_ids.astype(np.int64, copy=False))
        backproject_queries += int(region.pixels_xy.shape[0])
        backproject_hits += int(point_ids.shape[0])
        if point_ids.shape[0] < int(args.min_export_points):
            continue
        mask = np.zeros((exporter.scene_points.shape[0],), dtype=bool)
        mask[point_ids] = True
        pred_masks.append(mask)
        score = float(np.sqrt(max(region.pixel_count, 1)) * (0.5 + 0.5 * region.boundary_safe_ratio))
        pred_scores.append(score)
        kept_regions.append(region)
        point_sets.append(point_ids)

    pred_dir = Path("data/prediction") / f"{output_config}_class_agnostic"
    tmp_dir = Path("data/TMP") / output_config
    pred_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if pred_masks:
        masks = np.stack(pred_masks, axis=1).astype(bool, copy=False)
        scores = np.asarray(pred_scores, dtype=np.float32)
        pre_points = np.flatnonzero(np.any(masks, axis=1)).astype(np.int64)
    else:
        masks = np.zeros((exporter.scene_points.shape[0], 0), dtype=bool)
        scores = np.zeros((0,), dtype=np.float32)
        pre_points = np.zeros((0,), dtype=np.int64)
    np.savez_compressed(
        pred_dir / f"{scene}.npz",
        pred_masks=masks,
        pred_score=scores,
        pred_classes=np.zeros((masks.shape[1],), dtype=np.int32),
    )
    np.save(tmp_dir / f"{scene}_pre_points.npy", pre_points)
    diag = {
        "num_candidate_regions": int(len(regions)),
        "num_exported_regions": int(len(kept_regions)),
        "num_scene_points": int(exporter.scene_points.shape[0]),
        "num_exported_points": int(pre_points.shape[0]),
        "exported_pre_ratio": float(pre_points.shape[0] / max(exporter.scene_points.shape[0], 1)),
        "export_backproject_queries": int(backproject_queries),
        "export_backproject_hits": int(backproject_hits),
        "export_nn_hit_rate": float(backproject_hits / max(backproject_queries, 1)),
        "mean_region_pixels": float(np.mean([r.pixel_count for r in kept_regions])) if kept_regions else 0.0,
        "mean_region_surfels": float(np.mean([r.surfels.shape[0] for r in kept_regions])) if kept_regions else 0.0,
        "mean_boundary_safe_ratio": float(np.mean([r.boundary_safe_ratio for r in kept_regions])) if kept_regions else 0.0,
        "mean_visible_outside_negative_ratio": float(np.mean([r.visible_outside_negative_ratio for r in kept_regions]))
        if kept_regions
        else 0.0,
        "regions_per_mask_frame": float(len(kept_regions) / max(int(np.count_nonzero(bank.mask_frame_available)), 1)),
    }
    return diag, kept_regions, point_sets


def _save_region_bank(args: argparse.Namespace, scene: str, kept_regions: list[RegionCandidate], point_sets: list[np.ndarray]) -> Path:
    out_dir = Path(args.region_root) / args.mode / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    surfel_offsets = [0]
    surfels_flat: list[int] = []
    point_offsets = [0]
    points_flat: list[int] = []
    rows: list[dict[str, Any]] = []
    for local_idx, (region, point_ids) in enumerate(zip(kept_regions, point_sets)):
        surfels_flat.extend(int(v) for v in region.surfels.tolist())
        surfel_offsets.append(len(surfels_flat))
        points_flat.extend(int(v) for v in point_ids.tolist())
        point_offsets.append(len(points_flat))
        rows.append(
            {
                "local_region_index": int(local_idx),
                "region_id": int(region.region_id),
                "frame_id": int(region.frame_id),
                "mask_id": int(region.mask_id),
                "component_id": int(region.component_id),
                "pixel_count": int(region.pixel_count),
                "surfel_count": int(region.surfels.shape[0]),
                "point_count": int(point_ids.shape[0]),
                "bbox_xyxy": list(region.bbox_xyxy),
                "boundary_safe_ratio": float(region.boundary_safe_ratio),
                "visible_outside_negative_ratio": float(region.visible_outside_negative_ratio),
                "parent_area_ratio": float(region.parent_area_ratio),
                "split_mode": region.split_mode,
            }
        )
    path = out_dir / "region_bank.npz"
    np.savez_compressed(
        path,
        frame_id=np.asarray([r.frame_id for r in kept_regions], dtype=np.int64),
        mask_id=np.asarray([r.mask_id for r in kept_regions], dtype=np.int64),
        component_id=np.asarray([r.component_id for r in kept_regions], dtype=np.int64),
        region_id=np.asarray([r.region_id for r in kept_regions], dtype=np.int64),
        pixel_count=np.asarray([r.pixel_count for r in kept_regions], dtype=np.int64),
        surfel_offsets=np.asarray(surfel_offsets, dtype=np.int64),
        surfel_indices=np.asarray(surfels_flat, dtype=np.int64),
        point_offsets=np.asarray(point_offsets, dtype=np.int64),
        point_ids=np.asarray(points_flat, dtype=np.int64),
        boundary_safe_ratio=np.asarray([r.boundary_safe_ratio for r in kept_regions], dtype=np.float32),
        visible_outside_negative_ratio=np.asarray([r.visible_outside_negative_ratio for r in kept_regions], dtype=np.float32),
        meta_json=np.asarray(
            json.dumps(
                json_safe(
                    {
                        "algorithm": "v15_mask_region_measurements",
                        "mode": args.mode,
                        "split_grid": int(args.split_grid),
                        "min_surfels": int(args.min_surfels),
                        "min_region_pixels": int(args.min_region_pixels),
                    }
                ),
                sort_keys=True,
            )
        ),
    )
    (out_dir / "region_bank_summary.json").write_text(json.dumps(json_safe(rows), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _gt_region_diagnostic(root: Path, scene: str, output_config: str) -> dict[str, Any]:
    pred_path = root / "data" / "prediction" / f"{output_config}_class_agnostic" / f"{scene}.npz"
    gt_path = root / "data" / "scannet" / "gt" / f"{scene}.txt"
    with np.load(pred_path) as data:
        pred_masks = np.asarray(data["pred_masks"], dtype=bool)
    gt_ids = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
    gt_masks, _, gt_areas = _gt_instance_masks(gt_ids)
    if pred_masks.shape[1] == 0:
        return {
            "region_purity_mean": 0.0,
            "region_purity_area_weighted": 0.0,
            "region_completeness_mean": 0.0,
            "cross_object_contamination_ratio": 0.0,
            "best_region_iou_per_gt_mean": 0.0,
            **{f"gt_best_region_iou_ge_{str(th).replace('.', 'p')}": 0 for th in THRESHOLDS},
        }
    pred_area = np.count_nonzero(pred_masks, axis=0).astype(np.float64)
    if gt_masks.shape[0] == 0:
        purity = np.zeros((pred_masks.shape[1],), dtype=np.float64)
        completeness = np.zeros((pred_masks.shape[1],), dtype=np.float64)
        best_iou = np.zeros((0,), dtype=np.float64)
    else:
        inter = gt_masks.astype(np.int64) @ pred_masks.astype(np.int64)
        best_gt = np.argmax(inter, axis=0)
        max_inter = inter[best_gt, np.arange(pred_masks.shape[1])].astype(np.float64)
        purity = max_inter / np.maximum(pred_area, 1.0)
        completeness = max_inter / np.maximum(gt_areas[best_gt].astype(np.float64), 1.0)
        union = gt_areas[:, None].astype(np.float64) + pred_area[None, :] - inter.astype(np.float64)
        iou = inter.astype(np.float64) / np.maximum(union, 1.0)
        best_iou = np.max(iou, axis=1) if iou.size else np.zeros((0,), dtype=np.float64)
    weighted = float(np.sum(purity * pred_area) / max(float(np.sum(pred_area)), 1.0))
    out = {
        "region_purity_mean": float(np.mean(purity)) if purity.size else 0.0,
        "region_purity_area_weighted": weighted,
        "region_completeness_mean": float(np.mean(completeness)) if completeness.size else 0.0,
        "cross_object_contamination_ratio": float(1.0 - weighted),
        "best_region_iou_per_gt_mean": float(np.mean(best_iou)) if best_iou.size else 0.0,
    }
    for th in THRESHOLDS:
        out[f"gt_best_region_iou_ge_{str(th).replace('.', 'p')}"] = int(np.count_nonzero(best_iou >= th))
    return out


def _evaluate(output_config: str, root: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(Path("data/prediction") / f"{output_config}_class_agnostic"),
        "--gt_path",
        "data/scannet/gt",
        "--dataset",
        "scannet",
        "--output_file",
        str(Path("data/evaluation/scannet") / f"{output_config}_class_agnostic.txt"),
        "--tmp_root",
        "data/TMP",
        "--tmp_config",
        output_config,
        "--no_class",
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    subprocess.run(cmd, cwd=str(root), check=True)


def _parse_metric(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"ap": None, "ap50": None, "ap25": None}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {"ap": None, "ap50": None, "ap25": None}
    parts = lines[-1].replace("\t", ",").split(",")
    try:
        return {"ap": float(parts[0]), "ap50": float(parts[1]), "ap25": float(parts[2])}
    except Exception:
        return {"ap": None, "ap50": None, "ap25": None}


def _write_visuals(args: argparse.Namespace, scene: str, regions: list[RegionCandidate]) -> list[str]:
    if int(args.visual_limit) <= 0 or not regions:
        return []
    out_dir = Path(args.visual_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_root = Path(args.scannet_root) / scene
    by_frame: dict[int, list[RegionCandidate]] = {}
    for region in regions[: int(args.visual_limit)]:
        by_frame.setdefault(int(region.frame_id), []).append(region)
    written: list[str] = []
    rng = np.random.default_rng(15)
    for frame_id, frame_regions in by_frame.items():
        rgb = load_rgb(scene_root / "color" / f"{int(frame_id)}.jpg")
        if rgb is None:
            mask = load_mask(scene_root / f"output_{args.backbone}" / "mask" / f"{int(frame_id)}.png")
            if mask is None:
                continue
            rgb = np.repeat((mask > 0)[..., None].astype(np.uint8) * 180, 3, axis=2)
        canvas = rgb.astype(np.float32)
        for region in frame_regions:
            color = rng.integers(40, 255, size=(3,), dtype=np.uint8).astype(np.float32)
            xy = region.pixels_xy
            if xy.size == 0:
                continue
            x = np.clip(xy[:, 0], 0, canvas.shape[1] - 1)
            y = np.clip(xy[:, 1], 0, canvas.shape[0] - 1)
            canvas[y, x] = 0.35 * canvas[y, x] + 0.65 * color
        out = out_dir / f"v15_region_overlay_{scene}_{int(frame_id)}.png"
        cv2.imwrite(str(out), cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        written.append(str(out))
    return written


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    return {
        "num_scenes": int(len(rows)),
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
            for key in numeric_keys
            if any(row.get(key) is not None for row in rows)
        },
    }


def _write_summary(prefix: Path, payload: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    rows = payload["scenes"]
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys() if key != "visuals"})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    summary = payload["summary"]
    metric = summary.get("candidate_metric", {})
    lines = [
        "# Stream4D v15 Mask-Region Measurement Diagnostic",
        "",
        "Candidate generation does not read GT. GT diagnostic fields are computed after export when requested.",
        "",
        "## Aggregate",
        "",
        f"- output_config: `{summary.get('output_config')}`",
        f"- mode: `{summary.get('mode')}`",
        f"- candidate AP/AP50/AP25: `{metric.get('ap')}` / `{metric.get('ap50')}` / `{metric.get('ap25')}`",
        f"- region_gate_pass: `{summary.get('region_gate_pass')}`",
        "",
        "| scene | regions | pre% | hit rate | purity | contamination | best region IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scene")),
                    str(row.get("num_exported_regions")),
                    f"{float(row.get('exported_pre_ratio', 0.0)) * 100.0:.4f}",
                    f"{float(row.get('export_nn_hit_rate', 0.0)):.4f}",
                    f"{float(row.get('region_purity_area_weighted', 0.0)):.4f}",
                    f"{float(row.get('cross_object_contamination_ratio', 0.0)):.4f}",
                    f"{float(row.get('best_region_iou_per_gt_mean', 0.0)):.4f}",
                ]
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--bank-root", default="outputs/v14_measurement_bank_bank16_cropformer")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--region-root", default="outputs/v15_mask_region_measurements")
    parser.add_argument("--summary-prefix", default="outputs/audit/v15_phase2/mask_region_measurement_probe5")
    parser.add_argument("--visual-dir", default="outputs/audit/v15_phase2/visuals")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--mode", choices=["component", "seed_voronoi", "boundary_core"], default="seed_voronoi")
    parser.add_argument("--split-grid", type=int, default=2)
    parser.add_argument("--min-surfels", type=int, default=5)
    parser.add_argument("--min-region-pixels", type=int, default=80)
    parser.add_argument("--max-regions-per-scene", type=int, default=600)
    parser.add_argument("--boundary-safe-px", type=float, default=3.0)
    parser.add_argument("--erode-px", type=int, default=2)
    parser.add_argument("--pixel-stride", type=int, default=3)
    parser.add_argument("--max-pixels-per-region", type=int, default=12000)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--min-export-points", type=int, default=20)
    parser.add_argument("--gt-diagnostic", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--visual-limit", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows: list[dict[str, Any]] = []
    for scene in read_seq_list(root / args.seq_list):
        bank, regions, build_summary = _build_regions_for_scene(args, scene)
        visuals = _write_visuals(args, scene, regions)
        export_diag, kept_regions, point_sets = _export_regions(
            args=args,
            scene=scene,
            bank=bank,
            regions=regions,
            output_config=args.output_config,
        )
        region_bank_path = _save_region_bank(args, scene, kept_regions, point_sets)
        row = {
            **build_summary,
            **export_diag,
            "scene": scene,
            "region_bank_path": str(region_bank_path),
            "visuals": visuals,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": bool(args.gt_diagnostic),
            "is_method_result": False,
            "is_diagnostic_only": True,
        }
        if args.gt_diagnostic:
            row.update(_gt_region_diagnostic(root, scene, args.output_config))
        rows.append(row)

    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(args.bank_root)],
        pre_points_policy="recompute",
        support_policy=f"v15_mask_region_measurement:{args.mode}",
        notes="v15 mask-region measurement candidate; generated without GT and not a method result.",
        extra={
            "algorithm": "v15_mask_region_measurements",
            "mode": args.mode,
            "eval_policy": "measurement_candidate_diagnostic",
            "prediction_config": args.output_config,
            "pre_points_config": args.output_config,
            "support_source": "own_mask_region_candidate",
            "geometry_source": "rgbd_eval_bridge",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "gt_selected_output": False,
            "forbidden_for_method_table": True,
            "alignment_source": "none",
            "alignment_used_for_prediction": False,
            "alignment_used_for_diagnostic": False,
            "summary_path": str(Path(args.summary_prefix).with_suffix(".json")),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=root, pred_suffix="class_agnostic")
    if not args.skip_eval:
        _evaluate(args.output_config, root)
    metric = _parse_metric(root / "data" / "evaluation" / "scannet" / f"{args.output_config}_class_agnostic.txt")
    aggregate = _aggregate(rows)
    mean = aggregate["numeric_mean"]
    region_gate_pass = bool(
        metric["ap50"] is not None
        and metric["ap25"] is not None
        and float(metric["ap50"]) >= 0.60
        and float(metric["ap25"]) >= 0.78
        and float(mean.get("exported_pre_ratio", 0.0)) >= 0.25
        and float(mean.get("region_purity_area_weighted", 0.0)) >= 0.70
        and float(mean.get("cross_object_contamination_ratio", 1.0)) <= 0.20
    )
    payload = {
        "summary": {
            **aggregate,
            "algorithm": "v15_mask_region_measurements",
            "output_config": args.output_config,
            "mode": args.mode,
            "candidate_metric": metric,
            "region_gate_pass": region_gate_pass,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": bool(args.gt_diagnostic),
            "is_method_result": False,
            "is_diagnostic_only": True,
        },
        "scenes": rows,
    }
    _write_summary(root / args.summary_prefix, payload)
    print(json.dumps(json_safe(payload["summary"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
