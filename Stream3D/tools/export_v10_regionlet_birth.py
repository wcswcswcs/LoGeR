from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _frame_ids_for_carrier_file(carrier_path: Path, num_frames: int) -> list[int]:
    summary_path = carrier_path.with_name(carrier_path.name.replace("carriers_", "").replace(".npz", "_summary.json"))
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            frame_ids = [int(value) for value in payload.get("frame_ids", [])]
            if len(frame_ids) == num_frames:
                return frame_ids
        except Exception:
            pass
    return list(range(num_frames))


def _fallback_frame_ids(stream: ScanNetStream, stride: int, max_frames: int) -> list[int]:
    return stream.frame_ids(stride=int(stride), max_frames=int(max_frames) if max_frames > 0 else None)


def _available_mask_frame_ids(stream: ScanNetStream, max_frames: int) -> list[int]:
    frame_ids = sorted(int(path.stem) for path in stream.mask_dir.glob("*.png") if path.stem.isdigit())
    if max_frames > 0:
        frame_ids = frame_ids[: int(max_frames)]
    return frame_ids


def _load_d4rt_seed_pixels(
    *,
    stream: ScanNetStream,
    scene_debug_dir: Path,
    min_visibility: float,
    min_confidence: float,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any], list[int]]:
    seeds: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    frame_ids_seen: list[int] = []
    total_samples = 0
    valid_samples = 0
    positive_samples = 0
    carrier_paths = sorted(scene_debug_dir.glob("carriers_window*.npz"))
    if not carrier_paths:
        return {}, {"d4rt_seed_status": "missing_carriers", "d4rt_seed_positive_samples": 0}, []

    for carrier_path in carrier_paths:
        with np.load(carrier_path) as data:
            uv_pred = np.asarray(data["uv_pred"], dtype=np.float32)
            visibility = np.asarray(data["visibility_prob"], dtype=np.float32)
            confidence = np.asarray(data["confidence_prob"], dtype=np.float32)
        frame_ids = _frame_ids_for_carrier_file(carrier_path, uv_pred.shape[0])
        frame_ids_seen.extend(int(v) for v in frame_ids)
        for local_idx, frame_id in enumerate(frame_ids):
            try:
                mask = stream.load_mask(int(frame_id))
            except FileNotFoundError:
                continue
            h, w = mask.shape[:2]
            uv = uv_pred[local_idx]
            total_samples += int(uv.shape[0])
            x = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int64)
            y = np.rint(uv[:, 1] * float(max(h - 1, 1))).astype(np.int64)
            ok = (
                np.isfinite(uv).all(axis=1)
                & (x >= 0)
                & (x < w)
                & (y >= 0)
                & (y < h)
                & (visibility[local_idx] >= float(min_visibility))
                & (confidence[local_idx] >= float(min_confidence))
            )
            valid_samples += int(np.count_nonzero(ok))
            if not np.any(ok):
                continue
            mask_ids = mask[y[ok], x[ok]].astype(np.int64)
            xs = x[ok]
            ys = y[ok]
            positive = mask_ids > 0
            positive_samples += int(np.count_nonzero(positive))
            for px, py, mask_id in zip(xs[positive].tolist(), ys[positive].tolist(), mask_ids[positive].tolist()):
                seeds[(int(frame_id), int(mask_id))].append((int(px), int(py)))

    seed_arrays = {
        key: np.asarray(sorted(set(value)), dtype=np.int64).reshape(-1, 2)
        for key, value in seeds.items()
        if value
    }
    return seed_arrays, {
        "d4rt_seed_status": "ok",
        "d4rt_seed_carrier_files": int(len(carrier_paths)),
        "d4rt_seed_total_samples": int(total_samples),
        "d4rt_seed_valid_samples": int(valid_samples),
        "d4rt_seed_positive_samples": int(positive_samples),
        "d4rt_seed_positive_rate": float(positive_samples / max(valid_samples, 1)),
        "d4rt_seed_mask_observations": int(len(seed_arrays)),
    }, sorted(set(frame_ids_seen))


def _connected_regions(binary: np.ndarray, min_area: int) -> list[np.ndarray]:
    binary = np.asarray(binary, dtype=bool)
    if not np.any(binary):
        return []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    regions: list[np.ndarray] = []
    for label_id in range(1, int(num_labels)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        regions.append(labels == label_id)
    return regions


def _mask_core_regions(mask_bool: np.ndarray, min_area: int, boundary_px: float) -> list[np.ndarray]:
    regions: list[np.ndarray] = []
    for component in _connected_regions(mask_bool, min_area=1):
        dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)
        core = component & (dist >= float(boundary_px))
        if int(np.count_nonzero(core)) < int(min_area):
            finite = dist[component]
            if finite.size:
                core = component & (dist >= float(np.percentile(finite, 60)))
        regions.extend(_connected_regions(core, min_area=min_area))
    return regions


def _depth_split_regions(mask_bool: np.ndarray, depth: np.ndarray, min_area: int, depth_bin_m: float) -> list[np.ndarray]:
    valid = mask_bool & np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return []
    values = depth[valid]
    median = float(np.median(values))
    width = max(float(depth_bin_m), 1e-3)
    bins = np.floor((depth - median) / width).astype(np.int32)
    regions: list[np.ndarray] = []
    for bin_id in np.unique(bins[valid]).tolist():
        regions.extend(_connected_regions(valid & (bins == int(bin_id)), min_area=min_area))
    return regions


def _seed_near_regions(mask_bool: np.ndarray, seeds_xy: np.ndarray, min_area: int, radius_px: int) -> list[np.ndarray]:
    if seeds_xy.size == 0 or not np.any(mask_bool):
        return []
    h, w = mask_bool.shape[:2]
    seed_map = np.zeros((h, w), dtype=np.uint8)
    x = np.clip(seeds_xy[:, 0].astype(np.int64), 0, w - 1)
    y = np.clip(seeds_xy[:, 1].astype(np.int64), 0, h - 1)
    seed_map[y, x] = 1
    radius_px = max(1, int(radius_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px * 2 + 1, radius_px * 2 + 1))
    seed_regions = cv2.dilate(seed_map, kernel, iterations=1).astype(bool) & mask_bool
    regions = _connected_regions(seed_regions, min_area=min_area)
    if regions:
        return regions
    return _connected_regions(seed_map.astype(bool) & mask_bool, min_area=1)


def _variant_regions(
    *,
    variant: str,
    mask_bool: np.ndarray,
    depth: np.ndarray,
    seeds_xy: np.ndarray,
    min_area_2d: int,
    boundary_px: float,
    depth_bin_m: float,
    seed_radius_px: int,
) -> tuple[list[np.ndarray], str]:
    if variant == "R0_full_mask":
        return _connected_regions(mask_bool, min_area=min_area_2d), "full_mask"
    if variant == "R1_mask_core":
        return _mask_core_regions(mask_bool, min_area=min_area_2d, boundary_px=boundary_px), "mask_core_distance_transform"
    if variant == "R2_depth_split":
        return _depth_split_regions(mask_bool, depth=depth, min_area=min_area_2d, depth_bin_m=depth_bin_m), "depth_split"
    if variant == "R3_d4rt_seeded":
        return _seed_near_regions(mask_bool, seeds_xy=seeds_xy, min_area=min_area_2d, radius_px=seed_radius_px), "d4rt_seed_near"
    if variant == "R4_combined":
        depth_regions = _depth_split_regions(mask_bool, depth=depth, min_area=max(1, min_area_2d // 2), depth_bin_m=depth_bin_m)
        seed_regions = _seed_near_regions(mask_bool, seeds_xy=seeds_xy, min_area=max(1, min_area_2d // 2), radius_px=seed_radius_px)
        if not depth_regions and not seed_regions:
            return [], "combined_empty"
        if depth_regions and not seed_regions:
            return [item for item in depth_regions if int(np.count_nonzero(item)) >= min_area_2d], "combined_depth_fallback_no_seeds"
        if seed_regions and not depth_regions:
            return [item for item in seed_regions if int(np.count_nonzero(item)) >= min_area_2d], "combined_seed_fallback_no_depth"
        out: list[np.ndarray] = []
        for d_region in depth_regions:
            for s_region in seed_regions:
                inter = d_region & s_region
                if int(np.count_nonzero(inter)) >= int(min_area_2d):
                    out.extend(_connected_regions(inter, min_area=min_area_2d))
        return out, "combined_depth_seed_intersection"
    raise ValueError(f"Unsupported variant: {variant}")


def _sample_pixels(region: np.ndarray, max_pixels: int) -> np.ndarray:
    ys, xs = np.where(region)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if max_pixels > 0 and ys.size > int(max_pixels):
        keep = np.linspace(0, ys.size - 1, num=int(max_pixels), dtype=np.int64)
        xs = xs[keep]
        ys = ys[keep]
    return np.stack([xs.astype(np.int64), ys.astype(np.int64)], axis=1)


def _regionlet_stats(
    *,
    region: np.ndarray,
    depth: np.ndarray,
    seeds_xy: np.ndarray,
    point_ids: np.ndarray,
    frame_id: int,
    mask_id: int,
    split_reason: str,
) -> dict[str, Any]:
    values = depth[region]
    values = values[np.isfinite(values) & (values > 0.0)]
    surfel_count = 0
    if seeds_xy.size:
        h, w = region.shape[:2]
        x = np.clip(seeds_xy[:, 0].astype(np.int64), 0, w - 1)
        y = np.clip(seeds_xy[:, 1].astype(np.int64), 0, h - 1)
        surfel_count = int(np.count_nonzero(region[y, x]))
    return {
        "frame_id": int(frame_id),
        "mask_id": int(mask_id),
        "split_reason": split_reason,
        "area_2d": int(np.count_nonzero(region)),
        "area_3d": int(np.unique(point_ids).shape[0]),
        "num_d4rt_surfels": int(surfel_count),
        "surfel_density": float(surfel_count / max(int(np.count_nonzero(region)), 1)),
        "depth_variance": float(np.var(values)) if values.size else None,
        "empty_3d": bool(point_ids.size == 0),
    }


def _overlay_regionlets(
    stream: ScanNetStream,
    frame_id: int,
    frame_regions: list[dict[str, Any]],
    output_dir: Path,
    method: str,
    max_regions: int,
) -> None:
    if not frame_regions:
        return
    rgb = stream.load_rgb(int(frame_id))
    first_region = np.asarray(frame_regions[0]["region"], dtype=bool)
    if rgb.shape[:2] != first_region.shape[:2]:
        rgb = cv2.resize(rgb, (first_region.shape[1], first_region.shape[0]), interpolation=cv2.INTER_AREA)
    overlay = rgb.copy()
    rng = np.random.default_rng(abs(hash((stream.seq_name, method, frame_id))) % (2**32))
    for item in frame_regions[: int(max_regions)]:
        region = item["region"]
        color = rng.integers(32, 255, size=(3,), dtype=np.uint8)
        overlay[region] = (0.55 * overlay[region] + 0.45 * color).astype(np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{method}_frame{int(frame_id):06d}_regionlets.png"
    cv2.imwrite(str(png_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    sidecar = {
        "scene": stream.seq_name,
        "method": method,
        "frame_id": int(frame_id),
        "num_regionlets_in_frame": int(len(frame_regions)),
        "failure_tags": [],
    }
    png_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")


def _apply_regionlet_point_wta(object_dict: dict[int, dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[str, float]]:
    claims: dict[int, list[tuple[int, float]]] = defaultdict(list)
    total_assignments = 0
    for object_id, value in object_dict.items():
        stats = value.get("regionlet_stats", {})
        area_3d = float(stats.get("area_3d", 0.0))
        surfels = float(stats.get("num_d4rt_surfels", 0.0))
        depth_var = float(stats.get("depth_variance") or 0.0)
        score = area_3d + 8.0 * surfels - 0.1 * depth_var
        for point_id in np.asarray(value.get("point_ids", []), dtype=np.int64).tolist():
            claims[int(point_id)].append((int(object_id), float(score)))
            total_assignments += 1
    if not claims:
        return object_dict, {
            "regionlet_wta_enabled": 1.0,
            "regionlet_wta_conflict_points": 0.0,
            "regionlet_wta_removed_assignment_rate": 0.0,
        }
    winners = {
        point_id: max(values, key=lambda item: (float(item[1]), -int(item[0])))[0]
        for point_id, values in claims.items()
    }
    reassigned: dict[int, set[int]] = {int(object_id): set() for object_id in object_dict}
    conflict_points = 0
    removed = 0
    for point_id, values in claims.items():
        if len(values) > 1:
            conflict_points += 1
            removed += len(values) - 1
        reassigned[winners[point_id]].add(int(point_id))
    out: dict[int, dict[str, Any]] = {}
    for object_id, value in object_dict.items():
        kept = reassigned.get(int(object_id), set())
        if not kept:
            continue
        new_value = dict(value)
        new_value["point_ids"] = np.asarray(sorted(kept), dtype=np.int64)
        out[int(object_id)] = new_value
    return out, {
        "regionlet_wta_enabled": 1.0,
        "regionlet_wta_conflict_points": float(conflict_points),
        "regionlet_wta_removed_assignment_rate": float(removed / max(total_assignments, 1)),
    }


def _point_conflict_rate(object_dict: dict[int, dict[str, Any]]) -> tuple[int, int, float]:
    point_owner_counts = Counter()
    for value in object_dict.values():
        for point_id in np.asarray(value.get("point_ids", []), dtype=np.int64).tolist():
            point_owner_counts[int(point_id)] += 1
    conflict_points = int(sum(1 for count in point_owner_counts.values() if count > 1))
    union_points = int(len(point_owner_counts))
    return conflict_points, union_points, float(conflict_points / max(union_points, 1))


def _export_scene(args: argparse.Namespace, seq_name: str) -> dict[str, Any]:
    stream = ScanNetStream(seq_name=seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_support_mode="reuse_point_ids",
        export_min_points_per_object=int(args.min_points_per_object),
        export_score_mode=args.export_score_mode,
    )
    scene_debug_dir = Path(args.debug_root) / seq_name if args.debug_root else Path("__missing__")
    seeds_by_mask, seed_diag, d4rt_frame_ids = _load_d4rt_seed_pixels(
        stream=stream,
        scene_debug_dir=scene_debug_dir,
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
    )
    if args.frame_source == "available_masks":
        frame_ids = _available_mask_frame_ids(stream, int(args.max_mask_frames))
    else:
        frame_ids = d4rt_frame_ids or _fallback_frame_ids(stream, int(args.frame_stride), int(args.max_frames))
    object_dict: dict[int, dict[str, Any]] = {}
    regionlet_rows: list[dict[str, Any]] = []
    frame_visual_regions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    split_reasons = Counter()
    mask_count = 0

    for frame_id in frame_ids:
        try:
            mask = stream.load_mask(int(frame_id))
            depth = stream.load_depth(int(frame_id))
        except FileNotFoundError:
            continue
        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_ids = [int(v) for v in np.unique(mask) if int(v) > 0]
        mask_count += len(mask_ids)
        for mask_id in mask_ids:
            mask_bool = mask == int(mask_id)
            seeds_xy = seeds_by_mask.get((int(frame_id), int(mask_id)), np.empty((0, 2), dtype=np.int64))
            regions, split_reason = _variant_regions(
                variant=args.variant,
                mask_bool=mask_bool,
                depth=depth,
                seeds_xy=seeds_xy,
                min_area_2d=int(args.min_area_2d),
                boundary_px=float(args.boundary_px),
                depth_bin_m=float(args.depth_bin_m),
                seed_radius_px=int(args.seed_radius_px),
            )
            split_reasons[split_reason] += len(regions)
            for region in regions:
                xy = _sample_pixels(region, max_pixels=int(args.max_pixels_per_regionlet))
                point_ids, _ = exporter._backproject_xy(int(frame_id), xy, nn_radius=float(args.nn_radius))
                point_ids = np.unique(point_ids.astype(np.int64))
                stats = _regionlet_stats(
                    region=region,
                    depth=depth,
                    seeds_xy=seeds_xy,
                    point_ids=point_ids,
                    frame_id=int(frame_id),
                    mask_id=int(mask_id),
                    split_reason=split_reason,
                )
                regionlet_rows.append(stats)
                if point_ids.shape[0] < int(args.min_points_per_object):
                    continue
                object_id = len(object_dict)
                object_dict[object_id] = {
                    "point_ids": point_ids,
                    "mask_list": [(int(frame_id), int(mask_id), float(stats["area_2d"]))],
                    "carrier_ids": np.empty((0,), dtype=np.int64),
                    "regionlet_stats": stats,
                }
                if len(frame_visual_regions[int(frame_id)]) < int(args.visual_max_regions_per_frame):
                    frame_visual_regions[int(frame_id)].append({"region": region, "stats": stats})

    raw_conflict_points, raw_union_points, raw_conflict_rate = _point_conflict_rate(object_dict)
    wta_diag = {"regionlet_wta_enabled": 0.0, "regionlet_wta_conflict_points": 0.0, "regionlet_wta_removed_assignment_rate": 0.0}
    if args.enable_point_wta:
        object_dict, wta_diag = _apply_regionlet_point_wta(object_dict)
    export_diag = exporter.export_object_dict_points(object_dict)
    _, _, final_conflict_rate = _point_conflict_rate(object_dict)
    if args.write_visualizations:
        visual_root = Path(args.visualization_root) / "v10_regionlet_birth" / seq_name
        for frame_id in sorted(frame_visual_regions)[: int(args.visual_max_frames_per_scene)]:
            _overlay_regionlets(
                stream=stream,
                frame_id=frame_id,
                frame_regions=frame_visual_regions[frame_id],
                output_dir=visual_root,
                method=args.output_config,
                max_regions=int(args.visual_max_regions_per_frame),
            )

    areas_2d = [float(row["area_2d"]) for row in regionlet_rows]
    areas_3d = [float(row["area_3d"]) for row in regionlet_rows]
    seed_counts = [float(row["num_d4rt_surfels"]) for row in regionlet_rows]
    depth_vars = [float(row["depth_variance"]) for row in regionlet_rows if row.get("depth_variance") is not None]
    summary = {
        "seq_name": seq_name,
        "variant": args.variant,
        "num_frames": int(len(frame_ids)),
        "num_masks": int(mask_count),
        "num_regionlets": int(len(regionlet_rows)),
        "num_regionlets_per_frame": float(len(regionlet_rows) / max(len(frame_ids), 1)),
        "num_regionlets_per_mask": float(len(regionlet_rows) / max(mask_count, 1)),
        "num_exported_objects": int(len(object_dict)),
        "empty_regionlet_ratio": float(np.mean([row["empty_3d"] for row in regionlet_rows])) if regionlet_rows else 0.0,
        "over_small_regionlet_ratio": float(np.mean([row["area_3d"] < int(args.min_points_per_object) for row in regionlet_rows]))
        if regionlet_rows
        else 0.0,
        "regionlet_area_2d_mean": float(np.mean(areas_2d)) if areas_2d else 0.0,
        "regionlet_area_2d_median": float(np.median(areas_2d)) if areas_2d else 0.0,
        "regionlet_area_3d_mean": float(np.mean(areas_3d)) if areas_3d else 0.0,
        "regionlet_depth_variance_mean": float(np.mean(depth_vars)) if depth_vars else None,
        "regionlet_surfel_count_mean": float(np.mean(seed_counts)) if seed_counts else 0.0,
        "regionlet_conflict_points_before_wta": int(raw_conflict_points),
        "regionlet_union_points_before_wta": int(raw_union_points),
        "regionlet_conflict_rate_before_export": float(raw_conflict_rate),
        "regionlet_conflict_rate_after_wta": float(final_conflict_rate),
        "split_reasons": dict(split_reasons),
        **seed_diag,
        **wta_diag,
        **export_diag,
    }
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_summary_path = out_dir / f"{args.output_config}_{seq_name}_summary.json"
    scene_summary_path.write_text(
        json.dumps(_json_safe({"summary": summary, "regionlets": regionlet_rows}), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _write_aggregate(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> None:
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    numeric_keys = sorted(
        {
            key
            for row in summaries
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool)
        }
    )
    aggregate = {
        "args": vars(args),
        "algorithm": "v10_regionlet_birth",
        "variant": args.variant,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": False,
        "is_method_result": True,
        "is_diagnostic_only": False,
        "num_scenes": len(summaries),
        "scenes": summaries,
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in summaries if key in row]))
            for key in numeric_keys
            if any(key in row for row in summaries)
        },
    }
    aggregate_path = out_dir / f"{args.output_config}_summary.json"
    aggregate_path.write_text(json.dumps(_json_safe(aggregate), indent=2, sort_keys=True), encoding="utf-8")

    csv_path = out_dir / f"{args.output_config}_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seq_name"] + numeric_keys)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in ["seq_name"] + numeric_keys})

    md_path = out_dir / f"{args.output_config}_summary.md"
    lines = [
        f"# {args.output_config}",
        "",
        "| scene | frames | masks | regionlets | objects | regionlets/mask | empty | small | conflict | seed mean | export points |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {scene} | {frames} | {masks} | {regs} | {objs} | {rpm:.3f} | {empty:.3f} | {small:.3f} | {conf:.3f} | {seed:.3f} | {points:.0f} |".format(
                scene=row.get("seq_name"),
                frames=int(row.get("num_frames", 0)),
                masks=int(row.get("num_masks", 0)),
                regs=int(row.get("num_regionlets", 0)),
                objs=int(row.get("num_exported_objects", 0)),
                rpm=float(row.get("num_regionlets_per_mask", 0.0)),
                empty=float(row.get("empty_regionlet_ratio", 0.0)),
                small=float(row.get("over_small_regionlet_ratio", 0.0)),
                conf=float(row.get("regionlet_conflict_rate_before_export", 0.0)),
                seed=float(row.get("regionlet_surfel_count_mean", 0.0)),
                points=float(row.get("num_exported_points", 0.0)),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = build_prediction_manifest(
        root=".",
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(args.debug_root)] if args.debug_root else [],
        pre_points_policy="recompute",
        support_policy=f"v10_regionlet_birth:{args.variant}",
        notes=(
            "v10 regionlet object birth prototype. 2D masks are split into local regionlets before "
            "RGB-D evaluation-bridge backprojection. D4RT carriers are used only as non-GT seed evidence "
            "for R3/R4 variants. No ScanNet instance labels are read by the exporter."
        ),
        extra={
            "algorithm": "v10_regionlet_birth",
            "variant": args.variant,
            "frame_source": args.frame_source,
            "enable_point_wta": bool(args.enable_point_wta),
            "eval_policy": args.eval_policy,
            "prediction_config": args.output_config,
            "pre_points_config": args.output_config,
            "support_source": "own",
            "geometry_source": "mixed" if "d4rt" in args.variant.lower() or args.variant == "R4_combined" else "rgbd_eval_bridge",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "is_diagnostic_only": False,
            "is_method_result": True,
            "summary_path": str(aggregate_path),
            "seq_list": str(args.seq_list),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v10 regionlet object-birth ScanNet predictions.")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument(
        "--variant",
        choices=["R0_full_mask", "R1_mask_core", "R2_depth_split", "R3_d4rt_seeded", "R4_combined"],
        required=True,
    )
    parser.add_argument("--debug-root", default="")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--summary-root", default="outputs/v10_regionlet_birth")
    parser.add_argument("--frame-source", choices=["d4rt_or_fallback", "available_masks"], default="d4rt_or_fallback")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-mask-frames", type=int, default=32)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-area-2d", type=int, default=64)
    parser.add_argument("--boundary-px", type=float, default=4.0)
    parser.add_argument("--depth-bin-m", type=float, default=0.20)
    parser.add_argument("--seed-radius-px", type=int, default=16)
    parser.add_argument("--max-pixels-per-regionlet", type=int, default=8000)
    parser.add_argument("--nn-radius", type=float, default=0.05)
    parser.add_argument("--min-points-per-object", type=int, default=20)
    parser.add_argument("--enable-point-wta", action="store_true")
    parser.add_argument("--export-score-mode", default="area", choices=["one", "area", "reliability", "observations", "dense_quality", "selection_quality"])
    parser.add_argument("--eval-policy", default="own_recompute_paper_style")
    parser.add_argument("--write-visualizations", action="store_true")
    parser.add_argument("--visualization-root", default="outputs/visualization")
    parser.add_argument("--visual-max-frames-per-scene", type=int, default=2)
    parser.add_argument("--visual-max-regions-per-frame", type=int, default=80)
    args = parser.parse_args()

    summaries = [_export_scene(args, seq_name) for seq_name in _read_seq_list(Path(args.seq_list))]
    _write_aggregate(args, summaries)
    print(json.dumps(_json_safe({"output_config": args.output_config, "scenes": summaries}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
