#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS_DIR))

from serve_v98_1_da3_gt_dense_rgb_sim3_viewer import (  # noqa: E402
    _apply_sim3,
    _camera_residual_for_transform,
    _camera_rotation_residual_degrees,
    _fit_pose_orientation_sim3,
    _fit_trajectory_sim3,
    _json_default,
    _load_da3_dense_points,
    _load_da3_manifest,
    _read_gt_point_cloud,
    _refine_surface_sim3,
    _residual_stats,
    _sample_indices,
)


PHASE1 = ROOT / "Stream3D" / "outputs" / "audit" / "v98_phase1_provider_contract"
DEFAULT_INPUT_DIR = PHASE1 / "da3_streaming_d4rt32o3_scene0050_input119"
DEFAULT_OUTPUT_ROOT = ROOT / "Stream3D" / "outputs" / "audit" / "v98_1_da3_variant_geometry_quality_scene0050"

VARIANT_SPECS = {
    "streaming_base": {
        "display_name": "DA3-Streaming (DA3-BASE)",
        "model": "DA3-BASE",
        "repo_id": "depth-anything/DA3-BASE",
        "root": PHASE1 / "da3_streaming_d4rt32o3_scene0050_base_input119",
        "log": PHASE1 / "da3_streaming_d4rt32o3_scene0050_base_input119.log",
    },
    "small": {
        "display_name": "DA3-SMALL",
        "model": "DA3-SMALL",
        "repo_id": "depth-anything/DA3-SMALL",
        "root": PHASE1 / "da3_streaming_d4rt32o3_scene0050_small_input119",
        "log": PHASE1 / "da3_streaming_d4rt32o3_scene0050_small_input119.log",
    },
    "large": {
        "display_name": "DA3-LARGE",
        "model": "DA3-LARGE",
        "repo_id": "depth-anything/DA3-LARGE",
        "root": PHASE1 / "da3_streaming_d4rt32o3_scene0050_large_input119",
        "log": PHASE1 / "da3_streaming_d4rt32o3_scene0050_large_input119.log",
    },
    "giant": {
        "display_name": "DA3-GIANT",
        "model": "DA3-GIANT",
        "repo_id": "depth-anything/DA3-GIANT",
        "root": PHASE1 / "da3_streaming_d4rt32o3_scene0050_giant_input119",
        "log": PHASE1 / "da3_streaming_d4rt32o3_scene0050_giant_input119.log",
    },
    "nested_giant_large": {
        "display_name": "DA3NESTED-GIANT-LARGE",
        "model": "DA3NESTED-GIANT-LARGE",
        "repo_id": "depth-anything/DA3NESTED-GIANT-LARGE",
        "root": PHASE1 / "da3_streaming_d4rt32o3_scene0050_nested_giant_large_input119",
        "log": PHASE1 / "da3_streaming_d4rt32o3_scene0050_nested_giant_large_input119.log",
    },
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _stable_variant_offset(variant_key: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(variant_key))


def _parse_da3_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"log_path": str(path), "exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    processing_match = re.search(
        r"Processing\s+(\d+)\s+images\s+in\s+(\d+)\s+chunks\s+of\s+size\s+(\d+)\s+with\s+(\d+)\s+overlap",
        text,
    )
    total_points_match = re.search(r"Merge completed!\s+Total points:\s+(\d+)", text)
    return {
        "log_path": str(path),
        "exists": True,
        "completed": "DA3-Streaming done." in text,
        "frame_count": int(processing_match.group(1)) if processing_match else None,
        "chunk_count": int(processing_match.group(2)) if processing_match else None,
        "chunk_size": int(processing_match.group(3)) if processing_match else None,
        "overlap": int(processing_match.group(4)) if processing_match else None,
        "forward_pass_times_sec": [float(v) for v in re.findall(r"Model Forward Pass Done\. Time:\s+([0-9.eE+-]+)\s+seconds", text)],
        "overlap_matched_points": [int(v) for v in re.findall(r"The number of corresponding points matched:\s+(\d+)", text)],
        "overlap_alignment_mean_errors": [float(v) for v in re.findall(r"Mean error:\s+([0-9.eE+-]+)", text)],
        "combined_pcd_point_count": int(total_points_match.group(1)) if total_points_match else None,
    }


def _fscore_row(forward_dist: np.ndarray, backward_dist: np.ndarray, threshold: float) -> dict[str, float]:
    precision = float(np.mean(forward_dist <= threshold)) if forward_dist.size else float("nan")
    recall = float(np.mean(backward_dist <= threshold)) if backward_dist.size else float("nan")
    denom = precision + recall
    fscore = float(2.0 * precision * recall / denom) if denom > 0.0 else 0.0
    return {
        "threshold_m": float(threshold),
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def _chamfer_metrics(
    *,
    source_aligned: np.ndarray,
    target_gt: np.ndarray,
    target_gt_tree: cKDTree,
    thresholds: list[float],
) -> dict[str, Any]:
    source = np.asarray(source_aligned, dtype=np.float64)
    target = np.asarray(target_gt, dtype=np.float64)
    finite_source = np.isfinite(source).all(axis=1)
    source = source[finite_source]
    if source.shape[0] == 0:
        raise RuntimeError("empty finite source point cloud for Chamfer metrics")
    forward_dist, _ = target_gt_tree.query(source, k=1)
    source_tree = cKDTree(source)
    backward_dist, _ = source_tree.query(target, k=1)
    return {
        "source_point_count": int(source.shape[0]),
        "target_gt_point_count": int(target.shape[0]),
        "accuracy_da3_to_gt_m": _residual_stats(forward_dist),
        "completeness_gt_to_da3_m": _residual_stats(backward_dist),
        "chamfer_l2_mean_m": float(0.5 * (np.mean(forward_dist) + np.mean(backward_dist))),
        "chamfer_l2_squared_mean_m2": float(0.5 * (np.mean(forward_dist**2) + np.mean(backward_dist**2))),
        "fscore": {f"{threshold:.2f}m": _fscore_row(forward_dist, backward_dist, threshold) for threshold in thresholds},
    }


def _metric_row(variant_key: str, variant: dict[str, Any], transform_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "variant_key": variant_key,
        "display_name": variant["display_name"],
        "model": variant["model"],
        "transform": transform_name,
        "chamfer_l2_mean_m": metrics["chamfer_l2_mean_m"],
        "chamfer_l2_squared_mean_m2": metrics["chamfer_l2_squared_mean_m2"],
        "accuracy_mean_m": metrics["accuracy_da3_to_gt_m"]["mean"],
        "accuracy_p50_m": metrics["accuracy_da3_to_gt_m"]["p50"],
        "accuracy_p90_m": metrics["accuracy_da3_to_gt_m"]["p90"],
        "accuracy_p95_m": metrics["accuracy_da3_to_gt_m"]["p95"],
        "completeness_mean_m": metrics["completeness_gt_to_da3_m"]["mean"],
        "completeness_p50_m": metrics["completeness_gt_to_da3_m"]["p50"],
        "completeness_p90_m": metrics["completeness_gt_to_da3_m"]["p90"],
        "completeness_p95_m": metrics["completeness_gt_to_da3_m"]["p95"],
    }
    for key, value in metrics["fscore"].items():
        prefix = key.replace(".", "p")
        row[f"{prefix}_precision"] = value["precision"]
        row[f"{prefix}_recall"] = value["recall"]
        row[f"{prefix}_fscore"] = value["fscore"]
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ["min", "p50", "p90", "p95", "p99", "max"]}
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def _load_depth_meters(path: Path, depth_scale: float) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    return depth / float(depth_scale)


def _best_depth_abs_error(depth_m: np.ndarray, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
    h, w = depth_m.shape
    best = np.full(u.shape, np.inf, dtype=np.float64)
    for dv in (-1, 0, 1):
        vv = v + dv
        valid_v = (vv >= 0) & (vv < h)
        for du in (-1, 0, 1):
            uu = u + du
            inside = valid_v & (uu >= 0) & (uu < w)
            if not np.any(inside):
                continue
            observed = depth_m[vv[inside], uu[inside]].astype(np.float64)
            valid_depth = observed > 0.0
            if not np.any(valid_depth):
                continue
            idx_inside = np.flatnonzero(inside)
            idx = idx_inside[valid_depth]
            err = np.abs(z[idx] - observed[valid_depth])
            best[idx] = np.minimum(best[idx], err)
    return best


def _filter_gt_to_input_visible(
    *,
    gt_points: np.ndarray,
    manifest: Any,
    scene_root: Path,
    depth_scale: float,
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
    min_observations: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    intrinsic_path = scene_root / "intrinsic" / "intrinsic_depth.txt"
    intrinsic = np.loadtxt(intrinsic_path).reshape(4, 4)[:3, :3].astype(np.float64)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    counts = np.zeros(gt_points.shape[0], dtype=np.uint16)
    frame_rows = manifest.sort_values("da3_frame_index").reset_index(drop=True)
    used_frames = 0
    skipped_frames: list[int] = []
    for row in frame_rows.itertuples(index=False):
        frame_id = int(row.frame_id)
        pose_path = scene_root / "pose" / f"{frame_id}.txt"
        depth_path = scene_root / "depth" / f"{frame_id}.png"
        if not pose_path.is_file() or not depth_path.is_file():
            skipped_frames.append(frame_id)
            continue
        c2w = np.loadtxt(pose_path).reshape(4, 4).astype(np.float64)
        if not np.isfinite(c2w).all():
            skipped_frames.append(frame_id)
            continue
        w2c = np.linalg.inv(c2w)
        depth_m = _load_depth_meters(depth_path, depth_scale=depth_scale)
        h, w = depth_m.shape
        used_frames += 1
        for start in range(0, gt_points.shape[0], int(batch_size)):
            end = min(gt_points.shape[0], start + int(batch_size))
            pts = np.asarray(gt_points[start:end], dtype=np.float64)
            cam = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
            z = cam[:, 2]
            valid_z = np.isfinite(z) & (z > 0.05)
            u_float = fx * (cam[:, 0] / z) + cx
            v_float = fy * (cam[:, 1] / z) + cy
            u = np.rint(u_float).astype(np.int64)
            v = np.rint(v_float).astype(np.int64)
            inside = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not np.any(inside):
                continue
            local_idx = np.flatnonzero(inside)
            best_err = _best_depth_abs_error(depth_m, u[inside], v[inside], z[inside])
            tolerance = np.maximum(float(depth_abs_tolerance), float(depth_rel_tolerance) * z[inside])
            visible_local = local_idx[np.isfinite(best_err) & (best_err <= tolerance)]
            if visible_local.size:
                counts[start:end][visible_local] += 1
    visible = counts >= int(min_observations)
    info = {
        "mode": "input_visible_depth_consistent",
        "scene_root": str(scene_root),
        "intrinsic_depth": str(intrinsic_path),
        "depth_scale": float(depth_scale),
        "depth_abs_tolerance_m": float(depth_abs_tolerance),
        "depth_rel_tolerance": float(depth_rel_tolerance),
        "min_observations": int(min_observations),
        "batch_size": int(batch_size),
        "input_frame_count": int(frame_rows.shape[0]),
        "used_depth_frame_count": int(used_frames),
        "skipped_frame_count": int(len(skipped_frames)),
        "skipped_frame_ids": skipped_frames[:20],
        "full_gt_point_count": int(gt_points.shape[0]),
        "visible_gt_point_count": int(np.count_nonzero(visible)),
        "visible_fraction": float(np.mean(visible)),
        "visible_observation_count_stats": _residual_stats(counts[counts > 0].astype(np.float64)),
    }
    return visible, info


def _evaluate_variant(
    *,
    variant_key: str,
    variant: dict[str, Any],
    manifest: Any,
    gt_points: np.ndarray,
    gt_tree: cKDTree,
    scannet_pose_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    da3_root = Path(variant["root"])
    if not (da3_root / "camera_poses.txt").is_file():
        raise FileNotFoundError(da3_root / "camera_poses.txt")
    poses_da3 = np.loadtxt(da3_root / "camera_poses.txt").reshape(-1, 4, 4)
    dense_raw, dense_colors, dense_frame_ids = _load_da3_dense_points(
        da3_root=da3_root,
        manifest=manifest,
        poses_da3=poses_da3,
        step=int(args.da3_dense_step),
        conf_min=float(args.da3_conf_min),
    )
    trajectory = _fit_trajectory_sim3(manifest=manifest, poses_da3=poses_da3, scannet_pose_root=scannet_pose_root)
    pose_orientation = _fit_pose_orientation_sim3(trajectory)
    surface = _refine_surface_sim3(
        source_points=dense_raw,
        target_points=gt_points,
        initial=pose_orientation,
        sample_count=int(args.surface_fit_sample_count),
        iterations=int(args.surface_fit_iterations),
        keep_ratio=float(args.surface_fit_keep_ratio),
        seed=int(args.seed) + _stable_variant_offset(variant_key),
    )
    surface["axis_map"] = np.asarray(pose_orientation["axis_map"], dtype=np.float64)

    dense_pose = _apply_sim3(dense_raw, pose_orientation)
    dense_surface = _apply_sim3(dense_raw, surface)
    thresholds = [float(value) for value in args.fscore_thresholds]
    pose_metrics = _chamfer_metrics(
        source_aligned=dense_pose,
        target_gt=gt_points,
        target_gt_tree=gt_tree,
        thresholds=thresholds,
    )
    surface_metrics = _chamfer_metrics(
        source_aligned=dense_surface,
        target_gt=gt_points,
        target_gt_tree=gt_tree,
        thresholds=thresholds,
    )
    viewer_idx = _sample_indices(
        dense_raw.shape[0],
        int(args.viewer_da3_sample_count),
        int(args.seed) + 101 + _stable_variant_offset(variant_key),
    )
    return {
        "variant_key": variant_key,
        "display_name": variant["display_name"],
        "model": variant["model"],
        "repo_id": variant["repo_id"],
        "output_root": str(da3_root),
        "config_path": str(da3_root / "da3_streaming_d4rt32o3_config.yaml"),
        "run_log": _parse_da3_log(Path(variant["log"])),
        "dense_reconstruction": {
            "point_count": int(dense_raw.shape[0]),
            "frame_id_min": int(np.min(dense_frame_ids)) if dense_frame_ids.size else None,
            "frame_id_max": int(np.max(dense_frame_ids)) if dense_frame_ids.size else None,
            "dense_step_px": int(args.da3_dense_step),
            "conf_min": float(args.da3_conf_min),
        },
        "trajectory_sim3": {
            "scale": float(trajectory["scale"]),
            "rotation_det": float(trajectory["rotation_det"]),
            "camera_residual_m": _residual_stats(trajectory["camera_residual"]),
            "camera_rotation_residual_degrees": _residual_stats(_camera_rotation_residual_degrees(trajectory, trajectory)),
        },
        "pose_orientation_sim3": {
            "scale": float(pose_orientation["scale"]),
            "rotation_det": float(pose_orientation["rotation_det"]),
            "axis_map_id": int(pose_orientation["axis_map_id"]),
            "candidate_count": int(pose_orientation["candidate_count"]),
            "camera_residual_m": _residual_stats(pose_orientation["camera_residual"]),
            "camera_rotation_residual_degrees": _residual_stats(pose_orientation["camera_rotation_residual_degrees"]),
            "geometry_metrics": pose_metrics,
        },
        "surface_refined_sim3": {
            "scale": float(surface["scale"]),
            "rotation_det": float(surface["rotation_det"]),
            "fit_sample_count": int(surface["sample_count"]),
            "fit_iterations": int(args.surface_fit_iterations),
            "fit_keep_ratio": float(args.surface_fit_keep_ratio),
            "fit_history": surface["history"],
            "camera_residual_m": _camera_residual_for_transform(surface, trajectory),
            "camera_rotation_residual_degrees": _residual_stats(_camera_rotation_residual_degrees(surface, trajectory)),
            "geometry_metrics": surface_metrics,
        },
        "viewer_payload": {
            "pose_points": dense_pose[viewer_idx].astype(np.float32),
            "surface_points": dense_surface[viewer_idx].astype(np.float32),
            "colors": dense_colors[viewer_idx].astype(np.uint8),
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gt_ply = Path(args.gt_ply) if args.gt_ply else ROOT / "Stream3D" / "data" / "scannet" / "processed" / args.scene_id / f"{args.scene_id}_vh_clean_2.ply"
    gt_points_full, gt_colors_full = _read_gt_point_cloud(gt_ply)
    manifest = _load_da3_manifest(Path(args.da3_manifest))
    scene_root = ROOT / "Stream3D" / "data" / "scannet" / "processed" / args.scene_id
    gt_visibility_filter: dict[str, Any] = {"mode": "full_gt_no_input_visibility_filter"}
    if args.gt_filter == "input_visible":
        visible_mask, gt_visibility_filter = _filter_gt_to_input_visible(
            gt_points=gt_points_full,
            manifest=manifest,
            scene_root=scene_root,
            depth_scale=float(args.scannet_depth_scale),
            depth_abs_tolerance=float(args.gt_visible_depth_abs_tolerance),
            depth_rel_tolerance=float(args.gt_visible_depth_rel_tolerance),
            min_observations=int(args.gt_visible_min_observations),
            batch_size=int(args.gt_visible_batch_size),
        )
        if int(np.count_nonzero(visible_mask)) < 4:
            raise RuntimeError("input-visible GT filter left fewer than 4 GT points")
        gt_points = gt_points_full[visible_mask]
        gt_colors = gt_colors_full[visible_mask]
    else:
        gt_points = gt_points_full
        gt_colors = gt_colors_full
    gt_tree = cKDTree(gt_points)
    gt_viewer_idx = _sample_indices(gt_points.shape[0], int(args.viewer_gt_sample_count), int(args.seed) + 31)
    scannet_pose_root = scene_root / "pose"

    variant_keys = list(args.variants)
    variant_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    npz_payload: dict[str, Any] = {
        "gt_points": gt_points[gt_viewer_idx].astype(np.float32),
        "gt_colors": gt_colors[gt_viewer_idx].astype(np.uint8),
    }
    for variant_key in variant_keys:
        variant = VARIANT_SPECS[variant_key]
        row = _evaluate_variant(
            variant_key=variant_key,
            variant=variant,
            manifest=manifest,
            gt_points=gt_points,
            gt_tree=gt_tree,
            scannet_pose_root=scannet_pose_root,
            args=args,
        )
        payload = row.pop("viewer_payload")
        npz_payload[f"{variant_key}_pose_points"] = payload["pose_points"]
        npz_payload[f"{variant_key}_surface_points"] = payload["surface_points"]
        npz_payload[f"{variant_key}_colors"] = payload["colors"]
        variant_rows.append(row)
        csv_rows.append(_metric_row(variant_key, variant, "pose_orientation_sim3", row["pose_orientation_sim3"]["geometry_metrics"]))
        csv_rows.append(_metric_row(variant_key, variant, "surface_refined_sim3", row["surface_refined_sim3"]["geometry_metrics"]))

    npz_path = output_root / f"{args.scene_id}_da3_variant_geometry_viewer_points.npz"
    np.savez_compressed(npz_path, **npz_payload)
    csv_path = output_root / "geometry_quality_metrics.csv"
    _write_csv(csv_path, csv_rows)

    summary = {
        "scene_id": args.scene_id,
        "diagnostic_only": True,
        "method_result_allowed": False,
        "metric_note": (
            "All DA3 variants were run with official DA3-Streaming on the same stride=5 input, chunk_size=32, overlap=3, "
            "loop disabled, dense Sim3 overlap stitch enabled by the DA3-Streaming align step. "
            "pose_orientation_sim3 uses ScanNet camera poses for diagnostic coordinate alignment. "
            "surface_refined_sim3 additionally refines to GT mesh nearest neighbors and is therefore an optimistic visual/geometry diagnostic, not method prediction evidence."
        ),
        "input": {
            "input_dir": str(args.input_dir),
            "manifest": str(args.da3_manifest),
            "manifest_frame_count": int(manifest.shape[0]),
            "frame_id_min": int(manifest["frame_id"].min()),
            "frame_id_max": int(manifest["frame_id"].max()),
            "stride_frame_id": int(manifest["frame_id"].iloc[1] - manifest["frame_id"].iloc[0]) if manifest.shape[0] > 1 else None,
            "chunk_size": 32,
            "overlap": 3,
        },
        "gt": {
            "gt_ply": str(gt_ply),
            "gt_filter": str(args.gt_filter),
            "full_gt_point_count": int(gt_points_full.shape[0]),
            "eval_gt_point_count": int(gt_points.shape[0]),
            "viewer_gt_point_count": int(gt_viewer_idx.shape[0]),
            "full_gt_z_stats_m": _percentiles(gt_points_full[:, 2]),
            "eval_gt_z_stats_m": _percentiles(gt_points[:, 2]),
            "visibility_filter": gt_visibility_filter,
        },
        "outputs": {
            "summary_json": str(output_root / "geometry_quality_summary.json"),
            "metrics_csv": str(csv_path),
            "viewer_npz": str(npz_path),
        },
        "variants": variant_rows,
        "csv_rows": csv_rows,
    }
    summary_path = output_root / "geometry_quality_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps({"summary_json": str(summary_path), "metrics_csv": str(csv_path), "viewer_npz": str(npz_path)}, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute v98.1 DA3 variant geometry quality metrics against ScanNet GT.")
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--da3-manifest", default=str(DEFAULT_INPUT_DIR / "frame_manifest_rows.csv"))
    parser.add_argument("--gt-ply", default="")
    parser.add_argument("--gt-filter", choices=["full", "input_visible"], default="full")
    parser.add_argument("--scannet-depth-scale", type=float, default=1000.0)
    parser.add_argument("--gt-visible-depth-abs-tolerance", type=float, default=0.08)
    parser.add_argument("--gt-visible-depth-rel-tolerance", type=float, default=0.03)
    parser.add_argument("--gt-visible-min-observations", type=int, default=1)
    parser.add_argument("--gt-visible-batch-size", type=int, default=65536)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_SPECS.keys()), choices=list(VARIANT_SPECS.keys()))
    parser.add_argument("--seed", type=int, default=9801098)
    parser.add_argument("--da3-dense-step", type=int, default=8)
    parser.add_argument("--da3-conf-min", type=float, default=0.0)
    parser.add_argument("--surface-fit-sample-count", type=int, default=60000)
    parser.add_argument("--surface-fit-iterations", type=int, default=8)
    parser.add_argument("--surface-fit-keep-ratio", type=float, default=0.90)
    parser.add_argument("--viewer-gt-sample-count", type=int, default=180000)
    parser.add_argument("--viewer-da3-sample-count", type=int, default=120000)
    parser.add_argument("--fscore-thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.50])
    build(parser.parse_args())


if __name__ == "__main__":
    main()
