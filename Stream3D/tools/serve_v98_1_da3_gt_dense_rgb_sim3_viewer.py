#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import pandas as pd
import viser
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
STREAM3D_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STREAM3D_ROOT))

from stream4d_native.sim3 import apply_sim3_to_xyz, fit_sim3_umeyama  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _residual_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ["mean", "p50", "p75", "p90", "p95", "max"]}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p75": float(np.percentile(values, 75.0)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def _read_gt_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"failed to read GT point cloud: {path}")
    if colors.shape == points.shape and colors.size:
        colors_u8 = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        colors_u8 = np.full(points.shape, 185, dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors_u8[finite]


def _sample_indices(count: int, sample_count: int, seed: int) -> np.ndarray:
    if sample_count <= 0 or count <= sample_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(count, size=int(sample_count), replace=False)
    idx.sort()
    return idx.astype(np.int64)


def _load_phase5_surfels(path: Path, scene_id: str) -> np.ndarray:
    df = pd.read_csv(path)
    df = df[df["scene_id"] == scene_id].copy()
    if "surfel_valid" in df.columns:
        df = df[df["surfel_valid"].astype(str).str.lower().isin(["true", "1"])]
    return df[["xyz_x", "xyz_y", "xyz_z"]].to_numpy(dtype=np.float64)


def _load_phase3_smoke(path: Path, scene_id: str) -> np.ndarray:
    df = pd.read_csv(path)
    df = df[df["scene_id"] == scene_id].copy()
    if "stitch_valid" in df.columns:
        df = df[df["stitch_valid"].astype(str).str.lower().isin(["true", "1"])]
    return df[["xyz_stitched_x", "xyz_stitched_y", "xyz_stitched_z"]].to_numpy(dtype=np.float64)


def _load_da3_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"da3_frame_index", "frame_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.sort_values("da3_frame_index").reset_index(drop=True)


def _load_da3_dense_points(
    *,
    da3_root: Path,
    manifest: pd.DataFrame,
    poses_da3: np.ndarray,
    step: int,
    conf_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    frame_ids: list[np.ndarray] = []
    step = max(1, int(step))
    for row in manifest.itertuples(index=False):
        da3_idx = int(row.da3_frame_index)
        frame_id = int(row.frame_id)
        npz_path = da3_root / "results_output" / f"frame_{da3_idx}.npz"
        if da3_idx >= poses_da3.shape[0] or not npz_path.exists():
            continue
        with np.load(npz_path) as payload:
            image = np.asarray(payload["image"], dtype=np.uint8)
            depth = np.asarray(payload["depth"], dtype=np.float64)
            conf = np.asarray(payload["conf"], dtype=np.float64) if "conf" in payload.files else np.ones_like(depth)
            intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
        h, w = depth.shape
        yy, xx = np.mgrid[step // 2 : h : step, step // 2 : w : step]
        z = depth[yy, xx].reshape(-1)
        score = conf[yy, xx].reshape(-1)
        valid = np.isfinite(z) & (z > 0.0) & np.isfinite(score) & (score >= float(conf_min))
        if not np.any(valid):
            continue
        pix = np.stack(
            [
                xx.reshape(-1).astype(np.float64),
                yy.reshape(-1).astype(np.float64),
                np.ones(xx.size, dtype=np.float64),
            ],
            axis=0,
        )[:, valid]
        rays = np.linalg.inv(intrinsics) @ pix
        cam = rays.T * z[valid, None]
        hom = np.concatenate([cam, np.ones((cam.shape[0], 1), dtype=np.float64)], axis=1)
        world = (poses_da3[da3_idx] @ hom.T).T[:, :3]
        rgb = image[yy.reshape(-1)[valid], xx.reshape(-1)[valid]]
        finite = np.isfinite(world).all(axis=1)
        points.append(world[finite].astype(np.float32))
        colors.append(rgb[finite].astype(np.uint8))
        frame_ids.append(np.full(int(np.count_nonzero(finite)), frame_id, dtype=np.int32))
    if not points:
        raise RuntimeError("no dense DA3 points reconstructed")
    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0), np.concatenate(frame_ids, axis=0)


def _fit_trajectory_sim3(
    *,
    manifest: pd.DataFrame,
    poses_da3: np.ndarray,
    scannet_pose_root: Path,
) -> dict[str, Any]:
    source: list[np.ndarray] = []
    target: list[np.ndarray] = []
    source_rot: list[np.ndarray] = []
    target_rot: list[np.ndarray] = []
    frames: list[int] = []
    for row in manifest.itertuples(index=False):
        da3_idx = int(row.da3_frame_index)
        frame_id = int(row.frame_id)
        gt_pose_path = scannet_pose_root / f"{frame_id}.txt"
        if da3_idx >= poses_da3.shape[0] or not gt_pose_path.exists():
            continue
        gt_pose = np.loadtxt(gt_pose_path).reshape(4, 4)
        if not np.isfinite(gt_pose).all():
            continue
        source.append(poses_da3[da3_idx, :3, 3].astype(np.float64))
        target.append(gt_pose[:3, 3].astype(np.float64))
        source_rot.append(poses_da3[da3_idx, :3, :3].astype(np.float64))
        target_rot.append(gt_pose[:3, :3].astype(np.float64))
        frames.append(frame_id)
    if len(source) < 4:
        raise RuntimeError("not enough DA3/ScanNet camera-pose correspondences for trajectory Sim3")
    fit = fit_sim3_umeyama(np.stack(source), np.stack(target))
    transformed = apply_sim3_to_xyz(np.stack(source), transform=fit).astype(np.float64)
    cam_residual = np.linalg.norm(transformed - np.stack(target), axis=1)
    return {
        "scale": float(fit["scale"]),
        "rot": np.asarray(fit["rot"], dtype=np.float64),
        "trans": np.asarray(fit["trans"], dtype=np.float64),
        "rotation_det": float(fit["rotation_det"]),
        "source_camera_centers": np.stack(source),
        "target_camera_centers": np.stack(target),
        "source_camera_rotations": np.stack(source_rot),
        "target_camera_rotations": np.stack(target_rot),
        "frame_ids": frames,
        "camera_residual": cam_residual,
    }


def _proper_signed_permutation_mats() -> list[np.ndarray]:
    mats: list[np.ndarray] = []
    for perm in permutations(range(3)):
        perm_mat = np.zeros((3, 3), dtype=np.float64)
        for src_axis, dst_axis in enumerate(perm):
            perm_mat[dst_axis, src_axis] = 1.0
        for signs in product((-1.0, 1.0), repeat=3):
            q = perm_mat @ np.diag(np.asarray(signs, dtype=np.float64))
            if np.linalg.det(q) > 0.5:
                mats.append(q)
    return mats


def _fit_rotation_from_camera_axes(source_rot: np.ndarray, target_rot: np.ndarray, axis_map: np.ndarray) -> np.ndarray:
    src_axes: list[np.ndarray] = []
    tgt_axes: list[np.ndarray] = []
    for src_r, tgt_r in zip(source_rot, target_rot):
        mapped_src = np.asarray(src_r, dtype=np.float64) @ np.asarray(axis_map, dtype=np.float64)
        for axis in range(3):
            src_axes.append(mapped_src[:, axis])
            tgt_axes.append(np.asarray(tgt_r, dtype=np.float64)[:, axis])
    src = np.stack(src_axes)
    tgt = np.stack(tgt_axes)
    u, _s, vt = np.linalg.svd(src.T @ tgt)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    return rot.astype(np.float64)


def _fit_scale_trans_fixed_rotation(source: np.ndarray, target: np.ndarray, rot: np.ndarray) -> tuple[float, np.ndarray]:
    src_mean = np.mean(source, axis=0)
    tgt_mean = np.mean(target, axis=0)
    src_rot = (np.asarray(rot, dtype=np.float64) @ (source - src_mean).T).T
    tgt_centered = target - tgt_mean
    denom = float(np.sum(src_rot * src_rot))
    if denom <= 1e-12:
        raise ValueError("degenerate camera centers for fixed-rotation Sim3")
    scale = float(np.sum(src_rot * tgt_centered) / denom)
    trans = tgt_mean - scale * (np.asarray(rot, dtype=np.float64) @ src_mean)
    return scale, trans.astype(np.float64)


def _camera_rotation_residual_degrees(transform: dict[str, Any], trajectory: dict[str, Any]) -> np.ndarray:
    rot_global = np.asarray(transform["rot"], dtype=np.float64)
    axis_map = np.asarray(transform.get("axis_map", np.eye(3)), dtype=np.float64)
    source_rot = np.asarray(trajectory["source_camera_rotations"], dtype=np.float64)
    target_rot = np.asarray(trajectory["target_camera_rotations"], dtype=np.float64)
    angles: list[float] = []
    for src_r, tgt_r in zip(source_rot, target_rot):
        pred = rot_global @ (src_r @ axis_map)
        rerr = pred.T @ tgt_r
        cos_angle = float(np.clip((np.trace(rerr) - 1.0) * 0.5, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cos_angle))))
    return np.asarray(angles, dtype=np.float64)


def _fit_pose_orientation_sim3(trajectory: dict[str, Any]) -> dict[str, Any]:
    source = np.asarray(trajectory["source_camera_centers"], dtype=np.float64)
    target = np.asarray(trajectory["target_camera_centers"], dtype=np.float64)
    source_rot = np.asarray(trajectory["source_camera_rotations"], dtype=np.float64)
    target_rot = np.asarray(trajectory["target_camera_rotations"], dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    for axis_id, axis_map in enumerate(_proper_signed_permutation_mats()):
        rot = _fit_rotation_from_camera_axes(source_rot, target_rot, axis_map)
        scale, trans = _fit_scale_trans_fixed_rotation(source, target, rot)
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        pred = apply_sim3_to_xyz(source, scale=scale, rot=rot, trans=trans).astype(np.float64)
        camera_residual = np.linalg.norm(pred - target, axis=1)
        transform = {
            "scale": float(scale),
            "rot": rot,
            "trans": trans,
            "rotation_det": float(np.linalg.det(rot)),
            "axis_map": axis_map,
            "axis_map_id": int(axis_id),
        }
        angle_residual = _camera_rotation_residual_degrees(transform, trajectory)
        camera_stats = _residual_stats(camera_residual)
        angle_stats = _residual_stats(angle_residual)
        # 1 degree is scored as 1 cm here: enough to reject tilted camera
        # conventions without overwhelming camera-center fit quality.
        score = float(camera_stats["p90"] + 0.01 * angle_stats["p90"])
        candidates.append(
            {
                **transform,
                "camera_residual": camera_residual,
                "camera_rotation_residual_degrees": angle_residual,
                "camera_residual_stats": camera_stats,
                "camera_rotation_residual_degrees_stats": angle_stats,
                "score": score,
            }
        )
    if not candidates:
        raise RuntimeError("pose-orientation Sim3 fitting produced no valid positive-scale candidates")
    candidates.sort(key=lambda row: float(row["score"]))
    best = candidates[0]
    best["candidate_count"] = int(len(candidates))
    best["top_candidates"] = [
        {
            "axis_map_id": int(row["axis_map_id"]),
            "scale": float(row["scale"]),
            "score": float(row["score"]),
            "camera_residual_p90": float(row["camera_residual_stats"]["p90"]),
            "camera_rotation_residual_degrees_p90": float(row["camera_rotation_residual_degrees_stats"]["p90"]),
            "axis_map": np.asarray(row["axis_map"], dtype=np.float64),
        }
        for row in candidates[:5]
    ]
    return best


def _refine_surface_sim3(
    *,
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial: dict[str, Any],
    sample_count: int,
    iterations: int,
    keep_ratio: float,
    seed: int,
) -> dict[str, Any]:
    idx = _sample_indices(source_points.shape[0], int(sample_count), int(seed))
    src = np.asarray(source_points[idx], dtype=np.float64)
    tree = cKDTree(np.asarray(target_points, dtype=np.float64))
    scale = float(initial["scale"])
    rot = np.asarray(initial["rot"], dtype=np.float64)
    trans = np.asarray(initial["trans"], dtype=np.float64)
    history: list[dict[str, Any]] = []
    for iteration in range(int(iterations)):
        aligned = apply_sim3_to_xyz(src, scale=scale, rot=rot, trans=trans).astype(np.float64)
        dist, nn_idx = tree.query(aligned, k=1)
        finite = np.isfinite(dist)
        if int(np.count_nonzero(finite)) < 4:
            raise RuntimeError("surface Sim3 refinement lost finite nearest-neighbor pairs")
        threshold = float(np.percentile(dist[finite], 100.0 * float(keep_ratio)))
        keep = finite & (dist <= threshold)
        if int(np.count_nonzero(keep)) < 4:
            keep = finite
        fit = fit_sim3_umeyama(src[keep], target_points[nn_idx[keep]])
        scale = float(fit["scale"])
        rot = np.asarray(fit["rot"], dtype=np.float64)
        trans = np.asarray(fit["trans"], dtype=np.float64)
        history.append(
            {
                "iteration": int(iteration),
                "sample_count": int(src.shape[0]),
                "kept_pairs": int(np.count_nonzero(keep)),
                "keep_ratio": float(keep_ratio),
                "nn_distance_threshold": threshold,
                "pre_fit_nn_residual_mean": float(np.mean(dist[finite])),
                "pre_fit_nn_residual_p90": float(np.percentile(dist[finite], 90.0)),
                "anchor_residual_mean": float(np.mean(fit["residual"])),
                "anchor_residual_p90": float(np.percentile(fit["residual"], 90.0)),
            }
        )
    return {
        "scale": float(scale),
        "rot": rot,
        "trans": trans,
        "rotation_det": float(np.linalg.det(rot)),
        "history": history,
        "sample_count": int(src.shape[0]),
    }


def _apply_sim3(points: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    return apply_sim3_to_xyz(
        points,
        scale=float(transform["scale"]),
        rot=np.asarray(transform["rot"], dtype=np.float64),
        trans=np.asarray(transform["trans"], dtype=np.float64),
    ).astype(np.float32)


def _nearest_colors(source_points: np.ndarray, source_colors: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    if query_points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    _, idx = cKDTree(np.asarray(source_points, dtype=np.float64)).query(np.asarray(query_points, dtype=np.float64), k=1)
    return np.asarray(source_colors[idx], dtype=np.uint8)


def _nn_residual(points: np.ndarray, target_tree: cKDTree) -> dict[str, float]:
    dist, _ = target_tree.query(np.asarray(points, dtype=np.float64), k=1)
    return _residual_stats(dist)


def _camera_residual_for_transform(transform: dict[str, Any], traj: dict[str, Any]) -> dict[str, float]:
    aligned = _apply_sim3(np.asarray(traj["source_camera_centers"], dtype=np.float64), transform).astype(np.float64)
    residual = np.linalg.norm(aligned - np.asarray(traj["target_camera_centers"], dtype=np.float64), axis=1)
    return _residual_stats(residual)


def _prepare_payload(args: argparse.Namespace) -> dict[str, Any]:
    scene_id = args.scene_id
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    gt_ply = Path(args.gt_ply) if args.gt_ply else ROOT / "Stream3D" / "data" / "scannet" / "processed" / scene_id / f"{scene_id}_vh_clean_2.ply"
    gt_points, gt_colors = _read_gt_point_cloud(gt_ply)
    gt_idx = _sample_indices(gt_points.shape[0], int(args.viewer_gt_sample_count), int(args.seed) + 31)
    gt_viewer = gt_points[gt_idx]
    gt_viewer_colors = gt_colors[gt_idx]
    gt_tree = cKDTree(gt_points)

    da3_root = Path(args.da3_root)
    manifest = _load_da3_manifest(Path(args.da3_manifest))
    poses_da3 = np.loadtxt(da3_root / "camera_poses.txt").reshape(-1, 4, 4)
    dense_raw, dense_colors, dense_frame_ids = _load_da3_dense_points(
        da3_root=da3_root,
        manifest=manifest,
        poses_da3=poses_da3,
        step=int(args.da3_dense_step),
        conf_min=float(args.da3_conf_min),
    )

    trajectory = _fit_trajectory_sim3(
        manifest=manifest,
        poses_da3=poses_da3,
        scannet_pose_root=ROOT / "Stream3D" / "data" / "scannet" / "processed" / scene_id / "pose",
    )
    pose_orientation = _fit_pose_orientation_sim3(trajectory)
    surface = _refine_surface_sim3(
        source_points=dense_raw,
        target_points=gt_points,
        initial=pose_orientation,
        sample_count=int(args.surface_fit_sample_count),
        iterations=int(args.surface_fit_iterations),
        keep_ratio=float(args.surface_fit_keep_ratio),
        seed=int(args.seed) + 71,
    )
    surface["axis_map"] = np.asarray(pose_orientation["axis_map"], dtype=np.float64)

    dense_pose = _apply_sim3(dense_raw, pose_orientation)
    dense_surface = _apply_sim3(dense_raw, surface)
    dense_trajectory_idx = _sample_indices(dense_raw.shape[0], int(args.trajectory_viewer_sample_count), int(args.seed) + 91)
    dense_trajectory = _apply_sim3(dense_raw[dense_trajectory_idx], trajectory)
    dense_trajectory_colors = dense_colors[dense_trajectory_idx]

    phase5_raw = _load_phase5_surfels(Path(args.phase5_csv), scene_id)
    phase3_raw = _load_phase3_smoke(Path(args.phase3_csv), scene_id)
    phase5_pose = _apply_sim3(phase5_raw, pose_orientation)
    phase5_surface = _apply_sim3(phase5_raw, surface)
    phase5_trajectory = _apply_sim3(phase5_raw, trajectory)
    phase3_pose = _apply_sim3(phase3_raw, pose_orientation)
    phase3_surface = _apply_sim3(phase3_raw, surface)
    phase5_colors = _nearest_colors(dense_raw, dense_colors, phase5_raw)
    phase3_colors = np.tile(np.asarray([[38, 107, 220]], dtype=np.uint8), (phase3_surface.shape[0], 1))

    npz_path = output_root / f"{scene_id}_v98_da3_dense_rgb_sim3_viewer_points.npz"
    np.savez_compressed(
        npz_path,
        gt_points=gt_viewer.astype(np.float32),
        gt_colors=gt_viewer_colors.astype(np.uint8),
        da3_dense_pose_orientation_points=dense_pose.astype(np.float32),
        da3_dense_pose_orientation_colors=dense_colors.astype(np.uint8),
        da3_dense_surface_points=dense_surface.astype(np.float32),
        da3_dense_surface_colors=dense_colors.astype(np.uint8),
        da3_dense_trajectory_points=dense_trajectory.astype(np.float32),
        da3_dense_trajectory_colors=dense_trajectory_colors.astype(np.uint8),
        v98_phase5_pose_orientation_points=phase5_pose.astype(np.float32),
        v98_phase5_surface_points=phase5_surface.astype(np.float32),
        v98_phase5_trajectory_points=phase5_trajectory.astype(np.float32),
        v98_phase5_colors=phase5_colors.astype(np.uint8),
        v98_phase3_pose_orientation_points=phase3_pose.astype(np.float32),
        v98_phase3_surface_points=phase3_surface.astype(np.float32),
        v98_phase3_colors=phase3_colors.astype(np.uint8),
        da3_dense_raw_points=dense_raw.astype(np.float32),
        da3_dense_raw_colors=dense_colors.astype(np.uint8),
        da3_dense_frame_ids=dense_frame_ids.astype(np.int32),
    )

    summary = {
        "viewer": "v98_1_da3_gt_dense_rgb_sim3_viewer",
        "scene_id": scene_id,
        "diagnostic_only": True,
        "method_result_allowed": False,
        "default_visible_alignment": "pose_initialized_surface_refined_sim3",
        "alignment_warning": (
            "pose_orientation_sim3 is fitted from same-frame DA3 and ScanNet camera centers plus camera orientations. "
            "surface_refined_sim3 is initialized from pose_orientation_sim3, then refined to GT mesh nearest neighbors for visualization. "
            "center_only_trajectory_sim3 is retained as a hidden diagnostic; it fits centers but can leave large roll/tilt error. "
            "Neither transform is method prediction evidence or AP evidence."
        ),
        "source_files": {
            "gt_mesh": str(gt_ply),
            "da3_root": str(da3_root),
            "da3_manifest": str(args.da3_manifest),
            "phase5_fused_surfel_csv": str(args.phase5_csv),
            "phase3_stitched_smoke_csv": str(args.phase3_csv),
        },
        "output_files": {
            "viewer_npz": str(npz_path),
            "summary_json": str(output_root / "summary.json"),
        },
        "counts": {
            "gt_mesh_point_count": int(gt_points.shape[0]),
            "gt_viewer_point_count": int(gt_viewer.shape[0]),
            "da3_dense_rgb_point_count": int(dense_raw.shape[0]),
            "da3_dense_step_px": int(args.da3_dense_step),
            "da3_manifest_frame_count": int(manifest.shape[0]),
            "trajectory_viewer_point_count": int(dense_trajectory.shape[0]),
            "v98_phase5_fused_surfel_count": int(phase5_raw.shape[0]),
            "v98_phase3_stitched_da3_smoke_count": int(phase3_raw.shape[0]),
        },
        "trajectory_sim3": {
            "scale": float(trajectory["scale"]),
            "rotation_det": float(trajectory["rotation_det"]),
            "translation": np.asarray(trajectory["trans"], dtype=np.float64),
            "rotation": np.asarray(trajectory["rot"], dtype=np.float64),
            "camera_pair_count": int(len(trajectory["frame_ids"])),
            "camera_frame_id_min": int(min(trajectory["frame_ids"])),
            "camera_frame_id_max": int(max(trajectory["frame_ids"])),
            "camera_residual": _residual_stats(trajectory["camera_residual"]),
            "camera_rotation_residual_degrees": _residual_stats(_camera_rotation_residual_degrees(trajectory, trajectory)),
            "dense_to_full_gt_nn_residual": _nn_residual(_apply_sim3(dense_raw, trajectory), gt_tree),
            "phase5_to_full_gt_nn_residual": _nn_residual(phase5_trajectory, gt_tree),
        },
        "pose_orientation_sim3": {
            "scale": float(pose_orientation["scale"]),
            "rotation_det": float(pose_orientation["rotation_det"]),
            "translation": np.asarray(pose_orientation["trans"], dtype=np.float64),
            "rotation": np.asarray(pose_orientation["rot"], dtype=np.float64),
            "axis_map": np.asarray(pose_orientation["axis_map"], dtype=np.float64),
            "axis_map_id": int(pose_orientation["axis_map_id"]),
            "candidate_count": int(pose_orientation["candidate_count"]),
            "top_candidates": pose_orientation["top_candidates"],
            "camera_residual": _residual_stats(pose_orientation["camera_residual"]),
            "camera_rotation_residual_degrees": _residual_stats(pose_orientation["camera_rotation_residual_degrees"]),
            "dense_to_full_gt_nn_residual": _nn_residual(dense_pose, gt_tree),
            "phase5_to_full_gt_nn_residual": _nn_residual(phase5_pose, gt_tree),
            "phase3_smoke_to_full_gt_nn_residual": _nn_residual(phase3_pose, gt_tree),
        },
        "surface_refined_sim3": {
            "scale": float(surface["scale"]),
            "rotation_det": float(surface["rotation_det"]),
            "translation": np.asarray(surface["trans"], dtype=np.float64),
            "rotation": np.asarray(surface["rot"], dtype=np.float64),
            "fit_sample_count": int(surface["sample_count"]),
            "fit_keep_ratio": float(args.surface_fit_keep_ratio),
            "fit_iterations": int(args.surface_fit_iterations),
            "fit_history": surface["history"],
            "camera_residual": _camera_residual_for_transform(surface, trajectory),
            "camera_rotation_residual_degrees": _residual_stats(_camera_rotation_residual_degrees(surface, trajectory)),
            "dense_to_full_gt_nn_residual": _nn_residual(dense_surface, gt_tree),
            "phase5_to_full_gt_nn_residual": _nn_residual(phase5_surface, gt_tree),
            "phase3_smoke_to_full_gt_nn_residual": _nn_residual(phase3_surface, gt_tree),
        },
        "color_provenance": {
            "gt_colors": "ScanNet mesh vertex colors",
            "da3_dense_colors": "official DA3 frame npz image RGB at sampled depth pixels",
            "phase5_colors": "nearest raw dense DA3 RGB point in DA3 coordinates",
            "phase3_smoke_colors": "uniform blue diagnostic because Phase3 smoke rows have no RGB fields",
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "viewer_npz": npz_path,
        "gt_points": gt_viewer.astype(np.float32),
        "gt_colors": gt_viewer_colors.astype(np.uint8),
        "dense_pose_points": dense_pose.astype(np.float32),
        "dense_pose_colors": dense_colors.astype(np.uint8),
        "dense_surface_points": dense_surface.astype(np.float32),
        "dense_surface_colors": dense_colors.astype(np.uint8),
        "dense_trajectory_points": dense_trajectory.astype(np.float32),
        "dense_trajectory_colors": dense_trajectory_colors.astype(np.uint8),
        "phase5_pose_points": phase5_pose.astype(np.float32),
        "phase5_surface_points": phase5_surface.astype(np.float32),
        "phase5_colors": phase5_colors.astype(np.uint8),
        "phase3_pose_points": phase3_pose.astype(np.float32),
        "phase3_surface_points": phase3_surface.astype(np.float32),
        "phase3_colors": phase3_colors.astype(np.uint8),
    }


def _add_toggle(server: viser.ViserServer, label: str, handle: Any, visible: bool) -> None:
    toggle = server.gui.add_checkbox(label, visible)

    @toggle.on_update
    def _(_: Any) -> None:
        handle.visible = bool(toggle.value)


def serve(args: argparse.Namespace) -> dict[str, Any]:
    payload = _prepare_payload(args)
    server = viser.ViserServer(host=args.host, port=args.port, verbose=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid",
        width=float(args.grid_width),
        height=float(args.grid_width),
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
        position=(0.0, 0.0, -0.02),
    )
    gt = server.scene.add_point_cloud(
        "/GT ScanNet mesh RGB",
        points=payload["gt_points"],
        colors=payload["gt_colors"],
        point_size=float(args.gt_point_size),
        point_shape="circle",
        visible=True,
    )
    dense_pose = server.scene.add_point_cloud(
        "/DA3 dense RGB pose-orientation Sim3",
        points=payload["dense_pose_points"],
        colors=payload["dense_pose_colors"],
        point_size=float(args.da3_point_size),
        point_shape="circle",
        visible=False,
    )
    dense_surface = server.scene.add_point_cloud(
        "/DA3 dense RGB pose-initialized surface-refined Sim3",
        points=payload["dense_surface_points"],
        colors=payload["dense_surface_colors"],
        point_size=float(args.da3_point_size),
        point_shape="circle",
        visible=True,
    )
    dense_trajectory = server.scene.add_point_cloud(
        "/DA3 dense RGB trajectory Sim3 sample",
        points=payload["dense_trajectory_points"],
        colors=payload["dense_trajectory_colors"],
        point_size=float(args.da3_point_size),
        point_shape="circle",
        visible=False,
    )
    phase5 = server.scene.add_point_cloud(
        "/v98 Phase5 method surfels RGB pose-orientation",
        points=payload["phase5_pose_points"],
        colors=payload["phase5_colors"],
        point_size=float(args.phase5_point_size),
        point_shape="circle",
        visible=False,
    )
    phase3 = server.scene.add_point_cloud(
        "/v98 Phase3 smoke points pose-orientation",
        points=payload["phase3_pose_points"],
        colors=payload["phase3_colors"],
        point_size=float(args.phase3_point_size),
        point_shape="circle",
        visible=False,
    )

    _add_toggle(server, "GT ScanNet mesh RGB", gt, True)
    _add_toggle(server, "DA3 dense RGB pose-init surface Sim3", dense_surface, True)
    _add_toggle(server, "DA3 dense RGB pose-orientation Sim3", dense_pose, False)
    _add_toggle(server, "DA3 dense RGB trajectory Sim3 sample", dense_trajectory, False)
    _add_toggle(server, "v98 Phase5 method surfels RGB pose", phase5, False)
    _add_toggle(server, "v98 Phase3 smoke points", phase3, False)

    status = {
        "viewer": "v98_1_da3_gt_dense_rgb_sim3_viewer",
        "pid": int(os.getpid()),
        "host": args.host,
        "port": int(args.port),
        "url": f"http://localhost:{args.port}",
        "scene_id": args.scene_id,
        "viewer_npz": str(payload["viewer_npz"]),
        "summary_json": str(payload["summary_path"]),
        "diagnostic_only": True,
        "layers": [
            "GT ScanNet mesh RGB",
            "DA3 dense RGB pose-initialized surface-refined Sim3",
            "DA3 dense RGB pose-orientation Sim3",
            "DA3 dense RGB trajectory Sim3 sample",
            "v98 Phase5 method surfels RGB pose-orientation",
            "v98 Phase3 smoke points pose-orientation",
        ],
        "counts": payload["summary"]["counts"],
        "trajectory_sim3": {
            "scale": payload["summary"]["trajectory_sim3"]["scale"],
            "camera_residual": payload["summary"]["trajectory_sim3"]["camera_residual"],
            "camera_rotation_residual_degrees": payload["summary"]["trajectory_sim3"]["camera_rotation_residual_degrees"],
            "dense_to_full_gt_nn_residual": payload["summary"]["trajectory_sim3"]["dense_to_full_gt_nn_residual"],
        },
        "pose_orientation_sim3": {
            "scale": payload["summary"]["pose_orientation_sim3"]["scale"],
            "camera_residual": payload["summary"]["pose_orientation_sim3"]["camera_residual"],
            "camera_rotation_residual_degrees": payload["summary"]["pose_orientation_sim3"]["camera_rotation_residual_degrees"],
            "dense_to_full_gt_nn_residual": payload["summary"]["pose_orientation_sim3"]["dense_to_full_gt_nn_residual"],
        },
        "surface_refined_sim3": {
            "scale": payload["summary"]["surface_refined_sim3"]["scale"],
            "camera_residual": payload["summary"]["surface_refined_sim3"]["camera_residual"],
            "camera_rotation_residual_degrees": payload["summary"]["surface_refined_sim3"]["camera_rotation_residual_degrees"],
            "dense_to_full_gt_nn_residual": payload["summary"]["surface_refined_sim3"]["dense_to_full_gt_nn_residual"],
        },
    }
    if args.status_json:
        status_path = Path(args.status_json)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True, default=_json_default), flush=True)

    stop = {"value": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    if float(args.smoke_seconds) > 0:
        deadline = time.time() + float(args.smoke_seconds)
        while time.time() < deadline:
            time.sleep(0.25)
        server.stop()
        return status

    while not stop["value"]:
        time.sleep(1.0)
    server.stop()
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve dense RGB DA3 vs GT geometry in Viser with diagnostic Sim3 alignments.")
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--da3-root", default=str(ROOT / "Stream3D" / "outputs" / "audit" / "v98_phase1_provider_contract" / "da3_streaming_full_scene0050"))
    parser.add_argument("--da3-manifest", default=str(ROOT / "Stream3D" / "outputs" / "audit" / "v98_phase1_provider_contract" / "da3_streaming_full_scene0050_input" / "frame_manifest_rows.csv"))
    parser.add_argument("--phase5-csv", default=str(ROOT / "Stream3D" / "outputs" / "audit" / "v98_phase5_fused_surfel" / "fused_surfel_rows.csv"))
    parser.add_argument("--phase3-csv", default=str(ROOT / "Stream3D" / "outputs" / "audit" / "v98_phase3_da3_stitch" / "stitched_da3_point_rows.csv"))
    parser.add_argument("--gt-ply", default="")
    parser.add_argument("--output-root", default=str(ROOT / "Stream3D" / "outputs" / "audit" / "v98_1_da3_gt_dense_rgb_sim3_viewer_scene0050"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--status-json", default="")
    parser.add_argument("--seed", type=int, default=9801050)
    parser.add_argument("--viewer-gt-sample-count", type=int, default=0)
    parser.add_argument("--da3-dense-step", type=int, default=8)
    parser.add_argument("--da3-conf-min", type=float, default=0.0)
    parser.add_argument("--surface-fit-sample-count", type=int, default=60000)
    parser.add_argument("--surface-fit-iterations", type=int, default=8)
    parser.add_argument("--surface-fit-keep-ratio", type=float, default=0.90)
    parser.add_argument("--trajectory-viewer-sample-count", type=int, default=60000)
    parser.add_argument("--gt-point-size", type=float, default=0.008)
    parser.add_argument("--da3-point-size", type=float, default=0.01)
    parser.add_argument("--phase5-point-size", type=float, default=0.026)
    parser.add_argument("--phase3-point-size", type=float, default=0.055)
    parser.add_argument("--grid-width", type=float, default=8.0)
    parser.add_argument("--smoke-seconds", type=float, default=0.0)
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
