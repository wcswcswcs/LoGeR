#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    out = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    out[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    out[:, 0, 1] = 2.0 * (xy - wz)
    out[:, 0, 2] = 2.0 * (xz + wy)
    out[:, 1, 0] = 2.0 * (xy + wz)
    out[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    out[:, 1, 2] = 2.0 * (yz - wx)
    out[:, 2, 0] = 2.0 * (xz - wy)
    out[:, 2, 1] = 2.0 * (yz + wx)
    out[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return out


def _load_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) < 8:
            raise ValueError(f"Trajectory row must have >=8 columns: {line}")
        rows.append(vals[:8])
    if not rows:
        raise ValueError(f"No trajectory rows in {path}")
    arr = np.asarray(rows, dtype=np.float64)
    frames = np.rint(arr[:, 0]).astype(np.int64)
    poses = np.tile(np.eye(4, dtype=np.float64), (arr.shape[0], 1, 1))
    poses[:, :3, :3] = _quat_xyzw_to_mat(arr[:, 4:8])
    poses[:, :3, 3] = arr[:, 1:4]
    return frames, poses, poses[:, :3, 3].copy()


def _load_gt_poses(scene_root: Path, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    kept_frames: list[int] = []
    skipped_frames: list[dict[str, Any]] = []
    poses: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    for frame in frames.tolist():
        pose_path = scene_root / "pose" / f"{int(frame)}.txt"
        if not pose_path.exists():
            skipped_frames.append({"frame_id": int(frame), "reason": "missing_pose_file", "path": str(pose_path)})
            continue
        pose = np.loadtxt(pose_path).astype(np.float64)
        if pose.shape != (4, 4):
            skipped_frames.append({"frame_id": int(frame), "reason": f"bad_pose_shape_{pose.shape}", "path": str(pose_path)})
            continue
        if not np.isfinite(pose).all():
            skipped_frames.append({"frame_id": int(frame), "reason": "nonfinite_pose_values", "path": str(pose_path)})
            continue
        kept_frames.append(int(frame))
        poses.append(pose)
        centers.append(pose[:3, 3])
    if not poses:
        raise ValueError(f"No valid ScanNet GT poses matched frames from {scene_root}")
    return (
        np.asarray(kept_frames, dtype=np.int64),
        np.stack(poses, axis=0),
        np.stack(centers, axis=0),
        skipped_frames,
    )


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError(f"Need matched Nx3 arrays with N>=3, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    mx = src.mean(axis=0)
    my = dst.mean(axis=0)
    x = src - mx
    y = dst - my
    cov = (y.T @ x) / n
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0.0:
        d[-1, -1] = -1.0
    rot = u @ d @ vt
    var_x = float((x * x).sum() / n)
    scale = float(np.trace(np.diag(s) @ d) / max(var_x, 1e-12))
    trans = my - scale * (rot @ mx)
    return scale, rot, trans


def _apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (scale * (rot @ points.T)).T + trans[None]


def _sample_indices(num_items: int, max_items: int, seed: int) -> np.ndarray:
    if num_items <= max_items:
        return np.arange(num_items, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_items, size=max_items, replace=False).astype(np.int64))


def _stats(dist: np.ndarray) -> dict[str, float]:
    dist = np.asarray(dist, dtype=np.float64)
    return {
        "mean_m": float(np.mean(dist)),
        "median_m": float(np.median(dist)),
        "p90_m": float(np.percentile(dist, 90)),
        "p95_m": float(np.percentile(dist, 95)),
        "max_m": float(np.max(dist)),
    }


def _fscore(pred_to_gt: np.ndarray, gt_to_pred: np.ndarray, threshold: float) -> dict[str, float]:
    precision = float(np.mean(pred_to_gt <= threshold))
    recall = float(np.mean(gt_to_pred <= threshold))
    fscore = 0.0 if precision + recall <= 0.0 else float(2.0 * precision * recall / (precision + recall))
    return {
        f"precision@{threshold:.2f}m": precision,
        f"recall@{threshold:.2f}m": recall,
        f"fscore@{threshold:.2f}m": fscore,
    }


def _load_mesh_points(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(mesh_path))
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"Open3D read no points from {mesh_path}")
    if colors.shape != points.shape:
        colors = np.full(points.shape, 0.68, dtype=np.float32)
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return points, colors_u8


def _load_loger_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload)}")
    return payload


def _world_and_conf(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
    world = payload.get("world_points", payload.get("points"))
    if world is None:
        raise KeyError("LoGeR payload has neither world_points nor points")
    if not torch.is_tensor(world):
        world = torch.as_tensor(world)
    conf = payload.get("confidence", payload.get("conf"))
    if conf is not None and not torch.is_tensor(conf):
        conf = torch.as_tensor(conf)
    if conf is not None and conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    return world.float(), conf.float() if conf is not None else None


def _sample_loger_points(
    world: torch.Tensor,
    conf: torch.Tensor | None,
    *,
    conf_min: float | None,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if world.ndim != 4 or world.shape[-1] != 3:
        raise ValueError(f"Expected world points [T,H,W,3], got {tuple(world.shape)}")
    valid = torch.isfinite(world).all(dim=-1)
    conf_present = conf is not None
    if conf is not None:
        valid = valid & torch.isfinite(conf)
        if conf_min is not None:
            valid = valid & (conf >= float(conf_min))
    flat_valid = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    valid_count = int(flat_valid.numel())
    if valid_count == 0:
        raise ValueError("No valid LoGeR points after confidence/finite filtering")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if valid_count > max_points:
        perm = torch.randperm(valid_count, generator=generator)[: int(max_points)]
        chosen = flat_valid[perm]
    else:
        chosen = flat_valid
    chosen, _ = torch.sort(chosen)
    flat_world = world.reshape(-1, 3)
    points = flat_world[chosen].numpy().astype(np.float64)
    meta = {
        "world_shape": list(world.shape),
        "confidence_present": bool(conf_present),
        "confidence_min": None if conf_min is None else float(conf_min),
        "valid_point_count_after_filter": valid_count,
        "sampled_point_count": int(points.shape[0]),
        "sample_seed": int(seed),
    }
    return points, chosen.numpy().astype(np.int64), meta


def _color_loger_samples(
    scene_root: Path,
    frames: np.ndarray,
    flat_indices: np.ndarray,
    shape: tuple[int, int, int, int],
) -> np.ndarray:
    t_count, height, width, _ = shape
    plane = height * width
    t_idx = flat_indices // plane
    rem = flat_indices % plane
    y = rem // width
    x = rem % width
    colors = np.zeros((flat_indices.shape[0], 3), dtype=np.uint8)
    for t in np.unique(t_idx):
        if t < 0 or t >= t_count or t >= len(frames):
            continue
        frame = int(frames[int(t)])
        image_path = scene_root / "color" / f"{frame}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            colors[t_idx == t] = np.array([230, 90, 70], dtype=np.uint8)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = t_idx == t
        colors[mask] = image[y[mask], x[mask]]
    return colors


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_root = Path(args.scene_root)
    mesh_path = scene_root / f"{args.scene_id}_vh_clean_2.ply"

    frames, pred_poses, pred_centers = _load_tum(Path(args.trajectory))
    gt_frames, gt_poses, gt_centers, skipped_gt_frames = _load_gt_poses(scene_root, frames)
    frame_to_pred = {int(frame): i for i, frame in enumerate(frames.tolist())}
    pred_indices = np.asarray([frame_to_pred[int(frame)] for frame in gt_frames], dtype=np.int64)
    matched_pred_centers = pred_centers[pred_indices]

    scale, rot, trans = _umeyama_sim3(matched_pred_centers, gt_centers)
    aligned_pred_centers = _apply_sim3(matched_pred_centers, scale, rot, trans)
    ate_err = np.linalg.norm(aligned_pred_centers - gt_centers, axis=1)
    ate = {
        "matched_frame_count": int(gt_frames.shape[0]),
        "skipped_gt_frame_count": int(len(skipped_gt_frames)),
        "skipped_gt_frames": skipped_gt_frames,
        "aligned_ate_rmse_m": float(math.sqrt(float(np.mean(ate_err * ate_err)))),
        "aligned_ate_mean_m": float(np.mean(ate_err)),
        "aligned_ate_median_m": float(np.median(ate_err)),
        "aligned_ate_p90_m": float(np.percentile(ate_err, 90)),
        "aligned_ate_max_m": float(np.max(ate_err)),
        "sim3_scale": float(scale),
        "sim3_R": rot.tolist(),
        "sim3_t": trans.tolist(),
    }

    gt_points, gt_colors = _load_mesh_points(mesh_path)
    payload = _load_loger_payload(Path(args.geometry_pt))
    world, conf = _world_and_conf(payload)
    loger_points_raw, loger_flat_idx, loger_meta = _sample_loger_points(
        world,
        conf,
        conf_min=args.conf_min,
        max_points=args.max_pred_points,
        seed=args.seed,
    )
    loger_points = _apply_sim3(loger_points_raw, scale, rot, trans)

    gt_metric_idx = _sample_indices(gt_points.shape[0], args.max_gt_points, args.seed + 17)
    gt_metric_points = gt_points[gt_metric_idx]
    gt_tree = cKDTree(gt_metric_points)
    pred_tree = cKDTree(loger_points)
    pred_to_gt, _ = gt_tree.query(loger_points, workers=-1)
    gt_to_pred, _ = pred_tree.query(gt_metric_points, workers=-1)
    chamfer = {
        "definition": "chamfer_l1_m=0.5*(mean NN distance LoGeR->GT mesh + mean NN distance GT mesh->LoGeR); chamfer_l2_m2=0.5*(mean squared NN distance in both directions).",
        "gt_source": str(mesh_path),
        "gt_metric_point_count_total": int(gt_points.shape[0]),
        "gt_metric_point_count_sampled": int(gt_metric_points.shape[0]),
        "loger_metric": loger_meta,
        "loger_to_gt": _stats(pred_to_gt),
        "gt_to_loger": _stats(gt_to_pred),
        "chamfer_l1_m": float(0.5 * (np.mean(pred_to_gt) + np.mean(gt_to_pred))),
        "chamfer_l2_m2": float(0.5 * (np.mean(pred_to_gt * pred_to_gt) + np.mean(gt_to_pred * gt_to_pred))),
    }
    for threshold in (0.05, 0.10, 0.20):
        chamfer.update(_fscore(pred_to_gt, gt_to_pred, threshold))

    viewer_pred_idx = _sample_indices(loger_points.shape[0], args.max_viewer_pred_points, args.seed + 31)
    viewer_gt_idx = _sample_indices(gt_points.shape[0], args.max_viewer_gt_points, args.seed + 47)
    viewer_flat_idx = loger_flat_idx[viewer_pred_idx]
    loger_colors = _color_loger_samples(scene_root, frames, viewer_flat_idx, tuple(world.shape))
    viewer_npz = out_dir / "viewer_gt_loger_points.npz"
    np.savez_compressed(
        viewer_npz,
        gt_points=gt_points[viewer_gt_idx].astype(np.float32),
        gt_colors=gt_colors[viewer_gt_idx],
        loger_points=loger_points[viewer_pred_idx].astype(np.float32),
        loger_colors=loger_colors,
        gt_frames=gt_frames.astype(np.int64),
        gt_camera_centers=gt_centers.astype(np.float32),
        loger_camera_centers=aligned_pred_centers.astype(np.float32),
    )

    summary = {
        "scene_id": args.scene_id,
        "scene_root": str(scene_root),
        "input_frame_policy": (
            f"Numeric RGB frame ids from LoGeR trajectory; inferred frame step "
            f"{int(np.median(np.diff(frames))) if frames.shape[0] > 1 else 0}."
        ),
        "frame_id_count": int(frames.shape[0]),
        "frame_ids_first10": frames[:10].astype(int).tolist(),
        "frame_ids_last10": frames[-10:].astype(int).tolist(),
        "geometry_pt": str(args.geometry_pt),
        "trajectory": str(args.trajectory),
        "ate": ate,
        "chamfer": chamfer,
        "viewer_npz": str(viewer_npz),
    }
    summary_path = out_dir / f"{args.scene_id}_loger_geometry_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LoGeR geometry on one ScanNet scene.")
    parser.add_argument("--scene-id", default="scene0081_01")
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--geometry-pt", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--conf-min",
        type=float,
        default=None,
        help="Optional minimum confidence threshold. Omit to use all finite LoGeR points.",
    )
    parser.add_argument("--max-pred-points", type=int, default=500_000)
    parser.add_argument("--max-gt-points", type=int, default=300_000)
    parser.add_argument("--max-viewer-pred-points", type=int, default=150_000)
    parser.add_argument("--max-viewer-gt-points", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=65081)
    run_eval(parser.parse_args())


if __name__ == "__main__":
    main()
