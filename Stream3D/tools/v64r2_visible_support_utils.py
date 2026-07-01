from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from stream4d.scannet_stream import ScanNetStream


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def read_seq_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def frame_ids_from_debug_root(debug_root: str | Path, scene: str) -> list[int]:
    scene_dir = Path(debug_root) / scene
    out: set[int] = set()
    for carrier_path in sorted(scene_dir.glob("carriers_window*.npz")):
        manifest = carrier_path.with_name(f"{carrier_path.stem}_manifest.json")
        if manifest.exists():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                for key in ("raw_frame_ids", "frame_indices", "frame_ids"):
                    vals = [int(v) for v in payload.get(key, [])]
                    if vals:
                        out.update(vals)
                        break
            except Exception:
                pass
        if out:
            continue
        try:
            with np.load(carrier_path) as data:
                if "src_frame_global" in data.files:
                    vals = np.asarray(data["src_frame_global"], dtype=np.int64)
                    out.update(int(v) for v in np.unique(vals).tolist() if int(v) >= 0)
        except Exception:
            pass
    return sorted(out)


def scene_points_from_stream(stream: ScanNetStream) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required to load ScanNet scene mesh points") from exc
    points = np.asarray(o3d.io.read_point_cloud(str(stream.mesh_path)).points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"Failed to load scene mesh points from {stream.mesh_path}")
    return points


def visible_support_point_ids(
    stream: ScanNetStream,
    scene_points: np.ndarray,
    frame_ids: list[int],
    *,
    pixel_stride: int,
    nn_radius: float,
    mask_positive_only: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not frame_ids:
        return np.empty((0,), dtype=np.int64), {
            "frame_count": 0,
            "support_pixel_queries": 0,
            "support_nn_hits": 0,
            "support_unique_points": 0,
        }
    intr = stream.load_intrinsics()
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    tree = cKDTree(np.asarray(scene_points, dtype=np.float32))
    stride = max(1, int(pixel_stride))
    support: set[int] = set()
    queries = 0
    hits = 0
    valid_frames = 0
    skipped_bad_pose = 0
    skipped_missing_mask = 0
    for frame_id in frame_ids:
        depth = stream.load_depth(int(frame_id))
        pose = stream.load_pose(int(frame_id))
        if not np.isfinite(pose).all():
            skipped_bad_pose += 1
            continue
        mask = None
        if mask_positive_only:
            try:
                mask = stream.load_mask(int(frame_id))
            except FileNotFoundError:
                skipped_missing_mask += 1
                continue
            if mask.shape != depth.shape:
                mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        h, w = depth.shape[:2]
        yy, xx = np.mgrid[0:h:stride, 0:w:stride]
        z = depth[yy, xx].reshape(-1).astype(np.float32)
        x = xx.reshape(-1).astype(np.float32)
        y = yy.reshape(-1).astype(np.float32)
        ok = np.isfinite(z) & (z > 0.0)
        if mask_positive_only and mask is not None:
            m = mask[yy, xx].reshape(-1)
            ok &= m.astype(np.int64) > 0
        if not np.any(ok):
            continue
        cam = np.stack(
            [
                (x[ok] - cx) * z[ok] / fx,
                (y[ok] - cy) * z[ok] / fy,
                z[ok],
                np.ones((int(np.count_nonzero(ok)),), dtype=np.float32),
            ],
            axis=1,
        )
        world = (pose @ cam.T).T[:, :3].astype(np.float32)
        finite = np.isfinite(world).all(axis=1)
        if not np.any(finite):
            continue
        valid_frames += 1
        world = world[finite]
        queries += int(world.shape[0])
        dist, idx = tree.query(world, k=1, distance_upper_bound=float(nn_radius))
        hit = np.isfinite(dist) & (idx < scene_points.shape[0])
        hits += int(np.count_nonzero(hit))
        support.update(int(v) for v in idx[hit].tolist())
    ids = np.asarray(sorted(support), dtype=np.int64)
    return ids, {
        "frame_count": int(len(frame_ids)),
        "valid_frame_count": int(valid_frames),
        "skipped_bad_pose": int(skipped_bad_pose),
        "skipped_missing_mask": int(skipped_missing_mask),
        "support_pixel_stride": int(stride),
        "support_nn_radius": float(nn_radius),
        "support_mask_positive_only": bool(mask_positive_only),
        "support_pixel_queries": int(queries),
        "support_nn_hits": int(hits),
        "support_nn_hit_rate": float(hits / max(queries, 1)),
        "support_unique_points": int(ids.shape[0]),
        "support_scene_point_ratio": float(ids.shape[0] / max(scene_points.shape[0], 1)),
    }


def point_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    pred = pred[np.isfinite(pred).all(axis=1)]
    gt = gt[np.isfinite(gt).all(axis=1)]
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return {"status": "empty", "pred_point_count": int(pred.shape[0]), "gt_point_count": int(gt.shape[0])}
    gt_tree = cKDTree(gt)
    pred_tree = cKDTree(pred)
    pred_to_gt, _ = gt_tree.query(pred, k=1)
    gt_to_pred, _ = pred_tree.query(gt, k=1)
    out: dict[str, Any] = {
        "status": "ok",
        "pred_point_count": int(pred.shape[0]),
        "gt_point_count": int(gt.shape[0]),
        "chamfer_l1": float(np.mean(pred_to_gt) + np.mean(gt_to_pred)),
        "chamfer_l2": float(np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2)),
        "pred_to_gt_median": float(np.median(pred_to_gt)),
        "pred_to_gt_p90": float(np.percentile(pred_to_gt, 90)),
        "gt_to_pred_median": float(np.median(gt_to_pred)),
        "gt_to_pred_p90": float(np.percentile(gt_to_pred, 90)),
    }
    for tau in (0.01, 0.05, 0.10, 0.20):
        precision = float(np.mean(pred_to_gt < tau))
        recall = float(np.mean(gt_to_pred < tau))
        out[f"precision@{int(tau * 100)}cm"] = precision
        out[f"recall@{int(tau * 100)}cm"] = recall
        out[f"fscore@{int(tau * 100)}cm"] = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return out
