from __future__ import annotations

from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.sim3 import fit_sim3_umeyama


def backproject_xy_world(stream: ScanNetStream, frame_id: int, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Backproject image xy samples to ScanNet world coordinates.

    This helper intentionally lives under ``geometry_provider`` so provider
    code can use it without importing diagnostic scripts from ``tools``. Calls
    to it use ScanNet depth/pose and must remain diagnostic/eval-only.
    """

    xy = np.asarray(xy, dtype=np.float32)
    world = np.full((xy.shape[0], 3), np.nan, dtype=np.float32)
    valid = np.zeros((xy.shape[0],), dtype=bool)
    if xy.size == 0:
        return world, valid
    depth = stream.load_depth(int(frame_id))
    pose = stream.load_pose(int(frame_id))
    intrinsics = stream.load_intrinsics()
    if not np.isfinite(pose).all():
        return world, valid
    h, w = depth.shape[:2]
    x = np.rint(xy[:, 0]).astype(np.int64)
    y = np.rint(xy[:, 1]).astype(np.int64)
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not np.any(in_bounds):
        return world, valid
    z = depth[y[in_bounds], x[in_bounds]].astype(np.float32)
    depth_valid = np.isfinite(z) & (z > 0.0)
    source_indices = np.flatnonzero(in_bounds)[depth_valid]
    if source_indices.size == 0:
        return world, valid
    x_f = x[source_indices].astype(np.float32)
    y_f = y[source_indices].astype(np.float32)
    z_f = z[depth_valid]
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    cam = np.stack([(x_f - cx) * z_f / fx, (y_f - cy) * z_f / fy, z_f, np.ones_like(z_f)], axis=1)
    pts = (pose @ cam.T).T[:, :3].astype(np.float32)
    finite = np.isfinite(pts).all(axis=1)
    valid[source_indices[finite]] = True
    world[source_indices[finite]] = pts[finite]
    return world, valid


def fit_transform(source: np.ndarray, target: np.ndarray, robust_trim_percentile: float) -> dict[str, Any] | None:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if source.shape[0] < 4:
        return None
    fit = fit_sim3_umeyama(source, target)
    residual = np.asarray(fit["residual"], dtype=np.float64)
    trim = float(robust_trim_percentile)
    if 0.0 < trim < 100.0 and residual.size >= 8:
        keep = residual <= float(np.percentile(residual, trim))
        if np.count_nonzero(keep) >= 4 and np.count_nonzero(keep) < residual.size:
            fit = fit_sim3_umeyama(source[keep], target[keep])
            fit["robust_trim_percentile"] = trim
            fit["robust_kept_anchors"] = int(np.count_nonzero(keep))
    return fit

