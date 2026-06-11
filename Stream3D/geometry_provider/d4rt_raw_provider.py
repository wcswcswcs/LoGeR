from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .base import FrameProjection


class D4RTRawProvider:
    name = "d4rt_raw_provider"
    uses_rgbd_for_prediction = False
    uses_pose_for_prediction = False
    uses_scannet_mesh_for_prediction = False
    uses_gt_sim3_for_prediction = False
    is_diagnostic_only = False

    def __init__(self, *, geometry_root: str | Path, nn_radius: float = 0.05) -> None:
        self.geometry_root = Path(geometry_root)
        self.nn_radius = float(nn_radius)

    def _load_points(self, frame_id: int) -> np.ndarray:
        paths = sorted(self.geometry_root.glob(f"**/*frame{int(frame_id):06d}*_points.npy"))
        if not paths:
            paths = sorted(self.geometry_root.glob(f"**/*frame{int(frame_id):06d}*.npy"))
        if not paths:
            return np.empty((0, 3), dtype=np.float32)
        parts = [np.asarray(np.load(path), dtype=np.float32).reshape(-1, 3) for path in paths]
        return np.concatenate(parts, axis=0) if parts else np.empty((0, 3), dtype=np.float32)

    def project_frame_masks(
        self,
        *,
        dataset: object,
        scene_points: np.ndarray,
        mask_image: np.ndarray,
        frame_id: int,
        depth_max_pre: float,
    ) -> FrameProjection:
        del dataset, depth_max_pre
        d4rt_points = self._load_points(frame_id)
        mask_info: dict[int, set[int]] = {}
        if d4rt_points.size == 0:
            return FrameProjection(mask_info={}, frame_point_ids=[], depth_max=0.0, diagnostics={"provider": self.name, "projection_hit_rate": 0.0})
        finite = np.isfinite(d4rt_points).all(axis=1)
        d4rt_points = d4rt_points[finite]
        if d4rt_points.size == 0:
            return FrameProjection(mask_info={}, frame_point_ids=[], depth_max=0.0, diagnostics={"provider": self.name, "projection_hit_rate": 0.0})
        tree = cKDTree(np.asarray(scene_points, dtype=np.float32))
        dist, idx = tree.query(d4rt_points, k=1, distance_upper_bound=self.nn_radius)
        hit = np.isfinite(dist) & (idx < len(scene_points))
        frame_points = sorted(set(int(v) for v in idx[hit].tolist()))
        # Raw D4RT provider has no image-space mask ownership unless paired with a mapping file.
        if frame_points:
            positive_ids = np.unique(np.asarray(mask_image)[np.asarray(mask_image) > 0]).astype(np.int64)
            if positive_ids.size == 1:
                mask_info[int(positive_ids[0])] = set(frame_points)
        return FrameProjection(
            mask_info=mask_info,
            frame_point_ids=frame_points,
            depth_max=0.0,
            diagnostics={
                "provider": self.name,
                "local_point_count": int(d4rt_points.shape[0]),
                "projection_hit_rate": float(np.mean(hit)) if hit.size else 0.0,
                "num_frame_points": int(len(frame_points)),
            },
        )
