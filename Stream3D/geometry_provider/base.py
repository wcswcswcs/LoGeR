from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FrameProjection:
    mask_info: dict[int, set[int]]
    frame_point_ids: list[int]
    depth_max: float
    diagnostics: dict[str, float | int | str | bool | None]


class GeometryProvider(Protocol):
    name: str
    uses_rgbd_for_prediction: bool
    uses_pose_for_prediction: bool
    uses_scannet_mesh_for_prediction: bool
    uses_gt_sim3_for_prediction: bool
    is_diagnostic_only: bool

    def project_frame_masks(
        self,
        *,
        dataset: object,
        scene_points: np.ndarray,
        mask_image: np.ndarray,
        frame_id: int,
        depth_max_pre: float,
    ) -> FrameProjection:
        ...
