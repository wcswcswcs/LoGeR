from __future__ import annotations

import numpy as np

from .base import FrameProjection


class RGBDGeometryProvider:
    name = "rgbd_provider"
    uses_rgbd_for_prediction = True
    uses_pose_for_prediction = True
    uses_scannet_mesh_for_prediction = True
    uses_gt_sim3_for_prediction = False
    is_diagnostic_only = True

    def project_frame_masks(
        self,
        *,
        dataset: object,
        scene_points: np.ndarray,
        mask_image: np.ndarray,
        frame_id: int,
        depth_max_pre: float,
    ) -> FrameProjection:
        # Import lazily so native D4RT code never depends on this RGB-D bridge.
        from utils.mask_backprojection import turn_mask_to_point, turn_point_to_mask

        dataset_name = getattr(getattr(dataset, "args", None), "dataset", "")
        if dataset_name == "matterport3d":
            mask_info, _, frame_point_ids, depth_max = turn_point_to_mask(
                dataset,
                scene_points,
                mask_image,
                frame_id,
                depth_max_pre=depth_max_pre,
                threshold=0.1,
            )
        else:
            mask_info, _, frame_point_ids = turn_mask_to_point(
                dataset,
                scene_points,
                mask_image,
                frame_id,
                depth_max_pre=depth_max_pre,
            )
            depth_max = 0.1
        return FrameProjection(
            mask_info=mask_info,
            frame_point_ids=list(frame_point_ids),
            depth_max=float(depth_max),
            diagnostics={
                "provider": self.name,
                "num_masks_projected": int(len(mask_info)),
                "num_frame_points": int(len(frame_point_ids)),
            },
        )
