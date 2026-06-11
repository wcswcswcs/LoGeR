from __future__ import annotations

from .d4rt_raw_provider import D4RTRawProvider


class D4RTEvalSim3Provider(D4RTRawProvider):
    name = "d4rt_eval_sim3_provider"
    uses_rgbd_for_prediction = False
    uses_pose_for_prediction = False
    uses_scannet_mesh_for_prediction = False
    uses_gt_sim3_for_prediction = True
    is_diagnostic_only = True
