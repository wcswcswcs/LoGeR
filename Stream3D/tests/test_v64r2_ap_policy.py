from __future__ import annotations

import unittest

from stream4d_native.v64r2_scannet_exporters import build_v64r2_ap_contract


class V64R2APPolicyTest(unittest.TestCase):
    def test_rgbd_bridge_is_diagnostic_and_forbidden_for_method_table(self) -> None:
        payload = build_v64r2_ap_contract()
        rows = {row["exporter_name"]: row for row in payload["exporter_policy_rows"]}
        bridge = rows["E2_rgbd_pose_mesh_bridge_diagnostic"]
        self.assertTrue(bridge["uses_rgbd_pose_mesh_for_export"])
        self.assertTrue(bridge["is_diagnostic_only"])
        self.assertTrue(bridge["forbidden_for_method_table"])
        self.assertFalse(bridge["is_method_result"])

    def test_method_safe_projection_never_uses_rgbd_pose_mesh(self) -> None:
        payload = build_v64r2_ap_contract()
        rows = {row["exporter_name"]: row for row in payload["exporter_policy_rows"]}
        method = rows["E1_method_safe_projection_if_available"]
        self.assertFalse(method["uses_gt_for_prediction"])
        self.assertFalse(method["uses_rgbd_pose_mesh_for_export"])
        self.assertFalse(method["uses_scannet_mesh_for_prediction"])
        self.assertFalse(method["forbidden_for_method_table"])


if __name__ == "__main__":
    unittest.main()
