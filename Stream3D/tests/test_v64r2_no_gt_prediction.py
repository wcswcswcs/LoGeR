from __future__ import annotations

import unittest

from stream4d_native.v64r2_active_query_optional import build_v64r2_active_query_optional
from stream4d_native.v64r2_scannet_exporters import build_v64r2_ap_contract


class V64R2NoGTPredictionTest(unittest.TestCase):
    def test_method_candidate_policy_rows_do_not_use_gt_or_rgbd_bridge(self) -> None:
        payload = build_v64r2_ap_contract()
        for row in payload["exporter_policy_rows"]:
            if row["exporter_name"] in {"E0_native_component_field", "E1_method_safe_projection_if_available"}:
                self.assertFalse(row["uses_gt_for_prediction"])
                self.assertFalse(row["uses_rgbd_pose_mesh_for_export"])
                self.assertFalse(row["uses_scannet_mesh_for_prediction"])

    def test_active_query_failure_does_not_block_ap_or_dynamic(self) -> None:
        payload = build_v64r2_active_query_optional()
        self.assertFalse(payload["blocks_scannet_ap"])
        self.assertFalse(payload["blocks_dynamic"])
        self.assertIn(payload["active_query_status"], {"GO_ACTIVE_QUERY_EXTENSION", "REMOVE_ACTIVE_QUERY_FROM_MAIN"})


if __name__ == "__main__":
    unittest.main()
