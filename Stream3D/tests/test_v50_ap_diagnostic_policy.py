from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestV50APDiagnosticPolicy(unittest.TestCase):
    def test_ap_bridge_is_diagnostic_only_and_method_safe_ap_unavailable(self) -> None:
        payload = json.loads((ROOT / "outputs/audit/v50_ap_diagnostic/ap_export_summary.json").read_text())
        self.assertTrue(payload["gate"]["rgbd_bridge_ap_ran"])
        self.assertTrue(payload["gate"]["ap6_constant_score_min_region_ran"])
        self.assertTrue(payload["gate"]["ap7_wta_conflict_suppression_ran"])
        self.assertFalse(payload["gate"]["method_safe_ap_available"])
        ap5 = next(row for row in payload["ap_rows"] if row["variant"] == "AP5_v50_best_identity_rgbd_pose_mesh_bridge_diagnostic")
        self.assertFalse(ap5["is_method_result"])
        self.assertTrue(ap5["is_diagnostic_only"])
        self.assertTrue(ap5["forbidden_for_method_table"])
        self.assertTrue(ap5["uses_rgbd_pose_mesh_for_export"])
        self.assertFalse(ap5["uses_gt_for_prediction"])
        for variant in [
            "AP6_v50_best_identity_constant_score_min_region_sweep",
            "AP7_v50_best_identity_wta_conflict_suppression",
        ]:
            row = next(item for item in payload["ap_rows"] if item["variant"] == variant)
            self.assertEqual(row["status"], "ran")
            self.assertFalse(row["is_method_result"])
            self.assertTrue(row["is_diagnostic_only"])
            self.assertTrue(row["forbidden_for_method_table"])
            self.assertTrue(row["uses_rgbd_pose_mesh_for_export"])
            self.assertFalse(row["uses_gt_for_prediction"])


if __name__ == "__main__":
    unittest.main()
