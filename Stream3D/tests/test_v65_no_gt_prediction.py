from __future__ import annotations

import unittest

from stream4d_native.v65_ap_contract import build_v65_ap_contract
from stream4d_native.v65_instance_aggregation import build_v65_instance_aggregation


class V65NoGTPredictionTest(unittest.TestCase):
    def test_no_gt_or_rgbd_rows_are_method_results(self) -> None:
        payload = build_v65_ap_contract(command_rows=[])
        self.assertTrue(payload["evaluator_selfcheck_summary"]["soma_inference_policy_clean"])
        self.assertEqual(payload["evaluator_selfcheck_summary"]["soma_inference_policy_violations"], [])
        for row in payload["ap_contract_rows"]:
            if row["uses_gt_for_prediction"] or row["uses_gt_for_evaluation"] or row["uses_rgbd_pose_mesh_for_export"]:
                self.assertTrue(row["forbidden_for_method_table"])
                self.assertFalse(row["is_method_result"])

    def test_oracle_instance_aggregation_cannot_be_method_result(self) -> None:
        payload = build_v65_instance_aggregation(command_rows=[])
        for row in payload["aggregation_metric_rows"]:
            if row["variant"] == "I6":
                self.assertTrue(row["uses_gt_for_prediction"])
                self.assertTrue(row["forbidden_for_method_table"])
                self.assertEqual(row["status"], "ran")


if __name__ == "__main__":
    unittest.main()
