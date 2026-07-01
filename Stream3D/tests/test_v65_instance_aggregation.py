from __future__ import annotations

import unittest

from stream4d_native.v65_instance_aggregation import build_v65_instance_aggregation


class V65InstanceAggregationTest(unittest.TestCase):
    def test_oracle_rows_are_forbidden_for_method_table(self) -> None:
        payload = build_v65_instance_aggregation(command_rows=[])
        oracle_rows = [row for row in payload["aggregation_metric_rows"] if row["variant"] == "I6"]
        self.assertEqual(len(oracle_rows), 2)
        self.assertTrue(all(row["uses_gt_for_prediction"] for row in oracle_rows))
        self.assertTrue(all(row["forbidden_for_method_table"] for row in oracle_rows))

    def test_non_gt_aggregation_blocker_is_explicit(self) -> None:
        payload = build_v65_instance_aggregation(command_rows=[])
        summary = payload["summary"]
        self.assertFalse(summary["non_gt_aggregation_available"])
        self.assertEqual(summary["blocker"], "non_GT_fragment_aggregation_signal_insufficient")


if __name__ == "__main__":
    unittest.main()
