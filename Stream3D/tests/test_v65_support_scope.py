from __future__ import annotations

import unittest

from stream4d_native.v65_ap_contract import build_v65_ap_contract


class V65SupportScopeTest(unittest.TestCase):
    def test_bridge_stream3d_same_support_is_blocked_by_frame_policy(self) -> None:
        payload = build_v65_ap_contract(command_rows=[])
        matrix = payload["ap_comparability_matrix"]
        pair = next(row for row in matrix if row["left_row_id"] == "A1" and row["right_row_id"] == "A4")
        self.assertTrue(pair["support_scope_same"])
        self.assertTrue(pair["support_policy_hash_same"])
        self.assertFalse(pair["input_frame_policy_same"])
        self.assertEqual(pair["comparison_status"], "not_comparable_input_frame_policy")

    def test_no_pairs_are_comparable_without_same_support_hash_and_frame_policy(self) -> None:
        payload = build_v65_ap_contract(command_rows=[])
        self.assertEqual(payload["summary"]["comparison_allowed_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
