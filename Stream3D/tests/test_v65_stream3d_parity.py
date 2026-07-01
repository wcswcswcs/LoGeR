from __future__ import annotations

import unittest

from stream4d_native.v65_stream3d_parity import build_v65_stream3d_parity


class V65Stream3DParityTest(unittest.TestCase):
    def test_stream3d_rows_record_input_frame_policy(self) -> None:
        payload = build_v65_stream3d_parity(command_rows=[])
        rows = payload["stream3d_ap_rows"]
        self.assertGreaterEqual(len(rows), 5)
        self.assertTrue(all(row.get("input_frame_policy") for row in rows))
        self.assertTrue(payload["summary"]["gate"]["input_frame_policy_recorded"])

    def test_same_support_rows_are_diagnostic_not_official_win_loss(self) -> None:
        payload = build_v65_stream3d_parity(command_rows=[])
        bias_rows = payload["support_bias_rows"]
        same_support = [row for row in bias_rows if row["support_scope"] == "SAME_SUPPORT_ON_SOMA_SUPPORT_DIAGNOSTIC"]
        self.assertTrue(same_support)
        self.assertTrue(all(not row["can_use_for_official_win_loss"] for row in same_support))


if __name__ == "__main__":
    unittest.main()
