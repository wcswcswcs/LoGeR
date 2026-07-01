from __future__ import annotations

import unittest

from stream4d_native.v65_ap_contract import build_v65_ap_contract


class V65APContractTest(unittest.TestCase):
    def test_ap_contract_records_support_hash_and_frame_policy(self) -> None:
        payload = build_v65_ap_contract(command_rows=[])
        rows = payload["ap_contract_rows"]
        self.assertTrue(all(row["support_scope"] for row in rows))
        self.assertTrue(all(row["input_frame_policy"] for row in rows))
        self.assertTrue(all("support_policy_hash" in row for row in rows))

    def test_no_method_safe_ap_is_promoted(self) -> None:
        payload = build_v65_ap_contract(command_rows=[])
        summary = payload["summary"]
        self.assertEqual(summary["method_safe_rows_with_AP"], [])
        self.assertTrue(summary["gate"]["all_evaluated_rows_have_hash"])


if __name__ == "__main__":
    unittest.main()
