from __future__ import annotations

import unittest

from stream4d_native.v46_signed_mask_graph import stage2_eligibility


class V46Stage2BlockedContractTest(unittest.TestCase):
    def test_stage2_blocked_until_stage1_controls_and_scale_pass(self) -> None:
        payload = stage2_eligibility({"gate": {"pass": False, "controls_pass": True}}, {"gate": {"scale_guard_pass": True}})
        self.assertEqual(payload["status"], "STAGE2_BLOCKED")
        self.assertFalse(payload["gate"]["pass"])

    def test_stage2_allowed_when_all_entry_gates_pass(self) -> None:
        payload = stage2_eligibility({"gate": {"pass": True, "controls_pass": True}}, {"gate": {"scale_guard_pass": True}})
        self.assertEqual(payload["status"], "STAGE2_ALLOWED")
        self.assertTrue(payload["gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
