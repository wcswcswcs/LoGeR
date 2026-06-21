from __future__ import annotations

import unittest

from stream4d_native.v44_typed_mask_assembly import stage2_geometry_diagnostic


class V44Stage2BlockedTest(unittest.TestCase):
    def test_stage2_blocked_when_stage1_or_controls_fail(self) -> None:
        payload = stage2_geometry_diagnostic({"gate": {"pass": True}}, {"gate": {"pass": False}})
        self.assertFalse(payload["stage2_allowed"])
        self.assertEqual(payload["status"], "STAGE2_BLOCKED")


if __name__ == "__main__":
    unittest.main()
