from __future__ import annotations

import unittest

from stream4d_native.role_gated_geometry_optimization import geometry_stage_decision


class V432GeometryStageGateTest(unittest.TestCase):
    def test_geometry_stage_is_blocked_until_stage1_passes(self) -> None:
        summary = geometry_stage_decision({"final_label": "NO_GO_MATCHING_NOT_SIGNIFICANT"})
        self.assertFalse(summary["stage2_allowed"])
        self.assertEqual(summary["status"], "STAGE2_BLOCKED_MATCHING_NOT_SIGNIFICANT")


if __name__ == "__main__":
    unittest.main()
