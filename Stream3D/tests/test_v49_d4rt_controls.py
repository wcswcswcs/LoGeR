from __future__ import annotations

import unittest

from stream4d_native.v49_mosaic_stage1 import build_d4rt_control_audit


class TestV49D4RTControls(unittest.TestCase):
    def test_control_audit_has_required_control_rows(self) -> None:
        payload = build_d4rt_control_audit()
        controls = {row["control"] for row in payload["control_rows"]}
        self.assertIn("C0_real_D4RT", controls)
        self.assertIn("C3_no_temporal_component_support", controls)
        self.assertFalse(payload["uses_gt_for_prediction"])


if __name__ == "__main__":
    unittest.main()
