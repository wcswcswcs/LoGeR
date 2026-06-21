from __future__ import annotations

import unittest
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import write_json
from stream4d_native.v46_signed_mask_graph import stage2_eligibility


class V46ScaleGuardIntegrationTest(unittest.TestCase):
    def test_stage2_requires_scale_guard(self) -> None:
        stage1 = {"gate": {"pass": True, "controls_pass": True}}
        fact = {"gate": {"scale_guard_pass": False}}
        payload = stage2_eligibility(stage1, fact)
        self.assertEqual(payload["status"], "STAGE2_BLOCKED")
        self.assertFalse(payload["gate"]["pass"])

    def test_write_json_import_still_available_for_tools(self) -> None:
        self.assertTrue(callable(write_json))
        self.assertEqual(Path("x").name, "x")


if __name__ == "__main__":
    unittest.main()
