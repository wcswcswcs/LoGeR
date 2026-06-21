from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestV50Stage2Policy(unittest.TestCase):
    def test_stage2_blocked_until_method_gate_and_controls_pass(self) -> None:
        full = json.loads((ROOT / "outputs/audit/v50_full_stage1/full_stage1_summary.json").read_text())
        stage2 = json.loads((ROOT / "outputs/audit/v50_stage2/stage2_eligibility_summary.json").read_text())
        self.assertFalse(full["method_claim_gate"]["pass"])
        self.assertFalse(full["control_gate"]["pass"])
        self.assertFalse(stage2["stage2_allowed"])
        self.assertIn("stage1_not_passed", stage2["stage2_block_reason"])
        self.assertIn("controls_failed", stage2["stage2_block_reason"])


if __name__ == "__main__":
    unittest.main()
