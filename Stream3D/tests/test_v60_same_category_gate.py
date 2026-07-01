from __future__ import annotations

import unittest

from stream4d_native.v60_fact_lock import calibrated_same_category_gate, wilson_upper_95


class V60SameCategoryGateTest(unittest.TestCase):
    def test_low_baseline_zero_false_passes_without_negative_threshold(self) -> None:
        gate = calibrated_same_category_gate(method_false_count=0, method_pair_count=103, baseline_false_rate=0.0268)
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["mode"], "low_baseline_exact_or_wilson")
        self.assertLessEqual(gate["method_wilson_upper95"], 0.05)

    def test_low_baseline_real_false_merges_fail_when_wilson_high(self) -> None:
        gate = calibrated_same_category_gate(method_false_count=3, method_pair_count=50, baseline_false_rate=0.0268)
        self.assertFalse(gate["pass"])

    def test_wilson_upper_known_zero_case(self) -> None:
        upper = wilson_upper_95(0, 103)
        self.assertIsNotNone(upper)
        self.assertGreater(upper or 0.0, 0.0)
        self.assertLess(upper or 1.0, 0.05)


if __name__ == "__main__":
    unittest.main()
