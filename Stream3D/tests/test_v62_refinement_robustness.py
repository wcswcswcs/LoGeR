from __future__ import annotations

import unittest

from stream4d_native.v62_refinement_robustness import build_v62_refinement_robustness


class V62RefinementRobustnessTest(unittest.TestCase):
    def test_combined_refinement_reduces_pollution_under_perturbation(self) -> None:
        result = build_v62_refinement_robustness()
        self.assertTrue(result["summary"]["gate"]["P1_underseg_false_merge_reduction_ge_0_05"])
        self.assertTrue(result["summary"]["gate"]["P2_same_category_merge_reduction_ge_0_05"])


if __name__ == "__main__":
    unittest.main()

