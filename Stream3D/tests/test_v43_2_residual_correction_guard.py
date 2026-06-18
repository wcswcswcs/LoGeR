from __future__ import annotations

import unittest

from stream4d_native.regression_guarded_matching import phase_d_semantic_gate, regression_guard


class V432ResidualCorrectionGuardTest(unittest.TestCase):
    def test_regression_guard_rejects_no_gain_for_semantic_phase(self) -> None:
        baseline = {
            "4D_ARI": 0.426,
            "4D_purity": 0.868,
            "4D_completeness": 0.506,
            "temporal_span_mean": 1.70,
        }
        candidate = {**baseline, "changed_object_ratio": 0.0}
        gate = phase_d_semantic_gate(candidate, baseline, hard_scene_delta_ari=0.0)
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["ari_pass"])

    def test_small_purity_drop_with_ari_gain_can_pass_basic_regression_guard(self) -> None:
        baseline = {"4D_ARI": 0.42, "4D_purity": 0.868, "4D_completeness": 0.50, "temporal_span_mean": 1.70}
        candidate = {"4D_ARI": 0.46, "4D_purity": 0.866, "4D_completeness": 0.52, "temporal_span_mean": 1.69}
        self.assertTrue(regression_guard(candidate, baseline, min_delta_ari=0.035, min_delta_completeness=0.015)["pass"])


if __name__ == "__main__":
    unittest.main()
