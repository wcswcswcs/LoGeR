from __future__ import annotations

import unittest

from stream4d_native.matching_significance import minimum_significant_gate, significance_summary


class V432SignificantGateTest(unittest.TestCase):
    def test_minimum_gate_requires_real_improvement(self) -> None:
        gate = minimum_significant_gate(
            {
                "4D_ARI": 0.42599481039581194,
                "4D_purity": 0.8673519940549913,
                "4D_completeness": 0.5056972999752292,
                "temporal_span_mean": 1.702673104336451,
                "scene0081_ARI": 0.20073910315166837,
            }
        )
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["4D_ARI_pass"])

    def test_zero_scene_delta_fails_bootstrap_significance(self) -> None:
        rows = [
            {"scene": "scene0011_00", "4D_ARI": 0.5, "4D_completeness": 0.6},
            {"scene": "scene0081_01", "4D_ARI": 0.2, "4D_completeness": 0.4},
        ]
        sig = significance_summary(rows, rows, samples=100)
        self.assertFalse(sig["checks"]["pass"])
        self.assertEqual(sig["checks"]["median_scene_delta_ARI"], 0.0)


if __name__ == "__main__":
    unittest.main()
