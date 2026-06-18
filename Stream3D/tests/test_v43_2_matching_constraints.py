from __future__ import annotations

import unittest

from stream4d_native.matching_significance import compactness_gate, control_gate


class V432MatchingConstraintsTest(unittest.TestCase):
    def test_compactness_requires_no_birth_and_prediction_cap(self) -> None:
        metrics = {
            "birth_from_d4rt_tube_count": 0,
            "mean_predictions_per_scene": 64,
            "duplicate_rate": 0.0,
            "conflict_rate": 0.0,
            "unknown_tube_ratio": 0.2,
            "changed_object_ratio": 0.0,
        }
        self.assertTrue(compactness_gate(metrics)["pass"])

    def test_controls_fail_when_required_rows_are_missing(self) -> None:
        metrics = {"real_minus_no_temporal": 0.4}
        gate = control_gate(metrics)
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["real_minus_shuffled_pass"])
        self.assertFalse(gate["real_minus_mask_only_pass"])


if __name__ == "__main__":
    unittest.main()
