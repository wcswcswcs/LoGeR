from __future__ import annotations

import unittest

from stream4d_native.v62_decircularization import build_v62_decircularization


class V62DecircularizationTest(unittest.TestCase):
    def test_diagnostic_drop_and_typed_edges_pass_gate(self) -> None:
        result = build_v62_decircularization()
        summary = result["summary"]
        self.assertTrue(summary["gate"]["D1_delta_core_ARI_le_0_005"])
        self.assertTrue(summary["gate"]["D4_core_purity_ge_0_95"])
        self.assertTrue(summary["gate"]["uses_diagnostic_expected_in_prediction_false"])
        self.assertTrue(summary["gate"]["rebuilt_controls_available"])


if __name__ == "__main__":
    unittest.main()

