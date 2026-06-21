from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestV50HypothesisSelection(unittest.TestCase):
    def test_selection_gate_is_diagnostic_scope_not_method_claim(self) -> None:
        payload = json.loads((ROOT / "outputs/audit/v50_hypothesis_selection/selection_summary.json").read_text())
        best = payload["summary"]["best_real_row"]
        self.assertTrue(payload["gate"]["pass"])
        self.assertEqual(best["metric_scope"], "component_vote_diagnostic")
        self.assertEqual(best["real_minus_shuffled_ARI"], 0.0)
        self.assertEqual(best["real_minus_no_temporal_ARI"], 0.0)
        self.assertEqual(best["real_minus_mask_only_ARI"], 0.0)


if __name__ == "__main__":
    unittest.main()
