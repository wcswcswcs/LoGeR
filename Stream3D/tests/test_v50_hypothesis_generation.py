from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestV50HypothesisGeneration(unittest.TestCase):
    def test_hypothesis_pool_passes_after_weak_propagation_demotion(self) -> None:
        payload = json.loads((ROOT / "outputs/audit/v50_hypothesis_generation/hypothesis_summary.json").read_text())
        self.assertTrue(payload["gate"]["pass"])
        self.assertTrue(payload["summary"]["weak_propagation_score_demotion"])
        self.assertGreaterEqual(payload["summary"]["GT_object_has_hypothesis@0.50"], 0.45)
        self.assertLessEqual(payload["summary"]["hypothesis_conflict_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
