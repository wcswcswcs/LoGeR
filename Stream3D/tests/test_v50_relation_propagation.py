from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestV50RelationPropagation(unittest.TestCase):
    def test_propagation_does_not_beat_controls(self) -> None:
        payload = json.loads((ROOT / "outputs/audit/v50_relation_propagation/propagation_summary.json").read_text())
        self.assertFalse(payload["gate"]["pass"])
        self.assertTrue(payload["gate"]["relation_branch_weak"])
        self.assertFalse(payload["gate"]["propagation_real_minus_shuffled_AUC_pass"])
        self.assertFalse(payload["gate"]["propagation_real_minus_no_temporal_AUC_pass"])


if __name__ == "__main__":
    unittest.main()
