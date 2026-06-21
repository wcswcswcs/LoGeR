from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestV50SameViewRelations(unittest.TestCase):
    def test_relation_gate_records_unreliable_fallback_relations(self) -> None:
        payload = json.loads((ROOT / "outputs/audit/v50_same_view_relations/relation_summary.json").read_text())
        self.assertFalse(payload["gate"]["pass"])
        self.assertFalse(payload["gate"]["exact_same_view_relation_available"])
        self.assertLess(payload["summary"]["part_relation_precision"], 0.70)
        self.assertLess(payload["summary"]["sibling_relation_precision"], 0.60)


if __name__ == "__main__":
    unittest.main()
