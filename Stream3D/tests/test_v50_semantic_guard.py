from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestV50SemanticGuard(unittest.TestCase):
    def test_semantic_guard_is_diagnostic_only_when_dense_backend_missing(self) -> None:
        payload = json.loads((ROOT / "outputs/audit/v50_semantic_guard/semantic_guard_summary.json").read_text())
        self.assertFalse(payload["gate"]["component_dense_backend_available"])
        self.assertFalse(payload["gate"]["semantic_guard_enabled_for_selection"])
        self.assertEqual(payload["summary"]["selected_policy"], "S5_no_semantic_guard_for_selection_with_S0_diagnostic_negative_guard_recorded")


if __name__ == "__main__":
    unittest.main()
