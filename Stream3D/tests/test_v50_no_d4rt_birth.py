from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestV50NoD4RTBirth(unittest.TestCase):
    def test_selected_objects_do_not_birth_from_d4rt_or_maskless_path(self) -> None:
        payload = json.loads((ROOT / "outputs/audit/v50_hypothesis_selection/selection_summary.json").read_text())
        best = payload["summary"]["best_real_row"]
        self.assertEqual(best["birth_from_d4rt_tube_count"], 0)
        self.assertEqual(best["maskless_object_count"], 0)


if __name__ == "__main__":
    unittest.main()
