from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestV50KeyMaskSelection(unittest.TestCase):
    def test_key_masks_reduce_count_but_record_underseg_failure(self) -> None:
        payload = json.loads((ROOT / "outputs/audit/v50_key_masks/key_mask_summary.json").read_text())
        self.assertTrue(payload["gate"]["component_coverage_pass"])
        self.assertTrue(payload["gate"]["key_mask_ratio_pass"])
        self.assertFalse(payload["gate"]["large_underseg_reduction_pass"])
        self.assertLessEqual(payload["summary"]["key_mask_ratio"], 0.55)


if __name__ == "__main__":
    unittest.main()
