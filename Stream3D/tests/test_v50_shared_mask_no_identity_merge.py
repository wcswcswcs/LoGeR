from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestV50SharedMaskNoIdentityMerge(unittest.TestCase):
    def test_keymask_cosupport_does_not_create_multicomponent_identity_merge(self) -> None:
        payload = json.loads((ROOT / "outputs/audit/v50_relation_propagation/propagation_summary.json").read_text())
        self.assertEqual(payload["summary"]["mean_keymask_cosupport"], 0.0)
        self.assertFalse(payload["gate"]["keymask_multicomponent_support_available"])


if __name__ == "__main__":
    unittest.main()
