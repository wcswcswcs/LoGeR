from __future__ import annotations

import unittest

from stream4d_native.v62_increment_attribution import build_v62_increment_attribution


class V62IncrementAttributionTest(unittest.TestCase):
    def test_update_and_bridge_material_are_reported(self) -> None:
        result = build_v62_increment_attribution()
        summary = result["summary"]
        self.assertGreater(summary["update_new_material_count"], 0)
        self.assertGreaterEqual(summary["update_new_confirmed_rate"], 0.50)
        self.assertGreaterEqual(summary["shortcut_shared_quarantine_rate"], 0.80)


if __name__ == "__main__":
    unittest.main()

