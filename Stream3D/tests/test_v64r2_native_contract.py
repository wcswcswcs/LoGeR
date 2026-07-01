from __future__ import annotations

import unittest

from stream4d_native.v64r2_native_contract import build_v64r2_native_contract


class V64R2NativeContractTest(unittest.TestCase):
    def test_native_contract_exports_component_level_schema(self) -> None:
        payload = build_v64r2_native_contract()
        summary = payload["summary"]
        self.assertTrue(summary["gate"]["pass"])
        self.assertGreater(summary["object_count"], 0)
        self.assertGreater(summary["material_count"], 0)
        self.assertTrue(summary["component_level_available"])
        self.assertFalse(summary["uses_gt_for_prediction"])


if __name__ == "__main__":
    unittest.main()
