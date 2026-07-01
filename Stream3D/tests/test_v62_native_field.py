from __future__ import annotations

import unittest

from stream4d_native.v62_native_field import build_v62_native_field


class V62NativeFieldTest(unittest.TestCase):
    def test_component_field_is_method_safe_and_carrier_is_proxy(self) -> None:
        result = build_v62_native_field()
        summary = result["summary"]
        self.assertTrue(summary["gate"]["pass"])
        self.assertTrue(summary["component_level_field_available"])
        self.assertFalse(summary["carrier_level_field_available"])
        self.assertFalse(summary["uses_gt_for_prediction"])
        self.assertFalse(summary["uses_rgbd_pose_mesh_for_export"])


if __name__ == "__main__":
    unittest.main()

