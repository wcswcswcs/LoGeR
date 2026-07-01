from __future__ import annotations

import unittest

from stream4d_native.v56_core_update_component_native import build_v56_component_native_core_update


class V56ComponentNativeCoreUpdateTest(unittest.TestCase):
    def test_component_native_payload_has_no_gt_prediction(self) -> None:
        payload = build_v56_component_native_core_update(
            component_min_shared_support=999999,
            objectlet_min_total_shared_support=999999,
        )
        self.assertFalse(payload["summary"]["uses_gt_for_prediction"])
        self.assertEqual(payload["summary"]["confirmed_added_component_count"], 0)


if __name__ == "__main__":
    unittest.main()

