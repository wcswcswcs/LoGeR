from __future__ import annotations

import unittest

from stream4d_native.v53_local_control_gap_audit import _component_maps


class V53LocalControlGapAuditTest(unittest.TestCase):
    def test_component_maps_reads_component_ids_from_objectlet_rows(self) -> None:
        rows = [
            {"variant": "L6", "objectlet_id": "o1", "component_ids": '["c1", "c2"]'},
            {"variant": "L6", "objectlet_id": "o2", "component_ids": '["c3"]'},
            {"variant": "L9", "objectlet_id": "m1", "component_ids": '["c2"]'},
        ]
        maps = _component_maps(rows)
        self.assertEqual(maps["L6"]["c1"], "o1")
        self.assertEqual(maps["L6"]["c3"], "o2")
        self.assertEqual(maps["L9"]["c2"], "m1")


if __name__ == "__main__":
    unittest.main()
