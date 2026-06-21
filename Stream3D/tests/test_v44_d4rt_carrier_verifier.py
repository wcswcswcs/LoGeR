from __future__ import annotations

import unittest

from stream4d_native.v44_typed_mask_assembly import d4rt_score


class V44D4RTCarrierVerifierTest(unittest.TestCase):
    def test_visible_outside_lowers_d4rt_score(self) -> None:
        clean = {"shared_tube_count": 4, "trusted_material_tube_count": 0, "material_union_count": 4, "visible_outside_conflict_ratio": 0.0}
        conflict = dict(clean, visible_outside_conflict_ratio=1.0)
        self.assertGreater(d4rt_score(clean), d4rt_score(conflict))


if __name__ == "__main__":
    unittest.main()
