from __future__ import annotations

import unittest

from stream4d_native.v44_typed_mask_assembly import _duplicate_rate


class V44MaskLatticeSolverTest(unittest.TestCase):
    def test_duplicate_rate_detects_overlapping_fields(self) -> None:
        objects = [
            {"semantic_masklet_ids": [1, 2, 3]},
            {"semantic_masklet_ids": [1, 2, 3, 4]},
        ]
        self.assertEqual(_duplicate_rate(objects), 1.0)


if __name__ == "__main__":
    unittest.main()
