from __future__ import annotations

import unittest

from stream4d_native.v46_signed_mask_graph import material_support_score, shared_tube_jaccard


class V46ViewConsensusTest(unittest.TestCase):
    def test_material_support_can_exceed_sparse_shared_jaccard(self) -> None:
        row = {
            "shared_tube_count": "1",
            "trusted_material_tube_count": "3",
            "object_part_tube_count": "4",
            "material_union_count": "16",
        }
        self.assertEqual(shared_tube_jaccard(row), 1 / 16)
        self.assertEqual(material_support_score(row), 0.5)


if __name__ == "__main__":
    unittest.main()
