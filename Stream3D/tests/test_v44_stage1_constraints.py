from __future__ import annotations

import unittest

from stream4d_native.v44_typed_mask_assembly import _object_sets_from_edges


class V44Stage1ConstraintsTest(unittest.TestCase):
    def test_mixed_mask_cannot_birth_object(self) -> None:
        tokens = [
            {"token_id": 1, "diagnostic_gt_instance": 1},
            {"token_id": 2, "diagnostic_gt_instance": 1},
        ]
        roles = {1: {"role": "mixed"}, 2: {"role": "mixed"}}
        edges = [{"token_i": 1, "token_j": 2, "shared_tube_ids": "[]"}]
        objects, _pred, _gt = _object_sets_from_edges(tokens, edges, roles)
        self.assertEqual(objects, [])

    def test_part_only_singleton_does_not_birth_object(self) -> None:
        tokens = [{"token_id": 1, "diagnostic_gt_instance": 1}]
        roles = {1: {"role": "part"}}
        objects, _pred, _gt = _object_sets_from_edges(tokens, [], roles)
        self.assertEqual(objects, [])


if __name__ == "__main__":
    unittest.main()
