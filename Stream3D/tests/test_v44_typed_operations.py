from __future__ import annotations

import unittest

from stream4d_native.v44_typed_mask_assembly import _select_strategy_edges


class V44TypedOperationsTest(unittest.TestCase):
    def test_same_frame_cannot_link_vetoes_strategy_edges(self) -> None:
        rows = [
            {
                "token_i": 1,
                "token_j": 2,
                "semantic_affinity": 0.99,
                "object_affinity": 0.99,
                "same_frame_cannot_link": True,
                "visible_outside_conflict_ratio": 0.0,
            }
        ]
        roles = {1: {"role": "core"}, 2: {"role": "part"}}
        selected = _select_strategy_edges(rows, roles, {}, strategy="A_core_first_absorb_replace")
        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
