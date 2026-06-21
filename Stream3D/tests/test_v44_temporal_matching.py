from __future__ import annotations

import unittest

from stream4d_native.v44_typed_mask_assembly import _frame_rank_map


class V44TemporalMatchingTest(unittest.TestCase):
    def test_frame_rank_uses_sample_order(self) -> None:
        rows = [
            {"token_id": 1, "frame_id": 80},
            {"token_id": 2, "frame_id": 0},
            {"token_id": 3, "frame_id": 310},
        ]
        ranks = _frame_rank_map(rows)
        self.assertEqual(ranks[2], 0)
        self.assertEqual(ranks[1], 1)
        self.assertEqual(ranks[3], 2)


if __name__ == "__main__":
    unittest.main()
