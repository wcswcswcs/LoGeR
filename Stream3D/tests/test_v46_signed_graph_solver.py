from __future__ import annotations

import unittest

from stream4d_native.v46_signed_mask_graph import solve_edges


class V46SignedGraphSolverTest(unittest.TestCase):
    def test_hard_negative_veto_blocks_merge(self) -> None:
        token_rows = [{"token_id": "1"}, {"token_id": "2"}]
        labels, trace = solve_edges(
            token_rows,
            [
                {
                    "token_i": 1,
                    "token_j": 2,
                    "positive_weight": 0.95,
                    "negative_weight": 0.90,
                    "view_consensus_proxy": 0.95,
                    "edge_type": "same_frame+visible_outside",
                }
            ],
            solver="S3_greedy_signed_hard_veto",
        )
        self.assertNotEqual(labels[1], labels[2])
        self.assertEqual(trace, [])


if __name__ == "__main__":
    unittest.main()
