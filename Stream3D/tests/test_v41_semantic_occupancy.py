from __future__ import annotations

import unittest

import numpy as np

from stream4d_native.semantic_occupancy import run_semantic_occupancy_variants


class V41SemanticOccupancyTests(unittest.TestCase):
    def test_semantic_occupancy_improves_boundary_overlap_and_keeps_exploration(self) -> None:
        masks = np.zeros((4, 12, 12), dtype=np.int32)
        masks[:, 3:9, 3:9] = 1
        disagreement = np.zeros_like(masks, dtype=bool)
        disagreement[:, 5:7, 5:7] = True
        rows = run_semantic_occupancy_variants(
            masks,
            budget=40,
            overlap_frame_ranks=[1, 2],
            disagreement=disagreement,
        )
        by_variant = {row["variant"]: row for row in rows}
        b0 = by_variant["B0"]
        b5 = by_variant["B5"]
        b6 = by_variant["B6"]
        self.assertGreaterEqual(b5["mask_boundary_coverage"], b0["mask_boundary_coverage"] * 1.20)
        self.assertGreaterEqual(b5["overlap_anchor_coverage"], b0["overlap_anchor_coverage"] * 1.15)
        self.assertGreaterEqual(b6["exploration_outside_mask_ratio"], 0.10)
        self.assertGreater(b5["accepted_tube_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()

