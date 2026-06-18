from __future__ import annotations

import unittest

from stream4d_native.frame_index_map import FrameIndexMap


class V41FrameStrideTests(unittest.TestCase):
    def test_sparse_masks_use_observation_rank_while_d4rt_rgb_is_contiguous(self) -> None:
        fmap = FrameIndexMap.from_frame_ids(range(31), [0, 10, 20, 30])
        summary = fmap.audit_summary()
        self.assertEqual(summary["d4rt_encoder_stride"], 1)
        self.assertEqual(summary["mask_observation_stride"], [10])
        self.assertEqual(summary["rank_delta_distribution"], {"1": 3})
        self.assertTrue(summary["uses_contiguous_rgb_for_d4rt"])
        self.assertTrue(summary["uses_sparse_masks_as_measurements"])
        self.assertEqual(fmap.temporal_rank_delta(0, 10), 1)
        self.assertNotEqual(fmap.temporal_rank_delta(0, 10), 10)

    def test_detects_noncontiguous_rgb_loader(self) -> None:
        fmap = FrameIndexMap.from_frame_ids([0, 10, 20], [0, 10, 20])
        summary = fmap.audit_summary()
        self.assertEqual(summary["d4rt_encoder_stride"], "non_contiguous")
        self.assertFalse(summary["uses_contiguous_rgb_for_d4rt"])


if __name__ == "__main__":
    unittest.main()

