from __future__ import annotations

import unittest

from stream4d_native.merge_geometry_guard import GeometryRead, metric_merge_allowed


class V45NoCrossChunkLocalMergeTest(unittest.TestCase):
    def test_local_cross_chunk_metric_read_is_blocked(self) -> None:
        result = metric_merge_allowed(GeometryRead(source_frame="chunk_local", target_frame="method_canonical"))
        self.assertFalse(result["allowed"])
        self.assertTrue(result["cross_chunk_local_metric_read"])

    def test_eval_aligned_geometry_is_blocked_in_method(self) -> None:
        result = metric_merge_allowed(GeometryRead(source_frame="method_canonical", target_frame="eval_aligned_gt"))
        self.assertFalse(result["allowed"])
        self.assertTrue(result["cross_chunk_eval_aligned_read"])


if __name__ == "__main__":
    unittest.main()

