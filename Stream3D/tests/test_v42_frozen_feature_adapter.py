from __future__ import annotations

import unittest

import numpy as np

from stream4d_native.frozen_feature_adapter import FrozenFeatureAdapter


class V42FrozenFeatureAdapterTests(unittest.TestCase):
    def test_rgb_stats_pooling_and_affinity(self) -> None:
        frame = np.zeros((10, 12, 3), dtype=np.uint8)
        frame[:, :6, 0] = 255
        frame[:, 6:, 1] = 255
        left = np.zeros((10, 12), dtype=bool)
        left[:, :6] = True
        right = np.zeros((10, 12), dtype=bool)
        right[:, 6:] = True
        adapter = FrozenFeatureAdapter(backend="rgb_stats")
        fmap = adapter.extract_dense_features(frame)
        left_feature = adapter.pool_mask_feature(fmap, left)
        right_feature = adapter.pool_mask_feature(fmap, right)
        self.assertEqual(fmap.features.shape[:2], frame.shape[:2])
        self.assertEqual(left_feature.ndim, 1)
        self.assertLess(adapter.compute_token_affinity(left_feature, right_feature), 0.95)
        self.assertGreaterEqual(adapter.compute_boundary_contrast(fmap, left), 0.0)


if __name__ == "__main__":
    unittest.main()

