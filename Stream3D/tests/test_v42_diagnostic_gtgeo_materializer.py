from __future__ import annotations

import unittest

import numpy as np

from tools.run_v42_diagnostic_gtgeo_materializer import _mask_xy


class V42DiagnosticGtGeoMaterializerTests(unittest.TestCase):
    def test_mask_xy_resizes_and_samples_pixels(self) -> None:
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        xy = _mask_xy(mask, depth_shape=(4, 4), stride=1, max_pixels=0)
        self.assertGreaterEqual(xy.shape[0], 1)
        self.assertEqual(xy.shape[1], 2)
        self.assertTrue(np.all(xy[:, 0] >= 0))
        self.assertTrue(np.all(xy[:, 0] < 4))
        self.assertTrue(np.all(xy[:, 1] >= 0))
        self.assertTrue(np.all(xy[:, 1] < 4))

    def test_mask_xy_honors_max_pixels(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        xy = _mask_xy(mask, depth_shape=(10, 10), stride=1, max_pixels=7)
        self.assertEqual(xy.shape, (7, 2))


if __name__ == "__main__":
    unittest.main()
