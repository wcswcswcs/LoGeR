from __future__ import annotations

import unittest

import numpy as np

from stream4d_native.sim3 import apply_sim3_to_xyz
from tools.run_v42_calibrated_native_ap_bridge import _fit_robust_sim3, _parse_radii


class V42CalibratedNativeApBridgeTests(unittest.TestCase):
    def test_parse_radii(self) -> None:
        self.assertEqual(_parse_radii("0.02, 0.1,,0.5"), [0.02, 0.1, 0.5])

    def test_fit_robust_sim3_trims_outliers(self) -> None:
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        rot = np.eye(3, dtype=np.float64)
        trans = np.asarray([0.5, -1.0, 2.0], dtype=np.float64)
        target = 1.75 * source + trans
        target[-2:] += np.asarray([[8.0, -4.0, 2.0], [-6.0, 5.0, -3.0]], dtype=np.float64)

        fit = _fit_robust_sim3(source, target, trim_quantile=0.80, min_anchors=4)
        pred = apply_sim3_to_xyz(source[:8], transform=fit).astype(np.float64)

        self.assertLess(float(np.mean(np.linalg.norm(pred - (1.75 * source[:8] + trans), axis=1))), 1e-5)
        self.assertEqual(fit["kept_anchor_count"], 8)
        self.assertAlmostEqual(fit["scale"], 1.75, places=5)
        self.assertAlmostEqual(fit["rotation_det"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
