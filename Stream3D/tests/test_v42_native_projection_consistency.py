from __future__ import annotations

import unittest

import numpy as np

from tools.run_v42_native_projection_consistency import projection_consistency_stats


class V42NativeProjectionConsistencyTests(unittest.TestCase):
    def test_projection_consistency_gate_passes_for_camera_like_xyz(self) -> None:
        intrinsics = np.asarray(
            [
                [100.0, 0.0, 50.0],
                [0.0, 100.0, 50.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        xyz = np.asarray([[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]]], dtype=np.float32)
        uv = np.asarray([[[50.0 / 99.0, 50.0 / 99.0], [60.0 / 99.0, 50.0 / 99.0], [50.0 / 99.0, 60.0 / 99.0], [60.0 / 99.0, 60.0 / 99.0]]], dtype=np.float32)
        stats = projection_consistency_stats(xyz, uv, np.ones((1, 4), dtype=bool), intrinsics, width=100, height=100)
        self.assertTrue(stats["projection_consistency_gate"])
        self.assertLess(stats["projection_error_p90"], 1e-6)

    def test_projection_consistency_gate_fails_for_offset_uv(self) -> None:
        intrinsics = np.asarray(
            [
                [100.0, 0.0, 50.0],
                [0.0, 100.0, 50.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        xyz = np.asarray([[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.1, 0.1, 1.0]]], dtype=np.float32)
        uv = np.full((1, 4, 2), 0.0, dtype=np.float32)
        stats = projection_consistency_stats(xyz, uv, np.ones((1, 4), dtype=bool), intrinsics, width=100, height=100)
        self.assertFalse(stats["projection_consistency_gate"])
        self.assertGreater(stats["projection_error_p90"], 0.02)


if __name__ == "__main__":
    unittest.main()
