from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.trace_v25_real_geometry_flow import load_scene_chunks_from_cache


class TraceV25RealGeometryFlowTests(unittest.TestCase):
    def test_load_scene_chunks_materializes_selected_tubes_from_single_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene0000_00"
            scene_dir.mkdir()
            uv_pred = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2) / 100.0
            xyz_ref = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)
            np.savez_compressed(
                scene_dir / "carriers_window000.npz",
                carrier_id=np.asarray([100, 101, 102, 103], dtype=np.int64),
                persistent_tube_id=np.asarray([200, 201, 202, 203], dtype=np.int64),
                src_frame=np.asarray([0, 0, 1, 1], dtype=np.int64),
                src_frame_global=np.asarray([10, 10, 11, 11], dtype=np.int64),
                src_uv=np.asarray([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]], dtype=np.float32),
                src_xy=np.asarray([[8, 9], [10, 11], [12, 13], [14, 15]], dtype=np.int64),
                uv_pred=uv_pred,
                xyz_ref=xyz_ref,
                xyz_local=xyz_ref + 1.0,
                visibility_prob=np.ones((3, 4), dtype=np.float32),
                confidence_prob=np.full((3, 4), 0.5, dtype=np.float32),
                valid=np.ones((3, 4), dtype=bool),
            )

            chunks, diagnostics = load_scene_chunks_from_cache(
                scene_dir,
                max_tubes_per_window=2,
                image_width=640,
                image_height=480,
                prefer_source_pixel_id=False,
            )

        self.assertEqual(diagnostics["tube_count"], 2)
        self.assertEqual(len(chunks), 1)
        tubes = chunks[0]["tubes"]
        self.assertEqual([tube["carrier_id"] for tube in tubes], [100, 102])
        self.assertEqual([tube["persistent_tube_id"] for tube in tubes], [200, 202])
        self.assertEqual([tube["source_frame_global"] for tube in tubes], [10, 11])
        np.testing.assert_array_equal(tubes[0]["uv_norm"], uv_pred[:, 0, :])
        np.testing.assert_array_equal(tubes[1]["xyz_local"], xyz_ref[:, 2, :] + 1.0)
        self.assertFalse(tubes[0]["source_identity_from_fallback"])


if __name__ == "__main__":
    unittest.main()
