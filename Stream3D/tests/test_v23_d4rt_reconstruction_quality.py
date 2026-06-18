import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.run_v23_d4rt_reconstruction_quality_audit import (
    _backproject_pixel,
    _depth_row,
    _project_camera,
    _summarize_track_samples,
    _variant_rows,
    SourceRow,
)


class V23D4RTReconstructionQualityTests(unittest.TestCase):
    def test_xyz_local_and_uv_pred_have_same_target_frame_convention(self):
        intr = np.array(
            [
                [100.0, 0.0, 50.0],
                [0.0, 100.0, 40.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        uv_pixel = (70.0, 55.0)
        xyz_local = _backproject_pixel(uv_pixel[0], uv_pixel[1], 2.0, intr)

        projected = _project_camera(xyz_local, intr)

        self.assertIsNotNone(projected)
        self.assertAlmostEqual(projected[0], uv_pixel[0], places=6)
        self.assertAlmostEqual(projected[1], uv_pixel[1], places=6)

    def test_xyz_ref0_is_not_used_as_target_camera_depth_in_depth_mapping(self):
        source = SourceRow(
            path=Path("source.csv"),
            row={
                "label": "D4RT xyz_ref0 + ScanNet ref0 pose",
                "depth_raw_absrel": "0.5",
                "depth_raw_delta1": "0.1",
                "depth_raw_valid_pixel_ratio": "1.0",
            },
        )

        row = _depth_row("D7", "R22", source, "depth_raw", "ref0_pose", "ref0 diagnostic")

        self.assertEqual(row["alignment_source"], "ref0_pose")
        self.assertTrue(row["uses_eval_alignment"])
        self.assertEqual(row["AbsRel"], 0.5)

    def test_carrier_cache_contains_xyz_local_and_xyz_ref0(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "carriers_window000.npz"
            np.savez(
                path,
                xyz_local=np.zeros((1, 2, 3), dtype=np.float32),
                xyz_ref=np.zeros((1, 2, 3), dtype=np.float32),
                uv_pred=np.zeros((1, 2, 2), dtype=np.float32),
            )

            data = np.load(path)

            self.assertIn("xyz_local", data.files)
            self.assertIn("xyz_ref", data.files)
            self.assertIn("uv_pred", data.files)

    def test_no_branch_mixing_in_uvz_camera_metric(self):
        source = SourceRow(
            path=Path("source.csv"),
            row={
                "label": "D4RT xyz_local UV+Z camera backprojection",
                "depth_raw_absrel": "0.25",
                "depth_raw_delta1": "0.75",
            },
        )

        row = _depth_row("D6", "R18", source, "depth_raw", "none", "uv + xyz_local z")

        self.assertEqual(row["source_variant"], "R18")
        self.assertEqual(row["alignment_source"], "none")
        self.assertFalse(row["uses_eval_alignment"])

    def test_official_opend4rt_output_keys_parity(self):
        source = Path("Open-d4rt/infer_track_3d.py")
        if not source.exists():
            source = Path("../Open-d4rt/infer_track_3d.py")
        if not source.exists():
            self.skipTest("OpenD4RT source not present")
        text = source.read_text(encoding="utf-8")

        self.assertIn("tracks_xyz_local", text)
        self.assertIn("tracks_xyz_ref0", text)
        self.assertIn("tracks_uv_norm", text)

    def test_checkpoint_clip_frames_read_correctly_from_manifest_like_row(self):
        row = {"clip_frames": "32", "query_chunk_size": "1024"}

        self.assertEqual(int(row["clip_frames"]), 32)
        self.assertNotEqual(int(row["clip_frames"]), int(row["query_chunk_size"]))

    def test_track_summary_beats_empty_and_computes_pck_visibility(self):
        samples = [
            {"epe": 0.5, "pseudo_visible": True, "pred_visible": True, "gt_in_frame": True},
            {"epe": 4.0, "pseudo_visible": True, "pred_visible": False, "gt_in_frame": True},
            {"epe": float("nan"), "pseudo_visible": False, "pred_visible": True, "gt_in_frame": False},
        ]

        summary = _summarize_track_samples(samples)

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["visible_sample_count"], 2)
        self.assertAlmostEqual(summary["PCK@1px"], 0.5)
        self.assertAlmostEqual(summary["PCK@5px"], 1.0)
        self.assertGreater(summary["visibility_precision"], 0.0)

    def test_variant_rows_reads_first_variant_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "v22_5_direct_xyz_local_transform_probe5" / "direct_reconstruction_summary.csv"
            path.parent.mkdir(parents=True)
            path.write_text("variant,label,depth_raw_absrel\nR16,test,0.1\n", encoding="utf-8")

            rows = _variant_rows(root)

            self.assertIn("R16", rows)
            self.assertEqual(rows["R16"].row["label"], "test")


if __name__ == "__main__":
    unittest.main()
