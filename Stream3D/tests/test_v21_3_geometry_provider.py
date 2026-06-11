from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider
from tools.native_geometry_diagnostics import _matched_overlap


class V213GeometryProviderTest(unittest.TestCase):
    def test_carrier_provider_uses_src_frame_global_for_frame_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = "scene_test"
            scene_dir = root / scene
            scene_dir.mkdir(parents=True)
            xyz = np.asarray(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
                ],
                dtype=np.float32,
            )
            uv = np.asarray(
                [
                    [[0.0, 0.0], [1 / 3, 1 / 3], [2 / 3, 1 / 3], [1.0, 1.0]],
                    [[0.0, 0.0], [1 / 3, 1 / 3], [2 / 3, 1 / 3], [1.0, 1.0]],
                ],
                dtype=np.float32,
            )
            np.savez_compressed(
                scene_dir / "carriers_window000.npz",
                carrier_id=np.arange(4, dtype=np.int64),
                src_frame=np.asarray([0, 0, 1, 1], dtype=np.int64),
                src_frame_global=np.asarray([0, 0, 10, 10], dtype=np.int64),
                src_xy=np.zeros((4, 2), dtype=np.int64),
                src_mask_id=np.ones((4,), dtype=np.int64),
                xyz_ref=xyz,
                uv_pred=uv,
                visibility_prob=np.ones((2, 4), dtype=np.float32),
                confidence_prob=np.ones((2, 4), dtype=np.float32),
                valid=np.ones((2, 4), dtype=bool),
            )
            mask = np.zeros((4, 4), dtype=np.int64)
            mask[1, 1] = 7
            mask[1, 2] = 9
            provider = D4RTCarrierProjectionProvider(debug_root=root, mode="raw", nn_radius=0.02)
            out = provider.project_frame_masks(
                dataset=SimpleNamespace(seq_name=scene),
                scene_points=xyz[1],
                mask_image=mask,
                frame_id=10,
                depth_max_pre=0.0,
            )
            self.assertEqual(set(out.mask_info), {7, 9})
            self.assertEqual(out.mask_info[7], {1})
            self.assertEqual(out.mask_info[9], {2})
            self.assertEqual(out.diagnostics["source_windows"], 1)

    def test_carrier_provider_can_filter_mask_boundary_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = "scene_test"
            scene_dir = root / scene
            scene_dir.mkdir(parents=True)
            xyz = np.asarray(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ]
                ],
                dtype=np.float32,
            )
            uv = np.asarray(
                [
                    [
                        [1 / 4, 1 / 4],
                        [2 / 4, 2 / 4],
                    ]
                ],
                dtype=np.float32,
            )
            np.savez_compressed(
                scene_dir / "carriers_window000.npz",
                carrier_id=np.arange(2, dtype=np.int64),
                src_frame=np.asarray([0, 0], dtype=np.int64),
                src_frame_global=np.asarray([0, 0], dtype=np.int64),
                src_xy=np.zeros((2, 2), dtype=np.int64),
                src_mask_id=np.ones((2,), dtype=np.int64),
                xyz_ref=xyz,
                uv_pred=uv,
                visibility_prob=np.ones((1, 2), dtype=np.float32),
                confidence_prob=np.ones((1, 2), dtype=np.float32),
                valid=np.ones((1, 2), dtype=bool),
            )
            mask = np.zeros((5, 5), dtype=np.int64)
            mask[1:4, 1:4] = 7
            provider = D4RTCarrierProjectionProvider(
                debug_root=root,
                mode="raw",
                nn_radius=0.02,
                min_mask_interior_px=1.5,
            )
            out = provider.project_frame_masks(
                dataset=SimpleNamespace(seq_name=scene),
                scene_points=xyz[0],
                mask_image=mask,
                frame_id=0,
                depth_max_pre=0.0,
            )
            self.assertEqual(out.mask_info[7], {1})
            self.assertEqual(out.diagnostics["interior_filtered_point_count"], 1)
            self.assertEqual(out.diagnostics["min_mask_interior_px"], 1.5)

    def test_carrier_provider_best_confidence_overlap_policy_selects_one_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = "scene_test"
            scene_dir = root / scene
            scene_dir.mkdir(parents=True)
            mask = np.zeros((5, 5), dtype=np.int64)
            mask[1:4, 1:4] = 7
            uv = np.asarray([[[1 / 4, 1 / 4], [2 / 4, 2 / 4]]], dtype=np.float32)
            xyz0 = np.asarray([[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]], dtype=np.float32)
            xyz1 = np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
            for idx, (xyz, confidence) in enumerate(((xyz0, 0.6), (xyz1, 0.9))):
                np.savez_compressed(
                    scene_dir / f"carriers_window{idx:03d}.npz",
                    carrier_id=np.arange(2, dtype=np.int64),
                    src_frame=np.asarray([0, 0], dtype=np.int64),
                    src_frame_global=np.asarray([10, 10], dtype=np.int64),
                    src_xy=np.zeros((2, 2), dtype=np.int64),
                    src_mask_id=np.ones((2,), dtype=np.int64),
                    xyz_ref=xyz,
                    uv_pred=uv,
                    visibility_prob=np.ones((1, 2), dtype=np.float32),
                    confidence_prob=np.full((1, 2), confidence, dtype=np.float32),
                    valid=np.ones((1, 2), dtype=bool),
                )
            provider = D4RTCarrierProjectionProvider(
                debug_root=root,
                mode="raw",
                nn_radius=0.02,
                overlap_policy="best_confidence",
            )
            out = provider.project_frame_masks(
                dataset=SimpleNamespace(seq_name=scene),
                scene_points=xyz1[0],
                mask_image=mask,
                frame_id=10,
                depth_max_pre=0.0,
            )
            self.assertEqual(out.diagnostics["candidate_source_windows"], 2)
            self.assertEqual(out.diagnostics["source_windows"], 1)
            self.assertEqual(out.diagnostics["overlap_policy"], "best_confidence")
            self.assertEqual(out.diagnostics["duplicate_window_hit_rate"], 0.5)
            self.assertEqual(out.mask_info[7], {0, 1})

            provider_all = D4RTCarrierProjectionProvider(
                debug_root=root,
                mode="raw",
                nn_radius=0.02,
                overlap_policy="all_window_union",
            )
            out_all = provider_all.project_frame_masks(
                dataset=SimpleNamespace(seq_name=scene),
                scene_points=xyz1[0],
                mask_image=mask,
                frame_id=10,
                depth_max_pre=0.0,
            )
            self.assertEqual(out_all.diagnostics["source_windows"], 2)
            self.assertEqual(out_all.diagnostics["duplicate_window_hit_rate"], 0.0)

    def test_self_stitch_transform_maps_later_window_to_canonical_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = "scene_test"
            scene_dir = root / scene
            scene_dir.mkdir(parents=True)
            canonical = np.asarray(
                [
                    [10.0, 0.0, 0.0],
                    [12.0, 0.0, 0.0],
                    [10.0, 2.0, 0.0],
                    [10.0, 0.0, 2.0],
                ],
                dtype=np.float32,
            )
            translation = np.asarray([10.0, 0.0, 0.0], dtype=np.float32)
            local = (canonical - translation[None, :]) / 2.0
            uv = np.tile(
                np.asarray([[[0.25, 0.25], [0.50, 0.25], [0.25, 0.50], [0.50, 0.50]]], dtype=np.float32),
                (2, 1, 1),
            )
            np.savez_compressed(
                scene_dir / "carriers_window000.npz",
                carrier_id=np.arange(4, dtype=np.int64),
                src_frame=np.zeros((4,), dtype=np.int64),
                src_frame_global=np.zeros((4,), dtype=np.int64),
                xyz_ref=np.stack([canonical, canonical], axis=0),
                uv_pred=uv,
                visibility_prob=np.ones((2, 4), dtype=np.float32),
                confidence_prob=np.ones((2, 4), dtype=np.float32),
                valid=np.ones((2, 4), dtype=bool),
            )
            np.savez_compressed(
                scene_dir / "carriers_window001.npz",
                carrier_id=np.arange(4, dtype=np.int64),
                src_frame=np.zeros((4,), dtype=np.int64),
                src_frame_global=np.full((4,), 10, dtype=np.int64),
                xyz_ref=np.stack([local, local], axis=0),
                uv_pred=uv,
                visibility_prob=np.ones((2, 4), dtype=np.float32),
                confidence_prob=np.ones((2, 4), dtype=np.float32),
                valid=np.ones((2, 4), dtype=bool),
            )
            (scene_dir / "carriers_window000_manifest.json").write_text(
                '{"frame_ids": [0, 10]}',
                encoding="utf-8",
            )
            (scene_dir / "carriers_window001_manifest.json").write_text(
                '{"frame_ids": [10, 20]}',
                encoding="utf-8",
            )
            mask = np.zeros((5, 5), dtype=np.int64)
            mask[1:4, 1:4] = 7
            provider = D4RTCarrierProjectionProvider(
                debug_root=root,
                mode="self_stitched",
                nn_radius=0.02,
            )
            out = provider.project_frame_masks(
                dataset=SimpleNamespace(seq_name=scene),
                scene_points=canonical,
                mask_image=mask,
                frame_id=20,
                depth_max_pre=0.0,
            )
            self.assertEqual(out.diagnostics["self_stitch_pair_count"], 1)
            self.assertEqual(out.diagnostics["self_stitch_fail_count"], 0)
            self.assertIn("self_stitch_match_source_stable_id_count", out.diagnostics)
            self.assertEqual(out.mask_info[7], {0, 1, 2, 3})

    def test_self_stitch_three_window_nonidentity_sim3_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = "scene_test"
            scene_dir = root / scene
            scene_dir.mkdir(parents=True)
            canonical = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            rot1 = np.asarray(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            rot2 = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            )

            def invert_points(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
                return (((points.astype(np.float64) - trans[None, :]) / float(scale)) @ rot).astype(np.float32)

            local1 = invert_points(canonical, 1.5, rot1, np.asarray([0.5, -0.25, 0.75], dtype=np.float64))
            local2 = invert_points(canonical, 0.75, rot2, np.asarray([-0.5, 0.25, -0.75], dtype=np.float64))
            uv = np.tile(
                np.asarray([[[0.20, 0.20], [0.40, 0.20], [0.20, 0.40], [0.40, 0.40], [0.60, 0.60]]], dtype=np.float32),
                (2, 1, 1),
            )
            for idx, (frames, xyz) in enumerate(
                (
                    ([0, 10], np.stack([canonical, canonical], axis=0)),
                    ([10, 20], np.stack([local1, local1], axis=0)),
                    ([20, 30], np.stack([local2, local2], axis=0)),
                )
            ):
                np.savez_compressed(
                    scene_dir / f"carriers_window{idx:03d}.npz",
                    carrier_id=np.arange(canonical.shape[0], dtype=np.int64),
                    persistent_tube_id=np.arange(canonical.shape[0], dtype=np.int64),
                    src_frame=np.zeros((canonical.shape[0],), dtype=np.int64),
                    src_frame_global=np.full((canonical.shape[0],), frames[0], dtype=np.int64),
                    src_xy=np.stack([np.arange(canonical.shape[0]), np.arange(canonical.shape[0])], axis=1),
                    xyz_ref=xyz,
                    uv_pred=uv,
                    visibility_prob=np.ones((2, canonical.shape[0]), dtype=np.float32),
                    confidence_prob=np.ones((2, canonical.shape[0]), dtype=np.float32),
                    valid=np.ones((2, canonical.shape[0]), dtype=bool),
                )
                (scene_dir / f"carriers_window{idx:03d}_manifest.json").write_text(
                    json.dumps({"frame_ids": frames}),
                    encoding="utf-8",
                )
            mask = np.zeros((5, 5), dtype=np.int64)
            mask[1:4, 1:4] = 7
            provider = D4RTCarrierProjectionProvider(debug_root=root, mode="self_stitched", nn_radius=0.03)
            out = provider.project_frame_masks(
                dataset=SimpleNamespace(seq_name=scene),
                scene_points=canonical,
                mask_image=mask,
                frame_id=30,
                depth_max_pre=0.0,
            )
            self.assertEqual(out.diagnostics["self_stitch_pair_count"], 2)
            self.assertEqual(out.diagnostics["self_stitch_fail_count"], 0)
            self.assertGreaterEqual(out.diagnostics["self_stitch_match_source_stable_id_count"], 10)
            self.assertEqual(out.mask_info[7], {0, 1, 2, 3, 4})

    def test_phase_c_overlap_matching_uses_global_frame_ids(self) -> None:
        prev = {
            "uv_pred": np.asarray([[[0.2, 0.2]], [[0.4, 0.4]]], dtype=np.float32),
            "xyz_ref": np.asarray([[[99.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=np.float32),
            "valid": np.ones((2, 1), dtype=bool),
            "visibility": np.ones((2, 1), dtype=np.float32),
            "confidence": np.ones((2, 1), dtype=np.float32),
        }
        curr = {
            "uv_pred": np.asarray([[[0.4, 0.4]], [[0.9, 0.9]]], dtype=np.float32),
            "xyz_ref": np.asarray([[[2.0, 0.0, 0.0]], [[100.0, 0.0, 0.0]]], dtype=np.float32),
            "valid": np.ones((2, 1), dtype=bool),
            "visibility": np.ones((2, 1), dtype=np.float32),
            "confidence": np.ones((2, 1), dtype=np.float32),
        }
        prev_xyz, curr_xyz, overlap_count = _matched_overlap(
            prev,
            curr,
            prev_frame_ids=[0, 10],
            curr_frame_ids=[10, 20],
            uv_radius=0.01,
        )
        self.assertEqual(overlap_count, 1)
        self.assertEqual(prev_xyz.shape[0], 1)
        self.assertAlmostEqual(float(prev_xyz[0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(curr_xyz[0, 0, 0]), 2.0)


if __name__ == "__main__":
    unittest.main()
