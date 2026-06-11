from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from stream4d_native.chunk_alignment import (
    build_checkpoint_chunk_policy,
    make_sliding_window_clip_ranges,
    read_checkpoint_clip_frames,
)
from stream4d_native.opend4rt_long_video import make_anchor_clip_indices, validate_anchor_clip_indices
from stream4d_native.self_stitch import match_overlap_carriers, residual_diagnostics
from stream4d_native.sim3 import (
    Sim3Transform,
    apply_sim3_to_xyz,
    compose_sim3,
    estimate_overlap_sim3,
    invert_sim3,
)


class NativeChunkingAndSim3Test(unittest.TestCase):
    @staticmethod
    def _repo_path(relative: str) -> str:
        return str(Path(__file__).resolve().parents[2] / relative)

    def test_chunk_size_from_checkpoint_32clip(self) -> None:
        path = self._repo_path("Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
        value = read_checkpoint_clip_frames(path)
        self.assertEqual(value, 32)
        policy = build_checkpoint_chunk_policy(path)
        self.assertEqual(policy.temporal_chunk_size, 32)
        self.assertEqual(policy.temporal_chunk_stride, 16)
        self.assertEqual(policy.temporal_chunk_overlap, 16)

    def test_chunk_size_from_checkpoint_48clip(self) -> None:
        value = read_checkpoint_clip_frames(self._repo_path("Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml"))
        self.assertEqual(value, 48)

    def test_window_never_exceeds_checkpoint(self) -> None:
        ranges = make_sliding_window_clip_ranges(num_frames=77, clip_frames=32)
        self.assertTrue(all(end - start <= 32 for start, end in ranges))
        self.assertEqual(ranges[-1], (45, 77))

    def test_sliding_window_has_overlap(self) -> None:
        ranges = make_sliding_window_clip_ranges(num_frames=80, clip_frames=32)
        overlaps = [min(a[1], b[1]) - max(a[0], b[0]) for a, b in zip(ranges, ranges[1:])]
        self.assertTrue(all(value > 0 for value in overlaps))
        with self.assertRaises(ValueError):
            make_sliding_window_clip_ranges(num_frames=80, clip_frames=32, stride=32)

    def test_anchor_clip_contains_source_and_target(self) -> None:
        for target in (0, 1, 15, 31, 60):
            indices = make_anchor_clip_indices(num_frames=64, clip_frames=32, target_idx=target, source_idx=7)
            validate_anchor_clip_indices(indices, source_idx=7, target_idx=target, clip_frames=32)

    def test_estimate_overlap_sim3_recovers_known_transform(self) -> None:
        rng = np.random.default_rng(7)
        curr = rng.normal(size=(12, 5, 3)).astype(np.float32)
        scale = 1.7
        rot = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        trans = np.asarray([0.25, -0.5, 0.75], dtype=np.float64)
        prev = apply_sim3_to_xyz(curr, scale=scale, rot=rot, trans=trans)
        vis = np.ones(curr.shape[:2], dtype=bool)
        conf = np.ones(curr.shape[:2], dtype=np.float32)
        fit = estimate_overlap_sim3(prev, curr, vis, vis, conf, conf, min_points=16)
        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertAlmostEqual(float(fit["scale"]), scale, places=5)
        self.assertTrue(np.allclose(fit["rot"], rot, atol=1e-5))
        self.assertTrue(np.allclose(fit["trans"], trans, atol=1e-5))
        self.assertIn("inlier_ratio_abs010", fit)
        self.assertIn("residual_mad", fit)
        self.assertAlmostEqual(float(fit["inlier_ratio"]), float(fit["inlier_ratio_abs010"]))

    def test_residual_diagnostics_are_not_quantile_defined(self) -> None:
        residual = np.asarray([0.01, 0.02, 0.20, 0.30], dtype=np.float64)
        diag = residual_diagnostics(residual, scene_scale=10.0)
        self.assertAlmostEqual(float(diag["inlier_ratio_abs010"]), 0.5)
        self.assertAlmostEqual(float(diag["inlier_ratio_rel001"]), 0.5)
        self.assertLess(float(diag["inlier_ratio_abs010"]), 0.9)

    def test_overlap_matching_prefers_persistent_tube_id(self) -> None:
        prev = {
            "frame_ids": [10],
            "xyz": np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32),
            "uv": np.asarray([[[0.25, 0.25]]], dtype=np.float32),
            "valid": np.ones((1, 1), dtype=bool),
            "visibility": np.ones((1, 1), dtype=np.float32),
            "confidence": np.ones((1, 1), dtype=np.float32),
            "carrier_id": np.asarray([1], dtype=np.int64),
            "persistent_tube_id": np.asarray([42], dtype=np.int64),
            "src_frame_global": np.asarray([10], dtype=np.int64),
            "src_xy": np.asarray([[2, 2]], dtype=np.int64),
        }
        curr = {
            "frame_ids": [10],
            "xyz": np.asarray([[[2.0, 0.0, 0.0]]], dtype=np.float32),
            "uv": np.asarray([[[0.75, 0.75]]], dtype=np.float32),
            "valid": np.ones((1, 1), dtype=bool),
            "visibility": np.ones((1, 1), dtype=np.float32),
            "confidence": np.ones((1, 1), dtype=np.float32),
            "carrier_id": np.asarray([999], dtype=np.int64),
            "persistent_tube_id": np.asarray([42], dtype=np.int64),
            "src_frame_global": np.asarray([10], dtype=np.int64),
            "src_xy": np.asarray([[7, 7]], dtype=np.int64),
        }
        match = match_overlap_carriers(prev, curr, uv_radius=0.001)
        self.assertEqual(match.prev_xyz.shape[0], 1)
        self.assertEqual(match.stats["match_source_stable_id_count"], 1)
        self.assertEqual(match.stats["match_source_mutual_uv_count"], 0)

    def test_estimate_overlap_sim3_rejects_low_inliers(self) -> None:
        prev = np.zeros((2, 2, 3), dtype=np.float32)
        curr = np.ones((2, 2, 3), dtype=np.float32)
        vis = np.ones((2, 2), dtype=bool)
        fit = estimate_overlap_sim3(prev, curr, vis, vis, min_points=16)
        self.assertIsNone(fit)

    def test_apply_sim3_to_xyz_batch_shapes(self) -> None:
        xyz = np.zeros((2, 3, 4, 3), dtype=np.float32)
        out = apply_sim3_to_xyz(xyz, scale=2.0, rot=np.eye(3), trans=np.asarray([1.0, 2.0, 3.0]))
        self.assertEqual(out.shape, xyz.shape)
        self.assertTrue(np.allclose(out[..., 0], 1.0))

    def test_compose_and_invert_sim3_roundtrip(self) -> None:
        xyz = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]], dtype=np.float32)
        t = Sim3Transform(scale=2.5, rot=np.eye(3), trans=np.asarray([3.0, -2.0, 1.0]))
        inv = invert_sim3(t)
        identity = compose_sim3(t, inv)
        out = apply_sim3_to_xyz(xyz, transform=identity)
        self.assertTrue(np.allclose(out, xyz, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
