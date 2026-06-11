from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stream4d.carrier_store import CarrierBatch
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
from stream4d_native.occupancy_dense_tracker import QueryBudget, query_d4rt_tubes_with_spatiotemporal_occupancy
from stream4d_native.occupancy_state import OccupancyCoverageTargets, SpatioTemporalOccupancyState
from tools.export_v21_3_occupancy_carrier_cache import _assign_persistent_tube_ids, _export_d2, _tracks_to_batch


class FakeBatch:
    def __init__(self, t: int, q: int, uv: np.ndarray) -> None:
        self.uv_pred = np.tile(uv[None, :, :], (t, 1, 1)).astype(np.float32)
        self.xyz_ref = np.zeros((t, q, 3), dtype=np.float32)
        self.visibility_prob = np.ones((t, q), dtype=np.float32)
        self.confidence_prob = np.ones((t, q), dtype=np.float32)
        self.valid = np.ones((t, q), dtype=bool)


class FakeD4RTModel:
    def infer_carriers(self, *, video_rgb_uint8, src_uv_norm, src_frame_local, query_chunk_size):
        del src_frame_local, query_chunk_size
        return FakeBatch(int(video_rgb_uint8.shape[0]), int(src_uv_norm.shape[0]), np.asarray(src_uv_norm))


class FakeExportAdapter:
    def infer_carriers(
        self,
        *,
        video_rgb_uint8,
        src_uv_norm,
        src_frame_local,
        query_chunk_size,
        carrier_id,
        src_frame_global,
        src_xy,
        src_mask_id,
    ):
        del src_frame_local, query_chunk_size, src_frame_global, src_xy, src_mask_id
        frames = int(video_rgb_uint8.shape[0])
        queries = int(src_uv_norm.shape[0])
        uv = np.tile(np.asarray(src_uv_norm, dtype=np.float32)[None, :, :], (frames, 1, 1))
        xyz = np.zeros((frames, queries, 3), dtype=np.float32)
        return CarrierBatch(
            carrier_id=np.asarray(carrier_id, dtype=np.int64),
            src_frame=np.zeros((queries,), dtype=np.int64),
            src_uv=np.asarray(src_uv_norm, dtype=np.float32),
            xyz_ref=xyz,
            uv_pred=uv,
            visibility_prob=np.ones((frames, queries), dtype=np.float32),
            confidence_prob=np.ones((frames, queries), dtype=np.float32),
            valid=np.ones((frames, queries), dtype=bool),
        )


class FakeExportStream:
    def load_rgb(self, frame_id: int) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8) + int(frame_id)

    def load_mask(self, frame_id: int) -> np.ndarray:
        del frame_id
        mask = np.zeros((8, 8), dtype=np.int32)
        mask[2:6, 2:6] = 1
        return mask


class NativeOccupancyAndBuilderTest(unittest.TestCase):
    def test_occupancy_marks_all_target_frames(self) -> None:
        state = SpatioTemporalOccupancyState(num_frames=4, image_height=10, image_width=10)
        center = 5.0 / 9.0
        track = {
            "uv_norm": np.asarray([[center, center], [center, center], [center, center], [center, center]], dtype=np.float32),
            "visibility": np.ones((4,), dtype=np.float32),
            "confidence": np.ones((4,), dtype=np.float32),
            "valid": np.ones((4,), dtype=bool),
        }
        state.mark_visible_track_as_visited(track=track, tube_id=0, mark_radius_px=1)
        self.assertTrue(np.all(state.visited[:, 4:7, 4:7]))

    def test_occupancy_reduces_duplicate_queries(self) -> None:
        frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)

        def decode(source_points: np.ndarray) -> list[dict]:
            out = []
            for _, u, v in source_points:
                out.append(
                    {
                        "uv_norm": np.tile(np.asarray([[u, v]], dtype=np.float32), (3, 1)),
                        "visibility": np.ones((3,), dtype=np.float32),
                        "confidence": np.ones((3,), dtype=np.float32),
                        "valid": np.ones((3,), dtype=bool),
                    }
                )
            return out

        _, diag = query_d4rt_tubes_with_spatiotemporal_occupancy(
            frames=frames,
            masks=None,
            decode_source_points=decode,
            coverage_targets=OccupancyCoverageTargets(pixel_coverage_target=0.15, mark_radius_px=2),
            query_budget=QueryBudget(max_source_points=32, source_points_per_round=8),
        )
        self.assertTrue(diag["uses_spatiotemporal_occupancy"])
        self.assertGreater(diag["adaptive_speedup_vs_naive"], 1.0)
        self.assertLess(diag["actual_source_query_count"], diag["naive_source_query_count"])

    def test_occupancy_warmstart_marks_before_sampling(self) -> None:
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        center = 3.0 / 7.0
        warm_track = {
            "uv_norm": np.asarray([[center, center], [center, center]], dtype=np.float32),
            "visibility": np.ones((2,), dtype=np.float32),
            "confidence": np.ones((2,), dtype=np.float32),
            "valid": np.ones((2,), dtype=bool),
        }

        def decode(source_points: np.ndarray) -> list[dict]:
            return [
                {
                    "uv_norm": np.tile(np.asarray([[u, v]], dtype=np.float32), (2, 1)),
                    "visibility": np.ones((2,), dtype=np.float32),
                    "confidence": np.ones((2,), dtype=np.float32),
                    "valid": np.ones((2,), dtype=bool),
                }
                for _, u, v in source_points
            ]

        _, diag = query_d4rt_tubes_with_spatiotemporal_occupancy(
            frames=frames,
            masks=None,
            decode_source_points=decode,
            coverage_targets=OccupancyCoverageTargets(pixel_coverage_target=0.20, mark_radius_px=2),
            query_budget=QueryBudget(max_source_points=8, source_points_per_round=4),
            warmstart_tracks=[warm_track],
        )
        self.assertEqual(diag["warmstart_track_count"], 1)
        self.assertLessEqual(diag["actual_source_query_count"], 8)
        self.assertGreater(diag["pixel_occupancy_coverage_mean"], 0.0)

    def test_mask_aware_sampling_prefers_mask_pixels(self) -> None:
        masks = np.zeros((1, 8, 8), dtype=np.int32)
        masks[:, 2:6, 3:7] = 1
        state = SpatioTemporalOccupancyState(num_frames=1, image_height=8, image_width=8, masks=masks)
        points = state.sample_unvisited_source_points(
            batch_size=4,
            priority_order=["large_mask_interior_uncovered", "mask_boundary_uncovered"],
        )
        xs = np.rint(points[:, 1] * 7).astype(np.int64)
        ys = np.rint(points[:, 2] * 7).astype(np.int64)
        self.assertTrue(np.all(masks[0, ys, xs] == 1))

    def test_mask_coverage_summary_is_recorded(self) -> None:
        masks = np.zeros((2, 8, 8), dtype=np.int32)
        masks[:, 2:6, 2:6] = 1
        state = SpatioTemporalOccupancyState(num_frames=2, image_height=8, image_width=8, masks=masks)
        center = 3.0 / 7.0
        track = {
            "uv_norm": np.asarray([[center, center], [center, center]], dtype=np.float32),
            "visibility": np.ones((2,), dtype=np.float32),
            "confidence": np.ones((2,), dtype=np.float32),
            "valid": np.ones((2,), dtype=bool),
        }
        state.mark_visible_track_as_visited(track=track, tube_id=0, mark_radius_px=1)
        summary = state.summarize()
        self.assertIsNotNone(summary["mask_interior_coverage_mean"])
        self.assertIsNotNone(summary["mask_boundary_coverage_mean"])

    def test_occupancy_warmstart_uses_no_gt(self) -> None:
        import stream4d_native.occupancy_state as occupancy_state

        source = inspect.getsource(occupancy_state)
        forbidden = ("ScanNet", "load_depth", "load_pose", "vh_clean", "gt_instance")
        self.assertFalse(any(item in source for item in forbidden))

    def test_native_scene_builder_chunk_policy(self) -> None:
        frames = np.zeros((40, 8, 8, 3), dtype=np.uint8)
        builder = D4RTNativeSceneBuilder(FakeD4RTModel(), {"model": {"input": {"clip_frames": 16}}})
        chunks = builder.build_chunks(frames)
        self.assertTrue(all(chunk["num_frames"] <= 16 for chunk in chunks))
        self.assertGreater(builder.temporal_chunk_overlap, 0)

    def test_no_gt_geometry_in_native_scene_builder(self) -> None:
        import stream4d_native.d4rt_scene_builder as d4rt_scene_builder

        source = inspect.getsource(d4rt_scene_builder)
        forbidden = ("ScanNetStream", "load_depth", "load_pose", "mesh_path", "mask_backprojection", "gt_sim3")
        self.assertFalse(any(item in source for item in forbidden))

    def test_d2_export_uses_overlap_windows_with_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                output_root=str(Path(tmp) / "cache"),
                seq_name="scene_test",
                window_size=4,
                window_stride=2,
                mask_fixed_points_per_mask=4,
                mask_fixed_min_points_per_mask=2,
                query_chunk_size=16,
            )
            windows, row = _export_d2(
                stream=FakeExportStream(),
                frame_ids=[0, 1, 2, 3, 4, 5],
                adapter=FakeExportAdapter(),
                args=args,
                targets=OccupancyCoverageTargets(mark_radius_px=4),
            )
            self.assertEqual(row["num_windows"], 2)
            self.assertEqual(len(windows), 2)
            scene_dir = Path(args.output_root) / args.seq_name
            with np.load(scene_dir / "carriers_window000.npz") as first, np.load(
                scene_dir / "carriers_window001.npz"
            ) as second:
                shared = np.intersect1d(first["carrier_id"], second["carrier_id"])
            self.assertGreater(shared.shape[0], 0)

    def test_d5_identity_assignment_writes_persistent_fields(self) -> None:
        previous = [
            {
                "uv_norm": np.asarray([[0.2, 0.2], [0.4, 0.4]], dtype=np.float32),
                "valid": np.ones((2,), dtype=bool),
                "visibility": np.ones((2,), dtype=np.float32),
                "confidence": np.ones((2,), dtype=np.float32),
                "persistent_tube_id": 77,
                "carrier_id": 5,
                "window_index": 0,
            }
        ]
        current = [
            {
                "uv_norm": np.asarray([[0.4, 0.4], [0.6, 0.6]], dtype=np.float32),
                "valid": np.ones((2,), dtype=bool),
                "visibility": np.ones((2,), dtype=np.float32),
                "confidence": np.ones((2,), dtype=np.float32),
                "xyz": np.zeros((2, 3), dtype=np.float32),
                "carrier_id": 9,
                "source_frame": 0,
                "source_uv": np.asarray([0.4, 0.4], dtype=np.float32),
            }
        ]
        _, diag = _assign_persistent_tube_ids(
            tubes=current,
            previous_tracks=previous,
            previous_frame_ids=[0, 10],
            current_frame_ids=[10, 20],
            window_index=1,
            next_persistent_id=100,
            uv_radius=0.001,
        )
        self.assertEqual(current[0]["persistent_tube_id"], 77)
        self.assertTrue(current[0]["is_warmstarted"])
        self.assertEqual(diag["persistent_tube_retention_count"], 1)
        batch = _tracks_to_batch(current, [10, 20])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "carriers_window000.npz"
            batch.save_npz(path)
            with np.load(path) as data:
                self.assertEqual(int(data["persistent_tube_id"][0]), 77)
                self.assertTrue(bool(data["is_warmstarted"][0]))


if __name__ == "__main__":
    unittest.main()
