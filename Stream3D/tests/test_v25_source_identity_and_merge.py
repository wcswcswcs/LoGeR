from __future__ import annotations

import unittest

import numpy as np

from stream4d_native.d4rt_scene_builder import (
    D4RTNativeSceneBuilder,
    source_xy_from_uv,
    stable_source_carrier_id,
)
from stream4d_native.measurement_bank import MaskMeasurement, build_measurement_bank, count_pair_measurement_evidence
from stream4d_native.object_tube_io import TubeRecord
from stream4d_native.self_stitch import match_overlap_carriers
from stream4d_native.signed_tube_graph import TubeGraphEdge, build_signed_tube_graph
from stream4d_native.tube_cover import select_tube_cover
from stream4d_native.tube_partition import (
    filter_edges_by_min_score,
    filter_edges_by_mutual_topk,
    filter_edges_by_pair_evidence,
    partition_tube_graph,
)


class _FakeBatch:
    def __init__(self, frames: int, n: int) -> None:
        self.uv_pred = np.zeros((frames, n, 2), dtype=np.float32)
        self.xyz_ref = np.zeros((frames, n, 3), dtype=np.float32)
        for t in range(frames):
            self.uv_pred[t, :, 0] = np.linspace(0.1, 0.9, n, dtype=np.float32)
            self.uv_pred[t, :, 1] = 0.2 + 0.01 * t
            self.xyz_ref[t, :, 0] = np.arange(n, dtype=np.float32)
            self.xyz_ref[t, :, 1] = float(t)
            self.xyz_ref[t, :, 2] = 1.0
        self.visibility_prob = np.ones((frames, n), dtype=np.float32)
        self.confidence_prob = np.ones((frames, n), dtype=np.float32)
        self.valid = np.ones((frames, n), dtype=bool)


class _FakeModel:
    def infer_carriers(self, video_rgb_uint8, src_uv_norm, src_frame_local, query_chunk_size):
        return _FakeBatch(int(video_rgb_uint8.shape[0]), int(src_uv_norm.shape[0]))


def _builder() -> D4RTNativeSceneBuilder:
    return D4RTNativeSceneBuilder(_FakeModel(), {"model": {"input": {"clip_frames": 4}}}, temporal_chunk_size=4, temporal_chunk_stride=2)


def _window_without_identity(uv_offset: float) -> dict[str, np.ndarray]:
    xyz = np.zeros((1, 4, 3), dtype=np.float32)
    xyz[0, :, 0] = np.arange(4, dtype=np.float32)
    uv = np.zeros((1, 4, 2), dtype=np.float32)
    uv[0, :, 0] = np.linspace(0.1, 0.4, 4, dtype=np.float32) + float(uv_offset)
    uv[0, :, 1] = 0.2
    return {
        "frame_ids": np.asarray([0], dtype=np.int64),
        "xyz": xyz,
        "uv": uv,
        "valid": np.ones((1, 4), dtype=bool),
        "visibility": np.ones((1, 4), dtype=np.float32),
        "confidence": np.ones((1, 4), dtype=np.float32),
    }


def _record(tube_id: int, chunk_id: int, *, frame: str = "d4rt_canonical", allow: bool = True) -> TubeRecord:
    xyz = np.asarray([[float(tube_id), 0.0, 1.0], [float(tube_id), 0.1, 1.0]], dtype=np.float32)
    return TubeRecord(
        tube_id=int(tube_id),
        persistent_tube_id=int(tube_id),
        chunk_id=int(chunk_id),
        submap_id=0,
        source_frame_global=0,
        source_xy=(tube_id, tube_id + 1),
        source_uv=(0.1, 0.1),
        target_frames_global=np.asarray([0, 1], dtype=np.int64),
        uv=np.asarray([[0.25, 0.25], [0.26, 0.25]], dtype=np.float32),
        visibility=np.ones((2,), dtype=np.float32),
        confidence=np.ones((2,), dtype=np.float32),
        xyz_local=xyz,
        xyz_ref0=xyz,
        xyz_canonical=xyz if frame == "d4rt_canonical" else None,
        T_chunk_to_canonical={"scale": 1.0, "rot": np.eye(3).tolist(), "trans": [0.0, 0.0, 0.0]},
        alignment_quality={"pass_gate": allow},
        coordinate_frame=frame,
        scale_status="canonical" if frame == "d4rt_canonical" else "local",
        allow_metric_merge=bool(allow),
        alignment_source="d4rt_self_sim3" if chunk_id else "same_chunk_identity",
        transform_id=f"chunk{chunk_id}",
    )


def _record_uv(tube_id: int, uv_xy: tuple[float, float]) -> TubeRecord:
    tube = _record(tube_id, tube_id)
    tube.target_frames_global = np.asarray([0], dtype=np.int64)
    tube.uv = np.asarray([[float(uv_xy[0]), float(uv_xy[1])]], dtype=np.float32)
    tube.visibility = np.ones((1,), dtype=np.float32)
    tube.confidence = np.ones((1,), dtype=np.float32)
    tube.xyz_local = np.asarray([[float(tube_id), 0.0, 1.0]], dtype=np.float32)
    tube.xyz_ref0 = np.asarray([[float(tube_id), 0.0, 1.0]], dtype=np.float32)
    tube.xyz_canonical = np.asarray([[float(tube_id), 0.0, 1.0]], dtype=np.float32)
    return tube


class V25SourceIdentityAndMergeTest(unittest.TestCase):
    def test_source_identity_same_global_pixel_same_carrier_id(self) -> None:
        builder = _builder()
        frames = np.zeros((4, 10, 20, 3), dtype=np.uint8)
        source_points = np.asarray([[2, 0.50, 0.50]], dtype=np.float32)
        first = builder._decode_source_points(frames, source_points, frame_start=100)[0]
        second = builder._decode_source_points(frames, source_points, frame_start=100)[0]
        x, y = source_xy_from_uv(source_points[0, 1:3], image_width=20, image_height=10)
        expected = stable_source_carrier_id(102, x, y, 20)
        self.assertEqual(first["carrier_id"], expected)
        self.assertEqual(second["carrier_id"], expected)
        self.assertEqual(first["persistent_tube_id"], expected)
        self.assertEqual(first["source_pixel_key"], "102:%d:%d" % (x, y))

    def test_source_identity_different_chunk_same_local_index_not_same_carrier_id(self) -> None:
        builder = _builder()
        frames = np.zeros((4, 10, 20, 3), dtype=np.uint8)
        source_points = np.asarray([[0, 0.50, 0.50]], dtype=np.float32)
        first = builder._decode_source_points(frames, source_points, frame_start=0)[0]
        second = builder._decode_source_points(frames, source_points, frame_start=160)[0]
        self.assertNotEqual(first["carrier_id"], second["carrier_id"])
        self.assertNotEqual(first["source_frame_global"], second["source_frame_global"])

    def test_query_order_shuffle_preserves_self_stitch(self) -> None:
        base = _window_without_identity(uv_offset=0.0)
        ids = np.asarray([10, 11, 12, 13], dtype=np.int64)
        base["carrier_id"] = ids
        shuffled = {key: np.array(value, copy=True) for key, value in base.items()}
        order = np.asarray([2, 0, 3, 1], dtype=np.int64)
        shuffled["xyz"] = shuffled["xyz"][:, order, :]
        shuffled["uv"] = shuffled["uv"][:, order, :]
        shuffled["valid"] = shuffled["valid"][:, order]
        shuffled["visibility"] = shuffled["visibility"][:, order]
        shuffled["confidence"] = shuffled["confidence"][:, order]
        shuffled["carrier_id"] = ids[order]
        match = match_overlap_carriers(base, shuffled)
        self.assertEqual(match.stats["used_carrier_id_match_count"], 4)
        self.assertEqual(match.stats["match_source_stable_id_count"], 4)

    def test_missing_source_identity_disables_stable_matching(self) -> None:
        prev = _window_without_identity(uv_offset=0.0)
        curr = _window_without_identity(uv_offset=0.5)
        match = match_overlap_carriers(prev, curr, uv_radius=0.01)
        self.assertEqual(match.stats["used_carrier_id_match_count"], 0)
        self.assertEqual(match.stats["match_source_stable_id_count"], 0)
        self.assertEqual(match.stats["used_anchor_count"], 0)
        self.assertFalse(match.stats["default_range_id_detected"])

    def test_no_default_range_ids_used_for_cross_chunk_self_stitch(self) -> None:
        prev = _window_without_identity(uv_offset=0.0)
        curr = _window_without_identity(uv_offset=0.4)
        match = match_overlap_carriers(prev, curr, uv_radius=0.001)
        self.assertEqual(match.stats["raw_carrier_id_match_count"], 0)
        self.assertEqual(match.stats["used_total_match_count"], 0)

    def test_native_merge_uses_guarded_canonical_geometry(self) -> None:
        tubes = [_record(1, 0), _record(2, 1)]
        measurements, meas_diag = build_measurement_bank(tubes)
        cover = select_tube_cover(measurements)
        events: list[dict] = []
        graph = build_signed_tube_graph(tubes, cover.selected_measurements, event_logger=events.append, threshold_alpha=10.0)
        partition = partition_tube_graph([tube.tube_id for tube in tubes], graph.edges)
        self.assertFalse(meas_diag["measurement_uses_metric_geometry"])
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0]["geometry_field_used"], "xyz_canonical")
        self.assertEqual(events[0]["distance_threshold_type"], "spacing_normalized")
        self.assertEqual(partition.diagnostics["component_count"], 1)

    def test_native_merge_blocks_cross_chunk_local_geometry(self) -> None:
        tubes = [_record(1, 0, frame="chunk_local"), _record(2, 1, frame="chunk_local")]
        measurements, _ = build_measurement_bank(tubes)
        graph = build_signed_tube_graph(tubes, measurements, threshold_alpha=10.0)
        self.assertEqual(len(graph.edges), 0)
        self.assertEqual(graph.blocked_events[0]["guard_reason"], "cross_chunk_requires_xyz_canonical")

    def test_visible_outside_negative_evidence_counts_candidate_pairs(self) -> None:
        measurements = [
            MaskMeasurement(
                measurement_id="same",
                frame_global=0,
                mask_id=1,
                tube_ids=[1, 2],
                inside_tube_ids=[1, 2],
                outside_visible_tube_ids=[3],
            ),
            MaskMeasurement(
                measurement_id="split",
                frame_global=1,
                mask_id=2,
                tube_ids=[1],
                inside_tube_ids=[1],
                outside_visible_tube_ids=[2, 3],
            ),
        ]
        evidence = count_pair_measurement_evidence(measurements, {(1, 2), (1, 3)})
        self.assertEqual(evidence[(1, 2)]["same_mask_count"], 1)
        self.assertEqual(evidence[(1, 2)]["visible_outside_negative_count"], 1)
        self.assertEqual(evidence[(1, 3)]["same_mask_count"], 0)
        self.assertEqual(evidence[(1, 3)]["visible_outside_negative_count"], 2)

    def test_v27_measurement_bank_records_boundary_and_cannot_link_evidence(self) -> None:
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[1:8, 1:8] = 1
        tubes = [
            _record_uv(1, (4 / 9, 4 / 9)),
            _record_uv(2, (5 / 9, 4 / 9)),
            _record_uv(3, (1 / 9, 4 / 9)),
            _record_uv(4, (0 / 9, 4 / 9)),
        ]
        measurements, diag = build_measurement_bank(
            tubes,
            masks_by_frame={0: mask},
            interior_distance_px=2.0,
            boundary_distance_px=1.5,
            boundary_cross_radius_px=2.0,
        )
        evidence = count_pair_measurement_evidence(measurements, {(1, 2), (3, 4), (1, 4)})
        self.assertGreater(diag["num_boundary_safe_merge_pairs"], 0)
        self.assertGreater(diag["num_boundary_cross_cut_pairs"], 0)
        self.assertGreater(evidence[(1, 2)]["boundary_safe_count"], 0)
        self.assertGreater(evidence[(3, 4)]["boundary_cross_count"], 0)
        self.assertGreater(evidence[(1, 4)]["visible_outside_negative_count"], 0)

    def test_v27_measurement_bank_records_same_frame_different_mask_cannot_link(self) -> None:
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[:, :5] = 1
        mask[:, 5:] = 2
        tubes = [_record_uv(1, (2 / 9, 5 / 9)), _record_uv(2, (7 / 9, 5 / 9))]
        measurements, diag = build_measurement_bank(tubes, masks_by_frame={0: mask})
        evidence = count_pair_measurement_evidence(measurements, {(1, 2)})
        self.assertGreater(diag["num_same_frame_cannot_link_pairs"], 0)
        self.assertGreater(evidence[(1, 2)]["same_frame_cannot_link_count"], 0)

    def test_negative_evidence_filter_removes_boundary_edges(self) -> None:
        edges = [
            TubeGraphEdge(1, 2, 1, 0.8, 0.1, 0.2, "a", {}),
            TubeGraphEdge(1, 3, 1, 0.7, 0.1, 0.2, "b", {}),
        ]
        evidence = {
            (1, 2): {"same_mask_count": 2, "visible_outside_negative_count": 1},
            (1, 3): {"same_mask_count": 1, "visible_outside_negative_count": 2},
        }
        majority = filter_edges_by_pair_evidence(edges, evidence, mode="negative_majority")
        strict = filter_edges_by_pair_evidence(edges, evidence, mode="negative_strict")
        self.assertEqual([(edge.tube_i, edge.tube_j) for edge in majority], [(1, 2)])
        self.assertEqual(strict, [])

    def test_mutual_topk_keeps_only_reciprocal_strong_neighbors(self) -> None:
        edges = [
            TubeGraphEdge(1, 2, 1, 0.9, 0.1, 0.2, "a", {}),
            TubeGraphEdge(1, 3, 1, 0.8, 0.1, 0.2, "b", {}),
            TubeGraphEdge(2, 3, 1, 0.7, 0.1, 0.2, "c", {}),
        ]
        top1 = filter_edges_by_mutual_topk(edges, top_k=1)
        top2 = filter_edges_by_mutual_topk(edges, top_k=2)
        self.assertEqual([(edge.tube_i, edge.tube_j) for edge in top1], [(1, 2)])
        self.assertEqual([(edge.tube_i, edge.tube_j) for edge in top2], [(1, 2), (1, 3), (2, 3)])

    def test_min_score_filter_keeps_normalized_high_confidence_edges(self) -> None:
        edges = [
            TubeGraphEdge(1, 2, 1, 0.9, 0.1, 0.2, "a", {}),
            TubeGraphEdge(1, 3, 1, 0.4, 0.1, 0.2, "b", {}),
        ]
        kept = filter_edges_by_min_score(edges, min_score=0.5)
        self.assertEqual([(edge.tube_i, edge.tube_j) for edge in kept], [(1, 2)])
        with self.assertRaises(ValueError):
            filter_edges_by_min_score(edges, min_score=1.5)


if __name__ == "__main__":
    unittest.main()
