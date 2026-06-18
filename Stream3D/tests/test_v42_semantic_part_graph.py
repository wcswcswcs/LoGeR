from __future__ import annotations

import unittest

import numpy as np

from stream4d_native.frozen_feature_adapter import FrozenFeatureAdapter
from stream4d_native.measurement_bank import MaskMeasurement
from stream4d_native.semantic_material_mask_split import backfill_masks_by_material_support, split_masks_by_material_uv
from stream4d_native.semantic_material_part_graph import (
    build_material_part_graph_edges,
    build_token_material_support,
    summarize_material_part_graph,
)
from stream4d_native.semantic_part_graph import build_part_graph_edges, summarize_part_graph
from stream4d_native.semantic_part_tokens import (
    SemanticPartToken,
    build_semantic_part_tokens,
    merge_masks_by_feature_affinity,
    split_masks_by_feature_clusters,
)
from tools.run_v42_semantic_part_audit import _prepared_masks


class V42SemanticPartGraphTests(unittest.TestCase):
    def test_same_frame_cannot_link_penalizes_false_merge(self) -> None:
        tokens = [
            SemanticPartToken(0, 0, 1, 100, np.asarray([1.0, 0.0], dtype=np.float32), 0.0, 2.0, 2.0, 10, 0.95, 0.80),
            SemanticPartToken(1, 0, 2, 100, np.asarray([1.0, 0.0], dtype=np.float32), 0.0, 2.0, 8.0, 20, 0.95, 0.80),
            SemanticPartToken(2, 1, 3, 100, np.asarray([1.0, 0.0], dtype=np.float32), 0.0, 2.0, 2.5, 10, 0.95, 0.80),
        ]
        edges = build_part_graph_edges(tokens)
        same_frame = next(edge for edge in edges if {edge.token_i, edge.token_j} == {0, 1})
        same_object = next(edge for edge in edges if {edge.token_i, edge.token_j} == {0, 2})
        self.assertTrue(same_frame.same_frame_cannot_link)
        self.assertLess(same_frame.object_affinity, same_object.object_affinity)
        summary = summarize_part_graph(tokens, edges)
        self.assertIsNotNone(summary["semantic_affinity_AUC"])

    def test_twohop_structure_affinity_bridges_semantic_chain(self) -> None:
        angle = np.pi / 180.0
        tokens = [
            SemanticPartToken(0, 0, 1, 100, np.asarray([1.0, 0.0], dtype=np.float32), 0.0, 2.0, 2.0, 10, 0.95, 0.80),
            SemanticPartToken(
                1,
                1,
                2,
                100,
                np.asarray([np.cos(40 * angle), np.sin(40 * angle)], dtype=np.float32),
                0.0,
                2.0,
                2.0,
                10,
                0.95,
                0.80,
            ),
            SemanticPartToken(
                2,
                2,
                3,
                100,
                np.asarray([np.cos(80 * angle), np.sin(80 * angle)], dtype=np.float32),
                0.0,
                2.0,
                2.0,
                10,
                0.95,
                0.80,
            ),
        ]
        cosine_edges = build_part_graph_edges(tokens)
        structure_edges = build_part_graph_edges(
            tokens,
            semantic_affinity_mode="twohop_structure",
            structure_topk=2,
            structure_min_affinity=0.25,
            structure_decay=0.95,
        )
        cosine_ac = next(edge for edge in cosine_edges if {edge.token_i, edge.token_j} == {0, 2})
        structure_ac = next(edge for edge in structure_edges if {edge.token_i, edge.token_j} == {0, 2})
        self.assertLess(cosine_ac.semantic_affinity, 0.25)
        self.assertGreater(structure_ac.semantic_affinity, 0.70)

    def test_temporal_widest_structure_respects_frame_window(self) -> None:
        angle = np.pi / 180.0
        tokens = [
            SemanticPartToken(0, 0, 1, 100, np.asarray([1.0, 0.0], dtype=np.float32), 0.0, 2.0, 2.0, 10, 0.95, 0.80),
            SemanticPartToken(
                1,
                10,
                2,
                100,
                np.asarray([np.cos(40 * angle), np.sin(40 * angle)], dtype=np.float32),
                0.0,
                2.0,
                2.0,
                10,
                0.95,
                0.80,
            ),
            SemanticPartToken(
                2,
                20,
                3,
                100,
                np.asarray([np.cos(80 * angle), np.sin(80 * angle)], dtype=np.float32),
                0.0,
                2.0,
                2.0,
                10,
                0.95,
                0.80,
            ),
            SemanticPartToken(
                3,
                60,
                4,
                100,
                np.asarray([np.cos(80 * angle), np.sin(80 * angle)], dtype=np.float32),
                0.0,
                2.0,
                2.0,
                10,
                0.95,
                0.80,
            ),
        ]
        edges = build_part_graph_edges(
            tokens,
            semantic_affinity_mode="temporal_widest_structure",
            structure_topk=3,
            structure_min_affinity=0.25,
            structure_decay=0.95,
            structure_temporal_window=15,
        )
        near_chain = next(edge for edge in edges if {edge.token_i, edge.token_j} == {0, 2})
        far_pair = next(edge for edge in edges if {edge.token_i, edge.token_j} == {0, 3})
        self.assertGreater(near_chain.semantic_affinity, 0.70)
        self.assertLess(far_pair.semantic_affinity, 0.25)

    def test_temporal_chain_structure_uses_adjacent_observation_rank(self) -> None:
        angle = np.pi / 180.0
        tokens = [
            SemanticPartToken(0, 0, 1, 100, np.asarray([1.0, 0.0], dtype=np.float32), 0.0, 2.0, 2.0, 10, 0.95, 0.80),
            SemanticPartToken(
                1,
                100,
                2,
                100,
                np.asarray([np.cos(35 * angle), np.sin(35 * angle)], dtype=np.float32),
                0.0,
                2.0,
                2.0,
                10,
                0.95,
                0.80,
            ),
            SemanticPartToken(
                2,
                1000,
                3,
                100,
                np.asarray([np.cos(70 * angle), np.sin(70 * angle)], dtype=np.float32),
                0.0,
                2.0,
                2.0,
                10,
                0.95,
                0.80,
            ),
        ]
        cosine_edges = build_part_graph_edges(tokens)
        chain_edges = build_part_graph_edges(
            tokens,
            semantic_affinity_mode="temporal_chain_structure",
            structure_topk=2,
            structure_min_affinity=0.25,
            structure_decay=0.95,
            structure_temporal_rank_window=1,
        )
        cosine_ac = next(edge for edge in cosine_edges if {edge.token_i, edge.token_j} == {0, 2})
        chain_ac = next(edge for edge in chain_edges if {edge.token_i, edge.token_j} == {0, 2})
        self.assertLess(cosine_ac.semantic_affinity, 0.40)
        self.assertGreater(chain_ac.semantic_affinity, 0.75)

    def test_feature_cluster_split_divides_overmerged_mask(self) -> None:
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[:, :8, 0] = 255
        frame[:, 8:, 1] = 255
        overmerged = np.ones((16, 16), dtype=bool)
        adapter = FrozenFeatureAdapter(backend="rgb_stats")
        fmap = adapter.extract_dense_features(frame)
        fragments = split_masks_by_feature_clusters(
            [(1, overmerged)],
            fmap,
            image_shape=frame.shape[:2],
            min_area=16,
            max_splits=2,
            spatial_weight=0.0,
        )
        self.assertGreaterEqual(len(fragments), 2)
        self.assertEqual(sum(int(mask.sum()) for _mask_id, mask in fragments), int(overmerged.sum()))

    def test_local_contrast_token_feature_mode_appends_contrast(self) -> None:
        frame = np.zeros((12, 12, 3), dtype=np.uint8)
        frame[:, :6, 0] = 255
        frame[:, 6:, 1] = 255
        mask = np.zeros((12, 12), dtype=bool)
        mask[:, :6] = True
        adapter = FrozenFeatureAdapter(backend="rgb_stats")
        pooled = build_semantic_part_tokens(
            frame_id=0,
            frame=frame,
            masks=[(1, mask)],
            adapter=adapter,
            feature_mode="pooled",
        )[0]
        contrast = build_semantic_part_tokens(
            frame_id=0,
            frame=frame,
            masks=[(1, mask)],
            adapter=adapter,
            feature_mode="pooled_local_contrast",
        )[0]
        self.assertEqual(contrast.feature.shape[0], pooled.feature.shape[0] * 2)
        self.assertGreater(float(np.linalg.norm(contrast.feature)), 0.0)

    def test_feature_affinity_merge_combines_similar_fragments(self) -> None:
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[:, :10, 0] = 255
        frame[:, 10:, 1] = 255
        red_a = np.zeros((16, 16), dtype=bool)
        red_b = np.zeros((16, 16), dtype=bool)
        green = np.zeros((16, 16), dtype=bool)
        red_a[2:8, 2:6] = True
        red_b[2:8, 6:10] = True
        green[2:8, 11:15] = True
        adapter = FrozenFeatureAdapter(backend="rgb_stats")
        fmap = adapter.extract_dense_features(frame)
        merged = merge_masks_by_feature_affinity(
            [(1, red_a), (2, red_b), (3, green)],
            fmap,
            image_shape=frame.shape[:2],
            min_area=8,
            affinity_threshold=0.98,
            max_center_distance=0.40,
            max_group_size=3,
        )
        areas = sorted(int(mask.sum()) for _mask_id, mask in merged)
        self.assertEqual(areas, [24, 48])

    def test_material_support_boosts_shared_tube_and_penalizes_conflict(self) -> None:
        tokens = [
            SemanticPartToken(0, 0, 1, 100, np.asarray([1.0, 0.0], dtype=np.float32), 0.0, 2.0, 2.0, 10, 0.95, 0.40),
            SemanticPartToken(1, 1, 1, 100, np.asarray([1.0, 0.0], dtype=np.float32), 0.0, 2.0, 2.0, 10, 0.95, 0.40),
            SemanticPartToken(2, 0, 2, 100, np.asarray([1.0, 0.0], dtype=np.float32), 0.0, 2.0, 8.0, 20, 0.95, 0.40),
        ]
        semantic_edges = build_part_graph_edges(tokens)
        measurements = [
            MaskMeasurement("f000000_m0001", 0, 1, [1], [1], outside_visible_tube_ids=[2]),
            MaskMeasurement("f000001_m0001", 1, 1, [1], [1]),
            MaskMeasurement("f000000_m0002", 0, 2, [2], [2], outside_visible_tube_ids=[1]),
        ]
        support = build_token_material_support(tokens, measurements)
        material_edges = build_material_part_graph_edges(
            semantic_edges,
            support,
            material_weight=0.35,
            conflict_weight=0.35,
        )
        same_object = next(edge for edge in material_edges if {edge.token_i, edge.token_j} == {0, 1})
        conflict = next(edge for edge in material_edges if {edge.token_i, edge.token_j} == {0, 2})
        self.assertGreater(same_object.p4_semantic_material_affinity, same_object.semantic_object_affinity)
        self.assertLess(conflict.p5_semantic_material_boundary_affinity, conflict.p4_semantic_material_affinity)
        summary = summarize_material_part_graph(
            tokens,
            material_edges,
            support,
            semantic_false_merge_rate=0.5,
            coverage_at_010=0.8,
        )
        self.assertEqual(summary["material_supported_token_count"], 3)
        self.assertEqual(len(summary["variant_rows"]), 4)

    def test_material_support_min_shared_and_shrinkage_suppresses_singleton_boost(self) -> None:
        tokens = [
            SemanticPartToken(0, 0, 1, 100, np.asarray([1.0, 0.0], dtype=np.float32), 0.0, 2.0, 2.0, 10, 0.95, 0.40),
            SemanticPartToken(1, 1, 1, 100, np.asarray([1.0, 0.0], dtype=np.float32), 0.0, 2.0, 2.0, 10, 0.95, 0.40),
        ]
        semantic_edges = build_part_graph_edges(tokens)
        support = build_token_material_support(
            tokens,
            [
                MaskMeasurement("f000000_m0001", 0, 1, [7], [7]),
                MaskMeasurement("f000001_m0001", 1, 1, [7], [7]),
            ],
        )
        default_edge = build_material_part_graph_edges(semantic_edges, support)[0]
        gated_edge = build_material_part_graph_edges(semantic_edges, support, min_shared_tube_count=2)[0]
        shrunk_edge = build_material_part_graph_edges(semantic_edges, support, material_support_shrinkage=1.0)[0]
        self.assertEqual(default_edge.shared_tube_count, 1)
        self.assertEqual(default_edge.material_jaccard, 1.0)
        self.assertEqual(gated_edge.material_jaccard, 0.0)
        self.assertAlmostEqual(gated_edge.p4_semantic_material_affinity, gated_edge.semantic_object_affinity)
        self.assertAlmostEqual(shrunk_edge.material_jaccard, 0.5)

    def test_material_uv_split_divides_overmerged_mask(self) -> None:
        class FakeTube:
            def __init__(self, tube_id: int, uv: tuple[float, float]) -> None:
                self.tube_id = int(tube_id)
                self.target_frames_global = np.asarray([0], dtype=np.int64)
                self._uv = np.asarray([uv], dtype=np.float32)

            def get_geometry_for_measurement(self, *, field: str = "uv") -> np.ndarray:
                if field == "uv":
                    return self._uv
                if field in {"visibility", "confidence"}:
                    return np.ones((1,), dtype=np.float32)
                raise ValueError(field)

        mask = np.zeros((12, 12), dtype=bool)
        mask[2:10, 2:10] = True
        split_by_frame, diag = split_masks_by_material_uv(
            {0: [(1, mask)]},
            [FakeTube(1, (0.25, 0.50)), FakeTube(2, (0.75, 0.50))],
            min_area=8,
            max_splits=2,
            min_tubes=2,
            min_cluster_distance_px=2.0,
        )
        fragments = split_by_frame[0]
        self.assertEqual(len(fragments), 2)
        self.assertEqual(sum(int(fragment.sum()) for _mask_id, fragment in fragments), int(mask.sum()))
        self.assertEqual(diag["split_mask_count"], 1)

    def test_material_supported_backfill_keeps_only_anchored_masks(self) -> None:
        class FakeTube:
            def __init__(self, tube_id: int, uv: tuple[float, float]) -> None:
                self.tube_id = int(tube_id)
                self.target_frames_global = np.asarray([0], dtype=np.int64)
                self._uv = np.asarray([uv], dtype=np.float32)

            def get_geometry_for_measurement(self, *, field: str = "uv") -> np.ndarray:
                if field == "uv":
                    return self._uv
                if field in {"visibility", "confidence"}:
                    return np.ones((1,), dtype=np.float32)
                raise ValueError(field)

        primary = np.zeros((10, 10), dtype=bool)
        primary[:2, :2] = True
        anchored = np.zeros((10, 10), dtype=bool)
        anchored[4:7, 4:7] = True
        unanchored = np.zeros((10, 10), dtype=bool)
        unanchored[7:10, 7:10] = True
        out, diag = backfill_masks_by_material_support(
            {0: [(1, primary)]},
            [{0: [(10, anchored), (11, unanchored)]}],
            [FakeTube(1, (0.5, 0.5))],
            overlap_iou=0.10,
            max_backfill_per_frame=4,
            min_tubes=1,
        )
        self.assertEqual(len(out[0]), 2)
        self.assertEqual(diag["selected_backfill_count"], 1)
        self.assertEqual(diag["rejected_no_material_support_count"], 1)

    def test_material_supported_backfill_rejects_oversized_candidates(self) -> None:
        class FakeTube:
            def __init__(self, tube_id: int, uv: tuple[float, float]) -> None:
                self.tube_id = int(tube_id)
                self.target_frames_global = np.asarray([0], dtype=np.int64)
                self._uv = np.asarray([uv], dtype=np.float32)

            def get_geometry_for_measurement(self, *, field: str = "uv") -> np.ndarray:
                if field == "uv":
                    return self._uv
                if field in {"visibility", "confidence"}:
                    return np.ones((1,), dtype=np.float32)
                raise ValueError(field)

        primary = np.zeros((10, 10), dtype=bool)
        primary[:2, :2] = True
        oversized = np.ones((10, 10), dtype=bool)
        out, diag = backfill_masks_by_material_support(
            {0: [(1, primary)]},
            [{0: [(10, oversized)]}],
            [FakeTube(1, (0.5, 0.5))],
            overlap_iou=0.10,
            max_backfill_per_frame=4,
            min_tubes=1,
            max_candidate_area_fraction=0.25,
        )
        self.assertEqual(len(out[0]), 1)
        self.assertEqual(diag["selected_backfill_count"], 0)
        self.assertEqual(diag["rejected_oversize_count"], 1)
        self.assertEqual(diag["max_candidate_area_fraction"], 0.25)

    def test_prepared_masks_treats_missing_stride1_frames_as_empty_observations(self) -> None:
        class FakeStream:
            def load_mask(self, frame_id: int) -> np.ndarray:
                if int(frame_id) == 1:
                    raise FileNotFoundError("missing prepared mask")
                label = np.zeros((6, 6), dtype=np.int32)
                label[1:5, 1:5] = 3
                return label

        masks = _prepared_masks(FakeStream(), [0, 1], min_area=4)
        self.assertEqual(len(masks[0]), 1)
        self.assertEqual(masks[1], [])


if __name__ == "__main__":
    unittest.main()
