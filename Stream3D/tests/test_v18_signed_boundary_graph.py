from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stream4d.measurement_bank import MeasurementBank
from stream4d.signed_boundary_evidence import (
    SignedBoundaryEvidence,
    build_signed_boundary_evidence,
)
from stream4d.signed_graph_partition import partition_signed_graph, partition_to_object_dict
from stream4d.signed_surfel_graph import (
    EDGE_TYPE_IDS,
    SignedSurfelGraph,
    build_signed_surfel_graph,
    summarize_signed_surfel_graph,
)


def _toy_bank() -> MeasurementBank:
    side = 3
    num = side * side
    frames = np.asarray([0, 1, 2], dtype=np.int64)
    src_xy = np.asarray([[x, y] for y in range(side) for x in range(side)], dtype=np.int64)
    uv0 = src_xy.astype(np.float32) / float(side - 1)
    uv_pred = np.stack([uv0 + np.asarray([0.01 * frame, 0.0], dtype=np.float32) for frame in range(frames.size)], axis=0)
    uv_pred = np.clip(uv_pred, 0.0, 1.0)
    mask_by_node = np.where(src_xy[:, 0] < 2, 1, 2).astype(np.int64)
    target_mask_id = np.tile(mask_by_node[None, :], (frames.size, 1))
    positive = target_mask_id > 0
    rgb = np.stack(
        [
            np.where(mask_by_node == 1, 0.10, 0.80),
            np.linspace(0.0, 1.0, num),
            np.full((num,), 0.25, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return MeasurementBank(
        scene="toy_scene",
        frame_ids=frames,
        carrier_id=np.arange(num, dtype=np.int64),
        uv_pred=uv_pred.astype(np.float32),
        valid=np.ones((frames.size, num), dtype=bool),
        visibility=np.ones((frames.size, num), dtype=np.float32),
        confidence=np.ones((frames.size, num), dtype=np.float32),
        src_frame_global=np.zeros((num,), dtype=np.int64),
        src_mask_id=mask_by_node.copy(),
        src_xy=src_xy,
        src_rgb=rgb,
        target_mask_id=target_mask_id,
        target_in_bounds=np.ones((frames.size, num), dtype=bool),
        visible_ok=np.ones((frames.size, num), dtype=bool),
        boundary_distance=np.full((frames.size, num), 8.0, dtype=np.float32),
        source_boundary_distance=np.full((num,), 8.0, dtype=np.float32),
        mask_frame_available=np.ones((frames.size,), dtype=bool),
        positive_observation=positive,
        negative_observation=~positive,
        source_positive_propagated=positive.copy(),
        meta={"self_uv_error_p90_mean": 0.5, "cycle_uv_error_p90_mean": 1.0},
    )


def _edge_index(graph: SignedSurfelGraph, a: int, b: int) -> int:
    lo, hi = sorted((a, b))
    hits = np.flatnonzero((graph.src == lo) & (graph.dst == hi))
    if hits.size == 0:
        raise AssertionError(f"missing edge {lo}-{hi}")
    return int(hits[0])


class V18SignedBoundaryGraphTest(unittest.TestCase):
    def test_graph_builds_expected_edge_types_and_round_trips(self) -> None:
        bank = _toy_bank()
        graph = build_signed_surfel_graph(bank, knn_k=2, knn_max_frames=3, cross_frame_neighbors=2)
        self.assertEqual(graph.scene, "toy_scene")
        self.assertEqual(graph.num_nodes, bank.num_surfels)
        self.assertGreater(graph.num_edges, 0)
        self.assertIn(EDGE_TYPE_IDS["E_2d_grid"], set(graph.edge_type.tolist()))
        self.assertIn(EDGE_TYPE_IDS["E_2d_knn"], set(graph.edge_type.tolist()))
        summary = summarize_signed_surfel_graph(graph, bank)
        self.assertEqual(summary["status"], "ok")
        self.assertGreater(summary["largest_graph_component_ratio"], 0.0)
        self.assertEqual(summary["phase1_gate"]["cycle_uv_error_p90_le_5px"], True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.npz"
            graph.save(path)
            loaded = SignedSurfelGraph.load(path)
            np.testing.assert_array_equal(loaded.src, graph.src)
            np.testing.assert_array_equal(loaded.dst, graph.dst)

    def test_signed_evidence_separates_same_mask_from_cross_mask_edges(self) -> None:
        bank = _toy_bank()
        graph = build_signed_surfel_graph(bank, knn_k=2, knn_max_frames=3, cross_frame_neighbors=0)
        evidence = build_signed_boundary_evidence(bank, graph, variant="E5_full_signed")
        same = _edge_index(graph, 0, 1)
        cross = _edge_index(graph, 1, 2)
        self.assertGreater(float(evidence.merge_weight[same]), 0.0)
        self.assertGreater(float(evidence.cut_weight[cross]), float(evidence.cut_weight[same]))
        self.assertGreater(float(evidence.cut_score[cross]), float(evidence.cut_score[same]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.npz"
            evidence.save(path)
            loaded = SignedBoundaryEvidence.load(path)
            np.testing.assert_allclose(loaded.cut_score, evidence.cut_score)
            self.assertEqual(loaded.variant, "E5_full_signed")

    def test_partition_keeps_low_cut_components_and_exports_votes(self) -> None:
        bank = _toy_bank()
        graph = SignedSurfelGraph(
            scene=bank.scene,
            num_nodes=bank.num_surfels,
            src=np.asarray([0, 1, 3, 4], dtype=np.int64),
            dst=np.asarray([1, 2, 4, 5], dtype=np.int64),
            edge_type=np.asarray([0, 0, 0, 0], dtype=np.int16),
            num_visible_together=np.asarray([3, 3, 3, 3], dtype=np.int16),
            mean_uv_distance=np.ones((4,), dtype=np.float32),
            median_uv_distance=np.ones((4,), dtype=np.float32),
            mean_rgb_distance=np.ones((4,), dtype=np.float32),
            trajectory_relative_motion_variance=np.zeros((4,), dtype=np.float32),
            precut_keep=np.asarray([True, False, True, True], dtype=bool),
            meta={"algorithm": "toy"},
        )
        evidence = SignedBoundaryEvidence(
            scene=bank.scene,
            variant="toy",
            merge_weight=np.asarray([3.0, 0.0, 3.0, 3.0], dtype=np.float32),
            cut_weight=np.asarray([0.0, 3.0, 0.0, 0.0], dtype=np.float32),
            cut_score=np.asarray([0.20, 0.90, 0.20, 0.20], dtype=np.float32),
            num_frames_used=np.asarray([3, 3, 3, 3], dtype=np.int16),
            meta={},
        )
        result = partition_signed_graph(
            graph,
            evidence,
            mode="P2_agglomerative_signed",
            cut_threshold=0.62,
            merge_threshold=0.55,
            min_component_size=2,
            max_component_ratio=1.0,
        )
        sizes = sorted(comp.size for comp in result.components)
        self.assertIn(2, sizes)
        objects = partition_to_object_dict(bank, result, export_mode="G_core")
        self.assertGreaterEqual(len(objects), 1)
        self.assertIn("mask_list", next(iter(objects.values())))


if __name__ == "__main__":
    unittest.main()
