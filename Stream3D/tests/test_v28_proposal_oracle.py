from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from stream4d_native.measurement_bank import MaskMeasurement
from stream4d_native.object_tube_io import TubeRecord
from tools.run_v28_proposal_oracle import (
    Proposal,
    _canonical_clusters_for_core,
    _cluster_metrics,
    _feature_auc_rows,
    _filter_temporal_proposals,
    _proposal_diag,
    _shuffle_d4rt_records_for_control,
    _temporal_track_consensus_core_ids,
    _temporal_track_components,
    _visible_negative_eroded_core_ids,
    _visible_negative_pruned_core_ids,
)


class V28ProposalOracleTests(unittest.TestCase):
    def _tube(self, tube_id: int, xyz: tuple[float, float, float]) -> TubeRecord:
        arr = np.asarray([xyz, xyz], dtype=np.float32)
        return TubeRecord(
            tube_id=tube_id,
            persistent_tube_id=tube_id,
            chunk_id=0,
            submap_id=0,
            source_frame_global=0,
            source_xy=(0, 0),
            source_uv=(0.0, 0.0),
            target_frames_global=np.asarray([0, 1], dtype=np.int64),
            uv=np.zeros((2, 2), dtype=np.float32),
            visibility=np.ones((2,), dtype=np.float32),
            confidence=np.ones((2,), dtype=np.float32),
            xyz_local=arr,
            xyz_ref0=arr,
            xyz_canonical=arr,
        )

    def test_proposal_diag_computes_tube_level_purity_completeness_iou(self) -> None:
        proposal = Proposal(
            proposal_id="p0",
            scene="s",
            frame_id=0,
            mask_id=1,
            proposal_type="R0_full_mask_region",
            core_tube_ids=(1, 2, 3),
            fringe_tube_ids=(),
            boundary_tube_ids=(),
            region_area=3.0,
            features={},
        )
        diag = _proposal_diag(
            proposal,
            gt_labels={1: 7, 2: 7, 3: 8, 4: 7},
            gt_counts=Counter({7: 3, 8: 1}),
        )
        self.assertAlmostEqual(diag["proposal_purity"], 2.0 / 3.0)
        self.assertAlmostEqual(diag["proposal_completeness"], 2.0 / 3.0)
        self.assertAlmostEqual(diag["proposal_best_IoU"], 2.0 / 4.0)
        self.assertEqual(diag["proposal_best_GT"], 7)
        self.assertEqual(diag["_gt_overlap_counts"], {7: 2, 8: 1})
        self.assertEqual(diag["_proposal_labeled_tube_count"], 3)

    def test_cluster_metrics_penalizes_overmerge_and_oversplit(self) -> None:
        metrics = _cluster_metrics(
            labels_pred={1: 0, 2: 0, 3: 1, 4: 2},
            gt_labels={1: 10, 2: 20, 3: 20, 4: 20},
        )
        self.assertEqual(metrics["overmerge"], 1)
        self.assertEqual(metrics["oversplit"], 1)
        self.assertLess(metrics["purity"], 1.0)
        self.assertLess(metrics["completeness"], 1.0)

    def test_feature_auc_rows_use_gt_only_as_diagnostic_labels(self) -> None:
        rows = [
            {
                "scene": "scene0000_00",
                "mask_area": 100,
                "core_tube_count": 8,
                "proposal_purity": 0.95,
                "proposal_best_IoU": 0.40,
                "proposal_completeness": 0.60,
            },
            {
                "scene": "scene0000_00",
                "mask_area": 80,
                "core_tube_count": 2,
                "proposal_purity": 0.50,
                "proposal_best_IoU": 0.10,
                "proposal_completeness": 0.10,
            },
            {
                "scene": "scene0081_01",
                "mask_area": 120,
                "core_tube_count": 7,
                "proposal_purity": 0.90,
                "proposal_best_IoU": 0.30,
                "proposal_completeness": 0.55,
            },
            {
                "scene": "scene0081_01",
                "mask_area": 90,
                "core_tube_count": 1,
                "proposal_purity": 0.55,
                "proposal_best_IoU": 0.10,
                "proposal_completeness": 0.05,
            },
        ]
        auc_rows = {row["feature"]: row for row in _feature_auc_rows(rows)}
        self.assertIn("core_tube_count", auc_rows)
        self.assertIsNotNone(auc_rows["core_tube_count"]["purity_AUC"])
        self.assertGreaterEqual(auc_rows["core_tube_count"]["purity_AUC"], 0.5)
        self.assertIsNotNone(auc_rows["core_tube_count"]["scene0081_AUC"])

    def test_temporal_track_components_link_shared_tubes_without_mask_id_assumption(self) -> None:
        measurements = [
            MaskMeasurement(
                measurement_id="a",
                frame_global=0,
                mask_id=10,
                tube_ids=[1, 2, 3, 4],
                inside_tube_ids=[1, 2, 3, 4],
            ),
            MaskMeasurement(
                measurement_id="b",
                frame_global=4,
                mask_id=99,
                tube_ids=[2, 3, 4, 5],
                inside_tube_ids=[2, 3, 4, 5],
            ),
            MaskMeasurement(
                measurement_id="c",
                frame_global=4,
                mask_id=100,
                tube_ids=[20, 21, 22],
                inside_tube_ids=[20, 21, 22],
            ),
        ]
        components = _temporal_track_components(
            measurements,
            min_tubes=2,
            min_shared_tubes=2,
            max_frame_gap=8,
            min_overlap_ratio=0.5,
        )
        self.assertEqual(len(components), 1)
        self.assertEqual({item.measurement_id for item in components[0]}, {"a", "b"})

    def test_temporal_track_components_can_use_low_shared_tube_recall_mode(self) -> None:
        measurements = [
            MaskMeasurement(
                measurement_id="a",
                frame_global=0,
                mask_id=10,
                tube_ids=[1, 2, 3],
                inside_tube_ids=[1, 2, 3],
            ),
            MaskMeasurement(
                measurement_id="b",
                frame_global=4,
                mask_id=99,
                tube_ids=[3, 4, 5],
                inside_tube_ids=[3, 4, 5],
            ),
        ]
        strict = _temporal_track_components(
            measurements,
            min_tubes=2,
            min_shared_tubes=2,
            max_frame_gap=8,
            min_overlap_ratio=0.2,
        )
        recall = _temporal_track_components(
            measurements,
            min_tubes=2,
            min_shared_tubes=1,
            max_frame_gap=8,
            min_overlap_ratio=0.2,
        )
        self.assertEqual(strict, [])
        self.assertEqual(len(recall), 1)

    def test_temporal_track_consensus_core_requires_repeated_support(self) -> None:
        component = [
            MaskMeasurement(
                measurement_id="a",
                frame_global=0,
                mask_id=10,
                tube_ids=[1, 2, 3, 4],
                inside_tube_ids=[1, 2, 3, 4],
            ),
            MaskMeasurement(
                measurement_id="b",
                frame_global=4,
                mask_id=11,
                tube_ids=[2, 3, 4, 5],
                inside_tube_ids=[2, 3, 4, 5],
            ),
            MaskMeasurement(
                measurement_id="c",
                frame_global=8,
                mask_id=12,
                tube_ids=[3, 4, 5, 6],
                inside_tube_ids=[3, 4, 5, 6],
            ),
        ]
        self.assertEqual(
            _temporal_track_consensus_core_ids(component, min_vote_ratio=0.50, min_vote_count=2),
            (2, 3, 4, 5),
        )
        self.assertEqual(
            _temporal_track_consensus_core_ids(component, min_vote_ratio=0.67, min_vote_count=2),
            (3, 4),
        )

    def test_visible_negative_pruned_core_removes_only_visible_outside_tubes(self) -> None:
        measurement = MaskMeasurement(
            measurement_id="a",
            frame_global=0,
            mask_id=10,
            tube_ids=[1, 2],
            inside_tube_ids=[1, 2],
            outside_visible_tube_ids=[3, 5],
        )
        self.assertEqual(_visible_negative_pruned_core_ids((1, 2, 3, 4), measurement), (1, 2, 4))

    def test_visible_negative_eroded_core_keeps_temporal_nonvisible_and_eroded_inside(self) -> None:
        measurement = MaskMeasurement(
            measurement_id="a",
            frame_global=0,
            mask_id=10,
            tube_ids=[1, 2],
            inside_tube_ids=[1, 2],
            outside_visible_tube_ids=[3],
            mask_eroded_interior_flag_per_tube={1: True, 2: False},
        )
        self.assertEqual(_visible_negative_eroded_core_ids((1, 2, 3, 4), measurement), (1, 4))

    def test_canonical_clusters_for_core_splits_distant_temporal_core(self) -> None:
        by_id = {
            1: self._tube(1, (0.0, 0.0, 0.0)),
            2: self._tube(2, (0.1, 0.0, 0.0)),
            3: self._tube(3, (10.0, 0.0, 0.0)),
            4: self._tube(4, (10.1, 0.0, 0.0)),
        }
        clusters = _canonical_clusters_for_core((1, 2, 3, 4), by_id, max_clusters=2, min_tubes=2)
        self.assertEqual({tuple(cluster) for cluster in clusters}, {(1, 2), (3, 4)})

    def test_shuffle_d4rt_records_keeps_ids_and_moves_method_visible_fields(self) -> None:
        records = [
            self._tube(1, (1.0, 0.0, 0.0)),
            self._tube(2, (2.0, 0.0, 0.0)),
            self._tube(3, (3.0, 0.0, 0.0)),
        ]
        shuffled_a = _shuffle_d4rt_records_for_control(records, seed=11, scene="scene0000_00")
        shuffled_b = _shuffle_d4rt_records_for_control(records, seed=11, scene="scene0000_00")
        self.assertEqual([item.tube_id for item in shuffled_a], [1, 2, 3])
        self.assertEqual(
            [float(item.xyz_canonical[0, 0]) for item in shuffled_a],
            [float(item.xyz_canonical[0, 0]) for item in shuffled_b],
        )
        self.assertEqual(
            sorted(float(item.xyz_canonical[0, 0]) for item in shuffled_a),
            [1.0, 2.0, 3.0],
        )
        self.assertNotEqual(
            [float(item.xyz_canonical[0, 0]) for item in shuffled_a],
            [float(item.xyz_canonical[0, 0]) for item in records],
        )

    def test_filter_temporal_proposals_keeps_base_and_drops_high_cannot_link_temporal(self) -> None:
        base = Proposal(
            proposal_id="base",
            scene="s",
            frame_id=0,
            mask_id=1,
            proposal_type="R0_full_mask_region",
            core_tube_ids=(1, 2),
            fringe_tube_ids=(),
            boundary_tube_ids=(),
            region_area=2.0,
            features={"same_frame_cannot_link_rate": 999.0},
        )
        temporal_bad = Proposal(
            proposal_id="bad",
            scene="s",
            frame_id=0,
            mask_id=1,
            proposal_type="R10_temporal_tube_overlap_visible_negative_pruned_t20",
            core_tube_ids=(1, 2),
            fringe_tube_ids=(),
            boundary_tube_ids=(),
            region_area=2.0,
            features={"same_frame_cannot_link_rate": 6.0},
        )
        temporal_ok = Proposal(
            proposal_id="ok",
            scene="s",
            frame_id=0,
            mask_id=1,
            proposal_type="R10_temporal_tube_overlap_visible_negative_pruned_t20",
            core_tube_ids=(1, 2),
            fringe_tube_ids=(),
            boundary_tube_ids=(),
            region_area=2.0,
            features={"same_frame_cannot_link_rate": 5.0},
        )
        filtered = _filter_temporal_proposals([base, temporal_bad, temporal_ok], max_cannot_link_rate=5.0)
        self.assertEqual([item.proposal_id for item in filtered], ["base", "ok"])


if __name__ == "__main__":
    unittest.main()
