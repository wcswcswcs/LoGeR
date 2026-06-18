from __future__ import annotations

import unittest

from tools.run_v28_proposal_selection import (
    _selection_manifest_policy_fields,
    _rows_for_variant,
    _select_calibrated_ownership_expansion,
    _select_greedy,
    _select_local_search,
    _shuffle_candidate_memberships,
    _proposal_score,
)


class V28ProposalSelectionTests(unittest.TestCase):
    def test_selection_manifest_policy_fields_cover_plan_required_fields(self) -> None:
        fields = _selection_manifest_policy_fields()
        for key in {
            "is_method_result",
            "is_diagnostic_only",
            "forbidden_for_method_table",
            "uses_gt_for_prediction",
            "uses_gt_for_diagnostic_labels",
            "uses_rgbd_for_prediction",
            "uses_pose_for_prediction",
            "uses_scannet_mesh_for_prediction",
            "uses_eval_sim3_for_prediction",
            "uses_d4rt_self_sim3",
            "geometry_field",
            "coordinate_frame",
            "alignment_source",
        }:
            self.assertIn(key, fields)
        self.assertFalse(fields["forbidden_for_method_table"])
        self.assertFalse(fields["uses_gt_for_prediction"])
        self.assertFalse(fields["uses_rgbd_for_prediction"])
        self.assertFalse(fields["uses_pose_for_prediction"])
        self.assertFalse(fields["uses_eval_sim3_for_prediction"])

    def test_proposal_score_penalizes_cannot_link_and_visible_negative(self) -> None:
        clean = {
            "proposal_type": "R10_temporal_tube_overlap_visible_negative_pruned_t20",
            "num_core_tubes": 12,
            "eroded_interior_ratio": 0.8,
            "visibility_mean": 0.9,
            "confidence_mean": 0.9,
            "same_frame_cannot_link_rate": 0.0,
            "visible_outside_negative_rate": 0.0,
        }
        noisy = dict(clean)
        noisy["same_frame_cannot_link_rate"] = 10.0
        noisy["visible_outside_negative_rate"] = 10.0
        self.assertGreater(_proposal_score(clean), _proposal_score(noisy))

    def test_select_greedy_prefers_new_tube_gain_and_rejects_overlaps(self) -> None:
        rows = [
            {
                "proposal_id": "a",
                "proposal_type": "R10_temporal_tube_overlap_visible_negative_pruned_t20",
                "_core_tube_ids": [1, 2, 3, 4],
                "num_core_tubes": 4,
                "eroded_interior_ratio": 0.9,
            },
            {
                "proposal_id": "b",
                "proposal_type": "R10_temporal_tube_overlap_visible_negative_pruned_t20",
                "_core_tube_ids": [1, 2, 3, 5],
                "num_core_tubes": 4,
                "eroded_interior_ratio": 0.8,
            },
            {
                "proposal_id": "c",
                "proposal_type": "R1_boundary_eroded_interior",
                "_core_tube_ids": [6, 7, 8],
                "num_core_tubes": 3,
                "eroded_interior_ratio": 1.0,
            },
        ]
        selected = _select_greedy(rows, min_new_tubes=3, max_overlap_ratio=0.25, min_score=-999.0)
        self.assertEqual([row["proposal_id"] for row in selected], ["a", "c"])

    def test_rows_for_variant_separates_small_and_control_proposals(self) -> None:
        rows = [
            {"proposal_id": "full", "proposal_type": "R0_full_mask_region"},
            {"proposal_id": "water", "proposal_type": "R2_distance_watershed_region"},
            {"proposal_id": "seed", "proposal_type": "R3_d4rt_tube_seeded_voronoi"},
            {"proposal_id": "canon", "proposal_type": "R5_d4rt_canonical_adjacency_split"},
            {"proposal_id": "temp", "proposal_type": "R10_temporal_tube_overlap_visible_negative_pruned_t20"},
        ]
        self.assertEqual([row["proposal_id"] for row in _rows_for_variant(rows, "P2_watershed")], ["water"])
        self.assertEqual([row["proposal_id"] for row in _rows_for_variant(rows, "P3_d4rt_seeded")], ["seed", "canon"])
        self.assertEqual([row["proposal_id"] for row in _rows_for_variant(rows, "P8_shuffled_membership_control")], ["full", "water", "seed", "canon", "temp"])
        self.assertEqual([row["proposal_id"] for row in _rows_for_variant(rows, "P9_no_temporal_control")], ["full", "water", "seed", "canon"])
        self.assertEqual([row["proposal_id"] for row in _rows_for_variant(rows, "P11_calibrated_ownership_expansion")], ["full", "water", "seed", "canon", "temp"])

    def test_local_search_can_replace_overlapping_lower_objective_fragments(self) -> None:
        rows = [
            {
                "proposal_id": "broad",
                "proposal_type": "R10_temporal_tube_overlap_visible_negative_pruned_t20",
                "_core_tube_ids": [1, 2, 3, 4, 5, 6],
                "num_core_tubes": 6,
                "eroded_interior_ratio": 1.0,
                "visibility_mean": 1.0,
                "confidence_mean": 1.0,
            },
            {
                "proposal_id": "frag_a",
                "proposal_type": "R2_distance_watershed_region",
                "_core_tube_ids": [1, 2, 3],
                "num_core_tubes": 3,
                "eroded_interior_ratio": 0.2,
            },
            {
                "proposal_id": "frag_b",
                "proposal_type": "R2_distance_watershed_region",
                "_core_tube_ids": [4, 5, 6],
                "num_core_tubes": 3,
                "eroded_interior_ratio": 0.2,
            },
        ]
        selected = _select_local_search(
            rows,
            min_new_tubes=3,
            max_overlap_ratio=0.25,
            min_score=-999.0,
            score_kwargs={},
        )
        self.assertEqual([row["proposal_id"] for row in selected], ["broad"])

    def test_shuffle_candidate_memberships_is_deterministic_and_preserves_sizes(self) -> None:
        rows = [
            {
                "proposal_id": "a",
                "proposal_type": "R10_temporal_tube_overlap_visible_negative_pruned_t20",
                "_core_tube_ids": [1, 2, 3],
            },
            {
                "proposal_id": "b",
                "proposal_type": "R2_distance_watershed_region",
                "_core_tube_ids": [4, 5],
            },
        ]
        first = _shuffle_candidate_memberships(rows, seed=7, scene="scene0000_00")
        second = _shuffle_candidate_memberships(rows, seed=7, scene="scene0000_00")
        self.assertEqual([row["_core_tube_ids"] for row in first], [row["_core_tube_ids"] for row in second])
        self.assertEqual([len(row["_core_tube_ids"]) for row in first], [3, 2])
        self.assertEqual({tid for row in first for tid in row["_core_tube_ids"]}, {1, 2, 3, 4, 5})
        self.assertTrue(all(row["control_kind"] == "deterministic_tube_membership_shuffle_proxy" for row in first))

    def test_calibrated_ownership_expansion_adds_consensus_supported_tubes(self) -> None:
        rows = [
            {
                "proposal_id": "seed_a",
                "proposal_type": "R10_temporal_tube_overlap_visible_negative_pruned_t20",
                "_core_tube_ids": [1, 2, 3],
                "num_core_tubes": 3,
                "eroded_interior_ratio": 1.0,
                "visibility_mean": 1.0,
                "confidence_mean": 1.0,
            },
            {
                "proposal_id": "seed_b",
                "proposal_type": "R10_temporal_tube_overlap_visible_negative_pruned_t20",
                "_core_tube_ids": [10, 11, 12],
                "num_core_tubes": 3,
                "eroded_interior_ratio": 0.9,
                "visibility_mean": 1.0,
                "confidence_mean": 1.0,
            },
            {
                "proposal_id": "support_a1",
                "proposal_type": "R2_distance_watershed_region",
                "_core_tube_ids": [1, 2, 4],
                "num_core_tubes": 3,
                "eroded_interior_ratio": 0.9,
                "visibility_mean": 1.0,
                "confidence_mean": 1.0,
            },
            {
                "proposal_id": "support_a2",
                "proposal_type": "R3_d4rt_tube_seeded_voronoi",
                "_core_tube_ids": [1, 3, 4],
                "num_core_tubes": 3,
                "eroded_interior_ratio": 0.9,
                "visibility_mean": 1.0,
                "confidence_mean": 1.0,
            },
            {
                "proposal_id": "ambiguous",
                "proposal_type": "R3_d4rt_tube_seeded_voronoi",
                "_core_tube_ids": [2, 10, 99],
                "num_core_tubes": 3,
                "eroded_interior_ratio": 0.9,
                "visibility_mean": 1.0,
                "confidence_mean": 1.0,
            },
        ]
        selected = _select_calibrated_ownership_expansion(
            rows,
            min_new_tubes=3,
            seed_max_overlap_ratio=0.10,
            seed_min_score=-999.0,
            expand_min_score=-999.0,
            expand_min_overlap_ratio=0.50,
            expand_min_votes=1,
            expand_margin=1.25,
            max_expanded_core_ratio=2.0,
            score_kwargs={},
        )
        cores = {row["proposal_id"].split("_p11own")[0]: set(row["_core_tube_ids"]) for row in selected}
        self.assertIn(4, cores["seed_a"])
        self.assertNotIn(99, {tid for core in cores.values() for tid in core})
        self.assertTrue(all(row["selection_transform"] == "calibrated_ownership_expansion" for row in selected))


if __name__ == "__main__":
    unittest.main()
