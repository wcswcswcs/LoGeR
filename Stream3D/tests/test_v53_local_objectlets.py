from __future__ import annotations

import unittest

from stream4d_native.v53_local_objectlets import (
    _select_objectlets,
    _select_objectlets_dynamic_uncovered_gain,
    weighted_partition_metrics,
)


class V53LocalObjectletsTest(unittest.TestCase):
    def test_weighted_partition_metrics_separates_perfect_and_merged(self) -> None:
        perfect = weighted_partition_metrics([("p1", "a", 2.0), ("p2", "b", 2.0)])
        merged = weighted_partition_metrics([("p", "a", 2.0), ("p", "b", 2.0)])
        self.assertAlmostEqual(perfect["purity"], 1.0)
        self.assertAlmostEqual(perfect["completeness"], 1.0)
        self.assertLess(merged["purity"], perfect["purity"])

    def test_dynamic_uncovered_gain_can_accept_overlap_when_new_gain_is_positive(self) -> None:
        first_components = [f"c{i}" for i in range(22)]
        second_components = [f"c{i}" for i in range(20)] + ["c_new"]
        candidates = [
            {
                "candidate_id": "a",
                "scene": "s",
                "chunk_id": "ch",
                "source_mask_observation_id": "m0",
                "candidate_source": "R0",
                "component_ids": str(first_components).replace("'", '"'),
                "candidate_success_rate": 1.0,
                "outside_all_related_masks_ratio_mean": 0.0,
                "same_frame_exclusion_violation_rate": 0.0,
            },
            {
                "candidate_id": "b",
                "scene": "s",
                "chunk_id": "ch",
                "source_mask_observation_id": "m1",
                "candidate_source": "R0",
                "component_ids": str(second_components).replace("'", '"'),
                "candidate_success_rate": 1.0,
                "outside_all_related_masks_ratio_mean": 0.0,
                "same_frame_exclusion_violation_rate": 0.0,
            },
        ]
        component_to_object, object_rows = _select_objectlets_dynamic_uncovered_gain(
            candidates,
            [],
            {},
            variant="L11_test",
            duplicate_penalty=0.01,
            outside_weight=1.0,
            conflict_weight=1.0,
            object_penalty=0.0,
        )
        self.assertIn("c_new", component_to_object)
        self.assertEqual(len(object_rows), 2)
        self.assertEqual(object_rows[-1]["duplicate_component_count"], 20)

    def test_repeated_signature_sort_mode_prioritizes_r5_candidates(self) -> None:
        candidates = [
            {
                "candidate_id": "broad",
                "candidate_source": "R0_single_representative_mask",
                "scene": "s",
                "chunk_id": "ch",
                "source_mask_observation_id": "m0",
                "component_ids": '["c1", "c2", "c3"]',
                "candidate_success_rate": 1.0,
                "outside_all_related_masks_ratio_mean": 0.0,
                "same_frame_exclusion_violation_rate": 0.0,
            },
            {
                "candidate_id": "sig",
                "candidate_source": "R5_repeated_support_signature",
                "scene": "s",
                "chunk_id": "ch",
                "source_mask_observation_id": "m1",
                "component_ids": '["c1", "c2"]',
                "candidate_success_rate": 1.0,
                "outside_all_related_masks_ratio_mean": 0.0,
                "same_frame_exclusion_violation_rate": 0.0,
                "repeated_support_signature_len": "5",
            },
        ]
        _component_to_object, object_rows = _select_objectlets(
            candidates,
            [],
            {},
            variant="L12_test",
            max_components_per_objectlet=None,
            min_new_component_ratio=0.25,
            sort_mode="repeated_signature_first",
        )
        self.assertEqual(object_rows[0]["candidate_id"], "sig")


if __name__ == "__main__":
    unittest.main()
