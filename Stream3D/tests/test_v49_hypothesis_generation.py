from __future__ import annotations

import unittest

from stream4d_native.v49_mosaic_stage1 import _hypothesis_generation_support_score, build_component_profiles


class TestV49HypothesisGeneration(unittest.TestCase):
    def test_component_profiles_keep_gt_diagnostic_only(self) -> None:
        rows = [
            {
                "scene": "s",
                "predicted_component_object_id": "c1",
                "mask_observation_id": "m1",
                "frame_id": "0",
                "diagnostic_gt_instance": "7",
                "supporting_carrier_observation_count": "2",
                "supporting_unique_carrier_count": "2",
            }
        ]
        profiles = build_component_profiles(rows, [])
        self.assertEqual(profiles["s|c1"]["diagnostic_dominant_gt"], "7")
        self.assertFalse(profiles["s|c1"]["uses_gt_for_prediction"])
        self.assertTrue(profiles["s|c1"]["uses_gt_for_diagnostic_labels"])

    def test_hypothesis_generation_support_penalizes_false_shared_mask_family(self) -> None:
        clean_pair_edge = {
            "candidate_generation_source": "H2_shared_mask_temporal_semantic_set",
            "component_set_candidate_source": "pair_edge",
            "hypothesis_size": 2,
            "semantic_set_score": 0.98,
            "context_overlap_proxy": 0.02,
            "hypothesis_conflict_rate": 0.0,
            "mask_support_score": 8,
            "mask_reliability_min": 0.75,
            "mask_reliability_mean": 0.75,
        }
        noisy_neighborhood = dict(
            clean_pair_edge,
            component_set_candidate_source="pair_neighborhood",
            semantic_set_score=0.80,
            context_overlap_proxy=0.50,
            mask_support_score=90,
            hypothesis_size=3,
        )
        self.assertGreater(
            _hypothesis_generation_support_score(clean_pair_edge),
            _hypothesis_generation_support_score(noisy_neighborhood),
        )


if __name__ == "__main__":
    unittest.main()
