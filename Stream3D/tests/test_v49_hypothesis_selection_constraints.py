from __future__ import annotations

import unittest

from stream4d_native.v49_mosaic_stage1 import (
    _greedy_select,
    _mask_reliability_scores,
    _score_hypothesis,
    build_component_proxy_feature_audit,
    build_hypothesis_selection,
)


class TestV49HypothesisSelectionConstraints(unittest.TestCase):
    def test_zero_mask_support_hypothesis_is_not_selected(self) -> None:
        rows = [
            {
                "hypothesis_id": "h0",
                "scene": "s",
                "components": "s|c0",
                "mask_support_score": 0,
                "hypothesis_conflict_rate": 0,
                "score_full": 100,
            }
        ]
        self.assertEqual(_greedy_select(rows, score_key="score_full", max_per_scene=10), [])

    def test_duplicate_component_not_selected_twice(self) -> None:
        rows = [
            {"hypothesis_id": "h0", "scene": "s", "components": "s|c0", "mask_support_score": 3, "hypothesis_conflict_rate": 0, "score_full": 2},
            {"hypothesis_id": "h1", "scene": "s", "components": "s|c0;;s|c1", "mask_support_score": 3, "hypothesis_conflict_rate": 0, "score_full": 1},
        ]
        selected = _greedy_select(rows, score_key="score_full", max_per_scene=10)
        self.assertEqual([row["hypothesis_id"] for row in selected], ["h0"])

    def test_source_prefix_cap_limits_expanded_candidates_per_scene(self) -> None:
        rows = [
            {
                "hypothesis_id": f"h{idx}",
                "scene": "s",
                "components": f"s|c{idx}",
                "mask_support_score": 3,
                "hypothesis_conflict_rate": 0,
                "component_set_candidate_source": "expanded_low_overlap_semantic_star",
                "score_full": 10 - idx,
            }
            for idx in range(3)
        ]
        selected = _greedy_select(rows, score_key="score_full", max_per_scene=10, source_prefix_caps={"expanded_": 1})
        self.assertEqual([row["hypothesis_id"] for row in selected], ["h0"])

    def test_guarded_score_penalizes_large_support_risk(self) -> None:
        base = {
            "hypothesis_id": "h0",
            "scene": "s",
            "components": "s|c0;;s|c1",
            "mask_support_score": 30,
            "temporal_support_score": 0.7,
            "semantic_set_score": 0.8,
            "hypothesis_d4rt_specific_score": 0.7,
            "hypothesis_conflict_rate": 0.0,
            "hypothesis_size": 2,
        }
        low_risk = _score_hypothesis(dict(base, large_support_risk=0.0))
        high_risk = _score_hypothesis(dict(base, large_support_risk=1.0))
        self.assertGreater(low_risk["score_guarded_full"], high_risk["score_guarded_full"])

    def test_mask_reliability_downweights_large_sparse_masks(self) -> None:
        rows = [
            {
                "mask_observation_id": "compact",
                "mask_area": "100",
                "bbox_x0": "0",
                "bbox_y0": "0",
                "bbox_x1": "20",
                "bbox_y1": "20",
                "support_density": "0.05",
                "visible_carrier_count": "12",
            },
            {
                "mask_observation_id": "large_sparse",
                "mask_area": "10000",
                "bbox_x0": "0",
                "bbox_y0": "0",
                "bbox_x1": "110",
                "bbox_y1": "110",
                "support_density": "0.0001",
                "visible_carrier_count": "1",
            },
            {
                "mask_observation_id": "mid",
                "mask_area": "500",
                "bbox_x0": "0",
                "bbox_y0": "0",
                "bbox_x1": "40",
                "bbox_y1": "40",
                "support_density": "0.01",
                "visible_carrier_count": "5",
            },
        ]
        scores = _mask_reliability_scores(rows)
        self.assertGreater(scores["compact"], scores["large_sparse"])

    def test_boundary_prototype_context_guard_penalizes_proxy_contradiction(self) -> None:
        base = {
            "hypothesis_id": "h0",
            "scene": "s",
            "components": "s|c0;;s|c1",
            "mask_support_score": 30,
            "temporal_support_score": 0.4,
            "semantic_set_score": 0.8,
            "hypothesis_d4rt_specific_score": 0.4,
            "hypothesis_conflict_rate": 0.0,
            "hypothesis_size": 2,
            "large_support_risk": 0.0,
            "mask_reliability_mean": 0.9,
            "mask_reliability_min": 0.9,
            "mask_reliability_range": 0.0,
        }
        clean = _score_hypothesis(
            dict(base, prototype_diversity=0.0, context_overlap_proxy=0.0, boundary_proxy_instability=0.0, component_feature_variance=0.0)
        )
        noisy = _score_hypothesis(
            dict(base, prototype_diversity=0.8, context_overlap_proxy=0.8, boundary_proxy_instability=1.0, component_feature_variance=1.0)
        )
        self.assertGreater(clean["score_boundary_prototype_context_hard"], noisy["score_boundary_prototype_context_hard"])

    def test_d4rt_completion_guard_penalizes_no_temporal_singletons(self) -> None:
        base = {
            "hypothesis_id": "h0",
            "scene": "s",
            "mask_support_score": 20,
            "semantic_set_score": 0.8,
            "hypothesis_conflict_rate": 0.0,
            "large_support_risk": 0.0,
            "mask_reliability_mean": 0.85,
            "mask_reliability_min": 0.85,
            "mask_reliability_range": 0.0,
            "prototype_diversity": 0.1,
            "context_overlap_proxy": 0.0,
            "boundary_proxy_instability": 0.0,
            "component_feature_variance": 0.0,
        }
        singleton = _score_hypothesis(
            dict(base, components="s|c0", hypothesis_size=1, temporal_support_score=0.0, hypothesis_d4rt_specific_score=0.0)
        )
        temporal_pair = _score_hypothesis(
            dict(base, components="s|c0;;s|c1", hypothesis_size=2, temporal_support_score=0.7, hypothesis_d4rt_specific_score=0.7)
        )
        self.assertEqual(singleton["no_temporal_explainable"], 1.0)
        self.assertGreater(temporal_pair["score_d4rt_completion_guard"], singleton["score_d4rt_completion_guard"])

    def test_persistent_prefilter_allows_only_low_overlap_pair_edge(self) -> None:
        base = {
            "hypothesis_id": "h0",
            "scene": "s",
            "components": "s|c0;;s|c1",
            "hypothesis_size": 2,
            "mask_support_score": 10,
            "semantic_set_score": 0.98,
            "hypothesis_conflict_rate": 0.0,
            "large_support_risk": 0.0,
            "mask_reliability_mean": 0.9,
            "mask_reliability_min": 0.9,
            "mask_reliability_range": 0.0,
            "prototype_diversity": 0.02,
            "context_overlap_proxy": 0.02,
            "boundary_proxy_instability": 0.0,
            "component_feature_variance": 0.0,
            "coverage_gain_over_singletons": 1,
            "hypothesis_d4rt_specific_score": 0.02,
        }
        allowed = _score_hypothesis(dict(base, component_set_candidate_source="pair_edge", temporal_support_score=0.02))
        disallowed_source = _score_hypothesis(dict(base, component_set_candidate_source="pair_neighborhood", temporal_support_score=0.02))
        disallowed_temporal = _score_hypothesis(dict(base, component_set_candidate_source="pair_edge", temporal_support_score=0.8))
        self.assertTrue(allowed["persistent_prefilter_pair_edge_low_overlap_ok"])
        self.assertFalse(disallowed_source["persistent_contradiction_prefilter_ok"])
        self.assertFalse(disallowed_temporal["persistent_contradiction_prefilter_ok"])
        self.assertGreater(allowed["score_persistent_contradiction_prefilter"], disallowed_source["score_persistent_contradiction_prefilter"])

    def test_component_proxy_feature_audit_reports_guard_scores(self) -> None:
        payload = build_component_proxy_feature_audit()
        scores = {row["score"] for row in payload["score_auc_rows"]}
        self.assertIn("score_boundary_prototype_context_hard", scores)
        self.assertIn("score_d4rt_completion_guard", scores)
        self.assertIn("score_persistent_contradiction_prefilter", scores)
        self.assertIn("negative_context_overlap_proxy", scores)
        self.assertFalse(payload["gate"]["dense_semantic_backend_claimed"])

    def test_selection_reports_guarded_variant_and_controls(self) -> None:
        payload = build_hypothesis_selection()
        variants = {row["solver_variant"] for row in payload["selection_rows"]}
        self.assertIn("O13_guarded_completion_selection", variants)
        self.assertIn("O14_guarded_no_D4RT_control", variants)
        self.assertIn("O15_guarded_no_temporal_control", variants)
        self.assertIn("O16_guarded_mask_only_control", variants)
        self.assertIn("O17_split_entropy_reliability_selection", variants)
        self.assertIn("O18_split_entropy_no_temporal_control", variants)
        self.assertIn("O19_split_entropy_mask_only_control", variants)
        self.assertIn("O21_boundary_prototype_context_selection", variants)
        self.assertIn("O22_boundary_prototype_no_temporal_control", variants)
        self.assertIn("O23_boundary_prototype_mask_only_control", variants)
        self.assertIn("O24_boundary_prototype_no_context_boundary_control", variants)
        self.assertIn("O25_d4rt_completion_guard_selection", variants)
        self.assertIn("O26_d4rt_completion_no_temporal_control", variants)
        self.assertIn("O27_d4rt_completion_mask_only_control", variants)
        self.assertIn("O28_d4rt_completion_no_specific_control", variants)
        self.assertIn("O29_persistent_contradiction_prefilter_selection", variants)
        self.assertIn("O30_persistent_prefilter_no_temporal_control", variants)
        self.assertIn("O31_persistent_prefilter_mask_only_control", variants)
        self.assertIn("O32_persistent_prefilter_no_source_guard_control", variants)
        self.assertIn("O33_expanded_partwhole_cap_selection", variants)
        self.assertIn("O34_expanded_partwhole_no_temporal_control", variants)
        self.assertIn("O35_expanded_partwhole_mask_only_control", variants)
        self.assertIn("O36_expanded_partwhole_no_source_bonus_control", variants)
        self.assertIn(payload["best_real_variant"], variants)


if __name__ == "__main__":
    unittest.main()
