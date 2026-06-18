from __future__ import annotations

import unittest

from stream4d_native.object_field import ObjectFieldCandidate
from stream4d_native.semantic_material_inference import (
    SemanticMaterialInferenceConfig,
    TubeAttachmentScore,
    run_semantic_material_inference,
)
from tools.run_v41_1_native_support_metrics import _offset_labels


class V41SemanticMaterialConstraintTests(unittest.TestCase):
    def test_d4rt_tubes_cannot_birth_objects_and_ambiguous_tubes_remain_unknown(self) -> None:
        candidates = [
            ObjectFieldCandidate(0, (0, 1), (10,), 0.95, "semantic_masklet"),
            ObjectFieldCandidate(1, (2, 3), (11,), 0.90, "semantic_masklet"),
            ObjectFieldCandidate(2, (), (12,), 0.99, "d4rt_tube"),
        ]
        scores = [
            TubeAttachmentScore(10, 0, 0.90),
            TubeAttachmentScore(11, 1, 0.85),
            TubeAttachmentScore(12, 0, 0.60),
            TubeAttachmentScore(12, 1, 0.57),
        ]
        result = run_semantic_material_inference(
            candidates,
            scores,
            config=SemanticMaterialInferenceConfig(attach_threshold=0.50, attach_margin=0.10),
            diagnostic_metrics={"4D_ARI": 0.42, "purity": 0.87, "completeness": 0.52, "AP_bridge": None},
        )
        self.assertEqual(result.metrics["birth_from_d4rt_tube_count"], 0)
        self.assertEqual(result.metrics["rejected_forbidden_birth_candidate_count"], 1)
        self.assertEqual(result.tube_assignments[12], "unknown")
        self.assertTrue(result.constraint_audit["all_selected_have_semantic_birth"])
        self.assertEqual(result.constraint_audit["selected_forbidden_birth_count"], 0)
        self.assertEqual(result.constraint_audit["rejected_forbidden_birth_candidate_count"], 1)
        self.assertTrue(result.constraint_audit["ambiguous_tubes_remain_unknown"])
        self.assertLessEqual(result.metrics["predictions_per_scene"], 300)
        self.assertGreaterEqual(result.metrics["4D_ARI"], 0.40)
        self.assertGreaterEqual(result.metrics["purity"], 0.85)
        self.assertGreaterEqual(result.metrics["completeness"], 0.50)

    def test_duplicate_support_is_compacted_to_one_primary_field(self) -> None:
        candidates = [
            ObjectFieldCandidate(0, (0, 1, 2), (10,), 0.95),
            ObjectFieldCandidate(1, (0, 1, 2), (11,), 0.80),
        ]
        result = run_semantic_material_inference(candidates, [TubeAttachmentScore(10, 0, 0.90)])
        self.assertEqual(len(result.object_fields), 1)
        self.assertEqual(result.constraint_audit["duplicate_drop_count"], 1)
        self.assertTrue(result.constraint_audit["one_primary_field_per_object"])

    def test_material_overlap_penalty_is_opt_in(self) -> None:
        candidates = [
            ObjectFieldCandidate(0, (0,), (10, 11, 12), 0.95),
            ObjectFieldCandidate(1, (1,), (10, 11, 12), 0.90),
        ]
        scores = [TubeAttachmentScore(10, 0, 0.90), TubeAttachmentScore(11, 1, 0.90)]
        default_result = run_semantic_material_inference(candidates, scores)
        self.assertEqual(len(default_result.object_fields), 2)

        compact_result = run_semantic_material_inference(
            candidates,
            scores,
            config=SemanticMaterialInferenceConfig(duplicate_material_jaccard=0.90),
        )
        self.assertEqual(len(compact_result.object_fields), 1)
        self.assertEqual(compact_result.constraint_audit["material_duplicate_drop_count"], 1)

    def test_adaptive_attach_threshold_uses_no_gt_score_quantile(self) -> None:
        candidates = [
            ObjectFieldCandidate(0, (0,), (10,), 0.95),
            ObjectFieldCandidate(1, (1,), (11,), 0.90),
        ]
        scores = [
            TubeAttachmentScore(10, 0, 1.00),
            TubeAttachmentScore(11, 1, 0.70),
        ]
        result = run_semantic_material_inference(
            candidates,
            scores,
            config=SemanticMaterialInferenceConfig(
                attach_threshold=0.50,
                attach_margin=0.10,
                adaptive_attach_threshold=0.80,
                adaptive_attach_score_quantile=0.25,
                adaptive_attach_quantile_min=0.70,
            ),
        )
        self.assertTrue(result.constraint_audit["adaptive_attach_used"])
        self.assertAlmostEqual(result.constraint_audit["effective_attach_threshold"], 0.80)
        self.assertEqual(result.tube_assignments[10], 0)
        self.assertEqual(result.tube_assignments[11], "unknown")

    def test_scene_offset_avoids_unknown_label_collision(self) -> None:
        scene0_pred, _scene0_gt = _offset_labels(0, {1: 1_000_000}, {1: 7})
        scene1_pred, _scene1_gt = _offset_labels(1, {1: 0}, {1: 7})
        self.assertNotEqual(scene0_pred[1], scene1_pred[10_000_001])


if __name__ == "__main__":
    unittest.main()
