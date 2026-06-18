from __future__ import annotations

from types import SimpleNamespace
import unittest

from stream4d_native.part_gated_alignment import (
    TubeRoleAlignmentInfo,
    build_alignment_edge_evidence,
    evaluate_part_gated_alignment,
    select_alignment_variant,
)
from stream4d_native.semantic_material_part_graph import TokenMaterialSupport


class V42PartGatedAlignmentTests(unittest.TestCase):
    def _support(self) -> dict[int, TokenMaterialSupport]:
        return {
            0: TokenMaterialSupport(0, 0, 1, (2, 3), (), ()),
            1: TokenMaterialSupport(1, 1, 1, (3,), (), ()),
            2: TokenMaterialSupport(2, 0, 2, (2,), (), ()),
            3: TokenMaterialSupport(3, 1, 3, (4,), (), ()),
        }

    def _roles(self) -> dict[int, TubeRoleAlignmentInfo]:
        return {
            1: TubeRoleAlignmentInfo(1, "scene", 0.01, 0.10, scene_role_weight=0.8),
            2: TubeRoleAlignmentInfo(2, "object", 0.04, 0.90, object_role_weight=0.8),
            3: TubeRoleAlignmentInfo(3, "part", 0.02, 0.80, part_role_weight=0.8),
            4: TubeRoleAlignmentInfo(4, "unknown", 0.07, 0.40, unknown_role_weight=0.9),
        }

    def _edges(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                token_i=0,
                token_j=1,
                object_affinity=0.65,
                semantic_affinity=0.75,
                diagnostic_same_gt=True,
                same_frame_cannot_link=False,
            ),
            SimpleNamespace(
                token_i=0,
                token_j=2,
                object_affinity=0.90,
                semantic_affinity=0.90,
                diagnostic_same_gt=False,
                same_frame_cannot_link=True,
            ),
            SimpleNamespace(
                token_i=0,
                token_j=3,
                object_affinity=0.80,
                semantic_affinity=0.80,
                diagnostic_same_gt=None,
                same_frame_cannot_link=False,
            ),
        ]

    def test_part_gated_correspondences_reduce_mismatch_vs_all_points(self) -> None:
        evidences = build_alignment_edge_evidence(self._edges(), self._support(), self._roles())
        result = evaluate_part_gated_alignment(evidences, part_gate_semantic_threshold=0.10)
        rows = {row["variant"]: row for row in result["variant_rows"]}
        self.assertGreater(rows["O1_same_object_all_points"]["part_mismatch_rate"], 0.0)
        self.assertEqual(rows["O3_semantic_part_gated"]["part_mismatch_rate"], 0.0)
        self.assertGreater(rows["O3_semantic_part_gated"]["part_mismatch_reduction_vs_O1"], 0.30)

    def test_dynamic_object_does_not_update_static_scene_pose(self) -> None:
        evidences = build_alignment_edge_evidence(self._edges(), self._support(), self._roles())
        result = evaluate_part_gated_alignment(evidences, part_gate_semantic_threshold=0.10)
        rows = {row["variant"]: row for row in result["variant_rows"]}
        self.assertGreater(rows["O2_material_tube_only"]["static_scene_dynamic_leakage_ratio"], 0.0)
        self.assertEqual(rows["O3_semantic_part_gated"]["static_scene_dynamic_leakage_ratio"], 0.0)
        self.assertTrue(rows["O3_semantic_part_gated"]["role_gated_static_scene_update"])

    def test_wrong_part_negative_control_is_rejected(self) -> None:
        evidences = build_alignment_edge_evidence(self._edges(), self._support(), self._roles())
        result = evaluate_part_gated_alignment(evidences, part_gate_semantic_threshold=0.10)
        rows = {row["variant"]: row for row in result["variant_rows"]}
        self.assertTrue(rows["O6_wrong_part_alignment_negative_control"]["negative_control_rejected"])
        self.assertGreater(rows["O6_wrong_part_alignment_negative_control"]["part_mismatch_rate"], 0.0)

    def test_ambiguous_unknown_tube_is_not_part_gated(self) -> None:
        evidences = build_alignment_edge_evidence(self._edges(), self._support(), self._roles())
        selected = select_alignment_variant(evidences, "O3_semantic_part_gated", part_gate_semantic_threshold=0.10)
        selected_pairs = {(edge.token_i, edge.token_j) for edge in selected}
        self.assertIn((0, 1), selected_pairs)
        self.assertNotIn((0, 3), selected_pairs)


if __name__ == "__main__":
    unittest.main()
