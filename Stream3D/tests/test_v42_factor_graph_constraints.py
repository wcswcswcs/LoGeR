from __future__ import annotations

import unittest

from stream4d_native.object_field_v42 import (
    V42ObjectField,
    V42ObjectFieldConstraintError,
    build_v42_object_fields_from_alignment_rows,
)
from stream4d_native.semantic_material_factor_graph import summarize_v42_factor_graph


class V42FactorGraphConstraintTests(unittest.TestCase):
    def test_d4rt_tube_cannot_birth_object(self) -> None:
        field = V42ObjectField(0, 0, (1,), (10,), 0.9, birth_source="d4rt_tube")
        with self.assertRaises(V42ObjectFieldConstraintError):
            field.validate()

    def test_object_field_requires_semantic_masklet_support(self) -> None:
        field = V42ObjectField(0, 0, (), (10,), 0.9)
        with self.assertRaises(V42ObjectFieldConstraintError):
            field.validate()

    def test_unknown_attachment_allowed_for_unassigned_tubes(self) -> None:
        fields = [V42ObjectField(0, 0, (1, 2), (10,), 0.95)]
        summary = summarize_v42_factor_graph(fields, all_tube_ids={10, 20})
        self.assertGreater(summary.unknown_tube_ratio, 0.0)
        self.assertEqual(summary.birth_from_d4rt_tube_count, 0)

    def test_dynamic_object_does_not_update_static_scene_pose(self) -> None:
        fields = [V42ObjectField(0, 0, (1, 2), (10,), 0.95, static_scene_update_weight=0.0)]
        summary = summarize_v42_factor_graph(fields, all_tube_ids={10})
        self.assertEqual(summary.dynamic_static_update_violation_count, 0)
        self.assertTrue(summary.phase6_proxy_constraints_pass)

    def test_one_object_one_primary_field_by_default(self) -> None:
        rows = [
            {
                "token_i": "1",
                "token_j": "2",
                "shared_tube_ids": "[10, 11]",
                "residual_proxy": "0.01",
                "selected_O4_semantic_part_gated_robust_trim": "True",
            },
            {
                "token_i": "2",
                "token_j": "3",
                "shared_tube_ids": "[11, 12]",
                "residual_proxy": "0.02",
                "selected_O4_semantic_part_gated_robust_trim": "True",
            },
        ]
        fields = build_v42_object_fields_from_alignment_rows(rows)
        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].primary_field_id, 0)
        self.assertEqual(fields[0].semantic_masklet_ids, (1, 2, 3))
        self.assertEqual(fields[0].attached_tube_ids, (10, 11, 12))

    def test_safe_part_merge_adds_high_confidence_semantic_edges(self) -> None:
        rows = [
            {
                "token_i": "1",
                "token_j": "2",
                "shared_tube_ids": "[10]",
                "residual_proxy": "0.01",
                "selected_O2_material_tube_only": "True",
                "same_frame_cannot_link": "False",
                "role_conflict": "False",
                "semantic_affinity": "0.95",
                "object_affinity": "0.70",
                "visible_outside_conflict_ratio": "0.0",
            },
            {
                "token_i": "2",
                "token_j": "3",
                "shared_tube_ids": "[]",
                "residual_proxy": "0.20",
                "selected_O2_material_tube_only": "False",
                "same_frame_cannot_link": "False",
                "role_conflict": "False",
                "semantic_affinity": "0.92",
                "object_affinity": "0.55",
                "visible_outside_conflict_ratio": "0.1",
            },
            {
                "token_i": "3",
                "token_j": "4",
                "shared_tube_ids": "[]",
                "residual_proxy": "0.20",
                "selected_O2_material_tube_only": "False",
                "same_frame_cannot_link": "True",
                "role_conflict": "False",
                "semantic_affinity": "0.99",
                "object_affinity": "0.99",
                "visible_outside_conflict_ratio": "0.0",
            },
        ]
        fields = build_v42_object_fields_from_alignment_rows(
            rows,
            selected_column="selected_O2_material_tube_only",
            safe_merge_semantic_affinity=0.90,
            safe_merge_object_affinity=0.50,
        )
        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].semantic_masklet_ids, (1, 2, 3))

    def test_material_union_cap_prevents_global_material_glue(self) -> None:
        rows = [
            {
                "token_i": "1",
                "token_j": "2",
                "shared_tube_ids": "[10]",
                "residual_proxy": "0.01",
                "selected_O2_material_tube_only": "True",
                "material_union_count": "20",
            },
            {
                "token_i": "2",
                "token_j": "3",
                "shared_tube_ids": "[11]",
                "residual_proxy": "0.01",
                "selected_O2_material_tube_only": "True",
                "material_union_count": "999",
            },
        ]
        fields = build_v42_object_fields_from_alignment_rows(
            rows,
            selected_column="selected_O2_material_tube_only",
            max_material_union_count=320,
        )
        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].semantic_masklet_ids, (1, 2))


if __name__ == "__main__":
    unittest.main()
