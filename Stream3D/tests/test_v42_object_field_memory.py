from __future__ import annotations

import unittest

from stream4d_native.object_field import ObjectField
from stream4d_native.object_field_memory_v42 import (
    V42MemoryExpansionConfig,
    expand_v42_object_fields_with_token_support,
)
from stream4d_native.part_gated_alignment import TubeRoleAlignmentInfo
from stream4d_native.semantic_material_part_graph import TokenMaterialSupport


def _support(token_id: int, inside: tuple[int, ...], outside: tuple[int, ...] = ()) -> TokenMaterialSupport:
    return TokenMaterialSupport(
        token_id=token_id,
        frame_id=0,
        mask_id=token_id,
        inside_tube_ids=inside,
        boundary_tube_ids=(),
        outside_visible_tube_ids=outside,
    )


def _role(
    tube_id: int,
    role: str,
    *,
    scene_w: float = 0.0,
    object_w: float = 0.0,
    part_w: float = 0.0,
    unknown_w: float = 0.0,
    residual: float = 0.0,
) -> TubeRoleAlignmentInfo:
    return TubeRoleAlignmentInfo(
        tube_id=tube_id,
        role=role,
        residual_proxy=residual,
        object_masklet_consistency=0.0,
        scene_role_weight=scene_w,
        object_role_weight=object_w,
        part_role_weight=part_w,
        unknown_role_weight=unknown_w,
    )


class V42ObjectFieldMemoryTests(unittest.TestCase):
    def test_expands_existing_semantic_field_without_birth(self) -> None:
        fields = [
            ObjectField(
                object_id=0,
                primary_field_id=0,
                semantic_masklet_ids=[1],
                attached_tube_ids=[10],
                confidence=0.9,
            )
        ]
        support = {1: _support(1, (10, 20, 30, 40, 50))}
        roles = {
            20: _role(20, "object", object_w=0.8),
            30: _role(30, "part", part_w=0.7),
            40: _role(40, "unknown", unknown_w=0.8),
            50: _role(50, "scene", scene_w=0.6, unknown_w=0.3, residual=0.004),
        }
        result = expand_v42_object_fields_with_token_support(
            fields,
            support,
            roles,
            config=V42MemoryExpansionConfig(include_unknown_role=True),
        )
        self.assertEqual(len(result.object_fields), 1)
        self.assertEqual(result.object_fields[0].semantic_masklet_ids, [1])
        self.assertEqual(result.object_fields[0].attached_tube_ids, [10, 20, 30, 40, 50])
        self.assertEqual(result.summary["added_tube_count"], 4)
        self.assertFalse(result.summary["uses_gt_for_prediction"])

    def test_scene_role_requires_confidence_gate(self) -> None:
        fields = [ObjectField(object_id=0, primary_field_id=0, semantic_masklet_ids=[1], attached_tube_ids=[])]
        support = {1: _support(1, (20, 21, 22))}
        roles = {
            20: _role(20, "scene", scene_w=0.6, unknown_w=0.3, residual=0.004),
            21: _role(21, "scene", scene_w=0.6, unknown_w=0.9, residual=0.004),
            22: _role(22, "scene", scene_w=0.6, unknown_w=0.3, residual=0.050),
        }
        result = expand_v42_object_fields_with_token_support(fields, support, roles)
        self.assertEqual(result.object_fields[0].attached_tube_ids, [20])
        reasons = [row["reason"] for row in result.expansion_rows if row["action"] == "reject"]
        self.assertEqual(reasons.count("scene_role_confidence_rejected"), 2)

    def test_conflicting_memory_tube_gets_single_owner(self) -> None:
        fields = [
            ObjectField(object_id=0, primary_field_id=0, semantic_masklet_ids=[1], attached_tube_ids=[], confidence=0.1),
            ObjectField(object_id=1, primary_field_id=1, semantic_masklet_ids=[2], attached_tube_ids=[], confidence=0.9),
        ]
        support = {
            1: _support(1, (20,)),
            2: _support(2, (20,)),
        }
        roles = {20: _role(20, "object", object_w=0.8)}
        result = expand_v42_object_fields_with_token_support(fields, support, roles)
        owners = [field.object_id for field in result.object_fields if 20 in field.attached_tube_ids]
        self.assertEqual(owners, [1])
        reject_rows = [row for row in result.expansion_rows if row["reason"] == "duplicate_tube_claim_rejected"]
        self.assertEqual(len(reject_rows), 1)
        self.assertEqual(reject_rows[0]["owner_object_id"], 1)


if __name__ == "__main__":
    unittest.main()
