from __future__ import annotations

from .material_tube_roles import MaterialTubeEvidence, TubeRoleScores


def filter_scene_anchor_tubes(
    evidences: list[MaterialTubeEvidence],
    roles: list[TubeRoleScores],
    *,
    min_scene_weight: float = 0.35,
) -> list[MaterialTubeEvidence]:
    role_by_id = {int(role.tube_id): role for role in roles}
    out = []
    for evidence in evidences:
        role = role_by_id.get(int(evidence.tube_id))
        if role is None:
            continue
        if role.role == "scene" and role.scene_role_weight >= float(min_scene_weight):
            out.append(evidence)
    return out


def filter_object_support_tubes(
    evidences: list[MaterialTubeEvidence],
    roles: list[TubeRoleScores],
    *,
    min_object_weight: float = 0.35,
) -> list[MaterialTubeEvidence]:
    role_by_id = {int(role.tube_id): role for role in roles}
    out = []
    for evidence in evidences:
        role = role_by_id.get(int(evidence.tube_id))
        if role is None:
            continue
        if role.role == "object" and role.object_role_weight >= float(min_object_weight):
            out.append(evidence)
    return out

