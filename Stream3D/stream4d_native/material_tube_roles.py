from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class MaterialTubeEvidence:
    tube_id: int
    visibility: float
    confidence: float
    self_stitch_residual: float
    semantic_stability: float
    object_masklet_consistency: float
    motion_magnitude: float
    scale_proxy: float = 1.0


@dataclass(frozen=True)
class TubeRoleScores:
    tube_id: int
    scene_role_weight: float
    object_role_weight: float
    part_role_weight: float
    unknown_role_weight: float
    role: str


def infer_tube_role(
    evidence: MaterialTubeEvidence,
    *,
    residual_scale: float = 0.10,
    motion_scale: float = 0.20,
    threshold: float = 0.35,
    margin: float = 0.08,
    scene_dynamic_penalty: float = 0.50,
) -> TubeRoleScores:
    conf_vis = _clip01(evidence.visibility) * _clip01(evidence.confidence)
    residual_quality = _clip01(1.0 - float(evidence.self_stitch_residual) / max(float(residual_scale), 1e-6))
    static_motion = _clip01(1.0 - float(evidence.motion_magnitude) / max(float(motion_scale), 1e-6))
    dynamic_motion = _clip01(0.30 + float(evidence.motion_magnitude) / max(float(motion_scale), 1e-6))
    dynamic_penalty = _clip01(1.0 - float(scene_dynamic_penalty) * _clip01(evidence.object_masklet_consistency))
    scene = _clip01(conf_vis * residual_quality * _clip01(evidence.semantic_stability) * static_motion * dynamic_penalty)
    obj = _clip01(conf_vis * _clip01(evidence.object_masklet_consistency) * dynamic_motion)
    part = _clip01(conf_vis * _clip01(evidence.semantic_stability) * _clip01(evidence.object_masklet_consistency))
    best = max(scene, obj, part)
    top_two = sorted([scene, obj, part], reverse=True)[:2]
    ambiguity = 1.0 if len(top_two) >= 2 and abs(top_two[0] - top_two[1]) < float(margin) else 0.0
    unknown = _clip01((1.0 - best) + 0.25 * ambiguity)
    if best < float(threshold) or (ambiguity and unknown >= best):
        role = "unknown"
    elif scene == best and scene >= max(obj, part) + float(margin):
        role = "scene"
    elif obj == best and obj >= max(scene, part) + float(margin):
        role = "object"
    elif part == best and part >= max(scene, obj) + float(margin):
        role = "part"
    else:
        role = "unknown"
    return TubeRoleScores(
        tube_id=int(evidence.tube_id),
        scene_role_weight=float(scene),
        object_role_weight=float(obj),
        part_role_weight=float(part),
        unknown_role_weight=float(unknown),
        role=role,
    )


def infer_tube_roles(evidences: list[MaterialTubeEvidence], **kwargs: Any) -> list[TubeRoleScores]:
    return [infer_tube_role(evidence, **kwargs) for evidence in evidences]


def summarize_tube_roles(evidences: list[MaterialTubeEvidence], roles: list[TubeRoleScores]) -> dict[str, Any]:
    evidence_by_id = {int(e.tube_id): e for e in evidences}
    static_residuals = [
        float(evidence_by_id[r.tube_id].self_stitch_residual)
        for r in roles
        if r.role == "scene" and r.tube_id in evidence_by_id
    ]
    all_residuals = [float(e.self_stitch_residual) for e in evidences]
    object_consistency = [
        float(evidence_by_id[r.tube_id].object_masklet_consistency)
        for r in roles
        if r.role == "object" and r.tube_id in evidence_by_id
    ]
    rejected_consistency = [
        float(evidence_by_id[r.tube_id].object_masklet_consistency)
        for r in roles
        if r.role == "unknown" and r.tube_id in evidence_by_id
    ]
    part_consistency = [
        float(evidence_by_id[r.tube_id].object_masklet_consistency)
        for r in roles
        if r.role == "part" and r.tube_id in evidence_by_id
    ]
    role_probs = [
        float(sum(1 for r in roles if r.role == role) / max(len(roles), 1))
        for role in ("scene", "object", "part", "unknown")
    ]
    role_entropy = -sum(prob * np_log2(prob) for prob in role_probs if prob > 0.0)
    return {
        "tube_count": int(len(roles)),
        "scene_role_weight_mean": float(mean([r.scene_role_weight for r in roles])) if roles else 0.0,
        "object_role_weight_mean": float(mean([r.object_role_weight for r in roles])) if roles else 0.0,
        "part_role_weight_mean": float(mean([r.part_role_weight for r in roles])) if roles else 0.0,
        "unknown_role_ratio": float(sum(1 for r in roles if r.role == "unknown") / max(len(roles), 1)),
        "scene_role_count": int(sum(1 for r in roles if r.role == "scene")),
        "object_role_count": int(sum(1 for r in roles if r.role == "object")),
        "part_role_count": int(sum(1 for r in roles if r.role == "part")),
        "role_entropy": float(role_entropy),
        "self_stitch_residual_all_mean": float(mean(all_residuals)) if all_residuals else 0.0,
        "self_stitch_residual_scene_mean": float(mean(static_residuals)) if static_residuals else None,
        "object_support_consistency_object_mean": float(mean(object_consistency)) if object_consistency else None,
        "object_support_consistency_part_mean": float(mean(part_consistency)) if part_consistency else None,
        "object_support_consistency_unknown_mean": float(mean(rejected_consistency)) if rejected_consistency else None,
    }


def np_log2(value: float) -> float:
    import math

    return math.log(float(value), 2.0)
