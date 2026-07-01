from __future__ import annotations

from typing import Any


GT_GEOMETRY_INFERENCE_FLAGS = (
    "uses_gt_for_prediction",
    "uses_gt_geometry_for_prediction",
    "uses_gt_geometry_for_inference",
    "uses_gt_sim3_for_prediction",
    "uses_rgbd_for_prediction",
    "uses_pose_for_prediction",
    "uses_scannet_mesh_for_prediction",
    "uses_rgbd_pose_mesh_for_prediction",
    "alignment_used_for_prediction",
)

GT_GEOMETRY_EXPORT_OR_EVAL_FLAGS = (
    "uses_gt_for_evaluation",
    "uses_gt_for_diagnostic",
    "uses_gt_geometry_for_evaluation",
    "uses_gt_geometry_for_export",
    "uses_rgbd_for_evaluation",
    "uses_rgbd_for_evaluation_support",
    "uses_pose_for_evaluation",
    "uses_scannet_mesh_for_evaluation",
    "uses_rgbd_pose_mesh_for_export",
    "uses_scannet_mesh_for_export",
)

GT_GEOMETRY_TEXT_MARKERS = (
    "eval_sim3",
    "gt_sim3",
    "rgbd",
    "rgb-d",
    "pose_mesh",
    "scannet_mesh",
    "mesh_nearest",
    "mesh-nn",
)

GT_GEOMETRY_TEXT_FIELDS = (
    "mode",
    "geometry_source",
    "alignment_source",
    "support_policy",
    "eval_policy",
    "notes",
)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def gt_geometry_inference_reasons(record: dict[str, Any]) -> list[str]:
    """Return fields indicating GT geometry was used before/even during inference.

    Evaluation-only GT geometry is allowed by v65, but SOMA/method inference
    must not depend on GT depth, pose, ScanNet mesh, eval-Sim3, or RGB-D
    backprojection. This helper intentionally looks at explicit flags first and
    text markers only when the artifact claims to be a method result.
    """

    reasons = [flag for flag in GT_GEOMETRY_INFERENCE_FLAGS if bool_value(record.get(flag))]
    if bool_value(record.get("is_method_result")):
        for field in GT_GEOMETRY_TEXT_FIELDS:
            text = str(record.get(field, "")).lower()
            if any(marker in text for marker in GT_GEOMETRY_TEXT_MARKERS):
                reasons.append(f"{field}_mentions_gt_geometry")
    return sorted(set(reasons))


def gt_geometry_eval_or_export_reasons(record: dict[str, Any]) -> list[str]:
    return [flag for flag in GT_GEOMETRY_EXPORT_OR_EVAL_FLAGS if bool_value(record.get(flag))]


def method_table_forbidden_reasons(record: dict[str, Any]) -> list[str]:
    return sorted(set(gt_geometry_inference_reasons(record) + gt_geometry_eval_or_export_reasons(record)))


def normalize_reportability(record: dict[str, Any], *, context: str = "artifact") -> dict[str, Any]:
    """Normalize diagnostic/reportability flags and reject invalid method rows."""

    payload = dict(record)
    reasons = method_table_forbidden_reasons(payload)
    if reasons:
        payload["forbidden_for_method_table"] = True
        payload["is_diagnostic_only"] = True
    if bool_value(payload.get("is_method_result")) and reasons:
        raise ValueError(
            f"{context} cannot be a SOMA/method inference result because it uses GT geometry: "
            + ", ".join(reasons)
        )
    return payload


def policy_violation_reasons(record: dict[str, Any]) -> list[str]:
    reasons = method_table_forbidden_reasons(record)
    violations: list[str] = []
    if bool_value(record.get("is_method_result")) and reasons:
        violations.append("method_result_uses_gt_geometry")
    if reasons and not bool_value(record.get("forbidden_for_method_table")):
        violations.append("gt_geometry_not_forbidden_for_method_table")
    if gt_geometry_inference_reasons(record) and not bool_value(record.get("is_diagnostic_only")):
        violations.append("gt_geometry_inference_not_marked_diagnostic")
    return violations


__all__ = [
    "GT_GEOMETRY_EXPORT_OR_EVAL_FLAGS",
    "GT_GEOMETRY_INFERENCE_FLAGS",
    "bool_value",
    "gt_geometry_eval_or_export_reasons",
    "gt_geometry_inference_reasons",
    "method_table_forbidden_reasons",
    "normalize_reportability",
    "policy_violation_reasons",
]
