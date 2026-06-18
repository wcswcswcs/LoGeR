from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .semantic_material_part_graph import TokenMaterialSupport


@dataclass(frozen=True)
class TubeRoleAlignmentInfo:
    tube_id: int
    role: str
    residual_proxy: float
    object_masklet_consistency: float
    scene_role_weight: float = 0.0
    object_role_weight: float = 0.0
    part_role_weight: float = 0.0
    unknown_role_weight: float = 0.0


@dataclass(frozen=True)
class AlignmentEdgeEvidence:
    token_i: int
    token_j: int
    object_affinity: float
    semantic_affinity: float
    diagnostic_same_gt: bool | None
    same_frame_cannot_link: bool
    shared_tube_ids: tuple[int, ...]
    scene_tube_ids: tuple[int, ...]
    object_tube_ids: tuple[int, ...]
    part_tube_ids: tuple[int, ...]
    unknown_tube_ids: tuple[int, ...]
    material_union_count: int
    visible_outside_conflict_ratio: float
    residual_proxy: float
    role_conflict: bool

    @property
    def shared_tube_count(self) -> int:
        return len(self.shared_tube_ids)

    @property
    def object_part_tube_count(self) -> int:
        return len(set(self.object_tube_ids) | set(self.part_tube_ids))

    @property
    def trusted_material_tube_count(self) -> int:
        return len(set(self.scene_tube_ids) | set(self.object_tube_ids) | set(self.part_tube_ids))


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def build_alignment_edge_evidence(
    semantic_edges: list[Any],
    support_by_token: dict[int, TokenMaterialSupport],
    role_by_tube: dict[int, TubeRoleAlignmentInfo],
    *,
    no_material_residual_scale: float = 0.25,
) -> list[AlignmentEdgeEvidence]:
    out: list[AlignmentEdgeEvidence] = []
    for edge in semantic_edges:
        token_i = int(_attr(edge, "token_i"))
        token_j = int(_attr(edge, "token_j"))
        left = support_by_token.get(token_i)
        right = support_by_token.get(token_j)
        left_inside = set(left.inside_tube_ids) if left else set()
        right_inside = set(right.inside_tube_ids) if right else set()
        left_outside = set(left.outside_visible_tube_ids) if left else set()
        right_outside = set(right.outside_visible_tube_ids) if right else set()
        shared = tuple(sorted(int(v) for v in (left_inside & right_inside)))
        union = left_inside | right_inside
        conflict = (left_inside & right_outside) | (right_inside & left_outside)
        conflict_ratio = float(len(conflict) / max(len(union), 1)) if union else 0.0
        roles = [role_by_tube.get(int(tube_id)) for tube_id in shared]
        known_roles = [role for role in roles if role is not None]
        residual_values = [float(role.residual_proxy) for role in known_roles]
        object_affinity = float(_attr(edge, "object_affinity", 0.0) or 0.0)
        if residual_values:
            residual_proxy = float(np.mean(np.asarray(residual_values, dtype=np.float64)))
        else:
            residual_proxy = float(no_material_residual_scale * max(0.0, 1.0 - object_affinity))
        scene_tubes = tuple(sorted(int(role.tube_id) for role in known_roles if role.role == "scene"))
        object_tubes = tuple(sorted(int(role.tube_id) for role in known_roles if role.role == "object"))
        part_tubes = tuple(sorted(int(role.tube_id) for role in known_roles if role.role == "part"))
        unknown_tubes = tuple(sorted(int(role.tube_id) for role in known_roles if role.role == "unknown"))
        out.append(
            AlignmentEdgeEvidence(
                token_i=token_i,
                token_j=token_j,
                object_affinity=object_affinity,
                semantic_affinity=float(_attr(edge, "semantic_affinity", object_affinity) or 0.0),
                diagnostic_same_gt=_as_bool_or_none(_attr(edge, "diagnostic_same_gt", None)),
                same_frame_cannot_link=bool(_attr(edge, "same_frame_cannot_link", False)),
                shared_tube_ids=shared,
                scene_tube_ids=scene_tubes,
                object_tube_ids=object_tubes,
                part_tube_ids=part_tubes,
                unknown_tube_ids=unknown_tubes,
                material_union_count=int(len(union)),
                visible_outside_conflict_ratio=conflict_ratio,
                residual_proxy=residual_proxy,
                role_conflict=bool(scene_tubes and (object_tubes or part_tubes)),
            )
        )
    return out


def select_alignment_variant(
    evidences: list[AlignmentEdgeEvidence],
    variant: str,
    *,
    semantic_threshold: float = 0.50,
    part_gate_semantic_threshold: float = 0.15,
    min_shared_tubes: int = 1,
    min_part_role_tubes: int = 1,
    max_visible_outside_conflict: float = 0.35,
    robust_trim_quantile: float = 0.80,
) -> list[AlignmentEdgeEvidence]:
    if variant == "O0_no_object_alignment":
        return [edge for edge in evidences if not edge.same_frame_cannot_link]
    if variant == "O1_same_object_all_points":
        return [edge for edge in evidences if edge.object_affinity >= float(semantic_threshold)]
    if variant == "O2_material_tube_only":
        return [edge for edge in evidences if edge.shared_tube_count >= int(min_shared_tubes)]
    if variant == "O3_semantic_part_gated":
        return [
            edge
            for edge in evidences
            if edge.object_affinity >= float(part_gate_semantic_threshold)
            and (
                edge.object_part_tube_count >= int(min_part_role_tubes)
                or edge.trusted_material_tube_count >= int(min_shared_tubes)
            )
            and edge.visible_outside_conflict_ratio <= float(max_visible_outside_conflict)
            and not edge.same_frame_cannot_link
            and not edge.role_conflict
        ]
    if variant == "O4_semantic_part_gated_robust_trim":
        base = select_alignment_variant(
            evidences,
            "O3_semantic_part_gated",
            semantic_threshold=semantic_threshold,
            part_gate_semantic_threshold=part_gate_semantic_threshold,
            min_shared_tubes=min_shared_tubes,
            min_part_role_tubes=min_part_role_tubes,
            max_visible_outside_conflict=max_visible_outside_conflict,
            robust_trim_quantile=robust_trim_quantile,
        )
        if not base:
            return []
        threshold = float(np.quantile(np.asarray([edge.residual_proxy for edge in base], dtype=np.float64), float(robust_trim_quantile)))
        return [edge for edge in base if edge.residual_proxy <= threshold]
    if variant == "O6_wrong_part_alignment_negative_control":
        return [
            edge
            for edge in evidences
            if edge.object_affinity >= float(part_gate_semantic_threshold)
            and (edge.same_frame_cannot_link or edge.diagnostic_same_gt is False)
        ]
    raise ValueError(f"unsupported alignment variant: {variant}")


def summarize_alignment_variant(
    variant: str,
    selected: list[AlignmentEdgeEvidence],
    *,
    residual_inlier_threshold: float = 0.10,
    role_gated_static_scene_update: bool,
) -> dict[str, Any]:
    residuals = np.asarray([edge.residual_proxy for edge in selected], dtype=np.float64)
    labeled = [edge for edge in selected if edge.diagnostic_same_gt is not None]
    mismatched = [edge for edge in labeled if edge.diagnostic_same_gt is False]
    cannot = [edge for edge in selected if edge.same_frame_cannot_link]
    role_conflicts = [edge for edge in selected if edge.role_conflict]
    shared_counts = [edge.shared_tube_count for edge in selected]
    object_part_counts = [edge.object_part_tube_count for edge in selected]
    trusted_counts = [edge.trusted_material_tube_count for edge in selected]
    if role_gated_static_scene_update:
        static_dynamic_leakage_values = [0.0 for _edge in selected]
    else:
        static_dynamic_leakage_values = [
            float(edge.object_part_tube_count / max(edge.shared_tube_count, 1)) for edge in selected
        ]
    mismatch_rate = float(len(mismatched) / max(len(labeled), 1))
    cannot_rate = float(len(cannot) / max(len(selected), 1))
    negative_control_rejected = None
    if variant == "O6_wrong_part_alignment_negative_control":
        negative_control_rejected = bool(mismatch_rate >= 0.50 or cannot_rate >= 0.50)
    return {
        "variant": variant,
        "selected_edge_count": int(len(selected)),
        "gt_labeled_edge_count": int(len(labeled)),
        "object_alignment_residual_proxy_median": float(np.median(residuals)) if residuals.size else None,
        "object_alignment_residual_proxy_p90": float(np.quantile(residuals, 0.90)) if residuals.size else None,
        "inlier_ratio": float(np.mean(residuals <= float(residual_inlier_threshold))) if residuals.size else 0.0,
        "part_mismatch_rate": mismatch_rate,
        "same_frame_cannot_link_rate": cannot_rate,
        "role_conflict_rate": float(len(role_conflicts) / max(len(selected), 1)),
        "shared_tube_count_mean": float(np.mean(np.asarray(shared_counts, dtype=np.float64))) if shared_counts else 0.0,
        "object_part_tube_count_mean": float(np.mean(np.asarray(object_part_counts, dtype=np.float64))) if object_part_counts else 0.0,
        "trusted_material_tube_count_mean": float(np.mean(np.asarray(trusted_counts, dtype=np.float64))) if trusted_counts else 0.0,
        "visible_outside_conflict_ratio_mean": float(np.mean([edge.visible_outside_conflict_ratio for edge in selected])) if selected else 0.0,
        "static_scene_dynamic_leakage_ratio": float(np.mean(np.asarray(static_dynamic_leakage_values, dtype=np.float64))) if selected else 0.0,
        "role_gated_static_scene_update": bool(role_gated_static_scene_update),
        "negative_control_rejected": negative_control_rejected,
    }


def evaluate_part_gated_alignment(
    evidences: list[AlignmentEdgeEvidence],
    *,
    semantic_threshold: float = 0.50,
    part_gate_semantic_threshold: float = 0.15,
    min_shared_tubes: int = 1,
    min_part_role_tubes: int = 1,
    max_visible_outside_conflict: float = 0.35,
    residual_inlier_threshold: float = 0.10,
    robust_trim_quantile: float = 0.80,
) -> dict[str, Any]:
    variants = [
        "O0_no_object_alignment",
        "O1_same_object_all_points",
        "O2_material_tube_only",
        "O3_semantic_part_gated",
        "O4_semantic_part_gated_robust_trim",
        "O6_wrong_part_alignment_negative_control",
    ]
    selected_by_variant = {
        variant: select_alignment_variant(
            evidences,
            variant,
            semantic_threshold=semantic_threshold,
            part_gate_semantic_threshold=part_gate_semantic_threshold,
            min_shared_tubes=min_shared_tubes,
            min_part_role_tubes=min_part_role_tubes,
            max_visible_outside_conflict=max_visible_outside_conflict,
            robust_trim_quantile=robust_trim_quantile,
        )
        for variant in variants
    }
    rows = [
        summarize_alignment_variant(
            variant,
            selected_by_variant[variant],
            residual_inlier_threshold=residual_inlier_threshold,
            role_gated_static_scene_update=variant in {"O3_semantic_part_gated", "O4_semantic_part_gated_robust_trim"},
        )
        for variant in variants
    ]
    row_by_variant = {str(row["variant"]): row for row in rows}
    o0_residual = row_by_variant["O0_no_object_alignment"]["object_alignment_residual_proxy_median"]
    o1_mismatch = float(row_by_variant["O1_same_object_all_points"]["part_mismatch_rate"])
    o6_rejected = bool(row_by_variant["O6_wrong_part_alignment_negative_control"]["negative_control_rejected"])
    for row in rows:
        residual = row["object_alignment_residual_proxy_median"]
        if o0_residual is None or residual is None or float(o0_residual) <= 0.0:
            row["residual_proxy_improvement_vs_O0"] = None
        else:
            row["residual_proxy_improvement_vs_O0"] = float((float(o0_residual) - float(residual)) / max(float(o0_residual), 1e-9))
        if o1_mismatch <= 0.0:
            row["part_mismatch_reduction_vs_O1"] = None
        else:
            row["part_mismatch_reduction_vs_O1"] = float((o1_mismatch - float(row["part_mismatch_rate"])) / max(o1_mismatch, 1e-9))
        row["phase5_proxy_gate_pass"] = bool(
            row["variant"] in {"O3_semantic_part_gated", "O4_semantic_part_gated_robust_trim"}
            and row["selected_edge_count"] > 0
            and row["residual_proxy_improvement_vs_O0"] is not None
            and float(row["residual_proxy_improvement_vs_O0"]) >= 0.15
            and row["part_mismatch_reduction_vs_O1"] is not None
            and float(row["part_mismatch_reduction_vs_O1"]) >= 0.30
            and o6_rejected
        )
    return {
        "variant_rows": rows,
        "selected_by_variant": selected_by_variant,
        "phase5_proxy_gate_pass": bool(
            any(row["phase5_proxy_gate_pass"] for row in rows if row["variant"] in {"O3_semantic_part_gated", "O4_semantic_part_gated_robust_trim"})
        ),
        "phase5_gate_pass": False,
        "phase5_gate_blocker": "alignment diagnostic uses correspondence residual proxy, not optimized 3D object transforms",
    }
