from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .object_field import ObjectField
from .part_gated_alignment import TubeRoleAlignmentInfo
from .semantic_material_part_graph import TokenMaterialSupport


@dataclass(frozen=True)
class V42MemoryExpansionConfig:
    min_inside_count: int = 1
    max_outside_visible_count: int | None = None
    always_include_roles: tuple[str, ...] = ("object", "part")
    include_unknown_role: bool = False
    include_scene_role: bool = True
    scene_min_weight: float = 0.35
    scene_max_unknown_weight: float = 0.65
    scene_max_residual: float = 0.008
    outside_count_penalty: float = 0.01


@dataclass(frozen=True)
class V42MemoryExpansionResult:
    object_fields: list[ObjectField]
    expansion_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class _TubeProposal:
    field_index: int
    object_id: int
    primary_field_id: int
    tube_id: int
    role: str
    reason: str
    inside_count: int
    outside_count: int
    role_weight: float
    field_confidence: float
    score: float
    role_info: TubeRoleAlignmentInfo


def _float_attr(obj: Any, name: str, default: float = 0.0) -> float:
    value = getattr(obj, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _role_weight(role: str, info: TubeRoleAlignmentInfo) -> float:
    if role == "scene":
        return _float_attr(info, "scene_role_weight")
    if role == "object":
        return _float_attr(info, "object_role_weight")
    if role == "part":
        return _float_attr(info, "part_role_weight")
    if role == "unknown":
        return max(0.0, 1.0 - _float_attr(info, "unknown_role_weight"))
    return 0.0


def _role_info_row(info: TubeRoleAlignmentInfo | None) -> dict[str, Any]:
    if info is None:
        return {
            "role": "missing_role_evidence",
            "scene_role_weight": None,
            "object_role_weight": None,
            "part_role_weight": None,
            "unknown_role_weight": None,
            "residual_proxy": None,
            "object_masklet_consistency": None,
        }
    return {
        "role": str(info.role),
        "scene_role_weight": float(info.scene_role_weight),
        "object_role_weight": float(info.object_role_weight),
        "part_role_weight": float(info.part_role_weight),
        "unknown_role_weight": float(info.unknown_role_weight),
        "residual_proxy": float(info.residual_proxy),
        "object_masklet_consistency": float(info.object_masklet_consistency),
    }


def _base_row(field: ObjectField, tube_id: int, info: TubeRoleAlignmentInfo | None) -> dict[str, Any]:
    return {
        "object_id": int(field.object_id),
        "primary_field_id": int(field.primary_field_id),
        "tube_id": int(tube_id),
        "action": "keep",
        "source": "base_object_field",
        "reason": "base_attached_field_tube",
        "inside_count": None,
        "outside_count": None,
        "score": None,
        **_role_info_row(info),
    }


def _candidate_row(
    field: ObjectField,
    tube_id: int,
    *,
    action: str,
    reason: str,
    inside_count: int,
    outside_count: int,
    score: float | None,
    info: TubeRoleAlignmentInfo | None,
    owner_object_id: int | None = None,
) -> dict[str, Any]:
    row = {
        "object_id": int(field.object_id),
        "primary_field_id": int(field.primary_field_id),
        "tube_id": int(tube_id),
        "action": action,
        "source": "streaming_memory_token_support",
        "reason": reason,
        "inside_count": int(inside_count),
        "outside_count": int(outside_count),
        "score": score,
        **_role_info_row(info),
    }
    if owner_object_id is not None:
        row["owner_object_id"] = int(owner_object_id)
    return row


def _support_counts(
    field: ObjectField,
    support_by_token: dict[int, TokenMaterialSupport],
) -> tuple[Counter[int], Counter[int]]:
    inside: Counter[int] = Counter()
    outside: Counter[int] = Counter()
    for token_id in field.semantic_masklet_ids:
        support = support_by_token.get(int(token_id))
        if support is None:
            continue
        for tube_id in getattr(support, "inside_tube_ids", ()):
            inside[int(tube_id)] += 1
        for tube_id in getattr(support, "outside_visible_tube_ids", ()):
            outside[int(tube_id)] += 1
    return inside, outside


def _allowed_reason(
    info: TubeRoleAlignmentInfo | None,
    *,
    inside_count: int,
    outside_count: int,
    config: V42MemoryExpansionConfig,
) -> tuple[bool, str, str, float]:
    if int(inside_count) < int(config.min_inside_count):
        return False, "below_min_inside_count", "missing_role_evidence", 0.0
    if config.max_outside_visible_count is not None and int(outside_count) > int(config.max_outside_visible_count):
        return False, "outside_visible_conflict", "missing_role_evidence", 0.0
    if info is None:
        return False, "missing_role_evidence", "missing_role_evidence", 0.0
    role = str(info.role)
    role_weight = _role_weight(role, info)
    if role in set(config.always_include_roles):
        return True, f"{role}_role_support", role, role_weight
    if role == "unknown" and bool(config.include_unknown_role):
        return True, "unknown_role_support", role, role_weight
    if role == "scene" and bool(config.include_scene_role):
        if (
            float(info.scene_role_weight) >= float(config.scene_min_weight)
            and float(info.unknown_role_weight) <= float(config.scene_max_unknown_weight)
            and float(info.residual_proxy) <= float(config.scene_max_residual)
        ):
            return True, "scene_role_confidence_gate", role, role_weight
        return False, "scene_role_confidence_rejected", role, role_weight
    return False, f"{role}_role_rejected", role, role_weight


def _proposal_sort_key(proposal: _TubeProposal) -> tuple[float, int, float, float, int, int]:
    return (
        -float(proposal.score),
        -int(proposal.inside_count),
        -float(proposal.role_weight),
        -float(proposal.field_confidence),
        int(proposal.object_id),
        int(proposal.tube_id),
    )


def expand_v42_object_fields_with_token_support(
    fields: list[ObjectField],
    support_by_token: dict[int, TokenMaterialSupport],
    role_by_tube: dict[int, TubeRoleAlignmentInfo],
    *,
    config: V42MemoryExpansionConfig | None = None,
) -> V42MemoryExpansionResult:
    cfg = config or V42MemoryExpansionConfig()
    rows: list[dict[str, Any]] = []
    proposals: list[_TubeProposal] = []
    claimed_by_tube: dict[int, int] = {}
    claimed_object_by_tube: dict[int, int] = {}

    for field_index, field in enumerate(fields):
        if not field.semantic_masklet_ids:
            continue
        for tube_id in sorted({int(v) for v in field.attached_tube_ids}):
            claimed_by_tube.setdefault(int(tube_id), int(field_index))
            claimed_object_by_tube.setdefault(int(tube_id), int(field.object_id))
            rows.append(_base_row(field, int(tube_id), role_by_tube.get(int(tube_id))))

    for field_index, field in enumerate(fields):
        if not field.semantic_masklet_ids:
            continue
        base_tubes = {int(v) for v in field.attached_tube_ids}
        inside_counts, outside_counts = _support_counts(field, support_by_token)
        for tube_id, inside_count in sorted(inside_counts.items()):
            if int(tube_id) in base_tubes:
                continue
            outside_count = int(outside_counts.get(int(tube_id), 0))
            info = role_by_tube.get(int(tube_id))
            allowed, reason, role, role_weight = _allowed_reason(
                info,
                inside_count=int(inside_count),
                outside_count=outside_count,
                config=cfg,
            )
            score = (
                float(inside_count)
                + float(role_weight)
                + 0.10 * float(field.confidence)
                - float(cfg.outside_count_penalty) * float(outside_count)
            )
            if not allowed:
                rows.append(
                    _candidate_row(
                        field,
                        int(tube_id),
                        action="reject",
                        reason=reason,
                        inside_count=int(inside_count),
                        outside_count=outside_count,
                        score=float(score),
                        info=info,
                    )
                )
                continue
            proposals.append(
                _TubeProposal(
                    field_index=int(field_index),
                    object_id=int(field.object_id),
                    primary_field_id=int(field.primary_field_id),
                    tube_id=int(tube_id),
                    role=role,
                    reason=reason,
                    inside_count=int(inside_count),
                    outside_count=outside_count,
                    role_weight=float(role_weight),
                    field_confidence=float(field.confidence),
                    score=float(score),
                    role_info=info,
                )
            )

    additions_by_field: dict[int, list[int]] = defaultdict(list)
    field_by_index = {idx: field for idx, field in enumerate(fields)}
    for proposal in sorted(proposals, key=_proposal_sort_key):
        owner_index = claimed_by_tube.get(int(proposal.tube_id))
        field = field_by_index[int(proposal.field_index)]
        if owner_index is not None and int(owner_index) != int(proposal.field_index):
            rows.append(
                _candidate_row(
                    field,
                    int(proposal.tube_id),
                    action="reject",
                    reason="duplicate_tube_claim_rejected",
                    inside_count=int(proposal.inside_count),
                    outside_count=int(proposal.outside_count),
                    score=float(proposal.score),
                    info=proposal.role_info,
                    owner_object_id=claimed_object_by_tube.get(int(proposal.tube_id)),
                )
            )
            continue
        if owner_index is None:
            claimed_by_tube[int(proposal.tube_id)] = int(proposal.field_index)
            claimed_object_by_tube[int(proposal.tube_id)] = int(proposal.object_id)
        additions_by_field[int(proposal.field_index)].append(int(proposal.tube_id))
        rows.append(
            _candidate_row(
                field,
                int(proposal.tube_id),
                action="add",
                reason=proposal.reason,
                inside_count=int(proposal.inside_count),
                outside_count=int(proposal.outside_count),
                score=float(proposal.score),
                info=proposal.role_info,
            )
        )

    expanded: list[ObjectField] = []
    for field_index, field in enumerate(fields):
        expanded_tubes = sorted({int(v) for v in field.attached_tube_ids} | set(additions_by_field.get(field_index, [])))
        expanded.append(
            ObjectField(
                object_id=int(field.object_id),
                primary_field_id=int(field.primary_field_id),
                semantic_masklet_ids=[int(v) for v in field.semantic_masklet_ids],
                attached_tube_ids=expanded_tubes,
                confidence=float(field.confidence),
                birth_state=str(field.birth_state),
            )
        )

    action_counts = Counter(str(row.get("action", "")) for row in rows)
    reason_counts = Counter(str(row.get("reason", "")) for row in rows)
    added_role_counts = Counter(str(row.get("role", "")) for row in rows if str(row.get("action", "")) == "add")
    base_tube_count = int(sum(len({int(v) for v in field.attached_tube_ids}) for field in fields))
    expanded_tube_count = int(sum(len({int(v) for v in field.attached_tube_ids}) for field in expanded))
    summary = {
        "input_object_field_count": int(len(fields)),
        "output_object_field_count": int(len(expanded)),
        "base_attached_tube_count": base_tube_count,
        "expanded_attached_tube_count": expanded_tube_count,
        "added_tube_count": int(action_counts.get("add", 0)),
        "rejected_candidate_count": int(action_counts.get("reject", 0)),
        "kept_base_tube_count": int(action_counts.get("keep", 0)),
        "unique_expanded_tube_count": int(len({int(v) for field in expanded for v in field.attached_tube_ids})),
        "action_counts": {str(k): int(v) for k, v in sorted(action_counts.items())},
        "reason_counts": {str(k): int(v) for k, v in sorted(reason_counts.items())},
        "added_role_counts": {str(k): int(v) for k, v in sorted(added_role_counts.items())},
        "config": {
            "min_inside_count": int(cfg.min_inside_count),
            "max_outside_visible_count": cfg.max_outside_visible_count,
            "always_include_roles": list(cfg.always_include_roles),
            "include_unknown_role": bool(cfg.include_unknown_role),
            "include_scene_role": bool(cfg.include_scene_role),
            "scene_min_weight": float(cfg.scene_min_weight),
            "scene_max_unknown_weight": float(cfg.scene_max_unknown_weight),
            "scene_max_residual": float(cfg.scene_max_residual),
            "outside_count_penalty": float(cfg.outside_count_penalty),
        },
        "uses_gt_for_prediction": False,
    }
    return V42MemoryExpansionResult(object_fields=expanded, expansion_rows=rows, summary=summary)
