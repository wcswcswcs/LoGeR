from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .object_field_v42 import V42ObjectField


@dataclass(frozen=True)
class V42FactorGraphSummary:
    object_field_count: int
    birth_from_d4rt_tube_count: int
    mean_predictions_per_scene: float
    duplicate_rate: float
    conflict_rate: float
    unknown_tube_ratio: float
    dynamic_static_update_violation_count: int
    one_primary_field_per_object: bool
    phase6_proxy_constraints_pass: bool
    phase6_gate_pass: bool
    phase6_gate_blocker: str


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 0.0
    return float(len(left & right) / max(len(left | right), 1))


def summarize_v42_factor_graph(
    fields: list[V42ObjectField],
    *,
    all_tube_ids: set[int],
    scene_count: int = 1,
    duplicate_jaccard_threshold: float = 0.75,
) -> V42FactorGraphSummary:
    for field in fields:
        field.validate()
    forbidden_birth = int(sum(1 for field in fields if field.birth_source != "semantic_part_graph"))
    attached: set[int] = {int(tube_id) for field in fields for tube_id in field.attached_tube_ids}
    unknown = set(int(tube_id) for tube_id in all_tube_ids) - attached
    duplicate_pairs = 0
    total_pairs = 0
    conflict_tokens: set[int] = set()
    seen_tokens: dict[int, int] = {}
    for idx, left in enumerate(fields):
        left_tokens = set(int(v) for v in left.semantic_masklet_ids)
        for token_id in left_tokens:
            if token_id in seen_tokens:
                conflict_tokens.add(token_id)
            seen_tokens[token_id] = int(left.object_id)
        for right in fields[idx + 1 :]:
            total_pairs += 1
            if _jaccard(left_tokens, set(int(v) for v in right.semantic_masklet_ids)) >= float(duplicate_jaccard_threshold):
                duplicate_pairs += 1
    duplicate_rate = float(duplicate_pairs / max(total_pairs, 1))
    semantic_token_count = sum(len(field.semantic_masklet_ids) for field in fields)
    conflict_rate = float(len(conflict_tokens) / max(semantic_token_count, 1))
    one_primary = len({int(field.primary_field_id) for field in fields}) == len(fields)
    dynamic_static_violations = int(sum(1 for field in fields if float(field.static_scene_update_weight) > 0.0))
    proxy_pass = bool(
        forbidden_birth == 0
        and float(len(fields) / max(int(scene_count), 1)) <= 300.0
        and duplicate_rate <= 0.10
        and conflict_rate <= 0.15
        and dynamic_static_violations == 0
        and one_primary
    )
    return V42FactorGraphSummary(
        object_field_count=int(len(fields)),
        birth_from_d4rt_tube_count=forbidden_birth,
        mean_predictions_per_scene=float(len(fields) / max(int(scene_count), 1)),
        duplicate_rate=duplicate_rate,
        conflict_rate=conflict_rate,
        unknown_tube_ratio=float(len(unknown) / max(len(all_tube_ids), 1)),
        dynamic_static_update_violation_count=dynamic_static_violations,
        one_primary_field_per_object=bool(one_primary),
        phase6_proxy_constraints_pass=proxy_pass,
        phase6_gate_pass=False,
        phase6_gate_blocker="diagnostic factor graph has no full 4D ARI/purity/completeness/AP bridge evaluation",
    )


def summary_row(summary: V42FactorGraphSummary, *, scene: str, variant: str, source: str) -> dict[str, Any]:
    return {
        "scene": scene,
        "variant": variant,
        "source": source,
        "object_field_count": int(summary.object_field_count),
        "birth_from_d4rt_tube_count": int(summary.birth_from_d4rt_tube_count),
        "mean_predictions_per_scene": float(summary.mean_predictions_per_scene),
        "duplicate_rate": float(summary.duplicate_rate),
        "conflict_rate": float(summary.conflict_rate),
        "unknown_tube_ratio": float(summary.unknown_tube_ratio),
        "dynamic_static_update_violation_count": int(summary.dynamic_static_update_violation_count),
        "one_primary_field_per_object": bool(summary.one_primary_field_per_object),
        "phase6_proxy_constraints_pass": bool(summary.phase6_proxy_constraints_pass),
        "phase6_gate_pass": bool(summary.phase6_gate_pass),
        "phase6_gate_blocker": summary.phase6_gate_blocker,
    }
