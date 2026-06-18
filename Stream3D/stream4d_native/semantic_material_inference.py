from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import mean
from typing import Any

from .compact_object_field_selection import select_compact_object_fields
from .object_field import ObjectField, ObjectFieldCandidate


@dataclass(frozen=True)
class TubeAttachmentScore:
    tube_id: int
    object_id: int
    score: float


@dataclass(frozen=True)
class SemanticMaterialInferenceConfig:
    attach_threshold: float = 0.50
    attach_margin: float = 0.10
    max_fields: int = 300
    duplicate_support_jaccard: float = 0.75
    duplicate_material_jaccard: float = 1.01
    adaptive_attach_threshold: float = 0.0
    adaptive_attach_score_quantile: float = 0.25
    adaptive_attach_quantile_min: float = 1.01


@dataclass
class SemanticMaterialInferenceResult:
    object_fields: list[ObjectField]
    tube_assignments: dict[int, int | str]
    constraint_audit: dict[str, Any]
    metrics: dict[str, Any]


def _attach_tubes(
    selected: list[ObjectFieldCandidate],
    attachment_scores: list[TubeAttachmentScore],
    config: SemanticMaterialInferenceConfig,
) -> tuple[dict[int, int | str], dict[str, Any]]:
    selected_ids = {int(c.candidate_id) for c in selected}
    scores_by_tube: dict[int, list[TubeAttachmentScore]] = {}
    for score in attachment_scores:
        if int(score.object_id) in selected_ids:
            scores_by_tube.setdefault(int(score.tube_id), []).append(score)
    assignments: dict[int, int | str] = {}
    margins: list[float] = []
    for tube_id, scores in scores_by_tube.items():
        ordered = sorted(scores, key=lambda s: (-float(s.score), int(s.object_id)))
        top = ordered[0]
        second = ordered[1].score if len(ordered) > 1 else 0.0
        margin = float(top.score) - float(second)
        margins.append(margin)
        if float(top.score) < float(config.attach_threshold) or margin < float(config.attach_margin):
            assignments[int(tube_id)] = "unknown"
        else:
            assignments[int(tube_id)] = int(top.object_id)
    return assignments, {
        "tube_attachment_margin_mean": float(mean(margins)) if margins else None,
        "tube_attachment_margin_min": float(min(margins)) if margins else None,
        "unknown_tube_ratio": float(sum(1 for v in assignments.values() if v == "unknown") / max(len(assignments), 1)),
    }


def _selected_top_score_quantile(
    selected: list[ObjectFieldCandidate],
    attachment_scores: list[TubeAttachmentScore],
    quantile: float,
) -> float | None:
    selected_ids = {int(c.candidate_id) for c in selected}
    top_by_tube: dict[int, float] = {}
    for score in attachment_scores:
        if int(score.object_id) not in selected_ids:
            continue
        tube_id = int(score.tube_id)
        value = float(score.score)
        top_by_tube[tube_id] = max(value, top_by_tube.get(tube_id, value))
    if not top_by_tube:
        return None
    ordered = sorted(float(v) for v in top_by_tube.values())
    q = min(max(float(quantile), 0.0), 1.0)
    index = int(round(q * (len(ordered) - 1)))
    return float(ordered[index])


def run_semantic_material_inference(
    candidates: list[ObjectFieldCandidate],
    attachment_scores: list[TubeAttachmentScore],
    *,
    config: SemanticMaterialInferenceConfig | None = None,
    diagnostic_metrics: dict[str, float] | None = None,
) -> SemanticMaterialInferenceResult:
    cfg = config or SemanticMaterialInferenceConfig()
    selected, selection_diag = select_compact_object_fields(
        candidates,
        max_fields=int(cfg.max_fields),
        duplicate_support_jaccard=float(cfg.duplicate_support_jaccard),
        duplicate_material_jaccard=float(cfg.duplicate_material_jaccard),
    )
    selected_forbidden_birth_count = int(
        sum(
            1
            for candidate in selected
            if str(candidate.birth_source) != "semantic_masklet" or not tuple(candidate.semantic_masklet_ids)
        )
    )
    adaptive_quantile = _selected_top_score_quantile(
        selected,
        attachment_scores,
        float(cfg.adaptive_attach_score_quantile),
    )
    effective_cfg = cfg
    adaptive_attach_used = False
    if (
        float(cfg.adaptive_attach_threshold) > 0.0
        and adaptive_quantile is not None
        and float(adaptive_quantile) >= float(cfg.adaptive_attach_quantile_min)
    ):
        effective_cfg = replace(cfg, attach_threshold=max(float(cfg.attach_threshold), float(cfg.adaptive_attach_threshold)))
        adaptive_attach_used = True
    assignments, attachment_diag = _attach_tubes(selected, attachment_scores, effective_cfg)
    fields: list[ObjectField] = []
    for index, candidate in enumerate(selected):
        attached = sorted(int(tube_id) for tube_id, obj in assignments.items() if obj == int(candidate.candidate_id))
        fields.append(
            ObjectField(
                object_id=int(candidate.candidate_id),
                primary_field_id=index,
                semantic_masklet_ids=[int(v) for v in candidate.semantic_masklet_ids],
                attached_tube_ids=attached,
                confidence=float(candidate.score),
            )
        )
    metrics = {
        "object_count": int(len(fields)),
        "predictions_per_scene": int(len(fields)),
        "candidate_multiplicity": float(len(candidates) / max(len(fields), 1)),
        "duplicate_rate": selection_diag["duplicate_rate"],
        "conflict_rate": 0.0,
        "birth_from_d4rt_tube_count": selected_forbidden_birth_count,
        "rejected_forbidden_birth_candidate_count": selection_diag["forbidden_birth_count"],
        "effective_attach_threshold": float(effective_cfg.attach_threshold),
        "adaptive_attach_used": bool(adaptive_attach_used),
        "adaptive_attach_score_quantile": adaptive_quantile,
        "unknown_tube_ratio": attachment_diag["unknown_tube_ratio"],
        "tube_attachment_margin_mean": attachment_diag["tube_attachment_margin_mean"],
        "tube_attachment_margin_min": attachment_diag["tube_attachment_margin_min"],
    }
    if diagnostic_metrics:
        metrics.update(diagnostic_metrics)
    constraint_audit = {
        **selection_diag,
        **attachment_diag,
        "effective_attach_threshold": float(effective_cfg.attach_threshold),
        "adaptive_attach_used": bool(adaptive_attach_used),
        "adaptive_attach_score_quantile": adaptive_quantile,
        "birth_from_d4rt_tube_count": selected_forbidden_birth_count,
        "selected_forbidden_birth_count": selected_forbidden_birth_count,
        "rejected_forbidden_birth_candidate_count": selection_diag["forbidden_birth_count"],
        "all_selected_have_semantic_birth": all(c.birth_source == "semantic_masklet" and bool(c.semantic_masklet_ids) for c in selected),
        "ambiguous_tubes_remain_unknown": any(v == "unknown" for v in assignments.values()),
        "one_primary_field_per_object": len({f.primary_field_id for f in fields}) == len(fields),
    }
    return SemanticMaterialInferenceResult(
        object_fields=fields,
        tube_assignments=assignments,
        constraint_audit=constraint_audit,
        metrics=metrics,
    )
