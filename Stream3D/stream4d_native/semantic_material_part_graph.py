from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .measurement_bank import MaskMeasurement
from .semantic_part_graph import PartGraphEdge, auc_from_scores
from .semantic_part_tokens import SemanticPartToken


@dataclass(frozen=True)
class TokenMaterialSupport:
    token_id: int
    frame_id: int
    mask_id: int
    inside_tube_ids: tuple[int, ...]
    boundary_tube_ids: tuple[int, ...]
    outside_visible_tube_ids: tuple[int, ...]


@dataclass(frozen=True)
class MaterialPartGraphEdge:
    token_i: int
    token_j: int
    semantic_object_affinity: float
    material_jaccard: float
    shared_tube_count: int
    material_union_count: int
    visible_outside_conflict_ratio: float
    p3_d4rt_only_affinity: float
    p4_semantic_material_affinity: float
    p5_semantic_material_boundary_affinity: float
    diagnostic_same_gt: bool | None


def build_token_material_support(
    tokens: list[SemanticPartToken],
    measurements: list[MaskMeasurement],
) -> dict[int, TokenMaterialSupport]:
    by_frame_mask = {(int(m.frame_global), int(m.mask_id)): m for m in measurements}
    out: dict[int, TokenMaterialSupport] = {}
    for token in tokens:
        measurement = by_frame_mask.get((int(token.frame_id), int(token.mask_id)))
        inside = tuple(sorted({int(v) for v in measurement.inside_tube_ids})) if measurement else ()
        boundary = tuple(sorted({int(v) for v in measurement.boundary_tube_ids})) if measurement else ()
        outside = tuple(sorted({int(v) for v in measurement.outside_visible_tube_ids})) if measurement else ()
        out[int(token.token_id)] = TokenMaterialSupport(
            token_id=int(token.token_id),
            frame_id=int(token.frame_id),
            mask_id=int(token.mask_id),
            inside_tube_ids=inside,
            boundary_tube_ids=boundary,
            outside_visible_tube_ids=outside,
        )
    return out


def build_material_part_graph_edges(
    semantic_edges: list[PartGraphEdge],
    support_by_token: dict[int, TokenMaterialSupport],
    *,
    material_weight: float = 0.35,
    conflict_weight: float = 0.35,
    min_shared_tube_count: int = 1,
    material_support_shrinkage: float = 0.0,
) -> list[MaterialPartGraphEdge]:
    out: list[MaterialPartGraphEdge] = []
    for edge in semantic_edges:
        left = support_by_token.get(int(edge.token_i))
        right = support_by_token.get(int(edge.token_j))
        left_inside = set(left.inside_tube_ids) if left else set()
        right_inside = set(right.inside_tube_ids) if right else set()
        shared = left_inside & right_inside
        union = left_inside | right_inside
        shared_count = int(len(shared))
        material_jaccard = float(shared_count / max(len(union), 1)) if union else 0.0
        if shared_count < int(min_shared_tube_count):
            material_jaccard = 0.0
        if material_jaccard > 0.0 and float(material_support_shrinkage) > 0.0:
            shrink = float(shared_count) / (float(shared_count) + float(material_support_shrinkage))
            material_jaccard *= shrink
        left_outside = set(left.outside_visible_tube_ids) if left else set()
        right_outside = set(right.outside_visible_tube_ids) if right else set()
        conflict = (left_inside & right_outside) | (right_inside & left_outside)
        conflict_ratio = float(len(conflict) / max(len(union), 1)) if union else 0.0
        d4rt_only = material_jaccard - float(conflict_weight) * conflict_ratio
        sem_mat = float(edge.object_affinity) + float(material_weight) * material_jaccard
        sem_mat_boundary = sem_mat - float(conflict_weight) * conflict_ratio
        out.append(
            MaterialPartGraphEdge(
                token_i=int(edge.token_i),
                token_j=int(edge.token_j),
                semantic_object_affinity=float(edge.object_affinity),
                material_jaccard=float(material_jaccard),
                shared_tube_count=shared_count,
                material_union_count=int(len(union)),
                visible_outside_conflict_ratio=float(conflict_ratio),
                p3_d4rt_only_affinity=float(d4rt_only),
                p4_semantic_material_affinity=float(sem_mat),
                p5_semantic_material_boundary_affinity=float(sem_mat_boundary),
                diagnostic_same_gt=edge.diagnostic_same_gt,
            )
        )
    return out


def summarize_material_part_graph(
    tokens: list[SemanticPartToken],
    material_edges: list[MaterialPartGraphEdge],
    support_by_token: dict[int, TokenMaterialSupport],
    *,
    semantic_false_merge_rate: float | None,
    coverage_at_010: float,
    merge_threshold: float = 0.50,
) -> dict[str, Any]:
    supported = [s for s in support_by_token.values() if len(s.inside_tube_ids) > 0]
    tube_counts = [len(s.inside_tube_ids) for s in support_by_token.values()]
    rows = []
    variants = [
        ("P2_semantic_only", "semantic_object_affinity"),
        ("P3_d4rt_only", "p3_d4rt_only_affinity"),
        ("P4_semantic_material", "p4_semantic_material_affinity"),
        ("P5_semantic_material_boundary", "p5_semantic_material_boundary_affinity"),
    ]
    for variant, field in variants:
        rows.append(
            _variant_summary(
                variant,
                field,
                material_edges,
                coverage_at_010=coverage_at_010,
                merge_threshold=float(merge_threshold),
            )
        )
    semantic_row = next((row for row in rows if row["variant"] == "P2_semantic_only"), None)
    semantic_rate = float(semantic_row["false_merge_rate"]) if semantic_row is not None else None
    for row in rows:
        false_rate = float(row["false_merge_rate"])
        if semantic_rate is None or semantic_rate <= 0.0:
            row["false_merge_reduction_vs_semantic_graph"] = None
        else:
            row["false_merge_reduction_vs_semantic_graph"] = float((semantic_rate - false_rate) / max(semantic_rate, 1e-9))
        row["phase2_gate_pass"] = bool(
            row["false_merge_reduction_vs_semantic_graph"] is not None
            and float(row["false_merge_reduction_vs_semantic_graph"]) >= 0.10
            and float(row["mixed_split_success_improvement_vs_P0"]) >= 0.15
            and float(row["coverage_drop_vs_source"]) <= 0.10
        )
    return {
        "token_count": int(len(tokens)),
        "material_supported_token_count": int(len(supported)),
        "material_supported_token_ratio": float(len(supported) / max(len(tokens), 1)),
        "tube_count_per_token_mean": float(np.mean(tube_counts)) if tube_counts else 0.0,
        "tube_count_per_token_p90": float(np.quantile(tube_counts, 0.90)) if tube_counts else 0.0,
        "material_edge_count": int(len(material_edges)),
        "variant_rows": rows,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "measurement_geometry_fields": ["uv", "visibility", "confidence"],
        "measurement_uses_metric_geometry": False,
    }


def _variant_summary(
    variant: str,
    field: str,
    edges: list[MaterialPartGraphEdge],
    *,
    coverage_at_010: float,
    merge_threshold: float,
) -> dict[str, Any]:
    labeled = [edge for edge in edges if edge.diagnostic_same_gt is not None]
    labels = [bool(edge.diagnostic_same_gt) for edge in labeled]
    scores = [float(getattr(edge, field)) for edge in labeled]
    same = [edge for edge in labeled if bool(edge.diagnostic_same_gt)]
    diff = [edge for edge in labeled if not bool(edge.diagnostic_same_gt)]
    predicted = [edge for edge in labeled if float(getattr(edge, field)) >= float(merge_threshold)]
    false_merge = [edge for edge in predicted if not bool(edge.diagnostic_same_gt)]
    same_positive = [edge for edge in same if float(getattr(edge, field)) >= float(merge_threshold)]
    diff_positive = [edge for edge in diff if float(getattr(edge, field)) >= float(merge_threshold)]
    false_rate = float(len(false_merge) / max(len(predicted), 1))
    mixed_split_improvement = 0.0
    return {
        "variant": variant,
        "score_field": field,
        "edge_count": int(len(edges)),
        "gt_labeled_edge_count": int(len(labeled)),
        "object_part_compatibility_AUC": auc_from_scores(labels, scores) if labeled else None,
        "predicted_merge_count": int(len(predicted)),
        "false_merge_count": int(len(false_merge)),
        "false_merge_rate": false_rate,
        "false_merge_reduction_vs_semantic_graph": None,
        "same_GT_positive_edge_ratio": float(len(same_positive) / max(len(same), 1)),
        "different_GT_positive_edge_ratio": float(len(diff_positive) / max(len(diff), 1)),
        "coverage@0.10": float(coverage_at_010),
        "coverage_drop_vs_source": 0.0,
        "mixed_split_success_improvement_vs_P0": float(mixed_split_improvement),
        "phase2_gate_pass": False,
        "phase2_gate_note": "false because material graph did not implement D4RT-based token splitting",
    }
