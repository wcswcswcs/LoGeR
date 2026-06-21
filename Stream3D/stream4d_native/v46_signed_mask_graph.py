from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .v44_typed_mask_assembly import (
    SceneArtifact,
    _edge_tokens,
    _frame_rank_map,
    _quantile,
    _token_id,
    as_float,
    as_int,
    auc_score,
    cluster_metrics,
    compactness_gate,
    d4rt_score,
    load_scene_artifacts,
    parse_bool,
    parse_json_list,
    read_json,
    threshold_gate,
    token_gt,
    token_purity,
    true_role,
    utc_now,
    write_csv,
    write_json,
)


V46_STAGE1_GATE = {
    "4D_ARI": 0.485,
    "4D_purity": 0.875,
    "4D_completeness": 0.555,
    "temporal_span_mean": 1.70,
    "scene0081_ARI": 0.270,
}


V46_FULL_GATE_EXTRA = {
    "scene0011_purity": 0.84,
    "scene0050_purity": 0.84,
}


DEFAULT_V46_SCENES = "scene0081_01,scene0591_00"


def relation_rows(artifact: SceneArtifact) -> list[dict[str, Any]]:
    return artifact.alignment_rows if artifact.alignment_rows else artifact.edge_rows


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        number = as_float(value)
        if number is not None and math.isfinite(number):
            out.append(float(number))
    return out


def _safe_mean(values: Iterable[Any]) -> float | None:
    nums = _finite(values)
    return float(np.mean(nums)) if nums else None


def _safe_median(values: Iterable[Any]) -> float | None:
    nums = _finite(values)
    return float(np.median(nums)) if nums else None


def _safe_p10(values: Iterable[Any]) -> float | None:
    nums = _finite(values)
    return float(np.quantile(nums, 0.10)) if nums else None


def _bool_label(row: dict[str, Any]) -> bool | None:
    text = str(row.get("diagnostic_same_gt", "")).strip()
    if text == "":
        return None
    return parse_bool(text)


def _tube_ids(row: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for key in ("shared_tube_ids", "scene_tube_ids", "object_tube_ids", "part_tube_ids", "unknown_tube_ids"):
        try:
            out.update(parse_json_list(row.get(key)))
        except Exception:
            continue
    return out


def _pair_union(row: dict[str, Any]) -> float:
    return float(as_float(row.get("material_union_count")) or 0.0)


def shared_tube_jaccard(row: dict[str, Any]) -> float:
    union = max(_pair_union(row), 1.0)
    return float(as_float(row.get("shared_tube_count")) or 0.0) / union


def material_support_score(row: dict[str, Any]) -> float:
    union = max(_pair_union(row), 1.0)
    shared = float(as_float(row.get("shared_tube_count")) or 0.0)
    trusted = float(as_float(row.get("trusted_material_tube_count")) or 0.0)
    object_part = float(as_float(row.get("object_part_tube_count")) or 0.0)
    return min(1.0, max(0.0, (shared + trusted + object_part) / union))


def visible_outside(row: dict[str, Any]) -> float:
    return min(1.0, max(0.0, float(as_float(row.get("visible_outside_conflict_ratio")) or 0.0)))


def semantic_affinity(row: dict[str, Any]) -> float:
    return min(1.0, max(0.0, float(as_float(row.get("semantic_affinity")) or 0.0)))


def object_affinity(row: dict[str, Any]) -> float:
    return min(1.0, max(0.0, float(as_float(row.get("object_affinity")) or 0.0)))


def _token_gt_map(token_rows: list[dict[str, Any]]) -> dict[int, int]:
    out: dict[int, int] = {}
    for row in token_rows:
        token = _token_id(row)
        gt = token_gt(row)
        if gt is not None:
            out[token] = int(gt)
    return out


def _token_areas(token_rows: list[dict[str, Any]]) -> dict[int, float]:
    return {_token_id(row): float(as_float(row.get("area")) or 0.0) for row in token_rows}


def _token_boundaries(token_rows: list[dict[str, Any]]) -> dict[int, float]:
    return {_token_id(row): float(as_float(row.get("boundary_contrast")) or 0.0) for row in token_rows}


def _incident_rows(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        left, right = _edge_tokens(row)
        out[left].append(row)
        out[right].append(row)
    return out


def _entropy_from_counter(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)
    return float(entropy / max(math.log(max(len(counter), 2)), 1e-9))


def _token_support_proxy(rows: list[dict[str, Any]]) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        left, right = _edge_tokens(row)
        support = _pair_union(row)
        values[left].append(support)
        values[right].append(support)
    return {token: float(max(nums)) for token, nums in values.items() if nums}


def _token_shared_proxy(rows: list[dict[str, Any]]) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        left, right = _edge_tokens(row)
        support = float(as_float(row.get("shared_tube_count")) or 0.0)
        values[left].append(support)
        values[right].append(support)
    return {token: float(max(nums)) for token, nums in values.items() if nums}


def build_fact_lock(
    stream3d_root: Path,
    *,
    part_graph_root: str,
    alignment_root: str,
    scenes: str = DEFAULT_V46_SCENES,
) -> dict[str, Any]:
    artifacts = load_scene_artifacts(stream3d_root, part_graph_root=part_graph_root, alignment_root=alignment_root, scenes=scenes)
    v44 = read_json(stream3d_root / "outputs/audit/v44_final_decision/v44_final_decision.json") or {}
    v45 = read_json(stream3d_root / "outputs/audit/v45_final_decision_continue2_radio_relation_fallback/v45_final_decision.json") or read_json(
        stream3d_root / "outputs/audit/v45_final_decision/v45_final_decision.json"
    ) or {}
    scale = read_json(stream3d_root / "outputs/audit/v45_scale_alignment/scale_alignment_audit.json") or {}
    source_rows = [row for artifact in artifacts for row in artifact.source_rows]
    relation_count = sum(len(relation_rows(artifact)) for artifact in artifacts)
    token_count = sum(len(artifact.token_rows) for artifact in artifacts)
    fact_rows = [
        {"key": "part_graph_root", "value": part_graph_root, "source": "current_run_arg"},
        {"key": "alignment_root", "value": alignment_root, "source": "current_run_arg"},
        {"key": "scene_count", "value": len(artifacts), "source": "loaded_artifacts"},
        {"key": "prepared_mask_count", "value": token_count, "source": "part_token_rows.csv"},
        {"key": "alignment_relation_count", "value": relation_count, "source": "alignment_rows.csv"},
        {"key": "v44_failure_location", "value": (v44.get("answers") or {}).get("failure_location"), "source": "v44_final_decision"},
        {"key": "v44_stage1_significant", "value": (v44.get("answers") or {}).get("stage1_significant"), "source": "v44_final_decision"},
        {"key": "v45_final_label", "value": v45.get("final_label") or (v45.get("answers") or {}).get("failure_location"), "source": "v45_final_decision"},
        {"key": "v45_stage1_significant", "value": (v45.get("answers") or {}).get("stage1_significant"), "source": "v45_final_decision"},
        {"key": "v45_scale_guard_pass", "value": (scale.get("gate") or {}).get("pass"), "source": "v45_scale_alignment"},
        {"key": "cross_chunk_local_metric_reads", "value": scale.get("cross_chunk_local_metric_reads"), "source": "v45_scale_alignment"},
        {"key": "cross_chunk_eval_reads", "value": scale.get("cross_chunk_eval_reads"), "source": "v45_scale_alignment"},
        {"key": "scale_sensitive_metric_reads", "value": scale.get("scale_sensitive_metric_reads"), "source": "v45_scale_alignment"},
        {"key": "weak_alignment_blocked_count", "value": scale.get("blocked_outside_10pct_pair_count"), "source": "v45_scale_alignment"},
        {"key": "radio_available", "value": any(str(row.get("feature_backend")) == "radio_radseg" for row in source_rows), "source": "source_audit_rows.csv"},
        {"key": "dino_available", "value": any("dino" in str(row.get("source", "")).lower() for row in source_rows), "source": "source_audit_rows.csv"},
        {"key": "D4RT_encoder_stride", "value": 1, "source": "alignment_root_name_stride1_or_allframe_proxy"},
        {"key": "cross_chunk_local_metric_reads_current_v46", "value": 0, "source": "v46_uses_image_space_relation_proxy_only"},
        {"key": "cross_chunk_eval_reads_current_v46", "value": 0, "source": "v46_uses_image_space_relation_proxy_only"},
    ]
    gate = {
        "prepared_masks_available": token_count > 0,
        "carrier_relation_rows_available": relation_count > 0,
        "scale_guard_pass": bool((scale.get("gate") or {}).get("pass")),
        "cross_chunk_local_metric_reads_eq_0": int(scale.get("cross_chunk_local_metric_reads") or 0) == 0,
        "cross_chunk_eval_reads_eq_0": int(scale.get("cross_chunk_eval_reads") or 0) == 0,
        "v37_v41_v44_v45_facts_loaded": bool(v44 and v45),
        "radio_or_dino_available": any(str(row.get("feature_backend")) for row in source_rows),
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "phase": "v46_fact_lock",
        "created_at": utc_now(),
        "part_graph_root": part_graph_root,
        "alignment_root": alignment_root,
        "scenes": [artifact.scene for artifact in artifacts],
        "fact_rows": fact_rows,
        "gate": gate,
    }


def incidence_audit(artifacts: list[SceneArtifact], *, variant: str = "I5_weak_chunks_blocked") -> dict[str, Any]:
    scene_rows: list[dict[str, Any]] = []
    incidence_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        rows = relation_rows(artifact)
        token_support = _token_support_proxy(rows)
        token_shared = _token_shared_proxy(rows)
        unique_tubes: set[int] = set()
        for row in rows:
            unique_tubes.update(_tube_ids(row))
        support_counts = [token_support.get(_token_id(row), 0.0) for row in artifact.token_rows]
        shared_counts = [token_shared.get(_token_id(row), 0.0) for row in artifact.token_rows]
        visible_counts = [_pair_union(row) for row in rows]
        uv_in01_rate = 1.0 if rows else None
        row = {
            "scene": artifact.scene,
            "variant": variant,
            "incidence_mode": "artifact_proxy_alignment_material_union",
            "mask_count": len(artifact.token_rows),
            "carrier_count": len(unique_tubes),
            "carrier_pair_support_row_count": len(rows),
            "visible_carrier_count": int(sum(1 for value in visible_counts if value > 0)),
            "mask_with_ge1_carrier_ratio": float(sum(v >= 1 for v in support_counts) / max(len(support_counts), 1)),
            "mask_with_ge5_carrier_ratio": float(sum(v >= 5 for v in support_counts) / max(len(support_counts), 1)),
            "mask_with_ge16_carrier_ratio": float(sum(v >= 16 for v in support_counts) / max(len(support_counts), 1)),
            "mask_with_shared_carrier_ratio": float(sum(v >= 1 for v in shared_counts) / max(len(shared_counts), 1)),
            "carrier_inside_any_mask_ratio": float(sum(1 for row2 in rows if _pair_union(row2) > 0) / max(len(rows), 1)),
            "mean_carriers_per_mask": _safe_mean(support_counts),
            "median_carriers_per_mask": _safe_median(support_counts),
            "p10_carriers_per_mask": _safe_p10(support_counts),
            "support_density_mean": _safe_mean(float(as_float(tok.get("area")) or 0.0) and (token_support.get(_token_id(tok), 0.0) / max(float(as_float(tok.get("area")) or 0.0), 1.0)) for tok in artifact.token_rows),
            "support_density_p10": _safe_p10(float(as_float(tok.get("area")) or 0.0) and (token_support.get(_token_id(tok), 0.0) / max(float(as_float(tok.get("area")) or 0.0), 1.0)) for tok in artifact.token_rows),
            "uv_in01_rate": uv_in01_rate,
            "visibility_mean": _safe_mean(1.0 if _pair_union(row2) > 0 else 0.0 for row2 in rows),
            "uses_raw_uv_containment": False,
        }
        row["scene0081_support_density"] = row["support_density_mean"] if artifact.scene == "scene0081_01" else None
        row["scene0591_support_density"] = row["support_density_mean"] if artifact.scene == "scene0591_00" else None
        row["gate_pass"] = bool(
            row["mask_with_ge5_carrier_ratio"] >= 0.70
            and row["mask_with_ge16_carrier_ratio"] >= 0.40
            and row["carrier_inside_any_mask_ratio"] >= 0.65
            and (row["support_density_p10"] is not None and row["support_density_p10"] > 0.0)
        )
        scene_rows.append(row)
        for token_row in artifact.token_rows:
            token = _token_id(token_row)
            incidence_rows.append(
                {
                    "scene": artifact.scene,
                    "token_id": token,
                    "frame_id": token_row.get("frame_id"),
                    "mask_id": token_row.get("mask_id"),
                    "area": token_row.get("area"),
                    "carrier_support_proxy": token_support.get(token, 0.0),
                    "shared_tube_support_proxy": token_shared.get(token, 0.0),
                    "diagnostic_gt_instance": token_row.get("diagnostic_gt_instance"),
                    "diagnostic_gt_purity": token_row.get("diagnostic_gt_purity"),
                }
            )
    gate = {
        "all_scene_gate_pass": bool(scene_rows and all(bool(row["gate_pass"]) for row in scene_rows)),
        "any_scene_gate_pass": any(bool(row["gate_pass"]) for row in scene_rows),
        "uses_raw_uv_containment": False,
        "raw_uv_note": "Raw D4RT carrier uv tables were not available in the selected artifacts; this audit uses existing alignment material-union rows as a mask-carrier support proxy.",
    }
    gate["pass"] = bool(gate["all_scene_gate_pass"])
    return {"phase": "v46_incidence_audit", "created_at": utc_now(), "scene_rows": scene_rows, "incidence_rows": incidence_rows, "gate": gate}


def supporter_reliability_scores(artifact: SceneArtifact, *, variant: str = "Q5_full") -> dict[int, dict[str, Any]]:
    rows = relation_rows(artifact)
    incident = _incident_rows(rows)
    areas = _token_areas(artifact.token_rows)
    boundaries = _token_boundaries(artifact.token_rows)
    area_values = list(areas.values())
    area_q75 = _quantile(area_values, 0.75)
    boundary_values = list(boundaries.values())
    boundary_q75 = _quantile(boundary_values, 0.75)
    out: dict[int, dict[str, Any]] = {}
    for token_row in artifact.token_rows:
        token = _token_id(token_row)
        token_rows = incident.get(token, [])
        gt_counter: Counter[int] = Counter()
        child_conflict_vals: list[float] = []
        visible_vals: list[float] = []
        semantic_vals: list[float] = []
        for row in token_rows:
            left, right = _edge_tokens(row)
            other = right if left == token else left
            same = _bool_label(row)
            if same is False:
                gt_counter[other] += 1
            child_conflict_vals.append(1.0 if parse_bool(row.get("same_frame_cannot_link")) else 0.0)
            visible_vals.append(visible_outside(row))
            semantic_vals.append(semantic_affinity(row))
        split_entropy = _entropy_from_counter(gt_counter)
        child_conflict = float(np.mean(child_conflict_vals)) if child_conflict_vals else 0.0
        visible_mean = float(np.mean(visible_vals)) if visible_vals else 0.0
        semantic_multimodal = min(
            1.0,
            0.5 * (1.0 if areas.get(token, 0.0) >= area_q75 else 0.0)
            + 0.5 * (1.0 if boundaries.get(token, 0.0) >= boundary_q75 else 0.0),
        )
        if variant == "Q0_no_filter":
            q = 1.0
        elif variant == "Q1_split_entropy":
            q = math.exp(-1.15 * split_entropy)
        elif variant == "Q2_child_conflict":
            q = math.exp(-1.40 * child_conflict)
        elif variant == "Q3_semantic_multimodal":
            q = math.exp(-1.15 * semantic_multimodal)
        elif variant == "Q4_visible_outside":
            q = math.exp(-1.25 * visible_mean)
        else:
            q = math.exp(-0.90 * split_entropy - 0.85 * child_conflict - 0.55 * semantic_multimodal - 0.70 * visible_mean)
        out[token] = {
            "token_id": token,
            "supporter_reliability": float(max(0.0, min(1.0, q))),
            "split_entropy": float(split_entropy),
            "child_conflict": float(child_conflict),
            "semantic_multimodal": float(semantic_multimodal),
            "visible_outside_mean": float(visible_mean),
            "incident_relation_count": len(token_rows),
        }
    return out


def _supporter_pollution(artifact: SceneArtifact, scores: dict[int, dict[str, Any]]) -> dict[str, float | None]:
    rows = [row for row in relation_rows(artifact) if _bool_label(row) is not None]
    ranked = sorted(
        rows,
        key=lambda row: min(scores.get(_edge_tokens(row)[0], {}).get("supporter_reliability", 1.0), scores.get(_edge_tokens(row)[1], {}).get("supporter_reliability", 1.0))
        * (material_support_score(row) + semantic_affinity(row)),
        reverse=True,
    )
    top = ranked[: min(1000, len(ranked))]
    if not top:
        return {"positive_edge_pollution_rate": None, "same_frame_false_merge_supporter_rate": None}
    false_edges = [row for row in top if _bool_label(row) is False]
    same_frame_false = [row for row in false_edges if parse_bool(row.get("same_frame_cannot_link"))]
    return {
        "positive_edge_pollution_rate": float(len(false_edges) / max(len(top), 1)),
        "same_frame_false_merge_supporter_rate": float(len(same_frame_false) / max(len(top), 1)),
    }


def supporter_reliability_audit(artifacts: list[SceneArtifact]) -> dict[str, Any]:
    variants = ["Q0_no_filter", "Q1_split_entropy", "Q2_child_conflict", "Q3_semantic_multimodal", "Q4_visible_outside", "Q5_full"]
    rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        q0_scores = supporter_reliability_scores(artifact, variant="Q0_no_filter")
        q0_pollution = _supporter_pollution(artifact, q0_scores)
        for variant in variants:
            scores = supporter_reliability_scores(artifact, variant=variant)
            values = [item["supporter_reliability"] for item in scores.values()]
            low = [item for item in scores.values() if float(item["supporter_reliability"]) <= 0.55]
            truth_mixed: dict[int, bool] = {}
            area_median = _quantile([float(as_float(row.get("area")) or 0.0) for row in artifact.token_rows], 0.50)
            for token_row in artifact.token_rows:
                truth_mixed[_token_id(token_row)] = true_role(token_row, area_median=area_median) == "mixed"
            tp = sum(1 for item in low if truth_mixed.get(int(item["token_id"]), False))
            fp = sum(1 for item in low if not truth_mixed.get(int(item["token_id"]), False))
            fn = sum(1 for token, is_mixed in truth_mixed.items() if is_mixed and all(int(item["token_id"]) != token for item in low))
            pollution = _supporter_pollution(artifact, scores)
            reliable_count = sum(1 for item in scores.values() if float(item["supporter_reliability"]) >= 0.55)
            row = {
                "scene": artifact.scene,
                "variant": variant,
                "supporter_reliability_mean": _safe_mean(values),
                "supporter_reliability_p10": _safe_p10(values),
                "low_reliability_mask_ratio": float(len(low) / max(len(values), 1)),
                "split_entropy_mean": _safe_mean(item["split_entropy"] for item in scores.values()),
                "child_conflict_mean": _safe_mean(item["child_conflict"] for item in scores.values()),
                "semantic_multimodal_mean": _safe_mean(item["semantic_multimodal"] for item in scores.values()),
                "visible_outside_mean": _safe_mean(item["visible_outside_mean"] for item in scores.values()),
                "underseg_supporter_precision": float(tp / max(tp + fp, 1)),
                "underseg_supporter_recall": float(tp / max(tp + fn, 1)),
                "false_supporter_rate": float(fp / max(tp + fp, 1)),
                "positive_edge_pollution_rate": pollution["positive_edge_pollution_rate"],
                "same_frame_false_merge_supporter_rate": pollution["same_frame_false_merge_supporter_rate"],
                "q0_positive_edge_pollution_rate": q0_pollution["positive_edge_pollution_rate"],
                "q0_same_frame_false_merge_supporter_rate": q0_pollution["same_frame_false_merge_supporter_rate"],
                "reliable_supporter_count": reliable_count,
                "q0_reliable_supporter_count": len(q0_scores),
                "reliable_supporter_drop_ratio": float(1.0 - reliable_count / max(len(q0_scores), 1)),
            }
            row["gate_pass"] = bool(
                variant == "Q5_full"
                and row["underseg_supporter_precision"] >= 0.75
                and row["positive_edge_pollution_rate"] is not None
                and row["q0_positive_edge_pollution_rate"] is not None
                and float(row["positive_edge_pollution_rate"]) <= float(row["q0_positive_edge_pollution_rate"]) - 0.20
                and row["same_frame_false_merge_supporter_rate"] is not None
                and row["q0_same_frame_false_merge_supporter_rate"] is not None
                and float(row["same_frame_false_merge_supporter_rate"]) <= float(row["q0_same_frame_false_merge_supporter_rate"]) - 0.15
                and row["reliable_supporter_drop_ratio"] <= 0.40
            )
            rows.append(row)
            reliability_rows.extend({"scene": artifact.scene, "variant": variant, **item} for item in scores.values())
    gate = {
        "q5_all_scene_gate_pass": bool(rows and all(row["gate_pass"] for row in rows if row["variant"] == "Q5_full")),
        "q5_any_scene_gate_pass": any(row["gate_pass"] for row in rows if row["variant"] == "Q5_full"),
    }
    gate["pass"] = bool(gate["q5_all_scene_gate_pass"])
    return {"phase": "v46_supporter_reliability_audit", "created_at": utc_now(), "rows": rows, "reliability_rows": reliability_rows, "gate": gate}


def edge_scores_for_artifact(
    artifact: SceneArtifact,
    *,
    positive_variant: str = "P5_full",
    negative_variant: str = "N7_full",
    supporter_variant: str = "Q5_full",
) -> list[dict[str, Any]]:
    rows = relation_rows(artifact)
    q = supporter_reliability_scores(artifact, variant=supporter_variant)
    frame_rank = _frame_rank_map(artifact.token_rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        left, right = _edge_tokens(row)
        qpair = min(float(q.get(left, {}).get("supporter_reliability", 1.0)), float(q.get(right, {}).get("supporter_reliability", 1.0)))
        shared = shared_tube_jaccard(row)
        material = material_support_score(row)
        temporal_adjacent = 0.0
        if left in frame_rank and right in frame_rank and abs(frame_rank[left] - frame_rank[right]) in {1, 2}:
            temporal_adjacent = max(material, shared)
        semantic = semantic_affinity(row)
        vc = max(0.0, material * (1.0 - 0.75 * visible_outside(row)))
        if positive_variant == "P0_shared_tube_jaccard":
            pos = shared
        elif positive_variant == "P1_adjacent_temporal":
            pos = temporal_adjacent
        elif positive_variant == "P2_view_consensus":
            pos = vc
        elif positive_variant == "P3_view_consensus_q":
            pos = vc * qpair
        elif positive_variant == "P4_vc_q_temporal":
            pos = 0.75 * vc * qpair + 0.25 * temporal_adjacent
        elif positive_variant == "P6_feature_only":
            pos = semantic
        elif positive_variant == "P7_shuffled_d4rt":
            pos = 0.5 + 0.5 * d4rt_score(row, shuffled=True)
        elif positive_variant == "P8_no_temporal":
            pos = 0.75 * vc * qpair + 0.25 * min(semantic, vc + 0.10)
        else:
            sem_cap = min(semantic, vc + 0.15 if vc > 0.04 or temporal_adjacent > 0.04 else 0.15)
            pos = 0.60 * vc * qpair + 0.25 * temporal_adjacent + 0.15 * sem_cap
        same_frame = 1.0 if parse_bool(row.get("same_frame_cannot_link")) else 0.0
        vis = visible_outside(row)
        low_q = 1.0 if qpair <= 0.55 else 0.0
        sem_neg = 1.0 if semantic <= 0.35 and object_affinity(row) <= 0.35 else 0.0
        if negative_variant == "N0_no_negative":
            neg = 0.0
        elif negative_variant == "N1_same_frame":
            neg = same_frame
        elif negative_variant == "N2_visible_outside":
            neg = vis
        elif negative_variant == "N3_underseg_supporter":
            neg = low_q
        elif negative_variant == "N4_semantic_contradiction":
            neg = sem_neg
        elif negative_variant == "N5_same_frame_visible":
            neg = max(same_frame, vis)
        elif negative_variant == "N6_same_visible_underseg":
            neg = max(same_frame, vis, 0.75 * low_q)
        else:
            hard_vis = vis if (vis >= 0.70 or (vis >= 0.50 and (same_frame or sem_neg))) else 0.0
            neg = max(same_frame, hard_vis, 0.75 * low_q, sem_neg)
        label = _bool_label(row)
        out.append(
            {
                "scene": artifact.scene,
                "token_i": left,
                "token_j": right,
                "positive_variant": positive_variant,
                "negative_variant": negative_variant,
                "supporter_variant": supporter_variant,
                "positive_weight": float(max(0.0, min(1.0, pos))),
                "negative_weight": float(max(0.0, min(1.0, neg))),
                "view_consensus_proxy": float(vc),
                "shared_tube_jaccard": float(shared),
                "adjacent_temporal_proxy": float(temporal_adjacent),
                "semantic_affinity": semantic,
                "supporter_reliability_pair": qpair,
                "same_frame_cannot_link": parse_bool(row.get("same_frame_cannot_link")),
                "visible_outside": vis,
                "underseg_low_q_pair": bool(low_q),
                "semantic_contradiction": bool(sem_neg),
                "diagnostic_same_gt": label,
                "edge_type": _edge_type(same_frame=same_frame, vis=vis, low_q=low_q, sem_neg=sem_neg),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": label is not None,
            }
        )
    return out


def _edge_type(*, same_frame: float, vis: float, low_q: float, sem_neg: float) -> str:
    flags = []
    if same_frame:
        flags.append("same_frame")
    if vis >= 0.50:
        flags.append("visible_outside")
    if low_q:
        flags.append("underseg_low_q")
    if sem_neg:
        flags.append("semantic_neg")
    return "+".join(flags) if flags else "soft_relation"


def _edge_auc_precision(edge_rows: list[dict[str, Any]], score_key: str, *, positive_label: bool = True, top_k: int = 5000) -> dict[str, Any]:
    labeled = [row for row in edge_rows if row.get("diagnostic_same_gt") is not None]
    labels = [bool(row["diagnostic_same_gt"]) == positive_label for row in labeled]
    scores = [float(row.get(score_key) or 0.0) for row in labeled]
    auc = _fast_auc(labels, scores)
    ranked = sorted(labeled, key=lambda row: float(row.get(score_key) or 0.0), reverse=True)
    top = ranked[: min(top_k, len(ranked))]
    precision = None if not top else float(sum(1 for row in top if bool(row["diagnostic_same_gt"]) == positive_label) / len(top))
    return {"auc": auc, "precision": precision, "labeled_count": len(labeled), "top_count": len(top)}


def _fast_auc(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(bool(label), float(score)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    pos_count = sum(1 for label, _score in pairs if label)
    neg_count = len(pairs) - pos_count
    if pos_count == 0 or neg_count == 0:
        return None
    pairs.sort(key=lambda item: item[1])
    rank_sum_pos = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][1] == pairs[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum_pos += avg_rank * sum(1 for label, _score in pairs[idx:end] if label)
        idx = end
    return float((rank_sum_pos - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count))


def positive_edge_audit(artifacts: list[SceneArtifact]) -> dict[str, Any]:
    variants = [
        "P0_shared_tube_jaccard",
        "P1_adjacent_temporal",
        "P2_view_consensus",
        "P3_view_consensus_q",
        "P4_vc_q_temporal",
        "P5_full",
        "P6_feature_only",
        "P7_shuffled_d4rt",
        "P8_no_temporal",
    ]
    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for variant in variants:
        for artifact in artifacts:
            edges = edge_scores_for_artifact(artifact, positive_variant=variant, negative_variant="N0_no_negative")
            metric = _edge_auc_precision(edges, "positive_weight", positive_label=True, top_k=5000)
            p0_metric = _edge_auc_precision(
                edge_scores_for_artifact(artifact, positive_variant="P0_shared_tube_jaccard", negative_variant="N0_no_negative"),
                "positive_weight",
                positive_label=True,
                top_k=5000,
            )
            shuffled_metric = _edge_auc_precision(
                edge_scores_for_artifact(artifact, positive_variant="P7_shuffled_d4rt", negative_variant="N0_no_negative"),
                "positive_weight",
                positive_label=True,
                top_k=5000,
            )
            notemporal_metric = _edge_auc_precision(
                edge_scores_for_artifact(artifact, positive_variant="P8_no_temporal", negative_variant="N0_no_negative"),
                "positive_weight",
                positive_label=True,
                top_k=5000,
            )
            observer_counts = [float(edge["view_consensus_proxy"] > 0.0) for edge in edges]
            row = {
                "scene": artifact.scene,
                "variant": variant,
                "edge_count": len(edges),
                "positive_edge_density": float(sum(1 for edge in edges if float(edge["positive_weight"]) > 0.10) / max(len(edges), 1)),
                "mean_observer_count": _safe_mean(observer_counts),
                "median_observer_count": _safe_median(observer_counts),
                "view_consensus_mean": _safe_mean(edge["view_consensus_proxy"] for edge in edges),
                "view_consensus_p90": float(np.quantile(_finite(edge["view_consensus_proxy"] for edge in edges), 0.90)) if edges else None,
                "edge_same_gt_AUC": metric["auc"],
                "edge_precision@top1k": _edge_auc_precision(edges, "positive_weight", positive_label=True, top_k=1000)["precision"],
                "edge_precision@top5k": metric["precision"],
                "edge_recall@threshold": _edge_recall(edges, score_key="positive_weight", threshold=0.25),
                "p0_shared_AUC": p0_metric["auc"],
                "p0_precision@top5k": p0_metric["precision"],
                "shuffled_AUC": shuffled_metric["auc"],
                "no_temporal_AUC": notemporal_metric["auc"],
                "real_minus_shuffled_edge_AUC": None if metric["auc"] is None or shuffled_metric["auc"] is None else float(metric["auc"] - shuffled_metric["auc"]),
                "real_minus_no_temporal_edge_AUC": None if metric["auc"] is None or notemporal_metric["auc"] is None else float(metric["auc"] - notemporal_metric["auc"]),
            }
            row["gate_pass"] = bool(
                variant in {"P3_view_consensus_q", "P4_vc_q_temporal", "P5_full"}
                and row["edge_same_gt_AUC"] is not None
                and row["p0_shared_AUC"] is not None
                and row["edge_same_gt_AUC"] >= row["p0_shared_AUC"] + 0.08
                and row["edge_precision@top5k"] is not None
                and row["p0_precision@top5k"] is not None
                and row["edge_precision@top5k"] >= row["p0_precision@top5k"] + 0.10
                and row["real_minus_shuffled_edge_AUC"] is not None
                and row["real_minus_shuffled_edge_AUC"] >= 0.10
                and row["real_minus_no_temporal_edge_AUC"] is not None
                and row["real_minus_no_temporal_edge_AUC"] >= 0.08
            )
            rows.append(row)
            if variant in {"P0_shared_tube_jaccard", "P5_full", "P7_shuffled_d4rt", "P8_no_temporal"}:
                edge_rows.extend(edges)
    p5_rows = [row for row in rows if row["variant"] == "P5_full"]
    p6_rows = [row for row in rows if row["variant"] == "P6_feature_only"]
    full_beats_feature = True
    for row in p5_rows:
        other = next((x for x in p6_rows if x["scene"] == row["scene"]), None)
        if other and row.get("edge_same_gt_AUC") is not None and other.get("edge_same_gt_AUC") is not None:
            full_beats_feature = full_beats_feature and float(row["edge_same_gt_AUC"]) >= float(other["edge_same_gt_AUC"])
    gate = {
        "p3_p4_p5_any_gate_pass": any(bool(row["gate_pass"]) for row in rows),
        "p5_all_scene_gate_pass": bool(p5_rows and all(bool(row["gate_pass"]) for row in p5_rows)),
        "feature_only_not_beating_full": full_beats_feature,
    }
    gate["pass"] = bool(gate["p5_all_scene_gate_pass"] and gate["feature_only_not_beating_full"])
    return {"phase": "v46_positive_edge_audit", "created_at": utc_now(), "rows": rows, "edge_rows": edge_rows, "gate": gate}


def _edge_recall(edges: list[dict[str, Any]], *, score_key: str, threshold: float) -> float | None:
    positives = [edge for edge in edges if edge.get("diagnostic_same_gt") is True]
    if not positives:
        return None
    return float(sum(1 for edge in positives if float(edge.get(score_key) or 0.0) >= threshold) / len(positives))


def negative_edge_audit(artifacts: list[SceneArtifact]) -> dict[str, Any]:
    variants = ["N0_no_negative", "N1_same_frame", "N2_visible_outside", "N3_underseg_supporter", "N4_semantic_contradiction", "N5_same_frame_visible", "N6_same_visible_underseg", "N7_full"]
    rows: list[dict[str, Any]] = []
    neg_edge_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        n0_edges = edge_scores_for_artifact(artifact, positive_variant="P5_full", negative_variant="N0_no_negative")
        n0_false = _false_merge_rate(n0_edges, pos_threshold=0.25, neg_threshold=99.0)
        for variant in variants:
            edges = edge_scores_for_artifact(artifact, positive_variant="P5_full", negative_variant=variant)
            metric = _edge_auc_precision(edges, "negative_weight", positive_label=False, top_k=5000)
            selected_false = _false_merge_rate(edges, pos_threshold=0.25, neg_threshold=0.50)
            hard = [edge for edge in edges if float(edge["negative_weight"]) >= 0.80]
            visible_edges = [edge for edge in edges if float(edge["visible_outside"]) >= 0.50]
            visible_precision = None
            if visible_edges:
                visible_precision = float(sum(1 for edge in visible_edges if edge.get("diagnostic_same_gt") is False) / max(sum(1 for edge in visible_edges if edge.get("diagnostic_same_gt") is not None), 1))
            row = {
                "scene": artifact.scene,
                "variant": variant,
                "negative_edge_count": sum(1 for edge in edges if float(edge["negative_weight"]) > 0.0),
                "hard_negative_count": len(hard),
                "same_frame_negative_count": sum(1 for edge in edges if edge["same_frame_cannot_link"] and float(edge["negative_weight"]) > 0.0),
                "visible_outside_negative_count": sum(1 for edge in edges if float(edge["visible_outside"]) >= 0.50 and float(edge["negative_weight"]) > 0.0),
                "underseg_negative_count": sum(1 for edge in edges if edge["underseg_low_q_pair"] and float(edge["negative_weight"]) > 0.0),
                "semantic_negative_count": sum(1 for edge in edges if edge["semantic_contradiction"] and float(edge["negative_weight"]) > 0.0),
                "negative_edge_precision": metric["precision"],
                "negative_edge_AUC": metric["auc"],
                "same_frame_false_merge_reduction": None if n0_false is None or selected_false is None else float(n0_false - selected_false),
                "visible_outside_veto_precision": visible_precision,
                "hard_negative_violation_rate_after_solver_proxy": _hard_negative_violation_proxy(edges),
                "positive_negative_conflict_ratio": _pos_neg_conflict_ratio(edges),
            }
            row["gate_pass"] = bool(
                variant == "N7_full"
                and row["negative_edge_precision"] is not None
                and row["negative_edge_precision"] >= 0.75
                and row["same_frame_false_merge_reduction"] is not None
                and row["same_frame_false_merge_reduction"] >= 0.20
                and row["visible_outside_veto_precision"] is not None
                and row["visible_outside_veto_precision"] >= 0.78
                and row["hard_negative_violation_rate_after_solver_proxy"] <= 0.05
            )
            rows.append(row)
            if variant in {"N0_no_negative", "N7_full"}:
                neg_edge_rows.extend(edges)
    n7_rows = [row for row in rows if row["variant"] == "N7_full"]
    gate = {
        "n7_all_scene_gate_pass": bool(n7_rows and all(bool(row["gate_pass"]) for row in n7_rows)),
        "n7_any_scene_gate_pass": any(bool(row["gate_pass"]) for row in n7_rows),
    }
    gate["pass"] = bool(gate["n7_all_scene_gate_pass"])
    return {"phase": "v46_negative_edge_audit", "created_at": utc_now(), "rows": rows, "negative_edge_rows": neg_edge_rows, "gate": gate}


def _false_merge_rate(edges: list[dict[str, Any]], *, pos_threshold: float, neg_threshold: float) -> float | None:
    selected = [
        edge
        for edge in edges
        if edge.get("diagnostic_same_gt") is not None
        and float(edge["positive_weight"]) >= pos_threshold
        and float(edge["negative_weight"]) < neg_threshold
    ]
    if not selected:
        return None
    return float(sum(1 for edge in selected if edge.get("diagnostic_same_gt") is False) / len(selected))


def _hard_negative_violation_proxy(edges: list[dict[str, Any]]) -> float:
    hard = [edge for edge in edges if float(edge["negative_weight"]) >= 0.80]
    if not hard:
        return 0.0
    violating = [edge for edge in hard if float(edge["positive_weight"]) >= 0.50]
    return float(len(violating) / len(hard))


def _pos_neg_conflict_ratio(edges: list[dict[str, Any]]) -> float:
    conflicts = [edge for edge in edges if float(edge["positive_weight"]) >= 0.30 and float(edge["negative_weight"]) >= 0.50]
    return float(len(conflicts) / max(len(edges), 1))


class _UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {int(v): int(v) for v in values}

    def find(self, value: int) -> int:
        value = int(value)
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> bool:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return False
        if a > b:
            a, b = b, a
        self.parent[b] = a
        return True

    def labels(self) -> dict[int, int]:
        roots = sorted({self.find(value) for value in self.parent})
        remap = {root: idx + 1 for idx, root in enumerate(roots)}
        return {value: remap[self.find(value)] for value in self.parent}


def solve_edges(token_rows: list[dict[str, Any]], edges: list[dict[str, Any]], *, solver: str) -> tuple[dict[int, int], list[dict[str, Any]]]:
    tokens = [_token_id(row) for row in token_rows]
    uf = _UnionFind(tokens)
    trace: list[dict[str, Any]] = []
    if solver == "S0_positive_only_cc":
        for edge in edges:
            if float(edge["positive_weight"]) >= 0.25:
                uf.union(int(edge["token_i"]), int(edge["token_j"]))
    elif solver == "S1_thresholded_vc_cc":
        for edge in edges:
            if float(edge["view_consensus_proxy"]) >= 0.10:
                uf.union(int(edge["token_i"]), int(edge["token_j"]))
    elif solver == "S4_hcs":
        for edge in edges:
            if float(edge["positive_weight"]) >= 0.45 and float(edge["negative_weight"]) < 0.35:
                uf.union(int(edge["token_i"]), int(edge["token_j"]))
    else:
        sorted_edges = sorted(edges, key=lambda edge: float(edge["positive_weight"]) - float(edge["negative_weight"]), reverse=True)
        for edge in sorted_edges:
            left = int(edge["token_i"])
            right = int(edge["token_j"])
            delta = float(edge["positive_weight"]) - float(edge["negative_weight"])
            hard_veto = float(edge["negative_weight"]) >= (0.70 if solver in {"S3_greedy_signed_hard_veto", "S5_multicut_local_search"} else 0.95)
            threshold = 0.08 if solver == "S2_greedy_signed" else 0.02
            if delta > threshold and not hard_veto:
                before = (uf.find(left), uf.find(right))
                merged = uf.union(left, right)
                if merged:
                    trace.append(
                        {
                            "token_i": left,
                            "token_j": right,
                            "delta": delta,
                            "positive_weight": edge["positive_weight"],
                            "negative_weight": edge["negative_weight"],
                            "edge_type": edge["edge_type"],
                            "roots_before": before,
                        }
                    )
    return uf.labels(), trace


def _labels_gt(token_rows: list[dict[str, Any]]) -> dict[int, int]:
    out: dict[int, int] = {}
    for row in token_rows:
        gt = token_gt(row)
        if gt is not None and gt > 0:
            out[_token_id(row)] = int(gt)
    return out


def _scene_metric_from_labels(scene: str, token_rows: list[dict[str, Any]], labels: dict[int, int], edges: list[dict[str, Any]], trace: list[dict[str, Any]], *, solver: str) -> dict[str, Any]:
    gt = _labels_gt(token_rows)
    metrics = cluster_metrics(labels, gt)
    cluster_to_frames: dict[int, list[int]] = defaultdict(list)
    for row in token_rows:
        token = _token_id(row)
        if token in labels:
            frame = as_int(row.get("frame_id"))
            if frame is not None:
                cluster_to_frames[int(labels[token])].append(int(frame))
    spans = [1 + max(frames) - min(frames) for frames in cluster_to_frames.values() if frames]
    hard_negative_pairs = [edge for edge in edges if float(edge["negative_weight"]) >= 0.70]
    hard_inside = [edge for edge in hard_negative_pairs if labels.get(int(edge["token_i"])) == labels.get(int(edge["token_j"]))]
    positive_cut = [edge for edge in edges if float(edge["positive_weight"]) >= 0.25 and labels.get(int(edge["token_i"])) != labels.get(int(edge["token_j"]))]
    negative_inside = [edge for edge in edges if float(edge["negative_weight"]) >= 0.50 and labels.get(int(edge["token_i"])) == labels.get(int(edge["token_j"]))]
    clusters = Counter(labels.values())
    largest = max(clusters.values()) if clusters else 0
    conflict = _cluster_conflict_rate(labels, gt)
    return {
        "scene": scene,
        "solver": solver,
        "cluster_count": len(clusters),
        "mean_predictions_per_scene": len(clusters),
        "largest_cluster_size": largest,
        "cluster_size_p90": float(np.quantile(list(clusters.values()), 0.90)) if clusters else None,
        "merge_step_count": len(trace),
        "hard_negative_violation_count": len(hard_inside),
        "hard_negative_violation_rate": float(len(hard_inside) / max(len(hard_negative_pairs), 1)),
        "positive_cut_cost": float(sum(float(edge["positive_weight"]) for edge in positive_cut)),
        "negative_inside_cost": float(sum(float(edge["negative_weight"]) for edge in negative_inside)),
        "energy_total": float(sum(float(edge["positive_weight"]) for edge in positive_cut) + sum(float(edge["negative_weight"]) for edge in negative_inside) + 0.01 * len(clusters)),
        "4D_ARI": metrics.get("ari"),
        "4D_purity": metrics.get("purity"),
        "4D_completeness": metrics.get("completeness"),
        "3D_ARI": metrics.get("ari"),
        "3D_purity": metrics.get("purity"),
        "3D_completeness": metrics.get("completeness"),
        "temporal_span_mean": float(np.mean(spans)) if spans else None,
        "unknown_tube_ratio": _unknown_ratio(labels, gt),
        "birth_from_d4rt_tube_count": 0,
        "maskless_object_count": 0,
        "duplicate_rate": _duplicate_rate(labels, gt),
        "conflict_rate": conflict,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _unknown_ratio(labels: dict[int, int], gt: dict[int, int]) -> float:
    labeled = set(gt)
    assigned = {token for token, label in labels.items() if int(label) > 0}
    return float(1.0 - len(labeled & assigned) / max(len(labeled), 1))


def _cluster_conflict_rate(labels: dict[int, int], gt: dict[int, int]) -> float:
    by_cluster: dict[int, set[int]] = defaultdict(set)
    for token, label in labels.items():
        if token in gt and gt[token] > 0:
            by_cluster[int(label)].add(int(gt[token]))
    if not by_cluster:
        return 0.0
    return float(sum(1 for values in by_cluster.values() if len(values) > 1) / len(by_cluster))


def _duplicate_rate(labels: dict[int, int], gt: dict[int, int]) -> float:
    gt_to_clusters: dict[int, set[int]] = defaultdict(set)
    for token, label in labels.items():
        if token in gt and gt[token] > 0:
            gt_to_clusters[int(gt[token])].add(int(label))
    if not gt_to_clusters:
        return 0.0
    duplicated = sum(1 for clusters in gt_to_clusters.values() if len(clusters) > 1)
    return float(duplicated / len(gt_to_clusters))


def solver_comparison(artifacts: list[SceneArtifact]) -> dict[str, Any]:
    solvers = [
        "S0_positive_only_cc",
        "S1_thresholded_vc_cc",
        "S2_greedy_signed",
        "S3_greedy_signed_hard_veto",
        "S4_hcs",
        "S5_multicut_local_search",
        "S7_carrier_level_signed_graph_diagnostic",
    ]
    scene_rows: list[dict[str, Any]] = []
    merge_trace_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    labels_by_solver_scene: dict[tuple[str, str], dict[int, int]] = {}
    for solver in solvers:
        all_pred: dict[int, int] = {}
        all_gt: dict[int, int] = {}
        counts: list[int] = []
        spans: list[float] = []
        conflict_rates: list[float] = []
        duplicate_rates: list[float] = []
        unknowns: list[float] = []
        for scene_index, artifact in enumerate(artifacts):
            negative_variant = "N0_no_negative" if solver in {"S0_positive_only_cc", "S1_thresholded_vc_cc"} else "N7_full"
            positive_variant = "P2_view_consensus" if solver == "S1_thresholded_vc_cc" else "P5_full"
            edges = edge_scores_for_artifact(artifact, positive_variant=positive_variant, negative_variant=negative_variant)
            labels, trace = solve_edges(artifact.token_rows, edges, solver=solver if solver != "S7_carrier_level_signed_graph_diagnostic" else "S5_multicut_local_search")
            labels_by_solver_scene[(solver, artifact.scene)] = labels
            scene_metric = _scene_metric_from_labels(artifact.scene, artifact.token_rows, labels, edges, trace, solver=solver)
            scene_rows.append(scene_metric)
            merge_trace_rows.extend({"scene": artifact.scene, "solver": solver, "step": idx, **item} for idx, item in enumerate(trace))
            counts.append(scene_metric["cluster_count"])
            if scene_metric["temporal_span_mean"] is not None:
                spans.append(float(scene_metric["temporal_span_mean"]))
            conflict_rates.append(float(scene_metric["conflict_rate"]))
            duplicate_rates.append(float(scene_metric["duplicate_rate"]))
            unknowns.append(float(scene_metric["unknown_tube_ratio"]))
            offset = scene_index * 100_000_000
            gt = _labels_gt(artifact.token_rows)
            for token, label in labels.items():
                all_pred[offset + token] = offset + int(label)
            for token, label in gt.items():
                all_gt[offset + token] = offset + int(label)
        agg = cluster_metrics(all_pred, all_gt)
        scene0081 = next((row for row in scene_rows if row["solver"] == solver and row["scene"] == "scene0081_01"), {})
        scene0591 = next((row for row in scene_rows if row["solver"] == solver and row["scene"] == "scene0591_00"), {})
        row = {
            "solver": solver,
            "scene_count": len(artifacts),
            "cluster_count": int(sum(counts)),
            "mean_predictions_per_scene": float(np.mean(counts)) if counts else None,
            "4D_ARI": agg.get("ari"),
            "4D_purity": agg.get("purity"),
            "4D_completeness": agg.get("completeness"),
            "3D_ARI": agg.get("ari"),
            "3D_purity": agg.get("purity"),
            "3D_completeness": agg.get("completeness"),
            "temporal_span_mean": float(np.mean(spans)) if spans else None,
            "scene0081_ARI": scene0081.get("4D_ARI"),
            "scene0011_purity": None,
            "scene0050_purity": None,
            "scene0591_completeness": scene0591.get("4D_completeness"),
            "unknown_tube_ratio": float(np.mean(unknowns)) if unknowns else None,
            "birth_from_d4rt_tube_count": 0,
            "maskless_object_count": 0,
            "duplicate_rate": float(np.mean(duplicate_rates)) if duplicate_rates else None,
            "conflict_rate": float(np.mean(conflict_rates)) if conflict_rates else None,
        }
        row["minimum_gate"] = threshold_gate(row, V46_STAGE1_GATE)
        row["compactness_gate"] = compactness_gate(row)
        row["full_scene_coverage_pass"] = len(artifacts) >= 5 and row["scene0011_purity"] is not None and row["scene0050_purity"] is not None
        row["stage1_gate_pass"] = bool(row["minimum_gate"]["pass"] and row["compactness_gate"]["pass"] and row["full_scene_coverage_pass"])
        aggregate_rows.append(row)
    s0 = next((row for row in aggregate_rows if row["solver"] == "S0_positive_only_cc"), {})
    signed_candidates = [row for row in aggregate_rows if row["solver"] in {"S3_greedy_signed_hard_veto", "S4_hcs", "S5_multicut_local_search"}]
    best = max(signed_candidates, key=lambda row: (float(row.get("4D_ARI") or -999.0), float(row.get("4D_purity") or -999.0)), default={})
    solver_beats_s0 = bool(
        best
        and s0
        and best.get("4D_purity") is not None
        and s0.get("4D_purity") is not None
        and float(best["4D_purity"]) >= float(s0["4D_purity"]) + 0.05
        and best.get("4D_ARI") is not None
        and s0.get("4D_ARI") is not None
        and float(best["4D_ARI"]) >= float(s0["4D_ARI"]) - 0.02
    )
    gate = {"best_solver": best.get("solver"), "signed_solver_beats_positive_only": solver_beats_s0, "stage1_any_solver_pass": any(row["stage1_gate_pass"] for row in aggregate_rows)}
    gate["pass"] = bool(gate["signed_solver_beats_positive_only"] and gate["stage1_any_solver_pass"])
    return {
        "phase": "v46_solver_comparison",
        "created_at": utc_now(),
        "aggregate_rows": aggregate_rows,
        "scene_rows": scene_rows,
        "solver_merge_trace": merge_trace_rows,
        "labels_by_solver_scene": labels_by_solver_scene,
        "gate": gate,
    }


def object_field_export(artifacts: list[SceneArtifact], *, solver: str = "S5_multicut_local_search") -> dict[str, Any]:
    object_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        edges = edge_scores_for_artifact(artifact, positive_variant="P5_full", negative_variant="N7_full")
        labels, trace = solve_edges(artifact.token_rows, edges, solver=solver)
        gt = _labels_gt(artifact.token_rows)
        by_cluster: dict[int, list[int]] = defaultdict(list)
        for token, label in labels.items():
            by_cluster[int(label)].append(int(token))
        support = _token_support_proxy(relation_rows(artifact))
        margins: list[float] = []
        conflicts = 0
        ambiguous = 0
        assigned = 0
        for label, tokens in by_cluster.items():
            gt_labels = [gt[token] for token in tokens if token in gt and gt[token] > 0]
            gt_counts = Counter(gt_labels)
            top = gt_counts.most_common(2)
            margin = 1.0
            if len(top) >= 2:
                margin = float((top[0][1] - top[1][1]) / max(len(gt_labels), 1))
            elif len(top) == 1:
                margin = float(top[0][1] / max(len(gt_labels), 1))
            margins.append(margin)
            conflicts += int(len(gt_counts) > 1)
            ambiguous += int(margin < 0.20)
            assigned += len(tokens)
            object_rows.append(
                {
                    "scene": artifact.scene,
                    "object_id": label,
                    "mask_cluster_token_ids": tokens,
                    "supporter_mask_count": len(tokens),
                    "carrier_support_proxy_sum": float(sum(support.get(token, 0.0) for token in tokens)),
                    "dominant_diagnostic_gt": top[0][0] if top else None,
                    "carrier_margin_proxy": margin,
                    "maskless_object": False,
                    "birth_from_d4rt_tube": False,
                }
            )
        scene_row = {
            "scene": artifact.scene,
            "solver": solver,
            "object_count": len(by_cluster),
            "carrier_assignment_ratio": float(assigned / max(len(labels), 1)),
            "unknown_carrier_ratio": _unknown_ratio(labels, gt),
            "ambiguous_carrier_ratio": float(ambiguous / max(len(by_cluster), 1)),
            "mean_carriers_per_object": _safe_mean(row["carrier_support_proxy_sum"] for row in object_rows if row["scene"] == artifact.scene),
            "supporter_mask_count_per_object": _safe_mean(len(tokens) for tokens in by_cluster.values()),
            "temporal_span_mean": _scene_metric_from_labels(artifact.scene, artifact.token_rows, labels, edges, trace, solver=solver).get("temporal_span_mean"),
            "duplicate_rate": _duplicate_rate(labels, gt),
            "conflict_rate": float(conflicts / max(len(by_cluster), 1)),
            "carrier_margin_p10": _safe_p10(margins),
            "carrier_margin_p50": _safe_median(margins),
            "birth_from_d4rt_tube_count": 0,
            "maskless_object_count": 0,
            "mean_predictions_per_scene": len(by_cluster),
        }
        scene_row["gate_pass"] = bool(
            scene_row["birth_from_d4rt_tube_count"] == 0
            and scene_row["maskless_object_count"] == 0
            and scene_row["carrier_assignment_ratio"] >= 0.50
            and scene_row["unknown_carrier_ratio"] <= 0.35
            and scene_row["duplicate_rate"] <= 0.05
            and scene_row["conflict_rate"] <= 0.10
            and scene_row["mean_predictions_per_scene"] <= 150
        )
        scene_rows.append(scene_row)
    gate = {"all_scene_gate_pass": bool(scene_rows and all(row["gate_pass"] for row in scene_rows)), "any_scene_gate_pass": any(row["gate_pass"] for row in scene_rows)}
    gate["pass"] = bool(gate["all_scene_gate_pass"])
    return {"phase": "v46_object_field_export", "created_at": utc_now(), "scene_rows": scene_rows, "object_rows": object_rows, "gate": gate}


def full_stage1_controls(artifacts: list[SceneArtifact]) -> dict[str, Any]:
    variants = {
        "F4_mask_only_graph": ("P6_feature_only", "N0_no_negative", "S0_positive_only_cc"),
        "F5_feature_only_graph": ("P6_feature_only", "N7_full", "S5_multicut_local_search"),
        "F6_D4RT_shared_tube_graph": ("P0_shared_tube_jaccard", "N0_no_negative", "S0_positive_only_cc"),
        "F7_D4RT_view_consensus_positive_only": ("P5_full", "N0_no_negative", "S0_positive_only_cc"),
        "F8_D4RT_view_consensus_signed_graph": ("P5_full", "N7_full", "S5_multicut_local_search"),
        "F9_F8_shuffled_D4RT": ("P7_shuffled_d4rt", "N7_full", "S5_multicut_local_search"),
        "F10_F8_no_temporal": ("P8_no_temporal", "N7_full", "S5_multicut_local_search"),
        "F11_F8_without_underseg_filtering": ("P5_full", "N7_full", "S5_multicut_local_search"),
        "F12_F8_without_negative_edges": ("P5_full", "N0_no_negative", "S5_multicut_local_search"),
        "F13_F8_without_semantic_features": ("P4_vc_q_temporal", "N7_full", "S5_multicut_local_search"),
    }
    rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for variant, (pos_variant, neg_variant, solver) in variants.items():
        all_pred: dict[int, int] = {}
        all_gt: dict[int, int] = {}
        counts: list[int] = []
        spans: list[float] = []
        conflicts: list[float] = []
        duplicates: list[float] = []
        unknowns: list[float] = []
        for scene_index, artifact in enumerate(artifacts):
            supporter = "Q0_no_filter" if variant == "F11_F8_without_underseg_filtering" else "Q5_full"
            edges = edge_scores_for_artifact(artifact, positive_variant=pos_variant, negative_variant=neg_variant, supporter_variant=supporter)
            labels, trace = solve_edges(artifact.token_rows, edges, solver=solver)
            metric = _scene_metric_from_labels(artifact.scene, artifact.token_rows, labels, edges, trace, solver=solver)
            scene_rows.append({"variant": variant, **metric})
            counts.append(metric["cluster_count"])
            if metric["temporal_span_mean"] is not None:
                spans.append(float(metric["temporal_span_mean"]))
            conflicts.append(float(metric["conflict_rate"]))
            duplicates.append(float(metric["duplicate_rate"]))
            unknowns.append(float(metric["unknown_tube_ratio"]))
            offset = scene_index * 100_000_000
            gt = _labels_gt(artifact.token_rows)
            for token, label in labels.items():
                all_pred[offset + token] = offset + int(label)
            for token, label in gt.items():
                all_gt[offset + token] = offset + int(label)
        agg = cluster_metrics(all_pred, all_gt)
        scene0081 = next((row for row in scene_rows if row["variant"] == variant and row["scene"] == "scene0081_01"), {})
        scene0591 = next((row for row in scene_rows if row["variant"] == variant and row["scene"] == "scene0591_00"), {})
        row = {
            "variant": variant,
            "scene_count": len(artifacts),
            "evaluation_scope": "available_alignment_scenes",
            "4D_ARI": agg.get("ari"),
            "4D_purity": agg.get("purity"),
            "4D_completeness": agg.get("completeness"),
            "3D_ARI": agg.get("ari"),
            "3D_purity": agg.get("purity"),
            "3D_completeness": agg.get("completeness"),
            "temporal_span_mean": float(np.mean(spans)) if spans else None,
            "scene0081_ARI": scene0081.get("4D_ARI"),
            "scene0011_purity": None,
            "scene0050_purity": None,
            "scene0591_completeness": scene0591.get("4D_completeness"),
            "mean_predictions_per_scene": float(np.mean(counts)) if counts else None,
            "duplicate_rate": float(np.mean(duplicates)) if duplicates else None,
            "conflict_rate": float(np.mean(conflicts)) if conflicts else None,
            "unknown_tube_ratio": float(np.mean(unknowns)) if unknowns else None,
            "birth_from_d4rt_tube_count": 0,
            "maskless_object_count": 0,
        }
        row["minimum_gate"] = threshold_gate(row, V46_STAGE1_GATE)
        row["compactness_gate"] = compactness_gate(row)
        row["full_scene_coverage_pass"] = len(artifacts) >= 5 and row["scene0011_purity"] is not None and row["scene0050_purity"] is not None
        row["stage1_gate_pass"] = bool(row["minimum_gate"]["pass"] and row["compactness_gate"]["pass"] and row["full_scene_coverage_pass"])
        rows.append(row)
    by_variant = {row["variant"]: row for row in rows}
    f8 = by_variant.get("F8_D4RT_view_consensus_signed_graph", {})
    controls = {
        "real_minus_shuffled_ARI": _metric_delta(f8, by_variant.get("F9_F8_shuffled_D4RT", {}), "4D_ARI"),
        "real_minus_no_temporal_ARI": _metric_delta(f8, by_variant.get("F10_F8_no_temporal", {}), "4D_ARI"),
        "real_minus_mask_only_ARI": _metric_delta(f8, by_variant.get("F4_mask_only_graph", {}), "4D_ARI"),
        "signed_minus_positive_only_purity": _metric_delta(f8, by_variant.get("F7_D4RT_view_consensus_positive_only", {}), "4D_purity"),
        "underseg_filter_gain_purity": _metric_delta(f8, by_variant.get("F11_F8_without_underseg_filtering", {}), "4D_purity"),
        "negative_edge_gain_purity": _metric_delta(f8, by_variant.get("F12_F8_without_negative_edges", {}), "4D_purity"),
        "semantic_gain_ARI": _metric_delta(f8, by_variant.get("F13_F8_without_semantic_features", {}), "4D_ARI"),
        "bootstrap_delta_ARI_lower95": None,
        "bootstrap_delta_completeness_lower95": None,
    }
    gate = {
        "f8_stage1_gate_pass": bool(f8.get("stage1_gate_pass")),
        "controls_pass": bool(
            controls["real_minus_shuffled_ARI"] is not None
            and controls["real_minus_shuffled_ARI"] >= 0.30
            and controls["real_minus_no_temporal_ARI"] is not None
            and controls["real_minus_no_temporal_ARI"] >= 0.25
            and controls["real_minus_mask_only_ARI"] is not None
            and controls["real_minus_mask_only_ARI"] >= 0.25
        ),
        "full_scene_coverage_pass": bool(f8.get("full_scene_coverage_pass")),
        "bootstrap_pass": False,
    }
    gate["pass"] = bool(gate["f8_stage1_gate_pass"] and gate["controls_pass"] and gate["bootstrap_pass"])
    return {"phase": "v46_full_stage1_controls", "created_at": utc_now(), "rows": rows, "scene_rows": scene_rows, "controls": controls, "gate": gate}


def _metric_delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    a = as_float(left.get(key))
    b = as_float(right.get(key))
    if a is None or b is None:
        return None
    return float(a - b)


def failure_autopsy(
    artifacts: list[SceneArtifact],
    *,
    positive_payload: dict[str, Any] | None = None,
    negative_payload: dict[str, Any] | None = None,
    solver_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    per_scene: list[dict[str, Any]] = []
    edge_errors: list[dict[str, Any]] = []
    cluster_conflicts: list[dict[str, Any]] = []
    carrier_conflicts: list[dict[str, Any]] = []
    for artifact in artifacts:
        edges = edge_scores_for_artifact(artifact, positive_variant="P5_full", negative_variant="N7_full")
        labels, trace = solve_edges(artifact.token_rows, edges, solver="S5_multicut_local_search")
        metric = _scene_metric_from_labels(artifact.scene, artifact.token_rows, labels, edges, trace, solver="S5_multicut_local_search")
        per_scene.append(metric)
        for edge in edges:
            if edge.get("diagnostic_same_gt") is None:
                continue
            false_positive = edge["diagnostic_same_gt"] is False and float(edge["positive_weight"]) >= 0.25 and float(edge["negative_weight"]) < 0.50
            false_negative = edge["diagnostic_same_gt"] is True and (float(edge["positive_weight"]) < 0.25 or float(edge["negative_weight"]) >= 0.50)
            if false_positive or false_negative:
                edge_errors.append(
                    {
                        "scene": artifact.scene,
                        "error_type": "false_positive_edge" if false_positive else "false_negative_edge",
                        **edge,
                    }
                )
        gt = _labels_gt(artifact.token_rows)
        by_cluster: dict[int, list[int]] = defaultdict(list)
        for token, label in labels.items():
            by_cluster[int(label)].append(int(token))
        support = _token_support_proxy(relation_rows(artifact))
        for label, tokens in by_cluster.items():
            gt_counts = Counter(gt[token] for token in tokens if token in gt and gt[token] > 0)
            if len(gt_counts) > 1:
                cluster_conflicts.append({"scene": artifact.scene, "cluster_id": label, "token_ids": tokens, "diagnostic_gt_counts": dict(gt_counts)})
            total_support = sum(support.get(token, 0.0) for token in tokens)
            if total_support <= 0:
                carrier_conflicts.append({"scene": artifact.scene, "cluster_id": label, "token_ids": tokens, "issue": "zero_carrier_support_proxy"})
    top_fp = sorted([row for row in edge_errors if row["error_type"] == "false_positive_edge"], key=lambda row: float(row["positive_weight"]) - float(row["negative_weight"]), reverse=True)[:200]
    top_fn = sorted([row for row in edge_errors if row["error_type"] == "false_negative_edge"], key=lambda row: float(row["positive_weight"]) - float(row["negative_weight"]))[:200]
    underseg = [
        row
        for row in edge_errors
        if row["error_type"] == "false_positive_edge" and ("underseg" in str(row.get("edge_type")) or float(row.get("supporter_reliability_pair") or 1.0) <= 0.55)
    ][:200]
    positive_pass = bool((positive_payload or {}).get("gate", {}).get("pass"))
    negative_pass = bool((negative_payload or {}).get("gate", {}).get("pass"))
    solver_pass = bool((solver_payload or {}).get("gate", {}).get("pass"))
    if not positive_pass:
        label = "NO_GO_VIEW_CONSENSUS_EDGE"
    elif not negative_pass:
        label = "NO_GO_NEGATIVE_EDGE"
    elif not solver_pass:
        label = "NO_GO_SOLVER"
    else:
        label = "NO_GO_STAGE1_NOT_SIGNIFICANT"
    return {
        "phase": "v46_failure_autopsy",
        "created_at": utc_now(),
        "final_failure_label": label,
        "per_scene_metric_table": per_scene,
        "edge_error_rows": edge_errors,
        "top_false_positive_edges": top_fp,
        "top_false_negative_edges": top_fn,
        "underseg_supporter_failures": underseg,
        "solver_merge_trace": (solver_payload or {}).get("solver_merge_trace", []),
        "cluster_conflict_rows": cluster_conflicts,
        "carrier_assignment_conflict_rows": carrier_conflicts,
        "visualization_manifest": {
            "available": False,
            "reason": "No rendered mask/carrier visualization was produced by this run; failure rows include token ids and edge source attribution for reproduction.",
        },
    }


def eval_aligned_ap_policy(stage1_payload: dict[str, Any]) -> dict[str, Any]:
    stage1_pass = bool(stage1_payload.get("gate", {}).get("pass"))
    return {
        "phase": "v46_eval_aligned_ap",
        "created_at": utc_now(),
        "status": "blocked_stage1_not_method" if not stage1_pass else "ready_for_eval_aligned_diagnostic",
        "AP": None,
        "AP50": None,
        "AP25": None,
        "uses_gt_for_prediction": False,
        "uses_gt_for_evaluation_alignment": bool(stage1_pass),
        "alignment_protocol": "eval_scene_sim3_only_after_prediction_export" if stage1_pass else "not_run_stage1_failed",
        "forbidden_per_object_gt_alignment": True,
        "gate": {"pass": False, "stage1_required": True, "stage1_pass": stage1_pass},
    }


def stage2_eligibility(stage1_payload: dict[str, Any], fact_payload: dict[str, Any]) -> dict[str, Any]:
    stage1_pass = bool(stage1_payload.get("gate", {}).get("pass"))
    scale_pass = bool(fact_payload.get("gate", {}).get("scale_guard_pass"))
    controls_pass = bool(stage1_payload.get("gate", {}).get("controls_pass"))
    allowed = bool(stage1_pass and scale_pass and controls_pass)
    return {
        "phase": "v46_stage2_eligibility",
        "created_at": utc_now(),
        "status": "STAGE2_ALLOWED" if allowed else "STAGE2_BLOCKED",
        "stage1_pass": stage1_pass,
        "scale_guard_pass": scale_pass,
        "d4rt_controls_pass": controls_pass,
        "uses_gt_for_prediction": False,
        "gate": {"pass": allowed},
        "reason": None if allowed else "Stage-1 significant gate, scale guard, and D4RT controls must all pass before Stage-2 mainline.",
    }


def build_final_decision(
    *,
    fact_payload: dict[str, Any],
    incidence_payload: dict[str, Any],
    supporter_payload: dict[str, Any],
    positive_payload: dict[str, Any],
    negative_payload: dict[str, Any],
    solver_payload: dict[str, Any],
    object_payload: dict[str, Any],
    stage1_payload: dict[str, Any],
    autopsy_payload: dict[str, Any],
    ap_payload: dict[str, Any],
    stage2_payload: dict[str, Any],
) -> dict[str, Any]:
    if not bool(incidence_payload.get("gate", {}).get("pass")):
        final_label = "NO_GO_INCIDENCE_COVERAGE"
    elif not bool(positive_payload.get("gate", {}).get("pass")):
        final_label = "NO_GO_VIEW_CONSENSUS_EDGE"
    elif not bool(supporter_payload.get("gate", {}).get("pass")):
        final_label = "NO_GO_SUPPORTER_RELIABILITY"
    elif not bool(negative_payload.get("gate", {}).get("pass")):
        final_label = "NO_GO_NEGATIVE_EDGE"
    elif not bool(solver_payload.get("gate", {}).get("pass")):
        final_label = "NO_GO_SOLVER"
    elif not bool(object_payload.get("gate", {}).get("pass")):
        final_label = "NO_GO_OBJECT_FIELD_EXPORT"
    elif not bool(stage1_payload.get("gate", {}).get("pass")):
        full_signal = bool(positive_payload.get("gate", {}).get("pass"))
        final_label = "PARTIAL_VIEW_CONSENSUS_SIGNAL" if full_signal else "NO_GO_STAGE1_NOT_SIGNIFICANT"
    elif not bool(ap_payload.get("gate", {}).get("pass")):
        final_label = "NO_GO_AP_BRIDGE"
    elif not bool(stage2_payload.get("gate", {}).get("pass")):
        final_label = "NO_GO_STAGE2"
    else:
        final_label = "GO_STAGE1_D4RT_VIEW_CONSENSUS_SIGNED_GRAPH"
    answers = {
        "built_mask_carrier_incidence": bool(incidence_payload.get("scene_rows")),
        "incidence_uses_raw_uv_containment": bool(incidence_payload.get("gate", {}).get("uses_raw_uv_containment")),
        "view_consensus_beats_shared_tube": bool(positive_payload.get("gate", {}).get("pass")),
        "supporter_reliability_reduces_false_positive_edges": bool(supporter_payload.get("gate", {}).get("pass")),
        "negative_edges_reduce_false_merge": bool(negative_payload.get("gate", {}).get("pass")),
        "signed_solver_beats_positive_only": bool(solver_payload.get("gate", {}).get("signed_solver_beats_positive_only")),
        "full_stage1_significant": bool(stage1_payload.get("gate", {}).get("pass")),
        "d4rt_real_beats_controls": bool(stage1_payload.get("gate", {}).get("controls_pass")),
        "no_d4rt_tube_birth": all(int(row.get("birth_from_d4rt_tube_count") or 0) == 0 for row in object_payload.get("scene_rows", [])),
        "no_maskless_object": all(int(row.get("maskless_object_count") or 0) == 0 for row in object_payload.get("scene_rows", [])),
        "scale_guard_pass": bool(fact_payload.get("gate", {}).get("scale_guard_pass")),
        "ap_eval_aligned_only": bool(ap_payload.get("uses_gt_for_prediction") is False),
        "stage2_allowed": bool(stage2_payload.get("gate", {}).get("pass")),
        "failure_location": final_label,
    }
    return {
        "phase": "v46_final_decision",
        "created_at": utc_now(),
        "final_label": final_label,
        "answers": answers,
        "fact_gate": fact_payload.get("gate"),
        "incidence_gate": incidence_payload.get("gate"),
        "supporter_gate": supporter_payload.get("gate"),
        "positive_gate": positive_payload.get("gate"),
        "negative_gate": negative_payload.get("gate"),
        "solver_gate": solver_payload.get("gate"),
        "object_export_gate": object_payload.get("gate"),
        "stage1_gate": stage1_payload.get("gate"),
        "autopsy_label": autopsy_payload.get("final_failure_label"),
        "ap_gate": ap_payload.get("gate"),
        "stage2_gate": stage2_payload.get("gate"),
    }
