from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v44_typed_mask_assembly import (
    SceneArtifact,
    _edge_tokens,
    _frame_rank_map,
    _object_sets_from_edges,
    _quantile,
    _token_id,
    as_float,
    auc_score,
    cluster_metrics,
    compactness_gate,
    d4rt_score,
    load_scene_artifacts,
    parse_bool,
    read_json,
    threshold_gate,
    true_role,
    utc_now,
    write_csv,
    write_json,
)


V45_STAGE1_GATE = {
    "4D_ARI": 0.485,
    "4D_purity": 0.875,
    "4D_completeness": 0.555,
    "temporal_span_mean": 1.70,
    "scene0081_ARI": 0.270,
}


def _token_score_from_edges(edge_rows: list[dict[str, Any]], field: str) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for edge in edge_rows:
        left, right = _edge_tokens(edge)
        value = as_float(edge.get(field))
        if value is None:
            continue
        values[left].append(float(value))
        values[right].append(float(value))
    return {token: float(np.mean(vals)) for token, vals in values.items() if vals}


def _cannot_ratio(edge_rows: list[dict[str, Any]]) -> dict[int, float]:
    cannot: dict[int, int] = defaultdict(int)
    total: dict[int, int] = defaultdict(int)
    for edge in edge_rows:
        left, right = _edge_tokens(edge)
        total[left] += 1
        total[right] += 1
        if parse_bool(edge.get("same_frame_cannot_link")):
            cannot[left] += 1
            cannot[right] += 1
    return {token: float(cannot[token] / max(total[token], 1)) for token in total}


def _relation_rows(artifact: SceneArtifact) -> list[dict[str, Any]]:
    return artifact.edge_rows if artifact.edge_rows else artifact.alignment_rows


def descriptor_audit_v45(
    artifacts: list[SceneArtifact],
    *,
    baseline_artifacts: list[SceneArtifact] | None = None,
    feature_smokes: list[Path] | None = None,
) -> dict[str, Any]:
    baseline_by_scene = {artifact.scene: artifact for artifact in baseline_artifacts or []}
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        baseline = baseline_by_scene.get(artifact.scene)
        token_rows = artifact.token_rows
        edge_rows = artifact.edge_rows
        areas = [float(as_float(row.get("area")) or 0.0) for row in token_rows]
        boundaries = [float(as_float(row.get("boundary_contrast")) or 0.0) for row in token_rows]
        area_median = _quantile(areas, 0.50)
        roles = [true_role(row, area_median=area_median) for row in token_rows]
        mixed_labels = [role == "mixed" for role in roles if role is not None]
        boundary_scores = [boundaries[idx] for idx, role in enumerate(roles) if role is not None]
        part_labels = [role == "part" for role in roles if role is not None]
        object_by_token = _token_score_from_edges(edge_rows, "object_affinity")
        part_scores = [object_by_token.get(_token_id(row), 0.0) for row, role in zip(token_rows, roles) if role is not None]
        same_labels: list[bool] = []
        semantic_scores: list[float] = []
        object_scores: list[float] = []
        for edge in edge_rows:
            if str(edge.get("diagnostic_same_gt", "")).strip() == "":
                continue
            same_labels.append(parse_bool(edge.get("diagnostic_same_gt")))
            semantic_scores.append(float(as_float(edge.get("semantic_affinity")) or 0.0))
            object_scores.append(float(as_float(edge.get("object_affinity")) or 0.0))
        baseline_auc = None
        baseline_backend = None
        if baseline and baseline.source_rows:
            baseline_auc = as_float(baseline.source_rows[0].get("semantic_affinity_AUC"))
            baseline_backend = baseline.source_rows[0].get("feature_backend")
        semantic_auc = auc_score(same_labels, semantic_scores)
        object_auc = auc_score(same_labels, object_scores)
        mixed_auc = auc_score(mixed_labels, boundary_scores)
        part_absorb_auc = auc_score(part_labels, part_scores)
        source = artifact.source_rows[0] if artifact.source_rows else {}
        semantic_auc_source_fallback = False
        object_auc_source_fallback = False
        if semantic_auc is None:
            semantic_auc = as_float(source.get("semantic_affinity_AUC"))
            semantic_auc_source_fallback = semantic_auc is not None
        if object_auc is None:
            object_auc = as_float(source.get("object_part_compatibility_AUC"))
            object_auc_source_fallback = object_auc is not None
        row = {
            "scene": artifact.scene,
            "feature_backend_used": source.get("feature_backend"),
            "feature_checkpoint": source.get("feature_checkpoint"),
            "baseline_backend": baseline_backend,
            "descriptor_success_rate": float(sum(float(a) > 0.0 for a in areas) / max(len(areas), 1)),
            "core_nonempty_rate": float(sum(float(a) >= 16.0 for a in areas) / max(len(areas), 1)),
            "prototype_count_mean": float(np.mean([1 + int(a >= _quantile(areas, 0.75)) for a in areas])) if areas else None,
            "semantic_same_gt_AUC": semantic_auc,
            "semantic_same_gt_AUC_source_fallback": semantic_auc_source_fallback,
            "object_context_same_gt_AUC": object_auc,
            "object_context_same_gt_AUC_source_fallback": object_auc_source_fallback,
            "baseline_same_gt_AUC": baseline_auc,
            "semantic_minus_baseline_AUC": None if semantic_auc is None or baseline_auc is None else float(semantic_auc - baseline_auc),
            "mixed_mask_AUC": mixed_auc,
            "part_absorb_AUC": part_absorb_auc,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        row["gate_pass"] = bool(
            row["descriptor_success_rate"] >= 0.98
            and row["core_nonempty_rate"] >= 0.90
            and (
                (row["semantic_minus_baseline_AUC"] is not None and row["semantic_minus_baseline_AUC"] >= 0.05)
                or (mixed_auc is not None and mixed_auc >= 0.65)
                or (part_absorb_auc is not None and part_absorb_auc >= 0.65)
            )
        )
        rows.append(row)
    backend_rows: list[dict[str, Any]] = []
    for path in feature_smokes or []:
        payload = read_json(path) or {}
        backend_rows.append(
            {
                "path": str(path),
                "backend": payload.get("backend"),
                "checkpoint": payload.get("checkpoint"),
                "gate_pass": bool(payload.get("gate_pass")),
                "radio_lang_align": payload.get("radio_lang_align"),
            }
        )
    gate = {
        "descriptor_success_rate_pass": bool(rows and all(float(row["descriptor_success_rate"]) >= 0.98 for row in rows)),
        "semantic_descriptor_available": any(bool(row.get("gate_pass")) for row in backend_rows),
        "descriptor_signal_pass": any(bool(row.get("gate_pass")) for row in rows),
        "all_scene_gate_pass": bool(rows and all(bool(row.get("gate_pass")) for row in rows)),
    }
    gate["pass"] = bool(gate["all_scene_gate_pass"])
    return {"phase": "v45_mask_descriptor_audit", "created_at": utc_now(), "scene_rows": rows, "backend_rows": backend_rows, "gate": gate}


def infer_roles_v45(
    token_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    *,
    profile: str,
) -> dict[int, dict[str, Any]]:
    areas = [float(as_float(row.get("area")) or 0.0) for row in token_rows]
    boundaries = [float(as_float(row.get("boundary_contrast")) or 0.0) for row in token_rows]
    area_q55 = _quantile(areas, 0.55)
    area_q70 = _quantile(areas, 0.70)
    area_q85 = _quantile(areas, 0.85)
    boundary_q60 = _quantile(boundaries, 0.60)
    boundary_q75 = _quantile(boundaries, 0.75)
    cannot = _cannot_ratio(edge_rows)
    object_score = _token_score_from_edges(edge_rows, "object_affinity")
    semantic_score = _token_score_from_edges(edge_rows, "semantic_affinity")
    duplicate_hits: dict[int, int] = defaultdict(int)
    for edge in edge_rows:
        left, right = _edge_tokens(edge)
        if (
            not parse_bool(edge.get("same_frame_cannot_link"))
            and float(as_float(edge.get("semantic_affinity")) or 0.0) >= 0.88
            and float(as_float(edge.get("object_affinity")) or 0.0) >= 0.58
        ):
            duplicate_hits[left] += 1
            duplicate_hits[right] += 1
    roles: dict[int, dict[str, Any]] = {}
    for row in token_rows:
        token = _token_id(row)
        area = float(as_float(row.get("area")) or 0.0)
        boundary = float(as_float(row.get("boundary_contrast")) or 0.0)
        cannot_value = float(cannot.get(token, 0.0))
        obj = float(object_score.get(token, 0.0))
        sem = float(semantic_score.get(token, 0.0))
        mixed_signals = int(area >= area_q85) + int(boundary >= boundary_q75) + int(cannot_value >= 0.35)
        role = "unknown"
        if duplicate_hits.get(token, 0) >= 2 and cannot_value < 0.20:
            role = "duplicate"
        elif profile in {"R2_parent_child_lattice", "R4_full_role_lattice"} and mixed_signals >= 2:
            role = "mixed"
        elif profile in {"R3_d4rt_divergence", "R4_full_role_lattice"} and cannot_value >= 0.45 and boundary >= boundary_q60:
            role = "mixed"
        elif area >= area_q70 and sem >= 0.45 and cannot_value < 0.40:
            role = "core"
        elif area <= area_q55 and (obj >= 0.20 or sem >= 0.45) and cannot_value < 0.55:
            role = "part"
        elif area >= area_q70 and cannot_value < 0.25:
            role = "core"
        elif obj >= 0.35 and cannot_value < 0.40:
            role = "part"
        roles[token] = {
            "token_id": token,
            "role": role,
            "area": area,
            "boundary_contrast": boundary,
            "cannot_link_ratio": cannot_value,
            "duplicate_hits": int(duplicate_hits.get(token, 0)),
            "semantic_score": sem,
            "object_score": obj,
            "profile": profile,
        }
    return roles


def role_metrics_v45(token_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]], roles: dict[int, dict[str, Any]]) -> dict[str, Any]:
    areas = [float(as_float(row.get("area")) or 0.0) for row in token_rows]
    area_median = _quantile(areas, 0.50)
    out: dict[str, Any] = {}
    for role in ("core", "part", "mixed", "duplicate"):
        tp = fp = fn = 0
        for row in token_rows:
            truth = true_role(row, area_median=area_median)
            pred = roles.get(_token_id(row), {}).get("role")
            if pred == role and truth == role:
                tp += 1
            elif pred == role and truth != role:
                fp += 1
            elif pred != role and truth == role:
                fn += 1
        out[f"{role}_precision"] = float(tp / max(tp + fp, 1))
        out[f"{role}_recall"] = float(tp / max(tp + fn, 1))
        out[f"{role}_tp"] = int(tp)
        out[f"{role}_fp"] = int(fp)
        out[f"{role}_fn"] = int(fn)
    out["duplicate_precision_evaluable"] = bool(out["duplicate_tp"] + out["duplicate_fp"] + out["duplicate_fn"] > 0)
    out["duplicate_precision_gate_pass"] = bool(
        (not out["duplicate_precision_evaluable"] and out["duplicate_fp"] == 0)
        or (out["duplicate_precision_evaluable"] and out["duplicate_precision"] >= 0.75)
    )
    unknown = sum(1 for item in roles.values() if item["role"] == "unknown")
    out["unknown_rate"] = float(unknown / max(len(roles), 1))
    false_merge = total = 0
    for edge in edge_rows:
        left, right = _edge_tokens(edge)
        if roles.get(left, {}).get("role") in {"mixed", "unknown"} or roles.get(right, {}).get("role") in {"mixed", "unknown"}:
            continue
        if parse_bool(edge.get("same_frame_cannot_link")):
            continue
        if float(as_float(edge.get("object_affinity")) or 0.0) < 0.50:
            continue
        total += 1
        false_merge += int(not parse_bool(edge.get("diagnostic_same_gt")))
    out["same_frame_false_merge_rate"] = float(false_merge / max(total, 1))
    out["mixed_false_accept_rate"] = float((out["mixed_fn"]) / max(out["mixed_fn"] + out["mixed_tp"], 1))
    return out


def role_lattice_audit_v45(artifacts: list[SceneArtifact]) -> dict[str, Any]:
    profiles = ["R0_v44_heuristic", "R1_semantic_descriptor_role", "R2_parent_child_lattice", "R3_d4rt_divergence", "R4_full_role_lattice"]
    rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    for profile in profiles:
        for artifact in artifacts:
            relation_rows = _relation_rows(artifact)
            roles = infer_roles_v45(artifact.token_rows, relation_rows, profile=profile)
            metrics = role_metrics_v45(artifact.token_rows, relation_rows, roles)
            row = {"scene": artifact.scene, "profile": profile, **metrics}
            row["gate_pass"] = bool(
                row["core_precision"] >= 0.75
                and row["part_precision"] >= 0.72
                and row["mixed_precision"] >= 0.80
                and row["mixed_false_accept_rate"] <= 0.12
                and row["duplicate_precision_gate_pass"]
                and row["unknown_rate"] <= 0.40
            )
            rows.append(row)
            role_rows.extend({"scene": artifact.scene, **item} for item in roles.values())
    best_profile = max(
        profiles,
        key=lambda name: sum(
            float(row.get("core_precision", 0.0)) + float(row.get("part_precision", 0.0)) + float(row.get("mixed_precision", 0.0))
            for row in rows
            if row["profile"] == name
        ),
    ) if rows else None
    gate = {
        "best_profile": best_profile,
        "any_scene_gate_pass": any(bool(row["gate_pass"]) for row in rows),
        "all_scene_gate_pass": bool(rows and best_profile and all(bool(row["gate_pass"]) for row in rows if row["profile"] == best_profile)),
    }
    gate["pass"] = bool(gate["all_scene_gate_pass"])
    return {"phase": "v45_role_lattice_audit", "created_at": utc_now(), "rows": rows, "role_rows": role_rows, "best_profile": best_profile, "gate": gate}


def _alignment_by_pair(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        left, right = _edge_tokens(row)
        out[(left, right)] = row
        out[(right, left)] = row
    return out


def typed_operation_audit_v45(artifacts: list[SceneArtifact], *, role_profile: str = "R4_full_role_lattice") -> dict[str, Any]:
    profiles = [
        {"operation_profile": "O1_semantic_only", "semantic_min": 0.72, "object_min": 0.45, "d4rt_min": -2.0, "visible_max": 1.0},
        {"operation_profile": "O3_semantic_d4rt_visible_veto", "semantic_min": 0.72, "object_min": 0.50, "d4rt_min": -0.55, "visible_max": 0.55},
        {"operation_profile": "O7_full_typed_operations", "semantic_min": 0.78, "object_min": 0.58, "d4rt_min": -0.35, "visible_max": 0.45},
    ]
    rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    for profile in profiles:
        for artifact in artifacts:
            relation_rows = _relation_rows(artifact)
            roles = infer_roles_v45(artifact.token_rows, relation_rows, profile=role_profile)
            align = _alignment_by_pair(artifact.alignment_rows)
            truth = {
                _token_id(row): true_role(row, area_median=_quantile([float(as_float(r.get("area")) or 0.0) for r in artifact.token_rows], 0.50))
                for row in artifact.token_rows
            }
            absorb_tp = absorb_fp = absorb_fn = 0
            mixed_touch = mixed_false_accept = 0
            d4rt_labels: list[bool] = []
            d4rt_scores: list[float] = []
            shuffled_scores: list[float] = []
            visible_labels: list[bool] = []
            visible_scores: list[float] = []
            for edge in relation_rows:
                left, right = _edge_tokens(edge)
                left_role = roles.get(left, {}).get("role")
                right_role = roles.get(right, {}).get("role")
                same = parse_bool(edge.get("diagnostic_same_gt"))
                is_absorb = {left_role, right_role} == {"part", "core"}
                if is_absorb and same:
                    absorb_fn += 1
                arow = align.get((left, right), edge)
                real_d4rt = d4rt_score(arow)
                visible = float(as_float(arow.get("visible_outside_conflict_ratio")) or 0.0)
                if str(arow.get("diagnostic_same_gt", "")).strip() != "":
                    d4rt_labels.append(parse_bool(arow.get("diagnostic_same_gt")))
                    d4rt_scores.append(real_d4rt)
                    shuffled_scores.append(d4rt_score(arow, shuffled=True))
                    visible_labels.append(not parse_bool(arow.get("diagnostic_same_gt")))
                    visible_scores.append(visible)
                accept = (
                    is_absorb
                    and left_role != "mixed"
                    and right_role != "mixed"
                    and not parse_bool(edge.get("same_frame_cannot_link"))
                    and float(as_float(edge.get("semantic_affinity")) or 0.0) >= float(profile["semantic_min"])
                    and float(as_float(edge.get("object_affinity")) or 0.0) >= float(profile["object_min"])
                    and real_d4rt >= float(profile["d4rt_min"])
                    and visible <= float(profile["visible_max"])
                )
                if truth.get(left) == "mixed" or truth.get(right) == "mixed":
                    mixed_touch += 1
                    mixed_false_accept += int(accept)
                if accept:
                    absorb_tp += int(same)
                    absorb_fp += int(not same)
                    absorb_fn -= int(same)
                    accepted_rows.append({"scene": artifact.scene, **profile, **edge, "d4rt_score": real_d4rt, "visible_outside": visible})
            absorb_precision = float(absorb_tp / max(absorb_tp + absorb_fp, 1))
            absorb_recall = float(absorb_tp / max(absorb_tp + absorb_fn, 1))
            real_auc = auc_score(d4rt_labels, d4rt_scores)
            shuffled_auc = auc_score(d4rt_labels, shuffled_scores)
            visible_auc = auc_score(visible_labels, visible_scores)
            row = {
                "scene": artifact.scene,
                "role_profile": role_profile,
                **profile,
                "absorb_precision": absorb_precision,
                "absorb_recall": absorb_recall,
                "false_absorb_rate": float(absorb_fp / max(absorb_tp + absorb_fp, 1)),
                "mixed_false_accept_rate": float(mixed_false_accept / max(mixed_touch, 1)),
                    "mixed_reject_precision": role_metrics_v45(artifact.token_rows, relation_rows, roles)["mixed_precision"],
                "D4RT_real_absorb_AUC": real_auc,
                "D4RT_shuffled_absorb_AUC": shuffled_auc,
                "D4RT_real_minus_shuffled_absorb_AUC": None if real_auc is None or shuffled_auc is None else float(real_auc - shuffled_auc),
                "D4RT_visible_outside_veto_precision": visible_auc,
            }
            row["gate_pass"] = bool(
                row["absorb_precision"] >= 0.78
                and row["false_absorb_rate"] <= 0.15
                and row["mixed_reject_precision"] >= 0.80
                and row["mixed_false_accept_rate"] <= 0.12
                and row["D4RT_real_minus_shuffled_absorb_AUC"] is not None
                and row["D4RT_real_minus_shuffled_absorb_AUC"] >= 0.08
                and row["D4RT_visible_outside_veto_precision"] is not None
                and row["D4RT_visible_outside_veto_precision"] >= 0.78
            )
            rows.append(row)
    best_profile = max(
        {row["operation_profile"] for row in rows},
        key=lambda name: sum(float(row.get("absorb_precision", 0.0)) - float(row.get("false_absorb_rate", 0.0)) for row in rows if row["operation_profile"] == name),
    ) if rows else None
    gate = {
        "best_operation_profile": best_profile,
        "any_scene_gate_pass": any(bool(row["gate_pass"]) for row in rows),
        "all_scene_gate_pass": bool(rows and best_profile and all(bool(row["gate_pass"]) for row in rows if row["operation_profile"] == best_profile)),
    }
    gate["pass"] = bool(gate["all_scene_gate_pass"])
    return {"phase": "v45_typed_operation_audit", "created_at": utc_now(), "rows": rows, "accepted_rows": accepted_rows, "best_operation_profile": best_profile, "gate": gate}


def temporal_matching_audit_v45(artifacts: list[SceneArtifact], *, role_profile: str = "R4_full_role_lattice") -> dict[str, Any]:
    profiles = [
        {"profile": "T3_semantic_d4rt", "semantic_min": 0.66, "d4rt_min": -0.55, "visible_max": 0.75, "top1": False},
        {"profile": "T4_visible_outside_veto", "semantic_min": 0.70, "d4rt_min": -0.50, "visible_max": 0.50, "top1": False},
        {"profile": "T5_role_aware_typed_link", "semantic_min": 0.72, "d4rt_min": -0.45, "visible_max": 0.45, "top1": True},
    ]
    rows: list[dict[str, Any]] = []
    d4rt_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        labels: list[bool] = []
        real_scores: list[float] = []
        shuffled_scores: list[float] = []
        visible_labels: list[bool] = []
        visible_scores: list[float] = []
        for row in artifact.alignment_rows:
            if str(row.get("diagnostic_same_gt", "")).strip() == "":
                continue
            same = parse_bool(row.get("diagnostic_same_gt"))
            labels.append(same)
            real_scores.append(d4rt_score(row))
            shuffled_scores.append(d4rt_score(row, shuffled=True))
            visible_labels.append(not same)
            visible_scores.append(float(as_float(row.get("visible_outside_conflict_ratio")) or 0.0))
        real_auc = auc_score(labels, real_scores)
        shuffled_auc = auc_score(labels, shuffled_scores)
        visible_auc = auc_score(visible_labels, visible_scores)
        d4rt_rows.append(
            {
                "scene": artifact.scene,
                "real_D4RT_link_AUC": real_auc,
                "shuffled_link_AUC": shuffled_auc,
                "real_minus_shuffled_link_AUC": None if real_auc is None or shuffled_auc is None else float(real_auc - shuffled_auc),
                "visible_outside_veto_precision": visible_auc,
            }
        )
    for profile in profiles:
        for artifact in artifacts:
            relation_rows = _relation_rows(artifact)
            roles = infer_roles_v45(artifact.token_rows, relation_rows, profile=role_profile)
            frame_rank = _frame_rank_map(artifact.token_rows)
            candidates = []
            for row in artifact.alignment_rows:
                left, right = _edge_tokens(row)
                if left not in frame_rank or right not in frame_rank:
                    continue
                if abs(frame_rank[left] - frame_rank[right]) not in {1, 2}:
                    continue
                if roles.get(left, {}).get("role") in {"mixed", "unknown"} or roles.get(right, {}).get("role") in {"mixed", "unknown"}:
                    continue
                candidates.append(row)
            prelim = [
                row
                for row in candidates
                if float(as_float(row.get("semantic_affinity")) or 0.0) >= float(profile["semantic_min"])
                and d4rt_score(row) >= float(profile["d4rt_min"])
                and float(as_float(row.get("visible_outside_conflict_ratio")) or 0.0) <= float(profile["visible_max"])
            ]
            accepted = prelim
            if profile["top1"]:
                best: dict[int, dict[str, Any]] = {}
                for row in prelim:
                    left, _right = _edge_tokens(row)
                    score = float(as_float(row.get("semantic_affinity")) or 0.0) + d4rt_score(row)
                    old = best.get(left)
                    if old is None or score > float(old.get("_score", -999.0)):
                        current = dict(row)
                        current["_score"] = score
                        best[left] = current
                accepted = list(best.values())
            true_edges = sum(1 for row in candidates if parse_bool(row.get("diagnostic_same_gt")))
            tp = sum(1 for row in accepted if parse_bool(row.get("diagnostic_same_gt")))
            precision = float(tp / max(len(accepted), 1))
            recall = float(tp / max(true_edges, 1))
            role_conflict = sum(
                1
                for row in accepted
                if roles.get(_edge_tokens(row)[0], {}).get("role") in {"mixed", "unknown"}
                or roles.get(_edge_tokens(row)[1], {}).get("role") in {"mixed", "unknown"}
            )
            row = {
                "scene": artifact.scene,
                **profile,
                "candidate_count": int(len(candidates)),
                "accepted_count": int(len(accepted)),
                "link_edge_precision": precision,
                "link_edge_recall": recall,
                "short_masklet_purity": precision,
                "short_masklet_completeness": recall,
                "role_conflict_link_rate": float(role_conflict / max(len(accepted), 1)),
            }
            row["gate_pass"] = bool(
                row["link_edge_precision"] >= 0.82
                and row["short_masklet_purity"] >= 0.88
                and row["short_masklet_completeness"] >= 0.55
                and row["role_conflict_link_rate"] <= 0.10
            )
            rows.append(row)
    best_profile = max(
        {row["profile"] for row in rows},
        key=lambda name: sum(float(row.get("link_edge_precision", 0.0)) + float(row.get("link_edge_recall", 0.0)) for row in rows if row["profile"] == name),
    ) if rows else None
    d4rt_gate = bool(
        d4rt_rows
        and all(
            row.get("real_minus_shuffled_link_AUC") is not None
            and float(row["real_minus_shuffled_link_AUC"]) >= 0.10
            and row.get("visible_outside_veto_precision") is not None
            and float(row["visible_outside_veto_precision"]) >= 0.78
            for row in d4rt_rows
        )
    )
    gate = {
        "best_profile": best_profile,
        "matching_gate_pass": bool(rows and best_profile and all(bool(row["gate_pass"]) for row in rows if row["profile"] == best_profile)),
        "d4rt_gate_pass": d4rt_gate,
    }
    gate["pass"] = bool(gate["matching_gate_pass"] and gate["d4rt_gate_pass"])
    return {"phase": "v45_temporal_matching_audit", "created_at": utc_now(), "rows": rows, "d4rt_rows": d4rt_rows, "best_profile": best_profile, "gate": gate}


def typed_energy_selection_diagnostic_v45(artifacts: list[SceneArtifact], *, role_profile: str = "R4_full_role_lattice") -> dict[str, Any]:
    variants = [
        {"variant": "F3_descriptor_old_selection", "semantic_min": 0.74, "d4rt_min": -2.0, "visible_max": 1.0},
        {"variant": "F5_roles_operations_old_selection", "semantic_min": 0.74, "d4rt_min": -0.50, "visible_max": 0.55},
        {"variant": "F6_typed_energy_selection", "semantic_min": 0.78, "d4rt_min": -0.35, "visible_max": 0.45},
        {"variant": "F7_typed_energy_shuffled_d4rt", "semantic_min": 0.78, "d4rt_min": -0.35, "visible_max": 0.45, "shuffled": True},
        {"variant": "F8_no_temporal", "semantic_min": 0.78, "d4rt_min": -99.0, "visible_max": 0.45, "no_temporal": True},
        {"variant": "F9_mask_only", "semantic_min": 99.0, "d4rt_min": -99.0, "visible_max": 1.0},
    ]
    variant_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for variant in variants:
        aggregate_pred: dict[int, int] = {}
        aggregate_gt: dict[int, int] = {}
        counts: list[int] = []
        unknown_ratios: list[float] = []
        for scene_index, artifact in enumerate(artifacts):
            relation_rows = _relation_rows(artifact)
            roles = infer_roles_v45(artifact.token_rows, relation_rows, profile=role_profile)
            frame_rank = _frame_rank_map(artifact.token_rows)
            selected: list[dict[str, Any]] = []
            for row in artifact.alignment_rows if artifact.alignment_rows else relation_rows:
                left, right = _edge_tokens(row)
                if variant.get("no_temporal") and left in frame_rank and right in frame_rank and frame_rank[left] != frame_rank[right]:
                    continue
                if roles.get(left, {}).get("role") in {"mixed", "unknown"} or roles.get(right, {}).get("role") in {"mixed", "unknown"}:
                    continue
                score = d4rt_score(row, shuffled=bool(variant.get("shuffled", False)))
                if (
                    float(as_float(row.get("semantic_affinity")) or 0.0) >= float(variant["semantic_min"])
                    and score >= float(variant["d4rt_min"])
                    and float(as_float(row.get("visible_outside_conflict_ratio")) or 0.0) <= float(variant["visible_max"])
                ):
                    selected.append(row)
            objects, pred, gt = _object_sets_from_edges(artifact.token_rows, selected, roles, max_fields=150)
            for obj in objects:
                object_rows.append({"scene": artifact.scene, "variant": variant["variant"], **obj})
            off = scene_index * 100_000_000
            aggregate_pred.update({off + token: off + label for token, label in pred.items()})
            aggregate_gt.update({off + token: off + label for token, label in gt.items()})
            metrics = cluster_metrics(pred, gt)
            assigned = {token for token, label in pred.items() if int(label) < 10_000_000}
            unknown = float((len(gt) - len(assigned & set(gt))) / max(len(gt), 1))
            unknown_ratios.append(unknown)
            counts.append(len(objects))
            scene_rows.append({"scene": artifact.scene, "variant": variant["variant"], "4D_ARI": metrics.get("ari"), "4D_purity": metrics.get("purity"), "4D_completeness": metrics.get("completeness"), "object_count": len(objects), "unknown_tube_ratio": unknown})
        agg = cluster_metrics(aggregate_pred, aggregate_gt)
        row = {
            "variant": variant["variant"],
            "evaluation_scope": "semantic_token_diagnostic_available_scenes",
            "scene_count": int(len(artifacts)),
            "4D_ARI": agg.get("ari"),
            "4D_purity": agg.get("purity"),
            "4D_completeness": agg.get("completeness"),
            "temporal_span_mean": 1.0,
            "scene0081_ARI": next((r["4D_ARI"] for r in scene_rows if r["variant"] == variant["variant"] and r["scene"] == "scene0081_01"), None),
            "mean_predictions_per_scene": float(np.mean(counts)) if counts else None,
            "duplicate_rate": 0.0,
            "conflict_rate": 0.0,
            "unknown_tube_ratio": float(np.mean(unknown_ratios)) if unknown_ratios else None,
            "birth_from_d4rt_tube_count": 0,
            "mixed_birth_count": 0,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        }
        row["minimum_gate"] = threshold_gate(row, V45_STAGE1_GATE)
        row["compactness_gate"] = compactness_gate(row)
        row["stage1_gate_pass"] = bool(row["minimum_gate"]["pass"] and row["compactness_gate"]["pass"] and row["scene_count"] >= 5)
        variant_rows.append(row)
    by_variant = {row["variant"]: row for row in variant_rows}
    f6 = by_variant.get("F6_typed_energy_selection", {})
    f7 = by_variant.get("F7_typed_energy_shuffled_d4rt", {})
    f8 = by_variant.get("F8_no_temporal", {})
    f9 = by_variant.get("F9_mask_only", {})
    controls = {
        "real_minus_shuffled_ARI": _metric_delta(f6, f7, "4D_ARI"),
        "real_minus_no_temporal_ARI": _metric_delta(f6, f8, "4D_ARI"),
        "real_minus_mask_only_ARI": _metric_delta(f6, f9, "4D_ARI"),
    }
    gate = {
        "any_stage1_gate_pass": any(bool(row["stage1_gate_pass"]) for row in variant_rows),
        "full_probe5_scope_available": bool(len(artifacts) >= 5),
        "typed_energy_control_gate_pass": bool(
            controls["real_minus_shuffled_ARI"] is not None
            and controls["real_minus_shuffled_ARI"] >= 0.30
            and controls["real_minus_no_temporal_ARI"] is not None
            and controls["real_minus_no_temporal_ARI"] >= 0.25
            and controls["real_minus_mask_only_ARI"] is not None
            and controls["real_minus_mask_only_ARI"] >= 0.25
        ),
    }
    gate["pass"] = bool(gate["any_stage1_gate_pass"] and gate["typed_energy_control_gate_pass"])
    return {
        "phase": "v45_typed_energy_selection_diagnostic",
        "created_at": utc_now(),
        "variant_rows": variant_rows,
        "scene_rows": scene_rows,
        "object_rows": object_rows,
        "controls": controls,
        "gate": gate,
    }


def _metric_delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    lval = as_float(left.get(key))
    rval = as_float(right.get(key))
    return None if lval is None or rval is None else float(lval - rval)


def preconditioned_stage1_status(*, descriptor: dict[str, Any], role: dict[str, Any], operations: dict[str, Any], temporal: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if not descriptor.get("gate", {}).get("pass", False):
        reasons.append("descriptor_gate_failed")
    if not role.get("gate", {}).get("pass", False):
        reasons.append("role_lattice_gate_failed")
    if not operations.get("gate", {}).get("pass", False):
        reasons.append("typed_operation_gate_failed")
    if not temporal.get("gate", {}).get("pass", False):
        reasons.append("temporal_or_d4rt_gate_failed")
    return {
        "phase": "v45_stage1_full",
        "created_at": utc_now(),
        "status": "blocked_by_preconditions" if reasons else "ready_for_full_probe5",
        "stage1_run_as_method": bool(not reasons),
        "blocked_reasons": reasons,
        "gate": {"pass": bool(not reasons)},
    }
