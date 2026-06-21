from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from stream4d_native.v49_mosaic_stage1 import (
    _greedy_select,
    _unpack_components,
    build_hypothesis_scoring,
    evaluate_component_assignment,
    load_optional_csv,
    load_optional_json,
    parse_float,
    project_path,
    write_csv,
    write_json,
)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    if count <= 0:
        return {"count": 0}
    return {
        "count": count,
        "same_rate": sum(bool(row.get("same_GT_set")) for row in rows) / count,
        "purity_mean": sum(parse_float(row.get("hypothesis_purity")) for row in rows) / count,
        "completeness_mean": sum(parse_float(row.get("hypothesis_completeness")) for row in rows) / count,
        "coverage_gain_mean": sum(parse_float(row.get("coverage_gain_over_singletons")) for row in rows) / count,
        "temporal_mean": sum(parse_float(row.get("temporal_support_score")) for row in rows) / count,
        "semantic_mean": sum(parse_float(row.get("semantic_set_score")) for row in rows) / count,
        "context_mean": sum(parse_float(row.get("context_overlap_proxy")) for row in rows) / count,
        "conflict_mean": sum(parse_float(row.get("hypothesis_conflict_rate")) for row in rows) / count,
        "scene_counts": dict(Counter(str(row.get("scene")) for row in rows).most_common()),
    }


def _is_certificate(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    size = parse_float(row.get("hypothesis_size"), 1.0)
    return bool(
        str(row.get("component_set_candidate_source")) == rule["source"]
        and size > 1.0
        and size <= parse_float(rule["size_max"])
        and parse_float(row.get("semantic_set_score")) >= parse_float(rule["sem_min"])
        and parse_float(row.get("temporal_support_score")) >= parse_float(rule["temporal_min"])
        and parse_float(row.get("coverage_gain_over_singletons")) <= parse_float(rule["coverage_max"])
        and parse_float(row.get("context_overlap_proxy")) <= parse_float(rule["context_max"])
        and parse_float(row.get("hypothesis_conflict_rate")) <= parse_float(rule["conf_max"])
        and parse_float(row.get("mask_reliability_min")) >= parse_float(rule["rel_min"])
    )


def _with_birth_scores(
    hypotheses: list[dict[str, Any]],
    rule: dict[str, Any],
    *,
    bonus: float,
    multi_penalty: float,
    singleton_penalty: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in hypotheses:
        scored = dict(row)
        size = parse_float(row.get("hypothesis_size"), 1.0)
        certificate_ok = _is_certificate(row, rule)
        non_certificate_penalty = -float(multi_penalty) if size > 1.0 else -float(singleton_penalty)
        scored["d4rt_birth_certificate_ok"] = certificate_ok
        scored["score_d4rt_birth_certificate"] = parse_float(row.get("score_boundary_prototype_context_hard")) + (
            float(bonus) if certificate_ok else non_certificate_penalty
        )
        scored["score_d4rt_birth_no_temporal_control"] = (
            parse_float(row.get("score_boundary_prototype_no_temporal")) + non_certificate_penalty
        )
        scored["score_d4rt_birth_mask_only_control"] = (
            parse_float(row.get("score_boundary_prototype_mask_only")) + non_certificate_penalty
        )
        scored["score_d4rt_birth_no_certificate_control"] = (
            parse_float(row.get("score_boundary_prototype_context_hard")) + non_certificate_penalty
        )
        out.append(scored)
    return out


def _evaluate_selected(mask_vote_rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    component_to_object: dict[str, str] = {}
    for index, hypothesis in enumerate(selected):
        object_id = f"{hypothesis.get('scene')}|dry|h{index:04d}"
        for component in _unpack_components(hypothesis.get("components")):
            if component:
                component_to_object[component] = object_id
    metrics = evaluate_component_assignment(mask_vote_rows, component_to_object)
    return {
        "4D_ARI": metrics.get("4D_ARI"),
        "4D_purity": metrics.get("4D_purity"),
        "4D_completeness": metrics.get("4D_completeness"),
        "conflict_rate": metrics.get("conflict_rate"),
        "mean_predictions_per_scene": metrics.get("mean_predictions_per_scene"),
        "scene0591_completeness": metrics.get("scene0591_completeness"),
        "scene0081_ARI": metrics.get("scene0081_ARI"),
        "scene0011_purity": metrics.get("scene0011_purity"),
        "scene0050_purity": metrics.get("scene0050_purity"),
        "selected_object_count": metrics.get("selected_object_count"),
        "object_size_mean": (
            sum(parse_float(hypothesis.get("hypothesis_size"), 1.0) for hypothesis in selected) / len(selected)
            if selected
            else 1.0
        ),
        "cert_selected": sum(1 for hypothesis in selected if hypothesis.get("d4rt_birth_certificate_ok")),
        "multi_selected": sum(1 for hypothesis in selected if parse_float(hypothesis.get("hypothesis_size")) > 1.0),
    }


def build_d4rt_required_candidate_region_audit(
    *,
    scoring_root: str = "outputs/audit/v49_hypothesis_scoring",
    mask_vote_rows_path: str = "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_mask_vote_rows.csv",
) -> dict[str, Any]:
    scoring = load_optional_json(project_path(scoring_root) / "hypothesis_scoring_summary.json")
    if scoring.get("missing"):
        scoring = build_hypothesis_scoring()
    hypotheses = list(scoring.get("hypothesis_rows", []))
    mask_vote_rows = load_optional_csv(mask_vote_rows_path)
    if not mask_vote_rows:
        mask_vote_rows = load_optional_csv(
            "outputs/audit/v47_carrier_supertrack_union_32_metricfix/carrier_supertrack_mask_vote_rows.csv"
        )
    raw_metrics = evaluate_component_assignment(mask_vote_rows)

    temporal_multi = [
        row
        for row in hypotheses
        if parse_float(row.get("hypothesis_size")) > 1.0 and parse_float(row.get("temporal_support_score")) > 0.0
    ]
    source_summaries = []
    for source in ["pair_edge", "pair_neighborhood", "multi_scale_parent_containment"]:
        source_rows = [row for row in temporal_multi if str(row.get("component_set_candidate_source")) == source]
        source_summaries.append({"source": source, **_summarize(source_rows)})

    rules = [
        {
            "name": "hi_purity_scene0591_low_context",
            "source": "pair_edge",
            "size_max": 2.5,
            "sem_min": 0.98,
            "temporal_min": 0.02,
            "coverage_max": 1,
            "context_max": 0.10,
            "rel_min": 0.55,
            "conf_max": 0.05,
        },
        {
            "name": "broad_low_context_family",
            "source": "pair_edge",
            "size_max": 2.5,
            "sem_min": 0.90,
            "temporal_min": 0.02,
            "coverage_max": 10,
            "context_max": 0.25,
            "rel_min": 0.55,
            "conf_max": 0.05,
        },
        {
            "name": "temporal_heavy_top_same_rate",
            "source": "pair_edge",
            "size_max": 2.5,
            "sem_min": 0.99,
            "temporal_min": 0.75,
            "coverage_max": 20,
            "context_max": 1.0,
            "rel_min": 0.0,
            "conf_max": 0.05,
        },
        {
            "name": "o29_like_low_temporal_overlap_not_temporal_required",
            "source": "pair_edge",
            "size_max": 2.5,
            "sem_min": 0.95,
            "temporal_min": 0.0,
            "coverage_max": 1,
            "context_max": 0.02,
            "rel_min": 0.0,
            "conf_max": 0.05,
        },
    ]
    params = [
        (0.20, 1.25, 0.05),
        (0.35, 1.25, 0.05),
        (0.50, 1.25, 0.05),
        (0.35, 0.75, 0.05),
        (0.35, 1.75, 0.05),
        (0.35, 1.25, 0.10),
    ]

    dry_rows: list[dict[str, Any]] = []
    for rule in rules:
        certificate_rows = [row for row in hypotheses if _is_certificate(row, rule)]
        certificate_summary = _summarize(certificate_rows)
        for bonus, multi_penalty, singleton_penalty in params:
            scored = _with_birth_scores(
                hypotheses,
                rule,
                bonus=bonus,
                multi_penalty=multi_penalty,
                singleton_penalty=singleton_penalty,
            )
            real = _evaluate_selected(
                mask_vote_rows, _greedy_select(scored, score_key="score_d4rt_birth_certificate", max_per_scene=150)
            )
            no_temporal = _evaluate_selected(
                mask_vote_rows,
                _greedy_select(scored, score_key="score_d4rt_birth_no_temporal_control", max_per_scene=150),
            )
            mask_only = _evaluate_selected(
                mask_vote_rows, _greedy_select(scored, score_key="score_d4rt_birth_mask_only_control", max_per_scene=150)
            )
            no_certificate = _evaluate_selected(
                mask_vote_rows,
                _greedy_select(scored, score_key="score_d4rt_birth_no_certificate_control", max_per_scene=150),
            )
            real_minus_no_temporal = parse_float(real.get("4D_ARI")) - parse_float(no_temporal.get("4D_ARI"))
            row = {
                **{f"rule_{key}": value for key, value in rule.items()},
                "bonus": bonus,
                "multi_penalty": multi_penalty,
                "singleton_penalty": singleton_penalty,
                "cert_count": certificate_summary.get("count"),
                "cert_same_rate": certificate_summary.get("same_rate"),
                "cert_purity_mean": certificate_summary.get("purity_mean"),
                "cert_completeness_mean": certificate_summary.get("completeness_mean"),
                "real_4D_ARI": real.get("4D_ARI"),
                "real_4D_purity": real.get("4D_purity"),
                "real_4D_completeness": real.get("4D_completeness"),
                "real_conflict_rate": real.get("conflict_rate"),
                "real_mean_predictions_per_scene": real.get("mean_predictions_per_scene"),
                "real_scene0591_completeness": real.get("scene0591_completeness"),
                "real_scene0081_ARI": real.get("scene0081_ARI"),
                "real_object_size_mean": real.get("object_size_mean"),
                "real_cert_selected": real.get("cert_selected"),
                "real_multi_selected": real.get("multi_selected"),
                "no_temporal_4D_ARI": no_temporal.get("4D_ARI"),
                "mask_only_4D_ARI": mask_only.get("4D_ARI"),
                "no_certificate_4D_ARI": no_certificate.get("4D_ARI"),
                "real_minus_no_temporal_ARI": real_minus_no_temporal,
                "real_minus_mask_only_ARI": parse_float(real.get("4D_ARI")) - parse_float(mask_only.get("4D_ARI")),
                "real_minus_no_certificate_ARI": parse_float(real.get("4D_ARI"))
                - parse_float(no_certificate.get("4D_ARI")),
            }
            row["candidate_region_passlike"] = bool(
                parse_float(row.get("real_4D_purity")) >= 0.875
                and parse_float(row.get("real_4D_completeness")) >= parse_float(raw_metrics.get("4D_completeness")) + 0.02
                and real_minus_no_temporal >= 0.02
            )
            dry_rows.append(row)

    geometry_paths = [
        "outputs/audit/v47_component_raw_geometry_merge_union32_gap2_smoke_fast/component_raw_geometry_merge_summary.json",
        "outputs/audit/v47_component_raw_geometry_merge_union32_gap2_expanded/component_raw_geometry_merge_summary.json",
        "outputs/audit/v47_component_raw_geometry_merge_union32_gap2_smoke/component_raw_geometry_merge_summary.json",
    ]
    geometry_summaries = []
    for rel_path in geometry_paths:
        path = project_path(rel_path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        geometry_summaries.append(
            {
                "path": rel_path,
                "component_count": payload.get("component_count"),
                "candidate_pair_count": payload.get("candidate_pair_count"),
                "candidate_pair_count_with_raw_geometry": payload.get("candidate_pair_count_with_raw_geometry"),
                "best_real_minus_best_no_temporal_ARI": payload.get("best_real_minus_best_no_temporal_ARI"),
                "best_real_minus_best_shuffled_ARI": payload.get("best_real_minus_best_shuffled_ARI"),
                "gate": payload.get("gate"),
                "best_real_row": payload.get("best_real_row"),
                "best_no_temporal_row": payload.get("best_no_temporal_row"),
                "best_shuffled_row": payload.get("best_shuffled_row"),
                "raw_summary": payload.get("raw_summary"),
            }
        )

    best_dry = (
        max(
            dry_rows,
            key=lambda row: (
                int(bool(row.get("candidate_region_passlike"))),
                parse_float(row.get("real_minus_no_temporal_ARI")),
                parse_float(row.get("real_4D_ARI")),
                parse_float(row.get("real_4D_purity")),
            ),
        )
        if dry_rows
        else {}
    )
    gate = {
        "has_temporal_required_candidate_region": any(
            parse_float(row.get("cert_count")) >= 20
            and parse_float(row.get("cert_same_rate")) >= 0.45
            and parse_float(row.get("cert_purity_mean")) >= 0.875
            for row in dry_rows
        ),
        "dry_run_passlike": any(bool(row.get("candidate_region_passlike")) for row in dry_rows),
        "geometry_prior_existing_gate_pass": any(bool((row.get("gate") or {}).get("pass")) for row in geometry_summaries),
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "phase": "v49_d4rt_required_candidate_region_audit",
        "input_hypothesis_summary": str(project_path(scoring_root) / "hypothesis_scoring_summary.json"),
        "raw_component_metrics": {
            key: raw_metrics.get(key)
            for key in [
                "4D_ARI",
                "4D_purity",
                "4D_completeness",
                "conflict_rate",
                "mean_predictions_per_scene",
                "scene0591_completeness",
            ]
        },
        "base_temporal_multi_summary": _summarize(temporal_multi),
        "source_temporal_summaries": source_summaries,
        "tested_rule_count": len(rules),
        "dry_run_count": len(dry_rows),
        "passlike_count_purity_completeness_control002": sum(
            1 for row in dry_rows if bool(row.get("candidate_region_passlike"))
        ),
        "best_dry_run_by_control_gap": best_dry,
        "geometry_prior_summaries": geometry_summaries,
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_D4RT_REQUIRED_COMPLETION_REGION",
        "recommendation": (
            "Do not promote O33 D4RT-required birth certificate from current artifacts; temporal-required candidate "
            "regions are too weak and raw-geometry prior summaries remain control-negative."
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "_dry_run_rows": dry_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v49 D4RT-required candidate-region diagnostic.")
    parser.add_argument("--output-root", default="outputs/audit/v49_d4rt_required_candidate_region")
    args = parser.parse_args()

    payload = build_d4rt_required_candidate_region_audit()
    dry_rows = list(payload.pop("_dry_run_rows", []))
    out_root = project_path(args.output_root)
    write_json(out_root / "d4rt_required_candidate_region_summary.json", payload)
    write_csv(out_root / "d4rt_required_candidate_region_dry_run_rows.csv", dry_rows)
    print(
        {
            "summary": str(out_root / "d4rt_required_candidate_region_summary.json"),
            "dry_rows": str(out_root / "d4rt_required_candidate_region_dry_run_rows.csv"),
            "gate": payload["gate"],
            "failure_label": payload["failure_label"],
        }
    )


if __name__ == "__main__":
    main()
