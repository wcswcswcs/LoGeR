from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from stream4d_native.v47_common import ROOT, parse_bool, parse_float, read_csv, read_json, utc_now, write_csv, write_json


def _load_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _load_csv(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.exists() else []


def _fact_map(fact: dict[str, Any]) -> dict[str, Any]:
    return {str(row.get("key")): row.get("value") for row in fact.get("fact_rows", [])}


def _metric(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row.get(name) not in (None, ""):
            return row.get(name)
    return None


def _stage_row(
    *,
    variant: str,
    scheme: str,
    source: str,
    row: dict[str, Any] | None = None,
    status: str = "ok",
    note: str = "",
) -> dict[str, Any]:
    row = row or {}
    out = {
        "variant": variant,
        "scheme": scheme,
        "status": status,
        "source": source,
        "ARI": _metric(row, "ARI", "final_ARI", "best_ARI"),
        "purity": _metric(row, "purity", "final_purity", "best_purity"),
        "completeness": _metric(row, "completeness", "final_completeness", "best_completeness"),
        "temporal_span_mean": _metric(row, "temporal_span_mean"),
        "scene0081_ARI": _metric(row, "scene0081_ARI"),
        "scene0011_purity": _metric(row, "scene0011_purity"),
        "scene0050_purity": _metric(row, "scene0050_purity"),
        "scene0591_completeness": _metric(row, "scene0591_completeness"),
        "mean_predictions_per_scene": _metric(row, "mean_predictions_per_scene"),
        "selected_candidate_count": _metric(row, "selected_candidate_count", "selected_object_count", "track_count", "cluster_count"),
        "duplicate_rate": _metric(row, "duplicate_rate", "duplicate_component_ratio"),
        "conflict_rate": _metric(row, "conflict_rate"),
        "unknown_tube_ratio": _metric(row, "unknown_tube_ratio", "final_unknown_ratio"),
        "birth_from_d4rt_tube_count": row.get("birth_from_d4rt_tube_count", 0 if row else None),
        "maskless_object_count": row.get("maskless_object_count", 0 if row else None),
        "real_minus_shuffled_ARI": _metric(row, "real_minus_shuffled_ARI", "real_minus_shuffled_d4rt_ARI_same_params", "best_real_minus_best_shuffled_ARI"),
        "real_minus_no_temporal_ARI": _metric(row, "real_minus_no_temporal_ARI", "real_minus_no_temporal_ARI_same_params", "best_real_minus_best_no_temporal_ARI"),
        "real_minus_mask_only_ARI": _metric(row, "real_minus_mask_only_ARI"),
        "gate_pass": parse_bool(row.get("gate_pass")) if "gate_pass" in row else None,
        "uses_gt_for_prediction": row.get("uses_gt_for_prediction", False),
        "uses_gt_for_diagnostic_labels": row.get("uses_gt_for_diagnostic_labels", True),
        "note": note,
    }
    return out


def _best_by_ari(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("ARI") not in (None, "")]
    return max(valid, key=lambda row: parse_float(row.get("ARI"))) if valid else {}


def _fill_missing(row: dict[str, Any], donor: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out = dict(row)
    for key in keys:
        if out.get(key) in (None, "") and donor.get(key) not in (None, ""):
            out[key] = donor.get(key)
    return out


def _threshold(metric: str, value: Any, op: str, threshold: float | int) -> dict[str, Any]:
    if value in (None, ""):
        return {"metric": metric, "value": None, "op": op, "threshold": threshold, "pass": False, "reason": "unavailable"}
    value_f = parse_float(value)
    if op == ">=":
        passed = value_f >= float(threshold)
    elif op == "<=":
        passed = value_f <= float(threshold)
    elif op == "==":
        passed = value_f == float(threshold)
    else:
        raise ValueError(op)
    return {
        "metric": metric,
        "value": value_f,
        "op": op,
        "threshold": threshold,
        "pass": bool(passed),
        "reason": "ok" if passed else "threshold_not_met",
    }


def _prefixed(rows: list[dict[str, Any]], source: str, limit: int | None = None) -> list[dict[str, Any]]:
    selected = rows if limit is None else rows[:limit]
    return [dict(row, source=source) for row in selected]


def _case_rows(rows: list[dict[str, Any]], *, same_key: str, wanted: bool, limit: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if parse_bool(row.get(same_key)) == wanted:
            out.append(row)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v48 final Stage-1 decision and failure autopsy.")
    parser.add_argument("--output-root", default="outputs/audit/v48_final_decision")
    args = parser.parse_args()

    fact = _load_json(ROOT / "outputs/audit/v48_fact_lock/fact_lock.json")
    primitive = _load_json(ROOT / "outputs/audit/v48_primitive_audit/primitive_audit_summary.json")
    semantic = _load_json(ROOT / "outputs/audit/v48_semantic_features/semantic_feature_audit_summary.json")
    scheme_a = _load_json(ROOT / "outputs/audit/v48_object_explanation_ilp/object_explanation_summary.json")
    scheme_b = _load_json(ROOT / "outputs/audit/v48_true_flow/true_flow_summary.json")
    scheme_c = _load_json(ROOT / "outputs/audit/v48_latent_em/latent_em_summary.json")
    scheme_d = _load_json(ROOT / "outputs/audit/v48_common_fate/common_fate_summary.json")
    scheme_fg = _load_json(ROOT / "outputs/audit/v48_mask_set_cover_atoms/mask_set_cover_atoms_summary.json")
    d4rt_repair = _load_json(ROOT / "outputs/audit/v48_d4rt_control_repair/d4rt_control_repair_summary.json")
    contrast_repair = _load_json(ROOT / "outputs/audit/v48_control_contrast_component_merge/control_contrast_component_merge_summary.json")
    fragmentation = _load_json(ROOT / "outputs/audit/v48_fragmentation_autopsy/fragmentation_autopsy_summary.json")
    component_seeded_flow = _load_json(ROOT / "outputs/audit/v48_component_seeded_true_flow_repair/component_seeded_true_flow_repair_summary.json")
    d4rt_gated_bbox_flow = _load_json(ROOT / "outputs/audit/v48_d4rt_gated_bbox_flow_repair/component_seeded_true_flow_repair_summary.json")
    exact_subset = _load_json(ROOT / "outputs/audit/v48_exact_component_subset_diagnostic/exact_component_subset_summary.json")
    exact_subset_expanded = _load_json(ROOT / "outputs/audit/v48_exact_component_subset_diagnostic_expanded/exact_component_subset_summary.json")
    latent_em_d4rt_repair = _load_json(ROOT / "outputs/audit/v48_latent_em_d4rt_specific_repair/latent_em_d4rt_specific_repair_summary.json")
    common_fate_separation_repair = _load_json(ROOT / "outputs/audit/v48_common_fate_separation_repair/common_fate_separation_repair_summary.json")
    v47_stage1 = _load_json(ROOT / "outputs/audit/v47_stage1_final_gate_continued21_carrier_mdl_audit/stage1_final_gate_summary.json")
    v47_method_row = dict(v47_stage1.get("method_row") or {})

    facts = _fact_map(fact)
    primitive_row = dict(primitive.get("primary_primitive") or {})
    a_rows = list(scheme_a.get("rows") or [])
    a_best = max(
        [row for row in a_rows if str(row.get("status")) not in {"control"}],
        key=lambda row: parse_float(row.get("ARI")),
    ) if a_rows else {}
    a_best = _fill_missing(
        a_best,
        v47_method_row,
        [
            "mean_predictions_per_scene",
            "scene0011_purity",
            "scene0050_purity",
            "duplicate_rate",
            "conflict_rate",
            "unknown_tube_ratio",
            "birth_from_d4rt_tube_count",
            "maskless_object_count",
        ],
    )
    b_best = dict(scheme_b.get("best_real_row") or {})
    b_best.update(
        {
            "real_minus_shuffled_ARI": scheme_b.get("real_minus_shuffled_ARI"),
            "real_minus_no_temporal_ARI": scheme_b.get("real_minus_no_temporal_ARI"),
            "real_minus_mask_only_ARI": scheme_b.get("real_minus_mask_only_ARI"),
        }
    )
    c_rows = list(scheme_c.get("summary_rows") or [])
    c_best = max(c_rows, key=lambda row: parse_float(row.get("final_ARI"))) if c_rows else {}
    d_rows = list(scheme_d.get("summary_rows") or [])
    d_best = max(d_rows, key=lambda row: parse_float(row.get("ARI"))) if d_rows else {}
    fg_rows = list(scheme_fg.get("rows") or [])
    fg_best = max(fg_rows, key=lambda row: parse_float(row.get("ARI"))) if fg_rows else {}
    d4rt_repair_best = dict(d4rt_repair.get("best_row") or {})
    contrast_best = dict(contrast_repair.get("best_real_row") or {})
    component_seeded_flow_best = dict(component_seeded_flow.get("best_real_row") or {})
    component_seeded_flow_best.update(
        {
            "real_minus_shuffled_ARI": component_seeded_flow.get("real_minus_shuffled_ARI"),
            "real_minus_no_temporal_ARI": component_seeded_flow.get("real_minus_no_temporal_ARI"),
            "real_minus_mask_only_ARI": component_seeded_flow.get("real_minus_mask_only_ARI"),
        }
    )
    d4rt_gated_bbox_flow_best = dict(d4rt_gated_bbox_flow.get("best_real_row") or {})
    d4rt_gated_bbox_flow_best.update(
        {
            "real_minus_shuffled_ARI": d4rt_gated_bbox_flow.get("real_minus_shuffled_ARI"),
            "real_minus_no_temporal_ARI": d4rt_gated_bbox_flow.get("real_minus_no_temporal_ARI"),
            "real_minus_mask_only_ARI": d4rt_gated_bbox_flow.get("real_minus_mask_only_ARI"),
        }
    )
    exact_candidates = []
    for payload, source in [
        (exact_subset, "outputs/audit/v48_exact_component_subset_diagnostic/exact_component_subset_summary.json"),
        (exact_subset_expanded, "outputs/audit/v48_exact_component_subset_diagnostic_expanded/exact_component_subset_summary.json"),
    ]:
        if payload:
            row = dict(payload.get("best_real_row") or {})
            row.update(
                {
                    "real_minus_shuffled_ARI": payload.get("real_minus_shuffled_ARI"),
                    "real_minus_no_temporal_ARI": payload.get("real_minus_no_temporal_ARI"),
                    "real_minus_mask_only_ARI": payload.get("real_minus_mask_only_ARI"),
                    "_source": source,
                    "_gate_pass": parse_bool(payload.get("gate", {}).get("pass")),
                    "gate_pass": parse_bool(payload.get("gate", {}).get("pass")),
                }
            )
            exact_candidates.append(row)
    exact_subset_best = max(exact_candidates, key=lambda row: parse_float(row.get("ARI"))) if exact_candidates else {}
    latent_em_d4rt_best = dict(latent_em_d4rt_repair.get("best_real_row") or {})
    latent_em_d4rt_best.update(
        {
            "real_minus_shuffled_ARI": latent_em_d4rt_repair.get("real_minus_shuffled_ARI"),
            "real_minus_no_temporal_ARI": latent_em_d4rt_repair.get("real_minus_no_temporal_ARI"),
            "gate_pass": parse_bool(latent_em_d4rt_repair.get("gate", {}).get("pass")),
        }
    )
    common_fate_separation_best = dict(common_fate_separation_repair.get("best_real_row") or {})
    common_fate_separation_best.update(
        {
            "real_minus_shuffled_ARI": common_fate_separation_repair.get("real_minus_shuffled_ARI"),
            "real_minus_no_temporal_ARI": common_fate_separation_repair.get("real_minus_no_temporal_ARI"),
            "gate_pass": parse_bool(common_fate_separation_repair.get("gate", {}).get("pass")),
        }
    )

    stage_rows: list[dict[str, Any]] = [
        _stage_row(
            variant="Z0_v37_baseline",
            scheme="historical_baseline",
            source="outputs/audit/v48_fact_lock/fact_lock.json",
            row={
                "ARI": facts.get("v37_4D_ARI"),
                "purity": facts.get("v37_purity"),
                "completeness": facts.get("v37_completeness"),
                "temporal_span_mean": facts.get("v37_temporal_span"),
            },
            status="imported_prior",
        ),
        _stage_row(
            variant="Z1_v44_best",
            scheme="historical_baseline",
            source="outputs/audit/v48_fact_lock/fact_lock.json",
            row={"ARI": facts.get("v44_best_ARI")},
            status="imported_prior_partial_metrics",
            note="Only v44 best ARI is required by Phase 0 fact lock.",
        ),
        _stage_row(
            variant="Z2_v46_final",
            scheme="historical_baseline",
            source="outputs/audit/v46_final_decision/v46_final_decision.json",
            row={},
            status=str(facts.get("v46_final_label") or "imported_prior_label_only"),
        ),
        _stage_row(
            variant=str(facts.get("v47_selected_variant") or "Z3_v47_selected_partial"),
            scheme="historical_baseline",
            source="outputs/audit/v47_stage1_final_gate_continued21_carrier_mdl_audit/stage1_final_gate_summary.json",
            row=_fill_missing(
                {
                "ARI": facts.get("v47_selected_ARI"),
                "purity": facts.get("v47_selected_purity"),
                "completeness": facts.get("v47_selected_completeness"),
                },
                v47_method_row,
                [
                    "temporal_span_mean",
                    "scene0081_ARI",
                    "scene0011_purity",
                    "scene0050_purity",
                    "scene0591_completeness",
                    "mean_predictions_per_scene",
                    "birth_from_d4rt_tube_count",
                    "maskless_object_count",
                    "real_minus_shuffled_ARI",
                    "real_minus_no_temporal_ARI",
                ],
            ),
            status="imported_prior_partial",
        ),
        _stage_row(
            variant="Z4_best_primitive_P3_carrier_components",
            scheme="primitive",
            source="outputs/audit/v48_primitive_audit/primitive_audit_summary.json",
            row={
                "ARI": primitive_row.get("ARI"),
                "purity": primitive_row.get("primitive_purity_mean"),
                "completeness": primitive_row.get("primitive_completeness_mean"),
                "real_minus_shuffled_ARI": primitive_row.get("real_minus_shuffled_ARI"),
                "selected_candidate_count": primitive_row.get("primitive_count"),
            },
            status="primary_primitive",
        ),
        _stage_row(
            variant=f"Z5_scheme_A:{a_best.get('variant', '')}",
            scheme="A_object_explanation",
            source="outputs/audit/v48_object_explanation_ilp/object_explanation_summary.json",
            row=a_best,
            status="failed_gate" if not parse_bool(scheme_a.get("gate", {}).get("pass")) else "passed_gate",
        ),
        _stage_row(
            variant=f"Z6_scheme_B:{b_best.get('variant', '')}:{b_best.get('score_key', '')}",
            scheme="B_true_min_cost_flow",
            source="outputs/audit/v48_true_flow/true_flow_summary.json",
            row=b_best,
            status="failed_gate" if not parse_bool(scheme_b.get("gate", {}).get("pass")) else "passed_gate",
        ),
        _stage_row(
            variant=f"Z7_scheme_C:{c_best.get('variant', '')}",
            scheme="C_latent_EM",
            source="outputs/audit/v48_latent_em/latent_em_summary.json",
            row=c_best,
            status="failed_gate" if not parse_bool(scheme_c.get("gate", {}).get("pass")) else "passed_gate",
        ),
        _stage_row(
            variant=f"Z8_scheme_D:{d_best.get('variant', '')}",
            scheme="D_common_fate",
            source="outputs/audit/v48_common_fate/common_fate_summary.json",
            row=d_best,
            status="failed_gate" if not parse_bool(scheme_d.get("gate", {}).get("pass")) else "passed_gate",
        ),
        _stage_row(
            variant="Z9_scheme_E_semantic_guard",
            scheme="E_frozen_semantic_guard",
            source="outputs/audit/v48_semantic_features/semantic_feature_audit_summary.json",
            row={},
            status="blocked_no_positive_semantic_guard",
            note="RADIO is usable as contradiction guard only; no positive merge guard passed Phase 2.",
        ),
        _stage_row(
            variant=f"Z10_scheme_FG:{fg_best.get('variant', '')}",
            scheme="F_G_mask_set_cover_atoms",
            source="outputs/audit/v48_mask_set_cover_atoms/mask_set_cover_atoms_summary.json",
            row=fg_best,
            status="failed_gate" if not parse_bool(scheme_fg.get("gate", {}).get("pass")) else "passed_gate",
        ),
        _stage_row(
            variant=f"Z16_d4rt_control_repair:{d4rt_repair_best.get('variant', '')}",
            scheme="continuation_D4RT_control_repair",
            source="outputs/audit/v48_d4rt_control_repair/d4rt_control_repair_summary.json",
            row=d4rt_repair_best,
            status="failed_gate" if not parse_bool(d4rt_repair.get("gate", {}).get("pass")) else "passed_gate",
            note="Continuation after first No-Go: paired same-parameter D4RT-required repair.",
        ),
        _stage_row(
            variant=f"Z17_control_contrast_repair:{contrast_best.get('score_key', '')}",
            scheme="continuation_control_contrast_repair",
            source="outputs/audit/v48_control_contrast_component_merge/control_contrast_component_merge_summary.json",
            row=contrast_best,
            status="failed_gate" if not parse_bool(contrast_repair.get("gate", {}).get("pass")) else "passed_gate",
            note="Continuation after controls failed: select A5/A4 contrast against no-temporal and shuffled controls.",
        ),
        _stage_row(
            variant=f"Z18_component_seeded_true_flow_repair:{component_seeded_flow_best.get('score_key', '')}",
            scheme="continuation_component_seeded_true_flow",
            source="outputs/audit/v48_component_seeded_true_flow_repair/component_seeded_true_flow_repair_summary.json",
            row=component_seeded_flow_best,
            status="failed_gate" if not parse_bool(component_seeded_flow.get("gate", {}).get("pass")) else "passed_gate",
            note="Continuation after Scheme B under-merge: seed true min-cost flow with carrier components, then stitch component roots.",
        ),
        _stage_row(
            variant=f"Z19_d4rt_gated_bbox_flow_repair:{d4rt_gated_bbox_flow_best.get('score_key', '')}",
            scheme="continuation_d4rt_gated_bbox_flow",
            source="outputs/audit/v48_d4rt_gated_bbox_flow_repair/component_seeded_true_flow_repair_summary.json",
            row=d4rt_gated_bbox_flow_best,
            status="failed_gate" if not parse_bool(d4rt_gated_bbox_flow.get("gate", {}).get("pass")) else "passed_gate",
            note="Continuation after mask-only component-flow outperformed D4RT flow: require bbox temporal flow to be D4RT-confirmed.",
        ),
        _stage_row(
            variant=f"Z20_exact_component_subset_diagnostic:{exact_subset_best.get('score_key', '')}",
            scheme="continuation_exact_component_subset_diagnostic",
            source=str(exact_subset_best.get("_source") or "outputs/audit/v48_exact_component_subset_diagnostic/exact_component_subset_summary.json"),
            row=exact_subset_best,
            status="failed_gate" if not parse_bool(exact_subset_best.get("_gate_pass")) else "passed_gate",
            note="Continuation for Scheme A A6: bounded exact branch-and-bound over top component-pair subsets; diagnostic, not full large-scale ILP.",
        ),
        _stage_row(
            variant=f"Z21_latent_em_d4rt_specific_repair:{latent_em_d4rt_best.get('score_key', '')}",
            scheme="continuation_latent_em_d4rt_specific",
            source="outputs/audit/v48_latent_em_d4rt_specific_repair/latent_em_d4rt_specific_repair_summary.json",
            row=latent_em_d4rt_best,
            status="failed_gate" if not parse_bool(latent_em_d4rt_repair.get("gate", {}).get("pass")) else "passed_gate",
            note="Continuation for Scheme C: high-temporal-support EM threshold repair plus D4RT-specific margins against no-temporal/shuffled controls.",
        ),
        _stage_row(
            variant=f"Z22_common_fate_separation_repair:{common_fate_separation_best.get('score_key', '')}",
            scheme="continuation_common_fate_separation",
            source="outputs/audit/v48_common_fate_separation_repair/common_fate_separation_repair_summary.json",
            row=common_fate_separation_best,
            status="failed_gate" if not parse_bool(common_fate_separation_repair.get("gate", {}).get("pass")) else "passed_gate",
            note="Continuation for Scheme D: same-frame separation and relative-layout repair for static common-fate ambiguity.",
        ),
    ]
    method_candidates = [
        row
        for row in stage_rows
        if row["scheme"] in {
            "A_object_explanation",
            "B_true_min_cost_flow",
            "C_latent_EM",
            "D_common_fate",
            "F_G_mask_set_cover_atoms",
            "continuation_D4RT_control_repair",
            "continuation_control_contrast_repair",
            "continuation_component_seeded_true_flow",
            "continuation_d4rt_gated_bbox_flow",
            "continuation_exact_component_subset_diagnostic",
            "continuation_latent_em_d4rt_specific",
            "continuation_common_fate_separation",
            "primitive",
        }
        and row.get("ARI") not in (None, "")
    ]
    final_candidate = _best_by_ari(method_candidates)
    final_row = dict(final_candidate)
    final_row.update({"variant": f"Z11_final_candidate:{final_candidate.get('variant', 'none')}", "scheme": "final_candidate"})
    stage_rows.append(final_row)
    if scheme_a.get("rows"):
        for control_name in ["A7_shuffled_D4RT_control", "A8_no_temporal_control"]:
            control = next((row for row in scheme_a["rows"] if row.get("variant") == control_name), None)
            if control:
                stage_rows.append(
                    _stage_row(
                        variant=f"Z12_Z13_control:{control_name}",
                        scheme="control",
                        source="outputs/audit/v48_object_explanation_ilp/object_explanation_summary.json",
                        row=control,
                        status="control",
                    )
                )
    stage_rows.append(
        _stage_row(
            variant="Z14_mask_only_control:true_flow_mask_only_best",
            scheme="control",
            source="outputs/audit/v48_true_flow/true_flow_summary.json",
            row=scheme_b.get("best_mask_only_row") or {},
            status="control",
        )
    )
    stage_rows.append(
        _stage_row(
            variant="Z15_semantic_only_control",
            scheme="control",
            source="outputs/audit/v48_semantic_features/semantic_feature_audit_summary.json",
            row={},
            status="not_available",
            note="No promoted semantic-only component object row; Phase 2 positive semantic guard failed.",
        )
    )

    v37_ari = parse_float(facts.get("v37_4D_ARI"))
    v37_comp = parse_float(facts.get("v37_completeness"))
    final_metrics = {
        "4D_ARI": final_candidate.get("ARI"),
        "4D_purity": final_candidate.get("purity"),
        "4D_completeness": final_candidate.get("completeness"),
        "temporal_span_mean": final_candidate.get("temporal_span_mean"),
        "scene0081_ARI": final_candidate.get("scene0081_ARI"),
        "scene0011_purity": final_candidate.get("scene0011_purity"),
        "scene0050_purity": final_candidate.get("scene0050_purity"),
        "scene0591_completeness": final_candidate.get("scene0591_completeness"),
        "mean_predictions_per_scene": final_candidate.get("mean_predictions_per_scene"),
        "duplicate_rate": final_candidate.get("duplicate_rate"),
        "conflict_rate": final_candidate.get("conflict_rate"),
        "unknown_tube_ratio": final_candidate.get("unknown_tube_ratio"),
        "birth_from_d4rt_tube_count": final_candidate.get("birth_from_d4rt_tube_count"),
        "maskless_object_count": final_candidate.get("maskless_object_count"),
        "real_minus_shuffled_ARI": final_candidate.get("real_minus_shuffled_ARI"),
        "real_minus_no_temporal_ARI": final_candidate.get("real_minus_no_temporal_ARI"),
        "real_minus_mask_only_ARI": final_candidate.get("real_minus_mask_only_ARI"),
        "bootstrap_delta_ARI_lower95": None,
        "bootstrap_delta_completeness_lower95": None,
    }
    gate_rows = [
        _threshold("4D_ARI", final_metrics["4D_ARI"], ">=", 0.485),
        _threshold("4D_purity", final_metrics["4D_purity"], ">=", 0.875),
        _threshold("4D_completeness", final_metrics["4D_completeness"], ">=", 0.555),
        _threshold("temporal_span_mean", final_metrics["temporal_span_mean"], ">=", 1.70),
        _threshold("scene0081_ARI", final_metrics["scene0081_ARI"], ">=", 0.270),
        _threshold("scene0011_purity", final_metrics["scene0011_purity"], ">=", 0.84),
        _threshold("scene0050_purity", final_metrics["scene0050_purity"], ">=", 0.84),
        _threshold("mean_predictions_per_scene", final_metrics["mean_predictions_per_scene"], "<=", 150),
        _threshold("duplicate_rate", final_metrics["duplicate_rate"], "<=", 0.05),
        _threshold("conflict_rate", final_metrics["conflict_rate"], "<=", 0.10),
        _threshold("unknown_tube_ratio", final_metrics["unknown_tube_ratio"], "<=", 0.35),
        _threshold("birth_from_d4rt_tube_count", final_metrics["birth_from_d4rt_tube_count"], "==", 0),
        _threshold("maskless_object_count", final_metrics["maskless_object_count"], "==", 0),
        _threshold("real_minus_shuffled_ARI", final_metrics["real_minus_shuffled_ARI"], ">=", 0.30),
        _threshold("real_minus_no_temporal_ARI", final_metrics["real_minus_no_temporal_ARI"], ">=", 0.25),
        _threshold("real_minus_mask_only_ARI", final_metrics["real_minus_mask_only_ARI"], ">=", 0.25),
        _threshold("bootstrap_delta_ARI_lower95", final_metrics["bootstrap_delta_ARI_lower95"], ">=", 0.025),
        _threshold("bootstrap_delta_completeness_lower95", final_metrics["bootstrap_delta_completeness_lower95"], ">=", 0.020),
    ]
    stage1_gate = {
        "pass": all(row["pass"] for row in gate_rows),
        "passed_metric_count": sum(1 for row in gate_rows if row["pass"]),
        "failed_metric_count": sum(1 for row in gate_rows if not row["pass"]),
        "threshold_failed_metric_names": [row["metric"] for row in gate_rows if row["reason"] == "threshold_not_met"],
        "unavailable_metric_names": [row["metric"] for row in gate_rows if row["reason"] == "unavailable"],
        "rows": gate_rows,
    }

    labels = []
    if not parse_bool(semantic.get("gate", {}).get("pass")):
        labels.append("NO_GO_SEMANTIC_GUARD")
    if not parse_bool(scheme_a.get("gate", {}).get("pass")):
        labels.append("NO_GO_SET_COVER_ILP")
    if not parse_bool(scheme_b.get("gate", {}).get("pass")):
        labels.append("NO_GO_TRUE_FLOW")
    if not parse_bool(scheme_c.get("gate", {}).get("pass")):
        labels.append("NO_GO_EM_EXPLANATION")
    if not parse_bool(scheme_d.get("gate", {}).get("pass")):
        labels.append("NO_GO_COMMON_FATE")
    if not parse_bool(scheme_fg.get("gate", {}).get("pass")):
        labels.append("NO_GO_MASK_ATOM")
    if d4rt_repair and not parse_bool(d4rt_repair.get("gate", {}).get("pass")):
        labels.append("NO_GO_D4RT_CONTROL_REPAIR")
    if contrast_repair and not parse_bool(contrast_repair.get("gate", {}).get("pass")):
        labels.append("NO_GO_CONTROL_CONTRAST_REPAIR")
    if component_seeded_flow and not parse_bool(component_seeded_flow.get("gate", {}).get("pass")):
        labels.append("NO_GO_COMPONENT_SEEDED_TRUE_FLOW_REPAIR")
    if d4rt_gated_bbox_flow and not parse_bool(d4rt_gated_bbox_flow.get("gate", {}).get("pass")):
        labels.append("NO_GO_D4RT_GATED_BBOX_FLOW_REPAIR")
    if (exact_subset or exact_subset_expanded) and not parse_bool(exact_subset_best.get("_gate_pass")):
        labels.append("NO_GO_EXACT_COMPONENT_SUBSET_DIAGNOSTIC")
    if latent_em_d4rt_repair and not parse_bool(latent_em_d4rt_repair.get("gate", {}).get("pass")):
        labels.append("NO_GO_LATENT_EM_D4RT_SPECIFIC_REPAIR")
    if common_fate_separation_repair and not parse_bool(common_fate_separation_repair.get("gate", {}).get("pass")):
        labels.append("NO_GO_COMMON_FATE_SEPARATION_REPAIR")
    if not stage1_gate["pass"]:
        labels.append("NO_GO_STAGE1_NOT_SIGNIFICANT")
    final_label = "STAGE1_SIGNIFICANT_PASS" if stage1_gate["pass"] else "NO_GO_STAGE1_NOT_SIGNIFICANT"

    out = ROOT / str(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "v48_stage1_variant_rows.csv", stage_rows)
    write_csv(out / "v48_stage1_gate_rows.csv", gate_rows)
    write_csv(out / "component_merge_error_rows.csv", a_rows)
    write_csv(out / "flow_error_rows.csv", _load_csv(ROOT / "outputs/audit/v48_true_flow/true_flow_scan_rows.csv"))
    write_csv(out / "semantic_feature_error_rows.csv", semantic.get("backend_rows", []))
    write_csv(out / "common_fate_error_rows.csv", d_rows)
    write_csv(out / "set_cover_error_rows.csv", fg_rows)
    write_csv(
        out / "per_scene_metric_table.csv",
        [
            {
                "variant": row.get("variant"),
                "scheme": row.get("scheme"),
                "scene0081_ARI": row.get("scene0081_ARI"),
                "scene0011_purity": row.get("scene0011_purity"),
                "scene0050_purity": row.get("scene0050_purity"),
                "scene0591_completeness": row.get("scene0591_completeness"),
                "source": row.get("source"),
            }
            for row in stage_rows
        ],
    )

    true_flow_edges = _load_csv(ROOT / "outputs/audit/v48_true_flow/true_flow_best_real_selected_edges.csv")
    common_fate_pairs = _load_csv(ROOT / "outputs/audit/v48_common_fate/common_fate_selected_pairs.csv")
    false_merge_casebook = {
        "status": "aggregate_casebook_from_selected_edges",
        "notes": [
            "Examples are diagnostic-only selected pairs with diagnostic_same_gt=False.",
            "No GT labels are used for prediction; GT fields are used only to classify failure cases.",
        ],
        "true_flow_false_merge_examples": _case_rows(true_flow_edges, same_key="diagnostic_same_gt", wanted=False, limit=10),
        "common_fate_false_merge_examples": _case_rows(common_fate_pairs, same_key="diagnostic_same_gt", wanted=False, limit=10),
    }
    false_cut_casebook = {
        "status": "aggregate_casebook_from_high_score_nonpassing_routes",
        "notes": [
            "Full object-field false-cut export is unavailable in v48 artifacts.",
            "Examples below are same-GT candidates that did not rescue the gate because route-level metrics/controls failed.",
        ],
        "common_fate_same_gt_examples": _case_rows(
            _load_csv(ROOT / "outputs/audit/v48_common_fate/common_fate_pair_rows.csv"),
            same_key="diagnostic_same_gt",
            wanted=True,
            limit=10,
        ),
        "true_flow_selected_same_gt_examples": _case_rows(true_flow_edges, same_key="diagnostic_same_gt", wanted=True, limit=10),
    }
    write_json(out / "false_merge_casebook.json", false_merge_casebook)
    write_json(out / "false_cut_casebook.json", false_cut_casebook)

    failure_summary_lines = [
        "# Stream4D v48 Failure Summary",
        "",
        f"final_label: {final_label}",
        f"final_candidate: {final_candidate.get('variant')}",
        f"final_4D_ARI: {final_candidate.get('ARI')}",
        f"final_4D_purity: {final_candidate.get('purity')}",
        f"final_4D_completeness: {final_candidate.get('completeness')}",
        f"delta_ARI_vs_v37_point: {parse_float(final_candidate.get('ARI')) - v37_ari if final_candidate else None}",
        f"delta_completeness_vs_v37_point: {parse_float(final_candidate.get('completeness')) - v37_comp if final_candidate else None}",
        "",
        "## Labels",
        "",
        *[f"- {label}" for label in labels],
        "",
        "## Evidence Chain",
        "",
        "- Primitive: P3 carrier components pass Phase 1, but are only a partial signal.",
        "- Semantic: DINO/RADIO are available, but neither beats colorhist positive merge guard; RADIO is contradiction-only.",
        "- Scheme A: best A5 improves ARI by +0.0123 and completeness by +0.0175 vs raw, below gate and controls fail.",
        "- Scheme B: true NetworkX min-cost circulation improves proxy by only +0.00144 ARI and fails no-temporal/mask-only controls.",
        "- Scheme C: EM proxy reaches ARI 0.42948, +0.00478 over nuclei, below gate.",
        "- Scheme D: conflict-soft repair reaches ARI 0.42746, top100 precision 0.20 and AUC 0.4468, below gate.",
        "- Scheme F/G: budgeted set cover covers 0.3947 of components; full coverage needs 380 candidates; atom diagnostic does not improve component completion.",
        "- Continuation D4RT paired-control repair: best ARI 0.43060, real-minus-no-temporal same-params +0.00109, real-minus-shuffled +0.05225.",
        "- Continuation control-contrast repair: best real ARI 0.42491; no-temporal/shuffled contrast controls are stronger than real contrast.",
        "- Fragmentation autopsy: A5 reduces oracle missing merges from 472 to 190 but doubles conflict roots from 20 to 40 under scene-qualified diagnostic labels.",
        "- Continuation component-seeded true-flow repair: best real ARI 0.43238, completeness 0.43170; mask-only component flow reaches ARI 0.45339 and beats D4RT flow.",
        "- Continuation D4RT-gated bbox flow repair: best real ARI 0.42939, completeness 0.42295; no-temporal-confirmed bbox control reaches ARI 0.45387.",
        "- Continuation exact component-subset diagnostic: expanded top80/top20 exact subset reaches ARI 0.42921; no-temporal exact control reaches ARI 0.42932.",
        f"- Continuation latent-EM D4RT-specific repair: best real ARI {latent_em_d4rt_best.get('final_ARI')}, completeness {latent_em_d4rt_best.get('final_completeness')}; no-temporal EM control ARI {latent_em_d4rt_repair.get('best_no_temporal_row', {}).get('final_ARI') if latent_em_d4rt_repair else None}.",
        f"- Continuation common-fate separation repair: best real ARI {common_fate_separation_best.get('ARI')}, completeness {common_fate_separation_best.get('completeness')}; real motion AUC {common_fate_separation_repair.get('auc_by_score', {}).get('separation_real_score') if common_fate_separation_repair else None}, no-temporal AUC {common_fate_separation_repair.get('auc_by_score', {}).get('separation_no_temporal_score') if common_fate_separation_repair else None}.",
        "",
        "## Stage-1 Blockers",
        "",
        f"- threshold_failed_metric_names: {stage1_gate['threshold_failed_metric_names']}",
        f"- unavailable_metric_names: {stage1_gate['unavailable_metric_names']}",
        "- AP and Stage-2 method rows are blocked because Stage-1 significant gate failed.",
    ]
    (out / "failure_summary.md").write_text("\n".join(failure_summary_lines) + "\n", encoding="utf-8")

    payload = {
        "phase": "v48_final_decision",
        "created_at": utc_now(),
        "final_label": final_label,
        "supporting_failure_labels": labels,
        "final_candidate": final_candidate,
        "stage1_gate": stage1_gate,
        "answers": {
            "strongest_primitive": primitive.get("gate", {}).get("primary_primitive_type"),
            "semantic_guard_positive_pass": bool(semantic.get("gate", {}).get("pass")),
            "semantic_contradiction_guard_backend": semantic.get("gate", {}).get("recommended_contradiction_backend"),
            "scheme_A_pass": bool(scheme_a.get("gate", {}).get("pass")),
            "scheme_B_true_flow_pass": bool(scheme_b.get("gate", {}).get("pass")),
            "scheme_C_EM_pass": bool(scheme_c.get("gate", {}).get("pass")),
            "scheme_D_common_fate_pass": bool(scheme_d.get("gate", {}).get("pass")),
            "scheme_FG_mask_atom_pass": bool(scheme_fg.get("gate", {}).get("pass")),
            "d4rt_control_repair_pass": bool(d4rt_repair.get("gate", {}).get("pass")) if d4rt_repair else None,
            "control_contrast_repair_pass": bool(contrast_repair.get("gate", {}).get("pass")) if contrast_repair else None,
            "component_seeded_true_flow_repair_pass": bool(component_seeded_flow.get("gate", {}).get("pass")) if component_seeded_flow else None,
            "d4rt_gated_bbox_flow_repair_pass": bool(d4rt_gated_bbox_flow.get("gate", {}).get("pass")) if d4rt_gated_bbox_flow else None,
            "exact_component_subset_diagnostic_pass": bool(exact_subset_best.get("_gate_pass")) if exact_subset_best else None,
            "latent_em_d4rt_specific_repair_pass": bool(latent_em_d4rt_repair.get("gate", {}).get("pass")) if latent_em_d4rt_repair else None,
            "common_fate_separation_repair_pass": bool(common_fate_separation_repair.get("gate", {}).get("pass")) if common_fate_separation_repair else None,
            "full_stage1_significant": bool(stage1_gate["pass"]),
            "AP_method_rows_allowed": bool(stage1_gate["pass"]),
            "stage2_allowed": bool(stage1_gate["pass"]),
            "no_gt_for_prediction": True,
        },
        "point_delta_vs_v37": {
            "ARI": parse_float(final_candidate.get("ARI")) - v37_ari if final_candidate else None,
            "completeness": parse_float(final_candidate.get("completeness")) - v37_comp if final_candidate else None,
            "note": "Point deltas are not bootstrap lower bounds and do not satisfy Stage-1 significant gate.",
        },
        "artifact_paths": {
            "variant_rows": str(out / "v48_stage1_variant_rows.csv"),
            "gate_rows": str(out / "v48_stage1_gate_rows.csv"),
            "failure_summary": str(out / "failure_summary.md"),
            "false_merge_casebook": str(out / "false_merge_casebook.json"),
            "false_cut_casebook": str(out / "false_cut_casebook.json"),
            "fragmentation_autopsy": str(ROOT / "outputs/audit/v48_fragmentation_autopsy/fragmentation_autopsy_summary.json") if fragmentation else "",
            "component_seeded_true_flow_repair": str(ROOT / "outputs/audit/v48_component_seeded_true_flow_repair/component_seeded_true_flow_repair_summary.json") if component_seeded_flow else "",
            "d4rt_gated_bbox_flow_repair": str(ROOT / "outputs/audit/v48_d4rt_gated_bbox_flow_repair/component_seeded_true_flow_repair_summary.json") if d4rt_gated_bbox_flow else "",
            "exact_component_subset_diagnostic": str(ROOT / "outputs/audit/v48_exact_component_subset_diagnostic/exact_component_subset_summary.json") if exact_subset else "",
            "exact_component_subset_diagnostic_expanded": str(ROOT / "outputs/audit/v48_exact_component_subset_diagnostic_expanded/exact_component_subset_summary.json") if exact_subset_expanded else "",
            "latent_em_d4rt_specific_repair": str(ROOT / "outputs/audit/v48_latent_em_d4rt_specific_repair/latent_em_d4rt_specific_repair_summary.json") if latent_em_d4rt_repair else "",
            "common_fate_separation_repair": str(ROOT / "outputs/audit/v48_common_fate_separation_repair/common_fate_separation_repair_summary.json") if common_fate_separation_repair else "",
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    write_json(out / "v48_final_decision.json", payload)
    print({"summary": str(out / "v48_final_decision.json"), "final_label": final_label, "stage1_gate_pass": stage1_gate["pass"]})


if __name__ == "__main__":
    main()
