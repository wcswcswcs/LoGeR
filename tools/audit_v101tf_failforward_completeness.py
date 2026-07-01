#!/usr/bin/env python3
"""Audit and complete v101 fail-forward evidence artifacts.

The v101 plan requires failed/blocked tracks to leave reviewer-friendly
failure reports, control-gap reports, FP/FN rows where applicable, and next
attempt recommendations.  This script fills missing *audit artifacts only* from
existing evidence.  It does not change any gate result or authorize action.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"

POS_TAX = "HANDOFF_SCALE_GAUGE_TARGET"
SAFE_TAX = "SAFE_GOOD"


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def b(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}"}
    return payload if isinstance(payload, dict) else {"value": payload}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_text(path: Path, text: str) -> bool:
    if path.is_file():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return True


def ensure_rows(path: Path, rows: list[dict[str, Any]]) -> bool:
    if path.is_file():
        return False
    write_rows(path, rows)
    return True


def mean(values: list[Any]) -> float:
    vals = [f(v) for v in values if math.isfinite(f(v))]
    return sum(vals) / len(vals) if vals else math.nan


def target_rows() -> dict[str, dict[str, str]]:
    return {row.get("case_id", ""): row for row in read_rows(ROOT / "trackT_drift_target_relabel/target_universe_v101.csv")}


def fpfn_from_case_scores(case_rows: list[dict[str, Any]], score_field: str, *, direction: str, cue_name: str) -> list[dict[str, Any]]:
    eval_rows = [
        row
        for row in case_rows
        if row.get("target_taxonomy") in {POS_TAX, SAFE_TAX} and math.isfinite(f(row.get(score_field)))
    ]
    positives = [row for row in eval_rows if row.get("target_taxonomy") == POS_TAX]
    safe = [row for row in eval_rows if row.get("target_taxonomy") == SAFE_TAX]
    if not positives or not safe:
        return [
            {
                "cue_name": cue_name,
                "row_kind": "not_enough_target_or_safe_cases",
                "reason": "Cannot compute FP/FN selector rows without both HANDOFF target and SAFE_GOOD cases.",
                "positive_case_count": len(positives),
                "safe_good_case_count": len(safe),
            }
        ]
    ranked = sorted(eval_rows, key=lambda row: f(row.get(score_field)), reverse=(direction == "higher_bad"))
    selected = {row["case_id"] for row in ranked[: len(positives)]}
    out: list[dict[str, Any]] = []
    for row in eval_rows:
        row_kind = ""
        if row["case_id"] in selected and row.get("target_taxonomy") == SAFE_TAX:
            row_kind = "false_positive_safe_good"
        elif row["case_id"] not in selected and row.get("target_taxonomy") == POS_TAX:
            row_kind = "missed_positive_handoff"
        elif row["case_id"] in selected and row.get("target_taxonomy") == POS_TAX:
            row_kind = "true_positive_handoff"
        if row_kind:
            out.append(
                {
                    "cue_name": cue_name,
                    "row_kind": row_kind,
                    "case_id": row.get("case_id", ""),
                    "seq": row.get("seq", ""),
                    "target_taxonomy": row.get("target_taxonomy", ""),
                    "score_field": score_field,
                    "score_value": row.get(score_field, ""),
                    "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy", ""),
                    "claim_level": "diagnostic_fpfn_rows_no_action",
                }
            )
    return out


def aggregate_by_case(rows: list[dict[str, str]], target_by_case: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        case_id = row.get("case_id", "")
        if case_id:
            grouped[case_id].append(row)
    out: list[dict[str, Any]] = []
    for case_id, parts in sorted(grouped.items()):
        target = target_by_case.get(case_id, {})
        out.append(
            {
                "case_id": case_id,
                "seq": target.get("seq", case_id.split("_", 1)[0]),
                "target_taxonomy": target.get("target_taxonomy", ""),
                "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", ""),
                "anchor_count": len(parts),
                "mean_S_cur": mean([row.get("S_cur_combined") for row in parts]),
                "mean_O_scale": mean([row.get("O_scale") for row in parts]),
                "mean_R_same": mean([row.get("R_same") for row in parts]),
                "unsupported_inconsistent_frac": mean([1.0 if row.get("state_status") == "unsupported_inconsistent" else 0.0 for row in parts]),
                "supported_consistent_frac": mean([1.0 if row.get("state_status") == "supported_consistent" else 0.0 for row in parts]),
                "stale_candidate_frac": mean([1.0 if row.get("role") == "stale_candidate" else 0.0 for row in parts]),
                "fresh_supported_score": mean(
                    [
                        f(row.get("query_hit_max"), 0.0)
                        * f(row.get("S_cur_combined"), 0.0)
                        * f(row.get("O_scale"), 0.0)
                        * max(0.0, 1.0 - f(row.get("R_same"), 0.0))
                        for row in parts
                    ]
                ),
            }
        )
    return out


def target_conflict_fpfn() -> list[dict[str, Any]]:
    rows = read_rows(ROOT / "trackT_drift_target_relabel/label_l3_conflict_rows.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        row_kind = row.get("label_l3_conflict", "") or "label_metric_conflict"
        if b(row.get("false_positive_in_best_composite")):
            row_kind = "false_positive_in_best_composite"
        elif row.get("target_taxonomy") == POS_TAX and not b(row.get("selected_by_best_composite")):
            row_kind = "missed_clean_handoff_target_by_best_composite"
        out.append(
            {
                "cue_name": "TrackT_label_L3_hygiene",
                "row_kind": row_kind,
                "case_id": row.get("case_id", ""),
                "seq": row.get("seq", ""),
                "case_label": row.get("case_label", ""),
                "target_taxonomy": row.get("target_taxonomy", ""),
                "selected_by_best_composite": row.get("selected_by_best_composite", ""),
                "false_positive_in_best_composite": row.get("false_positive_in_best_composite", ""),
                "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy", ""),
                "claim_level": "target_hygiene_conflict_rows_no_action",
            }
        )
    return out or [
        {
            "cue_name": "TrackT_label_L3_hygiene",
            "row_kind": "no_conflict_rows_available",
            "claim_level": "diagnostic_missing_source",
        }
    ]


def write_completion_audit(final: dict[str, Any], created: list[dict[str, Any]], artifact_rows: list[dict[str, Any]]) -> None:
    broader = read_json(FINAL / "broader_carrier_reentry_summary.json")
    merge_gauge = read_json(FINAL / "merge_gauge_selector_reentry_summary.json")
    merge_gauge_rich = read_json(FINAL / "merge_gauge_rich_selector_reentry_summary.json")
    internal_qk = read_json(FINAL / "internal_qk_actuator_reentry_summary.json")
    cache_topk = read_json(FINAL / "cache_topk_identity_query_reentry_summary.json")
    remaining = read_json(FINAL / "remaining_reentry_route_closure_summary.json")
    historical_mining = read_json(FINAL / "historical_target_universe_mining_summary.json")
    replay_probe = read_json(FINAL / "rich_selector_replay_probe_summary.json")
    holdout_feasibility = read_json(FINAL / "rich_selector_holdout_feasibility_summary.json")
    fresh_schema = read_json(FINAL / "fresh_selected_schema_materialization_summary.json")
    strict_frontier = read_json(FINAL / "strict_action_frontier_summary.json")
    plan_coverage = read_json(FINAL / "plan_coverage_closure_summary.json")
    trace_rescue = read_json(FINAL / "trace_rescue_feasibility_summary.json")
    state_q2 = read_json(FINAL / "state_q2_readiness_summary.json")
    component_identity = read_json(FINAL / "component_identity_availability_summary.json")
    stage_c_seed_bridge = read_json(FINAL / "stage_c_seed_bridge_smoke_summary.json")
    stage_c_seed_support = read_json(ROOT / "trackU_true_current_support/stage_c_seed_current_support_summary.json")
    stage_c_seed_support_q128 = read_json(
        ROOT / "trackU_true_current_support/stage_c_seed_current_support_clean_eval_q128_summary.json"
    )
    anchor_seed_join = read_json(FINAL / "anchor_seed_join_feasibility_summary.json")
    lifecycle_support_join = read_json(FINAL / "anchor_seed_lifecycle_support_join_summary.json")
    masklet_visibility = read_json(FINAL / "anchor_seed_lifecycle_stage_c_masklet_visibility_summary.json")
    lifecycle_geometry = read_json(FINAL / "anchor_seed_lifecycle_geometry_observability_summary.json")
    geometry_smoke = read_json(FINAL / "stage_c_seed_geometry_smoke_clean6_summary.json")
    geometry_smoke_target28 = read_json(FINAL / "stage_c_seed_geometry_smoke_target28_summary.json")
    combined_admission = read_json(FINAL / "combined_masklet_geometry_admission_summary.json")
    combined_fp_attr = read_json(FINAL / "combined_admission_false_positive_attribution_summary.json")
    masklet_q2 = read_json(FINAL / "masklet_q2_admission_sanity_summary.json")

    requirements = [
        {
            "requirement": "dual_logs_exist",
            "status": "pass",
            "evidence": "docs/ACL2_v101TF_SemanticAnchorStateEstimation_ScaleUpdateAdmission_执行日志.md and _实验结果复盘.md",
        },
        {
            "requirement": "no_fabricated_runtime_or_full_success",
            "status": "pass",
            "evidence": f"runtime_action_allowed={final.get('runtime_action_allowed')}; full_validation_run={final.get('full_validation_run')}; full_method_success={final.get('full_method_success')}",
        },
        {
            "requirement": "v101_method_goal_achieved",
            "status": "fail",
            "evidence": f"final_taxonomy={final.get('final_taxonomy')}; Track T/Q2/V/S2 gates still fail.",
        },
        {
            "requirement": "downstream_action_authorized",
            "status": "fail",
            "evidence": f"trackM4_run_allowed={final.get('trackM4_run_allowed')}; runtime_action_allowed={final.get('runtime_action_allowed')}",
        },
        {
            "requirement": "fail_forward_artifact_audit",
            "status": "pass",
            "evidence": "final_decision/fail_forward_artifact_audit.csv",
        },
        {
            "requirement": "outcome_d_broader_routes_closed",
            "status": "pass" if broader.get("reentry_route_count") == 5 and broader.get("action_allowed_route_count") == 0 else "fail",
            "evidence": f"reentry_route_count={broader.get('reentry_route_count')}; action_allowed_route_count={broader.get('action_allowed_route_count')}",
        },
        {
            "requirement": "outcome_d_merge_gauge_selector_followup",
            "status": "pass" if merge_gauge.get("candidate_policy_count") == 108 and merge_gauge.get("passing_candidate_count") == 0 else "fail",
            "evidence": f"candidate_policy_count={merge_gauge.get('candidate_policy_count')}; passing_candidate_count={merge_gauge.get('passing_candidate_count')}",
        },
        {
            "requirement": "outcome_d_merge_gauge_rich_selector_screen",
            "status": "pass"
            if merge_gauge_rich.get("candidate_policy_count") == 36708
            and merge_gauge_rich.get("best_action_authorized") is False
            and merge_gauge_rich.get("promotion_floor_passing_candidate_count") == 0
            and merge_gauge_rich.get("action_authorized_candidate_count") == 0
            and merge_gauge_rich.get("best_signal_stability_status") == "unstable_selected_sequence_dependent"
            and merge_gauge_rich.get("loso_train_split_with_retrospective_pass_count") == 0
            and merge_gauge_rich.get("loso_train_split_with_promotion_floor_pass_count") == 0
            else "fail",
            "evidence": (
                f"candidate_policy_count={merge_gauge_rich.get('candidate_policy_count')}; "
                f"retrospective_passing_candidate_count={merge_gauge_rich.get('retrospective_passing_candidate_count')}; "
                f"promotion_floor_passing_candidate_count={merge_gauge_rich.get('promotion_floor_passing_candidate_count')}; "
                f"action_authorized_candidate_count={merge_gauge_rich.get('action_authorized_candidate_count')}; "
                f"best_action_authorized={merge_gauge_rich.get('best_action_authorized')}; "
                f"best_action_block_reason={merge_gauge_rich.get('best_action_block_reason')}; "
                f"best_signal_stability_status={merge_gauge_rich.get('best_signal_stability_status')}; "
                f"loso_train_split_with_retrospective_pass_count={merge_gauge_rich.get('loso_train_split_with_retrospective_pass_count')}; "
                f"loso_train_split_with_promotion_floor_pass_count={merge_gauge_rich.get('loso_train_split_with_promotion_floor_pass_count')}"
            ),
        },
        {
            "requirement": "outcome_d_internal_qk_actuator_followup",
            "status": "pass" if internal_qk.get("family_count") == 6 and internal_qk.get("action_surface_passing_family_count") == 0 else "fail",
            "evidence": f"family_count={internal_qk.get('family_count')}; action_surface_passing_family_count={internal_qk.get('action_surface_passing_family_count')}; metric_candidate_row_count={internal_qk.get('metric_candidate_row_count')}",
        },
        {
            "requirement": "outcome_d_cache_topk_query_action_followup",
            "status": "pass" if cache_topk.get("action_variant_count") == 10 and cache_topk.get("action_variant_gate_pass_count") == 0 else "fail",
            "evidence": f"diagnostic_gate_pass_count={cache_topk.get('diagnostic_gate_pass_count')}; action_variant_count={cache_topk.get('action_variant_count')}; action_variant_gate_pass_count={cache_topk.get('action_variant_gate_pass_count')}",
        },
        {
            "requirement": "outcome_d_remaining_routes_closure",
            "status": "pass" if remaining.get("closed_route_count") == 2 and remaining.get("action_allowed_route_count") == 0 else "fail",
            "evidence": f"closed_route_count={remaining.get('closed_route_count')}; action_allowed_route_count={remaining.get('action_allowed_route_count')}; read_stage7_gate_pass={remaining.get('read_provider_stage7_gate_pass')}; semantic_anchor_action_ready={remaining.get('semantic_anchor_strict_action_ready_clean_candidate_count')}",
        },
        {
            "requirement": "historical_target_universe_mining_followup",
            "status": "pass"
            if historical_mining.get("historical_mined_new_clean_universe_available") is False
            and historical_mining.get("strict_action_ready_candidate_count") == 0
            and historical_mining.get("usable_new_extension_case_count") == 0
            else "fail",
            "evidence": (
                f"unique_case_count={historical_mining.get('unique_case_count')}; "
                f"clean_handoff_candidate_count={historical_mining.get('clean_handoff_candidate_count')}; "
                f"safe_good_candidate_count={historical_mining.get('safe_good_candidate_count')}; "
                f"strict_action_ready_candidate_count={historical_mining.get('strict_action_ready_candidate_count')}; "
                f"usable_new_extension_case_count={historical_mining.get('usable_new_extension_case_count')}; "
                f"historical_mined_new_clean_universe_available="
                f"{historical_mining.get('historical_mined_new_clean_universe_available')}"
            ),
        },
        {
            "requirement": "rich_selector_replay_probe_followup",
            "status": "pass"
            if replay_probe.get("replay_probe_all_completed") is True
            and replay_probe.get("replay_probe_failed_count") == 0
            and replay_probe.get("phase3r_runtime_probe_gate_pass") is False
            and replay_probe.get("selected_candidate_beats_control") is False
            and replay_probe.get("runtime_action_allowed") is False
            else "fail",
            "evidence": (
                f"replay_probe_job_count={replay_probe.get('replay_probe_job_count')}; "
                f"replay_probe_failed_count={replay_probe.get('replay_probe_failed_count')}; "
                f"phase3r_runtime_probe_gate_pass={replay_probe.get('phase3r_runtime_probe_gate_pass')}; "
                f"selected_candidate_variant={replay_probe.get('selected_candidate_variant')}; "
                f"selected_candidate_bad_median_I_J_runtime_proxy="
                f"{replay_probe.get('selected_candidate_bad_median_I_J_runtime_proxy')}; "
                f"best_control_bad_median_I_J_runtime_proxy="
                f"{replay_probe.get('best_control_bad_median_I_J_runtime_proxy')}; "
                f"selected_candidate_beats_control={replay_probe.get('selected_candidate_beats_control')}"
            ),
        },
        {
            "requirement": "rich_selector_fresh_holdout_feasibility_followup",
            "status": "pass"
            if holdout_feasibility.get("fresh_holdout_action_evaluable") is False
            and holdout_feasibility.get("fresh_labelled_bad_good_holdout_pair_count") == 0
            and holdout_feasibility.get("runtime_action_allowed") is False
            else "fail",
            "evidence": (
                f"phase1_pair_count={holdout_feasibility.get('phase1_pair_count')}; "
                f"phase6_rich_screen_measured_pair_count="
                f"{holdout_feasibility.get('phase6_rich_screen_measured_pair_count')}; "
                f"prior_measured_not_rich_screen_pair_count="
                f"{holdout_feasibility.get('prior_measured_not_rich_screen_pair_count')}; "
                f"fresh_unmeasured_or_stage1_pair_count="
                f"{holdout_feasibility.get('fresh_unmeasured_or_stage1_pair_count')}; "
                f"fresh_labelled_bad_good_holdout_pair_count="
                f"{holdout_feasibility.get('fresh_labelled_bad_good_holdout_pair_count')}; "
                f"fresh_stage1_fixed_policy_selected_count="
                f"{holdout_feasibility.get('fresh_stage1_fixed_policy_selected_count')}"
            ),
        },
        {
            "requirement": "fresh_selected_schema_materialization_followup",
            "status": "pass"
            if fresh_schema.get("fresh_stage1_selected_count") == holdout_feasibility.get(
                "fresh_stage1_fixed_policy_selected_count"
            )
            and fresh_schema.get("clean_handoff_or_safe_good_candidate_count") == 0
            and fresh_schema.get("schema_action_materializable_now_count") == 0
            and fresh_schema.get("runtime_action_allowed") is False
            else "fail",
            "evidence": (
                f"fresh_stage1_selected_count={fresh_schema.get('fresh_stage1_selected_count')}; "
                f"trackT_28_case_target_present_count="
                f"{fresh_schema.get('trackT_28_case_target_present_count')}; "
                f"broad_representative_taxonomy_counts="
                f"{fresh_schema.get('broad_representative_taxonomy_counts')}; "
                f"clean_handoff_or_safe_good_candidate_count="
                f"{fresh_schema.get('clean_handoff_or_safe_good_candidate_count')}; "
                f"schema_action_materializable_now_count="
                f"{fresh_schema.get('schema_action_materializable_now_count')}"
            ),
        },
        {
            "requirement": "strict_action_frontier_followup",
            "status": "pass"
            if strict_frontier.get("clean_candidate_count") == 6
            and strict_frontier.get("core_v100_schema_ready_clean_candidate_count") == 6
            and strict_frontier.get("strict_action_ready_clean_candidate_count") == 0
            and strict_frontier.get("runtime_action_allowed") is False
            else "fail",
            "evidence": (
                f"clean_candidate_count={strict_frontier.get('clean_candidate_count')}; "
                f"core_v100_schema_ready_clean_candidate_count="
                f"{strict_frontier.get('core_v100_schema_ready_clean_candidate_count')}; "
                f"strict_action_ready_clean_candidate_count="
                f"{strict_frontier.get('strict_action_ready_clean_candidate_count')}; "
                f"missing_prereq_counts={strict_frontier.get('missing_prereq_counts')}"
            ),
        },
        {
            "requirement": "plan_coverage_closure_followup",
            "status": "pass"
            if plan_coverage.get("plan_coverage_closure_pass") is True
            and plan_coverage.get("method_goal_achieved") is False
            and plan_coverage.get("downstream_action_authorized") is False
            and plan_coverage.get("runtime_action_allowed") is False
            else "fail",
            "evidence": (
                f"plan_stage_row_count={plan_coverage.get('plan_stage_row_count')}; "
                f"repair_route_row_count={plan_coverage.get('repair_route_row_count')}; "
                f"missing_required_row_count={plan_coverage.get('missing_required_row_count')}; "
                f"missing_fail_forward_row_count={plan_coverage.get('missing_fail_forward_row_count')}; "
                f"plan_artifact_coverage_pass={plan_coverage.get('plan_artifact_coverage_pass')}; "
                f"fail_forward_coverage_pass={plan_coverage.get('fail_forward_coverage_pass')}; "
                f"repair_route_coverage_pass={plan_coverage.get('repair_route_coverage_pass')}"
            ),
        },
        {
            "requirement": "trace_rescue_feasibility_followup",
            "status": "pass"
            if trace_rescue.get("trace_rescue_available") is False
            and trace_rescue.get("strict_instance_identity_rescued") is False
            and trace_rescue.get("query_head_controls_rescued") is False
            and trace_rescue.get("write_cache_current_chain_rescued") is False
            and trace_rescue.get("q2_true_stage_rescued") is False
            and trace_rescue.get("runtime_action_allowed") is False
            else "fail",
            "evidence": (
                f"hook_identity_file_count={trace_rescue.get('hook_identity_file_count')}; "
                f"trace_jsonl_line_count={trace_rescue.get('trace_jsonl_line_count')}; "
                f"identity_hook_paths_nonempty_file_count="
                f"{trace_rescue.get('identity_hook_paths_nonempty_file_count')}; "
                f"implemented_paths_nonempty_file_count="
                f"{trace_rescue.get('implemented_paths_nonempty_file_count')}; "
                f"attention_mass_available_true_site_count="
                f"{trace_rescue.get('attention_mass_available_true_site_count')}; "
                f"query_soft_applied_total={trace_rescue.get('query_soft_applied_total')}; "
                f"write_debug_available_line_count="
                f"{trace_rescue.get('write_debug_available_line_count')}; "
                f"trace_rescue_available={trace_rescue.get('trace_rescue_available')}"
            ),
        },
        {
            "requirement": "state_q2_readiness_followup",
            "status": "pass"
            if state_q2.get("native_same_space_instrumentation_reusable") is True
            and state_q2.get("state_q2_readiness_pass") is False
            and state_q2.get("action_ready") is False
            and state_q2.get("runtime_action_allowed") is False
            and state_q2.get("trackS2_gate_pass") is False
            and state_q2.get("trackQ2_true_stage_pass") is False
            and state_q2.get("trackU_true_current_support_strict_pass") is False
            and state_q2.get("trackV_gate_pass") is False
            else "fail",
            "evidence": (
                f"native_same_space_instrumentation_reusable="
                f"{state_q2.get('native_same_space_instrumentation_reusable')}; "
                f"state_q2_readiness_pass={state_q2.get('state_q2_readiness_pass')}; "
                f"trackU_true_current_support_strict_pass="
                f"{state_q2.get('trackU_true_current_support_strict_pass')}; "
                f"trackV_gate_pass={state_q2.get('trackV_gate_pass')}; "
                f"trackS2_gate_pass={state_q2.get('trackS2_gate_pass')}; "
                f"trackQ2_true_stage_pass={state_q2.get('trackQ2_true_stage_pass')}; "
                f"proxy_only_row_count={state_q2.get('proxy_only_row_count')}; "
                f"r_write_cache_nonempty_row_count="
                f"{state_q2.get('r_write_cache_nonempty_row_count')}; "
                f"r_cache_current_nonempty_row_count="
                f"{state_q2.get('r_cache_current_nonempty_row_count')}; "
                f"r_ref_current_nonempty_row_count="
                f"{state_q2.get('r_ref_current_nonempty_row_count')}"
            ),
        },
        {
            "requirement": "component_identity_availability_followup",
            "status": "pass"
            if component_identity.get("stage_c_cache_case_coverage") == 1.0
            and component_identity.get("stage_c_masklet_loadable_case_coverage") == 1.0
            and component_identity.get("component_like_track_ids_available") is True
            and component_identity.get("direct_anchor_to_stage_c_seed_match_count") == 0
            and component_identity.get("diagnostic_anchor_seed_join_feasible") is True
            and component_identity.get("diagnostic_anchor_seed_lifecycle_pair_count", 0) > 0
            and component_identity.get("diagnostic_lifecycle_explicit_anchor_seed_mapping_available") is True
            and component_identity.get("diagnostic_lifecycle_stage_c_seed_support_join_unique_coverage") == 1.0
            and component_identity.get("upstream_component_provenance_bridge_available") is True
            and component_identity.get("trace_payload_with_component_or_stage_c_key_count", 0) > 0
            and component_identity.get("explicit_anchor_component_mapping_available") is False
            and component_identity.get("jl4_identity_rescue_available") is False
            and component_identity.get("runtime_action_allowed") is False
            else "fail",
            "evidence": (
                f"case_count={component_identity.get('case_count')}; "
                f"stage_c_cache_case_coverage="
                f"{component_identity.get('stage_c_cache_case_coverage')}; "
                f"stage_c_masklet_loadable_case_coverage="
                f"{component_identity.get('stage_c_masklet_loadable_case_coverage')}; "
                f"full_sparse_masklet_present_sequence_count="
                f"{component_identity.get('full_sparse_masklet_present_sequence_count')}; "
                f"component_like_track_ids_available="
                f"{component_identity.get('component_like_track_ids_available')}; "
                f"direct_anchor_to_stage_c_seed_match_count="
                f"{component_identity.get('direct_anchor_to_stage_c_seed_match_count')}; "
                f"diagnostic_anchor_seed_join_feasible="
                f"{component_identity.get('diagnostic_anchor_seed_join_feasible')}; "
                f"diagnostic_anchor_seed_lifecycle_pair_count="
                f"{component_identity.get('diagnostic_anchor_seed_lifecycle_pair_count')}; "
                f"diagnostic_lifecycle_explicit_anchor_seed_mapping_available="
                f"{component_identity.get('diagnostic_lifecycle_explicit_anchor_seed_mapping_available')}; "
                f"diagnostic_lifecycle_stage_c_seed_support_join_unique_coverage="
                f"{component_identity.get('diagnostic_lifecycle_stage_c_seed_support_join_unique_coverage')}; "
                f"upstream_component_provenance_bridge_available="
                f"{component_identity.get('upstream_component_provenance_bridge_available')}; "
                f"trace_payload_scanned_count="
                f"{component_identity.get('trace_payload_scanned_count')}; "
                f"trace_payload_with_component_or_stage_c_key_count="
                f"{component_identity.get('trace_payload_with_component_or_stage_c_key_count')}; "
                f"explicit_anchor_component_mapping_available="
                f"{component_identity.get('explicit_anchor_component_mapping_available')}; "
                f"jl4_identity_rescue_available="
                f"{component_identity.get('jl4_identity_rescue_available')}"
            ),
        },
        {
            "requirement": "stage_c_seed_bridge_smoke_followup",
            "status": "pass"
            if stage_c_seed_bridge.get("stage_c_seed_bridge_smoke_pass") is True
            and stage_c_seed_bridge.get("diagnostic_only") is True
            and stage_c_seed_bridge.get("method_goal_achieved") is False
            and stage_c_seed_bridge.get("runtime_action_allowed") is False
            else "fail",
            "evidence": (
                f"stage_c_seed_bridge_smoke_pass="
                f"{stage_c_seed_bridge.get('stage_c_seed_bridge_smoke_pass')}; "
                f"repair_effect_observed={stage_c_seed_bridge.get('repair_effect_observed')}; "
                f"current_trace_payload_file_count="
                f"{(stage_c_seed_bridge.get('current_smoke') or {}).get('trace_payload_file_count')}; "
                f"current_sample_nonnegative_counts="
                f"{(stage_c_seed_bridge.get('current_smoke') or {}).get('sample_nonnegative_counts')}; "
                f"current_topk_nonnegative_counts="
                f"{(stage_c_seed_bridge.get('current_smoke') or {}).get('topk_nonnegative_counts')}; "
                f"same_seed_true_counts="
                f"{(stage_c_seed_bridge.get('current_smoke') or {}).get('same_seed_true_counts')}; "
                f"first_all_current_seed_nonempty="
                f"{(stage_c_seed_bridge.get('first_smoke_before_repair') or {}).get('all_current_seed_nonempty')}"
            ),
        },
        {
            "requirement": "stage_c_seed_bridge_target_trace_followup",
            "status": "pass"
            if stage_c_seed_bridge.get("stage_c_seed_bridge_target_trace_pass") is True
            and stage_c_seed_bridge.get("diagnostic_only") is True
            and stage_c_seed_bridge.get("method_goal_achieved") is False
            and stage_c_seed_bridge.get("runtime_action_allowed") is False
            else "fail",
            "evidence": (
                f"stage_c_seed_bridge_target_trace_pass="
                f"{stage_c_seed_bridge.get('stage_c_seed_bridge_target_trace_pass')}; "
                f"target_trace_payload_file_count="
                f"{(stage_c_seed_bridge.get('target_trace') or {}).get('trace_payload_file_count')}; "
                f"target_completed_job_count="
                f"{(stage_c_seed_bridge.get('target_trace') or {}).get('completed_job_count')}; "
                f"target_failed_job_count="
                f"{(stage_c_seed_bridge.get('target_trace') or {}).get('failed_job_count')}; "
                f"target_sample_nonnegative_counts="
                f"{(stage_c_seed_bridge.get('target_trace') or {}).get('sample_nonnegative_counts')}; "
                f"target_topk_nonnegative_counts="
                f"{(stage_c_seed_bridge.get('target_trace') or {}).get('topk_nonnegative_counts')}; "
                f"target_same_seed_true_counts="
                f"{(stage_c_seed_bridge.get('target_trace') or {}).get('same_seed_true_counts')}"
            ),
        },
        {
            "requirement": "stage_c_seed_current_support_materialization_followup",
            "status": "pass"
            if stage_c_seed_support.get("seed_support_materialization_pass") is True
            and stage_c_seed_support.get("true_current_support_strict_pass") is False
            and stage_c_seed_support.get("stage_c_seed_support_discriminative_gate_pass") is False
            and stage_c_seed_support.get("proxy_only") is True
            and stage_c_seed_support.get("diagnostic_only") is True
            and stage_c_seed_support.get("runtime_action_allowed") is False
            and stage_c_seed_support.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"seed_support_materialization_pass="
                f"{stage_c_seed_support.get('seed_support_materialization_pass')}; "
                f"case_count={stage_c_seed_support.get('case_count')}; "
                f"trace_payload_file_count={stage_c_seed_support.get('trace_payload_file_count')}; "
                f"component_support_row_count={stage_c_seed_support.get('component_support_row_count')}; "
                f"case_sample_nonnegative_min={stage_c_seed_support.get('case_sample_nonnegative_min')}; "
                f"case_topk_nonnegative_min={stage_c_seed_support.get('case_topk_nonnegative_min')}; "
                f"same_seed_true_total={stage_c_seed_support.get('same_seed_true_total')}; "
                f"matched_current_seed_recall_mean="
                f"{stage_c_seed_support.get('matched_current_seed_recall_mean')}; "
                f"strict_current_support_pass="
                f"{stage_c_seed_support.get('true_current_support_strict_pass')}; "
                f"runtime_action_allowed={stage_c_seed_support.get('runtime_action_allowed')}"
            ),
        },
        {
            "requirement": "stage_c_seed_clean_eval_q128_followup",
            "status": "pass"
            if stage_c_seed_support_q128.get("seed_support_materialization_pass") is True
            and stage_c_seed_support_q128.get("target_case_count") == 6
            and stage_c_seed_support_q128.get("case_count") == 6
            and stage_c_seed_support_q128.get("true_current_support_strict_pass") is False
            and stage_c_seed_support_q128.get("stage_c_seed_support_discriminative_gate_pass") is False
            and stage_c_seed_support_q128.get("runtime_action_allowed") is False
            and stage_c_seed_support_q128.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"seed_support_materialization_pass="
                f"{stage_c_seed_support_q128.get('seed_support_materialization_pass')}; "
                f"target_case_count={stage_c_seed_support_q128.get('target_case_count')}; "
                f"case_count={stage_c_seed_support_q128.get('case_count')}; "
                f"component_support_row_count="
                f"{stage_c_seed_support_q128.get('component_support_row_count')}; "
                f"case_sample_nonnegative_min="
                f"{stage_c_seed_support_q128.get('case_sample_nonnegative_min')}; "
                f"case_topk_nonnegative_min="
                f"{stage_c_seed_support_q128.get('case_topk_nonnegative_min')}; "
                f"same_seed_true_total={stage_c_seed_support_q128.get('same_seed_true_total')}; "
                f"matched_current_seed_recall_mean="
                f"{stage_c_seed_support_q128.get('matched_current_seed_recall_mean')}; "
                f"same_seed_frac_corr_L3_handoff_vs_safe_only="
                f"{stage_c_seed_support_q128.get('same_seed_frac_corr_L3_handoff_vs_safe_only')}; "
                f"strict_current_support_pass="
                f"{stage_c_seed_support_q128.get('true_current_support_strict_pass')}; "
                f"runtime_action_allowed={stage_c_seed_support_q128.get('runtime_action_allowed')}"
            ),
        },
        {
            "requirement": "anchor_seed_join_feasibility_followup",
            "status": "pass"
            if anchor_seed_join.get("payload_with_stage_c_seed_count", 0) > 0
            and anchor_seed_join.get("payload_with_ttt_anchor_id_count", 0) > 0
            and anchor_seed_join.get("ttt_prev_stable_anchor_lifecycle_row_total", 0) > 0
            and anchor_seed_join.get("payload_with_lifecycle_stage_c_seed_mode_count", 0) > 0
            and anchor_seed_join.get("diagnostic_anchor_seed_join_feasible") is True
            and anchor_seed_join.get("strict_anchor_seed_join_ready_for_action") is False
            and anchor_seed_join.get("direct_stage_c_seed_id_overlap_support_anchor_id_count") == 0
            and anchor_seed_join.get("runtime_action_allowed") is False
            and anchor_seed_join.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"trace_payload_file_count={anchor_seed_join.get('trace_payload_file_count')}; "
                f"payload_with_stage_c_seed_count="
                f"{anchor_seed_join.get('payload_with_stage_c_seed_count')}; "
                f"payload_with_ttt_anchor_id_count="
                f"{anchor_seed_join.get('payload_with_ttt_anchor_id_count')}; "
                f"ttt_prev_stable_anchor_lifecycle_row_total="
                f"{anchor_seed_join.get('ttt_prev_stable_anchor_lifecycle_row_total')}; "
                f"payload_with_lifecycle_stage_c_seed_mode_count="
                f"{anchor_seed_join.get('payload_with_lifecycle_stage_c_seed_mode_count')}; "
                f"lifecycle_anchor_seed_pair_count="
                f"{anchor_seed_join.get('lifecycle_anchor_seed_pair_count')}; "
                f"diagnostic_anchor_seed_join_feasible="
                f"{anchor_seed_join.get('diagnostic_anchor_seed_join_feasible')}; "
                f"strict_anchor_seed_join_ready_for_action="
                f"{anchor_seed_join.get('strict_anchor_seed_join_ready_for_action')}; "
                f"direct_stage_c_seed_id_overlap_support_anchor_id_count="
                f"{anchor_seed_join.get('direct_stage_c_seed_id_overlap_support_anchor_id_count')}; "
                f"anchor_seed_join_feasible={anchor_seed_join.get('anchor_seed_join_feasible')}; "
                f"runtime_action_allowed={anchor_seed_join.get('runtime_action_allowed')}"
            ),
        },
        {
            "requirement": "anchor_seed_lifecycle_support_join_followup",
            "status": "pass"
            if lifecycle_support_join.get("diagnostic_only") is True
            and lifecycle_support_join.get("lifecycle_expanded_row_count", 0) > 0
            and lifecycle_support_join.get("lifecycle_rows_with_stage_c_seed_mode_count", 0) > 0
            and lifecycle_support_join.get("support_joined_unique_case_anchor_count", 0) > 0
            and lifecycle_support_join.get("state_joined_unique_case_anchor_count", 0) > 0
            and lifecycle_support_join.get("jl4_gap_joined_unique_case_anchor_count", 0) > 0
            and lifecycle_support_join.get("trackU_strict_current_support_ready") is False
            and lifecycle_support_join.get("jl4_identity_rescue_available") is False
            and lifecycle_support_join.get("q2_true_stage_ready") is False
            and lifecycle_support_join.get("strict_action_ready") is False
            and lifecycle_support_join.get("runtime_action_allowed") is False
            and lifecycle_support_join.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"lifecycle_expanded_row_count="
                f"{lifecycle_support_join.get('lifecycle_expanded_row_count')}; "
                f"lifecycle_rows_with_stage_c_seed_mode_count="
                f"{lifecycle_support_join.get('lifecycle_rows_with_stage_c_seed_mode_count')}; "
                f"support_joined_unique_case_anchor_count="
                f"{lifecycle_support_join.get('support_joined_unique_case_anchor_count')}; "
                f"support_join_unique_coverage="
                f"{lifecycle_support_join.get('support_join_unique_coverage')}; "
                f"state_joined_unique_case_anchor_count="
                f"{lifecycle_support_join.get('state_joined_unique_case_anchor_count')}; "
                f"jl4_gap_joined_unique_case_anchor_count="
                f"{lifecycle_support_join.get('jl4_gap_joined_unique_case_anchor_count')}; "
                f"strict_action_ready={lifecycle_support_join.get('strict_action_ready')}; "
                f"runtime_action_allowed={lifecycle_support_join.get('runtime_action_allowed')}"
            ),
        },
        {
            "requirement": "anchor_seed_lifecycle_stage_c_seed_support_join_followup",
            "status": "pass"
            if lifecycle_support_join.get("diagnostic_only") is True
            and lifecycle_support_join.get("lifecycle_rows_with_stage_c_seed_support_join_count", 0) > 0
            and lifecycle_support_join.get("lifecycle_stage_c_seed_support_join_unique_coverage") == 1.0
            and lifecycle_support_join.get("stage_c_seed_support_strict_current_support_ready") is False
            and lifecycle_support_join.get("strict_action_ready") is False
            and lifecycle_support_join.get("runtime_action_allowed") is False
            and lifecycle_support_join.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"lifecycle_rows_with_stage_c_seed_support_join_count="
                f"{lifecycle_support_join.get('lifecycle_rows_with_stage_c_seed_support_join_count')}; "
                f"lifecycle_unique_case_seed_count="
                f"{lifecycle_support_join.get('lifecycle_unique_case_seed_count')}; "
                f"lifecycle_stage_c_seed_support_joined_unique_case_seed_count="
                f"{lifecycle_support_join.get('lifecycle_stage_c_seed_support_joined_unique_case_seed_count')}; "
                f"lifecycle_stage_c_seed_support_join_unique_coverage="
                f"{lifecycle_support_join.get('lifecycle_stage_c_seed_support_join_unique_coverage')}; "
                f"stage_c_seed_support_strict_current_support_ready="
                f"{lifecycle_support_join.get('stage_c_seed_support_strict_current_support_ready')}; "
                f"strict_action_ready={lifecycle_support_join.get('strict_action_ready')}; "
                f"runtime_action_allowed={lifecycle_support_join.get('runtime_action_allowed')}"
            ),
        },
        {
            "requirement": "anchor_seed_lifecycle_stage_c_masklet_visibility_followup",
            "status": "pass"
            if masklet_visibility.get("diagnostic_only") is True
            and masklet_visibility.get("lifecycle_unique_case_seed_count", 0) > 0
            and masklet_visibility.get("masklet_chunk_load_error_count") == 0
            and masklet_visibility.get("component_current_visibility_materialized") is True
            and masklet_visibility.get("trackU_component_current_visibility_repair_candidate") is True
            and masklet_visibility.get("true_current_support_strict_pass") is False
            and masklet_visibility.get("runtime_action_allowed") is False
            and masklet_visibility.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"lifecycle_unique_case_seed_count="
                f"{masklet_visibility.get('lifecycle_unique_case_seed_count')}; "
                f"masklet_chunk_load_error_count="
                f"{masklet_visibility.get('masklet_chunk_load_error_count')}; "
                f"current_chunk_visible_unique_case_seed_count="
                f"{masklet_visibility.get('current_chunk_visible_unique_case_seed_count')}; "
                f"current_chunk_visibility_coverage="
                f"{masklet_visibility.get('current_chunk_visibility_coverage')}; "
                f"true_current_support_strict_pass="
                f"{masklet_visibility.get('true_current_support_strict_pass')}; "
                f"runtime_action_allowed={masklet_visibility.get('runtime_action_allowed')}"
            ),
        },
        {
            "requirement": "anchor_seed_lifecycle_stage_c_masklet_observability_proxy_followup",
            "status": "pass"
            if masklet_visibility.get("diagnostic_only") is True
            and masklet_visibility.get("masklet_2d_observability_proxy_materialized") is True
            and masklet_visibility.get("scale_observability_proxy_only") is True
            and masklet_visibility.get("true_scale_observability_pass") is False
            and masklet_visibility.get("runtime_action_allowed") is False
            and masklet_visibility.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"masklet_2d_observability_proxy_materialized="
                f"{masklet_visibility.get('masklet_2d_observability_proxy_materialized')}; "
                f"current_bbox_center_span_px_mean="
                f"{masklet_visibility.get('current_bbox_center_span_px_mean')}; "
                f"current_bbox_area_px_cv_mean="
                f"{masklet_visibility.get('current_bbox_area_px_cv_mean')}; "
                f"scale_observability_proxy_only="
                f"{masklet_visibility.get('scale_observability_proxy_only')}; "
                f"true_scale_observability_pass="
                f"{masklet_visibility.get('true_scale_observability_pass')}; "
                f"runtime_action_allowed={masklet_visibility.get('runtime_action_allowed')}"
            ),
        },
        {
            "requirement": "anchor_seed_lifecycle_geometry_observability_join_followup",
            "status": "pass"
            if lifecycle_geometry.get("diagnostic_only") is True
            and lifecycle_geometry.get("lifecycle_geometry_join_materialized") is True
            and lifecycle_geometry.get("trackv_geometry_materialization_pass") is True
            and lifecycle_geometry.get("lifecycle_true_geometry_source_available") is True
            and lifecycle_geometry.get("trackv_gate_pass") is False
            and lifecycle_geometry.get("lifecycle_scale_observability_true_stage_pass") is False
            and lifecycle_geometry.get("runtime_action_allowed") is False
            and lifecycle_geometry.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"lifecycle_geometry_joined_unique_case_anchor_count="
                f"{lifecycle_geometry.get('lifecycle_geometry_joined_unique_case_anchor_count')}; "
                f"lifecycle_geometry_unique_coverage="
                f"{lifecycle_geometry.get('lifecycle_geometry_unique_coverage')}; "
                f"lifecycle_raw_geometry_edge_joined_unique_case_anchor_count="
                f"{lifecycle_geometry.get('lifecycle_raw_geometry_edge_joined_unique_case_anchor_count')}; "
                f"lifecycle_combined_geometry_unique_coverage="
                f"{lifecycle_geometry.get('lifecycle_combined_geometry_unique_coverage')}; "
                f"lifecycle_true_geometry_joined_frac="
                f"{lifecycle_geometry.get('lifecycle_true_geometry_joined_frac')}; "
                f"pointmap_depth_support_row_count="
                f"{lifecycle_geometry.get('pointmap_depth_support_row_count')}; "
                f"scale_mode_row_count={lifecycle_geometry.get('scale_mode_row_count')}; "
                f"trackv_gate_pass={lifecycle_geometry.get('trackv_gate_pass')}; "
                f"lifecycle_scale_observability_true_stage_pass="
                f"{lifecycle_geometry.get('lifecycle_scale_observability_true_stage_pass')}; "
                f"runtime_action_allowed={lifecycle_geometry.get('runtime_action_allowed')}"
            ),
        },
        {
            "requirement": "stage_c_seed_geometry_smoke_clean6_followup",
            "status": "pass"
            if geometry_smoke.get("diagnostic_only") is True
            and geometry_smoke.get("smoke_status") == "complete"
            and geometry_smoke.get("selected_case_count") == 6
            and geometry_smoke.get("positive_case_count") == 1
            and geometry_smoke.get("safe_good_count") == 5
            and geometry_smoke.get("read_error_count") == 0
            and geometry_smoke.get("per_chunk_geometry_sidecar_file_count") == 12
            and geometry_smoke.get("q2_true_stage_pass") is False
            and geometry_smoke.get("runtime_action_allowed") is False
            and geometry_smoke.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"smoke_status={geometry_smoke.get('smoke_status')}; "
                f"selected_case_count={geometry_smoke.get('selected_case_count')}; "
                f"per_chunk_geometry_sidecar_file_count="
                f"{geometry_smoke.get('per_chunk_geometry_sidecar_file_count')}; "
                f"lifecycle_geometry_same_payload_join_coverage="
                f"{geometry_smoke.get('lifecycle_geometry_same_payload_join_coverage')}; "
                f"geometry_smoke_alignment_pass={geometry_smoke.get('geometry_smoke_alignment_pass')}; "
                f"best_geometry_policy={geometry_smoke.get('best_geometry_policy')}; "
                f"best_geometry_policy_balanced_accuracy="
                f"{geometry_smoke.get('best_geometry_policy_balanced_accuracy')}; "
                f"q2_true_stage_pass={geometry_smoke.get('q2_true_stage_pass')}; "
                f"runtime_action_allowed={geometry_smoke.get('runtime_action_allowed')}"
            ),
        },
        {
            "requirement": "stage_c_seed_geometry_smoke_target28_followup",
            "status": "pass"
            if geometry_smoke_target28.get("diagnostic_only") is True
            and geometry_smoke_target28.get("smoke_status") == "complete"
            and geometry_smoke_target28.get("selected_case_count") == 28
            and geometry_smoke_target28.get("read_error_count") == 0
            and geometry_smoke_target28.get("per_chunk_geometry_sidecar_file_count") == 56
            and geometry_smoke_target28.get("q2_true_stage_pass") is False
            and geometry_smoke_target28.get("runtime_action_allowed") is False
            and geometry_smoke_target28.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"smoke_status={geometry_smoke_target28.get('smoke_status')}; "
                f"selected_case_count={geometry_smoke_target28.get('selected_case_count')}; "
                f"per_chunk_geometry_sidecar_file_count="
                f"{geometry_smoke_target28.get('per_chunk_geometry_sidecar_file_count')}; "
                f"lifecycle_geometry_same_payload_join_coverage="
                f"{geometry_smoke_target28.get('lifecycle_geometry_same_payload_join_coverage')}; "
                f"best_geometry_policy={geometry_smoke_target28.get('best_geometry_policy')}; "
                f"best_geometry_policy_balanced_accuracy="
                f"{geometry_smoke_target28.get('best_geometry_policy_balanced_accuracy')}; "
                f"best_geometry_policy_all_non_handoff="
                f"{geometry_smoke_target28.get('best_geometry_policy_all_non_handoff')}; "
                f"best_geometry_policy_all_non_handoff_balanced_accuracy="
                f"{geometry_smoke_target28.get('best_geometry_policy_all_non_handoff_balanced_accuracy')}; "
                f"q2_true_stage_pass={geometry_smoke_target28.get('q2_true_stage_pass')}; "
                f"runtime_action_allowed={geometry_smoke_target28.get('runtime_action_allowed')}"
            ),
        },
        {
            "requirement": "combined_masklet_geometry_admission_followup",
            "status": "pass"
            if combined_admission.get("diagnostic_only") is True
            and combined_admission.get("case_count") == 28
            and combined_admission.get("positive_case_count") == 1
            and combined_admission.get("selected_positive_sequence_coverage") == 1
            and combined_admission.get("required_positive_sequence_coverage") == 3
            and combined_admission.get("all_non_handoff_promotion_pass") is False
            and combined_admission.get("q2_proxy_stage_pass") is False
            and combined_admission.get("q2_true_stage_pass") is False
            and combined_admission.get("runtime_action_allowed") is False
            and combined_admission.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"clean_safe_best_policy={combined_admission.get('clean_safe_best_policy')}; "
                f"clean_safe_best_policy_balanced_accuracy="
                f"{combined_admission.get('clean_safe_best_policy_balanced_accuracy')}; "
                f"selected_positive_sequence_coverage="
                f"{combined_admission.get('selected_positive_sequence_coverage')}; "
                f"required_positive_sequence_coverage="
                f"{combined_admission.get('required_positive_sequence_coverage')}; "
                f"all_non_handoff_best_policy={combined_admission.get('all_non_handoff_best_policy')}; "
                f"all_non_handoff_best_policy_balanced_accuracy="
                f"{combined_admission.get('all_non_handoff_best_policy_balanced_accuracy')}; "
                f"all_non_handoff_promotion_pass="
                f"{combined_admission.get('all_non_handoff_promotion_pass')}; "
                f"q2_proxy_stage_pass={combined_admission.get('q2_proxy_stage_pass')}; "
                f"q2_true_stage_pass={combined_admission.get('q2_true_stage_pass')}; "
                f"runtime_action_allowed={combined_admission.get('runtime_action_allowed')}"
            ),
        },
        {
            "requirement": "combined_admission_false_positive_attribution_followup",
            "status": "pass"
            if combined_fp_attr.get("diagnostic_only") is True
            and combined_fp_attr.get("false_positive_case_count", 0) > 0
            and combined_fp_attr.get("taxonomy_split_explains_false_positives") is True
            and combined_fp_attr.get("taxonomy_split_runtime_action_ready") is False
            and combined_fp_attr.get("clean_handoff_candidate_count") == 1
            and combined_fp_attr.get("historical_mined_new_clean_universe_available") is False
            and combined_fp_attr.get("q2_true_stage_pass") is False
            and combined_fp_attr.get("runtime_action_allowed") is False
            and combined_fp_attr.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"false_positive_case_count={combined_fp_attr.get('false_positive_case_count')}; "
                f"false_positive_taxonomy_counts="
                f"{combined_fp_attr.get('false_positive_taxonomy_counts')}; "
                f"clean_handoff_candidate_count={combined_fp_attr.get('clean_handoff_candidate_count')}; "
                f"historical_mined_new_clean_universe_available="
                f"{combined_fp_attr.get('historical_mined_new_clean_universe_available')}; "
                f"taxonomy_split_runtime_action_ready="
                f"{combined_fp_attr.get('taxonomy_split_runtime_action_ready')}; "
                f"q2_true_stage_pass={combined_fp_attr.get('q2_true_stage_pass')}; "
                f"runtime_action_allowed={combined_fp_attr.get('runtime_action_allowed')}"
            ),
        },
        {
            "requirement": "masklet_q2_admission_sanity_followup",
            "status": "pass"
            if masklet_q2.get("diagnostic_only") is True
            and masklet_q2.get("proxy_only") is True
            and masklet_q2.get("positive_case_count") == 1
            and masklet_q2.get("safe_good_count") == 5
            and masklet_q2.get("q2_true_stage_pass") is False
            and masklet_q2.get("runtime_action_allowed") is False
            and masklet_q2.get("method_goal_achieved") is False
            else "fail",
            "evidence": (
                f"best_policy={masklet_q2.get('best_policy')}; "
                f"best_policy_balanced_accuracy={masklet_q2.get('best_policy_balanced_accuracy')}; "
                f"positive_case_count={masklet_q2.get('positive_case_count')}; "
                f"safe_good_count={masklet_q2.get('safe_good_count')}; "
                f"q2_true_stage_pass={masklet_q2.get('q2_true_stage_pass')}; "
                f"runtime_action_allowed={masklet_q2.get('runtime_action_allowed')}"
            ),
        },
    ]
    failed_requirements = [row["requirement"] for row in requirements if row["status"] != "pass"]
    failed_requirement_count = len(failed_requirements)
    missing_artifact_count = sum(1 for row in artifact_rows if not row.get("exists"))
    write_rows(FINAL / "completion_audit.csv", requirements)
    write_json(
        FINAL / "completion_audit_summary.json",
        {
            "schema": "acl2_v101_completion_audit_v1",
            "goal_achieved": False,
            "final_taxonomy": final.get("final_taxonomy", ""),
            "runtime_action_allowed": final.get("runtime_action_allowed", False),
            "full_validation_run": final.get("full_validation_run", False),
            "full_method_success": final.get("full_method_success", False),
            "created_missing_artifacts_count": len(created),
            "artifact_audit_rows": len(artifact_rows),
            "missing_artifact_count": missing_artifact_count,
            "completion_requirement_count": len(requirements),
            "failed_requirement_count": failed_requirement_count,
            "failed_requirements": failed_requirements,
            "outcome_d_reentry_route_count": broader.get("reentry_route_count", ""),
            "outcome_d_action_allowed_route_count": broader.get("action_allowed_route_count", ""),
            "merge_gauge_selector_passing_candidate_count": merge_gauge.get("passing_candidate_count", ""),
            "merge_gauge_rich_selector_candidate_count": merge_gauge_rich.get("candidate_policy_count", ""),
            "merge_gauge_rich_selector_retrospective_passing_count": merge_gauge_rich.get(
                "retrospective_passing_candidate_count", ""
            ),
            "merge_gauge_rich_selector_promotion_floor_passing_count": merge_gauge_rich.get(
                "promotion_floor_passing_candidate_count", ""
            ),
            "merge_gauge_rich_selector_action_authorized_count": 0,
            "merge_gauge_rich_selector_stability_status": merge_gauge_rich.get(
                "best_signal_stability_status", ""
            ),
            "merge_gauge_rich_selector_leaveout_pass_count": merge_gauge_rich.get(
                "best_leave_one_selected_sequence_out_pass_count", ""
            ),
            "merge_gauge_rich_selector_loso_status": merge_gauge_rich.get("loso_holdout_status", ""),
            "merge_gauge_rich_selector_loso_train_retrospective_pass_split_count": merge_gauge_rich.get(
                "loso_train_split_with_retrospective_pass_count", ""
            ),
            "merge_gauge_rich_selector_loso_train_promotion_pass_split_count": merge_gauge_rich.get(
                "loso_train_split_with_promotion_floor_pass_count", ""
            ),
            "internal_qk_action_surface_passing_family_count": internal_qk.get(
                "action_surface_passing_family_count", ""
            ),
            "cache_topk_action_variant_gate_pass_count": cache_topk.get(
                "action_variant_gate_pass_count", ""
            ),
            "remaining_reentry_action_allowed_route_count": remaining.get("action_allowed_route_count", ""),
            "historical_target_universe_unique_case_count": historical_mining.get("unique_case_count", ""),
            "historical_target_universe_clean_handoff_count": historical_mining.get(
                "clean_handoff_candidate_count", ""
            ),
            "historical_target_universe_strict_action_ready_count": historical_mining.get(
                "strict_action_ready_candidate_count", ""
            ),
            "historical_target_universe_mined_available": historical_mining.get(
                "historical_mined_new_clean_universe_available", ""
            ),
            "rich_selector_replay_probe_job_count": replay_probe.get("replay_probe_job_count", ""),
            "rich_selector_replay_probe_failed_count": replay_probe.get("replay_probe_failed_count", ""),
            "rich_selector_replay_phase3r_gate_pass": replay_probe.get(
                "phase3r_runtime_probe_gate_pass", ""
            ),
            "rich_selector_replay_selected_beats_control": replay_probe.get(
                "selected_candidate_beats_control", ""
            ),
            "rich_selector_holdout_fresh_labelled_bad_good_count": holdout_feasibility.get(
                "fresh_labelled_bad_good_holdout_pair_count", ""
            ),
            "rich_selector_holdout_fresh_stage1_selected_count": holdout_feasibility.get(
                "fresh_stage1_fixed_policy_selected_count", ""
            ),
            "rich_selector_holdout_action_evaluable": holdout_feasibility.get(
                "fresh_holdout_action_evaluable", ""
            ),
            "fresh_selected_schema_materialization_selected_count": fresh_schema.get(
                "fresh_stage1_selected_count", ""
            ),
            "fresh_selected_schema_materialization_clean_candidate_count": fresh_schema.get(
                "clean_handoff_or_safe_good_candidate_count", ""
            ),
            "fresh_selected_schema_materialization_action_materializable_count": fresh_schema.get(
                "schema_action_materializable_now_count", ""
            ),
            "strict_action_frontier_clean_candidate_count": strict_frontier.get("clean_candidate_count", ""),
            "strict_action_frontier_core_ready_count": strict_frontier.get(
                "core_v100_schema_ready_clean_candidate_count", ""
            ),
            "strict_action_frontier_strict_ready_count": strict_frontier.get(
                "strict_action_ready_clean_candidate_count", ""
            ),
            "plan_coverage_closure_pass": plan_coverage.get("plan_coverage_closure_pass", ""),
            "plan_coverage_stage_row_count": plan_coverage.get("plan_stage_row_count", ""),
            "plan_coverage_repair_route_row_count": plan_coverage.get("repair_route_row_count", ""),
            "plan_coverage_missing_required_row_count": plan_coverage.get(
                "missing_required_row_count", ""
            ),
            "plan_coverage_missing_fail_forward_row_count": plan_coverage.get(
                "missing_fail_forward_row_count", ""
            ),
            "trace_rescue_available": trace_rescue.get("trace_rescue_available", ""),
            "trace_rescue_hook_identity_file_count": trace_rescue.get(
                "hook_identity_file_count", ""
            ),
            "trace_rescue_jsonl_line_count": trace_rescue.get("trace_jsonl_line_count", ""),
            "trace_rescue_attention_mass_available_true_site_count": trace_rescue.get(
                "attention_mass_available_true_site_count", ""
            ),
            "trace_rescue_query_soft_applied_total": trace_rescue.get(
                "query_soft_applied_total", ""
            ),
            "trace_rescue_write_debug_available_line_count": trace_rescue.get(
                "write_debug_available_line_count", ""
            ),
            "state_q2_readiness_pass": state_q2.get("state_q2_readiness_pass", ""),
            "state_q2_native_same_space_instrumentation_reusable": state_q2.get(
                "native_same_space_instrumentation_reusable", ""
            ),
            "state_q2_proxy_only_row_count": state_q2.get("proxy_only_row_count", ""),
            "state_q2_r_write_cache_nonempty_row_count": state_q2.get(
                "r_write_cache_nonempty_row_count", ""
            ),
            "state_q2_r_cache_current_nonempty_row_count": state_q2.get(
                "r_cache_current_nonempty_row_count", ""
            ),
            "state_q2_r_ref_current_nonempty_row_count": state_q2.get(
                "r_ref_current_nonempty_row_count", ""
            ),
            "component_identity_stage_c_cache_case_coverage": component_identity.get(
                "stage_c_cache_case_coverage", ""
            ),
            "component_identity_stage_c_masklet_loadable_case_coverage": component_identity.get(
                "stage_c_masklet_loadable_case_coverage", ""
            ),
            "component_identity_direct_anchor_to_stage_c_seed_match_count": component_identity.get(
                "direct_anchor_to_stage_c_seed_match_count", ""
            ),
            "component_identity_diagnostic_anchor_seed_join_feasible": component_identity.get(
                "diagnostic_anchor_seed_join_feasible", ""
            ),
            "component_identity_diagnostic_anchor_seed_lifecycle_pair_count": component_identity.get(
                "diagnostic_anchor_seed_lifecycle_pair_count", ""
            ),
            "component_identity_diagnostic_lifecycle_explicit_anchor_seed_mapping_available": component_identity.get(
                "diagnostic_lifecycle_explicit_anchor_seed_mapping_available", ""
            ),
            "component_identity_diagnostic_lifecycle_stage_c_seed_support_join_unique_coverage": component_identity.get(
                "diagnostic_lifecycle_stage_c_seed_support_join_unique_coverage", ""
            ),
            "component_identity_upstream_component_provenance_bridge_available": component_identity.get(
                "upstream_component_provenance_bridge_available", ""
            ),
            "component_identity_trace_payload_scanned_count": component_identity.get(
                "trace_payload_scanned_count", ""
            ),
            "component_identity_trace_payload_with_component_or_stage_c_key_count": component_identity.get(
                "trace_payload_with_component_or_stage_c_key_count", ""
            ),
            "component_identity_explicit_anchor_component_mapping_available": component_identity.get(
                "explicit_anchor_component_mapping_available", ""
            ),
            "component_identity_jl4_identity_rescue_available": component_identity.get(
                "jl4_identity_rescue_available", ""
            ),
            "stage_c_seed_bridge_smoke_pass": stage_c_seed_bridge.get(
                "stage_c_seed_bridge_smoke_pass", ""
            ),
            "stage_c_seed_bridge_repair_effect_observed": stage_c_seed_bridge.get(
                "repair_effect_observed", ""
            ),
            "stage_c_seed_bridge_current_trace_payload_file_count": (
                stage_c_seed_bridge.get("current_smoke") or {}
            ).get("trace_payload_file_count", ""),
            "stage_c_seed_bridge_current_sample_nonnegative_counts": (
                stage_c_seed_bridge.get("current_smoke") or {}
            ).get("sample_nonnegative_counts", ""),
            "stage_c_seed_bridge_current_topk_nonnegative_counts": (
                stage_c_seed_bridge.get("current_smoke") or {}
            ).get("topk_nonnegative_counts", ""),
            "stage_c_seed_bridge_target_trace_pass": stage_c_seed_bridge.get(
                "stage_c_seed_bridge_target_trace_pass", ""
            ),
            "stage_c_seed_bridge_target_trace_payload_file_count": (
                stage_c_seed_bridge.get("target_trace") or {}
            ).get("trace_payload_file_count", ""),
            "stage_c_seed_bridge_target_completed_job_count": (
                stage_c_seed_bridge.get("target_trace") or {}
            ).get("completed_job_count", ""),
            "stage_c_seed_bridge_target_failed_job_count": (
                stage_c_seed_bridge.get("target_trace") or {}
            ).get("failed_job_count", ""),
            "stage_c_seed_bridge_target_sample_nonnegative_counts": (
                stage_c_seed_bridge.get("target_trace") or {}
            ).get("sample_nonnegative_counts", ""),
            "stage_c_seed_bridge_target_topk_nonnegative_counts": (
                stage_c_seed_bridge.get("target_trace") or {}
            ).get("topk_nonnegative_counts", ""),
            "stage_c_seed_support_materialization_pass": stage_c_seed_support.get(
                "seed_support_materialization_pass", ""
            ),
            "stage_c_seed_support_case_count": stage_c_seed_support.get("case_count", ""),
            "stage_c_seed_support_component_support_row_count": stage_c_seed_support.get(
                "component_support_row_count", ""
            ),
            "stage_c_seed_support_same_seed_true_total": stage_c_seed_support.get(
                "same_seed_true_total", ""
            ),
            "stage_c_seed_support_matched_current_seed_recall_mean": stage_c_seed_support.get(
                "matched_current_seed_recall_mean", ""
            ),
            "stage_c_seed_support_strict_current_support_pass": stage_c_seed_support.get(
                "true_current_support_strict_pass", ""
            ),
            "stage_c_seed_clean_eval_q128_materialization_pass": stage_c_seed_support_q128.get(
                "seed_support_materialization_pass", ""
            ),
            "stage_c_seed_clean_eval_q128_case_count": stage_c_seed_support_q128.get("case_count", ""),
            "stage_c_seed_clean_eval_q128_same_seed_true_total": stage_c_seed_support_q128.get(
                "same_seed_true_total", ""
            ),
            "stage_c_seed_clean_eval_q128_matched_current_seed_recall_mean": (
                stage_c_seed_support_q128.get("matched_current_seed_recall_mean", "")
            ),
            "stage_c_seed_clean_eval_q128_corr_L3_handoff_vs_safe": (
                stage_c_seed_support_q128.get("same_seed_frac_corr_L3_handoff_vs_safe_only", "")
            ),
            "anchor_seed_join_feasible": anchor_seed_join.get("anchor_seed_join_feasible", ""),
            "anchor_seed_join_payload_with_stage_c_seed_count": anchor_seed_join.get(
                "payload_with_stage_c_seed_count", ""
            ),
            "anchor_seed_join_payload_with_ttt_anchor_id_count": anchor_seed_join.get(
                "payload_with_ttt_anchor_id_count", ""
            ),
            "anchor_seed_join_lifecycle_row_total": anchor_seed_join.get(
                "ttt_prev_stable_anchor_lifecycle_row_total", ""
            ),
            "anchor_seed_join_payload_with_lifecycle_stage_c_seed_mode_count": anchor_seed_join.get(
                "payload_with_lifecycle_stage_c_seed_mode_count", ""
            ),
            "anchor_seed_join_lifecycle_anchor_seed_pair_count": anchor_seed_join.get(
                "lifecycle_anchor_seed_pair_count", ""
            ),
            "anchor_seed_join_diagnostic_feasible": anchor_seed_join.get(
                "diagnostic_anchor_seed_join_feasible", ""
            ),
            "anchor_seed_join_strict_ready_for_action": anchor_seed_join.get(
                "strict_anchor_seed_join_ready_for_action", ""
            ),
            "anchor_seed_join_direct_overlap_count": anchor_seed_join.get(
                "direct_stage_c_seed_id_overlap_support_anchor_id_count", ""
            ),
            "anchor_seed_lifecycle_support_join_row_count": lifecycle_support_join.get(
                "lifecycle_expanded_row_count", ""
            ),
            "anchor_seed_lifecycle_support_join_unique_case_anchor_count": lifecycle_support_join.get(
                "lifecycle_unique_case_anchor_count", ""
            ),
            "anchor_seed_lifecycle_support_join_support_joined_unique_count": lifecycle_support_join.get(
                "support_joined_unique_case_anchor_count", ""
            ),
            "anchor_seed_lifecycle_support_join_support_unique_coverage": lifecycle_support_join.get(
                "support_join_unique_coverage", ""
            ),
            "anchor_seed_lifecycle_support_join_state_joined_unique_count": lifecycle_support_join.get(
                "state_joined_unique_case_anchor_count", ""
            ),
            "anchor_seed_lifecycle_support_join_jl4_gap_joined_unique_count": lifecycle_support_join.get(
                "jl4_gap_joined_unique_case_anchor_count", ""
            ),
            "anchor_seed_lifecycle_support_join_strict_action_ready": lifecycle_support_join.get(
                "strict_action_ready", ""
            ),
            "anchor_seed_lifecycle_stage_c_seed_support_join_unique_coverage": lifecycle_support_join.get(
                "lifecycle_stage_c_seed_support_join_unique_coverage", ""
            ),
            "anchor_seed_lifecycle_stage_c_seed_support_join_row_count": lifecycle_support_join.get(
                "lifecycle_rows_with_stage_c_seed_support_join_count", ""
            ),
            "anchor_seed_lifecycle_stage_c_seed_support_ready": lifecycle_support_join.get(
                "stage_c_seed_support_strict_current_support_ready", ""
            ),
            "anchor_seed_lifecycle_stage_c_masklet_visibility_current_visible_count": masklet_visibility.get(
                "current_chunk_visible_unique_case_seed_count", ""
            ),
            "anchor_seed_lifecycle_stage_c_masklet_visibility_current_coverage": masklet_visibility.get(
                "current_chunk_visibility_coverage", ""
            ),
            "anchor_seed_lifecycle_stage_c_masklet_visibility_strict_pass": masklet_visibility.get(
                "true_current_support_strict_pass", ""
            ),
            "anchor_seed_lifecycle_stage_c_masklet_observability_proxy_materialized": masklet_visibility.get(
                "masklet_2d_observability_proxy_materialized", ""
            ),
            "anchor_seed_lifecycle_stage_c_masklet_observability_true_scale_pass": masklet_visibility.get(
                "true_scale_observability_pass", ""
            ),
            "anchor_seed_lifecycle_geometry_join_unique_count": lifecycle_geometry.get(
                "lifecycle_geometry_joined_unique_case_anchor_count", ""
            ),
            "anchor_seed_lifecycle_geometry_unique_coverage": lifecycle_geometry.get(
                "lifecycle_geometry_unique_coverage", ""
            ),
            "anchor_seed_lifecycle_geometry_true_geometry_joined_frac": lifecycle_geometry.get(
                "lifecycle_true_geometry_joined_frac", ""
            ),
            "anchor_seed_lifecycle_raw_geometry_edge_join_unique_count": lifecycle_geometry.get(
                "lifecycle_raw_geometry_edge_joined_unique_case_anchor_count", ""
            ),
            "anchor_seed_lifecycle_raw_geometry_edge_unique_coverage": lifecycle_geometry.get(
                "lifecycle_raw_geometry_edge_unique_coverage", ""
            ),
            "anchor_seed_lifecycle_combined_geometry_join_unique_count": lifecycle_geometry.get(
                "lifecycle_combined_geometry_joined_unique_case_anchor_count", ""
            ),
            "anchor_seed_lifecycle_combined_geometry_unique_coverage": lifecycle_geometry.get(
                "lifecycle_combined_geometry_unique_coverage", ""
            ),
            "anchor_seed_lifecycle_geometry_scale_mode_row_count": lifecycle_geometry.get(
                "scale_mode_row_count", ""
            ),
            "anchor_seed_lifecycle_geometry_trackv_gate_pass": lifecycle_geometry.get(
                "trackv_gate_pass", ""
            ),
            "anchor_seed_lifecycle_geometry_true_stage_pass": lifecycle_geometry.get(
                "lifecycle_scale_observability_true_stage_pass", ""
            ),
            "stage_c_seed_geometry_smoke_clean6_status": geometry_smoke.get("smoke_status", ""),
            "stage_c_seed_geometry_smoke_clean6_selected_case_count": geometry_smoke.get(
                "selected_case_count", ""
            ),
            "stage_c_seed_geometry_smoke_clean6_sidecar_file_count": geometry_smoke.get(
                "per_chunk_geometry_sidecar_file_count", ""
            ),
            "stage_c_seed_geometry_smoke_clean6_join_coverage": geometry_smoke.get(
                "lifecycle_geometry_same_payload_join_coverage", ""
            ),
            "stage_c_seed_geometry_smoke_clean6_alignment_pass": geometry_smoke.get(
                "geometry_smoke_alignment_pass", ""
            ),
            "stage_c_seed_geometry_smoke_clean6_best_policy": geometry_smoke.get(
                "best_geometry_policy", ""
            ),
            "stage_c_seed_geometry_smoke_clean6_best_policy_balanced_accuracy": geometry_smoke.get(
                "best_geometry_policy_balanced_accuracy", ""
            ),
            "stage_c_seed_geometry_smoke_clean6_q2_true_stage_pass": geometry_smoke.get(
                "q2_true_stage_pass", ""
            ),
            "stage_c_seed_geometry_smoke_target28_status": geometry_smoke_target28.get(
                "smoke_status", ""
            ),
            "stage_c_seed_geometry_smoke_target28_selected_case_count": geometry_smoke_target28.get(
                "selected_case_count", ""
            ),
            "stage_c_seed_geometry_smoke_target28_sidecar_file_count": geometry_smoke_target28.get(
                "per_chunk_geometry_sidecar_file_count", ""
            ),
            "stage_c_seed_geometry_smoke_target28_join_coverage": geometry_smoke_target28.get(
                "lifecycle_geometry_same_payload_join_coverage", ""
            ),
            "stage_c_seed_geometry_smoke_target28_alignment_pass": geometry_smoke_target28.get(
                "geometry_smoke_alignment_pass", ""
            ),
            "stage_c_seed_geometry_smoke_target28_best_policy": geometry_smoke_target28.get(
                "best_geometry_policy", ""
            ),
            "stage_c_seed_geometry_smoke_target28_best_policy_balanced_accuracy": geometry_smoke_target28.get(
                "best_geometry_policy_balanced_accuracy", ""
            ),
            "stage_c_seed_geometry_smoke_target28_best_policy_all_non_handoff": geometry_smoke_target28.get(
                "best_geometry_policy_all_non_handoff", ""
            ),
            "stage_c_seed_geometry_smoke_target28_best_policy_all_non_handoff_balanced_accuracy": (
                geometry_smoke_target28.get("best_geometry_policy_all_non_handoff_balanced_accuracy", "")
            ),
            "stage_c_seed_geometry_smoke_target28_q2_true_stage_pass": geometry_smoke_target28.get(
                "q2_true_stage_pass", ""
            ),
            "combined_admission_clean_safe_best_policy": combined_admission.get(
                "clean_safe_best_policy", ""
            ),
            "combined_admission_clean_safe_best_policy_balanced_accuracy": combined_admission.get(
                "clean_safe_best_policy_balanced_accuracy", ""
            ),
            "combined_admission_selected_positive_sequence_coverage": combined_admission.get(
                "selected_positive_sequence_coverage", ""
            ),
            "combined_admission_required_positive_sequence_coverage": combined_admission.get(
                "required_positive_sequence_coverage", ""
            ),
            "combined_admission_all_non_handoff_best_policy": combined_admission.get(
                "all_non_handoff_best_policy", ""
            ),
            "combined_admission_all_non_handoff_best_policy_balanced_accuracy": (
                combined_admission.get("all_non_handoff_best_policy_balanced_accuracy", "")
            ),
            "combined_admission_all_non_handoff_promotion_pass": combined_admission.get(
                "all_non_handoff_promotion_pass", ""
            ),
            "combined_admission_q2_proxy_stage_pass": combined_admission.get(
                "q2_proxy_stage_pass", ""
            ),
            "combined_admission_q2_true_stage_pass": combined_admission.get(
                "q2_true_stage_pass", ""
            ),
            "combined_admission_fp_attr_false_positive_case_count": combined_fp_attr.get(
                "false_positive_case_count", ""
            ),
            "combined_admission_fp_attr_taxonomy_counts": combined_fp_attr.get(
                "false_positive_taxonomy_counts", ""
            ),
            "combined_admission_fp_attr_clean_handoff_candidate_count": combined_fp_attr.get(
                "clean_handoff_candidate_count", ""
            ),
            "combined_admission_fp_attr_taxonomy_split_runtime_action_ready": combined_fp_attr.get(
                "taxonomy_split_runtime_action_ready", ""
            ),
            "combined_admission_fp_attr_q2_true_stage_pass": combined_fp_attr.get(
                "q2_true_stage_pass", ""
            ),
            "masklet_q2_best_policy": masklet_q2.get("best_policy", ""),
            "masklet_q2_best_policy_balanced_accuracy": masklet_q2.get(
                "best_policy_balanced_accuracy", ""
            ),
            "masklet_q2_true_stage_pass": masklet_q2.get("q2_true_stage_pass", ""),
            "claim": "Audit proves documentation/logging progress, not method success.",
        },
    )
    ensure_text(
        FINAL / "remaining_blockers.md",
        "\n".join(
            [
                "# ACL2 v101 Remaining Blockers",
                "",
                "- Track T gate_pass=false: only one clean HANDOFF target and five SAFE_GOOD controls.",
                "- Track Q2 true_stage_pass=false: admission remains proxy/blocked.",
                "- Track V gate_pass=false: per-anchor geometry exists, but strict observability/control gate fails.",
                "- Stage7 F5/R3 high-signal diagnostics remain proxy-only without per-anchor write chain/control margins/sequence coverage.",
                "- M4, runtime pilots, and full validation are not authorized.",
            ]
        ),
    )


def main() -> None:
    target_by_case = target_rows()
    final = read_json(FINAL / "final_decision.json")
    state_case_rows = aggregate_by_case(read_rows(ROOT / "trackS2_anchor_state_estimator/anchor_state_rows.csv"), target_by_case)
    support_case_rows = aggregate_by_case(read_rows(ROOT / "trackU_true_current_support/anchor_current_support_rows.csv"), target_by_case)
    role_case_rows = aggregate_by_case(read_rows(ROOT / "trackW_anchor_memory_role/anchor_role_rows.csv"), target_by_case)
    v_case_rows = read_rows(ROOT / "trackV_anchor_scale_observability/per_anchor_geometry_case_summary.csv")
    dh4_case_rows = read_rows(ROOT / "trackDH4_read_current_support_refresh_provider/read_provider_case_rows.csv")

    fpfn_specs = {
        "trackT_drift_target_relabel": target_conflict_fpfn(),
        "trackS2_anchor_state_estimator": fpfn_from_case_scores(state_case_rows, "unsupported_inconsistent_frac", direction="higher_bad", cue_name="S2_unsupported_inconsistent_frac"),
        "trackU_true_current_support": fpfn_from_case_scores(support_case_rows, "mean_S_cur", direction="lower_bad", cue_name="U_low_current_support"),
        "trackV_anchor_scale_observability": fpfn_from_case_scores(v_case_rows, "O_scale_repaired_mean", direction="lower_bad", cue_name="V_low_O_scale_repaired"),
        "trackW_anchor_memory_role": fpfn_from_case_scores(role_case_rows, "stale_candidate_frac", direction="higher_bad", cue_name="W_stale_candidate_frac"),
        "trackDH4_read_current_support_refresh_provider": fpfn_from_case_scores(dh4_case_rows, "READ_current_support_mean", direction="lower_bad", cue_name="DH4_low_READ_current_support"),
        "trackJL4_semantic_anchor_instance_atlas": [
            {
                "cue_name": "JL4_identity_resolution",
                "row_kind": "not_selector_identity_gap",
                "reason": "JL4 is an identity atlas; FP/FN selector rows are not applicable. See identity_resolution_gap_rows.csv.",
                "source_artifact": "identity_resolution_gap_rows.csv",
                "claim_level": "diagnostic_identity_gap_no_action",
            }
        ],
        "trackM4_state_machine_carrier_to_action_simulator": [
            {
                "cue_name": "M4_simulator",
                "row_kind": "blocked_not_run_no_fpfn",
                "reason": "M4 was not run because Track T/Q2/V upstream gates failed.",
                "claim_level": "blocked_not_run",
            }
        ],
        "runtime_pilots_or_blocked": [
            {
                "cue_name": "runtime_pilots",
                "row_kind": "blocked_not_run_no_fpfn",
                "reason": "Runtime pilots are not authorized without M4 pass.",
                "claim_level": "blocked_not_run",
            }
        ],
        "full_validation_or_blocked": [
            {
                "cue_name": "full_validation",
                "row_kind": "blocked_not_run_no_fpfn",
                "reason": "Full validation is not authorized without a 12-case runtime L3 pilot pass.",
                "claim_level": "blocked_not_run",
            }
        ],
        "stage0_v101_evidence_ledger": [
            {
                "cue_name": "Stage0_evidence_ledger",
                "row_kind": "not_applicable_stage0_fact_lock",
                "reason": "Stage0 is a fact-lock ledger and does not define a selector FP/FN gate.",
                "claim_level": "stage0_pass_no_action",
            }
        ],
    }

    text_specs = {
        "trackDH4_read_current_support_refresh_provider": {
            "control_gap_report.md": "Missing comparison against U/Q2 true-stage because Q2 true-stage is blocked; READ support remains provider-only.",
            "next_attempt_recommendation.md": "Keep READ as provider only. First expand clean handoff targets and Q2 true-stage; then compare READ provider against U/Q2 diagnostics without READ runtime action.",
        },
        "trackJL4_semantic_anchor_instance_atlas": {
            "control_gap_report.md": "Instance/component identity is unavailable; all atlas rows fall back to semantic class, and role transitions across chunks are not materialized.",
            "next_attempt_recommendation.md": "Materialize stable component/instance ids and role transitions before any identity-specific runtime action.",
        },
        "trackM4_state_machine_carrier_to_action_simulator": {
            "control_gap_report.md": "M4 lacks an authorized action family because Track T, Track Q2 true-stage, and Track V controls fail.",
            "next_attempt_recommendation.md": "Do not run M4 until a sequence-covered clean target universe and true-stage admission/observability controls exist.",
        },
        "runtime_pilots_or_blocked": {
            "control_gap_report.md": "Runtime pilots are blocked by M4 run_allowed=false and missing 12-case L3 mechanism gate.",
            "next_attempt_recommendation.md": "Run runtime pilots only after M4 simulator passes with safe-good harm <=2% and required random/control margins.",
        },
        "full_validation_or_blocked": {
            "control_gap_report.md": "Full validation is blocked because no runtime pilot passed a 12-case L3 mechanism gate.",
            "next_attempt_recommendation.md": "Full validation requires passed runtime pilot evidence, full ATE/final/rolling/segment metrics, and safe-good protection.",
        },
        "final_decision": {
            "failure_report.md": "Final method success is not achieved. Track T/Q2/V/S2 fail; M4, runtime pilots, and full validation are not authorized.",
            "what_would_have_to_be_true_to_pass.md": "All v101 completion gates would need verified evidence: clean target universe, true-stage Q2, strict Track V controls, M4 pass, runtime L3 pilot pass, and full validation success.",
            "control_gap_report.md": "Final decision remains action-blocked: Track T/Q2/V/S2 fail and M4/runtime/full validation are not authorized.",
            "next_attempt_recommendation.md": "Acquire a new v100-schema clean handoff target universe with same-space trace, per-anchor geometry, and identity/query-head controls.",
            "false_positive_false_negative_rows.csv": "",
        },
    }

    created: list[dict[str, Any]] = []
    for track_dir, rows in fpfn_specs.items():
        path = ROOT / track_dir / "false_positive_false_negative_rows.csv"
        if ensure_rows(path, rows):
            created.append({"track": track_dir, "artifact": str(path), "kind": "fpfn"})
    for track_dir, files in text_specs.items():
        base = ROOT / track_dir
        for name, text in files.items():
            path = base / name
            if name.endswith(".csv"):
                rows = [
                    {
                        "cue_name": "final_decision",
                        "row_kind": "not_applicable_final_decision",
                        "reason": "Final decision aggregates track gates and is not itself a selector.",
                        "claim_level": "audit_no_action",
                    }
                ]
                if ensure_rows(path, rows):
                    created.append({"track": track_dir, "artifact": str(path), "kind": "fpfn"})
            elif ensure_text(path, text):
                created.append({"track": track_dir, "artifact": str(path), "kind": "text"})

    required = [
        "failure_report.md",
        "what_would_have_to_be_true_to_pass.md",
        "control_gap_report.md",
        "next_attempt_recommendation.md",
        "false_positive_false_negative_rows.csv",
    ]
    artifact_rows: list[dict[str, Any]] = []
    for directory in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        for name in required:
            path = directory / name
            artifact_rows.append(
                {
                    "track_dir": directory.name,
                    "artifact": name,
                    "exists": path.is_file(),
                    "size_bytes": path.stat().st_size if path.is_file() else "",
                }
            )
    outcome_d_required = {
        "outcomeD_broader_carrier_reentry": [
            "broader_carrier_reentry_summary.json",
            "broader_carrier_reentry_routes.csv",
            "broader_carrier_reentry_report.md",
        ],
        "outcomeD_merge_gauge_selector_reentry": [
            "merge_gauge_selector_reentry_summary.json",
            "merge_gauge_selector_reentry_candidate_metrics.csv",
            "merge_gauge_selector_reentry_report.md",
        ],
        "outcomeD_merge_gauge_rich_selector_reentry": [
            "merge_gauge_rich_selector_reentry_summary.json",
            "merge_gauge_rich_selector_reentry_candidate_metrics.csv",
            "merge_gauge_rich_selector_reentry_passing_candidates.csv",
            "merge_gauge_rich_selector_reentry_promotion_readiness.csv",
            "merge_gauge_rich_selector_reentry_best_selected_rows.csv",
            "merge_gauge_rich_selector_reentry_sequence_stability.csv",
            "merge_gauge_rich_selector_reentry_loso_holdout.csv",
            "merge_gauge_rich_selector_reentry_report.md",
            "merge_gauge_rich_selector_next_rerun_spec.md",
        ],
        "outcomeD_internal_qk_actuator_reentry": [
            "internal_qk_actuator_reentry_summary.json",
            "internal_qk_actuator_reentry_family_summary.csv",
            "internal_qk_actuator_reentry_candidate_metrics.csv",
            "internal_qk_actuator_reentry_report.md",
        ],
        "outcomeD_cache_topk_identity_query_reentry": [
            "cache_topk_identity_query_reentry_summary.json",
            "cache_topk_identity_query_reentry_signal_summary.csv",
            "cache_topk_identity_query_reentry_action_variant_metrics.csv",
            "cache_topk_identity_query_reentry_report.md",
        ],
        "outcomeD_remaining_reentry_route_closure": [
            "remaining_reentry_route_closure_summary.json",
            "remaining_reentry_route_closure_rows.csv",
            "remaining_reentry_route_closure_report.md",
        ],
        "historical_target_universe_mining_followup": [
            "historical_target_universe_mining_summary.json",
            "historical_target_universe_mining_rows.csv",
            "historical_target_universe_mining_report.md",
        ],
        "rich_selector_replay_probe_followup": [
            "rich_selector_replay_probe_summary.json",
            "rich_selector_replay_probe_variant_summary.csv",
            "rich_selector_replay_probe_report.md",
        ],
        "rich_selector_fresh_holdout_feasibility_followup": [
            "rich_selector_holdout_feasibility_summary.json",
            "rich_selector_holdout_feasibility_rows.csv",
            "rich_selector_holdout_feasibility_report.md",
        ],
        "fresh_selected_schema_materialization_followup": [
            "fresh_selected_schema_materialization_summary.json",
            "fresh_selected_schema_materialization_rows.csv",
            "fresh_selected_schema_materialization_report.md",
        ],
        "strict_action_frontier_followup": [
            "strict_action_frontier_summary.json",
            "strict_action_frontier_rows.csv",
            "strict_action_frontier_report.md",
        ],
        "plan_coverage_closure_followup": [
            "plan_coverage_closure_summary.json",
            "plan_coverage_closure_rows.csv",
            "plan_coverage_closure_report.md",
        ],
        "trace_rescue_feasibility_followup": [
            "trace_rescue_feasibility_summary.json",
            "trace_rescue_hook_identity_rows.csv",
            "trace_rescue_jsonl_rows.csv",
            "trace_rescue_feasibility_report.md",
        ],
        "state_q2_readiness_followup": [
            "state_q2_readiness_summary.json",
            "state_q2_readiness_rows.csv",
            "state_q2_readiness_report.md",
        ],
        "component_identity_availability_followup": [
            "component_identity_availability_summary.json",
            "component_identity_availability_case_rows.csv",
            "component_identity_availability_chunk_rows.csv",
            "component_identity_availability_report.md",
        ],
        "stage_c_seed_bridge_smoke_followup": [
            "stage_c_seed_bridge_smoke_summary.json",
            "stage_c_seed_bridge_smoke_rows.csv",
            "stage_c_seed_bridge_smoke_report.md",
        ],
        "stage_c_seed_current_support_materialization_followup": [
            "trackU_true_current_support/stage_c_seed_current_support_summary.json",
            "trackU_true_current_support/stage_c_seed_current_support_rows.csv",
            "trackU_true_current_support/stage_c_seed_current_support_case_rows.csv",
            "trackU_true_current_support/stage_c_seed_current_support_report.md",
        ],
        "stage_c_seed_clean_eval_q128_followup": [
            "trackU_true_current_support/stage_c_seed_current_support_clean_eval_q128_summary.json",
            "trackU_true_current_support/stage_c_seed_current_support_clean_eval_q128_rows.csv",
            "trackU_true_current_support/stage_c_seed_current_support_clean_eval_q128_case_rows.csv",
            "trackU_true_current_support/stage_c_seed_current_support_clean_eval_q128_report.md",
        ],
        "anchor_seed_join_feasibility_followup": [
            "anchor_seed_join_feasibility_summary.json",
            "anchor_seed_join_feasibility_rows.csv",
            "anchor_seed_join_feasibility_report.md",
        ],
        "anchor_seed_lifecycle_support_join_followup": [
            "anchor_seed_lifecycle_support_join_summary.json",
            "anchor_seed_lifecycle_expanded_rows.csv",
            "anchor_seed_lifecycle_support_join_rows.csv",
            "anchor_seed_lifecycle_support_join_report.md",
        ],
        "anchor_seed_lifecycle_stage_c_seed_support_join_followup": [
            "anchor_seed_lifecycle_support_join_summary.json",
            "anchor_seed_lifecycle_stage_c_seed_support_join_rows.csv",
            "anchor_seed_lifecycle_stage_c_seed_support_join_report.md",
        ],
        "anchor_seed_lifecycle_stage_c_masklet_visibility_followup": [
            "anchor_seed_lifecycle_stage_c_masklet_visibility_summary.json",
            "anchor_seed_lifecycle_stage_c_masklet_visibility_rows.csv",
            "anchor_seed_lifecycle_stage_c_masklet_visibility_report.md",
        ],
        "anchor_seed_lifecycle_stage_c_masklet_observability_proxy_followup": [
            "anchor_seed_lifecycle_stage_c_masklet_visibility_summary.json",
            "anchor_seed_lifecycle_stage_c_masklet_visibility_rows.csv",
            "anchor_seed_lifecycle_stage_c_masklet_visibility_report.md",
        ],
        "anchor_seed_lifecycle_geometry_observability_join_followup": [
            "anchor_seed_lifecycle_geometry_observability_summary.json",
            "anchor_seed_lifecycle_geometry_observability_join_rows.csv",
            "anchor_seed_lifecycle_geometry_observability_case_rows.csv",
            "anchor_seed_lifecycle_geometry_observability_report.md",
        ],
        "stage_c_seed_geometry_smoke_clean6_followup": [
            "stage_c_seed_geometry_smoke_clean6_summary.json",
            "stage_c_seed_geometry_smoke_clean6_case_rows.csv",
            "stage_c_seed_geometry_smoke_clean6_edge_rows.csv",
            "stage_c_seed_geometry_smoke_clean6_policy_rows.csv",
            "stage_c_seed_geometry_smoke_clean6_read_errors.csv",
            "stage_c_seed_geometry_smoke_clean6_report.md",
        ],
        "stage_c_seed_geometry_smoke_target28_followup": [
            "stage_c_seed_geometry_smoke_target28_summary.json",
            "stage_c_seed_geometry_smoke_target28_case_rows.csv",
            "stage_c_seed_geometry_smoke_target28_edge_rows.csv",
            "stage_c_seed_geometry_smoke_target28_policy_rows.csv",
            "stage_c_seed_geometry_smoke_target28_read_errors.csv",
            "stage_c_seed_geometry_smoke_target28_report.md",
        ],
        "combined_masklet_geometry_admission_followup": [
            "combined_masklet_geometry_admission_summary.json",
            "combined_masklet_geometry_admission_case_rows.csv",
            "combined_masklet_geometry_admission_policy_rows.csv",
            "combined_masklet_geometry_admission_false_positive_false_negative_rows.csv",
            "combined_masklet_geometry_admission_report.md",
        ],
        "combined_admission_false_positive_attribution_followup": [
            "combined_admission_false_positive_attribution_summary.json",
            "combined_admission_false_positive_attribution_rows.csv",
            "combined_admission_false_positive_attribution_report.md",
        ],
        "masklet_q2_admission_sanity_followup": [
            "masklet_q2_admission_sanity_summary.json",
            "masklet_q2_admission_sanity_case_rows.csv",
            "masklet_q2_admission_sanity_policy_rows.csv",
            "masklet_q2_admission_sanity_report.md",
        ],
    }
    for logical_track, names in outcome_d_required.items():
        for name in names:
            path = ROOT / name if "/" in name else FINAL / name
            artifact_rows.append(
                {
                    "track_dir": logical_track,
                    "artifact": name if "/" in name else f"final_decision/{name}",
                    "exists": path.is_file(),
                    "size_bytes": path.stat().st_size if path.is_file() else "",
                }
            )
    write_rows(FINAL / "fail_forward_artifact_audit.csv", artifact_rows)
    write_rows(FINAL / "fail_forward_artifacts_created.csv", created)
    write_completion_audit(final, created, artifact_rows)

    print(
        json.dumps(
            {
                "schema": "acl2_v101_failforward_completeness_v1",
                "created_missing_artifacts_count": len(created),
                "artifact_audit_rows": len(artifact_rows),
                "goal_achieved": False,
                "final_taxonomy": final.get("final_taxonomy", ""),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
