#!/usr/bin/env python3
"""Close the v101 plan coverage audit without changing gate outcomes.

This script reads existing v101 artifacts and writes a reviewer-facing coverage
table that maps the written plan stages to the available evidence, fail-forward
artifacts, and downstream authorization state.  It is intentionally read-only
with respect to experiment metrics: missing coverage is reported, not filled
with synthetic numbers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"

BASE_FAIL_FORWARD = [
    "failure_report.md",
    "what_would_have_to_be_true_to_pass.md",
    "control_gap_report.md",
    "next_attempt_recommendation.md",
    "false_positive_false_negative_rows.csv",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}"}
    return payload if isinstance(payload, dict) else {"value": payload}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: stringify(row.get(key, "")) for key in keys})


def stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_clean(value), ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [json_clean(v) for v in value]
    return value


def exists_all(base: Path, names: list[str]) -> tuple[bool, list[str]]:
    missing = [name for name in names if not (base / name).is_file()]
    return not missing, missing


def bool_from_json(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return ""


def gate_label(data: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in [
        "status",
        "gate_pass",
        "diagnostic_gate_pass",
        "proxy_stage_pass",
        "true_stage_pass",
        "run_allowed",
        "trackM4_run_allowed",
        "runtime_action_allowed",
        "full_validation_run",
        "full_method_success",
        "final_taxonomy",
    ]:
        if key in data:
            pieces.append(f"{key}={data[key]}")
    if not pieces:
        return "summary_missing_or_no_gate_fields"
    return "; ".join(pieces)


def plan_stage_specs() -> list[dict[str, Any]]:
    return [
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage0",
            "stage_name": "v100 evidence freeze and v101 ledger",
            "directory": "stage0_v101_evidence_ledger",
            "summary": "summary.json",
            "required": [
                "summary.json",
                "v100_fact_lock.md",
                "blocked_direction_list.md",
                "reusable_signal_list.md",
                "do_not_repeat_list.md",
                "case_artifact_manifest.csv",
            ],
            "expected_authorization": "diagnostic_ledger_only",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage1",
            "stage_name": "Track T drift target relabel and label-L3 hygiene",
            "directory": "trackT_drift_target_relabel",
            "summary": "target_taxonomy_summary.json",
            "required": [
                "target_universe_v101.csv",
                "target_taxonomy_summary.json",
                "per_sequence_target_distribution.csv",
                "safe_good_controls.csv",
                "ambiguous_cases.csv",
                "label_l3_conflict_rows.csv",
            ],
            "expected_authorization": "must_pass_before_action",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage2",
            "stage_name": "Track U true current-support materialization",
            "directory": "trackU_true_current_support",
            "summary": "current_support_summary.json",
            "required": [
                "current_support_summary.json",
                "anchor_current_support_rows.csv",
                "support_source_coverage.csv",
                "per_anchor_geometry_support_repair_summary.json",
            ],
            "expected_authorization": "materialization_only_no_action",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage3",
            "stage_name": "Track V true or stronger scale observability",
            "directory": "trackV_anchor_scale_observability",
            "summary": "observability_summary.json",
            "required": [
                "observability_summary.json",
                "anchor_observability_rows.csv",
                "per_anchor_geometry_case_summary.csv",
                "per_anchor_geometry_observability_summary.json",
            ],
            "expected_authorization": "must_pass_or_mark_proxy_blocked_before_action",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage4",
            "stage_name": "Track W semantic anchor memory role classifier",
            "directory": "trackW_anchor_memory_role",
            "summary": "role_summary.json",
            "required": [
                "role_summary.json",
                "anchor_role_rows.csv",
                "role_transition_rows.csv",
            ],
            "expected_authorization": "role_diagnostic_only",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage5",
            "stage_name": "Track S2 same-space state estimator with uncertainty",
            "directory": "trackS2_anchor_state_estimator",
            "summary": "state_estimator_summary.json",
            "required": [
                "state_estimator_summary.json",
                "anchor_state_rows.csv",
                "uncertainty_rows.csv",
                "gain_rows.csv",
            ],
            "expected_authorization": "must_pass_with_uncertainty_before_action",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage6",
            "stage_name": "Track Q2 chunk-level scale-update admission gate",
            "directory": "trackQ2_scale_update_admission",
            "summary": "Q2_summary.json",
            "required": [
                "Q2_summary.json",
                "admission_rows.csv",
                "admission_metric_summary.csv",
                "admission_false_positive_negative_report.md",
            ],
            "expected_authorization": "true_stage_pass_required_for_M4",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage7a",
            "stage_name": "Track N3 identity graph under cleaned targets",
            "directory": "trackN3_anchor_identity_graph_cleaned_targets",
            "summary": "N3_summary.json",
            "required": [
                "N3_summary.json",
                "anchor_graph_pattern_rows.csv",
                "metric_summary.csv",
                "blocked_summary.json",
            ],
            "expected_authorization": "diagnostic_blocked_without_downstream_M4",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage7b",
            "stage_name": "Track C5 identity-conditioned latent gauge with support",
            "directory": "trackC5_identity_latent_gauge_with_support",
            "summary": "C5_summary.json",
            "required": [
                "C5_summary.json",
                "latent_support_interaction_metrics.csv",
                "metric_summary.csv",
                "blocked_summary.json",
            ],
            "expected_authorization": "diagnostic_blocked_without_downstream_M4",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage7c",
            "stage_name": "Track F5 TTT write-to-use state chain",
            "directory": "trackF5_ttt_write_to_use_state_chain",
            "summary": "F5_summary.json",
            "required": [
                "F5_summary.json",
                "write_to_use_proxy_rows.csv",
                "write_to_use_materialization_audit.csv",
                "blocked_summary.json",
            ],
            "expected_authorization": "diagnostic_blocked_without_write_to_use_chain",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage7d",
            "stage_name": "Track R3 query/head/anchor edge audit with true support",
            "directory": "trackR3_query_head_anchor_edge_audit_true_support",
            "summary": "R3_summary.json",
            "required": [
                "R3_summary.json",
                "support_conditioned_anchor_edge_case_rows.csv",
                "edge_metric_summary.csv",
                "blocked_summary.json",
            ],
            "expected_authorization": "diagnostic_blocked_without_M4",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage8",
            "stage_name": "Track M4 state-machine carrier-to-action simulator",
            "directory": "trackM4_state_machine_carrier_to_action_simulator",
            "summary": "blocked_summary.json",
            "required": [
                "blocked_summary.json",
                "not_run_manifest.csv",
                "gate_checks.csv",
            ],
            "expected_authorization": "blocked_until_T_Q2_V_S2_prereqs_pass",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage9a",
            "stage_name": "Runtime pilots only after M4 pass",
            "directory": "runtime_pilots_or_blocked",
            "summary": "blocked_summary.json",
            "required": [
                "blocked_summary.json",
                "not_run_manifest.csv",
                "gate_checks.csv",
            ],
            "expected_authorization": "blocked_until_M4_pass",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage9b",
            "stage_name": "Track DH4 READ current-support refresh provider",
            "directory": "trackDH4_read_current_support_refresh_provider",
            "summary": "DH4_summary.json",
            "required": [
                "DH4_summary.json",
                "read_provider_case_rows.csv",
                "read_provider_anchor_rows.csv",
                "provider_report.md",
                "blocked_summary.json",
            ],
            "expected_authorization": "provider_only_no_runtime_action",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage9c",
            "stage_name": "Track JL4 semantic anchor instance atlas",
            "directory": "trackJL4_semantic_anchor_instance_atlas",
            "summary": "JL4_summary.json",
            "required": [
                "JL4_summary.json",
                "anchor_instance_atlas.csv",
                "identity_resolution_gap_rows.csv",
                "blocked_summary.json",
            ],
            "expected_authorization": "identity_gap_blocks_action",
        },
        {
            "coverage_group": "plan_stage",
            "stage_id": "Stage10",
            "stage_name": "Stage7 full validation only after L3 pilot passes",
            "directory": "full_validation_or_blocked",
            "summary": "blocked_summary.json",
            "required": [
                "blocked_summary.json",
                "not_run_manifest.csv",
                "gate_checks.csv",
            ],
            "expected_authorization": "blocked_until_runtime_L3_pilot_pass",
        },
    ]


def repair_route_specs() -> list[dict[str, Any]]:
    base = "final_decision"
    return [
        {
            "coverage_group": "repair_route",
            "stage_id": "OutcomeD0",
            "stage_name": "New v100-schema universe feasibility",
            "directory": base,
            "summary": "new_v100_schema_universe_feasibility_summary.json",
            "required": [
                "new_v100_schema_universe_feasibility_summary.json",
                "new_v100_schema_universe_feasibility_rows.csv",
                "new_v100_schema_universe_feasibility_report.md",
            ],
            "expected_authorization": "diagnostic_clean_universe_search",
        },
        {
            "coverage_group": "repair_route",
            "stage_id": "OutcomeD1",
            "stage_name": "Broader carrier route reentry audit",
            "directory": base,
            "summary": "broader_carrier_reentry_summary.json",
            "required": [
                "broader_carrier_reentry_summary.json",
                "broader_carrier_reentry_routes.csv",
                "broader_carrier_reentry_report.md",
            ],
            "expected_authorization": "broader_routes_checked_no_action_allowed",
        },
        {
            "coverage_group": "repair_route",
            "stage_id": "OutcomeD2",
            "stage_name": "Merge-gauge selector reentry audit",
            "directory": base,
            "summary": "merge_gauge_selector_reentry_summary.json",
            "required": [
                "merge_gauge_selector_reentry_summary.json",
                "merge_gauge_selector_reentry_candidate_metrics.csv",
                "merge_gauge_selector_reentry_report.md",
            ],
            "expected_authorization": "selector_screen_no_passing_policy",
        },
        {
            "coverage_group": "repair_route",
            "stage_id": "OutcomeD3",
            "stage_name": "Rich merge-gauge selector screen and stability audit",
            "directory": base,
            "summary": "merge_gauge_rich_selector_reentry_summary.json",
            "required": [
                "merge_gauge_rich_selector_reentry_summary.json",
                "merge_gauge_rich_selector_reentry_candidate_metrics.csv",
                "merge_gauge_rich_selector_reentry_passing_candidates.csv",
                "merge_gauge_rich_selector_reentry_loso_holdout.csv",
                "merge_gauge_rich_selector_reentry_report.md",
            ],
            "expected_authorization": "rich_selector_unstable_no_action",
        },
        {
            "coverage_group": "repair_route",
            "stage_id": "OutcomeD4",
            "stage_name": "Internal QK actuator reentry audit",
            "directory": base,
            "summary": "internal_qk_actuator_reentry_summary.json",
            "required": [
                "internal_qk_actuator_reentry_summary.json",
                "internal_qk_actuator_reentry_family_summary.csv",
                "internal_qk_actuator_reentry_candidate_metrics.csv",
                "internal_qk_actuator_reentry_report.md",
            ],
            "expected_authorization": "internal_QK_no_action_surface",
        },
        {
            "coverage_group": "repair_route",
            "stage_id": "OutcomeD5",
            "stage_name": "Cache/top-k identity-query reentry audit",
            "directory": base,
            "summary": "cache_topk_identity_query_reentry_summary.json",
            "required": [
                "cache_topk_identity_query_reentry_summary.json",
                "cache_topk_identity_query_reentry_signal_summary.csv",
                "cache_topk_identity_query_reentry_action_variant_metrics.csv",
                "cache_topk_identity_query_reentry_report.md",
            ],
            "expected_authorization": "cache_topk_no_action_variant",
        },
        {
            "coverage_group": "repair_route",
            "stage_id": "OutcomeD6",
            "stage_name": "Remaining reentry route closure",
            "directory": base,
            "summary": "remaining_reentry_route_closure_summary.json",
            "required": [
                "remaining_reentry_route_closure_summary.json",
                "remaining_reentry_route_closure_rows.csv",
                "remaining_reentry_route_closure_report.md",
            ],
            "expected_authorization": "remaining_routes_closed_no_action_allowed",
        },
        {
            "coverage_group": "repair_route",
            "stage_id": "OutcomeD7",
            "stage_name": "Historical target universe mining",
            "directory": base,
            "summary": "historical_target_universe_mining_summary.json",
            "required": [
                "historical_target_universe_mining_summary.json",
                "historical_target_universe_mining_rows.csv",
                "historical_target_universe_mining_report.md",
            ],
            "expected_authorization": "no_new_clean_action_universe",
        },
        {
            "coverage_group": "repair_route",
            "stage_id": "OutcomeD8",
            "stage_name": "Rich selector replay probe",
            "directory": base,
            "summary": "rich_selector_replay_probe_summary.json",
            "required": [
                "rich_selector_replay_probe_summary.json",
                "rich_selector_replay_probe_variant_summary.csv",
                "rich_selector_replay_probe_report.md",
            ],
            "expected_authorization": "probe_completed_no_candidate_beats_control",
        },
        {
            "coverage_group": "repair_route",
            "stage_id": "OutcomeD9",
            "stage_name": "Rich selector fresh holdout feasibility",
            "directory": base,
            "summary": "rich_selector_holdout_feasibility_summary.json",
            "required": [
                "rich_selector_holdout_feasibility_summary.json",
                "rich_selector_holdout_feasibility_rows.csv",
                "rich_selector_holdout_feasibility_report.md",
            ],
            "expected_authorization": "fresh_holdout_not_labelled_bad_good",
        },
        {
            "coverage_group": "repair_route",
            "stage_id": "OutcomeD10",
            "stage_name": "Fresh selected schema materialization",
            "directory": base,
            "summary": "fresh_selected_schema_materialization_summary.json",
            "required": [
                "fresh_selected_schema_materialization_summary.json",
                "fresh_selected_schema_materialization_rows.csv",
                "fresh_selected_schema_materialization_report.md",
            ],
            "expected_authorization": "fresh_selected_rows_not_action_materializable",
        },
        {
            "coverage_group": "repair_route",
            "stage_id": "OutcomeD11",
            "stage_name": "Strict action frontier common blocker audit",
            "directory": base,
            "summary": "strict_action_frontier_summary.json",
            "required": [
                "strict_action_frontier_summary.json",
                "strict_action_frontier_rows.csv",
                "strict_action_frontier_report.md",
            ],
            "expected_authorization": "strict_frontier_no_ready_candidate",
        },
    ]


def build_row(spec: dict[str, Any]) -> dict[str, Any]:
    base = ROOT / str(spec["directory"])
    summary_path = base / str(spec["summary"])
    summary = read_json(summary_path)
    required_present, missing_required = exists_all(base, list(spec["required"]))
    ff_present = ""
    missing_ff: list[str] = []
    if spec["coverage_group"] == "plan_stage":
        ff_present, missing_ff = exists_all(base, BASE_FAIL_FORWARD)
    runtime_action_allowed = bool_from_json(summary, "runtime_action_allowed")
    return {
        "coverage_group": spec["coverage_group"],
        "stage_id": spec["stage_id"],
        "stage_name": spec["stage_name"],
        "directory": str(spec["directory"]),
        "summary_artifact": str(summary_path),
        "summary_exists": summary_path.is_file(),
        "required_artifact_count": len(spec["required"]),
        "required_artifacts_present": required_present,
        "missing_required_artifacts": missing_required,
        "fail_forward_artifacts_checked": spec["coverage_group"] == "plan_stage",
        "fail_forward_artifacts_present": ff_present,
        "missing_fail_forward_artifacts": missing_ff,
        "gate_pass": bool_from_json(summary, "gate_pass"),
        "proxy_stage_pass": bool_from_json(summary, "proxy_stage_pass"),
        "true_stage_pass": bool_from_json(summary, "true_stage_pass"),
        "run_allowed": bool_from_json(summary, "run_allowed", "trackM4_run_allowed"),
        "runtime_action_allowed": runtime_action_allowed,
        "full_validation_run": bool_from_json(summary, "full_validation_run"),
        "full_method_success": bool_from_json(summary, "full_method_success"),
        "status": summary.get("status", ""),
        "final_taxonomy": summary.get("final_taxonomy", ""),
        "gate_status": gate_label(summary),
        "expected_authorization": spec["expected_authorization"],
        "claim_level": "coverage_audit_no_new_metric_no_action",
    }


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    failing_required = [row for row in rows if row["required_artifacts_present"] is not True]
    failing_ff = [
        row
        for row in rows
        if row["coverage_group"] == "plan_stage" and row["fail_forward_artifacts_present"] is not True
    ]
    action_allowed = [row for row in rows if row.get("runtime_action_allowed") is True or row.get("run_allowed") is True]
    lines = [
        "# ACL2 v101 Plan Coverage Closure",
        "",
        "This report audits coverage of the written v101 plan and follow-up repair routes.",
        "It does not claim method success and does not authorize runtime action.",
        "",
        "## Summary",
        "",
        f"- plan_artifact_coverage_pass: {summary['plan_artifact_coverage_pass']}",
        f"- fail_forward_coverage_pass: {summary['fail_forward_coverage_pass']}",
        f"- repair_route_coverage_pass: {summary['repair_route_coverage_pass']}",
        f"- method_goal_achieved: {summary['method_goal_achieved']}",
        f"- downstream_action_authorized: {summary['downstream_action_authorized']}",
        f"- runtime_action_allowed: {summary['runtime_action_allowed']}",
        f"- final_taxonomy: {summary['final_taxonomy']}",
        f"- missing_required_row_count: {len(failing_required)}",
        f"- missing_fail_forward_row_count: {len(failing_ff)}",
        f"- action_allowed_row_count: {len(action_allowed)}",
        "",
        "## Stage Gate Evidence",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row['stage_id']} {row['stage_name']}: "
            f"required_present={row['required_artifacts_present']}; "
            f"fail_forward_present={row['fail_forward_artifacts_present']}; "
            f"{row['gate_status']}"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Coverage is closed only as an audit/documentation state. The method remains No-Go because "
            "Track T/Q2/V/S2 and downstream M4/runtime/full gates do not authorize action.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    rows = [build_row(spec) for spec in plan_stage_specs()]
    rows.extend(build_row(spec) for spec in repair_route_specs())
    final = read_json(FINAL / "final_decision.json")
    completion = read_json(FINAL / "completion_audit_summary.json")

    plan_rows = [row for row in rows if row["coverage_group"] == "plan_stage"]
    repair_rows = [row for row in rows if row["coverage_group"] == "repair_route"]
    missing_required_count = sum(1 for row in rows if row["required_artifacts_present"] is not True)
    missing_ff_count = sum(
        1
        for row in plan_rows
        if row["fail_forward_artifacts_present"] is not True
    )
    runtime_action_allowed_count = sum(1 for row in rows if row.get("runtime_action_allowed") is True)
    run_allowed_count = sum(1 for row in rows if row.get("run_allowed") is True)

    track_passes = {
        "stage0_gate_pass": final.get("stage0_gate_pass"),
        "trackT_gate_pass": final.get("trackT_gate_pass"),
        "trackU_gate_pass": final.get("trackU_gate_pass"),
        "trackV_gate_pass": final.get("trackV_gate_pass"),
        "trackW_gate_pass": final.get("trackW_gate_pass"),
        "trackS2_gate_pass": final.get("trackS2_gate_pass"),
        "trackQ2_proxy_stage_pass": final.get("trackQ2_proxy_stage_pass"),
        "trackQ2_true_stage_pass": final.get("trackQ2_true_stage_pass"),
        "trackM4_run_allowed": final.get("trackM4_run_allowed"),
    }
    outcome_d_semantic_action_closed = (
        final.get("trackT_gate_pass") is False
        and final.get("trackV_gate_pass") is False
        and final.get("trackS2_gate_pass") is False
        and final.get("trackQ2_true_stage_pass") is False
        and final.get("trackM4_run_allowed") is False
        and runtime_action_allowed_count == 0
    )
    summary = {
        "schema": "acl2_v101_plan_coverage_closure_v1",
        "plan_stage_row_count": len(plan_rows),
        "repair_route_row_count": len(repair_rows),
        "coverage_row_count": len(rows),
        "missing_required_row_count": missing_required_count,
        "missing_fail_forward_row_count": missing_ff_count,
        "runtime_action_allowed_row_count": runtime_action_allowed_count,
        "run_allowed_row_count": run_allowed_count,
        "plan_artifact_coverage_pass": all(row["required_artifacts_present"] is True for row in plan_rows),
        "fail_forward_coverage_pass": all(row["fail_forward_artifacts_present"] is True for row in plan_rows),
        "repair_route_coverage_pass": all(row["required_artifacts_present"] is True for row in repair_rows),
        "method_goal_achieved": False,
        "downstream_action_authorized": False,
        "runtime_action_allowed": final.get("runtime_action_allowed", False),
        "trackM4_run_allowed": final.get("trackM4_run_allowed", False),
        "full_validation_run": final.get("full_validation_run", False),
        "full_method_success": final.get("full_method_success", False),
        "final_taxonomy": final.get("final_taxonomy", ""),
        "primary_blocker": final.get("primary_blocker", ""),
        "track_passes": track_passes,
        "outcome_a_active": False,
        "outcome_b_active": False,
        "outcome_c_active": False,
        "outcome_d_semantic_anchor_action_closed_under_existing_evidence": outcome_d_semantic_action_closed,
        "outcome_e_read_provider_only": read_json(ROOT / "trackDH4_read_current_support_refresh_provider/DH4_summary.json").get("runtime_action_allowed") is False,
        "completion_failed_requirement_count_before_this_audit": completion.get("failed_requirement_count", ""),
        "claim": "Plan coverage is closed for audit; v101 method success is not achieved and no runtime/full action is authorized.",
    }
    summary["plan_coverage_closure_pass"] = (
        summary["plan_artifact_coverage_pass"] is True
        and summary["fail_forward_coverage_pass"] is True
        and summary["repair_route_coverage_pass"] is True
        and summary["runtime_action_allowed"] is False
        and summary["trackM4_run_allowed"] is False
        and summary["full_validation_run"] is False
        and summary["full_method_success"] is False
    )

    write_rows(FINAL / "plan_coverage_closure_rows.csv", rows)
    write_json(FINAL / "plan_coverage_closure_summary.json", summary)
    write_report(FINAL / "plan_coverage_closure_report.md", summary, rows)
    print(json.dumps(json_clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
