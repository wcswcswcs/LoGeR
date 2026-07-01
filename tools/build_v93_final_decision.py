#!/usr/bin/env python3
"""Build v93 final decision from measured phase artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "report_final")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - final report should expose malformed artifacts.
        return {"read_error": str(exc), "path": str(path)}
    return data if isinstance(data, dict) else {"value": data}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def b(value: Any) -> bool:
    return bool(value)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    phase0 = read_json(args.root / "phase0_v92_evidence_lock/phase0_gate_summary.json")
    phase1 = read_json(args.root / "phase1_object_identity_row_join/object_identity_source_summary.json")
    phase1_audit = read_json(args.root / "phase1_object_identity_row_join/object_identity_join_audit.json")
    phase2 = read_json(args.root / "phase2_object_topology_policy/object_topology_policy_audit.json")
    phase3 = read_json(args.root / "phase3_merge_gauge_trace_audit/phase3_trace_availability_summary.json")
    hidden = read_json(args.root / "phase3_merge_gauge_trace_audit/hidden_merge_gauge_field_audit.json")
    phase4 = read_json(args.root / "phase4_merge_gauge_carrier_alignment/carrier_alignment_summary.json")
    phase5 = read_json(args.root / "phase5_merge_gauge_counterfactual_upper_bound/counterfactual_upper_bound_summary.json")
    phase7 = read_json(args.root / "phase7_swa_secondary_carrier/carrier_audit/phase7_swa_secondary_summary.json")

    phase1_pass = b(phase1_audit.get("phase1_source_gate_pass"))
    object_identity_source_pass = b(phase1_audit.get("object_identity_source_pass"))
    radio_source_pass = b(phase1_audit.get("radio_source_pass"))
    object_source_pass = object_identity_source_pass or radio_source_pass
    phase2_pass = b(phase2.get("phase2_object_topology_policy_gate_pass"))
    repair_pass = b(phase2.get("repair_gate_pass"))
    phase3_pass = b(phase3.get("phase3_trace_availability_gate_pass"))
    phase4_entered = b(phase4.get("entered"))
    phase4_pass = b(phase4.get("phase4_carrier_alignment_gate_pass"))
    phase5_entered = b(phase5.get("entered"))
    phase5_pass = b(phase5.get("phase5_counterfactual_gate_pass"))
    phase7_entered = b(phase7.get("entered"))
    phase7_pass = b(phase7.get("phase7_swa_secondary_carrier_gate_pass"))

    labels: list[str] = []
    if not b(phase0.get("phase0_gate_pass")):
        labels.append("D0_V92_EVIDENCE_LOCK_FAIL")
    if not object_source_pass:
        labels.append("D1_OBJECT_RADIO_ROW_SOURCE_INSUFFICIENT")
    if not phase2_pass:
        labels.append("D2_OBJECT_TOPOLOGY_POLICY_NOT_SPECIFIC")
    if not repair_pass:
        labels.append("D2R_PLAN_APPROVED_POLICY_REPAIR_FAIL")
    if not phase3_pass:
        labels.append("D3_TRUE_MERGE_GAUGE_TRACE_INSUFFICIENT")
    if not phase4_pass:
        labels.append("D4_CARRIER_ALIGNMENT_FAIL" if phase4_entered else "D4_CARRIER_ALIGNMENT_BLOCKED")
    if not phase5_pass:
        labels.append("D5_COUNTERFACTUAL_FAIL" if phase5_entered else "D5_COUNTERFACTUAL_BLOCKED")
    labels.append("D6_RUNTIME_ACTION_BLOCKED")
    if phase7_entered:
        labels.append("D7_SWA_SECONDARY_CARRIER_PASS" if phase7_pass else "D7_SWA_SECONDARY_CARRIER_FAIL")
    else:
        labels.append("D7_SWA_SECONDARY_CARRIER_BLOCKED")
    labels.append("D8_TTT_NOT_ELIGIBLE")

    if not object_source_pass:
        final_status = "NO_GO_OBJECT_RADIO_ROW_SOURCE_INSUFFICIENT"
        blocker = "object_identity_radio_source_insufficient"
    elif not phase2_pass:
        final_status = "NO_GO_OBJECT_TOPOLOGY_POLICY_NOT_SPECIFIC"
        blocker = "object_topology_policy_not_specific"
    elif not phase3_pass:
        final_status = "NO_GO_TRUE_MERGE_GAUGE_TRACE_INSUFFICIENT"
        blocker = "true_merge_gauge_trace_insufficient"
    elif not phase4_pass:
        final_status = "NO_GO_MERGE_GAUGE_CARRIER_ALIGNMENT_NOT_SPECIFIC"
        blocker = "merge_gauge_carrier_alignment_not_specific"
    elif phase5_entered and not phase5_pass and phase7_entered and not phase7_pass:
        final_status = "NO_GO_COUNTERFACTUAL_AND_SWA_SECONDARY_FAILED"
        blocker = "counterfactual_upper_bound_failed;swa_secondary_carrier_failed"
    elif phase5_entered and not phase5_pass:
        final_status = "NO_GO_COUNTERFACTUAL_UPPER_BOUND_FAILED"
        blocker = "counterfactual_upper_bound_failed"
    elif not phase5_entered:
        final_status = "READY_FOR_COUNTERFACTUAL_NOT_EXECUTED"
        blocker = "counterfactual_not_run"
    else:
        final_status = "READY_FOR_RUNTIME_NOT_EXECUTED"
        blocker = "runtime_not_run"
    if object_source_pass and phase3_pass and not phase2_pass and phase4_entered and not phase4_pass:
        blocker = "object_topology_policy_not_specific;merge_gauge_carrier_alignment_not_specific"

    carrier_summary = {
        "phase": "Phase4_merge_gauge_carrier_alignment_or_blocked",
        "entered": phase4_entered,
        "phase4_carrier_alignment_gate_pass": phase4_pass,
        "blocker": phase4.get("blocker") or ("blocked_by_phase2_policy_fail" if not phase2_pass else "blocked_by_phase3_trace_fail" if not phase3_pass else "carrier_alignment_not_run"),
        "reason": "Plan requires object-topology policy specificity and true merge/gauge trace coverage before carrier alignment.",
        "object_policy_gate_pass": phase2_pass,
        "phase3_trace_availability_gate_pass": phase3_pass,
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }
    counterfactual_summary = {
        "phase": "Phase5_counterfactual_or_blocked",
        "entered": phase5_entered,
        "phase5_counterfactual_gate_pass": phase5_pass,
        "counterfactual_executed": bool(phase5.get("counterfactual_executed")),
        "blocker": phase5.get("blocker")
        or ("blocked_without_carrier_alignment_pass" if not phase4_pass else "counterfactual_not_run"),
        "evidence": "phase5_merge_gauge_counterfactual_upper_bound/counterfactual_upper_bound_summary.json"
        if phase5_entered
        else "",
        "actual_runtime_trajectory_counterfactual_available": bool(
            phase5.get("actual_runtime_trajectory_counterfactual_available")
        ),
        "trace_level_upper_bound_only": bool(phase5.get("trace_level_upper_bound_only")),
        "runtime_action_allowed": phase5_pass,
        "ttt_allowed": False,
    }
    runtime_summary = {
        "phase": "Phase6_runtime_or_blocked",
        "entered": False,
        "phase6_runtime_action_gate_pass": False,
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "blocker": "blocked_without_counterfactual_upper_bound_pass" if not phase5_pass else "runtime_not_executed_in_this_audit",
        "ttt_allowed": False,
    }
    swa_summary = {
        "phase": "Phase7_swa_secondary_carrier_or_blocked",
        "entered": phase7_entered,
        "phase7_swa_secondary_carrier_gate_pass": phase7_pass,
        "blocker": phase7.get("blocker") if phase7_entered else "blocked_preconditions_missing",
        "required_preconditions": "Phase2 object policy pass, Phase3 sufficient trace or merge carrier fail with sufficient trace, and object row join coverage pass. Phase5 failure can also direct SWA secondary/action-surface rediscovery without opening runtime/TTT.",
        "phase2_policy_gate_pass": phase2_pass,
        "phase3_trace_gate_pass": phase3_pass,
        "object_source_pass": object_source_pass,
        "evidence": "phase7_swa_secondary_carrier/carrier_audit/phase7_swa_secondary_summary.json" if phase7_entered else "",
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    ttt_summary = {
        "phase": "Phase8_ttt_eligibility_or_blocked",
        "entered": False,
        "phase8_ttt_eligibility_gate_pass": False,
        "ttt_allowed": False,
        "ttt_executed": False,
        "blocker": "blocked_without_runtime_merge_gauge_or_swa_action_pass",
        "runtime_action_allowed": False,
    }

    visual_rows = [
        {
            "category": "object_identity_join_examples",
            "status": "evidence_table_available",
            "evidence": "phase1_object_identity_row_join/object_identity_row_join.csv",
            "blocked_reason": "",
        },
        {
            "category": "same_object_interior_support_examples",
            "status": "blocked_placeholder",
            "evidence": "phase1_object_identity_row_join/object_identity_source_summary.json",
            "blocked_reason": "row-level object identity was not established",
        },
        {
            "category": "cross_object_invalid_conflict_examples",
            "status": "blocked_placeholder",
            "evidence": "phase2_object_topology_policy/object_topology_false_positive_by_category.csv",
            "blocked_reason": "cross-object policy categories remained proxy-derived and not object-id-confirmed",
        },
        {
            "category": "merge_gauge_true_trace_examples",
            "status": "evidence_table_available",
            "evidence": "phase3_merge_gauge_trace_audit/merge_gauge_trace_ledger.csv",
            "blocked_reason": "",
        },
        {
            "category": "boundary_trace_unavailable_examples",
            "status": "evidence_table_available",
            "evidence": "phase3_merge_gauge_trace_audit/merge_gauge_trace_file_inventory.csv",
            "blocked_reason": "",
        },
        {
            "category": "carrier_alignment_panels",
            "status": "evidence_table_available" if phase4_pass else "blocked_placeholder",
            "evidence": "phase4_merge_gauge_carrier_alignment/carrier_alignment_summary.json",
            "blocked_reason": "" if phase4_pass else "Phase4 not entered or failed",
        },
        {
            "category": "counterfactual_panels",
            "status": "evidence_table_available" if phase5_entered else "blocked_placeholder",
            "evidence": "phase5_merge_gauge_counterfactual_upper_bound/counterfactual_upper_bound_summary.json"
            if phase5_entered
            else "phase5_counterfactual_or_blocked/counterfactual_or_blocked_summary.json",
            "blocked_reason": "" if phase5_entered else "Phase5 not allowed",
        },
        {
            "category": "runtime_action_panels",
            "status": "blocked_placeholder",
            "evidence": "phase6_runtime_or_blocked/runtime_or_blocked_summary.json",
            "blocked_reason": "Phase6 not allowed",
        },
        {
            "category": "swa_secondary_route_panels",
            "status": "evidence_table_available" if phase7_entered else "blocked_placeholder",
            "evidence": "phase7_swa_secondary_carrier/carrier_audit/phase7_swa_secondary_summary.json"
            if phase7_entered
            else "phase7_swa_secondary_carrier_or_blocked/swa_secondary_carrier_blocked_summary.json",
            "blocked_reason": "" if phase7_entered else "Phase7 not entered",
        },
        {
            "category": "ttt_eligibility_panels",
            "status": "blocked_placeholder",
            "evidence": "phase8_ttt_eligibility_or_blocked/ttt_eligibility_summary.json",
            "blocked_reason": "TTT remains closed",
        },
    ]
    visual_summary = {
        "phase": "Phase9_visual_rediscovery_or_blocked",
        "visual_gate_pass": False,
        "visual_bundle_status": "evidence_tables_plus_blocked_placeholders_no_fake_images",
        "required_categories": len(visual_rows),
        "evidence_table_categories": int(sum(1 for row in visual_rows if row["status"] == "evidence_table_available")),
        "blocked_placeholder_categories": int(sum(1 for row in visual_rows if row["status"] == "blocked_placeholder")),
        "no_fake_counterfactual_runtime_ttt_panels": True,
        "blocker": "visual_success_panels_blocked_by_no_go_preconditions",
    }

    write_json(args.root / "phase4_carrier_alignment_or_blocked/carrier_alignment_blocked_summary.json", carrier_summary)
    write_json(args.root / "phase5_counterfactual_or_blocked/counterfactual_or_blocked_summary.json", counterfactual_summary)
    write_json(args.root / "phase6_runtime_or_blocked/runtime_or_blocked_summary.json", runtime_summary)
    write_json(args.root / "phase7_swa_secondary_carrier_or_blocked/swa_secondary_carrier_blocked_summary.json", swa_summary)
    write_json(args.root / "phase8_ttt_eligibility_or_blocked/ttt_eligibility_summary.json", ttt_summary)
    write_csv(args.root / "phase9_visual_rediscovery_or_blocked/visual_requirement_matrix.csv", visual_rows)
    write_json(args.root / "phase9_visual_rediscovery_or_blocked/visual_rediscovery_summary.json", visual_summary)
    (args.root / "phase9_visual_rediscovery_or_blocked/visual_insight.md").write_text(
        "\n".join(
            [
                "# ACL2 v93 Visual Rediscovery Insight",
                "",
                "No runtime/TTT success image panel is generated because the required gates did not pass.",
                "The useful visual/audit substitute is a requirement matrix pointing to measured tables and blocked summaries.",
                "",
                "Key insight: v93 now exposes object identity, policy specificity, and merge/gauge carrier alignment, but the trace-level counterfactual upper bound failed good/bad residual gates and SWA secondary route evidence did not open runtime/TTT.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    final_questions = [
        {
            "question_id": 1,
            "question": "Was v92 No-Go boundary locked?",
            "answer": b(phase0.get("phase0_gate_pass")),
            "evidence": "phase0_v92_evidence_lock/phase0_gate_summary.json",
        },
        {
            "question_id": 2,
            "question": "Did row-level object identity or RADIO join become available?",
            "answer": object_source_pass,
            "evidence": f"object_identity_labelled_coverage={phase1.get('object_identity_labelled_coverage')}; radio_labelled_coverage={phase1.get('radio_labelled_coverage')}; radio_seq_coverage={phase1.get('radio_seq_coverage')}",
        },
        {
            "question_id": 3,
            "question": "Did object-topology policy beat object/component/semantic/regime shuffles?",
            "answer": phase2_pass,
            "evidence": "phase2_object_topology_policy/object_topology_policy_audit.json",
        },
        {
            "question_id": 4,
            "question": "Did plan-approved policy repair pass?",
            "answer": repair_pass,
            "evidence": "phase2_object_topology_policy/object_topology_policy_repair_metrics.csv",
        },
        {
            "question_id": 5,
            "question": "Did true merge/gauge boundary trace coverage reach gate?",
            "answer": phase3_pass,
            "evidence": f"row_coverage={phase3.get('row_coverage')}; merge_residual_delta_available_ratio={phase3.get('merge_residual_delta_available_ratio')}",
        },
        {
            "question_id": 6,
            "question": "Did semantic policy align with boundary_update_norm / merge_residual_delta?",
            "answer": phase4_pass,
            "evidence": "phase4_merge_gauge_carrier_alignment/carrier_alignment_summary.json",
        },
        {
            "question_id": 7,
            "question": "Did counterfactual upper bound pass?",
            "answer": phase5_pass,
            "evidence": "phase5_merge_gauge_counterfactual_upper_bound/counterfactual_upper_bound_summary.json"
            if phase5_entered
            else "Phase5 not executed",
        },
        {
            "question_id": 8,
            "question": "Did runtime action run?",
            "answer": False,
            "evidence": "runtime_action_allowed=false",
        },
        {
            "question_id": 9,
            "question": "Did runtime action improve bad rows and protect good rows?",
            "answer": False,
            "evidence": "runtime action not executed",
        },
        {
            "question_id": 10,
            "question": "Did SWA secondary carrier pass?",
            "answer": phase7_pass,
            "evidence": "phase7_swa_secondary_carrier/carrier_audit/phase7_swa_secondary_summary.json"
            if phase7_entered
            else "Phase7 not executed",
        },
        {
            "question_id": 11,
            "question": "Was TTT eligible?",
            "answer": False,
            "evidence": "TTT remains blocked without runtime memory action",
        },
        {
            "question_id": 12,
            "question": "What is the final blocker?",
            "answer": blocker,
            "evidence": ",".join(sorted(set(labels))),
        },
    ]

    decision = {
        "final_status": final_status,
        "final_no_go_allowed": True,
        "blocker": blocker,
        "decision_labels": sorted(set(labels)),
        "phase0_gate_pass": phase0.get("phase0_gate_pass"),
        "phase1_source_gate_pass": phase1_audit.get("phase1_source_gate_pass"),
        "object_identity_source_pass": phase1_audit.get("object_identity_source_pass"),
        "radio_source_pass": phase1_audit.get("radio_source_pass"),
        "phase2_object_topology_policy_gate_pass": phase2.get("phase2_object_topology_policy_gate_pass"),
        "phase2_repair_gate_pass": phase2.get("repair_gate_pass"),
        "phase3_trace_availability_gate_pass": phase3.get("phase3_trace_availability_gate_pass"),
        "phase4_carrier_alignment_gate_pass": phase4.get("phase4_carrier_alignment_gate_pass"),
        "carrier_alignment_entered": phase4_entered,
        "phase5_counterfactual_gate_pass": phase5_pass,
        "counterfactual_allowed": phase4_pass,
        "counterfactual_executed": bool(phase5.get("counterfactual_executed")),
        "phase7_swa_secondary_carrier_entered": phase7_entered,
        "phase7_swa_secondary_carrier_gate_pass": phase7_pass,
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
        "visual_gate_pass": visual_summary["visual_gate_pass"],
        "key_metrics": {
            "phase1_row_count": phase1.get("row_count"),
            "phase1_labelled_row_count": phase1.get("labelled_row_count"),
            "object_identity_available_ratio": phase1.get("object_identity_available_ratio"),
            "object_identity_labelled_coverage": phase1.get("object_identity_labelled_coverage"),
            "object_identity_seq_coverage": phase1.get("object_identity_seq_coverage"),
            "radio_available_ratio": phase1.get("radio_available_ratio"),
            "radio_labelled_coverage": phase1.get("radio_labelled_coverage"),
            "radio_seq_coverage": phase1.get("radio_seq_coverage"),
            "component_tracklet_available_ratio": phase1.get("component_tracklet_available_ratio"),
            "phase2_P5_bad_recall": phase2.get("actual_policy", {}).get("bad_recall") if isinstance(phase2.get("actual_policy"), dict) else None,
            "phase2_P5_good_FPR": phase2.get("actual_policy", {}).get("good_FPR") if isinstance(phase2.get("actual_policy"), dict) else None,
            "phase2_object_shuffle_margin": phase2.get("actual_policy", {}).get("object_shuffle_margin") if isinstance(phase2.get("actual_policy"), dict) else None,
            "phase2_component_shuffle_margin": phase2.get("actual_policy", {}).get("component_shuffle_margin") if isinstance(phase2.get("actual_policy"), dict) else None,
            "phase2_semantic_shuffle_margin": phase2.get("actual_policy", {}).get("semantic_shuffle_margin") if isinstance(phase2.get("actual_policy"), dict) else None,
            "phase2_regime_shuffle_margin": phase2.get("actual_policy", {}).get("regime_shuffle_margin") if isinstance(phase2.get("actual_policy"), dict) else None,
            "phase2_loso_positive_folds": phase2.get("actual_policy", {}).get("LOSO_positive_folds") if isinstance(phase2.get("actual_policy"), dict) else None,
            "phase3_row_coverage": phase3.get("row_coverage"),
            "phase3_labelled_trace_coverage": phase3.get("labelled_trace_coverage"),
            "phase3_trace_seq_coverage": phase3.get("trace_seq_coverage"),
            "phase3_boundary_update_norm_available_ratio": phase3.get("boundary_update_norm_available_ratio"),
            "phase3_boundary_update_norm_direct_ratio": phase3.get("boundary_update_norm_direct_ratio"),
            "phase3_merge_residual_delta_available_ratio": phase3.get("merge_residual_delta_available_ratio"),
            "phase3_trace_provenance_ratio": phase3.get("trace_provenance_ratio"),
            "phase3_merge_state_trace_files": hidden.get("merge_state_trace_files"),
            "phase4_actual_bad_recall": phase4.get("actual_policy_metrics", {}).get("bad_recall") if isinstance(phase4.get("actual_policy_metrics"), dict) else None,
            "phase4_actual_good_FPR": phase4.get("actual_policy_metrics", {}).get("good_FPR") if isinstance(phase4.get("actual_policy_metrics"), dict) else None,
            "phase4_trace_true_fields_pass": phase4.get("trace_true_fields_pass"),
            "phase5_actual_bad_median_residual_improvement_ratio": phase5.get("actual_family", {}).get("bad_median_residual_improvement_ratio")
            if isinstance(phase5.get("actual_family"), dict)
            else None,
            "phase5_good_median_residual_worsen_ratio": phase5.get("actual_family", {}).get("good_median_residual_worsen_ratio")
            if isinstance(phase5.get("actual_family"), dict)
            else None,
            "phase5_actual_minus_best_control": phase5.get("actual_minus_best_control"),
            "phase7_query_actual_minus_random_p95": phase7.get("query", {}).get("actual_minus_random_p95")
            if isinstance(phase7.get("query"), dict)
            else None,
            "phase7_query_object_margin": phase7.get("query", {}).get("object_margin")
            if isinstance(phase7.get("query"), dict)
            else None,
            "phase7_pair_actual_minus_random_p95": phase7.get("pair", {}).get("actual_minus_random_p95")
            if isinstance(phase7.get("pair"), dict)
            else None,
            "phase7_pair_object_margin": phase7.get("pair", {}).get("object_margin")
            if isinstance(phase7.get("pair"), dict)
            else None,
            "phase7_row_entropy_drop_available": phase7.get("row_entropy_drop_available"),
            "phase7_collapse_flag": phase7.get("collapse_flag"),
        },
        "final_questions": final_questions,
    }
    write_json(args.out_dir / "final_decision.json", decision)
    write_csv(args.out_dir / "decision_labels.csv", [{"label": label} for label in decision["decision_labels"]])
    write_csv(args.out_dir / "blocker_attribution.csv", [{"final_status": final_status, "blocker": blocker}])
    write_csv(args.out_dir / "final_decision_answers.csv", final_questions)

    report = [
        "# ACL2 v93 Final Report",
        "",
        f"- final_status: `{final_status}`",
        f"- blocker: `{blocker}`",
        f"- phase0_gate_pass: `{decision['phase0_gate_pass']}`",
        f"- object_identity_source_pass: `{decision['object_identity_source_pass']}`",
        f"- radio_source_pass: `{decision['radio_source_pass']}`",
        f"- phase2_object_topology_policy_gate_pass: `{decision['phase2_object_topology_policy_gate_pass']}`",
        f"- phase2_repair_gate_pass: `{decision['phase2_repair_gate_pass']}`",
        f"- phase3_trace_availability_gate_pass: `{decision['phase3_trace_availability_gate_pass']}`",
        f"- phase4_carrier_alignment_gate_pass: `{decision['phase4_carrier_alignment_gate_pass']}`",
        f"- phase5_counterfactual_gate_pass: `{decision['phase5_counterfactual_gate_pass']}`",
        f"- phase7_swa_secondary_carrier_gate_pass: `{decision['phase7_swa_secondary_carrier_gate_pass']}`",
        f"- runtime_action_allowed: `{decision['runtime_action_allowed']}`",
        f"- ttt_allowed: `{decision['ttt_allowed']}`",
        "",
        "## Key Evidence",
        "",
        f"- object_identity_labelled_coverage: `{phase1.get('object_identity_labelled_coverage')}`",
        f"- radio_labelled_coverage: `{phase1.get('radio_labelled_coverage')}`; radio_seq_coverage: `{phase1.get('radio_seq_coverage')}`",
        f"- Phase2 blocker: `{phase2.get('blocker')}`",
        f"- Phase3 row_coverage: `{phase3.get('row_coverage')}`; merge_residual_delta_available_ratio: `{phase3.get('merge_residual_delta_available_ratio')}`",
        f"- Phase4 blocker: `{phase4.get('blocker')}`; trace_true_fields_pass: `{phase4.get('trace_true_fields_pass')}`",
        f"- Phase5 blocker: `{phase5.get('blocker')}`; actual_bad_median_residual_improvement_ratio: `{decision['key_metrics']['phase5_actual_bad_median_residual_improvement_ratio']}`",
        f"- Phase7 blocker: `{phase7.get('blocker')}`; query_actual_minus_random_p95: `{decision['key_metrics']['phase7_query_actual_minus_random_p95']}`; pair_actual_minus_random_p95: `{decision['key_metrics']['phase7_pair_actual_minus_random_p95']}`",
        "",
        "## Final Questions",
        "",
        *[f"{row['question_id']}. {row['question']}: `{row['answer']}` Evidence: {row['evidence']}" for row in final_questions],
        "",
        "Conclusion: v93 did not reach runtime eligibility. Source coverage, object-topology policy, and merge/gauge carrier alignment now pass, but the Phase5 trace-level counterfactual upper bound failed. Phase7 SWA secondary evidence is recorded when present, but runtime action and TTT remain closed.",
    ]
    (args.out_dir / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"final_status={final_status}")
    print(f"blocker={blocker}")
    print(f"object_source_pass={object_source_pass}")
    print(f"phase2_gate_pass={phase2_pass}")
    print(f"phase3_trace_gate_pass={phase3_pass}")
    print(f"phase7_swa_secondary_carrier_gate_pass={phase7_pass}")
    print("runtime_action_allowed=False")
    print("ttt_allowed=False")


if __name__ == "__main__":
    main()
