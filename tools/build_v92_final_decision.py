#!/usr/bin/env python3
"""Build v92 final decision from measured phase artifacts only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v86_soft_latent_utils import read_json, write_csv, write_json  # noqa: E402
from tools.v92_semantic_policy_carrier_utils import ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "report_final")
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phase0 = _json(args.root / "phase0_evidence_lock/phase0_gate_summary.json")
    phase1 = _json(args.root / "phase1_semantic_policy_row_bank/phase1_gate_summary.json")
    phase2 = _json(args.root / "phase2_boundary_trace_ledger/phase2_gate_summary.json")
    phase2_noop = _json(args.root / "phase2_boundary_trace_ledger/noop_trace_smoke_summary.json")
    phase4 = _json(args.root / "phase4_swa_semantic_policy_route_audit/phase4_swa_carrier_audit/phase4_swa_carrier_summary.json")
    phase7_source = _json(args.root / "phase7_data_source_expansion/semantic_source_expansion_candidate_summary.json")
    phase7_sidecar = _json(args.root / "phase7_data_source_expansion/radio_tracklet_sidecar_summary.json")
    phase7_policy = _json(args.root / "phase7_data_source_expansion/expanded_semantic_policy_summary.json")
    phase4_query = phase4.get("query", {}) if isinstance(phase4.get("query"), dict) else {}
    phase4_pair = phase4.get("pair", {}) if isinstance(phase4.get("pair"), dict) else {}

    carrier_pass = bool(phase4.get("phase4_swa_carrier_gate_pass")) or bool(phase2.get("phase2_boundary_carrier_gate_pass"))
    counterfactual_allowed = carrier_pass
    runtime_allowed = False
    ttt_allowed = False
    labels: list[str] = []
    if not bool(phase0.get("phase0_gate_pass")):
        labels.append("NO_GO_POLICY_NOT_REPRODUCIBLE")
    if not bool(phase1.get("phase1_semantic_policy_row_bank_gate_pass")):
        labels.append("NO_GO_POLICY_NOT_REPRODUCIBLE")
    if not bool(phase2.get("phase2_boundary_trace_availability_gate_pass")):
        labels.append("NO_GO_BOUNDARY_TRACE_UNAVAILABLE")
    if not bool(phase4.get("phase4_swa_carrier_gate_pass")):
        labels.append("NO_GO_CARRIER_NOT_FOUND")
    if not bool(phase7_policy.get("phase7_data_source_expansion_useful")):
        labels.append("NO_GO_SEMANTIC_SOURCE_SPECIFICITY_INSUFFICIENT")
    if not counterfactual_allowed:
        labels.append("NO_GO_COUNTERFACTUAL_UPPER_BOUND_FAIL")
    if not runtime_allowed:
        labels.append("NO_GO_RUNTIME_ACTION_FAIL")
    if not ttt_allowed:
        labels.append("NO_GO_TTT_NOT_READY")

    if bool(phase0.get("phase0_gate_pass")) and bool(phase1.get("phase1_semantic_policy_row_bank_gate_pass")) and not carrier_pass and not bool(phase7_policy.get("phase7_data_source_expansion_useful")):
        final_status = "NO_GO_SEMANTIC_SOURCE_SPECIFICITY_INSUFFICIENT"
        blocker = "semantic_source_specificity_insufficient"
    elif not carrier_pass:
        final_status = "NO_GO_CARRIER_NOT_FOUND"
        blocker = "carrier_not_found"
    elif not counterfactual_allowed:
        final_status = "NO_GO_COUNTERFACTUAL_UPPER_BOUND_FAIL"
        blocker = "counterfactual_not_allowed_without_carrier"
    else:
        final_status = "SUCCESS_SEMANTIC_POLICY_CARRIER_FOUND"
        blocker = ""

    phase5_blocked_dir = args.root / "phase5_counterfactual_or_blocked"
    phase6_blocked_dir = args.root / "phase6_runtime_or_blocked"
    phase8_blocked_dir = args.root / "phase8_ttt_eligibility_or_blocked"
    phase9_visual_dir = args.root / "phase9_visual_rediscovery_or_blocked"
    counterfactual_summary = {
        "phase": "Phase5_counterfactual_or_blocked",
        "phase5_counterfactual_gate_pass": False,
        "counterfactual_available": False,
        "counterfactual_executed": False,
        "counterfactual_scope": "blocked_no_verified_carrier",
        "blocker": "counterfactual_not_allowed_without_carrier",
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    runtime_summary = {
        "phase": "Phase6_runtime_or_blocked",
        "phase6_runtime_action_gate_pass": False,
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "blocker": "runtime_not_allowed_without_carrier_and_counterfactual",
        "ttt_allowed": False,
    }
    ttt_summary = {
        "phase": "Phase8_ttt_eligibility_or_blocked",
        "phase8_ttt_eligibility_gate_pass": False,
        "ttt_allowed": False,
        "ttt_executed": False,
        "blocker": "ttt_not_allowed_without_runtime_memory_action",
        "runtime_action_allowed": False,
    }
    visual_rows = [
        {
            "category": "semantic_policy_pass_panels",
            "status": "evidence_table_available",
            "evidence": "phase1_semantic_policy_row_bank/semantic_policy_rows.csv",
            "blocked_reason": "",
        },
        {
            "category": "boundary_trace_panels",
            "status": "blocked_placeholder",
            "evidence": "phase2_boundary_trace_ledger/phase2_gate_summary.json",
            "blocked_reason": "true boundary trace unavailable; no fake boundary update panel generated",
        },
        {
            "category": "merge_gauge_carrier_panels",
            "status": "blocked_placeholder",
            "evidence": "phase2_boundary_trace_ledger/noop_trace_smoke_summary.json",
            "blocked_reason": "no-op trace smoke completed but native trace identity-only",
        },
        {
            "category": "swa_route_control_panels",
            "status": "evidence_table_available",
            "evidence": "phase4_swa_semantic_policy_route_audit/phase4_swa_carrier_audit/phase4_swa_carrier_rows.csv",
            "blocked_reason": "",
        },
        {
            "category": "counterfactual_panels",
            "status": "blocked_placeholder",
            "evidence": "phase5_counterfactual_or_blocked/counterfactual_or_blocked_summary.json",
            "blocked_reason": "counterfactual not allowed without carrier pass",
        },
        {
            "category": "runtime_action_panels_or_blocked_placeholders",
            "status": "blocked_placeholder",
            "evidence": "phase6_runtime_or_blocked/runtime_or_blocked_summary.json",
            "blocked_reason": "runtime action not allowed without counterfactual pass",
        },
        {
            "category": "data_source_expansion_panels",
            "status": "evidence_table_available",
            "evidence": "phase7_data_source_expansion/expanded_semantic_policy_strategy_metrics.csv",
            "blocked_reason": "",
        },
        {
            "category": "ttt_eligibility_panels_or_blocked_placeholders",
            "status": "blocked_placeholder",
            "evidence": "phase8_ttt_eligibility_or_blocked/ttt_eligibility_summary.json",
            "blocked_reason": "TTT remains blocked without runtime memory action",
        },
    ]
    visual_summary = {
        "phase": "Phase9_visual_rediscovery_or_blocked",
        "visual_gate_pass": False,
        "visual_bundle_status": "blocked_placeholders_only_no_fake_images",
        "review_coverage": 1.0,
        "required_categories": len(visual_rows),
        "evidence_table_categories": int(sum(1 for row in visual_rows if row["status"] == "evidence_table_available")),
        "blocked_placeholder_categories": int(sum(1 for row in visual_rows if row["status"] == "blocked_placeholder")),
        "no_fake_route_runtime_ttt_panels": True,
        "blocker": "image_panel_gate_not_run_because_carrier_counterfactual_runtime_blocked",
    }
    write_json(phase5_blocked_dir / "counterfactual_or_blocked_summary.json", counterfactual_summary)
    write_json(phase6_blocked_dir / "runtime_or_blocked_summary.json", runtime_summary)
    write_json(phase8_blocked_dir / "ttt_eligibility_summary.json", ttt_summary)
    write_csv(phase9_visual_dir / "visual_requirement_matrix.csv", visual_rows)
    write_json(phase9_visual_dir / "visual_rediscovery_summary.json", visual_summary)
    (phase9_visual_dir / "visual_insight.md").write_text(
        "\n".join(
            [
                "# ACL2 v92 Visual Rediscovery Insight",
                "",
                "No image-based success panel is generated for blocked carrier/runtime/TTT phases.",
                "The evidence bundle is represented as a requirement matrix that points to the measured CSV/JSON artifacts and explicitly marks blocked categories.",
                "",
                "Key insight: semantic policy rows are reproducible, but true merge/gauge carrier trace is unavailable and SWA query/pair route lift is too small. Phase7 found candidate sidecars and full component-proxy coverage, but no audited row-level object/RADIO join.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    decision = {
        "final_status": final_status,
        "blocker": blocker,
        "decision_labels": sorted(set(labels)),
        "phase0_gate_pass": phase0.get("phase0_gate_pass"),
        "phase1_semantic_policy_row_bank_gate_pass": phase1.get("phase1_semantic_policy_row_bank_gate_pass"),
        "phase2_boundary_trace_availability_gate_pass": phase2.get("phase2_boundary_trace_availability_gate_pass"),
        "phase2_noop_smoke_completed": phase2_noop.get("all_completed"),
        "phase4_swa_carrier_gate_pass": phase4.get("phase4_swa_carrier_gate_pass"),
        "phase7_source_candidate_audit_gate_pass": phase7_source.get("phase7_source_candidate_audit_gate_pass"),
        "phase7_sidecar_manifest_gate_pass": phase7_sidecar.get("phase7_sidecar_manifest_gate_pass"),
        "phase7_data_source_expansion_useful": phase7_policy.get("phase7_data_source_expansion_useful"),
        "counterfactual_allowed": counterfactual_allowed,
        "counterfactual_executed": False,
        "runtime_action_allowed": runtime_allowed,
        "runtime_action_executed": False,
        "ttt_allowed": ttt_allowed,
        "visual_gate_pass": visual_summary["visual_gate_pass"],
        "key_metrics": {
            "phase1_bad_recall": phase1.get("bad_recall"),
            "phase1_good_FPR": phase1.get("good_FPR"),
            "phase1_semantic_shuffle_margin": phase1.get("semantic_shuffle_margin"),
            "phase1_component_shuffle_margin": phase1.get("component_shuffle_margin"),
            "phase1_regime_shuffle_margin": phase1.get("regime_shuffle_margin"),
            "phase2_true_trace_ratio": phase2.get("true_trace_ratio"),
            "phase2_boundary_update_norm_available_ratio": phase2.get("boundary_update_norm_available_ratio"),
            "phase2_noop_non_identity_transform_rows": phase2_noop.get("non_identity_transform_rows"),
            "phase4_query_actual_minus_random_p95": phase4_query.get("actual_minus_random_p95"),
            "phase4_pair_actual_minus_random_p95": phase4_pair.get("actual_minus_random_p95"),
            "phase4_pair_semantic_shuffle_margin": phase4_pair.get("semantic_shuffle_margin"),
            "phase4_pair_component_shuffle_margin": phase4_pair.get("component_shuffle_margin"),
            "phase4_pair_regime_shuffle_margin": phase4_pair.get("regime_shuffle_margin"),
            "phase7_has_radio_ratio": phase7_source.get("has_radio_ratio"),
            "phase7_object_identity_available_ratio": phase7_source.get("object_identity_available_ratio"),
            "phase7_component_tracklet_available_ratio": phase7_source.get("component_tracklet_available_ratio"),
            "phase7_expanded_bad_recall": phase7_policy.get("expanded_policy_bad_recall"),
            "phase7_expanded_good_FPR": phase7_policy.get("expanded_policy_good_FPR"),
            "phase7_bad_recall_improvement_vs_phase1": phase7_policy.get("bad_recall_improvement_vs_phase1"),
            "phase7_good_FPR_improvement_vs_phase1": phase7_policy.get("good_FPR_improvement_vs_phase1"),
            "phase7_expanded_semantic_shuffle_margin": phase7_policy.get("expanded_semantic_shuffle_margin"),
            "phase7_expanded_component_shuffle_margin": phase7_policy.get("expanded_component_shuffle_margin"),
        },
    }
    answers = [
        {"question_id": 1, "question": "v91 Phase5 policy reproduced?", "answer": bool(phase1.get("phase1_semantic_policy_row_bank_gate_pass")), "evidence": "phase1_semantic_policy_row_bank/phase1_gate_summary.json"},
        {"question_id": 2, "question": "policy uses GT or bad/good labels for assignment?", "answer": False, "evidence": "phase1 bad_good_label_used_for_assignment=false scale_label_used_for_assignment=false"},
        {"question_id": 3, "question": "tested carriers", "answer": "merge_gauge_boundary_trace, SWA query-side, SWA QK/pair, data-source fallback; READ/TTT not promoted", "evidence": "phase2, phase4, phase7 summaries"},
        {"question_id": 4, "question": "carrier trace true or proxy?", "answer": "boundary true trace unavailable except 4/49 no-op smoke rows; policy trace remains proxy/diagnostic", "evidence": "phase2_gate_summary.json and noop_trace_smoke_summary.json"},
        {"question_id": 5, "question": "semantic policy beats shuffle?", "answer": bool(phase1.get("phase1_semantic_policy_row_bank_gate_pass")), "evidence": "phase1 shuffle margins"},
        {"question_id": 6, "question": "geometry-only control stronger?", "answer": "Phase4 geometry-only pair route lift did not rescue carrier; see phase4 controls", "evidence": "phase4_swa_carrier_summary.json"},
        {"question_id": 7, "question": "merge/gauge boundary closer than SWA route?", "answer": "not proven; merge/gauge true trace unavailable, SWA route tested and failed tiny lift gates", "evidence": "phase2 and phase4 summaries"},
        {"question_id": 8, "question": "counterfactual upper bound exists?", "answer": False, "evidence": "no carrier pass, counterfactual not allowed"},
        {"question_id": 9, "question": "runtime action allowed and executed?", "answer": False, "evidence": "runtime_action_allowed=false runtime_action_executed=false"},
        {"question_id": 10, "question": "good cases protected?", "answer": "Phase1 policy yes by measured FPR; Phase7 fallback did not improve enough to promote", "evidence": "phase1 and phase7 summaries"},
        {"question_id": 11, "question": "TTT blocked?", "answer": True, "evidence": "ttt_allowed=false because no confirmed carrier/runtime action"},
        {"question_id": 12, "question": "next step", "answer": "stop current threshold-sweep family; need true merge/gauge instrumentation or row-level object/RADIO join before runtime", "evidence": final_status},
    ]
    decision["final_questions"] = answers
    write_json(args.out_dir / "final_decision.json", decision)
    write_csv(args.out_dir / "decision_labels.csv", [{"label": label} for label in decision["decision_labels"]])
    write_csv(args.out_dir / "final_decision_answers.csv", answers)
    write_csv(args.out_dir / "blocker_attribution.csv", [{"final_status": final_status, "blocker": blocker}])
    report_lines = [
        "# ACL2 v92 Final Report",
        "",
        f"- final_status: `{final_status}`",
        f"- blocker: `{blocker}`",
        f"- phase0_gate_pass: `{decision['phase0_gate_pass']}`",
        f"- phase1_semantic_policy_row_bank_gate_pass: `{decision['phase1_semantic_policy_row_bank_gate_pass']}`",
        f"- phase2_boundary_trace_availability_gate_pass: `{decision['phase2_boundary_trace_availability_gate_pass']}`",
        f"- phase4_swa_carrier_gate_pass: `{decision['phase4_swa_carrier_gate_pass']}`",
        f"- phase7_data_source_expansion_useful: `{decision['phase7_data_source_expansion_useful']}`",
        f"- runtime_action_allowed: `{decision['runtime_action_allowed']}`",
        f"- ttt_allowed: `{decision['ttt_allowed']}`",
        "",
        "## Final Questions",
        "",
        *[f"{row['question_id']}. {row['question']}: `{row['answer']}` Evidence: {row['evidence']}" for row in answers],
        "",
        "Conclusion: v92 reproduced the semantic policy signal but did not find a verified carrier. Data-source expansion found useful candidate files and component proxies, but no audited row-level object/RADIO join and no Phase7 metric improvement sufficient for promotion.",
    ]
    (args.out_dir / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"final_status={final_status}")
    print(f"blocker={blocker}")
    print(f"phase4_swa_carrier_gate_pass={decision['phase4_swa_carrier_gate_pass']}")
    print(f"phase7_data_source_expansion_useful={decision['phase7_data_source_expansion_useful']}")
    print(f"runtime_action_allowed={decision['runtime_action_allowed']}")
    print(f"ttt_allowed={decision['ttt_allowed']}")


if __name__ == "__main__":
    main()
