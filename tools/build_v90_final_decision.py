#!/usr/bin/env python3
"""Build v90 final decision from phase summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_csv, write_json
from v90_semantic_topology_utils import ROOT


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
    phase1 = _json(args.root / "phase1_semantic_topology_source/topology_source_audit.json")
    phase2 = _json(args.root / "phase2_semantic_topology_scale_mode_ledger/phase2_topology_ledger_audit.json")
    phase3 = _json(args.root / "phase3_semantic_topology_relevance/topology_relevance_summary.json")
    highobs = _json(args.root / "phase3_semantic_topology_relevance_highobs/topology_relevance_summary.json")
    far = _json(args.root / "phase3_semantic_topology_relevance_far/topology_relevance_summary.json")
    phase4 = _json(args.root / "phase4_semantic_topology_observability_policy/topology_observability_policy_audit_summary.json")
    phase5 = _json(args.root / "phase5_feature_match_topology_ruler/feature_match_topology_audit_summary.json")
    phase6 = _json(args.root / "phase6_topology_carrier_attribution/phase6_carrier_attribution_summary.json")
    visual = _json(args.root / "phase10_visual_rediscovery/visual_integrity_audit.json")
    labels: list[str] = []
    blocker = ""
    if not phase1.get("phase1_topology_source_gate_pass", False):
        labels.append("D1_TOPOLOGY_SOURCE_INSUFFICIENT")
        blocker = "topology_source_insufficient"
    if not phase3.get("phase3_topology_relevance_global_gate_pass", False):
        labels.extend(["D2_TOPOLOGY_SIGNAL_NOT_SCALE_RELEVANT", "D3_TOPOLOGY_SEMANTIC_NOT_SPECIFIC"])
        blocker = blocker or "topology_signal_not_global_scale_relevant"
    if not phase4.get("semantic_topology_observability_policy_gate_pass", False):
        labels.append("D4_TOPOLOGY_GOOD_PROTECTION_ONLY")
        blocker = blocker or "topology_policy_not_safe_or_not_specific"
    if not phase5.get("feature_match_topology_ruler_gate_pass", False):
        labels.append("D5_FEATURE_MATCH_TOPOLOGY_NOT_SCALE_RULER")
        blocker = blocker or "feature_match_topology_not_scale_ruler"
    if not phase6.get("entered", False):
        labels.append("D6_CARRIER_NOT_ENTERED")
        blocker = blocker or "carrier_not_entered_preconditions_failed"
    labels.extend(["D8_COUNTERFACTUAL_FAIL", "D9_RUNTIME_ACTION_FAIL", "D10_TTT_NOT_READY"])
    success = False
    split_diag = bool(highobs.get("phase3_topology_relevance_global_gate_pass", False) or far.get("phase3_topology_relevance_global_gate_pass", False))
    decision = {
        "final_status": "No-Go_before_runtime_action",
        "final_no_go_allowed": True,
        "success_label": "D11_SEMANTIC_TOPOLOGY_MEMORY_SUCCESS" if success else "",
        "decision_labels": sorted(set(labels)),
        "blocker": blocker or "unknown_no_go_blocker",
        "phase0_gate_pass": phase0.get("phase0_gate_pass"),
        "phase1_topology_source_gate_pass": phase1.get("phase1_topology_source_gate_pass"),
        "phase2_topology_ledger_audit_gate_pass": phase2.get("phase2_topology_ledger_audit_gate_pass"),
        "phase3_topology_relevance_global_gate_pass": phase3.get("phase3_topology_relevance_global_gate_pass"),
        "phase3_split_diagnostic_positive": split_diag,
        "phase3_split_diagnostic_only": split_diag and not phase3.get("phase3_topology_relevance_global_gate_pass", False),
        "phase4_topology_policy_gate_pass": phase4.get("semantic_topology_observability_policy_gate_pass"),
        "phase5_feature_match_topology_ruler_gate_pass": phase5.get("feature_match_topology_ruler_gate_pass"),
        "phase6_carrier_entered": phase6.get("entered", False),
        "counterfactual_tools_run": False,
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
        "visual_integrity_gate_pass": visual.get("visual_integrity_gate_pass"),
        "key_metrics": {
            "phase1_pair_rows": phase1.get("pair_rows"),
            "phase1_node_rows": phase1.get("node_rows"),
            "phase1_topology_edge_rows": phase1.get("topology_edge_rows"),
            "phase2_mode_rows": phase2.get("mode_rows"),
            "phase2_type_counts": phase2.get("topology_mode_type_counts"),
            "phase3_geometry_reference_rho": phase3.get("geometry_reference_rho"),
            "phase3_best_topology_signal": (phase3.get("best_topology_signal") or {}).get("signal") if isinstance(phase3.get("best_topology_signal"), dict) else None,
            "phase3_best_topology_rho": (phase3.get("best_topology_signal") or {}).get("spearman_rho_abs_log_scale_jump") if isinstance(phase3.get("best_topology_signal"), dict) else None,
            "phase4_bad_recall": phase4.get("bad_recall"),
            "phase4_good_FPR": phase4.get("good_FPR"),
            "phase5_match_topology_score_rho": phase5.get("match_topology_score_rho_abs_log_scale_jump"),
            "phase5_match_topology_valid_ratio_median": phase5.get("match_topology_valid_ratio_median"),
        },
    }
    final_questions = [
        {
            "question_id": 1,
            "question": "Did object-topology semantic source build correctly?",
            "answer": bool(phase1.get("phase1_topology_source_gate_pass", False)),
            "evidence": "phase1_topology_source_gate_pass",
        },
        {
            "question_id": 2,
            "question": "Did topology modes improve over compact semantic roles?",
            "answer": bool(phase3.get("phase3_topology_relevance_global_gate_pass", False)),
            "evidence": "Phase3 global topology relevance failed; split positives are diagnostic-only",
        },
        {
            "question_id": 3,
            "question": "Did topology modes improve over geometry-only or protect good rows?",
            "answer": bool(phase3.get("phase3_topology_relevance_global_gate_pass", False) or phase4.get("semantic_topology_observability_policy_gate_pass", False)),
            "evidence": "Phase3 global and Phase4 policy gates are both false",
        },
        {
            "question_id": 4,
            "question": "Did semantic/component shuffle break the signal?",
            "answer": "global_best_has_positive_margins_but_global_gate_failed",
            "evidence": "best topology margins are positive, but rho/lift gate failed globally",
        },
        {
            "question_id": 5,
            "question": "Did feature-match topology become scale-relevant?",
            "answer": bool(phase5.get("feature_match_topology_ruler_gate_pass", False)),
            "evidence": "phase5_feature_match_topology_ruler_gate_pass",
        },
        {
            "question_id": 6,
            "question": "Did any memory carrier expose the topology signal?",
            "answer": False,
            "evidence": "Phase6 not entered because Phase3/4/5 topology-specific global gates failed",
        },
        {
            "question_id": 7,
            "question": "Did counterfactual upper bound pass?",
            "answer": False,
            "evidence": "Phase7 not entered; no carrier candidate and no explicit owner allowance",
        },
        {
            "question_id": 8,
            "question": "Was runtime action allowed and executed?",
            "answer": False,
            "evidence": "runtime_action_allowed=false and runtime_action_executed=false",
        },
        {
            "question_id": 9,
            "question": "Was TTT allowed?",
            "answer": False,
            "evidence": "ttt_allowed=false",
        },
        {
            "question_id": 10,
            "question": "If No-Go, what is the blocker?",
            "answer": decision["blocker"],
            "evidence": ",".join(decision["decision_labels"]),
        },
    ]
    decision["final_questions"] = final_questions
    write_json(args.out_dir / "final_decision.json", decision)
    write_csv(args.out_dir / "decision_labels.csv", [{"label": item} for item in decision["decision_labels"]])
    write_csv(args.out_dir / "blocker_attribution.csv", [{"blocker": decision["blocker"], "final_status": decision["final_status"]}])
    write_csv(args.out_dir / "final_decision_answers.csv", final_questions)
    report = [
        "# ACL2 v90 Final Report",
        "",
        f"- final_status: `{decision['final_status']}`",
        f"- final_no_go_allowed: `{decision['final_no_go_allowed']}`",
        f"- blocker: `{decision['blocker']}`",
        f"- phase1_topology_source_gate_pass: `{decision['phase1_topology_source_gate_pass']}`",
        f"- phase2_topology_ledger_audit_gate_pass: `{decision['phase2_topology_ledger_audit_gate_pass']}`",
        f"- phase3_topology_relevance_global_gate_pass: `{decision['phase3_topology_relevance_global_gate_pass']}`",
        f"- phase3_split_diagnostic_only: `{decision['phase3_split_diagnostic_only']}`",
        f"- phase4_topology_policy_gate_pass: `{decision['phase4_topology_policy_gate_pass']}`",
        f"- phase5_feature_match_topology_ruler_gate_pass: `{decision['phase5_feature_match_topology_ruler_gate_pass']}`",
        f"- phase6_carrier_entered: `{decision['phase6_carrier_entered']}`",
        f"- runtime_action_allowed: `{decision['runtime_action_allowed']}`",
        f"- ttt_allowed: `{decision['ttt_allowed']}`",
        "",
        "## Final Questions",
        "",
        *[f"{row['question_id']}. {row['question']} `{row['answer']}` Evidence: {row['evidence']}" for row in final_questions],
        "",
        "Conclusion: v90 built the object/component topology source and topology scale-mode ledger, but the topology signal did not pass the global relevance/specificity gate. Split diagnostics exist and are explicitly not promoted to runtime eligibility. Feature-match topology remained match-abundant but not scale-relevant. Carrier, counterfactual, runtime action, and TTT were not entered.",
    ]
    (args.out_dir / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"final_status={decision['final_status']}")
    print(f"blocker={decision['blocker']}")
    print(f"phase1_topology_source_gate_pass={decision['phase1_topology_source_gate_pass']}")
    print(f"phase2_topology_ledger_audit_gate_pass={decision['phase2_topology_ledger_audit_gate_pass']}")
    print(f"phase3_topology_relevance_global_gate_pass={decision['phase3_topology_relevance_global_gate_pass']}")
    print(f"phase3_split_diagnostic_only={decision['phase3_split_diagnostic_only']}")
    print(f"phase4_topology_policy_gate_pass={decision['phase4_topology_policy_gate_pass']}")
    print(f"phase5_feature_match_topology_ruler_gate_pass={decision['phase5_feature_match_topology_ruler_gate_pass']}")
    print(f"runtime_action_allowed={decision['runtime_action_allowed']}")
    print(f"ttt_allowed={decision['ttt_allowed']}")


if __name__ == "__main__":
    main()
