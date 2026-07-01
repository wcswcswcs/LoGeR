#!/usr/bin/env python3
"""Build v91 final decision from measured phase artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_csv, write_json
from v91_semantic_regime_utils import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "report_final")
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _best(summary: dict[str, Any]) -> dict[str, Any]:
    obj = summary.get("best_semantic_policy", {})
    return obj if isinstance(obj, dict) else {}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phase0 = _json(args.root / "phase0_evidence_lock/phase0_gate_summary.json")
    phase1 = _json(args.root / "phase1_semantic_topology_tracklets/semantic_topology_tracklet_audit.json")
    phase1_build = _json(args.root / "phase1_semantic_topology_tracklets/semantic_topology_tracklet_summary.json")
    phase2 = _json(args.root / "phase2_semantic_regime_classifier/semantic_regime_audit.json")
    phase2_build = _json(args.root / "phase2_semantic_regime_classifier/semantic_regime_summary.json")
    phase3 = _json(args.root / "phase3_regime_conditioned_semantic_relevance/regime_conditioned_relevance_summary.json")
    phase4 = _json(args.root / "phase4_tracklet_mode_disambiguation/tracklet_mode_disambiguation_summary.json")
    phase5 = _json(args.root / "phase5_memory_update_policy/policy_state_audit.json")
    phase6 = _json(args.root / "phase6_adaptive_memory_baseline/delayed_commit_audit.json")
    phase7 = _json(args.root / "phase7_carrier_attribution_or_blocked/phase7_carrier_summary.json")
    phase8 = _json(args.root / "phase8_counterfactual_or_blocked/counterfactual_or_blocked_summary.json")
    phase9 = _json(args.root / "phase9_runtime_or_blocked/runtime_or_blocked_summary.json")
    phase10 = _json(args.root / "phase10_ttt_or_blocked/ttt_or_blocked_summary.json")
    visual = _json(args.root / "phase11_visual_rediscovery/visual_integrity_audit.json")
    best = _best(phase3)
    labels: list[str] = []
    blocker = ""
    if not bool(phase1.get("phase1_tracklet_audit_gate_pass")):
        labels.append("D1_TRACKLET_SOURCE_INSUFFICIENT")
        blocker = blocker or "semantic_topology_tracklet_source_failed"
    if not bool(phase2.get("phase2_regime_classifier_audit_gate_pass")):
        labels.append("D2_REGIME_CLASSIFIER_FAILED")
        blocker = blocker or "semantic_regime_classifier_failed"
    if not bool(phase3.get("phase3_regime_semantic_gate_pass")):
        labels.append("D3_REGIME_SEMANTIC_NOT_GLOBAL_SCALE_RELEVANT")
        p3_blocker = str(phase3.get("blocker", ""))
        specificity_failed = min(
            float(best.get("semantic_shuffle_margin", 0.0) or 0.0),
            float(best.get("component_shuffle_margin", 0.0) or 0.0),
            float(best.get("regime_shuffle_margin", 0.0) or 0.0),
        ) < 0.05
        if "specific" in p3_blocker or "shuffle" in p3_blocker or specificity_failed:
            labels.append("D4_SEMANTIC_SPECIFICITY_FAILED")
            blocker = blocker or "semantic_specificity_failed"
        else:
            blocker = blocker or "semantic_regime_not_scale_relevant"
    if not bool(phase4.get("phase4_tracklet_mode_gate_pass")):
        labels.append("D5_TRACKLET_MODE_DISAMBIGUATION_FAILED")
    if not bool(phase5.get("phase5_memory_update_policy_gate_pass")):
        labels.append("D6_MEMORY_UPDATE_POLICY_UNSAFE_OR_WEAK")
        if blocker == "":
            if float(phase5.get("good_FPR", 1.0) or 1.0) > 0.25:
                blocker = "memory_update_policy_good_fpr_too_high"
            else:
                blocker = "memory_update_policy_gate_failed"
    if not bool(phase6.get("phase6_delayed_commit_gate_pass")):
        labels.append("D7_DELAYED_COMMIT_NOT_SAFE_OR_SPECIFIC")
    if not bool(phase7.get("phase7_carrier_gate_pass")):
        labels.append("D8_CARRIER_NOT_FOUND_OR_NOT_ENTERED")
        blocker = blocker or str(phase7.get("blocker", "carrier_not_found"))
    if not bool(phase8.get("phase8_counterfactual_gate_pass")):
        labels.append("D9_COUNTERFACTUAL_NOT_PASSED")
    if not bool(phase9.get("phase9_runtime_action_gate_pass")):
        labels.append("D10_RUNTIME_ACTION_NOT_EXECUTED")
    if not bool(phase10.get("phase10_ttt_gate_pass")):
        labels.append("D11_TTT_NOT_READY")
    if not bool(visual.get("visual_integrity_gate_pass")):
        labels.append("D12_VISUAL_INTEGRITY_FAILED")
        blocker = blocker or "visual_integrity_gate_failed"
    runtime_pass = bool(phase9.get("phase9_runtime_action_gate_pass") and phase9.get("runtime_action_executed"))
    semantic_entry_gate = bool(
        phase3.get("phase3_regime_semantic_gate_pass")
        or phase4.get("phase4_tracklet_mode_gate_pass")
        or phase5.get("phase5_memory_update_policy_gate_pass")
        or phase6.get("phase6_delayed_commit_gate_pass")
    )
    if semantic_entry_gate and not bool(phase7.get("phase7_carrier_gate_pass")):
        blocker = str(phase7.get("blocker", "carrier_not_found"))
    if runtime_pass:
        final_status = "SUCCESS_RUNTIME_MEMORY_CONTROL"
        blocker = ""
    elif semantic_entry_gate and not bool(phase7.get("phase7_carrier_gate_pass")):
        final_status = "NO_GO_CARRIER_NOT_FOUND"
    elif not bool(phase3.get("phase3_regime_semantic_gate_pass")):
        final_status = "NO_GO_SEMANTIC_SPECIFICITY_FAILED" if "D4_SEMANTIC_SPECIFICITY_FAILED" in labels else "NO_GO_SEMANTIC_REGIME_NOT_SCALE_RELEVANT"
    elif not bool(phase5.get("phase5_memory_update_policy_gate_pass")) and float(phase5.get("good_FPR", 1.0) or 1.0) > 0.25:
        final_status = "NO_GO_POLICY_UNSAFE_GOOD_FPR"
    elif not bool(phase7.get("phase7_carrier_gate_pass")):
        final_status = "NO_GO_CARRIER_NOT_FOUND"
    elif not bool(phase8.get("phase8_counterfactual_gate_pass")):
        final_status = "NO_GO_COUNTERFACTUAL_UPPER_BOUND_FAIL"
    elif not bool(phase9.get("phase9_runtime_action_gate_pass")):
        final_status = "NO_GO_RUNTIME_ACTION_FAIL"
    else:
        final_status = "NO_GO_TTT_NOT_READY"
    decision = {
        "final_status": final_status,
        "success_label": "D13_SEMANTIC_TOPOLOGY_REGIME_MEMORY_SUCCESS" if runtime_pass else "",
        "final_no_go_allowed": not runtime_pass,
        "decision_labels": sorted(set(labels)),
        "blocker": blocker or "no_verified_runtime_success",
        "phase0_gate_pass": phase0.get("phase0_gate_pass"),
        "phase1_tracklet_audit_gate_pass": phase1.get("phase1_tracklet_audit_gate_pass"),
        "phase2_regime_classifier_audit_gate_pass": phase2.get("phase2_regime_classifier_audit_gate_pass"),
        "phase3_regime_semantic_gate_pass": phase3.get("phase3_regime_semantic_gate_pass"),
        "phase4_tracklet_mode_gate_pass": phase4.get("phase4_tracklet_mode_gate_pass"),
        "phase5_memory_update_policy_gate_pass": phase5.get("phase5_memory_update_policy_gate_pass"),
        "phase6_delayed_commit_gate_pass": phase6.get("phase6_delayed_commit_gate_pass"),
        "phase7_carrier_gate_pass": phase7.get("phase7_carrier_gate_pass"),
        "phase8_counterfactual_gate_pass": phase8.get("phase8_counterfactual_gate_pass"),
        "phase9_runtime_action_gate_pass": phase9.get("phase9_runtime_action_gate_pass"),
        "phase10_ttt_gate_pass": phase10.get("phase10_ttt_gate_pass"),
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
        "visual_integrity_gate_pass": visual.get("visual_integrity_gate_pass"),
        "key_metrics": {
            "phase1_pair_rows": phase1_build.get("pair_rows"),
            "phase1_tracklet_rows": phase1_build.get("tracklet_rows"),
            "phase1_sequence_coverage": phase1_build.get("sequence_coverage"),
            "phase2_regime_rows": phase2_build.get("regime_rows"),
            "phase2_regime_counts": phase2_build.get("regime_counts"),
            "phase2_max_single_regime_ratio": phase2_build.get("max_single_regime_ratio"),
            "phase3_best_semantic_policy": best.get("signal"),
            "phase3_best_semantic_rho": best.get("spearman_rho_abs_log_scale_jump"),
            "phase3_best_semantic_bad_recall": best.get("bad_recall"),
            "phase3_best_semantic_good_FPR": best.get("good_FPR"),
            "phase3_best_semantic_shuffle_margin": best.get("semantic_shuffle_margin"),
            "phase3_best_component_shuffle_margin": best.get("component_shuffle_margin"),
            "phase3_best_regime_shuffle_margin": best.get("regime_shuffle_margin"),
            "phase4_mean_entropy_reduction": phase4.get("mean_entropy_reduction"),
            "phase4_tracklet_mode_rho_abs_scale_jump": phase4.get("tracklet_mode_rho_abs_scale_jump"),
            "phase5_bad_recall": phase5.get("bad_recall"),
            "phase5_good_FPR": phase5.get("good_FPR"),
            "phase5_semantic_good_protection_margin": phase5.get("semantic_good_protection_margin"),
            "phase6_bad_recall": phase6.get("bad_recall"),
            "phase6_good_FPR": phase6.get("good_FPR"),
            "phase6_premature_update_reduction": phase6.get("premature_update_reduction"),
            "phase7_direct_boundary_update_trace_proxy_rows": phase7.get("direct_boundary_update_trace_proxy_rows"),
            "visual_review_coverage": visual.get("review_coverage"),
        },
    }
    final_questions = [
        {"question_id": 1, "question": "Did semantic topology tracklet source build correctly?", "answer": bool(phase1.get("phase1_tracklet_audit_gate_pass")), "evidence": "phase1_semantic_topology_tracklets/semantic_topology_tracklet_audit.json"},
        {"question_id": 2, "question": "Did regime classifier cover all rows without GT labels?", "answer": bool(phase2.get("phase2_regime_classifier_audit_gate_pass")), "evidence": "phase2_semantic_regime_classifier/semantic_regime_audit.json"},
        {"question_id": 3, "question": "Did regime-conditioned semantic policy beat geometry-only?", "answer": bool(phase3.get("phase3_regime_semantic_gate_pass")), "evidence": "phase3_regime_conditioned_semantic_relevance/regime_conditioned_relevance_summary.json"},
        {"question_id": 4, "question": "Did semantic/component/regime shuffle break the signal?", "answer": bool(best.get("semantic_shuffle_margin", 0) and best.get("component_shuffle_margin", 0) and best.get("regime_shuffle_margin", 0) and min(float(best.get("semantic_shuffle_margin", 0) or 0), float(best.get("component_shuffle_margin", 0) or 0), float(best.get("regime_shuffle_margin", 0) or 0)) >= 0.05), "evidence": "best_semantic_policy shuffle margins"},
        {"question_id": 5, "question": "Did semantic topology reduce good false positives inside geometry-conflict rows?", "answer": bool(float(phase5.get("semantic_good_protection_margin", 0.0) or 0.0) >= 0.10 and float(phase5.get("good_FPR", 1.0) or 1.0) <= 0.25), "evidence": "phase5 policy_state_audit.json"},
        {"question_id": 6, "question": "Did tracklet modes reduce scale-mode entropy?", "answer": bool(phase4.get("phase4_tracklet_mode_gate_pass")), "evidence": "phase4_tracklet_mode_disambiguation_summary.json"},
        {"question_id": 7, "question": "Did adaptive baseline / delayed commit improve safety?", "answer": bool(phase6.get("phase6_delayed_commit_gate_pass")), "evidence": "phase6 delayed_commit_audit.json"},
        {"question_id": 8, "question": "Which memory carrier carried the signal?", "answer": "none_verified" if not phase7.get("phase7_carrier_gate_pass") else "see_phase7_carrier_metrics", "evidence": "phase7_carrier_summary.json"},
        {"question_id": 9, "question": "Did counterfactual upper bound pass?", "answer": bool(phase8.get("phase8_counterfactual_gate_pass")), "evidence": "phase8_counterfactual_or_blocked_summary.json"},
        {"question_id": 10, "question": "Was runtime action allowed and executed?", "answer": False, "evidence": "runtime_action_allowed=false runtime_action_executed=false"},
        {"question_id": 11, "question": "Was TTT allowed?", "answer": False, "evidence": "ttt_allowed=false"},
        {"question_id": 12, "question": "If No-Go, what exact blocker remains?", "answer": decision["blocker"], "evidence": ",".join(decision["decision_labels"])},
    ]
    decision["final_questions"] = final_questions
    write_json(args.out_dir / "final_decision.json", decision)
    write_csv(args.out_dir / "decision_labels.csv", [{"label": item} for item in decision["decision_labels"]])
    write_csv(args.out_dir / "blocker_attribution.csv", [{"blocker": decision["blocker"], "final_status": final_status}])
    write_csv(args.out_dir / "final_decision_answers.csv", final_questions)
    report = [
        "# ACL2 v91 Final Report",
        "",
        f"- final_status: `{final_status}`",
        f"- blocker: `{decision['blocker']}`",
        f"- phase1_tracklet_audit_gate_pass: `{decision['phase1_tracklet_audit_gate_pass']}`",
        f"- phase2_regime_classifier_audit_gate_pass: `{decision['phase2_regime_classifier_audit_gate_pass']}`",
        f"- phase3_regime_semantic_gate_pass: `{decision['phase3_regime_semantic_gate_pass']}`",
        f"- phase4_tracklet_mode_gate_pass: `{decision['phase4_tracklet_mode_gate_pass']}`",
        f"- phase5_memory_update_policy_gate_pass: `{decision['phase5_memory_update_policy_gate_pass']}`",
        f"- phase6_delayed_commit_gate_pass: `{decision['phase6_delayed_commit_gate_pass']}`",
        f"- phase7_carrier_gate_pass: `{decision['phase7_carrier_gate_pass']}`",
        f"- runtime_action_allowed: `{decision['runtime_action_allowed']}`",
        f"- ttt_allowed: `{decision['ttt_allowed']}`",
        f"- visual_integrity_gate_pass: `{decision['visual_integrity_gate_pass']}`",
        "",
        "## Final Questions",
        "",
        *[f"{row['question_id']}. {row['question']} `{row['answer']}` Evidence: {row['evidence']}" for row in final_questions],
        "",
        "Conclusion: v91 only claims the gates supported by the generated JSON/CSV/visual artifacts. Runtime action and TTT remain false unless a verified carrier and counterfactual route pass.",
    ]
    (args.out_dir / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"final_status={final_status}")
    print(f"blocker={decision['blocker']}")
    print(f"phase3_regime_semantic_gate_pass={decision['phase3_regime_semantic_gate_pass']}")
    print(f"phase5_memory_update_policy_gate_pass={decision['phase5_memory_update_policy_gate_pass']}")
    print(f"phase6_delayed_commit_gate_pass={decision['phase6_delayed_commit_gate_pass']}")
    print(f"phase7_carrier_gate_pass={decision['phase7_carrier_gate_pass']}")
    print(f"runtime_action_allowed={decision['runtime_action_allowed']}")
    print(f"ttt_allowed={decision['ttt_allowed']}")


if __name__ == "__main__":
    main()
