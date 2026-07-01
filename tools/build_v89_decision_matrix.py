#!/usr/bin/env python3
"""Build v89 final decision matrix from audited phase artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "phase10_decision_matrix")
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _best(summary: dict[str, Any]) -> dict[str, Any]:
    return dict(summary.get("best_semantic_signal") or {})


def _split_summaries(root: Path) -> dict[str, dict[str, Any]]:
    return {
        "all": _json(root / "phase2_semantic_mode_relevance/semantic_mode_relevance_summary.json"),
        "highobs": _json(root / "phase2_semantic_mode_relevance_highobs/semantic_mode_relevance_summary.json"),
        "nonseq01": _json(root / "phase2_semantic_mode_relevance_nonseq01/semantic_mode_relevance_summary.json"),
        "near": _json(root / "phase2_semantic_mode_relevance_near/semantic_mode_relevance_summary.json"),
        "far": _json(root / "phase2_semantic_mode_relevance_far/semantic_mode_relevance_summary.json"),
        "semantic_structure_rich": _json(root / "phase2_semantic_mode_relevance_semantic_structure_rich/semantic_mode_relevance_summary.json"),
        "semantic_lowobs": _json(root / "phase2_semantic_mode_relevance_semantic_lowobs/semantic_mode_relevance_summary.json"),
    }


def _best_split(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split, summary in summaries.items():
        best = _best(summary)
        if not best:
            continue
        best["split"] = split
        best["split_gate_pass"] = summary.get("phase2_semantic_mode_relevance_gate_pass")
        best["passing_semantic_signals"] = summary.get("passing_semantic_signals")
        best["geometry_reference_signal"] = summary.get("geometry_reference_signal")
        best["geometry_reference_rho"] = summary.get("geometry_reference_rho")
        rows.append(best)
    rows = [row for row in rows if row.get("spearman_rho_abs_log_scale_jump") is not None]
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda row: (
            bool(row.get("split_gate_pass")),
            float(row.get("spearman_rho_abs_log_scale_jump") or -9.0),
            float(row.get("semantic_shuffle_margin") or -9.0),
        ),
        reverse=True,
    )[0]


def main() -> None:
    args = parse_args()
    root = args.root
    phase0 = _json(root / "phase0_evidence_lock/phase0_gate_summary.json")
    phase1 = _json(root / "phase1_semantic_scale_mode_ledger/phase1_semantic_ledger_summary.json")
    phase1_audit = _json(root / "phase1_semantic_scale_mode_ledger/phase1_semantic_ledger_audit.json")
    phase2 = _split_summaries(root)
    phase3_build = _json(root / "phase3_feature_match_semantic_ruler/feature_match_build_summary.json")
    phase3 = _json(root / "phase3_feature_match_semantic_ruler/feature_match_audit_summary.json")
    phase4_build = _json(root / "phase4_semantic_observability_policy/semantic_observability_policy_build_summary.json")
    phase4 = _json(root / "phase4_semantic_observability_policy/semantic_observability_policy_audit_summary.json")
    phase7_build = _json(root / "phase7_semantic_mode_temporal_consistency/semantic_mode_temporal_consistency_summary.json")
    phase7 = _json(root / "phase7_semantic_mode_temporal_consistency/delayed_commit_policy_audit_summary.json")
    visual = _json(root / "phase10_visual_rediscovery/visual_integrity_audit.json")

    p2_all = phase2["all"]
    best_all = _best(p2_all)
    best_split = _best_split({k: v for k, v in phase2.items() if k != "all"})
    phase2_global_pass = bool(p2_all.get("phase2_semantic_mode_relevance_gate_pass"))
    phase2_any_split_pass = any(bool(v.get("phase2_semantic_mode_relevance_gate_pass")) for k, v in phase2.items() if k != "all")
    phase3_pass = bool(phase3.get("feature_match_semantic_ruler_gate_pass"))
    phase4_pass = bool(phase4.get("semantic_observability_policy_gate_pass"))
    phase7_pass = bool(phase7.get("delayed_commit_policy_gate_pass"))
    visual_pass = bool(visual.get("visual_integrity_gate_pass"))

    carrier_tools_run = bool(phase2_global_pass and phase3_pass and phase4_pass)
    counterfactual_tools_run = False
    runtime_action_allowed = False
    ttt_allowed = False
    final_no_go_allowed = bool(visual_pass and not runtime_action_allowed)

    active_labels: list[str] = []
    if not phase2_global_pass:
        active_labels.append("D1_GEOMETRY_MODE_SIGNAL_ONLY_SEMANTIC_NO_ADD")
    if phase4.get("semantic_good_protection_margin", 0) and not phase4_pass:
        active_labels.append("D5_OBSERVABILITY_HOLD_DIAGNOSTIC_ONLY")
    if not carrier_tools_run:
        active_labels.append("D6_CARRIER_ATTRIBUTION_NOT_ENTERED_PRECONDITIONS_FAILED")
    if not counterfactual_tools_run:
        active_labels.append("D8_COUNTERFACTUAL_UPPER_BOUND_NOT_ENTERED_NO_CARRIER")
    active_labels.extend(["D10_TTT_NOT_READY", "D11_SEMANTIC_NOT_SPECIFIC_CURRENT_DEFINITION", "D12_NO_GO_BEFORE_RUNTIME_ACTION"])

    blocker = (
        "global_semantic_relevance_failed_feature_match_not_scale_relevant_observability_bad_recall_delayed_commit_unsafe"
        if final_no_go_allowed
        else "visual_gate_incomplete_final_decision_blocked"
    )
    final_status = "No-Go_before_runtime_action" if final_no_go_allowed else "Incomplete_visual_audit_required"

    key_metrics = {
        "phase0_gate_pass": phase0.get("phase0_gate_pass"),
        "phase0_source": phase0.get("source"),
        "phase0_result_tree_unavailable": phase0.get("result_tree_unavailable"),
        "phase1_gate_pass": phase1.get("phase1_gate_pass"),
        "phase1_audit_gate_pass": phase1_audit.get("phase1_audit_gate_pass"),
        "phase1_pair_rows": phase1.get("pair_rows"),
        "phase1_edge_rows": phase1.get("edge_rows"),
        "phase1_sequence_coverage": phase1.get("sequence_coverage"),
        "phase1_semantic_mode_type_counts": phase1_audit.get("semantic_mode_type_counts"),
        "phase2_global_gate_pass": p2_all.get("phase2_semantic_mode_relevance_gate_pass"),
        "phase2_global_passing_semantic_signals": p2_all.get("passing_semantic_signals"),
        "phase2_global_geometry_reference_signal": p2_all.get("geometry_reference_signal"),
        "phase2_global_geometry_reference_rho": p2_all.get("geometry_reference_rho"),
        "phase2_global_best_semantic_signal": best_all.get("signal"),
        "phase2_global_best_semantic_rho": best_all.get("spearman_rho_abs_log_scale_jump"),
        "phase2_global_best_semantic_margin": best_all.get("semantic_shuffle_margin"),
        "phase2_global_best_semantic_recall": best_all.get("bad_recall"),
        "phase2_global_best_semantic_good_fpr": best_all.get("good_false_positive_rate"),
        "phase2_any_split_gate_pass": phase2_any_split_pass,
        "phase2_best_split": best_split.get("split"),
        "phase2_best_split_gate_pass": best_split.get("split_gate_pass"),
        "phase2_best_split_signal": best_split.get("signal"),
        "phase2_best_split_rho": best_split.get("spearman_rho_abs_log_scale_jump"),
        "phase2_best_split_margin": best_split.get("semantic_shuffle_margin"),
        "phase2_best_split_recall": best_split.get("bad_recall"),
        "phase2_best_split_good_fpr": best_split.get("good_false_positive_rate"),
        "phase3_matcher_available": phase3.get("matcher_available"),
        "phase3_matcher_type": phase3.get("matcher_type"),
        "phase3_build_verified_inlier_count_median": phase3_build.get("verified_inlier_count_median"),
        "phase3_gate_pass": phase3.get("feature_match_semantic_ruler_gate_pass"),
        "phase3_match_semantic_valid_ratio_median": phase3.get("match_semantic_valid_ratio_median"),
        "phase3_match_valid_score_rho": phase3.get("match_valid_score_rho_abs_log_scale_jump"),
        "phase3_semantic_shuffle_match_margin": phase3.get("semantic_shuffle_match_margin"),
        "phase4_build_state_counts": phase4_build.get("state_counts"),
        "phase4_gate_pass": phase4.get("semantic_observability_policy_gate_pass"),
        "phase4_bad_recall": phase4.get("bad_recall"),
        "phase4_good_FPR": phase4.get("good_FPR"),
        "phase4_semantic_good_protection_margin": phase4.get("semantic_good_protection_margin"),
        "phase4_semantic_shuffle_margin": phase4.get("semantic_shuffle_margin"),
        "phase7_build_delayed_commit_count": phase7_build.get("delayed_commit_count"),
        "phase7_build_persistent_rows": phase7_build.get("persistent_rows"),
        "phase7_gate_pass": phase7.get("delayed_commit_policy_gate_pass"),
        "phase7_bad_recall": phase7.get("bad_recall"),
        "phase7_good_FPR": phase7.get("good_FPR"),
        "phase7_semantic_shuffle_margin": phase7.get("semantic_shuffle_margin"),
        "visual_integrity_gate_pass": visual.get("visual_integrity_gate_pass"),
        "visual_manifest_rows": visual.get("manifest_rows"),
        "visual_question_rows": visual.get("question_rows"),
        "visual_review_coverage": visual.get("review_coverage"),
        "visual_no_fake_route_runtime_panels": visual.get("no_fake_route_runtime_panels"),
    }

    decision = {
        "phase": "Phase10_decision_matrix",
        "final_status": final_status,
        "final_no_go_allowed": final_no_go_allowed,
        "blocker": blocker,
        "active_decision_labels": active_labels,
        "runtime_action_allowed": runtime_action_allowed,
        "runtime_action_executed": False,
        "ttt_allowed": ttt_allowed,
        "carrier_tools_run": carrier_tools_run,
        "counterfactual_tools_run": counterfactual_tools_run,
        "key_metrics": key_metrics,
        "route_eligibility": {
            "semantic_mode_relevance_global_pass": phase2_global_pass,
            "feature_match_semantic_ruler_pass": phase3_pass,
            "semantic_observability_policy_pass": phase4_pass,
            "delayed_commit_policy_pass": phase7_pass,
            "carrier_attribution_entry_allowed": carrier_tools_run,
            "counterfactual_entry_allowed": counterfactual_tools_run,
            "runtime_entry_allowed": runtime_action_allowed,
            "skip_reason": "Phase2 global, Phase3, and Phase4 did not pass together; Phase5/6/8/9 are blocked by plan preconditions.",
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "decision_matrix.json", decision)
    write_csv(args.out_dir / "decision_labels.csv", [{"decision_label": label, "active": True} for label in active_labels])
    write_csv(
        args.out_dir / "blocker_attribution.csv",
        [
            {
                "blocker": "phase2_global_semantic_relevance_failed",
                "evidence": f"global pass={phase2_global_pass}; best={best_all.get('signal')} rho={best_all.get('spearman_rho_abs_log_scale_jump')} margin={best_all.get('semantic_shuffle_margin')}; geometry={p2_all.get('geometry_reference_signal')} rho={p2_all.get('geometry_reference_rho')}",
                "repair_attempted": "semantic role repair, highobs/nonseq01/near/far/structure-rich/lowobs splits, semantic shuffle controls",
                "status": "active_global; highobs/far split diagnostic only",
            },
            {
                "blocker": "feature_match_semantic_ruler_not_scale_relevant",
                "evidence": f"matcher={phase3.get('matcher_type')} valid_ratio={phase3.get('match_semantic_valid_ratio_median')} rho={phase3.get('match_valid_score_rho_abs_log_scale_jump')} shuffle_margin={phase3.get('semantic_shuffle_match_margin')}",
                "repair_attempted": "LightGlue-SIFT sparse matches with semantic-valid/cross-boundary filtering",
                "status": "active",
            },
            {
                "blocker": "observability_policy_not_safe",
                "evidence": f"bad_recall={phase4.get('bad_recall')} good_FPR={phase4.get('good_FPR')} protection_margin={phase4.get('semantic_good_protection_margin')} shuffle_margin={phase4.get('semantic_shuffle_margin')}",
                "repair_attempted": "UPDATE/HOLD/ABSTAIN/REJECT/DELAY deterministic policy with semantic shuffle comparison",
                "status": "active; good protection diagnostic but gate failed",
            },
            {
                "blocker": "delayed_commit_policy_not_safe",
                "evidence": f"bad_recall={phase7.get('bad_recall')} good_FPR={phase7.get('good_FPR')} shuffle_margin={phase7.get('semantic_shuffle_margin')}",
                "repair_attempted": "temporal consistency and delayed commit/hysteresis diagnostic",
                "status": "active",
            },
            {
                "blocker": "carrier_counterfactual_runtime_not_entered",
                "evidence": "Phase2 global, Phase3, and Phase4 did not pass together; carrier and counterfactual tools are blocked by the plan.",
                "repair_attempted": "no unsafe bypass; visual rediscovery completed with blocked placeholders",
                "status": "plan-compliant skip",
            },
        ],
    )
    (args.out_dir / "next_route_recommendation.md").write_text(
        "\n".join(
            [
                "# v89 Next Route Recommendation",
                "",
                "1. Do not run runtime memory action or TTT from current v89 evidence.",
                "2. Keep highobs/far Phase2 passes as diagnostic split evidence only; they are not no-GT runtime triggers by themselves.",
                "3. Current compact semantic labels can produce valid-support modes after role repair, but they are not specific enough to beat geometry/shuffle globally.",
                "4. Sparse feature matches are semantically valid, yet their scale relevance is negative; use them as a sanity check, not a carrier.",
                "5. If continuing this route, improve semantic source/ontology or object topology before rerunning carrier/counterfactual.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"final_status={decision['final_status']}")
    print(f"final_no_go_allowed={decision['final_no_go_allowed']}")
    print(f"runtime_action_allowed={decision['runtime_action_allowed']}")
    print(f"ttt_allowed={decision['ttt_allowed']}")
    print(f"carrier_tools_run={decision['carrier_tools_run']}")
    print(f"counterfactual_tools_run={decision['counterfactual_tools_run']}")
    print(f"active_decision_labels={decision['active_decision_labels']}")
    print(f"blocker={decision['blocker']}")


if __name__ == "__main__":
    main()
