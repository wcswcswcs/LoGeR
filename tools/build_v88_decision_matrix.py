#!/usr/bin/env python3
"""Build v88 Phase8 decision matrix from audited phase artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "phase8_decision_matrix")
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _signal(summary: dict[str, Any], name: str) -> dict[str, Any]:
    for row in summary.get("signals") or []:
        if row.get("signal") == name:
            return row
    return {}


def _best_split(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split, summary in summaries.items():
        best = dict(summary.get("best_signal") or {})
        if best:
            best["split"] = split
            best["split_gate_pass"] = summary.get("phase2_mode_relevance_gate_pass")
            best["passing_signals"] = summary.get("passing_signals")
            best["semantic_aware_pass"] = summary.get("semantic_aware_pass")
            rows.append(best)
    rows = [row for row in rows if row.get("spearman_rho_abs_log_scale_jump") is not None]
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda row: (bool(row.get("split_gate_pass")), float(row.get("spearman_rho_abs_log_scale_jump") or -9), float(row.get("rho_margin_vs_shuffled") or -9)),
        reverse=True,
    )[0]


def main() -> None:
    args = parse_args()
    root = args.root
    phase0 = _json(root / "phase0_evidence_lock/phase0_gate_summary.json")
    phase1 = _json(root / "phase1_scale_mode_consensus_universe/phase1_gate_summary.json")
    phase2_all = _json(root / "phase2_scale_mode_relevance/scale_mode_relevance_summary.json")
    phase2_splits = {
        "all": phase2_all,
        "highobs": _json(root / "phase2_scale_mode_relevance_highobs/scale_mode_relevance_summary.json"),
        "nonseq01": _json(root / "phase2_scale_mode_relevance_nonseq01/scale_mode_relevance_summary.json"),
        "near": _json(root / "phase2_scale_mode_relevance_near/scale_mode_relevance_summary.json"),
        "far": _json(root / "phase2_scale_mode_relevance_far/scale_mode_relevance_summary.json"),
    }
    phase3 = _json(root / "phase3_native_gauge_update_attribution/native_gauge_update_attribution_summary.json")
    phase4_swa = _json(root / "phase4_swa_mode_route_audit/swa_mode_route_audit_summary.json")
    phase4_merge = _json(root / "phase4_merge_gauge_mode_carrier/merge_gauge_mode_carrier_summary.json")
    phase5 = _json(root / "phase5_mode_aware_counterfactual/mode_aware_counterfactual_summary.json")
    visual = _json(root / "phase7_visual_rediscovery/visual_integrity_audit.json")

    global_phase2_pass = bool(phase2_all.get("phase2_mode_relevance_gate_pass"))
    any_split_phase2_pass = any(bool(s.get("phase2_mode_relevance_gate_pass")) for s in phase2_splits.values())
    semantic_any_pass = any(bool(s.get("semantic_aware_pass")) for s in phase2_splits.values())
    phase3_pass = bool(phase3.get("phase3_native_update_attribution_gate_pass"))
    swa_pass = bool(phase4_swa.get("swa_route_carrier_gate_pass"))
    merge_pass = bool(phase4_merge.get("merge_gauge_mode_carrier_gate_pass"))
    cf_pass = bool(phase5.get("scale_label_gate_pass") and phase5.get("raw_residual_gate_pass"))
    runtime_candidate = bool(global_phase2_pass and phase3_pass and (swa_pass or merge_pass) and cf_pass)
    final_no_go_allowed = bool(visual.get("visual_integrity_gate_pass"))

    labels: list[str] = []
    if not global_phase2_pass:
        labels.append("D1_NO_SCALE_MODE_SIGNAL")
    if any_split_phase2_pass and not semantic_any_pass:
        labels.append("D2_GEOMETRY_MODE_ONLY_SEMANTIC_NO_ADD")
    if phase3_pass:
        labels.append("D3_NATIVE_UPDATE_MISMATCH_EXPLAINS_BAD")
    else:
        labels.append("D4_NATIVE_UPDATE_MISMATCH_UNSAFE")
    if swa_pass:
        labels.append("D5_SWA_MODE_ROUTE_CARRIER_PASS")
    else:
        labels.append("D6_SWA_MODE_ROUTE_NOT_CARRIER")
    if merge_pass:
        labels.append("D7_MERGE_GAUGE_MODE_CARRIER_PASS")
    if not cf_pass:
        labels.append("D8_MERGE_GAUGE_MODE_COUNTERFACTUAL_FAIL")
    if runtime_candidate:
        labels.append("D9_RUNTIME_CANDIDATE")
    else:
        labels.append("D10_TTT_NOT_READY")

    best_all = phase2_all.get("best_signal") or {}
    best_split = _best_split(phase2_splits)
    best_variant = phase3.get("best_variant") or {}
    best_family = phase5.get("best_family") or {}
    blocker = (
        "no_global_mode_signal_native_mismatch_not_attributive_carrier_absent_counterfactual_good_protection_fail"
        if final_no_go_allowed
        else "visual_gate_not_complete_final_no_go_not_allowed"
    )
    final_status = "No-Go_before_runtime_action" if final_no_go_allowed else "Incomplete_visual_audit_required"
    key_metrics = {
        "phase0_gate_pass": phase0.get("phase0_gate_pass"),
        "phase1_gate_pass": phase1.get("phase1_gate_pass"),
        "phase1_pair_rows": phase1.get("pair_rows"),
        "phase1_sequence_coverage": phase1.get("sequence_coverage"),
        "phase2_global_gate_pass": phase2_all.get("phase2_mode_relevance_gate_pass"),
        "phase2_global_passing_signals": phase2_all.get("passing_signals"),
        "phase2_global_best_signal": best_all.get("signal"),
        "phase2_global_best_rho": best_all.get("spearman_rho_abs_log_scale_jump"),
        "phase2_global_best_recall": best_all.get("high_scale_jump_recall"),
        "phase2_global_best_good_low_fpr": best_all.get("good_low_scale_fpr"),
        "phase2_best_split": best_split.get("split"),
        "phase2_best_split_signal": best_split.get("signal"),
        "phase2_best_split_rho": best_split.get("spearman_rho_abs_log_scale_jump"),
        "phase2_best_split_margin": best_split.get("rho_margin_vs_shuffled"),
        "phase2_best_split_passing_signals": best_split.get("passing_signals"),
        "phase2_semantic_any_pass": semantic_any_pass,
        "phase3_gate_pass": phase3.get("phase3_native_update_attribution_gate_pass"),
        "phase3_best_variant": best_variant.get("variant"),
        "phase3_mismatch_bad_recall": best_variant.get("MISMATCH_BAD_recall"),
        "phase3_mismatch_good_fpr": best_variant.get("MISMATCH_GOOD_FPR"),
        "phase3_mismatch_rho": best_variant.get("native_mode_mismatch_rho_abs_log_scale_jump"),
        "phase3_rho_margin_vs_shape_shuffle": best_variant.get("rho_margin_vs_shape_shuffle"),
        "phase4_swa_gate_pass": phase4_swa.get("swa_route_carrier_gate_pass"),
        "phase4_swa_blocker": phase4_swa.get("blocker"),
        "phase4_merge_gate_pass": phase4_merge.get("merge_gauge_mode_carrier_gate_pass"),
        "phase4_merge_blocker": phase4_merge.get("blocker"),
        "phase5_scale_label_gate_pass": phase5.get("scale_label_gate_pass"),
        "phase5_raw_residual_counterfactual_available": phase5.get("raw_residual_counterfactual_available"),
        "phase5_raw_residual_gate_pass": phase5.get("raw_residual_gate_pass"),
        "phase5_best_family": best_family.get("family"),
        "phase5_best_bad_median_I_scale": best_family.get("bad_median_I_scale"),
        "phase5_best_good_max_scale_error_worsen": best_family.get("good_max_scale_error_worsen"),
        "phase5_control_bad_best_median_I_scale": phase5.get("control_bad_best_median_I_scale"),
        "visual_integrity_gate_pass": visual.get("visual_integrity_gate_pass"),
        "visual_manifest_rows": visual.get("manifest_rows"),
        "visual_question_rows": visual.get("question_rows"),
        "visual_review_coverage": visual.get("review_coverage"),
    }
    decision = {
        "phase": "Phase8_decision_matrix",
        "final_status": final_status,
        "final_no_go_allowed": final_no_go_allowed,
        "blocker": blocker,
        "active_decision_labels": labels,
        "runtime_action_allowed": runtime_candidate,
        "ttt_allowed": False,
        "key_metrics": key_metrics,
        "route_eligibility": {
            "swa_runtime_route": bool(global_phase2_pass and phase3_pass and swa_pass and cf_pass),
            "merge_gauge_runtime_route": bool(global_phase2_pass and phase3_pass and merge_pass and cf_pass),
            "phase6_runtime_skipped": not runtime_candidate,
            "skip_reason": "Phase2 global, Phase3, Phase4, and Phase5 did not all pass for any route.",
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "decision_matrix.json", decision)
    write_csv(
        args.out_dir / "decision_labels.csv",
        [{"decision_label": label, "active": True} for label in labels],
    )
    write_csv(
        args.out_dir / "blocker_attribution.csv",
        [
            {
                "blocker": "phase2_global_mode_relevance_failed",
                "evidence": f"global passing_signals={phase2_all.get('passing_signals')} best={best_all.get('signal')} rho={best_all.get('spearman_rho_abs_log_scale_jump')}",
                "repair_attempted": "highobs, nonseq01, near, far splits plus mode entropy/gap/sign variants",
                "status": "active_global; split_diagnostic_only" if any_split_phase2_pass else "active",
            },
            {
                "blocker": "semantic_no_add",
                "evidence": f"semantic_any_pass={semantic_any_pass}; highobs semantic_aware_pass={phase2_splits['highobs'].get('semantic_aware_pass')}",
                "repair_attempted": "semantic-aware signal compared against geometry-only and shuffled semantic controls",
                "status": "active",
            },
            {
                "blocker": "native_mode_mismatch_attribution_failed",
                "evidence": f"phase3_pass={phase3_pass}; recall={best_variant.get('MISMATCH_BAD_recall')} FPR={best_variant.get('MISMATCH_GOOD_FPR')} rho={best_variant.get('native_mode_mismatch_rho_abs_log_scale_jump')}",
                "repair_attempted": "residual-low, entropy/gap, highobs, semantic-stable, lowobs abstain, combined guards",
                "status": "active",
            },
            {
                "blocker": "carrier_absent",
                "evidence": f"SWA blocker={phase4_swa.get('blocker')}; merge blocker={phase4_merge.get('blocker')}",
                "repair_attempted": "separate SWA mode-route and merge/gauge carrier audits",
                "status": "active",
            },
            {
                "blocker": "mode_counterfactual_failed",
                "evidence": f"best_family={best_family.get('family')} bad_I={best_family.get('bad_median_I_scale')} good_worsen={best_family.get('good_max_scale_error_worsen')} raw_available={phase5.get('raw_residual_counterfactual_available')}",
                "repair_attempted": "dominant, trimmed, robust, hold, multimode-abstain, semantic-guarded, random/shuffle controls",
                "status": "active",
            },
        ],
    )
    (args.out_dir / "next_route_recommendation.md").write_text(
        "\n".join(
            [
                "# v88 Next Route Recommendation",
                "",
                "1. Do not run Phase6 runtime or TTT from current v88 evidence.",
                "2. Treat scale-mode statistics as diagnostic split evidence only; the global Phase2 gate failed.",
                "3. Build a per-head/per-layer SWA route dump before making any SWA carrier claim.",
                "4. For merge/gauge, require raw residual counterfactual evidence and good-row protection before runtime.",
                "5. Future work should pre-register transition-regime splits; post-hoc split success is not a runtime trigger.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"final_status={decision['final_status']}")
    print(f"final_no_go_allowed={decision['final_no_go_allowed']}")
    print(f"runtime_action_allowed={decision['runtime_action_allowed']}")
    print(f"ttt_allowed={decision['ttt_allowed']}")
    print(f"active_decision_labels={decision['active_decision_labels']}")
    print(f"blocker={decision['blocker']}")


if __name__ == "__main__":
    main()
