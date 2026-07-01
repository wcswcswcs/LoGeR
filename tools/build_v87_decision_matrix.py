#!/usr/bin/env python3
"""Build v87 Phase11 decision matrix and final report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from v86_soft_latent_utils import read_json, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_ROOT / "phase1_scale_conditioned_pair_universe_k16_r1_median_abs")
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_ROOT / "phase2_scale_relevance_k16_r1_median_abs_highobs")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "phase11_decision_matrix")
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _signal(summary: dict[str, Any], name: str) -> dict[str, Any]:
    for row in summary.get("signals") or []:
        if row.get("signal") == name:
            return row
    return {}


def main() -> None:
    args = parse_args()
    phase0 = _json(args.root / "phase0_evidence_lock/phase0_gate_summary.json")
    phase1_default = _json(args.root / "phase1_scale_conditioned_pair_universe/phase1_gate_summary.json")
    phase1 = _json(args.phase1_dir / "phase1_gate_summary.json")
    phase2 = _json(args.phase2_dir / "proxy_relevance_summary.json")
    phase3 = _json(args.root / "phase3_state_conditioned_latent_transport/state_conditioned_alignment_summary.json")
    phase4 = _json(args.root / "phase4_no_refresh_guard/no_refresh_guard_summary.json")
    phase8 = _json(args.root / "phase8_merge_gauge_direct_pair_weighting/merge_gauge_direct_pair_summary.json")
    visual = _json(args.root / "phase12_visual_rediscovery/visual_integrity_audit.json")

    s_shape = _signal(phase2, "S_shape")
    s_overlap = _signal(phase2, "S_overlap")
    s_geometry = _signal(phase2, "S_geometry_only")
    active_labels = []
    if phase2.get("phase2_scale_proxy_gate_pass") and phase2.get("geometry_only_pass") and not phase2.get("semantic_signal_pass"):
        active_labels.append("D2_SCALE_PROXY_GEOMETRY_ONLY_SEMANTIC_NO_ADD")
    if not phase3.get("phase3_alignment_gate_pass"):
        active_labels.append("D5_SWA_QK_NOT_SCALE_GAUGE_CARRIER")
    if not phase8.get("phase8_merge_gauge_gate_pass"):
        active_labels.append("D8_MERGE_GAUGE_DIRECT_PAIR_UPPER_BOUND_FAIL")
    active_labels.append("D9_TTT_NOT_READY")
    final_no_go_allowed = bool(visual.get("visual_integrity_gate_pass"))
    final_status = "No-Go_before_runtime_action" if final_no_go_allowed else "Incomplete_visual_audit_required"
    blocker = "geometry_only_scale_proxy_but_no_support_C_no_safe_no_refresh_raw_overlap_direct_pair_counterfactual_failed_no_runtime_merge_geometry"

    key_metrics = {
        "phase0_gate_pass": phase0.get("phase0_gate_pass"),
        "phase1_default_gate_pass": phase1_default.get("phase1_gate_pass"),
        "phase1_selected_gate_pass": phase1.get("phase1_gate_pass"),
        "phase1_selected_raw_shape_availability_high_quality": phase1.get("raw_overlap_local_shape_proxy_availability_high_quality"),
        "phase1_selected_support_or_conflict_effective_pairs": phase1.get("support_or_conflict_effective_pairs"),
        "phase2_highobs_gate_pass": phase2.get("phase2_scale_proxy_gate_pass"),
        "phase2_S_shape_rho": s_shape.get("spearman_rho_abs_log_scale_jump"),
        "phase2_S_shape_good_low_fpr": s_shape.get("good_low_scale_fpr"),
        "phase2_S_overlap_rho": s_overlap.get("spearman_rho_abs_log_scale_jump"),
        "phase2_S_overlap_good_low_fpr": s_overlap.get("good_low_scale_fpr"),
        "phase2_S_geometry_only_rho": s_geometry.get("spearman_rho_abs_log_scale_jump"),
        "phase2_semantic_signal_pass": phase2.get("semantic_signal_pass"),
        "phase3_valid_rows": phase3.get("valid_rows"),
        "phase3_blocker": phase3.get("blocker"),
        "phase4_bad_recall": phase4.get("bad_recall"),
        "phase4_good_FPR": phase4.get("good_FPR"),
        "phase4_prior_available_rows": phase4.get("prior_available_rows"),
        "phase8_actual_geometry_counterfactual_available": phase8.get("actual_geometry_counterfactual_available"),
        "phase8_raw_overlap_geometry_counterfactual_available": phase8.get("raw_overlap_geometry_counterfactual_available"),
        "phase8_raw_overlap_gate_pass": phase8.get("phase8_raw_overlap_geometry_gate_pass"),
        "phase8_raw_overlap_valid_rows": phase8.get("raw_overlap_valid_rows"),
        "phase8_raw_overlap_sequence_coverage": phase8.get("raw_overlap_sequence_coverage"),
        "phase8_bad_raw_overlap_median_improvement_vs_native": phase8.get("bad_raw_overlap_median_improvement_vs_native"),
        "phase8_good_raw_overlap_median_worsen_vs_native": phase8.get("good_raw_overlap_median_worsen_vs_native"),
        "phase8_raw_overlap_control_margins": phase8.get("raw_overlap_control_margins"),
        "phase8_bad_proxy_recall": phase8.get("bad_proxy_recall"),
        "phase8_good_proxy_fpr": phase8.get("good_proxy_fpr"),
        "visual_manifest_rows": visual.get("manifest_rows"),
        "visual_integrity_gate_pass": visual.get("visual_integrity_gate_pass"),
    }
    final = {
        "phase": "Phase11_decision_matrix",
        "final_status": final_status,
        "final_no_go_allowed": final_no_go_allowed,
        "blocker": blocker,
        "active_decision_labels": active_labels,
        "key_metrics": key_metrics,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "selected_phase1_dir": str(args.phase1_dir),
        "selected_phase2_dir": str(args.phase2_dir),
        "phase3_gate_pass": phase3.get("phase3_alignment_gate_pass"),
        "phase4_gate_pass": phase4.get("phase4_no_refresh_guard_gate_pass"),
        "phase8_gate_pass": phase8.get("phase8_merge_gauge_gate_pass"),
        "visual_gate_pass": visual.get("visual_integrity_gate_pass"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "final_decision.json", final)
    write_csv(
        args.out_dir / "blocker_attribution.csv",
        [
            {
                "blocker": "phase2_default_not_enough",
                "evidence": "default S_shape/S_overlap recall positive but good-low-scale FPR too high",
                "resolved_by": "high_observability_only repair",
                "status": "partial_resolved_geometry_only",
            },
            {
                "blocker": "semantic_no_add",
                "evidence": "semantic_signal_pass=false in selected Phase2",
                "resolved_by": "",
                "status": "active",
            },
            {
                "blocker": "support_state_absent_no_C_fit",
                "evidence": "Phase3 valid_rows=0 invalid_reason=no_support_state_rows",
                "resolved_by": "",
                "status": "active",
            },
            {
                "blocker": "no_refresh_guard_not_safe",
                "evidence": "Phase4 bad_recall=1.0 but good_FPR=1.0 and prior_available_rows=0",
                "resolved_by": "",
                "status": "active",
            },
            {
                "blocker": "merge_gauge_counterfactual_failed",
                "evidence": "Phase8 raw-overlap direct pair counterfactual available but phase8_raw_overlap_geometry_gate_pass=false; actual runtime trajectory geometry counterfactual still unavailable",
                "resolved_by": "",
                "status": "active",
            },
        ],
    )

    (args.out_dir / "scale_proxy_report.md").write_text(
        "\n".join(
            [
                "# Scale Proxy Report",
                "",
                f"- default Phase2 gate: `False` from `{args.root / 'phase2_scale_relevance_k16_r1_median_abs'}`",
                f"- selected repair: `{args.phase2_dir.name}`",
                f"- selected gate pass: `{phase2.get('phase2_scale_proxy_gate_pass')}`",
                f"- geometry_only_pass: `{phase2.get('geometry_only_pass')}`",
                f"- semantic_signal_pass: `{phase2.get('semantic_signal_pass')}`",
                f"- S_shape rho / FPR: `{s_shape.get('spearman_rho_abs_log_scale_jump')} / {s_shape.get('good_low_scale_fpr')}`",
                f"- S_overlap rho / FPR: `{s_overlap.get('spearman_rho_abs_log_scale_jump')} / {s_overlap.get('good_low_scale_fpr')}`",
                "",
                "Conclusion: no-GT scale proxy exists only as high-observability geometry/local-shape evidence. No semantic add-value claim is supported.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "latent_transport_report.md").write_text(
        "\n".join(
            [
                "# Latent Transport Report",
                "",
                f"- phase3 gate pass: `{phase3.get('phase3_alignment_gate_pass')}`",
                f"- valid rows: `{phase3.get('valid_rows')}`",
                f"- blocker: `{phase3.get('blocker')}`",
                f"- invalid reasons: `{phase3.get('invalid_reason_counts')}`",
                "",
                "Conclusion: state-conditioned C was not fit because selected v87 states contain no SUPPORT rows. Conflict rows were not used as positive anchors.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "route_carrier_report.md").write_text(
        "\n".join(
            [
                "# Route Carrier Report",
                "",
                "- Phase5 was not entered because Phase3 and Phase4 did not pass.",
                "- Pooled Q/K diagnostic features were not promoted to per-head route carrier.",
                "- Runtime QK action remains blocked.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "merge_gauge_report.md").write_text(
        "\n".join(
            [
                "# Merge/Gauge Report",
                "",
                f"- phase8 gate pass: `{phase8.get('phase8_merge_gauge_gate_pass')}`",
                f"- actual geometry counterfactual available: `{phase8.get('actual_geometry_counterfactual_available')}`",
                f"- raw-overlap geometry counterfactual available: `{phase8.get('raw_overlap_geometry_counterfactual_available')}`",
                f"- raw-overlap geometry gate pass: `{phase8.get('phase8_raw_overlap_geometry_gate_pass')}`",
                f"- raw-overlap valid rows / sequence coverage: `{phase8.get('raw_overlap_valid_rows')} / {phase8.get('raw_overlap_sequence_coverage')}`",
                f"- bad raw-overlap median improvement vs native: `{phase8.get('bad_raw_overlap_median_improvement_vs_native')}`",
                f"- good raw-overlap median worsen vs native: `{phase8.get('good_raw_overlap_median_worsen_vs_native')}`",
                f"- raw-overlap control margins: `{phase8.get('raw_overlap_control_margins')}`",
                f"- bad proxy recall / good proxy fpr: `{phase8.get('bad_proxy_recall')} / {phase8.get('good_proxy_fpr')}`",
                "",
                "Conclusion: direct raw-pair weights now have a v87 raw-overlap weighted-Sim3 counterfactual, but it failed the gate: bad median residual did not improve by 10%. Old v84 support-map fallback was not reused, and no runtime/HMC trajectory counterfactual is available.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "next_route_recommendation.md").write_text(
        "\n".join(
            [
                "# Next Route Recommendation",
                "",
                "1. Preserve the geometry-only high-observability scale proxy as a control insight, not as semantic memory success.",
                "2. Do not continue generic Q/K C fitting until a SUPPORT state with good-FPR guard exists.",
                "3. Do not promote current direct raw-pair weights: the raw-overlap weighted-Sim3 counterfactual failed the bad-improvement gate.",
                "4. If this route continues, redesign the direct-pair weighting or build an actual runtime/HMC trajectory counterfactual rather than reusing support-map fallback.",
                "5. Generate per-head/per-layer route dumps before any future SWA QK carrier claim.",
                "6. Keep TTT blocked until SWA or direct merge/gauge has confirmed evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report_dir = args.root / "report_final"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "final_decision.json", final)
    final_report_lines = [
        "# ACL2 v87TF Final Report",
        "",
        "## Final Decision",
        "",
        f"- Final status: `{final_status}`",
        f"- Final No-Go allowed: `{final_no_go_allowed}`",
        f"- Blocker: `{blocker}`",
        f"- Active decision labels: `{', '.join(active_labels)}`",
        "- Runtime action allowed: `False`",
        "- TTT allowed: `False`",
        "",
        "## Required Questions",
        "",
        "1. Did no-GT local shape / overlap scale proxy correlate with offline scale jump?",
        f"   Yes, but only after high-observability filtering and only as geometry/local-shape evidence. S_overlap rho={s_overlap.get('spearman_rho_abs_log_scale_jump')} with good-low-scale FPR={s_overlap.get('good_low_scale_fpr')}.",
        "2. Did SUPPORT / CONFLICT / ABSENCE state separate bad/good protected pairs?",
        "   No. Selected Phase1 is conflict/stress dominated and Phase4 risk decisions had good_FPR=1.0.",
        "3. Did semantic information add value beyond geometry-only local shape proxy?",
        f"   No. semantic_signal_pass={phase2.get('semantic_signal_pass')}.",
        "4. Did scale-conditioned C beat random/shuffle on heldout support pairs?",
        "   Not evaluated as a valid C: Phase3 had valid_rows=0 because no SUPPORT state rows existed.",
        "5. Did C preserve conflict mismatch rather than over-aligning conflict pairs?",
        "   Not applicable; C was not fit.",
        "6. Did any Q/K signal correlate with scale jump after conditioning on local shape proxy?",
        "   No Q/K carrier signal passed; pooled Q/K stayed diagnostic-only.",
        "7. Was true per-head/per-layer SWA route carrier available?",
        "   No. Phase5 was not entered and pooled Q/K was not promoted.",
        "8. Did SWA route support/conflict signals beat same-count random and shuffle controls?",
        "   Not run because Phase3/4 carrier entry failed.",
        "9. Did QK counterfactual have a meaningful upper bound?",
        "   Not run; runtime QK remained blocked.",
        "10. If SWA failed, did merge/gauge direct pair weighting have upper bound?",
        f"   No. A raw-overlap direct-pair weighted-Sim3 counterfactual was computed with valid_rows={phase8.get('raw_overlap_valid_rows')} and sequence_coverage={phase8.get('raw_overlap_sequence_coverage')}, but phase8_raw_overlap_geometry_gate_pass={phase8.get('phase8_raw_overlap_geometry_gate_pass')}: bad median improvement vs native={phase8.get('bad_raw_overlap_median_improvement_vs_native')} while the required gate is >=0.10. actual_geometry_counterfactual_available={phase8.get('actual_geometry_counterfactual_available')} for runtime/HMC trajectory geometry.",
        "11. Was TTT kept blocked unless SWA/merge confirmed evidence passed?",
        "   Yes. ttt_allowed=false throughout.",
        "12. If No-Go, is the blocker scale proxy, latent transport, route carrier, action surface, merge/gauge interface, or TTT readiness?",
        "   The blocker chain is: geometry-only scale proxy -> no SUPPORT C fit -> unsafe no-refresh guard -> raw-overlap direct-pair merge/gauge counterfactual failed and runtime trajectory counterfactual is unavailable -> TTT not ready.",
        "",
        "## Visual Evidence",
        "",
        f"- visual_integrity_gate_pass: `{visual.get('visual_integrity_gate_pass')}`",
        f"- manifest_rows: `{visual.get('manifest_rows')}`",
    ]
    (report_dir / "final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")

    print(f"final_status={final_status}")
    print(f"final_no_go_allowed={final_no_go_allowed}")
    print(f"active_decision_labels={active_labels}")


if __name__ == "__main__":
    main()
