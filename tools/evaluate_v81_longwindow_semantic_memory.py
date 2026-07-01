#!/usr/bin/env python3
"""Build final ACL2 v81 long-window semantic memory decision artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REPORT_ROOT = Path("results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/report_final")
DEFAULT_OUT_JSON = REPORT_ROOT / "v81_final_decision.json"
DEFAULT_OUT_MD = REPORT_ROOT / "v81_final_report.md"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _phase_gate(value: Any) -> bool:
    return bool(value)


def _build_decision() -> dict[str, Any]:
    phase0 = _read_json(REPORT_ROOT / "phase0_v80_evidence_lock/v80_evidence_lock.json")
    phase1 = _read_json(REPORT_ROOT / "phase1_long_window_cluster_bank/long_window_cluster_resummary.json")
    phase2 = _read_json(REPORT_ROOT / "phase2_long_window_visual_confirmation/visual_integrity_audit.json")
    phase3 = _read_json(REPORT_ROOT / "phase3_selected_write_risk_rule/bad_good_confusion_matrix.json")
    phase4 = _read_json(REPORT_ROOT / "phase4_read_swa_confirmation/confirmation_quality_summary.json")
    phase6 = _read_json(REPORT_ROOT / "phase6_merge_boundary_typeb_rescue/typeb_rescue_summary.json")
    phase11 = _read_json(REPORT_ROOT / "phase11_rediscovery/visual_integrity_audit.json")
    phase13 = _read_json(
        REPORT_ROOT
        / "phase13_direct_merge_state_projection/typeb_overlap_outlier_projection_bad/direct_merge_state_projection_summary.json"
    )
    phase13_control = _read_json(
        REPORT_ROOT
        / "phase13_control_aware_merge_promotion/typeb_overlap_outlier_random_qscale_gate_bad/control_aware_merge_promotion_summary.json"
    )
    phase12_seq05 = _read_json(
        REPORT_ROOT
        / "phase12_typeb_coverage_seq05_goodprotect/seq05_typeb_goodprotect_summary.json"
    )
    phase14_qscale = _read_json(
        REPORT_ROOT / "phase14_qscale_hold_refresh/qscale_hold_refresh_summary.json"
    )

    phase0_pass = _phase_gate(phase0.get("gate_pass"))
    phase1_pass = _phase_gate(phase1.get("gate_pass"))
    phase2_pass = _phase_gate(phase2.get("gate_pass"))
    phase3_pass = _phase_gate(phase3.get("best_profile_gate_pass"))
    phase4_pass = _phase_gate(phase4.get("gate_pass"))
    phase5_blocked = bool(not (phase0_pass and phase1_pass and phase2_pass and phase3_pass and phase4_pass))
    phase6_pass = _phase_gate(phase6.get("gate_pass"))
    phase7_blocked = bool(not (phase5_blocked is False or phase6_pass))
    phase8_blocked = bool(not phase6_pass)
    phase11_pass = _phase_gate(phase11.get("visual_audit_gate_pass"))
    phase13_pass = _phase_gate(phase13.get("phaseE_gate_pass"))
    phase13_control_pass = _phase_gate(phase13_control.get("phaseE_gate_pass")) and bool(
        phase13_control.get("promotion_pass_chunks")
    )
    phase12_seq05_pass = _phase_gate(phase12_seq05.get("phase12_typeb_seq05_goodprotect_pass"))
    phase14_qscale_pass = _phase_gate(phase14_qscale.get("phase14_qscale_hold_refresh_pass"))

    final_status = (
        "No-Go_method_not_achieved_with_rediscovery_complete"
        if phase11_pass
        else "No-Go_pending_phase11_visual_audit"
    )
    method_gate_claimed = False
    return {
        "schema": "acl2_v81_final_decision_v1",
        "final_status": final_status,
        "method_gate_claimed": method_gate_claimed,
        "v81_goal_achieved": False,
        "training_free": True,
        "official_704F_success": False,
        "official_704F_run": False,
        "official_704F_blocker": "Phase5/6/7 method gates did not pass; Phase8 official entry is only allowed if Phase5/6/7 pass.",
        "phase_gates": {
            "phase0_v80_evidence_lock": phase0_pass,
            "phase1_long_window_bank": phase1_pass,
            "phase2_visual_confirmation": phase2_pass,
            "phase3_selected_write_risk_rule": phase3_pass,
            "phase4_read_swa_confirmation": phase4_pass,
            "phase5_ttt_write_less_onehop": "blocked_by_phase3_phase4",
            "phase6_typeb_merge_boundary_rescue": phase6_pass,
            "phase7_cross_memory_handshake": "blocked_by_missing_action_gate",
            "phase8_heldout_official_validation": "blocked_by_missing_phase5_6_7_pass",
            "phase9_radio_sidecar_expansion": "not_promoted_radio_unavailable_for_multiseq_action",
            "phase11_rediscovery_visual_audit": phase11_pass,
            "phase12_seq05_typeb_goodprotect_coverage": phase12_seq05_pass,
            "phase13_direct_merge_state_projection": phase13_pass,
            "phase13_control_aware_merge_promotion": phase13_control_pass,
            "phase14_qscale_hold_refresh": phase14_qscale_pass,
        },
        "key_metrics": {
            "phase1_bad_long_windows": phase1.get("bad_long_windows"),
            "phase1_good_or_false_positive_windows": phase1.get("good_or_false_positive_windows"),
            "phase1_seqs_covered": phase1.get("seqs_covered"),
            "phase2_review_coverage": phase2.get("review_coverage"),
            "phase3_best_profile": phase3.get("best_profile"),
            "phase3_best_profile_metrics": phase3.get("profiles", {}).get(str(phase3.get("best_profile")), {}),
            "phase4_read_swa_alignment_mean": phase4.get("read_swa_alignment_mean"),
            "phase4_actual_beats_random": phase4.get("actual_beats_random"),
            "phase6_typeb_rows": phase6.get("typeb_rows"),
            "phase6_targeted_smoke_phaseE_gate_pass": phase6.get("targeted_smoke_phaseE_gate_pass"),
            "phase6_targeted_smoke_head_tail_pass_count": phase6.get("targeted_smoke_head_tail_pass_count"),
            "phase6_targeted_smoke_overlap_pass_count": phase6.get("targeted_smoke_overlap_pass_count"),
            "phase11_question_count": phase11.get("question_count"),
            "phase11_group_counts": phase11.get("group_counts"),
            "phase13_direct_projection_job_count": phase13.get("job_count"),
            "phase13_direct_projection_failed_jobs_count": phase13.get("failed_jobs_count"),
            "phase13_direct_projection_guard_rejected_chunks": (phase13.get("candidate_trace") or {}).get("guard_rejected_chunks"),
            "phase13_direct_projection_direct_semantic_chunks": (phase13.get("candidate_trace") or {}).get("direct_semantic_chunks"),
            "phase13_direct_projection_projection_accepted_chunks": (phase13.get("candidate_trace") or {}).get("projection_accepted_chunks"),
            "phase13_direct_projection_head_tail_pass_count": phase13.get("head_tail_pass_count"),
            "phase13_direct_projection_overlap_pass_count": phase13.get("overlap_pass_count"),
            "phase13_direct_projection_head_tail_median_improvement_vs_baseline_ratio": phase13.get("head_tail_median_improvement_vs_baseline_ratio"),
            "phase13_direct_projection_overlap_median_improvement_vs_baseline_ratio": phase13.get("overlap_median_improvement_vs_baseline_ratio"),
            "phase13_control_promotion_job_count": phase13_control.get("job_count"),
            "phase13_control_promotion_failed_jobs_count": phase13_control.get("failed_jobs_count"),
            "phase13_control_promotion_pass_chunks": phase13_control.get("promotion_pass_chunks"),
            "phase13_control_promotion_rejected_chunks": phase13_control.get("promotion_rejected_chunks"),
            "phase13_control_promotion_guard_rejected_chunks": phase13_control.get("guard_rejected_chunks"),
            "phase13_control_promotion_projection_accepted_chunks": phase13_control.get("projection_accepted_chunks"),
            "phase13_control_promotion_max_random_qscale_gap": phase13_control.get("max_random_qscale_gap"),
            "phase13_control_promotion_head_tail_pass_count": phase13_control.get("head_tail_pass_count"),
            "phase13_control_promotion_overlap_pass_count": phase13_control.get("overlap_pass_count"),
            "phase13_control_promotion_head_tail_median_improvement_vs_baseline_ratio": phase13_control.get("head_tail_median_improvement_vs_baseline_ratio"),
            "phase13_control_promotion_overlap_median_improvement_vs_baseline_ratio": phase13_control.get("overlap_median_improvement_vs_baseline_ratio"),
            "phase12_seq05_goodprotect_support_map_count": phase12_seq05.get("support_map_count"),
            "phase12_seq05_goodprotect_trace_target_row_count": phase12_seq05.get("trace_target_row_count"),
            "phase12_seq05_goodprotect_support_weighted_mass_min": phase12_seq05.get("support_weighted_mass_min"),
            "phase12_seq05_goodprotect_support_weighted_mass_max": phase12_seq05.get("support_weighted_mass_max"),
            "phase12_seq05_goodprotect_plain_phaseE_gate_pass": (phase12_seq05.get("plain") or {}).get("phaseE_gate_pass"),
            "phase12_seq05_goodprotect_plain_head_tail_pass_count": (phase12_seq05.get("plain") or {}).get("head_tail_pass_count"),
            "phase12_seq05_goodprotect_plain_overlap_pass_count": (phase12_seq05.get("plain") or {}).get("overlap_pass_count"),
            "phase12_seq05_goodprotect_plain_head_tail_median_improvement_vs_baseline_ratio": (phase12_seq05.get("plain") or {}).get("head_tail_median_improvement_vs_baseline_ratio"),
            "phase12_seq05_goodprotect_plain_overlap_median_improvement_vs_baseline_ratio": (phase12_seq05.get("plain") or {}).get("overlap_median_improvement_vs_baseline_ratio"),
            "phase12_seq05_goodprotect_randomgap_phaseE_gate_pass": (phase12_seq05.get("randomgap") or {}).get("phaseE_gate_pass"),
            "phase12_seq05_goodprotect_randomgap_head_tail_pass_count": (phase12_seq05.get("randomgap") or {}).get("head_tail_pass_count"),
            "phase12_seq05_goodprotect_randomgap_overlap_pass_count": (phase12_seq05.get("randomgap") or {}).get("overlap_pass_count"),
            "phase12_seq05_goodprotect_randomgap_head_tail_median_improvement_vs_baseline_ratio": (phase12_seq05.get("randomgap") or {}).get("head_tail_median_improvement_vs_baseline_ratio"),
            "phase12_seq05_goodprotect_randomgap_overlap_median_improvement_vs_baseline_ratio": (phase12_seq05.get("randomgap") or {}).get("overlap_median_improvement_vs_baseline_ratio"),
            "phase14_qscale_default_phaseE_gate_pass": (phase14_qscale.get("default") or {}).get("phaseE_gate_pass"),
            "phase14_qscale_default_head_tail_pass_count": (phase14_qscale.get("default") or {}).get("head_tail_pass_count"),
            "phase14_qscale_default_overlap_pass_count": (phase14_qscale.get("default") or {}).get("overlap_pass_count"),
            "phase14_qscale_default_head_tail_median_improvement_vs_baseline_ratio": (phase14_qscale.get("default") or {}).get("head_tail_median_improvement_vs_baseline_ratio"),
            "phase14_qscale_default_overlap_median_improvement_vs_baseline_ratio": (phase14_qscale.get("default") or {}).get("overlap_median_improvement_vs_baseline_ratio"),
            "phase14_qscale_strict_phaseE_gate_pass": (phase14_qscale.get("strict") or {}).get("phaseE_gate_pass"),
            "phase14_qscale_strict_head_tail_pass_count": (phase14_qscale.get("strict") or {}).get("head_tail_pass_count"),
            "phase14_qscale_strict_overlap_pass_count": (phase14_qscale.get("strict") or {}).get("overlap_pass_count"),
            "phase14_qscale_strict_head_tail_median_improvement_vs_baseline_ratio": (phase14_qscale.get("strict") or {}).get("head_tail_median_improvement_vs_baseline_ratio"),
            "phase14_qscale_strict_overlap_median_improvement_vs_baseline_ratio": (phase14_qscale.get("strict") or {}).get("overlap_median_improvement_vs_baseline_ratio"),
            "phase14_qscale_default_factor_min": phase14_qscale.get("default_qscale_factor_min"),
            "phase14_qscale_default_factor_max": phase14_qscale.get("default_qscale_factor_max"),
            "phase14_qscale_strict_factor_min": phase14_qscale.get("strict_qscale_factor_min"),
            "phase14_qscale_strict_factor_max": phase14_qscale.get("strict_qscale_factor_max"),
            "phase14_qscale_default_guard_rejected_chunks": phase14_qscale.get("default_guard_rejected_chunks"),
            "phase14_qscale_strict_guard_rejected_chunks": phase14_qscale.get("strict_guard_rejected_chunks"),
        },
        "conclusion": (
            "Phase1/2 established an auditable long-window case bank and visual confirmation. "
            "Phase3 and Phase4 failed the required action gates, so TTT write-less/one-hop was not allowed. "
            "Phase6 Type-B overlap-outlier rescue was run on eligible seq01 chunks with controls but did not pass. "
            "A continuation Phase13 direct merge-state projection smoke enabled residual-safe projection for candidate and controls; "
            "all jobs completed, but no projection was accepted and PhaseE still failed. "
            "A seq05 Type-B good-protection coverage check found real support maps, but both plain and random-gap variants failed PhaseE/controls. "
            "A further control-aware random-qscale promotion gate rejected all Type-B target chunks or fell back to native, "
            "and PhaseE still failed. "
            "A qscale hold-refresh continuation attenuated merge alpha under a stricter unit reference, but PhaseE still failed. "
            "Phase11 rediscovery bundle is complete, so the honest final decision is No-Go for v81 method success."
        ),
    }


def _write_report(path: Path, decision: Mapping[str, Any]) -> None:
    metrics = decision["key_metrics"]
    phase_gates = decision["phase_gates"]
    lines = [
        "# ACL2 v81 Final Report",
        "",
        f"Final status: `{decision['final_status']}`",
        f"method_gate_claimed: `{decision['method_gate_claimed']}`",
        f"v81_goal_achieved: `{decision['v81_goal_achieved']}`",
        "",
        "## Phase Gates",
        "",
    ]
    for key, value in phase_gates.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Required Questions",
            "",
            f"1. Long-window bank established: `{phase_gates['phase1_long_window_bank']}`; seqs={metrics.get('phase1_seqs_covered')}; KITTI08 remained blocked by missing baseline trajectory from Phase0.",
            "2. selected-write low-support is continuous for seq02 62-70, but not broad enough for multi-seq action coverage.",
            f"3. Risk rule did not sufficiently distinguish bad cluster and good false-positive: best={metrics.get('phase3_best_profile')} metrics={metrics.get('phase3_best_profile_metrics')}.",
            f"4. Visual confirmation completed: `{phase_gates['phase2_visual_confirmation']}`, review_coverage={metrics.get('phase2_review_coverage')}.",
            f"5. READ/SWA maps exist but confirmation failed: alignment_mean={metrics.get('phase4_read_swa_alignment_mean')}, actual_beats_random={metrics.get('phase4_actual_beats_random')}.",
            "6. TTT write-less / one-hop was not run because Phase3/4 gates failed.",
            "7. Good-window protection is not proven for runtime action; Phase6 bad-only smoke has good_pair_coverage=false.",
            "8. Semantic candidate did not beat geometry/random/shuffle controls in Phase6.",
            "9. Type-B cases need merge/gauge analysis rather than TTT, but existing overlap-outlier rescue failed.",
            f"10. Direct merge-state projection continuation pass: `{phase_gates['phase13_direct_merge_state_projection']}`; projection_accepted_chunks={metrics.get('phase13_direct_projection_projection_accepted_chunks')}.",
            f"11. Seq05 Type-B good-protection coverage pass: `{phase_gates['phase12_seq05_typeb_goodprotect_coverage']}`; support_maps={metrics.get('phase12_seq05_goodprotect_support_map_count')}, plain_phaseE={metrics.get('phase12_seq05_goodprotect_plain_phaseE_gate_pass')}, randomgap_phaseE={metrics.get('phase12_seq05_goodprotect_randomgap_phaseE_gate_pass')}.",
            f"12. Control-aware merge promotion pass: `{phase_gates['phase13_control_aware_merge_promotion']}`; promotion_pass_chunks={metrics.get('phase13_control_promotion_pass_chunks')}, max_random_qscale_gap={metrics.get('phase13_control_promotion_max_random_qscale_gap')}.",
            f"13. Qscale hold-refresh pass: `{phase_gates['phase14_qscale_hold_refresh']}`; default_factor_range={metrics.get('phase14_qscale_default_factor_min')}..{metrics.get('phase14_qscale_default_factor_max')}, strict_factor_range={metrics.get('phase14_qscale_strict_factor_min')}..{metrics.get('phase14_qscale_strict_factor_max')}.",
            "14. Held-out / 704F was not run; official entry is blocked by missing Phase5/6/7/12/13/14 pass.",
            "15. Failure chain: risk rule coverage, READ/SWA confirmation, merge/gauge actuator, seq05 good-protection coverage, direct state projection, control-aware carrier promotion, and qscale hold-refresh all failed; Phase11 rediscovery completed.",
            "",
            "## Conclusion",
            "",
            str(decision["conclusion"]),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()
    decision = _build_decision()
    _write_json(args.out_json, decision)
    _write_report(args.out_md, decision)
    print(json.dumps(_jsonable(decision), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
