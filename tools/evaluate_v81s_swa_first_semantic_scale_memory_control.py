#!/usr/bin/env python3
"""Build final ACL2 v81S SWA-first semantic scale memory decision artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REPORT_ROOT = Path("results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/report_final")
V81TF_ROOT = Path("results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/report_final")
DEFAULT_OUT_JSON = REPORT_ROOT / "v81s_final_decision.json"
DEFAULT_OUT_MD = REPORT_ROOT / "v81s_final_report.md"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _gate(payload: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _build_decision() -> dict[str, Any]:
    s1_default = _read_json(REPORT_ROOT / "phaseS1_multiseq_swa_overlap_repair/phaseS1_overlap_pair_audit_summary.json")
    s1_minconf0 = _read_json(REPORT_ROOT / "phaseS1_multiseq_swa_overlap_repair_minconf0/phaseS1_overlap_pair_audit_summary.json")
    s1_conf = _read_json(REPORT_ROOT / "phaseS1_multiseq_swa_overlap_repair_minconf0/seq01_minconf0_geometry_confidence_audit.json")
    s2 = _read_json(REPORT_ROOT / "phaseS2_swa_good_bad_pair_bank/swa_pair_bank_resummary.json")
    if not s2:
        s2 = _read_json(REPORT_ROOT / "phaseS2_swa_good_bad_pair_bank/swa_good_bad_pair_bank_summary.json")
    s3 = _read_json(REPORT_ROOT / "phaseS3_swa_visual_confirmation/visual_integrity_audit.json")
    s5 = _read_json(REPORT_ROOT / "phaseS5_swa_action_route_audit/swa_action_route_audit_summary.json")
    s14 = _read_json(REPORT_ROOT / "phaseS14_rediscovery/visual_integrity_audit.json")
    v81tf = _read_json(V81TF_ROOT / "v81_final_decision.json")
    v81tf_phase13 = _read_json(
        V81TF_ROOT
        / "phase13_direct_merge_state_projection/typeb_overlap_outlier_projection_bad/direct_merge_state_projection_summary.json"
    )
    v81tf_phase13_control = _read_json(
        V81TF_ROOT
        / "phase13_control_aware_merge_promotion/typeb_overlap_outlier_random_qscale_gate_bad/control_aware_merge_promotion_summary.json"
    )
    v81tf_phase12_seq05 = _read_json(
        V81TF_ROOT
        / "phase12_typeb_coverage_seq05_goodprotect/seq05_typeb_goodprotect_summary.json"
    )
    v81tf_phase14_qscale = _read_json(V81TF_ROOT / "phase14_qscale_hold_refresh/qscale_hold_refresh_summary.json")

    s1_default_pass = bool(_gate(s1_default, "gate", "phaseS1_gate_pass"))
    s1_alt_pass = bool(_gate(s1_minconf0, "gate", "phaseS1_gate_pass"))
    s2_pass = bool(_gate(s2, "gate", "phaseS2_gate_pass") if isinstance(s2.get("gate"), Mapping) else s2.get("phaseS2_gate_pass"))
    s3_visual_pass = bool(_gate(s3, "gate", "phaseS3_visual_artifact_gate_pass"))
    s3_full_pass = bool(_gate(s3, "gate", "phaseS3_gate_pass"))
    s5_action_fidelity_pass = bool(_gate(s5, "gate", "phaseS5_action_fidelity_pass"))
    s5_geometry_pass = bool(_gate(s5, "gate", "phaseS5_geometry_metric_gate_pass"))
    s5_pass = bool(_gate(s5, "gate", "phaseS5_gate_pass"))
    s8_pass = bool(False)
    s14_pass = bool(s14.get("visual_audit_gate_pass"))

    final_status = (
        "No-Go_method_not_achieved_with_swa_route_and_rediscovery_complete"
        if s14_pass
        else "No-Go_pending_phaseS14_rediscovery"
    )
    phase_gates = {
        "phaseS1_default_overlap_repair": s1_default_pass,
        "phaseS1_minconf0_overlap_repair_alt": s1_alt_pass,
        "phaseS1_alt_quality_caveat": "seq01_minconf0_zero_confidence_high",
        "phaseS2_swa_pair_bank": s2_pass,
        "phaseS3_visual_artifact_gate": s3_visual_pass,
        "phaseS3_full_true_route_coverage_gate": s3_full_pass,
        "phaseS5_swa_action_fidelity": s5_action_fidelity_pass,
        "phaseS5_swa_geometry_metric_gate": s5_geometry_pass,
        "phaseS5_swa_action_gate": s5_pass,
        "phaseS6_ttt_after_swa": "blocked_by_phaseS5_geometry_gate",
        "phaseS8_merge_gauge_fallback": s8_pass,
        "phaseS12_heldout_official_validation": "blocked_by_missing_phaseS5_or_phaseS8_pass",
        "phaseS14_rediscovery_visual_audit": s14_pass,
    }
    return {
        "schema": "acl2_v81s_final_decision_v1",
        "final_status": final_status,
        "method_gate_claimed": False,
        "v81s_goal_achieved": False,
        "training_free": True,
        "official_704F_run": False,
        "official_704F_success": False,
        "official_704F_blocker": "Phase S5/S8 method gates did not pass; Phase S12 official entry is only allowed after SWA/merge-gauge pass.",
        "phase_gates": phase_gates,
        "key_metrics": {
            "phaseS1_default_allowed_seqs": _gate(s1_default, "gate", "swa_action_allowed_seqs"),
            "phaseS1_alt_allowed_seqs": _gate(s1_minconf0, "gate", "swa_action_allowed_seqs"),
            "phaseS1_seq01_minconf0_either_zero_ratio": s1_conf.get("either_zero_ratio"),
            "phaseS1_seq01_minconf0_both_zero_ratio": s1_conf.get("both_zero_ratio"),
            "phaseS2_rows": s2.get("rows"),
            "phaseS2_case_counts": s2.get("case_counts"),
            "phaseS2_seq_coverage": s2.get("seq_coverage"),
            "phaseS3_route_smoke_pair_count": s3.get("route_smoke_pair_count"),
            "phaseS3_route_smoke_pairs": s3.get("route_smoke_pairs"),
            "phaseS5_route_row_count": s5.get("route_row_count"),
            "phaseS5_route_mask_row_count": s5.get("route_mask_row_count"),
            "phaseS5_route_file_count": s5.get("route_file_count"),
            "phaseS5_seq_coverage": s5.get("seq_coverage"),
            "phaseS5_candidate_summaries": s5.get("candidate_summaries"),
            "phaseS14_group_counts": s14.get("group_counts"),
            "phaseS14_failed_swa_question_count": s14.get("failed_swa_question_count"),
            "v81tf_shared_typeb_phase_gates": {
                key: value
                for key, value in (v81tf.get("phase_gates") or {}).items()
                if str(key).startswith("phase10")
                or key == "phase6_typeb_merge_boundary_rescue"
                or key == "phase12_seq05_typeb_goodprotect_coverage"
                or key == "phase13_direct_merge_state_projection"
                or key == "phase13_control_aware_merge_promotion"
                or key == "phase14_qscale_hold_refresh"
            },
            "v81tf_shared_typeb_selected_metrics": {
                key: value
                for key, value in (v81tf.get("key_metrics") or {}).items()
                if str(key).startswith("phase10_robust_semoverlap")
                or str(key).startswith("phase10_latent_scale_projection")
                or str(key).startswith("phase10_thingstuff_dense")
                or str(key).startswith("phase12_seq05_goodprotect")
                or str(key).startswith("phase13_direct_projection")
                or str(key).startswith("phase13_control_promotion")
                or str(key).startswith("phase14_qscale")
                or str(key).startswith("phase6_targeted")
            },
            "v81tf_shared_phase13_direct_projection_summary": {
                "phaseE_gate_pass": v81tf_phase13.get("phaseE_gate_pass"),
                "job_count": v81tf_phase13.get("job_count"),
                "failed_jobs_count": v81tf_phase13.get("failed_jobs_count"),
                "projection_accepted_chunks": (v81tf_phase13.get("candidate_trace") or {}).get("projection_accepted_chunks"),
                "direct_semantic_chunks": (v81tf_phase13.get("candidate_trace") or {}).get("direct_semantic_chunks"),
                "guard_rejected_chunks": (v81tf_phase13.get("candidate_trace") or {}).get("guard_rejected_chunks"),
                "head_tail_pass_count": v81tf_phase13.get("head_tail_pass_count"),
                "overlap_pass_count": v81tf_phase13.get("overlap_pass_count"),
            },
            "v81tf_shared_phase13_control_aware_promotion_summary": {
                "phaseE_gate_pass": v81tf_phase13_control.get("phaseE_gate_pass"),
                "job_count": v81tf_phase13_control.get("job_count"),
                "failed_jobs_count": v81tf_phase13_control.get("failed_jobs_count"),
                "promotion_pass_chunks": v81tf_phase13_control.get("promotion_pass_chunks"),
                "promotion_rejected_chunks": v81tf_phase13_control.get("promotion_rejected_chunks"),
                "guard_rejected_chunks": v81tf_phase13_control.get("guard_rejected_chunks"),
                "projection_accepted_chunks": v81tf_phase13_control.get("projection_accepted_chunks"),
                "max_random_qscale_gap": v81tf_phase13_control.get("max_random_qscale_gap"),
                "head_tail_pass_count": v81tf_phase13_control.get("head_tail_pass_count"),
                "overlap_pass_count": v81tf_phase13_control.get("overlap_pass_count"),
            },
            "v81tf_shared_phase12_seq05_goodprotect_summary": {
                "phase12_typeb_seq05_goodprotect_pass": v81tf_phase12_seq05.get("phase12_typeb_seq05_goodprotect_pass"),
                "decision": v81tf_phase12_seq05.get("decision"),
                "support_map_count": v81tf_phase12_seq05.get("support_map_count"),
                "support_weighted_mass_min": v81tf_phase12_seq05.get("support_weighted_mass_min"),
                "support_weighted_mass_max": v81tf_phase12_seq05.get("support_weighted_mass_max"),
                "plain_phaseE_gate_pass": (v81tf_phase12_seq05.get("plain") or {}).get("phaseE_gate_pass"),
                "plain_head_tail_pass_count": (v81tf_phase12_seq05.get("plain") or {}).get("head_tail_pass_count"),
                "plain_overlap_pass_count": (v81tf_phase12_seq05.get("plain") or {}).get("overlap_pass_count"),
                "plain_head_tail_median_improvement_vs_baseline_ratio": (v81tf_phase12_seq05.get("plain") or {}).get("head_tail_median_improvement_vs_baseline_ratio"),
                "plain_overlap_median_improvement_vs_baseline_ratio": (v81tf_phase12_seq05.get("plain") or {}).get("overlap_median_improvement_vs_baseline_ratio"),
                "randomgap_phaseE_gate_pass": (v81tf_phase12_seq05.get("randomgap") or {}).get("phaseE_gate_pass"),
                "randomgap_head_tail_pass_count": (v81tf_phase12_seq05.get("randomgap") or {}).get("head_tail_pass_count"),
                "randomgap_overlap_pass_count": (v81tf_phase12_seq05.get("randomgap") or {}).get("overlap_pass_count"),
                "randomgap_head_tail_median_improvement_vs_baseline_ratio": (v81tf_phase12_seq05.get("randomgap") or {}).get("head_tail_median_improvement_vs_baseline_ratio"),
                "randomgap_overlap_median_improvement_vs_baseline_ratio": (v81tf_phase12_seq05.get("randomgap") or {}).get("overlap_median_improvement_vs_baseline_ratio"),
            },
            "v81tf_shared_phase14_qscale_hold_refresh_summary": {
                "phase14_qscale_hold_refresh_pass": v81tf_phase14_qscale.get("phase14_qscale_hold_refresh_pass"),
                "decision": v81tf_phase14_qscale.get("decision"),
                "default_phaseE_gate_pass": (v81tf_phase14_qscale.get("default") or {}).get("phaseE_gate_pass"),
                "default_head_tail_pass_count": (v81tf_phase14_qscale.get("default") or {}).get("head_tail_pass_count"),
                "default_overlap_pass_count": (v81tf_phase14_qscale.get("default") or {}).get("overlap_pass_count"),
                "default_overlap_median_improvement_vs_baseline_ratio": (v81tf_phase14_qscale.get("default") or {}).get("overlap_median_improvement_vs_baseline_ratio"),
                "strict_phaseE_gate_pass": (v81tf_phase14_qscale.get("strict") or {}).get("phaseE_gate_pass"),
                "strict_head_tail_pass_count": (v81tf_phase14_qscale.get("strict") or {}).get("head_tail_pass_count"),
                "strict_overlap_pass_count": (v81tf_phase14_qscale.get("strict") or {}).get("overlap_pass_count"),
                "strict_overlap_median_improvement_vs_baseline_ratio": (v81tf_phase14_qscale.get("strict") or {}).get("overlap_median_improvement_vs_baseline_ratio"),
                "default_qscale_factor_min": v81tf_phase14_qscale.get("default_qscale_factor_min"),
                "default_qscale_factor_max": v81tf_phase14_qscale.get("default_qscale_factor_max"),
                "strict_qscale_factor_min": v81tf_phase14_qscale.get("strict_qscale_factor_min"),
                "strict_qscale_factor_max": v81tf_phase14_qscale.get("strict_qscale_factor_max"),
                "default_guard_rejected_chunks": v81tf_phase14_qscale.get("default_guard_rejected_chunks"),
                "strict_guard_rejected_chunks": v81tf_phase14_qscale.get("strict_guard_rejected_chunks"),
            },
        },
        "conclusion": (
            "v81S repaired multi-sequence overlap artifacts and produced a balanced SWA adjacent-pair bank. "
            "Q/K/V visual artifacts are complete, and true route-smoke maps/action fidelity were produced on a multi-sequence bad/good sample. "
            "However S5 geometry gains are far below the declared thresholds and no candidate passes controls. "
            "The paired v81TF Type-B fallback family, including seq05 good-protection coverage, direct merge-state projection, control-aware carrier promotion, and qscale hold-refresh continuations, also failed controls, so held-out/704F validation remains blocked."
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, decision: Mapping[str, Any]) -> None:
    lines = [
        "# ACL2 v81S Final Report",
        "",
        f"Final status: `{decision['final_status']}`",
        f"method_gate_claimed: `{decision['method_gate_claimed']}`",
        f"v81s_goal_achieved: `{decision['v81s_goal_achieved']}`",
        "",
        "## Phase Gates",
        "",
    ]
    for key, value in (decision.get("phase_gates") or {}).items():
        lines.append(f"- {key}: `{value}`")
    metrics = decision.get("key_metrics") or {}
    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. S1 default overlap repair pass: `{decision['phase_gates']['phaseS1_default_overlap_repair']}`; minconf0 alt pass: `{decision['phase_gates']['phaseS1_minconf0_overlap_repair_alt']}`.",
            f"2. S1 caveat: seq01 minconf0 either_zero_ratio=`{metrics.get('phaseS1_seq01_minconf0_either_zero_ratio')}`, both_zero_ratio=`{metrics.get('phaseS1_seq01_minconf0_both_zero_ratio')}`.",
            f"3. S2 case bank rows=`{metrics.get('phaseS2_rows')}`, case_counts=`{metrics.get('phaseS2_case_counts')}`, seqs=`{metrics.get('phaseS2_seq_coverage')}`.",
            f"4. S3 visual artifact gate=`{decision['phase_gates']['phaseS3_visual_artifact_gate']}`, full true-route coverage gate=`{decision['phase_gates']['phaseS3_full_true_route_coverage_gate']}`.",
            f"5. S5 action fidelity=`{decision['phase_gates']['phaseS5_swa_action_fidelity']}`, geometry metric gate=`{decision['phase_gates']['phaseS5_swa_geometry_metric_gate']}`.",
            f"6. S5 route rows=`{metrics.get('phaseS5_route_row_count')}`, route masks=`{metrics.get('phaseS5_route_mask_row_count')}`, seqs=`{metrics.get('phaseS5_seq_coverage')}`.",
            "7. TTT after SWA was not opened because S5 geometry gate failed.",
            "8. Shared Type-B merge/gauge fallback evidence from v81TF Phase6/10/12/13/14 remains false across overlap-outlier, scale-state, retrieval, robust, latent, ThingStuff, seq05 good-protection coverage, direct projection, control-aware promotion, and qscale hold-refresh variants.",
            "9. Held-out/704F was not run because no SWA or merge/gauge method gate passed.",
            "10. Phase S14 rediscovery visual audit completed; next work must be a new direct merge/gauge interface or control-aware carrier, not alpha sweeping.",
            "",
            "## Conclusion",
            "",
            str(decision.get("conclusion")),
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
