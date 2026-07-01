#!/usr/bin/env python3
"""Build v92 Phase0 evidence lock from the updated v91 final artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, write_csv, write_json
from v92_semantic_policy_carrier_utils import ROOT, V91_PHASE5, V91_PHASE7, V91_ROOT


EXPECTED_PHASE5 = {
    "bad_recall": 0.6923076923076923,
    "good_FPR": 0.14285714285714285,
    "semantic_shuffle_margin": 0.4065934065934066,
    "component_shuffle_margin": 0.36263736263736274,
    "regime_shuffle_margin": 0.22527472527472525,
}
EXPECTED_COUNTS = {"HOLD": 25, "RESET_RISK": 16, "DELAY": 7, "REJECT": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase0_evidence_lock")
    parser.add_argument("--v91-root", type=Path, default=V91_ROOT)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _near(a: Any, b: float, tol: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    out = args.out_dir
    v91_root = args.v91_root
    phase5_dir = v91_root / "phase5_memory_update_policy"
    phase7_dir = v91_root / "phase7_carrier_attribution_or_blocked"
    final_path = v91_root / "report_final/final_decision.json"
    phase5_audit_path = phase5_dir / "policy_state_audit.json"
    phase5_summary_path = phase5_dir / "policy_state_summary.json"
    phase7_summary_path = phase7_dir / "phase7_carrier_summary.json"
    required = [
        Path("docs/ACL2_v91TF_SemanticTopologyRegimeAdaptiveMemoryControl_实验结果复盘.md"),
        Path("docs/ACL2_v91TF_SemanticTopologyRegimeAdaptiveMemoryControl_执行日志.md"),
        final_path,
        phase5_dir / "policy_state_rows.csv",
        phase5_audit_path,
        phase5_summary_path,
        phase7_summary_path,
        phase7_dir / "direct_boundary_update_trace_proxy.csv",
        phase7_dir / "v91_external_mask_materialization/anchor_route_mask_positions.csv",
        phase7_dir / "route_dump_smoke",
    ]
    required_rows = [{"path": str(path), "exists": path.exists()} for path in required]
    missing = [row for row in required_rows if not row["exists"]]
    final = _json(final_path)
    phase5_audit = _json(phase5_audit_path)
    phase5_summary = _json(phase5_summary_path)
    phase7 = _json(phase7_summary_path)
    forbidden = [
        "repeat_v84_v91_source_side_external_mask_alpha_small_sweep",
        "repeat_v85_v86_qk_latent_c_or_fdim_small_sweep",
        "repeat_v88_signed_mode_threshold_or_mismatch_sweep",
        "repeat_v89_compact_semantic_valid_ratio_sweep",
        "repeat_v90_topology_invalid_or_support_threshold_sweep",
        "repeat_v91_policy_state_threshold_sweep_without_bug_repair",
        "run_ttt_without_confirmed_carrier",
        "claim_route_smoke_as_runtime_method_success",
        "claim_visual_audit_as_action_success",
    ]
    stale_zip = Path("code_audit_pack/acl2_v91tf_semantic_topology_regime_adaptive_memory_control_core_audit_20260625_050135.zip")
    final_zip = Path("code_audit_pack/acl2_v91tf_semantic_topology_regime_adaptive_memory_control_core_audit_20260625_052602.zip")
    phase5_numbers_match = all(_near(phase5_audit.get(key), value) for key, value in EXPECTED_PHASE5.items())
    phase5_counts_match = dict(phase5_audit.get("state_counts") or {}) == EXPECTED_COUNTS
    gate = bool(
        not missing
        and final.get("final_status") == "NO_GO_CARRIER_NOT_FOUND"
        and final.get("blocker") == "true_route_smoke_available_controls_incomplete_not_promoted"
        and final.get("runtime_action_allowed") is False
        and final.get("runtime_action_executed") is False
        and final.get("ttt_allowed") is False
        and phase5_audit.get("phase5_memory_update_policy_gate_pass") is True
        and phase5_numbers_match
        and phase5_counts_match
        and phase7.get("entered") is True
        and phase7.get("phase7_carrier_gate_pass") is False
        and phase7.get("route_dump_smoke_available") is True
        and int(phase7.get("route_dump_seq_coverage") or 0) >= 4
        and int(phase7.get("route_dump_successful_jobs") or 0) >= 8
        and phase7.get("runtime_action_allowed") is False
        and phase7.get("ttt_allowed") is False
        and len(forbidden) > 0
        and (not stale_zip.exists())
        and final_zip.exists()
    )
    summary = {
        "phase": "Phase0_v91_updated_evidence_lock",
        "phase0_gate_pass": gate,
        "missing_required_inputs": len(missing),
        "v91_final_status": final.get("final_status"),
        "v91_blocker": final.get("blocker"),
        "phase5_policy_gate_pass": phase5_audit.get("phase5_memory_update_policy_gate_pass"),
        "phase5_policy_rows": phase5_summary.get("policy_rows"),
        "phase5_state_counts": phase5_audit.get("state_counts"),
        "phase5_state_counts_match_expected": phase5_counts_match,
        "phase5_bad_recall": phase5_audit.get("bad_recall"),
        "phase5_good_FPR": phase5_audit.get("good_FPR"),
        "phase5_semantic_good_protection_margin": phase5_audit.get("semantic_good_protection_margin"),
        "phase5_semantic_shuffle_margin": phase5_audit.get("semantic_shuffle_margin"),
        "phase5_component_shuffle_margin": phase5_audit.get("component_shuffle_margin"),
        "phase5_regime_shuffle_margin": phase5_audit.get("regime_shuffle_margin"),
        "phase5_numbers_match_expected": phase5_numbers_match,
        "phase7_entered": phase7.get("entered"),
        "phase7_carrier_gate_pass": phase7.get("phase7_carrier_gate_pass"),
        "true_route_or_trace_available": phase7.get("true_route_or_trace_available"),
        "external_mask_materialization_feasible": phase7.get("external_mask_materialization_feasible"),
        "route_dump_seq_coverage": phase7.get("route_dump_seq_coverage"),
        "route_dump_successful_jobs": phase7.get("route_dump_successful_jobs"),
        "route_dump_failed_jobs": phase7.get("route_dump_failed_jobs"),
        "delta_selected_mean": phase7.get("route_dump_actual_minus_random_selected_lift_mean"),
        "delta_headmax_mean": phase7.get("route_dump_actual_minus_random_headmax_lift_mean"),
        "runtime_action_allowed": final.get("runtime_action_allowed"),
        "runtime_action_executed": final.get("runtime_action_executed"),
        "ttt_allowed": final.get("ttt_allowed"),
        "stale_050135_zip_exists": stale_zip.exists(),
        "fresh_052602_zip_exists": final_zip.exists(),
        "forbidden_repeat_count": len(forbidden),
    }
    if not gate:
        summary["blocker"] = "phase0_v91_evidence_lock_failed"
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "evidence_lock.json",
        {
            "v91_final": final,
            "phase5_audit": phase5_audit,
            "phase5_summary": phase5_summary,
            "phase7_summary": phase7,
            "required_inputs": required_rows,
            "forbidden_repeats": forbidden,
            "expected_phase5": {"state_counts": EXPECTED_COUNTS, **EXPECTED_PHASE5},
        },
    )
    write_json(out / "phase0_gate_summary.json", summary)
    write_csv(out / "required_inputs.csv", required_rows)
    write_csv(out / "forbidden_repeats.csv", [{"forbidden_repeat": item} for item in forbidden])
    write_csv(
        out / "v92_hypothesis_matrix.csv",
        [
            {"hypothesis": "H1_v91_phase5_policy_real_signal", "phase": "Phase1", "status": "locked_for_reproduction"},
            {"hypothesis": "H2_merge_gauge_boundary_carrier", "phase": "Phase2_Phase3", "status": "primary_to_test"},
            {"hypothesis": "H3_swa_qk_or_query_secondary_carrier", "phase": "Phase4", "status": "secondary_if_boundary_fails"},
            {"hypothesis": "H4_data_source_needed_if_policy_has_no_carrier", "phase": "Phase7", "status": "repair_ladder_if_carrier_fails"},
            {"hypothesis": "H5_ttt_only_after_confirmed_runtime_carrier", "phase": "Phase8", "status": "blocked_until_runtime_pass"},
        ],
    )
    (out / "forbidden_repeats.md").write_text(
        "\n".join(["# v92 Forbidden Repeats", "", *[f"- {item}" for item in forbidden]]) + "\n",
        encoding="utf-8",
    )
    (out / "stale_closeout_warning.md").write_text(
        "\n".join(
            [
                "# Stale Closeout Warning",
                "",
                "The v91 20260625_050135 package is superseded by continuation evidence and must not be used as final.",
                f"- stale_050135_zip_exists: `{stale_zip.exists()}`",
                f"- fresh_052602_zip_exists: `{final_zip.exists()}`",
                f"- final_status: `{final.get('final_status')}`",
                f"- blocker: `{final.get('blocker')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "v91_updated_no_go_boundary.md").write_text(
        "\n".join(
            [
                "# v91 Updated No-Go Boundary for v92",
                "",
                f"- final_status: `{final.get('final_status')}`",
                f"- blocker: `{final.get('blocker')}`",
                f"- phase5_memory_update_policy_gate_pass: `{phase5_audit.get('phase5_memory_update_policy_gate_pass')}`",
                f"- phase5_state_counts: `{phase5_audit.get('state_counts')}`",
                f"- phase5_bad_recall: `{phase5_audit.get('bad_recall')}`",
                f"- phase5_good_FPR: `{phase5_audit.get('good_FPR')}`",
                f"- phase7_carrier_gate_pass: `{phase7.get('phase7_carrier_gate_pass')}`",
                f"- route_dump_seq_coverage: `{phase7.get('route_dump_seq_coverage')}`",
                f"- route_dump_successful_jobs: `{phase7.get('route_dump_successful_jobs')}`",
                f"- delta_selected_mean: `{phase7.get('route_dump_actual_minus_random_selected_lift_mean')}`",
                f"- delta_headmax_mean: `{phase7.get('route_dump_actual_minus_random_headmax_lift_mean')}`",
                f"- runtime_action_allowed: `{final.get('runtime_action_allowed')}`",
                f"- runtime_action_executed: `{final.get('runtime_action_executed')}`",
                f"- ttt_allowed: `{final.get('ttt_allowed')}`",
                "",
                "v92 may reuse the v91 semantic policy as a signal, but must still find a real memory carrier before counterfactual, runtime action, or TTT.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"phase0_gate_pass={summary['phase0_gate_pass']}")
    print(f"missing_required_inputs={summary['missing_required_inputs']}")
    print(f"v91_final_status={summary['v91_final_status']}")
    print(f"phase5_policy_gate_pass={summary['phase5_policy_gate_pass']}")
    print(f"phase5_numbers_match_expected={summary['phase5_numbers_match_expected']}")
    print(f"phase5_state_counts_match_expected={summary['phase5_state_counts_match_expected']}")
    print(f"phase7_carrier_gate_pass={summary['phase7_carrier_gate_pass']}")
    print(f"route_dump_seq_coverage={summary['route_dump_seq_coverage']}")
    print(f"route_dump_successful_jobs={summary['route_dump_successful_jobs']}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")
    print(f"ttt_allowed={summary['ttt_allowed']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
