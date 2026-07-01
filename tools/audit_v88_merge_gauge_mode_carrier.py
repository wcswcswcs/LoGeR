#!/usr/bin/env python3
"""Audit v88 merge/gauge mode carrier gate from Phase3 and Phase5 evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from v86_soft_latent_utils import read_json, write_csv, write_json


DEFAULT_PHASE3 = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase3_native_gauge_update_attribution")
DEFAULT_PHASE5 = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase5_mode_aware_counterfactual")
DEFAULT_OUT = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase4_merge_gauge_mode_carrier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--phase5-dir", type=Path, default=DEFAULT_PHASE5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phase3 = read_json(args.phase3_dir / "native_gauge_update_attribution_summary.json")
    phase5 = read_json(args.phase5_dir / "mode_aware_counterfactual_summary.json")
    best3 = phase3.get("best_variant") or {}
    best5 = phase5.get("best_family") or {}
    rows = [
        {
            "audit_item": "phase3_native_update_attribution",
            "gate_pass": phase3.get("phase3_native_update_attribution_gate_pass"),
            "evidence": f"best_variant={best3.get('variant')} recall={best3.get('MISMATCH_BAD_recall')} good_fpr={best3.get('MISMATCH_GOOD_FPR')} rho={best3.get('native_mode_mismatch_rho_abs_log_scale_jump')}",
        },
        {
            "audit_item": "phase5_mode_aware_counterfactual",
            "gate_pass": phase5.get("scale_label_gate_pass") and phase5.get("raw_residual_gate_pass"),
            "evidence": f"best_family={best5.get('family')} bad_I_scale={best5.get('bad_median_I_scale')} good_worsen={best5.get('good_max_scale_error_worsen')} raw_available={phase5.get('raw_residual_counterfactual_available')}",
        },
    ]
    merge_pass = bool(phase3.get("phase3_native_update_attribution_gate_pass") and phase5.get("scale_label_gate_pass") and phase5.get("raw_residual_gate_pass"))
    summary = {
        "phase": "Phase4_merge_gauge_mode_carrier",
        "merge_gauge_mode_carrier_gate_pass": merge_pass,
        "bad_recall": best3.get("MISMATCH_BAD_recall"),
        "good_FPR": best3.get("MISMATCH_GOOD_FPR"),
        "native_mode_mismatch_rho": best3.get("native_mode_mismatch_rho_abs_log_scale_jump"),
        "counterfactual_scale_label_gate_pass": phase5.get("scale_label_gate_pass"),
        "counterfactual_raw_residual_gate_pass": phase5.get("raw_residual_gate_pass"),
        "blocker": "" if merge_pass else "phase3_attribution_failed_and_phase5_counterfactual_failed_or_raw_unavailable",
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_csv(args.out_dir / "merge_gauge_mode_carrier_rows.csv", rows)
    write_json(args.out_dir / "merge_gauge_mode_carrier_summary.json", summary)
    report = [
        "# v88 Phase4 Merge/Gauge Mode Carrier",
        "",
        f"- merge_gauge_mode_carrier_gate_pass: `{merge_pass}`",
        f"- phase3 best recall/good_FPR/rho: `{summary['bad_recall']} / {summary['good_FPR']} / {summary['native_mode_mismatch_rho']}`",
        f"- phase5 scale/raw gates: `{summary['counterfactual_scale_label_gate_pass']} / {summary['counterfactual_raw_residual_gate_pass']}`",
        f"- blocker: `{summary['blocker']}`",
    ]
    (args.out_dir / "merge_gauge_mode_carrier_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"merge_gauge_mode_carrier_gate_pass={merge_pass}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
