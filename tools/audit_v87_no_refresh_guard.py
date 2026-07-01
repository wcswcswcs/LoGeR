#!/usr/bin/env python3
"""Audit v87 Phase4 no-refresh guard."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v86_soft_latent_utils import spearman_rho, stable_hash_float, write_csv, write_json


DEFAULT_IN = Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase4_no_refresh_guard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    return parser.parse_args()


def _risk_flag(df: pd.DataFrame) -> pd.Series:
    return df["state_decision"].isin(["RESET_RISK", "ABSTAIN"])


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.in_dir / "no_refresh_guard_rows.csv")
    labelled = rows[rows["base_case_type"].isin(["bad", "good"])].copy()
    flags = _risk_flag(labelled)
    bad = labelled["base_case_type"] == "bad"
    good = labelled["base_case_type"] == "good"
    bad_recall = float((flags & bad).sum() / max(int(bad.sum()), 1)) if len(labelled) else 0.0
    good_fpr = float((flags & good).sum() / max(int(good.sum()), 1)) if len(labelled) else 1.0
    seq_cov = int(labelled["seq"].astype(str).str.zfill(2).nunique()) if len(labelled) else 0
    prior_available_rows = int(rows["prior_available"].astype(str).str.lower().eq("true").sum())
    rho = spearman_rho(rows["prior_mismatch_score"].tolist(), rows["abs_log_scale_jump_gt"].tolist())
    random_recalls = []
    for salt in range(64):
        order = sorted(range(len(labelled)), key=lambda i: stable_hash_float("v87_phase4_random", salt, i))
        same_count = int(flags.sum())
        random_flag = pd.Series(False, index=labelled.index)
        random_flag.iloc[order[:same_count]] = True
        random_recalls.append(float((random_flag & bad).sum() / max(int(bad.sum()), 1)))
    random_p95 = float(np.quantile(random_recalls, 0.95)) if random_recalls else 0.0
    margin = bad_recall - random_p95
    checks = {
        "bad_recall_ge_0p60": bad_recall >= 0.60,
        "good_fpr_le_0p25": good_fpr <= 0.25,
        "sequence_coverage_ge_3": seq_cov >= 3,
        "state_signal_beats_same_count_random_by_0p05": margin >= 0.05,
        "prior_mismatch_rho_ge_0p30": rho is not None and rho >= 0.30,
    }
    gate_pass = all(checks.values())
    control_rows = [{"control": "same_count_random_p95_bad_recall", "value": random_p95, "actual_bad_recall": bad_recall, "margin": margin}]
    write_csv(args.in_dir / "no_refresh_guard_controls.csv", control_rows)
    summary = {
        "phase": "Phase4_no_refresh_guard_audit",
        "phase4_no_refresh_guard_gate_pass": gate_pass,
        "checks": checks,
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "sequence_coverage": seq_cov,
        "prior_available_rows": prior_available_rows,
        "prior_mismatch_abs_scale_spearman_rho": rho,
        "same_count_random_bad_recall_p95": random_p95,
        "state_signal_margin_vs_random": margin,
        "state_decision_counts": rows["state_decision"].value_counts().to_dict(),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "next_action": "phase5_route_carrier" if gate_pass else "phase8_merge_gauge_counterfactual_or_phase12_visual",
    }
    write_json(args.in_dir / "no_refresh_guard_summary.json", summary)
    report = [
        "# v87 Phase4 No-Refresh Guard",
        "",
        f"- phase4_no_refresh_guard_gate_pass: `{gate_pass}`",
        f"- bad_recall: `{bad_recall}`",
        f"- good_FPR: `{good_fpr}`",
        f"- sequence_coverage: `{seq_cov}`",
        f"- prior_available_rows: `{prior_available_rows}`",
        f"- prior_mismatch_abs_scale_spearman_rho: `{rho}`",
        f"- state_signal_margin_vs_random: `{margin}`",
        "",
        "No runtime action is authorized by Phase4.",
    ]
    (args.in_dir / "no_refresh_guard_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"phase4_no_refresh_guard_gate_pass={gate_pass}")
    print(f"bad_recall={bad_recall}")
    print(f"good_FPR={good_fpr}")
    print(f"prior_available_rows={prior_available_rows}")
    print(f"prior_mismatch_abs_scale_spearman_rho={rho}")


if __name__ == "__main__":
    main()
