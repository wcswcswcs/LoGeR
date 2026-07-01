#!/usr/bin/env python3
"""Audit ACL2 v86 Phase2 alignment gain gate."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from v86_soft_latent_utils import write_json


DEFAULT_IN = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase2_robust_transport")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    return parser.parse_args()


def _finite(values: pd.Series) -> list[float]:
    out: list[float] = []
    for value in values.tolist():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.in_dir / "c_fit_rows.csv")
    candidate = rows[
        (rows["direction"] == "current_to_history")
        & (~rows["C_family"].isin(["C0_identity", "C6_full_rank_ridge_upper_bound"]))
    ].copy()
    valid = candidate[candidate["valid_for_next_phase"].astype(str).str.lower() == "true"].copy()
    gains = _finite(valid["alignment_gain"]) if len(valid) else []
    random_margins = _finite(valid["actual_minus_random_p95"]) if len(valid) else []
    shuffle_margins = _finite(valid["actual_minus_shuffle_p95"]) if len(valid) else []
    gaps = _finite(valid["train_heldout_gap"]) if len(valid) else []
    sequence_coverage = int(valid["seq"].astype(str).str.zfill(2).nunique()) if len(valid) else 0
    bad_valid = valid[valid["case_label"] == "bad"]
    checks = {
        "valid_support_sufficient_pair_rows_ge_8": int(len(valid[["seq", "prev_chunk", "curr_chunk"]].drop_duplicates())) >= 8,
        "bad_valid_pair_rows_ge_3": int(len(bad_valid[["seq", "prev_chunk", "curr_chunk"]].drop_duplicates())) >= 3,
        "median_heldout_alignment_gain_ge_0p05": bool(gains) and float(np.median(gains)) >= 0.05,
        "actual_minus_random_p95_ge_0p03": bool(random_margins) and float(np.median(random_margins)) >= 0.03,
        "actual_minus_shuffle_p95_ge_0p03": bool(shuffle_margins) and float(np.median(shuffle_margins)) >= 0.03,
        "train_heldout_gap_le_0p20": bool(gaps) and float(np.max(gaps)) <= 0.20,
        "no_overfit_for_passing_rows": bool(len(valid) > 0) and not valid["overfit_flag"].astype(str).str.lower().eq("true").any(),
        "sequence_coverage_ge_3": sequence_coverage >= 3,
    }
    gate_pass = all(checks.values())
    summary = {
        "phase": "Phase2_alignment_gain",
        "phase2_alignment_gate_pass": gate_pass,
        "checks": checks,
        "candidate_rows": int(len(candidate)),
        "valid_rows": int(len(valid)),
        "valid_pair_rows": int(len(valid[["seq", "prev_chunk", "curr_chunk"]].drop_duplicates())) if len(valid) else 0,
        "bad_valid_pair_rows": int(len(bad_valid[["seq", "prev_chunk", "curr_chunk"]].drop_duplicates())) if len(bad_valid) else 0,
        "sequence_coverage": sequence_coverage,
        "median_alignment_gain": float(np.median(gains)) if gains else None,
        "median_actual_minus_random_p95": float(np.median(random_margins)) if random_margins else None,
        "median_actual_minus_shuffle_p95": float(np.median(shuffle_margins)) if shuffle_margins else None,
        "max_train_heldout_gap_valid": float(np.max(gaps)) if gaps else None,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "next_action": "phase3_phase4_prior_and_scale_audit"
        if not gate_pass
        else "phase3_phase4_before_route_carrier",
    }
    write_json(args.in_dir / "alignment_gain_gate_summary.json", summary)
    report = [
        "# Phase2 Alignment Gain Report",
        "",
        f"- phase2_alignment_gate_pass: `{str(gate_pass).lower()}`",
        f"- valid rows: `{summary['valid_rows']}`",
        f"- valid pair rows: `{summary['valid_pair_rows']}`",
        f"- bad valid pair rows: `{summary['bad_valid_pair_rows']}`",
        f"- sequence coverage: `{summary['sequence_coverage']}`",
        f"- median alignment gain: `{summary['median_alignment_gain']}`",
        f"- median actual minus random p95: `{summary['median_actual_minus_random_p95']}`",
        f"- median actual minus shuffle p95: `{summary['median_actual_minus_shuffle_p95']}`",
        f"- max train/heldout gap among valid rows: `{summary['max_train_heldout_gap_valid']}`",
        "",
        "Runtime action remains blocked in Phase2 regardless of this audit result.",
    ]
    (args.in_dir / "alignment_gain_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"phase2_alignment_gate_pass={gate_pass}")
    print(f"valid_rows={summary['valid_rows']}")
    print(f"valid_pair_rows={summary['valid_pair_rows']}")
    print(f"bad_valid_pair_rows={summary['bad_valid_pair_rows']}")
    print(f"median_alignment_gain={summary['median_alignment_gain']}")


if __name__ == "__main__":
    main()
