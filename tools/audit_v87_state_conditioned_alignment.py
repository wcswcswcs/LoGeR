#!/usr/bin/env python3
"""Audit v87 Phase3 state-conditioned alignment gate."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from v86_soft_latent_utils import spearman_rho, write_json


DEFAULT_IN = Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase3_state_conditioned_latent_transport")
DEFAULT_PHASE1 = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase1_scale_conditioned_pair_universe_k16_r1_median_abs"
)
DEFAULT_PHASE2_SUMMARY = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase2_scale_relevance_k16_r1_median_abs_highobs/proxy_relevance_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    return parser.parse_args()


def _finite(series: pd.Series) -> list[float]:
    out = []
    for value in series.tolist():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def main() -> None:
    args = parse_args()
    fit = pd.read_csv(args.in_dir / "state_conditioned_c_fit_rows.csv")
    by_pair = pd.read_csv(args.phase1_dir / "scale_conditioned_pair_by_adjacent.csv")
    valid = fit[fit["fit_status"].astype(str) == "ok"].copy()
    bad_valid = valid[valid["base_case_type"] == "bad"].copy()
    gains = _finite(valid["support_alignment_gain"]) if len(valid) else []
    random_margins = _finite(valid["support_actual_minus_random_p95"]) if len(valid) else []
    shuffle_margins = _finite(valid["support_actual_minus_shuffle_p95"]) if len(valid) else []
    gaps = _finite(valid["train_heldout_gap"]) if len(valid) else []
    support_conflict_gaps = _finite(valid["support_conflict_gap"]) if len(valid) else []
    scale_rho = spearman_rho(valid["support_conflict_gap"].tolist(), valid["abs_log_scale_jump_gt"].tolist()) if len(valid) else None
    sequence_coverage = int(valid["seq"].astype(str).str.zfill(2).nunique()) if len(valid) else 0
    bad_state_rows = by_pair[
        (by_pair["base_case_type"] == "bad") & (by_pair["state_label"].isin(["CONFLICT", "ABSENCE", "STRESS"]))
    ]
    checks = {
        "valid_pair_rows_ge_8": int(len(valid[["seq", "prev_chunk", "curr_chunk"]].drop_duplicates())) >= 8 if len(valid) else False,
        "bad_valid_pair_rows_ge_3_or_bad_conflict_absence_ge_5": (
            int(len(bad_valid[["seq", "prev_chunk", "curr_chunk"]].drop_duplicates())) >= 3 if len(bad_valid) else False
        )
        or int(len(bad_state_rows)) >= 5,
        "median_support_alignment_gain_ge_0p05": bool(gains) and float(np.median(gains)) >= 0.05,
        "actual_minus_random_p95_ge_0p03": bool(random_margins) and float(np.median(random_margins)) >= 0.03,
        "actual_minus_shuffle_p95_ge_0p03": bool(shuffle_margins) and float(np.median(shuffle_margins)) >= 0.03,
        "support_conflict_gap_positive": bool(support_conflict_gaps) and float(np.median(support_conflict_gaps)) > 0.0,
        "support_conflict_gap_scale_relevant": scale_rho is not None and scale_rho >= 0.30,
        "train_heldout_gap_le_0p20": bool(gaps) and float(np.max(gaps)) <= 0.20,
        "no_overfit_flag": bool(len(valid) > 0) and not valid["overfit_flag"].astype(str).str.lower().eq("true").any(),
        "sequence_coverage_ge_3": sequence_coverage >= 3,
    }
    gate_pass = all(checks.values())
    invalid_counts = fit["invalid_reason"].fillna("").value_counts().to_dict()
    if not len(valid) and invalid_counts.get("no_support_state_rows", 0) == len(fit):
        route = "phase4_no_refresh_guard"
        blocker = "support_state_absent_no_C_fit"
    elif not gate_pass and scale_rho is not None and scale_rho < 0.30:
        route = "phase4_or_stop_route_action"
        blocker = "appearance_alignment_not_gauge"
    elif gate_pass:
        route = "phase4_then_phase5_route_carrier"
        blocker = ""
    else:
        route = "phase3_repair_or_phase4_if_state_scale_relevant"
        blocker = "phase3_gate_failed"
    summary = {
        "phase": "Phase3_state_conditioned_alignment_audit",
        "phase3_alignment_gate_pass": gate_pass,
        "checks": checks,
        "valid_rows": int(len(valid)),
        "valid_pair_rows": int(len(valid[["seq", "prev_chunk", "curr_chunk"]].drop_duplicates())) if len(valid) else 0,
        "bad_valid_pair_rows": int(len(bad_valid[["seq", "prev_chunk", "curr_chunk"]].drop_duplicates())) if len(bad_valid) else 0,
        "bad_conflict_absence_stress_pair_rows": int(len(bad_state_rows)),
        "sequence_coverage": sequence_coverage,
        "median_support_alignment_gain": float(np.median(gains)) if gains else None,
        "median_actual_minus_random_p95": float(np.median(random_margins)) if random_margins else None,
        "median_actual_minus_shuffle_p95": float(np.median(shuffle_margins)) if shuffle_margins else None,
        "median_support_conflict_gap": float(np.median(support_conflict_gaps)) if support_conflict_gaps else None,
        "support_conflict_gap_abs_scale_spearman_rho": scale_rho,
        "invalid_reason_counts": invalid_counts,
        "blocker": blocker,
        "next_action": route,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.in_dir / "state_conditioned_alignment_summary.json", summary)
    report = [
        "# v87 Phase3 State-Conditioned Alignment",
        "",
        f"- phase3_alignment_gate_pass: `{gate_pass}`",
        f"- valid_rows: `{summary['valid_rows']}`",
        f"- valid_pair_rows: `{summary['valid_pair_rows']}`",
        f"- bad_valid_pair_rows: `{summary['bad_valid_pair_rows']}`",
        f"- bad_conflict_absence_stress_pair_rows: `{summary['bad_conflict_absence_stress_pair_rows']}`",
        f"- blocker: `{blocker}`",
        f"- invalid_reason_counts: `{invalid_counts}`",
        "",
        "Runtime action remains blocked.",
    ]
    (args.in_dir / "state_conditioned_alignment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"phase3_alignment_gate_pass={gate_pass}")
    print(f"valid_rows={summary['valid_rows']}")
    print(f"bad_conflict_absence_stress_pair_rows={summary['bad_conflict_absence_stress_pair_rows']}")
    print(f"blocker={blocker}")
    print(f"next_action={route}")


if __name__ == "__main__":
    main()
