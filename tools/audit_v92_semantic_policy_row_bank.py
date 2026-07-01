#!/usr/bin/env python3
"""Audit v92 Phase1 semantic policy row bank reproducibility and specificity."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v91_semantic_regime_utils import policy_metric
from v92_semantic_policy_carrier_utils import ROOT, bool_text, nseries


DEFAULT_OUT = ROOT / "phase1_semantic_policy_row_bank"
EXPECTED_COUNTS = {"HOLD": 25, "RESET_RISK": 16, "DELAY": 7, "REJECT": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _balanced(metric: dict[str, Any]) -> float:
    return 0.5 * (float(metric.get("bad_recall", 0.0)) + 1.0 - float(metric.get("good_FPR", 1.0)))


def main() -> None:
    args = parse_args()
    rows_path = args.out_dir / "semantic_policy_rows.csv"
    df = pd.read_csv(rows_path)
    score = nseries(df, "P_update") + nseries(df, "P_reject") + nseries(df, "P_delay") + nseries(df, "P_reset_risk")
    actual = policy_metric(df, score, "v92_phase1_policy_state", df["policy_state"])
    geom = policy_metric(df, nseries(df, "geometry_dominant_mode_mu").abs() + nseries(df, "H_mode"), "geometry")
    sem_ctrl = policy_metric(df, score, "semantic_shuffle", df["semantic_shuffle_state"])
    comp_ctrl = policy_metric(df, score, "component_shuffle", df["component_shuffle_state"])
    reg_ctrl = policy_metric(df, score, "regime_shuffle", df["regime_shuffle_state"])
    actual_bal = _balanced(actual)
    sem_margin = actual_bal - _balanced(sem_ctrl)
    comp_margin = actual_bal - _balanced(comp_ctrl)
    reg_margin = actual_bal - _balanced(reg_ctrl)
    good_protection = float(geom.get("good_FPR", 1.0)) - float(actual.get("good_FPR", 1.0))
    state_counts = {str(k): int(v) for k, v in df["policy_state"].astype(str).value_counts().to_dict().items()}
    no_bad_good_label = not df.get("bad_good_label_used_for_assignment", pd.Series([], dtype=str)).map(bool_text).any()
    no_scale_label = not df.get("scale_label_used_for_assignment", pd.Series([], dtype=str)).map(bool_text).any()
    gate = bool(
        len(df) >= 49
        and int(actual.get("sequence_coverage", 0)) >= 4
        and state_counts == EXPECTED_COUNTS
        and float(actual.get("bad_recall", 0.0)) >= 0.65
        and float(actual.get("good_FPR", 1.0)) <= 0.20
        and sem_margin >= 0.20
        and comp_margin >= 0.20
        and reg_margin >= 0.15
        and no_bad_good_label
        and no_scale_label
    )
    controls = [
        {"control": "geometry", **geom},
        {"control": "semantic_shuffle", **sem_ctrl},
        {"control": "component_shuffle", **comp_ctrl},
        {"control": "regime_shuffle", **reg_ctrl},
    ]
    summary = {
        "phase": "Phase1_semantic_policy_row_bank_audit",
        "phase1_semantic_policy_row_bank_gate_pass": gate,
        "row_count": int(len(df)),
        "sequence_coverage": int(actual.get("sequence_coverage", 0)),
        "state_counts": state_counts,
        "state_counts_match_expected": state_counts == EXPECTED_COUNTS,
        "bad_recall": actual.get("bad_recall"),
        "good_FPR": actual.get("good_FPR"),
        "semantic_good_protection_margin": good_protection,
        "semantic_shuffle_margin": sem_margin,
        "component_shuffle_margin": comp_margin,
        "regime_shuffle_margin": reg_margin,
        "actual_balanced_accuracy": actual_bal,
        "no_bad_good_label_used_for_assignment": no_bad_good_label,
        "no_scale_label_used_for_assignment": no_scale_label,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        summary["blocker"] = "phase1_semantic_policy_row_bank_gate_failed"
    write_csv(args.out_dir / "semantic_policy_row_bank_audit_controls.csv", controls)
    write_json(args.out_dir / "semantic_policy_row_bank_audit.json", summary)
    write_json(args.out_dir / "phase1_gate_summary.json", summary)
    print(f"phase1_semantic_policy_row_bank_gate_pass={summary['phase1_semantic_policy_row_bank_gate_pass']}")
    print(f"row_count={summary['row_count']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"state_counts={summary['state_counts']}")
    print(f"bad_recall={summary['bad_recall']}")
    print(f"good_FPR={summary['good_FPR']}")
    print(f"semantic_good_protection_margin={summary['semantic_good_protection_margin']}")
    print(f"semantic_shuffle_margin={summary['semantic_shuffle_margin']}")
    print(f"component_shuffle_margin={summary['component_shuffle_margin']}")
    print(f"regime_shuffle_margin={summary['regime_shuffle_margin']}")
    print(f"no_bad_good_label_used_for_assignment={summary['no_bad_good_label_used_for_assignment']}")
    print(f"no_scale_label_used_for_assignment={summary['no_scale_label_used_for_assignment']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
