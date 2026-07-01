#!/usr/bin/env python3
"""Audit v91 semantic memory update policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v91_semantic_regime_utils import ROOT, nseries, policy_metric
from v86_soft_latent_utils import write_csv, write_json


DEFAULT_OUT = ROOT / "phase5_memory_update_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.out_dir / "policy_state_rows.csv")
    score = nseries(df, "P_update") + nseries(df, "P_reject") + nseries(df, "P_delay") + nseries(df, "P_reset_risk")
    actual = policy_metric(df, score, "P5_policy_state", df["policy_state"])
    geom = policy_metric(df, nseries(df, "geometry_dominant_mode_mu").abs() + nseries(df, "H_mode"), "geometry")
    sem_ctrl = policy_metric(df, score, "semantic_shuffle", df["semantic_shuffle_state"])
    comp_ctrl = policy_metric(df, score, "component_shuffle", df["component_shuffle_state"])
    reg_ctrl = policy_metric(df, score, "regime_shuffle", df["regime_shuffle_state"])
    bal = 0.5 * (actual["bad_recall"] + 1 - actual["good_FPR"])
    sem_margin = bal - 0.5 * (sem_ctrl["bad_recall"] + 1 - sem_ctrl["good_FPR"])
    comp_margin = bal - 0.5 * (comp_ctrl["bad_recall"] + 1 - comp_ctrl["good_FPR"])
    reg_margin = bal - 0.5 * (reg_ctrl["bad_recall"] + 1 - reg_ctrl["good_FPR"])
    good_margin = geom["good_FPR"] - actual["good_FPR"]
    update_rows = df[df["policy_state"] == "UPDATE"]
    reject_delay = df[df["policy_state"].isin(["REJECT", "DELAY", "RESET_RISK"])]
    update_lowconf_context_ratio = float(((nseries(update_rows, "S_context") > update_rows["S_context"].quantile(0.75)) | (nseries(update_rows, "S_lowobs") > update_rows["S_lowobs"].quantile(0.75))).mean()) if len(update_rows) else 0.0
    reject_delay_good_ratio = float(reject_delay["base_case_type"].astype(str).eq("good").mean()) if len(reject_delay) else 0.0
    gate = bool(
        actual["bad_recall"] >= 0.55
        and actual["good_FPR"] <= 0.25
        and good_margin >= 0.10
        and sem_margin >= 0.05
        and comp_margin >= 0.05
        and reg_margin >= 0.05
        and update_lowconf_context_ratio <= 0.50
        and reject_delay_good_ratio <= 0.50
        and actual["sequence_coverage"] >= 3
    )
    controls = [
        {"control": "geometry", **geom},
        {"control": "semantic_shuffle", **sem_ctrl},
        {"control": "component_shuffle", **comp_ctrl},
        {"control": "regime_shuffle", **reg_ctrl},
    ]
    summary = {
        "phase": "Phase5_semantic_memory_update_policy_audit",
        "phase5_memory_update_policy_gate_pass": gate,
        "bad_recall": actual["bad_recall"],
        "good_FPR": actual["good_FPR"],
        "semantic_good_protection_margin": good_margin,
        "semantic_shuffle_margin": sem_margin,
        "component_shuffle_margin": comp_margin,
        "regime_shuffle_margin": reg_margin,
        "update_lowconf_context_ratio": update_lowconf_context_ratio,
        "reject_delay_good_ratio": reject_delay_good_ratio,
        "sequence_coverage": actual["sequence_coverage"],
        "state_counts": df["policy_state"].value_counts().to_dict(),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        summary["blocker"] = "semantic_memory_update_policy_gate_failed"
    write_csv(args.out_dir / "policy_state_audit_controls.csv", controls)
    write_json(args.out_dir / "policy_state_audit.json", summary)
    print(f"phase5_memory_update_policy_gate_pass={summary['phase5_memory_update_policy_gate_pass']}")
    print(f"bad_recall={summary['bad_recall']}")
    print(f"good_FPR={summary['good_FPR']}")
    print(f"semantic_good_protection_margin={summary['semantic_good_protection_margin']}")
    print(f"semantic_shuffle_margin={summary['semantic_shuffle_margin']}")
    print(f"component_shuffle_margin={summary['component_shuffle_margin']}")
    print(f"regime_shuffle_margin={summary['regime_shuffle_margin']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
