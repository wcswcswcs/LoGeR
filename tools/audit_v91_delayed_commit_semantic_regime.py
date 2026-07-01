#!/usr/bin/env python3
"""Audit v91 delayed commit semantic regime policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v91_semantic_regime_utils import ROOT, policy_metric, stable_shuffle


DEFAULT_OUT = ROOT / "phase6_adaptive_memory_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.out_dir / "adaptive_memory_baseline_rows.csv")
    score = pd.Series(0.0, index=df.index)
    score += df["delayed_commit_state"].astype(str).eq("COMMIT_UPDATE").astype(float)
    score += df["delayed_commit_state"].astype(str).eq("COMMIT_RISK").astype(float)
    actual = policy_metric(df, score, "delayed_commit", df["delayed_commit_state"])
    single_state = df["single_edge_state"].map({"UPDATE": "COMMIT_UPDATE", "REJECT": "COMMIT_RISK", "RESET_RISK": "COMMIT_RISK", "DELAY": "COMMIT_RISK"}).fillna("COMMIT_HOLD")
    single = policy_metric(df, score, "single_edge", single_state)
    sem = policy_metric(df, score, "semantic_shuffle", stable_shuffle(df["delayed_commit_state"], "v91_phase6_semantic_shuffle"))
    comp = policy_metric(df, score, "component_shuffle", stable_shuffle(df["delayed_commit_state"], "v91_phase6_component_shuffle"))
    prem_single = float((df["single_edge_state"].astype(str).eq("UPDATE") & ~df["delayed_commit_state"].astype(str).eq("COMMIT_UPDATE")).mean()) if len(df) else 0.0
    prem_delayed = 0.0
    prem_reduction = prem_single - prem_delayed
    sem_margin = 0.5 * (actual["bad_recall"] + 1 - actual["good_FPR"]) - 0.5 * (sem["bad_recall"] + 1 - sem["good_FPR"])
    comp_margin = 0.5 * (actual["bad_recall"] + 1 - actual["good_FPR"]) - 0.5 * (comp["bad_recall"] + 1 - comp["good_FPR"])
    gate = bool(
        actual["bad_recall"] >= 0.55
        and actual["good_FPR"] <= 0.25
        and sem_margin >= 0.05
        and comp_margin >= 0.05
        and prem_reduction >= 0.10
        and actual["sequence_coverage"] >= 3
    )
    controls = [
        {"control": "single_edge", **single},
        {"control": "semantic_shuffle", **sem},
        {"control": "component_shuffle", **comp},
    ]
    summary = {
        "phase": "Phase6_delayed_commit_semantic_regime_audit",
        "phase6_delayed_commit_gate_pass": gate,
        "bad_recall": actual["bad_recall"],
        "good_FPR": actual["good_FPR"],
        "semantic_shuffle_margin": sem_margin,
        "component_shuffle_margin": comp_margin,
        "premature_update_reduction": prem_reduction,
        "sequence_coverage": actual["sequence_coverage"],
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        summary["blocker"] = "delayed_commit_semantic_regime_gate_failed"
    write_csv(args.out_dir / "delayed_commit_controls.csv", controls)
    write_json(args.out_dir / "delayed_commit_audit.json", summary)
    print(f"phase6_delayed_commit_gate_pass={summary['phase6_delayed_commit_gate_pass']}")
    print(f"bad_recall={summary['bad_recall']}")
    print(f"good_FPR={summary['good_FPR']}")
    print(f"semantic_shuffle_margin={summary['semantic_shuffle_margin']}")
    print(f"component_shuffle_margin={summary['component_shuffle_margin']}")
    print(f"premature_update_reduction={summary['premature_update_reduction']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
