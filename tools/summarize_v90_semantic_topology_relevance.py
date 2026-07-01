#!/usr/bin/env python3
"""Summarize v90 Phase3 semantic topology relevance output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import read_json, write_json
from v90_semantic_topology_utils import ROOT


DEFAULT_OUT = ROOT / "phase3_semantic_topology_relevance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_json(args.out_dir / "topology_relevance_summary.json")
    by_signal = pd.read_csv(args.out_dir / "topology_relevance_by_signal.csv")
    topology = by_signal[by_signal["is_topology_conditioned"].astype(str).str.lower().isin(["true", "1"])].copy()
    if len(topology):
        topology["_rho"] = pd.to_numeric(topology["spearman_rho_abs_log_scale_jump"], errors="coerce").fillna(-999)
        topology["_recall"] = pd.to_numeric(topology["bad_recall"], errors="coerce").fillna(0)
        topology["_fpr"] = pd.to_numeric(topology["good_false_positive_rate"], errors="coerce").fillna(1)
        best = topology.sort_values(["phase3_global_signal_pass", "_rho", "_recall", "_fpr"], ascending=[False, False, False, True]).iloc[0].to_dict()
    else:
        best = {}
    out = {
        "phase": "Phase3_semantic_topology_relevance_summary",
        "global_gate_pass": summary.get("phase3_topology_relevance_global_gate_pass"),
        "passing_topology_signals": summary.get("passing_topology_signals", []),
        "best_topology_signal": best.get("signal"),
        "best_topology_rho": best.get("spearman_rho_abs_log_scale_jump"),
        "best_topology_bad_recall": best.get("bad_recall"),
        "best_topology_good_fpr": best.get("good_false_positive_rate"),
        "best_topology_semantic_shuffle_margin": best.get("semantic_shuffle_margin"),
        "best_topology_component_shuffle_margin": best.get("component_shuffle_margin"),
        "blocker": summary.get("blocker", ""),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "topology_relevance_compact_summary.json", out)
    print(f"global_gate_pass={out['global_gate_pass']}")
    print(f"best_topology_signal={out['best_topology_signal']}")
    print(f"best_topology_rho={out['best_topology_rho']}")
    print(f"best_topology_semantic_shuffle_margin={out['best_topology_semantic_shuffle_margin']}")
    print(f"best_topology_component_shuffle_margin={out['best_topology_component_shuffle_margin']}")
    if out.get("blocker"):
        print(f"blocker={out['blocker']}")


if __name__ == "__main__":
    main()
