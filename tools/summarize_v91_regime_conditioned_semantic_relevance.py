#!/usr/bin/env python3
"""Compact summary for v91 Phase3 relevance."""

from __future__ import annotations

import argparse
from pathlib import Path

from v86_soft_latent_utils import read_json, write_json
from v91_semantic_regime_utils import ROOT


DEFAULT_OUT = ROOT / "phase3_regime_conditioned_semantic_relevance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_json(args.out_dir / "regime_conditioned_relevance_summary.json")
    best = summary.get("best_semantic_policy", {})
    compact = {
        "phase": "Phase3_regime_conditioned_semantic_relevance_compact",
        "gate_pass": summary.get("phase3_regime_semantic_gate_pass"),
        "passing_policies": summary.get("passing_policies", []),
        "best_semantic_policy": best.get("signal"),
        "best_semantic_rho": best.get("spearman_rho_abs_log_scale_jump"),
        "best_semantic_bad_recall": best.get("bad_recall"),
        "best_semantic_good_FPR": best.get("good_FPR"),
        "best_semantic_shuffle_margin": best.get("semantic_shuffle_margin"),
        "best_component_shuffle_margin": best.get("component_shuffle_margin"),
        "best_regime_shuffle_margin": best.get("regime_shuffle_margin"),
        "blocker": summary.get("blocker", ""),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "regime_conditioned_relevance_compact_summary.json", compact)
    print(f"gate_pass={compact['gate_pass']}")
    print(f"best_semantic_policy={compact['best_semantic_policy']}")
    print(f"best_semantic_rho={compact['best_semantic_rho']}")
    print(f"best_semantic_bad_recall={compact['best_semantic_bad_recall']}")
    print(f"best_semantic_good_FPR={compact['best_semantic_good_FPR']}")
    if compact.get("blocker"):
        print(f"blocker={compact['blocker']}")


if __name__ == "__main__":
    main()
