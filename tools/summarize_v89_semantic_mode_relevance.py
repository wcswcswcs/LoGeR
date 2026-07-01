#!/usr/bin/env python3
"""Print v89 semantic mode relevance summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from v86_soft_latent_utils import read_json


DEFAULT_DIR = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control/phase2_semantic_mode_relevance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relevance-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_json(args.relevance_dir / "semantic_mode_relevance_summary.json")
    best = summary.get("best_semantic_signal") or {}
    print(f"phase2_semantic_mode_relevance_gate_pass={summary.get('phase2_semantic_mode_relevance_gate_pass')}")
    print(f"filter={summary.get('filter')}")
    print(f"passing_semantic_signals={summary.get('passing_semantic_signals')}")
    print(f"geometry_reference_signal={summary.get('geometry_reference_signal')}")
    print(f"geometry_reference_rho={summary.get('geometry_reference_rho')}")
    print(f"best_semantic_signal={best.get('signal')}")
    print(f"best_semantic_rho={best.get('spearman_rho_abs_log_scale_jump')}")
    print(f"best_semantic_margin={best.get('semantic_shuffle_margin')}")
    print(f"best_semantic_recall={best.get('bad_recall')}")
    print(f"best_semantic_good_fpr={best.get('good_false_positive_rate')}")
    print(f"semantic_valid_support_pair_mass_nonzero_rows={summary.get('semantic_valid_support_pair_mass_nonzero_rows')}")
    print(f"blocker={summary.get('blocker', '')}")


if __name__ == "__main__":
    main()
