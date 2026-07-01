#!/usr/bin/env python3
"""Print a compact v88 Phase5 counterfactual summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from v86_soft_latent_utils import read_json


DEFAULT_PHASE5 = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase5_mode_aware_counterfactual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5-dir", type=Path, default=DEFAULT_PHASE5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_json(args.phase5_dir / "mode_aware_counterfactual_summary.json")
    print(f"scale_label_gate_pass={summary.get('scale_label_gate_pass')}")
    print(f"raw_residual_counterfactual_available={summary.get('raw_residual_counterfactual_available')}")
    print(f"raw_residual_gate_pass={summary.get('raw_residual_gate_pass')}")
    print(f"passing_families={summary.get('passing_families')}")
    best = summary.get("best_family") or {}
    print(f"best_family={best.get('family')}")
    print(f"best_bad_median_I_scale={best.get('bad_median_I_scale')}")
    print(f"best_good_max_scale_error_worsen={best.get('good_max_scale_error_worsen')}")
    print(f"blocker={summary.get('blocker', '')}")


if __name__ == "__main__":
    main()
