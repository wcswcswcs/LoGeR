#!/usr/bin/env python3
"""Record v88 SWA mode route audit availability."""

from __future__ import annotations

import argparse
from pathlib import Path

from v86_soft_latent_utils import read_json, write_csv, write_json


DEFAULT_MASKS = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase4_mode_route_masks")
DEFAULT_OUT = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase4_swa_mode_route_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASKS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_summary = read_json(args.mask_dir / "mode_route_mask_summary.json")
    rows = [
        {
            "audit_item": "per_head_per_layer_route_dump",
            "available": False,
            "reason": "v88 did not enter runtime/SWA route dump path because Phase3 native mismatch attribution failed",
        },
        {
            "audit_item": "diagnostic_pair_mode_masks",
            "available": True,
            "rows": mask_summary.get("mask_rows"),
            "reason": "pair-level Phase1 mode masks exist but cannot prove SWA carrier",
        },
    ]
    write_csv(args.out_dir / "swa_mode_route_audit_rows.csv", rows)
    summary = {
        "phase": "Phase4_swa_mode_route_audit",
        "swa_route_carrier_gate_pass": False,
        "dominant_mode_route_lift_available": False,
        "outlier_mode_route_excess_available": False,
        "route_entropy_available": False,
        "coverage_ge_3_sequences": False,
        "blocker": "no_per_head_per_layer_mode_route_dump_phase3_failed",
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "swa_mode_route_audit_summary.json", summary)
    print("swa_route_carrier_gate_pass=False")
    print("blocker=no_per_head_per_layer_mode_route_dump_phase3_failed")


if __name__ == "__main__":
    main()
