#!/usr/bin/env python3
"""Audit v88 SWA mode route carrier gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from v86_soft_latent_utils import read_json


DEFAULT_DIR = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase4_swa_mode_route_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_json(args.audit_dir / "swa_mode_route_audit_summary.json")
    print(f"swa_route_carrier_gate_pass={summary.get('swa_route_carrier_gate_pass')}")
    print(f"dominant_mode_route_lift_available={summary.get('dominant_mode_route_lift_available')}")
    print(f"route_entropy_available={summary.get('route_entropy_available')}")
    print(f"blocker={summary.get('blocker')}")


if __name__ == "__main__":
    main()
