#!/usr/bin/env python3
"""Build v88 diagnostic mode route mask rows from Phase1 pair ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import write_csv, write_json


DEFAULT_PHASE1 = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase1_scale_mode_consensus_universe")
DEFAULT_OUT = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase4_mode_route_masks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.phase1_dir / "scale_mode_pair_rows.csv")
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "seq": str(row["seq"]).zfill(2),
                "prev_chunk": int(row["prev_chunk"]),
                "curr_chunk": int(row["curr_chunk"]),
                "base_case_type": row.get("base_case_type", ""),
                "quality_type": row.get("quality_type", ""),
                "dominant_mode_center": row.get("weighted_mode_mu", ""),
                "dominant_mode_mass": row.get("mode_mass_top1", ""),
                "outlier_mode_mass_proxy": row.get("mode_mass_top2", ""),
                "mode_entropy": row.get("mode_entropy", ""),
                "native_mode_mismatch": row.get("native_mode_mismatch", ""),
                "source_path": row.get("source_path", ""),
                "mask_authority": "diagnostic_phase1_mode_bins_only",
                "runtime_action_allowed": False,
            }
        )
    write_csv(args.out_dir / "mode_route_mask_rows.csv", rows)
    summary = {
        "phase": "Phase4_mode_route_masks",
        "mask_rows": len(rows),
        "sequence_coverage": int(df["seq"].astype(str).str.zfill(2).nunique()),
        "per_head_route_dump_available": False,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "note": "These are diagnostic pair-level mode masks only. They are not per-head/per-layer SWA route dumps.",
    }
    write_json(args.out_dir / "mode_route_mask_summary.json", summary)
    print(f"mask_rows={summary['mask_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print("per_head_route_dump_available=False")


if __name__ == "__main__":
    main()
