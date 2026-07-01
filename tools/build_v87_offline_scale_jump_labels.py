#!/usr/bin/env python3
"""Build ACL2 v87 Phase2 offline scale-jump labels from v86 audit labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import seq_norm, write_csv, write_json


DEFAULT_V86_LABELS = Path(
    "results/acl2_v86tf_robust_soft_latent_gauge_transport/phase4_offline_scale_labels/offline_scale_jump_rows.csv"
)
DEFAULT_PHASE1 = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase1_scale_conditioned_pair_universe/scale_conditioned_pair_by_adjacent.csv"
)
DEFAULT_OUT = Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase2_scale_relevance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v86-labels", type=Path, default=DEFAULT_V86_LABELS)
    parser.add_argument("--phase1-by-adjacent", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v86 = pd.read_csv(args.v86_labels)
    phase1 = pd.read_csv(args.phase1_by_adjacent)
    for frame in (v86, phase1):
        frame["seq"] = frame["seq"].map(seq_norm)
        frame["prev_chunk"] = frame["prev_chunk"].astype(int)
        frame["curr_chunk"] = frame["curr_chunk"].astype(int)

    merged = phase1[["seq", "prev_chunk", "curr_chunk", "state_label"]].merge(
        v86,
        on=["seq", "prev_chunk", "curr_chunk"],
        how="left",
        suffixes=("", "_v86"),
    )
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "seq": row.get("seq"),
                "prev_chunk": int(row.get("prev_chunk")),
                "curr_chunk": int(row.get("curr_chunk")),
                "case_label": row.get("case_label"),
                "quality_label": row.get("quality_label"),
                "state_label": row.get("state_label"),
                "chunk_scale_prev": row.get("chunk_scale_prev"),
                "chunk_scale_curr": row.get("chunk_scale_curr"),
                "adjacent_log_scale_jump": row.get("adjacent_log_scale_jump"),
                "abs_log_scale_jump": row.get("abs_log_scale_jump"),
                "scale_jump_sign": row.get("scale_jump_sign"),
                "full_ATE_contribution_proxy": row.get("full_ATE_contribution_proxy"),
                "prev_trajectory": row.get("prev_trajectory"),
                "curr_trajectory": row.get("curr_trajectory"),
                "scale_label_available": row.get("scale_label_available"),
                "missing_reason": row.get("missing_reason"),
                "offline_audit_label_only": True,
                "no_gt_runtime_feature": False,
                "label_source": str(args.v86_labels),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "scale_jump_labels.csv", rows)
    available = [r for r in rows if str(r["scale_label_available"]).lower() == "true"]
    summary = {
        "phase": "Phase2_offline_scale_jump_labels",
        "label_rows": len(rows),
        "scale_label_available_rows": len(available),
        "sequence_coverage": len({r["seq"] for r in available}),
        "offline_audit_label_only": True,
        "no_gt_runtime_feature": False,
        "source": str(args.v86_labels),
        "note": "v87 reuses v86 trajectory-derived offline Sim3 scale labels as audit-only labels.",
    }
    write_json(args.out_dir / "scale_jump_label_summary.json", summary)
    print(f"label_rows={summary['label_rows']}")
    print(f"scale_label_available_rows={summary['scale_label_available_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")


if __name__ == "__main__":
    main()
