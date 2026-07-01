#!/usr/bin/env python3
"""Build v90 feature-match topology ruler rows from v89 match evidence and v90 topology summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v90_semantic_topology_utils import ROOT, V89_FEATURE


DEFAULT_SOURCE = ROOT / "phase1_semantic_topology_source"
DEFAULT_OUT = ROOT / "phase5_feature_match_topology_ruler"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v89-feature-dir", type=Path, default=V89_FEATURE)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(args.v89_feature_dir / "feature_match_semantic_rows.csv")
    pair_base = pd.read_csv(args.v89_feature_dir / "feature_match_pair_summary.csv")
    topo = pd.read_csv(args.source_dir / "topology_pair_summary.csv")
    for df in (matches, pair_base, topo):
        df["seq"] = df["seq"].astype(str).str.zfill(2)
        df["prev_chunk"] = df["prev_chunk"].astype(int)
        df["curr_chunk"] = df["curr_chunk"].astype(int)
    topo_keep = topo[
        [
            "seq",
            "prev_chunk",
            "curr_chunk",
            "same_label_support_ratio",
            "cross_component_boundary_ratio",
            "feature_match_support_count",
            "raw_overlap_support_count",
        ]
    ]
    matches = matches.merge(topo_keep, on=["seq", "prev_chunk", "curr_chunk"], how="left")
    same_label = _bool(matches["same_label"])
    low_conf = _bool(matches["low_conf_flag"])
    cross = _bool(matches["cross_boundary"])
    dynamic = _bool(matches["dynamic_flag"])
    matches["match_topology_valid"] = same_label & (~low_conf) & (~cross) & (~dynamic) & (pd.to_numeric(matches["same_label_support_ratio"], errors="coerce").fillna(0.0) >= 0.50)
    matches["match_topology_invalid"] = cross | dynamic | (pd.to_numeric(matches["cross_component_boundary_ratio"], errors="coerce").fillna(0.0) >= 0.50)
    matches["match_topology_type"] = np.where(matches["match_topology_valid"], "MATCH_TOPOLOGY_VALID", np.where(matches["match_topology_invalid"], "MATCH_TOPOLOGY_INVALID_BOUNDARY", "MATCH_TOPOLOGY_CONTEXT_LOWOBS"))
    pair_rows: list[dict[str, Any]] = []
    for key, group in matches.groupby(["seq", "prev_chunk", "curr_chunk"], sort=False):
        seq, prev, curr = str(key[0]).zfill(2), int(key[1]), int(key[2])
        base = pair_base[(pair_base["seq"] == seq) & (pair_base["prev_chunk"] == prev) & (pair_base["curr_chunk"] == curr)]
        base_row = base.iloc[0].to_dict() if len(base) else {}
        valid_ratio = float(group["match_topology_valid"].astype(float).mean()) if len(group) else 0.0
        invalid_ratio = float(group["match_topology_invalid"].astype(float).mean()) if len(group) else 0.0
        signed = pd.to_numeric(group["signed_match_scale_ratio"], errors="coerce").fillna(0.0)
        entropy = 0.0
        if len(signed):
            hist = pd.cut(signed, bins=10, duplicates="drop").value_counts(normalize=True)
            entropy = float(-(hist * np.log(hist + 1e-12)).sum()) if len(hist) else 0.0
        pair_rows.append(
            {
                "seq": seq,
                "prev_chunk": prev,
                "curr_chunk": curr,
                "matcher_available": base_row.get("matcher_available", True),
                "matcher_type": base_row.get("matcher_type", ""),
                "verified_inlier_count": int(len(group)),
                "match_topology_valid_ratio": valid_ratio,
                "match_topology_invalid_ratio": invalid_ratio,
                "match_topology_context_ratio": float(1.0 - valid_ratio - invalid_ratio),
                "match_topology_mode_entropy": entropy,
                "match_topology_valid_score": float(len(group) * valid_ratio * (1.0 - min(1.0, invalid_ratio)) * (1.0 - min(1.0, entropy / max(np.log(10), 1e-12)))),
                "raw_match_count_baseline": int(len(group)),
                "abs_log_scale_jump_gt": base_row.get("abs_log_scale_jump_gt", ""),
                "base_case_type": base_row.get("base_case_type", ""),
            }
        )
    write_csv(args.out_dir / "feature_match_topology_rows.csv", matches.to_dict("records"))
    write_csv(args.out_dir / "feature_match_topology_pair_summary.csv", pair_rows)
    pair_df = pd.DataFrame(pair_rows)
    summary = {
        "phase": "Phase5_feature_match_topology_ruler_build",
        "match_rows": int(len(matches)),
        "pair_rows": int(len(pair_df)),
        "sequence_coverage": int(pair_df["seq"].nunique()) if len(pair_df) else 0,
        "matcher_available": bool(len(pair_df) > 0 and pair_df["matcher_available"].astype(str).str.lower().isin(["true", "1"]).any()),
        "verified_inlier_count_median": float(pd.to_numeric(pair_df["verified_inlier_count"], errors="coerce").median()) if len(pair_df) else 0.0,
        "match_topology_valid_ratio_median": float(pd.to_numeric(pair_df["match_topology_valid_ratio"], errors="coerce").median()) if len(pair_df) else 0.0,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "feature_match_topology_build_summary.json", summary)
    print(f"match_rows={summary['match_rows']}")
    print(f"pair_rows={summary['pair_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"matcher_available={summary['matcher_available']}")
    print(f"verified_inlier_count_median={summary['verified_inlier_count_median']}")
    print(f"match_topology_valid_ratio_median={summary['match_topology_valid_ratio_median']}")


if __name__ == "__main__":
    main()
