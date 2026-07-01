#!/usr/bin/env python3
"""Audit ACL2 v87 Phase1 scale-conditioned pair universe gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import write_json


DEFAULT_IN = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase1_scale_conditioned_pair_universe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    return parser.parse_args()


def _truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.in_dir / "scale_conditioned_pair_rows.csv")
    by_pair = pd.read_csv(args.in_dir / "scale_conditioned_pair_by_adjacent.csv")
    labelled = by_pair[by_pair["base_case_type"].isin(["bad", "good"])].copy()
    labelled_rows = rows[rows["base_case_type"].isin(["bad", "good"])].copy()
    nonstress_rows = labelled_rows[labelled_rows["state_label"] != "STRESS"].copy()
    high_quality_rows = rows[rows["quality_type"] != "low_conf_stress"].copy()
    bad_pairs = labelled[labelled["base_case_type"] == "bad"].copy()

    q_avail = float(_truth(nonstress_rows["q_feature_available"]).mean()) if len(nonstress_rows) else 0.0
    k_avail = float(_truth(nonstress_rows["k_feature_available"]).mean()) if len(nonstress_rows) else 0.0
    raw_shape_avail = (
        float((high_quality_rows["local_shape_proxy_source"].astype(str) == "raw_overlap_knn").mean()) if len(high_quality_rows) else 0.0
    )
    support_or_conflict_pairs = by_pair[pd.to_numeric(by_pair["support_or_conflict_effective_sample_size"], errors="coerce") >= 10]
    bad_classified = bad_pairs["state_label"].isin(["SUPPORT", "CONFLICT", "ABSENCE", "STRESS"]).all() if len(bad_pairs) else False

    zero_conf_rows = rows[_truth(rows["either_zero_conf"])]
    high_zero_support = zero_conf_rows[pd.to_numeric(zero_conf_rows["support_weight"], errors="coerce") >= 0.05]
    zero_conf_high_positive_support_ratio = float(len(high_zero_support) / max(len(zero_conf_rows), 1))

    seq01_low_conf = rows[
        (rows["seq"].astype(str).str.zfill(2) == "01")
        & ((rows["quality_type"].astype(str) == "low_conf_stress") | rows["stress_type"].astype(str).str.contains("minconf0|zero", na=False))
    ]
    seq01_low_conf_support = int((seq01_low_conf["state_label"].astype(str) == "SUPPORT").sum()) if len(seq01_low_conf) else 0

    checks = {
        "adjacent_labelled_rows_ge_24": int(len(labelled)) >= 24,
        "sequence_coverage_ge_4": int(labelled["seq"].astype(str).str.zfill(2).nunique()) >= 4,
        "q_feature_availability_ge_90pct_nonstress": q_avail >= 0.90,
        "k_feature_availability_ge_90pct_nonstress": k_avail >= 0.90,
        "raw_overlap_local_shape_proxy_availability_ge_80pct_high_quality": raw_shape_avail >= 0.80,
        "support_or_conflict_effective_pairs_ge_10": int(len(support_or_conflict_pairs)) >= 10,
        "bad_rows_explicitly_classified": bool(bad_classified),
        "zero_conf_high_positive_support_ratio_le_05": zero_conf_high_positive_support_ratio <= 0.05,
        "seq01_low_conf_rows_not_support": seq01_low_conf_support == 0,
    }
    gate_pass = all(checks.values())
    summary = {
        "phase": "Phase1_scale_conditioned_pair_universe_audit",
        "phase1_gate_pass": gate_pass,
        "checks": checks,
        "adjacent_labelled_rows": int(len(labelled)),
        "sequence_coverage": int(labelled["seq"].astype(str).str.zfill(2).nunique()) if len(labelled) else 0,
        "q_feature_availability_nonstress": q_avail,
        "k_feature_availability_nonstress": k_avail,
        "raw_overlap_local_shape_proxy_availability_high_quality": raw_shape_avail,
        "support_or_conflict_effective_pairs": int(len(support_or_conflict_pairs)),
        "bad_pair_state_counts": bad_pairs["state_label"].value_counts().to_dict(),
        "pair_state_counts": by_pair["state_label"].value_counts().to_dict(),
        "row_state_counts": rows["state_label"].value_counts().to_dict(),
        "zero_conf_high_positive_support_ratio": zero_conf_high_positive_support_ratio,
        "seq01_low_conf_row_count": int(len(seq01_low_conf)),
        "seq01_low_conf_support_rows": seq01_low_conf_support,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "next_action": "phase2_no_gt_scale_proxy_relevance" if gate_pass else "phase1_repair",
    }
    write_json(args.in_dir / "phase1_gate_summary.json", summary)
    print(f"phase1_gate_pass={gate_pass}")
    print(f"adjacent_labelled_rows={summary['adjacent_labelled_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"raw_overlap_local_shape_proxy_availability_high_quality={raw_shape_avail:.6g}")
    print(f"support_or_conflict_effective_pairs={summary['support_or_conflict_effective_pairs']}")
    print(f"zero_conf_high_positive_support_ratio={zero_conf_high_positive_support_ratio:.6g}")


if __name__ == "__main__":
    main()
