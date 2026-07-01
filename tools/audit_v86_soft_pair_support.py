#!/usr/bin/env python3
"""Audit ACL2 v86 Phase1 soft pair support gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import write_json


DEFAULT_IN = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase1_soft_pair_universe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.in_dir / "soft_pair_rows.csv")
    by_pair = pd.read_csv(args.in_dir / "soft_pair_by_seq_chunk.csv")
    labelled = by_pair[by_pair["case_label"].isin(["bad", "good"])].copy()
    labelled_nonstress = labelled[labelled["quality_label"] != "low_conf_stress"].copy()
    row_labelled_nonstress = rows[rows["case_label"].isin(["bad", "good"]) & (rows["quality_label"] != "low_conf_stress")].copy()
    bad = labelled[labelled["case_label"] == "bad"].copy()

    high_zero = rows[(rows["zero_conf_flag"].astype(str).str.lower() == "true") & (rows["w_fit"] >= 0.05)]
    zero_rows = rows[rows["zero_conf_flag"].astype(str).str.lower() == "true"]
    zero_conf_high_weight_ratio = float(len(high_zero) / max(len(zero_rows), 1))
    support_sufficient = by_pair[by_pair["support_sufficient_for_dim8"].astype(str).str.lower() == "true"]
    bad_support_sufficient = bad[bad["support_sufficient_for_dim8"].astype(str).str.lower() == "true"]
    support_insufficient_bad = bad[bad["support_sufficient_for_dim8"].astype(str).str.lower() != "true"]

    q_avail = float(row_labelled_nonstress["q_feature_available"].astype(str).str.lower().eq("true").mean())
    k_avail = float(row_labelled_nonstress["k_feature_available"].astype(str).str.lower().eq("true").mean())
    checks = {
        "adjacent_labelled_rows_ge_24": int(len(labelled)) >= 24,
        "sequence_coverage_ge_4": int(labelled["seq"].nunique()) >= 4,
        "q_feature_availability_ge_90pct": q_avail >= 0.90,
        "k_feature_availability_ge_90pct": k_avail >= 0.90,
        "labelled_nonstress_pair_count_ge_18": int(len(labelled_nonstress)) >= 18,
        "weighted_support_sufficient_pairs_ge_10": int(len(support_sufficient)) >= 10,
        "bad_weighted_support_or_explicit_insufficient": int(len(bad_support_sufficient)) >= 4
        or int(len(support_insufficient_bad)) >= 4,
        "zero_conf_high_weight_ratio_le_05": zero_conf_high_weight_ratio <= 0.05,
    }
    gate_pass = all(checks.values())
    summary = {
        "phase": "Phase1_soft_pair_support",
        "phase1_gate_pass": gate_pass,
        "checks": checks,
        "adjacent_labelled_rows": int(len(labelled)),
        "sequence_coverage": int(labelled["seq"].nunique()),
        "q_feature_availability": q_avail,
        "k_feature_availability": k_avail,
        "labelled_nonstress_pair_count": int(len(labelled_nonstress)),
        "weighted_support_sufficient_pairs": int(len(support_sufficient)),
        "bad_weighted_support_sufficient_pairs": int(len(bad_support_sufficient)),
        "support_insufficient_bad_pairs_explicitly_classified": int(len(support_insufficient_bad)),
        "zero_conf_high_weight_ratio": zero_conf_high_weight_ratio,
        "support_state_counts": by_pair["support_state_preliminary"].value_counts().to_dict(),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "next_action": "phase2_robust_C_audit" if gate_pass else "phase1_repair_or_phase3_absence_route",
    }
    write_json(args.in_dir / "soft_pair_support_summary.json", summary)
    print(f"phase1_gate_pass={gate_pass}")
    print(f"weighted_support_sufficient_pairs={summary['weighted_support_sufficient_pairs']}")
    print(f"bad_weighted_support_sufficient_pairs={summary['bad_weighted_support_sufficient_pairs']}")
    print(f"support_insufficient_bad_pairs_explicitly_classified={summary['support_insufficient_bad_pairs_explicitly_classified']}")
    print(f"zero_conf_high_weight_ratio={zero_conf_high_weight_ratio:.6g}")


if __name__ == "__main__":
    main()
