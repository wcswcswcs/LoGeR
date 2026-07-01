#!/usr/bin/env python3
"""Audit ACL2 v86 anchor absence and historical-prior signal."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v86_soft_latent_utils import spearman_rho, stable_hash_float, write_csv, write_json


DEFAULT_PHASE1 = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase1_soft_pair_universe")
DEFAULT_PRIOR = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase3_historical_prior")
DEFAULT_SCALE = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase4_offline_scale_labels")
DEFAULT_OUT = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase3_anchor_absence_signal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--prior-dir", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--scale-dir", type=Path, default=DEFAULT_SCALE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--absence-threshold", type=float, default=0.35)
    return parser.parse_args()


def _metrics(df: pd.DataFrame, flag_col: str) -> dict[str, float | int]:
    labelled = df[df["case_label"].isin(["bad", "good"])]
    bad = labelled[labelled["case_label"] == "bad"]
    good = labelled[labelled["case_label"] == "good"]
    bad_recall = float(bad[flag_col].mean()) if len(bad) else 0.0
    good_fpr = float(good[flag_col].mean()) if len(good) else 0.0
    return {
        "flagged_rows": int(labelled[flag_col].sum()) if len(labelled) else 0,
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "sequence_coverage": int(labelled[labelled[flag_col]]["seq"].astype(str).str.zfill(2).nunique()),
    }


def main() -> None:
    args = parse_args()
    by_pair = pd.read_csv(args.phase1_dir / "soft_pair_by_seq_chunk.csv")
    prior = pd.read_csv(args.prior_dir / "historical_prior_rows.csv")
    scale = pd.read_csv(args.scale_dir / "offline_scale_jump_rows.csv")
    for frame in (by_pair, prior, scale):
        frame["seq"] = frame["seq"].astype(str).str.zfill(2)
    df = by_pair.merge(prior, on=["seq", "prev_chunk", "curr_chunk", "case_label", "quality_label"], how="left")
    df = df.merge(
        scale[["seq", "prev_chunk", "curr_chunk", "abs_log_scale_jump", "scale_label_available"]],
        on=["seq", "prev_chunk", "curr_chunk"],
        how="left",
    )
    df["anchor_absence_score"] = pd.to_numeric(df["anchor_absence_score_x"], errors="coerce").fillna(
        pd.to_numeric(df["anchor_absence_score_y"], errors="coerce")
    )
    df["absence_flag"] = df["anchor_absence_score"] >= args.absence_threshold
    # Good-protection branch: require low support plus either risk/low observability or prior conflict.
    prior_conflict = pd.to_numeric(df["prior_conflict_score"], errors="coerce").fillna(0.0)
    risk = pd.to_numeric(df["mean_risk_score"], errors="coerce").fillna(0.0)
    df["protected_anchor_absence_score"] = df["anchor_absence_score"] * (0.5 + 0.5 * risk) * (1.0 + np.clip(prior_conflict, 0.0, 1.0))
    df["protected_absence_flag"] = df["protected_anchor_absence_score"] >= args.absence_threshold
    labelled = df[df["case_label"].isin(["bad", "good"])].copy()
    k = int(labelled["absence_flag"].sum())
    if k > 0 and len(labelled) > 0:
        labelled["random_score"] = [
            stable_hash_float(row.seq, row.prev_chunk, row.curr_chunk, "absence_random")
            for row in labelled.itertuples(index=False)
        ]
        random_top = labelled.sort_values("random_score", ascending=False).head(k)
        random_bad_recall = float((random_top["case_label"] == "bad").sum() / max((labelled["case_label"] == "bad").sum(), 1))
    else:
        random_bad_recall = 0.0
    absence = _metrics(df, "absence_flag")
    protected = _metrics(df, "protected_absence_flag")
    prior_rho = spearman_rho(df["prior_mismatch_score"].tolist(), df["abs_log_scale_jump"].tolist())
    beats_random = float(absence["bad_recall"] - random_bad_recall)
    gate_pass = (
        absence["bad_recall"] >= 0.60
        and absence["good_FPR"] <= 0.25
        and absence["sequence_coverage"] >= 3
        and beats_random >= 0.05
        and prior_rho is not None
        and prior_rho >= 0.30
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_rows = df.drop(columns=[col for col in ["anchor_absence_score_x", "anchor_absence_score_y"] if col in df.columns]).to_dict("records")
    write_csv(args.out_dir / "anchor_absence_signal_rows.csv", out_rows)
    summary = {
        "phase": "Phase3_anchor_absence_signal",
        "phase3_anchor_absence_gate_pass": gate_pass,
        "absence_threshold": args.absence_threshold,
        "absence_metrics": absence,
        "protected_absence_metrics": protected,
        "same_count_random_bad_recall": random_bad_recall,
        "anchor_absence_beats_same_count_random_by": beats_random,
        "prior_mismatch_abs_scale_spearman_rho": prior_rho,
        "scale_label_available_rows": int(pd.to_numeric(df["scale_label_available"], errors="coerce").fillna(False).astype(bool).sum()),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "notes": [
            "Protected absence is a plan-directed good-protection attempt; it is reported separately and not used to hide the raw FPR.",
            "Prior mismatch uses only historical prior rows built from earlier valid current-to-history C updates.",
        ],
    }
    write_json(args.out_dir / "anchor_absence_signal_summary.json", summary)
    print(f"phase3_anchor_absence_gate_pass={gate_pass}")
    print(f"bad_recall={absence['bad_recall']}")
    print(f"good_FPR={absence['good_FPR']}")
    print(f"sequence_coverage={absence['sequence_coverage']}")
    print(f"anchor_absence_beats_random={beats_random}")
    print(f"prior_mismatch_abs_scale_spearman_rho={prior_rho}")


if __name__ == "__main__":
    main()
