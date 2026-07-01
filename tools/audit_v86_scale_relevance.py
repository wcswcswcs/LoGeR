#!/usr/bin/env python3
"""Audit ACL2 v86 Phase4 offline scale relevance of alignment/absence signals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import spearman_rho, stable_hash_float, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_ROOT / "phase2_robust_transport_ridge10")
    parser.add_argument("--prior-dir", type=Path, default=DEFAULT_ROOT / "phase3_historical_prior")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "phase4_scale_relevance")
    parser.add_argument("--absence-threshold", type=float, default=0.35)
    return parser.parse_args()


def _signal_flags(values: pd.Series, signal: str, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if signal in {"prior_mismatch_score", "prior_conflict_score"}:
        # A non-positive prior mismatch is not a conflict under the v86 plan's M_prior definition.
        return numeric > max(float(threshold), 0.0)
    if signal == "anchor_absence_score":
        return numeric >= threshold
    return numeric >= threshold


def _signal_recall_fpr(df: pd.DataFrame, signal: str, threshold: float) -> dict[str, Any]:
    labelled = df[df["case_label"].isin(["bad", "good"])].copy()
    labelled = labelled[pd.to_numeric(labelled["abs_log_scale_jump"], errors="coerce").notna()].copy()
    if len(labelled) == 0:
        return {"recall": None, "fpr": None, "threshold": threshold, "flagged": 0}
    scale_threshold = float(np.quantile(pd.to_numeric(labelled["abs_log_scale_jump"], errors="coerce"), 0.75))
    labelled["high_scale"] = pd.to_numeric(labelled["abs_log_scale_jump"], errors="coerce") >= scale_threshold
    labelled["low_scale"] = ~labelled["high_scale"]
    labelled["flag"] = _signal_flags(labelled[signal], signal, threshold).fillna(False)
    recall = float((labelled["flag"] & labelled["high_scale"]).sum() / max(int(labelled["high_scale"].sum()), 1))
    fpr = float((labelled["flag"] & labelled["low_scale"]).sum() / max(int(labelled["low_scale"].sum()), 1))
    return {
        "recall": recall,
        "fpr": fpr,
        "threshold": threshold,
        "scale_high_threshold_q75": scale_threshold,
        "flagged": int(labelled["flag"].sum()),
    }


def main() -> None:
    args = parse_args()
    scale = pd.read_csv(args.root / "phase4_offline_scale_labels/offline_scale_jump_rows.csv")
    by_pair = pd.read_csv(args.root / "phase1_soft_pair_universe/soft_pair_by_seq_chunk.csv")
    prior = pd.read_csv(args.prior_dir / "historical_prior_rows.csv")
    cfit = pd.read_csv(args.phase2_dir / "c_fit_rows.csv")
    for frame in (scale, by_pair, prior, cfit):
        frame["seq"] = frame["seq"].astype(str).str.zfill(2)
    best_c = (
        cfit[cfit["direction"] == "current_to_history"]
        .sort_values("alignment_gain", ascending=False)
        .drop_duplicates(["seq", "prev_chunk", "curr_chunk"])
    )
    df = scale.merge(
        by_pair[["seq", "prev_chunk", "curr_chunk", "anchor_absence_score", "mean_risk_score", "mean_w_fit"]],
        on=["seq", "prev_chunk", "curr_chunk"],
        how="left",
    )
    df = df.merge(
        prior[["seq", "prev_chunk", "curr_chunk", "prior_mismatch_score", "prior_conflict_score"]],
        on=["seq", "prev_chunk", "curr_chunk"],
        how="left",
    )
    df = df.merge(
        best_c[
            [
                "seq",
                "prev_chunk",
                "curr_chunk",
                "heldout_identity_residual",
                "heldout_aligned_residual",
                "fro_norm_C_minus_I",
                "alignment_gain",
                "actual_minus_random_p95",
                "actual_minus_shuffle_p95",
            ]
        ],
        on=["seq", "prev_chunk", "curr_chunk"],
        how="left",
    )
    df["heldout_residual_drop"] = pd.to_numeric(df["heldout_identity_residual"], errors="coerce") - pd.to_numeric(
        df["heldout_aligned_residual"], errors="coerce"
    )
    signals = [
        "heldout_identity_residual",
        "heldout_aligned_residual",
        "heldout_residual_drop",
        "fro_norm_C_minus_I",
        "alignment_gain",
        "anchor_absence_score",
        "prior_mismatch_score",
        "prior_conflict_score",
    ]
    rows: list[dict[str, Any]] = []
    gate_any = False
    for signal in signals:
        rho = spearman_rho(df[signal].tolist(), df["abs_log_scale_jump"].tolist())
        available = df[pd.to_numeric(df[signal], errors="coerce").notna() & pd.to_numeric(df["abs_log_scale_jump"], errors="coerce").notna()]
        threshold_source = "q75"
        if signal == "anchor_absence_score":
            threshold = float(args.absence_threshold)
            threshold_source = "phase3_fixed_absence_threshold"
        elif signal in {"prior_mismatch_score", "prior_conflict_score"}:
            if len(available):
                threshold = max(0.0, float(np.quantile(pd.to_numeric(available[signal], errors="coerce"), 0.75)))
            else:
                threshold = float("nan")
            threshold_source = "max_zero_q75_positive_prior_conflict"
        elif len(available):
            threshold = float(np.quantile(pd.to_numeric(available[signal], errors="coerce"), 0.75))
        else:
            threshold = float("nan")
        rf = _signal_recall_fpr(df, signal, threshold)
        shuffled = df.copy()
        if len(shuffled) > 1:
            order = sorted(range(len(shuffled)), key=lambda i: stable_hash_float(signal, i))
            shuffled_vals = pd.to_numeric(shuffled["abs_log_scale_jump"], errors="coerce").to_numpy()
            shuffled_vals = shuffled_vals[order]
            shuffled["shuffled_abs_log_scale_jump"] = np.roll(shuffled_vals, 1)
        else:
            shuffled["shuffled_abs_log_scale_jump"] = shuffled["abs_log_scale_jump"]
        shuffled_rho = spearman_rho(df[signal].tolist(), shuffled["shuffled_abs_log_scale_jump"].tolist())
        margin = None if rho is None or shuffled_rho is None else float(rho - shuffled_rho)
        signal_pass = (
            (
                (rho is not None and rho >= 0.30)
                or (rf["recall"] is not None and rf["recall"] >= 0.60 and rf["fpr"] is not None and rf["fpr"] <= 0.25)
            )
            and margin is not None
            and margin >= 0.05
            and int(available["seq"].astype(str).str.zfill(2).nunique()) >= 3
        )
        gate_any = gate_any or signal_pass
        rows.append(
            {
                "signal": signal,
                "spearman_rho_abs_log_scale_jump": rho,
                "shuffled_spearman_rho": shuffled_rho,
                "rho_margin_vs_shuffled": margin,
                "high_scale_jump_recall": rf["recall"],
                "good_low_scale_jump_fpr": rf["fpr"],
                "signal_threshold_q75": rf["threshold"],
                "threshold_source": threshold_source,
                "scale_high_threshold_q75": rf.get("scale_high_threshold_q75"),
                "available_rows": int(len(available)),
                "sequence_coverage": int(available["seq"].astype(str).str.zfill(2).nunique()) if len(available) else 0,
                "signal_pass": signal_pass,
            }
        )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "scale_relevance_signal_rows.csv", df.to_dict("records"))
    write_csv(out_dir / "scale_relevance_summary_rows.csv", rows)
    summary = {
        "phase": "Phase4_scale_relevance",
        "phase4_scale_relevance_gate_pass": gate_any,
        "signals": rows,
        "scale_label_rows": int(pd.to_numeric(df["abs_log_scale_jump"], errors="coerce").notna().sum()),
        "sequence_coverage": int(df[pd.to_numeric(df["abs_log_scale_jump"], errors="coerce").notna()]["seq"].astype(str).str.zfill(2).nunique()),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "prior_dir": str(args.prior_dir),
        "phase2_dir": str(args.phase2_dir),
        "note": "Offline scale labels are diagnostic only and are never used as runtime triggers.",
    }
    write_json(out_dir / "scale_relevance_summary.json", summary)
    print(f"phase4_scale_relevance_gate_pass={gate_any}")
    for row in rows:
        print(
            f"{row['signal']}: rho={row['spearman_rho_abs_log_scale_jump']} margin={row['rho_margin_vs_shuffled']} "
            f"recall={row['high_scale_jump_recall']} fpr={row['good_low_scale_jump_fpr']} pass={row['signal_pass']}"
        )


if __name__ == "__main__":
    main()
