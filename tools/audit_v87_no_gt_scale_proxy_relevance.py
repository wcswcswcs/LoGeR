#!/usr/bin/env python3
"""Audit ACL2 v87 Phase2 no-GT scale proxy relevance."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import spearman_rho, stable_hash_float, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier")
DEFAULT_PHASE1 = DEFAULT_ROOT / "phase1_scale_conditioned_pair_universe"
DEFAULT_PHASE2 = DEFAULT_ROOT / "phase2_scale_relevance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--labels", type=Path, default=DEFAULT_PHASE2 / "scale_jump_labels.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--exclude-seq01-stress", action="store_true")
    parser.add_argument("--high-observability-only", action="store_true")
    parser.add_argument("--depth-bin", choices=["all", "near", "far"], default="all")
    return parser.parse_args()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _norm(series: pd.Series) -> pd.Series:
    values = _numeric(series)
    lo = values.quantile(0.05)
    hi = values.quantile(0.95)
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or abs(float(hi - lo)) < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def _deterministic_shuffle(values: pd.Series, salt: str) -> pd.Series:
    arr = values.to_numpy(copy=True)
    if len(arr) <= 1:
        return pd.Series(arr, index=values.index)
    order = sorted(range(len(arr)), key=lambda i: stable_hash_float(salt, i))
    shuffled = arr[order]
    shuffled = np.roll(shuffled, 1)
    out = np.empty_like(arr)
    for dst, src_value in zip(order, shuffled):
        out[dst] = src_value
    return pd.Series(out, index=values.index)


def _same_sequence_shuffle(df: pd.DataFrame, label: str) -> pd.Series:
    out = df[label].copy()
    for seq, idx in df.groupby("seq").groups.items():
        local = df.loc[idx, label].reset_index(drop=True)
        shuffled = _deterministic_shuffle(local, f"same_seq:{seq}:{label}")
        out.loc[idx] = shuffled.to_numpy()
    return out


def _signal_recall_fpr(df: pd.DataFrame, signal: str, threshold: float, y_threshold: float) -> dict[str, Any]:
    labelled = df[df["base_case_type"].isin(["bad", "good"])].copy()
    labelled = labelled[_numeric(labelled["abs_log_scale_jump"]).notna() & _numeric(labelled[signal]).notna()].copy()
    if len(labelled) == 0:
        return {"recall": None, "good_low_scale_fpr": None, "flagged": 0, "scale_high_threshold_q75": y_threshold}
    labelled["high_scale"] = _numeric(labelled["abs_log_scale_jump"]) >= y_threshold
    labelled["good_low_scale"] = (labelled["base_case_type"] == "good") & (_numeric(labelled["abs_log_scale_jump"]) < y_threshold)
    labelled["flag"] = _numeric(labelled[signal]) >= threshold
    recall = float((labelled["flag"] & labelled["high_scale"]).sum() / max(int(labelled["high_scale"].sum()), 1))
    fpr = float((labelled["flag"] & labelled["good_low_scale"]).sum() / max(int(labelled["good_low_scale"].sum()), 1))
    return {
        "recall": recall,
        "good_low_scale_fpr": fpr,
        "flagged": int(labelled["flag"].sum()),
        "scale_high_threshold_q75": y_threshold,
    }


def _dominance(df: pd.DataFrame, signal: str, threshold: float) -> float:
    flagged = df[_numeric(df[signal]) >= threshold]
    if len(flagged) == 0:
        return 0.0
    seq01_stress = flagged[
        (flagged["seq"].astype(str).str.zfill(2) == "01")
        & ((flagged["quality_type"].astype(str) == "low_conf_stress") | (flagged["state_label"].astype(str) == "STRESS"))
    ]
    return float(len(seq01_stress) / len(flagged))


def _build_proxy_rows(by_pair: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    for frame in (by_pair, labels):
        frame["seq"] = frame["seq"].astype(str).str.zfill(2)
        frame["prev_chunk"] = frame["prev_chunk"].astype(int)
        frame["curr_chunk"] = frame["curr_chunk"].astype(int)
    df = by_pair.merge(
        labels[
            [
                "seq",
                "prev_chunk",
                "curr_chunk",
                "case_label",
                "quality_label",
                "abs_log_scale_jump",
                "adjacent_log_scale_jump",
                "offline_audit_label_only",
                "no_gt_runtime_feature",
            ]
        ],
        on=["seq", "prev_chunk", "curr_chunk"],
        how="left",
    )
    if "base_case_type" not in df.columns:
        df["base_case_type"] = df["case_label"]
    if "quality_type" not in df.columns:
        df["quality_type"] = df["quality_label"]
    df["S_shape"] = _numeric(df["weighted_median_local_shape_log_ratio"])
    df["S_conflict"] = _numeric(df["conflict_effective_sample_size"])
    df["S_support"] = _numeric(df["support_effective_sample_size"])
    df["S_absence"] = _numeric(df["absence_score"])
    df["S_obs"] = 1.0 - _numeric(df["observability_mean"])
    df["S_overlap"] = _numeric(df["mean_confidence_weighted_overlap_residual"])
    state_map = {"SUPPORT": 0.0, "CONFLICT": 0.75, "ABSENCE": 1.0, "STRESS": 0.80}
    df["S_state"] = df["state_label"].map(state_map).fillna(0.50)
    df["S_combined"] = (
        0.35 * _norm(df["S_shape"])
        + 0.25 * _norm(df["S_conflict"])
        + 0.20 * _norm(df["S_absence"])
        + 0.20 * _norm(df["S_overlap"])
    )
    df["S_geometry_only"] = 0.55 * _norm(df["S_shape"]) + 0.45 * _norm(df["S_overlap"])
    df["S_semantic_aware"] = (
        0.30 * _norm(df["S_shape"])
        + 0.25 * _norm(df["S_conflict"])
        + 0.20 * _norm(df["S_absence"])
        + 0.15 * _norm(df["S_overlap"])
        + 0.10 * _norm(df["S_state"])
    )
    df["S_confidence_only"] = _norm(df["S_obs"])
    df["S_semantic_only"] = _norm(df["S_state"])
    return df


def main() -> None:
    args = parse_args()
    by_pair = pd.read_csv(args.phase1_dir / "scale_conditioned_pair_by_adjacent.csv")
    labels = pd.read_csv(args.labels)
    df = _build_proxy_rows(by_pair, labels)

    filter_notes = []
    if args.exclude_seq01_stress:
        before = len(df)
        df = df[~((df["seq"].astype(str).str.zfill(2) == "01") & ((df["quality_type"].astype(str) == "low_conf_stress") | (df["state_label"].astype(str) == "STRESS")))].copy()
        filter_notes.append(f"exclude_seq01_stress:{before}->{len(df)}")
    if args.high_observability_only:
        before = len(df)
        threshold = _numeric(df["observability_mean"]).quantile(0.50)
        df = df[_numeric(df["observability_mean"]) >= threshold].copy()
        filter_notes.append(f"high_observability_only:q50={threshold}:{before}->{len(df)}")
    if args.depth_bin != "all":
        before = len(df)
        depth = _numeric(df["median_depth"])
        threshold = depth.quantile(0.50)
        if args.depth_bin == "near":
            df = df[depth <= threshold].copy()
        else:
            df = df[depth > threshold].copy()
        filter_notes.append(f"depth_bin={args.depth_bin}:q50={threshold}:{before}->{len(df)}")

    signals = [
        "S_shape",
        "S_conflict",
        "S_support",
        "S_absence",
        "S_obs",
        "S_overlap",
        "S_state",
        "S_combined",
        "S_geometry_only",
        "S_semantic_aware",
        "S_confidence_only",
        "S_semantic_only",
    ]
    available_y = _numeric(df["abs_log_scale_jump"]).dropna()
    y_threshold = float(available_y.quantile(0.75)) if len(available_y) else float("nan")
    rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    gate_any = False
    geometry_only_pass = False
    semantic_pass = False
    for signal in signals:
        available = df[_numeric(df[signal]).notna() & _numeric(df["abs_log_scale_jump"]).notna()].copy()
        rho = spearman_rho(available[signal].tolist(), available["abs_log_scale_jump"].tolist()) if len(available) else None
        shuffled_label = _deterministic_shuffle(available["abs_log_scale_jump"], f"scale_shuffle:{signal}") if len(available) else pd.Series(dtype=float)
        same_seq_label = _same_sequence_shuffle(available, "abs_log_scale_jump") if len(available) else pd.Series(dtype=float)
        shuffle_rho = spearman_rho(available[signal].tolist(), shuffled_label.tolist()) if len(available) else None
        same_seq_rho = spearman_rho(available[signal].tolist(), same_seq_label.tolist()) if len(available) else None
        margin = None if rho is None or shuffle_rho is None else float(rho - shuffle_rho)
        threshold = float(_numeric(available[signal]).quantile(0.75)) if len(available) else float("nan")
        rf = _signal_recall_fpr(available, signal, threshold, y_threshold) if len(available) else {"recall": None, "good_low_scale_fpr": None, "flagged": 0, "scale_high_threshold_q75": y_threshold}
        dominance = _dominance(available, signal, threshold) if len(available) else 0.0
        seq_cov = int(available["seq"].astype(str).str.zfill(2).nunique()) if len(available) else 0
        signal_pass = bool(
            (
                (rho is not None and rho >= 0.30)
                or (rf["recall"] is not None and rf["recall"] >= 0.60 and rf["good_low_scale_fpr"] is not None and rf["good_low_scale_fpr"] <= 0.25)
            )
            and margin is not None
            and margin >= 0.05
            and seq_cov >= 3
            and dominance <= 0.50
        )
        gate_any = gate_any or signal_pass
        if signal_pass and signal == "S_geometry_only":
            geometry_only_pass = True
        if signal_pass and signal in {"S_semantic_aware", "S_state", "S_semantic_only", "S_combined"}:
            semantic_pass = True
        rows.append(
            {
                "signal": signal,
                "spearman_rho_abs_log_scale_jump": rho,
                "shuffled_spearman_rho": shuffle_rho,
                "same_sequence_shuffled_spearman_rho": same_seq_rho,
                "rho_margin_vs_shuffled": margin,
                "high_scale_jump_recall": rf["recall"],
                "good_low_scale_fpr": rf["good_low_scale_fpr"],
                "signal_threshold_q75": threshold,
                "scale_high_threshold_q75": y_threshold,
                "available_rows": int(len(available)),
                "sequence_coverage": seq_cov,
                "seq01_stress_flagged_fraction": dominance,
                "signal_pass": signal_pass,
            }
        )
        controls.append(
            {
                "signal": signal,
                "scale_label_shuffle_rho": shuffle_rho,
                "same_sequence_shuffle_rho": same_seq_rho,
                "confidence_only_reference_rho": spearman_rho(available["S_confidence_only"].tolist(), available["abs_log_scale_jump"].tolist())
                if len(available) and "S_confidence_only" in available
                else None,
                "geometry_only_reference_rho": spearman_rho(available["S_geometry_only"].tolist(), available["abs_log_scale_jump"].tolist())
                if len(available) and "S_geometry_only" in available
                else None,
                "semantic_only_reference_rho": spearman_rho(available["S_semantic_only"].tolist(), available["abs_log_scale_jump"].tolist())
                if len(available) and "S_semantic_only" in available
                else None,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "no_gt_scale_proxy_rows.csv", df.to_dict("records"))
    write_csv(args.out_dir / "proxy_relevance_by_signal.csv", rows)
    write_csv(args.out_dir / "proxy_relevance_controls.csv", controls)
    best = sorted(
        rows,
        key=lambda r: (
            bool(r["signal_pass"]),
            -1e9 if r["spearman_rho_abs_log_scale_jump"] is None else float(r["spearman_rho_abs_log_scale_jump"]),
        ),
        reverse=True,
    )[0] if rows else {}
    summary = {
        "phase": "Phase2_no_gt_scale_proxy_relevance",
        "phase2_scale_proxy_gate_pass": gate_any,
        "geometry_only_pass": geometry_only_pass,
        "semantic_signal_pass": semantic_pass,
        "best_signal": best,
        "signals": rows,
        "scale_label_rows": int(_numeric(df["abs_log_scale_jump"]).notna().sum()),
        "sequence_coverage": int(df[_numeric(df["abs_log_scale_jump"]).notna()]["seq"].astype(str).str.zfill(2).nunique()),
        "filters": filter_notes,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "note": "Offline scale labels are audit-only. All S_* signals are computed from v87 no-GT Phase1 aggregates.",
    }
    write_json(args.out_dir / "proxy_relevance_summary.json", summary)
    report_lines = [
        "# v87 Phase2 No-GT Scale Proxy Relevance",
        "",
        f"- phase2_scale_proxy_gate_pass: `{gate_any}`",
        f"- geometry_only_pass: `{geometry_only_pass}`",
        f"- semantic_signal_pass: `{semantic_pass}`",
        f"- scale_label_rows: `{summary['scale_label_rows']}`",
        f"- sequence_coverage: `{summary['sequence_coverage']}`",
        f"- filters: `{filter_notes}`",
        "",
        "## Signals",
        "",
    ]
    for row in rows:
        report_lines.append(
            f"- {row['signal']}: rho={row['spearman_rho_abs_log_scale_jump']} margin={row['rho_margin_vs_shuffled']} "
            f"recall={row['high_scale_jump_recall']} good_low_fpr={row['good_low_scale_fpr']} "
            f"seq01_stress_fraction={row['seq01_stress_flagged_fraction']} pass={row['signal_pass']}"
        )
    (args.out_dir / "proxy_relevance_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"phase2_scale_proxy_gate_pass={gate_any}")
    print(f"geometry_only_pass={geometry_only_pass}")
    print(f"semantic_signal_pass={semantic_pass}")
    for row in rows:
        print(
            f"{row['signal']}: rho={row['spearman_rho_abs_log_scale_jump']} "
            f"margin={row['rho_margin_vs_shuffled']} recall={row['high_scale_jump_recall']} "
            f"good_low_fpr={row['good_low_scale_fpr']} pass={row['signal_pass']}"
        )


if __name__ == "__main__":
    main()
