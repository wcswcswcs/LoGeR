#!/usr/bin/env python3
"""Audit v88 Phase2 scale-mode relevance and controls."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import safe_float, spearman_rho, stable_hash_float, write_csv, write_json


DEFAULT_PHASE1 = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase1_scale_mode_consensus_universe")
DEFAULT_V87_PHASE2 = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase2_scale_relevance_k16_r1_median_abs_highobs/no_gt_scale_proxy_rows.csv"
)
DEFAULT_OUT = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase2_scale_mode_relevance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--v87-phase2-rows", type=Path, default=DEFAULT_V87_PHASE2)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--filter", choices=["all", "highobs", "nonseq01", "near", "far"], default="all")
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


def _shuffle_series(values: pd.Series, salt: str) -> pd.Series:
    arr = values.to_numpy(copy=True)
    out = arr.copy()
    if len(arr) <= 1:
        return pd.Series(out, index=values.index)
    order = sorted(range(len(arr)), key=lambda i: stable_hash_float(salt, i))
    shuffled = arr[order]
    shuffled = np.roll(shuffled, 1)
    for dst, value in zip(order, shuffled):
        out[dst] = value
    return pd.Series(out, index=values.index)


def _same_sequence_shuffle(df: pd.DataFrame, column: str) -> pd.Series:
    out = df[column].copy()
    for seq, idx in df.groupby("seq").groups.items():
        local = df.loc[idx, column].reset_index(drop=True)
        out.loc[idx] = _shuffle_series(local, f"same_seq:{seq}:{column}").to_numpy()
    return out


def _quality_stratified_random_flags(df: pd.DataFrame, count: int, salt: str) -> pd.Series:
    flags = pd.Series(False, index=df.index)
    if count <= 0 or len(df) == 0:
        return flags
    remaining = count
    for _, idx in df.groupby("quality_type").groups.items():
        frac = len(idx) / max(len(df), 1)
        local_count = min(len(idx), int(round(count * frac)))
        if local_count <= 0:
            continue
        local_order = sorted(idx, key=lambda i: stable_hash_float(salt, i))
        flags.loc[local_order[:local_count]] = True
        remaining -= local_count
    if remaining > 0:
        candidates = [i for i in df.index if not bool(flags.loc[i])]
        order = sorted(candidates, key=lambda i: stable_hash_float(salt, "rem", i))
        flags.loc[order[:remaining]] = True
    return flags


def _flag_metrics(df: pd.DataFrame, flags: pd.Series, y_threshold: float) -> dict[str, Any]:
    labelled_y = _numeric(df["abs_log_scale_jump_gt"]).notna()
    high_scale = _numeric(df["abs_log_scale_jump_gt"]) >= y_threshold
    good_low = (df["base_case_type"] == "good") & (_numeric(df["abs_log_scale_jump_gt"]) < y_threshold)
    bad_label = df["base_case_type"] == "bad"
    good_label = df["base_case_type"] == "good"
    return {
        "high_scale_jump_recall": float((flags & high_scale).sum() / max(int(high_scale.sum()), 1)),
        "good_low_scale_fpr": float((flags & good_low).sum() / max(int(good_low.sum()), 1)),
        "bad_label_recall": float((flags & bad_label).sum() / max(int(bad_label.sum()), 1)),
        "good_label_fpr": float((flags & good_label).sum() / max(int(good_label.sum()), 1)),
        "labelled_rows": int(labelled_y.sum()),
        "flagged_rows": int(flags.sum()),
    }


def _seq01_stress_fraction(df: pd.DataFrame, flags: pd.Series) -> float:
    flagged = df[flags]
    if len(flagged) == 0:
        return 0.0
    seq01_stress = flagged[
        (flagged["seq"].astype(str).str.zfill(2) == "01")
        & (flagged["quality_type"].astype(str).str.contains("stress|low_conf", case=False, regex=True))
    ]
    return float(len(seq01_stress) / len(flagged))


def _prepare_rows(phase1: Path, v87_phase2: Path) -> pd.DataFrame:
    df = pd.read_csv(phase1 / "scale_mode_pair_rows.csv")
    df["seq"] = df["seq"].astype(str).str.zfill(2)
    df["prev_chunk"] = df["prev_chunk"].astype(int)
    df["curr_chunk"] = df["curr_chunk"].astype(int)
    if v87_phase2.exists():
        v87 = pd.read_csv(v87_phase2)
        v87["seq"] = v87["seq"].astype(str).str.zfill(2)
        v87["prev_chunk"] = v87["prev_chunk"].astype(int)
        v87["curr_chunk"] = v87["curr_chunk"].astype(int)
        keep = [
            "seq",
            "prev_chunk",
            "curr_chunk",
            "S_overlap",
            "S_shape",
            "S_geometry_only",
            "S_semantic_aware",
            "S_confidence_only",
        ]
        df = df.merge(v87[keep], on=["seq", "prev_chunk", "curr_chunk"], how="left", suffixes=("", "_v87"))
    else:
        for col in ["S_overlap", "S_shape", "S_geometry_only", "S_semantic_aware", "S_confidence_only"]:
            df[col] = np.nan
    sigma = float(_numeric(df["weighted_mode_mad"]).dropna().median())
    if not math.isfinite(sigma) or sigma <= 1e-9:
        sigma = 0.10
    obs = _numeric(df["observability_score"]).fillna(0.0).clip(0.0, 1.0)
    mass = _numeric(df["mode_mass_top1"]).fillna(0.0).clip(0.0, 1.0)
    mad_kernel = np.exp(-_numeric(df["weighted_mode_mad"]).fillna(sigma) / max(sigma, 1e-12))
    static_mass = _numeric(df["semantic_static_mass_in_mode"]).fillna(0.0).clip(0.0, 1.0)
    dynamic_mass = _numeric(df["semantic_dynamic_or_boundary_mass_in_mode"]).fillna(0.0).clip(0.0, 1.0)
    df["M0_v87_S_overlap_baseline"] = _numeric(df["S_overlap"])
    df["M1_weighted_mode_abs_mu"] = _numeric(df["weighted_mode_abs_mu"])
    df["M2_weighted_mode_mu_signed"] = _numeric(df["weighted_mode_mu"])
    df["M3_mode_mad"] = _numeric(df["weighted_mode_mad"])
    df["M4_mode_entropy"] = _numeric(df["mode_entropy"])
    df["M5_mode_gap_top1_top2"] = _numeric(df["mode_gap_top1_top2"])
    df["M6_native_mode_mismatch"] = _numeric(df["native_mode_mismatch"])
    df["M7_native_mode_sign_mismatch"] = df["native_mode_sign_mismatch"].astype(str).str.lower().isin(["true", "1"]).astype(float)
    df["M8_mode_risk_score"] = _numeric(df["native_mode_mismatch"]).fillna(0.0) * mass * mad_kernel * obs
    df["M9_semantic_aware_mode_risk"] = df["M8_mode_risk_score"] * (0.5 + 0.5 * static_mass) * (1.0 - 0.5 * dynamic_mass)
    df["M10_confidence_only_mode"] = _norm(df["effective_edge_sample_size"]) * obs
    df["M11_geometry_only_mode"] = _numeric(df["weighted_mode_abs_mu"]).fillna(0.0) * mass * mad_kernel * obs
    df["M12_observability_only"] = obs
    df["M13_semantic_conf_shuffle_control"] = df["M8_mode_risk_score"] * (0.5 + 0.5 * _shuffle_series(static_mass, "semantic_static_shuffle"))
    return df


def _apply_filter(df: pd.DataFrame, filter_name: str) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    out = df.copy()
    if filter_name == "highobs":
        before = len(out)
        q = _numeric(out["observability_score"]).quantile(0.50)
        out = out[_numeric(out["observability_score"]) >= q].copy()
        notes.append(f"highobs:q50={q}:{before}->{len(out)}")
    elif filter_name == "nonseq01":
        before = len(out)
        out = out[out["seq"].astype(str).str.zfill(2) != "01"].copy()
        notes.append(f"nonseq01:{before}->{len(out)}")
    elif filter_name in {"near", "far"}:
        # Approximate depth split with native_delta/mode magnitude when raw median depth is not available in v88 pair rows.
        before = len(out)
        q = _numeric(out["weighted_mode_abs_mu"]).quantile(0.50)
        if filter_name == "near":
            out = out[_numeric(out["weighted_mode_abs_mu"]) <= q].copy()
        else:
            out = out[_numeric(out["weighted_mode_abs_mu"]) > q].copy()
        notes.append(f"{filter_name}_mode_abs_mu_split:q50={q}:{before}->{len(out)}")
    return out, notes


def _evaluate_signal(df: pd.DataFrame, signal: str, y_threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    available = df[_numeric(df[signal]).notna() & _numeric(df["abs_log_scale_jump_gt"]).notna()].copy()
    controls: list[dict[str, Any]] = []
    if len(available) == 0:
        return {
            "signal": signal,
            "available_rows": 0,
            "sequence_coverage": 0,
            "spearman_rho_abs_log_scale_jump": None,
            "rho_margin_vs_shuffled": None,
            "high_scale_jump_recall": None,
            "good_low_scale_fpr": None,
            "bad_label_recall": None,
            "good_label_fpr": None,
            "seq01_stress_flagged_fraction": None,
            "signal_pass": False,
        }, controls
    rho = spearman_rho(available[signal].tolist(), available["abs_log_scale_jump_gt"].tolist())
    label_shuffle = _shuffle_series(available["abs_log_scale_jump_gt"], f"label:{signal}")
    same_seq_shuffle = _same_sequence_shuffle(available, "abs_log_scale_jump_gt")
    signal_shuffle = _shuffle_series(available[signal], f"signal:{signal}")
    label_shuffle_rho = spearman_rho(available[signal].tolist(), label_shuffle.tolist())
    same_seq_rho = spearman_rho(available[signal].tolist(), same_seq_shuffle.tolist())
    signal_shuffle_rho = spearman_rho(signal_shuffle.tolist(), available["abs_log_scale_jump_gt"].tolist())
    control_rhos = [v for v in [label_shuffle_rho, same_seq_rho, signal_shuffle_rho] if v is not None]
    max_control_rho = max(control_rhos) if control_rhos else None
    margin = None if rho is None or max_control_rho is None else float(rho - max_control_rho)
    threshold = float(_numeric(available[signal]).quantile(0.75))
    flags = _numeric(available[signal]) >= threshold
    flag_metrics = _flag_metrics(available, flags, y_threshold)
    seq_cov = int(available["seq"].astype(str).str.zfill(2).nunique())
    stress_fraction = _seq01_stress_fraction(available, flags)
    signal_pass = bool(
        len(available) >= 12
        and seq_cov >= 3
        and rho is not None
        and rho >= 0.30
        and margin is not None
        and margin >= 0.05
        and flag_metrics["high_scale_jump_recall"] >= 0.60
        and flag_metrics["good_low_scale_fpr"] <= 0.25
        and stress_fraction <= 0.50
    )
    random_flags = _quality_stratified_random_flags(available, int(flags.sum()), f"quality_random:{signal}")
    seq01_flags = (available["seq"].astype(str).str.zfill(2) == "01") & available["quality_type"].astype(str).str.contains(
        "stress|low_conf", case=False, regex=True
    )
    controls.extend(
        [
            {
                "signal": signal,
                "control": "same_sequence_scale_label_shuffle",
                "spearman_rho": same_seq_rho,
            },
            {
                "signal": signal,
                "control": "global_scale_label_shuffle",
                "spearman_rho": label_shuffle_rho,
            },
            {
                "signal": signal,
                "control": "shape_ratio_or_signal_shuffle",
                "spearman_rho": signal_shuffle_rho,
            },
            {
                "signal": signal,
                "control": "quality_stratified_random",
                **_flag_metrics(available, random_flags, y_threshold),
            },
            {
                "signal": signal,
                "control": "seq01_stress_only_control",
                **_flag_metrics(available, seq01_flags, y_threshold),
            },
        ]
    )
    if signal == "M9_semantic_aware_mode_risk":
        semantic_shuffle_rho = spearman_rho(available["M13_semantic_conf_shuffle_control"].tolist(), available["abs_log_scale_jump_gt"].tolist())
        controls.append({"signal": signal, "control": "semantic_conf_shuffle", "spearman_rho": semantic_shuffle_rho})
    return {
        "signal": signal,
        "available_rows": int(len(available)),
        "sequence_coverage": seq_cov,
        "spearman_rho_abs_log_scale_jump": rho,
        "max_control_rho": max_control_rho,
        "rho_margin_vs_shuffled": margin,
        "high_scale_jump_recall": flag_metrics["high_scale_jump_recall"],
        "good_low_scale_fpr": flag_metrics["good_low_scale_fpr"],
        "bad_label_recall": flag_metrics["bad_label_recall"],
        "good_label_fpr": flag_metrics["good_label_fpr"],
        "signal_threshold_q75": threshold,
        "scale_high_threshold_q75": y_threshold,
        "seq01_stress_flagged_fraction": stress_fraction,
        "signal_pass": signal_pass,
    }, controls


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _prepare_rows(args.phase1_dir, args.v87_phase2_rows)
    df, filter_notes = _apply_filter(df, args.filter)
    y = _numeric(df["abs_log_scale_jump_gt"]).dropna()
    y_threshold = float(y.quantile(0.75)) if len(y) else float("nan")
    signals = [
        "M0_v87_S_overlap_baseline",
        "M1_weighted_mode_abs_mu",
        "M2_weighted_mode_mu_signed",
        "M3_mode_mad",
        "M4_mode_entropy",
        "M5_mode_gap_top1_top2",
        "M6_native_mode_mismatch",
        "M7_native_mode_sign_mismatch",
        "M8_mode_risk_score",
        "M9_semantic_aware_mode_risk",
        "M10_confidence_only_mode",
        "M11_geometry_only_mode",
        "M12_observability_only",
    ]
    signal_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for signal in signals:
        row, controls = _evaluate_signal(df, signal, y_threshold)
        signal_rows.append(row)
        control_rows.extend(controls)
    pass_rows = [row for row in signal_rows if row["signal_pass"]]
    geometry_row = next((row for row in signal_rows if row["signal"] == "M11_geometry_only_mode"), {})
    semantic_row = next((row for row in signal_rows if row["signal"] == "M9_semantic_aware_mode_risk"), {})
    semantic_shuffle = next(
        (row for row in control_rows if row.get("signal") == "M9_semantic_aware_mode_risk" and row.get("control") == "semantic_conf_shuffle"),
        {},
    )
    semantic_lift = None
    if semantic_row.get("spearman_rho_abs_log_scale_jump") is not None and geometry_row.get("spearman_rho_abs_log_scale_jump") is not None:
        semantic_lift = float(semantic_row["spearman_rho_abs_log_scale_jump"] - geometry_row["spearman_rho_abs_log_scale_jump"])
    semantic_shuffle_margin = None
    if semantic_row.get("spearman_rho_abs_log_scale_jump") is not None and semantic_shuffle.get("spearman_rho") is not None:
        semantic_shuffle_margin = float(semantic_row["spearman_rho_abs_log_scale_jump"] - semantic_shuffle["spearman_rho"])
    semantic_aware_pass = bool(
        semantic_row.get("signal_pass")
        and semantic_lift is not None
        and semantic_lift >= 0.05
        and semantic_shuffle_margin is not None
        and semantic_shuffle_margin >= 0.05
        and semantic_row.get("good_low_scale_fpr", 1.0) <= geometry_row.get("good_low_scale_fpr", 0.0)
    )
    best = sorted(
        signal_rows,
        key=lambda r: (
            bool(r["signal_pass"]),
            -1e9 if r["spearman_rho_abs_log_scale_jump"] is None else float(r["spearman_rho_abs_log_scale_jump"]),
        ),
        reverse=True,
    )[0]
    summary = {
        "phase": "Phase2_scale_mode_relevance",
        "filter": args.filter,
        "filters": filter_notes,
        "phase2_mode_relevance_gate_pass": len(pass_rows) > 0,
        "passing_signals": [row["signal"] for row in pass_rows],
        "best_signal": best,
        "scale_label_rows": int(_numeric(df["abs_log_scale_jump_gt"]).notna().sum()),
        "sequence_coverage": int(df[_numeric(df["abs_log_scale_jump_gt"]).notna()]["seq"].astype(str).str.zfill(2).nunique()),
        "semantic_aware_pass": semantic_aware_pass,
        "semantic_lift_over_geometry_only": semantic_lift,
        "semantic_conf_shuffle_margin": semantic_shuffle_margin,
        "geometry_only_mode_pass": bool(geometry_row.get("signal_pass")),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "signals": signal_rows,
        "note": "Offline scale labels are audit-only. Signals are fixed-rule no-GT mode/mismatch statistics from Phase1.",
    }
    if not summary["phase2_mode_relevance_gate_pass"]:
        summary["blocker"] = "no_mode_or_mismatch_signal_passed"
    elif not semantic_aware_pass:
        summary["decision_label_hint"] = "D2_GEOMETRY_MODE_ONLY_SEMANTIC_NO_ADD" if summary["geometry_only_mode_pass"] else "semantic_no_add"
    write_csv(args.out_dir / "scale_mode_relevance_rows.csv", df.to_dict("records"))
    write_csv(args.out_dir / "scale_mode_relevance_by_signal.csv", signal_rows)
    write_csv(args.out_dir / "scale_mode_relevance_controls.csv", control_rows)
    write_json(args.out_dir / "scale_mode_relevance_summary.json", summary)
    report = [
        "# v88 Phase2 Scale-Mode Relevance",
        "",
        f"- filter: `{args.filter}`",
        f"- phase2_mode_relevance_gate_pass: `{summary['phase2_mode_relevance_gate_pass']}`",
        f"- passing_signals: `{summary['passing_signals']}`",
        f"- scale_label_rows: `{summary['scale_label_rows']}`",
        f"- sequence_coverage: `{summary['sequence_coverage']}`",
        f"- semantic_aware_pass: `{semantic_aware_pass}`",
        f"- semantic_lift_over_geometry_only: `{semantic_lift}`",
        "",
        "## Signals",
        "",
    ]
    for row in signal_rows:
        report.append(
            f"- {row['signal']}: rho={row['spearman_rho_abs_log_scale_jump']} margin={row['rho_margin_vs_shuffled']} "
            f"recall={row['high_scale_jump_recall']} good_low_fpr={row['good_low_scale_fpr']} "
            f"bad_recall={row['bad_label_recall']} good_fpr={row['good_label_fpr']} pass={row['signal_pass']}"
        )
    (args.out_dir / "scale_mode_relevance_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"phase2_mode_relevance_gate_pass={summary['phase2_mode_relevance_gate_pass']}")
    print(f"passing_signals={summary['passing_signals']}")
    print(f"semantic_aware_pass={semantic_aware_pass}")
    print(f"scale_label_rows={summary['scale_label_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    for row in signal_rows:
        print(
            f"{row['signal']}: rho={row['spearman_rho_abs_log_scale_jump']} margin={row['rho_margin_vs_shuffled']} "
            f"recall={row['high_scale_jump_recall']} good_low_fpr={row['good_low_scale_fpr']} pass={row['signal_pass']}"
        )


if __name__ == "__main__":
    main()
