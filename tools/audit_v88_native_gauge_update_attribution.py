#!/usr/bin/env python3
"""Audit v88 Phase3 native gauge-update attribution variants."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v86_soft_latent_utils import safe_float, spearman_rho, stable_hash_float, write_csv, write_json


DEFAULT_PHASE1 = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase1_scale_mode_consensus_universe")
DEFAULT_V87_CF = Path(
    "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase8_merge_gauge_direct_pair_weighting/raw_overlap_geometry_counterfactual_rows.csv"
)
DEFAULT_OUT = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase3_native_gauge_update_attribution")

ATTRIBUTION_CLASSES = [
    "AGREE_GOOD",
    "AGREE_BAD",
    "MISMATCH_BAD",
    "MISMATCH_GOOD",
    "MULTIMODE_UNSAFE",
    "LOWOBS_ABSTAIN",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--v87-counterfactual-rows", type=Path, default=DEFAULT_V87_CF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--filter", choices=["all", "highobs", "nonseq01", "near", "far"], default="all")
    return parser.parse_args()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


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


def _load_rows(phase1_dir: Path, v87_cf: Path) -> pd.DataFrame:
    df = pd.read_csv(phase1_dir / "scale_mode_pair_rows.csv")
    df["seq"] = df["seq"].astype(str).str.zfill(2)
    df["prev_chunk"] = df["prev_chunk"].astype(int)
    df["curr_chunk"] = df["curr_chunk"].astype(int)
    if v87_cf.exists():
        cf = pd.read_csv(v87_cf)
        cf["seq"] = cf["seq"].astype(str).str.zfill(2)
        cf["prev_chunk"] = cf["prev_chunk"].astype(int)
        cf["curr_chunk"] = cf["curr_chunk"].astype(int)
        keep = ["seq", "prev_chunk", "curr_chunk", "native_eval_rmse"]
        df = df.merge(cf[keep], on=["seq", "prev_chunk", "curr_chunk"], how="left")
    else:
        df["native_eval_rmse"] = np.nan
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
        before = len(out)
        q = _numeric(out["weighted_mode_abs_mu"]).quantile(0.50)
        if filter_name == "near":
            out = out[_numeric(out["weighted_mode_abs_mu"]) <= q].copy()
        else:
            out = out[_numeric(out["weighted_mode_abs_mu"]) > q].copy()
        notes.append(f"{filter_name}_mode_abs_mu_split:q50={q}:{before}->{len(out)}")
    return out, notes


def _variant_flags(df: pd.DataFrame, variant: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mismatch = _numeric(df["native_mode_mismatch"])
    sign = df["native_mode_sign_mismatch"].astype(str).str.lower().isin(["true", "1"])
    mismatch_q75 = mismatch.quantile(0.75)
    base_flag = (mismatch >= mismatch_q75) | sign
    entropy = _numeric(df["mode_entropy"])
    gap = _numeric(df["mode_gap_top1_top2"])
    obs = _numeric(df["observability_score"])
    static = _numeric(df["semantic_static_mass_in_mode"])
    dynamic = _numeric(df["semantic_dynamic_or_boundary_mass_in_mode"])
    native_residual = _numeric(df["native_eval_rmse"])
    multimode = (entropy >= entropy.quantile(0.75)) | (gap <= gap.quantile(0.25))
    lowobs = obs < obs.quantile(0.25)
    if variant == "mismatch_q75":
        flag = base_flag
    elif variant == "sign_mismatch_only":
        flag = sign
    elif variant == "entropy_gap_guard":
        flag = base_flag & (~multimode)
    elif variant == "highobs_guard":
        flag = base_flag & (obs >= obs.quantile(0.50))
    elif variant == "semantic_stable_guard":
        flag = base_flag & (static >= static.quantile(0.50)) & (dynamic <= dynamic.quantile(0.50))
    elif variant == "native_residual_low_guard":
        flag = base_flag & (native_residual >= native_residual.quantile(0.50))
    elif variant == "combined_guard":
        flag = base_flag & (~multimode) & (obs >= obs.quantile(0.50)) & (static >= static.quantile(0.50)) & (dynamic <= dynamic.quantile(0.50))
    else:
        raise ValueError(f"unknown variant {variant}")
    flag = flag.fillna(False)
    return flag, multimode.fillna(False), lowobs.fillna(False), mismatch


def _classes(df: pd.DataFrame, flags: pd.Series, multimode: pd.Series, lowobs: pd.Series, y_threshold: float) -> pd.Series:
    high_or_bad = (_numeric(df["abs_log_scale_jump_gt"]) >= y_threshold) | (df["base_case_type"] == "bad")
    good_or_low = (df["base_case_type"] == "good") | (_numeric(df["abs_log_scale_jump_gt"]) < y_threshold)
    classes = pd.Series("LOWOBS_ABSTAIN", index=df.index, dtype=object)
    classes.loc[~lowobs] = "AGREE_GOOD"
    classes.loc[(~lowobs) & high_or_bad] = "AGREE_BAD"
    classes.loc[(~lowobs) & flags & high_or_bad] = "MISMATCH_BAD"
    classes.loc[(~lowobs) & flags & good_or_low & (~high_or_bad)] = "MISMATCH_GOOD"
    classes.loc[(~lowobs) & multimode & (~flags)] = "MULTIMODE_UNSAFE"
    return classes


def _metrics(df: pd.DataFrame, flags: pd.Series, classes: pd.Series, mismatch_score: pd.Series, y_threshold: float) -> dict[str, Any]:
    high_or_bad = (_numeric(df["abs_log_scale_jump_gt"]) >= y_threshold) | (df["base_case_type"] == "bad")
    good_or_low = (df["base_case_type"] == "good") | (_numeric(df["abs_log_scale_jump_gt"]) < y_threshold)
    good_low = (df["base_case_type"] == "good") & (_numeric(df["abs_log_scale_jump_gt"]) < y_threshold)
    agree_good = classes == "AGREE_GOOD"
    rho = spearman_rho(mismatch_score.tolist(), df["abs_log_scale_jump_gt"].tolist())
    shuffled_mode = _shuffle_series(_numeric(df["weighted_mode_mu"]), "phase3_shape_ratio_shuffle")
    shape_shuffle_score = np.abs(_numeric(df["native_delta_log_scale"]) - shuffled_mode)
    shape_shuffle_rho = spearman_rho(shape_shuffle_score.tolist(), df["abs_log_scale_jump_gt"].tolist())
    random_flags = pd.Series(False, index=df.index)
    order = sorted(list(df.index), key=lambda i: stable_hash_float("phase3_same_count_random", i))
    random_flags.loc[order[: int(flags.sum())]] = True
    random_metrics = {
        "same_count_random_bad_recall": float((random_flags & high_or_bad).sum() / max(int(high_or_bad.sum()), 1)),
        "same_count_random_good_fpr": float((random_flags & good_low).sum() / max(int(good_low.sum()), 1)),
    }
    margin = None if rho is None or shape_shuffle_rho is None else float(rho - shape_shuffle_rho)
    class_counts = classes.value_counts().to_dict()
    return {
        "MISMATCH_BAD_recall": float(((classes == "MISMATCH_BAD") & high_or_bad).sum() / max(int(high_or_bad.sum()), 1)),
        "MISMATCH_GOOD_FPR": float(((classes == "MISMATCH_GOOD") & good_low).sum() / max(int(good_low.sum()), 1)),
        "AGREE_GOOD_precision": float(((agree_good) & good_or_low).sum() / max(int(agree_good.sum()), 1)),
        "MULTIMODE_UNSAFE_rate_bad_or_high": float(((classes == "MULTIMODE_UNSAFE") & high_or_bad).sum() / max(int(high_or_bad.sum()), 1)),
        "MULTIMODE_UNSAFE_rate_good_low": float(((classes == "MULTIMODE_UNSAFE") & good_low).sum() / max(int(good_low.sum()), 1)),
        "native_mode_mismatch_rho_abs_log_scale_jump": rho,
        "shape_ratio_shuffle_rho": shape_shuffle_rho,
        "rho_margin_vs_shape_shuffle": margin,
        "sequence_coverage": int(df[_numeric(df["abs_log_scale_jump_gt"]).notna()]["seq"].astype(str).str.zfill(2).nunique()),
        "visual_sample_classes": sorted([key for key, value in class_counts.items() if value > 0]),
        "class_counts": class_counts,
        **random_metrics,
    }


def _plot_samples(df: pd.DataFrame, classes: pd.Series, out_dir: Path, variant: str) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for cls in ATTRIBUTION_CLASSES:
        subset = df[classes == cls]
        if len(subset) == 0:
            continue
        row = subset.iloc[0]
        name = f"{variant}_{cls}_{row['seq']}_{int(row['prev_chunk']):03d}_{int(row['curr_chunk']):03d}.png"
        path = out_dir / name
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        labels = ["mode_mu", "native_delta", "mismatch", "entropy", "gap"]
        values = [
            safe_float(row.get("weighted_mode_mu")) or 0.0,
            safe_float(row.get("native_delta_log_scale")) or 0.0,
            safe_float(row.get("native_mode_mismatch")) or 0.0,
            safe_float(row.get("mode_entropy")) or 0.0,
            safe_float(row.get("mode_gap_top1_top2")) or 0.0,
        ]
        ax.bar(labels, values, color=["#3366AA", "#228833", "#BB2222", "#AA7733", "#663399"])
        ax.set_title(f"{variant} {cls} {row['seq']} {int(row['prev_chunk'])}->{int(row['curr_chunk'])}")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        samples.append(
            {
                "variant": variant,
                "attribution_class": cls,
                "seq": row["seq"],
                "prev_chunk": int(row["prev_chunk"]),
                "curr_chunk": int(row["curr_chunk"]),
                "panel_path": str(path),
            }
        )
    return samples


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_rows(args.phase1_dir, args.v87_counterfactual_rows)
    df, filter_notes = _apply_filter(df, args.filter)
    y = _numeric(df["abs_log_scale_jump_gt"]).dropna()
    y_threshold = float(y.quantile(0.75)) if len(y) else float("nan")
    variants = [
        "mismatch_q75",
        "sign_mismatch_only",
        "entropy_gap_guard",
        "highobs_guard",
        "semantic_stable_guard",
        "native_residual_low_guard",
        "combined_guard",
    ]
    metric_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for variant in variants:
        flags, multimode, lowobs, score = _variant_flags(df, variant)
        classes = _classes(df, flags, multimode, lowobs, y_threshold)
        metrics = _metrics(df, flags, classes, score, y_threshold)
        sample_rows = _plot_samples(df, classes, args.out_dir / "attribution_visual_samples", variant)
        samples.extend(sample_rows)
        visual_sample_complete = all(cls in {row["attribution_class"] for row in sample_rows} for cls in metrics["visual_sample_classes"])
        variant_pass = bool(
            metrics["MISMATCH_BAD_recall"] >= 0.60
            and metrics["MISMATCH_GOOD_FPR"] <= 0.25
            and metrics["sequence_coverage"] >= 3
            and metrics["native_mode_mismatch_rho_abs_log_scale_jump"] is not None
            and metrics["native_mode_mismatch_rho_abs_log_scale_jump"] >= 0.30
            and metrics["rho_margin_vs_shape_shuffle"] is not None
            and metrics["rho_margin_vs_shape_shuffle"] >= 0.05
            and visual_sample_complete
        )
        metric_row = {
            "variant": variant,
            "variant_pass": variant_pass,
            "visual_sample_complete_for_present_classes": visual_sample_complete,
            **metrics,
        }
        metric_rows.append(metric_row)
        for idx, row in df.iterrows():
            attribution_rows.append(
                {
                    "variant": variant,
                    "seq": row["seq"],
                    "prev_chunk": int(row["prev_chunk"]),
                    "curr_chunk": int(row["curr_chunk"]),
                    "base_case_type": row.get("base_case_type", ""),
                    "quality_type": row.get("quality_type", ""),
                    "abs_log_scale_jump_gt": row.get("abs_log_scale_jump_gt", ""),
                    "offline_audit_label_only": True,
                    "native_delta_log_scale": row.get("native_delta_log_scale", ""),
                    "weighted_mode_mu": row.get("weighted_mode_mu", ""),
                    "native_mode_mismatch": row.get("native_mode_mismatch", ""),
                    "native_mode_sign_mismatch": row.get("native_mode_sign_mismatch", ""),
                    "mode_entropy": row.get("mode_entropy", ""),
                    "mode_gap_top1_top2": row.get("mode_gap_top1_top2", ""),
                    "observability_score": row.get("observability_score", ""),
                    "flagged_mismatch": bool(flags.loc[idx]),
                    "attribution_class": classes.loc[idx],
                }
            )
    pass_rows = [row for row in metric_rows if row["variant_pass"]]
    best = sorted(
        metric_rows,
        key=lambda r: (
            bool(r["variant_pass"]),
            float(r["MISMATCH_BAD_recall"]),
            -float(r["MISMATCH_GOOD_FPR"]),
            -1e9
            if r["native_mode_mismatch_rho_abs_log_scale_jump"] is None
            else float(r["native_mode_mismatch_rho_abs_log_scale_jump"]),
        ),
        reverse=True,
    )[0]
    summary = {
        "phase": "Phase3_native_gauge_update_attribution",
        "filter": args.filter,
        "filters": filter_notes,
        "phase3_native_update_attribution_gate_pass": len(pass_rows) > 0,
        "passing_variants": [row["variant"] for row in pass_rows],
        "best_variant": best,
        "scale_label_rows": int(_numeric(df["abs_log_scale_jump_gt"]).notna().sum()),
        "sequence_coverage": int(df[_numeric(df["abs_log_scale_jump_gt"]).notna()]["seq"].astype(str).str.zfill(2).nunique()),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "note": "GT-derived scale labels are used only for offline audit metrics. Variants are fixed-rule guard audits.",
    }
    if not summary["phase3_native_update_attribution_gate_pass"]:
        summary["blocker"] = "native_mode_mismatch_did_not_meet_recall_fpr_rho_control_gate"
    write_csv(args.out_dir / "native_gauge_update_attribution_rows.csv", attribution_rows)
    write_csv(args.out_dir / "native_gauge_update_attribution_by_variant.csv", metric_rows)
    write_csv(args.out_dir / "attribution_visual_samples.csv", samples)
    write_json(args.out_dir / "native_gauge_update_attribution_summary.json", summary)
    report = [
        "# v88 Phase3 Native Gauge-Update Attribution",
        "",
        f"- filter: `{args.filter}`",
        f"- phase3 gate pass: `{summary['phase3_native_update_attribution_gate_pass']}`",
        f"- passing_variants: `{summary['passing_variants']}`",
        f"- scale_label_rows: `{summary['scale_label_rows']}`",
        f"- sequence_coverage: `{summary['sequence_coverage']}`",
        "",
        "## Variants",
        "",
    ]
    for row in metric_rows:
        report.append(
            f"- {row['variant']}: pass={row['variant_pass']} recall={row['MISMATCH_BAD_recall']} "
            f"good_fpr={row['MISMATCH_GOOD_FPR']} rho={row['native_mode_mismatch_rho_abs_log_scale_jump']} "
            f"margin={row['rho_margin_vs_shape_shuffle']} classes={row['class_counts']}"
        )
    (args.out_dir / "native_gauge_update_attribution_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"phase3_native_update_attribution_gate_pass={summary['phase3_native_update_attribution_gate_pass']}")
    print(f"passing_variants={summary['passing_variants']}")
    print(f"scale_label_rows={summary['scale_label_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    for row in metric_rows:
        print(
            f"{row['variant']}: pass={row['variant_pass']} recall={row['MISMATCH_BAD_recall']} "
            f"good_fpr={row['MISMATCH_GOOD_FPR']} rho={row['native_mode_mismatch_rho_abs_log_scale_jump']} "
            f"margin={row['rho_margin_vs_shape_shuffle']}"
        )


if __name__ == "__main__":
    main()
