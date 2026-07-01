#!/usr/bin/env python3
"""Diagnose v88 Phase5 mode-aware counterfactual upper bounds."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import stable_hash_float, write_csv, write_json


DEFAULT_PHASE1 = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase1_scale_mode_consensus_universe")
DEFAULT_OUT = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase5_mode_aware_counterfactual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _shuffle(values: pd.Series, salt: str) -> pd.Series:
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


def _candidate_deltas(df: pd.DataFrame) -> dict[str, pd.Series]:
    native = _numeric(df["native_delta_log_scale"]).fillna(0.0)
    mode = _numeric(df["weighted_mode_mu"]).fillna(0.0)
    trimmed = _numeric(df["weighted_trimmed_mean"]).fillna(mode)
    mismatch = _numeric(df["native_mode_mismatch"]).fillna(0.0)
    mass = _numeric(df["mode_mass_top1"]).fillna(0.0).clip(0.0, 1.0)
    confidence = _numeric(df["mode_confidence"]).fillna(0.0).clip(0.0, 1.0)
    entropy = _numeric(df["mode_entropy"]).fillna(0.0)
    gap = _numeric(df["mode_gap_top1_top2"]).fillna(0.0)
    static = _numeric(df["semantic_static_mass_in_mode"]).fillna(0.0).clip(0.0, 1.0)
    dynamic = _numeric(df["semantic_dynamic_or_boundary_mass_in_mode"]).fillna(0.0).clip(0.0, 1.0)
    sign_mismatch = df["native_mode_sign_mismatch"].astype(str).str.lower().isin(["true", "1"])
    mismatch_high = mismatch >= mismatch.quantile(0.75)
    multimode = (entropy >= entropy.quantile(0.75)) | (gap <= gap.quantile(0.25))
    robust_alpha = (0.25 + 0.75 * confidence * mass).clip(0.0, 1.0)
    semantic_alpha = (robust_alpha * (0.5 + 0.5 * static) * (1.0 - 0.5 * dynamic)).clip(0.0, 1.0)
    shuffled_mode = _shuffle(mode, "phase5_shape_ratio_shuffle")
    shuffled_static = _shuffle(static, "phase5_semantic_conf_shuffle")
    random_delta = _shuffle(native + robust_alpha * (mode - native), "phase5_same_count_random")
    return {
        "CF0_native": native,
        "CF1_dominant_mode_only": mode,
        "CF2_outlier_mode_downweight": trimmed,
        "CF3_mode_consensus_robust": native + robust_alpha * (mode - native),
        "CF4_native_mode_mismatch_hold": native.where(~(mismatch_high | sign_mismatch), 0.0),
        "CF5_multimode_abstain": native.where(~multimode, 0.0),
        "CF6_geometry_only_mode": native + (0.25 + 0.75 * mass).clip(0.0, 1.0) * (mode - native),
        "CF7_semantic_guarded_mode": native + semantic_alpha * (mode - native),
        "CF8_same_count_random": random_delta,
        "CF9_shape_ratio_shuffle": shuffled_mode,
        "CF10_semantic_conf_shuffle": native + (0.25 + 0.75 * shuffled_static).clip(0.0, 1.0) * (mode - native),
        "CF11_confidence_only": native + confidence * (mode - native),
    }


def _summarize_family(df: pd.DataFrame, family: str, cf_delta: pd.Series, native_abs_error: pd.Series) -> dict[str, Any]:
    labelled = df[_numeric(df["abs_log_scale_jump_gt"]).notna()].copy()
    cf_delta = cf_delta.loc[labelled.index]
    native_err = native_abs_error.loc[labelled.index]
    gt_abs = _numeric(labelled["abs_log_scale_jump_gt"])
    cf_abs_error = (cf_delta.abs() - gt_abs).abs()
    improvement = (native_err - cf_abs_error) / native_err.clip(lower=1e-9)
    bad = labelled["base_case_type"] == "bad"
    good = labelled["base_case_type"] == "good"
    controls = {"CF8_same_count_random", "CF9_shape_ratio_shuffle", "CF10_semantic_conf_shuffle", "CF11_confidence_only"}
    return {
        "family": family,
        "labelled_rows": int(len(labelled)),
        "sequence_coverage": int(labelled["seq"].astype(str).str.zfill(2).nunique()),
        "bad_rows": int(bad.sum()),
        "good_rows": int(good.sum()),
        "bad_median_I_scale": float(improvement[bad].median()) if int(bad.sum()) else None,
        "good_max_scale_error_worsen": float(np.maximum(-improvement[good], 0.0).max()) if int(good.sum()) else None,
        "all_median_I_scale": float(improvement.median()) if len(improvement) else None,
        "is_control": family in controls,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.phase1_dir / "scale_mode_pair_rows.csv")
    df["seq"] = df["seq"].astype(str).str.zfill(2)
    native = _numeric(df["native_delta_log_scale"]).fillna(0.0)
    gt_abs = _numeric(df["abs_log_scale_jump_gt"])
    native_abs_error = (native.abs() - gt_abs).abs()
    families = _candidate_deltas(df)
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for family, delta in families.items():
        rows.append(_summarize_family(df, family, delta, native_abs_error))
        labelled = df[_numeric(df["abs_log_scale_jump_gt"]).notna()].copy()
        for idx, item in labelled.iterrows():
            native_err = float(native_abs_error.loc[idx])
            cf_err = abs(abs(float(delta.loc[idx])) - float(item["abs_log_scale_jump_gt"]))
            detail_rows.append(
                {
                    "family": family,
                    "seq": item["seq"],
                    "prev_chunk": int(item["prev_chunk"]),
                    "curr_chunk": int(item["curr_chunk"]),
                    "base_case_type": item.get("base_case_type", ""),
                    "quality_type": item.get("quality_type", ""),
                    "offline_audit_label_only": True,
                    "abs_log_scale_jump_gt": item["abs_log_scale_jump_gt"],
                    "native_delta_log_scale": native.loc[idx],
                    "cf_delta_log_scale": delta.loc[idx],
                    "native_abs_scale_error": native_err,
                    "cf_abs_scale_error": cf_err,
                    "I_scale": (native_err - cf_err) / max(native_err, 1e-9),
                }
            )
    non_control = [row for row in rows if not row["is_control"] and row["family"] != "CF0_native"]
    control_bad_best = max(
        [row["bad_median_I_scale"] for row in rows if row["is_control"] and row["bad_median_I_scale"] is not None],
        default=None,
    )
    for row in rows:
        row["beats_control_bad_median"] = (
            False
            if row["bad_median_I_scale"] is None or control_bad_best is None or row["is_control"]
            else row["bad_median_I_scale"] > control_bad_best
        )
        row["scale_label_gate_pass"] = bool(
            not row["is_control"]
            and row["family"] != "CF0_native"
            and row["bad_median_I_scale"] is not None
            and row["bad_median_I_scale"] >= 0.10
            and row["good_max_scale_error_worsen"] is not None
            and row["good_max_scale_error_worsen"] <= 0.02
            and row["sequence_coverage"] >= 3
            and row["beats_control_bad_median"]
        )
    pass_rows = [row for row in rows if row["scale_label_gate_pass"]]
    best = sorted(
        [row for row in non_control if row["bad_median_I_scale"] is not None],
        key=lambda row: (bool(row["scale_label_gate_pass"]), float(row["bad_median_I_scale"]), -float(row["good_max_scale_error_worsen"] or 1e9)),
        reverse=True,
    )[0]
    summary = {
        "phase": "Phase5_mode_aware_counterfactual",
        "scale_label_gate_pass": len(pass_rows) > 0,
        "raw_residual_gate_pass": False,
        "raw_residual_counterfactual_available": False,
        "passing_families": [row["family"] for row in pass_rows],
        "best_family": best,
        "control_bad_best_median_I_scale": control_bad_best,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "note": "This is an offline audit-only scale-label upper-bound. It does not apply runtime scale correction and does not compute mode-aware raw residual fitting.",
    }
    if not pass_rows:
        summary["blocker"] = "all_mode_aware_scale_label_counterfactuals_failed"
    elif not summary["raw_residual_counterfactual_available"]:
        summary["blocker"] = "scale_label_counterfactual_pass_without_raw_residual_or_carrier_runtime_invalid"
    write_csv(args.out_dir / "mode_aware_counterfactual_rows.csv", detail_rows)
    write_csv(args.out_dir / "mode_aware_counterfactual_by_family.csv", rows)
    write_json(args.out_dir / "mode_aware_counterfactual_summary.json", summary)
    report = [
        "# v88 Phase5 Mode-Aware Counterfactual",
        "",
        f"- scale_label_gate_pass: `{summary['scale_label_gate_pass']}`",
        f"- raw_residual_counterfactual_available: `{summary['raw_residual_counterfactual_available']}`",
        f"- raw_residual_gate_pass: `{summary['raw_residual_gate_pass']}`",
        f"- passing_families: `{summary['passing_families']}`",
        f"- blocker: `{summary.get('blocker', '')}`",
        "",
        "## Families",
        "",
    ]
    for row in rows:
        report.append(
            f"- {row['family']}: bad_median_I_scale={row['bad_median_I_scale']} "
            f"good_max_worsen={row['good_max_scale_error_worsen']} beats_control={row['beats_control_bad_median']} "
            f"scale_gate={row['scale_label_gate_pass']}"
        )
    (args.out_dir / "mode_aware_counterfactual_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"scale_label_gate_pass={summary['scale_label_gate_pass']}")
    print(f"raw_residual_counterfactual_available={summary['raw_residual_counterfactual_available']}")
    print(f"raw_residual_gate_pass={summary['raw_residual_gate_pass']}")
    print(f"passing_families={summary['passing_families']}")
    print(f"best_family={best['family']}")
    print(f"best_bad_median_I_scale={best['bad_median_I_scale']}")
    print(f"best_good_max_scale_error_worsen={best['good_max_scale_error_worsen']}")
    print(f"blocker={summary.get('blocker', '')}")


if __name__ == "__main__":
    main()
