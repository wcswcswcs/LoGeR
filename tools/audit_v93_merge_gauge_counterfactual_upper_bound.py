#!/usr/bin/env python3
"""Audit v93 Phase5 trace-level merge/gauge counterfactual upper bounds."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from tools.v86_soft_latent_utils import safe_float, write_csv, write_json  # noqa: E402
from tools.v93_semantic_object_identity_utils import ROOT, seq_text  # noqa: E402


POSITIVE_STATES = {
    "RESET_RISK",
    "DELAY",
    "REJECT",
    "UPDATE_OBJECT_GAUGE",
    "REJECT_OBJECT_CONFLICT",
    "DELAY_COMMIT",
    "GEOMETRY_RISK",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-dir", type=Path, default=ROOT / "phase4_merge_gauge_carrier_alignment")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase5_merge_gauge_counterfactual_upper_bound")
    return parser.parse_args()


def is_positive(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(POSITIVE_STATES)


def num_series(series: pd.Series) -> pd.Series:
    return series.map(safe_float)


def stable_key(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def median_or_none(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.median())


def mean_or_none(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def same_count_random_mask(df: pd.DataFrame, reference: pd.Series) -> pd.Series:
    out = pd.Series(False, index=df.index)
    group_cols = ["seq", "quality_type"]
    for (seq, quality), group in df.groupby(group_cols, dropna=False):
        count = int(reference.loc[group.index].sum())
        if count <= 0:
            continue
        ordered = sorted(group.index, key=lambda idx: stable_key("v93_cf9_same_count", seq, quality, df.loc[idx, "pair_id"]))
        out.loc[ordered[: min(count, len(ordered))]] = True
    return out


def family_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    cf6 = is_positive(df["p5_combined_object_policy"])
    masks = {
        "CF0_native_trace": pd.Series(False, index=df.index),
        "CF1_geometry_only_hold": is_positive(df["p6_geometry_only_control"]),
        "CF2_object_cross_boundary_reject": is_positive(df["p2_cross_object_reject"]),
        "CF3_same_object_interior_update": ~df["p1_object_interior_update"].astype(str).eq("UPDATE_OBJECT_GAUGE"),
        "CF4_lowobs_hold": df["p3_lowobs_hold"].astype(str).eq("HOLD_GAUGE"),
        "CF5_multimode_delay": is_positive(df["p4_multimode_delay"]),
        "CF6_full_object_policy": cf6,
        "CF7_object_shuffle_control": is_positive(df["p7_object_shuffle_control"]),
        "CF8_component_shuffle_control": is_positive(df["p8_component_shuffle_control"]),
    }
    masks["CF9_same_count_random_control"] = same_count_random_mask(df, masks["CF6_full_object_policy"])
    return masks


def counterfactual_metrics(df: pd.DataFrame, family: str, mask: pd.Series) -> dict[str, Any]:
    labelled = df["base_case_type"].astype(str).isin(["bad", "good"])
    bad = df["base_case_type"].astype(str).eq("bad")
    good = df["base_case_type"].astype(str).eq("good")
    native_delta = num_series(df["merge_residual_delta"])
    native_update = num_series(df["boundary_update_norm"])

    # Conservative trace-level hold/reject/delay model:
    # selected rows cancel the observed native boundary update, so residual_delta
    # and boundary_update_norm become zero. This is an upper-bound proxy, not a
    # rerun trajectory counterfactual.
    cf_delta = native_delta.copy()
    cf_update = native_update.copy()
    cf_delta.loc[mask] = 0.0
    cf_update.loc[mask] = 0.0

    residual_improvement = native_delta - cf_delta
    update_reduction = native_update - cf_update
    residual_worsen = cf_delta - native_delta

    bad_native_med = median_or_none(native_delta[bad])
    bad_cf_med = median_or_none(cf_delta[bad])
    bad_improve_med = median_or_none(residual_improvement[bad])
    good_worsen_med = median_or_none(residual_worsen[good])
    good_worsen_max = float(pd.to_numeric(residual_worsen[good], errors="coerce").dropna().max()) if good.any() else None
    bad_scale = median_or_none(native_delta[bad].abs())
    good_scale = median_or_none(native_delta[good].abs())
    bad_ratio = float(bad_improve_med / max(bad_scale or 0.0, 1e-9)) if finite(bad_improve_med) else None
    good_ratio = float(good_worsen_med / max(good_scale or 0.0, 1e-9)) if finite(good_worsen_med) else None
    good_max_ratio = float(good_worsen_max / max(good_scale or 0.0, 1e-9)) if finite(good_worsen_max) else None

    action_labelled = mask & labelled
    action_bad = mask & bad
    action_good = mask & good
    return {
        "family": family,
        "action_row_count": int(mask.sum()),
        "labelled_action_row_count": int(action_labelled.sum()),
        "bad_rows": int(bad.sum()),
        "good_rows": int(good.sum()),
        "bad_action_rows": int(action_bad.sum()),
        "good_action_rows": int(action_good.sum()),
        "bad_action_recall": float(action_bad.sum() / bad.sum()) if bad.any() else 0.0,
        "good_action_FPR": float(action_good.sum() / good.sum()) if good.any() else 0.0,
        "sequence_coverage": int(df.loc[action_labelled, "seq"].nunique()) if action_labelled.any() else 0,
        "bad_native_merge_residual_delta_median": bad_native_med,
        "bad_counterfactual_merge_residual_delta_median": bad_cf_med,
        "bad_median_residual_improvement": bad_improve_med,
        "bad_median_residual_improvement_ratio": bad_ratio,
        "good_median_residual_worsen": good_worsen_med,
        "good_median_residual_worsen_ratio": good_ratio,
        "good_max_residual_worsen": good_worsen_max,
        "good_max_residual_worsen_ratio": good_max_ratio,
        "bad_mean_boundary_update_norm_reduction": mean_or_none(update_reduction[bad]),
        "good_mean_boundary_update_norm_reduction": mean_or_none(update_reduction[good]),
        "counterfactual_model": "trace_hold_sets_selected_merge_residual_delta_and_boundary_update_norm_to_zero",
        "actual_runtime_trajectory_counterfactual_available": False,
    }


def effect_rows(df: pd.DataFrame, mask: pd.Series) -> list[dict[str, Any]]:
    native_delta = num_series(df["merge_residual_delta"])
    native_update = num_series(df["boundary_update_norm"])
    cf_delta = native_delta.copy()
    cf_update = native_update.copy()
    cf_delta.loc[mask] = 0.0
    cf_update.loc[mask] = 0.0
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        before = safe_float(native_delta.loc[idx])
        after = safe_float(cf_delta.loc[idx])
        update_before = safe_float(native_update.loc[idx])
        update_after = safe_float(cf_update.loc[idx])
        improvement = before - after if before is not None and after is not None else None
        rows.append(
            {
                "pair_id": row.get("pair_id"),
                "seq": row.get("seq"),
                "prev_chunk": row.get("prev_chunk"),
                "curr_chunk": row.get("curr_chunk"),
                "base_case_type": row.get("base_case_type"),
                "quality_type": row.get("quality_type"),
                "cf6_action": bool(mask.loc[idx]),
                "p5_combined_object_policy": row.get("p5_combined_object_policy"),
                "p2_cross_object_reject": row.get("p2_cross_object_reject"),
                "p4_multimode_delay": row.get("p4_multimode_delay"),
                "native_merge_residual_delta": before,
                "cf_merge_residual_delta": after,
                "merge_residual_improvement": improvement,
                "native_boundary_update_norm": update_before,
                "cf_boundary_update_norm": update_after,
                "residual_effect_sign": "improves" if improvement is not None and improvement > 0 else "worsens" if improvement is not None and improvement < 0 else "neutral",
                "trace_path": row.get("merge_state_trace_path"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.phase4_dir / "carrier_alignment_rows.csv"
    df = pd.read_csv(rows_path)
    df["seq"] = df["seq"].map(seq_text)
    masks = family_masks(df)
    rows = [counterfactual_metrics(df, family, mask) for family, mask in masks.items()]
    by_family = {row["family"]: row for row in rows}
    control_families = ["CF1_geometry_only_hold", "CF7_object_shuffle_control", "CF8_component_shuffle_control", "CF9_same_count_random_control"]
    control_best_bad_ratio = max(
        [safe_float(by_family[name].get("bad_median_residual_improvement_ratio")) for name in control_families]
        or [None],
        key=lambda value: -1e9 if value is None else value,
    )
    actual = by_family["CF6_full_object_policy"]
    actual_bad_ratio = safe_float(actual.get("bad_median_residual_improvement_ratio"))
    actual_minus_best_control = (
        float(actual_bad_ratio - control_best_bad_ratio)
        if actual_bad_ratio is not None and control_best_bad_ratio is not None
        else None
    )
    for row in rows:
        bad_ratio = safe_float(row.get("bad_median_residual_improvement_ratio"))
        good_ratio = safe_float(row.get("good_median_residual_worsen_ratio"))
        good_max_ratio = safe_float(row.get("good_max_residual_worsen_ratio"))
        row["bad_residual_improvement_gate"] = bad_ratio is not None and bad_ratio >= 0.10
        row["good_median_protection_gate"] = good_ratio is not None and good_ratio <= 0.02
        row["good_catastrophic_worsen_absent"] = good_max_ratio is not None and good_max_ratio <= 0.20
        row["sequence_coverage_gate"] = int(row.get("sequence_coverage") or 0) >= 3
        row["actual_minus_best_control"] = actual_minus_best_control if row["family"] == "CF6_full_object_policy" else ""
        row["beats_best_control_gate"] = (
            row["family"] == "CF6_full_object_policy"
            and actual_minus_best_control is not None
            and actual_minus_best_control >= 0.05
        )
        row["counterfactual_gate_pass"] = bool(
            row["family"] == "CF6_full_object_policy"
            and row["bad_residual_improvement_gate"]
            and row["good_median_protection_gate"]
            and row["good_catastrophic_worsen_absent"]
            and row["beats_best_control_gate"]
            and row["sequence_coverage_gate"]
        )

    pass_rows = [row for row in rows if row["counterfactual_gate_pass"]]
    cf6_effect_rows = effect_rows(df, masks["CF6_full_object_policy"])
    sign_counts: dict[str, int] = {}
    for row in cf6_effect_rows:
        if not row["cf6_action"]:
            continue
        key = f"{row.get('base_case_type')}::{row.get('residual_effect_sign')}"
        sign_counts[key] = sign_counts.get(key, 0) + 1
    blocker_parts = []
    if not actual.get("bad_residual_improvement_gate"):
        blocker_parts.append("bad_residual_improvement_below_gate")
    if not actual.get("good_median_protection_gate") or not actual.get("good_catastrophic_worsen_absent"):
        blocker_parts.append("good_residual_worsen_exceeds_gate")
    if not actual.get("beats_best_control_gate"):
        blocker_parts.append("does_not_beat_controls")
    if not actual.get("sequence_coverage_gate"):
        blocker_parts.append("sequence_coverage_below_gate")
    summary = {
        "phase": "Phase5_merge_gauge_counterfactual_upper_bound",
        "entered": True,
        "phase5_counterfactual_gate_pass": bool(pass_rows),
        "counterfactual_executed": True,
        "actual_runtime_trajectory_counterfactual_available": False,
        "trace_level_upper_bound_only": True,
        "counterfactual_model": "selected hold/reject/delay rows cancel observed native merge/gauge trace deltas; no trajectory rerun is claimed",
        "actual_family": actual,
        "control_best_bad_median_residual_improvement_ratio": control_best_bad_ratio,
        "actual_minus_best_control": actual_minus_best_control,
        "cf6_action_residual_effect_sign_counts": sign_counts,
        "passing_families": [row["family"] for row in pass_rows],
        "blocker": "" if pass_rows else ";".join(blocker_parts),
        "runtime_action_allowed": bool(pass_rows),
        "ttt_allowed": False,
    }
    write_csv(args.out_dir / "counterfactual_upper_bound_rows.csv", rows)
    write_csv(args.out_dir / "counterfactual_cf6_effect_rows.csv", cf6_effect_rows)
    write_json(args.out_dir / "counterfactual_upper_bound_summary.json", summary)
    print(f"phase5_counterfactual_gate_pass={summary['phase5_counterfactual_gate_pass']}")
    print(f"blocker={summary['blocker']}")
    print(f"actual_bad_median_residual_improvement_ratio={actual.get('bad_median_residual_improvement_ratio')}")
    print(f"good_median_residual_worsen_ratio={actual.get('good_median_residual_worsen_ratio')}")
    print(f"actual_minus_best_control={actual_minus_best_control}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")


if __name__ == "__main__":
    main()
