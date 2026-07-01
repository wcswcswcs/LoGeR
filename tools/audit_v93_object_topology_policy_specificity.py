#!/usr/bin/env python3
"""Audit v93 Phase2 fixed object-topology policy specificity gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from tools.v86_soft_latent_utils import write_csv, write_json  # noqa: E402
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

POLICIES = {
    "P0_v92_policy_baseline": "p0_v92_policy_baseline",
    "P1_object_interior_update": "p1_object_interior_update",
    "P2_cross_object_reject": "p2_cross_object_reject",
    "P3_lowobs_hold": "p3_lowobs_hold",
    "P4_multimode_delay": "p4_multimode_delay",
    "P5_combined_object_policy": "p5_combined_object_policy",
    "P6_geometry_only_control": "p6_geometry_only_control",
    "P7_object_shuffle_control": "p7_object_shuffle_control",
    "P8_component_shuffle_control": "p8_component_shuffle_control",
    "P9_semantic_label_shuffle_control": "p9_semantic_label_shuffle_control",
    "P10_regime_shuffle_control": "p10_regime_shuffle_control",
    "P5_repair_require_match_or_radio": "p5_repair_require_match_or_radio",
    "P5_repair_multimode_delay": "p5_repair_multimode_delay",
    "P5_repair_radio_guarded": "p5_repair_radio_guarded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=ROOT / "phase2_object_topology_policy")
    return parser.parse_args()


def is_positive(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(POSITIVE_STATES)


def metric(df: pd.DataFrame, col: str, label: str) -> dict[str, Any]:
    labelled = df[df["base_case_type"].astype(str).isin(["bad", "good"])].copy()
    if labelled.empty:
        return {"policy": label, "available_rows": 0}
    positive = is_positive(labelled[col])
    bad = labelled["base_case_type"].astype(str).eq("bad")
    good = labelled["base_case_type"].astype(str).eq("good")
    bad_recall = float(positive[bad].mean()) if bad.any() else 0.0
    good_fpr = float(positive[good].mean()) if good.any() else 0.0
    ba = float((bad_recall + (1.0 - good_fpr)) / 2.0)
    triggered = labelled[positive]
    return {
        "policy": label,
        "column": col,
        "available_rows": int(len(labelled)),
        "bad_rows": int(bad.sum()),
        "good_rows": int(good.sum()),
        "triggered_rows": int(positive.sum()),
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": ba,
        "sequence_coverage": int(labelled["seq"].nunique()),
        "triggered_seq_coverage": int(triggered["seq"].nunique()) if len(triggered) else 0,
        "object_identity_coverage_on_triggered_rows": float(triggered["has_object_identity"].astype(str).str.lower().isin(["true", "1"]).mean())
        if len(triggered)
        else 0.0,
        "radio_coverage_on_triggered_rows": float(triggered["radio_available"].astype(str).str.lower().isin(["true", "1"]).mean())
        if len(triggered)
        else 0.0,
    }


def loso_positive_folds(df: pd.DataFrame, actual_col: str, control_cols: list[str]) -> int:
    total = 0
    for seq, fold in df[df["base_case_type"].astype(str).isin(["bad", "good"])].groupby("seq"):
        if fold["base_case_type"].nunique() < 2:
            continue
        actual = metric(fold, actual_col, "actual")["balanced_accuracy"]
        controls = [metric(fold, col, col)["balanced_accuracy"] for col in control_cols]
        if actual - max(controls) >= 0.05:
            total += 1
    return total


def fp_categories(row: pd.Series) -> str:
    cats: list[str] = []
    if str(row.get("has_object_identity")).lower() in {"true", "1"} and float(row.get("boundary_global_cross_ratio") or 0.0) >= 0.34:
        cats.append("exact_global_id_cross_boundary")
    if str(row.get("has_object_identity")).lower() in {"true", "1"} and float(row.get("object_identity_confidence") or 0.0) < 0.35:
        cats.append("object_id_low_confidence")
    if float(row.get("same_object_ratio") or 0.0) >= 0.80 and float(row.get("temporal_stability") or 0.0) >= 0.20:
        cats.append("stable_same_object_boundary")
    if float(row.get("same_object_ratio") or 0.0) >= 0.80 and str(row.get("has_object_identity")).lower() not in {"true", "1"}:
        cats.append("high_same_label_or_component_ratio_but_no_object_id")
    if "component_proxy" in str(row.get("source_scope")) and str(row.get("has_object_identity")).lower() not in {"true", "1"}:
        cats.append("compact_proxy_artifact")
    if float(row.get("object_boundary_ratio") or 0.0) >= 0.90:
        cats.append("high_boundary_without_true_residual_conflict_trace")
    if not cats:
        cats.append("other_false_positive")
    return ";".join(cats)


def fn_categories(row: pd.Series) -> str:
    cats: list[str] = []
    if str(row.get("has_object_identity")).lower() not in {"true", "1"}:
        cats.append("no_object_id")
    if str(row.get("has_object_identity")).lower() in {"true", "1"} and float(row.get("object_identity_confidence") or 0.0) < 0.35:
        cats.append("object_id_low_confidence")
    try:
        match_conf = float(row.get("match_support_confidence") or 0.0)
    except ValueError:
        match_conf = 0.0
    if match_conf < 0.50:
        cats.append("low_tracklet_confidence")
    if float(row.get("cross_object_ratio") or 0.0) >= 0.90 and float(row.get("object_boundary_ratio") or 0.0) >= 0.80:
        cats.append("cross_object_boundary_ambiguous")
    if float(row.get("temporal_stability") or 0.0) < 0.20:
        cats.append("lowobs_context")
    if str(row.get("seq")) == "01":
        cats.append("seq01_stress")
    if not cats:
        cats.append("other_false_negative")
    return ";".join(cats)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.phase2_dir / "object_topology_policy_rows.csv")
    df["seq"] = df["seq"].map(seq_text)
    metrics = [metric(df, col, name) for name, col in POLICIES.items()]
    by_name = {row["policy"]: row for row in metrics}
    baseline = by_name["P0_v92_policy_baseline"]
    actual = by_name["P5_combined_object_policy"]
    geometry = by_name["P6_geometry_only_control"]
    controls = [
        by_name["P7_object_shuffle_control"],
        by_name["P8_component_shuffle_control"],
        by_name["P9_semantic_label_shuffle_control"],
        by_name["P10_regime_shuffle_control"],
    ]
    margins = {
        "object_shuffle_margin": actual["balanced_accuracy"] - by_name["P7_object_shuffle_control"]["balanced_accuracy"],
        "component_shuffle_margin": actual["balanced_accuracy"] - by_name["P8_component_shuffle_control"]["balanced_accuracy"],
        "semantic_shuffle_margin": actual["balanced_accuracy"] - by_name["P9_semantic_label_shuffle_control"]["balanced_accuracy"],
        "regime_shuffle_margin": actual["balanced_accuracy"] - by_name["P10_regime_shuffle_control"]["balanced_accuracy"],
        "semantic_good_protection_margin_vs_geometry": geometry["good_FPR"] - actual["good_FPR"],
        "semantic_good_protection_margin_vs_v92": baseline["good_FPR"] - actual["good_FPR"],
    }
    loso = loso_positive_folds(
        df,
        "p5_combined_object_policy",
        [
            "p7_object_shuffle_control",
            "p8_component_shuffle_control",
            "p9_semantic_label_shuffle_control",
            "p10_regime_shuffle_control",
        ],
    )
    actual_with_deltas = {
        **actual,
        "bad_recall_delta_vs_v92": actual["bad_recall"] - baseline["bad_recall"],
        "good_FPR_delta_vs_v92": actual["good_FPR"] - baseline["good_FPR"],
        "bad_recall_delta_vs_geometry": actual["bad_recall"] - geometry["bad_recall"],
        "good_FPR_delta_vs_geometry": actual["good_FPR"] - geometry["good_FPR"],
        **margins,
        "LOSO_positive_folds": loso,
    }
    primary_gate_pass = (
        actual["bad_recall"] >= 0.60
        and actual["good_FPR"] <= 0.25
        and actual["sequence_coverage"] >= 3
        and min(
            margins["object_shuffle_margin"],
            margins["component_shuffle_margin"],
            margins["semantic_shuffle_margin"],
            margins["regime_shuffle_margin"],
        )
        >= 0.05
        and loso >= 3
    )
    good_protection_pass = (
        max(margins["semantic_good_protection_margin_vs_geometry"], margins["semantic_good_protection_margin_vs_v92"]) >= 0.10
        and (baseline["bad_recall"] - actual["bad_recall"]) <= 0.05
        and min(
            margins["object_shuffle_margin"],
            margins["component_shuffle_margin"],
            margins["semantic_shuffle_margin"],
            margins["regime_shuffle_margin"],
        )
        >= 0.05
    )

    false_neg = df[
        df["base_case_type"].astype(str).eq("bad") & (~is_positive(df["p5_combined_object_policy"]))
    ].copy()
    false_pos = df[
        df["base_case_type"].astype(str).eq("good") & is_positive(df["p5_combined_object_policy"])
    ].copy()
    false_neg["false_negative_categories"] = false_neg.apply(fn_categories, axis=1) if len(false_neg) else []
    false_pos["false_positive_categories"] = false_pos.apply(fp_categories, axis=1) if len(false_pos) else []
    fn_rows = (
        false_neg.groupby("source_scope", dropna=False)
        .size()
        .reset_index(name="false_negative_count")
        .sort_values("false_negative_count", ascending=False)
        .to_dict("records")
    )
    fp_rows = (
        false_pos.groupby("source_scope", dropna=False)
        .size()
        .reset_index(name="false_positive_count")
        .sort_values("false_positive_count", ascending=False)
        .to_dict("records")
    )
    fp_category_rows = (
        false_pos.assign(false_positive_category=false_pos["false_positive_categories"].str.split(";"))
        .explode("false_positive_category")
        .groupby("false_positive_category", dropna=False)
        .size()
        .reset_index(name="false_positive_count")
        .sort_values("false_positive_count", ascending=False)
        .to_dict("records")
        if len(false_pos)
        else []
    )
    fn_category_rows = (
        false_neg.assign(false_negative_category=false_neg["false_negative_categories"].str.split(";"))
        .explode("false_negative_category")
        .groupby("false_negative_category", dropna=False)
        .size()
        .reset_index(name="false_negative_count")
        .sort_values("false_negative_count", ascending=False)
        .to_dict("records")
        if len(false_neg)
        else []
    )

    repair_names = [
        "P5_repair_require_match_or_radio",
        "P5_repair_multimode_delay",
        "P5_repair_radio_guarded",
    ]
    repair_rows = []
    for name in repair_names:
        row = by_name[name]
        repair_margins = {
            "object_shuffle_margin": row["balanced_accuracy"] - by_name["P7_object_shuffle_control"]["balanced_accuracy"],
            "component_shuffle_margin": row["balanced_accuracy"] - by_name["P8_component_shuffle_control"]["balanced_accuracy"],
            "semantic_shuffle_margin": row["balanced_accuracy"] - by_name["P9_semantic_label_shuffle_control"]["balanced_accuracy"],
            "regime_shuffle_margin": row["balanced_accuracy"] - by_name["P10_regime_shuffle_control"]["balanced_accuracy"],
        }
        repair_rows.append(
            {
                **row,
                "repair_rule": name,
                **repair_margins,
                "bad_recall_delta_vs_v92": row["bad_recall"] - baseline["bad_recall"],
                "good_FPR_delta_vs_v92": row["good_FPR"] - baseline["good_FPR"],
                "repair_gate_pass": row["bad_recall"] >= 0.60
                and row["good_FPR"] <= 0.25
                and row["sequence_coverage"] >= 3
                and min(repair_margins.values()) >= 0.05,
            }
        )
    any_repair_pass = any(row["repair_gate_pass"] for row in repair_rows)
    gate_pass = bool(primary_gate_pass or good_protection_pass or any_repair_pass)
    labelled_mask = df["base_case_type"].astype(str).isin(["bad", "good"])
    has_object_mask = df["has_object_identity"].astype(str).str.lower().isin(["true", "1"])
    object_identity_labelled_coverage = float(has_object_mask[labelled_mask].mean()) if labelled_mask.any() else 0.0
    blocker_parts = []
    if actual["bad_recall"] < 0.60:
        blocker_parts.append("low_bad_recall")
    if actual["good_FPR"] > 0.25:
        blocker_parts.append("high_good_FPR")
    if min(margins["object_shuffle_margin"], margins["component_shuffle_margin"], margins["semantic_shuffle_margin"], margins["regime_shuffle_margin"]) < 0.05:
        blocker_parts.append("shuffle_specificity_fail")
    if actual["object_identity_coverage_on_triggered_rows"] == 0.0:
        blocker_parts.append("object_identity_absent_on_triggered_rows")
    if not blocker_parts:
        blocker_parts.append("policy_gate_failed")
    summary = {
        "phase": "Phase2_object_topology_policy_specificity",
        "phase2_object_topology_policy_gate_pass": gate_pass,
        "primary_gate_pass": primary_gate_pass,
        "good_protection_pass": good_protection_pass,
        "repair_gate_pass": any_repair_pass,
        "blocker": "" if gate_pass else ";".join(blocker_parts),
        "allowed_next_scope": "carrier_alignment_if_phase3_trace_pass" if gate_pass else "phase3_trace_instrumentation_or_source_expansion_no_policy_promotion",
        "actual_policy": actual_with_deltas,
        "baseline_policy": baseline,
        "geometry_control": geometry,
        "control_policies": controls,
        "repair_attempts": repair_rows,
        "false_negative_by_scope": fn_rows,
        "false_positive_by_scope": fp_rows,
        "false_negative_by_category": fn_category_rows,
        "false_positive_by_category": fp_category_rows,
        "object_identity_labelled_coverage": object_identity_labelled_coverage,
        "no_object_identity_success_claim": object_identity_labelled_coverage < 0.50,
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
        "analysis": "Phase2 uses exact SAM31 boundary global-id evidence where available. Repairs are limited to stable same-object boundary guarding, object-confidence guarded multimode DELAY, and match/RADIO guarded evidence; no global threshold sweep is used.",
    }
    write_csv(args.phase2_dir / "object_topology_policy_metrics.csv", metrics)
    write_csv(args.phase2_dir / "object_topology_policy_repair_metrics.csv", repair_rows)
    write_csv(args.phase2_dir / "object_topology_false_negative_by_scope.csv", fn_rows)
    write_csv(args.phase2_dir / "object_topology_false_positive_by_scope.csv", fp_rows)
    write_csv(args.phase2_dir / "object_topology_false_negative_by_category.csv", fn_category_rows)
    write_csv(args.phase2_dir / "object_topology_false_positive_by_category.csv", fp_category_rows)
    write_csv(
        args.phase2_dir / "object_topology_false_negative_rows.csv",
        false_neg[
            [
                "pair_id",
                "seq",
                "prev_chunk",
                "curr_chunk",
                "source_scope",
                "false_negative_categories",
                "p5_combined_object_policy",
                "has_object_identity",
                "object_identity_confidence",
                "same_object_ratio",
                "cross_object_ratio",
                "boundary_global_cross_ratio",
                "boundary_new_id_ratio",
                "boundary_valid_global_rows",
                "object_boundary_ratio",
                "temporal_stability",
                "match_support_confidence",
            ]
        ].to_dict("records")
        if len(false_neg)
        else [],
    )
    write_csv(
        args.phase2_dir / "object_topology_false_positive_rows.csv",
        false_pos[
            [
                "pair_id",
                "seq",
                "prev_chunk",
                "curr_chunk",
                "source_scope",
                "false_positive_categories",
                "p5_combined_object_policy",
                "has_object_identity",
                "object_identity_confidence",
                "same_object_ratio",
                "cross_object_ratio",
                "boundary_global_cross_ratio",
                "boundary_new_id_ratio",
                "boundary_valid_global_rows",
                "object_boundary_ratio",
                "temporal_stability",
                "match_support_confidence",
            ]
        ].to_dict("records")
        if len(false_pos)
        else [],
    )
    write_json(args.phase2_dir / "object_topology_policy_audit.json", summary)
    print(f"phase2_object_topology_policy_gate_pass={gate_pass}")
    print(f"primary_gate_pass={primary_gate_pass}")
    print(f"good_protection_pass={good_protection_pass}")
    print(f"repair_gate_pass={any_repair_pass}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
