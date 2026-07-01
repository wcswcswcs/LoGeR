#!/usr/bin/env python3
"""Audit v93 Phase4 merge/gauge carrier alignment from measured traces."""

from __future__ import annotations

import argparse
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

POLICIES = {
    "P0_v92_policy_baseline": "p0_v92_policy_baseline",
    "P5_combined_object_policy": "p5_combined_object_policy",
    "P6_geometry_only_control": "p6_geometry_only_control",
    "P7_object_shuffle_control": "p7_object_shuffle_control",
    "P8_component_shuffle_control": "p8_component_shuffle_control",
    "P9_semantic_label_shuffle_control": "p9_semantic_label_shuffle_control",
    "P10_regime_shuffle_control": "p10_regime_shuffle_control",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=ROOT / "phase2_object_topology_policy")
    parser.add_argument("--phase3-dir", type=Path, default=ROOT / "phase3_merge_gauge_trace_audit")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase4_merge_gauge_carrier_alignment")
    return parser.parse_args()


def is_positive(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(POSITIVE_STATES)


def num_series(series: pd.Series) -> pd.Series:
    return series.map(safe_float)


def auc_binary(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(bool(label), float(score)) for label, score in zip(labels, scores) if score is not None and math.isfinite(float(score))]
    pos = [score for label, score in pairs if label]
    neg = [score for label, score in pairs if not label]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / total) if total else None


def labelled_metric(df: pd.DataFrame, col: str, policy: str) -> dict[str, Any]:
    labelled = df[df["base_case_type"].astype(str).isin(["bad", "good"])].copy()
    positive = is_positive(labelled[col]) if len(labelled) else pd.Series(dtype=bool)
    bad = labelled["base_case_type"].astype(str).eq("bad") if len(labelled) else pd.Series(dtype=bool)
    good = labelled["base_case_type"].astype(str).eq("good") if len(labelled) else pd.Series(dtype=bool)
    bad_recall = float(positive[bad].mean()) if bad.any() else 0.0
    good_fpr = float(positive[good].mean()) if good.any() else 0.0
    return {
        "policy": policy,
        "column": col,
        "labelled_rows": int(len(labelled)),
        "bad_rows": int(bad.sum()) if len(labelled) else 0,
        "good_rows": int(good.sum()) if len(labelled) else 0,
        "triggered_rows": int(positive.sum()) if len(labelled) else 0,
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": float((bad_recall + (1.0 - good_fpr)) / 2.0),
        "sequence_coverage": int(labelled["seq"].nunique()) if len(labelled) else 0,
    }


def trace_group_stats(df: pd.DataFrame, col: str, policy: str) -> dict[str, Any]:
    positive = is_positive(df[col])
    update = num_series(df["boundary_update_norm"])
    residual = num_series(df["merge_residual_delta"])
    non_null_update = update.dropna()
    non_null_residual = residual.dropna()
    risk_update = update[positive].dropna()
    safe_update = update[~positive].dropna()
    risk_residual = residual[positive].dropna()
    safe_residual = residual[~positive].dropna()
    labelled = df["base_case_type"].astype(str).isin(["bad", "good"])
    bad = df["base_case_type"].astype(str).eq("bad")
    policy_pos = list(positive[labelled])
    boundary_scores = [safe_float(v) for v in update[labelled]]
    residual_scores = [safe_float(v) for v in residual[labelled]]
    bad_labels = list(bad[labelled])
    return {
        "policy": policy,
        "rows": int(len(df)),
        "positive_rows": int(positive.sum()),
        "safe_rows": int((~positive).sum()),
        "boundary_update_norm_available_ratio": float(update.notna().mean()) if len(df) else 0.0,
        "merge_residual_delta_available_ratio": float(residual.notna().mean()) if len(df) else 0.0,
        "boundary_update_norm_mean": float(non_null_update.mean()) if len(non_null_update) else None,
        "merge_residual_delta_mean": float(non_null_residual.mean()) if len(non_null_residual) else None,
        "risk_state_update_norm_mean": float(risk_update.mean()) if len(risk_update) else None,
        "safe_state_update_norm_mean": float(safe_update.mean()) if len(safe_update) else None,
        "risk_state_residual_delta_mean": float(risk_residual.mean()) if len(risk_residual) else None,
        "safe_state_residual_delta_mean": float(safe_residual.mean()) if len(safe_residual) else None,
        "policy_state_vs_boundary_update_norm_auc": auc_binary(policy_pos, boundary_scores),
        "policy_state_vs_merge_residual_delta_auc": auc_binary(policy_pos, residual_scores),
        "bad_label_vs_boundary_update_norm_auc": auc_binary(bad_labels, boundary_scores),
        "bad_label_vs_merge_residual_delta_auc": auc_binary(bad_labels, residual_scores),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = pd.read_csv(args.phase2_dir / "object_topology_policy_rows.csv")
    trace = pd.read_csv(args.phase3_dir / "merge_gauge_trace_ledger.csv")
    policy["seq"] = policy["seq"].map(seq_text)
    trace["seq"] = trace["seq"].map(seq_text)
    df = policy.merge(
        trace[
            [
                "pair_id",
                "merge_state_trace_path",
                "trace_source",
                "boundary_update_norm",
                "boundary_update_norm_source",
                "merge_residual_delta",
                "non_identity_transform_flag",
                "non_identity_transform_flag_source",
                "trace_provenance",
            ]
        ],
        on="pair_id",
        how="left",
    )
    policy_metrics = [labelled_metric(df, col, name) for name, col in POLICIES.items()]
    trace_stats = [trace_group_stats(df, col, name) for name, col in POLICIES.items()]
    by_name = {row["policy"]: row for row in policy_metrics}
    actual = by_name["P5_combined_object_policy"]
    geometry = by_name["P6_geometry_only_control"]
    margins = {
        "actual_vs_object_shuffle_margin": actual["balanced_accuracy"] - by_name["P7_object_shuffle_control"]["balanced_accuracy"],
        "actual_vs_component_shuffle_margin": actual["balanced_accuracy"] - by_name["P8_component_shuffle_control"]["balanced_accuracy"],
        "actual_vs_semantic_shuffle_margin": actual["balanced_accuracy"] - by_name["P9_semantic_label_shuffle_control"]["balanced_accuracy"],
        "actual_vs_regime_shuffle_margin": actual["balanced_accuracy"] - by_name["P10_regime_shuffle_control"]["balanced_accuracy"],
        "actual_vs_geometry_only_margin": actual["balanced_accuracy"] - geometry["balanced_accuracy"],
        "good_protection_margin_vs_geometry": geometry["good_FPR"] - actual["good_FPR"],
    }
    trace_true = bool(
        df["boundary_update_norm_source"].astype(str).eq("direct").mean() >= 0.80
        and df["merge_residual_delta"].map(lambda value: safe_float(value) is not None).mean() >= 0.60
        and df["trace_provenance"].astype(str).str.len().mean() >= 0.95
    )
    gate_pass = bool(
        actual["bad_recall"] >= 0.60
        and actual["good_FPR"] <= 0.25
        and actual["sequence_coverage"] >= 3
        and min(
            margins["actual_vs_object_shuffle_margin"],
            margins["actual_vs_component_shuffle_margin"],
            margins["actual_vs_semantic_shuffle_margin"],
            margins["actual_vs_regime_shuffle_margin"],
        )
        >= 0.05
        and (margins["actual_vs_geometry_only_margin"] >= 0.05 or margins["good_protection_margin_vs_geometry"] >= 0.10)
        and trace_true
    )
    blockers = []
    if actual["good_FPR"] > 0.25:
        blockers.append("carrier_good_FPR_too_high")
    if min(
        margins["actual_vs_object_shuffle_margin"],
        margins["actual_vs_component_shuffle_margin"],
        margins["actual_vs_semantic_shuffle_margin"],
        margins["actual_vs_regime_shuffle_margin"],
    ) < 0.05:
        blockers.append("carrier_shuffle_specificity_fail")
    if not trace_true:
        blockers.append("trace_fields_not_true_or_incomplete")
    if actual["bad_recall"] < 0.60:
        blockers.append("carrier_bad_recall_low")
    summary = {
        "phase": "Phase4_merge_gauge_carrier_alignment",
        "entered": True,
        "phase4_carrier_alignment_gate_pass": gate_pass,
        "blocker": "" if gate_pass else ";".join(blockers),
        "actual_policy_metrics": {**actual, **margins},
        "trace_true_fields_pass": trace_true,
        "trace_group_stats_actual": next(row for row in trace_stats if row["policy"] == "P5_combined_object_policy"),
        "control_policy_metrics": [
            by_name["P7_object_shuffle_control"],
            by_name["P8_component_shuffle_control"],
            by_name["P9_semantic_label_shuffle_control"],
            by_name["P10_regime_shuffle_control"],
            by_name["P6_geometry_only_control"],
        ],
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }
    write_csv(args.out_dir / "carrier_alignment_rows.csv", df.to_dict("records"))
    write_csv(args.out_dir / "carrier_alignment_policy_metrics.csv", policy_metrics)
    write_csv(args.out_dir / "carrier_alignment_trace_group_stats.csv", trace_stats)
    write_json(args.out_dir / "carrier_alignment_summary.json", summary)
    print(f"phase4_carrier_alignment_gate_pass={gate_pass}")
    print(f"blocker={summary['blocker']}")
    print(f"actual_bad_recall={actual['bad_recall']}")
    print(f"actual_good_FPR={actual['good_FPR']}")
    print(f"trace_true_fields_pass={trace_true}")


if __name__ == "__main__":
    main()
