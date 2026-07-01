#!/usr/bin/env python3
"""Audit whether v92 Phase7 source expansion improves semantic policy gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v86_soft_latent_utils import read_json, write_csv, write_json  # noqa: E402
from tools.v92_semantic_policy_carrier_utils import ROOT, RISK_STATES  # noqa: E402


DEFAULT_OUT = ROOT / "phase7_data_source_expansion"
DEFAULT_PHASE1 = ROOT / "phase1_semantic_policy_row_bank/phase1_gate_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-candidate-rows", type=Path, default=DEFAULT_OUT / "semantic_source_expansion_candidate_rows.csv")
    parser.add_argument("--phase1-summary", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.lower().isin({"true", "1", "yes"})


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _risk_state(series: pd.Series) -> pd.Series:
    return series.astype(str).isin(RISK_STATES)


def _metrics(df: pd.DataFrame, state_col: str) -> dict[str, Any]:
    labelled = df[df["base_case_type"].astype(str).isin(["bad", "good"])].copy()
    if labelled.empty:
        return {
            "labelled_rows": 0,
            "bad_rows": 0,
            "good_rows": 0,
            "bad_recall": None,
            "good_FPR": None,
            "balanced_accuracy": None,
            "positive_count": int(_risk_state(df[state_col]).sum()),
        }
    bad = labelled["base_case_type"].astype(str).eq("bad")
    good = labelled["base_case_type"].astype(str).eq("good")
    pred = _risk_state(labelled[state_col])
    bad_recall = float((pred & bad).sum() / max(1, int(bad.sum())))
    good_fpr = float((pred & good).sum() / max(1, int(good.sum())))
    return {
        "labelled_rows": int(len(labelled)),
        "bad_rows": int(bad.sum()),
        "good_rows": int(good.sum()),
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": float(0.5 * (bad_recall + (1.0 - good_fpr))),
        "positive_count": int(_risk_state(df[state_col]).sum()),
    }


def _component_fallback_state(
    df: pd.DataFrame,
    base_col: str,
    *,
    strategy: str,
    rotate_component_proxy: bool = False,
) -> pd.Series:
    base = df[base_col].astype(str).copy()
    if strategy == "baseline":
        return base
    invalid = _num(df, "cross_object_boundary_ratio")
    boundary = _num(df, "object_boundary_ratio")
    interior = _num(df, "object_interior_ratio")
    temporal = _num(df, "temporal_stability")
    same_object = _num(df, "same_object_ratio")
    component_available = _bool_series(df, "component_tracklet_available")

    high_risk = component_available & (invalid > 0.50) & (same_object < 0.90)
    stable_same = component_available & (same_object >= 0.90) & (temporal >= 0.25)
    delay_low_stability = component_available & (~high_risk) & (~stable_same) & (temporal < 0.08) & (boundary >= 0.90)

    if rotate_component_proxy and len(df):
        rotated_high = pd.Series(False, index=df.index)
        rotated_stable = pd.Series(False, index=df.index)
        rotated_delay = pd.Series(False, index=df.index)
        for _, group in df.groupby("seq"):
            order = list(group.sort_values(["prev_chunk", "curr_chunk"]).index)
            if not order:
                continue
            target = order[1:] + order[:1]
            rotated_high.loc[target] = high_risk.loc[order].to_numpy()
            rotated_stable.loc[target] = stable_same.loc[order].to_numpy()
            rotated_delay.loc[target] = delay_low_stability.loc[order].to_numpy()
        high_risk = rotated_high
        stable_same = rotated_stable
        delay_low_stability = rotated_delay

    out = base.copy()
    if strategy in {"continuity_protection", "combined_continuity_invalid"}:
        out.loc[stable_same & _risk_state(out)] = "HOLD"
    if strategy in {"invalid_majority", "combined_continuity_invalid"}:
        out.loc[high_risk & (~stable_same)] = "RESET_RISK"
    if strategy == "combined_continuity_invalid":
        out.loc[delay_low_stability & (~_risk_state(out))] = "DELAY"
    return out


def _control_rows(df: pd.DataFrame, strategy: str) -> list[dict[str, Any]]:
    rows = []
    for label, col in [
        (f"{strategy}_actual", f"{strategy}_actual_state"),
        (f"{strategy}_semantic_shuffle", f"{strategy}_semantic_shuffle_state"),
        (f"{strategy}_component_shuffle", f"{strategy}_component_shuffle_state"),
    ]:
        metrics = _metrics(df, col)
        rows.append({"strategy": strategy, "policy": label, "state_col": col, **metrics})
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phase1 = read_json(args.phase1_summary) if args.phase1_summary.exists() else {}
    df = pd.read_csv(args.source_candidate_rows)
    strategies = [
        "baseline",
        "continuity_protection",
        "invalid_majority",
        "combined_continuity_invalid",
    ]
    all_control_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    for strategy in strategies:
        actual_col = f"{strategy}_actual_state"
        sem_col = f"{strategy}_semantic_shuffle_state"
        comp_col = f"{strategy}_component_shuffle_state"
        df[actual_col] = _component_fallback_state(df, "policy_state", strategy=strategy, rotate_component_proxy=False)
        df[sem_col] = _component_fallback_state(df, "semantic_shuffle_state", strategy=strategy, rotate_component_proxy=False)
        df[comp_col] = _component_fallback_state(df, "component_shuffle_state", strategy=strategy, rotate_component_proxy=True)
        rows = _control_rows(df, strategy)
        all_control_rows.extend(rows)
        by_policy = {row["policy"]: row for row in rows}
        actual = by_policy[f"{strategy}_actual"]
        sem = by_policy[f"{strategy}_semantic_shuffle"]
        comp = by_policy[f"{strategy}_component_shuffle"]
        baseline_metrics = _metrics(df, "baseline_actual_state")
        bad_improve = float((actual["bad_recall"] or 0.0) - (baseline_metrics["bad_recall"] or 0.0))
        fpr_improve = float((baseline_metrics["good_FPR"] or 0.0) - (actual["good_FPR"] or 0.0))
        fpr_worse = float((actual["good_FPR"] or 0.0) - (baseline_metrics["good_FPR"] or 0.0))
        bad_worse = float((baseline_metrics["bad_recall"] or 0.0) - (actual["bad_recall"] or 0.0))
        sem_margin = float((actual["balanced_accuracy"] or 0.0) - (sem["balanced_accuracy"] or 0.0))
        comp_margin = float((actual["balanced_accuracy"] or 0.0) - (comp["balanced_accuracy"] or 0.0))
        strategy_rows.append(
            {
                "strategy": strategy,
                "bad_recall": actual["bad_recall"],
                "good_FPR": actual["good_FPR"],
                "balanced_accuracy": actual["balanced_accuracy"],
                "positive_count": actual["positive_count"],
                "changed_policy_rows": int((df[actual_col].astype(str) != df["policy_state"].astype(str)).sum()),
                "bad_recall_improvement_vs_phase7_baseline": bad_improve,
                "good_FPR_improvement_vs_phase7_baseline": fpr_improve,
                "good_FPR_worsening_vs_phase7_baseline": fpr_worse,
                "bad_recall_worsening_vs_phase7_baseline": bad_worse,
                "semantic_shuffle_margin": sem_margin,
                "component_shuffle_margin": comp_margin,
            }
        )

    best = sorted(
        strategy_rows,
        key=lambda row: (float(row.get("balanced_accuracy") or -1.0), float(row.get("good_FPR_improvement_vs_phase7_baseline") or 0.0)),
        reverse=True,
    )[0]
    best_strategy = str(best["strategy"])
    df["expanded_policy_state"] = df[f"{best_strategy}_actual_state"]
    df["expanded_semantic_shuffle_state"] = df[f"{best_strategy}_semantic_shuffle_state"]
    df["expanded_component_shuffle_state"] = df[f"{best_strategy}_component_shuffle_state"]
    df["component_fallback_changed_policy"] = df["expanded_policy_state"].astype(str) != df["policy_state"].astype(str)
    df["component_fallback_reason"] = ""
    df.loc[df["expanded_policy_state"].astype(str).eq("RESET_RISK") & ~df["policy_state"].astype(str).eq("RESET_RISK"), "component_fallback_reason"] = "component_proxy_invalid_majority_low_same_object"
    df.loc[df["expanded_policy_state"].astype(str).eq("HOLD") & _risk_state(df["policy_state"]), "component_fallback_reason"] = "component_proxy_stable_same_object_hold"
    df.loc[df["expanded_policy_state"].astype(str).eq("DELAY") & ~_risk_state(df["policy_state"]), "component_fallback_reason"] = "component_proxy_low_temporal_stability_delay"

    phase1_metrics = _metrics(df, "policy_state")
    expanded = _metrics(df, "expanded_policy_state")

    labelled = df[df["base_case_type"].astype(str).isin(["bad", "good"])]
    coverage = float(_bool_series(labelled, "component_tracklet_available").sum() / max(1, len(labelled))) if len(labelled) else 0.0
    object_identity_coverage = float(_bool_series(labelled, "object_identity_available").sum() / max(1, len(labelled))) if len(labelled) else 0.0

    bad_improvement = float((expanded["bad_recall"] or 0.0) - (phase1_metrics["bad_recall"] or 0.0))
    fpr_improvement = float((phase1_metrics["good_FPR"] or 0.0) - (expanded["good_FPR"] or 0.0))
    fpr_worsening = float((expanded["good_FPR"] or 0.0) - (phase1_metrics["good_FPR"] or 0.0))
    bad_worsening = float((phase1_metrics["bad_recall"] or 0.0) - (expanded["bad_recall"] or 0.0))
    expanded_semantic_shuffle_margin = float(best.get("semantic_shuffle_margin") or 0.0)
    expanded_component_shuffle_margin = float(best.get("component_shuffle_margin") or 0.0)
    phase1_semantic_margin = float(phase1.get("semantic_shuffle_margin", 0.0) or 0.0)
    phase1_component_margin = float(phase1.get("component_shuffle_margin", 0.0) or 0.0)
    margin_improve = min(
        expanded_semantic_shuffle_margin - phase1_semantic_margin,
        expanded_component_shuffle_margin - phase1_component_margin,
    )

    useful = bool(
        coverage >= 0.50
        and (
            (bad_improvement >= 0.10 and fpr_worsening <= 0.05)
            or (fpr_improvement >= 0.10 and bad_worsening <= 0.05)
            or margin_improve >= 0.05
        )
    )
    summary = {
        "phase": "Phase7_expanded_semantic_policy_audit",
        "phase7_data_source_expansion_useful": useful,
        "phase7_expanded_policy_gate_pass": useful,
        "best_fixed_fallback_strategy": best_strategy,
        "fixed_fallback_strategy_count": int(len(strategy_rows)),
        "coverage_on_labelled_rows": coverage,
        "object_identity_labelled_coverage": object_identity_coverage,
        "source_scope": "component_tracklet_proxy",
        "expanded_policy_bad_recall": expanded["bad_recall"],
        "expanded_policy_good_FPR": expanded["good_FPR"],
        "expanded_balanced_accuracy": expanded["balanced_accuracy"],
        "phase1_bad_recall_recomputed": phase1_metrics["bad_recall"],
        "phase1_good_FPR_recomputed": phase1_metrics["good_FPR"],
        "phase1_bad_recall_summary": phase1.get("bad_recall"),
        "phase1_good_FPR_summary": phase1.get("good_FPR"),
        "phase1_metric_scope_note": "Phase7 recomputed metrics use the 24 labelled rows present in semantic_source_expansion_candidate_rows.csv; Phase1 summary values keep the original v91/v92 row-bank audit scope.",
        "bad_recall_improvement_vs_phase1": bad_improvement,
        "good_FPR_improvement_vs_phase1": fpr_improvement,
        "good_FPR_worsening_vs_phase1": fpr_worsening,
        "bad_recall_worsening_vs_phase1": bad_worsening,
        "expanded_semantic_shuffle_margin": expanded_semantic_shuffle_margin,
        "expanded_component_shuffle_margin": expanded_component_shuffle_margin,
        "phase1_semantic_shuffle_margin": phase1_semantic_margin,
        "phase1_component_shuffle_margin": phase1_component_margin,
        "min_shuffle_margin_improvement_vs_phase1": margin_improve,
        "changed_policy_rows": int(df["component_fallback_changed_policy"].sum()),
        "object_identity_available_ratio": float(_bool_series(df, "object_identity_available").mean()) if len(df) else 0.0,
        "component_tracklet_available_ratio": float(_bool_series(df, "component_tracklet_available").mean()) if len(df) else 0.0,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "gate_rule": "coverage>=0.50 and bad_recall/good_FPR/margin improvement per plan Phase7",
    }
    if not useful:
        summary["blocker"] = (
            "semantic_source_specificity_insufficient"
            if object_identity_coverage < 0.50
            else "component_tracklet_fallback_does_not_improve_policy_gate"
        )
    write_csv(args.out_dir / "expanded_semantic_policy_rows.csv", df.to_dict("records"))
    write_csv(args.out_dir / "expanded_semantic_policy_control_metrics.csv", all_control_rows)
    write_csv(args.out_dir / "expanded_semantic_policy_strategy_metrics.csv", strategy_rows)
    write_json(args.out_dir / "expanded_semantic_policy_summary.json", summary)
    print(f"phase7_expanded_policy_gate_pass={summary['phase7_expanded_policy_gate_pass']}")
    print(f"coverage_on_labelled_rows={summary['coverage_on_labelled_rows']}")
    print(f"object_identity_labelled_coverage={summary['object_identity_labelled_coverage']}")
    print(f"expanded_policy_bad_recall={summary['expanded_policy_bad_recall']}")
    print(f"expanded_policy_good_FPR={summary['expanded_policy_good_FPR']}")
    print(f"bad_recall_improvement_vs_phase1={summary['bad_recall_improvement_vs_phase1']}")
    print(f"good_FPR_improvement_vs_phase1={summary['good_FPR_improvement_vs_phase1']}")
    print(f"expanded_semantic_shuffle_margin={summary['expanded_semantic_shuffle_margin']}")
    print(f"expanded_component_shuffle_margin={summary['expanded_component_shuffle_margin']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
