#!/usr/bin/env python3
"""Audit v91 regime-conditioned semantic relevance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v91_semantic_regime_utils import ROOT, nseries, policy_metric, positive_loso_folds, stable_shuffle


DEFAULT_REGIME = ROOT / "phase2_semantic_regime_classifier"
DEFAULT_OUT = ROOT / "phase3_regime_conditioned_semantic_relevance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime-dir", type=Path, default=DEFAULT_REGIME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _semantic_score(df: pd.DataFrame) -> pd.Series:
    scores = []
    for _, row in df.iterrows():
        regime = str(row["regime"])
        if regime == "REGIME_FAR_OPEN_HIGHOBS":
            val = row["S_invalid"] + row["boundary_mass"] + max(0.0, row["H_mode"] - row["H_topo"])
        elif regime == "REGIME_NEAR_STRUCTURED":
            val = row["S_valid"] - row["S_invalid"]
        elif regime == "REGIME_BOUNDARY_RICH":
            val = row["S_invalid"] + row["boundary_mass"] + row.get("split_merge_score", 0.0) - 0.5 * row["S_valid"]
        elif regime == "REGIME_LOWOBS_CONTEXT":
            val = row["S_context"] + row["S_lowobs"] - row["S_valid"]
        elif regime == "REGIME_MULTIMODE_CONFLICT":
            val = row["H_mode"] * max(0.0, 1.0 - row["S_valid"])
        else:
            val = row["S_valid"] * (1.0 - row["regime_confidence"]) + abs(row["geometry_dominant_mode_mu"])
        scores.append(float(val))
    return pd.Series(scores, index=df.index)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.regime_dir / "semantic_regime_rows.csv")
    df["seq"] = df["seq"].astype(str).str.zfill(2)
    geometry = nseries(df, "geometry_dominant_mode_mu").abs() + 0.25 * nseries(df, "H_mode")
    v90_topology = nseries(df, "S_invalid") + nseries(df, "boundary_mass") - 0.25 * nseries(df, "S_valid")
    regime_geometry = geometry * (1.0 + nseries(df, "regime_confidence").clip(0, 1))
    semantic = _semantic_score(df)
    semantic_match = semantic * (1.0 + (nseries(df, "feature_match_support_count") / max(float(nseries(df, "feature_match_support_count").max()), 1.0)).clip(0, 1))
    policy_state_score = semantic_match + nseries(df, "S_invalid") + nseries(df, "S_context")
    semantic_shuffle = stable_shuffle(semantic_match, "v91_phase3_semantic_shuffle")
    component_shuffle = stable_shuffle(semantic_match, "v91_phase3_component_shuffle")
    regime_shuffle = semantic_match.copy()
    shuffled_regime = stable_shuffle(df["regime"], "v91_phase3_regime_shuffle")
    df_regime_shuffled = df.copy()
    df_regime_shuffled["regime"] = shuffled_regime
    regime_shuffle = _semantic_score(df_regime_shuffled)
    score_map = {
        "P0_GEOMETRY_ONLY_GLOBAL": geometry,
        "P1_V90_TOPOLOGY_GLOBAL": v90_topology,
        "P2_REGIME_GEOMETRY_ONLY": regime_geometry,
        "P3_REGIME_SEMANTIC_TOPOLOGY": semantic,
        "P4_REGIME_SEMANTIC_PLUS_MATCH": semantic_match,
        "P5_REGIME_SEMANTIC_POLICY_STATE": policy_state_score,
        "P6_REGIME_SHUFFLED_SEMANTIC_CONTROL": regime_shuffle,
        "P7_COMPONENT_SHUFFLED_CONTROL": component_shuffle,
        "P8_SEMANTIC_LABEL_SHUFFLED_CONTROL": semantic_shuffle,
    }
    metrics: list[dict[str, Any]] = []
    for name, score in score_map.items():
        row = policy_metric(df, score, name)
        row["LOSO_positive_folds"] = positive_loso_folds(df, score, geometry)
        metrics.append(row)
    by_name = {row["signal"]: row for row in metrics}
    geom_ref = by_name["P2_REGIME_GEOMETRY_ONLY"]
    sem_candidates = [by_name["P3_REGIME_SEMANTIC_TOPOLOGY"], by_name["P4_REGIME_SEMANTIC_PLUS_MATCH"], by_name["P5_REGIME_SEMANTIC_POLICY_STATE"]]
    for row in metrics:
        if row["signal"] in {"P3_REGIME_SEMANTIC_TOPOLOGY", "P4_REGIME_SEMANTIC_PLUS_MATCH", "P5_REGIME_SEMANTIC_POLICY_STATE"}:
            ctrl_sem = by_name["P8_SEMANTIC_LABEL_SHUFFLED_CONTROL"]
            ctrl_comp = by_name["P7_COMPONENT_SHUFFLED_CONTROL"]
            ctrl_reg = by_name["P6_REGIME_SHUFFLED_SEMANTIC_CONTROL"]
            row["semantic_shuffle_margin"] = float((row.get("balanced_accuracy") or (0.5 * (row["bad_recall"] + 1 - row["good_FPR"]))) - (0.5 * (ctrl_sem["bad_recall"] + 1 - ctrl_sem["good_FPR"])))
            row["component_shuffle_margin"] = float((0.5 * (row["bad_recall"] + 1 - row["good_FPR"])) - (0.5 * (ctrl_comp["bad_recall"] + 1 - ctrl_comp["good_FPR"])))
            row["regime_shuffle_margin"] = float((0.5 * (row["bad_recall"] + 1 - row["good_FPR"])) - (0.5 * (ctrl_reg["bad_recall"] + 1 - ctrl_reg["good_FPR"])))
            row["geometry_lift_margin"] = None if row.get("spearman_rho_abs_log_scale_jump") is None or geom_ref.get("spearman_rho_abs_log_scale_jump") is None else float(row["spearman_rho_abs_log_scale_jump"] - geom_ref["spearman_rho_abs_log_scale_jump"])
            row["semantic_good_protection_margin"] = float(geom_ref["good_FPR"] - row["good_FPR"])
            row["phase3_gate_pass"] = bool(
                row["bad_recall"] >= 0.60
                and row["good_FPR"] <= 0.25
                and row["sequence_coverage"] >= 3
                and row["semantic_shuffle_margin"] >= 0.05
                and row["component_shuffle_margin"] >= 0.05
                and row["regime_shuffle_margin"] >= 0.05
                and ((row["geometry_lift_margin"] is not None and row["geometry_lift_margin"] >= 0.05) or row["semantic_good_protection_margin"] >= 0.10)
                and row["LOSO_positive_folds"] >= 3
            )
        else:
            row["phase3_gate_pass"] = False
    passing = [row for row in metrics if row.get("phase3_gate_pass")]
    best_sem = sorted(
        sem_candidates,
        key=lambda row: (bool(row.get("phase3_gate_pass")), row.get("spearman_rho_abs_log_scale_jump") if row.get("spearman_rho_abs_log_scale_jump") is not None else -999, row["bad_recall"], -row["good_FPR"]),
        reverse=True,
    )[0]
    out_rows = df.copy()
    for name, score in score_map.items():
        out_rows[name] = score
    summary = {
        "phase": "Phase3_regime_conditioned_semantic_relevance",
        "phase3_regime_semantic_gate_pass": bool(passing),
        "passing_policies": [row["signal"] for row in passing],
        "geometry_reference_policy": "P2_REGIME_GEOMETRY_ONLY",
        "geometry_reference_rho": geom_ref.get("spearman_rho_abs_log_scale_jump"),
        "geometry_reference_bad_recall": geom_ref.get("bad_recall"),
        "geometry_reference_good_FPR": geom_ref.get("good_FPR"),
        "best_semantic_policy": best_sem,
        "scale_label_rows": int(pd.to_numeric(df["abs_log_scale_jump_gt"], errors="coerce").notna().sum()),
        "sequence_coverage": int(df[pd.to_numeric(df["abs_log_scale_jump_gt"], errors="coerce").notna()]["seq"].nunique()),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "note": "Scale labels are audit-only; regime assignment was built before this audit and records label non-use.",
    }
    if not passing:
        if by_name["P2_REGIME_GEOMETRY_ONLY"]["bad_recall"] >= 0.60 and by_name["P2_REGIME_GEOMETRY_ONLY"]["good_FPR"] <= 0.25:
            summary["blocker"] = "regime_geometry_only_not_semantic_success"
        elif max(row["semantic_shuffle_margin"] for row in sem_candidates) < 0.05:
            summary["blocker"] = "semantic_not_specific_shuffle_margin_failed"
        else:
            summary["blocker"] = "regime_conditioned_semantic_policy_gate_failed"
    write_csv(args.out_dir / "regime_conditioned_policy_metrics.csv", metrics)
    write_csv(args.out_dir / "regime_conditioned_policy_rows.csv", out_rows.to_dict("records"))
    write_json(args.out_dir / "regime_conditioned_relevance_summary.json", summary)
    print(f"phase3_regime_semantic_gate_pass={summary['phase3_regime_semantic_gate_pass']}")
    print(f"passing_policies={summary['passing_policies']}")
    print(f"geometry_reference_rho={summary['geometry_reference_rho']}")
    print(f"best_semantic_policy={best_sem.get('signal')}")
    print(f"best_semantic_rho={best_sem.get('spearman_rho_abs_log_scale_jump')}")
    print(f"best_semantic_bad_recall={best_sem.get('bad_recall')}")
    print(f"best_semantic_good_FPR={best_sem.get('good_FPR')}")
    print(f"best_semantic_shuffle_margin={best_sem.get('semantic_shuffle_margin')}")
    print(f"best_component_shuffle_margin={best_sem.get('component_shuffle_margin')}")
    print(f"best_regime_shuffle_margin={best_sem.get('regime_shuffle_margin')}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
