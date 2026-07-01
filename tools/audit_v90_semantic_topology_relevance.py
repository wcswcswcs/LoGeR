#!/usr/bin/env python3
"""Audit v90 topology scale-mode relevance against geometry, compact semantic, and shuffle controls."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import write_csv, write_json
from v90_semantic_topology_utils import ROOT, metric_for_signal


DEFAULT_LEDGER = ROOT / "phase2_semantic_topology_scale_mode_ledger"
DEFAULT_OUT = ROOT / "phase3_semantic_topology_relevance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--filter",
        choices=["all", "highobs", "nonseq01", "near", "far", "topology_support_rich", "topology_boundary_rich", "topology_lowobs"],
        default="all",
    )
    return parser.parse_args()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)


def _scores(df: pd.DataFrame) -> dict[str, pd.Series]:
    geom = _num(df, "geometry_dominant_mode_mu").abs()
    compact = (_num(df, "native_delta_log_scale") - _num(df, "compact_semantic_valid_dominant_mode_mu")).abs()
    tvalid = _num(df, "topology_valid_dominant_mode_mu").abs() * (1.0 + _num(df, "topology_valid_mass"))
    tinvalid = _num(df, "topology_invalid_dominant_mode_mu").abs() * _num(df, "topology_invalid_mass")
    return {
        "G0_geometry_dominant_mode": geom,
        "G1_geometry_entropy": _num(df, "geometry_mode_entropy"),
        "G2_observability": _num(df, "observability_score"),
        "C0_v89_compact_semantic_best": compact,
        "T0_topology_valid_mode": tvalid,
        "T1_topology_invalid_conflict": tinvalid,
        "T2_topology_boundary_invalid": _num(df, "topology_boundary_conflict") * (1.0 + tinvalid),
        "T3_topology_component_support": _num(df, "topology_component_support") * (1.0 + _num(df, "topology_valid_mass")),
        "T4_feature_match_topology_support": _num(df, "feature_match_topology_support") * (1.0 + _num(df, "topology_valid_mass")),
        "T5_topology_valid_native_mismatch": (_num(df, "native_delta_log_scale") - _num(df, "topology_valid_dominant_mode_mu")).abs() * (1.0 + _num(df, "topology_valid_mass")),
        "T6_topology_entropy_reduction": _num(df, "topology_entropy_reduction").clip(lower=0.0),
        "T7_topology_observability": _num(df, "O_topology_scale"),
        "T8_topology_combined_policy_score": tvalid + tinvalid + _num(df, "topology_boundary_conflict") - 0.5 * _num(df, "topology_context_mass"),
    }


def _apply_filter(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    out = df.copy()
    if name == "highobs":
        q = _num(out, "observability_score").quantile(0.50)
        before = len(out)
        out = out[_num(out, "observability_score") >= q].copy()
        notes.append(f"highobs:q50={q}:{before}->{len(out)}")
    elif name == "nonseq01":
        before = len(out)
        out = out[out["seq"].astype(str).str.zfill(2) != "01"].copy()
        notes.append(f"nonseq01:{before}->{len(out)}")
    elif name in {"near", "far"}:
        q = _num(out, "geometry_dominant_mode_mu").abs().quantile(0.50)
        before = len(out)
        mask = _num(out, "geometry_dominant_mode_mu").abs() <= q
        out = out[mask if name == "near" else ~mask].copy()
        notes.append(f"{name}_abs_geometry_mode_split:q50={q}:{before}->{len(out)}")
    elif name == "topology_support_rich":
        q = _num(out, "topology_valid_mass").quantile(0.75)
        before = len(out)
        out = out[_num(out, "topology_valid_mass") >= q].copy()
        notes.append(f"topology_support_rich:q75={q}:{before}->{len(out)}")
    elif name == "topology_boundary_rich":
        q = _num(out, "topology_boundary_conflict").quantile(0.75)
        before = len(out)
        out = out[_num(out, "topology_boundary_conflict") >= q].copy()
        notes.append(f"topology_boundary_rich:q75={q}:{before}->{len(out)}")
    elif name == "topology_lowobs":
        q = _num(out, "topology_lowobs_mass").quantile(0.75)
        before = len(out)
        out = out[_num(out, "topology_lowobs_mass") >= q].copy()
        notes.append(f"topology_lowobs:q75={q}:{before}->{len(out)}")
    return out, notes


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.ledger_dir / "topology_pair_rows.csv")
    df["seq"] = df["seq"].astype(str).str.zfill(2)
    df, filter_notes = _apply_filter(df, args.filter)
    scores = _scores(df)
    geometry_metrics = []
    dummy = {"spearman_rho_abs_log_scale_jump": -1e9, "bad_recall": 0.0, "good_false_positive_rate": 1.0}
    for name in ["G0_geometry_dominant_mode", "G1_geometry_entropy", "G2_observability"]:
        geometry_metrics.append(metric_for_signal(df, name, scores[name], dummy))
    geometry_ref = sorted(
        geometry_metrics,
        key=lambda row: (row.get("spearman_rho_abs_log_scale_jump") if row.get("spearman_rho_abs_log_scale_jump") is not None else -1e9, row.get("bad_recall", 0.0), -row.get("good_false_positive_rate", 1.0)),
        reverse=True,
    )[0]
    metrics: list[dict[str, Any]] = geometry_metrics
    for name, vals in scores.items():
        if name.startswith("G"):
            continue
        metrics.append(metric_for_signal(df, name, vals, geometry_ref))
    topology_entropy_reduction_mean = float(_num(df, "topology_entropy_reduction").clip(lower=0.0).mean()) if len(df) else 0.0
    for row in metrics:
        row["global_gate_components_pass"] = bool(
            row.get("available_rows", 0) >= 12
            and row.get("sequence_coverage", 0) >= 3
            and (
                not row.get("is_topology_conditioned", False)
                or (
                    row.get("semantic_shuffle_margin") is not None
                    and row.get("component_shuffle_margin") is not None
                    and row.get("semantic_shuffle_margin") >= 0.05
                    and row.get("component_shuffle_margin") >= 0.05
                )
            )
        )
        lift = (
            row.get("is_topology_conditioned", False)
            and row.get("spearman_rho_abs_log_scale_jump") is not None
            and geometry_ref.get("spearman_rho_abs_log_scale_jump") is not None
            and row["spearman_rho_abs_log_scale_jump"] >= geometry_ref["spearman_rho_abs_log_scale_jump"] + 0.05
        )
        good_protection = row.get("is_topology_conditioned", False) and row.get("good_false_positive_rate", 1.0) <= 0.25 and row.get("bad_recall", 0.0) >= max(0.45, geometry_ref.get("bad_recall", 0.0) - 0.10)
        entropy_reduction = bool(
            row.get("is_topology_conditioned", False)
            and row.get("signal") == "T6_topology_entropy_reduction"
            and topology_entropy_reduction_mean >= 0.10
            and row.get("bad_recall", 0.0) >= 0.50
        )
        row["phase3_global_signal_pass"] = bool(
            row.get("is_topology_conditioned", False)
            and row["global_gate_components_pass"]
            and (
                (row.get("spearman_rho_abs_log_scale_jump") is not None and row["spearman_rho_abs_log_scale_jump"] >= 0.30)
                or row.get("bad_recall", 0.0) >= 0.55
            )
            and row.get("good_false_positive_rate", 1.0) <= 0.25
            and (lift or good_protection or entropy_reduction)
        )
    pass_rows = [row for row in metrics if row.get("phase3_global_signal_pass")]
    topo_rows = [row for row in metrics if row.get("is_topology_conditioned")]
    best_topology = sorted(
        topo_rows,
        key=lambda row: (bool(row.get("phase3_global_signal_pass")), row.get("spearman_rho_abs_log_scale_jump") if row.get("spearman_rho_abs_log_scale_jump") is not None else -1e9, row.get("bad_recall", 0.0), -row.get("good_false_positive_rate", 1.0)),
        reverse=True,
    )[0] if topo_rows else {}
    rows_out = df.copy()
    for name, vals in scores.items():
        rows_out[name] = vals
    summary = {
        "phase": "Phase3_semantic_topology_relevance",
        "filter": args.filter,
        "filters": filter_notes,
        "phase3_topology_relevance_global_gate_pass": bool(pass_rows),
        "passing_topology_signals": [row["signal"] for row in pass_rows],
        "geometry_reference_signal": geometry_ref.get("signal"),
        "geometry_reference_rho": geometry_ref.get("spearman_rho_abs_log_scale_jump"),
        "geometry_reference_bad_recall": geometry_ref.get("bad_recall"),
        "geometry_reference_good_fpr": geometry_ref.get("good_false_positive_rate"),
        "best_topology_signal": best_topology,
        "scale_label_rows": int(pd.to_numeric(df["abs_log_scale_jump_gt"], errors="coerce").notna().sum()),
        "sequence_coverage": int(df[pd.to_numeric(df["abs_log_scale_jump_gt"], errors="coerce").notna()]["seq"].nunique()),
        "semantic_shuffle_controls": "computed_per_signal",
        "component_shuffle_controls": "computed_per_signal",
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "note": "Offline scale labels are audit-only; topology scores are deterministic functions of Phase1/Phase2 artifacts.",
    }
    if not pass_rows:
        summary["blocker"] = "topology_signal_not_scale_relevant_or_not_specific"
    write_csv(args.out_dir / "topology_relevance_by_signal.csv", metrics)
    write_csv(args.out_dir / "topology_relevance_rows.csv", rows_out.to_dict("records"))
    write_json(args.out_dir / "topology_relevance_summary.json", summary)
    report = [
        "# v90 Phase3 Semantic Topology Relevance",
        "",
        f"- global_gate_pass: `{summary['phase3_topology_relevance_global_gate_pass']}`",
        f"- geometry_reference_signal: `{summary['geometry_reference_signal']}`",
        f"- geometry_reference_rho: `{summary['geometry_reference_rho']}`",
        f"- best_topology_signal: `{best_topology.get('signal')}`",
        f"- best_topology_rho: `{best_topology.get('spearman_rho_abs_log_scale_jump')}`",
        f"- best_topology_semantic_shuffle_margin: `{best_topology.get('semantic_shuffle_margin')}`",
        f"- best_topology_component_shuffle_margin: `{best_topology.get('component_shuffle_margin')}`",
        "",
        "Scale labels are audit-only and are not runtime features.",
    ]
    if summary.get("blocker"):
        report.append(f"- blocker: `{summary['blocker']}`")
    (args.out_dir / "topology_relevance_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"phase3_topology_relevance_global_gate_pass={summary['phase3_topology_relevance_global_gate_pass']}")
    print(f"passing_topology_signals={summary['passing_topology_signals']}")
    print(f"geometry_reference_signal={summary['geometry_reference_signal']}")
    print(f"geometry_reference_rho={summary['geometry_reference_rho']}")
    print(f"best_topology_signal={best_topology.get('signal')}")
    print(f"best_topology_rho={best_topology.get('spearman_rho_abs_log_scale_jump')}")
    print(f"best_topology_semantic_shuffle_margin={best_topology.get('semantic_shuffle_margin')}")
    print(f"best_topology_component_shuffle_margin={best_topology.get('component_shuffle_margin')}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
