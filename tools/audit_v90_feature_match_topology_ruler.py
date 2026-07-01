#!/usr/bin/env python3
"""Audit v90 feature-match topology ruler relevance."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from v86_soft_latent_utils import spearman_rho, write_csv, write_json
from v90_semantic_topology_utils import ROOT, stable_shuffle


DEFAULT_OUT = ROOT / "phase5_feature_match_topology_ruler"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)


def main() -> None:
    args = parse_args()
    pair = pd.read_csv(args.out_dir / "feature_match_topology_pair_summary.csv")
    pair["seq"] = pair["seq"].astype(str).str.zfill(2)
    labelled = pair[pd.to_numeric(pair["abs_log_scale_jump_gt"], errors="coerce").notna()].copy()
    score = _num(labelled, "match_topology_valid_score")
    baseline = _num(labelled, "raw_match_count_baseline")
    y = _num(labelled, "abs_log_scale_jump_gt")
    rho = spearman_rho(score.tolist(), y.tolist())
    baseline_rho = spearman_rho(baseline.tolist(), y.tolist())
    sem_shuffle = stable_shuffle(score, "v90_phase5_semantic_shuffle")
    comp_shuffle = stable_shuffle(score, "v90_phase5_component_shuffle")
    sem_rho = spearman_rho(sem_shuffle.tolist(), y.tolist())
    comp_rho = spearman_rho(comp_shuffle.tolist(), y.tolist())
    semantic_margin = None if rho is None or sem_rho is None else float(rho - sem_rho)
    component_margin = None if rho is None or comp_rho is None else float(rho - comp_rho)
    threshold = float(score.quantile(0.75)) if len(score) else 0.0
    flags = score >= threshold
    high = y >= float(y.quantile(0.75)) if len(y) else pd.Series(False, index=labelled.index)
    bad = labelled["base_case_type"].astype(str).eq("bad")
    good_low = labelled["base_case_type"].astype(str).eq("good") & (~high)
    bad_or_high = bad | high
    bad_recall = float((flags & bad_or_high).sum() / max(int(bad_or_high.sum()), 1)) if len(labelled) else 0.0
    good_fpr = float((flags & good_low).sum() / max(int(good_low.sum()), 1)) if len(labelled) else 1.0
    matcher_available = bool(pair.get("matcher_available", pd.Series([False])).astype(str).str.lower().isin(["true", "1"]).any())
    verified_median = float(_num(pair, "verified_inlier_count").median()) if len(pair) else 0.0
    valid_ratio_median = float(_num(pair, "match_topology_valid_ratio").median()) if len(pair) else 0.0
    beats_baseline = bool(rho is not None and baseline_rho is not None and rho >= baseline_rho + 0.05)
    gate = bool(
        matcher_available
        and verified_median >= 100
        and valid_ratio_median >= 0.30
        and ((rho is not None and rho >= 0.30) or bad_recall >= 0.55)
        and good_fpr <= 0.25
        and semantic_margin is not None
        and semantic_margin >= 0.05
        and component_margin is not None
        and component_margin >= 0.05
        and beats_baseline
    )
    controls = [
        {"control": "raw_match_count_baseline", "rho": baseline_rho},
        {"control": "semantic_shuffle", "rho": sem_rho},
        {"control": "component_shuffle", "rho": comp_rho},
    ]
    summary = {
        "phase": "Phase5_feature_match_topology_ruler_audit",
        "feature_match_topology_ruler_gate_pass": gate,
        "matcher_available": matcher_available,
        "verified_inlier_count_median": verified_median,
        "match_topology_valid_ratio_median": valid_ratio_median,
        "match_topology_score_rho_abs_log_scale_jump": rho,
        "raw_match_count_baseline_rho": baseline_rho,
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "semantic_shuffle_rho": sem_rho,
        "component_shuffle_rho": comp_rho,
        "semantic_shuffle_match_margin": semantic_margin,
        "component_shuffle_match_margin": component_margin,
        "beats_raw_match_count_baseline_by_0_05": beats_baseline,
        "sequence_coverage": int(labelled["seq"].nunique()) if len(labelled) else 0,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        summary["blocker"] = "feature_match_topology_ruler_gate_failed"
    write_csv(args.out_dir / "feature_match_topology_audit_controls.csv", controls)
    write_json(args.out_dir / "feature_match_topology_audit_summary.json", summary)
    report = [
        "# v90 Phase5 Feature-Match Topology Ruler",
        "",
        f"- gate_pass: `{gate}`",
        f"- matcher_available: `{matcher_available}`",
        f"- verified_inlier_count_median: `{verified_median}`",
        f"- match_topology_valid_ratio_median: `{valid_ratio_median}`",
        f"- rho: `{rho}`",
        f"- semantic_shuffle_match_margin: `{semantic_margin}`",
        f"- component_shuffle_match_margin: `{component_margin}`",
        f"- beats_raw_match_count_baseline_by_0_05: `{beats_baseline}`",
    ]
    if summary.get("blocker"):
        report.append(f"- blocker: `{summary['blocker']}`")
    (args.out_dir / "feature_match_topology_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"feature_match_topology_ruler_gate_pass={summary['feature_match_topology_ruler_gate_pass']}")
    print(f"matcher_available={summary['matcher_available']}")
    print(f"verified_inlier_count_median={summary['verified_inlier_count_median']}")
    print(f"match_topology_valid_ratio_median={summary['match_topology_valid_ratio_median']}")
    print(f"match_topology_score_rho_abs_log_scale_jump={summary['match_topology_score_rho_abs_log_scale_jump']}")
    print(f"semantic_shuffle_match_margin={summary['semantic_shuffle_match_margin']}")
    print(f"component_shuffle_match_margin={summary['component_shuffle_match_margin']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
