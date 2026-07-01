#!/usr/bin/env python3
"""Audit v89 Phase3 feature-match semantic ruler evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import read_json, spearman_rho, stable_hash_float, write_csv, write_json


DEFAULT_DIR = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control/phase3_feature_match_semantic_ruler")
DEFAULT_PHASE2 = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control/phase2_semantic_mode_relevance/semantic_mode_relevance_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--phase2-summary", type=Path, default=DEFAULT_PHASE2)
    return parser.parse_args()


def _num(series: pd.Series) -> pd.Series:
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


def main() -> None:
    args = parse_args()
    build = read_json(args.match_dir / "feature_match_build_summary.json")
    phase2 = read_json(args.phase2_summary) if args.phase2_summary.exists() else {}
    pairs = pd.read_csv(args.match_dir / "feature_match_pair_summary.csv")
    labelled = pairs[_num(pairs["abs_log_scale_jump_gt"]).notna()].copy() if len(pairs) else pairs
    score = _num(labelled.get("match_valid_score", pd.Series(dtype=float))).fillna(0.0)
    y = _num(labelled.get("abs_log_scale_jump_gt", pd.Series(dtype=float)))
    rho = spearman_rho(score.tolist(), y.tolist()) if len(labelled) else None
    semantic_shuffle_rho = spearman_rho(_shuffle(score, "phase3_match_semantic_shuffle").tolist(), y.tolist()) if len(labelled) else None
    random_rho = spearman_rho(_shuffle(score, "phase3_match_same_count_random").tolist(), y.tolist()) if len(labelled) else None
    max_control = max([v for v in [semantic_shuffle_rho, random_rho] if v is not None], default=None)
    margin = None if rho is None or max_control is None else float(rho - max_control)
    geometry_rho = phase2.get("geometry_reference_rho")
    high_quality = pairs[pairs.get("verified_inlier_count", 0).notna()] if len(pairs) else pairs
    valid_ratio = float(_num(pairs.get("match_semantic_valid_ratio", pd.Series(dtype=float))).median()) if len(pairs) else 0.0
    cross_ratio = float(_num(pairs.get("match_cross_boundary_ratio", pd.Series(dtype=float))).median()) if len(pairs) else 1.0
    gate = bool(
        build.get("matcher_available")
        and build.get("verified_inlier_count_median", 0) >= 30
        and build.get("sequence_coverage", 0) >= 3
        and valid_ratio >= 0.50
        and cross_ratio <= 0.25
        and (
            (rho is not None and geometry_rho is not None and rho >= geometry_rho + 0.05)
            or (margin is not None and margin >= 0.05)
        )
        and (semantic_shuffle_rho is None or rho is None or rho >= semantic_shuffle_rho + 0.05)
        and (random_rho is None or rho is None or rho >= random_rho + 0.05)
    )
    audit = {
        "phase": "Phase3_feature_match_semantic_ruler_audit",
        "feature_match_semantic_ruler_gate_pass": gate,
        "matcher_available": build.get("matcher_available"),
        "matcher_type": build.get("matcher_type"),
        "verified_inlier_count_median": build.get("verified_inlier_count_median"),
        "sequence_coverage": build.get("sequence_coverage"),
        "match_semantic_valid_ratio_median": valid_ratio,
        "cross_boundary_match_ratio_median": cross_ratio,
        "match_valid_score_rho_abs_log_scale_jump": rho,
        "geometry_reference_rho": geometry_rho,
        "semantic_shuffle_rho": semantic_shuffle_rho,
        "same_count_random_match_rho": random_rho,
        "semantic_shuffle_match_margin": margin,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    if not gate:
        audit["blocker"] = "feature_match_semantic_ruler_gate_failed"
    write_json(args.match_dir / "feature_match_audit_summary.json", audit)
    write_csv(
        args.match_dir / "feature_match_audit_controls.csv",
        [
            {"control": "semantic_shuffle", "rho": semantic_shuffle_rho},
            {"control": "same_count_random_match", "rho": random_rho},
        ],
    )
    report = [
        "# v89 Phase3 Feature-Match Semantic Ruler Audit",
        "",
        f"- gate_pass: `{audit['feature_match_semantic_ruler_gate_pass']}`",
        f"- matcher_type: `{audit['matcher_type']}`",
        f"- verified_inlier_count_median: `{audit['verified_inlier_count_median']}`",
        f"- match_semantic_valid_ratio_median: `{audit['match_semantic_valid_ratio_median']}`",
        f"- cross_boundary_match_ratio_median: `{audit['cross_boundary_match_ratio_median']}`",
        f"- rho: `{audit['match_valid_score_rho_abs_log_scale_jump']}`",
        f"- semantic_shuffle_match_margin: `{audit['semantic_shuffle_match_margin']}`",
        f"- blocker: `{audit.get('blocker', '')}`",
    ]
    (args.match_dir / "feature_match_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"feature_match_semantic_ruler_gate_pass={audit['feature_match_semantic_ruler_gate_pass']}")
    print(f"matcher_type={audit['matcher_type']}")
    print(f"verified_inlier_count_median={audit['verified_inlier_count_median']}")
    print(f"match_semantic_valid_ratio_median={audit['match_semantic_valid_ratio_median']}")
    print(f"cross_boundary_match_ratio_median={audit['cross_boundary_match_ratio_median']}")
    print(f"match_valid_score_rho={audit['match_valid_score_rho_abs_log_scale_jump']}")
    print(f"semantic_shuffle_match_margin={audit['semantic_shuffle_match_margin']}")
    if audit.get("blocker"):
        print(f"blocker={audit['blocker']}")


if __name__ == "__main__":
    main()
