#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_v113hs_baseline_metric_summary import summarize_sequence


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rel_improvement(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None or abs(base) <= 1e-12:
        return None
    return float((base - cand) / base)


def summarize_pair(
    seq: str,
    baseline_root: Path,
    candidate_root: Path,
    candidate_name: str,
    baseline_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base = summarize_sequence(baseline_root, seq)
    cand = summarize_sequence(candidate_root, seq)
    base["variant"] = baseline_name
    cand["variant"] = candidate_name

    cmp_row: dict[str, Any] = {
        "seq": seq,
        "baseline_variant": baseline_name,
        "candidate_variant": candidate_name,
        "baseline_output_root": str(baseline_root),
        "candidate_output_root": str(candidate_root),
    }
    for metric in [
        "full_ATE_sim3_rmse",
        "rolling_ate_p90",
        "final_error_sim3_aligned",
        "segment_scale_log_error_median_abs",
        "adjacent_log_scale_jump_p90_abs",
        "rpe_delta1_translation_mean",
        "rpe_delta1_rotation_deg_mean",
        "global_sim3_scale",
    ]:
        b = base.get(metric)
        c = cand.get(metric)
        cmp_row[f"baseline_{metric}"] = b
        cmp_row[f"candidate_{metric}"] = c
        cmp_row[f"{metric}_rel_improvement"] = rel_improvement(b, c)
        if b is not None and c is not None:
            cmp_row[f"{metric}_abs_delta_candidate_minus_baseline"] = float(c - b)
    return base, cand, cmp_row


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals = []
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            vals.append(float(value))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v113-HS action-vs-baseline metrics.")
    parser.add_argument("--results-root", default="results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence")
    parser.add_argument("--seqs", default="00,02")
    parser.add_argument("--baseline-name", default="sliding10_baseline")
    parser.add_argument("--candidate-name", default="HS_A1_patch_only_mild_no_mrt_sliding10")
    parser.add_argument("--baseline-template", default="outputs/stage4_sliding10_baseline_kitti_{seq}")
    parser.add_argument("--baseline-template-by-seq-json", default="")
    parser.add_argument("--candidate-template", default="outputs/stage4_sliding10_hs_a1_patch_only_mild_no_mrt_kitti_{seq}")
    parser.add_argument("--output-prefix", default="stage4_sliding10_hs_a1_patch_only_mild_no_mrt")
    parser.add_argument("--claim-boundary", default="")
    parser.add_argument("--median-improvement-threshold", type=float, default=0.05)
    parser.add_argument("--max-full-ate-harm-rel-threshold", type=float, default=0.02)
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    seqs = [seq.strip() for seq in args.seqs.split(",") if seq.strip()]
    baseline_template_by_seq = json.loads(args.baseline_template_by_seq_json) if args.baseline_template_by_seq_json.strip() else {}

    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for seq in seqs:
        baseline_template = baseline_template_by_seq.get(seq, args.baseline_template)
        baseline_root = root / baseline_template.format(seq=seq)
        candidate_root = root / args.candidate_template.format(seq=seq)
        base, cand, cmp_row = summarize_pair(
            seq=seq,
            baseline_root=baseline_root,
            candidate_root=candidate_root,
            candidate_name=args.candidate_name,
            baseline_name=args.baseline_name,
        )
        metric_rows.extend([base, cand])
        comparison_rows.append(cmp_row)

    full_ate_improvements = finite_values(comparison_rows, "full_ATE_sim3_rmse_rel_improvement")
    rolling_improvements = finite_values(comparison_rows, "rolling_ate_p90_rel_improvement")
    segment_improvements = finite_values(comparison_rows, "segment_scale_log_error_median_abs_rel_improvement")
    full_ate_harms = [max(0.0, -v) for v in full_ate_improvements]
    claim_boundary = args.claim_boundary.strip()
    if not claim_boundary:
        if "sliding10" in args.baseline_name or "sliding10" in args.baseline_template or "sliding10" in args.candidate_template:
            claim_boundary = "Sliding-size-10 action comparison is a memory-safe pilot branch and is not the default sliding-size-21 full-pilot result."
        else:
            claim_boundary = "Default HorizonStream configuration comparison against the Stage1 default local baseline."

    aggregate = {
        "seqs": seqs,
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "median_full_ATE_rel_improvement": float(np.median(full_ate_improvements)) if full_ate_improvements else None,
        "median_rolling_p90_rel_improvement": float(np.median(rolling_improvements)) if rolling_improvements else None,
        "median_segment_scale_rel_improvement": float(np.median(segment_improvements)) if segment_improvements else None,
        "max_full_ATE_harm_rel": float(max(full_ate_harms)) if full_ate_harms else None,
        "improved_seq_count_full_ATE": int(sum(v > 0 for v in full_ate_improvements)),
        "segment_scale_not_worse_all": bool(segment_improvements and all(v >= 0 for v in segment_improvements)),
        "pilot_geometry_gate": {
            "thresholds": {
                "median_full_ATE_rel_improvement_ge": float(args.median_improvement_threshold),
                "or_median_rolling_p90_rel_improvement_ge": float(args.median_improvement_threshold),
                "max_full_ATE_harm_rel_le": float(args.max_full_ate_harm_rel_threshold),
                "segment_scale_not_worse_all": True,
            },
            "pass": bool(
                full_ate_improvements
                and rolling_improvements
                and segment_improvements
                and (
                    float(np.median(full_ate_improvements)) >= float(args.median_improvement_threshold)
                    or float(np.median(rolling_improvements)) >= float(args.median_improvement_threshold)
                )
                and max(full_ate_harms) <= float(args.max_full_ate_harm_rel_threshold)
                and all(v >= 0 for v in segment_improvements)
            ),
        },
        "semantic_causality_gate": {
            "required_controls": [
                "semantic_shuffle_by_frame",
                "role_rotation_dynamic_stable",
                "same_count_high_risk_frame_random",
                "low_risk_reverse",
            ],
            "status": "not_run",
            "pass": False,
        },
        "claim_boundary": claim_boundary,
    }

    out_dir = root / "diagnostics"
    write_csv(out_dir / f"{args.output_prefix}_metrics_rows.csv", metric_rows)
    write_csv(out_dir / f"{args.output_prefix}_comparison_rows.csv", comparison_rows)
    summary = {"metric_rows": metric_rows, "comparison_rows": comparison_rows, "aggregate": aggregate}
    (out_dir / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
