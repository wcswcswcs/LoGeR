#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def finite_stats(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if v == v]
    if not vals:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": len(vals),
        "min": min(vals),
        "mean": mean(vals),
        "median": median(vals),
        "max": max(vals),
    }


def aggregate_from_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"status": "missing"}
    aggregate = summary.get("aggregate", {})
    return {
        "status": "present",
        "median_full_ATE_rel_improvement": aggregate.get("median_full_ATE_rel_improvement"),
        "median_rolling_p90_rel_improvement": aggregate.get("median_rolling_p90_rel_improvement"),
        "median_segment_scale_rel_improvement": aggregate.get("median_segment_scale_rel_improvement"),
        "max_full_ATE_harm_rel": aggregate.get("max_full_ATE_harm_rel"),
        "segment_scale_not_worse_all": aggregate.get("segment_scale_not_worse_all"),
        "pilot_geometry_gate_pass": aggregate.get("pilot_geometry_gate", {}).get("pass"),
        "claim_boundary": aggregate.get("claim_boundary"),
    }


def comparison_by_seq(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not summary:
        return []
    rows = []
    for row in summary.get("comparison_rows", []):
        rows.append(
            {
                "seq": row.get("seq"),
                "full_ATE_rel_improvement": row.get("full_ATE_sim3_rmse_rel_improvement"),
                "rolling_p90_rel_improvement": row.get("rolling_ate_p90_rel_improvement"),
                "segment_scale_rel_improvement": row.get("segment_scale_log_error_median_abs_rel_improvement"),
                "candidate_full_ATE_sim3_rmse": row.get("candidate_full_ATE_sim3_rmse"),
                "candidate_segment_scale_log_error_median_abs": row.get("candidate_segment_scale_log_error_median_abs"),
                "baseline_full_ATE_sim3_rmse": row.get("baseline_full_ATE_sim3_rmse"),
                "baseline_segment_scale_log_error_median_abs": row.get("baseline_segment_scale_log_error_median_abs"),
            }
        )
    return rows


def summarize_audit(case_dir: Path) -> dict[str, Any]:
    lq_rows = read_csv_rows(case_dir / "hs_lq_action_gate_rows.csv")
    mrt_rows = read_csv_rows(case_dir / "hs_mrt_readout_probe_rows.csv")
    neutral_counts: dict[str, int] = {}
    for row in lq_rows:
        key = row.get("rowmean_neutral", "")
        neutral_counts[key] = neutral_counts.get(key, 0) + 1

    scale_deltas = [
        v
        for v in (as_float(row.get("predicted_metric_scale_delta")) for row in mrt_rows)
        if v is not None
    ]
    scale_values = [
        v
        for v in (as_float(row.get("predicted_metric_scale")) for row in mrt_rows)
        if v is not None
    ]
    return {
        "case_dir": str(case_dir),
        "hs_lq_action_gate_rows": len(lq_rows),
        "rowmean_neutral_counts": neutral_counts,
        "gate_mean": finite_stats([v for v in (as_float(row.get("gate_mean")) for row in lq_rows) if v is not None]),
        "changed_token_count_abs_gt_1e_4": finite_stats(
            [v for v in (as_float(row.get("changed_token_count_abs_gt_1e_4")) for row in lq_rows) if v is not None]
        ),
        "semantic_risk_mean": finite_stats(
            [v for v in (as_float(row.get("semantic_risk_mean")) for row in lq_rows) if v is not None]
        ),
        "semantic_stable_mean": finite_stats(
            [v for v in (as_float(row.get("semantic_stable_mean")) for row in lq_rows) if v is not None]
        ),
        "hs_mrt_readout_probe_rows": len(mrt_rows),
        "predicted_metric_scale": finite_stats(scale_values),
        "predicted_metric_scale_delta": finite_stats(scale_deltas),
    }


def branch_status(fresh: dict[str, Any], rowmean: dict[str, Any]) -> str:
    if fresh.get("status") != "present" or rowmean.get("status") != "present":
        return "INCOMPLETE_SUMMARY"
    if fresh.get("pilot_geometry_gate_pass") is True:
        return "CARRIER_REPAIR_GATE_PASS_DIAGNOSTIC_ONLY"
    scale_repaired = rowmean.get("segment_scale_not_worse_all") is True
    geometry_positive = (
        (fresh.get("median_full_ATE_rel_improvement") or 0.0) > 0.0
        or (fresh.get("median_rolling_p90_rel_improvement") or 0.0) > 0.0
    )
    if scale_repaired and geometry_positive:
        return "PARTIAL_SCALE_REPAIR_GEOMETRY_GATE_FAIL"
    if not scale_repaired and geometry_positive:
        return "TRADEOFF_REMAINS_SCALE_NOT_FIXED"
    return "NO_GO_GEOMETRY_OR_SCALE"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# V116 Carrier Repair Diagnostic",
        "",
        "Scope: generic rowmean value carrier plus MRT risk/scale-delta safety diagnostics. This report does not promote a semantic-direction claim.",
        "",
        "## Branch Matrix",
        "",
        "| branch | status | vs fresh full | vs fresh rolling | vs fresh scale | vs rowmean scale | fresh gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in payload["branches"]:
        fresh = branch["vs_fresh_noaction"]["aggregate"]
        rowmean = branch["vs_rowmean_only"]["aggregate"]
        lines.append(
            "| {name} | {status} | {full} | {rolling} | {fresh_scale} | {rowmean_scale} | {gate} |".format(
                name=branch["name"],
                status=branch["diagnostic_status"],
                full=fresh.get("median_full_ATE_rel_improvement"),
                rolling=fresh.get("median_rolling_p90_rel_improvement"),
                fresh_scale=fresh.get("median_segment_scale_rel_improvement"),
                rowmean_scale=rowmean.get("median_segment_scale_rel_improvement"),
                gate=fresh.get("pilot_geometry_gate_pass"),
            )
        )
    lines.extend(["", "## Audit Activation", ""])
    for branch in payload["branches"]:
        lines.append(f"### {branch['name']}")
        for seq, audit in branch["audit_by_seq"].items():
            lines.append(
                "- seq {seq}: lq_rows={lq}, rowmean_neutral_counts={neutral}, gate_mean={gate}, changed_tokens={changed}, mrt_rows={mrt}, scale_delta={delta}".format(
                    seq=seq,
                    lq=audit["hs_lq_action_gate_rows"],
                    neutral=json.dumps(audit["rowmean_neutral_counts"], sort_keys=True),
                    gate=audit["gate_mean"],
                    changed=audit["changed_token_count_abs_gt_1e_4"],
                    mrt=audit["hs_mrt_readout_probe_rows"],
                    delta=audit["predicted_metric_scale_delta"],
                )
            )
        lines.append("")
    lines.extend(["## Verdict", "", payload["verdict"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v116 carrier repair diagnostic matrix.")
    parser.add_argument("--results-root", default="results/acl2_v116tf_fast_semantic_causal_memory_influence")
    parser.add_argument("--seqs", default="00,02")
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    diagnostics = root / "diagnostics"
    seqs = [seq.strip() for seq in args.seqs.split(",") if seq.strip()]
    branch_defs = [
        {
            "name": "rowmean_mrt_scaledelta",
            "case_prefix": "v116tf_carrier_rowmean_mrt_scaledelta_fullpilot_full_kitti",
            "fresh_summary": "v116tf_carrier_rowmean_mrt_scaledelta_fullpilot_vs_fresh_noaction_summary.json",
            "rowmean_summary": "v116tf_carrier_rowmean_mrt_scaledelta_fullpilot_vs_rowmean_only_summary.json",
        },
        {
            "name": "rowmean_mrt_scaledelta_tight",
            "case_prefix": "v116tf_carrier_rowmean_mrt_scaledelta_tight_fullpilot_full_kitti",
            "fresh_summary": "v116tf_carrier_rowmean_mrt_scaledelta_tight_fullpilot_vs_fresh_noaction_summary.json",
            "rowmean_summary": "v116tf_carrier_rowmean_mrt_scaledelta_tight_fullpilot_vs_rowmean_only_summary.json",
        },
    ]

    branches = []
    for branch_def in branch_defs:
        fresh_summary = read_json(diagnostics / branch_def["fresh_summary"])
        rowmean_summary = read_json(diagnostics / branch_def["rowmean_summary"])
        fresh_aggregate = aggregate_from_summary(fresh_summary)
        rowmean_aggregate = aggregate_from_summary(rowmean_summary)
        branches.append(
            {
                "name": branch_def["name"],
                "diagnostic_status": branch_status(fresh_aggregate, rowmean_aggregate),
                "vs_fresh_noaction": {
                    "summary_json": str(diagnostics / branch_def["fresh_summary"]),
                    "aggregate": fresh_aggregate,
                    "comparison_by_seq": comparison_by_seq(fresh_summary),
                },
                "vs_rowmean_only": {
                    "summary_json": str(diagnostics / branch_def["rowmean_summary"]),
                    "aggregate": rowmean_aggregate,
                    "comparison_by_seq": comparison_by_seq(rowmean_summary),
                },
                "audit_by_seq": {
                    seq: summarize_audit(diagnostics / f"{branch_def['case_prefix']}_{seq}") for seq in seqs
                },
            }
        )

    pass_branches = [b["name"] for b in branches if b["diagnostic_status"] == "CARRIER_REPAIR_GATE_PASS_DIAGNOSTIC_ONLY"]
    partial_branches = [b["name"] for b in branches if b["diagnostic_status"] == "PARTIAL_SCALE_REPAIR_GEOMETRY_GATE_FAIL"]
    tradeoff_branches = [b["name"] for b in branches if b["diagnostic_status"] == "TRADEOFF_REMAINS_SCALE_NOT_FIXED"]
    if pass_branches:
        verdict = (
            "At least one generic carrier repair branch passes the geometry/scale diagnostic gate versus fresh noaction. "
            "This is still diagnostic-only; semantic-direction promotion remains blocked by Task4 controls."
        )
    elif partial_branches:
        verdict = (
            "Scale repair improved the rowmean-only tradeoff for at least one branch, but the fresh-noaction pilot gate did not pass. "
            "This supports carrier/safety analysis only, not semantic promotion."
        )
    elif tradeoff_branches:
        verdict = (
            "The generic rowmean carrier still shows a scale or geometry tradeoff after MRT scale-delta safety. "
            "The allowed repair does not resolve the v116 carrier blocker."
        )
    else:
        verdict = "Required summaries are incomplete or both carrier repair branches failed geometry/scale evidence."

    payload = {
        "results_root": str(root),
        "seqs": seqs,
        "branches": branches,
        "pass_branches": pass_branches,
        "partial_scale_repair_branches": partial_branches,
        "tradeoff_remaining_branches": tradeoff_branches,
        "verdict": verdict,
    }

    out_dir = root / "carrier_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "CARRIER_REPAIR_DIAGNOSTIC_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir / "CARRIER_REPAIR_DIAGNOSTIC_REPORT.md", payload)
    print(json.dumps({"verdict": verdict, "branch_statuses": {b["name"]: b["diagnostic_status"] for b in branches}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
