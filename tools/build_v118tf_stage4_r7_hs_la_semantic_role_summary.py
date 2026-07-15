#!/usr/bin/env python3
"""Summarize v118 Stage4-R7 HS-LA semantic-role-only selected-query pilot."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

import build_v118tf_stage4_r6_hs_la_full_calibrated_summary as base


base.STAGE = base.RESULT_ROOT / "stage4_r7_hs_la_semantic_role_only"
base.OUT = base.STAGE / "summary"
base.PREFIX = "stage4_r7_hs_la2_semantic_role_only_tiny_tight"
base.VARIANT = "HS_LA2_semantic_role_only_tiny_tight"
base.CONTROL = "semantic_role_only"
base.SEQS = ["00", "02"]


def report_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    agg = summary["aggregate"]
    lines = [
        "# ACL2 v118-TF Stage4-R7 HS-LA Semantic-Role-Only Selected-Query Pilot",
        "",
        f"- candidate: `{agg['candidate_name']}`",
        f"- control: `{agg['control']}`",
        f"- pilot_gate_plan13_2_pass: `{agg['pilot_gate_plan13_2']['pass']}`",
        f"- median_full_ATE_rel_improvement: `{agg['median_full_ATE_rel_improvement']}`",
        f"- median_rolling_p90_rel_improvement: `{agg['median_rolling_p90_rel_improvement']}`",
        f"- max_full_ATE_harm_rel: `{agg['max_full_ATE_harm_rel']}`",
        f"- segment_scale_not_worse_all: `{agg['segment_scale_not_worse_all']}`",
        "",
        "| seq | baseline ATE | candidate ATE | full ATE rel | rolling p90 rel | segment scale rel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seq']} | {row['baseline_full_ATE_sim3_rmse']} | {row['candidate_full_ATE_sim3_rmse']} | "
            f"{row['full_ATE_sim3_rmse_rel_improvement']} | {row['rolling_ate_p90_rel_improvement']} | "
            f"{row['segment_scale_log_error_median_abs_rel_improvement']} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "This run tests the pre-registered HS-LA semantic-role-only selected-query logit intervention. It does not use internal QK or selected-query read reliability as a decision factor, so it is an ablation, not a full calibrated semantic-aware method.",
    ]
    return "\n".join(lines)


def main() -> None:
    base.OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = [base.manifest_status("00", max_frames=128)] + [base.manifest_status(seq) for seq in base.SEQS]
    audit_rows = [base.action_audit_summary("00", max_frames=128)] + [
        base.action_audit_summary(seq) for seq in base.SEQS
    ]
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for seq in base.SEQS:
        baseline, candidate, comparison = base.summarize_pair(seq)
        metric_rows.extend([baseline, candidate])
        comparison_rows.append(comparison)

    full = base.finite_values(comparison_rows, "full_ATE_sim3_rmse_rel_improvement")
    rolling = base.finite_values(comparison_rows, "rolling_ate_p90_rel_improvement")
    segment = base.finite_values(comparison_rows, "segment_scale_log_error_median_abs_rel_improvement")
    harms = [max(0.0, -value) for value in full]
    median_full = float(np.median(full)) if full else None
    median_rolling = float(np.median(rolling)) if rolling else None
    max_harm = float(max(harms)) if harms else None
    pilot_pass = bool(
        full
        and rolling
        and median_full is not None
        and median_rolling is not None
        and max_harm is not None
        and median_full >= 0.03
        and median_rolling > 0.0
        and max_harm <= 0.01
    )
    segment_not_worse_all = bool(segment and all(value >= 0 for value in segment))
    aggregate = {
        "schema": "acl2_v118tf_stage4_r7_hs_la_semantic_role_metric_summary_v1",
        "candidate_name": base.VARIANT,
        "control": base.CONTROL,
        "seqs": base.SEQS,
        "baseline_name": "v113_baseline_default_no_loop",
        "median_full_ATE_rel_improvement": median_full,
        "median_rolling_p90_rel_improvement": median_rolling,
        "median_segment_scale_rel_improvement": float(np.median(segment)) if segment else None,
        "max_full_ATE_harm_rel": max_harm,
        "improved_seq_count_full_ATE": int(sum(value > 0 for value in full)),
        "full_ATE_not_worse_all": bool(full and all(value >= -0.01 for value in full)),
        "segment_scale_not_worse_all": segment_not_worse_all,
        "pilot_gate_plan13_2": {
            "thresholds": {
                "median_full_ATE_rel_improvement_ge": 0.03,
                "median_rolling_p90_rel_improvement_gt": 0.0,
                "max_full_ATE_harm_rel_le": 0.01,
            },
            "pass": pilot_pass,
        },
        "semantic_causality_gate": {
            "status": "pending_controls_after_pilot_gate_pass" if pilot_pass else "not_triggered_pilot_gate_failed",
            "pass": False,
            "reason": (
                "The HS-LA semantic-role-only action passed the plan 13.2 pilot gate, but controls have not been run."
                if pilot_pass
                else "The HS-LA semantic-role-only action did not reach the plan 13.2 pilot gate."
            ),
        },
    }
    summary = {
        "schema": "acl2_v118tf_stage4_r7_hs_la_semantic_role_summary_v1",
        "aggregate": aggregate,
        "metric_rows": metric_rows,
        "comparison_rows": comparison_rows,
        "manifest_rows": manifest_rows,
        "action_audit_rows": audit_rows,
        "outputs": {
            "summary": base.rel(base.OUT / "stage4_r7_hs_la_semantic_role_summary.json"),
            "report": base.rel(base.OUT / "STAGE4_R7_HS_LA_SEMANTIC_ROLE_REPORT.md"),
            "comparison_rows": base.rel(base.OUT / "stage4_r7_hs_la_semantic_role_comparison_rows.csv"),
            "metric_rows": base.rel(base.OUT / "stage4_r7_hs_la_semantic_role_metric_rows.csv"),
            "action_audit_summary": base.rel(base.OUT / "stage4_r7_hs_la_semantic_role_action_audit_summary.csv"),
            "manifest_summary": base.rel(base.OUT / "stage4_r7_hs_la_semantic_role_manifest_summary.csv"),
        },
    }
    base.write_csv(base.OUT / "stage4_r7_hs_la_semantic_role_metric_rows.csv", metric_rows)
    base.write_csv(base.OUT / "stage4_r7_hs_la_semantic_role_comparison_rows.csv", comparison_rows)
    base.write_csv(base.OUT / "stage4_r7_hs_la_semantic_role_action_audit_summary.csv", audit_rows)
    base.write_csv(base.OUT / "stage4_r7_hs_la_semantic_role_manifest_summary.csv", manifest_rows)
    base.write_json(base.OUT / "stage4_r7_hs_la_semantic_role_summary.json", summary)
    base.write_text(base.OUT / "STAGE4_R7_HS_LA_SEMANTIC_ROLE_REPORT.md", report_text(summary, comparison_rows))
    base.add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R7",
            "surface_or_branch": "HS-LA",
            "status": "PASS_PILOT_GATE_PENDING_CONTROLS" if pilot_pass else "NO_GO_PILOT_GATE_FAILED",
            "artifact": base.rel(base.OUT / "stage4_r7_hs_la_semantic_role_summary.json"),
            "notes": "HS-LA semantic-role-only selected-query full 00/02 pilot complete; controls not triggered unless pilot gate passes",
        }
    )
    print(json.dumps(base.clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
