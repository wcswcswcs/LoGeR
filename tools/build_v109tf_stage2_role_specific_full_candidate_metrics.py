#!/usr/bin/env python3
"""Build v109TF Stage2 role-specific full-candidate metrics."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v109tf_stage2_role_specific_pilot_metrics as rolem  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
OUT = RESULT_ROOT / "stage2_role_specific_full_candidates"
SEQUENCES = ("00", "01", "02", "05")


def full_candidate_taxonomy(
    metric_complete: bool,
    all_action_fidelity: bool,
    summary_rows: list[dict[str, Any]],
) -> tuple[str, bool, str]:
    if not metric_complete:
        return "ROLE_FULL_CANDIDATE_METRICS_NOT_COMPLETE", False, "role_full_candidate_metrics_not_complete"
    if not all_action_fidelity:
        return "ROLE_FULL_CANDIDATE_ACTION_FIDELITY_FAIL", False, "role_full_candidate_action_fidelity_fail"
    pre_gate_rows = [row for row in summary_rows if bool(row.get("role_pilot_pre_gate_pass"))]
    if not pre_gate_rows:
        return "ROLE_FULL_CANDIDATE_NO_ROLE_SURPASSES_GATE", False, "no_role_specific_policy_passed_full_candidate_gate"
    best_vs_f1 = max(
        rolem.safe_float(row.get("median_full_ATE_relative_improvement_vs_stage2_F1_same_seq", "nan"))
        for row in pre_gate_rows
    )
    if math.isfinite(best_vs_f1) and best_vs_f1 >= -rolem.PILOT_F1_TOL:
        return "ROLE_FULL_CANDIDATE_MATCHES_STAGE2_F1", True, ""
    return (
        "ROLE_FULL_CANDIDATE_BEATS_BASELINE_BUT_NOT_STAGE2_F1",
        False,
        "role_full_candidate_does_not_match_stage2_f1",
    )


def full_report(summary: dict[str, Any], rows: list[dict[str, str]]) -> str:
    lines = [
        "# ACL2 v109TF Stage2 Role-Specific Full-Candidate Report",
        "",
        f"metric_complete: {summary['metric_complete']}",
        f"role_full_candidate_pre_gate_any_pass: {summary['role_full_candidate_pre_gate_any_pass']}",
        f"taxonomy: {summary['taxonomy']}",
        f"blocker: {summary['blocker']}",
        f"observed_run_workers: {summary['observed_run_worker_count']}/{summary['expected_run_worker_count']}",
        f"observed_evaluate_rows: {summary['observed_evaluate_count']}/{summary['expected_run_worker_count']}",
        f"observed_report_rows: {summary['observed_report_count']}/{summary['expected_run_worker_count']}",
        "",
        "## Policy Summary",
        "",
    ]
    for row in rows:
        lines.append(
            "- {policy_id}: median_full_rel={median_full_rel_improvement} "
            "improved={num_seq_improved}/{sequence_count} max_harm={max_harm} "
            "local_max_harm={local_window_max_harm} vs_F1_median={median_full_ATE_relative_improvement_vs_stage2_F1_same_seq} "
            "pre_gate={role_pilot_pre_gate_pass}".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a full KITTI 00/01/02/05 role-specific candidate check promoted from the 00/02 role pilot. It can support a role-specific semantic explanation only if it survives full-sequence controls and matches or beats the Stage2 F1 same-sequence reference within tolerance.",
        ]
    )
    return "\n".join(lines)


def configure() -> None:
    rolem.OUT = OUT
    rolem.CONFIG_ROWS = OUT / "action_config_rows.csv"
    rolem.RUN_RESULTS = OUT / "run_results.csv"
    rolem.WORKSPACE = OUT / "workspace"
    rolem.SEQUENCES = SEQUENCES
    rolem.SUMMARY_ROW_SCHEMA = "acl2_v109tf_stage2_role_specific_full_candidate_summary_row_v1"
    rolem.SUMMARY_SCHEMA = "acl2_v109tf_stage2_role_specific_full_candidate_summary_v1"
    rolem.SUMMARY_JSON = "role_specific_full_candidate_summary.json"
    rolem.REPORT_MD = "role_specific_full_candidate_report.md"
    rolem.REPORT_TITLE = "# ACL2 v109TF Stage2 Role-Specific Full-Candidate Report"
    rolem.SCOPE_NOTE = "full KITTI 00/01/02/05 role-specific candidate; not a method success claim by itself"
    rolem.GATE_SCOPE = "KITTI 00/01/02/05 full candidate"
    rolem.INTERPRETATION_TEXT = (
        "This is a full KITTI role-specific candidate check. Passing it can support a role-specific "
        "semantic explanation only if it also matches or beats Stage2 F1 same-sequence reference within tolerance."
    )
    rolem.taxonomy = full_candidate_taxonomy


def build() -> dict[str, Any]:
    configure()
    summary = rolem.build()
    summary["role_full_candidate_pass"] = summary.get("role_pilot_pass", False)
    summary["role_full_candidate_pre_gate_any_pass"] = summary.get("role_pilot_pre_gate_any_pass", False)
    summary["candidate_scope"] = "full KITTI 00/01/02/05"
    summary["outputs"]["role_specific_full_candidate_summary"] = rolem.rel(OUT / rolem.SUMMARY_JSON)
    summary["outputs"]["role_specific_full_candidate_report"] = rolem.rel(OUT / rolem.REPORT_MD)
    rolem.write_json(OUT / rolem.SUMMARY_JSON, summary)
    rows = rolem.read_csv(OUT / "role_specific_summary_rows.csv")
    rolem.write_text(OUT / rolem.REPORT_MD, full_report(summary, rows))
    return summary


def main() -> None:
    print(json.dumps(rolem.base.clean_json(build()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
