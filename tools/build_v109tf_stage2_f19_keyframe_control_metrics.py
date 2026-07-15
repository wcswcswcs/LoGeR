#!/usr/bin/env python3
"""Build v109TF Stage2 F19 exact-count keyframe random control metrics."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v109tf_stage2_role_specific_pilot_metrics as rolem  # noqa: E402


base = rolem.base

RESULT_ROOT = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
OUT = RESULT_ROOT / "stage2_f19_keyframe_controls"
F19 = RESULT_ROOT / "stage2_role_specific_safety_candidates"
CONTROL_TOL = 0.005
CONTROL_MATCH_SEQUENCE_FAIL_COUNT = 2


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    rolem.write_csv(path, rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    rolem.write_json(path, data)


def write_text(path: Path, text: str) -> None:
    rolem.write_text(path, text)


def safe_float(value: Any) -> float:
    return rolem.safe_float(value)


def row_rel(row: dict[str, Any] | None) -> float:
    return rolem.row_rel(row)


def configure() -> None:
    rolem.OUT = OUT
    rolem.CONFIG_ROWS = OUT / "action_config_rows.csv"
    rolem.RUN_RESULTS = OUT / "run_results.csv"
    rolem.WORKSPACE = OUT / "workspace"
    rolem.SEQUENCES = ("00", "01", "02", "05")
    rolem.SUMMARY_ROW_SCHEMA = "acl2_v109tf_stage2_f19_keyframe_control_role_summary_row_v1"
    rolem.SUMMARY_SCHEMA = "acl2_v109tf_stage2_f19_keyframe_control_role_summary_v1"
    rolem.SUMMARY_JSON = "f19_keyframe_control_role_summary.json"
    rolem.REPORT_MD = "f19_keyframe_control_role_report.md"
    rolem.REPORT_TITLE = "# ACL2 v109TF Stage2 F19 Keyframe Control Role-Metric Report"
    rolem.SCOPE_NOTE = "full KITTI 00/01/02/05 F19 exact-selected-count keyframe random controls"
    rolem.GATE_SCOPE = "KITTI 00/01/02/05 F19 exact-count keyframe controls"
    rolem.INTERPRETATION_TEXT = (
        "These rows reuse the role-metric extraction path only for metric completeness. "
        "The causal decision is made in f19_keyframe_control_summary.json by comparing controls to F19."
    )


def f19_rows_by_seq() -> dict[str, dict[str, str]]:
    return {
        row["seq"]: row
        for row in read_csv(F19 / "full_metric_rows.csv")
        if row.get("policy_id") == "F19_dynamic_or_special_admitted_high_risk_else_weak_context"
    }


def control_comparison_rows(full_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    f19_by_seq = f19_rows_by_seq()
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        by_policy[row["policy_id"]].append(row)

    f19_rels = [row_rel(f19_by_seq.get(seq)) for seq in rolem.SEQUENCES]
    f19_median = base.median(f19_rels)
    rows: list[dict[str, Any]] = []
    best_by_seq: dict[str, dict[str, Any]] = {}
    for policy_id in sorted(by_policy):
        policy_rows = sorted(by_policy[policy_id], key=lambda row: row["seq"])
        rels = [row_rel(row) for row in policy_rows]
        gaps: list[float] = []
        vs_f19_ate: list[float] = []
        within_count = 0
        beats_count = 0
        for row in policy_rows:
            seq = row["seq"]
            ctrl_rel = row_rel(row)
            f19_rel = row_rel(f19_by_seq.get(seq))
            if math.isfinite(ctrl_rel) and math.isfinite(f19_rel):
                gap = ctrl_rel - f19_rel
                gaps.append(gap)
                if ctrl_rel >= f19_rel - CONTROL_TOL:
                    within_count += 1
            ctrl_ate = safe_float(row.get("full_ATE_sim3", "nan"))
            f19_ate = safe_float(f19_by_seq.get(seq, {}).get("full_ATE_sim3", "nan"))
            rel_vs_f19 = base.rel_improvement(f19_ate, ctrl_ate)
            vs_f19_ate.append(rel_vs_f19)
            if math.isfinite(rel_vs_f19) and rel_vs_f19 > 0.0:
                beats_count += 1
            current_best = best_by_seq.get(seq)
            if current_best is None or ctrl_rel > safe_float(current_best.get("control_rel", "nan")):
                best_by_seq[seq] = {
                    "seq": seq,
                    "policy_id": policy_id,
                    "control_rel": ctrl_rel,
                    "f19_rel": f19_rel,
                    "gap_vs_f19_rel": ctrl_rel - f19_rel if math.isfinite(ctrl_rel) and math.isfinite(f19_rel) else float("nan"),
                    "control_full_ATE_sim3": ctrl_ate,
                    "f19_full_ATE_sim3": f19_ate,
                    "control_full_ATE_relative_improvement_vs_f19": rel_vs_f19,
                }
        rows.append(
            {
                "schema": "acl2_v109tf_stage2_f19_keyframe_control_summary_row_v1",
                "policy_id": policy_id,
                "policy_family": policy_rows[0].get("policy_family", "") if policy_rows else "",
                "sequence_count": len(policy_rows),
                "median_full_rel_improvement": base.median(rels),
                "mean_full_rel_improvement": base.mean(rels),
                "num_seq_improved": sum(1 for value in rels if math.isfinite(value) and value > 0.0),
                "max_harm": base.max_rel_harm(rels),
                "f19_median_full_rel_improvement": f19_median,
                "median_gap_vs_f19_full_rel": base.median(gaps),
                "mean_gap_vs_f19_full_rel": base.mean(gaps),
                "control_within_0p005_of_f19_sequence_count": within_count,
                "control_beats_f19_ate_sequence_count": beats_count,
                "median_full_ATE_relative_improvement_vs_f19": base.median(vs_f19_ate),
                "mean_full_ATE_relative_improvement_vs_f19": base.mean(vs_f19_ate),
            }
        )

    best_same_seq_rows = [best_by_seq[seq] for seq in rolem.SEQUENCES if seq in best_by_seq]
    best_same_seq_match_count = sum(
        1
        for row in best_same_seq_rows
        if math.isfinite(safe_float(row.get("gap_vs_f19_rel", "nan")))
        and safe_float(row["gap_vs_f19_rel"]) >= -CONTROL_TOL
    )
    strongest = max(rows, key=lambda row: safe_float(row["median_full_rel_improvement"])) if rows else {}
    aggregate = {
        "f19_median_full_rel_improvement": f19_median,
        "strongest_control_policy_id": strongest.get("policy_id", ""),
        "strongest_control_median_full_rel_improvement": strongest.get("median_full_rel_improvement", ""),
        "strongest_control_median_gap_vs_f19_full_rel": strongest.get("median_gap_vs_f19_full_rel", ""),
        "best_same_seq_control_match_f19_count": best_same_seq_match_count,
        "best_same_seq_rows": best_same_seq_rows,
    }
    return rows, aggregate


def taxonomy(metric_complete: bool, all_action_fidelity: bool, aggregate: dict[str, Any]) -> tuple[str, bool, str]:
    if not metric_complete:
        return "F19_KEYFRAME_CONTROL_METRICS_NOT_COMPLETE", False, "f19_keyframe_control_metrics_not_complete"
    if not all_action_fidelity:
        return "F19_KEYFRAME_CONTROL_ACTION_FIDELITY_FAIL", False, "f19_keyframe_control_action_fidelity_fail"
    strongest = safe_float(aggregate.get("strongest_control_median_full_rel_improvement", "nan"))
    f19 = safe_float(aggregate.get("f19_median_full_rel_improvement", "nan"))
    match_count = int(aggregate.get("best_same_seq_control_match_f19_count", 0))
    if math.isfinite(strongest) and math.isfinite(f19) and strongest >= f19 - CONTROL_TOL:
        return (
            "F19_KEYFRAME_CONTROL_MATCHES_MEDIAN_EFFECT",
            False,
            "same_count_keyframe_control_matches_f19_median_effect",
        )
    if match_count >= CONTROL_MATCH_SEQUENCE_FAIL_COUNT:
        return (
            "F19_KEYFRAME_CONTROL_MATCHES_MULTI_SEQUENCE_EFFECT",
            False,
            "same_count_keyframe_control_matches_f19_on_multiple_sequences",
        )
    return "F19_KEYFRAME_CONTROL_DOES_NOT_MATCH_F19_EFFECT", True, ""


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ACL2 v109TF Stage2 F19 Keyframe Control Report",
        "",
        f"metric_complete: {summary['metric_complete']}",
        f"all_action_fidelity: {summary['all_action_fidelity']}",
        f"f19_keyframe_control_supports_f19_causality: {summary['f19_keyframe_control_supports_f19_causality']}",
        f"taxonomy: {summary['taxonomy']}",
        f"blocker: {summary['blocker']}",
        f"f19_median_full_rel_improvement: {summary['f19_median_full_rel_improvement']}",
        f"strongest_control_policy_id: {summary['strongest_control_policy_id']}",
        f"strongest_control_median_full_rel_improvement: {summary['strongest_control_median_full_rel_improvement']}",
        f"best_same_seq_control_match_f19_count: {summary['best_same_seq_control_match_f19_count']}",
        "",
        "## Control Summary",
        "",
    ]
    for row in rows:
        lines.append(
            "- {policy_id}: median_full_rel={median_full_rel_improvement} "
            "improved={num_seq_improved}/{sequence_count} max_harm={max_harm} "
            "median_gap_vs_F19={median_gap_vs_f19_full_rel} "
            "within_0p005_seq={control_within_0p005_of_f19_sequence_count}".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Pass here means the exact-count random keyframe controls do not match F19 under the registered tolerance. "
            "It is evidence against a simple count-only/random-keyframe explanation, but it does not by itself remove every possible schedule confound.",
        ]
    )
    return "\n".join(lines)


def build() -> dict[str, Any]:
    configure()
    role_summary = rolem.build()
    full_rows = read_csv(OUT / "full_metric_rows.csv")
    comparison_rows, aggregate = control_comparison_rows(full_rows)
    control_pass, supports, blocker = taxonomy(
        bool(role_summary.get("metric_complete")),
        bool(role_summary.get("all_action_fidelity")),
        aggregate,
    )
    summary = {
        "schema": "acl2_v109tf_stage2_f19_keyframe_control_summary_v1",
        "metric_complete": bool(role_summary.get("metric_complete")),
        "all_action_fidelity": bool(role_summary.get("all_action_fidelity")),
        "f19_keyframe_control_supports_f19_causality": supports,
        "taxonomy": control_pass,
        "blocker": blocker,
        "expected_run_worker_count": role_summary.get("expected_run_worker_count"),
        "observed_run_worker_count": role_summary.get("observed_run_worker_count"),
        "observed_evaluate_count": role_summary.get("observed_evaluate_count"),
        "observed_report_count": role_summary.get("observed_report_count"),
        "full_metric_row_count": role_summary.get("full_metric_row_count"),
        "rolling_metric_row_count": role_summary.get("rolling_metric_row_count"),
        "local_handoff_metric_row_count": role_summary.get("local_handoff_metric_row_count"),
        "action_fidelity_row_count": role_summary.get("action_fidelity_row_count"),
        "control_tolerance": CONTROL_TOL,
        "control_match_sequence_fail_count": CONTROL_MATCH_SEQUENCE_FAIL_COUNT,
        **aggregate,
        "outputs": {
            "full_metric_rows": rolem.rel(OUT / "full_metric_rows.csv"),
            "rolling_metric_rows": rolem.rel(OUT / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rolem.rel(OUT / "local_handoff_metric_rows.csv"),
            "action_fidelity_rows": rolem.rel(OUT / "action_fidelity_rows.csv"),
            "control_comparison_rows": rolem.rel(OUT / "f19_keyframe_control_summary_rows.csv"),
            "role_metric_summary": rolem.rel(OUT / rolem.SUMMARY_JSON),
            "summary": rolem.rel(OUT / "f19_keyframe_control_summary.json"),
            "report": rolem.rel(OUT / "f19_keyframe_control_report.md"),
        },
    }
    write_csv(OUT / "f19_keyframe_control_summary_rows.csv", comparison_rows)
    write_json(OUT / "f19_keyframe_control_summary.json", summary)
    write_text(OUT / "f19_keyframe_control_report.md", build_report(summary, comparison_rows))
    return summary


def main() -> None:
    print(json.dumps(base.clean_json(build()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
