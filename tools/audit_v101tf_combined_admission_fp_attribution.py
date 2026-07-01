#!/usr/bin/env python3
"""Attribute combined-admission false positives by target taxonomy.

This is a no-action follow-up to the combined masklet+geometry admission audit.
It explains which offline target-taxonomy buckets produce false positives under
the candidate policies.  It does not turn target taxonomy into a runtime gate.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"

FPFN_ROWS = FINAL / "combined_masklet_geometry_admission_false_positive_false_negative_rows.csv"
CASE_ROWS = FINAL / "combined_masklet_geometry_admission_case_rows.csv"
SUMMARY_IN = FINAL / "combined_masklet_geometry_admission_summary.json"
HISTORICAL_MINING = FINAL / "historical_target_universe_mining_summary.json"

ATTR_ROWS_OUT = FINAL / "combined_admission_false_positive_attribution_rows.csv"
SUMMARY_OUT = FINAL / "combined_admission_false_positive_attribution_summary.json"
REPORT_OUT = FINAL / "combined_admission_false_positive_attribution_report.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}"}
    return payload if isinstance(payload, dict) else {"value": payload}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def main() -> None:
    fpfn = read_rows(FPFN_ROWS)
    case_by_id = {row.get("case_id", ""): row for row in read_rows(CASE_ROWS)}
    combined_summary = read_json(SUMMARY_IN)
    historical = read_json(HISTORICAL_MINING)

    false_positive_rows = [
        row for row in fpfn if row.get("eval_scope") == "all_non_handoff" and row.get("row_kind") == "false_positive_control"
    ]
    missed_positive_rows = [
        row for row in fpfn if row.get("eval_scope") == "all_non_handoff" and row.get("row_kind") == "missed_handoff_positive"
    ]
    fp_by_taxonomy = Counter(row.get("target_taxonomy", "") for row in false_positive_rows)
    fp_by_case = Counter(row.get("case_id", "") for row in false_positive_rows)
    policies_by_case: dict[str, list[str]] = defaultdict(list)
    for row in false_positive_rows:
        policies_by_case[row.get("case_id", "")].append(row.get("policy_name", ""))

    attr_rows: list[dict[str, Any]] = []
    for case_id, count in sorted(fp_by_case.items(), key=lambda item: (-item[1], item[0])):
        case = case_by_id.get(case_id, {})
        policies = sorted(set(policies_by_case.get(case_id, [])))
        attr_rows.append(
            {
                "case_id": case_id,
                "target_taxonomy": case.get("target_taxonomy", ""),
                "false_positive_policy_count": count,
                "false_positive_policies": ";".join(policies),
                "masklet_bbox_center_span_mean": case.get("masklet_bbox_center_span_mean", ""),
                "geometry_local_scale_mode_entropy": case.get("geometry_local_scale_mode_entropy", ""),
                "geometry_abs_log_depth_ratio_mean": case.get("geometry_abs_log_depth_ratio_mean", ""),
                "geometry_query_world_spread_svd_ratio": case.get("geometry_query_world_spread_svd_ratio", ""),
                "geometry_lifecycle_join_coverage": case.get("geometry_lifecycle_join_coverage", ""),
                "L3_handoff_transfer_penalty_proxy": case.get("L3_handoff_transfer_penalty_proxy", ""),
                "claim_level": "combined_admission_false_positive_attribution_no_action",
            }
        )

    clean_handoff_count = int(historical.get("clean_handoff_candidate_count", 0) or 0)
    required_positive_sequence_coverage = int(combined_summary.get("required_positive_sequence_coverage", 3) or 3)
    summary = {
        "schema": "acl2_v101_combined_admission_false_positive_attribution_v1",
        "diagnostic_only": True,
        "all_non_handoff_false_positive_row_count": len(false_positive_rows),
        "all_non_handoff_missed_positive_row_count": len(missed_positive_rows),
        "false_positive_case_count": len(fp_by_case),
        "false_positive_taxonomy_counts": dict(sorted(fp_by_taxonomy.items())),
        "false_positive_cases": ";".join(sorted(fp_by_case)),
        "most_common_false_positive_case": fp_by_case.most_common(1)[0][0] if fp_by_case else "",
        "most_common_false_positive_case_policy_count": fp_by_case.most_common(1)[0][1] if fp_by_case else 0,
        "clean_handoff_candidate_count": clean_handoff_count,
        "required_positive_sequence_coverage": required_positive_sequence_coverage,
        "historical_mined_new_clean_universe_available": historical.get(
            "historical_mined_new_clean_universe_available", False
        ),
        "taxonomy_split_explains_false_positives": bool(fp_by_taxonomy),
        "taxonomy_split_runtime_action_ready": False,
        "q2_proxy_stage_pass": False,
        "q2_true_stage_pass": False,
        "runtime_action_allowed": False,
        "method_goal_achieved": False,
        "blocker": (
            "False positives are explainable by offline target taxonomy, but only one clean HANDOFF positive exists "
            "and target taxonomy is not a runtime admission feature."
        ),
    }
    write_rows(ATTR_ROWS_OUT, attr_rows)
    write_json(SUMMARY_OUT, summary)
    REPORT_OUT.write_text(
        "\n".join(
            [
                "# ACL2 v101 Combined Admission False Positive Attribution",
                "",
                "This report attributes all-non-handoff false positives from the combined admission audit.",
                "",
                "## Summary",
                "",
                f"- false_positive_case_count: {summary['false_positive_case_count']}",
                f"- false_positive_taxonomy_counts: {summary['false_positive_taxonomy_counts']}",
                f"- clean_handoff_candidate_count: {summary['clean_handoff_candidate_count']}",
                f"- required_positive_sequence_coverage: {summary['required_positive_sequence_coverage']}",
                f"- historical_mined_new_clean_universe_available: {summary['historical_mined_new_clean_universe_available']}",
                f"- taxonomy_split_runtime_action_ready: {summary['taxonomy_split_runtime_action_ready']}",
                f"- q2_true_stage_pass: {summary['q2_true_stage_pass']}",
                "",
                "## Blocker",
                "",
                summary["blocker"],
                "",
                "## Artifacts",
                "",
                f"- `{ATTR_ROWS_OUT}`",
                f"- `{SUMMARY_OUT}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "false_positive_case_count": summary["false_positive_case_count"],
                "false_positive_taxonomy_counts": summary["false_positive_taxonomy_counts"],
                "clean_handoff_candidate_count": summary["clean_handoff_candidate_count"],
                "taxonomy_split_runtime_action_ready": summary["taxonomy_split_runtime_action_ready"],
                "q2_true_stage_pass": summary["q2_true_stage_pass"],
                "runtime_action_allowed": summary["runtime_action_allowed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
