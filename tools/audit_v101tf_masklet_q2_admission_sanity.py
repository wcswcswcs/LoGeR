#!/usr/bin/env python3
"""Q2 sanity audit for lifecycle Stage-C masklet proxy signals.

This evaluates simple case-level selectors from the diagnostic masklet
visibility/2D observability rows.  It is proxy-only and cannot authorize Q2
true-stage admission because the clean HANDOFF target count is one.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"
VIS_ROWS = FINAL / "anchor_seed_lifecycle_stage_c_masklet_visibility_rows.csv"
TARGET_ROWS = ROOT / "trackT_drift_target_relabel/target_universe_v101.csv"
ROWS_OUT = FINAL / "masklet_q2_admission_sanity_case_rows.csv"
POLICY_OUT = FINAL / "masklet_q2_admission_sanity_policy_rows.csv"
SUMMARY_OUT = FINAL / "masklet_q2_admission_sanity_summary.json"
REPORT_OUT = FINAL / "masklet_q2_admission_sanity_report.md"

POS_TAX = "HANDOFF_SCALE_GAUGE_TARGET"
SAFE_TAX = "SAFE_GOOD"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def f(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def mean(values: list[Any]) -> float:
    finite = [f(value) for value in values if math.isfinite(f(value))]
    return sum(finite) / len(finite) if finite else math.nan


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    target = {row.get("case_id", ""): row for row in read_rows(TARGET_ROWS)}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(VIS_ROWS):
        if row.get("case_id"):
            grouped[row["case_id"]].append(row)

    case_rows: list[dict[str, Any]] = []
    for case_id, rows in sorted(grouped.items()):
        target_row = target.get(case_id, {})
        seed_count = len(rows)
        current_visible = [row for row in rows if bool_text(row.get("current_chunk_seed_visible"))]
        current_source_visible = [
            row
            for row in rows
            if bool_text(row.get("current_chunk_seed_visible"))
            and bool_text(row.get("source_chunk_seed_visible"))
        ]
        case_rows.append(
            {
                "case_id": case_id,
                "target_taxonomy": target_row.get("target_taxonomy", ""),
                "L3_handoff_transfer_penalty_proxy": target_row.get("L3_handoff_transfer_penalty_proxy", ""),
                "seed_count": seed_count,
                "current_visible_frac": len(current_visible) / seed_count if seed_count else math.nan,
                "current_source_visible_frac": len(current_source_visible) / seed_count if seed_count else math.nan,
                "bbox_center_span_mean": mean([row.get("current_chunk_bbox_center_span_px") for row in rows]),
                "bbox_area_cv_mean": mean([row.get("current_chunk_bbox_area_px_cv") for row in rows]),
                "area_ratio_max_mean": mean([row.get("current_chunk_area_ratio_max") for row in rows]),
                "area_ratio_std_mean": mean([row.get("current_chunk_area_ratio_std") for row in rows]),
                "claim_level": "masklet_q2_proxy_case_signal_no_action",
            }
        )
    write_rows(ROWS_OUT, case_rows)

    clean_rows = [row for row in case_rows if row.get("target_taxonomy") in {POS_TAX, SAFE_TAX}]
    positives = [row for row in clean_rows if row.get("target_taxonomy") == POS_TAX]
    safe = [row for row in clean_rows if row.get("target_taxonomy") == SAFE_TAX]
    policies = {
        "low_current_visible_frac": ("current_visible_frac", False),
        "low_current_source_visible_frac": ("current_source_visible_frac", False),
        "high_bbox_center_span_mean": ("bbox_center_span_mean", True),
        "high_bbox_area_cv_mean": ("bbox_area_cv_mean", True),
        "low_area_ratio_max_mean": ("area_ratio_max_mean", False),
        "high_area_ratio_std_mean": ("area_ratio_std_mean", True),
    }
    policy_rows: list[dict[str, Any]] = []
    for policy_name, (field, higher_bad) in policies.items():
        eval_rows = [row for row in clean_rows if math.isfinite(f(row.get(field)))]
        ranked = sorted(eval_rows, key=lambda row: f(row.get(field)), reverse=higher_bad)
        selected = {row["case_id"] for row in ranked[: len(positives)]}
        true_positive = [row["case_id"] for row in positives if row["case_id"] in selected]
        false_positive = [row["case_id"] for row in safe if row["case_id"] in selected]
        missed_positive = [row["case_id"] for row in positives if row["case_id"] not in selected]
        bad_recall = len(true_positive) / len(positives) if positives else math.nan
        good_fpr = len(false_positive) / len(safe) if safe else math.nan
        ba = (bad_recall + (1.0 - good_fpr)) * 0.5 if math.isfinite(bad_recall) and math.isfinite(good_fpr) else math.nan
        policy_rows.append(
            {
                "policy": policy_name,
                "score_field": field,
                "higher_is_bad": higher_bad,
                "positive_case_count": len(positives),
                "safe_good_count": len(safe),
                "selected_case_count": len(selected),
                "selected_cases": ";".join(sorted(selected)),
                "true_positive_cases": ";".join(sorted(true_positive)),
                "false_positive_cases": ";".join(sorted(false_positive)),
                "missed_positive_cases": ";".join(sorted(missed_positive)),
                "bad_recall": bad_recall,
                "good_FPR": good_fpr,
                "balanced_accuracy": ba,
                "claim_level": "masklet_q2_proxy_policy_no_action",
            }
        )
    write_rows(POLICY_OUT, policy_rows)

    best = max(policy_rows, key=lambda row: f(row.get("balanced_accuracy")), default={})
    summary = {
        "schema": "acl2_v101_masklet_q2_admission_sanity_v1",
        "diagnostic_only": True,
        "proxy_only": True,
        "runtime_action_allowed": False,
        "method_goal_achieved": False,
        "case_count": len(case_rows),
        "clean_eval_case_count": len(clean_rows),
        "positive_case_count": len(positives),
        "safe_good_count": len(safe),
        "policy_count": len(policy_rows),
        "best_policy": best.get("policy", ""),
        "best_policy_score_field": best.get("score_field", ""),
        "best_policy_balanced_accuracy": best.get("balanced_accuracy", ""),
        "best_policy_bad_recall": best.get("bad_recall", ""),
        "best_policy_good_FPR": best.get("good_FPR", ""),
        "best_policy_selected_cases": best.get("selected_cases", ""),
        "best_policy_true_positive_cases": best.get("true_positive_cases", ""),
        "q2_true_stage_pass": False,
        "q2_proxy_action_authorized": False,
        "blocker": (
            "Masklet proxy can be screened at case level, but clean positive count is one and the signal is 2D/proxy-only; "
            "this cannot authorize Q2 true-stage admission, M4, runtime, or full validation."
        ),
    }
    write_json(SUMMARY_OUT, summary)
    REPORT_OUT.write_text(
        "\n".join(
            [
                "# ACL2 v101 Masklet Q2 Admission Sanity",
                "",
                "This report evaluates simple Q2 proxy selectors from lifecycle Stage-C masklet visibility/2D observability rows. It is no-action.",
                "",
                "## Summary",
                "",
                f"- clean_eval_case_count: {summary['clean_eval_case_count']}",
                f"- positive_case_count: {summary['positive_case_count']}",
                f"- safe_good_count: {summary['safe_good_count']}",
                f"- best_policy: {summary['best_policy']}",
                f"- best_policy_balanced_accuracy: {summary['best_policy_balanced_accuracy']}",
                f"- best_policy_selected_cases: {summary['best_policy_selected_cases']}",
                f"- q2_true_stage_pass: {summary['q2_true_stage_pass']}",
                "",
                "## Blocker",
                "",
                summary["blocker"],
                "",
                "## Artifacts",
                "",
                f"- `{ROWS_OUT}`",
                f"- `{POLICY_OUT}`",
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
                "best_policy": summary["best_policy"],
                "best_policy_balanced_accuracy": summary["best_policy_balanced_accuracy"],
                "positive_case_count": summary["positive_case_count"],
                "safe_good_count": summary["safe_good_count"],
                "q2_true_stage_pass": summary["q2_true_stage_pass"],
                "runtime_action_allowed": summary["runtime_action_allowed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
