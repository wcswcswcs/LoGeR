#!/usr/bin/env python3
"""Audit combined masklet + lifecycle-geometry admission signals for v101.

This is a formalized version of the admission-level quick check.  It joins the
masklet Q2 proxy case rows with target28 lifecycle-aligned geometry smoke case
rows, evaluates simple combined policies under clean-safe and all-non-handoff
scopes, and records why the result is still no-action.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"

MASKLET_CASE_ROWS = FINAL / "masklet_q2_admission_sanity_case_rows.csv"
GEOMETRY_CASE_ROWS = FINAL / "stage_c_seed_geometry_smoke_target28_case_rows.csv"

CASE_ROWS_OUT = FINAL / "combined_masklet_geometry_admission_case_rows.csv"
POLICY_ROWS_OUT = FINAL / "combined_masklet_geometry_admission_policy_rows.csv"
FPFN_ROWS_OUT = FINAL / "combined_masklet_geometry_admission_false_positive_false_negative_rows.csv"
SUMMARY_OUT = FINAL / "combined_masklet_geometry_admission_summary.json"
REPORT_OUT = FINAL / "combined_masklet_geometry_admission_report.md"

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


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def frac(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else math.nan


def normalized(rows: list[dict[str, Any]], row: dict[str, Any], field: str) -> float:
    vals = [f(item.get(field)) for item in rows if math.isfinite(f(item.get(field)))]
    value = f(row.get(field))
    if not vals or not math.isfinite(value):
        return math.nan
    lo = min(vals)
    hi = max(vals)
    return (value - lo) / (hi - lo) if hi > lo else 0.0


def joined_case_rows() -> list[dict[str, Any]]:
    masklet = {row.get("case_id", ""): row for row in read_rows(MASKLET_CASE_ROWS)}
    geometry = {row.get("case_id", ""): row for row in read_rows(GEOMETRY_CASE_ROWS)}
    rows: list[dict[str, Any]] = []
    for case_id in sorted(set(masklet) & set(geometry)):
        m = masklet[case_id]
        g = geometry[case_id]
        row: dict[str, Any] = {
            "case_id": case_id,
            "target_taxonomy": m.get("target_taxonomy", g.get("target_taxonomy", "")),
            "L3_handoff_transfer_penalty_proxy": m.get(
                "L3_handoff_transfer_penalty_proxy", g.get("L3_handoff_transfer_penalty_proxy", "")
            ),
            "masklet_seed_count": m.get("seed_count", ""),
            "masklet_current_visible_frac": m.get("current_visible_frac", ""),
            "masklet_current_source_visible_frac": m.get("current_source_visible_frac", ""),
            "masklet_bbox_center_span_mean": m.get("bbox_center_span_mean", ""),
            "masklet_bbox_area_cv_mean": m.get("bbox_area_cv_mean", ""),
            "masklet_area_ratio_max_mean": m.get("area_ratio_max_mean", ""),
            "masklet_area_ratio_std_mean": m.get("area_ratio_std_mean", ""),
            "geometry_lifecycle_join_coverage": g.get("lifecycle_geometry_same_payload_join_coverage", ""),
            "geometry_edge_row_count": g.get("geometry_edge_row_count", ""),
            "geometry_abs_log_depth_ratio_mean": g.get("abs_log_depth_ratio_mean", ""),
            "geometry_abs_log_depth_ratio_std": g.get("abs_log_depth_ratio_std", ""),
            "geometry_query_inverse_depth_std": g.get("query_inverse_depth_std", ""),
            "geometry_query_world_spread_svd_ratio": g.get("query_world_spread_svd_ratio", ""),
            "geometry_cache_world_spread_svd_ratio": g.get("cache_world_spread_svd_ratio", ""),
            "geometry_local_scale_mode_count": g.get("local_scale_mode_count", ""),
            "geometry_local_scale_mode_entropy": g.get("local_scale_mode_entropy", ""),
            "claim_level": "combined_masklet_geometry_case_signal_no_action",
        }
        rows.append(row)

    for row in rows:
        bbox = normalized(rows, row, "masklet_bbox_center_span_mean")
        entropy = normalized(rows, row, "geometry_local_scale_mode_entropy")
        visible = f(row.get("masklet_current_visible_frac"), 0.0)
        row["score_entropy_bbox_product"] = entropy * bbox if math.isfinite(entropy) and math.isfinite(bbox) else math.nan
        row["score_entropy_plus_bbox"] = entropy + bbox if math.isfinite(entropy) and math.isfinite(bbox) else math.nan
        row["score_entropy_visible_bbox"] = (
            entropy * bbox * visible if math.isfinite(entropy) and math.isfinite(bbox) else math.nan
        )
        row["score_entropy_bbox_join_product"] = (
            entropy * bbox * f(row.get("geometry_lifecycle_join_coverage"), 0.0)
            if math.isfinite(entropy) and math.isfinite(bbox)
            else math.nan
        )
    return rows


def scoped_eval_rows(rows: list[dict[str, Any]], scope: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if scope == "clean_safe":
        eval_rows = [row for row in rows if row.get("target_taxonomy") in {POS_TAX, SAFE_TAX}]
        positives = [row for row in eval_rows if row.get("target_taxonomy") == POS_TAX]
        controls = [row for row in eval_rows if row.get("target_taxonomy") == SAFE_TAX]
        return eval_rows, positives, controls
    if scope == "all_non_handoff":
        eval_rows = [row for row in rows if row.get("target_taxonomy")]
        positives = [row for row in eval_rows if row.get("target_taxonomy") == POS_TAX]
        controls = [row for row in eval_rows if row.get("target_taxonomy") != POS_TAX]
        return eval_rows, positives, controls
    return [], [], []


def evaluate_policies(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    policies = [
        ("high_masklet_bbox_center_span_mean", "masklet_bbox_center_span_mean", "higher_bad"),
        ("high_geometry_local_scale_mode_entropy", "geometry_local_scale_mode_entropy", "higher_bad"),
        ("high_entropy_bbox_product", "score_entropy_bbox_product", "higher_bad"),
        ("high_entropy_plus_bbox", "score_entropy_plus_bbox", "higher_bad"),
        ("high_entropy_visible_bbox", "score_entropy_visible_bbox", "higher_bad"),
        ("high_entropy_bbox_join_product", "score_entropy_bbox_join_product", "higher_bad"),
        ("high_geometry_abs_log_depth_ratio_mean", "geometry_abs_log_depth_ratio_mean", "higher_bad"),
        ("high_geometry_query_inverse_depth_std", "geometry_query_inverse_depth_std", "higher_bad"),
        ("low_geometry_query_world_spread_svd_ratio", "geometry_query_world_spread_svd_ratio", "lower_bad"),
    ]
    policy_rows: list[dict[str, Any]] = []
    fpfn_rows: list[dict[str, Any]] = []
    best_by_scope: dict[str, dict[str, Any]] = {}
    for scope in ["clean_safe", "all_non_handoff"]:
        eval_rows, positives, controls = scoped_eval_rows(rows, scope)
        scope_policy_rows: list[dict[str, Any]] = []
        for policy_name, field, direction in policies:
            usable = [row for row in eval_rows if math.isfinite(f(row.get(field)))]
            selected_count = len(positives)
            if selected_count <= 0 or not usable:
                continue
            ranked = sorted(usable, key=lambda row: f(row.get(field)), reverse=(direction == "higher_bad"))
            selected = ranked[:selected_count]
            selected_cases = {row["case_id"] for row in selected}
            tp_rows = [row for row in positives if row["case_id"] in selected_cases]
            fp_rows = [row for row in controls if row["case_id"] in selected_cases]
            fn_rows = [row for row in positives if row["case_id"] not in selected_cases]
            bad_recall = frac(len(tp_rows), len(positives))
            control_fpr = frac(len(fp_rows), len(controls))
            balanced_accuracy = (
                0.5 * (bad_recall + (1.0 - control_fpr))
                if math.isfinite(bad_recall) and math.isfinite(control_fpr)
                else math.nan
            )
            row = {
                "eval_scope": scope,
                "policy_name": policy_name,
                "score_field": field,
                "direction": direction,
                "selected_count": selected_count,
                "selected_cases": ";".join(sorted(selected_cases)),
                "true_positive_cases": ";".join(sorted(item["case_id"] for item in tp_rows)),
                "false_positive_cases": ";".join(sorted(item["case_id"] for item in fp_rows)),
                "false_negative_cases": ";".join(sorted(item["case_id"] for item in fn_rows)),
                "bad_recall": bad_recall,
                "control_FPR": control_fpr,
                "balanced_accuracy": balanced_accuracy,
                "claim_level": "combined_masklet_geometry_policy_sanity_no_action",
            }
            policy_rows.append(row)
            scope_policy_rows.append(row)
            for item in tp_rows:
                fpfn_rows.append(
                    fpfn_row(scope, policy_name, field, "true_positive_handoff", item)
                )
            for item in fp_rows:
                fpfn_rows.append(
                    fpfn_row(scope, policy_name, field, "false_positive_control", item)
                )
            for item in fn_rows:
                fpfn_rows.append(
                    fpfn_row(scope, policy_name, field, "missed_handoff_positive", item)
                )
        best_by_scope[scope] = max(scope_policy_rows, key=lambda row: f(row.get("balanced_accuracy"), -1.0), default={})
    return policy_rows, fpfn_rows, best_by_scope.get("clean_safe", {}), best_by_scope.get("all_non_handoff", {})


def fpfn_row(scope: str, policy: str, field: str, row_kind: str, case_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_scope": scope,
        "policy_name": policy,
        "score_field": field,
        "row_kind": row_kind,
        "case_id": case_row.get("case_id", ""),
        "target_taxonomy": case_row.get("target_taxonomy", ""),
        "score_value": case_row.get(field, ""),
        "L3_handoff_transfer_penalty_proxy": case_row.get("L3_handoff_transfer_penalty_proxy", ""),
        "masklet_bbox_center_span_mean": case_row.get("masklet_bbox_center_span_mean", ""),
        "geometry_local_scale_mode_entropy": case_row.get("geometry_local_scale_mode_entropy", ""),
        "geometry_lifecycle_join_coverage": case_row.get("geometry_lifecycle_join_coverage", ""),
        "claim_level": "combined_masklet_geometry_fpfn_no_action",
    }


def main() -> None:
    case_rows = joined_case_rows()
    policy_rows, fpfn_rows, best_clean, best_all = evaluate_policies(case_rows)
    clean_eval_rows, clean_pos, clean_controls = scoped_eval_rows(case_rows, "clean_safe")
    all_eval_rows, all_pos, all_controls = scoped_eval_rows(case_rows, "all_non_handoff")
    selected_positive_sequence_coverage = len(clean_pos)
    required_positive_sequence_coverage = 3
    clean_metric_pass = (
        f(best_clean.get("bad_recall")) >= 0.65
        and f(best_clean.get("control_FPR")) <= 0.25
        and selected_positive_sequence_coverage >= required_positive_sequence_coverage
    )
    all_non_handoff_promotion_pass = (
        f(best_all.get("bad_recall")) >= 0.65
        and f(best_all.get("control_FPR")) <= 0.25
    )
    summary = {
        "schema": "acl2_v101_combined_masklet_geometry_admission_v1",
        "diagnostic_only": True,
        "case_count": len(case_rows),
        "clean_eval_case_count": len(clean_eval_rows),
        "positive_case_count": len(clean_pos),
        "safe_good_count": len(clean_controls),
        "all_non_handoff_eval_case_count": len(all_eval_rows),
        "all_non_handoff_control_count": len(all_controls),
        "policy_count": len(policy_rows),
        "clean_safe_best_policy": best_clean.get("policy_name", ""),
        "clean_safe_best_policy_balanced_accuracy": best_clean.get("balanced_accuracy", ""),
        "clean_safe_best_policy_selected_cases": best_clean.get("selected_cases", ""),
        "clean_safe_best_policy_true_positive_cases": best_clean.get("true_positive_cases", ""),
        "clean_safe_metric_pass_before_coverage": (
            f(best_clean.get("bad_recall")) >= 0.65 and f(best_clean.get("control_FPR")) <= 0.25
        ),
        "selected_positive_sequence_coverage": selected_positive_sequence_coverage,
        "required_positive_sequence_coverage": required_positive_sequence_coverage,
        "clean_safe_diagnostic_pass": clean_metric_pass,
        "all_non_handoff_best_policy": best_all.get("policy_name", ""),
        "all_non_handoff_best_policy_balanced_accuracy": best_all.get("balanced_accuracy", ""),
        "all_non_handoff_best_policy_selected_cases": best_all.get("selected_cases", ""),
        "all_non_handoff_best_policy_true_positive_cases": best_all.get("true_positive_cases", ""),
        "all_non_handoff_best_policy_false_positive_cases": best_all.get("false_positive_cases", ""),
        "all_non_handoff_promotion_pass": all_non_handoff_promotion_pass,
        "same_count_margin_available": False,
        "semantic_rotation_margin_available": False,
        "anchor_id_rotation_margin_available": False,
        "proxy_stage_signal_observed": f(best_clean.get("balanced_accuracy")) == 1.0,
        "q2_proxy_stage_pass": False,
        "q2_true_stage_pass": False,
        "runtime_action_allowed": False,
        "method_goal_achieved": False,
        "blocker": (
            "Combined masklet+geometry signals separate the single clean HANDOFF positive from SAFE_GOOD, "
            "but selected positive sequence coverage is 1<3 and all-non-handoff promotion fails."
        ),
    }
    write_rows(CASE_ROWS_OUT, case_rows)
    write_rows(POLICY_ROWS_OUT, policy_rows)
    write_rows(FPFN_ROWS_OUT, fpfn_rows)
    write_json(SUMMARY_OUT, summary)
    REPORT_OUT.write_text(
        "\n".join(
            [
                "# ACL2 v101 Combined Masklet + Geometry Admission",
                "",
                "This audit joins masklet 2D proxy rows with target28 lifecycle-aligned geometry smoke rows. It is no-action.",
                "",
                "## Summary",
                "",
                f"- case_count: {summary['case_count']}",
                f"- clean_eval_case_count: {summary['clean_eval_case_count']}",
                f"- positive_case_count: {summary['positive_case_count']}",
                f"- safe_good_count: {summary['safe_good_count']}",
                f"- clean_safe_best_policy: {summary['clean_safe_best_policy']}",
                f"- clean_safe_best_policy_balanced_accuracy: {summary['clean_safe_best_policy_balanced_accuracy']}",
                f"- selected_positive_sequence_coverage: {summary['selected_positive_sequence_coverage']}",
                f"- required_positive_sequence_coverage: {summary['required_positive_sequence_coverage']}",
                f"- all_non_handoff_best_policy: {summary['all_non_handoff_best_policy']}",
                f"- all_non_handoff_best_policy_balanced_accuracy: {summary['all_non_handoff_best_policy_balanced_accuracy']}",
                f"- q2_proxy_stage_pass: {summary['q2_proxy_stage_pass']}",
                f"- q2_true_stage_pass: {summary['q2_true_stage_pass']}",
                "",
                "## Blocker",
                "",
                summary["blocker"],
                "",
                "## Artifacts",
                "",
                f"- `{CASE_ROWS_OUT}`",
                f"- `{POLICY_ROWS_OUT}`",
                f"- `{FPFN_ROWS_OUT}`",
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
                "clean_safe_best_policy": summary["clean_safe_best_policy"],
                "clean_safe_best_policy_balanced_accuracy": summary["clean_safe_best_policy_balanced_accuracy"],
                "selected_positive_sequence_coverage": summary["selected_positive_sequence_coverage"],
                "required_positive_sequence_coverage": summary["required_positive_sequence_coverage"],
                "all_non_handoff_best_policy": summary["all_non_handoff_best_policy"],
                "all_non_handoff_best_policy_balanced_accuracy": summary[
                    "all_non_handoff_best_policy_balanced_accuracy"
                ],
                "q2_proxy_stage_pass": summary["q2_proxy_stage_pass"],
                "q2_true_stage_pass": summary["q2_true_stage_pass"],
                "runtime_action_allowed": summary["runtime_action_allowed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
