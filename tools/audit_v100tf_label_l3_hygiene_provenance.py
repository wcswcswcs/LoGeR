#!/usr/bin/env python3
"""Audit label/L3/hygiene provenance for ACL2 v100 diagnostics.

This is diagnostic-only.  It does not relabel cases and it does not promote a
filtered view to an action-ready gate.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
V98_ROOT = Path("results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control")
V99_ROOT = Path("results/acl2_v99tf_semantic_anchor_identity_lifecycle_multiroute_memory_control")
OUT = ROOT / "trackD4_read_current_support_provider"

V98_STAGE1 = V98_ROOT / "stage1_trackK_swa_v2_strict_stable_eligibility/case_universe_rows.csv"
V99_GRAPH = V99_ROOT / "trackN_semantic_anchor_identity_memory_graph/graph_case_rows.csv"
V100_L2 = ROOT / "trackL2_anchor_scale_observability/rows.csv"
CURRENT_CASES = OUT / "missed_positive_l3_case_rows.csv"
CURRENT_SUMMARY = OUT / "missed_positive_l3_consistency_summary.json"


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
            writer.writerow({key: row.get(key, "") for key in keys})


def by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("case_id", "")): row for row in rows if row.get("case_id")}


def split_cases(value: Any) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def pearson(xs: list[Any], ys: list[Any]) -> float:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        fx = f(x)
        fy = f(y)
        if math.isfinite(fx) and math.isfinite(fy):
            pairs.append((fx, fy))
    if len(pairs) < 2:
        return math.nan
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def selector_metrics(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    positives = [row for row in rows if row.get("case_label") != "good"]
    negatives = [row for row in rows if row.get("case_label") == "good"]
    selected = [row for row in rows if b(row.get("selected_by_best_composite"))]
    tp = [row for row in selected if row.get("case_label") != "good"]
    fp = [row for row in selected if row.get("case_label") == "good"]
    missed = [row for row in positives if not b(row.get("selected_by_best_composite"))]
    recall = len(tp) / len(positives) if positives else math.nan
    fpr = len(fp) / len(negatives) if negatives else math.nan
    seq_counts = Counter(str(row.get("seq", "")) for row in tp)
    max_frac = (max(seq_counts.values()) / len(tp)) if tp and seq_counts else math.nan
    corr = pearson(
        [1.0 if b(row.get("selected_by_best_composite")) else 0.0 for row in rows],
        [row.get("L3_handoff_transfer_penalty_proxy") for row in rows],
    )
    return {
        "view": name,
        "case_count": len(rows),
        "good_count": len(negatives),
        "non_good_count": len(positives),
        "selected_case_count": len(selected),
        "bad_recall": recall,
        "good_FPR": fpr,
        "balanced_accuracy": ((recall + (1.0 - fpr)) / 2.0) if math.isfinite(recall) and math.isfinite(fpr) else math.nan,
        "selector_corr_L3": corr,
        "selector_abs_corr_L3": abs(corr) if math.isfinite(corr) else math.nan,
        "true_positive_cases": ";".join(row.get("case_id", "") for row in tp),
        "false_positive_cases": ";".join(row.get("case_id", "") for row in fp),
        "missed_positive_cases": ";".join(row.get("case_id", "") for row in missed),
        "selected_positive_sequence_coverage": len(seq_counts),
        "selected_positive_sequence_max_frac": max_frac,
        "selected_positive_sequence_counts": dict(seq_counts),
    }


def main() -> None:
    v98_rows, v98_fields = read_rows(V98_STAGE1)
    v99_rows, v99_fields = read_rows(V99_GRAPH)
    v100_rows, v100_fields = read_rows(V100_L2)
    current_rows, current_fields = read_rows(CURRENT_CASES)
    current_summary = read_json(CURRENT_SUMMARY)

    v98 = by_case(v98_rows)
    v99 = by_case(v99_rows)
    v100 = by_case(v100_rows)
    current = by_case(current_rows)

    best = current_summary.get("best", {}) if isinstance(current_summary.get("best", {}), dict) else {}
    oracle = current_summary.get("oracle", {}) if isinstance(current_summary.get("oracle", {}), dict) else {}
    false_positive_cases = set(split_cases(best.get("false_positive_cases", "")))
    good_high_l3_cases = set(split_cases(oracle.get("good_high_l3_cases", "")))
    bad_low_l3_cases = set(split_cases(oracle.get("bad_low_l3_cases", "")))

    rows: list[dict[str, Any]] = []
    for case_id in sorted(v100):
        row98 = v98.get(case_id, {})
        row99 = v99.get(case_id, {})
        row100 = v100.get(case_id, {})
        rowcur = current.get(case_id, {})
        hygiene_include = row98.get("good_control_hygiene_include_for_repair", "")
        hygiene_excluded = row98.get("case_label") == "good" and hygiene_include != "" and not b(hygiene_include)
        selected = b(rowcur.get("selected_by_best_composite"))
        case_label = row100.get("case_label", row99.get("case_label", row98.get("case_label", "")))
        out_row = {
            "case_id": case_id,
            "seq": row100.get("seq", row99.get("seq", row98.get("seq", ""))),
            "case_label": case_label,
            "failure_type": row100.get("failure_type", row99.get("failure_type", row98.get("failure_type", ""))),
            "L3_handoff_transfer_penalty_proxy": row100.get(
                "L3_handoff_transfer_penalty_proxy",
                row99.get("L3_handoff_transfer_penalty_proxy", row98.get("L3_handoff_transfer_penalty_proxy", "")),
            ),
            "v98_present": bool(row98),
            "v98_case_label": row98.get("case_label", ""),
            "v98_L3": row98.get("L3_handoff_transfer_penalty_proxy", ""),
            "v98_failure_type": row98.get("failure_type", ""),
            "v98_universe_split": row98.get("universe_split", ""),
            "v98_good_control_hygiene_include_for_repair": hygiene_include,
            "v98_good_control_hygiene_status": row98.get("good_control_hygiene_status", ""),
            "v98_good_control_hygiene_l3_threshold": row98.get("good_control_hygiene_l3_threshold", ""),
            "v98_hygiene_excluded_good_control": hygiene_excluded,
            "v99_present": bool(row99),
            "v99_case_label": row99.get("case_label", ""),
            "v99_L3": row99.get("L3_handoff_transfer_penalty_proxy", ""),
            "v99_failure_type": row99.get("failure_type", ""),
            "v99_has_hygiene_field": "good_control_hygiene_include_for_repair" in v99_fields,
            "v100_present": bool(row100),
            "v100_case_label": row100.get("case_label", ""),
            "v100_L3": row100.get("L3_handoff_transfer_penalty_proxy", ""),
            "v100_failure_type": row100.get("failure_type", ""),
            "v100_has_hygiene_field": "good_control_hygiene_include_for_repair" in v100_fields,
            "label_l3_conflict": rowcur.get("label_l3_conflict", ""),
            "good_high_l3_from_current_oracle": case_id in good_high_l3_cases,
            "bad_low_l3_from_current_oracle": case_id in bad_low_l3_cases,
            "selected_by_best_composite": selected,
            "false_positive_in_best_composite": case_id in false_positive_cases,
            "selected_good_high_l3": selected and case_id in good_high_l3_cases,
            "selected_and_v98_hygiene_excluded": selected and hygiene_excluded,
            "false_positive_and_v98_hygiene_excluded": case_id in false_positive_cases and hygiene_excluded,
        }
        rows.append(out_row)

    hygiene_excluded_cases = [row["case_id"] for row in rows if b(row.get("v98_hygiene_excluded_good_control"))]
    selected_hygiene_excluded = [row["case_id"] for row in rows if b(row.get("selected_and_v98_hygiene_excluded"))]
    fp_hygiene_excluded = [row["case_id"] for row in rows if b(row.get("false_positive_and_v98_hygiene_excluded"))]
    good_high_l3_hygiene_excluded = [
        row["case_id"]
        for row in rows
        if b(row.get("good_high_l3_from_current_oracle")) and b(row.get("v98_hygiene_excluded_good_control"))
    ]

    all_metrics = selector_metrics(rows, "all_28_current_labels")
    hygiene_filtered_rows = [
        row for row in rows if not (row.get("case_label") == "good" and b(row.get("v98_hygiene_excluded_good_control")))
    ]
    hygiene_metrics = selector_metrics(hygiene_filtered_rows, "diagnostic_v98_hygiene_filtered")

    hygiene_field_lost_after_v98 = (
        "good_control_hygiene_include_for_repair" in v98_fields
        and "good_control_hygiene_include_for_repair" not in v99_fields
        and "good_control_hygiene_include_for_repair" not in v100_fields
    )
    blocker_text = (
        "v98 marked high-L3 good controls for hygiene repair exclusion, but current v99/v100 artifacts still drop that field; "
        "the current best composite's remaining false positive is one of those v98-excluded good controls."
        if hygiene_field_lost_after_v98
        else (
            "v98 marked high-L3 good controls for hygiene repair exclusion, and current v99/v100 artifacts now carry that field; "
            "the current best composite's remaining false positive is one of those v98-excluded good controls."
        )
    )

    summary = {
        "schema": "acl2_v100_label_l3_hygiene_provenance_v1",
        "status": "complete",
        "note": "Diagnostic-only provenance audit; does not relabel cases or authorize runtime action.",
        "inputs": {
            "v98_stage1_case_universe_rows": str(V98_STAGE1),
            "v99_graph_case_rows": str(V99_GRAPH),
            "v100_trackL2_rows": str(V100_L2),
            "current_missed_positive_case_rows": str(CURRENT_CASES),
            "current_missed_positive_summary": str(CURRENT_SUMMARY),
        },
        "field_availability": {
            "v98_case_count": len(v98_rows),
            "v99_case_count": len(v99_rows),
            "v100_l2_case_count": len(v100_rows),
            "current_case_count": len(current_rows),
            "v98_has_good_control_hygiene": "good_control_hygiene_include_for_repair" in v98_fields,
            "v99_has_good_control_hygiene": "good_control_hygiene_include_for_repair" in v99_fields,
            "v100_l2_has_good_control_hygiene": "good_control_hygiene_include_for_repair" in v100_fields,
            "current_has_selected_by_best_composite": "selected_by_best_composite" in current_fields,
            "hygiene_field_lost_after_v98": hygiene_field_lost_after_v98,
        },
        "v98_hygiene": {
            "excluded_good_control_count": len(hygiene_excluded_cases),
            "excluded_good_control_cases": ";".join(hygiene_excluded_cases),
            "good_high_l3_cases_from_current_oracle": ";".join(sorted(good_high_l3_cases)),
            "good_high_l3_excluded_by_v98_hygiene": ";".join(good_high_l3_hygiene_excluded),
        },
        "current_best_composite": {
            "cue_name": best.get("cue_name", ""),
            "false_positive_cases": best.get("false_positive_cases", ""),
            "missed_positive_cases": best.get("missed_positive_cases", ""),
            "selector_abs_corr_L3": best.get("selector_abs_corr_L3", math.nan),
            "gate_like": best.get("gate_like", False),
            "selected_hygiene_excluded_cases": ";".join(selected_hygiene_excluded),
            "false_positive_hygiene_excluded_cases": ";".join(fp_hygiene_excluded),
        },
        "diagnostic_selector_metrics": {
            "all_28_current_labels": all_metrics,
            "v98_hygiene_filtered": hygiene_metrics,
        },
        "conclusion": {
            "blocker": blocker_text,
            "no_go_reason": (
                "The hygiene-filtered view is a provenance diagnostic, not a full-universe v100 gate. "
                "Track S/M3/E4/runtime remain blocked unless documented gates pass on their required evidence."
            ),
        },
    }

    report_lines = [
        "# Label/L3/Hygiene Provenance Audit",
        "",
        "This audit is diagnostic-only. It does not relabel cases and does not authorize runtime action.",
        "",
        "## Field Availability",
        "",
        f"- v98 case rows: {len(v98_rows)}; hygiene field present: {summary['field_availability']['v98_has_good_control_hygiene']}",
        f"- v99 graph case rows: {len(v99_rows)}; hygiene field present: {summary['field_availability']['v99_has_good_control_hygiene']}",
        f"- v100 L2 rows: {len(v100_rows)}; hygiene field present: {summary['field_availability']['v100_l2_has_good_control_hygiene']}",
        f"- Hygiene field lost after v98: {summary['field_availability']['hygiene_field_lost_after_v98']}",
        "",
        "## v98 Hygiene Exclusions",
        "",
        f"- Excluded good controls: {summary['v98_hygiene']['excluded_good_control_cases'] or 'none'}",
        f"- Current oracle good-high-L3 cases: {summary['v98_hygiene']['good_high_l3_cases_from_current_oracle'] or 'none'}",
        f"- Good-high-L3 cases excluded by v98 hygiene: {summary['v98_hygiene']['good_high_l3_excluded_by_v98_hygiene'] or 'none'}",
        "",
        "## Current Best Composite Interaction",
        "",
        f"- Best cue: {summary['current_best_composite']['cue_name']}",
        f"- Current false positives: {summary['current_best_composite']['false_positive_cases'] or 'none'}",
        f"- False positives also excluded by v98 hygiene: {summary['current_best_composite']['false_positive_hygiene_excluded_cases'] or 'none'}",
        "",
        "## Selector Metrics",
        "",
        f"- All 28 current labels: BA={all_metrics['balanced_accuracy']}, recall={all_metrics['bad_recall']}, good_FPR={all_metrics['good_FPR']}, abs_corr_L3={all_metrics['selector_abs_corr_L3']}",
        f"- Diagnostic v98-hygiene-filtered view: case_count={hygiene_metrics['case_count']}, BA={hygiene_metrics['balanced_accuracy']}, recall={hygiene_metrics['bad_recall']}, good_FPR={hygiene_metrics['good_FPR']}, abs_corr_L3={hygiene_metrics['selector_abs_corr_L3']}",
        "",
        "## Conclusion",
        "",
        summary["conclusion"]["blocker"],
        "",
        summary["conclusion"]["no_go_reason"],
        "",
    ]

    write_rows(OUT / "label_l3_hygiene_provenance_rows.csv", rows)
    write_json(OUT / "label_l3_hygiene_provenance_summary.json", summary)
    write_text(OUT / "label_l3_hygiene_provenance_report.md", "\n".join(report_lines))


if __name__ == "__main__":
    main()
