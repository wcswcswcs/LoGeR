#!/usr/bin/env python3
"""Audit whether v102 can expand clean SWA handoff positives for Stage3.

This fail-forward audit follows the v102 plan guidance: when strict handoff
targets are too few, inspect drift-onset base cases and mark ambiguity instead
of forcing promotion.  It does not authorize Stage4 runtime/action work.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
STAGE1_ROWS = ROOT / "stage1_drift_source_autopsy/drift_source_taxonomy.csv"
STAGE1_SUMMARY = ROOT / "stage1_drift_source_autopsy/stage1_summary.json"
FULL_CONTROL_CASE_ROWS = ROOT / "stage3_semantic_oracle_upper_bound/stage3_full_control_semantic_rotation_case_rows.csv"
V101_TRACE_ROOT = Path(
    "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/stage_c_seed_bridge_geometry_smoke_target28"
)
V102_TRACE_EXTENSION_ROOT = (
    ROOT / "stage3_semantic_oracle_upper_bound/broader_drift_onset_trace_extension_traces"
)
OUT_ROWS = ROOT / "stage3_semantic_oracle_upper_bound/stage3_clean_handoff_candidate_expansion_rows.csv"
OUT_BROADER_ROWS = ROOT / "stage3_semantic_oracle_upper_bound/stage3_broader_drift_onset_candidate_rows.csv"
OUT_SUMMARY = ROOT / "stage3_semantic_oracle_upper_bound/stage3_clean_handoff_candidate_expansion_summary.json"
OUT_REPORT = ROOT / "stage3_semantic_oracle_upper_bound/stage3_clean_handoff_candidate_expansion_report.md"


def f(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(jsonable(value), sort_keys=True)
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["case_id"]: row for row in rows if row.get("case_id")}


def trace_sidecar_counts_at(root: Path, case_id: str) -> tuple[int, int]:
    run = root / case_id / "READ_NO_ACTION"
    trace = len(list((run / "swa_raw_transport_trace").glob("*.pt"))) if (run / "swa_raw_transport_trace").is_dir() else 0
    sidecar = len(list((run / "per_chunk_geometry").glob("chunk_*.pt"))) if (run / "per_chunk_geometry").is_dir() else 0
    return trace, sidecar


def trace_sidecar_counts(case_id: str) -> tuple[int, int]:
    return trace_sidecar_counts_at(V101_TRACE_ROOT, case_id)


def combined_trace_sidecar_counts(case_id: str) -> tuple[int, int, int, int, str]:
    v101_trace, v101_sidecar = trace_sidecar_counts_at(V101_TRACE_ROOT, case_id)
    v102_trace, v102_sidecar = trace_sidecar_counts_at(V102_TRACE_EXTENSION_ROOT, case_id)
    source = []
    if v101_trace and v101_sidecar >= 2:
        source.append("v101_target28")
    if v102_trace and v102_sidecar >= 2:
        source.append("v102_broader_extension")
    return v101_trace, v101_sidecar, v102_trace, v102_sidecar, ";".join(source)


def ambiguity_reasons(row: dict[str, str], *, require_swa: bool) -> list[str]:
    labels = str(row.get("drift_source_labels", ""))
    tax = str(row.get("target_taxonomy_v101", ""))
    reasons: list[str] = []
    if require_swa and "SWA_HANDOFF" not in labels:
        reasons.append("not_swa_handoff_label")
    if "UNRELIABLE_OVERLAP" in labels:
        reasons.append("unreliable_overlap")
    if "READ_LOCAL_SCALE" in labels:
        reasons.append("also_read_local")
    if "LABEL_L3_CONFLICT" in labels:
        reasons.append("label_l3_conflict")
    if str(row.get("label_original")) == "good":
        reasons.append("original_good_label")
    if tax != "HANDOFF_SCALE_GAUGE_TARGET":
        reasons.append(f"v101_taxonomy:{tax or 'missing'}")
    return reasons


def broader_promotion_role(row: dict[str, str], *, local_goodish: bool, high_l3: bool) -> str:
    labels = str(row.get("drift_source_labels", ""))
    tax = str(row.get("target_taxonomy_v101", ""))
    if not (local_goodish and high_l3):
        return "not_broader_drift_onset_candidate"
    if tax == "HANDOFF_SCALE_GAUGE_TARGET":
        return "strict_clean_handoff_validation_positive"
    if "SWA_HANDOFF" in labels:
        return "exploration_ambiguous_swa_handoff"
    return "exploration_non_handoff_drift_onset"


def main() -> int:
    stage1_rows = read_rows(STAGE1_ROWS)
    thresholds = read_json(STAGE1_SUMMARY).get("thresholds", {})
    full_control = by_case(read_rows(FULL_CONTROL_CASE_ROWS))
    l1_q70 = f(thresholds.get("L1_q70"))
    l2_q70 = f(thresholds.get("L2_cv_q70"))
    l3_q65 = f(thresholds.get("L3_q65"))
    l3_q75 = f(thresholds.get("L3_q75"))
    rows: list[dict[str, Any]] = []
    for row in stage1_rows:
        labels = str(row.get("drift_source_labels", ""))
        if "SWA_HANDOFF" not in labels:
            continue
        case_id = row["case_id"]
        l1 = f(row.get("L1_local_sim3_ate"))
        l2 = f(row.get("L2_intra_scale_cv"))
        l3 = f(row.get("L3_handoff_transfer_penalty_proxy"))
        tax = str(row.get("target_taxonomy_v101", ""))
        trace_count, sidecar_count = trace_sidecar_counts(case_id)
        local_goodish = l1 <= l1_q70 and l2 <= l2_q70
        high_l3 = l3 >= l3_q65
        strong_l3 = l3 >= l3_q75
        strict_clean = tax == "HANDOFF_SCALE_GAUGE_TARGET"
        ambiguity: list[str] = []
        if "UNRELIABLE_OVERLAP" in labels:
            ambiguity.append("unreliable_overlap")
        if "READ_LOCAL_SCALE" in labels:
            ambiguity.append("also_read_local")
        if "LABEL_L3_CONFLICT" in labels:
            ambiguity.append("label_l3_conflict")
        if str(row.get("label_original")) == "good":
            ambiguity.append("original_good_label")
        if tax != "HANDOFF_SCALE_GAUGE_TARGET":
            ambiguity.append(f"v101_taxonomy:{tax or 'missing'}")
        fc = full_control.get(case_id, {})
        rows.append(
            {
                "case_id": case_id,
                "seq": row.get("seq", ""),
                "label_original": row.get("label_original", ""),
                "target_taxonomy_v101": tax,
                "drift_source_labels": labels,
                "primary_drift_source": row.get("primary_drift_source", ""),
                "L1_local_sim3_ate": l1,
                "L2_intra_scale_cv": l2,
                "L3_handoff_transfer_penalty_proxy": l3,
                "L3_adjacent_log_scale_jump": row.get("L3_adjacent_log_scale_jump", ""),
                "local_goodish_under_q70": local_goodish,
                "high_l3_under_q65": high_l3,
                "strong_l3_under_q75": strong_l3,
                "clean_handoff_like_candidate": local_goodish and high_l3,
                "strict_clean_handoff_positive": strict_clean,
                "usable_for_exploration_only": (local_goodish and high_l3 and not strict_clean),
                "ambiguity_reasons": ";".join(ambiguity),
                "v101_trace_count": trace_count,
                "v101_sidecar_count": sidecar_count,
                "trace_sidecar_ready": bool(trace_count and sidecar_count >= 2),
                "semantic_unreliable_anchor_frac": fc.get("semantic_unreliable_anchor_frac", ""),
                "unreliable_or_low_observable_frac": fc.get("unreliable_or_low_observable_frac", ""),
                "O_scale_mean": fc.get("O_scale_mean", ""),
                "stable_observable_frac_q50": fc.get("stable_observable_frac_q50", ""),
                "claim_level": "v102_clean_handoff_candidate_expansion_diagnostic_no_action",
            }
        )
    clean_like = [row for row in rows if row["clean_handoff_like_candidate"]]
    strict = [row for row in rows if row["strict_clean_handoff_positive"]]
    exploration_only = [row for row in rows if row["usable_for_exploration_only"]]
    ready_clean_like = [row for row in clean_like if row["trace_sidecar_ready"]]
    selected_by_unreliable = [
        row
        for row in clean_like
        if f(row.get("semantic_unreliable_anchor_frac")) >= 0.948051948051948
    ]

    broader_rows: list[dict[str, Any]] = []
    for row in stage1_rows:
        case_id = row["case_id"]
        l1 = f(row.get("L1_local_sim3_ate"))
        l2 = f(row.get("L2_intra_scale_cv"))
        l3 = f(row.get("L3_handoff_transfer_penalty_proxy"))
        local_goodish = l1 <= l1_q70 and l2 <= l2_q70
        high_l3 = l3 >= l3_q65
        if not (local_goodish and high_l3):
            continue
        v101_trace_count, v101_sidecar_count, v102_trace_count, v102_sidecar_count, trace_source = combined_trace_sidecar_counts(case_id)
        fc = full_control.get(case_id, {})
        role = broader_promotion_role(row, local_goodish=local_goodish, high_l3=high_l3)
        broader_rows.append(
            {
                "case_id": case_id,
                "seq": row.get("seq", ""),
                "label_original": row.get("label_original", ""),
                "target_taxonomy_v101": row.get("target_taxonomy_v101", ""),
                "drift_source_labels": row.get("drift_source_labels", ""),
                "primary_drift_source": row.get("primary_drift_source", ""),
                "L1_local_sim3_ate": l1,
                "L2_intra_scale_cv": l2,
                "L3_handoff_transfer_penalty_proxy": l3,
                "L3_adjacent_log_scale_jump": row.get("L3_adjacent_log_scale_jump", ""),
                "local_goodish_under_q70": local_goodish,
                "high_l3_under_q65": high_l3,
                "has_swa_handoff_label": "SWA_HANDOFF" in str(row.get("drift_source_labels", "")),
                "strict_clean_handoff_positive": row.get("target_taxonomy_v101", "") == "HANDOFF_SCALE_GAUGE_TARGET",
                "promotion_role": role,
                "usable_for_strict_stage3_promotion": role == "strict_clean_handoff_validation_positive",
                "usable_for_exploration_only": role != "strict_clean_handoff_validation_positive",
                "ambiguity_reasons": ";".join(ambiguity_reasons(row, require_swa=True)),
                "v101_trace_count": v101_trace_count,
                "v101_sidecar_count": v101_sidecar_count,
                "v102_trace_extension_count": v102_trace_count,
                "v102_trace_extension_sidecar_count": v102_sidecar_count,
                "trace_sidecar_ready": bool(trace_source),
                "trace_sidecar_source": trace_source,
                "full_control_case_row_available": bool(fc),
                "semantic_unreliable_anchor_frac": fc.get("semantic_unreliable_anchor_frac", ""),
                "unreliable_or_low_observable_frac": fc.get("unreliable_or_low_observable_frac", ""),
                "O_scale_mean": fc.get("O_scale_mean", ""),
                "stable_observable_frac_q50": fc.get("stable_observable_frac_q50", ""),
                "claim_level": "v102_broader_drift_onset_exploration_diagnostic_no_action",
            }
        )
    broader_strict = [row for row in broader_rows if row["usable_for_strict_stage3_promotion"]]
    broader_exploration = [row for row in broader_rows if row["usable_for_exploration_only"]]
    broader_missing_full_control = [row for row in broader_rows if not row["full_control_case_row_available"]]
    broader_non_swa = [row for row in broader_rows if not row["has_swa_handoff_label"]]
    broader_v102_extension_ready = [
        row for row in broader_rows if "v102_broader_extension" in str(row.get("trace_sidecar_source", ""))
    ]
    broader_selected_by_unreliable = [
        row
        for row in broader_rows
        if f(row.get("semantic_unreliable_anchor_frac")) >= 0.948051948051948
    ]
    summary = {
        "schema": "acl2_v102_stage3_clean_handoff_candidate_expansion_v1",
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "swa_handoff_candidate_count": len(rows),
        "swa_handoff_sequence_count": len({row["seq"] for row in rows}),
        "clean_handoff_like_candidate_count": len(clean_like),
        "clean_handoff_like_sequence_count": len({row["seq"] for row in clean_like}),
        "clean_handoff_like_trace_sidecar_ready_count": len(ready_clean_like),
        "strict_clean_handoff_positive_count": len(strict),
        "strict_clean_handoff_positive_cases": ";".join(row["case_id"] for row in strict),
        "exploration_only_candidate_count": len(exploration_only),
        "exploration_only_candidate_cases": ";".join(row["case_id"] for row in exploration_only),
        "best_full_control_unreliable_semantic_threshold": 0.948051948051948,
        "clean_like_selected_by_best_unreliable_semantic_count": len(selected_by_unreliable),
        "clean_like_selected_by_best_unreliable_semantic_cases": ";".join(row["case_id"] for row in selected_by_unreliable),
        "broader_drift_onset_definition": "Stage1 rows with L1<=L1_q70, L2_cv<=L2_cv_q70, and L3>=L3_q65; includes non-SWA drift-onset cases but marks ambiguity.",
        "broader_local_goodish_high_l3_count": len(broader_rows),
        "broader_local_goodish_high_l3_sequence_count": len({row["seq"] for row in broader_rows}),
        "broader_strict_clean_handoff_positive_count": len(broader_strict),
        "broader_strict_clean_handoff_positive_cases": ";".join(row["case_id"] for row in broader_strict),
        "broader_exploration_only_count": len(broader_exploration),
        "broader_exploration_only_cases": ";".join(row["case_id"] for row in broader_exploration),
        "broader_non_swa_drift_onset_count": len(broader_non_swa),
        "broader_non_swa_drift_onset_cases": ";".join(row["case_id"] for row in broader_non_swa),
        "broader_full_control_case_row_available_count": len(broader_rows) - len(broader_missing_full_control),
        "broader_missing_full_control_case_row_count": len(broader_missing_full_control),
        "broader_missing_full_control_cases": ";".join(row["case_id"] for row in broader_missing_full_control),
        "broader_trace_sidecar_ready_count": len([row for row in broader_rows if row["trace_sidecar_ready"]]),
        "broader_v102_trace_extension_ready_count": len(broader_v102_extension_ready),
        "broader_v102_trace_extension_ready_cases": ";".join(row["case_id"] for row in broader_v102_extension_ready),
        "broader_selected_by_best_unreliable_semantic_count": len(broader_selected_by_unreliable),
        "broader_selected_by_best_unreliable_semantic_cases": ";".join(row["case_id"] for row in broader_selected_by_unreliable),
        "strict_stage3_coverage_repaired": False,
        "blocker": (
            "Stage1 has SWA clean-handoff-like and broader drift-onset exploration candidates, "
            "but only one strict HANDOFF_SCALE_GAUGE_TARGET remains.  Broader candidates are ambiguous/non-SWA/good-label contaminated "
            "or lack full-control support, so they cannot repair strict Stage3 promotion."
        ),
        "outputs": {
            "rows": OUT_ROWS.as_posix(),
            "broader_rows": OUT_BROADER_ROWS.as_posix(),
            "report": OUT_REPORT.as_posix(),
        },
    }
    write_rows(OUT_ROWS, rows)
    write_rows(OUT_BROADER_ROWS, broader_rows)
    write_json(OUT_SUMMARY, summary)
    write_text(
        OUT_REPORT,
        "\n".join(
            [
                "# Stage3 Clean Handoff Candidate Expansion",
                "",
                f"- swa_handoff_candidate_count: {summary['swa_handoff_candidate_count']}",
                f"- clean_handoff_like_candidate_count: {summary['clean_handoff_like_candidate_count']}",
                f"- clean_handoff_like_sequence_count: {summary['clean_handoff_like_sequence_count']}",
                f"- clean_handoff_like_trace_sidecar_ready_count: {summary['clean_handoff_like_trace_sidecar_ready_count']}",
                f"- strict_clean_handoff_positive_count: {summary['strict_clean_handoff_positive_count']}",
                f"- exploration_only_candidate_count: {summary['exploration_only_candidate_count']}",
                f"- clean_like_selected_by_best_unreliable_semantic_count: {summary['clean_like_selected_by_best_unreliable_semantic_count']}",
                f"- broader_local_goodish_high_l3_count: {summary['broader_local_goodish_high_l3_count']}",
                f"- broader_local_goodish_high_l3_sequence_count: {summary['broader_local_goodish_high_l3_sequence_count']}",
                f"- broader_strict_clean_handoff_positive_count: {summary['broader_strict_clean_handoff_positive_count']}",
                f"- broader_exploration_only_count: {summary['broader_exploration_only_count']}",
                f"- broader_non_swa_drift_onset_count: {summary['broader_non_swa_drift_onset_count']}",
                f"- broader_missing_full_control_case_row_count: {summary['broader_missing_full_control_case_row_count']}",
                f"- broader_missing_full_control_cases: {summary['broader_missing_full_control_cases']}",
                f"- broader_trace_sidecar_ready_count: {summary['broader_trace_sidecar_ready_count']}",
                f"- broader_v102_trace_extension_ready_cases: {summary['broader_v102_trace_extension_ready_cases']}",
                "",
                "Conclusion:",
                "",
                summary["blocker"],
                "",
                "Broader drift-onset candidates:",
                "",
                "| case_id | seq | role | target_taxonomy_v101 | primary_drift_source | ambiguity_reasons | full_control_case_row_available | trace_sidecar_ready | trace_sidecar_source |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                *[
                    (
                        f"| {row['case_id']} | {row['seq']} | {row['promotion_role']} | "
                        f"{row['target_taxonomy_v101']} | {row['primary_drift_source']} | "
                        f"{row['ambiguity_reasons']} | {row['full_control_case_row_available']} | {row['trace_sidecar_ready']} | "
                        f"{row['trace_sidecar_source']} |"
                    )
                    for row in broader_rows
                ],
            ]
        ),
    )
    print(json.dumps(jsonable(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
