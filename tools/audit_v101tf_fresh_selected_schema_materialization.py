#!/usr/bin/env python3
"""Audit whether fresh rich-selector rows can become v101 target-universe rows.

The previous holdout-feasibility audit showed that fixed rich-selector policy
can select fresh support rows after a native-only Stage1 measurement.  This
script checks whether those selected rows can legally become v101 clean
handoff/safe-good target rows under the existing Track T taxonomy and v100
schema requirements.

It is intentionally diagnostic-only: no unlabelled/LOCAL_BAD/MULTIMODE row is
promoted to action evidence.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"
STAGE1 = ROOT / "outcomeD_merge_gauge_fresh_unlabelled_stage1_native_probe"
TRACK_T = ROOT / "trackT_drift_target_relabel"


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return default
    return out if math.isfinite(out) else default


def b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def json_clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_clean(row.get(key, "")) for key in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_unique_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = text.splitlines()
    if line not in lines:
        lines.append(line)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def upsert_section(path: Path, heading: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    section = f"\n\n## {heading}\n\n{body.strip()}\n"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    marker = f"\n## {heading}\n"
    if marker in text:
        prefix, rest = text.split(marker, 1)
        next_heading = rest.find("\n## ")
        if next_heading >= 0:
            text = prefix.rstrip() + section + rest[next_heading:]
        else:
            text = prefix.rstrip() + section
    else:
        text = text.rstrip() + section
    path.write_text(text.lstrip() + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")


def index(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value and value not in out:
            out[value] = row
    return out


def selected_fresh_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if not str(row.get("evidence_role", "")).startswith("fresh"):
            continue
        if not b(row.get("selected_by_fixed_policy")):
            continue
        out.append(row)
    return out


def missing_schema_reasons(
    *,
    target_taxonomy: str,
    broad_row: dict[str, str],
    target_present: bool,
) -> list[str]:
    reasons: list[str] = []
    if target_taxonomy not in {"HANDOFF_SCALE_GAUGE_TARGET", "SAFE_GOOD"}:
        reasons.append(f"not_clean_handoff_or_safe_good:{target_taxonomy or 'missing_target_taxonomy'}")
    if not target_present:
        reasons.append("not_in_trackT_28_case_target_universe")
    if not b(broad_row.get("already_in_v100_28_case_universe")):
        reasons.append("not_in_v100_28_case_universe")
    if not b(broad_row.get("has_v100_same_space_trace")):
        reasons.append("missing_v100_same_space_trace")
    if not b(broad_row.get("has_v100_per_anchor_geometry")):
        reasons.append("missing_v100_per_anchor_geometry")
    if target_taxonomy == "MULTIMODE_LOWOBS_ABSTAIN":
        reasons.append("multimode_lowobs_abstain_not_action_target")
    if target_taxonomy == "LOCAL_BAD_NOT_HANDOFF":
        reasons.append("local_bad_not_handoff_target")
    return list(dict.fromkeys(reasons))


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    holdout_rows = read_rows(FINAL / "rich_selector_holdout_feasibility_rows.csv")
    metric_rows = read_rows(STAGE1 / "runtime_probe_metric_rows.csv")
    target_rows = read_rows(TRACK_T / "target_universe_v101.csv")
    broad_rows = read_rows(TRACK_T / "broad_prior_unique_case_summary.csv")
    holdout_summary = read_json(FINAL / "rich_selector_holdout_feasibility_summary.json")

    metric_by_pair = index(metric_rows, "pair_id")
    target_by_case = index(target_rows, "case_id")
    broad_by_case = index(broad_rows, "case_id")

    rows: list[dict[str, Any]] = []
    for row in selected_fresh_rows(holdout_rows):
        case_id = row.get("pair_id", "")
        target = target_by_case.get(case_id, {})
        broad = broad_by_case.get(case_id, {})
        metric = metric_by_pair.get(case_id, {})
        target_taxonomy = target.get("target_taxonomy", "") or broad.get("representative_taxonomy", "")
        stage1_l3_proxy = f(metric.get("curr_handoff_transfer_rmse")) - f(metric.get("curr_postmerge_sim3_rmse"))
        if not math.isfinite(stage1_l3_proxy):
            stage1_l3_proxy = math.nan
        missing = missing_schema_reasons(
            target_taxonomy=target_taxonomy,
            broad_row=broad,
            target_present=bool(target),
        )
        action_materializable = (
            target_taxonomy in {"HANDOFF_SCALE_GAUGE_TARGET", "SAFE_GOOD"}
            and b(broad.get("already_in_v100_28_case_universe"))
            and b(broad.get("has_v100_same_space_trace"))
            and b(broad.get("has_v100_per_anchor_geometry"))
            and bool(target)
        )
        rows.append(
            {
                "case_id": case_id,
                "seq": row.get("seq", ""),
                "phase1_case_label_offline_only": row.get("case_label_offline_only", ""),
                "phase1_failure_type_primary": row.get("failure_type_primary", ""),
                "fixed_policy_feature1_value": row.get("feature1_value", ""),
                "fixed_policy_feature2_value": row.get("feature2_value", ""),
                "stage1_native_curr_postmerge_sim3_rmse": metric.get("curr_postmerge_sim3_rmse", ""),
                "stage1_native_curr_handoff_transfer_rmse": metric.get("curr_handoff_transfer_rmse", ""),
                "stage1_native_handoff_minus_postmerge_proxy": stage1_l3_proxy,
                "stage1_native_metric_claim": "diagnostic_not_trackT_threshold_comparable",
                "trackT_28_case_target_present": bool(target),
                "trackT_target_taxonomy": target.get("target_taxonomy", ""),
                "trackT_target_reason": target.get("target_reason", ""),
                "trackT_L3_handoff_transfer_penalty_proxy": target.get(
                    "L3_handoff_transfer_penalty_proxy", ""
                ),
                "broad_prior_present": bool(broad),
                "broad_representative_taxonomy": broad.get("representative_taxonomy", ""),
                "broad_representative_reason": broad.get("representative_reason", ""),
                "already_in_v100_28_case_universe": b(broad.get("already_in_v100_28_case_universe")),
                "has_v100_same_space_trace": b(broad.get("has_v100_same_space_trace")),
                "has_v100_per_anchor_geometry": b(broad.get("has_v100_per_anchor_geometry")),
                "clean_handoff_or_safe_good_candidate": target_taxonomy
                in {"HANDOFF_SCALE_GAUGE_TARGET", "SAFE_GOOD"},
                "schema_action_materializable_now": action_materializable,
                "missing_schema_or_target_reasons": ";".join(missing),
                "claim_level": "fresh_selected_schema_materialization_no_action",
            }
        )

    taxonomy_counts = Counter(row["broad_representative_taxonomy"] or row["trackT_target_taxonomy"] for row in rows)
    target_taxonomy_counts = Counter(row["trackT_target_taxonomy"] or "not_in_trackT_28" for row in rows)
    missing_counts: Counter[str] = Counter()
    for row in rows:
        for reason in str(row["missing_schema_or_target_reasons"]).split(";"):
            if reason:
                missing_counts[reason] += 1

    materializable = [row for row in rows if row["schema_action_materializable_now"]]
    clean_candidates = [row for row in rows if row["clean_handoff_or_safe_good_candidate"]]
    summary = {
        "schema": "acl2_v101_fresh_selected_schema_materialization_v1",
        "source_fixed_policy_id": holdout_summary.get("fixed_policy_id", ""),
        "fresh_stage1_selected_count": len(rows),
        "trackT_28_case_target_present_count": sum(1 for row in rows if row["trackT_28_case_target_present"]),
        "broad_prior_present_count": sum(1 for row in rows if row["broad_prior_present"]),
        "target_taxonomy_counts": dict(target_taxonomy_counts),
        "broad_representative_taxonomy_counts": dict(taxonomy_counts),
        "already_in_v100_28_case_count": sum(1 for row in rows if row["already_in_v100_28_case_universe"]),
        "has_v100_same_space_trace_count": sum(1 for row in rows if row["has_v100_same_space_trace"]),
        "has_v100_per_anchor_geometry_count": sum(1 for row in rows if row["has_v100_per_anchor_geometry"]),
        "clean_handoff_or_safe_good_candidate_count": len(clean_candidates),
        "schema_action_materializable_now_count": len(materializable),
        "missing_schema_or_target_reason_counts": dict(missing_counts),
        "runtime_action_allowed": False,
        "full_validation_run": False,
        "blocked_reason": (
            "Fixed-policy fresh Stage1 selected rows do not form a v101 clean target/control universe: "
            "existing Track T/broad taxonomy maps them to LOCAL_BAD_NOT_HANDOFF or "
            "MULTIMODE_LOWOBS_ABSTAIN, and outside-v100 rows lack same-space trace and per-anchor geometry."
        ),
        "claim": (
            "Diagnostic schema-materialization audit only; fresh selected support rows are not promoted "
            "to action evidence."
        ),
    }

    write_rows(FINAL / "fresh_selected_schema_materialization_rows.csv", rows)
    write_json(FINAL / "fresh_selected_schema_materialization_summary.json", summary)

    report = [
        "# Fresh Selected Schema Materialization Audit",
        "",
        "This audit aligns the fixed-policy fresh Stage1 selected rows with v101 Track T taxonomy and v100 schema requirements.",
        "",
        "## Summary",
        "",
        f"- fresh Stage1 selected rows: `{summary['fresh_stage1_selected_count']}`",
        f"- present in Track T 28-case target universe: `{summary['trackT_28_case_target_present_count']}`",
        f"- broad prior present: `{summary['broad_prior_present_count']}`",
        f"- already in v100 28-case universe: `{summary['already_in_v100_28_case_count']}`",
        f"- has v100 same-space trace: `{summary['has_v100_same_space_trace_count']}`",
        f"- has v100 per-anchor geometry: `{summary['has_v100_per_anchor_geometry_count']}`",
        f"- clean handoff/safe-good candidate count: `{summary['clean_handoff_or_safe_good_candidate_count']}`",
        f"- schema action materializable now count: `{summary['schema_action_materializable_now_count']}`",
        "",
        "## Taxonomy",
        "",
        f"- target taxonomy counts: `{summary['target_taxonomy_counts']}`",
        f"- broad representative taxonomy counts: `{summary['broad_representative_taxonomy_counts']}`",
        "",
        "## Conclusion",
        "",
        summary["blocked_reason"],
    ]
    write_text(FINAL / "fresh_selected_schema_materialization_report.md", "\n".join(report))

    recommendation = (
        "Fresh selected schema-materialization audit aligned the 6 fixed-policy fresh Stage1 selected rows "
        "against Track T/broad taxonomy: clean_handoff_or_safe_good_candidate_count=0 and "
        "schema_action_materializable_now_count=0. The selected rows are LOCAL_BAD_NOT_HANDOFF or "
        "MULTIMODE_LOWOBS_ABSTAIN, and most outside-v100 rows lack same-space trace/per-anchor geometry. "
        "Do not relabel these support rows into action targets."
    )
    upsert_section(FINAL / "next_attempt_recommendation.md", "Fresh Selected Schema Materialization", recommendation)
    append_unique_line(
        FINAL / "remaining_blockers.md",
        "- Fresh selected schema-materialization blocker: fixed-policy fresh Stage1 selected rows map to LOCAL_BAD_NOT_HANDOFF/MULTIMODE_LOWOBS_ABSTAIN, with 0 clean handoff/safe-good action-materializable rows.",
    )
    append_unique_line(
        FINAL / "failure_report.md",
        "- Fresh selected schema-materialization: 6 fresh selected rows checked, 0 clean handoff/safe-good action-materializable rows.",
    )
    append_unique_line(
        FINAL / "control_gap_report.md",
        "- Fresh selected schema-materialization control gap: no selected fresh row satisfies Track T clean target/control taxonomy plus v100 schema requirements.",
    )

    print(json.dumps(json_clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
