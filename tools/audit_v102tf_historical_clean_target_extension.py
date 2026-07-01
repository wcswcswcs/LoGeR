#!/usr/bin/env python3
"""Audit whether historical artifacts can extend v102 clean handoff targets.

This Stage3 fail-forward audit checks existing v94-v101 mining/feasibility
artifacts for additional v100-schema clean handoff positives.  It is read-only
and does not authorize action.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
OUT_DIR = ROOT / "stage3_semantic_oracle_upper_bound"
OUT_ROWS = OUT_DIR / "stage3_historical_clean_target_extension_rows.csv"
OUT_SUMMARY = OUT_DIR / "stage3_historical_clean_target_extension_summary.json"
OUT_REPORT = OUT_DIR / "historical_clean_target_extension_report.md"

P_V101 = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
SRC = {
    "historical_summary": P_V101 / "final_decision/historical_target_universe_mining_summary.json",
    "historical_rows": P_V101 / "final_decision/historical_target_universe_mining_rows.csv",
    "new_universe_summary": P_V101 / "final_decision/new_v100_schema_universe_feasibility_summary.json",
    "new_universe_rows": P_V101 / "final_decision/new_v100_schema_universe_feasibility_rows.csv",
    "rich_holdout_summary": P_V101 / "final_decision/rich_selector_holdout_feasibility_summary.json",
    "rich_holdout_rows": P_V101 / "final_decision/rich_selector_holdout_feasibility_rows.csv",
    "trace_rescue_summary": P_V101 / "final_decision/trace_rescue_feasibility_summary.json",
    "remaining_closure_summary": P_V101 / "final_decision/remaining_reentry_route_closure_summary.json",
}
V102_STAGE1_ROWS = ROOT / "stage1_drift_source_autopsy/drift_source_taxonomy.csv"
V102_BROADER_ROWS = OUT_DIR / "stage3_broader_drift_onset_candidate_rows.csv"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
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
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def b(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def split_cases(value: Any) -> list[str]:
    if not value:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in str(value).replace(",", ";").split(";"):
        case_id = item.strip()
        if case_id and case_id not in seen:
            out.append(case_id)
            seen.add(case_id)
    return out


def by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["case_id"]: row for row in rows if row.get("case_id")}


def main() -> int:
    historical_summary = read_json(SRC["historical_summary"])
    new_summary = read_json(SRC["new_universe_summary"])
    rich_summary = read_json(SRC["rich_holdout_summary"])
    trace_summary = read_json(SRC["trace_rescue_summary"])
    closure_summary = read_json(SRC["remaining_closure_summary"])
    v102_stage1 = by_case(read_rows(V102_STAGE1_ROWS))
    broader = by_case(read_rows(V102_BROADER_ROWS))
    historical_rows = read_rows(SRC["historical_rows"])
    new_rows = read_rows(SRC["new_universe_rows"])
    rich_rows = read_rows(SRC["rich_holdout_rows"])

    rows: list[dict[str, Any]] = []
    for row in historical_rows:
        case_id = row.get("case_id", "")
        stage1 = v102_stage1.get(case_id, {})
        broad = broader.get(case_id, {})
        clean_handoff = b(row.get("clean_handoff_target"))
        clean_candidate = b(row.get("clean_candidate_any"))
        safe_good = b(row.get("safe_good_control"))
        new_extension = b(row.get("usable_new_extension_from_v94_v95"))
        strict_ready = b(row.get("strict_action_ready"))
        rows.append(
            {
                "case_id": case_id,
                "seq": row.get("seq", ""),
                "representative_taxonomy": row.get("representative_taxonomy", ""),
                "v102_primary_drift_source": stage1.get("primary_drift_source", ""),
                "v102_drift_source_labels": stage1.get("drift_source_labels", ""),
                "v102_broader_promotion_role": broad.get("promotion_role", ""),
                "already_in_v100_28_case_universe": row.get("already_in_v100_28_case_universe", ""),
                "has_v100_same_space_trace": row.get("has_v100_same_space_trace", ""),
                "has_v100_per_anchor_geometry": row.get("has_v100_per_anchor_geometry", ""),
                "clean_candidate_any": clean_candidate,
                "clean_handoff_target": clean_handoff,
                "safe_good_control": safe_good,
                "core_v100_schema_ready": row.get("core_v100_schema_ready", ""),
                "strict_action_ready": strict_ready,
                "usable_new_extension_from_v94_v95": new_extension,
                "v102_strict_extension_candidate": clean_handoff and new_extension and strict_ready,
                "mining_decision": row.get("mining_decision", ""),
                "missing_or_blocking_reasons": row.get("missing_or_blocking_reasons", ""),
                "claim_level": "historical_clean_target_extension_no_action",
            }
        )

    rich_selected = split_cases(rich_summary.get("fresh_stage1_fixed_policy_selected_pairs"))
    rich_labelled = int(float(rich_summary.get("fresh_labelled_bad_good_holdout_pair_count") or 0))
    historical_new = [row for row in rows if row["v102_strict_extension_candidate"]]
    clean_handoff_cases = [row["case_id"] for row in rows if row["clean_handoff_target"]]
    safe_good_cases = [row["case_id"] for row in rows if row["safe_good_control"]]
    usable_new_cases = [row["case_id"] for row in rows if row["usable_new_extension_from_v94_v95"]]
    new_universe_strict_ready = [row.get("case_id", "") for row in new_rows if b(row.get("strict_action_ready"))]
    summary = {
        "schema": "acl2_v102_historical_clean_target_extension_v1",
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "stage3_strict_repaired": False,
        "historical_row_count": len(historical_rows),
        "historical_unique_case_count": historical_summary.get("unique_case_count"),
        "historical_clean_candidate_count": historical_summary.get("clean_candidate_count"),
        "historical_clean_handoff_candidate_count": historical_summary.get("clean_handoff_candidate_count"),
        "historical_clean_handoff_candidate_cases": ";".join(clean_handoff_cases),
        "historical_safe_good_candidate_count": historical_summary.get("safe_good_candidate_count"),
        "historical_safe_good_candidate_cases": ";".join(safe_good_cases),
        "historical_mined_new_clean_universe_available": historical_summary.get("historical_mined_new_clean_universe_available"),
        "usable_new_extension_case_count": len(usable_new_cases),
        "usable_new_extension_cases": ";".join(usable_new_cases),
        "v102_strict_extension_candidate_count": len(historical_new),
        "v102_strict_extension_candidate_cases": ";".join(row["case_id"] for row in historical_new),
        "new_v100_schema_universe_available_from_existing_artifacts": new_summary.get("new_universe_available_from_existing_artifacts"),
        "new_universe_strict_action_ready_clean_candidate_count": new_summary.get("strict_action_ready_clean_candidate_count"),
        "new_universe_strict_ready_cases": ";".join(new_universe_strict_ready),
        "rich_holdout_fresh_labelled_bad_good_pair_count": rich_labelled,
        "rich_holdout_fresh_stage1_fixed_policy_selected_count": rich_summary.get("fresh_stage1_fixed_policy_selected_count"),
        "rich_holdout_fresh_stage1_fixed_policy_selected_pairs": ";".join(rich_selected),
        "trace_rescue_available": trace_summary.get("trace_rescue_available"),
        "strict_instance_identity_rescued": trace_summary.get("strict_instance_identity_rescued"),
        "query_head_controls_rescued": trace_summary.get("query_head_controls_rescued"),
        "write_cache_current_chain_rescued": trace_summary.get("write_cache_current_chain_rescued"),
        "q2_true_stage_rescued": trace_summary.get("q2_true_stage_rescued"),
        "remaining_reentry_action_allowed_route_count": closure_summary.get("action_allowed_route_count"),
        "blocker": (
            "Existing historical artifacts do not provide a new strict v100-schema clean handoff universe: "
            "the only clean handoff target remains 02_017_018, fresh holdout rows are unlabelled/no bad-good gate, "
            "and trace rescue does not recover instance identity, query-head controls, write/cache/current chain, or Q2 true-stage."
        ),
        "next_repair": (
            "A real repair needs newly materialized labelled clean handoff targets with same-space trace, per-anchor geometry, "
            "strict instance identity, query-head controls, write/cache/current chain, and Q2 true-stage evidence."
        ),
        "outputs": {"rows": OUT_ROWS.as_posix(), "report": OUT_REPORT.as_posix()},
    }
    write_rows(OUT_ROWS, rows)
    write_json(OUT_SUMMARY, summary)
    write_text(
        OUT_REPORT,
        "# Historical Clean Target Extension\n\n"
        "This read-only audit checks whether existing v94-v101 artifacts can expand the v102 strict clean handoff target universe.\n\n"
        f"- historical_row_count: {summary['historical_row_count']}\n"
        f"- historical_clean_handoff_candidate_count: {summary['historical_clean_handoff_candidate_count']}\n"
        f"- historical_clean_handoff_candidate_cases: {summary['historical_clean_handoff_candidate_cases']}\n"
        f"- usable_new_extension_case_count: {summary['usable_new_extension_case_count']}\n"
        f"- v102_strict_extension_candidate_count: {summary['v102_strict_extension_candidate_count']}\n"
        f"- rich_holdout_fresh_labelled_bad_good_pair_count: {summary['rich_holdout_fresh_labelled_bad_good_pair_count']}\n"
        f"- trace_rescue_available: {summary['trace_rescue_available']}\n\n"
        "Conclusion:\n\n"
        + summary["blocker"]
        + "\n\nNext repair:\n\n"
        + summary["next_repair"]
        + "\n",
    )
    print(json.dumps(jsonable(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
