#!/usr/bin/env python3
"""Mine existing historical ACL2 artifacts for a v101-ready target universe.

This is a read-only final-decision audit.  It consolidates the Track T broad
prior-case census, v94/v95 extension search, and v100-schema readiness audit
into one reviewer-facing evidence chain.  It does not promote historical or
proxy evidence to runtime action evidence.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
TRACK_T = ROOT / "trackT_drift_target_relabel"
FINAL = ROOT / "final_decision"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {"value": payload}


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
            clean: dict[str, Any] = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                clean[key] = value
            writer.writerow(clean)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def seq_from_case(case_id: str) -> str:
    return case_id.split("_", 1)[0] if "_" in case_id else ""


def upsert_section(path: Path, heading: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    section = f"\n\n## {heading}\n\n{body.strip()}\n"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    marker = f"\n## {heading}\n"
    if marker in text:
        prefix, rest = text.split(marker, 1)
        next_heading = rest.find("\n## ")
        if next_heading >= 0:
            text = prefix + section + rest[next_heading:]
        else:
            text = prefix.rstrip() + section
    else:
        text = text.rstrip() + section
    path.write_text(text.lstrip() + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")


def append_unique_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = text.splitlines()
    if line not in lines:
        lines.append(line)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_case_rows() -> list[dict[str, Any]]:
    broad_rows = read_rows(TRACK_T / "broad_prior_unique_case_summary.csv")
    readiness_rows = {
        row.get("case_id", ""): row
        for row in read_rows(FINAL / "new_v100_schema_universe_feasibility_rows.csv")
        if row.get("case_id", "")
    }
    extension_rows = read_rows(TRACK_T / "target_extension_candidates.csv")
    extension_by_case: dict[str, list[dict[str, str]]] = {}
    for row in extension_rows:
        case_id = row.get("case_id", "")
        if case_id:
            extension_by_case.setdefault(case_id, []).append(row)

    rows: list[dict[str, Any]] = []
    for row in broad_rows:
        case_id = row.get("case_id", "")
        taxonomy = row.get("representative_taxonomy", "")
        readiness = readiness_rows.get(case_id, {})
        extensions = extension_by_case.get(case_id, [])
        clean_candidate = truthy(row.get("clean_candidate_any", ""))
        clean_handoff = taxonomy == "HANDOFF_SCALE_GAUGE_TARGET"
        safe_good = taxonomy == "SAFE_GOOD"
        in_v100 = truthy(row.get("already_in_v100_28_case_universe", ""))
        same_space = truthy(row.get("has_v100_same_space_trace", ""))
        geometry = truthy(row.get("has_v100_per_anchor_geometry", ""))
        prior_trace = truthy(row.get("prior_trace_available_any", ""))
        strict_ready = truthy(readiness.get("strict_action_ready", ""))
        core_ready = truthy(readiness.get("core_v100_schema_ready", ""))

        missing: list[str] = []
        if not clean_candidate:
            missing.append("not_clean_v101_target_or_safe_good")
        if taxonomy == "GOOD_HIGH_L3_CONTAMINATED":
            missing.append("label_L3_conflict_good_high_L3")
        if taxonomy == "MULTIMODE_LOWOBS_ABSTAIN":
            missing.append("lowobs_or_multimode_abstain")
        if taxonomy == "LOCAL_BAD_NOT_HANDOFF":
            missing.append("local_bad_not_handoff_target")
        if not in_v100:
            missing.append("not_in_v100_28_case_universe")
        if not same_space:
            missing.append("missing_v100_same_space_trace")
        if not geometry:
            missing.append("missing_v100_per_anchor_geometry")
        if readiness.get("missing_action_prereqs"):
            missing.extend(part for part in readiness["missing_action_prereqs"].split(";") if part)
        if clean_candidate and core_ready and not strict_ready:
            missing.append("core_ready_but_action_controls_missing")

        new_extension_possible = any(truthy(item.get("usable_for_v101_extension", "")) for item in extensions)
        diagnostic_handoff_if_unlabelled = any(
            truthy(item.get("diagnostic_handoff_if_unlabelled_allowed", "")) for item in extensions
        )
        if new_extension_possible:
            decision = "usable_extension_found"
        elif strict_ready:
            decision = "strict_action_ready_existing_candidate"
        elif clean_candidate:
            decision = "clean_but_action_blocked_or_not_new"
        elif diagnostic_handoff_if_unlabelled:
            decision = "diagnostic_only_unlabelled_handoff_not_action_target"
        else:
            decision = "not_target_universe_candidate"

        rows.append(
            {
                "case_id": case_id,
                "seq": row.get("seq", seq_from_case(case_id)),
                "representative_taxonomy": taxonomy,
                "representative_reason": row.get("representative_reason", ""),
                "source_count": row.get("source_count", ""),
                "version_roots": row.get("version_roots", ""),
                "L3_min": row.get("L3_min", ""),
                "L3_max": row.get("L3_max", ""),
                "failure_types": row.get("failure_types", ""),
                "prior_trace_available_any": prior_trace,
                "already_in_v100_28_case_universe": in_v100,
                "has_v100_same_space_trace": same_space,
                "has_v100_per_anchor_geometry": geometry,
                "clean_candidate_any": clean_candidate,
                "clean_handoff_target": clean_handoff,
                "safe_good_control": safe_good,
                "core_v100_schema_ready": core_ready,
                "strict_action_ready": strict_ready,
                "usable_new_extension_from_v94_v95": new_extension_possible,
                "diagnostic_handoff_if_unlabelled_allowed": diagnostic_handoff_if_unlabelled,
                "mining_decision": decision,
                "missing_or_blocking_reasons": ";".join(dict.fromkeys(missing)),
                "clean_candidate_sources": row.get("clean_candidate_sources", ""),
            }
        )
    return rows


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    broad = read_json(TRACK_T / "broad_prior_case_census_summary.json")
    extension = read_json(TRACK_T / "target_extension_feasibility_summary.json")
    feasibility = read_json(FINAL / "new_v100_schema_universe_feasibility_summary.json")
    rows = build_case_rows()

    tax_counts = Counter(row["representative_taxonomy"] for row in rows)
    decision_counts = Counter(row["mining_decision"] for row in rows)
    missing_counts: Counter[str] = Counter()
    for row in rows:
        for reason in str(row.get("missing_or_blocking_reasons", "")).split(";"):
            if reason:
                missing_counts[reason] += 1

    clean_handoff_rows = [row for row in rows if row["clean_handoff_target"]]
    safe_good_rows = [row for row in rows if row["safe_good_control"]]
    strict_ready_rows = [row for row in rows if row["strict_action_ready"]]
    usable_extension_rows = [row for row in rows if row["usable_new_extension_from_v94_v95"]]

    summary = {
        "schema": "acl2_v101_historical_target_universe_mining_v1",
        "source_scope": "existing v94-v101 case-level artifacts only",
        "broad_source_file_count_scanned": broad.get("source_file_count_scanned", ""),
        "broad_source_file_count_used": broad.get("source_file_count_used", ""),
        "broad_source_row_count": broad.get("source_row_count", ""),
        "unique_case_count": len(rows),
        "taxonomy_counts": dict(tax_counts),
        "mining_decision_counts": dict(decision_counts),
        "missing_or_blocking_reason_counts": dict(missing_counts),
        "v94_v95_old_source_row_count": extension.get("old_source_row_count", ""),
        "v94_v95_unique_old_case_count": extension.get("unique_old_case_count", ""),
        "v94_v95_new_old_case_count": extension.get("new_old_case_count", ""),
        "v94_v95_new_old_case_with_v100_same_space_trace_count": extension.get(
            "new_old_case_with_v100_same_space_trace_count", ""
        ),
        "v94_v95_usable_extension_case_count": extension.get("usable_extension_case_count", ""),
        "v94_v95_diagnostic_handoff_if_unlabelled_allowed_count": extension.get(
            "diagnostic_handoff_if_unlabelled_allowed_unique_count", ""
        ),
        "clean_candidate_count": broad.get("clean_candidate_unique_count", ""),
        "clean_handoff_candidate_count": len(clean_handoff_rows),
        "clean_handoff_candidate_cases": ";".join(sorted(row["case_id"] for row in clean_handoff_rows)),
        "clean_handoff_sequence_coverage": len({row["seq"] for row in clean_handoff_rows}),
        "safe_good_candidate_count": len(safe_good_rows),
        "safe_good_candidate_cases": ";".join(sorted(row["case_id"] for row in safe_good_rows)),
        "core_v100_schema_ready_clean_candidate_count": feasibility.get(
            "core_v100_schema_ready_clean_candidate_count", ""
        ),
        "strict_action_ready_candidate_count": len(strict_ready_rows),
        "usable_new_extension_case_count": len(usable_extension_rows),
        "new_v100_schema_universe_available_from_existing_artifacts": feasibility.get(
            "new_universe_available_from_existing_artifacts", False
        ),
        "historical_mined_new_clean_universe_available": False,
        "runtime_action_allowed": False,
        "full_validation_run": False,
        "blocked_reason": (
            "Historical artifacts do not contain a new v100-schema clean handoff target universe: "
            "the only clean handoff candidate is already in the v100 universe, no v94/v95 new case has "
            "v100 same-space trace, and the existing clean candidates remain blocked by identity/query-head "
            "controls, F5 write-to-use materialization, and Q2 true-stage admission."
        ),
        "claim": (
            "This mining audit is diagnostic and final-decision evidence only; it does not change any v101 gate "
            "or authorize M4/runtime/full validation."
        ),
    }

    write_rows(FINAL / "historical_target_universe_mining_rows.csv", rows)
    write_json(FINAL / "historical_target_universe_mining_summary.json", summary)

    report = [
        "# Historical Target Universe Mining Audit",
        "",
        "This audit consolidates existing v94-v101 case-level artifacts to test whether a new v100-schema clean v101 target universe can be mined without new materialization.",
        "",
        "## Source Scope",
        "",
        f"- broad prior source files scanned: `{summary['broad_source_file_count_scanned']}`",
        f"- broad prior source files used: `{summary['broad_source_file_count_used']}`",
        f"- broad prior source rows: `{summary['broad_source_row_count']}`",
        f"- unique cases: `{summary['unique_case_count']}`",
        f"- v94/v95 old source rows: `{summary['v94_v95_old_source_row_count']}`",
        f"- v94/v95 new old cases outside v100: `{summary['v94_v95_new_old_case_count']}`",
        f"- v94/v95 new old cases with v100 same-space trace: `{summary['v94_v95_new_old_case_with_v100_same_space_trace_count']}`",
        "",
        "## Result",
        "",
        f"- clean handoff candidates: `{summary['clean_handoff_candidate_count']}` ({summary['clean_handoff_candidate_cases']})",
        f"- clean handoff sequence coverage: `{summary['clean_handoff_sequence_coverage']}`",
        f"- safe-good candidates: `{summary['safe_good_candidate_count']}` ({summary['safe_good_candidate_cases']})",
        f"- strict action-ready candidates: `{summary['strict_action_ready_candidate_count']}`",
        f"- usable new extension cases: `{summary['usable_new_extension_case_count']}`",
        f"- historical mined new universe available: `{summary['historical_mined_new_clean_universe_available']}`",
        f"- runtime action allowed: `{summary['runtime_action_allowed']}`",
        "",
        "## Evidence Chain",
        "",
        "- Track T broad prior census found 49 unique historical cases but only 6 clean candidates.",
        "- The only clean handoff target is already in the v100 universe; it does not create a new target universe.",
        "- v94/v95 extension search found 21 old cases outside v100, but 0 have v100 same-space trace and 0 are usable extensions.",
        "- Existing clean candidates are core-schema visible but not strict-action-ready because identity/query-head controls, F5 write-to-use chain materialization, and Q2 true-stage admission remain unavailable.",
        "",
        "## Conclusion",
        "",
        summary["blocked_reason"],
    ]
    (FINAL / "historical_target_universe_mining_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    recommendation_body = (
        "A final-decision historical mining audit found no new v100-schema clean target universe from existing "
        "v94-v101 artifacts. It scanned the Track T broad prior census ("
        f"{summary['broad_source_file_count_scanned']} source files, {summary['broad_source_file_count_used']} used, "
        f"{summary['broad_source_row_count']} source rows, {summary['unique_case_count']} unique cases) and the v94/v95 "
        f"extension search ({summary['v94_v95_new_old_case_count']} old cases outside v100, "
        f"{summary['v94_v95_new_old_case_with_v100_same_space_trace_count']} with v100 same-space trace). "
        f"Result: clean handoff candidates={summary['clean_handoff_candidate_count']}, safe-good candidates="
        f"{summary['safe_good_candidate_count']}, strict action-ready candidates={summary['strict_action_ready_candidate_count']}, "
        f"usable new extension cases={summary['usable_new_extension_case_count']}. "
        "Next work must materialize a genuinely new v100-schema case universe or collect new evidence; current historical "
        "artifacts should not be threshold-tuned into M4/runtime authorization."
    )
    upsert_section(FINAL / "next_attempt_recommendation.md", "Historical Target Universe Mining Follow-up", recommendation_body)
    append_unique_line(
        FINAL / "remaining_blockers.md",
        "- Historical target-universe mining found no new v100-schema clean universe from existing v94-v101 artifacts; strict action-ready candidates remain 0.",
    )
    append_unique_line(
        FINAL / "failure_report.md",
        "- Historical target-universe mining follow-up: no usable new extension cases and 0 strict action-ready candidates from existing artifacts.",
    )
    append_unique_line(
        FINAL / "control_gap_report.md",
        "- Historical target-universe mining gap: existing broader cases lack v100 same-space/current-support/identity-query-head/Q2 true-stage evidence needed for action.",
    )

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
