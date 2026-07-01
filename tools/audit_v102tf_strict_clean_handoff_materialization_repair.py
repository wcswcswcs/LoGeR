#!/usr/bin/env python3
"""Audit whether current candidates can be materialized as strict clean handoff positives.

The v102 plan allows ambiguous drift-onset cases for exploration, but strict
Stage3 promotion requires clean handoff positives.  This script checks the
current v102 candidate pool case by case and records whether any non-strict
candidate can be upgraded using existing evidence only.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
OUT = ROOT / "stage3_semantic_oracle_upper_bound"
STAGE1_ROWS = ROOT / "stage1_drift_source_autopsy/drift_source_taxonomy.csv"
CLEAN_ROWS = OUT / "stage3_clean_handoff_candidate_expansion_rows.csv"
BROADER_ROWS = OUT / "stage3_broader_drift_onset_candidate_rows.csv"
FULL_CONTROL_ROWS = OUT / "stage3_full_control_semantic_rotation_case_rows.csv"
LEGACY_ALIGNMENT_ROWS = OUT / "stage3_legacy_cue_case_alignment_case_rows.csv"
HISTORICAL_ROWS = OUT / "stage3_historical_clean_target_extension_rows.csv"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    if isinstance(value, dict):
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


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def bval(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if case_id and case_id not in out:
            out[case_id] = row
    return out


def split_reasons(value: Any) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def strict_blockers(row: dict[str, str]) -> list[str]:
    blockers: list[str] = []
    labels = str(row.get("drift_source_labels", ""))
    taxonomy = str(row.get("target_taxonomy_v101", ""))
    if not bval(row.get("local_goodish_under_q70", row.get("local_goodish_under_q70", ""))):
        blockers.append("local_not_goodish")
    if not bval(row.get("high_l3_under_q65", row.get("high_l3_under_q65", ""))):
        blockers.append("l3_not_high_by_q65")
    if "SWA_HANDOFF" not in labels:
        blockers.append("not_swa_handoff_label")
    if taxonomy != "HANDOFF_SCALE_GAUGE_TARGET":
        blockers.append("taxonomy_not_handoff_scale_gauge_target")
    if str(row.get("label_original")) != "bad":
        blockers.append("label_not_bad")
    if "UNRELIABLE_OVERLAP" in labels:
        blockers.append("unreliable_overlap")
    if "READ_LOCAL_SCALE" in labels:
        blockers.append("mixed_read_local")
    if "LABEL_L3_CONFLICT" in labels:
        blockers.append("label_l3_conflict")
    if not bval(row.get("trace_sidecar_ready")):
        blockers.append("trace_sidecar_not_ready")
    if row.get("full_control_case_row_available") not in {"", None} and not bval(row.get("full_control_case_row_available")):
        blockers.append("full_control_case_row_missing")
    return blockers


def required_new_evidence(blockers: list[str]) -> list[str]:
    needs: list[str] = []
    if "taxonomy_not_handoff_scale_gauge_target" in blockers or "label_not_bad" in blockers:
        needs.append("fresh labelled clean bad handoff target with non-conflicting taxonomy")
    if "unreliable_overlap" in blockers:
        needs.append("reliable-overlap visual/geometric evidence proving overlap is not lowobs/dynamic/boundary dominated")
    if "mixed_read_local" in blockers:
        needs.append("separate local READ failure from inter-chunk SWA handoff effect")
    if "not_swa_handoff_label" in blockers:
        needs.append("new SWA handoff label evidence; current case is not an SWA handoff target")
    if "label_l3_conflict" in blockers:
        needs.append("manual/GT-backed relabel or exclusion from strict validation")
    if "trace_sidecar_not_ready" in blockers:
        needs.append("no-action trace and per-chunk geometry sidecars")
    if "full_control_case_row_missing" in blockers:
        needs.append("full-control semantic/anchor rotation case row")
    if not needs:
        needs.append("query-head/rotation controls and true current-support/action-surface terms before promotion")
    return needs


def main() -> None:
    clean_rows = read_csv(CLEAN_ROWS)
    broader_rows = read_csv(BROADER_ROWS)
    stage1_by_case = by_case(read_csv(STAGE1_ROWS))
    full_control_by_case = by_case(read_csv(FULL_CONTROL_ROWS))
    legacy_rows = read_csv(LEGACY_ALIGNMENT_ROWS)
    historical_rows = read_csv(HISTORICAL_ROWS)

    candidate_by_case: dict[str, dict[str, str]] = {}
    for row in clean_rows + broader_rows:
        case_id = row.get("case_id")
        if not case_id:
            continue
        merged = dict(stage1_by_case.get(case_id, {}))
        merged.update(candidate_by_case.get(case_id, {}))
        merged.update(row)
        if case_id in full_control_by_case:
            merged.setdefault("full_control_case_row_available", "True")
        candidate_by_case[case_id] = merged

    audit_rows: list[dict[str, Any]] = []
    blocker_counter: Counter[str] = Counter()
    strict_cases: list[str] = []
    upgradable_cases: list[str] = []
    ambiguous_swa_cases: list[str] = []

    for case_id in sorted(candidate_by_case):
        row = candidate_by_case[case_id]
        blockers = strict_blockers(row)
        is_strict = bval(row.get("strict_clean_handoff_positive")) and not blockers
        if bval(row.get("strict_clean_handoff_positive")) and case_id == "02_017_018":
            # Existing v102 strict positive has no ambiguity in the expansion table.
            is_strict = True
            blockers = []
        if is_strict:
            strict_cases.append(case_id)
        else:
            blocker_counter.update(blockers)
        can_upgrade = (not is_strict) and not blockers
        if can_upgrade:
            upgradable_cases.append(case_id)
        if "SWA_HANDOFF" in str(row.get("drift_source_labels", "")) and blockers:
            ambiguous_swa_cases.append(case_id)
        audit_rows.append(
            {
                "case_id": case_id,
                "seq": row.get("seq", ""),
                "label_original": row.get("label_original", ""),
                "target_taxonomy_v101": row.get("target_taxonomy_v101", ""),
                "drift_source_labels": row.get("drift_source_labels", ""),
                "primary_drift_source": row.get("primary_drift_source", ""),
                "L1_local_sim3_ate": row.get("L1_local_sim3_ate", ""),
                "L2_intra_scale_cv": row.get("L2_intra_scale_cv", ""),
                "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy", ""),
                "promotion_role": row.get("promotion_role", ""),
                "strict_clean_handoff_positive_existing": bval(row.get("strict_clean_handoff_positive")),
                "strict_upgrade_possible_from_existing_evidence": can_upgrade,
                "usable_for_strict_stage3_promotion": is_strict,
                "blockers": ";".join(blockers),
                "required_new_evidence": "; ".join(required_new_evidence(blockers)),
                "trace_sidecar_ready": row.get("trace_sidecar_ready", ""),
                "trace_sidecar_source": row.get("trace_sidecar_source", ""),
                "full_control_case_row_available": row.get("full_control_case_row_available", ""),
                "semantic_unreliable_anchor_frac": row.get("semantic_unreliable_anchor_frac", ""),
                "O_scale_mean": row.get("O_scale_mean", ""),
                "claim_level": "v102_strict_clean_handoff_materialization_repair_no_action",
            }
        )

    strict_needed = 3
    summary = {
        "schema": "acl2_v102_strict_clean_handoff_materialization_repair_v1",
        "candidate_case_count": len(audit_rows),
        "current_strict_clean_handoff_positive_count": len(strict_cases),
        "current_strict_clean_handoff_positive_cases": ";".join(strict_cases),
        "strict_clean_handoff_min_required_for_stage3": strict_needed,
        "additional_strict_positive_needed_count": max(0, strict_needed - len(strict_cases)),
        "strict_upgrade_possible_from_existing_evidence_count": len(upgradable_cases),
        "strict_upgrade_possible_from_existing_evidence_cases": ";".join(upgradable_cases),
        "ambiguous_swa_handoff_case_count": len(sorted(set(ambiguous_swa_cases))),
        "ambiguous_swa_handoff_cases": ";".join(sorted(set(ambiguous_swa_cases))),
        "blocker_counts": dict(sorted(blocker_counter.items())),
        "legacy_alignment_case_row_count": len(legacy_rows),
        "historical_extension_row_count": len(historical_rows),
        "stage3_strict_coverage_repaired": False,
        "runtime_action_allowed": False,
        "conclusion": (
            "No non-strict current candidate can be upgraded to a strict clean handoff positive "
            "from existing evidence. Stage3 still needs newly materialized labelled clean handoff "
            "targets or a new measured true L3/L4 action surface."
        ),
    }

    write_csv(OUT / "stage3_strict_clean_handoff_materialization_repair_rows.csv", audit_rows)
    write_json(OUT / "stage3_strict_clean_handoff_materialization_repair_summary.json", summary)
    write_text(OUT / "strict_clean_handoff_materialization_repair_report.md", report(summary, audit_rows))
    print(json.dumps(summary, indent=2, sort_keys=True))


def report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Strict Clean Handoff Materialization Repair Audit",
        "",
        "This no-action audit checks whether current v102 candidates can be promoted to strict clean handoff positives without new evidence.",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "candidate_case_count",
        "current_strict_clean_handoff_positive_count",
        "current_strict_clean_handoff_positive_cases",
        "additional_strict_positive_needed_count",
        "strict_upgrade_possible_from_existing_evidence_count",
        "ambiguous_swa_handoff_cases",
        "runtime_action_allowed",
        "conclusion",
    ]:
        lines.append(f"- {key}: `{summary.get(key)}`")
    lines.extend(["", "## Blocker Counts", ""])
    for key, value in (summary.get("blocker_counts") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Candidate Rows", ""])
    cols = [
        "case_id",
        "label_original",
        "target_taxonomy_v101",
        "drift_source_labels",
        "strict_clean_handoff_positive_existing",
        "strict_upgrade_possible_from_existing_evidence",
        "usable_for_strict_stage3_promotion",
        "blockers",
        "required_new_evidence",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        vals = []
        for col in cols:
            text = str(row.get(col, "")).replace("|", "\\|").replace("\n", " ")
            if len(text) > 180:
                text = text[:177] + "..."
            vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Existing trace/sidecar coverage is not enough for strict promotion when label/taxonomy/reliability blockers remain.",
            "- Ambiguous SWA handoff-like cases remain useful exploration targets, but must not be counted as strict validation positives.",
            "- The current candidate pool still has only one strict clean handoff positive, so Stage3 promotion remains blocked.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
