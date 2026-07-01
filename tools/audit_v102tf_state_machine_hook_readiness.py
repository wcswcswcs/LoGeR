#!/usr/bin/env python3
"""Audit readiness for the v102 SWA/TTT state-machine action hooks.

The v102 plan forbids promoting old source-gate/source-replace/query-soft
families as the new action body.  This audit checks whether the codebase and
current artifacts already contain a non-forbidden state-machine hook with true
L3/L4 measurement closure.  It does not run runtime actions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
OUT = ROOT / "stage4_memory_action_surface_oracle"

STAGE3_SUMMARY = ROOT / "stage3_semantic_oracle_upper_bound/stage3_summary.json"
EXHAUSTIVE_TARGET_SUMMARY = ROOT / "stage3_semantic_oracle_upper_bound/stage3_exhaustive_clean_handoff_target_mining_summary.json"
STAGE4_SUMMARY = OUT / "stage4_summary.json"
TRUE_L3_SUMMARY = OUT / "action_surface_true_l3_upper_bound_feasibility_summary.json"

CODE_FILES = [
    Path("loger/models/pi3.py"),
    Path("loger/models/layers/attention.py"),
    Path("run_pipeline_abc_v2.py"),
    Path("loger/pipeline/hybrid_memory_controller.py"),
]

SUMMARY_PATH = OUT / "state_machine_hook_readiness_summary.json"
ROWS_PATH = OUT / "state_machine_hook_readiness_rows.csv"
CODE_LOCI_PATH = OUT / "state_machine_hook_code_loci.csv"
REPORT_PATH = OUT / "state_machine_hook_readiness_report.md"


OLD_OR_FORBIDDEN_PATTERNS = {
    "swa_overlap_source_gate": "old SWA source_gate family",
    "swa_overlap_source_replace": "old SWA source_replace family",
    "prev_ttt_anchor_query_soft": "old TTT/SWA query-soft family",
    "merge_alpha": "old merge-alpha/simple selector family",
}

EXPECTED_NEW_PATTERNS = {
    "v102_swa_state_machine": "generic v102 SWA state machine",
    "TRANSMIT_SUPPORTED_ANCHORS": "SWA transmit supported anchors",
    "REJECT_UNRELIABLE_ANCHORS": "SWA reject unreliable anchors",
    "DELAY_UPDATE": "SWA delay update",
    "HOLD_PREV_REFERENCE": "SWA hold previous reference",
    "CONTEXT_ONLY_DEMOTION": "SWA context-only demotion",
    "WRITE_CONFIRMED_ANCHORS_ONLY": "TTT write confirmed anchors",
    "EXPIRE_UNSUPPORTED_STALE_ANCHORS": "TTT expire unsupported stale anchors",
    "REFRESH_SUPPORTED_STALE_ANCHORS": "TTT refresh supported stale anchors",
    "WRITE_CONTEXT_ONLY": "TTT context-only write",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in keys})


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def find_code_loci(patterns: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in CODE_FILES:
        if not file_path.is_file():
            rows.append(
                {
                    "pattern": "",
                    "description": "",
                    "path": file_path.as_posix(),
                    "line": "",
                    "status": "file_missing",
                    "line_text": "",
                }
            )
            continue
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for pattern, description in patterns.items():
            for idx, line in enumerate(lines, start=1):
                if pattern in line:
                    rows.append(
                        {
                            "pattern": pattern,
                            "description": description,
                            "path": file_path.as_posix(),
                            "line": idx,
                            "status": "present",
                            "line_text": line.strip()[:220],
                        }
                    )
    return rows


def present_patterns(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("pattern")) for row in rows if row.get("status") == "present" and row.get("pattern")}


def short_loci(rows: list[dict[str, Any]], pattern_names: set[str]) -> str:
    selected = [
        f"{row['path']}:{row['line']}:{row['pattern']}"
        for row in rows
        if row.get("pattern") in pattern_names and row.get("status") == "present"
    ]
    return ";".join(selected[:12])


def action_rows(
    old_loci: list[dict[str, Any]],
    new_present: set[str],
    strict_target_ready: bool,
    true_l3_ready: bool,
) -> list[dict[str, Any]]:
    old_swa = {"swa_overlap_source_gate", "swa_overlap_source_replace"}
    old_ttt = {"prev_ttt_anchor_query_soft"}
    rows: list[dict[str, Any]] = []
    actions = [
        ("C2a", "SWA", "TRANSMIT_SUPPORTED_ANCHORS", old_swa),
        ("C2b", "SWA", "REJECT_UNRELIABLE_ANCHORS", old_swa),
        ("C2c", "SWA", "DELAY_UPDATE", old_swa),
        ("C2d", "SWA", "HOLD_PREV_REFERENCE", old_swa),
        ("C2e", "SWA", "CONTEXT_ONLY_DEMOTION", old_swa),
        ("C3a", "TTT", "WRITE_CONFIRMED_ANCHORS_ONLY", old_ttt),
        ("C3b", "TTT", "EXPIRE_UNSUPPORTED_STALE_ANCHORS", old_ttt),
        ("C3c", "TTT", "REFRESH_SUPPORTED_STALE_ANCHORS", old_ttt),
        ("C3d", "TTT", "WRITE_CONTEXT_ONLY", old_ttt),
    ]
    for action_id, memory_body, action_name, related_old in actions:
        has_new_hook = action_name in new_present or "v102_swa_state_machine" in new_present
        blockers = []
        if not strict_target_ready:
            blockers.append("stage3_strict_clean_handoff_targets_not_ready")
        if not has_new_hook:
            blockers.append("new_non_forbidden_state_machine_hook_missing")
        if not true_l3_ready:
            blockers.append("measured_true_l3_l4_effect_missing")
        blockers.append("old_related_hook_family_is_forbidden_or_already_failed")
        if has_new_hook:
            next_required = (
                "keep the v102 diagnostic scaffold default-off, add measured true L3/L4 closure and strict target "
                "coverage evidence, then only promote a non-mutating diagnostic hook to runtime action after the "
                "plan gates pass; do not reuse old source_gate/source_replace/query_soft families as the action body"
            )
        else:
            next_required = (
                "implement a new diagnostic-first state-machine hook with anchor identity, current support, "
                "O_scale/R_same, query-head controls, and true L3/L4 evaluator; do not reuse old source_gate/"
                "source_replace/query_soft families as the action body"
            )
        rows.append(
            {
                "action_id": action_id,
                "memory_body": memory_body,
                "plan_action": action_name,
                "new_state_machine_hook_detected": has_new_hook,
                "old_related_hook_loci": short_loci(old_loci, related_old),
                "strict_target_ready": strict_target_ready,
                "measured_true_l3_l4_effect_available": true_l3_ready,
                "runtime_action_allowed": False,
                "readiness_status": "not_ready" if blockers else "ready",
                "blockers": blockers,
                "next_required": next_required,
                "claim_level": "v102_state_machine_hook_readiness_no_action",
            }
        )
    return rows


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            text = str(csv_value(row.get(col, ""))).replace("|", "\\|").replace("\n", " ")
            if len(text) > 180:
                text = text[:177] + "..."
            values.append(text)
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    stage3 = read_json(STAGE3_SUMMARY)
    exhaustive_targets = read_json(EXHAUSTIVE_TARGET_SUMMARY)
    stage4 = read_json(STAGE4_SUMMARY)
    true_l3 = read_json(TRUE_L3_SUMMARY)

    old_loci = find_code_loci(OLD_OR_FORBIDDEN_PATTERNS)
    new_loci = find_code_loci(EXPECTED_NEW_PATTERNS)
    old_present = present_patterns(old_loci)
    new_present = present_patterns(new_loci)

    strict_target_ready = bool(stage3.get("stage3_strict_semantic_oracle_pass")) or bool(
        exhaustive_targets.get("stage3_strict_coverage_repaired")
    )
    true_l3_ready = bool(true_l3.get("stage4_strict_memory_action_surface_pass")) or bool(
        true_l3.get("strict_action_surface_upper_bound_pass_count")
    )
    rows = action_rows(old_loci, new_present, strict_target_ready, true_l3_ready)
    blocking_action_count = sum(1 for row in rows if row["readiness_status"] != "ready")
    if new_present:
        conclusion = (
            "A default-off v102 diagnostic state-machine scaffold is present, but no state-machine action is ready "
            "because Stage3 strict target coverage is false and measured true L3/L4 closure is missing. Old "
            "source_gate/source_replace/query_soft hook families remain forbidden or already failed as v102 action bodies."
        )
        repair_interpretation = (
            "The detected v102 scaffold is diagnostic-only; a real repair now requires strict target coverage and a "
            "true L3/L4 evaluator before any Stage5/6/7 promotion."
        )
    else:
        conclusion = (
            "No non-forbidden v102 state-machine hook with measured true L3/L4 closure is currently ready. "
            "The codebase contains old source_gate/source_replace/query_soft hook families, but those are "
            "explicitly forbidden or already failed as action bodies for v102."
        )
        repair_interpretation = (
            "A real repair requires a new diagnostic-first hook and true L3/L4 evaluator before any Stage5/6/7 promotion."
        )
    summary = {
        "schema": "acl2_v102_state_machine_hook_readiness_v1",
        "stage3_strict_semantic_oracle_pass": bool(stage3.get("stage3_strict_semantic_oracle_pass")),
        "stage3_strict_coverage_repaired": bool(exhaustive_targets.get("stage3_strict_coverage_repaired")),
        "strict_clean_handoff_positive_count": exhaustive_targets.get("exhaustive_strict_clean_handoff_positive_count"),
        "additional_strict_positive_needed_count": exhaustive_targets.get("additional_strict_positive_needed_count"),
        "stage4_strict_memory_action_surface_pass": bool(stage4.get("strict_memory_action_surface_pass")),
        "existing_true_l3_upper_bound_pass": bool(true_l3.get("stage4_strict_memory_action_surface_pass")),
        "old_or_forbidden_hook_pattern_count": len(old_present),
        "old_or_forbidden_hook_patterns_present": ";".join(sorted(old_present)),
        "new_v102_state_machine_hook_pattern_count": len(new_present),
        "new_v102_state_machine_hook_patterns_present": ";".join(sorted(new_present)),
        "state_machine_action_count": len(rows),
        "state_machine_action_ready_count": len(rows) - blocking_action_count,
        "state_machine_action_blocked_count": blocking_action_count,
        "measured_true_l3_l4_effect_available": true_l3_ready,
        "runtime_action_allowed": False,
        "stage5_allowed": False,
        "conclusion": conclusion,
    }

    write_csv(CODE_LOCI_PATH, old_loci + new_loci)
    write_csv(ROWS_PATH, rows)
    write_json(SUMMARY_PATH, summary)
    report = [
        "# State-Machine Hook Readiness Audit",
        "",
        "This no-action audit checks whether the v102 SWA/TTT state-machine action bodies already exist and have true L3/L4 measurement closure.",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "stage3_strict_semantic_oracle_pass",
        "stage3_strict_coverage_repaired",
        "strict_clean_handoff_positive_count",
        "additional_strict_positive_needed_count",
        "stage4_strict_memory_action_surface_pass",
        "old_or_forbidden_hook_patterns_present",
        "new_v102_state_machine_hook_pattern_count",
        "measured_true_l3_l4_effect_available",
        "state_machine_action_ready_count",
        "state_machine_action_blocked_count",
        "runtime_action_allowed",
        "conclusion",
    ]:
        report.append(f"- {key}: `{summary.get(key)}`")
    report.extend(
        [
            "",
            "## Action Readiness Rows",
            "",
            md_table(
                rows,
                [
                    "action_id",
                    "memory_body",
                    "plan_action",
                    "new_state_machine_hook_detected",
                    "strict_target_ready",
                    "measured_true_l3_l4_effect_available",
                    "readiness_status",
                    "blockers",
                ],
            ),
            "## Interpretation",
            "",
            "- Existing source-gate/source-replace/query-soft hooks are not acceptable v102 state-machine action bodies.",
            f"- {repair_interpretation}",
            "- Because Stage3 strict coverage is still false, this audit does not run or authorize runtime action.",
        ]
    )
    write_text(REPORT_PATH, "\n".join(report))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
