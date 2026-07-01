#!/usr/bin/env python3
"""Build v102 diagnostic state-machine scaffold trace targets.

This target list is for no-action scaffold trace materialization only. It does
not upgrade ambiguous cases to strict positives and does not authorize runtime
actions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
STAGE2_BASE = ROOT / "stage2_base_case_selection/base_case_rows.csv"
STAGE3_EXHAUSTIVE = ROOT / "stage3_semantic_oracle_upper_bound/stage3_exhaustive_clean_handoff_target_mining_rows.csv"
OUT_DIR = ROOT / "stage4_memory_action_surface_oracle"
OUT_CSV = OUT_DIR / "v102_state_machine_scaffold_trace_targets.csv"
OUT_SUMMARY = OUT_DIR / "v102_state_machine_scaffold_trace_targets_summary.json"
OUT_REPORT = OUT_DIR / "v102_state_machine_scaffold_trace_targets_report.md"


SELECTED_CASES = [
    ("02_017_018", "strict_clean_handoff_positive", "only current strict clean handoff positive"),
    ("05_018_019", "ambiguous_materialization_candidate", "top external materialization worklist candidate"),
    ("05_019_020", "ambiguous_materialization_candidate", "top external materialization worklist candidate"),
    ("00_012_013", "ambiguous_materialization_candidate", "broader drift-onset exploration candidate"),
    ("02_004_005", "safe_good_control", "safe-good/control case also present in materialization worklist"),
    ("05_007_008", "safe_good_control", "safe-good/control case also present in materialization worklist"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def parse_case(case_id: str) -> tuple[str, int, int]:
    seq, prev_chunk, curr_chunk = case_id.split("_")
    return f"{int(seq):02d}", int(prev_chunk), int(curr_chunk)


def main() -> None:
    base_rows = {row.get("case_id", ""): row for row in read_rows(STAGE2_BASE)}
    exhaustive_rows = {row.get("case_id", ""): row for row in read_rows(STAGE3_EXHAUSTIVE)}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for case_id, role, reason in SELECTED_CASES:
        base = base_rows.get(case_id)
        if not base:
            missing.append(case_id)
            continue
        exhaustive = exhaustive_rows.get(case_id, {})
        seq, prev_chunk, curr_chunk = parse_case(case_id)
        rows.append(
            {
                "case_id": case_id,
                "seq": seq,
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "target_taxonomy": f"V102_STATE_MACHINE_SCAFFOLD_{role.upper()}",
                "target_reason": reason,
                "case_label": role,
                "failure_type": base.get("primary_drift_source") or base.get("drift_source_labels", ""),
                "L1_local_sim3_ate": base.get("L1_local_sim3_ate", ""),
                "L2_head_tail_proxy_error": base.get("L2_head_tail_proxy_error", ""),
                "L3_handoff_transfer_penalty_proxy": base.get("L3_handoff_transfer_penalty_proxy", ""),
                "L3_J_handoff": base.get("L3_handoff_transfer_penalty_proxy", ""),
                "v96_recommended_next_track": base.get("target_memory_body", ""),
                "action_response_labels": "diagnostic_v102_state_machine_scaffold_no_action",
                "strict_clean_handoff_positive": bool(
                    str(exhaustive.get("exhaustive_strict_clean_handoff_positive", "")).lower() == "true"
                ),
                "ambiguous_or_control_role": role,
                "runtime_action_allowed": False,
            }
        )
    summary = {
        "schema": "acl2_v102_state_machine_scaffold_trace_targets_v1",
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "target_count": len(rows),
        "missing_case_count": len(missing),
        "missing_cases": ";".join(missing),
        "strict_positive_target_count": sum(1 for row in rows if row["strict_clean_handoff_positive"]),
        "ambiguous_materialization_candidate_count": sum(
            1 for row in rows if row["ambiguous_or_control_role"] == "ambiguous_materialization_candidate"
        ),
        "safe_good_control_count": sum(1 for row in rows if row["ambiguous_or_control_role"] == "safe_good_control"),
        "stage3_strict_coverage_repaired": False,
        "stage4_runtime_action_allowed": False,
        "output_csv": OUT_CSV.as_posix(),
    }
    write_rows(OUT_CSV, rows)
    write_json(OUT_SUMMARY, summary)
    write_text(
        OUT_REPORT,
        "\n".join(
            [
                "# v102 State-Machine Scaffold Trace Targets",
                "",
                "Diagnostic-only target list for default-off state-machine scaffold trace materialization.",
                "",
                f"- target_count: {summary['target_count']}",
                f"- strict_positive_target_count: {summary['strict_positive_target_count']}",
                f"- ambiguous_materialization_candidate_count: {summary['ambiguous_materialization_candidate_count']}",
                f"- safe_good_control_count: {summary['safe_good_control_count']}",
                f"- runtime_action_allowed: {summary['runtime_action_allowed']}",
                "",
                "These targets do not repair Stage3 strict coverage by themselves.",
            ]
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
