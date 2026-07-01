#!/usr/bin/env python3
"""Summarize the remaining strict-action frontier for v101 clean candidates."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"


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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: row.get(key, "") for key in fields})


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


def b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def first_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    clean_rows = read_rows(FINAL / "new_v100_schema_universe_feasibility_rows.csv")
    jl4 = read_json(ROOT / "trackJL4_semantic_anchor_instance_atlas/JL4_summary.json")
    r3 = read_json(ROOT / "trackR3_query_head_anchor_edge_audit_true_support/R3_summary.json")
    f5 = read_json(ROOT / "trackF5_ttt_write_to_use_state_chain/F5_summary.json")
    q2 = read_json(ROOT / "trackQ2_scale_update_admission/Q2_summary.json")
    r3_metric = first_row(read_rows(ROOT / "trackR3_query_head_anchor_edge_audit_true_support/metric_summary.csv"))
    f5_audit = first_row(read_rows(ROOT / "trackF5_ttt_write_to_use_state_chain/write_to_use_materialization_audit.csv"))
    q2_metric = first_row(read_rows(ROOT / "trackQ2_scale_update_admission/admission_metric_summary.csv"))

    blocker_counts: Counter[str] = Counter()
    for row in clean_rows:
        for item in str(row.get("missing_action_prereqs", "")).split(";"):
            if item:
                blocker_counts[item] += 1

    prereq_evidence = {
        "strict_instance_identity_unavailable": {
            "track": "JL4",
            "gate_pass": jl4.get("gate_pass", ""),
            "evidence": (
                f"distinguishes_region_label_from_instance={jl4.get('distinguishes_region_label_from_instance')}; "
                f"records_role_transitions_across_chunks={jl4.get('records_role_transitions_across_chunks')}; "
                f"identity_resolution_level_counts={jl4.get('identity_resolution_level_counts')}"
            ),
            "repair_direction": "materialize stable instance/component ids and role transitions before identity-specific action",
        },
        "query_head_control_margins_unavailable": {
            "track": "R3",
            "gate_pass": r3.get("gate_pass", ""),
            "evidence": (
                f"control_margins_available={r3_metric.get('control_margins_available')}; "
                f"query_head_random_margin={r3_metric.get('query_head_random_margin')}; "
                f"anchor_id_rotation_margin={r3_metric.get('anchor_id_rotation_margin')}; "
                f"selected_positive_sequence_coverage={r3_metric.get('selected_positive_sequence_coverage')}"
            ),
            "repair_direction": "add query-head random plus anchor/semantic rotation controls after target universe has sequence coverage",
        },
        "write_cache_current_chain_not_materialized": {
            "track": "F5",
            "gate_pass": f5.get("gate_pass", ""),
            "evidence": (
                f"r_write_cache_nonempty={f5_audit.get('r_write_cache_nonempty')}; "
                f"r_cache_current_nonempty={f5_audit.get('r_cache_current_nonempty')}; "
                f"r_ref_current_nonempty={f5_audit.get('r_ref_current_nonempty')}; "
                f"anchor_state_row_count={f5_audit.get('anchor_state_row_count')}"
            ),
            "repair_direction": "materialize per-anchor write/cache/current residual chain before using TTT write-to-use cues",
        },
        "q2_true_stage_unavailable": {
            "track": "Q2",
            "gate_pass": q2.get("gate_pass", ""),
            "evidence": (
                f"true_stage_pass={q2.get('true_stage_pass')}; proxy_only={q2.get('proxy_only')}; "
                f"bad_recall={q2_metric.get('bad_recall')}; good_FPR={q2_metric.get('good_FPR')}; "
                f"selected_positive_sequence_coverage={q2_metric.get('selected_positive_sequence_coverage')}"
            ),
            "repair_direction": "rebuild admission with true current support/scale observability and enough clean target sequence coverage",
        },
    }

    rows: list[dict[str, Any]] = []
    for clean in clean_rows:
        for prereq in str(clean.get("missing_action_prereqs", "")).split(";"):
            if not prereq:
                continue
            evidence = prereq_evidence.get(prereq, {})
            rows.append(
                {
                    "case_id": clean.get("case_id", ""),
                    "candidate_kind": clean.get("candidate_kind", ""),
                    "core_v100_schema_ready": clean.get("core_v100_schema_ready", ""),
                    "strict_action_ready": clean.get("strict_action_ready", ""),
                    "missing_prereq": prereq,
                    "evidence_track": evidence.get("track", ""),
                    "evidence_track_gate_pass": evidence.get("gate_pass", ""),
                    "evidence": evidence.get("evidence", ""),
                    "repair_direction": evidence.get("repair_direction", ""),
                    "runtime_action_allowed": False,
                    "claim_level": "strict_action_frontier_no_action",
                }
            )

    clean_handoff = [row for row in clean_rows if row.get("candidate_kind") == "HANDOFF_SCALE_GAUGE_TARGET"]
    safe_good = [row for row in clean_rows if row.get("candidate_kind") == "SAFE_GOOD"]
    strict_ready = [row for row in clean_rows if b(row.get("strict_action_ready"))]
    core_ready = [row for row in clean_rows if b(row.get("core_v100_schema_ready"))]
    summary = {
        "schema": "acl2_v101_strict_action_frontier_v1",
        "clean_candidate_count": len(clean_rows),
        "clean_handoff_candidate_count": len(clean_handoff),
        "safe_good_candidate_count": len(safe_good),
        "core_v100_schema_ready_clean_candidate_count": len(core_ready),
        "strict_action_ready_clean_candidate_count": len(strict_ready),
        "missing_prereq_counts": dict(blocker_counts),
        "frontier_row_count": len(rows),
        "jl4_gate_pass": jl4.get("gate_pass", ""),
        "jl4_identity_resolution_level_counts": jl4.get("identity_resolution_level_counts", {}),
        "r3_gate_pass": r3.get("gate_pass", ""),
        "r3_control_margins_available": r3_metric.get("control_margins_available", ""),
        "f5_gate_pass": f5.get("gate_pass", ""),
        "f5_r_write_cache_nonempty": f5_audit.get("r_write_cache_nonempty", ""),
        "f5_r_cache_current_nonempty": f5_audit.get("r_cache_current_nonempty", ""),
        "f5_r_ref_current_nonempty": f5_audit.get("r_ref_current_nonempty", ""),
        "q2_gate_pass": q2.get("gate_pass", ""),
        "q2_true_stage_pass": q2.get("true_stage_pass", ""),
        "q2_proxy_only": q2.get("proxy_only", ""),
        "q2_good_FPR": q2_metric.get("good_FPR", ""),
        "q2_selected_positive_sequence_coverage": q2_metric.get("selected_positive_sequence_coverage", ""),
        "runtime_action_allowed": False,
        "full_validation_run": False,
        "blocked_reason": (
            "The six clean candidates are core-v100-schema-ready but none is strict-action-ready: all share "
            "missing instance identity, query-head controls, write/cache/current materialization, and Q2 true-stage. "
            "Additionally Track T still has only one clean handoff target and five safe-good controls."
        ),
        "claim": "Frontier audit only; it prioritizes remaining instrumentation/control blockers without authorizing action.",
    }

    write_rows(FINAL / "strict_action_frontier_rows.csv", rows)
    write_json(FINAL / "strict_action_frontier_summary.json", summary)

    report = [
        "# Strict Action Frontier",
        "",
        "This audit summarizes why the current clean candidates still cannot enter M4/runtime.",
        "",
        f"- clean candidates: `{summary['clean_candidate_count']}`",
        f"- clean handoff candidates: `{summary['clean_handoff_candidate_count']}`",
        f"- safe-good candidates: `{summary['safe_good_candidate_count']}`",
        f"- core-v100-schema-ready clean candidates: `{summary['core_v100_schema_ready_clean_candidate_count']}`",
        f"- strict-action-ready clean candidates: `{summary['strict_action_ready_clean_candidate_count']}`",
        f"- missing prereq counts: `{summary['missing_prereq_counts']}`",
        "",
        "## Track Evidence",
        "",
        f"- JL4 gate pass: `{summary['jl4_gate_pass']}`, identity levels: `{summary['jl4_identity_resolution_level_counts']}`",
        f"- R3 gate pass: `{summary['r3_gate_pass']}`, control margins available: `{summary['r3_control_margins_available']}`",
        f"- F5 gate pass: `{summary['f5_gate_pass']}`, write/cache/current nonempty: `{summary['f5_r_write_cache_nonempty']}/{summary['f5_r_cache_current_nonempty']}/{summary['f5_r_ref_current_nonempty']}`",
        f"- Q2 gate pass: `{summary['q2_gate_pass']}`, true stage pass: `{summary['q2_true_stage_pass']}`, proxy only: `{summary['q2_proxy_only']}`",
        "",
        "## Conclusion",
        "",
        summary["blocked_reason"],
    ]
    write_text(FINAL / "strict_action_frontier_report.md", "\n".join(report))

    recommendation = (
        "Strict-action frontier audit shows all 6 clean candidates are core-v100-schema-ready but 0 are "
        "strict-action-ready. Shared blockers are strict instance identity, query-head controls, F5 "
        "write/cache/current materialization, and Q2 true-stage; Track T still has only 1 clean handoff "
        "target and 5 safe-good controls. Do not run M4/runtime until this frontier changes."
    )
    upsert_section(FINAL / "next_attempt_recommendation.md", "Strict Action Frontier", recommendation)
    append_unique_line(
        FINAL / "remaining_blockers.md",
        "- Strict-action frontier: 6 clean candidates are core-schema-ready, but 0 are strict-action-ready because JL4/R3/F5/Q2 prereqs all remain blocked.",
    )
    append_unique_line(
        FINAL / "failure_report.md",
        "- Strict-action frontier: no clean candidate satisfies instance identity, query-head controls, write-to-use materialization, and Q2 true-stage together.",
    )
    append_unique_line(
        FINAL / "control_gap_report.md",
        "- Strict-action frontier control gap: R3/F5 controls and Q2 true-stage remain unavailable despite core v100-schema readiness.",
    )

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
