#!/usr/bin/env python3
"""Refresh ACL2 v118 current closure after the post-R62 LB-AR R79-R81 repair ladder.

This is a closure/reporting script only. It reads completed artifacts and writes
current decision files; it does not run geometry or synthesize missing metrics.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
BRANCH_ROOT = RESULT_ROOT / "branches"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
MATRIX_PATH = RESULT_ROOT / "V118_CURRENT_BRANCH_COMPLETION_MATRIX.csv"
R82_SUMMARY_PATH = RESULT_ROOT / "V118_POSTFINAL_R82_CURRENT_CLOSURE_SUMMARY.json"
FINAL_DECISION_SUMMARY = RESULT_ROOT / "V118_FINAL_DECISION_SUMMARY.json"
FINAL_DECISION_REPORT = RESULT_ROOT / "V118_FINAL_DECISION_REPORT.md"
FINAL_ANSWER_SUMMARY = RESULT_ROOT / "V118_FINAL_SCIENTIFIC_ANSWER_AUDIT.json"
FINAL_ANSWER_REPORT = RESULT_ROOT / "V118_FINAL_SCIENTIFIC_ANSWER_AUDIT.md"
BOUNDARY_REPORT = RESULT_ROOT / "V118_METHOD_AND_NO_GO_BOUNDARIES.md"

R61_SUMMARY_PATH = (
    RESULT_ROOT
    / "stage4_r61_lingbot_ar_control_dominance_repair_audit/summary/stage4_r61_control_dominance_repair_summary.json"
)
R79_SUMMARY_PATH = (
    RESULT_ROOT
    / "stage4_r79_lingbot_ar_fresh_g1p25_source_subset_audit_08/summary/stage4_r79_g1p25_source_subset_summary.json"
)
R80_SUMMARY_PATH = (
    RESULT_ROOT
    / "stage4_r80_lingbot_ar_fresh_s125_gain_control_matrix_08/summary/stage4_r80_s125_gain_control_summary.json"
)
R81_SUMMARY_PATH = (
    RESULT_ROOT
    / "stage4_r81_lingbot_ar_fresh_s125_query_bias_matrix_08/summary/stage4_r81_s125_query_bias_summary.json"
)

BRANCHES = [
    ("LB-AI", "LingBot", "Anchor initialization", "LB-Anchor"),
    ("LB-AR", "LingBot", "Anchor read", "LB-Anchor"),
    ("LB-LR", "LingBot", "Local read", "LB-Local"),
    ("LB-TA", "LingBot", "Trajectory admission", "LB-Trajectory"),
    ("LB-TR", "LingBot", "Trajectory retrieval", "LB-Trajectory"),
    ("LB-TE", "LingBot", "Retention / eviction", "LB-Trajectory"),
    ("LB-CT", "LingBot", "Compact context token routing", "LB-Local"),
    ("HS-LA", "HorizonStream", "Local Attention", "HS-Local"),
    ("HS-HG", "HorizonStream", "Head reliability", "HS-Local"),
    ("HS-GW", "HorizonStream", "GLA write", "HS-GLA"),
    ("HS-GR", "HorizonStream", "GLA state reliability / retention", "HS-GLA"),
    ("HS-MR", "HorizonStream", "MRT safety/readout", "HS-MRT"),
]

TERMINAL_STATUSES = {
    "COMPLETE_PASS",
    "COMPLETE_GEOMETRY_PASS_SEMANTIC_FAIL",
    "COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS",
    "STRUCTURAL_BLOCKED_AFTER_THREE_REPAIRS",
}

LBAR_R82_EVIDENCE = [
    "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r59_lingbot_ar_forced_action_diagnostic/summary/stage4_r59_forced_action_diagnostic_summary.json",
    "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r61_lingbot_ar_control_dominance_repair_audit/summary/stage4_r61_control_dominance_repair_summary.json",
    "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r68_lingbot_ar_control_safe_boundary_aggregate_audit/summary/stage4_r68_control_safe_boundary_summary.json",
    "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r79_lingbot_ar_fresh_g1p25_source_subset_audit_08/summary/stage4_r79_g1p25_source_subset_summary.json",
    "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r80_lingbot_ar_fresh_s125_gain_control_matrix_08/summary/stage4_r80_s125_gain_control_summary.json",
    "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r81_lingbot_ar_fresh_s125_query_bias_matrix_08/summary/stage4_r81_s125_query_bias_summary.json",
]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def require_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(rel(path))
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_registry_row(row: dict[str, Any]) -> None:
    rows = read_csv(REGISTRY)
    fields: list[str] = []
    for old in rows:
        for key in old:
            if key not in fields:
                fields.append(key)
    for key in row:
        if key not in fields:
            fields.append(key)
    kept = [
        old
        for old in rows
        if not (
            old.get("stage") == row.get("stage")
            and old.get("surface_or_branch") == row.get("surface_or_branch")
            and old.get("artifact") == row.get("artifact")
        )
    ]
    kept.append({key: row.get(key, "") for key in fields})
    with REGISTRY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def branch_summary_path(branch: str) -> Path:
    return BRANCH_ROOT / branch / f"{branch}_DECISION_SUMMARY.json"


def branch_report_path(branch: str) -> Path:
    return BRANCH_ROOT / branch / f"{branch}_REPORT.md"


def best_row(summary: dict[str, Any]) -> dict[str, Any]:
    for key in ("best_candidate_row", "best_row"):
        row = summary.get(key)
        if isinstance(row, dict) and row:
            return dict(row)
    rows = [
        row
        for row in summary.get("result_rows", [])
        if isinstance(row, dict) and row.get("eval_exists") and row.get("ate") is not None
    ]
    if rows:
        return dict(min(rows, key=lambda row: float(row["ate"])))
    return {}


def refresh_lbar_branch() -> dict[str, Any]:
    old = read_json(branch_summary_path("LB-AR"))
    r61 = read_json(R61_SUMMARY_PATH)
    r79 = require_json(R79_SUMMARY_PATH)
    r80 = require_json(R80_SUMMARY_PATH)
    r81 = require_json(R81_SUMMARY_PATH)
    r79_best = best_row(r79)
    r80_best = best_row(r80)
    r81_best = best_row(r81)
    latest_decision = "NO_GO_LB_AR_R82_S125_REPAIR_LADDER_NOT_PROMOTABLE"

    updated = dict(old)
    updated.update(
        {
            "schema": "acl2_v118tf_branch_decision_summary_v5",
            "branch": "LB-AR",
            "model": "LingBot",
            "operation": "Anchor read",
            "surface": "LB-Anchor",
            "status": "COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS",
            "runtime_launched": True,
            "gpu_runtime_launched": True,
            "global_goal_achieved": False,
            "latest_decision": latest_decision,
            "previous_latest_decision_before_r82": old.get("latest_decision", ""),
            "latest_r79_decision": r79.get("stage4_r79_decision", ""),
            "latest_r80_decision": r80.get("stage4_r80_decision", ""),
            "latest_r81_decision": r81.get("stage4_r81_decision", ""),
            "primary_blocker": (
                "Post-R62 LB-AR reopening found a real seq08 source-value local signal, but R79 stayed below "
                "the 0.03 pilot gate, R80 stronger same-source gains did not repair it, and R81 selected-query "
                "attention-logit bias was harmful/control-failed. Prior R59/R61 evidence still blocks generalization "
                "and semantic-specific control dominance."
            ),
            "current_closure_evidence": LBAR_R82_EVIDENCE,
            "post_r62_repair_stages": [
                "Stage4-R63",
                "Stage4-R64",
                "Stage4-R65",
                "Stage4-R66",
                "Stage4-R67",
                "Stage4-R68",
                "Stage4-R70",
                "Stage4-R71",
                "Stage4-R72",
                "Stage4-R73",
                "Stage4-R74",
                "Stage4-R75",
                "Stage4-R76",
                "Stage4-R77",
                "Stage4-R78",
                "Stage4-R79",
                "Stage4-R80",
                "Stage4-R81",
            ],
            "r61_decision": r61.get("stage4_r61_decision", ""),
            "r79_best_method": r79_best.get("method", ""),
            "r79_best_ate": r79_best.get("ate"),
            "r79_best_rel_vs_baseline": r79_best.get("rel_vs_baseline"),
            "r79_geometry_gate": r79_best.get("geometry_gate"),
            "r80_best_method": r80_best.get("method", ""),
            "r80_best_ate": r80_best.get("ate"),
            "r80_best_rel_vs_baseline": r80_best.get("rel_vs_baseline"),
            "r80_geometry_gate": r80_best.get("geometry_gate"),
            "r81_best_method": r81_best.get("method", ""),
            "r81_best_ate": r81_best.get("ate"),
            "r81_best_rel_vs_baseline": r81_best.get("rel_vs_baseline"),
            "r81_geometry_gate": r81_best.get("geometry_gate"),
            "r81_attention_mask_applied_rows": r81_best.get("attention_mask_applied_rows"),
            "boundary": (
                "LB-AR was reopened repeatedly after the older full-plan No-Go. R67/R68 found a useful local/partial "
                "action surface, and R79 found a stronger seq08 source-value row, but R79-R81 did not convert that "
                "signal into a gate-passing and control-safe policy. The branch remains "
                "COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS."
            ),
        }
    )
    outputs = dict(updated.get("outputs", {}))
    outputs.update({"summary": rel(branch_summary_path("LB-AR")), "report": rel(branch_report_path("LB-AR"))})
    updated["outputs"] = outputs
    write_json(branch_summary_path("LB-AR"), updated)
    write_lbar_report(updated)
    return updated


def write_lbar_report(data: dict[str, Any]) -> None:
    lines = [
        "# ACL2 v118-TF LB-AR Current Closure Report",
        "",
        f"- status: `{data['status']}`",
        f"- latest_decision: `{data['latest_decision']}`",
        f"- global_goal_achieved: `{data['global_goal_achieved']}`",
        f"- latest_r79_decision: `{data['latest_r79_decision']}`",
        f"- latest_r80_decision: `{data['latest_r80_decision']}`",
        f"- latest_r81_decision: `{data['latest_r81_decision']}`",
        "",
        "## Current Blocker",
        "",
        data["primary_blocker"],
        "",
        "## Latest Repair Metrics",
        "",
        "| stage | method | ATE | rel_vs_baseline | geometry_gate |",
        "|---|---|---:|---:|---:|",
        f"| R79 | `{data['r79_best_method']}` | {data['r79_best_ate']} | {data['r79_best_rel_vs_baseline']} | {data['r79_geometry_gate']} |",
        f"| R80 | `{data['r80_best_method']}` | {data['r80_best_ate']} | {data['r80_best_rel_vs_baseline']} | {data['r80_geometry_gate']} |",
        f"| R81 | `{data['r81_best_method']}` | {data['r81_best_ate']} | {data['r81_best_rel_vs_baseline']} | {data['r81_geometry_gate']} |",
        "",
        "## Evidence",
        "",
    ]
    for artifact in data["current_closure_evidence"]:
        lines.append(f"- `{artifact}`")
    lines += ["", "## Boundary", "", data["boundary"]]
    branch_report_path("LB-AR").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_matrix_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch, model, operation, surface in BRANCHES:
        data = read_json(branch_summary_path(branch))
        status = data.get("status", "")
        rows.append(
            {
                "schema": "acl2_v118tf_current_branch_completion_row_v3",
                "branch": branch,
                "model": model,
                "operation": operation,
                "surface": surface,
                "status": status,
                "terminal": status in TERMINAL_STATUSES,
                "runtime_launched": bool(data.get("runtime_launched")),
                "gpu_runtime_launched": bool(data.get("gpu_runtime_launched")),
                "latest_decision": data.get("latest_decision", ""),
                "global_goal_achieved": data.get("global_goal_achieved", ""),
                "primary_blocker": data.get("primary_blocker", ""),
                "evidence_count": len(data.get("current_closure_evidence", []) or []),
                "artifact": rel(branch_summary_path(branch)),
            }
        )
    return rows


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("status", ""))] = counts.get(str(row.get("status", "")), 0) + 1
    return counts


def build_current_summary(rows: list[dict[str, Any]], lbar: dict[str, Any]) -> dict[str, Any]:
    nonterminal = [row["branch"] for row in rows if not row["terminal"]]
    pass_branches = [row["branch"] for row in rows if row["status"] == "COMPLETE_PASS"]
    return {
        "schema": "acl2_v118tf_postfinal_r82_current_closure_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "V118_POSTFINAL_R82_ALL_BRANCHES_TERMINAL_NO_GO_GLOBAL_FALSE",
        "taxonomy": "V118_STRUCTURAL_BLOCKERS_REMAIN_AFTER_EXHAUSTIVE_REPAIRS",
        "global_goal_achieved": False,
        "all_branches_terminal": not nonterminal,
        "nonterminal_branches": nonterminal,
        "complete_pass_branches": pass_branches,
        "branch_count": len(rows),
        "runtime_branch_count": sum(1 for row in rows if row["runtime_launched"]),
        "terminal_status_counts": status_counts(rows),
        "lbar_latest_decision": lbar.get("latest_decision", ""),
        "lbar_latest_r79_decision": lbar.get("latest_r79_decision", ""),
        "lbar_latest_r80_decision": lbar.get("latest_r80_decision", ""),
        "lbar_latest_r81_decision": lbar.get("latest_r81_decision", ""),
        "lbar_r82_evidence": LBAR_R82_EVIDENCE,
        "outputs": {
            "summary": rel(R82_SUMMARY_PATH),
            "current_completion_matrix": rel(MATRIX_PATH),
            "final_decision_summary": rel(FINAL_DECISION_SUMMARY),
            "final_decision_report": rel(FINAL_DECISION_REPORT),
            "final_scientific_answer_audit": rel(FINAL_ANSWER_SUMMARY),
            "method_boundaries": rel(BOUNDARY_REPORT),
            "registry": rel(REGISTRY),
            "lbar_summary": rel(branch_summary_path("LB-AR")),
            "lbar_report": rel(branch_report_path("LB-AR")),
        },
        "boundary": (
            "This closure records the current evidence state after the post-R62 LB-AR R79-R81 repair ladder. "
            "It is not a success claim: all branches remain terminal No-Go or structural blocked, and the v118 "
            "semantic-carrier calibration goal remains false."
        ),
    }


def write_final_decision(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    payload = {
        "schema": "acl2_v118tf_current_final_decision_summary_v2",
        "created_at_utc": summary["created_at_utc"],
        "decision": summary["decision"],
        "taxonomy": summary["taxonomy"],
        "global_goal_achieved": summary["global_goal_achieved"],
        "all_branches_terminal": summary["all_branches_terminal"],
        "branch_count": summary["branch_count"],
        "runtime_branch_count": summary["runtime_branch_count"],
        "complete_pass_branches": summary["complete_pass_branches"],
        "nonterminal_branches": summary["nonterminal_branches"],
        "terminal_status_counts": summary["terminal_status_counts"],
        "latest_lbar_decision": summary["lbar_latest_decision"],
        "outputs": summary["outputs"],
        "boundary": summary["boundary"],
    }
    write_json(FINAL_DECISION_SUMMARY, payload)
    lines = [
        "# ACL2 v118 Final Decision Report",
        "",
        f"- decision: `{payload['decision']}`",
        f"- taxonomy: `{payload['taxonomy']}`",
        f"- global_goal_achieved: `{payload['global_goal_achieved']}`",
        f"- all_branches_terminal: `{payload['all_branches_terminal']}`",
        f"- complete_pass_branches: `{','.join(payload['complete_pass_branches'])}`",
        f"- nonterminal_branches: `{','.join(payload['nonterminal_branches'])}`",
        "",
        "## Current Branch Matrix",
        "",
        "| branch | status | runtime | latest decision | blocker |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['branch']}` | `{row['status']}` | {row['runtime_launched']} | "
            f"`{row['latest_decision']}` | {row['primary_blocker']} |"
        )
    lines += ["", "## Boundary", "", payload["boundary"]]
    FINAL_DECISION_REPORT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def compact_branch_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["branch", "operation", "status", "latest_decision", "primary_blocker"]
    return [{key: row.get(key, "") for key in keys} for row in rows]


def build_final_answers(rows: list[dict[str, Any]], summary: dict[str, Any], lbar: dict[str, Any]) -> dict[str, Any]:
    runtime_no_go = [
        str(row["branch"])
        for row in rows
        if row["status"] in {"COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS", "COMPLETE_GEOMETRY_PASS_SEMANTIC_FAIL"}
    ]
    structural = [str(row["branch"]) for row in rows if row["status"] == "STRUCTURAL_BLOCKED_AFTER_THREE_REPAIRS"]
    answers = [
        {
            "id": "q1_lingbot_operation_fit",
            "answer": (
                "No LingBot operation is promotable as a stable semantic-management success in v118. "
                "The reopened LB-AR path still fails after R79-R81: R79 reached rel_vs_baseline="
                f"{lbar.get('r79_best_rel_vs_baseline')} below the 0.03 gate, R80 did not repair it with stronger "
                "same-source gains, and R81 selected-query attention bias was harmful/control-failed."
            ),
            "evidence": [
                "V118_CURRENT_BRANCH_COMPLETION_MATRIX.csv",
                rel(R79_SUMMARY_PATH),
                rel(R80_SUMMARY_PATH),
                rel(R81_SUMMARY_PATH),
            ],
        },
        {
            "id": "q2_append_only_cache_reliability",
            "answer": (
                "v118 did not validate a promotable append-only cache reliability definition. Runtime LingBot cache "
                "branches either failed controls/default baselines or remained structurally blocked for true internal "
                "candidate/reliability rows."
            ),
            "evidence": ["LB-TA/LB-TE/LB-AR matrix rows", "LB-LR/LB-CT structural rows"],
        },
        {
            "id": "q3_horizonstream_recal3r_calibration",
            "answer": (
                "No HorizonStream branch produced a promoted full-KITTI semantic-specific geometry improvement. "
                "HS-LA and HS-HG are runtime No-Go; HS-GW, HS-GR, and HS-MR remain structural/repair blocked."
            ),
            "evidence": ["HS-LA/HS-HG/HS-GW/HS-GR/HS-MR matrix rows"],
        },
        {
            "id": "q4_persistent_dynamic_extra_info",
            "answer": (
                "Persistent/dynamic identity carries local signals, but not a promotable signal beyond generic controls. "
                "R79 is the strongest latest source-value row, yet it stays below the gate; R80 and R81 fail to convert "
                "it into a control-safe promoted policy."
            ),
            "evidence": [rel(R79_SUMMARY_PATH), rel(R80_SUMMARY_PATH), rel(R81_SUMMARY_PATH)],
        },
        {
            "id": "q5_stable_operation_specific_full_geometry",
            "answer": (
                "None verified. The current closure has complete_pass_branches=[] and global_goal_achieved=False. "
                "No operation-specific action survived the full plan gates as a stable semantic-carrier calibration success."
            ),
            "evidence": [rel(R82_SUMMARY_PATH), "V118_CURRENT_BRANCH_COMPLETION_MATRIX.csv"],
        },
        {
            "id": "q6_falsified_vs_structural",
            "answer": (
                f"Runtime No-Go branches: {', '.join(runtime_no_go)}. "
                f"Structural/repair-blocked branches: {', '.join(structural)}."
            ),
            "evidence": [compact_branch_table(rows)],
        },
    ]
    return {
        "schema": "acl2_v118tf_r82_final_scientific_answer_audit_v1",
        "created_at_utc": summary["created_at_utc"],
        "decision": "V118_R82_FINAL_SCIENTIFIC_ANSWERS_COMPLETE_NO_GO",
        "global_goal_achieved": False,
        "all_branches_terminal": summary["all_branches_terminal"],
        "complete_pass_branches": summary["complete_pass_branches"],
        "nonterminal_branches": summary["nonterminal_branches"],
        "branch_count": summary["branch_count"],
        "scientific_answers": answers,
        "boundary": summary["boundary"],
        "outputs": {
            "summary": rel(FINAL_ANSWER_SUMMARY),
            "report": rel(FINAL_ANSWER_REPORT),
            "boundary_report": rel(BOUNDARY_REPORT),
            "current_closure_summary": rel(R82_SUMMARY_PATH),
        },
    }


def write_final_answer_report(payload: dict[str, Any]) -> None:
    lines = [
        "# ACL2 v118 Final Scientific Answer Audit",
        "",
        f"- decision: `{payload['decision']}`",
        f"- global_goal_achieved: `{payload['global_goal_achieved']}`",
        f"- all_branches_terminal: `{payload['all_branches_terminal']}`",
        f"- complete_pass_branches: `{','.join(payload['complete_pass_branches'])}`",
        "",
        "## Answers",
        "",
    ]
    for answer in payload["scientific_answers"]:
        lines += [f"### {answer['id']}", "", answer["answer"], "", "Evidence:"]
        for item in answer["evidence"]:
            if isinstance(item, str):
                lines.append(f"- {item}")
            else:
                lines.append(f"- `{json.dumps(item, sort_keys=True, ensure_ascii=False)}`")
        lines.append("")
    lines += ["## Boundary", "", payload["boundary"]]
    FINAL_ANSWER_REPORT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_boundary_report(summary: dict[str, Any], rows: list[dict[str, Any]], final_answers: dict[str, Any]) -> None:
    lines = [
        "# ACL2 v118-TF Method And No-Go Boundaries",
        "",
        "- R82 supersedes R60/R62 wording by incorporating post-R62 LB-AR R79-R81 repair evidence.",
        "- Runtime evidence exists only for the branch/stage artifacts listed in the current completion matrix.",
        "- No missing runtime metric is backfilled. Structural branches keep structural blocker status.",
        "- LB-AR remains No-Go after R79/R80/R81: the best source-value local gain stays below 3%, stronger gains fail, and selected-query logit bias is harmful/control-failed.",
        "- The global v118 semantic-carrier calibration goal remains false.",
        "",
        "## Current Branch Matrix",
        "",
        "| branch | status | runtime | latest decision | blocker |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['branch']}` | `{row['status']}` | {row['runtime_launched']} | "
            f"`{row['latest_decision']}` | {row['primary_blocker']} |"
        )
    lines += [
        "",
        "## R82 Current Closure Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
        "## R82 Final Scientific Answer Audit",
        "",
        f"- decision: `{final_answers['decision']}`",
        f"- global_goal_achieved: `{final_answers['global_goal_achieved']}`",
        f"- artifact: `{final_answers['outputs']['report']}`",
    ]
    for answer in final_answers["scientific_answers"]:
        lines.append(f"- {answer['id']}: {answer['answer']}")
    BOUNDARY_REPORT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    lbar = refresh_lbar_branch()
    rows = build_matrix_rows()
    summary = build_current_summary(rows, lbar)
    final_answers = build_final_answers(rows, summary, lbar)

    write_csv(MATRIX_PATH, rows)
    write_json(R82_SUMMARY_PATH, summary)
    write_final_decision(summary, rows)
    write_json(FINAL_ANSWER_SUMMARY, final_answers)
    write_final_answer_report(final_answers)
    write_boundary_report(summary, rows, final_answers)

    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "PostFinal-R82-LBAR-R81",
            "surface_or_branch": "ALL/LB-AR",
            "status": summary["decision"],
            "artifact": rel(R82_SUMMARY_PATH),
            "notes": "Refreshes current closure after LB-AR R79-R81 repair ladder; global goal remains false.",
        }
    )
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "PostFinal-R82-ScientificAnswerAudit",
            "surface_or_branch": "ALL",
            "status": final_answers["decision"],
            "artifact": rel(FINAL_ANSWER_SUMMARY),
            "notes": "Final six-answer audit refreshed after LB-AR R79-R81; No-Go, no new geometry.",
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
