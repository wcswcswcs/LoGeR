#!/usr/bin/env python3
"""Refresh ACL2 v118 current closure after LB-AR R59 diagnostics.

This is an audit/closure builder only. It does not synthesize metrics or run
geometry. It updates LB-AR branch closure to include the R51-R59 reopening and
writes a current branch matrix that supersedes stale post-final R33 wording.
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
SUMMARY_PATH = RESULT_ROOT / "V118_POSTFINAL_R60_CURRENT_CLOSURE_SUMMARY.json"
MATRIX_PATH = RESULT_ROOT / "V118_CURRENT_BRANCH_COMPLETION_MATRIX.csv"
REPORT_PATH = RESULT_ROOT / "V118_METHOD_AND_NO_GO_BOUNDARIES.md"

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

LBAR_R59_EVIDENCE = [
    "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r51_lingbot_ar_fresh_r47_policy_validation/summary/stage4_r51_fresh_policy_summary.json",
    "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r52_lingbot_ar_fresh_r47_policy_validation_06_07/summary/stage4_r51_fresh_policy_summary.json",
    "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r55_lingbot_ar_fresh_guarded_risk_policy_validation_06_07_gpu/summary/stage4_r51_fresh_policy_summary.json",
    "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r58_lingbot_ar_fresh_guarded_risk_policy_validation_08_09/summary/stage4_r51_fresh_policy_summary.json",
    "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r59_lingbot_ar_forced_action_diagnostic/summary/stage4_r59_forced_action_diagnostic_summary.json",
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


def branch_summary_path(branch: str) -> Path:
    return BRANCH_ROOT / branch / f"{branch}_DECISION_SUMMARY.json"


def branch_report_path(branch: str) -> Path:
    return BRANCH_ROOT / branch / f"{branch}_REPORT.md"


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


def refresh_lbar_branch() -> dict[str, Any]:
    path = branch_summary_path("LB-AR")
    old = read_json(path)
    r59 = read_json(RESULT_ROOT / "stage4_r59_lingbot_ar_forced_action_diagnostic/summary/stage4_r59_forced_action_diagnostic_summary.json")
    if not r59:
        raise FileNotFoundError("R59 summary missing; run build_v118tf_stage4_r59_lingbot_ar_forced_action_diagnostic.py first")
    updated = dict(old)
    old_decision = old.get("latest_decision", "")
    updated.update(
        {
            "schema": "acl2_v118tf_branch_decision_summary_v4",
            "branch": "LB-AR",
            "model": "LingBot",
            "operation": "Anchor read",
            "surface": "LB-Anchor",
            "status": "COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS",
            "runtime_launched": True,
            "gpu_runtime_launched": True,
            "global_goal_achieved": False,
            "latest_decision": r59.get("stage4_r59_decision", ""),
            "previous_latest_decision_before_r59": old_decision,
            "primary_blocker": (
                "R59 fresh forced-action diagnostic blocks promotion of LB-AR stable_guarded_risk: "
                "R58 is complete/action-fidelity-clean but selected actions lose to random/risk controls on 08/09."
            ),
            "repair_attempt_count": 11,
            "stage4_runtime_stages": [
                "Stage4-R28",
                "Stage4-R29",
                "Stage4-R30",
                "Stage4-R31",
                "Stage4-R32",
                "Stage4-R40",
                "Stage4-R41",
                "Stage4-R42",
                "Stage4-R43",
                "Stage4-R44",
                "Stage4-R45",
                "Stage4-R46",
                "Stage4-R47",
                "Stage4-R51",
                "Stage4-R52",
                "Stage4-R55",
                "Stage4-R58",
                "Stage4-R59",
            ],
            "current_closure_evidence": LBAR_R59_EVIDENCE,
            "r59_sequence_winner_actions": r59.get("sequence_winner_actions", {}),
            "r59_selected_action_mismatches": r59.get("stable_guarded_selected_action_mismatch_rows", []),
            "boundary": (
                "LB-AR was reopened after the older R32 closure; R51 gave a local fresh pass, but R52/R55/R58/R59 "
                "show the sign/ratio rule does not generalize and controls can beat selected semantic actions. "
                "The branch remains COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS."
            ),
        }
    )
    outputs = dict(updated.get("outputs", {}))
    outputs.update({"summary": rel(path), "report": rel(branch_report_path("LB-AR"))})
    updated["outputs"] = outputs
    write_json(path, updated)

    lines = [
        "# ACL2 v118-TF LB-AR Current Closure Report",
        "",
        f"- status: `{updated['status']}`",
        f"- latest_decision: `{updated['latest_decision']}`",
        f"- previous_latest_decision_before_r59: `{old_decision}`",
        f"- global_goal_achieved: `{updated['global_goal_achieved']}`",
        "",
        "## Current Blocker",
        "",
        updated["primary_blocker"],
        "",
        "## R59 Winner Actions",
        "",
        "```json",
        json.dumps(updated["r59_sequence_winner_actions"], indent=2, sort_keys=True),
        "```",
        "",
        "## R59 Selected-Action Mismatches",
        "",
        "```json",
        json.dumps(updated["r59_selected_action_mismatches"], indent=2, sort_keys=True),
        "```",
        "",
        "## Evidence",
        "",
    ]
    for artifact in LBAR_R59_EVIDENCE:
        lines.append(f"- `{artifact}`")
    lines += ["", "## Boundary", "", updated["boundary"]]
    branch_report_path("LB-AR").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return updated


def build_matrix_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch, model, operation, surface in BRANCHES:
        data = read_json(branch_summary_path(branch))
        status = data.get("status", "")
        rows.append(
            {
                "schema": "acl2_v118tf_current_branch_completion_row_v2",
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


def write_boundary_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ACL2 v118-TF Method And No-Go Boundaries",
        "",
        "- The original structural-only final is superseded by current post-final closure artifacts.",
        "- R60 supersedes R33/R32 wording for LB-AR by incorporating R51-R59 fresh forced-action evidence.",
        "- Runtime evidence exists only for the branch/stage artifacts listed in the current completion matrix.",
        "- No missing runtime metric is backfilled. Structural branches keep structural blocker status.",
        "- LB-AR stable_guarded_risk is No-Go after fresh 08/09 action-surface diagnostic: selected actions lose to random/risk controls.",
        "- LB-AI, LB-AR, LB-TA, LB-TE, HS-LA, and HS-HG are runtime No-Go after required variants/control checks.",
        "- LB-LR, LB-CT, LB-TR, HS-GW, HS-GR, and HS-MR remain structural/repair blocked as recorded in branch artifacts.",
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
    lines += ["", "## Summary", "", "```json", json.dumps(summary, indent=2, sort_keys=True), "```"]
    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    refreshed_lbar = refresh_lbar_branch()
    rows = build_matrix_rows()
    nonterminal = [row["branch"] for row in rows if not row["terminal"]]
    pass_branches = [row["branch"] for row in rows if row["status"] == "COMPLETE_PASS"]
    summary = {
        "schema": "acl2_v118tf_postfinal_r60_current_closure_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "V118_POSTFINAL_R60_ALL_BRANCHES_TERMINAL_NO_GO_GLOBAL_FALSE",
        "global_goal_achieved": False,
        "all_branches_terminal": not nonterminal,
        "nonterminal_branches": nonterminal,
        "complete_pass_branches": pass_branches,
        "branch_count": len(rows),
        "runtime_branch_count": sum(1 for row in rows if row["runtime_launched"]),
        "lbar_latest_decision": refreshed_lbar.get("latest_decision", ""),
        "lbar_r59_evidence": LBAR_R59_EVIDENCE,
        "outputs": {
            "summary": rel(SUMMARY_PATH),
            "matrix": rel(MATRIX_PATH),
            "boundary_report": rel(REPORT_PATH),
            "lbar_summary": rel(branch_summary_path("LB-AR")),
            "lbar_report": rel(branch_report_path("LB-AR")),
        },
        "boundary": (
            "This closure records the current evidence state after LB-AR R59. It is not a success claim: "
            "all branches are terminal No-Go or structural blocked, and the v118 global goal remains false."
        ),
    }
    write_csv(MATRIX_PATH, rows)
    write_json(SUMMARY_PATH, summary)
    write_boundary_report(summary, rows)
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "PostFinal-R60-LBAR-R59",
            "surface_or_branch": "ALL/LB-AR",
            "status": summary["decision"],
            "artifact": rel(SUMMARY_PATH),
            "notes": "Refreshes current closure after LB-AR R59 fresh forced-action diagnostic; global goal remains false.",
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
