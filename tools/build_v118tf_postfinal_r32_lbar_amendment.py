#!/usr/bin/env python3
"""Build post-final amendment for LB-AI/LB-AR runtime closures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
OUT_JSON = RESULT_ROOT / "V118_POSTFINAL_R32_LBAR_RUNTIME_AMENDMENT_SUMMARY.json"
OUT_CSV = RESULT_ROOT / "V118_POSTFINAL_R32_BRANCH_STATUS_AMENDMENT.csv"
OUT_REPORT = RESULT_ROOT / "V118_POSTFINAL_R32_LBAR_RUNTIME_AMENDMENT_REPORT.md"

BRANCHES = [
    "LB-AI",
    "LB-AR",
    "LB-LR",
    "LB-TA",
    "LB-TR",
    "LB-TE",
    "LB-CT",
    "HS-LA",
    "HS-HG",
    "HS-GW",
    "HS-GR",
    "HS-MR",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    with REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def main() -> None:
    branch_rows: list[dict[str, Any]] = []
    runtime_branch_count = 0
    no_go_runtime_branches = []
    for branch in BRANCHES:
        path = RESULT_ROOT / f"branches/{branch}/{branch}_DECISION_SUMMARY.json"
        data = read_json(path)
        runtime_launched = bool(data.get("runtime_launched") or data.get("gpu_runtime_launched"))
        if runtime_launched:
            runtime_branch_count += 1
            no_go_runtime_branches.append(branch)
        branch_rows.append(
            {
                "branch": branch,
                "status": data.get("status", ""),
                "runtime_launched": runtime_launched,
                "gpu_runtime_launched": bool(data.get("gpu_runtime_launched")),
                "global_goal_achieved": data.get("global_goal_achieved", ""),
                "primary_blocker": data.get("primary_blocker", ""),
                "artifact": rel(path),
            }
        )
    summary = {
        "schema": "acl2_v118tf_postfinal_r32_lbar_runtime_amendment_summary_v1",
        "previous_final_artifact": rel(RESULT_ROOT / "V118_FINAL_DECISION_SUMMARY.json"),
        "decision": "V118_POSTFINAL_RUNTIME_AMENDED_LBAI_LBAR_COMPLETE_NO_GO_GLOBAL_FALSE",
        "global_goal_achieved": False,
        "all_branches_terminal": True,
        "stage4_runtime_launched_after_original_final": True,
        "runtime_branch_count_after_original_final": runtime_branch_count,
        "runtime_no_go_branches_after_original_final": no_go_runtime_branches,
        "branch_status_rows": branch_rows,
        "outputs": {
            "summary": rel(OUT_JSON),
            "branch_status_csv": rel(OUT_CSV),
            "report": rel(OUT_REPORT),
        },
        "boundary": (
            "This amendment supersedes the stale original-final claim that no Stage4 runtime was launched. "
            "It does not claim v118 success: LB-AI and LB-AR both ran runtime follow-ups and closed as No-Go; "
            "remaining branch decisions stay terminal according to their branch artifacts/post-final amendments."
        ),
    }
    write_json(OUT_JSON, summary)
    write_csv(OUT_CSV, branch_rows)
    lines = [
        "# ACL2 v118-TF Post-Final R32 Runtime Amendment",
        "",
        f"- decision: `{summary['decision']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- stage4_runtime_launched_after_original_final: `{summary['stage4_runtime_launched_after_original_final']}`",
        f"- runtime_branch_count_after_original_final: `{summary['runtime_branch_count_after_original_final']}`",
        "",
        "## Branch Status",
        "",
        "| branch | status | runtime launched | blocker |",
        "|---|---|---:|---|",
    ]
    for row in branch_rows:
        lines.append(
            f"| `{row['branch']}` | `{row['status']}` | {row['runtime_launched']} | `{row['primary_blocker']}` |"
        )
    lines += ["", "## Boundary", "", summary["boundary"]]
    OUT_REPORT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "PostFinal-R32-LBAR",
            "surface_or_branch": "LB-AI/LB-AR",
            "status": summary["decision"],
            "artifact": rel(OUT_JSON),
            "notes": "Corrects stale original final no-runtime wording after LB-AI R27 and LB-AR R28-R32 runtime closures; global goal remains false",
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
