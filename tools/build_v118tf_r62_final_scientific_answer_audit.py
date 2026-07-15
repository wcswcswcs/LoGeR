#!/usr/bin/env python3
"""Build the final ACL2 v118 scientific-answer audit.

This is a closure/reporting script only. It reads the current R60/R61 closure
artifacts and answers the six scientific questions required by the v118 plan.
It does not run geometry, compute new metrics, or backfill missing evidence.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
MATRIX_PATH = RESULT_ROOT / "V118_CURRENT_BRANCH_COMPLETION_MATRIX.csv"
R60_SUMMARY_PATH = RESULT_ROOT / "V118_POSTFINAL_R60_CURRENT_CLOSURE_SUMMARY.json"
R61_SUMMARY_PATH = (
    RESULT_ROOT
    / "stage4_r61_lingbot_ar_control_dominance_repair_audit/summary/stage4_r61_control_dominance_repair_summary.json"
)
R61_ROWS_PATH = (
    RESULT_ROOT
    / "stage4_r61_lingbot_ar_control_dominance_repair_audit/summary/stage4_r61_control_dominance_rows.csv"
)
SUMMARY_PATH = RESULT_ROOT / "V118_FINAL_SCIENTIFIC_ANSWER_AUDIT.json"
REPORT_PATH = RESULT_ROOT / "V118_FINAL_SCIENTIFIC_ANSWER_AUDIT.md"
BOUNDARY_REPORT = RESULT_ROOT / "V118_METHOD_AND_NO_GO_BOUNDARIES.md"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_registry_row(row: dict[str, Any]) -> None:
    rows = read_csv(REGISTRY) if REGISTRY.exists() else []
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


def compact_branch_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keys = ["branch", "operation", "status", "latest_decision", "primary_blocker"]
    return [{key: row.get(key, "") for key in keys} for row in rows]


def make_answers(matrix_rows: list[dict[str, str]], r61_summary: dict[str, Any]) -> list[dict[str, Any]]:
    branch_table = compact_branch_table(matrix_rows)
    runtime_no_go = [
        row["branch"]
        for row in matrix_rows
        if row["status"] in {"COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS", "COMPLETE_GEOMETRY_PASS_SEMANTIC_FAIL"}
    ]
    structural = [row["branch"] for row in matrix_rows if row["status"] == "STRUCTURAL_BLOCKED_AFTER_THREE_REPAIRS"]
    return [
        {
            "id": "q1_lingbot_operation_fit",
            "question": "LingBot 中，语义最适合管理 Anchor initialization、Local read、Trajectory admission、retrieval 还是 retention?",
            "answer": (
                "No LingBot operation is promotable as a stable semantic-management success in v118. "
                "LB-AI, LB-AR, LB-TA, and LB-TE reached runtime No-Go; LB-LR and LB-CT remained structural; "
                "LB-TR had retrieval signal relative to controls but failed/default-baseline or exact-policy requirements. "
                "LB-AR R61 preserves only one local seq07 pass, which is insufficient for promotion."
            ),
            "evidence": [
                "V118_CURRENT_BRANCH_COMPLETION_MATRIX.csv rows for LB-AI/LB-AR/LB-LR/LB-TA/LB-TR/LB-TE/LB-CT",
                "R61 semantic_claim_pass_count=1 and rejected_sequences=[06,08,09]",
            ],
        },
        {
            "id": "q2_append_only_cache_reliability",
            "question": "对 append-only cache，memory reliability 应由什么内部量定义?",
            "answer": (
                "v118 did not validate a promotable append-only cache reliability definition. "
                "Semantic persistence/stable-risk proxies and source-frame aggregates were repeatedly rejected as insufficient; "
                "true lifecycle/internal reliability either failed controls or remained unavailable for structural branches."
            ),
            "evidence": [
                "LB-TA/LB-TE runtime No-Go rows",
                "LB-LR/LB-CT structural blocker rows",
                "current branch blockers describe missing true internal candidate or reliability rows",
            ],
        },
        {
            "id": "q3_horizonstream_recal3r_calibration",
            "question": "对 HorizonStream recurrent state，ReCal3R-style candidate/reliability calibration 是否有效?",
            "answer": (
                "No verified success. HS-LA and HS-HG reached runtime No-Go; HS-GW, HS-GR, and HS-MR remained structural/"
                "repair blocked or deterministic-OOM/no-pilot-pass. No HorizonStream branch produced a promoted full-KITTI "
                "semantic-specific geometry improvement."
            ),
            "evidence": [
                "V118_CURRENT_BRANCH_COMPLETION_MATRIX.csv rows for HS-LA/HS-HG/HS-GW/HS-GR/HS-MR",
                "V118_METHOD_AND_NO_GO_BOUNDARIES.md current branch matrix",
            ],
        },
        {
            "id": "q4_persistent_dynamic_extra_info",
            "question": "persistent landmark 与 dynamic object identity 是否提供超过 class-level risk 和 generic controls 的额外信息?",
            "answer": (
                "Not in a promotable way. Several branches found action surfaces or local signals, but matched random, reverse, "
                "generic, or control actions either matched/exceeded candidates or exposed non-semantic action sensitivity. "
                "R61 explicitly shows random/control winners on fresh LB-AR seq08/seq09."
            ),
            "evidence": [
                "R61 control_or_random_winner_sequences=[08,09]",
                "R60 matrix marks zero COMPLETE_PASS branches",
            ],
        },
        {
            "id": "q5_stable_operation_specific_full_geometry",
            "question": "哪种 operation-specific action 能在相同预算下稳定改善 full KITTI geometry?",
            "answer": (
                "None verified in v118. R60 records complete_pass_branches=[] and global_goal_achieved=False. "
                "The R61 oracle best-action median rel is explicitly an upper bound from observed outcomes, not a deployable rule."
            ),
            "evidence": [
                "V118_POSTFINAL_R60_CURRENT_CLOSURE_SUMMARY.json complete_pass_branches=[]",
                "R61 oracle_boundary says the oracle upper bound is not deployable or semantic-specific",
            ],
        },
        {
            "id": "q6_falsified_vs_structural",
            "question": "哪些方向被彻底证伪，哪些只是代码结构受限?",
            "answer": (
                f"Runtime No-Go branches: {', '.join(runtime_no_go)}. "
                f"Structural/repair-blocked branches: {', '.join(structural)}. "
                "The detailed per-branch blocker text is preserved in the completion matrix and branch reports."
            ),
            "evidence": ["V118_CURRENT_BRANCH_COMPLETION_MATRIX.csv", branch_table],
        },
    ]


def write_report(payload: dict[str, Any]) -> None:
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
        lines += [
            f"### {answer['id']}",
            "",
            f"Question: {answer['question']}",
            "",
            f"Answer: {answer['answer']}",
            "",
            "Evidence:",
        ]
        for item in answer["evidence"]:
            if isinstance(item, str):
                lines.append(f"- {item}")
            else:
                lines.append(f"- `{json.dumps(item, sort_keys=True, ensure_ascii=False)}`")
        lines.append("")
    lines += [
        "## Boundary",
        "",
        payload["boundary"],
        "",
        "## Outputs",
        "",
        f"- summary: `{payload['outputs']['summary']}`",
        f"- report: `{payload['outputs']['report']}`",
        f"- boundary_report: `{payload['outputs']['boundary_report']}`",
    ]
    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def update_boundary_report(payload: dict[str, Any]) -> None:
    if not BOUNDARY_REPORT.exists():
        return
    text = BOUNDARY_REPORT.read_text(encoding="utf-8")
    marker = "\n## R62 Final Scientific Answer Audit\n"
    base = text.split(marker)[0].rstrip()
    lines = [
        "",
        "## R62 Final Scientific Answer Audit",
        "",
        f"- decision: `{payload['decision']}`",
        f"- global_goal_achieved: `{payload['global_goal_achieved']}`",
        "- The six plan-level scientific questions are answered in the final audit artifact.",
        f"- artifact: `{payload['outputs']['report']}`",
    ]
    for answer in payload["scientific_answers"]:
        lines.append(f"- {answer['id']}: {answer['answer']}")
    BOUNDARY_REPORT.write_text(base + marker + "\n".join(lines[2:]).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    matrix_rows = read_csv(MATRIX_PATH)
    r60_summary = read_json(R60_SUMMARY_PATH)
    r61_summary = read_json(R61_SUMMARY_PATH)
    r61_rows = read_csv(R61_ROWS_PATH)
    answers = make_answers(matrix_rows, r61_summary)
    payload = {
        "schema": "acl2_v118tf_r62_final_scientific_answer_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "V118_R62_FINAL_SCIENTIFIC_ANSWERS_COMPLETE_NO_GO",
        "global_goal_achieved": False,
        "all_branches_terminal": bool(r60_summary.get("all_branches_terminal")),
        "complete_pass_branches": r60_summary.get("complete_pass_branches", []),
        "nonterminal_branches": r60_summary.get("nonterminal_branches", []),
        "branch_count": len(matrix_rows),
        "r61_semantic_claim_pass_count": r61_summary.get("semantic_claim_pass_count"),
        "r61_rejected_sequences": r61_summary.get("rejected_sequences"),
        "r61_control_or_random_winner_sequences": r61_summary.get("control_or_random_winner_sequences"),
        "r61_row_count": len(r61_rows),
        "scientific_answers": answers,
        "boundary": (
            "This final audit answers the plan questions from current artifacts only. It is a No-Go scientific closure: "
            "the v118 execution is complete, but the semantic-carrier calibration hypothesis is not achieved."
        ),
        "outputs": {
            "summary": rel(SUMMARY_PATH),
            "report": rel(REPORT_PATH),
            "boundary_report": rel(BOUNDARY_REPORT),
        },
    }
    write_json(SUMMARY_PATH, payload)
    write_report(payload)
    update_boundary_report(payload)
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "PostFinal-R62-ScientificAnswerAudit",
            "surface_or_branch": "ALL",
            "status": payload["decision"],
            "artifact": rel(SUMMARY_PATH),
            "notes": "Final six-question scientific-answer audit from R60/R61 current closure; No-Go, no new geometry.",
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
