#!/usr/bin/env python3
"""Build current ACL2 v118-TF closure artifacts after post-final runtime follow-ups."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
BRANCH_ROOT = RESULT_ROOT / "branches"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"

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
FORBIDDEN_STATUSES = {"not_run", "skipped_due_to_other_branch", "pending_without_reason", ""}

EVIDENCE = {
    "HS-LA": {
        "status": "COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS",
        "runtime_launched": True,
        "gpu_runtime_launched": True,
        "repair_attempt_count": 7,
        "latest_decision": "HS_LA_STAGE4_NO_PILOT_GATE_PASS_CONTROLS_RECORDED",
        "primary_blocker": "HLA1-HLA5 selected-query candidates and same-magnitude controls completed; zero candidate pilot gates passed and random controls matched or exceeded candidates.",
        "evidence": [
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_hs_la_control_matrix/summary/hs_la_stage4_control_matrix_summary.json"
        ],
    },
    "HS-HG": {
        "status": "COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS",
        "runtime_launched": True,
        "gpu_runtime_launched": True,
        "repair_attempt_count": 5,
        "latest_decision": "V118_R4_HS_HG_STAGE4_NO_GO_REST_STRUCTURAL_BLOCKERS",
        "primary_blocker": "Stage3-R4 reopened HS-HG, but all four Stage4-R4 head-gate variants failed the geometry gate.",
        "evidence": [
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/V118_POSTFINAL_R4_AMENDMENT_SUMMARY.json"
        ],
    },
    "HS-GW": {
        "status": "STRUCTURAL_BLOCKED_AFTER_THREE_REPAIRS",
        "runtime_launched": True,
        "gpu_runtime_launched": True,
        "repair_attempt_count": 6,
        "latest_decision": "HGW3_REDUCED_FULL_CLEAN_GPU_RETRY_DETERMINISTIC_OOM",
        "primary_blocker": "HS-GW full GLA write matrix OOMed; reduced HGW5 did not pass pilot gate and HGW3 clean-GPU retry remained deterministic OOM.",
        "evidence": [
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r12_hs_gla_oom_repair/summary/hs_gla_oom_repair_summary.json",
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r13_hs_gla_hgw3_serial_retry/summary/hgw3_clean_gpu_retry_summary.json",
        ],
    },
    "HS-GR": {
        "status": "STRUCTURAL_BLOCKED_AFTER_THREE_REPAIRS",
        "runtime_launched": True,
        "gpu_runtime_launched": True,
        "repair_attempt_count": 6,
        "latest_decision": "HS_GLA_REPAIR_RUNTIME_INCOMPLETE",
        "primary_blocker": "HS-GR direct gamma/fixed-reference reliability stayed unavailable; reduced HGR4 candidate/control completed but did not pass the pilot gate.",
        "evidence": [
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r12_hs_gla_oom_repair/summary/hs_gla_oom_repair_summary.json"
        ],
    },
    "LB-TA": {
        "status": "COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS",
        "runtime_launched": True,
        "gpu_runtime_launched": True,
        "repair_attempt_count": 6,
        "latest_decision": "TA_SOFT_CONTEXT_GATE_CONTROL_OR_BASELINE_NO_GO",
        "primary_blocker": "R22 hard no-append harmed seq02 and failed controls; R33 soft context-token gate had action fidelity but produced zero geometry delta versus default and matched controls.",
        "evidence": [
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r22_lingbot_ta_hard_noappend/summary/stage4_r22_lingbot_ta_hard_noappend_summary.json",
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r33_lingbot_ta_soft_context_gate/summary/stage4_r33_lingbot_ta_soft_context_gate_summary.json",
        ],
    },
    "LB-TR": {
        "status": "STRUCTURAL_BLOCKED_AFTER_THREE_REPAIRS",
        "runtime_launched": True,
        "gpu_runtime_launched": True,
        "repair_attempt_count": 7,
        "latest_decision": "FRAME_SUPPORTED_BRIDGE_PASS_EXACT_TRAJECTORY_SEMANTIC_POLICY_BLOCKED",
        "primary_blocker": "TR topK/diversity action variants failed default-baseline promotion; exact trajectory semantic policy stayed blocked by missing direct token/object semantic fields on trajectory-special entries.",
        "evidence": [
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r17_lingbot_retrieval_retention_action/summary/stage4_r17_lingbot_action_summary.json",
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r18_lingbot_stage4_variant_expansion/summary/stage4_r18_lingbot_variant_summary.json",
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r19_lingbot_tr_topk_calibration/summary/stage4_r19_lingbot_tr_topk_summary.json",
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r20_lingbot_semantic_bridge_audit/summary/stage4_r20_lingbot_semantic_bridge_audit_summary.json",
        ],
    },
    "LB-TE": {
        "status": "COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS",
        "runtime_launched": True,
        "gpu_runtime_launched": True,
        "repair_attempt_count": 6,
        "latest_decision": "TE3_SEMANTIC_EVICTION_CONTROL_OR_BASELINE_NO_GO",
        "primary_blocker": "TE1/TE2/TE3 retention-eviction variants completed; TE3 had semantic support and action fidelity but failed matched-random/default-baseline gates.",
        "evidence": [
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r17_lingbot_retrieval_retention_action/summary/stage4_r17_lingbot_action_summary.json",
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r18_lingbot_stage4_variant_expansion/summary/stage4_r18_lingbot_variant_summary.json",
            "results/acl2_v118tf_operation_specific_semantic_carrier_calibration/stage4_r21_lingbot_te3_semantic_eviction/summary/stage4_r21_lingbot_te3_semantic_eviction_summary.json",
        ],
    },
}


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def branch_summary_path(branch: str) -> Path:
    return BRANCH_ROOT / branch / f"{branch}_DECISION_SUMMARY.json"


def branch_report_path(branch: str) -> Path:
    return BRANCH_ROOT / branch / f"{branch}_REPORT.md"


def update_branch_artifacts(branch: str, meta: dict[str, Any], model: str, operation: str, surface: str) -> dict[str, Any]:
    path = branch_summary_path(branch)
    old = read_json(path)
    if branch not in EVIDENCE:
        return old
    evidence = EVIDENCE[branch]
    updated = dict(old)
    updated.update(
        {
            "schema": "acl2_v118tf_branch_decision_summary_v3",
            "branch": branch,
            "model": model,
            "operation": operation,
            "surface": surface,
            "status": evidence["status"],
            "runtime_launched": evidence["runtime_launched"],
            "gpu_runtime_launched": evidence["gpu_runtime_launched"],
            "global_goal_achieved": False,
            "latest_decision": evidence["latest_decision"],
            "primary_blocker": evidence["primary_blocker"],
            "repair_attempt_count": evidence["repair_attempt_count"],
            "current_closure_evidence": evidence["evidence"],
            "previous_status_before_current_closure": old.get("status", ""),
            "boundary": "Current closure merges branch summaries with post-final runtime/amendment evidence; it does not claim v118 success.",
        }
    )
    outputs = dict(updated.get("outputs", {}))
    outputs.update(
        {
            "summary": rel(path),
            "report": rel(branch_report_path(branch)),
        }
    )
    updated["outputs"] = outputs
    write_json(path, updated)
    lines = [
        f"# ACL2 v118-TF {branch} Current Closure Report",
        "",
        f"- status: `{updated['status']}`",
        f"- latest_decision: `{updated['latest_decision']}`",
        f"- runtime_launched: `{updated['runtime_launched']}`",
        f"- gpu_runtime_launched: `{updated['gpu_runtime_launched']}`",
        f"- global_goal_achieved: `{updated['global_goal_achieved']}`",
        "",
        "## Primary Blocker",
        "",
        updated["primary_blocker"],
        "",
        "## Evidence",
        "",
    ]
    for artifact in evidence["evidence"]:
        lines.append(f"- `{artifact}`")
    lines += [
        "",
        "## Boundary",
        "",
        "This branch closure supersedes stale structural-only wording where post-final runtime evidence exists. It does not fabricate missing metrics or change any stage summary values.",
    ]
    write_text(branch_report_path(branch), "\n".join(lines))
    return updated


def matrix_row(branch: str, model: str, operation: str, surface: str) -> dict[str, Any]:
    summary = read_json(branch_summary_path(branch))
    return {
        "schema": "acl2_v118tf_current_branch_completion_matrix_row_v1",
        "branch": branch,
        "model": summary.get("model", model),
        "operation": summary.get("operation", operation),
        "surface": summary.get("surface", surface),
        "status": summary.get("status", ""),
        "runtime_launched": bool(summary.get("runtime_launched", False)),
        "gpu_runtime_launched": bool(summary.get("gpu_runtime_launched", False)),
        "repair_attempt_count": summary.get("repair_attempt_count", ""),
        "global_goal_achieved": bool(summary.get("global_goal_achieved", False)),
        "latest_decision": summary.get("latest_decision", summary.get("status", "")),
        "primary_blocker": summary.get("primary_blocker", ""),
        "evidence": ";".join(summary.get("current_closure_evidence", []) or [rel(branch_summary_path(branch))]),
        "report": summary.get("outputs", {}).get("report", rel(branch_report_path(branch))),
        "summary": rel(branch_summary_path(branch)),
    }


def main() -> None:
    for branch, model, operation, surface in BRANCHES:
        update_branch_artifacts(branch, {}, model, operation, surface)

    rows = [matrix_row(*entry) for entry in BRANCHES]
    status_counts = dict(Counter(row["status"] for row in rows))
    forbidden_rows = [
        row["branch"]
        for row in rows
        if str(row["status"]).strip().lower() in FORBIDDEN_STATUSES
    ]
    nonterminal_rows = [row["branch"] for row in rows if row["status"] not in TERMINAL_STATUSES]
    global_success_rows = [row["branch"] for row in rows if row["global_goal_achieved"]]
    runtime_rows = [row["branch"] for row in rows if row["runtime_launched"]]
    taxonomy = "V118_STRUCTURAL_BLOCKERS_REMAIN_AFTER_EXHAUSTIVE_REPAIRS"
    all_terminal = not forbidden_rows and not nonterminal_rows

    write_csv(RESULT_ROOT / "V118_BRANCH_COMPLETION_MATRIX.csv", rows)
    write_csv(RESULT_ROOT / "V118_CURRENT_BRANCH_COMPLETION_MATRIX.csv", rows)

    summary = {
        "schema": "acl2_v118tf_current_final_decision_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "taxonomy": taxonomy,
        "global_goal_achieved": False,
        "all_branches_terminal": all_terminal,
        "branch_count": len(rows),
        "terminal_status_counts": status_counts,
        "forbidden_status_rows_present": bool(forbidden_rows),
        "forbidden_status_branch_ids": forbidden_rows,
        "nonterminal_branch_ids": nonterminal_rows,
        "global_success_branch_ids": global_success_rows,
        "runtime_branch_count": len(runtime_rows),
        "runtime_branch_ids": runtime_rows,
        "stage4_5_6_runtime_launched": bool(runtime_rows),
        "boundary": (
            "This current final supersedes the stale original-final no-runtime wording. "
            "Post-final runtime branches are included, but structural blockers remain and no branch achieved the global semantic-carrier calibration goal."
        ),
        "outputs": {
            "completion_matrix": rel(RESULT_ROOT / "V118_BRANCH_COMPLETION_MATRIX.csv"),
            "current_completion_matrix": rel(RESULT_ROOT / "V118_CURRENT_BRANCH_COMPLETION_MATRIX.csv"),
            "final_report": rel(RESULT_ROOT / "V118_FINAL_DECISION_REPORT.md"),
            "method_boundaries": rel(RESULT_ROOT / "V118_METHOD_AND_NO_GO_BOUNDARIES.md"),
            "carrier_attribution_report": rel(RESULT_ROOT / "CARRIER_ATTRIBUTION_REPORT.md"),
            "current_closure_summary": rel(RESULT_ROOT / "V118_POSTFINAL_R33_CURRENT_CLOSURE_SUMMARY.json"),
            "registry": rel(REGISTRY),
        },
    }
    write_json(RESULT_ROOT / "V118_FINAL_DECISION_SUMMARY.json", summary)
    write_json(RESULT_ROOT / "V118_POSTFINAL_R33_CURRENT_CLOSURE_SUMMARY.json", summary)

    report_lines = [
        "# ACL2 v118-TF Current Final Decision Report",
        "",
        f"- taxonomy: `{taxonomy}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- branch_count: `{summary['branch_count']}`",
        f"- all_branches_terminal: `{summary['all_branches_terminal']}`",
        f"- stage4_5_6_runtime_launched: `{summary['stage4_5_6_runtime_launched']}`",
        f"- runtime_branch_count: `{summary['runtime_branch_count']}`",
        "",
        "## Branch Matrix",
        "",
        "| branch | status | runtime | latest decision | blocker |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        report_lines.append(
            f"| `{row['branch']}` | `{row['status']}` | {row['runtime_launched']} | `{row['latest_decision']}` | {row['primary_blocker']} |"
        )
    report_lines += [
        "",
        "## Boundary",
        "",
        summary["boundary"],
        "",
        "## Evidence Chain",
        "",
        f"- Completion matrix: `{rel(RESULT_ROOT / 'V118_BRANCH_COMPLETION_MATRIX.csv')}`",
        f"- Current closure summary: `{rel(RESULT_ROOT / 'V118_POSTFINAL_R33_CURRENT_CLOSURE_SUMMARY.json')}`",
        f"- R33 LB-TA soft context gate: `{rel(RESULT_ROOT / 'stage4_r33_lingbot_ta_soft_context_gate/summary/stage4_r33_lingbot_ta_soft_context_gate_summary.json')}`",
        f"- R32 LB-AR amendment: `{rel(RESULT_ROOT / 'V118_POSTFINAL_R32_LBAR_RUNTIME_AMENDMENT_SUMMARY.json')}`",
        f"- R4 HS-HG amendment: `{rel(RESULT_ROOT / 'V118_POSTFINAL_R4_AMENDMENT_SUMMARY.json')}`",
    ]
    write_text(RESULT_ROOT / "V118_FINAL_DECISION_REPORT.md", "\n".join(report_lines))

    write_text(
        RESULT_ROOT / "CARRIER_ATTRIBUTION_REPORT.md",
        "\n".join(
            [
                "# ACL2 v118-TF Carrier Attribution Report",
                "",
                "Post-final runtime follow-ups were launched for selected LingBot and HorizonStream branches. None produced a global semantic-carrier calibration success.",
                "",
                "Branches with runtime No-Go or runtime-backed blockers are recorded in the current completion matrix. Branches that remained structurally blocked are not assigned fabricated ATE, Spearman, AUROC, uplift, or semantic-causality metrics.",
                "",
                f"- Completion matrix: `{rel(RESULT_ROOT / 'V118_BRANCH_COMPLETION_MATRIX.csv')}`",
                f"- Final decision summary: `{rel(RESULT_ROOT / 'V118_FINAL_DECISION_SUMMARY.json')}`",
            ]
        ),
    )
    write_text(
        RESULT_ROOT / "V118_METHOD_AND_NO_GO_BOUNDARIES.md",
        "\n".join(
            [
                "# ACL2 v118-TF Method And No-Go Boundaries",
                "",
                "- The original structural-only final is superseded by current post-final closure artifacts.",
                "- Runtime evidence exists only for the branch/stage artifacts listed in the current completion matrix.",
                "- No missing runtime metric is backfilled. Structural branches keep structural blocker status.",
                "- LB-TA R33 soft context-token gate had action fidelity but zero geometry delta versus default and matched controls.",
                "- LB-AI and LB-AR are runtime No-Go after fresh/control/cue-ablation follow-ups.",
                "- HS-GW/HS-GR remain bounded by GLA OOM/direct-gamma reliability blockers.",
                "- The global v118 semantic-carrier calibration goal remains false.",
            ]
        ),
    )
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "PostFinal-R33-CurrentClosure",
            "surface_or_branch": "all_branches",
            "status": taxonomy,
            "artifact": rel(RESULT_ROOT / "V118_POSTFINAL_R33_CURRENT_CLOSURE_SUMMARY.json"),
            "notes": "Refreshes final matrix/report from branch summaries plus R4/R21/R22/R27/R32/R33 evidence; global goal false",
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
