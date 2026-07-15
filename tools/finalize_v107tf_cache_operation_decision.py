#!/usr/bin/env python3
"""Write ACL2 v107TF final decision from completed stage artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V107 = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
FINAL = V107 / "final_decision"

ALLOWED_TAXONOMY = {
    "CACHE_OPERATION_TRACE_BLOCKED",
    "CACHE_OPERATION_TRACE_READY_NO_ACTION",
    "VERIFIER_PROXY_ONLY_NO_ACTION",
    "NO_MEMORY_OPERATION_LEVER_FOUND",
    "MEMORY_OPERATION_LEVER_DIAGNOSTIC_PASS_ACTION_BLOCKED",
    "SEMANTIC_INCREMENT_FAIL_GEOMETRY_ONLY",
    "ROLE_DISAMBIGUATION_PASS_ACTION_NOT_READY",
    "ACTION_SURFACE_PASS_RUNTIME_READY",
    "RUNTIME_PILOT_PASS_FULL_VALIDATION_READY",
    "LOCAL_MEMORY_ACTION_PASS_FULL_PENDING",
    "FULL_METHOD_SUCCESS",
    "NO_GO_GOOD_HARM",
    "NO_GO_TRACE_PASS_NO_L3_EFFECT",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    s0 = load_json(V107 / "stage0_evidence_freeze/stage0_summary.json")
    s1_code = load_json(V107 / "stage1_cache_operation_instrumentation/code_audit_summary.json")
    s1 = load_json(V107 / "stage1_cache_operation_instrumentation/operation_trace_summary.json")
    s2 = load_json(V107 / "stage2_metric_reliability_verifier/verifier_coverage_summary.json")
    s3 = load_json(V107 / "stage3_operation_discovery/stage3_summary.json")

    taxonomy = s3.get("final_taxonomy_if_stop_here") or "CACHE_OPERATION_TRACE_BLOCKED"
    if taxonomy not in ALLOWED_TAXONOMY:
        raise ValueError(f"Unexpected v107TF taxonomy: {taxonomy}")

    length = s3.get("length_matched_96f") or {}
    top = length.get("top_cue") or {}
    runtime_action_run = False
    full_validation_run = False
    reason = (
        "Stage3 length-matched operation discovery found zero levers passing controls; "
        "same-count random margin for the top cue is 0.0, so Stage4/5/6 are blocked by plan."
    )
    decision = {
        "schema": "acl2_v107tf_final_decision_v1",
        "taxonomy": taxonomy,
        "reason": reason,
        "stage0_pass": bool(s0.get("stage0_pass")),
        "stage1_code_audit_pass": bool(s1_code.get("code_audit_pass", True)),
        "stage1_trace_parity_pass": bool(s1.get("trace_parity_pass")),
        "stage1_operation_row_count": s1.get("operation_row_count"),
        "stage2_pass": bool(s2.get("stage2_pass")),
        "stage2_verifier_coverage": s2.get("verifier_coverage"),
        "stage2_proxy_only": s2.get("proxy_only", False),
        "stage3_pass": bool(s3.get("stage3_pass")),
        "stage3_main_universe": s3.get("main_universe"),
        "stage3_length_matched_case_count": length.get("case_count"),
        "stage3_length_matched_operation_row_count": length.get("operation_row_count"),
        "stage3_length_matched_lever_pass_count": length.get("lever_pass_count"),
        "stage3_top_cue": top,
        "runtime_action_run": runtime_action_run,
        "full_validation_run": full_validation_run,
        "full_kitti_ate_new_run_available": False,
        "result_root": rel(V107),
        "evidence_files": [
            rel(V107 / "stage0_evidence_freeze/stage0_summary.json"),
            rel(V107 / "stage1_cache_operation_instrumentation/operation_trace_summary.json"),
            rel(V107 / "stage2_metric_reliability_verifier/verifier_coverage_summary.json"),
            rel(V107 / "stage3_operation_discovery/stage3_summary.json"),
            rel(V107 / "stage3_operation_discovery/NO_MEMORY_OPERATION_LEVER_FOUND.md"),
        ],
    }
    write_json(FINAL / "final_decision.json", decision)
    report = [
        "# ACL2 v107TF Final Report",
        "",
        f"Taxonomy: `{taxonomy}`",
        "",
        f"Reason: {reason}",
        "",
        "Gate summary:",
        "",
        f"- Stage0 pass: {decision['stage0_pass']}",
        f"- Stage1 trace parity pass: {decision['stage1_trace_parity_pass']}",
        f"- Stage1 operation rows: {decision['stage1_operation_row_count']}",
        f"- Stage2 pass: {decision['stage2_pass']}",
        f"- Stage2 verifier coverage: {decision['stage2_verifier_coverage']}",
        f"- Stage3 pass: {decision['stage3_pass']}",
        f"- Length-matched case count: {decision['stage3_length_matched_case_count']}",
        f"- Length-matched operation rows: {decision['stage3_length_matched_operation_row_count']}",
        f"- Length-matched lever pass count: {decision['stage3_length_matched_lever_pass_count']}",
        "",
        "Action/full validation:",
        "",
        "- Runtime action was not run because Stage3 failed.",
        "- Full validation was not run because Stage5/6 were never reached.",
        "- No new full KITTI ATE is available from v107TF.",
        "",
        "Top failed cue:",
        "",
        f"- {top.get('cue_name', '')}",
        f"- bad_recall: {top.get('bad_recall', '')}",
        f"- good_FPR: {top.get('good_FPR', '')}",
        f"- balanced_accuracy: {top.get('balanced_accuracy', '')}",
        f"- same_count_random_p95: {top.get('same_count_random_BA_p95', '')}",
        f"- same_count_random_margin: {top.get('same_count_random_margin', '')}",
        "",
        "Evidence files:",
        "",
        *[f"- {path}" for path in decision["evidence_files"]],
    ]
    write_text(FINAL / "final_report.md", "\n".join(report))


if __name__ == "__main__":
    main()
