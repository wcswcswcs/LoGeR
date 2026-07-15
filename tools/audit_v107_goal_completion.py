#!/usr/bin/env python3
"""Audit ACL2 v107TF + v107R goal completion artifacts."""

from __future__ import annotations

import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
V107TF = RESULTS / "acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
V107R = RESULTS / "acl2_v107r_lingbot_semantic_memory_decision_cue_operation_control"


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    for rel in [
        "stage0_evidence_freeze/stage0_summary.json",
        "stage1_cache_operation_instrumentation/code_audit_summary.json",
        "stage1_cache_operation_instrumentation/operation_trace_summary.json",
        "stage1_cache_operation_instrumentation/operation_trace_rows.csv",
        "stage1_cache_operation_instrumentation/operation_trace_parity_rows.csv",
        "stage2_metric_reliability_verifier/verifier_coverage_summary.json",
        "stage2_metric_reliability_verifier/operation_verifier_join_rows.csv",
        "stage3_operation_discovery/operation_discovery_rows.csv",
        "stage3_operation_discovery/operation_lever_rank.csv",
        "stage3_operation_discovery/semantic_increment_by_operation.csv",
        "stage3_operation_discovery/sequence_loso_rows.csv",
        "stage3_operation_discovery/control_margin_rows.csv",
        "stage3_operation_discovery/top_lever_reports",
        "stage3_operation_discovery/stage3_summary.json",
        "stage3_operation_discovery/NO_MEMORY_OPERATION_LEVER_FOUND.md",
        "final_decision/final_decision.json",
        "final_decision/final_report.md",
    ]:
        add("v107TF exists " + rel, (V107TF / rel).exists())

    tf_s0 = load_json(V107TF / "stage0_evidence_freeze/stage0_summary.json")
    tf_s1 = load_json(V107TF / "stage1_cache_operation_instrumentation/operation_trace_summary.json")
    tf_s2 = load_json(V107TF / "stage2_metric_reliability_verifier/verifier_coverage_summary.json")
    tf_s3 = load_json(V107TF / "stage3_operation_discovery/stage3_summary.json")
    tf_fd = load_json(V107TF / "final_decision/final_decision.json")
    length = tf_s3.get("length_matched_96f") or {}
    top = length.get("top_cue") or {}
    add("v107TF stage0 pass", tf_s0.get("stage0_pass") is True)
    add(
        "v107TF stage1 pass",
        tf_s1.get("stage1_pass") is True
        and tf_s1.get("trace_parity_pass") is True
        and int(tf_s1.get("operation_row_count")) == 20080,
    )
    add(
        "v107TF stage2 pass",
        tf_s2.get("stage2_pass") is True
        and float(tf_s2.get("verifier_coverage")) == 1.0
        and tf_s2.get("proxy_only") is False,
    )
    add(
        "v107TF stage3 no lever terminal",
        tf_s3.get("stage3_pass") is False
        and tf_s3.get("final_taxonomy_if_stop_here") == "NO_MEMORY_OPERATION_LEVER_FOUND"
        and int(length.get("lever_pass_count")) == 0
        and float(top.get("same_count_random_margin")) == 0.0,
    )
    add(
        "v107TF final taxonomy terminal",
        tf_fd.get("taxonomy") == "NO_MEMORY_OPERATION_LEVER_FOUND"
        and tf_fd.get("runtime_action_run") is False
        and tf_fd.get("full_validation_run") is False
        and tf_fd.get("full_kitti_ate_new_run_available") is False,
    )

    for rel in [
        "stage0_v107r_evidence_freeze/v107r_known_facts.json",
        "stage0_v107r_evidence_freeze/forbidden_repeat_list.md",
        "stage0_v107r_evidence_freeze/allowed_lingbot_memory_operations.csv",
        "stage0_v107r_evidence_freeze/why_no_external_depth_or_postprocessing.md",
        "stage0_v107r_evidence_freeze/stage0_summary.json",
        "stage1_semantic_cue_bank/token_grid_audit.md",
        "stage1_semantic_cue_bank/token_grid_rows.csv",
        "stage1_semantic_cue_bank/token_semantic_rows.csv",
        "stage1_semantic_cue_bank/frame_semantic_summary.csv",
        "stage1_semantic_cue_bank/semantic_role_mapping.csv",
        "stage1_semantic_cue_bank/semantic_continuity_rows.csv",
        "stage1_semantic_cue_bank/semantic_cue_summary.json",
        "stage2_operation_cue_join/operation_semantic_rows.csv",
        "stage2_operation_cue_join/operation_type_coverage.csv",
        "stage2_operation_cue_join/missing_operation_type_report.md",
        "stage2_operation_cue_join/stage2_summary.json",
        "stage3_operation_cue_matrix/operation_cue_pattern_metrics.csv",
        "stage3_operation_cue_matrix/semantic_increment_by_operation.csv",
        "stage3_operation_cue_matrix/top_memory_lever_candidates.csv",
        "stage3_operation_cue_matrix/operation_cue_failure_panels.md",
        "stage3_operation_cue_matrix/semantic_increment_failure.md",
        "stage3_operation_cue_matrix/stage3_summary.json",
        "stage4_memory_role_disambiguation/stage4_summary.json",
        "stage5_action_surface_selection/stage5_summary.json",
        "stage6_runtime_pilot_or_blocked/stage6_summary.json",
        "stage7_full_validation_or_blocked/stage7_summary.json",
        "final_decision/final_decision.json",
        "final_decision/final_report.md",
    ]:
        add("v107R exists " + rel, (V107R / rel).exists())

    r_s0 = load_json(V107R / "stage0_v107r_evidence_freeze/stage0_summary.json")
    r_s1 = load_json(V107R / "stage1_semantic_cue_bank/semantic_cue_summary.json")
    r_s2 = load_json(V107R / "stage2_operation_cue_join/stage2_summary.json")
    r_s3 = load_json(V107R / "stage3_operation_cue_matrix/stage3_summary.json")
    r_fd = load_json(V107R / "final_decision/final_decision.json")
    add("v107R stage0 pass", r_s0.get("stage0_pass") is True and r_s0.get("missing_required_artifacts") == [])
    add(
        "v107R stage1 pass",
        r_s1.get("stage1_pass") is True
        and r_s1.get("token_alignment_pass") is True
        and float(r_s1.get("semantic_projection_coverage")) >= 0.95
        and float(r_s1.get("semantic_patch_nonvoid_ratio")) >= 0.95
        and float(r_s1.get("semantic_patch_purity_mean")) >= 0.70
        and float(r_s1.get("semantic_role_coverage")) >= 0.95,
    )
    add(
        "v107R stage2 pass",
        r_s2.get("stage2_pass") is True
        and r_s2.get("non_readout_operation_present") is True
        and float(r_s2.get("semantic_join_coverage")) >= 0.80
        and r_s2.get("trace_parity_pass") is True,
    )
    add(
        "v107R stage3 terminal",
        r_s3.get("stage3_diagnostic_pass") is False
        and r_s3.get("stage3_semantic_increment_pass") is False
        and r_s3.get("stage3_action_entry_pass") is False
        and int(r_s3.get("semantic_increment_pass_count")) == 0,
    )
    add(
        "v107R final taxonomy terminal",
        r_fd.get("taxonomy") == "SEMANTIC_INCREMENT_FAIL_INTERNAL_ONLY_DOMINATES"
        and r_fd.get("runtime_action_run") is False
        and r_fd.get("full_validation_run") is False,
    )

    for rel in [
        "docs/ACL2_v107TF_LingBot_CacheOperationObservability_SemanticAwareMemoryUpdateRetention_执行日志.md",
        "docs/ACL2_v107TF_LingBot_CacheOperationObservability_SemanticAwareMemoryUpdateRetention_实验结果复盘.md",
        "docs/ACL2_v107R_LingBot_SemanticOnlyMemoryDecisionCue_OperationControl_执行日志.md",
        "docs/ACL2_v107R_LingBot_SemanticOnlyMemoryDecisionCue_OperationControl_实验结果复盘.md",
        "tools/finalize_v107tf_cache_operation_decision.py",
        "tools/build_v107r_semantic_memory_decision_cue_operation_control.py",
        "tools/audit_v107_goal_completion.py",
    ]:
        add("workspace file exists " + rel, (ROOT / rel).exists())

    failed = [item for item in checks if not item[1]]
    print(json.dumps({"check_count": len(checks), "fail_count": len(failed), "failed": failed}, indent=2, ensure_ascii=False))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
