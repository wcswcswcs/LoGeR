#!/usr/bin/env python3
"""Build ACL2 v108TF Stage0 evidence freeze artifacts.

Stage0 records what is already proven by v105-v107 artifacts and locks the
forbidden repeat list.  It does not promote any old result into a v108 success.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search/stage0_evidence_freeze"
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V107TF = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
V107R = ROOT / "results/acl2_v107r_lingbot_semantic_memory_decision_cue_operation_control"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def process_rows() -> list[str]:
    proc = subprocess.run(
        "ps -eo pid,ppid,stat,etime,cmd | rg 'lingbot|run_worker|build_v10|ACL2_V10|conda run -n loger python' || true",
        shell=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    rows = [line for line in proc.stdout.splitlines() if line.strip()]
    self_markers = {
        "rg lingbot|run_worker",
        "build_v108tf_stage0_evidence_freeze.py",
        "ps -eo pid,ppid,stat,etime,cmd",
    }
    filtered = [
        line for line in rows
        if not any(marker in line for marker in self_markers)
    ]
    return filtered


def artifact_manifest() -> list[dict[str, Any]]:
    artifacts = [
        ("v105_stage0_summary", V105 / "stage0_repo_env_audit/stage0_summary.json", "required"),
        ("v105_full_kitti_metrics_csv", V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv", "required"),
        ("v105_full_kitti_summary", V105 / "stage1_lingbot_baseline/full_sequence_metrics/stage1_full_metric_summary.json", "required"),
        ("v105_trace_parity_rows", V105 / "stage2_gca_trace/no_action_parity_rows.csv", "required"),
        ("v105_trace_summary", V105 / "stage2_gca_trace/trace_summary.json", "required"),
        ("v105_stage4_action_summary", V105 / "stage4_lingbot_action_pilot_or_blocked/stage4_summary.json", "optional_known_failure"),
        ("v105_stage4_action_aggregate", V105 / "stage4_lingbot_action_pilot_or_blocked/action_aggregate_metrics.csv", "optional_known_failure"),
        ("v107tf_operation_trace_summary", V107TF / "stage1_cache_operation_instrumentation/operation_trace_summary.json", "required"),
        ("v107tf_operation_trace_rows", V107TF / "stage1_cache_operation_instrumentation/operation_trace_rows.csv", "required"),
        ("v107tf_operation_trace_parity_rows", V107TF / "stage1_cache_operation_instrumentation/operation_trace_parity_rows.csv", "required"),
        ("v107r_semantic_cue_summary", V107R / "stage1_semantic_cue_bank/semantic_cue_summary.json", "required"),
        ("v107r_frame_semantic_summary", V107R / "stage1_semantic_cue_bank/frame_semantic_summary.csv", "required"),
        ("v107r_stage3_summary", V107R / "stage3_operation_cue_matrix/stage3_summary.json", "required"),
        ("v107r_stage6_summary", V107R / "stage6_runtime_pilot_or_blocked/stage6_summary.json", "optional_prior_runtime_repair"),
        ("v107r_stage7b_summary", V107R / "stage7b_full_sequence_keyframe_aware_policy/stage7b_summary.json", "optional_prior_runtime_repair"),
    ]
    rows: list[dict[str, Any]] = []
    for name, path, requirement in artifacts:
        row_count = ""
        if path.exists() and path.suffix == ".csv":
            row_count = len(read_csv(path))
        rows.append(
            {
                "schema": "acl2_v108tf_stage0_artifact_manifest_row_v1",
                "artifact_id": name,
                "path": rel(path),
                "requirement": requirement,
                "exists": path.exists(),
                "suffix": path.suffix,
                "row_count": row_count,
            }
        )
    return rows


def full_kitti_baseline_table() -> list[dict[str, Any]]:
    src = V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"
    rows: list[dict[str, Any]] = []
    for row in read_csv(src):
        rows.append(
            {
                "schema": "acl2_v108tf_stage0_full_kitti_baseline_row_v1",
                "seq": row.get("seq", ""),
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "frames": row.get("frames", ""),
                "ATE_full_sim3_m": row.get("ATE_full_sim3_m", ""),
                "benchmark_rpe_rot": row.get("benchmark_rpe_rot", ""),
                "benchmark_rpe_trans": row.get("benchmark_rpe_trans", ""),
                "final_error_m": row.get("final_error_m", ""),
                "rolling_ATE_mean": row.get("rolling_ATE_mean", ""),
                "rolling_ATE_p90": row.get("rolling_ATE_p90", ""),
                "rolling_ATE_max": row.get("rolling_ATE_max", ""),
                "rolling_worse_fraction_gt_0p05": row.get("rolling_worse_fraction_gt_0p05", ""),
                "full_global_sim3_scale": row.get("full_global_sim3_scale", ""),
                "full_global_sim3_yaw_rad": row.get("full_global_sim3_yaw_rad", ""),
                "local_window_ATE_median": row.get("local_window_ATE_median", ""),
                "adjacent_log_scale_jump_median": row.get("adjacent_log_scale_jump_median", ""),
                "handoff_transfer_penalty_median": row.get("handoff_transfer_penalty_median", ""),
                "source_metric_scope_note": row.get("metric_scope_note", ""),
            }
        )
    return rows


def known_facts() -> dict[str, Any]:
    baseline_rows = full_kitti_baseline_table()
    trace_summary = read_json(V105 / "stage2_gca_trace/trace_summary.json")
    v107tf_trace = read_json(V107TF / "stage1_cache_operation_instrumentation/operation_trace_summary.json")
    semantic_summary = read_json(V107R / "stage1_semantic_cue_bank/semantic_cue_summary.json")
    v107r_stage3 = read_json(V107R / "stage3_operation_cue_matrix/stage3_summary.json")
    v107r_stage7b = read_json(V107R / "stage7b_full_sequence_keyframe_aware_policy/stage7b_summary.json")
    v105_stage4 = read_json(V105 / "stage4_lingbot_action_pilot_or_blocked/stage4_summary.json")
    aggregate = v105_stage4.get("aggregate_metrics", [])
    relaxed = next((row for row in aggregate if row.get("action_label") == "semantic_headlocal_relaxed_context_only_demote"), {})

    return {
        "schema": "acl2_v108tf_stage0_known_facts_v1",
        "v105_full_kitti_baseline_available": len(baseline_rows) == 4,
        "v105_full_kitti_baseline_ate_by_seq": {row["seq"]: row["ATE_full_sim3_m"] for row in baseline_rows},
        "v105_full_kitti_baseline_rows": baseline_rows,
        "v105_trace_parity_pass": bool(trace_summary.get("stage2_trace_parity_pass", False)),
        "v105_headlocal_relaxed_bad_l3_improvement": relaxed.get("bad_l3_median_improvement", "missing"),
        "v105_headlocal_relaxed_good_harm": relaxed.get("good_median_harm", "missing"),
        "v105_headlocal_relaxed_good_max_harm": relaxed.get("good_max_harm", "missing"),
        "v106r_readout_only_no_go": "not_recomputed_in_stage0; see v106R docs/results if needed",
        "v106r_missing_nonreadout_operation_types": "superseded_by_v107tf_non_readout_operation_types_observed",
        "v107tf_operation_trace_parity_pass": bool(v107tf_trace.get("trace_parity_pass", False)),
        "v107tf_observed_operation_types": v107tf_trace.get("observed_operation_types", []),
        "v107tf_operation_row_count": v107tf_trace.get("operation_row_count", 0),
        "v107r_semantic_cue_bank_available": bool(semantic_summary.get("stage1_pass", False)),
        "v107r_semantic_projection_coverage": semantic_summary.get("semantic_projection_coverage", "missing"),
        "v107r_semantic_patch_nonvoid_ratio": semantic_summary.get("semantic_patch_nonvoid_ratio", "missing"),
        "v107r_semantic_patch_purity_mean": semantic_summary.get("semantic_patch_purity_mean", "missing"),
        "v107r_stage3_semantic_increment_status": {
            "stage3_semantic_increment_pass": v107r_stage3.get("stage3_semantic_increment_pass", "missing"),
            "semantic_increment_pass_count": v107r_stage3.get("semantic_increment_pass_count", "missing"),
            "diagnostic_pass_count": v107r_stage3.get("diagnostic_pass_count", "missing"),
            "taxonomy_hint": "semantic_increment_diagnostic_failed; do not use this as v108 final stop",
        },
        "v107r_any_full_kitti_action_result_if_available": {
            "available": bool(v107r_stage7b),
            "stage7b_pass": v107r_stage7b.get("stage7_pass", "missing"),
            "stage7b_ate_complete": v107r_stage7b.get("stage7_ate_complete", "missing"),
            "mean_rel_improvement_vs_baseline": v107r_stage7b.get("mean_rel_improvement_vs_baseline", "missing"),
            "median_rel_improvement_vs_baseline": v107r_stage7b.get("median_rel_improvement_vs_baseline", "missing"),
            "max_rel_harm_vs_baseline": v107r_stage7b.get("max_rel_harm_vs_baseline", "missing"),
            "caution": "prior v107R repair evidence only; v108 still requires action surface search and semantic controls",
        },
    }


def forbidden_repeat_text() -> str:
    return """# v108TF Forbidden Repeat List

The following repeats are forbidden for v108TF:

1. readout attention mass threshold as action
2. frame-level semantic write filter without keyframe/cache fidelity
3. headlocal relaxed context-only demotion as method
4. semantic-label-only detector
5. selected-window-only action success
6. action without full KITTI 00/02 pilot
7. action without semantic controls
8. external depth / MoGe / LingBot-Depth runtime cue
9. post-hoc Sim(3) / SLAM correction

Additional audit note:
- v107R Stage7B is useful prior evidence, but it is not a v108TF final method claim because v108TF requires systematic action-surface search and semantic control comparisons.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_rows = full_kitti_baseline_table()
    manifest_rows = artifact_manifest()
    facts = known_facts()
    pending = process_rows()
    required_missing = [row["artifact_id"] for row in manifest_rows if row["requirement"] == "required" and not boolish(row["exists"])]
    baseline_ok = {row["seq"] for row in baseline_rows} == {"00", "01", "02", "05"}
    v107tf_ok = facts["v107tf_operation_trace_parity_pass"] and len(facts["v107tf_observed_operation_types"]) > 0
    semantic_ok = bool(facts["v107r_semantic_cue_bank_available"])
    stage0_pass = baseline_ok and v107tf_ok and semantic_ok and not required_missing and not pending

    write_csv(OUT / "full_kitti_baseline_table.csv", baseline_rows)
    write_csv(OUT / "available_artifact_manifest.csv", manifest_rows)
    write_json(OUT / "v105_v107_known_facts.json", facts)
    (OUT / "forbidden_repeat_list.md").write_text(forbidden_repeat_text(), encoding="utf-8")

    summary = {
        "schema": "acl2_v108tf_stage0_summary_v1",
        "stage0_pass": stage0_pass,
        "baseline_ok": baseline_ok,
        "v107tf_operation_trace_artifacts_readable": v107tf_ok,
        "v107r_semantic_cue_bank_artifacts_readable": semantic_ok,
        "required_missing_artifacts": required_missing,
        "pending_lingbot_process_rows": pending,
        "full_kitti_baseline_seq_count": len({row["seq"] for row in baseline_rows}),
        "observed_operation_types": facts["v107tf_observed_operation_types"],
        "forbidden_repeat_list_written": (OUT / "forbidden_repeat_list.md").exists(),
        "outputs": {
            "stage0_summary": rel(OUT / "stage0_summary.json"),
            "v105_v107_known_facts": rel(OUT / "v105_v107_known_facts.json"),
            "forbidden_repeat_list": rel(OUT / "forbidden_repeat_list.md"),
            "full_kitti_baseline_table": rel(OUT / "full_kitti_baseline_table.csv"),
            "available_artifact_manifest": rel(OUT / "available_artifact_manifest.csv"),
        },
    }
    write_json(OUT / "stage0_summary.json", summary)
    if not stage0_pass:
        (OUT / "STAGE0_EVIDENCE_FREEZE_BLOCKED.md").write_text(
            "# Stage0 Evidence Freeze Blocked\n\n"
            f"- required_missing_artifacts: `{required_missing}`\n"
            f"- pending_lingbot_process_rows: `{pending}`\n"
            f"- baseline_ok: `{baseline_ok}`\n"
            f"- v107tf_operation_trace_artifacts_readable: `{v107tf_ok}`\n"
            f"- v107r_semantic_cue_bank_artifacts_readable: `{semantic_ok}`\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
