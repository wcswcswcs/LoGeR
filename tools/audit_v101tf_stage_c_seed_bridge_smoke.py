#!/usr/bin/env python3
"""Audit the v101 Stage-C seed provenance bridge smoke traces.

This is an instrumentation audit only.  It verifies that diagnostic no-action
smoke runs can carry Stage-C ``seed_global_track_idx`` into current query tokens
and SWA cache top-k source tokens.  It does not authorize runtime action.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"
SMOKE = ROOT / "stage_c_seed_bridge_smoke_v2"
FIRST_SMOKE = ROOT / "stage_c_seed_bridge_smoke"
TARGET_TRACE = ROOT / "stage_c_seed_bridge_target_traces"
TARGET_TRACE_PLAN = ROOT / "stage_c_seed_bridge_target_traces_plan"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}"}
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
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def tensor_nonnegative_count(value: Any) -> int:
    if not torch.is_tensor(value):
        return 0
    return int((value.detach().cpu().long().reshape(-1) >= 0).sum().item())


def tensor_true_count(value: Any) -> int:
    if not torch.is_tensor(value):
        return 0
    return int(value.detach().cpu().bool().reshape(-1).sum().item())


def tensor_shape(value: Any) -> str:
    if not torch.is_tensor(value):
        return ""
    return json.dumps([int(x) for x in value.shape])


def audit_smoke(root: Path, *, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = read_json(root / "summary.json")
    job_rows = read_rows(root / "job_results.csv")
    job_by_case = {row.get("case_id", ""): row for row in job_rows}
    trace_paths = sorted(root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt"))
    rows: list[dict[str, Any]] = []
    for path in trace_paths:
        case_id = path.parents[2].name
        job = job_by_case.get(case_id, {})
        bucket = str(job.get("bucket", ""))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sample = payload.get("sampled_query_stage_c_seed_global_track_idx")
        topk = payload.get("current_Q_to_cache_K_topk_cache_stage_c_seed_global_track_idx")
        same = payload.get("current_Q_to_cache_K_topk_same_stage_c_seed_global_track_idx")
        lifecycle = payload.get("ttt_prev_anchor_lifecycle_rows") or []
        rows.append(
            {
                "smoke_label": label,
                "case_id": case_id,
                "seq": job.get("seq", ""),
                "target_taxonomy": bucket[len("V101_") :] if bucket.startswith("V101_") else bucket,
                "trace_payload_path": str(path),
                "schema": payload.get("schema", ""),
                "chunk_idx": payload.get("chunk_idx", ""),
                "current_stage_c_seed_trace_available": bool(
                    payload.get("current_stage_c_seed_global_track_idx_trace_available")
                ),
                "cache_stage_c_seed_trace_available": bool(
                    payload.get("cache_stage_c_seed_global_track_idx_trace_available")
                ),
                "sample_shape": tensor_shape(sample),
                "topk_shape": tensor_shape(topk),
                "same_shape": tensor_shape(same),
                "sample_nonnegative_count": tensor_nonnegative_count(sample),
                "topk_nonnegative_count": tensor_nonnegative_count(topk),
                "same_seed_true_count": tensor_true_count(same),
                "same_seed_frac_mean": payload.get(
                    "current_Q_to_cache_K_topk_same_stage_c_seed_global_track_idx_frac_mean",
                    "",
                ),
                "lifecycle_row_count": len(lifecycle),
                "lifecycle_seed_mode_nonnull_count": sum(
                    1 for item in lifecycle if item.get("source_stage_c_seed_global_track_idx_mode") is not None
                ),
            }
        )
    first_cmds = [row.get("cmd_shell", "") for row in job_rows]
    job_returncodes = [int(row.get("returncode", "1") or 1) for row in job_rows]
    out = {
        "smoke_label": label,
        "smoke_root": str(root),
        "status": summary.get("status", "missing"),
        "selected_case_count": summary.get("selected_case_count", ""),
        "completed_job_count": summary.get("completed_job_count", 0),
        "failed_job_count": summary.get("failed_job_count", ""),
        "trace_payload_file_count": len(trace_paths),
        "job_results_row_count": len(job_rows),
        "all_jobs_returncode_zero": bool(job_rows) and all(code == 0 for code in job_returncodes),
        "all_variants_read_no_action": bool(job_rows) and all(row.get("variant") == "READ_NO_ACTION" for row in job_rows),
        "all_beta_frame_zero": bool(job_rows) and all("--beta_frame 0.0" in cmd for cmd in first_cmds),
        "all_hybrid_mode_read_path_only": bool(job_rows) and all(
            "--hybrid_memory_mode read_path_only" in cmd for cmd in first_cmds
        ),
        "all_stage_c_cache_read": bool(job_rows) and all("--stage_c_cache_mode read" in cmd for cmd in first_cmds),
        "all_current_seed_trace_available": bool(rows)
        and all(row["current_stage_c_seed_trace_available"] for row in rows),
        "all_cache_seed_trace_available": bool(rows)
        and all(row["cache_stage_c_seed_trace_available"] for row in rows),
        "all_current_seed_nonempty": bool(rows) and all(int(row["sample_nonnegative_count"]) > 0 for row in rows),
        "all_cache_seed_nonempty": bool(rows) and all(int(row["topk_nonnegative_count"]) > 0 for row in rows),
        "sample_nonnegative_counts": [int(row["sample_nonnegative_count"]) for row in rows],
        "topk_nonnegative_counts": [int(row["topk_nonnegative_count"]) for row in rows],
        "same_seed_true_counts": [int(row["same_seed_true_count"]) for row in rows],
        "same_seed_frac_means": [row["same_seed_frac_mean"] for row in rows],
    }
    taxonomy_agg: dict[str, dict[str, Any]] = {}
    for row in rows:
        taxonomy = str(row.get("target_taxonomy", "") or "UNKNOWN")
        agg = taxonomy_agg.setdefault(
            taxonomy,
            {
                "count": 0,
                "sample_nonnegative_sum": 0,
                "topk_nonnegative_sum": 0,
                "same_seed_true_sum": 0,
                "same_seed_frac_values": [],
            },
        )
        agg["count"] += 1
        agg["sample_nonnegative_sum"] += int(row["sample_nonnegative_count"])
        agg["topk_nonnegative_sum"] += int(row["topk_nonnegative_count"])
        agg["same_seed_true_sum"] += int(row["same_seed_true_count"])
        if row["same_seed_frac_mean"] not in {"", None}:
            agg["same_seed_frac_values"].append(float(row["same_seed_frac_mean"]))
    for agg in taxonomy_agg.values():
        vals = agg.pop("same_seed_frac_values")
        agg["same_seed_frac_mean_avg"] = sum(vals) / len(vals) if vals else None
    out["same_seed_true_total"] = sum(int(row["same_seed_true_count"]) for row in rows)
    out["min_sample_nonnegative_count"] = min([int(row["sample_nonnegative_count"]) for row in rows], default=0)
    out["min_topk_nonnegative_count"] = min([int(row["topk_nonnegative_count"]) for row in rows], default=0)
    out["taxonomy_agg"] = taxonomy_agg
    return out, rows


def write_smoke_fail_forward_docs(root: Path, summary: dict[str, Any], *, label: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    passed = bool(summary.get("all_current_seed_nonempty") and summary.get("all_cache_seed_nonempty"))
    status = str(summary.get("status", "missing"))
    root.joinpath("failure_report.md").write_text(
        "\n".join(
            [
                f"# Stage-C Seed Bridge Smoke {label}",
                "",
                f"- status: {status}",
                f"- provenance_bridge_smoke_pass: {passed}",
                f"- diagnostic_only: true",
                "- runtime_action_allowed: false",
                "",
                "This smoke does not evaluate method success. It only checks whether Stage-C seed provenance reaches trace payloads.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    root.joinpath("what_would_have_to_be_true_to_pass.md").write_text(
        "\n".join(
            [
                f"# Stage-C Seed Bridge Smoke Pass Conditions {label}",
                "",
                "- all planned jobs must return code 0",
                "- variant must remain READ_NO_ACTION with beta_frame=0",
                "- Stage-C cache mode must be read-only",
                "- current sampled query seed and cache top-k seed tensors must both be present and nonempty",
                "- passing this smoke still does not authorize Track U/V/Q2/M4/runtime action",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    root.joinpath("control_gap_report.md").write_text(
        "\n".join(
            [
                f"# Stage-C Seed Bridge Smoke Control Gap {label}",
                "",
                "The smoke is no-action and diagnostic-only. It does not measure L3 improvement, M4 state-machine outcome, or full-sequence ATE.",
                "The remaining gap is to use the provenance bridge to rebuild strict current-support / identity rows, then re-run Track U/V/Q2 gates.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    root.joinpath("next_attempt_recommendation.md").write_text(
        "\n".join(
            [
                f"# Stage-C Seed Bridge Smoke Next Attempt {label}",
                "",
                "Use the new trace fields to materialize anchor/component current-support rows.",
                "Do not run action until strict current support, scale observability, Q2 true-stage, and M4 gates pass.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_rows(
        root / "false_positive_false_negative_rows.csv",
        [
            {
                "cue_name": "stage_c_seed_bridge_smoke",
                "row_kind": "not_selector_provenance_smoke",
                "smoke_label": label,
                "provenance_bridge_smoke_pass": passed,
                "reason": "No FP/FN selector is defined for this instrumentation smoke.",
                "claim_level": "diagnostic_no_action",
            }
        ],
    )


def main() -> None:
    current_summary, current_rows = audit_smoke(SMOKE, label="v2_after_dense_sparse_seed_merge")
    first_summary, first_rows = audit_smoke(FIRST_SMOKE, label="v1_before_dense_sparse_seed_merge")
    plan_summary, _ = audit_smoke(ROOT / "stage_c_seed_bridge_smoke_plan", label="planned_no_run")
    target_summary, target_rows = audit_smoke(TARGET_TRACE, label="target_universe_28_after_seed_merge")
    target_plan_summary, _ = audit_smoke(TARGET_TRACE_PLAN, label="target_universe_28_planned_no_run")
    write_smoke_fail_forward_docs(SMOKE, current_summary, label="v2_after_dense_sparse_seed_merge")
    write_smoke_fail_forward_docs(FIRST_SMOKE, first_summary, label="v1_before_dense_sparse_seed_merge")
    write_smoke_fail_forward_docs(ROOT / "stage_c_seed_bridge_smoke_plan", plan_summary, label="planned_no_run")
    write_smoke_fail_forward_docs(TARGET_TRACE, target_summary, label="target_universe_28_after_seed_merge")
    write_smoke_fail_forward_docs(TARGET_TRACE_PLAN, target_plan_summary, label="target_universe_28_planned_no_run")
    rows = first_rows + current_rows + target_rows
    bridge_pass = (
        current_summary.get("all_jobs_returncode_zero") is True
        and current_summary.get("trace_payload_file_count") == 2
        and current_summary.get("all_variants_read_no_action") is True
        and current_summary.get("all_beta_frame_zero") is True
        and current_summary.get("all_hybrid_mode_read_path_only") is True
        and current_summary.get("all_stage_c_cache_read") is True
        and current_summary.get("all_current_seed_trace_available") is True
        and current_summary.get("all_cache_seed_trace_available") is True
        and current_summary.get("all_current_seed_nonempty") is True
        and current_summary.get("all_cache_seed_nonempty") is True
    )
    target_pass = (
        target_summary.get("all_jobs_returncode_zero") is True
        and target_summary.get("trace_payload_file_count") == 28
        and target_summary.get("all_variants_read_no_action") is True
        and target_summary.get("all_beta_frame_zero") is True
        and target_summary.get("all_hybrid_mode_read_path_only") is True
        and target_summary.get("all_stage_c_cache_read") is True
        and target_summary.get("all_current_seed_trace_available") is True
        and target_summary.get("all_cache_seed_trace_available") is True
        and target_summary.get("all_current_seed_nonempty") is True
        and target_summary.get("all_cache_seed_nonempty") is True
    )
    summary = {
        "schema": "acl2_v101_stage_c_seed_bridge_smoke_audit_v1",
        "diagnostic_only": True,
        "method_goal_achieved": False,
        "runtime_action_allowed": False,
        "stage_c_seed_bridge_smoke_pass": bool(bridge_pass),
        "stage_c_seed_bridge_target_trace_pass": bool(target_pass),
        "current_smoke": current_summary,
        "target_trace": target_summary,
        "first_smoke_before_repair": first_summary,
        "repair_effect_observed": bool(
            first_summary.get("all_current_seed_nonempty") is False
            and current_summary.get("all_current_seed_nonempty") is True
        ),
        "repair_note": (
            "Merged sparse MaskletOutput.seed_global_track_idx projection into dense semantic prior path; "
            "kept READ_NO_ACTION beta=0 smoke diagnostic-only."
        ),
    }
    write_json(FINAL / "stage_c_seed_bridge_smoke_summary.json", summary)
    write_rows(FINAL / "stage_c_seed_bridge_smoke_rows.csv", rows)
    report = [
        "# ACL2 v101 Stage-C Seed Bridge Smoke",
        "",
        f"- diagnostic_only: {summary['diagnostic_only']}",
        f"- stage_c_seed_bridge_smoke_pass: {summary['stage_c_seed_bridge_smoke_pass']}",
        f"- runtime_action_allowed: {summary['runtime_action_allowed']}",
        f"- current trace payloads: {current_summary.get('trace_payload_file_count')}",
        f"- current seed nonnegative counts: {current_summary.get('sample_nonnegative_counts')}",
        f"- cache seed nonnegative counts: {current_summary.get('topk_nonnegative_counts')}",
        f"- same seed true counts: {current_summary.get('same_seed_true_counts')}",
        f"- target trace pass: {summary['stage_c_seed_bridge_target_trace_pass']}",
        f"- target trace payloads: {target_summary.get('trace_payload_file_count')}",
        f"- target completed jobs: {target_summary.get('completed_job_count')}",
        f"- target failed jobs: {target_summary.get('failed_job_count')}",
        f"- target min current seed nonnegative count: {target_summary.get('min_sample_nonnegative_count')}",
        f"- target min cache top-k seed nonnegative count: {target_summary.get('min_topk_nonnegative_count')}",
        f"- target same seed true total: {target_summary.get('same_seed_true_total')}",
        f"- target taxonomy aggregate: `{json.dumps(target_summary.get('taxonomy_agg', {}), sort_keys=True)}`",
        f"- first smoke current seed nonempty before repair: {first_summary.get('all_current_seed_nonempty')}",
        "",
        "This audit verifies a provenance/instrumentation bridge only; it does not pass Track U/V/Q2/M4 or authorize runtime action.",
    ]
    (FINAL / "stage_c_seed_bridge_smoke_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
