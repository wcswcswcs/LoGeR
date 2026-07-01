#!/usr/bin/env python3
"""Audit v102 state-machine diagnostic action-probe closure.

This verifies the default-off SWA state-machine action probe as an audit
artifact only.  A passing closure here means the probe actually changed the
SWA KV-cache read path and produced paired trajectory metrics; it does not
promote Stage4 because Stage3 strict oracle and Stage4 harm/control gates still
must pass separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
STAGE4 = ROOT / "stage4_memory_action_surface_oracle"
DEFAULT_TARGET_CSV = STAGE4 / "v102_state_machine_scaffold_trace_targets.csv"
DEFAULT_TRACE_ROOT = STAGE4 / "v102_state_machine_action_probe_reject_unreliable_v1"
DEFAULT_METRIC_SUMMARY = STAGE4 / "state_machine_action_probe_reject_unreliable_v1_metrics/state_machine_trace_run_metrics_summary.json"
DEFAULT_OUTPUT_PREFIX = STAGE4 / "state_machine_action_probe_reject_unreliable_v1_closure"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def prefixed_path(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}{suffix}"


def raw_payload_row(case_id: str, trace_root: Path) -> dict[str, Any]:
    files = sorted((trace_root / case_id / "READ_NO_ACTION" / "swa_raw_transport_trace").glob("*.pt"))
    row: dict[str, Any] = {
        "raw_payload_file_count": len(files),
        "raw_payload_path": files[0].as_posix() if files else "",
        "raw_trace_available": False,
        "raw_trace_applied": False,
        "raw_scaffold_only": False,
        "raw_runtime_action_allowed": False,
        "raw_action": "",
        "raw_reason": "",
        "raw_probe_impl": "",
        "raw_rejected_history_tokens": 0,
        "raw_kept_history_tokens": 0,
        "raw_rejected_history_frac": 0.0,
        "raw_unreliable_d_high_tokens": 0,
        "raw_unreliable_g_high_tokens": 0,
        "raw_read_error": "",
    }
    if not files:
        row["raw_read_error"] = "missing_raw_payload"
        return row
    try:
        payload = torch.load(files[0], map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        row["raw_read_error"] = f"{type(exc).__name__}: {exc}"
        return row
    if not isinstance(payload, dict):
        row["raw_read_error"] = f"payload_not_dict:{type(payload).__name__}"
        return row
    row.update({
        "raw_trace_available": bool(payload.get("v102_swa_state_machine_trace_available", False)),
        "raw_trace_applied": bool(payload.get("v102_swa_state_machine_trace_applied", False)),
        "raw_scaffold_only": bool(payload.get("v102_swa_state_machine_scaffold_only", False)),
        "raw_runtime_action_allowed": bool(payload.get("v102_swa_state_machine_runtime_action_allowed", False)),
        "raw_action": str(payload.get("v102_swa_state_machine_action", "") or ""),
        "raw_reason": str(payload.get("v102_swa_state_machine_reason", "") or ""),
        "raw_probe_impl": str(payload.get("v102_swa_state_machine_probe_impl", "") or ""),
        "raw_rejected_history_tokens": int(payload.get("v102_swa_state_machine_rejected_history_tokens", 0) or 0),
        "raw_kept_history_tokens": int(payload.get("v102_swa_state_machine_kept_history_tokens", 0) or 0),
        "raw_rejected_history_frac": float(payload.get("v102_swa_state_machine_rejected_history_frac", 0.0) or 0.0),
        "raw_unreliable_d_high_tokens": int(payload.get("v102_swa_state_machine_unreliable_d_high_tokens", 0) or 0),
        "raw_unreliable_g_high_tokens": int(payload.get("v102_swa_state_machine_unreliable_g_high_tokens", 0) or 0),
        "raw_supported_d_low_tokens": int(payload.get("v102_swa_state_machine_supported_d_low_tokens", 0) or 0),
        "raw_supported_semantic_static_tokens": int(
            payload.get("v102_swa_state_machine_supported_semantic_static_tokens", 0) or 0
        ),
        "raw_supported_k_stable_tokens": int(payload.get("v102_swa_state_machine_supported_k_stable_tokens", 0) or 0),
        "raw_supported_history_tokens": int(payload.get("v102_swa_state_machine_supported_history_tokens", 0) or 0),
        "raw_supported_fallback_used": bool(payload.get("v102_swa_state_machine_supported_fallback_used", False)),
        "raw_soft_unsupported_min_keep": float(payload.get("v102_swa_state_machine_soft_unsupported_min_keep", 0.0) or 0.0),
        "raw_hold_prev_frames": int(payload.get("v102_swa_state_machine_hold_prev_frames", 0) or 0),
        "raw_hold_history_frames": int(payload.get("v102_swa_state_machine_hold_history_frames", 0) or 0),
        "raw_hold_reference_tokens": int(payload.get("v102_swa_state_machine_hold_reference_tokens", 0) or 0),
        "raw_hold_d_low_tokens": int(payload.get("v102_swa_state_machine_hold_d_low_tokens", 0) or 0),
        "raw_hold_semantic_static_tokens": int(
            payload.get("v102_swa_state_machine_hold_semantic_static_tokens", 0) or 0
        ),
        "raw_hold_k_stable_tokens": int(payload.get("v102_swa_state_machine_hold_k_stable_tokens", 0) or 0),
        "raw_hold_soft_min_keep": float(payload.get("v102_swa_state_machine_hold_soft_min_keep", 0.0) or 0.0),
        "raw_delay_current_tokens": int(payload.get("v102_swa_state_machine_delay_current_tokens", 0) or 0),
        "raw_delay_current_frac": float(payload.get("v102_swa_state_machine_delay_current_frac", 0.0) or 0.0),
        "raw_delay_current_soft_min_keep": float(
            payload.get("v102_swa_state_machine_delay_current_soft_min_keep", 0.0) or 0.0
        ),
        "raw_context_semantic_tokens": int(payload.get("v102_swa_state_machine_context_semantic_tokens", 0) or 0),
        "raw_context_scale_observable_tokens": int(
            payload.get("v102_swa_state_machine_context_scale_observable_tokens", 0) or 0
        ),
        "raw_context_d_low_tokens": int(payload.get("v102_swa_state_machine_context_d_low_tokens", 0) or 0),
        "raw_context_demoted_tokens": int(payload.get("v102_swa_state_machine_context_demoted_tokens", 0) or 0),
        "raw_context_soft_min_keep": float(payload.get("v102_swa_state_machine_context_soft_min_keep", 0.0) or 0.0),
    })
    return row


def hmc_summary_row(case_id: str, trace_root: Path) -> dict[str, Any]:
    path = trace_root / case_id / "READ_NO_ACTION" / "hmc_control_summary.jsonl"
    row: dict[str, Any] = {
        "hmc_summary_path": path.as_posix() if path.is_file() else "",
        "hmc_summary_line": "",
        "hmc_trace_available": False,
        "hmc_trace_applied_count": 0,
        "hmc_runtime_action_allowed_count": 0,
        "hmc_mean_rejected_history_tokens": "",
        "hmc_mean_attention_mass_removed_before": "",
        "hmc_actions": "",
        "hmc_reasons": "",
        "hmc_probe_impls": "",
        "hmc_read_error": "",
    }
    if not path.is_file():
        row["hmc_read_error"] = "missing_hmc_control_summary"
        return row
    best: dict[str, Any] | None = None
    best_line = -1
    for line_idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        swa = (
            payload.get("control_trace", {})
            .get("hook_effect_summary", {})
            .get("swa_read", {})
        )
        if isinstance(swa, dict) and int(swa.get("num_v102_swa_state_machine_trace_available", 0) or 0) > 0:
            best = swa
            best_line = line_idx
    if best is None:
        row["hmc_read_error"] = "no_v102_swa_read_summary"
        return row
    row.update({
        "hmc_summary_line": best_line,
        "hmc_trace_available": True,
        "hmc_trace_applied_count": int(best.get("num_v102_swa_state_machine_trace_applied", 0) or 0),
        "hmc_runtime_action_allowed_count": int(best.get("num_v102_swa_state_machine_runtime_action_allowed", 0) or 0),
        "hmc_mean_rejected_history_tokens": best.get("mean_v102_swa_state_machine_rejected_history_tokens", ""),
        "hmc_mean_attention_mass_removed_before": best.get("mean_attention_mass_removed_before", ""),
        "hmc_actions": ";".join(str(x) for x in best.get("values_v102_swa_state_machine_action", []) or []),
        "hmc_reasons": ";".join(str(x) for x in best.get("values_v102_swa_state_machine_reason", []) or []),
        "hmc_probe_impls": ";".join(str(x) for x in best.get("values_v102_swa_state_machine_probe_impl", []) or []),
    })
    return row


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# v102 State-Machine Action-Probe Closure Audit",
        "",
        f"- trace_root: `{summary.get('trace_root')}`",
        f"- target_count: {summary.get('target_count')}",
        f"- completed_job_count: {summary.get('completed_job_count')}",
        f"- failed_job_count: {summary.get('failed_job_count')}",
        f"- raw_trace_applied_count: {summary.get('raw_trace_applied_count')}",
        f"- raw_runtime_action_allowed_count: {summary.get('raw_runtime_action_allowed_count')}",
        f"- paired_metric_case_count: {summary.get('paired_metric_case_count')}",
        f"- action_probe_materialization_pass: {summary.get('action_probe_materialization_pass')}",
        f"- stage4_strict_memory_action_surface_pass: {summary.get('stage4_strict_memory_action_surface_pass')}",
        "",
        "Metric summary:",
        "",
        f"- relative_improvement_vs_baseline_head10_median: {summary.get('relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median')}",
        f"- relative_improvement_vs_baseline_overlap_future_median: {summary.get('relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median')}",
        f"- strict_positive_head10_relative_improvement_median: {summary.get('strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median')}",
        f"- safe_good_control_scale_cv_relative_improvement_median: {summary.get('safe_good_control_relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median')}",
        "",
        "Conclusion:",
        "",
        str(summary.get("conclusion", "")),
        "",
        "| case_id | role | raw_applied | runtime_allowed | rejected_history | attention_removed_before | reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {role} | {raw_applied} | {runtime_allowed} | {rejected} | {mass} | {reason} |".format(
                case_id=row.get("case_id", ""),
                role=row.get("ambiguous_or_control_role", ""),
                raw_applied=row.get("raw_trace_applied", ""),
                runtime_allowed=row.get("raw_runtime_action_allowed", ""),
                rejected=row.get("raw_rejected_history_tokens", ""),
                mass=row.get("hmc_mean_attention_mass_removed_before", ""),
                reason=str(row.get("raw_reason", "")).replace("|", "\\|"),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--metric-summary", type=Path, default=DEFAULT_METRIC_SUMMARY)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_rows = read_csv_rows(args.target_csv)
    run_summary = read_json(args.trace_root / "summary.json")
    metric_summary = read_json(args.metric_summary)
    rows: list[dict[str, Any]] = []
    for target in target_rows:
        case_id = str(target.get("case_id", "")).strip()
        row: dict[str, Any] = dict(target)
        row.update(raw_payload_row(case_id, args.trace_root))
        row.update(hmc_summary_row(case_id, args.trace_root))
        row["case_action_probe_materialized"] = (
            bool(row.get("raw_trace_available"))
            and bool(row.get("hmc_trace_available"))
            and bool(row.get("raw_trace_applied"))
            and int(row.get("hmc_trace_applied_count", 0) or 0) > 0
            and not bool(row.get("raw_runtime_action_allowed"))
            and int(row.get("hmc_runtime_action_allowed_count", 0) or 0) == 0
        )
        rows.append(row)

    target_count = len(target_rows)
    raw_trace_applied_count = sum(1 for row in rows if bool(row.get("raw_trace_applied")))
    raw_runtime_action_allowed_count = sum(1 for row in rows if bool(row.get("raw_runtime_action_allowed")))
    raw_supported_fallback_used_count = sum(1 for row in rows if bool(row.get("raw_supported_fallback_used")))
    hmc_runtime_action_allowed_count = sum(
        int(row.get("hmc_runtime_action_allowed_count", 0) or 0) for row in rows
    )
    failed_job_count = int(run_summary.get("failed_job_count", 1) or 0)
    paired_metric_count = int(metric_summary.get("paired_baseline_case_count", 0) or 0)
    action_probe_materialization_pass = (
        target_count > 0
        and int(run_summary.get("completed_job_count", 0) or 0) == target_count
        and failed_job_count == 0
        and raw_trace_applied_count == target_count
        and raw_runtime_action_allowed_count == 0
        and hmc_runtime_action_allowed_count == 0
        and all(bool(row.get("case_action_probe_materialized")) for row in rows)
    )
    true_l3_measurement_ready = paired_metric_count == target_count and int(metric_summary.get("ok_case_count", 0) or 0) == target_count
    summary = {
        "schema": "acl2_v102_state_machine_action_probe_closure_v1",
        "target_csv": args.target_csv.as_posix(),
        "trace_root": args.trace_root.as_posix(),
        "metric_summary": args.metric_summary.as_posix(),
        "target_count": target_count,
        "completed_job_count": run_summary.get("completed_job_count"),
        "failed_job_count": run_summary.get("failed_job_count"),
        "trace_payload_file_count": run_summary.get("trace_payload_file_count"),
        "per_chunk_geometry_sidecar_file_count": run_summary.get("per_chunk_geometry_sidecar_file_count"),
        "raw_trace_applied_count": raw_trace_applied_count,
        "raw_runtime_action_allowed_count": raw_runtime_action_allowed_count,
        "hmc_runtime_action_allowed_count": hmc_runtime_action_allowed_count,
        "raw_rejected_history_tokens_mean": (
            float(sum(float(row.get("raw_rejected_history_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_rejected_history_frac_mean": (
            float(sum(float(row.get("raw_rejected_history_frac", 0.0) or 0.0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_supported_history_tokens_mean": (
            float(sum(float(row.get("raw_supported_history_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_supported_d_low_tokens_mean": (
            float(sum(float(row.get("raw_supported_d_low_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_supported_semantic_static_tokens_mean": (
            float(sum(float(row.get("raw_supported_semantic_static_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_supported_k_stable_tokens_mean": (
            float(sum(float(row.get("raw_supported_k_stable_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_supported_fallback_used_count": raw_supported_fallback_used_count,
        "raw_soft_unsupported_min_keep_mean": (
            float(sum(float(row.get("raw_soft_unsupported_min_keep", 0.0) or 0.0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_hold_prev_frames_mean": (
            float(sum(float(row.get("raw_hold_prev_frames", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_hold_history_frames_mean": (
            float(sum(float(row.get("raw_hold_history_frames", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_hold_reference_tokens_mean": (
            float(sum(float(row.get("raw_hold_reference_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_hold_d_low_tokens_mean": (
            float(sum(float(row.get("raw_hold_d_low_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_hold_semantic_static_tokens_mean": (
            float(sum(float(row.get("raw_hold_semantic_static_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_hold_k_stable_tokens_mean": (
            float(sum(float(row.get("raw_hold_k_stable_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_hold_soft_min_keep_mean": (
            float(sum(float(row.get("raw_hold_soft_min_keep", 0.0) or 0.0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_delay_current_tokens_mean": (
            float(sum(float(row.get("raw_delay_current_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_delay_current_frac_mean": (
            float(sum(float(row.get("raw_delay_current_frac", 0.0) or 0.0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_delay_current_soft_min_keep_mean": (
            float(sum(float(row.get("raw_delay_current_soft_min_keep", 0.0) or 0.0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_context_semantic_tokens_mean": (
            float(sum(float(row.get("raw_context_semantic_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_context_scale_observable_tokens_mean": (
            float(sum(float(row.get("raw_context_scale_observable_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_context_d_low_tokens_mean": (
            float(sum(float(row.get("raw_context_d_low_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_context_demoted_tokens_mean": (
            float(sum(float(row.get("raw_context_demoted_tokens", 0) or 0) for row in rows) / target_count)
            if target_count else None
        ),
        "raw_context_soft_min_keep_mean": (
            float(sum(float(row.get("raw_context_soft_min_keep", 0.0) or 0.0) for row in rows) / target_count)
            if target_count else None
        ),
        "paired_metric_case_count": paired_metric_count,
        "true_l3_measurement_ready": true_l3_measurement_ready,
        "action_probe_materialization_pass": action_probe_materialization_pass,
        "stage3_strict_coverage_repaired": False,
        "stage4_strict_memory_action_surface_pass": False,
        "runtime_action_allowed": False,
        "stage5_allowed": False,
        "stage6_runtime_pilot_allowed": False,
        "stage7_full_validation_allowed": False,
        "relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median": metric_summary.get(
            "relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
        ),
        "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median": metric_summary.get(
            "relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
        ),
        "relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median": metric_summary.get(
            "relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
        ),
        "relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median": metric_summary.get(
            "relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median"
        ),
        "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median": metric_summary.get(
            "strict_clean_handoff_positive_relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m_median"
        ),
        "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median": metric_summary.get(
            "strict_clean_handoff_positive_relative_improvement_vs_baseline_overlap3_to_future_pose_sim3_rmse_m_median"
        ),
        "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median": metric_summary.get(
            "safe_good_control_relative_improvement_vs_baseline_local_sim3_ate_rmse_m_median"
        ),
        "safe_good_control_relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median": metric_summary.get(
            "safe_good_control_relative_improvement_vs_baseline_scale_cv_head_mid_tail_pose_sim3_median"
        ),
        "conclusion": (
            "The v102 diagnostic SWA action probe was materialized and did change the KV-cache read path "
            "on all selected cases while runtime_action_allowed stayed false.  Paired trajectory metrics "
            "are available, but this is not a Stage4 pass: Stage3 strict coverage remains false, the single "
            "strict clean handoff positive worsened on L3/scale proxy, and safe controls show local/scale harm."
        ),
    }
    write_csv(prefixed_path(args.output_prefix, "_rows.csv"), rows)
    write_json(prefixed_path(args.output_prefix, "_summary.json"), summary)
    write_text(prefixed_path(args.output_prefix, "_report.md"), build_report(summary, rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
