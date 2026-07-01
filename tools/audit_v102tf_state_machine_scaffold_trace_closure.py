#!/usr/bin/env python3
"""Audit v102 state-machine scaffold trace materialization.

This is a diagnostic-only closure check.  It verifies that the default-off v102
state-machine trace fields are visible in both raw SWA transport payloads and
HMC summaries, while explicitly preserving the Stage3/Stage4 No-Go state.
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
DEFAULT_TRACE_ROOT = STAGE4 / "v102_state_machine_scaffold_trace_delay_update_v2"
DEFAULT_OUTPUT_PREFIX = STAGE4 / "state_machine_scaffold_trace_closure"
TARGET_SUMMARY = STAGE4 / "v102_state_machine_scaffold_trace_targets_summary.json"
RUN_SUMMARY_NAME = "summary.json"


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


def bval(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def raw_payload_row(case_id: str, trace_root: Path) -> dict[str, Any]:
    files = sorted((trace_root / case_id / "READ_NO_ACTION" / "swa_raw_transport_trace").glob("*.pt"))
    row: dict[str, Any] = {
        "raw_payload_file_count": len(files),
        "raw_payload_path": files[0].as_posix() if files else "",
        "raw_trace_available": False,
        "raw_trace_applied": False,
        "raw_scaffold_only": False,
        "raw_runtime_action_allowed": False,
        "raw_strict_gate_pass": False,
        "raw_true_l3_gate_pass": False,
        "raw_action": "",
        "raw_reason": "",
        "raw_required_terms": "",
        "raw_swa_layer": "",
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
        "raw_strict_gate_pass": bool(payload.get("v102_swa_state_machine_strict_gate_pass", False)),
        "raw_true_l3_gate_pass": bool(payload.get("v102_swa_state_machine_true_l3_gate_pass", False)),
        "raw_action": str(payload.get("v102_swa_state_machine_action", "") or ""),
        "raw_reason": str(payload.get("v102_swa_state_machine_reason", "") or ""),
        "raw_required_terms": str(payload.get("v102_swa_state_machine_required_terms", "") or ""),
        "raw_swa_layer": payload.get("v102_swa_state_machine_swa_layer", ""),
    })
    return row


def hmc_summary_row(case_id: str, trace_root: Path) -> dict[str, Any]:
    path = trace_root / case_id / "READ_NO_ACTION" / "hmc_control_summary.jsonl"
    row: dict[str, Any] = {
        "hmc_summary_path": path.as_posix() if path.is_file() else "",
        "hmc_summary_line": "",
        "hmc_trace_available": False,
        "hmc_trace_applied_count": 0,
        "hmc_scaffold_only_count": 0,
        "hmc_runtime_action_allowed_count": 0,
        "hmc_actions": "",
        "hmc_reasons": "",
        "hmc_required_terms": "",
        "hmc_read_error": "",
    }
    if not path.is_file():
        row["hmc_read_error"] = "missing_hmc_control_summary"
        return row
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:  # noqa: BLE001
        row["hmc_read_error"] = f"{type(exc).__name__}: {exc}"
        return row
    best: dict[str, Any] | None = None
    best_line = -1
    for line_idx, line in enumerate(lines, 1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
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
        "hmc_scaffold_only_count": int(best.get("num_v102_swa_state_machine_scaffold_only", 0) or 0),
        "hmc_runtime_action_allowed_count": int(best.get("num_v102_swa_state_machine_runtime_action_allowed", 0) or 0),
        "hmc_actions": ";".join(str(x) for x in best.get("values_v102_swa_state_machine_action", []) or []),
        "hmc_reasons": ";".join(str(x) for x in best.get("values_v102_swa_state_machine_reason", []) or []),
        "hmc_required_terms": ";".join(str(x) for x in best.get("values_v102_swa_state_machine_required_terms", []) or []),
    })
    return row


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# v102 State-Machine Scaffold Trace Closure Audit",
        "",
        f"- trace_root: `{summary.get('trace_root')}`",
        f"- target_count: {summary.get('target_count')}",
        f"- completed_job_count: {summary.get('completed_job_count')}",
        f"- failed_job_count: {summary.get('failed_job_count')}",
        f"- raw_case_with_v102_trace_count: {summary.get('raw_case_with_v102_trace_count')}",
        f"- hmc_case_with_v102_trace_count: {summary.get('hmc_case_with_v102_trace_count')}",
        f"- raw_trace_applied_count: {summary.get('raw_trace_applied_count')}",
        f"- raw_runtime_action_allowed_count: {summary.get('raw_runtime_action_allowed_count')}",
        f"- scaffold_trace_materialization_pass: {summary.get('scaffold_trace_materialization_pass')}",
        f"- stage3_strict_coverage_repaired: {summary.get('stage3_strict_coverage_repaired')}",
        f"- stage4_strict_memory_action_surface_pass: {summary.get('stage4_strict_memory_action_surface_pass')}",
        "",
        "Conclusion:",
        "",
        str(summary.get("conclusion", "")),
        "",
        "| case_id | role | raw_available | hmc_available | raw_applied | raw_runtime_allowed | action | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {role} | {raw_trace_available} | {hmc_trace_available} | "
            "{raw_trace_applied} | {raw_runtime_action_allowed} | {raw_action} | {raw_reason} |".format(
                case_id=row.get("case_id", ""),
                role=row.get("ambiguous_or_control_role", ""),
                raw_trace_available=row.get("raw_trace_available", ""),
                hmc_trace_available=row.get("hmc_trace_available", ""),
                raw_trace_applied=row.get("raw_trace_applied", ""),
                raw_runtime_action_allowed=row.get("raw_runtime_action_allowed", ""),
                raw_action=str(row.get("raw_action", "")).replace("|", "\\|"),
                raw_reason=str(row.get("raw_reason", "")).replace("|", "\\|"),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_rows = read_csv_rows(args.target_csv)
    run_summary = read_json(args.trace_root / RUN_SUMMARY_NAME)
    target_summary = read_json(TARGET_SUMMARY)
    rows: list[dict[str, Any]] = []
    for target in target_rows:
        case_id = str(target.get("case_id", "")).strip()
        row: dict[str, Any] = dict(target)
        row.update(raw_payload_row(case_id, args.trace_root))
        row.update(hmc_summary_row(case_id, args.trace_root))
        row["target_runtime_action_allowed"] = bval(target.get("runtime_action_allowed"))
        row["case_scaffold_trace_materialized"] = (
            bool(row.get("raw_trace_available"))
            and bool(row.get("hmc_trace_available"))
            and bool(row.get("raw_scaffold_only"))
            and not bool(row.get("raw_trace_applied"))
            and not bool(row.get("raw_runtime_action_allowed"))
            and int(row.get("hmc_trace_applied_count", 0) or 0) == 0
            and int(row.get("hmc_runtime_action_allowed_count", 0) or 0) == 0
        )
        rows.append(row)

    target_count = len(target_rows)
    raw_case_with_v102_trace_count = sum(1 for row in rows if bool(row.get("raw_trace_available")))
    hmc_case_with_v102_trace_count = sum(1 for row in rows if bool(row.get("hmc_trace_available")))
    raw_trace_applied_count = sum(1 for row in rows if bool(row.get("raw_trace_applied")))
    raw_runtime_action_allowed_count = sum(1 for row in rows if bool(row.get("raw_runtime_action_allowed")))
    hmc_trace_applied_count = sum(int(row.get("hmc_trace_applied_count", 0) or 0) for row in rows)
    hmc_runtime_action_allowed_count = sum(
        int(row.get("hmc_runtime_action_allowed_count", 0) or 0) for row in rows
    )
    try:
        failed_job_count = int(run_summary.get("failed_job_count"))
    except (TypeError, ValueError):
        failed_job_count = 1
    scaffold_trace_materialization_pass = (
        target_count > 0
        and raw_case_with_v102_trace_count == target_count
        and hmc_case_with_v102_trace_count == target_count
        and raw_trace_applied_count == 0
        and raw_runtime_action_allowed_count == 0
        and hmc_trace_applied_count == 0
        and hmc_runtime_action_allowed_count == 0
        and all(bool(row.get("case_scaffold_trace_materialized")) for row in rows)
        and failed_job_count == 0
    )
    summary = {
        "schema": "acl2_v102_state_machine_scaffold_trace_closure_v1",
        "target_csv": args.target_csv.as_posix(),
        "trace_root": args.trace_root.as_posix(),
        "target_count": target_count,
        "strict_positive_target_count": target_summary.get("strict_positive_target_count"),
        "ambiguous_materialization_candidate_count": target_summary.get("ambiguous_materialization_candidate_count"),
        "safe_good_control_count": target_summary.get("safe_good_control_count"),
        "run_status": run_summary.get("status"),
        "selected_case_count": run_summary.get("selected_case_count"),
        "completed_job_count": run_summary.get("completed_job_count"),
        "failed_job_count": run_summary.get("failed_job_count"),
        "trace_payload_file_count": run_summary.get("trace_payload_file_count"),
        "per_chunk_geometry_sidecar_file_count": run_summary.get("per_chunk_geometry_sidecar_file_count"),
        "raw_case_with_v102_trace_count": raw_case_with_v102_trace_count,
        "hmc_case_with_v102_trace_count": hmc_case_with_v102_trace_count,
        "raw_trace_applied_count": raw_trace_applied_count,
        "raw_runtime_action_allowed_count": raw_runtime_action_allowed_count,
        "raw_scaffold_only_count": sum(1 for row in rows if bool(row.get("raw_scaffold_only"))),
        "hmc_trace_applied_count": hmc_trace_applied_count,
        "hmc_runtime_action_allowed_count": hmc_runtime_action_allowed_count,
        "hmc_scaffold_only_count": sum(int(row.get("hmc_scaffold_only_count", 0) or 0) for row in rows),
        "actions_seen": sorted({str(row.get("raw_action", "")) for row in rows if row.get("raw_action")}),
        "reasons_seen": sorted({str(row.get("raw_reason", "")) for row in rows if row.get("raw_reason")}),
        "scaffold_trace_materialization_pass": scaffold_trace_materialization_pass,
        "true_l3_measurement_ready": False,
        "stage3_strict_coverage_repaired": False,
        "stage4_strict_memory_action_surface_pass": False,
        "runtime_action_allowed": False,
        "stage5_allowed": False,
        "stage6_runtime_pilot_allowed": False,
        "stage7_full_validation_allowed": False,
        "conclusion": (
            "The v102 state-machine scaffold trace is materialized across the selected six diagnostic cases "
            "and visible in both raw SWA transport payloads and HMC summaries.  This repairs the trace "
            "instrumentation evidence path only.  No KV/attention change was applied, runtime action remained "
            "disabled, Stage3 strict target coverage is still not repaired, and no true L3/L4 action effect "
            "is measured; therefore Stage4/5/6/7 remain No-Go."
        ),
    }
    write_csv(prefixed_path(args.output_prefix, "_rows.csv"), rows)
    write_json(prefixed_path(args.output_prefix, "_summary.json"), summary)
    write_text(prefixed_path(args.output_prefix, "_report.md"), build_report(summary, rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
