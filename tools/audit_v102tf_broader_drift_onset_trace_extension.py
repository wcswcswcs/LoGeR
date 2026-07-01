#!/usr/bin/env python3
"""Audit v102 broader drift-onset trace extension materialization.

This diagnostic verifies the no-action trace/sidecar run used to repair a
missing full-control support gap for broader drift-onset exploration cases. It
does not evaluate a runtime action and cannot authorize Stage4/5/6/7.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
STAGE3 = ROOT / "stage3_semantic_oracle_upper_bound"
TARGET_CSV = STAGE3 / "broader_drift_onset_trace_extension_targets.csv"
TRACE_ROOT = STAGE3 / "broader_drift_onset_trace_extension_traces"
OUT_ROWS = STAGE3 / "broader_drift_onset_trace_extension_audit_rows.csv"
OUT_SUMMARY = STAGE3 / "broader_drift_onset_trace_extension_summary.json"
OUT_REPORT = STAGE3 / "broader_drift_onset_trace_extension_report.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


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
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def shape(value: Any) -> str:
    if torch.is_tensor(value):
        return "x".join(str(int(dim)) for dim in value.shape)
    return ""


def tensor_present(payload: dict[str, Any], key: str) -> bool:
    return torch.is_tensor(payload.get(key))


def load_torch(path: Path) -> tuple[Any, str]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}:{exc}"


def audit_case(target: dict[str, str]) -> dict[str, Any]:
    case_id = target.get("case_id", "")
    run = TRACE_ROOT / case_id / "READ_NO_ACTION"
    trace_files = sorted((run / "swa_raw_transport_trace").glob("*.pt"))
    sidecar_files = sorted((run / "per_chunk_geometry").glob("chunk_*.pt"))
    pose_trace = run / "per_chunk_pose_trace.jsonl"
    payload, payload_error = load_torch(trace_files[0]) if trace_files else (None, "missing_trace_payload")
    payload_is_dict = isinstance(payload, dict)
    lifecycle_rows = payload.get("ttt_prev_stable_anchor_lifecycle_rows", []) if payload_is_dict else []
    hit = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask") if payload_is_dict else None
    anchor_ids = payload.get("current_Q_to_cache_K_topk_ttt_prev_stable_anchor_ids") if payload_is_dict else None
    hit_count = int(hit.detach().cpu().bool().sum().item()) if torch.is_tensor(hit) else 0
    nonnegative_anchor_count = (
        int((anchor_ids.detach().cpu().long() >= 0).sum().item()) if torch.is_tensor(anchor_ids) else 0
    )
    sidecar_read_ok = 0
    sidecar_schema_ok = 0
    sidecar_shapes: list[dict[str, str]] = []
    sidecar_errors: list[str] = []
    for path in sidecar_files:
        geo, error = load_torch(path)
        if error:
            sidecar_errors.append(f"{path.name}:{error}")
            continue
        if not isinstance(geo, dict):
            sidecar_errors.append(f"{path.name}:sidecar_not_dict")
            continue
        sidecar_read_ok += 1
        required = {
            "camera_poses": shape(geo.get("camera_poses")),
            "local_points": shape(geo.get("local_points")),
            "points": shape(geo.get("points")),
            "conf": shape(geo.get("conf")),
        }
        if all(required.values()):
            sidecar_schema_ok += 1
        sidecar_shapes.append({"file": str(path), **required})
    trace_materialized = (
        len(trace_files) >= 1
        and payload_is_dict
        and payload_error == ""
        and len(sidecar_files) >= 2
        and sidecar_schema_ok >= 2
        and pose_trace.is_file()
    )
    return {
        "case_id": case_id,
        "target_taxonomy": target.get("target_taxonomy", ""),
        "case_label": target.get("case_label", ""),
        "failure_type": target.get("failure_type", ""),
        "L3_handoff_transfer_penalty_proxy": target.get("L3_handoff_transfer_penalty_proxy", ""),
        "runner_status": read_json(TRACE_ROOT / "summary.json").get("status", ""),
        "trace_payload_count": len(trace_files),
        "trace_payload": str(trace_files[0]) if trace_files else "",
        "payload_read_ok": payload_is_dict and payload_error == "",
        "payload_error": payload_error,
        "lifecycle_row_count": len(lifecycle_rows) if isinstance(lifecycle_rows, list) else 0,
        "lifecycle_rows_with_seed_mode": sum(
            1
            for row in lifecycle_rows
            if isinstance(row, dict) and row.get("source_stage_c_seed_global_track_idx_mode") not in (None, "")
        )
        if isinstance(lifecycle_rows, list)
        else 0,
        "topk_hit_mask_present": tensor_present(payload, "current_Q_to_cache_K_topk_ttt_prev_stable_anchor_hit_mask")
        if payload_is_dict
        else False,
        "topk_hit_mask_shape": shape(hit),
        "topk_hit_count": hit_count,
        "nonnegative_anchor_id_count": nonnegative_anchor_count,
        "sampled_query_indices_shape": shape(payload.get("sampled_query_indices")) if payload_is_dict else "",
        "sidecar_count": len(sidecar_files),
        "sidecar_read_ok_count": sidecar_read_ok,
        "sidecar_schema_ok_count": sidecar_schema_ok,
        "sidecar_shapes": sidecar_shapes,
        "sidecar_errors": ";".join(sidecar_errors),
        "pose_trace_exists": pose_trace.is_file(),
        "trace_sidecar_materialized": trace_materialized,
        "strict_stage3_promotion_role": "exploration_only_non_swa_read_case",
        "runtime_action_allowed": False,
        "claim_level": "v102_broader_drift_onset_trace_extension_diagnostic_no_action",
    }


def main() -> int:
    targets = read_rows(TARGET_CSV)
    rows = [audit_case(target) for target in targets]
    materialized = [row for row in rows if row["trace_sidecar_materialized"]]
    summary = {
        "schema": "acl2_v102_broader_drift_onset_trace_extension_audit_v1",
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "target_count": len(targets),
        "runner_status": read_json(TRACE_ROOT / "summary.json").get("status", ""),
        "completed_job_count": read_json(TRACE_ROOT / "summary.json").get("completed_job_count"),
        "failed_job_count": read_json(TRACE_ROOT / "summary.json").get("failed_job_count"),
        "trace_sidecar_materialized_count": len(materialized),
        "trace_sidecar_materialized_cases": ";".join(row["case_id"] for row in materialized),
        "all_targets_trace_sidecar_materialized": bool(targets) and len(materialized) == len(targets),
        "strict_stage3_coverage_repaired": False,
        "stage4_allowed": False,
        "blocker": (
            "Trace/sidecar materialization repaired the missing support artifact for the broader drift-onset case, "
            "but the case remains exploration-only READ/non-SWA evidence and does not add a strict clean handoff positive "
            "or a true L3 action-surface effect."
        ),
        "outputs": {
            "rows": OUT_ROWS.as_posix(),
            "report": OUT_REPORT.as_posix(),
        },
    }
    write_rows(OUT_ROWS, rows)
    write_json(OUT_SUMMARY, summary)
    write_text(
        OUT_REPORT,
        "\n".join(
            [
                "# Broader Drift-Onset Trace Extension Audit",
                "",
                f"- target_count: {summary['target_count']}",
                f"- runner_status: {summary['runner_status']}",
                f"- completed_job_count: {summary['completed_job_count']}",
                f"- failed_job_count: {summary['failed_job_count']}",
                f"- trace_sidecar_materialized_count: {summary['trace_sidecar_materialized_count']}",
                f"- trace_sidecar_materialized_cases: {summary['trace_sidecar_materialized_cases']}",
                f"- strict_stage3_coverage_repaired: {summary['strict_stage3_coverage_repaired']}",
                f"- runtime_action_allowed: {summary['runtime_action_allowed']}",
                "",
                "Conclusion:",
                "",
                summary["blocker"],
            ]
        ),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
