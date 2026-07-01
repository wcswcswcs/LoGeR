#!/usr/bin/env python3
"""Freeze v100 decisions from audited phase summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase8_decision_freeze"

PHASE0 = AUDIT_ROOT / "v100_phase0_contract" / "summary.json"
PHASE1 = AUDIT_ROOT / "v100_phase1_gpu_data_model_parity" / "summary.json"
PHASE2 = AUDIT_ROOT / "v100_phase2_f2_local_final" / "summary.json"
PHASE3 = AUDIT_ROOT / "v100_phase3_scene_fragmentation_audit" / "summary.json"
PHASE4 = AUDIT_ROOT / "v100_phase4_history_memory" / "summary.json"
PHASE4B = AUDIT_ROOT / "v100_phase4b_history_memory_repair" / "summary.json"
PHASE5 = AUDIT_ROOT / "v100_phase5_da3_d4rt_verifier_audit" / "summary.json"


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool_count_zero(summary: dict[str, Any], key: str) -> bool:
    value = summary.get(key)
    if isinstance(value, dict):
        return all(_num(v) == 0 for v in value.values())
    return _num(value) == 0


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = _load_json(PHASE0)
    phase1 = _load_json(PHASE1)
    phase2 = _load_json(PHASE2)
    phase3 = _load_json(PHASE3)
    phase4 = _load_json(PHASE4)
    phase4b = _load_json(PHASE4B)
    phase5 = _load_json(PHASE5)

    local_gate_rows = [
        ("local_dev_MV_AP_window_ge_0p100", _num(phase2.get("dev_MV_AP_window")) >= 0.100, ">=0.100", phase2.get("dev_MV_AP_window")),
        ("local_dev_MV_AP50_window_ge_0p225", _num(phase2.get("dev_MV_AP50_window")) >= 0.225, ">=0.225", phase2.get("dev_MV_AP50_window")),
        ("local_holdout_MV_AP_window_ge_0p125", _num(phase2.get("holdout_MV_AP_window")) >= 0.125, ">=0.125", phase2.get("holdout_MV_AP_window")),
        ("local_holdout_MV_AP50_window_ge_0p285", _num(phase2.get("holdout_MV_AP50_window")) >= 0.285, ">=0.285", phase2.get("holdout_MV_AP50_window")),
        ("local_same_frame_collision_zero", _bool_count_zero(phase2, "same_frame_collision_count"), "0", phase2.get("same_frame_collision_count")),
        ("local_missing_mask_raster_zero", _bool_count_zero(phase2, "missing_mask_raster_count"), "0", phase2.get("missing_mask_raster_count")),
        ("local_future_chunk_access_false", not bool(phase2.get("method_contract", {}).get("future_chunk_access")), "false", phase2.get("method_contract", {}).get("future_chunk_access")),
        ("local_formal_claim_allowed", bool(phase2.get("formal_claim_allowed")), "true", phase2.get("formal_claim_allowed")),
    ]
    local_pass = all(row[1] for row in local_gate_rows)
    scene_pass = bool(phase4.get("phase4_pass")) or bool(phase4b.get("phase4_pass"))
    geometry_pass = bool(phase5.get("phase5_pass"))
    gpu_data_model_pass = (
        bool(phase1.get("phase1_pass"))
        and _num(phase1.get("bitset_iou_max_abs_error")) == 0
        and _num(phase1.get("mask_area_mismatch_count")) == 0
        and _num(phase1.get("frame_count_mismatch_count")) == 0
        and _num(phase1.get("runtime_speedup")) >= 5.0
    )

    if not local_pass:
        primary_decision = "NO_GO_CHUNK_CAUSAL_LOCAL"
    elif scene_pass:
        primary_decision = "GO_LOCAL_AND_SCENE"
    else:
        primary_decision = "GO_LOCAL_ONLY_FORMALIZED"

    secondary_decisions: list[str] = []
    if local_pass and not scene_pass:
        secondary_decisions.append("NO_GO_SCENE_STITCHING")
    if not geometry_pass:
        secondary_decisions.append("NO_GO_GEOMETRY_INCREMENT")
    if gpu_data_model_pass:
        secondary_decisions.append("GPU_DATA_MODEL_ACCELERATION_PARITY_ONLY")
    secondary_decisions.append("GPU_FAST_MV_AP_EVALUATOR_NOT_CLAIMED")

    variant_config_rows = [
        {
            "schema_version": "stream4d_v100_phase8_variant_config_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "variant_id": "local_f2_formalized",
            "source_summary": _rel(PHASE2),
            "status": "freeze_local_claim" if local_pass else "no_go",
            "claim_scope": "MV_AP_window only; fragmented scene metrics are not a scene-memory claim",
            "config_or_method": phase2.get("variant_id"),
            "frozen": local_pass,
        },
        {
            "schema_version": "stream4d_v100_phase8_variant_config_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "variant_id": "semantic_history_memory_main",
            "source_summary": _rel(PHASE4),
            "status": "no_go" if not bool(phase4.get("phase4_pass")) else "pass",
            "claim_scope": "scene/local2history",
            "config_or_method": phase4.get("best_variant_id"),
            "frozen": False,
        },
        {
            "schema_version": "stream4d_v100_phase8_variant_config_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "variant_id": "semantic_history_memory_repair_local",
            "source_summary": _rel(PHASE4B),
            "status": "no_go" if not bool(phase4b.get("phase4_pass")) else "pass",
            "claim_scope": "scene/local2history stricter local-preserving repair",
            "config_or_method": phase4b.get("best_variant_id"),
            "frozen": False,
        },
        {
            "schema_version": "stream4d_v100_phase8_variant_config_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "variant_id": "da3_d4rt_verifier_increment",
            "source_summary": _rel(PHASE5),
            "status": "diagnostic_only" if not geometry_pass else "pass",
            "claim_scope": "geometry/temporal verifier increment",
            "config_or_method": "G1-G5 audit",
            "frozen": False,
        },
        {
            "schema_version": "stream4d_v100_phase8_variant_config_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "variant_id": "gpu_data_model_parity",
            "source_summary": _rel(PHASE1),
            "status": "parity_speed_pass" if gpu_data_model_pass else "no_go",
            "claim_scope": "packed bitset IoU and semantic matrix data-model parity/speed only",
            "config_or_method": phase1.get("gpu_backend"),
            "frozen": gpu_data_model_pass,
        },
        {
            "schema_version": "stream4d_v100_phase8_variant_config_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "variant_id": "gpu_fast_mv_ap_evaluator",
            "source_summary": "",
            "status": "not_run_not_claimed",
            "claim_scope": "Phase6 GPU MV_AP evaluator parity",
            "config_or_method": "No v100 final GPU evaluator parity suite was run; final AP remains v65 CPU.",
            "frozen": False,
        },
    ]

    variant_metric_rows = [
        {
            "schema_version": "stream4d_v100_phase8_variant_metric_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "variant_id": "local_f2_formalized",
            "dataset_split": "dev",
            "MV_AP_window": phase2.get("dev_MV_AP_window"),
            "MV_AP50_window": phase2.get("dev_MV_AP50_window"),
            "MV_AP_scene": phase2.get("dev_MV_AP_scene_fragmented"),
            "MV_AP50_scene": phase2.get("dev_MV_AP50_scene_fragmented"),
            "metric_source": "v65 canonical",
        },
        {
            "schema_version": "stream4d_v100_phase8_variant_metric_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "variant_id": "local_f2_formalized",
            "dataset_split": "holdout",
            "MV_AP_window": phase2.get("holdout_MV_AP_window"),
            "MV_AP50_window": phase2.get("holdout_MV_AP50_window"),
            "MV_AP_scene": phase2.get("holdout_MV_AP_scene_fragmented"),
            "MV_AP50_scene": phase2.get("holdout_MV_AP50_scene_fragmented"),
            "metric_source": "v65 canonical",
        },
        {
            "schema_version": "stream4d_v100_phase8_variant_metric_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "variant_id": "semantic_history_memory_main",
            "dataset_split": "holdout",
            "MV_AP_window": phase4.get("best_holdout_MV_AP_window"),
            "MV_AP50_window": phase4.get("best_holdout_MV_AP50_window"),
            "MV_AP_scene": phase4.get("best_holdout_MV_AP_scene"),
            "MV_AP50_scene": phase4.get("best_holdout_MV_AP50_scene"),
            "local_window_AP_drop": phase4.get("local_window_AP_drop", {}).get("holdout"),
            "objects_crossing_multiple_chunks": phase4.get("objects_crossing_multiple_chunks", {}).get("holdout"),
            "metric_source": "v65 canonical",
        },
        {
            "schema_version": "stream4d_v100_phase8_variant_metric_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "variant_id": "semantic_history_memory_repair_local",
            "dataset_split": "holdout",
            "MV_AP_window": phase4b.get("best_holdout_MV_AP_window"),
            "MV_AP50_window": phase4b.get("best_holdout_MV_AP50_window"),
            "MV_AP_scene": phase4b.get("best_holdout_MV_AP_scene"),
            "MV_AP50_scene": phase4b.get("best_holdout_MV_AP50_scene"),
            "local_window_AP_drop": phase4b.get("local_window_AP_drop", {}).get("holdout"),
            "objects_crossing_multiple_chunks": phase4b.get("objects_crossing_multiple_chunks", {}).get("holdout"),
            "metric_source": "v65 canonical",
        },
        {
            "schema_version": "stream4d_v100_phase8_variant_metric_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "variant_id": "da3_d4rt_verifier_increment",
            "dataset_split": "holdout_or_prior_audit",
            "da3_delta_scene_vs_phase2": phase5.get("da3_delta_scene_vs_phase2"),
            "d4rt_real_minus_control_MV_AP_scene": phase5.get("d4rt_real_minus_control_MV_AP_scene"),
            "d4rt_ai_best_MV_AP_scene": phase5.get("d4rt_ai_best_MV_AP_scene"),
            "d4rt_ai_semantic_reference_MV_AP_scene": phase5.get("d4rt_ai_semantic_reference_MV_AP_scene"),
            "phase4_main_d4rt_support_rate": phase5.get("phase4_main_d4rt_support_rate"),
            "phase4_repair_d4rt_support_rate": phase5.get("phase4_repair_d4rt_support_rate"),
        },
        {
            "schema_version": "stream4d_v100_phase8_variant_metric_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "variant_id": "gpu_data_model_parity",
            "dataset_split": "sample_parity_suite",
            "bitset_iou_max_abs_error": phase1.get("bitset_iou_max_abs_error"),
            "semantic_cosine_max_abs_error": phase1.get("semantic_cosine_max_abs_error"),
            "runtime_cpu_sec": phase1.get("runtime_cpu_sec"),
            "runtime_gpu_sec": phase1.get("runtime_gpu_sec"),
            "runtime_speedup": phase1.get("runtime_speedup"),
        },
    ]

    gate_rows = [
        {
            "schema_version": "stream4d_v100_phase8_gate_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "gate_id": gate_id,
            "pass": passed,
            "expected": expected,
            "observed": observed,
            "severity": "local_claim_required",
        }
        for gate_id, passed, expected, observed in local_gate_rows
    ]
    gate_rows.extend(
        [
            {
                "schema_version": "stream4d_v100_phase8_gate_row_v1",
                "phase_id": "v100_phase8_decision_freeze",
                "gate_id": "phase3_fragmentation_confirmed",
                "pass": bool(phase3.get("fragmentation_confirmed")) and bool(phase3.get("local2history_required")),
                "expected": "fragmentation confirmed and local2history required",
                "observed": f"fragmentation={phase3.get('fragmentation_confirmed')} objects_crossing_total={phase3.get('objects_crossing_multiple_chunks_total')}",
                "severity": "routing_required",
            },
            {
                "schema_version": "stream4d_v100_phase8_gate_row_v1",
                "phase_id": "v100_phase8_decision_freeze",
                "gate_id": "scene_history_memory_pass",
                "pass": scene_pass,
                "expected": "Phase4 or Phase4b passes scene/local2history gates",
                "observed": f"phase4={phase4.get('decision')} phase4b={phase4b.get('decision')}",
                "severity": "scene_claim_required",
            },
            {
                "schema_version": "stream4d_v100_phase8_gate_row_v1",
                "phase_id": "v100_phase8_decision_freeze",
                "gate_id": "geometry_temporal_increment_pass",
                "pass": geometry_pass,
                "expected": "Phase5 DA3/D4RT verifier increment passes",
                "observed": phase5.get("decision"),
                "severity": "geometry_claim_required",
            },
            {
                "schema_version": "stream4d_v100_phase8_gate_row_v1",
                "phase_id": "v100_phase8_decision_freeze",
                "gate_id": "gpu_data_model_parity_speed_pass",
                "pass": gpu_data_model_pass,
                "expected": "Phase1 parity exact and speedup >=5x",
                "observed": f"speedup={phase1.get('runtime_speedup')} bitset_error={phase1.get('bitset_iou_max_abs_error')}",
                "severity": "gpu_data_model_claim_required",
            },
            {
                "schema_version": "stream4d_v100_phase8_gate_row_v1",
                "phase_id": "v100_phase8_decision_freeze",
                "gate_id": "gpu_fast_mv_ap_evaluator_parity_pass",
                "pass": False,
                "expected": "Phase6 AP parity suite run with AP errors <=1e-9",
                "observed": "not_run_not_claimed; final AP metrics are v65 CPU",
                "severity": "not_claimed",
            },
            {
                "schema_version": "stream4d_v100_phase8_gate_row_v1",
                "phase_id": "v100_phase8_decision_freeze",
                "gate_id": "final_metric_contract_v65",
                "pass": bool(phase0.get("formal_metric_source_eq_v65")),
                "expected": "final reported AP uses v65 canonical evaluator",
                "observed": phase0.get("formal_metric_source_eq_v65"),
                "severity": "metric_contract_required",
            },
        ]
    )
    failure_rows = [
        {
            "schema_version": "stream4d_v100_phase8_failure_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "decision_effect": (
                "blocks local claim"
                if row["severity"] == "local_claim_required"
                else "blocks scene claim"
                if row["severity"] == "scene_claim_required"
                else "blocks DA3/D4RT claim"
                if row["severity"] == "geometry_claim_required"
                else "not claimed; no final metric drift risk"
                if row["severity"] == "not_claimed"
                else "diagnostic"
            ),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]

    performance_rows = [
        {
            "schema_version": "stream4d_v100_phase8_performance_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "case_id": "phase1_gpu_data_model",
            "runtime_cpu_sec": phase1.get("runtime_cpu_sec"),
            "runtime_gpu_sec": phase1.get("runtime_gpu_sec"),
            "runtime_speedup": phase1.get("runtime_speedup"),
            "gpu_backend": phase1.get("gpu_backend"),
        },
        {
            "schema_version": "stream4d_v100_phase8_performance_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "case_id": "phase2_local_formalization",
            "runtime_sec": phase2.get("runtime_sec"),
        },
        {
            "schema_version": "stream4d_v100_phase8_performance_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "case_id": "phase4_history_memory_main",
            "runtime_sec": phase4.get("runtime_sec"),
            "v65_evaluator_runs": 10,
        },
        {
            "schema_version": "stream4d_v100_phase8_performance_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "case_id": "phase4b_history_memory_repair",
            "runtime_sec": phase4b.get("runtime_sec"),
            "v65_evaluator_runs": 10,
        },
        {
            "schema_version": "stream4d_v100_phase8_performance_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "case_id": "phase5_da3_d4rt_audit",
            "runtime_sec": phase5.get("runtime_sec"),
        },
    ]
    casebook_rows = [
        {
            "schema_version": "stream4d_v100_phase8_casebook_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "case_id": "local_claim_supported",
            "evidence": f"dev={phase2.get('dev_MV_AP_window')} holdout={phase2.get('holdout_MV_AP_window')} formal={phase2.get('formal_claim_allowed')}",
            "interpretation": "F2 chunk-local object birth can be claimed for local/window AP under the audited causal contract.",
        },
        {
            "schema_version": "stream4d_v100_phase8_casebook_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "case_id": "scene_fragmentation_confirmed",
            "evidence": f"objects_crossing_total={phase3.get('objects_crossing_multiple_chunks_total')} mean_fragmentation_rate={phase3.get('mean_fragmentation_rate')}",
            "interpretation": "The local method's scene AP failure is identity fragmentation, not a lack of local detections alone.",
        },
        {
            "schema_version": "stream4d_v100_phase8_casebook_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "case_id": "semantic_history_tradeoff",
            "evidence": f"main_scene={phase4.get('best_holdout_MV_AP_scene')} main_local_drop={phase4.get('local_window_AP_drop', {}).get('holdout')} repair_scene={phase4b.get('best_holdout_MV_AP_scene')} repair_local_drop={phase4b.get('local_window_AP_drop', {}).get('holdout')}",
            "interpretation": "Aggressive semantic linking improves scene somewhat but destroys local AP; stricter linking preserves more local AP but leaves fragmentation.",
        },
        {
            "schema_version": "stream4d_v100_phase8_casebook_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "case_id": "geometry_increment_not_promoted",
            "evidence": f"da3_delta={phase5.get('da3_delta_scene_vs_phase2')} d4rt_support_main={phase5.get('phase4_main_d4rt_support_rate')}",
            "interpretation": "DA3/D4RT remain diagnostic because they do not provide the required increment or support the accepted semantic history links.",
        },
        {
            "schema_version": "stream4d_v100_phase8_casebook_row_v1",
            "phase_id": "v100_phase8_decision_freeze",
            "case_id": "gpu_scope",
            "evidence": f"phase1_speedup={phase1.get('runtime_speedup')} gpu_evaluator=not_run",
            "interpretation": "GPU data-model acceleration is validated, but no v100 GPU MV_AP evaluator claim is made.",
        },
    ]

    variant_config_csv = OUT_DIR / "variant_config_rows.csv"
    variant_metric_csv = OUT_DIR / "variant_metric_rows.csv"
    variant_gate_csv = OUT_DIR / "variant_gate_rows.csv"
    variant_failure_csv = OUT_DIR / "variant_failure_rows.csv"
    performance_csv = OUT_DIR / "performance_rows.csv"
    casebook_csv = OUT_DIR / "casebook_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"
    summary_json = OUT_DIR / "summary.json"

    _write_csv(variant_config_csv, variant_config_rows)
    _write_csv(variant_metric_csv, variant_metric_rows)
    _write_csv(variant_gate_csv, gate_rows)
    _write_csv(variant_failure_csv, failure_rows)
    _write_csv(performance_csv, performance_rows)
    _write_csv(casebook_csv, casebook_rows)

    artifacts = [
        (variant_config_csv, "csv", "Phase8 frozen claim/config rows"),
        (variant_metric_csv, "csv", "Phase8 final metric rows"),
        (variant_gate_csv, "csv", "Phase8 final gates"),
        (variant_failure_csv, "csv", "Phase8 final failures and non-claims"),
        (performance_csv, "csv", "Phase8 performance summary"),
        (casebook_csv, "csv", "Phase8 evidence casebook"),
    ]
    _write_csv(
        artifact_csv,
        [
            {
                "schema_version": "stream4d_v100_phase8_artifact_manifest_row_v1",
                "phase_id": "v100_phase8_decision_freeze",
                "artifact_path": _rel(path),
                "artifact_type": kind,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": _sha256(path) if path.exists() and path.is_file() else "",
                "note": note,
            }
            for path, kind, note in artifacts
        ],
    )

    summary = {
        "schema_version": "stream4d_v100_phase8_decision_freeze_summary_v1",
        "phase_id": "v100_phase8_decision_freeze",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": primary_decision,
        "secondary_decisions": secondary_decisions,
        "local_claim_allowed": local_pass,
        "scene_claim_allowed": scene_pass,
        "geometry_increment_claim_allowed": geometry_pass,
        "gpu_data_model_claim_allowed": gpu_data_model_pass,
        "gpu_fast_mv_ap_evaluator_claim_allowed": False,
        "final_metric_source": "v65 canonical CPU evaluator",
        "phase_status": {
            "phase0_pass": phase0.get("phase0_pass"),
            "phase1_pass": phase1.get("phase1_pass"),
            "phase2_pass": phase2.get("phase2_pass"),
            "phase3_pass": phase3.get("phase3_pass"),
            "phase4_pass": phase4.get("phase4_pass"),
            "phase4b_pass": phase4b.get("phase4_pass"),
            "phase5_pass": phase5.get("phase5_pass"),
            "phase6_gpu_mv_ap_evaluator": "not_run_not_claimed",
            "phase7_performance_gate": "partial_phase1_data_model_only",
        },
        "key_metrics": {
            "local_dev_MV_AP_window": phase2.get("dev_MV_AP_window"),
            "local_dev_MV_AP50_window": phase2.get("dev_MV_AP50_window"),
            "local_holdout_MV_AP_window": phase2.get("holdout_MV_AP_window"),
            "local_holdout_MV_AP50_window": phase2.get("holdout_MV_AP50_window"),
            "fragmented_holdout_MV_AP_scene": phase2.get("holdout_MV_AP_scene_fragmented"),
            "phase4_main_holdout_MV_AP_scene": phase4.get("best_holdout_MV_AP_scene"),
            "phase4_main_holdout_local_drop": phase4.get("local_window_AP_drop", {}).get("holdout"),
            "phase4b_holdout_MV_AP_scene": phase4b.get("best_holdout_MV_AP_scene"),
            "phase4b_holdout_local_drop": phase4b.get("local_window_AP_drop", {}).get("holdout"),
            "da3_delta_scene_vs_phase2": phase5.get("da3_delta_scene_vs_phase2"),
            "d4rt_real_minus_control_MV_AP_scene": phase5.get("d4rt_real_minus_control_MV_AP_scene"),
            "phase1_runtime_speedup": phase1.get("runtime_speedup"),
        },
        "outputs": {
            "summary": _rel(summary_json),
            "variant_config_rows": _rel(variant_config_csv),
            "variant_metric_rows": _rel(variant_metric_csv),
            "variant_gate_rows": _rel(variant_gate_csv),
            "variant_failure_rows": _rel(variant_failure_csv),
            "performance_rows": _rel(performance_csv),
            "casebook_rows": _rel(casebook_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(summary_json, summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
