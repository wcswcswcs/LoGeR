#!/usr/bin/env python3
"""Build Stream4D v103 Phase0 contract artifacts.

Phase0 is intentionally read-only with respect to method predictions. It locks
the evaluator, current F2 baseline, prior v100/v101/v102 boundaries, causal
contract, provider/input availability, and GPU/package environment before any
primitive-affinity method experiment is allowed to run.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v103_phase0_contract"

PLAN_DOC = ROOT / "docs" / "stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"
METHOD_DOC = ROOT / "docs" / "stream4d_v103_method_thinking_training_free_primitive_affinity_field.md"
V65_EVALUATOR = STREAM3D / "tools" / "run_v65_scene_multiview_ap.py"

PHASE2C_DIR = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
V100_FINAL_DIR = AUDIT_ROOT / "v100_phase8e_final_decision_freeze"
V101_FINAL_DIR = AUDIT_ROOT / "v101_final_closeout"
V101_PHASE2_DIR = AUDIT_ROOT / "v101_phase2_geometry_provider_capability"
V102_FINAL_DIR = AUDIT_ROOT / "v102_final_closeout"
V102_PHASE5C_DIR = AUDIT_ROOT / "v102_phase5c_semantic_barrier_bridge_repair"
V102_PHASE7J_DIR = AUDIT_ROOT / "v102_phase7j_f2_primitive_support_fill_diagnostic"

SOURCE_ROWS_DEV = AUDIT_ROOT / "v95_phase1_physical_source_registry/source_container_rows.csv"
SOURCE_ROWS_HOLDOUT = AUDIT_ROOT / "v98_phase13_holdout/source_container_rows.csv"
RADIO_DEV = AUDIT_ROOT / "v91_radio_mask_features_npz/mask_features.npz"
RADIO_DEV_SCENE0011 = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_features.npz"
RADIO_DEV_SCENE0050 = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_features.npz"
RADIO_HOLDOUT = AUDIT_ROOT / "v98_phase13_holdout_radio_features_npz/mask_features.npz"
CROPFORMER_DEV = STREAM3D / "outputs/cache/v66_cropformer_chunk_masks"
CROPFORMER_HOLDOUT = STREAM3D / "outputs/cache/v65_cropformer_chunk_masks"
D4RT_ROOT = ROOT / "Open-d4rt"
D4RT_CKPT_32 = D4RT_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG"
D4RT_QUERY_TOOL = STREAM3D / "tools/run_v63_d4rt_query.py"
D4RT_PROVIDER_AUDIT = AUDIT_ROOT / "v99_phase10ae_d4rt_da3grid_provider_audit/summary.json"

EXPECTED_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_count(path: Path) -> int | str:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".csv":
        return ""
    with path.open(newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_v65_thresholds() -> tuple[list[float], str, bool, bool]:
    if str(STREAM3D) not in sys.path:
        sys.path.insert(0, str(STREAM3D))
    try:
        from tools import run_v65_scene_multiview_ap as v65  # noqa: WPS433

        thresholds = [float(x) for x in v65.AP_THRESHOLDS]
        return thresholds, "", hasattr(v65, "SparseSceneIoU"), hasattr(v65, "_summarize_iou")
    except Exception as exc:
        return [], repr(exc), False, False


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return ""


def _nvidia_rows() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used",
        "--format=csv,noheader",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    rows: list[dict[str, Any]] = []
    if proc.returncode != 0:
        return [
            {
                "schema_version": "stream4d_v103_phase0_gpu_env_row_v1",
                "phase_id": "v103_phase0_contract",
                "probe": "nvidia_smi",
                "probe_returncode": proc.returncode,
                "stderr": proc.stderr.strip(),
            }
        ]
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "schema_version": "stream4d_v103_phase0_gpu_env_row_v1",
                "phase_id": "v103_phase0_contract",
                "probe": "nvidia_smi",
                "gpu_index": parts[0],
                "gpu_name": parts[1],
                "memory_total": parts[2],
                "memory_used": parts[3],
                "gpu_requested_by_plan": parts[0] in {"6", "7"},
                "gpu_available_for_v103": parts[0] in {"6", "7"},
            }
        )
    return rows


def _torch_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "torch_import_ok": False,
        "torch_cuda_available": False,
        "torch_visible_device_count": 0,
    }
    try:
        import torch

        summary.update(
            {
                "torch_import_ok": True,
                "torch_version": getattr(torch, "__version__", ""),
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }
        )
        if torch.cuda.is_available():
            for local_idx in range(torch.cuda.device_count()):
                rows.append(
                    {
                        "schema_version": "stream4d_v103_phase0_gpu_env_row_v1",
                        "phase_id": "v103_phase0_contract",
                        "probe": "torch_cuda",
                        "local_device_index": local_idx,
                        "device_name": torch.cuda.get_device_name(local_idx),
                        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    }
                )
    except Exception as exc:
        summary["torch_import_error"] = repr(exc)
    return rows, summary


def _artifact_row(role: str, path: Path, required: bool, note: str) -> dict[str, Any]:
    exists = path.exists()
    return {
        "schema_version": "stream4d_v103_phase0_input_artifact_row_v1",
        "phase_id": "v103_phase0_contract",
        "role": role,
        "path": _rel(path),
        "required": required,
        "exists": exists,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size_bytes": path.stat().st_size if exists and path.is_file() else "",
        "csv_row_count": _row_count(path),
        "sha256": _sha256(path) if exists and path.is_file() else "",
        "note": note,
    }


def _contract_artifacts() -> list[dict[str, Any]]:
    return [
        _artifact_row("v103_method_doc", METHOD_DOC, True, "method-thinking doc requested by user"),
        _artifact_row("v103_plan_doc", PLAN_DOC, True, "experiment plan requested by user"),
        _artifact_row("canonical_v65_evaluator", V65_EVALUATOR, True, "SparseSceneIoU/_summarize_iou AP source"),
        _artifact_row("v100_phase2c_summary", PHASE2C_DIR / "summary.json", True, "current strong F2 local baseline summary"),
        _artifact_row("v100_phase2c_metric_rows", PHASE2C_DIR / "variant_metric_rows.csv", True, "current strong F2 local/window and fragmented scene metrics"),
        _artifact_row("v100_phase2c_gate_rows", PHASE2C_DIR / "variant_gate_rows.csv", True, "current strong F2 safety gates"),
        _artifact_row("v100_phase2c_object_rows", PHASE2C_DIR / "mv_object_rows.parquet", True, "F2 object rows for baseline comparison"),
        _artifact_row("v100_phase2c_frame_mask_rows", PHASE2C_DIR / "mv_object_frame_mask_rows.parquet", True, "F2 object-frame-mask rows for adapters"),
        _artifact_row("v100_final_summary", V100_FINAL_DIR / "summary.json", True, "v100 local-only / scene-No-Go boundary"),
        _artifact_row("v101_final_summary", V101_FINAL_DIR / "summary.json", True, "v101 provider bridge No-Go boundary"),
        _artifact_row("v101_provider_capability_summary", V101_PHASE2_DIR / "summary.json", True, "D4RT/DA3 provider bridge capability context"),
        _artifact_row("v102_final_summary", V102_FINAL_DIR / "summary.json", True, "v102 bridge pass / Phase6 blocked boundary"),
        _artifact_row("v102_phase5c_summary", V102_PHASE5C_DIR / "summary.json", True, "semantic barrier bridge gate context"),
        _artifact_row("v102_phase7j_summary", V102_PHASE7J_DIR / "summary.json", True, "same-chunk primitive support fill diagnostic boundary"),
        _artifact_row("dev_source_rows", SOURCE_ROWS_DEV, True, "dev source/mask registry rows"),
        _artifact_row("holdout_source_rows", SOURCE_ROWS_HOLDOUT, True, "holdout source/mask registry rows"),
        _artifact_row("dev_cropformer_mask_cache", CROPFORMER_DEV, True, "dev CropFormer mask registry root"),
        _artifact_row("holdout_cropformer_mask_cache", CROPFORMER_HOLDOUT, True, "holdout CropFormer mask registry root"),
        _artifact_row("radio_dev_feature_store", RADIO_DEV, True, "dev RADIO mask feature store"),
        _artifact_row("radio_dev_scene0011_feature_store", RADIO_DEV_SCENE0011, True, "dev scene0011 RADIO feature store"),
        _artifact_row("radio_dev_scene0050_feature_store", RADIO_DEV_SCENE0050, True, "dev scene0050 RADIO feature store"),
        _artifact_row("radio_holdout_feature_store", RADIO_HOLDOUT, True, "holdout RADIO mask feature store"),
        _artifact_row("d4rt_root", D4RT_ROOT, True, "local Open-d4rt checkout"),
        _artifact_row("d4rt_checkpoint_32clip", D4RT_CKPT_32, True, "OpenD4RT 32CLIP checkpoint directory"),
        _artifact_row("d4rt_query_tool", D4RT_QUERY_TOOL, True, "existing D4RT query tool entrypoint"),
        _artifact_row("d4rt_provider_audit_summary", D4RT_PROVIDER_AUDIT, True, "prior D4RT chunk32/o3 provider audit"),
    ]


def _baseline_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase2c_metrics = _read_csv(PHASE2C_DIR / "variant_metric_rows.csv")
    for row in phase2c_metrics:
        rows.append(
            {
                "schema_version": "stream4d_v103_phase0_baseline_metric_row_v1",
                "phase_id": "v103_phase0_contract",
                "source_phase_id": row.get("phase_id", ""),
                "source_artifact": _rel(PHASE2C_DIR / "variant_metric_rows.csv"),
                "baseline_role": "current_strong_local_baseline",
                "variant_id": row.get("variant_id", ""),
                "dataset_split": row.get("dataset_split", ""),
                "MV_AP_window": row.get("MV_AP_window", ""),
                "MV_AP50_window": row.get("MV_AP50_window", ""),
                "MV_AP25_window": row.get("MV_AP25_window", ""),
                "ScoreFreeMatch50_window": row.get("ScoreFreeMatch50_window", ""),
                "fragmented_MV_AP_scene": row.get("MV_AP_scene", ""),
                "fragmented_MV_AP50_scene": row.get("MV_AP50_scene", ""),
                "same_frame_collision_count": row.get("same_frame_collision_count", ""),
                "pixel_collision_rate": row.get("pixel_collision_rate", ""),
                "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
                "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
                "uses_future": row.get("uses_future", ""),
                "method_chunk_size": row.get("method_chunk_size", ""),
                "frame_stride": row.get("frame_stride", ""),
                "chunk_overlap": 3,
                "metric_source": row.get("metric_source", ""),
            }
        )

    v100_final = _read_json(V100_FINAL_DIR / "summary.json")
    v101_final = _read_json(V101_FINAL_DIR / "summary.json")
    v101_phase2 = _read_json(V101_PHASE2_DIR / "summary.json")
    v102_final = _read_json(V102_FINAL_DIR / "summary.json")
    v102_phase5c = _read_json(V102_PHASE5C_DIR / "summary.json")
    v102_phase7j = _read_json(V102_PHASE7J_DIR / "summary.json")
    rows.extend(
        [
            {
                "schema_version": "stream4d_v103_phase0_prior_boundary_row_v1",
                "phase_id": "v103_phase0_contract",
                "baseline_role": "prior_boundary",
                "source": "v100_phase8e_final_decision",
                "decision": v100_final.get("decision", ""),
                "local_claim_allowed": v100_final.get("local_claim_allowed", ""),
                "scene_claim_allowed": v100_final.get("scene_claim_allowed", ""),
                "note": "v100 positive claim is Phase2c local F2 only; scene stitching remains No-Go.",
            },
            {
                "schema_version": "stream4d_v103_phase0_prior_boundary_row_v1",
                "phase_id": "v103_phase0_contract",
                "baseline_role": "prior_boundary",
                "source": "v101_provider_capability/final_closeout",
                "decision": v101_final.get("final_decision", ""),
                "provider_decision": v101_phase2.get("decision", ""),
                "method_success_claim_allowed": v101_final.get("method_success_claim_allowed", ""),
                "note": "v101 proved raw fragmentation but provider bridge was insufficient for direct fragment repair.",
            },
            {
                "schema_version": "stream4d_v103_phase0_prior_boundary_row_v1",
                "phase_id": "v103_phase0_contract",
                "baseline_role": "prior_boundary",
                "source": "v102_final_closeout/phase5c/phase7j",
                "decision": v102_final.get("decision", ""),
                "phase5_bridge_pass": v102_phase5c.get("any_phase5_formal_bridge_gate_pass", ""),
                "phase6_ap_repair_allowed": v102_final.get("phase6_ap_repair_allowed", ""),
                "phase7j_decision": v102_phase7j.get("decision", ""),
                "phase7j_scene_metric_computed": v102_phase7j.get("scene_metric_computed", ""),
                "note": "v102 got a semantic-barrier bridge pass, but Phase6 remained blocked by repair-space; Phase7j was same-chunk local diagnostic only.",
            },
        ]
    )
    return rows


def _metric_contract() -> dict[str, Any]:
    thresholds, import_error, has_sparse, has_summarize = _load_v65_thresholds()
    phase2c_metrics = _read_csv(PHASE2C_DIR / "variant_metric_rows.csv")
    metric_sources = [str(row.get("metric_source", "")) for row in phase2c_metrics]
    formal_metric_source_eq_v65 = (
        V65_EVALUATOR.exists()
        and thresholds == EXPECTED_THRESHOLDS
        and has_sparse
        and has_summarize
        and all("v65" in src.lower() for src in metric_sources)
    )
    return {
        "schema_version": "stream4d_v103_phase0_metric_contract_v1",
        "phase_id": "v103_phase0_contract",
        "canonical_evaluator_path": _rel(V65_EVALUATOR),
        "canonical_evaluator_sha256": _sha256(V65_EVALUATOR),
        "v65_import_error": import_error,
        "has_sparse_scene_iou": has_sparse,
        "has_summarize_iou": has_summarize,
        "formal_metric_source_eq_v65": bool(formal_metric_source_eq_v65),
        "AP_thresholds_actual": thresholds,
        "AP_thresholds_expected": EXPECTED_THRESHOLDS,
        "ap_thresholds_match": thresholds == EXPECTED_THRESHOLDS,
        "local_support_policy": "local_window_gt_projection",
        "scene_support_policy": "scene_level_object_id_vs_raw_scene_gt_id",
        "method_chunk_size": 32,
        "frame_stride": 5,
        "chunk_overlap": 3,
        "future_chunk_access_allowed": False,
        "history_update_after_assignment_only": True,
        "metric_sources_locked": metric_sources,
    }


def _causality_rows() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "stream4d_v103_phase0_causality_contract_row_v1",
            "phase_id": "v103_phase0_contract",
            "contract_id": "streaming_protocol",
            "frame_stride": 5,
            "method_chunk_size": 32,
            "chunk_overlap": 3,
            "causality_scope": "chunk_causal",
            "future_chunk_access_allowed": False,
            "history_update_after_assignment_only": True,
            "pass": True,
        },
        {
            "schema_version": "stream4d_v103_phase0_causality_contract_row_v1",
            "phase_id": "v103_phase0_contract",
            "contract_id": "prediction_gt_boundary",
            "uses_gt_for_prediction_allowed": False,
            "uses_gt_for_eval_allowed": True,
            "uses_gt_for_diagnostic_allowed": True,
            "threshold_selection_from_gt_allowed": False,
            "pass": True,
        },
        {
            "schema_version": "stream4d_v103_phase0_causality_contract_row_v1",
            "phase_id": "v103_phase0_contract",
            "contract_id": "history_memory_order",
            "pre_update_history_memory": "H^{r-1}",
            "post_update_history_memory": "H^r",
            "carrier_to_history_support_uses": "H^{r-1}_only",
            "history_update_timing": "after_current_chunk_assignment",
            "pass": True,
        },
    ]


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_contract = _metric_contract()
    baseline_rows = _baseline_rows()
    artifact_rows = _contract_artifacts()
    causality_rows = _causality_rows()
    gpu_rows = _nvidia_rows()
    torch_gpu_rows, torch_summary = _torch_rows()
    gpu_rows.extend(torch_gpu_rows)
    gpu_rows.append(
        {
            "schema_version": "stream4d_v103_phase0_gpu_env_row_v1",
            "phase_id": "v103_phase0_contract",
            "probe": "python_packages",
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "torch": torch_summary.get("torch_version", _pkg_version("torch")),
            "cupy": _pkg_version("cupy") or _pkg_version("cupy-cuda12x") or _pkg_version("cupy-cuda11x"),
            "triton": _pkg_version("triton"),
            "numpy": _pkg_version("numpy"),
            "pandas": _pkg_version("pandas"),
            "pyarrow": _pkg_version("pyarrow"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
    )

    baseline_by_split = {
        str(row.get("dataset_split")): row
        for row in baseline_rows
        if row.get("baseline_role") == "current_strong_local_baseline"
    }
    dev = baseline_by_split.get("dev", {})
    holdout = baseline_by_split.get("holdout", {})

    required_missing = [row for row in artifact_rows if row.get("required") and not row.get("exists")]
    same_frame_collision_total = sum(
        int(_num(row.get("same_frame_collision_count")))
        for row in baseline_rows
        if row.get("baseline_role") == "current_strong_local_baseline"
    )
    max_pixel_collision = max(
        (
            _num(row.get("pixel_collision_rate"), 999.0)
            for row in baseline_rows
            if row.get("baseline_role") == "current_strong_local_baseline"
        ),
        default=999.0,
    )
    missing_mask_total = sum(
        int(_num(row.get("missing_mask_raster_count")))
        for row in baseline_rows
        if row.get("baseline_role") == "current_strong_local_baseline"
    )
    uses_gt_any = any(
        _bool(row.get("uses_gt_for_prediction"))
        for row in baseline_rows
        if row.get("baseline_role") == "current_strong_local_baseline"
    )
    uses_future_any = any(
        _bool(row.get("uses_future"))
        for row in baseline_rows
        if row.get("baseline_role") == "current_strong_local_baseline"
    )
    gpus_6_7_available = {str(row.get("gpu_index")) for row in gpu_rows if row.get("gpu_available_for_v103")}
    gpu67_pass = {"6", "7"}.issubset(gpus_6_7_available)

    gate_rows = [
        {
            "gate_id": "formal_metric_source_eq_v65",
            "pass": bool(metric_contract["formal_metric_source_eq_v65"]),
            "expected": "v65 SparseSceneIoU/_summarize_iou, thresholds 0.50..0.90, metric_source contains v65",
            "observed": f"thresholds={metric_contract['AP_thresholds_actual']} metric_sources={metric_contract['metric_sources_locked']}",
            "severity": "required",
        },
        {
            "gate_id": "current_strong_f2_baseline_readable",
            "pass": bool(dev and holdout),
            "expected": "dev and holdout F2 Phase2c rows readable",
            "observed": f"dev={dev.get('MV_AP_window', '')} holdout={holdout.get('MV_AP_window', '')}",
            "severity": "required",
        },
        {
            "gate_id": "baseline_safety_contract",
            "pass": same_frame_collision_total == 0 and max_pixel_collision <= 0.02 and missing_mask_total == 0,
            "expected": "same_frame_collision=0, pixel_collision<=0.02, missing_mask=0",
            "observed": f"same_frame_collision={same_frame_collision_total} max_pixel_collision={max_pixel_collision} missing_mask={missing_mask_total}",
            "severity": "required",
        },
        {
            "gate_id": "baseline_no_gt_future_prediction",
            "pass": not uses_gt_any and not uses_future_any,
            "expected": "uses_gt_for_prediction=false and uses_future=false",
            "observed": f"uses_gt_any={uses_gt_any} uses_future_any={uses_future_any}",
            "severity": "required",
        },
        {
            "gate_id": "required_input_artifacts_exist",
            "pass": len(required_missing) == 0,
            "expected": "all required artifact paths exist",
            "observed": "|".join(str(row.get("role")) for row in required_missing),
            "severity": "required",
        },
        {
            "gate_id": "gpu_6_7_available",
            "pass": gpu67_pass,
            "expected": "physical GPUs 6 and 7 visible in nvidia-smi",
            "observed": ",".join(sorted(gpus_6_7_available)),
            "severity": "required",
        },
        {
            "gate_id": "torch_cuda_available",
            "pass": bool(torch_summary.get("torch_cuda_available")),
            "expected": "torch can import and cuda is available",
            "observed": json.dumps(_jsonable(torch_summary), sort_keys=True),
            "severity": "required",
        },
    ]

    failure_rows = [
        {
            "schema_version": "stream4d_v103_phase0_failure_row_v1",
            "phase_id": "v103_phase0_contract",
            "gate_id": row["gate_id"],
            "severity": row["severity"],
            "observed": row["observed"],
            "repair_direction": {
                "formal_metric_source_eq_v65": "repair evaluator adapter/path; do not run algorithm experiments",
                "current_strong_f2_baseline_readable": "restore/read v100 Phase2c metrics/artifacts before continuing",
                "baseline_safety_contract": "inspect baseline materialization and mask raster paths before reuse",
                "baseline_no_gt_future_prediction": "remove any prediction GT/future dependency before continuing",
                "required_input_artifacts_exist": "repair artifact path/adapter boundary before D4RT query",
                "gpu_6_7_available": "repair GPU visibility or explicitly record fallback before Phase1",
                "torch_cuda_available": "use loger env or repair torch CUDA import before GPU parity",
            }.get(str(row["gate_id"]), "repair Phase0 contract before continuing"),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]

    phase0_pass = not failure_rows
    summary = {
        "schema_version": "stream4d_v103_phase0_contract_summary_v1",
        "phase_id": "v103_phase0_contract",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "phase0_pass": phase0_pass,
        "plan_doc": _rel(PLAN_DOC),
        "method_doc": _rel(METHOD_DOC),
        "formal_metric_source_eq_v65": metric_contract["formal_metric_source_eq_v65"],
        "AP_thresholds_actual": metric_contract["AP_thresholds_actual"],
        "local_support_policy": metric_contract["local_support_policy"],
        "scene_support_policy": metric_contract["scene_support_policy"],
        "method_chunk_size": 32,
        "frame_stride": 5,
        "chunk_overlap": 3,
        "future_chunk_access_allowed": False,
        "history_update_after_assignment_only": True,
        "current_strong_local_baseline_variant": dev.get("variant_id", holdout.get("variant_id", "")),
        "dev_MV_AP_window": dev.get("MV_AP_window", ""),
        "dev_MV_AP50_window": dev.get("MV_AP50_window", ""),
        "holdout_MV_AP_window": holdout.get("MV_AP_window", ""),
        "holdout_MV_AP50_window": holdout.get("MV_AP50_window", ""),
        "holdout_fragmented_MV_AP_scene": holdout.get("fragmented_MV_AP_scene", ""),
        "holdout_fragmented_MV_AP50_scene": holdout.get("fragmented_MV_AP50_scene", ""),
        "d4rt_root_available": D4RT_ROOT.exists(),
        "d4rt_checkpoint_32clip_available": D4RT_CKPT_32.exists(),
        "cropformer_mask_registry_available": CROPFORMER_DEV.exists() and CROPFORMER_HOLDOUT.exists(),
        "radio_mask_feature_available": RADIO_DEV.exists() and RADIO_HOLDOUT.exists(),
        "gpu_6_7_available": gpu67_pass,
        "torch_cuda_available": bool(torch_summary.get("torch_cuda_available")),
        "failure_count": len(failure_rows),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "metric_contract": _rel(OUT_DIR / "metric_contract.json"),
            "baseline_metric_rows": _rel(OUT_DIR / "baseline_metric_rows.csv"),
            "input_artifact_rows": _rel(OUT_DIR / "input_artifact_rows.csv"),
            "causality_contract_rows": _rel(OUT_DIR / "causality_contract_rows.csv"),
            "gpu_env_rows": _rel(OUT_DIR / "gpu_env_rows.csv"),
            "gate_rows": _rel(OUT_DIR / "gate_rows.csv"),
            "failure_rows": _rel(OUT_DIR / "failure_rows.csv"),
        },
        "truthfulness_note": "Phase0 is read-only; it locks inputs and contracts only, and does not create method predictions.",
    }

    _write_json(OUT_DIR / "metric_contract.json", metric_contract)
    _write_csv(OUT_DIR / "baseline_metric_rows.csv", baseline_rows)
    _write_csv(OUT_DIR / "input_artifact_rows.csv", artifact_rows)
    _write_csv(OUT_DIR / "causality_contract_rows.csv", causality_rows)
    _write_csv(OUT_DIR / "gpu_env_rows.csv", gpu_rows)
    _write_csv(OUT_DIR / "gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "failure_rows.csv", failure_rows)
    _write_json(OUT_DIR / "summary.json", summary)

    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase0_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
