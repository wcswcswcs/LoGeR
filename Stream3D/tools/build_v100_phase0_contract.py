#!/usr/bin/env python3
"""Build Stream4D v100 Phase0 metric/baseline/artifact/GPU contract lock.

This phase is intentionally read-only with respect to method outputs. It locks
the v65 metric contract, the v99 F2/repaired-local facts, the known scene
fragmentation boundary, and the GPU library state before v100 method work.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase0_contract"

EXPECTED_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(9)]

V99_PHASE0 = AUDIT_ROOT / "v99_phase0_fact_lock/summary.json"
V99_PHASE10L = AUDIT_ROOT / "v99_phase10l_frozen_p2d2_regenerated_birth_holdout/summary.json"
V99_PHASE10M = AUDIT_ROOT / "v99_phase10m_repaired_projection_final_decision/summary.json"
V99_PHASE10N = AUDIT_ROOT / "v99_phase10n_scene_fragmentation_audit/summary.json"
V99_PHASE10P = AUDIT_ROOT / "v99_phase10p_overlap3_scene_stitch_semantic_sweep/summary.json"
V99_PHASE10AI = AUDIT_ROOT / "v99_phase10ai_prefix_sim3_d4rt_semantic_scene_repair_cupy/summary.json"
V99_CUPY_PARITY = AUDIT_ROOT / "v99_cupy_sparse_iou_parity/summary.json"


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_row_count(path: Path) -> int | str:
    if not path.exists() or not path.is_file():
        return ""
    if path.suffix.lower() != ".csv":
        return ""
    with path.open(newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def _module_from_path(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric_contract() -> dict[str, Any]:
    v65 = STREAM3D_ROOT / "tools/run_v65_scene_multiview_ap.py"
    check = STREAM3D_ROOT / "tools/check_mv_ap_contract.py"
    local_adapter = STREAM3D_ROOT / "tools/run_v89_recalc_point_projected_mv_ap.py"
    scene_adapter = STREAM3D_ROOT / "tools/build_v98_1_canonical_scene_metrics.py"
    proc = subprocess.run(
        [sys.executable, _rel(check)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    import_error = ""
    thresholds: list[float] = []
    has_sparse = has_summarize = False
    try:
        module = _module_from_path(v65)
        thresholds = [round(float(x), 2) for x in getattr(module, "AP_THRESHOLDS", [])]
        has_sparse = hasattr(module, "SparseSceneIoU")
        has_summarize = hasattr(module, "_summarize_iou")
    except Exception as exc:
        import_error = repr(exc)
    local_text = local_adapter.read_text(encoding="utf-8") if local_adapter.exists() else ""
    scene_text = scene_adapter.read_text(encoding="utf-8") if scene_adapter.exists() else ""
    return {
        "schema_version": "stream4d_v100_metric_contract_v1",
        "formal_metric_source_eq_v65": bool(proc.returncode == 0 and has_sparse and has_summarize),
        "check_returncode": proc.returncode,
        "check_stdout": proc.stdout.strip(),
        "check_stderr": proc.stderr.strip(),
        "v65_file": _rel(v65),
        "v65_import_error": import_error,
        "has_sparse_scene_iou": has_sparse,
        "has_summarize_iou": has_summarize,
        "ap_thresholds": thresholds,
        "expected_ap_thresholds": EXPECTED_THRESHOLDS,
        "ap_thresholds_match": thresholds == EXPECTED_THRESHOLDS,
        "local_support_policy": "local_window_gt_projection",
        "local_window_adapter": _rel(local_adapter),
        "local_window_adapter_mentions_summarize_iou": "_summarize_iou" in local_text,
        "scene_adapter": _rel(scene_adapter),
        "scene_adapter_mentions_summarize_iou": "_summarize_iou" in scene_text,
    }


def _baseline_rows(v99p0: dict[str, Any], v99l: dict[str, Any], v99m: dict[str, Any], v99n: dict[str, Any], v99p: dict[str, Any], v99ai: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "row_id": "F2_base_full_dev",
            "reference_kind": "locked_baseline",
            "split": "full_dev",
            "variant_id": v99p0.get("F2_base_variant_id", ""),
            "MV_AP_window": v99p0.get("F2_base_full_dev_MV_AP_window", ""),
            "MV_AP50_window": v99p0.get("F2_base_full_dev_MV_AP50_window", ""),
            "MV_AP_scene": v99p0.get("F2_base_full_dev_MV_AP_scene", ""),
            "MV_AP50_scene": v99p0.get("F2_base_full_dev_MV_AP50_scene", ""),
            "formal_claim_allowed": True,
            "source_summary": _rel(V99_PHASE0),
        },
        {
            "row_id": "F2_base_holdout",
            "reference_kind": "locked_baseline",
            "split": "same_scene_temporal_holdout",
            "variant_id": v99p0.get("F2_base_variant_id", ""),
            "MV_AP_window": v99p0.get("F2_base_holdout_MV_AP_window", ""),
            "MV_AP50_window": v99p0.get("F2_base_holdout_MV_AP50_window", ""),
            "MV_AP_scene": v99p0.get("F2_base_holdout_MV_AP_scene", ""),
            "MV_AP50_scene": v99p0.get("F2_base_holdout_MV_AP50_scene", ""),
            "formal_claim_allowed": True,
            "source_summary": _rel(V99_PHASE0),
        },
        {
            "row_id": "v99_repaired_local_dev",
            "reference_kind": "repaired_local_candidate",
            "split": "full_dev",
            "variant_id": v99l.get("fixed_dev_variant_id", ""),
            "birth_variant_id": v99l.get("fixed_birth_variant_id", ""),
            "MV_AP_window": v99l.get("dev_MV_AP_window", ""),
            "MV_AP50_window": v99l.get("dev_MV_AP50_window", ""),
            "MV_AP_scene": v99l.get("dev_MV_AP_scene", ""),
            "MV_AP50_scene": v99l.get("dev_MV_AP50_scene", ""),
            "metric_gate_pass": v99l.get("dev_gate_pass", ""),
            "formal_claim_allowed": v99l.get("formal_claim_allowed", False),
            "formal_blocker": v99l.get("formal_blocker", ""),
            "source_summary": _rel(V99_PHASE10L),
        },
        {
            "row_id": "v99_repaired_local_holdout",
            "reference_kind": "repaired_local_candidate",
            "split": "same_scene_temporal_holdout",
            "variant_id": v99l.get("holdout_variant_id", ""),
            "birth_variant_id": v99l.get("fixed_birth_variant_id", ""),
            "MV_AP_window": v99l.get("holdout_MV_AP_window", ""),
            "MV_AP50_window": v99l.get("holdout_MV_AP50_window", ""),
            "MV_AP_scene": v99l.get("holdout_MV_AP_scene", ""),
            "MV_AP50_scene": v99l.get("holdout_MV_AP50_scene", ""),
            "metric_gate_pass": v99l.get("holdout_gate_pass", ""),
            "formal_claim_allowed": v99l.get("formal_claim_allowed", False),
            "formal_blocker": v99l.get("formal_blocker", ""),
            "source_summary": _rel(V99_PHASE10L),
        },
        {
            "row_id": "v99_repaired_final_decision",
            "reference_kind": "decision_boundary",
            "split": "dev_and_holdout",
            "decision": v99m.get("decision", ""),
            "local_metric_gate_pass": v99m.get("local_metric_gate_pass", ""),
            "scene_gate_pass": v99m.get("scene_gate_pass", ""),
            "formal_claim_allowed": v99m.get("formal_claim_allowed", ""),
            "source_summary": _rel(V99_PHASE10M),
        },
        {
            "row_id": "v99_scene_fragmentation_audit",
            "reference_kind": "scene_failure_diagnostic",
            "split": "same_scene_temporal_holdout",
            "decision": v99n.get("decision", ""),
            "MV_AP_window": v99n.get("phase10l_holdout_MV_AP_window", ""),
            "MV_AP50_window": v99n.get("phase10l_holdout_MV_AP50_window", ""),
            "MV_AP_scene": v99n.get("phase10l_holdout_MV_AP_scene", ""),
            "MV_AP50_scene": v99n.get("phase10l_holdout_MV_AP50_scene", ""),
            "source_summary": _rel(V99_PHASE10N),
        },
        {
            "row_id": "v99_best_non_gt_scene_repair",
            "reference_kind": "scene_repair_diagnostic",
            "split": "same_scene_temporal_holdout",
            "variant_id": v99ai.get("best_variant_id") or v99p.get("best_variant_id", ""),
            "MV_AP_window": v99ai.get("best_MV_AP_window") or v99p.get("best_MV_AP_window", ""),
            "MV_AP50_window": v99ai.get("best_MV_AP50_window") or v99p.get("best_MV_AP50_window", ""),
            "MV_AP_scene": v99ai.get("best_MV_AP_scene") or v99p.get("best_MV_AP_scene", ""),
            "MV_AP50_scene": v99ai.get("best_MV_AP50_scene") or v99p.get("best_MV_AP50_scene", ""),
            "metric_gate_pass": v99ai.get("metric_gate_pass", v99p.get("metric_gate_pass", "")),
            "source_summary": _rel(V99_PHASE10AI if V99_PHASE10AI.exists() else V99_PHASE10P),
        },
    ]


def _artifact_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.append(
            {
                "artifact_path": _rel(path),
                "exists": path.exists(),
                "artifact_type": "directory" if path.is_dir() else path.suffix.lstrip(".") if path.exists() else "",
                "row_count_or_shape": _csv_row_count(path),
                "dtype": "",
                "key_columns": "",
                "sha256_or_fast_hash": _sha256(path),
                "method_input_allowed": False,
                "diagnostic_only": True,
                "notes": "Phase0 fact-lock input",
            }
        )
    return rows


def _version(module_name: str, package_name: str | None = None) -> tuple[bool, str, str]:
    package_name = package_name or module_name
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False, "", "not_found"
        try:
            version = importlib.metadata.version(package_name)
        except Exception:
            version = "unknown"
        return True, version, ""
    except Exception as exc:
        return False, "", repr(exc)


def _gpu_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    wanted = {"6", "7"}
    seen: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, used, total, util = parts[:5]
        if idx not in wanted:
            continue
        seen.add(idx)
        rows.append(
            {
                "row_id": f"gpu_{idx}",
                "resource": "nvidia_gpu",
                "device_index": idx,
                "name": name,
                "memory_used_mib": used,
                "memory_total_mib": total,
                "utilization_gpu_percent": util,
                "available": True,
                "stdout": "",
                "stderr": proc.stderr.strip(),
            }
        )
    for idx in sorted(wanted - seen):
        rows.append(
            {
                "row_id": f"gpu_{idx}",
                "resource": "nvidia_gpu",
                "device_index": idx,
                "available": False,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )
    for module_name, package_name in [
        ("torch", "torch"),
        ("cupy", "cupy-cuda12x"),
        ("triton", "triton"),
        ("numpy", "numpy"),
    ]:
        ok, version, error = _version(module_name, package_name)
        rows.append(
            {
                "row_id": f"python_module_{module_name}",
                "resource": "python_module",
                "module": module_name,
                "package": package_name,
                "available": ok,
                "version": version,
                "error": error,
                "python": sys.executable,
            }
        )
    return rows


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v99p0 = _read_json(V99_PHASE0)
    v99l = _read_json(V99_PHASE10L)
    v99m = _read_json(V99_PHASE10M)
    v99n = _read_json(V99_PHASE10N)
    v99p = _read_json(V99_PHASE10P)
    v99ai = _read_json(V99_PHASE10AI)
    cupy_parity = _read_json(V99_CUPY_PARITY)

    metric_contract = _metric_contract()
    baseline_rows = _baseline_rows(v99p0, v99l, v99m, v99n, v99p, v99ai)
    artifact_rows = _artifact_rows(
        [
            Path("docs/stream4d_v100_f2_gpu_history_memory_plan.md"),
            Path("docs/stream4d_v99_执行日志.md"),
            Path("docs/stream4d_v99_实验结果复盘.md"),
            V99_PHASE0,
            V99_PHASE10L,
            V99_PHASE10M,
            V99_PHASE10N,
            V99_PHASE10P,
            V99_PHASE10AI,
            V99_CUPY_PARITY,
            AUDIT_ROOT / "v99_phase10l_frozen_p2d2_regenerated_birth_holdout/mv_object_frame_mask_rows.csv",
            AUDIT_ROOT / "v99_phase10o_overlap3_scene_stitch_repair/mv_object_frame_mask_rows.csv",
            STREAM3D_ROOT / "tools/build_v99_phase1_f2_base_reproduction.py",
            STREAM3D_ROOT / "tools/build_v99_phase10o_overlap3_scene_stitch_repair.py",
            STREAM3D_ROOT / "tools/v99_cupy_sparse_iou.py",
        ]
    )
    gpu_rows = _gpu_rows()

    gpu_available = all(
        _bool(row.get("available")) for row in gpu_rows if row.get("resource") == "nvidia_gpu" and row.get("device_index") in {"6", "7"}
    )
    module_available = {str(row.get("module")): _bool(row.get("available")) for row in gpu_rows if row.get("resource") == "python_module"}
    f2_metrics_locked = all(
        v99p0.get(key) not in ("", None)
        for key in [
            "F2_base_full_dev_MV_AP_window",
            "F2_base_full_dev_MV_AP50_window",
            "F2_base_holdout_MV_AP_window",
            "F2_base_holdout_MV_AP50_window",
        ]
    )
    repaired_local_locked = bool(v99l.get("metric_gate_pass")) and v99l.get("holdout_MV_AP_window") not in ("", None)
    fragmentation_exists = V99_PHASE10N.exists() and v99n.get("decision") == "SCENE_LOW_EXPECTED_FROM_CHUNK_FRAGMENTATION"
    cupy_or_torch = bool(module_available.get("cupy") or module_available.get("torch"))
    triton_available = bool(module_available.get("triton"))

    gate_rows = [
        {
            "gate_id": "formal_metric_source_eq_v65",
            "pass": metric_contract["formal_metric_source_eq_v65"],
            "expected": "check_mv_ap_contract passes and v65 exposes SparseSceneIoU/_summarize_iou",
            "observed": metric_contract["check_stdout"] or metric_contract["v65_import_error"],
            "severity": "required",
        },
        {
            "gate_id": "ap_thresholds_match_contract",
            "pass": metric_contract["ap_thresholds_match"],
            "expected": EXPECTED_THRESHOLDS,
            "observed": metric_contract["ap_thresholds"],
            "severity": "required",
        },
        {
            "gate_id": "adapters_reference_v65_summarize_iou",
            "pass": bool(metric_contract["local_window_adapter_mentions_summarize_iou"] and metric_contract["scene_adapter_mentions_summarize_iou"]),
            "expected": "local and scene adapters mention _summarize_iou",
            "observed": {
                "local": metric_contract["local_window_adapter_mentions_summarize_iou"],
                "scene": metric_contract["scene_adapter_mentions_summarize_iou"],
            },
            "severity": "required",
        },
        {
            "gate_id": "F2_base_metrics_locked",
            "pass": f2_metrics_locked,
            "expected": "F2_base full-dev and holdout MV_AP_window/MV_AP50_window available",
            "observed": {
                "full_dev": v99p0.get("F2_base_full_dev_MV_AP_window"),
                "holdout": v99p0.get("F2_base_holdout_MV_AP_window"),
            },
            "severity": "required",
        },
        {
            "gate_id": "repaired_local_metrics_locked",
            "pass": repaired_local_locked,
            "expected": "v99 Phase10L metric_gate_pass=true and holdout metrics present",
            "observed": {
                "decision": v99l.get("decision"),
                "metric_gate_pass": v99l.get("metric_gate_pass"),
                "holdout_MV_AP_window": v99l.get("holdout_MV_AP_window"),
            },
            "severity": "required",
        },
        {
            "gate_id": "phase10n_fragmentation_audit_exists",
            "pass": fragmentation_exists,
            "expected": "scene low explained by chunk fragmentation audit",
            "observed": v99n.get("decision"),
            "severity": "required",
        },
        {
            "gate_id": "gpu_6_7_available",
            "pass": gpu_available,
            "expected": "GPU 6 and GPU 7 visible in nvidia-smi",
            "observed": [row for row in gpu_rows if row.get("resource") == "nvidia_gpu"],
            "severity": "required_gpu",
        },
        {
            "gate_id": "cupy_or_torch_available",
            "pass": cupy_or_torch,
            "expected": "CuPy or PyTorch importable in current interpreter",
            "observed": module_available,
            "severity": "required_gpu_fallback",
        },
        {
            "gate_id": "triton_available_or_fallback_recorded",
            "pass": triton_available or cupy_or_torch,
            "expected": "Triton available, or CuPy/PyTorch fallback recorded",
            "observed": module_available,
            "severity": "gpu_kernel_fallback",
        },
        {
            "gate_id": "previous_cupy_parity_passed",
            "pass": bool(cupy_parity.get("parity_pass")),
            "expected": "v99 CuPy parity smoke passed",
            "observed": cupy_parity.get("decision"),
            "severity": "phase1_recheck_required",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "Repair v65 contract/path before entering Phase1."
                if row["severity"] == "required"
                else "Run CPU/CuPy/PyTorch fallback parity subset before any full GPU method branch."
            ),
        }
        for row in gate_rows
        if not bool(row["pass"]) and row["severity"] in {"required", "required_gpu", "required_gpu_fallback"}
    ]
    phase0_pass = not failure_rows
    summary = {
        "schema_version": "stream4d_v100_phase0_contract_summary_v1",
        "phase_id": "v100_phase0_contract",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "PASS_ENTER_PHASE1" if phase0_pass else "BLOCK_PHASE1_REPAIR_CONTRACT",
        "phase0_pass": phase0_pass,
        "failure_count": len(failure_rows),
        "formal_metric_source_eq_v65": metric_contract["formal_metric_source_eq_v65"],
        "ap_thresholds": metric_contract["ap_thresholds"],
        "local_support_policy": metric_contract["local_support_policy"],
        "F2_base_full_dev_MV_AP_window": v99p0.get("F2_base_full_dev_MV_AP_window"),
        "F2_base_full_dev_MV_AP50_window": v99p0.get("F2_base_full_dev_MV_AP50_window"),
        "F2_base_holdout_MV_AP_window": v99p0.get("F2_base_holdout_MV_AP_window"),
        "F2_base_holdout_MV_AP50_window": v99p0.get("F2_base_holdout_MV_AP50_window"),
        "repaired_local_dev_MV_AP_window": v99l.get("dev_MV_AP_window"),
        "repaired_local_dev_MV_AP50_window": v99l.get("dev_MV_AP50_window"),
        "repaired_local_holdout_MV_AP_window": v99l.get("holdout_MV_AP_window"),
        "repaired_local_holdout_MV_AP50_window": v99l.get("holdout_MV_AP50_window"),
        "repaired_local_holdout_MV_AP_scene": v99l.get("holdout_MV_AP_scene"),
        "repaired_local_formal_claim_allowed": v99l.get("formal_claim_allowed"),
        "repaired_local_formal_blocker": v99l.get("formal_blocker"),
        "fragmentation_audit_decision": v99n.get("decision"),
        "best_non_gt_scene_repair_MV_AP_scene": v99ai.get("best_MV_AP_scene") or v99p.get("best_MV_AP_scene"),
        "gpu_6_7_available": gpu_available,
        "torch_available": module_available.get("torch", False),
        "cupy_available": module_available.get("cupy", False),
        "triton_available": triton_available,
        "previous_cupy_parity_pass": cupy_parity.get("parity_pass", False),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "metric_contract": _rel(OUT_DIR / "metric_contract.json"),
            "baseline_metric_rows": _rel(OUT_DIR / "baseline_metric_rows.csv"),
            "artifact_boundary_rows": _rel(OUT_DIR / "artifact_boundary_rows.csv"),
            "gpu_env_rows": _rel(OUT_DIR / "gpu_env_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
        },
    }

    _write_json(OUT_DIR / "metric_contract.json", metric_contract)
    _write_csv(OUT_DIR / "baseline_metric_rows.csv", baseline_rows)
    _write_csv(OUT_DIR / "artifact_boundary_rows.csv", artifact_rows)
    _write_csv(OUT_DIR / "gpu_env_rows.csv", gpu_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase0_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
