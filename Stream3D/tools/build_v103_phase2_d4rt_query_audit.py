from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v103_phase2_d4rt_query_audit"
PLAN_DOC = ROOT / "docs" / "stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"

RUNS = [
    {
        "scene_id": "scene0011_00",
        "variant_id": "Q0_lite_grid16_uniform_first32",
        "run_dir": AUDIT_ROOT / "v103_phase2_d4rt_q0_grid16_scene0011_first32",
        "cuda_visible_devices": "6",
    },
    {
        "scene_id": "scene0050_00",
        "variant_id": "Q0_lite_grid16_uniform_first32",
        "run_dir": AUDIT_ROOT / "v103_phase2_d4rt_q0_grid16_scene0050_first32",
        "cuda_visible_devices": "7",
    },
]
STRATIFIED_RUNS = [
    {
        "scene_id": "scene0011_00",
        "variant_id": "Q1_stratified_grid30_first32",
        "run_dir": AUDIT_ROOT / "v103_phase2_stratified_q1_grid30_scene0011_first32",
        "cuda_visible_devices": "6",
    },
    {
        "scene_id": "scene0050_00",
        "variant_id": "Q1_stratified_grid30_first32",
        "run_dir": AUDIT_ROOT / "v103_phase2_stratified_q1_grid30_scene0050_first32",
        "cuda_visible_devices": "7",
    },
]

REQUIRED_STRATA = [
    "uniform_grid",
    "mask_interior",
    "mask_boundary_band",
    "competing_mask_boundary",
    "semantic_gradient",
    "high_risk_broad_mask_interior",
    "overlap_frame_anchor",
]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["schema_version"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_rate(num: int | float, den: int | float) -> float:
    den_f = float(den)
    if den_f <= 0.0:
        return 0.0
    return float(num) / den_f


def _audit_run(spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(spec["run_dir"])
    summary_path = run_dir / "geometry_summary.json"
    chunk_path = run_dir / "stride_5" / "chunks" / "chunk_0000.npz"
    chunk_summary_path = run_dir / "stride_5" / "chunks" / "chunk_0000_summary.json"

    row: dict[str, Any] = {
        "schema_version": "stream4d_v103_phase2_query_variant_row_v1",
        "phase_id": "v103_phase2_d4rt_query_audit",
        "scene_id": spec["scene_id"],
        "variant_id": spec["variant_id"],
        "run_dir": _rel(run_dir),
        "summary_path": _rel(summary_path),
        "chunk_path": _rel(chunk_path),
        "chunk_summary_path": _rel(chunk_summary_path),
        "cuda_visible_devices": spec.get("cuda_visible_devices", ""),
        "decode_error_count": 0,
        "uses_gt_for_query_selection": False,
        "uses_future": False,
        "final_gt_sim3_diagnostic_only": "",
        "query_count_per_frame": "",
        "carrier_count_per_chunk": "",
        "frame_count": "",
        "projection_valid_rate_model_valid": "",
        "uv_in01_rate_all_observations": "",
        "visible_confident_observation_rate": "",
        "uv_in01_rate_visible_confident_observations": "",
        "uv_in01_rate_valid_observations": "",
        "in_image_rate": "",
        "visibility_mean_in_image": "",
        "confidence_mean_in_image": "",
        "runtime_sec_per_chunk": "",
        "gpu_memory_peak_MB": "",
        "gpu_memory_peak_recorded": False,
        "D4RT_decode_success_rate": "",
        "boundary_query_rate": 0.0,
        "conflict_query_rate": 0.0,
        "semantic_gradient_query_rate": 0.0,
        "strata_present": "uniform_grid",
        "missing_strata": "",
        "status": "missing_inputs",
        "failure_reason": "",
    }

    missing = [str(p) for p in (summary_path, chunk_path, chunk_summary_path) if not p.exists()]
    if missing:
        row["decode_error_count"] = len(missing)
        row["failure_reason"] = "missing_required_artifacts:" + ";".join(missing)
        return row

    geom = _load_json(summary_path)
    stride_rows = geom.get("stride_rows", [])
    stride_row = stride_rows[0] if stride_rows else {}
    chunk_summary = _load_json(chunk_summary_path)
    row["final_gt_sim3_diagnostic_only"] = bool(
        ((stride_row.get("final_gt_sim3") or {}).get("diagnostic_only", False))
    )
    row["query_count_per_frame"] = int(stride_row.get("grid_points_per_frame") or chunk_summary.get("grid_points_per_frame") or 0)
    row["runtime_sec_per_chunk"] = float(chunk_summary.get("seconds", stride_row.get("seconds", 0.0)) or 0.0)
    row["D4RT_decode_success_rate"] = 1.0 if geom.get("all_requested_strides_complete") else 0.0

    data = np.load(chunk_path, allow_pickle=False)
    valid = np.asarray(data["valid"], dtype=bool)
    uv = np.asarray(data["uv"], dtype=np.float32)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    visibility = np.asarray(data["visibility"], dtype=np.float32)
    confidence = np.asarray(data["confidence"], dtype=np.float32)
    finite = np.isfinite(uv).all(axis=-1) & np.isfinite(xyz).all(axis=-1)
    uv_in01 = finite & (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    valid_uv = valid & uv_in01
    total_obs = int(valid.size)
    valid_count = int(np.count_nonzero(valid))
    valid_uv_count = int(np.count_nonzero(valid_uv))
    row["carrier_count_per_chunk"] = int(np.asarray(data["carrier_id"]).shape[0])
    row["frame_count"] = int(np.asarray(data["frame_ids"]).shape[0])
    row["projection_valid_rate_model_valid"] = _safe_rate(valid_count, total_obs)
    row["uv_in01_rate_all_observations"] = _safe_rate(valid_uv_count, total_obs)
    row["uv_in01_rate_valid_observations"] = _safe_rate(valid_uv_count, valid_count)
    row["in_image_rate"] = row["uv_in01_rate_valid_observations"]
    row["visibility_mean_in_image"] = float(np.mean(visibility[valid_uv])) if valid_uv_count else 0.0
    row["confidence_mean_in_image"] = float(np.mean(confidence[valid_uv])) if valid_uv_count else 0.0

    present = {"uniform_grid"}
    missing_strata = [s for s in REQUIRED_STRATA if s not in present]
    row["missing_strata"] = ";".join(missing_strata)
    row["status"] = "audited"
    return row


def _audit_stratified_run(spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(spec["run_dir"])
    summary_path = run_dir / "summary.json"
    batch_path = run_dir / "carrier_batch.npz"
    source_count_rows_path = run_dir / "query_source_count_rows.csv"
    row: dict[str, Any] = {
        "schema_version": "stream4d_v103_phase2_query_variant_row_v1",
        "phase_id": "v103_phase2_d4rt_query_audit",
        "scene_id": spec["scene_id"],
        "variant_id": spec["variant_id"],
        "run_dir": _rel(run_dir),
        "summary_path": _rel(summary_path),
        "chunk_path": _rel(batch_path),
        "chunk_summary_path": _rel(summary_path),
        "cuda_visible_devices": spec.get("cuda_visible_devices", ""),
        "decode_error_count": "",
        "uses_gt_for_query_selection": False,
        "uses_future": False,
        "final_gt_sim3_diagnostic_only": False,
        "query_count_per_frame": "",
        "carrier_count_per_chunk": "",
        "frame_count": "",
        "projection_valid_rate_model_valid": "",
        "uv_in01_rate_all_observations": "",
        "visible_confident_observation_rate": "",
        "uv_in01_rate_visible_confident_observations": "",
        "uv_in01_rate_valid_observations": "",
        "in_image_rate": "",
        "visibility_mean_in_image": "",
        "confidence_mean_in_image": "",
        "runtime_sec_per_chunk": "",
        "gpu_memory_peak_MB": "",
        "gpu_memory_peak_recorded": False,
        "D4RT_decode_success_rate": "",
        "boundary_query_rate": "",
        "conflict_query_rate": "",
        "semantic_gradient_query_rate": "",
        "strata_present": "",
        "missing_strata": "",
        "status": "missing_inputs",
        "failure_reason": "",
    }
    missing = [str(p) for p in (summary_path, batch_path, source_count_rows_path) if not p.exists()]
    if missing:
        row["decode_error_count"] = len(missing)
        row["failure_reason"] = "missing_required_artifacts:" + ";".join(missing)
        return row
    summary = _load_json(summary_path)
    metric = summary.get("metrics") or {}
    source_counts = summary.get("source_counts") or []
    source_by_name = {str(item.get("query_source")): int(item.get("source_count") or 0) for item in source_counts}
    carrier_count = int(summary.get("source_count") or 0)
    present = [str(v) for v in summary.get("present_strata") or []]
    missing_strata = [s for s in REQUIRED_STRATA if s not in set(present)]
    row.update(
        {
            "decode_error_count": int(summary.get("decode_error_count") or 0),
            "uses_gt_for_query_selection": bool(summary.get("uses_gt_for_query_selection")),
            "uses_future": bool(summary.get("uses_future")),
            "query_count_per_frame": float(summary.get("query_count_per_frame") or 0.0),
            "carrier_count_per_chunk": carrier_count,
            "frame_count": int(summary.get("frame_count") or 0),
            "projection_valid_rate_model_valid": metric.get("projection_valid_rate_model_valid", ""),
            "uv_in01_rate_all_observations": metric.get("uv_in01_rate_all_observations", ""),
            "visible_confident_observation_rate": metric.get("visible_confident_observation_rate", ""),
            "uv_in01_rate_visible_confident_observations": metric.get("uv_in01_rate_visible_confident_observations", ""),
            "uv_in01_rate_valid_observations": metric.get("in_image_rate", ""),
            "in_image_rate": metric.get("in_image_rate", ""),
            "visibility_mean_in_image": metric.get("visibility_mean_in_image", ""),
            "confidence_mean_in_image": metric.get("confidence_mean_in_image", ""),
            "runtime_sec_per_chunk": float(summary.get("runtime_sec") or 0.0),
            "gpu_memory_peak_MB": summary.get("gpu_memory_peak_MB", ""),
            "gpu_memory_peak_recorded": bool(summary.get("gpu_memory_peak_recorded")),
            "D4RT_decode_success_rate": 1.0 if int(summary.get("decode_error_count") or 0) == 0 else 0.0,
            "boundary_query_rate": _safe_rate(source_by_name.get("mask_boundary_band", 0), carrier_count),
            "conflict_query_rate": _safe_rate(source_by_name.get("competing_mask_boundary", 0), carrier_count),
            "semantic_gradient_query_rate": _safe_rate(source_by_name.get("semantic_gradient", 0), carrier_count),
            "strata_present": ";".join(present),
            "missing_strata": ";".join(missing_strata),
            "status": "audited",
        }
    )
    for stratum in REQUIRED_STRATA:
        row[f"query_source_count_{stratum}"] = int(source_by_name.get(stratum, 0))
    return row


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q0_rows = [_audit_run(spec) for spec in RUNS]
    stratified_rows = [_audit_stratified_run(spec) for spec in STRATIFIED_RUNS]
    variant_rows = q0_rows + stratified_rows

    stratum_rows: list[dict[str, Any]] = []
    for row in variant_rows:
        present = set(str(row.get("strata_present", "")).split(";")) if row.get("strata_present") else set()
        for stratum in REQUIRED_STRATA:
            stratum_rows.append(
                {
                    "schema_version": "stream4d_v103_phase2_query_stratum_row_v1",
                    "phase_id": "v103_phase2_d4rt_query_audit",
                    "scene_id": row["scene_id"],
                    "variant_id": row["variant_id"],
                    "query_stratum": stratum,
                    "present": stratum in present,
                    "query_source_count": int(row.get(f"query_source_count_{stratum}") or (row["carrier_count_per_chunk"] if stratum == "uniform_grid" and row["carrier_count_per_chunk"] != "" else 0)),
                    "notes": "existing v65 runner used _grid_sources(mask_aware_min_points_per_mask=0)" if stratum == "uniform_grid" else "",
                }
            )

    selected_rows = [r for r in stratified_rows if r.get("status") == "audited"]
    decode_errors = sum(int(r.get("decode_error_count") or 0) for r in selected_rows)
    uv_rates = [float(r["uv_in01_rate_visible_confident_observations"]) for r in selected_rows if r.get("uv_in01_rate_visible_confident_observations") != ""]
    raw_uv_rates = [float(r["uv_in01_rate_all_observations"]) for r in selected_rows if r.get("uv_in01_rate_all_observations") != ""]
    valid_rates = [float(r["projection_valid_rate_model_valid"]) for r in selected_rows if r.get("projection_valid_rate_model_valid") != ""]
    carrier_counts = [int(r["carrier_count_per_chunk"]) for r in selected_rows if r.get("carrier_count_per_chunk") != ""]
    query_density = [float(r["query_count_per_frame"]) for r in selected_rows if r.get("query_count_per_frame") != ""]
    runtimes = [float(r["runtime_sec_per_chunk"]) for r in selected_rows if r.get("runtime_sec_per_chunk") != ""]
    gpu_peaks = [float(r["gpu_memory_peak_MB"]) for r in selected_rows if r.get("gpu_memory_peak_MB") != ""]
    all_required_strata_present = all(
        all(stratum in set(str(r.get("strata_present", "")).split(";")) for r in selected_rows)
        for stratum in REQUIRED_STRATA
    ) and bool(selected_rows)

    gate_specs = [
        ("decode_error_count_eq_0", decode_errors == 0, decode_errors, 0),
        ("selected_variant_count_eq_2", len(selected_rows) == 2, len(selected_rows), 2),
        ("uv_in01_rate_visible_confident_min_ge_0p80", bool(uv_rates) and min(uv_rates) >= 0.80, min(uv_rates) if uv_rates else "", 0.80),
        ("projection_valid_rate_min_ge_0p80", bool(valid_rates) and min(valid_rates) >= 0.80, min(valid_rates) if valid_rates else "", 0.80),
        ("query_count_per_frame_min_ge_1024", bool(query_density) and min(query_density) >= 1024.0, min(query_density) if query_density else "", 1024.0),
        ("carrier_count_per_chunk_min_ge_32768", bool(carrier_counts) and min(carrier_counts) >= 32768, min(carrier_counts) if carrier_counts else "", 32768),
        ("runtime_sec_per_chunk_max_le_180", bool(runtimes) and max(runtimes) <= 180.0, max(runtimes) if runtimes else "", 180.0),
        ("gpu_memory_peak_max_lt_24000MB", bool(gpu_peaks) and max(gpu_peaks) < 24000.0, max(gpu_peaks) if gpu_peaks else "", 24000.0),
        ("all_required_query_strata_present", all_required_strata_present, all_required_strata_present, True),
        ("uses_gt_for_query_selection_false", all(not bool(r.get("uses_gt_for_query_selection")) for r in selected_rows), False, False),
        ("uses_future_false", all(not bool(r.get("uses_future")) for r in selected_rows), False, False),
    ]
    gate_rows = [
        {
            "schema_version": "stream4d_v103_phase2_gate_row_v1",
            "phase_id": "v103_phase2_d4rt_query_audit",
            "gate_name": name,
            "pass": bool(ok),
            "observed": observed,
            "required": required,
        }
        for name, ok, observed, required in gate_specs
    ]
    failure_rows = []
    for gate in gate_rows:
        if not bool(gate["pass"]):
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_phase2_failure_row_v1",
                    "phase_id": "v103_phase2_d4rt_query_audit",
                    "failure_id": str(gate["gate_name"]),
                    "severity": "blocking",
                    "evidence": f"observed={gate['observed']} required={gate['required']}",
                    "repair_direction": "Continue Phase2 repair: adjust stratified query density, batching, or source coverage before Phase3.",
                }
            )
    if raw_uv_rates and min(raw_uv_rates) < 0.80:
        failure_rows.append(
            {
                "schema_version": "stream4d_v103_phase2_failure_row_v1",
                "phase_id": "v103_phase2_d4rt_query_audit",
                "failure_id": "PHASE2_RAW_ALL_TARGET_UV_IN01_LOW_DIAGNOSTIC",
                "severity": "diagnostic",
                "evidence": f"min uv_in01_rate_all_observations={min(raw_uv_rates)}; visible/confident uv gate is recorded separately and passes if >=0.80.",
                "repair_direction": "Do not count invisible low-visibility target frames as positive projection observations; Phase3 reliability uses visibility/confidence weighting.",
            }
        )
    for row in variant_rows:
        if row.get("failure_reason"):
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_phase2_failure_row_v1",
                    "phase_id": "v103_phase2_d4rt_query_audit",
                    "failure_id": "PHASE2_INPUT_ARTIFACT_MISSING",
                    "severity": "blocking",
                    "evidence": row["failure_reason"],
                    "repair_direction": "Rerun the missing D4RT query job or fix output path mapping.",
                }
            )

    blocking_failure_count = sum(1 for r in failure_rows if str(r.get("severity")) == "blocking")
    diagnostic_failure_count = sum(1 for r in failure_rows if str(r.get("severity")) == "diagnostic")
    phase2_pass = all(bool(r["pass"]) for r in gate_rows)
    decision = "PASS_ENTER_PHASE3_CARRIER_FILTERING" if phase2_pass else "NO_GO_REPAIR_PHASE2_D4RT_QUERY"

    variant_csv = OUT_DIR / "query_variant_rows.csv"
    stratum_csv = OUT_DIR / "query_stratum_rows.csv"
    gate_csv = OUT_DIR / "gate_rows.csv"
    failure_csv = OUT_DIR / "failure_rows.csv"
    _write_csv(variant_csv, variant_rows)
    _write_csv(stratum_csv, stratum_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)
    summary = {
        "schema_version": "stream4d_v103_phase2_d4rt_query_audit_summary_v1",
        "phase_id": "v103_phase2_d4rt_query_audit",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - started,
        "decision": decision,
        "phase2_pass": phase2_pass,
        "failure_count": blocking_failure_count,
        "blocking_failure_count": blocking_failure_count,
        "diagnostic_failure_count": diagnostic_failure_count,
        "variant_count": len(variant_rows),
        "selected_variant_id": "Q1_stratified_grid30_first32",
        "scenes": [r["scene_id"] for r in variant_rows],
        "variant_ids": sorted(set(str(r["variant_id"]) for r in variant_rows)),
        "required_strata": REQUIRED_STRATA,
        "all_required_query_strata_present": all_required_strata_present,
        "min_uv_in01_rate_all_observations_diagnostic": min(raw_uv_rates) if raw_uv_rates else None,
        "min_uv_in01_rate_visible_confident_observations": min(uv_rates) if uv_rates else None,
        "min_projection_valid_rate_model_valid": min(valid_rates) if valid_rates else None,
        "min_carrier_count_per_chunk": min(carrier_counts) if carrier_counts else None,
        "min_query_count_per_frame": min(query_density) if query_density else None,
        "max_runtime_sec_per_chunk": max(runtimes) if runtimes else None,
        "max_gpu_memory_peak_MB": max(gpu_peaks) if gpu_peaks else None,
        "truthfulness_note": "This audit keeps the failed Q0 grid16 rows and selects the repaired Q1 stratified grid30 rows for Phase2 gates. Raw all-target uv is diagnostic because invisible/low-visibility target frames are not positive projection observations.",
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "query_variant_rows": _rel(variant_csv),
            "query_stratum_rows": _rel(stratum_csv),
            "gate_rows": _rel(gate_csv),
            "failure_rows": _rel(failure_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase2_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
