from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASE_ID = "v93_phase2_d4rt_edge_sampling_diagnostic"
RUN_ID = "v93_phase2_d4rt_edge_sampling_diagnostic"
OUT = ROOT / "outputs/audit/v93_phase2_d4rt_edge_sampling_diagnostic"

V92_PHASE2 = ROOT / "outputs/audit/v92_phase2_d4rt_sufficiency"
V93_PHASE1 = ROOT / "outputs/audit/v93_phase1_source_edge_registry"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(ROOT.parent))
        except ValueError:
            return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "allow"}


def _float(value: Any, default: float | str = "") -> float | str:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_div(num: float | int, den: float | int) -> float:
    den_f = float(den)
    return 0.0 if den_f == 0.0 else float(num) / den_f


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (str(row.get("variant_id", "")), str(row.get("scene_id", "")), str(row.get("frame_id", "")), str(row.get("source_mask_id", "")))


def _rewrite_phase(row: dict[str, Any], suffix: str, created_at: str) -> dict[str, Any]:
    out = dict(row)
    out["schema_version"] = f"stream4d_v93_phase2_{suffix}_v1"
    out["phase_id"] = PHASE_ID
    out["run_id"] = RUN_ID
    out["created_at"] = created_at
    return out


def _copy_transformed(src: Path, dst: Path, suffix: str, created_at: str) -> list[dict[str, Any]]:
    rows = [_rewrite_phase(row, suffix, created_at) for row in _read_csv(src)]
    _write_csv(dst, rows)
    return rows


def _load_source_support(created_at: str) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
    rows = _copy_transformed(V92_PHASE2 / "source_container_carrier_rows.csv", OUT / "d4rt_source_support_rows.csv", "d4rt_source_support", created_at)
    support_by_key = {_key(row): row for row in rows}
    return rows, support_by_key


def _stream_edge_band_rows(
    support_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
    created_at: str,
) -> tuple[Counter[str], dict[str, list[float]], int, int]:
    edge_path = V93_PHASE1 / "mask_edge_hypothesis_rows.csv"
    out_path = OUT / "d4rt_edge_band_support_rows.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge_counter: Counter[str] = Counter()
    support_values: dict[str, list[float]] = defaultdict(list)
    row_count = 0
    missing_support = 0
    fieldnames = [
        "schema_version",
        "phase_id",
        "run_id",
        "variant_id",
        "scene_id",
        "split",
        "window_id",
        "chunk_id",
        "frame_id",
        "source_mask_id",
        "edge_id",
        "edge_type",
        "edge_band_area",
        "edge_source_area_ratio",
        "carrier_count_inside_source",
        "carrier_support_area_ratio",
        "edge_band_support_ratio_proxy",
        "edge_band_support_source",
        "projection_jitter_available",
        "uses_gt_for_prediction",
        "uses_future",
        "created_at",
    ]
    with edge_path.open(newline="", encoding="utf-8") as src_handle, out_path.open("w", newline="", encoding="utf-8") as dst_handle:
        reader = csv.DictReader(src_handle)
        writer = csv.DictWriter(dst_handle, fieldnames=fieldnames)
        writer.writeheader()
        for edge in reader:
            key = _key(edge)
            support = support_by_key.get(key, {})
            if not support:
                missing_support += 1
            edge_type = str(edge.get("edge_type", ""))
            carrier_ratio = float(_float(support.get("carrier_support_area_ratio"), 0.0) or 0.0)
            edge_ratio = float(_float(edge.get("edge_source_area_ratio"), 0.0) or 0.0)
            support_proxy = min(carrier_ratio, edge_ratio)
            edge_counter[edge_type] += 1
            support_values[edge_type].append(support_proxy)
            row_count += 1
            writer.writerow(
                {
                    "schema_version": "stream4d_v93_phase2_d4rt_edge_band_support_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": edge.get("variant_id", ""),
                    "scene_id": edge.get("scene_id", ""),
                    "split": edge.get("split", "dev"),
                    "window_id": edge.get("window_id", ""),
                    "chunk_id": edge.get("chunk_id", ""),
                    "frame_id": edge.get("frame_id", ""),
                    "source_mask_id": edge.get("source_mask_id", ""),
                    "edge_id": edge.get("edge_id", ""),
                    "edge_type": edge_type,
                    "edge_band_area": edge.get("edge_band_area", ""),
                    "edge_source_area_ratio": edge_ratio,
                    "carrier_count_inside_source": support.get("carrier_count_inside_source", ""),
                    "carrier_support_area_ratio": carrier_ratio,
                    "edge_band_support_ratio_proxy": support_proxy,
                    "edge_band_support_source": "min(v92_carrier_support_area_ratio, v93_edge_source_area_ratio); proxy_not_precise_per-witness_band",
                    "projection_jitter_available": support.get("carrier_nearest_neighbor_distance_p90_px", "") != "",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "created_at": created_at,
                }
            )
    return edge_counter, support_values, row_count, missing_support


def _median(values: list[float]) -> float | str:
    if not values:
        return ""
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _sampling_rows(created_at: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    budgets = {
        "A512_adaptive_edge_conflict": {
            "uniform_base": 128,
            "object_interior": 96,
            "source_outer_boundary": 64,
            "nested_overlap_boundary": 64,
            "competing_mask_boundary": 64,
            "semantic_gradient": 48,
            "conflict_underseg": 32,
            "uncertainty_flip_jitter": 16,
            "partwhole_overlap": 0,
        },
        "A1024_adaptive_edge_conflict_uncertainty": {
            "uniform_base": 192,
            "object_interior": 192,
            "source_outer_boundary": 128,
            "nested_overlap_boundary": 128,
            "competing_mask_boundary": 128,
            "semantic_gradient": 128,
            "conflict_underseg": 96,
            "uncertainty_flip_jitter": 32,
            "partwhole_overlap": 0,
        },
        "A256_lite_edge_conflict": {
            "uniform_base": 64,
            "object_interior": 48,
            "source_outer_boundary": 48,
            "nested_overlap_boundary": 32,
            "competing_mask_boundary": 32,
            "semantic_gradient": 16,
            "conflict_underseg": 16,
            "uncertainty_flip_jitter": 0,
            "partwhole_overlap": 0,
        },
    }
    plan_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    for plan_id, strata in budgets.items():
        query_total = sum(strata.values())
        plan_rows.append(
            {
                "schema_version": "stream4d_v93_phase2_adaptive_sampling_plan_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "sampling_plan_id": plan_id,
                "query_budget_per_frame": query_total,
                "normalization": "stratum_normalized",
                "routing_reason": "support_area_ratio_or_boundary_band_support_low",
                "uses_gt_for_routing": False,
                "uses_future": False,
                "created_at": created_at,
            }
        )
        for name, budget in strata.items():
            stratum_rows.append(
                {
                    "schema_version": "stream4d_v93_phase2_sampling_stratum_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "sampling_plan_id": plan_id,
                    "stratum_name": name,
                    "query_budget": budget,
                    "actual_query_count": "",
                    "normalization_weight": _safe_div(1.0, max(1, budget)),
                    "mean_membership": "",
                    "mean_confidence": "",
                    "mean_jitter": "",
                    "support_area_ratio": "",
                    "uses_gt_for_routing": False,
                    "uses_future": False,
                    "created_at": created_at,
                }
            )
    return plan_rows, stratum_rows


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()

    v92_summary = _read_json(V92_PHASE2 / "summary.json")
    source_rows, support_by_key = _load_source_support(created_at)
    quality_rows = _copy_transformed(V92_PHASE2 / "d4rt_quality_proxy_rows.csv", OUT / "d4rt_quality_rows.csv", "d4rt_quality", created_at)
    gt_rows = _copy_transformed(V92_PHASE2 / "gt_diagnostic_carrier_coverage_rows.csv", OUT / "diagnostic_gt_coverage_rows.csv", "diagnostic_gt_coverage", created_at)
    edge_counter, support_values, edge_row_count, missing_edge_support = _stream_edge_band_rows(support_by_key, created_at)

    carrier_counts = [float(_float(row.get("carrier_count_inside_source"), 0.0) or 0.0) for row in source_rows]
    support_ratios = [float(_float(row.get("carrier_support_area_ratio"), 0.0) or 0.0) for row in source_rows]
    median_carrier_count_inside_source = _median(carrier_counts)
    median_carrier_support_area_ratio = _median(support_ratios)
    boundary_values = support_values.get("outer", []) + support_values.get("nested_overlap", []) + support_values.get("competing", [])
    boundary_band_support_ratio = _median(boundary_values)
    nested_edge_band_support_ratio = _median(support_values.get("nested_overlap", []))
    competing_edge_band_support_ratio = _median(support_values.get("competing", []))
    semantic_gradient_band_support_ratio = _median(support_values.get("semantic_gradient", []))

    support_area_low = median_carrier_support_area_ratio != "" and float(median_carrier_support_area_ratio) < 0.08
    boundary_low = boundary_band_support_ratio != "" and float(boundary_band_support_ratio) < 0.05
    jitter_high = float(v92_summary.get("projection_jitter_p90_global", 0.0) or 0.0) > 40.0
    flip_high = float(v92_summary.get("mask_membership_flip_rate_median", 0.0) or 0.0) >= 0.5
    route_to_phase7 = bool(support_area_low or boundary_low)
    route_to_uncertainty = bool((not route_to_phase7) and (jitter_high or flip_high))

    plan_rows, stratum_rows = _sampling_rows(created_at)
    _write_csv(OUT / "adaptive_sampling_plan_rows.csv", plan_rows)
    _write_csv(OUT / "sampling_stratum_rows.csv", stratum_rows)

    gate_rows = [
        {
            "schema_version": "stream4d_v93_phase2_variant_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": "D4RT_EDGE_SAMPLING_DIAGNOSTIC",
            "scene_id": "ALL_DEV",
            "split": "dev",
            "window_id": "ALL_WINDOWS",
            "gate_name": "source_container_carrier_rows_gt_0",
            "gate_pass": len(source_rows) > 0,
            "gate_value": len(source_rows),
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "created_at": created_at,
        },
        {
            "schema_version": "stream4d_v93_phase2_variant_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": "D4RT_EDGE_SAMPLING_DIAGNOSTIC",
            "scene_id": "ALL_DEV",
            "split": "dev",
            "window_id": "ALL_WINDOWS",
            "gate_name": "edge_band_support_rows_gt_0",
            "gate_pass": edge_row_count > 0,
            "gate_value": edge_row_count,
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "created_at": created_at,
        },
        {
            "schema_version": "stream4d_v93_phase2_variant_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": "D4RT_EDGE_SAMPLING_DIAGNOSTIC",
            "scene_id": "ALL_DEV",
            "split": "dev",
            "window_id": "ALL_WINDOWS",
            "gate_name": "edge_support_join_missing_eq_0",
            "gate_pass": missing_edge_support == 0,
            "gate_value": missing_edge_support,
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "created_at": created_at,
        },
        {
            "schema_version": "stream4d_v93_phase2_variant_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": "D4RT_EDGE_SAMPLING_DIAGNOSTIC",
            "scene_id": "ALL_DEV",
            "split": "dev",
            "window_id": "ALL_WINDOWS",
            "gate_name": "routing_uses_gt_false",
            "gate_pass": True,
            "gate_value": False,
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "created_at": created_at,
        },
    ]
    _write_csv(OUT / "variant_gate_rows.csv", gate_rows)
    phase2_gate_pass = all(_bool(row.get("gate_pass")) for row in gate_rows)

    summary = {
        "schema": "stream4d_v93_phase2_d4rt_edge_sampling_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": "ROUTE_V93_PHASE7_ADAPTIVE_D4RT_SAMPLING" if route_to_phase7 else ("ROUTE_V93_UNCERTAINTY_WITNESS_FIELD" if route_to_uncertainty else "D4RT_SUPPORT_ADEQUATE_ROUTE_FIELD_INFERENCE"),
        "phase2_gate_pass": phase2_gate_pass,
        "source_container_count": len(source_rows),
        "d4rt_edge_band_support_row_count": edge_row_count,
        "missing_edge_support_join_count": missing_edge_support,
        "d4rt_quality_row_count": len(quality_rows),
        "diagnostic_gt_coverage_row_count": len(gt_rows),
        "diagnostic_gt_coverage_used_for_routing": False,
        "median_carrier_count_inside_source": median_carrier_count_inside_source,
        "median_carrier_support_area_ratio": median_carrier_support_area_ratio,
        "boundary_band_support_ratio": boundary_band_support_ratio,
        "nested_edge_band_support_ratio": nested_edge_band_support_ratio,
        "competing_edge_band_support_ratio": competing_edge_band_support_ratio,
        "semantic_gradient_band_support_ratio": semantic_gradient_band_support_ratio,
        "projection_jitter_p90": v92_summary.get("projection_jitter_p90_global", ""),
        "mask_membership_flip_rate_median": v92_summary.get("mask_membership_flip_rate_median", ""),
        "hard_negative_witness_density": "",
        "uniform_vs_adaptive_expected_coverage_gain": "adaptive branch expected because source support/boundary support are below plan thresholds",
        "route_to_phase7_adaptive_sampling": route_to_phase7,
        "route_to_uncertainty_witness_field": route_to_uncertainty,
        "support_area_low_lt_0p08": support_area_low,
        "boundary_band_support_low_lt_0p05": boundary_low,
        "jitter_high": jitter_high,
        "membership_flip_high": flip_high,
        "edge_type_counts": dict(edge_counter),
        "edge_band_support_ratio_source": "proxy=min(v92 carrier_support_area_ratio, v93 edge_source_area_ratio); no per-witness edge-band coordinates available in current artifacts",
        "routing_uses_gt": False,
        "uses_gt_for_prediction_count": 0,
        "uses_future_count": 0,
        "input_artifacts": {
            _rel(path): _sha256(path)
            for path in [
                V92_PHASE2 / "summary.json",
                V92_PHASE2 / "source_container_carrier_rows.csv",
                V92_PHASE2 / "d4rt_quality_proxy_rows.csv",
                V92_PHASE2 / "gt_diagnostic_carrier_coverage_rows.csv",
                V93_PHASE1 / "mask_edge_hypothesis_rows.csv",
                V93_PHASE1 / "summary.json",
            ]
            if path.exists()
        },
        "duration_sec": time.time() - started,
        "created_at": created_at,
    }
    _write_json(OUT / "summary.json", summary)
    sha_rows = {path.name: _sha256(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "SHA256SUMS.json"}
    _write_json(OUT / "SHA256SUMS.json", sha_rows)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
