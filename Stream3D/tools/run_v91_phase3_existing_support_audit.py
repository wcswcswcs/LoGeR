from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V90_PHASE3 = ROOT / "outputs/audit/v90_phase3_carrier_supported_carving"
V91_OUT = ROOT / "outputs/audit/v91_phase3_carrier_visible_support"
V91_PHASE8 = ROOT / "outputs/audit/v91_phase8_dev_selection"


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _phase8_baselines() -> dict[str, dict[str, Any]]:
    return {row.get("variant_id", ""): row for row in _read_csv(V91_PHASE8 / "all_variant_metric_rows.csv")}


def _gate_rows(metric_rows: list[dict[str, str]], baselines: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    b0 = baselines.get("B0_local_only", {})
    control = baselines.get("P3_C0_area_semantic_hybrid_score", {})
    out: list[dict[str, Any]] = []
    for row in metric_rows:
        mv_ap = _num(row.get("mean_MV_AP_window"))
        mv_ap50 = _num(row.get("mean_MV_AP50_window"))
        gates = {
            "variant_id": row.get("variant_id", ""),
            "MV_AP_window": mv_ap,
            "MV_AP50_window": mv_ap50,
            "MV_AP25_window": _num(row.get("mean_MV_AP25_window")),
            "mean_score_free_Match50_window": _num(row.get("mean_score_free_Match50_window")),
            "same_frame_collision_count": _int(row.get("same_frame_collision_count")),
            "missing_mask_raster_count": _int(row.get("missing_mask_raster_count")),
            "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
            "uses_future": _bool(row.get("uses_future")),
            "real_minus_B0_MV_AP_window": mv_ap - _num(b0.get("mean_MV_AP_window")),
            "real_minus_B0_MV_AP50_window": mv_ap50 - _num(b0.get("mean_MV_AP50_window")),
            "real_minus_best_control_MV_AP_window": mv_ap - _num(control.get("mean_MV_AP_window")),
            "real_minus_best_control_MV_AP50_window": mv_ap50 - _num(control.get("mean_MV_AP50_window")),
            "required_MV_AP50_window_for_v91_phase8_control_gate": _num(control.get("mean_MV_AP50_window")) + 0.010,
        }
        gate_checks = {
            "MV_AP_window_ge_B0_plus_0p010": mv_ap >= _num(b0.get("mean_MV_AP_window")) + 0.010,
            "MV_AP50_window_ge_B0_plus_0p020": mv_ap50 >= _num(b0.get("mean_MV_AP50_window")) + 0.020,
            "MV_AP_window_ge_control_plus_0p005": mv_ap >= _num(control.get("mean_MV_AP_window")) + 0.005,
            "MV_AP50_window_ge_control_plus_0p010": mv_ap50 >= _num(control.get("mean_MV_AP50_window")) + 0.010,
            "same_frame_collision_count_eq_0": gates["same_frame_collision_count"] == 0,
            "missing_mask_raster_count_eq_0": gates["missing_mask_raster_count"] == 0,
            "uses_gt_for_prediction_false": not gates["uses_gt_for_prediction"],
            "uses_future_false": not gates["uses_future"],
        }
        out.append({**row, **gates, **{f"gate_{k}": v for k, v in gate_checks.items()}, "v91_phase8_progress_gate_pass": all(gate_checks.values())})
    return out


def _support_quality_rows(heatmap_rows: list[dict[str, str]], generated_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    generated_by_key = {
        (
            row.get("variant_id", ""),
            row.get("scene_id", ""),
            str(row.get("frame_id", "")),
            str(row.get("source_mask_id", "")),
        ): row
        for row in generated_rows
    }
    out: list[dict[str, Any]] = []
    for row in heatmap_rows:
        key = (
            row.get("variant_id", ""),
            row.get("scene_id", ""),
            str(row.get("frame_id", "")),
            str(row.get("source_mask_id", "")),
        )
        gen = generated_by_key.get(key, {})
        selected_area = max(1.0, _num(row.get("source_mask_area")))
        generated_area = max(1.0, _num(gen.get("generated_mask_area")))
        support_area = max(1.0, _num(row.get("support_area")))
        support_inside = _num(row.get("support_inside_source_mask_ratio"))
        out.append(
            {
                "variant_id": row.get("variant_id", ""),
                "scene_id": row.get("scene_id", ""),
                "window_id": row.get("window_id", ""),
                "mv_object_id": row.get("mv_object_id", ""),
                "frame_id": row.get("frame_id", ""),
                "support_carrier_count": _int(row.get("carrier_support_count")),
                "support_heatmap_area": support_area,
                "selected_mask_area": selected_area,
                "generated_mask_area": generated_area,
                "support_to_mask_ratio": support_area / selected_area,
                "mask_to_support_ratio": generated_area / support_area,
                "broad_risk": _bool(gen.get("broad_risk")),
                "hard_negative_density": 1.0 - support_inside,
                "same_frame_collision_flag": False,
                "missing_raster_flag": False,
                "support_inside_source_mask_ratio": support_inside,
                "carrier_projection_coverage_rate": _num(row.get("carrier_projection_coverage_rate")),
                "source_v90_artifact": _rel(V90_PHASE3 / "carrier_support_heatmap_rows.csv"),
            }
        )
    return out


def run() -> dict[str, Any]:
    started = time.time()
    V91_OUT.mkdir(parents=True, exist_ok=True)
    v90_summary = _read_json(V90_PHASE3 / "summary.json")
    metric_rows = _read_csv(V90_PHASE3 / "mv_metric_rows.csv")
    aggregate_rows = _read_csv(V90_PHASE3 / "mv_metric_aggregate_rows.csv")
    generated_rows = _read_csv(V90_PHASE3 / "generated_mask_rows.csv")
    eval_rows = _read_csv(V90_PHASE3 / "eval_frame_mask_rows.csv")
    heatmap_rows = _read_csv(V90_PHASE3 / "carrier_support_heatmap_rows.csv")
    baselines = _phase8_baselines()
    control_rows = _gate_rows(aggregate_rows, baselines)
    support_rows = _support_quality_rows(heatmap_rows, generated_rows)
    variant_config_rows = [
        {
            "variant_id": row.get("variant_id", ""),
            "source_phase": "v90_phase3_carrier_supported_carving",
            "source_summary": _rel(V90_PHASE3 / "summary.json"),
            "readout_family": "carrier_visible_support",
            "MV_AP_window": row.get("mean_MV_AP_window", ""),
            "MV_AP50_window": row.get("mean_MV_AP50_window", ""),
            "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
            "uses_future": row.get("uses_future", ""),
        }
        for row in aggregate_rows
    ]
    best = max(control_rows, key=lambda row: _num(row.get("MV_AP50_window")), default={})
    a0 = next((row for row in aggregate_rows if row.get("variant_id") == "A0_whole_mask_adapter"), {})
    best_a = v90_summary.get("best_carved_metrics", {})
    phase3_plan_progress_gate = bool(best_a) and (
        _num(best_a.get("mean_MV_AP_window")) >= _num(a0.get("mean_MV_AP_window")) + 0.005
        and _num(best_a.get("mean_MV_AP50_window")) >= _num(a0.get("mean_MV_AP50_window")) + 0.010
        and _int(best_a.get("same_frame_collision_count")) == 0
        and not _bool(best_a.get("uses_gt_for_prediction"))
        and not _bool(best_a.get("uses_future"))
    )
    summary = {
        "phase": "v91_phase3_carrier_visible_support",
        "schema": "stream4d_v91_phase3_existing_full_support_audit_v1",
        "source_phase": "v90_phase3_carrier_supported_carving",
        "source_reuse_policy": "no_rerun; re-audit existing full-support Phase3 artifacts under v91 gates",
        "source_summary_sha256": _sha256(V90_PHASE3 / "summary.json"),
        "native_support_rows": v90_summary.get("inputs", {}).get("native_carrier_support_rows", ""),
        "A0_whole_mask_adapter": a0,
        "best_A_variant": v90_summary.get("best_carved_variant", ""),
        "best_A_metrics": best_a,
        "best_by_AP50_under_v91_gate": best,
        "phase3_plan_progress_gate_pass": phase3_plan_progress_gate,
        "any_v91_phase8_progress_gate_pass": any(_bool(row.get("v91_phase8_progress_gate_pass")) for row in control_rows),
        "decision": "PHASE3_VISIBLE_SUPPORT_NO_GO_EXISTING_FULL_SUPPORT",
        "next_action": "continue Phase4 extent/materialization repair; do not run Phase9 holdout",
        "support_sparsity": v90_summary.get("support_sparsity", {}),
        "row_counts": {
            "variant_config_rows": len(variant_config_rows),
            "generated_mask_rows": len(generated_rows),
            "mv_object_frame_mask_rows": len(eval_rows),
            "mv_metric_rows": len(metric_rows),
            "support_quality_rows": len(support_rows),
            "control_metric_rows": len(control_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - started,
    }
    _write_csv(V91_OUT / "variant_config_rows.csv", variant_config_rows)
    _write_csv(V91_OUT / "generated_mask_rows.csv", generated_rows)
    _write_csv(V91_OUT / "mv_object_frame_mask_rows.csv", eval_rows)
    _write_csv(V91_OUT / "mv_metric_rows.csv", metric_rows)
    _write_csv(V91_OUT / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(V91_OUT / "control_metric_rows.csv", control_rows)
    _write_csv(V91_OUT / "support_quality_rows.csv", support_rows)
    _write_json(V91_OUT / "summary.json", summary)
    shutil.copy2(V90_PHASE3 / "generated_mask_manifest.json", V91_OUT / "source_generated_mask_manifest.json")
    outputs = [
        V91_OUT / "variant_config_rows.csv",
        V91_OUT / "generated_mask_rows.csv",
        V91_OUT / "mv_object_frame_mask_rows.csv",
        V91_OUT / "mv_metric_rows.csv",
        V91_OUT / "mv_metric_aggregate_rows.csv",
        V91_OUT / "control_metric_rows.csv",
        V91_OUT / "support_quality_rows.csv",
        V91_OUT / "summary.json",
        V91_OUT / "source_generated_mask_manifest.json",
    ]
    _write_json(V91_OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
