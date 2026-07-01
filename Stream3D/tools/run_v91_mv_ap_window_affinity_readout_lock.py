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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


V90_PHASE0 = ROOT / "outputs/audit/v90_phase0_mv_ap_contract"
V90_PHASE4 = ROOT / "outputs/audit/v90_phase4_geo_semantic_witness_cover"
V90_PHASE7F = ROOT / "outputs/audit/v90_phase7f_dev_scene_balanced_score_repair"
V90_PHASE9 = ROOT / "outputs/audit/v90_phase9_holdout_mv_ap"
V90_PHASE11 = ROOT / "outputs/audit/v90_phase11_casebook"

OUT0 = ROOT / "outputs/audit/v91_phase0_mv_ap_contract"
OUT1 = ROOT / "outputs/audit/v91_phase1_variant_resurrection"
OUT2 = ROOT / "outputs/audit/v91_phase2_failure_decomposition"
OUT8 = ROOT / "outputs/audit/v91_phase8_dev_selection"

PRIMARY_SEED = "F_W9a_drop_broad_low_h9_5_broad_scene_orig_ge060"
MINIMAL_SEED = "F_W9a_drop_broad_low_h9_2_broad_scene_orig_ge060"


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


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _copy_if_exists(src: Path, dst_root: Path, dst_name: str | None = None) -> Path | None:
    if not src.exists():
        return None
    dst = dst_root / (dst_name or src.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _write_sha(root: Path) -> None:
    payload: dict[str, str] = {}
    for path in sorted(root.glob("*")):
        if path.is_file() and path.name != "SHA256SUMS.json":
            payload[_rel(path)] = _sha256(path)
    _write_json(root / "SHA256SUMS.json", payload)


def _phase0(started: float) -> dict[str, Any]:
    OUT0.mkdir(parents=True, exist_ok=True)
    v90 = _read_json(V90_PHASE0 / "summary.json")
    contract = _read_json(V90_PHASE0 / "mv_ap_contract.json")
    dev_windows = _read_csv(V90_PHASE0 / "window_support_rows.csv")
    holdout_windows = _read_csv(V90_PHASE9 / "holdout_window_support_rows.csv")
    evaluator_rows = _read_csv(V90_PHASE0 / "evaluator_source_rows.csv")
    _copy_if_exists(V90_PHASE0 / "mv_ap_contract.json", OUT0)
    _write_csv(OUT0 / "window_support_rows.csv", dev_windows + [{**row, "split": "holdout"} for row in holdout_windows])
    _write_csv(OUT0 / "evaluator_source_rows.csv", evaluator_rows)
    metric_boundary_rows = [
        {
            "artifact_id": "v90_phase0_contract",
            "path": _rel(V90_PHASE0 / "summary.json"),
            "exists": (V90_PHASE0 / "summary.json").exists(),
            "sha256": _sha256(V90_PHASE0 / "summary.json") if (V90_PHASE0 / "summary.json").exists() else "",
            "role": "v91_contract_source",
            "method_mode_allowed": True,
        },
        {
            "artifact_id": "v90_phase7f_seed",
            "path": _rel(V90_PHASE7F / "best_variant_summary.json"),
            "exists": (V90_PHASE7F / "best_variant_summary.json").exists(),
            "sha256": _sha256(V90_PHASE7F / "best_variant_summary.json") if (V90_PHASE7F / "best_variant_summary.json").exists() else "",
            "role": "v91_pre_registered_dev_seed_source",
            "method_mode_allowed": True,
            "notes": "dev-only source; not a holdout result",
        },
        {
            "artifact_id": "v90_phase9_holdout",
            "path": _rel(V90_PHASE9 / "summary.json"),
            "exists": (V90_PHASE9 / "summary.json").exists(),
            "sha256": _sha256(V90_PHASE9 / "summary.json") if (V90_PHASE9 / "summary.json").exists() else "",
            "role": "v90_holdout_reference_do_not_tune_v91",
            "method_mode_allowed": False,
            "notes": "read only to preserve no-retune boundary",
        },
    ]
    _write_csv(OUT0 / "metric_boundary_rows.csv", metric_boundary_rows)
    v90_holdout = _read_json(V90_PHASE9 / "summary.json")
    pass_conditions = {
        "formal_metric_source_eq_v65": bool(v90.get("formal_metric_source_eq_v65")),
        "local_support_policy_eq_local_window_gt_projection": v90.get("support_policy_local_window") == "local_window_gt_projection",
        "missing_mask_raster_count_eq_0": _num(v90.get("missing_mask_raster_count")) == 0,
        "dev_duplicate_frame_mask_conflict_count_eq_0": _num(v90.get("duplicate_frame_mask_conflict_count")) == 0,
        "uses_gt_for_prediction_count_eq_0": _num(v90.get("uses_gt_for_prediction_count")) == 0,
        "uses_future_count_eq_0": _num(v90.get("uses_future_count")) == 0,
    }
    summary = {
        "phase": "v91_phase0_mv_ap_contract",
        "schema": "stream4d_v91_phase0_mv_ap_contract_v1",
        "source_phase": "v90_phase0_mv_ap_contract plus v90 holdout window manifest",
        "AP_thresholds_actual": v90.get("AP_thresholds_actual", contract.get("ap_thresholds_actual")),
        "AP_threshold_contract_name": "MV_AP_window@0.50:0.90",
        "AP_threshold_list_hash": v90.get("AP_threshold_list_hash", ""),
        "formal_metric_source": v90.get("formal_metric_source", contract.get("formal_metric_source")),
        "formal_metric_source_eq_v65": bool(v90.get("formal_metric_source_eq_v65")),
        "local_support_policy": v90.get("support_policy_local_window", contract.get("support_policy_local_window")),
        "window_count_dev": len(dev_windows),
        "window_count_holdout": len(holdout_windows),
        "window_count_external": 0,
        "missing_mask_raster_count": _num(v90.get("missing_mask_raster_count")),
        "same_frame_collision_count_dev_contract": _num(v90.get("duplicate_frame_mask_conflict_count")),
        "holdout_B0_same_frame_collision_count_diagnostic": _num(v90_holdout.get("holdout_B0", {}).get("same_frame_collision_count")),
        "holdout_best_real_same_frame_collision_count_reference": _num(v90_holdout.get("holdout_best_real", {}).get("same_frame_collision_count")),
        "uses_gt_for_prediction_count": _num(v90.get("uses_gt_for_prediction_count")),
        "uses_future_count": _num(v90.get("uses_future_count")),
        "pass_conditions": pass_conditions,
        "phase0_contract_pass": all(pass_conditions.values()),
        "caveat": "v90 holdout B0 diagnostic has collisions, but v91 method candidates must be WTA/collision-free before holdout scoring.",
        "runtime_sec": time.time() - started,
    }
    _write_json(OUT0 / "summary.json", summary)
    _write_sha(OUT0)
    return summary


def _variant_rows_from_sources() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v90_phase0 = _read_json(V90_PHASE0 / "summary.json")
    v90_final = _read_json(V90_PHASE11 / "final_decision.json")
    phase7f = _read_json(V90_PHASE7F / "best_variant_summary.json")
    phase7f_agg = {row.get("variant_id", ""): row for row in _read_csv(V90_PHASE7F / "variant_metric_aggregate_rows.csv")}
    phase7f_gate = {row.get("variant_id", ""): row for row in _read_csv(V90_PHASE7F / "variant_gate_rows.csv")}
    phase4_agg = {row.get("variant_id", ""): row for row in _read_csv(V90_PHASE4 / "mv_metric_aggregate_rows.csv")}

    def add_metric(
        *,
        variant_id: str,
        method_family: str,
        source_artifact: str,
        mean_MV_AP_window: Any,
        mean_MV_AP50_window: Any,
        mean_MV_AP25_window: Any = "",
        mean_score_free_Match50_window: Any = "",
        same_frame_collision_count: Any = 0,
        missing_mask_raster_count: Any = 0,
        uses_gt_for_prediction: Any = False,
        uses_future: Any = False,
        risk_penalty_mean: Any = "",
        risk_safe_vs_B0_proxy: Any = "",
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "variant_id": variant_id,
                "method_family": method_family,
                "source_artifact": source_artifact,
                "mean_MV_AP_window": mean_MV_AP_window,
                "mean_MV_AP50_window": mean_MV_AP50_window,
                "mean_MV_AP25_window": mean_MV_AP25_window,
                "mean_score_free_Match50_window": mean_score_free_Match50_window,
                "same_frame_collision_count": same_frame_collision_count,
                "missing_mask_raster_count": missing_mask_raster_count,
                "uses_gt_for_prediction": uses_gt_for_prediction,
                "uses_future": uses_future,
                "risk_penalty_mean": risk_penalty_mean,
                "risk_safe_vs_B0_proxy": risk_safe_vs_B0_proxy,
                "notes": notes,
            }
        )

    add_metric(
        variant_id="B0_local_only",
        method_family="baseline_real",
        source_artifact=_rel(V90_PHASE0 / "summary.json"),
        mean_MV_AP_window=v90_phase0.get("B0_MV_AP_window", ""),
        mean_MV_AP50_window=v90_phase0.get("B0_MV_AP50_window", ""),
        notes="dev baseline from v90 corrected MV_AP_window contract",
    )
    add_metric(
        variant_id="C0_semantic_only_control",
        method_family="control",
        source_artifact=_rel(V90_PHASE0 / "summary.json"),
        mean_MV_AP_window=v90_phase0.get("C0_MV_AP_window", ""),
        mean_MV_AP50_window=v90_phase0.get("C0_MV_AP50_window", ""),
        notes="semantic-only control",
    )
    control = phase7f.get("control_threshold_metrics", {})
    add_metric(
        variant_id=control.get("variant_id", "P3_C0_area_semantic_hybrid_score"),
        method_family="control",
        source_artifact=_rel(V90_PHASE7F / "best_variant_summary.json"),
        mean_MV_AP_window=control.get("mean_MV_AP_window", ""),
        mean_MV_AP50_window=control.get("mean_MV_AP50_window", ""),
        mean_MV_AP25_window=control.get("mean_MV_AP25_window", ""),
        mean_score_free_Match50_window=control.get("mean_score_free_Match50_window", ""),
        same_frame_collision_count=control.get("same_frame_collision_count", ""),
        missing_mask_raster_count=control.get("missing_mask_raster_count", ""),
        uses_gt_for_prediction=control.get("uses_gt_for_prediction", False),
        uses_future=control.get("uses_future", False),
        notes="v91 mandatory area-semantic hybrid control from v90 Phase7f",
    )
    s3d = v90_final.get("Stream3D_local_diagnostic", {})
    add_metric(
        variant_id="S3D_L1_local_merged_masks",
        method_family="stream3d_local_diagnostic",
        source_artifact=_rel(V90_PHASE11 / "final_decision.json"),
        mean_MV_AP_window=s3d.get("mean_MV_AP_window", v90_phase0.get("Stream3D_S3D_L1_MV_AP_window", "")),
        mean_MV_AP50_window=s3d.get("mean_MV_AP50_window", v90_phase0.get("Stream3D_S3D_L1_MV_AP50_window", "")),
        mean_MV_AP25_window=s3d.get("mean_MV_AP25_window", ""),
        mean_score_free_Match50_window=s3d.get("mean_score_free_Match50_window", ""),
        notes="diagnostic only; uses Stream3D local geometry path",
    )
    for variant_id, alias in [
        ("W9b_risk_balanced_p165_plus_carving", "W9b_v90_risk_safe"),
        ("W9a_risk_balanced_p135_plus_carving", "W9a_v90_aggressive"),
    ]:
        row = phase4_agg.get(variant_id, {})
        add_metric(
            variant_id=alias,
            method_family="ours_real",
            source_artifact=_rel(V90_PHASE4 / "mv_metric_aggregate_rows.csv"),
            mean_MV_AP_window=row.get("mean_MV_AP_window", ""),
            mean_MV_AP50_window=row.get("mean_MV_AP50_window", ""),
            mean_MV_AP25_window=row.get("mean_MV_AP25_window", ""),
            mean_score_free_Match50_window=row.get("mean_score_free_Match50_window", ""),
            same_frame_collision_count=row.get("same_frame_collision_count", ""),
            uses_gt_for_prediction=row.get("uses_gt_for_prediction", False),
            uses_future=row.get("uses_future", False),
            risk_penalty_mean=row.get("risk_penalty_mean", ""),
            notes=f"v90 Phase4 source variant {variant_id}",
        )
    for variant_id, label in [(PRIMARY_SEED, "primary_seed"), (MINIMAL_SEED, "minimal_ablation_seed")]:
        row = phase7f_agg.get(variant_id, {})
        gate = phase7f_gate.get(variant_id, {})
        add_metric(
            variant_id=variant_id,
            method_family="ours_real_pre_registered_v91_seed",
            source_artifact=_rel(V90_PHASE7F / "variant_metric_aggregate_rows.csv"),
            mean_MV_AP_window=row.get("mean_MV_AP_window", ""),
            mean_MV_AP50_window=row.get("mean_MV_AP50_window", ""),
            mean_MV_AP25_window=row.get("mean_MV_AP25_window", ""),
            mean_score_free_Match50_window=row.get("mean_score_free_Match50_window", ""),
            same_frame_collision_count=row.get("same_frame_collision_count", ""),
            missing_mask_raster_count=row.get("missing_mask_raster_count", ""),
            uses_gt_for_prediction=row.get("uses_gt_for_prediction", False),
            uses_future=row.get("uses_future", False),
            risk_penalty_mean=gate.get("risk_penalty_mean", ""),
            risk_safe_vs_B0_proxy=gate.get("risk_safe_vs_B0_proxy", ""),
            notes=f"{label}; dev-only v90 Phase7f provenance, must be revalidated by v91 gates",
        )
    return rows, phase7f


def _phase1(started: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    OUT1.mkdir(parents=True, exist_ok=True)
    rows, phase7f = _variant_rows_from_sources()
    b0 = next((row for row in rows if row["variant_id"] == "B0_local_only"), {})
    c0 = next((row for row in rows if row["variant_id"] == "C0_semantic_only_control"), {})
    b0_ap = _num(b0.get("mean_MV_AP_window"))
    c0_ap = _num(c0.get("mean_MV_AP_window"))
    gate_rows: list[dict[str, Any]] = []
    todo_rows: list[dict[str, Any]] = []
    for row in rows:
        ap = _num(row.get("mean_MV_AP_window"))
        sf50 = _num(row.get("mean_score_free_Match50_window"))
        ap50 = _num(row.get("mean_MV_AP50_window"))
        is_method = str(row.get("method_family", "")).startswith("ours_real")
        candidate_label = "diagnostic_or_control"
        if is_method and ap >= b0_ap + 0.005:
            candidate_label = "phase2_diagnosis_candidate"
        if is_method and ap >= c0_ap + 0.005:
            candidate_label = "phase3_4_repair_pool_candidate"
        if is_method and sf50 and sf50 - ap50 >= 0.10:
            candidate_label += "+ranking_candidate"
        gate_rows.append(
            {
                **row,
                "B0_gap_MV_AP_window": ap - b0_ap,
                "C0_gap_MV_AP_window": ap - c0_ap,
                "phase1_candidate_label": candidate_label,
                "materialized": row.get("mean_MV_AP_window") not in {"", None},
            }
        )
        if row.get("mean_MV_AP_window") in {"", None}:
            todo_rows.append(
                {
                    "variant_id": row.get("variant_id", ""),
                    "missing_fields": "MV_AP_window/MV_AP50_window",
                    "required_fields": "frame_id,mask_id,mv_object_id,score,raster_path",
                    "provenance": row.get("source_artifact", ""),
                }
            )
    best_real = max(
        [row for row in rows if str(row.get("method_family", "")).startswith("ours_real")],
        key=lambda row: _num(row.get("mean_MV_AP_window")),
        default={},
    )
    summary = {
        "phase": "v91_phase1_variant_resurrection",
        "schema": "stream4d_v91_phase1_variant_resurrection_v1",
        "source": "v90/v91 pre-registered dev metric provenance lock; no new holdout scoring",
        "variant_count": len(rows),
        "materialization_todo_count": len(todo_rows),
        "B0_MV_AP_window": b0.get("mean_MV_AP_window", ""),
        "C0_MV_AP_window": c0.get("mean_MV_AP_window", ""),
        "best_real_variant": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
        "phase1_candidate_found": bool(best_real),
        "dev_seed_source_decision": phase7f.get("decision", ""),
        "runtime_sec": time.time() - started,
    }
    _write_csv(OUT1 / "all_variant_metric_rows.csv", rows)
    _write_csv(OUT1 / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT1 / "materialization_adapter_todo_rows.csv", todo_rows)
    _write_json(OUT1 / "summary.json", summary)
    _write_sha(OUT1)
    return summary, rows


def _phase2(started: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    OUT2.mkdir(parents=True, exist_ok=True)
    primary = next((row for row in rows if row.get("variant_id") == PRIMARY_SEED), {})
    control_rows = [row for row in rows if row.get("method_family") == "control"]
    best_control = max(control_rows, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    score_free = _num(primary.get("mean_score_free_Match50_window"))
    ap50 = _num(primary.get("mean_MV_AP50_window"))
    best_control_ap = _num(best_control.get("mean_MV_AP_window"))
    best_control_ap50 = _num(best_control.get("mean_MV_AP50_window"))
    primary_ap = _num(primary.get("mean_MV_AP_window"))
    extent = score_free < 0.30
    ranking = (score_free - ap50) >= 0.10
    control_bias = best_control_ap >= primary_ap or best_control_ap50 >= ap50
    support_bug = (
        _num(primary.get("same_frame_collision_count")) > 0
        or _num(primary.get("missing_mask_raster_count")) > 0
        or _bool(primary.get("uses_gt_for_prediction"))
        or _bool(primary.get("uses_future"))
    )
    control_bias_rows = [
        {
            "best_real_variant": PRIMARY_SEED,
            "best_real_MV_AP_window": primary_ap,
            "best_real_MV_AP50_window": ap50,
            "best_control_variant": best_control.get("variant_id", ""),
            "best_control_MV_AP_window": best_control_ap,
            "best_control_MV_AP50_window": best_control_ap50,
            "control_ge_real_MV_AP_window": best_control_ap >= primary_ap,
            "control_ge_real_MV_AP50_window": best_control_ap50 >= ap50,
            "control_bias_blocker": control_bias,
        }
    ]
    failure_casebook_rows = [
        {
            "variant_id": PRIMARY_SEED,
            "failure_type": "EXTENT_BLOCKER" if extent else "",
            "evidence": "score_free_Match50_window < 0.30",
            "score_free_Match50_window": score_free,
            "threshold": 0.30,
        },
        {
            "variant_id": PRIMARY_SEED,
            "failure_type": "CONTROL_BIAS_BLOCKER" if control_bias else "",
            "evidence": "v91 requires MV_AP and MV_AP50 control margins; best control MV_AP50 is not beaten",
            "best_real_MV_AP50_window": ap50,
            "best_control_MV_AP50_window": best_control_ap50,
        },
    ]
    summary = {
        "phase": "v91_phase2_failure_decomposition",
        "schema": "stream4d_v91_phase2_failure_decomposition_v1",
        "diagnosed_variant": PRIMARY_SEED,
        "EXTENT_BLOCKER": extent,
        "RANKING_BLOCKER": ranking,
        "GROUPING_BLOCKER": False,
        "CONTROL_BIAS_BLOCKER": control_bias,
        "SUPPORT_OR_EVALUATOR_BUG": support_bug,
        "score_free_Match50_window": score_free,
        "MV_AP50_window": ap50,
        "best_control_variant": best_control.get("variant_id", ""),
        "best_control_MV_AP_window": best_control_ap,
        "best_control_MV_AP50_window": best_control_ap50,
        "next_action": "Phase3/4 extent/control repair before freezing; do not run holdout from current F seed" if control_bias or extent else "Phase8 selection may proceed",
        "runtime_sec": time.time() - started,
    }
    _write_csv(OUT2 / "control_bias_rows.csv", control_bias_rows)
    _write_csv(OUT2 / "failure_casebook_rows.csv", failure_casebook_rows)
    _write_csv(OUT2 / "gt_top_iou_rows.csv", [])
    _write_csv(OUT2 / "pred_top_iou_rows.csv", [])
    _write_csv(OUT2 / "grouping_error_rows.csv", [])
    _write_csv(OUT2 / "ranking_error_rows.csv", [])
    _write_json(OUT2 / "summary.json", summary)
    _write_sha(OUT2)
    return summary


def _phase8(started: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    OUT8.mkdir(parents=True, exist_ok=True)
    b0 = next((row for row in rows if row["variant_id"] == "B0_local_only"), {})
    controls = [row for row in rows if row.get("method_family") == "control"]
    best_control = max(controls, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    real_rows = [row for row in rows if str(row.get("method_family", "")).startswith("ours_real")]
    best_real = max(real_rows, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    b0_ap = _num(b0.get("mean_MV_AP_window"))
    b0_ap50 = _num(b0.get("mean_MV_AP50_window"))
    real_ap = _num(best_real.get("mean_MV_AP_window"))
    real_ap50 = _num(best_real.get("mean_MV_AP50_window"))
    real_ap25 = _num(best_real.get("mean_MV_AP25_window"))
    ctrl_ap = _num(best_control.get("mean_MV_AP_window"))
    ctrl_ap50 = _num(best_control.get("mean_MV_AP50_window"))
    gate = {
        "best_real_MV_AP_window_ge_B0_plus_0p010": real_ap >= b0_ap + 0.010,
        "best_real_MV_AP50_window_ge_B0_plus_0p020": real_ap50 >= b0_ap50 + 0.020,
        "best_real_MV_AP_window_ge_control_plus_0p005": real_ap >= ctrl_ap + 0.005,
        "best_real_MV_AP50_window_ge_control_plus_0p010": real_ap50 >= ctrl_ap50 + 0.010,
        "same_frame_collision_count_eq_0": _num(best_real.get("same_frame_collision_count")) == 0,
        "missing_mask_raster_count_eq_0": _num(best_real.get("missing_mask_raster_count")) == 0,
        "uses_gt_for_prediction_false": not _bool(best_real.get("uses_gt_for_prediction")),
        "uses_future_false": not _bool(best_real.get("uses_future")),
    }
    strong_gate = {
        "best_real_MV_AP_window_ge_0p060": real_ap >= 0.060,
        "best_real_MV_AP50_window_ge_0p160": real_ap50 >= 0.160,
        "best_real_MV_AP25_window_ge_0p380": real_ap25 >= 0.380,
        "best_real_MV_AP_window_ge_control_plus_0p005": real_ap >= ctrl_ap + 0.005,
        "risk_safe": _bool(best_real.get("risk_safe_vs_B0_proxy")) or _num(best_real.get("risk_penalty_mean"), 1.0) <= 0.6680042860969729,
    }
    dev_pass = all(gate.values())
    strong_pass = all(strong_gate.values())
    frozen_config = {
        "frozen": bool(dev_pass and strong_pass),
        "variant_id": best_real.get("variant_id", ""),
        "source_artifact": best_real.get("source_artifact", ""),
        "reason": "dev gates pass" if dev_pass and strong_pass else "dev gate failed; holdout must not run",
        "gate": gate,
        "strong_gate": strong_gate,
    }
    control_comparison = [
        {
            "best_real_variant": best_real.get("variant_id", ""),
            "best_real_MV_AP_window": real_ap,
            "best_real_MV_AP50_window": real_ap50,
            "best_control_variant": best_control.get("variant_id", ""),
            "best_control_MV_AP_window": ctrl_ap,
            "best_control_MV_AP50_window": ctrl_ap50,
            "real_minus_control_MV_AP_window": real_ap - ctrl_ap,
            "real_minus_control_MV_AP50_window": real_ap50 - ctrl_ap50,
        }
    ]
    casebook = [
        {
            "variant_id": best_real.get("variant_id", ""),
            "case_type": "dev_gate_failure" if not dev_pass else "dev_gate_pass",
            "details": "MV_AP50 control margin failed" if not gate["best_real_MV_AP50_window_ge_control_plus_0p010"] else "",
        }
    ]
    summary = {
        "phase": "v91_phase8_dev_selection",
        "schema": "stream4d_v91_phase8_dev_selection_v1",
        "best_real_variant": best_real.get("variant_id", ""),
        "best_control_variant": best_control.get("variant_id", ""),
        "B0_variant": b0.get("variant_id", ""),
        "best_real_MV_AP_window": real_ap,
        "best_real_MV_AP50_window": real_ap50,
        "best_real_MV_AP25_window": real_ap25,
        "best_control_MV_AP_window": ctrl_ap,
        "best_control_MV_AP50_window": ctrl_ap50,
        "dev_progress_gate": gate,
        "dev_progress_gate_pass": dev_pass,
        "strong_local_dev_gate": strong_gate,
        "strong_local_dev_gate_pass": strong_pass,
        "decision": "FREEZE_CANDIDATE" if dev_pass and strong_pass else "DEV_GATE_FAIL_CONTINUE_REPAIR",
        "next_action": "CONTROL_BIAS/EXTENT repair: improve AP50 real-control margin before Phase9; no holdout yet" if not dev_pass else "freeze and run Phase9 once",
        "runtime_sec": time.time() - started,
    }
    _write_csv(OUT8 / "all_variant_metric_rows.csv", rows)
    _write_csv(OUT8 / "control_comparison_rows.csv", control_comparison)
    _write_csv(OUT8 / "casebook_rows.csv", casebook)
    _write_json(OUT8 / "frozen_candidate_config.json", frozen_config)
    _write_json(OUT8 / "summary.json", summary)
    _write_sha(OUT8)
    return summary


def run() -> dict[str, Any]:
    started = time.time()
    phase0 = _phase0(started)
    phase1, rows = _phase1(started)
    phase2 = _phase2(started, rows)
    phase8 = _phase8(started, rows)
    summary = {
        "phase": "v91_mv_ap_window_affinity_readout_lock",
        "schema": "stream4d_v91_lock_v1",
        "phase0_contract_pass": phase0.get("phase0_contract_pass"),
        "phase1_best_real_variant": phase1.get("best_real_variant"),
        "phase2_EXTENT_BLOCKER": phase2.get("EXTENT_BLOCKER"),
        "phase2_CONTROL_BIAS_BLOCKER": phase2.get("CONTROL_BIAS_BLOCKER"),
        "phase8_decision": phase8.get("decision"),
        "phase8_dev_progress_gate_pass": phase8.get("dev_progress_gate_pass"),
        "phase8_strong_local_dev_gate_pass": phase8.get("strong_local_dev_gate_pass"),
        "next_action": phase8.get("next_action"),
        "runtime_sec": time.time() - started,
    }
    _write_json(ROOT / "outputs/audit/v91_lock_summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
