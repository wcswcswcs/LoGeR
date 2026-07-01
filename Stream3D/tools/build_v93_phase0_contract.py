from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASE_ID = "v93_phase0_contract"
RUN_ID = "v93_phase0_contract_lock"
OUT = ROOT / "outputs/audit/v93_phase0_contract"

V65_EVALUATOR = ROOT / "tools/run_v65_scene_multiview_ap.py"
V89_PHASE0 = ROOT / "outputs/audit/v89_phase0_mv_ap_contract"
V90_PHASE0 = ROOT / "outputs/audit/v90_phase0_mv_ap_contract"
V91_PHASE0 = ROOT / "outputs/audit/v91_phase0_mv_ap_contract"
V91_PHASE8 = ROOT / "outputs/audit/v91_phase8_dev_selection"
V91_FINAL = ROOT / "outputs/audit/v91_final_decision"
V92_PHASE0 = ROOT / "outputs/audit/v92_phase0_mv_ap_contract"
V92_FINAL = ROOT / "outputs/audit/v92_phase9_casebook"


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


def _num(value: Any, default: float | str = "") -> float | str:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "allow"}


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _all_rows() -> list[dict[str, str]]:
    return _read_csv(V91_PHASE8 / "all_variant_metric_rows.csv")


def _row(rows: list[dict[str, str]], variant_id: str) -> dict[str, str]:
    return next((row for row in rows if row.get("variant_id") == variant_id), {})


def _metric_row(variant_id: str, family: str, source_artifact: Path, row: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v93_phase0_baseline_metric_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": variant_id,
        "variant_family": family,
        "source_version": source_artifact.parent.name,
        "support_policy": "local_window_gt_projection",
        "metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "scene_id": "ALL_DEV",
        "split": "dev",
        "window_id": "ALL_DEV_WINDOWS",
        "chunk_id": "",
        "MV_AP_window": _num(row.get("mean_MV_AP_window")),
        "MV_AP50_window": _num(row.get("mean_MV_AP50_window")),
        "MV_AP25_window": _num(row.get("mean_MV_AP25_window")),
        "ScoreFreeMatch50_window": _num(row.get("mean_score_free_Match50_window")),
        "ScoreFreeMatch25_window": _num(row.get("mean_score_free_Match25_window")),
        "same_frame_collision_count": _num(row.get("same_frame_collision_count"), 0.0),
        "missing_mask_raster_count": _num(row.get("missing_mask_raster_count"), 0.0),
        "uses_gt_for_prediction_count": 1 if _bool(row.get("uses_gt_for_prediction")) else 0,
        "uses_future_count": 1 if _bool(row.get("uses_future")) else 0,
        "uses_rgbd_pose_mesh_count": 1 if _bool(row.get("uses_rgbd_pose_mesh")) else 0,
        "source_artifact": _rel(source_artifact),
        "source_artifact_sha256": _sha256(source_artifact) if source_artifact.exists() else "",
        "created_at": _created_at(),
        "notes": notes,
    }


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()

    v89 = _read_json(V89_PHASE0 / "summary.json")
    v90 = _read_json(V90_PHASE0 / "summary.json")
    v91 = _read_json(V91_PHASE0 / "summary.json")
    v92 = _read_json(V92_PHASE0 / "summary.json")
    phase8 = _read_json(V91_PHASE8 / "summary.json")
    final91 = _read_json(V91_FINAL / "decision_summary.json")
    final92 = _read_json(V92_FINAL / "final_decision.json")

    rows = _all_rows()
    b0 = _row(rows, "B0_local_only")
    c0 = _row(rows, "C0_semantic_only_control")
    p3 = _row(rows, "P3_C0_area_semantic_hybrid_score")
    s3d = _row(rows, "S3D_L1_local_merged_masks")
    best_id = str(phase8.get("best_real_variant", final91.get("best_variant", "")))
    best = _row(rows, best_id)
    if not best and best_id:
        best = {
            "mean_MV_AP_window": phase8.get("best_real_MV_AP_window", final91.get("best_real_MV_AP_window", "")),
            "mean_MV_AP50_window": phase8.get("best_real_MV_AP50_window", final91.get("best_real_MV_AP50_window", "")),
            "mean_MV_AP25_window": phase8.get("best_real_MV_AP25_window", final91.get("best_real_MV_AP25_window", "")),
            "mean_score_free_Match50_window": final91.get("best_score_free_Match50_window", ""),
            "same_frame_collision_count": 0,
            "missing_mask_raster_count": 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }

    baseline_rows = [
        _metric_row("B0_local_only", "baseline_real", V91_PHASE8 / "all_variant_metric_rows.csv", b0, "B0 local-only baseline locked from v91 Phase8 rows."),
        _metric_row("C0_semantic_only_control", "control", V91_PHASE8 / "all_variant_metric_rows.csv", c0, "Semantic-only control locked from v91 Phase8 rows."),
        _metric_row("P3_C0_area_semantic_hybrid_score", "control", V91_PHASE8 / "all_variant_metric_rows.csv", p3, "Best area-semantic control from v91 Phase8."),
        _metric_row(best_id, "v91_best_real", V91_PHASE8 / "all_variant_metric_rows.csv", best, "Best formal v91 local-window method candidate."),
        _metric_row("S3D_L1_local_merged_masks", "stream3d_local_diagnostic", V91_PHASE8 / "all_variant_metric_rows.csv", s3d, "Corrected Stream3D local-window diagnostic baseline; not a v93 method row."),
    ]
    baseline_rows = [row for row in baseline_rows if row["variant_id"]]

    evaluator_text = V65_EVALUATOR.read_text(encoding="utf-8") if V65_EVALUATOR.exists() else ""
    formal_metric_source_eq_v65 = bool(
        V65_EVALUATOR.exists()
        and "SparseSceneIoU" in evaluator_text
        and "_summarize_iou" in evaluator_text
        and _bool(v90.get("formal_metric_source_eq_v65"))
        and _bool(v91.get("formal_metric_source_eq_v65"))
        and _bool(v92.get("formal_metric_source_eq_v65"))
    )
    local_support_policy = str(v92.get("local_support_policy", v91.get("local_support_policy", v90.get("support_policy_local_window", ""))))
    thresholds = list(v92.get("AP_thresholds_actual", v91.get("AP_thresholds_actual", v90.get("AP_thresholds_actual", []))))

    method_rows = [row for row in baseline_rows if row.get("variant_family") != "stream3d_local_diagnostic"]
    missing_baseline_ids = [row["variant_id"] for row in baseline_rows if row.get("MV_AP_window") == "" or row.get("MV_AP50_window") == ""]
    same_frame_collision_count = max(float(_num(row.get("same_frame_collision_count"), 0.0) or 0.0) for row in method_rows) if method_rows else 0.0
    missing_mask_raster_count = max(float(_num(row.get("missing_mask_raster_count"), 0.0) or 0.0) for row in method_rows) if method_rows else 0.0
    uses_gt_for_prediction_count = sum(int(_num(row.get("uses_gt_for_prediction_count"), 0) or 0) for row in method_rows)
    uses_future_count = sum(int(_num(row.get("uses_future_count"), 0) or 0) for row in method_rows)
    uses_rgbd_pose_mesh_count = sum(int(_num(row.get("uses_rgbd_pose_mesh_count"), 0) or 0) for row in method_rows)

    best_control_mv = _num(phase8.get("best_control_MV_AP_window"))
    best_control_ap50 = _num(phase8.get("best_control_MV_AP50_window"))
    pass_conditions = {
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "local_support_policy_eq_local_window_gt_projection": local_support_policy == "local_window_gt_projection",
        "ap_thresholds_actual_recorded": len(thresholds) > 0,
        "v91_best_rows_available": best_id != "" and best.get("mean_MV_AP_window", "") != "",
        "best_control_rows_available": best_control_mv != "" and best_control_ap50 != "",
        "stream3d_corrected_local_baseline_available": s3d.get("mean_MV_AP_window", "") != "" or v90.get("Stream3D_S3D_L1_MV_AP_window", "") != "",
        "same_frame_collision_count_eq_0": same_frame_collision_count == 0.0,
        "missing_mask_raster_count_eq_0": missing_mask_raster_count == 0.0,
        "uses_gt_for_prediction_count_eq_0": uses_gt_for_prediction_count == 0,
        "uses_future_count_eq_0": uses_future_count == 0,
        "uses_rgbd_pose_mesh_count_eq_0": uses_rgbd_pose_mesh_count == 0,
        "baseline_metric_rows_complete": not missing_baseline_ids,
    }
    phase0_pass = all(pass_conditions.values())

    mv_ap_contract = {
        "schema": "stream4d_v93_mv_ap_contract_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "formal_metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "primary_local_metric": "MV_AP_window",
        "local_support_policy": local_support_policy,
        "support_definition": v90.get("support_definition", "local_window_gt_projection split by window_index and gt_id"),
        "AP_thresholds_actual": thresholds,
        "AP_threshold_list_hash": v92.get("AP_threshold_list_hash", v91.get("AP_threshold_list_hash", v90.get("AP_threshold_list_hash", ""))),
        "MV_AP_window_name": "MV_AP_window@0.50:0.90",
        "diagnostic_only_metrics": ["native-carrier AP", "local SF50", "GT_best_IoU", "RADIO/DINO same-GT AUC", "carrier density"],
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "created_at": created_at,
    }

    evaluator_rows = [
        {
            "schema_version": "stream4d_v93_phase0_evaluator_source_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "source_name": "v65_sparse_scene_iou_evaluator",
            "source_artifact": _rel(V65_EVALUATOR),
            "source_artifact_exists": V65_EVALUATOR.exists(),
            "source_artifact_sha256": _sha256(V65_EVALUATOR) if V65_EVALUATOR.exists() else "",
            "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "created_at": created_at,
        }
    ]

    boundary_sources = [
        ("v89_phase0_contract", V89_PHASE0 / "summary.json", "historical evaluator import / v65 formula guard"),
        ("v90_phase0_contract", V90_PHASE0 / "summary.json", "corrected local-window support and Stream3D baseline"),
        ("v91_phase0_contract", V91_PHASE0 / "summary.json", "v91 local-window contract"),
        ("v91_phase8_dev_selection", V91_PHASE8 / "summary.json", "best/control dev metric rows"),
        ("v91_final_decision", V91_FINAL / "decision_summary.json", "v91 No-Go boundary"),
        ("v92_phase0_contract", V92_PHASE0 / "summary.json", "v92 contract and baseline relock"),
        ("v92_final_decision", V92_FINAL / "final_decision.json", "v92 No-Go boundary and v93 starting blocker"),
    ]
    artifact_boundary_rows = []
    for name, path, role in boundary_sources:
        artifact_boundary_rows.append(
            {
                "schema_version": "stream4d_v93_phase0_artifact_boundary_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "artifact_name": name,
                "source_artifact": _rel(path),
                "source_artifact_exists": path.exists(),
                "source_artifact_sha256": _sha256(path) if path.exists() else "",
                "role": role,
                "allowed_for_method_selection": False,
                "allowed_for_metric_contract": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "created_at": created_at,
            }
        )

    variant_gate_rows = []
    for gate_name, gate_pass in pass_conditions.items():
        variant_gate_rows.append(
            {
                "schema_version": "stream4d_v93_phase0_variant_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": "ALL",
                "scene_id": "ALL_DEV",
                "split": "dev",
                "window_id": "ALL_DEV_WINDOWS",
                "gate_name": gate_name,
                "gate_pass": bool(gate_pass),
                "gate_value": {
                    "same_frame_collision_count_eq_0": same_frame_collision_count,
                    "missing_mask_raster_count_eq_0": missing_mask_raster_count,
                    "uses_gt_for_prediction_count_eq_0": uses_gt_for_prediction_count,
                    "uses_future_count_eq_0": uses_future_count,
                    "uses_rgbd_pose_mesh_count_eq_0": uses_rgbd_pose_mesh_count,
                    "baseline_metric_rows_complete": ";".join(missing_baseline_ids),
                }.get(gate_name, ""),
                "created_at": created_at,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    failure_repairs = {
        "formal_metric_source_eq_v65": "stop algorithm phases and repair evaluator adapter",
        "local_support_policy_eq_local_window_gt_projection": "rebuild local-window support rows",
        "same_frame_collision_count_eq_0": "run global WTA and repeat contract lock",
        "missing_mask_raster_count_eq_0": "repair materializer/mask raster references before method phases",
        "baseline_metric_rows_complete": "regenerate missing baseline rows from source artifacts; do not infer from logs",
    }
    variant_failure_rows = [
        {
            "schema_version": "stream4d_v93_phase0_variant_failure_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": "ALL",
            "scene_id": "ALL_DEV",
            "split": "dev",
            "window_id": "ALL_DEV_WINDOWS",
            "failure_type": row["gate_name"],
            "gate_value": row["gate_value"],
            "repair_direction": failure_repairs.get(row["gate_name"], "repair source artifact and rerun Phase0"),
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for row in variant_gate_rows
        if not _bool(row.get("gate_pass"))
    ]

    summary = {
        "schema": "stream4d_v93_phase0_contract_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": "PASS_V93_PHASE0_CONTRACT" if phase0_pass else "STOP_V93_PHASE0_CONTRACT_FAILED",
        "phase0_pass": phase0_pass,
        "formal_metric_source": mv_ap_contract["formal_metric_source"],
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "AP_thresholds_actual": thresholds,
        "local_support_policy": local_support_policy,
        "B0_MV_AP_window": _num(b0.get("mean_MV_AP_window")),
        "B0_MV_AP50_window": _num(b0.get("mean_MV_AP50_window")),
        "best_control_variant": phase8.get("best_control_variant", ""),
        "best_control_MV_AP_window": best_control_mv,
        "best_control_MV_AP50_window": best_control_ap50,
        "v91_best_variant": best_id,
        "v91_best_MV_AP_window": _num(phase8.get("best_real_MV_AP_window")),
        "v91_best_MV_AP50_window": _num(phase8.get("best_real_MV_AP50_window")),
        "v91_best_MV_AP25_window": _num(phase8.get("best_real_MV_AP25_window")),
        "S3D_local_window_MV_AP_window": _num(s3d.get("mean_MV_AP_window"), v90.get("Stream3D_S3D_L1_MV_AP_window", "")),
        "S3D_local_window_MV_AP50_window": _num(s3d.get("mean_MV_AP50_window"), v90.get("Stream3D_S3D_L1_MV_AP50_window", "")),
        "same_frame_collision_count": same_frame_collision_count,
        "missing_mask_raster_count": missing_mask_raster_count,
        "uses_gt_for_prediction_count": uses_gt_for_prediction_count,
        "uses_future_count": uses_future_count,
        "uses_rgbd_pose_mesh_count": uses_rgbd_pose_mesh_count,
        "v91_final_decision_label": final91.get("decision_label", ""),
        "v92_final_decision": final92.get("final_decision", ""),
        "v92_primary_blocker": final92.get("primary_blocker", ""),
        "v92_secondary_blocker": final92.get("secondary_blocker", ""),
        "pass_conditions": pass_conditions,
        "row_counts": {
            "baseline_metric_rows": len(baseline_rows),
            "evaluator_source_rows": len(evaluator_rows),
            "artifact_boundary_rows": len(artifact_boundary_rows),
            "variant_gate_rows": len(variant_gate_rows),
            "variant_failure_rows": len(variant_failure_rows),
        },
        "runtime_sec": time.time() - started,
        "created_at": created_at,
    }

    baseline_fields = [
        "schema_version",
        "phase_id",
        "run_id",
        "variant_id",
        "variant_family",
        "source_version",
        "support_policy",
        "metric_source",
        "MV_AP_window",
        "MV_AP50_window",
        "MV_AP25_window",
        "ScoreFreeMatch50_window",
        "ScoreFreeMatch25_window",
        "same_frame_collision_count",
        "missing_mask_raster_count",
        "uses_gt_for_prediction_count",
        "uses_future_count",
        "notes",
        "scene_id",
        "split",
        "window_id",
        "chunk_id",
        "uses_rgbd_pose_mesh_count",
        "source_artifact",
        "source_artifact_sha256",
        "created_at",
    ]
    _write_json(OUT / "summary.json", summary)
    _write_json(OUT / "mv_ap_contract.json", mv_ap_contract)
    _write_csv(OUT / "baseline_metric_rows.csv", baseline_rows, baseline_fields)
    _write_csv(OUT / "evaluator_source_rows.csv", evaluator_rows)
    _write_csv(OUT / "artifact_boundary_rows.csv", artifact_boundary_rows)
    _write_csv(OUT / "variant_gate_rows.csv", variant_gate_rows)
    _write_csv(OUT / "variant_failure_rows.csv", variant_failure_rows)
    outputs = [
        OUT / "summary.json",
        OUT / "mv_ap_contract.json",
        OUT / "baseline_metric_rows.csv",
        OUT / "evaluator_source_rows.csv",
        OUT / "artifact_boundary_rows.csv",
        OUT / "variant_gate_rows.csv",
        OUT / "variant_failure_rows.csv",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
