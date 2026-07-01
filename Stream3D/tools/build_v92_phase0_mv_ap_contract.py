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


OUT = ROOT / "outputs/audit/v92_phase0_mv_ap_contract"
V65_EVALUATOR = ROOT / "tools/run_v65_scene_multiview_ap.py"
V89_PHASE0 = ROOT / "outputs/audit/v89_phase0_mv_ap_contract"
V90_PHASE0 = ROOT / "outputs/audit/v90_phase0_mv_ap_contract"
V91_PHASE0 = ROOT / "outputs/audit/v91_phase0_mv_ap_contract"
V91_PHASE8 = ROOT / "outputs/audit/v91_phase8_dev_selection"
V91_FINAL = ROOT / "outputs/audit/v91_final_decision"


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


def _all_rows() -> list[dict[str, str]]:
    return _read_csv(V91_PHASE8 / "all_variant_metric_rows.csv")


def _row(rows: list[dict[str, str]], variant_id: str) -> dict[str, str]:
    return next((row for row in rows if row.get("variant_id") == variant_id), {})


def _metric_row(variant_id: str, family: str, source_artifact: Path, row: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v92_phase0_baseline_metric_v1",
        "phase_id": "v92_phase0_mv_ap_contract",
        "run_id": "v92_phase0_lock",
        "variant_id": variant_id,
        "variant_family": family,
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
        "uses_gt_for_prediction": bool(_bool(row.get("uses_gt_for_prediction"))),
        "uses_future": bool(_bool(row.get("uses_future"))),
        "uses_rgbd_pose_mesh": bool(_bool(row.get("uses_rgbd_pose_mesh"))),
        "source_artifact": _rel(source_artifact),
        "source_artifact_sha256": _sha256(source_artifact) if source_artifact.exists() else "",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "notes": notes,
    }


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    v89 = _read_json(V89_PHASE0 / "summary.json")
    v90 = _read_json(V90_PHASE0 / "summary.json")
    v91 = _read_json(V91_PHASE0 / "summary.json")
    phase8 = _read_json(V91_PHASE8 / "summary.json")
    final = _read_json(V91_FINAL / "decision_summary.json")
    rows = _all_rows()
    b0 = _row(rows, "B0_local_only")
    c0 = _row(rows, "C0_semantic_only_control")
    p3 = _row(rows, "P3_C0_area_semantic_hybrid_score")
    s3d = _row(rows, "S3D_L1_local_merged_masks")
    best_id = str(phase8.get("best_real_variant", final.get("best_variant", "")))
    best = _row(rows, best_id)
    if not best and best_id:
        best = {
            "mean_MV_AP_window": final.get("best_real_MV_AP_window", ""),
            "mean_MV_AP50_window": final.get("best_real_MV_AP50_window", ""),
            "mean_MV_AP25_window": final.get("best_real_MV_AP25_window", ""),
            "mean_score_free_Match50_window": final.get("best_score_free_Match50_window", ""),
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
        _metric_row("S3D_L1_local_merged_masks", "stream3d_local_diagnostic", V91_PHASE8 / "all_variant_metric_rows.csv", s3d, "Corrected Stream3D local-window diagnostic baseline; not a v92 method row."),
    ]
    baseline_rows = [row for row in baseline_rows if row["variant_id"]]

    evaluator_text = V65_EVALUATOR.read_text(encoding="utf-8") if V65_EVALUATOR.exists() else ""
    formal_metric_source_eq_v65 = bool(
        V65_EVALUATOR.exists()
        and "SparseSceneIoU" in evaluator_text
        and "_summarize_iou" in evaluator_text
        and _bool(v90.get("formal_metric_source_eq_v65"))
        and _bool(v91.get("formal_metric_source_eq_v65"))
    )
    local_support_policy = str(v91.get("local_support_policy", v90.get("support_policy_local_window", "")))
    same_frame_collision_count = max(
        float(_num(best.get("same_frame_collision_count"), 0.0) or 0.0),
        float(_num(v91.get("same_frame_collision_count_dev_contract"), 0.0) or 0.0),
    )
    missing_mask_raster_count = max(
        float(_num(best.get("missing_mask_raster_count"), 0.0) or 0.0),
        float(_num(v91.get("missing_mask_raster_count"), 0.0) or 0.0),
    )
    uses_gt_for_prediction_count = sum(1 for row in baseline_rows if _bool(row.get("uses_gt_for_prediction")))
    uses_future_count = sum(1 for row in baseline_rows if _bool(row.get("uses_future")))
    method_rows = [row for row in baseline_rows if row.get("variant_family") != "stream3d_local_diagnostic"]
    method_uses_gt_count = sum(1 for row in method_rows if _bool(row.get("uses_gt_for_prediction")))
    method_uses_future_count = sum(1 for row in method_rows if _bool(row.get("uses_future")))

    pass_conditions = {
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "local_support_policy_eq_local_window_gt_projection": local_support_policy == "local_window_gt_projection",
        "missing_mask_raster_count_eq_0": missing_mask_raster_count == 0.0,
        "same_frame_collision_count_eq_0_after_WTA": same_frame_collision_count == 0.0,
        "method_uses_gt_for_prediction_count_eq_0": method_uses_gt_count == 0,
        "method_uses_future_count_eq_0": method_uses_future_count == 0,
    }
    phase0_pass = all(pass_conditions.values())

    thresholds = list(v91.get("AP_thresholds_actual", v90.get("AP_thresholds_actual", [])))
    mv_ap_contract = {
        "schema": "stream4d_v92_mv_ap_contract_v1",
        "phase": "v92_phase0_mv_ap_contract",
        "formal_metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "primary_local_metric": "MV_AP_window",
        "primary_scene_metric": "MV_AP_scene",
        "local_support_policy": local_support_policy,
        "support_definition": v90.get("support_definition", "local_window_gt_projection split by window_index and gt_id"),
        "AP_thresholds_actual": thresholds,
        "AP_threshold_list_hash": v91.get("AP_threshold_list_hash", v90.get("AP_threshold_list_hash", "")),
        "score_free_metrics_diagnostic_only": ["ScoreFreeMatch50_window", "ScoreFreeMatch25_window"],
        "forbidden_success_metrics": [
            "native-carrier AP",
            "local SF50",
            "GT_best_IoU",
            "carrier density",
            "RADIO/DINO same-GT AUC",
        ],
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }

    evaluator_rows = [
        {
            "schema_version": "stream4d_v92_phase0_evaluator_source_v1",
            "phase_id": "v92_phase0_mv_ap_contract",
            "run_id": "v92_phase0_lock",
            "source_name": "v65_sparse_scene_iou_evaluator",
            "source_artifact": _rel(V65_EVALUATOR),
            "source_artifact_exists": V65_EVALUATOR.exists(),
            "source_artifact_sha256": _sha256(V65_EVALUATOR) if V65_EVALUATOR.exists() else "",
            "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    ]
    boundary_sources = [
        ("v89_phase0_contract", V89_PHASE0 / "summary.json", "historical evaluator import / v65 formula guard"),
        ("v90_phase0_contract", V90_PHASE0 / "summary.json", "corrected local-window support and Stream3D baseline"),
        ("v91_phase0_contract", V91_PHASE0 / "summary.json", "v91 local-window contract"),
        ("v91_phase8_dev_selection", V91_PHASE8 / "summary.json", "best/control dev metric rows"),
        ("v91_final_decision", V91_FINAL / "decision_summary.json", "latest formal No-Go boundary"),
    ]
    artifact_boundary_rows = []
    for name, path, role in boundary_sources:
        artifact_boundary_rows.append(
            {
                "schema_version": "stream4d_v92_phase0_artifact_boundary_v1",
                "phase_id": "v92_phase0_mv_ap_contract",
                "run_id": "v92_phase0_lock",
                "artifact_name": name,
                "source_artifact": _rel(path),
                "source_artifact_exists": path.exists(),
                "source_artifact_sha256": _sha256(path) if path.exists() else "",
                "role": role,
                "allowed_for_method_selection": False,
                "allowed_for_metric_contract": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )

    variant_config_rows = [
        {
            "schema_version": "stream4d_v92_phase0_variant_config_v1",
            "phase_id": "v92_phase0_mv_ap_contract",
            "run_id": "v92_phase0_lock",
            "variant_id": row["variant_id"],
            "scene_id": "ALL_DEV",
            "split": row["split"],
            "window_id": row["window_id"],
            "chunk_id": "",
            "config_role": row["variant_family"],
            "source_artifact": row["source_artifact"],
            "source_artifact_sha256": row["source_artifact_sha256"],
            "created_at": row["created_at"],
            "uses_gt_for_prediction": row["uses_gt_for_prediction"],
            "uses_future": row["uses_future"],
            "uses_rgbd_pose_mesh": row["uses_rgbd_pose_mesh"],
        }
        for row in baseline_rows
    ]
    variant_gate_rows = [
        {
            "schema_version": "stream4d_v92_phase0_variant_gate_v1",
            "phase_id": "v92_phase0_mv_ap_contract",
            "run_id": "v92_phase0_lock",
            "variant_id": row["variant_id"],
            "scene_id": row["scene_id"],
            "split": row["split"],
            "window_id": row["window_id"],
            "chunk_id": "",
            "gate_name": "phase0_contract_row_safe",
            "gate_pass": (not _bool(row["uses_gt_for_prediction"])) and (not _bool(row["uses_future"])) and float(row.get("same_frame_collision_count") or 0.0) == 0.0 and float(row.get("missing_mask_raster_count") or 0.0) == 0.0,
            "source_artifact": row["source_artifact"],
            "source_artifact_sha256": row["source_artifact_sha256"],
            "created_at": row["created_at"],
            "uses_gt_for_prediction": row["uses_gt_for_prediction"],
            "uses_future": row["uses_future"],
            "uses_rgbd_pose_mesh": row["uses_rgbd_pose_mesh"],
        }
        for row in baseline_rows
    ]
    variant_failure_rows = [
        {**row, "failure_type": "phase0_contract_row_not_safe"}
        for row in variant_gate_rows
        if not _bool(row.get("gate_pass"))
    ]
    casebook_rows = [
        {
            "schema_version": "stream4d_v92_phase0_casebook_v1",
            "phase_id": "v92_phase0_mv_ap_contract",
            "run_id": "v92_phase0_lock",
            "case_id": "v91_best_no_go_boundary",
            "variant_id": best_id,
            "scene_id": "ALL_DEV",
            "split": "dev",
            "window_id": "ALL_DEV_WINDOWS",
            "chunk_id": "",
            "case_type": "contract_boundary",
            "evidence": "v91 best misses AP50 control-margin gate; no holdout/local2history claim allowed.",
            "source_artifact": _rel(V91_FINAL / "decision_summary.json"),
            "source_artifact_sha256": _sha256(V91_FINAL / "decision_summary.json") if (V91_FINAL / "decision_summary.json").exists() else "",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "uses_gt_for_prediction": False,
            "uses_future": False,
            "uses_rgbd_pose_mesh": False,
        }
    ]

    summary = {
        "schema": "stream4d_v92_phase0_mv_ap_contract_v1",
        "phase": "v92_phase0_mv_ap_contract",
        "phase0_pass": phase0_pass,
        "decision": "PASS_V92_PHASE0_CONTRACT" if phase0_pass else "STOP_V92_PHASE0_CONTRACT_FAILED",
        "formal_metric_source": mv_ap_contract["formal_metric_source"],
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "AP_thresholds_actual": thresholds,
        "local_support_policy": local_support_policy,
        "B0_MV_AP_window": _num(b0.get("mean_MV_AP_window")),
        "B0_MV_AP50_window": _num(b0.get("mean_MV_AP50_window")),
        "C0_MV_AP_window": _num(c0.get("mean_MV_AP_window")),
        "C0_MV_AP50_window": _num(c0.get("mean_MV_AP50_window")),
        "best_control_variant": phase8.get("best_control_variant", ""),
        "best_control_MV_AP_window": _num(phase8.get("best_control_MV_AP_window")),
        "best_control_MV_AP50_window": _num(phase8.get("best_control_MV_AP50_window")),
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
        "method_uses_gt_for_prediction_count": method_uses_gt_count,
        "method_uses_future_count": method_uses_future_count,
        "v91_final_decision_label": final.get("decision_label", ""),
        "v91_phase10_decision": final.get("phase10_decision", ""),
        "pass_conditions": pass_conditions,
        "row_counts": {
            "baseline_metric_rows": len(baseline_rows),
            "evaluator_source_rows": len(evaluator_rows),
            "artifact_boundary_rows": len(artifact_boundary_rows),
            "variant_config_rows": len(variant_config_rows),
            "variant_metric_rows": len(baseline_rows),
            "variant_gate_rows": len(variant_gate_rows),
            "variant_failure_rows": len(variant_failure_rows),
            "casebook_rows": len(casebook_rows),
        },
        "runtime_sec": time.time() - started,
    }

    baseline_fields = [
        "schema_version",
        "phase_id",
        "run_id",
        "variant_id",
        "variant_family",
        "scene_id",
        "split",
        "window_id",
        "chunk_id",
        "MV_AP_window",
        "MV_AP50_window",
        "MV_AP25_window",
        "ScoreFreeMatch50_window",
        "ScoreFreeMatch25_window",
        "same_frame_collision_count",
        "missing_mask_raster_count",
        "uses_gt_for_prediction",
        "uses_future",
        "uses_rgbd_pose_mesh",
        "source_artifact",
        "source_artifact_sha256",
        "created_at",
        "notes",
    ]
    _write_json(OUT / "mv_ap_contract.json", mv_ap_contract)
    _write_json(OUT / "summary.json", summary)
    _write_csv(OUT / "baseline_metric_rows.csv", baseline_rows, baseline_fields)
    _write_csv(OUT / "variant_metric_rows.csv", baseline_rows, baseline_fields)
    _write_csv(OUT / "evaluator_source_rows.csv", evaluator_rows)
    _write_csv(OUT / "artifact_boundary_rows.csv", artifact_boundary_rows)
    _write_csv(OUT / "variant_config_rows.csv", variant_config_rows)
    _write_csv(OUT / "variant_gate_rows.csv", variant_gate_rows)
    _write_csv(OUT / "variant_failure_rows.csv", variant_failure_rows)
    _write_csv(OUT / "casebook_rows.csv", casebook_rows)
    outputs = [
        OUT / "summary.json",
        OUT / "mv_ap_contract.json",
        OUT / "baseline_metric_rows.csv",
        OUT / "variant_metric_rows.csv",
        OUT / "evaluator_source_rows.csv",
        OUT / "artifact_boundary_rows.csv",
        OUT / "variant_config_rows.csv",
        OUT / "variant_gate_rows.csv",
        OUT / "variant_failure_rows.csv",
        OUT / "casebook_rows.csv",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
