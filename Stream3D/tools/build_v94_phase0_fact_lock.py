#!/usr/bin/env python3
"""Build Stream4D v94 Phase0 fact/evaluator lock from v93/v91 evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE_ID = "v94_phase0_fact_lock"
RUN_ID = "v94_phase0_fact_lock"
OUT = ROOT / "outputs/audit/v94_phase0_fact_lock"

V65_EVALUATOR = ROOT / "tools/run_v65_scene_multiview_ap.py"
V93_PHASE0 = ROOT / "outputs/audit/v93_phase0_contract"
V93_PHASE1 = ROOT / "outputs/audit/v93_phase1_source_edge_registry"
V93_PHASE2 = ROOT / "outputs/audit/v93_phase2_d4rt_edge_sampling_diagnostic"
V93_PHASE3 = ROOT / "outputs/audit/v93_phase3_region_edge_graph"
V93_PHASE4_EDGE = ROOT / "outputs/audit/v93_phase4_edge_only_materialization"
V93_PHASE5 = ROOT / "outputs/audit/v93_phase5_boundary_affinity_field"
V93_PHASE5B = ROOT / "outputs/audit/v93_phase5b_unknown_background_field"
V93_DA3_READY = ROOT / "outputs/audit/v93_da3_conditional_readiness"
V93_FINAL = ROOT / "outputs/audit/v93_final_decision"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _num(value: Any, default: float | str = "") -> float | str:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _artifact_row(name: str, path: Path) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v94_phase0_artifact_boundary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "artifact_name": name,
        "artifact_path": _rel(path),
        "artifact_exists": path.exists(),
        "artifact_sha256": _sha256(path) if path.is_file() and path.exists() else "",
        "artifact_kind": "file" if path.is_file() else "directory",
        "created_at": _created_at(),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _phase0_baseline_rows(created_at: str) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(V93_PHASE0 / "baseline_metric_rows.csv"):
        out = dict(row)
        out["schema_version"] = "stream4d_v94_phase0_baseline_metric_v1"
        out["phase_id"] = PHASE_ID
        out["run_id"] = RUN_ID
        out["created_at"] = created_at
        out["source_artifact"] = _rel(V93_PHASE0 / "baseline_metric_rows.csv")
        out["source_artifact_sha256"] = _sha256(V93_PHASE0 / "baseline_metric_rows.csv")
        out["notes"] = f"v94 fact lock reuses v93 Phase0 locked row: {row.get('notes', '')}"
        rows.append(out)

    final = _read_json(V93_FINAL / "summary.json")
    best = final.get("best_attempt", {}) if isinstance(final.get("best_attempt"), dict) else {}
    if best:
        rows.append(
            {
                "schema_version": "stream4d_v94_phase0_baseline_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": best.get("best_variant_id", ""),
                "variant_family": "v93_best_attempt",
                "source_version": "v93_final_decision",
                "support_policy": "local_window_gt_projection",
                "metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
                "MV_AP_window": best.get("MV_AP_window", ""),
                "MV_AP50_window": best.get("MV_AP50_window", ""),
                "MV_AP25_window": "",
                "ScoreFreeMatch50_window": "",
                "ScoreFreeMatch25_window": "",
                "same_frame_collision_count": 0,
                "missing_mask_raster_count": 0,
                "uses_gt_for_prediction_count": best.get("uses_gt_for_prediction", 0),
                "uses_future_count": best.get("uses_future", 0),
                "uses_rgbd_pose_mesh_count": 0,
                "notes": f"v93 final best attempt: {best.get('attempt_id', '')}",
                "scene_id": "ALL_DEV",
                "split": "dev",
                "window_id": "ALL_DEV_WINDOWS",
                "chunk_id": "",
                "source_artifact": _rel(V93_FINAL / "summary.json"),
                "source_artifact_sha256": _sha256(V93_FINAL / "summary.json"),
                "created_at": created_at,
            }
        )
    return rows


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()

    v93_phase0 = _read_json(V93_PHASE0 / "summary.json")
    v93_phase1 = _read_json(V93_PHASE1 / "summary.json")
    v93_phase2 = _read_json(V93_PHASE2 / "summary.json")
    v93_phase3 = _read_json(V93_PHASE3 / "summary.json")
    v93_phase4_edge = _read_json(V93_PHASE4_EDGE / "summary.json")
    v93_phase5 = _read_json(V93_PHASE5 / "summary.json")
    v93_phase5b = _read_json(V93_PHASE5B / "summary.json")
    v93_da3 = _read_json(V93_DA3_READY / "summary.json")
    v93_final = _read_json(V93_FINAL / "summary.json")

    evaluator_text = V65_EVALUATOR.read_text(encoding="utf-8") if V65_EVALUATOR.exists() else ""
    formal_metric_source_eq_v65 = (
        V65_EVALUATOR.exists()
        and "SparseSceneIoU" in evaluator_text
        and "_summarize_iou" in evaluator_text
        and _bool(v93_phase0.get("formal_metric_source_eq_v65"))
    )
    local_support_policy = str(v93_phase0.get("local_support_policy", ""))
    ap_thresholds = v93_phase0.get("AP_thresholds_actual", [])

    baseline_rows = _phase0_baseline_rows(created_at)
    method_rows = [row for row in baseline_rows if row.get("variant_family") != "stream3d_local_diagnostic"]
    same_frame_collision_count = max(float(_num(row.get("same_frame_collision_count"), 0.0) or 0.0) for row in method_rows)
    missing_mask_raster_count = max(float(_num(row.get("missing_mask_raster_count"), 0.0) or 0.0) for row in method_rows)
    uses_gt_for_prediction_count = sum(int(float(_num(row.get("uses_gt_for_prediction_count"), 0) or 0)) for row in method_rows)
    uses_future_count = sum(int(float(_num(row.get("uses_future_count"), 0) or 0)) for row in method_rows)
    uses_rgbd_pose_mesh_count = sum(int(float(_num(row.get("uses_rgbd_pose_mesh_count"), 0) or 0)) for row in method_rows)

    required_mv_ap = max(
        float(_num(v93_phase0.get("v91_best_MV_AP_window"), 0.0) or 0.0) + 0.006,
        float(_num(v93_phase0.get("best_control_MV_AP_window"), 0.0) or 0.0) + 0.010,
    )
    required_ap50 = max(
        float(_num(v93_phase0.get("v91_best_MV_AP50_window"), 0.0) or 0.0) + 0.012,
        float(_num(v93_phase0.get("best_control_MV_AP50_window"), 0.0) or 0.0) + 0.015,
    )

    evaluator_contract = {
        "schema": "stream4d_v94_evaluator_contract_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "formal_metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "local_support_policy": local_support_policy,
        "AP_thresholds_actual": ap_thresholds,
        "primary_local_metric": "MV_AP_window",
        "secondary_local_metrics": ["MV_AP50_window", "MV_AP25_window"],
        "diagnostic_only_metrics": ["native carrier AP", "local SF50", "GT-best-IoU", "RADIO/DINO AUC"],
        "created_at": created_at,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }

    pass_conditions = {
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "local_support_policy_eq_local_window_gt_projection": local_support_policy == "local_window_gt_projection",
        "ap_thresholds_match_0p50_to_0p90": ap_thresholds == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        "v93_final_decision_no_go_locked": v93_final.get("decision") == "NO_GO_V93_BOUNDARY_AWARE_AFFINITY_READOUT",
        "same_frame_collision_count_eq_0": same_frame_collision_count == 0.0,
        "missing_mask_raster_count_eq_0": missing_mask_raster_count == 0.0,
        "uses_gt_for_prediction_count_eq_0": uses_gt_for_prediction_count == 0,
        "uses_future_count_eq_0": uses_future_count == 0,
        "uses_rgbd_pose_mesh_count_eq_0": uses_rgbd_pose_mesh_count == 0,
        "v93_source_chain_available": bool(v93_phase1 and v93_phase2 and v93_phase3),
    }
    phase0_pass = all(pass_conditions.values())

    gate_rows = [
        {
            "schema_version": "stream4d_v94_phase0_variant_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": "ALL",
            "scene_id": "ALL_DEV",
            "split": "dev",
            "window_id": "ALL_DEV_WINDOWS",
            "gate_name": key,
            "gate_pass": value,
            "gate_value": {
                "same_frame_collision_count_eq_0": same_frame_collision_count,
                "missing_mask_raster_count_eq_0": missing_mask_raster_count,
                "uses_gt_for_prediction_count_eq_0": uses_gt_for_prediction_count,
                "uses_future_count_eq_0": uses_future_count,
            }.get(key, ""),
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for key, value in pass_conditions.items()
    ]

    failure_rows = [
        {
            "schema_version": "stream4d_v94_phase0_failure_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "failure_id": key,
            "failure_label": "PHASE0_CONTRACT_FAIL",
            "repair_direction": "Repair evaluator/support/baseline contract before any v94 algorithm variant.",
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for key, passed in pass_conditions.items()
        if not passed
    ]

    artifact_rows = [
        _artifact_row("v65_evaluator", V65_EVALUATOR),
        _artifact_row("v93_phase0_contract", V93_PHASE0 / "summary.json"),
        _artifact_row("v93_phase1_source_edge_registry", V93_PHASE1 / "summary.json"),
        _artifact_row("v93_phase2_d4rt_edge_sampling_diagnostic", V93_PHASE2 / "summary.json"),
        _artifact_row("v93_phase3_region_edge_graph", V93_PHASE3 / "summary.json"),
        _artifact_row("v93_phase4_edge_only_materialization", V93_PHASE4_EDGE / "summary.json"),
        _artifact_row("v93_phase5_boundary_affinity_field", V93_PHASE5 / "summary.json"),
        _artifact_row("v93_phase5b_unknown_background_field", V93_PHASE5B / "summary.json"),
        _artifact_row("v93_da3_conditional_readiness", V93_DA3_READY / "summary.json"),
        _artifact_row("v93_final_decision", V93_FINAL / "summary.json"),
    ]

    summary = {
        "schema": "stream4d_v94_phase0_fact_lock_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": "PASS_V94_PHASE0_FACT_LOCK" if phase0_pass else "BLOCK_V94_PHASE0_FACT_LOCK",
        "phase0_pass": phase0_pass,
        "formal_metric_source": evaluator_contract["formal_metric_source"],
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "local_support_policy": local_support_policy,
        "AP_thresholds_actual": ap_thresholds,
        "v91_best_variant": v93_phase0.get("v91_best_variant"),
        "v91_best_MV_AP_window": v93_phase0.get("v91_best_MV_AP_window"),
        "v91_best_MV_AP50_window": v93_phase0.get("v91_best_MV_AP50_window"),
        "best_control_variant": v93_phase0.get("best_control_variant"),
        "best_control_MV_AP_window": v93_phase0.get("best_control_MV_AP_window"),
        "best_control_MV_AP50_window": v93_phase0.get("best_control_MV_AP50_window"),
        "S3D_local_window_MV_AP_window": v93_phase0.get("S3D_local_window_MV_AP_window"),
        "S3D_local_window_MV_AP50_window": v93_phase0.get("S3D_local_window_MV_AP50_window"),
        "v93_final_decision": v93_final.get("decision"),
        "v93_best_attempt": v93_final.get("best_attempt"),
        "v93_phase5_best_real_MV_AP_window": v93_phase5.get("best_real_MV_AP_window"),
        "v93_phase5b_best_real_MV_AP_window": v93_phase5b.get("best_real_MV_AP_window"),
        "v93_da3_conditional_branch_ready": v93_da3.get("da3_conditional_branch_ready"),
        "required_MV_AP_window": required_mv_ap,
        "required_MV_AP50_window": required_ap50,
        "same_frame_collision_count": same_frame_collision_count,
        "missing_mask_raster_count": missing_mask_raster_count,
        "uses_gt_for_prediction_count": uses_gt_for_prediction_count,
        "uses_future_count": uses_future_count,
        "uses_rgbd_pose_mesh_count": uses_rgbd_pose_mesh_count,
        "v93_source_container_count": v93_phase1.get("source_container_count"),
        "v93_object_hypothesis_count": v93_phase1.get("object_hypothesis_count"),
        "v93_region_feature_available_rate": v93_phase3.get("region_feature_available_rate"),
        "v93_projection_jitter_p90": v93_phase2.get("projection_jitter_p90"),
        "v93_mask_membership_flip_rate_median": v93_phase2.get("mask_membership_flip_rate_median"),
        "pass_conditions": pass_conditions,
        "row_counts": {
            "baseline_metric_rows": len(baseline_rows),
            "artifact_boundary_rows": len(artifact_rows),
            "variant_gate_rows": len(gate_rows),
            "variant_failure_rows": len(failure_rows),
        },
        "runtime_sec": time.time() - started,
    }

    _write_json(OUT / "summary.json", summary)
    _write_json(OUT / "evaluator_contract.json", evaluator_contract)
    _write_csv(OUT / "baseline_metric_rows.csv", baseline_rows)
    _write_csv(OUT / "artifact_boundary_rows.csv", artifact_rows)
    _write_csv(OUT / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT / "variant_failure_rows.csv", failure_rows)
    output_files = [
        OUT / "summary.json",
        OUT / "evaluator_contract.json",
        OUT / "baseline_metric_rows.csv",
        OUT / "artifact_boundary_rows.csv",
        OUT / "variant_gate_rows.csv",
        OUT / "variant_failure_rows.csv",
    ]
    _write_json(OUT / "SHA256SUMS.json", {path.name: _sha256(path) for path in output_files})
    return summary


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
