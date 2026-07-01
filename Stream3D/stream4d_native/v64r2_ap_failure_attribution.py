from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .v47_common import ROOT, parse_float, read_csv, read_json, utc_now, write_csv, write_json


AP_EVAL_SCOPE = "ap_eval_pre_points_no_class"
FULL_SCENE_SCOPE = "full_scene_all_gt_instances"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _load_dict(path: str | Path) -> dict[str, Any]:
    path_obj = _project(path)
    if not path_obj.exists():
        return {}
    payload = read_json(path_obj)
    return payload if isinstance(payload, dict) else {}


def _category_for_gt(row: dict[str, str]) -> str:
    best_iou = parse_float(row.get("best_iou"), 0.0)
    duplicate_count = parse_float(row.get("duplicate_predictions_at_0p25"), 0.0)
    failure_reason = str(row.get("failure_reason") or "")
    if failure_reason:
        if "missing" in failure_reason or "mismatch" in failure_reason or "format" in failure_reason:
            return "A_evaluator_format"
        return "A_coverage"
    if duplicate_count > 1:
        return "A_duplicate"
    if best_iou <= 0.0:
        return "A_materialization"
    if best_iou < 0.25:
        return "A_coverage"
    if best_iou < 0.50:
        return "A_scoring"
    return "covered"


def _gt_failure_rows(
    rows: list[dict[str, str]],
    *,
    failure_scope: str,
    prediction_state_mix: str,
    materialization_policy: str,
) -> list[dict[str, Any]]:
    failure_rows: list[dict[str, Any]] = []
    for row in rows:
        category = _category_for_gt(row)
        if category == "covered":
            continue
        failure_rows.append(
            {
                "failure_category": category,
                "failure_scope": failure_scope,
                "iou_scope": row.get("iou_scope") or failure_scope,
                "scene_id": row.get("scene_id"),
                "gt_instance_id": row.get("gt_instance_id"),
                "best_prediction_id": row.get("best_prediction_id"),
                "best_iou": row.get("best_iou"),
                "prediction_state_mix": prediction_state_mix,
                "confirmed_ratio": None,
                "tentative_ratio": None,
                "shared_ratio": None,
                "quarantine_ratio": None,
                "score": row.get("best_prediction_score"),
                "is_duplicate": parse_float(row.get("duplicate_predictions_at_0p25"), 0.0) > 1.0,
                "is_overmerge": None,
                "materialization_policy": materialization_policy,
                "evidence": row.get("failure_reason"),
            }
        )
    return failure_rows


def build_v64r2_ap_failure_attribution(
    *,
    ap_summary_path: str | Path = "outputs/audit/v64r2_scannet_ap_probe5/ap_smoke_summary.json",
    ap_metric_rows_path: str | Path = "outputs/audit/v64r2_scannet_ap_probe5/ap_metric_rows.csv",
    per_gt_iou_rows_path: str | Path = "outputs/audit/v64r2_scannet_ap_probe5/per_gt_iou_rows.csv",
    full_scene_coverage_rows_path: str | Path = "outputs/audit/v64r2_scannet_ap_probe5/full_scene_coverage_rows.csv",
) -> dict[str, Any]:
    ap_summary = _load_dict(ap_summary_path)
    ap_rows = read_csv(_project(ap_metric_rows_path)) if _project(ap_metric_rows_path).exists() else []
    per_gt_rows = read_csv(_project(per_gt_iou_rows_path)) if _project(per_gt_iou_rows_path).exists() else []
    full_scene_rows = (
        read_csv(_project(full_scene_coverage_rows_path)) if _project(full_scene_coverage_rows_path).exists() else []
    )
    ap_eval_failure_rows = _gt_failure_rows(
        per_gt_rows,
        failure_scope=AP_EVAL_SCOPE,
        prediction_state_mix="rgbd_pose_mesh_bridge_diagnostic_no_native_state_trace",
        materialization_policy="rgbd_pose_mesh_bridge_wta_diagnostic",
    )
    full_scene_failure_rows = _gt_failure_rows(
        full_scene_rows,
        failure_scope=FULL_SCENE_SCOPE,
        prediction_state_mix="rgbd_pose_mesh_bridge_diagnostic_full_scene_coverage_profile",
        materialization_policy="rgbd_pose_mesh_bridge_wta_diagnostic_full_scene_profile",
    )
    failure_rows: list[dict[str, Any]] = list(ap_eval_failure_rows)
    method_rows = [
        row
        for row in ap_rows
        if str(row.get("forbidden_for_method_table")).lower() == "false"
        and str(row.get("uses_rgbd_pose_mesh_for_export")).lower() == "false"
    ]
    method_blocked = [row for row in method_rows if not str(row.get("AP") or "").strip()]
    if method_blocked:
        failure_rows.append(
            {
                "failure_category": "A_materialization",
                "failure_scope": "method_safe_gate",
                "iou_scope": "",
                "scene_id": "ALL",
                "gt_instance_id": "",
                "best_prediction_id": "",
                "best_iou": None,
                "prediction_state_mix": "v62_component_level_ownership_field",
                "confirmed_ratio": None,
                "tentative_ratio": None,
                "shared_ratio": None,
                "quarantine_ratio": None,
                "score": None,
                "is_duplicate": False,
                "is_overmerge": None,
                "materialization_policy": "method_safe_native_component_to_scannet_mask",
                "evidence": "method-safe AP rows lack native component/carrier to ScanNet point/mesh materializer",
            }
        )
    failure_rows.extend(full_scene_failure_rows)
    ap_eval_counts = Counter(str(row.get("failure_category")) for row in ap_eval_failure_rows)
    full_scene_counts = Counter(str(row.get("failure_category")) for row in full_scene_failure_rows)
    all_counts = Counter(str(row.get("failure_category")) for row in failure_rows)
    failed_gt_count = sum(1 for row in per_gt_rows if _category_for_gt(row) != "covered")
    full_scene_failed_gt_count = sum(1 for row in full_scene_rows if _category_for_gt(row) != "covered")
    attribution_coverage = (
        1.0
        if failed_gt_count == 0
        else min(1.0, len([r for r in ap_eval_failure_rows if r.get("gt_instance_id")]) / failed_gt_count)
    )
    top_failure_category = ap_eval_counts.most_common(1)[0][0] if ap_eval_counts else "none"
    top_full_scene_failure_category = full_scene_counts.most_common(1)[0][0] if full_scene_counts else "none"
    summary = {
        "phase": "v64r2_ap_failure_attribution",
        "created_at": utc_now(),
        "input_paths": {
            "ap_summary": _rel(ap_summary_path),
            "ap_metric_rows": _rel(ap_metric_rows_path),
            "per_gt_iou_rows": _rel(per_gt_iou_rows_path),
            "full_scene_coverage_rows": _rel(full_scene_coverage_rows_path),
        },
        "scannet_ap_status": ap_summary.get("scannet_ap_status"),
        "method_safe_AP_available": ap_summary.get("method_safe_AP_available"),
        "diagnostic_AP_available": ap_summary.get("diagnostic_AP_available"),
        "ap_eval_scope": AP_EVAL_SCOPE,
        "full_scene_coverage_scope": FULL_SCENE_SCOPE,
        "failed_gt_count": failed_gt_count,
        "full_scene_failed_gt_count": full_scene_failed_gt_count,
        "failure_row_count": len(failure_rows),
        "failure_category_counts": dict(ap_eval_counts),
        "full_scene_failure_category_counts": dict(full_scene_counts),
        "all_failure_category_counts": dict(all_counts),
        "top_failure_category": top_failure_category,
        "top_full_scene_failure_category": top_full_scene_failure_category,
        "method_safe_materialization_blocked": bool(method_blocked),
        "method_safe_blocker_count": len(method_blocked),
        "attribution_coverage": attribution_coverage,
        "gate": {
            "attribution_coverage_ge_0_90": attribution_coverage >= 0.90,
            "top_failure_category_identified": top_failure_category != "none",
        },
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return {"summary": summary, "failure_rows": failure_rows}


def write_v64r2_ap_failure_attribution(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "failure_summary.json", payload["summary"])
    write_csv(out / "failure_rows.csv", payload["failure_rows"])
