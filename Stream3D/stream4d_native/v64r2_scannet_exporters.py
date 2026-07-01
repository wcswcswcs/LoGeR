from __future__ import annotations

from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json
from .v53_ap_diagnostic import native_method_export_repair_audit


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


def build_v64r2_ap_contract(
    *,
    native_contract_summary_path: str | Path = "outputs/audit/v64r2_native_contract/native_contract_summary.json",
    v53_native_carrier_summary_path: str | Path = "outputs/audit/v53_native_carrier_materialization/native_carrier_summary.json",
) -> dict[str, Any]:
    native_contract = _load_dict(native_contract_summary_path)
    repair = native_method_export_repair_audit(v53_native_carrier_summary_path=v53_native_carrier_summary_path)
    object_count = native_contract.get("object_count")
    material_count = native_contract.get("material_count")
    component_available = bool(native_contract.get("component_level_available"))
    method_safe_native_ap = bool(repair.get("method_safe_native_ap_export_available"))
    method_safe_support = bool(repair.get("method_safe_native_support_available"))
    rows = [
        {
            "exporter_name": "E0_native_component_field",
            "prediction_count": object_count,
            "object_count": object_count,
            "material_count": material_count,
            "mesh_mask_count": 0,
            "point_mask_count": 0,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "uses_scannet_mesh_for_prediction": False,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": False,
            "export_status": "native_component_field_available" if component_available else "blocked_missing_native_contract",
            "failure_reason": "" if component_available else "v64r2 native contract missing or empty",
        },
        {
            "exporter_name": "E1_method_safe_projection_if_available",
            "prediction_count": 0,
            "object_count": object_count,
            "material_count": material_count,
            "mesh_mask_count": 0,
            "point_mask_count": 0,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "uses_scannet_mesh_for_prediction": False,
            "is_method_result": bool(method_safe_native_ap),
            "is_diagnostic_only": not bool(method_safe_native_ap),
            "forbidden_for_method_table": False,
            "export_status": "ok" if method_safe_native_ap else "blocked_method_safe_materializer",
            "failure_reason": "" if method_safe_native_ap else repair.get("blocked_reason"),
            "repair_result": repair.get("repair_result"),
            "required_future_change": repair.get("required_future_change"),
            "method_safe_native_support_available": method_safe_support,
        },
        {
            "exporter_name": "E2_rgbd_pose_mesh_bridge_diagnostic",
            "prediction_count": None,
            "object_count": None,
            "material_count": None,
            "mesh_mask_count": None,
            "point_mask_count": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": True,
            "uses_scannet_mesh_for_prediction": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "export_status": "available_for_diagnostic_run",
            "failure_reason": "",
        },
        {
            "exporter_name": "E3_eval_aligned_sim3_diagnostic",
            "prediction_count": None,
            "object_count": None,
            "material_count": None,
            "mesh_mask_count": None,
            "point_mask_count": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": True,
            "uses_rgbd_pose_mesh_for_export": True,
            "uses_scannet_mesh_for_prediction": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "export_status": "not_run_eval_aligned_not_needed_for_probe5_smoke",
            "failure_reason": "kept as diagnostic policy row only; not used for method table",
        },
        {
            "exporter_name": "E4_stream3d_same_support_diagnostic",
            "prediction_count": None,
            "object_count": None,
            "material_count": None,
            "mesh_mask_count": None,
            "point_mask_count": None,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": None,
            "uses_scannet_mesh_for_prediction": None,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "export_status": "not_available_in_current_artifacts",
            "failure_reason": "no locked Stream3D-CropFormer same-support adapter artifact found in v64r2 inputs",
        },
    ]
    gate = {
        "policy_rows_complete": all("forbidden_for_method_table" in row for row in rows),
        "diagnostic_bridge_available": any(
            row["exporter_name"] == "E2_rgbd_pose_mesh_bridge_diagnostic"
            and row["export_status"] == "available_for_diagnostic_run"
            for row in rows
        ),
        "method_safe_projection_available": method_safe_native_ap,
        "method_safe_native_support_available": method_safe_support,
    }
    gate["ap_smoke_can_run"] = bool(gate["diagnostic_bridge_available"] or gate["method_safe_projection_available"])
    summary = {
        "phase": "v64r2_ap_contract",
        "created_at": utc_now(),
        "input_paths": {
            "native_contract_summary": _rel(native_contract_summary_path),
            "v53_native_carrier_summary": _rel(v53_native_carrier_summary_path),
        },
        "gate": gate,
        "ap_status_before_probe": "diagnostic_bridge_available_method_safe_blocked"
        if gate["diagnostic_bridge_available"] and not method_safe_native_ap
        else ("method_safe_available" if method_safe_native_ap else "blocked"),
        "native_method_export_repair": repair,
    }
    return {"summary": summary, "exporter_policy_rows": rows}


def write_v64r2_ap_contract(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "ap_export_contract.json", payload["summary"])
    write_csv(out / "exporter_policy_rows.csv", payload["exporter_policy_rows"])
