from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_csv, read_json, utc_now, write_csv, write_json


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


def _state_bucket() -> dict[str, list[str]]:
    return {
        "confirmed": [],
        "tentative": [],
        "shared": [],
        "quarantine": [],
        "unknown": [],
    }


def build_v64r2_native_contract(
    *,
    native_summary_path: str | Path = "outputs/audit/v62_native_field/native_field_summary.json",
    component_state_rows_path: str | Path = "outputs/audit/v62_native_field/native_component_state_rows.csv",
    carrier_state_rows_path: str | Path = "outputs/audit/v62_native_field/native_carrier_state_rows.csv",
) -> dict[str, Any]:
    native_summary = _load_dict(native_summary_path)
    component_rows = read_csv(_project(component_state_rows_path)) if _project(component_state_rows_path).exists() else []
    carrier_rows = read_csv(_project(carrier_state_rows_path)) if _project(carrier_state_rows_path).exists() else []
    carrier_by_material = {
        str(row.get("material_node_id") or ""): row for row in carrier_rows if str(row.get("material_node_id") or "")
    }
    object_acc: dict[tuple[str, str], dict[str, Any]] = {}
    material_rows: list[dict[str, Any]] = []
    state_label_complete = True
    for row in component_rows:
        scene = str(row.get("scene") or row.get("scene_id") or "")
        material_id = str(row.get("material_node_id") or "")
        component_id = str(row.get("component_id") or "")
        state = str(row.get("state") or "unknown")
        history_id = str(row.get("predicted_history_id") or "")
        support_ids = _json_list(row.get("support_observation_ids_json") or row.get("supporting_observation_ids_json") or row.get("support_observation_ids"))
        if state not in {"confirmed", "tentative", "shared", "quarantine", "unknown"}:
            state_label_complete = False
        carrier_row = carrier_by_material.get(material_id, {})
        carrier_id = str(carrier_row.get("carrier_id") or "")
        key = (scene, history_id or "unknown_history")
        if key not in object_acc:
            object_acc[key] = {
                "scene_id": scene,
                "history_id": history_id or "unknown_history",
                "semantic_modes": [],
                "confirmed_material_ids": [],
                "tentative_material_ids": [],
                "shared_material_ids": [],
                "quarantine_material_ids": [],
                "unknown_material_ids": [],
                "supporting_mask_ids": [],
                "supporting_frame_ids": [],
                "state_timeline": [],
                "confidence": 1.0,
                "score_policy": "v62_verified_ownership_state",
                "material_count": 0,
            }
        obj = object_acc[key]
        obj["material_count"] += 1
        bucket_name = f"{state}_material_ids" if state in _state_bucket() else "unknown_material_ids"
        obj.setdefault(bucket_name, []).append(material_id)
        obj.setdefault("supporting_observation_ids", [])
        obj["supporting_observation_ids"].extend(support_ids)
        for support_id in support_ids:
            parsed = _parse_mask_observation_id(str(support_id))
            if parsed:
                _obs_scene, frame_id, mask_id = parsed
                obj["supporting_frame_ids"].append(frame_id)
                obj["supporting_mask_ids"].append(mask_id)
        obj["state_timeline"].append({"material_id": material_id, "state": state})
        material_rows.append(
            {
                "scene_id": scene,
                "material_id": material_id,
                "component_id": component_id,
                "carrier_id_if_available": carrier_id,
                "history_id": history_id,
                "state": state,
                "state_confidence": 1.0,
                "support_observation_ids": support_ids,
                "source_evidence_types": ["v62_decircularized_solver_component_state"],
                "frame_ids": [parsed[1] for parsed in (_parse_mask_observation_id(str(obs)) for obs in support_ids) if parsed],
                "uv_tracks_if_available": [],
                "xyz_tracks_if_available": [],
                "uses_gt_for_prediction": str(row.get("uses_gt_for_prediction")).lower() == "true",
                "uses_rgbd_pose_mesh_for_export": str(row.get("uses_rgbd_pose_mesh_for_export")).lower() == "true",
            }
        )
    object_rows = []
    for row in object_acc.values():
        row["supporting_observation_ids"] = sorted(set(row.get("supporting_observation_ids", [])))
        row["supporting_frame_ids"] = sorted(set(row.get("supporting_frame_ids", [])))
        row["supporting_mask_ids"] = sorted(set(row.get("supporting_mask_ids", [])))
        object_rows.append(row)
    material_count = len(material_rows)
    object_count = len(object_rows)
    confirmed_material_count = sum(1 for row in material_rows if row["state"] == "confirmed")
    summary = {
        "phase": "v64r2_native_contract",
        "created_at": utc_now(),
        "input_paths": {
            "native_summary": _rel(native_summary_path),
            "component_state_rows": _rel(component_state_rows_path),
            "carrier_state_rows": _rel(carrier_state_rows_path),
        },
        "object_count": object_count,
        "material_count": material_count,
        "confirmed_material_count": confirmed_material_count,
        "state_label_complete": bool(state_label_complete and material_count > 0),
        "component_level_available": bool(native_summary.get("component_level_field_available") and material_count > 0),
        "carrier_level_available": bool(native_summary.get("carrier_level_field_available")),
        "component_to_carrier_mapping_available": bool(native_summary.get("component_to_carrier_mapping_available")),
        "uv_track_available": False,
        "xyz_track_available": False,
        "support_mask_available": any(bool(row.get("support_observation_ids")) for row in material_rows),
        "semantic_mode_available": False,
        "uses_gt_for_prediction": any(bool(row.get("uses_gt_for_prediction")) for row in material_rows),
        "uses_rgbd_pose_mesh_for_export": any(bool(row.get("uses_rgbd_pose_mesh_for_export")) for row in material_rows),
        "native_field_limitation": native_summary.get("native_field_limitation"),
    }
    gate = {
        "object_count_gt_0": object_count > 0,
        "material_count_gt_0": material_count > 0,
        "state_label_complete": bool(summary["state_label_complete"]),
        "component_level_available": bool(summary["component_level_available"]),
        "uses_gt_for_prediction_false": not bool(summary["uses_gt_for_prediction"]),
    }
    gate["pass"] = bool(all(gate.values()))
    summary["gate"] = gate
    summary["native_contract_status"] = "pass" if gate["pass"] else "fail"
    return {
        "summary": summary,
        "object_field_rows": object_rows,
        "material_state_rows": material_rows,
    }


def write_v64r2_native_contract(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "native_contract_summary.json", payload["summary"])
    write_csv(out / "object_field_rows.csv", payload["object_field_rows"])
    write_csv(out / "material_state_rows.csv", payload["material_state_rows"])


def _json_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str):
        return [value]
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def _parse_mask_observation_id(obs_id: str) -> tuple[str, int, int] | None:
    parts = str(obs_id).split(":")
    if len(parts) == 4 and parts[0] == "m":
        scene, frame, mask = parts[1], parts[2], parts[3]
    elif len(parts) == 3:
        scene, frame, mask = parts
    else:
        return None
    try:
        return scene, int(float(frame)), int(float(mask))
    except ValueError:
        return None
