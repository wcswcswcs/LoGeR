from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .soma_inference_policy import normalize_reportability
from .v47_common import ROOT, parse_bool, parse_float, utc_now, write_csv, write_json


DEFAULT_OBJECT_ROWS = "outputs/audit/v64r2_native_contract/object_field_rows.csv"
DEFAULT_MATERIAL_ROWS = "outputs/audit/v64r2_native_contract/material_state_rows.csv"
DEFAULT_CARRIER_ROWS = "outputs/audit/v53_native_carrier_materialization/objectlet_native_carrier_rows.csv"


@dataclass(frozen=True)
class V65SOMAObjectBankConfig:
    object_rows_path: str | Path = DEFAULT_OBJECT_ROWS
    material_rows_path: str | Path = DEFAULT_MATERIAL_ROWS
    carrier_rows_path: str | Path = DEFAULT_CARRIER_ROWS
    output_root: str | Path = "outputs/audit/v65_soma_object_bank"
    allow_unverified_component_join: bool = False


def build_v65_soma_object_bank(config: V65SOMAObjectBankConfig | None = None) -> dict[str, Any]:
    cfg = config or V65SOMAObjectBankConfig()
    object_rows = _read_csv_large(_project(cfg.object_rows_path))
    material_rows = _read_csv_large(_project(cfg.material_rows_path))
    carrier_rows = _read_csv_large(_project(cfg.carrier_rows_path))

    materials_by_history: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    material_component_keys: set[tuple[str, str]] = set()
    material_carrier_keys: set[tuple[str, str]] = set()
    for row in material_rows:
        scene = _scene(row)
        history_id = str(row.get("history_id", ""))
        materials_by_history[(scene, history_id)].append(row)
        component_id = str(row.get("component_id", "")).strip()
        carrier_id = str(row.get("carrier_id_if_available", "")).strip()
        if component_id:
            material_component_keys.add((scene, component_id))
        if carrier_id:
            material_carrier_keys.add((scene, carrier_id))

    carrier_index = _build_carrier_index(carrier_rows)
    carrier_component_keys = set(carrier_index["by_component"].keys())
    carrier_id_keys = set(carrier_index["by_carrier_id"].keys())
    carrier_global_keys = set(carrier_index["by_carrier_global_id"].keys())

    object_bank_rows: list[dict[str, Any]] = []
    object_material_rows: list[dict[str, Any]] = []
    object_support_rows: list[dict[str, Any]] = []

    for object_row in object_rows:
        scene = _scene(object_row)
        history_id = str(object_row.get("history_id", ""))
        object_id = _object_id_from_history(history_id)
        material_ids_by_state = _material_ids_by_state(object_row)
        object_materials = materials_by_history.get((scene, history_id), [])
        declared_mask_support_rows = _object_declared_support_rows(scene, history_id, object_id, object_row)
        material_support_rows: list[dict[str, Any]] = []
        native_join_rows: list[dict[str, Any]] = []
        state_counts: Counter[str] = Counter()

        material_rows_by_id = {str(row.get("material_id", "")): row for row in object_materials}
        for state, material_ids in material_ids_by_state.items():
            state_counts[state] += len(material_ids)
            for material_id in material_ids:
                material_row = material_rows_by_id.get(material_id, {})
                if not material_row:
                    material_row = {
                        "scene_id": scene,
                        "history_id": history_id,
                        "material_id": material_id,
                        "state": state,
                    }
                component_id = str(material_row.get("component_id", "")).strip()
                carrier_id = str(material_row.get("carrier_id_if_available", "")).strip()
                object_material_rows.append(
                    {
                        "scene_id": scene,
                        "object_id": object_id,
                        "history_id": history_id,
                        "material_id": material_id,
                        "component_id": component_id,
                        "carrier_id_if_available": carrier_id,
                        "state": str(material_row.get("state") or state),
                        "state_confidence": material_row.get("state_confidence", ""),
                        "uses_gt_for_prediction": False,
                        "uses_rgbd_pose_mesh_for_export": False,
                        "is_scannet_ap_export": False,
                    }
                )
                material_support_rows.extend(
                    _material_declared_support_rows(scene, history_id, object_id, material_row)
                )
                native_join_rows.extend(
                    _native_carrier_support_rows(
                        scene,
                        history_id,
                        object_id,
                        material_row,
                        carrier_index,
                        allow_unverified_component_join=cfg.allow_unverified_component_join,
                    )
                )

        support_rows_for_object = declared_mask_support_rows + material_support_rows + _dedupe_support_rows(native_join_rows)
        object_support_rows.extend(support_rows_for_object)
        support_kind_counts = Counter(str(row.get("support_kind", "")) for row in support_rows_for_object)
        view_support_count = sum(1 for row in support_rows_for_object if _row_has_view_mask_support(row))
        point_support_count = sum(1 for row in support_rows_for_object if _row_has_point_or_carrier_support(row))
        joined_native_support_count = _native_support_count(support_kind_counts)
        object_bank_rows.append(
            normalize_reportability(
                {
                    "scene_id": scene,
                    "object_id": object_id,
                    "history_id": history_id,
                    "semantic_modes": _json_list(object_row.get("semantic_modes")),
                    "confidence": parse_float(object_row.get("confidence"), 0.0),
                    "score_policy": object_row.get("score_policy", ""),
                    "material_count_declared": _int_from_text(object_row.get("material_count")),
                    "material_count_expanded": sum(len(values) for values in material_ids_by_state.values()),
                    "confirmed_material_count": state_counts.get("confirmed", 0),
                    "tentative_material_count": state_counts.get("tentative", 0),
                    "shared_material_count": state_counts.get("shared", 0),
                    "quarantine_material_count": state_counts.get("quarantine", 0),
                    "unknown_material_count": state_counts.get("unknown", 0),
                    "supporting_mask_count_declared": len(_json_list(object_row.get("supporting_mask_ids"))),
                    "supporting_frame_count_declared": len(_json_list(object_row.get("supporting_frame_ids"))),
                    "object_support_row_count": len(support_rows_for_object),
                    "view_mask_support_row_count": view_support_count,
                    "native_point_or_carrier_support_row_count": point_support_count,
                    "joined_native_carrier_support_row_count": joined_native_support_count,
                    "support_kind_counts": dict(sorted(support_kind_counts.items())),
                    "has_view_mask_support": view_support_count > 0,
                    "has_native_point_or_carrier_support": point_support_count > 0,
                    "native_support_join_available": joined_native_support_count > 0,
                    "method_safe_inference_artifact": True,
                    "is_method_result": True,
                    "method_result_type": "soma_object_bank_not_scannet_ap",
                    "is_scannet_ap_export": False,
                    "uses_gt_for_prediction": False,
                    "uses_gt_geometry_for_inference": False,
                    "uses_rgbd_pose_mesh_for_export": False,
                    "evaluation_adapter_required_for_scannet_ap": True,
                    "forbidden_for_method_table": False,
                    "is_diagnostic_only": False,
                },
                context=f"v65 SOMA object bank row {history_id}",
            )
        )

    summary = _build_summary(
        cfg=cfg,
        object_bank_rows=object_bank_rows,
        object_material_rows=object_material_rows,
        object_support_rows=object_support_rows,
        object_rows=object_rows,
        material_rows=material_rows,
        carrier_rows=carrier_rows,
        material_component_keys=material_component_keys,
        material_carrier_keys=material_carrier_keys,
        carrier_component_keys=carrier_component_keys,
        carrier_id_keys=carrier_id_keys,
        carrier_global_keys=carrier_global_keys,
    )
    return {
        "summary": summary,
        "object_bank_rows": object_bank_rows,
        "object_material_rows": object_material_rows,
        "object_support_rows": object_support_rows,
    }


def write_v65_soma_object_bank(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    paths = {
        "summary": root / "soma_object_bank_summary.json",
        "object_bank_rows": root / "soma_object_bank_rows.csv",
        "object_material_rows": root / "soma_object_material_rows.csv",
        "object_support_rows": root / "soma_object_support_rows.csv",
    }
    write_json(paths["summary"], result["summary"])
    write_csv(paths["object_bank_rows"], result["object_bank_rows"])
    write_csv(paths["object_material_rows"], result["object_material_rows"])
    write_csv(paths["object_support_rows"], result["object_support_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def _build_summary(
    *,
    cfg: V65SOMAObjectBankConfig,
    object_bank_rows: list[dict[str, Any]],
    object_material_rows: list[dict[str, Any]],
    object_support_rows: list[dict[str, Any]],
    object_rows: list[dict[str, str]],
    material_rows: list[dict[str, str]],
    carrier_rows: list[dict[str, str]],
    material_component_keys: set[tuple[str, str]],
    material_carrier_keys: set[tuple[str, str]],
    carrier_component_keys: set[tuple[str, str]],
    carrier_id_keys: set[tuple[str, str]],
    carrier_global_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    object_scene_counts = Counter(str(row.get("scene_id", "")) for row in object_bank_rows)
    material_scene_counts = Counter(str(row.get("scene_id", "")) for row in object_material_rows)
    support_scene_counts = Counter(str(row.get("scene_id", "")) for row in object_support_rows)
    support_kind_counts = Counter(str(row.get("support_kind", "")) for row in object_support_rows)
    objects_with_support = sum(1 for row in object_bank_rows if int(row.get("object_support_row_count") or 0) > 0)
    objects_with_view_mask_support = sum(1 for row in object_bank_rows if parse_bool(row.get("has_view_mask_support")))
    objects_with_native_support = sum(1 for row in object_bank_rows if parse_bool(row.get("has_native_point_or_carrier_support")))
    native_join_rows = _native_support_count(support_kind_counts)
    material_component_matches = material_component_keys & carrier_component_keys
    material_carrier_matches = (material_carrier_keys & carrier_id_keys) | (material_carrier_keys & carrier_global_keys)

    blockers: list[str] = []
    if not object_bank_rows:
        blockers.append("object_bank_empty")
    if not object_material_rows:
        blockers.append("object_material_assignment_empty")
    if not object_support_rows:
        blockers.append("object_to_view_mask_or_point_support_missing")
    if native_join_rows == 0:
        blockers.append("verified_object_to_native_carrier_mapping_missing")

    summary = normalize_reportability(
        {
            "phase": "v65_soma_object_bank",
            "created_at": utc_now(),
            "soma_output_contract": "method-safe object bank: object/history rows with material assignments plus view-mask/native-carrier support rows when present",
            "object_count": len(object_bank_rows),
            "raw_object_row_count": len(object_rows),
            "material_assignment_count": len(object_material_rows),
            "raw_material_row_count": len(material_rows),
            "raw_carrier_row_count": len(carrier_rows),
            "object_support_row_count": len(object_support_rows),
            "native_carrier_support_row_count": native_join_rows,
            "objects_with_any_support_count": objects_with_support,
            "objects_with_view_mask_support_count": objects_with_view_mask_support,
            "objects_with_native_point_or_carrier_support_count": objects_with_native_support,
            "object_support_coverage_ratio": _ratio(objects_with_support, len(object_bank_rows)),
            "object_view_mask_support_coverage_ratio": _ratio(objects_with_view_mask_support, len(object_bank_rows)),
            "object_native_support_coverage_ratio": _ratio(objects_with_native_support, len(object_bank_rows)),
            "support_kind_counts": dict(sorted(support_kind_counts.items())),
            "object_scene_counts": dict(sorted(object_scene_counts.items())),
            "material_scene_counts": dict(sorted(material_scene_counts.items())),
            "support_scene_counts": dict(sorted(support_scene_counts.items())),
            "material_component_key_count": len(material_component_keys),
            "carrier_component_key_count": len(carrier_component_keys),
            "material_component_keys_with_carrier_count": len(material_component_matches),
            "material_carrier_key_count": len(material_carrier_keys),
            "carrier_id_key_count": len(carrier_id_keys),
            "carrier_global_key_count": len(carrier_global_keys),
            "material_carrier_keys_with_carrier_count": len(material_carrier_matches),
            "component_join_policy": "experimental_component_id_overlap"
            if cfg.allow_unverified_component_join
            else "disabled_without_verified_component_to_carrier_mapping",
            "unverified_component_id_overlap_count": len(material_component_matches),
            "unverified_component_id_overlap_is_support": bool(cfg.allow_unverified_component_join),
            "native_support_join_available": native_join_rows > 0,
            "object_bank_available": bool(object_bank_rows and object_material_rows),
            "method_safe_inference_artifact": True,
            "is_method_result": True,
            "method_result_type": "soma_object_bank_not_scannet_ap",
            "is_scannet_ap_export": False,
            "scannet_ap_status": _scannet_ap_status(
                object_support_row_count=len(object_support_rows),
                native_join_row_count=native_join_rows,
            ),
            "evaluation_adapter_required_for_scannet_ap": True,
            "uses_gt_for_prediction": False,
            "uses_gt_geometry_for_inference": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "forbidden_for_method_table": False,
            "is_diagnostic_only": False,
            "blockers": blockers,
            "input_paths": {
                "object_rows": _rel(cfg.object_rows_path),
                "material_rows": _rel(cfg.material_rows_path),
                "carrier_rows": _rel(cfg.carrier_rows_path),
            },
            "mapping_policy_note": (
                "v64r2 object/material component ids and v53 native carrier component ids are only joined when "
                "allow_unverified_component_join=True. The default artifact does not treat same text component_id "
                "across L11 object histories and L6 objectlets as verified support."
            ),
            "gate": {
                "object_bank_available": bool(object_bank_rows and object_material_rows),
                "uses_gt_for_prediction_false": True,
                "uses_rgbd_pose_mesh_for_export_false": True,
                "not_scannet_ap_export": True,
                "has_view_mask_or_native_point_support": bool(object_support_rows),
                "native_carrier_join_available": native_join_rows > 0,
            },
        },
        context="v65 SOMA object bank summary",
    )
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return summary


def _read_csv_large(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _scene(row: dict[str, Any]) -> str:
    return str(row.get("scene_id") or row.get("scene") or "")


def _object_id_from_history(history_id: str) -> str:
    if "|" in history_id:
        return history_id.rsplit("|", 1)[-1]
    return history_id


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
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _as_string_list(value: Any) -> list[str]:
    return [str(item) for item in _json_list(value) if str(item) != ""]


def _material_ids_by_state(row: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "confirmed": _as_string_list(row.get("confirmed_material_ids")),
        "tentative": _as_string_list(row.get("tentative_material_ids")),
        "shared": _as_string_list(row.get("shared_material_ids")),
        "quarantine": _as_string_list(row.get("quarantine_material_ids")),
        "unknown": _as_string_list(row.get("unknown_material_ids")),
    }


def _object_declared_support_rows(
    scene: str,
    history_id: str,
    object_id: str,
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    observation_ids = _as_string_list(row.get("supporting_observation_ids"))
    if observation_ids:
        out = []
        for idx, obs_id in enumerate(observation_ids):
            parsed = _parse_mask_observation_id(obs_id)
            frame_id = parsed[1] if parsed else ""
            mask_id = parsed[2] if parsed else ""
            out.append(
                {
                    "scene_id": scene,
                    "object_id": object_id,
                    "history_id": history_id,
                    "material_id": "",
                    "component_id": "",
                    "carrier_id_if_available": "",
                    "support_kind": "object_declared_mask_observation_support",
                    "support_index": idx,
                    "frame_id": frame_id,
                    "observed_mask_id": mask_id,
                    "support_mask_observation_id": obs_id,
                    "uv_x": "",
                    "uv_y": "",
                    "xyz": "",
                    "carrier_global_id": "",
                    "uses_gt_for_prediction": False,
                    "uses_rgbd_pose_mesh_for_export": False,
                    "is_scannet_ap_export": False,
                }
            )
        return out
    mask_ids = _as_string_list(row.get("supporting_mask_ids"))
    frame_ids = _as_string_list(row.get("supporting_frame_ids"))
    count = max(len(mask_ids), len(frame_ids))
    out = []
    for idx in range(count):
        out.append(
            {
                "scene_id": scene,
                "object_id": object_id,
                "history_id": history_id,
                "material_id": "",
                "component_id": "",
                "carrier_id_if_available": "",
                "support_kind": "object_declared_mask_frame_support",
                "support_index": idx,
                "frame_id": frame_ids[idx] if idx < len(frame_ids) else "",
                "observed_mask_id": mask_ids[idx] if idx < len(mask_ids) else "",
                "support_mask_observation_id": mask_ids[idx] if idx < len(mask_ids) else "",
                "uv_x": "",
                "uv_y": "",
                "xyz": "",
                "carrier_global_id": "",
                "uses_gt_for_prediction": False,
                "uses_rgbd_pose_mesh_for_export": False,
                "is_scannet_ap_export": False,
            }
        )
    return out


def _material_declared_support_rows(
    scene: str,
    history_id: str,
    object_id: str,
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    support_ids = _as_string_list(row.get("support_observation_ids"))
    frame_ids = _as_string_list(row.get("frame_ids"))
    uv_tracks = _json_list(row.get("uv_tracks_if_available"))
    xyz_tracks = _json_list(row.get("xyz_tracks_if_available"))
    count = max(len(support_ids), len(frame_ids), len(uv_tracks), len(xyz_tracks))
    out = []
    for idx in range(count):
        uv = uv_tracks[idx] if idx < len(uv_tracks) else []
        xyz = xyz_tracks[idx] if idx < len(xyz_tracks) else []
        support_id = support_ids[idx] if idx < len(support_ids) else ""
        parsed = _parse_mask_observation_id(support_id)
        frame_id = frame_ids[idx] if idx < len(frame_ids) else (parsed[1] if parsed else "")
        mask_id = parsed[2] if parsed else ""
        out.append(
            {
                "scene_id": scene,
                "object_id": object_id,
                "history_id": history_id,
                "material_id": row.get("material_id", ""),
                "component_id": row.get("component_id", ""),
                "carrier_id_if_available": row.get("carrier_id_if_available", ""),
                "state": row.get("state", ""),
                "support_kind": "material_declared_support",
                "support_index": idx,
                "frame_id": frame_id,
                "observed_mask_id": mask_id,
                "support_mask_observation_id": support_id,
                "uv_x": _list_item(uv, 0),
                "uv_y": _list_item(uv, 1),
                "xyz": xyz,
                "carrier_global_id": "",
                "uses_gt_for_prediction": False,
                "uses_rgbd_pose_mesh_for_export": False,
                "is_scannet_ap_export": False,
            }
        )
    return out


def _native_carrier_support_rows(
    scene: str,
    history_id: str,
    object_id: str,
    row: dict[str, Any],
    carrier_index: dict[str, dict[tuple[str, str], list[dict[str, str]]]],
    *,
    allow_unverified_component_join: bool,
) -> list[dict[str, Any]]:
    if not allow_unverified_component_join:
        return []
    candidate_rows: list[dict[str, str]] = []
    component_id = str(row.get("component_id", "")).strip()
    carrier_id = str(row.get("carrier_id_if_available", "")).strip()
    if component_id:
        candidate_rows.extend(carrier_index["by_component"].get((scene, component_id), []))
    if carrier_id:
        candidate_rows.extend(carrier_index["by_carrier_id"].get((scene, carrier_id), []))
        candidate_rows.extend(carrier_index["by_carrier_global_id"].get((scene, carrier_id), []))
    out = []
    for idx, carrier in enumerate(_dedupe_carriers(candidate_rows)):
        out.append(
            {
                "scene_id": scene,
                "object_id": object_id,
                "history_id": history_id,
                "material_id": row.get("material_id", ""),
                "component_id": component_id,
                "carrier_id_if_available": carrier_id,
                "state": row.get("state", ""),
                "support_kind": "experimental_native_carrier_component_id_overlap",
                "support_index": idx,
                "frame_id": carrier.get("frame_id", ""),
                "observed_mask_id": carrier.get("observed_mask_id", ""),
                "support_mask_observation_id": carrier.get("support_mask_observation_id", ""),
                "uv_x": carrier.get("uv_x", ""),
                "uv_y": carrier.get("uv_y", ""),
                "xyz": "",
                "carrier_global_id": carrier.get("carrier_global_id", ""),
                "carrier_id": carrier.get("carrier_id", ""),
                "chunk_id": carrier.get("chunk_id", ""),
                "window_index": carrier.get("window_index", ""),
                "carrier_index": carrier.get("carrier_index", ""),
                "confidence": carrier.get("confidence", ""),
                "visibility_prob": carrier.get("visibility_prob", ""),
                "visible": carrier.get("visible", ""),
                "valid": carrier.get("valid", ""),
                "native_support_kind": carrier.get("native_support_kind", ""),
                "uses_gt_for_prediction": False,
                "uses_rgbd_pose_mesh_for_export": False,
                "is_scannet_ap_export": False,
            }
        )
    return out


def _build_carrier_index(rows: Iterable[dict[str, str]]) -> dict[str, dict[tuple[str, str], list[dict[str, str]]]]:
    index: dict[str, dict[tuple[str, str], list[dict[str, str]]]] = {
        "by_component": defaultdict(list),
        "by_carrier_id": defaultdict(list),
        "by_carrier_global_id": defaultdict(list),
    }
    for row in rows:
        scene = _scene(row)
        component_id = str(row.get("component_id", "")).strip()
        carrier_id = str(row.get("carrier_id", "")).strip()
        carrier_global_id = str(row.get("carrier_global_id", "")).strip()
        if component_id:
            index["by_component"][(scene, component_id)].append(row)
        if carrier_id:
            index["by_carrier_id"][(scene, carrier_id)].append(row)
        if carrier_global_id:
            index["by_carrier_global_id"][(scene, carrier_global_id)].append(row)
    return index


def _dedupe_carriers(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (
            _scene(row),
            str(row.get("carrier_global_id", "")),
            str(row.get("frame_id", "")),
            str(row.get("observed_mask_id", "")),
            str(row.get("support_mask_observation_id", "")),
            str(row.get("chunk_id", "")),
            str(row.get("window_index", "")),
            str(row.get("carrier_index", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _dedupe_support_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("scene_id", "")),
            str(row.get("history_id", "")),
            str(row.get("material_id", "")),
            str(row.get("support_kind", "")),
            str(row.get("frame_id", "")),
            str(row.get("observed_mask_id", "")),
            str(row.get("support_mask_observation_id", "")),
            str(row.get("carrier_global_id", "")),
            str(row.get("chunk_id", "")),
            str(row.get("window_index", "")),
            str(row.get("carrier_index", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _row_has_view_mask_support(row: dict[str, Any]) -> bool:
    return bool(str(row.get("observed_mask_id", "")).strip() or str(row.get("support_mask_observation_id", "")).strip())


def _row_has_point_or_carrier_support(row: dict[str, Any]) -> bool:
    if str(row.get("carrier_global_id", "")).strip():
        return True
    if str(row.get("uv_x", "")).strip() and str(row.get("uv_y", "")).strip():
        return True
    return bool(_json_list(row.get("xyz")))


def _native_support_count(counts: Counter[str]) -> int:
    return int(
        counts.get("native_carrier_observation_join", 0)
        + counts.get("experimental_native_carrier_component_id_overlap", 0)
    )


def _scannet_ap_status(*, object_support_row_count: int, native_join_row_count: int) -> str:
    if object_support_row_count <= 0:
        return "blocked_missing_object_support"
    if native_join_row_count <= 0:
        return "view_mask_support_ready_missing_native_carrier_or_ap_adapter"
    return "not_exported_requires_evaluation_adapter"


def _list_item(value: Any, idx: int) -> Any:
    if isinstance(value, (list, tuple)) and len(value) > idx:
        return value[idx]
    return ""


def _int_from_text(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


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


__all__ = [
    "DEFAULT_CARRIER_ROWS",
    "DEFAULT_MATERIAL_ROWS",
    "DEFAULT_OBJECT_ROWS",
    "V65SOMAObjectBankConfig",
    "build_v65_soma_object_bank",
    "write_v65_soma_object_bank",
]
