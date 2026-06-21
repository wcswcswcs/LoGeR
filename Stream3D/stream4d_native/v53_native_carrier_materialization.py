from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .v47_common import ROOT, parse_float, parse_int, read_csv, read_json, utc_now, write_csv, write_json
from .v53_local_objectlets import _load_component_ids
from .v53_mask_component_support import _build_components, _carrier_global_id


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _load_best_variant(summary_path: str | Path, fallback: str) -> str:
    path = _project(summary_path)
    if not path.exists():
        return fallback
    payload = read_json(path)
    if not isinstance(payload, dict):
        return fallback
    best = payload.get("best_real_row", {}) if isinstance(payload.get("best_real_row"), dict) else {}
    return str(payload.get("best_real_variant") or best.get("variant") or fallback)


def _mask_observation_id(scene: str, frame_id: int, mask_id: int) -> str:
    return f"{scene}:{int(frame_id)}:{int(mask_id)}" if mask_id > 0 else ""


def build_native_carrier_materialization(
    *,
    carrier_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv",
    mask_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv",
    objectlet_summary_path: str | Path = "outputs/audit/v53_local_objectlets_k0_conflict_veto025_skip_repeated_sig_l12/local_objectlet_summary.json",
    objectlet_rows_path: str | Path = "outputs/audit/v53_local_objectlets_k0_conflict_veto025_skip_repeated_sig_l12/objectlet_rows.csv",
    objectlet_variant: str | None = None,
    max_union_unique_carriers: int = 32,
    min_visibility_prob: float = 0.5,
    min_confidence: float = 0.5,
) -> dict[str, Any]:
    input_paths = {
        "carrier_table_path": _project(carrier_table_path),
        "mask_table_path": _project(mask_table_path),
        "objectlet_rows_path": _project(objectlet_rows_path),
    }
    missing_inputs = [name for name, path in input_paths.items() if not path.exists()]
    variant = objectlet_variant or _load_best_variant(objectlet_summary_path, "L6_coverage_first_minnew025")
    if missing_inputs:
        summary = {
            "phase": "v53_native_carrier_materialization",
            "created_at": utc_now(),
            "native_carrier_materialization_pass": False,
            "method_safe_native_support_available": False,
            "method_safe_ap_available": False,
            "missing_inputs": missing_inputs,
            "objectlet_variant": variant,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "is_scannet_ap_export": False,
        }
        return {"summary": summary, "carrier_rows": [], "objectlet_summary_rows": [], "component_rows": []}

    carrier_rows = read_csv(input_paths["carrier_table_path"])
    mask_rows = read_csv(input_paths["mask_table_path"])
    objectlet_rows = read_csv(input_paths["objectlet_rows_path"])
    selected_objectlets = [row for row in objectlet_rows if str(row.get("variant")) == str(variant)]

    component_payload = _build_components(
        carrier_rows=carrier_rows,
        mask_rows=mask_rows,
        max_union_unique_carriers=max_union_unique_carriers,
        min_visibility_prob=min_visibility_prob,
        min_confidence=min_confidence,
    )
    visible_rows = component_payload["visible_rows"]
    component_by_carrier = component_payload["component_by_carrier"]

    objectlets_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_count_by_objectlet: dict[str, int] = {}
    for objectlet in selected_objectlets:
        objectlet_id = str(objectlet.get("objectlet_id") or "")
        components = _load_component_ids(objectlet.get("component_ids"))
        component_count_by_objectlet[objectlet_id] = len(components)
        for component_id in components:
            objectlets_by_component[component_id].append(objectlet)

    carrier_support_rows: list[dict[str, Any]] = []
    objectlet_observation_count: Counter[str] = Counter()
    objectlet_frame_ids: dict[str, set[int]] = defaultdict(set)
    objectlet_native_carriers: dict[str, set[str]] = defaultdict(set)
    component_observation_count: Counter[str] = Counter()
    component_native_carriers: dict[str, set[str]] = defaultdict(set)
    scene_observation_count: Counter[str] = Counter()
    all_native_carriers: set[str] = set()

    for carrier_row in visible_rows:
        carrier_global_id = _carrier_global_id(carrier_row)
        component_id = component_by_carrier.get(carrier_global_id)
        if not component_id:
            continue
        objectlets = objectlets_by_component.get(component_id)
        if not objectlets:
            continue
        scene = str(carrier_row.get("scene") or "")
        frame_id = parse_int(carrier_row.get("frame_id"))
        observed_mask_id = parse_int(carrier_row.get("observed_mask_id"))
        support_mask_id = _mask_observation_id(scene, frame_id, observed_mask_id)
        for objectlet in objectlets:
            objectlet_id = str(objectlet.get("objectlet_id") or "")
            row = {
                "variant": variant,
                "objectlet_id": objectlet_id,
                "scene": scene,
                "chunk_id": objectlet.get("chunk_id") or carrier_row.get("chunk_id"),
                "component_id": component_id,
                "carrier_global_id": carrier_global_id,
                "carrier_id": carrier_row.get("carrier_id"),
                "frame_id": frame_id,
                "carrier_observation_chunk_id": carrier_row.get("chunk_id"),
                "submap_id": carrier_row.get("submap_id"),
                "window_index": carrier_row.get("window_index"),
                "carrier_index": carrier_row.get("carrier_index"),
                "uv_x": parse_float(carrier_row.get("uv_x")),
                "uv_y": parse_float(carrier_row.get("uv_y")),
                "confidence": parse_float(carrier_row.get("confidence")),
                "visibility_prob": parse_float(carrier_row.get("visibility_prob")),
                "visible": carrier_row.get("visible"),
                "valid": carrier_row.get("valid"),
                "valid_uv": carrier_row.get("valid_uv"),
                "observed_mask_id": observed_mask_id,
                "support_mask_observation_id": support_mask_id,
                "native_support_kind": "d4rt_carrier_global_id",
                "is_scannet_ap_export": False,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
                "uses_rgbd_pose_mesh_for_export": False,
            }
            carrier_support_rows.append(row)
            objectlet_observation_count[objectlet_id] += 1
            objectlet_frame_ids[objectlet_id].add(frame_id)
            objectlet_native_carriers[objectlet_id].add(carrier_global_id)
            component_observation_count[component_id] += 1
            component_native_carriers[component_id].add(carrier_global_id)
            scene_observation_count[scene] += 1
            all_native_carriers.add(carrier_global_id)

    objectlet_summary_rows: list[dict[str, Any]] = []
    for objectlet in selected_objectlets:
        objectlet_id = str(objectlet.get("objectlet_id") or "")
        frame_ids = sorted(objectlet_frame_ids.get(objectlet_id, set()))
        objectlet_summary_rows.append(
            {
                "variant": variant,
                "objectlet_id": objectlet_id,
                "scene": objectlet.get("scene"),
                "chunk_id": objectlet.get("chunk_id"),
                "candidate_id": objectlet.get("candidate_id"),
                "source_mask_observation_id": objectlet.get("source_mask_observation_id"),
                "component_count": component_count_by_objectlet.get(objectlet_id, 0),
                "native_observation_count": int(objectlet_observation_count.get(objectlet_id, 0)),
                "unique_native_carrier_count": int(len(objectlet_native_carriers.get(objectlet_id, set()))),
                "support_frame_count": int(len(frame_ids)),
                "support_frame_ids": frame_ids,
                "has_native_carrier_support": int(objectlet_observation_count.get(objectlet_id, 0)) > 0,
                "native_support_kind": "d4rt_carrier_global_id",
                "is_scannet_ap_export": False,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
                "uses_rgbd_pose_mesh_for_export": False,
            }
        )

    component_rows = [
        {
            "variant": variant,
            "component_id": component_id,
            "objectlet_count": len(objectlets),
            "objectlet_ids": [str(row.get("objectlet_id") or "") for row in objectlets],
            "native_observation_count": int(component_observation_count.get(component_id, 0)),
            "unique_native_carrier_count": int(len(component_native_carriers.get(component_id, set()))),
            "has_native_carrier_support": int(component_observation_count.get(component_id, 0)) > 0,
            "native_support_kind": "d4rt_carrier_global_id",
            "is_scannet_ap_export": False,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
            "uses_rgbd_pose_mesh_for_export": False,
        }
        for component_id, objectlets in sorted(objectlets_by_component.items())
    ]

    selected_component_count = len(objectlets_by_component)
    duplicate_selected_component_count = sum(max(len(objectlets) - 1, 0) for objectlets in objectlets_by_component.values())
    objectlet_count_with_carriers = sum(1 for count in objectlet_observation_count.values() if count > 0)
    component_count_with_carriers = sum(1 for count in component_observation_count.values() if count > 0)
    gate = {
        "selected_objectlet_count_gt_0": len(selected_objectlets) > 0,
        "selected_component_count_gt_0": selected_component_count > 0,
        "native_observation_row_count_gt_0": len(carrier_support_rows) > 0,
        "native_unique_carrier_count_gt_0": len(all_native_carriers) > 0,
        "uses_gt_for_prediction_eq_false": True,
        "uses_gt_for_diagnostic_labels_eq_false": True,
        "uses_rgbd_pose_mesh_for_export_eq_false": True,
        "is_scannet_ap_export_eq_false": True,
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v53_native_carrier_materialization",
        "created_at": utc_now(),
        "carrier_table_path": _rel(carrier_table_path),
        "mask_table_path": _rel(mask_table_path),
        "objectlet_summary_path": _rel(objectlet_summary_path),
        "objectlet_rows_path": _rel(objectlet_rows_path),
        "objectlet_variant": variant,
        "max_union_unique_carriers": int(max_union_unique_carriers),
        "min_visibility_prob": float(min_visibility_prob),
        "min_confidence": float(min_confidence),
        "input_carrier_observation_count": len(carrier_rows),
        "visible_carrier_observation_count": len(visible_rows),
        "selected_objectlet_count": len(selected_objectlets),
        "selected_component_count": selected_component_count,
        "duplicate_selected_component_count": duplicate_selected_component_count,
        "native_observation_row_count": len(carrier_support_rows),
        "native_unique_carrier_count": len(all_native_carriers),
        "objectlet_count_with_native_carriers": objectlet_count_with_carriers,
        "component_count_with_native_carriers": component_count_with_carriers,
        "scene_native_observation_counts": dict(sorted(scene_observation_count.items())),
        "native_support_kind": "d4rt_carrier_global_id",
        "AP_bridge_status": "not_evaluated_native_carrier_support_not_scannet_ap",
        "real_method_ap_status": "not_run",
        "is_scannet_ap_export": False,
        "method_safe_native_support_available": bool(gate["pass"]),
        "method_safe_ap_available": False,
        "native_carrier_materialization_pass": bool(gate["pass"]),
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
        "uses_rgbd_pose_mesh_for_export": False,
        "required_future_change": (
            "connect audited v53 D4RT carrier ids to ScanNet scene/mesh point ids or add a native-carrier evaluator; "
            "this artifact is method-safe D4RT support but is not a ScanNet AP prediction mask."
        ),
    }
    return {
        "summary": summary,
        "carrier_rows": carrier_support_rows,
        "objectlet_summary_rows": objectlet_summary_rows,
        "component_rows": component_rows,
    }


def write_native_carrier_materialization(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "native_carrier_summary.json", payload["summary"])
    write_csv(out / "objectlet_native_carrier_rows.csv", payload["carrier_rows"])
    write_csv(out / "objectlet_native_carrier_summary_rows.csv", payload["objectlet_summary_rows"])
    write_csv(out / "component_native_carrier_rows.csv", payload["component_rows"])


__all__ = ["build_native_carrier_materialization", "write_native_carrier_materialization"]
