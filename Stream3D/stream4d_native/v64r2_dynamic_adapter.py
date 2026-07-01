from __future__ import annotations

from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


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


def build_v64r2_dynamic_adapter(
    *,
    dynamic_env_summary_path: str | Path = "outputs/audit/v64r2_dynamic_env/dynamic_env_summary.json",
) -> dict[str, Any]:
    env = _load_dict(dynamic_env_summary_path)
    dyn_level = int(env.get("dyn_level") or 0)
    masks_available = bool(env.get("instance_masks_exist") or env.get("object_ids_exist"))
    can_run_method = bool(dyn_level >= 1 and masks_available)
    method_rows: list[dict[str, Any]] = []
    for name in [
        "D0_mask_only_memory",
        "D1_D4RT_material_only",
        "D2_v62_SOMA_ownership_static_policy",
        "D3_v64r2_SOMA_dynamic_adapter",
        "D4_Stream3D_style_baseline_if_available",
        "D5_no_temporal_control",
        "D6_shuffled_D4RT_control",
    ]:
        method_rows.append(
            {
                "method_row": name,
                "scene_id": "ALL",
                "sequence_id": "",
                "frame_count": env.get("actual_file_counts", {}).get("image_exists") if isinstance(env.get("actual_file_counts"), dict) else None,
                "mask_count": 0,
                "material_node_count": 0,
                "history_object_count": 0,
                "confirmed_material_count": 0,
                "tentative_material_count": 0,
                "shared_material_count": 0,
                "quarantine_material_count": 0,
                "runtime_per_frame": None,
                "uses_gt_for_prediction": False,
                "status": "blocked_missing_actual_masks_or_object_id_maps" if not can_run_method else "not_run_needs_d4rt_dynamic_cache",
                "failure_reason": (
                    "Dynamic Replica annotation declares mask/object fields, but actual mask/object-id files are missing in the extracted tree."
                    if not masks_available
                    else "D4RT dynamic cache/generator not available in current v64r2 run."
                ),
            }
        )
    gate = {
        "method_runs_on_at_least_3_sequences": False,
        "history_object_count_gt_0": False,
        "material_node_count_gt_0": False,
        "uses_gt_for_prediction_false": True,
    }
    summary = {
        "phase": "v64r2_dynamic_adapter",
        "created_at": utc_now(),
        "input_paths": {"dynamic_env_summary": _rel(dynamic_env_summary_path)},
        "dyn_level": dyn_level,
        "dyn_level_label": env.get("dyn_level_label"),
        "method_adapter_status": "blocked_missing_actual_masks_or_object_id_maps"
        if not masks_available
        else "blocked_missing_d4rt_dynamic_cache",
        "method_runs_on_at_least_3_sequences": False,
        "history_object_count": 0,
        "material_node_count": 0,
        "uses_gt_for_prediction": False,
        "gate": gate,
        "blocked_reason": method_rows[0]["failure_reason"] if method_rows else "",
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "summary": summary,
        "dynamic_object_rows": method_rows,
        "dynamic_material_rows": [],
    }


def write_v64r2_dynamic_adapter(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "adapter_summary.json", payload["summary"])
    write_csv(out / "dynamic_object_rows.csv", payload["dynamic_object_rows"])
    write_csv(out / "dynamic_material_rows.csv", payload["dynamic_material_rows"])
