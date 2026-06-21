from __future__ import annotations

from pathlib import Path
from typing import Any

from stream4d_native.v47_common import ROOT, read_json, utc_now, write_csv, write_json


DEFAULT_SCENES = [
    "scene0011_00",
    "scene0030_00",
    "scene0050_00",
    "scene0081_01",
    "scene0591_00",
]


DEFAULT_ROOTS = {
    "v37_v44_summary": "outputs/audit/v44_native_full_probe5_core_first_l034/v44_native_typed_summary.json",
    "v46_fact": "outputs/audit/v46_fact_lock/fact_lock.json",
    "v46_final": "outputs/audit/v46_final_decision/v46_final_decision.json",
    "v47_fact": "outputs/audit/v47_fact_lock/fact_lock.json",
    "v47_observation": "outputs/audit/v47_observation_tables_metricfix/observation_table_summary.json",
    "v47_final": "outputs/audit/v47_final_decision_phase9_continued21_carrier_mdl_audit/v47_final_decision.json",
    "v47_stage1": "outputs/audit/v47_stage1_final_gate_continued21_carrier_mdl_audit/stage1_final_gate_summary.json",
    "v47_tracklet": "outputs/audit/v47_tracklets_strict_veto_A5/tracklet_construction.json",
    "v47_carrier_supertrack": "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_summary.json",
    "v47_carrier_mdl": "outputs/audit/v47_carrier_component_mdl_semantic_continued19/carrier_component_mdl_semantic_summary.json",
    "v47_matching_flow": "outputs/audit/v47_matching_flow_gap2_global_proxy/matching_flow_summary.json",
    "semantic_source": "outputs/audit/v46_loger_env_radio_radseg_availability_recheck_20260619/source_availability.json",
    "radio_vipe": "outputs/audit/v46_loger_env_radio_radseg_availability_recheck_20260619/radio_vipe_availability.json",
}


def project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def rel(path: str | Path) -> str:
    path_obj = project_path(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def load_optional_json(path: str | Path) -> dict[str, Any]:
    path_obj = project_path(path)
    if not path_obj.exists():
        return {"missing": True, "path": rel(path_obj)}
    payload = read_json(path_obj)
    if isinstance(payload, dict):
        return payload
    return {"payload": payload, "path": rel(path_obj)}


def nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def metric_row(
    *,
    key: str,
    value: Any,
    source: str | Path,
    note: str = "",
    required: bool = True,
) -> dict[str, Any]:
    return {
        "key": key,
        "value": value,
        "source": rel(source),
        "note": note,
        "required": required,
        "available": value is not None,
    }


def write_payload_bundle(output_root: str | Path, name: str, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    out = project_path(output_root)
    write_json(out / f"{name}.json", payload)
    write_csv(out / f"{name}_rows.csv", rows)

