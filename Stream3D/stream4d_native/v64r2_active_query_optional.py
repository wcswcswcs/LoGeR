from __future__ import annotations

from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_json


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


def build_v64r2_active_query_optional(
    *,
    v63_final_path: str | Path = "outputs/audit/v63_final/final_decision.json",
    ap_probe_summary_path: str | Path = "outputs/audit/v64r2_scannet_ap_probe5/ap_smoke_summary.json",
    dynamic_env_summary_path: str | Path = "outputs/audit/v64r2_dynamic_env/dynamic_env_summary.json",
) -> dict[str, Any]:
    v63 = _load_dict(v63_final_path)
    ap = _load_dict(ap_probe_summary_path)
    dyn = _load_dict(dynamic_env_summary_path)
    v63_label = str(v63.get("decision_label") or "")
    active_pass = v63_label.startswith("GO_") and "ACTIVE_QUERY" in v63_label
    status = "GO_ACTIVE_QUERY_EXTENSION" if active_pass else "REMOVE_ACTIVE_QUERY_FROM_MAIN"
    payload = {
        "phase": "v64r2_active_query_optional",
        "created_at": utc_now(),
        "input_paths": {
            "v63_final": _rel(v63_final_path),
            "ap_probe_summary": _rel(ap_probe_summary_path),
            "dynamic_env_summary": _rel(dynamic_env_summary_path),
        },
        "active_query_status": status,
        "main_method_unchanged": True,
        "blocks_scannet_ap": False,
        "blocks_dynamic": False,
        "v63_decision_label": v63_label,
        "v63_key_metrics": v63.get("key_metrics", {}),
        "ap_already_launched_or_completed": bool(ap),
        "dynamic_env_already_launched_or_completed": bool(dyn),
        "evidence": "v63 final decision did not pass active-query method contribution gate; v64-r2 demotes it to optional/future work.",
        "gate": {
            "active_query_failure_cannot_block_scannet_ap": True,
            "active_query_failure_cannot_block_dynamic": True,
            "active_query_claim_removed_if_no_go": not active_pass,
            "pass": True,
        },
    }
    return payload


def write_v64r2_active_query_optional(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "optional_query_summary.json", payload)
