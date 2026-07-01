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


def build_v64r2_final_decision(
    *,
    main_fact_lock_path: str | Path = "outputs/audit/v64r2_phaseA0_main_fact_lock/main_fact_lock_summary.json",
    native_contract_path: str | Path = "outputs/audit/v64r2_native_contract/native_contract_summary.json",
    ap_probe_path: str | Path = "outputs/audit/v64r2_scannet_ap_probe5/ap_smoke_summary.json",
    ap_failure_path: str | Path = "outputs/audit/v64r2_ap_failure_attribution/failure_summary.json",
    dynamic_env_path: str | Path = "outputs/audit/v64r2_dynamic_env/dynamic_env_summary.json",
    dynamic_small_path: str | Path = "outputs/audit/v64r2_dynamic_small/dynamic_small_summary.json",
    active_query_path: str | Path = "outputs/audit/v64r2_active_query_optional/optional_query_summary.json",
) -> dict[str, Any]:
    main = _load_dict(main_fact_lock_path)
    native = _load_dict(native_contract_path)
    ap = _load_dict(ap_probe_path)
    ap_failure = _load_dict(ap_failure_path)
    dyn = _load_dict(dynamic_env_path)
    dyn_small = _load_dict(dynamic_small_path)
    query = _load_dict(active_query_path)
    main_status = main.get("summary", {}).get("main_ownership_status") if isinstance(main.get("summary"), dict) else None
    if main_status is None:
        main_status = "GO_MAIN_OWNERSHIP_FIELD" if main.get("gate", {}).get("pass") else "NO_GO_MAIN_OWNERSHIP_FIELD"
    method_safe_ap_available = bool(ap.get("method_safe_AP_available"))
    diagnostic_ap_available = bool(ap.get("diagnostic_AP_available"))
    if method_safe_ap_available:
        scannet_status = "GO_SCANNET_AP_METHOD"
    elif diagnostic_ap_available:
        scannet_status = "PARTIAL_SCANNET_AP_DIAGNOSTIC"
    else:
        scannet_status = "NO_GO_SCANNET_MATERIALIZATION"
    dyn_level = int(dyn.get("dyn_level") or 0)
    if dyn_small.get("gate", {}).get("pass"):
        dynamic_status = "GO_DYNAMIC_REPLICA"
    elif dyn_level >= 1 and bool(dyn_small):
        dynamic_status = "NO_GO_DYNAMIC_DATA"
    else:
        dynamic_status = "NO_GO_DYNAMIC_DATA"
    active_status = query.get("active_query_status") or "REMOVE_ACTIVE_QUERY_FROM_MAIN"
    blocked_claims: list[str] = []
    if not method_safe_ap_available:
        blocked_claims.append("scannet_method_ap_claim")
        blocked_claims.append("stream3d_cropformer_win_claim")
    if dynamic_status != "GO_DYNAMIC_REPLICA":
        blocked_claims.append("dynamic_official_tracking_or_4d_claim")
    if active_status != "GO_ACTIVE_QUERY_EXTENSION":
        blocked_claims.append("active_query_extension_claim")
    if main_status != "GO_MAIN_OWNERSHIP_FIELD":
        blocked_claims.append("main_ownership_field_claim")
    final_claim_allowed = []
    if main_status == "GO_MAIN_OWNERSHIP_FIELD":
        final_claim_allowed.append("SOMA-Manifold ownership field")
    if scannet_status == "PARTIAL_SCANNET_AP_DIAGNOSTIC":
        final_claim_allowed.append("diagnostic ScanNet AP / materialization analysis only")
    if dynamic_status == "GO_DYNAMIC_REPLICA":
        final_claim_allowed.append("Dynamic Replica tracking under available GT level")
    metric_rows = [
        {
            "track": "A_main_ownership",
            "status": main_status,
            "key_metric": "core_purity",
            "value": main.get("summary", {}).get("core_purity") if isinstance(main.get("summary"), dict) else None,
            "claim_allowed": main_status == "GO_MAIN_OWNERSHIP_FIELD",
        },
        {
            "track": "B_scannet_ap",
            "status": scannet_status,
            "key_metric": "best_diagnostic_AP",
            "value": ap.get("best_diagnostic_AP"),
            "claim_allowed": scannet_status == "GO_SCANNET_AP_METHOD",
        },
        {
            "track": "B_ap_failure",
            "status": ap_failure.get("top_failure_category"),
            "key_metric": "attribution_coverage",
            "value": ap_failure.get("attribution_coverage"),
            "claim_allowed": False,
        },
        {
            "track": "C_dynamic",
            "status": dynamic_status,
            "key_metric": "dyn_level",
            "value": dyn_level,
            "claim_allowed": dynamic_status == "GO_DYNAMIC_REPLICA",
        },
        {
            "track": "D_active_query",
            "status": active_status,
            "key_metric": "blocks_scannet_ap_or_dynamic",
            "value": bool(query.get("blocks_scannet_ap")) or bool(query.get("blocks_dynamic")),
            "claim_allowed": active_status == "GO_ACTIVE_QUERY_EXTENSION",
        },
    ]
    final_gate = {
        "main_ownership_go": main_status == "GO_MAIN_OWNERSHIP_FIELD",
        "active_query_does_not_block_ap": not bool(query.get("blocks_scannet_ap")),
        "active_query_does_not_block_dynamic": not bool(query.get("blocks_dynamic")),
        "scannet_ap_evaluated_or_blocked_explicitly": bool(ap),
        "dynamic_env_checked": bool(dyn.get("dynamic_env_check_complete")),
        "no_method_ap_claim_without_method_safe_ap": not method_safe_ap_available,
    }
    payload = {
        "phase": "v64r2_final",
        "created_at": utc_now(),
        "input_paths": {
            "main_fact_lock": _rel(main_fact_lock_path),
            "native_contract": _rel(native_contract_path),
            "ap_probe": _rel(ap_probe_path),
            "ap_failure": _rel(ap_failure_path),
            "dynamic_env": _rel(dynamic_env_path),
            "dynamic_small": _rel(dynamic_small_path),
            "active_query": _rel(active_query_path),
        },
        "decision_label": "PARTIAL_V64R2_EVALUATION_FIRST_DIAGNOSTIC_AP" if main_status == "GO_MAIN_OWNERSHIP_FIELD" else "NO_GO_V64R2_MAIN_FIELD",
        "main_ownership_status": main_status,
        "scannet_ap_status": scannet_status,
        "dynamic_status": dynamic_status,
        "active_query_status": active_status,
        "method_safe_ap_available": method_safe_ap_available,
        "diagnostic_ap_available": diagnostic_ap_available,
        "dynamic_data_level": dyn.get("dyn_level_label"),
        "final_claim_allowed": final_claim_allowed,
        "blocked_claims": blocked_claims,
        "top_ap_failure_category": ap_failure.get("top_failure_category"),
        "native_contract_limitation": native.get("native_field_limitation"),
        "dynamic_blocked_reason": dyn_small.get("blocked_reason"),
        "final_gate": final_gate,
        "metric_rows": metric_rows,
    }
    return payload


def write_v64r2_final_decision(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "final_decision.json", payload)
    write_csv(out / "final_metric_rows.csv", payload["metric_rows"])
