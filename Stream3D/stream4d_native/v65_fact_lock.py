from __future__ import annotations

from pathlib import Path
from typing import Any

from .v65_common import first_row, load_dict, load_rows, project, rel, sha256_file, write_standard_outputs


REQUIRED_ARTIFACTS = {
    "final_decision": "outputs/audit/v64r2_final/final_decision.json",
    "ap_smoke_summary": "outputs/audit/v64r2_scannet_ap_probe5/ap_smoke_summary.json",
    "ap_metric_rows": "outputs/audit/v64r2_scannet_ap_probe5/ap_metric_rows.csv",
    "ap_failure_summary": "outputs/audit/v64r2_ap_failure_attribution/failure_summary.json",
    "bridge_used_support": "outputs/audit/v64r2_bridge_wta_used_frame_support_check/v64r2_probe5_v53_bridge_wta_used_support_used_frame_support_summary.json",
    "bridge_support_contrast": "outputs/audit/v64r2_bridge_wta_support_contrast_iou_probe5/ap_iou_distribution_summary.json",
    "d4rt_chunk_scale_first": "outputs/audit/v64r2_d4rt_chunk_scale_first_ap_probe5/D4RT_geometry_replacement_stream3d_probe5.csv",
    "dynamic_env": "outputs/audit/v64r2_dynamic_env/dynamic_env_summary.json",
    "main_fact_lock": "outputs/audit/v64r2_phaseA0_main_fact_lock/main_fact_lock_summary.json",
    "native_contract": "outputs/audit/v64r2_native_contract/native_contract_summary.json",
}


def _stable_final_decision_rebuild_matches() -> bool:
    try:
        from .v64r2_final_eval import build_v64r2_final_decision

        rebuilt = build_v64r2_final_decision()
    except Exception:
        return False
    existing = load_dict(REQUIRED_ARTIFACTS["final_decision"])
    keys = [
        "decision_label",
        "main_ownership_status",
        "scannet_ap_status",
        "dynamic_status",
        "active_query_status",
        "method_safe_ap_available",
        "diagnostic_ap_available",
    ]
    return bool(existing) and all(existing.get(key) == rebuilt.get(key) for key in keys)


def build_v65_fact_lock() -> dict[str, Any]:
    artifact_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for name, path in REQUIRED_ARTIFACTS.items():
        path_obj = project(path)
        row = {
            "artifact": name,
            "path": rel(path),
            "exists": path_obj.exists(),
            "sha256": sha256_file(path_obj) if path_obj.is_file() else "",
        }
        artifact_rows.append(row)
        if not row["exists"]:
            missing_rows.append({**row, "repair_attempt": _repair_attempt_for(name)})

    final_decision = load_dict(REQUIRED_ARTIFACTS["final_decision"])
    ap_summary = load_dict(REQUIRED_ARTIFACTS["ap_smoke_summary"])
    failure_summary = load_dict(REQUIRED_ARTIFACTS["ap_failure_summary"])
    dynamic = load_dict(REQUIRED_ARTIFACTS["dynamic_env"])
    main = load_dict(REQUIRED_ARTIFACTS["main_fact_lock"])
    native = load_dict(REQUIRED_ARTIFACTS["native_contract"])
    ap_rows = load_rows(REQUIRED_ARTIFACTS["ap_metric_rows"])
    d4rt_rows = load_rows(REQUIRED_ARTIFACTS["d4rt_chunk_scale_first"])
    bridge_contrast = load_dict(REQUIRED_ARTIFACTS["bridge_support_contrast"])

    main_metric = {}
    if isinstance(main.get("metric_rows"), list) and main["metric_rows"]:
        main_metric = main["metric_rows"][0]
    ap4 = first_row(ap_rows, "variant", "AP4_v64r2_rgbd_pose_mesh_bridge_diagnostic")
    g11 = first_row(d4rt_rows, "variant", "G11")
    g12 = first_row(d4rt_rows, "variant", "G12")
    contrast_rows = bridge_contrast.get("rows") if isinstance(bridge_contrast.get("rows"), list) else []
    old_bridge_support = load_dict(REQUIRED_ARTIFACTS["bridge_used_support"])
    support_numeric = old_bridge_support.get("numeric_mean") if isinstance(old_bridge_support.get("numeric_mean"), dict) else {}

    current_metric_rows = [
        _metric("main_ownership_status", final_decision.get("main_ownership_status"), REQUIRED_ARTIFACTS["final_decision"]),
        _metric("core_purity", main_metric.get("core_purity"), REQUIRED_ARTIFACTS["main_fact_lock"]),
        _metric("core_completeness", main_metric.get("core_completeness"), REQUIRED_ARTIFACTS["main_fact_lock"]),
        _metric("state_coverage_rate", main_metric.get("state_coverage_rate"), REQUIRED_ARTIFACTS["main_fact_lock"]),
        _metric("method_safe_AP_available", ap_summary.get("method_safe_AP_available"), REQUIRED_ARTIFACTS["ap_smoke_summary"]),
        _metric("diagnostic_AP_available", ap_summary.get("diagnostic_AP_available"), REQUIRED_ARTIFACTS["ap_smoke_summary"]),
        _metric("best_diagnostic_AP", ap_summary.get("best_diagnostic_AP"), REQUIRED_ARTIFACTS["ap_smoke_summary"]),
        _metric("best_diagnostic_AP50", ap_summary.get("best_diagnostic_AP50"), REQUIRED_ARTIFACTS["ap_smoke_summary"]),
        _metric("best_diagnostic_AP25", ap_summary.get("best_diagnostic_AP25"), REQUIRED_ARTIFACTS["ap_smoke_summary"]),
        _metric("old_bridge_wta_AP", ap4.get("AP"), REQUIRED_ARTIFACTS["ap_metric_rows"]),
        _metric("old_bridge_wta_support_scope", "PREDICTION_UNION_ISLAND", REQUIRED_ARTIFACTS["ap_metric_rows"]),
        _metric("old_bridge_wta_pre_points_count_mean", support_numeric.get("prediction_union_count"), REQUIRED_ARTIFACTS["bridge_used_support"]),
        _metric("old_bridge_wta_gt_instance_count_mean", None, REQUIRED_ARTIFACTS["bridge_support_contrast"]),
        _metric("used_support_AP", _eval_tail("data/evaluation/scannet/v64r2_probe5_v53_bridge_wta_used_support_class_agnostic.txt", "AP"), "data/evaluation/scannet/v64r2_probe5_v53_bridge_wta_used_support_class_agnostic.txt"),
        _metric("used_support_scope", "USED_FRAME_VISIBLE_SUPPORT", "data/prediction/v64r2_probe5_v53_bridge_wta_used_support_class_agnostic/config_manifest.json"),
        _metric("used_support_pre_points_count_mean", support_numeric.get("support_count"), REQUIRED_ARTIFACTS["bridge_used_support"]),
        _metric("used_support_gt_instance_count_mean", None, REQUIRED_ARTIFACTS["bridge_used_support"]),
        _metric("G11_AP", g11.get("ap"), REQUIRED_ARTIFACTS["d4rt_chunk_scale_first"]),
        _metric("G11_AP50", g11.get("ap50"), REQUIRED_ARTIFACTS["d4rt_chunk_scale_first"]),
        _metric("G11_AP25", g11.get("ap25"), REQUIRED_ARTIFACTS["d4rt_chunk_scale_first"]),
        _metric("G11_raw_pred_count_mean", g11.get("num_pred_per_scene"), REQUIRED_ARTIFACTS["d4rt_chunk_scale_first"]),
        _metric("G12_AP", g12.get("ap"), REQUIRED_ARTIFACTS["d4rt_chunk_scale_first"]),
        _metric("G12_AP50", g12.get("ap50"), REQUIRED_ARTIFACTS["d4rt_chunk_scale_first"]),
        _metric("G12_AP25", g12.get("ap25"), REQUIRED_ARTIFACTS["d4rt_chunk_scale_first"]),
        _metric("G12_raw_pred_count_mean", g12.get("num_pred_per_scene"), REQUIRED_ARTIFACTS["d4rt_chunk_scale_first"]),
        _metric("dynamic_data_level", dynamic.get("dyn_level_label"), REQUIRED_ARTIFACTS["dynamic_env"]),
        _metric("dynamic_missing_files", "; ".join(dynamic.get("blocked_official_metric_reasons", [])), REQUIRED_ARTIFACTS["dynamic_env"]),
        _metric("native_field_limitation", native.get("native_field_limitation"), REQUIRED_ARTIFACTS["native_contract"]),
    ]
    for prefix in ["G11", "G12"]:
        config = f"v64r2_d4rt_chunk_scale_first_ap_probe5_{prefix.lower()}"
        per_config = [row for row in contrast_rows if str(row.get("config")) == config]
        current_metric_rows.extend(
            [
                _metric(f"{prefix}_kept_pred_count_mean", _mean(per_config, "kept_pred_count"), REQUIRED_ARTIFACTS["bridge_support_contrast"]),
                _metric(f"{prefix}_dropped_pred_lt100_mean", _mean(per_config, "dropped_pred_lt100"), REQUIRED_ARTIFACTS["bridge_support_contrast"]),
                _metric(f"{prefix}_pred_best_iou_median", _mean(per_config, "pred_best_iou_median"), REQUIRED_ARTIFACTS["bridge_support_contrast"]),
            ]
        )

    summary = {
        "phase": "v65_phase0_fact_lock",
        "all_required_artifacts_found": not missing_rows,
        "final_decision_reproducible": _stable_final_decision_rebuild_matches(),
        "AP_rows_parsed": bool(ap_rows),
        "support_contrast_rows_parsed": bool(contrast_rows),
        "D4RT_G11_G12_rows_parsed": bool(g11 and g12),
        "dynamic_env_parsed": bool(dynamic),
        "main_ownership_status": final_decision.get("main_ownership_status"),
        "method_safe_AP_available": ap_summary.get("method_safe_AP_available"),
        "diagnostic_AP_available": ap_summary.get("diagnostic_AP_available"),
        "best_diagnostic_AP": ap_summary.get("best_diagnostic_AP"),
        "best_diagnostic_AP50": ap_summary.get("best_diagnostic_AP50"),
        "best_diagnostic_AP25": ap_summary.get("best_diagnostic_AP25"),
        "dynamic_data_level": dynamic.get("dyn_level_label"),
        "missing_artifact_count": len(missing_rows),
        "artifact_rows": artifact_rows,
    }
    summary["gate"] = {
        "all_required_artifacts_found": summary["all_required_artifacts_found"],
        "final_decision_reproducible": summary["final_decision_reproducible"],
        "AP_rows_parsed": summary["AP_rows_parsed"],
        "support_contrast_rows_parsed": summary["support_contrast_rows_parsed"],
        "D4RT_G11_G12_rows_parsed": summary["D4RT_G11_G12_rows_parsed"],
        "dynamic_env_parsed": summary["dynamic_env_parsed"],
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return {"summary": summary, "current_metric_rows": current_metric_rows, "missing_artifact_rows": missing_rows}


def write_v65_fact_lock(output_root: str | Path, payload: dict[str, Any]) -> None:
    write_standard_outputs(
        output_root,
        {
            "fact_lock_summary.json": payload["summary"],
            "current_metric_rows.csv": payload["current_metric_rows"],
            "missing_artifact_rows.csv": payload["missing_artifact_rows"],
        },
    )


def _metric(metric: str, value: Any, source_path: str) -> dict[str, Any]:
    return {"metric": metric, "value": value, "source_path": rel(source_path), "source_sha256": sha256_file(source_path)}


def _repair_attempt_for(name: str) -> str:
    return {
        "final_decision": "rebuild from v64r2 final eval aggregator",
        "ap_metric_rows": "rerun run_v64r2_scannet_ap_probe5.py from existing prediction configs",
        "bridge_used_support": "rerun build_v64r2_used_frame_support_config.py",
        "bridge_support_contrast": "rerun diagnose_v64r2_ap_iou_distribution.py",
        "d4rt_chunk_scale_first": "locate d4rt_chunk_scale_first outputs and rerun AP parsing only",
    }.get(name, "record missing; do not infer values")


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return float(sum(values) / len(values)) if values else None


def _eval_tail(path: str, key: str) -> float | None:
    from .v65_common import parse_eval_metric_file

    return parse_eval_metric_file(path).get(key)
