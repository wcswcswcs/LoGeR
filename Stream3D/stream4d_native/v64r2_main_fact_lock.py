from __future__ import annotations

from pathlib import Path
from typing import Any

from .v47_common import ROOT, parse_float, read_json, utc_now, write_csv, write_json


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


def _claim_status(final_decision: dict[str, Any], claim: str) -> str:
    table = final_decision.get("claim_table", {})
    item = table.get(claim, {}) if isinstance(table, dict) else {}
    if not isinstance(item, dict):
        return "missing"
    return "pass" if bool(item.get("pass")) else "fail"


def build_v64r2_main_fact_lock(
    *,
    v62_final_path: str | Path = "outputs/audit/v62_final/final_decision.json",
    v62_solver_summary_path: str | Path = "outputs/audit/v62_solver_v2/solver_summary.json",
    v62_native_summary_path: str | Path = "outputs/audit/v62_native_field/native_field_summary.json",
    v62_increment_summary_path: str | Path = "outputs/audit/v62_increment_attribution/increment_summary.json",
    v62_stress_summary_path: str | Path = "outputs/audit/v62_stress_regen/stress_regen_summary.json",
) -> dict[str, Any]:
    final_decision = _load_dict(v62_final_path)
    solver = _load_dict(v62_solver_summary_path)
    native = _load_dict(v62_native_summary_path)
    increment = _load_dict(v62_increment_summary_path)
    stress = _load_dict(v62_stress_summary_path)
    key_metrics = final_decision.get("key_metrics", {}) if isinstance(final_decision.get("key_metrics"), dict) else {}
    decision_label = str(final_decision.get("decision_label") or "")
    core_purity = parse_float(key_metrics.get("core_purity", solver.get("full_solver_core_purity")), -1.0)
    core_completeness = parse_float(
        key_metrics.get("core_completeness", solver.get("full_solver_core_completeness")), -1.0
    )
    expanded_completeness = core_completeness
    state_coverage_rate = parse_float(key_metrics.get("state_coverage"), -1.0)
    real_minus_shuffled_ari = parse_float(
        key_metrics.get("real_minus_shuffled_ARI", solver.get("full_solver_real_minus_shuffled_ARI")), -1.0
    )
    real_minus_no_temporal_ari = parse_float(
        key_metrics.get("real_minus_no_temporal_ARI", solver.get("full_solver_real_minus_no_temporal_ARI")), -1.0
    )
    uses_gt_for_prediction = bool(final_decision.get("uses_gt_for_prediction", True)) or bool(
        solver.get("uses_gt_for_prediction", True)
    )
    uses_rgbd_pose_mesh_for_prediction = bool(native.get("uses_rgbd_pose_mesh_for_export", True))
    gate = {
        "v62_decision_label_is_GO_V62_VERIFIED_OWNERSHIP_FIELD": decision_label
        == "GO_V62_VERIFIED_OWNERSHIP_FIELD",
        "claim_A_status_pass": _claim_status(final_decision, "Claim A") == "pass",
        "core_purity_ge_0_95": core_purity >= 0.95,
        "core_completeness_ge_0_90": core_completeness >= 0.90,
        "state_coverage_rate_ge_0_95": state_coverage_rate >= 0.95,
        "real_minus_shuffled_ARI_ge_0_30": real_minus_shuffled_ari >= 0.30,
        "real_minus_no_temporal_ARI_ge_0_25": real_minus_no_temporal_ari >= 0.25,
        "uses_gt_for_prediction_false": not uses_gt_for_prediction,
        "uses_rgbd_pose_mesh_for_prediction_false": not uses_rgbd_pose_mesh_for_prediction,
    }
    gate["pass"] = bool(all(gate.values()))
    metric_row = {
        "row_id": "A0_v62_verified_ownership_field",
        "v62_decision_label": decision_label,
        "claim_A_status": _claim_status(final_decision, "Claim A"),
        "claim_B_status": _claim_status(final_decision, "Claim B"),
        "claim_C_status": _claim_status(final_decision, "Claim C"),
        "claim_D_status": _claim_status(final_decision, "Claim D"),
        "core_purity": core_purity,
        "core_completeness": core_completeness,
        "expanded_completeness": expanded_completeness,
        "state_coverage_rate": state_coverage_rate,
        "confirmed_count": native.get("confirmed_component_count"),
        "tentative_count": native.get("tentative_carrier_count"),
        "shared_count": native.get("shared_carrier_count"),
        "quarantine_count": native.get("quarantine_carrier_count"),
        "unknown_count": native.get("unknown_carrier_count"),
        "real_minus_shuffled_ARI": real_minus_shuffled_ari,
        "real_minus_no_temporal_ARI": real_minus_no_temporal_ari,
        "same_category_merge_rate": solver.get("same_category_merge_rate"),
        "underseg_false_merge_rate": solver.get("underseg_false_merge_rate"),
        "new_material_gain_vs_anchor_only": increment.get("new_material_gain_vs_anchor_only"),
        "stress_pass_count": stress.get("stress_regen_real_minus_mask_only_pass_count"),
        "uses_gt_for_prediction": uses_gt_for_prediction,
        "uses_rgbd_pose_mesh_for_prediction": uses_rgbd_pose_mesh_for_prediction,
        "gate_pass": gate["pass"],
    }
    return {
        "phase": "v64r2_phaseA0_main_fact_lock",
        "created_at": utc_now(),
        "input_paths": {
            "v62_final": _rel(v62_final_path),
            "v62_solver": _rel(v62_solver_summary_path),
            "v62_native": _rel(v62_native_summary_path),
            "v62_increment": _rel(v62_increment_summary_path),
            "v62_stress": _rel(v62_stress_summary_path),
        },
        "summary": {
            "main_ownership_status": "GO_MAIN_OWNERSHIP_FIELD" if gate["pass"] else "NO_GO_MAIN_OWNERSHIP_FIELD",
            "v62_decision_label": decision_label,
            "core_purity": core_purity,
            "core_completeness": core_completeness,
            "state_coverage_rate": state_coverage_rate,
            "real_minus_shuffled_ARI": real_minus_shuffled_ari,
            "real_minus_no_temporal_ARI": real_minus_no_temporal_ari,
            "uses_gt_for_prediction": uses_gt_for_prediction,
            "uses_rgbd_pose_mesh_for_prediction": uses_rgbd_pose_mesh_for_prediction,
        },
        "gate": gate,
        "metric_rows": [metric_row],
        "source_final_decision": final_decision,
    }


def write_v64r2_main_fact_lock(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "main_fact_lock_summary.json", payload)
    write_csv(out / "main_metric_rows.csv", payload["metric_rows"])
