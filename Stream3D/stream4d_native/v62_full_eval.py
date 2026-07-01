from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


SUMMARY_PATHS = {
    "integrity": "outputs/audit/v62_phase0_integrity/integrity_summary.json",
    "decircularization": "outputs/audit/v62_decircularization/decircularization_summary.json",
    "increment": "outputs/audit/v62_increment_attribution/increment_summary.json",
    "solver": "outputs/audit/v62_solver_v2/solver_summary.json",
    "refinement": "outputs/audit/v62_refinement_robustness/refinement_summary.json",
    "query": "outputs/audit/v62_active_query_refresh/query_refresh_summary.json",
    "stress": "outputs/audit/v62_stress_regen/stress_regen_summary.json",
    "native": "outputs/audit/v62_native_field/native_field_summary.json",
    "v56_core": "outputs/audit/v56_core_update/core_update_summary.json",
    "v56_tentative": "outputs/audit/v56_tentative_support/tentative_support_summary.json",
    "v61_final": "outputs/audit/v61_final_decision/final_decision.json",
}


@dataclass(frozen=True)
class V62FullEvalConfig:
    output_root: str | Path = "outputs/audit/v62_final"
    visualization_root: str | Path = "outputs/audit/v62_visualizations"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def build_v62_final_decision(config: V62FullEvalConfig | None = None) -> dict[str, Any]:
    cfg = config or V62FullEvalConfig()
    summaries = {name: _read(path) for name, path in SUMMARY_PATHS.items()}
    claim_a = _gate(summaries["integrity"]) and _gate(summaries["decircularization"]) and _gate(summaries["solver"]) and _gate(summaries["native"])
    claim_b = _gate(summaries["refinement"])
    claim_c = _gate(summaries["query"])
    claim_d = _gate(summaries["stress"])
    blocked_claims = []
    if not claim_b:
        blocked_claims.append("robust_manifold_refinement_claim")
    if not claim_c:
        blocked_claims.append("active_query_refresh_claim")
    if not claim_d:
        blocked_claims.append("dynamic_ready_regen_stress_claim")
    if summaries["native"].get("carrier_level_field_available") is False:
        blocked_claims.append("true_carrier_level_native_field_claim")
    if summaries["native"].get("ap_diagnostic_status") != "method_safe":
        blocked_claims.append("native_AP_or_mesh_materialization_claim")

    decision_label = "GO_V62_VERIFIED_OWNERSHIP_FIELD" if claim_a else _no_go_label(summaries)
    final_rows = _final_rows(summaries)
    final_decision = {
        "phase": "v62_final",
        "created_at": utc_now(),
        "decision_label": decision_label,
        "claim_table": {
            "Claim A": {
                "label": "GO_VERIFIED_GLOBAL_OWNERSHIP_FIELD",
                "pass": claim_a,
                "evidence": "Phase0 integrity + Phase1 de-circularization + Phase3 solver + Phase7 native component field",
            },
            "Claim B": {
                "label": "GO_ROBUST_MANIFOLD_REFINEMENT" if claim_b else "PARTIAL_REFINEMENT_ROBUSTNESS",
                "pass": claim_b,
                "evidence": "Phase4 perturbation-row robustness metrics",
            },
            "Claim C": {
                "label": "GO_ACTIVE_QUERY_REFRESH" if claim_c else "PARTIAL_QUERY_REFRESH_SIGNAL",
                "pass": claim_c,
                "evidence": "Phase5A candidate pool only; no real D4RT query outcomes",
            },
            "Claim D": {
                "label": "GO_DYNAMIC_READY_REGEN_STRESS" if claim_d else "NO_GO_STRESS_REGEN",
                "pass": claim_d,
                "evidence": "Phase6 lightweight graph-regeneration stress",
            },
        },
        "blocked_claims": blocked_claims,
        "key_metrics": {
            "core_purity": summaries["solver"].get("full_solver_core_purity"),
            "core_completeness": summaries["solver"].get("full_solver_core_completeness"),
            "real_minus_shuffled_ARI": summaries["solver"].get("full_solver_real_minus_shuffled_ARI"),
            "real_minus_no_temporal_ARI": summaries["solver"].get("full_solver_real_minus_no_temporal_ARI"),
            "state_coverage": _state_coverage_from_solver(),
            "new_material_gain": summaries["increment"].get("new_material_gain_vs_anchor_only"),
            "stress_pass_count": summaries["stress"].get("stress_regen_real_minus_mask_only_pass_count"),
            "native_field_available": summaries["native"].get("component_level_field_available"),
            "ap_status": summaries["native"].get("ap_diagnostic_status"),
        },
        "required_answers": _required_answers(summaries, claim_a, claim_b, claim_c, claim_d),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "input_paths": SUMMARY_PATHS,
    }
    return {"final_decision": final_decision, "final_metric_rows": final_rows}


def write_v62_final_decision(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "final_decision": root / "final_decision.json",
        "final_metric_rows": root / "final_metric_rows.csv",
    }
    write_json(paths["final_decision"], result["final_decision"])
    write_csv(paths["final_metric_rows"], result["final_metric_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def _read(path: str) -> dict[str, Any]:
    full = ROOT / path
    return read_json(full) if full.exists() else {}


def _gate(summary: dict[str, Any]) -> bool:
    return bool((summary.get("gate") or {}).get("pass"))


def _no_go_label(summaries: dict[str, dict[str, Any]]) -> str:
    if not _gate(summaries["decircularization"]):
        return "NO_GO_DECIRCULARIZATION"
    if not _gate(summaries["increment"]):
        return "NO_GO_INCREMENTAL_LOCAL2HISTORY"
    if not _gate(summaries["solver"]):
        return "NO_GO_SOLVER_V2"
    if not _gate(summaries["native"]):
        return "NO_GO_NATIVE_FIELD"
    return "PARTIAL_V62_UNRESOLVED"


def _final_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    v56_core = summaries["v56_core"]
    v56_tent = summaries["v56_tentative"]
    v61 = summaries["v61_final"].get("key_metrics", {})
    solver = summaries["solver"]
    query = summaries["query"]
    stress = summaries["stress"]
    native = summaries["native"]
    return [
        _row("F0_v56_confirmed_core", v56_core.get("core_purity"), v56_core.get("core_completeness"), None, v56_core.get("real_minus_shuffled_ARI"), v56_core.get("real_minus_no_temporal_ARI"), None, None, None, None, None, native),
        _row("F1_v56_expanded", v56_tent.get("expanded_purity"), v56_tent.get("expanded_completeness"), v56_tent.get("expanded_completeness"), None, None, None, None, None, None, None, native),
        _row("F2_v61_global_embedding_original", v61.get("core_purity"), v61.get("core_completeness"), v61.get("expanded_completeness"), v61.get("real_minus_shuffled_ARI"), v61.get("real_minus_no_temporal_ARI"), None, v61.get("same_category_merge_rate"), None, None, None, native),
        _row("F3_v62_decircularized_solver", solver.get("full_solver_core_purity"), solver.get("full_solver_core_completeness"), solver.get("full_solver_core_completeness"), solver.get("full_solver_real_minus_shuffled_ARI"), solver.get("full_solver_real_minus_no_temporal_ARI"), None, None, None, _state_coverage_from_solver(), summaries["increment"].get("new_material_gain_vs_anchor_only"), native),
        _row("F4_v62_solver_refinement_robustness", solver.get("full_solver_core_purity"), solver.get("full_solver_core_completeness"), solver.get("full_solver_core_completeness"), solver.get("full_solver_real_minus_shuffled_ARI"), solver.get("full_solver_real_minus_no_temporal_ARI"), None, None, None, _state_coverage_from_solver(), summaries["increment"].get("new_material_gain_vs_anchor_only"), native),
        _row("F5_v62_active_query_refresh", None, None, None, query.get("real_minus_shuffled_query_AUC"), query.get("real_minus_no_temporal_query_AUC"), None, None, None, None, None, native),
        _row("F6_v62_graph_regeneration_stress", solver.get("full_solver_core_purity"), solver.get("full_solver_core_completeness"), solver.get("full_solver_core_completeness"), solver.get("full_solver_real_minus_shuffled_ARI"), solver.get("full_solver_real_minus_no_temporal_ARI"), None, None, None, _state_coverage_from_solver(), summaries["increment"].get("new_material_gain_vs_anchor_only"), native, stress_pass_count=stress.get("stress_regen_real_minus_mask_only_pass_count")),
        _row("F7_mask_only_control", None, None, None, None, None, None, None, None, None, None, native),
        _row("F8_shuffled_D4RT_control", None, None, None, 0.0, None, None, None, None, None, None, native),
        _row("F9_no_temporal_control", None, None, None, None, 0.0, None, None, None, None, None, native),
        _row("F10_semantic_only_control", None, None, None, None, None, None, None, None, None, None, native),
    ]


def _row(
    row_id: str,
    core_purity: Any,
    core_completeness: Any,
    expanded_completeness: Any,
    real_minus_shuffled_ARI: Any,
    real_minus_no_temporal_ARI: Any,
    real_minus_mask_only_ARI: Any,
    same_category_merge_rate: Any,
    underseg_false_merge_rate: Any,
    state_coverage: Any,
    new_material_gain: Any,
    native: dict[str, Any],
    *,
    stress_pass_count: Any = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "core_purity": core_purity,
        "core_completeness": core_completeness,
        "expanded_completeness": expanded_completeness,
        "real_minus_shuffled_ARI": real_minus_shuffled_ARI,
        "real_minus_no_temporal_ARI": real_minus_no_temporal_ARI,
        "real_minus_mask_only_ARI": real_minus_mask_only_ARI,
        "same_category_merge_rate": same_category_merge_rate,
        "underseg_false_merge_rate": underseg_false_merge_rate,
        "state_coverage": state_coverage,
        "new_material_gain": new_material_gain,
        "stress_pass_count": stress_pass_count,
        "native_field_available": native.get("component_level_field_available"),
        "ap_status": native.get("ap_diagnostic_status"),
        "uses_gt_for_prediction": False,
    }


def _state_coverage_from_solver() -> float | None:
    path = ROOT / "outputs/audit/v62_solver_v2/energy_rows.csv"
    if not path.exists():
        return None
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == "S6_full_SOMA_Manifold_v2_solver":
                material = float(row.get("material_node_count") or 0.0)
                unknown = float(row.get("unknown_material_count") or 0.0)
                return None if material == 0 else (material - unknown) / material
    return None


def _required_answers(summaries: dict[str, dict[str, Any]], claim_a: bool, claim_b: bool, claim_c: bool, claim_d: bool) -> dict[str, str]:
    native = summaries["native"]
    return {
        "q1_v61_zip_artifact_complete": str(_gate(summaries["integrity"])),
        "q2_de_circularized_GO_still_holds": str(_gate(summaries["decircularization"])),
        "q3_K_mat_provenance_prediction_safe": "method candidate evidence from material_continuity; see v62_decircularization/provenance_rows.csv",
        "q4_material_gain_anchor_or_update": f"new_material_gain_vs_anchor_only={summaries['increment'].get('new_material_gain_vs_anchor_only')}",
        "q5_M1_sufficient_or_complex_solver_needed": summaries["solver"].get("solver_complexity_claim", ""),
        "q6_refinement_stress_value": str(claim_b),
        "q7_active_query_refresh_status": summaries["query"].get("claim_status", ""),
        "q8_graph_regeneration_stress_dynamic_ready": str(claim_d),
        "q9_native_field_level": native.get("native_field_limitation", ""),
        "q10_AP_status": str(native.get("ap_diagnostic_status")),
        "q11_failure_layer": "active_query_refresh" if not claim_c else ("none_for_claim_A" if claim_a else "claim_A_blocked"),
    }


