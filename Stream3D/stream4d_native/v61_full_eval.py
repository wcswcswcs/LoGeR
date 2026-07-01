from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v47_common import ROOT, utc_now, write_csv, write_json


@dataclass(frozen=True)
class V61FullEvalConfig:
    phase0_summary_path: str | Path = "outputs/audit/v61_phase0_failure_lock/failure_lock.json"
    graph_summary_path: str | Path = "outputs/audit/v61_graph_v3/graph_v3_summary.json"
    embedding_summary_path: str | Path = "outputs/audit/v61_global_embedding/embedding_summary.json"
    refinement_summary_path: str | Path = "outputs/audit/v61_refinement/refinement_summary.json"
    query_summary_path: str | Path = "outputs/audit/v61_manifold_query/query_summary.json"
    stress_summary_path: str | Path = "outputs/audit/v61_stress/stress_summary.json"
    native_summary_path: str | Path = "outputs/audit/v61_native_field/native_field_summary.json"
    v56_core_summary_path: str | Path = "outputs/audit/v56_core_update/core_update_summary.json"
    v56_tentative_summary_path: str | Path = "outputs/audit/v56_tentative_support/tentative_support_summary.json"
    output_root: str | Path = "outputs/audit/v61_final_decision"


def build_v61_final_decision(config: V61FullEvalConfig | None = None) -> dict[str, Any]:
    cfg = config or V61FullEvalConfig()
    phase0 = _read_json(cfg.phase0_summary_path)
    graph = _read_json(cfg.graph_summary_path)
    embedding = _read_json(cfg.embedding_summary_path)
    refinement = _read_json(cfg.refinement_summary_path)
    query = _read_json(cfg.query_summary_path)
    stress = _read_json(cfg.stress_summary_path)
    native = _read_json(cfg.native_summary_path)
    v56_core = _read_json_if_exists(cfg.v56_core_summary_path)
    v56_tentative = _read_json_if_exists(cfg.v56_tentative_summary_path)

    global_pass = _gate_pass(embedding)
    graph_pass = _gate_pass(graph)
    phase0_pass = _gate_pass(phase0)
    separation_pass = (
        float(embedding.get("same_category_merge_rate", 1.0)) <= 0.05
        and (refinement.get("quarantine_precision_diagnostic") is not None)
        and float(refinement.get("quarantine_precision_diagnostic", 0.0)) >= 0.80
    )
    stress_pass = _gate_pass(stress)
    native_pass = _gate_pass(native)
    query_pass = _gate_pass(query)
    refinement_pass = _gate_pass(refinement)
    core_go = bool(phase0_pass and graph_pass and global_pass and separation_pass and stress_pass and native_pass)
    decision_label = _decision_label(
        phase0_pass=phase0_pass,
        graph_pass=graph_pass,
        global_pass=global_pass,
        separation_pass=separation_pass,
        stress_pass=stress_pass,
        native_pass=native_pass,
        core_go=core_go,
    )
    blocked_claims = []
    if not refinement_pass:
        blocked_claims.append("refinement_core_gain_and_promotion_claim")
    if not query_pass:
        blocked_claims.append("active_material_query_claim")
    if native.get("ap_diagnostic_status") != "run":
        blocked_claims.append("native_AP_or_mesh_materialization_claim")
    if query.get("method_note", "").find("not a newly executed D4RT tracker pass") >= 0:
        blocked_claims.append("new_D4RT_query_tracking_claim")

    final_rows = _final_rows(embedding, refinement, query, stress, native, v56_core, v56_tentative)
    summary = {
        "phase": "v61_final_decision",
        "created_at": utc_now(),
        "decision_label": decision_label,
        "core_global_embedding_go": core_go,
        "query_gate_pass": query_pass,
        "refinement_gate_pass": refinement_pass,
        "blocked_claims": blocked_claims,
        "go_gate": {
            "phase0_unit_mismatch_lock_pass": phase0_pass,
            "graph_v3_candidate_gate_pass": graph_pass,
            "global_embedding_gate_pass": global_pass,
            "separation_gate_pass": separation_pass,
            "stress_gate_pass": stress_pass,
            "native_field_gate_pass": native_pass,
            "pass": core_go,
        },
        "required_answers": _required_answers(phase0, graph, embedding, refinement, query, stress, native, decision_label),
        "key_metrics": {
            "core_purity": embedding.get("core_purity"),
            "core_completeness": embedding.get("core_completeness"),
            "expanded_completeness": embedding.get("expanded_completeness"),
            "real_minus_shuffled_ARI": embedding.get("real_minus_shuffled_ARI"),
            "real_minus_no_temporal_ARI": embedding.get("real_minus_no_temporal_ARI"),
            "same_category_merge_rate": embedding.get("same_category_merge_rate"),
            "shortcut_quarantine_precision": refinement.get("quarantine_precision_diagnostic"),
            "query_to_confirm_or_quarantine_rate": query.get("query_to_confirm_or_quarantine_rate"),
            "stress_real_minus_mask_only_ARI_pass_count": stress.get("stress_real_minus_mask_only_ARI_pass_count"),
            "native_field_available": native.get("method_safe_native_support_available"),
            "best_AP_diagnostic": None,
        },
        "input_paths": {
            "phase0_summary": _rel(cfg.phase0_summary_path),
            "graph_summary": _rel(cfg.graph_summary_path),
            "embedding_summary": _rel(cfg.embedding_summary_path),
            "refinement_summary": _rel(cfg.refinement_summary_path),
            "query_summary": _rel(cfg.query_summary_path),
            "stress_summary": _rel(cfg.stress_summary_path),
            "native_summary": _rel(cfg.native_summary_path),
            "v56_core_summary": _rel(cfg.v56_core_summary_path),
            "v56_tentative_summary": _rel(cfg.v56_tentative_summary_path),
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"summary": summary, "final_metric_rows": final_rows}


def write_v61_final_decision(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "final_decision": root / "final_decision.json",
        "final_metric_rows": root / "final_metric_rows.csv",
    }
    write_json(paths["final_decision"], result["summary"])
    write_csv(paths["final_metric_rows"], result["final_metric_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def _decision_label(
    *,
    phase0_pass: bool,
    graph_pass: bool,
    global_pass: bool,
    separation_pass: bool,
    stress_pass: bool,
    native_pass: bool,
    core_go: bool,
) -> str:
    if core_go:
        return "GO_SOMA_MANIFOLD_GLOBAL_EMBEDDING"
    if not phase0_pass or not graph_pass:
        return "NO_GO_CANDIDATE_COVERAGE"
    if not global_pass:
        return "NO_GO_EMBEDDING"
    if not separation_pass:
        return "NO_GO_SHORTCUT_SEPARATION"
    if not stress_pass:
        return "NO_GO_DYNAMIC_READY"
    if not native_pass:
        return "NO_GO_NATIVE_FIELD"
    return "PARTIAL_GLOBAL_EMBEDDING_SIGNAL"


def _final_rows(
    embedding: dict[str, Any],
    refinement: dict[str, Any],
    query: dict[str, Any],
    stress: dict[str, Any],
    native: dict[str, Any],
    v56_core: dict[str, Any],
    v56_tentative: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_variant = next((row for row in embedding.get("variant_rows", []) if row.get("variant") == embedding.get("selected_variant")), None)
    variant_baseline = selected_variant or (embedding.get("variant_rows", [{}])[0] if embedding.get("variant_rows") else {})
    rows = [
        {
            "row": "F0_v56_confirmed_core",
            "core_purity": v56_core.get("core_purity"),
            "core_completeness": v56_core.get("core_completeness") or variant_baseline.get("v56_core_completeness"),
            "expanded_completeness": "",
            "real_minus_shuffled_ARI": v56_core.get("real_minus_shuffled_ARI") or variant_baseline.get("v56_core_real_minus_shuffled_ARI"),
            "real_minus_no_temporal_ARI": v56_core.get("real_minus_no_temporal_ARI") or variant_baseline.get("v56_core_real_minus_no_temporal_ARI"),
        },
        {
            "row": "F1_v56_expanded_tentative",
            "core_purity": v56_tentative.get("expanded_purity"),
            "core_completeness": "",
            "expanded_completeness": v56_tentative.get("expanded_completeness") or variant_baseline.get("v56_expanded_completeness"),
            "real_minus_shuffled_ARI": "",
            "real_minus_no_temporal_ARI": "",
        },
        {
            "row": "F4_v61_global_embedding",
            "core_purity": embedding.get("core_purity"),
            "core_completeness": embedding.get("core_completeness"),
            "expanded_completeness": embedding.get("expanded_completeness"),
            "real_minus_shuffled_ARI": embedding.get("real_minus_shuffled_ARI"),
            "real_minus_no_temporal_ARI": embedding.get("real_minus_no_temporal_ARI"),
            "same_category_merge_rate": embedding.get("same_category_merge_rate"),
            "underseg_false_merge_rate": embedding.get("underseg_false_merge_rate"),
        },
        {
            "row": "F6_v61_refined_manifold",
            "core_purity": refinement.get("core_purity"),
            "core_completeness": refinement.get("core_completeness"),
            "expanded_completeness": refinement.get("expanded_completeness"),
            "shortcut_quarantine_precision": refinement.get("quarantine_precision_diagnostic"),
            "refinement_gate_pass": _gate_pass(refinement),
        },
        {
            "row": "F7_v61_full_plus_query",
            "query_gate_pass": _gate_pass(query),
            "query_to_confirm_or_quarantine_rate": query.get("query_to_confirm_or_quarantine_rate"),
            "valid_material_evidence_rate": query.get("valid_material_evidence_rate"),
            "state_entropy_reduction": query.get("state_entropy_reduction"),
        },
        {
            "row": "F_stress_native",
            "stress_pass_count": stress.get("stress_real_minus_mask_only_ARI_pass_count"),
            "native_field_available": native.get("method_safe_native_support_available"),
            "best_AP_diagnostic": "",
            "ap_diagnostic_status": native.get("ap_diagnostic_status"),
        },
    ]
    for row in rows:
        row["uses_gt_for_prediction"] = False
        row["uses_gt_for_diagnostic_labels"] = True
    return rows


def _required_answers(
    phase0: dict[str, Any],
    graph: dict[str, Any],
    embedding: dict[str, Any],
    refinement: dict[str, Any],
    query: dict[str, Any],
    stress: dict[str, Any],
    native: dict[str, Any],
    decision_label: str,
) -> dict[str, str]:
    return {
        "q1_v60_failure_unit_mismatch": f"{_gate_pass(phase0)}; material_state_coverage_rate={phase0.get('material_state_coverage_rate')}",
        "q2_material_candidates_cover_enough_nodes": f"{_gate_pass(graph)}; material_nodes_with_candidate_rate={graph.get('material_nodes_with_candidate_rate')}",
        "q3_global_embedding_improves_completeness": f"{_gate_pass(embedding)}; core={embedding.get('core_completeness')}, expanded={embedding.get('expanded_completeness')}",
        "q4_core_purity_and_controls": f"{_gate_pass(embedding)}; purity={embedding.get('core_purity')}, real_minus_shuffled={embedding.get('real_minus_shuffled_ARI')}, real_minus_no_temporal={embedding.get('real_minus_no_temporal_ARI')}",
        "q5_underseg_shortcut_shared_quarantine": f"{refinement.get('quarantine_precision_diagnostic')}; quarantined={refinement.get('quarantined_node_count')}",
        "q6_same_category_merge": f"{embedding.get('same_category_merge_rate')}",
        "q7_refinement_safe_promotion": f"{_gate_pass(refinement)}; promoted={refinement.get('promoted_node_count')}, core_gain={refinement.get('core_purity_gain')}",
        "q8_query_beats_baselines": f"{_gate_pass(query)}; rate={query.get('query_to_confirm_or_quarantine_rate')}, best_fixed={query.get('best_fixed_query_to_confirm_or_quarantine_rate')}",
        "q9_stress_beats_mask_only_v56": f"{_gate_pass(stress)}; mask_only_pass_count={stress.get('stress_real_minus_mask_only_ARI_pass_count')}, v56_pass_count={stress.get('stress_real_minus_v56_expanded_ARI_pass_count')}",
        "q10_native_field_method_safe": f"{_gate_pass(native)}; available={native.get('method_safe_native_support_available')}",
        "q11_failure_layer_if_any": f"{decision_label}; blocked_claims are in final_decision.json",
    }


def _gate_pass(summary: dict[str, Any]) -> bool:
    return bool((summary.get("gate") or {}).get("pass"))


def _read_json(path: str | Path) -> dict[str, Any]:
    path_obj = _project(path)
    return json.loads(path_obj.read_text(encoding="utf-8"))


def _read_json_if_exists(path: str | Path) -> dict[str, Any]:
    path_obj = _project(path)
    if not path_obj.exists():
        return {}
    return json.loads(path_obj.read_text(encoding="utf-8"))


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)
