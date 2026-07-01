from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


@dataclass(frozen=True)
class V60FinalDecisionConfig:
    output_root: str | Path = "outputs/audit/v60_final_decision"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def build_v60_final_decision(config: V60FinalDecisionConfig | None = None) -> dict[str, Any]:
    fact = read_json(ROOT / "outputs/audit/v60_phase0_fact_lock/fact_lock.json")
    graph = read_json(ROOT / "outputs/audit/v60_graph_v2/graph_summary.json")
    path = read_json(ROOT / "outputs/audit/v60_manifold_paths_v2/path_summary.json")
    embedding = read_json(ROOT / "outputs/audit/v60_manifold_embedding/embedding_summary.json")
    refinement = read_json(ROOT / "outputs/audit/v60_manifold_refinement/refinement_summary.json")
    query = read_json(ROOT / "outputs/audit/v60_manifold_query/query_summary.json")
    v56_core = read_json(ROOT / "outputs/audit/v56_core_update/core_update_summary.json")
    v56_tentative = read_json(ROOT / "outputs/audit/v56_tentative_support/tentative_support_summary.json")
    v58 = read_json(ROOT / "outputs/audit/v58_counterfactual_explanation_dino_full_repair6/explanation_summary.json")
    v59_final = read_json(ROOT / "outputs/audit/v59_final_decision/final_decision.json")

    rows = [
        _row("F0_v56_confirmed_core", v56_core.get("core_ARI"), v56_core.get("core_purity"), v56_core.get("core_completeness"), None, None, None, v56_core.get("history_temporal_span_mean"), None, None, None, None, None, None, v56_core.get("real_minus_shuffled_ARI"), v56_core.get("real_minus_no_temporal_ARI"), v56_core.get("real_minus_mask_only_ARI"), None, None, "v56 core baseline"),
        _row("F1_v56_expanded_tentative", v56_tentative.get("confirmed_core_ARI"), v56_tentative.get("confirmed_core_purity"), v56_tentative.get("confirmed_core_completeness"), v56_tentative.get("expanded_ARI"), v56_tentative.get("expanded_purity"), v56_tentative.get("expanded_completeness"), None, None, v56_tentative.get("tentative_underseg_rate"), None, None, None, None, None, None, None, None, None, "v56 expanded baseline"),
        _row("F2_v58_SOMA_diagnostic", v58.get("ARI"), v58.get("purity"), v58.get("completeness"), None, None, None, v58.get("temporal_span_mean"), None, None, None, None, None, None, v58.get("real_minus_shuffled_ARI"), v58.get("real_minus_no_temporal_ARI"), v58.get("real_minus_mask_only_ARI"), None, None, "diagnostic observation-support projection"),
        _row("F3_v59_path_only_manifold", None, None, None, None, None, None, None, v59_final.get("phase2_same_category_false_path_rate"), None, v59_final.get("phase2_shortcut_quarantine_precision"), None, None, None, None, None, None, None, None, v59_final.get("partial_label")),
        _row("F4_v60_calibrated_path", None, None, None, None, None, path.get("path_recall_proxy"), None, path.get("same_category_false_path_rate_calibrated"), None, path.get("shortcut_quarantine_precision"), None, None, None, None, None, None, None, None, "Phase2 path/shortcut pass under calibrated gate"),
        _row("F5_v60_global_embedding", embedding.get("core_ARI"), embedding.get("core_purity"), embedding.get("core_completeness"), embedding.get("expanded_ARI"), embedding.get("expanded_purity"), embedding.get("expanded_completeness"), embedding.get("temporal_span_mean"), embedding.get("same_category_merge_rate"), embedding.get("underseg_false_merge_rate"), path.get("shortcut_quarantine_precision"), None, None, None, embedding.get("real_minus_shuffled_ARI"), embedding.get("real_minus_no_temporal_ARI"), embedding.get("real_minus_mask_only_ARI_static"), None, False, "Phase3 gate fail"),
        _row("F6_v60_embedding_refinement", embedding.get("core_ARI"), embedding.get("core_purity"), embedding.get("core_completeness"), embedding.get("expanded_ARI"), embedding.get("expanded_purity"), embedding.get("expanded_completeness"), embedding.get("temporal_span_mean"), embedding.get("same_category_merge_rate"), embedding.get("underseg_false_merge_rate"), refinement.get("quarantine_precision_diagnostic"), None, None, None, embedding.get("real_minus_shuffled_ARI"), embedding.get("real_minus_no_temporal_ARI"), embedding.get("real_minus_mask_only_ARI_static"), None, False, "diagnostic bypass; refinement gate fail"),
        _row("F7_v60_full_plus_query", embedding.get("core_ARI"), embedding.get("core_purity"), embedding.get("core_completeness"), embedding.get("expanded_ARI"), embedding.get("expanded_purity"), embedding.get("expanded_completeness"), embedding.get("temporal_span_mean"), embedding.get("same_category_merge_rate"), embedding.get("underseg_false_merge_rate"), refinement.get("quarantine_precision_diagnostic"), None, query.get("valid_material_evidence_rate"), None, embedding.get("real_minus_shuffled_ARI"), embedding.get("real_minus_no_temporal_ARI"), embedding.get("real_minus_mask_only_ARI_static"), None, False, "diagnostic bypass; query gate fail"),
    ]

    success_criteria = {
        "calibrated_same_category_gate": bool((fact.get("gate") or {}).get("same_category_calibrated_gate_pass")),
        "path_and_shortcut": bool((path.get("gate") or {}).get("pass")),
        "embedding": bool((embedding.get("gate") or {}).get("pass")),
        "controls": bool((embedding.get("gate") or {}).get("real_minus_shuffled_ARI_ge_v56_core_plus_0_03") and (embedding.get("gate") or {}).get("real_minus_no_temporal_ARI_ge_v56_core_plus_0_02")),
        "refinement_query": bool((refinement.get("gate") or {}).get("pass") and (query.get("gate") or {}).get("pass")),
        "stress": False,
        "native_field": False,
    }
    decision = {
        "phase": "v60_final_decision",
        "created_at": utc_now(),
        "goal_achieved": False,
        "final_label": "NO_GO_EMBEDDING",
        "partial_label": "PARTIAL_GRAPH_PATH_SIGNAL",
        "first_hard_blocker": "Phase3 global embedding fails completeness/control gate after plan-directed repair.",
        "secondary_blockers": [
            "Phase4 refinement diagnostic-only fails quarantine precision and cannot safely promote tentative nodes.",
            "Phase5 manifold-aware query diagnostic-only fails all query gates and does not beat fixed baselines.",
            "Phase6 stress and Phase7 native field method claims are blocked because embedding/query gates failed.",
        ],
        "success_criteria": success_criteria,
        "answers_to_report_questions": [
            "v59 No-Go was caused by the old same-category gate using an impossible negative required max rate under a low baseline; v60 calibrated same-category gate passes with 0/103 false paths and Wilson upper95=0.03595475822864816.",
            "v60 calibrated same-category gate passes.",
            "typed graph v2 preserves the locked invariants: graph_v2 gate passes, shortcut_edge_count=2648, hard_constraint_violation_count=0.",
            "part-to-core path and shortcut quarantine remain reliable at Phase2: path precision=0.8543689320388349 and shortcut precision=0.9537037037037037.",
            "global manifold embedding is not better as a final method gate: repaired core purity passes but core/expanded completeness and controls fail.",
            "confirmed core does not exceed v56 core; v60 core completeness=0.021058315334773217 vs v56 core=0.6189200933776642, and expanded completeness=0.1900647948164147 vs v56 expanded=0.6814716751050162.",
            "underseg/shortcut handling is not a full pass: Phase2 shortcut quarantine is reliable, but Phase3/4 quarantine overcuts with diagnostic precision=0.5052083333333334.",
            "manifold-aware Q7 query does not beat random/boundary/v58 Q6; valid material evidence=0.0078125 and query_to_confirm_or_quarantine=0.015625.",
            "stress under mask-only/v58 was not run as a method claim because embedding and query gates failed.",
            "native semantic-material field was not exported as a method-safe success claim because embedding/query gates failed.",
            "The primary failure layer is embedding; query is a secondary diagnostic failure; stress/native are blocked downstream claims.",
        ],
        "analysis_conclusions": [
            "The v60 calibration repair is valid: the old v59 same-category gate was mathematically overstrict for a 0.0268 baseline false rate.",
            "Typed graph and path evidence are real partial signals, but they only cover 412 accepted paths out of 1852 diagnostic observations.",
            "Moving low-margin paths to tentative repairs purity but destroys completeness; this is a precision/coverage bottleneck rather than a code crash.",
            "Margin-only refinement is unsafe and material-query evidence is too sparse on the v60 tentative/quarantine subset.",
            "The honest closure is No-Go embedding with partial graph/path signal, not a successful SOMA-Manifold v2 method.",
        ],
        "input_paths": {
            "fact_lock": "outputs/audit/v60_phase0_fact_lock/fact_lock.json",
            "graph_summary": "outputs/audit/v60_graph_v2/graph_summary.json",
            "path_summary": "outputs/audit/v60_manifold_paths_v2/path_summary.json",
            "embedding_summary": "outputs/audit/v60_manifold_embedding/embedding_summary.json",
            "refinement_summary": "outputs/audit/v60_manifold_refinement/refinement_summary.json",
            "query_summary": "outputs/audit/v60_manifold_query/query_summary.json",
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"decision": decision, "final_eval_rows": rows}


def write_v60_final_decision(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "final_decision": root / "final_decision.json",
        "final_eval_rows": root / "final_eval_rows.csv",
    }
    write_json(paths["final_decision"], result["decision"])
    write_csv(paths["final_eval_rows"], result["final_eval_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def _row(
    row: str,
    core_ari: Any,
    core_purity: Any,
    core_completeness: Any,
    expanded_ari: Any,
    expanded_purity: Any,
    expanded_completeness: Any,
    temporal_span_mean: Any,
    same_category_false_path_rate: Any,
    underseg_false_merge_rate: Any,
    shortcut_quarantine_precision: Any,
    promotion_precision: Any,
    query_valid_rate: Any,
    stress_pass_count: Any,
    real_minus_shuffled_ari: Any,
    real_minus_no_temporal_ari: Any,
    real_minus_mask_only_ari_static: Any,
    best_ap_diagnostic: Any,
    native_field_available: Any,
    note: str,
) -> dict[str, Any]:
    return {
        "row": row,
        "core_ARI": core_ari,
        "core_purity": core_purity,
        "core_completeness": core_completeness,
        "expanded_ARI": expanded_ari,
        "expanded_purity": expanded_purity,
        "expanded_completeness": expanded_completeness,
        "temporal_span_mean": temporal_span_mean,
        "same_category_false_path_rate": same_category_false_path_rate,
        "underseg_false_merge_rate": underseg_false_merge_rate,
        "shortcut_quarantine_precision": shortcut_quarantine_precision,
        "promotion_precision": promotion_precision,
        "query_valid_rate": query_valid_rate,
        "stress_pass_count": stress_pass_count,
        "real_minus_shuffled_ARI": real_minus_shuffled_ari,
        "real_minus_no_temporal_ARI": real_minus_no_temporal_ari,
        "real_minus_mask_only_ARI_static": real_minus_mask_only_ari_static,
        "native_field_available": native_field_available,
        "best_AP_diagnostic": best_ap_diagnostic,
        "note": note,
    }
