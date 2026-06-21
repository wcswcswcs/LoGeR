from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _read_json(path: str | Path) -> dict[str, Any]:
    with _project(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _gate_pass(summary: dict[str, Any]) -> bool:
    gate = summary.get("gate")
    return bool(isinstance(gate, dict) and gate.get("pass") is True)


def _phase4_row(name: str, path: str) -> dict[str, Any]:
    summary = _read_json(path)
    return {
        "name": name,
        "path": path,
        "gate_pass": _gate_pass(summary),
        "history_update_variant": summary.get("history_update_variant"),
        "history_evidence_roles": summary.get("history_evidence_roles"),
        "cosupport_seed_ratio_min": summary.get("cosupport_seed_ratio_min"),
        "mask_cosupport_enabled": summary.get("mask_cosupport_enabled"),
        "cosupport_native_gate_enabled": summary.get("cosupport_native_gate_enabled"),
        "cosupport_native_min_support": summary.get("cosupport_native_min_support"),
        "cosupport_native_gate_reject_count": summary.get("cosupport_native_gate_reject_count"),
        "native_boundary_projection_enabled": summary.get("native_boundary_projection_enabled"),
        "native_carrier_rows_path": summary.get("native_carrier_rows_path"),
        "native_boundary_accepted_count": summary.get("native_boundary_accepted_count"),
        "native_boundary_added_component_count": summary.get("native_boundary_added_component_count"),
        "native_uv_projection_enabled": summary.get("native_uv_projection_enabled"),
        "native_uv_accepted_count": summary.get("native_uv_accepted_count"),
        "native_uv_added_component_count": summary.get("native_uv_added_component_count"),
        "native_uv_duplicate_noop_count": summary.get("native_uv_duplicate_noop_count"),
        "native_history_mask_projection_enabled": summary.get("native_history_mask_projection_enabled"),
        "native_history_mask_min_support": summary.get("native_history_mask_min_support"),
        "native_history_mask_min_ratio": summary.get("native_history_mask_min_ratio"),
        "native_history_mask_min_dominance": summary.get("native_history_mask_min_dominance"),
        "native_history_mask_min_mask_ratio": summary.get("native_history_mask_min_mask_ratio"),
        "native_history_mask_component_gate_enabled": summary.get("native_history_mask_component_gate_enabled"),
        "native_history_mask_component_min_support": summary.get("native_history_mask_component_min_support"),
        "native_history_mask_component_gate_candidate_component_count": summary.get(
            "native_history_mask_component_gate_candidate_component_count"
        ),
        "native_history_mask_component_gate_direct_component_count": summary.get(
            "native_history_mask_component_gate_direct_component_count"
        ),
        "native_history_mask_component_gate_filtered_component_count": summary.get(
            "native_history_mask_component_gate_filtered_component_count"
        ),
        "native_history_mask_component_accumulation_gate_enabled": summary.get(
            "native_history_mask_component_accumulation_gate_enabled"
        ),
        "native_history_mask_component_accumulation_min_support": summary.get(
            "native_history_mask_component_accumulation_min_support"
        ),
        "native_history_mask_component_accumulation_min_masks": summary.get(
            "native_history_mask_component_accumulation_min_masks"
        ),
        "native_history_mask_component_accumulation_min_frames": summary.get(
            "native_history_mask_component_accumulation_min_frames"
        ),
        "native_history_mask_component_accumulation_candidate_component_count": summary.get(
            "native_history_mask_component_accumulation_candidate_component_count"
        ),
        "native_history_mask_component_accumulation_eligible_component_count": summary.get(
            "native_history_mask_component_accumulation_eligible_component_count"
        ),
        "native_history_mask_component_accumulation_filtered_component_count": summary.get(
            "native_history_mask_component_accumulation_filtered_component_count"
        ),
        "native_history_mask_component_support_gate_enabled": summary.get(
            "native_history_mask_component_support_gate_enabled"
        ),
        "native_history_mask_component_max_selected_rank": summary.get(
            "native_history_mask_component_max_selected_rank"
        ),
        "native_history_mask_component_min_w_visible": summary.get(
            "native_history_mask_component_min_w_visible"
        ),
        "native_history_mask_component_min_r_mask": summary.get(
            "native_history_mask_component_min_r_mask"
        ),
        "native_history_mask_component_require_dominant": summary.get(
            "native_history_mask_component_require_dominant"
        ),
        "native_history_mask_component_support_gate_candidate_component_count": summary.get(
            "native_history_mask_component_support_gate_candidate_component_count"
        ),
        "native_history_mask_component_support_gate_pass_component_count": summary.get(
            "native_history_mask_component_support_gate_pass_component_count"
        ),
        "native_history_mask_component_support_gate_filtered_component_count": summary.get(
            "native_history_mask_component_support_gate_filtered_component_count"
        ),
        "native_history_mask_cannot_link_guard_enabled": summary.get(
            "native_history_mask_cannot_link_guard_enabled"
        ),
        "native_history_mask_other_seed_min_support": summary.get("native_history_mask_other_seed_min_support"),
        "native_history_mask_other_seed_min_ratio": summary.get("native_history_mask_other_seed_min_ratio"),
        "native_history_mask_second_native_min_support": summary.get("native_history_mask_second_native_min_support"),
        "native_history_mask_second_native_min_ratio": summary.get("native_history_mask_second_native_min_ratio"),
        "native_history_mask_semantic_guard_enabled": summary.get("native_history_mask_semantic_guard_enabled"),
        "native_history_mask_semantic_backend": summary.get("native_history_mask_semantic_backend"),
        "native_history_mask_semantic_min_cosine": summary.get("native_history_mask_semantic_min_cosine"),
        "native_history_mask_semantic_feature_success_rate": summary.get(
            "native_history_mask_semantic_feature_success_rate"
        ),
        "native_history_mask_accepted_count": summary.get("native_history_mask_accepted_count"),
        "native_history_mask_added_component_count": summary.get("native_history_mask_added_component_count"),
        "native_history_mask_duplicate_noop_count": summary.get("native_history_mask_duplicate_noop_count"),
        "native_history_mask_cannot_link_reject_count": summary.get(
            "native_history_mask_cannot_link_reject_count"
        ),
        "native_history_mask_semantic_reject_count": summary.get("native_history_mask_semantic_reject_count"),
        "history_temporal_span_mean": summary.get("history_temporal_span_mean"),
        "anchor_only_temporal_span_mean": summary.get("anchor_only_temporal_span_mean"),
        "history_ARI": summary.get("history_ARI"),
        "history_purity": summary.get("history_purity"),
        "history_completeness": summary.get("history_completeness"),
        "update_precision_diagnostic": summary.get("update_precision_diagnostic"),
        "real_minus_shuffled_ARI": summary.get("real_minus_shuffled_ARI"),
        "real_minus_no_temporal_ARI": summary.get("real_minus_no_temporal_ARI"),
        "gate": summary.get("gate"),
    }


def build_final_decision() -> dict[str, Any]:
    phase0 = _read_json("outputs/audit/v55_phase0_fact_lock/fact_lock.json")
    phase1 = _read_json("outputs/audit/v55_chunk_roles/chunk_role_summary.json")
    phase2 = _read_json("outputs/audit/v55_atoms/atom_summary.json")
    phase3 = _read_json("outputs/audit/v55_anchor_birth/anchor_birth_summary.json")
    native_materialization = _read_json(
        "outputs/audit/v55_native_carrier_materialization_q4096_l11/native_carrier_summary.json"
    )
    semantic_diagnostic = _read_json(
        "outputs/audit/v55_semantic_memory_diagnostic_dinov2_scripted_u8/semantic_memory_summary.json"
    )
    phase4_rows = [
        _phase4_row(
            "U1_objectlet_atom_overlap_baseline",
            "outputs/audit/v55_history_update_u1_objectlet_atom_overlap/history_update_summary.json",
        ),
        _phase4_row(
            "U3_update_only_cosupport_seed020",
            "outputs/audit/v55_history_update_repair_update_only_seed020/history_update_summary.json",
        ),
        _phase4_row(
            "U3_bridge_update_cosupport_seed038_selected",
            "outputs/audit/v55_history_update_repair_bridge_update_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U3_bridge_update_cosupport_seed050_conservative",
            "outputs/audit/v55_history_update_repair_bridge_update_seed050/history_update_summary.json",
        ),
        _phase4_row(
            "U4_native_boundary_only_s100_j001_m3",
            "outputs/audit/v55_history_update_repair_native_boundary_only_s100_j001_m3/history_update_summary.json",
        ),
        _phase4_row(
            "U4_native_boundary_plus_U3_cosupport_seed038_selected",
            "outputs/audit/v55_history_update_repair_native_boundary_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_native_boundary_plus_U3_cosupport_seed050_conservative",
            "outputs/audit/v55_history_update_repair_native_boundary_plus_cosupport_seed050/history_update_summary.json",
        ),
        _phase4_row(
            "U5_native_uv_only_iou005_dist010_m3",
            "outputs/audit/v55_history_update_repair_native_uv_only_iou005_dist010_m3/history_update_summary.json",
        ),
        _phase4_row(
            "U5_native_uv_plus_U3_cosupport_seed038",
            "outputs/audit/v55_history_update_repair_native_uv_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_native_boundary_uv_plus_U3_cosupport_seed038_selected",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_native_boundary_uv_plus_U3_cosupport_seed050_conservative",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_plus_cosupport_seed050/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_plus_U3g_cosupport_nativegate_s20_seed038",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_cosupport_nativegate_s20_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_plus_U3g_cosupport_nativegate_s100_seed038",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_cosupport_nativegate_s100_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_no_U3_fixed",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_fixed/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_plus_U3_selected",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_plus_cosupport_seed038_fixed/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s200_r09_d15_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s200_r09_d15_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s500_r09_d15_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s500_r09_d15_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r095_d15_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r095_d15_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_mr030_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_mr030_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_mr040_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_mr040_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_mr045_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_mr045_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_mr050_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_mr050_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_cannotlink_second003_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_cl_second003_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_cannotlink_seed005_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_cl_seed005_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_cannotlink_seed001_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_cl_seed001_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_semantic_dino094_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_sem_dino094_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_semantic_dino096_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_sem_dino096_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_componentgate_min1_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_componentgate_min1_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6_historymask_s100_r09_d15_componentgate_min3_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_componentgate_min3_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6a_historymask_component_accum_m2f2_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_accum_m2f2_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6a_historymask_component_accum_m3f3_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_accum_m3f3_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6a_historymask_component_accum_m2f2_no_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_accum_m2f2_no_cosupport/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6b_historymask_component_rank100_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_rank100_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6b_historymask_component_rank50_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_rank50_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6b_historymask_component_rank500_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_rank500_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6b_historymask_component_rank1000_plus_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_rank1000_plus_cosupport_seed038/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6b_historymask_component_rank100_no_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_rank100_no_cosupport/history_update_summary.json",
        ),
        _phase4_row(
            "U4_U5_U6b_historymask_component_rank500_no_U3",
            "outputs/audit/v55_history_update_repair_native_boundary_uv_historymask_s100_r09_d15_rank500_no_cosupport/history_update_summary.json",
        ),
    ]
    selected_name = "U4_U5_U6_historymask_s100_r09_d15_plus_U3_selected"
    selected = next(row for row in phase4_rows if row["name"] == selected_name)
    phase0_to_3_pass = all(_gate_pass(summary) for summary in (phase0, phase1, phase2, phase3))
    phase4_pass = bool(selected["gate_pass"])
    final_label = "NO_GO_HISTORY_UPDATE" if phase0_to_3_pass and not phase4_pass else "UNKNOWN_REVIEW_REQUIRED"
    if phase0_to_3_pass and phase4_pass:
        final_label = "PARTIAL_HISTORY_NEEDS_STRESS_NATIVE_EVAL"
    payload = {
        "phase": "v55_final_decision",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "final_label": final_label,
        "goal_achieved": False,
        "partial_label": "PARTIAL_ANCHOR_GO_HISTORY_WEAK" if final_label == "NO_GO_HISTORY_UPDATE" else None,
        "stopped_at_phase": "phase4_history_conditioned_update",
        "stop_reason": (
            "Phase 4 selected U4+U5+U6+U3 repair improves static ARI/completeness and "
            "history-control margins over U4+U5+U3, but still fails purity and the hard "
            "history-control gates; follow-up cannot-link, semantic drift, mask-ratio, "
            "direct component-level native carrier gates, and component accumulation gates trade away the U6 control-margin "
            "gain or leave the hard control gates far below threshold."
        ),
        "phase_gates": {
            "phase0_fact_lock": _gate_pass(phase0),
            "phase1_chunk_roles": _gate_pass(phase1),
            "phase2_material_atoms": _gate_pass(phase2),
            "phase3_anchor_birth": _gate_pass(phase3),
            "phase4_history_update_selected": phase4_pass,
        },
        "key_phase0_metrics": {
            "v54_final_label": phase0.get("v54_final_label"),
            "v54_local_ARI_mean": phase0.get("v54_local_ARI_mean"),
            "v54_local_purity_mean": phase0.get("v54_local_purity_mean"),
            "v54_local_completeness_mean": phase0.get("v54_local_completeness_mean"),
        },
        "key_phase1_metrics": {
            "anchor_chunk_count": phase1.get("anchor_chunk_count"),
            "update_chunk_count": phase1.get("update_chunk_count"),
            "bridge_chunk_count": phase1.get("bridge_chunk_count"),
            "role_separation_score": phase1.get("role_separation_score"),
        },
        "key_phase2_metrics": {
            "atom_variant": phase2.get("atom_variant"),
            "component_count": phase2.get("component_count"),
            "atom_count": phase2.get("atom_count"),
            "atom_purity_diagnostic": phase2.get("atom_purity_diagnostic"),
            "same_frame_conflict_rate": phase2.get("same_frame_conflict_rate"),
            "real_minus_no_temporal_atom_AUC": phase2.get("real_minus_no_temporal_atom_AUC"),
        },
        "key_phase3_metrics": {
            "accepted_birth_count": phase3.get("accepted_birth_count"),
            "birth_from_d4rt_only_count": phase3.get("birth_from_d4rt_only_count"),
            "birth_purity_diagnostic": phase3.get("birth_purity_diagnostic"),
            "birth_completeness_diagnostic": phase3.get("birth_completeness_diagnostic"),
            "accepted_birth_to_GT_object_ratio_diagnostic": phase3.get(
                "accepted_birth_to_GT_object_ratio_diagnostic"
            ),
        },
        "key_native_carrier_materialization_metrics": {
            "path": "outputs/audit/v55_native_carrier_materialization_q4096_l11/native_carrier_summary.json",
            "native_carrier_materialization_pass": native_materialization.get("native_carrier_materialization_pass"),
            "selected_objectlet_count": native_materialization.get("selected_objectlet_count"),
            "selected_component_count": native_materialization.get("selected_component_count"),
            "native_observation_row_count": native_materialization.get("native_observation_row_count"),
            "native_unique_carrier_count": native_materialization.get("native_unique_carrier_count"),
            "method_safe_ap_available": native_materialization.get("method_safe_ap_available"),
        },
        "key_semantic_memory_diagnostic_metrics": {
            "path": "outputs/audit/v55_semantic_memory_diagnostic_dinov2_scripted_u8/semantic_memory_summary.json",
            "backend": semantic_diagnostic.get("backend"),
            "feature_success_rate": semantic_diagnostic.get("feature_success_rate"),
            "confirmed_feature_success_rate": semantic_diagnostic.get("confirmed_feature_success_rate"),
            "false_update_count_diagnostic": semantic_diagnostic.get("false_update_count_diagnostic"),
            "confirmed_false_update_count_diagnostic": semantic_diagnostic.get(
                "confirmed_false_update_count_diagnostic"
            ),
            "semantic_drift_detection_AUC_diagnostic": semantic_diagnostic.get(
                "semantic_drift_detection_AUC_diagnostic"
            ),
        },
        "phase4_selected_metrics": selected,
        "phase4_repair_attempts": phase4_rows,
        "not_run_downstream_phases": [
            "phase5_reactivation_duplicate_repair",
            "phase6_dynamic_readiness_stress",
            "phase8_native_carrier_field_ap",
        ],
        "not_run_reason": (
            "Downstream method promotion is not justified because Phase 4 fails real_minus_shuffled_ARI "
            "and real_minus_no_temporal_ARI hard gates."
        ),
        "analysis_conclusions": [
            "A1 shared-component atoms do not directly link anchor objectlets to bridge/update objectlets; objectlet atom overlap remains zero.",
            "v54 q4096 native carrier materialization succeeds for L11 objectlets, but exact carrier_global_id reuse across chunks is nearly absent in the cache.",
            "Native-boundary projection is a high-precision but low-coverage repair: U4 native-only passes update precision but reaches only 10 confirmed updates and temporal span 1.0510204081632653.",
            "U5 native UV/bbox projection exposes additional non-GT same-frame geometry evidence, but strict thresholds are required: U5-only accepts 11 updates with update_precision_diagnostic 0.913840830449827 and temporal span 1.0459183673469388.",
            "U3g carrier-gated cosupport removes weak mask-only span evidence but does not increase control margins enough: s20 keeps real_minus_shuffled_ARI at 0.11911742496551031 and fails temporal span, while s100 drops to 0.11799254150864924.",
            "U6 native history-mask projection is the strongest new Phase 4 mechanism: U6-only improves real_minus_shuffled_ARI to 0.13206863798461527 but fails temporal span; U6+U3 improves it to 0.14400340767391928 but introduces a small purity failure.",
            "U6 mask-ratio guards expose a Pareto boundary: mr030 still fails purity with real_minus_shuffled_ARI 0.1277480205576933, while mr050 passes purity but loses the U6 control-margin gain.",
            "Intermediate mask-ratio guards confirm the boundary: mr040 and mr045 pass the purity gate but collapse back to real_minus_shuffled_ARI 0.119115655362867 and real_minus_no_temporal_ARI 0.09812035133786201.",
            "Same-frame cannot-link guards did not solve the failure: second-history native conflict guard rejects 17 U6 masks and drops real_minus_shuffled_ARI to 0.13641106402805647, while other-anchor seed guards either leave metrics unchanged or reduce them to 0.12899311667247776.",
            "Frozen DINOv2 semantic drift guard is available and diagnostic-strong: scripted feature_success_rate is 1.0 and semantic_drift_detection_AUC_diagnostic is 0.9659090909090909.",
            "DINO semantic guard is still insufficient as a Phase 4 repair: threshold 0.94 rejects 3 history-mask rows, yet purity remains below gate and real_minus_shuffled_ARI is only 0.1433097218966003.",
            "A stronger semantic threshold 0.96 rejects 14 history-mask rows but overcuts useful updates: history_ARI falls to 0.6080377613744454 and real_minus_shuffled_ARI to 0.1277480205576933, while purity still does not pass.",
            "Direct component-level native carrier gating removes the non-carrier part of U6 mask expansion: min1 sees 38351 candidate components, 23233 directly supported components, filters 15118, but all 45 U6 history-mask rows become duplicate_noop with 0 added components.",
            "The direct component gate passes purity at 0.8889843326292954 but collapses the U6 control-margin gain back to real_minus_shuffled_ARI 0.119115655362867 and real_minus_no_temporal_ARI 0.09812035133786201.",
            "Component accumulation over repeated history-supported masks is a plausible carrier-conditioned completion mechanism, but m2f2 accepts only 6 U6 rows and 235 components after U4/U5, leaving metrics unchanged from U4+U5+U3: real_minus_shuffled_ARI 0.119115655362867 and real_minus_no_temporal_ARI 0.09812035133786201.",
            "Without U3 cosupport, the same m2f2 accumulation gate is even weaker than original U6-only: real_minus_shuffled_ARI 0.08456312097586371 versus original U6-only 0.13206863798461527, so it is not a standalone history-continuity repair.",
            "U6b component support-rank gates test the plan's weak-evidence rule by keeping only high-rank mask-supported components as hard updates: rank50/rank100 pass purity but collapse to U4+U5+U3 control margins, rank500 nearly reaches the purity gate but still fails purity and reaches only real_minus_shuffled_ARI 0.13160210898753716, and rank1000 approaches original U6 coverage while still failing purity and controls.",
            "Without U3 cosupport, U6b rank100 and rank500 both fail temporal span and remain weaker than original U6-only, so support-rank hard-merge gating is not a standalone history-continuity repair.",
            "Visible-mask co-support over bridge+update chunks is a real method-safe signal and improves temporal span from 1.0 to 1.4387755102040816.",
            "The selected U4+U5+U6+U3 repair improves ARI from 0.5034937586311461 to 0.624253092545248 and completeness from 0.5675387978804752 to 0.6814716751050162.",
            "The selected U4+U5+U6+U3 repair still does not establish a robust history method: purity is 0.888004069271944 against the 0.8887398057663739 gate, real_minus_shuffled_ARI is 0.14400340767391928, and real_minus_no_temporal_ARI is 0.12075933391410187.",
            "The correct failure layer is history update/control strength, not chunk role, atom layer, or anchor birth.",
        ],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    selected = payload["phase4_selected_metrics"]
    lines = [
        "# Stream4D v55 Final Decision",
        "",
        f"Final label: `{payload['final_label']}`",
        f"Goal achieved: `{payload['goal_achieved']}`",
        f"Stopped at: `{payload['stopped_at_phase']}`",
        "",
        "## Phase Gates",
    ]
    for key, value in payload["phase_gates"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Selected Phase 4 Metrics",
            "",
            f"- history_temporal_span_mean: {selected['history_temporal_span_mean']}",
            f"- history_ARI: {selected['history_ARI']}",
            f"- history_purity: {selected['history_purity']}",
            f"- history_completeness: {selected['history_completeness']}",
            f"- update_precision_diagnostic: {selected['update_precision_diagnostic']}",
            f"- real_minus_shuffled_ARI: {selected['real_minus_shuffled_ARI']}",
            f"- real_minus_no_temporal_ARI: {selected['real_minus_no_temporal_ARI']}",
            "",
            "## Conclusion",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["analysis_conclusions"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v55 final decision from landed phase summaries.")
    parser.add_argument("--output-root", default="outputs/audit/v55_final_decision")
    args = parser.parse_args()
    out = _project(args.output_root)
    payload = build_final_decision()
    _write_json(out / "final_decision.json", payload)
    write_markdown(out / "final_decision.md", payload)
    print({"final_decision": str(out / "final_decision.json"), "final_label": payload["final_label"]})


if __name__ == "__main__":
    main()
