#!/usr/bin/env python3
"""Build ACL2 v78 Phase8 PCA rediscovery question and hypothesis files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


QUESTION_FIELDS = [
    "failed_phase",
    "failed_candidate",
    "failure_reason",
    "old_layer",
    "old_tap",
    "old_action",
    "old_visual_evidence_file",
    "what_visual_evidence_was_missing",
    "new_visual_question",
    "new_tap_or_layer_to_dump",
    "new_overlay_required",
    "new_candidate_hypothesis",
]

CANDIDATE_FIELDS = [
    "hypothesis_id",
    "memory_body",
    "tap",
    "layer",
    "new_visual_question",
    "source_failed_phase",
    "source_failed_candidate",
    "expected_mechanism_metric",
    "required_controls",
    "stop_rule",
]


def _json_load(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _first_existing(paths: Iterable[Path]) -> str:
    for path in paths:
        if path.exists():
            return str(path)
    return ""


def _phase_summary(v78_root: Path) -> Dict[str, Any]:
    phase1_csv = v78_root / "phase1_l13_controls/full_chunk06_10case/phase1_l13_metrics.csv"
    phase2_csv = v78_root / "phase2_l07_l13_twostage/smoke_chunk06_g0_g6_semfix/phase2_l07_l13_metrics.csv"
    phase3_csv = v78_root / "phase3_swa_handoff/smoke_chunk06_context2_4case/phase3_swa_handoff_metrics.csv"
    phase4_decision = v78_root / "phase4_ttt_write_update/smoke_chunk06_context2_role_control_v1/phase4_ttt_write_role_decision.json"
    phase5_decision = v78_root / "phase5_frame_l18_tail/smoke_chunk06_context2_v1/phase5_frame_l18_tail_decision.json"
    phase9_root = v78_root / "phase9_swa_cache_value_carryover"
    phase4_payload = _json_load(phase4_decision) or {}
    phase5_payload = _json_load(phase5_decision) or {}
    phase9_decisions = sorted(str(path) for path in phase9_root.glob("*/phase9_swa_cache_value_decision.json"))
    return {
        "phase1_metrics_csv": str(phase1_csv) if phase1_csv.exists() else "",
        "phase1_rows": len(_read_csv(phase1_csv)),
        "phase2_metrics_csv": str(phase2_csv) if phase2_csv.exists() else "",
        "phase2_rows": len(_read_csv(phase2_csv)),
        "phase3_metrics_csv": str(phase3_csv) if phase3_csv.exists() else "",
        "phase3_rows": len(_read_csv(phase3_csv)),
        "phase4_decision_json": str(phase4_decision) if phase4_decision.exists() else "",
        "phase4_any_gate_pass": bool(phase4_payload.get("phase4_any_gate_pass", False)),
        "phase5_decision_json": str(phase5_decision) if phase5_decision.exists() else "",
        "phase5_any_gate_pass": bool(phase5_payload.get("phase5_any_gate_pass", False)),
        "phase9_decision_jsons": phase9_decisions,
    }


def _question_rows(v78_root: Path) -> List[Dict[str, Any]]:
    registry = v78_root / "phase0_pca_visual_registry/pca_visual_registry.csv"
    phase8_root = v78_root / "phase8_pca_rediscovery"
    phase8_after_phase9_root = v78_root / "phase8_pca_rediscovery_after_phase9_v1"
    phase8_after_attention_bias_root = v78_root / "phase8_pca_rediscovery_after_phase9_attention_bias_v2"
    phase4_visual = v78_root / "phase4_ttt_write_update/visual_smoke_chunk06_output_separated_r3/chunk_006_TTT_operator_output_L06_L14_L18.png"
    phase4_role = v78_root / "phase4_ttt_write_update/visual_smoke_chunk06_output_separated_r3/chunk_006_TTT_write_role_mass_panel.png"
    phase9_attention_bias_feature = (
        v78_root
        / "phase9_swa_cache_value_carryover/smoke_chunk06_context2_v10_attention_bias_beta070/chunk06/"
        / "P9_20_ATTENTION_BIAS_STABLE_AGREEMENT_LAST/swa_overlap_feature_maps/"
        / "chunk_006_swa_overlap_source_bias_geometric_layer_03.pt"
    )
    phase9_head6_per_head = (
        v78_root
        / "phase9_swa_cache_value_carryover/per_head_attention_mass/"
        / "p9_36_chunk06_v21_head6_per_head.json"
    )
    phase9_heads068_boundary = (
        v78_root
        / "phase9_swa_cache_value_carryover/boundary_residual_audit_v1/"
        / "p9_38_chunk06_heads0_6_8_boundary_residuals.json"
    )
    phase0_global_v = ""
    phase0_frame_v = ""
    phase0_swa = ""
    phase0_rows = _read_csv(registry)
    for row in phase0_rows:
        clue = row.get("clue_id", "")
        if clue == "V78-CLUE-GLOBAL-V-L13":
            phase0_global_v = row.get("representative_contact_sheet", "")
        elif clue == "V78-CLUE-FRAME-V-L18":
            phase0_frame_v = row.get("representative_contact_sheet", "")
        elif clue == "V78-CLUE-SWA-CURRENT-Q-L18":
            phase0_swa = row.get("representative_contact_sheet", "")

    return [
        {
            "failed_phase": "phase1",
            "failed_candidate": "L13_NEG_DAMP_ACTUAL",
            "failure_reason": "single Global-V L13 actual did not pass strong controls and worsened local ATE on chunk06",
            "old_layer": "13",
            "old_tap": "pca_attn_global_v_layers",
            "old_action": "short_term_global_attention",
            "old_visual_evidence_file": phase0_global_v,
            "what_visual_evidence_was_missing": "candidate-level failure overlays for whether L13 V selects semantic structure or only lowstuff/high-D composition",
            "new_visual_question": "Does Global-Q L13 separate geometry/failure regions better than Global-V L13 after overlaying D_geo, future/head-tail/scale failure, and action-vs-random masks?",
            "new_tap_or_layer_to_dump": "pca_attn_global_q_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-001",
        },
        {
            "failed_phase": "phase2",
            "failed_candidate": "G3_G4_G5_G6_L07_MASK_L13",
            "failure_reason": "L07-to-L13 two-stage candidates did not beat controls on the formal chunk06 gate",
            "old_layer": "21",
            "old_tap": "pca_attn_global_k_layers",
            "old_action": "short_term_global_attention",
            "old_visual_evidence_file": phase0_global_v,
            "what_visual_evidence_was_missing": "full-layer global K/V mismatch view beyond L07/L13/L17",
            "new_visual_question": "Do later Global-K layers around L21 preserve road/corridor layout while avoiding the overbroad L13 action mask?",
            "new_tap_or_layer_to_dump": "pca_attn_global_k_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-002",
        },
        {
            "failed_phase": "phase3",
            "failed_candidate": "SWA_ROLE_REWEIGHT",
            "failure_reason": "SWA overlap role reweight changed the hook path but did not improve future/head-tail/scale_cv",
            "old_layer": "18",
            "old_tap": "pca_swa_current_q_layers",
            "old_action": "mid_term_swa",
            "old_visual_evidence_file": phase0_swa,
            "what_visual_evidence_was_missing": "SWA current/cache mismatch and overlap residual overlay tied to failed route mass",
            "new_visual_question": "Does SWA current-Q L18 align with overlap residual/future badness, or is the semantic route acting on the wrong memory object?",
            "new_tap_or_layer_to_dump": "pca_swa_current_q_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-003",
        },
        {
            "failed_phase": "phase3",
            "failed_candidate": "SWA_CACHE_V_ROUTE",
            "failure_reason": "SWA handoff did not show mechanism-level improvement; cache/value side may be the wrong or missing visual target",
            "old_layer": "18",
            "old_tap": "pca_swa_cache_v_layers",
            "old_action": "mid_term_swa",
            "old_visual_evidence_file": phase0_swa,
            "what_visual_evidence_was_missing": "cache-V visual confirmation with failure overlays and actual-vs-random route masks",
            "new_visual_question": "Does SWA cache-V L18 contain persistent layout/failure structure that current-Q route masks missed?",
            "new_tap_or_layer_to_dump": "pca_swa_cache_v_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-004",
        },
        {
            "failed_phase": "phase4",
            "failed_candidate": "T1_T2_T3_TTT_WRITE_ROLE",
            "failure_reason": "TTT write-role control changed prior/action/hash but did not produce Phase4 metric passes",
            "old_layer": "14",
            "old_tap": "pca_ttt_operator_output_layers",
            "old_action": "long_term_ttt",
            "old_visual_evidence_file": _first_existing([phase4_visual, phase4_role]),
            "what_visual_evidence_was_missing": "layer-specific operator/update comparison beyond L18 and a visual reason for state change without trajectory gain",
            "new_visual_question": "Does TTT operator-output L14 highlight road/corridor geometry more cleanly than L18, explaining why broad write-role mass did not help?",
            "new_tap_or_layer_to_dump": "pca_ttt_operator_output_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-005",
        },
        {
            "failed_phase": "phase4",
            "failed_candidate": "T1_T2_T3_TTT_WRITE_ROLE",
            "failure_reason": "TTT update-term visual hotspot was patchy and action magnitude increased without geometry correction",
            "old_layer": "18",
            "old_tap": "pca_ttt_update_term_layers",
            "old_action": "long_term_ttt",
            "old_visual_evidence_file": _first_existing([phase4_visual, phase4_role]),
            "what_visual_evidence_was_missing": "update-term overlay with same-mass/group-stratified action masks on failure frames",
            "new_visual_question": "Does TTT update-term L18 align with high-D dynamic boundaries or only broad lowstuff/sky structure?",
            "new_tap_or_layer_to_dump": "pca_ttt_update_term_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-006",
        },
        {
            "failed_phase": "phase5",
            "failed_candidate": "F1_F2_F3_FRAME_L18_TAIL",
            "failure_reason": "Frame-attention L18 tail source bias had action fidelity but no 10% head-tail/scale_cv improvement",
            "old_layer": "18",
            "old_tap": "pca_attn_frame_v_layers",
            "old_action": "short_term_frame_attention",
            "old_visual_evidence_file": phase0_frame_v,
            "what_visual_evidence_was_missing": "frame Q/K/V comparison to see whether V L18 is the wrong signal for source-side skip",
            "new_visual_question": "Does Frame-K L18 or Frame-Q L18 align with the tail source-skip attention mass better than Frame-V L18?",
            "new_tap_or_layer_to_dump": "pca_attn_frame_k_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-007",
        },
        {
            "failed_phase": "phase5",
            "failed_candidate": "F3_FRAME_L18_HEAD_TAIL_REBALANCE",
            "failure_reason": "mid-tail source skip improved future about 3% but did not reach Phase5 gate and remained small",
            "old_layer": "24",
            "old_tap": "pca_attn_frame_q_layers",
            "old_action": "short_term_frame_attention",
            "old_visual_evidence_file": phase0_frame_v,
            "what_visual_evidence_was_missing": "deeper frame-Q visual confirmation for head/tail scale residual alignment",
            "new_visual_question": "Do deeper Frame-Q layers around L24 show a cleaner tail/scale residual boundary than L18?",
            "new_tap_or_layer_to_dump": "pca_attn_frame_q_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-008",
        },
        {
            "failed_phase": "phase9",
            "failed_candidate": "P9_4/P9_6/P9_8_SWA_SOURCE_VALUE_ACTIONS",
            "failure_reason": "SWA cache-V/source-gate actuators had action fidelity but failed future/head-tail/scale_cv gates and did not beat same-mass controls",
            "old_layer": "18",
            "old_tap": "pca_swa_cache_k_layers",
            "old_action": "mid_term_swa",
            "old_visual_evidence_file": _first_existing([
                phase8_root / "new_pca_contact_sheets/HYP-PCA-REDISC-004_contact_sheet.png",
                phase8_root / "new_pca_contact_sheets/HYP-PCA-REDISC-003_contact_sheet.png",
            ]),
            "what_visual_evidence_was_missing": "SWA cache K/V mismatch view after V-only replace/gate failed; key-side source alignment may be the actual carry-over boundary",
            "new_visual_question": "Does SWA cache-K L18 show a road/corridor or failure-aligned key structure that cache-V source replacement/gating missed?",
            "new_tap_or_layer_to_dump": "pca_swa_cache_k_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-009",
        },
        {
            "failed_phase": "phase9",
            "failed_candidate": "P9_4/P9_6/P9_8_SWA_SOURCE_VALUE_ACTIONS",
            "failure_reason": "V-side source actions degraded native trajectory, suggesting the current value stream or source/current mismatch was not visualized precisely enough",
            "old_layer": "18",
            "old_tap": "pca_swa_current_v_layers",
            "old_action": "mid_term_swa",
            "old_visual_evidence_file": _first_existing([
                phase8_root / "new_pca_contact_sheets/HYP-PCA-REDISC-004_contact_sheet.png",
                phase8_root / "new_pca_contact_sheets/HYP-PCA-REDISC-003_contact_sheet.png",
            ]),
            "what_visual_evidence_was_missing": "current-V overlay paired with cache-V/source-gate failures; current-Q alone may be the wrong side of the SWA value mismatch",
            "new_visual_question": "Does SWA current-V L18 expose a cleaner current-head corridor or residual boundary than current-Q/cache-V, explaining why source-V history edits hurt?",
            "new_tap_or_layer_to_dump": "pca_swa_current_v_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-010",
        },
        {
            "failed_phase": "phase9",
            "failed_candidate": "P9_4/P9_6/P9_8_SWA_SOURCE_VALUE_ACTIONS",
            "failure_reason": "current-Q/cache-V visual clues were confirmed but V-only actions failed; the missing action may be key-side stabilization rather than value replacement",
            "old_layer": "18",
            "old_tap": "pca_swa_current_k_layers",
            "old_action": "mid_term_swa",
            "old_visual_evidence_file": _first_existing([
                phase8_root / "new_pca_contact_sheets/HYP-PCA-REDISC-003_contact_sheet.png",
                phase8_root / "new_pca_contact_sheets/HYP-PCA-REDISC-004_contact_sheet.png",
            ]),
            "what_visual_evidence_was_missing": "current-K view and key-value mismatch after current-Q/cache-V evidence failed as an action family",
            "new_visual_question": "Does SWA current-K L18 show the overlap key mismatch/failure boundary that should gate K or KV instead of V only?",
            "new_tap_or_layer_to_dump": "pca_swa_current_k_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-011",
        },
        {
            "failed_phase": "phase9",
            "failed_candidate": "P9_4/P9_6/P9_8_SWA_SOURCE_VALUE_ACTIONS",
            "failure_reason": "L18 SWA cache/current actions failed, so the next PCA search should test whether the cache-V structure shifts to a different SWA layer",
            "old_layer": "26",
            "old_tap": "pca_swa_cache_v_layers",
            "old_action": "mid_term_swa",
            "old_visual_evidence_file": _first_existing([
                phase8_root / "new_pca_contact_sheets/HYP-PCA-REDISC-004_contact_sheet.png",
            ]),
            "what_visual_evidence_was_missing": "cross-layer cache-V comparison after L18 visual clue failed as a control action",
            "new_visual_question": "Does deeper SWA cache-V L26 contain a more failure-aligned layout/corridor signal than L18 cache-V?",
            "new_tap_or_layer_to_dump": "pca_swa_cache_v_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-012",
        },
        {
            "failed_phase": "phase9b",
            "failed_candidate": "P9_20_ATTENTION_BIAS_STABLE_AGREEMENT_LAST",
            "failure_reason": "stable-agreement SWA overlap attention-bias had action fidelity and tiny head-tail/future gains, but remained far below the 10% mechanism gate after beta=0.70",
            "old_layer": "26",
            "old_tap": "pca_swa_cache_k_layers",
            "old_action": "mid_term_swa_attention_bias",
            "old_visual_evidence_file": _first_existing([
                phase9_attention_bias_feature,
                phase8_after_phase9_root / "new_pca_contact_sheets/HYP-PCA-REDISC-009_contact_sheet.png",
            ]),
            "what_visual_evidence_was_missing": "cross-layer cache-K view after L18 attention-bias gave only sub-percent mechanism gains",
            "new_visual_question": "Does deeper SWA cache-K L26 show a cleaner routeable corridor/failure structure than L18 cache-K for attention-bias handoff?",
            "new_tap_or_layer_to_dump": "pca_swa_cache_k_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-013",
        },
        {
            "failed_phase": "phase9b",
            "failed_candidate": "P9_20_ATTENTION_BIAS_STABLE_AGREEMENT_LAST",
            "failure_reason": "attention-bias improved head-tail/future only weakly, suggesting the route score may still be on the wrong SWA current/cache side",
            "old_layer": "26",
            "old_tap": "pca_swa_current_v_layers",
            "old_action": "mid_term_swa_attention_bias",
            "old_visual_evidence_file": _first_existing([
                phase9_attention_bias_feature,
                phase8_after_phase9_root / "new_pca_contact_sheets/HYP-PCA-REDISC-010_contact_sheet.png",
            ]),
            "what_visual_evidence_was_missing": "deeper current-V view paired with attention-bias feature dump after L18 source route underperformed",
            "new_visual_question": "Does SWA current-V L26 isolate a current-head corridor/value structure that should drive route bias instead of L18 stable-agreement D maps?",
            "new_tap_or_layer_to_dump": "pca_swa_current_v_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-014",
        },
        {
            "failed_phase": "phase9b",
            "failed_candidate": "P9_20_ATTENTION_BIAS_STABLE_AGREEMENT_LAST",
            "failure_reason": "SWA-only route bias did not move scale/head-tail enough; the missing cue may be a short-term frame-attention tail/scale residual rather than SWA source routing",
            "old_layer": "24",
            "old_tap": "pca_attn_frame_k_layers",
            "old_action": "short_term_frame_attention",
            "old_visual_evidence_file": _first_existing([
                phase9_attention_bias_feature,
                phase8_after_phase9_root / "new_pca_contact_sheets/HYP-PCA-REDISC-011_contact_sheet.png",
            ]),
            "what_visual_evidence_was_missing": "frame-attention K/Q tail-scale comparison after SWA attention-bias remained sub-percent",
            "new_visual_question": "Does Frame-K L24 align with head/tail scale residual or boundary jump better than SWA L18/L26 route maps?",
            "new_tap_or_layer_to_dump": "pca_attn_frame_k_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-015",
        },
        {
            "failed_phase": "phase9b",
            "failed_candidate": "P9_20_ATTENTION_BIAS_STABLE_AGREEMENT_LAST",
            "failure_reason": "read-route attention-bias produced only weak positive signal; the remaining failure may be post-read write/update conflict rather than read routing",
            "old_layer": "18",
            "old_tap": "pca_ttt_update_term_layers",
            "old_action": "long_term_ttt",
            "old_visual_evidence_file": _first_existing([
                phase9_attention_bias_feature,
                phase8_after_phase9_root / "new_pca_contact_sheets/HYP-PCA-REDISC-012_contact_sheet.png",
            ]),
            "what_visual_evidence_was_missing": "TTT update-term overlay after SWA read-route bias showed weak positive but no mechanism gate",
            "new_visual_question": "Does TTT update-term L18 concentrate on boundary-local failure regions that could explain why SWA read-route improvements do not persist?",
            "new_tap_or_layer_to_dump": "pca_ttt_update_term_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-016",
        },
        {
            "failed_phase": "phase9c",
            "failed_candidate": "P9_36_HEAD6_ROUTE_BIAS",
            "failure_reason": "head6-only route-bias changed the intended head attention mass but did not produce stable boundary/geometry gains across KITTI01 chunk06 and KITTI02 chunk14",
            "old_layer": "10",
            "old_tap": "pca_swa_current_q_layers",
            "old_action": "mid_term_swa_boundary_selector",
            "old_visual_evidence_file": _first_existing([
                phase9_head6_per_head,
                phase8_after_phase9_root / "new_pca_contact_sheets/HYP-PCA-REDISC-009_contact_sheet.png",
            ]),
            "what_visual_evidence_was_missing": "boundary-position view of current-head overlap queries; per-head mass alone did not identify which source positions matter",
            "new_visual_question": "Does earlier SWA current-Q L10 localize boundary-overlap query positions more sharply than L18/L26, suggesting a boundary-position gate instead of more head/beta scaling?",
            "new_tap_or_layer_to_dump": "pca_swa_current_q_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;overlap/boundary residual;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-017",
        },
        {
            "failed_phase": "phase9c",
            "failed_candidate": "P9_36_HEAD6_ROUTE_BIAS",
            "failure_reason": "selected-token attention mass could be increased, but selected quality did not dominate same-mass random strongly enough to amplify the weak positive signal",
            "old_layer": "34",
            "old_tap": "pca_swa_cache_k_layers",
            "old_action": "mid_term_swa_selected_source_quality",
            "old_visual_evidence_file": _first_existing([
                phase9_head6_per_head,
                phase8_after_attention_bias_root / "new_pca_contact_sheets/HYP-PCA-REDISC-013_contact_sheet.png",
            ]),
            "what_visual_evidence_was_missing": "late cache-key source-quality view that separates useful stable-topq80 source tokens from road/static tokens that behave like random controls",
            "new_visual_question": "Does late SWA cache-K L34 separate high-quality source tokens from random-like stable road/static tokens, giving a stricter source-quality selector than stable-topq80?",
            "new_tap_or_layer_to_dump": "pca_swa_cache_k_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;overlap/boundary residual;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-018",
        },
        {
            "failed_phase": "phase9c",
            "failed_candidate": "P9_38_HEADS0_6_8_ROUTE_BIAS",
            "failure_reason": "adding common positive heads 0,6,8 made the SWA boundary/overlap behavior less stable; common positive heads were not sufficient to define the correct handoff position",
            "old_layer": "34",
            "old_tap": "pca_swa_current_k_layers",
            "old_action": "mid_term_swa_boundary_selector",
            "old_visual_evidence_file": _first_existing([
                phase9_heads068_boundary,
                phase8_after_phase9_root / "new_pca_contact_sheets/HYP-PCA-REDISC-011_contact_sheet.png",
            ]),
            "what_visual_evidence_was_missing": "late current-key mismatch view across the previous-tail/current-head boundary after heads0/6/8 route-bias lost overlap/future metrics",
            "new_visual_question": "Does late SWA current-K L34 reveal boundary-local key mismatch that explains why adding heads0/6/8 harms overlap-to-future metrics?",
            "new_tap_or_layer_to_dump": "pca_swa_current_k_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;overlap/boundary residual;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-019",
        },
        {
            "failed_phase": "phase9c",
            "failed_candidate": "P9_36_P9_38_HEAD_SELECTIVE_ROUTE_BIAS",
            "failure_reason": "head-selective bias showed that head masking is a real actuator, but the remaining failure is likely boundary-local selector/position quality rather than global selected mass",
            "old_layer": "18",
            "old_tap": "pca_swa_cache_v_layers",
            "old_action": "mid_term_swa_boundary_selector",
            "old_visual_evidence_file": _first_existing([
                phase9_heads068_boundary,
                phase9_head6_per_head,
            ]),
            "what_visual_evidence_was_missing": "cache-value view specifically interpreted as boundary-local carry-over body rather than whole selected mass",
            "new_visual_question": "Does SWA cache-V L18 contain a narrow boundary-local carry-over body that should be gated by source position rather than boosting all selected stable tokens?",
            "new_tap_or_layer_to_dump": "pca_swa_cache_v_layers",
            "new_overlay_required": "RGB;semantic;confidence;D_geo;overlap/boundary residual;future/head_tail/scale;actual;same_mass_random;group_stratified_random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-020",
        },
    ]


def _candidate_rows(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for q in questions:
        expected = "head_to_tail / future_after_overlap / scale_cv"
        if q["failed_phase"] == "phase4":
            expected = "future_after_overlap / head_to_tail / next_probe_state_hash"
        elif q["failed_phase"] in {"phase3", "phase9", "phase9b", "phase9c"}:
            expected = "SWA boundary residual / overlap3_to_future / adjacent-pair Sim3 / head_to_tail"
        rows.append(
            {
                "hypothesis_id": q["new_candidate_hypothesis"],
                "memory_body": q["old_action"],
                "tap": q["new_tap_or_layer_to_dump"],
                "layer": q["old_layer"],
                "new_visual_question": q["new_visual_question"],
                "source_failed_phase": q["failed_phase"],
                "source_failed_candidate": q["failed_candidate"],
                "expected_mechanism_metric": expected,
                "required_controls": "same-mass random; group-stratified random; label/confidence shuffle where applicable",
                "stop_rule": "stop this hypothesis if visual review is rejected/ambiguous or if actual loses to group-stratified random without unique structure",
            }
        )
    return rows


def _hypothesis_bank(candidates: List[Dict[str, Any]], questions: List[Dict[str, Any]]) -> str:
    by_hid = {row["new_candidate_hypothesis"]: row for row in questions}
    lines = [
        "# v78 Phase8 New Hypothesis Bank",
        "",
        "Status: pca_rediscovery_required",
        "",
        "These hypotheses are generated after HMC candidates failed to pass their gates. They are not success claims.",
        "",
    ]
    for row in candidates:
        q = by_hid.get(row["hypothesis_id"], {})
        lines.extend(
            [
                f"## {row['hypothesis_id']}",
                "",
                f"Memory body: {row['memory_body']}",
                f"Tap/layer: {row['tap']} L{row['layer']}",
                f"Visual evidence files: to be generated from Phase8 visualizer using failed_action_to_visual_question.csv",
                f"Observed visual pattern: pending visual review; question = {row['new_visual_question']}",
                f"Why previous action failed: {q.get('failure_reason', '')}",
                f"New action point: pending; only allowed if Phase8 visual review confirms a non-random pattern",
                f"Expected mechanism metric: {row['expected_mechanism_metric']}",
                f"Required controls: {row['required_controls']}",
                "Required visual outputs: contact sheet; single-frame overlay; temporal filmstrip; action-vs-random panel",
                f"Stop rule: {row['stop_rule']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v78-root", type=Path, required=True)
    parser.add_argument("--failed-phases", default="phase1,phase2,phase3,phase4,phase5,phase6,phase7")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested = {part.strip() for part in str(args.failed_phases).split(",") if part.strip()}
    questions = [row for row in _question_rows(args.v78_root) if row["failed_phase"] in requested]
    candidates = _candidate_rows(questions)
    summary = {
        "schema": "acl2_v78_phase8_rediscovery_trigger_v1",
        "status": "pca_rediscovery_required",
        "v78_root": str(args.v78_root),
        "failed_phases_requested": sorted(requested),
        "num_questions": len(questions),
        "num_new_layer_tap_candidates": len(candidates),
        "phase_summary": _phase_summary(args.v78_root),
        "note": (
            "Phase6/Phase7 are not executed because no single path passed Phase1-5; "
            "this is not a semantic/PCA route exhaustion claim."
        ),
    }
    _write_csv(args.out_dir / "failed_action_to_visual_question.csv", questions, QUESTION_FIELDS)
    _write_csv(args.out_dir / "new_layer_tap_candidates.csv", candidates, CANDIDATE_FIELDS)
    (args.out_dir / "new_hypothesis_bank.md").write_text(_hypothesis_bank(candidates, questions), encoding="utf-8")
    (args.out_dir / "rediscovery_trigger_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
