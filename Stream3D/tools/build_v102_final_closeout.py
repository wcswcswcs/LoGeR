from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STREAM3D = ROOT / "Stream3D"
AUDIT_ROOT = STREAM3D / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v102_final_closeout"
PLAN_DOC = ROOT / "docs" / "stream4d_v102_mask_pair_primitive_bridge_da3_giant_3dgs_plan.md"

PHASE_SUMMARIES = {
    "phase0": AUDIT_ROOT / "v102_phase0_fact_lock" / "summary.json",
    "phase1": AUDIT_ROOT / "v102_phase1_fragment_casebook" / "summary.json",
    "phase1b_repair_space": AUDIT_ROOT / "v102_phase1b_repair_space_policy_sweep" / "summary.json",
    "phase2": AUDIT_ROOT / "v102_phase2_provider_ladder_audit" / "summary.json",
    "phase2b_chunk32": AUDIT_ROOT / "v102_phase2b_da3_giant_chunk32_audit" / "summary.json",
    "phase3": AUDIT_ROOT / "v102_phase3_da3_giant_3dgs_visual_audit" / "summary.json",
    "phase3b_chunk32": AUDIT_ROOT / "v102_phase3b_da3_giant_chunk32_visual_audit" / "summary.json",
    "phase3c_chunk32_gsplat_video": AUDIT_ROOT
    / "v102_phase3c_da3_giant_chunk32_gsplat_video"
    / "summary.json",
    "phase4": AUDIT_ROOT / "v102_phase4_persistent_primitive_diagnostic" / "summary.json",
    "phase5": AUDIT_ROOT / "v102_phase5_mask_pair_bridge_smoke" / "summary.json",
    "phase5b_chunk32_bridge": AUDIT_ROOT / "v102_phase5b_chunk32_short_range_bridge_repair" / "summary.json",
    "phase5c_semantic_barrier": AUDIT_ROOT / "v102_phase5c_semantic_barrier_bridge_repair" / "summary.json",
    "phase7a_bridge_rebirth": AUDIT_ROOT / "v102_phase7a_bridge_rebirth_diagnostic" / "summary.json",
    "phase7b_constrained_rebirth": AUDIT_ROOT
    / "v102_phase7b_constrained_bridge_graph_repair"
    / "summary.json",
    "phase7c_node_quality_rebirth": AUDIT_ROOT
    / "v102_phase7c_node_quality_constrained_rebirth"
    / "summary.json",
    "phase7d_materialized_ap": AUDIT_ROOT
    / "v102_phase7d_phase7c_materialized_ap_diagnostic"
    / "summary.json",
    "phase7e_score_calibration": AUDIT_ROOT
    / "v102_phase7e_gtfree_score_calibration_diagnostic"
    / "summary.json",
    "phase7f_component_stitch": AUDIT_ROOT
    / "v102_phase7f_gtfree_component_stitch_diagnostic"
    / "summary.json",
    "phase7g_missing_frame_expansion": AUDIT_ROOT
    / "v102_phase7g_semantic_missing_frame_expansion"
    / "summary.json",
    "phase7h_primitive_support_shape": AUDIT_ROOT
    / "v102_phase7h_chunk32_primitive_support_shape_diagnostic"
    / "summary.json",
    "phase7i_same_chunk_f2_comparison": AUDIT_ROOT
    / "v102_phase7i_same_chunk_f2_vs_phase7h_comparison"
    / "summary.json",
    "phase7j_f2_primitive_support_fill": AUDIT_ROOT
    / "v102_phase7j_f2_primitive_support_fill_diagnostic"
    / "summary.json",
}


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phases = {name: _read_json(path) for name, path in PHASE_SUMMARIES.items()}

    phase1_repair_candidates = int(phases["phase1"].get("repair_candidate_pair_count", 0))
    phase1b_safe_candidates = int(phases["phase1b_repair_space"].get("safe_promotable_candidate_count_max", 0))
    phase5_smoke_formal_pass = bool(phases["phase5"].get("phase5_formal_pass_for_phase6"))
    phase5c_formal_pass = bool(phases["phase5c_semantic_barrier"].get("any_phase5_formal_bridge_gate_pass"))
    phase5_formal_pass = bool(phase5_smoke_formal_pass or phase5c_formal_pass)
    phase6_allowed = bool(phase5_formal_pass and phase1_repair_candidates >= 30 and phase1b_safe_candidates >= 30)

    final_decision = (
        "GO_F2_FRAGMENT_REPAIR_WITH_PRIMITIVE_BRIDGE"
        if phase6_allowed
        else "PARTIAL_PHASE5_BRIDGE_PASS__NO_PHASE6_AP_REPAIR_PHASE1_REPAIR_SPACE_BLOCKED"
        if phase5c_formal_pass
        else "DIAGNOSTIC_PROVIDER_ADVANCE_ONLY__NO_PHASE6_AP_REPAIR"
    )
    gate_rows = [
        {
            "gate_id": "phase0_fact_lock",
            "pass": bool(phases["phase0"].get("phase0_pass")),
            "observed": phases["phase0"].get("decision"),
            "required_for": "all_v102",
        },
        {
            "gate_id": "phase1_repair_candidate_pair_count",
            "pass": phase1_repair_candidates >= 30,
            "expected": ">=30 for AP repair",
            "observed": phase1_repair_candidates,
            "required_for": "phase6_ap_repair",
        },
        {
            "gate_id": "phase1b_safe_promotable_repair_space",
            "pass": phase1b_safe_candidates >= 30,
            "expected": ">=30 safe positive-union repair candidates",
            "observed": phase1b_safe_candidates,
            "required_for": "phase6_ap_repair",
        },
        {
            "gate_id": "phase1_broad_contamination_rate",
            "pass": float(phases["phase1"].get("broad_contamination_rate", 1.0)) <= 0.5,
            "expected": "<=0.5 before direct merge",
            "observed": phases["phase1"].get("broad_contamination_rate"),
            "required_for": "direct_merge_or_ap_repair",
        },
        {
            "gate_id": "phase2_3dgs_smoke",
            "pass": bool(phases["phase2"].get("phase2_pass_for_3dgs_promotion")),
            "observed": phases["phase2"].get("decision"),
            "required_for": "phase3",
        },
        {
            "gate_id": "phase3_visual_artifact",
            "pass": bool(phases["phase3"].get("phase3_pass_for_visual_artifact")),
            "observed": phases["phase3"].get("decision"),
            "required_for": "phase4",
        },
        {
            "gate_id": "phase2b_chunk32_3dgs_export",
            "pass": bool(phases["phase2b_chunk32"].get("chunk32_success")),
            "observed": phases["phase2b_chunk32"].get("decision"),
            "required_for": "chunk_level_provider_audit",
        },
        {
            "gate_id": "phase3b_chunk32_visual_artifact",
            "pass": bool(phases["phase3b_chunk32"].get("chunk32_visual_artifact_pass")),
            "observed": phases["phase3b_chunk32"].get("decision"),
            "required_for": "chunk_level_visual_audit",
        },
        {
            "gate_id": "phase3c_chunk32_gsplat_interpolated_pose_video",
            "pass": bool(phases["phase3c_chunk32_gsplat_video"].get("render_success")),
            "observed": phases["phase3c_chunk32_gsplat_video"].get("decision"),
            "required_for": "chunk_level_gsplat_video_audit",
        },
        {
            "gate_id": "phase4_mask_participation_and_purity",
            "pass": bool(phases["phase4"].get("phase4_pass_for_phase5_bridge")),
            "observed": phases["phase4"].get("decision"),
            "required_for": "phase5_smoke",
        },
        {
            "gate_id": "phase5_same_object_bridge_recall",
            "pass": phases["phase5"].get("fixed_same_object_bridge_recall", 0.0) >= 0.35,
            "expected": ">=0.35",
            "observed": phases["phase5"].get("fixed_same_object_bridge_recall"),
            "required_for": "formal_phase5",
        },
        {
            "gate_id": "phase5b_chunk32_geometric_bridge",
            "pass": bool(phases["phase5b_chunk32_bridge"].get("any_phase5_formal_bridge_gate_pass")),
            "expected": "Phase5 formal gate pass without semantic barrier",
            "observed": phases["phase5b_chunk32_bridge"].get("decision"),
            "required_for": "bridge_repair_diagnostic",
        },
        {
            "gate_id": "phase5c_semantic_barrier_bridge",
            "pass": phase5c_formal_pass,
            "expected": "Phase5 formal gate pass with GT-free semantic residual barrier",
            "observed": phases["phase5c_semantic_barrier"].get("decision"),
            "required_for": "phase6_ap_repair",
        },
        {
            "gate_id": "phase5_same_semantic_diff_gt_false_bridge",
            "pass": phase5c_formal_pass
            and phases["phase5c_semantic_barrier"].get(
                "best_variant_same_semantic_different_gt_false_bridge_among_accepted", 1.0
            )
            <= 0.20,
            "expected": "<=0.20 with same-semantic hard negatives",
            "observed": phases["phase5c_semantic_barrier"].get(
                "best_variant_same_semantic_different_gt_false_bridge_among_accepted"
            ),
            "required_for": "formal_phase5",
        },
        {
            "gate_id": "phase5_auc",
            "pass": phases["phase5c_semantic_barrier"].get("bridge_auc", 0.0) >= 0.65,
            "expected": ">=0.65",
            "observed": phases["phase5c_semantic_barrier"].get("bridge_auc"),
            "required_for": "formal_phase5",
        },
        {
            "gate_id": "phase6_ap_repair_allowed",
            "pass": phase6_allowed,
            "expected": "Phase5 formal pass and Phase1 repair candidates >=30",
            "observed": phase6_allowed,
            "required_for": "phase6",
        },
        {
            "gate_id": "phase7a_primitive_rebirth_components_safe",
            "pass": bool(phases["phase7a_bridge_rebirth"].get("any_variant_safe_for_primitive_rebirth")),
            "expected": "no same-frame collision / no multi-GT unsafe components",
            "observed": phases["phase7a_bridge_rebirth"].get("decision"),
            "required_for": "optional_phase7_rebirth",
        },
        {
            "gate_id": "phase7b_constrained_rebirth_components_safe",
            "pass": bool(phases["phase7b_constrained_rebirth"].get("any_variant_safe_for_primitive_rebirth")),
            "expected": "same-frame cannot-link plus centroid veto removes unsafe components",
            "observed": phases["phase7b_constrained_rebirth"].get("decision"),
            "required_for": "optional_phase7_rebirth",
        },
        {
            "gate_id": "phase7c_node_quality_rebirth_components_safe",
            "pass": bool(phases["phase7c_node_quality_rebirth"].get("any_variant_safe_for_primitive_rebirth")),
            "expected": "GT-free node-quality filtering produces clean diagnostic components",
            "observed": phases["phase7c_node_quality_rebirth"].get("decision"),
            "required_for": "phase7d_materialized_diagnostic",
        },
        {
            "gate_id": "phase7d_materialized_local_ap_recorded",
            "pass": bool(phases["phase7d_materialized_ap"].get("local_diagnostic_ap_recorded")),
            "expected": "materialize Phase7c components and record chunk32 diagnostic AP",
            "observed": phases["phase7d_materialized_ap"].get("decision"),
            "required_for": "optional_phase7_rebirth_diagnostic_only",
        },
        {
            "gate_id": "phase7e_gtfree_score_calibration_local_ap50_improves",
            "pass": bool(
                phases["phase7e_score_calibration"].get(
                    "local_gtfree_score_ap50_improves_over_phase7d"
                )
            ),
            "expected": "GT-free score ordering improves local chunk32 AP50 over Phase7d",
            "observed": phases["phase7e_score_calibration"].get("decision"),
            "required_for": "optional_phase7_rebirth_diagnostic_only",
        },
        {
            "gate_id": "phase7f_gtfree_component_stitch_local_improves",
            "pass": bool(phases["phase7f_component_stitch"].get("local_diagnostic_improves")),
            "expected": "GT-free component stitching improves local score-free recall or AP50",
            "observed": phases["phase7f_component_stitch"].get("decision"),
            "required_for": "optional_phase7_rebirth_diagnostic_only",
        },
        {
            "gate_id": "phase7g_semantic_missing_frame_expansion_local_improves",
            "pass": bool(phases["phase7g_missing_frame_expansion"].get("local_diagnostic_improves")),
            "expected": "GT-free missing-frame semantic support improves local AP50 or ScoreFreeMatch50",
            "observed": phases["phase7g_missing_frame_expansion"].get("decision"),
            "required_for": "optional_phase7_rebirth_diagnostic_only",
        },
        {
            "gate_id": "phase7h_chunk32_primitive_support_shape_local_improves",
            "pass": bool(phases["phase7h_primitive_support_shape"].get("local_diagnostic_improves")),
            "expected": "chunk32 DA3-GIANT primitive support expansion improves local AP50 or ScoreFreeMatch50 over Phase7e",
            "observed": phases["phase7h_primitive_support_shape"].get("decision"),
            "required_for": "optional_phase7_rebirth_diagnostic_only",
        },
        {
            "gate_id": "phase7i_phase7h_beats_same_chunk_f2",
            "pass": bool(phases["phase7i_same_chunk_f2_comparison"].get("local_diagnostic_beats_same_chunk_f2")),
            "expected": "Phase7h P2 primitive re-birth beats same scene0050/c0000 v100 F2 overlap3 local baseline",
            "observed": phases["phase7i_same_chunk_f2_comparison"].get("decision"),
            "required_for": "phase7_p2_vs_f2_claim",
        },
        {
            "gate_id": "phase7j_f2_primitive_support_fill_local_improves",
            "pass": bool(phases["phase7j_f2_primitive_support_fill"].get("local_diagnostic_improves")),
            "expected": "Phase7h primitive-support fill/replacement improves same scene0050/c0000 F2 local baseline",
            "observed": phases["phase7j_f2_primitive_support_fill"].get("decision"),
            "required_for": "phase6_f2_repair_candidate",
        },
    ]
    gate_csv = OUT_DIR / "final_gate_rows.csv"
    _write_csv(gate_csv, gate_rows)

    code_change_rows = [
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase0_fact_lock.py",
            "change": "added v102 fact lock against v100/v101/v65 boundaries",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase1_fragment_casebook.py",
            "change": "added effective fragment, repair candidate, and broad contamination casebook",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase1b_repair_space_policy_sweep.py",
            "change": "added Phase1 repair-space policy sweep to test whether broad-risk filtering or relaxed thresholds create safe Phase6 candidates",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase2_provider_ladder_audit.py",
            "change": "added provider ladder audit, official DA3-GIANT-1.1 gs_ply_only smoke, and deterministic mini_npz sidecar export",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase3_da3_giant_3dgs_visual_audit.py",
            "change": "added PLY parsing, snapshots, provider quality rows, and reprojection diagnostics",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase2b_da3_giant_chunk32_audit.py",
            "change": "added chunk-level DA3-GIANT-1.1 3DGS audit with Smoke-8 and Chunk-32 attempts",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase3b_da3_giant_chunk32_visual_audit.py",
            "change": "added chunk-32 3DGS visual artifact, snapshots, and reprojection diagnostics",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase3c_da3_giant_chunk32_gsplat_video.py",
            "change": "added gsplat interpolated-pose video render for the chunk-32 DA3-GIANT-1.1 prediction",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase4_persistent_primitive_diagnostic.py",
            "change": "added Gaussian primitive/projection rows plus two-frame CropFormer mask participation and diagnostic purity",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase5_mask_pair_bridge_smoke.py",
            "change": "added two-frame mask-pair primitive bridge smoke and recall/broad-veto repair variants",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase5b_chunk32_short_range_bridge_repair.py",
            "change": "added chunk-32 short-range shared-Gaussian bridge repair variants and same-semantic diagnostic hard negatives",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase5c_semantic_barrier_bridge_repair.py",
            "change": "added GT-free v91 RADIO semantic residual barrier variants for high false-bridge repair",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase7a_bridge_rebirth_diagnostic.py",
            "change": "added optional bridge-component primitive re-birth diagnostic and same-frame-collision component audit",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase7b_constrained_bridge_graph_repair.py",
            "change": "added same-frame cannot-link and RADIO centroid-veto constrained bridge forest repair",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase7c_node_quality_constrained_rebirth.py",
            "change": "added GT-free RADIO entropy/margin node-quality filter before constrained primitive re-birth",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase7d_phase7c_materialized_ap_diagnostic.py",
            "change": "added Phase7c component materialization into CropFormer mask predictions and local chunk32 diagnostic AP readout",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase7e_gtfree_score_calibration_diagnostic.py",
            "change": "added GT-free score calibration sweep over Phase7d local component predictions",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase7f_gtfree_component_stitch_diagnostic.py",
            "change": "added GT-free RADIO centroid/prototype component stitching diagnostic over Phase7d components",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase7g_semantic_missing_frame_expansion.py",
            "change": "added GT-free RADIO semantic missing-frame support expansion diagnostic for Phase7d components",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase7h_chunk32_primitive_support_shape_diagnostic.py",
            "change": "added chunk32 DA3-GIANT 3DGS primitive-support missing-frame expansion diagnostic and fixed expansion rows to carry dataset_split=dev",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase7i_same_chunk_f2_vs_phase7h_comparison.py",
            "change": "added same scene0050/c0000 local diagnostic comparison between v100 F2 overlap3 and Phase7h best rows; fixed dataset_split filtering with fillna",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_phase7j_f2_primitive_support_fill_diagnostic.py",
            "change": "added F2-backbone primitive-support fill/replacement diagnostic using Phase7h rows mapped to v100 F2 objects by shared frame/mask observations",
        },
        {
            "schema_version": "stream4d_v102_code_change_row_v1",
            "path": "Stream3D/tools/build_v102_final_closeout.py",
            "change": "added final v102 gate closeout, Phase7b/7c/7d/7e/7f/7g/7h/7i/7j status, and code-change manifest",
        },
    ]
    code_change_csv = OUT_DIR / "code_change_rows.csv"
    _write_csv(code_change_csv, code_change_rows)

    summary = {
        "schema_version": "stream4d_v102_final_closeout_summary_v1",
        "phase_id": "v102_final_closeout",
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": final_decision,
        "phase6_ap_repair_run": False,
        "phase6_ap_repair_allowed": phase6_allowed,
        "new_ap_metrics": {},
        "current_positive_method_remains": phases["phase0"].get("current_positive_method"),
        "F2_holdout_MV_AP_window_locked": phases["phase0"].get("F2_holdout_MV_AP_window"),
        "F2_holdout_fragmented_MV_AP_scene_locked": phases["phase0"].get("F2_holdout_fragmented_MV_AP_scene"),
        "provider_3dgs_success": bool(phases["phase2"].get("official_3dgs_export_success")),
        "provider_chunk32_3dgs_success": bool(phases["phase2b_chunk32"].get("chunk32_success")),
        "provider_chunk32_ply_file": phases["phase2b_chunk32"].get("best_chunk32_ply_file"),
        "provider_chunk32_ply_file_size_MB": phases["phase2b_chunk32"].get("best_chunk32_ply_file_size_MB"),
        "phase3_gaussian_count": phases["phase3"].get("gaussian_count"),
        "phase3_reprojection_valid_rate": phases["phase3"].get("reprojection_valid_rate"),
        "phase3b_chunk32_gaussian_count": phases["phase3b_chunk32"].get("gaussian_count"),
        "phase3b_chunk32_reprojection_valid_rate": phases["phase3b_chunk32"].get("reprojection_valid_rate"),
        "phase3b_chunk32_visual_artifact_pass": phases["phase3b_chunk32"].get("chunk32_visual_artifact_pass"),
        "phase3c_chunk32_gsplat_video_render_success": phases["phase3c_chunk32_gsplat_video"].get(
            "render_success"
        ),
        "phase3c_chunk32_gsplat_video_file": phases["phase3c_chunk32_gsplat_video"].get("video_file"),
        "phase3c_chunk32_gsplat_video_file_size_MB": phases["phase3c_chunk32_gsplat_video"].get(
            "video_file_size_MB"
        ),
        "phase3c_chunk32_gsplat_video_frame_count_input": phases["phase3c_chunk32_gsplat_video"].get(
            "frame_count"
        ),
        "phase3c_chunk32_gsplat_video_trj_mode": phases["phase3c_chunk32_gsplat_video"].get("trj_mode"),
        "phase4_primitive_mask_participation_row_count": phases["phase4"].get("primitive_mask_participation_row_count"),
        "phase4_primitive_mask_purity_row_count": phases["phase4"].get("primitive_mask_purity_row_count"),
        "phase5_candidate_pair_count": phases["phase5"].get("candidate_pair_count"),
        "phase5_bridge_auc": phases["phase5"].get("bridge_auc"),
        "phase5_fixed_same_object_bridge_recall": phases["phase5"].get("fixed_same_object_bridge_recall"),
        "phase5_fixed_different_gt_false_bridge_proxy": phases["phase5"].get(
            "fixed_different_gt_false_bridge_among_accepted_proxy"
        ),
        "phase5_same_semantic_false_bridge_available": phases["phase5"].get(
            "same_semantic_diff_gt_false_bridge_available"
        ),
        "phase1_repair_candidate_pair_count": phase1_repair_candidates,
        "phase1b_safe_promotable_candidate_count_max": phases["phase1b_repair_space"].get(
            "safe_promotable_candidate_count_max"
        ),
        "phase1b_all_positive_union_pair_count": phases["phase1b_repair_space"].get(
            "all_positive_union_pair_count"
        ),
        "phase1b_no_collision_positive_union_pair_count": phases["phase1b_repair_space"].get(
            "no_collision_positive_union_pair_count"
        ),
        "phase1_broad_contamination_rate": phases["phase1"].get("broad_contamination_rate"),
        "phase5b_chunk32_best_variant_id_by_recall": phases["phase5b_chunk32_bridge"].get(
            "best_variant_id_by_recall"
        ),
        "phase5b_chunk32_best_same_object_bridge_recall": phases["phase5b_chunk32_bridge"].get(
            "best_variant_same_object_bridge_recall"
        ),
        "phase5b_chunk32_best_different_gt_false_bridge": phases["phase5b_chunk32_bridge"].get(
            "best_variant_different_gt_false_bridge_among_accepted"
        ),
        "phase5c_semantic_barrier_phase5_pass": phase5c_formal_pass,
        "phase5c_semantic_barrier_best_variant_id": phases["phase5c_semantic_barrier"].get("best_variant_id"),
        "phase5c_semantic_barrier_best_same_object_bridge_recall": phases["phase5c_semantic_barrier"].get(
            "best_variant_same_object_bridge_recall"
        ),
        "phase5c_semantic_barrier_best_different_gt_false_bridge": phases["phase5c_semantic_barrier"].get(
            "best_variant_different_gt_false_bridge_among_accepted"
        ),
        "phase5c_semantic_barrier_best_same_semantic_diff_gt_false_bridge": phases[
            "phase5c_semantic_barrier"
        ].get("best_variant_same_semantic_different_gt_false_bridge_among_accepted"),
        "phase5c_semantic_barrier_bridge_auc": phases["phase5c_semantic_barrier"].get("bridge_auc"),
        "phase7a_bridge_rebirth_decision": phases["phase7a_bridge_rebirth"].get("decision"),
        "phase7a_any_variant_safe_for_primitive_rebirth": phases["phase7a_bridge_rebirth"].get(
            "any_variant_safe_for_primitive_rebirth"
        ),
        "phase7a_best_variant_id": phases["phase7a_bridge_rebirth"].get("best_variant_id"),
        "phase7a_best_variant_same_frame_collision_component_count": phases["phase7a_bridge_rebirth"].get(
            "best_variant_same_frame_collision_component_count"
        ),
        "phase7a_best_variant_multi_gt_component_count": phases["phase7a_bridge_rebirth"].get(
            "best_variant_multi_gt_component_count"
        ),
        "phase7a_best_variant_component_purity_p10": phases["phase7a_bridge_rebirth"].get(
            "best_variant_component_purity_p10"
        ),
        "phase7b_constrained_rebirth_decision": phases["phase7b_constrained_rebirth"].get("decision"),
        "phase7b_any_variant_safe_for_primitive_rebirth": phases["phase7b_constrained_rebirth"].get(
            "any_variant_safe_for_primitive_rebirth"
        ),
        "phase7b_best_variant_id": phases["phase7b_constrained_rebirth"].get("best_variant_id"),
        "phase7b_best_variant_same_frame_collision_component_count": phases[
            "phase7b_constrained_rebirth"
        ].get("best_variant_same_frame_collision_component_count"),
        "phase7b_best_variant_multi_gt_component_count": phases["phase7b_constrained_rebirth"].get(
            "best_variant_multi_gt_component_count"
        ),
        "phase7b_best_variant_component_purity_p10": phases["phase7b_constrained_rebirth"].get(
            "best_variant_component_purity_p10"
        ),
        "phase7c_node_quality_rebirth_decision": phases["phase7c_node_quality_rebirth"].get("decision"),
        "phase7c_any_variant_safe_for_primitive_rebirth": phases["phase7c_node_quality_rebirth"].get(
            "any_variant_safe_for_primitive_rebirth"
        ),
        "phase7c_best_variant_id": phases["phase7c_node_quality_rebirth"].get("best_variant_id"),
        "phase7c_best_variant_component_count": phases["phase7c_node_quality_rebirth"].get(
            "best_variant_component_count"
        ),
        "phase7c_best_variant_clean_component_proxy_count": phases["phase7c_node_quality_rebirth"].get(
            "best_variant_clean_component_proxy_count"
        ),
        "phase7c_best_variant_largest_component_size": phases["phase7c_node_quality_rebirth"].get(
            "best_variant_largest_component_size"
        ),
        "phase7c_best_variant_same_frame_collision_component_count": phases[
            "phase7c_node_quality_rebirth"
        ].get("best_variant_same_frame_collision_component_count"),
        "phase7c_best_variant_multi_gt_component_count": phases["phase7c_node_quality_rebirth"].get(
            "best_variant_multi_gt_component_count"
        ),
        "phase7c_best_variant_component_purity_p10": phases["phase7c_node_quality_rebirth"].get(
            "best_variant_component_purity_p10"
        ),
        "phase7d_materialized_ap_decision": phases["phase7d_materialized_ap"].get("decision"),
        "phase7d_local_diagnostic_ap_recorded": phases["phase7d_materialized_ap"].get(
            "local_diagnostic_ap_recorded"
        ),
        "phase7d_formal_v102_target_achieved": phases["phase7d_materialized_ap"].get(
            "formal_v102_target_achieved"
        ),
        "phase7d_best_MV_AP_window": phases["phase7d_materialized_ap"].get("MV_AP_window"),
        "phase7d_best_MV_AP50_window": phases["phase7d_materialized_ap"].get("MV_AP50_window"),
        "phase7d_best_MV_AP_scene": None,
        "phase7d_best_MV_AP50_scene": None,
        "phase7d_scene_metric_computed": False,
        "phase7d_scene_metric_not_computed_reason": "Phase7d is a scene0050_00/c0000 chunk32 local diagnostic; full-scene/local2history MV_AP_scene is not computed.",
        "phase7d_score_free_match50_window": phases["phase7d_materialized_ap"].get(
            "ScoreFreeMatch50_window"
        ),
        "phase7d_score_free_match25_window": phases["phase7d_materialized_ap"].get(
            "ScoreFreeMatch25_window"
        ),
        "phase7e_score_calibration_decision": phases["phase7e_score_calibration"].get("decision"),
        "phase7e_best_score_variant_id": phases["phase7e_score_calibration"].get("best_score_variant_id"),
        "phase7e_base_MV_AP_window": phases["phase7e_score_calibration"].get("base_MV_AP_window"),
        "phase7e_base_MV_AP50_window": phases["phase7e_score_calibration"].get("base_MV_AP50_window"),
        "phase7e_best_MV_AP_window": phases["phase7e_score_calibration"].get("best_MV_AP_window"),
        "phase7e_best_MV_AP50_window": phases["phase7e_score_calibration"].get("best_MV_AP50_window"),
        "phase7e_best_delta_MV_AP_window_vs_phase7d": phases["phase7e_score_calibration"].get(
            "best_delta_MV_AP_window_vs_phase7d"
        ),
        "phase7e_best_delta_MV_AP50_window_vs_phase7d": phases["phase7e_score_calibration"].get(
            "best_delta_MV_AP50_window_vs_phase7d"
        ),
        "phase7e_local_gtfree_score_ap50_improves_over_phase7d": phases[
            "phase7e_score_calibration"
        ].get("local_gtfree_score_ap50_improves_over_phase7d"),
        "phase7e_formal_v102_target_achieved": phases["phase7e_score_calibration"].get(
            "formal_v102_target_achieved"
        ),
        "phase7f_component_stitch_decision": phases["phase7f_component_stitch"].get("decision"),
        "phase7f_best_variant_id": phases["phase7f_component_stitch"].get("best_variant_id"),
        "phase7f_best_MV_AP_window": phases["phase7f_component_stitch"].get("best_MV_AP_window"),
        "phase7f_best_MV_AP50_window": phases["phase7f_component_stitch"].get("best_MV_AP50_window"),
        "phase7f_best_ScoreFreeMatch50_window": phases["phase7f_component_stitch"].get(
            "best_ScoreFreeMatch50_window"
        ),
        "phase7f_best_delta_MV_AP50_window_vs_phase7e": phases["phase7f_component_stitch"].get(
            "best_delta_MV_AP50_window_vs_phase7e"
        ),
        "phase7f_best_delta_ScoreFreeMatch50_window_vs_base": phases["phase7f_component_stitch"].get(
            "best_delta_ScoreFreeMatch50_window_vs_base"
        ),
        "phase7f_local_diagnostic_improves": phases["phase7f_component_stitch"].get(
            "local_diagnostic_improves"
        ),
        "phase7f_formal_v102_target_achieved": phases["phase7f_component_stitch"].get(
            "formal_v102_target_achieved"
        ),
        "phase7g_missing_frame_expansion_decision": phases["phase7g_missing_frame_expansion"].get(
            "decision"
        ),
        "phase7g_best_variant_id": phases["phase7g_missing_frame_expansion"].get("best_variant_id"),
        "phase7g_best_MV_AP_window": phases["phase7g_missing_frame_expansion"].get("best_MV_AP_window"),
        "phase7g_best_MV_AP50_window": phases["phase7g_missing_frame_expansion"].get(
            "best_MV_AP50_window"
        ),
        "phase7g_best_ScoreFreeMatch50_window": phases["phase7g_missing_frame_expansion"].get(
            "best_ScoreFreeMatch50_window"
        ),
        "phase7g_best_delta_MV_AP50_window_vs_phase7e": phases[
            "phase7g_missing_frame_expansion"
        ].get("best_delta_MV_AP50_window_vs_phase7e"),
        "phase7g_best_delta_ScoreFreeMatch50_window_vs_phase7e": phases[
            "phase7g_missing_frame_expansion"
        ].get("best_delta_ScoreFreeMatch50_window_vs_phase7e"),
        "phase7g_best_accepted_expansion_count": phases["phase7g_missing_frame_expansion"].get(
            "best_accepted_expansion_count"
        ),
        "phase7g_best_accepted_expansion_same_gt_rate": phases[
            "phase7g_missing_frame_expansion"
        ].get("best_accepted_expansion_same_gt_rate"),
        "phase7g_local_diagnostic_improves": phases["phase7g_missing_frame_expansion"].get(
            "local_diagnostic_improves"
        ),
        "phase7g_formal_v102_target_achieved": phases["phase7g_missing_frame_expansion"].get(
            "formal_v102_target_achieved"
        ),
        "phase7h_primitive_support_shape_decision": phases["phase7h_primitive_support_shape"].get(
            "decision"
        ),
        "phase7h_best_variant_id": phases["phase7h_primitive_support_shape"].get("best_variant_id"),
        "phase7h_best_MV_AP_window": phases["phase7h_primitive_support_shape"].get("best_MV_AP_window"),
        "phase7h_best_MV_AP50_window": phases["phase7h_primitive_support_shape"].get(
            "best_MV_AP50_window"
        ),
        "phase7h_best_MV_AP_scene": phases["phase7h_primitive_support_shape"].get("best_MV_AP_scene"),
        "phase7h_best_MV_AP50_scene": phases["phase7h_primitive_support_shape"].get(
            "best_MV_AP50_scene"
        ),
        "phase7h_scene_metric_computed": phases["phase7h_primitive_support_shape"].get(
            "scene_metric_computed"
        ),
        "phase7h_scene_metric_not_computed_reason": phases["phase7h_primitive_support_shape"].get(
            "scene_metric_not_computed_reason"
        ),
        "phase7h_best_ScoreFreeMatch50_window": phases["phase7h_primitive_support_shape"].get(
            "best_ScoreFreeMatch50_window"
        ),
        "phase7h_best_delta_MV_AP50_window_vs_phase7e": phases[
            "phase7h_primitive_support_shape"
        ].get("best_delta_MV_AP50_window_vs_phase7e"),
        "phase7h_best_delta_ScoreFreeMatch50_window_vs_phase7e": phases[
            "phase7h_primitive_support_shape"
        ].get("best_delta_ScoreFreeMatch50_window_vs_phase7e"),
        "phase7h_best_accepted_expansion_count": phases["phase7h_primitive_support_shape"].get(
            "best_accepted_expansion_count"
        ),
        "phase7h_best_accepted_expansion_same_gt_rate": phases[
            "phase7h_primitive_support_shape"
        ].get("best_accepted_expansion_same_gt_rate"),
        "phase7h_local_diagnostic_improves": phases["phase7h_primitive_support_shape"].get(
            "local_diagnostic_improves"
        ),
        "phase7h_formal_v102_target_achieved": phases["phase7h_primitive_support_shape"].get(
            "formal_v102_target_achieved"
        ),
        "phase7i_same_chunk_f2_comparison_decision": phases["phase7i_same_chunk_f2_comparison"].get(
            "decision"
        ),
        "phase7i_f2_MV_AP_window": phases["phase7i_same_chunk_f2_comparison"].get(
            "f2_MV_AP_window"
        ),
        "phase7i_f2_MV_AP50_window": phases["phase7i_same_chunk_f2_comparison"].get(
            "f2_MV_AP50_window"
        ),
        "phase7i_f2_MV_AP50_scene": phases["phase7i_same_chunk_f2_comparison"].get(
            "f2_MV_AP50_scene"
        ),
        "phase7i_scene_metric_computed": phases["phase7i_same_chunk_f2_comparison"].get(
            "scene_metric_computed"
        ),
        "phase7i_phase7h_MV_AP_window": phases["phase7i_same_chunk_f2_comparison"].get(
            "phase7h_MV_AP_window"
        ),
        "phase7i_phase7h_MV_AP50_window": phases["phase7i_same_chunk_f2_comparison"].get(
            "phase7h_MV_AP50_window"
        ),
        "phase7i_phase7h_MV_AP50_scene": phases["phase7i_same_chunk_f2_comparison"].get(
            "phase7h_MV_AP50_scene"
        ),
        "phase7i_delta_MV_AP_window_vs_same_chunk_f2": phases[
            "phase7i_same_chunk_f2_comparison"
        ].get("delta_MV_AP_window_vs_same_chunk_f2"),
        "phase7i_delta_MV_AP50_window_vs_same_chunk_f2": phases[
            "phase7i_same_chunk_f2_comparison"
        ].get("delta_MV_AP50_window_vs_same_chunk_f2"),
        "phase7i_delta_ScoreFreeMatch50_window_vs_same_chunk_f2": phases[
            "phase7i_same_chunk_f2_comparison"
        ].get("delta_ScoreFreeMatch50_window_vs_same_chunk_f2"),
        "phase7i_local_diagnostic_beats_same_chunk_f2": phases[
            "phase7i_same_chunk_f2_comparison"
        ].get("local_diagnostic_beats_same_chunk_f2"),
        "phase7j_f2_primitive_support_fill_decision": phases["phase7j_f2_primitive_support_fill"].get(
            "decision"
        ),
        "phase7j_best_variant_id": phases["phase7j_f2_primitive_support_fill"].get("best_variant_id"),
        "phase7j_f2_base_MV_AP_window": phases["phase7j_f2_primitive_support_fill"].get(
            "f2_base_MV_AP_window"
        ),
        "phase7j_f2_base_MV_AP50_window": phases["phase7j_f2_primitive_support_fill"].get(
            "f2_base_MV_AP50_window"
        ),
        "phase7j_f2_base_MV_AP50_scene": phases["phase7j_f2_primitive_support_fill"].get(
            "f2_base_MV_AP50_scene"
        ),
        "phase7j_scene_metric_computed": phases["phase7j_f2_primitive_support_fill"].get(
            "scene_metric_computed"
        ),
        "phase7j_best_MV_AP_window": phases["phase7j_f2_primitive_support_fill"].get(
            "best_MV_AP_window"
        ),
        "phase7j_best_MV_AP50_window": phases["phase7j_f2_primitive_support_fill"].get(
            "best_MV_AP50_window"
        ),
        "phase7j_best_MV_AP50_scene": phases["phase7j_f2_primitive_support_fill"].get(
            "best_MV_AP50_scene"
        ),
        "phase7j_best_delta_MV_AP_window_vs_f2": phases["phase7j_f2_primitive_support_fill"].get(
            "best_delta_MV_AP_window_vs_f2"
        ),
        "phase7j_best_delta_MV_AP50_window_vs_f2": phases["phase7j_f2_primitive_support_fill"].get(
            "best_delta_MV_AP50_window_vs_f2"
        ),
        "phase7j_best_delta_ScoreFreeMatch50_window_vs_f2": phases[
            "phase7j_f2_primitive_support_fill"
        ].get("best_delta_ScoreFreeMatch50_window_vs_f2"),
        "phase7j_best_accepted_fill_count": phases["phase7j_f2_primitive_support_fill"].get(
            "best_accepted_fill_count"
        ),
        "phase7j_best_accepted_fill_same_gt_rate": phases["phase7j_f2_primitive_support_fill"].get(
            "best_accepted_fill_same_gt_rate"
        ),
        "phase7j_local_diagnostic_improves": phases["phase7j_f2_primitive_support_fill"].get(
            "local_diagnostic_improves"
        ),
        "truthfulness_note": (
            "v102 now has a chunk-32 Phase5 bridge pass after adding a GT-free RADIO semantic residual barrier, "
            "but Phase6 AP repair is still blocked by Phase1 repair-space evidence. Phase7c produced clean local "
            "diagnostic components; Phase7d records their chunk32 AP; Phase7e improves local AP50 by GT-free score "
            "calibration; Phase7f GT-free stitching and Phase7g missing-frame expansion did not improve local AP50/"
            "ScoreFreeMatch50; Phase7h DA3-GIANT chunk32 primitive support improves P2 local diagnostics, but Phase7i "
            "shows it still does not beat the same scene0050/c0000 v100 F2 overlap3 local baseline; Phase7j maps "
            "Phase7h support to F2 for fill/replacement but also produces no local gain. These are not formal full-dev/"
            "holdout AP improvement claims."
        ),
        "plan_doc": _rel(PLAN_DOC),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "final_gate_rows": _rel(gate_csv),
            "code_change_rows": _rel(code_change_csv),
        },
        "phase_summaries": {name: _rel(path) for name, path in PHASE_SUMMARIES.items()},
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
