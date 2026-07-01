#!/usr/bin/env python3
"""Build v94 Phase8 dev decision from completed v94 diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/v94_phase8_dev_decision"
PHASE_ID = "v94_phase8_dev_decision"
RUN_ID = "v94_phase8_dev_decision"
PHASE_ROOTS = [
    ("phase3A_object_axis_full_dev", ROOT / "outputs/audit/v94_phase3A_object_axis_full_dev_combined"),
    ("phase3A_object_axis_competition_softening", ROOT / "outputs/audit/v94_phase3A_object_axis_full_dev_repair_competition_softening"),
    ("phase3B_object_axis_propagation", ROOT / "outputs/audit/v94_phase3B_object_axis_propagation"),
    ("phase3C_object_axis_constrained_cut", ROOT / "outputs/audit/v94_phase3C_object_axis_constrained_cut"),
    ("phase3D_object_axis_component_pooling", ROOT / "outputs/audit/v94_phase3D_object_axis_component_pooling"),
    ("phase3A_edge_repair", ROOT / "outputs/audit/v94_phase3A_greedy_assignment_edge_repair"),
    ("phase3B_random_walker", ROOT / "outputs/audit/v94_phase3B_random_walker"),
    ("phase3C_constrained_cut", ROOT / "outputs/audit/v94_phase3C_constrained_cut"),
    ("phase3D_component_pooling", ROOT / "outputs/audit/v94_phase3D_component_pooling"),
]
PHASE0 = ROOT / "outputs/audit/v94_phase0_fact_lock/summary.json"
PHASE4 = ROOT / "outputs/audit/v94_phase4_controls/summary.json"
PHASE5 = ROOT / "outputs/audit/v94_phase5_failure_decomposition/blocker_summary.json"
PHASE6 = ROOT / "outputs/audit/v94_phase6_adaptive_d4rt_prior_audit/summary.json"
PHASE7 = ROOT / "outputs/audit/v94_phase7_competition_registry_audit/summary.json"
PHASE7B = ROOT / "outputs/audit/v94_phase7b_object_specific_field_readiness/summary.json"
PHASE7C = ROOT / "outputs/audit/v94_phase7c_object_axis_field_smoke/summary.json"
PHASE7C_FULL_SCENE0011 = ROOT / "outputs/audit/v94_phase7c_object_axis_field_scene0011_full_dev/summary.json"
PHASE7C_FULL_SCENE0050 = ROOT / "outputs/audit/v94_phase7c_object_axis_field_scene0050_full_dev/summary.json"
PHASE3A_OBJECT_AXIS_FULL = ROOT / "outputs/audit/v94_phase3A_object_axis_full_dev_combined/summary.json"
PHASE3A_OBJECT_AXIS_REPAIR = ROOT / "outputs/audit/v94_phase3A_object_axis_full_dev_repair_competition_softening/summary.json"
PHASE3B_OBJECT_AXIS = ROOT / "outputs/audit/v94_phase3B_object_axis_propagation/summary.json"
PHASE3C_OBJECT_AXIS = ROOT / "outputs/audit/v94_phase3C_object_axis_constrained_cut/summary.json"
PHASE3D_OBJECT_AXIS = ROOT / "outputs/audit/v94_phase3D_object_axis_component_pooling/summary.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _bool_false(value: Any) -> bool:
    return str(value).lower() in ("false", "0", "")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _variant_family(variant_id: str) -> str:
    lower = variant_id.lower()
    if "ctrl" in lower or "control" in lower:
        return "control"
    if "whole_source" in lower or variant_id.endswith("whole_source_replay"):
        return "baseline"
    return "real"


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    phase0 = _read_json(PHASE0)
    phase4 = _read_json(PHASE4)
    phase5 = _read_json(PHASE5)
    phase6 = _read_json(PHASE6)
    phase7 = _read_json(PHASE7)
    phase7b = _read_json(PHASE7B)
    phase7c = _read_json(PHASE7C)
    phase7c_full_0011 = _read_json(PHASE7C_FULL_SCENE0011)
    phase7c_full_0050 = _read_json(PHASE7C_FULL_SCENE0050)
    phase3a_object_axis_full = _read_json(PHASE3A_OBJECT_AXIS_FULL)
    phase3a_object_axis_repair = _read_json(PHASE3A_OBJECT_AXIS_REPAIR)
    phase3b_object_axis = _read_json(PHASE3B_OBJECT_AXIS)
    phase3c_object_axis = _read_json(PHASE3C_OBJECT_AXIS)
    phase3d_object_axis = _read_json(PHASE3D_OBJECT_AXIS)
    rows: list[dict[str, Any]] = []
    for phase_name, root in PHASE_ROOTS:
        for row in _read_csv(root / "variant_metric_rows.csv"):
            variant_id = row.get("variant_id", "")
            family = _variant_family(variant_id)
            rows.append(
                {
                    "schema_version": "stream4d_v94_phase8_variant_rank_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "source_phase": phase_name,
                    "source_artifact": _rel(root / "variant_metric_rows.csv"),
                    "variant_id": variant_id,
                    "family": family,
                    "MV_AP_window": _num(row.get("mean_MV_AP_window")),
                    "MV_AP50_window": _num(row.get("mean_MV_AP50_window")),
                    "MV_AP25_window": _num(row.get("mean_MV_AP25_window")),
                    "ScoreFreeMatch50_window": _num(row.get("mean_score_free_Match50_window")),
                    "mean_generated_area_ratio": _num(row.get("mean_generated_area_ratio")),
                    "same_frame_collision_count": _num(row.get("same_frame_collision_count")),
                    "missing_mask_raster_count": _num(row.get("missing_mask_raster_count")),
                    "uses_gt_for_prediction": row.get("uses_gt_for_prediction", "False"),
                    "uses_future": row.get("uses_future", "False"),
                }
            )
    rows.sort(key=lambda row: (row["family"] == "real", row["MV_AP_window"], row["MV_AP50_window"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    real_rows = [row for row in rows if row["family"] == "real"]
    best_real = max(real_rows, key=lambda row: (row["MV_AP_window"], row["MV_AP50_window"]), default={})
    required_ap = _num(phase0.get("required_MV_AP_window"))
    required_ap50 = _num(phase0.get("required_MV_AP50_window"))
    progress_gate = bool(best_real and best_real["MV_AP_window"] >= required_ap and best_real["MV_AP50_window"] >= required_ap50)
    provenance_gate = bool(
        best_real
        and best_real["same_frame_collision_count"] == 0
        and best_real["missing_mask_raster_count"] == 0
        and _bool_false(best_real["uses_gt_for_prediction"])
        and _bool_false(best_real["uses_future"])
    )
    control_gate = bool(phase4.get("control_gate_pass", False)) if phase4 else bool(
        best_real
        and best_real["MV_AP_window"] >= _num(phase0.get("best_control_MV_AP_window")) + 0.010
        and best_real["MV_AP50_window"] >= _num(phase0.get("best_control_MV_AP50_window")) + 0.015
    )
    competition_registry_gate = bool(phase7.get("competition_registry_gate_pass", False)) if phase7 else False
    phase7c_object_axis_smoke_gate = bool(phase7c.get("object_specific_field_input_gate_pass", False)) if phase7c else False
    phase7c_full_gate = bool(
        phase7c_full_0011.get("object_specific_field_input_gate_pass", False)
        and phase7c_full_0050.get("object_specific_field_input_gate_pass", False)
    )
    phase7b_object_specific_field_input_gate = bool(phase7b.get("object_specific_field_input_gate_pass", False)) if phase7b else False
    object_specific_field_input_gate = phase7c_full_gate or phase7c_object_axis_smoke_gate or phase7b_object_specific_field_input_gate
    object_specific_field_input_gate_scope = (
        "phase7c_full_dev_field_shards_with_source_skips"
        if phase7c_full_gate
        else ("phase7c_smoke_not_full_dev_materialization" if phase7c_object_axis_smoke_gate else "phase7b_existing_artifact_readiness")
    )
    decision_rows = [
        {
            "schema_version": "stream4d_v94_phase8_decision_matrix_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "dev_progress_gate",
            "pass": progress_gate,
            "observed_MV_AP_window": best_real.get("MV_AP_window", ""),
            "observed_MV_AP50_window": best_real.get("MV_AP50_window", ""),
            "required_MV_AP_window": required_ap,
            "required_MV_AP50_window": required_ap50,
        },
        {
            "schema_version": "stream4d_v94_phase8_decision_matrix_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "control_attribution_gate",
            "pass": control_gate,
            "phase4_decision": phase4.get("decision", ""),
            "best_control_id": phase4.get("best_control_id", ""),
            "best_control_variant_id": phase4.get("best_control_variant_id", ""),
            "best_control_MV_AP_window": phase4.get("best_control_MV_AP_window", phase0.get("best_control_MV_AP_window", "")),
            "best_control_MV_AP50_window": phase4.get("best_control_MV_AP50_window", phase0.get("best_control_MV_AP50_window", "")),
            "real_minus_best_control_MV_AP_window": phase4.get("real_minus_best_control_MV_AP_window", ""),
            "real_minus_edge_only_MV_AP_window": phase4.get("real_minus_edge_only_MV_AP_window", ""),
            "real_minus_random_edge_MV_AP_window": phase4.get("real_minus_random_edge_MV_AP_window", ""),
            "shuffled_D4RT_control_available": phase4.get("shuffled_D4RT_control_available", ""),
        },
        {
            "schema_version": "stream4d_v94_phase8_decision_matrix_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "provenance_materializer_gate",
            "pass": provenance_gate,
            "same_frame_collision_count": best_real.get("same_frame_collision_count", ""),
            "missing_mask_raster_count": best_real.get("missing_mask_raster_count", ""),
            "uses_gt_for_prediction": best_real.get("uses_gt_for_prediction", ""),
            "uses_future": best_real.get("uses_future", ""),
        },
        {
            "schema_version": "stream4d_v94_phase8_decision_matrix_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "phase6_adaptive_d4rt_gate",
            "pass": False,
            "phase6_decision": phase6.get("decision", ""),
            "A512_minus_G16_MV_AP_window": phase6.get("A512_minus_G16_MV_AP_window", ""),
            "runtime_budget_pass": phase6.get("runtime_budget_pass", ""),
        },
        {
            "schema_version": "stream4d_v94_phase8_decision_matrix_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "competition_registry_gate",
            "pass": competition_registry_gate,
            "phase7_decision": phase7.get("decision", "not_available"),
            "competition_blocker": phase7.get("competition_blocker", ""),
            "canonical_multi_object_rate_plan_key": phase7.get("canonical_multi_object_rate_plan_key", ""),
            "duplicate_drop_rate_raw_to_canonical": phase7.get("duplicate_drop_rate_raw_to_canonical", ""),
            "field_multi_object_plan_key_rate": phase7.get("field_multi_object_plan_key_rate", ""),
            "safe_to_materialize_current_v94": phase7.get("safe_to_materialize_current_v94", ""),
        },
        {
            "schema_version": "stream4d_v94_phase8_decision_matrix_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "object_specific_field_input_gate",
            "pass": object_specific_field_input_gate,
            "gate_scope": object_specific_field_input_gate_scope,
            "phase7b_decision": phase7b.get("decision", "not_available"),
            "phase7c_decision": phase7c.get("decision", "not_available"),
            "phase7c_processed_source_count": phase7c.get("processed_source_count", ""),
            "phase7c_field_unary_count_shard": phase7c.get("field_unary_count_shard", ""),
            "phase7c_cpu_parity_max_abs_diff": phase7c.get("cpu_parity_max_abs_diff", ""),
            "phase7c_cosine_backend_counts": phase7c.get("cosine_backend_counts", ""),
            "phase7c_full_gate_pass": phase7c_full_gate,
            "phase7c_full_processed_source_count": _num(phase7c_full_0011.get("processed_source_count")) + _num(phase7c_full_0050.get("processed_source_count")),
            "phase7c_full_selected_source_count": _num(phase7c_full_0011.get("selected_source_count")) + _num(phase7c_full_0050.get("selected_source_count")),
            "phase7c_full_field_unary_count_shard": _num(phase7c_full_0011.get("field_unary_count_shard")) + _num(phase7c_full_0050.get("field_unary_count_shard")),
            "phase7c_full_failure_count": _num(phase7c_full_0011.get("failure_count")) + _num(phase7c_full_0050.get("failure_count")),
            "phase3A_object_axis_full_decision": phase3a_object_axis_full.get("decision", "not_available"),
            "phase3A_object_axis_full_best_real_variant_id": phase3a_object_axis_full.get("best_real_variant_id", ""),
            "phase3A_object_axis_full_best_real_MV_AP_window": phase3a_object_axis_full.get("best_real_MV_AP_window", ""),
            "phase3A_object_axis_full_best_real_MV_AP50_window": phase3a_object_axis_full.get("best_real_MV_AP50_window", ""),
            "phase3A_object_axis_full_dev_progress_gate_pass": phase3a_object_axis_full.get("dev_progress_gate_pass", ""),
            "phase3A_object_axis_repair_decision": phase3a_object_axis_repair.get("decision", "not_available"),
            "phase3A_object_axis_repair_best_real_variant_id": phase3a_object_axis_repair.get("best_real_variant_id", ""),
            "phase3A_object_axis_repair_best_real_MV_AP_window": phase3a_object_axis_repair.get("best_real_MV_AP_window", ""),
            "phase3A_object_axis_repair_best_real_MV_AP50_window": phase3a_object_axis_repair.get("best_real_MV_AP50_window", ""),
            "phase3A_object_axis_repair_dev_progress_gate_pass": phase3a_object_axis_repair.get("dev_progress_gate_pass", ""),
            "phase3B_object_axis_decision": phase3b_object_axis.get("decision", "not_available"),
            "phase3B_object_axis_best_real_variant_id": phase3b_object_axis.get("best_real_variant_id", ""),
            "phase3B_object_axis_best_real_MV_AP_window": phase3b_object_axis.get("best_real_MV_AP_window", ""),
            "phase3B_object_axis_best_real_MV_AP50_window": phase3b_object_axis.get("best_real_MV_AP50_window", ""),
            "phase3B_object_axis_dev_progress_gate_pass": phase3b_object_axis.get("dev_progress_gate_pass", ""),
            "phase3C_object_axis_decision": phase3c_object_axis.get("decision", "not_available"),
            "phase3C_object_axis_best_real_variant_id": phase3c_object_axis.get("best_real_variant_id", ""),
            "phase3C_object_axis_best_real_MV_AP_window": phase3c_object_axis.get("best_real_MV_AP_window", ""),
            "phase3C_object_axis_best_real_MV_AP50_window": phase3c_object_axis.get("best_real_MV_AP50_window", ""),
            "phase3C_object_axis_candidate_gate_pass": phase3c_object_axis.get("phase3C_candidate_gate_pass", ""),
            "phase3C_object_axis_dev_progress_gate_pass": phase3c_object_axis.get("dev_progress_gate_pass", ""),
            "phase3D_object_axis_decision": phase3d_object_axis.get("decision", "not_available"),
            "phase3D_object_axis_best_real_variant_id": phase3d_object_axis.get("best_real_variant_id", ""),
            "phase3D_object_axis_best_real_MV_AP_window": phase3d_object_axis.get("best_real_MV_AP_window", ""),
            "phase3D_object_axis_best_real_MV_AP50_window": phase3d_object_axis.get("best_real_MV_AP50_window", ""),
            "phase3D_object_axis_candidate_gate_pass": phase3d_object_axis.get("phase3D_candidate_gate_pass", ""),
            "phase3D_object_axis_dev_progress_gate_pass": phase3d_object_axis.get("dev_progress_gate_pass", ""),
            "object_specific_field_blocker": phase7b.get("object_specific_field_blocker", ""),
            "appearance_feature_hash_nonempty_count": phase7b.get("appearance_feature_hash_nonempty_count", ""),
            "region_feature_header_has_region_vectors": phase7b.get("region_feature_header_has_region_vectors", ""),
            "v91_mask_feature_vector_count": phase7b.get("v91_mask_feature_vector_count", ""),
            "v91_slot_proto_diagnostic_gt_rate": phase7b.get("v91_slot_proto_diagnostic_gt_rate", ""),
        },
        {
            "schema_version": "stream4d_v94_phase8_decision_matrix_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "score_calibration_gate",
            "pass": bool(phase5.get("score_calibration_enter_gate_pass", False)),
            "AP_to_scorefree_gap": phase5.get("AP_to_scorefree_gap", ""),
        },
    ]
    final_pass = progress_gate and control_gate and provenance_gate and competition_registry_gate and object_specific_field_input_gate
    summary = {
        "schema": "stream4d_v94_phase8_dev_decision_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "final_decision": "PASS_V94_FREEZE_READY" if final_pass else "NO_GO_LOCAL_MV_AP_WINDOW",
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_phase": best_real.get("source_phase", ""),
        "best_real_MV_AP_window": best_real.get("MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("MV_AP50_window", ""),
        "required_MV_AP_window": required_ap,
        "required_MV_AP50_window": required_ap50,
        "dev_progress_gate_pass": progress_gate,
        "control_attribution_gate_pass": control_gate,
        "phase4_control_decision": phase4.get("decision", "not_available"),
        "phase4_best_control_id": phase4.get("best_control_id", ""),
        "phase4_best_control_variant_id": phase4.get("best_control_variant_id", ""),
        "phase4_real_minus_best_control_MV_AP_window": phase4.get("real_minus_best_control_MV_AP_window", ""),
        "phase4_real_minus_edge_only_MV_AP_window": phase4.get("real_minus_edge_only_MV_AP_window", ""),
        "phase4_shuffled_D4RT_control_available": phase4.get("shuffled_D4RT_control_available", False),
        "provenance_materializer_gate_pass": provenance_gate,
        "score_calibration_enter_gate_pass": bool(phase5.get("score_calibration_enter_gate_pass", False)),
        "phase6_adaptive_d4rt_gate_pass": False,
        "competition_registry_gate_pass": competition_registry_gate,
        "phase7_competition_registry_decision": phase7.get("decision", "not_available"),
        "phase7_competition_blocker": phase7.get("competition_blocker", ""),
        "phase7_canonical_multi_object_rate_plan_key": phase7.get("canonical_multi_object_rate_plan_key", ""),
        "phase7_duplicate_drop_rate_raw_to_canonical": phase7.get("duplicate_drop_rate_raw_to_canonical", ""),
        "phase7_field_multi_object_plan_key_rate": phase7.get("field_multi_object_plan_key_rate", ""),
        "phase7_safe_to_materialize_current_v94": phase7.get("safe_to_materialize_current_v94", ""),
        "phase7_recommended_repair_direction": phase7.get("recommended_repair_direction", ""),
        "object_specific_field_input_gate_pass": object_specific_field_input_gate,
        "object_specific_field_input_gate_scope": object_specific_field_input_gate_scope,
        "phase7b_object_specific_field_decision": phase7b.get("decision", "not_available"),
        "phase7b_object_specific_field_blocker": phase7b.get("object_specific_field_blocker", ""),
        "phase7b_appearance_feature_hash_nonempty_count": phase7b.get("appearance_feature_hash_nonempty_count", ""),
        "phase7b_region_feature_header_has_region_vectors": phase7b.get("region_feature_header_has_region_vectors", ""),
        "phase7b_v91_mask_feature_vector_count": phase7b.get("v91_mask_feature_vector_count", ""),
        "phase7b_v91_slot_proto_diagnostic_gt_rate": phase7b.get("v91_slot_proto_diagnostic_gt_rate", ""),
        "phase7b_recommended_repair_direction": phase7b.get("recommended_repair_direction", ""),
        "phase7c_object_axis_field_smoke_decision": phase7c.get("decision", "not_available"),
        "phase7c_object_axis_field_smoke_gate_pass": phase7c_object_axis_smoke_gate,
        "phase7c_processed_source_count": phase7c.get("processed_source_count", ""),
        "phase7c_field_unary_count_shard": phase7c.get("field_unary_count_shard", ""),
        "phase7c_cpu_parity_max_abs_diff": phase7c.get("cpu_parity_max_abs_diff", ""),
        "phase7c_cosine_backend_counts": phase7c.get("cosine_backend_counts", ""),
        "phase7c_recommended_repair_direction": phase7c.get("recommended_repair_direction", ""),
        "phase7c_full_gate_pass": phase7c_full_gate,
        "phase7c_full_processed_source_count": _num(phase7c_full_0011.get("processed_source_count")) + _num(phase7c_full_0050.get("processed_source_count")),
        "phase7c_full_selected_source_count": _num(phase7c_full_0011.get("selected_source_count")) + _num(phase7c_full_0050.get("selected_source_count")),
        "phase7c_full_field_unary_count_shard": _num(phase7c_full_0011.get("field_unary_count_shard")) + _num(phase7c_full_0050.get("field_unary_count_shard")),
        "phase7c_full_failure_count": _num(phase7c_full_0011.get("failure_count")) + _num(phase7c_full_0050.get("failure_count")),
        "phase7c_full_cpu_parity_max_abs_diff": max(_num(phase7c_full_0011.get("cpu_parity_max_abs_diff")), _num(phase7c_full_0050.get("cpu_parity_max_abs_diff"))),
        "phase3A_object_axis_full_decision": phase3a_object_axis_full.get("decision", "not_available"),
        "phase3A_object_axis_full_best_real_variant_id": phase3a_object_axis_full.get("best_real_variant_id", ""),
        "phase3A_object_axis_full_best_real_MV_AP_window": phase3a_object_axis_full.get("best_real_MV_AP_window", ""),
        "phase3A_object_axis_full_best_real_MV_AP50_window": phase3a_object_axis_full.get("best_real_MV_AP50_window", ""),
        "phase3A_object_axis_full_dev_progress_gate_pass": phase3a_object_axis_full.get("dev_progress_gate_pass", ""),
        "phase3A_object_axis_repair_decision": phase3a_object_axis_repair.get("decision", "not_available"),
        "phase3A_object_axis_repair_best_real_variant_id": phase3a_object_axis_repair.get("best_real_variant_id", ""),
        "phase3A_object_axis_repair_best_real_MV_AP_window": phase3a_object_axis_repair.get("best_real_MV_AP_window", ""),
        "phase3A_object_axis_repair_best_real_MV_AP50_window": phase3a_object_axis_repair.get("best_real_MV_AP50_window", ""),
        "phase3A_object_axis_repair_dev_progress_gate_pass": phase3a_object_axis_repair.get("dev_progress_gate_pass", ""),
        "phase3B_object_axis_decision": phase3b_object_axis.get("decision", "not_available"),
        "phase3B_object_axis_best_real_variant_id": phase3b_object_axis.get("best_real_variant_id", ""),
        "phase3B_object_axis_best_real_MV_AP_window": phase3b_object_axis.get("best_real_MV_AP_window", ""),
        "phase3B_object_axis_best_real_MV_AP50_window": phase3b_object_axis.get("best_real_MV_AP50_window", ""),
        "phase3B_object_axis_dev_progress_gate_pass": phase3b_object_axis.get("dev_progress_gate_pass", ""),
        "phase3C_object_axis_decision": phase3c_object_axis.get("decision", "not_available"),
        "phase3C_object_axis_best_real_variant_id": phase3c_object_axis.get("best_real_variant_id", ""),
        "phase3C_object_axis_best_real_MV_AP_window": phase3c_object_axis.get("best_real_MV_AP_window", ""),
        "phase3C_object_axis_best_real_MV_AP50_window": phase3c_object_axis.get("best_real_MV_AP50_window", ""),
        "phase3C_object_axis_candidate_gate_pass": phase3c_object_axis.get("phase3C_candidate_gate_pass", ""),
        "phase3C_object_axis_dev_progress_gate_pass": phase3c_object_axis.get("dev_progress_gate_pass", ""),
        "phase3D_object_axis_decision": phase3d_object_axis.get("decision", "not_available"),
        "phase3D_object_axis_best_real_variant_id": phase3d_object_axis.get("best_real_variant_id", ""),
        "phase3D_object_axis_best_real_MV_AP_window": phase3d_object_axis.get("best_real_MV_AP_window", ""),
        "phase3D_object_axis_best_real_MV_AP50_window": phase3d_object_axis.get("best_real_MV_AP50_window", ""),
        "phase3D_object_axis_candidate_gate_pass": phase3d_object_axis.get("phase3D_candidate_gate_pass", ""),
        "phase3D_object_axis_dev_progress_gate_pass": phase3d_object_axis.get("dev_progress_gate_pass", ""),
        "holdout_executed": False,
        "local2history_blocked": True,
        "frozen_config_created": False,
        "no_go_reason": (
            "best real variant remains below v91/control/dev progress gates; Phase4 control attribution, A-D methods, adaptive D4RT evidence, "
            "and competition-registry audit did not clear local MV_AP_window or safe multi-object materialization gates. "
            "Phase7c repaired object-axis field inputs; object-axis Phase3A/B/C/D full-dev variants also remain below gate."
        ),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "row_counts": {
            "variant_rank_rows": len(rows),
            "decision_matrix_rows": len(decision_rows),
        },
    }
    frozen_config = {
        "schema": "stream4d_v94_phase8_frozen_config_v1",
        "frozen": False,
        "reason": "No dev candidate passed progress/control gates; holdout/local2history blocked.",
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_phase": best_real.get("source_phase", ""),
    }
    _write_csv(OUT / "variant_rank_rows.csv", rows)
    _write_csv(OUT / "decision_matrix_rows.csv", decision_rows)
    _write_json(OUT / "summary.json", summary)
    _write_json(OUT / "frozen_config.json", frozen_config)
    outputs = [OUT / "variant_rank_rows.csv", OUT / "decision_matrix_rows.csv", OUT / "summary.json", OUT / "frozen_config.json"]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
