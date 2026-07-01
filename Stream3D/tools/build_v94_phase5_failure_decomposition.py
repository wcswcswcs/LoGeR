#!/usr/bin/env python3
"""Build v94 Phase5 failure decomposition from completed A-D artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/v94_phase5_failure_decomposition"
PHASE_ID = "v94_phase5_failure_decomposition"
RUN_ID = "v94_phase5_failure_decomposition"

PHASES = [
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
PHASE0 = ROOT / "outputs/audit/v94_phase0_fact_lock"
PHASE2 = ROOT / "outputs/audit/v94_phase2_seed_boundary_diagnostic"
PHASE4 = ROOT / "outputs/audit/v94_phase4_controls"
PHASE7 = ROOT / "outputs/audit/v94_phase7_competition_registry_audit"
PHASE7B = ROOT / "outputs/audit/v94_phase7b_object_specific_field_readiness"
PHASE7C = ROOT / "outputs/audit/v94_phase7c_object_axis_field_smoke"
PHASE7C_FULL_SCENE0011 = ROOT / "outputs/audit/v94_phase7c_object_axis_field_scene0011_full_dev"
PHASE7C_FULL_SCENE0050 = ROOT / "outputs/audit/v94_phase7c_object_axis_field_scene0050_full_dev"
PHASE3A_OBJECT_AXIS_FULL = ROOT / "outputs/audit/v94_phase3A_object_axis_full_dev_combined"
PHASE3A_OBJECT_AXIS_REPAIR = ROOT / "outputs/audit/v94_phase3A_object_axis_full_dev_repair_competition_softening"
PHASE3B_OBJECT_AXIS = ROOT / "outputs/audit/v94_phase3B_object_axis_propagation"
PHASE3C_OBJECT_AXIS = ROOT / "outputs/audit/v94_phase3C_object_axis_constrained_cut"
PHASE3D_OBJECT_AXIS = ROOT / "outputs/audit/v94_phase3D_object_axis_component_pooling"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


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


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _variant_family(variant_id: str) -> str:
    lower = variant_id.lower()
    if "ctrl" in lower or "control" in lower:
        return "control"
    if "whole_source" in lower or variant_id.endswith("whole_source_replay"):
        return "baseline"
    return "real"


def _collect_variant_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase_name, root in PHASES:
        for row in _read_csv(root / "variant_metric_rows.csv"):
            row = dict(row)
            row["source_phase"] = phase_name
            row["source_root"] = _rel(root)
            row["family"] = _variant_family(row.get("variant_id", ""))
            rows.append(row)
    return rows


def _best_real(rows: list[dict[str, Any]]) -> dict[str, Any]:
    real = [row for row in rows if row.get("family") == "real"]
    return max(real, key=lambda row: (_num(row.get("mean_MV_AP_window")), _num(row.get("mean_MV_AP50_window"))), default={})


def _metric_scene_rows(best: dict[str, Any]) -> list[dict[str, str]]:
    root = ROOT / str(best.get("source_root", ""))
    variant_id = str(best.get("variant_id", ""))
    return [row for row in _read_csv(root / "mv_metric_rows.csv") if row.get("variant_id") == variant_id or row.get("variant") == variant_id]


def _casebook_rows(best: dict[str, Any], limit: int = 400) -> list[dict[str, Any]]:
    root = ROOT / str(best.get("source_root", ""))
    variant_id = str(best.get("variant_id", ""))
    rows = []
    for row in _read_csv(root / "casebook_rows.csv"):
        if row.get("variant_id") == variant_id or row.get("variant") == variant_id:
            rows.append(
                {
                    "schema_version": "stream4d_v94_phase5_field_error_casebook_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    **row,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        if len(rows) >= limit:
            break
    return rows


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    phase0 = _read_json(PHASE0 / "summary.json")
    phase2 = _read_json(PHASE2 / "summary.json")
    phase4 = _read_json(PHASE4 / "summary.json")
    phase7 = _read_json(PHASE7 / "summary.json")
    phase7b = _read_json(PHASE7B / "summary.json")
    phase7c = _read_json(PHASE7C / "summary.json")
    phase7c_full_0011 = _read_json(PHASE7C_FULL_SCENE0011 / "summary.json")
    phase7c_full_0050 = _read_json(PHASE7C_FULL_SCENE0050 / "summary.json")
    phase3a_object_axis_full = _read_json(PHASE3A_OBJECT_AXIS_FULL / "summary.json")
    phase3a_object_axis_repair = _read_json(PHASE3A_OBJECT_AXIS_REPAIR / "summary.json")
    phase3b_object_axis = _read_json(PHASE3B_OBJECT_AXIS / "summary.json")
    phase3c_object_axis = _read_json(PHASE3C_OBJECT_AXIS / "summary.json")
    phase3d_object_axis = _read_json(PHASE3D_OBJECT_AXIS / "summary.json")
    rows = _collect_variant_rows()
    best = _best_real(rows)
    scene_rows = _metric_scene_rows(best)
    best_variant = str(best.get("variant_id", ""))
    best_phase = str(best.get("source_phase", ""))
    mv_ap = _num(best.get("mean_MV_AP_window"))
    ap50 = _num(best.get("mean_MV_AP50_window"))
    scorefree = _num(best.get("mean_score_free_Match50_window"))
    ap_to_scorefree_gap = scorefree - ap50
    control_ap = _num(phase4.get("best_control_MV_AP_window", phase0.get("best_control_MV_AP_window")))
    control_ap50 = _num(phase4.get("best_control_MV_AP50_window", phase0.get("best_control_MV_AP50_window")))
    required_ap = _num(phase0.get("required_MV_AP_window"))
    required_ap50 = _num(phase0.get("required_MV_AP50_window"))

    gt_top_iou_rows: list[dict[str, Any]] = []
    pred_top_iou_rows: list[dict[str, Any]] = []
    extent_error_rows: list[dict[str, Any]] = []
    grouping_error_rows: list[dict[str, Any]] = []
    ranking_error_rows: list[dict[str, Any]] = []
    for row in scene_rows:
        gt_count = _num(row.get("gt_object_count"))
        pred_count = _num(row.get("pred_object_count"))
        gt_top_iou_rows.append(
            {
                "schema_version": "stream4d_v94_phase5_gt_top_iou_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": best_variant,
                "scene_id": row.get("scene_id", ""),
                "gt_best_iou_mean": row.get("gt_best_iou_mean", ""),
                "gt_best_iou_median": row.get("gt_best_iou_median", ""),
                "gt_best_iou_max": row.get("gt_best_iou_max", ""),
                "gt_recall_best_iou_ge_050": row.get("gt_recall_best_iou_ge_050", ""),
                "diagnostic_only": True,
            }
        )
        pred_top_iou_rows.append(
            {
                "schema_version": "stream4d_v94_phase5_pred_top_iou_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": best_variant,
                "scene_id": row.get("scene_id", ""),
                "pred_best_iou_mean": row.get("pred_best_iou_mean", ""),
                "pred_object_count": row.get("pred_object_count", ""),
                "gt_object_count": row.get("gt_object_count", ""),
                "diagnostic_only": True,
            }
        )
        extent_error_rows.append(
            {
                "schema_version": "stream4d_v94_phase5_extent_error_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": best_variant,
                "scene_id": row.get("scene_id", ""),
                "MV_AP50_window": row.get("MV_AP50_window", row.get("MV_AP50", "")),
                "ScoreFreeMatch50_window": row.get("score_free_Match50_window", ""),
                "AP_to_scorefree_gap": _num(row.get("score_free_Match50_window")) - _num(row.get("MV_AP50_window", row.get("MV_AP50"))),
                "gt_best_iou_mean": row.get("gt_best_iou_mean", ""),
                "gt_best_iou_median": row.get("gt_best_iou_median", ""),
                "boundary_miss_rate_diagnostic_proxy": 1.0 - _num(row.get("MV_AP50_window", row.get("MV_AP50"))),
                "diagnostic_only": True,
            }
        )
        pred_gt_ratio = pred_count / max(1.0, gt_count)
        grouping_error_rows.append(
            {
                "schema_version": "stream4d_v94_phase5_grouping_error_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": best_variant,
                "scene_id": row.get("scene_id", ""),
                "pred_object_count": pred_count,
                "gt_object_count": gt_count,
                "pred_to_gt_object_ratio": pred_gt_ratio,
                "fragmentation_rate_proxy": max(0.0, pred_gt_ratio - 1.0),
                "overmerge_rate_proxy": max(0.0, 1.0 - pred_gt_ratio),
                "proxy_note": "Count-ratio proxy; not a GT association fragmentation decomposition.",
                "diagnostic_only": True,
            }
        )
        ranking_error_rows.append(
            {
                "schema_version": "stream4d_v94_phase5_ranking_error_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": best_variant,
                "scene_id": row.get("scene_id", ""),
                "MV_AP50_window": row.get("MV_AP50_window", row.get("MV_AP50", "")),
                "ScoreFreeMatch50_window": row.get("score_free_Match50_window", ""),
                "AP_to_scorefree_gap": _num(row.get("score_free_Match50_window")) - _num(row.get("MV_AP50_window", row.get("MV_AP50"))),
                "ranking_blocker_scene_gate": (_num(row.get("score_free_Match50_window")) - _num(row.get("MV_AP50_window", row.get("MV_AP50")))) >= 0.10,
                "diagnostic_only": True,
            }
        )

    def metric(variant_id: str, phase: str | None = None) -> dict[str, Any]:
        candidates = [row for row in rows if row.get("variant_id") == variant_id and (phase is None or row.get("source_phase") == phase)]
        return candidates[0] if candidates else {}

    a1 = metric("A1r_greedy_d4rt_radio_no_edge_replay", "phase3A_edge_repair")
    a5 = metric("A5_greedy_d4rt_radio_all_edges", "phase3A_main")
    a9 = metric("A9_source_preserve_soft_edge", "phase3A_edge_repair")
    a10 = metric("A10_relaxed_no_edge_source_preserve", "phase3A_edge_repair")
    c4 = metric("C4_cut_with_strong_edge_barrier", "phase3C_constrained_cut")
    d4 = metric("D4_component_pool_strong_boundary", "phase3D_component_pooling")
    d5 = metric("D5_component_pool_large_merge", "phase3D_component_pooling")
    ob2 = metric("OB2_edge_smooth_wta_high_recall", "phase3B_object_axis_propagation")
    oc3 = metric("OC3_alpha_expansion_approx_3round", "phase3C_object_axis_constrained_cut")
    od5 = metric("OD5_component_pool_large_merge", "phase3D_object_axis_component_pooling")
    # A5 lives in the main Phase3A artifact, which is not in PHASES. Read it directly.
    a5_rows = _read_csv(ROOT / "outputs/audit/v94_phase3A_greedy_assignment/variant_metric_rows.csv")
    if a5_rows:
        a5 = next((row for row in a5_rows if row.get("variant_id") == "A5_greedy_d4rt_radio_all_edges"), {})

    edge_overcut_evidence = {
        "A5_minus_A1r_MV_AP": _num(a5.get("mean_MV_AP_window")) - _num(a1.get("mean_MV_AP_window")),
        "A5_minus_A1r_AP50": _num(a5.get("mean_MV_AP50_window")) - _num(a1.get("mean_MV_AP50_window")),
        "C4_minus_C3_MV_AP": _num(c4.get("mean_MV_AP_window")) - _num(metric("C3_alpha_expansion_approx_3round", "phase3C_constrained_cut").get("mean_MV_AP_window")),
        "D4_minus_D5_MV_AP": _num(d4.get("mean_MV_AP_window")) - _num(d5.get("mean_MV_AP_window")),
        "object_axis_OC3_minus_OB2_MV_AP": _num(oc3.get("mean_MV_AP_window")) - _num(ob2.get("mean_MV_AP_window")),
        "object_axis_OC3_minus_OB2_AP50": _num(oc3.get("mean_MV_AP50_window")) - _num(ob2.get("mean_MV_AP50_window")),
        "object_axis_OD5_minus_OC3_MV_AP": _num(od5.get("mean_MV_AP_window")) - _num(oc3.get("mean_MV_AP_window")),
        "object_axis_OD5_minus_OC3_AP50": _num(od5.get("mean_MV_AP50_window")) - _num(oc3.get("mean_MV_AP50_window")),
    }

    blockers = {
        "MATERIALIZER_BUG": False,
        "SEED_BLOCKER": False,
        "BOUNDARY_BLOCKER": True,
        "COMPETITION_BLOCKER": phase7.get("competition_blocker", "inconclusive_current_variant_key_multi_object_rate_is_sparse"),
        "RADIO_BLOCKER": "not_primary_RADIO_diagnostic_auc_available_but_readout_does_not_convert_to_AP",
        "D4RT_BLOCKER": True,
        "EDGE_BLOCKER": True,
        "RANKING_BLOCKER": ap_to_scorefree_gap >= 0.10,
        "CONTROL_BIAS_BLOCKER": (mv_ap < control_ap) or (ap50 < control_ap50),
    }
    summary = {
        "schema": "stream4d_v94_phase5_failure_decomposition_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": "NO_GO_V94_FAILURE_DECOMPOSITION",
        "best_real_variant_id": best_variant,
        "best_real_phase": best_phase,
        "best_real_MV_AP_window": mv_ap,
        "best_real_MV_AP50_window": ap50,
        "best_real_MV_AP25_window": _num(best.get("mean_MV_AP25_window")),
        "ScoreFreeMatch50_window": scorefree,
        "AP_to_scorefree_gap": ap_to_scorefree_gap,
        "score_calibration_enter_gate_pass": ap_to_scorefree_gap >= 0.10,
        "required_MV_AP_window": required_ap,
        "required_MV_AP50_window": required_ap50,
        "best_control_MV_AP_window": control_ap,
        "best_control_MV_AP50_window": control_ap50,
        "phase4_control_decision": phase4.get("decision", "not_available"),
        "phase4_control_gate_pass": bool(phase4.get("control_gate_pass", False)),
        "phase4_best_control_id": phase4.get("best_control_id", ""),
        "phase4_best_control_variant_id": phase4.get("best_control_variant_id", ""),
        "phase4_real_minus_best_control_MV_AP_window": phase4.get("real_minus_best_control_MV_AP_window", ""),
        "phase4_real_minus_edge_only_MV_AP_window": phase4.get("real_minus_edge_only_MV_AP_window", ""),
        "phase4_real_minus_random_edge_MV_AP_window": phase4.get("real_minus_random_edge_MV_AP_window", ""),
        "phase4_shuffled_D4RT_control_available": phase4.get("shuffled_D4RT_control_available", False),
        "phase4_missing_control_ids": phase4.get("missing_control_ids", []),
        "phase7_competition_registry_decision": phase7.get("decision", "not_available"),
        "phase7_competition_registry_gate_pass": bool(phase7.get("competition_registry_gate_pass", False)),
        "phase7_canonical_multi_object_rate_plan_key": phase7.get("canonical_multi_object_rate_plan_key", ""),
        "phase7_duplicate_drop_rate_raw_to_canonical": phase7.get("duplicate_drop_rate_raw_to_canonical", ""),
        "phase7_field_multi_object_plan_key_rate": phase7.get("field_multi_object_plan_key_rate", ""),
        "phase7_safe_to_materialize_current_v94": phase7.get("safe_to_materialize_current_v94", ""),
        "phase7_recommended_repair_direction": phase7.get("recommended_repair_direction", ""),
        "phase7b_object_specific_field_decision": phase7b.get("decision", "not_available"),
        "phase7b_object_specific_field_input_gate_pass": bool(phase7b.get("object_specific_field_input_gate_pass", False)),
        "phase7b_object_specific_field_blocker": phase7b.get("object_specific_field_blocker", ""),
        "phase7b_appearance_feature_hash_nonempty_count": phase7b.get("appearance_feature_hash_nonempty_count", ""),
        "phase7b_region_feature_header_has_region_vectors": phase7b.get("region_feature_header_has_region_vectors", ""),
        "phase7b_v91_slot_proto_diagnostic_gt_rate": phase7b.get("v91_slot_proto_diagnostic_gt_rate", ""),
        "phase7b_recommended_repair_direction": phase7b.get("recommended_repair_direction", ""),
        "phase7c_object_axis_field_smoke_decision": phase7c.get("decision", "not_available"),
        "phase7c_object_axis_field_smoke_gate_pass": bool(phase7c.get("object_specific_field_input_gate_pass", False)),
        "phase7c_processed_source_count": phase7c.get("processed_source_count", ""),
        "phase7c_field_unary_count_shard": phase7c.get("field_unary_count_shard", ""),
        "phase7c_cpu_parity_max_abs_diff": phase7c.get("cpu_parity_max_abs_diff", ""),
        "phase7c_cosine_backend_counts": phase7c.get("cosine_backend_counts", ""),
        "phase7c_recommended_repair_direction": phase7c.get("recommended_repair_direction", ""),
        "phase7c_full_gate_pass": bool(phase7c_full_0011.get("object_specific_field_input_gate_pass", False) and phase7c_full_0050.get("object_specific_field_input_gate_pass", False)),
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
        "mean_GT_best_IoU_window": sum(_num(row.get("gt_best_iou_mean")) for row in scene_rows) / max(1, len(scene_rows)),
        "median_GT_best_IoU_window": sum(_num(row.get("gt_best_iou_median")) for row in scene_rows) / max(1, len(scene_rows)),
        "fragmentation_rate_proxy_mean": sum(_num(row.get("fragmentation_rate_proxy")) for row in grouping_error_rows) / max(1, len(grouping_error_rows)),
        "overmerge_rate_proxy_mean": sum(_num(row.get("overmerge_rate_proxy")) for row in grouping_error_rows) / max(1, len(grouping_error_rows)),
        "unknown_overuse_rate_available": False,
        "background_overuse_rate_available": False,
        "whole_source_fallback_rate_proxy": 1.0 if _num(best.get("mean_generated_area_ratio")) >= 0.95 else 0.0,
        "seed_undercoverage_rate_proxy": 1.0 - _num(phase2.get("seed_coverage_rate"), 1.0),
        "seed_area_ratio_available": bool(phase2.get("seed_area_ratio_available", False)),
        "edge_overcut_evidence": edge_overcut_evidence,
        "boundary_miss_rate_diagnostic_proxy": 1.0 - ap50,
        "blockers": blockers,
        "blocker_conclusion": (
            "A-D converge to high-recall/source-preserving masks but remain below whole-source, v91, and locked controls. "
            "Hard edge/barrier/negative variants repeatedly reduce AP. D4RT boundary support is low and noisy from Phase2, "
            "but prior adaptive sampling evidence says density alone did not improve MV_AP. No score calibration because AP-to-scorefree gap is below 0.10. "
            "The promoted object-axis full-dev A plus B/C/D propagation, constrained-cut, and component-pooling repairs fix the missing object-specific field input but still lose to whole-source/global best and dev gates."
        ),
        "phase6_adaptive_d4rt_recommendation": "prepare_branch_only_do_not_launch_without_new_sampling_budget_and_readout_fix",
        "holdout_executed": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "diagnostic_gt_used_for_thresholds": False,
        "row_counts": {
            "gt_top_iou_rows": len(gt_top_iou_rows),
            "pred_top_iou_rows": len(pred_top_iou_rows),
            "extent_error_rows": len(extent_error_rows),
            "grouping_error_rows": len(grouping_error_rows),
            "ranking_error_rows": len(ranking_error_rows),
        },
    }

    field_casebook = _casebook_rows(best)
    _write_csv(OUT / "gt_top_iou_rows.csv", gt_top_iou_rows)
    _write_csv(OUT / "pred_top_iou_rows.csv", pred_top_iou_rows)
    _write_csv(OUT / "extent_error_rows.csv", extent_error_rows)
    _write_csv(OUT / "grouping_error_rows.csv", grouping_error_rows)
    _write_csv(OUT / "ranking_error_rows.csv", ranking_error_rows)
    _write_csv(OUT / "field_error_casebook_rows.csv", field_casebook)
    _write_json(OUT / "blocker_summary.json", summary)
    outputs = [
        OUT / "gt_top_iou_rows.csv",
        OUT / "pred_top_iou_rows.csv",
        OUT / "extent_error_rows.csv",
        OUT / "grouping_error_rows.csv",
        OUT / "ranking_error_rows.csv",
        OUT / "field_error_casebook_rows.csv",
        OUT / "blocker_summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
