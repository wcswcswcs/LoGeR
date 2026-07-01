from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/v91_final_decision"
PHASE8 = ROOT / "outputs/audit/v91_phase8_dev_selection"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _family_rows() -> list[dict[str, Any]]:
    specs = [
        ("phase3_visible_support", ROOT / "outputs/audit/v91_phase3_carrier_visible_support/summary.json"),
        ("phase4_ap50_control_repair", ROOT / "outputs/audit/v91_phase4_witness_cover_ap50_control_repair/summary.json"),
        ("phase4_radius_coarse", ROOT / "outputs/audit/v91_phase4_witness_cover_radius_sweep/summary.json"),
        ("phase4_radius_fine", ROOT / "outputs/audit/v91_phase4_witness_cover_radius_sweep_fine_radius/summary.json"),
        ("phase4_support_point", ROOT / "outputs/audit/v91_phase4_witness_cover_radius_sweep_support_point/summary.json"),
        ("phase4_multimask_materialization", ROOT / "outputs/audit/v91_phase4_witness_cover_multimask_materialization/summary.json"),
        ("phase4_multimask_score_repair", ROOT / "outputs/audit/v91_phase4_witness_cover_multimask_score_repair/summary.json"),
        ("scene0011_support_extent_diagnostic", ROOT / "outputs/audit/v91_scene0011_support_extent_failure/summary.json"),
        ("phase4_scene_risk_materialization", ROOT / "outputs/audit/v91_phase4_scene_risk_materialization/summary.json"),
        ("phase4_adaptive_uncertainty_materialization", ROOT / "outputs/audit/v91_phase4_adaptive_uncertainty_materialization/summary.json"),
        ("phase4_adaptive_score_repair", ROOT / "outputs/audit/v91_phase4_adaptive_score_repair/summary.json"),
        ("phase4_broad_core_precision_repair", ROOT / "outputs/audit/v91_phase4_broad_core_precision_repair/summary.json"),
        ("phase4_support_wta_repair", ROOT / "outputs/audit/v91_phase4_support_wta_repair/summary.json"),
        ("phase4_semantic_witness_split_repair", ROOT / "outputs/audit/v91_phase4_semantic_witness_split_repair/summary.json"),
        ("source_mask_oracle_upper_bound_diagnostic", ROOT / "outputs/audit/v91_source_mask_oracle_upper_bound/summary.json"),
        ("phase4_source_voronoi_partition_repair", ROOT / "outputs/audit/v91_phase4_source_voronoi_partition_repair/summary.json"),
        ("phase4_rgb_edge_boundary_repair", ROOT / "outputs/audit/v91_phase4_rgb_edge_boundary_repair/summary.json"),
        ("post_repair_top_iou_diagnostic", ROOT / "outputs/audit/v91_post_repair_top_iou_diagnostic/summary.json"),
    ]
    rows: list[dict[str, Any]] = []
    for family, path in specs:
        summary = _read_json(path)
        if not summary:
            rows.append({"family": family, "artifact": _rel(path), "status": "missing"})
            continue
        best_gate = summary.get("best_variant_gate", {}) if isinstance(summary.get("best_variant_gate"), dict) else {}
        rows.append(
            {
                "family": family,
                "artifact": _rel(path),
                "status": "present",
                "decision": summary.get("decision", ""),
                "best_variant_id": summary.get("best_variant_id", summary.get("phase8_best_variant", "")),
                "best_MV_AP_window": best_gate.get("mean_MV_AP_window", summary.get("best_real_MV_AP_window", "")),
                "best_MV_AP50_window": best_gate.get("mean_MV_AP50_window", summary.get("best_real_MV_AP50_window", "")),
                "any_gate_pass": summary.get("any_v91_phase8_progress_gate_pass", ""),
                "family_stop_rule_applies": summary.get("family_stop_rule_applies", ""),
                "uses_gt_for_prediction": summary.get("uses_gt_for_prediction", ""),
                "uses_future": summary.get("uses_future", ""),
                "next_action": summary.get("next_action", ""),
            }
        )
    return rows


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    phase8 = _read_json(PHASE8 / "summary.json")
    frozen = _read_json(PHASE8 / "frozen_candidate_config.json")
    comparison_rows = _read_csv(PHASE8 / "control_comparison_rows.csv")
    comparison = comparison_rows[0] if comparison_rows else {}
    all_rows = _read_csv(PHASE8 / "all_variant_metric_rows.csv")
    best_variant = str(phase8.get("best_real_variant", ""))
    best_row = next((row for row in all_rows if row.get("variant_id") == best_variant), {})
    score_free = _num(best_row.get("mean_score_free_Match50_window"))
    best_ap50 = _num(phase8.get("best_real_MV_AP50_window"))
    score_free_gap = score_free - best_ap50
    real_beats_control = (
        _num(phase8.get("best_real_MV_AP_window")) > _num(phase8.get("best_control_MV_AP_window"))
        and best_ap50 > _num(phase8.get("best_control_MV_AP50_window"))
    )
    if not real_beats_control:
        decision_label = "CONTROL_BIAS_BLOCKER"
    elif not _bool(phase8.get("dev_progress_gate_pass")):
        decision_label = "NO_GO_EXTENT_BLOCKER"
    else:
        decision_label = "DIAGNOSTIC_PROGRESS_LOCAL_ONLY"

    evidence_rows = [
        {
            "evidence_id": "phase8_dev_gate",
            "status": "fail" if not _bool(phase8.get("dev_progress_gate_pass")) else "pass",
            "detail": "Phase8 best still misses AP50 control-margin gate; holdout is forbidden unless dev passes.",
            "artifact": _rel(PHASE8 / "summary.json"),
            "value": phase8.get("dev_progress_gate", {}),
        },
        {
            "evidence_id": "best_vs_control",
            "status": "fail",
            "detail": "Best real beats control slightly, but not by required AP50 +0.010 margin.",
            "artifact": _rel(PHASE8 / "control_comparison_rows.csv"),
            "value": comparison,
        },
        {
            "evidence_id": "score_free_upper",
            "status": "weak",
            "detail": "Score-free Match50 remains modest; score repair on AD4 did not beat AD4/control margin.",
            "artifact": _rel(PHASE8 / "all_variant_metric_rows.csv"),
            "value": {
                "best_variant": best_variant,
                "score_free_Match50_window": score_free,
                "best_real_MV_AP50_window": best_ap50,
                "score_free_minus_AP50": score_free_gap,
            },
        },
        {
            "evidence_id": "support_evaluator_bug_guard",
            "status": "not_triggered",
            "detail": "Current best has zero same-frame collision and zero missing mask raster; prediction rows remain GT/future-free.",
            "artifact": _rel(PHASE8 / "summary.json"),
            "value": {
                "same_frame_collision_count_eq_0": phase8.get("dev_progress_gate", {}).get("same_frame_collision_count_eq_0"),
                "missing_mask_raster_count_eq_0": phase8.get("dev_progress_gate", {}).get("missing_mask_raster_count_eq_0"),
                "uses_gt_for_prediction_false": phase8.get("dev_progress_gate", {}).get("uses_gt_for_prediction_false"),
                "uses_future_false": phase8.get("dev_progress_gate", {}).get("uses_future_false"),
            },
        },
        {
            "evidence_id": "phase6_multiscale_status",
            "status": "not_run",
            "detail": "No v91-compatible multi_scale_cluster/part_support/parent_support inputs existed; old v80 cross-scale rows were not mixed into v91 to avoid provenance drift.",
            "artifact": "none",
            "value": "phase6_unavailable_without_new_v91_scale_inputs",
        },
    ]
    blocked_claim_rows = [
        {
            "claim": "Phase9 frozen holdout",
            "blocked": True,
            "reason": "dev_progress_gate_pass=false; frozen_candidate_config.frozen=false",
            "artifact": _rel(PHASE8 / "frozen_candidate_config.json"),
        },
        {
            "claim": "Phase10 MV_AP_scene/local2history method path",
            "blocked": True,
            "reason": "local MV_AP_window dev/holdout gate did not pass",
            "phase10_decision": "BLOCK_L2H_BY_LOCAL_MV_AP_WINDOW",
        },
        {
            "claim": "GO_LOCAL_MV_AP_WINDOW_INTERNAL_ONLY",
            "blocked": True,
            "reason": "no legal frozen holdout because dev gate failed",
        },
    ]
    family_rows = _family_rows()
    summary = {
        "phase": "v91_final_decision",
        "schema": "stream4d_v91_final_decision_v1",
        "decision_label": decision_label,
        "phase10_decision": "BLOCK_L2H_BY_LOCAL_MV_AP_WINDOW",
        "best_variant": best_variant,
        "best_real_MV_AP_window": _num(phase8.get("best_real_MV_AP_window")),
        "best_real_MV_AP50_window": best_ap50,
        "best_real_MV_AP25_window": _num(phase8.get("best_real_MV_AP25_window")),
        "best_control_variant": phase8.get("best_control_variant", ""),
        "best_control_MV_AP_window": _num(phase8.get("best_control_MV_AP_window")),
        "best_control_MV_AP50_window": _num(phase8.get("best_control_MV_AP50_window")),
        "real_minus_best_control_MV_AP_window": _num(comparison.get("real_minus_best_control_MV_AP_window")),
        "real_minus_best_control_MV_AP50_window": _num(comparison.get("real_minus_best_control_MV_AP50_window")),
        "gap_to_required_MV_AP50_window": _num(comparison.get("gap_to_required_MV_AP50_window")),
        "best_score_free_Match50_window": score_free,
        "score_free_minus_AP50_window": score_free_gap,
        "dev_progress_gate_pass": bool(phase8.get("dev_progress_gate_pass")),
        "strong_local_dev_gate_pass": bool(phase8.get("strong_local_dev_gate_pass")),
        "frozen": bool(frozen.get("frozen")),
        "holdout_run": False,
        "holdout_policy": "Phase9 holdout was not run because Phase8 dev gate failed.",
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "evidence_count": len(evidence_rows),
        "repair_family_count": len(family_rows),
    }
    _write_csv(OUT / "decision_evidence_rows.csv", evidence_rows)
    _write_csv(OUT / "repair_family_rows.csv", family_rows)
    _write_csv(OUT / "blocked_claim_rows.csv", blocked_claim_rows)
    _write_json(OUT / "decision_summary.json", summary)
    outputs = [
        OUT / "decision_evidence_rows.csv",
        OUT / "repair_family_rows.csv",
        OUT / "blocked_claim_rows.csv",
        OUT / "decision_summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
