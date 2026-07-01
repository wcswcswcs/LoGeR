#!/usr/bin/env python3
"""Aggregate v97 No-Go evidence into Phase9 blocker taxonomy artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v97_phase9_failure_decomposition"
RUN_ID = "v97_phase9_failure_decomposition"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "docs", "Open-d4rt"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["schema_version"])
        writer.writeheader()
        for row in materialized:
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
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _base_row(blocker_type: str, source_artifact: Path, evidence: str, observed: Any, threshold: Any, interpretation: str, repair: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v97_phase9_blocker_metric_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "blocker_type": blocker_type,
        "source_artifact": _rel(source_artifact),
        "evidence_metric": evidence,
        "observed": observed,
        "threshold_or_required": threshold,
        "interpretation": interpretation,
        "repair_direction": repair,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    phase2_root = _project(args.phase2_root)
    stitch_root = _project(args.stitch_root)
    phase4_root = _project(args.phase4_root)
    phase5_root = _project(args.phase5_root)
    phase6_root = _project(args.phase6_root)
    phase7_root = _project(args.phase7_root)
    phase0_root = _project(args.phase0_root)

    phase0 = _read_json(phase0_root / "summary.json")
    phase2 = _read_json(phase2_root / "summary.json")
    stitch = _read_json(stitch_root / "summary.json")
    phase4 = _read_json(phase4_root / "summary.json")
    phase5 = _read_json(phase5_root / "summary.json")
    phase6 = _read_json(phase6_root / "summary.json")
    phase7 = _read_json(phase7_root / "summary.json")
    stitch_rows = _read_csv(stitch_root / "overlap_stitch_rows.csv")
    object_rows = _read_csv(phase5_root / "object_birth_metric_rows.csv")
    readout_quality_rows = _read_csv(phase7_root / "readout_quality_rows.csv")
    phase7_metric_rows = _read_csv(phase7_root / "variant_metric_rows.csv")

    blocker_rows: list[dict[str, Any]] = []
    d4rt_rows: list[dict[str, Any]] = []
    for row in stitch_rows:
        residual_p90 = _num(row.get("residual_p90_curr_to_prev"))
        residual_median = _num(row.get("residual_median_curr_to_prev"))
        scale = _num(row.get("scale_curr_to_prev"))
        transform_scale = _num(row.get("transform_scale_to_method"))
        case = {
            "schema_version": "stream4d_v97_phase9_d4rt_geometry_case_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "prev_window_id": row.get("prev_window_id", ""),
            "curr_window_id": row.get("curr_window_id", ""),
            "used_anchor_count": row.get("used_anchor_count", ""),
            "scale_curr_to_prev": scale,
            "transform_scale_to_method": transform_scale,
            "residual_median_curr_to_prev": residual_median,
            "residual_p90_curr_to_prev": residual_p90,
            "inlier_ratio_abs010_curr_to_prev": row.get("inlier_ratio_abs010_curr_to_prev", ""),
            "blocker_triggered": residual_p90 > float(args.geometry_residual_p90_threshold) or transform_scale < float(args.method_scale_min),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        d4rt_rows.append(case)
    if any(row["blocker_triggered"] for row in d4rt_rows):
        worst_p90 = max(_num(row.get("residual_p90_curr_to_prev")) for row in d4rt_rows)
        min_scale = min(_num(row.get("transform_scale_to_method"), 1.0) for row in d4rt_rows)
        blocker_rows.append(
            _base_row(
                "D4RT_GEOMETRY_BLOCKER",
                stitch_root / "overlap_stitch_rows.csv",
                "overlap_stitch_residual_p90_or_cumulative_scale",
                f"worst_residual_p90={worst_p90}, min_transform_scale_to_method={min_scale}",
                f"residual_p90<={args.geometry_residual_p90_threshold}, transform_scale_to_method>={args.method_scale_min}",
                "D4RT-only overlap stitch technically runs, but cross-window self-consistency remains poor and scale collapses.",
                "Repair D4RT overlap/self-stitch geometry before further readout tuning; do not use final GT-Sim3 in method artifacts.",
            )
        )

    semantic_loaded = _bool(phase4.get("semantic_tensor_loaded")) or str(phase4.get("semantic_source", "")).startswith("real_")
    if not semantic_loaded or not _bool(phase4.get("full_semantic_gate_pass")):
        blocker_rows.append(
            _base_row(
                "SEMANTIC_FEATURE_BLOCKER",
                phase4_root / "summary.json",
                "full_semantic_gate_pass",
                phase4.get("full_semantic_gate_pass"),
                "true with real RADIO/DINO dense or region feature contribution",
                "Phase4 remains proxy/neutral rather than real semantic-aware affinity.",
                "Load and verify real RADIO/DINO features at micro-primitive coordinates before claiming semantic-aware affinity.",
            )
        )

    object_case_rows: list[dict[str, Any]] = []
    for row in object_rows:
        assigned = _num(row.get("assigned_micro_primitive_count"))
        unassigned = _num(row.get("unassigned_micro_primitive_count"))
        total = assigned + unassigned
        coverage = float(assigned / max(1.0, total))
        key_cov = _num(row.get("keypoint_coverage_rate"))
        object_count_mean = _num(row.get("object_count_per_window_mean"))
        triggered = (coverage < float(args.object_assignment_coverage_min)) or (key_cov < float(args.keypoint_coverage_min)) or not (20 <= object_count_mean <= 400)
        object_case_rows.append(
            {
                "schema_version": "stream4d_v97_phase9_object_birth_case_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": row.get("variant_id", ""),
                "birth_family": row.get("birth_family", ""),
                "object_count_per_window_mean": object_count_mean,
                "keypoint_coverage_rate": key_cov,
                "assigned_micro_primitive_count": assigned,
                "unassigned_micro_primitive_count": unassigned,
                "assignment_coverage_rate": coverage,
                "cannot_link_violation_count": row.get("cannot_link_violation_count", ""),
                "same_frame_violation_count": row.get("same_frame_violation_count", ""),
                "blocker_triggered": triggered,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    best_phase5 = phase5.get("best_variant", {}) if isinstance(phase5.get("best_variant", {}), dict) else {}
    if _num(best_phase5.get("keypoint_coverage_rate")) < float(args.keypoint_coverage_min):
        blocker_rows.append(
            _base_row(
                "OBJECT_BIRTH_GROUPING_BLOCKER",
                phase5_root / "summary.json",
                "best_variant_keypoint_coverage_rate",
                best_phase5.get("keypoint_coverage_rate"),
                f">={args.keypoint_coverage_min}",
                "Best Phase5 diagnostic variant passes object-count sanity by assigning only a small minority of primitives.",
                "Add coverage-aware gate/repair; do not advance low-coverage object birth as a method success candidate.",
            )
        )

    render_case_rows: list[dict[str, Any]] = []
    metric_by_variant = {row.get("variant_id", ""): row for row in phase7_metric_rows}
    for row in readout_quality_rows:
        variant = row.get("variant_id", "")
        metric = metric_by_variant.get(variant, {})
        support_iou = _num(row.get("support_to_selected_mask_IoU_mean"))
        precision = _num(row.get("mask_precision_mean"))
        render_case_rows.append(
            {
                "schema_version": "stream4d_v97_phase9_render_snap_case_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant,
                "support_to_selected_mask_IoU_mean": support_iou,
                "mask_precision_mean": precision,
                "readout_mode_counts": row.get("readout_mode_counts", ""),
                "MV_AP_window": metric.get("MV_AP_window", ""),
                "MV_AP50_window": metric.get("MV_AP50_window", ""),
                "MV_AP_scene": metric.get("MV_AP_scene", ""),
                "MV_AP50_scene": metric.get("MV_AP50_scene", ""),
                "blocker_triggered": support_iou < float(args.support_iou_min) or precision < float(args.mask_precision_min),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    if any(row["blocker_triggered"] for row in render_case_rows):
        best_iou = max(_num(row.get("support_to_selected_mask_IoU_mean")) for row in render_case_rows) if render_case_rows else 0.0
        blocker_rows.append(
            _base_row(
                "RENDER_SUPPORT_ALIGNMENT_BLOCKER",
                phase7_root / "readout_quality_rows.csv",
                "support_to_selected_mask_IoU_mean",
                f"best={best_iou}",
                f">={args.support_iou_min}",
                "Rendered support and CropFormer readout remain poorly aligned; AP is zero under all readout variants.",
                "Inspect support heatmap alignment, snap/carve rules, and upstream geometry before threshold tuning.",
            )
        )

    ranking_rows: list[dict[str, Any]] = []
    ranking_triggered = False
    for row in phase7_metric_rows:
        gap = _num(row.get("ScoreFreeMatch50_window")) - _num(row.get("MV_AP50_window"))
        triggered = gap > float(args.ranking_gap_threshold)
        ranking_triggered = ranking_triggered or triggered
        ranking_rows.append(
            {
                "schema_version": "stream4d_v97_phase9_ranking_case_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": row.get("variant_id", ""),
                "MV_AP50_window": row.get("MV_AP50_window", ""),
                "ScoreFreeMatch50_window": row.get("ScoreFreeMatch50_window", ""),
                "scorefree_minus_ap50_window": gap,
                "ranking_blocker_triggered": triggered,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    if ranking_triggered:
        blocker_rows.append(
            _base_row(
                "RANKING_BLOCKER",
                phase7_root / "variant_metric_rows.csv",
                "ScoreFreeMatch50_window_minus_MV_AP50_window",
                f">{args.ranking_gap_threshold}",
                f"<={args.ranking_gap_threshold}",
                "Score-free matching is much higher than AP50, so ranking/calibration is a primary blocker.",
                "Only in this case tune scores/calibration; otherwise return to geometry/object birth/readout.",
            )
        )

    if not _bool(phase7.get("full_dev_gate_pass")):
        blocker_rows.append(
            _base_row(
                "BLOCK_FULL_DEV_SCOPE",
                phase7_root / "summary.json",
                "full_dev_gate_pass",
                phase7.get("full_dev_gate_pass"),
                "true",
                "Current q512+B4 readout is segment_diagnostic, not full-dev.",
                "Do not freeze; either extend corrected overlap-stitch chain to full-dev scope or report No-Go.",
            )
        )

    control_bias_rows = [
        {
            "schema_version": "stream4d_v97_phase9_control_bias_case_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "status": "full_dev_controls_not_run",
            "reason": "Phase7 MV_AP_window is zero and full_dev_gate_pass=false; Phase8 controls were not reached.",
            "best_control_MV_AP_window": phase0.get("best_control_MV_AP_window", ""),
            "best_control_MV_AP50_window": phase0.get("best_control_MV_AP50_window", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]

    semantic_rows = [
        {
            "schema_version": "stream4d_v97_phase9_semantic_feature_case_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "phase4_decision": phase4.get("decision", ""),
            "semantic_tensor_loaded": phase4.get("semantic_tensor_loaded", ""),
            "semantic_source": phase4.get("semantic_source", ""),
            "full_semantic_gate_pass": phase4.get("full_semantic_gate_pass", ""),
            "radio_feature_available_rate": phase4.get("radio_feature_available_rate", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]

    next_actions = [
        {
            "schema_version": "stream4d_v97_phase9_next_action_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "priority": 1,
            "blocker_type": "D4RT_GEOMETRY_BLOCKER",
            "next_action": "Repair method-side D4RT overlap/self-stitch geometry or switch to a better geometry scaffold; final GT-Sim3 remains diagnostic-only.",
        },
        {
            "schema_version": "stream4d_v97_phase9_next_action_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "priority": 2,
            "blocker_type": "OBJECT_BIRTH_GROUPING_BLOCKER",
            "next_action": "Require coverage-aware object birth; B4 low-coverage object-count pass should not advance to AP as success.",
        },
        {
            "schema_version": "stream4d_v97_phase9_next_action_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "priority": 3,
            "blocker_type": "SEMANTIC_FEATURE_BLOCKER",
            "next_action": "Attach real RADIO/DINO features to micro-primitives and rerun Phase4/5; proxy semantic status is insufficient.",
        },
        {
            "schema_version": "stream4d_v97_phase9_next_action_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "priority": 4,
            "blocker_type": "RENDER_SUPPORT_ALIGNMENT_BLOCKER",
            "next_action": "Inspect support heatmap to CropFormer mask alignment; do not tune AP scores while score-free match is also zero.",
        },
    ]

    primary_blockers = sorted({row["blocker_type"] for row in blocker_rows})
    summary = {
        "schema": "stream4d_v97_phase9_failure_decomposition_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "NO_GO_V97_PHASE9_FAILURE_DECOMPOSITION",
        "target_achieved": False,
        "primary_blockers": primary_blockers,
        "source_artifacts": {
            "phase2_root": _rel(phase2_root),
            "stitch_root": _rel(stitch_root),
            "phase4_root": _rel(phase4_root),
            "phase5_root": _rel(phase5_root),
            "phase6_root": _rel(phase6_root),
            "phase7_root": _rel(phase7_root),
        },
        "phase2_decision": phase2.get("decision"),
        "stitch_decision": stitch.get("decision"),
        "phase4_decision": phase4.get("decision"),
        "phase5_decision": phase5.get("decision"),
        "phase6_decision": phase6.get("decision"),
        "phase7_decision": phase7.get("decision"),
        "best_MV_AP_window": (phase7.get("best_variant", {}) if isinstance(phase7.get("best_variant", {}), dict) else {}).get("MV_AP_window", ""),
        "best_MV_AP_scene": (phase7.get("best_variant", {}) if isinstance(phase7.get("best_variant", {}), dict) else {}).get("MV_AP_scene", ""),
        "scene_metric_name": phase7.get("scene_metric_name", "MV_AP_scene"),
        "scene_comparator_available": phase7.get("scene_comparator_available"),
        "runtime_sec": float(time.time() - started),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }

    _write_json(output_root / "blocker_summary.json", summary)
    _write_csv(output_root / "blocker_metric_rows.csv", blocker_rows)
    _write_csv(output_root / "object_birth_case_rows.csv", object_case_rows)
    _write_csv(output_root / "render_snap_case_rows.csv", render_case_rows)
    _write_csv(output_root / "semantic_feature_case_rows.csv", semantic_rows)
    _write_csv(output_root / "d4rt_geometry_case_rows.csv", d4rt_rows)
    _write_csv(output_root / "control_bias_case_rows.csv", control_bias_rows)
    _write_csv(output_root / "ranking_case_rows.csv", ranking_rows)
    _write_csv(output_root / "next_action_rows.csv", next_actions)
    print(json.dumps({"decision": summary["decision"], "primary_blockers": primary_blockers, "output_root": _rel(output_root)}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-root", default="Stream3D/outputs/audit/v97_phase0_fact_lock")
    parser.add_argument("--phase2-root", default="Stream3D/outputs/audit/v97_phase2_d4rt_micro_tracks_overlap48_48clip_all4_q512_gpu7")
    parser.add_argument("--stitch-root", default="Stream3D/outputs/audit/v97_phase2_d4rt_micro_tracks_overlap48_48clip_all4_q512_stitched")
    parser.add_argument("--phase4-root", default="Stream3D/outputs/audit/v97_phase4_micro_affinity_feature_overlap48_48clip_q512_stitched_500k_gpu7")
    parser.add_argument("--phase5-root", default="Stream3D/outputs/audit/v97_phase5_object_birth_overlap48_48clip_q512_stitched_500k")
    parser.add_argument("--phase6-root", default="Stream3D/outputs/audit/v97_phase6_render_splat_B4_overlap48_48clip_q512_stitched_500k_gpu7")
    parser.add_argument("--phase7-root", default="Stream3D/outputs/audit/v97_phase7_support_iou_readout_overlap48_48clip_q512_B4_stitched_gpu7_sigma2_window_scene")
    parser.add_argument("--output-root", default="Stream3D/outputs/audit/v97_phase9_failure_decomposition")
    parser.add_argument("--geometry-residual-p90-threshold", type=float, default=0.5)
    parser.add_argument("--method-scale-min", type=float, default=0.5)
    parser.add_argument("--object-assignment-coverage-min", type=float, default=0.5)
    parser.add_argument("--keypoint-coverage-min", type=float, default=0.5)
    parser.add_argument("--support-iou-min", type=float, default=0.25)
    parser.add_argument("--mask-precision-min", type=float, default=0.10)
    parser.add_argument("--ranking-gap-threshold", type=float, default=0.10)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
