from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v91_phase4_adaptive_uncertainty_materialization as adaptive  # noqa: E402
from tools import run_v91_phase4_ap50_control_repair as v91repair  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402
from tools import run_v91_phase4_scene_risk_materialization as scene_risk  # noqa: E402


OUT = ROOT / "outputs/audit/v91_phase4_semantic_witness_split_repair"
SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
REFERENCE_PHASE8 = ROOT / "outputs/audit/v91_phase8_dev_selection/summary.json"


def _variant_specs() -> list[dict[str, Any]]:
    base = {
        "high_risk_max_masks": 4,
        "high_risk_extra_score_delta": 0.65,
        "high_risk_allow_broad_extra": True,
        "low_risk_max_masks": 2,
        "low_risk_extra_score_delta": 0.35,
        "low_risk_allow_broad_extra": False,
        "broad_rate_threshold": 0.65,
        "drop_per_selected_threshold": 1.0,
        "radius": 16,
        "support_point_radius": 3,
    }
    return [
        {**base, "variant_id": "V91_WS1_highrisk_proto1_split_r16", "split_mode": "highrisk_proto1"},
        {**base, "variant_id": "V91_WS2_highrisk_proto2_split_r16", "split_mode": "highrisk_proto2"},
        {**base, "variant_id": "V91_WS3_highrisk_broad_proto2_split_r16", "split_mode": "highrisk_broad_proto2"},
        {**base, "variant_id": "V91_WS4_highrisk_proto2_uvquad_split_r16", "split_mode": "highrisk_proto2_uvquad"},
        {**base, "variant_id": "V91_WS5_allscene_proto1_split_r16", "split_mode": "allscene_proto1"},
    ]


def _safe_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text).strip())[:96] or "unknown"


def _proto_token(row: dict[str, Any], depth: int) -> str:
    parts = [part for part in str(row.get("semantic_prototype_id", "")).split("|") if part]
    if not parts:
        return "proto_unknown"
    if parts[0] == "dino":
        parts = parts[1:]
    return _safe_token("_".join(parts[: max(1, depth)]))


def _uv_quad(row: dict[str, Any]) -> str:
    x = adaptive._num(row.get("uv_x_mean"), 0.5)
    y = adaptive._num(row.get("uv_y_mean"), 0.5)
    return f"q{int(x >= 0.5)}{int(y >= 0.5)}"


def _split_key(row: dict[str, Any], spec: dict[str, Any]) -> str:
    mode = str(spec.get("split_mode", "none"))
    high_risk = adaptive._bool(row.get("scene_policy_is_high_risk"))
    broad = adaptive._bool(row.get("broad_background_risk"))
    if mode == "highrisk_proto1" and high_risk:
        return f"p1_{_proto_token(row, 1)}"
    if mode == "highrisk_proto2" and high_risk:
        return f"p2_{_proto_token(row, 2)}"
    if mode == "highrisk_broad_proto2" and high_risk and broad:
        return f"bp2_{_proto_token(row, 2)}"
    if mode == "highrisk_proto2_uvquad" and high_risk:
        return f"p2q_{_proto_token(row, 2)}_{_uv_quad(row)}"
    if mode == "allscene_proto1":
        return f"ap1_{_proto_token(row, 1)}"
    return "base"


def _apply_semantic_split(source_rows: list[dict[str, Any]], source_variant: str, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in source_rows:
        split_key = _split_key(row, spec)
        original_object = str(row.get("mv_object_id", ""))
        if split_key == "base":
            new_object = original_object
        else:
            new_object = f"{original_object}:witness_{split_key}"
        new_row = {
            **row,
            "mv_object_id": new_object,
            "semantic_witness_split_key": split_key,
            "semantic_witness_split_applied": split_key != "base",
            "semantic_witness_split_mode": spec["split_mode"],
        }
        out.append(new_row)
        audit_rows.append(
            {
                "variant_id": spec["variant_id"],
                "source_variant": source_variant,
                "scene_id": row.get("scene_id", ""),
                "frame_id": row.get("frame_id", ""),
                "mask_id": row.get("mask_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "original_mv_object_id": original_object,
                "split_mv_object_id": new_object,
                "semantic_prototype_id": row.get("semantic_prototype_id", ""),
                "split_key": split_key,
                "split_applied": split_key != "base",
                "scene_policy_is_high_risk": row.get("scene_policy_is_high_risk", ""),
                "broad_background_risk": row.get("broad_background_risk", ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out, audit_rows


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            adaptive._num(row.get("dev_gate_min_margin"), -999.0),
            adaptive._num(row.get("mean_MV_AP50_window"), -999.0),
            adaptive._num(row.get("mean_MV_AP_window"), -999.0),
        ),
        default={},
    )


def run(_args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    phase4.OUT = OUT
    radius_sweep.OUT = OUT
    mask_dirs = phase4._mask_dir_by_scene()
    frame_to_window_index, frame_to_window_id = phase4._window_maps()
    _source_rows, slot_to_obj, slot_to_proto, slot_to_area = phase4._load_source_rows()
    semantic_features = phase4._load_semantic_features()
    candidates, support_points = phase4._load_support_candidates(SUPPORT_ROWS, set(slot_to_obj), semantic_features, mask_dirs)
    slot_to_proto, slot_to_area = phase4._fill_slot_priors_from_candidates(candidates, slot_to_proto, slot_to_area)
    baselines = v91repair._phase8_baselines()
    phase8 = json.loads(REFERENCE_PHASE8.read_text(encoding="utf-8")) if REFERENCE_PHASE8.exists() else {}
    profile_rows, scene_profile = scene_risk._scene_profile_rows(
        candidates,
        slot_to_obj,
        slot_to_proto,
        slot_to_area,
        frame_to_window_index,
        frame_to_window_id,
    )

    config_rows: list[dict[str, Any]] = []
    scene_policy_rows: list[dict[str, Any]] = []
    source_selection_rows_all: list[dict[str, Any]] = []
    split_assignment_rows_all: list[dict[str, Any]] = []
    dropped_source_rows_all: list[dict[str, Any]] = []
    generated_rows_all: list[dict[str, Any]] = []
    selected_rows_all: list[dict[str, Any]] = []
    eval_rows_all: list[dict[str, Any]] = []
    scored_rows_all: list[dict[str, Any]] = []
    support_quality_all: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []

    for spec in _variant_specs():
        variant_id = str(spec["variant_id"])
        source_variant = f"{variant_id}_source"
        source_rows, dropped_source_rows, policy_rows = scene_risk._select_scene_risk_rows(
            candidates,
            slot_to_obj,
            slot_to_proto,
            slot_to_area,
            frame_to_window_index,
            frame_to_window_id,
            scene_profile,
            spec,
        )
        split_source_rows, split_assignment_rows = _apply_semantic_split(source_rows, source_variant, spec)
        generated_rows, selected_rows, eval_rows = phase4._generate_carved_masks(
            split_source_rows,
            support_points,
            mask_dirs,
            radius=int(spec["radius"]),
            support_point_radius=int(spec["support_point_radius"]),
            variant=variant_id,
            source_variant=source_variant,
        )
        feature_map = radius_sweep._feature_map(selected_rows, generated_rows)
        scored_rows, keep_flags = v91repair._variant_rows(
            eval_rows,
            feature_map,
            variant_id=variant_id,
            risk_mode="drop_broad_low_h9_5",
            score_mode="broad_scene_orig_ge065",
            group_name="semantic_witness_split_repair",
        )
        support_quality = v91repair._support_quality_rows(scored_rows, feature_map, variant_id)
        metrics, cases = radius_sweep._evaluate_variant(variant_id, scored_rows)

        scene_policy_rows.extend(policy_rows)
        source_selection_rows_all.extend(source_rows)
        split_assignment_rows_all.extend(split_assignment_rows)
        dropped_source_rows_all.extend(dropped_source_rows)
        generated_rows_all.extend(generated_rows)
        selected_rows_all.extend(selected_rows)
        eval_rows_all.extend(eval_rows)
        scored_rows_all.extend(scored_rows)
        support_quality_all.extend(support_quality)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)

        broad_values = [1.0 if adaptive._bool(row.get("broad_risk")) else 0.0 for row in support_quality]
        split_count = sum(1 for row in split_assignment_rows if adaptive._bool(row.get("split_applied")))
        risk_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
                "split_mode": spec["split_mode"],
                "radius": int(spec["radius"]),
                "support_point_radius": int(spec["support_point_radius"]),
                "source_selected_rows": len(source_rows),
                "source_conflict_dropped_rows": len(dropped_source_rows),
                "split_assignment_rows": len(split_assignment_rows),
                "split_applied_rows": split_count,
                "selected_rows": len(scored_rows),
                "pre_filter_eval_rows": len(eval_rows),
                "dropped_rows": len(eval_rows) - int(sum(1 for flag in keep_flags if flag)),
                "risk_penalty_mean": adaptive._mean(broad_values),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        config_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
                "changed_module": "phase4_geo_semantic_witness_object_tube_split",
                "changed_parameters": (
                    f"split_mode={spec['split_mode']}; radius={int(spec['radius'])}; "
                    f"support_point_radius={int(spec['support_point_radius'])}; "
                    f"high_risk=max{int(spec['high_risk_max_masks'])}/delta{float(spec['high_risk_extra_score_delta'])}/"
                    f"allowBroad{bool(spec['high_risk_allow_broad_extra'])}; "
                    "risk_filter=drop_broad_low_h9_5; score=broad_scene_orig_ge065"
                ),
                "reason_for_change": (
                    "v91 Phase4 priority includes geo-semantic witness cover with object-scale witness sampling; "
                    "split high-risk object tubes by GT-free semantic prototype/UV witness tokens to test whether AD4 is over-broad at object identity level"
                ),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "EXTENT_BLOCKER+CONTROL_BIAS_BLOCKER+SCENE_IMBALANCE_BLOCKER",
            }
        )

    aggregate_rows = phase7d._aggregate(metric_rows)
    control_rows = v91repair._add_gate_rows(aggregate_rows, baselines)
    best = _best_row(control_rows)
    passing = [row for row in control_rows if adaptive._bool(row.get("v91_phase8_progress_gate_pass"))]
    reference_mv_ap = adaptive._num(phase8.get("best_real_MV_AP_window"))
    reference_mv_ap50 = adaptive._num(phase8.get("best_real_MV_AP50_window"))
    best_delta_mv_ap = adaptive._num(best.get("mean_MV_AP_window")) - reference_mv_ap
    best_delta_mv_ap50 = adaptive._num(best.get("mean_MV_AP50_window")) - reference_mv_ap50

    variant_gate_rows: list[dict[str, Any]] = []
    variant_failure_rows: list[dict[str, Any]] = []
    for row in control_rows:
        gate_row = {
            "variant_id": row.get("variant_id", ""),
            "parent_variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
            "changed_terms": "GT-free semantic witness object-tube split",
            "changed_parameters": next((cfg.get("changed_parameters", "") for cfg in config_rows if cfg.get("variant_id") == row.get("variant_id")), ""),
            "reason_for_change": next((cfg.get("reason_for_change", "") for cfg in config_rows if cfg.get("variant_id") == row.get("variant_id")), ""),
            "uses_gt_for_prediction": row.get("uses_gt_for_prediction", "False"),
            "uses_future": row.get("uses_future", "False"),
            "metric_source": "v65 local-window evaluator via radius_sweep._evaluate_variant",
            "MV_AP_window": row.get("mean_MV_AP_window", ""),
            "MV_AP50_window": row.get("mean_MV_AP50_window", ""),
            "MV_AP25_window": row.get("mean_MV_AP25_window", ""),
            "score_free_Match50_window": row.get("mean_score_free_Match50_window", ""),
            "best_control_MV_AP_window": row.get("best_control_MV_AP_window", ""),
            "best_control_MV_AP50_window": row.get("best_control_MV_AP50_window", ""),
            "real_minus_best_control_MV_AP_window": row.get("real_minus_best_control_MV_AP_window", ""),
            "real_minus_best_control_MV_AP50_window": row.get("real_minus_best_control_MV_AP50_window", ""),
            "same_frame_collision_count": row.get("same_frame_collision_count", ""),
            "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
            "gate_pass": row.get("v91_phase8_progress_gate_pass", ""),
            "failure_type": adaptive._control_failure(row),
        }
        variant_gate_rows.append(gate_row)
        if not adaptive._bool(row.get("v91_phase8_progress_gate_pass")):
            variant_failure_rows.append(gate_row)

    next_action = {
        "decision": "DEV_GATE_PASS_READY_FOR_PHASE8_FREEZE" if passing else "NO_PROGRESS_IN_SEMANTIC_WITNESS_SPLIT_FAMILY",
        "reason": (
            "At least one semantic witness split variant passed v91 Phase8 progress gate."
            if passing
            else "Five semantic witness split variants did not pass the v91 Phase8 progress gate; stop this family unless a new source universe is introduced."
        ),
        "best_variant_id": best.get("variant_id", ""),
        "best_delta_vs_phase8_best_MV_AP_window": best_delta_mv_ap,
        "best_delta_vs_phase8_best_MV_AP50_window": best_delta_mv_ap50,
        "do_not_run_holdout": not bool(passing),
        "recommended_next": "refresh Phase8 and freeze before holdout" if passing else "new source/mask universe diagnostic or stop v91 local repair",
    }
    summary = {
        "phase": "v91_phase4_semantic_witness_split_repair",
        "schema": "stream4d_v91_phase4_semantic_witness_split_repair_v1",
        "variant_count": len(config_rows),
        "best_variant_id": best.get("variant_id", ""),
        "best_variant_gate": best,
        "reference_phase8_best_variant": phase8.get("best_real_variant", ""),
        "reference_phase8_best_MV_AP_window": reference_mv_ap,
        "reference_phase8_best_MV_AP50_window": reference_mv_ap50,
        "best_delta_vs_phase8_best_MV_AP_window": best_delta_mv_ap,
        "best_delta_vs_phase8_best_MV_AP50_window": best_delta_mv_ap50,
        "family_stop_rule_applies": (not passing) and best_delta_mv_ap < 0.002,
        "any_v91_phase8_progress_gate_pass": bool(passing),
        "decision": next_action["decision"],
        "next_action": next_action["recommended_next"],
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "scene_profile_rows": len(profile_rows),
            "scene_policy_rows": len(scene_policy_rows),
            "source_selection_rows": len(source_selection_rows_all),
            "split_assignment_rows": len(split_assignment_rows_all),
            "dropped_source_rows": len(dropped_source_rows_all),
            "generated_mask_rows": len(generated_rows_all),
            "selected_masklet_rows": len(selected_rows_all),
            "pre_filter_eval_rows": len(eval_rows_all),
            "scored_frame_mask_rows": len(scored_rows_all),
            "support_quality_rows": len(support_quality_all),
            "mv_metric_rows": len(metric_rows),
            "control_metric_rows": len(control_rows),
            "casebook_rows": len(case_rows),
            "variant_gate_rows": len(variant_gate_rows),
            "variant_failure_rows": len(variant_failure_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "holdout_policy": "No holdout is used or touched by this dev-only semantic witness split repair.",
        "runtime_sec": time.time() - started,
    }

    adaptive._write_csv(OUT / "variant_config_rows.csv", config_rows)
    adaptive._write_csv(OUT / "scene_profile_rows.csv", profile_rows)
    adaptive._write_csv(OUT / "scene_policy_rows.csv", scene_policy_rows)
    adaptive._write_csv(OUT / "source_selection_rows.csv", source_selection_rows_all)
    adaptive._write_csv(OUT / "split_assignment_rows.csv", split_assignment_rows_all)
    adaptive._write_csv(OUT / "dropped_source_rows.csv", dropped_source_rows_all)
    adaptive._write_csv(OUT / "generated_mask_rows.csv", generated_rows_all)
    adaptive._write_csv(OUT / "selected_masklet_rows.csv", selected_rows_all)
    adaptive._write_csv(OUT / "mv_object_frame_mask_rows.csv", scored_rows_all)
    adaptive._write_csv(OUT / "support_quality_rows.csv", support_quality_all)
    adaptive._write_csv(OUT / "risk_rows.csv", risk_rows)
    adaptive._write_csv(OUT / "mv_metric_rows.csv", metric_rows)
    adaptive._write_csv(OUT / "mv_metric_aggregate_rows.csv", aggregate_rows)
    adaptive._write_csv(OUT / "control_metric_rows.csv", control_rows)
    adaptive._write_csv(OUT / "casebook_rows.csv", case_rows)
    adaptive._write_csv(OUT / "variant_gate_rows.csv", variant_gate_rows)
    adaptive._write_csv(OUT / "variant_failure_rows.csv", variant_failure_rows)
    adaptive._write_json(OUT / "best_variant_summary.json", best)
    adaptive._write_json(OUT / "next_action_recommendation.json", next_action)
    adaptive._write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "variant_config_rows.csv",
        OUT / "scene_profile_rows.csv",
        OUT / "scene_policy_rows.csv",
        OUT / "source_selection_rows.csv",
        OUT / "split_assignment_rows.csv",
        OUT / "dropped_source_rows.csv",
        OUT / "generated_mask_rows.csv",
        OUT / "selected_masklet_rows.csv",
        OUT / "mv_object_frame_mask_rows.csv",
        OUT / "support_quality_rows.csv",
        OUT / "risk_rows.csv",
        OUT / "mv_metric_rows.csv",
        OUT / "mv_metric_aggregate_rows.csv",
        OUT / "control_metric_rows.csv",
        OUT / "casebook_rows.csv",
        OUT / "variant_gate_rows.csv",
        OUT / "variant_failure_rows.csv",
        OUT / "best_variant_summary.json",
        OUT / "next_action_recommendation.json",
        OUT / "summary.json",
    ]
    adaptive._write_json(OUT / "SHA256SUMS.json", {adaptive._rel(path): adaptive._sha256(path) for path in outputs if path.exists()})
    print(json.dumps(adaptive._jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v91 Phase4 semantic-witness object-tube split repair on dev only.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
