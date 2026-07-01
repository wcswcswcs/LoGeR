from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v90_carrier_supported_carving as phase3  # noqa: E402
from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v91_phase4_adaptive_uncertainty_materialization as adaptive  # noqa: E402
from tools import run_v91_phase4_ap50_control_repair as v91repair  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402
from tools import run_v91_phase4_scene_risk_materialization as scene_risk  # noqa: E402


OUT = ROOT / "outputs/audit/v91_phase4_broad_core_precision_repair"
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
    }
    return [
        {
            **base,
            "variant_id": "V91_BC1_highrisk_broad_core_r12_spr3",
            "radius": 12,
            "support_point_radius": 3,
            "broad_core_mode": "intersect",
            "min_core_area": 16,
        },
        {
            **base,
            "variant_id": "V91_BC2_highrisk_broad_core_r16_spr3",
            "radius": 16,
            "support_point_radius": 3,
            "broad_core_mode": "intersect",
            "min_core_area": 16,
        },
        {
            **base,
            "variant_id": "V91_BC3_highrisk_broad_core_r20_spr3",
            "radius": 20,
            "support_point_radius": 3,
            "broad_core_mode": "intersect",
            "min_core_area": 16,
        },
        {
            **base,
            "variant_id": "V91_BC4_highrisk_broad_core_r16_spr5",
            "radius": 16,
            "support_point_radius": 5,
            "broad_core_mode": "intersect",
            "min_core_area": 16,
        },
        {
            **base,
            "variant_id": "V91_BC5_highrisk_broad_core_top3_r16_spr3",
            "high_risk_max_masks": 3,
            "high_risk_extra_score_delta": 0.50,
            "radius": 16,
            "support_point_radius": 3,
            "broad_core_mode": "intersect",
            "min_core_area": 16,
        },
    ]


def _should_core_clip(row: dict[str, Any]) -> bool:
    return adaptive._bool(row.get("scene_policy_is_high_risk")) and adaptive._bool(row.get("broad_background_risk"))


def _generate_core_masks(
    source_rows: list[dict[str, Any]],
    support_points: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    mask_dirs: dict[str, Path],
    variant: str,
    source_variant: str,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    generated_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    core_rows: list[dict[str, Any]] = []

    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in source_rows:
        key = (str(row.get("scene_id", "")), adaptive._int(row.get("frame_id"), -1))
        by_frame.setdefault(key, []).append(row)

    for (scene, frame_id), rows in sorted(by_frame.items()):
        label = phase4._read_label(mask_dirs[scene] / f"{int(frame_id)}.png")
        shape = label.shape
        frame_items: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda r: adaptive._num(r.get("selection_score")), reverse=True):
            slot = str(row.get("local_slot_id", ""))
            mask_id = adaptive._int(row.get("mask_id"), -1)
            points = support_points.get((scene, slot, frame_id, mask_id), [])
            if not points:
                continue
            source_mask = label == int(mask_id)
            if not np.any(source_mask):
                continue

            _heat, support = phase3._paint_support(points, shape, int(spec["support_point_radius"]))
            dilated = phase3._dilate(support, int(spec["radius"]))
            component = phase3._connected_component_around_support(source_mask, dilated)
            core = source_mask & dilated
            use_core = _should_core_clip(row)
            if use_core and int(np.count_nonzero(core)) >= int(spec["min_core_area"]):
                generated = core
                carving_mode = "high_risk_broad_mask_intersect_dilated_carrier_core"
            else:
                generated = component
                carving_mode = "non_broad_or_low_risk_connected_component"
            if not np.any(generated):
                continue

            source_area = int(np.count_nonzero(source_mask))
            support_area = int(np.count_nonzero(dilated))
            generated_area_pre_wta = int(np.count_nonzero(generated))
            frame_items.append(
                {
                    **row,
                    "generated_mask": generated,
                    "carving_mode": carving_mode,
                    "source_mask_area": source_area,
                    "support_area": support_area,
                    "core_area": int(np.count_nonzero(core)),
                    "component_area": int(np.count_nonzero(component)),
                    "pre_wta_generated_mask_area": generated_area_pre_wta,
                    "core_clip_applied": bool(use_core),
                    "core_support_to_source_ratio": support_area / max(1, source_area),
                    "core_generated_to_support_ratio": generated_area_pre_wta / max(1, support_area),
                }
            )

        if not frame_items:
            continue
        label_out = np.zeros(shape, dtype=np.uint16)
        for item in frame_items:
            new_mask_id = int(np.max(label_out)) + 1
            write_mask = item["generated_mask"] & (label_out == 0)
            if np.any(write_mask):
                label_out[write_mask] = new_mask_id
                item["new_mask_id"] = new_mask_id
        out_dir = OUT / "generated_masks" / variant / scene / "mask"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{int(frame_id)}.png"
        if not cv2.imwrite(str(out_path), label_out):
            raise RuntimeError(f"failed to write {out_path}")

        for item in frame_items:
            new_mask_id = adaptive._int(item.get("new_mask_id"), -1)
            if new_mask_id <= 0:
                continue
            final_area = int(np.count_nonzero(label_out == new_mask_id))
            if final_area <= 0:
                continue
            mv_object_id = str(item.get("mv_object_id", "")).replace(f"{source_variant}:", f"{variant}:")
            generated_rows.append(
                {
                    "variant_id": variant,
                    "source_variant": source_variant,
                    "scene_id": scene,
                    "window_id": item.get("window_id", ""),
                    "window_index": item.get("window_index", ""),
                    "mv_object_id": mv_object_id,
                    "local_slot_id": item.get("local_slot_id", ""),
                    "frame_id": int(frame_id),
                    "source_mask_id": adaptive._int(item.get("mask_id"), -1),
                    "new_mask_id": int(new_mask_id),
                    "generated_mask_path": adaptive._rel(out_path),
                    "carving_mode": item["carving_mode"],
                    "carrier_support_count": adaptive._int(item.get("support_count"), 0),
                    "support_area": adaptive._int(item.get("support_area"), 0),
                    "source_mask_area": adaptive._int(item.get("source_mask_area"), 0),
                    "generated_mask_area": int(final_area),
                    "object_score": adaptive._num(item.get("selection_score"), 1.0),
                    "core_clip_applied": bool(item["core_clip_applied"]),
                    "core_area": adaptive._int(item.get("core_area"), 0),
                    "component_area": adaptive._int(item.get("component_area"), 0),
                    "pre_wta_generated_mask_area": adaptive._int(item.get("pre_wta_generated_mask_area"), 0),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                }
            )
            audit_item = {k: v for k, v in item.items() if k != "generated_mask"}
            selection_rows.append(
                {
                    **audit_item,
                    "variant_id": variant,
                    "mv_object_id": mv_object_id,
                    "new_mask_id": int(new_mask_id),
                    "generated_mask_path": adaptive._rel(out_path),
                    "generated_mask_area": int(final_area),
                    "selection_stage": "post_scene_risk_plus_broad_core_precision",
                }
            )
            eval_rows.append(
                {
                    "split": "dev",
                    "scene_id": scene,
                    "source_variant": variant,
                    "variant": variant,
                    "mv_object_id": mv_object_id,
                    "frame_id": int(frame_id),
                    "mask_id": int(new_mask_id),
                    "frame_mask_score": adaptive._num(item.get("selection_score"), 1.0),
                    "object_score": adaptive._num(item.get("selection_score"), 1.0),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                    "materializable": True,
                    "selection_reason": f"phase4_{variant}_{source_variant}_broad_core_precision",
                }
            )
            core_rows.append(
                {
                    "variant_id": variant,
                    "scene_id": scene,
                    "frame_id": int(frame_id),
                    "mask_id": adaptive._int(item.get("mask_id"), -1),
                    "new_mask_id": int(new_mask_id),
                    "mv_object_id": mv_object_id,
                    "scene_policy_is_high_risk": adaptive._bool(item.get("scene_policy_is_high_risk")),
                    "broad_background_risk": adaptive._bool(item.get("broad_background_risk")),
                    "core_clip_applied": bool(item["core_clip_applied"]),
                    "carving_mode": item["carving_mode"],
                    "source_mask_area": adaptive._int(item.get("source_mask_area"), 0),
                    "support_area": adaptive._int(item.get("support_area"), 0),
                    "core_area": adaptive._int(item.get("core_area"), 0),
                    "component_area": adaptive._int(item.get("component_area"), 0),
                    "pre_wta_generated_mask_area": adaptive._int(item.get("pre_wta_generated_mask_area"), 0),
                    "generated_mask_area": int(final_area),
                    "core_generated_to_support_ratio": adaptive._num(item.get("core_generated_to_support_ratio")),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return generated_rows, selection_rows, eval_rows, core_rows


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
    dropped_source_rows_all: list[dict[str, Any]] = []
    generated_rows_all: list[dict[str, Any]] = []
    selected_rows_all: list[dict[str, Any]] = []
    core_clip_rows_all: list[dict[str, Any]] = []
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
        generated_rows, selected_rows, eval_rows, core_rows = _generate_core_masks(
            source_rows,
            support_points,
            mask_dirs,
            variant=variant_id,
            source_variant=source_variant,
            spec=spec,
        )
        feature_map = radius_sweep._feature_map(selected_rows, generated_rows)
        scored_rows, keep_flags = v91repair._variant_rows(
            eval_rows,
            feature_map,
            variant_id=variant_id,
            risk_mode="drop_broad_low_h9_5",
            score_mode="broad_scene_orig_ge065",
            group_name="broad_core_precision_repair",
        )
        support_quality = v91repair._support_quality_rows(scored_rows, feature_map, variant_id)
        metrics, cases = radius_sweep._evaluate_variant(variant_id, scored_rows)

        source_selection_rows_all.extend(source_rows)
        dropped_source_rows_all.extend(dropped_source_rows)
        scene_policy_rows.extend(policy_rows)
        generated_rows_all.extend(generated_rows)
        selected_rows_all.extend(selected_rows)
        core_clip_rows_all.extend(core_rows)
        eval_rows_all.extend(eval_rows)
        scored_rows_all.extend(scored_rows)
        support_quality_all.extend(support_quality)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)

        broad_values = [1.0 if adaptive._bool(row.get("broad_risk")) else 0.0 for row in support_quality]
        clipped = [row for row in core_rows if adaptive._bool(row.get("core_clip_applied"))]
        risk_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
                "radius": int(spec["radius"]),
                "support_point_radius": int(spec["support_point_radius"]),
                "high_risk_max_masks": int(spec["high_risk_max_masks"]),
                "high_risk_extra_score_delta": float(spec["high_risk_extra_score_delta"]),
                "high_risk_allow_broad_extra": bool(spec["high_risk_allow_broad_extra"]),
                "low_risk_max_masks": int(spec["low_risk_max_masks"]),
                "low_risk_extra_score_delta": float(spec["low_risk_extra_score_delta"]),
                "low_risk_allow_broad_extra": bool(spec["low_risk_allow_broad_extra"]),
                "broad_core_mode": spec["broad_core_mode"],
                "source_selected_rows": len(source_rows),
                "source_conflict_dropped_rows": len(dropped_source_rows),
                "selected_rows": len(scored_rows),
                "pre_filter_eval_rows": len(eval_rows),
                "dropped_rows": len(eval_rows) - int(sum(1 for flag in keep_flags if flag)),
                "core_clip_rows": len(clipped),
                "risk_penalty_mean": adaptive._mean(broad_values),
                "core_generated_to_support_ratio_mean": adaptive._mean(
                    [adaptive._num(row.get("core_generated_to_support_ratio")) for row in clipped]
                ),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        config_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
                "changed_module": "phase4_high_risk_broad_core_precision_readout",
                "changed_parameters": (
                    f"profile=broad_rate>={float(spec['broad_rate_threshold'])} or "
                    f"drop_per_selected>={float(spec['drop_per_selected_threshold'])}; "
                    f"high_risk=max{int(spec['high_risk_max_masks'])}/delta{float(spec['high_risk_extra_score_delta'])}/"
                    f"allowBroad{bool(spec['high_risk_allow_broad_extra'])}; "
                    f"low_risk=max{int(spec['low_risk_max_masks'])}/delta{float(spec['low_risk_extra_score_delta'])}/"
                    f"allowBroad{bool(spec['low_risk_allow_broad_extra'])}; "
                    f"radius={int(spec['radius'])}; support_point_radius={int(spec['support_point_radius'])}; "
                    f"broad_core_mode={spec['broad_core_mode']}; risk_filter=drop_broad_low_h9_5; "
                    "score=broad_scene_orig_ge065"
                ),
                "reason_for_change": (
                    "v91 best AD4 remains blocked by AP50 control margin and scene0011 broad-risk/extent weakness; "
                    "for high-risk broad masks, clip generated mask to mask∩dilated D4RT carrier support instead of preserving the whole connected component"
                ),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "EXTENT_BLOCKER+CONTROL_BIAS_BLOCKER+SCENE_IMBALANCE_BLOCKER",
            }
        )

    aggregate_rows = phase7d._aggregate(metric_rows)
    control_rows = v91repair._add_gate_rows(aggregate_rows, baselines)
    best = adaptive._best_row(control_rows)
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
            "changed_terms": "high-risk broad-mask carrier-core precision clipping; existing D4RT risk filter and score",
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
        "decision": "DEV_GATE_PASS_READY_FOR_PHASE8_FREEZE" if passing else "NO_PROGRESS_IN_BROAD_CORE_PRECISION_FAMILY",
        "reason": (
            "At least one broad-core precision variant passed v91 Phase8 progress gate."
            if passing
            else "Five broad-core precision variants did not pass the v91 Phase8 progress gate; stop this family unless a new D4RT residual term is introduced."
        ),
        "best_variant_id": best.get("variant_id", ""),
        "best_delta_vs_phase8_best_MV_AP_window": best_delta_mv_ap,
        "best_delta_vs_phase8_best_MV_AP50_window": best_delta_mv_ap50,
        "do_not_run_holdout": not bool(passing),
        "recommended_next": "refresh Phase8 and freeze before holdout" if passing else "return to remaining Phase3/4 mechanisms; do not run holdout",
    }
    summary = {
        "phase": "v91_phase4_broad_core_precision_repair",
        "schema": "stream4d_v91_phase4_broad_core_precision_repair_v1",
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
            "dropped_source_rows": len(dropped_source_rows_all),
            "generated_mask_rows": len(generated_rows_all),
            "selected_masklet_rows": len(selected_rows_all),
            "core_clip_rows": len(core_clip_rows_all),
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
        "holdout_policy": "No holdout is used or touched by this dev-only broad-core precision repair.",
        "runtime_sec": time.time() - started,
    }

    adaptive._write_csv(OUT / "variant_config_rows.csv", config_rows)
    adaptive._write_csv(OUT / "scene_profile_rows.csv", profile_rows)
    adaptive._write_csv(OUT / "scene_policy_rows.csv", scene_policy_rows)
    adaptive._write_csv(OUT / "source_selection_rows.csv", source_selection_rows_all)
    adaptive._write_csv(OUT / "dropped_source_rows.csv", dropped_source_rows_all)
    adaptive._write_csv(OUT / "generated_mask_rows.csv", generated_rows_all)
    adaptive._write_csv(OUT / "selected_masklet_rows.csv", selected_rows_all)
    adaptive._write_csv(OUT / "core_clip_rows.csv", core_clip_rows_all)
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
        OUT / "dropped_source_rows.csv",
        OUT / "generated_mask_rows.csv",
        OUT / "selected_masklet_rows.csv",
        OUT / "core_clip_rows.csv",
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
    parser = argparse.ArgumentParser(description="Run v91 Phase4 high-risk broad-mask core precision repair on dev only.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
