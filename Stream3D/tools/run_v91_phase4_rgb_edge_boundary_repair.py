from __future__ import annotations

import argparse
import csv
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

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from tools import run_v90_carrier_supported_carving as phase3  # noqa: E402
from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v91_phase4_adaptive_uncertainty_materialization as adaptive  # noqa: E402
from tools import run_v91_phase4_ap50_control_repair as v91repair  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402
from tools import run_v91_phase4_scene_risk_materialization as scene_risk  # noqa: E402


OUT = ROOT / "outputs/audit/v91_phase4_rgb_edge_boundary_repair"
SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
REFERENCE_PHASE8 = ROOT / "outputs/audit/v91_phase8_dev_selection/summary.json"


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
            writer.writerow(adaptive._jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adaptive._jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        "min_edge_area": 32,
        "fallback_to_component": True,
    }
    return [
        {**base, "variant_id": "V91_EB1_highrisk_broad_rgbedge_p85_r24", "trigger": "highrisk_broad", "edge_percentile": 85, "edge_barrier_dilate": 1, "radius": 24, "post_expand": 0},
        {**base, "variant_id": "V91_EB2_highrisk_broad_rgbedge_p90_r24", "trigger": "highrisk_broad", "edge_percentile": 90, "edge_barrier_dilate": 1, "radius": 24, "post_expand": 0},
        {**base, "variant_id": "V91_EB3_highrisk_broad_rgbedge_p95_r24", "trigger": "highrisk_broad", "edge_percentile": 95, "edge_barrier_dilate": 0, "radius": 24, "post_expand": 0},
        {**base, "variant_id": "V91_EB4_highrisk_broad_rgbedge_p90_r32_expand6", "trigger": "highrisk_broad", "edge_percentile": 90, "edge_barrier_dilate": 1, "radius": 32, "post_expand": 6},
        {**base, "variant_id": "V91_EB5_allhighrisk_rgbedge_p90_r24", "trigger": "all_highrisk", "edge_percentile": 90, "edge_barrier_dilate": 1, "radius": 24, "post_expand": 0},
    ]


class _RgbGradientCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, tuple[int, int]], np.ndarray] = {}
        self._streams: dict[str, ScanNetStream] = {}

    def get(self, scene: str, frame_id: int, shape: tuple[int, int]) -> np.ndarray:
        key = (scene, int(frame_id), tuple(shape))
        if key in self._cache:
            return self._cache[key]
        if scene not in self._streams:
            self._streams[scene] = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        rgb = self._streams[scene].load_rgb(int(frame_id))
        if rgb.shape[:2] != tuple(shape):
            rgb = cv2.resize(rgb, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        max_grad = float(np.max(grad)) if grad.size else 0.0
        if max_grad > 0:
            grad = grad / max_grad
        self._cache[key] = grad.astype(np.float32, copy=False)
        return self._cache[key]


def _should_edge(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    high_risk = adaptive._bool(row.get("scene_policy_is_high_risk"))
    broad = adaptive._bool(row.get("broad_background_risk"))
    trigger = str(spec.get("trigger", "highrisk_broad"))
    if trigger == "all_highrisk":
        return high_risk
    if trigger == "highrisk_broad":
        return high_risk and broad
    return False


def _edge_component(
    *,
    source_mask: np.ndarray,
    support: np.ndarray,
    gradient: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if not np.any(source_mask) or not np.any(support & source_mask):
        return np.zeros_like(source_mask, dtype=bool), {
            "edge_threshold": 0.0,
            "edge_barrier_area": 0,
            "edge_component_area": 0,
            "edge_used": False,
            "edge_fallback_reason": "empty_source_or_support",
        }
    source_grad = gradient[source_mask]
    threshold = float(np.percentile(source_grad, float(spec["edge_percentile"]))) if source_grad.size else 1.0
    barrier = source_mask & (gradient >= threshold)
    if int(spec.get("edge_barrier_dilate", 0)) > 0 and np.any(barrier):
        k = int(spec["edge_barrier_dilate"]) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        barrier = cv2.dilate(barrier.astype(np.uint8), kernel, iterations=1).astype(bool) & source_mask
    seed = phase3._dilate(support & source_mask, int(spec["radius"])) & source_mask
    allowed = source_mask & (~barrier | seed)
    labels_count, labels = cv2.connectedComponents(allowed.astype(np.uint8), connectivity=8)
    keep = phase3._connected_component_from_labels(labels_count, labels, seed, allowed)
    if int(spec.get("post_expand", 0)) > 0 and np.any(keep):
        keep = phase3._dilate(keep, int(spec["post_expand"])) & source_mask
    edge_area = int(np.count_nonzero(keep))
    if edge_area < int(spec["min_edge_area"]):
        return keep, {
            "edge_threshold": threshold,
            "edge_barrier_area": int(np.count_nonzero(barrier)),
            "edge_component_area": edge_area,
            "edge_used": False,
            "edge_fallback_reason": "edge_component_too_small",
        }
    return keep, {
        "edge_threshold": threshold,
        "edge_barrier_area": int(np.count_nonzero(barrier)),
        "edge_component_area": edge_area,
        "edge_used": True,
        "edge_fallback_reason": "",
    }


def _generate_edge_masks(
    source_rows: list[dict[str, Any]],
    support_points: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    mask_dirs: dict[str, Path],
    *,
    variant: str,
    source_variant: str,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    generated_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    gradient_cache = _RgbGradientCache()
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in source_rows:
        key = (str(row.get("scene_id", "")), adaptive._int(row.get("frame_id"), -1))
        by_frame.setdefault(key, []).append(row)

    for (scene, frame_id), rows in sorted(by_frame.items()):
        label = phase4._read_label(mask_dirs[scene] / f"{int(frame_id)}.png")
        shape = label.shape
        gradient = gradient_cache.get(scene, frame_id, shape)
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
            fallback_component = phase3._connected_component_around_support(source_mask, dilated)
            use_edge = _should_edge(row, spec)
            edge_info = {
                "edge_threshold": "",
                "edge_barrier_area": "",
                "edge_component_area": "",
                "edge_used": False,
                "edge_fallback_reason": "trigger_not_matched",
            }
            generated = fallback_component
            mode = f"fallback_witness_connected_component_r{int(spec['radius']):02d}"
            if use_edge:
                edge_component, edge_info = _edge_component(
                    source_mask=source_mask,
                    support=support,
                    gradient=gradient,
                    spec=spec,
                )
                if adaptive._bool(edge_info.get("edge_used")) or not bool(spec.get("fallback_to_component", True)):
                    generated = edge_component
                    mode = (
                        f"rgb_edge_boundary_p{int(spec['edge_percentile'])}_"
                        f"r{int(spec['radius'])}_bd{int(spec['edge_barrier_dilate'])}_expand{int(spec['post_expand'])}"
                    )
                else:
                    mode = f"edge_failed_{edge_info.get('edge_fallback_reason')}_fallback_component"
            if not np.any(generated):
                continue
            frame_items.append(
                {
                    **row,
                    "generated_mask": generated,
                    "carving_mode": mode,
                    "source_mask_area": int(np.count_nonzero(source_mask)),
                    "support_area": int(np.count_nonzero(dilated)),
                    "pre_wta_generated_mask_area": int(np.count_nonzero(generated)),
                    "rgb_edge_triggered": bool(use_edge),
                    **edge_info,
                }
            )

        if not frame_items:
            continue
        label_out = np.zeros(shape, dtype=np.uint16)
        for item in sorted(frame_items, key=lambda r: adaptive._num(r.get("selection_score")), reverse=True):
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
            row_base = {
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
                "pre_wta_generated_mask_area": adaptive._int(item.get("pre_wta_generated_mask_area"), 0),
                "object_score": adaptive._num(item.get("selection_score"), 1.0),
                "rgb_edge_triggered": bool(item.get("rgb_edge_triggered", False)),
                "rgb_edge_used": adaptive._bool(item.get("edge_used")),
                "edge_threshold": item.get("edge_threshold", ""),
                "edge_barrier_area": item.get("edge_barrier_area", ""),
                "edge_component_area": item.get("edge_component_area", ""),
                "edge_fallback_reason": item.get("edge_fallback_reason", ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_rgb_image_boundary": True,
                "uses_rgbd_pose_mesh": False,
            }
            generated_rows.append(row_base)
            audit_item = {k: v for k, v in item.items() if k != "generated_mask"}
            selection_rows.append({**audit_item, **row_base, "selection_stage": "post_rgb_edge_boundary_readout"})
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
                    "uses_rgb_image_boundary": True,
                    "uses_rgbd_pose_mesh": False,
                    "materializable": True,
                    "selection_reason": f"phase4_{variant}_{source_variant}_rgb_edge_boundary_readout",
                }
            )
            edge_rows.append(
                {
                    **row_base,
                    "scene_policy_is_high_risk": adaptive._bool(item.get("scene_policy_is_high_risk")),
                    "broad_background_risk": adaptive._bool(item.get("broad_background_risk")),
                    "trigger": spec["trigger"],
                    "edge_percentile": int(spec["edge_percentile"]),
                    "edge_barrier_dilate": int(spec["edge_barrier_dilate"]),
                    "post_expand": int(spec["post_expand"]),
                }
            )
    return generated_rows, selection_rows, eval_rows, edge_rows


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
    edge_rows_all: list[dict[str, Any]] = []
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
        generated_rows, selected_rows, eval_rows, edge_rows = _generate_edge_masks(
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
            group_name="rgb_edge_boundary_repair",
        )
        support_quality = v91repair._support_quality_rows(scored_rows, feature_map, variant_id)
        metrics, cases = radius_sweep._evaluate_variant(variant_id, scored_rows)

        scene_policy_rows.extend(policy_rows)
        source_selection_rows_all.extend(source_rows)
        dropped_source_rows_all.extend(dropped_source_rows)
        generated_rows_all.extend(generated_rows)
        selected_rows_all.extend(selected_rows)
        edge_rows_all.extend(edge_rows)
        eval_rows_all.extend(eval_rows)
        scored_rows_all.extend(scored_rows)
        support_quality_all.extend(support_quality)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)

        broad_values = [1.0 if adaptive._bool(row.get("broad_risk")) else 0.0 for row in support_quality]
        edge_used = [row for row in edge_rows if adaptive._bool(row.get("rgb_edge_used"))]
        risk_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
                "trigger": spec["trigger"],
                "edge_percentile": int(spec["edge_percentile"]),
                "edge_barrier_dilate": int(spec["edge_barrier_dilate"]),
                "radius": int(spec["radius"]),
                "post_expand": int(spec["post_expand"]),
                "source_selected_rows": len(source_rows),
                "source_conflict_dropped_rows": len(dropped_source_rows),
                "selected_rows": len(scored_rows),
                "pre_filter_eval_rows": len(eval_rows),
                "dropped_rows": len(eval_rows) - int(sum(1 for flag in keep_flags if flag)),
                "rgb_edge_trigger_rows": int(sum(1 for row in edge_rows if adaptive._bool(row.get("rgb_edge_triggered")))),
                "rgb_edge_used_rows": len(edge_used),
                "risk_penalty_mean": adaptive._mean(broad_values),
                "generated_to_source_area_ratio_mean": adaptive._mean(
                    [
                        adaptive._num(row.get("generated_mask_area")) / max(1.0, adaptive._num(row.get("source_mask_area"), 1.0))
                        for row in edge_rows
                    ]
                ),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_rgb_image_boundary": True,
            }
        )
        config_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
                "changed_module": "phase4_rgb_edge_boundary_readout",
                "changed_parameters": (
                    f"trigger={spec['trigger']}; edge_percentile={int(spec['edge_percentile'])}; "
                    f"edge_barrier_dilate={int(spec['edge_barrier_dilate'])}; radius={int(spec['radius'])}; "
                    f"post_expand={int(spec['post_expand'])}; risk_filter=drop_broad_low_h9_5; "
                    "score=broad_scene_orig_ge065"
                ),
                "reason_for_change": (
                    "v91 source-mask oracle shows GT boundary clipping has high upper bound; "
                    "use legal RGB image gradients as source-mask internal boundary barriers for D4RT carrier-supported region growing."
                ),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "uses_rgb_image_boundary": True,
                "uses_rgbd_pose_mesh": False,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "SOURCE_BOUNDARY_READOUT_BLOCKER",
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
        cfg = next((item for item in config_rows if item.get("variant_id") == row.get("variant_id")), {})
        gate_row = {
            "variant_id": row.get("variant_id", ""),
            "parent_variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
            "changed_terms": "RGB edge-boundary source-mask readout; existing D4RT risk filter and score",
            "changed_parameters": cfg.get("changed_parameters", ""),
            "reason_for_change": cfg.get("reason_for_change", ""),
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
        "decision": "DEV_GATE_PASS_READY_FOR_PHASE8_FREEZE" if passing else "NO_PROGRESS_IN_RGB_EDGE_BOUNDARY_FAMILY",
        "reason": (
            "At least one RGB edge-boundary variant passed v91 Phase8 progress gate."
            if passing
            else "Five RGB edge-boundary variants did not pass the v91 Phase8 progress gate; stop this family unless a new non-GT boundary signal is introduced."
        ),
        "best_variant_id": best.get("variant_id", ""),
        "best_delta_vs_phase8_best_MV_AP_window": best_delta_mv_ap,
        "best_delta_vs_phase8_best_MV_AP50_window": best_delta_mv_ap50,
        "do_not_run_holdout": not bool(passing),
        "recommended_next": "refresh Phase8 and freeze before holdout" if passing else "do not run holdout; source-boundary readout remains blocked",
    }
    summary = {
        "phase": "v91_phase4_rgb_edge_boundary_repair",
        "schema": "stream4d_v91_phase4_rgb_edge_boundary_repair_v1",
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
            "generated_rows": len(generated_rows_all),
            "selected_rows": len(selected_rows_all),
            "edge_rows": len(edge_rows_all),
            "eval_rows": len(eval_rows_all),
            "scored_rows": len(scored_rows_all),
            "support_quality_rows": len(support_quality_all),
            "metric_rows": len(metric_rows),
            "casebook_rows": len(case_rows),
            "control_metric_rows": len(control_rows),
            "variant_gate_rows": len(variant_gate_rows),
            "variant_failure_rows": len(variant_failure_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_rgb_image_boundary": True,
        "uses_rgbd_pose_mesh": False,
        "metric_source": "v65 local-window evaluator via radius_sweep._evaluate_variant",
        "runtime_sec": time.time() - started,
    }

    _write_csv(OUT / "variant_config_rows.csv", config_rows)
    _write_csv(OUT / "scene_profile_rows.csv", profile_rows)
    _write_csv(OUT / "scene_policy_rows.csv", scene_policy_rows)
    _write_csv(OUT / "source_selection_rows.csv", source_selection_rows_all)
    _write_csv(OUT / "dropped_source_rows.csv", dropped_source_rows_all)
    _write_csv(OUT / "generated_rows.csv", generated_rows_all)
    _write_csv(OUT / "selected_rows.csv", selected_rows_all)
    _write_csv(OUT / "edge_rows.csv", edge_rows_all)
    _write_csv(OUT / "mv_object_frame_mask_rows.csv", eval_rows_all)
    _write_csv(OUT / "scored_frame_mask_rows.csv", scored_rows_all)
    _write_csv(OUT / "support_quality_rows.csv", support_quality_all)
    _write_csv(OUT / "mv_metric_rows.csv", metric_rows)
    _write_csv(OUT / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(OUT / "control_metric_rows.csv", control_rows)
    _write_csv(OUT / "variant_gate_rows.csv", variant_gate_rows)
    _write_csv(OUT / "variant_failure_rows.csv", variant_failure_rows)
    _write_csv(OUT / "risk_rows.csv", risk_rows)
    _write_csv(OUT / "casebook_rows.csv", case_rows)
    _write_json(OUT / "best_variant_summary.json", best)
    _write_json(OUT / "next_action_recommendation.json", next_action)
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "variant_config_rows.csv",
        OUT / "scene_profile_rows.csv",
        OUT / "scene_policy_rows.csv",
        OUT / "source_selection_rows.csv",
        OUT / "dropped_source_rows.csv",
        OUT / "generated_rows.csv",
        OUT / "selected_rows.csv",
        OUT / "edge_rows.csv",
        OUT / "mv_object_frame_mask_rows.csv",
        OUT / "scored_frame_mask_rows.csv",
        OUT / "support_quality_rows.csv",
        OUT / "mv_metric_rows.csv",
        OUT / "mv_metric_aggregate_rows.csv",
        OUT / "control_metric_rows.csv",
        OUT / "variant_gate_rows.csv",
        OUT / "variant_failure_rows.csv",
        OUT / "risk_rows.csv",
        OUT / "casebook_rows.csv",
        OUT / "best_variant_summary.json",
        OUT / "next_action_recommendation.json",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {adaptive._rel(path): adaptive._sha256(path) for path in outputs if path.exists()})
    print(json.dumps(adaptive._jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v91 Phase4 RGB edge-boundary source-mask repair.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
