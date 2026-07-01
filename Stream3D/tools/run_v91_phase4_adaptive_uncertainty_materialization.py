from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
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
from tools import run_v91_phase4_ap50_control_repair as v91repair  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402
from tools import run_v91_phase4_scene_risk_materialization as scene_risk  # noqa: E402


OUT = ROOT / "outputs/audit/v91_phase4_adaptive_uncertainty_materialization"
SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
REFERENCE_PHASE8 = ROOT / "outputs/audit/v91_phase8_dev_selection/summary.json"


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


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
            "variant_id": "V91_AD1_sr2_adapt_sig4_b05_j05_r16",
            "base_radius": 16,
            "base_support_point_radius": 3,
            "sigma0": 4.0,
            "beta": 0.5,
            "lambda_jitter": 0.5,
            "radius_scale": 1.0,
            "support_point_scale": 0.25,
            "max_radius": 24,
            "max_support_point_radius": 6,
        },
        {
            **base,
            "variant_id": "V91_AD2_sr2_adapt_sig8_b05_j075_r16",
            "base_radius": 16,
            "base_support_point_radius": 3,
            "sigma0": 8.0,
            "beta": 0.5,
            "lambda_jitter": 0.75,
            "radius_scale": 1.0,
            "support_point_scale": 0.25,
            "max_radius": 28,
            "max_support_point_radius": 7,
        },
        {
            **base,
            "variant_id": "V91_AD3_sr2_adapt_sig12_b05_j1_r16",
            "base_radius": 16,
            "base_support_point_radius": 3,
            "sigma0": 12.0,
            "beta": 0.5,
            "lambda_jitter": 1.0,
            "radius_scale": 1.0,
            "support_point_scale": 0.30,
            "max_radius": 32,
            "max_support_point_radius": 8,
        },
        {
            **base,
            "variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
            "base_radius": 12,
            "base_support_point_radius": 3,
            "sigma0": 8.0,
            "beta": 0.5,
            "lambda_jitter": 0.75,
            "radius_scale": 1.0,
            "support_point_scale": 0.25,
            "max_radius": 28,
            "max_support_point_radius": 7,
        },
        {
            **base,
            "variant_id": "V91_AD5_sr2_adapt_sig8_b05_j075_r20",
            "base_radius": 20,
            "base_support_point_radius": 3,
            "sigma0": 8.0,
            "beta": 0.5,
            "lambda_jitter": 0.75,
            "radius_scale": 1.0,
            "support_point_scale": 0.25,
            "max_radius": 32,
            "max_support_point_radius": 7,
        },
    ]


def _adaptive_radii(row: dict[str, Any], shape: tuple[int, int], spec: dict[str, Any]) -> dict[str, float | int]:
    h, w = shape
    confidence = max(0.0, min(1.0, _num(row.get("confidence_mean"), 1.0)))
    jitter_px = math.sqrt((_num(row.get("uv_x_std")) * w) ** 2 + (_num(row.get("uv_y_std")) * h) ** 2)
    jitter_norm = min(2.0, jitter_px / 32.0)
    sigma = float(spec["sigma0"]) * math.exp(-float(spec["beta"]) * confidence) * (1.0 + float(spec["lambda_jitter"]) * jitter_norm)
    radius = int(round(float(spec["base_radius"]) + sigma * float(spec["radius_scale"])))
    support_radius = int(round(float(spec["base_support_point_radius"]) + sigma * float(spec["support_point_scale"])))
    radius = max(1, min(int(spec["max_radius"]), radius))
    support_radius = max(1, min(int(spec["max_support_point_radius"]), support_radius))
    return {
        "adaptive_sigma": float(sigma),
        "adaptive_jitter_px": float(jitter_px),
        "adaptive_jitter_norm": float(jitter_norm),
        "adaptive_confidence": float(confidence),
        "adaptive_radius": int(radius),
        "adaptive_support_point_radius": int(support_radius),
    }


def _generate_adaptive_masks(
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
    adaptive_rows: list[dict[str, Any]] = []
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_frame[(row["scene_id"], _int(row.get("frame_id"), -1))].append(row)
    for (scene, frame_id), rows in sorted(by_frame.items()):
        label = phase4._read_label(mask_dirs[scene] / f"{int(frame_id)}.png")
        shape = label.shape
        frame_items: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda r: _num(r.get("selection_score")), reverse=True):
            slot = row["local_slot_id"]
            mask_id = _int(row.get("mask_id"), -1)
            points = support_points.get((scene, slot, frame_id, mask_id), [])
            if not points:
                continue
            source_mask = label == int(mask_id)
            if not np.any(source_mask):
                continue
            radii = _adaptive_radii(row, shape, spec)
            _heat, support = phase3._paint_support(points, shape, int(radii["adaptive_support_point_radius"]))
            dilated = phase3._dilate(support, int(radii["adaptive_radius"]))
            carved = phase3._connected_component_around_support(source_mask, dilated)
            if not np.any(carved):
                continue
            frame_items.append(
                {
                    **row,
                    **radii,
                    "generated_mask": carved,
                    "source_mask_area": int(np.count_nonzero(source_mask)),
                    "support_area": int(np.count_nonzero(dilated)),
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
            new_mask_id = _int(item.get("new_mask_id"), -1)
            if new_mask_id <= 0:
                continue
            final_area = int(np.count_nonzero(label_out == new_mask_id))
            if final_area <= 0:
                continue
            generated_rows.append(
                {
                    "variant_id": variant,
                    "source_variant": source_variant,
                    "scene_id": scene,
                    "window_id": item.get("window_id", ""),
                    "window_index": item.get("window_index", ""),
                    "mv_object_id": item.get("mv_object_id", "").replace(f"{source_variant}:", f"{variant}:"),
                    "local_slot_id": item.get("local_slot_id", ""),
                    "frame_id": int(frame_id),
                    "source_mask_id": _int(item.get("mask_id"), -1),
                    "new_mask_id": int(new_mask_id),
                    "generated_mask_path": _rel(out_path),
                    "carving_mode": "phase4_adaptive_uncertainty_connected_component",
                    "carrier_support_count": _int(item.get("support_count"), 0),
                    "support_area": _int(item.get("support_area"), 0),
                    "source_mask_area": _int(item.get("source_mask_area"), 0),
                    "generated_mask_area": int(final_area),
                    "object_score": _num(item.get("selection_score"), 1.0),
                    "adaptive_sigma": item["adaptive_sigma"],
                    "adaptive_jitter_px": item["adaptive_jitter_px"],
                    "adaptive_radius": item["adaptive_radius"],
                    "adaptive_support_point_radius": item["adaptive_support_point_radius"],
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
                    "mv_object_id": generated_rows[-1]["mv_object_id"],
                    "new_mask_id": int(new_mask_id),
                    "generated_mask_path": _rel(out_path),
                    "generated_mask_area": int(final_area),
                    "selection_stage": "post_scene_risk_plus_adaptive_uncertainty_carving",
                }
            )
            eval_rows.append(
                {
                    "split": "dev",
                    "scene_id": scene,
                    "source_variant": variant,
                    "variant": variant,
                    "mv_object_id": generated_rows[-1]["mv_object_id"],
                    "frame_id": int(frame_id),
                    "mask_id": int(new_mask_id),
                    "frame_mask_score": _num(item.get("selection_score"), 1.0),
                    "object_score": _num(item.get("selection_score"), 1.0),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                    "materializable": True,
                    "selection_reason": f"phase4_{variant}_{source_variant}_adaptive_uncertainty_carving",
                }
            )
            adaptive_rows.append(
                {
                    "variant_id": variant,
                    "scene_id": scene,
                    "frame_id": int(frame_id),
                    "mask_id": _int(item.get("mask_id"), -1),
                    "new_mask_id": int(new_mask_id),
                    "mv_object_id": generated_rows[-1]["mv_object_id"],
                    "confidence_mean": _num(item.get("confidence_mean"), 1.0),
                    "uv_x_std": _num(item.get("uv_x_std")),
                    "uv_y_std": _num(item.get("uv_y_std")),
                    "adaptive_sigma": item["adaptive_sigma"],
                    "adaptive_jitter_px": item["adaptive_jitter_px"],
                    "adaptive_jitter_norm": item["adaptive_jitter_norm"],
                    "adaptive_radius": item["adaptive_radius"],
                    "adaptive_support_point_radius": item["adaptive_support_point_radius"],
                    "source_mask_area": _int(item.get("source_mask_area"), 0),
                    "support_area": _int(item.get("support_area"), 0),
                    "generated_mask_area": int(final_area),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return generated_rows, selection_rows, eval_rows, adaptive_rows


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            _num(row.get("dev_gate_min_margin"), -999.0),
            _num(row.get("mean_MV_AP50_window"), -999.0),
            _num(row.get("mean_MV_AP_window"), -999.0),
        ),
        default={},
    )


def _control_failure(row: dict[str, Any]) -> str:
    if _bool(row.get("v91_phase8_progress_gate_pass")):
        return ""
    failed = []
    for key in [
        "best_real_MV_AP_window_ge_B0_plus_0p010",
        "best_real_MV_AP50_window_ge_B0_plus_0p020",
        "best_real_MV_AP_window_ge_control_plus_0p005",
        "best_real_MV_AP50_window_ge_control_plus_0p010",
        "same_frame_collision_count_eq_0",
        "missing_mask_raster_count_eq_0",
        "uses_gt_for_prediction_false",
        "uses_future_false",
    ]:
        if not _bool(row.get(key, row.get(f"gate_{key}", False))):
            failed.append(key)
    return ";".join(failed)


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
    adaptive_rows_all: list[dict[str, Any]] = []
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
        generated_rows, selected_rows, eval_rows, adaptive_rows = _generate_adaptive_masks(
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
            group_name="adaptive_uncertainty_materialization",
        )
        support_quality = v91repair._support_quality_rows(scored_rows, feature_map, variant_id)
        metrics, cases = radius_sweep._evaluate_variant(variant_id, scored_rows)

        source_selection_rows_all.extend(source_rows)
        dropped_source_rows_all.extend(dropped_source_rows)
        scene_policy_rows.extend(policy_rows)
        generated_rows_all.extend(generated_rows)
        selected_rows_all.extend(selected_rows)
        adaptive_rows_all.extend(adaptive_rows)
        eval_rows_all.extend(eval_rows)
        scored_rows_all.extend(scored_rows)
        support_quality_all.extend(support_quality)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)
        broad_values = [1.0 if _bool(row.get("broad_risk")) else 0.0 for row in support_quality]
        risk_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": "V91_SR2_highrisk_broad_top4_r16_drop5",
                "sigma0": float(spec["sigma0"]),
                "beta": float(spec["beta"]),
                "lambda_jitter": float(spec["lambda_jitter"]),
                "base_radius": int(spec["base_radius"]),
                "base_support_point_radius": int(spec["base_support_point_radius"]),
                "radius_scale": float(spec["radius_scale"]),
                "support_point_scale": float(spec["support_point_scale"]),
                "source_selected_rows": len(source_rows),
                "source_conflict_dropped_rows": len(dropped_source_rows),
                "selected_rows": len(scored_rows),
                "pre_filter_eval_rows": len(eval_rows),
                "dropped_rows": len(eval_rows) - int(sum(1 for flag in keep_flags if flag)),
                "risk_penalty_mean": _mean(broad_values),
                "adaptive_radius_mean": _mean([_num(row.get("adaptive_radius")) for row in adaptive_rows]),
                "adaptive_support_point_radius_mean": _mean([_num(row.get("adaptive_support_point_radius")) for row in adaptive_rows]),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        config_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": "V91_SR2_highrisk_broad_top4_r16_drop5",
                "changed_module": "phase4_adaptive_d4rt_uncertainty_dilation",
                "changed_parameters": (
                    f"selection=SR2 scene-risk max4/high-risk broad; base_radius={int(spec['base_radius'])}; "
                    f"sigma0={float(spec['sigma0'])}; beta={float(spec['beta'])}; lambda_jitter={float(spec['lambda_jitter'])}; "
                    f"radius_scale={float(spec['radius_scale'])}; support_point_scale={float(spec['support_point_scale'])}; "
                    f"max_radius={int(spec['max_radius'])}; max_support_point_radius={int(spec['max_support_point_radius'])}; "
                    "risk_filter=drop_broad_low_h9_5; score=broad_scene_orig_ge065"
                ),
                "reason_for_change": (
                    "v91 plan Phase3/16 calls for adaptive D4RT uncertainty dilation when fixed support/radius sweeps are insufficient; "
                    "use confidence and UV jitter from GT-free carrier support rows to adapt support and dilation radii"
                ),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "EXTENT_BLOCKER+SCENE_IMBALANCE_BLOCKER",
            }
        )

    aggregate_rows = phase7d._aggregate(metric_rows)
    control_rows = v91repair._add_gate_rows(aggregate_rows, baselines)
    best = _best_row(control_rows)
    passing = [row for row in control_rows if _bool(row.get("v91_phase8_progress_gate_pass"))]
    reference_mv_ap = _num(phase8.get("best_real_MV_AP_window"))
    reference_mv_ap50 = _num(phase8.get("best_real_MV_AP50_window"))
    best_delta_mv_ap = _num(best.get("mean_MV_AP_window")) - reference_mv_ap
    best_delta_mv_ap50 = _num(best.get("mean_MV_AP50_window")) - reference_mv_ap50
    variant_gate_rows: list[dict[str, Any]] = []
    variant_failure_rows: list[dict[str, Any]] = []
    for row in control_rows:
        gate_row = {
            "variant_id": row.get("variant_id", ""),
            "parent_variant_id": "V91_SR2_highrisk_broad_top4_r16_drop5",
            "changed_terms": "adaptive D4RT uncertainty dilation from confidence and UV jitter",
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
            "real_minus_best_control_MV_AP_window": row.get("real_minus_best_control_MV_AP_window", ""),
            "same_frame_collision_count": row.get("same_frame_collision_count", ""),
            "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
            "gate_pass": row.get("v91_phase8_progress_gate_pass", ""),
            "failure_type": _control_failure(row),
        }
        variant_gate_rows.append(gate_row)
        if not _bool(row.get("v91_phase8_progress_gate_pass")):
            variant_failure_rows.append(gate_row)

    next_action = {
        "decision": "DEV_GATE_PASS_READY_FOR_PHASE8_FREEZE" if passing else "NO_PROGRESS_IN_ADAPTIVE_UNCERTAINTY_FAMILY",
        "reason": (
            "At least one adaptive uncertainty variant passed v91 Phase8 progress gate."
            if passing
            else "Five adaptive uncertainty variants did not improve MV_AP_window by >=0.002 over current Phase8 best; stop this family."
        ),
        "best_variant_id": best.get("variant_id", ""),
        "best_delta_vs_phase8_best_MV_AP_window": best_delta_mv_ap,
        "best_delta_vs_phase8_best_MV_AP50_window": best_delta_mv_ap50,
        "do_not_run_holdout": not bool(passing),
        "recommended_next": "refresh Phase8 and freeze before holdout" if passing else "return to remaining Phase3/4 mechanisms or lock No-Go if exhausted",
    }
    summary = {
        "phase": "v91_phase4_adaptive_uncertainty_materialization",
        "schema": "stream4d_v91_phase4_adaptive_uncertainty_materialization_v1",
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
            "adaptive_radius_rows": len(adaptive_rows_all),
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
        "holdout_policy": "No holdout is used or touched by this dev-only adaptive uncertainty materialization repair.",
        "runtime_sec": time.time() - started,
    }
    _write_csv(OUT / "variant_config_rows.csv", config_rows)
    _write_csv(OUT / "scene_profile_rows.csv", profile_rows)
    _write_csv(OUT / "scene_policy_rows.csv", scene_policy_rows)
    _write_csv(OUT / "source_selection_rows.csv", source_selection_rows_all)
    _write_csv(OUT / "dropped_source_rows.csv", dropped_source_rows_all)
    _write_csv(OUT / "generated_mask_rows.csv", generated_rows_all)
    _write_csv(OUT / "selected_masklet_rows.csv", selected_rows_all)
    _write_csv(OUT / "adaptive_radius_rows.csv", adaptive_rows_all)
    _write_csv(OUT / "mv_object_frame_mask_rows.csv", scored_rows_all)
    _write_csv(OUT / "support_quality_rows.csv", support_quality_all)
    _write_csv(OUT / "risk_rows.csv", risk_rows)
    _write_csv(OUT / "mv_metric_rows.csv", metric_rows)
    _write_csv(OUT / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(OUT / "control_metric_rows.csv", control_rows)
    _write_csv(OUT / "casebook_rows.csv", case_rows)
    _write_csv(OUT / "variant_gate_rows.csv", variant_gate_rows)
    _write_csv(OUT / "variant_failure_rows.csv", variant_failure_rows)
    _write_json(OUT / "best_variant_summary.json", best)
    _write_json(OUT / "next_action_recommendation.json", next_action)
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "variant_config_rows.csv",
        OUT / "scene_profile_rows.csv",
        OUT / "scene_policy_rows.csv",
        OUT / "source_selection_rows.csv",
        OUT / "dropped_source_rows.csv",
        OUT / "generated_mask_rows.csv",
        OUT / "selected_masklet_rows.csv",
        OUT / "adaptive_radius_rows.csv",
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
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v91 Phase4 adaptive D4RT uncertainty materialization on dev only.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
