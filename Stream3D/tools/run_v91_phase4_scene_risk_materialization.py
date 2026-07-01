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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v91_phase4_ap50_control_repair as v91repair  # noqa: E402
from tools import run_v91_phase4_multimask_materialization as multimask  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


OUT = ROOT / "outputs/audit/v91_phase4_scene_risk_materialization"
SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
SCORE_SOURCE_VARIANT = "W8a_risk_balanced_p135_witnesses"
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


def _median(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else 0.0


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "V91_SR1_highrisk_broad_top3_r16_drop5",
            "high_risk_max_masks": 3,
            "high_risk_extra_score_delta": 0.50,
            "high_risk_allow_broad_extra": True,
            "low_risk_max_masks": 2,
            "low_risk_extra_score_delta": 0.35,
            "low_risk_allow_broad_extra": False,
            "radius": 16,
            "support_point_radius": 3,
            "broad_rate_threshold": 0.65,
            "drop_per_selected_threshold": 1.0,
        },
        {
            "variant_id": "V91_SR2_highrisk_broad_top4_r16_drop5",
            "high_risk_max_masks": 4,
            "high_risk_extra_score_delta": 0.65,
            "high_risk_allow_broad_extra": True,
            "low_risk_max_masks": 2,
            "low_risk_extra_score_delta": 0.35,
            "low_risk_allow_broad_extra": False,
            "radius": 16,
            "support_point_radius": 3,
            "broad_rate_threshold": 0.65,
            "drop_per_selected_threshold": 1.0,
        },
        {
            "variant_id": "V91_SR3_highrisk_broad_top5_r16_drop5",
            "high_risk_max_masks": 5,
            "high_risk_extra_score_delta": 0.80,
            "high_risk_allow_broad_extra": True,
            "low_risk_max_masks": 2,
            "low_risk_extra_score_delta": 0.35,
            "low_risk_allow_broad_extra": False,
            "radius": 16,
            "support_point_radius": 3,
            "broad_rate_threshold": 0.65,
            "drop_per_selected_threshold": 1.0,
        },
        {
            "variant_id": "V91_SR4_highrisk_broad_top3_r12_drop5",
            "high_risk_max_masks": 3,
            "high_risk_extra_score_delta": 0.50,
            "high_risk_allow_broad_extra": True,
            "low_risk_max_masks": 2,
            "low_risk_extra_score_delta": 0.35,
            "low_risk_allow_broad_extra": False,
            "radius": 12,
            "support_point_radius": 3,
            "broad_rate_threshold": 0.65,
            "drop_per_selected_threshold": 1.0,
        },
        {
            "variant_id": "V91_SR5_highrisk_broad_top4_r12_drop5",
            "high_risk_max_masks": 4,
            "high_risk_extra_score_delta": 0.65,
            "high_risk_allow_broad_extra": True,
            "low_risk_max_masks": 2,
            "low_risk_extra_score_delta": 0.35,
            "low_risk_allow_broad_extra": False,
            "radius": 12,
            "support_point_radius": 3,
            "broad_rate_threshold": 0.65,
            "drop_per_selected_threshold": 1.0,
        },
    ]


def _scene_profile_rows(
    candidates: list[dict[str, Any]],
    slot_to_obj: dict[tuple[str, str], str],
    slot_to_proto: dict[tuple[str, str], str],
    slot_to_area: dict[tuple[str, str], float],
    frame_to_window_index: dict[tuple[str, int], int],
    frame_to_window_id: dict[tuple[str, int], str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    profile_spec = {
        "variant_id": "V91_SR0_profile_m1_policy",
        "max_masks": 2,
        "extra_score_delta": 0.35,
        "allow_broad_extra": False,
        "radius": 16,
        "support_point_radius": 3,
    }
    source_rows, dropped_rows = multimask._select_multimask_rows(
        candidates,
        slot_to_obj,
        slot_to_proto,
        slot_to_area,
        frame_to_window_index,
        frame_to_window_id,
        profile_spec,
    )
    selected_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dropped_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        selected_by_scene[str(row.get("scene_id", ""))].append(row)
    for row in dropped_rows:
        dropped_by_scene[str(row.get("scene_id", ""))].append(row)

    rows: list[dict[str, Any]] = []
    profile: dict[str, dict[str, float]] = {}
    for scene in sorted(set(selected_by_scene) | set(dropped_by_scene)):
        selected = selected_by_scene.get(scene, [])
        dropped = dropped_by_scene.get(scene, [])
        broad_rate = _mean([1.0 if _bool(row.get("broad_background_risk")) else 0.0 for row in selected])
        drop_per_selected = len(dropped) / max(1, len(selected))
        selected_support = [_num(row.get("support_count")) for row in selected]
        dropped_support = [_num(row.get("support_count")) for row in dropped]
        area_ratio = [_num(row.get("area_ratio")) for row in selected]
        record = {
            "scene_id": scene,
            "profile_source_variant": profile_spec["variant_id"],
            "profile_policy": "M1 max2 delta0.35 no-broad-extra post-WTA",
            "source_selected_rows": len(selected),
            "source_dropped_rows": len(dropped),
            "source_drop_per_selected": drop_per_selected,
            "selected_broad_risk_rate": broad_rate,
            "selected_support_count_mean": _mean(selected_support),
            "selected_support_count_median": _median(selected_support),
            "dropped_support_count_mean": _mean(dropped_support),
            "selected_area_ratio_mean": _mean(area_ratio),
            "selected_area_ratio_max": max(area_ratio) if area_ratio else 0.0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        rows.append(record)
        profile[scene] = {key: _num(value) for key, value in record.items() if key not in {"scene_id", "profile_source_variant", "profile_policy"}}
    return rows, profile


def _is_high_risk_scene(scene_profile: dict[str, float], spec: dict[str, Any]) -> bool:
    return (
        scene_profile.get("selected_broad_risk_rate", 0.0) >= float(spec["broad_rate_threshold"])
        or scene_profile.get("source_drop_per_selected", 0.0) >= float(spec["drop_per_selected_threshold"])
    )


def _select_scene_risk_rows(
    candidates: list[dict[str, Any]],
    slot_to_obj: dict[tuple[str, str], str],
    slot_to_proto: dict[tuple[str, str], str],
    slot_to_area: dict[tuple[str, str], float],
    frame_to_window_index: dict[tuple[str, int], int],
    frame_to_window_id: dict[tuple[str, int], str],
    scene_profile: dict[str, dict[str, float]],
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_variant = f"{spec['variant_id']}_source"
    by_slot_frame: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_slot_frame[(str(row["scene_id"]), str(row["local_slot_id"]), _int(row.get("frame_id"), -1))].append(row)

    policy_rows_by_scene: dict[str, dict[str, Any]] = {}
    pre_rows: list[dict[str, Any]] = []
    for (scene, slot, frame_id), items in sorted(by_slot_frame.items()):
        slot_key = (scene, slot)
        if slot_key not in slot_to_obj:
            continue
        high_risk = _is_high_risk_scene(scene_profile.get(scene, {}), spec)
        max_masks = int(spec["high_risk_max_masks"] if high_risk else spec["low_risk_max_masks"])
        extra_delta = float(spec["high_risk_extra_score_delta"] if high_risk else spec["low_risk_extra_score_delta"])
        allow_broad = bool(spec["high_risk_allow_broad_extra"] if high_risk else spec["low_risk_allow_broad_extra"])
        policy_rows_by_scene[scene] = {
            "variant_id": spec["variant_id"],
            "scene_id": scene,
            "scene_policy_is_high_risk": high_risk,
            "scene_profile_selected_broad_risk_rate": scene_profile.get(scene, {}).get("selected_broad_risk_rate", 0.0),
            "scene_profile_source_drop_per_selected": scene_profile.get(scene, {}).get("source_drop_per_selected", 0.0),
            "broad_rate_threshold": float(spec["broad_rate_threshold"]),
            "drop_per_selected_threshold": float(spec["drop_per_selected_threshold"]),
            "max_masks": max_masks,
            "extra_score_delta": extra_delta,
            "allow_broad_extra": allow_broad,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            score = phase4._score_candidate(
                item,
                SCORE_SOURCE_VARIANT,
                slot_to_proto.get(slot_key, ""),
                slot_to_area.get(slot_key, 0.0),
            )
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored:
            continue
        selected: list[tuple[float, dict[str, Any]]] = [scored[0]]
        top_score = scored[0][0]
        for score, item in scored[1:]:
            if len(selected) >= max_masks:
                break
            if not allow_broad and _bool(item.get("broad_background_risk")):
                continue
            if score >= top_score - extra_delta:
                selected.append((score, item))
        for rank, (score, item) in enumerate(selected, start=1):
            pre_rows.append(
                {
                    **item,
                    "variant_id": source_variant,
                    "mv_object_id": f"{source_variant}:{slot_to_obj[slot_key]}",
                    "window_index": int(frame_to_window_index.get((scene, frame_id), -1)),
                    "window_id": frame_to_window_id.get((scene, frame_id), ""),
                    "selection_score": float(score),
                    "selection_rank": int(rank),
                    "selection_stage": "pre_conflict_wta_scene_risk",
                    "selection_reason": (
                        f"v91_scene_risk_{SCORE_SOURCE_VARIANT}_highRisk{high_risk}"
                        f"_max{max_masks}_delta{extra_delta:.2f}_allowBroad{allow_broad}"
                    ),
                    "scene_policy_is_high_risk": high_risk,
                    "scene_profile_selected_broad_risk_rate": scene_profile.get(scene, {}).get("selected_broad_risk_rate", 0.0),
                    "scene_profile_source_drop_per_selected": scene_profile.get(scene, {}).get("source_drop_per_selected", 0.0),
                    "risk_penalty": 1.0 if _bool(item.get("broad_background_risk")) else 0.0,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    kept: dict[tuple[str, int, int], dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    for row in sorted(pre_rows, key=lambda r: _num(r.get("selection_score")), reverse=True):
        key = (str(row["scene_id"]), _int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))
        old = kept.get(key)
        if old is None:
            kept[key] = {**row, "selection_stage": "post_conflict_wta_scene_risk", "conflict_dropped": False}
        else:
            dropped.append(
                {
                    **row,
                    "selection_stage": "dropped_by_same_frame_mask_wta_scene_risk",
                    "conflict_dropped": True,
                    "kept_mv_object_id": old.get("mv_object_id", ""),
                }
            )
    final_rows = sorted(
        kept.values(),
        key=lambda r: (r["variant_id"], r["scene_id"], r["local_slot_id"], _int(r.get("frame_id")), -_num(r.get("selection_score"))),
    )
    return final_rows, dropped, list(policy_rows_by_scene.values())


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
        value = row.get(key, row.get(f"gate_{key}", False))
        if not _bool(value):
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
    profile_rows, scene_profile = _scene_profile_rows(
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
    eval_rows_all: list[dict[str, Any]] = []
    scored_rows_all: list[dict[str, Any]] = []
    support_quality_all: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []

    for spec in _variant_specs():
        variant_id = str(spec["variant_id"])
        source_variant = f"{variant_id}_source"
        source_rows, dropped_source_rows, policy_rows = _select_scene_risk_rows(
            candidates,
            slot_to_obj,
            slot_to_proto,
            slot_to_area,
            frame_to_window_index,
            frame_to_window_id,
            scene_profile,
            spec,
        )
        generated_rows, selected_rows, eval_rows = phase4._generate_carved_masks(
            source_rows,
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
            group_name="scene_risk_materialization",
        )
        support_quality = v91repair._support_quality_rows(scored_rows, feature_map, variant_id)
        metrics, cases = radius_sweep._evaluate_variant(variant_id, scored_rows)

        source_selection_rows_all.extend(source_rows)
        dropped_source_rows_all.extend(dropped_source_rows)
        scene_policy_rows.extend(policy_rows)
        generated_rows_all.extend(generated_rows)
        selected_rows_all.extend(selected_rows)
        eval_rows_all.extend(eval_rows)
        scored_rows_all.extend(scored_rows)
        support_quality_all.extend(support_quality)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)
        broad_values = [1.0 if _bool(row.get("broad_risk")) else 0.0 for row in support_quality]
        risk_rows.append(
            {
                "variant_id": variant_id,
                "score_source_variant": SCORE_SOURCE_VARIANT,
                "scene_risk_profile_source": "GT-free M1-style source selection rows",
                "high_risk_max_masks": int(spec["high_risk_max_masks"]),
                "high_risk_extra_score_delta": float(spec["high_risk_extra_score_delta"]),
                "high_risk_allow_broad_extra": bool(spec["high_risk_allow_broad_extra"]),
                "low_risk_max_masks": int(spec["low_risk_max_masks"]),
                "low_risk_extra_score_delta": float(spec["low_risk_extra_score_delta"]),
                "low_risk_allow_broad_extra": bool(spec["low_risk_allow_broad_extra"]),
                "radius": int(spec["radius"]),
                "support_point_radius": int(spec["support_point_radius"]),
                "risk_filter_mode": "drop_broad_low_h9_5",
                "score_mode": "broad_scene_orig_ge065",
                "source_selected_rows": len(source_rows),
                "source_conflict_dropped_rows": len(dropped_source_rows),
                "selected_rows": len(scored_rows),
                "pre_filter_eval_rows": len(eval_rows),
                "dropped_rows": len(eval_rows) - int(sum(1 for flag in keep_flags if flag)),
                "risk_penalty_mean": _mean(broad_values),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        config_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": "V91_M1_W8a_top2_r16_drop5_sceneorig",
                "changed_module": "phase4_scene_risk_conditioned_multimask_materialization",
                "changed_parameters": (
                    f"profile=broad_rate>={float(spec['broad_rate_threshold'])} or "
                    f"drop_per_selected>={float(spec['drop_per_selected_threshold'])}; "
                    f"high_risk=max{int(spec['high_risk_max_masks'])}/delta{float(spec['high_risk_extra_score_delta'])}/"
                    f"allowBroad{bool(spec['high_risk_allow_broad_extra'])}; "
                    f"low_risk=max{int(spec['low_risk_max_masks'])}/delta{float(spec['low_risk_extra_score_delta'])}/"
                    f"allowBroad{bool(spec['low_risk_allow_broad_extra'])}; "
                    f"carving_radius={int(spec['radius'])}; support_point_radius={int(spec['support_point_radius'])}; "
                    "risk_filter=drop_broad_low_h9_5; score=broad_scene_orig_ge065"
                ),
                "reason_for_change": (
                    "v91 scene0011 diagnostic showed high GT-free broad-risk/drop pressure and low AP50; "
                    "try scene-risk-conditioned broad multi-mask materialization without using scene id, GT, or holdout"
                ),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "SCENE_IMBALANCE_BLOCKER+EXTENT_BLOCKER+CONTROL_BIAS_BLOCKER",
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
            "parent_variant_id": "V91_M1_W8a_top2_r16_drop5_sceneorig",
            "changed_terms": "scene-risk-conditioned broad multi-mask allowance; existing D4RT risk filter and score",
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
        "decision": "DEV_GATE_PASS_READY_FOR_PHASE8_FREEZE" if passing else "NO_PROGRESS_IN_SCENE_RISK_MATERIALIZATION_FAMILY",
        "reason": (
            "At least one scene-risk materialization variant passed v91 Phase8 progress gate."
            if passing
            else "Five scene-risk materialization variants did not improve MV_AP_window by >=0.002 over current Phase8 best; stop this family."
        ),
        "best_variant_id": best.get("variant_id", ""),
        "best_delta_vs_phase8_best_MV_AP_window": best_delta_mv_ap,
        "best_delta_vs_phase8_best_MV_AP50_window": best_delta_mv_ap50,
        "do_not_run_holdout": not bool(passing),
        "recommended_next": "refresh Phase8 and freeze before holdout" if passing else "return to another Phase3/4 extent mechanism or lock No-Go if all planned families are exhausted",
    }
    summary = {
        "phase": "v91_phase4_scene_risk_materialization",
        "schema": "stream4d_v91_phase4_scene_risk_materialization_v1",
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
        "holdout_policy": "No holdout is used or touched by this dev-only scene-risk materialization repair.",
        "runtime_sec": time.time() - started,
    }
    _write_csv(OUT / "variant_config_rows.csv", config_rows)
    _write_csv(OUT / "scene_profile_rows.csv", profile_rows)
    _write_csv(OUT / "scene_policy_rows.csv", scene_policy_rows)
    _write_csv(OUT / "source_selection_rows.csv", source_selection_rows_all)
    _write_csv(OUT / "dropped_source_rows.csv", dropped_source_rows_all)
    _write_csv(OUT / "generated_mask_rows.csv", generated_rows_all)
    _write_csv(OUT / "selected_masklet_rows.csv", selected_rows_all)
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
    parser = argparse.ArgumentParser(description="Run v91 Phase4 scene-risk-conditioned multi-mask materialization on dev only.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
