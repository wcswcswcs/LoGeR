from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v90_dev_risk_gate_repair as phase7e  # noqa: E402


OUT = ROOT / "outputs/audit/v91_phase4_witness_cover_ap50_control_repair"
PHASE4_ROOT = ROOT / "outputs/audit/v90_phase4_geo_semantic_witness_cover"
PHASE8_ROOT = ROOT / "outputs/audit/v91_phase8_dev_selection"
BASE_VARIANT = "W9a_risk_balanced_p135_plus_carving"
PRIMARY_SEED = "F_W9a_drop_broad_low_h9_5_broad_scene_orig_ge060"


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _rank01(values: list[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0 for _ in values]
    denom = max(1, len(values) - 1)
    for rank, idx in enumerate(order):
        out[idx] = float(rank / denom)
    return out


def _rank_by_scene(rows: list[dict[str, Any]], values: list[float]) -> list[float]:
    out = [0.0 for _ in values]
    by_scene: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_scene.setdefault(str(row.get("scene_id", "")), []).append(idx)
    for indices in by_scene.values():
        ranks = _rank01([values[i] for i in indices])
        for idx, rank in zip(indices, ranks):
            out[idx] = rank
    return out


def _area_bin(area_ratio: float) -> str:
    if area_ratio < 0.002:
        return "tiny"
    if area_ratio < 0.015:
        return "small"
    if area_ratio < 0.08:
        return "medium"
    return "large"


def _feature_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("scene_id", "")), _int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))


def _feature(row: dict[str, Any], feature_map: dict[tuple[str, int, int], dict[str, Any]]) -> dict[str, Any]:
    return feature_map.get(_feature_key(row), {})


def _feature_values(row: dict[str, Any], feature_map: dict[tuple[str, int, int], dict[str, Any]]) -> dict[str, float | str | bool]:
    feat = _feature(row, feature_map)
    support_count = _num(feat.get("support_count"), _num(feat.get("generated_carrier_support_count"), 0.0))
    carrier_count = _num(feat.get("carrier_count_unique"), support_count)
    confidence = _num(feat.get("confidence_mean"), 1.0)
    visibility = _num(feat.get("visibility_mean"), 1.0)
    density = _num(feat.get("observed_density_mean"), 0.0)
    source_area = max(1.0, _num(feat.get("source_mask_area"), _num(feat.get("generated_source_mask_area"), 1.0)))
    generated_area = max(1.0, _num(feat.get("generated_mask_area"), _num(feat.get("generated_generated_mask_area"), 1.0)))
    support_area = max(1.0, _num(feat.get("support_area"), _num(feat.get("generated_support_area"), 1.0)))
    area_ratio = _num(feat.get("area_ratio"), generated_area / 1254528.0)
    shrink = generated_area / source_area
    internal_affinity = math.log1p(carrier_count) * confidence * visibility + 40.0 * density
    support_coverage = support_area / generated_area
    support_density = support_count / math.sqrt(generated_area)
    broad = _bool(feat.get("broad_background_risk"))
    return {
        "window_id": str(feat.get("window_id", "")),
        "semantic_prototype_id": str(feat.get("semantic_prototype_id", "")),
        "area_bin": _area_bin(area_ratio),
        "support_count": float(support_count),
        "carrier_count": float(carrier_count),
        "confidence": float(confidence),
        "visibility": float(visibility),
        "observed_density": float(density),
        "source_area": float(source_area),
        "generated_area": float(generated_area),
        "support_area": float(support_area),
        "area_ratio": float(area_ratio),
        "shrink_penalty": float(abs(math.log(max(1e-6, shrink)))),
        "internal_affinity": float(internal_affinity),
        "support_coverage": float(support_coverage),
        "support_density": float(support_density),
        "broad_risk": bool(broad),
    }


def _residual(values: list[float], groups: list[tuple[str, ...]]) -> list[float]:
    by_group: dict[tuple[str, ...], list[float]] = {}
    for value, group in zip(values, groups):
        by_group.setdefault(group, []).append(value)
    means = {group: _mean(group_values) for group, group_values in by_group.items()}
    return [float(value - means[group]) for value, group in zip(values, groups)]


def _scene_broad_rates(rows: list[dict[str, Any]], feature_map: dict[tuple[str, int, int], dict[str, Any]]) -> dict[str, float]:
    by_scene: dict[str, list[float]] = {}
    for row in rows:
        values = _feature_values(row, feature_map)
        by_scene.setdefault(str(row.get("scene_id", "")), []).append(1.0 if bool(values["broad_risk"]) else 0.0)
    return {scene: _mean(values) for scene, values in by_scene.items()}


def _raw_scores(
    rows: list[dict[str, Any]],
    feature_map: dict[tuple[str, int, int], dict[str, Any]],
    score_mode: str,
) -> list[float]:
    h9 = [phase7e._raw_h9(dict(row), feature_map) for row in rows]
    original = [_num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)) for row in rows]
    features = [_feature_values(row, feature_map) for row in rows]
    groups_scene_sem_area = [
        (str(row.get("scene_id", "")), str(feat["semantic_prototype_id"]), str(feat["area_bin"]))
        for row, feat in zip(rows, features)
    ]
    groups_window_sem_area = [
        (str(row.get("scene_id", "")), str(feat["window_id"]), str(feat["semantic_prototype_id"]), str(feat["area_bin"]))
        for row, feat in zip(rows, features)
    ]
    if score_mode == "h9_density_heavy":
        return h9
    if score_mode == "support_consistency":
        return [0.75 * float(feat["support_coverage"]) + 0.25 * float(feat["support_density"]) for feat in features]
    if score_mode == "hard_negative_residual_scene":
        return _residual(h9, groups_scene_sem_area)
    if score_mode == "hard_negative_residual_window":
        return _residual(h9, groups_window_sem_area)
    if score_mode == "support_consistency_residual_scene":
        base = [0.75 * float(feat["support_coverage"]) + 0.25 * float(feat["support_density"]) for feat in features]
        return _residual(base, groups_scene_sem_area)
    if score_mode == "broad_scene_orig_ge065":
        scene_rates = _scene_broad_rates(rows, feature_map)
        h9_rank = _rank_by_scene(rows, h9)
        original_rank = _rank_by_scene(rows, original)
        return [
            original_rank[idx] if scene_rates.get(str(row.get("scene_id", "")), 0.0) >= 0.65 else h9_rank[idx]
            for idx, row in enumerate(rows)
        ]
    raise ValueError(score_mode)


def _variant_rows(
    base_rows: list[dict[str, str]],
    feature_map: dict[tuple[str, int, int], dict[str, Any]],
    *,
    variant_id: str,
    risk_mode: str,
    score_mode: str,
    group_name: str,
) -> tuple[list[dict[str, Any]], list[bool]]:
    keep = phase7e._keep_flags(base_rows, feature_map, risk_mode)
    kept_rows = [row for row, flag in zip(base_rows, keep) if flag]
    raw = _raw_scores([dict(row) for row in kept_rows], feature_map, score_mode)
    scores = raw if score_mode == "broad_scene_orig_ge065" else _rank_by_scene([dict(row) for row in kept_rows], raw)
    out: list[dict[str, Any]] = []
    for row, raw_score, score in zip(kept_rows, raw, scores):
        obj = str(row.get("mv_object_id", ""))
        if obj.startswith(f"{BASE_VARIANT}:"):
            obj = obj.replace(f"{BASE_VARIANT}:", f"{variant_id}:", 1)
        out.append(
            {
                **row,
                "variant": variant_id,
                "source_variant": variant_id,
                "mv_object_id": obj,
                "frame_mask_score": float(score),
                "object_score": float(score),
                "raw_v91_score": float(raw_score),
                "score_mode_detail": score_mode,
                "risk_filter_mode": risk_mode,
                "base_extent_variant": BASE_VARIANT,
                "selection_reason": f"v91_phase4_{group_name}_{risk_mode}_{score_mode}_from_{BASE_VARIANT}",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out, keep


def _support_quality_rows(
    rows: list[dict[str, Any]],
    feature_map: dict[tuple[str, int, int], dict[str, Any]],
    variant_id: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        values = _feature_values(row, feature_map)
        selected_area = max(1.0, _num(row.get("mask_area"), float(values["source_area"])))
        generated_area = max(1.0, float(values["generated_area"]))
        support_area = max(1.0, float(values["support_area"]))
        out.append(
            {
                "variant_id": variant_id,
                "scene_id": row.get("scene_id", ""),
                "window_id": values["window_id"],
                "mv_object_id": row.get("mv_object_id", ""),
                "frame_id": row.get("frame_id", ""),
                "mask_id": row.get("mask_id", ""),
                "support_carrier_count": values["support_count"],
                "support_heatmap_area": support_area,
                "selected_mask_area": selected_area,
                "generated_mask_area": generated_area,
                "support_to_mask_ratio": support_area / selected_area,
                "mask_to_support_ratio": generated_area / support_area,
                "broad_risk": values["broad_risk"],
                "hard_negative_density": 1.0 - float(values["observed_density"]),
                "same_frame_collision_flag": False,
                "missing_raster_flag": False,
                "semantic_prototype_id": values["semantic_prototype_id"],
                "area_ratio": values["area_ratio"],
                "support_coverage": values["support_coverage"],
                "support_density": values["support_density"],
            }
        )
    return out


def _variants_for_group(group: str) -> list[dict[str, Any]]:
    risk_strength = [
        ("V91_R1_W9a_drop_broad_low_h9_7_h9", "drop_broad_low_h9_7", "h9_density_heavy"),
        ("V91_R2_W9a_drop_broad_low_h9_10_h9", "drop_broad_low_h9_10", "h9_density_heavy"),
        ("V91_R3_W9a_drop_broad_low_h9_15_h9", "drop_broad_low_h9_15", "h9_density_heavy"),
        ("V91_R4_W9a_drop_broad_low_h9_20_h9", "drop_broad_low_h9_20", "h9_density_heavy"),
        ("V91_R5_W9a_drop_broad_low_h9_25_h9", "drop_broad_low_h9_25", "h9_density_heavy"),
    ]
    d4rt_residual = [
        ("V91_D1_W9a_drop5_hardneg_residual_scene", "drop_broad_low_h9_5", "hard_negative_residual_scene"),
        ("V91_D2_W9a_drop5_hardneg_residual_window", "drop_broad_low_h9_5", "hard_negative_residual_window"),
        ("V91_D3_W9a_drop5_support_consistency", "drop_broad_low_h9_5", "support_consistency"),
        ("V91_D4_W9a_drop5_support_consistency_residual", "drop_broad_low_h9_5", "support_consistency_residual_scene"),
        ("V91_D5_W9a_drop5_scene_broad_orig_ge065", "drop_broad_low_h9_5", "broad_scene_orig_ge065"),
    ]
    selected = {"risk_strength": risk_strength, "d4rt_residual": d4rt_residual}
    if group == "all":
        rows = risk_strength + d4rt_residual
    else:
        rows = selected[group]
    return [
        {
            "variant_id": variant_id,
            "risk_mode": risk_mode,
            "score_mode": score_mode,
            "group": "risk_strength" if variant_id.startswith("V91_R") else "d4rt_residual",
        }
        for variant_id, risk_mode, score_mode in rows
    ]


def _phase8_baselines() -> dict[str, dict[str, Any]]:
    rows = _read_csv(PHASE8_ROOT / "all_variant_metric_rows.csv")
    return {row.get("variant_id", ""): row for row in rows}


def _gate_fields(row: dict[str, Any], baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    b0 = baselines.get("B0_local_only", {})
    control = baselines.get("P3_C0_area_semantic_hybrid_score", {})
    mv_ap = _num(row.get("mean_MV_AP_window"))
    mv_ap50 = _num(row.get("mean_MV_AP50_window"))
    return {
        "B0_MV_AP_window": _num(b0.get("mean_MV_AP_window")),
        "B0_MV_AP50_window": _num(b0.get("mean_MV_AP50_window")),
        "best_control_variant": control.get("variant_id", "P3_C0_area_semantic_hybrid_score"),
        "best_control_MV_AP_window": _num(control.get("mean_MV_AP_window")),
        "best_control_MV_AP50_window": _num(control.get("mean_MV_AP50_window")),
        "real_minus_B0_MV_AP_window": mv_ap - _num(b0.get("mean_MV_AP_window")),
        "real_minus_B0_MV_AP50_window": mv_ap50 - _num(b0.get("mean_MV_AP50_window")),
        "real_minus_best_control_MV_AP_window": mv_ap - _num(control.get("mean_MV_AP_window")),
        "real_minus_best_control_MV_AP50_window": mv_ap50 - _num(control.get("mean_MV_AP50_window")),
        "required_MV_AP_window_for_control_gate": _num(control.get("mean_MV_AP_window")) + 0.005,
        "required_MV_AP50_window_for_control_gate": _num(control.get("mean_MV_AP50_window")) + 0.010,
        "gap_to_required_MV_AP_window": mv_ap - (_num(control.get("mean_MV_AP_window")) + 0.005),
        "gap_to_required_MV_AP50_window": mv_ap50 - (_num(control.get("mean_MV_AP50_window")) + 0.010),
    }


def _add_gate_rows(aggregate_rows: list[dict[str, Any]], baselines: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in aggregate_rows:
        gates = _gate_fields(row, baselines)
        mv_ap = _num(row.get("mean_MV_AP_window"))
        mv_ap50 = _num(row.get("mean_MV_AP50_window"))
        b0 = baselines.get("B0_local_only", {})
        control = baselines.get("P3_C0_area_semantic_hybrid_score", {})
        dev_gate = {
            "best_real_MV_AP_window_ge_B0_plus_0p010": mv_ap >= _num(b0.get("mean_MV_AP_window")) + 0.010,
            "best_real_MV_AP50_window_ge_B0_plus_0p020": mv_ap50 >= _num(b0.get("mean_MV_AP50_window")) + 0.020,
            "best_real_MV_AP_window_ge_control_plus_0p005": mv_ap >= _num(control.get("mean_MV_AP_window")) + 0.005,
            "best_real_MV_AP50_window_ge_control_plus_0p010": mv_ap50 >= _num(control.get("mean_MV_AP50_window")) + 0.010,
            "same_frame_collision_count_eq_0": _int(row.get("same_frame_collision_count")) == 0,
            "missing_mask_raster_count_eq_0": _int(row.get("missing_mask_raster_count")) == 0,
            "uses_gt_for_prediction_false": not _bool(row.get("uses_gt_for_prediction")),
            "uses_future_false": not _bool(row.get("uses_future")),
        }
        out.append(
            {
                **row,
                **gates,
                **{f"gate_{key}": value for key, value in dev_gate.items()},
                "v91_phase8_progress_gate_pass": all(dev_gate.values()),
                "dev_gate_min_margin": min(
                    gates["real_minus_B0_MV_AP_window"] - 0.010,
                    gates["real_minus_B0_MV_AP50_window"] - 0.020,
                    gates["real_minus_best_control_MV_AP_window"] - 0.005,
                    gates["real_minus_best_control_MV_AP50_window"] - 0.010,
                ),
            }
        )
    return out


def _summarize(
    *,
    output_root: Path,
    group: str,
    started: float,
    config_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    best = max(
        control_rows,
        key=lambda row: (
            _num(row.get("dev_gate_min_margin"), -999.0),
            _num(row.get("mean_MV_AP50_window"), -999.0),
            _num(row.get("mean_MV_AP_window"), -999.0),
        ),
        default={},
    )
    passing = [row for row in control_rows if _bool(row.get("v91_phase8_progress_gate_pass"))]
    summary = {
        "phase": "v91_phase4_witness_cover_ap50_control_repair",
        "schema": "stream4d_v91_phase4_ap50_control_repair_v1",
        "group": group,
        "base_extent_variant": BASE_VARIANT,
        "primary_seed_reference": PRIMARY_SEED,
        "repair_scope": "dev_only_phase4_CONTROL_BIAS_EXTENT_repair_no_holdout",
        "variant_count": len(config_rows),
        "best_variant_id": best.get("variant_id", ""),
        "best_variant_gate": best,
        "any_v91_phase8_progress_gate_pass": bool(passing),
        "decision": "DEV_GATE_PASS_READY_FOR_PHASE8_FREEZE" if passing else "DEV_GATE_FAIL_CONTINUE_PHASE3_4_REPAIR",
        "next_action": "freeze candidate in Phase8 before holdout" if passing else "continue CONTROL_BIAS/EXTENT repair; do not run Phase9 holdout",
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "selected_masklet_rows": len(selected_rows),
            "mv_metric_rows": len(metric_rows),
            "mv_metric_aggregate_rows": len(aggregate_rows),
            "control_metric_rows": len(control_rows),
            "support_quality_rows": len(support_rows),
            "risk_rows": len(risk_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "holdout_policy": "No holdout is used or touched by this dev-only repair.",
        "runtime_sec": time.time() - started,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def _run_variants(group: str, output_root: Path) -> dict[str, Any]:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    variants = _variants_for_group(group)
    base_rows = [row for row in _read_csv(PHASE4_ROOT / "mv_object_frame_mask_rows.csv") if row.get("variant") == BASE_VARIANT]
    feature_map = phase7d._load_feature_rows(BASE_VARIANT)
    baselines = _phase8_baselines()
    b0_risk = _num(_read_json(PHASE4_ROOT / "summary.json").get("B0_risk_penalty_mean_proxy"), 1.0)

    config_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    witness_rows: list[dict[str, Any]] = []
    witness_cover_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []

    for variant in variants:
        variant_id = str(variant["variant_id"])
        risk_mode = str(variant["risk_mode"])
        score_mode = str(variant["score_mode"])
        rows, keep_flags = _variant_rows(
            base_rows,
            feature_map,
            variant_id=variant_id,
            risk_mode=risk_mode,
            score_mode=score_mode,
            group_name=str(variant["group"]),
        )
        selected_rows.extend(rows)
        support_rows_for_variant = _support_quality_rows(rows, feature_map, variant_id)
        support_rows.extend(support_rows_for_variant)
        metrics, cases = phase7d._evaluate_variant(BASE_VARIANT, variant_id, rows)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)

        broad_values = [1.0 if bool(row.get("broad_risk")) else 0.0 for row in support_rows_for_variant]
        risk_mean = _mean(broad_values)
        dropped_count = len(base_rows) - int(sum(1 for flag in keep_flags if flag))
        risk_rows.append(
            {
                "variant_id": variant_id,
                "risk_filter_mode": risk_mode,
                "score_mode": score_mode,
                "selected_rows": len(rows),
                "dropped_rows": dropped_count,
                "risk_penalty_mean": risk_mean,
                "B0_risk_penalty_mean_proxy": b0_risk,
                "risk_safe_vs_B0_proxy": risk_mean <= b0_risk + 1e-12,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        config_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": BASE_VARIANT,
                "variant_group": variant["group"],
                "changed_parameters": f"risk_filter_mode={risk_mode}; score_mode={score_mode}",
                "changed_module": "phase4_witness_cover_d4rt_risk_guard_ap50_repair",
                "reason_for_change": "v91 Phase8 failed AP50 control margin; try GT-free D4RT risk/residual guard before holdout",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "CONTROL_BIAS_BLOCKER+EXTENT_BLOCKER",
            }
        )
        for scene in sorted({str(row.get("scene_id", "")) for row in rows}):
            scene_support = [row for row in support_rows_for_variant if str(row.get("scene_id", "")) == scene]
            witness_rows.append(
                {
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "witness_type": "D4RT_carrier_risk_residual",
                    "witness_row_count": len(scene_support),
                    "support_carrier_count_mean": _mean([_num(row.get("support_carrier_count")) for row in scene_support]),
                    "support_coverage_mean": _mean([_num(row.get("support_coverage")) for row in scene_support]),
                    "broad_risk_rate": _mean([1.0 if _bool(row.get("broad_risk")) else 0.0 for row in scene_support]),
                }
            )
            witness_cover_rows.append(
                {
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "selected_masklet_count": len(scene_support),
                    "support_to_mask_ratio_mean": _mean([_num(row.get("support_to_mask_ratio")) for row in scene_support]),
                    "mask_to_support_ratio_mean": _mean([_num(row.get("mask_to_support_ratio")) for row in scene_support]),
                    "hard_negative_density_mean": _mean([_num(row.get("hard_negative_density")) for row in scene_support]),
                }
            )

    aggregate_rows = phase7d._aggregate(metric_rows)
    control_rows = _add_gate_rows(aggregate_rows, baselines)
    _write_csv(output_root / "variant_config_rows.csv", config_rows)
    _write_csv(output_root / "witness_rows.csv", witness_rows)
    _write_csv(output_root / "witness_cover_rows.csv", witness_cover_rows)
    _write_csv(output_root / "selected_masklet_rows.csv", selected_rows)
    _write_csv(output_root / "support_quality_rows.csv", support_rows)
    _write_csv(output_root / "risk_rows.csv", risk_rows)
    _write_csv(output_root / "mv_metric_rows.csv", metric_rows)
    _write_csv(output_root / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(output_root / "control_metric_rows.csv", control_rows)
    _write_csv(output_root / "casebook_rows.csv", case_rows)
    summary = _summarize(
        output_root=output_root,
        group=group,
        started=started,
        config_rows=config_rows,
        metric_rows=metric_rows,
        aggregate_rows=aggregate_rows,
        control_rows=control_rows,
        selected_rows=selected_rows,
        support_rows=support_rows,
        risk_rows=risk_rows,
    )
    outputs = [
        output_root / "variant_config_rows.csv",
        output_root / "witness_rows.csv",
        output_root / "witness_cover_rows.csv",
        output_root / "selected_masklet_rows.csv",
        output_root / "support_quality_rows.csv",
        output_root / "risk_rows.csv",
        output_root / "mv_metric_rows.csv",
        output_root / "mv_metric_aggregate_rows.csv",
        output_root / "control_metric_rows.csv",
        output_root / "casebook_rows.csv",
        output_root / "summary.json",
    ]
    _write_json(output_root / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def _merge_groups(output_root: Path, group_roots: list[Path]) -> dict[str, Any]:
    started = time.time()
    file_names = [
        "variant_config_rows.csv",
        "witness_rows.csv",
        "witness_cover_rows.csv",
        "selected_masklet_rows.csv",
        "support_quality_rows.csv",
        "risk_rows.csv",
        "mv_metric_rows.csv",
        "mv_metric_aggregate_rows.csv",
        "control_metric_rows.csv",
        "casebook_rows.csv",
    ]
    merged: dict[str, list[dict[str, Any]]] = {name: [] for name in file_names}
    for root in group_roots:
        for name in file_names:
            merged[name].extend(_read_csv(root / name))
    baselines = _phase8_baselines()
    aggregate_rows = phase7d._aggregate(merged["mv_metric_rows.csv"])
    control_rows = _add_gate_rows(aggregate_rows, baselines)
    merged["mv_metric_aggregate_rows.csv"] = aggregate_rows
    merged["control_metric_rows.csv"] = control_rows
    output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in merged.items():
        _write_csv(output_root / name, rows)
    summary = _summarize(
        output_root=output_root,
        group="merged_parallel_groups",
        started=started,
        config_rows=merged["variant_config_rows.csv"],
        metric_rows=merged["mv_metric_rows.csv"],
        aggregate_rows=aggregate_rows,
        control_rows=control_rows,
        selected_rows=merged["selected_masklet_rows.csv"],
        support_rows=merged["support_quality_rows.csv"],
        risk_rows=merged["risk_rows.csv"],
    )
    outputs = [output_root / name for name in file_names] + [output_root / "summary.json"]
    _write_json(output_root / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v91 Phase4 AP50/control-margin repair on dev artifacts only.")
    parser.add_argument("--group", choices=["risk_strength", "d4rt_residual", "all"], default="all")
    parser.add_argument("--merge", action="store_true", help="Merge risk_strength and d4rt_residual subdir outputs into the final root.")
    args = parser.parse_args()
    if args.merge:
        _merge_groups(OUT, [OUT / "risk_strength", OUT / "d4rt_residual"])
    else:
        output_root = OUT if args.group == "all" else OUT / args.group
        _run_variants(args.group, output_root)


if __name__ == "__main__":
    main()
