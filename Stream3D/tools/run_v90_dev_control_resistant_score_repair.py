from __future__ import annotations

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

from tools import run_v89_recalc_point_projected_mv_ap as recalc  # noqa: E402
from tools import run_v90_mv_ap_window_resurrection as phase1  # noqa: E402


OUT = ROOT / "outputs/audit/v90_phase7c_dev_control_resistant_score_repair"
PHASE4_ROOT = ROOT / "outputs/audit/v90_phase4_geo_semantic_witness_cover"
PHASE7 = ROOT / "outputs/audit/v90_phase7_dev_score_calibration_repair"
PHASE7B = ROOT / "outputs/audit/v90_phase7b_dev_score_control_audit"
BASE_VARIANT = "W9b_risk_balanced_p165_plus_carving"
REAL_Q0 = "Q0_W9b_original_score"
CONTROL_BEST = "P3_C0_area_semantic_hybrid_score"
LOCAL_EXPORT_ROOT = ROOT / "outputs/audit/v89_recalc_point_projected_mv_ap"
SCENES = ["scene0011_00", "scene0050_00"]


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


def _rank01(values: list[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0 for _ in values]
    denom = max(1, len(values) - 1)
    for rank, idx in enumerate(order):
        out[idx] = float(rank / denom)
    return out


def _area_bin(area_ratio: float) -> str:
    if area_ratio < 0.002:
        return "tiny"
    if area_ratio < 0.015:
        return "small"
    if area_ratio < 0.08:
        return "medium"
    return "large"


def _load_feature_rows() -> dict[tuple[str, int, int], dict[str, Any]]:
    selection = _read_csv(PHASE4_ROOT / "witness_cover_selection_rows.csv")
    generated = _read_csv(PHASE4_ROOT / "generated_mask_rows.csv")
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in selection:
        if row.get("variant_id") != BASE_VARIANT:
            continue
        key = (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("new_mask_id"), -1))
        if key[0] and key[1] >= 0 and key[2] > 0:
            by_key[key] = dict(row)
    for row in generated:
        if row.get("variant_id") != BASE_VARIANT:
            continue
        key = (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("new_mask_id"), -1))
        if key in by_key:
            by_key[key].update({f"generated_{k}": v for k, v in row.items()})
        elif key[0] and key[1] >= 0 and key[2] > 0:
            by_key[key] = {f"generated_{k}": v for k, v in row.items()}
    return by_key


def _load_baselines() -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    phase7_rows = _read_csv(PHASE7 / "variant_metric_aggregate_rows.csv")
    phase7b_summary_path = PHASE7B / "best_variant_summary.json"
    phase7b_summary = json.loads(phase7b_summary_path.read_text(encoding="utf-8")) if phase7b_summary_path.exists() else {}
    q0 = next((row for row in phase7_rows if row.get("variant_id") == REAL_Q0), {})
    control = dict(phase7b_summary.get("best_control_metrics", {}))
    return q0, control, phase7b_summary


def _raw_features(rows: list[dict[str, Any]], feature_map: dict[tuple[str, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    feats: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))
        feat = feature_map.get(key, {})
        support_count = _num(feat.get("support_count"), _num(feat.get("generated_carrier_support_count"), 0.0))
        carrier_count = _num(feat.get("carrier_count_unique"), support_count)
        confidence = _num(feat.get("confidence_mean"), 1.0)
        visibility = _num(feat.get("visibility_mean"), 1.0)
        density = _num(feat.get("observed_density_mean"), 0.0)
        source_area = max(1.0, _num(feat.get("source_mask_area"), _num(feat.get("generated_source_mask_area"), 1.0)))
        generated_area = max(1.0, _num(feat.get("generated_mask_area"), _num(feat.get("generated_generated_mask_area"), 1.0)))
        support_area = max(1.0, _num(feat.get("support_area"), _num(feat.get("generated_support_area"), 1.0)))
        margin = _num(feat.get("semantic_prototype_margin"), 0.0)
        entropy = _num(feat.get("semantic_entropy"), 1.0)
        broad = 1.0 if _bool(feat.get("broad_background_risk")) else 0.0
        area_ratio = _num(feat.get("area_ratio"), generated_area / 1254528.0)
        shrink = generated_area / source_area
        internal_affinity = math.log1p(carrier_count) * confidence * visibility + 40.0 * density
        support_coverage = support_area / generated_area
        support_density = support_count / math.sqrt(generated_area)
        feats.append(
            {
                "semantic": str(feat.get("semantic_prototype_id", "")),
                "area_bin": _area_bin(area_ratio),
                "window_id": str(feat.get("window_id", "")),
                "selection_score": _num(feat.get("selection_score"), _num(row.get("object_score"), 1.0)),
                "internal_affinity": internal_affinity,
                "support_coverage": support_coverage,
                "support_density": support_density,
                "margin": margin,
                "entropy": entropy,
                "broad": broad,
                "area_ratio": area_ratio,
                "shrink_penalty": abs(math.log(max(1e-6, shrink))),
            }
        )
    return feats


def _residual(values: list[float], groups: list[tuple[str, ...]]) -> list[float]:
    by_group: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        by_group[group].append(value)
    means = {group: _mean(vals) for group, vals in by_group.items()}
    return [float(value - means[group]) for value, group in zip(values, groups)]


def _score_values(rows: list[dict[str, Any]], feature_map: dict[tuple[str, int, int], dict[str, Any]], mode: str) -> list[float]:
    feats = _raw_features(rows, feature_map)
    if mode == "original":
        return [_num(row.get("object_score"), 1.0) for row in rows]
    groups_sem_area = [(row.get("scene_id", ""), f["semantic"], f["area_bin"]) for row, f in zip(rows, feats)]
    groups_sem_area_window = [(row.get("scene_id", ""), f["window_id"], f["semantic"], f["area_bin"]) for row, f in zip(rows, feats)]
    internal = [f["internal_affinity"] for f in feats]
    coverage = [f["support_coverage"] for f in feats]
    density = [f["support_density"] for f in feats]
    if mode == "d4rt_internal_only":
        raw = internal
    elif mode == "within_semantic_internal_residual":
        raw = _residual(internal, groups_sem_area)
    elif mode == "within_window_internal_residual":
        raw = _residual(internal, groups_sem_area_window)
    elif mode == "hard_negative_d4rt":
        raw = [
            f["internal_affinity"]
            + 0.55 * f["support_coverage"]
            + 0.15 * f["support_density"]
            - 0.90 * f["broad"]
            - 0.30 * f["area_ratio"]
            - 0.20 * f["shrink_penalty"]
            for f in feats
        ]
    elif mode == "hard_negative_support_heavy":
        raw = [
            f["internal_affinity"]
            + 1.10 * f["support_coverage"]
            + 0.30 * f["support_density"]
            - 0.90 * f["broad"]
            - 0.30 * f["area_ratio"]
            - 0.20 * f["shrink_penalty"]
            for f in feats
        ]
    elif mode == "hard_negative_area_strong":
        raw = [
            f["internal_affinity"]
            + 0.55 * f["support_coverage"]
            + 0.15 * f["support_density"]
            - 1.20 * f["broad"]
            - 0.70 * f["area_ratio"]
            - 0.30 * f["shrink_penalty"]
            for f in feats
        ]
    elif mode == "hard_negative_density_heavy":
        raw = [
            f["internal_affinity"]
            + 0.35 * f["support_coverage"]
            + 0.85 * f["support_density"]
            - 0.90 * f["broad"]
            - 0.45 * f["area_ratio"]
            - 0.20 * f["shrink_penalty"]
            for f in feats
        ]
    elif mode == "hard_negative_selection_blend":
        raw = [
            0.70 * f["internal_affinity"]
            + 0.30 * f["selection_score"]
            + 0.55 * f["support_coverage"]
            + 0.15 * f["support_density"]
            - 0.90 * f["broad"]
            - 0.30 * f["area_ratio"]
            - 0.20 * f["shrink_penalty"]
            for f in feats
        ]
    elif mode == "hard_negative_no_semantic_penalty":
        raw = [
            f["internal_affinity"]
            + 0.55 * f["support_coverage"]
            + 0.15 * f["support_density"]
            - 0.35 * f["area_ratio"]
            - 0.20 * f["shrink_penalty"]
            for f in feats
        ]
    elif mode == "residual_hard_negative":
        base = [
            f["internal_affinity"]
            + 0.55 * f["support_coverage"]
            + 0.15 * f["support_density"]
            - 0.90 * f["broad"]
            - 0.30 * f["area_ratio"]
            - 0.20 * f["shrink_penalty"]
            for f in feats
        ]
        raw = _residual(base, groups_sem_area)
    elif mode == "coverage_density_residual":
        base = [0.75 * cov + 0.25 * den for cov, den in zip(coverage, density)]
        raw = _residual(base, groups_sem_area)
    else:
        raise ValueError(mode)
    return [float(v) for v in raw]


def _variant_rows(base_rows: list[dict[str, str]], feature_map: dict[tuple[str, int, int], dict[str, Any]], variant_id: str, mode: str) -> list[dict[str, Any]]:
    raw = _score_values([dict(row) for row in base_rows], feature_map, mode)
    scores = raw if mode == "original" else _rank01(raw)
    out: list[dict[str, Any]] = []
    for row, raw_score, score in zip(base_rows, raw, scores):
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
                "raw_control_resistant_score": float(raw_score),
                "score_repair_mode": mode,
                "selection_reason": f"phase7c_control_resistant_score_{mode}_from_{BASE_VARIANT}",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def _evaluate_variant(variant_id: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original_mask_dir = recalc._mask_dir
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    scope = recalc._frame_scope()
    try:
        for scene in SCENES:
            recalc._mask_dir = lambda scene_id, _base=BASE_VARIANT: PHASE4_ROOT / "generated_masks" / _base / scene_id / "mask"
            scene_rows = [row for row in rows if row.get("scene_id") == scene]
            if not scene_rows:
                continue
            metric, cases, _tops, _window_rows = recalc._evaluate_frame_mask_variant_local_window(
                scene=scene,
                split="dev",
                variant=variant_id,
                frame_ids=scope.get(("dev", scene)),
                rows=scene_rows,
                score_mode="input",
                local_export_root=LOCAL_EXPORT_ROOT,
                window_source_step="S3D_L1_local_merged_masks",
            )
            metric_rows.append(
                {
                    **metric,
                    "variant_id": variant_id,
                    "MV_AP_window": metric.get("MV_AP"),
                    "MV_AP50_window": metric.get("MV_AP50"),
                    "MV_AP25_window": metric.get("MV_AP25"),
                    "score_free_Match50_window": phase1._f1(metric.get("SF50_precision"), metric.get("SF50_recall")),
                    "score_free_Match50_precision_window": metric.get("SF50_precision"),
                    "score_free_Match50_recall_window": metric.get("SF50_recall"),
                    "same_frame_collision_count": int(_int(metric.get("duplicate_frame_mask_conflict_count"), 0)),
                    "metric_scope": "local_window_gt_projection",
                }
            )
            case_rows.extend(cases)
    finally:
        recalc._mask_dir = original_mask_dir
    return metric_rows, case_rows


def _aggregate(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row.get("variant_id", ""))].append(row)
    out: list[dict[str, Any]] = []
    for variant, rows in sorted(grouped.items()):
        out.append(
            {
                "variant_id": variant,
                "scene_count": len(rows),
                "mean_MV_AP_window": _mean([_num(row.get("MV_AP_window")) for row in rows]),
                "mean_MV_AP50_window": _mean([_num(row.get("MV_AP50_window")) for row in rows]),
                "mean_MV_AP25_window": _mean([_num(row.get("MV_AP25_window")) for row in rows]),
                "mean_score_free_Match50_window": _mean([_num(row.get("score_free_Match50_window")) for row in rows]),
                "mean_gt_object_count": _mean([_num(row.get("gt_object_count")) for row in rows]),
                "mean_pred_object_count": _mean([_num(row.get("pred_object_count")) for row in rows]),
                "same_frame_collision_count": int(sum(_int(row.get("same_frame_collision_count")) for row in rows)),
                "missing_mask_raster_count": int(sum(_int(row.get("missing_mask_raster_count")) for row in rows)),
                "uses_gt_for_prediction": any(_bool(row.get("uses_gt_for_prediction")) for row in rows),
                "uses_future": any(_bool(row.get("uses_future")) for row in rows),
            }
        )
    return out


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    base_rows = [row for row in _read_csv(PHASE4_ROOT / "mv_object_frame_mask_rows.csv") if row.get("variant") == BASE_VARIANT]
    feature_map = _load_feature_rows()
    real_q0, best_control, phase7b_summary = _load_baselines()
    control_threshold = _num(best_control.get("mean_MV_AP_window"))
    base_mv_ap = _num(real_q0.get("mean_MV_AP_window"))
    variants = [
        ("H0_W9b_original_score", "original", "baseline parity; same masks and original W9b score"),
        ("H1_W9b_d4rt_internal_only", "d4rt_internal_only", "CONTROL_BIAS try D4RT witness-cover score without semantic/area positive terms"),
        ("H2_W9b_within_semantic_internal_residual", "within_semantic_internal_residual", "CONTROL_BIAS try within-semantic residual readout"),
        ("H3_W9b_within_window_internal_residual", "within_window_internal_residual", "CONTROL_BIAS try within-window hard-negative residual readout"),
        ("H4_W9b_hard_negative_d4rt", "hard_negative_d4rt", "CONTROL_BIAS try hard-negative instance separation score"),
        ("H5_W9b_residual_hard_negative", "residual_hard_negative", "CONTROL_BIAS try residual hard-negative instance separation"),
        ("H6_W9b_coverage_density_residual", "coverage_density_residual", "CONTROL_BIAS try carrier-supported support-density residual"),
        ("H7_W9b_hard_negative_support_heavy", "hard_negative_support_heavy", "CONTROL_BIAS hard-negative weight sweep: support-heavy"),
        ("H8_W9b_hard_negative_area_strong", "hard_negative_area_strong", "CONTROL_BIAS hard-negative weight sweep: stronger area/background penalties"),
        ("H9_W9b_hard_negative_density_heavy", "hard_negative_density_heavy", "CONTROL_BIAS hard-negative weight sweep: density-heavy"),
        ("H10_W9b_hard_negative_selection_blend", "hard_negative_selection_blend", "CONTROL_BIAS hard-negative weight sweep: selection-score blend"),
        ("H11_W9b_hard_negative_no_semantic_penalty", "hard_negative_no_semantic_penalty", "CONTROL_BIAS hard-negative weight sweep: remove semantic background penalty"),
    ]
    config_rows: list[dict[str, Any]] = []
    frame_mask_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for variant_id, mode, reason in variants:
        rows = _variant_rows(base_rows, feature_map, variant_id, mode)
        frame_mask_rows.extend(rows)
        metrics, cases = _evaluate_variant(variant_id, rows)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)
        config_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": BASE_VARIANT,
                "changed_parameters": f"control_resistant_score_mode={mode}",
                "changed_module": "object_score_only_control_resistant",
                "reason_for_change": reason,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "CONTROL_BIAS_BLOCKER",
            }
        )
    aggregate_rows = _aggregate(metric_rows)
    by_variant = {row["variant_id"]: row for row in aggregate_rows}
    best_real = max(aggregate_rows, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for config in config_rows:
        variant_id = config["variant_id"]
        row = by_variant.get(variant_id, {})
        lift = _num(row.get("mean_MV_AP_window")) - base_mv_ap
        control_gap = _num(row.get("mean_MV_AP_window")) - control_threshold
        pass_gate = lift > 0.002 and control_gap > 0.005 and not _bool(row.get("uses_gt_for_prediction")) and not _bool(row.get("uses_future"))
        gate = {
            **config,
            "actual_blocker": "NEEDS_HOLDOUT_IN_NEXT_VERSION" if pass_gate else "CONTROL_BIAS_BLOCKER",
            "MV_AP_window": row.get("mean_MV_AP_window", ""),
            "MV_AP50_window": row.get("mean_MV_AP50_window", ""),
            "MV_AP25_window": row.get("mean_MV_AP25_window", ""),
            "best_control_gap": control_gap,
            "B0_gap": "",
            "base_W9b_gap": lift,
            "control_threshold_variant": best_control.get("variant_id", CONTROL_BEST),
            "control_threshold_MV_AP_window": control_threshold,
            "same_frame_collision_count": row.get("same_frame_collision_count", ""),
            "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
            "gate_pass_control_resistant_dev_repair": pass_gate,
        }
        gate_rows.append(gate)
        if not pass_gate:
            failure_rows.append(
                {
                    "variant_id": variant_id,
                    "parent_variant_id": BASE_VARIANT,
                    "expected_blocker": "CONTROL_BIAS_BLOCKER",
                    "actual_blocker": "CONTROL_BIAS_BLOCKER",
                    "failure_reason": "not_above_base_by_0p002_and_control_by_0p005",
                    "MV_AP_window": row.get("mean_MV_AP_window", ""),
                    "base_W9b_gap": lift,
                    "best_control_gap": control_gap,
                    "control_threshold_variant": best_control.get("variant_id", CONTROL_BEST),
                    "control_threshold_MV_AP_window": control_threshold,
                }
            )
    any_pass = any(row.get("gate_pass_control_resistant_dev_repair") for row in gate_rows)
    summary = {
        "phase": "v90_phase7c_dev_control_resistant_score_repair",
        "schema": "stream4d_v90_phase7c_dev_control_resistant_score_repair_v1",
        "repair_scope": "dev_only_after_phase7b_control_bias_blocker_no_phase9_retune",
        "base_variant": BASE_VARIANT,
        "baseline_Q0": real_q0,
        "phase7b_control_audit": phase7b_summary,
        "control_threshold_variant": best_control.get("variant_id", CONTROL_BEST),
        "control_threshold_metrics": best_control,
        "best_real_variant": best_real.get("variant_id", ""),
        "best_real_metrics": best_real,
        "any_control_resistant_gate_pass": any_pass,
        "decision": "CONTROL_BIAS_REPAIRED_ON_DEV_ONLY" if any_pass else "CONTROL_BIAS_BLOCKER_REMAINS",
        "holdout_policy": "No v90 holdout rerun is allowed; this dev-only loop only tests whether real scores can clear the synchronous control threshold.",
        "row_counts": {
            "base_frame_mask_rows": len(base_rows),
            "feature_map_rows": len(feature_map),
            "variant_frame_mask_rows": len(frame_mask_rows),
            "metric_rows": len(metric_rows),
            "case_rows": len(case_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - started,
    }
    outputs = {
        "variant_config_rows": OUT / "variant_config_rows.csv",
        "variant_frame_mask_rows": OUT / "variant_frame_mask_rows.csv",
        "variant_metric_rows": OUT / "variant_metric_rows.csv",
        "variant_metric_aggregate_rows": OUT / "variant_metric_aggregate_rows.csv",
        "variant_gate_rows": OUT / "variant_gate_rows.csv",
        "variant_failure_rows": OUT / "variant_failure_rows.csv",
        "variant_case_rows": OUT / "variant_case_rows.csv",
        "best_variant_summary": OUT / "best_variant_summary.json",
    }
    _write_csv(outputs["variant_config_rows"], config_rows)
    _write_csv(outputs["variant_frame_mask_rows"], frame_mask_rows)
    _write_csv(outputs["variant_metric_rows"], metric_rows)
    _write_csv(outputs["variant_metric_aggregate_rows"], aggregate_rows)
    _write_csv(outputs["variant_gate_rows"], gate_rows)
    _write_csv(outputs["variant_failure_rows"], failure_rows)
    _write_csv(outputs["variant_case_rows"], case_rows)
    _write_json(outputs["best_variant_summary"], summary)
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs.values() if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
