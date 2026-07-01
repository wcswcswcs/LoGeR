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


OUT = ROOT / "outputs/audit/v90_phase7_dev_score_calibration_repair"
PHASE1_AGG = ROOT / "outputs/audit/v90_phase1_variant_resurrection/mv_metric_aggregate_rows.csv"
PHASE4_ROOT = ROOT / "outputs/audit/v90_phase4_geo_semantic_witness_cover"
BASE_VARIANT = "W9b_risk_balanced_p165_plus_carving"
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


def _score_values(rows: list[dict[str, Any]], feature_map: dict[tuple[str, int, int], dict[str, Any]], mode: str) -> list[float]:
    raw: list[float] = []
    semantic_groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))
        feat = feature_map.get(key, {})
        selection_score = _num(feat.get("selection_score"), _num(row.get("object_score"), 1.0))
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
        support_density = support_count / math.sqrt(generated_area)
        support_coverage = support_area / generated_area
        shrink_ratio = generated_area / source_area
        internal_affinity = math.log1p(carrier_count) * confidence * visibility + 40.0 * density
        if mode == "original":
            score = _num(row.get("object_score"), selection_score)
        elif mode == "pred_area":
            score = math.log1p(generated_area)
        elif mode == "support_coverage":
            score = math.log1p(support_count) + 1.5 * support_coverage - 0.25 * math.log1p(generated_area)
        elif mode == "internal_affinity":
            score = internal_affinity
        elif mode == "hybrid_fixed":
            score = 0.45 * selection_score + 0.25 * internal_affinity + 0.20 * support_coverage - 0.35 * broad - 0.15 * abs(math.log(max(1e-6, shrink_ratio)))
        elif mode == "control_matched_hybrid":
            score = 0.30 * selection_score + 0.35 * internal_affinity + 0.25 * support_density + 0.10 * margin - 0.10 * entropy - 0.50 * broad - 0.20 * area_ratio
            group = (row.get("scene_id", ""), str(feat.get("window_id", "")), str(feat.get("semantic_prototype_id", "")))
            semantic_groups[group].append(idx)
        else:
            raise ValueError(mode)
        raw.append(float(score))
    if mode != "control_matched_hybrid":
        return raw
    centered = list(raw)
    for indices in semantic_groups.values():
        group_mean = _mean([raw[idx] for idx in indices])
        for idx in indices:
            centered[idx] = raw[idx] - group_mean
    return centered


def _variant_rows(base_rows: list[dict[str, str]], feature_map: dict[tuple[str, int, int], dict[str, Any]], variant_id: str, mode: str) -> list[dict[str, Any]]:
    raw_scores = _score_values([dict(row) for row in base_rows], feature_map, mode)
    scores = raw_scores if mode == "original" else _rank01(raw_scores)
    out: list[dict[str, Any]] = []
    for row, raw_score, score in zip(base_rows, raw_scores, scores):
        obj = str(row.get("mv_object_id", ""))
        if obj.startswith(f"{BASE_VARIANT}:"):
            obj = obj.replace(f"{BASE_VARIANT}:", f"{variant_id}:", 1)
        item = {
            **row,
            "variant": variant_id,
            "source_variant": variant_id,
            "mv_object_id": obj,
            "frame_mask_score": float(score),
            "object_score": float(score),
            "raw_repair_score": float(raw_score),
            "score_repair_mode": mode,
            "selection_reason": f"phase7_dev_score_repair_{mode}_from_{BASE_VARIANT}",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        out.append(item)
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


def _lookup_phase_rows() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    phase1_rows = _read_csv(PHASE1_AGG)
    phase4_rows = _read_csv(PHASE4_ROOT / "mv_metric_aggregate_rows.csv")
    b0 = next((row for row in phase1_rows if row.get("variant_id") == "B0_local_only"), {})
    c0 = next((row for row in phase4_rows if row.get("variant_id") == "C0_W0_semantic_control"), {})
    w9b = next((row for row in phase4_rows if row.get("variant_id") == BASE_VARIANT), {})
    return b0, c0, w9b


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    base_rows = [row for row in _read_csv(PHASE4_ROOT / "mv_object_frame_mask_rows.csv") if row.get("variant") == BASE_VARIANT]
    feature_map = _load_feature_rows()
    b0, control, w9b = _lookup_phase_rows()
    variants = [
        ("Q0_W9b_original_score", "original", "baseline parity; same masks and original W9b selection score"),
        ("Q1_W9b_pred_area_score", "pred_area", "RANKING_BLOCKER try pred_area"),
        ("Q2_W9b_support_coverage_score", "support_coverage", "RANKING_BLOCKER try support coverage"),
        ("Q3_W9b_internal_affinity_score", "internal_affinity", "RANKING_BLOCKER try internal affinity"),
        ("Q4_W9b_hybrid_fixed_score", "hybrid_fixed", "RANKING_BLOCKER try hybrid fixed score"),
        ("Q5_W9b_control_matched_hybrid_score", "control_matched_hybrid", "RANKING_BLOCKER try control-matched hybrid score"),
    ]
    config_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    frame_mask_rows: list[dict[str, Any]] = []
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
                "changed_parameters": f"score_repair_mode={mode}",
                "changed_module": "object_score_only",
                "reason_for_change": reason,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "RANKING_BLOCKER",
            }
        )
    aggregate_rows = _aggregate(metric_rows)
    by_variant = {row["variant_id"]: row for row in aggregate_rows}
    base = by_variant.get("Q0_W9b_original_score", {})
    best = max(aggregate_rows, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for config in config_rows:
        variant_id = config["variant_id"]
        row = by_variant.get(variant_id, {})
        b0_gap = _num(row.get("mean_MV_AP_window")) - _num(b0.get("mean_MV_AP_window"))
        control_gap = _num(row.get("mean_MV_AP_window")) - _num(control.get("mean_MV_AP_window"))
        base_gap = _num(row.get("mean_MV_AP_window")) - _num(base.get("mean_MV_AP_window"))
        improves_over_base = base_gap > 0.002
        control_safe = control_gap > 0.005
        real_safe = not _bool(row.get("uses_gt_for_prediction")) and not _bool(row.get("uses_future"))
        gate_rows.append(
            {
                **config,
                "actual_blocker": "RANKING_BLOCKER" if not improves_over_base else "NEEDS_HOLDOUT_IN_NEXT_VERSION",
                "MV_AP_window": row.get("mean_MV_AP_window", ""),
                "MV_AP50_window": row.get("mean_MV_AP50_window", ""),
                "MV_AP25_window": row.get("mean_MV_AP25_window", ""),
                "best_control_gap": control_gap,
                "B0_gap": b0_gap,
                "base_W9b_gap": base_gap,
                "improves_over_base_gt_0p002": improves_over_base,
                "control_gap_gt_0p005": control_safe,
                "same_frame_collision_count": row.get("same_frame_collision_count", ""),
                "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
                "gate_pass_dev_ranking_repair": bool(improves_over_base and control_safe and real_safe),
            }
        )
        if not improves_over_base:
            failure_rows.append(
                {
                    "variant_id": variant_id,
                    "parent_variant_id": BASE_VARIANT,
                    "expected_blocker": "RANKING_BLOCKER",
                    "actual_blocker": "RANKING_BLOCKER",
                    "failure_reason": "dev_MV_AP_window_not_improved_over_Q0_by_0p002",
                    "MV_AP_window": row.get("mean_MV_AP_window", ""),
                    "base_W9b_gap": base_gap,
                    "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
                    "uses_future": row.get("uses_future", ""),
                }
            )
    best_gate = next((row for row in gate_rows if row.get("variant_id") == best.get("variant_id")), {})
    consecutive_no_lift = sum(1 for row in gate_rows if row["variant_id"] != "Q0_W9b_original_score" and not row["improves_over_base_gt_0p002"])
    stop_ranking_direction = consecutive_no_lift >= 5
    summary = {
        "phase": "v90_phase7_dev_score_calibration_repair",
        "schema": "stream4d_v90_phase7_dev_score_calibration_repair_v1",
        "repair_scope": "dev_only_after_v90_holdout_failure_no_phase9_retune",
        "base_variant": BASE_VARIANT,
        "best_variant_id": best.get("variant_id", ""),
        "best_variant_metrics": best,
        "best_variant_gate": best_gate,
        "Q0_original_score_metrics": base,
        "B0_local_only": b0,
        "best_control": control,
        "stop_ranking_direction": stop_ranking_direction,
        "stop_reason": "five_nonbaseline_score_variants_failed_to_improve_dev_MV_AP_window_by_0p002" if stop_ranking_direction else "",
        "holdout_policy": "No v90 holdout rerun is allowed from this repair loop; any improved dev variant is v91 candidate only.",
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
