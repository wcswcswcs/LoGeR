from __future__ import annotations

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

from tools import run_v89_recalc_point_projected_mv_ap as recalc  # noqa: E402
from tools import run_v90_mv_ap_window_resurrection as phase1  # noqa: E402


OUT = ROOT / "outputs/audit/v90_phase7d_dev_extent_score_cross_audit"
PHASE4_ROOT = ROOT / "outputs/audit/v90_phase4_geo_semantic_witness_cover"
PHASE7B = ROOT / "outputs/audit/v90_phase7b_dev_score_control_audit"
LOCAL_EXPORT_ROOT = ROOT / "outputs/audit/v89_recalc_point_projected_mv_ap"
SCENES = ["scene0011_00", "scene0050_00"]
BASE_VARIANTS = [
    "W4_witness_cover_plus_carving",
    "W7_risk_controlled_witness_cover_plus_carving",
    "W9a_risk_balanced_p135_plus_carving",
    "W9b_risk_balanced_p165_plus_carving",
    "W9c_risk_balanced_p195_plus_carving",
]


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


def _load_feature_rows(base_variant: str) -> dict[tuple[str, int, int], dict[str, Any]]:
    selection = _read_csv(PHASE4_ROOT / "witness_cover_selection_rows.csv")
    generated = _read_csv(PHASE4_ROOT / "generated_mask_rows.csv")
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in selection:
        if row.get("variant_id") != base_variant:
            continue
        key = (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("new_mask_id"), -1))
        if key[0] and key[1] >= 0 and key[2] > 0:
            by_key[key] = dict(row)
    for row in generated:
        if row.get("variant_id") != base_variant:
            continue
        key = (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("new_mask_id"), -1))
        if key in by_key:
            by_key[key].update({f"generated_{k}": v for k, v in row.items()})
        elif key[0] and key[1] >= 0 and key[2] > 0:
            by_key[key] = {f"generated_{k}": v for k, v in row.items()}
    return by_key


def _h9_score(row: dict[str, Any], feat: dict[str, Any]) -> float:
    support_count = _num(feat.get("support_count"), _num(feat.get("generated_carrier_support_count"), 0.0))
    carrier_count = _num(feat.get("carrier_count_unique"), support_count)
    confidence = _num(feat.get("confidence_mean"), 1.0)
    visibility = _num(feat.get("visibility_mean"), 1.0)
    density = _num(feat.get("observed_density_mean"), 0.0)
    source_area = max(1.0, _num(feat.get("source_mask_area"), _num(feat.get("generated_source_mask_area"), 1.0)))
    generated_area = max(1.0, _num(feat.get("generated_mask_area"), _num(feat.get("generated_generated_mask_area"), 1.0)))
    support_area = max(1.0, _num(feat.get("support_area"), _num(feat.get("generated_support_area"), 1.0)))
    broad = 1.0 if _bool(feat.get("broad_background_risk")) else 0.0
    area_ratio = _num(feat.get("area_ratio"), generated_area / 1254528.0)
    shrink = generated_area / source_area
    internal_affinity = math.log1p(carrier_count) * confidence * visibility + 40.0 * density
    support_coverage = support_area / generated_area
    support_density = support_count / math.sqrt(generated_area)
    return float(
        internal_affinity
        + 0.35 * support_coverage
        + 0.85 * support_density
        - 0.90 * broad
        - 0.45 * area_ratio
        - 0.20 * abs(math.log(max(1e-6, shrink)))
    )


def _variant_rows(base_variant: str, rows: list[dict[str, str]], feature_map: dict[tuple[str, int, int], dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    raw: list[float] = []
    for row in rows:
        key = (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))
        feat = feature_map.get(key, {})
        if mode == "original":
            raw.append(_num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)))
        elif mode == "h9_density_heavy":
            raw.append(_h9_score(dict(row), feat))
        else:
            raise ValueError(mode)
    scores = raw if mode == "original" else _rank01(raw)
    variant_id = f"X_{base_variant}_{mode}"
    out: list[dict[str, Any]] = []
    for row, raw_score, score in zip(rows, raw, scores):
        obj = str(row.get("mv_object_id", ""))
        if obj.startswith(f"{base_variant}:"):
            obj = obj.replace(f"{base_variant}:", f"{variant_id}:", 1)
        out.append(
            {
                **row,
                "variant": variant_id,
                "source_variant": variant_id,
                "mv_object_id": obj,
                "frame_mask_score": float(score),
                "object_score": float(score),
                "raw_cross_score": float(raw_score),
                "score_mode_detail": mode,
                "base_extent_variant": base_variant,
                "selection_reason": f"phase7d_extent_score_cross_{mode}_from_{base_variant}",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def _evaluate_variant(base_variant: str, variant_id: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original_mask_dir = recalc._mask_dir
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    scope = recalc._frame_scope()
    try:
        for scene in SCENES:
            recalc._mask_dir = lambda scene_id, _base=base_variant: PHASE4_ROOT / "generated_masks" / _base / scene_id / "mask"
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
                    "base_extent_variant": base_variant,
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
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped.setdefault(str(row.get("variant_id", "")), []).append(row)
    out: list[dict[str, Any]] = []
    for variant, rows in sorted(grouped.items()):
        out.append(
            {
                "variant_id": variant,
                "base_extent_variant": rows[0].get("base_extent_variant", "") if rows else "",
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
    phase7b_summary_path = PHASE7B / "best_variant_summary.json"
    phase7b = json.loads(phase7b_summary_path.read_text(encoding="utf-8")) if phase7b_summary_path.exists() else {}
    control = dict(phase7b.get("best_control_metrics", {}))
    control_threshold = _num(control.get("mean_MV_AP_window"))
    config_rows: list[dict[str, Any]] = []
    frame_mask_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for base_variant in BASE_VARIANTS:
        rows = [row for row in _read_csv(PHASE4_ROOT / "mv_object_frame_mask_rows.csv") if row.get("variant") == base_variant]
        feature_map = _load_feature_rows(base_variant)
        for mode in ["original", "h9_density_heavy"]:
            variant_id = f"X_{base_variant}_{mode}"
            out_rows = _variant_rows(base_variant, rows, feature_map, mode)
            frame_mask_rows.extend(out_rows)
            metrics, cases = _evaluate_variant(base_variant, variant_id, out_rows)
            metric_rows.extend(metrics)
            case_rows.extend({**row, "variant_id": variant_id} for row in cases)
            config_rows.append(
                {
                    "variant_id": variant_id,
                    "parent_variant_id": base_variant,
                    "changed_parameters": f"extent_variant={base_variant}; score_mode={mode}",
                    "changed_module": "existing_extent_plus_object_score",
                    "reason_for_change": "CONTROL_BIAS/EXTENT cross-check using existing Phase4 carved masks",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "dev_only_or_holdout": "dev_only",
                    "expected_blocker": "CONTROL_BIAS_BLOCKER+EXTENT_BLOCKER",
                }
            )
    aggregate_rows = _aggregate(metric_rows)
    best = max(aggregate_rows, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for config in config_rows:
        row = next((item for item in aggregate_rows if item.get("variant_id") == config["variant_id"]), {})
        control_gap = _num(row.get("mean_MV_AP_window")) - control_threshold
        gate_pass = control_gap > 0.005 and not _bool(row.get("uses_gt_for_prediction")) and not _bool(row.get("uses_future"))
        gate_rows.append(
            {
                **config,
                "actual_blocker": "NEEDS_HOLDOUT_IN_NEXT_VERSION" if gate_pass else "CONTROL_BIAS_BLOCKER_OR_EXTENT_BLOCKER",
                "MV_AP_window": row.get("mean_MV_AP_window", ""),
                "MV_AP50_window": row.get("mean_MV_AP50_window", ""),
                "MV_AP25_window": row.get("mean_MV_AP25_window", ""),
                "best_control_gap": control_gap,
                "B0_gap": "",
                "control_threshold_variant": control.get("variant_id", ""),
                "control_threshold_MV_AP_window": control_threshold,
                "same_frame_collision_count": row.get("same_frame_collision_count", ""),
                "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
                "gate_pass_extent_score_cross_dev": gate_pass,
            }
        )
        if not gate_pass:
            failure_rows.append(
                {
                    "variant_id": config["variant_id"],
                    "parent_variant_id": config["parent_variant_id"],
                    "expected_blocker": "CONTROL_BIAS_BLOCKER+EXTENT_BLOCKER",
                    "actual_blocker": "CONTROL_BIAS_BLOCKER_OR_EXTENT_BLOCKER",
                    "failure_reason": "not_above_synchronous_control_by_0p005",
                    "MV_AP_window": row.get("mean_MV_AP_window", ""),
                    "best_control_gap": control_gap,
                    "control_threshold_variant": control.get("variant_id", ""),
                    "control_threshold_MV_AP_window": control_threshold,
                }
            )
    any_pass = any(row.get("gate_pass_extent_score_cross_dev") for row in gate_rows)
    summary = {
        "phase": "v90_phase7d_dev_extent_score_cross_audit",
        "schema": "stream4d_v90_phase7d_dev_extent_score_cross_audit_v1",
        "repair_scope": "dev_only_existing_phase4_extents_after_phase7c_control_bias",
        "control_threshold_variant": control.get("variant_id", ""),
        "control_threshold_metrics": control,
        "best_variant_id": best.get("variant_id", ""),
        "best_variant_metrics": best,
        "any_extent_score_cross_gate_pass": any_pass,
        "decision": "DEV_EXTENT_SCORE_CANDIDATE_FOUND" if any_pass else "CONTROL_BIAS_OR_EXTENT_BLOCKER_REMAINS",
        "holdout_policy": "No v90 holdout rerun is allowed; this only audits existing dev extents and scores for v91 direction.",
        "row_counts": {
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
