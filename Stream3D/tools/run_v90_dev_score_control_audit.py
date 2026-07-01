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

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v89_recalc_point_projected_mv_ap as recalc  # noqa: E402
from tools import run_v90_geo_semantic_witness_cover as phase4  # noqa: E402
from tools import run_v90_mv_ap_window_resurrection as phase1  # noqa: E402


OUT = ROOT / "outputs/audit/v90_phase7b_dev_score_control_audit"
PHASE4_ROOT = ROOT / "outputs/audit/v90_phase4_geo_semantic_witness_cover"
PHASE7_REPAIR = ROOT / "outputs/audit/v90_phase7_dev_score_calibration_repair"
CONTROL_BASE_VARIANT = "C0_W0_semantic_control"
REAL_BASE_VARIANT = "Q0_W9b_original_score"
REAL_BEST_VARIANT = "Q4_W9b_hybrid_fixed_score"
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


def _read_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int64, copy=False)


def _mask_area_cache(mask_dirs: dict[str, Path]):
    cache: dict[tuple[str, int], tuple[np.ndarray, int]] = {}

    def area(scene: str, frame_id: int, mask_id: int) -> tuple[int, int]:
        key = (scene, frame_id)
        if key not in cache:
            label = _read_label(mask_dirs[scene] / f"{int(frame_id)}.png")
            counts = np.bincount(label.reshape(-1).astype(np.int64))
            cache[key] = (counts, int(label.size))
        counts, image_area = cache[key]
        mask_area = int(counts[mask_id]) if 0 <= mask_id < counts.shape[0] else 0
        return mask_area, image_area

    return area


def _load_real_repair_metrics() -> tuple[dict[str, str], dict[str, str]]:
    rows = _read_csv(PHASE7_REPAIR / "variant_metric_aggregate_rows.csv")
    q0 = next((row for row in rows if row.get("variant_id") == REAL_BASE_VARIANT), {})
    q4 = next((row for row in rows if row.get("variant_id") == REAL_BEST_VARIANT), {})
    return q0, q4


def _control_raw_scores(
    rows: list[dict[str, Any]],
    mode: str,
    semantic: dict[tuple[str, int, int], dict[str, Any]],
    area_fn,
) -> list[float]:
    out: list[float] = []
    for row in rows:
        scene = row.get("scene_id", "")
        frame_id = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("mask_id"), -1)
        feat = semantic.get((scene, frame_id, mask_id), {})
        mask_area, image_area = area_fn(scene, frame_id, mask_id)
        area_ratio = float(mask_area / max(1, image_area))
        margin = _num(feat.get("semantic_prototype_margin"), 0.0)
        entropy = _num(feat.get("semantic_entropy"), 1.0)
        broad = 1.0 if _bool(feat.get("broad_background_risk")) or _bool(feat.get("semantic_background_score_proxy")) else 0.0
        original = _num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0))
        if mode == "original":
            score = original
        elif mode == "pred_area":
            score = math.log1p(mask_area)
        elif mode == "semantic_margin":
            score = margin - 0.20 * entropy - 0.50 * broad
        elif mode == "area_semantic_hybrid":
            score = 0.45 * math.log1p(mask_area) + 0.55 * margin - 0.15 * entropy - 0.50 * broad - 0.20 * area_ratio
        elif mode == "small_mask_semantic":
            score = 0.60 * margin - 0.20 * entropy - 0.70 * broad - 0.35 * area_ratio
        else:
            raise ValueError(mode)
        out.append(float(score))
    return out


def _variant_rows(
    base_rows: list[dict[str, str]],
    variant_id: str,
    mode: str,
    semantic: dict[tuple[str, int, int], dict[str, Any]],
    area_fn,
) -> list[dict[str, Any]]:
    raw = _control_raw_scores([dict(row) for row in base_rows], mode, semantic, area_fn)
    scores = raw if mode == "original" else _rank01(raw)
    out: list[dict[str, Any]] = []
    for row, raw_score, score in zip(base_rows, raw, scores):
        obj = str(row.get("mv_object_id", ""))
        if obj.startswith(f"{CONTROL_BASE_VARIANT}:"):
            obj = obj.replace(f"{CONTROL_BASE_VARIANT}:", f"{variant_id}:", 1)
        out.append(
            {
                **row,
                "variant": variant_id,
                "source_variant": variant_id,
                "mv_object_id": obj,
                "frame_mask_score": float(score),
                "object_score": float(score),
                "raw_control_score": float(raw_score),
                "score_control_mode": mode,
                "selection_reason": f"phase7b_dev_control_score_audit_{mode}_from_{CONTROL_BASE_VARIANT}",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def _evaluate_variant(variant_id: str, rows: list[dict[str, Any]], mask_dirs: dict[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original_mask_dir = recalc._mask_dir
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    scope = recalc._frame_scope()
    try:
        for scene in SCENES:
            recalc._mask_dir = lambda scene_id, _mask_dirs=mask_dirs: _mask_dirs[scene_id]
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
    base_rows = [row for row in _read_csv(PHASE4_ROOT / "mv_object_frame_mask_rows.csv") if row.get("variant") == CONTROL_BASE_VARIANT]
    mask_dirs = phase4._mask_dir_by_scene()
    semantic = phase4._load_semantic_features()
    area_fn = _mask_area_cache(mask_dirs)
    real_base, real_best = _load_real_repair_metrics()
    real_lift = _num(real_best.get("mean_MV_AP_window")) - _num(real_base.get("mean_MV_AP_window"))
    variants = [
        ("P0_C0_original_score", "original", "control baseline parity; same masks and original C0 score"),
        ("P1_C0_pred_area_score", "pred_area", "same-protocol control try pred_area"),
        ("P2_C0_semantic_margin_score", "semantic_margin", "same-protocol control try semantic margin"),
        ("P3_C0_area_semantic_hybrid_score", "area_semantic_hybrid", "same-protocol control try area-semantic hybrid score"),
        ("P4_C0_small_mask_semantic_score", "small_mask_semantic", "same-protocol control try small-mask semantic score"),
    ]
    config_rows: list[dict[str, Any]] = []
    frame_mask_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for variant_id, mode, reason in variants:
        rows = _variant_rows(base_rows, variant_id, mode, semantic, area_fn)
        frame_mask_rows.extend(rows)
        metrics, cases = _evaluate_variant(variant_id, rows, mask_dirs)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)
        config_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": CONTROL_BASE_VARIANT,
                "changed_parameters": f"score_control_mode={mode}",
                "changed_module": "control_object_score_only",
                "reason_for_change": reason,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "dev_only_or_holdout": "dev_only_control_audit",
                "expected_blocker": "CONTROL_BIAS_BLOCKER",
            }
        )
    aggregate_rows = _aggregate(metric_rows)
    by_variant = {row["variant_id"]: row for row in aggregate_rows}
    control_base = by_variant.get("P0_C0_original_score", {})
    best_control = max(aggregate_rows, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    control_lift = _num(best_control.get("mean_MV_AP_window")) - _num(control_base.get("mean_MV_AP_window"))
    real_best_mv = _num(real_best.get("mean_MV_AP_window"))
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for config in config_rows:
        variant_id = config["variant_id"]
        row = by_variant.get(variant_id, {})
        lift = _num(row.get("mean_MV_AP_window")) - _num(control_base.get("mean_MV_AP_window"))
        beats_real = _num(row.get("mean_MV_AP_window")) >= real_best_mv
        improves_more_than_real = lift >= real_lift
        gate_pass = not beats_real and not improves_more_than_real
        gate_rows.append(
            {
                **config,
                "actual_blocker": "CONTROL_BIAS_BLOCKER" if not gate_pass else "CONTROL_AUDIT_PASS",
                "MV_AP_window": row.get("mean_MV_AP_window", ""),
                "MV_AP50_window": row.get("mean_MV_AP50_window", ""),
                "MV_AP25_window": row.get("mean_MV_AP25_window", ""),
                "best_control_gap": _num(row.get("mean_MV_AP_window")) - _num(control_base.get("mean_MV_AP_window")),
                "B0_gap": "",
                "control_lift_vs_P0": lift,
                "real_Q4_lift_vs_Q0": real_lift,
                "control_MV_AP_ge_real_Q4": beats_real,
                "control_lift_ge_real_lift": improves_more_than_real,
                "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
                "uses_future": row.get("uses_future", ""),
                "same_frame_collision_count": row.get("same_frame_collision_count", ""),
                "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
                "gate_pass_control_audit": gate_pass,
            }
        )
        if not gate_pass:
            failure_rows.append(
                {
                    "variant_id": variant_id,
                    "parent_variant_id": CONTROL_BASE_VARIANT,
                    "expected_blocker": "CONTROL_BIAS_BLOCKER",
                    "actual_blocker": "CONTROL_BIAS_BLOCKER",
                    "failure_reason": "control_beats_real_Q4_or_control_lift_ge_real_lift",
                    "MV_AP_window": row.get("mean_MV_AP_window", ""),
                    "control_lift_vs_P0": lift,
                    "real_Q4_lift_vs_Q0": real_lift,
                    "control_MV_AP_ge_real_Q4": beats_real,
                    "control_lift_ge_real_lift": improves_more_than_real,
                }
            )
    control_audit_pass = not any(row.get("control_MV_AP_ge_real_Q4") or row.get("control_lift_ge_real_lift") for row in gate_rows)
    summary = {
        "phase": "v90_phase7b_dev_score_control_audit",
        "schema": "stream4d_v90_phase7b_dev_score_control_audit_v1",
        "repair_scope": "dev_only_control_audit_after_Q4_score_repair",
        "control_base_variant": CONTROL_BASE_VARIANT,
        "real_base_variant": REAL_BASE_VARIANT,
        "real_best_variant": REAL_BEST_VARIANT,
        "real_Q0_metrics": real_base,
        "real_Q4_metrics": real_best,
        "real_Q4_lift_vs_Q0": real_lift,
        "best_control_variant": best_control.get("variant_id", ""),
        "best_control_metrics": best_control,
        "best_control_lift_vs_P0": control_lift,
        "control_audit_pass": control_audit_pass,
        "decision": "Q4_dev_gain_not_explained_by_synchronous_controls" if control_audit_pass else "CONTROL_BIAS_BLOCKER",
        "holdout_policy": "No v90 holdout rerun is allowed; this audit only checks whether the dev-only Q4 score gain is control-dominated.",
        "row_counts": {
            "base_control_frame_mask_rows": len(base_rows),
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
