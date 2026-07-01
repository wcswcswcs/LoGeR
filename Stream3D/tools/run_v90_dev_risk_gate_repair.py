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

from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402


OUT = ROOT / "outputs/audit/v90_phase7e_dev_risk_gate_repair"
PHASE4_ROOT = ROOT / "outputs/audit/v90_phase4_geo_semantic_witness_cover"
PHASE7B = ROOT / "outputs/audit/v90_phase7b_dev_score_control_audit"
BASE_VARIANT = "W9a_risk_balanced_p135_plus_carving"


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


def _phase4_summary() -> dict[str, Any]:
    path = PHASE4_ROOT / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _control_threshold() -> dict[str, Any]:
    path = PHASE7B / "best_variant_summary.json"
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")).get("best_control_metrics", {}))


def _risk(row: dict[str, Any], feature_map: dict[tuple[str, int, int], dict[str, Any]]) -> float:
    key = (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))
    feat = feature_map.get(key, {})
    return 1.0 if _bool(feat.get("broad_background_risk")) else 0.0


def _area_ratio(row: dict[str, Any], feature_map: dict[tuple[str, int, int], dict[str, Any]]) -> float:
    key = (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))
    return _num(feature_map.get(key, {}).get("area_ratio"), 0.0)


def _raw_h9(row: dict[str, Any], feature_map: dict[tuple[str, int, int], dict[str, Any]]) -> float:
    key = (row.get("scene_id", ""), _int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))
    return phase7d._h9_score(row, feature_map.get(key, {}))


def _keep_flags(rows: list[dict[str, str]], feature_map: dict[tuple[str, int, int], dict[str, Any]], mode: str) -> list[bool]:
    if mode == "all":
        return [True] * len(rows)
    risk = [_risk(row, feature_map) for row in rows]
    area = [_area_ratio(row, feature_map) for row in rows]
    score = [_raw_h9(row, feature_map) for row in rows]
    keep = [True] * len(rows)
    broad_indices = [i for i, value in enumerate(risk) if value >= 0.5]
    if mode.startswith("drop_broad_low_h9_"):
        frac = float(mode.rsplit("_", 1)[-1]) / 100.0
        count = max(1, int(round(len(broad_indices) * frac)))
        for idx in sorted(broad_indices, key=lambda i: score[i])[:count]:
            keep[idx] = False
    elif mode == "drop_broad_large_area":
        for i in broad_indices:
            if area[i] >= 0.25:
                keep[i] = False
    elif mode == "drop_broad_large_area_low_h9_5":
        cutoff_count = max(1, int(round(len(broad_indices) * 0.05)))
        low_score = set(sorted(broad_indices, key=lambda i: score[i])[:cutoff_count])
        for i in broad_indices:
            if area[i] >= 0.25 or i in low_score:
                keep[i] = False
    elif mode == "drop_broad_large_area_low_h9_10":
        cutoff_count = max(1, int(round(len(broad_indices) * 0.10)))
        low_score = set(sorted(broad_indices, key=lambda i: score[i])[:cutoff_count])
        for i in broad_indices:
            if area[i] >= 0.25 or i in low_score:
                keep[i] = False
    else:
        raise ValueError(mode)
    return keep


def _variant_rows(
    base_rows: list[dict[str, str]],
    feature_map: dict[tuple[str, int, int], dict[str, Any]],
    mode: str,
    variant_id: str,
) -> list[dict[str, Any]]:
    keep = _keep_flags(base_rows, feature_map, mode)
    kept_rows = [row for row, flag in zip(base_rows, keep) if flag]
    raw = [_raw_h9(dict(row), feature_map) for row in kept_rows]
    scores = _rank01(raw)
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
                "raw_h9_score": float(raw_score),
                "risk_filter_mode": mode,
                "base_extent_variant": BASE_VARIANT,
                "selection_reason": f"phase7e_risk_gate_{mode}_from_{BASE_VARIANT}_h9",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    phase4_summary = _phase4_summary()
    b0_risk = _num(phase4_summary.get("B0_risk_penalty_mean_proxy"), 1.0)
    control = _control_threshold()
    control_threshold = _num(control.get("mean_MV_AP_window"))
    feature_map = phase7d._load_feature_rows(BASE_VARIANT)
    base_rows = [row for row in _read_csv(PHASE4_ROOT / "mv_object_frame_mask_rows.csv") if row.get("variant") == BASE_VARIANT]
    modes = [
        "all",
        "drop_broad_low_h9_1",
        "drop_broad_low_h9_2",
        "drop_broad_low_h9_5",
        "drop_broad_low_h9_10",
        "drop_broad_large_area",
        "drop_broad_large_area_low_h9_5",
        "drop_broad_large_area_low_h9_10",
    ]
    config_rows: list[dict[str, Any]] = []
    frame_mask_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for mode in modes:
        variant_id = f"R_W9a_h9_{mode}"
        rows = _variant_rows(base_rows, feature_map, mode, variant_id)
        frame_mask_rows.extend(rows)
        metrics, cases = phase7d._evaluate_variant(BASE_VARIANT, variant_id, rows)
        metric_rows.extend(metrics)
        case_rows.extend({**row, "variant_id": variant_id} for row in cases)
        risks = [_risk(row, feature_map) for row in rows]
        risk_row = {
            "variant_id": variant_id,
            "risk_filter_mode": mode,
            "selected_rows": len(rows),
            "dropped_rows": len(base_rows) - len(rows),
            "risk_penalty_mean": _mean(risks),
            "B0_risk_penalty_mean_proxy": b0_risk,
            "risk_safe_vs_B0_proxy": _mean(risks) <= b0_risk + 1e-12,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        risk_rows.append(risk_row)
        config_rows.append(
            {
                "variant_id": variant_id,
                "parent_variant_id": BASE_VARIANT,
                "changed_parameters": f"risk_filter_mode={mode}; score_mode=h9_density_heavy",
                "changed_module": "existing_extent_plus_risk_filter_plus_object_score",
                "reason_for_change": "EXTENT/CONTROL_BIAS risk gate repair for W9a+H9 dev candidate",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "dev_only_or_holdout": "dev_only",
                "expected_blocker": "CONTROL_BIAS_BLOCKER+EXTENT_RISK_BLOCKER",
            }
        )
    aggregate_rows = phase7d._aggregate(metric_rows)
    by_variant = {row["variant_id"]: row for row in aggregate_rows}
    risk_by_variant = {row["variant_id"]: row for row in risk_rows}
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for config in config_rows:
        variant_id = config["variant_id"]
        metric = by_variant.get(variant_id, {})
        risk = risk_by_variant.get(variant_id, {})
        control_gap = _num(metric.get("mean_MV_AP_window")) - control_threshold
        gate_pass = bool(risk.get("risk_safe_vs_B0_proxy")) and control_gap > 0.005 and not _bool(metric.get("uses_gt_for_prediction")) and not _bool(metric.get("uses_future"))
        gate = {
            **config,
            "actual_blocker": "NEEDS_HOLDOUT_IN_NEXT_VERSION" if gate_pass else "EXTENT_RISK_OR_CONTROL_BIAS_BLOCKER",
            "MV_AP_window": metric.get("mean_MV_AP_window", ""),
            "MV_AP50_window": metric.get("mean_MV_AP50_window", ""),
            "MV_AP25_window": metric.get("mean_MV_AP25_window", ""),
            "best_control_gap": control_gap,
            "B0_gap": "",
            "risk_penalty_mean": risk.get("risk_penalty_mean", ""),
            "risk_safe_vs_B0_proxy": risk.get("risk_safe_vs_B0_proxy", ""),
            "selected_rows": risk.get("selected_rows", ""),
            "dropped_rows": risk.get("dropped_rows", ""),
            "control_threshold_variant": control.get("variant_id", ""),
            "control_threshold_MV_AP_window": control_threshold,
            "same_frame_collision_count": metric.get("same_frame_collision_count", ""),
            "missing_mask_raster_count": metric.get("missing_mask_raster_count", ""),
            "gate_pass_risk_gate_dev_repair": gate_pass,
        }
        gate_rows.append(gate)
        if not gate_pass:
            failure_rows.append(
                {
                    "variant_id": variant_id,
                    "parent_variant_id": BASE_VARIANT,
                    "expected_blocker": "CONTROL_BIAS_BLOCKER+EXTENT_RISK_BLOCKER",
                    "actual_blocker": "EXTENT_RISK_OR_CONTROL_BIAS_BLOCKER",
                    "failure_reason": "not_risk_safe_and_control_gap_gt_0p005",
                    "MV_AP_window": metric.get("mean_MV_AP_window", ""),
                    "best_control_gap": control_gap,
                    "risk_penalty_mean": risk.get("risk_penalty_mean", ""),
                    "risk_safe_vs_B0_proxy": risk.get("risk_safe_vs_B0_proxy", ""),
                }
            )
    passing = [row for row in gate_rows if row.get("gate_pass_risk_gate_dev_repair")]
    best = max(passing or gate_rows, key=lambda row: _num(row.get("MV_AP_window")), default={})
    any_pass = bool(passing)
    summary = {
        "phase": "v90_phase7e_dev_risk_gate_repair",
        "schema": "stream4d_v90_phase7e_dev_risk_gate_repair_v1",
        "repair_scope": "dev_only_risk_gate_repair_for_phase7d_W9a_H9_candidate",
        "base_extent_variant": BASE_VARIANT,
        "B0_risk_penalty_mean_proxy": b0_risk,
        "control_threshold_variant": control.get("variant_id", ""),
        "control_threshold_metrics": control,
        "best_variant_id": best.get("variant_id", ""),
        "best_variant_gate": best,
        "any_risk_gate_dev_pass": any_pass,
        "decision": "DEV_RISK_SAFE_EXTENT_SCORE_CANDIDATE_FOUND" if any_pass else "EXTENT_RISK_OR_CONTROL_BIAS_BLOCKER_REMAINS",
        "holdout_policy": "No v90 holdout rerun is allowed; this dev-only loop only tests whether W9a+H9 can satisfy risk and control gates.",
        "row_counts": {
            "base_frame_mask_rows": len(base_rows),
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
        "variant_risk_rows": OUT / "variant_risk_rows.csv",
        "variant_case_rows": OUT / "variant_case_rows.csv",
        "best_variant_summary": OUT / "best_variant_summary.json",
    }
    _write_csv(outputs["variant_config_rows"], config_rows)
    _write_csv(outputs["variant_frame_mask_rows"], frame_mask_rows)
    _write_csv(outputs["variant_metric_rows"], metric_rows)
    _write_csv(outputs["variant_metric_aggregate_rows"], aggregate_rows)
    _write_csv(outputs["variant_gate_rows"], gate_rows)
    _write_csv(outputs["variant_failure_rows"], failure_rows)
    _write_csv(outputs["variant_risk_rows"], risk_rows)
    _write_csv(outputs["variant_case_rows"], case_rows)
    _write_json(outputs["best_variant_summary"], summary)
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs.values() if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
