#!/usr/bin/env python3
"""Holdout-only support-score diagnostic for v99 Phase10.

This diagnostic is not a method claim because the support_iou/mask_precision
fields are present in the fixed holdout adapter rows but not in the Phase2
full-dev rows with the same schema. It tests whether those non-GT holdout-side
features have enough ranking signal to explain the Phase10C oracle gap.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v98_1_canonical_holdout_metrics as holdout  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10e_holdout_support_score_diagnostic"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
HOLDOUT_SOURCE_ROWS = AUDIT_ROOT / "v98_phase13_holdout/source_container_rows.csv"
HOLDOUT_FIXED_ROWS = PHASE2_DIR / "holdout_mv_object_frame_mask_rows.csv"


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _norm(values: dict[str, float]) -> dict[str, float]:
    vals = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not vals:
        return {key: 0.0 for key in values}
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (float(value) - lo) / (hi - lo) for key, value in values.items()}


def _object_features(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        oid = str(row.get("mv_object_id", ""))
        if not oid:
            continue
        grouped[oid]["score"].append(_num(row.get("score")))
        grouped[oid]["support_iou"].append(_num(row.get("support_iou")))
        grouped[oid]["mask_precision"].append(_num(row.get("mask_precision")))
        grouped[oid]["support_area"].append(_num(row.get("support_area")))
    parent = {oid: max(vals["score"]) if vals["score"] else 0.0 for oid, vals in grouped.items()}
    support_iou_mean = {oid: float(np.mean(vals["support_iou"])) if vals["support_iou"] else 0.0 for oid, vals in grouped.items()}
    support_iou_max = {oid: max(vals["support_iou"]) if vals["support_iou"] else 0.0 for oid, vals in grouped.items()}
    precision_mean = {oid: float(np.mean(vals["mask_precision"])) if vals["mask_precision"] else 0.0 for oid, vals in grouped.items()}
    precision_max = {oid: max(vals["mask_precision"]) if vals["mask_precision"] else 0.0 for oid, vals in grouped.items()}
    area_mean = {oid: float(np.mean(vals["support_area"])) if vals["support_area"] else 0.0 for oid, vals in grouped.items()}
    parent_n = _norm(parent)
    support_iou_mean_n = _norm(support_iou_mean)
    support_iou_max_n = _norm(support_iou_max)
    precision_mean_n = _norm(precision_mean)
    precision_max_n = _norm(precision_max)
    area_mean_n = _norm(area_mean)
    out: dict[str, dict[str, float]] = {}
    for oid in grouped:
        out[oid] = {
            "parent": parent[oid],
            "parent_norm": parent_n.get(oid, 0.0),
            "support_iou_mean_norm": support_iou_mean_n.get(oid, 0.0),
            "support_iou_max_norm": support_iou_max_n.get(oid, 0.0),
            "mask_precision_mean_norm": precision_mean_n.get(oid, 0.0),
            "mask_precision_max_norm": precision_max_n.get(oid, 0.0),
            "support_area_mean_norm": area_mean_n.get(oid, 0.0),
        }
    return out


def _score(config: dict[str, Any], f: dict[str, float]) -> float:
    mode = str(config["mode"])
    if mode == "parent":
        return f["parent"]
    if mode == "support_iou_mean_only":
        return f["support_iou_mean_norm"]
    if mode == "support_iou_max_only":
        return f["support_iou_max_norm"]
    if mode == "mask_precision_mean_only":
        return f["mask_precision_mean_norm"]
    if mode == "support_area_mean_only":
        return f["support_area_mean_norm"]
    if mode == "inverse_support_area_mean_only":
        return 1.0 - f["support_area_mean_norm"]
    if mode.startswith("blend_"):
        w = float(config["parent_weight"])
        feature = str(config["feature"])
        return w * f["parent_norm"] + (1.0 - w) * f[feature]
    raise ValueError(f"unknown mode {mode}")


def _make_rows(parent_rows: list[dict[str, Any]], features: dict[str, dict[str, float]], config: dict[str, Any]) -> list[dict[str, Any]]:
    variant = f"V99P10E_holdout_{config['name']}"
    out: list[dict[str, Any]] = []
    for row in parent_rows:
        oid = str(row.get("mv_object_id", ""))
        f = features.get(oid)
        if f is None:
            continue
        new = dict(row)
        new["variant_id"] = variant
        new["variant"] = variant
        new["score"] = float(_score(config, f))
        new["score_policy"] = config["score_policy"]
        new["phase10e_parent_variant_id"] = row.get("variant_id", "")
        new["phase10e_mode"] = config["mode"]
        new["phase10e_parent_weight"] = config.get("parent_weight", "")
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    scope = holdout._load_source_scope(HOLDOUT_SOURCE_ROWS)
    parent_rows = [dict(row) for row in _read_csv(HOLDOUT_FIXED_ROWS)]
    features = _object_features(parent_rows)
    configs = [
        {"name": "S0_parent", "mode": "parent", "score_policy": "phase10e_parent_score_replay"},
        {"name": "S1_support_iou_mean_only", "mode": "support_iou_mean_only", "score_policy": "phase10e_support_iou_mean_only"},
        {"name": "S2_support_iou_max_only", "mode": "support_iou_max_only", "score_policy": "phase10e_support_iou_max_only"},
        {"name": "S3_mask_precision_mean_only", "mode": "mask_precision_mean_only", "score_policy": "phase10e_mask_precision_mean_only"},
        {"name": "S4_support_area_mean_only", "mode": "support_area_mean_only", "score_policy": "phase10e_support_area_mean_only"},
        {"name": "S5_inverse_support_area_mean_only", "mode": "inverse_support_area_mean_only", "score_policy": "phase10e_inverse_support_area_mean_only"},
        {"name": "S6_blend_parent_support_iou_90_10", "mode": "blend_support", "parent_weight": 0.90, "feature": "support_iou_mean_norm", "score_policy": "phase10e_0p90_parent_0p10_support_iou_mean"},
        {"name": "S7_blend_parent_precision_90_10", "mode": "blend_precision", "parent_weight": 0.90, "feature": "mask_precision_mean_norm", "score_policy": "phase10e_0p90_parent_0p10_mask_precision_mean"},
        {"name": "S8_blend_parent_support_area_90_10", "mode": "blend_support_area", "parent_weight": 0.90, "feature": "support_area_mean_norm", "score_policy": "phase10e_0p90_parent_0p10_support_area_mean"},
    ]
    config_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for config in configs:
        rows = _make_rows(parent_rows, features, config)
        metrics, cases, tops = holdout._evaluate_variant(f"V99P10E_holdout_{config['name']}", rows, scope)
        all_rows.extend(rows)
        metric_rows.extend(metrics)
        case_rows.extend(cases)
        top_rows.extend(tops)
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase10e_variant_config_v1",
                "phase_id": "v99_phase10e_holdout_support_score_diagnostic",
                "name": config["name"],
                "mode": config["mode"],
                "parent_weight": config.get("parent_weight", ""),
                "feature": config.get("feature", ""),
                "score_policy": config["score_policy"],
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
                "formal_claim_blocker": "holdout_only_feature_schema_no_full_dev_counterpart",
            }
        )
    agg = holdout._aggregate(metric_rows, family="v99_phase10e_holdout_support_score_diagnostic")
    f2_hold_window = float(phase0["F2_base_holdout_MV_AP_window"])
    f2_hold_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"])
    paired: list[dict[str, Any]] = []
    for row in agg:
        name = str(row["variant_id"]).replace("V99P10E_holdout_", "")
        gate = _num(row.get("mean_MV_AP_window")) >= f2_hold_window + 0.005 and _num(row.get("mean_MV_AP50_window")) >= f2_hold_ap50 + 0.010
        paired.append(
            {
                "schema_version": "stream4d_v99_phase10e_holdout_metric_v1",
                "phase_id": "v99_phase10e_holdout_support_score_diagnostic",
                "name": name,
                "holdout_variant_id": row["variant_id"],
                "holdout_MV_AP_window": row.get("mean_MV_AP_window"),
                "holdout_MV_AP50_window": row.get("mean_MV_AP50_window"),
                "holdout_MV_AP25_window": row.get("mean_MV_AP25_window"),
                "holdout_delta_vs_F2_base_window": _num(row.get("mean_MV_AP_window")) - f2_hold_window,
                "holdout_gate_pass": gate,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )
    best = max(paired, key=lambda row: (_num(row["holdout_MV_AP_window"]), _num(row["holdout_MV_AP50_window"])))
    any_pass = any(bool(row["holdout_gate_pass"]) for row in paired)
    gate_rows = [
        {
            "gate_id": "holdout_only_support_score_passes_strict_holdout_gate",
            "pass": any_pass,
            "expected": f"MV_AP_window>={f2_hold_window + 0.005} and MV_AP50_window>={f2_hold_ap50 + 0.010}",
            "observed": f"best={best['name']} MV_AP_window={best['holdout_MV_AP_window']} MV_AP50_window={best['holdout_MV_AP50_window']}",
            "severity": "diagnostic",
        },
        {
            "gate_id": "formal_claim_allowed",
            "pass": False,
            "expected": "same feature schema on dev and frozen holdout",
            "observed": "support_iou/mask_precision/support_area are holdout-side diagnostic fields here",
            "severity": "formal_claim_blocker",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If support fields pass holdout-only, implement same-schema support features on full-dev and require fresh frozen holdout; otherwise support-score ranking is insufficient.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10e_holdout_support_score_diagnostic_summary_v1",
        "phase_id": "v99_phase10e_holdout_support_score_diagnostic",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "DIAGNOSTIC_SUPPORT_SCORE_HOLDOUT_ONLY_PASSES_REQUIRES_DEV_SCHEMA_AND_FRESH_HOLDOUT" if any_pass else "NO_GO_HOLDOUT_SUPPORT_SCORE_DIAGNOSTIC",
        "formal_claim_allowed": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "variant_count": len(configs),
        "holdout_object_count": len(features),
        "best_holdout_name": best["name"],
        "best_holdout_MV_AP_window": float(_num(best["holdout_MV_AP_window"])),
        "best_holdout_MV_AP50_window": float(_num(best["holdout_MV_AP50_window"])),
        "best_holdout_delta_vs_F2_base_window": float(_num(best["holdout_delta_vs_F2_base_window"])),
        "any_holdout_variant_passes_strict_gate": any_pass,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "holdout_metric_rows": _rel(OUT_DIR / "holdout_metric_rows.csv"),
            "holdout_metric_aggregate_rows": _rel(OUT_DIR / "holdout_metric_aggregate_rows.csv"),
            "paired_metric_rows": _rel(OUT_DIR / "paired_metric_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "holdout_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "holdout_metric_aggregate_rows.csv", agg)
    _write_csv(OUT_DIR / "paired_metric_rows.csv", paired)
    _write_csv(OUT_DIR / "holdout_case_rows.csv", case_rows)
    _write_csv(OUT_DIR / "holdout_top_iou_rows.csv", top_rows)
    _write_csv(OUT_DIR / "holdout_mv_object_frame_mask_rows.csv", all_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
