#!/usr/bin/env python3
"""Post-final v99 Phase10H same-schema support object-quality holdout test.

Phase10F found a dev-positive object-quality score using support_surfel_count.
Phase10G could only project it to holdout with support_area as a proxy. This
script repairs that schema mismatch by joining the holdout object-birth
object_frame_support_rows.csv back onto the fixed holdout rows, restoring the
same support_surfel_count field before rerunning holdout metrics.
"""

from __future__ import annotations

import csv
import json
import math
import os
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
from tools import build_v99_phase10g_dense_object_quality_holdout as p10g  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10h_same_schema_support_quality_holdout"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
HOLDOUT_SOURCE_ROWS = AUDIT_ROOT / "v98_phase13_holdout/source_container_rows.csv"
HOLDOUT_FIXED_ROWS = PHASE2_DIR / "holdout_mv_object_frame_mask_rows.csv"
HOLDOUT_SUPPORT_ROWS = AUDIT_ROOT / "v98_phase13_holdout_phase8_object_birth/object_frame_support_rows.csv"


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


def _join_support_surfel_count(parent_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    support_rows = _read_csv(HOLDOUT_SUPPORT_ROWS)
    support_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in support_rows:
        key = (
            str(row.get("object_id", "")),
            str(row.get("scene_id", "")),
            str(row.get("frame_id", "")),
            str(row.get("selected_mask_id_if_any", "")),
        )
        if key[0] and key[3]:
            support_by_key[key] = row
    out: list[dict[str, Any]] = []
    missing: list[tuple[str, str, str, str]] = []
    support_values: list[float] = []
    for row in parent_rows:
        key = (
            str(row.get("mv_object_id", "")),
            str(row.get("scene_id", "")),
            str(row.get("frame_id", "")),
            str(row.get("mask_id", "")),
        )
        support = support_by_key.get(key)
        new = dict(row)
        if support is None:
            missing.append(key)
            new["support_surfel_count"] = 0.0
            new["support_confidence"] = 0.0
            new["support_source_join_status"] = "missing"
        else:
            val = _num(support.get("support_surfel_count"))
            support_values.append(val)
            new["support_surfel_count"] = val
            new["support_confidence"] = _num(support.get("support_confidence"))
            new["support_source_join_status"] = "matched_object_frame_support_rows"
        out.append(new)
    diag = {
        "support_row_count": len(support_rows),
        "parent_row_count": len(parent_rows),
        "matched_parent_row_count": len(parent_rows) - len(missing),
        "missing_parent_row_count": len(missing),
        "support_surfel_count_min": float(min(support_values)) if support_values else 0.0,
        "support_surfel_count_max": float(max(support_values)) if support_values else 0.0,
        "support_surfel_count_mean": float(np.mean(support_values)) if support_values else 0.0,
        "missing_examples": missing[:10],
    }
    return out, diag


def _semantic_features(rows: list[dict[str, Any]], scope: dict[str, Any]) -> dict[str, dict[str, float]]:
    device = os.environ.get("V99_DINO_DEVICE", "cuda:0")
    radio_residuals = p10g._load_holdout_radio_residuals()
    dino_residuals, _dino_feature_rows, _dino_stats = p10g._extract_or_load_dino(rows, scope, device=device, short_side=518)
    return p10g._features(rows, radio_residuals, dino_residuals)


def _feature_table(rows: list[dict[str, Any]], semantic: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    support_sum: dict[str, float] = defaultdict(float)
    support_n: dict[str, int] = defaultdict(int)
    parent_max: dict[str, float] = {}
    for row in rows:
        oid = str(row["mv_object_id"])
        parent_max[oid] = max(parent_max.get(oid, 0.0), _num(row.get("score")))
        support_sum[oid] += _num(row.get("support_surfel_count"))
        support_n[oid] += 1
    support_mean = {oid: support_sum[oid] / max(1, support_n[oid]) for oid in support_n}
    support_norm = _norm(support_mean)
    parent_norm = _norm(parent_max)
    out: dict[str, dict[str, float]] = {}
    for oid in sorted({str(row["mv_object_id"]) for row in rows}):
        sem = semantic.get(oid, {})
        out[oid] = {
            "parent_max": parent_max.get(oid, 0.0),
            "parent_norm": parent_norm.get(oid, 0.0),
            "radio_norm": sem.get("radio_norm", 0.0),
            "dino_norm": sem.get("dino_norm", 0.0),
            "radio_coherence": sem.get("radio_coherence", 0.0),
            "dino_coherence": sem.get("dino_coherence", 0.0),
            "support_surfel_count_mean": support_mean.get(oid, 0.0),
            "support_surfel_count_norm": support_norm.get(oid, 0.0),
        }
    return out


def _score(config: dict[str, Any], f: dict[str, float]) -> float:
    sem = 0.5 * f["radio_norm"] + 0.5 * f["dino_norm"]
    mode = str(config["mode"])
    if mode == "parent":
        return f["parent_max"]
    if mode == "q5_exact_parent60_sem30_support10":
        return 0.6 * f["parent_norm"] + 0.3 * sem + 0.1 * f["support_surfel_count_norm"]
    if mode == "parent70_sem20_support10":
        return 0.7 * f["parent_norm"] + 0.2 * sem + 0.1 * f["support_surfel_count_norm"]
    if mode == "parent60_sem20_support20":
        return 0.6 * f["parent_norm"] + 0.2 * sem + 0.2 * f["support_surfel_count_norm"]
    if mode == "parent50_sem30_support20":
        return 0.5 * f["parent_norm"] + 0.3 * sem + 0.2 * f["support_surfel_count_norm"]
    raise ValueError(f"unknown mode {mode}")


def _make_rows(parent_rows: list[dict[str, Any]], features: dict[str, dict[str, float]], config: dict[str, Any]) -> list[dict[str, Any]]:
    variant = f"V99P10H_holdout_{config['name']}"
    out: list[dict[str, Any]] = []
    for row in parent_rows:
        oid = str(row["mv_object_id"])
        f = features[oid]
        new = dict(row)
        new["variant_id"] = variant
        new["variant"] = variant
        new["score"] = float(_score(config, f))
        new["score_policy"] = config["score_policy"]
        new["phase10h_parent_variant_id"] = row.get("variant_id", "")
        new["phase10h_mode"] = config["mode"]
        new["phase10h_support_surfel_count_mean"] = f["support_surfel_count_mean"]
        new["phase10h_support_surfel_count_norm"] = f["support_surfel_count_norm"]
        new["phase10h_radio_coherence"] = f["radio_coherence"]
        new["phase10h_dino_coherence"] = f["dino_coherence"]
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
    joined_rows, join_diag = _join_support_surfel_count(parent_rows)
    semantic = _semantic_features(joined_rows, scope)
    features = _feature_table(joined_rows, semantic)
    configs = [
        {"name": "H0_parent", "mode": "parent", "score_policy": "phase10h_parent_score_replay"},
        {"name": "H1_Q5_exact_parent60_sem30_support10", "mode": "q5_exact_parent60_sem30_support10", "score_policy": "phase10h_0p60_parent_0p30_semantic_0p10_support_surfel_count"},
        {"name": "H2_parent70_sem20_support10", "mode": "parent70_sem20_support10", "score_policy": "phase10h_0p70_parent_0p20_semantic_0p10_support_surfel_count"},
        {"name": "H3_parent60_sem20_support20", "mode": "parent60_sem20_support20", "score_policy": "phase10h_0p60_parent_0p20_semantic_0p20_support_surfel_count"},
        {"name": "H4_parent50_sem30_support20", "mode": "parent50_sem30_support20", "score_policy": "phase10h_0p50_parent_0p30_semantic_0p20_support_surfel_count"},
    ]
    all_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for config in configs:
        rows = _make_rows(joined_rows, features, config)
        metrics, cases, tops = holdout._evaluate_variant(f"V99P10H_holdout_{config['name']}", rows, scope)
        all_rows.extend(rows)
        metric_rows.extend(metrics)
        case_rows.extend(cases)
        top_rows.extend(tops)
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase10h_variant_config_v1",
                "phase_id": "v99_phase10h_same_schema_support_quality_holdout",
                "name": config["name"],
                "mode": config["mode"],
                "score_policy": config["score_policy"],
                "support_source": _rel(HOLDOUT_SUPPORT_ROWS),
                "support_schema": "support_surfel_count joined by object_id/scene/frame/mask",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )
    agg = holdout._aggregate(metric_rows, family="v99_phase10h_same_schema_support_quality_holdout")
    f2_hold_window = float(phase0["F2_base_holdout_MV_AP_window"])
    f2_hold_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"])
    parent = next(row for row in agg if str(row["variant_id"]).endswith("H0_parent"))
    paired: list[dict[str, Any]] = []
    for row in agg:
        name = str(row["variant_id"]).replace("V99P10H_holdout_", "")
        gate = _num(row.get("mean_MV_AP_window")) >= f2_hold_window + 0.005 and _num(row.get("mean_MV_AP50_window")) >= f2_hold_ap50 + 0.010
        paired.append(
            {
                "schema_version": "stream4d_v99_phase10h_holdout_metric_v1",
                "phase_id": "v99_phase10h_same_schema_support_quality_holdout",
                "name": name,
                "holdout_variant_id": row["variant_id"],
                "holdout_MV_AP_window": row.get("mean_MV_AP_window"),
                "holdout_MV_AP50_window": row.get("mean_MV_AP50_window"),
                "holdout_MV_AP25_window": row.get("mean_MV_AP25_window"),
                "delta_vs_parent_window": _num(row.get("mean_MV_AP_window")) - _num(parent.get("mean_MV_AP_window")),
                "delta_vs_parent_AP50_window": _num(row.get("mean_MV_AP50_window")) - _num(parent.get("mean_MV_AP50_window")),
                "holdout_delta_vs_F2_base_window": _num(row.get("mean_MV_AP_window")) - f2_hold_window,
                "holdout_gate_pass": gate,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )
    repair = [row for row in paired if row["name"] != "H0_parent"]
    best = max(repair, key=lambda row: (_num(row["holdout_MV_AP_window"]), _num(row["holdout_MV_AP50_window"])))
    any_pass = any(bool(row["holdout_gate_pass"]) for row in repair)
    gate_rows = [
        {
            "gate_id": "same_schema_support_join_complete",
            "pass": int(join_diag["missing_parent_row_count"]) == 0,
            "expected": "0 missing fixed holdout rows after support_surfel_count join",
            "observed": join_diag["missing_parent_row_count"],
            "severity": "required_data_contract",
        },
        {
            "gate_id": "same_schema_support_quality_holdout_gate",
            "pass": any_pass,
            "expected": f"MV_AP_window>={f2_hold_window + 0.005} and MV_AP50_window>={f2_hold_ap50 + 0.010}",
            "observed": f"best={best['name']} MV_AP_window={best['holdout_MV_AP_window']} MV_AP50_window={best['holdout_MV_AP50_window']}",
            "severity": "method_gate",
        },
        {
            "gate_id": "formal_claim_allowed_after_post_final_diagnostic",
            "pass": False,
            "expected": "fresh frozen holdout",
            "observed": "post-final diagnostic after prior holdout feedback",
            "severity": "formal_claim_blocker",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If same-schema support quality still fails holdout, close v99 as KEEP_F2_AS_MAIN_METHOD unless a new object-birth plan is opened.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    feature_rows = [
        {
            "schema_version": "stream4d_v99_phase10h_object_feature_v1",
            "phase_id": "v99_phase10h_same_schema_support_quality_holdout",
            "mv_object_id": oid,
            **vals,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for oid, vals in sorted(features.items())
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10h_same_schema_support_quality_holdout_summary_v1",
        "phase_id": "v99_phase10h_same_schema_support_quality_holdout",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "DIAGNOSTIC_SAME_SCHEMA_SUPPORT_HOLDOUT_PASS_REQUIRES_FRESH_HOLDOUT" if any_pass else "NO_GO_SAME_SCHEMA_SUPPORT_QUALITY_HOLDOUT",
        "formal_claim_allowed": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "variant_count": len(configs),
        "object_count": len(features),
        "join_diag": join_diag,
        "best_holdout_name": best["name"],
        "best_holdout_MV_AP_window": float(_num(best["holdout_MV_AP_window"])),
        "best_holdout_MV_AP50_window": float(_num(best["holdout_MV_AP50_window"])),
        "best_holdout_delta_vs_F2_base_window": float(_num(best["holdout_delta_vs_F2_base_window"])),
        "any_holdout_variant_passes_strict_gate": any_pass,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "paired_metric_rows": _rel(OUT_DIR / "paired_metric_rows.csv"),
            "holdout_metric_rows": _rel(OUT_DIR / "holdout_metric_rows.csv"),
            "holdout_metric_aggregate_rows": _rel(OUT_DIR / "holdout_metric_aggregate_rows.csv"),
            "object_feature_rows": _rel(OUT_DIR / "object_feature_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "paired_metric_rows.csv", paired)
    _write_csv(OUT_DIR / "holdout_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "holdout_metric_aggregate_rows.csv", agg)
    _write_csv(OUT_DIR / "holdout_case_rows.csv", case_rows)
    _write_csv(OUT_DIR / "holdout_top_iou_rows.csv", top_rows)
    _write_csv(OUT_DIR / "holdout_mv_object_frame_mask_rows.csv", all_rows)
    _write_csv(OUT_DIR / "object_feature_rows.csv", feature_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
