#!/usr/bin/env python3
"""Post-final v99 Phase10F dense/RADIO object-quality dev diagnostic.

This tests the v99 Phase17.5 repair direction without another tiny eps sweep:
use object-level semantic coherence as the ranking score. It is diagnostic
because it is run after the Phase10 holdout decision; if a variant passed dev it
would still require a fresh frozen holdout before any formal claim.
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

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10f_dense_object_quality_dev"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE6_DIR = AUDIT_ROOT / "v99_phase6_dense_semantic_residual"


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


def _phase2_parent_rows() -> tuple[str, list[dict[str, Any]]]:
    summary = json.loads((PHASE2_DIR / "best_variant_summary.json").read_text(encoding="utf-8"))
    variant = str(summary["best_variant_id"])
    rows = [dict(row) for row in _read_csv(PHASE2_DIR / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == variant]
    if not rows:
        raise RuntimeError(f"missing Phase2 rows for {variant}")
    return variant, rows


def _load_dino_residuals() -> dict[tuple[str, int, int], np.ndarray]:
    path = PHASE6_DIR / "dino_dense_feature_store.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=True)
    residuals = np.asarray(payload["residuals"], dtype=np.float32)
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for idx in range(residuals.shape[0]):
        out[(str(payload["scene_id"][idx]), int(payload["frame_id"][idx]), int(payload["mask_id"][idx]))] = residuals[idx]
    return out


def _semantic_consistency(rows: list[dict[str, Any]], residuals: dict[tuple[str, int, int], np.ndarray]) -> dict[str, dict[str, float]]:
    by_object: dict[str, list[np.ndarray]] = defaultdict(list)
    by_object_frames: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        oid = str(row["mv_object_id"])
        key = (str(row["scene_id"]), int(row["frame_id"]), int(row["selected_mask_id"]))
        feat = residuals.get(key)
        if feat is None:
            continue
        by_object[oid].append(feat)
        by_object_frames[oid].add((str(row["scene_id"]), int(row["frame_id"])))
    out: dict[str, dict[str, float]] = {}
    for oid in {str(row["mv_object_id"]) for row in rows}:
        feats = by_object.get(oid, [])
        if len(feats) < 2:
            out[oid] = {"coherence": 0.0, "spread_penalty": 1.0, "valid_count": float(len(feats)), "frame_count": float(len(by_object_frames.get(oid, set())))}
            continue
        mat = np.stack(feats).astype(np.float32)
        centroid = p1._normalize_rows(np.mean(mat, axis=0, keepdims=True))[0]
        cos = [p1._cosine(row, centroid) for row in mat]
        out[oid] = {
            "coherence": float(np.mean(cos)),
            "spread_penalty": float(np.std(cos)),
            "valid_count": float(len(feats)),
            "frame_count": float(len(by_object_frames.get(oid, set()))),
        }
    return out


def _object_area_features(rows: list[dict[str, Any]], scope: dict[str, Any]) -> dict[str, dict[str, float]]:
    area_by_key: dict[tuple[str, int, int], float] = {}
    for row in scope["source_rows"]:
        scene = str(row.get("scene_id", ""))
        frame = int(_num(row.get("frame_id"), -1))
        mask_id = int(_num(row.get("source_mask_id"), -1))
        if scene and frame >= 0 and mask_id > 0:
            area_by_key[(scene, frame, mask_id)] = _num(row.get("mask_area_ratio"))
    by_object: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["scene_id"]), int(row["frame_id"]), int(row["selected_mask_id"]))
        by_object[str(row["mv_object_id"])].append(area_by_key.get(key, 0.0))
    out: dict[str, dict[str, float]] = {}
    for oid in {str(row["mv_object_id"]) for row in rows}:
        vals = by_object.get(oid, [])
        out[oid] = {
            "area_mean": float(np.mean(vals)) if vals else 0.0,
            "area_max": float(np.max(vals)) if vals else 0.0,
            "area_std": float(np.std(vals)) if vals else 0.0,
        }
    return out


def _features(rows: list[dict[str, Any]], scope: dict[str, Any]) -> dict[str, dict[str, float]]:
    radio_residuals, _radio_tau = p1._load_radio_residual_features()
    dino_residuals = _load_dino_residuals()
    radio = _semantic_consistency(rows, radio_residuals)
    dino = _semantic_consistency(rows, dino_residuals)
    area = _object_area_features(rows, scope)
    parent_max: dict[str, float] = {}
    support_mean: dict[str, float] = defaultdict(float)
    support_count: dict[str, int] = defaultdict(int)
    for row in rows:
        oid = str(row["mv_object_id"])
        parent_max[oid] = max(parent_max.get(oid, 0.0), _num(row.get("score")))
        support_mean[oid] += _num(row.get("support_surfel_count"))
        support_count[oid] += 1
    support = {oid: support_mean[oid] / max(1, support_count[oid]) for oid in support_count}
    parent_n = _norm(parent_max)
    support_n = _norm(support)
    radio_n = _norm({oid: vals["coherence"] for oid, vals in radio.items()})
    dino_n = _norm({oid: vals["coherence"] for oid, vals in dino.items()})
    radio_spread_n = _norm({oid: vals["spread_penalty"] for oid, vals in radio.items()})
    dino_spread_n = _norm({oid: vals["spread_penalty"] for oid, vals in dino.items()})
    area_mean_n = _norm({oid: vals["area_mean"] for oid, vals in area.items()})
    out: dict[str, dict[str, float]] = {}
    for oid in sorted({str(row["mv_object_id"]) for row in rows}):
        out[oid] = {
            "parent_max": parent_max.get(oid, 0.0),
            "parent_norm": parent_n.get(oid, 0.0),
            "support_norm": support_n.get(oid, 0.0),
            "radio_coherence": radio.get(oid, {}).get("coherence", 0.0),
            "radio_norm": radio_n.get(oid, 0.0),
            "radio_spread_norm": radio_spread_n.get(oid, 0.0),
            "dino_coherence": dino.get(oid, {}).get("coherence", 0.0),
            "dino_norm": dino_n.get(oid, 0.0),
            "dino_spread_norm": dino_spread_n.get(oid, 0.0),
            "area_mean_norm": area_mean_n.get(oid, 0.0),
            "radio_valid_count": radio.get(oid, {}).get("valid_count", 0.0),
            "dino_valid_count": dino.get(oid, {}).get("valid_count", 0.0),
        }
    return out


def _score(config: dict[str, Any], f: dict[str, float]) -> float:
    mode = str(config["mode"])
    if mode == "parent":
        return f["parent_max"]
    if mode == "radio_only":
        return f["radio_norm"]
    if mode == "dino_only":
        return f["dino_norm"]
    if mode == "radio_dino_mean":
        return 0.5 * f["radio_norm"] + 0.5 * f["dino_norm"]
    if mode == "parent80_sem20":
        sem = 0.5 * f["radio_norm"] + 0.5 * f["dino_norm"]
        return 0.8 * f["parent_norm"] + 0.2 * sem
    if mode == "parent60_sem30_support10":
        sem = 0.5 * f["radio_norm"] + 0.5 * f["dino_norm"]
        return 0.6 * f["parent_norm"] + 0.3 * sem + 0.1 * f["support_norm"]
    raise ValueError(f"unknown score mode {mode}")


def _make_rows(parent_rows: list[dict[str, Any]], features: dict[str, dict[str, float]], config: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    variant = f"V99P10F_{config['name']}"
    for row in parent_rows:
        oid = str(row["mv_object_id"])
        f = features[oid]
        new = dict(row)
        new["variant_id"] = variant
        new["variant"] = variant
        new["score"] = float(_score(config, f))
        new["score_policy"] = config["score_policy"]
        new["phase10f_parent_variant_id"] = row.get("variant_id", "")
        new["phase10f_mode"] = config["mode"]
        new["phase10f_radio_coherence"] = f["radio_coherence"]
        new["phase10f_dino_coherence"] = f["dino_coherence"]
        new["phase10f_support_norm"] = f["support_norm"]
        new["phase10f_area_mean_norm"] = f["area_mean_norm"]
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    parent_variant, parent_rows = _phase2_parent_rows()
    scope = p1._load_source_scope()
    feature_by_object = _features(parent_rows, scope)
    configs = [
        {"name": "Q0_parent", "mode": "parent", "score_policy": "phase10f_parent_score_replay"},
        {"name": "Q1_radio_only", "mode": "radio_only", "score_policy": "phase10f_radio_object_coherence_only"},
        {"name": "Q2_dino_only", "mode": "dino_only", "score_policy": "phase10f_dino_object_coherence_only"},
        {"name": "Q3_radio_dino_mean", "mode": "radio_dino_mean", "score_policy": "phase10f_radio_dino_object_coherence_mean"},
        {"name": "Q4_parent80_sem20", "mode": "parent80_sem20", "score_policy": "phase10f_0p80_parent_0p20_radio_dino_semantic"},
        {"name": "Q5_parent60_sem30_support10", "mode": "parent60_sem30_support10", "score_policy": "phase10f_0p60_parent_0p30_semantic_0p10_support"},
    ]
    config_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    metric_scene_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for config in configs:
        rows = _make_rows(parent_rows, feature_by_object, config)
        metrics, frames = p1._evaluate_variant(f"V99P10F_{config['name']}", rows, scope)
        all_rows.extend(rows)
        metric_scene_rows.extend(metrics)
        frame_rows.extend(frames)
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase10f_variant_config_v1",
                "phase_id": "v99_phase10f_dense_object_quality_dev",
                "name": config["name"],
                "mode": config["mode"],
                "score_policy": config["score_policy"],
                "parent_variant_id": parent_variant,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )
    aggregate_rows = p1._aggregate_metrics(metric_scene_rows)
    by_name = {str(row["variant_id"]).replace("V99P10F_", ""): row for row in aggregate_rows}
    parent = by_name["Q0_parent"]
    f2_window = float(phase0["F2_base_full_dev_MV_AP_window"])
    f2_ap50 = float(phase0["F2_base_full_dev_MV_AP50_window"])
    paired_rows: list[dict[str, Any]] = []
    for config in configs:
        name = str(config["name"])
        row = by_name[name]
        delta_parent = _num(row.get("MV_AP_window")) - _num(parent.get("MV_AP_window"))
        delta_parent_ap50 = _num(row.get("MV_AP50_window")) - _num(parent.get("MV_AP50_window"))
        local_gate = _num(row.get("MV_AP_window")) >= f2_window + 0.005 and _num(row.get("MV_AP50_window")) >= f2_ap50 + 0.010
        min_progress = delta_parent >= 0.002 or delta_parent_ap50 >= 0.004
        paired_rows.append(
            {
                "schema_version": "stream4d_v99_phase10f_paired_metric_v1",
                "phase_id": "v99_phase10f_dense_object_quality_dev",
                "name": name,
                "variant_id": row["variant_id"],
                "MV_AP_window": row.get("MV_AP_window"),
                "MV_AP50_window": row.get("MV_AP50_window"),
                "MV_AP_scene": row.get("MV_AP_scene"),
                "delta_vs_parent_window": delta_parent,
                "delta_vs_parent_AP50_window": delta_parent_ap50,
                "delta_vs_F2_base_window": _num(row.get("MV_AP_window")) - f2_window,
                "dev_local_gate_pass": local_gate,
                "min_progress_gate_pass": min_progress,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )
    repair_rows = [row for row in paired_rows if row["name"] != "Q0_parent"]
    best = max(repair_rows, key=lambda row: (_num(row["MV_AP_window"]), _num(row["MV_AP50_window"])))
    any_progress = any(bool(row["min_progress_gate_pass"]) for row in repair_rows)
    any_local_gate = any(bool(row["dev_local_gate_pass"]) for row in repair_rows)
    gate_rows = [
        {
            "gate_id": "dense_object_quality_any_repair_min_progress",
            "pass": any_progress,
            "expected": "some repair improves parent by >=0.002 MV_AP_window or >=0.004 MV_AP50_window",
            "observed": f"best={best['name']} delta_window={best['delta_vs_parent_window']} delta_AP50={best['delta_vs_parent_AP50_window']}",
            "severity": "repair_family_progress",
        },
        {
            "gate_id": "dense_object_quality_any_repair_dev_local_gate",
            "pass": any_local_gate,
            "expected": f"MV_AP_window>={f2_window + 0.005} and MV_AP50_window>={f2_ap50 + 0.010}",
            "observed": f"best={best['name']} MV_AP_window={best['MV_AP_window']} MV_AP50_window={best['MV_AP50_window']}",
            "severity": "method_gate",
        },
        {
            "gate_id": "formal_claim_allowed_after_post_final_diagnostic",
            "pass": False,
            "expected": "fresh frozen holdout after dev-only freeze",
            "observed": "post-final diagnostic after previous holdout feedback",
            "severity": "formal_claim_blocker",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If no min-progress, stop dense object quality family per v99 Phase17.5 and keep mask-level proxy.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    feature_rows = [
        {
            "schema_version": "stream4d_v99_phase10f_object_feature_v1",
            "phase_id": "v99_phase10f_dense_object_quality_dev",
            "mv_object_id": oid,
            **vals,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for oid, vals in sorted(feature_by_object.items())
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10f_dense_object_quality_dev_summary_v1",
        "phase_id": "v99_phase10f_dense_object_quality_dev",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "DEV_CANDIDATE_REQUIRES_FRESH_HOLDOUT" if any_local_gate else ("DENSE_OBJECT_QUALITY_HAS_MIN_PROGRESS_NO_METHOD_GATE" if any_progress else "NO_GO_DENSE_OBJECT_QUALITY_DEV"),
        "formal_claim_allowed": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "parent_variant_id": parent_variant,
        "variant_count": len(configs),
        "object_count": len(feature_by_object),
        "best_repair_name": best["name"],
        "best_repair_MV_AP_window": float(_num(best["MV_AP_window"])),
        "best_repair_MV_AP50_window": float(_num(best["MV_AP50_window"])),
        "best_repair_delta_vs_parent_window": float(_num(best["delta_vs_parent_window"])),
        "best_repair_delta_vs_parent_AP50_window": float(_num(best["delta_vs_parent_AP50_window"])),
        "any_repair_min_progress": any_progress,
        "any_repair_dev_local_gate_pass": any_local_gate,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "paired_metric_rows": _rel(OUT_DIR / "paired_metric_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "object_feature_rows": _rel(OUT_DIR / "object_feature_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "paired_metric_rows.csv", paired_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", metric_scene_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", all_rows)
    _write_csv(OUT_DIR / "object_feature_rows.csv", feature_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if any_local_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
