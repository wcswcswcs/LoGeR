#!/usr/bin/env python3
"""Post-final v99 Phase10G holdout projection for dense object quality.

This projects the Phase10F dev-positive dense/RADIO object-quality direction to
the same-scene temporal holdout. It is still diagnostic because this is run
after prior holdout feedback; a real claim would require a fresh frozen holdout.
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

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.frozen_feature_adapter import FrozenFeatureAdapter, locate_default_dinov2_checkpoint  # noqa: E402
from tools import build_v98_1_canonical_holdout_metrics as holdout  # noqa: E402
from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10g_dense_object_quality_holdout"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
HOLDOUT_SOURCE_ROWS = AUDIT_ROOT / "v98_phase13_holdout/source_container_rows.csv"
HOLDOUT_FIXED_ROWS = PHASE2_DIR / "holdout_mv_object_frame_mask_rows.csv"
HOLDOUT_RADIO_FEATURES = AUDIT_ROOT / "v98_phase13_holdout_radio_features_npz/mask_features.npz"
DEV_SEMANTIC_CONSTANTS = AUDIT_ROOT / "v98_phase6_semantic_residual_constants/semantic_constants.json"
SCANNET_ROOT = STREAM3D_ROOT / "data/scannet/processed"


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


def _load_holdout_radio_residuals() -> dict[tuple[str, int, int], np.ndarray]:
    constants = json.loads(DEV_SEMANTIC_CONSTANTS.read_text(encoding="utf-8"))
    mu = np.asarray(np.load(holdout._project(constants["radio_mu_vector_path"])), dtype=np.float32)
    payload = np.load(HOLDOUT_RADIO_FEATURES, allow_pickle=True)
    features = np.asarray(payload["features"], dtype=np.float32)
    residual = p1._normalize_rows(features - mu[None, :])
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for idx in range(residual.shape[0]):
        out[(str(payload["scene_id"][idx]), int(payload["frame_id"][idx]), int(payload["mask_id"][idx]))] = residual[idx]
    return out


def _extract_or_load_dino(rows: list[dict[str, Any]], scope: dict[str, Any], *, device: str, short_side: int) -> tuple[dict[tuple[str, int, int], np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    store = OUT_DIR / "dino_holdout_feature_store.npz"
    feature_rows_path = OUT_DIR / "dino_holdout_feature_rows.csv"
    if store.exists() and feature_rows_path.exists():
        payload = np.load(store, allow_pickle=True)
        residuals = np.asarray(payload["residuals"], dtype=np.float32)
        out: dict[tuple[str, int, int], np.ndarray] = {}
        for idx in range(residuals.shape[0]):
            out[(str(payload["scene_id"][idx]), int(payload["frame_id"][idx]), int(payload["mask_id"][idx]))] = residuals[idx]
        return out, _read_csv(feature_rows_path), {
            "feature_cache_hit": True,
            "dino_feature_count": len(out),
            "dinov2_checkpoint": str(payload["dinov2_checkpoint"].item()) if "dinov2_checkpoint" in payload.files else "",
        }

    checkpoint = locate_default_dinov2_checkpoint()
    if checkpoint is None:
        raise RuntimeError("no DINOv2 checkpoint found")
    adapter = FrozenFeatureAdapter(
        backend="dinov2_timm",
        device=device,
        checkpoint=checkpoint,
        short_side=short_side,
    )
    keys_by_frame: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        keys_by_frame[(str(row["scene_id"]), int(row["frame_id"]))].add(int(row["mask_id"]))
    streams: dict[str, ScanNetStream] = {}
    raw_features: dict[tuple[str, int, int], np.ndarray] = {}
    feature_rows: list[dict[str, Any]] = []
    missing_mask_count = 0
    empty_pool_count = 0
    shape_hist: dict[str, int] = defaultdict(int)
    for (scene, frame), mask_ids in sorted(keys_by_frame.items()):
        stream = streams.setdefault(scene, ScanNetStream(scene, root=SCANNET_ROOT))
        rgb = stream.load_rgb(int(frame))
        fmap = adapter.extract_dense_features(rgb)
        shape_hist[f"{fmap.features.shape[0]}x{fmap.features.shape[1]}x{fmap.features.shape[2]}"] += 1
        mask_path = scope["mask_path_by_frame"].get((scene, frame))
        if mask_path is None or not mask_path.exists():
            missing_mask_count += len(mask_ids)
            continue
        label = p1._read_label(mask_path)
        for mask_id in sorted(mask_ids):
            mask = label == int(mask_id)
            pooled = adapter.pool_mask_feature(fmap, mask)
            valid = bool(mask.any() and np.linalg.norm(pooled) > 1e-8)
            if not valid:
                empty_pool_count += 1
            if valid:
                raw_features[(scene, frame, int(mask_id))] = pooled.astype(np.float32)
            feature_rows.append(
                {
                    "schema_version": "stream4d_v99_phase10g_dino_holdout_feature_v1",
                    "phase_id": "v99_phase10g_dense_object_quality_holdout",
                    "scene_id": scene,
                    "frame_id": frame,
                    "mask_id": int(mask_id),
                    "feature_valid": valid,
                    "mask_area_px": int(np.count_nonzero(mask)),
                    "feature_dim": int(pooled.shape[0]) if pooled.ndim == 1 else 0,
                    "feature_norm": float(np.linalg.norm(pooled)),
                    "dense_grid_shape": f"{fmap.features.shape[0]}x{fmap.features.shape[1]}",
                    "patch_size": fmap.patch_size,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    keys = sorted(raw_features)
    mat = np.stack([raw_features[key] for key in keys]).astype(np.float32) if keys else np.zeros((0, 0), dtype=np.float32)
    if mat.size:
        mu = np.mean(mat, axis=0).astype(np.float32)
        residual_mat = p1._normalize_rows(mat - mu[None, :])
    else:
        residual_mat = np.zeros_like(mat)
    residuals = {key: residual_mat[idx] for idx, key in enumerate(keys)}
    np.savez_compressed(
        store,
        scene_id=np.asarray([key[0] for key in keys], dtype=object),
        frame_id=np.asarray([key[1] for key in keys], dtype=np.int32),
        mask_id=np.asarray([key[2] for key in keys], dtype=np.int32),
        residuals=residual_mat,
        dinov2_checkpoint=np.asarray(str(checkpoint), dtype=object),
    )
    _write_csv(feature_rows_path, feature_rows)
    return residuals, feature_rows, {
        "feature_cache_hit": False,
        "dino_feature_count": len(residuals),
        "missing_mask_count": missing_mask_count,
        "empty_pool_count": empty_pool_count,
        "dense_feature_shape_histogram": dict(shape_hist),
        "dinov2_checkpoint": checkpoint,
    }


def _semantic_consistency(rows: list[dict[str, Any]], residuals: dict[tuple[str, int, int], np.ndarray]) -> dict[str, dict[str, float]]:
    by_object: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = (str(row["scene_id"]), int(row["frame_id"]), int(row["mask_id"]))
        feat = residuals.get(key)
        if feat is not None:
            by_object[str(row["mv_object_id"])].append(feat)
    out: dict[str, dict[str, float]] = {}
    for oid in {str(row["mv_object_id"]) for row in rows}:
        feats = by_object.get(oid, [])
        if len(feats) < 2:
            out[oid] = {"coherence": 0.0, "valid_count": float(len(feats))}
            continue
        mat = np.stack(feats).astype(np.float32)
        centroid = p1._normalize_rows(np.mean(mat, axis=0, keepdims=True))[0]
        cos = [p1._cosine(row, centroid) for row in mat]
        out[oid] = {"coherence": float(np.mean(cos)), "valid_count": float(len(feats))}
    return out


def _features(rows: list[dict[str, Any]], radio_residuals: dict[tuple[str, int, int], np.ndarray], dino_residuals: dict[tuple[str, int, int], np.ndarray]) -> dict[str, dict[str, float]]:
    radio = _semantic_consistency(rows, radio_residuals)
    dino = _semantic_consistency(rows, dino_residuals)
    parent_max: dict[str, float] = {}
    support_area_mean: dict[str, float] = defaultdict(float)
    support_count: dict[str, int] = defaultdict(int)
    for row in rows:
        oid = str(row["mv_object_id"])
        parent_max[oid] = max(parent_max.get(oid, 0.0), _num(row.get("score")))
        support_area_mean[oid] += _num(row.get("support_area"))
        support_count[oid] += 1
    support_area = {oid: support_area_mean[oid] / max(1, support_count[oid]) for oid in support_count}
    parent_n = _norm(parent_max)
    radio_n = _norm({oid: vals["coherence"] for oid, vals in radio.items()})
    dino_n = _norm({oid: vals["coherence"] for oid, vals in dino.items()})
    support_n = _norm(support_area)
    out: dict[str, dict[str, float]] = {}
    for oid in sorted({str(row["mv_object_id"]) for row in rows}):
        out[oid] = {
            "parent_max": parent_max.get(oid, 0.0),
            "parent_norm": parent_n.get(oid, 0.0),
            "radio_coherence": radio.get(oid, {}).get("coherence", 0.0),
            "radio_norm": radio_n.get(oid, 0.0),
            "dino_coherence": dino.get(oid, {}).get("coherence", 0.0),
            "dino_norm": dino_n.get(oid, 0.0),
            "support_area_norm": support_n.get(oid, 0.0),
            "radio_valid_count": radio.get(oid, {}).get("valid_count", 0.0),
            "dino_valid_count": dino.get(oid, {}).get("valid_count", 0.0),
        }
    return out


def _score(config: dict[str, Any], f: dict[str, float]) -> float:
    sem = 0.5 * f["radio_norm"] + 0.5 * f["dino_norm"]
    mode = str(config["mode"])
    if mode == "parent":
        return f["parent_max"]
    if mode == "q4_strict_parent80_sem20":
        return 0.8 * f["parent_norm"] + 0.2 * sem
    if mode == "parent70_sem30":
        return 0.7 * f["parent_norm"] + 0.3 * sem
    if mode == "parent60_sem40":
        return 0.6 * f["parent_norm"] + 0.4 * sem
    if mode == "q5h_parent60_sem30_support_area10":
        return 0.6 * f["parent_norm"] + 0.3 * sem + 0.1 * f["support_area_norm"]
    raise ValueError(f"unknown mode {mode}")


def _make_rows(parent_rows: list[dict[str, Any]], features: dict[str, dict[str, float]], config: dict[str, Any]) -> list[dict[str, Any]]:
    variant = f"V99P10G_holdout_{config['name']}"
    out: list[dict[str, Any]] = []
    for row in parent_rows:
        oid = str(row["mv_object_id"])
        f = features[oid]
        new = dict(row)
        new["variant_id"] = variant
        new["variant"] = variant
        new["score"] = float(_score(config, f))
        new["score_policy"] = config["score_policy"]
        new["phase10g_parent_variant_id"] = row.get("variant_id", "")
        new["phase10g_mode"] = config["mode"]
        new["phase10g_schema_note"] = config["schema_note"]
        new["phase10g_radio_coherence"] = f["radio_coherence"]
        new["phase10g_dino_coherence"] = f["dino_coherence"]
        new["phase10g_support_area_norm"] = f["support_area_norm"]
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
    if not parent_rows:
        raise RuntimeError(f"missing holdout parent rows: {HOLDOUT_FIXED_ROWS}")
    device = os.environ.get("V99_DINO_DEVICE", "cuda:0")
    radio_residuals = _load_holdout_radio_residuals()
    dino_residuals, dino_feature_rows, dino_stats = _extract_or_load_dino(parent_rows, scope, device=device, short_side=518)
    feature_by_object = _features(parent_rows, radio_residuals, dino_residuals)
    configs = [
        {"name": "G0_parent", "mode": "parent", "score_policy": "phase10g_parent_score_replay", "schema_note": "same_as_phase2_holdout_parent"},
        {"name": "G1_Q4_strict_parent80_sem20", "mode": "q4_strict_parent80_sem20", "score_policy": "phase10g_0p80_parent_0p20_radio_dino_semantic", "schema_note": "strict_common_parent_plus_radio_dino"},
        {"name": "G2_parent70_sem30", "mode": "parent70_sem30", "score_policy": "phase10g_0p70_parent_0p30_radio_dino_semantic", "schema_note": "strict_common_parent_plus_radio_dino"},
        {"name": "G3_parent60_sem40", "mode": "parent60_sem40", "score_policy": "phase10g_0p60_parent_0p40_radio_dino_semantic", "schema_note": "strict_common_parent_plus_radio_dino"},
        {"name": "G4_Q5H_parent60_sem30_supportArea10", "mode": "q5h_parent60_sem30_support_area10", "score_policy": "phase10g_0p60_parent_0p30_semantic_0p10_holdout_support_area_proxy", "schema_note": "diagnostic_support_schema_mismatch_dev_used_support_surfel_count_holdout_uses_support_area"},
    ]
    config_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for config in configs:
        rows = _make_rows(parent_rows, feature_by_object, config)
        metrics, cases, tops = holdout._evaluate_variant(f"V99P10G_holdout_{config['name']}", rows, scope)
        all_rows.extend(rows)
        metric_rows.extend(metrics)
        case_rows.extend(cases)
        top_rows.extend(tops)
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase10g_variant_config_v1",
                "phase_id": "v99_phase10g_dense_object_quality_holdout",
                "name": config["name"],
                "mode": config["mode"],
                "score_policy": config["score_policy"],
                "schema_note": config["schema_note"],
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )
    agg = holdout._aggregate(metric_rows, family="v99_phase10g_dense_object_quality_holdout")
    f2_hold_window = float(phase0["F2_base_holdout_MV_AP_window"])
    f2_hold_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"])
    paired: list[dict[str, Any]] = []
    parent = next(row for row in agg if str(row["variant_id"]).endswith("G0_parent"))
    for row in agg:
        name = str(row["variant_id"]).replace("V99P10G_holdout_", "")
        gate = _num(row.get("mean_MV_AP_window")) >= f2_hold_window + 0.005 and _num(row.get("mean_MV_AP50_window")) >= f2_hold_ap50 + 0.010
        paired.append(
            {
                "schema_version": "stream4d_v99_phase10g_holdout_metric_v1",
                "phase_id": "v99_phase10g_dense_object_quality_holdout",
                "name": name,
                "holdout_variant_id": row["variant_id"],
                "holdout_MV_AP_window": row.get("mean_MV_AP_window"),
                "holdout_MV_AP50_window": row.get("mean_MV_AP50_window"),
                "holdout_MV_AP25_window": row.get("mean_MV_AP25_window"),
                "delta_vs_parent_window": _num(row.get("mean_MV_AP_window")) - _num(parent.get("mean_MV_AP_window")),
                "delta_vs_parent_AP50_window": _num(row.get("mean_MV_AP50_window")) - _num(parent.get("mean_MV_AP50_window")),
                "holdout_delta_vs_F2_base_window": _num(row.get("mean_MV_AP_window")) - f2_hold_window,
                "holdout_gate_pass": gate,
                "schema_note": next((cfg["schema_note"] for cfg in configs if cfg["name"] == name), ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )
    repair = [row for row in paired if row["name"] != "G0_parent"]
    best = max(repair, key=lambda row: (_num(row["holdout_MV_AP_window"]), _num(row["holdout_MV_AP50_window"])))
    any_pass = any(bool(row["holdout_gate_pass"]) for row in repair)
    strict_rows = [row for row in repair if "schema_mismatch" not in str(row.get("schema_note", ""))]
    best_strict = max(strict_rows, key=lambda row: (_num(row["holdout_MV_AP_window"]), _num(row["holdout_MV_AP50_window"]))) if strict_rows else {}
    gate_rows = [
        {
            "gate_id": "strict_common_dense_quality_holdout_gate",
            "pass": bool(best_strict) and bool(best_strict.get("holdout_gate_pass")),
            "expected": f"MV_AP_window>={f2_hold_window + 0.005} and MV_AP50_window>={f2_hold_ap50 + 0.010}",
            "observed": f"best_strict={best_strict.get('name')} MV_AP_window={best_strict.get('holdout_MV_AP_window')} MV_AP50_window={best_strict.get('holdout_MV_AP50_window')}",
            "severity": "method_gate",
        },
        {
            "gate_id": "any_dense_quality_holdout_diagnostic_gate",
            "pass": any_pass,
            "expected": f"any dense quality variant reaches strict holdout gate; schema-mismatch rows diagnostic only",
            "observed": f"best={best['name']} MV_AP_window={best['holdout_MV_AP_window']} MV_AP50_window={best['holdout_MV_AP50_window']}",
            "severity": "diagnostic",
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
            "repair_direction": "If strict dense-quality holdout fails, do not promote dense semantic; if only schema-mismatch support proxy helps, implement same-schema support first and require fresh holdout.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    feature_rows = [
        {
            "schema_version": "stream4d_v99_phase10g_object_feature_v1",
            "phase_id": "v99_phase10g_dense_object_quality_holdout",
            "mv_object_id": oid,
            **vals,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for oid, vals in sorted(feature_by_object.items())
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10g_dense_object_quality_holdout_summary_v1",
        "phase_id": "v99_phase10g_dense_object_quality_holdout",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "DIAGNOSTIC_DENSE_QUALITY_HOLDOUT_PASS_REQUIRES_FRESH_HOLDOUT" if any_pass else "NO_GO_DENSE_OBJECT_QUALITY_HOLDOUT",
        "formal_claim_allowed": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "variant_count": len(configs),
        "object_count": len(feature_by_object),
        "best_holdout_name": best["name"],
        "best_holdout_MV_AP_window": float(_num(best["holdout_MV_AP_window"])),
        "best_holdout_MV_AP50_window": float(_num(best["holdout_MV_AP50_window"])),
        "best_holdout_delta_vs_F2_base_window": float(_num(best["holdout_delta_vs_F2_base_window"])),
        "best_strict_name": best_strict.get("name", ""),
        "best_strict_MV_AP_window": float(_num(best_strict.get("holdout_MV_AP_window", 0.0))) if best_strict else 0.0,
        "best_strict_MV_AP50_window": float(_num(best_strict.get("holdout_MV_AP50_window", 0.0))) if best_strict else 0.0,
        "any_holdout_variant_passes_strict_gate": any_pass,
        "strict_common_variant_passes_holdout_gate": bool(best_strict) and bool(best_strict.get("holdout_gate_pass")),
        "dino_stats": dino_stats,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "paired_metric_rows": _rel(OUT_DIR / "paired_metric_rows.csv"),
            "holdout_metric_rows": _rel(OUT_DIR / "holdout_metric_rows.csv"),
            "holdout_metric_aggregate_rows": _rel(OUT_DIR / "holdout_metric_aggregate_rows.csv"),
            "object_feature_rows": _rel(OUT_DIR / "object_feature_rows.csv"),
            "dino_holdout_feature_rows": _rel(OUT_DIR / "dino_holdout_feature_rows.csv"),
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
