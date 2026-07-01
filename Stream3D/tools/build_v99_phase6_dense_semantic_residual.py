#!/usr/bin/env python3
"""Evaluate dense DINO semantic residuals on top of the Phase2 F2 rows."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.frozen_feature_adapter import (  # noqa: E402
    FrozenFeatureAdapter,
    locate_default_dinov2_checkpoint,
    locate_default_radio_checkpoint,
)
from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase6_dense_semantic_residual"
PHASE0_SUMMARY = AUDIT_ROOT / "v99_phase0_fact_lock/summary.json"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE2_SUMMARY = PHASE2_DIR / "best_variant_summary.json"
AVAILABILITY = AUDIT_ROOT / "v99_phase6_dense_semantic_availability_loger_gpu6/radio_vipe_availability.json"
RADIO_MASK_FEATURES = AUDIT_ROOT / "v91_radio_mask_features_npz/mask_features.npz"
RADIO_CONSTANTS = AUDIT_ROOT / "v98_phase6_semantic_residual_constants/semantic_constants.json"
SCANNET_ROOT = STREAM3D_ROOT / "data/scannet/processed"
RANDOM_SEED = 9906
EPS = 1e-4


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


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, 1e-12)).astype(np.float32, copy=False)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _safe_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _phase2_best_rows() -> tuple[str, list[dict[str, Any]]]:
    summary = json.loads(PHASE2_SUMMARY.read_text(encoding="utf-8"))
    variant = str(summary["best_variant_id"])
    rows = [dict(row) for row in _read_csv(PHASE2_DIR / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == variant]
    if not rows:
        raise RuntimeError(f"no rows for Phase2 best variant {variant}")
    return variant, rows


def _load_radio_residuals() -> dict[tuple[str, int, int], np.ndarray]:
    if not RADIO_MASK_FEATURES.exists() or not RADIO_CONSTANTS.exists():
        return {}
    constants = json.loads(RADIO_CONSTANTS.read_text(encoding="utf-8"))
    mu_path = p1._project(constants.get("radio_mu_vector_path", ""))
    if not mu_path.exists():
        return {}
    mu = np.asarray(np.load(mu_path), dtype=np.float32)
    payload = np.load(RADIO_MASK_FEATURES, allow_pickle=True)
    feats = np.asarray(payload["features"], dtype=np.float32)
    residual = _normalize_rows(feats - mu[None, :])
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for idx in range(residual.shape[0]):
        out[(str(payload["scene_id"][idx]), int(payload["frame_id"][idx]), int(payload["mask_id"][idx]))] = residual[idx]
    return out


def _extract_dense_dino_features(
    parent_rows: list[dict[str, Any]],
    scope: dict[str, Any],
    *,
    device: str,
    short_side: int,
) -> tuple[dict[tuple[str, int, int], np.ndarray], list[dict[str, Any]], dict[str, Any]]:
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
    for row in parent_rows:
        keys_by_frame[(str(row["scene_id"]), int(row["frame_id"]))].add(int(row["selected_mask_id"]))

    streams: dict[str, ScanNetStream] = {}
    feature_rows: list[dict[str, Any]] = []
    features: dict[tuple[str, int, int], np.ndarray] = {}
    frame_feature_shape: dict[str, int] = defaultdict(int)
    missing_mask_count = 0
    empty_pool_count = 0
    for (scene, frame), mask_ids in sorted(keys_by_frame.items()):
        stream = streams.setdefault(scene, ScanNetStream(scene, root=SCANNET_ROOT))
        rgb = stream.load_rgb(int(frame))
        fmap = adapter.extract_dense_features(rgb)
        frame_feature_shape[f"{fmap.features.shape[0]}x{fmap.features.shape[1]}x{fmap.features.shape[2]}"] += 1
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
                features[(scene, frame, int(mask_id))] = pooled.astype(np.float32)
            feature_rows.append(
                {
                    "schema_version": "stream4d_v99_phase6_dense_feature_row_v1",
                    "phase_id": "v99_phase6_dense_semantic_residual",
                    "feature_provider": "DINOv2_timm_dense_patch_pool",
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
    stats = {
        "selected_mask_observation_count": len({(r["scene_id"], int(r["frame_id"]), int(r["selected_mask_id"])) for r in parent_rows}),
        "dense_feature_valid_count": len(features),
        "dense_feature_valid_rate": float(len(features) / max(1, len({(r["scene_id"], int(r["frame_id"]), int(r["selected_mask_id"])) for r in parent_rows}))),
        "missing_mask_count": missing_mask_count,
        "empty_pool_count": empty_pool_count,
        "dense_feature_shape_histogram": dict(frame_feature_shape),
        "dinov2_checkpoint": checkpoint,
        "dinov2_short_side": short_side,
    }
    return features, feature_rows, stats


def _residualize(features: dict[tuple[str, int, int], np.ndarray]) -> tuple[dict[tuple[str, int, int], np.ndarray], np.ndarray, float]:
    keys = sorted(features)
    mat = np.stack([features[key] for key in keys]).astype(np.float32) if keys else np.zeros((0, 0), dtype=np.float32)
    if mat.size == 0:
        return {}, np.zeros((0,), dtype=np.float32), 0.0
    mu = np.mean(mat, axis=0).astype(np.float32)
    residual = _normalize_rows(mat - mu[None, :])
    rng = np.random.default_rng(RANDOM_SEED)
    if residual.shape[0] >= 2:
        pairs = rng.integers(0, residual.shape[0], size=(min(20000, residual.shape[0] * 4), 2))
        vals = np.asarray([_cos(residual[int(a)], residual[int(b)]) for a, b in pairs if int(a) != int(b)], dtype=np.float32)
        tau = float(np.quantile(vals, 0.75)) if vals.size else 0.0
    else:
        tau = 0.0
    return {key: residual[idx] for idx, key in enumerate(keys)}, mu, tau


def _semantic_score(cosine: float, tau: float) -> float:
    return float(max(0.0, min(1.0, (cosine - tau) / max(1e-6, 1.0 - tau))))


def _pair_distribution(
    parent_rows: list[dict[str, Any]],
    dense_residuals: dict[tuple[str, int, int], np.ndarray],
    radio_residuals: dict[tuple[str, int, int], np.ndarray],
    tau: float,
) -> tuple[list[dict[str, Any]], float]:
    rng = np.random.default_rng(RANDOM_SEED)
    keys = sorted(dense_residuals)
    pair_groups: dict[str, list[tuple[tuple[str, int, int], tuple[str, int, int]]]] = defaultdict(list)
    if len(keys) >= 2:
        for a, b in rng.integers(0, len(keys), size=(min(30000, len(keys) * 4), 2)):
            if int(a) != int(b):
                pair_groups["random_pair"].append((keys[int(a)], keys[int(b)]))
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in parent_rows:
        by_object[str(row["mv_object_id"])].append(row)
        by_frame[(str(row["scene_id"]), int(row["frame_id"]))].append(row)
    for vals in by_object.values():
        ordered = sorted(vals, key=lambda r: (str(r["scene_id"]), str(r["chunk_id"]), int(r["frame_id"]), int(r["selected_mask_id"])))
        for a, b in zip(ordered[:-1], ordered[1:]):
            if a["scene_id"] == b["scene_id"] and int(a["frame_id"]) != int(b["frame_id"]):
                ka = (str(a["scene_id"]), int(a["frame_id"]), int(a["selected_mask_id"]))
                kb = (str(b["scene_id"]), int(b["frame_id"]), int(b["selected_mask_id"]))
                if ka in dense_residuals and kb in dense_residuals:
                    pair_groups["same_source_pair"].append((ka, kb))
    for (_scene, _frame), vals in by_frame.items():
        ordered = sorted(vals, key=lambda r: int(r["selected_mask_id"]))
        for a, b in zip(ordered[:-1], ordered[1:]):
            ka = (str(a["scene_id"]), int(a["frame_id"]), int(a["selected_mask_id"]))
            kb = (str(b["scene_id"]), int(b["frame_id"]), int(b["selected_mask_id"]))
            if ka in dense_residuals and kb in dense_residuals:
                pair_groups["cross_boundary_pair"].append((ka, kb))

    rows: list[dict[str, Any]] = []
    agreement_dense: list[float] = []
    agreement_radio: list[float] = []
    for pair_type, pairs in sorted(pair_groups.items()):
        if len(pairs) > 20000:
            take = rng.choice(len(pairs), size=20000, replace=False)
            pairs = [pairs[int(i)] for i in take]
        dense_vals: list[float] = []
        radio_vals: list[float] = []
        both = 0
        for ka, kb in pairs:
            dcos = _cos(dense_residuals[ka], dense_residuals[kb])
            dense_vals.append(dcos)
            if ka in radio_residuals and kb in radio_residuals:
                rcos = _cos(radio_residuals[ka], radio_residuals[kb])
                radio_vals.append(rcos)
                agreement_dense.append(dcos)
                agreement_radio.append(rcos)
                both += 1
        arr = np.asarray(dense_vals, dtype=np.float32)
        rarr = np.asarray(radio_vals, dtype=np.float32)
        rows.append(
            {
                "schema_version": "stream4d_v99_phase6_semantic_distribution_v1",
                "phase_id": "v99_phase6_dense_semantic_residual",
                "feature_provider": "DINOv2_timm_dense_patch_pool",
                "pair_type": pair_type,
                "sample_count": int(arr.shape[0]),
                "random_pair_cos_mean": float(np.mean(arr)) if arr.size else "",
                "random_pair_cos_p75": float(np.quantile(arr, 0.75)) if arr.size else "",
                "same_source_pair_cos_mean": float(np.mean(arr)) if arr.size and pair_type == "same_source_pair" else "",
                "cross_boundary_pair_cos_mean": float(np.mean(arr)) if arr.size and pair_type == "cross_boundary_pair" else "",
                "cos_p50": float(np.quantile(arr, 0.50)) if arr.size else "",
                "cos_p90": float(np.quantile(arr, 0.90)) if arr.size else "",
                "semantic_residual_margin_mean": float(np.mean([_semantic_score(v, tau) for v in dense_vals])) if dense_vals else "",
                "radio_pair_count": both,
                "radio_cos_mean": float(np.mean(rarr)) if rarr.size else "",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows, _safe_corr(agreement_dense, agreement_radio)


def _object_semantic_metrics(
    parent_rows: list[dict[str, Any]],
    residuals: dict[tuple[str, int, int], np.ndarray],
    tau: float,
) -> dict[str, dict[str, float]]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parent_rows:
        by_object[str(row["mv_object_id"])].append(row)
    out: dict[str, dict[str, float]] = {}
    for oid, vals in by_object.items():
        ordered = sorted(vals, key=lambda r: (str(r["scene_id"]), str(r["chunk_id"]), int(r["frame_id"]), int(r["selected_mask_id"])))
        scores: list[float] = []
        conflicts = 0
        valid_links = 0
        for a, b in zip(ordered[:-1], ordered[1:]):
            if a["scene_id"] != b["scene_id"] or a["chunk_id"] != b["chunk_id"] or int(a["frame_id"]) == int(b["frame_id"]):
                continue
            ka = (str(a["scene_id"]), int(a["frame_id"]), int(a["selected_mask_id"]))
            kb = (str(b["scene_id"]), int(b["frame_id"]), int(b["selected_mask_id"]))
            if ka not in residuals or kb not in residuals:
                continue
            valid_links += 1
            score = _semantic_score(_cos(residuals[ka], residuals[kb]), tau)
            scores.append(score)
            if score <= 0.02:
                conflicts += 1
        out[oid] = {
            "dense_semantic_score_mean": float(np.mean(scores)) if scores else 0.0,
            "dense_semantic_valid_link_count": float(valid_links),
            "dense_semantic_conflict_count": float(conflicts),
        }
    return out


def _norm(values: dict[str, float]) -> dict[str, float]:
    vals = list(values.values())
    if not vals:
        return {}
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def _shuffled_residuals(residuals: dict[tuple[str, int, int], np.ndarray]) -> dict[tuple[str, int, int], np.ndarray]:
    keys = sorted(residuals)
    values = [residuals[key] for key in keys]
    rnd = random.Random(RANDOM_SEED)
    shuffled = values[:]
    rnd.shuffle(shuffled)
    return {key: shuffled[idx] for idx, key in enumerate(keys)}


def _variant_rows(parent_rows: list[dict[str, Any]], real_metrics: dict[str, dict[str, float]], shuffled_metrics: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    real_score_norm = _norm({oid: vals["dense_semantic_score_mean"] for oid, vals in real_metrics.items()})
    real_conflict_norm = _norm({oid: vals["dense_semantic_conflict_count"] for oid, vals in real_metrics.items()})
    shuf_score_norm = _norm({oid: vals["dense_semantic_score_mean"] for oid, vals in shuffled_metrics.items()})
    shuf_conflict_norm = _norm({oid: vals["dense_semantic_conflict_count"] for oid, vals in shuffled_metrics.items()})
    variants = {
        "P6_B0_phase2_best_no_dense_semantic": ("phase2_score_replay", real_metrics, {}, {}, "replay"),
        "P6_D1_dense_dino_boost": ("phase2_score_plus_1e-4_dense_dino_semantic", real_metrics, real_score_norm, real_conflict_norm, "boost"),
        "P6_D2_dense_dino_veto": ("phase2_score_minus_1e-4_dense_dino_conflict", real_metrics, real_score_norm, real_conflict_norm, "veto"),
        "P6_D3_dense_dino_boost_plus_veto": ("phase2_score_plus_dense_dino_minus_conflict", real_metrics, real_score_norm, real_conflict_norm, "boost_veto"),
        "P6_D4_dense_dino_boost_eps3e-4": ("phase2_score_plus_3e-4_dense_dino_semantic", real_metrics, real_score_norm, real_conflict_norm, "boost3"),
        "P6_C1_shuffled_dense_dino_boost_plus_veto": ("control_shuffled_dense_dino_boost_plus_veto", shuffled_metrics, shuf_score_norm, shuf_conflict_norm, "boost_veto"),
    }
    out: list[dict[str, Any]] = []
    for variant, (policy, metrics, score_norm, conflict_norm, mode) in variants.items():
        for row in parent_rows:
            oid = str(row["mv_object_id"])
            score = _num(row.get("score"), 1.0)
            if mode in {"boost", "boost_veto"}:
                score += EPS * score_norm.get(oid, 0.0)
            if mode == "boost3":
                score += 3.0 * EPS * score_norm.get(oid, 0.0)
            if mode in {"veto", "boost_veto"}:
                score -= EPS * conflict_norm.get(oid, 0.0)
            new = dict(row)
            new["variant_id"] = variant
            new["score"] = float(score)
            new["score_policy"] = policy
            new["phase6_parent_variant_id"] = row["variant_id"]
            new["dense_semantic_provider"] = "DINOv2_timm_dense_patch_pool" if not variant.startswith("P6_C1") else "DINOv2_timm_dense_patch_pool_shuffled_control"
            new["dense_semantic_score_mean"] = metrics.get(oid, {}).get("dense_semantic_score_mean", 0.0)
            new["dense_semantic_valid_link_count"] = metrics.get(oid, {}).get("dense_semantic_valid_link_count", 0.0)
            new["dense_semantic_conflict_count"] = metrics.get(oid, {}).get("dense_semantic_conflict_count", 0.0)
            new["uses_gt_for_prediction"] = False
            new["uses_future"] = False
            out.append(new)
    return out


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    phase2_summary = json.loads(PHASE2_SUMMARY.read_text(encoding="utf-8"))
    availability = json.loads(AVAILABILITY.read_text(encoding="utf-8")) if AVAILABILITY.exists() else {}
    parent_variant, parent_rows = _phase2_best_rows()
    scope = p1._load_source_scope()

    dense_features, dense_feature_rows, feature_stats = _extract_dense_dino_features(
        parent_rows,
        scope,
        device="cuda:0",
        short_side=518,
    )
    dense_residuals, mu, tau = _residualize(dense_features)
    np.save(OUT_DIR / "dino_dense_mu_vector.npy", mu)
    np.savez_compressed(
        OUT_DIR / "dino_dense_feature_store.npz",
        scene_id=np.asarray([key[0] for key in sorted(dense_features)], dtype=object),
        frame_id=np.asarray([key[1] for key in sorted(dense_features)], dtype=np.int32),
        mask_id=np.asarray([key[2] for key in sorted(dense_features)], dtype=np.int32),
        features=np.stack([dense_features[key] for key in sorted(dense_features)]).astype(np.float32) if dense_features else np.zeros((0, 0), dtype=np.float32),
        residuals=np.stack([dense_residuals[key] for key in sorted(dense_residuals)]).astype(np.float32) if dense_residuals else np.zeros((0, 0), dtype=np.float32),
    )
    radio_residuals = _load_radio_residuals()
    distribution_rows, proxy_agreement = _pair_distribution(parent_rows, dense_residuals, radio_residuals, tau)
    real_metrics = _object_semantic_metrics(parent_rows, dense_residuals, tau)
    shuffled_metrics = _object_semantic_metrics(parent_rows, _shuffled_residuals(dense_residuals), tau)
    all_rows = _variant_rows(parent_rows, real_metrics, shuffled_metrics)

    metric_scene_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for variant in sorted({row["variant_id"] for row in all_rows}):
        rows = [row for row in all_rows if row["variant_id"] == variant]
        metrics, frames = p1._evaluate_variant(variant, rows, scope)
        metric_scene_rows.extend(metrics)
        frame_rows.extend(frames)
    aggregate_rows = p1._aggregate_metrics(metric_scene_rows)

    valid_links = [vals["dense_semantic_valid_link_count"] for vals in real_metrics.values()]
    real_ids = {
        "P6_D1_dense_dino_boost",
        "P6_D2_dense_dino_veto",
        "P6_D3_dense_dino_boost_plus_veto",
        "P6_D4_dense_dino_boost_eps3e-4",
    }
    best_real = max([row for row in aggregate_rows if row["variant_id"] in real_ids], key=lambda row: (float(row["MV_AP_window"]), float(row["MV_AP_scene"]), float(row["MV_AP50_window"])))
    shuffled = next(row for row in aggregate_rows if row["variant_id"] == "P6_C1_shuffled_dense_dino_boost_plus_veto")
    baseline = next(row for row in aggregate_rows if row["variant_id"] == "P6_B0_phase2_best_no_dense_semantic")
    base_window = float(phase0["F2_base_full_dev_MV_AP_window"])
    base_ap50 = float(phase0["F2_base_full_dev_MV_AP50_window"])
    success_local = bool(float(best_real["MV_AP_window"]) >= base_window + 0.005 and float(best_real["MV_AP50_window"]) >= base_ap50 + 0.010)
    real_minus_shuffled_window = float(best_real["MV_AP_window"]) - float(shuffled["MV_AP_window"])
    real_minus_shuffled_scene = float(best_real["MV_AP_scene"]) - float(shuffled["MV_AP_scene"])
    valid_rate_pass = bool(feature_stats["dense_feature_valid_rate"] >= 0.95)
    control_margin_pass = bool(real_minus_shuffled_window >= 0.003 or real_minus_shuffled_scene >= 0.006)
    safety_pass = bool(
        int(best_real["same_frame_collision_count"]) == 0
        and float(best_real["pixel_collision_rate"]) <= 0.02
        and int(best_real["missing_mask_raster_count"]) == 0
    )
    phase6_pass = bool(valid_rate_pass and success_local and control_margin_pass and safety_pass)
    gate_rows = [
        {
            "gate_id": "dense_semantic_valid_rate_ge_0p95",
            "pass": valid_rate_pass,
            "expected": ">=0.95",
            "observed": feature_stats["dense_feature_valid_rate"],
            "severity": "required_feature_extraction",
        },
        {
            "gate_id": "dense_semantic_local_success_vs_F2_base",
            "pass": success_local,
            "expected": f"MV_AP_window>={base_window + 0.005} and MV_AP50_window>={base_ap50 + 0.010}",
            "observed": f"MV_AP_window={best_real['MV_AP_window']}; MV_AP50_window={best_real['MV_AP50_window']}",
            "severity": "plan_success",
        },
        {
            "gate_id": "dense_semantic_real_minus_shuffled_margin",
            "pass": control_margin_pass,
            "expected": "real-shuffled >=0.003 MV_AP_window or >=0.006 MV_AP_scene",
            "observed": f"window_margin={real_minus_shuffled_window}; scene_margin={real_minus_shuffled_scene}",
            "severity": "control",
        },
        {
            "gate_id": "same_frame_collision_count_eq_0",
            "pass": int(best_real["same_frame_collision_count"]) == 0,
            "expected": "0",
            "observed": best_real["same_frame_collision_count"],
            "severity": "required",
        },
        {
            "gate_id": "pixel_collision_rate_le_0p02",
            "pass": float(best_real["pixel_collision_rate"]) <= 0.02,
            "expected": "<=0.02",
            "observed": best_real["pixel_collision_rate"],
            "severity": "required",
        },
        {
            "gate_id": "missing_mask_raster_count_eq_0",
            "pass": int(best_real["missing_mask_raster_count"]) == 0,
            "expected": "0",
            "observed": best_real["missing_mask_raster_count"],
            "severity": "required",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "if dense feature extraction is valid but AP/control fails, keep the existing mask-level RADIO proxy and do not promote dense semantic",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    constant_rows = [
        {
            "schema_version": "stream4d_v99_phase6_semantic_constant_v1",
            "phase_id": "v99_phase6_dense_semantic_residual",
            "feature_provider": "DINOv2_timm_dense_patch_pool",
            "mu_vector_path": _rel(OUT_DIR / "dino_dense_mu_vector.npy"),
            "tau_sem": tau,
            "semantic_feature_valid_rate": feature_stats["dense_feature_valid_rate"],
            "mask_proxy_vs_dense_agreement": proxy_agreement,
            "radio_available": availability.get("radio_available", ""),
            "dinov2_checkpoint": feature_stats["dinov2_checkpoint"],
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]
    for row in distribution_rows:
        row["semantic_feature_valid_rate"] = feature_stats["dense_feature_valid_rate"]
        row["mask_proxy_vs_dense_agreement"] = proxy_agreement
    summary = {
        "schema_version": "stream4d_v99_phase6_dense_semantic_residual_summary_v1",
        "phase_id": "v99_phase6_dense_semantic_residual",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "PASS_DENSE_SEMANTIC_CONTRIBUTION" if phase6_pass else "NO_GO_DENSE_SEMANTIC_KEEP_MASK_PROXY",
        "phase6_pass": phase6_pass,
        "parent_phase2_variant": parent_variant,
        "best_real_variant": best_real["variant_id"],
        "best_real_MV_AP_window": float(best_real["MV_AP_window"]),
        "best_real_MV_AP50_window": float(best_real["MV_AP50_window"]),
        "best_real_MV_AP_scene": float(best_real["MV_AP_scene"]),
        "best_real_MV_AP50_scene": float(best_real["MV_AP50_scene"]),
        "baseline_MV_AP_window": float(baseline["MV_AP_window"]),
        "baseline_MV_AP_scene": float(baseline["MV_AP_scene"]),
        "shuffled_MV_AP_window": float(shuffled["MV_AP_window"]),
        "shuffled_MV_AP_scene": float(shuffled["MV_AP_scene"]),
        "real_minus_shuffled_MV_AP_window": real_minus_shuffled_window,
        "real_minus_shuffled_MV_AP_scene": real_minus_shuffled_scene,
        "dense_feature_valid_rate": feature_stats["dense_feature_valid_rate"],
        "semantic_feature_valid_rate": feature_stats["dense_feature_valid_rate"],
        "mask_proxy_vs_dense_agreement": proxy_agreement,
        "dense_semantic_valid_link_mean": float(np.mean(valid_links)) if valid_links else 0.0,
        "radio_available": availability.get("radio_available", ""),
        "dinov2_checkpoint": feature_stats["dinov2_checkpoint"],
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "memory_MB": "",
        "blocking_failure_count": len(failure_rows),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "outputs": {
            "semantic_distribution_rows": _rel(OUT_DIR / "semantic_distribution_rows.csv"),
            "semantic_constant_rows": _rel(OUT_DIR / "semantic_constant_rows.csv"),
            "dense_feature_rows": _rel(OUT_DIR / "dense_feature_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "summary": _rel(OUT_DIR / "summary.json"),
        },
    }
    _write_csv(OUT_DIR / "dense_feature_rows.csv", dense_feature_rows)
    _write_csv(OUT_DIR / "semantic_distribution_rows.csv", distribution_rows)
    _write_csv(OUT_DIR / "semantic_constant_rows.csv", constant_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", metric_scene_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", all_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase6_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
