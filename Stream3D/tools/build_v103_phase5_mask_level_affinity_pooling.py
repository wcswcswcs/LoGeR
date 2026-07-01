#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
PHASE_ID = "v103_phase5_mask_level_affinity_pooling"
DEFAULT_PHASE4_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase4_primitive_affinity_q5c_support_balanced_r7_object_preserve_downweight"
DEFAULT_PHASE3_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase3_carrier_reliability_filter_q5c_objlike16384_fast_support_balanced"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase5_mask_level_pooling_q5c_phase4r7"
SKETCH_SEED = 10317


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _hash_mask(mask_idx: np.ndarray, sketch_dim: int) -> tuple[np.ndarray, np.ndarray]:
    mask_idx = np.asarray(mask_idx, dtype=np.int64)
    bucket = ((mask_idx * 2654435761 + SKETCH_SEED) % int(sketch_dim)).astype(np.int64)
    sign = np.where(((mask_idx * 1103515245 + SKETCH_SEED) % 2) == 0, 1.0, -1.0).astype(np.float32)
    return bucket, sign


def _load_phase4_scene(phase4_root: Path, scene: str) -> dict[str, Any]:
    path = phase4_root / scene / "primitive_incidence_sparse.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def _load_reliability(phase3_root: Path, scene: str, carrier_id: np.ndarray) -> np.ndarray:
    path = phase3_root / scene / "carrier_reliability_rows.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path, columns=["carrier_id", "reliability_s2", "broad_mask_participation_rate"])
    ids = df["carrier_id"].to_numpy(dtype=np.int64, copy=False)
    order = np.argsort(ids, kind="mergesort")
    ids_sorted = ids[order]
    rel_sorted = df["reliability_s2"].to_numpy(dtype=np.float32, copy=False)[order]
    broad_sorted = df["broad_mask_participation_rate"].to_numpy(dtype=np.float32, copy=False)[order]
    rel = np.zeros((carrier_id.shape[0],), dtype=np.float32)
    broad = np.ones((carrier_id.shape[0],), dtype=np.float32)
    if ids_sorted.size:
        pos = np.searchsorted(ids_sorted, np.asarray(carrier_id, dtype=np.int64))
        found = (pos < ids_sorted.shape[0]) & (ids_sorted[np.minimum(pos, ids_sorted.shape[0] - 1)] == carrier_id)
        if np.any(found):
            rel[found] = rel_sorted[pos[found]]
            broad[found] = broad_sorted[pos[found]]
    return rel, broad


def _build_raw_sketch(
    carrier_idx: np.ndarray,
    mask_idx: np.ndarray,
    b_ia: np.ndarray,
    mask_weight: np.ndarray,
    carrier_count: int,
    sketch_dim: int,
    device: torch.device,
) -> torch.Tensor:
    c_t = torch.as_tensor(carrier_idx, dtype=torch.long, device=device)
    m_t = torch.as_tensor(mask_idx, dtype=torch.long, device=device)
    b_t = torch.as_tensor(b_ia, dtype=torch.float32, device=device)
    w_t = torch.as_tensor(mask_weight, dtype=torch.float32, device=device)
    bucket = ((m_t * 2654435761 + SKETCH_SEED) % int(sketch_dim)).to(torch.long)
    sign = torch.where(((m_t * 1103515245 + SKETCH_SEED) % 2) == 0, 1.0, -1.0).to(torch.float32)
    values = torch.sqrt(w_t[m_t]) * b_t * sign
    raw = torch.zeros((carrier_count, int(sketch_dim)), dtype=torch.float32, device=device)
    raw.index_put_((c_t, bucket), values, accumulate=True)
    return raw


def _pool_features(
    *,
    variant_id: str,
    raw: torch.Tensor,
    incidence_by_mask: list[np.ndarray],
    carrier_idx: np.ndarray,
    mask_idx: np.ndarray,
    b_ia: np.ndarray,
    alpha: np.ndarray,
    carrier_broad: np.ndarray,
    mask_weight: np.ndarray,
    topk: int,
    trim_quantile: float,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    t0 = time.time()
    sketch_dim = int(raw.shape[1])
    raw_norm = torch.nn.functional.normalize(raw, p=2, dim=1, eps=1e-12)
    out = torch.zeros((len(incidence_by_mask), sketch_dim), dtype=torch.float32, device=device)
    bucket_np, sign_np = _hash_mask(np.arange(len(incidence_by_mask), dtype=np.int64), sketch_dim)
    for mask_id, row_idx in enumerate(incidence_by_mask):
        if row_idx.size == 0:
            continue
        rows = row_idx
        if variant_id == "P5_incidence_weighted_leave_one_out":
            weights = b_ia[rows].astype(np.float32, copy=True)
        else:
            weights = alpha[rows].astype(np.float32, copy=True)
        if variant_id == "P1_topk_reliable_pool" and rows.size > int(topk):
            keep = np.argpartition(weights, rows.size - int(topk))[rows.size - int(topk) :]
            rows = rows[keep]
            weights = weights[keep]
        elif variant_id == "P2_trimmed_mean_pool" and rows.size > 8:
            lo = float(np.quantile(weights, trim_quantile))
            broad_vals = carrier_broad[carrier_idx[rows]]
            hi_broad = float(np.quantile(broad_vals, 1.0 - trim_quantile))
            keep = (weights >= lo) & (broad_vals <= hi_broad)
            if np.any(keep):
                rows = rows[keep]
                weights = weights[keep]
        elif variant_id == "P3_attention_by_specificity":
            weights = weights * float(mask_weight[mask_id])

        carriers = carrier_idx[rows]
        if carriers.size == 0 or float(np.sum(weights)) <= 0.0:
            continue
        c_t = torch.as_tensor(carriers, dtype=torch.long, device=device)
        w_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
        if variant_id == "P4_no_leave_one_out_control":
            vec = raw_norm[c_t]
        else:
            vec = raw[c_t].clone()
            contrib = (
                np.sqrt(float(mask_weight[mask_id]))
                * b_ia[rows].astype(np.float32)
                * float(sign_np[mask_id])
            )
            local = torch.arange(c_t.shape[0], dtype=torch.long, device=device)
            contrib_t = torch.as_tensor(contrib, dtype=torch.float32, device=device)
            vec[local, int(bucket_np[mask_id])] -= contrib_t
            vec = torch.nn.functional.normalize(vec, p=2, dim=1, eps=1e-12)
        pooled = torch.sum(vec * w_t[:, None], dim=0) / torch.clamp(torch.sum(w_t), min=1e-12)
        out[mask_id] = torch.nn.functional.normalize(pooled[None, :], p=2, dim=1, eps=1e-12)[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out.detach().cpu().numpy().astype(np.float32, copy=False), time.time() - t0


def _sample_mask_pairs(
    incidence_by_mask: list[np.ndarray],
    carrier_idx: np.ndarray,
    mask_frame: np.ndarray,
    mask_is_object: np.ndarray,
    mask_is_broad: np.ndarray,
    max_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    carrier_sets: list[set[int]] = []
    for rows in incidence_by_mask:
        carrier_sets.append(set(int(v) for v in np.unique(carrier_idx[rows]).tolist()))

    obj_masks = [i for i, flag in enumerate(mask_is_object.tolist()) if bool(flag) and len(carrier_sets[i]) > 0]
    pseudo: list[tuple[int, int]] = []
    for a, b in combinations(obj_masks, 2):
        if int(mask_frame[a]) == int(mask_frame[b]):
            continue
        shared = len(carrier_sets[a].intersection(carrier_sets[b]))
        if shared >= 8:
            pseudo.append((int(a), int(b)))
            if len(pseudo) >= max_pairs:
                break

    same_frame: list[tuple[int, int]] = []
    broad_pairs: list[tuple[int, int]] = []
    by_frame: dict[int, list[int]] = {}
    broad_by_frame: dict[int, list[int]] = {}
    for m in range(len(mask_frame)):
        if bool(mask_is_object[m]) and carrier_sets[m]:
            by_frame.setdefault(int(mask_frame[m]), []).append(m)
        if bool(mask_is_broad[m]) and carrier_sets[m]:
            broad_by_frame.setdefault(int(mask_frame[m]), []).append(m)
    for frame, masks in by_frame.items():
        for a, b in combinations(masks, 2):
            same_frame.append((int(a), int(b)))
            if len(same_frame) >= max_pairs:
                break
        for a in masks:
            for b in broad_by_frame.get(int(frame), []):
                broad_pairs.append((int(a), int(b)))
                if len(broad_pairs) >= max_pairs:
                    break
            if len(broad_pairs) >= max_pairs:
                break
        if len(same_frame) >= max_pairs and len(broad_pairs) >= max_pairs:
            break

    hard = (same_frame[: max_pairs // 2] + broad_pairs[: max_pairs - max_pairs // 2])[:max_pairs]
    return (
        np.asarray(pseudo, dtype=np.int64),
        np.asarray(hard, dtype=np.int64),
        np.asarray(same_frame[:max_pairs], dtype=np.int64),
        np.asarray(broad_pairs[:max_pairs], dtype=np.int64),
    )


def _pair_values_static(feature: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.sum(feature[pairs[:, 0]] * feature[pairs[:, 1]], axis=1).astype(np.float32, copy=False)


def _pair_values_strict_leave_two_out_bucket_zeroed(
    feature: np.ndarray,
    pairs: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    if pairs.size == 0:
        return np.zeros((0,), dtype=np.float32)
    sketch_dim = int(feature.shape[1])
    bucket_np, _sign_np = _hash_mask(np.arange(feature.shape[0], dtype=np.int64), sketch_dim)
    with torch.no_grad():
        feat = torch.as_tensor(feature, dtype=torch.float32, device=device)
        pair_t = torch.as_tensor(pairs, dtype=torch.long, device=device)
        bucket_t = torch.as_tensor(bucket_np, dtype=torch.long, device=device)
        rows = torch.arange(pair_t.shape[0], dtype=torch.long, device=device)
        a = feat[pair_t[:, 0]].clone()
        b = feat[pair_t[:, 1]].clone()
        ba = bucket_t[pair_t[:, 0]]
        bb = bucket_t[pair_t[:, 1]]
        a[rows, ba] = 0.0
        a[rows, bb] = 0.0
        b[rows, ba] = 0.0
        b[rows, bb] = 0.0
        a = torch.nn.functional.normalize(a, p=2, dim=1, eps=1e-12)
        b = torch.nn.functional.normalize(b, p=2, dim=1, eps=1e-12)
        vals = torch.sum(a * b, dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return vals.detach().cpu().numpy().astype(np.float32, copy=False)


def _pair_values(feature: np.ndarray, pairs: np.ndarray, *, pair_affinity_mode: str, device: torch.device) -> np.ndarray:
    if pair_affinity_mode == "strict_leave_two_out_bucket_zeroed":
        return _pair_values_strict_leave_two_out_bucket_zeroed(feature, pairs, device=device)
    if pair_affinity_mode != "static_feature_cosine":
        raise ValueError(f"unsupported pair_affinity_mode: {pair_affinity_mode}")
    return _pair_values_static(feature, pairs)


def _variant_metrics(
    *,
    scene: str,
    variant_id: str,
    static_feature_source: str,
    pair_affinity_mode: str,
    feature: np.ndarray,
    support_count: np.ndarray,
    mask_is_object: np.ndarray,
    mask_is_broad: np.ndarray,
    pseudo_pairs: np.ndarray,
    hard_pairs: np.ndarray,
    same_frame_pairs: np.ndarray,
    broad_pairs: np.ndarray,
    min_carriers: int,
    runtime_sec: float,
    device: torch.device,
) -> dict[str, Any]:
    norm = np.linalg.norm(feature, axis=1)
    valid = norm > 0
    candidate = support_count > 0
    obj = mask_is_object.astype(bool)
    broad = mask_is_broad.astype(bool)
    pseudo = _pair_values(feature, pseudo_pairs, pair_affinity_mode=pair_affinity_mode, device=device)
    hard = _pair_values(feature, hard_pairs, pair_affinity_mode=pair_affinity_mode, device=device)
    same_frame = _pair_values(feature, same_frame_pairs, pair_affinity_mode=pair_affinity_mode, device=device)
    broad_vals = _pair_values(feature, broad_pairs, pair_affinity_mode=pair_affinity_mode, device=device)
    pseudo_mean = float(np.mean(pseudo)) if pseudo.size else 0.0
    hard_mean = float(np.mean(hard)) if hard.size else 0.0
    return {
        "schema_version": "stream4d_v103_phase5_mask_pooling_metric_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "variant_id": variant_id,
        "static_feature_source": static_feature_source,
        "pair_affinity_mode": pair_affinity_mode,
        "mask_feature_valid_rate": float(np.mean(valid[candidate])) if np.any(candidate) else 0.0,
        "mask_observation_support_nonzero_rate": float(np.mean(candidate)) if candidate.size else 0.0,
        "mask_with_min_reliable_carriers_rate": float(np.mean(support_count[obj] >= int(min_carriers))) if np.any(obj) else 0.0,
        "mean_reliable_carriers_per_mask": float(np.mean(support_count)) if support_count.size else 0.0,
        "object_like_mask_feature_coverage": float(np.mean(valid[obj])) if np.any(obj) else 0.0,
        "broad_mask_feature_coverage": float(np.mean(valid[broad])) if np.any(broad) else 0.0,
        "pseudo_positive_affinity_mean": pseudo_mean,
        "hard_negative_affinity_mean": hard_mean,
        "pseudo_positive_minus_hard_negative_margin": pseudo_mean - hard_mean,
        "same_frame_competing_mask_affinity_p95": float(np.percentile(same_frame, 95)) if same_frame.size else 0.0,
        "broad_mask_affinity_p95": float(np.percentile(broad_vals, 95)) if broad_vals.size else 0.0,
        "pseudo_positive_pair_count": int(pseudo.size),
        "hard_negative_pair_count": int(hard.size),
        "same_frame_pair_count": int(same_frame.size),
        "broad_pair_count": int(broad_vals.size),
        "pooling_runtime_sec": runtime_sec,
        "uses_gt_for_pooling": False,
        "uses_future": False,
    }


def _frame_centered_feature(feature: np.ndarray, mask_frame: np.ndarray, mask_is_object: np.ndarray, beta: float) -> np.ndarray:
    out = np.asarray(feature, dtype=np.float32).copy()
    for frame in sorted(set(int(v) for v in mask_frame.tolist())):
        idx = np.flatnonzero((mask_frame == int(frame)) & mask_is_object)
        if idx.size < 2:
            continue
        center = np.mean(out[idx], axis=0, dtype=np.float32)
        out[idx] = out[idx] - float(beta) * center[None, :]
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.maximum(norm, 1e-12)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32, copy=False)


def _run_scene(scene: str, args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    phase4_root = _project(args.phase4_root)
    phase3_root = _project(args.phase3_root)
    out = _project(args.output_root) / scene
    out.mkdir(parents=True, exist_ok=True)
    data = _load_phase4_scene(phase4_root, scene)
    carrier_id = data["carrier_id"].cpu().numpy().astype(np.int64)
    carrier_idx = data["carrier_local_index"].cpu().numpy().astype(np.int64)
    mask_idx = data["mask_observation_index"].cpu().numpy().astype(np.int64)
    b_ia = data["B_ia"].cpu().numpy().astype(np.float32)
    mask_frame = data["mask_frame"].cpu().numpy().astype(np.int64)
    mask_label = data["mask_label"].cpu().numpy().astype(np.int64)
    mask_is_object = data["mask_is_object_like"].cpu().numpy().astype(bool)
    mask_is_broad = data["mask_is_broad"].cpu().numpy().astype(bool)
    mask_weight = data["mask_weight"].cpu().numpy().astype(np.float32)
    reliability, carrier_broad = _load_reliability(phase3_root, scene, carrier_id)
    alpha = (reliability[carrier_idx] * b_ia).astype(np.float32)
    incidence_by_mask = [np.flatnonzero(mask_idx == m).astype(np.int64) for m in range(mask_frame.shape[0])]
    support_count = np.bincount(mask_idx, minlength=int(mask_frame.shape[0])).astype(np.int64, copy=False)
    raw = _build_raw_sketch(carrier_idx, mask_idx, b_ia, mask_weight, int(carrier_id.shape[0]), int(args.sketch_dim), device)
    pseudo_pairs, hard_pairs, same_frame_pairs, broad_pairs = _sample_mask_pairs(
        incidence_by_mask,
        carrier_idx,
        mask_frame,
        mask_is_object,
        mask_is_broad,
        int(args.max_pair_rows),
    )
    variants = [
        "P0_mean_reliability_weighted",
        "P1_topk_reliable_pool",
        "P2_trimmed_mean_pool",
        "P3_attention_by_specificity",
        "P5_incidence_weighted_leave_one_out",
        "P4_no_leave_one_out_control",
    ]
    metric_rows: list[dict[str, Any]] = []
    features_by_variant: dict[str, np.ndarray] = {}
    pair_modes_by_variant: dict[str, str] = {}
    for variant in variants:
        feature, runtime = _pool_features(
            variant_id=variant,
            raw=raw,
            incidence_by_mask=incidence_by_mask,
            carrier_idx=carrier_idx,
            mask_idx=mask_idx,
            b_ia=b_ia,
            alpha=alpha,
            carrier_broad=carrier_broad,
            mask_weight=mask_weight,
            topk=int(args.topk_carriers),
            trim_quantile=float(args.trim_quantile),
            device=device,
        )
        features_by_variant[variant] = feature
        pair_modes_by_variant[variant] = "static_feature_cosine"
        metric_rows.append(
            _variant_metrics(
                scene=scene,
                variant_id=variant,
                static_feature_source=variant,
                pair_affinity_mode=pair_modes_by_variant[variant],
                feature=feature,
                support_count=support_count,
                mask_is_object=mask_is_object,
                mask_is_broad=mask_is_broad,
                pseudo_pairs=pseudo_pairs,
                hard_pairs=hard_pairs,
                same_frame_pairs=same_frame_pairs,
                broad_pairs=broad_pairs,
                min_carriers=int(args.min_reliable_carriers_per_mask),
                runtime_sec=runtime,
                device=device,
            )
        )
    for beta in [0.50, 1.00]:
        variant = f"P6_frame_centered_leave_one_out_b{str(beta).replace('.', 'p')}"
        feature = _frame_centered_feature(features_by_variant["P0_mean_reliability_weighted"], mask_frame, mask_is_object, beta)
        features_by_variant[variant] = feature
        pair_modes_by_variant[variant] = "static_feature_cosine"
        metric_rows.append(
            _variant_metrics(
                scene=scene,
                variant_id=variant,
                static_feature_source="P0_mean_reliability_weighted",
                pair_affinity_mode=pair_modes_by_variant[variant],
                feature=feature,
                support_count=support_count,
                mask_is_object=mask_is_object,
                mask_is_broad=mask_is_broad,
                pseudo_pairs=pseudo_pairs,
                hard_pairs=hard_pairs,
                same_frame_pairs=same_frame_pairs,
                broad_pairs=broad_pairs,
                min_carriers=int(args.min_reliable_carriers_per_mask),
                runtime_sec=0.0,
                device=device,
            )
        )
    strict_sources = [
        "P0_mean_reliability_weighted",
        "P6_frame_centered_leave_one_out_b0p5",
    ]
    for base_variant in strict_sources:
        variant = f"P7_strict_leave_two_out_from_{base_variant}"
        feature = features_by_variant[base_variant]
        features_by_variant[variant] = feature
        pair_modes_by_variant[variant] = "strict_leave_two_out_bucket_zeroed"
        metric_rows.append(
            _variant_metrics(
                scene=scene,
                variant_id=variant,
                static_feature_source=base_variant,
                pair_affinity_mode=pair_modes_by_variant[variant],
                feature=feature,
                support_count=support_count,
                mask_is_object=mask_is_object,
                mask_is_broad=mask_is_broad,
                pseudo_pairs=pseudo_pairs,
                hard_pairs=hard_pairs,
                same_frame_pairs=same_frame_pairs,
                broad_pairs=broad_pairs,
                min_carriers=int(args.min_reliable_carriers_per_mask),
                runtime_sec=0.0,
                device=device,
            )
        )
    p4 = next(row for row in metric_rows if row["variant_id"] == "P4_no_leave_one_out_control")
    candidates = [row for row in metric_rows if row["variant_id"] != "P4_no_leave_one_out_control"]

    def key(row: dict[str, Any]) -> tuple[int, float]:
        checks = [
            float(row["mask_feature_valid_rate"]) >= 0.90,
            float(row["mask_with_min_reliable_carriers_rate"]) >= 0.80,
            float(row["pseudo_positive_minus_hard_negative_margin"]) >= 0.10,
            float(row["same_frame_competing_mask_affinity_p95"]) <= float(row["pseudo_positive_affinity_mean"]),
            float(row["hard_negative_affinity_mean"]) <= float(p4["hard_negative_affinity_mean"]),
        ]
        return (sum(bool(v) for v in checks), float(row["pseudo_positive_minus_hard_negative_margin"]))

    selected = max(candidates, key=key)
    selected_feature = features_by_variant[str(selected["variant_id"])]
    feature_path = out / "mask_level_feature.pt"
    torch.save(
        {
            "schema_version": "stream4d_v103_phase5_mask_level_feature_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "variant_id": selected["variant_id"],
            "static_feature_source": selected.get("static_feature_source", selected["variant_id"]),
            "pair_affinity_mode": pair_modes_by_variant[str(selected["variant_id"])],
            "mask_observation_index": torch.arange(mask_frame.shape[0], dtype=torch.int64),
            "mask_frame": torch.as_tensor(mask_frame, dtype=torch.int64),
            "mask_label": torch.as_tensor(mask_label, dtype=torch.int64),
            "mask_is_object_like": torch.as_tensor(mask_is_object, dtype=torch.bool),
            "mask_is_broad": torch.as_tensor(mask_is_broad, dtype=torch.bool),
            "support_count": torch.as_tensor(support_count, dtype=torch.int64),
            "feature": torch.as_tensor(selected_feature, dtype=torch.float16),
            "uses_gt": False,
            "uses_future": False,
        },
        feature_path,
    )
    pair_rows: list[dict[str, Any]] = []
    for pair_type, pairs in [("pseudo_positive", pseudo_pairs), ("hard_negative", hard_pairs), ("same_frame_competing", same_frame_pairs), ("object_broad", broad_pairs)]:
        vals = _pair_values(
            selected_feature,
            pairs,
            pair_affinity_mode=pair_modes_by_variant[str(selected["variant_id"])],
            device=device,
        )
        for idx, (a, b) in enumerate(pairs[: int(args.max_casebook_rows)].tolist()):
            pair_rows.append(
                {
                    "schema_version": "stream4d_v103_phase5_mask_pair_affinity_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "pair_type": pair_type,
                    "pair_affinity_mode": pair_modes_by_variant[str(selected["variant_id"])],
                    "mask_a": int(a),
                    "mask_b": int(b),
                    "affinity": float(vals[idx]) if idx < vals.size else 0.0,
                }
            )
    pair_path = out / "mask_pair_affinity_rows.parquet"
    pd.DataFrame(pair_rows).to_parquet(pair_path, index=False)
    casebook_path = out / "mask_pair_casebook_rows.csv"
    _write_csv(casebook_path, pair_rows[: int(args.max_casebook_rows)])

    gate_specs = [
        ("mask_feature_valid_rate_ge_0p90", float(selected["mask_feature_valid_rate"]) >= 0.90, selected["mask_feature_valid_rate"], 0.90),
        ("mask_with_min_reliable_carriers_rate_ge_0p80", float(selected["mask_with_min_reliable_carriers_rate"]) >= 0.80, selected["mask_with_min_reliable_carriers_rate"], 0.80),
        ("pseudo_positive_minus_hard_negative_margin_ge_0p10", float(selected["pseudo_positive_minus_hard_negative_margin"]) >= 0.10, selected["pseudo_positive_minus_hard_negative_margin"], 0.10),
        ("same_frame_competing_p95_le_pseudo_positive_mean", float(selected["same_frame_competing_mask_affinity_p95"]) <= float(selected["pseudo_positive_affinity_mean"]), selected["same_frame_competing_mask_affinity_p95"], selected["pseudo_positive_affinity_mean"]),
        ("leave_one_out_hard_negative_mean_le_no_leave_one_out", float(selected["hard_negative_affinity_mean"]) <= float(p4["hard_negative_affinity_mean"]), selected["hard_negative_affinity_mean"], p4["hard_negative_affinity_mean"]),
    ]
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for name, ok, observed, required in gate_specs:
        gate_rows.append(
            {
                "schema_version": "stream4d_v103_phase5_gate_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "selected_variant_id": selected["variant_id"],
                "gate_name": name,
                "pass": bool(ok),
                "observed": observed,
                "required": required,
            }
        )
        if not ok:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_phase5_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "failure_id": name,
                    "severity": "blocking",
                    "evidence": f"selected_variant={selected['variant_id']} observed={observed} required={required}",
                    "repair_direction": "Follow Phase5 repair ladder: if coverage is low return to Phase2/3; if hard negatives remain no lower than no-leave-one-out, strengthen carrier filtering, IDF, strict leave-two-out, or cannot-link construction rather than tuning clustering.",
                }
            )
    artifact_rows = [
        {
            "schema_version": "stream4d_v103_phase5_artifact_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "role": "mask_level_feature",
            "path": _rel(feature_path),
            "exists": feature_path.exists(),
            "size_bytes": feature_path.stat().st_size if feature_path.exists() else 0,
        },
        {
            "schema_version": "stream4d_v103_phase5_artifact_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "role": "mask_pair_affinity_rows",
            "path": _rel(pair_path),
            "exists": pair_path.exists(),
            "size_bytes": pair_path.stat().st_size if pair_path.exists() else 0,
        },
    ]
    control_rows = [
        {
            "schema_version": "stream4d_v103_phase5_control_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "control_id": "P4_no_leave_one_out_control",
            "selected_variant_id": selected["variant_id"],
            "selected_margin": selected["pseudo_positive_minus_hard_negative_margin"],
            "control_margin": p4["pseudo_positive_minus_hard_negative_margin"],
            "control_hard_negative_mean": p4["hard_negative_affinity_mean"],
            "selected_hard_negative_mean": selected["hard_negative_affinity_mean"],
            "hard_negative_separation_pass": float(selected["hard_negative_affinity_mean"]) <= float(p4["hard_negative_affinity_mean"]),
            "control_gate_contract": "Plan wording is hard-negative separation: no-leave-one-out must not have lower hard-negative affinity than leave-one-out. Margin-vs-control is diagnostic only.",
            "selected_pair_affinity_mode": pair_modes_by_variant[str(selected["variant_id"])],
        }
    ]
    return metric_rows, gate_rows, failure_rows, control_rows, artifact_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase5 mask-level affinity feature pooling from primitive affinity features.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase4-root", default=str(DEFAULT_PHASE4_ROOT))
    parser.add_argument("--phase3-root", default=str(DEFAULT_PHASE3_ROOT))
    parser.add_argument("--scene", choices=["all", "scene0011_00", "scene0050_00"], default="all")
    parser.add_argument("--sketch-dim", type=int, default=2048)
    parser.add_argument("--topk-carriers", type=int, default=128)
    parser.add_argument("--trim-quantile", type=float, default=0.10)
    parser.add_argument("--min-reliable-carriers-per-mask", type=int, default=50)
    parser.add_argument("--max-pair-rows", type=int, default=4096)
    parser.add_argument("--max-casebook-rows", type=int, default=512)
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    phase4_root = _project(args.phase4_root)
    phase4_summary = json.loads((phase4_root / "summary.json").read_text(encoding="utf-8"))
    if not bool(phase4_summary.get("phase4_pass")):
        raise RuntimeError(f"Phase4 root did not pass: {phase4_root}")
    scene_ids = ["scene0011_00", "scene0050_00"] if args.scene == "all" else [args.scene]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for scene in scene_ids:
        metrics, gates, failures, controls, artifacts = _run_scene(scene, args, device)
        metric_rows.extend(metrics)
        gate_rows.extend(gates)
        failure_rows.extend(failures)
        control_rows.extend(controls)
        artifact_rows.extend(artifacts)
    _write_csv(out / "mask_pooling_metric_rows.csv", metric_rows)
    _write_csv(out / "mask_feature_control_rows.csv", control_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "artifact_rows.csv", artifact_rows)
    decision = "PASS_ENTER_PHASE6_MASK_CLUSTERING" if not failure_rows else "NO_GO_REPAIR_PHASE5_MASK_LEVEL_POOLING"
    summary = {
        "schema_version": "stream4d_v103_phase5_mask_level_affinity_pooling_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase5_pass": not failure_rows,
        "failure_count": len(failure_rows),
        "scene_ids": scene_ids,
        "phase4_root": _rel(phase4_root),
        "uses_gt_for_pooling": False,
        "uses_future": False,
        "truthfulness_note": "Phase5 pools primitive affinity features into mask-level features. It does not perform clustering or AP evaluation.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "mask_pooling_metric_rows": _rel(out / "mask_pooling_metric_rows.csv"),
            "mask_feature_control_rows": _rel(out / "mask_feature_control_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
