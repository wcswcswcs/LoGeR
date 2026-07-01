#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v103_phase3_fast_carrier_reliability_filter import (  # noqa: E402
    PHASE_ID as PHASE3_ID,
    ALL_SUPPORT_BALANCED_VARIANTS,
    SEMANTIC_CONTRADICTION_THRESHOLD,
    _apply_support_balanced_backfill,
    _compute_scene_arrays,
    _ensure_mmap_cache,
    _load_cached,
    _project,
    _variant_hard_ok,
    _variant_scores_and_candidate,
)


PHASE_ID = "v103_phase4_primitive_affinity_feature"
PLAN_DOC = REPO_ROOT / "docs" / "stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"
DEFAULT_PHASE3_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase3_carrier_reliability_filter_q5c_objlike16384_fast_support_balanced"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase4_primitive_affinity_q5c_support_balanced"
SKETCH_SEED = 10317


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


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


def _normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, eps)


def _variant_by_id(variant_id: str) -> dict[str, Any]:
    for variant in ALL_SUPPORT_BALANCED_VARIANTS:
        if str(variant["variant_id"]) == str(variant_id):
            return dict(variant)
    raise KeyError(f"unsupported support-balanced variant: {variant_id}")


def _score_for_variant(variant: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return _variant_scores_and_candidate(variant, arrays)


def _retained_for_variant(variant: dict[str, Any], arrays: dict[str, np.ndarray], diag: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    scores, candidate = _score_for_variant(variant, arrays)
    n = int(scores.shape[0])
    keep_n = max(1, int(round(float(variant["top_rate"]) * n)))
    candidate_count = int(np.count_nonzero(candidate))
    if candidate_count <= keep_n:
        retained = candidate.copy()
        threshold = float(np.min(scores[retained])) if np.any(retained) else -float("inf")
    else:
        order = np.argpartition(scores, n - keep_n)
        keep = order[n - keep_n :]
        retained = np.zeros((n,), dtype=bool)
        retained[keep] = True
        threshold = float(np.min(scores[keep]))
    if bool(variant.get("hard_veto")):
        hard_ok = _variant_hard_ok(variant, arrays)
        retained &= hard_ok
        candidate &= hard_ok
    retained, added_object, added_boundary = _apply_support_balanced_backfill(
        diag=diag,
        scores=scores,
        candidate=candidate,
        retained=retained,
        min_object_like_support_per_mask=int(variant.get("min_object_like_support_per_mask", 0)),
        min_boundary_support_per_mask=int(variant.get("min_boundary_support_per_mask", 0)),
    )
    return retained, {
        "threshold": threshold,
        "candidate_count": candidate_count,
        "support_backfill_added_object": int(added_object),
        "support_backfill_added_boundary": int(added_boundary),
    }


def _carrier_affinity_risk_weight(variant: dict[str, Any], arrays: dict[str, np.ndarray], mode: str) -> np.ndarray:
    n = int(np.asarray(arrays["carrier_id"]).shape[0])
    weights = np.ones((n,), dtype=np.float32)
    if mode == "base":
        return weights
    if mode not in {"variant_source_penalty", "source_and_competing_penalty"}:
        raise ValueError(f"unsupported affinity_risk_mode: {mode}")
    if variant.get("source_penalty"):
        source = np.asarray(arrays["query_source_code"], dtype=np.int16)
        for code, value in dict(variant["source_penalty"]).items():
            weights[source == int(code)] *= float(value)
    if mode == "source_and_competing_penalty":
        competing = np.asarray(arrays.get("competing_mask_conflict_rate", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
        source_risk = np.asarray(arrays.get("source_risk_score", np.zeros((n,), dtype=np.float32)), dtype=np.float32)
        weights *= np.square(np.clip(1.0 - competing, 0.05, 1.0)).astype(np.float32)
        weights *= np.clip(1.0 - source_risk, 0.05, 1.0).astype(np.float32)
    return np.clip(weights, 0.01, 2.0).astype(np.float32, copy=False)


def _hash_mask(mask_idx: np.ndarray, sketch_dim: int) -> tuple[np.ndarray, np.ndarray]:
    mask_idx = np.asarray(mask_idx, dtype=np.int64)
    bucket = ((mask_idx * 2654435761 + SKETCH_SEED) % int(sketch_dim)).astype(np.int64)
    sign = np.where(((mask_idx * 1103515245 + SKETCH_SEED) % 2) == 0, 1.0, -1.0).astype(np.float32)
    return bucket, sign


def _mask_observations(diag: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    masks = diag["masks"]
    object_like_by_frame = diag["object_like_by_frame"]
    broad_map = diag.get("broad_map")
    object_map = diag.get("object_map")
    rows: list[tuple[int, int, int, int]] = []
    lookup: dict[tuple[int, int], int] = {}
    for fi in range(masks.shape[0]):
        object_labels = {int(v) for v in np.asarray(object_like_by_frame.get(fi, []), dtype=np.int32).tolist()}
        labels, counts = np.unique(masks[fi], return_counts=True)
        frame_area = float(max(masks.shape[1] * masks.shape[2], 1))
        for label, count in zip(labels.tolist(), counts.tolist()):
            label = int(label)
            if label <= 0:
                continue
            area_ratio = float(count) / frame_area
            is_object = int(label in object_labels)
            is_broad = int((not is_object) and area_ratio >= 0.12)
            obs_idx = len(rows)
            lookup[(int(fi), label)] = obs_idx
            if broad_map is not None and label < np.asarray(broad_map).shape[1]:
                is_broad = int(bool(np.asarray(broad_map)[fi, label]))
            if object_map is not None and label < np.asarray(object_map).shape[1]:
                is_object = int(bool(np.asarray(object_map)[fi, label]))
            rows.append((fi, label, is_object, is_broad))
    if not rows:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=bool),
            lookup,
        )
    arr = np.asarray(rows, dtype=np.int32)
    return arr[:, 0], arr[:, 1], arr[:, 2].astype(bool), arr[:, 3].astype(bool), lookup


def _build_incidence(
    *,
    diag: dict[str, Any],
    arrays: dict[str, np.ndarray],
    batch: dict[str, np.ndarray],
    carrier_indices: np.ndarray,
    obs_lookup: dict[tuple[int, int], int],
    variant: dict[str, Any],
    affinity_risk_mode: str,
) -> np.ndarray:
    labels = np.asarray(diag["labels"], dtype=np.int32)
    in_image = np.asarray(diag["in_image"], dtype=bool)
    valid = np.asarray(batch["valid"], dtype=bool)
    visibility = np.asarray(batch["visibility_prob"], dtype=np.float32)
    confidence = np.asarray(batch["confidence_prob"], dtype=np.float32)
    reliability = np.asarray(arrays["reliability_s2"], dtype=np.float32)
    affinity_weight = _carrier_affinity_risk_weight(variant, arrays, affinity_risk_mode)
    rows: list[np.ndarray] = []
    local_carrier = np.arange(carrier_indices.shape[0], dtype=np.int64)
    for fi in range(labels.shape[0]):
        carrier_idx = carrier_indices
        lab = labels[fi, carrier_idx]
        ok = in_image[fi, carrier_idx] & valid[fi, carrier_idx] & (lab > 0)
        if not np.any(ok):
            continue
        local_ok = local_carrier[ok]
        global_ok = carrier_idx[ok]
        lab_ok = lab[ok].astype(np.int32)
        obs_idx = np.asarray([obs_lookup.get((int(fi), int(label)), -1) for label in lab_ok.tolist()], dtype=np.int64)
        obs_ok = obs_idx >= 0
        if not np.any(obs_ok):
            continue
        local_ok = local_ok[obs_ok]
        global_ok = global_ok[obs_ok]
        lab_ok = lab_ok[obs_ok]
        obs_idx = obs_idx[obs_ok]
        b_val = (
            reliability[global_ok].astype(np.float32)
            * visibility[fi, global_ok].astype(np.float32)
            * confidence[fi, global_ok].astype(np.float32)
            * affinity_weight[global_ok].astype(np.float32)
        )
        good = np.isfinite(b_val) & (b_val > 0.0)
        if not np.any(good):
            continue
        frame_col = np.full(int(np.count_nonzero(good)), int(fi), dtype=np.int64)
        rows.append(
            np.stack(
                [
                    local_ok[good].astype(np.float64),
                    obs_idx[good].astype(np.float64),
                    frame_col.astype(np.float64),
                    lab_ok[good].astype(np.float64),
                    b_val[good].astype(np.float64),
                ],
                axis=1,
            )
        )
    if not rows:
        return np.zeros((0, 5), dtype=np.float64)
    return np.concatenate(rows, axis=0).astype(np.float64, copy=False)


def _mask_weights(
    *,
    incidence: np.ndarray,
    mask_count: int,
    mask_frame: np.ndarray,
    mask_is_object: np.ndarray,
    mask_is_broad: np.ndarray,
    visible_reliable_by_frame: np.ndarray,
    specificity_mode: str = "idf_boost",
    specificity_alpha: float = 1.0,
    no_idf: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    if incidence.size:
        support = np.bincount(incidence[:, 1].astype(np.int64), minlength=mask_count).astype(np.float64)
    else:
        support = np.zeros((mask_count,), dtype=np.float64)
    denom = np.maximum(visible_reliable_by_frame[mask_frame.astype(np.int64)].astype(np.float64), 1.0)
    rho = support / denom
    if no_idf:
        idf = np.ones((mask_count,), dtype=np.float64)
    else:
        idf_raw = np.log(1.0 / np.maximum(rho, 1e-6))
        positive_raw = idf_raw[np.isfinite(idf_raw) & (idf_raw > 0.0)]
        p95 = float(np.percentile(positive_raw, 95)) if positive_raw.size else 1.0
        median = float(np.median(positive_raw)) if positive_raw.size else 1.0
        if specificity_mode == "idf_boost":
            idf = 1.0 + idf_raw
        elif specificity_mode == "idf_downweight":
            idf = np.clip(idf_raw / max(p95, 1e-6), 0.05, 1.0)
        elif specificity_mode == "idf_object_preserve_downweight":
            down = np.clip(idf_raw / max(p95, 1e-6), 0.05, 1.0)
            idf = np.where(mask_is_object, 1.0, down)
        elif specificity_mode == "idf_centered_exp":
            idf = np.clip(np.exp(float(specificity_alpha) * (idf_raw - median)), 0.10, 4.0)
        elif specificity_mode == "common_penalty":
            idf = 1.0 / (1.0 + float(specificity_alpha) * rho)
        else:
            raise ValueError(f"unsupported specificity_mode: {specificity_mode}")
    quality = np.where(mask_is_object, 1.0, np.where(mask_is_broad, 0.05, 0.40)).astype(np.float64)
    weights = np.maximum(quality * np.maximum(idf, 0.0), 1e-4).astype(np.float32)
    return weights, {
        "mean_carriers_per_mask": float(np.mean(support)) if support.size else 0.0,
        "mask_support_nonzero_rate": float(np.mean(support > 0)) if support.size else 0.0,
        "specificity_mode": specificity_mode if not no_idf else "none",
        "specificity_alpha": float(specificity_alpha),
    }


def _countsketch(
    incidence: np.ndarray,
    mask_weights: np.ndarray,
    carrier_count: int,
    sketch_dim: int,
    device: torch.device,
    shuffle_mask: bool = False,
) -> tuple[np.ndarray, float]:
    if incidence.size == 0 or carrier_count == 0:
        return np.zeros((carrier_count, sketch_dim), dtype=np.float32), 0.0
    t0 = time.time()
    carrier_idx = torch.as_tensor(incidence[:, 0].astype(np.int64), dtype=torch.long, device=device)
    mask_idx_np = incidence[:, 1].astype(np.int64)
    if shuffle_mask:
        rng = np.random.default_rng(SKETCH_SEED)
        mask_idx_np = rng.integers(0, int(mask_weights.shape[0]), size=mask_idx_np.shape[0], dtype=np.int64)
    mask_idx = torch.as_tensor(mask_idx_np, dtype=torch.long, device=device)
    b_val = torch.as_tensor(incidence[:, 4].astype(np.float32), dtype=torch.float32, device=device)
    weights = torch.as_tensor(mask_weights, dtype=torch.float32, device=device)
    values = torch.sqrt(weights[mask_idx]) * b_val
    bucket = ((mask_idx * 2654435761 + SKETCH_SEED) % int(sketch_dim)).to(torch.long)
    sign = torch.where(((mask_idx * 1103515245 + SKETCH_SEED) % 2) == 0, 1.0, -1.0).to(torch.float32)
    out = torch.zeros((carrier_count, int(sketch_dim)), dtype=torch.float32, device=device)
    out.index_put_((carrier_idx, bucket), sign * values, accumulate=True)
    out = torch.nn.functional.normalize(out, p=2, dim=1, eps=1e-12)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out.detach().cpu().numpy().astype(np.float32, copy=False), time.time() - t0


def _exact_dense_subset(
    incidence: np.ndarray,
    mask_weights: np.ndarray,
    carrier_count: int,
    mask_count: int,
    subset_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    present = np.unique(incidence[:, 0].astype(np.int64)) if incidence.size else np.zeros((0,), dtype=np.int64)
    subset = present[: min(int(subset_size), present.size)]
    dense = np.zeros((subset.shape[0], mask_count), dtype=np.float32)
    if subset.size == 0:
        return subset, dense
    remap = {int(v): i for i, v in enumerate(subset.tolist())}
    keep = np.isin(incidence[:, 0].astype(np.int64), subset)
    for row in incidence[keep]:
        ci = remap[int(row[0])]
        mi = int(row[1])
        dense[ci, mi] += math.sqrt(float(mask_weights[mi])) * float(row[4])
    return subset, _normalize_rows(dense.astype(np.float32, copy=False))


def _pair_error(exact: np.ndarray, sketch: np.ndarray, subset: np.ndarray) -> tuple[float, float]:
    if subset.size < 2:
        return 0.0, 0.0
    sk = sketch[subset]
    cos_exact = exact @ exact.T
    cos_sketch = sk @ sk.T
    err = np.abs(cos_exact - cos_sketch)
    return float(np.percentile(err, 95)), float(np.max(err))


def _sample_pairs(
    incidence: np.ndarray,
    mask_is_object: np.ndarray,
    mask_is_broad: np.ndarray,
    mask_frame: np.ndarray,
    max_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    if incidence.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    carriers_by_mask_all: dict[int, np.ndarray] = {}
    carriers_by_mask_object: dict[int, np.ndarray] = {}
    for mask_idx in np.unique(incidence[:, 1].astype(np.int64)).tolist():
        carriers = np.unique(incidence[incidence[:, 1].astype(np.int64) == int(mask_idx), 0].astype(np.int64))
        if carriers.size:
            carriers_by_mask_all[int(mask_idx)] = carriers[: min(carriers.size, 256)]
        if bool(mask_is_object[int(mask_idx)]) and carriers.size >= 2:
            carriers_by_mask_object[int(mask_idx)] = carriers[: min(carriers.size, 256)]
    pos: list[tuple[int, int]] = []
    for carriers in carriers_by_mask_object.values():
        anchor_count = min(8, carriers.size)
        for i in range(anchor_count):
            j = min(carriers.size - 1, i + 1)
            if i == j:
                continue
            pos.append((int(carriers[i]), int(carriers[j])))
            if len(pos) >= max_pairs:
                break
        if len(pos) >= max_pairs:
            break
    neg_object: list[tuple[int, int]] = []
    neg_broad: list[tuple[int, int]] = []
    object_quota = max(1, int(round(float(max_pairs) * 0.50)))
    broad_quota = max(1, int(max_pairs) - object_quota)
    object_masks_by_frame: dict[int, list[int]] = {}
    broad_masks_by_frame: dict[int, list[int]] = {}
    for mask_idx, frame in enumerate(mask_frame.astype(np.int64).tolist()):
        if bool(mask_is_object[mask_idx]) and mask_idx in carriers_by_mask_object:
            object_masks_by_frame.setdefault(int(frame), []).append(mask_idx)
        if bool(mask_is_broad[mask_idx]) and mask_idx in carriers_by_mask_all:
            broad_masks_by_frame.setdefault(int(frame), []).append(mask_idx)
    for frame, masks in object_masks_by_frame.items():
        for a_pos, a in enumerate(masks):
            for b in masks[a_pos + 1 :]:
                if len(neg_object) >= object_quota:
                    break
                ca = carriers_by_mask_object.get(a, np.zeros((0,), dtype=np.int64))
                cb = carriers_by_mask_object.get(b, np.zeros((0,), dtype=np.int64))
                if ca.size and cb.size:
                    pair_count = min(8, ca.size, cb.size)
                    for k in range(pair_count):
                        neg_object.append((int(ca[k]), int(cb[k])))
                        if len(neg_object) >= object_quota:
                            break
            if len(neg_object) >= object_quota:
                break
        for a in masks:
            if len(neg_broad) >= broad_quota:
                break
            ca = carriers_by_mask_object.get(a, np.zeros((0,), dtype=np.int64))
            for b in broad_masks_by_frame.get(int(frame), []):
                cb = carriers_by_mask_all.get(b, np.zeros((0,), dtype=np.int64))
                if ca.size and cb.size:
                    pair_count = min(8, ca.size, cb.size)
                    for k in range(pair_count):
                        neg_broad.append((int(ca[k]), int(cb[k])))
                        if len(neg_broad) >= broad_quota:
                            break
                if len(neg_broad) >= broad_quota:
                    break
    if len(neg_broad) < broad_quota:
        extra = object_quota + (broad_quota - len(neg_broad))
        neg = neg_object[:extra] + neg_broad
    else:
        neg = neg_object[:object_quota] + neg_broad[:broad_quota]
    return np.asarray(pos, dtype=np.int64), np.asarray(neg[: int(max_pairs)], dtype=np.int64)


def _margin(feature: np.ndarray, pos_pairs: np.ndarray, neg_pairs: np.ndarray) -> dict[str, float]:
    def vals(pairs: np.ndarray) -> np.ndarray:
        if pairs.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.sum(feature[pairs[:, 0]] * feature[pairs[:, 1]], axis=1)

    pos = vals(pos_pairs)
    neg = vals(neg_pairs)
    pos_mean = float(np.mean(pos)) if pos.size else 0.0
    neg_mean = float(np.mean(neg)) if neg.size else 0.0
    return {
        "pseudo_positive_affinity_mean": pos_mean,
        "hard_negative_affinity_mean": neg_mean,
        "pseudo_positive_minus_hard_negative_margin": pos_mean - neg_mean,
        "pseudo_positive_pair_count": int(pos.size),
        "hard_negative_pair_count": int(neg.size),
    }


def _scene_specs(scene0011_phase2: str, scene0050_phase2: str) -> dict[str, dict[str, Path]]:
    audit = STREAM3D_ROOT / "outputs/audit"
    return {
        "scene0011_00": {
            "phase2_root": _project(scene0011_phase2),
            "semantic_npz": audit / "v91_radio_mask_features_npz_scene0011/mask_features.npz",
            "semantic_rows": audit / "v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
        },
        "scene0050_00": {
            "phase2_root": _project(scene0050_phase2),
            "semantic_npz": audit / "v91_radio_mask_features_npz_scene0050/mask_features.npz",
            "semantic_rows": audit / "v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
        },
    }


def _run_scene(scene: str, spec: dict[str, Path], args: argparse.Namespace, selected_variant_id: str, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scene_t0 = time.time()
    scene_out = _project(args.output_root) / scene
    scene_out.mkdir(parents=True, exist_ok=True)
    diag, _unused_a, _unused_b, arrays = _compute_scene_arrays(scene, spec, scene_out, int(args.cupy_device_id))
    variant = _variant_by_id(selected_variant_id)
    retained, retain_meta = _retained_for_variant(variant, arrays, diag)
    carrier_indices = np.flatnonzero(retained).astype(np.int64)
    cache_dir, _manifest = _ensure_mmap_cache(spec["phase2_root"])
    batch = _load_cached(cache_dir)

    mask_frame, mask_label, mask_is_object, mask_is_broad, obs_lookup = _mask_observations(diag)
    incidence = _build_incidence(
        diag=diag,
        arrays=arrays,
        batch=batch,
        carrier_indices=carrier_indices,
        obs_lookup=obs_lookup,
        variant=variant,
        affinity_risk_mode=str(args.affinity_risk_mode),
    )
    visible_reliable = np.zeros((len(diag["frame_ids"]),), dtype=np.int64)
    for fi in range(len(visible_reliable)):
        visible_reliable[fi] = int(np.count_nonzero(diag["in_image"][fi, carrier_indices]))
    weights, weight_meta = _mask_weights(
        incidence=incidence,
        mask_count=int(mask_frame.shape[0]),
        mask_frame=mask_frame,
        mask_is_object=mask_is_object,
        mask_is_broad=mask_is_broad,
        visible_reliable_by_frame=visible_reliable,
        specificity_mode=str(args.specificity_mode),
        specificity_alpha=float(args.specificity_alpha),
        no_idf=False,
    )
    no_idf_weights, _no_idf_meta = _mask_weights(
        incidence=incidence,
        mask_count=int(mask_frame.shape[0]),
        mask_frame=mask_frame,
        mask_is_object=mask_is_object,
        mask_is_broad=mask_is_broad,
        visible_reliable_by_frame=visible_reliable,
        specificity_mode=str(args.specificity_mode),
        specificity_alpha=float(args.specificity_alpha),
        no_idf=True,
    )
    feature, sketch_runtime = _countsketch(incidence, weights, len(carrier_indices), int(args.sketch_dim), device)
    feature_no_idf, no_idf_runtime = _countsketch(incidence, no_idf_weights, len(carrier_indices), int(args.sketch_dim), device)
    feature_shuffled, shuffled_runtime = _countsketch(incidence, weights, len(carrier_indices), int(args.sketch_dim), device, shuffle_mask=True)
    subset, exact = _exact_dense_subset(incidence, weights, len(carrier_indices), int(mask_frame.shape[0]), int(args.exact_subset_size))
    p95_error, max_error = _pair_error(exact, feature, subset)

    if incidence.size:
        mass = np.sqrt(weights[incidence[:, 1].astype(np.int64)]) * np.abs(incidence[:, 4].astype(np.float32))
        obs_idx = incidence[:, 1].astype(np.int64)
        total_mass = float(np.sum(mass))
        broad_mass = float(np.sum(mass[mask_is_broad[obs_idx]])) if total_mass else 0.0
        object_mass = float(np.sum(mass[mask_is_object[obs_idx]])) if total_mass else 0.0
        masks_per_carrier = np.bincount(incidence[:, 0].astype(np.int64), minlength=len(carrier_indices))
    else:
        total_mass = broad_mass = object_mass = 0.0
        masks_per_carrier = np.zeros((len(carrier_indices),), dtype=np.int64)
    bucket, _sign = _hash_mask(np.arange(mask_frame.shape[0], dtype=np.int64), int(args.sketch_dim))
    bucket_load_all = np.bincount(bucket, minlength=int(args.sketch_dim)).astype(np.float64)
    bucket_load = bucket_load_all[bucket_load_all > 0]
    bucket_mean = float(np.mean(bucket_load)) if bucket_load.size else 0.0
    bucket_p95 = float(np.percentile(bucket_load, 95)) if bucket_load.size else 0.0
    collision_mass = float(np.sum(np.maximum(bucket_load - 1.0, 0.0)) / max(mask_frame.shape[0], 1))
    valid_rate = float(np.mean(np.linalg.norm(feature, axis=1) > 0.0)) if feature.size else 0.0

    pos_pairs, neg_pairs = _sample_pairs(incidence, mask_is_object, mask_is_broad, mask_frame, int(args.control_pair_count))
    real_margin = _margin(feature, pos_pairs, neg_pairs)
    no_idf_margin = _margin(feature_no_idf, pos_pairs, neg_pairs)
    shuffled_margin = _margin(feature_shuffled, pos_pairs, neg_pairs)

    control_rows = [
        {
            "schema_version": "stream4d_v103_phase4_control_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "control_id": "filtered_real_idf",
            **real_margin,
        },
        {
            "schema_version": "stream4d_v103_phase4_control_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "control_id": "C1_no_IDF",
            **no_idf_margin,
        },
        {
            "schema_version": "stream4d_v103_phase4_control_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "control_id": "C3_shuffled_mask_id",
            **shuffled_margin,
        },
    ]

    incidence_path = scene_out / "primitive_incidence_sparse.pt"
    feature_path = scene_out / "primitive_affinity_feature.pt"
    torch.save(
        {
            "schema_version": "stream4d_v103_phase4_primitive_incidence_sparse_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "carrier_id": torch.as_tensor(np.asarray(batch["carrier_id"])[carrier_indices], dtype=torch.int64),
            "carrier_local_index": torch.as_tensor(incidence[:, 0].astype(np.int64), dtype=torch.int64),
            "mask_observation_index": torch.as_tensor(incidence[:, 1].astype(np.int64), dtype=torch.int64),
            "frame_local_index": torch.as_tensor(incidence[:, 2].astype(np.int64), dtype=torch.int64),
            "mask_id": torch.as_tensor(incidence[:, 3].astype(np.int64), dtype=torch.int64),
            "B_ia": torch.as_tensor(incidence[:, 4].astype(np.float32), dtype=torch.float32),
            "mask_frame": torch.as_tensor(mask_frame, dtype=torch.int64),
            "mask_label": torch.as_tensor(mask_label, dtype=torch.int64),
            "mask_is_object_like": torch.as_tensor(mask_is_object, dtype=torch.bool),
            "mask_is_broad": torch.as_tensor(mask_is_broad, dtype=torch.bool),
            "mask_weight": torch.as_tensor(weights, dtype=torch.float32),
            "selected_phase3_variant": selected_variant_id,
            "affinity_risk_mode": str(args.affinity_risk_mode),
            "B_ia_formula": "reliability_s2 * visibility_prob * confidence_prob * carrier_affinity_risk_weight",
            "uses_gt": False,
            "uses_future": False,
        },
        incidence_path,
    )
    torch.save(
        {
            "schema_version": "stream4d_v103_phase4_primitive_affinity_feature_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "carrier_id": torch.as_tensor(np.asarray(batch["carrier_id"])[carrier_indices], dtype=torch.int64),
            "feature": torch.as_tensor(feature, dtype=torch.float16),
            "feature_norm_source_dtype": "float32",
            "sketch_dim": int(args.sketch_dim),
            "sketch_seed": SKETCH_SEED,
            "selected_phase3_variant": selected_variant_id,
            "affinity_risk_mode": str(args.affinity_risk_mode),
            "uses_gt": False,
            "uses_future": False,
        },
        feature_path,
    )

    metric = {
        "schema_version": "stream4d_v103_phase4_feature_metric_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "selected_phase3_variant": selected_variant_id,
        "specificity_mode": str(args.specificity_mode),
        "specificity_alpha": float(args.specificity_alpha),
        "affinity_risk_mode": str(args.affinity_risk_mode),
        "reliable_carrier_count": int(len(carrier_indices)),
        "incidence_row_count": int(incidence.shape[0]),
        "mask_observation_count": int(mask_frame.shape[0]),
        "mean_masks_per_carrier": float(np.mean(masks_per_carrier)) if masks_per_carrier.size else 0.0,
        "mean_carriers_per_mask": weight_meta["mean_carriers_per_mask"],
        "feature_valid_rate": valid_rate,
        "sketch_dim": int(args.sketch_dim),
        "sketch_bucket_load_mean": bucket_mean,
        "sketch_bucket_load_p95": bucket_p95,
        "sketch_bucket_load_policy": "occupied_buckets_only",
        "sketch_collision_mass": collision_mass,
        "exact_vs_sketch_cosine_p95_error": p95_error,
        "exact_vs_sketch_cosine_max_error": max_error,
        "broad_mask_feature_contribution_ratio": broad_mass / max(total_mass, 1e-12),
        "object_like_mask_feature_contribution_ratio": object_mass / max(total_mass, 1e-12),
        "pseudo_positive_minus_hard_negative_margin": real_margin["pseudo_positive_minus_hard_negative_margin"],
        "no_idf_margin": no_idf_margin["pseudo_positive_minus_hard_negative_margin"],
        "shuffled_margin": shuffled_margin["pseudo_positive_minus_hard_negative_margin"],
        "uses_gt_for_feature": False,
        "uses_future": False,
    }
    gate_specs = [
        ("feature_valid_rate_ge_0p95", metric["feature_valid_rate"] >= 0.95, metric["feature_valid_rate"], 0.95),
        ("exact_vs_sketch_cosine_p95_error_le_0p02", metric["exact_vs_sketch_cosine_p95_error"] <= 0.02, metric["exact_vs_sketch_cosine_p95_error"], 0.02),
        ("sketch_bucket_load_p95_le_5x_mean", metric["sketch_bucket_load_p95"] <= 5.0 * max(metric["sketch_bucket_load_mean"], 1e-9), metric["sketch_bucket_load_p95"], 5.0 * max(metric["sketch_bucket_load_mean"], 1e-9)),
        ("broad_mask_feature_contribution_ratio_le_0p30", metric["broad_mask_feature_contribution_ratio"] <= 0.30, metric["broad_mask_feature_contribution_ratio"], 0.30),
        ("object_like_mask_feature_contribution_ratio_ge_0p40", metric["object_like_mask_feature_contribution_ratio"] >= 0.40, metric["object_like_mask_feature_contribution_ratio"], 0.40),
        ("real_margin_gt_no_idf", metric["pseudo_positive_minus_hard_negative_margin"] > metric["no_idf_margin"], metric["pseudo_positive_minus_hard_negative_margin"], f">{metric['no_idf_margin']}"),
        ("real_margin_gt_shuffled", metric["pseudo_positive_minus_hard_negative_margin"] > metric["shuffled_margin"], metric["pseudo_positive_minus_hard_negative_margin"], f">{metric['shuffled_margin']}"),
    ]
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for name, ok, observed, required in gate_specs:
        gate_rows.append(
            {
                "schema_version": "stream4d_v103_phase4_gate_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "gate_name": name,
                "pass": bool(ok),
                "observed": observed,
                "required": required,
            }
        )
        if not ok:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_phase4_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "failure_id": name,
                    "severity": "blocking",
                    "evidence": f"observed={observed} required={required}",
                    "repair_direction": "Follow Phase4 repair ladder: increase sketch_dim/hash seed for sketch error, strengthen IDF/broad quality downweight for broad contribution, or revisit Phase3 if filtered feature loses to controls.",
                }
            )
    artifact_rows = [
        {
            "schema_version": "stream4d_v103_phase4_artifact_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "role": "primitive_incidence_sparse",
            "path": _rel(incidence_path),
            "exists": incidence_path.exists(),
            "size_bytes": incidence_path.stat().st_size if incidence_path.exists() else 0,
        },
        {
            "schema_version": "stream4d_v103_phase4_artifact_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "role": "primitive_affinity_feature",
            "path": _rel(feature_path),
            "exists": feature_path.exists(),
            "size_bytes": feature_path.stat().st_size if feature_path.exists() else 0,
        },
    ]
    perf_rows = [
        {
            "schema_version": "stream4d_v103_phase4_performance_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "carrier_count": int(len(carrier_indices)),
            "incidence_row_count": int(incidence.shape[0]),
            "sketch_runtime_sec": sketch_runtime,
            "no_idf_sketch_runtime_sec": no_idf_runtime,
            "shuffled_sketch_runtime_sec": shuffled_runtime,
            "scene_runtime_sec": time.time() - scene_t0,
            "device": str(device),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
    ]
    _write_json(scene_out / "scene_summary.json", {"metric": metric, "failure_count": len(failure_rows), "outputs": {"incidence": _rel(incidence_path), "feature": _rel(feature_path)}})
    return metric, gate_rows, failure_rows, control_rows, artifact_rows + perf_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase4 primitive-mask incidence and CountSketch primitive affinity feature.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase3-root", default=str(DEFAULT_PHASE3_ROOT))
    parser.add_argument("--scene0011-phase2-root", default="outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0011_first32")
    parser.add_argument("--scene0050-phase2-root", default="outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0050_first32")
    parser.add_argument("--scene", choices=["all", "scene0011_00", "scene0050_00"], default="all")
    parser.add_argument("--scene0011-selected-variant-id", default="")
    parser.add_argument("--scene0050-selected-variant-id", default="")
    parser.add_argument("--sketch-dim", type=int, default=2048)
    parser.add_argument("--exact-subset-size", type=int, default=512)
    parser.add_argument("--control-pair-count", type=int, default=4096)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument(
        "--affinity-risk-mode",
        choices=["base", "variant_source_penalty", "source_and_competing_penalty"],
        default="base",
    )
    parser.add_argument(
        "--specificity-mode",
        choices=["idf_boost", "idf_downweight", "idf_object_preserve_downweight", "idf_centered_exp", "common_penalty"],
        default="idf_boost",
    )
    parser.add_argument("--specificity-alpha", type=float, default=1.0)
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    phase3_root = _project(args.phase3_root)
    phase3_summary = json.loads((phase3_root / "summary.json").read_text(encoding="utf-8"))
    selected_by_scene = {str(k): str(v) for k, v in dict(phase3_summary["selected_variant_by_scene"]).items()}
    selected_override_by_scene = {
        "scene0011_00": str(args.scene0011_selected_variant_id),
        "scene0050_00": str(args.scene0050_selected_variant_id),
    }
    selected_override_by_scene = {scene: variant for scene, variant in selected_override_by_scene.items() if variant}
    selected_by_scene.update(selected_override_by_scene)
    specs = _scene_specs(args.scene0011_phase2_root, args.scene0050_phase2_root)
    scene_ids = list(specs) if args.scene == "all" else [args.scene]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    perf_rows: list[dict[str, Any]] = []
    for scene in scene_ids:
        metric, gates, failures, controls, artifacts_and_perf = _run_scene(
            scene=scene,
            spec=specs[scene],
            args=args,
            selected_variant_id=selected_by_scene[scene],
            device=device,
        )
        metric_rows.append(metric)
        gate_rows.extend(gates)
        failure_rows.extend(failures)
        control_rows.extend(controls)
        artifact_rows.extend([row for row in artifacts_and_perf if row["schema_version"].endswith("artifact_row_v1")])
        perf_rows.extend([row for row in artifacts_and_perf if row["schema_version"].endswith("performance_row_v1")])
    _write_csv(out / "primitive_feature_metric_rows.csv", metric_rows)
    _write_csv(out / "sketch_parity_rows.csv", [row for row in gate_rows if "sketch" in str(row["gate_name"]) or "feature_valid" in str(row["gate_name"])])
    _write_csv(out / "feature_control_rows.csv", control_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "artifact_rows.csv", artifact_rows)
    _write_csv(out / "performance_rows.csv", perf_rows)
    decision = "PASS_ENTER_PHASE5_MASK_LEVEL_POOLING" if not failure_rows else "NO_GO_REPAIR_PHASE4_PRIMITIVE_AFFINITY"
    summary = {
        "schema_version": "stream4d_v103_phase4_primitive_affinity_feature_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "phase4_pass": not failure_rows,
        "failure_count": len(failure_rows),
        "scene_ids": scene_ids,
        "phase3_root": _rel(phase3_root),
        "selected_phase3_variant_by_scene": selected_by_scene,
        "selected_variant_override_by_scene": selected_override_by_scene,
        "sketch_dim": int(args.sketch_dim),
        "sketch_seed": SKETCH_SEED,
        "specificity_mode": str(args.specificity_mode),
        "specificity_alpha": float(args.specificity_alpha),
        "affinity_risk_mode": str(args.affinity_risk_mode),
        "uses_gt_for_feature": False,
        "uses_future": False,
        "truthfulness_note": "Phase4 constructs carrier-level primitive affinity features from GT-free reliable carriers. It does not compute AP or perform mask clustering.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "primitive_feature_metric_rows": _rel(out / "primitive_feature_metric_rows.csv"),
            "sketch_parity_rows": _rel(out / "sketch_parity_rows.csv"),
            "feature_control_rows": _rel(out / "feature_control_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
            "performance_rows": _rel(out / "performance_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
