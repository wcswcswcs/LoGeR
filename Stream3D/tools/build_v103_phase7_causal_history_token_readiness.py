#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from build_v103_phase3_fast_carrier_reliability_filter import (  # noqa: E402
    ALL_SUPPORT_BALANCED_VARIANTS,
    _variant_hard_ok,
    _variant_scores_and_candidate,
)


PHASE_ID = "v103_phase7_causal_history_token_readiness"
PLAN_DOC = REPO_ROOT / "docs" / "stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase7_causal_history_token_readiness_r1"
DEFAULT_HISTORY_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase6d_f2_skeleton_affinity_merge_phase9n_r8_i14_e3_veto_ratio100"
DEFAULT_PHASE3_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase3_carrier_reliability_filter_q5c_objlike16384_c0001_cap24576_competing_repair5"
DEFAULT_SCENE0011_PHASE2 = STREAM3D_ROOT / "outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576"
DEFAULT_SCENE0050_PHASE2 = STREAM3D_ROOT / "outputs/audit/v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576"
SEMANTIC_NPZ = {
    "scene0011_00": STREAM3D_ROOT / "outputs/audit/v91_radio_mask_features_npz_scene0011/mask_features.npz",
    "scene0050_00": STREAM3D_ROOT / "outputs/audit/v91_radio_mask_features_npz_scene0050/mask_features.npz",
}
SEMANTIC_ROWS = {
    "scene0011_00": STREAM3D_ROOT / "outputs/audit/v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
    "scene0050_00": STREAM3D_ROOT / "outputs/audit/v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
}
PHASE2_MMAP_BATCH_KEYS = (
    "carrier_id",
    "uv_pred",
    "xyz_ref",
    "valid",
    "visibility_prob",
    "confidence_prob",
)


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: str | Path) -> str:
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_obs(obs_id: str) -> tuple[str, int, int]:
    scene, frame, mask = str(obs_id).split(":")
    return scene, int(frame), int(mask)


def _load_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int32)


def _normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, eps)


def _variant_by_id(variant_id: str) -> dict[str, Any]:
    for variant in ALL_SUPPORT_BALANCED_VARIANTS:
        if str(variant.get("variant_id")) == str(variant_id):
            return dict(variant)
    raise KeyError(f"unsupported Phase3 variant id: {variant_id}")


def _scene_phase2_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }


def _scene_phase3_roots(args: argparse.Namespace) -> dict[str, Path]:
    common = _project(args.phase3_root)
    return {
        "scene0011_00": _project(args.scene0011_phase3_root) if str(args.scene0011_phase3_root).strip() else common,
        "scene0050_00": _project(args.scene0050_phase3_root) if str(args.scene0050_phase3_root).strip() else common,
    }


def _scene_da3_supplement_roots(args: argparse.Namespace) -> dict[str, Path | None]:
    roots: dict[str, Path | None] = {}
    for scene, value in {
        "scene0011_00": args.scene0011_da3_overlap_supplement_root,
        "scene0050_00": args.scene0050_da3_overlap_supplement_root,
    }.items():
        roots[scene] = _project(value) if str(value).strip() else None
    return roots


def _load_phase2_scene(root: Path) -> tuple[dict[str, Any], np.ndarray, Any]:
    summary = _read_json(root / "summary.json")
    frame_ids = [int(v) for v in summary["frame_ids"]]
    mask_root = _project(summary["mask_root"])
    masks = np.stack([_load_mask(mask_root / f"{frame_id}.png") for frame_id in frame_ids], axis=0)
    cache_dir = root / "carrier_batch_mmap_cache"
    if cache_dir.exists() and all((cache_dir / f"{key}.npy").exists() for key in PHASE2_MMAP_BATCH_KEYS):
        batch = {key: np.load(cache_dir / f"{key}.npy", mmap_mode="r") for key in PHASE2_MMAP_BATCH_KEYS}
        return summary, masks, batch
    batch = np.load(root / "carrier_batch.npz", allow_pickle=False)
    return summary, masks, batch


def _load_semantic(scene: str) -> tuple[dict[tuple[int, int], int], np.ndarray, dict[str, Any]]:
    pack = np.load(SEMANTIC_NPZ[scene], allow_pickle=False)
    features = _normalize_rows(pack["features"].astype(np.float32))
    frame_ids = pack["frame_id"].astype(np.int64)
    mask_ids = pack["mask_id"].astype(np.int64)
    lookup = {(int(f), int(m)): int(i) for i, (f, m) in enumerate(zip(frame_ids.tolist(), mask_ids.tolist()))}
    rng = np.random.default_rng(10307)
    pair_count = min(8192, max(0, features.shape[0] * 2))
    if features.shape[0] >= 2 and pair_count > 0:
        a = rng.integers(0, features.shape[0], size=pair_count)
        b = rng.integers(0, features.shape[0], size=pair_count)
        neq = a != b
        sims = np.sum(features[a[neq]] * features[b[neq]], axis=1)
    else:
        sims = np.asarray([], dtype=np.float32)
    constants = {
        "semantic_source": _rel(SEMANTIC_NPZ[scene]),
        "semantic_rows": _rel(SEMANTIC_ROWS[scene]),
        "mu_sem_used": float(np.mean(sims)) if sims.size else 0.0,
        "mu_sem_random_pair_count": int(sims.shape[0]),
    }
    return lookup, features, constants


def _history_variant_id(history_root: Path, requested: str) -> str:
    requested = str(requested).strip()
    if requested:
        return requested
    summary = _read_json(history_root / "summary.json")
    return str(summary["best_variant_id"])


def _load_history_objects(history_root: Path, variant_id: str) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows = pd.read_csv(history_root / "merge_selected_rows.csv")
    rows = rows[(rows["variant_id"].astype(str) == str(variant_id)) & (~rows["uses_gt_for_prediction"].astype(bool))]
    objects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    object_rows: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        scene = str(row["scene_id"])
        hist_id = str(row["object_id"])
        record = {
            "scene_id": scene,
            "history_id": hist_id,
            "frame_id": int(row["frame_id"]),
            "frame_local_index": int(row.get("frame_local_index", -1)),
            "mask_id": int(row["selected_mask_id"]),
            "score": float(row.get("score", row.get("object_score", 0.0))),
            "support_count": int(float(row.get("support_count", 0) or 0)),
        }
        objects[scene].append(record)
    for scene, scene_rows in sorted(objects.items()):
        by_obj: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scene_rows:
            by_obj[str(row["history_id"])].append(row)
        objects[scene] = []
        for hist_id, obj_rows in sorted(by_obj.items()):
            obj_rows = sorted(obj_rows, key=lambda r: (int(r["frame_id"]), int(r["mask_id"])))
            objects[scene].extend(obj_rows)
            frames = sorted({int(r["frame_id"]) for r in obj_rows})
            object_rows.append(
                {
                    "schema_version": "stream4d_v103_phase7_history_object_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "history_id": hist_id,
                    "history_variant_id": variant_id,
                    "frame_count": len(frames),
                    "mask_observation_count": len(obj_rows),
                    "first_frame_id": frames[0] if frames else "",
                    "last_frame_id": frames[-1] if frames else "",
                    "uses_gt_for_prediction": False,
                }
            )
    return dict(objects), object_rows


def _group_history(
    scene_rows: list[dict[str, Any]],
    semantic_lookup: dict[tuple[int, int], int],
) -> tuple[list[str], dict[str, list[tuple[int, int]]], dict[str, np.ndarray]]:
    by_obj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
        by_obj[str(row["history_id"])].append(row)
    history_ids = sorted(by_obj)
    masks_by_obj: dict[str, list[tuple[int, int]]] = {}
    feature_idx_by_obj: dict[str, np.ndarray] = {}
    for hist_id in history_ids:
        pairs = sorted({(int(r["frame_id"]), int(r["mask_id"])) for r in by_obj[hist_id]})
        masks_by_obj[hist_id] = pairs
        feat_idx = [semantic_lookup[pair] for pair in pairs if pair in semantic_lookup]
        feature_idx_by_obj[hist_id] = np.asarray(sorted(set(feat_idx)), dtype=np.int64)
    return history_ids, masks_by_obj, feature_idx_by_obj


def _reconstruct_retained(
    *,
    phase3_root: Path,
    scene: str,
    max_retained_carriers: int,
    variant_override: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    summary = _read_json(phase3_root / "summary.json")
    variant_id = str(variant_override).strip() or str(summary["selected_variant_by_scene"][scene])
    variant = _variant_by_id(variant_id)
    df = pd.read_parquet(phase3_root / scene / "carrier_reliability_rows.parquet")
    arrays = {col: df[col].to_numpy(copy=False) for col in df.columns if col not in {"schema_version", "phase_id", "scene_id"}}
    scores, candidate = _variant_scores_and_candidate(variant, arrays)
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
    retained_idx = np.flatnonzero(retained).astype(np.int64)
    capped = False
    if int(max_retained_carriers) > 0 and retained_idx.shape[0] > int(max_retained_carriers):
        order = np.argsort(np.asarray(arrays["reliability_s2"], dtype=np.float32)[retained_idx], kind="mergesort")
        retained_idx = retained_idx[order[-int(max_retained_carriers) :]]
        capped = True
    metric_df = pd.read_csv(phase3_root / "carrier_filter_metric_rows.csv")
    metric = metric_df[(metric_df["scene_id"].astype(str) == scene) & (metric_df["variant_id"].astype(str) == variant_id)]
    metric_retained = int(metric.iloc[0]["retained_carrier_count"]) if not metric.empty else ""
    return retained_idx, {
        "selected_phase3_variant_id": variant_id,
        "retained_reconstruction": "top_rate_candidate_hard_veto_pre_support_backfill",
        "reconstructed_threshold": threshold,
        "candidate_count_after_gtfree_prefilter": candidate_count,
        "reconstructed_retained_count": int(retained_idx.shape[0]),
        "phase3_metric_retained_count": metric_retained,
        "phase3_metric_root": _rel(phase3_root),
        "max_retained_carriers": int(max_retained_carriers),
        "max_retained_cap_applied": capped,
        "truthfulness_note": (
            "Phase7 uses the precision-anchor retained set reconstructed from Phase3 parquet with candidate filters, "
            "top-rate threshold, and hard veto. Phase3 support-backfill carriers are intentionally not added here."
        ),
    }


def _project_labels_for_indices(
    *,
    batch: np.lib.npyio.NpzFile,
    masks: np.ndarray,
    carrier_indices: np.ndarray,
    cupy_device_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, float]:
    t0 = time.time()
    frame_count = int(batch["valid"].shape[0])
    height, width = masks.shape[1:]
    n = int(carrier_indices.shape[0])
    labels = np.full((frame_count, n), -1, dtype=np.int32)
    ok = np.zeros((frame_count, n), dtype=bool)
    weights = np.zeros((frame_count, n), dtype=np.float32)
    xs_all = np.zeros((frame_count, n), dtype=np.int16)
    try:
        import cupy as cp
    except Exception:
        cp = None  # type: ignore[assignment]
    if cp is not None:
        backend = "cupy_retained_framewise_projection"
        with cp.cuda.Device(int(cupy_device_id)):
            idx_g = cp.asarray(carrier_indices.astype(np.int64))
            for fi in range(frame_count):
                uv = cp.asarray(batch["uv_pred"][fi])[idx_g]
                xyz = cp.asarray(batch["xyz_ref"][fi])[idx_g]
                valid = cp.asarray(batch["valid"][fi])[idx_g]
                visibility = cp.asarray(batch["visibility_prob"][fi])[idx_g]
                confidence = cp.asarray(batch["confidence_prob"][fi])[idx_g]
                mask_g = cp.asarray(masks[fi], dtype=cp.int32)
                finite = cp.isfinite(uv).all(axis=1) & cp.isfinite(xyz).all(axis=1)
                in_img = valid & finite & (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
                xs = cp.rint(cp.clip(uv[:, 0], 0.0, 1.0) * float(max(width - 1, 1))).astype(cp.int32)
                ys = cp.rint(cp.clip(uv[:, 1], 0.0, 1.0) * float(max(height - 1, 1))).astype(cp.int32)
                lab = cp.full((n,), -1, dtype=cp.int32)
                lab[in_img] = mask_g[ys[in_img], xs[in_img]]
                labels[fi] = cp.asnumpy(lab)
                ok[fi] = cp.asnumpy(in_img)
                weights[fi] = cp.asnumpy(cp.where(in_img, visibility * confidence, 0.0)).astype(np.float32, copy=False)
                xs_all[fi] = cp.asnumpy(xs).astype(np.int16, copy=False)
            cp.cuda.Stream.null.synchronize()
        return labels, ok, weights, xs_all, backend, time.time() - t0

    backend = "numpy_retained_framewise_projection_fallback"
    for fi in range(frame_count):
        uv = np.asarray(batch["uv_pred"][fi, carrier_indices], dtype=np.float32)
        xyz = np.asarray(batch["xyz_ref"][fi, carrier_indices], dtype=np.float32)
        valid = np.asarray(batch["valid"][fi, carrier_indices], dtype=bool)
        visibility = np.asarray(batch["visibility_prob"][fi, carrier_indices], dtype=np.float32)
        confidence = np.asarray(batch["confidence_prob"][fi, carrier_indices], dtype=np.float32)
        finite = np.isfinite(uv).all(axis=1) & np.isfinite(xyz).all(axis=1)
        in_img = valid & finite & (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
        xs = np.rint(np.clip(uv[:, 0], 0.0, 1.0) * float(max(width - 1, 1))).astype(np.int32)
        ys = np.rint(np.clip(uv[:, 1], 0.0, 1.0) * float(max(height - 1, 1))).astype(np.int32)
        labels[fi, in_img] = masks[fi, ys[in_img], xs[in_img]]
        ok[fi] = in_img
        weights[fi] = np.where(in_img, visibility * confidence, 0.0).astype(np.float32, copy=False)
        xs_all[fi] = xs.astype(np.int16, copy=False)
    return labels, ok, weights, xs_all, backend, time.time() - t0


def _overlap_support(
    *,
    frame_ids: list[int],
    labels: np.ndarray,
    ok: np.ndarray,
    weights: np.ndarray,
    history_ids: list[str],
    masks_by_obj: dict[str, list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = labels.shape[1]
    m = len(history_ids)
    support = np.zeros((n, m), dtype=np.float32)
    hit_any = np.zeros((n, m), dtype=bool)
    owner_hit_count = np.zeros((n,), dtype=np.int16)
    overlap_frames = []
    frame_to_local = {int(frame_id): int(i) for i, frame_id in enumerate(frame_ids)}
    owner_by_frame_label: dict[tuple[int, int], int] = {}
    for obj_idx, hist_id in enumerate(history_ids):
        for frame_id, mask_id in masks_by_obj[hist_id]:
            if int(frame_id) in frame_to_local:
                owner_by_frame_label[(int(frame_id), int(mask_id))] = int(obj_idx)
    overlap_frame_ids = sorted({int(frame) for frame, _mask in owner_by_frame_label})
    denom = np.zeros((n,), dtype=np.float32)
    for frame_id in overlap_frame_ids:
        fi = frame_to_local[int(frame_id)]
        overlap_frames.append(int(frame_id))
        lab = labels[fi]
        good = ok[fi] & (lab > 0)
        w = weights[fi]
        denom += np.where(good, w, 0.0).astype(np.float32, copy=False)
        if not np.any(good):
            continue
        for mask_id in sorted({mask for f, mask in owner_by_frame_label if f == int(frame_id)}):
            obj_idx = owner_by_frame_label[(int(frame_id), int(mask_id))]
            hit = good & (lab == int(mask_id))
            if np.any(hit):
                support[hit, obj_idx] += w[hit]
                hit_any[hit, obj_idx] = True
    support = support / np.maximum(denom[:, None], 1e-6)
    owner_hit_count = np.sum(hit_any, axis=1).astype(np.int16)
    hard_conflict = np.zeros((n, m), dtype=bool)
    if m:
        any_owner = owner_hit_count > 0
        hard_conflict[any_owner] = ~hit_any[any_owner]
    return support, hard_conflict, {
        "overlap_frame_ids": overlap_frames,
        "overlap_history_mask_count": int(len(owner_by_frame_label)),
        "carrier_with_any_overlap_owner_hit_rate": float(np.mean(owner_hit_count > 0)) if n else 0.0,
    }


def _da3_overlap_support(
    *,
    scene: str,
    da3_supplement_root: Path | None,
    frame_ids: list[int],
    labels: np.ndarray,
    ok: np.ndarray,
    weights: np.ndarray,
    history_ids: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    n = labels.shape[1]
    m = len(history_ids)
    support = np.zeros((n, m), dtype=np.float32)
    if da3_supplement_root is None:
        return support, {
            "da3_overlap_supplement_available": False,
            "da3_overlap_supplement_root": "",
            "da3_overlap_assignment_row_count": 0,
            "da3_overlap_assigned_mask_count": 0,
            "carrier_with_any_da3_overlap_hit_rate": 0.0,
        }
    assignment_path = da3_supplement_root / "da3_overlap_assignment_rows.csv"
    if not assignment_path.exists():
        return support, {
            "da3_overlap_supplement_available": False,
            "da3_overlap_supplement_root": _rel(da3_supplement_root),
            "da3_overlap_assignment_row_count": 0,
            "da3_overlap_assigned_mask_count": 0,
            "carrier_with_any_da3_overlap_hit_rate": 0.0,
            "da3_overlap_missing_assignment_path": _rel(assignment_path),
        }
    hist_to_idx = {str(hist_id): int(i) for i, hist_id in enumerate(history_ids)}
    rows = pd.read_csv(assignment_path)
    if "scene_id" in rows.columns:
        rows = rows[rows["scene_id"].astype(str) == str(scene)]
    if "assigned" in rows.columns:
        rows = rows[rows["assigned"].astype(bool)]
    assigned_by_mask: dict[tuple[int, int], tuple[int, float]] = {}
    for row in rows.to_dict("records"):
        hist_id = str(row.get("top1_history_id", ""))
        if hist_id not in hist_to_idx:
            continue
        try:
            obs_scene, frame_id, mask_id = _parse_obs(str(row["current_overlap_mask_observation_id"]))
        except Exception:
            continue
        if str(obs_scene) != str(scene):
            continue
        assigned_by_mask[(int(frame_id), int(mask_id))] = (hist_to_idx[hist_id], float(row.get("top1_score", 0.0) or 0.0))
    denom = np.zeros((n,), dtype=np.float32)
    hit_any = np.zeros((n,), dtype=bool)
    frame_to_local = {int(frame_id): int(i) for i, frame_id in enumerate(frame_ids)}
    matched_frame_ids: list[int] = []
    for frame_id in sorted({int(frame) for frame, _mask in assigned_by_mask if int(frame) in frame_to_local}):
        fi = frame_to_local[int(frame_id)]
        matched_frame_ids.append(int(frame_id))
        lab = labels[fi]
        good = ok[fi] & (lab > 0)
        if not np.any(good):
            continue
        for (assigned_frame, mask_id), (obj_idx, da3_score) in assigned_by_mask.items():
            if int(assigned_frame) != int(frame_id):
                continue
            hit = good & (lab == int(mask_id))
            if not np.any(hit):
                continue
            w = weights[fi, hit].astype(np.float32, copy=False)
            support[hit, int(obj_idx)] += w * float(da3_score)
            denom[hit] += w
            hit_any[hit] = True
    support = support / np.maximum(denom[:, None], 1e-6)
    return support, {
        "da3_overlap_supplement_available": True,
        "da3_overlap_supplement_root": _rel(da3_supplement_root),
        "da3_overlap_assignment_row_count": int(len(rows)),
        "da3_overlap_assigned_mask_count": int(len(assigned_by_mask)),
        "da3_overlap_matched_frame_ids": matched_frame_ids,
        "carrier_with_any_da3_overlap_hit_rate": float(np.mean(hit_any)) if n else 0.0,
    }


def _semantic_scores_by_current_mask(
    *,
    scene: str,
    current_frame_ids: list[int],
    semantic_lookup: dict[tuple[int, int], int],
    features: np.ndarray,
    history_ids: list[str],
    feature_idx_by_obj: dict[str, np.ndarray],
    mu_sem: float,
    semantic_frame_max_gap: int,
    history_max_frame: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    current_feature_indices = []
    current_keys = []
    max_allowed_frame = int(history_max_frame) + int(semantic_frame_max_gap)
    for frame_id in current_frame_ids:
        if int(frame_id) > max_allowed_frame:
            continue
        for (f, mask_id), idx in semantic_lookup.items():
            if int(f) == int(frame_id):
                current_keys.append((int(f), int(mask_id)))
                current_feature_indices.append(int(idx))
    if not current_feature_indices:
        return {}, {
            "semantic_current_mask_count": 0,
            "semantic_history_object_count_with_features": 0,
            "semantic_frame_max_gap": int(semantic_frame_max_gap),
            "semantic_max_allowed_frame": int(max_allowed_frame),
        }
    cur_feat = features[np.asarray(current_feature_indices, dtype=np.int64)]
    score_rows = np.zeros((len(current_feature_indices), len(history_ids)), dtype=np.float32)
    with_feat = 0
    for obj_idx, hist_id in enumerate(history_ids):
        hist_idx = feature_idx_by_obj.get(hist_id, np.zeros((0,), dtype=np.int64))
        if hist_idx.size == 0:
            continue
        with_feat += 1
        sim = cur_feat @ features[hist_idx].T
        topk = min(3, sim.shape[1])
        if topk <= 0:
            continue
        vals = np.partition(sim, sim.shape[1] - topk, axis=1)[:, -topk:]
        cal = np.clip((vals - float(mu_sem)) / max(1.0 - float(mu_sem), 1e-6), 0.0, 1.0)
        score_rows[:, obj_idx] = np.mean(cal, axis=1).astype(np.float32, copy=False)
    out = {int(idx): score_rows[i] for i, idx in enumerate(current_feature_indices)}
    return out, {
        "semantic_current_mask_count": int(len(current_feature_indices)),
        "semantic_history_object_count_with_features": int(with_feat),
        "semantic_frame_max_gap": int(semantic_frame_max_gap),
        "semantic_max_allowed_frame": int(max_allowed_frame),
        "semantic_topk_per_history_object": 3,
        "semantic_scene_id": scene,
    }


def _semantic_support(
    *,
    frame_ids: list[int],
    labels: np.ndarray,
    ok: np.ndarray,
    semantic_lookup: dict[tuple[int, int], int],
    mask_feature_score: dict[int, np.ndarray],
    history_count: int,
    topk_observations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = labels.shape[1]
    k = max(1, int(topk_observations))
    top_vals = np.full((n, history_count, k), -np.inf, dtype=np.float32)
    obs_count = np.zeros((n,), dtype=np.int16)
    for fi, frame_id in enumerate(frame_ids):
        lab = labels[fi]
        good = ok[fi] & (lab > 0)
        if not np.any(good):
            continue
        unique_labels = sorted({int(v) for v in np.unique(lab[good]).tolist() if int(v) > 0})
        for mask_id in unique_labels:
            feat_idx = semantic_lookup.get((int(frame_id), int(mask_id)), -1)
            if int(feat_idx) < 0 or int(feat_idx) not in mask_feature_score:
                continue
            carriers = np.flatnonzero(good & (lab == int(mask_id)))
            if carriers.size == 0:
                continue
            scores = np.asarray(mask_feature_score[int(feat_idx)], dtype=np.float32)
            merged = np.concatenate(
                [
                    top_vals[carriers],
                    np.broadcast_to(scores[None, :, None], (carriers.size, history_count, 1)),
                ],
                axis=2,
            )
            top_vals[carriers] = np.partition(merged, merged.shape[2] - k, axis=2)[:, :, -k:]
            obs_count[carriers] += 1
    valid = np.isfinite(top_vals)
    sum_vals = np.where(valid, top_vals, 0.0).sum(axis=2)
    cnt_vals = np.maximum(valid.sum(axis=2), 1)
    support = (sum_vals / cnt_vals).astype(np.float32, copy=False)
    return support, {
        "semantic_carrier_observation_rate": float(np.mean(obs_count > 0)) if n else 0.0,
        "semantic_topk_observations": int(k),
    }


def _assignment_metrics(
    *,
    scene: str,
    variant_id: str,
    support: np.ndarray,
    hard_conflict: np.ndarray | None,
    history_ids: list[str],
    tau_hist: float,
    tau_margin: float,
    tau_entropy: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n, m = support.shape
    if hard_conflict is not None:
        u = np.where(hard_conflict, 0.0, support).astype(np.float32, copy=False)
    else:
        u = support.astype(np.float32, copy=True)
    if m == 0 or n == 0:
        top1_idx = np.zeros((n,), dtype=np.int64)
        top1 = np.zeros((n,), dtype=np.float32)
        top2 = np.zeros((n,), dtype=np.float32)
    else:
        top1_idx = np.argmax(u, axis=1).astype(np.int64)
        top1 = u[np.arange(n), top1_idx].astype(np.float32)
        if m >= 2:
            part = np.partition(u, m - 2, axis=1)[:, -2:]
            top2 = np.min(part, axis=1).astype(np.float32)
        else:
            top2 = np.zeros((n,), dtype=np.float32)
    margin = (top1 - top2).astype(np.float32)
    total = np.sum(u, axis=1, keepdims=True)
    p = np.divide(u, np.maximum(total, 1e-8), out=np.zeros_like(u), where=total > 0)
    entropy_raw = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0), axis=1)
    entropy = (entropy_raw / max(math.log(max(m, 2)), 1e-6)).astype(np.float32)
    entropy[total[:, 0] <= 0] = 1.0
    if hard_conflict is not None and m > 0:
        hard_top1 = np.asarray(hard_conflict[np.arange(n), top1_idx], dtype=bool) & (top1 > 0)
    else:
        hard_top1 = np.zeros((n,), dtype=bool)
    assigned = (top1 >= float(tau_hist)) & (margin >= float(tau_margin)) & (entropy <= float(tau_entropy))
    metric = {
        "schema_version": "stream4d_v103_phase7_history_assignment_metric_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "variant_id": variant_id,
        "carrier_count": int(n),
        "history_object_count": int(m),
        "history_token_assignment_rate": float(np.mean(assigned)) if n else 0.0,
        "carrier_history_top1_mean": float(np.mean(top1)) if n else 0.0,
        "carrier_history_margin_mean": float(np.mean(margin)) if n else 0.0,
        "carrier_history_entropy_mean": float(np.mean(entropy)) if n else 1.0,
        "carrier_history_hard_conflict_rate": float(np.mean(hard_top1)) if n else 0.0,
        "tau_hist": float(tau_hist),
        "tau_margin": float(tau_margin),
        "tau_entropy": float(tau_entropy),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    return metric, {
        "U": u,
        "top1_idx": top1_idx,
        "top1_score": top1,
        "top2_score": top2,
        "margin": margin,
        "entropy": entropy,
        "assigned": assigned,
        "top1_history_id": np.asarray([history_ids[int(i)] if history_ids else "" for i in top1_idx], dtype=object),
    }


def _casebook_rows(
    *,
    scene: str,
    carrier_ids: np.ndarray,
    assignment: dict[str, np.ndarray],
    variant_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if carrier_ids.size == 0:
        return rows
    order = np.argsort(-assignment["margin"])[: max(0, int(limit))]
    for rank, idx in enumerate(order.tolist()):
        rows.append(
            {
                "schema_version": "stream4d_v103_phase7_history_token_casebook_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "variant_id": variant_id,
                "rank": int(rank),
                "carrier_id": int(carrier_ids[int(idx)]),
                "top1_history_id": str(assignment["top1_history_id"][int(idx)]),
                "top1_score": float(assignment["top1_score"][int(idx)]),
                "top2_score": float(assignment["top2_score"][int(idx)]),
                "margin": float(assignment["margin"][int(idx)]),
                "entropy": float(assignment["entropy"][int(idx)]),
                "assigned": bool(assignment["assigned"][int(idx)]),
            }
        )
    return rows


def _evaluate_scene(
    *,
    scene: str,
    phase2_root: Path,
    phase3_root: Path,
    history_rows: list[dict[str, Any]],
    history_variant_id: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary, masks, batch = _load_phase2_scene(phase2_root)
    frame_ids = [int(v) for v in summary["frame_ids"]]
    semantic_lookup, features, semantic_constants = _load_semantic(scene)
    history_ids, masks_by_obj, feature_idx_by_obj = _group_history(history_rows, semantic_lookup)
    retained_idx, retain_meta = _reconstruct_retained(
        phase3_root=phase3_root,
        scene=scene,
        max_retained_carriers=int(args.max_retained_carriers),
        variant_override=str(args.scene0011_selected_variant_id if scene == "scene0011_00" else args.scene0050_selected_variant_id),
    )
    labels, ok, weights, _xs, projection_backend, projection_runtime = _project_labels_for_indices(
        batch=batch,
        masks=masks,
        carrier_indices=retained_idx,
        cupy_device_id=int(args.cupy_device_id),
    )
    e_overlap, hard_conflict, overlap_meta = _overlap_support(
        frame_ids=frame_ids,
        labels=labels,
        ok=ok,
        weights=weights,
        history_ids=history_ids,
        masks_by_obj=masks_by_obj,
    )
    da3_roots = _scene_da3_supplement_roots(args)
    e_da3_overlap, da3_overlap_meta = _da3_overlap_support(
        scene=scene,
        da3_supplement_root=da3_roots.get(scene),
        frame_ids=frame_ids,
        labels=labels,
        ok=ok,
        weights=weights,
        history_ids=history_ids,
    )
    history_max_frame = max([int(row["frame_id"]) for row in history_rows], default=max(frame_ids))
    mask_sem_scores, semantic_meta_a = _semantic_scores_by_current_mask(
        scene=scene,
        current_frame_ids=frame_ids,
        semantic_lookup=semantic_lookup,
        features=features,
        history_ids=history_ids,
        feature_idx_by_obj=feature_idx_by_obj,
        mu_sem=float(semantic_constants["mu_sem_used"]),
        semantic_frame_max_gap=int(args.semantic_frame_max_gap),
        history_max_frame=int(history_max_frame),
    )
    e_sem, semantic_meta_b = _semantic_support(
        frame_ids=frame_ids,
        labels=labels,
        ok=ok,
        semantic_lookup=semantic_lookup,
        mask_feature_score=mask_sem_scores,
        history_count=len(history_ids),
        topk_observations=int(args.semantic_topk_observations),
    )
    rng = np.random.default_rng(10317)
    if len(history_ids) > 1:
        perm = rng.permutation(len(history_ids))
    else:
        perm = np.arange(len(history_ids))
    random_u = rng.random(e_overlap.shape, dtype=np.float32) if e_overlap.size else e_overlap.copy()
    da3_weighted = float(args.da3_overlap_weight) * e_da3_overlap
    e_overlap_for_h2 = np.maximum(e_overlap, da3_weighted).astype(np.float32, copy=False)
    overlap_present = np.max(e_overlap_for_h2, axis=1) > 0 if e_overlap_for_h2.size else np.zeros((e_overlap_for_h2.shape[0],), dtype=bool)
    h2_support = float(args.overlap_weight) * e_overlap_for_h2 + float(args.semantic_weight) * e_sem
    h2_support = np.where(overlap_present[:, None], h2_support, e_sem).astype(np.float32, copy=False)
    shuffled_support = np.array(e_sem[:, perm], dtype=np.float32, copy=True)
    variants = {
        "H1_overlap_only_history_token": (e_overlap, hard_conflict),
        "H2_overlap_semantic_viewset_history_token": (h2_support, hard_conflict),
        "H2a_overlap_semantic_without_da3_supplement_control": (
            np.where(
                (np.max(e_overlap, axis=1) > 0 if e_overlap.size else np.zeros((e_overlap.shape[0],), dtype=bool))[:, None],
                float(args.overlap_weight) * e_overlap + float(args.semantic_weight) * e_sem,
                e_sem,
            ).astype(np.float32, copy=False),
            hard_conflict,
        ),
        "H5_semantic_only_history_token_control": (e_sem, None),
        "H5_shuffled_history_token_control": (shuffled_support, None),
        "H5_random_history_token_control": (random_u, None),
    }
    metric_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    best_assignment: dict[str, np.ndarray] | None = None
    carrier_ids = np.asarray(batch["carrier_id"], dtype=np.int64)[retained_idx]
    token_payload: dict[str, Any] = {
        "scene_id": scene,
        "history_variant_id": history_variant_id,
        "history_ids": history_ids,
        "carrier_indices": torch.as_tensor(retained_idx.astype(np.int64), dtype=torch.int64),
        "carrier_ids": torch.as_tensor(carrier_ids.astype(np.int64), dtype=torch.int64),
        "e_overlap": torch.as_tensor(e_overlap.astype(np.float32), dtype=torch.float32),
        "e_da3_overlap": torch.as_tensor(e_da3_overlap.astype(np.float32), dtype=torch.float32),
        "e_overlap_for_h2": torch.as_tensor(e_overlap_for_h2.astype(np.float32), dtype=torch.float32),
        "e_sem": torch.as_tensor(e_sem.astype(np.float32), dtype=torch.float32),
        "hard_conflict": torch.as_tensor(hard_conflict.astype(np.bool_), dtype=torch.bool),
    }
    assignments_by_variant: dict[str, dict[str, np.ndarray]] = {}
    for variant_id, (support, conflict) in variants.items():
        metric, assignment = _assignment_metrics(
            scene=scene,
            variant_id=variant_id,
            support=np.asarray(support, dtype=np.float32),
            hard_conflict=conflict,
            history_ids=history_ids,
            tau_hist=float(args.tau_hist),
            tau_margin=float(args.tau_margin),
            tau_entropy=float(args.tau_entropy),
        )
        metric_rows.append(metric)
        assignments_by_variant[variant_id] = assignment
        if "control" in variant_id:
            control_rows.append(
                {
                    "schema_version": "stream4d_v103_phase7_history_control_row_v1",
                    "phase_id": PHASE_ID,
                    **metric,
                    "control_id": variant_id,
                }
            )
        if variant_id == "H2_overlap_semantic_viewset_history_token":
            best_assignment = assignment
            casebook_rows.extend(
                _casebook_rows(
                    scene=scene,
                    carrier_ids=carrier_ids,
                    assignment=assignment,
                    variant_id=variant_id,
                    limit=int(args.casebook_limit),
                )
            )
    if best_assignment is not None:
        token_payload["h2_top1_history_index"] = torch.as_tensor(best_assignment["top1_idx"].astype(np.int64), dtype=torch.int64)
        token_payload["h2_top1_score"] = torch.as_tensor(best_assignment["top1_score"].astype(np.float32), dtype=torch.float32)
        token_payload["h2_margin"] = torch.as_tensor(best_assignment["margin"].astype(np.float32), dtype=torch.float32)
        token_payload["h2_entropy"] = torch.as_tensor(best_assignment["entropy"].astype(np.float32), dtype=torch.float32)
        token_payload["h2_assigned"] = torch.as_tensor(best_assignment["assigned"].astype(np.bool_), dtype=torch.bool)
    h2 = assignments_by_variant["H2_overlap_semantic_viewset_history_token"]
    h1 = assignments_by_variant["H1_overlap_only_history_token"]
    sem_only = assignments_by_variant["H5_semantic_only_history_token_control"]
    carrier_support_df = pd.DataFrame(
        {
            "schema_version": "stream4d_v103_phase7_carrier_history_support_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "carrier_index": retained_idx.astype(np.int64),
            "carrier_id": carrier_ids.astype(np.int64),
            "history_variant_id": history_variant_id,
            "selected_phase3_variant_id": str(retain_meta["selected_phase3_variant_id"]),
            "h2_top1_history_id": h2["top1_history_id"].astype(str),
            "h2_top1_score": h2["top1_score"].astype(np.float32),
            "h2_top2_score": h2["top2_score"].astype(np.float32),
            "h2_margin": h2["margin"].astype(np.float32),
            "h2_entropy": h2["entropy"].astype(np.float32),
            "h2_assigned": h2["assigned"].astype(bool),
            "h1_overlap_top1_score": h1["top1_score"].astype(np.float32),
            "da3_overlap_top1_score": np.max(e_da3_overlap, axis=1).astype(np.float32) if e_da3_overlap.size else np.zeros_like(h1["top1_score"], dtype=np.float32),
            "semantic_only_top1_score": sem_only["top1_score"].astype(np.float32),
            "hard_conflict_any": np.any(hard_conflict, axis=1).astype(bool),
        }
    )
    scene_meta = {
        "scene_id": scene,
        "phase2_root": _rel(phase2_root),
        "phase3_root": _rel(phase3_root),
        "history_variant_id": history_variant_id,
        "frame_ids": frame_ids,
        "history_object_count": int(len(history_ids)),
        "history_mask_observation_count": int(len(history_rows)),
        "retained": retain_meta,
        "projection_backend": projection_backend,
        "projection_runtime_sec": projection_runtime,
        **overlap_meta,
        **da3_overlap_meta,
        **semantic_constants,
        **semantic_meta_a,
        **semantic_meta_b,
        "history_d4rt_anchor_provider_available": False,
        "history_spatial_provider_available": False,
        "h3_h4_note": "H3/H4 full D4RT/spatial anchor terms are not claimed in this diagnostic; H2 tests overlap+semantic token readiness.",
        "h2_overlap_absent_semantic_backoff": True,
        "h2_da3_overlap_supplement_policy": "max(original_overlap, da3_overlap_weight * assigned_da3_overlap_support)",
        "da3_overlap_weight": float(args.da3_overlap_weight),
        "shuffled_control_policy": "semantic view-set columns shuffled; overlap owner masks intentionally disabled to avoid column-permutation invariant margin/entropy.",
    }
    return metric_rows, control_rows, casebook_rows, {"meta": scene_meta, "token_payload": token_payload, "carrier_support_df": carrier_support_df}


def _build_gates(metric_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    scenes = sorted({str(row["scene_id"]) for row in metric_rows})
    for scene in scenes:
        by_variant = {str(row["variant_id"]): row for row in metric_rows if str(row["scene_id"]) == scene}
        real = by_variant.get("H2_overlap_semantic_viewset_history_token")
        sem = by_variant.get("H5_semantic_only_history_token_control")
        shuf = by_variant.get("H5_shuffled_history_token_control")
        rand = by_variant.get("H5_random_history_token_control")
        if not real or not sem:
            continue
        gate_specs = [
            (
                "history_token_assignment_rate_between_0p10_0p70",
                0.10 <= float(real["history_token_assignment_rate"]) <= 0.70,
                real["history_token_assignment_rate"],
                "0.10..0.70",
            ),
            (
                "entropy_mean_le_semantic_only_minus_0p05",
                float(real["carrier_history_entropy_mean"]) <= float(sem["carrier_history_entropy_mean"]) - 0.05,
                real["carrier_history_entropy_mean"],
                f"<= {float(sem['carrier_history_entropy_mean']) - 0.05:.6f}",
            ),
            (
                "margin_mean_ge_semantic_only_plus_0p03",
                float(real["carrier_history_margin_mean"]) >= float(sem["carrier_history_margin_mean"]) + 0.03,
                real["carrier_history_margin_mean"],
                f">= {float(sem['carrier_history_margin_mean']) + 0.03:.6f}",
            ),
            (
                "hard_conflict_rate_le_0p20",
                float(real["carrier_history_hard_conflict_rate"]) <= 0.20,
                real["carrier_history_hard_conflict_rate"],
                "<= 0.20",
            ),
        ]
        if shuf:
            gate_specs.append(
                (
                    "real_entropy_and_margin_beat_shuffled",
                    float(real["carrier_history_entropy_mean"]) < float(shuf["carrier_history_entropy_mean"])
                    and float(real["carrier_history_margin_mean"]) > float(shuf["carrier_history_margin_mean"]),
                    f"real_entropy={real['carrier_history_entropy_mean']} real_margin={real['carrier_history_margin_mean']}",
                    f"beat shuffled entropy={shuf['carrier_history_entropy_mean']} margin={shuf['carrier_history_margin_mean']}",
                )
            )
        if rand:
            gate_specs.append(
                (
                    "real_entropy_and_margin_beat_random",
                    float(real["carrier_history_entropy_mean"]) < float(rand["carrier_history_entropy_mean"])
                    and float(real["carrier_history_margin_mean"]) > float(rand["carrier_history_margin_mean"]),
                    f"real_entropy={real['carrier_history_entropy_mean']} real_margin={real['carrier_history_margin_mean']}",
                    f"beat random entropy={rand['carrier_history_entropy_mean']} margin={rand['carrier_history_margin_mean']}",
                )
            )
        for name, ok, observed, required in gate_specs:
            row = {
                "schema_version": "stream4d_v103_phase7_gate_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "variant_id": "H2_overlap_semantic_viewset_history_token",
                "gate_name": name,
                "pass": bool(ok),
                "observed": observed,
                "required": required,
            }
            gate_rows.append(row)
            if not ok:
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v103_phase7_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "scene_id": scene,
                        "failure_id": name,
                        "severity": "blocking",
                        "evidence": f"observed={observed} required={required}",
                        "repair_direction": (
                            "Diagnose whether reliable D4RT carrier overlap is too sparse, semantic view-set is non-discriminative, "
                            "or D4RT-only coverage is insufficient; DA3 may only be introduced as a same-formula primitive provider."
                        ),
                    }
                )
    return gate_rows, failure_rows, len(failure_rows) == 0 and bool(gate_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v103 Phase7 causal carrier-to-history token readiness diagnostic.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--history-root", default=str(DEFAULT_HISTORY_ROOT))
    parser.add_argument("--history-variant-id", default="")
    parser.add_argument("--phase3-root", default=str(DEFAULT_PHASE3_ROOT))
    parser.add_argument("--scene0011-phase3-root", default="")
    parser.add_argument("--scene0050-phase3-root", default="")
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    parser.add_argument("--scene0011-selected-variant-id", default="")
    parser.add_argument("--scene0050-selected-variant-id", default="")
    parser.add_argument("--scene0011-da3-overlap-supplement-root", default="")
    parser.add_argument("--scene0050-da3-overlap-supplement-root", default="")
    parser.add_argument("--scene", choices=["all", "scene0011_00", "scene0050_00"], default="all")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--max-retained-carriers", type=int, default=0)
    parser.add_argument("--semantic-frame-max-gap", type=int, default=60)
    parser.add_argument("--semantic-topk-observations", type=int, default=3)
    parser.add_argument("--overlap-weight", type=float, default=0.70)
    parser.add_argument("--da3-overlap-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=0.30)
    parser.add_argument("--tau-hist", type=float, default=0.55)
    parser.add_argument("--tau-margin", type=float, default=0.10)
    parser.add_argument("--tau-entropy", type=float, default=0.75)
    parser.add_argument("--casebook-limit", type=int, default=64)
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    history_root = _project(args.history_root)
    phase3_root = _project(args.phase3_root)
    history_variant_id = _history_variant_id(history_root, str(args.history_variant_id))
    history_by_scene, history_object_rows = _load_history_objects(history_root, history_variant_id)
    phase2_roots = _scene_phase2_roots(args)
    phase3_roots = _scene_phase3_roots(args)
    da3_supplement_roots = _scene_da3_supplement_roots(args)
    scene_ids = list(phase2_roots) if args.scene == "all" else [str(args.scene)]
    metric_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    scene_meta: list[dict[str, Any]] = []
    token_payloads: dict[str, Any] = {}
    carrier_support_frames: list[pd.DataFrame] = []
    for scene in scene_ids:
        rows = history_by_scene.get(scene, [])
        if not rows:
            raise RuntimeError(f"no history rows for {scene} in {history_root} variant {history_variant_id}")
        metrics, controls, cases, extra = _evaluate_scene(
            scene=scene,
            phase2_root=phase2_roots[scene],
            phase3_root=phase3_roots[scene],
            history_rows=rows,
            history_variant_id=history_variant_id,
            args=args,
        )
        metric_rows.extend(metrics)
        control_rows.extend(controls)
        casebook_rows.extend(cases)
        scene_meta.append(extra["meta"])
        token_payloads[scene] = extra["token_payload"]
        carrier_support_frames.append(extra["carrier_support_df"])
    gate_rows, failure_rows, gate_pass = _build_gates(metric_rows)
    support_df = pd.concat(carrier_support_frames, ignore_index=True) if carrier_support_frames else pd.DataFrame()
    support_path = out / "carrier_history_support_rows.parquet"
    support_df.to_parquet(support_path, index=False)
    torch.save(
        {
            "schema_version": "stream4d_v103_phase7_history_token_feature_rows_v1",
            "phase_id": PHASE_ID,
            "history_root": _rel(history_root),
            "history_variant_id": history_variant_id,
            "phase3_root": _rel(phase3_root),
            "payload_by_scene": token_payloads,
            "truthfulness_note": "This is a token-readiness diagnostic payload; it is not a Phase8 history-aware AP materialization.",
        },
        out / "history_token_feature_rows.pt",
    )
    _write_csv(out / "history_object_rows.csv", history_object_rows)
    _write_csv(out / "history_assignment_metric_rows.csv", metric_rows)
    _write_csv(out / "history_control_rows.csv", control_rows)
    _write_csv(out / "history_token_casebook_rows.csv", casebook_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    summary = {
        "schema_version": "stream4d_v103_phase7_causal_history_token_readiness_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_ENTER_PHASE8_HISTORY_AWARE_CLUSTERING" if gate_pass else "NO_GO_PHASE7_HISTORY_TOKEN_READINESS",
        "phase7_pass": bool(gate_pass),
        "failure_count": len(failure_rows),
        "history_root": _rel(history_root),
        "history_variant_id": history_variant_id,
        "phase3_root": _rel(phase3_root),
        "phase3_root_by_scene": {scene: _rel(root) for scene, root in sorted(phase3_roots.items())},
        "da3_overlap_supplement_root_by_scene": {
            scene: "" if root is None else _rel(root) for scene, root in sorted(da3_supplement_roots.items())
        },
        "plan_doc": _rel(PLAN_DOC),
        "scene_meta": scene_meta,
        "truthfulness_note": (
            "Phase7 uses pre-update c0000 history rows and current c0001 carriers only. "
            "No GT is used for prediction or threshold choice. H3/H4 D4RT/spatial anchor terms are not claimed here."
        ),
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "carrier_history_support_rows": _rel(support_path),
            "history_token_feature_rows": _rel(out / "history_token_feature_rows.pt"),
            "history_object_rows": _rel(out / "history_object_rows.csv"),
            "history_assignment_metric_rows": _rel(out / "history_assignment_metric_rows.csv"),
            "history_control_rows": _rel(out / "history_control_rows.csv"),
            "history_token_casebook_rows": _rel(out / "history_token_casebook_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
