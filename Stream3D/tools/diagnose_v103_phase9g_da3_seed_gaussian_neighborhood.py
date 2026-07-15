#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v103_phase3_fast_carrier_reliability_filter import _compute_scene_arrays, _project  # noqa: E402
from build_v103_phase9b_da3_provider_readiness import (  # noqa: E402
    SCENES as PHASE9B_SCENES,
    _artifact_paths,
    _frame_manifest,
    _load_xyz,
    _mask_meta,
    _project_masks,
)


PHASE_ID = "v103_phase9g_da3_seed_gaussian_neighborhood"
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
PLAN_DOC = REPO_ROOT / "docs/stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"
DEFAULT_PHASE9E_ROOT = AUDIT_ROOT / "v103_phase9e_d4rt_anchor_da3_induced_carriers_r5_target_object"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase9g_da3_seed_gaussian_neighborhood_r1"
DEFAULT_SCENE0011_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_SCENE0050_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_first32"

SCENE_SPECS = {
    "scene0011_00": {
        "phase2_root": DEFAULT_SCENE0011_PHASE2,
        "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_features.npz",
        "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
    },
    "scene0050_00": {
        "phase2_root": DEFAULT_SCENE0050_PHASE2,
        "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_features.npz",
        "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
    },
}

VARIANTS = [
    {
        "variant_id": "g1_seed_hit2_ratio005",
        "seed_hit_min": 2,
        "seed_ratio_min": 0.05,
        "veto_ratio_max": 1.01,
        "knn_k": 0,
        "max_seed_query": 0,
    },
    {
        "variant_id": "g2_seed_hit3_ratio010",
        "seed_hit_min": 3,
        "seed_ratio_min": 0.10,
        "veto_ratio_max": 1.01,
        "knn_k": 0,
        "max_seed_query": 0,
    },
    {
        "variant_id": "g3_seed_hit2_ratio005_knn8_top20000",
        "seed_hit_min": 2,
        "seed_ratio_min": 0.05,
        "veto_ratio_max": 1.01,
        "knn_k": 8,
        "max_seed_query": 20000,
    },
    {
        "variant_id": "g4_seed_hit3_ratio010_knn16_top20000",
        "seed_hit_min": 3,
        "seed_ratio_min": 0.10,
        "veto_ratio_max": 1.01,
        "knn_k": 16,
        "max_seed_query": 20000,
    },
    {
        "variant_id": "g5_seed_hit2_ratio005_veto050",
        "seed_hit_min": 2,
        "seed_ratio_min": 0.05,
        "veto_ratio_max": 0.50,
        "knn_k": 0,
        "max_seed_query": 0,
    },
    {
        "variant_id": "g6_seed_hit2_ratio005_veto050_knn8_top10000",
        "seed_hit_min": 2,
        "seed_ratio_min": 0.05,
        "veto_ratio_max": 0.50,
        "knn_k": 8,
        "max_seed_query": 10000,
    },
    {
        "variant_id": "g7_objseed_hit1_ratio003",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 1.01,
        "knn_k": 0,
        "max_seed_query": 0,
    },
    {
        "variant_id": "g8_objseed_hit2_ratio005",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 2,
        "seed_ratio_min": 0.05,
        "veto_ratio_max": 1.01,
        "knn_k": 0,
        "max_seed_query": 0,
    },
    {
        "variant_id": "g9_objseed_hit1_ratio003_knn16_top20000",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 1.01,
        "knn_k": 16,
        "max_seed_query": 20000,
    },
    {
        "variant_id": "g10_objseed_hit2_ratio005_knn16_top20000",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 2,
        "seed_ratio_min": 0.05,
        "veto_ratio_max": 1.01,
        "knn_k": 16,
        "max_seed_query": 20000,
    },
    {
        "variant_id": "g11_objseed_hit1_ratio003_veto050_knn8_top10000",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 0.50,
        "knn_k": 8,
        "max_seed_query": 10000,
    },
    {
        "variant_id": "g12_objseed_hit1_ratio003_obsratio010_count16",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 1.01,
        "knn_k": 0,
        "max_seed_query": 0,
        "obs_selected_ratio_min": 0.10,
        "obs_selected_count_min": 16,
    },
    {
        "variant_id": "g13_objseed_hit1_ratio003_obsratio020_count16",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 1.01,
        "knn_k": 0,
        "max_seed_query": 0,
        "obs_selected_ratio_min": 0.20,
        "obs_selected_count_min": 16,
    },
    {
        "variant_id": "g14_objseed_hit2_ratio005_obsratio010_count16",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 2,
        "seed_ratio_min": 0.05,
        "veto_ratio_max": 1.01,
        "knn_k": 0,
        "max_seed_query": 0,
        "obs_selected_ratio_min": 0.10,
        "obs_selected_count_min": 16,
    },
    {
        "variant_id": "g15_objseed_hit2_ratio005_obsratio020_count16",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 2,
        "seed_ratio_min": 0.05,
        "veto_ratio_max": 1.01,
        "knn_k": 0,
        "max_seed_query": 0,
        "obs_selected_ratio_min": 0.20,
        "obs_selected_count_min": 16,
    },
    {
        "variant_id": "g16_objseed_hit1_ratio003_veto050_obsratio010_count8",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 0.50,
        "knn_k": 0,
        "max_seed_query": 0,
        "obs_selected_ratio_min": 0.10,
        "obs_selected_count_min": 8,
    },
    {
        "variant_id": "g17_objseed_hit1_ratio003_broadratio050_obsratio010_count8",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 1.01,
        "broad_ratio_max": 0.50,
        "knn_k": 0,
        "max_seed_query": 0,
        "obs_selected_ratio_min": 0.10,
        "obs_selected_count_min": 8,
    },
    {
        "variant_id": "g18_objseed_hit1_ratio003_broadratio035_obsratio010_count8",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 1,
        "seed_ratio_min": 0.03,
        "veto_ratio_max": 1.01,
        "broad_ratio_max": 0.35,
        "knn_k": 0,
        "max_seed_query": 0,
        "obs_selected_ratio_min": 0.10,
        "obs_selected_count_min": 8,
    },
    {
        "variant_id": "g19_objseed_hit2_ratio005_broadratio050_obsratio010_count8",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 2,
        "seed_ratio_min": 0.05,
        "veto_ratio_max": 1.01,
        "broad_ratio_max": 0.50,
        "knn_k": 0,
        "max_seed_query": 0,
        "obs_selected_ratio_min": 0.10,
        "obs_selected_count_min": 8,
    },
    {
        "variant_id": "g20_objseed_hit2_ratio005_broadratio035_obsratio010_count8",
        "anchor_source": "object_like_non_broad_positive",
        "seed_hit_min": 2,
        "seed_ratio_min": 0.05,
        "veto_ratio_max": 1.01,
        "broad_ratio_max": 0.35,
        "knn_k": 0,
        "max_seed_query": 0,
        "obs_selected_ratio_min": 0.10,
        "obs_selected_count_min": 8,
    },
]


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


def _project_phase(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _load_phase9e_supports(phase9e_root: Path, scene_id: str) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    scene_dir = phase9e_root / scene_id
    positive_path = scene_dir / "d4rt_positive_anchor_support_rows.csv"
    veto_path = scene_dir / "d4rt_veto_support_rows.csv"
    if not positive_path.exists() or not veto_path.exists():
        raise FileNotFoundError(f"missing Phase9e support rows under {scene_dir}")

    def load(path: Path) -> set[tuple[int, int]]:
        df = pd.read_csv(path)
        return set((int(row.frame_id), int(row.mask_id)) for row in df.itertuples(index=False))

    return load(positive_path), load(veto_path)


def _obs_meta_from_phase3(scene_id: str, device_id: int, out: Path) -> dict[str, dict[str, Any]]:
    spec = dict(SCENE_SPECS[scene_id])
    spec["phase2_root"] = _project(spec["phase2_root"])
    diag, _unused_a, _unused_b, _arrays = _compute_scene_arrays(scene_id, spec, out / f"{scene_id}_phase3_meta", int(device_id))
    frame_ids = [int(v) for v in diag["frame_ids"]]
    object_like_by_frame = {int(k): np.asarray(v, dtype=np.int32) for k, v in dict(diag["object_like_by_frame"]).items()}
    broad_map = np.asarray(diag["broad_map"], dtype=bool)
    object_map = np.asarray(diag["object_map"], dtype=bool)
    meta: dict[str, dict[str, Any]] = {}
    for fi, frame_id in enumerate(frame_ids):
        object_like_labels = set(int(v) for v in object_like_by_frame.get(fi, np.asarray([], dtype=np.int32)).tolist())
        for label in np.unique(diag["masks"][fi]).astype(int).tolist():
            if label <= 0:
                continue
            safe = min(int(label), broad_map.shape[1] - 1)
            meta[f"{scene_id}:{frame_id}:{int(label)}"] = {
                "frame_id": int(frame_id),
                "mask_id": int(label),
                "is_object_like": bool(int(label) in object_like_labels or object_map[fi, safe]),
                "is_broad": bool(broad_map[fi, safe]),
            }
    return meta


def _load_or_project_da3(
    scene_id: str,
    out: Path,
    projection_cache_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scene_cache = out / scene_id / "da3_projection_cache"
    read_cache = (
        projection_cache_root / scene_id / "da3_projection_cache"
        if projection_cache_root is not None
        else scene_cache
    )
    if projection_cache_root is not None and not read_cache.exists():
        read_cache = scene_cache
    mask_path = read_cache / "mask_by_frame.npy"
    xyz_path = read_cache / "xyz.npy"
    frame_path = read_cache / "frame_manifest_rows.csv"
    meta_path = read_cache / "mask_meta_rows.csv"
    manifest_path = read_cache / "manifest.json"
    if mask_path.exists() and xyz_path.exists() and frame_path.exists() and meta_path.exists():
        mask_by_frame = np.load(mask_path, mmap_mode="r")
        xyz = np.load(xyz_path, mmap_mode="r")
        frame_df = pd.read_csv(frame_path)
        meta = pd.read_csv(meta_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest = dict(manifest)
        manifest["cache_mode"] = "read_existing"
        manifest["cache_root"] = _rel(read_cache)
        return np.asarray(mask_by_frame), np.asarray(xyz), frame_df, meta, manifest

    scene_cache.mkdir(parents=True, exist_ok=True)
    spec = PHASE9B_SCENES[scene_id]
    ply_path, mini_npz = _artifact_paths(spec)
    if ply_path is None or mini_npz is None:
        raise FileNotFoundError(f"missing DA3 artifacts for {scene_id}")
    xyz = _load_xyz(ply_path)
    with np.load(mini_npz) as data:
        mini = {key: np.asarray(data[key]) for key in data.files}
    camera_count = int(min(len(mini["extrinsics"]), len(mini["intrinsics"]), len(mini["depth"])))
    frame_df = _frame_manifest(spec["input_manifest"], camera_count)
    frame_ids = frame_df["frame_id"].astype(int).tolist()
    meta = _mask_meta(scene_id, frame_ids, spec)
    mask_by_frame, source_rows, reprojection_valid_any = _project_masks(
        scene_id=scene_id,
        xyz=xyz,
        mini=mini,
        frame_df=frame_df,
        spec=spec,
    )
    np.save(mask_path, mask_by_frame)
    np.save(xyz_path, xyz.astype(np.float32, copy=False))
    frame_df.to_csv(frame_path, index=False)
    meta.to_csv(meta_path, index=False)
    _write_csv(scene_cache / "mask_projection_source_rows.csv", source_rows)
    manifest = {
        "scene_id": scene_id,
        "ply_path": _rel(ply_path),
        "mini_npz": _rel(mini_npz),
        "mask_by_frame": _rel(mask_path),
        "xyz": _rel(xyz_path),
        "reprojection_valid_rate": reprojection_valid_any,
        "gaussian_count": int(len(xyz)),
        "frame_count": int(mask_by_frame.shape[0]),
    }
    _write_json(manifest_path, manifest)
    return mask_by_frame, xyz.astype(np.float32, copy=False), frame_df, meta, manifest


def _frame_label_lookup(frame_df: pd.DataFrame, masks: set[tuple[int, int]], max_label: int) -> np.ndarray:
    table = np.zeros((len(frame_df), max_label + 1), dtype=bool)
    frame_to_idx = {int(row.frame_id): int(i) for i, row in enumerate(frame_df.itertuples(index=False))}
    for frame_id, mask_id in masks:
        fi = frame_to_idx.get(int(frame_id))
        if fi is not None and 0 <= int(mask_id) <= max_label:
            table[fi, int(mask_id)] = True
    return table


def _hit_counts(mask_by_frame: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    out = np.zeros(mask_by_frame.shape[1], dtype=np.int16)
    max_label = lookup.shape[1] - 1
    for fi in range(mask_by_frame.shape[0]):
        labels = np.minimum(np.asarray(mask_by_frame[fi], dtype=np.int32), max_label)
        out += lookup[fi, labels].astype(np.int16)
    return out


def _obs_sets(
    mask_by_frame: np.ndarray,
    frame_df: pd.DataFrame,
    selected: np.ndarray,
    scene_id: str,
    obs_selected_ratio_min: float = 0.0,
    obs_selected_count_min: int = 1,
) -> set[str]:
    selected = np.asarray(selected, dtype=bool)
    obs: set[str] = set()
    if not np.any(selected):
        return obs
    for fi in range(mask_by_frame.shape[0]):
        frame_id = int(frame_df.iloc[fi]["frame_id"])
        frame_labels = np.asarray(mask_by_frame[fi], dtype=np.int32)
        selected_labels = frame_labels[selected]
        selected_labels = selected_labels[selected_labels > 0]
        if selected_labels.size == 0:
            continue
        selected_counts = np.bincount(selected_labels)
        if obs_selected_ratio_min > 0:
            total_labels = frame_labels[frame_labels > 0]
            total_counts = np.bincount(total_labels, minlength=selected_counts.shape[0])
        else:
            total_counts = np.ones_like(selected_counts)
        for label in np.flatnonzero(selected_counts).astype(int).tolist():
            count = int(selected_counts[label])
            if count < int(obs_selected_count_min):
                continue
            denom = int(total_counts[label]) if label < int(total_counts.shape[0]) else count
            ratio = float(count / max(denom, 1))
            if ratio >= float(obs_selected_ratio_min):
                obs.add(f"{scene_id}:{frame_id}:{int(label)}")
    return obs


def _summarize_obs(prefix: str, obs_set: set[str], obs_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    known = [obs for obs in obs_set if obs in obs_meta]
    object_like = [obs for obs in known if bool(obs_meta[obs]["is_object_like"])]
    broad = [obs for obs in known if bool(obs_meta[obs]["is_broad"])]
    return {
        f"{prefix}_obs_count": int(len(obs_set)),
        f"{prefix}_known_obs_count": int(len(known)),
        f"{prefix}_object_like_obs_count": int(len(object_like)),
        f"{prefix}_object_like_obs_rate": float(len(object_like) / max(len(known), 1)),
        f"{prefix}_broad_obs_count": int(len(broad)),
        f"{prefix}_broad_obs_rate": float(len(broad) / max(len(known), 1)),
    }


def _rank_seed_indices(seed_idx: np.ndarray, seed_hit: np.ndarray, seed_ratio: np.ndarray, veto_ratio: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or seed_idx.size <= limit:
        return seed_idx
    order = np.lexsort((veto_ratio[seed_idx], -seed_ratio[seed_idx], -seed_hit[seed_idx]))
    return seed_idx[order[:limit]]


def _knn_expand(xyz: np.ndarray, seed_idx: np.ndarray, k: int, limit: int, seed_hit: np.ndarray, seed_ratio: np.ndarray, veto_ratio: np.ndarray) -> tuple[np.ndarray, str, int]:
    if k <= 0 or seed_idx.size == 0:
        return seed_idx, "none", 0
    query_idx = _rank_seed_indices(seed_idx, seed_hit, seed_ratio, veto_ratio, int(limit))
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return seed_idx, "scipy_missing_no_expansion", int(query_idx.size)
    tree = cKDTree(np.asarray(xyz, dtype=np.float32))
    kk = min(int(k) + 1, int(xyz.shape[0]))
    _dist, neigh = tree.query(np.asarray(xyz[query_idx], dtype=np.float32), k=kk, workers=-1)
    expanded = np.unique(np.concatenate([seed_idx.astype(np.int64, copy=False), np.asarray(neigh).reshape(-1).astype(np.int64)]))
    return expanded, "scipy_ckdtree_knn", int(query_idx.size)


def _score_variant(
    *,
    scene_id: str,
    variant: dict[str, Any],
    mask_by_frame: np.ndarray,
    xyz: np.ndarray,
    frame_df: pd.DataFrame,
    positive_obs: set[str],
    seed_hit: np.ndarray,
    veto_hit: np.ndarray,
    broad_hit: np.ndarray,
    visible_count: np.ndarray,
    obs_meta: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    seed_ratio = np.divide(seed_hit.astype(np.float32), np.maximum(visible_count, 1).astype(np.float32), out=np.zeros_like(seed_hit, dtype=np.float32), where=visible_count > 0)
    veto_ratio = np.divide(veto_hit.astype(np.float32), np.maximum(visible_count, 1).astype(np.float32), out=np.zeros_like(veto_hit, dtype=np.float32), where=visible_count > 0)
    broad_ratio = np.divide(broad_hit.astype(np.float32), np.maximum(visible_count, 1).astype(np.float32), out=np.zeros_like(broad_hit, dtype=np.float32), where=visible_count > 0)
    seed = (
        (visible_count > 0)
        & (seed_hit >= int(variant["seed_hit_min"]))
        & (seed_ratio >= float(variant["seed_ratio_min"]))
        & (veto_ratio <= float(variant["veto_ratio_max"]))
        & (broad_ratio <= float(variant.get("broad_ratio_max", 1.01)))
    )
    seed_idx = np.flatnonzero(seed)
    selected_idx, backend, queried_seed_count = _knn_expand(
        xyz=xyz,
        seed_idx=seed_idx,
        k=int(variant.get("knn_k", 0)),
        limit=int(variant.get("max_seed_query", 0)),
        seed_hit=seed_hit,
        seed_ratio=seed_ratio,
        veto_ratio=veto_ratio,
    )
    selected = np.zeros(mask_by_frame.shape[1], dtype=bool)
    selected[selected_idx] = True
    obs_selected_ratio_min = float(variant.get("obs_selected_ratio_min", 0.0))
    obs_selected_count_min = int(variant.get("obs_selected_count_min", 1))
    selected_obs = _obs_sets(
        mask_by_frame,
        frame_df,
        selected,
        scene_id,
        obs_selected_ratio_min=obs_selected_ratio_min,
        obs_selected_count_min=obs_selected_count_min,
    )
    induced_obs = selected_obs - positive_obs
    row: dict[str, Any] = {
        "schema_version": "stream4d_v103_phase9g_gaussian_neighborhood_variant_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "variant_id": str(variant["variant_id"]),
        "anchor_source": str(variant.get("anchor_source", "all_positive")),
        "seed_hit_min": int(variant["seed_hit_min"]),
        "seed_ratio_min": float(variant["seed_ratio_min"]),
        "veto_ratio_max": float(variant["veto_ratio_max"]),
        "broad_ratio_max": float(variant.get("broad_ratio_max", 1.01)),
        "knn_k": int(variant.get("knn_k", 0)),
        "max_seed_query": int(variant.get("max_seed_query", 0)),
        "obs_selected_ratio_min": obs_selected_ratio_min,
        "obs_selected_count_min": obs_selected_count_min,
        "knn_backend": backend,
        "knn_queried_seed_count": queried_seed_count,
        "seed_gaussian_count": int(seed_idx.size),
        "selected_gaussian_count": int(selected_idx.size),
        "selected_gaussian_rate": float(selected_idx.size / max(mask_by_frame.shape[1], 1)),
        "seed_hit_mean_selected": float(np.mean(seed_hit[selected_idx])) if selected_idx.size else 0.0,
        "seed_ratio_mean_selected": float(np.mean(seed_ratio[selected_idx])) if selected_idx.size else 0.0,
        "veto_ratio_mean_selected": float(np.mean(veto_ratio[selected_idx])) if selected_idx.size else 0.0,
        "broad_ratio_mean_selected": float(np.mean(broad_ratio[selected_idx])) if selected_idx.size else 0.0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_coverage_gate": False,
    }
    row.update(_summarize_obs("selected", selected_obs, obs_meta))
    row.update(_summarize_obs("induced", induced_obs, obs_meta))
    induced_known = max(int(row["induced_known_obs_count"]), 1)
    gate = (
        int(row["induced_object_like_obs_count"]) >= 30
        and float(row["induced_broad_obs_count"]) / float(induced_known) <= 0.50
        and int(row["selected_gaussian_count"]) <= 300000
    )
    row["phase9g_object_like_extension_gate_pass"] = bool(gate)
    obs_rows = [
        {
            "schema_version": "stream4d_v103_phase9g_selected_observation_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene_id,
            "variant_id": str(variant["variant_id"]),
            "mask_observation_id": obs,
            "is_induced": bool(obs in induced_obs),
            "is_object_like": bool(obs_meta.get(obs, {}).get("is_object_like", False)),
            "is_broad": bool(obs_meta.get(obs, {}).get("is_broad", True)),
            "uses_gt_for_prediction": False,
        }
        for obs in sorted(selected_obs)
    ]
    return row, pd.DataFrame(obs_rows)


def _process_scene(
    scene_id: str,
    phase9e_root: Path,
    out: Path,
    device_id: int,
    projection_cache_root: Path | None,
) -> dict[str, Any]:
    scene_out = out / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)
    positive_masks, veto_masks = _load_phase9e_supports(phase9e_root, scene_id)
    mask_by_frame, xyz, frame_df, _meta, manifest = _load_or_project_da3(scene_id, out, projection_cache_root)
    obs_meta = _obs_meta_from_phase3(scene_id, int(device_id), out)
    positive_obs = {f"{scene_id}:{frame_id}:{mask_id}" for frame_id, mask_id in positive_masks}
    object_positive_masks = set()
    for frame_id, mask_id in positive_masks:
        obs = f"{scene_id}:{frame_id}:{mask_id}"
        bits = obs_meta.get(obs, {})
        if bool(bits.get("is_object_like", False)) and not bool(bits.get("is_broad", True)):
            object_positive_masks.add((int(frame_id), int(mask_id)))
    broad_masks = {
        (int(bits["frame_id"]), int(bits["mask_id"]))
        for bits in obs_meta.values()
        if bool(bits.get("is_broad", False))
    }
    max_label = int(np.max(mask_by_frame)) if mask_by_frame.size else 0
    positive_lookup = _frame_label_lookup(frame_df, positive_masks, max_label)
    object_positive_lookup = _frame_label_lookup(frame_df, object_positive_masks, max_label)
    veto_lookup = _frame_label_lookup(frame_df, veto_masks, max_label)
    broad_lookup = _frame_label_lookup(frame_df, broad_masks, max_label)
    seed_hit = _hit_counts(mask_by_frame, positive_lookup)
    object_seed_hit = _hit_counts(mask_by_frame, object_positive_lookup)
    veto_hit = _hit_counts(mask_by_frame, veto_lookup)
    broad_hit = _hit_counts(mask_by_frame, broad_lookup)
    visible_count = np.sum(mask_by_frame > 0, axis=0).astype(np.int16)

    variant_rows: list[dict[str, Any]] = []
    obs_by_variant: dict[str, pd.DataFrame] = {}
    for variant in VARIANTS:
        use_seed_hit = object_seed_hit if str(variant.get("anchor_source", "all_positive")) == "object_like_non_broad_positive" else seed_hit
        row, obs_rows = _score_variant(
            scene_id=scene_id,
            variant=variant,
            mask_by_frame=mask_by_frame,
            xyz=xyz,
            frame_df=frame_df,
            positive_obs=positive_obs,
            seed_hit=use_seed_hit,
            veto_hit=veto_hit,
            broad_hit=broad_hit,
            visible_count=visible_count,
            obs_meta=obs_meta,
        )
        variant_rows.append(row)
        obs_by_variant[str(variant["variant_id"])] = obs_rows

    variant_path = scene_out / "gaussian_neighborhood_variant_rows.csv"
    _write_csv(variant_path, variant_rows)
    best = max(
        variant_rows,
        key=lambda r: (
            bool(r.get("phase9g_object_like_extension_gate_pass", False)),
            int(r.get("induced_object_like_obs_count", 0)),
            -float(r.get("induced_broad_obs_rate", 1.0)),
            -int(r.get("selected_gaussian_count", 0)),
        ),
    )
    best_obs_path = scene_out / "best_variant_selected_observation_rows.csv"
    obs_by_variant[str(best["variant_id"])].to_csv(best_obs_path, index=False)
    pass_any = any(bool(r["phase9g_object_like_extension_gate_pass"]) for r in variant_rows)
    seed_gaussians_hit1 = int(np.count_nonzero(seed_hit > 0))
    seed_gaussians_hit2 = int(np.count_nonzero(seed_hit >= 2))
    return {
        "schema_version": "stream4d_v103_phase9g_scene_summary_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "phase9e_root": _rel(phase9e_root),
        "da3_projection_manifest": manifest,
        "positive_anchor_mask_observation_count": int(len(positive_obs)),
        "object_like_non_broad_positive_anchor_mask_observation_count": int(len(object_positive_masks)),
        "veto_mask_observation_count": int(len(veto_masks)),
        "broad_mask_observation_count": int(len(broad_masks)),
        "gaussian_count": int(mask_by_frame.shape[1]),
        "seed_gaussian_hit1_count": seed_gaussians_hit1,
        "seed_gaussian_hit2_count": seed_gaussians_hit2,
        "object_seed_gaussian_hit1_count": int(np.count_nonzero(object_seed_hit > 0)),
        "object_seed_gaussian_hit2_count": int(np.count_nonzero(object_seed_hit >= 2)),
        "variant_count": len(VARIANTS),
        "phase9g_object_like_extension_gate_pass": pass_any,
        "best_variant_id": best["variant_id"],
        "best_selected_gaussian_count": best["selected_gaussian_count"],
        "best_induced_obs_count": best["induced_obs_count"],
        "best_induced_object_like_obs_count": best["induced_object_like_obs_count"],
        "best_induced_broad_obs_count": best["induced_broad_obs_count"],
        "best_induced_broad_obs_rate": best["induced_broad_obs_rate"],
        "blocker": "" if pass_any else "da3_seed_gaussian_neighborhood_extension_not_object_like_or_broad_safe",
        "uses_gt_for_prediction": False,
        "uses_gt_for_coverage_gate": False,
        "outputs": {
            "gaussian_neighborhood_variant_rows": _rel(variant_path),
            "best_variant_selected_observation_rows": _rel(best_obs_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose DA3 Gaussian neighborhoods induced by reliable D4RT seed carriers.")
    parser.add_argument("--phase9e-root", default=str(DEFAULT_PHASE9E_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--projection-cache-root",
        default="",
        help="Optional prior Phase9g output root containing per-scene da3_projection_cache directories.",
    )
    parser.add_argument("--scene", choices=["all", "scene0011_00", "scene0050_00"], default="all")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    phase9e_root = _project_phase(args.phase9e_root)
    out = _project_phase(args.output_root)
    projection_cache_root = _project_phase(args.projection_cache_root) if str(args.projection_cache_root).strip() else None
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    scenes = list(SCENE_SPECS.keys()) if args.scene == "all" else [str(args.scene)]
    scene_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for scene in scenes:
        try:
            scene_rows.append(_process_scene(scene, phase9e_root, out, int(args.cupy_device_id), projection_cache_root))
        except Exception as exc:
            failure = {
                "schema_version": "stream4d_v103_phase9g_failure_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "blocker": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "uses_gt_for_prediction": False,
            }
            scene_rows.append(failure)
            failure_rows.append(failure)

    pass_count = sum(bool(row.get("phase9g_object_like_extension_gate_pass", False)) for row in scene_rows)
    _write_csv(out / "scene_summary_rows.csv", scene_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    decision = (
        "PASS_PHASE9G_DA3_SEED_GAUSSIAN_NEIGHBORHOOD_EXTENSION"
        if pass_count == len(scenes) and not failure_rows
        else "PARTIAL_PHASE9G_DA3_SEED_GAUSSIAN_NEIGHBORHOOD_EXTENSION"
        if pass_count > 0 and not failure_rows
        else "NO_GO_PHASE9G_DA3_SEED_GAUSSIAN_NEIGHBORHOOD_EXTENSION"
    )
    summary = {
        "schema_version": "stream4d_v103_phase9g_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "scene_count": len(scenes),
        "pass_scene_count": pass_count,
        "failure_count": len(failure_rows),
        "plan_doc": _rel(PLAN_DOC),
        "projection_cache_root": "" if projection_cache_root is None else _rel(projection_cache_root),
        "truthfulness_note": (
            "This diagnostic selects DA3 Gaussians using GT-free D4RT positive-anchor mask hits and optional seed-local "
            "kNN expansion. Object-like/broad coverage gates use GT-free v103 mask metadata; no AP predictions are emitted."
        ),
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "scene_summary_rows": _rel(out / "scene_summary_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if decision.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
