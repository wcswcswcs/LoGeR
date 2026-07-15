#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v104_lingbot_v103_affinity_field_adapter"
DEFAULT_OUT = AUDIT_ROOT / "v104_lingbot_map_only_phase11_v103_affinity_field_adapter_first32"
DEFAULT_SUPPORT_ROWS = AUDIT_ROOT / "v104_lingbot_map_only_phase7_real_mask_support_rows/real_mask_support_rows.csv"
DEFAULT_SELECTED_ROWS = AUDIT_ROOT / "v87_phase1_mv_input_generation/frame_mask_selected_rows.csv"
DEFAULT_PHASE2_SCENE0011 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_PHASE2_SCENE0050 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_first32"
SKETCH_SEED = 10317

if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from geometry_provider.lingbot_map_provider import LingBotMapGeometryProvider  # noqa: E402


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _history_priority(history_id: str) -> int:
    value = str(history_id)
    if "confirmed_gain" in value:
        return 4
    if value.startswith("confirmed:"):
        return 3
    if "state_priority:confirmed" in value:
        return 2
    return 1


def _dedupe_frame_mask_rows(
    rows: list[dict[str, str]],
    selected_meta: dict[str, dict[str, str]],
    *,
    enabled: bool,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], int]:
    if not enabled:
        return rows, [], 0
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("scene_id", ""),
            int(float(row.get("source_frame_id", -1))),
            int(float(row.get("mask_id", -1))),
        )
        grouped[key].append(row)

    kept: list[dict[str, str]] = []
    duplicate_rows: list[dict[str, Any]] = []
    duplicate_group_count = 0
    for key, group in grouped.items():
        if len(group) <= 1:
            kept.extend(group)
            continue
        duplicate_group_count += 1

        def rank(row: dict[str, str]) -> tuple[int, int, int, int]:
            meta = selected_meta.get(row.get("candidate_row_id", ""), {})
            broad = _as_bool(meta.get("broad_mask_flag", "False"))
            object_like = _as_bool(meta.get("object_mask_ownership_allowed", "True")) and _as_bool(
                meta.get("adapter_candidate_valid", "True")
            )
            candidate = int(float(row.get("candidate_row_id", 0) or 0))
            return (int(not broad), int(object_like), _history_priority(row.get("history_id", "")), -candidate)

        chosen = max(group, key=rank)
        kept.append(chosen)
        chosen_id = chosen.get("candidate_row_id", "")
        for row in group:
            if row is chosen:
                continue
            duplicate_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_v103_affinity_dedupe_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": key[0],
                    "source_frame_id": key[1],
                    "mask_id": key[2],
                    "kept_candidate_row_id": chosen_id,
                    "dropped_candidate_row_id": row.get("candidate_row_id", ""),
                    "kept_history_id": chosen.get("history_id", ""),
                    "dropped_history_id": row.get("history_id", ""),
                    "dedupe_key": "scene_source_frame_mask",
                    "uses_gt_for_prediction": False,
                }
            )
    return kept, duplicate_rows, duplicate_group_count


def _load_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return np.empty((0, 0), dtype=np.int64)
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _mask_stats(mask_path: Path, mask_id: int) -> tuple[int, bool]:
    label = _load_label(mask_path)
    if label.size == 0:
        return 0, False
    count = int(np.count_nonzero(label == int(mask_id)))
    return count, count > 0


def _filter_mask_interior_ids(
    *,
    label: np.ndarray,
    mask_id: int,
    sample_xy: np.ndarray | None,
    sample_image_shape: tuple[int, int] | None,
    valid_ids: np.ndarray,
    erode_pixels: float,
) -> np.ndarray:
    if erode_pixels <= 0.0 or valid_ids.size == 0 or label.size == 0 or sample_xy is None:
        return valid_ids
    xy = np.asarray(sample_xy, dtype=np.float32).reshape(-1, 2)
    valid_ids = valid_ids[valid_ids < xy.shape[0]]
    if valid_ids.size == 0:
        return valid_ids
    mask = (label == int(mask_id)).astype(np.uint8)
    if not np.any(mask):
        return valid_ids[:0]
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    mh, mw = label.shape[:2]
    if sample_image_shape is None:
        sh, sw = mh, mw
    else:
        sh, sw = sample_image_shape
    scale_x = float(max(mw - 1, 1)) / float(max(sw - 1, 1))
    scale_y = float(max(mh - 1, 1)) / float(max(sh - 1, 1))
    pts = xy[valid_ids]
    mx = np.rint(pts[:, 0] * scale_x).astype(np.int64)
    my = np.rint(pts[:, 1] * scale_y).astype(np.int64)
    in_bounds = (mx >= 0) & (mx < mw) & (my >= 0) & (my < mh)
    keep = np.zeros((valid_ids.shape[0],), dtype=bool)
    if np.any(in_bounds):
        keep[in_bounds] = dist[my[in_bounds], mx[in_bounds]] >= float(erode_pixels)
    return valid_ids[keep]


def _carrier_key_array(
    points: np.ndarray,
    *,
    voxel_size: float,
    carrier_key_mode: str,
    view_ray_bin_size: float,
    normal_bin_size: float,
    camera_center: np.ndarray | None,
    normals: np.ndarray | None,
) -> np.ndarray:
    voxels = np.floor(points.astype(np.float64) / float(voxel_size)).astype(np.int64)
    if carrier_key_mode == "world_voxel":
        return voxels
    if carrier_key_mode not in {"surface_view_voxel", "selective_surface_view_voxel", "surface_normal_voxel"}:
        raise ValueError(f"unknown carrier key mode: {carrier_key_mode}")
    if carrier_key_mode == "surface_normal_voxel":
        if normals is None or normals.shape[0] != voxels.shape[0]:
            normal_bins = np.zeros((voxels.shape[0], 3), dtype=np.int64)
        else:
            ns = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
            ns /= np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-12)
            largest = np.argmax(np.abs(ns), axis=1)
            signs = np.take_along_axis(ns, largest[:, None], axis=1).reshape(-1)
            ns[signs < 0.0] *= -1.0
            bin_size = max(float(normal_bin_size), 1e-6)
            normal_bins = np.floor((ns + 1.0) / bin_size).astype(np.int64)
        return np.concatenate([voxels, normal_bins], axis=1)
    if camera_center is None:
        ray_bins = np.zeros((voxels.shape[0], 3), dtype=np.int64)
    else:
        rays = points.astype(np.float64) - np.asarray(camera_center, dtype=np.float64).reshape(1, 3)
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        bin_size = max(float(view_ray_bin_size), 1e-6)
        ray_bins = np.floor((rays + 1.0) / bin_size).astype(np.int64)
    return np.concatenate([voxels, ray_bins], axis=1)


def _estimate_sample_normals(points: np.ndarray, xy: np.ndarray | None, *, k: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    if xy is None:
        return np.zeros_like(pts, dtype=np.float32)
    pixels = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    n = min(pts.shape[0], pixels.shape[0])
    normals = np.zeros_like(pts, dtype=np.float32)
    if n < 4:
        return normals
    pts_n = pts[:n]
    pixels_n = pixels[:n]
    finite = np.isfinite(pts_n).all(axis=1) & np.isfinite(pixels_n).all(axis=1)
    valid_idx = np.flatnonzero(finite)
    if valid_idx.size < 4:
        return normals
    query_xy = pixels_n[valid_idx]
    query_pts = pts_n[valid_idx]
    kk = max(4, min(int(k), int(valid_idx.size)))
    tree = cKDTree(query_xy)
    _dist, nbr_idx = tree.query(query_xy, k=kk)
    if nbr_idx.ndim == 1:
        nbr_idx = nbr_idx[:, None]
    neighbors = query_pts[nbr_idx]
    centered = neighbors - query_pts[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered, centered) / float(max(kk, 1))
    vals, vecs = np.linalg.eigh(cov)
    del vals
    local_normals = vecs[:, :, 0].astype(np.float32, copy=False)
    largest = np.argmax(np.abs(local_normals), axis=1)
    signs = np.take_along_axis(local_normals, largest[:, None], axis=1).reshape(-1)
    local_normals[signs < 0.0] *= -1.0
    local_normals /= np.maximum(np.linalg.norm(local_normals, axis=1, keepdims=True), 1e-12)
    normals[valid_idx] = local_normals
    return normals


def _hash_bucket(mask_idx: np.ndarray, sketch_dim: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(mask_idx, dtype=np.int64)
    bucket = ((idx * 2654435761 + SKETCH_SEED) % int(sketch_dim)).astype(np.int64)
    sign = np.where(((idx * 1103515245 + SKETCH_SEED) % 2) == 0, 1.0, -1.0).astype(np.float32)
    return bucket, sign


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    out = arr / np.maximum(norm, 1e-12)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32, copy=False)


def _frame_centered_feature(feature: np.ndarray, mask_frame: np.ndarray, mask_is_object: np.ndarray, beta: float) -> np.ndarray:
    out = np.asarray(feature, dtype=np.float32).copy()
    if float(beta) <= 0.0:
        return out
    for frame in sorted(set(int(v) for v in mask_frame.tolist())):
        idx = np.flatnonzero((mask_frame == int(frame)) & mask_is_object)
        if idx.size < 2:
            continue
        center = np.mean(out[idx], axis=0, dtype=np.float32)
        out[idx] -= float(beta) * center[None, :]
    return _normalize_rows(out)


def _mask_weights(support_count: np.ndarray, mask_is_object: np.ndarray, mask_is_broad: np.ndarray) -> np.ndarray:
    support = np.asarray(support_count, dtype=np.float64)
    positive = support[support > 0]
    p95 = float(np.percentile(positive, 95)) if positive.size else 1.0
    rarity = np.clip(np.log1p(p95) / np.maximum(np.log1p(support), 1e-6), 0.05, 4.0)
    quality = np.where(mask_is_object, 1.0, np.where(mask_is_broad, 0.05, 0.40)).astype(np.float64)
    return np.maximum(quality * rarity, 1e-4).astype(np.float32)


def _provider(
    cache: dict[str, LingBotMapGeometryProvider],
    root: Path,
    *,
    max_points_per_frame: int,
    min_confidence: float | None,
) -> LingBotMapGeometryProvider:
    key = root.as_posix()
    if key not in cache:
        cache[key] = LingBotMapGeometryProvider(
            geometry_root=root,
            max_points_per_frame=max_points_per_frame,
            min_confidence=min_confidence,
        )
    return cache[key]


def _build_scene(scene: str, rows: list[dict[str, str]], phase2_summary: dict[str, Any], selected_meta: dict[str, dict[str, str]], args: argparse.Namespace, out: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    scene_out = out / scene
    scene_out.mkdir(parents=True, exist_ok=True)
    frame_ids = [int(v) for v in phase2_summary["frame_ids"]]
    frame_to_local = {frame: idx for idx, frame in enumerate(frame_ids)}
    mask_root = _project(phase2_summary["mask_root"])
    filtered_raw = [row for row in rows if row.get("scene_id") == scene and int(float(row.get("source_frame_id", -1))) in frame_to_local]
    filtered, dedupe_rows, duplicate_group_count = _dedupe_frame_mask_rows(
        filtered_raw,
        selected_meta,
        enabled=bool(args.dedupe_frame_mask),
    )

    providers: dict[str, LingBotMapGeometryProvider] = {}
    sample_cache: dict[tuple[str, int], Any] = {}
    normal_cache: dict[tuple[str, int], np.ndarray] = {}
    carrier_key_to_idx: dict[tuple[int, ...], int] = {}
    carrier_frames: dict[int, set[int]] = defaultdict(set)
    carrier_object_hits: dict[int, int] = defaultdict(int)
    carrier_broad_hits: dict[int, int] = defaultdict(int)
    carrier_incidence_count: dict[int, int] = defaultdict(int)
    carrier_subkeys: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    obs_rows: list[dict[str, Any]] = []
    obs_key_records: list[dict[str, Any]] = []
    world_observation_count: dict[tuple[int, int, int], int] = defaultdict(int)
    inc_acc: dict[tuple[int, int], float] = defaultdict(float)
    failure_rows: list[dict[str, Any]] = []

    for row in filtered:
        candidate_row_id = row.get("candidate_row_id", "")
        source_frame = int(float(row.get("source_frame_id", -1)))
        bss_frame = int(float(row.get("bss_frame_id", -1)))
        mask_id = int(float(row.get("mask_id", -1)))
        support_path = _project(row.get("support_point_ids_path", ""))
        lingbot_root = _project(row.get("lingbot_root", ""))
        mask_path = mask_root / f"{source_frame}.png"
        meta = selected_meta.get(candidate_row_id, {})
        object_like = _as_bool(meta.get("object_mask_ownership_allowed", "True")) and _as_bool(
            meta.get("adapter_candidate_valid", "True")
        )
        broad = _as_bool(meta.get("broad_mask_flag", "False"))
        label = _load_label(mask_path)
        mask_area = int(np.count_nonzero(label == int(mask_id))) if label.size else 0
        mask_present = mask_area > 0
        if not mask_present:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_v103_affinity_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "candidate_row_id": candidate_row_id,
                    "failure_id": "MISSING_MASK_PIXELS",
                    "mask_path": _rel(mask_path),
                    "mask_id": mask_id,
                }
            )
            continue
        try:
            support_ids = np.asarray(np.load(support_path), dtype=np.int64)
        except Exception as exc:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_v103_affinity_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "candidate_row_id": candidate_row_id,
                    "failure_id": "SUPPORT_IDS_LOAD_FAILED",
                    "error": str(exc),
                }
            )
            continue
        prov = _provider(
            providers,
            lingbot_root,
            max_points_per_frame=int(args.max_points_per_frame),
            min_confidence=args.min_confidence,
        )
        sample_key = (lingbot_root.as_posix(), bss_frame)
        if sample_key not in sample_cache:
            sample_cache[sample_key] = prov.load_frame_samples(bss_frame)
        samples = sample_cache[sample_key]
        points = np.asarray(samples.points, dtype=np.float32).reshape(-1, 3)
        if sample_key not in normal_cache:
            normal_cache[sample_key] = _estimate_sample_normals(points, samples.xy, k=int(args.normal_knn))
        valid_ids = support_ids[(support_ids >= 0) & (support_ids < points.shape[0])]
        raw_valid_support_count = int(valid_ids.size)
        valid_ids = _filter_mask_interior_ids(
            label=label,
            mask_id=mask_id,
            sample_xy=samples.xy,
            sample_image_shape=samples.image_shape,
            valid_ids=valid_ids,
            erode_pixels=float(args.mask_interior_erode_pixels),
        )
        if valid_ids.size == 0:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_v103_affinity_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "candidate_row_id": candidate_row_id,
                    "failure_id": "EMPTY_VALID_SUPPORT",
                    "support_point_count": int(support_ids.size),
                    "frame_point_count": int(points.shape[0]),
                    "raw_valid_support_count": raw_valid_support_count,
                    "mask_interior_erode_pixels": float(args.mask_interior_erode_pixels),
                }
            )
            continue
        obs_idx = len(obs_rows)
        local_frame = int(frame_to_local[source_frame])
        obs_rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_v103_affinity_mask_observation_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "mask_observation_index": obs_idx,
                "candidate_row_id": candidate_row_id,
                "frame_local_index": local_frame,
                "source_frame_id": source_frame,
                "mask_id": mask_id,
                "history_id": row.get("history_id", ""),
                "mask_is_object_like": bool(object_like),
                "mask_is_broad": bool(broad),
                "mask_area": mask_area,
                "support_point_count": int(valid_ids.size),
                "raw_valid_support_count": raw_valid_support_count,
                "mask_interior_erode_pixels": float(args.mask_interior_erode_pixels),
                "uses_d4rt_for_prediction": False,
                "uses_da3_for_prediction": False,
                "uses_gt_for_prediction": False,
            }
        )
        point_subset = points[valid_ids]
        normal_subset = normal_cache[sample_key][valid_ids] if normal_cache[sample_key].shape[0] >= points.shape[0] else None
        pose = getattr(prov, "_poses", {}).get(int(bss_frame))
        camera_center = None if pose is None else np.asarray(pose[:3, 3], dtype=np.float64)
        carrier_keys = _carrier_key_array(
            point_subset,
            voxel_size=float(args.voxel_size),
            carrier_key_mode=str(args.carrier_key_mode)
            if str(args.carrier_key_mode) == "surface_normal_voxel"
            else "surface_view_voxel",
            view_ray_bin_size=float(args.view_ray_bin_size),
            normal_bin_size=float(args.normal_bin_size),
            camera_center=camera_center,
            normals=normal_subset,
        )
        unique_keys, counts = np.unique(carrier_keys, axis=0, return_counts=True)
        world_keys = np.unique(unique_keys[:, :3], axis=0)
        for world_key in world_keys.tolist():
            world_observation_count[tuple(int(value) for value in world_key)] += 1
        obs_key_records.append(
            {
                "obs_idx": int(obs_idx),
                "source_frame": int(source_frame),
                "object_like": bool(object_like),
                "broad": bool(broad),
                "carrier_keys": unique_keys,
                "counts": counts,
            }
        )

    for record in obs_key_records:
        local_counts: dict[tuple[int, ...], int] = defaultdict(int)
        local_subkeys: dict[tuple[int, ...], set[tuple[int, ...]]] = defaultdict(set)
        for carrier_key, count in zip(record["carrier_keys"].tolist(), record["counts"].tolist()):
            world_key = tuple(int(value) for value in carrier_key[:3])
            subkey = tuple(int(value) for value in carrier_key[3:])
            mode = str(args.carrier_key_mode)
            if mode == "world_voxel":
                key: tuple[int, ...] = world_key
            elif mode in {"surface_view_voxel", "surface_normal_voxel"}:
                key = tuple(int(value) for value in carrier_key)
            elif mode == "selective_surface_view_voxel":
                if world_observation_count[world_key] >= int(args.selective_surface_view_min_world_observations):
                    key = tuple(int(value) for value in carrier_key)
                else:
                    key = world_key
            else:
                raise ValueError(f"unknown carrier key mode: {mode}")
            local_counts[key] += int(count)
            local_subkeys[key].add(subkey)
        for key, count in local_counts.items():
            carrier_idx = carrier_key_to_idx.setdefault(key, len(carrier_key_to_idx))
            carrier_frames[carrier_idx].add(int(record["source_frame"]))
            carrier_subkeys[carrier_idx].update(local_subkeys[key])
            carrier_incidence_count[carrier_idx] += 1
            if bool(record["object_like"]):
                carrier_object_hits[carrier_idx] += 1
            if bool(record["broad"]):
                carrier_broad_hits[carrier_idx] += 1
            inc_acc[(carrier_idx, int(record["obs_idx"]))] += min(
                1.0,
                np.sqrt(float(count)) / max(float(args.support_count_scale), 1e-6),
            )

    mask_count = len(obs_rows)
    carrier_count = len(carrier_key_to_idx)
    if mask_count == 0 or carrier_count == 0:
        raise RuntimeError(f"{scene}: no valid LingBot primitive incidence rows")

    mask_frame = np.asarray([int(row["frame_local_index"]) for row in obs_rows], dtype=np.int64)
    mask_label = np.asarray([int(row["mask_id"]) for row in obs_rows], dtype=np.int64)
    mask_is_object = np.asarray([bool(row["mask_is_object_like"]) for row in obs_rows], dtype=bool)
    mask_is_broad = np.asarray([bool(row["mask_is_broad"]) for row in obs_rows], dtype=bool)
    raw_support_count = np.zeros((mask_count,), dtype=np.int64)
    carrier_idx = []
    mask_idx = []
    b_ia = []
    for (ci, mi), val in sorted(inc_acc.items()):
        carrier_idx.append(ci)
        mask_idx.append(mi)
        b_ia.append(float(val))
        raw_support_count[mi] += 1
    carrier_idx_np = np.asarray(carrier_idx, dtype=np.int64)
    mask_idx_np = np.asarray(mask_idx, dtype=np.int64)
    b_ia_np = np.asarray(b_ia, dtype=np.float32)

    frame_count = np.asarray([len(carrier_frames[i]) for i in range(carrier_count)], dtype=np.int64)
    incidence_count = np.asarray([carrier_incidence_count[i] for i in range(carrier_count)], dtype=np.float32)
    subkey_count = np.asarray([len(carrier_subkeys[i]) for i in range(carrier_count)], dtype=np.float32)
    object_hits = np.asarray([carrier_object_hits[i] for i in range(carrier_count)], dtype=np.float32)
    broad_hits = np.asarray([carrier_broad_hits[i] for i in range(carrier_count)], dtype=np.float32)
    object_ratio = object_hits / np.maximum(incidence_count, 1.0)
    broad_ratio = broad_hits / np.maximum(incidence_count, 1.0)
    is_anchor = (frame_count >= int(args.anchor_min_frames)) & (object_ratio >= float(args.anchor_min_object_ratio)) & (broad_ratio <= float(args.anchor_max_broad_ratio))
    is_support = np.ones((carrier_count,), dtype=bool)
    is_veto = broad_ratio >= float(args.veto_broad_ratio)

    weights = _mask_weights(raw_support_count, mask_is_object, mask_is_broad)
    carrier_idf_alpha = float(args.carrier_incidence_idf_alpha)
    if carrier_idf_alpha > 0.0:
        carrier_idf = np.power(np.maximum(incidence_count, 1.0), -carrier_idf_alpha).astype(np.float32)
    else:
        carrier_idf = np.ones((carrier_count,), dtype=np.float32)
    subkey_idf_alpha = float(args.carrier_subkey_idf_alpha)
    if subkey_idf_alpha > 0.0:
        carrier_subkey_idf = np.power(np.maximum(subkey_count, 1.0), -subkey_idf_alpha).astype(np.float32)
    else:
        carrier_subkey_idf = np.ones((carrier_count,), dtype=np.float32)
    role_factor = np.where(is_anchor[carrier_idx_np], 1.0, float(args.support_lambda)).astype(np.float32)
    role_factor *= np.where(is_veto[carrier_idx_np], float(args.veto_attenuation), 1.0).astype(np.float32)
    role_factor *= carrier_idf[carrier_idx_np]
    role_factor *= carrier_subkey_idf[carrier_idx_np]
    weighted_b = b_ia_np * role_factor
    keep = np.isfinite(weighted_b) & (weighted_b > 0.0)
    carrier_idx_np = carrier_idx_np[keep]
    mask_idx_np = mask_idx_np[keep]
    weighted_b = weighted_b[keep]

    bucket, sign = _hash_bucket(mask_idx_np, int(args.sketch_dim))
    raw = np.zeros((carrier_count, int(args.sketch_dim)), dtype=np.float32)
    values = np.sqrt(weights[mask_idx_np]).astype(np.float32) * weighted_b.astype(np.float32) * sign
    np.add.at(raw, (carrier_idx_np, bucket), values)
    primitive = _normalize_rows(raw)

    feature = np.zeros((mask_count, int(args.sketch_dim)), dtype=np.float32)
    incidence_by_mask: list[np.ndarray] = [np.flatnonzero(mask_idx_np == i).astype(np.int64) for i in range(mask_count)]
    for mi, rows_for_mask in enumerate(incidence_by_mask):
        if rows_for_mask.size == 0:
            continue
        carriers = carrier_idx_np[rows_for_mask]
        vals = weighted_b[rows_for_mask].astype(np.float32)
        vecs = raw[carriers].copy()
        own_bucket, own_sign = _hash_bucket(np.asarray([mi], dtype=np.int64), int(args.sketch_dim))
        own_contrib = np.sqrt(float(weights[mi])) * vals * float(own_sign[0])
        vecs[:, int(own_bucket[0])] -= own_contrib
        vecs = _normalize_rows(vecs)
        pooled = np.sum(vecs * vals[:, None], axis=0) / max(float(np.sum(vals)), 1e-12)
        norm = float(np.linalg.norm(pooled))
        if norm > 0.0:
            feature[mi] = pooled / norm
    feature = _frame_centered_feature(feature, mask_frame, mask_is_object, float(args.frame_center_beta))

    feature_path = scene_out / "mask_level_feature.pt"
    torch.save(
        {
            "schema_version": "stream4d_v104_lingbot_v103_affinity_mask_level_feature_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "variant_id": "LBPF1_lingbot_voxel_primitive_field_anchor_support",
            "static_feature_source": "lingbot_voxel_primitive_incidence_countsketch_leave_one_out",
            "pair_affinity_mode": "static_feature_cosine",
            "mask_observation_index": torch.arange(mask_count, dtype=torch.int64),
            "mask_frame": torch.as_tensor(mask_frame, dtype=torch.int64),
            "mask_label": torch.as_tensor(mask_label, dtype=torch.int64),
            "mask_is_object_like": torch.as_tensor(mask_is_object, dtype=torch.bool),
            "mask_is_broad": torch.as_tensor(mask_is_broad, dtype=torch.bool),
            "support_count": torch.as_tensor(raw_support_count, dtype=torch.int64),
            "feature": torch.as_tensor(feature, dtype=torch.float16),
            "carrier_id": torch.arange(carrier_count, dtype=torch.int64),
            "is_A_anchor": torch.as_tensor(is_anchor, dtype=torch.bool),
            "is_S_support": torch.as_tensor(is_support, dtype=torch.bool),
            "is_V_veto": torch.as_tensor(is_veto, dtype=torch.bool),
            "voxel_size": float(args.voxel_size),
            "carrier_key_mode": str(args.carrier_key_mode),
            "view_ray_bin_size": float(args.view_ray_bin_size),
            "normal_bin_size": float(args.normal_bin_size),
            "normal_knn": int(args.normal_knn),
            "selective_surface_view_min_world_observations": int(args.selective_surface_view_min_world_observations),
            "sketch_dim": int(args.sketch_dim),
            "support_lambda": float(args.support_lambda),
            "veto_attenuation": float(args.veto_attenuation),
            "frame_center_beta": float(args.frame_center_beta),
            "carrier_incidence_idf_alpha": float(args.carrier_incidence_idf_alpha),
            "carrier_subkey_idf_alpha": float(args.carrier_subkey_idf_alpha),
            "mask_interior_erode_pixels": float(args.mask_interior_erode_pixels),
            "uses_gt": False,
            "uses_future": False,
        },
        feature_path,
    )
    incidence_path = scene_out / "primitive_incidence_sparse.pt"
    torch.save(
        {
            "schema_version": "stream4d_v104_lingbot_v103_affinity_primitive_incidence_sparse_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "carrier_local_index": torch.as_tensor(carrier_idx_np, dtype=torch.int64),
            "mask_observation_index": torch.as_tensor(mask_idx_np, dtype=torch.int64),
            "frame_local_index": torch.as_tensor(mask_frame[mask_idx_np], dtype=torch.int64),
            "mask_id": torch.as_tensor(mask_label[mask_idx_np], dtype=torch.int64),
            "B_ia": torch.as_tensor(weighted_b.astype(np.float32), dtype=torch.float32),
            "mask_frame": torch.as_tensor(mask_frame, dtype=torch.int64),
            "mask_label": torch.as_tensor(mask_label, dtype=torch.int64),
            "mask_is_object_like": torch.as_tensor(mask_is_object, dtype=torch.bool),
            "mask_is_broad": torch.as_tensor(mask_is_broad, dtype=torch.bool),
            "mask_weight": torch.as_tensor(weights, dtype=torch.float32),
            "is_A_anchor": torch.as_tensor(is_anchor, dtype=torch.bool),
            "is_S_support": torch.as_tensor(is_support, dtype=torch.bool),
            "is_V_veto": torch.as_tensor(is_veto, dtype=torch.bool),
            "uses_gt": False,
            "uses_future": False,
        },
        incidence_path,
    )
    _write_csv(scene_out / "mask_observation_rows.csv", obs_rows)
    _write_csv(scene_out / "failure_rows.csv", failure_rows)
    _write_csv(scene_out / "dedupe_rows.csv", dedupe_rows)
    role_rows = [
        {
            "schema_version": "stream4d_v104_lingbot_v103_affinity_carrier_role_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "carrier_local_index": int(i),
            "frame_count": int(frame_count[i]),
            "incidence_count": int(incidence_count[i]),
            "subkey_count": int(subkey_count[i]),
            "object_ratio": float(object_ratio[i]),
            "broad_ratio": float(broad_ratio[i]),
            "is_A_anchor": bool(is_anchor[i]),
            "is_S_support": bool(is_support[i]),
            "is_V_veto": bool(is_veto[i]),
        }
        for i in range(carrier_count)
    ]
    _write_csv(scene_out / "carrier_role_rows.csv", role_rows[: int(args.max_role_rows)])
    summary = {
        "schema_version": "stream4d_v104_lingbot_v103_affinity_scene_summary_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "frame_count": len(frame_ids),
        "mask_observation_count": mask_count,
        "raw_mask_observation_count": len(filtered_raw),
        "dedupe_frame_mask": bool(args.dedupe_frame_mask),
        "duplicate_frame_mask_group_count": int(duplicate_group_count),
        "duplicate_dropped_observation_count": int(len(dedupe_rows)),
        "carrier_count": carrier_count,
        "incidence_count": int(carrier_idx_np.shape[0]),
        "anchor_carrier_count": int(np.count_nonzero(is_anchor)),
        "support_carrier_count": int(np.count_nonzero(is_support)),
        "veto_carrier_count": int(np.count_nonzero(is_veto)),
        "carrier_key_mode": str(args.carrier_key_mode),
        "view_ray_bin_size": float(args.view_ray_bin_size),
        "normal_bin_size": float(args.normal_bin_size),
        "normal_knn": int(args.normal_knn),
        "selective_surface_view_min_world_observations": int(args.selective_surface_view_min_world_observations),
        "carrier_incidence_idf_alpha": float(args.carrier_incidence_idf_alpha),
        "carrier_subkey_idf_alpha": float(args.carrier_subkey_idf_alpha),
        "mask_interior_erode_pixels": float(args.mask_interior_erode_pixels),
        "mask_feature_valid_rate": float(np.mean(np.linalg.norm(feature, axis=1) > 0.0)) if mask_count else 0.0,
        "object_like_mask_feature_valid_rate": float(np.mean(np.linalg.norm(feature[mask_is_object], axis=1) > 0.0)) if np.any(mask_is_object) else 0.0,
        "feature_path": _rel(feature_path),
        "incidence_path": _rel(incidence_path),
        "failure_count": len(failure_rows),
        "uses_d4rt_for_prediction": False,
        "uses_da3_for_prediction": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(scene_out / "summary.json", summary)
    return summary, obs_rows, failure_rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    support_rows = _read_csv(_project(args.support_rows))
    selected_meta = {row.get("candidate_row_id", ""): row for row in _read_csv(_project(args.selected_rows))}
    phase2_roots = {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }
    scene_summaries: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    for scene, root in phase2_roots.items():
        summary, _obs, failures = _build_scene(scene, support_rows, _read_json(root / "summary.json"), selected_meta, args, out)
        scene_summaries.append(summary)
        all_failures.extend(failures)

    artifact_rows = [
        {
            "schema_version": "stream4d_v104_lingbot_v103_affinity_artifact_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": row["scene_id"],
            "role": "mask_level_feature",
            "path": row["feature_path"],
            "exists": _project(row["feature_path"]).exists(),
        }
        for row in scene_summaries
    ]
    _write_csv(out / "artifact_rows.csv", artifact_rows)
    _write_csv(out / "failure_rows.csv", all_failures)
    adapter_pass = bool(scene_summaries) and all(row["failure_count"] == 0 for row in scene_summaries)
    summary = {
        "schema_version": "stream4d_v104_lingbot_v103_affinity_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix": time.time(),
        "adapter_pass": adapter_pass,
        "phase5_pass": adapter_pass,
        "decision": "PASS_ENTER_V103_PHASE6_MASK_CLUSTERING" if adapter_pass else "NO_GO_REPAIR_LINGBOT_V103_AFFINITY_ADAPTER",
        "method_boundary": "v103 primitive affinity field preserved; D4RT/DA3 carrier provider replaced by LingBot voxel primitive carriers",
        "scene_summaries": scene_summaries,
        "support_rows": _rel(_project(args.support_rows)),
        "selected_rows": _rel(_project(args.selected_rows)),
        "voxel_size": float(args.voxel_size),
        "carrier_key_mode": str(args.carrier_key_mode),
        "view_ray_bin_size": float(args.view_ray_bin_size),
        "normal_bin_size": float(args.normal_bin_size),
        "normal_knn": int(args.normal_knn),
        "selective_surface_view_min_world_observations": int(args.selective_surface_view_min_world_observations),
        "sketch_dim": int(args.sketch_dim),
        "support_lambda": float(args.support_lambda),
        "veto_attenuation": float(args.veto_attenuation),
        "frame_center_beta": float(args.frame_center_beta),
        "carrier_incidence_idf_alpha": float(args.carrier_incidence_idf_alpha),
        "carrier_subkey_idf_alpha": float(args.carrier_subkey_idf_alpha),
        "mask_interior_erode_pixels": float(args.mask_interior_erode_pixels),
        "dedupe_frame_mask": bool(args.dedupe_frame_mask),
        "uses_d4rt_for_prediction": False,
        "uses_da3_for_prediction": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
        "runtime_sec": round(time.time() - t0, 3),
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a LingBot-backed v103/R5 primitive affinity field adapter.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--support-rows", default=str(DEFAULT_SUPPORT_ROWS))
    parser.add_argument("--selected-rows", default=str(DEFAULT_SELECTED_ROWS))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument(
        "--carrier-key-mode",
        choices=["world_voxel", "surface_view_voxel", "selective_surface_view_voxel", "surface_normal_voxel"],
        default="world_voxel",
    )
    parser.add_argument("--view-ray-bin-size", type=float, default=0.50)
    parser.add_argument("--normal-bin-size", type=float, default=0.50)
    parser.add_argument("--normal-knn", type=int, default=8)
    parser.add_argument("--selective-surface-view-min-world-observations", type=int, default=4)
    parser.add_argument("--sketch-dim", type=int, default=2048)
    parser.add_argument("--support-lambda", type=float, default=0.20)
    parser.add_argument("--veto-attenuation", type=float, default=0.25)
    parser.add_argument("--frame-center-beta", type=float, default=0.0)
    parser.add_argument("--mask-interior-erode-pixels", type=float, default=0.0)
    parser.add_argument("--dedupe-frame-mask", action="store_true")
    parser.add_argument("--veto-broad-ratio", type=float, default=0.50)
    parser.add_argument("--anchor-min-frames", type=int, default=2)
    parser.add_argument("--anchor-min-object-ratio", type=float, default=0.50)
    parser.add_argument("--anchor-max-broad-ratio", type=float, default=0.30)
    parser.add_argument("--carrier-incidence-idf-alpha", type=float, default=0.0)
    parser.add_argument("--carrier-subkey-idf-alpha", type=float, default=0.0)
    parser.add_argument("--support-count-scale", type=float, default=4.0)
    parser.add_argument("--max-points-per-frame", type=int, default=20000)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--max-role-rows", type=int, default=50000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args)
    return 0 if summary.get("adapter_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
