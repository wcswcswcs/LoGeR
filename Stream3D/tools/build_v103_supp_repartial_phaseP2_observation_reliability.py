#!/usr/bin/env python3
"""Build v103 supplement R2 Phase P2 observation-level reliability.

P2 assigns q_geo/q_mask/q_sem/q_final to each foreground carrier observation in
the current c0001 scope. The output universe is intentionally restricted to
valid, in-image, foreground mask observations because only those observations
can participate in primitive-mask incidence in P3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_supp_r2_phaseP2_observation_reliability"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
DEFAULT_P1_ROOT = AUDIT_ROOT / "v103_supp_r2_phaseP1_semantic_features"
S1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers"

PHASE2_ROOTS = {
    "scene0011_00": AUDIT_ROOT
    / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
    "scene0050_00": AUDIT_ROOT
    / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8",
}

NEEDED_CACHE_KEYS = [
    "carrier_id",
    "src_frame",
    "src_uv",
    "xyz_ref",
    "uv_pred",
    "visibility_prob",
    "confidence_prob",
    "valid",
    "src_frame_global",
    "src_xy",
    "src_mask_id",
    "query_source_code",
]

BROAD_AREA_RATIO = 0.12
OBJECT_LIKE_AREA_MIN = 0.001
OBJECT_LIKE_AREA_MAX = 0.20
SELF_ERROR_SIGMA_NORM = 0.015
SEMANTIC_DELTA_LOCAL = 3

ROLE_TO_CODE = {
    "A_obs_anchor": 1,
    "S_obs_support": 2,
    "V_obs_veto": 3,
    "U_obs_uncertain": 4,
}


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
            out: dict[str, Any] = {}
            for key in fields:
                value = _jsonable(row.get(key, ""))
                out[key] = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            writer.writerow(out)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_short(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as f:
        while read < max_bytes:
            chunk = f.read(min(1024 * 1024, max_bytes - read))
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def _load_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int32)


def _ensure_mmap_cache(phase2_root: Path) -> tuple[Path, dict[str, Any]]:
    batch_path = phase2_root / "carrier_batch.npz"
    cache_dir = phase2_root / "carrier_batch_mmap_cache"
    manifest_path = cache_dir / "manifest.json"
    if not batch_path.exists():
        raise FileNotFoundError(batch_path)
    stat = batch_path.stat()
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        valid = (
            manifest.get("source_path") == _rel(batch_path)
            and int(manifest.get("source_size_bytes", -1)) == int(stat.st_size)
            and all((cache_dir / f"{key}.npy").exists() for key in NEEDED_CACHE_KEYS)
        )
        if valid:
            manifest["cache_reused"] = True
            return cache_dir, manifest
    t0 = time.time()
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_rows: list[dict[str, Any]] = []
    with np.load(batch_path, allow_pickle=False) as data:
        missing = [key for key in NEEDED_CACHE_KEYS if key not in data.files]
        if missing:
            raise KeyError(f"carrier_batch missing keys: {missing}")
        for key in NEEDED_CACHE_KEYS:
            arr = np.asarray(data[key])
            out = cache_dir / f"{key}.npy"
            np.save(out, arr)
            array_rows.append(
                {
                    "key": key,
                    "path": _rel(out),
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "size_bytes": out.stat().st_size,
                    "sha256_first64m": _sha256_short(out),
                }
            )
    manifest = {
        "schema_version": "stream4d_v103_supp_r2_phaseP2_mmap_cache_manifest_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "source_path": _rel(batch_path),
        "source_size_bytes": stat.st_size,
        "cache_dir": _rel(cache_dir),
        "cache_reused": False,
        "array_rows": array_rows,
    }
    _write_json(manifest_path, manifest)
    return cache_dir, manifest


def _load_cached(cache_dir: Path) -> dict[str, np.ndarray]:
    return {key: np.load(cache_dir / f"{key}.npy", mmap_mode="r") for key in NEEDED_CACHE_KEYS}


def _boundary_any_maps(masks: np.ndarray) -> np.ndarray:
    out = np.zeros(masks.shape, dtype=bool)
    kernel = np.ones((5, 5), dtype=np.uint8)
    for fi in range(masks.shape[0]):
        frame = masks[fi]
        for label in np.unique(frame).tolist():
            label = int(label)
            if label <= 0:
                continue
            binary = (frame == label).astype(np.uint8)
            out[fi] |= cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel).astype(bool)
    return out


def _same_frame_competing_boundary_maps(masks: np.ndarray, object_map: np.ndarray, broad_map: np.ndarray) -> np.ndarray:
    out = np.zeros(masks.shape, dtype=bool)
    for fi in range(masks.shape[0]):
        frame = masks[fi]
        safe = np.clip(frame, 0, object_map.shape[1] - 1)
        foreground = frame > 0
        object_pix = object_map[fi][safe]
        broad_pix = broad_map[fi][safe]

        def mark(a_slice: tuple[slice, slice], b_slice: tuple[slice, slice]) -> None:
            a = frame[a_slice]
            b = frame[b_slice]
            diff = foreground[a_slice] & foreground[b_slice] & (a != b)
            if not np.any(diff):
                return
            obj_pair = object_pix[a_slice] | object_pix[b_slice]
            broad_object_pair = (object_pix[a_slice] & broad_pix[b_slice]) | (object_pix[b_slice] & broad_pix[a_slice])
            conflict = diff & (obj_pair | broad_object_pair)
            out[fi][a_slice] |= conflict
            out[fi][b_slice] |= conflict

        mark((slice(1, None), slice(None)), (slice(None, -1), slice(None)))
        mark((slice(None), slice(1, None)), (slice(None), slice(None, -1)))
    return out


def _load_scene_masks(scene_id: str, phase2_root: Path, max_frames: int = 0) -> tuple[dict[str, Any], list[int], np.ndarray]:
    summary = _read_json(phase2_root / "summary.json")
    frame_ids = [int(v) for v in summary["frame_ids"]]
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    mask_root = _project(summary["mask_root"])
    masks = np.stack([_load_mask(mask_root / f"{frame_id}.png") for frame_id in frame_ids], axis=0)
    return summary, frame_ids, masks


def _load_p1_features(p1_root: Path) -> tuple[pd.DataFrame, np.ndarray, dict[tuple[str, str], float]]:
    rows = pd.read_parquet(p1_root / "semantic_feature_rows.parquet")
    pack = np.load(p1_root / "semantic_features_compact_fp16.npz", allow_pickle=False)
    features = np.asarray(pack["compact_features"], dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)
    baseline_rows = pd.read_csv(p1_root / "semantic_baseline_rows.csv")
    mu: dict[tuple[str, str], float] = {}
    for row in baseline_rows.to_dict("records"):
        source = str(row["semantic_source_id"])
        scene = str(row["scene_id"])
        value = row.get("random_pair_mean", "")
        try:
            mu[(source, scene)] = float(value)
        except Exception:
            continue
    return rows, features.astype(np.float32), mu


def _mask_meta_maps(
    scene_id: str,
    frame_ids: list[int],
    masks: np.ndarray,
    p1_rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    max_label = int(np.max(masks)) if masks.size else 0
    t = len(frame_ids)
    broad_map = np.zeros((t, max_label + 1), dtype=bool)
    object_map = np.zeros((t, max_label + 1), dtype=bool)
    area_ratio_map = np.zeros((t, max_label + 1), dtype=np.float32)
    feature_maps = {
        "E_pool_radio": np.full((t, max_label + 1), -1, dtype=np.int32),
        "E_pool_dino": np.full((t, max_label + 1), -1, dtype=np.int32),
    }
    frame_to_fi = {int(frame_id): fi for fi, frame_id in enumerate(frame_ids)}
    p1_scene = p1_rows[p1_rows["scene_id"] == scene_id]
    semantic_broad: dict[tuple[int, int], bool] = {}
    for row in p1_scene.to_dict("records"):
        key = (int(row["frame_id"]), int(row["mask_id"]))
        semantic_broad[key] = semantic_broad.get(key, False) or bool(row.get("broad_background_risk", False))
        source = str(row["semantic_source_id"])
        if source in feature_maps and int(row["frame_id"]) in frame_to_fi and int(row["mask_id"]) <= max_label:
            feature_maps[source][frame_to_fi[int(row["frame_id"])], int(row["mask_id"])] = int(row["compact_feature_row_index"])
    h, w = masks.shape[1:]
    denom = float(max(h * w, 1))
    object_like_count = 0
    broad_count = 0
    for fi, frame_id in enumerate(frame_ids):
        labels, counts = np.unique(masks[fi], return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            label = int(label)
            if label <= 0:
                continue
            area = float(count) / denom
            area_ratio_map[fi, label] = area
            broad = bool(semantic_broad.get((int(frame_id), label), False)) or area >= BROAD_AREA_RATIO
            obj = (OBJECT_LIKE_AREA_MIN <= area <= OBJECT_LIKE_AREA_MAX) and not broad
            broad_map[fi, label] = broad
            object_map[fi, label] = obj
            broad_count += int(broad)
            object_like_count += int(obj)
    meta = {
        "max_label": max_label,
        "height": h,
        "width": w,
        "object_like_mask_count": object_like_count,
        "broad_mask_count": broad_count,
    }
    return broad_map, object_map, area_ratio_map, feature_maps, meta


def _project_labels(batch: dict[str, np.ndarray], masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uv_pred = batch["uv_pred"]
    xyz_ref = batch["xyz_ref"]
    valid = batch["valid"]
    frame_count, carrier_count = valid.shape
    h, w = masks.shape[1:]
    labels = np.full((frame_count, carrier_count), -1, dtype=np.int16)
    xs_all = np.zeros((frame_count, carrier_count), dtype=np.int16)
    ys_all = np.zeros((frame_count, carrier_count), dtype=np.int16)
    in_image = np.zeros((frame_count, carrier_count), dtype=bool)
    for fi in range(frame_count):
        uv = np.asarray(uv_pred[fi], dtype=np.float32)
        xyz = np.asarray(xyz_ref[fi], dtype=np.float32)
        finite = np.isfinite(uv).all(axis=1) & np.isfinite(xyz).all(axis=1)
        ok = valid[fi] & finite & (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
        xs = np.rint(np.clip(uv[:, 0], 0.0, 1.0) * float(max(w - 1, 1))).astype(np.int32)
        ys = np.rint(np.clip(uv[:, 1], 0.0, 1.0) * float(max(h - 1, 1))).astype(np.int32)
        lab = np.full((carrier_count,), -1, dtype=np.int16)
        lab[ok] = masks[fi, ys[ok], xs[ok]].astype(np.int16)
        labels[fi] = lab
        xs_all[fi] = xs.astype(np.int16)
        ys_all[fi] = ys.astype(np.int16)
        in_image[fi] = ok
    return labels, xs_all, ys_all, in_image


def _feature_idx_by_source(labels: np.ndarray, feature_map: np.ndarray) -> np.ndarray:
    out = np.full(labels.shape, -1, dtype=np.int32)
    max_label = feature_map.shape[1] - 1
    for fi in range(labels.shape[0]):
        lab = np.clip(labels[fi], 0, max_label)
        vals = feature_map[fi, lab]
        out[fi] = np.where(labels[fi] > 0, vals, -1).astype(np.int32)
    return out


def _semantic_q(
    feature_idx: np.ndarray,
    features: np.ndarray,
    mu_sem: float,
    delta: int = SEMANTIC_DELTA_LOCAL,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count, carrier_count = feature_idx.shape
    slots = np.full((delta * 2, frame_count, carrier_count), np.nan, dtype=np.float16)
    slot = 0
    denom = max(1.0 - float(mu_sem), 1e-6)
    for d in range(1, delta + 1):
        for direction in [-1, 1]:
            for fi in range(frame_count):
                fj = fi + direction * d
                if fj < 0 or fj >= frame_count:
                    continue
                a = feature_idx[fi]
                b = feature_idx[fj]
                ok = (a >= 0) & (b >= 0)
                if not np.any(ok):
                    continue
                sim = np.einsum("ij,ij->i", features[a[ok]], features[b[ok]]).astype(np.float32)
                q = np.clip((sim - float(mu_sem)) / denom, 0.0, 1.0)
                row = slots[slot, fi]
                row[ok] = q.astype(np.float16)
            slot += 1
    with np.errstate(all="ignore"):
        q = np.nanmedian(slots, axis=0).astype(np.float32)
    has_pair = np.isfinite(q)
    q[~has_pair] = 0.5
    return q.astype(np.float32), has_pair


def _load_previous_anchor_sets() -> dict[str, set[int]]:
    path = S1_ROOT / "carrier_role_rows.parquet"
    if not path.exists():
        return {scene_id: set() for scene_id in PHASE2_ROOTS}
    rows = pd.read_parquet(path, columns=["scene_id", "carrier_id", "is_A_anchor"])
    out: dict[str, set[int]] = {}
    for scene_id in PHASE2_ROOTS:
        vals = rows[(rows["scene_id"] == scene_id) & (rows["is_A_anchor"])]["carrier_id"].astype(np.int64).tolist()
        out[scene_id] = set(int(v) for v in vals)
    return out


def _previous_gate_values() -> dict[str, dict[str, float]]:
    path = S1_ROOT / "gate_rows.csv"
    out: dict[str, dict[str, float]] = {scene_id: {} for scene_id in PHASE2_ROOTS}
    if not path.exists():
        return out
    rows = pd.read_csv(path)
    name_map = {
        "A_anchor_broad_mask_participation_rate_le_0p15": "previous_broad_anchor_rate",
        "A_anchor_short_range_semantic_contradiction_rate_le_0p20": "previous_semantic_contradiction_anchor_rate",
        "A_anchor_competing_mask_conflict_rate_le_0p15": "previous_competing_anchor_rate",
    }
    for row in rows.to_dict("records"):
        metric = name_map.get(str(row.get("gate_name", "")))
        if not metric:
            continue
        try:
            out[str(row["scene_id"])][metric] = float(row["observed"])
        except Exception:
            pass
    return out


def _role_arrays(
    q_geo: np.ndarray,
    q_mask: np.ndarray,
    q_sem_final: np.ndarray,
    q_final: np.ndarray,
    broad: np.ndarray,
    boundary: np.ndarray,
    competing: np.ndarray,
    disagreement: np.ndarray,
    visibility: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    veto = broad | boundary | competing | (disagreement > 0.45) | (q_geo < 0.05)
    anchor = (~veto) & (visibility >= 0.40) & (q_geo >= 0.35) & (q_mask >= 0.80) & (q_sem_final >= 0.60) & (q_final >= 0.20)
    support = (~veto) & (~anchor) & (q_geo >= 0.15) & (q_mask >= 0.25) & (q_sem_final >= 0.45) & (q_final >= 0.05)
    role_code = np.full(q_final.shape, ROLE_TO_CODE["U_obs_uncertain"], dtype=np.int8)
    role_code[veto] = ROLE_TO_CODE["V_obs_veto"]
    role_code[support] = ROLE_TO_CODE["S_obs_support"]
    role_code[anchor] = ROLE_TO_CODE["A_obs_anchor"]
    role_name = np.empty(q_final.shape, dtype=object)
    role_name[role_code == ROLE_TO_CODE["A_obs_anchor"]] = "A_obs_anchor"
    role_name[role_code == ROLE_TO_CODE["S_obs_support"]] = "S_obs_support"
    role_name[role_code == ROLE_TO_CODE["V_obs_veto"]] = "V_obs_veto"
    role_name[role_code == ROLE_TO_CODE["U_obs_uncertain"]] = "U_obs_uncertain"
    return role_code, role_name


def _arrow_table(frame_rows: dict[str, Any]) -> pa.Table:
    return pa.table(frame_rows)


def _scene_process(
    scene_id: str,
    output_writer: pq.ParquetWriter | None,
    p1_rows: pd.DataFrame,
    p1_features: np.ndarray,
    semantic_mu: dict[tuple[str, str], float],
    previous_anchor_ids: set[int],
    previous_rates: dict[str, float],
    max_frames: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    t0 = time.time()
    phase2_root = PHASE2_ROOTS[scene_id]
    cache_dir, cache_manifest = _ensure_mmap_cache(phase2_root)
    batch = _load_cached(cache_dir)
    summary, frame_ids, masks = _load_scene_masks(scene_id, phase2_root, max_frames=max_frames)
    if max_frames > 0:
        batch = {
            key: value[: len(frame_ids)] if getattr(value, "ndim", 0) >= 2 and value.shape[0] == 32 else value
            for key, value in batch.items()
        }
    labels, xs, ys, in_image = _project_labels(batch, masks)
    broad_map, object_map, area_ratio_map, feature_maps, mask_meta = _mask_meta_maps(scene_id, frame_ids, masks, p1_rows)
    boundary_maps = _boundary_any_maps(masks)
    competing_maps = _same_frame_competing_boundary_maps(masks, object_map, broad_map)

    feat_idx_radio = _feature_idx_by_source(labels, feature_maps["E_pool_radio"])
    feat_idx_dino = _feature_idx_by_source(labels, feature_maps["E_pool_dino"])
    mu_radio = semantic_mu.get(("E_pool_radio", scene_id), semantic_mu.get(("E_pool_radio", "all"), 0.0))
    mu_dino = semantic_mu.get(("E_pool_dino", scene_id), semantic_mu.get(("E_pool_dino", "all"), 0.0))
    q_radio, radio_has_pair = _semantic_q(feat_idx_radio, p1_features, mu_radio)
    q_dino, dino_has_pair = _semantic_q(feat_idx_dino, p1_features, mu_dino)
    both_sem = radio_has_pair & dino_has_pair
    disagreement = np.where(both_sem, np.abs(q_radio - q_dino), 0.0).astype(np.float32)
    q_sem_final = (np.maximum(q_radio, q_dino) * (1.0 - disagreement)).astype(np.float32)

    carrier_id = np.asarray(batch["carrier_id"], dtype=np.int64)
    src_frame = np.asarray(batch["src_frame"], dtype=np.int16)
    src_uv = np.asarray(batch["src_uv"], dtype=np.float32)
    src_frame_clip = np.clip(src_frame.astype(np.int64), 0, len(frame_ids) - 1)
    carrier_idx = np.arange(carrier_id.shape[0], dtype=np.int64)
    self_uv = np.asarray(batch["uv_pred"][src_frame_clip, carrier_idx], dtype=np.float32)
    h, w = masks.shape[1:]
    scale = np.asarray([max(w - 1, 1), max(h - 1, 1)], dtype=np.float32)
    diag = math.sqrt(float(w * w + h * h))
    self_error_px = np.linalg.norm((self_uv - src_uv) * scale, axis=1).astype(np.float32)
    self_error_px[~np.isfinite(self_error_px)] = diag
    self_error_norm = np.clip(self_error_px / max(diag, 1.0), 0.0, 1.0).astype(np.float32)
    self_jitter_q = np.exp(-self_error_norm / SELF_ERROR_SIGMA_NORM).astype(np.float32)

    previous_anchor_mask = np.asarray([int(v) in previous_anchor_ids for v in carrier_id.tolist()], dtype=bool)
    scene_counters: dict[str, Any] = {
        "scene_id": scene_id,
        "total_carrier_frame_observation_count": int(labels.size),
        "in_image_observation_count": int(in_image.sum()),
        "foreground_observation_count": 0,
        "previous_A_anchor_carrier_count": int(len(previous_anchor_ids)),
        "previous_A_anchor_foreground_observation_count": 0,
        "A_obs_anchor_count": 0,
        "S_obs_support_count": 0,
        "V_obs_veto_count": 0,
        "U_obs_uncertain_count": 0,
        "anchor_broad_count": 0,
        "anchor_boundary_count": 0,
        "anchor_competing_count": 0,
        "anchor_semantic_contradiction_count": 0,
        "anchor_mask_support_counts": [],
        "q_final_sum": 0.0,
        "q_final_count": 0,
        "semantic_both_pair_count": int(both_sem.sum()),
        "semantic_disagreement_sum": float(disagreement[both_sem].sum()) if np.any(both_sem) else 0.0,
        "cache_reused": bool(cache_manifest.get("cache_reused")),
    }

    for fi, frame_id in enumerate(frame_ids):
        lab = labels[fi]
        foreground = in_image[fi] & (lab > 0)
        if not np.any(foreground):
            continue
        lab_clip = np.clip(lab, 0, area_ratio_map.shape[1] - 1)
        visibility = np.asarray(batch["visibility_prob"][fi], dtype=np.float32)
        confidence = np.asarray(batch["confidence_prob"][fi], dtype=np.float32)
        q_geo_all = (confidence * visibility * in_image[fi].astype(np.float32) * self_jitter_q).astype(np.float32)
        broad_all = broad_map[fi, lab_clip]
        object_all = object_map[fi, lab_clip]
        area_all = area_ratio_map[fi, lab_clip]
        boundary_all = boundary_maps[fi, ys[fi], xs[fi]]
        competing_all = competing_maps[fi, ys[fi], xs[fi]]
        q_object = np.where(object_all, 1.0, np.where(broad_all, 0.0, 0.25)).astype(np.float32)
        q_mask_all = (
            (1.0 - broad_all.astype(np.float32))
            * (1.0 - boundary_all.astype(np.float32))
            * (1.0 - competing_all.astype(np.float32))
            * q_object
        ).astype(np.float32)
        q_final_all = (q_geo_all * q_mask_all * q_sem_final[fi]).astype(np.float32)
        role_code_all, role_name_all = _role_arrays(
            q_geo_all,
            q_mask_all,
            q_sem_final[fi],
            q_final_all,
            broad_all,
            boundary_all,
            competing_all,
            disagreement[fi],
            visibility,
        )

        idx = np.flatnonzero(foreground)
        role_code = role_code_all[idx]
        anchor = role_code == ROLE_TO_CODE["A_obs_anchor"]
        support = role_code == ROLE_TO_CODE["S_obs_support"]
        veto = role_code == ROLE_TO_CODE["V_obs_veto"]
        uncertain = role_code == ROLE_TO_CODE["U_obs_uncertain"]
        scene_counters["foreground_observation_count"] += int(idx.size)
        scene_counters["previous_A_anchor_foreground_observation_count"] += int((foreground & previous_anchor_mask).sum())
        scene_counters["A_obs_anchor_count"] += int(anchor.sum())
        scene_counters["S_obs_support_count"] += int(support.sum())
        scene_counters["V_obs_veto_count"] += int(veto.sum())
        scene_counters["U_obs_uncertain_count"] += int(uncertain.sum())
        scene_counters["anchor_broad_count"] += int(broad_all[idx][anchor].sum())
        scene_counters["anchor_boundary_count"] += int(boundary_all[idx][anchor].sum())
        scene_counters["anchor_competing_count"] += int(competing_all[idx][anchor].sum())
        scene_counters["anchor_semantic_contradiction_count"] += int((disagreement[fi, idx][anchor] > 0.45).sum())
        scene_counters["q_final_sum"] += float(q_final_all[idx].sum())
        scene_counters["q_final_count"] += int(idx.size)
        if np.any(anchor):
            key = lab[idx][anchor].astype(np.int64) + int(frame_id) * 100000
            counts = np.unique(key, return_counts=True)[1]
            scene_counters["anchor_mask_support_counts"].extend([int(v) for v in counts.tolist()])

        if output_writer is not None:
            table = _arrow_table(
                {
                    "scene_id": np.full(idx.size, scene_id),
                    "chunk_id": np.full(idx.size, "c0001"),
                    "carrier_id": carrier_id[idx],
                    "frame_id": np.full(idx.size, int(frame_id), dtype=np.int32),
                    "x": xs[fi, idx].astype(np.int16),
                    "y": ys[fi, idx].astype(np.int16),
                    "mask_id": lab[idx].astype(np.int16),
                    "mask_area_ratio": area_all[idx].astype(np.float32),
                    "q_geo": q_geo_all[idx].astype(np.float32),
                    "q_mask": q_mask_all[idx].astype(np.float32),
                    "q_sem_raw_radio": np.full(idx.size, np.nan, dtype=np.float32),
                    "q_sem_pool_radio": q_radio[fi, idx].astype(np.float32),
                    "q_sem_pool_dino": q_dino[fi, idx].astype(np.float32),
                    "q_sem_clip_crop": np.full(idx.size, np.nan, dtype=np.float32),
                    "q_sem_consensus": np.full(idx.size, np.nan, dtype=np.float32),
                    "q_sem_final": q_sem_final[fi, idx].astype(np.float32),
                    "q_final": q_final_all[idx].astype(np.float32),
                    "visibility_prob": visibility[idx].astype(np.float32),
                    "confidence_prob": confidence[idx].astype(np.float32),
                    "broad_risk": broad_all[idx].astype(bool),
                    "boundary_risk": boundary_all[idx].astype(bool),
                    "competing_mask_risk": competing_all[idx].astype(bool),
                    "semantic_source_disagreement": disagreement[fi, idx].astype(np.float32),
                    "observation_role": role_name_all[idx],
                    "observation_role_code": role_code.astype(np.int8),
                    "query_source_code": np.asarray(batch["query_source_code"], dtype=np.int16)[idx],
                    "src_frame_global": np.asarray(batch["src_frame_global"], dtype=np.int32)[idx],
                    "prev_whole_carrier_A_anchor": previous_anchor_mask[idx],
                    "uses_gt_for_prediction": np.zeros(idx.size, dtype=bool),
                    "uses_future": np.zeros(idx.size, dtype=bool),
                }
            )
            output_writer.write_table(table)

    anchor_count = max(int(scene_counters["A_obs_anchor_count"]), 1)
    scene_counters["broad_anchor_rate"] = float(scene_counters["anchor_broad_count"] / anchor_count)
    scene_counters["boundary_anchor_rate"] = float(scene_counters["anchor_boundary_count"] / anchor_count)
    scene_counters["competing_anchor_rate"] = float(scene_counters["anchor_competing_count"] / anchor_count)
    scene_counters["semantic_contradiction_anchor_rate"] = float(scene_counters["anchor_semantic_contradiction_count"] / anchor_count)
    supports = np.asarray(scene_counters.pop("anchor_mask_support_counts"), dtype=np.int32)
    scene_counters["object_like_anchor_mask_support_p10"] = float(np.percentile(supports, 10)) if supports.size else 0.0
    scene_counters["q_final_mean"] = float(scene_counters["q_final_sum"] / max(int(scene_counters["q_final_count"]), 1))
    scene_counters["semantic_source_disagreement_mean"] = float(
        scene_counters["semantic_disagreement_sum"] / max(int(scene_counters["semantic_both_pair_count"]), 1)
    )
    for key, value in previous_rates.items():
        scene_counters[key] = value
    scene_counters["runtime_sec"] = time.time() - t0
    agreement = {
        "schema_version": "stream4d_v103_supp_r2_phaseP2_semantic_source_agreement_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene_id,
        "semantic_source_pair": "E_pool_radio__E_pool_dino",
        "both_source_short_range_pair_observation_count": int(scene_counters["semantic_both_pair_count"]),
        "semantic_source_disagreement_mean": scene_counters["semantic_source_disagreement_mean"],
        "semantic_source_disagreement_gt_0p45_rate": float((disagreement[both_sem] > 0.45).mean()) if np.any(both_sem) else 0.0,
        "mu_sem_pool_radio": mu_radio,
        "mu_sem_pool_dino": mu_dino,
        "uses_gt": False,
    }
    return scene_counters, agreement


def _gate_row(gate_id: str, passed: bool, observed: Any, required: Any, repair_direction: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r2_phaseP2_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_id": gate_id,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "repair_direction": repair_direction,
    }


def _build_gates(summary_rows: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    if max_frames > 0:
        return [
            _gate_row(
                "diagnostic_subset_not_full_phaseP2",
                False,
                {"max_frames_per_scene": max_frames},
                "max_frames_per_scene=0 for full P2 gate",
                "Rerun without --max-frames-per-scene before continuing.",
            )
        ]
    gates: list[dict[str, Any]] = []
    for row in summary_rows:
        scene = row["scene_id"]
        gates.append(
            _gate_row(
                f"{scene}_anchor_observation_count_gt_previous_whole_carrier_anchor_observations",
                int(row["A_obs_anchor_count"]) > int(row["previous_A_anchor_foreground_observation_count"]),
                {"current": row["A_obs_anchor_count"], "previous": row["previous_A_anchor_foreground_observation_count"]},
                "current > previous foreground observations from S1 A_anchor carriers",
                "Relax observation thresholds or inspect q_geo/q_mask/q_sem source disagreement.",
            )
        )
        gates.append(
            _gate_row(
                f"{scene}_object_like_anchor_mask_support_p10_gt_0",
                float(row["object_like_anchor_mask_support_p10"]) > 0.0,
                row["object_like_anchor_mask_support_p10"],
                ">0",
                "Repair mask assignment/object-like gate before P3.",
            )
        )
        gates.append(
            _gate_row(
                f"{scene}_broad_anchor_rate_le_previous_plus_0p02",
                float(row["broad_anchor_rate"]) <= float(row.get("previous_broad_anchor_rate", 1.0)) + 0.02,
                {"current": row["broad_anchor_rate"], "previous": row.get("previous_broad_anchor_rate", "")},
                "current <= previous + 0.02",
                "Tighten broad/object-like q_mask before P3.",
            )
        )
        gates.append(
            _gate_row(
                f"{scene}_semantic_contradiction_anchor_rate_le_previous_plus_0p02",
                float(row["semantic_contradiction_anchor_rate"]) <= float(row.get("previous_semantic_contradiction_anchor_rate", 1.0)) + 0.02,
                {"current": row["semantic_contradiction_anchor_rate"], "previous": row.get("previous_semantic_contradiction_anchor_rate", "")},
                "current <= previous + 0.02",
                "Tighten semantic disagreement veto before P3.",
            )
        )
        gates.append(
            _gate_row(
                f"{scene}_competing_anchor_rate_le_previous_plus_0p02",
                float(row["competing_anchor_rate"]) <= float(row.get("previous_competing_anchor_rate", 1.0)) + 0.02,
                {"current": row["competing_anchor_rate"], "previous": row.get("previous_competing_anchor_rate", "")},
                "current <= previous + 0.02",
                "Tighten competing/boundary veto before P3.",
            )
        )
    return gates


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    p1_root = _project(args.p1_root)
    output_root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    p1_rows, p1_features, semantic_mu = _load_p1_features(p1_root)
    previous_anchor_sets = _load_previous_anchor_sets()
    previous_rates = _previous_gate_values()

    observation_parquet = output_root / "carrier_observation_reliability_rows.parquet"
    summary_csv = output_root / "carrier_observation_summary_rows.csv"
    agreement_csv = output_root / "semantic_source_agreement_rows.csv"
    gate_csv = output_root / "gate_rows.csv"
    failure_csv = output_root / "failure_rows.csv"
    summary_path = output_root / "summary.json"

    class _LazyParquetWriter:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.writer: pq.ParquetWriter | None = None

        def write_table(self, table: pa.Table) -> None:
            if self.writer is None:
                self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd", use_dictionary=True)
            self.writer.write_table(table)

        def close(self) -> None:
            if self.writer is not None:
                self.writer.close()
                self.writer = None

    if observation_parquet.exists():
        observation_parquet.unlink()
    writer = _LazyParquetWriter(observation_parquet)
    scene_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    try:
        for scene_id in PHASE2_ROOTS:
            row, agreement = _scene_process(
                scene_id=scene_id,
                output_writer=writer,
                p1_rows=p1_rows,
                p1_features=p1_features,
                semantic_mu=semantic_mu,
                previous_anchor_ids=previous_anchor_sets.get(scene_id, set()),
                previous_rates=previous_rates.get(scene_id, {}),
                max_frames=int(args.max_frames_per_scene),
            )
            scene_rows.append(row)
            agreement_rows.append(agreement)
    finally:
        writer.close()

    gates = _build_gates(scene_rows, int(args.max_frames_per_scene))
    failure_rows = [row for row in gates if not row["pass"]]
    phaseP2_pass = len(failure_rows) == 0

    _write_csv(summary_csv, scene_rows)
    _write_csv(agreement_csv, agreement_rows)
    _write_csv(gate_csv, gates)
    _write_csv(failure_csv, failure_rows)

    total_anchor = int(sum(int(row["A_obs_anchor_count"]) for row in scene_rows))
    total_foreground = int(sum(int(row["foreground_observation_count"]) for row in scene_rows))
    summary = {
        "schema_version": "stream4d_v103_supp_r2_phaseP2_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - start,
        "decision": "PASS_ENTER_PHASEP3_CARRIER_SEGMENTATION" if phaseP2_pass else "NO_GO_REPAIR_PHASEP2_OBSERVATION_RELIABILITY",
        "phaseP2_pass": bool(phaseP2_pass),
        "failure_count": len(failure_rows),
        "observation_universe_policy": "valid_in_image_foreground_mask_observations_only",
        "total_foreground_observation_count": total_foreground,
        "total_A_obs_anchor_count": total_anchor,
        "max_frames_per_scene": int(args.max_frames_per_scene),
        "p1_root": _rel(p1_root),
        "semantic_sources": ["E_pool_radio", "E_pool_dino"],
        "raw_radio_available": False,
        "clip_crop_available": False,
        "consensus_available": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "outputs": {
            "summary": _rel(summary_path),
            "carrier_observation_reliability_rows": _rel(observation_parquet),
            "carrier_observation_summary_rows": _rel(summary_csv),
            "semantic_source_agreement_rows": _rel(agreement_csv),
            "gate_rows": _rel(gate_csv),
            "failure_rows": _rel(failure_csv),
        },
        "truthfulness_note": (
            "P2 computes observation-level reliability only. It does not segment carriers, build mask graph edges, "
            "compute AP, or use GT/future frames. Background/out-of-image observations are counted in summary but not written to the wide row table."
        ),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--p1-root", default=str(DEFAULT_P1_ROOT))
    parser.add_argument("--max-frames-per-scene", type=int, default=0)
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True, ensure_ascii=False))
    raise SystemExit(0 if summary["phaseP2_pass"] else 2)


if __name__ == "__main__":
    main()
