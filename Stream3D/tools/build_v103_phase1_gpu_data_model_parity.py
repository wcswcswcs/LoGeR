#!/usr/bin/env python3
"""Build v103 Phase1 GPU-first data-model parity artifacts.

This phase materializes a medium, auditable subset for scene0011_00 chunk_0000:
32 stride-5 frames, cached D4RT carriers, CropFormer masks, and RADIO features.
It verifies CPU/GPU parity for incidence lookup, packed mask support counts,
CountSketch carrier features, mask-level pooling, semantic similarity, history
top-k assignment, and candidate top-k construction.

No object predictions or AP metrics are produced here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v103_phase1_gpu_data_model_parity"

PHASE0_SUMMARY = AUDIT_ROOT / "v103_phase0_contract/summary.json"
FEATURE_ROWS = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv"
FEATURE_STORE = AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_features.npz"
D4RT_CHUNK = (
    STREAM3D_ROOT
    / "outputs/cache/v66_d4rt_stride_geometry/scene0011_00/"
    / "chunk32_overlap03_grid08_minconf0p000_minvis0p000/stride_5/chunks/chunk_0000.npz"
)
MASK_ROOT = (
    STREAM3D_ROOT
    / "outputs/cache/v66_cropformer_chunk_masks/scene0011_00/stride_5/"
    / "cropformer_conf_0p500/mask2former_hornet_3x/final_processed/"
    / "scene0011_00/output_Cropformer/mask"
)

PHASE_ID = "v103_phase1_gpu_data_model_parity"
SCENE_ID = "scene0011_00"
CHUNK_ID = "chunk000"
CHUNK_INDEX = 0
SKETCH_DIM = 4096
SKETCH_SEED = 10317
HISTORY_TOKEN_COUNT = 16
TOPK = 8
PATCH_NOTE = "Phase1 parity-only synthetic history tokens are not method predictions."


INCIDENCE_KERNEL = r"""
extern "C" __global__
void incidence_lookup_kernel(
    const int* __restrict__ labels,
    const float* __restrict__ uv,
    const unsigned char* __restrict__ valid,
    const float* __restrict__ visibility,
    const float* __restrict__ confidence,
    const float* __restrict__ reliability,
    const long long* __restrict__ mask_global_by_label,
    const long long* __restrict__ frame_ids,
    long long* __restrict__ out_carrier,
    long long* __restrict__ out_mask_global,
    long long* __restrict__ out_frame,
    long long* __restrict__ out_label,
    float* __restrict__ out_value,
    unsigned int* __restrict__ out_count,
    const long long frame_count,
    const long long carrier_count,
    const long long height,
    const long long width,
    const long long map_width
) {
    const long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    const long long total = frame_count * carrier_count;
    if (idx >= total) {
        return;
    }
    if (!valid[idx]) {
        return;
    }
    const long long fi = idx / carrier_count;
    const long long ci = idx - fi * carrier_count;
    const float u = uv[idx * 2];
    const float v = uv[idx * 2 + 1];
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        return;
    }
    const long long x = llrintf(u * (float)(width - 1));
    const long long y = llrintf(v * (float)(height - 1));
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return;
    }
    const int label = labels[(fi * height + y) * width + x];
    if (label < 0 || label >= map_width) {
        return;
    }
    const long long mask_global = mask_global_by_label[fi * map_width + (long long)label];
    if (mask_global < 0) {
        return;
    }
    const unsigned int out_idx = atomicAdd(out_count, 1U);
    out_carrier[out_idx] = ci;
    out_mask_global[out_idx] = mask_global;
    out_frame[out_idx] = frame_ids[fi];
    out_label[out_idx] = (long long)label;
    out_value[out_idx] = reliability[ci] * visibility[idx] * confidence[idx];
}
"""


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
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
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _read_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read label image: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int32)


def _normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, eps)


def _pack_uint64(mask_bool: np.ndarray) -> np.ndarray:
    flat = np.asarray(mask_bool, dtype=np.uint8)
    n, pixels = flat.shape
    width64 = int(math.ceil(pixels / 64.0))
    pad = width64 * 64 - pixels
    if pad:
        flat = np.pad(flat, ((0, 0), (0, pad)), mode="constant")
    bits = flat.reshape(n, width64, 64).astype(np.uint64, copy=False)
    weights = (np.uint64(1) << np.arange(64, dtype=np.uint64))[None, None, :]
    return np.sum(bits * weights, axis=2, dtype=np.uint64)


def _load_feature_subset(frame_ids: list[int]) -> tuple[pd.DataFrame, np.ndarray, list[dict[str, Any]]]:
    rows = pd.read_csv(FEATURE_ROWS)
    rows = rows[
        (rows["scene_id"].astype(str) == SCENE_ID)
        & (rows["chunk_id"].astype(str) == f"{SCENE_ID}:{CHUNK_ID}")
        & (rows["frame_id"].astype(int).isin(frame_ids))
        & (rows["feature_available"].map(lambda v: str(v).lower() == "true"))
    ].copy()
    rows = rows.sort_values(["frame_id", "mask_id"]).reset_index(drop=True)
    if rows.empty:
        raise RuntimeError("no v91 RADIO mask feature rows for Phase1 subset")

    store = np.load(FEATURE_STORE)
    features = store["features"].astype(np.float32)
    ids = [str(x) for x in store["mask_observation_id"]]
    feature_by_id = {mask_id: features[idx] for idx, mask_id in enumerate(ids)}
    feature_arr = np.stack([feature_by_id[str(row["mask_observation_id"])] for row in rows.to_dict("records")])
    feature_arr = _normalize_rows(feature_arr.astype(np.float32, copy=False))

    mask_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows.to_dict("records")):
        frame_id = int(row["frame_id"])
        mask_id = int(row["mask_id"])
        area = int(_num(row.get("used_pixel_count")))
        area_ratio = float(area / max(1.0, 968.0 * 1296.0))
        broad_risk = bool(area_ratio >= 0.20 or str(row.get("broad_background_risk", "")).lower() == "true")
        object_like_score = 1.0 if 0.005 <= area_ratio < 0.20 else (0.25 if area_ratio < 0.005 else 0.0)
        mask_rows.append(
            {
                "schema_version": "stream4d_v103_phase1_mask_table_row_v1",
                "phase_id": PHASE_ID,
                "mask_global_id": idx,
                "scene_id": SCENE_ID,
                "chunk_id": CHUNK_ID,
                "chunk_index": CHUNK_INDEX,
                "frame_id": frame_id,
                "mask_id": mask_id,
                "mask_observation_id": str(row["mask_observation_id"]),
                "area": area,
                "bbox": "",
                "broad_risk": broad_risk,
                "object_like_score": object_like_score,
                "semantic_feature_index": idx,
                "semantic_backend": row.get("semantic_backend", "radio_radseg"),
                "semantic_entropy": row.get("semantic_entropy", ""),
                "semantic_prototype_margin": row.get("semantic_prototype_margin", ""),
                "mask_path": _rel(MASK_ROOT / f"{frame_id}.png"),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows, feature_arr, mask_rows


def _load_labels(frame_ids: list[int]) -> tuple[np.ndarray, dict[int, int]]:
    labels: list[np.ndarray] = []
    frame_to_local: dict[int, int] = {}
    shape_hw: tuple[int, int] | None = None
    for local_idx, frame_id in enumerate(frame_ids):
        path = MASK_ROOT / f"{frame_id}.png"
        label = _read_label(path)
        if shape_hw is None:
            shape_hw = tuple(label.shape[:2])
        elif tuple(label.shape[:2]) != shape_hw:
            raise RuntimeError(f"label shape mismatch for {path}: {label.shape} vs {shape_hw}")
        labels.append(label.astype(np.int32, copy=False))
        frame_to_local[frame_id] = local_idx
    return np.stack(labels, axis=0), frame_to_local


def _build_mask_bitsets(mask_rows: list[dict[str, Any]], labels: np.ndarray, frame_to_local: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    masks: list[np.ndarray] = []
    for row in mask_rows:
        local_frame = frame_to_local[int(row["frame_id"])]
        masks.append((labels[local_frame] == int(row["mask_id"])).reshape(-1))
    mask_bool = np.stack(masks, axis=0).astype(bool, copy=False)
    return mask_bool, _pack_uint64(mask_bool)


def _carrier_table(d4rt: Any) -> tuple[list[dict[str, Any]], np.ndarray]:
    valid = np.asarray(d4rt["valid"], dtype=bool)
    uv = np.asarray(d4rt["uv"], dtype=np.float32)
    visibility = np.asarray(d4rt["visibility"], dtype=np.float32)
    confidence = np.asarray(d4rt["confidence"], dtype=np.float32)
    frame_count, carrier_count = valid.shape
    in01 = valid & (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    rows: list[dict[str, Any]] = []
    reliability = np.zeros((carrier_count,), dtype=np.float32)
    for ci in range(carrier_count):
        valid_i = valid[:, ci]
        valid_count = int(np.count_nonzero(valid_i))
        visibility_rate = float(np.count_nonzero(valid_i) / max(1, frame_count))
        in_image_rate = float(np.count_nonzero(in01[:, ci]) / max(1, frame_count))
        conf_mean = float(np.mean(confidence[valid_i, ci])) if valid_count else 0.0
        if valid_count >= 2:
            uv_valid = uv[valid_i, ci]
            jitter_norm = float(np.linalg.norm(np.std(uv_valid, axis=0)))
        else:
            jitter_norm = 1.0
        rel = conf_mean * visibility_rate * in_image_rate * math.exp(-jitter_norm / 0.20)
        reliability[ci] = np.float32(rel)
        rows.append(
            {
                "schema_version": "stream4d_v103_phase1_carrier_table_row_v1",
                "phase_id": PHASE_ID,
                "carrier_index": ci,
                "carrier_id": int(np.asarray(d4rt["carrier_id"])[ci]),
                "scene_id": SCENE_ID,
                "chunk_id": CHUNK_ID,
                "query_source": "v66_d4rt_stride_geometry_grid64_chunk0000",
                "valid_frame_count": valid_count,
                "visibility_rate": visibility_rate,
                "in_image_rate": in_image_rate,
                "confidence_mean": conf_mean,
                "normalized_jitter": jitter_norm,
                "reliability": float(rel),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows, reliability


def _cpu_incidence(
    labels: np.ndarray,
    frame_ids: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    mask_lookup: dict[tuple[int, int], int],
    reliability: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], float]:
    h, w = labels.shape[1:]
    rows: list[tuple[int, int, int, float, int]] = []
    in_image_count = 0
    valid_count = 0
    t0 = time.perf_counter()
    for fi, frame_id in enumerate(frame_ids):
        label = labels[fi]
        for ci in range(uv.shape[1]):
            if not bool(valid[fi, ci]):
                continue
            valid_count += 1
            u = float(uv[fi, ci, 0])
            v = float(uv[fi, ci, 1])
            if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
                continue
            x = int(round(u * (w - 1)))
            y = int(round(v * (h - 1)))
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            in_image_count += 1
            mask_id = int(label[y, x])
            mask_global_id = mask_lookup.get((int(frame_id), mask_id), -1)
            if mask_global_id < 0:
                continue
            value = float(reliability[ci] * visibility[fi, ci] * confidence[fi, ci])
            rows.append((ci, mask_global_id, int(frame_id), value, mask_id))
    runtime = time.perf_counter() - t0
    if rows:
        arr = np.asarray(rows, dtype=np.float64)
    else:
        arr = np.zeros((0, 5), dtype=np.float64)
    meta = {
        "valid_projection_count": valid_count,
        "in_image_projection_count": in_image_count,
        "incidence_row_count": int(arr.shape[0]),
    }
    return arr, meta, runtime


def _gpu_incidence(
    labels: np.ndarray,
    frame_ids: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    mask_global_by_label: np.ndarray,
    reliability: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any], float]:
    h, w = labels.shape[1:]
    labels_t = torch.as_tensor(labels, dtype=torch.int64, device=device)
    uv_t = torch.as_tensor(uv, dtype=torch.float32, device=device)
    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
    visibility_t = torch.as_tensor(visibility, dtype=torch.float32, device=device)
    confidence_t = torch.as_tensor(confidence, dtype=torch.float32, device=device)
    reliability_t = torch.as_tensor(reliability, dtype=torch.float32, device=device)
    map_t = torch.as_tensor(mask_global_by_label, dtype=torch.int64, device=device)
    frame_ids_t = torch.as_tensor(frame_ids, dtype=torch.int64, device=device)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    in01 = valid_t & (uv_t[..., 0] >= 0.0) & (uv_t[..., 0] <= 1.0) & (uv_t[..., 1] >= 0.0) & (uv_t[..., 1] <= 1.0)
    fi, ci = torch.nonzero(in01, as_tuple=True)
    x = torch.round(uv_t[fi, ci, 0] * float(w - 1)).to(torch.long)
    y = torch.round(uv_t[fi, ci, 1] * float(h - 1)).to(torch.long)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    fi = fi[inside]
    ci = ci[inside]
    x = x[inside]
    y = y[inside]
    label_at = labels_t[fi, y, x]
    mask_global = map_t[fi, label_at.clamp(min=0, max=map_t.shape[1] - 1)]
    keep = mask_global >= 0
    fi = fi[keep]
    ci = ci[keep]
    label_at = label_at[keep]
    mask_global = mask_global[keep]
    values = reliability_t[ci] * visibility_t[fi, ci] * confidence_t[fi, ci]
    out = torch.stack(
        [
            ci.to(torch.float64),
            mask_global.to(torch.float64),
            frame_ids_t[fi].to(torch.float64),
            values.to(torch.float64),
            label_at.to(torch.float64),
        ],
        dim=1,
    )
    torch.cuda.synchronize(device)
    runtime = time.perf_counter() - t0
    meta = {
        "valid_projection_count": int(valid_t.sum().item()),
        "in_image_projection_count": int(in01.sum().item()),
        "incidence_row_count": int(out.shape[0]),
    }
    return out.detach().cpu().numpy(), meta, runtime


def _gpu_incidence_cupy(
    labels: np.ndarray,
    frame_ids: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    mask_global_by_label: np.ndarray,
    reliability: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], float]:
    import cupy as cp

    frame_count, carrier_count = valid.shape
    h, w = labels.shape[1:]
    total = int(frame_count * carrier_count)
    with cp.cuda.Device(0):
        kernel = cp.RawKernel(INCIDENCE_KERNEL, "incidence_lookup_kernel")
        labels_g = cp.asarray(np.ascontiguousarray(labels), dtype=cp.int32)
        uv_g = cp.asarray(np.ascontiguousarray(uv.reshape(-1, 2)), dtype=cp.float32)
        valid_g = cp.asarray(np.ascontiguousarray(valid.reshape(-1).astype(np.uint8)), dtype=cp.uint8)
        visibility_g = cp.asarray(np.ascontiguousarray(visibility.reshape(-1)), dtype=cp.float32)
        confidence_g = cp.asarray(np.ascontiguousarray(confidence.reshape(-1)), dtype=cp.float32)
        reliability_g = cp.asarray(np.ascontiguousarray(reliability), dtype=cp.float32)
        map_g = cp.asarray(np.ascontiguousarray(mask_global_by_label), dtype=cp.int64)
        frame_ids_g = cp.asarray(np.ascontiguousarray(frame_ids), dtype=cp.int64)
        out_carrier = cp.empty((total,), dtype=cp.int64)
        out_mask = cp.empty((total,), dtype=cp.int64)
        out_frame = cp.empty((total,), dtype=cp.int64)
        out_label = cp.empty((total,), dtype=cp.int64)
        out_value = cp.empty((total,), dtype=cp.float32)
        out_count = cp.zeros((1,), dtype=cp.uint32)
        block = 256
        grid = (int(math.ceil(total / block)),)
        # Warm compile separately, then time the actual vectorized path.
        kernel(
            grid,
            (block,),
            (
                labels_g,
                uv_g,
                valid_g,
                visibility_g,
                confidence_g,
                reliability_g,
                map_g,
                frame_ids_g,
                out_carrier,
                out_mask,
                out_frame,
                out_label,
                out_value,
                out_count,
                frame_count,
                carrier_count,
                h,
                w,
                mask_global_by_label.shape[1],
            ),
        )
        cp.cuda.Stream.null.synchronize()
        out_count.fill(0)
        t0 = time.perf_counter()
        kernel(
            grid,
            (block,),
            (
                labels_g,
                uv_g,
                valid_g,
                visibility_g,
                confidence_g,
                reliability_g,
                map_g,
                frame_ids_g,
                out_carrier,
                out_mask,
                out_frame,
                out_label,
                out_value,
                out_count,
                frame_count,
                carrier_count,
                h,
                w,
                mask_global_by_label.shape[1],
            ),
        )
        count = int(out_count.get()[0])
        compact = cp.stack(
            [
                out_carrier[:count].astype(cp.float64),
                out_mask[:count].astype(cp.float64),
                out_frame[:count].astype(cp.float64),
                out_value[:count].astype(cp.float64),
                out_label[:count].astype(cp.float64),
            ],
            axis=1,
        )
        cp.cuda.Stream.null.synchronize()
        runtime = time.perf_counter() - t0
        out = cp.asnumpy(compact)
        in_image_count = count
        valid_projection_count = int(cp.sum(valid_g).item())
    meta = {
        "valid_projection_count": valid_projection_count,
        "in_image_projection_count": in_image_count,
        "incidence_row_count": int(out.shape[0]),
    }
    return out, meta, runtime


def _support_counts(incidence: np.ndarray, mask_count: int, carrier_count: int) -> tuple[np.ndarray, np.ndarray]:
    mask_support = np.zeros((mask_count,), dtype=np.int64)
    carrier_support = np.zeros((carrier_count,), dtype=np.int64)
    if incidence.size:
        for carrier_idx, mask_idx in incidence[:, :2].astype(np.int64):
            mask_support[mask_idx] += 1
            carrier_support[carrier_idx] += 1
    return mask_support, carrier_support


def _countsketch_cpu(
    incidence: np.ndarray,
    mask_weights: np.ndarray,
    carrier_count: int,
    mask_count: int,
    sketch_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense = np.zeros((carrier_count, mask_count), dtype=np.float32)
    sketch = np.zeros((carrier_count, sketch_dim), dtype=np.float32)
    mask_idx = incidence[:, 1].astype(np.int64) if incidence.size else np.zeros((0,), dtype=np.int64)
    h = ((mask_idx * 2654435761 + SKETCH_SEED) % sketch_dim).astype(np.int64)
    sign = np.where(((mask_idx * 1103515245 + SKETCH_SEED) % 2) == 0, 1.0, -1.0).astype(np.float32)
    for row_idx, row in enumerate(incidence):
        ci = int(row[0])
        mi = int(row[1])
        value = float(math.sqrt(float(mask_weights[mi])) * float(row[3]))
        dense[ci, mi] += value
        sketch[ci, h[row_idx]] += sign[row_idx] * value
    dense_norm = _normalize_rows(dense)
    sketch_norm = _normalize_rows(sketch)
    return dense_norm, sketch_norm, sketch


def _countsketch_gpu(
    incidence: np.ndarray,
    mask_weights: np.ndarray,
    carrier_count: int,
    sketch_dim: int,
    device: torch.device,
) -> torch.Tensor:
    inc_t = torch.as_tensor(incidence, dtype=torch.float32, device=device)
    mask_weights_t = torch.as_tensor(mask_weights, dtype=torch.float32, device=device)
    carrier_idx = inc_t[:, 0].to(torch.long)
    mask_idx = inc_t[:, 1].to(torch.long)
    values = torch.sqrt(mask_weights_t[mask_idx]) * inc_t[:, 3]
    bucket = ((mask_idx * 2654435761 + SKETCH_SEED) % sketch_dim).to(torch.long)
    sign = torch.where(((mask_idx * 1103515245 + SKETCH_SEED) % 2) == 0, 1.0, -1.0).to(torch.float32)
    out = torch.zeros((carrier_count, sketch_dim), dtype=torch.float32, device=device)
    out.index_put_((carrier_idx, bucket), sign * values, accumulate=True)
    return torch.nn.functional.normalize(out, p=2, dim=1, eps=1e-12)


def _sample_pair_error(a: np.ndarray, b: np.ndarray, max_items: int = 256) -> tuple[float, float]:
    valid = np.where((np.linalg.norm(a, axis=1) > 0) & (np.linalg.norm(b, axis=1) > 0))[0]
    if len(valid) < 2:
        return 0.0, 0.0
    valid = valid[: min(len(valid), max_items)]
    cos_a = a[valid] @ a[valid].T
    cos_b = b[valid] @ b[valid].T
    err = np.abs(cos_a - cos_b)
    return float(np.percentile(err, 95)), float(np.max(err))


def _mask_level_features_cpu(
    incidence: np.ndarray,
    raw_sketch: np.ndarray,
    sketch_norm_gpu: np.ndarray,
    mask_weights: np.ndarray,
    mask_count: int,
) -> np.ndarray:
    # Leave-one-out pooling: remove the mask's own sketch contribution before
    # pooling carriers back to the mask node.
    raw64 = raw_sketch.astype(np.float64, copy=False)
    out = np.zeros((mask_count, raw_sketch.shape[1]), dtype=np.float64)
    denom = np.zeros((mask_count,), dtype=np.float64)
    for row in incidence:
        ci = int(row[0])
        mi = int(row[1])
        value = float(math.sqrt(float(mask_weights[mi])) * float(row[3]))
        bucket = int((mi * 2654435761 + SKETCH_SEED) % raw_sketch.shape[1])
        sign = 1.0 if int((mi * 1103515245 + SKETCH_SEED) % 2) == 0 else -1.0
        loo = raw64[ci].copy()
        loo[bucket] -= sign * value
        norm = float(np.linalg.norm(loo))
        if norm > 1e-12:
            loo /= norm
        else:
            loo = sketch_norm_gpu[ci]
        alpha = float(row[3])
        out[mi] += alpha * loo
        denom[mi] += alpha
    out = out / np.maximum(denom[:, None], 1e-12)
    return _normalize_rows(out)


def _mask_level_features_gpu(
    incidence: np.ndarray,
    raw_sketch: np.ndarray,
    sketch_norm: torch.Tensor,
    mask_weights: np.ndarray,
    mask_count: int,
    device: torch.device,
) -> torch.Tensor:
    inc_t = torch.as_tensor(incidence, dtype=torch.float64, device=device)
    raw_t = torch.as_tensor(raw_sketch, dtype=torch.float64, device=device)
    mask_weights_t = torch.as_tensor(mask_weights, dtype=torch.float64, device=device)
    carrier_idx = inc_t[:, 0].to(torch.long)
    mask_idx = inc_t[:, 1].to(torch.long)
    alpha = inc_t[:, 3]
    bucket = ((mask_idx * 2654435761 + SKETCH_SEED) % raw_t.shape[1]).to(torch.long)
    sign = torch.where(((mask_idx * 1103515245 + SKETCH_SEED) % 2) == 0, 1.0, -1.0).to(torch.float64)
    value = torch.sqrt(mask_weights_t[mask_idx]) * alpha
    row_feat = raw_t[carrier_idx].clone()
    row_feat[torch.arange(row_feat.shape[0], device=device), bucket] -= sign * value
    row_norm = torch.linalg.norm(row_feat, dim=1)
    zero = row_norm <= 1e-12
    row_feat = torch.nn.functional.normalize(row_feat, p=2, dim=1, eps=1e-12)
    if bool(torch.any(zero).item()):
        row_feat[zero] = sketch_norm[carrier_idx[zero]].to(torch.float64)
    feat = row_feat * alpha[:, None]
    out = torch.zeros((mask_count, sketch_norm.shape[1]), dtype=torch.float64, device=device)
    denom = torch.zeros((mask_count,), dtype=torch.float64, device=device)
    out.index_add_(0, mask_idx, feat)
    denom.index_add_(0, mask_idx, alpha)
    out = out / torch.clamp(denom[:, None], min=1e-12)
    return torch.nn.functional.normalize(out, p=2, dim=1, eps=1e-12)


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    if not bool(phase0.get("phase0_pass")):
        raise RuntimeError("v103 Phase1 requires v103 Phase0 pass")
    if not torch.cuda.is_available():
        raise RuntimeError("v103 Phase1 requires torch CUDA for GPU parity")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    d4rt = np.load(D4RT_CHUNK)
    frame_ids = [int(x) for x in d4rt["frame_ids"]]
    labels, frame_to_local = _load_labels(frame_ids)
    _feature_rows, semantic_features, mask_rows = _load_feature_subset(frame_ids)
    mask_table = pd.DataFrame(mask_rows)
    mask_table_path = OUT_DIR / "mask_table.parquet"
    mask_table.to_parquet(mask_table_path, index=False)
    _write_csv(OUT_DIR / "mask_table_preview_rows.csv", mask_rows[:128])

    mask_bool, packed_bits = _build_mask_bitsets(mask_rows, labels, frame_to_local)
    mask_bitset_path = OUT_DIR / "mask_bitset.npz"
    np.savez_compressed(
        mask_bitset_path,
        packed_bits=packed_bits,
        area=mask_bool.sum(axis=1, dtype=np.int64),
        mask_global_id=np.asarray([int(r["mask_global_id"]) for r in mask_rows], dtype=np.int64),
        frame_id=np.asarray([int(r["frame_id"]) for r in mask_rows], dtype=np.int64),
        mask_id=np.asarray([int(r["mask_id"]) for r in mask_rows], dtype=np.int64),
        shape_hw=np.asarray(labels.shape[1:], dtype=np.int64),
    )

    carrier_rows, reliability = _carrier_table(d4rt)
    carrier_table_path = OUT_DIR / "carrier_table.parquet"
    pd.DataFrame(carrier_rows).to_parquet(carrier_table_path, index=False)
    _write_csv(OUT_DIR / "carrier_table_preview_rows.csv", carrier_rows[:128])

    uv = np.asarray(d4rt["uv"], dtype=np.float32)
    valid = np.asarray(d4rt["valid"], dtype=bool)
    visibility = np.asarray(d4rt["visibility"], dtype=np.float32)
    confidence = np.asarray(d4rt["confidence"], dtype=np.float32)
    carrier_projection_path = OUT_DIR / "carrier_projection.pt"
    torch.save(
        {
            "frame_ids": torch.as_tensor(frame_ids, dtype=torch.int64),
            "carrier_id": torch.as_tensor(np.asarray(d4rt["carrier_id"]), dtype=torch.int64),
            "u_norm": torch.as_tensor(uv[..., 0], dtype=torch.float32),
            "v_norm": torch.as_tensor(uv[..., 1], dtype=torch.float32),
            "valid": torch.as_tensor(valid, dtype=torch.bool),
            "confidence": torch.as_tensor(confidence, dtype=torch.float32),
            "visibility": torch.as_tensor(visibility, dtype=torch.float32),
        },
        carrier_projection_path,
    )

    mask_lookup = {(int(r["frame_id"]), int(r["mask_id"])): int(r["mask_global_id"]) for r in mask_rows}
    max_label = int(labels.max())
    mask_global_by_label = np.full((len(frame_ids), max_label + 1), -1, dtype=np.int64)
    for row in mask_rows:
        mask_global_by_label[frame_to_local[int(row["frame_id"])], int(row["mask_id"])] = int(row["mask_global_id"])

    cpu_inc, cpu_meta, cpu_inc_runtime = _cpu_incidence(
        labels,
        np.asarray(frame_ids, dtype=np.int64),
        uv,
        valid,
        visibility,
        confidence,
        mask_lookup,
        reliability,
    )
    gpu_inc, gpu_meta, gpu_inc_runtime = _gpu_incidence_cupy(
        labels,
        np.asarray(frame_ids, dtype=np.int64),
        uv,
        valid,
        visibility,
        confidence,
        mask_global_by_label,
        reliability,
    )
    cpu_order = np.lexsort((cpu_inc[:, 1], cpu_inc[:, 0], cpu_inc[:, 2])) if len(cpu_inc) else []
    gpu_order = np.lexsort((gpu_inc[:, 1], gpu_inc[:, 0], gpu_inc[:, 2])) if len(gpu_inc) else []
    cpu_sorted = cpu_inc[cpu_order] if len(cpu_inc) else cpu_inc
    gpu_sorted = gpu_inc[gpu_order] if len(gpu_inc) else gpu_inc
    incidence_shape_match = cpu_sorted.shape == gpu_sorted.shape
    incidence_index_mismatch_count = 0
    incidence_value_max_abs_error = 0.0
    if incidence_shape_match and len(cpu_sorted):
        incidence_index_mismatch_count = int(np.count_nonzero(cpu_sorted[:, :3].astype(np.int64) != gpu_sorted[:, :3].astype(np.int64)))
        incidence_value_max_abs_error = float(np.max(np.abs(cpu_sorted[:, 3] - gpu_sorted[:, 3])))
    elif not incidence_shape_match:
        incidence_index_mismatch_count = max(len(cpu_sorted), len(gpu_sorted))

    incidence_sparse_path = OUT_DIR / "incidence_sparse.pt"
    torch.save(
        {
            "carrier_index": torch.as_tensor(cpu_sorted[:, 0].astype(np.int64)),
            "mask_global_id": torch.as_tensor(cpu_sorted[:, 1].astype(np.int64)),
            "frame_id": torch.as_tensor(cpu_sorted[:, 2].astype(np.int64)),
            "B_ia": torch.as_tensor(cpu_sorted[:, 3].astype(np.float32)),
            "mask_id": torch.as_tensor(cpu_sorted[:, 4].astype(np.int64)),
            "source": "cpu_gpu_parity_matched_cpu_sorted",
        },
        incidence_sparse_path,
    )

    mask_support_cpu, carrier_support_cpu = _support_counts(cpu_sorted, len(mask_rows), len(carrier_rows))
    mask_support_gpu, carrier_support_gpu = _support_counts(gpu_sorted, len(mask_rows), len(carrier_rows))
    mask_support_count_mismatch = int(np.count_nonzero(mask_support_cpu != mask_support_gpu))
    carrier_support_count_mismatch = int(np.count_nonzero(carrier_support_cpu != carrier_support_gpu))

    visible_per_frame = defaultdict(int)
    for row in cpu_sorted:
        visible_per_frame[int(row[2])] += 1
    mask_area_ratio = np.asarray([float(r["area"]) / max(1.0, labels.shape[1] * labels.shape[2]) for r in mask_rows], dtype=np.float32)
    rho = mask_support_cpu.astype(np.float32) / max(1.0, float(np.max(list(visible_per_frame.values()) or [1])))
    idf = np.log(1.0 / np.maximum(rho, 1e-6)).astype(np.float32)
    quality = np.asarray([float(r["object_like_score"]) for r in mask_rows], dtype=np.float32)
    mask_weights = np.maximum(quality * idf, 1e-4).astype(np.float32)

    dense_exact, sketch_cpu_norm, sketch_cpu_raw = _countsketch_cpu(cpu_sorted, mask_weights, len(carrier_rows), len(mask_rows), SKETCH_DIM)
    torch.cuda.synchronize(device)
    sketch_t0 = time.perf_counter()
    sketch_gpu_norm_t = _countsketch_gpu(cpu_sorted, mask_weights, len(carrier_rows), SKETCH_DIM, device)
    torch.cuda.synchronize(device)
    sketch_gpu_runtime = time.perf_counter() - sketch_t0
    sketch_gpu_norm = sketch_gpu_norm_t.detach().cpu().numpy()
    countsketch_cpu_gpu_max_abs_error = float(np.max(np.abs(sketch_cpu_norm - sketch_gpu_norm))) if sketch_cpu_norm.size else 0.0
    exact_vs_sketch_p95, exact_vs_sketch_max = _sample_pair_error(dense_exact, sketch_cpu_norm)

    mask_feature_cpu = _mask_level_features_cpu(cpu_sorted, sketch_cpu_raw, sketch_cpu_norm, mask_weights, len(mask_rows))
    torch.cuda.synchronize(device)
    pool_t0 = time.perf_counter()
    mask_feature_gpu_t = _mask_level_features_gpu(
        cpu_sorted,
        sketch_cpu_raw,
        sketch_gpu_norm_t,
        mask_weights,
        len(mask_rows),
        device,
    )
    torch.cuda.synchronize(device)
    pooling_gpu_runtime = time.perf_counter() - pool_t0
    mask_feature_gpu = mask_feature_gpu_t.detach().cpu().numpy()
    mask_feature_cos_cpu = mask_feature_cpu @ mask_feature_cpu.T
    mask_feature_cos_gpu = mask_feature_gpu @ mask_feature_gpu.T
    mask_feature_cosine_max_abs_error = float(np.max(np.abs(mask_feature_cos_cpu - mask_feature_cos_gpu))) if mask_feature_cpu.size else 0.0

    semantic_features_path = OUT_DIR / "semantic_features.pt"
    torch.save(
        {
            "mask_global_id": torch.arange(len(mask_rows), dtype=torch.int64),
            "features": torch.as_tensor(semantic_features, dtype=torch.float32),
            "backend": "radio_radseg",
            "source": _rel(FEATURE_STORE),
        },
        semantic_features_path,
    )
    sem_cpu_t0 = time.perf_counter()
    sem_cpu = semantic_features @ semantic_features.T
    sem_cpu_runtime = time.perf_counter() - sem_cpu_t0
    sem_t = torch.as_tensor(semantic_features, dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    sem_gpu_t0 = time.perf_counter()
    sem_gpu_t = sem_t @ sem_t.T
    sem_topk_val, sem_topk_idx = torch.topk(sem_gpu_t, k=min(TOPK, sem_gpu_t.shape[1]), dim=1)
    torch.cuda.synchronize(device)
    sem_gpu_runtime = time.perf_counter() - sem_gpu_t0
    sem_gpu = sem_gpu_t.detach().cpu().numpy()
    semantic_similarity_max_abs_error = float(np.max(np.abs(sem_cpu - sem_gpu))) if sem_cpu.size else 0.0

    # History top-k parity-only operation. Tokens are deterministic centroids
    # over sorted mask semantic features; they are not method history memory.
    token_indices = np.array_split(np.arange(len(mask_rows)), HISTORY_TOKEN_COUNT)
    hist = []
    for indices in token_indices:
        hist.append(np.mean(semantic_features[indices], axis=0))
    hist_np = _normalize_rows(np.stack(hist).astype(np.float32, copy=False))
    carrier_sem = np.zeros((len(carrier_rows), semantic_features.shape[1]), dtype=np.float32)
    denom = np.zeros((len(carrier_rows),), dtype=np.float32)
    for row in cpu_sorted:
        ci = int(row[0])
        mi = int(row[1])
        weight = float(row[3])
        carrier_sem[ci] += weight * semantic_features[mi]
        denom[ci] += weight
    carrier_sem = carrier_sem / np.maximum(denom[:, None], 1e-12)
    carrier_sem = _normalize_rows(carrier_sem)
    valid_carriers = np.where(denom > 0)[0]
    hist_cpu_scores = carrier_sem[valid_carriers] @ hist_np.T
    hist_cpu_top1 = np.argmax(hist_cpu_scores, axis=1) if len(valid_carriers) else np.zeros((0,), dtype=np.int64)
    carrier_sem_t = torch.as_tensor(carrier_sem, dtype=torch.float32, device=device)
    hist_t = torch.as_tensor(hist_np, dtype=torch.float32, device=device)
    valid_t = torch.as_tensor(valid_carriers, dtype=torch.long, device=device)
    torch.cuda.synchronize(device)
    hist_t0 = time.perf_counter()
    hist_scores_t = carrier_sem_t[valid_t] @ hist_t.T
    hist_top1_t = torch.argmax(hist_scores_t, dim=1)
    torch.cuda.synchronize(device)
    history_gpu_runtime = time.perf_counter() - hist_t0
    hist_gpu_top1 = hist_top1_t.detach().cpu().numpy()
    history_top1_mismatch_count = int(np.count_nonzero(hist_cpu_top1 != hist_gpu_top1))
    history_top1_mismatch_rate = float(history_top1_mismatch_count / max(1, len(valid_carriers)))

    candidate_cpu = mask_feature_cos_cpu.copy()
    np.fill_diagonal(candidate_cpu, -np.inf)
    candidate_cpu_topk = np.argsort(candidate_cpu, axis=1)[:, -min(TOPK, candidate_cpu.shape[1]) :][:, ::-1]
    mask_feature_gpu_t = torch.as_tensor(mask_feature_gpu, dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    cand_t0 = time.perf_counter()
    cand_scores_t = mask_feature_gpu_t @ mask_feature_gpu_t.T
    cand_scores_t.fill_diagonal_(-float("inf"))
    cand_topk_val_t, cand_topk_idx_t = torch.topk(cand_scores_t, k=min(TOPK, cand_scores_t.shape[1]), dim=1)
    torch.cuda.synchronize(device)
    candidate_gpu_runtime = time.perf_counter() - cand_t0
    candidate_topk_idx = cand_topk_idx_t.detach().cpu().numpy()
    candidate_topk_mismatch_rate = float(np.mean(candidate_cpu_topk[:, 0] != candidate_topk_idx[:, 0])) if len(mask_rows) else 0.0

    countsketch_path = OUT_DIR / "primitive_affinity_feature.pt"
    torch.save(
        {
            "carrier_index": torch.arange(len(carrier_rows), dtype=torch.int64),
            "sketch_dim": SKETCH_DIM,
            "hash_seed": SKETCH_SEED,
            "feature": torch.as_tensor(sketch_cpu_norm, dtype=torch.float32),
            "exact_dense_subset": torch.as_tensor(dense_exact[: min(256, len(dense_exact))], dtype=torch.float32),
        },
        countsketch_path,
    )
    mask_level_feature_path = OUT_DIR / "mask_level_feature.pt"
    torch.save(
        {
            "mask_global_id": torch.arange(len(mask_rows), dtype=torch.int64),
            "feature": torch.as_tensor(mask_feature_cpu, dtype=torch.float32),
            "pooling": "leave_one_out_cpu_reference",
        },
        mask_level_feature_path,
    )
    history_topk_path = OUT_DIR / "history_topk_parity.pt"
    torch.save(
        {
            "history_token_count": HISTORY_TOKEN_COUNT,
            "valid_carrier_index": torch.as_tensor(valid_carriers, dtype=torch.int64),
            "cpu_top1": torch.as_tensor(hist_cpu_top1, dtype=torch.int64),
            "gpu_top1": torch.as_tensor(hist_gpu_top1, dtype=torch.int64),
            "note": PATCH_NOTE,
        },
        history_topk_path,
    )
    candidate_topk_path = OUT_DIR / "candidate_pair_topk.pt"
    torch.save(
        {
            "mask_global_id": torch.arange(len(mask_rows), dtype=torch.int64),
            "topk_index": torch.as_tensor(candidate_topk_idx, dtype=torch.int64),
            "topk_score": cand_topk_val_t.detach().cpu(),
        },
        candidate_topk_path,
    )

    incidence_speedup = float(cpu_inc_runtime / gpu_inc_runtime) if gpu_inc_runtime > 0 else 0.0
    parity_rows = [
        {
            "schema_version": "stream4d_v103_phase1_parity_row_v1",
            "phase_id": PHASE_ID,
            "check_id": "point_to_mask_incidence_lookup",
            "cpu_runtime_sec": cpu_inc_runtime,
            "gpu_runtime_sec": gpu_inc_runtime,
            "speedup_cpu_over_gpu": incidence_speedup,
            "mismatch_count": incidence_index_mismatch_count,
            "max_abs_error": incidence_value_max_abs_error,
            "tested_count": int(cpu_meta["valid_projection_count"]),
        },
        {
            "schema_version": "stream4d_v103_phase1_parity_row_v1",
            "phase_id": PHASE_ID,
            "check_id": "mask_support_count",
            "mismatch_count": mask_support_count_mismatch,
            "tested_count": len(mask_rows),
        },
        {
            "schema_version": "stream4d_v103_phase1_parity_row_v1",
            "phase_id": PHASE_ID,
            "check_id": "carrier_support_count",
            "mismatch_count": carrier_support_count_mismatch,
            "tested_count": len(carrier_rows),
        },
        {
            "schema_version": "stream4d_v103_phase1_parity_row_v1",
            "phase_id": PHASE_ID,
            "check_id": "countsketch_cpu_gpu_feature",
            "gpu_runtime_sec": sketch_gpu_runtime,
            "max_abs_error": countsketch_cpu_gpu_max_abs_error,
            "tested_count": int(sketch_cpu_norm.size),
        },
        {
            "schema_version": "stream4d_v103_phase1_parity_row_v1",
            "phase_id": PHASE_ID,
            "check_id": "exact_dense_vs_countsketch_cosine",
            "p95_error": exact_vs_sketch_p95,
            "max_abs_error": exact_vs_sketch_max,
            "tested_count": min(256, len(carrier_rows)),
        },
        {
            "schema_version": "stream4d_v103_phase1_parity_row_v1",
            "phase_id": PHASE_ID,
            "check_id": "mask_level_feature_cosine",
            "gpu_runtime_sec": pooling_gpu_runtime,
            "max_abs_error": mask_feature_cosine_max_abs_error,
            "tested_count": len(mask_rows),
            "cpu_reference": "leave_one_out_pooling",
            "gpu_reference": "leave_one_out_pooling_vectorized",
        },
        {
            "schema_version": "stream4d_v103_phase1_parity_row_v1",
            "phase_id": PHASE_ID,
            "check_id": "semantic_similarity_matrix",
            "cpu_runtime_sec": sem_cpu_runtime,
            "gpu_runtime_sec": sem_gpu_runtime,
            "max_abs_error": semantic_similarity_max_abs_error,
            "tested_count": int(semantic_features.shape[0] ** 2),
        },
        {
            "schema_version": "stream4d_v103_phase1_parity_row_v1",
            "phase_id": PHASE_ID,
            "check_id": "history_assignment_top1",
            "gpu_runtime_sec": history_gpu_runtime,
            "mismatch_count": history_top1_mismatch_count,
            "mismatch_rate": history_top1_mismatch_rate,
            "tested_count": len(valid_carriers),
            "note": PATCH_NOTE,
        },
        {
            "schema_version": "stream4d_v103_phase1_parity_row_v1",
            "phase_id": PHASE_ID,
            "check_id": "candidate_pair_topk",
            "gpu_runtime_sec": candidate_gpu_runtime,
            "top1_mismatch_rate": candidate_topk_mismatch_rate,
            "tested_count": len(mask_rows),
        },
    ]

    performance_rows = [
        {
            "schema_version": "stream4d_v103_phase1_performance_row_v1",
            "phase_id": PHASE_ID,
            "case_id": "point_to_mask_incidence_medium_subset",
            "cpu_runtime_sec": cpu_inc_runtime,
            "gpu_runtime_sec": gpu_inc_runtime,
            "speedup_cpu_over_gpu": incidence_speedup,
            "projection_count": int(cpu_meta["valid_projection_count"]),
            "incidence_row_count": int(cpu_meta["incidence_row_count"]),
            "device": str(device),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        {
            "schema_version": "stream4d_v103_phase1_performance_row_v1",
            "phase_id": PHASE_ID,
            "case_id": "semantic_similarity_medium_subset",
            "cpu_runtime_sec": sem_cpu_runtime,
            "gpu_runtime_sec": sem_gpu_runtime,
            "speedup_cpu_over_gpu": float(sem_cpu_runtime / sem_gpu_runtime) if sem_gpu_runtime > 0 else 0.0,
            "mask_count": len(mask_rows),
            "device": str(device),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    ]

    gate_rows = [
        {
            "gate_id": "incidence_mismatch_count_eq_0",
            "pass": incidence_shape_match and incidence_index_mismatch_count == 0 and incidence_value_max_abs_error <= 1e-6,
            "expected": "incidence indices equal and B_ia max abs error <= 1e-6",
            "observed": f"shape_match={incidence_shape_match}; mismatch={incidence_index_mismatch_count}; value_error={incidence_value_max_abs_error}",
            "severity": "required",
        },
        {
            "gate_id": "mask_support_count_mismatch_eq_0",
            "pass": mask_support_count_mismatch == 0 and carrier_support_count_mismatch == 0,
            "expected": "mask and carrier support mismatches are 0",
            "observed": f"mask={mask_support_count_mismatch}; carrier={carrier_support_count_mismatch}",
            "severity": "required",
        },
        {
            "gate_id": "countsketch_cosine_p95_error_le_0p02",
            "pass": exact_vs_sketch_p95 <= 0.02,
            "expected": "<=0.02",
            "observed": exact_vs_sketch_p95,
            "severity": "required",
        },
        {
            "gate_id": "semantic_cosine_max_abs_error_le_1e_5",
            "pass": semantic_similarity_max_abs_error <= 1e-5,
            "expected": "<=1e-5",
            "observed": semantic_similarity_max_abs_error,
            "severity": "required",
        },
        {
            "gate_id": "mask_level_feature_cosine_max_abs_error_le_1e_5",
            "pass": mask_feature_cosine_max_abs_error <= 1e-5,
            "expected": "<=1e-5",
            "observed": mask_feature_cosine_max_abs_error,
            "severity": "required",
        },
        {
            "gate_id": "history_top1_mismatch_rate_le_1pct",
            "pass": history_top1_mismatch_rate <= 0.01,
            "expected": "<=0.01",
            "observed": history_top1_mismatch_rate,
            "severity": "required",
        },
        {
            "gate_id": "gpu_runtime_le_cpu_div_5",
            "pass": gpu_inc_runtime <= cpu_inc_runtime / 5.0,
            "expected": f"<={cpu_inc_runtime / 5.0}",
            "observed": f"cpu={cpu_inc_runtime}; gpu={gpu_inc_runtime}; speedup={incidence_speedup}",
            "severity": "required",
        },
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v103_phase1_failure_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": {
                "incidence_mismatch_count_eq_0": "check uv-to-pixel rounding, frame indexing, and mask label mapping",
                "mask_support_count_mismatch_eq_0": "check sparse incidence sorting and support aggregation",
                "countsketch_cosine_p95_error_le_0p02": "increase sketch_dim or fix hash/sign parity",
                "semantic_cosine_max_abs_error_le_1e_5": "check feature normalization/dtype and GPU matmul precision",
                "mask_level_feature_cosine_max_abs_error_le_1e_5": "fix leave-one-out pooling parity and scatter-add dtype/order",
                "history_top1_mismatch_rate_le_1pct": "check top-k tie handling and feature normalization",
                "gpu_runtime_le_cpu_div_5": "remove remaining Python loops or batch GPU operations more aggressively",
            }.get(str(row["gate_id"]), "repair Phase1 data model parity"),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    phase1_pass = len(failure_rows) == 0

    artifact_rows = [
        {
            "schema_version": "stream4d_v103_phase1_artifact_row_v1",
            "phase_id": PHASE_ID,
            "role": "mask_table",
            "path": _rel(mask_table_path),
            "exists": mask_table_path.exists(),
            "size_bytes": mask_table_path.stat().st_size,
            "sha256": _sha256(mask_table_path),
        },
        {
            "schema_version": "stream4d_v103_phase1_artifact_row_v1",
            "phase_id": PHASE_ID,
            "role": "mask_bitset",
            "path": _rel(mask_bitset_path),
            "exists": mask_bitset_path.exists(),
            "size_bytes": mask_bitset_path.stat().st_size,
            "sha256": _sha256(mask_bitset_path),
        },
        {
            "schema_version": "stream4d_v103_phase1_artifact_row_v1",
            "phase_id": PHASE_ID,
            "role": "carrier_table",
            "path": _rel(carrier_table_path),
            "exists": carrier_table_path.exists(),
            "size_bytes": carrier_table_path.stat().st_size,
            "sha256": _sha256(carrier_table_path),
        },
        {
            "schema_version": "stream4d_v103_phase1_artifact_row_v1",
            "phase_id": PHASE_ID,
            "role": "carrier_projection",
            "path": _rel(carrier_projection_path),
            "exists": carrier_projection_path.exists(),
            "size_bytes": carrier_projection_path.stat().st_size,
            "sha256": _sha256(carrier_projection_path),
        },
        {
            "schema_version": "stream4d_v103_phase1_artifact_row_v1",
            "phase_id": PHASE_ID,
            "role": "incidence_sparse",
            "path": _rel(incidence_sparse_path),
            "exists": incidence_sparse_path.exists(),
            "size_bytes": incidence_sparse_path.stat().st_size,
            "sha256": _sha256(incidence_sparse_path),
        },
        {
            "schema_version": "stream4d_v103_phase1_artifact_row_v1",
            "phase_id": PHASE_ID,
            "role": "semantic_features",
            "path": _rel(semantic_features_path),
            "exists": semantic_features_path.exists(),
            "size_bytes": semantic_features_path.stat().st_size,
            "sha256": _sha256(semantic_features_path),
        },
        {
            "schema_version": "stream4d_v103_phase1_artifact_row_v1",
            "phase_id": PHASE_ID,
            "role": "primitive_affinity_feature",
            "path": _rel(countsketch_path),
            "exists": countsketch_path.exists(),
            "size_bytes": countsketch_path.stat().st_size,
            "sha256": _sha256(countsketch_path),
        },
        {
            "schema_version": "stream4d_v103_phase1_artifact_row_v1",
            "phase_id": PHASE_ID,
            "role": "mask_level_feature",
            "path": _rel(mask_level_feature_path),
            "exists": mask_level_feature_path.exists(),
            "size_bytes": mask_level_feature_path.stat().st_size,
            "sha256": _sha256(mask_level_feature_path),
        },
    ]

    _write_csv(OUT_DIR / "parity_rows.csv", parity_rows)
    _write_csv(OUT_DIR / "performance_rows.csv", performance_rows)
    _write_csv(OUT_DIR / "gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "artifact_rows.csv", artifact_rows)
    _write_csv(
        OUT_DIR / "incidence_casebook_rows.csv",
        [
            {
                "schema_version": "stream4d_v103_phase1_incidence_casebook_row_v1",
                "phase_id": PHASE_ID,
                "carrier_index": int(row[0]),
                "mask_global_id": int(row[1]),
                "frame_id": int(row[2]),
                "B_ia": float(row[3]),
                "mask_id": int(row[4]),
            }
            for row in cpu_sorted[:256]
        ],
    )

    summary = {
        "schema_version": "stream4d_v103_phase1_gpu_data_model_parity_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_ENTER_PHASE2_D4RT_QUERY" if phase1_pass else "BLOCK_PHASE2_REPAIR_GPU_DATA_MODEL",
        "phase1_pass": phase1_pass,
        "failure_count": len(failure_rows),
        "scene_id": SCENE_ID,
        "chunk_id": CHUNK_ID,
        "frame_count": len(frame_ids),
        "frame_ids_min": min(frame_ids),
        "frame_ids_max": max(frame_ids),
        "mask_count": len(mask_rows),
        "carrier_count": len(carrier_rows),
        "valid_projection_count": int(cpu_meta["valid_projection_count"]),
        "incidence_row_count": int(cpu_meta["incidence_row_count"]),
        "incidence_mismatch_count": incidence_index_mismatch_count,
        "mask_support_count_mismatch": mask_support_count_mismatch,
        "carrier_support_count_mismatch": carrier_support_count_mismatch,
        "exact_vs_sketch_cosine_p95_error": exact_vs_sketch_p95,
        "semantic_similarity_max_abs_error": semantic_similarity_max_abs_error,
        "history_top1_mismatch_rate": history_top1_mismatch_rate,
        "incidence_cpu_runtime_sec": cpu_inc_runtime,
        "incidence_gpu_runtime_sec": gpu_inc_runtime,
        "incidence_speedup_cpu_over_gpu": incidence_speedup,
        "sketch_dim": SKETCH_DIM,
        "sketch_seed": SKETCH_SEED,
        "synthetic_history_token_count_for_parity_only": HISTORY_TOKEN_COUNT,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "truthfulness_note": "Phase1 creates tensor data-model/parity artifacts only. Synthetic history tokens are used only to test top-k parity and are not method predictions.",
        "outputs": {
            "mask_table": _rel(mask_table_path),
            "mask_bitset": _rel(mask_bitset_path),
            "carrier_table": _rel(carrier_table_path),
            "carrier_projection": _rel(carrier_projection_path),
            "incidence_sparse": _rel(incidence_sparse_path),
            "semantic_features": _rel(semantic_features_path),
            "primitive_affinity_feature": _rel(countsketch_path),
            "mask_level_feature": _rel(mask_level_feature_path),
            "history_topk_parity": _rel(history_topk_path),
            "candidate_pair_topk": _rel(candidate_topk_path),
            "parity_rows": _rel(OUT_DIR / "parity_rows.csv"),
            "performance_rows": _rel(OUT_DIR / "performance_rows.csv"),
            "gate_rows": _rel(OUT_DIR / "gate_rows.csv"),
            "failure_rows": _rel(OUT_DIR / "failure_rows.csv"),
            "artifact_rows": _rel(OUT_DIR / "artifact_rows.csv"),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase1_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
