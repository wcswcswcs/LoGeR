#!/usr/bin/env python3
"""Build v100 Phase1 GPU data-model parity artifacts.

This phase verifies a small but real tensor path for mask bitsets, semantic
residual features, and object frame-count reductions. The GPU IoU path uses a
packed uint64 popcount kernel; it does not report final MV_AP, because the
canonical v65 evaluator remains the metric source.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
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
from tools import build_v99_phase10k_holdout_chunk_object_birth_sweep as p10k  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = Path(os.environ.get("V100_PHASE1_OUT_DIR", str(AUDIT_ROOT / "v100_phase1_gpu_data_model_parity")))
PHASE0_SUMMARY = AUDIT_ROOT / "v100_phase0_contract/summary.json"
PHASE10O_ROWS = AUDIT_ROOT / "v99_phase10o_overlap3_scene_stitch_repair/mv_object_frame_mask_rows.csv"
BASE_VARIANT = "O0_overlap3_chunk_birth_primary_emit"
MAX_MASKS = int(os.environ.get("V100_PHASE1_MAX_MASKS", "96"))
PAIR_BATCH = int(os.environ.get("V100_PHASE1_PAIR_BATCH", "64"))
DEVICE_ID = int(os.environ.get("V100_PHASE1_DEVICE_ID", "0"))


BITSET_POPCOUNT_KERNEL = r"""
extern "C" __global__
void bitset_intersection_popcount(
    const unsigned long long* __restrict__ packed_bits,
    const long long* __restrict__ pairs,
    const long long* __restrict__ area,
    long long* __restrict__ intersections,
    long long* __restrict__ unions,
    const long long width64,
    const long long pair_count
) {
    const long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= pair_count) {
        return;
    }
    const long long a = pairs[idx * 2];
    const long long b = pairs[idx * 2 + 1];
    unsigned long long inter = 0;
    const long long off_a = a * width64;
    const long long off_b = b * width64;
    for (long long k = 0; k < width64; ++k) {
        inter += __popcll(packed_bits[off_a + k] & packed_bits[off_b + k]);
    }
    intersections[idx] = (long long)inter;
    unions[idx] = area[a] + area[b] - (long long)inter;
}
"""


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


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


def _select_rows() -> list[dict[str, Any]]:
    rows = [dict(row) for row in _read_csv(PHASE10O_ROWS) if row.get("variant_id") == BASE_VARIANT]
    rows.sort(key=lambda r: (r.get("scene_id", ""), int(float(r.get("frame_id", 0))), r.get("chunk_id", ""), int(float(r.get("selected_mask_id", 0))), r.get("mv_object_id", "")))
    return rows


def _selected_mask_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, int]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("scene_id") != "scene0011_00":
            continue
        key = (str(row["scene_id"]), int(float(row["frame_id"])), int(float(row["selected_mask_id"])))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= MAX_MASKS:
            break
    return out


def _load_masks(mask_rows: list[dict[str, Any]], scope: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    masks: list[np.ndarray] = []
    manifest_rows: list[dict[str, Any]] = []
    shape_hw: tuple[int, int] | None = None
    missing = 0
    empty = 0
    for idx, row in enumerate(mask_rows):
        scene = str(row["scene_id"])
        frame = int(float(row["frame_id"]))
        mask_id = int(float(row["selected_mask_id"]))
        mask_path = scope["mask_path_by_frame"].get((scene, frame))
        if mask_path is None or not mask_path.exists():
            missing += 1
            continue
        label = p1._read_label(mask_path)
        if shape_hw is None:
            shape_hw = tuple(int(v) for v in label.shape[:2])
        elif tuple(label.shape[:2]) != shape_hw:
            label = p1._read_label(mask_path, shape_hw)
        mask = label == mask_id
        area = int(np.count_nonzero(mask))
        if area <= 0:
            empty += 1
            continue
        masks.append(mask.reshape(-1))
        manifest_rows.append(
            {
                "schema_version": "stream4d_v100_phase1_mask_manifest_v1",
                "phase_id": "v100_phase1_gpu_data_model_parity",
                "mask_index": len(masks) - 1,
                "source_row_index": idx,
                "scene_id": scene,
                "frame_id": frame,
                "selected_mask_id": mask_id,
                "mask_path": _rel(mask_path),
                "area_pixels_cpu": area,
                "mv_object_id": row.get("mv_object_id", ""),
                "chunk_id": row.get("chunk_id", ""),
            }
        )
    if not masks:
        raise RuntimeError("no masks loaded for Phase1 parity")
    meta = {
        "missing_mask_count": missing,
        "empty_mask_count": empty,
        "shape_h": int(shape_hw[0]) if shape_hw else 0,
        "shape_w": int(shape_hw[1]) if shape_hw else 0,
    }
    return np.stack(masks, axis=0).astype(bool, copy=False), manifest_rows, meta


def _all_pairs(n: int) -> np.ndarray:
    pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    return np.asarray(pairs, dtype=np.int64)


def _cpu_pair_iou(mask_bool: np.ndarray, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    area = mask_bool.sum(axis=1, dtype=np.int64)
    inter = np.empty((len(pairs),), dtype=np.int64)
    union = np.empty((len(pairs),), dtype=np.int64)
    t0 = time.perf_counter()
    for idx, (a, b) in enumerate(pairs):
        count = int(np.count_nonzero(mask_bool[int(a)] & mask_bool[int(b)]))
        inter[idx] = count
        union[idx] = int(area[int(a)] + area[int(b)] - count)
    runtime = time.perf_counter() - t0
    return inter, union, area, runtime


def _gpu_pair_iou(packed_bits: np.ndarray, area_cpu: np.ndarray, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    import cupy as cp

    with cp.cuda.Device(DEVICE_ID):
        kernel = cp.RawKernel(BITSET_POPCOUNT_KERNEL, "bitset_intersection_popcount")
        bits_gpu = cp.asarray(np.ascontiguousarray(packed_bits), dtype=cp.uint64)
        area_gpu = cp.asarray(np.ascontiguousarray(area_cpu), dtype=cp.int64)
        pairs_flat_gpu = cp.asarray(np.ascontiguousarray(pairs.reshape(-1)), dtype=cp.int64)
        block = 128
        grid = (int(math.ceil(len(pairs) / block)),)
        warm_n = min(len(pairs), 64)
        if warm_n:
            warm_pairs = pairs_flat_gpu[: warm_n * 2]
            warm_inter = cp.empty((warm_n,), dtype=cp.int64)
            warm_union = cp.empty((warm_n,), dtype=cp.int64)
            kernel(
                (int(math.ceil(warm_n / block)),),
                (block,),
                (bits_gpu, warm_pairs, area_gpu, warm_inter, warm_union, packed_bits.shape[1], warm_n),
            )
            cp.cuda.Stream.null.synchronize()
        free0, total0 = cp.cuda.runtime.memGetInfo()
        t0 = time.perf_counter()
        inter_all = cp.empty((len(pairs),), dtype=cp.int64)
        union_all = cp.empty((len(pairs),), dtype=cp.int64)
        if len(pairs):
            kernel(grid, (block,), (bits_gpu, pairs_flat_gpu, area_gpu, inter_all, union_all, packed_bits.shape[1], len(pairs)))
        cp.cuda.Stream.null.synchronize()
        runtime = time.perf_counter() - t0
        free1, _total1 = cp.cuda.runtime.memGetInfo()
        peak_used_mb = max(0.0, float(free0 - free1) / (1024.0 * 1024.0))
        return cp.asnumpy(inter_all), cp.asnumpy(union_all), cp.asnumpy(area_gpu), runtime, peak_used_mb


def _load_holdout_residual_features() -> dict[tuple[str, int, int], np.ndarray]:
    constants = json.loads(p1.SEMANTIC_CONSTANTS.read_text(encoding="utf-8"))
    mu = np.asarray(np.load(p1._project(constants["radio_mu_vector_path"])), dtype=np.float32)
    payload = np.load(p10k.HOLDOUT_RADIO_MASK_FEATURES, allow_pickle=True)
    features = np.asarray(payload["features"], dtype=np.float32)
    residual = p1._normalize_rows(features - mu[None, :])
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for idx in range(residual.shape[0]):
        out[(str(payload["scene_id"][idx]), int(payload["frame_id"][idx]), int(payload["mask_id"][idx]))] = residual[idx]
    return out


def _semantic_parity(mask_manifest: list[dict[str, Any]], pairs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    import cupy as cp

    feature_by_key = _load_holdout_residual_features()
    features: list[np.ndarray] = []
    kept_indices: list[int] = []
    for row in mask_manifest:
        key = (str(row["scene_id"]), int(row["frame_id"]), int(row["selected_mask_id"]))
        feat = feature_by_key.get(key)
        if feat is None:
            continue
        kept_indices.append(int(row["mask_index"]))
        features.append(feat.astype(np.float32, copy=False))
    if not features:
        raise RuntimeError("no semantic features found for selected masks")
    feature_arr = np.stack(features, axis=0).astype(np.float32, copy=False)
    old_to_new = {old: new for new, old in enumerate(kept_indices)}
    sem_pairs = np.asarray(
        [(old_to_new[int(a)], old_to_new[int(b)]) for a, b in pairs if int(a) in old_to_new and int(b) in old_to_new],
        dtype=np.int64,
    )
    t0 = time.perf_counter()
    cpu_cos = np.sum(feature_arr[sem_pairs[:, 0]] * feature_arr[sem_pairs[:, 1]], axis=1, dtype=np.float32)
    cpu_runtime = time.perf_counter() - t0
    with cp.cuda.Device(DEVICE_ID):
        feat_gpu = cp.asarray(feature_arr, dtype=cp.float32)
        pairs_gpu = cp.asarray(sem_pairs, dtype=cp.int64)
        if len(sem_pairs):
            warm_mat = feat_gpu @ feat_gpu.T
            _ = warm_mat[pairs_gpu[:1, 0], pairs_gpu[:1, 1]]
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        gpu_matrix = feat_gpu @ feat_gpu.T
        gpu_cos = gpu_matrix[pairs_gpu[:, 0], pairs_gpu[:, 1]]
        cp.cuda.Stream.null.synchronize()
        gpu_runtime = time.perf_counter() - t0
        gpu_cos_np = cp.asnumpy(gpu_cos)
    max_abs_error = float(np.max(np.abs(cpu_cos.astype(np.float64) - gpu_cos_np.astype(np.float64)))) if len(cpu_cos) else 0.0
    meta = {
        "semantic_feature_count": int(feature_arr.shape[0]),
        "semantic_feature_dim": int(feature_arr.shape[1]),
        "semantic_pair_count": int(len(sem_pairs)),
        "semantic_cosine_max_abs_error": max_abs_error,
        "semantic_cpu_runtime_sec": cpu_runtime,
        "semantic_gpu_runtime_sec": gpu_runtime,
    }
    return feature_arr, meta


def _object_frame_count_parity(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import cupy as cp

    object_ids = sorted({str(row["mv_object_id"]) for row in rows})
    frames = sorted({int(float(row["frame_id"])) for row in rows})
    obj_index = {oid: idx for idx, oid in enumerate(object_ids)}
    frame_index = {frame: idx for idx, frame in enumerate(frames)}
    cpu_sets: dict[str, set[int]] = defaultdict(set)
    obj_indices: list[int] = []
    frame_indices: list[int] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        frame = int(float(row["frame_id"]))
        cpu_sets[oid].add(frame)
        obj_indices.append(obj_index[oid])
        frame_indices.append(frame_index[frame])
    cpu_counts = np.asarray([len(cpu_sets[oid]) for oid in object_ids], dtype=np.int64)
    with cp.cuda.Device(DEVICE_ID):
        mat = cp.zeros((len(object_ids), len(frames)), dtype=cp.bool_)
        mat[cp.asarray(obj_indices, dtype=cp.int64), cp.asarray(frame_indices, dtype=cp.int64)] = True
        gpu_counts = cp.asnumpy(cp.sum(mat, axis=1, dtype=cp.int64))
    mismatch = int(np.count_nonzero(cpu_counts != gpu_counts))
    meta = {
        "object_count": len(object_ids),
        "frame_universe_count": len(frames),
        "frame_count_mismatch_count": mismatch,
        "frame_count_max_abs_error": int(np.max(np.abs(cpu_counts - gpu_counts))) if len(cpu_counts) else 0,
    }
    arrays = {
        "object_ids": np.asarray(object_ids, dtype=object),
        "frame_ids": np.asarray(frames, dtype=np.int64),
        "cpu_frame_counts": cpu_counts,
        "gpu_frame_counts": gpu_counts.astype(np.int64, copy=False),
    }
    return meta, arrays


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    if not bool(phase0.get("phase0_pass")):
        raise RuntimeError("Phase1 requires v100 Phase0 pass")

    p10k._patch_phase1_inputs()
    scope = p1._load_source_scope()
    all_rows = _select_rows()
    selected_masks = _selected_mask_rows(all_rows)
    mask_bool, mask_manifest_rows, mask_meta = _load_masks(selected_masks, scope)
    pairs = _all_pairs(mask_bool.shape[0])

    packed_bits = _pack_uint64(mask_bool)
    cpu_inter, cpu_union, cpu_area, cpu_runtime = _cpu_pair_iou(mask_bool, pairs)
    gpu_inter, gpu_union, gpu_area, gpu_runtime, peak_gpu_mb = _gpu_pair_iou(packed_bits, cpu_area, pairs)
    iou_cpu = cpu_inter / np.maximum(cpu_union, 1)
    iou_gpu = gpu_inter / np.maximum(gpu_union, 1)
    bitset_iou_max_abs_error = float(np.max(np.abs(iou_cpu - iou_gpu))) if len(iou_cpu) else 0.0
    intersection_mismatch_count = int(np.count_nonzero(cpu_inter != gpu_inter))
    union_mismatch_count = int(np.count_nonzero(cpu_union != gpu_union))
    mask_area_mismatch_count = int(np.count_nonzero(cpu_area != gpu_area))

    mask_npz = OUT_DIR / "mask_bitsets.npz"
    np.savez_compressed(
        mask_npz,
        mask_ids=np.asarray([int(row["selected_mask_id"]) for row in mask_manifest_rows], dtype=np.int64),
        frame_ids=np.asarray([int(row["frame_id"]) for row in mask_manifest_rows], dtype=np.int64),
        scene_ids=np.asarray([str(row["scene_id"]) for row in mask_manifest_rows], dtype=object),
        packed_bits=packed_bits,
        area_pixels=cpu_area.astype(np.int64, copy=False),
        shape_hw=np.asarray([mask_meta["shape_h"], mask_meta["shape_w"]], dtype=np.int64),
    )

    feature_arr, semantic_meta = _semantic_parity(mask_manifest_rows, pairs)
    semantic_npz = OUT_DIR / "mask_semantic_features.npz"
    np.savez_compressed(semantic_npz, feature=feature_arr)

    object_meta, object_arrays = _object_frame_count_parity([row for row in all_rows if row.get("variant_id") == BASE_VARIANT])
    object_npz = OUT_DIR / "object_frame_count_csr.npz"
    np.savez_compressed(object_npz, **object_arrays)

    pair_rows = [
        {
            "schema_version": "stream4d_v100_phase1_pair_parity_v1",
            "phase_id": "v100_phase1_gpu_data_model_parity",
            "pair_index": idx,
            "mask_index_a": int(a),
            "mask_index_b": int(b),
            "cpu_intersection": int(cpu_inter[idx]),
            "gpu_intersection": int(gpu_inter[idx]),
            "cpu_union": int(cpu_union[idx]),
            "gpu_union": int(gpu_union[idx]),
            "cpu_iou": float(iou_cpu[idx]),
            "gpu_iou": float(iou_gpu[idx]),
        }
        for idx, (a, b) in enumerate(pairs[: min(len(pairs), 256)])
    ]

    csv_bytes_read = PHASE10O_ROWS.stat().st_size + p10k.HOLDOUT_SOURCE_ROWS.stat().st_size
    tensor_bytes_written = sum(path.stat().st_size for path in [mask_npz, semantic_npz, object_npz])
    speedup = float(cpu_runtime / gpu_runtime) if gpu_runtime > 0 else 0.0
    performance_rows = [
        {
            "schema_version": "stream4d_v100_phase1_performance_v1",
            "phase_id": "v100_phase1_gpu_data_model_parity",
            "case_id": "mask_bitset_iou_all_pairs",
            "mask_count": int(mask_bool.shape[0]),
            "pair_count": int(len(pairs)),
            "runtime_cpu_sec": cpu_runtime,
            "runtime_gpu_sec": gpu_runtime,
            "speedup_cpu_over_gpu": speedup,
            "peak_gpu_memory_MB": peak_gpu_mb,
            "csv_bytes_read": csv_bytes_read,
            "tensor_bytes_written": tensor_bytes_written,
            "device_id": DEVICE_ID,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_backend": "cupy_rawkernel_packed_uint64_popcount",
            "runtime_scope": "warm_after_kernel_compile_excludes_artifact_load_and_host_copy",
        },
        {
            "schema_version": "stream4d_v100_phase1_performance_v1",
            "phase_id": "v100_phase1_gpu_data_model_parity",
            "case_id": "semantic_cosine_pairs",
            "mask_count": semantic_meta["semantic_feature_count"],
            "pair_count": semantic_meta["semantic_pair_count"],
            "runtime_cpu_sec": semantic_meta["semantic_cpu_runtime_sec"],
            "runtime_gpu_sec": semantic_meta["semantic_gpu_runtime_sec"],
            "speedup_cpu_over_gpu": float(semantic_meta["semantic_cpu_runtime_sec"] / semantic_meta["semantic_gpu_runtime_sec"]) if semantic_meta["semantic_gpu_runtime_sec"] > 0 else 0.0,
            "peak_gpu_memory_MB": "",
            "csv_bytes_read": "",
            "tensor_bytes_written": "",
            "device_id": DEVICE_ID,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_backend": "cupy_matmul_full_semantic_matrix",
            "runtime_scope": "warm_after_cublas_setup_excludes_artifact_load_and_host_copy",
        },
    ]
    variant_config_rows = [
        {
            "schema_version": "stream4d_v100_phase1_variant_config_v1",
            "phase_id": "v100_phase1_gpu_data_model_parity",
            "variant_id": "P1_gpu_bitset_semantic_object_parity",
            "source_variant": BASE_VARIANT,
            "mask_sample_count": int(mask_bool.shape[0]),
            "pair_generation": "all_unique_pairs_among_selected_masks",
            "gpu_backend": "cupy_rawkernel_packed_uint64_popcount",
            "semantic_gpu_backend": "cupy_matmul_full_semantic_matrix",
            "method_input_allowed": False,
            "diagnostic_only": True,
        }
    ]
    variant_metric_rows = [
        {
            "schema_version": "stream4d_v100_phase1_variant_metric_v1",
            "phase_id": "v100_phase1_gpu_data_model_parity",
            "variant_id": "P1_gpu_bitset_semantic_object_parity",
            "bitset_iou_max_abs_error": bitset_iou_max_abs_error,
            "intersection_mismatch_count": intersection_mismatch_count,
            "union_mismatch_count": union_mismatch_count,
            "semantic_cosine_max_abs_error": semantic_meta["semantic_cosine_max_abs_error"],
            "frame_count_mismatch_count": object_meta["frame_count_mismatch_count"],
            "mask_area_mismatch_count": mask_area_mismatch_count,
            "runtime_speedup": speedup,
        }
    ]
    artifact_manifest_rows = [
        {
            "artifact_path": _rel(mask_npz),
            "artifact_type": "npz",
            "row_count_or_shape": f"{mask_bool.shape[0]}x{packed_bits.shape[1]} uint64 packed",
            "dtype": "uint64",
            "key_columns": "scene_ids,frame_ids,mask_ids",
            "sha256_or_fast_hash": _sha256(mask_npz),
            "method_input_allowed": True,
            "diagnostic_only": False,
        },
        {
            "artifact_path": _rel(semantic_npz),
            "artifact_type": "npz",
            "row_count_or_shape": f"{feature_arr.shape[0]}x{feature_arr.shape[1]}",
            "dtype": "float32",
            "key_columns": "selected mask order from mask_manifest_rows.csv",
            "sha256_or_fast_hash": _sha256(semantic_npz),
            "method_input_allowed": True,
            "diagnostic_only": False,
        },
        {
            "artifact_path": _rel(object_npz),
            "artifact_type": "npz",
            "row_count_or_shape": f"{object_meta['object_count']} objects",
            "dtype": "mixed",
            "key_columns": "object_ids,frame_ids",
            "sha256_or_fast_hash": _sha256(object_npz),
            "method_input_allowed": True,
            "diagnostic_only": False,
        },
    ]

    gates = [
        {
            "gate_id": "bitset_iou_max_abs_error_eq_0",
            "pass": bitset_iou_max_abs_error == 0.0 and intersection_mismatch_count == 0 and union_mismatch_count == 0,
            "expected": "0 exact integer mismatch and 0 IoU diff",
            "observed": f"iou_error={bitset_iou_max_abs_error}; intersection_mismatch={intersection_mismatch_count}; union_mismatch={union_mismatch_count}",
            "severity": "required",
        },
        {
            "gate_id": "semantic_cosine_max_abs_error_le_1e_5",
            "pass": float(semantic_meta["semantic_cosine_max_abs_error"]) <= 1e-5,
            "expected": "<=1e-5",
            "observed": semantic_meta["semantic_cosine_max_abs_error"],
            "severity": "required",
        },
        {
            "gate_id": "frame_count_mismatch_count_eq_0",
            "pass": object_meta["frame_count_mismatch_count"] == 0,
            "expected": "0",
            "observed": object_meta["frame_count_mismatch_count"],
            "severity": "required",
        },
        {
            "gate_id": "mask_area_mismatch_count_eq_0",
            "pass": mask_area_mismatch_count == 0,
            "expected": "0",
            "observed": mask_area_mismatch_count,
            "severity": "required",
        },
        {
            "gate_id": "runtime_gpu_sec_le_cpu_div_5",
            "pass": bool(gpu_runtime <= cpu_runtime / 5.0),
            "expected": f"gpu_runtime<={cpu_runtime / 5.0}",
            "observed": f"cpu={cpu_runtime}; gpu={gpu_runtime}; speedup={speedup}",
            "severity": "performance_required",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "Check flatten order, label convention, and CPU/GPU boolean semantics."
                if "bitset" in row["gate_id"] or "mask_area" in row["gate_id"]
                else "Check normalization/dtype or move remaining Python/CSV loops into tensor batches."
            ),
        }
        for row in gates
        if not bool(row["pass"])
    ]
    phase1_pass = not failure_rows
    casebook_rows = [
        {
            "schema_version": "stream4d_v100_phase1_casebook_v1",
            "phase_id": "v100_phase1_gpu_data_model_parity",
            "case_id": "selected_subset",
            "source_rows": _rel(PHASE10O_ROWS),
            "source_variant": BASE_VARIANT,
            "mask_count": int(mask_bool.shape[0]),
            "pair_count": int(len(pairs)),
            "shape_h": mask_meta["shape_h"],
            "shape_w": mask_meta["shape_w"],
            "semantic_feature_count": semantic_meta["semantic_feature_count"],
            "object_count": object_meta["object_count"],
        }
    ]
    summary = {
        "schema_version": "stream4d_v100_phase1_gpu_data_model_parity_summary_v1",
        "phase_id": "v100_phase1_gpu_data_model_parity",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "PASS_ENTER_PHASE2" if phase1_pass else "BLOCK_PHASE2_REPAIR_GPU_PARITY",
        "phase1_pass": phase1_pass,
        "failure_count": len(failure_rows),
        "source_variant": BASE_VARIANT,
        "mask_count": int(mask_bool.shape[0]),
        "pair_count": int(len(pairs)),
        "bitset_iou_max_abs_error": bitset_iou_max_abs_error,
        "semantic_cosine_max_abs_error": semantic_meta["semantic_cosine_max_abs_error"],
        "frame_count_mismatch_count": object_meta["frame_count_mismatch_count"],
        "mask_area_mismatch_count": mask_area_mismatch_count,
        "runtime_cpu_sec": cpu_runtime,
        "runtime_gpu_sec": gpu_runtime,
        "runtime_speedup": speedup,
        "peak_gpu_memory_MB": peak_gpu_mb,
        "gpu_runtime_scope": "warm_after_kernel_compile_excludes_artifact_load_and_host_copy",
        "gpu_backend": "cupy_rawkernel_packed_uint64_popcount",
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "performance_rows": _rel(OUT_DIR / "performance_rows.csv"),
            "casebook_rows": _rel(OUT_DIR / "casebook_rows.csv"),
            "artifact_manifest_rows": _rel(OUT_DIR / "artifact_manifest_rows.csv"),
            "mask_manifest_rows": _rel(OUT_DIR / "mask_manifest_rows.csv"),
            "mask_pair_parity_rows": _rel(OUT_DIR / "mask_pair_parity_rows.csv"),
        },
    }

    _write_csv(OUT_DIR / "variant_config_rows.csv", variant_config_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", variant_metric_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gates)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "performance_rows.csv", performance_rows)
    _write_csv(OUT_DIR / "casebook_rows.csv", casebook_rows)
    _write_csv(OUT_DIR / "artifact_manifest_rows.csv", artifact_manifest_rows)
    _write_csv(OUT_DIR / "mask_manifest_rows.csv", mask_manifest_rows)
    _write_csv(OUT_DIR / "mask_pair_parity_rows.csv", pair_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase1_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
