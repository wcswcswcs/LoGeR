#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
STREAM3D_ROOT = ROOT / "Stream3D"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(STREAM3D_ROOT))

from build_v98_1_da3_variant_geometry_quality import (  # noqa: E402
    _chamfer_metrics,
    _filter_gt_to_input_visible,
    _load_da3_manifest,
    _read_gt_point_cloud,
    _residual_stats,
    _sample_indices,
    _write_csv,
    _write_json,
)
from geometry_provider.common import backproject_xy_world, fit_transform  # noqa: E402
from serve_v98_1_da3_gt_dense_rgb_sim3_viewer import _json_default  # noqa: E402
from stream4d.d4rt_adapter import D4RTAdapter  # noqa: E402
from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.sim3 import apply_sim3_to_xyz  # noqa: E402


DEFAULT_BASE_ROOT = STREAM3D_ROOT / "outputs" / "audit" / "v98_1_da3_variant_geometry_quality_scene0050_input_visible_gt"
DEFAULT_OUTPUT_ROOT = STREAM3D_ROOT / "outputs" / "audit" / "v98_1_d4rt_dense120x160_geometry_comparison_scene0050_input_visible_gt"
DEFAULT_D4RT_ROOT = ROOT / "Open-d4rt"
DEFAULT_D4RT_CONFIG = DEFAULT_D4RT_ROOT / "checkpoints" / "OpenD4RT_32CLIP_9Dataset_NoAUG" / "model.yaml"
DEFAULT_D4RT_CKPT = DEFAULT_D4RT_ROOT / "checkpoints" / "OpenD4RT_32CLIP_9Dataset_NoAUG" / "opend4rt.ckpt"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float32), -60.0, 60.0)))).astype(np.float32)


def _build_uv_grid(width: int, height: int, rows: int, cols: int, margin_ratio: float) -> np.ndarray:
    width = max(1, int(width))
    height = max(1, int(height))
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    margin_x = float(np.clip(margin_ratio, 0.0, 0.49)) * float(max(width - 1, 1))
    margin_y = float(np.clip(margin_ratio, 0.0, 0.49)) * float(max(height - 1, 1))
    xs = np.linspace(margin_x, float(width - 1) - margin_x, num=cols, dtype=np.float32)
    ys = np.linspace(margin_y, float(height - 1) - margin_y, num=rows, dtype=np.float32)
    return np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2).astype(np.float32)


def _make_overlap_windows(count: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    count = max(0, int(count))
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    if count <= chunk_size:
        return [(0, count)]
    step = chunk_size - overlap
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < count:
        end = min(count, start + chunk_size)
        ranges.append((int(start), int(end)))
        if end >= count:
            break
        start += step
    last_start = max(0, count - chunk_size)
    if ranges[-1][0] != last_start:
        ranges.append((int(last_start), int(count)))
    return ranges


def _sample_rgb_sequence(video_rgb: np.ndarray, uv_px: np.ndarray) -> np.ndarray:
    video = np.asarray(video_rgb, dtype=np.uint8)
    uv = np.asarray(uv_px, dtype=np.float32)
    x = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, max(video.shape[2] - 1, 0))
    y = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, max(video.shape[1] - 1, 0))
    out = np.empty((video.shape[0], uv.shape[0], 3), dtype=np.uint8)
    for idx in range(video.shape[0]):
        out[idx] = video[idx, y, x]
    return out


def _decode_dense_chunk(
    *,
    adapter: D4RTAdapter,
    video_rgb_uint8: np.ndarray,
    query_uv_norm: np.ndarray,
    query_chunk_size: int,
    aspect_source: str,
    camera_frame_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from src.eval.tasks import _encode_model_memory, _run_model_for_queries

    started = time.time()
    video_rgb_uint8 = np.asarray(video_rgb_uint8, dtype=np.uint8)
    num_frames = int(video_rgb_uint8.shape[0])
    num_queries = int(query_uv_norm.shape[0])
    if num_frames > int(adapter.clip_frames):
        raise ValueError(f"D4RT dense chunk has {num_frames} frames but model clip_frames={adapter.clip_frames}")

    resize_t0 = time.time()
    video_model = adapter._resize_video(video_rgb_uint8)
    seconds_resize = float(time.time() - resize_t0)
    native_h, native_w = int(video_rgb_uint8.shape[1]), int(video_rgb_uint8.shape[2])
    model_h, model_w = int(video_model.shape[1]), int(video_model.shape[2])
    if str(aspect_source) == "native_rgb":
        aspect_value = float(native_w) / float(max(native_h, 1))
    elif str(aspect_source) == "model_input":
        aspect_value = float(model_w) / float(max(model_h, 1))
    else:
        raise ValueError(f"unsupported aspect_source={aspect_source!r}")
    aspect = np.asarray([[aspect_value]], dtype=np.float32)
    video_tensor = (
        torch.from_numpy(video_model)
        .to(device=adapter.device, dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        / 255.0
    )
    aspect_tensor = torch.from_numpy(aspect).to(device=adapter.device, dtype=torch.float32)

    target_ids = np.arange(num_frames, dtype=np.int64)
    repeated_uv = np.tile(np.asarray(query_uv_norm, dtype=np.float32), (num_frames, 1))
    t_src = np.repeat(target_ids, num_queries)
    t_tgt = np.repeat(target_ids, num_queries)
    if str(camera_frame_mode) == "ref0":
        t_cam = np.zeros_like(t_tgt)
    elif str(camera_frame_mode) == "target_local":
        t_cam = t_tgt.copy()
    else:
        raise ValueError(f"unsupported camera_frame_mode={camera_frame_mode!r}")
    query = {
        "u": torch.from_numpy(repeated_uv[:, 0]).to(device=adapter.device, dtype=torch.float32),
        "v": torch.from_numpy(repeated_uv[:, 1]).to(device=adapter.device, dtype=torch.float32),
        "t_src": torch.from_numpy(t_src).to(device=adapter.device, dtype=torch.long),
        "t_tgt": torch.from_numpy(t_tgt).to(device=adapter.device, dtype=torch.long),
        "t_cam": torch.from_numpy(t_cam).to(device=adapter.device, dtype=torch.long),
    }

    with torch.inference_mode():
        encode_t0 = time.time()
        memory = _encode_model_memory(model=adapter.model, video_b=video_tensor, aspect_b=aspect_tensor)
        seconds_encode = float(time.time() - encode_t0)
        decode_t0 = time.time()
        pred = _run_model_for_queries(
            model=adapter.model,
            video_b=video_tensor,
            aspect_b=aspect_tensor,
            query=query,
            chunk_size=max(1, int(query_chunk_size)),
            memory_b=memory,
        )
        seconds_decode = float(time.time() - decode_t0)

    xyz = pred["xyz_3d"].numpy().astype(np.float32).reshape(num_frames, num_queries, 3)
    confidence_raw = pred.get("confidence", torch.ones((num_frames * num_queries,), dtype=torch.float32)).numpy().astype(np.float32)
    confidence_prob = _sigmoid_np(confidence_raw).reshape(num_frames, num_queries)
    if "visibility" in pred:
        visibility_prob = _sigmoid_np(pred["visibility"].numpy().astype(np.float32)).reshape(num_frames, num_queries)
    else:
        visibility_prob = np.ones((num_frames, num_queries), dtype=np.float32)
    valid = np.isfinite(xyz).all(axis=-1)
    info = {
        "num_frames": int(num_frames),
        "num_queries_per_frame": int(num_queries),
        "num_queries_total": int(num_frames * num_queries),
        "query_chunk_size": int(query_chunk_size),
        "image_hw_model": [int(adapter.image_hw[0]), int(adapter.image_hw[1])],
        "aspect_ratio_source": str(aspect_source),
        "aspect_ratio_value": float(aspect_value),
        "camera_frame_mode": str(camera_frame_mode),
        "seconds_resize": seconds_resize,
        "seconds_encode": seconds_encode,
        "seconds_decode": seconds_decode,
        "seconds_total": float(time.time() - started),
    }
    return xyz, valid, visibility_prob, confidence_prob, info


def _fit_overlap_sim3(
    *,
    prev_xyz: np.ndarray,
    curr_xyz: np.ndarray,
    prev_valid: np.ndarray,
    curr_valid: np.ndarray,
    prev_conf: np.ndarray,
    curr_conf: np.ndarray,
    max_anchors: int,
    robust_trim_percentile: float,
    seed: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    valid = (
        np.asarray(prev_valid, dtype=bool)
        & np.asarray(curr_valid, dtype=bool)
        & np.isfinite(prev_xyz).all(axis=-1)
        & np.isfinite(curr_xyz).all(axis=-1)
    )
    candidate_count = int(np.count_nonzero(valid))
    row: dict[str, Any] = {"candidate_count": candidate_count}
    if candidate_count < 4:
        row["pass"] = False
        row["reason"] = "not_enough_overlap_points"
        return None, row
    src = np.asarray(curr_xyz[valid], dtype=np.float32)
    dst = np.asarray(prev_xyz[valid], dtype=np.float32)
    score = np.minimum(np.asarray(prev_conf[valid], dtype=np.float32), np.asarray(curr_conf[valid], dtype=np.float32))
    if src.shape[0] > int(max_anchors) > 0:
        rng = np.random.default_rng(int(seed))
        if np.isfinite(score).any():
            order = np.argsort(np.nan_to_num(score, nan=-np.inf))
            top = order[-int(max_anchors) :]
            if top.size < int(max_anchors):
                fill = rng.choice(src.shape[0], size=int(max_anchors) - top.size, replace=False)
                keep = np.unique(np.concatenate([top, fill]))
            else:
                keep = top
        else:
            keep = rng.choice(src.shape[0], size=int(max_anchors), replace=False)
        src_fit = src[keep]
        dst_fit = dst[keep]
    else:
        src_fit = src
        dst_fit = dst
    fit = fit_transform(src_fit, dst_fit, robust_trim_percentile=float(robust_trim_percentile))
    if fit is None:
        row["pass"] = False
        row["reason"] = "fit_transform_failed"
        row["fit_anchor_count"] = int(src_fit.shape[0])
        return None, row
    aligned = apply_sim3_to_xyz(src_fit, transform=fit)
    residual = np.linalg.norm(aligned.astype(np.float64) - dst_fit.astype(np.float64), axis=1)
    row.update(
        {
            "pass": True,
            "fit_anchor_count": int(src_fit.shape[0]),
            "robust_trim_percentile": float(robust_trim_percentile),
            "robust_kept_anchors": int(fit.get("robust_kept_anchors", src_fit.shape[0])),
            "scale": float(fit["scale"]),
            "rotation_det": float(fit.get("rotation_det", np.linalg.det(np.asarray(fit["rotation"], dtype=np.float64)))),
            "translation_norm_m": float(np.linalg.norm(np.asarray(fit["translation"], dtype=np.float64))),
            "residual_m": _residual_stats(residual),
        }
    )
    return fit, row


def _stitch_dense_chunks(
    *,
    chunk_records: list[dict[str, Any]],
    frame_ids: np.ndarray,
    point_count: int,
    max_overlap_anchors: int,
    robust_trim_percentile: float,
    seed: int,
    camera_frame_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    frame_to_index = {int(frame_id): idx for idx, frame_id in enumerate(np.asarray(frame_ids, dtype=np.int64).tolist())}
    out_xyz = np.full((len(frame_ids), point_count, 3), np.nan, dtype=np.float32)
    out_valid = np.zeros((len(frame_ids), point_count), dtype=bool)
    out_visibility = np.zeros((len(frame_ids), point_count), dtype=np.float32)
    out_confidence = np.full((len(frame_ids), point_count), np.nan, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    weak_count = 0

    for chunk_idx, record in enumerate(chunk_records):
        chunk_frame_ids = np.asarray(record["frame_ids"], dtype=np.int64)
        xyz = np.asarray(record["xyz"], dtype=np.float32)
        valid = np.asarray(record["valid"], dtype=bool)
        visibility = np.asarray(record["visibility"], dtype=np.float32)
        confidence = np.asarray(record["confidence"], dtype=np.float32)
        transform: dict[str, Any] = {
            "scale": 1.0,
            "rotation": np.eye(3, dtype=np.float64),
            "translation": np.zeros((3,), dtype=np.float64),
        }
        row: dict[str, Any] = {
            "chunk_index": int(chunk_idx),
            "frame_id_min": int(chunk_frame_ids.min()),
            "frame_id_max": int(chunk_frame_ids.max()),
            "frame_count": int(chunk_frame_ids.shape[0]),
        }
        if chunk_idx > 0:
            overlap_global = [int(fid) for fid in chunk_frame_ids.tolist() if int(fid) in frame_to_index and np.any(out_valid[frame_to_index[int(fid)]])]
            if overlap_global:
                row["overlap_frame_ids"] = overlap_global
                row["overlap_frame_count"] = int(len(overlap_global))
                if str(camera_frame_mode) == "target_local":
                    row.update(
                        {
                            "pass": True,
                            "reason": "target_local_duplicate_confidence_fusion_no_sim3",
                            "scale": 1.0,
                            "sim3_applied": False,
                        }
                    )
                else:
                    curr_local = [int(np.flatnonzero(chunk_frame_ids == fid)[0]) for fid in overlap_global]
                    prev_global = [frame_to_index[fid] for fid in overlap_global]
                    fit, fit_row = _fit_overlap_sim3(
                        prev_xyz=out_xyz[prev_global],
                        curr_xyz=xyz[curr_local],
                        prev_valid=out_valid[prev_global],
                        curr_valid=valid[curr_local],
                        prev_conf=out_confidence[prev_global],
                        curr_conf=confidence[curr_local],
                        max_anchors=int(max_overlap_anchors),
                        robust_trim_percentile=float(robust_trim_percentile),
                        seed=int(seed) + chunk_idx,
                    )
                    row.update(fit_row)
                    if fit is not None:
                        transform = fit
                        xyz = apply_sim3_to_xyz(xyz, transform=transform).astype(np.float32)
                    else:
                        weak_count += 1
                        row["fallback"] = "identity_current_chunk_left_in_own_ref0"
            else:
                weak_count += 1
                row.update({"pass": False, "reason": "no_overlap_with_existing_global", "overlap_frame_count": 0})
        else:
            row.update({"pass": True, "reason": "first_chunk_identity", "overlap_frame_count": 0, "scale": 1.0})

        for local_idx, frame_id in enumerate(chunk_frame_ids.tolist()):
            global_idx = frame_to_index[int(frame_id)]
            current_ok = valid[local_idx] & np.isfinite(xyz[local_idx]).all(axis=-1)
            existing_ok = out_valid[global_idx]
            current_conf = confidence[local_idx]
            existing_conf = out_confidence[global_idx]
            better = (~existing_ok & current_ok) | (
                existing_ok
                & current_ok
                & np.isfinite(current_conf)
                & (~np.isfinite(existing_conf) | (current_conf >= existing_conf))
            )
            if not np.any(better):
                continue
            out_xyz[global_idx, better] = xyz[local_idx, better]
            out_valid[global_idx, better] = current_ok[better]
            out_visibility[global_idx, better] = visibility[local_idx, better]
            out_confidence[global_idx, better] = confidence[local_idx, better]
        rows.append(row)

    diagnostics = {
        "mode": "fixed_stride5_chunk32_overlap3_dense_point_cloud_self_stitch" if str(camera_frame_mode) == "ref0" else "fixed_stride5_chunk32_overlap3_target_local_duplicate_confidence_fusion",
        "camera_frame_mode": str(camera_frame_mode),
        "sim3_self_stitch_applied": bool(str(camera_frame_mode) == "ref0"),
        "chunk_count": int(len(chunk_records)),
        "weak_alignment_chunk_count": int(weak_count),
        "all_pairs_pass": bool(weak_count == 0),
        "rows": rows,
    }
    return out_xyz, out_valid, out_visibility, out_confidence, rows, diagnostics


def _fit_dense_to_scannet(
    *,
    stream: ScanNetStream,
    frame_ids: np.ndarray,
    xyz_ref: np.ndarray,
    valid: np.ndarray,
    uv_px: np.ndarray,
    rgb_width: int,
    rgb_height: int,
    max_anchors: int,
    robust_trim_percentile: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    anchor_candidates = 0
    depth_hits = 0
    for frame_index, frame_id in enumerate(np.asarray(frame_ids, dtype=np.int64).tolist()):
        ok = np.asarray(valid[frame_index], dtype=bool) & np.isfinite(xyz_ref[frame_index]).all(axis=1)
        anchor_candidates += int(np.count_nonzero(ok))
        if not np.any(ok):
            continue
        depth = stream.load_depth(int(frame_id))
        depth_h, depth_w = int(depth.shape[0]), int(depth.shape[1])
        uv_depth_px = np.asarray(uv_px[ok], dtype=np.float32).copy()
        uv_depth_px[:, 0] *= float(max(depth_w - 1, 1)) / float(max(int(rgb_width) - 1, 1))
        uv_depth_px[:, 1] *= float(max(depth_h - 1, 1)) / float(max(int(rgb_height) - 1, 1))
        world, world_ok = backproject_xy_world(stream, int(frame_id), uv_depth_px)
        if not np.any(world_ok):
            continue
        source_parts.append(np.asarray(xyz_ref[frame_index][ok][world_ok], dtype=np.float32))
        target_parts.append(np.asarray(world[world_ok], dtype=np.float32))
        depth_hits += int(np.count_nonzero(world_ok))
    if not source_parts:
        raise RuntimeError("no dense D4RT->ScanNet depth/pose anchors available")
    source = np.concatenate(source_parts, axis=0).astype(np.float32)
    target = np.concatenate(target_parts, axis=0).astype(np.float32)
    if int(max_anchors) > 0 and source.shape[0] > int(max_anchors):
        keep = np.linspace(0, source.shape[0] - 1, num=int(max_anchors), dtype=np.int64)
    else:
        keep = np.arange(source.shape[0], dtype=np.int64)
    fit = fit_transform(source[keep], target[keep], robust_trim_percentile=float(robust_trim_percentile))
    if fit is None:
        raise RuntimeError("dense D4RT->ScanNet depth/pose Sim3 fit returned None")
    aligned_fit = apply_sim3_to_xyz(source[keep], transform=fit)
    residual = np.linalg.norm(aligned_fit.astype(np.float64) - target[keep].astype(np.float64), axis=1)
    info = {
        "alignment_type": "diagnostic_scannet_depth_pose_sim3",
        "alignment_source": "ScanNet depth + pose backprojection at dense D4RT query UV after RGB-pixel to depth-pixel scaling",
        "uses_rgbd_pose_for_alignment": True,
        "uses_gt_mesh_for_alignment": False,
        "rgb_hw": [int(rgb_height), int(rgb_width)],
        "anchor_candidates": int(anchor_candidates),
        "depth_pose_anchor_count": int(source.shape[0]),
        "depth_pose_hit_count": int(depth_hits),
        "fit_anchor_count": int(keep.shape[0]),
        "robust_trim_percentile": float(robust_trim_percentile),
        "robust_kept_anchors": int(fit.get("robust_kept_anchors", keep.shape[0])),
        "scale": float(fit["scale"]),
        "rotation_det": float(fit.get("rotation_det", np.linalg.det(np.asarray(fit["rotation"], dtype=np.float64)))),
        "translation_norm_m": float(np.linalg.norm(np.asarray(fit["translation"], dtype=np.float64))),
        "fit_anchor_residual_m": _residual_stats(residual),
    }
    return fit, info


def _backproject_xy_camera(stream: ScanNetStream, frame_id: int, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float32)
    camera = np.full((xy.shape[0], 3), np.nan, dtype=np.float32)
    valid = np.zeros((xy.shape[0],), dtype=bool)
    if xy.size == 0:
        return camera, valid
    depth = stream.load_depth(int(frame_id))
    intrinsics = stream.load_intrinsics()
    h, w = depth.shape[:2]
    x = np.rint(xy[:, 0]).astype(np.int64)
    y = np.rint(xy[:, 1]).astype(np.int64)
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not np.any(in_bounds):
        return camera, valid
    z = depth[y[in_bounds], x[in_bounds]].astype(np.float32)
    depth_valid = np.isfinite(z) & (z > 0.0)
    source_indices = np.flatnonzero(in_bounds)[depth_valid]
    if source_indices.size == 0:
        return camera, valid
    x_f = x[source_indices].astype(np.float32)
    y_f = y[source_indices].astype(np.float32)
    z_f = z[depth_valid]
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    pts = np.stack([(x_f - cx) * z_f / fx, (y_f - cy) * z_f / fy, z_f], axis=1).astype(np.float32)
    finite = np.isfinite(pts).all(axis=1)
    valid[source_indices[finite]] = True
    camera[source_indices[finite]] = pts[finite]
    return camera, valid


def _fit_dense_local_to_scannet_camera(
    *,
    stream: ScanNetStream,
    frame_ids: np.ndarray,
    xyz_local: np.ndarray,
    valid: np.ndarray,
    uv_px: np.ndarray,
    rgb_width: int,
    rgb_height: int,
    max_anchors: int,
    robust_trim_percentile: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    anchor_candidates = 0
    depth_hits = 0
    for frame_index, frame_id in enumerate(np.asarray(frame_ids, dtype=np.int64).tolist()):
        ok = np.asarray(valid[frame_index], dtype=bool) & np.isfinite(xyz_local[frame_index]).all(axis=1)
        anchor_candidates += int(np.count_nonzero(ok))
        if not np.any(ok):
            continue
        depth = stream.load_depth(int(frame_id))
        depth_h, depth_w = int(depth.shape[0]), int(depth.shape[1])
        uv_depth_px = np.asarray(uv_px[ok], dtype=np.float32).copy()
        uv_depth_px[:, 0] *= float(max(depth_w - 1, 1)) / float(max(int(rgb_width) - 1, 1))
        uv_depth_px[:, 1] *= float(max(depth_h - 1, 1)) / float(max(int(rgb_height) - 1, 1))
        camera, camera_ok = _backproject_xy_camera(stream, int(frame_id), uv_depth_px)
        if not np.any(camera_ok):
            continue
        source_parts.append(np.asarray(xyz_local[frame_index][ok][camera_ok], dtype=np.float32))
        target_parts.append(np.asarray(camera[camera_ok], dtype=np.float32))
        depth_hits += int(np.count_nonzero(camera_ok))
    if not source_parts:
        raise RuntimeError("no dense D4RT-local->ScanNet-camera depth anchors available")
    source = np.concatenate(source_parts, axis=0).astype(np.float32)
    target = np.concatenate(target_parts, axis=0).astype(np.float32)
    if int(max_anchors) > 0 and source.shape[0] > int(max_anchors):
        keep = np.linspace(0, source.shape[0] - 1, num=int(max_anchors), dtype=np.int64)
    else:
        keep = np.arange(source.shape[0], dtype=np.int64)
    fit = fit_transform(source[keep], target[keep], robust_trim_percentile=float(robust_trim_percentile))
    if fit is None:
        raise RuntimeError("dense D4RT-local->ScanNet-camera Sim3 fit returned None")
    aligned_fit = apply_sim3_to_xyz(source[keep], transform=fit)
    residual = np.linalg.norm(aligned_fit.astype(np.float64) - target[keep].astype(np.float64), axis=1)
    info = {
        "alignment_type": "diagnostic_scannet_camera_depth_sim3_then_scannet_pose",
        "alignment_source": "ScanNet depth camera backprojection at dense D4RT query UV, then ScanNet pose maps camera points to world",
        "uses_rgbd_pose_for_alignment": True,
        "uses_gt_mesh_for_alignment": False,
        "rgb_hw": [int(rgb_height), int(rgb_width)],
        "anchor_candidates": int(anchor_candidates),
        "depth_pose_anchor_count": int(source.shape[0]),
        "depth_pose_hit_count": int(depth_hits),
        "fit_anchor_count": int(keep.shape[0]),
        "robust_trim_percentile": float(robust_trim_percentile),
        "robust_kept_anchors": int(fit.get("robust_kept_anchors", keep.shape[0])),
        "scale": float(fit["scale"]),
        "rotation_det": float(fit.get("rotation_det", np.linalg.det(np.asarray(fit["rotation"], dtype=np.float64)))),
        "translation_norm_m": float(np.linalg.norm(np.asarray(fit["translation"], dtype=np.float64))),
        "fit_anchor_residual_m": _residual_stats(residual),
    }
    return fit, info


def _target_local_to_world_flat(
    *,
    stream: ScanNetStream,
    frame_ids: np.ndarray,
    xyz_local: np.ndarray,
    valid: np.ndarray,
    fit: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    world_parts: list[np.ndarray] = []
    flat_valid_parts: list[np.ndarray] = []
    for frame_index, frame_id in enumerate(np.asarray(frame_ids, dtype=np.int64).tolist()):
        ok = np.asarray(valid[frame_index], dtype=bool) & np.isfinite(xyz_local[frame_index]).all(axis=1)
        flat_valid_parts.append(ok)
        if not np.any(ok):
            continue
        pose = stream.load_pose(int(frame_id))
        local_fit = apply_sim3_to_xyz(np.asarray(xyz_local[frame_index][ok], dtype=np.float32), transform=fit)
        hom = np.concatenate([local_fit.astype(np.float32), np.ones((local_fit.shape[0], 1), dtype=np.float32)], axis=1)
        world = (pose @ hom.T).T[:, :3].astype(np.float32)
        world_parts.append(world)
    flat_valid = np.concatenate(flat_valid_parts, axis=0) if flat_valid_parts else np.zeros((0,), dtype=bool)
    flat_world = np.concatenate(world_parts, axis=0).astype(np.float32) if world_parts else np.empty((0, 3), dtype=np.float32)
    return flat_world, flat_valid


def _load_eval_gt(base_summary: dict[str, Any], manifest: Any, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    gt_ply = Path(base_summary.get("gt", {}).get("gt_ply", "")) if base_summary.get("gt", {}).get("gt_ply") else (
        STREAM3D_ROOT / "data" / "scannet" / "processed" / args.scene_id / f"{args.scene_id}_vh_clean_2.ply"
    )
    gt_points_full, gt_colors_full = _read_gt_point_cloud(gt_ply)
    gt_mode = str(base_summary.get("gt", {}).get("gt_filter", args.gt_filter))
    if gt_mode == "input_visible":
        visibility = base_summary.get("gt", {}).get("visibility_filter", {})
        scene_root = STREAM3D_ROOT / "data" / "scannet" / "processed" / args.scene_id
        mask, info = _filter_gt_to_input_visible(
            gt_points=gt_points_full,
            manifest=manifest,
            scene_root=scene_root,
            depth_scale=float(visibility.get("depth_scale", args.scannet_depth_scale)),
            depth_abs_tolerance=float(visibility.get("depth_abs_tolerance_m", args.gt_visible_depth_abs_tolerance)),
            depth_rel_tolerance=float(visibility.get("depth_rel_tolerance", args.gt_visible_depth_rel_tolerance)),
            min_observations=int(visibility.get("min_observations", args.gt_visible_min_observations)),
            batch_size=int(args.gt_visible_batch_size),
        )
        return gt_points_full[mask], gt_colors_full[mask], info
    return gt_points_full, gt_colors_full, {"mode": "full_gt_no_input_visibility_filter"}


def _metric_row(metrics: dict[str, Any], *, rows: int, cols: int, camera_frame_mode: str) -> dict[str, Any]:
    if str(camera_frame_mode) == "target_local":
        variant_key = f"d4rt_dense{int(rows)}x{int(cols)}_target_local_pose_world"
        display_name = f"D4RT dense {int(rows)}x{int(cols)} target-local pose-world"
        transform = "target_local_camera_sim3_then_scannet_pose"
    else:
        variant_key = f"d4rt_dense{int(rows)}x{int(cols)}_self_stitched"
        display_name = f"D4RT dense {int(rows)}x{int(cols)} self-stitched"
        transform = "dense_point_cloud_overlap_self_stitch_then_depth_pose_sim3"
    row = {
        "variant_key": variant_key,
        "display_name": display_name,
        "model": "OpenD4RT_32CLIP_9Dataset_NoAUG",
        "transform": transform,
        "chamfer_l2_mean_m": metrics["chamfer_l2_mean_m"],
        "chamfer_l2_squared_mean_m2": metrics["chamfer_l2_squared_mean_m2"],
        "accuracy_mean_m": metrics["accuracy_da3_to_gt_m"]["mean"],
        "accuracy_p50_m": metrics["accuracy_da3_to_gt_m"]["p50"],
        "accuracy_p90_m": metrics["accuracy_da3_to_gt_m"]["p90"],
        "accuracy_p95_m": metrics["accuracy_da3_to_gt_m"]["p95"],
        "completeness_mean_m": metrics["completeness_gt_to_da3_m"]["mean"],
        "completeness_p50_m": metrics["completeness_gt_to_da3_m"]["p50"],
        "completeness_p90_m": metrics["completeness_gt_to_da3_m"]["p90"],
        "completeness_p95_m": metrics["completeness_gt_to_da3_m"]["p95"],
    }
    for key, value in metrics["fscore"].items():
        prefix = key.replace(".", "p")
        row[f"{prefix}_precision"] = value["precision"]
        row[f"{prefix}_recall"] = value["recall"]
        row[f"{prefix}_fscore"] = value["fscore"]
    return row


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base_root = Path(args.base_output_root)
    base_summary_path = Path(args.base_summary_json) if args.base_summary_json else base_root / "geometry_quality_summary.json"
    base_summary = _load_json(base_summary_path)
    base_npz_path = Path(args.base_viewer_npz) if args.base_viewer_npz else Path(base_summary["outputs"]["viewer_npz"])
    base_csv_path = Path(base_summary["outputs"]["metrics_csv"])
    manifest = _load_da3_manifest(Path(args.da3_manifest))
    manifest = manifest.sort_values("da3_frame_index").reset_index(drop=True)
    frame_ids = manifest["frame_id"].to_numpy(dtype=np.int64)
    if frame_ids.shape[0] <= 0:
        raise RuntimeError("empty DA3 frame manifest")

    stream = ScanNetStream(
        seq_name=args.scene_id,
        backbone="Cropformer",
        root=STREAM3D_ROOT / "data" / "scannet" / "processed",
    )
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))
    first_rgb = stream.load_rgb(int(frame_ids[0]))
    image_h, image_w = int(first_rgb.shape[0]), int(first_rgb.shape[1])
    uv_px = _build_uv_grid(image_w, image_h, int(args.rows), int(args.cols), float(args.grid_margin_ratio))
    uv_norm = uv_px.copy()
    uv_norm[:, 0] /= float(max(image_w - 1, 1))
    uv_norm[:, 1] /= float(max(image_h - 1, 1))

    adapter = D4RTAdapter(
        d4rt_root=Path(args.d4rt_root),
        model_config=Path(args.d4rt_config),
        ckpt_path=Path(args.d4rt_ckpt),
        device=str(args.device),
    )
    windows = _make_overlap_windows(frame_ids.shape[0], int(args.chunk_size), int(args.overlap))
    chunk_records: list[dict[str, Any]] = []
    for chunk_index, (start, end) in enumerate(windows):
        chunk_frame_ids = frame_ids[start:end]
        video_rgb = np.stack([stream.load_rgb(int(fid)) for fid in chunk_frame_ids.tolist()], axis=0).astype(np.uint8)
        chunk_t0 = time.time()
        xyz, valid, visibility, confidence, infer_info = _decode_dense_chunk(
            adapter=adapter,
            video_rgb_uint8=video_rgb,
            query_uv_norm=uv_norm,
            query_chunk_size=int(args.query_chunk_size),
            aspect_source=str(args.aspect_source),
            camera_frame_mode=str(args.camera_frame_mode),
        )
        colors = _sample_rgb_sequence(video_rgb, uv_px)
        keep = (
            valid
            & (visibility >= float(args.min_visibility))
            & (confidence >= float(args.min_confidence))
            & np.isfinite(xyz).all(axis=-1)
        )
        chunk_records.append(
            {
                "chunk_index": int(chunk_index),
                "frame_ids": chunk_frame_ids.astype(np.int64),
                "xyz": xyz.astype(np.float32),
                "valid": keep.astype(bool),
                "visibility": visibility.astype(np.float32),
                "confidence": confidence.astype(np.float32),
                "colors": colors.astype(np.uint8),
                "infer_info": infer_info,
            }
        )
        print(
            json.dumps(
                {
                    "event": "dense_d4rt_chunk_done",
                    "chunk_index": int(chunk_index),
                    "frame_id_min": int(chunk_frame_ids.min()),
                    "frame_id_max": int(chunk_frame_ids.max()),
                    "frame_count": int(chunk_frame_ids.shape[0]),
                    "valid_points": int(np.count_nonzero(keep)),
                    "seconds_chunk_total": float(time.time() - chunk_t0),
                    **infer_info,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    point_count = int(uv_px.shape[0])
    xyz_ref, valid_ref, visibility_ref, confidence_ref, stitch_rows, stitch_diag = _stitch_dense_chunks(
        chunk_records=chunk_records,
        frame_ids=frame_ids,
        point_count=point_count,
        max_overlap_anchors=int(args.max_overlap_anchors),
        robust_trim_percentile=float(args.self_stitch_trim_percentile),
        seed=int(args.seed),
        camera_frame_mode=str(args.camera_frame_mode),
    )
    color_global = np.zeros((frame_ids.shape[0], point_count, 3), dtype=np.uint8)
    frame_to_index = {int(fid): idx for idx, fid in enumerate(frame_ids.tolist())}
    for record in chunk_records:
        chunk_frame_ids = np.asarray(record["frame_ids"], dtype=np.int64)
        colors = np.asarray(record["colors"], dtype=np.uint8)
        for local_idx, frame_id in enumerate(chunk_frame_ids.tolist()):
            global_idx = frame_to_index[int(frame_id)]
            color_global[global_idx] = colors[local_idx]

    if str(args.camera_frame_mode) == "target_local":
        fit, alignment_info = _fit_dense_local_to_scannet_camera(
            stream=stream,
            frame_ids=frame_ids,
            xyz_local=xyz_ref,
            valid=valid_ref,
            uv_px=uv_px,
            rgb_width=int(image_w),
            rgb_height=int(image_h),
            max_anchors=int(args.max_sim3_anchors),
            robust_trim_percentile=float(args.robust_trim_percentile),
        )
        flat_points_aligned, flat_valid = _target_local_to_world_flat(
            stream=stream,
            frame_ids=frame_ids,
            xyz_local=xyz_ref,
            valid=valid_ref,
            fit=fit,
        )
    else:
        fit, alignment_info = _fit_dense_to_scannet(
            stream=stream,
            frame_ids=frame_ids,
            xyz_ref=xyz_ref,
            valid=valid_ref,
            uv_px=uv_px,
            rgb_width=int(image_w),
            rgb_height=int(image_h),
            max_anchors=int(args.max_sim3_anchors),
            robust_trim_percentile=float(args.robust_trim_percentile),
        )
        flat_valid = valid_ref.reshape(-1) & np.isfinite(xyz_ref.reshape(-1, 3)).all(axis=1)
        flat_points_ref = xyz_ref.reshape(-1, 3)[flat_valid]
        flat_points_aligned = apply_sim3_to_xyz(flat_points_ref, transform=fit).astype(np.float32)
    flat_colors = color_global.reshape(-1, 3)[flat_valid]
    flat_frame_ids = np.repeat(frame_ids.astype(np.int32), point_count)[flat_valid]
    flat_point_indices = np.tile(np.arange(point_count, dtype=np.int32), frame_ids.shape[0])[flat_valid]

    gt_points, gt_colors, gt_filter_info = _load_eval_gt(base_summary, manifest, args)
    gt_tree = cKDTree(gt_points.astype(np.float64))
    metric_idx = _sample_indices(flat_points_aligned.shape[0], int(args.max_metric_points), int(args.seed) + 707) if int(args.max_metric_points) > 0 else np.arange(flat_points_aligned.shape[0], dtype=np.int64)
    metrics = _chamfer_metrics(
        source_aligned=flat_points_aligned[metric_idx],
        target_gt=gt_points,
        target_gt_tree=gt_tree,
        thresholds=[float(value) for value in args.fscore_thresholds],
    )
    metrics["source_total_point_count_before_metric_sampling"] = int(flat_points_aligned.shape[0])
    metrics["source_metric_point_count"] = int(metric_idx.shape[0])

    with np.load(base_npz_path) as payload:
        npz_payload = {key: np.asarray(payload[key]) for key in payload.files}
    viewer_idx = _sample_indices(flat_points_aligned.shape[0], int(args.viewer_d4rt_sample_count), int(args.seed) + 808)
    npz_payload["d4rt_points"] = flat_points_aligned[viewer_idx].astype(np.float32)
    npz_payload["d4rt_colors"] = flat_colors[viewer_idx].astype(np.uint8)
    npz_payload["d4rt_frame_ids"] = flat_frame_ids[viewer_idx].astype(np.int32)
    npz_payload["d4rt_point_indices"] = flat_point_indices[viewer_idx].astype(np.int32)
    npz_path = output_root / f"{args.scene_id}_da3_d4rt_dense_geometry_viewer_points.npz"
    np.savez_compressed(npz_path, **npz_payload)

    cache_mode = "target_local_pose_world" if str(args.camera_frame_mode) == "target_local" else "self_stitched"
    dense_cache_path = output_root / f"{args.scene_id}_d4rt_dense{int(args.rows)}x{int(args.cols)}_{cache_mode}_points.npz"
    np.savez_compressed(
        dense_cache_path,
        frame_ids=frame_ids.astype(np.int64),
        uv_px=uv_px.astype(np.float32),
        xyz_ref0_self_stitched=xyz_ref.astype(np.float32),
        xyz_source=xyz_ref.astype(np.float32),
        camera_frame_mode=np.asarray([str(args.camera_frame_mode)]),
        valid=valid_ref.astype(bool),
        visibility=visibility_ref.astype(np.float32),
        confidence=confidence_ref.astype(np.float32),
        colors=color_global.astype(np.uint8),
        sim3_to_scannet_scale=np.asarray([float(fit["scale"])], dtype=np.float64),
        sim3_to_scannet_rotation=np.asarray(fit["rotation"], dtype=np.float64),
        sim3_to_scannet_translation=np.asarray(fit["translation"], dtype=np.float64),
    )

    csv_rows = _read_csv(base_csv_path)
    d4rt_row = _metric_row(metrics, rows=int(args.rows), cols=int(args.cols), camera_frame_mode=str(args.camera_frame_mode))
    csv_rows.append(d4rt_row)
    csv_path = output_root / "geometry_quality_metrics_with_d4rt_dense.csv"
    _write_csv(csv_path, csv_rows)

    d4rt_geometry = {
        "display_name": str(d4rt_row["display_name"]),
        "diagnostic_only": True,
        "method_result_allowed": False,
        "source": {
            "d4rt_root": str(Path(args.d4rt_root)),
            "d4rt_config": str(Path(args.d4rt_config)),
            "d4rt_ckpt": str(Path(args.d4rt_ckpt)),
            "paper_contract": {
                "query_interface": "(u,v,t_src,t_tgt,t_cam)->xyz_3d",
                "dense_geometry_query": "per-frame point cloud uses dense uv grid with t_src=t_tgt=frame and t_cam=chunk_ref0 before overlap self-stitch"
                if str(args.camera_frame_mode) == "ref0"
                else "per-frame point cloud uses dense uv grid with t_src=t_tgt=t_cam=frame, duplicate overlap confidence fusion, local-camera Sim3, then ScanNet pose to world",
                "not_sparse_carrier_tracks": True,
            },
            "sampling_contract": {
                "rows": int(args.rows),
                "cols": int(args.cols),
                "queries_per_frame": int(point_count),
                "min_required_rows": 120,
                "min_required_cols": 160,
                "meets_requested_density": bool(int(args.rows) >= 120 and int(args.cols) >= 160),
                "fresh_samples_per_image": True,
                "chunk_size": int(args.chunk_size),
                "overlap": int(args.overlap),
                "stride_frame_id": int(frame_ids[1] - frame_ids[0]) if frame_ids.shape[0] > 1 else None,
                "grid_margin_ratio": float(args.grid_margin_ratio),
                "aspect_source": str(args.aspect_source),
                "camera_frame_mode": str(args.camera_frame_mode),
            },
        },
        "input_scope": {
            "da3_manifest": str(args.da3_manifest),
            "manifest_frame_count": int(manifest.shape[0]),
            "frame_id_min": int(frame_ids.min()),
            "frame_id_max": int(frame_ids.max()),
            "only_da3_input_frames_retained_for_viewer_and_metrics": True,
        },
        "filters": {
            "min_visibility_prob": float(args.min_visibility),
            "min_confidence_prob": float(args.min_confidence),
            "finite_xyz_required": True,
        },
        "observation_counts": {
            "dense_query_point_count_per_frame": int(point_count),
            "raw_query_observation_count": int(frame_ids.shape[0] * point_count),
            "valid_observation_count_after_filters": int(np.count_nonzero(valid_ref)),
            "viewer_d4rt_point_count": int(viewer_idx.shape[0]),
            "metric_d4rt_point_count": int(metric_idx.shape[0]),
        },
        "chunk_inference": [record["infer_info"] | {"chunk_index": int(record["chunk_index"])} for record in chunk_records],
        "overlap_self_stitch": stitch_diag,
        "alignment_to_scannet": alignment_info,
        "gt_filter_for_metrics": gt_filter_info,
        "geometry_metrics_against_input_visible_gt": metrics,
        "csv_row": d4rt_row,
    }

    summary = dict(base_summary)
    summary["outputs"] = dict(base_summary["outputs"])
    summary["outputs"].update(
        {
            "base_summary_json": str(base_summary_path),
            "base_viewer_npz": str(base_npz_path),
            "summary_json": str(output_root / "geometry_quality_summary_with_d4rt_dense.json"),
            "metrics_csv": str(csv_path),
            "viewer_npz": str(npz_path),
            "dense_d4rt_cache_npz": str(dense_cache_path),
        }
    )
    summary["metric_note"] = (
        str(base_summary.get("metric_note", ""))
        + " D4RT dense point-cloud geometry is added as a diagnostic comparison layer with >=120x160 per-frame query grid, "
        + (
            "fixed chunk=32 overlap=3 ref0 self-stitch, and diagnostic ScanNet depth/pose Sim3 alignment before Chamfer/F-score."
            if str(args.camera_frame_mode) == "ref0"
            else "fixed chunk=32 overlap=3 target-local duplicate fusion, diagnostic local-camera Sim3, and ScanNet pose-to-world before Chamfer/F-score."
        )
    )
    summary["d4rt_geometry"] = d4rt_geometry
    summary["csv_rows"] = csv_rows
    summary_path = output_root / "geometry_quality_summary_with_d4rt_dense.json"
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary_json": str(summary_path),
                "metrics_csv": str(csv_path),
                "viewer_npz": str(npz_path),
                "dense_d4rt_cache_npz": str(dense_cache_path),
                "camera_frame_mode": str(args.camera_frame_mode),
                "d4rt_dense_queries_per_frame": int(point_count),
                "d4rt_dense_valid_observation_count": int(np.count_nonzero(valid_ref)),
                "d4rt_dense_metric_point_count": int(metric_idx.shape[0]),
                "d4rt_dense_chamfer_l2_mean_m": float(metrics["chamfer_l2_mean_m"]),
                "d4rt_dense_fscore_0p10m": float(metrics["fscore"]["0.10m"]["fscore"]),
                "self_stitch_all_pairs_pass": bool(stitch_diag["all_pairs_pass"]),
                "self_stitch_weak_alignment_chunk_count": int(stitch_diag["weak_alignment_chunk_count"]),
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense D4RT point-cloud geometry comparison for v98.1 Viser/metrics.")
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--base-output-root", default=str(DEFAULT_BASE_ROOT))
    parser.add_argument("--base-summary-json", default="")
    parser.add_argument("--base-viewer-npz", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--da3-manifest", default=str(STREAM3D_ROOT / "outputs" / "audit" / "v98_phase1_provider_contract" / "da3_streaming_d4rt32o3_scene0050_input119" / "frame_manifest_rows.csv"))
    parser.add_argument("--d4rt-root", default=str(DEFAULT_D4RT_ROOT))
    parser.add_argument("--d4rt-config", default=str(DEFAULT_D4RT_CONFIG))
    parser.add_argument("--d4rt-ckpt", default=str(DEFAULT_D4RT_CKPT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--cols", type=int, default=160)
    parser.add_argument("--grid-margin-ratio", type=float, default=0.02)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--aspect-source", choices=["model_input", "native_rgb"], default="model_input")
    parser.add_argument("--camera-frame-mode", choices=["ref0", "target_local"], default="ref0")
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-overlap-anchors", type=int, default=100000)
    parser.add_argument("--self-stitch-trim-percentile", type=float, default=90.0)
    parser.add_argument("--max-sim3-anchors", type=int, default=120000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--max-metric-points", type=int, default=0)
    parser.add_argument("--viewer-d4rt-sample-count", type=int, default=180000)
    parser.add_argument("--seed", type=int, default=9801098)
    parser.add_argument("--fscore-thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.50])
    parser.add_argument("--gt-filter", choices=["full", "input_visible"], default="input_visible")
    parser.add_argument("--scannet-depth-scale", type=float, default=1000.0)
    parser.add_argument("--gt-visible-depth-abs-tolerance", type=float, default=0.08)
    parser.add_argument("--gt-visible-depth-rel-tolerance", type=float, default=0.03)
    parser.add_argument("--gt-visible-min-observations", type=int, default=1)
    parser.add_argument("--gt-visible-batch-size", type=int, default=65536)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
