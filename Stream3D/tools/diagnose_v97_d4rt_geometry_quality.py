#!/usr/bin/env python3
"""Diagnostic-only D4RT geometry quality audit for v97 carrier batches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry_provider.common import backproject_xy_world, fit_transform
from stream4d.scannet_stream import ScanNetStream


PHASE_ID = "v97_phase8_d4rt_geometry_quality"
RUN_ID = "v97_phase8_d4rt_geometry_quality"
DEFAULT_PHASE2 = ROOT / "outputs/audit/v97_phase2_d4rt_micro_tracks_full_D3_gpu7_clamp002"
DEFAULT_OUT = ROOT / "outputs/audit/v97_phase8_d4rt_geometry_quality_D3"
DEFAULT_SCANNET = ROOT / "data/scannet/processed"


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def _norm_stats(prefix: str, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_p50": None,
            f"{prefix}_p90": None,
            f"{prefix}_p95": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def _apply_fit(points: np.ndarray, fit: dict[str, Any] | None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if fit is None:
        return np.full_like(points, np.nan, dtype=np.float32)
    scale = float(fit["scale"])
    rotation = np.asarray(fit["rotation"], dtype=np.float64)
    translation = np.asarray(fit["translation"], dtype=np.float64)
    return (scale * (points @ rotation.T) + translation).astype(np.float32)


def _stable_sample_indices(count: int, cap: int) -> np.ndarray:
    if cap <= 0 or count <= cap:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=int(cap), dtype=np.int64)


def _split_train_eval(count: int, holdout_mod: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(count, dtype=np.int64)
    if count < 8 or holdout_mod <= 1:
        return idx, idx
    holdout = (idx % int(holdout_mod)) == 0
    train = ~holdout
    if int(np.count_nonzero(train)) < 4 or int(np.count_nonzero(holdout)) < 4:
        return idx, idx
    return idx[train], idx[holdout]


def _fit_rigid_no_scale(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 4:
        raise ValueError("source and target must be Nx3 arrays with at least four points")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    x = src - mu_src
    y = dst - mu_dst
    u, _, vt = np.linalg.svd((x.T @ y) / float(src.shape[0]))
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1.0
        rot = vt.T @ u.T
    trans = mu_dst - rot @ mu_src
    residual = np.linalg.norm((src @ rot.T) + trans - dst, axis=1)
    return rot.astype(np.float64), trans.astype(np.float64), residual.astype(np.float64)


def _fit_scope(
    *,
    scope: str,
    scope_id: str,
    source: np.ndarray,
    target: np.ndarray,
    robust_trim_percentile: float,
    holdout_mod: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    train_idx, eval_idx = _split_train_eval(source.shape[0], holdout_mod)
    fit = fit_transform(source[train_idx], target[train_idx], robust_trim_percentile=robust_trim_percentile)
    if fit is None:
        row = {
            "schema_version": "stream4d_v97_phase8_sim3_fit_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "metric_level": "scene_alignment" if scope == "scene" else "single_window_metric",
            "scope": scope,
            "scope_id": scope_id,
            "anchor_count_total": int(source.shape[0]),
            "anchor_count_train": int(train_idx.shape[0]),
            "anchor_count_holdout": int(eval_idx.shape[0]),
            "fit_status": "insufficient_anchors",
            "coordinate_frame": "d4rt_xyz_ref_to_scannet_world",
            "alignment_type": "eval_only_sim3",
            "alignment_source": "scannet_depth_pose_backprojection",
            "scale_source": "eval_sim3_fit",
            "uses_gt_for_metric": True,
            "uses_gt_for_prediction": False,
            "uses_rgbd_pose_mesh_for_metric": True,
            "uses_rgbd_pose_for_diagnostic": True,
            "is_method_safe_metric": False,
            "is_diagnostic_metric": True,
        }
        return None, row
    train_res = np.linalg.norm(_apply_fit(source[train_idx], fit) - target[train_idx], axis=1)
    eval_res = np.linalg.norm(_apply_fit(source[eval_idx], fit) - target[eval_idx], axis=1)
    row = {
        "schema_version": "stream4d_v97_phase8_sim3_fit_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "metric_level": "scene_alignment" if scope == "scene" else "single_window_metric",
        "scope": scope,
        "scope_id": scope_id,
        "anchor_count_total": int(source.shape[0]),
        "anchor_count_train": int(train_idx.shape[0]),
        "anchor_count_holdout": int(eval_idx.shape[0]),
        "fit_status": "ok",
        "sim3_scale": float(fit["scale"]),
        "sim3_rotation_det": float(fit.get("rotation_det", np.linalg.det(np.asarray(fit["rotation"], dtype=np.float64)))),
        "sim3_translation_norm": float(np.linalg.norm(np.asarray(fit["translation"], dtype=np.float64))),
        "robust_trim_percentile": float(robust_trim_percentile),
        "robust_kept_anchors": int(fit.get("robust_kept_anchors", train_idx.shape[0])),
        "train_residual_mean_m": _mean(train_res),
        "train_residual_p50_m": _quantile(train_res, 50),
        "train_residual_p90_m": _quantile(train_res, 90),
        "train_residual_p95_m": _quantile(train_res, 95),
        "holdout_residual_mean_m": _mean(eval_res),
        "holdout_residual_p50_m": _quantile(eval_res, 50),
        "holdout_residual_p90_m": _quantile(eval_res, 90),
        "holdout_residual_p95_m": _quantile(eval_res, 95),
        "coordinate_frame": "d4rt_xyz_ref_to_scannet_world",
        "alignment_type": "eval_only_sim3",
        "alignment_source": "scannet_depth_pose_backprojection",
        "scale_source": "eval_sim3_fit",
        "uses_gt_for_metric": True,
        "uses_gt_for_prediction": False,
        "uses_rgbd_pose_mesh_for_metric": True,
        "uses_rgbd_pose_for_diagnostic": True,
        "is_method_safe_metric": False,
        "is_diagnostic_metric": True,
    }
    return fit, row


def _chamfer_stats(source_aligned: np.ndarray, target_world: np.ndarray) -> dict[str, Any]:
    source_aligned = np.asarray(source_aligned, dtype=np.float32)
    target_world = np.asarray(target_world, dtype=np.float32)
    finite = np.isfinite(source_aligned).all(axis=1) & np.isfinite(target_world).all(axis=1)
    source_aligned = source_aligned[finite]
    target_world = target_world[finite]
    if source_aligned.shape[0] < 2 or target_world.shape[0] < 2:
        return {
            "paired_residual_mean_m": None,
            "paired_residual_p50_m": None,
            "paired_residual_p90_m": None,
            "d4rt_to_depth_nn_mean_m": None,
            "d4rt_to_depth_nn_p90_m": None,
            "depth_to_d4rt_nn_mean_m": None,
            "depth_to_d4rt_nn_p90_m": None,
            "symmetric_chamfer_mean_m": None,
            "symmetric_chamfer_p90_mean_m": None,
        }
    paired = np.linalg.norm(source_aligned - target_world, axis=1)
    tree_t = cKDTree(target_world)
    tree_s = cKDTree(source_aligned)
    d_st, _ = tree_t.query(source_aligned, k=1)
    d_ts, _ = tree_s.query(target_world, k=1)
    return {
        "paired_residual_mean_m": _mean(paired),
        "paired_residual_p50_m": _quantile(paired, 50),
        "paired_residual_p90_m": _quantile(paired, 90),
        "d4rt_to_depth_nn_mean_m": _mean(d_st),
        "d4rt_to_depth_nn_p90_m": _quantile(d_st, 90),
        "depth_to_d4rt_nn_mean_m": _mean(d_ts),
        "depth_to_d4rt_nn_p90_m": _quantile(d_ts, 90),
        "symmetric_chamfer_mean_m": None if _mean(d_st) is None or _mean(d_ts) is None else 0.5 * float(_mean(d_st) + _mean(d_ts)),
        "symmetric_chamfer_p90_mean_m": None if _quantile(d_st, 90) is None or _quantile(d_ts, 90) is None else 0.5 * float(_quantile(d_st, 90) + _quantile(d_ts, 90)),
    }


def _load_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, list[int]]]:
    phase2_root = _project(args.phase2_root)
    batch_root = phase2_root / "carrier_batches" / args.decode_variant
    paths = sorted(batch_root.glob("*/*.npz"))
    if not paths:
        raise FileNotFoundError(f"No carrier batch npz files under {batch_root}")
    streams: dict[str, ScanNetStream] = {}
    frame_rows: list[dict[str, Any]] = []
    source_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    row_chunks: dict[str, list[int]] = defaultdict(list)
    total_query_slots = 0
    total_valid_uv = 0
    total_accepted = 0
    total_depth_hits = 0
    for batch_path in paths:
        scene = batch_path.parent.name
        window = batch_path.stem
        if scene not in streams:
            streams[scene] = ScanNetStream(seq_name=scene, backbone=args.backbone, root=_project(args.scannet_root))
            errors = streams[scene].validate(require_masks=False)
            if errors:
                raise RuntimeError("; ".join(errors))
        stream = streams[scene]
        with np.load(batch_path, allow_pickle=True) as data:
            frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
            uv = np.asarray(data["uv_pred"], dtype=np.float32)
            xyz = np.asarray(data["xyz_ref"], dtype=np.float32)
            xyz_local = np.asarray(data.get("xyz_local", np.full_like(xyz, np.nan)), dtype=np.float32)
            visibility = np.asarray(data["visibility_prob"], dtype=np.float32)
            confidence = np.asarray(data["confidence_prob"], dtype=np.float32)
            valid = np.asarray(data.get("valid", np.ones(uv.shape[:2], dtype=bool)), dtype=bool)
        for local_idx, frame_id in enumerate(frame_ids.tolist()):
            uv_frame = uv[local_idx]
            xyz_frame = xyz[local_idx]
            xyz_local_frame = xyz_local[local_idx]
            query_count = int(uv_frame.shape[0])
            uv_in01 = (
                np.isfinite(uv_frame).all(axis=1)
                & (uv_frame[:, 0] >= 0.0)
                & (uv_frame[:, 0] <= 1.0)
                & (uv_frame[:, 1] >= 0.0)
                & (uv_frame[:, 1] <= 1.0)
            )
            accepted = (
                valid[local_idx]
                & uv_in01
                & np.isfinite(xyz_frame).all(axis=1)
                & (visibility[local_idx] >= float(args.min_visibility))
                & (confidence[local_idx] >= float(args.min_confidence))
            )
            accepted_indices = np.flatnonzero(accepted)
            sample_local = accepted_indices[_stable_sample_indices(int(accepted_indices.shape[0]), int(args.max_points_per_frame))]
            total_query_slots += query_count
            total_valid_uv += int(np.count_nonzero(uv_in01))
            total_accepted += int(accepted_indices.shape[0])
            if sample_local.size == 0:
                frame_rows.append(
                    {
                        "schema_version": "stream4d_v97_phase8_frame_geometry_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "decode_variant": args.decode_variant,
                        "scene_id": scene,
                        "window_id": window,
                        "frame_id": int(frame_id),
                        "query_count": query_count,
                        "uv_in01_count": int(np.count_nonzero(uv_in01)),
                        "accepted_count": int(accepted_indices.shape[0]),
                        "sampled_count": 0,
                        "depth_hit_count": 0,
                        "depth_hit_rate_sampled": 0.0,
                        "metric_level": "single_window_metric",
                        "coordinate_frame": "d4rt_xyz_ref_and_xyz_local",
                        "alignment_type": "none_for_uv_reprojection;eval_only_sim3_for_world_chamfer",
                        "alignment_source": "scannet_depth_pose_backprojection",
                        "uses_gt_for_metric": True,
                        "uses_gt_for_prediction": False,
                        "uses_rgbd_pose_mesh_for_metric": True,
                        "uses_rgbd_pose_for_diagnostic": True,
                        "is_method_safe_metric": False,
                        "is_diagnostic_metric": True,
                    }
                )
                continue
            depth = stream.load_depth(int(frame_id))
            intrinsics = stream.load_intrinsics()
            h, w = depth.shape[:2]
            xy = np.stack(
                [
                    uv_frame[sample_local, 0] * float(max(w - 1, 1)),
                    uv_frame[sample_local, 1] * float(max(h - 1, 1)),
                ],
                axis=1,
            )
            world, world_ok = backproject_xy_world(stream, int(frame_id), xy)
            src = xyz_frame[sample_local][world_ok].astype(np.float32, copy=False)
            tgt = world[world_ok].astype(np.float32, copy=False)
            depth_hits = int(src.shape[0])
            total_depth_hits += depth_hits
            local_xyz = xyz_local_frame[sample_local].astype(np.float64, copy=False)
            local_z = local_xyz[:, 2]
            positive_local_z = np.isfinite(local_z) & (local_z > 1e-6) & np.isfinite(local_xyz).all(axis=1)
            reproj_err = np.empty((0,), dtype=np.float64)
            if np.any(positive_local_z):
                fx = float(intrinsics[0, 0])
                fy = float(intrinsics[1, 1])
                cx = float(intrinsics[0, 2])
                cy = float(intrinsics[1, 2])
                x_proj = local_xyz[positive_local_z, 0] * fx / local_z[positive_local_z] + cx
                y_proj = local_xyz[positive_local_z, 1] * fy / local_z[positive_local_z] + cy
                x_uv = xy[positive_local_z, 0]
                y_uv = xy[positive_local_z, 1]
                reproj_err = np.sqrt((x_proj - x_uv) ** 2 + (y_proj - y_uv) ** 2)
                reproj_err = reproj_err[np.isfinite(reproj_err)]
            point_start = sum(chunk.shape[0] for chunk in source_chunks)
            if depth_hits:
                source_chunks.append(src)
                target_chunks.append(tgt)
                point_indices = list(range(point_start, point_start + depth_hits))
                row_chunks[f"scene:{scene}"].extend(point_indices)
                row_chunks[f"window:{scene}:{window}"].extend(point_indices)
                frame_key = f"frame:{scene}:{window}:{int(frame_id)}"
                row_chunks[frame_key].extend(point_indices)
            frame_rows.append(
                {
                    "schema_version": "stream4d_v97_phase8_frame_geometry_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "decode_variant": args.decode_variant,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": int(frame_id),
                    "query_count": query_count,
                    "uv_in01_count": int(np.count_nonzero(uv_in01)),
                    "accepted_count": int(accepted_indices.shape[0]),
                    "sampled_count": int(sample_local.shape[0]),
                    "depth_hit_count": depth_hits,
                    "depth_hit_rate_sampled": float(depth_hits / max(1, int(sample_local.shape[0]))),
                    "local_xyz_positive_z_rate_sampled": float(np.count_nonzero(positive_local_z) / max(1, int(sample_local.shape[0]))),
                    "local_xyz_reproj_error_px_count": int(reproj_err.shape[0]),
                    "local_xyz_reproj_error_px_mean": _mean(reproj_err),
                    "local_xyz_reproj_error_px_p50": _quantile(reproj_err, 50),
                    "local_xyz_reproj_error_px_p90": _quantile(reproj_err, 90),
                    "metric_level": "single_window_metric",
                    "coordinate_frame": "d4rt_xyz_ref_and_xyz_local",
                    "alignment_type": "none_for_uv_reprojection;eval_only_sim3_for_world_chamfer",
                    "alignment_source": "scannet_depth_pose_backprojection",
                    "uses_gt_for_metric": True,
                    "uses_gt_for_prediction": False,
                    "uses_rgbd_pose_mesh_for_metric": True,
                    "uses_rgbd_pose_for_diagnostic": True,
                    "is_method_safe_metric": False,
                    "is_diagnostic_metric": True,
                }
            )
    source = np.concatenate(source_chunks, axis=0) if source_chunks else np.empty((0, 3), dtype=np.float32)
    target = np.concatenate(target_chunks, axis=0) if target_chunks else np.empty((0, 3), dtype=np.float32)
    totals = {
        "total_query_slots": [int(total_query_slots)],
        "total_valid_uv": [int(total_valid_uv)],
        "total_accepted": [int(total_accepted)],
        "total_depth_hits": [int(total_depth_hits)],
    }
    for key, value in totals.items():
        row_chunks[key] = value
    return frame_rows, source, target, row_chunks


def _window_sort_key(window_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(window_id) if ch.isdigit())
    return (int(digits) if digits else 0, str(window_id))


def _build_scale_pair_rows(fit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fit_rows:
        if row.get("scope") != "window" or row.get("fit_status") != "ok":
            continue
        scope_id = str(row.get("scope_id", ""))
        parts = scope_id.split(":", 1)
        if len(parts) != 2:
            continue
        scene, window = parts
        item = dict(row)
        item["scene_id"] = scene
        item["window_id"] = window
        by_scene[scene].append(item)
    out: list[dict[str, Any]] = []
    for scene, rows in sorted(by_scene.items()):
        rows = sorted(rows, key=lambda item: _window_sort_key(str(item["window_id"])))
        for prev, nxt in zip(rows[:-1], rows[1:]):
            prev_scale = float(prev["sim3_scale"])
            next_scale = float(nxt["sim3_scale"])
            ratio = next_scale / prev_scale if abs(prev_scale) > 1e-12 else float("nan")
            abs_log = abs(math.log(ratio)) if np.isfinite(ratio) and ratio > 0.0 else float("nan")
            out.append(
                {
                    "schema_version": "stream4d_v97_phase8_scale_pair_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "metric_level": "chunk_stitching_scale_eval",
                    "scene_id": scene,
                    "prev_window_id": prev["window_id"],
                    "next_window_id": nxt["window_id"],
                    "prev_sim3_scale": prev_scale,
                    "next_sim3_scale": next_scale,
                    "scale_next_over_prev": ratio,
                    "abs_log_scale_ratio": abs_log,
                    "scale_aligned_within_10pct": bool(np.isfinite(abs_log) and abs_log <= math.log(1.10)),
                    "prev_holdout_residual_p90_m": prev.get("holdout_residual_p90_m"),
                    "next_holdout_residual_p90_m": nxt.get("holdout_residual_p90_m"),
                    "coordinate_frame": "per_window_d4rt_xyz_ref_to_scannet_world",
                    "alignment_type": "eval_only_window_sim3_scale_ratio",
                    "alignment_source": "scannet_depth_pose_backprojection",
                    "scale_source": "window_eval_sim3_fit",
                    "uses_gt_for_metric": True,
                    "uses_gt_for_prediction": False,
                    "uses_rgbd_pose_mesh_for_metric": True,
                    "uses_rgbd_pose_for_diagnostic": True,
                    "is_method_safe_metric": False,
                    "is_diagnostic_metric": True,
                }
            )
    return out


def _ate_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "ate_frame_count": 0,
            "ate_sim3_rmse_m": None,
            "ate_sim3_median_m": None,
            "ate_sim3_p90_m": None,
            "ate_sim3_max_m": None,
        }
    return {
        "ate_frame_count": int(values.size),
        "ate_sim3_rmse_m": float(math.sqrt(float(np.mean(values**2)))),
        "ate_sim3_median_m": float(np.median(values)),
        "ate_sim3_p90_m": float(np.percentile(values, 90)),
        "ate_sim3_max_m": float(np.max(values)),
    }


def _trajectory_ate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    phase2_root = _project(args.phase2_root)
    batch_root = phase2_root / "carrier_batches" / args.decode_variant
    paths = sorted(batch_root.glob("*/*.npz"))
    streams: dict[str, ScanNetStream] = {}
    frame_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    all_residuals: list[float] = []
    for batch_path in paths:
        scene = batch_path.parent.name
        window = batch_path.stem
        if scene not in streams:
            streams[scene] = ScanNetStream(seq_name=scene, backbone=args.backbone, root=_project(args.scannet_root))
        stream = streams[scene]
        with np.load(batch_path, allow_pickle=True) as data:
            if "xyz_local" not in data.files:
                window_rows.append(
                    {
                        "schema_version": "stream4d_v97_phase8_trajectory_ate_window_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "metric_level": "trajectory_ate",
                        "scene_id": scene,
                        "window_id": window,
                        "status": "missing_xyz_local",
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_metric": True,
                        "uses_rgbd_pose_mesh_for_metric": True,
                        "is_method_safe_metric": False,
                        "is_diagnostic_metric": True,
                    }
                )
                continue
            frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
            uv = np.asarray(data["uv_pred"], dtype=np.float64)
            xyz_ref = np.asarray(data["xyz_ref"], dtype=np.float64)
            xyz_local = np.asarray(data["xyz_local"], dtype=np.float64)
            visibility = np.asarray(data["visibility_prob"], dtype=np.float64)
            confidence = np.asarray(data["confidence_prob"], dtype=np.float64)
            valid = np.asarray(data.get("valid", np.ones(uv.shape[:2], dtype=bool)), dtype=bool)
        pred_centers: list[np.ndarray] = []
        gt_centers: list[np.ndarray] = []
        local_frame_rows: list[dict[str, Any]] = []
        per_frame_cap = max(4, int(args.ate_max_anchors_per_window) // max(1, int(frame_ids.shape[0]) - 1))
        for local_idx, frame_id in enumerate(frame_ids.tolist()):
            pose = stream.load_pose(int(frame_id))
            if not np.isfinite(pose).all():
                continue
            gt_center = np.asarray(pose[:3, 3], dtype=np.float64)
            if local_idx == 0:
                pred_center = np.zeros((3,), dtype=np.float64)
                rigid_residual = np.empty((0,), dtype=np.float64)
                anchor_count = 0
                status = "reference_frame_zero_center"
            else:
                ok = (
                    valid[local_idx]
                    & np.isfinite(xyz_ref[local_idx]).all(axis=1)
                    & np.isfinite(xyz_local[local_idx]).all(axis=1)
                    & np.isfinite(uv[local_idx]).all(axis=1)
                    & (uv[local_idx, :, 0] >= 0.0)
                    & (uv[local_idx, :, 0] <= 1.0)
                    & (uv[local_idx, :, 1] >= 0.0)
                    & (uv[local_idx, :, 1] <= 1.0)
                    & (visibility[local_idx] >= float(args.min_visibility))
                    & (confidence[local_idx] >= float(args.min_confidence))
                )
                indices = np.flatnonzero(ok)
                if indices.shape[0] < 4:
                    continue
                indices = indices[_stable_sample_indices(int(indices.shape[0]), per_frame_cap)]
                try:
                    rot, trans, rigid_residual = _fit_rigid_no_scale(xyz_ref[local_idx, indices], xyz_local[local_idx, indices])
                except Exception:
                    continue
                pred_center = -rot.T @ trans
                anchor_count = int(indices.shape[0])
                status = "ok"
            pred_centers.append(pred_center)
            gt_centers.append(gt_center)
            local_frame_rows.append(
                {
                    "schema_version": "stream4d_v97_phase8_trajectory_ate_frame_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "metric_level": "trajectory_ate",
                    "decode_variant": args.decode_variant,
                    "scene_id": scene,
                    "window_id": window,
                    "frame_id": int(frame_id),
                    "local_idx": int(local_idx),
                    "status": status,
                    "rigid_anchor_count": int(anchor_count),
                    "rigid_residual_median_m": _quantile(rigid_residual, 50),
                    "rigid_residual_p90_m": _quantile(rigid_residual, 90),
                    "pred_center_x": float(pred_center[0]),
                    "pred_center_y": float(pred_center[1]),
                    "pred_center_z": float(pred_center[2]),
                    "gt_center_x": float(gt_center[0]),
                    "gt_center_y": float(gt_center[1]),
                    "gt_center_z": float(gt_center[2]),
                    "center_convention": "fit xyz_ref->xyz_local rigid transform, then C_ref=-R^T t",
                    "coordinate_frame": "d4rt_ref_camera_centers_to_scannet_pose_centers",
                    "alignment_type": "eval_only_sim3",
                    "alignment_source": "scannet_pose_centers",
                    "scale_source": "trajectory_eval_sim3_fit",
                    "uses_gt_for_metric": True,
                    "uses_gt_for_prediction": False,
                    "uses_rgbd_pose_mesh_for_metric": True,
                    "uses_rgbd_pose_for_diagnostic": True,
                    "is_method_safe_metric": False,
                    "is_diagnostic_metric": True,
                }
            )
        pred = np.asarray(pred_centers, dtype=np.float64)
        gt = np.asarray(gt_centers, dtype=np.float64)
        if pred.shape[0] < 4:
            frame_rows.extend(local_frame_rows)
            window_rows.append(
                {
                    "schema_version": "stream4d_v97_phase8_trajectory_ate_window_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "metric_level": "trajectory_ate",
                    "decode_variant": args.decode_variant,
                    "scene_id": scene,
                    "window_id": window,
                    "status": "too_few_frames",
                    "frame_count": int(pred.shape[0]),
                    "uses_gt_for_metric": True,
                    "uses_gt_for_prediction": False,
                    "uses_rgbd_pose_mesh_for_metric": True,
                    "is_method_safe_metric": False,
                    "is_diagnostic_metric": True,
                }
            )
            continue
        fit = fit_transform(pred, gt, robust_trim_percentile=100.0)
        if fit is None:
            frame_rows.extend(local_frame_rows)
            continue
        residual = np.asarray(fit["residual"], dtype=np.float64)
        for row, value in zip(local_frame_rows, residual):
            row["ate_sim3_residual_m"] = float(value)
        frame_rows.extend(local_frame_rows)
        all_residuals.extend(float(v) for v in residual.tolist())
        window_rows.append(
            {
                "schema_version": "stream4d_v97_phase8_trajectory_ate_window_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "metric_level": "trajectory_ate",
                "decode_variant": args.decode_variant,
                "scene_id": scene,
                "window_id": window,
                "status": "ok",
                "frame_count": int(pred.shape[0]),
                "sim3_scale": float(fit["scale"]),
                "sim3_rotation_det": float(fit["rotation_det"]),
                **_ate_stats(residual),
                "center_convention": "fit xyz_ref->xyz_local rigid transform, then C_ref=-R^T t",
                "coordinate_frame": "d4rt_ref_camera_centers_to_scannet_pose_centers",
                "alignment_type": "eval_only_sim3",
                "alignment_source": "scannet_pose_centers",
                "scale_source": "trajectory_eval_sim3_fit",
                "uses_gt_for_metric": True,
                "uses_gt_for_prediction": False,
                "uses_rgbd_pose_mesh_for_metric": True,
                "uses_rgbd_pose_for_diagnostic": True,
                "is_method_safe_metric": False,
                "is_diagnostic_metric": True,
            }
        )
    summary = {
        "trajectory_ate_status": "ok" if all_residuals else "no_valid_ate_windows",
        **_ate_stats(np.asarray(all_residuals, dtype=np.float64)),
    }
    return frame_rows, window_rows, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frame_rows, source, target, chunks = _load_samples(args)
    fit_rows: list[dict[str, Any]] = []
    scene_fits: dict[str, dict[str, Any] | None] = {}
    window_fits: dict[str, dict[str, Any] | None] = {}
    for key in sorted(k for k in chunks if k.startswith("scene:")):
        idx = np.asarray(chunks[key], dtype=np.int64)
        fit, row = _fit_scope(
            scope="scene",
            scope_id=key.split(":", 1)[1],
            source=source[idx],
            target=target[idx],
            robust_trim_percentile=float(args.robust_trim_percentile),
            holdout_mod=int(args.holdout_mod),
        )
        scene_fits[key] = fit
        fit_rows.append(row)
    for key in sorted(k for k in chunks if k.startswith("window:")):
        idx = np.asarray(chunks[key], dtype=np.int64)
        fit, row = _fit_scope(
            scope="window",
            scope_id=key.split(":", 1)[1],
            source=source[idx],
            target=target[idx],
            robust_trim_percentile=float(args.robust_trim_percentile),
            holdout_mod=int(args.holdout_mod),
        )
        window_fits[key] = fit
        fit_rows.append(row)

    for row in frame_rows:
        frame_key = f"frame:{row['scene_id']}:{row['window_id']}:{int(row['frame_id'])}"
        idx = np.asarray(chunks.get(frame_key, []), dtype=np.int64)
        if idx.size == 0:
            continue
        scene_fit = scene_fits.get(f"scene:{row['scene_id']}")
        window_fit = window_fits.get(f"window:{row['scene_id']}:{row['window_id']}")
        scene_stats = _chamfer_stats(_apply_fit(source[idx], scene_fit), target[idx])
        window_stats = _chamfer_stats(_apply_fit(source[idx], window_fit), target[idx])
        for key, value in scene_stats.items():
            row[f"scene_fit_{key}"] = value
        for key, value in window_stats.items():
            row[f"window_fit_{key}"] = value

    scene_paired = np.asarray(
        [row.get("scene_fit_paired_residual_p50_m") for row in frame_rows if row.get("scene_fit_paired_residual_p50_m") is not None],
        dtype=np.float64,
    )
    window_paired = np.asarray(
        [row.get("window_fit_paired_residual_p50_m") for row in frame_rows if row.get("window_fit_paired_residual_p50_m") is not None],
        dtype=np.float64,
    )
    scene_chamfer = np.asarray(
        [row.get("scene_fit_symmetric_chamfer_mean_m") for row in frame_rows if row.get("scene_fit_symmetric_chamfer_mean_m") is not None],
        dtype=np.float64,
    )
    window_chamfer = np.asarray(
        [row.get("window_fit_symmetric_chamfer_mean_m") for row in frame_rows if row.get("window_fit_symmetric_chamfer_mean_m") is not None],
        dtype=np.float64,
    )
    local_reproj_p50 = np.asarray(
        [row.get("local_xyz_reproj_error_px_p50") for row in frame_rows if row.get("local_xyz_reproj_error_px_p50") is not None],
        dtype=np.float64,
    )
    local_reproj_p90 = np.asarray(
        [row.get("local_xyz_reproj_error_px_p90") for row in frame_rows if row.get("local_xyz_reproj_error_px_p90") is not None],
        dtype=np.float64,
    )
    scale_pair_rows = _build_scale_pair_rows(fit_rows)
    scale_abs_logs = np.asarray(
        [row.get("abs_log_scale_ratio") for row in scale_pair_rows if row.get("abs_log_scale_ratio") is not None],
        dtype=np.float64,
    )
    outside_scale_10pct = sum(1 for row in scale_pair_rows if not bool(row.get("scale_aligned_within_10pct")))
    ate_frame_rows, ate_window_rows, ate_summary = _trajectory_ate(args)
    total_query_slots = int(chunks.get("total_query_slots", [0])[0])
    total_valid_uv = int(chunks.get("total_valid_uv", [0])[0])
    total_accepted = int(chunks.get("total_accepted", [0])[0])
    total_depth_hits = int(chunks.get("total_depth_hits", [0])[0])
    summary = {
        "schema": "stream4d_v97_phase8_d4rt_geometry_quality_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "DIAGNOSTIC_ONLY_V97_D4RT_GEOMETRY_QUALITY",
        "decode_variant": args.decode_variant,
        "phase2_root": _rel(_project(args.phase2_root)),
        "output_root": _rel(output_root),
        "frame_geometry_rows": _rel(output_root / "frame_geometry_rows.csv"),
        "sim3_fit_rows": _rel(output_root / "sim3_fit_rows.csv"),
        "scale_pair_rows": _rel(output_root / "scale_pair_rows.csv"),
        "trajectory_ate_frame_rows": _rel(output_root / "trajectory_ate_frame_rows.csv"),
        "trajectory_ate_window_rows": _rel(output_root / "trajectory_ate_window_rows.csv"),
        "total_query_slots": total_query_slots,
        "total_valid_uv": total_valid_uv,
        "total_accepted_by_visibility_confidence": total_accepted,
        "total_depth_hits_after_sampling": total_depth_hits,
        "uv_in01_rate": float(total_valid_uv / max(1, total_query_slots)),
        "accepted_rate_of_query_slots": float(total_accepted / max(1, total_query_slots)),
        "depth_hit_rate_of_accepted_sample": float(total_depth_hits / max(1, sum(int(row.get("sampled_count", 0)) for row in frame_rows))),
        "frame_count": len(frame_rows),
        "sampled_point_count": int(source.shape[0]),
        "max_points_per_frame": int(args.max_points_per_frame),
        "min_visibility": float(args.min_visibility),
        "min_confidence": float(args.min_confidence),
        "robust_trim_percentile": float(args.robust_trim_percentile),
        "holdout_mod": int(args.holdout_mod),
        "scene_fit_frame_paired_residual_p50_m_mean": _mean(scene_paired),
        "scene_fit_frame_paired_residual_p50_m_p90": _quantile(scene_paired, 90),
        "window_fit_frame_paired_residual_p50_m_mean": _mean(window_paired),
        "window_fit_frame_paired_residual_p50_m_p90": _quantile(window_paired, 90),
        "scene_fit_frame_symmetric_chamfer_mean_m_mean": _mean(scene_chamfer),
        "scene_fit_frame_symmetric_chamfer_mean_m_p90": _quantile(scene_chamfer, 90),
        "window_fit_frame_symmetric_chamfer_mean_m_mean": _mean(window_chamfer),
        "window_fit_frame_symmetric_chamfer_mean_m_p90": _quantile(window_chamfer, 90),
        "local_xyz_reproj_error_px_p50_frame_mean": _mean(local_reproj_p50),
        "local_xyz_reproj_error_px_p90_frame_mean": _mean(local_reproj_p90),
        "scale_pair_count": int(len(scale_pair_rows)),
        "scale_pair_outside_10pct_count": int(outside_scale_10pct),
        "scale_pair_abs_log_ratio_max": _quantile(scale_abs_logs, 100),
        "scale_pair_abs_log_ratio_p90": _quantile(scale_abs_logs, 90),
        **ate_summary,
        "metric_levels_present": sorted(
            set(str(row.get("metric_level", "")) for row in frame_rows + fit_rows + scale_pair_rows + ate_frame_rows + ate_window_rows if row.get("metric_level"))
        ),
        "uses_gt_for_metric": True,
        "uses_gt_for_prediction": False,
        "uses_rgbd_pose_mesh_for_metric": True,
        "uses_rgbd_pose_for_diagnostic": True,
        "is_method_safe_metric": False,
        "is_diagnostic_metric": True,
        "diagnostic_note": "ScanNet depth/pose are used only to audit D4RT metric geometry; this artifact must not be used as prediction evidence.",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "runtime_sec": float(time.time() - started),
    }
    _write_csv(output_root / "frame_geometry_rows.csv", frame_rows)
    _write_csv(output_root / "sim3_fit_rows.csv", fit_rows)
    _write_csv(output_root / "scale_pair_rows.csv", scale_pair_rows)
    _write_csv(output_root / "trajectory_ate_frame_rows.csv", ate_frame_rows)
    _write_csv(output_root / "trajectory_ate_window_rows.csv", ate_window_rows)
    _write_json(output_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "decode_variant": args.decode_variant,
                "sampled_point_count": summary["sampled_point_count"],
                "scene_fit_frame_paired_residual_p50_m_mean": summary["scene_fit_frame_paired_residual_p50_m_mean"],
                "window_fit_frame_paired_residual_p50_m_mean": summary["window_fit_frame_paired_residual_p50_m_mean"],
                "local_xyz_reproj_error_px_p50_frame_mean": summary["local_xyz_reproj_error_px_p50_frame_mean"],
                "scale_pair_outside_10pct_count": summary["scale_pair_outside_10pct_count"],
                "ate_sim3_rmse_m": summary["ate_sim3_rmse_m"],
                "runtime_sec": summary["runtime_sec"],
            },
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-root", default=str(DEFAULT_PHASE2))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--decode-variant", default="D3_adaptive1024")
    parser.add_argument("--scannet-root", default=str(DEFAULT_SCANNET))
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-points-per-frame", type=int, default=2048)
    parser.add_argument("--ate-max-anchors-per-window", type=int, default=8000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--holdout-mod", type=int, default=5)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
