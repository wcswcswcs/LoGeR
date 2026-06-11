from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from stream4d.carrier_sampler import CarrierSampler
from stream4d.carrier_store import CarrierSources
from stream4d.d4rt_adapter import D4RTAdapter
from stream4d.scannet_stream import ScanNetStream


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _parse_csv_modes(raw: str) -> list[str]:
    modes = [part.strip() for part in str(raw).split(",") if part.strip()]
    return modes or ["raw"]


def _transform_xyz_hypothesis(xyz: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(xyz, dtype=np.float32)
    if mode == "raw":
        return arr
    if mode == "signed_log1p":
        return (np.sign(arr) * np.log1p(np.abs(arr))).astype(np.float32)
    if mode == "signed_expm1":
        abs_clipped = np.clip(np.abs(arr), 0.0, 20.0)
        return (np.sign(arr) * np.expm1(abs_clipped)).astype(np.float32)
    raise ValueError(f"Unsupported xyz transform mode: {mode}")


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _sample_indices(indices: np.ndarray, max_count: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.shape[0] <= int(max_count):
        return indices
    keep = np.linspace(0, indices.shape[0] - 1, num=int(max_count), dtype=np.int64)
    return indices[keep]


def _slice_sources(sources: CarrierSources, keep: np.ndarray) -> CarrierSources:
    keep = np.asarray(keep, dtype=np.int64)
    return CarrierSources(
        carrier_id=np.asarray(sources.carrier_id)[keep],
        src_frame=np.asarray(sources.src_frame)[keep],
        src_frame_global=np.asarray(sources.src_frame_global)[keep],
        src_xy=np.asarray(sources.src_xy)[keep],
        src_uv=np.asarray(sources.src_uv)[keep],
        src_mask_id=np.asarray(sources.src_mask_id)[keep],
    )


def _depth_errors(pred_depth: np.ndarray, gt_depth: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred_depth, dtype=np.float64)
    gt = np.asarray(gt_depth, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 1e-6) & (gt > 1e-6)
    if np.count_nonzero(valid) == 0:
        return {"valid_ratio": 0.0}
    pred = pred[valid]
    gt = gt[valid]
    ratio = np.maximum(pred / gt, gt / pred)
    diff = pred - gt
    return {
        "valid_ratio": float(np.mean(valid)),
        "absrel": float(np.mean(np.abs(diff) / gt)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


def _fit_depth(raw: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(raw, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.isfinite(raw) & np.isfinite(gt) & (raw > 1e-6) & (gt > 1e-6)
    if np.count_nonzero(valid) < 8:
        return {"fit_status": "failed_insufficient_anchors", "anchor_count": int(np.count_nonzero(valid))}
    raw = raw[valid]
    gt = gt[valid]
    scale = float(np.median(gt / raw))
    design = np.stack([raw, np.ones(raw.shape[0], dtype=np.float64)], axis=1)
    linear_scale, linear_shift = np.linalg.lstsq(design, gt, rcond=None)[0]
    out: dict[str, Any] = {
        "fit_status": "ok",
        "anchor_count": int(raw.shape[0]),
        "median_scale": scale,
        "linear_scale": float(linear_scale),
        "linear_shift": float(linear_shift),
    }
    out.update({f"raw_{key}": value for key, value in _depth_errors(raw, gt).items()})
    out.update({f"median_{key}": value for key, value in _depth_errors(raw * scale, gt).items()})
    out.update({f"linear_{key}": value for key, value in _depth_errors(raw * float(linear_scale) + float(linear_shift), gt).items()})
    return out


def _point_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    pred = pred[np.isfinite(pred).all(axis=1)]
    gt = gt[np.isfinite(gt).all(axis=1)]
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return {"status": "empty", "pred_point_count": int(pred.shape[0]), "gt_point_count": int(gt.shape[0])}
    gt_tree = cKDTree(gt)
    pred_tree = cKDTree(pred)
    pred_to_gt, _ = gt_tree.query(pred, k=1)
    gt_to_pred, _ = pred_tree.query(gt, k=1)
    out: dict[str, Any] = {
        "status": "ok",
        "pred_point_count": int(pred.shape[0]),
        "gt_point_count": int(gt.shape[0]),
        "chamfer_l1": float(np.mean(pred_to_gt) + np.mean(gt_to_pred)),
        "outlier_rate_20cm": float(np.mean(pred_to_gt > 0.20)),
        "pred_to_gt_median": float(np.median(pred_to_gt)),
    }
    for tau in (0.05, 0.10, 0.20):
        precision = float(np.mean(pred_to_gt < tau))
        recall = float(np.mean(gt_to_pred < tau))
        out[f"precision@{int(tau * 100)}cm"] = precision
        out[f"recall@{int(tau * 100)}cm"] = recall
        out[f"fscore@{int(tau * 100)}cm"] = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return out


def _gt_camera_points(stream: ScanNetStream, frame_id: int, *, stride: int, max_points: int) -> np.ndarray:
    depth = stream.load_depth(int(frame_id))
    intr = stream.load_intrinsics()
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    h, w = depth.shape[:2]
    yy, xx = np.mgrid[0:h:int(stride), 0:w:int(stride)]
    z = depth[yy, xx].reshape(-1).astype(np.float32)
    x = xx.reshape(-1).astype(np.float32)
    y = yy.reshape(-1).astype(np.float32)
    valid = np.isfinite(z) & (z > 0.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    points = np.stack(
        [(x[valid] - cx) * z[valid] / fx, (y[valid] - cy) * z[valid] / fy, z[valid]],
        axis=1,
    ).astype(np.float32)
    if points.shape[0] > int(max_points):
        keep = np.linspace(0, points.shape[0] - 1, num=int(max_points), dtype=np.int64)
        points = points[keep]
    return points


def _target_samples(
    stream: ScanNetStream,
    xyz: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    frame_ids: list[int],
    *,
    min_visibility: float,
    min_confidence: float,
    max_anchors: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    raw_depths: list[np.ndarray] = []
    gt_depths: list[np.ndarray] = []
    reproj_errors: list[np.ndarray] = []
    xyz_points: list[np.ndarray] = []
    uvz_points: list[np.ndarray] = []
    xyz_frames: list[np.ndarray] = []
    uvz_frames: list[np.ndarray] = []
    raw_uv_pix: list[np.ndarray] = []
    raw_frames: list[np.ndarray] = []
    frame_slots = max(1, len(frame_ids))
    per_frame_cap = max(16, int(max_anchors) // frame_slots)
    intr = stream.load_intrinsics()
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    for local_idx, frame_id in enumerate(frame_ids):
        depth = stream.load_depth(int(frame_id))
        h, w = depth.shape[:2]
        ok = (
            np.asarray(valid[local_idx], dtype=bool)
            & (np.asarray(visibility[local_idx], dtype=np.float32) >= float(min_visibility))
            & (np.asarray(confidence[local_idx], dtype=np.float32) >= float(min_confidence))
            & np.isfinite(xyz[local_idx]).all(axis=1)
            & np.isfinite(uv[local_idx]).all(axis=1)
            & (uv[local_idx, :, 0] >= 0.0)
            & (uv[local_idx, :, 0] <= 1.0)
            & (uv[local_idx, :, 1] >= 0.0)
            & (uv[local_idx, :, 1] <= 1.0)
        )
        indices = _sample_indices(np.flatnonzero(ok), per_frame_cap)
        if indices.size == 0:
            continue
        branch = np.asarray(xyz[local_idx, indices], dtype=np.float64)
        uv_sel = np.asarray(uv[local_idx, indices], dtype=np.float64)
        x_pix = uv_sel[:, 0] * float(max(w - 1, 1))
        y_pix = uv_sel[:, 1] * float(max(h - 1, 1))
        xi = np.rint(x_pix).astype(np.int64)
        yi = np.rint(y_pix).astype(np.int64)
        in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if not np.any(in_bounds):
            continue
        branch = branch[in_bounds]
        x_pix = x_pix[in_bounds]
        y_pix = y_pix[in_bounds]
        gt_z = depth[yi[in_bounds], xi[in_bounds]].astype(np.float64)
        pred_z = branch[:, 2].astype(np.float64)
        raw_depths.append(pred_z)
        gt_depths.append(gt_z)
        raw_uv_pix.append(np.stack([x_pix, y_pix], axis=1).astype(np.float64))
        raw_frames.append(np.full((pred_z.shape[0],), int(frame_id), dtype=np.int64))
        positive = np.isfinite(pred_z) & (pred_z > 1e-6)
        if np.any(positive):
            x_proj = branch[positive, 0] * fx / pred_z[positive] + cx
            y_proj = branch[positive, 1] * fy / pred_z[positive] + cy
            err = np.sqrt((x_proj - x_pix[positive]) ** 2 + (y_proj - y_pix[positive]) ** 2)
            err = err[np.isfinite(err)]
            if err.size:
                reproj_errors.append(err)
            uvz = np.stack(
                [
                    (x_pix[positive] - cx) * pred_z[positive] / fx,
                    (y_pix[positive] - cy) * pred_z[positive] / fy,
                    pred_z[positive],
                ],
                axis=1,
            )
            uvz_points.append(uvz.astype(np.float32))
            uvz_frames.append(np.full((uvz.shape[0],), int(frame_id), dtype=np.int64))
        xyz_points.append(branch.astype(np.float32))
        xyz_frames.append(np.full((branch.shape[0],), int(frame_id), dtype=np.int64))
    raw = np.concatenate(raw_depths, axis=0) if raw_depths else np.empty((0,), dtype=np.float64)
    gt = np.concatenate(gt_depths, axis=0) if gt_depths else np.empty((0,), dtype=np.float64)
    reproj = np.concatenate(reproj_errors, axis=0) if reproj_errors else np.empty((0,), dtype=np.float64)
    xyz_out = np.concatenate(xyz_points, axis=0) if xyz_points else np.empty((0, 3), dtype=np.float32)
    uvz_out = np.concatenate(uvz_points, axis=0) if uvz_points else np.empty((0, 3), dtype=np.float32)
    xyz_frames_out = np.concatenate(xyz_frames, axis=0) if xyz_frames else np.empty((0,), dtype=np.int64)
    uvz_frames_out = np.concatenate(uvz_frames, axis=0) if uvz_frames else np.empty((0,), dtype=np.int64)
    raw_uv_pix_out = np.concatenate(raw_uv_pix, axis=0) if raw_uv_pix else np.empty((0, 2), dtype=np.float64)
    raw_frames_out = np.concatenate(raw_frames, axis=0) if raw_frames else np.empty((0,), dtype=np.int64)
    return raw, gt, reproj, xyz_out, uvz_out, xyz_frames_out, uvz_frames_out, raw_uv_pix_out, raw_frames_out


def _apply_depth_fit_to_raw(raw: np.ndarray, fit: dict[str, Any], mode: str) -> np.ndarray:
    depth = np.asarray(raw, dtype=np.float64)
    if mode == "median":
        return depth * float(fit["median_scale"])
    if mode == "linear":
        return depth * float(fit["linear_scale"]) + float(fit["linear_shift"])
    raise ValueError(f"Unsupported fit mode: {mode}")


def _uvz_points_from_depth(
    stream: ScanNetStream,
    depth: np.ndarray,
    uv_pix: np.ndarray,
    frame_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    intr = stream.load_intrinsics()
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    uv_pix = np.asarray(uv_pix, dtype=np.float64).reshape(-1, 2)
    frame_ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    if uv_pix.shape[0] != depth.shape[0] or frame_ids.shape[0] != depth.shape[0]:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
    valid = (
        np.isfinite(depth)
        & (depth > 1e-6)
        & np.isfinite(uv_pix).all(axis=1)
    )
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
    z = depth[valid]
    uv = uv_pix[valid]
    points = np.stack([(uv[:, 0] - cx) * z / fx, (uv[:, 1] - cy) * z / fy, z], axis=1).astype(np.float32)
    return points, frame_ids[valid]


def _camera_metrics_by_frame(
    stream: ScanNetStream,
    points: np.ndarray,
    frame_ids: np.ndarray,
    *,
    gt_stride: int,
    max_gt_points_per_frame: int,
) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float32)
    frame_ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    if points.shape[0] == 0 or frame_ids.shape[0] != points.shape[0]:
        return {"status": "empty", "frame_count": 0}
    rows: list[dict[str, Any]] = []
    total_pred = 0
    total_gt = 0
    for frame_id in sorted(set(int(v) for v in frame_ids.tolist())):
        idx = np.flatnonzero(frame_ids == int(frame_id))
        if idx.size == 0:
            continue
        gt = _gt_camera_points(stream, int(frame_id), stride=gt_stride, max_points=max_gt_points_per_frame)
        metrics = _point_metrics(points[idx], gt)
        if metrics.get("status") != "ok":
            continue
        total_pred += int(metrics.get("pred_point_count", idx.size))
        total_gt += int(metrics.get("gt_point_count", gt.shape[0]))
        rows.append(metrics)
    if not rows:
        return {"status": "empty", "frame_count": 0, "pred_point_count": int(points.shape[0]), "gt_point_count": 0}
    out: dict[str, Any] = {
        "status": "ok",
        "frame_count": int(len(rows)),
        "pred_point_count": int(total_pred),
        "gt_point_count": int(total_gt),
    }
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and key not in {"pred_point_count", "gt_point_count"}
        }
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        if values:
            out[key] = float(np.mean(values))
    return out


def _source_self_samples(
    stream: ScanNetStream,
    xyz: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    sources: CarrierSources,
    frame_ids: list[int],
    *,
    max_anchors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_depths: list[np.ndarray] = []
    gt_depths: list[np.ndarray] = []
    uv_errors: list[np.ndarray] = []
    intr_cache: dict[int, tuple[int, int]] = {}
    indices = np.arange(np.asarray(sources.carrier_id).shape[0], dtype=np.int64)
    indices = _sample_indices(indices, int(max_anchors))
    for q in indices:
        local_idx = int(np.asarray(sources.src_frame)[q])
        if local_idx < 0 or local_idx >= len(frame_ids) or local_idx >= xyz.shape[0]:
            continue
        if not bool(valid[local_idx, q]):
            continue
        frame_id = int(frame_ids[local_idx])
        if frame_id not in intr_cache:
            depth = stream.load_depth(frame_id)
            intr_cache[frame_id] = depth.shape[:2]
        else:
            depth = stream.load_depth(frame_id)
        h, w = intr_cache[frame_id]
        pred_uv = np.asarray(uv[local_idx, q], dtype=np.float64)
        src_uv = np.asarray(sources.src_uv[q], dtype=np.float64)
        if not np.isfinite(pred_uv).all() or not np.isfinite(src_uv).all():
            continue
        uv_errors.append(
            np.asarray(
                [
                    np.sqrt(
                        ((pred_uv[0] - src_uv[0]) * float(max(w - 1, 1))) ** 2
                        + ((pred_uv[1] - src_uv[1]) * float(max(h - 1, 1))) ** 2
                    )
                ],
                dtype=np.float64,
            )
        )
        if sources.src_xy is not None:
            x = int(np.asarray(sources.src_xy)[q, 0])
            y = int(np.asarray(sources.src_xy)[q, 1])
        else:
            x = int(np.rint(src_uv[0] * float(max(w - 1, 1))))
            y = int(np.rint(src_uv[1] * float(max(h - 1, 1))))
        if 0 <= x < w and 0 <= y < h:
            pred_z = float(xyz[local_idx, q, 2])
            gt_z = float(depth[y, x])
            raw_depths.append(np.asarray([pred_z], dtype=np.float64))
            gt_depths.append(np.asarray([gt_z], dtype=np.float64))
    raw = np.concatenate(raw_depths, axis=0) if raw_depths else np.empty((0,), dtype=np.float64)
    gt = np.concatenate(gt_depths, axis=0) if gt_depths else np.empty((0,), dtype=np.float64)
    uv_err = np.concatenate(uv_errors, axis=0) if uv_errors else np.empty((0,), dtype=np.float64)
    return raw, gt, uv_err


def _branch_row(
    *,
    stream: ScanNetStream,
    branch_name: str,
    xyz_transform: str,
    xyz: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    sources: CarrierSources,
    frame_ids: list[int],
    min_visibility: float,
    min_confidence: float,
    max_anchors: int,
    gt_stride: int,
    max_gt_points_per_frame: int,
) -> dict[str, Any]:
    raw, gt, reproj, xyz_points, uvz_points, xyz_frame_ids, uvz_frame_ids, raw_uv_pix, raw_frame_ids = _target_samples(
        stream,
        xyz,
        uv,
        valid,
        visibility,
        confidence,
        frame_ids,
        min_visibility=min_visibility,
        min_confidence=min_confidence,
        max_anchors=max_anchors,
    )
    src_raw, src_gt, src_uv_err = _source_self_samples(
        stream,
        xyz,
        uv,
        valid,
        sources,
        frame_ids,
        max_anchors=max_anchors,
    )
    row: dict[str, Any] = {
        "branch": branch_name,
        "xyz_transform": xyz_transform,
        "target_anchor_count": int(raw.shape[0]),
        "target_positive_z_rate": float(np.mean(raw > 1e-6)) if raw.size else 0.0,
        "source_anchor_count": int(src_raw.shape[0]),
        "source_self_uv_count": int(src_uv_err.shape[0]),
    }
    if reproj.size:
        row.update(
            {
                "target_reproj_error_px_median": float(np.median(reproj)),
                "target_reproj_error_px_p90": float(np.percentile(reproj, 90)),
                "target_reproj_error_px_mean": float(np.mean(reproj)),
            }
        )
    if src_uv_err.size:
        row.update(
            {
                "source_self_uv_error_px_median": float(np.median(src_uv_err)),
                "source_self_uv_error_px_p90": float(np.percentile(src_uv_err, 90)),
            }
        )
    target_depth_fit = _fit_depth(raw, gt)
    row.update({f"target_depth_{key}": value for key, value in target_depth_fit.items()})
    row.update({f"source_depth_{key}": value for key, value in _fit_depth(src_raw, src_gt).items()})
    row.update(
        {
            f"target_xyz_camera_{key}": value
            for key, value in _camera_metrics_by_frame(
                stream,
                xyz_points,
                xyz_frame_ids,
                gt_stride=gt_stride,
                max_gt_points_per_frame=max_gt_points_per_frame,
            ).items()
        }
    )
    row.update(
        {
            f"target_uvz_camera_{key}": value
            for key, value in _camera_metrics_by_frame(
                stream,
                uvz_points,
                uvz_frame_ids,
                gt_stride=gt_stride,
                max_gt_points_per_frame=max_gt_points_per_frame,
            ).items()
        }
    )
    if target_depth_fit.get("fit_status") == "ok":
        for mode in ("median", "linear"):
            calibrated_depth = _apply_depth_fit_to_raw(raw, target_depth_fit, mode)
            calibrated_points, calibrated_frames = _uvz_points_from_depth(
                stream,
                calibrated_depth,
                raw_uv_pix,
                raw_frame_ids,
            )
            row.update(
                {
                    f"target_uvz_{mode}_camera_{key}": value
                    for key, value in _camera_metrics_by_frame(
                        stream,
                        calibrated_points,
                        calibrated_frames,
                        gt_stride=gt_stride,
                        max_gt_points_per_frame=max_gt_points_per_frame,
                    ).items()
                }
            )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stream4D v22 D4RT Local-vs-Ref Diagnostic",
        "",
        "Diagnostic-only. Uses ScanNet depth/pose only for evaluation, not for prediction.",
        "",
        "| scene | branch | transform | target anchors | reproj median px | raw d1 | linear d1 | xyz F@10 | uvz F@10 | uvz-linear F@10 | source uv median px | source raw d1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scene} | {branch} | {transform} | {anchors} | {reproj} | {raw_d1} | {lin_d1} | {xyz_f10} | {uvz_f10} | {uvz_lin_f10} | {src_uv} | {src_d1} |".format(
                scene=row.get("scene", ""),
                branch=row.get("branch", ""),
                transform=row.get("xyz_transform", "raw"),
                anchors=_fmt(row.get("target_anchor_count")),
                reproj=_fmt(row.get("target_reproj_error_px_median")),
                raw_d1=_fmt(row.get("target_depth_raw_delta1")),
                lin_d1=_fmt(row.get("target_depth_linear_delta1")),
                xyz_f10=_fmt(row.get("target_xyz_camera_fscore@10cm")),
                uvz_f10=_fmt(row.get("target_uvz_camera_fscore@10cm")),
                uvz_lin_f10=_fmt(row.get("target_uvz_linear_camera_fscore@10cm")),
                src_uv=_fmt(row.get("source_self_uv_error_px_median")),
                src_d1=_fmt(row.get("source_depth_raw_delta1")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.6g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare D4RT xyz_local and xyz_ref0 against target UV/depth on ScanNet.")
    parser.add_argument("--d4rt-root", required=True)
    parser.add_argument("--d4rt-config", required=True)
    parser.add_argument("--d4rt-ckpt", required=True)
    parser.add_argument("--seq-name", default=None)
    parser.add_argument("--seq-list", default=None)
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-points-per-mask", type=int, default=16)
    parser.add_argument("--min-points-per-mask", type=int, default=4)
    parser.add_argument("--sampling-strategy", default="grid_inside_mask")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-source-points", type=int, default=1024)
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument(
        "--xyz-transform-modes",
        default="raw",
        help="Comma-separated xyz hypothesis transforms: raw,signed_log1p,signed_expm1.",
    )
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--gt-depth-stride", type=int, default=12)
    parser.add_argument("--max-gt-points-per-frame", type=int, default=1500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--cache-root", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    xyz_transform_modes = _parse_csv_modes(args.xyz_transform_modes)
    if args.seq_list:
        scenes = _read_seq_list(Path(args.seq_list))
    elif args.seq_name:
        scenes = [args.seq_name]
    else:
        raise ValueError("Provide --seq-name or --seq-list")

    adapter = D4RTAdapter(
        d4rt_root=args.d4rt_root,
        model_config=args.d4rt_config,
        ckpt_path=args.d4rt_ckpt,
        device=args.device,
    )
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        stream = ScanNetStream(seq_name=scene, backbone=args.backbone, root=args.scannet_root)
        errors = stream.validate(require_masks=True)
        if errors:
            rows.append({"scene": scene, "status": "failed", "error": "; ".join(errors)})
            continue
        frame_ids = stream.frame_ids(stride=int(args.frame_stride), max_frames=int(args.max_frames))
        data = stream.load_window(frame_ids)
        sampler = CarrierSampler(
            max_points_per_mask=int(args.max_points_per_mask),
            min_points_per_mask=int(args.min_points_per_mask),
            strategy=args.sampling_strategy,
            seed=int(args.seed),
        )
        sources = sampler.sample(masks=np.asarray(data["mask"]), frame_ids=frame_ids)
        if sources.carrier_id.shape[0] > int(args.max_source_points):
            keep = _sample_indices(np.arange(sources.carrier_id.shape[0], dtype=np.int64), int(args.max_source_points))
            sources = _slice_sources(sources, keep)
        batch = adapter.infer_carriers(
            video_rgb_uint8=np.asarray(data["rgb"]),
            src_uv_norm=sources.src_uv,
            src_frame_local=sources.src_frame,
            carrier_id=sources.carrier_id,
            src_frame_global=sources.src_frame_global,
            src_xy=sources.src_xy,
            src_mask_id=sources.src_mask_id,
            query_chunk_size=int(args.query_chunk_size),
        )
        if args.cache_root:
            cache_dir = Path(args.cache_root) / scene
            cache_dir.mkdir(parents=True, exist_ok=True)
            sources.save_npz(cache_dir / "carrier_sources_window000.npz")
            batch.save_npz(cache_dir / "carriers_window000.npz")
            (cache_dir / "carriers_window000_manifest.json").write_text(
                json.dumps(
                    {
                        "seq_name": scene,
                        "frame_ids": [int(v) for v in frame_ids],
                        "raw_frame_ids": [int(v) for v in frame_ids],
                        "num_frames": int(len(frame_ids)),
                        "num_carriers": int(batch.carrier_id.shape[0]),
                        "diagnostic_only": True,
                        "is_method_result": False,
                        "contains_xyz_local": bool(batch.xyz_local is not None),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        branches = {
            "xyz_ref0": np.asarray(batch.xyz_ref, dtype=np.float32),
            "xyz_local": np.asarray(batch.xyz_local, dtype=np.float32)
            if batch.xyz_local is not None
            else np.full_like(batch.xyz_ref, np.nan),
        }
        for branch_name, xyz in branches.items():
            for xyz_transform in xyz_transform_modes:
                transformed_xyz = _transform_xyz_hypothesis(xyz, xyz_transform)
                row = {
                    "scene": scene,
                    "status": "ok",
                    "diagnostic_only": True,
                    "is_method_result": False,
                    "frame_count": int(len(frame_ids)),
                    "source_count": int(batch.carrier_id.shape[0]),
                    "adapter_seconds_d4rt_decode_local": adapter.last_infer_diagnostics.get("seconds_d4rt_decode_local"),
                    "adapter_seconds_d4rt_decode_ref": adapter.last_infer_diagnostics.get("seconds_d4rt_decode_ref"),
                }
                row.update(
                    _branch_row(
                        stream=stream,
                        branch_name=branch_name,
                        xyz_transform=xyz_transform,
                        xyz=transformed_xyz,
                        uv=np.asarray(batch.uv_pred, dtype=np.float32),
                        valid=np.asarray(batch.valid, dtype=bool),
                        visibility=np.asarray(batch.visibility_prob, dtype=np.float32),
                        confidence=np.asarray(batch.confidence_prob, dtype=np.float32),
                        sources=sources,
                        frame_ids=frame_ids,
                        min_visibility=float(args.min_visibility),
                        min_confidence=float(args.min_confidence),
                        max_anchors=int(args.max_anchors),
                        gt_stride=int(args.gt_depth_stride),
                        max_gt_points_per_frame=int(args.max_gt_points_per_frame),
                    )
                )
                rows.append(row)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(prefix.with_suffix(".csv"), rows)
    prefix.with_suffix(".json").write_text(json.dumps(_json_safe(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(prefix.with_suffix(".md"), rows)
    print(json.dumps(_json_safe({"rows": rows}), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
