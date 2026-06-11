from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider, _apply_fit
from geometry_provider.common import backproject_xy_world, fit_transform
from stream4d.scannet_stream import ScanNetStream


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    cache_root: str
    provider_mode: str
    first_window_only: bool = False
    eval_per_chunk: bool = False
    point_mode: str = "xyz"
    depth_calibration: str = "none"
    xyz_field: str = "xyz_ref"
    xyz_transform: str = "raw"
    notes: str = ""


VARIANTS: list[VariantSpec] = [
    VariantSpec("R0", "D4RT single-chunk raw", "outputs/stream4d_debug_full_32f_ioc075_fixmem", "raw", first_window_only=True),
    VariantSpec("R1", "D4RT sliding-window raw", "outputs/stream4d_debug_scene0050_128f_ioc075_fixmem", "raw"),
    VariantSpec("R2", "D4RT sliding-window self-Sim3", "outputs/stream4d_debug_scene0050_128f_ioc075_fixmem", "self_stitched"),
    VariantSpec(
        "R3",
        "D4RT sliding-window scale-normalized self-Sim3",
        "outputs/stream4d_debug_scene0050_128f_ioc075_fixmem",
        "self_stitched_scale_normalized",
    ),
    VariantSpec("R4", "D4RT eval-only scene Sim3", "outputs/stream4d_debug_full_32f_ioc075_fixmem", "eval_sim3"),
    VariantSpec(
        "R5",
        "D4RT eval-only per-chunk Sim3",
        "outputs/stream4d_debug_full_32f_ioc075_fixmem",
        "raw",
        eval_per_chunk=True,
    ),
    VariantSpec(
        "R6",
        "D4RT occupancy-dense warmstart tubes",
        "outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1",
        "raw",
        notes="D5 warmstart cache; raw D4RT coordinates unless paired with eval rows.",
    ),
    VariantSpec(
        "R7",
        "D4RT dense128 grid fixed tubes",
        "outputs/stream4d_debug_v21_3_dense128_grid_probe5_merged_r1",
        "raw",
    ),
    VariantSpec(
        "R8",
        "D4RT mask-aware fixed D2r4 tubes",
        "outputs/stream4d_debug_v21_3_occupancy_d2r4_probe5_r1",
        "raw",
    ),
    VariantSpec(
        "R9",
        "D4RT mask-aware occupancy/warmstart tubes",
        "outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1",
        "self_stitched",
        notes="Uses available D5 cache as closest v21.3 mask-aware occupancy tube artifact.",
    ),
    VariantSpec(
        "R10",
        "D4RT single-chunk UV+Z camera backprojection",
        "outputs/stream4d_debug_full_32f_ioc075_fixmem",
        "raw",
        first_window_only=True,
        point_mode="uvz_camera",
        notes="Diagnostic-only: rebuilds camera-space xyz from D4RT uv and z with ScanNet intrinsics; no depth/pose used for prediction.",
    ),
    VariantSpec(
        "R11",
        "D4RT D2r4 UV+Z camera backprojection",
        "outputs/stream4d_debug_v21_3_occupancy_d2r4_probe5_r1",
        "raw",
        point_mode="uvz_camera",
        notes="Diagnostic-only: rebuilds camera-space xyz from D4RT uv and z with ScanNet intrinsics; no depth/pose used for prediction.",
    ),
    VariantSpec(
        "R12",
        "D4RT single-chunk UV+Z + eval-only median depth calibration",
        "outputs/stream4d_debug_full_32f_ioc075_fixmem",
        "raw",
        first_window_only=True,
        point_mode="uvz_camera",
        depth_calibration="median",
        notes="Diagnostic-only and forbidden for method table: uses ScanNet depth to median-scale D4RT z before UV+Z camera backprojection.",
    ),
    VariantSpec(
        "R13",
        "D4RT single-chunk UV+Z + eval-only linear depth calibration",
        "outputs/stream4d_debug_full_32f_ioc075_fixmem",
        "raw",
        first_window_only=True,
        point_mode="uvz_camera",
        depth_calibration="linear",
        notes="Diagnostic-only and forbidden for method table: uses ScanNet depth to fit z scale/shift before UV+Z camera backprojection.",
    ),
    VariantSpec(
        "R14",
        "D4RT D2r4 UV+Z + eval-only median depth calibration",
        "outputs/stream4d_debug_v21_3_occupancy_d2r4_probe5_r1",
        "raw",
        point_mode="uvz_camera",
        depth_calibration="median",
        notes="Diagnostic-only and forbidden for method table: uses ScanNet depth to median-scale D2r4 z before UV+Z camera backprojection.",
    ),
    VariantSpec(
        "R15",
        "D4RT D2r4 UV+Z + eval-only linear depth calibration",
        "outputs/stream4d_debug_v21_3_occupancy_d2r4_probe5_r1",
        "raw",
        point_mode="uvz_camera",
        depth_calibration="linear",
        notes="Diagnostic-only and forbidden for method table: uses ScanNet depth to fit D2r4 z scale/shift before UV+Z camera backprojection.",
    ),
    VariantSpec(
        "R16",
        "D4RT xyz_local raw",
        "outputs/stream4d_debug_v22_local_xyz_probe5_r1",
        "raw",
        xyz_field="xyz_local",
        notes="Diagnostic-only: uses target-frame pred_local xyz saved by v22.3.",
    ),
    VariantSpec(
        "R17",
        "D4RT xyz_local signed-log1p",
        "outputs/stream4d_debug_v22_local_xyz_probe5_r1",
        "raw",
        xyz_field="xyz_local",
        xyz_transform="signed_log1p",
        notes="Diagnostic-only: applies OpenD4RT loss-space sign(x)*log1p(abs(x)) hypothesis to xyz_local.",
    ),
    VariantSpec(
        "R18",
        "D4RT xyz_local UV+Z camera backprojection",
        "outputs/stream4d_debug_v22_local_xyz_probe5_r1",
        "raw",
        point_mode="uvz_camera",
        xyz_field="xyz_local",
        notes="Diagnostic-only: rebuilds camera-space xyz from uv and xyz_local z.",
    ),
    VariantSpec(
        "R19",
        "D4RT xyz_local signed-log1p UV+Z camera backprojection",
        "outputs/stream4d_debug_v22_local_xyz_probe5_r1",
        "raw",
        point_mode="uvz_camera",
        xyz_field="xyz_local",
        xyz_transform="signed_log1p",
        notes="Diagnostic-only: rebuilds camera-space xyz from uv and signed-log1p xyz_local z.",
    ),
    VariantSpec(
        "R20",
        "D4RT xyz_local eval-only scene Sim3",
        "outputs/stream4d_debug_v22_local_xyz_probe5_r1",
        "eval_sim3",
        xyz_field="xyz_local",
        notes="Diagnostic-only and forbidden for method table: fits ScanNet depth/pose scene Sim3 on xyz_local.",
    ),
    VariantSpec(
        "R21",
        "D4RT xyz_local signed-log1p eval-only scene Sim3",
        "outputs/stream4d_debug_v22_local_xyz_probe5_r1",
        "eval_sim3",
        xyz_field="xyz_local",
        xyz_transform="signed_log1p",
        notes="Diagnostic-only and forbidden for method table: fits ScanNet depth/pose scene Sim3 on signed-log1p xyz_local.",
    ),
    VariantSpec(
        "R22",
        "D4RT xyz_ref0 + ScanNet ref0 pose",
        "outputs/stream4d_debug_full_32f_ioc075_fixmem",
        "ref0_pose",
        notes="Diagnostic-only and forbidden for method table: maps xyz_ref0 through the ScanNet pose of each window ref0 frame.",
    ),
    VariantSpec(
        "R23",
        "D4RT xyz_ref0 + ScanNet ref0 pose + eval-only scale",
        "outputs/stream4d_debug_full_32f_ioc075_fixmem",
        "ref0_pose_scale",
        notes="Diagnostic-only and forbidden for method table: fixes ref0 pose and fits only global scale from ScanNet depth/pose anchors.",
    ),
    VariantSpec(
        "R24",
        "D4RT xyz_ref0 + ref0 pose + local/ref median-norm scale",
        "outputs/stream4d_debug_v22_local_xyz_probe5_r1",
        "ref0_pose_scale_local_median_norm",
        notes="Diagnostic-only: uses ScanNet ref0 pose but estimates scale from D4RT xyz_local/xyz_ref median norms, without ScanNet depth scale fitting.",
    ),
    VariantSpec(
        "R25",
        "D4RT xyz_ref0 + ref0 pose + local/ref RMS-norm scale",
        "outputs/stream4d_debug_v22_local_xyz_probe5_r1",
        "ref0_pose_scale_local_rms_norm",
        notes="Diagnostic-only: uses ScanNet ref0 pose but estimates scale from D4RT xyz_local/xyz_ref RMS norms, without ScanNet depth scale fitting.",
    ),
    VariantSpec(
        "R26",
        "D4RT xyz_ref0 + ref0 pose + source-frame z scale",
        "outputs/stream4d_debug_v22_local_xyz_probe5_r1",
        "ref0_pose_scale_source_z",
        notes="Diagnostic-only: uses ScanNet ref0 pose but estimates scale from source-frame D4RT xyz_local/xyz_ref z statistics, without ScanNet depth scale fitting.",
    ),
    VariantSpec(
        "R27",
        "D4RT xyz_ref0 + ref0 pose + pose-trajectory scale",
        "outputs/stream4d_debug_v22_local_xyz_probe5_r1",
        "ref0_pose_scale_pose_trajectory",
        notes="Diagnostic-only and forbidden for method table: estimates scale from ScanNet pose trajectory length divided by D4RT ref/local rigid translation length, without ScanNet depth scale fitting.",
    ),
]


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


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _sample_rows(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points)
    if points.shape[0] <= int(max_points):
        return points
    keep = np.linspace(0, points.shape[0] - 1, num=int(max_points), dtype=np.int64)
    return points[keep]


def _load_gt_points(
    stream: ScanNetStream,
    frame_ids: list[int],
    *,
    depth_sample_stride: int,
    max_gt_points_per_scene: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    inst_parts: list[np.ndarray] = []
    intr = stream.load_intrinsics()
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    for frame_id in frame_ids:
        depth = stream.load_depth(int(frame_id))
        pose = stream.load_pose(int(frame_id))
        if not np.isfinite(pose).all():
            continue
        h, w = depth.shape[:2]
        yy, xx = np.mgrid[0:h:int(depth_sample_stride), 0:w:int(depth_sample_stride)]
        z = depth[yy, xx].reshape(-1).astype(np.float32)
        x = xx.reshape(-1).astype(np.float32)
        y = yy.reshape(-1).astype(np.float32)
        valid = np.isfinite(z) & (z > 0.0)
        if not np.any(valid):
            continue
        cam = np.stack(
            [(x[valid] - cx) * z[valid] / fx, (y[valid] - cy) * z[valid] / fy, z[valid], np.ones_like(z[valid])],
            axis=1,
        )
        world = (pose @ cam.T).T[:, :3].astype(np.float32)
        finite = np.isfinite(world).all(axis=1)
        if not np.any(finite):
            continue
        pts_parts.append(world[finite])
        frame_parts.append(np.full((int(np.count_nonzero(finite)),), int(frame_id), dtype=np.int64))
        inst_path = stream.root / "instance" / "instance" / f"{int(frame_id)}.png"
        if inst_path.exists():
            import cv2

            inst = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED)
            if inst is not None:
                inst_ids = inst[yy.reshape(-1)[valid], xx.reshape(-1)[valid]].astype(np.int64)[finite]
            else:
                inst_ids = np.full((int(np.count_nonzero(finite)),), -1, dtype=np.int64)
        else:
            inst_ids = np.full((int(np.count_nonzero(finite)),), -1, dtype=np.int64)
        inst_parts.append(inst_ids)
    if not pts_parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    pts = np.concatenate(pts_parts, axis=0)
    frames = np.concatenate(frame_parts, axis=0)
    inst = np.concatenate(inst_parts, axis=0)
    if pts.shape[0] > int(max_gt_points_per_scene):
        keep = np.linspace(0, pts.shape[0] - 1, num=int(max_gt_points_per_scene), dtype=np.int64)
        pts = pts[keep]
        frames = frames[keep]
        inst = inst[keep]
    return pts, frames, inst


def _load_gt_camera_points_by_frame(
    stream: ScanNetStream,
    frame_ids: list[int],
    *,
    depth_sample_stride: int,
    max_points_per_frame: int,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    intr = stream.load_intrinsics()
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    for frame_id in frame_ids:
        depth = stream.load_depth(int(frame_id))
        h, w = depth.shape[:2]
        yy, xx = np.mgrid[0:h:int(depth_sample_stride), 0:w:int(depth_sample_stride)]
        z = depth[yy, xx].reshape(-1).astype(np.float32)
        x = xx.reshape(-1).astype(np.float32)
        y = yy.reshape(-1).astype(np.float32)
        valid = np.isfinite(z) & (z > 0.0)
        if not np.any(valid):
            out[int(frame_id)] = np.empty((0, 3), dtype=np.float32)
            continue
        cam = np.stack(
            [(x[valid] - cx) * z[valid] / fx, (y[valid] - cy) * z[valid] / fy, z[valid]],
            axis=1,
        ).astype(np.float32)
        if cam.shape[0] > int(max_points_per_frame):
            keep = np.linspace(0, cam.shape[0] - 1, num=int(max_points_per_frame), dtype=np.int64)
            cam = cam[keep]
        out[int(frame_id)] = cam
    return out


def _sample_indices(indices: np.ndarray, max_count: int) -> np.ndarray:
    if indices.shape[0] <= int(max_count):
        return indices
    keep = np.linspace(0, indices.shape[0] - 1, num=int(max_count), dtype=np.int64)
    return indices[keep]


def _transform_xyz_hypothesis(xyz: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(xyz, dtype=np.float32)
    if mode == "raw":
        return arr
    if mode == "signed_log1p":
        return (np.sign(arr) * np.log1p(np.abs(arr))).astype(np.float32)
    if mode == "signed_expm1":
        clipped = np.minimum(np.abs(arr), 20.0)
        return (np.sign(arr) * np.expm1(clipped)).astype(np.float32)
    raise ValueError(f"Unsupported xyz transform: {mode}")


def _apply_variant_xyz_to_windows(windows: list[Any], spec: VariantSpec) -> None:
    if spec.xyz_field == "xyz_ref" and spec.xyz_transform == "raw":
        return
    if spec.xyz_field not in {"xyz_ref", "xyz_local"}:
        raise ValueError(f"Unsupported xyz_field: {spec.xyz_field}")
    for window in windows:
        with np.load(window.path) as data:
            if spec.xyz_field not in data.files:
                raise KeyError(f"{window.path} does not contain {spec.xyz_field}")
            xyz = np.asarray(data[spec.xyz_field], dtype=np.float32)
        window.xyz = _transform_xyz_hypothesis(xyz, spec.xyz_transform)


def _fit_chunk_transform(stream: ScanNetStream, window: Any, *, robust_trim_percentile: float, max_anchors: int) -> dict[str, Any] | None:
    src_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    per_frame_cap = max(16, int(max_anchors) // max(len(window.frame_ids), 1))
    for local_idx, frame_id in enumerate(window.frame_ids):
        depth = stream.load_depth(int(frame_id))
        h, w = depth.shape[:2]
        uv = np.asarray(window.uv[local_idx], dtype=np.float32)
        xyz = np.asarray(window.xyz[local_idx], dtype=np.float32)
        ok = (
            window.valid[local_idx]
            & (window.visibility[local_idx] >= 0.5)
            & (window.confidence[local_idx] >= 0.5)
            & np.isfinite(uv).all(axis=1)
            & np.isfinite(xyz).all(axis=1)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= 1.0)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= 1.0)
        )
        if not np.any(ok):
            continue
        indices = _sample_indices(np.flatnonzero(ok), per_frame_cap)
        xy = np.stack([uv[indices, 0] * float(max(w - 1, 1)), uv[indices, 1] * float(max(h - 1, 1))], axis=1)
        world, valid = backproject_xy_world(stream, int(frame_id), xy)
        if np.any(valid):
            src_parts.append(xyz[indices][valid])
            tgt_parts.append(world[valid])
    if not src_parts:
        return None
    source = np.concatenate(src_parts, axis=0)
    target = np.concatenate(tgt_parts, axis=0)
    if source.shape[0] > int(max_anchors):
        keep = np.linspace(0, source.shape[0] - 1, num=int(max_anchors), dtype=np.int64)
        source = source[keep]
        target = target[keep]
    return fit_transform(source, target, robust_trim_percentile=float(robust_trim_percentile))


def _fit_scene_transform_sampled(
    stream: ScanNetStream,
    windows: list[Any],
    *,
    robust_trim_percentile: float,
    max_anchors: int,
) -> dict[str, Any] | None:
    src_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    frame_slots = sum(len(window.frame_ids) for window in windows)
    per_frame_cap = max(16, int(max_anchors) // max(frame_slots, 1))
    for window in windows:
        for local_idx, frame_id in enumerate(window.frame_ids):
            depth = stream.load_depth(int(frame_id))
            h, w = depth.shape[:2]
            uv = np.asarray(window.uv[local_idx], dtype=np.float32)
            xyz = np.asarray(window.xyz[local_idx], dtype=np.float32)
            ok = (
                window.valid[local_idx]
                & (window.visibility[local_idx] >= 0.5)
                & (window.confidence[local_idx] >= 0.5)
                & np.isfinite(uv).all(axis=1)
                & np.isfinite(xyz).all(axis=1)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
            )
            if not np.any(ok):
                continue
            indices = _sample_indices(np.flatnonzero(ok), per_frame_cap)
            xy = np.stack([uv[indices, 0] * float(max(w - 1, 1)), uv[indices, 1] * float(max(h - 1, 1))], axis=1)
            world, valid = backproject_xy_world(stream, int(frame_id), xy)
            if np.any(valid):
                src_parts.append(xyz[indices][valid])
                tgt_parts.append(world[valid])
    if not src_parts:
        return None
    return fit_transform(np.concatenate(src_parts, axis=0), np.concatenate(tgt_parts, axis=0), robust_trim_percentile=float(robust_trim_percentile))


def _pose_fit(pose: np.ndarray, *, scale: float = 1.0, residual: np.ndarray | None = None, anchor_count: int = 0) -> dict[str, Any] | None:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    if residual is None:
        residual = np.empty((0,), dtype=np.float64)
    return {
        "scale": float(scale),
        "rotation": pose[:3, :3].astype(np.float64),
        "translation": pose[:3, 3].astype(np.float64),
        "residual": np.asarray(residual, dtype=np.float64),
        "anchor_count": int(anchor_count),
    }


def _fit_ref0_pose_scale(
    stream: ScanNetStream,
    window: Any,
    *,
    robust_trim_percentile: float,
    max_anchors: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not window.frame_ids:
        return None, {"ref0_pose_scale_status": "no_frames"}
    pose0 = stream.load_pose(int(window.frame_ids[0]))
    if not np.isfinite(pose0).all():
        return None, {"ref0_pose_scale_status": "invalid_pose0"}
    pose0_inv = np.linalg.inv(pose0.astype(np.float64))
    src_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    per_frame_cap = max(16, int(max_anchors) // max(len(window.frame_ids), 1))
    for local_idx, frame_id in enumerate(window.frame_ids):
        depth = stream.load_depth(int(frame_id))
        h, w = depth.shape[:2]
        uv = np.asarray(window.uv[local_idx], dtype=np.float32)
        xyz = np.asarray(window.xyz[local_idx], dtype=np.float32)
        ok = (
            window.valid[local_idx]
            & (window.visibility[local_idx] >= 0.5)
            & (window.confidence[local_idx] >= 0.5)
            & np.isfinite(uv).all(axis=1)
            & np.isfinite(xyz).all(axis=1)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= 1.0)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= 1.0)
        )
        if not np.any(ok):
            continue
        indices = _sample_indices(np.flatnonzero(ok), per_frame_cap)
        xy = np.stack([uv[indices, 0] * float(max(w - 1, 1)), uv[indices, 1] * float(max(h - 1, 1))], axis=1)
        world, valid = backproject_xy_world(stream, int(frame_id), xy)
        if not np.any(valid):
            continue
        world_h = np.concatenate([world[valid].astype(np.float64), np.ones((int(np.count_nonzero(valid)), 1), dtype=np.float64)], axis=1)
        target_ref0 = (pose0_inv @ world_h.T).T[:, :3]
        src_parts.append(xyz[indices][valid].astype(np.float64))
        tgt_parts.append(target_ref0.astype(np.float64))
    if not src_parts:
        return None, {"ref0_pose_scale_status": "no_anchors"}
    source = np.concatenate(src_parts, axis=0)
    target = np.concatenate(tgt_parts, axis=0)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < 4:
        return None, {"ref0_pose_scale_status": "too_few_anchors", "ref0_pose_scale_anchor_count": int(source.shape[0])}
    if source.shape[0] > int(max_anchors):
        keep = np.linspace(0, source.shape[0] - 1, num=int(max_anchors), dtype=np.int64)
        source = source[keep]
        target = target[keep]

    def solve_scale(src: np.ndarray, dst: np.ndarray) -> float:
        denom = float(np.sum(src * src))
        if denom <= 1e-12:
            return float("nan")
        return float(np.sum(src * dst) / denom)

    scale = solve_scale(source, target)
    if not np.isfinite(scale) or scale <= 0.0:
        return None, {"ref0_pose_scale_status": "invalid_scale", "ref0_pose_scale_anchor_count": int(source.shape[0])}
    residual = np.linalg.norm(float(scale) * source - target, axis=1)
    trim = float(robust_trim_percentile)
    if 0.0 < trim < 100.0 and residual.size >= 8:
        keep = residual <= float(np.percentile(residual, trim))
        if np.count_nonzero(keep) >= 4 and np.count_nonzero(keep) < residual.size:
            scale = solve_scale(source[keep], target[keep])
            residual = np.linalg.norm(float(scale) * source - target, axis=1)
    fit = _pose_fit(pose0, scale=scale, residual=residual, anchor_count=int(source.shape[0]))
    diag = {
        "ref0_pose_scale_status": "ok",
        "ref0_pose_frame": int(window.frame_ids[0]),
        "ref0_pose_scale": float(scale),
        "ref0_pose_scale_anchor_count": int(source.shape[0]),
        "ref0_pose_scale_residual_median": float(np.median(residual)),
        "ref0_pose_scale_residual_p90": float(np.percentile(residual, 90)),
    }
    return fit, diag


_REF0_LOCAL_SCALE_MODES = {
    "ref0_pose_scale_local_median_norm": "local_median_norm",
    "ref0_pose_scale_local_rms_norm": "local_rms_norm",
    "ref0_pose_scale_source_z": "source_z",
}

_REF0_TRAJECTORY_SCALE_MODE = "ref0_pose_scale_pose_trajectory"


def _estimate_ref0_local_scale(window: Any, *, mode: str, max_anchors: int) -> tuple[float | None, dict[str, Any]]:
    if mode not in set(_REF0_LOCAL_SCALE_MODES.values()):
        return None, {"ref0_local_scale_status": "unsupported_mode", "ref0_local_scale_mode": str(mode)}
    with np.load(window.path) as data:
        if "xyz_local" not in data.files or "xyz_ref" not in data.files:
            return None, {"ref0_local_scale_status": "missing_xyz_local", "ref0_local_scale_mode": str(mode)}
        xyz_ref = np.asarray(data["xyz_ref"], dtype=np.float64)
        xyz_local = np.asarray(data["xyz_local"], dtype=np.float64)
        src_frame = np.asarray(data.get("src_frame", np.zeros((xyz_ref.shape[1],), dtype=np.int64)), dtype=np.int64)
    finite = np.isfinite(xyz_ref).all(axis=-1) & np.isfinite(xyz_local).all(axis=-1)
    valid = np.asarray(window.valid, dtype=bool) & finite
    visibility = np.asarray(window.visibility, dtype=np.float64)
    confidence = np.asarray(window.confidence, dtype=np.float64)
    ok = valid & (visibility >= 0.5) & (confidence >= 0.5)

    if mode in {"local_median_norm", "local_rms_norm"}:
        flat = np.flatnonzero(ok.reshape(-1))
        if flat.size == 0:
            return None, {"ref0_local_scale_status": "no_anchors", "ref0_local_scale_mode": str(mode)}
        flat = _sample_indices(flat, int(max_anchors))
        ref_sel = xyz_ref.reshape(-1, 3)[flat]
        local_sel = xyz_local.reshape(-1, 3)[flat]
        ref_norm = np.linalg.norm(ref_sel, axis=1)
        local_norm = np.linalg.norm(local_sel, axis=1)
        finite_norm = np.isfinite(ref_norm) & np.isfinite(local_norm) & (ref_norm > 1e-8) & (local_norm > 1e-8)
        ref_norm = ref_norm[finite_norm]
        local_norm = local_norm[finite_norm]
        if ref_norm.size < 4:
            return None, {"ref0_local_scale_status": "too_few_anchors", "ref0_local_scale_mode": str(mode), "ref0_local_scale_anchor_count": int(ref_norm.size)}
        if mode == "local_median_norm":
            scale = float(np.median(local_norm) / max(float(np.median(ref_norm)), 1e-12))
        else:
            scale = float(np.sqrt(np.mean(local_norm * local_norm)) / max(float(np.sqrt(np.mean(ref_norm * ref_norm))), 1e-12))
        diag = {
            "ref0_local_scale_status": "ok",
            "ref0_local_scale_mode": str(mode),
            "ref0_local_scale": float(scale),
            "ref0_local_scale_anchor_count": int(ref_norm.size),
            "ref0_local_scale_ref_norm_median": float(np.median(ref_norm)),
            "ref0_local_scale_local_norm_median": float(np.median(local_norm)),
        }
        return scale, diag

    num_frames, num_points = xyz_ref.shape[:2]
    q = np.arange(num_points, dtype=np.int64)
    local_idx = np.clip(src_frame.reshape(-1)[:num_points], 0, max(num_frames - 1, 0))
    src_ok = ok[local_idx, q]
    q = q[src_ok]
    local_idx = local_idx[src_ok]
    if q.size == 0:
        return None, {"ref0_local_scale_status": "no_source_anchors", "ref0_local_scale_mode": str(mode)}
    keep = _sample_indices(np.arange(q.shape[0], dtype=np.int64), int(max_anchors))
    q = q[keep]
    local_idx = local_idx[keep]
    ref_z = np.abs(xyz_ref[local_idx, q, 2])
    local_z = np.abs(xyz_local[local_idx, q, 2])
    finite_z = np.isfinite(ref_z) & np.isfinite(local_z) & (ref_z > 1e-8) & (local_z > 1e-8)
    ref_z = ref_z[finite_z]
    local_z = local_z[finite_z]
    if ref_z.size < 4:
        return None, {"ref0_local_scale_status": "too_few_source_anchors", "ref0_local_scale_mode": str(mode), "ref0_local_scale_anchor_count": int(ref_z.size)}
    scale = float(np.median(local_z) / max(float(np.median(ref_z)), 1e-12))
    diag = {
        "ref0_local_scale_status": "ok",
        "ref0_local_scale_mode": str(mode),
        "ref0_local_scale": float(scale),
        "ref0_local_scale_anchor_count": int(ref_z.size),
        "ref0_local_scale_ref_z_median": float(np.median(ref_z)),
        "ref0_local_scale_local_z_median": float(np.median(local_z)),
    }
    return scale, diag


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


def _estimate_ref0_pose_trajectory_scale(
    stream: ScanNetStream,
    window: Any,
    *,
    max_anchors: int,
) -> tuple[float | None, dict[str, Any]]:
    if not window.frame_ids:
        return None, {"ref0_trajectory_scale_status": "no_frames"}
    pose0 = stream.load_pose(int(window.frame_ids[0]))
    if not np.isfinite(pose0).all():
        return None, {"ref0_trajectory_scale_status": "invalid_pose0"}
    with np.load(window.path) as data:
        if "xyz_local" not in data.files:
            return None, {"ref0_trajectory_scale_status": "missing_xyz_local"}
        xyz_local = np.asarray(data["xyz_local"], dtype=np.float64)
    if xyz_local.shape != np.asarray(window.xyz).shape:
        return None, {"ref0_trajectory_scale_status": "shape_mismatch"}

    ratios: list[float] = []
    d4rt_trans: list[float] = []
    pose_trans: list[float] = []
    residual_p90: list[float] = []
    per_frame_cap = max(4, int(max_anchors) // max(len(window.frame_ids) - 1, 1))
    for local_idx, frame_id in enumerate(window.frame_ids):
        if local_idx == 0:
            continue
        pose = stream.load_pose(int(frame_id))
        if not np.isfinite(pose).all():
            continue
        ok = (
            np.asarray(window.valid[local_idx], dtype=bool)
            & np.isfinite(window.xyz[local_idx]).all(axis=1)
            & np.isfinite(xyz_local[local_idx]).all(axis=1)
            & (np.asarray(window.visibility[local_idx], dtype=np.float64) >= 0.5)
            & (np.asarray(window.confidence[local_idx], dtype=np.float64) >= 0.5)
            & np.isfinite(window.uv[local_idx]).all(axis=1)
            & (window.uv[local_idx, :, 0] >= 0.0)
            & (window.uv[local_idx, :, 0] <= 1.0)
            & (window.uv[local_idx, :, 1] >= 0.0)
            & (window.uv[local_idx, :, 1] <= 1.0)
        )
        indices = np.flatnonzero(ok)
        if indices.shape[0] < 4:
            continue
        indices = _sample_indices(indices, per_frame_cap)
        try:
            _, trans, residual = _fit_rigid_no_scale(window.xyz[local_idx, indices], xyz_local[local_idx, indices])
        except Exception:
            continue
        d4rt_len = float(np.linalg.norm(trans))
        pose_len = float(np.linalg.norm(np.asarray(pose[:3, 3], dtype=np.float64) - np.asarray(pose0[:3, 3], dtype=np.float64)))
        if d4rt_len <= 1e-6 or pose_len <= 1e-6:
            continue
        ratios.append(pose_len / d4rt_len)
        d4rt_trans.append(d4rt_len)
        pose_trans.append(pose_len)
        residual_p90.append(float(np.percentile(residual, 90)))

    if len(ratios) < 2:
        return None, {
            "ref0_trajectory_scale_status": "too_few_frames",
            "ref0_trajectory_scale_frame_count": int(len(ratios)),
        }
    values = np.asarray(ratios, dtype=np.float64)
    scale = float(np.median(values))
    diag = {
        "ref0_trajectory_scale_status": "ok",
        "ref0_trajectory_scale": scale,
        "ref0_trajectory_scale_frame_count": int(values.shape[0]),
        "ref0_trajectory_scale_mean": float(np.mean(values)),
        "ref0_trajectory_scale_std": float(np.std(values)),
        "ref0_trajectory_scale_min": float(np.min(values)),
        "ref0_trajectory_scale_max": float(np.max(values)),
        "ref0_trajectory_d4rt_translation_median": float(np.median(np.asarray(d4rt_trans, dtype=np.float64))),
        "ref0_trajectory_pose_translation_median": float(np.median(np.asarray(pose_trans, dtype=np.float64))),
        "ref0_trajectory_rigid_residual_p90_median": float(np.median(np.asarray(residual_p90, dtype=np.float64))),
    }
    return scale, diag


def _load_windows(
    spec: VariantSpec,
    scene: str,
    *,
    nn_radius: float,
    density_alpha: float,
    robust_trim_percentile: float,
    max_anchors: int,
    max_windows_per_scene: int | None,
) -> tuple[list[Any], dict[str, Any]]:
    if spec.provider_mode.startswith("self_stitched"):
        raw_provider = D4RTCarrierProjectionProvider(
            debug_root=spec.cache_root,
            mode="raw",
            nn_radius=nn_radius,
            density_alpha=density_alpha,
            robust_trim_percentile=robust_trim_percentile,
            max_anchors=max_anchors,
        )
        cache = raw_provider._load_scene(scene)
        windows = list(cache["windows"])
        if spec.first_window_only:
            windows = windows[:1]
        if max_windows_per_scene is not None:
            windows = windows[: int(max_windows_per_scene)]
        _apply_variant_xyz_to_windows(windows, spec)
        stitch_provider = D4RTCarrierProjectionProvider(
            debug_root=spec.cache_root,
            mode=spec.provider_mode,
            nn_radius=nn_radius,
            density_alpha=density_alpha,
            robust_trim_percentile=robust_trim_percentile,
            max_anchors=max_anchors,
        )
        transforms, stitch_diag = stitch_provider._self_stitch_transforms(windows)
        for window, transform in zip(windows, transforms):
            window.transform = transform
        cache["stitch_diag"] = stitch_diag
        return windows, cache
    if spec.provider_mode == "eval_sim3":
        raw_provider = D4RTCarrierProjectionProvider(
            debug_root=spec.cache_root,
            mode="raw",
            nn_radius=nn_radius,
            density_alpha=density_alpha,
            robust_trim_percentile=robust_trim_percentile,
            max_anchors=max_anchors,
        )
        cache = raw_provider._load_scene(scene)
        windows = list(cache["windows"])
        if spec.first_window_only:
            windows = windows[:1]
        if max_windows_per_scene is not None:
            windows = windows[: int(max_windows_per_scene)]
        _apply_variant_xyz_to_windows(windows, spec)
        stream = ScanNetStream(seq_name=scene, backbone="Cropformer")
        scene_fit = _fit_scene_transform_sampled(
            stream,
            windows,
            robust_trim_percentile=robust_trim_percentile,
            max_anchors=max_anchors,
        )
        for window in windows:
            window.transform = scene_fit
        cache["scene_fit"] = scene_fit
        return windows, cache
    if spec.provider_mode in {"ref0_pose", "ref0_pose_scale", _REF0_TRAJECTORY_SCALE_MODE} or spec.provider_mode in _REF0_LOCAL_SCALE_MODES:
        raw_provider = D4RTCarrierProjectionProvider(
            debug_root=spec.cache_root,
            mode="raw",
            nn_radius=nn_radius,
            density_alpha=density_alpha,
            robust_trim_percentile=robust_trim_percentile,
            max_anchors=max_anchors,
        )
        cache = raw_provider._load_scene(scene)
        windows = list(cache["windows"])
        if spec.first_window_only:
            windows = windows[:1]
        if max_windows_per_scene is not None:
            windows = windows[: int(max_windows_per_scene)]
        _apply_variant_xyz_to_windows(windows, spec)
        stream = ScanNetStream(seq_name=scene, backbone="Cropformer")
        alignment_diag: dict[str, Any] = {}
        for idx, window in enumerate(windows):
            if spec.provider_mode == "ref0_pose":
                pose0 = stream.load_pose(int(window.frame_ids[0])) if window.frame_ids else np.full((4, 4), np.nan)
                window.transform = _pose_fit(pose0)
                if idx == 0:
                    alignment_diag.update({"ref0_pose_status": "ok" if window.transform is not None else "failed", "ref0_pose_frame": int(window.frame_ids[0]) if window.frame_ids else -1})
            elif spec.provider_mode == "ref0_pose_scale":
                fit, diag = _fit_ref0_pose_scale(
                    stream,
                    window,
                    robust_trim_percentile=robust_trim_percentile,
                    max_anchors=max_anchors,
                )
                window.transform = fit
                if idx == 0:
                    alignment_diag.update(diag)
            elif spec.provider_mode == _REF0_TRAJECTORY_SCALE_MODE:
                pose0 = stream.load_pose(int(window.frame_ids[0])) if window.frame_ids else np.full((4, 4), np.nan)
                scale, diag = _estimate_ref0_pose_trajectory_scale(
                    stream,
                    window,
                    max_anchors=max_anchors,
                )
                window.transform = _pose_fit(pose0, scale=float(scale)) if scale is not None else None
                if idx == 0:
                    alignment_diag.update(
                        {
                            "ref0_pose_status": "ok" if np.isfinite(pose0).all() else "failed",
                            "ref0_pose_frame": int(window.frame_ids[0]) if window.frame_ids else -1,
                            **diag,
                        }
                    )
            else:
                pose0 = stream.load_pose(int(window.frame_ids[0])) if window.frame_ids else np.full((4, 4), np.nan)
                scale, diag = _estimate_ref0_local_scale(
                    window,
                    mode=_REF0_LOCAL_SCALE_MODES[spec.provider_mode],
                    max_anchors=max_anchors,
                )
                window.transform = _pose_fit(pose0, scale=float(scale)) if scale is not None else None
                if idx == 0:
                    alignment_diag.update(
                        {
                            "ref0_pose_status": "ok" if np.isfinite(pose0).all() else "failed",
                            "ref0_pose_frame": int(window.frame_ids[0]) if window.frame_ids else -1,
                            **diag,
                        }
                    )
        cache["alignment_diag"] = alignment_diag
        return windows, cache
    provider = D4RTCarrierProjectionProvider(
        debug_root=spec.cache_root,
        mode=spec.provider_mode,
        nn_radius=nn_radius,
        density_alpha=density_alpha,
        robust_trim_percentile=robust_trim_percentile,
        max_anchors=max_anchors,
    )
    cache = provider._load_scene(scene)
    windows = list(cache["windows"])
    if spec.first_window_only:
        windows = windows[:1]
    if max_windows_per_scene is not None:
        windows = windows[: int(max_windows_per_scene)]
    _apply_variant_xyz_to_windows(windows, spec)
    if spec.eval_per_chunk:
        stream = ScanNetStream(seq_name=scene, backbone="Cropformer")
        for window in windows:
            window.transform = _fit_chunk_transform(
                stream,
                window,
                robust_trim_percentile=robust_trim_percentile,
                max_anchors=max_anchors,
            )
    return windows, cache


def _window_uv_xyz_indices(
    window: Any,
    local_idx: int,
    *,
    min_visibility: float,
    min_confidence: float,
) -> np.ndarray:
    uv = np.asarray(window.uv[local_idx], dtype=np.float32)
    xyz = np.asarray(window.xyz[local_idx], dtype=np.float32)
    ok = (
        window.valid[local_idx]
        & (window.visibility[local_idx] >= float(min_visibility))
        & (window.confidence[local_idx] >= float(min_confidence))
        & np.isfinite(uv).all(axis=1)
        & np.isfinite(xyz).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= 1.0)
    )
    return np.flatnonzero(ok)


def _fit_depth_calibration(
    stream: ScanNetStream,
    windows: list[Any],
    *,
    min_visibility: float,
    min_confidence: float,
    max_anchors: int,
) -> dict[str, Any] | None:
    raw_depths: list[np.ndarray] = []
    gt_depths: list[np.ndarray] = []
    frame_slots = sum(len(window.frame_ids) for window in windows)
    per_frame_cap = max(16, int(max_anchors) // max(frame_slots, 1))
    for window in windows:
        for local_idx, frame_id in enumerate(window.frame_ids):
            depth = stream.load_depth(int(frame_id))
            h, w = depth.shape[:2]
            uv = np.asarray(window.uv[local_idx], dtype=np.float32)
            xyz = _apply_fit(window.xyz[local_idx], window.transform)
            idx = _window_uv_xyz_indices(
                window,
                local_idx,
                min_visibility=float(min_visibility),
                min_confidence=float(min_confidence),
            )
            if idx.size == 0:
                continue
            idx = _sample_indices(idx, per_frame_cap)
            x = np.rint(uv[idx, 0] * float(max(w - 1, 1))).astype(np.int64)
            y = np.rint(uv[idx, 1] * float(max(h - 1, 1))).astype(np.int64)
            in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            if not np.any(in_bounds):
                continue
            pred_z = xyz[idx[in_bounds], 2].astype(np.float64)
            gt_z = depth[y[in_bounds], x[in_bounds]].astype(np.float64)
            valid = np.isfinite(pred_z) & np.isfinite(gt_z) & (pred_z > 1e-6) & (gt_z > 1e-6)
            if not np.any(valid):
                continue
            raw_depths.append(pred_z[valid])
            gt_depths.append(gt_z[valid])
    if not raw_depths:
        return None
    raw = np.concatenate(raw_depths, axis=0)
    gt = np.concatenate(gt_depths, axis=0)
    if raw.shape[0] > int(max_anchors):
        keep = np.linspace(0, raw.shape[0] - 1, num=int(max_anchors), dtype=np.int64)
        raw = raw[keep]
        gt = gt[keep]
    scale = float(np.median(gt / raw))
    design = np.stack([raw, np.ones(raw.shape[0], dtype=np.float64)], axis=1)
    linear_scale, linear_shift = np.linalg.lstsq(design, gt, rcond=None)[0]
    out: dict[str, Any] = {
        "anchor_count": int(raw.shape[0]),
        "median_scale": scale,
        "linear_scale": float(linear_scale),
        "linear_shift": float(linear_shift),
    }
    out.update({f"raw_{key}": value for key, value in _depth_errors(raw, gt).items()})
    out.update({f"median_{key}": value for key, value in _depth_errors(raw * scale, gt).items()})
    out.update({f"linear_{key}": value for key, value in _depth_errors(raw * float(linear_scale) + float(linear_shift), gt).items()})
    return out


def _apply_depth_calibration(z: np.ndarray, fit: dict[str, Any] | None, mode: str) -> np.ndarray:
    depth = np.asarray(z, dtype=np.float32)
    if fit is None or mode == "none":
        return depth
    if mode == "median":
        return (depth.astype(np.float64) * float(fit["median_scale"])).astype(np.float32)
    if mode == "linear":
        return (depth.astype(np.float64) * float(fit["linear_scale"]) + float(fit["linear_shift"])).astype(np.float32)
    raise ValueError(f"Unsupported depth calibration mode: {mode}")


def _raw_uvz_signal_metrics(
    stream: ScanNetStream,
    windows: list[Any],
    *,
    min_visibility: float,
    min_confidence: float,
    max_anchors: int,
) -> dict[str, Any]:
    intr = stream.load_intrinsics()
    fx = float(intr[0, 0])
    fy = float(intr[1, 1])
    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    z_parts: list[np.ndarray] = []
    uv_err_parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    uv_pix_parts: list[np.ndarray] = []
    frame_slots = sum(len(window.frame_ids) for window in windows)
    per_frame_cap = max(16, int(max_anchors) // max(frame_slots, 1))
    for window in windows:
        for local_idx, frame_id in enumerate(window.frame_ids):
            depth = stream.load_depth(int(frame_id))
            h, w = depth.shape[:2]
            uv = np.asarray(window.uv[local_idx], dtype=np.float32)
            xyz = np.asarray(window.xyz[local_idx], dtype=np.float32)
            idx = _window_uv_xyz_indices(
                window,
                local_idx,
                min_visibility=float(min_visibility),
                min_confidence=float(min_confidence),
            )
            if idx.size == 0:
                continue
            idx = _sample_indices(idx, per_frame_cap)
            raw = xyz[idx].astype(np.float64)
            uv_sel = uv[idx].astype(np.float64)
            z = raw[:, 2]
            raw_parts.append(raw)
            uv_pix_parts.append(
                np.stack(
                    [
                        uv_sel[:, 0] * float(max(w - 1, 1)),
                        uv_sel[:, 1] * float(max(h - 1, 1)),
                    ],
                    axis=1,
                )
            )
            finite_z = z[np.isfinite(z)]
            if finite_z.size:
                z_parts.append(finite_z)
            positive = np.isfinite(z) & (z > 1e-6)
            if np.any(positive):
                x_proj = raw[positive, 0] * fx / z[positive] + cx
                y_proj = raw[positive, 1] * fy / z[positive] + cy
                x_uv = uv_sel[positive, 0] * float(max(w - 1, 1))
                y_uv = uv_sel[positive, 1] * float(max(h - 1, 1))
                err = np.sqrt((x_proj - x_uv) ** 2 + (y_proj - y_uv) ** 2)
                err = err[np.isfinite(err)]
                if err.size:
                    uv_err_parts.append(err.astype(np.float64))
    if not z_parts:
        return {"raw_uvz_anchor_count": 0}
    z_all = np.concatenate(z_parts, axis=0)
    out: dict[str, Any] = {
        "raw_uvz_anchor_count": int(z_all.shape[0]),
        "raw_uvz_positive_z_rate": float(np.mean(z_all > 1e-6)),
        "raw_uvz_z_median": float(np.median(z_all)),
        "raw_uvz_z_p05": float(np.percentile(z_all, 5)),
        "raw_uvz_z_p95": float(np.percentile(z_all, 95)),
    }
    if uv_err_parts:
        err = np.concatenate(uv_err_parts, axis=0)
        out.update(
            {
                "raw_uvz_reproj_error_px_median": float(np.median(err)),
                "raw_uvz_reproj_error_px_p90": float(np.percentile(err, 90)),
                "raw_uvz_reproj_error_px_mean": float(np.mean(err)),
            }
        )
    if raw_parts and uv_pix_parts:
        raw_all = np.concatenate(raw_parts, axis=0)
        uv_pix_all = np.concatenate(uv_pix_parts, axis=0)
        if raw_all.shape[0] > int(max_anchors):
            keep = np.linspace(0, raw_all.shape[0] - 1, num=int(max_anchors), dtype=np.int64)
            raw_all = raw_all[keep]
            uv_pix_all = uv_pix_all[keep]
        best: dict[str, Any] | None = None
        for axes in itertools.permutations((0, 1, 2), 3):
            xi, yi, zi = axes
            for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3):
                depth = sz * raw_all[:, zi]
                valid = np.isfinite(depth) & (depth > 1e-6)
                if np.count_nonzero(valid) < 32:
                    continue
                x_cam = sx * raw_all[valid, xi]
                y_cam = sy * raw_all[valid, yi]
                z_cam = depth[valid]
                x_proj = x_cam * fx / z_cam + cx
                y_proj = y_cam * fy / z_cam + cy
                err = np.sqrt((x_proj - uv_pix_all[valid, 0]) ** 2 + (y_proj - uv_pix_all[valid, 1]) ** 2)
                err = err[np.isfinite(err)]
                if err.size == 0:
                    continue
                candidate = {
                    "median": float(np.median(err)),
                    "p90": float(np.percentile(err, 90)),
                    "mean": float(np.mean(err)),
                    "positive_z_rate": float(np.mean(valid)),
                    "convention": f"{'-' if sx < 0 else '+'}{xi},{'-' if sy < 0 else '+'}{yi},{'-' if sz < 0 else '+'}{zi}",
                }
                if best is None or candidate["median"] < float(best["median"]):
                    best = candidate
        if best is not None:
            out.update(
                {
                    "raw_uvz_best_reproj_convention": best["convention"],
                    "raw_uvz_best_reproj_positive_z_rate": best["positive_z_rate"],
                    "raw_uvz_best_reproj_error_px_median": best["median"],
                    "raw_uvz_best_reproj_error_px_p90": best["p90"],
                    "raw_uvz_best_reproj_error_px_mean": best["mean"],
                }
            )
    return out


def _collect_pred_points(
    windows: list[Any],
    *,
    stream: ScanNetStream | None = None,
    point_mode: str = "xyz",
    depth_calibration_fit: dict[str, Any] | None = None,
    depth_calibration: str = "none",
    max_points_per_frame: int,
    min_visibility: float,
    min_confidence: float,
    min_pred_z: float | None = None,
) -> dict[str, np.ndarray]:
    pts_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    for window in windows:
        for local_idx, frame_id in enumerate(window.frame_ids):
            uv = np.asarray(window.uv[local_idx], dtype=np.float32)
            xyz = _apply_fit(window.xyz[local_idx], window.transform)
            ok = (
                window.valid[local_idx]
                & (window.visibility[local_idx] >= float(min_visibility))
                & (window.confidence[local_idx] >= float(min_confidence))
                & np.isfinite(uv).all(axis=1)
                & np.isfinite(xyz).all(axis=1)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
            )
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                continue
            if idx.size > int(max_points_per_frame):
                keep = np.linspace(0, idx.size - 1, num=int(max_points_per_frame), dtype=np.int64)
                idx = idx[keep]
            points = xyz[idx]
            if point_mode == "uvz_camera":
                if stream is None:
                    raise ValueError("stream is required for uvz_camera point mode")
                intr = stream.load_intrinsics()
                fx = float(intr[0, 0])
                fy = float(intr[1, 1])
                cx = float(intr[0, 2])
                cy = float(intr[1, 2])
                height, width = stream.load_depth(int(frame_id)).shape[:2]
                uv_sel = uv[idx]
                z = _apply_depth_calibration(points[:, 2].astype(np.float32), depth_calibration_fit, depth_calibration)
                z_valid = np.isfinite(z)
                if min_pred_z is not None:
                    z_valid &= z >= float(min_pred_z)
                if not np.any(z_valid):
                    continue
                uv_sel = uv_sel[z_valid]
                idx = idx[z_valid]
                z = z[z_valid]
                x_pix = uv_sel[:, 0] * float(max(width - 1, 1))
                y_pix = uv_sel[:, 1] * float(max(height - 1, 1))
                points = np.stack([(x_pix - cx) * z / fx, (y_pix - cy) * z / fy, z], axis=1).astype(np.float32)
            elif point_mode != "xyz":
                raise ValueError(f"Unsupported point_mode: {point_mode}")
            pts_parts.append(points)
            uv_parts.append(uv[idx])
            frame_parts.append(np.full((idx.shape[0],), int(frame_id), dtype=np.int64))
    if not pts_parts:
        return {
            "points": np.empty((0, 3), dtype=np.float32),
            "uv": np.empty((0, 2), dtype=np.float32),
            "frame_ids": np.empty((0,), dtype=np.int64),
        }
    return {
        "points": np.concatenate(pts_parts, axis=0).astype(np.float32),
        "uv": np.concatenate(uv_parts, axis=0).astype(np.float32),
        "frame_ids": np.concatenate(frame_parts, axis=0).astype(np.int64),
    }


def _point_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    pred = pred[np.isfinite(pred).all(axis=1)]
    gt = gt[np.isfinite(gt).all(axis=1)]
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return {"status": "empty"}
    gt_tree = cKDTree(gt)
    pred_tree = cKDTree(pred)
    pred_to_gt, _ = gt_tree.query(pred, k=1)
    gt_to_pred, _ = pred_tree.query(gt, k=1)
    out: dict[str, Any] = {
        "pred_point_count": int(pred.shape[0]),
        "gt_point_count": int(gt.shape[0]),
        "chamfer_l1": float(np.mean(pred_to_gt) + np.mean(gt_to_pred)),
        "chamfer_l2": float(np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2)),
        "outlier_rate_20cm": float(np.mean(pred_to_gt > 0.20)),
        "outlier_rate_50cm": float(np.mean(pred_to_gt > 0.50)),
        "pred_to_gt_median": float(np.median(pred_to_gt)),
        "pred_to_gt_p90": float(np.percentile(pred_to_gt, 90)),
    }
    for tau in (0.01, 0.05, 0.10, 0.20):
        precision = float(np.mean(pred_to_gt < tau))
        recall = float(np.mean(gt_to_pred < tau))
        out[f"accuracy@{int(tau * 100)}cm"] = precision
        out[f"completeness@{int(tau * 100)}cm"] = recall
        out[f"precision@{int(tau * 100)}cm"] = precision
        out[f"recall@{int(tau * 100)}cm"] = recall
        out[f"fscore@{int(tau * 100)}cm"] = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    out["_pred_to_gt_dist"] = pred_to_gt.astype(np.float32)
    return out


def _camera_space_point_metrics(
    stream: ScanNetStream,
    pred: dict[str, np.ndarray],
    *,
    points_are_world: bool,
    depth_sample_stride: int,
    max_gt_points_per_scene: int,
) -> dict[str, Any]:
    frame_ids = sorted({int(v) for v in np.asarray(pred["frame_ids"], dtype=np.int64).tolist()})
    if not frame_ids:
        return {"camera_space_frame_count": 0}
    per_frame_cap = max(32, int(max_gt_points_per_scene) // max(len(frame_ids), 1))
    gt_by_frame = _load_gt_camera_points_by_frame(
        stream,
        frame_ids,
        depth_sample_stride=int(depth_sample_stride),
        max_points_per_frame=per_frame_cap,
    )
    pose_cache: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    total_pred = 0
    total_gt = 0
    for frame_id in frame_ids:
        pred_idx = np.flatnonzero(np.asarray(pred["frame_ids"], dtype=np.int64) == int(frame_id))
        if pred_idx.size == 0:
            continue
        pred_points = np.asarray(pred["points"][pred_idx], dtype=np.float32)
        if points_are_world:
            if frame_id not in pose_cache:
                pose_cache[frame_id] = np.linalg.inv(stream.load_pose(int(frame_id)).astype(np.float64))
            homo = np.concatenate([pred_points.astype(np.float64), np.ones((pred_points.shape[0], 1), dtype=np.float64)], axis=1)
            pred_points = (pose_cache[frame_id] @ homo.T).T[:, :3].astype(np.float32)
        gt_points = gt_by_frame.get(int(frame_id), np.empty((0, 3), dtype=np.float32))
        metrics = _point_metrics(pred_points, gt_points)
        if metrics.get("status") == "empty":
            continue
        total_pred += int(metrics.get("pred_point_count", pred_points.shape[0]))
        total_gt += int(metrics.get("gt_point_count", gt_points.shape[0]))
        metrics.pop("_pred_to_gt_dist", None)
        rows.append(metrics)
    out: dict[str, Any] = {
        "camera_space_frame_count": int(len(rows)),
        "camera_space_pred_point_count": int(total_pred),
        "camera_space_gt_point_count": int(total_gt),
    }
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and not str(key).startswith("_")
        }
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
        if values:
            out[f"camera_space_{key}"] = float(np.mean(values))
    return out


def _depth_errors(pred_depth: np.ndarray, gt_depth: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred_depth, dtype=np.float64)
    gt = np.asarray(gt_depth, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 1e-6) & (gt > 1e-6)
    if np.count_nonzero(valid) == 0:
        return {"valid_pixel_ratio": 0.0}
    pred = pred[valid]
    gt = gt[valid]
    ratio = np.maximum(pred / gt, gt / pred)
    diff = pred - gt
    return {
        "valid_pixel_ratio": float(np.mean(valid)),
        "absrel": float(np.mean(np.abs(diff) / gt)),
        "sqrel": float(np.mean((diff**2) / gt)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


def _depth_metrics(stream: ScanNetStream, pred: dict[str, np.ndarray], *, points_are_world: bool) -> dict[str, Any]:
    raw_depths: list[float] = []
    gt_depths: list[float] = []
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for point, uv, frame_id in zip(pred["points"], pred["uv"], pred["frame_ids"]):
        frame_id = int(frame_id)
        if frame_id not in cache:
            cache[frame_id] = (stream.load_depth(frame_id), stream.load_pose(frame_id))
        depth, pose = cache[frame_id]
        h, w = depth.shape[:2]
        x = int(np.rint(float(uv[0]) * float(max(w - 1, 1))))
        y = int(np.rint(float(uv[1]) * float(max(h - 1, 1))))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        gt = float(depth[y, x])
        if not np.isfinite(gt) or gt <= 0.0:
            continue
        if points_are_world:
            cam = np.linalg.inv(pose.astype(np.float64)) @ np.asarray([point[0], point[1], point[2], 1.0], dtype=np.float64)
            pred_z = float(cam[2])
        else:
            pred_z = float(point[2])
        raw_depths.append(pred_z)
        gt_depths.append(gt)
    raw = np.asarray(raw_depths, dtype=np.float64)
    gt = np.asarray(gt_depths, dtype=np.float64)
    out = {f"depth_raw_{k}": v for k, v in _depth_errors(raw, gt).items()}
    valid = np.isfinite(raw) & np.isfinite(gt) & (raw > 1e-6) & (gt > 1e-6)
    if np.count_nonzero(valid):
        scale = float(np.median(gt[valid] / raw[valid]))
        out.update({f"depth_median_{k}": v for k, v in _depth_errors(raw * scale, gt).items()})
        a, b = np.linalg.lstsq(np.stack([raw[valid], np.ones(np.count_nonzero(valid))], axis=1), gt[valid], rcond=None)[0]
        out.update({f"depth_ls_{k}": v for k, v in _depth_errors(raw * float(a) + float(b), gt).items()})
        out["depth_scale_median"] = scale
        out["depth_ls_scale"] = float(a)
        out["depth_ls_shift"] = float(b)
    return out


def _instance_coverage(stream: ScanNetStream, pred: dict[str, np.ndarray]) -> dict[str, Any]:
    gt_instances: set[int] = set()
    hit_instances: set[int] = set()
    cache: dict[int, tuple[np.ndarray | None, set[int]]] = {}
    for uv, frame_id in zip(pred["uv"], pred["frame_ids"]):
        frame_id = int(frame_id)
        if frame_id not in cache:
            path = stream.root / "instance" / "instance" / f"{frame_id}.png"
            inst = None
            ids: set[int] = set()
            if path.exists():
                import cv2

                inst = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if inst is not None:
                    ids = {int(v) for v in np.unique(inst).tolist() if int(v) > 0}
                    gt_instances.update(ids)
            cache[frame_id] = (inst, ids)
        inst, ids = cache[frame_id]
        if inst is None:
            continue
        gt_instances.update(ids)
        h, w = inst.shape[:2]
        x = int(np.rint(float(uv[0]) * float(max(w - 1, 1))))
        y = int(np.rint(float(uv[1]) * float(max(h - 1, 1))))
        if 0 <= x < w and 0 <= y < h and int(inst[y, x]) > 0:
            hit_instances.add(int(inst[y, x]))
    return {
        "gt_instance_count_2d_support": int(len(gt_instances)),
        "covered_instance_count_2d_support": int(len(hit_instances)),
        "per_instance_covered_gt_ratio": float(len(hit_instances) / max(len(gt_instances), 1)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row if not key.startswith("_")})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _write_markdown(path: Path, rows: list[dict[str, Any]], scene_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stream4D v22 Direct D4RT Reconstruction Benchmark",
        "",
        "Evaluation-only GT inputs: ScanNet depth/pose/instance maps. Rows with raw/self D4RT coordinates are not method results and are not GT-aligned unless the label says eval-only.",
        "",
        "| variant | status | depth AbsRel median | depth delta1 median | Chamfer-L1 | F@5cm | F@10cm | camera F@10cm | completeness@20cm | outlier@20cm | covered GT inst | notes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {variant} {label} | {status} | {absrel} | {delta1} | {chamfer} | {f5} | {f10} | {camf10} | {comp20} | {out20} | {inst} | {notes} |".format(
                variant=row.get("variant", ""),
                label=row.get("label", ""),
                status=row.get("status", ""),
                absrel=_fmt(row.get("depth_median_absrel")),
                delta1=_fmt(row.get("depth_median_delta1")),
                chamfer=_fmt(row.get("chamfer_l1")),
                f5=_fmt(row.get("fscore@5cm")),
                f10=_fmt(row.get("fscore@10cm")),
                camf10=_fmt(row.get("camera_space_fscore@10cm")),
                comp20=_fmt(row.get("completeness@20cm")),
                out20=_fmt(row.get("outlier_rate_20cm")),
                inst=_fmt(row.get("per_instance_covered_gt_ratio")),
                notes=row.get("notes", ""),
            )
        )
    lines.extend(["", "## Per-Scene Rows", "", "| variant | scene | status | pred pts | gt pts | F@10cm | completeness@20cm | outlier@20cm |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for row in scene_rows:
        lines.append(
            "| {variant} | {scene} | {status} | {pred} | {gt} | {f10} | {comp20} | {out20} |".format(
                variant=row.get("variant", ""),
                scene=row.get("scene", ""),
                status=row.get("status", ""),
                pred=_fmt(row.get("pred_point_count")),
                gt=_fmt(row.get("gt_point_count")),
                f10=_fmt(row.get("fscore@10cm")),
                comp20=_fmt(row.get("completeness@20cm")),
                out20=_fmt(row.get("outlier_rate_20cm")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.6g}"


def _plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    ok = [row for row in rows if row.get("status") == "ok" and row.get("fscore@10cm") is not None]
    if not ok:
        return
    labels = [str(row["variant"]) for row in ok]
    values = [float(row["fscore@10cm"]) for row in ok]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.ylabel("F-score@10cm")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _spec_outputs_world(spec: VariantSpec) -> bool:
    return (
        spec.provider_mode.startswith("eval_sim3")
        or spec.eval_per_chunk
        or spec.provider_mode.startswith("ref0_pose")
    ) and spec.point_mode == "xyz"


def _plot_hist(path: Path, dist: np.ndarray, title: str) -> None:
    values = np.asarray(dist, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(np.clip(values, 0.0, 1.0), bins=50)
    plt.xlabel("pred->GT distance (m), clipped at 1m")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {"num_scenes": int(len(rows))}
    for key in sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.generic)) and not str(key).startswith("_")}):
        values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
        if values:
            out[key] = float(np.mean(values))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v22 direct D4RT reconstruction benchmark from carrier caches.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--audit-root", default="outputs/audit/v22_direct_reconstruction")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--depth-sample-stride", type=int, default=12)
    parser.add_argument("--max-gt-points-per-scene", type=int, default=60000)
    parser.add_argument("--max-pred-points-per-frame", type=int, default=1500)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-pred-z", type=float, default=None)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--nn-radius", type=float, default=0.05)
    parser.add_argument("--density-alpha", type=float, default=2.0)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-windows-per-scene", type=int, default=None)
    parser.add_argument("--debug-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    scenes = _read_seq_list(Path(args.seq_list))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    requested = None if args.variants == "all" else {item.strip() for item in args.variants.split(",") if item.strip()}
    variants = [spec for spec in VARIANTS if requested is None or spec.name in requested]
    scene_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in variants:
        per_variant: list[dict[str, Any]] = []
        for scene in scenes:
            if args.debug_progress:
                print(f"[v22-direct] start {spec.name} {scene}", file=sys.stderr, flush=True)
            scene_dir = Path(spec.cache_root) / scene
            row: dict[str, Any] = {
                "variant": spec.name,
                "label": spec.label,
                "scene": scene,
                "cache_root": spec.cache_root,
                "provider_mode": spec.provider_mode,
                "point_mode": spec.point_mode,
                "depth_calibration": spec.depth_calibration,
                "xyz_field": spec.xyz_field,
                "xyz_transform": spec.xyz_transform,
                "notes": spec.notes,
            }
            if not scene_dir.exists():
                row.update({"status": "missing_cache"})
                scene_rows.append(row)
                continue
            try:
                stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
                if args.debug_progress:
                    print(f"[v22-direct] load windows {spec.name} {scene}", file=sys.stderr, flush=True)
                windows, cache = _load_windows(
                    spec,
                    scene,
                    nn_radius=float(args.nn_radius),
                    density_alpha=float(args.density_alpha),
                    robust_trim_percentile=float(args.robust_trim_percentile),
                    max_anchors=int(args.max_anchors),
                    max_windows_per_scene=args.max_windows_per_scene,
                )
                frame_ids = sorted({int(frame_id) for window in windows for frame_id in window.frame_ids})
                if args.debug_progress:
                    print(f"[v22-direct] load gt {spec.name} {scene} frames={len(frame_ids)}", file=sys.stderr, flush=True)
                gt_points, _, _ = _load_gt_points(
                    stream,
                    frame_ids,
                    depth_sample_stride=int(args.depth_sample_stride),
                    max_gt_points_per_scene=int(args.max_gt_points_per_scene),
                )
                if args.debug_progress:
                    print(f"[v22-direct] collect pred {spec.name} {scene}", file=sys.stderr, flush=True)
                row.update(_raw_uvz_signal_metrics(
                    stream,
                    windows,
                    min_visibility=float(args.min_visibility),
                    min_confidence=float(args.min_confidence),
                    max_anchors=int(args.max_anchors),
                ))
                depth_fit = None
                if spec.depth_calibration != "none":
                    depth_fit = _fit_depth_calibration(
                        stream,
                        windows,
                        min_visibility=float(args.min_visibility),
                        min_confidence=float(args.min_confidence),
                        max_anchors=int(args.max_anchors),
                    )
                    if depth_fit is None:
                        row["depth_calibration_status"] = "failed_no_anchors"
                    else:
                        row["depth_calibration_status"] = "ok"
                        for key, value in depth_fit.items():
                            row[f"depth_calibration_{key}"] = value
                pred = _collect_pred_points(
                    windows,
                    stream=stream,
                    point_mode=spec.point_mode,
                    depth_calibration_fit=depth_fit,
                    depth_calibration=spec.depth_calibration,
                    max_points_per_frame=int(args.max_pred_points_per_frame),
                    min_visibility=float(args.min_visibility),
                    min_confidence=float(args.min_confidence),
                    min_pred_z=args.min_pred_z,
                )
                pred_points = _sample_rows(pred["points"], int(args.max_gt_points_per_scene))
                gt_points = _sample_rows(gt_points, int(args.max_gt_points_per_scene))
                if args.debug_progress:
                    print(f"[v22-direct] point metrics {spec.name} {scene} pred={pred_points.shape[0]} gt={gt_points.shape[0]}", file=sys.stderr, flush=True)
                metrics = _point_metrics(pred_points, gt_points)
                if metrics.get("status") == "empty":
                    row.update({"status": "empty"})
                else:
                    dist = metrics.pop("_pred_to_gt_dist")
                    points_are_world = _spec_outputs_world(spec)
                    row.update(metrics)
                    if args.debug_progress:
                        print(f"[v22-direct] depth metrics {spec.name} {scene}", file=sys.stderr, flush=True)
                    row.update(_depth_metrics(stream, pred, points_are_world=points_are_world))
                    if args.debug_progress:
                        print(f"[v22-direct] camera-space metrics {spec.name} {scene}", file=sys.stderr, flush=True)
                    row.update(
                        _camera_space_point_metrics(
                            stream,
                            pred,
                            points_are_world=points_are_world,
                            depth_sample_stride=int(args.depth_sample_stride),
                            max_gt_points_per_scene=int(args.max_gt_points_per_scene),
                        )
                    )
                    if args.debug_progress:
                        print(f"[v22-direct] instance metrics {spec.name} {scene}", file=sys.stderr, flush=True)
                    row.update(_instance_coverage(stream, pred))
                    row.update(cache.get("stitch_diag", {}))
                    row.update(cache.get("alignment_diag", {}))
                    row["status"] = "ok"
                    _plot_hist(audit_root / f"{spec.name}_{scene}_residual_hist.png", dist, f"{spec.name} {scene}")
                row["num_windows"] = int(len(windows))
            except Exception as exc:
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            scene_rows.append(row)
            if row.get("status") == "ok":
                per_variant.append(row)
        summary = {
            "variant": spec.name,
            "label": spec.label,
            "status": "ok" if per_variant else "missing_or_failed",
            "cache_root": spec.cache_root,
            "provider_mode": spec.provider_mode,
            "point_mode": spec.point_mode,
            "depth_calibration": spec.depth_calibration,
            "xyz_field": spec.xyz_field,
            "xyz_transform": spec.xyz_transform,
            "notes": spec.notes,
            **_aggregate(per_variant),
        }
        summary_rows.append(summary)
    _write_csv(audit_root / "direct_reconstruction_scene_rows.csv", scene_rows)
    _write_csv(audit_root / "direct_reconstruction_summary.csv", summary_rows)
    (audit_root / "direct_reconstruction_scene_rows.json").write_text(json.dumps(_json_safe(scene_rows), indent=2, sort_keys=True), encoding="utf-8")
    (audit_root / "direct_reconstruction_summary.json").write_text(json.dumps(_json_safe(summary_rows), indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(audit_root / "direct_reconstruction_summary.md", summary_rows, scene_rows)
    _plot_summary(audit_root / "direct_reconstruction_f10_by_variant.png", summary_rows)
    print(json.dumps(_json_safe({"summary": summary_rows}), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
