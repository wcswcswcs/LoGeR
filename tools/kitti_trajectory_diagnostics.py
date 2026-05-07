#!/usr/bin/env python3
"""Plot and diagnose KITTI camera trajectory differences.

Inputs:
  * KITTI GT pose file: one 3x4 camera-to-world pose per line.
  * One or more LoGeR/TUM-style predictions:
      timestamp tx ty tz qx qy qz qw

Outputs:
  * top-down trajectory comparison plot.
  * error-over-frame plots.
  * per-frame and segment error CSV files.
  * aligned/raw camera trajectories in NPZ and TUM text format.
  * diagnosis.md / summary.json with likely failure modes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name.strip("_") or "run"


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[..., 0, 1] = 2.0 * (xy - wz)
    R[..., 0, 2] = 2.0 * (xz + wy)
    R[..., 1, 0] = 2.0 * (xy + wz)
    R[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[..., 1, 2] = 2.0 * (yz - wx)
    R[..., 2, 0] = 2.0 * (xz - wy)
    R[..., 2, 1] = 2.0 * (yz + wx)
    R[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def _mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to xyzw quaternions."""
    R = np.asarray(R, dtype=np.float64)
    q = np.empty(R.shape[:-2] + (4,), dtype=np.float64)
    flat_R = R.reshape(-1, 3, 3)
    flat_q = q.reshape(-1, 4)
    for i, m in enumerate(flat_R):
        tr = float(np.trace(m))
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        qq = np.array([qx, qy, qz, qw], dtype=np.float64)
        flat_q[i] = qq / max(np.linalg.norm(qq), 1e-12)
    return q


def _load_kitti_gt(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.shape[1] != 12:
        raise ValueError(f"KITTI GT must have 12 columns, got {arr.shape} from {path}")
    poses = np.tile(np.eye(4, dtype=np.float64), (arr.shape[0], 1, 1))
    poses[:, :3, :4] = arr.reshape(-1, 3, 4)
    frames = np.arange(arr.shape[0], dtype=np.int64)
    return frames, poses, poses[:, :3, 3].copy()


def _load_tum_prediction(path: Path, n_gt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 8:
                raise ValueError(f"Prediction row must have >=8 columns in {path}: {line}")
            rows.append(vals[:8])
    if not rows:
        raise ValueError(f"No trajectory rows found in {path}")
    arr = np.asarray(rows, dtype=np.float64)
    frames = np.rint(arr[:, 0]).astype(np.int64)
    valid = (frames >= 0) & (frames < n_gt)
    frames = frames[valid]
    arr = arr[valid]
    order = np.argsort(frames)
    frames = frames[order]
    arr = arr[order]

    poses = np.tile(np.eye(4, dtype=np.float64), (frames.shape[0], 1, 1))
    poses[:, :3, :3] = _quat_xyzw_to_mat(arr[:, 4:8])
    poses[:, :3, 3] = arr[:, 1:4]
    return frames, poses, poses[:, :3, 3].copy()


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """Find Sim(3)/SE(3) transform mapping src positions to dst positions."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError(f"Need matched Nx3 arrays with N>=3, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    mx = src.mean(axis=0)
    my = dst.mean(axis=0)
    X = src - mx
    Y = dst - my
    cov = (Y.T @ X) / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0.0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt
    if with_scale:
        var_x = float((X * X).sum() / n)
        scale = float(np.trace(np.diag(S) @ D) / max(var_x, 1e-12))
    else:
        scale = 1.0
    t = my - scale * (R @ mx)
    return scale, R, t


def _apply_alignment(poses: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    aligned = poses.copy()
    aligned[:, :3, :3] = R[None] @ poses[:, :3, :3]
    aligned[:, :3, 3] = (scale * (R @ poses[:, :3, 3].T)).T + t[None]
    return aligned


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.asarray(x, dtype=np.float64) ** 2)))


def _axis_indices(axes: str) -> Tuple[int, int, str, str]:
    mapping = {"x": 0, "y": 1, "z": 2}
    axes = axes.lower()
    if len(axes) != 2 or axes[0] not in mapping or axes[1] not in mapping:
        raise ValueError("--axes must be one of xy/xz/yz/zx/etc.")
    return mapping[axes[0]], mapping[axes[1]], axes[0], axes[1]


def _yaw_from_pose(poses: np.ndarray, axes: str = "xz") -> np.ndarray:
    """Approximate yaw from the camera forward vector projected to plot plane."""
    a, b, _, _ = _axis_indices(axes)
    forward = poses[:, :3, 2]
    return np.arctan2(forward[:, a], forward[:, b])


def _angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return np.degrees(d)


def _write_tum(path: Path, frames: np.ndarray, poses: np.ndarray) -> None:
    q = _mat_to_quat_xyzw(poses[:, :3, :3])
    t = poses[:, :3, 3]
    with path.open("w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for frame, tt, qq in zip(frames, t, q):
            f.write(
                f"{float(frame):.6f} {tt[0]:.9f} {tt[1]:.9f} {tt[2]:.9f} "
                f"{qq[0]:.9f} {qq[1]:.9f} {qq[2]:.9f} {qq[3]:.9f}\n"
            )


@dataclass
class RunData:
    name: str
    safe: str
    path: Path
    frames: np.ndarray
    raw_poses: np.ndarray
    aligned_poses: np.ndarray
    scale: float
    R: np.ndarray
    t: np.ndarray
    summary: Dict[str, object]


def _parse_pred_arg(arg: str) -> Tuple[str, Path]:
    if "=" in arg:
        name, path = arg.split("=", 1)
    elif ":" in arg:
        name, path = arg.split(":", 1)
    else:
        p = Path(arg)
        name, path = p.parent.name or p.stem, arg
    p = Path(path)
    if p.is_dir():
        p = p / "01.txt"
    return name.strip(), p


def _segment_stats(
    name: str,
    frames: np.ndarray,
    aligned_pos: np.ndarray,
    gt_pos_by_frame: np.ndarray,
    segment_lengths: Iterable[int],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {}
    full_pred = np.full_like(gt_pos_by_frame, np.nan, dtype=np.float64)
    full_pred[frames] = aligned_pos
    for seg_len in segment_lengths:
        vals = []
        for start in range(0, gt_pos_by_frame.shape[0], seg_len):
            end = min(start + seg_len, gt_pos_by_frame.shape[0])
            if end - start < max(5, seg_len // 2):
                continue
            seg_pred = full_pred[start:end]
            seg_gt = gt_pos_by_frame[start:end]
            valid = np.isfinite(seg_pred).all(axis=1)
            if valid.sum() < 3:
                ate = float("nan")
                axis_rmse = [float("nan")] * 3
            else:
                err = seg_pred[valid] - seg_gt[valid]
                ate = _rmse(np.linalg.norm(err, axis=1))
                axis_rmse = [_rmse(err[:, i]) for i in range(3)]
            row = {
                "run": name,
                "segment_len": int(seg_len),
                "start": int(start),
                "end": int(end),
                "num_valid": int(valid.sum()),
                "ate_rmse_m": ate,
                "axis_rmse_x_m": axis_rmse[0],
                "axis_rmse_y_m": axis_rmse[1],
                "axis_rmse_z_m": axis_rmse[2],
            }
            rows.append(row)
            if math.isfinite(ate):
                vals.append(row)
        if vals:
            worst = max(vals, key=lambda r: float(r["ate_rmse_m"]))
            best = min(vals, key=lambda r: float(r["ate_rmse_m"]))
            summary[str(seg_len)] = {
                "mean_ate_rmse_m": float(np.mean([r["ate_rmse_m"] for r in vals])),
                "median_ate_rmse_m": float(np.median([r["ate_rmse_m"] for r in vals])),
                "worst": worst,
                "best": best,
            }
    return rows, summary


def _make_chunks(n_frames: int, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, int, int]]:
    if chunk_size <= 0:
        return []
    step = max(1, chunk_size - max(0, chunk_overlap))
    chunks: List[Tuple[int, int, int]] = []
    start = 0
    chunk_idx = 0
    while start < n_frames:
        end = min(start + chunk_size, n_frames)
        chunks.append((chunk_idx, start, end))
        if end >= n_frames:
            break
        start += step
        chunk_idx += 1
    return chunks


def _primary_chunk_idx(frame: int, chunks: List[Tuple[int, int, int]], chunk_size: int, chunk_overlap: int) -> int:
    if not chunks:
        return -1
    step = max(1, chunk_size - max(0, chunk_overlap))
    idx = min(int(frame) // step, len(chunks) - 1)
    return int(chunks[idx][0])


def _find_run(runs: List[RunData], focus_run: Optional[str]) -> RunData:
    if not runs:
        raise ValueError("No runs available")
    if not focus_run:
        return runs[-1]
    target = focus_run.strip()
    for run in runs:
        if run.name == target or run.safe == target:
            return run
    known = ", ".join(r.name for r in runs)
    raise ValueError(f"--focus_run={focus_run!r} not found. Known runs: {known}")


def _build_chunk_rows(
    runs: List[RunData],
    gt_pos_by_frame: np.ndarray,
    chunks: List[Tuple[int, int, int]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not chunks:
        return rows
    for run in runs:
        full_pred = np.full_like(gt_pos_by_frame, np.nan, dtype=np.float64)
        full_pred[run.frames] = run.aligned_poses[:, :3, 3]
        for chunk_idx, start, end in chunks:
            seg_pred = full_pred[start:end]
            seg_gt = gt_pos_by_frame[start:end]
            valid = np.isfinite(seg_pred).all(axis=1)
            if valid.sum() == 0:
                rows.append({
                    "run": run.name,
                    "chunk_idx": int(chunk_idx),
                    "start": int(start),
                    "end": int(end),
                    "num_valid": 0,
                    "rmse_m": float("nan"),
                    "mean_m": float("nan"),
                    "max_m": float("nan"),
                    "worst_frame": -1,
                    "start_error_m": float("nan"),
                    "end_error_m": float("nan"),
                    "axis_rmse_x_m": float("nan"),
                    "axis_rmse_y_m": float("nan"),
                    "axis_rmse_z_m": float("nan"),
                })
                continue
            err = seg_pred[valid] - seg_gt[valid]
            norm = np.linalg.norm(err, axis=1)
            valid_frames = np.arange(start, end)[valid]
            worst_i = int(np.nanargmax(norm))
            start_err = (
                np.linalg.norm(full_pred[start] - gt_pos_by_frame[start])
                if np.isfinite(full_pred[start]).all() else float("nan")
            )
            end_frame = end - 1
            end_err = (
                np.linalg.norm(full_pred[end_frame] - gt_pos_by_frame[end_frame])
                if np.isfinite(full_pred[end_frame]).all() else float("nan")
            )
            rows.append({
                "run": run.name,
                "chunk_idx": int(chunk_idx),
                "start": int(start),
                "end": int(end),
                "num_valid": int(valid.sum()),
                "rmse_m": _rmse(norm),
                "mean_m": float(np.nanmean(norm)),
                "max_m": float(np.nanmax(norm)),
                "worst_frame": int(valid_frames[worst_i]),
                "start_error_m": float(start_err),
                "end_error_m": float(end_err),
                "axis_rmse_x_m": _rmse(err[:, 0]),
                "axis_rmse_y_m": _rmse(err[:, 1]),
                "axis_rmse_z_m": _rmse(err[:, 2]),
            })
    return rows


def _diagnose(summary: Dict[str, object]) -> List[str]:
    notes: List[str] = []
    scale = float(summary["sim3_scale"])
    raw = float(summary["raw_ate_rmse_m"])
    aligned = float(summary["aligned_ate_rmse_m"])
    gain = raw / max(aligned, 1e-9)
    axis = summary["largest_axis"]
    final_err = float(summary["final_error_m"])
    slope = float(summary["error_slope_m_per_100f"])
    yaw = float(summary.get("yaw_rmse_deg", 0.0))

    if abs(scale - 1.0) > 0.05:
        notes.append(
            f"Raw trajectory has a large global scale offset: Sim(3) scale={scale:.4f}, "
            f"{100*(scale-1):+.2f}% from 1. This is absorbed by Sim(3) alignment, "
            "so it explains raw-coordinate mismatch, not necessarily the aligned KITTI ATE."
        )
    if gain > 1.5:
        notes.append(f"Raw ATE is {gain:.2f}x aligned ATE, so a large part is global Sim(3) mismatch rather than only local drift.")
    if slope > 0.5:
        notes.append(f"Error grows over time ({slope:.3f} m / 100 frames), suggesting accumulated drift.")
    if final_err > aligned * 1.5:
        notes.append(f"Final-frame error ({final_err:.3f} m) is much larger than RMSE, suggesting tail drift.")
    notes.append(f"Largest aligned axis RMSE is {axis}, so inspect that component first.")
    if yaw > 3.0:
        notes.append(f"Yaw RMSE is {yaw:.2f} deg after alignment, so orientation/turning drift may be contributing.")
    if not notes:
        notes.append("No single dominant failure mode found; inspect segment CSV and per-frame plots.")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, help="KITTI GT pose file, e.g. .../poses/01.txt")
    parser.add_argument("--pred", action="append", required=True,
                        help="Prediction as NAME=path/to/01.txt, NAME=dir, or path. Can be repeated.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seq_name", default="KITTI 01")
    parser.add_argument("--axes", default="xz", help="Top-down plane, e.g. xz for KITTI, xy for indoor plots.")
    parser.add_argument("--align", choices=["sim3", "se3", "none"], default="sim3")
    parser.add_argument("--segment_lengths", type=int, nargs="*", default=[50, 100, 200])
    parser.add_argument("--chunk_size", type=int, default=0,
                        help="Optional pipeline chunk size for chunk-aware trajectory diagnostics.")
    parser.add_argument("--chunk_overlap", type=int, default=0,
                        help="Optional pipeline chunk overlap used with --chunk_size.")
    parser.add_argument("--focus_run", default=None,
                        help="Run name/safe name to annotate with arrows and chunk failures. Defaults to last --pred.")
    parser.add_argument("--arrow_stride", type=int, default=50,
                        help="Draw GT-to-pred error arrows every N frames for the focus run.")
    parser.add_argument("--top_error_count", type=int, default=12,
                        help="Annotate this many worst focus-run frames on the chunk trajectory plot.")
    parser.add_argument("--chunk_label_every", type=int, default=5,
                        help="Label every Nth chunk boundary on the trajectory plot.")
    parser.add_argument("--chunk_error_percentile", type=float, default=95.0,
                        help="Colorbar vmax percentile for chunk RMSE on the chunk trajectory plot.")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
    gt_by_frame = gt_pos.copy()

    runs: List[RunData] = []
    all_frame_rows: List[Dict[str, object]] = []
    all_segment_rows: List[Dict[str, object]] = []
    npz_payload: Dict[str, np.ndarray] = {
        "gt_frames": gt_frames,
        "gt_poses": gt_poses,
        "gt_positions": gt_pos,
    }

    for pred_arg in args.pred:
        name, path = _parse_pred_arg(pred_arg)
        safe = _safe_name(name)
        frames, raw_poses, raw_pos = _load_tum_prediction(path, gt_pos.shape[0])
        matched_gt = gt_by_frame[frames]
        valid = np.isfinite(raw_pos).all(axis=1)
        frames = frames[valid]
        raw_poses = raw_poses[valid]
        raw_pos = raw_pos[valid]
        matched_gt = matched_gt[valid]
        if args.align == "none":
            scale, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            scale, R, t = _umeyama_sim3(raw_pos, matched_gt, with_scale=(args.align == "sim3"))
        aligned_poses = _apply_alignment(raw_poses, scale, R, t)
        aligned_pos = aligned_poses[:, :3, 3]

        raw_err = raw_pos - matched_gt
        aligned_err = aligned_pos - matched_gt
        raw_norm = np.linalg.norm(raw_err, axis=1)
        aligned_norm = np.linalg.norm(aligned_err, axis=1)
        yaw_err = _angle_diff_deg(_yaw_from_pose(aligned_poses, args.axes), _yaw_from_pose(gt_poses[frames], args.axes))
        axis_rmse = np.array([_rmse(aligned_err[:, i]) for i in range(3)])
        largest_axis = ["x", "y", "z"][int(np.nanargmax(axis_rmse))]
        slope = 0.0
        if frames.shape[0] >= 2:
            coef = np.polyfit(frames.astype(np.float64), aligned_norm, deg=1)
            slope = float(coef[0] * 100.0)
        summary = {
            "name": name,
            "path": str(path),
            "num_frames": int(frames.shape[0]),
            "align": args.align,
            "sim3_scale": float(scale),
            "sim3_R": R.tolist(),
            "sim3_t": t.tolist(),
            "raw_ate_rmse_m": _rmse(raw_norm),
            "aligned_ate_rmse_m": _rmse(aligned_norm),
            "aligned_ate_mean_m": float(np.nanmean(aligned_norm)),
            "aligned_ate_median_m": float(np.nanmedian(aligned_norm)),
            "aligned_ate_p90_m": float(np.nanpercentile(aligned_norm, 90)),
            "aligned_ate_max_m": float(np.nanmax(aligned_norm)),
            "axis_rmse_x_m": float(axis_rmse[0]),
            "axis_rmse_y_m": float(axis_rmse[1]),
            "axis_rmse_z_m": float(axis_rmse[2]),
            "largest_axis": largest_axis,
            "final_error_m": float(aligned_norm[-1]),
            "final_error_xyz_m": aligned_err[-1].tolist(),
            "error_slope_m_per_100f": slope,
            "yaw_rmse_deg": _rmse(yaw_err),
        }

        for frame, raw_e, aligned_e, yaw_e in zip(frames, raw_norm, aligned_norm, yaw_err):
            idx = np.where(frames == frame)[0][0]
            err_vec = aligned_err[idx]
            all_frame_rows.append({
                "run": name,
                "frame": int(frame),
                "raw_error_m": float(raw_e),
                "aligned_error_m": float(aligned_e),
                "aligned_error_x_m": float(err_vec[0]),
                "aligned_error_y_m": float(err_vec[1]),
                "aligned_error_z_m": float(err_vec[2]),
                "yaw_error_deg": float(yaw_e),
            })

        seg_rows, seg_summary = _segment_stats(name, frames, aligned_pos, gt_by_frame, args.segment_lengths)
        all_segment_rows.extend(seg_rows)
        summary["segment_summary"] = seg_summary
        summary["diagnosis"] = _diagnose(summary)

        _write_tum(out_dir / f"{safe}_raw.tum.txt", frames, raw_poses)
        _write_tum(out_dir / f"{safe}_aligned.tum.txt", frames, aligned_poses)
        npz_payload[f"{safe}_frames"] = frames
        npz_payload[f"{safe}_raw_poses"] = raw_poses
        npz_payload[f"{safe}_aligned_poses"] = aligned_poses
        npz_payload[f"{safe}_raw_positions"] = raw_pos
        npz_payload[f"{safe}_aligned_positions"] = aligned_pos
        npz_payload[f"{safe}_sim3_scale"] = np.array([scale], dtype=np.float64)
        npz_payload[f"{safe}_sim3_R"] = R
        npz_payload[f"{safe}_sim3_t"] = t

        runs.append(RunData(name, safe, path, frames, raw_poses, aligned_poses, scale, R, t, summary))

    chunks = _make_chunks(gt_pos.shape[0], args.chunk_size, args.chunk_overlap)
    chunk_rows = _build_chunk_rows(runs, gt_by_frame, chunks)
    if chunks:
        for row in all_frame_rows:
            row["chunk_idx"] = _primary_chunk_idx(
                int(row["frame"]), chunks, args.chunk_size, args.chunk_overlap
            )
    else:
        for row in all_frame_rows:
            row["chunk_idx"] = -1

    _write_tum(out_dir / "ground_truth.tum.txt", gt_frames, gt_poses)
    np.savez_compressed(out_dir / "camera_trajectories.npz", **npz_payload)

    with (out_dir / "per_frame_errors.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["run", "frame", "chunk_idx", "raw_error_m", "aligned_error_m",
                      "aligned_error_x_m", "aligned_error_y_m", "aligned_error_z_m", "yaw_error_deg"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_frame_rows)

    with (out_dir / "segment_errors.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["run", "segment_len", "start", "end", "num_valid", "ate_rmse_m",
                      "axis_rmse_x_m", "axis_rmse_y_m", "axis_rmse_z_m"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_segment_rows)

    if chunks:
        with (out_dir / "chunk_errors.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "run", "chunk_idx", "start", "end", "num_valid", "rmse_m", "mean_m",
                "max_m", "worst_frame", "start_error_m", "end_error_m",
                "axis_rmse_x_m", "axis_rmse_y_m", "axis_rmse_z_m",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(chunk_rows)

    summary_json = {
        "sequence": args.seq_name,
        "gt": str(Path(args.gt)),
        "axes": args.axes,
        "align": args.align,
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "chunks": [
            {"chunk_idx": int(idx), "start": int(start), "end": int(end)}
            for idx, start, end in chunks
        ],
        "runs": [r.summary for r in runs],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    # Import matplotlib lazily so the metric-only path is easy to debug.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    a, b, an, bn = _axis_indices(args.axes)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.plot(gt_pos[:, a], gt_pos[:, b], "--", color="black", linewidth=1.2, alpha=0.65, label="Ground Truth")
    for i, run in enumerate(runs):
        pos = run.aligned_poses[:, :3, 3]
        ax.plot(pos[:, a], pos[:, b], linewidth=1.2, color=colors[i % len(colors)], label=run.name)
    ax.scatter([gt_pos[0, a]], [gt_pos[0, b]], marker="o", color="black", s=24, label="Start")
    ax.scatter([gt_pos[-1, a]], [gt_pos[-1, b]], marker="x", color="black", s=36, label="GT End")
    ax.set_title(f"Trajectory Comparison: {args.seq_name}")
    ax.set_xlabel(f"{an} (m)")
    ax.set_ylabel(f"{bn} (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"trajectory_{args.axes}_{args.align}.png", dpi=args.dpi)
    fig.savefig(out_dir / f"trajectory_{args.axes}_{args.align}.pdf")
    plt.close(fig)

    if chunks:
        focus = _find_run(runs, args.focus_run)
        focus_pos = focus.aligned_poses[:, :3, 3]
        focus_frame_to_idx = {int(frame): i for i, frame in enumerate(focus.frames)}
        focus_frame_rows = [r for r in all_frame_rows if r["run"] == focus.name]
        focus_chunk_rows = [
            r for r in chunk_rows
            if r["run"] == focus.name and math.isfinite(float(r["rmse_m"]))
        ]
        chunk_rmse = {int(r["chunk_idx"]): float(r["rmse_m"]) for r in focus_chunk_rows}
        rmse_vals = np.asarray(list(chunk_rmse.values()), dtype=np.float64)
        vmax = float(np.nanpercentile(
            rmse_vals,
            min(100.0, max(0.0, float(args.chunk_error_percentile))),
        )) if rmse_vals.size else 1.0
        vmax = max(vmax, 1e-6)
        norm = plt.Normalize(vmin=0.0, vmax=vmax)

        segments = []
        segment_values = []
        for chunk_idx, start, end in chunks:
            if chunk_idx not in chunk_rmse:
                continue
            pts = gt_pos[start:end, :][:, [a, b]]
            if pts.shape[0] < 2:
                continue
            segments.extend(np.stack([pts[:-1], pts[1:]], axis=1))
            segment_values.extend([chunk_rmse[chunk_idx]] * (pts.shape[0] - 1))

        top_frame_rows = sorted(
            focus_frame_rows,
            key=lambda r: float(r["aligned_error_m"]),
            reverse=True,
        )[:max(0, int(args.top_error_count))]
        top_frames = {int(r["frame"]) for r in top_frame_rows}
        worst_chunk_ids = {
            int(r["chunk_idx"])
            for r in sorted(focus_chunk_rows, key=lambda r: float(r["rmse_m"]), reverse=True)[:5]
        }

        fig, ax = plt.subplots(figsize=(9.5, 7.2))
        if segments:
            lc = LineCollection(
                segments,
                cmap="magma",
                norm=norm,
                linewidths=2.5,
                alpha=0.92,
                label=f"GT colored by {focus.name} chunk RMSE",
            )
            lc.set_array(np.asarray(segment_values, dtype=np.float64))
            ax.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label("Chunk RMSE (m)")
        else:
            ax.plot(gt_pos[:, a], gt_pos[:, b], "--", color="black", linewidth=1.2, alpha=0.65, label="Ground Truth")

        ax.plot(focus_pos[:, a], focus_pos[:, b], color="#1f77b4", linewidth=1.15, alpha=0.9, label=f"{focus.name} aligned")
        ax.scatter([gt_pos[0, a]], [gt_pos[0, b]], marker="o", color="black", s=24, label="Start")
        ax.scatter([gt_pos[-1, a]], [gt_pos[-1, b]], marker="x", color="black", s=36, label="GT End")

        label_every = max(1, int(args.chunk_label_every))
        for chunk_idx, start, end in chunks:
            is_worst_chunk = int(chunk_idx) in worst_chunk_ids
            if chunk_idx % label_every != 0 and not is_worst_chunk:
                continue
            pos = gt_pos[start]
            label = f"c{chunk_idx}"
            if is_worst_chunk:
                label += f"\n{chunk_rmse.get(chunk_idx, float('nan')):.1f}m"
            ax.scatter([pos[a]], [pos[b]], marker="|", color="black", s=32, alpha=0.65)
            ax.text(
                pos[a], pos[b], label,
                fontsize=7,
                color="black",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.65),
            )

        arrow_stride = max(1, int(args.arrow_stride))
        arrow_frames = set(range(0, gt_pos.shape[0], arrow_stride)) | top_frames
        arrow_label_done = False
        top_label_done = False
        for frame in sorted(arrow_frames):
            pred_idx = focus_frame_to_idx.get(int(frame))
            if pred_idx is None:
                continue
            gt_xy = gt_pos[frame, [a, b]]
            pred_xy = focus_pos[pred_idx, [a, b]]
            is_top = int(frame) in top_frames
            color = "#d62728" if is_top else "#555555"
            alpha = 0.85 if is_top else 0.22
            lw = 1.15 if is_top else 0.55
            ax.annotate(
                "",
                xy=(pred_xy[0], pred_xy[1]),
                xytext=(gt_xy[0], gt_xy[1]),
                arrowprops=dict(arrowstyle="->", color=color, alpha=alpha, linewidth=lw, shrinkA=0, shrinkB=0),
            )
            if not arrow_label_done and not is_top:
                ax.plot([], [], color="#555555", alpha=0.5, linewidth=0.8, label=f"Error arrows every {arrow_stride}f")
                arrow_label_done = True
            if not top_label_done and is_top:
                ax.plot([], [], color="#d62728", linewidth=1.2, label=f"Top {len(top_frames)} frame errors")
                top_label_done = True
            if is_top:
                row = next(r for r in top_frame_rows if int(r["frame"]) == int(frame))
                ax.text(
                    pred_xy[0],
                    pred_xy[1],
                    f"f{frame}/c{row['chunk_idx']}\n{float(row['aligned_error_m']):.1f}m",
                    fontsize=7,
                    color="#d62728",
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#d62728", alpha=0.75),
                )

        ax.set_title(f"Chunk Error Map: {args.seq_name} ({focus.name})")
        ax.set_xlabel(f"{an} (m)")
        ax.set_ylabel(f"{bn} (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", fontsize=8)
        ax.autoscale()
        fig.tight_layout()
        fig.savefig(out_dir / f"trajectory_chunk_errors_{focus.safe}_{args.axes}_{args.align}.png", dpi=args.dpi)
        fig.savefig(out_dir / f"trajectory_chunk_errors_{focus.safe}_{args.axes}_{args.align}.pdf")
        plt.close(fig)

        if focus_chunk_rows:
            x = np.asarray([int(r["chunk_idx"]) for r in focus_chunk_rows], dtype=np.int64)
            rmse = np.asarray([float(r["rmse_m"]) for r in focus_chunk_rows], dtype=np.float64)
            max_err = np.asarray([float(r["max_m"]) for r in focus_chunk_rows], dtype=np.float64)
            start_err = np.asarray([float(r["start_error_m"]) for r in focus_chunk_rows], dtype=np.float64)
            end_err = np.asarray([float(r["end_error_m"]) for r in focus_chunk_rows], dtype=np.float64)

            fig, ax = plt.subplots(figsize=(10.0, 4.8))
            ax.bar(x, rmse, color="#4c78a8", alpha=0.78, label="Chunk RMSE")
            ax.plot(x, max_err, color="#d62728", linewidth=1.0, marker=".", markersize=3, label="Chunk max frame error")
            ax.plot(x, start_err, color="#2ca02c", linewidth=0.9, alpha=0.75, label="Start-frame error")
            ax.plot(x, end_err, color="#ff7f0e", linewidth=0.9, alpha=0.75, label="End-frame error")
            for row in sorted(focus_chunk_rows, key=lambda r: float(r["rmse_m"]), reverse=True)[:8]:
                ci = int(row["chunk_idx"])
                ax.text(
                    ci,
                    float(row["rmse_m"]),
                    f"c{ci}\n[{row['start']},{row['end']})\nf{row['worst_frame']}",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    rotation=0,
                )
            ax.set_title(f"Chunk Error Timeline: {args.seq_name} ({focus.name})")
            ax.set_xlabel("Chunk index")
            ax.set_ylabel("Error (m)")
            ax.grid(True, axis="y", alpha=0.3)
            if len(x) > 0:
                tick_step = max(1, len(x) // 12)
                ax.set_xticks(x[::tick_step])
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"chunk_error_timeline_{focus.safe}.png", dpi=args.dpi)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for i, run in enumerate(runs):
        rows = [r for r in all_frame_rows if r["run"] == run.name]
        ax.plot([r["frame"] for r in rows], [r["aligned_error_m"] for r in rows],
                linewidth=1.0, color=colors[i % len(colors)], label=run.name)
    ax.set_title(f"Aligned Position Error: {args.seq_name}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Position error (m)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "aligned_error_over_frame.png", dpi=args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True)
    for i, run in enumerate(runs):
        rows = [r for r in all_frame_rows if r["run"] == run.name]
        frames = [r["frame"] for r in rows]
        for j, axis_name in enumerate(["x", "y", "z"]):
            axes[j].plot(frames, [r[f"aligned_error_{axis_name}_m"] for r in rows],
                         linewidth=0.9, color=colors[i % len(colors)], label=run.name)
            axes[j].set_ylabel(f"{axis_name} err (m)")
            axes[j].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Frame")
    fig.suptitle(f"Aligned Axis Errors: {args.seq_name}")
    fig.tight_layout()
    fig.savefig(out_dir / "aligned_axis_errors.png", dpi=args.dpi)
    plt.close(fig)

    for seg_len in args.segment_lengths:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        width = 0.8 / max(1, len(runs))
        starts = sorted({int(r["start"]) for r in all_segment_rows if int(r["segment_len"]) == seg_len})
        x = np.arange(len(starts), dtype=np.float64)
        for i, run in enumerate(runs):
            vals_by_start = {
                int(r["start"]): float(r["ate_rmse_m"])
                for r in all_segment_rows
                if r["run"] == run.name and int(r["segment_len"]) == seg_len
            }
            vals = [vals_by_start.get(s, np.nan) for s in starts]
            ax.bar(x + (i - (len(runs) - 1) / 2.0) * width, vals, width=width,
                   color=colors[i % len(colors)], label=run.name)
        ax.set_title(f"Segment ATE RMSE ({seg_len} frames): {args.seq_name}")
        ax.set_xlabel("Segment start frame")
        ax.set_ylabel("ATE RMSE (m)")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in starts], rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"segment_ate_{seg_len}.png", dpi=args.dpi)
        plt.close(fig)

    lines = [
        f"# Trajectory Diagnostics: {args.seq_name}",
        "",
        f"- GT: `{Path(args.gt)}`",
        f"- Alignment: `{args.align}`",
        f"- Plot axes: `{args.axes}`",
        f"- Saved trajectories: `ground_truth.tum.txt`, `*_raw.tum.txt`, `*_aligned.tum.txt`, `camera_trajectories.npz`",
        "",
        "## Summary",
        "",
        "| Run | Frames | Raw ATE | Aligned ATE | Scale | Axis RMSE x/y/z | Final Err | Err Slope /100f | Yaw RMSE |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for run in runs:
        s = run.summary
        lines.append(
            f"| {run.name} | {s['num_frames']} | {s['raw_ate_rmse_m']:.4f} | "
            f"{s['aligned_ate_rmse_m']:.4f} | {s['sim3_scale']:.6f} | "
            f"{s['axis_rmse_x_m']:.3f}/{s['axis_rmse_y_m']:.3f}/{s['axis_rmse_z_m']:.3f} | "
            f"{s['final_error_m']:.3f} | {s['error_slope_m_per_100f']:.4f} | "
            f"{s['yaw_rmse_deg']:.3f} |"
        )
    lines.extend(["", "## Diagnosis Notes", ""])
    for run in runs:
        lines.append(f"### {run.name}")
        lines.append("")
        for note in run.summary["diagnosis"]:
            lines.append(f"- {note}")
        for seg_len, seg in run.summary["segment_summary"].items():
            worst = seg["worst"]
            lines.append(
                f"- Worst {seg_len}-frame segment: [{worst['start']}, {worst['end']}) "
                f"ATE={worst['ate_rmse_m']:.4f} m."
            )
        lines.append("")
    if chunks:
        focus = _find_run(runs, args.focus_run)
        focus_chunk_rows = [
            r for r in chunk_rows
            if r["run"] == focus.name and math.isfinite(float(r["rmse_m"]))
        ]
        lines.extend([
            "## Chunk Diagnostics",
            "",
            f"- Chunk size / overlap: `{args.chunk_size}` / `{args.chunk_overlap}`",
            f"- Focus run: `{focus.name}`",
            f"- Saved chunk CSV: `chunk_errors.csv`",
            f"- Saved chunk trajectory map: `trajectory_chunk_errors_{focus.safe}_{args.axes}_{args.align}.png`",
            f"- Saved chunk timeline: `chunk_error_timeline_{focus.safe}.png`",
            "",
            "| Rank | Chunk | Frame Range | RMSE | Max Err | Worst Frame | Start Err | End Err |",
            "|---:|---:|---|---:|---:|---:|---:|---:|",
        ])
        for rank, row in enumerate(
            sorted(focus_chunk_rows, key=lambda r: float(r["rmse_m"]), reverse=True)[:12],
            start=1,
        ):
            lines.append(
                f"| {rank} | {row['chunk_idx']} | [{row['start']}, {row['end']}) | "
                f"{float(row['rmse_m']):.4f} | {float(row['max_m']):.4f} | "
                f"{row['worst_frame']} | {float(row['start_error_m']):.4f} | "
                f"{float(row['end_error_m']):.4f} |"
            )
        lines.append("")
    (out_dir / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved trajectory diagnostics to {out_dir}")
    if chunks:
        focus = _find_run(runs, args.focus_run)
        print(
            f"Chunk diagnostics: focus={focus.name}, "
            f"chunk_size={args.chunk_size}, overlap={args.chunk_overlap}"
        )
    for run in runs:
        s = run.summary
        print(
            f"{run.name}: aligned ATE={s['aligned_ate_rmse_m']:.4f} m, "
            f"scale={s['sim3_scale']:.6f}, largest_axis={s['largest_axis']}, "
            f"final_err={s['final_error_m']:.3f} m"
        )


if __name__ == "__main__":
    main()
