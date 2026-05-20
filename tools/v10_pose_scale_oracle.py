#!/usr/bin/env python3
"""Offline pose-scale oracle diagnostics for v10 target-25 experiments.

The oracle applies GT-fitted window transforms to an existing trajectory.  It is
not a deployment method; it only measures whether the remaining error is mostly
window trajectory state, scale, yaw/translation, or something else.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import (  # noqa: E402
    _angle_diff_deg,
    _apply_alignment,
    _load_kitti_gt,
    _load_tum_prediction,
    _mat_to_quat_xyzw,
    _rmse,
    _umeyama_sim3,
    _yaw_from_pose,
)


@dataclass
class Window:
    name: str
    start: int
    end: int


def _write_tum(path: Path, frames: np.ndarray, poses: np.ndarray) -> None:
    q = _mat_to_quat_xyzw(poses[:, :3, :3])
    t = poses[:, :3, 3]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# timestamp tx ty tz qx qy qz qw\n")
        for frame, tt, qq in zip(frames, t, q):
            handle.write(
                f"{float(frame):.6f} {tt[0]:.9f} {tt[1]:.9f} {tt[2]:.9f} "
                f"{qq[0]:.9f} {qq[1]:.9f} {qq[2]:.9f} {qq[3]:.9f}\n"
            )


def _make_chunk_starts(n_frames: int, chunk_size: int, chunk_overlap: int) -> List[int]:
    step = max(1, int(chunk_size) - max(0, int(chunk_overlap)))
    starts = list(range(0, n_frames, step))
    if starts[-1] != n_frames:
        starts.append(n_frames)
    return starts


def _reset_windows(n_frames: int, chunk_size: int, chunk_overlap: int, reset_every: int) -> List[Window]:
    step = max(1, int(chunk_size) - max(0, int(chunk_overlap)))
    starts = [i * step for i in range(0, 10000, int(reset_every))]
    starts = [s for s in starts if s < n_frames]
    if starts[0] != 0:
        starts.insert(0, 0)
    starts.append(n_frames)
    out = []
    for idx, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        if end - start >= 3:
            out.append(Window(f"reset_{idx:02d}", int(start), int(end)))
    return out


def _semantic_windows(n_frames: int) -> List[Window]:
    bounds = [
        ("pre", 0, 145),
        ("body", 145, 290),
        ("exit", 290, 377),
        ("mid", 377, 464),
        ("c16_handoff", 464, 551),
        ("post", 551, n_frames),
    ]
    return [Window(name, max(0, start), min(n_frames, end)) for name, start, end in bounds if min(n_frames, end) - max(0, start) >= 3]


def _fit_scale_only(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    mx = src.mean(axis=0)
    my = dst.mean(axis=0)
    x = src - mx
    y = dst - my
    denom = float((x * x).sum())
    scale = float((x * y).sum() / max(denom, 1e-12))
    R = np.eye(3, dtype=np.float64)
    t = my - scale * mx
    return scale, R, t


def _fit_yaw_translation(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    src2 = src[:, [0, 2]]
    dst2 = dst[:, [0, 2]]
    mx = src2.mean(axis=0)
    my = dst2.mean(axis=0)
    x = src2 - mx
    y = dst2 - my
    H = x.T @ y
    U, _S, Vt = np.linalg.svd(H)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0.0:
        Vt[-1, :] *= -1.0
        R2 = Vt.T @ U.T
    R = np.eye(3, dtype=np.float64)
    R[0, 0] = R2[0, 0]
    R[0, 2] = R2[0, 1]
    R[2, 0] = R2[1, 0]
    R[2, 2] = R2[1, 1]
    t = dst.mean(axis=0) - (R @ src.mean(axis=0))
    return 1.0, R, t


def _apply_window_oracle(
    poses: np.ndarray,
    frames: np.ndarray,
    gt_pos_by_frame: np.ndarray,
    windows: Iterable[Window],
    mode: str,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    out = poses.copy()
    rows: List[Dict[str, object]] = []
    pos = poses[:, :3, 3]
    for window in windows:
        mask = (frames >= window.start) & (frames < window.end)
        if int(mask.sum()) < 3:
            rows.append({"window": window.name, "start": window.start, "end": window.end, "num_frames": int(mask.sum()), "skipped": True})
            continue
        src = pos[mask]
        dst = gt_pos_by_frame[frames[mask]]
        if mode == "sim3":
            scale, R, t = _umeyama_sim3(src, dst, with_scale=True)
        elif mode == "se3":
            scale, R, t = _umeyama_sim3(src, dst, with_scale=False)
        elif mode == "scale_only":
            scale, R, t = _fit_scale_only(src, dst)
        elif mode == "yaw_translation":
            scale, R, t = _fit_yaw_translation(src, dst)
        else:
            raise ValueError(f"Unsupported oracle mode: {mode}")
        out[mask, :3, :3] = R[None] @ out[mask, :3, :3]
        out[mask, :3, 3] = (scale * (R @ out[mask, :3, 3].T)).T + t[None]
        before = np.linalg.norm(src - dst, axis=1)
        after = np.linalg.norm(out[mask, :3, 3] - dst, axis=1)
        rows.append(
            {
                "window": window.name,
                "start": int(window.start),
                "end": int(window.end),
                "num_frames": int(mask.sum()),
                "mode": mode,
                "scale": float(scale),
                "rmse_before": _rmse(before),
                "rmse_after": _rmse(after),
                "gain": _rmse(before) - _rmse(after),
                "skipped": False,
            }
        )
    return out, rows


def _fixed_segment(frames: np.ndarray, poses: np.ndarray, gt_pos_by_frame: np.ndarray, start: int, end: int) -> float:
    mask = (frames >= start) & (frames < end)
    if int(mask.sum()) < 3:
        return float("nan")
    err = poses[mask, :3, 3] - gt_pos_by_frame[frames[mask]]
    return _rmse(np.linalg.norm(err, axis=1))


def _summarize(name: str, frames: np.ndarray, poses: np.ndarray, gt_poses: np.ndarray, gt_pos_by_frame: np.ndarray) -> Dict[str, object]:
    pos = poses[:, :3, 3]
    gt_pos = gt_pos_by_frame[frames]
    err = pos - gt_pos
    norm = np.linalg.norm(err, axis=1)
    yaw_err = _angle_diff_deg(_yaw_from_pose(poses, "xz"), _yaw_from_pose(gt_poses[frames], "xz"))
    return {
        "run": name,
        "ate_rmse": _rmse(norm),
        "final_err": float(norm[-1]),
        "yaw_rmse": _rmse(yaw_err),
        "axis_rmse_x": _rmse(err[:, 0]),
        "axis_rmse_y": _rmse(err[:, 1]),
        "axis_rmse_z": _rmse(err[:, 2]),
        "seg_200_300": _fixed_segment(frames, poses, gt_pos_by_frame, 200, 300),
        "seg_200_400": _fixed_segment(frames, poses, gt_pos_by_frame, 200, 400),
        "seg_400_500": _fixed_segment(frames, poses, gt_pos_by_frame, 400, 500),
        "seg_400_600": _fixed_segment(frames, poses, gt_pos_by_frame, 400, 600),
        "gap_to_25": _rmse(norm) - 25.0,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{val:.{digits}f}" if math.isfinite(val) else "nan"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True, help="NAME=path/to/01.txt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--reset-every", type=int, default=5)
    args = parser.parse_args()

    if "=" in args.pred:
        pred_name, pred_path_text = args.pred.split("=", 1)
    else:
        pred_path = Path(args.pred)
        pred_name, pred_path_text = pred_path.parent.name, str(pred_path)
    pred_path = Path(pred_path_text)
    if pred_path.is_dir():
        pred_path = pred_path / "01.txt"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
    frames, raw_poses, raw_pos = _load_tum_prediction(pred_path, gt_pos.shape[0])
    scale, R, t = _umeyama_sim3(raw_pos, gt_pos[frames], with_scale=True)
    baseline_poses = _apply_alignment(raw_poses, scale, R, t)

    reset_windows = _reset_windows(gt_pos.shape[0], args.chunk_size, args.chunk_overlap, args.reset_every)
    semantic_windows = _semantic_windows(gt_pos.shape[0])
    variants = {
        "baseline_global_sim3": baseline_poses,
    }
    window_rows: List[Dict[str, object]] = []
    for label, windows, mode in (
        ("POSEORACLE_01_per_reset_sim3", reset_windows, "sim3"),
        ("POSEORACLE_02_semantic_window_sim3", semantic_windows, "sim3"),
        ("POSEORACLE_03_per_reset_scale_only", reset_windows, "scale_only"),
        ("POSEORACLE_04_per_reset_yaw_translation", reset_windows, "yaw_translation"),
        ("POSEORACLE_05_per_reset_se3_no_scale", reset_windows, "se3"),
    ):
        poses, rows = _apply_window_oracle(baseline_poses, frames, gt_pos, windows, mode)
        variants[label] = poses
        for row in rows:
            row["oracle"] = label
            window_rows.append(row)

    summary_rows = []
    for name, poses in variants.items():
        _write_tum(out_dir / f"{name}.tum.txt", frames, poses)
        summary_rows.append(_summarize(name, frames, poses, gt_poses, gt_pos))
    _write_csv(out_dir / "pose_oracle_summary.csv", summary_rows)
    _write_csv(out_dir / "pose_oracle_window_fits.csv", window_rows)
    (out_dir / "pose_oracle_summary.json").write_text(
        json.dumps({"source_run": pred_name, "source_path": str(pred_path), "global_sim3_scale": scale, "runs": summary_rows}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# V10 Pose-Scale Oracle",
        "",
        f"source_run = `{pred_name}`",
        f"source_path = `{pred_path}`",
        f"global_sim3_scale = `{_fmt(scale, 6)}`",
        "",
        "| Oracle | ATE | gap_to_25 | FinalErr | YawRMSE | [200,300) | [200,400) | [400,600) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda item: float(item["ate_rmse"])):
        lines.append(
            f"| `{row['run']}` | {_fmt(row['ate_rmse'])} | {_fmt(row['gap_to_25'])} | "
            f"{_fmt(row['final_err'])} | {_fmt(row['yaw_rmse'])} | {_fmt(row['seg_200_300'])} | "
            f"{_fmt(row['seg_200_400'])} | {_fmt(row['seg_400_600'])} |"
        )
    best = min(summary_rows, key=lambda item: float(item["ate_rmse"]))
    lines += [
        "",
        "## Gate Read",
        "",
        f"best_oracle = `{best['run']}` with ATE `{_fmt(best['ate_rmse'])}`.",
    ]
    if float(best["ate_rmse"]) <= 25.0:
        lines.append("Pose-scale oracle reaches the target-25 gate; v10 should prioritize a no-GT window pose/scale proxy.")
    elif float(best["ate_rmse"]) <= 30.0:
        lines.append("Pose-scale oracle is weak-positive for target-30 but not target-25; read + pose may both be needed.")
    else:
        lines.append("Pose-scale oracle does not reach target-30; current geometry/read output likely remains limiting.")
    (out_dir / "pose_scale_oracle_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'pose_scale_oracle_report.md'}")


if __name__ == "__main__":
    main()
