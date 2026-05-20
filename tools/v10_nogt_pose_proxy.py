#!/usr/bin/env python3
"""No-GT trajectory-state proxy diagnostics for v10 target-25 experiments.

This tool intentionally does not use GT to build corrections.  It applies
simple window-local scale proxies derived only from the predicted trajectory's
own step-length statistics, then evaluates the resulting trajectories with the
standard global Sim(3) KITTI-style alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


@dataclass(frozen=True)
class Window:
    name: str
    start: int
    end: int


def _write_tum(path: Path, frames: np.ndarray, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q = _mat_to_quat_xyzw(poses[:, :3, :3])
    t = poses[:, :3, 3]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# timestamp tx ty tz qx qy qz qw\n")
        for frame, tt, qq in zip(frames, t, q):
            handle.write(
                f"{float(frame):.6f} {tt[0]:.9f} {tt[1]:.9f} {tt[2]:.9f} "
                f"{qq[0]:.9f} {qq[1]:.9f} {qq[2]:.9f} {qq[3]:.9f}\n"
            )


def _reset_windows(n_frames: int, chunk_size: int, chunk_overlap: int, reset_every: int) -> List[Window]:
    step = max(1, int(chunk_size) - max(0, int(chunk_overlap)))
    starts = [idx * step for idx in range(0, 10000, max(1, int(reset_every)))]
    starts = [start for start in starts if start < n_frames]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    starts.append(n_frames)
    return [
        Window(f"reset_{idx:02d}", int(start), int(end))
        for idx, (start, end) in enumerate(zip(starts[:-1], starts[1:]))
        if int(end) - int(start) >= 3
    ]


def _semantic_windows(n_frames: int) -> List[Window]:
    bounds = [
        ("pre", 0, 145),
        ("body", 145, 290),
        ("exit", 290, 377),
        ("mid", 377, 464),
        ("c16_handoff", 464, 551),
        ("post", 551, n_frames),
    ]
    return [
        Window(name, max(0, start), min(n_frames, end))
        for name, start, end in bounds
        if min(n_frames, end) - max(0, start) >= 3
    ]


def _step_median(pos: np.ndarray, window: Window) -> float:
    start = max(0, int(window.start))
    end = min(pos.shape[0], int(window.end))
    if end - start < 3:
        return float("nan")
    steps = np.linalg.norm(np.diff(pos[start:end], axis=0), axis=1)
    finite = steps[np.isfinite(steps)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _robust_global_step(step_medians: Sequence[float]) -> float:
    vals = np.asarray([x for x in step_medians if math.isfinite(float(x)) and float(x) > 1e-9], dtype=np.float64)
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))


def _neighbor_target(step_medians: Sequence[float], idx: int) -> float:
    vals = []
    for j in range(max(0, idx - 1), min(len(step_medians), idx + 2)):
        val = float(step_medians[j])
        if math.isfinite(val) and val > 1e-9:
            vals.append(val)
    if not vals:
        return 1.0
    return float(np.median(vals))


def _scale_windows_continuous(
    poses: np.ndarray,
    windows: Iterable[Window],
    scales: Sequence[float],
) -> np.ndarray:
    """Apply per-window translational scales while preserving boundary continuity."""
    out = poses.copy()
    raw_pos = poses[:, :3, 3]
    new_pos = raw_pos.copy()
    windows = list(windows)
    for window, scale in zip(windows, scales):
        start = max(0, int(window.start))
        end = min(raw_pos.shape[0], int(window.end))
        if end - start < 2:
            continue
        if start == 0:
            anchor_new = new_pos[start].copy()
        else:
            anchor_new = new_pos[start - 1] + float(scale) * (raw_pos[start] - raw_pos[start - 1])
        anchor_raw = raw_pos[start].copy()
        new_pos[start:end] = anchor_new[None, :] + float(scale) * (raw_pos[start:end] - anchor_raw[None, :])
    out[:, :3, 3] = new_pos
    return out


def _policy_scales(
    name: str,
    pos: np.ndarray,
    windows: Sequence[Window],
    clip: Tuple[float, float],
    active_indices: Sequence[int] | None = None,
    target_multiplier: float = 1.0,
) -> Tuple[List[float], List[Dict[str, object]]]:
    medians = [_step_median(pos, window) for window in windows]
    global_target = _robust_global_step(medians)
    active = set(int(idx) for idx in active_indices) if active_indices is not None else None
    scales: List[float] = []
    rows: List[Dict[str, object]] = []
    for idx, (window, cur) in enumerate(zip(windows, medians)):
        if active is not None and idx not in active:
            target = cur
            scale = 1.0
        elif not math.isfinite(float(cur)) or float(cur) <= 1e-9:
            target = cur
            scale = 1.0
        elif "neighbor" in name:
            target = _neighbor_target(medians, idx)
            target = float(target) * float(target_multiplier)
            scale = target / float(cur)
        else:
            target = float(global_target) * float(target_multiplier)
            scale = target / float(cur)
        clipped = float(np.clip(scale, clip[0], clip[1]))
        scales.append(clipped)
        rows.append(
            {
                "policy": name,
                "window": window.name,
                "start": int(window.start),
                "end": int(window.end),
                "step_median": float(cur),
                "target_step_median": float(target),
                "raw_scale": float(scale),
                "applied_scale": clipped,
            }
        )
    return scales, rows


def _active_reset_indices(windows: Sequence[Window], *, start_min: int = 0, end_max: int | None = None) -> List[int]:
    out: List[int] = []
    for idx, window in enumerate(windows):
        if int(window.end) <= int(start_min):
            continue
        if end_max is not None and int(window.start) >= int(end_max):
            continue
        out.append(idx)
    return out


def _fixed_segment(frames: np.ndarray, poses: np.ndarray, gt_pos_by_frame: np.ndarray, start: int, end: int) -> float:
    mask = (frames >= start) & (frames < end)
    if int(mask.sum()) < 3:
        return float("nan")
    err = poses[mask, :3, 3] - gt_pos_by_frame[frames[mask]]
    return _rmse(np.linalg.norm(err, axis=1))


def _summarize_aligned(
    name: str,
    frames: np.ndarray,
    raw_poses: np.ndarray,
    gt_poses: np.ndarray,
    gt_pos_by_frame: np.ndarray,
) -> Dict[str, object]:
    raw_pos = raw_poses[:, :3, 3]
    gt_pos = gt_pos_by_frame[frames]
    scale, R, t = _umeyama_sim3(raw_pos, gt_pos, with_scale=True)
    aligned = _apply_alignment(raw_poses, scale, R, t)
    err = aligned[:, :3, 3] - gt_pos
    norm = np.linalg.norm(err, axis=1)
    yaw_err = _angle_diff_deg(_yaw_from_pose(aligned, "xz"), _yaw_from_pose(gt_poses[frames], "xz"))
    return {
        "run": name,
        "ate_rmse": _rmse(norm),
        "final_err": float(norm[-1]),
        "yaw_rmse": _rmse(yaw_err),
        "sim3_scale": float(scale),
        "axis_rmse_x": _rmse(err[:, 0]),
        "axis_rmse_y": _rmse(err[:, 1]),
        "axis_rmse_z": _rmse(err[:, 2]),
        "seg_200_300": _fixed_segment(frames, aligned, gt_pos_by_frame, 200, 300),
        "seg_200_400": _fixed_segment(frames, aligned, gt_pos_by_frame, 200, 400),
        "seg_400_500": _fixed_segment(frames, aligned, gt_pos_by_frame, 400, 500),
        "seg_400_600": _fixed_segment(frames, aligned, gt_pos_by_frame, 400, 600),
        "gap_to_25": _rmse(norm) - 25.0,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
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
        pred_name, pred_path_s = args.pred.split("=", 1)
    else:
        pred_name, pred_path_s = Path(args.pred).stem, args.pred
    pred_path = Path(pred_path_s)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt))
    frames, raw_poses, _raw_pos = _load_tum_prediction(pred_path, gt_pos.shape[0])
    frames = frames.astype(np.int64)
    gt_pos_by_frame = gt_poses[:, :3, 3]
    pos = raw_poses[:, :3, 3]

    reset_windows = _reset_windows(gt_pos.shape[0], args.chunk_size, args.chunk_overlap, args.reset_every)
    semantic_windows = _semantic_windows(gt_pos.shape[0])
    candidates: Dict[str, np.ndarray] = {"baseline_raw": raw_poses}
    window_rows: List[Dict[str, object]] = []

    policy_specs = [
        ("NOGTPOSE_01_reset_neighbor_clip10", reset_windows, (0.90, 1.10), None, 1.0),
        ("NOGTPOSE_02_reset_global_clip10", reset_windows, (0.90, 1.10), None, 1.0),
        ("NOGTPOSE_03_reset_global_clip25", reset_windows, (0.80, 1.25), None, 1.0),
        ("NOGTPOSE_04_semantic_global_clip15", semantic_windows, (0.85, 1.15), None, 1.0),
        ("NOGTPOSE_05_reset_global_clip15", reset_windows, (0.85, 1.15), None, 1.0),
        ("NOGTPOSE_06_reset_global_clip10_pre600", reset_windows, (0.90, 1.10), _active_reset_indices(reset_windows, end_max=600), 1.0),
        ("NOGTPOSE_07_reset_global_clip15_pre600", reset_windows, (0.85, 1.15), _active_reset_indices(reset_windows, end_max=600), 1.0),
        ("NOGTPOSE_08_reset_neighbor_clip10_pre600", reset_windows, (0.90, 1.10), _active_reset_indices(reset_windows, end_max=600), 1.0),
        ("NOGTPOSE_09_reset_global_clip10_body600", reset_windows, (0.90, 1.10), _active_reset_indices(reset_windows, start_min=145, end_max=600), 1.0),
        ("NOGTPOSE_10_reset_global_clip12_pre600", reset_windows, (0.88, 1.12), _active_reset_indices(reset_windows, end_max=600), 1.0),
        ("NOGTPOSE_11_reset_global_clip18_pre600", reset_windows, (0.82, 1.18), _active_reset_indices(reset_windows, end_max=600), 1.0),
        ("NOGTPOSE_12_reset_global_clip20_pre600", reset_windows, (0.80, 1.20), _active_reset_indices(reset_windows, end_max=600), 1.0),
        ("NOGTPOSE_13_reset_global_clip25_pre600", reset_windows, (0.75, 1.25), _active_reset_indices(reset_windows, end_max=600), 1.0),
        ("NOGTPOSE_14_reset_global_clip15_body600", reset_windows, (0.85, 1.15), _active_reset_indices(reset_windows, start_min=145, end_max=600), 1.0),
        ("NOGTPOSE_15_reset_global_clip20_body600", reset_windows, (0.80, 1.20), _active_reset_indices(reset_windows, start_min=145, end_max=600), 1.0),
        ("NOGTPOSE_16_reset_global_clip30_pre600", reset_windows, (0.70, 1.30), _active_reset_indices(reset_windows, end_max=600), 1.0),
        ("NOGTPOSE_17_reset_global_clip35_pre600", reset_windows, (0.65, 1.35), _active_reset_indices(reset_windows, end_max=600), 1.0),
        ("NOGTPOSE_18_reset_global_clip40_pre600", reset_windows, (0.60, 1.40), _active_reset_indices(reset_windows, end_max=600), 1.0),
        ("NOGTPOSE_19_reset_global_clip25_body600", reset_windows, (0.75, 1.25), _active_reset_indices(reset_windows, start_min=145, end_max=600), 1.0),
        ("NOGTPOSE_20_reset_global_clip30_body600", reset_windows, (0.70, 1.30), _active_reset_indices(reset_windows, start_min=145, end_max=600), 1.0),
        ("NOGTPOSE_21_reset_global_clip32_body600", reset_windows, (0.68, 1.32), _active_reset_indices(reset_windows, start_min=145, end_max=600), 1.0),
        ("NOGTPOSE_22_reset_global_clip35_body600", reset_windows, (0.65, 1.35), _active_reset_indices(reset_windows, start_min=145, end_max=600), 1.0),
        ("NOGTPOSE_23_reset_global_clip40_body600", reset_windows, (0.60, 1.40), _active_reset_indices(reset_windows, start_min=145, end_max=600), 1.0),
        ("NOGTPOSE_24_reset_global_clip45_body600", reset_windows, (0.55, 1.45), _active_reset_indices(reset_windows, start_min=145, end_max=600), 1.0),
        ("NOGTPOSE_25_reset_global_clip35_body600_t095", reset_windows, (0.65, 1.35), _active_reset_indices(reset_windows, start_min=145, end_max=600), 0.95),
        ("NOGTPOSE_26_reset_global_clip35_body600_t090", reset_windows, (0.65, 1.35), _active_reset_indices(reset_windows, start_min=145, end_max=600), 0.90),
        ("NOGTPOSE_27_reset_global_clip35_body600_t105", reset_windows, (0.65, 1.35), _active_reset_indices(reset_windows, start_min=145, end_max=600), 1.05),
    ]
    for name, windows, clip, active_indices, target_multiplier in policy_specs:
        scales, rows = _policy_scales(
            name,
            pos,
            windows,
            clip,
            active_indices=active_indices,
            target_multiplier=target_multiplier,
        )
        candidates[name] = _scale_windows_continuous(raw_poses, windows, scales)
        window_rows.extend(rows)

    summary_rows: List[Dict[str, object]] = []
    for name, poses in candidates.items():
        _write_tum(out_dir / f"{name}.tum.txt", frames, poses)
        summary_rows.append(_summarize_aligned(name, frames, poses, gt_poses, gt_pos_by_frame))

    summary_rows.sort(key=lambda row: float(row["ate_rmse"]))
    _write_csv(out_dir / "nogt_pose_proxy_summary.csv", summary_rows)
    _write_csv(out_dir / "nogt_pose_proxy_window_scales.csv", window_rows)
    (out_dir / "nogt_pose_proxy_summary.json").write_text(
        json.dumps({"source_run": pred_name, "source_path": str(pred_path), "runs": summary_rows}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# V10 No-GT Pose Proxy",
        "",
        f"source_run = `{pred_name}`",
        f"source_path = `{pred_path}`",
        "",
        "| Run | ATE | gap_to_25 | FinalErr | YawRMSE | [200,300) | [200,400) | [400,600) | Sim3 scale |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['run']}` | {_fmt(row['ate_rmse'])} | {_fmt(row['gap_to_25'])} | "
            f"{_fmt(row['final_err'])} | {_fmt(row['yaw_rmse'])} | {_fmt(row['seg_200_300'])} | "
            f"{_fmt(row['seg_200_400'])} | {_fmt(row['seg_400_600'])} | {_fmt(row['sim3_scale'], 6)} |"
        )
    lines.extend(["", "## Gate Read", ""])
    best = summary_rows[0]
    baseline = next(row for row in summary_rows if row["run"] == "baseline_raw")
    gain = float(baseline["ate_rmse"]) - float(best["ate_rmse"])
    if best["run"] == "baseline_raw":
        lines.append("No-GT pose proxy did not improve the source trajectory.")
    else:
        lines.append(f"best_proxy = `{best['run']}` with ATE `{_fmt(best['ate_rmse'])}`, gain vs baseline `{_fmt(gain)}`.")
    if float(best["ate_rmse"]) <= 25.0:
        lines.append("No-GT proxy reaches target-25.")
    elif gain >= 0.20:
        lines.append("No-GT proxy has a measurable positive signal but does not reach target-25.")
    else:
        lines.append("No-GT proxy signal is too weak for target-25; the pose-scale oracle remains an upper bound, not an implemented method.")
    (out_dir / "nogt_pose_proxy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'nogt_pose_proxy_report.md'}")


if __name__ == "__main__":
    main()
