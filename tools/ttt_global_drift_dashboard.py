#!/usr/bin/env python3
"""Generate v8 global drift diagnostics for KITTI trajectory runs."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import (  # noqa: E402
    _angle_diff_deg,
    _apply_alignment,
    _load_kitti_gt,
    _load_tum_prediction,
    _rmse,
    _umeyama_sim3,
    _yaw_from_pose,
)


@dataclass
class Run:
    name: str
    frames: np.ndarray
    aligned_poses: np.ndarray
    gt_poses: np.ndarray
    aligned_pos: np.ndarray
    gt_pos: np.ndarray
    err: np.ndarray
    err_norm: np.ndarray
    yaw_err: np.ndarray
    scale: float


def _parse_run(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        p = Path(spec)
        return p.parent.name or p.stem, p
    name, path = spec.split("=", 1)
    p = Path(path)
    if p.is_dir():
        p = p / "01.txt"
    return name, p


def _parse_segments(spec: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        start, end = part.split(":", 1)
        out.append((int(start), int(end)))
    return out


def _chunks(n_frames: int, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, int, int]]:
    if chunk_size <= 0:
        return []
    stride = max(1, chunk_size - chunk_overlap)
    chunks = []
    idx = 0
    for start in range(0, n_frames, stride):
        end = min(start + chunk_size, n_frames)
        if end - start < max(5, chunk_size // 2):
            break
        chunks.append((idx, start, end))
        idx += 1
        if end >= n_frames:
            break
    return chunks


def _load_run(name: str, path: Path, gt_poses_all: np.ndarray, gt_pos_all: np.ndarray, align: str) -> Run:
    frames, raw_poses, raw_pos = _load_tum_prediction(path, gt_pos_all.shape[0])
    gt_pos = gt_pos_all[frames]
    if align == "none":
        scale, R, t = 1.0, np.eye(3), np.zeros(3)
    else:
        scale, R, t = _umeyama_sim3(raw_pos, gt_pos, with_scale=(align == "sim3"))
    aligned_poses = _apply_alignment(raw_poses, scale, R, t)
    aligned_pos = aligned_poses[:, :3, 3]
    err = aligned_pos - gt_pos
    err_norm = np.linalg.norm(err, axis=1)
    gt_poses = gt_poses_all[frames]
    yaw_err = _angle_diff_deg(_yaw_from_pose(aligned_poses, "xz"), _yaw_from_pose(gt_poses, "xz"))
    return Run(name, frames, aligned_poses, gt_poses, aligned_pos, gt_pos, err, err_norm, yaw_err, scale)


def _window_values(run: Run, start: int, end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (run.frames >= start) & (run.frames < end)
    return run.err[mask], run.err_norm[mask], run.yaw_err[mask]


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", required=True)
    p.add_argument("--run", action="append", required=True, help="NAME=path/to/01.txt or run directory")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--align", choices=["sim3", "se3", "none"], default="sim3")
    p.add_argument("--segments", default="100:200,200:250,200:300,200:400,300:400,400:500,400:600,600:800")
    p.add_argument("--segment_lengths", default="50,100,200")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--chunk_overlap", type=int, default=3)
    p.add_argument("--reset_every", type=int, default=5)
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, gt_poses_all, gt_pos_all = _load_kitti_gt(Path(args.gt))
    runs = [_load_run(name, path, gt_poses_all, gt_pos_all, args.align) for name, path in map(_parse_run, args.run)]
    segments = _parse_segments(args.segments)
    segment_lengths = [int(x) for x in args.segment_lengths.split(",") if x.strip()]
    chunks = _chunks(gt_pos_all.shape[0], args.chunk_size, args.chunk_overlap)

    global_rows = []
    for run in runs:
        axis = [_rmse(run.err[:, i]) for i in range(3)]
        global_rows.append(
            {
                "run_id": run.name,
                "ATE": _rmse(run.err_norm),
                "FinalErr": float(run.err_norm[-1]),
                "YawRMSE": _rmse(run.yaw_err),
                "Sim3Scale": run.scale,
                "RMSE_x": axis[0],
                "RMSE_y": axis[1],
                "RMSE_z": axis[2],
            }
        )
    _write_csv(out_dir / "global_metrics.csv", global_rows, list(global_rows[0].keys()))

    segment_rows = []
    for run in runs:
        for start, end in segments:
            err, err_norm, yaw = _window_values(run, start, end)
            segment_rows.append(
                {
                    "run_id": run.name,
                    "segment_start": start,
                    "segment_end": end,
                    "ATE_segment": _rmse(err_norm) if err_norm.size else float("nan"),
                    "Yaw_segment": _rmse(yaw) if yaw.size else float("nan"),
                    "RMSE_x": _rmse(err[:, 0]) if err.size else float("nan"),
                    "RMSE_y": _rmse(err[:, 1]) if err.size else float("nan"),
                    "RMSE_z": _rmse(err[:, 2]) if err.size else float("nan"),
                }
            )
        for length in segment_lengths:
            for start in range(0, gt_pos_all.shape[0], length):
                end = min(start + length, gt_pos_all.shape[0])
                if end - start < max(5, length // 2):
                    continue
                err, err_norm, yaw = _window_values(run, start, end)
                segment_rows.append(
                    {
                        "run_id": run.name,
                        "segment_start": start,
                        "segment_end": end,
                        "ATE_segment": _rmse(err_norm) if err_norm.size else float("nan"),
                        "Yaw_segment": _rmse(yaw) if yaw.size else float("nan"),
                        "RMSE_x": _rmse(err[:, 0]) if err.size else float("nan"),
                        "RMSE_y": _rmse(err[:, 1]) if err.size else float("nan"),
                        "RMSE_z": _rmse(err[:, 2]) if err.size else float("nan"),
                    }
                )
    _write_csv(out_dir / "segment_metrics.csv", segment_rows, list(segment_rows[0].keys()))

    reset_rows = []
    groups = [chunks[i : i + args.reset_every] for i in range(0, len(chunks), args.reset_every)]
    for run in runs:
        prev_end_err = float("nan")
        for group_id, group in enumerate(groups):
            if not group:
                continue
            chunk_start = group[0][0]
            chunk_end = group[-1][0]
            start = group[0][1]
            end = group[-1][2]
            err, err_norm, yaw = _window_values(run, start, end)
            start_err = float(err_norm[0]) if err_norm.size else float("nan")
            end_err = float(err_norm[-1]) if err_norm.size else float("nan")
            reset_rows.append(
                {
                    "run_id": run.name,
                    "reset_group_id": group_id,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "frame_start": start,
                    "frame_end": end,
                    "ATE_group": _rmse(err_norm) if err_norm.size else float("nan"),
                    "FinalErr_group": end_err,
                    "Yaw_group": _rmse(yaw) if yaw.size else float("nan"),
                    "BoundaryJump_prev": abs(start_err - prev_end_err) if np.isfinite(prev_end_err) else float("nan"),
                    "BoundaryJump_next": float("nan"),
                }
            )
            prev_end_err = end_err
    _write_csv(out_dir / "reset_group_metrics.csv", reset_rows, list(reset_rows[0].keys()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_trajectory(fname: str, axes: Tuple[int, int]) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(gt_pos_all[:, axes[0]], gt_pos_all[:, axes[1]], "--", color="black", linewidth=1.0, label="GT")
        for run in runs:
            ax.plot(run.aligned_pos[:, axes[0]], run.aligned_pos[:, axes[1]], linewidth=1.0, label=run.name)
        ax.axis("equal")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=args.dpi)
        plt.close(fig)

    save_trajectory("trajectory_xy_overlay.png", (0, 1))
    save_trajectory("trajectory_xz_overlay.png", (0, 2))

    fig, ax = plt.subplots(figsize=(10, 4))
    for run in runs:
        ax.plot(run.frames, run.err_norm, linewidth=0.9, label=run.name)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Translation error (m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "per_frame_translation_error.png", dpi=args.dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for run in runs:
        ax.plot(run.frames, run.yaw_err, linewidth=0.9, label=run.name)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Yaw error (deg)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_yaw_drift.png", dpi=args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for run in runs:
        for i, axis in enumerate("xyz"):
            axes[i].plot(run.frames, run.err[:, i], linewidth=0.9, label=run.name)
            axes[i].set_ylabel(f"{axis} err (m)")
            axes[i].grid(alpha=0.25)
    axes[-1].set_xlabel("Frame")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_xyz_drift.png", dpi=args.dpi)
    plt.close(fig)

    for length in (100, 200):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        starts = sorted({r["segment_start"] for r in segment_rows if r["segment_end"] - r["segment_start"] == length})
        width = 0.8 / max(len(runs), 1)
        x = np.arange(len(starts))
        for i, run in enumerate(runs):
            vals = []
            for start in starts:
                match = [
                    r for r in segment_rows
                    if r["run_id"] == run.name and r["segment_start"] == start and r["segment_end"] == start + length
                ]
                vals.append(match[0]["ATE_segment"] if match else np.nan)
            ax.bar(x + (i - (len(runs) - 1) / 2) * width, vals, width=width, label=run.name)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in starts], rotation=45, ha="right")
        ax.set_ylabel(f"{length}f ATE (m)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"per_{length}f_ATE_bar.png", dpi=args.dpi)
        plt.close(fig)

    for value_key, fname in (("ATE_group", "reset_group_ATE_heatmap.png"), ("BoundaryJump_prev", "reset_group_boundary_jump_heatmap.png")):
        matrix = []
        for run in runs:
            vals = [r[value_key] for r in reset_rows if r["run_id"] == run.name]
            matrix.append(vals)
        fig, ax = plt.subplots(figsize=(10, 2.5 + 0.3 * len(runs)))
        im = ax.imshow(np.asarray(matrix, dtype=float), aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(runs)))
        ax.set_yticklabels([r.name for r in runs])
        ax.set_xlabel("Reset group")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=args.dpi)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot([r.name for r in runs], [r.scale for r in runs], marker="o")
    ax.set_ylabel("Global Sim3 scale")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "sim3_scale_over_time.png", dpi=args.dpi)
    plt.close(fig)

    print(f"Saved v8 global drift dashboard to {out_dir}")


if __name__ == "__main__":
    main()
