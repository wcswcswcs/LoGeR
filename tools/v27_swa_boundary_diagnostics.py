#!/usr/bin/env python3
"""Boundary diagnostics for v27 SWA semantic source/cache experiments.

This script is intentionally offline: it reads landed short-rollout
trajectories and compares candidate boundary-local errors against the matching
H9/reference rollout. It does not launch candidates or change any results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.kitti_trajectory_diagnostics import (  # noqa: E402
    _apply_alignment,
    _load_kitti_gt,
    _load_tum_prediction,
    _rmse,
    _umeyama_sim3,
)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _align(frames: np.ndarray, poses: np.ndarray, gt_pos: np.ndarray) -> np.ndarray:
    raw_pos = poses[:, :3, 3]
    matched_gt_pos = gt_pos[frames.astype(np.int64)]
    scale, rot, trans = _umeyama_sim3(raw_pos, matched_gt_pos, with_scale=True)
    return _apply_alignment(poses, scale, rot, trans)


def _pose_lookup(frames: np.ndarray, poses: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(frame): pose for frame, pose in zip(frames, poses)}


def _subset_reference(frames: np.ndarray, ref_lookup: Mapping[int, np.ndarray]) -> np.ndarray:
    poses: List[np.ndarray] = []
    for frame in frames.astype(np.int64):
        pose = ref_lookup.get(int(frame))
        if pose is None:
            raise KeyError(f"reference trajectory lacks frame {int(frame)}")
        poses.append(pose)
    return np.stack(poses, axis=0)


def _boundary_centers(first: int, last: int, chunk_size: int, overlap: int) -> List[int]:
    stride = chunk_size - overlap
    centers: List[int] = []
    cur = first + stride
    while cur <= last:
        centers.append(cur)
        cur += stride
    return centers


def _window_ate(
    frames: np.ndarray,
    aligned: np.ndarray,
    gt_pos: np.ndarray,
    center: int,
    radius: int,
) -> Tuple[float, int]:
    mask = (frames >= center - radius) & (frames < center + radius)
    count = int(mask.sum())
    if count < 3:
        return float("nan"), count
    idx = frames[mask].astype(np.int64)
    err = aligned[mask, :3, 3] - gt_pos[idx]
    return _rmse(np.linalg.norm(err, axis=1)), count


def _step_norms(frames: np.ndarray, poses: np.ndarray) -> Dict[int, float]:
    out: Dict[int, float] = {}
    order = np.argsort(frames)
    sorted_frames = frames[order].astype(np.int64)
    sorted_poses = poses[order]
    for i in range(1, len(sorted_frames)):
        prev = int(sorted_frames[i - 1])
        cur = int(sorted_frames[i])
        if cur != prev + 1:
            continue
        delta = sorted_poses[i, :3, 3] - sorted_poses[i - 1, :3, 3]
        out[cur] = float(np.linalg.norm(delta))
    return out


def _mean_finite(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _candidate_id_from_run(run: Path) -> str:
    name = run.name
    if "_SWA_SEM_" in name:
        tail = name.split("_SWA_SEM_", 1)[1]
        return "SWA_SEM_" + tail.split("_", 1)[0]
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--candidate-run", action="append", default=[])
    parser.add_argument("--gt-path", default="/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--improvement-threshold", type=float, default=0.10)
    args = parser.parse_args()

    ref_run = Path(args.reference_run)
    candidate_runs = [Path(p) for p in args.candidate_run]
    if not candidate_runs:
        raise SystemExit("No --candidate-run supplied")

    _gt_frames, gt_poses, gt_pos = _load_kitti_gt(Path(args.gt_path))
    ref_frames, ref_poses, _ref_quat = _load_tum_prediction(ref_run / "01.txt", gt_poses.shape[0])
    ref_lookup = _pose_lookup(ref_frames, ref_poses)

    boundary_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for run in candidate_runs:
        candidate_id = _candidate_id_from_run(run)
        frames, poses, _quat = _load_tum_prediction(run / "01.txt", gt_poses.shape[0])
        ref_subset = _subset_reference(frames, ref_lookup)
        aligned = _align(frames, poses, gt_pos)
        ref_aligned = _align(frames, ref_subset, gt_pos)
        cand_steps = _step_norms(frames, poses)
        ref_steps = _step_norms(frames, ref_subset)
        centers = _boundary_centers(int(frames[0]), int(frames[-1]), args.chunk_size, args.overlap)

        cand10: List[float] = []
        ref10: List[float] = []
        cand20: List[float] = []
        ref20: List[float] = []
        jump_deltas: List[float] = []
        for center in centers:
            b10, n10 = _window_ate(frames, aligned, gt_pos, center, 10)
            r10, _ = _window_ate(frames, ref_aligned, gt_pos, center, 10)
            b20, n20 = _window_ate(frames, aligned, gt_pos, center, 20)
            r20, _ = _window_ate(frames, ref_aligned, gt_pos, center, 20)
            cand_jump = cand_steps.get(center, float("nan"))
            ref_jump = ref_steps.get(center, float("nan"))
            jump_delta = cand_jump - ref_jump if math.isfinite(cand_jump) and math.isfinite(ref_jump) else float("nan")
            boundary_rows.append(
                {
                    "candidate_id": candidate_id,
                    "run_dir": str(run),
                    "boundary_frame": center,
                    "boundary_10f_ATE": b10,
                    "boundary_10f_H9_ATE": r10,
                    "boundary_10f_delta_vs_H9": b10 - r10 if math.isfinite(b10) and math.isfinite(r10) else float("nan"),
                    "boundary_10f_frame_count": n10,
                    "boundary_20f_ATE": b20,
                    "boundary_20f_H9_ATE": r20,
                    "boundary_20f_delta_vs_H9": b20 - r20 if math.isfinite(b20) and math.isfinite(r20) else float("nan"),
                    "boundary_20f_frame_count": n20,
                    "chunk_boundary_pose_jump": cand_jump,
                    "chunk_boundary_pose_jump_H9": ref_jump,
                    "chunk_boundary_pose_jump_delta_vs_H9": jump_delta,
                }
            )
            cand10.append(b10)
            ref10.append(r10)
            cand20.append(b20)
            ref20.append(r20)
            jump_deltas.append(jump_delta)

        mean_c10 = _mean_finite(cand10)
        mean_r10 = _mean_finite(ref10)
        mean_c20 = _mean_finite(cand20)
        mean_r20 = _mean_finite(ref20)
        improvement10 = (mean_r10 - mean_c10) / mean_r10 if math.isfinite(mean_r10) and mean_r10 > 0 else float("nan")
        improvement20 = (mean_r20 - mean_c20) / mean_r20 if math.isfinite(mean_r20) and mean_r20 > 0 else float("nan")
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "run_dir": str(run),
                "boundary_count": len(centers),
                "mean_boundary_10f_ATE": mean_c10,
                "mean_boundary_10f_H9_ATE": mean_r10,
                "mean_boundary_10f_delta_vs_H9": mean_c10 - mean_r10 if math.isfinite(mean_c10) and math.isfinite(mean_r10) else float("nan"),
                "mean_boundary_10f_improvement_ratio": improvement10,
                "mean_boundary_20f_ATE": mean_c20,
                "mean_boundary_20f_H9_ATE": mean_r20,
                "mean_boundary_20f_delta_vs_H9": mean_c20 - mean_r20 if math.isfinite(mean_c20) and math.isfinite(mean_r20) else float("nan"),
                "mean_boundary_20f_improvement_ratio": improvement20,
                "mean_pose_jump_delta_vs_H9": _mean_finite(jump_deltas),
                "h3_boundary_gate_pass": (
                    (math.isfinite(improvement10) and improvement10 >= args.improvement_threshold)
                    or (math.isfinite(improvement20) and improvement20 >= args.improvement_threshold)
                ),
            }
        )

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "swa_boundary_by_candidate_boundary.csv", boundary_rows)
    _write_csv(out_dir / "swa_boundary_summary.csv", summary_rows)
    gate_pass = sorted(row["candidate_id"] for row in summary_rows if row["h3_boundary_gate_pass"])
    best10 = min(
        summary_rows,
        key=lambda row: float(row["mean_boundary_10f_delta_vs_H9"])
        if math.isfinite(float(row["mean_boundary_10f_delta_vs_H9"]))
        else float("inf"),
        default=None,
    )
    _write_json(
        out_dir / "swa_boundary_summary.json",
        {
            "num_candidates": len(summary_rows),
            "boundary_gate_pass_candidates": gate_pass,
            "best_boundary_10f_candidate": best10["candidate_id"] if best10 else None,
            "best_boundary_10f_delta_vs_H9": best10["mean_boundary_10f_delta_vs_H9"] if best10 else None,
            "improvement_threshold": args.improvement_threshold,
            "counts_as_deployable_online_success": False,
        },
    )


if __name__ == "__main__":
    main()
