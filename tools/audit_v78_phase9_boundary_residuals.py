#!/usr/bin/env python3
"""Audit v78 Phase9 runs at the actual target chunk boundary.

The Phase9 smoke runner uses a two-chunk context window for chunk06
(`start_frame=145`, `target_start_frame=174`).  The existing smoke evaluator is
trajectory-only and fits its overlap-to-future proxy on the first emitted
frames. This script adds a diagnostic-only view at the actual target boundary:
previous tail frames 171-173 into current head/future frames 174+.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

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


DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--target-start-frame", type=int, default=174)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--run", action="append", required=True, help="NAME=run_dir containing 01.txt")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_run(text: str) -> tuple[str, Path]:
    if "=" not in str(text):
        path = Path(text)
        return path.name, path
    name, path = str(text).split("=", 1)
    return name.strip(), Path(path)


def _indices_for_frames(frames: np.ndarray, want: list[int]) -> np.ndarray:
    positions = {int(frame): idx for idx, frame in enumerate(frames.tolist())}
    return np.asarray([positions[f] for f in want if f in positions], dtype=np.int64)


def _fit_eval(
    *,
    frames: np.ndarray,
    raw_pos: np.ndarray,
    gt_pos_all: np.ndarray,
    fit_frames: list[int],
    eval_frames: list[int],
) -> dict[str, Any]:
    fit_idx = _indices_for_frames(frames, fit_frames)
    eval_idx = _indices_for_frames(frames, eval_frames)
    out: dict[str, Any] = {
        "fit_frames": fit_frames,
        "eval_frames": eval_frames,
        "fit_n": int(fit_idx.size),
        "eval_n": int(eval_idx.size),
        "rmse_m": None,
        "mean_m": None,
        "p90_m": None,
        "sim3_scale": None,
        "valid": False,
    }
    if int(fit_idx.size) < 3 or int(eval_idx.size) < 1:
        out["reason"] = "insufficient_frames"
        return out
    try:
        scale, rot, trans = _umeyama_sim3(raw_pos[fit_idx], gt_pos_all[frames[fit_idx]], with_scale=True)
    except Exception as exc:
        out["reason"] = f"fit_failed:{type(exc).__name__}"
        return out
    aligned = (scale * (rot @ raw_pos[eval_idx].T)).T + trans
    err = np.linalg.norm(aligned - gt_pos_all[frames[eval_idx]], axis=1)
    out.update({
        "rmse_m": float(_rmse(err)),
        "mean_m": float(np.mean(err)),
        "p90_m": float(np.percentile(err, 90)),
        "sim3_scale": float(scale),
        "valid": True,
    })
    return out


def _eval_run(
    name: str,
    run_dir: Path,
    *,
    gt_pos_all: np.ndarray,
    gt_poses_all: np.ndarray,
    target_start: int,
    overlap: int,
) -> dict[str, Any]:
    traj = run_dir / "01.txt"
    frames, raw_poses, raw_pos = _load_tum_prediction(traj, gt_pos_all.shape[0])
    if int(frames.size) < 3:
        raise ValueError(f"{traj}: need at least 3 frames")
    scale, rot, trans = _umeyama_sim3(raw_pos, gt_pos_all[frames], with_scale=True)
    aligned = _apply_alignment(raw_poses, scale, rot, trans)
    aligned_pos = aligned[:, :3, 3]
    err = np.linalg.norm(aligned_pos - gt_pos_all[frames], axis=1)

    source_tail_frames = list(range(int(target_start) - int(overlap), int(target_start)))
    query_head_frames = list(range(int(target_start), int(target_start) + int(overlap)))
    future_frames = [int(f) for f in frames.tolist() if int(f) >= int(target_start)]
    after_head_frames = [int(f) for f in frames.tolist() if int(f) >= int(target_start) + int(overlap)]
    source_tail_idx = _indices_for_frames(frames, source_tail_frames)
    query_head_idx = _indices_for_frames(frames, query_head_frames)
    future_idx = _indices_for_frames(frames, future_frames)
    after_head_idx = _indices_for_frames(frames, after_head_frames)

    boundary_step_error = None
    prev_frame = int(target_start) - 1
    if prev_frame in set(int(x) for x in frames.tolist()) and int(target_start) in set(int(x) for x in frames.tolist()):
        prev_idx = int(np.where(frames == prev_frame)[0][0])
        cur_idx = int(np.where(frames == int(target_start))[0][0])
        pred_step = aligned_pos[cur_idx] - aligned_pos[prev_idx]
        gt_step = gt_pos_all[int(target_start)] - gt_pos_all[prev_frame]
        boundary_step_error = float(np.linalg.norm(pred_step - gt_step))

    tail_to_head = _fit_eval(
        frames=frames,
        raw_pos=raw_pos,
        gt_pos_all=gt_pos_all,
        fit_frames=source_tail_frames,
        eval_frames=query_head_frames,
    )
    tail_to_future = _fit_eval(
        frames=frames,
        raw_pos=raw_pos,
        gt_pos_all=gt_pos_all,
        fit_frames=source_tail_frames,
        eval_frames=future_frames,
    )
    tail_to_after_head = _fit_eval(
        frames=frames,
        raw_pos=raw_pos,
        gt_pos_all=gt_pos_all,
        fit_frames=source_tail_frames,
        eval_frames=after_head_frames,
    )

    def _err_rmse(idx: np.ndarray) -> float | None:
        if int(idx.size) <= 0:
            return None
        return float(_rmse(err[idx]))

    row = {
        "run": name,
        "run_dir": str(run_dir),
        "trajectory": str(traj),
        "frame_start": int(frames.min()),
        "frame_end_exclusive": int(frames.max()) + 1,
        "target_start_frame": int(target_start),
        "source_tail_frames": source_tail_frames,
        "query_head_frames": query_head_frames,
        "future_frame_start": int(target_start),
        "after_head_frame_start": int(target_start) + int(overlap),
        "global_sim3_scale": float(scale),
        "global_all_rmse_m": float(_rmse(err)),
        "global_source_tail_rmse_m": _err_rmse(source_tail_idx),
        "global_query_head_rmse_m": _err_rmse(query_head_idx),
        "global_future_from_boundary_rmse_m": _err_rmse(future_idx),
        "global_after_head_future_rmse_m": _err_rmse(after_head_idx),
        "global_query_minus_source_rmse_m": (
            _err_rmse(query_head_idx) - _err_rmse(source_tail_idx)
            if _err_rmse(query_head_idx) is not None and _err_rmse(source_tail_idx) is not None
            else None
        ),
        "boundary_step_error_m": boundary_step_error,
        "tail3_to_head3_sim3_rmse_m": tail_to_head.get("rmse_m"),
        "tail3_to_future_from_boundary_sim3_rmse_m": tail_to_future.get("rmse_m"),
        "tail3_to_after_head_future_sim3_rmse_m": tail_to_after_head.get("rmse_m"),
        "tail3_fit_sim3_scale": tail_to_head.get("sim3_scale"),
        "tail3_to_head3_valid": tail_to_head.get("valid"),
        "tail3_to_future_valid": tail_to_future.get("valid"),
        "tail3_to_after_head_valid": tail_to_after_head.get("valid"),
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (list, dict)) else row.get(key)
                for key in keys
            })


def _build_comparisons(rows: list[dict[str, Any]], baseline: str, controls: list[str]) -> dict[str, Any]:
    by_name = {str(row["run"]): row for row in rows}
    metrics = [
        "global_query_head_rmse_m",
        "global_future_from_boundary_rmse_m",
        "global_after_head_future_rmse_m",
        "global_query_minus_source_rmse_m",
        "boundary_step_error_m",
        "tail3_to_head3_sim3_rmse_m",
        "tail3_to_future_from_boundary_sim3_rmse_m",
        "tail3_to_after_head_future_sim3_rmse_m",
    ]
    out: dict[str, Any] = {}
    base = by_name.get(baseline) if baseline else None
    for row in rows:
        name = str(row["run"])
        if name == baseline:
            continue
        comp: dict[str, Any] = {}
        for key in metrics:
            cand = _finite(row.get(key))
            base_v = _finite(base.get(key)) if base else None
            ctrl_vals = [_finite(by_name.get(c, {}).get(key)) for c in controls]
            ctrl_vals = [v for v in ctrl_vals if v is not None]
            best_control = min(ctrl_vals) if ctrl_vals else None
            comp[key] = {
                "candidate": cand,
                "baseline": base_v,
                "best_control": best_control,
                "candidate_minus_baseline": cand - base_v if cand is not None and base_v is not None else None,
                "candidate_minus_best_control": cand - best_control if cand is not None and best_control is not None else None,
                "beats_controls": cand is not None and best_control is not None and cand < best_control,
            }
        out[name] = comp
    return out


def main() -> None:
    args = parse_args()
    _, gt_poses_all, gt_pos_all = _load_kitti_gt(args.gt)
    runs = [_parse_run(spec) for spec in args.run]
    rows = [
        _eval_run(
            name,
            run_dir,
            gt_pos_all=gt_pos_all,
            gt_poses_all=gt_poses_all,
            target_start=int(args.target_start_frame),
            overlap=int(args.overlap),
        )
        for name, run_dir in runs
    ]
    payload = {
        "schema": "acl2_v78_phase9_boundary_residual_audit_v1",
        "gt": str(args.gt),
        "target_start_frame": int(args.target_start_frame),
        "overlap": int(args.overlap),
        "runs": rows,
        "baseline": str(args.baseline),
        "controls": list(args.control),
        "comparisons": _build_comparisons(rows, str(args.baseline), list(args.control)),
        "decision": {
            "diagnostic_only": True,
            "method_gate_claimed": False,
            "reason": "Boundary residual attribution does not replace Phase9 method gate.",
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.out_csv, rows)
    print(json.dumps(payload["comparisons"], indent=2, ensure_ascii=False, sort_keys=True))
    print(f"wrote_json={args.out_json}")
    print(f"wrote_csv={args.out_csv}")


if __name__ == "__main__":
    main()
