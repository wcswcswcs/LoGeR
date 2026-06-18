#!/usr/bin/env python3
"""Segment-level trajectory diagnostics for ACL2 v67 experiments.

This script intentionally recomputes local Sim(3) metrics from trajectories so
short-window claims do not get mixed with full-run ATE numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from kitti_trajectory_diagnostics import _load_kitti_gt, _load_tum_prediction, _umeyama_sim3
except ImportError:  # pragma: no cover
    from tools.kitti_trajectory_diagnostics import _load_kitti_gt, _load_tum_prediction, _umeyama_sim3


DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")


def _parse_pred(arg: str) -> Tuple[str, Path]:
    if "=" in arg:
        name, path = arg.split("=", 1)
        return name.strip(), Path(path)
    path = Path(arg)
    return path.parent.name or path.stem, path


def _parse_interval(text: str) -> Tuple[int, int]:
    if ":" in text:
        a, b = text.split(":", 1)
    elif "," in text:
        a, b = text.split(",", 1)
    else:
        raise ValueError(f"Interval must be START:END, got {text!r}")
    start, end = int(a), int(b)
    if end <= start:
        raise ValueError(f"Invalid interval {text!r}: end <= start")
    return start, end


def _safe_float(x: float) -> Optional[float]:
    return float(x) if math.isfinite(float(x)) else None


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def _trajectory_path(path: Path) -> Path:
    if path.is_dir():
        return path / "01.txt"
    return path


def _map_frames(frames: np.ndarray, start_frame: Optional[int], mode: str) -> np.ndarray:
    frames = frames.astype(np.int64, copy=True)
    if start_frame is None:
        return frames
    if mode == "local":
        return frames + int(start_frame)
    if mode == "absolute":
        return frames
    if mode != "auto":
        raise ValueError(f"Unknown frame mode: {mode}")
    if frames.size and int(frames.min()) < int(start_frame):
        return frames + int(start_frame)
    return frames


def _fit_eval_sim3(
    frames: np.ndarray,
    pred_pos: np.ndarray,
    gt_pos: np.ndarray,
    fit_start: int,
    fit_end: int,
    eval_start: int,
    eval_end: int,
) -> Dict[str, Any]:
    fit_mask = (frames >= fit_start) & (frames < fit_end)
    eval_mask = (frames >= eval_start) & (frames < eval_end)
    out: Dict[str, Any] = {
        "fit_start": int(fit_start),
        "fit_end": int(fit_end),
        "eval_start": int(eval_start),
        "eval_end": int(eval_end),
        "fit_n": int(fit_mask.sum()),
        "eval_n": int(eval_mask.sum()),
        "ate_rmse_m": None,
        "sim3_scale": None,
        "valid": False,
    }
    if out["fit_n"] < 3 or out["eval_n"] < 1:
        out["reason"] = "insufficient_frames"
        return out
    fit_frames = frames[fit_mask]
    eval_frames = frames[eval_mask]
    try:
        scale, rot, trans = _umeyama_sim3(pred_pos[fit_mask], gt_pos[fit_frames], with_scale=True)
    except Exception as exc:
        out["reason"] = f"sim3_fit_failed:{type(exc).__name__}"
        return out
    aligned = (scale * (rot @ pred_pos[eval_mask].T)).T + trans
    errors = np.linalg.norm(aligned - gt_pos[eval_frames], axis=1)
    out.update({
        "ate_rmse_m": _safe_float(_rmse(errors)),
        "ate_mean_m": _safe_float(float(np.mean(errors))),
        "ate_p90_m": _safe_float(float(np.percentile(errors, 90))),
        "sim3_scale": _safe_float(float(scale)),
        "valid": True,
    })
    return out


def _window_rows(
    run_name: str,
    frames: np.ndarray,
    pred_pos: np.ndarray,
    gt_pos: np.ndarray,
    intervals: Sequence[Tuple[int, int]],
    row_type: str,
) -> List[Dict[str, Any]]:
    rows = []
    for start, end in intervals:
        row = _fit_eval_sim3(frames, pred_pos, gt_pos, start, end, start, end)
        row.update({"run": run_name, "type": row_type, "start": int(start), "end": int(end)})
        rows.append(row)
    return rows


def _rolling_intervals(first: int, last_exclusive: int, length: int, stride: int) -> List[Tuple[int, int]]:
    intervals = []
    start = int(first)
    while start + length <= last_exclusive:
        intervals.append((start, start + length))
        start += stride
    return intervals


def _chunk_rows(
    run_name: str,
    frames: np.ndarray,
    pred_pos: np.ndarray,
    gt_pos: np.ndarray,
    chunk_size: int,
    overlap: int,
    head_len: int,
) -> List[Dict[str, Any]]:
    if chunk_size <= 0:
        return []
    step = max(1, chunk_size - max(0, overlap))
    pred_first, pred_last = int(frames.min()), int(frames.max()) + 1
    first_chunk = max(0, pred_first // step - 1)
    last_chunk = (pred_last + step - 1) // step + 1
    rows: List[Dict[str, Any]] = []
    for chunk_idx in range(first_chunk, last_chunk):
        start = chunk_idx * step
        end = start + chunk_size
        if end <= pred_first or start >= pred_last:
            continue
        whole = _fit_eval_sim3(frames, pred_pos, gt_pos, start, end, start, end)
        third = max(3, chunk_size // 3)
        parts = {
            "head": (start, min(start + third, end)),
            "mid": (start + third, min(start + 2 * third, end)),
            "tail": (max(start + 2 * third, end - third), end),
        }
        scales: Dict[str, Optional[float]] = {}
        rmses: Dict[str, Optional[float]] = {}
        for label, (a, b) in parts.items():
            part = _fit_eval_sim3(frames, pred_pos, gt_pos, a, b, a, b)
            scales[f"{label}_scale"] = part.get("sim3_scale")
            rmses[f"{label}_ate_rmse_m"] = part.get("ate_rmse_m")
        finite_scales = np.asarray([v for v in scales.values() if v is not None], dtype=np.float64)
        scale_cv = float(np.std(finite_scales) / max(abs(float(np.mean(finite_scales))), 1e-12)) if finite_scales.size else float("nan")
        head = min(head_len, chunk_size)
        head_to_tail = _fit_eval_sim3(frames, pred_pos, gt_pos, start, start + head, end - head, end)
        head_to_future = _fit_eval_sim3(frames, pred_pos, gt_pos, start, start + head, end, end + head)
        overlap_to_future = _fit_eval_sim3(
            frames,
            pred_pos,
            gt_pos,
            max(start, end - overlap),
            end,
            end,
            end + head,
        )
        row = {
            "run": run_name,
            "chunk_idx": int(chunk_idx),
            "start": int(start),
            "end": int(end),
            "whole_ate_rmse_m": whole.get("ate_rmse_m"),
            "whole_scale": whole.get("sim3_scale"),
            "whole_fit_n": whole.get("fit_n"),
            "scale_cv_head_mid_tail": _safe_float(scale_cv),
            "head_to_tail_ate_rmse_m": head_to_tail.get("ate_rmse_m"),
            "head_to_tail_scale": head_to_tail.get("sim3_scale"),
            "head_to_future_ate_rmse_m": head_to_future.get("ate_rmse_m"),
            "head_to_future_eval_n": head_to_future.get("eval_n"),
            "overlap_to_future_ate_rmse_m": overlap_to_future.get("ate_rmse_m"),
            "overlap_to_future_fit_n": overlap_to_future.get("fit_n"),
            "overlap_to_future_eval_n": overlap_to_future.get("eval_n"),
        }
        row.update(scales)
        row.update(rmses)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(values: Iterable[Any]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None


def _run_summary(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--pred", action="append", required=True, help="NAME=run_dir_or_01.txt; repeatable")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pred-start-frame", type=int, default=None)
    parser.add_argument("--frame-mode", choices=["auto", "local", "absolute"], default="auto")
    parser.add_argument("--interval", action="append", default=[])
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--head-len", type=int, default=10)
    parser.add_argument("--rolling-len", type=int, default=100)
    parser.add_argument("--rolling-stride", type=int, default=29)
    args = parser.parse_args()

    _, _, gt_pos = _load_kitti_gt(args.gt)
    interval_specs = [_parse_interval(x) for x in args.interval]
    all_interval_rows: List[Dict[str, Any]] = []
    all_chunk_rows: List[Dict[str, Any]] = []
    all_rolling_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "gt": str(args.gt),
        "pred_start_frame": args.pred_start_frame,
        "frame_mode": args.frame_mode,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "head_len": args.head_len,
        "rolling_len": args.rolling_len,
        "rolling_stride": args.rolling_stride,
        "runs": {},
    }

    for pred_arg in args.pred:
        name, raw_path = _parse_pred(pred_arg)
        path = _trajectory_path(raw_path)
        frames0, _, pred_pos = _load_tum_prediction(path, gt_pos.shape[0])
        frames = _map_frames(frames0, args.pred_start_frame, args.frame_mode)
        valid = (frames >= 0) & (frames < gt_pos.shape[0])
        frames = frames[valid]
        pred_pos = pred_pos[valid]
        if frames.shape[0] < 3:
            raise ValueError(f"{name}: need at least 3 valid frames after mapping, got {frames.shape[0]}")

        overall = _fit_eval_sim3(frames, pred_pos, gt_pos, int(frames.min()), int(frames.max()) + 1, int(frames.min()), int(frames.max()) + 1)
        interval_rows = _window_rows(name, frames, pred_pos, gt_pos, interval_specs, "local_sim3_interval")
        rolling = _rolling_intervals(int(frames.min()), int(frames.max()) + 1, args.rolling_len, args.rolling_stride)
        rolling_rows = _window_rows(name, frames, pred_pos, gt_pos, rolling, "local_sim3_rolling")
        chunk_rows = _chunk_rows(name, frames, pred_pos, gt_pos, args.chunk_size, args.chunk_overlap, args.head_len)
        all_interval_rows.extend(interval_rows)
        all_rolling_rows.extend(rolling_rows)
        all_chunk_rows.extend(chunk_rows)
        summary["runs"][name] = {
            "path": str(path),
            "raw_frame_min": int(frames0.min()),
            "raw_frame_max": int(frames0.max()),
            "mapped_frame_min": int(frames.min()),
            "mapped_frame_max": int(frames.max()),
            "frame_count": int(frames.shape[0]),
            "overall_local_sim3": overall,
            "interval_ate_rmse_m": {f"{r['start']}:{r['end']}": r.get("ate_rmse_m") for r in interval_rows},
            "chunk_whole_ate": _run_summary(chunk_rows, "whole_ate_rmse_m"),
            "chunk_scale_cv": _run_summary(chunk_rows, "scale_cv_head_mid_tail"),
            "chunk_head_to_tail_ate": _run_summary(chunk_rows, "head_to_tail_ate_rmse_m"),
            "chunk_head_to_future_ate": _run_summary(chunk_rows, "head_to_future_ate_rmse_m"),
            "chunk_overlap_to_future_ate": _run_summary(chunk_rows, "overlap_to_future_ate_rmse_m"),
            "rolling_ate": _run_summary(rolling_rows, "ate_rmse_m"),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "interval_metrics.csv", all_interval_rows)
    _write_csv(args.out_dir / "chunk_metrics.csv", all_chunk_rows)
    _write_csv(args.out_dir / "rolling_metrics.csv", all_rolling_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
