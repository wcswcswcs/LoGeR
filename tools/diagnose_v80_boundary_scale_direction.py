#!/usr/bin/env python3
"""Diagnose whether post-boundary scale direction can explain a v80 merge failure.

This is a diagnostic-only tool.  It perturbs an existing native trajectory after
the target boundary and reuses the boundary metrics from the v78/v80 audits.  It
does not claim a runnable method gate because the perturbation is synthetic.
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


DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/00.txt")
KEY_METRICS = [
    "global_future_from_boundary_rmse_m",
    "global_after_head_future_rmse_m",
    "tail3_to_future_from_boundary_sim3_rmse_m",
    "boundary_step_error_m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing 01.txt, or a TUM file")
    parser.add_argument("--target-start-frame", type=int, required=True)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--actual-candidate-scale", type=float, default=None)
    parser.add_argument("--scale-factor", action="append", type=float, default=[])
    parser.add_argument(
        "--anchor-mode",
        action="append",
        choices=["origin", "boundary_prev", "source_tail_centroid"],
        default=[],
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def _trajectory_path(path: Path) -> Path:
    return path / "01.txt" if path.is_dir() else path


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
        "fit_n": int(fit_idx.size),
        "eval_n": int(eval_idx.size),
        "rmse_m": None,
        "sim3_scale": None,
        "valid": False,
    }
    if int(fit_idx.size) < 3 or int(eval_idx.size) < 1:
        out["reason"] = "insufficient_frames"
        return out
    try:
        scale, rot, trans = _umeyama_sim3(raw_pos[fit_idx], gt_pos_all[frames[fit_idx]], with_scale=True)
    except Exception as exc:  # pragma: no cover - diagnostic failure payload
        out["reason"] = f"fit_failed:{type(exc).__name__}"
        return out
    aligned = (scale * (rot @ raw_pos[eval_idx].T)).T + trans
    err = np.linalg.norm(aligned - gt_pos_all[frames[eval_idx]], axis=1)
    out.update({"rmse_m": float(_rmse(err)), "sim3_scale": float(scale), "valid": True})
    return out


def _anchor(
    mode: str,
    *,
    frames: np.ndarray,
    raw_pos: np.ndarray,
    target_start: int,
    overlap: int,
) -> np.ndarray:
    if mode == "origin":
        return np.zeros(3, dtype=np.float64)
    if mode == "boundary_prev":
        idx = _indices_for_frames(frames, [int(target_start) - 1])
        if int(idx.size) != 1:
            raise ValueError(f"missing boundary previous frame {int(target_start) - 1}")
        return raw_pos[int(idx[0])].copy()
    if mode == "source_tail_centroid":
        source_tail = list(range(int(target_start) - int(overlap), int(target_start)))
        idx = _indices_for_frames(frames, source_tail)
        if int(idx.size) < 1:
            raise ValueError(f"missing source tail frames {source_tail}")
        return raw_pos[idx].mean(axis=0)
    raise ValueError(f"unknown anchor mode: {mode}")


def _perturb_post_boundary(
    poses: np.ndarray,
    frames: np.ndarray,
    *,
    target_start: int,
    factor: float,
    anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out = poses.copy()
    mask = frames >= int(target_start)
    pos = out[:, :3, 3].copy()
    pos[mask] = anchor[None] + float(factor) * (pos[mask] - anchor[None])
    out[:, :3, 3] = pos
    return out, pos


def _eval_positions(
    *,
    frames: np.ndarray,
    raw_poses: np.ndarray,
    raw_pos: np.ndarray,
    gt_pos_all: np.ndarray,
    target_start: int,
    overlap: int,
) -> dict[str, Any]:
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

    return {
        "global_sim3_scale": float(scale),
        "global_all_rmse_m": float(_rmse(err)),
        "global_source_tail_rmse_m": _err_rmse(source_tail_idx),
        "global_query_head_rmse_m": _err_rmse(query_head_idx),
        "global_future_from_boundary_rmse_m": _err_rmse(future_idx),
        "global_after_head_future_rmse_m": _err_rmse(after_head_idx),
        "boundary_step_error_m": boundary_step_error,
        "tail3_to_head3_sim3_rmse_m": tail_to_head.get("rmse_m"),
        "tail3_to_future_from_boundary_sim3_rmse_m": tail_to_future.get("rmse_m"),
        "tail3_to_after_head_future_sim3_rmse_m": tail_to_after_head.get("rmse_m"),
        "tail3_fit_sim3_scale": tail_to_head.get("sim3_scale"),
    }


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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
            writer.writerow(row)


def _default_scales(actual: float | None, user_scales: list[float]) -> list[float]:
    scales = [
        0.990,
        0.995,
        0.9975,
        0.999,
        1.0,
        1.001,
        1.0025,
        1.005,
        1.010,
    ]
    if actual is not None:
        scales.append(float(actual))
    scales.extend(float(x) for x in user_scales)
    return sorted({round(float(x), 12) for x in scales})


def _build_summary(rows: list[dict[str, Any]], actual_scale: float | None) -> dict[str, Any]:
    by_anchor: dict[str, Any] = {}
    for anchor_mode in sorted({str(row["anchor_mode"]) for row in rows}):
        anchor_rows = [row for row in rows if str(row["anchor_mode"]) == anchor_mode]
        native = min(anchor_rows, key=lambda row: abs(float(row["scale_factor"]) - 1.0))
        native_metrics = {key: _finite(native.get(key)) for key in KEY_METRICS}
        metric_best: dict[str, Any] = {}
        for metric in KEY_METRICS:
            valid = [row for row in anchor_rows if _finite(row.get(metric)) is not None]
            if not valid:
                continue
            best = min(valid, key=lambda row: float(row[metric]))
            value = float(best[metric])
            native_value = native_metrics.get(metric)
            metric_best[metric] = {
                "best_scale_factor": float(best["scale_factor"]),
                "best_value": value,
                "native_value": native_value,
                "best_minus_native": value - native_value if native_value is not None else None,
                "best_direction": (
                    "down" if float(best["scale_factor"]) < 1.0 else "up" if float(best["scale_factor"]) > 1.0 else "hold"
                ),
            }
        actual_row = None
        if actual_scale is not None:
            actual_row = min(anchor_rows, key=lambda row: abs(float(row["scale_factor"]) - float(actual_scale)))
        actual_comp: dict[str, Any] | None = None
        if actual_row is not None:
            actual_comp = {}
            for metric in KEY_METRICS:
                actual_value = _finite(actual_row.get(metric))
                native_value = native_metrics.get(metric)
                actual_comp[metric] = {
                    "actual_scale_value": actual_value,
                    "native_value": native_value,
                    "actual_minus_native": (
                        actual_value - native_value
                        if actual_value is not None and native_value is not None
                        else None
                    ),
                    "beats_native": (
                        actual_value is not None and native_value is not None and actual_value < native_value
                    ),
                }
        by_anchor[anchor_mode] = {
            "native_scale_factor": float(native["scale_factor"]),
            "native_metrics": native_metrics,
            "best_by_metric": metric_best,
            "actual_candidate_scale": actual_scale,
            "actual_candidate_scale_nearest_row": float(actual_row["scale_factor"]) if actual_row is not None else None,
            "actual_vs_native": actual_comp,
        }

    primary = by_anchor.get("boundary_prev") or next(iter(by_anchor.values()))
    actual_vs_native = primary.get("actual_vs_native") or {}
    actual_beats_all = bool(actual_vs_native) and all(
        bool(actual_vs_native.get(metric, {}).get("beats_native")) for metric in KEY_METRICS
    )
    best_dirs = [
        str(primary.get("best_by_metric", {}).get(metric, {}).get("best_direction"))
        for metric in KEY_METRICS
        if metric in primary.get("best_by_metric", {})
    ]
    actual_direction = None
    if actual_scale is not None:
        actual_direction = "down" if actual_scale < 1.0 else "up" if actual_scale > 1.0 else "hold"
    return {
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "key_metrics": KEY_METRICS,
        "actual_candidate_scale": actual_scale,
        "actual_candidate_scale_direction": actual_direction,
        "by_anchor": by_anchor,
        "primary_anchor_mode": "boundary_prev" if "boundary_prev" in by_anchor else next(iter(by_anchor.keys())),
        "primary_best_directions": best_dirs,
        "actual_scale_beats_native_on_all_key_metrics": actual_beats_all,
        "scale_only_runtime_branch_recommended": False,
        "reason": (
            "Synthetic perturbation is diagnostic-only; promote only if a later real run beats native and controls. "
            "This summary only decides whether the current scale direction is worth more runtime."
        ),
    }


def main() -> None:
    args = parse_args()
    _, _, gt_pos_all = _load_kitti_gt(args.gt)
    traj = _trajectory_path(args.run_dir)
    frames, raw_poses, raw_pos = _load_tum_prediction(traj, gt_pos_all.shape[0])
    anchor_modes = list(args.anchor_mode) or ["origin", "boundary_prev", "source_tail_centroid"]
    scale_factors = _default_scales(args.actual_candidate_scale, args.scale_factor)

    rows: list[dict[str, Any]] = []
    for anchor_mode in anchor_modes:
        anchor = _anchor(
            anchor_mode,
            frames=frames,
            raw_pos=raw_pos,
            target_start=int(args.target_start_frame),
            overlap=int(args.overlap),
        )
        for factor in scale_factors:
            perturbed_poses, perturbed_pos = _perturb_post_boundary(
                raw_poses,
                frames,
                target_start=int(args.target_start_frame),
                factor=float(factor),
                anchor=anchor,
            )
            row = {
                "anchor_mode": anchor_mode,
                "scale_factor": float(factor),
                "scale_direction": "down" if factor < 1.0 else "up" if factor > 1.0 else "hold",
            }
            row.update(
                _eval_positions(
                    frames=frames,
                    raw_poses=perturbed_poses,
                    raw_pos=perturbed_pos,
                    gt_pos_all=gt_pos_all,
                    target_start=int(args.target_start_frame),
                    overlap=int(args.overlap),
                )
            )
            rows.append(row)

    payload = {
        "schema": "acl2_v80_boundary_scale_direction_diagnostic_v1",
        "gt": str(args.gt),
        "trajectory": str(traj),
        "target_start_frame": int(args.target_start_frame),
        "overlap": int(args.overlap),
        "rows": rows,
        "summary": _build_summary(rows, args.actual_candidate_scale),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.out_csv, rows)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False, sort_keys=True))
    print(f"wrote_json={args.out_json}")
    print(f"wrote_csv={args.out_csv}")


if __name__ == "__main__":
    main()
