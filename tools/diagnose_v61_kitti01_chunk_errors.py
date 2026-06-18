#!/usr/bin/env python3
"""Decompose KITTI01 v61 errors into intra-chunk and inter-chunk terms.

This diagnostic is intentionally geometry-only. It reads landed v61 artifacts
and does not use semantic masks or actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_RUN_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v61_clean_semantic_residual_read_ttt_scale_state/"
    "phase2_baseline_repair/rollouts/V61R2_A0_H35_CLEAN_REPEAT_PAIRBIAS_704F"
)


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        val = _safe_float(value)
        if math.isfinite(val):
            out.append(val)
    return out


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.mean(vals)) if vals else None


def _median(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.median(vals)) if vals else None


def _max(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.max(vals)) if vals else None


def _std(values: Iterable[Any]) -> Optional[float]:
    vals = _finite(values)
    return float(np.std(vals)) if vals else None


def _pose_centers_from_txt(path: Path) -> np.ndarray:
    rows: List[np.ndarray] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                vals = [float(x) for x in raw.split()]
            except ValueError:
                continue
            if len(vals) == 12:
                mat = np.eye(4, dtype=np.float64)
                mat[:3, :4] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
                rows.append(mat[:3, 3].copy())
            elif len(vals) >= 8:
                rows.append(np.asarray(vals[1:4], dtype=np.float64))
    if not rows:
        raise ValueError(f"No 3x4 poses found in {path}")
    return np.stack(rows, axis=0)


def _pose_centers_from_tensor(value: Any) -> np.ndarray:
    arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    arr = arr.astype(np.float64, copy=False)
    if arr.ndim != 3 or arr.shape[-2:] != (4, 4):
        raise ValueError(f"Expected camera_poses as (N,4,4), got {arr.shape}")
    return arr[:, :3, 3].copy()


def _load_chunks(geometry_dir: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for path in sorted(geometry_dir.glob("chunk_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or "camera_poses" not in data:
            continue
        centers = _pose_centers_from_tensor(data["camera_poses"])
        chunks.append(
            {
                "path": path,
                "chunk_id": int(data.get("chunk_idx", len(chunks))),
                "start": int(data.get("start_frame", 0)),
                "end": int(data.get("end_frame", int(data.get("start_frame", 0)) + centers.shape[0])),
                "centers": centers,
            }
        )
    chunks.sort(key=lambda row: int(row["chunk_id"]))
    return chunks


def _fit_sim3(src: np.ndarray, dst: np.ndarray) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    mask = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[mask].astype(np.float64, copy=False)
    dst = dst[mask].astype(np.float64, copy=False)
    if src.shape[0] < 3:
        return None
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = float(np.mean(np.sum(src_c * src_c, axis=1)))
    if var_src <= 1e-12:
        return None
    cov = (dst_c.T @ src_c) / float(src.shape[0])
    u, singular_values, vt = np.linalg.svd(cov)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rot = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular_values * sign) / var_src)
    trans = mu_dst - scale * (rot @ mu_src)
    if not math.isfinite(scale) or abs(scale) <= 1e-12:
        return None
    return scale, rot, trans


def _apply_sim3(points: np.ndarray, fit: Tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rot, trans = fit
    return scale * (points @ rot.T) + trans[None, :]


def _compose_sim3(
    second: Tuple[float, np.ndarray, np.ndarray],
    first: Tuple[float, np.ndarray, np.ndarray],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return transform equivalent to second(first(x))."""

    s1, r1, t1 = first
    s2, r2, t2 = second
    return s2 * s1, r2 @ r1, s2 * (r2 @ t1) + t2


def _residual_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    mask = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if not np.any(mask):
        return {
            "count": 0,
            "rmse": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    residual = np.linalg.norm(pred[mask] - gt[mask], axis=1)
    return {
        "count": int(residual.size),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "median": float(np.median(residual)),
        "p90": float(np.percentile(residual, 90)),
        "max": float(np.max(residual)),
    }


def _step_lengths(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def _step_ratio(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    pred_steps = _step_lengths(pred)
    gt_steps = _step_lengths(gt)
    n = min(pred_steps.size, gt_steps.size)
    if n <= 0:
        return None
    pred_steps = pred_steps[:n]
    gt_steps = gt_steps[:n]
    mask = np.isfinite(pred_steps) & np.isfinite(gt_steps) & (gt_steps > 1e-9)
    if not np.any(mask):
        return None
    return float(np.median(pred_steps[mask] / gt_steps[mask]))


def _slice_gt(gt: np.ndarray, start: int, end: int, count: int) -> np.ndarray:
    stop = min(end, start + count, gt.shape[0])
    return gt[start:stop]


def _slice_centers(row: Mapping[str, Any], start: int, end: int) -> np.ndarray:
    row_start = int(row["start"])
    centers = np.asarray(row["centers"], dtype=np.float64)
    local_start = max(0, start - row_start)
    local_end = min(centers.shape[0], end - row_start)
    if local_end <= local_start:
        return np.zeros((0, 3), dtype=np.float64)
    return centers[local_start:local_end]


def _scale_lookup(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = _read_csv(path)
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        try:
            cid = int(row.get("chunk_id", ""))
        except ValueError:
            continue
        out[cid] = row
    return out


def _top_rows(rows: Sequence[Mapping[str, Any]], key: str, n: int = 5, *, abs_value: bool = False) -> List[Dict[str, Any]]:
    def score(row: Mapping[str, Any]) -> float:
        val = _safe_float(row.get(key))
        if not math.isfinite(val):
            return -float("inf")
        return abs(val) if abs_value else val

    selected = sorted(rows, key=score, reverse=True)[:n]
    return [{k: row.get(k) for k in row.keys() if k in {"chunk_id", "prev_chunk_id", "curr_chunk_id", key, "frame_start", "frame_end", "overlap_start", "overlap_end"}} for row in selected]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--gt-poses", type=Path, default=DEFAULT_GT)
    parser.add_argument("--pred-poses", type=Path, default=None)
    parser.add_argument("--geometry-dir", type=Path, default=None)
    parser.add_argument("--scale-metrics", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir
    pred_path = args.pred_poses or run_dir / "01.txt"
    geometry_dir = args.geometry_dir or run_dir / "per_chunk_geometry"
    scale_metrics_path = args.scale_metrics or run_dir / "scale_metrics" / "per_chunk_scale_metrics.csv"
    out_dir = args.out_dir or run_dir / "chunk_error_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_global = _pose_centers_from_txt(pred_path)
    gt_global = _pose_centers_from_txt(args.gt_poses)
    n_global = min(pred_global.shape[0], gt_global.shape[0])
    pred_global = pred_global[:n_global]
    gt_for_global = gt_global[:n_global]
    global_fit = _fit_sim3(pred_global, gt_for_global)
    if global_fit is None:
        raise RuntimeError("Could not fit global Sim3 between prediction and GT")
    pred_global_aligned = _apply_sim3(pred_global, global_fit)
    global_residual = np.linalg.norm(pred_global_aligned - gt_for_global, axis=1)

    chunks = _load_chunks(geometry_dir)
    scale_rows = _scale_lookup(scale_metrics_path)
    chunk_rows: List[Dict[str, Any]] = []
    fit_to_gt_by_chunk: Dict[int, Tuple[float, np.ndarray, np.ndarray]] = {}

    for chunk in chunks:
        chunk_id = int(chunk["chunk_id"])
        start = int(chunk["start"])
        end = int(chunk["end"])
        centers = np.asarray(chunk["centers"], dtype=np.float64)
        gt = _slice_gt(gt_global, start, end, centers.shape[0])
        n = min(centers.shape[0], gt.shape[0])
        centers = centers[:n]
        gt = gt[:n]
        fit = _fit_sim3(centers, gt)
        aligned = _apply_sim3(centers, fit) if fit is not None else np.full_like(centers, np.nan)
        stats = _residual_stats(aligned, gt)
        if fit is not None:
            fit_to_gt_by_chunk[chunk_id] = fit

        global_start = min(start, n_global)
        global_end = min(start + n, n_global)
        global_stats = _residual_stats(
            pred_global_aligned[global_start:global_end],
            gt_for_global[global_start:global_end],
        )
        scale_row = scale_rows.get(chunk_id, {})
        row = {
            "chunk_id": chunk_id,
            "frame_start": start,
            "frame_end": start + n,
            "frame_count": n,
            "intra_chunk_sim3_rmse": stats["rmse"],
            "intra_chunk_sim3_median": stats["median"],
            "intra_chunk_sim3_p90": stats["p90"],
            "intra_chunk_sim3_max": stats["max"],
            "chunk_sim3_scale_to_gt": fit[0] if fit is not None else None,
            "chunk_step_ratio_to_gt": _step_ratio(centers, gt),
            "global_aligned_chunk_rmse": global_stats["rmse"],
            "global_aligned_chunk_median": global_stats["median"],
            "global_aligned_chunk_p90": global_stats["p90"],
            "global_minus_intra_rmse": (
                global_stats["rmse"] - stats["rmse"]
                if global_stats["rmse"] is not None and stats["rmse"] is not None
                else None
            ),
            "scale_metric_step_length_ratio_internal": _safe_float(scale_row.get("step_length_ratio")),
            "pointmap_depth_scale_ratio_internal": _safe_float(scale_row.get("pointmap_depth_scale_ratio")),
            "overlap_log_scale_residual_all_from_prev": _safe_float(scale_row.get("log_scale_residual_all")),
            "overlap_sim3_residual_all_from_prev": _safe_float(scale_row.get("overlap_sim3_residual_all")),
        }
        chunk_rows.append(row)

    inter_rows: List[Dict[str, Any]] = []
    for prev, curr in zip(chunks[:-1], chunks[1:]):
        prev_id = int(prev["chunk_id"])
        curr_id = int(curr["chunk_id"])
        start = max(int(prev["start"]), int(curr["start"]))
        end = min(int(prev["end"]), int(curr["end"]), gt_global.shape[0])
        prev_overlap = _slice_centers(prev, start, end)
        curr_overlap = _slice_centers(curr, start, end)
        n_overlap = min(prev_overlap.shape[0], curr_overlap.shape[0])
        prev_overlap = prev_overlap[:n_overlap]
        curr_overlap = curr_overlap[:n_overlap]
        fit_curr_to_prev = _fit_sim3(curr_overlap, prev_overlap)
        overlap_stats = (
            _residual_stats(_apply_sim3(curr_overlap, fit_curr_to_prev), prev_overlap)
            if fit_curr_to_prev is not None
            else _residual_stats(np.full_like(curr_overlap, np.nan), prev_overlap)
        )

        curr_centers = np.asarray(curr["centers"], dtype=np.float64)
        curr_gt = _slice_gt(gt_global, int(curr["start"]), int(curr["end"]), curr_centers.shape[0])
        n_curr = min(curr_centers.shape[0], curr_gt.shape[0])
        curr_centers = curr_centers[:n_curr]
        curr_gt = curr_gt[:n_curr]
        prev_to_gt = fit_to_gt_by_chunk.get(prev_id)
        curr_local_to_gt = fit_to_gt_by_chunk.get(curr_id)
        chain_stats = _residual_stats(np.full_like(curr_centers, np.nan), curr_gt)
        chain_minus_local = None
        if fit_curr_to_prev is not None and prev_to_gt is not None:
            chain_fit = _compose_sim3(prev_to_gt, fit_curr_to_prev)
            chain_stats = _residual_stats(_apply_sim3(curr_centers, chain_fit), curr_gt)
            if curr_local_to_gt is not None and chain_stats["rmse"] is not None:
                local_stats = _residual_stats(_apply_sim3(curr_centers, curr_local_to_gt), curr_gt)
                if local_stats["rmse"] is not None:
                    chain_minus_local = chain_stats["rmse"] - local_stats["rmse"]

        scale_row = scale_rows.get(curr_id, {})
        inter_rows.append(
            {
                "prev_chunk_id": prev_id,
                "curr_chunk_id": curr_id,
                "overlap_start": start,
                "overlap_end": end,
                "overlap_frame_count": n_overlap,
                "pose_overlap_sim3_scale_curr_to_prev": fit_curr_to_prev[0] if fit_curr_to_prev is not None else None,
                "pose_overlap_sim3_rmse": overlap_stats["rmse"],
                "pose_overlap_sim3_median": overlap_stats["median"],
                "pose_overlap_sim3_p90": overlap_stats["p90"],
                "chain_to_gt_rmse_after_overlap_sim3": chain_stats["rmse"],
                "chain_to_gt_p90_after_overlap_sim3": chain_stats["p90"],
                "chain_minus_local_rmse": chain_minus_local,
                "pointmap_overlap_sim3_scale_all": _safe_float(scale_row.get("overlap_sim3_scale_all")),
                "pointmap_log_scale_residual_all": _safe_float(scale_row.get("log_scale_residual_all")),
                "pointmap_overlap_sim3_residual_all_p90": _safe_float(scale_row.get("overlap_sim3_residual_all")),
                "pointmap_inlier_ratio_all": _safe_float(scale_row.get("inlier_ratio_all")),
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "pred_poses": str(pred_path),
        "gt_poses": str(args.gt_poses),
        "geometry_dir": str(geometry_dir),
        "scale_metrics": str(scale_metrics_path),
        "frame_count_pred": int(pred_global.shape[0]),
        "frame_count_gt_used": int(gt_for_global.shape[0]),
        "chunk_count": len(chunk_rows),
        "inter_chunk_pair_count": len(inter_rows),
        "global_sim3_scale_pred_to_gt": global_fit[0],
        "global_sim3_rmse": float(np.sqrt(np.mean(global_residual * global_residual))),
        "global_sim3_median": float(np.median(global_residual)),
        "global_sim3_p90": float(np.percentile(global_residual, 90)),
        "intra_chunk_sim3_rmse_mean": _mean(row.get("intra_chunk_sim3_rmse") for row in chunk_rows),
        "intra_chunk_sim3_rmse_median": _median(row.get("intra_chunk_sim3_rmse") for row in chunk_rows),
        "intra_chunk_sim3_rmse_max": _max(row.get("intra_chunk_sim3_rmse") for row in chunk_rows),
        "global_aligned_chunk_rmse_mean": _mean(row.get("global_aligned_chunk_rmse") for row in chunk_rows),
        "global_aligned_chunk_rmse_max": _max(row.get("global_aligned_chunk_rmse") for row in chunk_rows),
        "global_minus_intra_rmse_mean": _mean(row.get("global_minus_intra_rmse") for row in chunk_rows),
        "chunk_sim3_scale_to_gt_mean": _mean(row.get("chunk_sim3_scale_to_gt") for row in chunk_rows),
        "chunk_sim3_scale_to_gt_std": _std(row.get("chunk_sim3_scale_to_gt") for row in chunk_rows),
        "chunk_sim3_scale_to_gt_min": min(_finite(row.get("chunk_sim3_scale_to_gt") for row in chunk_rows), default=None),
        "chunk_sim3_scale_to_gt_max": max(_finite(row.get("chunk_sim3_scale_to_gt") for row in chunk_rows), default=None),
        "chunk_step_ratio_to_gt_mean": _mean(row.get("chunk_step_ratio_to_gt") for row in chunk_rows),
        "chunk_step_ratio_to_gt_std": _std(row.get("chunk_step_ratio_to_gt") for row in chunk_rows),
        "chunk_step_ratio_to_gt_min": min(_finite(row.get("chunk_step_ratio_to_gt") for row in chunk_rows), default=None),
        "chunk_step_ratio_to_gt_max": max(_finite(row.get("chunk_step_ratio_to_gt") for row in chunk_rows), default=None),
        "pose_overlap_sim3_rmse_mean": _mean(row.get("pose_overlap_sim3_rmse") for row in inter_rows),
        "pose_overlap_sim3_p90_max": _max(row.get("pose_overlap_sim3_p90") for row in inter_rows),
        "chain_to_gt_rmse_after_overlap_sim3_mean": _mean(row.get("chain_to_gt_rmse_after_overlap_sim3") for row in inter_rows),
        "chain_to_gt_rmse_after_overlap_sim3_max": _max(row.get("chain_to_gt_rmse_after_overlap_sim3") for row in inter_rows),
        "chain_minus_local_rmse_mean": _mean(row.get("chain_minus_local_rmse") for row in inter_rows),
        "pointmap_overlap_sim3_residual_all_p90_mean": _mean(row.get("pointmap_overlap_sim3_residual_all_p90") for row in inter_rows),
        "pointmap_overlap_sim3_residual_all_p90_max": _max(row.get("pointmap_overlap_sim3_residual_all_p90") for row in inter_rows),
        "top_intra_chunk_rmse": _top_rows(chunk_rows, "intra_chunk_sim3_rmse"),
        "top_global_aligned_chunk_rmse": _top_rows(chunk_rows, "global_aligned_chunk_rmse"),
        "top_chunk_scale_abs_log": _top_rows(
            [
                {
                    **row,
                    "chunk_scale_abs_log": abs(math.log(max(_safe_float(row.get("chunk_sim3_scale_to_gt")), 1e-12)))
                    if math.isfinite(_safe_float(row.get("chunk_sim3_scale_to_gt")))
                    else None,
                }
                for row in chunk_rows
            ],
            "chunk_scale_abs_log",
        ),
        "top_inter_chain_rmse": _top_rows(inter_rows, "chain_to_gt_rmse_after_overlap_sim3"),
        "top_inter_pose_overlap_p90": _top_rows(inter_rows, "pose_overlap_sim3_p90"),
        "top_pointmap_overlap_residual": _top_rows(inter_rows, "pointmap_overlap_sim3_residual_all_p90"),
    }

    _write_csv(out_dir / "chunk_error_rows.csv", chunk_rows)
    _write_json(out_dir / "chunk_error_rows.json", chunk_rows)
    _write_csv(out_dir / "inter_chunk_error_rows.csv", inter_rows)
    _write_json(out_dir / "inter_chunk_error_rows.json", inter_rows)
    _write_csv(out_dir / "chunk_error_summary.csv", [summary])
    _write_json(out_dir / "chunk_error_summary.json", summary)
    _write_markdown(out_dir / "chunk_error_diagnostic.md", summary, chunk_rows, inter_rows)
    print(json.dumps(_clean(summary), ensure_ascii=False, sort_keys=True))


def _fmt(value: Any, digits: int = 6) -> str:
    val = _safe_float(value)
    return f"{val:.{digits}f}" if math.isfinite(val) else "NA"


def _write_markdown(
    path: Path,
    summary: Mapping[str, Any],
    chunk_rows: Sequence[Mapping[str, Any]],
    inter_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines: List[str] = [
        "# v61 KITTI01 Chunk Error Diagnostic",
        "",
        "Diagnostic-only geometry decomposition. No semantic method result is claimed.",
        "",
        "## Summary",
        "",
        f"- Global trajectory Sim3 RMSE: `{_fmt(summary.get('global_sim3_rmse'))}` m; p90 `{_fmt(summary.get('global_sim3_p90'))}` m.",
        f"- Intra-chunk Sim3 RMSE mean/median/max: `{_fmt(summary.get('intra_chunk_sim3_rmse_mean'))}` / `{_fmt(summary.get('intra_chunk_sim3_rmse_median'))}` / `{_fmt(summary.get('intra_chunk_sim3_rmse_max'))}` m.",
        f"- Global-aligned chunk RMSE mean/max: `{_fmt(summary.get('global_aligned_chunk_rmse_mean'))}` / `{_fmt(summary.get('global_aligned_chunk_rmse_max'))}` m.",
        f"- Inter-chunk chain-to-GT RMSE after overlap Sim3 mean/max: `{_fmt(summary.get('chain_to_gt_rmse_after_overlap_sim3_mean'))}` / `{_fmt(summary.get('chain_to_gt_rmse_after_overlap_sim3_max'))}` m.",
        f"- Chunk Sim3 scale-to-GT mean/std/range: `{_fmt(summary.get('chunk_sim3_scale_to_gt_mean'))}` / `{_fmt(summary.get('chunk_sim3_scale_to_gt_std'))}` / `{_fmt(summary.get('chunk_sim3_scale_to_gt_min'))}..{_fmt(summary.get('chunk_sim3_scale_to_gt_max'))}`.",
        f"- Chunk step-ratio-to-GT mean/std/range: `{_fmt(summary.get('chunk_step_ratio_to_gt_mean'))}` / `{_fmt(summary.get('chunk_step_ratio_to_gt_std'))}` / `{_fmt(summary.get('chunk_step_ratio_to_gt_min'))}..{_fmt(summary.get('chunk_step_ratio_to_gt_max'))}`.",
        f"- Pointmap overlap residual p90 mean/max: `{_fmt(summary.get('pointmap_overlap_sim3_residual_all_p90_mean'))}` / `{_fmt(summary.get('pointmap_overlap_sim3_residual_all_p90_max'))}` m.",
        "",
        "## Top Intra-Chunk Shape Errors",
        "",
        "| chunk | frames | intra Sim3 RMSE | intra p90 | scale to GT | step ratio to GT | global chunk RMSE |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    top_chunks = sorted(chunk_rows, key=lambda row: _safe_float(row.get("intra_chunk_sim3_rmse")), reverse=True)[:8]
    for row in top_chunks:
        lines.append(
            f"| {row.get('chunk_id')} | {row.get('frame_start')}..{row.get('frame_end')} | "
            f"{_fmt(row.get('intra_chunk_sim3_rmse'))} | {_fmt(row.get('intra_chunk_sim3_p90'))} | "
            f"{_fmt(row.get('chunk_sim3_scale_to_gt'))} | {_fmt(row.get('chunk_step_ratio_to_gt'))} | "
            f"{_fmt(row.get('global_aligned_chunk_rmse'))} |"
        )
    lines.extend(
        [
            "",
            "## Top Inter-Chunk Errors After Overlap Sim3",
            "",
            "| pair | overlap | pose overlap p90 | chain-to-GT RMSE | chain-local RMSE delta | pointmap p90 | log scale residual |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    top_inter = sorted(inter_rows, key=lambda row: _safe_float(row.get("chain_to_gt_rmse_after_overlap_sim3")), reverse=True)[:8]
    for row in top_inter:
        lines.append(
            f"| {row.get('prev_chunk_id')}->{row.get('curr_chunk_id')} | {row.get('overlap_start')}..{row.get('overlap_end')} | "
            f"{_fmt(row.get('pose_overlap_sim3_p90'))} | {_fmt(row.get('chain_to_gt_rmse_after_overlap_sim3'))} | "
            f"{_fmt(row.get('chain_minus_local_rmse'))} | {_fmt(row.get('pointmap_overlap_sim3_residual_all_p90'))} | "
            f"{_fmt(row.get('pointmap_log_scale_residual_all'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `intra_chunk_sim3_*` removes each chunk's own Sim3 gauge and measures within-chunk trajectory shape error.",
            "- `chain_to_gt_*` aligns current chunk to the previous chunk using only overlap Sim3, then maps through the previous chunk's GT Sim3 gauge. It measures inter-chunk stitching error that remains after Sim3.",
            "- `chunk_sim3_scale_to_gt` and `chunk_step_ratio_to_gt` are the direct intra-chunk scale-offset diagnostics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
