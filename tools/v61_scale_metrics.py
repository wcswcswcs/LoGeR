#!/usr/bin/env python3
"""Compute ACL2 v61 per-chunk scale diagnostics from landed artifacts.

The overlap Sim(3) metrics are computed only when per-chunk geometry debug
files are present. Missing semantic point weights are reported as unavailable;
they are never replaced with zeros.
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


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        val = _safe_float(value)
        if math.isfinite(val):
            out.append(val)
    return out


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _load_chunk(path: Path) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a dict")
    return data


def _as_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    arr = np.asarray(value)
    return arr


def _chunk_rows(geometry_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(geometry_dir.glob("chunk_*.pt")):
        data = _load_chunk(path)
        points = _as_numpy(data.get("points"))
        local = _as_numpy(data.get("local_points"))
        poses = _as_numpy(data.get("camera_poses"))
        conf = _as_numpy(data.get("conf"))
        if points is None or poses is None:
            continue
        rows.append({
            "path": path,
            "chunk_id": int(data.get("chunk_idx", len(rows))),
            "start": int(data.get("start_frame", 0)),
            "end": int(data.get("end_frame", 0)),
            "points": points,
            "local_points": local,
            "camera_poses": poses,
            "conf": conf,
            "semantic_weight": _as_numpy(data.get("semantic_weight")),
        })
    rows.sort(key=lambda r: int(r["chunk_id"]))
    return rows


def _weighted_sim3(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    mask = (
        np.isfinite(src).all(axis=1)
        & np.isfinite(dst).all(axis=1)
        & np.isfinite(weights)
        & (weights > 0)
    )
    src = src[mask].astype(np.float64, copy=False)
    dst = dst[mask].astype(np.float64, copy=False)
    weights = weights[mask].astype(np.float64, copy=False)
    if src.shape[0] < 6 or float(weights.sum()) <= 1e-12:
        return None
    weights = weights / max(float(weights.sum()), 1e-12)
    mu_src = (weights[:, None] * src).sum(axis=0)
    mu_dst = (weights[:, None] * dst).sum(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (weights[:, None] * dst_c).T @ src_c
    u, _, vt = np.linalg.svd(cov)
    sfix = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sfix[-1, -1] = -1.0
    rot = u @ sfix @ vt
    var_src = float((weights * np.sum(src_c * src_c, axis=1)).sum())
    if var_src <= 1e-12:
        return None
    scale = float(np.trace(np.diag(sfix.diagonal()) @ np.diag(np.linalg.svd(cov, compute_uv=False))) / var_src)
    # The trace expression above is intentionally conservative, but for
    # numerical stability recompute it from the rotated covariance.
    scale = float(np.sum(weights * np.sum((dst_c @ rot) * src_c, axis=1)) / var_src)
    trans = mu_dst - scale * (rot @ mu_src)
    if not math.isfinite(scale) or abs(scale) <= 1e-12:
        return None
    return scale, rot, trans


def _sim3_metrics(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> Dict[str, Any]:
    fit = _weighted_sim3(src, dst, weights)
    if fit is None:
        return {"available": False}
    scale, rot, trans = fit
    pred = scale * (src @ rot.T) + trans[None, :]
    residual = np.linalg.norm(pred - dst, axis=1)
    valid = np.isfinite(residual) & np.isfinite(weights) & (weights > 0)
    residual = residual[valid]
    if residual.size == 0:
        return {"available": False}
    med = float(np.median(residual))
    p90 = float(np.percentile(residual, 90))
    inlier_thr = max(0.05, 2.5 * med)
    return {
        "available": True,
        "scale": float(scale),
        "log_scale_residual": float(math.log(max(abs(scale), 1e-12))),
        "residual_median": med,
        "residual_p90": p90,
        "inlier_ratio": float(np.mean(residual <= inlier_thr)),
        "inlier_threshold": float(inlier_thr),
    }


def _sample_overlap(
    prev: Mapping[str, Any],
    curr: Mapping[str, Any],
    *,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    start = max(int(prev["start"]), int(curr["start"]))
    end = min(int(prev["end"]), int(curr["end"]))
    if end <= start:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            False,
        )
    prev_pts = prev["points"]
    curr_pts = curr["points"]
    prev_conf = prev.get("conf")
    curr_conf = curr.get("conf")
    prev_sem = prev.get("semantic_weight")
    curr_sem = curr.get("semantic_weight")
    src_chunks: List[np.ndarray] = []
    dst_chunks: List[np.ndarray] = []
    geo_w_chunks: List[np.ndarray] = []
    sem_w_chunks: List[np.ndarray] = []
    sem_available = prev_sem is not None and curr_sem is not None
    for frame in range(start, end):
        pi = frame - int(prev["start"])
        ci = frame - int(curr["start"])
        if pi < 0 or ci < 0 or pi >= prev_pts.shape[0] or ci >= curr_pts.shape[0]:
            continue
        dst = prev_pts[pi].reshape(-1, 3)
        src = curr_pts[ci].reshape(-1, 3)
        n = min(dst.shape[0], src.shape[0])
        if n <= 0:
            continue
        dst = dst[:n]
        src = src[:n]
        if prev_conf is not None and curr_conf is not None:
            w0 = prev_conf[pi].reshape(-1)[:n]
            w1 = curr_conf[ci].reshape(-1)[:n]
            geo_w = np.sqrt(np.clip(w0, 0.0, None) * np.clip(w1, 0.0, None))
        else:
            geo_w = np.ones((n,), dtype=np.float64)
        if sem_available:
            sw0 = prev_sem[pi].reshape(-1)[:n]
            sw1 = curr_sem[ci].reshape(-1)[:n]
            sem_w = geo_w * np.sqrt(np.clip(sw0, 0.0, None) * np.clip(sw1, 0.0, None))
        else:
            sem_w = np.zeros((n,), dtype=np.float64)
        src_chunks.append(src)
        dst_chunks.append(dst)
        geo_w_chunks.append(geo_w)
        sem_w_chunks.append(sem_w)
    if not src_chunks:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            sem_available,
        )
    src_all = np.concatenate(src_chunks, axis=0)
    dst_all = np.concatenate(dst_chunks, axis=0)
    geo_w_all = np.concatenate(geo_w_chunks, axis=0)
    sem_w_all = np.concatenate(sem_w_chunks, axis=0)
    finite = np.isfinite(src_all).all(axis=1) & np.isfinite(dst_all).all(axis=1) & np.isfinite(geo_w_all)
    src_all = src_all[finite]
    dst_all = dst_all[finite]
    geo_w_all = geo_w_all[finite]
    sem_w_all = sem_w_all[finite]
    if max_points > 0 and src_all.shape[0] > max_points:
        idx = np.linspace(0, src_all.shape[0] - 1, max_points).astype(np.int64)
        src_all = src_all[idx]
        dst_all = dst_all[idx]
        geo_w_all = geo_w_all[idx]
        sem_w_all = sem_w_all[idx]
    return src_all, dst_all, geo_w_all, sem_w_all, sem_available


def _pose_step_stats(chunks: Sequence[Mapping[str, Any]]) -> Tuple[Dict[int, float], float]:
    per: Dict[int, float] = {}
    all_steps: List[float] = []
    for row in chunks:
        poses = np.asarray(row.get("camera_poses"), dtype=np.float64)
        if poses.ndim < 3 or poses.shape[0] < 2:
            continue
        pos = poses[:, :3, 3]
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        vals = [float(x) for x in steps if math.isfinite(float(x))]
        if vals:
            per[int(row["chunk_id"])] = float(np.median(vals))
            all_steps.extend(vals)
    global_med = float(np.median(all_steps)) if all_steps else float("nan")
    return per, global_med


def _point_depth_median(row: Mapping[str, Any]) -> float:
    local = row.get("local_points")
    if local is None:
        return float("nan")
    arr = np.asarray(local, dtype=np.float64)
    if arr.ndim < 4:
        return float("nan")
    z = arr[..., 2].reshape(-1)
    z = z[np.isfinite(z)]
    return float(np.median(np.abs(z))) if z.size else float("nan")


def _hmc_by_chunk(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in _load_jsonl(path):
        try:
            chunk = int(row.get("chunk_idx"))
        except (TypeError, ValueError):
            continue
        out[chunk] = row
    return out


def _prefixed(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    if not metrics.get("available"):
        return {
            f"overlap_sim3_scale_{prefix}": None,
            f"log_scale_residual_{prefix}": None,
            f"overlap_sim3_residual_{prefix}": None,
            f"inlier_ratio_{prefix}": None,
        }
    return {
        f"overlap_sim3_scale_{prefix}": metrics.get("scale"),
        f"log_scale_residual_{prefix}": metrics.get("log_scale_residual"),
        f"overlap_sim3_residual_{prefix}": metrics.get("residual_p90"),
        f"inlier_ratio_{prefix}": metrics.get("inlier_ratio"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--geometry-dir", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-points", type=int, default=24000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    geometry_dir = Path(args.geometry_dir) if args.geometry_dir else run_dir / "per_chunk_geometry"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hmc_rows = _hmc_by_chunk(run_dir / "hmc_state_hash.jsonl")
    chunks = _chunk_rows(geometry_dir) if geometry_dir.is_dir() else []
    step_by_chunk, global_step_median = _pose_step_stats(chunks)
    depth_vals = [_point_depth_median(row) for row in chunks]
    depth_global = float(np.nanmedian(depth_vals)) if any(math.isfinite(v) for v in depth_vals) else float("nan")

    rows: List[Dict[str, Any]] = []
    for idx, curr in enumerate(chunks):
        chunk_id = int(curr["chunk_id"])
        hmc = hmc_rows.get(chunk_id, {})
        row: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "frame_start": int(curr["start"]),
            "frame_end": int(curr["end"]),
            "step_length_median": step_by_chunk.get(chunk_id),
            "step_length_ratio": (
                step_by_chunk.get(chunk_id) / global_step_median
                if math.isfinite(global_step_median) and global_step_median > 0 and step_by_chunk.get(chunk_id) is not None
                else None
            ),
            "pointmap_depth_scale_ratio": (
                _point_depth_median(curr) / depth_global
                if math.isfinite(depth_global) and depth_global > 0 and math.isfinite(_point_depth_median(curr))
                else None
            ),
            "semantic_anchor_ratio": hmc.get("prior_semantic_anchor_token_ratio"),
            "semantic_anchor_spatial_entropy": hmc.get("prior_semantic_anchor_spatial_entropy"),
            "semantic_point_weights_available": bool(curr.get("semantic_weight") is not None),
        }
        if idx == 0:
            row.update(_prefixed("all", {"available": False}))
            row.update(_prefixed("geo", {"available": False}))
            row.update(_prefixed("sem", {"available": False}))
            rows.append(row)
            continue
        prev = chunks[idx - 1]
        src, dst, geo_w, sem_w, sem_available = _sample_overlap(prev, curr, max_points=int(args.max_points))
        all_w = np.ones_like(geo_w)
        all_metrics = _sim3_metrics(src, dst, all_w)
        geo_metrics = _sim3_metrics(src, dst, geo_w)
        sem_metrics = _sim3_metrics(src, dst, sem_w) if sem_available and float(sem_w.sum()) > 0 else {"available": False}
        row["overlap_frame_count"] = max(0, min(int(prev["end"]), int(curr["end"])) - max(int(prev["start"]), int(curr["start"])))
        row["overlap_point_pairs"] = int(src.shape[0])
        row["semantic_point_weights_available"] = bool(sem_available)
        row.update(_prefixed("all", all_metrics))
        row.update(_prefixed("geo", geo_metrics))
        row.update(_prefixed("sem", sem_metrics))
        rows.append(row)

    def _variance(key: str) -> Optional[float]:
        vals = _finite_values(row.get(key) for row in rows)
        return float(np.var(vals)) if vals else None

    def _mean_abs(key: str) -> Optional[float]:
        vals = _finite_values(row.get(key) for row in rows)
        return float(np.mean(np.abs(vals))) if vals else None

    summary = {
        "run_dir": str(run_dir),
        "geometry_dir": str(geometry_dir),
        "chunk_count": len(chunks),
        "scale_rows": len(rows),
        "overlap_point_pairs_available": any(int(row.get("overlap_point_pairs") or 0) > 0 for row in rows),
        "semantic_point_weights_available": any(bool(row.get("semantic_point_weights_available")) for row in rows),
        "variance_all": _variance("log_scale_residual_all"),
        "variance_geo": _variance("log_scale_residual_geo"),
        "variance_sem": _variance("log_scale_residual_sem"),
        "mean_abs_log_scale_all": _mean_abs("log_scale_residual_all"),
        "mean_abs_log_scale_geo": _mean_abs("log_scale_residual_geo"),
        "mean_abs_log_scale_sem": _mean_abs("log_scale_residual_sem"),
        "corr_log_scale_with_rolling100": None,
        "corr_anchor_quality_with_scale_stability": None,
        "note": "sem metrics are NA unless explicit point-level semantic weights are saved",
    }
    _write_csv(out_dir / "per_chunk_scale_metrics.csv", rows)
    _write_json(out_dir / "per_chunk_scale_metrics.json", rows)
    _write_csv(out_dir / "scale_residual_summary.csv", [summary])
    _write_json(out_dir / "scale_residual_summary.json", summary)
    print(json.dumps(_clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
