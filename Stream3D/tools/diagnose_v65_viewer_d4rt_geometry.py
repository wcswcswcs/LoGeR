from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def _sample_indices(count: int, max_count: int, seed: str) -> np.ndarray:
    count = int(count)
    max_count = int(max_count)
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_count <= 0 or count <= max_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(_stable_seed(seed))
    return np.sort(rng.choice(count, size=max_count, replace=False).astype(np.int64))


def _stats(dist: np.ndarray, prefix: str) -> dict[str, Any]:
    values = np.asarray(dist, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    out: dict[str, Any] = {f"{prefix}_count": int(values.size)}
    if values.size == 0:
        for key in ("mean", "median", "p75", "p90", "p95", "p99", "max"):
            out[f"{prefix}_{key}"] = None
        for threshold in (0.02, 0.05, 0.10, 0.20, 0.50, 1.00):
            out[f"{prefix}_frac_le_{str(threshold).replace('.', 'p')}m"] = None
        return out
    out.update(
        {
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_p75": float(np.percentile(values, 75)),
            f"{prefix}_p90": float(np.percentile(values, 90)),
            f"{prefix}_p95": float(np.percentile(values, 95)),
            f"{prefix}_p99": float(np.percentile(values, 99)),
            f"{prefix}_max": float(np.max(values)),
        }
    )
    for threshold in (0.02, 0.05, 0.10, 0.20, 0.50, 1.00):
        out[f"{prefix}_frac_le_{str(threshold).replace('.', 'p')}m"] = float(np.mean(values <= threshold))
    return out


def _bounds(points: np.ndarray, prefix: str) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return {f"{prefix}_finite_count": 0, f"{prefix}_min": None, f"{prefix}_max": None}
    return {
        f"{prefix}_finite_count": int(pts.shape[0]),
        f"{prefix}_min": [float(v) for v in np.min(pts, axis=0).tolist()],
        f"{prefix}_max": [float(v) for v in np.max(pts, axis=0).tolist()],
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    layers_npz = Path(args.layers_npz)
    output_json = Path(args.output_json)
    with np.load(layers_npz) as payload:
        d4rt = np.asarray(payload["d4rt_geo_points"], dtype=np.float32)
        gt = np.asarray(payload["gt_geo_points"], dtype=np.float32)
    d4rt = d4rt[np.isfinite(d4rt).all(axis=1)]
    gt = gt[np.isfinite(gt).all(axis=1)]
    d4rt_idx = _sample_indices(d4rt.shape[0], int(args.max_d4rt_points), seed=f"{layers_npz}:d4rt_to_gt")
    gt_idx = _sample_indices(gt.shape[0], int(args.max_gt_points), seed=f"{layers_npz}:gt_to_d4rt")
    d4rt_sample = d4rt[d4rt_idx]
    gt_sample = gt[gt_idx]
    gt_tree = cKDTree(gt)
    d4rt_tree = cKDTree(d4rt_sample) if d4rt_sample.shape[0] else None
    d4rt_to_gt, _ = gt_tree.query(d4rt_sample, k=1) if d4rt_sample.shape[0] else (np.asarray([]), np.asarray([]))
    gt_to_d4rt = np.asarray([], dtype=np.float64)
    if d4rt_tree is not None and gt_sample.shape[0]:
        gt_to_d4rt, _ = d4rt_tree.query(gt_sample, k=1)
    summary = {
        "phase": "v65_viewer_d4rt_geometry_diagnostic",
        "layers_npz": str(layers_npz),
        "layers_npz_sha256": _sha256(layers_npz),
        "d4rt_point_count": int(d4rt.shape[0]),
        "gt_point_count": int(gt.shape[0]),
        "sampled_d4rt_point_count": int(d4rt_sample.shape[0]),
        "sampled_gt_point_count": int(gt_sample.shape[0]),
        **_bounds(d4rt, "d4rt"),
        **_bounds(gt, "gt"),
        **_stats(d4rt_to_gt, "viewer_d4rt_to_gt_mesh_nn"),
        **_stats(gt_to_d4rt, "viewer_gt_mesh_to_d4rt_nn"),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute D4RT-vs-GT geometry metrics directly from a v65 Viser layer NPZ.")
    parser.add_argument("--layers-npz", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-d4rt-points", type=int, default=500000)
    parser.add_argument("--max-gt-points", type=int, default=250000)
    return parser.parse_args()


if __name__ == "__main__":
    diagnose(parse_args())
