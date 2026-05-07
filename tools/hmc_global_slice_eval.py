#!/usr/bin/env python3
"""Global-continuity slice diagnostics for HMC Phase C gates.

The regular KITTI benchmark locally aligns each slice, which is useful but too
forgiving for Phase C v4.  This helper fixes the Sim(3) transform from a full
no-control trajectory and applies it to stateful slice predictions so local
shape wins cannot hide endpoint drift or path-length damage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[..., 0, 1] = 2.0 * (xy - wz)
    R[..., 0, 2] = 2.0 * (xz + wy)
    R[..., 1, 0] = 2.0 * (xy + wz)
    R[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[..., 1, 2] = 2.0 * (yz - wx)
    R[..., 2, 0] = 2.0 * (xz - wy)
    R[..., 2, 1] = 2.0 * (yz + wx)
    R[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def _load_gt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None]
    poses = np.tile(np.eye(4, dtype=np.float64), (arr.shape[0], 1, 1))
    poses[:, :3, :4] = arr.reshape(-1, 3, 4)
    return np.arange(arr.shape[0], dtype=np.int64), poses[:, :3, 3].copy()


def _load_tum(path: Path, n_gt: int) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 8:
                raise ValueError(f"Bad TUM row in {path}: {line}")
            rows.append(vals[:8])
    if not rows:
        raise ValueError(f"No trajectory rows in {path}")
    arr = np.asarray(rows, dtype=np.float64)
    frames = np.rint(arr[:, 0]).astype(np.int64)
    valid = (frames >= 0) & (frames < n_gt)
    frames = frames[valid]
    arr = arr[valid]
    order = np.argsort(frames)
    frames = frames[order]
    arr = arr[order]
    return frames, arr[:, 1:4].copy()


def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError(f"Need matched Nx3 arrays with N>=3, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    mx = src.mean(axis=0)
    my = dst.mean(axis=0)
    X = src - mx
    Y = dst - my
    cov = (Y.T @ X) / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0.0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt
    scale = float(np.trace(np.diag(S) @ D) / max(float((X * X).sum() / n), 1e-12)) if with_scale else 1.0
    t = my - scale * (R @ mx)
    return scale, R, t


def _apply_pos(pos: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (R @ pos.T)).T + t[None]


def _rmse_norm(pos: np.ndarray, gt: np.ndarray) -> float:
    err = np.linalg.norm(pos - gt, axis=1)
    return float(np.sqrt(np.nanmean(err * err)))


def _path_length(pos: np.ndarray) -> float:
    if pos.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def _parse_pair(spec: str) -> Tuple[str, int, Path, Path]:
    # name,chunk,base,cand
    parts = spec.split(",", 3)
    if len(parts) != 4:
        raise ValueError("--pair must be name,chunk_idx,base_txt,cand_txt")
    name, chunk, base, cand = parts
    return name, int(chunk), Path(base), Path(cand)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--full_ref", required=True, help="Full no-control TUM trajectory used to fit global Sim(3).")
    parser.add_argument("--pair", action="append", required=True,
                        help="name,chunk_idx,base_slice_txt,candidate_slice_txt")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_summary", default=None)
    args = parser.parse_args()

    _, gt_pos = _load_gt(Path(args.gt))
    full_frames, full_pos = _load_tum(Path(args.full_ref), gt_pos.shape[0])
    scale, R, t = _umeyama(full_pos, gt_pos[full_frames], with_scale=True)

    rows: List[Dict[str, object]] = []
    for spec in args.pair:
        name, chunk_idx, base_path, cand_path = _parse_pair(spec)
        base_frames, base_raw = _load_tum(base_path, gt_pos.shape[0])
        cand_frames, cand_raw = _load_tum(cand_path, gt_pos.shape[0])
        common = np.intersect1d(base_frames, cand_frames)
        if common.shape[0] < 3:
            raise ValueError(f"Need >=3 common frames for {name}, got {common.shape[0]}")
        b_idx = np.searchsorted(base_frames, common)
        c_idx = np.searchsorted(cand_frames, common)
        base_raw = base_raw[b_idx]
        cand_raw = cand_raw[c_idx]
        gt = gt_pos[common]

        b_s, b_R, b_t = _umeyama(base_raw, gt, with_scale=True)
        c_s, c_R, c_t = _umeyama(cand_raw, gt, with_scale=True)
        base_local = _apply_pos(base_raw, b_s, b_R, b_t)
        cand_local = _apply_pos(cand_raw, c_s, c_R, c_t)
        base_global = _apply_pos(base_raw, scale, R, t)
        cand_global = _apply_pos(cand_raw, scale, R, t)

        base_local_ate = _rmse_norm(base_local, gt)
        cand_local_ate = _rmse_norm(cand_local, gt)
        base_global_ate = _rmse_norm(base_global, gt)
        cand_global_ate = _rmse_norm(cand_global, gt)
        base_end = float(np.linalg.norm(base_global[-1] - gt[-1]))
        cand_end = float(np.linalg.norm(cand_global[-1] - gt[-1]))
        path_ratio = _path_length(cand_global) / max(_path_length(base_global), 1e-12)
        rows.append({
            "name": name,
            "chunk_idx": chunk_idx,
            "start_frame": int(common[0]),
            "end_frame": int(common[-1] + 1),
            "num_frames": int(common.shape[0]),
            "base_local_sim3_ate_m": base_local_ate,
            "cand_local_sim3_ate_m": cand_local_ate,
            "delta_local_sim3_ate_m": cand_local_ate - base_local_ate,
            "base_global_fixed_ate_m": base_global_ate,
            "cand_global_fixed_ate_m": cand_global_ate,
            "delta_global_fixed_ate_m": cand_global_ate - base_global_ate,
            "base_endpoint_error_m": base_end,
            "cand_endpoint_error_m": cand_end,
            "delta_endpoint_error_m": cand_end - base_end,
            "path_length_ratio_cand_base": path_ratio,
            "local_improved": cand_local_ate <= base_local_ate,
            "global_improved": cand_global_ate <= base_global_ate,
            "endpoint_not_bad": (cand_end - base_end) <= 0.15,
            "path_ratio_pass": 0.99 <= path_ratio <= 1.01,
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_slices": len(rows),
        "global_alignment_source": str(args.full_ref),
        "global_sim3_scale": scale,
        "local_improved_count": sum(1 for r in rows if bool(r["local_improved"])),
        "global_improved_count": sum(1 for r in rows if bool(r["global_improved"])),
        "mean_delta_endpoint_error_m": float(np.mean([float(r["delta_endpoint_error_m"]) for r in rows])),
        "max_delta_endpoint_error_m": float(np.max([float(r["delta_endpoint_error_m"]) for r in rows])),
        "path_ratio_all_pass": all(bool(r["path_ratio_pass"]) for r in rows),
    }
    if args.out_summary:
        Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_summary).open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
