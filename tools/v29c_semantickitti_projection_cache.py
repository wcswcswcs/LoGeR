#!/usr/bin/env python3
"""Generate v29C SemanticKITTI sparse 3D-to-2D projection cache."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.gt_semantic_provider import (  # noqa: E402
    IGNORE_LABEL,
    SEMANTIC_KITTI_ID_TO_NAME,
    read_kitti_calib,
)


CHUNK_STARTS = {6: 174, 10: 290, 16: 464}


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
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


def _parse_int_list(text: str, default: Sequence[int]) -> List[int]:
    if not str(text or "").strip():
        return list(default)
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _selected_frames(chunks: Sequence[int], horizons: Sequence[int]) -> List[int]:
    frames = set()
    for chunk in chunks:
        start = CHUNK_STARTS[int(chunk)]
        for horizon in horizons:
            end = start + 32 + int(horizon) * 29
            frames.update(range(start, end))
    return sorted(frames)


def _homogeneous_4x4(mat: np.ndarray) -> np.ndarray:
    if mat.shape == (4, 4):
        return mat.astype(np.float64)
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = mat.astype(np.float64)
        return out
    raise ValueError(f"Unsupported transform shape: {mat.shape}")


def _projection_matrices(calib_path: Path) -> Tuple[np.ndarray, np.ndarray, str, bool]:
    calib = read_kitti_calib(calib_path)
    if "P2" not in calib:
        raise ValueError(f"Missing P2 in {calib_path}")
    tr_key = ""
    tr = None
    for key in ("Tr", "Tr_velo_to_cam", "Tr_velo_cam"):
        if key in calib:
            tr = calib[key]
            tr_key = key
            break
    if tr is None:
        raise ValueError(f"Missing Tr/Tr_velo_to_cam in {calib_path}")
    r0 = calib.get("R0_rect")
    r0_missing = r0 is None
    if r0 is None:
        r0 = np.eye(3, dtype=np.float64)
    r0_4 = np.eye(4, dtype=np.float64)
    r0_4[:3, :3] = r0.astype(np.float64)
    return calib["P2"].astype(np.float64), r0_4 @ _homogeneous_4x4(tr), tr_key, r0_missing


def _image_size(image_dir: Path, frame: int) -> Tuple[int, int]:
    from PIL import Image  # type: ignore

    path = image_dir / f"{frame:06d}.png"
    with Image.open(path) as img:
        return int(img.size[0]), int(img.size[1])


def _project_frame(seq_dir: Path, out_dir: Path, frame: int, p2: np.ndarray, velo_to_rect: np.ndarray) -> Dict[str, object]:
    image_dir = seq_dir / "image_2"
    width, height = _image_size(image_dir, frame)
    points_path = seq_dir / "velodyne" / f"{frame:06d}.bin"
    labels_path = seq_dir / "labels" / f"{frame:06d}.label"
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile(labels_path, dtype=np.uint32)
    if points.shape[0] != labels.shape[0]:
        raise ValueError(f"Point/label mismatch frame {frame}: {points.shape[0]} vs {labels.shape[0]}")

    semantic_ids = (labels & np.uint32(0xFFFF)).astype(np.int32)
    instance_ids = (labels >> np.uint32(16)).astype(np.int32)
    pts_h = np.ones((points.shape[0], 4), dtype=np.float64)
    pts_h[:, :3] = points[:, :3].astype(np.float64)
    rect = (velo_to_rect @ pts_h.T).T
    in_front = rect[:, 2] > 1e-6
    proj = (p2 @ rect.T).T
    uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    valid = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)

    sem = np.full((height, width), IGNORE_LABEL, dtype=np.uint16)
    inst = np.zeros((height, width), dtype=np.uint16)
    depth = np.zeros((height, width), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=np.uint8)
    if np.any(valid):
        flat = v[valid] * width + u[valid]
        z = rect[valid, 2].astype(np.float64)
        sem_v = semantic_ids[valid]
        inst_v = instance_ids[valid]
        order = np.lexsort((z, flat))
        flat_sorted = flat[order]
        first = np.r_[True, flat_sorted[1:] != flat_sorted[:-1]]
        chosen = order[first]
        chosen_flat = flat[chosen]
        yy = chosen_flat // width
        xx = chosen_flat % width
        sem[yy, xx] = sem_v[chosen].astype(np.uint16)
        inst[yy, xx] = np.clip(inst_v[chosen], 0, np.iinfo(np.uint16).max).astype(np.uint16)
        depth[yy, xx] = z[chosen].astype(np.float32)
        valid_mask[yy, xx] = 1

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{frame:06d}"
    np.save(str(prefix) + "_sem_sparse.npy", sem)
    np.save(str(prefix) + "_inst_sparse.npy", inst)
    np.save(str(prefix) + "_depth_sparse.npy", depth)
    np.save(str(prefix) + "_valid_mask.npy", valid_mask)

    valid_count = int(valid_mask.sum())
    valid_sem = sem[valid_mask.astype(bool)].astype(np.int32)
    unique, counts = np.unique(valid_sem, return_counts=True) if valid_count else (np.array([], dtype=np.int32), np.array([], dtype=np.int64))
    label_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    label_names = {str(int(k)): SEMANTIC_KITTI_ID_TO_NAME.get(int(k), f"id{int(k)}") for k in unique}
    meta = {
        "frame": int(frame),
        "width": int(width),
        "height": int(height),
        "num_points": int(points.shape[0]),
        "num_projected_points_before_zbuffer": int(valid.sum()),
        "num_unique_projected_pixels": valid_count,
        "projected_pixel_coverage": float(valid_count / max(1, height * width)),
        "semantic_label_counts": label_counts,
        "semantic_label_names": label_names,
        "source_points": str(points_path),
        "source_labels": str(labels_path),
        "ignore_id": int(IGNORE_LABEL),
    }
    (prefix.with_name(prefix.name + "_meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    depths = depth[valid_mask.astype(bool)]
    return {
        "frame": int(frame),
        "num_points": int(points.shape[0]),
        "num_projected_points_before_zbuffer": int(valid.sum()),
        "num_unique_projected_pixels": valid_count,
        "projected_pixel_coverage": float(valid_count / max(1, height * width)),
        "depth_mean": float(depths.mean()) if depths.size else 0.0,
        "depth_p50": float(np.quantile(depths, 0.50)) if depths.size else 0.0,
        "depth_p90": float(np.quantile(depths, 0.90)) if depths.size else 0.0,
        "label_count": int(len(unique)),
        "top_labels": " ".join(f"{int(k)}:{SEMANTIC_KITTI_ID_TO_NAME.get(int(k), f'id{int(k)}')}:{int(v)}" for k, v in zip(unique[:12], counts[:12])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semkitti-root", default="/mnt/data/users/chengshun.wang/data/semantickitti_odometry")
    parser.add_argument("--sequence", default="01")
    parser.add_argument("--results-root", default="results/kitti01_hmc_v2/acl2_v29c_semantickitti_download_projection_videomasklet")
    parser.add_argument("--chunks", default="6,10,16")
    parser.add_argument("--horizons", default="10,15")
    args = parser.parse_args()

    repo = REPO_ROOT
    results = Path(args.results_root)
    if not results.is_absolute():
        results = repo / results
    seq_dir = Path(args.semkitti_root) / "dataset" / "sequences" / str(args.sequence)
    cache_dir = results / "projection_cache" / f"seq{int(args.sequence):02d}"
    report_dir = results / "phase1_projection_cache"
    chunks = _parse_int_list(args.chunks, [6, 10, 16])
    horizons = _parse_int_list(args.horizons, [10, 15])
    frames = _selected_frames(chunks, horizons)
    p2, velo_to_rect, tr_key, r0_missing = _projection_matrices(seq_dir / "calib.txt")

    rows: List[Dict[str, object]] = []
    for idx, frame in enumerate(frames, start=1):
        rows.append(_project_frame(seq_dir, cache_dir, frame, p2, velo_to_rect))
        if idx % 50 == 0 or idx == len(frames):
            print(json.dumps({"projected_frames": idx, "total_frames": len(frames), "last_frame": frame}), flush=True)

    _write_csv(report_dir / "projection_frame_metrics.csv", rows)
    coverages = np.array([float(r["projected_pixel_coverage"]) for r in rows], dtype=np.float64)
    unique_pixels = np.array([int(r["num_unique_projected_pixels"]) for r in rows], dtype=np.int64)
    focus_rows = [r for r in rows if 200 <= int(r["frame"]) < 300]
    focus_cov = float(np.mean([float(r["projected_pixel_coverage"]) for r in focus_rows])) if focus_rows else 0.0
    summary = {
        "phase": "v29c_phase1_semantickitti_sparse_projection_cache",
        "sequence": str(args.sequence),
        "frames_expected": len(frames),
        "frames_projected": len(rows),
        "frame_hit_rate": len(rows) / max(1, len(frames)),
        "median_unique_projected_pixels": float(np.median(unique_pixels)) if unique_pixels.size else 0.0,
        "mean_unique_projected_pixels": float(np.mean(unique_pixels)) if unique_pixels.size else 0.0,
        "mean_projected_pixel_coverage": float(np.mean(coverages)) if coverages.size else 0.0,
        "p10_projected_pixel_coverage": float(np.quantile(coverages, 0.10)) if coverages.size else 0.0,
        "focus_200_300_projected_coverage": focus_cov,
        "projection_cache_gate_pass": bool(
            len(rows) / max(1, len(frames)) >= 0.95
            and (float(np.median(unique_pixels)) if unique_pixels.size else 0.0) >= 2000.0
            and focus_cov > 0.0
        ),
        "cache_dir": str(cache_dir),
        "calib_tr_key": tr_key,
        "r0_rect_missing_identity_used": bool(r0_missing),
        "ignore_id": int(IGNORE_LABEL),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "projection_cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if bool(summary["projection_cache_gate_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
