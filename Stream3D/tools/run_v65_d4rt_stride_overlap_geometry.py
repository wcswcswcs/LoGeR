#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry_provider.common import backproject_xy_world
from stream4d.d4rt_adapter import D4RTAdapter
from stream4d.scannet_stream import ScanNetStream
from stream4d_native.self_stitch import fit_sim3_with_diagnostics, match_overlap_carriers, residual_diagnostics
from stream4d_native.sim3 import Sim3Transform, apply_sim3_to_xyz, compose_sim3, fit_sim3_umeyama
from stream4d_native.v65_common import rel, sha256_file
from stream4d_native.v65_visualization_export import _id_colors, _load_gt, _load_scene_mesh, _window_colors
from tools.export_d4rt_grid_surfel_field_v8 import _grid_sources


DEFAULT_STRIDES = (1, 2, 5, 10)
DEFAULT_OUTPUT_ROOT = "outputs/audit/v65_d4rt_stride_overlap_geometry_scene0050"


@dataclass
class ChunkRecord:
    chunk_index: int
    frame_ids: list[int]
    xyz: np.ndarray
    uv: np.ndarray
    valid: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray
    carrier_id: np.ndarray
    src_frame_global: np.ndarray
    src_xy: np.ndarray
    transform_to_scene: Sim3Transform

    def raw_dict(self) -> dict[str, np.ndarray | list[int]]:
        return {
            "frame_ids": self.frame_ids,
            "xyz": self.xyz,
            "uv": self.uv,
            "valid": self.valid,
            "visibility": self.visibility,
            "confidence": self.confidence,
            "carrier_id": self.carrier_id,
            "src_frame_global": self.src_frame_global,
            "src_xy": self.src_xy,
        }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_only:
        summarize_outputs(args)
        return
    if args.serve:
        serve_viewer(args)
        return

    out_root = project(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    command_path = out_root / "last_command.txt"
    command_path.write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    adapter = D4RTAdapter(
        d4rt_root=resolve_repo(args.d4rt_root),
        model_config=resolve_repo(args.d4rt_config),
        ckpt_path=resolve_repo(args.d4rt_ckpt),
        device=args.device,
    )
    all_rows: list[dict[str, Any]] = []
    for stride in args.strides:
        row = run_stride(args, adapter, int(stride))
        all_rows.append(row)
    summarize_outputs(args, extra_rows=all_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh D4RT geometry recompute with stride-specific 32-frame chunks, "
            "3 selected-frame overlap stitching, final diagnostic Sim3 to ScanNet GT, metrics, and Viser view."
        )
    )
    parser.add_argument("--scene", default="scene0050_00")
    parser.add_argument("--strides", nargs="+", type=int, default=list(DEFAULT_STRIDES))
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--d4rt-root", default="Open-d4rt")
    parser.add_argument("--d4rt-config", default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--d4rt-ckpt", default="Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scannet-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--overlap-frames", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--grid-margin-ratio", type=float, default=0.02)
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--uv-radius", type=float, default=0.002)
    parser.add_argument("--max-matches-per-frame", type=int, default=4096)
    parser.add_argument("--fit-trim-percentile", type=float, default=90.0)
    parser.add_argument("--max-sim3-anchors", type=int, default=120000)
    parser.add_argument("--max-metric-points", type=int, default=250000)
    parser.add_argument("--max-gt-metric-points", type=int, default=250000)
    parser.add_argument("--max-visual-points-per-stride", type=int, default=180000)
    parser.add_argument("--save-chunks", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    return parser


def project(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "Stream3D":
        return REPO_ROOT / path
    return ROOT / path


def resolve_repo(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def run_stride(args: argparse.Namespace, adapter: D4RTAdapter, stride: int) -> dict[str, Any]:
    started = time.time()
    out = project(args.output_root) / f"stride_{stride}"
    chunks_dir = out / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    stream = ScanNetStream(seq_name=args.scene, root=resolve_repo(args.scannet_root))
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))
    frame_ids = stream.frame_ids(stride=stride, max_frames=int(args.max_frames) if args.max_frames > 0 else None)
    windows = make_overlap_windows(frame_ids, int(args.chunk_size), int(args.overlap_frames))
    if not windows:
        raise RuntimeError(f"No windows for scene={args.scene} stride={stride}")

    print(
        f"[v65-stride-geom] scene={args.scene} stride={stride} frames={len(frame_ids)} "
        f"chunks={len(windows)} chunk_size={args.chunk_size} overlap={args.overlap_frames} grid={args.grid_size}",
        flush=True,
    )

    chunk_rows: list[dict[str, Any]] = []
    stitch_rows: list[dict[str, Any]] = []
    chunks: list[ChunkRecord] = []
    previous: ChunkRecord | None = None
    for chunk_index, frame_window in enumerate(windows):
        chunk_t0 = time.time()
        data = load_window_without_masks(stream, frame_window)
        sources, source_diag = _grid_sources(
            masks=np.asarray(data["mask"]),
            frame_ids=frame_window,
            grid_size=int(args.grid_size),
            grid_margin_ratio=float(args.grid_margin_ratio),
            mask_aware_min_points_per_mask=0,
            min_mask_area=1,
        )
        batch = adapter.infer_carriers(
            video_rgb_uint8=np.asarray(data["rgb"]),
            src_uv_norm=sources.src_uv,
            src_frame_local=sources.src_frame,
            carrier_id=sources.carrier_id,
            src_frame_global=sources.src_frame_global,
            src_xy=sources.src_xy,
            src_mask_id=sources.src_mask_id,
            query_chunk_size=int(args.query_chunk_size),
        )
        raw = ChunkRecord(
            chunk_index=chunk_index,
            frame_ids=[int(v) for v in frame_window],
            xyz=np.asarray(batch.xyz_ref, dtype=np.float32),
            uv=np.asarray(batch.uv_pred, dtype=np.float32),
            valid=np.asarray(batch.valid, dtype=bool),
            visibility=np.asarray(batch.visibility_prob, dtype=np.float32),
            confidence=np.asarray(batch.confidence_prob, dtype=np.float32),
            carrier_id=np.asarray(batch.carrier_id, dtype=np.int64),
            src_frame_global=np.asarray(batch.src_frame_global, dtype=np.int64),
            src_xy=np.asarray(batch.src_xy, dtype=np.int64),
            transform_to_scene=Sim3Transform(scale=1.0, rot=np.eye(3, dtype=np.float64), trans=np.zeros(3, dtype=np.float64)),
        )
        if previous is not None:
            stitch = fit_overlap_transform(previous, raw, args)
            raw.transform_to_scene = compose_sim3(stitch["transform_curr_to_prev"], previous.transform_to_scene)
            stitch_rows.append(
                {
                    "scene": args.scene,
                    "stride": stride,
                    "prev_chunk_index": int(previous.chunk_index),
                    "curr_chunk_index": int(chunk_index),
                    "prev_frames": frame_range_text(previous.frame_ids),
                    "curr_frames": frame_range_text(raw.frame_ids),
                    **stitch["row"],
                }
            )
        chunks.append(raw)
        previous = raw
        chunk_row = {
            "scene": args.scene,
            "stride": int(stride),
            "chunk_index": int(chunk_index),
            "frame_ids": ",".join(str(int(v)) for v in frame_window),
            "frame_start": int(frame_window[0]),
            "frame_end": int(frame_window[-1]),
            "num_frames": int(len(frame_window)),
            "source_count": int(sources.carrier_id.shape[0]),
            "valid_observation_count": int(np.count_nonzero(raw_valid_mask(raw, args))),
            "seconds": float(time.time() - chunk_t0),
            **source_diag,
            **adapter.last_infer_diagnostics,
            "transform_scale_to_scene_before_final_gt": float(raw.transform_to_scene.scale),
            "transform_trans_norm_to_scene_before_final_gt": float(np.linalg.norm(raw.transform_to_scene.trans)),
        }
        chunk_rows.append(chunk_row)
        if int(args.save_chunks):
            save_chunk_npz(chunks_dir / f"chunk_{chunk_index:04d}.npz", raw)
        write_json(chunks_dir / f"chunk_{chunk_index:04d}_summary.json", chunk_row)
        print(
            f"[v65-stride-geom] stride={stride} chunk={chunk_index + 1}/{len(windows)} "
            f"frames={frame_window[0]}..{frame_window[-1]} valid={chunk_row['valid_observation_count']} "
            f"sec={chunk_row['seconds']:.2f}",
            flush=True,
        )

    stitched = collect_stitched_observations(chunks, args)
    final_fit = fit_final_gt_sim3(stitched, stream, args)
    final_points = apply_sim3_to_xyz(stitched["xyz"], transform=final_fit["transform"])
    metrics = compute_metrics(args.scene, final_points, args)
    visual = save_visual_points(out, final_points, stitched, stride, args)

    summary = {
        "phase": "v65_fresh_d4rt_stride_overlap_geometry",
        "scene": args.scene,
        "stride": int(stride),
        "fresh_recompute": True,
        "cache_policy": "does_not_read_old_d4rt_debug_cache; writes this run outputs only",
        "d4rt_root": str(resolve_repo(args.d4rt_root)),
        "d4rt_config": str(resolve_repo(args.d4rt_config)),
        "d4rt_config_sha256": sha256_file(resolve_repo(args.d4rt_config)),
        "d4rt_ckpt": str(resolve_repo(args.d4rt_ckpt)),
        "d4rt_ckpt_size_bytes": int(resolve_repo(args.d4rt_ckpt).stat().st_size),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "frame_count": int(len(frame_ids)),
        "chunk_size": int(args.chunk_size),
        "overlap_frames": int(args.overlap_frames),
        "chunk_step_selected_frames": int(args.chunk_size - args.overlap_frames),
        "chunk_count": int(len(chunks)),
        "grid_size": int(args.grid_size),
        "grid_points_per_frame": int(args.grid_size * args.grid_size),
        "min_visibility": float(args.min_visibility),
        "min_confidence": float(args.min_confidence),
        "uv_radius": float(args.uv_radius),
        "raw_stitched_observation_count": int(stitched["xyz"].shape[0]),
        "final_gt_sim3": final_fit["summary"],
        "final_gt_sim3_transform": {
            "scale": float(final_fit["transform"].scale),
            "rot": np.asarray(final_fit["transform"].rot, dtype=np.float64),
            "trans": np.asarray(final_fit["transform"].trans, dtype=np.float64),
        },
        "metrics": metrics,
        "visualization": visual,
        "seconds": float(time.time() - started),
        "outputs": {
            "stride_summary_json": rel(out / "stride_summary.json"),
            "chunk_rows_csv": rel(out / "chunk_rows.csv"),
            "stitch_rows_csv": rel(out / "stitch_rows.csv"),
            "metric_rows_csv": rel(out / "metric_rows.csv"),
            "visual_points_npz": rel(out / "visual_points.npz"),
        },
    }
    write_json(out / "stride_summary.json", summary)
    write_csv(out / "chunk_rows.csv", chunk_rows)
    write_csv(out / "stitch_rows.csv", stitch_rows)
    write_csv(out / "metric_rows.csv", flatten_metric_rows(args.scene, stride, metrics, final_fit["summary"]))
    print(json.dumps({"stride": stride, "metrics": metrics, "final_gt_sim3": final_fit["summary"]}, indent=2, sort_keys=True))
    return summary


def make_overlap_windows(frame_ids: list[int], chunk_size: int, overlap_frames: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap_frames < 0 or overlap_frames >= chunk_size:
        raise ValueError("overlap_frames must satisfy 0 <= overlap < chunk_size")
    if not frame_ids:
        return []
    step = chunk_size - overlap_frames
    windows: list[list[int]] = []
    start = 0
    while start < len(frame_ids):
        window = frame_ids[start : min(start + chunk_size, len(frame_ids))]
        if window:
            windows.append([int(v) for v in window])
        if start + chunk_size >= len(frame_ids):
            break
        start += step
    return windows


def load_window_without_masks(stream: ScanNetStream, frame_ids: list[int]) -> dict[str, np.ndarray | list[int]]:
    """Load the geometry/RGB inputs needed by D4RT without probing 2D mask files."""

    rgbs = [stream.load_rgb(fid) for fid in frame_ids]
    depths = [stream.load_depth(fid) for fid in frame_ids]
    poses = [stream.load_pose(fid) for fid in frame_ids]
    masks = [np.zeros(rgb.shape[:2], dtype=np.int32) for rgb in rgbs]
    return {
        "frame_ids": list(frame_ids),
        "rgb": np.stack(rgbs, axis=0),
        "mask": np.stack(masks, axis=0),
        "depth": np.stack(depths, axis=0),
        "pose": np.stack(poses, axis=0),
        "intrinsics": stream.load_intrinsics(),
    }


def raw_valid_mask(chunk: ChunkRecord, args: argparse.Namespace) -> np.ndarray:
    uv = chunk.uv
    return (
        chunk.valid
        & np.isfinite(chunk.xyz).all(axis=-1)
        & np.isfinite(uv).all(axis=-1)
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] <= 1.0)
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] <= 1.0)
        & (chunk.visibility >= float(args.min_visibility))
        & (chunk.confidence >= float(args.min_confidence))
    )


def fit_overlap_transform(prev: ChunkRecord, curr: ChunkRecord, args: argparse.Namespace) -> dict[str, Any]:
    match = match_overlap_carriers(
        prev.raw_dict(),
        curr.raw_dict(),
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
        uv_radius=float(args.uv_radius),
        max_matches_per_frame=int(args.max_matches_per_frame),
    )
    source = match.curr_xyz.reshape(-1, 3)
    target = match.prev_xyz.reshape(-1, 3)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < 4:
        raise RuntimeError(
            f"Not enough overlap anchors for chunks {prev.chunk_index}->{curr.chunk_index}: "
            f"{source.shape[0]} anchors, stats={match.stats}"
        )
    first = fit_sim3_with_diagnostics(source, target)
    residual = np.asarray(first["residual"], dtype=np.float64)
    trim = float(args.fit_trim_percentile)
    kept = np.ones((source.shape[0],), dtype=bool)
    if 0.0 < trim < 100.0 and source.shape[0] >= 16:
        kept = residual <= float(np.percentile(residual, trim))
        if int(np.count_nonzero(kept)) >= 4 and int(np.count_nonzero(kept)) < source.shape[0]:
            fit = fit_sim3_with_diagnostics(source[kept], target[kept])
        else:
            fit = first
            kept = np.ones((source.shape[0],), dtype=bool)
    else:
        fit = first
    transform = Sim3Transform(
        scale=float(fit["scale"]),
        rot=np.asarray(fit["rot"], dtype=np.float64),
        trans=np.asarray(fit["trans"], dtype=np.float64),
    )
    row = json_safe(
        {
            **match.stats,
            "fit_anchor_count": int(source.shape[0]),
            "fit_kept_anchor_count": int(np.count_nonzero(kept)),
            "fit_trim_percentile": trim,
            "scale_curr_to_prev": float(transform.scale),
            "rotation_det_curr_to_prev": float(np.linalg.det(transform.rot)),
            "translation_norm_curr_to_prev": float(np.linalg.norm(transform.trans)),
            "residual_median_curr_to_prev": fit.get("residual_median"),
            "residual_p90_curr_to_prev": fit.get("residual_p90"),
            "residual_p95_curr_to_prev": fit.get("residual_p95"),
            "inlier_ratio_abs005_curr_to_prev": fit.get("inlier_ratio_abs005"),
            "inlier_ratio_abs010_curr_to_prev": fit.get("inlier_ratio_abs010"),
        }
    )
    return {"transform_curr_to_prev": transform, "row": row}


def collect_stitched_observations(chunks: list[ChunkRecord], args: argparse.Namespace) -> dict[str, np.ndarray]:
    xyz_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    chunk_parts: list[np.ndarray] = []
    conf_parts: list[np.ndarray] = []
    for chunk in chunks:
        ok = raw_valid_mask(chunk, args)
        if not np.any(ok):
            continue
        local_idx, carrier_idx = np.where(ok)
        xyz = apply_sim3_to_xyz(chunk.xyz[ok], transform=chunk.transform_to_scene)
        uv = chunk.uv[ok]
        frame_ids = np.asarray([chunk.frame_ids[int(i)] for i in local_idx.tolist()], dtype=np.int64)
        xyz_parts.append(xyz.astype(np.float32))
        uv_parts.append(uv.astype(np.float32))
        frame_parts.append(frame_ids)
        chunk_parts.append(np.full((xyz.shape[0],), int(chunk.chunk_index), dtype=np.int64))
        conf_parts.append(chunk.confidence[ok].astype(np.float32))
    if not xyz_parts:
        raise RuntimeError("No valid stitched D4RT observations")
    return {
        "xyz": np.concatenate(xyz_parts, axis=0),
        "uv": np.concatenate(uv_parts, axis=0),
        "frame_id": np.concatenate(frame_parts, axis=0),
        "chunk_id": np.concatenate(chunk_parts, axis=0),
        "confidence": np.concatenate(conf_parts, axis=0),
    }


def fit_final_gt_sim3(stitched: dict[str, np.ndarray], stream: ScanNetStream, args: argparse.Namespace) -> dict[str, Any]:
    xyz = np.asarray(stitched["xyz"], dtype=np.float32)
    uv = np.asarray(stitched["uv"], dtype=np.float32)
    frame_ids = np.asarray(stitched["frame_id"], dtype=np.int64)
    conf = np.asarray(stitched["confidence"], dtype=np.float32)
    max_anchors = int(args.max_sim3_anchors)
    candidate = np.flatnonzero(np.isfinite(xyz).all(axis=1) & np.isfinite(uv).all(axis=1))
    if candidate.size > max_anchors:
        order = np.argsort(np.nan_to_num(conf[candidate], nan=-np.inf))
        candidate = candidate[order[-max_anchors:]]
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    valid_by_frame: dict[int, int] = {}
    for frame_id in sorted(set(frame_ids[candidate].tolist())):
        idx = candidate[frame_ids[candidate] == int(frame_id)]
        if idx.size == 0:
            continue
        depth = stream.load_depth(int(frame_id))
        h, w = depth.shape[:2]
        xy = np.stack(
            [
                uv[idx, 0] * float(max(w - 1, 1)),
                uv[idx, 1] * float(max(h - 1, 1)),
            ],
            axis=1,
        )
        world, valid = backproject_xy_world(stream, int(frame_id), xy)
        if np.any(valid):
            source_parts.append(xyz[idx][valid])
            target_parts.append(world[valid])
            valid_by_frame[int(frame_id)] = int(np.count_nonzero(valid))
    if not source_parts:
        raise RuntimeError("No valid RGB-D/pose anchors for final GT Sim3")
    source = np.concatenate(source_parts, axis=0).astype(np.float32)
    target = np.concatenate(target_parts, axis=0).astype(np.float32)
    first = fit_sim3_umeyama(source, target)
    residual = np.asarray(first["residual"], dtype=np.float64)
    trim = float(args.fit_trim_percentile)
    kept = np.ones((source.shape[0],), dtype=bool)
    if 0.0 < trim < 100.0 and source.shape[0] >= 16:
        kept = residual <= float(np.percentile(residual, trim))
        if int(np.count_nonzero(kept)) >= 4 and int(np.count_nonzero(kept)) < source.shape[0]:
            fit = fit_sim3_umeyama(source[kept], target[kept])
        else:
            fit = first
            kept = np.ones((source.shape[0],), dtype=bool)
    else:
        fit = first
    final_residual = np.asarray(fit["residual"], dtype=np.float64)
    diag = residual_diagnostics(final_residual)
    transform = Sim3Transform(
        scale=float(fit["scale"]),
        rot=np.asarray(fit["rot"], dtype=np.float64),
        trans=np.asarray(fit["trans"], dtype=np.float64),
    )
    summary = json_safe(
        {
            "diagnostic_only": True,
            "gt_data_used": "ScanNet depth/pose RGB-D backprojection for final sequence-level Sim3 only",
            "candidate_anchor_count": int(candidate.size),
            "valid_anchor_count_before_trim": int(source.shape[0]),
            "valid_anchor_count_after_trim": int(np.count_nonzero(kept)),
            "frame_count_with_valid_anchors": int(len(valid_by_frame)),
            "fit_trim_percentile": trim,
            "scale_d4rt_to_gt": float(transform.scale),
            "rotation_d4rt_to_gt": transform.rot.astype(np.float64),
            "rotation_det_d4rt_to_gt": float(np.linalg.det(transform.rot)),
            "translation_d4rt_to_gt": transform.trans.astype(np.float64),
            "translation_norm_d4rt_to_gt": float(np.linalg.norm(transform.trans)),
            **{f"anchor_{key}": value for key, value in diag.items()},
        }
    )
    return {"transform": transform, "summary": summary}


def compute_metrics(scene: str, final_points: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    scene_points, _colors, mesh_path = _load_scene_mesh(scene)
    pts = np.asarray(final_points, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    metric_idx = sample_indices(pts.shape[0], int(args.max_metric_points), seed=f"{scene}:metric:d4rt")
    metric_pts = pts[metric_idx]
    gt_idx = sample_indices(scene_points.shape[0], int(args.max_gt_metric_points), seed=f"{scene}:metric:gt")
    metric_gt = scene_points[gt_idx]
    gt_tree = cKDTree(scene_points)
    d4rt_tree = cKDTree(metric_pts) if metric_pts.shape[0] > 0 else None
    d4rt_to_gt, _ = gt_tree.query(metric_pts, k=1) if metric_pts.shape[0] > 0 else (np.asarray([]), np.asarray([]))
    gt_to_d4rt = np.asarray([], dtype=np.float64)
    if d4rt_tree is not None and metric_gt.shape[0] > 0:
        gt_to_d4rt, _ = d4rt_tree.query(metric_gt, k=1)
    return json_safe(
        {
            "mesh_path": rel(mesh_path),
            "mesh_path_sha256": sha256_file(mesh_path),
            "full_d4rt_point_count_after_final_gt_sim3": int(pts.shape[0]),
            "sampled_d4rt_point_count_for_metrics": int(metric_pts.shape[0]),
            "sampled_gt_point_count_for_metrics": int(metric_gt.shape[0]),
            **dist_stats(d4rt_to_gt, "d4rt_to_gt_mesh_nn"),
            **dist_stats(gt_to_d4rt, "gt_mesh_to_d4rt_nn"),
        }
    )


def save_visual_points(
    out: Path,
    final_points: np.ndarray,
    stitched: dict[str, np.ndarray],
    stride: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pts = np.asarray(final_points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    chunk = np.asarray(stitched["chunk_id"], dtype=np.int64)[finite]
    frame = np.asarray(stitched["frame_id"], dtype=np.int64)[finite]
    conf = np.asarray(stitched["confidence"], dtype=np.float32)[finite]
    uv = np.asarray(stitched["uv"], dtype=np.float32)[finite]
    idx = sample_indices(pts.shape[0], int(args.max_visual_points_per_stride), seed=f"{args.scene}:stride:{stride}:visual")
    colors = sample_rgb_colors(args.scene, frame[idx], uv[idx], args)
    np.savez_compressed(
        out / "visual_points.npz",
        points=pts[idx],
        colors=colors,
        uv=uv[idx],
        chunk_id=chunk[idx],
        frame_id=frame[idx],
        confidence=conf[idx],
        stride=np.full((idx.shape[0],), int(stride), dtype=np.int64),
    )
    return {
        "visual_points_npz": rel(out / "visual_points.npz"),
        "visual_point_count": int(idx.shape[0]),
        "sampled": bool(idx.shape[0] < pts.shape[0]),
        "max_visual_points_per_stride": int(args.max_visual_points_per_stride),
        "color_source": "ScanNet RGB sampled at D4RT predicted uv/frame_id",
    }


def sample_rgb_colors(scene: str, frame_ids: np.ndarray, uv: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    stream = ScanNetStream(seq_name=scene, root=resolve_repo(args.scannet_root))
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float32)
    colors = np.zeros((frame_ids.shape[0], 3), dtype=np.uint8)
    for frame_id in sorted(set(frame_ids.tolist())):
        sel = frame_ids == int(frame_id)
        rgb = stream.load_rgb(int(frame_id))
        h, w = rgb.shape[:2]
        xy = np.rint(
            np.stack(
                [
                    uv[sel, 0] * float(max(w - 1, 1)),
                    uv[sel, 1] * float(max(h - 1, 1)),
                ],
                axis=1,
            )
        ).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, max(w - 1, 0))
        xy[:, 1] = np.clip(xy[:, 1], 0, max(h - 1, 0))
        colors[sel] = rgb[xy[:, 1], xy[:, 0], :3]
    return colors


def summarize_outputs(args: argparse.Namespace, extra_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    root = project(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for stride in args.strides:
        path = root / f"stride_{int(stride)}" / "stride_summary.json"
        if path.exists():
            rows.append(load_json(path))
    if extra_rows:
        by_stride = {int(row["stride"]): row for row in rows}
        for row in extra_rows:
            by_stride[int(row["stride"])] = row
        rows = [by_stride[k] for k in sorted(by_stride)]
    summary = {
        "phase": "v65_d4rt_stride_overlap_geometry_aggregate",
        "scene": args.scene,
        "requested_strides": [int(v) for v in args.strides],
        "completed_strides": [int(row["stride"]) for row in rows],
        "all_requested_strides_complete": sorted(int(row["stride"]) for row in rows)
        == sorted(int(v) for v in args.strides),
        "output_root": rel(root),
        "stride_rows": rows,
    }
    write_json(root / "geometry_summary.json", summary)
    write_csv(root / "geometry_metric_rows.csv", aggregate_metric_rows(rows))
    write_viewer_index(root, args, rows)
    print(json.dumps({"summary": rel(root / "geometry_summary.json"), "completed_strides": summary["completed_strides"]}, indent=2))
    return summary


def write_viewer_index(root: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    scene_points, scene_colors, mesh_path = _load_scene_mesh(args.scene)
    gt_labels = _load_gt(args.scene)
    gt_positive = gt_labels > 0
    gt_sem_colors = _id_colors(gt_labels[gt_positive])
    stride_payload: dict[str, Any] = {}
    for row in rows:
        stride = int(row["stride"])
        npz_path = root / f"stride_{stride}" / "visual_points.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path) as payload:
            pts = np.asarray(payload["points"], dtype=np.float32)
            if "colors" in payload:
                colors = np.asarray(payload["colors"], dtype=np.uint8)
            else:
                chunk = np.asarray(payload["chunk_id"], dtype=np.int64)
                colors = _window_colors(chunk)
        stride_payload[f"stride_{stride}_points"] = pts
        stride_payload[f"stride_{stride}_colors"] = colors
    np.savez_compressed(
        root / "viewer_layers.npz",
        gt_geometry_points=scene_points,
        gt_geometry_colors=scene_colors,
        gt_sem_points=scene_points[gt_positive],
        gt_sem_colors=gt_sem_colors,
        **stride_payload,
    )
    index = {
        "phase": "v65_d4rt_stride_overlap_geometry_viewer_export",
        "scene": args.scene,
        "viewer_layers_npz": rel(root / "viewer_layers.npz"),
        "mesh_path": rel(mesh_path),
        "mesh_path_sha256": sha256_file(mesh_path),
        "layers": ["gt_geometry", "gt_sem"]
        + [f"stride_{int(row['stride'])}" for row in rows if (root / f"stride_{int(row['stride'])}" / "visual_points.npz").exists()],
        "layer_controls_required": True,
        "viser_command": (
            f"PYTHONPATH=Stream3D /mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n loger python "
            f"Stream3D/tools/run_v65_d4rt_stride_overlap_geometry.py --serve --scene {args.scene} "
            f"--output-root {rel(root)} --port {args.port}"
        ),
    }
    write_json(root / "viewer_index.json", index)


def serve_viewer(args: argparse.Namespace) -> None:
    root = project(args.output_root)
    index = load_json(root / "viewer_index.json")
    layer_path = project(index["viewer_layers_npz"])
    with np.load(layer_path) as payload:
        layers = {key: np.asarray(payload[key]) for key in payload.files}
    import viser  # type: ignore

    server = viser.ViserServer(host=args.host, port=int(args.port), verbose=True)
    handles: dict[str, list[Any]] = {}

    def add(layer: str, handle: Any, visible: bool = True) -> None:
        handle.visible = bool(visible)
        handles.setdefault(layer, []).append(handle)

    def set_visible(layer: str, value: bool) -> None:
        for handle in handles.get(layer, []):
            handle.visible = bool(value)

    server.scene.add_grid("/v65_stride_overlap/grid", width=8.0, height=8.0, plane="xy", cell_size=1.0, section_size=4.0)
    add(
        "gt_geometry",
        server.scene.add_point_cloud(
            "/v65_stride_overlap/gt_geometry",
            points=layers["gt_geometry_points"],
            colors=layers["gt_geometry_colors"],
            point_size=0.006,
            point_shape="circle",
            precision="float32",
        ),
        True,
    )
    add(
        "gt_sem",
        server.scene.add_point_cloud(
            "/v65_stride_overlap/gt_sem",
            points=layers["gt_sem_points"],
            colors=layers["gt_sem_colors"],
            point_size=0.014,
            point_shape="circle",
            precision="float32",
        ),
        False,
    )
    stride_layers = sorted(
        int(key.removeprefix("stride_").removesuffix("_points"))
        for key in layers
        if key.startswith("stride_") and key.endswith("_points")
    )
    for stride in stride_layers:
        key = f"stride_{stride}"
        add(
            key,
            server.scene.add_point_cloud(
                f"/v65_stride_overlap/{key}",
                points=layers[f"{key}_points"],
                colors=layers[f"{key}_colors"],
                point_size=0.018,
                point_shape="circle",
                precision="float32",
            ),
            stride == 10,
        )
    with server.gui.add_folder("v65 stride overlap layers"):
        controls: dict[str, Any] = {}
        for layer in ["gt_geometry", "gt_sem", *[f"stride_{s}" for s in stride_layers]]:
            controls[layer] = server.gui.add_checkbox(layer, initial_value=(layer == "gt_geometry" or layer == "stride_10"))
            controls[layer].on_update(lambda event, layer=layer: set_visible(layer, bool(controls[layer].value)))
        show_all = server.gui.add_button("show all")
        hide_d4rt = server.gui.add_button("hide d4rt")

    @show_all.on_click
    def _(_: Any) -> None:
        for layer, control in controls.items():
            control.value = True
            set_visible(layer, True)

    @hide_d4rt.on_click
    def _(_: Any) -> None:
        for stride in stride_layers:
            layer = f"stride_{stride}"
            controls[layer].value = False
            set_visible(layer, False)

    status = {
        "phase": "v65_d4rt_stride_overlap_geometry_live_viewer",
        "scene": args.scene,
        "host": args.host,
        "port": int(args.port),
        "url": f"http://localhost:{int(args.port)}",
        "viewer_index": rel(root / "viewer_index.json"),
        "layer_count": int(len(handles)),
        "layers": sorted(handles.keys()),
        "controls": list(controls.keys()) + ["show all", "hide d4rt"],
    }
    write_json(root / "live_viewer_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()


def save_chunk_npz(path: Path, chunk: ChunkRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_ids=np.asarray(chunk.frame_ids, dtype=np.int64),
        xyz=chunk.xyz,
        uv=chunk.uv,
        valid=chunk.valid,
        visibility=chunk.visibility,
        confidence=chunk.confidence,
        carrier_id=chunk.carrier_id,
        src_frame_global=chunk.src_frame_global,
        src_xy=chunk.src_xy,
        transform_scale_to_scene=np.asarray([chunk.transform_to_scene.scale], dtype=np.float64),
        transform_rot_to_scene=chunk.transform_to_scene.rot,
        transform_trans_to_scene=chunk.transform_to_scene.trans,
    )


def dist_stats(dist: np.ndarray, prefix: str) -> dict[str, Any]:
    values = np.asarray(dist, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    out: dict[str, Any] = {f"{prefix}_count": int(values.size)}
    if values.size == 0:
        for key in ["mean", "median", "p75", "p90", "p95", "p99", "max"]:
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


def sample_indices(count: int, max_count: int, *, seed: str) -> np.ndarray:
    count = int(count)
    max_count = int(max_count)
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_count <= 0 or count <= max_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(stable_seed(seed))
    return np.sort(rng.choice(count, size=max_count, replace=False).astype(np.int64))


def stable_seed(text: str) -> int:
    value = 2166136261
    for ch in text.encode("utf-8"):
        value ^= int(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def frame_range_text(frame_ids: list[int]) -> str:
    if not frame_ids:
        return ""
    return f"{int(frame_ids[0])}..{int(frame_ids[-1])}({len(frame_ids)})"


def flatten_metric_rows(scene: str, stride: int, metrics: dict[str, Any], final_fit: dict[str, Any]) -> list[dict[str, Any]]:
    row = {
        "scene": scene,
        "stride": int(stride),
        **{key: value for key, value in metrics.items() if not isinstance(value, (dict, list))},
        **{f"final_sim3_{key}": value for key, value in final_fit.items() if not isinstance(value, (dict, list))},
    }
    return [json_safe(row)]


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics", {})
        final_fit = row.get("final_gt_sim3", {})
        out.extend(flatten_metric_rows(str(row.get("scene", "")), int(row.get("stride", -1)), metrics, final_fit))
    return out


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(json_safe(row.get(key))) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in fieldnames})


if __name__ == "__main__":
    main()
