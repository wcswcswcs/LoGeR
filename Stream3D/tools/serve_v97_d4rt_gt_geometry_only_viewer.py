#!/usr/bin/env python3
"""Serve only D4RT geometry and ScanNet GT geometry for audit visualization."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry_provider.common import backproject_xy_world  # noqa: E402
from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.self_stitch import residual_diagnostics  # noqa: E402
from stream4d_native.sim3 import fit_sim3_umeyama  # noqa: E402
from stream4d_native.v65_visualization_export import _load_scene_mesh  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs/audit/v97_corrected_d4rt_overlap_geometry_scene0011_stride5"


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs", "tools"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: str | Path) -> str:
    p = _project(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tag_suffix(args: argparse.Namespace) -> str:
    tag = str(getattr(args, "viewer_tag", "") or "").strip()
    if not tag:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tag)
    return f"_{safe}"


def _apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.full_like(pts, np.nan, dtype=np.float64)
    ok = np.isfinite(pts).all(axis=1)
    if np.any(ok):
        r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        t = np.asarray(trans, dtype=np.float64).reshape(3)
        out[ok] = float(scale) * (pts[ok] @ r.T) + t
    return out.astype(np.float32)


def _load_final_transform(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing stride summary for final GT-Sim3: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    transform = payload.get("final_gt_sim3_transform")
    if not isinstance(transform, dict):
        raise KeyError(f"{path} has no final_gt_sim3_transform")
    return transform


def _sample_rgb(scene: str, frame_ids: np.ndarray, uv: np.ndarray) -> np.ndarray:
    stream = ScanNetStream(seq_name=scene, root=ROOT / "data/scannet/processed")
    frame_ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
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


def _load_d4rt_from_chunks(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    output_root = _project(args.output_root)
    chunks_dir = _project(args.chunks_dir) if args.chunks_dir else output_root / "stride_5" / "chunks"
    summary_path = _project(args.stride_summary_json) if args.stride_summary_json else output_root / "stride_5" / "stride_summary.json"
    paths = sorted(chunks_dir.glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No chunk npz files under {chunks_dir}")
    final = _load_final_transform(summary_path)
    final_scale = float(final["scale"])
    final_rot = np.asarray(final["rot"], dtype=np.float64)
    final_trans = np.asarray(final["trans"], dtype=np.float64)
    point_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    chunk_parts: list[np.ndarray] = []
    total_slots = 0
    valid_count = 0
    finite_count = 0
    uv_inbounds_count = 0
    kept_before_cap = 0
    visibility_filtered_out = 0
    visibility_equal_threshold_count = 0
    confidence_filtered_out = 0
    for chunk_path in paths:
        with np.load(chunk_path) as data:
            xyz = np.asarray(data["xyz"], dtype=np.float32)
            uv = np.asarray(data["uv"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=bool)
            visibility = np.asarray(data["visibility"], dtype=np.float32)
            confidence = np.asarray(data["confidence"], dtype=np.float32)
            frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
            chunk_index = int(chunk_path.stem.split("_")[-1])
            chunk_scale = float(np.asarray(data["transform_scale_to_scene"]).reshape(-1)[0])
            chunk_rot = np.asarray(data["transform_rot_to_scene"], dtype=np.float64)
            chunk_trans = np.asarray(data["transform_trans_to_scene"], dtype=np.float64)
        frame_grid = np.repeat(frame_ids.reshape(-1, 1), xyz.shape[1], axis=1)
        total_slots += int(xyz.shape[0] * xyz.shape[1])
        base = valid & np.isfinite(xyz).all(axis=-1) & np.isfinite(uv).all(axis=-1)
        valid_count += int(np.count_nonzero(valid))
        finite_count += int(np.count_nonzero(base))
        inbounds = (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
        uv_inbounds_count += int(np.count_nonzero(base & inbounds))
        keep = base.copy()
        if not bool(args.include_uv_out_of_bounds):
            keep &= inbounds
        if float(args.min_visibility) >= 0.0:
            before = keep.copy()
            threshold = float(args.min_visibility)
            visibility_equal_threshold_count += int(np.count_nonzero(before & (visibility == threshold)))
            if bool(args.visibility_strict_gt):
                keep &= visibility > threshold
            else:
                keep &= visibility >= threshold
            visibility_filtered_out += int(np.count_nonzero(before & ~keep))
        if float(args.min_confidence) >= 0.0:
            before = keep.copy()
            keep &= confidence >= float(args.min_confidence)
            confidence_filtered_out += int(np.count_nonzero(before & ~keep))
        if not np.any(keep):
            continue
        flat = xyz[keep].reshape(-1, 3)
        stitched = _apply_sim3(flat, chunk_scale, chunk_rot, chunk_trans)
        aligned = _apply_sim3(stitched, final_scale, final_rot, final_trans)
        final_ok = np.isfinite(aligned).all(axis=1)
        aligned = aligned[final_ok]
        point_parts.append(aligned)
        uv_parts.append(uv[keep].reshape(-1, 2)[final_ok])
        frame_parts.append(frame_grid[keep].reshape(-1)[final_ok])
        chunk_parts.append(np.full((aligned.shape[0],), chunk_index, dtype=np.int64))
        kept_before_cap += int(aligned.shape[0])
    if not point_parts:
        raise RuntimeError("No D4RT points survived chunk export filtering")
    points = np.concatenate(point_parts, axis=0).astype(np.float32)
    uv_all = np.concatenate(uv_parts, axis=0).astype(np.float32)
    frames = np.concatenate(frame_parts, axis=0).astype(np.int64)
    chunks = np.concatenate(chunk_parts, axis=0).astype(np.int64)
    cap = int(args.max_d4rt_points)
    sampled = False
    if cap > 0 and points.shape[0] > cap:
        sampled = True
        idx = np.linspace(0, points.shape[0] - 1, num=cap, dtype=np.int64)
        points = points[idx]
        uv_all = uv_all[idx]
        frames = frames[idx]
        chunks = chunks[idx]
    colors = _sample_rgb(args.scene, frames, uv_all) if not bool(args.include_uv_out_of_bounds) else np.full((points.shape[0], 3), 96, dtype=np.uint8)
    filter_summary = {
        "d4rt_source": "fresh_chunk_npz_reconstructed",
        "chunks_dir": _rel(chunks_dir),
        "stride_summary_json": _rel(summary_path),
        "chunk_file_count": int(len(paths)),
        "total_slots": int(total_slots),
        "valid_count": int(valid_count),
        "finite_xyz_uv_count": int(finite_count),
        "uv_inbounds_count": int(uv_inbounds_count),
        "min_visibility": float(args.min_visibility),
        "visibility_comparator": ">" if bool(args.visibility_strict_gt) else ">=",
        "visibility_equal_threshold_count": int(visibility_equal_threshold_count),
        "min_confidence": float(args.min_confidence),
        "confidence_comparator": ">=",
        "include_uv_out_of_bounds": bool(args.include_uv_out_of_bounds),
        "visibility_filtered_out": int(visibility_filtered_out),
        "confidence_filtered_out": int(confidence_filtered_out),
        "kept_before_cap": int(kept_before_cap),
        "exported_count": int(points.shape[0]),
        "sampled_by_max_d4rt_points": bool(sampled),
        "max_d4rt_points": int(cap),
    }
    return points, colors, frames, chunks, filter_summary


def _chunk_paths(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    output_root = _project(args.output_root)
    chunks_dir = _project(args.chunks_dir) if args.chunks_dir else output_root / "stride_5" / "chunks"
    summary_path = _project(args.stride_summary_json) if args.stride_summary_json else output_root / "stride_5" / "stride_summary.json"
    paths = sorted(chunks_dir.glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No chunk npz files under {chunks_dir}")
    return chunks_dir, summary_path, paths


def _load_d4rt_candidates_from_chunks(args: argparse.Namespace) -> dict[str, Any]:
    chunks_dir, summary_path, paths = _chunk_paths(args)
    point_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    chunk_parts: list[np.ndarray] = []
    visibility_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    total_slots = 0
    valid_count = 0
    finite_count = 0
    uv_inbounds_count = 0
    for chunk_path in paths:
        with np.load(chunk_path) as data:
            xyz = np.asarray(data["xyz"], dtype=np.float32)
            uv = np.asarray(data["uv"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=bool)
            visibility = np.asarray(data["visibility"], dtype=np.float32)
            confidence = np.asarray(data["confidence"], dtype=np.float32)
            frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
            chunk_index = int(chunk_path.stem.split("_")[-1])
            chunk_scale = float(np.asarray(data["transform_scale_to_scene"]).reshape(-1)[0])
            chunk_rot = np.asarray(data["transform_rot_to_scene"], dtype=np.float64)
            chunk_trans = np.asarray(data["transform_trans_to_scene"], dtype=np.float64)
        frame_grid = np.repeat(frame_ids.reshape(-1, 1), xyz.shape[1], axis=1)
        total_slots += int(xyz.shape[0] * xyz.shape[1])
        base = valid & np.isfinite(xyz).all(axis=-1) & np.isfinite(uv).all(axis=-1)
        valid_count += int(np.count_nonzero(valid))
        finite_count += int(np.count_nonzero(base))
        inbounds = (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
        uv_inbounds_count += int(np.count_nonzero(base & inbounds))
        if not bool(args.include_uv_out_of_bounds):
            base &= inbounds
        if not np.any(base):
            continue
        stitched = _apply_sim3(xyz[base].reshape(-1, 3), chunk_scale, chunk_rot, chunk_trans)
        ok = np.isfinite(stitched).all(axis=1)
        if not np.any(ok):
            continue
        point_parts.append(stitched[ok])
        uv_parts.append(uv[base].reshape(-1, 2)[ok])
        frame_parts.append(frame_grid[base].reshape(-1)[ok])
        chunk_parts.append(np.full((int(np.count_nonzero(ok)),), chunk_index, dtype=np.int64))
        visibility_parts.append(visibility[base].reshape(-1)[ok])
        confidence_parts.append(confidence[base].reshape(-1)[ok])
    if not point_parts:
        raise RuntimeError("No D4RT candidate points survived base chunk filtering")
    points = np.concatenate(point_parts, axis=0).astype(np.float32)
    uv_all = np.concatenate(uv_parts, axis=0).astype(np.float32)
    frames = np.concatenate(frame_parts, axis=0).astype(np.int64)
    chunks = np.concatenate(chunk_parts, axis=0).astype(np.int64)
    visibility_all = np.concatenate(visibility_parts, axis=0).astype(np.float32)
    confidence_all = np.concatenate(confidence_parts, axis=0).astype(np.float32)
    colors = (
        _sample_rgb(args.scene, frames, uv_all)
        if not bool(args.include_uv_out_of_bounds)
        else np.full((points.shape[0], 3), 96, dtype=np.uint8)
    )
    return {
        "points_pre_final_sim3": points,
        "uv": uv_all,
        "frame_id": frames,
        "chunk_id": chunks,
        "visibility": visibility_all,
        "confidence": confidence_all,
        "colors": colors,
        "base_summary": {
            "d4rt_source": "fresh_chunk_npz_reconstructed_interactive_candidates",
            "chunks_dir": _rel(chunks_dir),
            "stride_summary_json": _rel(summary_path),
            "chunk_file_count": int(len(paths)),
            "total_slots": int(total_slots),
            "valid_count": int(valid_count),
            "finite_xyz_uv_count": int(finite_count),
            "uv_inbounds_count": int(uv_inbounds_count),
            "include_uv_out_of_bounds": bool(args.include_uv_out_of_bounds),
            "pre_threshold_candidate_count": int(points.shape[0]),
            "pre_final_sim3_space": "D4RT overlap-stitched scene space; final GT-Sim3 is recomputed after GUI threshold filtering",
        },
    }


def _filter_candidate_mask(
    candidates: dict[str, Any],
    *,
    min_visibility: float,
    visibility_strict_gt: bool,
    min_confidence: float,
    max_d4rt_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    visibility = np.asarray(candidates["visibility"], dtype=np.float32)
    confidence = np.asarray(candidates["confidence"], dtype=np.float32)
    keep = np.ones((visibility.shape[0],), dtype=bool)
    visibility_filtered_out = 0
    visibility_equal_threshold_count = 0
    confidence_filtered_out = 0
    if float(min_visibility) >= 0.0:
        before = keep.copy()
        threshold = float(min_visibility)
        visibility_equal_threshold_count = int(np.count_nonzero(before & (visibility == threshold)))
        if bool(visibility_strict_gt):
            keep &= visibility > threshold
        else:
            keep &= visibility >= threshold
        visibility_filtered_out = int(np.count_nonzero(before & ~keep))
    if float(min_confidence) >= 0.0:
        before = keep.copy()
        keep &= confidence >= float(min_confidence)
        confidence_filtered_out = int(np.count_nonzero(before & ~keep))
    base = dict(candidates.get("base_summary", {}))
    base.update(
        {
            "min_visibility": float(min_visibility),
            "visibility_comparator": ">" if bool(visibility_strict_gt) else ">=",
            "visibility_equal_threshold_count": int(visibility_equal_threshold_count),
            "min_confidence": float(min_confidence),
            "confidence_comparator": ">=",
            "visibility_filtered_out": int(visibility_filtered_out),
            "confidence_filtered_out": int(confidence_filtered_out),
            "kept_before_cap": int(np.count_nonzero(keep)),
            "max_d4rt_points": int(max_d4rt_points),
            "sampled_by_max_d4rt_points": False,
        }
    )
    return keep, base


def _fit_dynamic_final_gt_sim3(
    candidates: dict[str, Any],
    keep: np.ndarray,
    stream: ScanNetStream,
    args: argparse.Namespace,
) -> dict[str, Any]:
    xyz = np.asarray(candidates["points_pre_final_sim3"], dtype=np.float32)[keep]
    uv = np.asarray(candidates["uv"], dtype=np.float32)[keep]
    frame_ids = np.asarray(candidates["frame_id"], dtype=np.int64)[keep]
    confidence = np.asarray(candidates["confidence"], dtype=np.float32)[keep]
    max_anchors = int(args.max_sim3_anchors)
    candidate = np.flatnonzero(np.isfinite(xyz).all(axis=1) & np.isfinite(uv).all(axis=1))
    if candidate.size > max_anchors:
        order = np.argsort(np.nan_to_num(confidence[candidate], nan=-np.inf))
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
        raise RuntimeError("No valid RGB-D/pose anchors for dynamic final GT-Sim3")
    source = np.concatenate(source_parts, axis=0).astype(np.float32)
    target = np.concatenate(target_parts, axis=0).astype(np.float32)
    if source.shape[0] < 4:
        raise RuntimeError(f"Need at least 4 anchors for dynamic final GT-Sim3, got {source.shape[0]}")
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
    transform = {
        "scale": float(fit["scale"]),
        "rot": np.asarray(fit["rot"], dtype=np.float64),
        "trans": np.asarray(fit["trans"], dtype=np.float64),
    }
    summary = {
        "diagnostic_only": True,
        "gt_data_used": "ScanNet depth/pose RGB-D backprojection for GUI threshold-specific final Sim3 only",
        "candidate_anchor_count": int(candidate.size),
        "valid_anchor_count_before_trim": int(source.shape[0]),
        "valid_anchor_count_after_trim": int(np.count_nonzero(kept)),
        "frame_count_with_valid_anchors": int(len(valid_by_frame)),
        "fit_trim_percentile": float(trim),
        "scale_d4rt_to_gt": float(transform["scale"]),
        "rotation_d4rt_to_gt": np.asarray(transform["rot"], dtype=np.float64),
        "rotation_det_d4rt_to_gt": float(np.linalg.det(np.asarray(transform["rot"], dtype=np.float64))),
        "translation_d4rt_to_gt": np.asarray(transform["trans"], dtype=np.float64),
        "translation_norm_d4rt_to_gt": float(np.linalg.norm(np.asarray(transform["trans"], dtype=np.float64))),
        **{f"anchor_{key}": value for key, value in diag.items()},
    }
    return {"transform": transform, "summary": summary}


def _dynamic_points_for_threshold(
    candidates: dict[str, Any],
    stream: ScanNetStream,
    args: argparse.Namespace,
    *,
    min_visibility: float,
    visibility_strict_gt: bool,
    min_confidence: float,
) -> dict[str, Any]:
    keep, filter_summary = _filter_candidate_mask(
        candidates,
        min_visibility=float(min_visibility),
        visibility_strict_gt=bool(visibility_strict_gt),
        min_confidence=float(min_confidence),
        max_d4rt_points=int(args.max_d4rt_points),
    )
    if int(np.count_nonzero(keep)) <= 0:
        raise RuntimeError("No D4RT points survived current GUI thresholds")
    fit = _fit_dynamic_final_gt_sim3(candidates, keep, stream, args)
    transform = fit["transform"]
    points = _apply_sim3(
        np.asarray(candidates["points_pre_final_sim3"], dtype=np.float32)[keep],
        float(transform["scale"]),
        np.asarray(transform["rot"], dtype=np.float64),
        np.asarray(transform["trans"], dtype=np.float64),
    )
    colors = np.asarray(candidates["colors"], dtype=np.uint8)[keep]
    frames = np.asarray(candidates["frame_id"], dtype=np.int64)[keep]
    chunks = np.asarray(candidates["chunk_id"], dtype=np.int64)[keep]
    final_ok = np.isfinite(points).all(axis=1)
    points = points[final_ok]
    colors = colors[final_ok]
    frames = frames[final_ok]
    chunks = chunks[final_ok]
    cap = int(args.max_d4rt_points)
    sampled = False
    if cap > 0 and points.shape[0] > cap:
        sampled = True
        idx = np.linspace(0, points.shape[0] - 1, num=cap, dtype=np.int64)
        points = points[idx]
        colors = colors[idx]
        frames = frames[idx]
        chunks = chunks[idx]
    filter_summary.update(
        {
            "exported_count": int(points.shape[0]),
            "sampled_by_max_d4rt_points": bool(sampled),
            "dynamic_final_gt_sim3_after_threshold": True,
        }
    )
    return {
        "points": points.astype(np.float32),
        "colors": colors.astype(np.uint8),
        "frame_id": frames.astype(np.int64),
        "chunk_id": chunks.astype(np.int64),
        "filter_summary": filter_summary,
        "final_gt_sim3": fit["summary"],
        "final_gt_sim3_transform": transform,
    }


def export_layers(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    gt_points, gt_colors, mesh_path = _load_scene_mesh(args.scene)
    filter_summary: dict[str, Any]
    visual_path = _project(args.visual_points_npz) if args.visual_points_npz else output_root / "stride_5" / "visual_points.npz"
    if bool(args.from_chunks):
        d4rt_points, d4rt_colors, d4rt_frame, d4rt_chunk, filter_summary = _load_d4rt_from_chunks(args)
    else:
        if not visual_path.exists():
            raise FileNotFoundError(f"Missing D4RT visual points: {visual_path}")
        with np.load(visual_path) as payload:
            d4rt_points = np.asarray(payload["points"], dtype=np.float32)
            if "colors" in payload.files:
                d4rt_colors = np.asarray(payload["colors"], dtype=np.uint8)
            else:
                d4rt_colors = np.full((d4rt_points.shape[0], 3), 80, dtype=np.uint8)
            d4rt_frame = np.asarray(payload["frame_id"], dtype=np.int64) if "frame_id" in payload.files else np.zeros((d4rt_points.shape[0],), dtype=np.int64)
            d4rt_chunk = np.asarray(payload["chunk_id"], dtype=np.int64) if "chunk_id" in payload.files else np.zeros((d4rt_points.shape[0],), dtype=np.int64)
        filter_summary = {
            "d4rt_source": "visual_points_npz",
            "visual_points_npz": _rel(visual_path),
            "exported_count": int(d4rt_points.shape[0]),
        }
    if int(args.max_gt_points) > 0 and gt_points.shape[0] > int(args.max_gt_points):
        idx = np.linspace(0, gt_points.shape[0] - 1, num=int(args.max_gt_points), dtype=np.int64)
        gt_points = gt_points[idx]
        gt_colors = gt_colors[idx]
    suffix = _tag_suffix(args)
    layer_path = output_root / f"geometry_only_layers{suffix}.npz"
    np.savez_compressed(
        layer_path,
        gt_geometry_points=gt_points.astype(np.float32),
        gt_geometry_colors=gt_colors.astype(np.uint8),
        d4rt_geometry_points=d4rt_points.astype(np.float32),
        d4rt_geometry_colors=d4rt_colors.astype(np.uint8),
        d4rt_frame_id=d4rt_frame.astype(np.int64),
        d4rt_chunk_id=d4rt_chunk.astype(np.int64),
    )
    index = {
        "phase": "v97_d4rt_gt_geometry_only_viewer_export",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scene": args.scene,
        "output_root": _rel(output_root),
        "viewer_tag": str(args.viewer_tag),
        "visual_points_npz": _rel(visual_path) if visual_path.exists() else "",
        "visual_points_npz_sha256": _sha256_file(visual_path),
        "geometry_only_layers_npz": _rel(layer_path),
        "geometry_only_layers_npz_sha256": _sha256_file(layer_path),
        "mesh_path": _rel(mesh_path),
        "mesh_path_sha256": _sha256_file(mesh_path),
        "layers": ["GT geometry", "D4RT geometry"],
        "no_semantic_layers": True,
        "gt_geometry_point_count": int(gt_points.shape[0]),
        "d4rt_geometry_point_count": int(d4rt_points.shape[0]),
        "d4rt_color_source": "ScanNet RGB sampled at D4RT predicted uv/frame_id; geometry only, no semantic labels",
        "filter_summary": filter_summary,
        "diagnostic_contract": {
            "uses_gt_for_prediction": False,
            "uses_gt_for_visual_alignment": True,
            "uses_rgbd_pose_mesh_for_visual_alignment": True,
            "is_method_safe": False,
            "is_diagnostic_visualization": True,
            "geometry_alignment": "D4RT-only overlap stitch followed by one final diagnostic GT-Sim3 from the upstream visual_points artifact",
        },
    }
    _write_json(output_root / f"geometry_only_viewer_index{suffix}.json", index)
    return index


def serve_interactive_threshold_gui(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    suffix = _tag_suffix(args)
    gt_points, gt_colors, mesh_path = _load_scene_mesh(args.scene)
    if int(args.max_gt_points) > 0 and gt_points.shape[0] > int(args.max_gt_points):
        idx = np.linspace(0, gt_points.shape[0] - 1, num=int(args.max_gt_points), dtype=np.int64)
        gt_points = gt_points[idx]
        gt_colors = gt_colors[idx]
    candidates = _load_d4rt_candidates_from_chunks(args)
    stream = ScanNetStream(seq_name=args.scene, root=ROOT / "data/scannet/processed")
    current = _dynamic_points_for_threshold(
        candidates,
        stream,
        args,
        min_visibility=float(args.min_visibility),
        visibility_strict_gt=bool(args.visibility_strict_gt),
        min_confidence=float(args.min_confidence),
    )
    candidate_index = {
        "phase": "v97_d4rt_gt_geometry_only_interactive_candidates",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scene": args.scene,
        "output_root": _rel(output_root),
        "viewer_tag": str(args.viewer_tag),
        "layers": ["GT geometry", "D4RT geometry"],
        "no_semantic_layers": True,
        "interactive_threshold_gui": True,
        "dynamic_final_gt_sim3_after_threshold": True,
        "candidate_base_summary": candidates.get("base_summary", {}),
        "mesh_path": _rel(mesh_path),
        "mesh_path_sha256": _sha256_file(mesh_path),
    }
    _write_json(output_root / f"geometry_only_interactive_candidate_index{suffix}.json", candidate_index)

    import viser  # type: ignore

    server = viser.ViserServer(host=args.host, port=int(args.port), verbose=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/v97_d4rt_gt_geometry_only/grid",
        width=8.0,
        height=8.0,
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
        position=(0.0, 0.0, -0.02),
    )
    gt_handle = server.scene.add_point_cloud(
        "/v97_d4rt_gt_geometry_only/GT geometry",
        points=gt_points,
        colors=gt_colors,
        point_size=float(args.gt_point_size),
        point_shape="circle",
        visible=True,
        precision="float32",
    )
    d4rt_handle = server.scene.add_point_cloud(
        "/v97_d4rt_gt_geometry_only/D4RT geometry",
        points=current["points"],
        colors=current["colors"],
        point_size=float(args.d4rt_point_size),
        point_shape="circle",
        visible=True,
        precision="float32",
    )
    gt_toggle = server.gui.add_checkbox("GT geometry", True)
    d4rt_toggle = server.gui.add_checkbox("D4RT geometry", True)
    vis_slider = server.gui.add_slider(
        "visibility",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=float(args.min_visibility),
        marks=((0.0, "0"), (0.5, "0.5"), (1.0, "1")),
    )
    conf_slider = server.gui.add_slider(
        "confidence",
        min=0.0,
        max=1.0,
        step=0.01,
        initial_value=float(args.min_confidence),
        marks=((0.0, "0"), (0.5, "0.5"), (1.0, "1")),
    )
    strict_toggle = server.gui.add_checkbox("visibility > threshold", bool(args.visibility_strict_gt))
    auto_toggle = server.gui.add_checkbox("auto apply", bool(args.auto_apply_gui_thresholds))
    apply_button = server.gui.add_button("Apply thresholds + Sim3")
    count_handle = server.gui.add_number("D4RT points", int(current["points"].shape[0]), disabled=True)
    scale_handle = server.gui.add_number(
        "Sim3 scale",
        float(current["final_gt_sim3"].get("scale_d4rt_to_gt") or 0.0),
        disabled=True,
    )
    p90_handle = server.gui.add_number(
        "Sim3 residual p90",
        float(current["final_gt_sim3"].get("anchor_residual_p90") or 0.0),
        disabled=True,
    )
    status_text = server.gui.add_text("status", "ready", disabled=True)

    @gt_toggle.on_update
    def _(_: Any) -> None:
        gt_handle.visible = bool(gt_toggle.value)

    @d4rt_toggle.on_update
    def _(_: Any) -> None:
        d4rt_handle.visible = bool(d4rt_toggle.value)

    def status_payload(display: dict[str, Any], trigger: str, error: str | None = None) -> dict[str, Any]:
        return {
            "phase": "v97_d4rt_gt_geometry_only_live_viewer",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "scene": args.scene,
            "host": args.host,
            "port": int(args.port),
            "url": f"http://localhost:{int(args.port)}",
            "viewer_index": _rel(output_root / f"geometry_only_interactive_candidate_index{suffix}.json"),
            "geometry_only_layers_npz": "",
            "layers": ["GT geometry", "D4RT geometry"],
            "toggles": ["GT geometry", "D4RT geometry"],
            "no_semantic_layers": True,
            "interactive_threshold_gui": True,
            "auto_apply_gui_thresholds": bool(auto_toggle.value),
            "dynamic_final_gt_sim3_after_threshold": True,
            "last_update_trigger": trigger,
            "last_update_error": error,
            "gt_geometry_point_count": int(gt_points.shape[0]),
            "d4rt_geometry_point_count": int(display["points"].shape[0]),
            "filter_summary": display.get("filter_summary", {}),
            "final_gt_sim3": display.get("final_gt_sim3", {}),
            "final_gt_sim3_transform": display.get("final_gt_sim3_transform", {}),
            "candidate_base_summary": candidates.get("base_summary", {}),
            "diagnostic_contract": {
                "uses_gt_for_prediction": False,
                "uses_gt_for_visual_alignment": True,
                "uses_rgbd_pose_mesh_for_visual_alignment": True,
                "is_method_safe": False,
                "is_diagnostic_visualization": True,
                "geometry_alignment": "D4RT overlap stitch, GUI threshold filter, then threshold-specific final diagnostic GT-Sim3",
            },
        }

    status_path = output_root / f"geometry_only_live_viewer_status{suffix}.json"
    current_box: dict[str, Any] = {"display": current}
    _write_json(status_path, status_payload(current, "initial"))
    print(json.dumps(_jsonable(status_payload(current, "initial")), indent=2, sort_keys=True), flush=True)
    update_lock = threading.Lock()

    def apply_current_thresholds(trigger: str) -> None:
        if not update_lock.acquire(blocking=False):
            status_text.value = "busy; skipped overlapping update"
            return
        try:
            status_text.value = "fitting Sim3..."
            updated = _dynamic_points_for_threshold(
                candidates,
                stream,
                args,
                min_visibility=float(vis_slider.value),
                visibility_strict_gt=bool(strict_toggle.value),
                min_confidence=float(conf_slider.value),
            )
            d4rt_handle.points = updated["points"]
            d4rt_handle.colors = updated["colors"]
            count_handle.value = int(updated["points"].shape[0])
            scale_handle.value = float(updated["final_gt_sim3"].get("scale_d4rt_to_gt") or 0.0)
            p90_handle.value = float(updated["final_gt_sim3"].get("anchor_residual_p90") or 0.0)
            current_box["display"] = updated
            _write_json(status_path, status_payload(updated, trigger))
            status_text.value = (
                f"points={updated['points'].shape[0]} "
                f"scale={float(updated['final_gt_sim3'].get('scale_d4rt_to_gt') or 0.0):.6f} "
                f"p90={float(updated['final_gt_sim3'].get('anchor_residual_p90') or 0.0):.6f}"
            )
        except Exception as exc:  # noqa: BLE001 - audit viewer should report exact GUI update failure.
            current_display = current_box["display"]
            _write_json(status_path, status_payload(current_display, trigger, error=repr(exc)))
            status_text.value = f"error: {exc!r}"
            print(f"[interactive-threshold-gui] update failed trigger={trigger}: {exc!r}", flush=True)
        finally:
            update_lock.release()

    @apply_button.on_click
    def _(_: Any) -> None:
        apply_current_thresholds("button")

    def maybe_auto(trigger: str) -> None:
        if bool(auto_toggle.value):
            apply_current_thresholds(trigger)

    @vis_slider.on_update
    def _(_: Any) -> None:
        maybe_auto("visibility_slider")

    @conf_slider.on_update
    def _(_: Any) -> None:
        maybe_auto("confidence_slider")

    @strict_toggle.on_update
    def _(_: Any) -> None:
        maybe_auto("strict_visibility_toggle")

    stop = {"value": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not stop["value"]:
        time.sleep(1.0)
    server.stop()
    return status_payload(current_box["display"], "shutdown")


def serve(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.interactive_threshold_gui):
        if not bool(args.from_chunks):
            raise ValueError("--interactive-threshold-gui requires --from-chunks")
        return serve_interactive_threshold_gui(args)
    output_root = _project(args.output_root)
    suffix = _tag_suffix(args)
    index_path = output_root / f"geometry_only_viewer_index{suffix}.json"
    if args.rebuild_layers or not index_path.exists():
        index = export_layers(args)
    else:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    layer_path = _project(index["geometry_only_layers_npz"])
    with np.load(layer_path) as payload:
        gt_points = np.asarray(payload["gt_geometry_points"], dtype=np.float32)
        gt_colors = np.asarray(payload["gt_geometry_colors"], dtype=np.uint8)
        d4rt_points = np.asarray(payload["d4rt_geometry_points"], dtype=np.float32)
        d4rt_colors = np.asarray(payload["d4rt_geometry_colors"], dtype=np.uint8)

    import viser  # type: ignore

    server = viser.ViserServer(host=args.host, port=int(args.port), verbose=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/v97_d4rt_gt_geometry_only/grid",
        width=8.0,
        height=8.0,
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
        position=(0.0, 0.0, -0.02),
    )
    gt_handle = server.scene.add_point_cloud(
        "/v97_d4rt_gt_geometry_only/GT geometry",
        points=gt_points,
        colors=gt_colors,
        point_size=float(args.gt_point_size),
        point_shape="circle",
        visible=True,
        precision="float32",
    )
    d4rt_handle = server.scene.add_point_cloud(
        "/v97_d4rt_gt_geometry_only/D4RT geometry",
        points=d4rt_points,
        colors=d4rt_colors,
        point_size=float(args.d4rt_point_size),
        point_shape="circle",
        visible=True,
        precision="float32",
    )
    gt_toggle = server.gui.add_checkbox("GT geometry", True)
    d4rt_toggle = server.gui.add_checkbox("D4RT geometry", True)

    @gt_toggle.on_update
    def _(_: Any) -> None:
        gt_handle.visible = bool(gt_toggle.value)

    @d4rt_toggle.on_update
    def _(_: Any) -> None:
        d4rt_handle.visible = bool(d4rt_toggle.value)

    status = {
        "phase": "v97_d4rt_gt_geometry_only_live_viewer",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scene": args.scene,
        "host": args.host,
        "port": int(args.port),
        "url": f"http://localhost:{int(args.port)}",
        "viewer_index": _rel(index_path),
        "geometry_only_layers_npz": index.get("geometry_only_layers_npz"),
        "layers": ["GT geometry", "D4RT geometry"],
        "toggles": ["GT geometry", "D4RT geometry"],
        "no_semantic_layers": True,
        "gt_geometry_point_count": int(gt_points.shape[0]),
        "d4rt_geometry_point_count": int(d4rt_points.shape[0]),
        "filter_summary": index.get("filter_summary", {}),
        "diagnostic_contract": index.get("diagnostic_contract", {}),
    }
    _write_json(output_root / f"geometry_only_live_viewer_status{suffix}.json", status)
    print(json.dumps(_jsonable(status), indent=2, sort_keys=True), flush=True)

    stop = {"value": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not stop["value"]:
        time.sleep(1.0)
    server.stop()
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="scene0011_00")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--visual-points-npz", default="")
    parser.add_argument("--from-chunks", action="store_true")
    parser.add_argument("--chunks-dir", default="")
    parser.add_argument("--stride-summary-json", default="")
    parser.add_argument("--min-visibility", type=float, default=-1.0)
    parser.add_argument("--visibility-strict-gt", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=-1.0)
    parser.add_argument("--include-uv-out-of-bounds", action="store_true")
    parser.add_argument("--max-d4rt-points", type=int, default=0)
    parser.add_argument("--fit-trim-percentile", type=float, default=90.0)
    parser.add_argument("--max-sim3-anchors", type=int, default=120000)
    parser.add_argument("--interactive-threshold-gui", action="store_true")
    parser.add_argument("--auto-apply-gui-thresholds", action="store_true")
    parser.add_argument("--viewer-tag", default="")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--gt-point-size", type=float, default=0.006)
    parser.add_argument("--d4rt-point-size", type=float, default=0.014)
    parser.add_argument("--max-gt-points", type=int, default=0)
    parser.add_argument("--rebuild-layers", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.export_only:
        print(json.dumps(_jsonable(export_layers(args)), indent=2, sort_keys=True))
    else:
        serve(args)


if __name__ == "__main__":
    main()
