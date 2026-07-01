#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
STREAM3D_ROOT = ROOT / "Stream3D"
sys.path.insert(0, str(STREAM3D_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry_provider.common import fit_transform  # noqa: E402
from stream4d_native.sim3 import apply_sim3_to_xyz  # noqa: E402


DEFAULT_SCENE = "scene0011_00"
DEFAULT_DA3_ROOT = STREAM3D_ROOT / "outputs" / "audit" / "v99_phase10s_da3_holdout_chunk32o3_scene0011_base"
DEFAULT_DA3_MANIFEST = (
    STREAM3D_ROOT
    / "outputs"
    / "audit"
    / "v99_phase10s_da3_holdout_chunk32o3_scene0011_input"
    / "frame_manifest_rows.csv"
)
DEFAULT_D4RT_ROWS = (
    STREAM3D_ROOT
    / "outputs"
    / "audit"
    / "v99_phase10w_d4rt_holdout_chunk32o3_stitched_scene0011"
    / "micro_track_rows.csv"
)
DEFAULT_D4RT_SUMMARY = (
    STREAM3D_ROOT
    / "outputs"
    / "audit"
    / "v99_phase10w_d4rt_holdout_chunk32o3_stitched_scene0011"
    / "summary.json"
)
DEFAULT_OUTPUT_ROOT = (
    STREAM3D_ROOT
    / "outputs"
    / "audit"
    / "v99_phase10aa_da3_d4rt_sim3_alignment_scene0011"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _residual_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ["mean", "p50", "p75", "p90", "p95", "p99", "max"]}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p75": float(np.percentile(values, 75.0)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def _sample_indices(count: int, sample_count: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if sample_count <= 0 or count <= sample_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(count, size=int(sample_count), replace=False)
    idx.sort()
    return idx.astype(np.int64)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_path"] = str(path)
    payload["_exists"] = True
    return payload


def _load_manifest(path: Path, scene_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"scene_id", "da3_frame_index", "frame_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df[df["scene_id"].astype(str) == str(scene_id)].copy()
    if df.empty:
        raise RuntimeError(f"{path} has no rows for scene_id={scene_id}")
    df["da3_frame_index"] = df["da3_frame_index"].astype(np.int64)
    df["frame_id"] = df["frame_id"].astype(np.int64)
    return df.sort_values("da3_frame_index").reset_index(drop=True)


def _parse_da3_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    processing_match = re.search(
        r"Processing\s+(\d+)\s+images\s+in\s+(\d+)\s+chunks\s+of\s+size\s+(\d+)\s+with\s+(\d+)\s+overlap",
        text,
    )
    total_points_match = re.search(r"Merge completed!\s+Total points:\s+(\d+)", text)
    return {
        "path": str(path),
        "exists": True,
        "completed": "DA3-Streaming done." in text,
        "frame_count": int(processing_match.group(1)) if processing_match else None,
        "chunk_count": int(processing_match.group(2)) if processing_match else None,
        "chunk_size": int(processing_match.group(3)) if processing_match else None,
        "overlap": int(processing_match.group(4)) if processing_match else None,
        "combined_pcd_point_count": int(total_points_match.group(1)) if total_points_match else None,
        "overlap_matched_points": [int(value) for value in re.findall(r"The number of corresponding points matched:\s+(\d+)", text)],
        "overlap_alignment_mean_errors": [float(value) for value in re.findall(r"Mean error:\s+([0-9.eE+-]+)", text)],
    }


class DA3FrameCache:
    def __init__(self, da3_root: Path, manifest: pd.DataFrame) -> None:
        self.da3_root = da3_root
        self.frame_to_da3 = {
            int(row.frame_id): int(row.da3_frame_index)
            for row in manifest.itertuples(index=False)
        }
        poses_path = da3_root / "camera_poses.txt"
        if not poses_path.is_file():
            raise FileNotFoundError(f"missing DA3 camera poses: {poses_path}")
        self.poses = np.loadtxt(poses_path).reshape(-1, 4, 4).astype(np.float64)
        self._cache: dict[int, dict[str, np.ndarray]] = {}

    def load(self, frame_id: int) -> dict[str, np.ndarray]:
        frame_id = int(frame_id)
        if frame_id in self._cache:
            return self._cache[frame_id]
        if frame_id not in self.frame_to_da3:
            raise KeyError(f"frame_id={frame_id} not in DA3 manifest")
        da3_index = int(self.frame_to_da3[frame_id])
        npz_path = self.da3_root / "results_output" / f"frame_{da3_index}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"missing DA3 frame npz: {npz_path}")
        with np.load(npz_path) as payload:
            image = np.asarray(payload["image"], dtype=np.uint8)
            depth = np.asarray(payload["depth"], dtype=np.float64)
            conf = np.asarray(payload["conf"], dtype=np.float64) if "conf" in payload.files else np.ones_like(depth)
            intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
        if da3_index >= self.poses.shape[0]:
            raise IndexError(f"DA3 pose index {da3_index} outside camera_poses length {self.poses.shape[0]}")
        frame = {
            "image": image,
            "depth": depth,
            "conf": conf,
            "intrinsics": intrinsics,
            "pose_c2w": self.poses[da3_index],
            "da3_frame_index": np.asarray(da3_index, dtype=np.int64),
        }
        self._cache[frame_id] = frame
        return frame

    def backproject(self, frame_id: int, xy_px: np.ndarray, conf_min: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = self.load(int(frame_id))
        depth = np.asarray(frame["depth"], dtype=np.float64)
        image = np.asarray(frame["image"], dtype=np.uint8)
        conf = np.asarray(frame["conf"], dtype=np.float64)
        intrinsics = np.asarray(frame["intrinsics"], dtype=np.float64)
        pose = np.asarray(frame["pose_c2w"], dtype=np.float64)
        h, w = depth.shape[:2]
        xy_px = np.asarray(xy_px, dtype=np.float64)
        x = np.rint(xy_px[:, 0]).astype(np.int64)
        y = np.rint(xy_px[:, 1]).astype(np.int64)
        in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        world = np.full((xy_px.shape[0], 3), np.nan, dtype=np.float32)
        colors = np.zeros((xy_px.shape[0], 3), dtype=np.uint8)
        valid = np.zeros((xy_px.shape[0],), dtype=bool)
        if not np.any(in_bounds):
            return world, colors, valid
        z = depth[y[in_bounds], x[in_bounds]].astype(np.float64)
        score = conf[y[in_bounds], x[in_bounds]].astype(np.float64)
        depth_ok = np.isfinite(z) & (z > 0.0) & np.isfinite(score) & (score >= float(conf_min))
        source_indices = np.flatnonzero(in_bounds)[depth_ok]
        if source_indices.size == 0:
            return world, colors, valid
        pix = np.stack(
            [
                x[source_indices].astype(np.float64),
                y[source_indices].astype(np.float64),
                np.ones(source_indices.shape[0], dtype=np.float64),
            ],
            axis=0,
        )
        rays = np.linalg.inv(intrinsics) @ pix
        cam = rays.T * z[depth_ok, None]
        hom = np.concatenate([cam, np.ones((cam.shape[0], 1), dtype=np.float64)], axis=1)
        pts = (pose @ hom.T).T[:, :3]
        finite = np.isfinite(pts).all(axis=1)
        final_indices = source_indices[finite]
        valid[final_indices] = True
        world[final_indices] = pts[finite].astype(np.float32)
        colors[final_indices] = image[y[final_indices], x[final_indices]].astype(np.uint8)
        return world, colors, valid


def _append_frame_reservoir(
    reservoir: dict[int, list[np.ndarray]],
    frame_id: int,
    arrays: list[np.ndarray],
    cap: int,
    rng: np.random.Generator,
) -> None:
    if arrays[0].shape[0] == 0:
        return
    existing = reservoir.get(int(frame_id))
    if existing is None:
        merged = [np.asarray(value) for value in arrays]
    else:
        merged = [np.concatenate([old, np.asarray(new)], axis=0) for old, new in zip(existing, arrays)]
    if int(cap) > 0 and merged[0].shape[0] > int(cap):
        keep = rng.choice(merged[0].shape[0], size=int(cap), replace=False)
        keep.sort()
        merged = [value[keep] for value in merged]
    reservoir[int(frame_id)] = merged


def _concat_reservoir(reservoir: dict[int, list[np.ndarray]], key_count: int) -> list[np.ndarray]:
    if not reservoir:
        return [np.zeros((0,), dtype=np.float32) for _ in range(key_count)]
    parts = [reservoir[key] for key in sorted(reservoir)]
    out: list[np.ndarray] = []
    for idx in range(key_count):
        out.append(np.concatenate([part[idx] for part in parts], axis=0))
    return out


def _collect_d4rt_da3_pairs(
    *,
    d4rt_rows: Path,
    scene_id: str,
    frame_to_da3: dict[int, int],
    da3_cache: DA3FrameCache,
    min_visibility: float,
    min_confidence: float,
    da3_conf_min: float,
    chunksize: int,
    max_pairs_per_frame: int,
    max_input_rows: int,
    d4rt_uv_width: int,
    d4rt_uv_height: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    usecols = [
        "scene_id",
        "target_frame_id",
        "u_tgt",
        "v_tgt",
        "x_3d",
        "y_3d",
        "z_3d",
        "visibility",
        "confidence",
        "uv_in01",
        "overlap_stitch_applied",
        "geometry_coordinate_mode",
    ]
    reservoir: dict[int, list[np.ndarray]] = {}
    frame_hits: dict[int, int] = {}
    frame_candidate_hits: dict[int, int] = {}
    rows_seen = 0
    rows_scene = 0
    rows_candidate = 0
    rows_da3_valid = 0
    skipped_bad_mode = 0
    uv_raw_min = np.asarray([np.inf, np.inf], dtype=np.float64)
    uv_raw_max = np.asarray([-np.inf, -np.inf], dtype=np.float64)
    uv_da3_min = np.asarray([np.inf, np.inf], dtype=np.float64)
    uv_da3_max = np.asarray([-np.inf, -np.inf], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    target_frames = set(int(value) for value in frame_to_da3)

    for chunk in pd.read_csv(d4rt_rows, usecols=usecols, chunksize=int(chunksize)):
        rows_seen += int(chunk.shape[0])
        if int(max_input_rows) > 0 and rows_seen > int(max_input_rows):
            chunk = chunk.iloc[: max(0, int(chunk.shape[0]) - (rows_seen - int(max_input_rows)))]
        if chunk.empty:
            break
        chunk = chunk[chunk["scene_id"].astype(str) == str(scene_id)].copy()
        rows_scene += int(chunk.shape[0])
        if chunk.empty:
            if int(max_input_rows) > 0 and rows_seen >= int(max_input_rows):
                break
            continue
        mode_ok = chunk["geometry_coordinate_mode"].astype(str).eq("d4rt_overlap_self_stitched_no_final_gt_sim3")
        skipped_bad_mode += int(np.count_nonzero(~mode_ok.to_numpy()))
        uv_bool = chunk["uv_in01"].astype(str).str.lower().isin(["true", "1"])
        stitch_bool = chunk["overlap_stitch_applied"].astype(str).str.lower().isin(["true", "1"])
        frame_ok = chunk["target_frame_id"].astype(np.int64).isin(target_frames)
        numeric = chunk[["u_tgt", "v_tgt", "x_3d", "y_3d", "z_3d", "visibility", "confidence"]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        finite = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
        keep = (
            mode_ok.to_numpy()
            & uv_bool.to_numpy()
            & stitch_bool.to_numpy()
            & frame_ok.to_numpy()
            & finite
            & (numeric["visibility"].to_numpy(dtype=np.float64) >= float(min_visibility))
            & (numeric["confidence"].to_numpy(dtype=np.float64) >= float(min_confidence))
        )
        if not np.any(keep):
            if int(max_input_rows) > 0 and rows_seen >= int(max_input_rows):
                break
            continue
        kept = chunk.loc[keep, ["target_frame_id"]].copy()
        vals = numeric.loc[keep].reset_index(drop=True)
        kept = kept.reset_index(drop=True)
        rows_candidate += int(kept.shape[0])
        uv_raw = vals[["u_tgt", "v_tgt"]].to_numpy(dtype=np.float64)
        uv_raw_min = np.minimum(uv_raw_min, np.nanmin(uv_raw, axis=0))
        uv_raw_max = np.maximum(uv_raw_max, np.nanmax(uv_raw, axis=0))
        xyz = vals[["x_3d", "y_3d", "z_3d"]].to_numpy(dtype=np.float32)
        visibility = vals["visibility"].to_numpy(dtype=np.float32)
        confidence = vals["confidence"].to_numpy(dtype=np.float32)
        frame_ids = kept["target_frame_id"].to_numpy(dtype=np.int64)

        for frame_id in np.unique(frame_ids).tolist():
            frame_mask = frame_ids == int(frame_id)
            frame_candidate_hits[int(frame_id)] = frame_candidate_hits.get(int(frame_id), 0) + int(np.count_nonzero(frame_mask))
            frame = da3_cache.load(int(frame_id))
            da3_h, da3_w = np.asarray(frame["depth"]).shape[:2]
            uv_norm = np.asarray(uv_raw[frame_mask], dtype=np.float64).copy()
            uv_norm[:, 0] /= float(max(int(d4rt_uv_width) - 1, 1))
            uv_norm[:, 1] /= float(max(int(d4rt_uv_height) - 1, 1))
            uv_da3 = uv_norm.copy()
            uv_da3[:, 0] *= float(max(int(da3_w) - 1, 1))
            uv_da3[:, 1] *= float(max(int(da3_h) - 1, 1))
            uv_da3_min = np.minimum(uv_da3_min, np.nanmin(uv_da3, axis=0))
            uv_da3_max = np.maximum(uv_da3_max, np.nanmax(uv_da3, axis=0))
            da3_world, colors, da3_ok = da3_cache.backproject(int(frame_id), uv_da3, conf_min=float(da3_conf_min))
            if not np.any(da3_ok):
                continue
            d4rt_frame_xyz = xyz[frame_mask][da3_ok].astype(np.float32)
            da3_frame_xyz = da3_world[da3_ok].astype(np.float32)
            colors_frame = colors[da3_ok].astype(np.uint8)
            uv_raw_frame = uv_raw[frame_mask][da3_ok].astype(np.float32)
            uv_norm_frame = uv_norm[da3_ok].astype(np.float32)
            uv_da3_frame = uv_da3[da3_ok].astype(np.float32)
            vis_frame = visibility[frame_mask][da3_ok].astype(np.float32)
            conf_frame = confidence[frame_mask][da3_ok].astype(np.float32)
            frame_col = np.full((d4rt_frame_xyz.shape[0],), int(frame_id), dtype=np.int32)
            da3_idx_col = np.full((d4rt_frame_xyz.shape[0],), int(frame_to_da3[int(frame_id)]), dtype=np.int32)
            rows_da3_valid += int(d4rt_frame_xyz.shape[0])
            frame_hits[int(frame_id)] = frame_hits.get(int(frame_id), 0) + int(d4rt_frame_xyz.shape[0])
            _append_frame_reservoir(
                reservoir,
                int(frame_id),
                [
                    d4rt_frame_xyz,
                    da3_frame_xyz,
                    colors_frame,
                    uv_raw_frame,
                    uv_norm_frame,
                    uv_da3_frame,
                    vis_frame,
                    conf_frame,
                    frame_col,
                    da3_idx_col,
                ],
                cap=int(max_pairs_per_frame),
                rng=rng,
            )
        if int(max_input_rows) > 0 and rows_seen >= int(max_input_rows):
            break

    arrays = _concat_reservoir(reservoir, 10)
    names = [
        "d4rt_points_raw",
        "da3_points_at_d4rt_uv",
        "colors_rgb",
        "uv_d4rt_raw_px",
        "uv_norm",
        "uv_da3_px",
        "visibility",
        "confidence",
        "frame_ids",
        "da3_frame_indices",
    ]
    out = {name: value for name, value in zip(names, arrays)}
    info = {
        "d4rt_rows": str(d4rt_rows),
        "rows_seen": int(rows_seen),
        "rows_scene": int(rows_scene),
        "rows_candidate_after_d4rt_filters": int(rows_candidate),
        "rows_with_valid_da3_depth": int(rows_da3_valid),
        "sampled_pair_count": int(out["d4rt_points_raw"].shape[0]),
        "max_pairs_per_frame": int(max_pairs_per_frame),
        "d4rt_uv_source_width": int(d4rt_uv_width),
        "d4rt_uv_source_height": int(d4rt_uv_height),
        "uv_alignment_contract": "canonical uv is normalized [0,1]; u_norm=u_d4rt_px/(d4rt_width-1), v_norm=v_d4rt_px/(d4rt_height-1), then u_da3=u_norm*(da3_width-1), v_da3=v_norm*(da3_height-1)",
        "frame_count_with_candidates": int(len(frame_candidate_hits)),
        "frame_count_with_valid_pairs": int(len(frame_hits)),
        "uv_d4rt_raw_px_min": uv_raw_min.tolist() if np.isfinite(uv_raw_min).all() else None,
        "uv_d4rt_raw_px_max": uv_raw_max.tolist() if np.isfinite(uv_raw_max).all() else None,
        "uv_da3_px_min": uv_da3_min.tolist() if np.isfinite(uv_da3_min).all() else None,
        "uv_da3_px_max": uv_da3_max.tolist() if np.isfinite(uv_da3_max).all() else None,
        "skipped_bad_geometry_coordinate_mode": int(skipped_bad_mode),
        "per_frame_candidate_min": int(min(frame_candidate_hits.values())) if frame_candidate_hits else 0,
        "per_frame_candidate_max": int(max(frame_candidate_hits.values())) if frame_candidate_hits else 0,
        "per_frame_valid_pair_min": int(min(frame_hits.values())) if frame_hits else 0,
        "per_frame_valid_pair_max": int(max(frame_hits.values())) if frame_hits else 0,
    }
    return out, info


def _load_da3_dense_points(
    *,
    manifest: pd.DataFrame,
    da3_cache: DA3FrameCache,
    step: int,
    conf_min: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    raw_candidates = 0
    valid_count = 0
    for row in manifest.itertuples(index=False):
        frame_id = int(row.frame_id)
        frame = da3_cache.load(frame_id)
        depth = np.asarray(frame["depth"], dtype=np.float64)
        conf = np.asarray(frame["conf"], dtype=np.float64)
        image = np.asarray(frame["image"], dtype=np.uint8)
        h, w = depth.shape[:2]
        stride = max(1, int(step))
        yy, xx = np.mgrid[stride // 2 : h : stride, stride // 2 : w : stride]
        xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float64)
        raw_candidates += int(xy.shape[0])
        z = depth[xy[:, 1].astype(np.int64), xy[:, 0].astype(np.int64)]
        score = conf[xy[:, 1].astype(np.int64), xy[:, 0].astype(np.int64)]
        ok0 = np.isfinite(z) & (z > 0.0) & np.isfinite(score) & (score >= float(conf_min))
        if not np.any(ok0):
            continue
        world, colors, ok = da3_cache.backproject(frame_id, xy[ok0], conf_min=float(conf_min))
        if not np.any(ok):
            continue
        pts = world[ok].astype(np.float32)
        rgb = colors[ok].astype(np.uint8)
        valid_count += int(pts.shape[0])
        point_parts.append(pts)
        color_parts.append(rgb)
        frame_parts.append(np.full((pts.shape[0],), frame_id, dtype=np.int32))
    if not point_parts:
        raise RuntimeError("no DA3 dense points survived depth/conf filters")
    points = np.concatenate(point_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    frames = np.concatenate(frame_parts, axis=0)
    idx = _sample_indices(points.shape[0], int(max_points), int(seed))
    info = {
        "dense_grid_step_px": int(step),
        "raw_grid_candidate_count": int(raw_candidates),
        "valid_dense_point_count_before_sample": int(valid_count),
        "sampled_dense_point_count": int(idx.shape[0]),
        "conf_min": float(conf_min),
    }
    return points[idx], colors[idx], frames[idx], info


def _chamfer(source: np.ndarray, target: np.ndarray, sample_count: int, seed: int) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    if source.shape[0] == 0 or target.shape[0] == 0:
        return {"valid": False, "reason": "empty_source_or_target"}
    src_idx = _sample_indices(source.shape[0], int(sample_count), int(seed) + 1)
    tgt_idx = _sample_indices(target.shape[0], int(sample_count), int(seed) + 2)
    src = source[src_idx]
    tgt = target[tgt_idx]
    src_to_tgt, _ = cKDTree(tgt).query(src, k=1)
    tgt_to_src, _ = cKDTree(src).query(tgt, k=1)
    return {
        "valid": True,
        "source_sample_count": int(src.shape[0]),
        "target_sample_count": int(tgt.shape[0]),
        "source_to_target_m": _residual_stats(src_to_tgt),
        "target_to_source_m": _residual_stats(tgt_to_src),
        "chamfer_l2_mean_m": float(0.5 * (np.mean(src_to_tgt) + np.mean(tgt_to_src))),
        "chamfer_l2_squared_mean_m2": float(0.5 * (np.mean(src_to_tgt**2) + np.mean(tgt_to_src**2))),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    scene_id = str(args.scene_id)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(Path(args.da3_manifest), scene_id=scene_id)
    frame_to_da3 = {
        int(row.frame_id): int(row.da3_frame_index)
        for row in manifest.itertuples(index=False)
    }
    da3_cache = DA3FrameCache(Path(args.da3_root), manifest)
    pairs, pair_info = _collect_d4rt_da3_pairs(
        d4rt_rows=Path(args.d4rt_micro_track_rows),
        scene_id=scene_id,
        frame_to_da3=frame_to_da3,
        da3_cache=da3_cache,
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
        da3_conf_min=float(args.da3_conf_min),
        chunksize=int(args.chunksize),
        max_pairs_per_frame=int(args.max_pairs_per_frame),
        max_input_rows=int(args.max_input_rows),
        d4rt_uv_width=int(args.d4rt_uv_width),
        d4rt_uv_height=int(args.d4rt_uv_height),
        seed=int(args.seed),
    )
    if pairs["d4rt_points_raw"].shape[0] < 4:
        raise RuntimeError("not enough D4RT/DA3 correspondence pairs for Sim3")
    fit_idx = _sample_indices(
        pairs["d4rt_points_raw"].shape[0],
        int(args.max_fit_anchors),
        int(args.seed) + 10,
    )
    fit = fit_transform(
        pairs["d4rt_points_raw"][fit_idx],
        pairs["da3_points_at_d4rt_uv"][fit_idx],
        robust_trim_percentile=float(args.robust_trim_percentile),
    )
    if fit is None:
        raise RuntimeError("D4RT->DA3 Sim3 fit returned None")
    d4rt_aligned = apply_sim3_to_xyz(pairs["d4rt_points_raw"], transform=fit).astype(np.float32)
    raw_pair_residual = np.linalg.norm(
        pairs["d4rt_points_raw"].astype(np.float64) - pairs["da3_points_at_d4rt_uv"].astype(np.float64),
        axis=1,
    )
    aligned_pair_residual = np.linalg.norm(
        d4rt_aligned.astype(np.float64) - pairs["da3_points_at_d4rt_uv"].astype(np.float64),
        axis=1,
    )
    fit_residual = np.asarray(fit["residual"], dtype=np.float64)
    dense_points, dense_colors, dense_frames, dense_info = _load_da3_dense_points(
        manifest=manifest,
        da3_cache=da3_cache,
        step=int(args.da3_dense_step),
        conf_min=float(args.da3_conf_min),
        max_points=int(args.viewer_da3_dense_points),
        seed=int(args.seed) + 20,
    )
    viewer_idx = _sample_indices(
        d4rt_aligned.shape[0],
        int(args.viewer_pair_points),
        int(args.seed) + 30,
    )
    npz_path = output_root / "da3_d4rt_sim3_alignment_layers.npz"
    np.savez_compressed(
        npz_path,
        da3_dense_points=dense_points.astype(np.float32),
        da3_dense_colors=dense_colors.astype(np.uint8),
        da3_dense_frame_ids=dense_frames.astype(np.int32),
        da3_correspondence_points=pairs["da3_points_at_d4rt_uv"][viewer_idx].astype(np.float32),
        da3_correspondence_colors=pairs["colors_rgb"][viewer_idx].astype(np.uint8),
        d4rt_raw_points=pairs["d4rt_points_raw"][viewer_idx].astype(np.float32),
        d4rt_raw_colors=pairs["colors_rgb"][viewer_idx].astype(np.uint8),
        d4rt_aligned_points=d4rt_aligned[viewer_idx].astype(np.float32),
        d4rt_aligned_colors=pairs["colors_rgb"][viewer_idx].astype(np.uint8),
        pair_frame_ids=pairs["frame_ids"][viewer_idx].astype(np.int32),
        pair_da3_frame_indices=pairs["da3_frame_indices"][viewer_idx].astype(np.int32),
        pair_uv_d4rt_raw_px=pairs["uv_d4rt_raw_px"][viewer_idx].astype(np.float32),
        pair_uv_norm=pairs["uv_norm"][viewer_idx].astype(np.float32),
        pair_uv_da3_px=pairs["uv_da3_px"][viewer_idx].astype(np.float32),
        pair_visibility=pairs["visibility"][viewer_idx].astype(np.float32),
        pair_confidence=pairs["confidence"][viewer_idx].astype(np.float32),
    )
    chamfer_raw = _chamfer(
        pairs["d4rt_points_raw"],
        pairs["da3_points_at_d4rt_uv"],
        sample_count=int(args.chamfer_sample_count),
        seed=int(args.seed) + 40,
    )
    chamfer_aligned = _chamfer(
        d4rt_aligned,
        pairs["da3_points_at_d4rt_uv"],
        sample_count=int(args.chamfer_sample_count),
        seed=int(args.seed) + 50,
    )
    metrics = {
        "raw_pair_residual_m": _residual_stats(raw_pair_residual),
        "aligned_pair_residual_m": _residual_stats(aligned_pair_residual),
        "fit_anchor_residual_m": _residual_stats(fit_residual),
        "chamfer_raw_d4rt_to_da3_pairs": chamfer_raw,
        "chamfer_aligned_d4rt_to_da3_pairs": chamfer_aligned,
    }
    metrics_rows = [
        {
            "name": "raw_no_sim3",
            "pair_residual_mean_m": metrics["raw_pair_residual_m"]["mean"],
            "pair_residual_p50_m": metrics["raw_pair_residual_m"]["p50"],
            "pair_residual_p90_m": metrics["raw_pair_residual_m"]["p90"],
            "pair_residual_p95_m": metrics["raw_pair_residual_m"]["p95"],
            "chamfer_l2_mean_m": chamfer_raw.get("chamfer_l2_mean_m"),
            "chamfer_l2_squared_mean_m2": chamfer_raw.get("chamfer_l2_squared_mean_m2"),
        },
        {
            "name": "d4rt_to_da3_sim3",
            "pair_residual_mean_m": metrics["aligned_pair_residual_m"]["mean"],
            "pair_residual_p50_m": metrics["aligned_pair_residual_m"]["p50"],
            "pair_residual_p90_m": metrics["aligned_pair_residual_m"]["p90"],
            "pair_residual_p95_m": metrics["aligned_pair_residual_m"]["p95"],
            "chamfer_l2_mean_m": chamfer_aligned.get("chamfer_l2_mean_m"),
            "chamfer_l2_squared_mean_m2": chamfer_aligned.get("chamfer_l2_squared_mean_m2"),
        },
    ]
    csv_path = output_root / "da3_d4rt_sim3_alignment_metrics.csv"
    _write_csv(csv_path, metrics_rows)
    d4rt_summary = _read_json_if_exists(Path(args.d4rt_summary_json))
    da3_log = _parse_da3_log(Path(args.da3_root).with_suffix(Path(args.da3_root).suffix + ".log"))
    if not da3_log.get("exists", False):
        da3_log = _parse_da3_log(Path(str(args.da3_root) + ".log"))
    summary = {
        "phase": "v99_phase10aa_da3_d4rt_sim3_alignment_visual",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scene_id": scene_id,
        "decision": "AUDIT_BUILT_NO_METHOD_CLAIM",
        "output_root": str(output_root),
        "outputs": {
            "summary_json": str(output_root / "summary.json"),
            "layers_npz": str(npz_path),
            "metrics_csv": str(csv_path),
        },
        "contract": {
            "da3_chunk_size": da3_log.get("chunk_size"),
            "da3_overlap": da3_log.get("overlap"),
            "d4rt_applies_overlap_stitch": d4rt_summary.get("d4rt_applies_overlap_stitch"),
            "d4rt_geometry_coordinate_mode": d4rt_summary.get("geometry_coordinate_mode"),
            "d4rt_applies_final_gt_sim3": d4rt_summary.get("d4rt_applies_final_gt_sim3"),
            "cross_model_alignment": "D4RT self-stitched points fitted to DA3 self-stitched backprojected points by same-frame same-pixel Sim3",
            "uses_gt_for_cross_model_alignment": False,
            "uses_future_for_prediction": False,
            "visualization_layers": [
                "DA3 dense",
                "DA3 correspondence",
                "D4RT raw self-stitched",
                "D4RT aligned to DA3 by Sim3",
            ],
        },
        "inputs": {
            "da3_root": str(args.da3_root),
            "da3_manifest": str(args.da3_manifest),
            "d4rt_micro_track_rows": str(args.d4rt_micro_track_rows),
            "d4rt_summary_json": str(args.d4rt_summary_json),
            "da3_log": da3_log,
            "d4rt_summary_selected": {
                "query_chunk_size": d4rt_summary.get("query_chunk_size"),
                "selected_group_count": d4rt_summary.get("selected_group_count"),
                "overlap_stitch_edge_count": d4rt_summary.get("overlap_stitch_edge_count"),
                "required_overlap_stitch_edge_count": d4rt_summary.get("required_overlap_stitch_edge_count"),
                "min_visibility": d4rt_summary.get("min_visibility"),
                "min_confidence": d4rt_summary.get("min_confidence"),
            },
        },
        "manifest": {
            "frame_count": int(manifest.shape[0]),
            "frame_id_min": int(manifest["frame_id"].min()),
            "frame_id_max": int(manifest["frame_id"].max()),
            "stride_values": sorted(int(value) for value in np.unique(np.diff(manifest["frame_id"].to_numpy(dtype=np.int64))).tolist()),
        },
        "filters_and_sampling": {
            "min_visibility": float(args.min_visibility),
            "min_confidence": float(args.min_confidence),
            "da3_conf_min": float(args.da3_conf_min),
            "chunksize": int(args.chunksize),
            "max_input_rows": int(args.max_input_rows),
            "max_pairs_per_frame": int(args.max_pairs_per_frame),
            "d4rt_uv_width": int(args.d4rt_uv_width),
            "d4rt_uv_height": int(args.d4rt_uv_height),
            "max_fit_anchors": int(args.max_fit_anchors),
            "viewer_pair_points": int(args.viewer_pair_points),
            "viewer_da3_dense_points": int(args.viewer_da3_dense_points),
            "da3_dense_step": int(args.da3_dense_step),
            "chamfer_sample_count": int(args.chamfer_sample_count),
            "seed": int(args.seed),
        },
        "pair_collection": pair_info,
        "dense_da3": dense_info,
        "sim3_d4rt_to_da3": {
            "scale": float(fit["scale"]),
            "rotation_det": float(fit.get("rotation_det", np.linalg.det(np.asarray(fit["rotation"], dtype=np.float64)))),
            "translation_norm": float(np.linalg.norm(np.asarray(fit["translation"], dtype=np.float64))),
            "anchor_count": int(fit.get("anchor_count", fit_idx.shape[0])),
            "fit_anchor_count_requested": int(fit_idx.shape[0]),
            "robust_trim_percentile": float(args.robust_trim_percentile),
            "robust_kept_anchors": int(fit.get("robust_kept_anchors", fit_idx.shape[0])),
            "rotation": np.asarray(fit["rotation"], dtype=np.float64),
            "translation": np.asarray(fit["translation"], dtype=np.float64),
        },
        "metrics": metrics,
        "runtime_sec": float(time.time() - t0),
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE)
    parser.add_argument("--da3-root", default=str(DEFAULT_DA3_ROOT))
    parser.add_argument("--da3-manifest", default=str(DEFAULT_DA3_MANIFEST))
    parser.add_argument("--d4rt-micro-track-rows", default=str(DEFAULT_D4RT_ROWS))
    parser.add_argument("--d4rt-summary-json", default=str(DEFAULT_D4RT_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--da3-conf-min", type=float, default=0.0)
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument("--max-input-rows", type=int, default=0)
    parser.add_argument("--max-pairs-per-frame", type=int, default=1200)
    parser.add_argument("--d4rt-uv-width", type=int, default=1296)
    parser.add_argument("--d4rt-uv-height", type=int, default=968)
    parser.add_argument("--max-fit-anchors", type=int, default=120000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--viewer-pair-points", type=int, default=180000)
    parser.add_argument("--viewer-da3-dense-points", type=int, default=220000)
    parser.add_argument("--da3-dense-step", type=int, default=7)
    parser.add_argument("--chamfer-sample-count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=991001)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
