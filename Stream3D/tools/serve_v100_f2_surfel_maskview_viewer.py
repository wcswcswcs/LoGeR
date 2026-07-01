#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
SCANNET_PROCESSED_ROOT = STREAM3D_ROOT / "data" / "scannet" / "processed"
DEFAULT_PHASE2_ROOT = STREAM3D_ROOT / "outputs" / "audit" / "v100_phase2_f2_local_final"
DEFAULT_OUTPUT_ROOT = STREAM3D_ROOT / "outputs" / "audit" / "v100_f2_surfel_maskview_viewer"
DEFAULT_SOURCE_REGISTRY = STREAM3D_ROOT / "outputs" / "audit" / "v95_phase1_physical_source_registry" / "source_container_rows.csv"
DEV_SURFEL_ROOT = STREAM3D_ROOT / "outputs" / "audit" / "v98_phase5_fused_surfel"
HOLDOUT_SURFEL_ROOT = STREAM3D_ROOT / "outputs" / "audit" / "v98_phase13_holdout_phase5_fused_surfel"
DA3_CONTRACT_ROOT = STREAM3D_ROOT / "outputs" / "audit" / "v98_phase1_provider_contract"
DEV_DA3_DENSE_ROOT_BY_SCENE = {
    "scene0011_00": DA3_CONTRACT_ROOT / "da3_streaming_full_scene0011",
    "scene0050_00": DA3_CONTRACT_ROOT / "da3_streaming_full_scene0050",
}
DEV_DA3_DENSE_MANIFEST_BY_SCENE = {
    "scene0011_00": DA3_CONTRACT_ROOT / "da3_streaming_full_scene0011_input" / "frame_manifest_rows.csv",
    "scene0050_00": DA3_CONTRACT_ROOT / "da3_streaming_full_scene0050_input" / "frame_manifest_rows.csv",
}
HOLDOUT_DA3_DENSE_ROOT_BY_SCENE = {
    "scene0011_00": DA3_CONTRACT_ROOT / "da3_streaming_holdout_scene0011",
    "scene0050_00": DA3_CONTRACT_ROOT / "da3_streaming_holdout_scene0050",
}
HOLDOUT_DA3_DENSE_MANIFEST_BY_SCENE = {
    "scene0011_00": DA3_CONTRACT_ROOT / "da3_streaming_holdout_scene0011_input" / "frame_manifest_rows.csv",
    "scene0050_00": DA3_CONTRACT_ROOT / "da3_streaming_holdout_scene0050_input" / "frame_manifest_rows.csv",
}
VARIANT_ID = "F2_v100_chunk32_surfel_maskview_thr018_p2d2_formalized"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _project_source_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) & 0xFFFFFFFF


def _color_from_text(text: str, *, sat: float = 0.72, val: float = 0.95) -> tuple[int, int, int]:
    seed = _stable_seed(text)
    hue = (seed % 4096) / 4096.0
    red, green, blue = colorsys.hsv_to_rgb(hue, sat, val)
    return int(red * 255), int(green * 255), int(blue * 255)


def _colors_for(values: pd.Series, *, salt: str) -> np.ndarray:
    return np.asarray([_color_from_text(f"{salt}:{value}") for value in values.astype(str).tolist()], dtype=np.uint8)


def _score_colors(scores: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(scores, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if vals.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if hi - lo <= 1e-12:
        t = np.zeros_like(vals)
    else:
        t = (vals - lo) / (hi - lo)
    colors = np.stack(
        [
            55.0 + 200.0 * t,
            110.0 + 115.0 * (1.0 - np.abs(t - 0.5) * 2.0),
            230.0 * (1.0 - t) + 35.0 * t,
        ],
        axis=1,
    )
    return np.clip(colors, 0, 255).astype(np.uint8)


def _to_int_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(-1).round().astype(np.int64)


def _read_label(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    import cv2  # type: ignore

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask label: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    label = np.asarray(image, dtype=np.int64)
    if shape_hw is not None and tuple(label.shape[:2]) != tuple(shape_hw):
        label = cv2.resize(label, (int(shape_hw[1]), int(shape_hw[0])), interpolation=cv2.INTER_NEAREST)
    return np.asarray(label, dtype=np.int64)


def _load_mask_path_by_frame(source_registry: Path) -> tuple[dict[tuple[str, str, int], Path], dict[str, Any]]:
    if not source_registry.is_file():
        raise FileNotFoundError(source_registry)
    rows = pd.read_csv(
        source_registry,
        usecols=["scene_id", "split", "frame_id", "mask_path", "uses_gt_for_prediction", "uses_future"],
    )
    out: dict[tuple[str, str, int], Path] = {}
    source_uses_gt = False
    source_uses_future = False
    duplicate_keys = 0
    for row in rows.itertuples(index=False):
        scene = str(row.scene_id)
        split = str(row.split)
        frame = int(row.frame_id)
        raw = str(row.mask_path or "")
        if not scene or not split or not raw:
            continue
        key = (split, scene, frame)
        path = _project_source_path(raw)
        if key in out and out[key] != path:
            duplicate_keys += 1
            continue
        out.setdefault(key, path)
        source_uses_gt = source_uses_gt or str(row.uses_gt_for_prediction).strip().lower() in {"1", "true", "yes", "y"}
        source_uses_future = source_uses_future or str(row.uses_future).strip().lower() in {"1", "true", "yes", "y"}
    diag = {
        "source_registry": _rel(source_registry),
        "source_registry_sha256": _sha256(source_registry),
        "source_registry_rows": int(len(rows)),
        "mask_path_frame_count": int(len(out)),
        "duplicate_frame_mask_path_keys": int(duplicate_keys),
        "source_uses_gt_for_prediction": bool(source_uses_gt),
        "source_uses_future": bool(source_uses_future),
    }
    return out, diag


def _frame_mask_assignment(frame_rows: pd.DataFrame) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    assignments: dict[int, dict[str, Any]] = {}
    duplicate_mask_object_keys = 0
    duplicate_mask_row_count = 0
    if frame_rows.empty:
        return assignments, {
            "selected_mask_count": 0,
            "duplicate_mask_object_keys": 0,
            "duplicate_mask_row_count": 0,
        }
    for mask_id, group in frame_rows.groupby("selected_mask_id_i", sort=True):
        group = group.copy()
        object_count = int(group["mv_object_id"].astype(str).nunique())
        if object_count > 1:
            duplicate_mask_object_keys += 1
        if len(group) > 1:
            duplicate_mask_row_count += int(len(group) - 1)
        group = group.sort_values(["score_f", "mv_object_id"], ascending=[False, True])
        row = group.iloc[0]
        object_id = str(row["mv_object_id"])
        assignments[int(mask_id)] = {
            "mv_object_id": object_id,
            "object_color": _color_from_text(f"object:{object_id}"),
            "mask_color": _color_from_text(f"mask:{int(mask_id)}"),
            "score": _safe_float(row.get("score_f")),
        }
    return assignments, {
        "selected_mask_count": int(len(assignments)),
        "duplicate_mask_object_keys": int(duplicate_mask_object_keys),
        "duplicate_mask_row_count": int(duplicate_mask_row_count),
    }


def _sample_frame(df: pd.DataFrame, max_points: int, *, seed: str) -> tuple[pd.DataFrame, bool]:
    if max_points <= 0 or len(df) <= max_points:
        return df, False
    rng = np.random.default_rng(_stable_seed(seed))
    indices = np.sort(rng.choice(len(df), size=int(max_points), replace=False))
    return df.iloc[indices].copy(), True


def _load_phase2_rows(phase2_root: Path, split: str, scene: str, chunk: str) -> pd.DataFrame:
    rows_path = phase2_root / "mv_object_frame_mask_rows.parquet"
    if not rows_path.is_file():
        raise FileNotFoundError(rows_path)
    rows = pd.read_parquet(rows_path)
    rows = rows[rows["variant_id"].astype(str) == VARIANT_ID].copy()
    if split != "all":
        rows = rows[rows["dataset_split"].astype(str) == split].copy()
    if scene != "all":
        rows = rows[rows["scene_id"].astype(str) == scene].copy()
    if rows.empty:
        raise RuntimeError(f"no v100 rows for split={split} scene={scene} variant={VARIANT_ID}")
    resolved_chunk = "all"
    if chunk != "all":
        chunk_values = sorted(rows["chunk_id"].astype(str).unique().tolist())
        if not chunk_values:
            raise RuntimeError(f"no chunks found for split={split} scene={scene}")
        resolved_chunk = chunk_values[0] if chunk == "first" else chunk
        rows = rows[rows["chunk_id"].astype(str) == resolved_chunk].copy()
        if rows.empty:
            raise RuntimeError(f"no v100 rows for split={split} scene={scene} chunk={resolved_chunk}")
    rows["frame_id_i"] = _to_int_series(rows["frame_id"])
    rows["selected_mask_id_i"] = _to_int_series(rows["selected_mask_id"])
    rows["score_f"] = pd.to_numeric(rows["score"], errors="coerce").fillna(0.0).astype(float)
    rows["object_frame_count_i"] = _to_int_series(rows.get("object_frame_count_post_nms", rows["frame_id"]))
    rows.attrs["resolved_chunk"] = resolved_chunk
    return rows


def _surfel_root_for_split(split: str) -> Path:
    if split == "dev":
        return DEV_SURFEL_ROOT
    if split == "holdout":
        return HOLDOUT_SURFEL_ROOT
    raise ValueError(f"unknown split: {split}")


def _da3_dense_source_for(split: str, scene: str) -> tuple[Path, Path]:
    if split == "dev":
        roots = DEV_DA3_DENSE_ROOT_BY_SCENE
        manifests = DEV_DA3_DENSE_MANIFEST_BY_SCENE
    elif split == "holdout":
        roots = HOLDOUT_DA3_DENSE_ROOT_BY_SCENE
        manifests = HOLDOUT_DA3_DENSE_MANIFEST_BY_SCENE
    else:
        raise ValueError(f"unknown split for DA3 dense source: {split}")
    if scene not in roots or scene not in manifests:
        raise KeyError(f"no DA3 dense source configured for split={split} scene={scene}")
    return roots[scene], manifests[scene]


def _load_da3_dense_chunk_points(
    *,
    split: str,
    scene: str,
    frame_ids: list[int],
    selected_rows: pd.DataFrame,
    mask_path_by_frame: dict[tuple[str, str, int], Path],
    offset: np.ndarray,
    step: int,
    conf_min: float | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    da3_root, manifest_path = _da3_dense_source_for(split, scene)
    poses_path = da3_root / "camera_poses.txt"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not poses_path.is_file():
        raise FileNotFoundError(poses_path)

    manifest = pd.read_csv(manifest_path)
    required = {"da3_frame_index", "frame_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{manifest_path} missing columns: {sorted(missing)}")
    if "scene_id" in manifest.columns:
        manifest = manifest[manifest["scene_id"].astype(str) == scene].copy()
    if "split" in manifest.columns:
        manifest = manifest[manifest["split"].astype(str) == split].copy()
    manifest["frame_id_i"] = _to_int_series(manifest["frame_id"])
    manifest["da3_frame_index_i"] = _to_int_series(manifest["da3_frame_index"])
    requested_frames = sorted({int(v) for v in frame_ids})
    selected = manifest[manifest["frame_id_i"].isin(requested_frames)].copy()
    selected = selected.sort_values("da3_frame_index_i").reset_index(drop=True)
    found_frames = sorted(selected["frame_id_i"].astype(int).unique().tolist())
    missing_frames = sorted(set(requested_frames) - set(found_frames))
    if selected.empty:
        raise RuntimeError(f"no DA3 manifest rows for split={split} scene={scene} requested_frames={requested_frames}")

    poses_da3 = np.loadtxt(poses_path).reshape(-1, 4, 4)
    step_i = max(1, int(step))
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    mask_object_points: list[np.ndarray] = []
    mask_object_colors: list[np.ndarray] = []
    mask_id_colors: list[np.ndarray] = []
    frame_diags: list[dict[str, Any]] = []
    total_grid_pixels = 0
    valid_depth_points = 0
    confidence_filtered_points = 0
    finite_world_points = 0
    mask_da3_points = 0
    mask_label_hit_points = 0
    selected_mask_raster_area_pixels = 0
    missing_mask_rasters = 0
    resized_mask_rasters = 0
    duplicate_mask_object_keys = 0
    duplicate_mask_row_count = 0
    missing_npz: list[str] = []
    skipped_pose_indices: list[int] = []

    for row in selected.itertuples(index=False):
        da3_idx = int(row.da3_frame_index_i)
        frame_id = int(row.frame_id_i)
        npz_path = da3_root / "results_output" / f"frame_{da3_idx}.npz"
        frame_selected_rows = selected_rows[_to_int_series(selected_rows["frame_id"]) == frame_id].copy()
        assignments, assignment_diag = _frame_mask_assignment(frame_selected_rows)
        duplicate_mask_object_keys += int(assignment_diag["duplicate_mask_object_keys"])
        duplicate_mask_row_count += int(assignment_diag["duplicate_mask_row_count"])
        if da3_idx >= poses_da3.shape[0]:
            skipped_pose_indices.append(da3_idx)
            continue
        if not npz_path.is_file():
            missing_npz.append(_rel(npz_path))
            continue

        with np.load(npz_path) as payload:
            image = np.asarray(payload["image"], dtype=np.uint8)
            depth = np.asarray(payload["depth"], dtype=np.float64)
            conf = np.asarray(payload["conf"], dtype=np.float64) if "conf" in payload.files else None
            intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)

        h, w = depth.shape
        yy, xx = np.mgrid[0:h:step_i, 0:w:step_i]
        z = depth[yy, xx].reshape(-1)
        valid = np.isfinite(z) & (z > 0.0)
        depth_valid_count = int(np.count_nonzero(valid))
        if conf_min is not None:
            score = conf[yy, xx].reshape(-1) if conf is not None else np.ones_like(z)
            valid = valid & np.isfinite(score) & (score >= float(conf_min))
        confidence_kept_count = int(np.count_nonzero(valid))
        total_grid_pixels += int(z.size)
        valid_depth_points += depth_valid_count
        confidence_filtered_points += max(0, depth_valid_count - confidence_kept_count)
        if not np.any(valid):
            frame_diags.append(
                {
                    "da3_frame_index": da3_idx,
                    "frame_id": frame_id,
                    "npz": _rel(npz_path),
                    "height": int(h),
                    "width": int(w),
                    "grid_pixels_after_step": int(z.size),
                    "valid_depth_points": depth_valid_count,
                    "rendered_points": 0,
                }
            )
            continue

        pix = np.stack(
            [
                xx.reshape(-1).astype(np.float64),
                yy.reshape(-1).astype(np.float64),
                np.ones(xx.size, dtype=np.float64),
            ],
            axis=0,
        )[:, valid]
        rays = np.linalg.inv(intrinsics) @ pix
        cam = rays.T * z[valid, None]
        hom = np.concatenate([cam, np.ones((cam.shape[0], 1), dtype=np.float64)], axis=1)
        world = (poses_da3[da3_idx] @ hom.T).T[:, :3]
        rgb = image[yy.reshape(-1)[valid], xx.reshape(-1)[valid]]
        finite = np.isfinite(world).all(axis=1)
        rendered = int(np.count_nonzero(finite))
        frame_mask_da3_points = 0
        frame_mask_label_hit_points = 0
        frame_selected_mask_raster_area_pixels = 0
        mask_path = mask_path_by_frame.get((split, scene, frame_id))
        mask_path_rel = _rel(mask_path) if mask_path is not None else ""
        mask_raster_missing = bool(mask_path is None or not mask_path.is_file())
        mask_raster_resized = False
        selected_mask_ids = sorted(assignments)
        finite_world_points += rendered
        if rendered > 0:
            rendered_world = (world[finite].astype(np.float32) + offset.astype(np.float32)[None, :]).astype(np.float32)
            points.append(rendered_world)
            colors.append(rgb[finite].astype(np.uint8))
            if assignments and not mask_raster_missing:
                label_raw = _read_label(mask_path)
                mask_raster_resized = tuple(label_raw.shape[:2]) != (int(h), int(w))
                label = _read_label(mask_path, shape_hw=(int(h), int(w)))
                if mask_raster_resized:
                    resized_mask_rasters += 1
                max_label = int(max(int(np.nanmax(label)), max(selected_mask_ids)))
                selected_lut = np.zeros(max_label + 1, dtype=bool)
                object_color_lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
                mask_color_lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
                for mask_id, payload in assignments.items():
                    if 0 <= mask_id <= max_label:
                        selected_lut[mask_id] = True
                        object_color_lut[mask_id] = np.asarray(payload["object_color"], dtype=np.uint8)
                        mask_color_lut[mask_id] = np.asarray(payload["mask_color"], dtype=np.uint8)
                        frame_selected_mask_raster_area_pixels += int(np.count_nonzero(label == mask_id))
                label_values = label[yy.reshape(-1)[valid], xx.reshape(-1)[valid]][finite]
                in_range = (label_values >= 0) & (label_values <= max_label)
                selected_hit = np.zeros(label_values.shape, dtype=bool)
                selected_hit[in_range] = selected_lut[label_values[in_range]]
                frame_mask_label_hit_points = int(np.count_nonzero(selected_hit))
                if frame_mask_label_hit_points > 0:
                    hit_labels = label_values[selected_hit].astype(np.int64, copy=False)
                    mask_object_points.append(rendered_world[selected_hit])
                    mask_object_colors.append(object_color_lut[hit_labels])
                    mask_id_colors.append(mask_color_lut[hit_labels])
                    frame_mask_da3_points = frame_mask_label_hit_points
            elif assignments and mask_raster_missing:
                missing_mask_rasters += 1
        frame_diags.append(
            {
                "da3_frame_index": da3_idx,
                "frame_id": frame_id,
                "npz": _rel(npz_path),
                "mask_raster": mask_path_rel,
                "mask_raster_missing": mask_raster_missing,
                "mask_raster_resized_to_da3_shape": mask_raster_resized,
                "height": int(h),
                "width": int(w),
                "grid_pixels_after_step": int(z.size),
                "valid_depth_points": depth_valid_count,
                "confidence_kept_points": confidence_kept_count,
                "rendered_points": rendered,
                "selected_object_mask_rows": int(len(frame_selected_rows)),
                "selected_mask_count": int(assignment_diag["selected_mask_count"]),
                "selected_mask_raster_area_pixels_da3_shape": int(frame_selected_mask_raster_area_pixels),
                "mask_label_hit_points": int(frame_mask_label_hit_points),
                "mask_da3_points": int(frame_mask_da3_points),
            }
        )
        mask_da3_points += frame_mask_da3_points
        mask_label_hit_points += frame_mask_label_hit_points
        selected_mask_raster_area_pixels += frame_selected_mask_raster_area_pixels

    if not points:
        raise RuntimeError(f"no dense DA3 points reconstructed for split={split} scene={scene}")

    arrays = {
        "dense_points": np.concatenate(points, axis=0),
        "dense_rgb_colors": np.concatenate(colors, axis=0),
        "mask_object_points": (
            np.concatenate(mask_object_points, axis=0) if mask_object_points else np.zeros((0, 3), dtype=np.float32)
        ),
        "mask_object_colors": (
            np.concatenate(mask_object_colors, axis=0) if mask_object_colors else np.zeros((0, 3), dtype=np.uint8)
        ),
        "mask_id_colors": np.concatenate(mask_id_colors, axis=0) if mask_id_colors else np.zeros((0, 3), dtype=np.uint8),
    }
    diag = {
        "split": split,
        "scene_id": scene,
        "da3_root": _rel(da3_root),
        "manifest": _rel(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "camera_poses": _rel(poses_path),
        "camera_poses_sha256": _sha256(poses_path),
        "requested_frame_count": int(len(requested_frames)),
        "manifest_frame_count": int(len(selected)),
        "rendered_frame_count": int(len(frame_diags)),
        "requested_frames": requested_frames,
        "missing_manifest_frames": missing_frames,
        "missing_npz": missing_npz,
        "skipped_pose_indices": skipped_pose_indices,
        "dense_step": step_i,
        "stride_sampling_enabled": bool(step_i != 1),
        "confidence_filter_min": conf_min,
        "confidence_filter_enabled": bool(conf_min is not None),
        "grid_pixels_after_step": int(total_grid_pixels),
        "valid_depth_points": int(valid_depth_points),
        "confidence_filtered_points": int(confidence_filtered_points),
        "rendered_points": int(finite_world_points),
        "mask_da3_points": int(mask_da3_points),
        "mask_label_hit_points": int(mask_label_hit_points),
        "selected_mask_raster_area_pixels_da3_shape": int(selected_mask_raster_area_pixels),
        "missing_mask_raster_count": int(missing_mask_rasters),
        "resized_mask_raster_count": int(resized_mask_rasters),
        "duplicate_mask_object_keys": int(duplicate_mask_object_keys),
        "duplicate_mask_row_count": int(duplicate_mask_row_count),
        "frame_diags": frame_diags,
    }
    return arrays, diag


def _load_gt_dense_chunk_points(
    *,
    split: str,
    scene: str,
    frame_ids: list[int],
    selected_rows: pd.DataFrame,
    mask_path_by_frame: dict[tuple[str, str, int], Path],
    offset: np.ndarray,
    step: int,
    depth_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import cv2  # type: ignore

    scene_root = SCANNET_PROCESSED_ROOT / scene
    intrinsic_path = scene_root / "intrinsic" / "intrinsic_depth.txt"
    if not intrinsic_path.is_file():
        raise FileNotFoundError(intrinsic_path)
    intrinsic = np.loadtxt(intrinsic_path).reshape(4, 4)[:3, :3].astype(np.float64)
    inv_intrinsic = np.linalg.inv(intrinsic)
    step_i = max(1, int(step))
    requested_frames = sorted({int(v) for v in frame_ids})

    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    mask_object_points: list[np.ndarray] = []
    mask_object_colors: list[np.ndarray] = []
    mask_id_colors: list[np.ndarray] = []
    frame_diags: list[dict[str, Any]] = []

    total_grid_pixels = 0
    valid_depth_points = 0
    finite_world_points = 0
    mask_gt_points = 0
    mask_label_hit_points = 0
    selected_mask_raster_area_pixels = 0
    missing_mask_rasters = 0
    resized_mask_rasters = 0
    duplicate_mask_object_keys = 0
    duplicate_mask_row_count = 0
    missing_depth_frames: list[int] = []
    missing_pose_frames: list[int] = []
    invalid_pose_frames: list[int] = []
    missing_color_frames: list[int] = []

    for frame_id in requested_frames:
        depth_path = scene_root / "depth" / f"{frame_id}.png"
        pose_path = scene_root / "pose" / f"{frame_id}.txt"
        color_path = scene_root / "color" / f"{frame_id}.jpg"
        frame_selected_rows = selected_rows[_to_int_series(selected_rows["frame_id"]) == frame_id].copy()
        assignments, assignment_diag = _frame_mask_assignment(frame_selected_rows)
        duplicate_mask_object_keys += int(assignment_diag["duplicate_mask_object_keys"])
        duplicate_mask_row_count += int(assignment_diag["duplicate_mask_row_count"])

        if not depth_path.is_file():
            missing_depth_frames.append(frame_id)
            continue
        if not pose_path.is_file():
            missing_pose_frames.append(frame_id)
            continue
        pose = np.loadtxt(pose_path).reshape(4, 4).astype(np.float64)
        if not np.isfinite(pose).all():
            invalid_pose_frames.append(frame_id)
            continue
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            missing_depth_frames.append(frame_id)
            continue
        depth = np.asarray(depth_raw, dtype=np.float64) / float(depth_scale)
        h, w = depth.shape[:2]
        yy, xx = np.mgrid[0:h:step_i, 0:w:step_i]
        z = depth[yy, xx].reshape(-1)
        valid = np.isfinite(z) & (z > 0.0)
        depth_valid_count = int(np.count_nonzero(valid))
        total_grid_pixels += int(z.size)
        valid_depth_points += depth_valid_count
        if not np.any(valid):
            frame_diags.append(
                {
                    "frame_id": frame_id,
                    "depth": _rel(depth_path),
                    "pose": _rel(pose_path),
                    "height": int(h),
                    "width": int(w),
                    "grid_pixels_after_step": int(z.size),
                    "valid_depth_points": depth_valid_count,
                    "rendered_points": 0,
                    "selected_object_mask_rows": int(len(frame_selected_rows)),
                    "selected_mask_count": int(assignment_diag["selected_mask_count"]),
                    "mask_gt_points": 0,
                }
            )
            continue

        pix = np.stack(
            [
                xx.reshape(-1).astype(np.float64),
                yy.reshape(-1).astype(np.float64),
                np.ones(xx.size, dtype=np.float64),
            ],
            axis=0,
        )[:, valid]
        rays = inv_intrinsic @ pix
        cam = rays.T * z[valid, None]
        hom = np.concatenate([cam, np.ones((cam.shape[0], 1), dtype=np.float64)], axis=1)
        world = (pose @ hom.T).T[:, :3]
        finite = np.isfinite(world).all(axis=1)
        rendered = int(np.count_nonzero(finite))
        finite_world_points += rendered

        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_bgr is None:
            missing_color_frames.append(frame_id)
            rgb_full = np.zeros((h, w, 3), dtype=np.uint8)
            norm = np.clip(depth / max(float(np.nanpercentile(depth[depth > 0], 95)) if np.any(depth > 0) else 1.0, 1e-6), 0, 1)
            rgb_full[..., :] = (255.0 * (1.0 - norm[..., None])).astype(np.uint8)
        else:
            rgb_full = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            if tuple(rgb_full.shape[:2]) != (int(h), int(w)):
                rgb_full = cv2.resize(rgb_full, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)

        frame_mask_gt_points = 0
        frame_mask_label_hit_points = 0
        frame_selected_mask_raster_area_pixels = 0
        mask_path = mask_path_by_frame.get((split, scene, frame_id))
        mask_path_rel = _rel(mask_path) if mask_path is not None else ""
        mask_raster_missing = bool(mask_path is None or not mask_path.is_file())
        mask_raster_resized = False
        selected_mask_ids = sorted(assignments)
        if rendered > 0:
            rendered_world = (world[finite].astype(np.float32) + offset.astype(np.float32)[None, :]).astype(np.float32)
            rgb = rgb_full[yy.reshape(-1)[valid], xx.reshape(-1)[valid]][finite]
            points.append(rendered_world)
            colors.append(rgb.astype(np.uint8))
            if assignments and not mask_raster_missing:
                label_raw = _read_label(mask_path)
                mask_raster_resized = tuple(label_raw.shape[:2]) != (int(h), int(w))
                label = _read_label(mask_path, shape_hw=(int(h), int(w)))
                if mask_raster_resized:
                    resized_mask_rasters += 1
                max_label = int(max(int(np.nanmax(label)), max(selected_mask_ids)))
                selected_lut = np.zeros(max_label + 1, dtype=bool)
                object_color_lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
                mask_color_lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
                for mask_id, payload in assignments.items():
                    if 0 <= mask_id <= max_label:
                        selected_lut[mask_id] = True
                        object_color_lut[mask_id] = np.asarray(payload["object_color"], dtype=np.uint8)
                        mask_color_lut[mask_id] = np.asarray(payload["mask_color"], dtype=np.uint8)
                        frame_selected_mask_raster_area_pixels += int(np.count_nonzero(label == mask_id))
                label_values = label[yy.reshape(-1)[valid], xx.reshape(-1)[valid]][finite]
                in_range = (label_values >= 0) & (label_values <= max_label)
                selected_hit = np.zeros(label_values.shape, dtype=bool)
                selected_hit[in_range] = selected_lut[label_values[in_range]]
                frame_mask_label_hit_points = int(np.count_nonzero(selected_hit))
                if frame_mask_label_hit_points > 0:
                    hit_labels = label_values[selected_hit].astype(np.int64, copy=False)
                    mask_object_points.append(rendered_world[selected_hit])
                    mask_object_colors.append(object_color_lut[hit_labels])
                    mask_id_colors.append(mask_color_lut[hit_labels])
                    frame_mask_gt_points = frame_mask_label_hit_points
            elif assignments and mask_raster_missing:
                missing_mask_rasters += 1

        frame_diags.append(
            {
                "frame_id": frame_id,
                "depth": _rel(depth_path),
                "pose": _rel(pose_path),
                "color": _rel(color_path),
                "mask_raster": mask_path_rel,
                "mask_raster_missing": mask_raster_missing,
                "mask_raster_resized_to_gt_depth_shape": mask_raster_resized,
                "height": int(h),
                "width": int(w),
                "grid_pixels_after_step": int(z.size),
                "valid_depth_points": depth_valid_count,
                "rendered_points": rendered,
                "selected_object_mask_rows": int(len(frame_selected_rows)),
                "selected_mask_count": int(assignment_diag["selected_mask_count"]),
                "selected_mask_raster_area_pixels_gt_shape": int(frame_selected_mask_raster_area_pixels),
                "mask_label_hit_points": int(frame_mask_label_hit_points),
                "mask_gt_points": int(frame_mask_gt_points),
            }
        )
        mask_gt_points += frame_mask_gt_points
        mask_label_hit_points += frame_mask_label_hit_points
        selected_mask_raster_area_pixels += frame_selected_mask_raster_area_pixels

    if not points:
        raise RuntimeError(f"no GT depth/pose points reconstructed for split={split} scene={scene}")

    arrays = {
        "dense_points": np.concatenate(points, axis=0),
        "dense_rgb_colors": np.concatenate(colors, axis=0),
        "mask_object_points": (
            np.concatenate(mask_object_points, axis=0) if mask_object_points else np.zeros((0, 3), dtype=np.float32)
        ),
        "mask_object_colors": (
            np.concatenate(mask_object_colors, axis=0) if mask_object_colors else np.zeros((0, 3), dtype=np.uint8)
        ),
        "mask_id_colors": np.concatenate(mask_id_colors, axis=0) if mask_id_colors else np.zeros((0, 3), dtype=np.uint8),
    }
    diag = {
        "split": split,
        "scene_id": scene,
        "gt_scene_root": _rel(scene_root),
        "intrinsic_depth": _rel(intrinsic_path),
        "intrinsic_depth_sha256": _sha256(intrinsic_path),
        "depth_scale": float(depth_scale),
        "requested_frame_count": int(len(requested_frames)),
        "rendered_frame_count": int(len(frame_diags)),
        "requested_frames": requested_frames,
        "missing_depth_frames": missing_depth_frames,
        "missing_pose_frames": missing_pose_frames,
        "invalid_pose_frames": invalid_pose_frames,
        "missing_color_frames": missing_color_frames,
        "dense_step": step_i,
        "stride_sampling_enabled": bool(step_i != 1),
        "grid_pixels_after_step": int(total_grid_pixels),
        "valid_depth_points": int(valid_depth_points),
        "rendered_points": int(finite_world_points),
        "mask_gt_points": int(mask_gt_points),
        "mask_label_hit_points": int(mask_label_hit_points),
        "selected_mask_raster_area_pixels_gt_shape": int(selected_mask_raster_area_pixels),
        "missing_mask_raster_count": int(missing_mask_rasters),
        "resized_mask_raster_count": int(resized_mask_rasters),
        "duplicate_mask_object_keys": int(duplicate_mask_object_keys),
        "duplicate_mask_row_count": int(duplicate_mask_row_count),
        "frame_diags": frame_diags,
    }
    return arrays, diag


def _load_surfel_xyz_for_split(split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = _surfel_root_for_split(split)
    obs_path = root / "surfel_observation_rows.csv"
    fused_path = root / "fused_surfel_rows.csv"
    if not obs_path.is_file():
        raise FileNotFoundError(obs_path)
    if not fused_path.is_file():
        raise FileNotFoundError(fused_path)

    obs = pd.read_csv(
        obs_path,
        usecols=[
            "surfel_id",
            "scene_id",
            "frame_id",
            "mask_ids_covering",
            "provider_confidence",
            "projection_valid",
        ],
    )
    obs["projection_valid_bool"] = obs["projection_valid"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    obs["frame_id_i"] = _to_int_series(obs["frame_id"])
    obs["selected_mask_id_i"] = _to_int_series(obs["mask_ids_covering"])
    obs["provider_confidence_f"] = pd.to_numeric(obs["provider_confidence"], errors="coerce").fillna(0.0).astype(float)

    fused = pd.read_csv(
        fused_path,
        usecols=[
            "surfel_id",
            "scene_id",
            "xyz_x",
            "xyz_y",
            "xyz_z",
            "observation_count",
            "observed_frame_count",
            "mean_confidence",
            "surfel_valid",
        ],
    )
    fused["surfel_valid_bool"] = fused["surfel_valid"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    fused = fused[fused["surfel_valid_bool"]].copy()
    for col in ["xyz_x", "xyz_y", "xyz_z", "observation_count", "observed_frame_count", "mean_confidence"]:
        fused[col] = pd.to_numeric(fused[col], errors="coerce")
    fused = fused[np.isfinite(fused[["xyz_x", "xyz_y", "xyz_z"]]).all(axis=1)].copy()

    joined = obs.merge(
        fused.drop(columns=["scene_id"]),
        on="surfel_id",
        how="inner",
        validate="many_to_one",
    )
    out = joined[
        [
            "scene_id",
            "frame_id_i",
            "selected_mask_id_i",
            "surfel_id",
            "projection_valid_bool",
            "provider_confidence_f",
            "xyz_x",
            "xyz_y",
            "xyz_z",
            "observation_count",
            "observed_frame_count",
            "mean_confidence",
        ]
    ].copy()
    diag = {
        "split": split,
        "surfel_root": _rel(root),
        "surfel_observation_rows": _rel(obs_path),
        "surfel_observation_rows_sha256": _sha256(obs_path),
        "fused_surfel_rows": _rel(fused_path),
        "fused_surfel_rows_sha256": _sha256(fused_path),
        "observation_rows": int(len(obs)),
        "projection_valid_observation_rows": int(obs["projection_valid_bool"].sum()),
        "valid_fused_surfel_rows": int(len(fused)),
        "joined_observation_xyz_rows": int(len(out)),
    }
    return out, diag


def _load_metrics(phase2_root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    scene_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    scene_path = phase2_root / "mv_metric_scene_fragmented_rows.csv"
    window_path = phase2_root / "mv_metric_window_rows.csv"
    if scene_path.is_file():
        scene_df = pd.read_csv(scene_path)
        for _, row in scene_df.iterrows():
            scene_metrics[(str(row.get("dataset_split")), str(row.get("scene_id")))] = {
                "MV_AP_scene": _safe_float(row.get("MV_AP_scene")),
                "MV_AP50_scene": _safe_float(row.get("MV_AP50_scene")),
                "pred_object_count_scene": _safe_int(row.get("pred_object_count")),
                "gt_object_count_scene": _safe_int(row.get("gt_object_count")),
                "frame_count_scene": _safe_int(row.get("frame_count")),
            }
    if window_path.is_file():
        window_df = pd.read_csv(window_path)
        local_rows = window_df[window_df["metric_scope"].astype(str) == "local_window_gt_projection_chunk32"]
        for _, row in local_rows.iterrows():
            key = (str(row.get("dataset_split")), str(row.get("scene_id")))
            scene_metrics.setdefault(key, {}).update(
                {
                    "MV_AP_window": _safe_float(row.get("MV_AP_window")),
                    "MV_AP50_window": _safe_float(row.get("MV_AP50_window")),
                    "pred_object_count_window": _safe_int(row.get("pred_object_count")),
                    "gt_object_count_window": _safe_int(row.get("gt_object_count")),
                    "frame_count_window": _safe_int(row.get("frame_count")),
                }
            )
        aggregate: dict[str, dict[str, Any]] = {}
        agg_rows = window_df[window_df["schema_version"].astype(str).str.endswith("metric_aggregate_row_v1")]
        for _, row in agg_rows.iterrows():
            aggregate[str(row.get("dataset_split"))] = {
                "MV_AP_window": _safe_float(row.get("MV_AP_window")),
                "MV_AP50_window": _safe_float(row.get("MV_AP50_window")),
                "MV_AP_scene_fragmented": _safe_float(row.get("MV_AP_scene")),
                "MV_AP50_scene_fragmented": _safe_float(row.get("MV_AP50_scene")),
                "same_frame_collision_count": _safe_int(row.get("same_frame_collision_count")),
                "pixel_collision_rate": _safe_float(row.get("pixel_collision_rate")),
                "missing_mask_raster_count": _safe_int(row.get("missing_mask_raster_count")),
            }
        return scene_metrics, aggregate
    return scene_metrics, {}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _safe_int(value: Any) -> int | None:
    try:
        out = int(round(float(value)))
    except Exception:
        return None
    return out


def _points_from(df: pd.DataFrame, offsets: dict[str, np.ndarray]) -> np.ndarray:
    if df.empty:
        return np.zeros((0, 3), dtype=np.float32)
    base = df[["xyz_x", "xyz_y", "xyz_z"]].to_numpy(dtype=np.float32)
    scene_keys = (df["dataset_split"].astype(str) + ":" + df["scene_id"].astype(str)).tolist()
    off = np.asarray([offsets[key] for key in scene_keys], dtype=np.float32)
    return np.asarray(base + off, dtype=np.float32)


def _make_offsets(scene_keys: list[str], spacing: float) -> dict[str, np.ndarray]:
    offsets: dict[str, np.ndarray] = {}
    cols = max(1, int(np.ceil(np.sqrt(len(scene_keys)))))
    for idx, key in enumerate(scene_keys):
        row = idx // cols
        col = idx % cols
        offsets[key] = np.asarray([float(col) * spacing, float(row) * spacing, 0.0], dtype=np.float32)
    return offsets


def _confidence_colors(confidence: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(confidence, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if vals.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo = float(np.nanpercentile(vals, 5))
    hi = float(np.nanpercentile(vals, 95))
    if hi - lo <= 1e-12:
        t = np.zeros_like(vals)
    else:
        t = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
    colors = np.stack(
        [
            70.0 + 120.0 * t,
            120.0 + 110.0 * t,
            125.0 + 55.0 * (1.0 - t),
        ],
        axis=1,
    )
    return np.clip(colors, 0, 255).astype(np.uint8)


def _build_viewer_payload(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    phase2_root = Path(args.phase2_root)
    v100_rows = _load_phase2_rows(phase2_root, args.split, args.scene, args.chunk)
    resolved_chunk = str(v100_rows.attrs.get("resolved_chunk", "all"))
    scene_metrics, aggregate_metrics = _load_metrics(phase2_root)
    mask_path_by_frame, mask_source_diag = _load_mask_path_by_frame(Path(args.source_registry))

    merged_parts: list[pd.DataFrame] = []
    da3_geometry_parts: list[pd.DataFrame] = []
    split_diags: list[dict[str, Any]] = []
    missing_key_count = 0
    for split in sorted(v100_rows["dataset_split"].astype(str).unique()):
        surfel_xyz, diag = _load_surfel_xyz_for_split(split)
        surfel_xyz = surfel_xyz.copy()
        surfel_xyz["dataset_split"] = split
        selected = v100_rows[v100_rows["dataset_split"].astype(str) == split].copy()
        chunk_frames = selected[["scene_id", "frame_id_i"]].drop_duplicates()
        da3_geometry = surfel_xyz.merge(
            chunk_frames,
            on=["scene_id", "frame_id_i"],
            how="inner",
        )
        da3_geometry["chunk_id"] = resolved_chunk
        da3_geometry_parts.append(da3_geometry)
        keys = selected[["dataset_split", "scene_id", "frame_id_i", "selected_mask_id_i"]].drop_duplicates()
        surfel_keys = surfel_xyz[["dataset_split", "scene_id", "frame_id_i", "selected_mask_id_i"]].drop_duplicates()
        key_check = keys.merge(
            surfel_keys,
            on=["dataset_split", "scene_id", "frame_id_i", "selected_mask_id_i"],
            how="left",
            indicator=True,
        )
        split_missing = int((key_check["_merge"] == "left_only").sum())
        missing_key_count += split_missing
        diag["requested_selected_mask_keys"] = int(len(keys))
        diag["missing_selected_mask_keys"] = split_missing
        merged = selected.merge(
            surfel_xyz,
            on=["dataset_split", "scene_id", "frame_id_i", "selected_mask_id_i"],
            how="left",
        )
        merged = merged[merged["surfel_id"].notna()].copy()
        merged_parts.append(merged)
        diag["v100_frame_mask_rows"] = int(len(selected))
        diag["joined_v100_observation_rows"] = int(len(merged))
        diag["da3_chunk_all_geometry_observation_rows"] = int(len(da3_geometry))
        split_diags.append(diag)

    if not merged_parts:
        raise RuntimeError("no joined surfel rows available for selected v100 rows")
    observations = pd.concat(merged_parts, ignore_index=True)
    observations = observations[np.isfinite(observations[["xyz_x", "xyz_y", "xyz_z"]]).all(axis=1)].copy()
    if observations.empty:
        raise RuntimeError("joined v100/surfel rows have no finite xyz points")
    da3_geometry_all = pd.concat(da3_geometry_parts, ignore_index=True) if da3_geometry_parts else pd.DataFrame()
    da3_geometry_all = da3_geometry_all[np.isfinite(da3_geometry_all[["xyz_x", "xyz_y", "xyz_z"]]).all(axis=1)].copy()
    if da3_geometry_all.empty:
        raise RuntimeError("DA3 chunk geometry layer has no finite xyz points")

    unique = observations.drop_duplicates(["dataset_split", "scene_id", "mv_object_id", "surfel_id"]).copy()
    unique_sampled, aggregate_sampled = _sample_frame(unique, int(args.max_aggregate_points), seed="v100_f2_unique")
    obs_sampled, obs_density_sampled = _sample_frame(observations, int(args.max_observation_points), seed="v100_f2_observations")
    scene_keys = sorted((unique["dataset_split"].astype(str) + ":" + unique["scene_id"].astype(str)).unique().tolist())
    offsets = _make_offsets(scene_keys, float(args.scene_spacing))

    da3_sampled, da3_sampled_flag = _sample_frame(
        da3_geometry_all,
        int(args.max_da3_geometry_points),
        seed="v100_da3_chunk_all_geometry",
    )
    da3_points = _points_from(da3_sampled, offsets)

    gt_points_parts: list[np.ndarray] = []
    gt_color_parts: list[np.ndarray] = []
    mask_object_points_parts: list[np.ndarray] = []
    mask_object_color_parts: list[np.ndarray] = []
    mask_id_color_parts: list[np.ndarray] = []
    gt_diags: list[dict[str, Any]] = []
    for (split, scene), source in v100_rows.groupby(["dataset_split", "scene_id"], sort=True):
        key = f"{split}:{scene}"
        frame_ids = sorted(_to_int_series(source["frame_id"]).astype(int).unique().tolist())
        gt_arrays, gt_diag = _load_gt_dense_chunk_points(
            split=str(split),
            scene=str(scene),
            frame_ids=frame_ids,
            selected_rows=source.copy(),
            mask_path_by_frame=mask_path_by_frame,
            offset=offsets[key],
            step=int(args.gt_dense_step),
            depth_scale=float(args.gt_depth_scale),
        )
        gt_points_parts.append(gt_arrays["dense_points"])
        gt_color_parts.append(gt_arrays["dense_rgb_colors"])
        mask_object_points_parts.append(gt_arrays["mask_object_points"])
        mask_object_color_parts.append(gt_arrays["mask_object_colors"])
        mask_id_color_parts.append(gt_arrays["mask_id_colors"])
        gt_diags.append(gt_diag)
    dense_gt_points = np.concatenate(gt_points_parts, axis=0) if gt_points_parts else np.zeros((0, 3), dtype=np.float32)
    dense_gt_colors = np.concatenate(gt_color_parts, axis=0) if gt_color_parts else np.zeros((0, 3), dtype=np.uint8)
    mask_object_points = (
        np.concatenate(mask_object_points_parts, axis=0) if mask_object_points_parts else np.zeros((0, 3), dtype=np.float32)
    )
    mask_object_colors = (
        np.concatenate(mask_object_color_parts, axis=0) if mask_object_color_parts else np.zeros((0, 3), dtype=np.uint8)
    )
    mask_id_colors = np.concatenate(mask_id_color_parts, axis=0) if mask_id_color_parts else np.zeros((0, 3), dtype=np.uint8)
    if dense_gt_points.size == 0:
        raise RuntimeError("GT depth/pose geometry layer has no finite xyz points")
    if mask_object_points.size == 0:
        raise RuntimeError("F2 selected-mask GT geometry layer has no finite xyz points")

    aggregate_points = _points_from(unique_sampled, offsets)
    observation_points = _points_from(obs_sampled, offsets)
    centroid_rows = (
        unique.groupby(["dataset_split", "scene_id", "chunk_id", "mv_object_id"], as_index=False)
        .agg(
            xyz_x=("xyz_x", "mean"),
            xyz_y=("xyz_y", "mean"),
            xyz_z=("xyz_z", "mean"),
            point_count=("surfel_id", "nunique"),
            frame_count=("frame_id_i", "nunique"),
            score_f=("score_f", "max"),
        )
        .copy()
    )
    centroid_rows, centroid_sampled = _sample_frame(centroid_rows, int(args.max_centroid_count), seed="v100_f2_centroids")
    centroid_points = _points_from(centroid_rows, offsets)

    scene_rows: list[dict[str, Any]] = []
    for key in scene_keys:
        split, scene = key.split(":", 1)
        sub = unique[(unique["dataset_split"].astype(str) == split) & (unique["scene_id"].astype(str) == scene)]
        source = v100_rows[(v100_rows["dataset_split"].astype(str) == split) & (v100_rows["scene_id"].astype(str) == scene)]
        metric = scene_metrics.get((split, scene), {})
        frame_values = _to_int_series(source["frame_id"])
        object_count = int(source["mv_object_id"].astype(str).nunique())
        gt_scene = metric.get("gt_object_count_scene")
        pred_scene = metric.get("pred_object_count_scene")
        scene_rows.append(
            {
                "split": split,
                "scene_id": scene,
                "offset": offsets[key].tolist(),
                "frame_mask_rows": int(len(source)),
                "object_count": object_count,
                "unique_surfel_count": int(sub["surfel_id"].astype(str).nunique()),
                "chunk_count": int(source["chunk_id"].astype(str).nunique()),
                "frame_min": int(frame_values.min()) if len(frame_values) else None,
                "frame_max": int(frame_values.max()) if len(frame_values) else None,
                "frame_count": int(frame_values.nunique()),
                "MV_AP_window": metric.get("MV_AP_window"),
                "MV_AP50_window": metric.get("MV_AP50_window"),
                "MV_AP_scene": metric.get("MV_AP_scene"),
                "MV_AP50_scene": metric.get("MV_AP50_scene"),
                "gt_object_count_scene": gt_scene,
                "pred_object_count_scene": pred_scene,
                "pred_to_gt_scene_object_ratio": (
                    float(pred_scene) / float(gt_scene)
                    if isinstance(pred_scene, int) and isinstance(gt_scene, int) and gt_scene > 0
                    else None
                ),
            }
        )

    arrays = {
        "gt_dense_points": dense_gt_points,
        "gt_dense_rgb_colors": dense_gt_colors,
        "mask_object_points": mask_object_points,
        "mask_object_colors": mask_object_colors,
        "mask_id_colors": mask_id_colors,
        "da3_geometry_points": da3_points,
        "da3_geometry_frame_colors": _colors_for(da3_sampled["frame_id_i"], salt="da3_frame"),
        "da3_geometry_confidence_colors": _confidence_colors(da3_sampled["provider_confidence_f"]),
        "aggregate_points": aggregate_points,
        "aggregate_object_colors": _colors_for(unique_sampled["mv_object_id"], salt="object"),
        "aggregate_chunk_colors": _colors_for(unique_sampled["dataset_split"].astype(str) + ":" + unique_sampled["scene_id"].astype(str) + ":" + unique_sampled["chunk_id"].astype(str), salt="chunk"),
        "aggregate_score_colors": _score_colors(unique_sampled["score_f"]),
        "observation_points": observation_points,
        "observation_object_colors": _colors_for(obs_sampled["mv_object_id"], salt="object"),
        "centroid_points": centroid_points,
        "centroid_object_colors": _colors_for(centroid_rows["mv_object_id"], salt="object"),
    }
    dataframes = {
        "da3_geometry": da3_sampled,
        "unique": unique_sampled,
        "observations": obs_sampled,
        "centroids": centroid_rows,
    }
    status = {
        "viewer": "v100_f2_surfel_maskview_viewer",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": int(os.getpid()),
        "host": args.host,
        "port": int(args.port),
        "url": f"http://localhost:{int(args.port)}",
        "variant_id": VARIANT_ID,
        "phase2_root": _rel(phase2_root),
        "output_root": _rel(args.output_root),
        "split_filter": args.split,
        "scene_filter": args.scene,
        "chunk_filter": args.chunk,
        "resolved_chunk": resolved_chunk,
        "contract": {
            "visualization_uses_gt_for_prediction": False,
            "visualization_uses_gt_geometry": True,
            "visualization_gt_usage_note": "GT ScanNet depth/pose/intrinsics are used only for this requested visualization, not as a method metric result.",
            "visualization_source": "primary object layer colors GT ScanNet RGB-D/pose geometry points whose resized 2D label raster equals each v100 selected_mask_id; sampled full GT RGB geometry is also available as a background toggle",
            "prediction_scope": "single-scene first-chunk diagnostic by default; chunk-causal local/window result, not a scene-memory claim",
            "selected_mask_geometry_coloring": "for each frame, load the source mask raster, nearest-resize to the GT depth shape, and color GT sampled depth pixels where label == selected_mask_id by mv_object_id",
            "gt_geometry_filtering": "GT RGB-D geometry uses ScanNet depth>0 and finite pose/xyz; gt_dense_step controls explicit pixel-stride sampling.",
            "phase5_surfel_overlay_note": "Phase5 fused surfel observations are kept only as hidden diagnostic layers and are not the primary geometry layer",
        },
        "source_files": {
            "mv_object_frame_mask_rows": _rel(phase2_root / "mv_object_frame_mask_rows.parquet"),
            "mv_object_frame_mask_rows_sha256": _sha256(phase2_root / "mv_object_frame_mask_rows.parquet"),
            "summary": _rel(phase2_root / "summary.json"),
            "summary_sha256": _sha256(phase2_root / "summary.json"),
        },
        "mask_source": mask_source_diag,
        "split_sources": split_diags,
        "gt_geometry_sources": gt_diags,
        "scene_rows": scene_rows,
        "aggregate_metrics": aggregate_metrics,
        "counts": {
            "v100_frame_mask_rows_selected": int(len(v100_rows)),
            "gt_dense_geometry_points": int(dense_gt_points.shape[0]),
            "gt_dense_geometry_frames": int(sum(int(item["rendered_frame_count"]) for item in gt_diags)),
            "gt_dense_grid_pixels_after_step": int(sum(int(item["grid_pixels_after_step"]) for item in gt_diags)),
            "gt_dense_valid_depth_points": int(sum(int(item["valid_depth_points"]) for item in gt_diags)),
            "f2_selected_mask_gt_points": int(mask_object_points.shape[0]),
            "f2_selected_mask_gt_label_hit_points": int(sum(int(item["mask_label_hit_points"]) for item in gt_diags)),
            "f2_selected_mask_raster_area_pixels_gt_shape": int(
                sum(int(item["selected_mask_raster_area_pixels_gt_shape"]) for item in gt_diags)
            ),
            "f2_selected_mask_missing_raster_count": int(sum(int(item["missing_mask_raster_count"]) for item in gt_diags)),
            "f2_selected_mask_resized_raster_count": int(sum(int(item["resized_mask_raster_count"]) for item in gt_diags)),
            "f2_duplicate_mask_object_keys": int(sum(int(item["duplicate_mask_object_keys"]) for item in gt_diags)),
            "f2_duplicate_mask_row_count": int(sum(int(item["duplicate_mask_row_count"]) for item in gt_diags)),
            "da3_chunk_all_geometry_rows": int(len(da3_geometry_all)),
            "da3_chunk_all_geometry_rows_rendered": int(len(da3_sampled)),
            "joined_observation_rows": int(len(observations)),
            "unique_object_surfel_rows": int(len(unique)),
            "unique_object_surfel_rows_rendered": int(len(unique_sampled)),
            "observation_rows_rendered": int(len(obs_sampled)),
            "centroid_rows_rendered": int(len(centroid_rows)),
            "object_count": int(v100_rows["mv_object_id"].astype(str).nunique()),
            "scene_panel_count": int(len(scene_keys)),
            "missing_selected_mask_key_count": int(missing_key_count),
        },
        "sampling": {
            "gt_dense_step": int(args.gt_dense_step),
            "gt_dense_stride_sampling_enabled": bool(int(args.gt_dense_step) != 1),
            "gt_depth_scale": float(args.gt_depth_scale),
            "da3_geometry_sampled": da3_sampled_flag,
            "max_da3_geometry_points": int(args.max_da3_geometry_points),
            "aggregate_sampled": aggregate_sampled,
            "observation_density_sampled": obs_density_sampled,
            "centroid_sampled": centroid_sampled,
            "max_aggregate_points": int(args.max_aggregate_points),
            "max_observation_points": int(args.max_observation_points),
            "max_centroid_count": int(args.max_centroid_count),
        },
        "layers": [
            "F2 selected-mask GT geometry - object colors",
            "F2 selected-mask GT geometry - mask-id colors",
            "GT RGB-D geometry - sampled depth/color",
            "Phase5 fused surfel observations - frame colors",
            "Phase5 fused surfel observations - confidence colors",
            "F2 object aggregate - object colors",
            "F2 object aggregate - chunk colors",
            "F2 object aggregate - score colors",
            "F2 observation density - object colors",
            "F2 object centroids",
        ],
    }
    return status, arrays, dataframes


def _add_toggle(server: Any, label: str, handle: Any, visible: bool) -> None:
    handle.visible = bool(visible)
    toggle = server.gui.add_checkbox(label, bool(visible))

    @toggle.on_update
    def _(_: Any) -> None:
        handle.visible = bool(toggle.value)


def serve(args: argparse.Namespace) -> dict[str, Any]:
    status, arrays, dfs = _build_viewer_payload(args)

    import viser  # type: ignore

    server = viser.ViserServer(host=args.host, port=int(args.port), verbose=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/v100_f2/grid",
        width=max(8.0, float(status["counts"]["scene_panel_count"]) * float(args.scene_spacing)),
        height=max(8.0, float(status["counts"]["scene_panel_count"]) * float(args.scene_spacing)),
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
        position=(0.0, 0.0, -0.03),
    )

    mask_object_geometry = server.scene.add_point_cloud(
        "/v100_f2/F2 selected-mask GT geometry - object colors",
        points=arrays["mask_object_points"],
        colors=arrays["mask_object_colors"],
        point_size=float(args.mask_object_point_size),
        point_shape="circle",
        visible=True,
        precision="float32",
    )
    mask_id_geometry = server.scene.add_point_cloud(
        "/v100_f2/F2 selected-mask GT geometry - mask-id colors",
        points=arrays["mask_object_points"],
        colors=arrays["mask_id_colors"],
        point_size=float(args.mask_object_point_size),
        point_shape="circle",
        visible=False,
        precision="float32",
    )
    gt_dense_rgb = server.scene.add_point_cloud(
        "/v100_f2/GT RGB-D geometry - sampled depth/color",
        points=arrays["gt_dense_points"],
        colors=arrays["gt_dense_rgb_colors"],
        point_size=float(args.gt_dense_point_size),
        point_shape="circle",
        visible=False,
        precision="float32",
    )
    da3_geometry = server.scene.add_point_cloud(
        "/v100_f2/Phase5 fused surfel observations - frame colors",
        points=arrays["da3_geometry_points"],
        colors=arrays["da3_geometry_frame_colors"],
        point_size=float(args.da3_geometry_point_size),
        point_shape="circle",
        visible=False,
        precision="float32",
    )
    da3_confidence = server.scene.add_point_cloud(
        "/v100_f2/Phase5 fused surfel observations - confidence colors",
        points=arrays["da3_geometry_points"],
        colors=arrays["da3_geometry_confidence_colors"],
        point_size=float(args.da3_geometry_point_size),
        point_shape="circle",
        visible=False,
        precision="float32",
    )
    aggregate_object = server.scene.add_point_cloud(
        "/v100_f2/F2 object aggregate - object colors",
        points=arrays["aggregate_points"],
        colors=arrays["aggregate_object_colors"],
        point_size=float(args.aggregate_point_size),
        point_shape="circle",
        visible=False,
        precision="float32",
    )
    aggregate_chunk = server.scene.add_point_cloud(
        "/v100_f2/F2 object aggregate - chunk colors",
        points=arrays["aggregate_points"],
        colors=arrays["aggregate_chunk_colors"],
        point_size=float(args.aggregate_point_size),
        point_shape="circle",
        visible=False,
        precision="float32",
    )
    aggregate_score = server.scene.add_point_cloud(
        "/v100_f2/F2 object aggregate - score colors",
        points=arrays["aggregate_points"],
        colors=arrays["aggregate_score_colors"],
        point_size=float(args.aggregate_point_size),
        point_shape="circle",
        visible=False,
        precision="float32",
    )
    observation_density = server.scene.add_point_cloud(
        "/v100_f2/F2 observation density - object colors",
        points=arrays["observation_points"],
        colors=arrays["observation_object_colors"],
        point_size=float(args.observation_point_size),
        point_shape="circle",
        visible=False,
        precision="float32",
    )
    centroids = server.scene.add_point_cloud(
        "/v100_f2/F2 object centroids",
        points=arrays["centroid_points"],
        colors=arrays["centroid_object_colors"],
        point_size=float(args.centroid_point_size),
        point_shape="sparkle",
        visible=False,
        precision="float32",
    )

    _add_toggle(server, "F2 mask GT object colors", mask_object_geometry, True)
    _add_toggle(server, "F2 mask GT mask-id colors", mask_id_geometry, False)
    _add_toggle(server, "GT RGB-D sampled background", gt_dense_rgb, False)
    _add_toggle(server, "Phase5 surfel frame colors", da3_geometry, False)
    _add_toggle(server, "Phase5 surfel confidence colors", da3_confidence, False)
    _add_toggle(server, "Sparse object surfel colors", aggregate_object, False)
    _add_toggle(server, "Chunk colors", aggregate_chunk, False)
    _add_toggle(server, "Score colors", aggregate_score, False)
    _add_toggle(server, "Observation density", observation_density, False)
    _add_toggle(server, "Object centroids", centroids, False)

    for row in status["scene_rows"]:
        offset = np.asarray(row["offset"], dtype=np.float32)
        server.scene.add_frame(
            f"/v100_f2/{row['split']}/{row['scene_id']}/origin",
            position=tuple(float(v) for v in offset),
            axes_length=0.25,
            axes_radius=0.01,
        )
        label = (
            f"{row['split']} {row['scene_id']} {status['resolved_chunk']}\n"
            f"window AP={_fmt_metric(row.get('MV_AP_window'))} AP50={_fmt_metric(row.get('MV_AP50_window'))}\n"
            f"scene AP={_fmt_metric(row.get('MV_AP_scene'))} AP50={_fmt_metric(row.get('MV_AP50_scene'))}\n"
            f"objects={row['object_count']} chunks={row['chunk_count']} surfels={row['unique_surfel_count']}\n"
            f"F2 mask GT pts={status['counts']['f2_selected_mask_gt_points']}\n"
            f"GT sampled bg pts={status['counts']['gt_dense_geometry_points']}\n"
            f"Phase5 surfel obs={status['counts']['da3_chunk_all_geometry_rows']}\n"
            f"pred/gt(scene)={row.get('pred_object_count_scene')}/{row.get('gt_object_count_scene')}"
        )
        server.scene.add_label(
            f"/v100_f2/{row['split']}/{row['scene_id']}/metric_label",
            label,
            position=tuple(float(v) for v in offset + np.asarray([-1.8, 1.8, 1.8], dtype=np.float32)),
            font_screen_scale=0.82,
            anchor="top-left",
        )

    label_rows = dfs["centroids"].sort_values(["frame_count", "point_count", "score_f"], ascending=False).head(int(args.label_top_k))
    for idx, row in label_rows.iterrows():
        key = f"{row['dataset_split']}:{row['scene_id']}"
        offset = np.asarray(next(item["offset"] for item in status["scene_rows"] if f"{item['split']}:{item['scene_id']}" == key), dtype=np.float32)
        position = np.asarray([float(row["xyz_x"]), float(row["xyz_y"]), float(row["xyz_z"])], dtype=np.float32) + offset
        short_id = str(row["mv_object_id"]).split(":")[-1]
        server.scene.add_label(
            f"/v100_f2/object_label/{idx}",
            f"{row['chunk_id']} {short_id}\nframes={int(row['frame_count'])} score={float(row['score_f']):.3f}",
            position=tuple(float(v) for v in position + np.asarray([0.0, 0.0, 0.08], dtype=np.float32)),
            font_screen_scale=0.45,
            anchor="middle",
        )

    server.gui.add_text("variant", VARIANT_ID, disabled=True)
    server.gui.add_text("scope", f"{status['split_filter']} {status['scene_filter']} {status['resolved_chunk']}", disabled=True)
    server.gui.add_number("F2 mask GT pts", int(status["counts"]["f2_selected_mask_gt_points"]), disabled=True)
    server.gui.add_number("GT sampled bg pts", int(status["counts"]["gt_dense_geometry_points"]), disabled=True)
    server.gui.add_number("Phase5 surfel obs", int(status["counts"]["da3_chunk_all_geometry_rows"]), disabled=True)
    server.gui.add_number("unique surfels", int(status["counts"]["unique_object_surfel_rows"]), disabled=True)
    server.gui.add_number("objects", int(status["counts"]["object_count"]), disabled=True)
    server.gui.add_number("joined obs", int(status["counts"]["joined_observation_rows"]), disabled=True)
    server.gui.add_text("claim", "local/window only; scene layer fragmented diagnostic", disabled=True)
    status["server_started"] = True
    status["gui_controls_available"] = True
    status["status_json"] = _rel(args.status_json) if args.status_json else _rel(Path(args.output_root) / "viewer_status.json")
    status["gate"] = {
        "viser_server_started": True,
        "f2_selected_mask_gt_points_nonempty": bool(arrays["mask_object_points"].shape[0] > 0),
        "f2_selected_mask_missing_raster_count_eq_0": int(status["counts"]["f2_selected_mask_missing_raster_count"]) == 0,
        "gt_dense_points_nonempty": bool(arrays["gt_dense_points"].shape[0] > 0),
        "phase5_surfel_points_nonempty": bool(arrays["da3_geometry_points"].shape[0] > 0),
        "phase5_surfel_not_sampled": not bool(status["sampling"]["da3_geometry_sampled"]),
        "aggregate_points_nonempty": bool(arrays["aggregate_points"].shape[0] > 0),
        "object_centroids_nonempty": bool(arrays["centroid_points"].shape[0] > 0),
        "missing_selected_mask_key_count_eq_0": int(status["counts"]["missing_selected_mask_key_count"]) == 0,
        "scene_panel_count_positive": int(status["counts"]["scene_panel_count"]) > 0,
    }
    status["gate"]["pass"] = bool(all(status["gate"].values()))

    status_path = Path(args.status_json) if args.status_json else Path(args.output_root) / "viewer_status.json"
    _write_json(status_path, status)
    _write_json(Path(args.output_root) / "viewer_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True, default=_json_default), flush=True)

    stop = {"value": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    if float(args.smoke_seconds) > 0.0:
        deadline = time.time() + float(args.smoke_seconds)
        while time.time() < deadline:
            time.sleep(0.25)
        server.stop()
        return status
    while not stop["value"]:
        time.sleep(1.0)
    server.stop()
    return status


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "NA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the v100 F2 surfel-maskview local/window result in Viser.")
    parser.add_argument("--phase2-root", default=str(DEFAULT_PHASE2_ROOT))
    parser.add_argument("--source-registry", default=str(DEFAULT_SOURCE_REGISTRY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", choices=["all", "dev", "holdout"], default="dev")
    parser.add_argument("--scene", default="scene0011_00", help="Scene id, or all.")
    parser.add_argument("--chunk", default="first", help="Chunk id, 'first', or all.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--status-json", default="")
    parser.add_argument("--scene-spacing", type=float, default=4.5)
    parser.add_argument("--max-da3-geometry-points", type=int, default=0)
    parser.add_argument("--max-aggregate-points", type=int, default=0)
    parser.add_argument("--max-observation-points", type=int, default=0)
    parser.add_argument("--max-centroid-count", type=int, default=2000)
    parser.add_argument("--label-top-k", type=int, default=36)
    parser.add_argument("--da3-dense-step", type=int, default=1, help="Pixel stride for DA3 dense geometry; 1 keeps every pixel.")
    parser.add_argument("--da3-dense-conf-min", type=float, default=None, help="Optional DA3 confidence threshold; omitted means no confidence filtering.")
    parser.add_argument("--da3-dense-point-size", type=float, default=0.004)
    parser.add_argument("--gt-dense-step", type=int, default=4, help="Pixel stride for GT ScanNet RGB-D geometry; user requested 4x sampling.")
    parser.add_argument("--gt-depth-scale", type=float, default=1000.0)
    parser.add_argument("--gt-dense-point-size", type=float, default=0.012)
    parser.add_argument("--mask-object-point-size", type=float, default=0.010)
    parser.add_argument("--da3-geometry-point-size", type=float, default=0.010)
    parser.add_argument("--aggregate-point-size", type=float, default=0.018)
    parser.add_argument("--observation-point-size", type=float, default=0.010)
    parser.add_argument("--centroid-point-size", type=float, default=0.075)
    parser.add_argument("--smoke-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    serve(parse_args())


if __name__ == "__main__":
    main()
