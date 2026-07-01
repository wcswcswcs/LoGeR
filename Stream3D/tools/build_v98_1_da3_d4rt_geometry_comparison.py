#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
STREAM3D_ROOT = ROOT / "Stream3D"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(STREAM3D_ROOT))

from build_v98_1_da3_variant_geometry_quality import (  # noqa: E402
    _chamfer_metrics,
    _fscore_row,
    _load_da3_manifest,
    _residual_stats,
    _sample_indices,
    _write_csv,
    _write_json,
)
from geometry_provider.common import backproject_xy_world, fit_transform  # noqa: E402
from serve_v98_1_da3_gt_dense_rgb_sim3_viewer import _json_default  # noqa: E402
from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder  # noqa: E402
from stream4d_native.sim3 import apply_sim3_to_xyz  # noqa: E402


DEFAULT_BASE_ROOT = STREAM3D_ROOT / "outputs" / "audit" / "v98_1_da3_variant_geometry_quality_scene0050_input_visible_gt"
DEFAULT_OUTPUT_ROOT = STREAM3D_ROOT / "outputs" / "audit" / "v98_1_da3_d4rt_geometry_comparison_scene0050_input_visible_gt"
DEFAULT_D4RT_ROOT = (
    STREAM3D_ROOT
    / "outputs"
    / "audit"
    / "v65_soma_fullscene_pipeline_scene0050_stride5_conf02_integrated_d4rt"
    / "carrier_cache"
    / "scene0050_00"
)
DEFAULT_D4RT_CHUNK_ROOT = Path("")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_carrier_window(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = _load_json(path.with_name(f"{path.stem}_manifest.json"))
    with np.load(path) as payload:
        data = {key: np.asarray(payload[key]) for key in payload.files}
    return manifest, data


def _selected_d4rt_paths(carrier_root: Path, input_frame_ids: set[int]) -> list[Path]:
    selected: list[Path] = []
    for manifest_path in sorted(carrier_root.glob("carriers_window*_manifest.json")):
        manifest = _load_json(manifest_path)
        frame_ids = {int(value) for value in manifest.get("frame_ids", [])}
        if frame_ids & input_frame_ids:
            selected.append(manifest_path.with_name(manifest_path.name.replace("_manifest.json", ".npz")))
    if not selected:
        raise FileNotFoundError(f"no D4RT windows under {carrier_root} intersect the DA3 input manifest")
    return selected


def _selected_d4rt_chunk_paths(chunk_root: Path, input_frame_ids: set[int]) -> list[Path]:
    selected: list[Path] = []
    for chunk_path in sorted(chunk_root.glob("chunk_*.npz")):
        with np.load(chunk_path) as payload:
            if "frame_ids" not in payload.files:
                continue
            frame_ids = {int(value) for value in np.asarray(payload["frame_ids"], dtype=np.int64).tolist()}
        if frame_ids & input_frame_ids:
            selected.append(chunk_path)
    if not selected:
        raise FileNotFoundError(f"no D4RT chunk npz files under {chunk_root} intersect the DA3 input manifest")
    return selected


def _window_overlap_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifests = [_load_window_manifest(path) for path in paths]
    for idx in range(1, len(paths)):
        prev_ids = [int(value) for value in manifests[idx - 1]["frame_ids"]]
        curr_ids = [int(value) for value in manifests[idx]["frame_ids"]]
        overlap = sorted(set(prev_ids) & set(curr_ids))
        rows.append(
            {
                "prev_window": paths[idx - 1].name,
                "curr_window": paths[idx].name,
                "overlap_frame_count": int(len(overlap)),
                "overlap_frame_ids": overlap,
            }
        )
    return rows


def _d4rt_materialization_params(source_root: Path, source_kind: str) -> dict[str, Any]:
    if source_kind == "chunk_npz":
        stride_summary_path = source_root.parent / "stride_summary.json"
        last_command_path = source_root.parent.parent / "last_command.txt"
        if not stride_summary_path.is_file():
            return {"summary_path": str(stride_summary_path), "exists": False, "source_kind": source_kind}
        payload = _load_json(stride_summary_path)
        return {
            "summary_path": str(stride_summary_path),
            "exists": True,
            "source_kind": source_kind,
            "d4rt_grid_size": payload.get("grid_size"),
            "grid_points_per_frame": payload.get("grid_points_per_frame"),
            "d4rt_grid_margin_ratio": None,
            "d4rt_query_chunk_size": None,
            "d4rt_chunk_size": payload.get("chunk_size"),
            "d4rt_overlap_frames": payload.get("overlap_frames"),
            "stride": payload.get("stride"),
            "d4rt_min_visibility": payload.get("min_visibility"),
            "d4rt_min_confidence": payload.get("min_confidence"),
            "d4rt_ckpt": payload.get("d4rt_ckpt"),
            "d4rt_ckpt_sha256": payload.get("d4rt_ckpt_sha256"),
            "source_command": last_command_path.read_text(encoding="utf-8").strip() if last_command_path.is_file() else None,
        }
    summary_path = source_root.parent.parent / "d4rt_geometry_materialization_summary.json"
    if not summary_path.is_file():
        return {"summary_path": str(summary_path), "exists": False, "source_kind": source_kind}
    payload = _load_json(summary_path)
    expected = payload.get("expected_params", {})
    return {
        "summary_path": str(summary_path),
        "exists": True,
        "source_kind": source_kind,
        "d4rt_grid_size": expected.get("d4rt_grid_size"),
        "grid_points_per_frame": None if expected.get("d4rt_grid_size") is None else int(expected.get("d4rt_grid_size")) ** 2,
        "d4rt_grid_margin_ratio": expected.get("d4rt_grid_margin_ratio"),
        "d4rt_query_chunk_size": expected.get("d4rt_query_chunk_size"),
        "d4rt_chunk_size": expected.get("d4rt_chunk_size"),
        "d4rt_overlap_frames": expected.get("d4rt_overlap_frames"),
        "stride": expected.get("stride"),
        "d4rt_min_visibility": expected.get("d4rt_min_visibility"),
        "d4rt_min_confidence": expected.get("d4rt_min_confidence"),
        "d4rt_ckpt": expected.get("d4rt_ckpt"),
        "d4rt_ckpt_sha256": expected.get("d4rt_ckpt_sha256"),
        "source_command": next(
            (row.get("command") for row in _read_csv(carrier_root.parent.parent / "d4rt_geometry_process_rows.csv")),
            None,
        )
        if (carrier_root.parent.parent / "d4rt_geometry_process_rows.csv").is_file()
        else None,
    }


def _load_window_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_name(f"{path.stem}_manifest.json")
    if manifest_path.is_file():
        return _load_json(manifest_path)
    summary_path = path.with_name(f"{path.stem}_summary.json")
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    with np.load(path) as payload:
        frame_ids = [int(value) for value in np.asarray(payload["frame_ids"], dtype=np.int64).tolist()]
        carrier_count = int(np.asarray(payload["xyz" if "xyz" in payload.files else "xyz_ref"]).shape[1])
    return {
        "carrier_count": carrier_count,
        "frame_ids": frame_ids,
        "input_stride": int(frame_ids[1] - frame_ids[0]) if len(frame_ids) > 1 else None,
        "pipeline_stage": summary.get("phase", "d4rt_raw_chunk_npz"),
        "scene": summary.get("scene"),
        "source_chunk_npz": str(path),
        "uses_gt_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_rgbd_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "grid_points_per_frame": summary.get("grid_points_per_frame"),
        "grid_margin_ratio": summary.get("grid_margin_ratio"),
        "query_chunk_size": summary.get("query_chunk_size"),
    }


def _chunks_from_d4rt_windows(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for window_index, path in enumerate(paths):
        manifest, data = _load_carrier_window(path) if path.with_name(f"{path.stem}_manifest.json").is_file() else (_load_window_manifest(path), _load_npz(path))
        frame_ids = [int(value) for value in manifest["frame_ids"]]
        xyz = np.asarray(data["xyz_ref"] if "xyz_ref" in data else data["xyz"], dtype=np.float32)
        uv = np.asarray(data["uv_pred"] if "uv_pred" in data else data["uv"], dtype=np.float32)
        valid = np.asarray(data.get("valid", np.ones(xyz.shape[:2], dtype=bool)), dtype=bool)
        visibility = np.asarray(data.get("visibility_prob", data.get("visibility", np.ones(xyz.shape[:2], dtype=np.float32))), dtype=np.float32)
        confidence = np.asarray(data.get("confidence_prob", data.get("confidence", np.ones(xyz.shape[:2], dtype=np.float32))), dtype=np.float32)
        carrier_id = np.asarray(data.get("carrier_id", np.arange(xyz.shape[1])), dtype=np.int64)
        persistent_tube_id = np.asarray(
            data.get("persistent_tube_id", data.get("source_carrier_id", data.get("carrier_id", np.full((xyz.shape[1],), -1)))),
            dtype=np.int64,
        )
        src_frame_global = np.asarray(data.get("src_frame_global", np.full((xyz.shape[1],), -1)), dtype=np.int64)
        src_xy = np.asarray(data.get("src_xy", np.full((xyz.shape[1], 2), -1)), dtype=np.int64)

        tubes: list[dict[str, Any]] = []
        for tube_index in range(xyz.shape[1]):
            source_frame = int(src_frame_global[tube_index])
            source_frame_local = frame_ids.index(source_frame) if source_frame in frame_ids else 0
            tubes.append(
                {
                    "carrier_id": int(carrier_id[tube_index]),
                    "persistent_tube_id": int(persistent_tube_id[tube_index]),
                    "uv_norm": uv[:, tube_index, :],
                    "xyz": xyz[:, tube_index, :],
                    "xyz_ref0": xyz[:, tube_index, :],
                    "visibility": visibility[:, tube_index],
                    "confidence": confidence[:, tube_index],
                    "valid": valid[:, tube_index],
                    "source_frame_local": int(source_frame_local),
                    "source_frame_global": int(source_frame),
                    "source_xy": tuple(int(value) for value in src_xy[tube_index].tolist()),
                }
            )
        chunks.append(
            {
                "chunk": {
                    "chunk_id": int(window_index),
                    "start": int(frame_ids[0]),
                    "end": int(frame_ids[-1] + int(manifest.get("input_stride", 1))),
                    "frame_ids": frame_ids,
                    "carrier_npz": str(path),
                    "source_chunk_npz": str(manifest.get("source_chunk_npz", "")),
                },
                "tubes": tubes,
            }
        )
        window_rows.append(
            {
                "window_index": int(window_index),
                "path": str(path),
                "frame_id_min": int(frame_ids[0]),
                "frame_id_max": int(frame_ids[-1]),
                "frame_count": int(len(frame_ids)),
                "carrier_count": int(xyz.shape[1]),
                "grid_points_per_frame": manifest.get("grid_points_per_frame"),
                "input_stride": int(manifest.get("input_stride", -1)),
                "source_chunk_npz": str(manifest.get("source_chunk_npz", "")),
                "uses_gt_for_prediction": bool(manifest.get("uses_gt_for_prediction", False)),
                "uses_pose_for_prediction": bool(manifest.get("uses_pose_for_prediction", False)),
                "uses_rgbd_for_prediction": bool(manifest.get("uses_rgbd_for_prediction", False)),
                "uses_scannet_mesh_for_prediction": bool(manifest.get("uses_scannet_mesh_for_prediction", False)),
            }
        )
    return chunks, window_rows


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _rgb_for_uv(stream: ScanNetStream, frame_id: int, uv: np.ndarray) -> np.ndarray:
    image = stream.load_rgb(int(frame_id))
    h, w = image.shape[:2]
    x = np.rint(np.asarray(uv[:, 0], dtype=np.float64) * float(max(w - 1, 1))).astype(np.int64)
    y = np.rint(np.asarray(uv[:, 1], dtype=np.float64) * float(max(h - 1, 1))).astype(np.int64)
    x = np.clip(x, 0, max(w - 1, 0))
    y = np.clip(y, 0, max(h - 1, 0))
    return image[y, x].astype(np.uint8)


def _collect_d4rt_observations(
    *,
    builder: D4RTNativeSceneBuilder,
    stitched_chunks: list[dict[str, Any]],
    stream: ScanNetStream,
    input_frame_ids: set[int],
    min_visibility: float,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    window_parts: list[np.ndarray] = []
    submap_parts: list[np.ndarray] = []
    raw_input_observations = 0
    finite_uv_observations = 0
    filtered_observations = 0
    frame_slot_count = 0
    used_frame_ids: set[int] = set()
    input_frame_array = np.asarray(sorted(input_frame_ids), dtype=np.int64)

    for window_index, chunk in enumerate(stitched_chunks):
        window = builder._chunk_to_window_data(chunk, use_canonical_xyz=True)
        frame_ids = np.asarray(window["frame_ids"], dtype=np.int64)
        frame_in_input = np.isin(frame_ids, input_frame_array)
        submap_id = int(chunk.get("submap_id", 0))
        for local_idx, frame_id in enumerate(frame_ids.tolist()):
            if not bool(frame_in_input[local_idx]):
                continue
            frame_slot_count += 1
            xyz = np.asarray(window["xyz"][local_idx], dtype=np.float32)
            uv = np.asarray(window["uv"][local_idx], dtype=np.float32)
            valid = np.asarray(window["valid"][local_idx], dtype=bool)
            visibility = np.asarray(window["visibility"][local_idx], dtype=np.float32)
            confidence = np.asarray(window["confidence"][local_idx], dtype=np.float32)
            raw_input_observations += int(xyz.shape[0])
            finite_uv = (
                valid
                & np.isfinite(xyz).all(axis=1)
                & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
            )
            finite_uv_observations += int(np.count_nonzero(finite_uv))
            keep = finite_uv & (visibility >= float(min_visibility)) & (confidence >= float(min_confidence))
            if not np.any(keep):
                continue
            pts = xyz[keep]
            colors = _rgb_for_uv(stream, int(frame_id), uv[keep])
            count = int(pts.shape[0])
            filtered_observations += count
            used_frame_ids.add(int(frame_id))
            point_parts.append(pts)
            color_parts.append(colors)
            frame_parts.append(np.full((count,), int(frame_id), dtype=np.int32))
            window_parts.append(np.full((count,), int(window_index), dtype=np.int16))
            submap_parts.append(np.full((count,), int(submap_id), dtype=np.int16))

    if not point_parts:
        raise RuntimeError("D4RT self-stitched input-scope observation set is empty after filters")
    points = np.concatenate(point_parts, axis=0).astype(np.float32)
    colors = np.concatenate(color_parts, axis=0).astype(np.uint8)
    frames = np.concatenate(frame_parts, axis=0).astype(np.int32)
    windows = np.concatenate(window_parts, axis=0).astype(np.int16)
    submaps = np.concatenate(submap_parts, axis=0).astype(np.int16)
    info = {
        "raw_input_observation_count": int(raw_input_observations),
        "finite_uv_observation_count": int(finite_uv_observations),
        "filtered_observation_count": int(filtered_observations),
        "input_window_frame_slot_count": int(frame_slot_count),
        "unique_frame_count_after_filter": int(len(used_frame_ids)),
        "frame_id_min_after_filter": int(min(used_frame_ids)) if used_frame_ids else None,
        "frame_id_max_after_filter": int(max(used_frame_ids)) if used_frame_ids else None,
        "submap_ids_after_filter": sorted(int(value) for value in np.unique(submaps).tolist()),
    }
    return points, colors, frames, windows, submaps, info


def _fit_d4rt_to_scannet(
    *,
    points_canonical: np.ndarray,
    frames: np.ndarray,
    windows: np.ndarray,
    stream: ScanNetStream,
    stitched_chunks: list[dict[str, Any]],
    builder: D4RTNativeSceneBuilder,
    max_anchors: int,
    robust_trim_percentile: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del points_canonical, windows
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    anchor_candidates = 0
    depth_hits = 0
    for chunk in stitched_chunks:
        window = builder._chunk_to_window_data(chunk, use_canonical_xyz=True)
        frame_ids = np.asarray(window["frame_ids"], dtype=np.int64)
        for local_idx, frame_id in enumerate(frame_ids.tolist()):
            if int(frame_id) not in set(int(value) for value in np.unique(frames).tolist()):
                continue
            xyz = np.asarray(window["xyz"][local_idx], dtype=np.float32)
            uv = np.asarray(window["uv"][local_idx], dtype=np.float32)
            valid = np.asarray(window["valid"][local_idx], dtype=bool)
            visibility = np.asarray(window["visibility"][local_idx], dtype=np.float32)
            confidence = np.asarray(window["confidence"][local_idx], dtype=np.float32)
            ok = (
                valid
                & np.isfinite(xyz).all(axis=1)
                & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
                & (visibility >= 0.5)
                & (confidence >= 0.5)
            )
            anchor_candidates += int(np.count_nonzero(ok))
            if not np.any(ok):
                continue
            h, w = stream.load_depth(int(frame_id)).shape[:2]
            xy = np.stack(
                [
                    uv[ok, 0] * float(max(w - 1, 1)),
                    uv[ok, 1] * float(max(h - 1, 1)),
                ],
                axis=1,
            )
            world, world_ok = backproject_xy_world(stream, int(frame_id), xy)
            if not np.any(world_ok):
                continue
            source_parts.append(xyz[ok][world_ok])
            target_parts.append(world[world_ok])
            depth_hits += int(np.count_nonzero(world_ok))
    if not source_parts:
        raise RuntimeError("no D4RT->ScanNet depth/pose anchors available")
    source = np.concatenate(source_parts, axis=0).astype(np.float32)
    target = np.concatenate(target_parts, axis=0).astype(np.float32)
    if source.shape[0] > int(max_anchors):
        keep = np.linspace(0, source.shape[0] - 1, num=int(max_anchors), dtype=np.int64)
    else:
        keep = np.arange(source.shape[0], dtype=np.int64)
    fit = fit_transform(source[keep], target[keep], robust_trim_percentile=float(robust_trim_percentile))
    if fit is None:
        raise RuntimeError("D4RT->ScanNet depth/pose Sim3 fit returned None")
    aligned_all = apply_sim3_to_xyz(source, transform=fit).astype(np.float64)
    residual = np.linalg.norm(aligned_all - target.astype(np.float64), axis=1)
    info = {
        "alignment_type": "diagnostic_scannet_depth_pose_sim3",
        "alignment_source": "ScanNet depth + pose backprojection of D4RT uv_pred",
        "uses_rgbd_pose_for_alignment": True,
        "uses_gt_mesh_for_alignment": False,
        "anchor_candidates": int(anchor_candidates),
        "depth_pose_anchor_count": int(source.shape[0]),
        "depth_pose_hit_count": int(depth_hits),
        "fit_anchor_count": int(keep.shape[0]),
        "robust_trim_percentile": float(robust_trim_percentile),
        "robust_kept_anchors": int(fit.get("robust_kept_anchors", keep.shape[0])),
        "scale": float(fit["scale"]),
        "rotation_det": float(fit.get("rotation_det", np.linalg.det(np.asarray(fit["rotation"], dtype=np.float64)))),
        "translation_norm_m": float(np.linalg.norm(np.asarray(fit["translation"], dtype=np.float64))),
        "all_anchor_residual_m": _residual_stats(residual),
    }
    return fit, info


def _d4rt_metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "variant_key": "d4rt_self_stitched",
        "display_name": "D4RT self-stitched",
        "model": "D4RT",
        "transform": "overlap_self_stitch_then_depth_pose_sim3",
        "chamfer_l2_mean_m": metrics["chamfer_l2_mean_m"],
        "chamfer_l2_squared_mean_m2": metrics["chamfer_l2_squared_mean_m2"],
        "accuracy_mean_m": metrics["accuracy_da3_to_gt_m"]["mean"],
        "accuracy_p50_m": metrics["accuracy_da3_to_gt_m"]["p50"],
        "accuracy_p90_m": metrics["accuracy_da3_to_gt_m"]["p90"],
        "accuracy_p95_m": metrics["accuracy_da3_to_gt_m"]["p95"],
        "completeness_mean_m": metrics["completeness_gt_to_da3_m"]["mean"],
        "completeness_p50_m": metrics["completeness_gt_to_da3_m"]["p50"],
        "completeness_p90_m": metrics["completeness_gt_to_da3_m"]["p90"],
        "completeness_p95_m": metrics["completeness_gt_to_da3_m"]["p95"],
    }
    for key, value in metrics["fscore"].items():
        prefix = key.replace(".", "p")
        row[f"{prefix}_precision"] = value["precision"]
        row[f"{prefix}_recall"] = value["recall"]
        row[f"{prefix}_fscore"] = value["fscore"]
    return row


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build(args: argparse.Namespace) -> dict[str, Any]:
    base_root = Path(args.base_output_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base_summary_path = Path(args.base_summary_json) if args.base_summary_json else base_root / "geometry_quality_summary.json"
    base_summary = _load_json(base_summary_path)
    base_npz_path = Path(args.base_viewer_npz) if args.base_viewer_npz else Path(base_summary["outputs"]["viewer_npz"])
    base_csv_path = Path(base_summary["outputs"]["metrics_csv"])
    manifest = _load_da3_manifest(Path(args.da3_manifest))
    input_frame_ids = {int(value) for value in manifest["frame_id"].tolist()}
    chunk_root = Path(args.d4rt_chunk_root) if args.d4rt_chunk_root else Path("")
    if chunk_root and str(chunk_root) != "." and chunk_root.is_dir():
        source_root = chunk_root
        source_kind = "chunk_npz"
        d4rt_paths = _selected_d4rt_chunk_paths(chunk_root, input_frame_ids)
    else:
        source_root = Path(args.d4rt_carrier_root)
        source_kind = "carrier_cache"
        d4rt_paths = _selected_d4rt_paths(source_root, input_frame_ids)
    chunks, window_rows = _chunks_from_d4rt_windows(d4rt_paths)
    builder = D4RTNativeSceneBuilder.__new__(D4RTNativeSceneBuilder)
    stitched = builder.stitch_to_canonical(chunks)
    stream = ScanNetStream(
        seq_name=args.scene_id,
        backbone="Cropformer",
        root=ROOT / "Stream3D" / "data" / "scannet" / "processed",
    )
    d4rt_points_canonical, d4rt_colors, d4rt_frames, d4rt_windows, d4rt_submaps, observation_info = _collect_d4rt_observations(
        builder=builder,
        stitched_chunks=stitched["chunks"],
        stream=stream,
        input_frame_ids=input_frame_ids,
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
    )
    fit, alignment_info = _fit_d4rt_to_scannet(
        points_canonical=d4rt_points_canonical,
        frames=d4rt_frames,
        windows=d4rt_windows,
        stream=stream,
        stitched_chunks=stitched["chunks"],
        builder=builder,
        max_anchors=int(args.max_sim3_anchors),
        robust_trim_percentile=float(args.robust_trim_percentile),
    )
    d4rt_points_aligned = apply_sim3_to_xyz(d4rt_points_canonical, transform=fit).astype(np.float32)
    with np.load(base_npz_path) as payload:
        npz_payload = {key: np.asarray(payload[key]) for key in payload.files}
    gt_points = np.asarray(npz_payload["gt_points"], dtype=np.float32)
    gt_tree = cKDTree(np.asarray(gt_points, dtype=np.float64))
    thresholds = [float(value) for value in args.fscore_thresholds]
    metrics = _chamfer_metrics(
        source_aligned=d4rt_points_aligned,
        target_gt=gt_points,
        target_gt_tree=gt_tree,
        thresholds=thresholds,
    )
    viewer_idx = _sample_indices(
        d4rt_points_aligned.shape[0],
        int(args.viewer_d4rt_sample_count),
        int(args.seed) + 404,
    )
    npz_payload["d4rt_points"] = d4rt_points_aligned[viewer_idx].astype(np.float32)
    npz_payload["d4rt_colors"] = d4rt_colors[viewer_idx].astype(np.uint8)
    npz_payload["d4rt_frame_ids"] = d4rt_frames[viewer_idx].astype(np.int32)
    npz_payload["d4rt_window_ids"] = d4rt_windows[viewer_idx].astype(np.int16)
    npz_payload["d4rt_submap_ids"] = d4rt_submaps[viewer_idx].astype(np.int16)
    npz_path = output_root / f"{args.scene_id}_da3_d4rt_geometry_viewer_points.npz"
    np.savez_compressed(npz_path, **npz_payload)

    csv_rows = _read_csv(base_csv_path)
    d4rt_row = _d4rt_metric_row(metrics)
    csv_rows.append(d4rt_row)
    csv_path = output_root / "geometry_quality_metrics_with_d4rt.csv"
    _write_csv(csv_path, csv_rows)

    diagnostics = stitched["diagnostics"]
    d4rt_geometry = {
        "display_name": "D4RT self-stitched",
        "diagnostic_only": True,
        "method_result_allowed": False,
        "source": {
            "source_root": str(source_root),
            "source_kind": source_kind,
            "materialization_params": _d4rt_materialization_params(source_root, source_kind),
            "selected_window_count": int(len(d4rt_paths)),
            "selected_windows": window_rows,
            "window_overlap_rows": _window_overlap_rows(d4rt_paths),
            "self_stitch_code_path": "Stream3D/stream4d_native/d4rt_scene_builder.py::D4RTNativeSceneBuilder.stitch_to_canonical",
            "source_persistent_id_field": "source_carrier_id mapped to persistent_tube_id when persistent_tube_id is absent",
            "sampling_contract": {
                "carrier_count_per_window": int(window_rows[0]["carrier_count"]) if window_rows else None,
                "source_queries_per_frame": int(window_rows[0]["grid_points_per_frame"])
                if window_rows and window_rows[0].get("grid_points_per_frame") is not None
                else None,
                "frames_per_window": int(window_rows[0]["frame_count"]) if window_rows else None,
                "fresh_samples_per_image": False,
                "interpretation": (
                    "D4RT source queries are sampled on every source frame. With grid_size=G this gives G*G source "
                    "queries per frame and G*G*frames_per_window carriers per window. D4RT then propagates every carrier "
                    "across the whole window, so a target frame can receive up to carrier_count_per_window observations "
                    "from one window. Because this comparison keeps all self-stitched overlap-window observations, an "
                    "overlap target frame can contribute observations from two windows before filtering."
                ),
                "overlap_window_policy_for_viewer_and_metrics": "all_window_union_after_self_stitch",
            },
        },
        "input_scope": {
            "da3_manifest": str(args.da3_manifest),
            "manifest_frame_count": int(manifest.shape[0]),
            "frame_id_min": int(manifest["frame_id"].min()),
            "frame_id_max": int(manifest["frame_id"].max()),
            "only_da3_input_frames_retained_for_viewer_and_metrics": True,
        },
        "filters": {
            "min_visibility": float(args.min_visibility),
            "min_confidence": float(args.min_confidence),
            "uv_in_bounds_required": True,
            "finite_xyz_required": True,
        },
        "observation_counts": {
            **observation_info,
            "viewer_d4rt_point_count": int(viewer_idx.shape[0]),
        },
        "overlap_self_stitch": diagnostics,
        "alignment_to_scannet": alignment_info,
        "geometry_metrics_against_input_visible_gt": metrics,
        "csv_row": d4rt_row,
    }
    summary = dict(base_summary)
    summary["outputs"] = dict(base_summary["outputs"])
    summary["outputs"].update(
        {
            "base_summary_json": str(base_summary_path),
            "base_viewer_npz": str(base_npz_path),
            "summary_json": str(output_root / "geometry_quality_summary_with_d4rt.json"),
            "metrics_csv": str(csv_path),
            "viewer_npz": str(npz_path),
        }
    )
    summary["metric_note"] = (
        str(base_summary.get("metric_note", ""))
        + " D4RT self-stitched geometry is added as a diagnostic comparison layer: D4RT windows are stitched with overlap self-Sim3, "
        "then aligned to ScanNet world by ScanNet depth/pose backprojection anchors for visualization and Chamfer diagnostics."
    )
    summary["d4rt_geometry"] = d4rt_geometry
    summary["csv_rows"] = csv_rows
    summary_path = output_root / "geometry_quality_summary_with_d4rt.json"
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary_json": str(summary_path),
                "metrics_csv": str(csv_path),
                "viewer_npz": str(npz_path),
                "d4rt_filtered_observation_count": int(observation_info["filtered_observation_count"]),
                "d4rt_viewer_point_count": int(viewer_idx.shape[0]),
                "d4rt_chamfer_l2_mean_m": float(metrics["chamfer_l2_mean_m"]),
                "d4rt_fscore_0p10m": float(metrics["fscore"]["0.10m"]["fscore"]),
                "self_stitch_submap_count": int(diagnostics["submap_count"]),
                "self_stitch_weak_alignment_chunk_count": int(diagnostics["weak_alignment_chunk_count"]),
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Add D4RT overlap self-stitched geometry to the v98.1 DA3/GT Viser comparison.")
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--base-output-root", default=str(DEFAULT_BASE_ROOT))
    parser.add_argument("--base-summary-json", default="")
    parser.add_argument("--base-viewer-npz", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--da3-manifest", default=str(DEFAULT_BASE_ROOT.parent / "v98_phase1_provider_contract" / "da3_streaming_d4rt32o3_scene0050_input119" / "frame_manifest_rows.csv"))
    parser.add_argument("--d4rt-carrier-root", default=str(DEFAULT_D4RT_ROOT))
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-sim3-anchors", type=int, default=100000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--viewer-d4rt-sample-count", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=9801098)
    parser.add_argument("--fscore-thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.50])
    build(parser.parse_args())


if __name__ == "__main__":
    main()
