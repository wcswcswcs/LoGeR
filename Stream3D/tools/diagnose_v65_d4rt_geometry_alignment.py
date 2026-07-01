#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry_provider.common import backproject_xy_world, fit_transform
from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider, _apply_fit
from stream4d.scannet_stream import ScanNetStream
from stream4d_native.v47_common import write_json
from stream4d_native.v65_common import project, rel, sha256_file
from stream4d_native.v65_visualization_export import (
    D4RT_DEBUG_ROOT,
    _load_aligned_d4rt_scene_points,
    _load_prediction_overlay,
    _load_scene_mesh,
)


DEFAULT_MODES = [
    "raw",
    "self_stitched",
    "self_stitched_scale_normalized",
    "eval_sim3",
    "self_stitched_scale_normalized_eval_sim3",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose v65 D4RT geometry alignment against ScanNet mesh.")
    parser.add_argument("--scene", default="scene0050_00")
    parser.add_argument("--debug-root", default=D4RT_DEBUG_ROOT)
    parser.add_argument("--pred-config", default="v64r2_d4rt_chunk_scale_first_ap_probe5_g11")
    parser.add_argument("--output-root", default="outputs/audit/v65_d4rt_geometry_alignment")
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    out = project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    scene_points, _colors, mesh_path = _load_scene_mesh(args.scene)
    scene_tree = cKDTree(scene_points)
    summary_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for mode in args.modes:
        points, _windows, diag = _load_aligned_d4rt_scene_points(
            args.scene,
            debug_root=args.debug_root,
            mode=mode,
            max_points=0,
            min_visibility=args.min_visibility,
            min_confidence=args.min_confidence,
        )
        dist = _nn_dist(scene_tree, scene_points.shape[0], points)
        summary_rows.append(
            {
                "scene": args.scene,
                "mode": mode,
                "point_count": int(points.shape[0]),
                **_dist_stats(dist, prefix="mesh_nn"),
                **_bbox_stats(points, prefix="d4rt"),
                "diag": diag,
            }
        )
        provider = D4RTCarrierProjectionProvider(
            debug_root=project(args.debug_root),
            mode=mode,
            max_anchors=args.max_anchors,
            robust_trim_percentile=args.robust_trim_percentile,
            min_visibility=args.min_visibility,
            min_confidence=args.min_confidence,
        )
        old_cwd = Path.cwd()
        try:
            # Provider internals resolve ScanNet stream paths relative to the Stream3D root.
            import os

            os.chdir(project("."))
            cache = provider._load_scene(args.scene)
        finally:
            os.chdir(old_cwd)
        for window_index, window in enumerate(cache["windows"]):
            parts: list[np.ndarray] = []
            for local_idx, frame_id in enumerate(window.frame_ids):
                xyz = _apply_fit(window.xyz[local_idx], window.transform)
                uv = window.uv[local_idx]
                ok = _valid_mask(window, local_idx, uv, xyz, args.min_visibility, args.min_confidence)
                if np.any(ok):
                    parts.append(xyz[ok])
            pts = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float32)
            row = {
                "scene": args.scene,
                "mode": mode,
                "window_index": int(window_index),
                "path": rel(window.path),
                "frame_ids": ",".join(str(int(x)) for x in window.frame_ids),
                "point_count": int(pts.shape[0]),
                **_dist_stats(_nn_dist(scene_tree, scene_points.shape[0], pts), prefix="mesh_nn"),
                **_bbox_stats(pts, prefix="d4rt"),
            }
            window_rows.append(row)

    anchor_rows = _uv_variant_anchor_rows(
        args.scene,
        project(args.debug_root),
        max_anchors=args.max_anchors,
        robust_trim_percentile=args.robust_trim_percentile,
        min_visibility=args.min_visibility,
        min_confidence=args.min_confidence,
    )

    pred_overlay = _load_prediction_overlay(args.scene, args.pred_config, scene_points.shape[0])
    pred_points = np.asarray(pred_overlay.get("_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    pred_dist = _nn_dist(scene_tree, scene_points.shape[0], pred_points)
    final_mode = "self_stitched_scale_normalized_eval_sim3"
    final_d4rt_points, _final_windows, _final_diag = _load_aligned_d4rt_scene_points(
        args.scene,
        debug_root=args.debug_root,
        mode=final_mode,
        max_points=0,
        min_visibility=args.min_visibility,
        min_confidence=args.min_confidence,
    )
    pred_summary = {
        "pred_config": args.pred_config,
        "pred_path": pred_overlay.get("pred_path", ""),
        "pred_path_sha256": pred_overlay.get("pred_path_sha256", ""),
        "pred_pre_points_path": pred_overlay.get("pre_points_path", ""),
        "pred_pre_points_path_sha256": pred_overlay.get("pre_points_path_sha256", ""),
        "pred_pre_points_count": pred_overlay.get("pre_points_count", 0),
        "pred_vertex_count": pred_overlay.get("pred_vertex_count", 0),
        "pred_instance_count": pred_overlay.get("pred_instance_count", 0),
        "pred_mask_contract": pred_overlay.get("mask_contract", ""),
        **_dist_stats(pred_dist, prefix="pred_sem_mesh_nn"),
        **_cross_dist_stats(final_d4rt_points, pred_points, left_name="final_d4rt", right_name="pred_sem"),
        **_bbox_stats(pred_points, prefix="pred_sem"),
    }

    summary = {
        "phase": "v65_d4rt_geometry_alignment_diagnostic",
        "scene": args.scene,
        "debug_root": rel(args.debug_root),
        "mesh_path": rel(mesh_path),
        "mesh_path_sha256": sha256_file(mesh_path),
        "mesh_point_count": int(scene_points.shape[0]),
        "modes": args.modes,
        "min_visibility": float(args.min_visibility),
        "min_confidence": float(args.min_confidence),
        "max_anchors": int(args.max_anchors),
        "robust_trim_percentile": float(args.robust_trim_percentile),
        "mode_rows": summary_rows,
        "anchor_variant_rows": anchor_rows,
        "pred_sem_summary": pred_summary,
        "provider_probe5_rows": _provider_probe5_rows(),
        "ap_failure_attribution": _ap_failure_attribution(args.scene),
        "interpretation_notes": [
            "mesh_nn distances measure geometric overlay against the ScanNet mesh vertices.",
            "eval_sim3 modes use ScanNet depth/pose anchors and remain diagnostic-only.",
            "pred_sem points are loaded from the AP evaluator pred_masks overlay; they should lie exactly on mesh vertices when mask_contract is full_scene_vertex_mask.",
        ],
    }
    write_json(out / "alignment_summary.json", summary)
    _write_csv(out / "mode_alignment_rows.csv", _flatten_rows(summary_rows))
    _write_csv(out / "window_alignment_rows.csv", _flatten_rows(window_rows))
    _write_csv(out / "uv_anchor_variant_rows.csv", _flatten_rows(anchor_rows))
    print(json.dumps(summary, indent=2, sort_keys=True))


def _valid_mask(window: Any, local_idx: int, uv: np.ndarray, xyz: np.ndarray, min_visibility: float, min_confidence: float) -> np.ndarray:
    return (
        window.valid[local_idx]
        & np.isfinite(xyz).all(axis=1)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= 1.0)
        & (window.visibility[local_idx] >= float(min_visibility))
        & (window.confidence[local_idx] >= float(min_confidence))
    )


def _nn_dist(tree: cKDTree, point_count: int, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    dist, idx = tree.query(points, k=1)
    valid = np.isfinite(dist) & (idx < int(point_count))
    return np.asarray(dist[valid], dtype=np.float64)


def _dist_stats(dist: np.ndarray, *, prefix: str) -> dict[str, Any]:
    dist = np.asarray(dist, dtype=np.float64)
    dist = dist[np.isfinite(dist)]
    out: dict[str, Any] = {f"{prefix}_count": int(dist.shape[0])}
    if dist.size == 0:
        for key in ["mean", "median", "p75", "p90", "p95", "p99", "max"]:
            out[f"{prefix}_{key}"] = None
        for threshold in [0.02, 0.05, 0.10, 0.20, 0.50, 1.00]:
            out[f"{prefix}_frac_le_{str(threshold).replace('.', 'p')}m"] = None
        return out
    out.update(
        {
            f"{prefix}_mean": float(np.mean(dist)),
            f"{prefix}_median": float(np.median(dist)),
            f"{prefix}_p75": float(np.percentile(dist, 75)),
            f"{prefix}_p90": float(np.percentile(dist, 90)),
            f"{prefix}_p95": float(np.percentile(dist, 95)),
            f"{prefix}_p99": float(np.percentile(dist, 99)),
            f"{prefix}_max": float(np.max(dist)),
        }
    )
    for threshold in [0.02, 0.05, 0.10, 0.20, 0.50, 1.00]:
        out[f"{prefix}_frac_le_{str(threshold).replace('.', 'p')}m"] = float(np.mean(dist <= threshold))
    return out


def _bbox_stats(points: np.ndarray, *, prefix: str) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return {
            f"{prefix}_centroid": None,
            f"{prefix}_bbox_min": None,
            f"{prefix}_bbox_max": None,
            f"{prefix}_bbox_extent": None,
        }
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return {
        f"{prefix}_centroid": points.mean(axis=0).astype(float).tolist(),
        f"{prefix}_bbox_min": mins.astype(float).tolist(),
        f"{prefix}_bbox_max": maxs.astype(float).tolist(),
        f"{prefix}_bbox_extent": (maxs - mins).astype(float).tolist(),
    }


def _cross_dist_stats(left: np.ndarray, right: np.ndarray, *, left_name: str, right_name: str) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float32).reshape(-1, 3)
    right = np.asarray(right, dtype=np.float32).reshape(-1, 3)
    left = left[np.isfinite(left).all(axis=1)]
    right = right[np.isfinite(right).all(axis=1)]
    if left.shape[0] == 0 or right.shape[0] == 0:
        return {
            f"{left_name}_to_{right_name}_available": False,
            f"{right_name}_to_{left_name}_available": False,
        }
    right_tree = cKDTree(right)
    left_tree = cKDTree(left)
    left_to_right, _ = right_tree.query(left, k=1)
    right_to_left, _ = left_tree.query(right, k=1)
    return {
        f"{left_name}_point_count": int(left.shape[0]),
        f"{right_name}_point_count": int(right.shape[0]),
        **_dist_stats(left_to_right, prefix=f"{left_name}_to_{right_name}_nn"),
        **_dist_stats(right_to_left, prefix=f"{right_name}_to_{left_name}_nn"),
    }


def _uv_variant_anchor_rows(
    scene: str,
    debug_root: Path,
    *,
    max_anchors: int,
    robust_trim_percentile: float,
    min_visibility: float,
    min_confidence: float,
) -> list[dict[str, Any]]:
    import os

    old_cwd = Path.cwd()
    rows: list[dict[str, Any]] = []
    variants = []
    for swap in [False, True]:
        for flip_x in [False, True]:
            for flip_y in [False, True]:
                variants.append((swap, flip_x, flip_y))
    try:
        os.chdir(project("."))
        stream = ScanNetStream(seq_name=scene, backbone="Cropformer")
        for swap, flip_x, flip_y in variants:
            source_parts: list[np.ndarray] = []
            target_parts: list[np.ndarray] = []
            total = 0
            valid_total = 0
            for carrier_path in sorted((debug_root / scene).glob("carriers_window*.npz")):
                with np.load(carrier_path) as data:
                    xyz = np.asarray(data["xyz_ref"], dtype=np.float32)
                    uv = np.asarray(data["uv_pred"], dtype=np.float32)
                    valid = np.asarray(data.get("valid", np.ones(xyz.shape[:2], dtype=bool)), dtype=bool)
                    visibility = np.asarray(data.get("visibility_prob", np.ones(xyz.shape[:2], dtype=np.float32)), dtype=np.float32)
                    confidence = np.asarray(data.get("confidence_prob", np.ones(xyz.shape[:2], dtype=np.float32)), dtype=np.float32)
                    frame_ids = _frame_ids_for_window(carrier_path, data, int(uv.shape[0]))
                    for local_idx, frame_id in enumerate(frame_ids):
                        h, w = stream.load_depth(int(frame_id)).shape[:2]
                        uv_local = _uv_variant(uv[local_idx], swap=swap, flip_x=flip_x, flip_y=flip_y)
                        ok = (
                            valid[local_idx]
                            & np.isfinite(xyz[local_idx]).all(axis=1)
                            & np.isfinite(uv_local).all(axis=1)
                            & (uv_local[:, 0] >= 0.0)
                            & (uv_local[:, 0] <= 1.0)
                            & (uv_local[:, 1] >= 0.0)
                            & (uv_local[:, 1] <= 1.0)
                            & (visibility[local_idx] >= float(min_visibility))
                            & (confidence[local_idx] >= float(min_confidence))
                        )
                        total += int(ok.shape[0])
                        if not np.any(ok):
                            continue
                        xy = np.stack(
                            [
                                uv_local[ok, 0] * float(max(w - 1, 1)),
                                uv_local[ok, 1] * float(max(h - 1, 1)),
                            ],
                            axis=1,
                        )
                        world, world_ok = backproject_xy_world(stream, int(frame_id), xy)
                        if not np.any(world_ok):
                            continue
                        source_parts.append(xyz[local_idx, ok][world_ok])
                        target_parts.append(world[world_ok])
                        valid_total += int(np.count_nonzero(world_ok))
            name = f"{'yx' if swap else 'xy'}{'_flipx' if flip_x else ''}{'_flipy' if flip_y else ''}"
            if not source_parts:
                rows.append(
                    {
                        "scene": scene,
                        "uv_variant": name,
                        "anchor_candidates": int(total),
                        "anchor_valid": int(valid_total),
                        "fit_available": False,
                    }
                )
                continue
            source = np.concatenate(source_parts, axis=0)
            target = np.concatenate(target_parts, axis=0)
            if source.shape[0] > int(max_anchors):
                keep = np.linspace(0, source.shape[0] - 1, num=int(max_anchors), dtype=np.int64)
                source = source[keep]
                target = target[keep]
            fit = fit_transform(source, target, robust_trim_percentile=float(robust_trim_percentile))
            residual = np.asarray([] if fit is None else fit.get("residual", []), dtype=np.float64)
            rows.append(
                {
                    "scene": scene,
                    "uv_variant": name,
                    "swap": bool(swap),
                    "flip_x": bool(flip_x),
                    "flip_y": bool(flip_y),
                    "anchor_candidates": int(total),
                    "anchor_valid": int(valid_total),
                    "fit_available": fit is not None,
                    "sim3_scale": None if fit is None else float(fit["scale"]),
                    "rotation_det": None if fit is None else float(fit.get("rotation_det", np.nan)),
                    **_dist_stats(residual, prefix="anchor_residual"),
                }
            )
    finally:
        os.chdir(old_cwd)
    return rows


def _provider_probe5_rows() -> list[dict[str, Any]]:
    path = project("outputs/audit/v64r2_d4rt_chunk_scale_first_ap_probe5/D4RT_geometry_replacement_stream3d_probe5.csv")
    if not path.exists():
        return []
    keep = {
        "variant",
        "label",
        "mode",
        "geometry_source",
        "ap",
        "ap50",
        "ap25",
        "projection_hit_rate_mean",
        "mask_projection_empty_rate_mean",
        "positive_mask_point_count_mean",
        "hit_point_count_mean",
        "nn_radius_mean",
        "pre_points_ratio",
        "prediction_union_ratio",
        "num_pred_per_scene",
        "uses_gt_sim3_for_prediction",
        "uses_d4rt_self_sim3",
        "sim3_residual_median_mean",
        "sim3_residual_p90_mean",
        "status",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [{key: _number_or_text(value) for key, value in row.items() if key in keep} for row in csv.DictReader(handle)]
    return rows


def _ap_failure_attribution(scene: str) -> dict[str, Any]:
    root = project("outputs/audit/v65_ap_failure_decomp")
    failure_path = root / "failure_rows.csv"
    fragmentation_path = root / "fragmentation_rows.csv"
    summary_path = root / "failure_summary.json"
    if not failure_path.exists() or not fragmentation_path.exists():
        return {
            "available": False,
            "failure_rows_path": rel(failure_path),
            "fragmentation_rows_path": rel(fragmentation_path),
        }
    with failure_path.open(newline="", encoding="utf-8") as handle:
        failure_rows = list(csv.DictReader(handle))
    with fragmentation_path.open(newline="", encoding="utf-8") as handle:
        fragmentation_rows = list(csv.DictReader(handle))
    d4rt_ids = {"A5", "A6", "A7", "A8"}
    out: dict[str, Any] = {
        "available": True,
        "failure_summary_path": rel(summary_path),
        "failure_rows_path": rel(failure_path),
        "fragmentation_rows_path": rel(fragmentation_path),
        "global_failure_summary": json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {},
        "by_row_id": {},
        "scene": scene,
    }
    for row_id in sorted(d4rt_ids):
        rows = [row for row in failure_rows if row.get("variant_row_id") == row_id]
        scene_rows = [row for row in rows if row.get("scene_id") == scene]
        fragments = [row for row in fragmentation_rows if row.get("variant_row_id") == row_id]
        scene_fragments = [row for row in fragments if row.get("scene_id") == scene]
        out["by_row_id"][row_id] = {
            "global_failure_count": int(len(rows)),
            "global_failure_category_counts": dict(Counter(row.get("failure_category", "") for row in rows)),
            "scene_failure_count": int(len(scene_rows)),
            "scene_failure_category_counts": dict(Counter(row.get("failure_category", "") for row in scene_rows)),
            "scene_fragmentation_rows": [_coerce_row(row) for row in scene_fragments],
        }
    return out


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _number_or_text(value) for key, value in row.items()}


def _number_or_text(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if text == "":
        return ""
    if text in {"True", "False"}:
        return text == "True"
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _uv_variant(uv: np.ndarray, *, swap: bool, flip_x: bool, flip_y: bool) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float32)
    out = uv[:, [1, 0]].copy() if swap else uv.copy()
    if flip_x:
        out[:, 0] = 1.0 - out[:, 0]
    if flip_y:
        out[:, 1] = 1.0 - out[:, 1]
    return out


def _frame_ids_for_window(carrier_path: Path, data: Any, num_frames: int) -> list[int]:
    manifest_path = carrier_path.with_name(carrier_path.stem + "_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        frame_ids = manifest.get("frame_ids")
        if isinstance(frame_ids, list) and len(frame_ids) == num_frames:
            return [int(x) for x in frame_ids]
    src_frame = np.asarray(data.get("src_frame", []), dtype=np.int64)
    src_global = np.asarray(data.get("src_frame_global", []), dtype=np.int64)
    if src_frame.shape == src_global.shape and src_frame.size:
        out: list[int] = []
        for local_idx in range(num_frames):
            vals = src_global[src_frame == local_idx]
            if vals.size:
                uniq, counts = np.unique(vals, return_counts=True)
                out.append(int(uniq[np.argmax(counts)]))
            else:
                out.append(int(local_idx))
        return out
    return list(range(num_frames))


def _flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat_rows = []
    for row in rows:
        flat: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                flat[key] = json.dumps(value, sort_keys=True)
            else:
                flat[key] = value
        flat_rows.append(flat)
    return flat_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
