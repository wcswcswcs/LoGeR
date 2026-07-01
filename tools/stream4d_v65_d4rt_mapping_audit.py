#!/usr/bin/env python3
"""Audit D4RT/SOMA mask-to-ScanNet-vertex materialization.

The goal is to inspect the mapping chain before AP is computed.  It compares
nearest-1 and k-neighbor radius assignment from D4RT positive mask samples to
ScanNet scene vertices, without running Stream3D clustering.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from dataset.scannet import ScanNetDataset
from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider, _apply_fit
from stream4d.scannet_stream import ScanNetStream


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _hash_path(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _existing_mask_frames(scene: str, backbone: str) -> set[int]:
    mask_dir = Path("data/scannet/processed") / scene / f"output_{backbone}" / "mask"
    return {int(path.stem) for path in mask_dir.glob("*.png")}


def _debug_frames(debug_root: Path, scene: str) -> set[int]:
    out: set[int] = set()
    for manifest in sorted((debug_root / scene).glob("*_manifest.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("raw_frame_ids", "frame_indices", "frame_ids"):
            values = payload.get(key)
            if values:
                out.update(int(v) for v in values)
                break
    return out


def _sample_mask(mask_dir: Path, frame_id: int) -> np.ndarray:
    path = mask_dir / f"{int(frame_id)}.png"
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _collect_positive_samples(
    provider: D4RTCarrierProjectionProvider,
    *,
    scene: str,
    frame_id: int,
    mask_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache = provider._load_scene(scene)
    height, width = mask_np.shape[:2]
    points_parts: list[np.ndarray] = []
    mask_id_parts: list[np.ndarray] = []
    local_points = 0
    positive_points = 0
    selected_windows = provider._windows_for_frame(cache["windows"], int(frame_id))
    for window, local_idx in selected_windows:
        xyz = _apply_fit(window.xyz[local_idx], window.transform)
        uv = window.uv[local_idx]
        ok = (
            window.valid[local_idx]
            & np.isfinite(xyz).all(axis=1)
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= 1.0)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= 1.0)
            & (window.visibility[local_idx] >= provider.min_visibility)
            & (window.confidence[local_idx] >= provider.min_confidence)
        )
        local_points += int(np.count_nonzero(ok))
        if not np.any(ok):
            continue
        x = np.rint(uv[ok, 0] * float(max(width - 1, 1))).astype(np.int64)
        y = np.rint(uv[ok, 1] * float(max(height - 1, 1))).astype(np.int64)
        in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not np.any(in_bounds):
            continue
        points = xyz[ok][in_bounds]
        mask_ids = mask_np[y[in_bounds], x[in_bounds]].astype(np.int64)
        positive = mask_ids > 0
        positive_points += int(np.count_nonzero(positive))
        if not np.any(positive):
            continue
        points_parts.append(points[positive])
        mask_id_parts.append(mask_ids[positive])
    if points_parts:
        points = np.concatenate(points_parts, axis=0).astype(np.float32)
        mask_ids = np.concatenate(mask_id_parts, axis=0).astype(np.int64)
    else:
        points = np.empty((0, 3), dtype=np.float32)
        mask_ids = np.empty((0,), dtype=np.int64)
    return points, mask_ids, {
        "local_point_count": int(local_points),
        "positive_mask_point_count": int(positive_points),
        "selected_source_windows": int(len(selected_windows)),
        **cache.get("scene_fit", {}),
        **cache.get("stitch_diag", {}),
    }


def _assign_vertices(
    tree: cKDTree,
    scene_point_count: int,
    points: np.ndarray,
    mask_ids: np.ndarray,
    *,
    radius: float,
    k: int,
) -> tuple[dict[int, set[int]], dict[str, Any]]:
    mask_to_points: dict[int, set[int]] = defaultdict(set)
    if points.shape[0] == 0:
        return mask_to_points, {
            "hit_sample_count": 0,
            "unique_vertex_count": 0,
            "hit_rate": 0.0,
            "distance_median": None,
            "distance_p90": None,
        }
    dist, idx = tree.query(points, k=int(k), distance_upper_bound=float(radius))
    dist = np.asarray(dist)
    idx = np.asarray(idx)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    valid = np.isfinite(dist) & (idx >= 0) & (idx < scene_point_count)
    hit_samples = np.any(valid, axis=1)
    for row, mask_id in enumerate(mask_ids.tolist()):
        cols = idx[row][valid[row]]
        if cols.size:
            mask_to_points[int(mask_id)].update(int(v) for v in cols.tolist())
    hit_dists = dist[valid]
    unique_vertices = set()
    for values in mask_to_points.values():
        unique_vertices.update(values)
    return mask_to_points, {
        "hit_sample_count": int(np.count_nonzero(hit_samples)),
        "unique_vertex_count": int(len(unique_vertices)),
        "hit_rate": float(np.count_nonzero(hit_samples) / max(points.shape[0], 1)),
        "distance_median": float(np.median(hit_dists)) if hit_dists.size else None,
        "distance_p90": float(np.percentile(hit_dists, 90)) if hit_dists.size else None,
        "mask_ids_with_vertices": int(len(mask_to_points)),
        "mean_vertices_per_mask": float(np.mean([len(v) for v in mask_to_points.values()])) if mask_to_points else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--debug-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--mode", default="self_stitched_eval_sim3")
    parser.add_argument("--frame-stride", default=5, type=int)
    parser.add_argument("--min-visibility", default=0.0, type=float)
    parser.add_argument("--min-confidence", default=0.2, type=float)
    parser.add_argument("--max-anchors", default=120000, type=int)
    parser.add_argument("--robust-trim-percentile", default=90.0, type=float)
    parser.add_argument("--stitch-uv-radius", default=0.002, type=float)
    parser.add_argument("--stitch-max-matches-per-frame", default=4096, type=int)
    parser.add_argument("--stitch-fit-trim-percentile", default=90.0, type=float)
    parser.add_argument("--overlap-policy", default="best_confidence")
    parser.add_argument("--radii", default="0.03,0.05,0.075,0.1,0.15,0.2")
    parser.add_argument("--ks", default="1,5,20")
    parser.add_argument("--max-frames", default=0, type=int, help="0 means all selected frames")
    args = parser.parse_args()

    stream = ScanNetStream(seq_name=args.scene, backbone=args.backbone)
    dataset = ScanNetDataset(args.scene, args.backbone)
    scene_points = np.asarray(dataset.get_scene_points(), dtype=np.float32)
    scene_tree = cKDTree(scene_points)
    provider = D4RTCarrierProjectionProvider(
        debug_root=args.debug_root,
        mode=args.mode,
        nn_radius=0.05,
        min_visibility=args.min_visibility,
        min_confidence=args.min_confidence,
        max_anchors=args.max_anchors,
        robust_trim_percentile=args.robust_trim_percentile,
        overlap_policy=args.overlap_policy,
        stitch_uv_radius=args.stitch_uv_radius,
        stitch_max_matches_per_frame=args.stitch_max_matches_per_frame,
        stitch_fit_trim_percentile=args.stitch_fit_trim_percentile,
    )

    frames = set(dataset.get_frame_list(args.frame_stride))
    debug_frames = _debug_frames(args.debug_root, args.scene)
    if not debug_frames:
        cache = provider._load_scene(args.scene)
        for window in cache["windows"]:
            debug_frames.update(int(v) for v in window.frame_ids)
    frames &= debug_frames
    frames &= _existing_mask_frames(args.scene, args.backbone)
    frame_ids = sorted(frames)
    if args.max_frames > 0:
        frame_ids = frame_ids[: int(args.max_frames)]

    radii = [float(item) for item in args.radii.split(",") if item.strip()]
    ks = [int(item) for item in args.ks.split(",") if item.strip()]
    aggregate: dict[str, dict[str, Any]] = {
        f"r{radius:g}_k{k}": {
            "radius": float(radius),
            "k": int(k),
            "positive_sample_count": 0,
            "hit_sample_count": 0,
            "unique_vertex_ids": set(),
            "mask_vertex_counts": [],
            "distance_values": [],
        }
        for radius in radii
        for k in ks
    }
    frame_rows: list[dict[str, Any]] = []
    mask_dir = Path("data/scannet/processed") / args.scene / f"output_{args.backbone}" / "mask"

    for frame_id in frame_ids:
        mask_np = _sample_mask(mask_dir, frame_id)
        points, mask_ids, base_diag = _collect_positive_samples(
            provider,
            scene=args.scene,
            frame_id=int(frame_id),
            mask_np=mask_np,
        )
        row: dict[str, Any] = {
            "scene": args.scene,
            "frame_id": int(frame_id),
            "positive_sample_count": int(points.shape[0]),
            **base_diag,
        }
        for radius in radii:
            for k in ks:
                key = f"r{radius:g}_k{k}"
                mask_to_points, stats = _assign_vertices(
                    scene_tree,
                    scene_points.shape[0],
                    points,
                    mask_ids,
                    radius=radius,
                    k=k,
                )
                row[key] = stats
                agg = aggregate[key]
                agg["positive_sample_count"] += int(points.shape[0])
                agg["hit_sample_count"] += int(stats["hit_sample_count"])
                for values in mask_to_points.values():
                    agg["unique_vertex_ids"].update(values)
                    agg["mask_vertex_counts"].append(len(values))
                if stats["distance_median"] is not None:
                    # Store frame medians/p90s, not every point, to keep output compact.
                    agg["distance_values"].append(float(stats["distance_median"]))
        frame_rows.append(row)

    aggregate_out = {}
    for key, value in aggregate.items():
        counts = value["mask_vertex_counts"]
        dist_vals = value["distance_values"]
        aggregate_out[key] = {
            "radius": value["radius"],
            "k": value["k"],
            "positive_sample_count": int(value["positive_sample_count"]),
            "hit_sample_count": int(value["hit_sample_count"]),
            "hit_rate": float(value["hit_sample_count"] / max(value["positive_sample_count"], 1)),
            "unique_vertex_count": int(len(value["unique_vertex_ids"])),
            "unique_vertex_ratio": float(len(value["unique_vertex_ids"]) / max(scene_points.shape[0], 1)),
            "mask_vertex_count_mean": float(np.mean(counts)) if counts else 0.0,
            "mask_vertex_count_median": float(np.median(counts)) if counts else 0.0,
            "frame_distance_median_median": float(np.median(dist_vals)) if dist_vals else None,
        }

    output = {
        "scene": args.scene,
        "scene_vertex_count": int(scene_points.shape[0]),
        "debug_root": str(args.debug_root),
        "debug_root_exists": bool(args.debug_root.exists()),
        "mask_dir": str(mask_dir),
        "frame_count": int(len(frame_ids)),
        "frame_ids": frame_ids,
        "mode": args.mode,
        "frame_stride": int(args.frame_stride),
        "min_visibility": float(args.min_visibility),
        "min_confidence": float(args.min_confidence),
        "overlap_policy": args.overlap_policy,
        "stitch_uv_radius": float(args.stitch_uv_radius),
        "stitch_max_matches_per_frame": int(args.stitch_max_matches_per_frame),
        "stitch_fit_trim_percentile": float(args.stitch_fit_trim_percentile),
        "aggregate": aggregate_out,
        "provider_scene_cache": {
            "scene_fit": provider._load_scene(args.scene).get("scene_fit", {}),
            "stitch_diag": provider._load_scene(args.scene).get("stitch_diag", {}),
            "anchor_diag": provider._load_scene(args.scene).get("anchor_diag", {}),
        },
        "frame_rows": frame_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(_json_safe(output), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_json_sha256": _hash_path(args.output_json),
        "scene": args.scene,
        "frame_count": len(frame_ids),
        "aggregate": aggregate_out,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
