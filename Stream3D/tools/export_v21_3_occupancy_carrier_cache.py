from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.carrier_sampler import CarrierSampler
from stream4d.carrier_store import CarrierBatch
from stream4d.d4rt_adapter import D4RTAdapter
from stream4d.scannet_stream import ScanNetStream
from stream4d_native.occupancy_dense_tracker import (
    QueryBudget,
    query_d4rt_tubes_with_spatiotemporal_occupancy,
)
from stream4d_native.d4rt_scene_builder import source_xy_from_uv, stable_source_carrier_id
from stream4d_native.occupancy_state import OccupancyCoverageTargets
from tools.run_v21_3_native_occupancy_ablation import (
    _frame_ids,
    _tracks_from_batch,
    _windows,
)


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


def _load_rgb_sparse_mask_window(stream: ScanNetStream, frame_ids: list[int]) -> dict[str, Any]:
    rgbs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    present: list[int] = []
    missing: list[int] = []
    for frame_id in frame_ids:
        rgb = stream.load_rgb(int(frame_id))
        rgbs.append(rgb)
        try:
            mask = stream.load_mask(int(frame_id))
            present.append(int(frame_id))
        except FileNotFoundError:
            mask = np.zeros(rgb.shape[:2], dtype=np.int32)
            missing.append(int(frame_id))
        masks.append(np.asarray(mask))
    return {
        "rgb": np.stack(rgbs, axis=0),
        "mask": np.stack(masks, axis=0),
        "sparse_mask_present_frame_ids": present,
        "sparse_mask_missing_frame_ids": missing,
    }


def _write_manifest(path: Path, *, frame_ids: list[int], variant: str, scene: str, window_index: int) -> None:
    payload = {
        "scene": scene,
        "variant": variant,
        "window_index": int(window_index),
        "raw_frame_ids": [int(v) for v in frame_ids],
        "frame_ids": [int(v) for v in frame_ids],
        "eval_policy": "v21_3_occupancy_carrier_cache_diagnostic",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _save_batch(
    *,
    batch: CarrierBatch,
    out_dir: Path,
    frame_ids: list[int],
    variant: str,
    scene: str,
    window_index: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"carriers_window{window_index:03d}.npz"
    batch.save_npz(npz_path)
    _write_manifest(
        npz_path.with_name(f"{npz_path.stem}_manifest.json"),
        frame_ids=frame_ids,
        variant=variant,
        scene=scene,
        window_index=window_index,
    )
    valid = np.asarray(batch.valid, dtype=bool)
    uv = np.asarray(batch.uv_pred, dtype=np.float32)
    in_bounds = (
        valid
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] <= 1.0)
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] <= 1.0)
    )
    return {
        "path": str(npz_path),
        "window_index": int(window_index),
        "frame_start": int(frame_ids[0]) if frame_ids else None,
        "frame_end": int(frame_ids[-1]) if frame_ids else None,
        "num_frames": int(len(frame_ids)),
        "num_carriers": int(batch.carrier_id.shape[0]),
        "persistent_tube_count": int(np.unique(batch.persistent_tube_id).shape[0])
        if batch.persistent_tube_id is not None and batch.persistent_tube_id.size
        else 0,
        "warmstarted_carrier_count": int(np.count_nonzero(batch.is_warmstarted))
        if batch.is_warmstarted is not None
        else 0,
        "uv_in01_rate": float(np.mean(in_bounds)) if in_bounds.size else 0.0,
        "valid_rate": float(np.mean(valid)) if valid.size else 0.0,
    }


def _track_source_frame(track: dict[str, Any], frame_count: int) -> int:
    value = int(track.get("source_frame", 0))
    return min(max(value, 0), max(frame_count - 1, 0))


def _source_arrays_from_points(
    *,
    source_points: np.ndarray,
    frame_ids: list[int],
    image_width: int,
    image_height: int,
    masks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_points = np.asarray(source_points, dtype=np.float32)
    src_frame = source_points[:, 0].astype(np.int64)
    src_xy = np.zeros((source_points.shape[0], 2), dtype=np.int64)
    src_frame_global = np.zeros((source_points.shape[0],), dtype=np.int64)
    src_mask_id = np.full((source_points.shape[0],), -1, dtype=np.int64)
    frame_arr = np.asarray(frame_ids, dtype=np.int64)
    for idx, local_frame in enumerate(src_frame.tolist()):
        clamped_frame = int(np.clip(local_frame, 0, max(len(frame_ids) - 1, 0)))
        src_frame_global[idx] = int(frame_arr[clamped_frame]) if frame_arr.size else int(clamped_frame)
        x, y = source_xy_from_uv(
            source_points[idx, 1:3],
            image_width=int(image_width),
            image_height=int(image_height),
        )
        src_xy[idx] = (int(x), int(y))
        if masks is not None and masks.size:
            src_mask_id[idx] = int(np.asarray(masks)[clamped_frame, int(y), int(x)])
    return src_frame_global, src_xy, src_mask_id


def _tracks_to_batch(tracks: list[dict[str, Any]], frame_ids: list[int]) -> CarrierBatch:
    num_frames = int(len(frame_ids))
    num_tracks = int(len(tracks))
    xyz = np.empty((num_frames, num_tracks, 3), dtype=np.float32)
    xyz_local = np.empty((num_frames, num_tracks, 3), dtype=np.float32)
    uv = np.empty((num_frames, num_tracks, 2), dtype=np.float32)
    visibility = np.zeros((num_frames, num_tracks), dtype=np.float32)
    confidence = np.zeros((num_frames, num_tracks), dtype=np.float32)
    valid = np.zeros((num_frames, num_tracks), dtype=bool)
    xyz.fill(np.nan)
    xyz_local.fill(np.nan)
    uv.fill(np.nan)
    carrier_id = np.empty((num_tracks,), dtype=np.int64)
    src_frame = np.zeros((num_tracks,), dtype=np.int64)
    src_frame_global = np.zeros((num_tracks,), dtype=np.int64)
    src_xy = np.full((num_tracks, 2), -1, dtype=np.int64)
    src_uv = np.zeros((num_tracks, 2), dtype=np.float32)
    src_mask_id = np.full((num_tracks,), -1, dtype=np.int64)
    persistent_tube_id = np.full((num_tracks,), -1, dtype=np.int64)
    parent_tube_id = np.full((num_tracks,), -1, dtype=np.int64)
    warmstart_source_chunk = np.full((num_tracks,), -1, dtype=np.int64)
    warmstart_source_frame = np.full((num_tracks,), -1, dtype=np.int64)
    is_warmstarted = np.zeros((num_tracks,), dtype=bool)
    has_xyz_local = False
    for idx, track in enumerate(tracks):
        xyz[:, idx, :] = np.asarray(track.get("xyz", np.full((num_frames, 3), np.nan)), dtype=np.float32)
        if track.get("xyz_local") is not None:
            xyz_local[:, idx, :] = np.asarray(track["xyz_local"], dtype=np.float32)
            has_xyz_local = True
        uv[:, idx, :] = np.asarray(track.get("uv_norm", np.full((num_frames, 2), np.nan)), dtype=np.float32)
        visibility[:, idx] = np.asarray(track.get("visibility", np.zeros((num_frames,))), dtype=np.float32)
        confidence[:, idx] = np.asarray(track.get("confidence", np.zeros((num_frames,))), dtype=np.float32)
        valid[:, idx] = np.asarray(track.get("valid", np.zeros((num_frames,), dtype=bool)), dtype=bool)
        carrier_id[idx] = int(track.get("carrier_id", idx))
        src_frame[idx] = _track_source_frame(track, num_frames)
        src_frame_global[idx] = int(track.get("source_frame_global", frame_ids[src_frame[idx]] if frame_ids else 0))
        if track.get("source_xy") is not None:
            src_xy[idx] = np.asarray(track["source_xy"], dtype=np.int64).reshape(2)
        src_uv[idx] = np.asarray(track.get("source_uv", [0.0, 0.0]), dtype=np.float32)
        src_mask_id[idx] = int(track.get("src_mask_id", track.get("source_mask_id", -1)))
        persistent_tube_id[idx] = int(track.get("persistent_tube_id", carrier_id[idx]))
        parent_tube_id[idx] = int(track.get("parent_tube_id", -1))
        warmstart_source_chunk[idx] = int(track.get("warmstart_source_chunk", -1))
        warmstart_source_frame[idx] = int(track.get("warmstart_source_frame", -1))
        is_warmstarted[idx] = bool(track.get("is_warmstarted", False))
    return CarrierBatch(
        carrier_id=carrier_id,
        src_frame=src_frame,
        src_uv=src_uv,
        xyz_ref=xyz,
        uv_pred=uv,
        visibility_prob=visibility,
        confidence_prob=confidence,
        valid=valid,
        xyz_local=xyz_local if has_xyz_local else None,
        src_frame_global=src_frame_global,
        src_xy=src_xy,
        src_mask_id=src_mask_id,
        persistent_tube_id=persistent_tube_id,
        parent_tube_id=parent_tube_id,
        warmstart_source_chunk=warmstart_source_chunk,
        warmstart_source_frame=warmstart_source_frame,
        is_warmstarted=is_warmstarted,
    )


def _tracks_from_batch_with_source(
    batch: CarrierBatch,
    source_points: np.ndarray,
    carrier_ids: np.ndarray,
    *,
    frame_ids: list[int],
    image_width: int,
    image_height: int,
    masks: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    tracks = _tracks_from_batch(batch)
    source_points = np.asarray(source_points, dtype=np.float32)
    carrier_ids = np.asarray(carrier_ids, dtype=np.int64)
    fallback_frame_global, fallback_src_xy, fallback_mask_id = _source_arrays_from_points(
        source_points=source_points,
        frame_ids=frame_ids,
        image_width=int(image_width),
        image_height=int(image_height),
        masks=masks,
    )
    batch_frame_global = getattr(batch, "src_frame_global", None)
    batch_src_xy = getattr(batch, "src_xy", None)
    batch_src_mask_id = getattr(batch, "src_mask_id", None)
    batch_xyz_local = getattr(batch, "xyz_local", None)
    for idx, track in enumerate(tracks):
        track["source_frame"] = int(source_points[idx, 0])
        track["source_uv"] = np.asarray(source_points[idx, 1:3], dtype=np.float32)
        track["source_frame_global"] = (
            int(np.asarray(batch_frame_global, dtype=np.int64).reshape(-1)[idx])
            if batch_frame_global is not None
            else int(fallback_frame_global[idx])
        )
        track["source_xy"] = (
            tuple(int(v) for v in np.asarray(batch_src_xy, dtype=np.int64).reshape(-1, 2)[idx].tolist())
            if batch_src_xy is not None
            else tuple(int(v) for v in fallback_src_xy[idx].tolist())
        )
        track["src_mask_id"] = (
            int(np.asarray(batch_src_mask_id, dtype=np.int64).reshape(-1)[idx])
            if batch_src_mask_id is not None
            else int(fallback_mask_id[idx])
        )
        if batch_xyz_local is not None:
            track["xyz_local"] = np.asarray(batch_xyz_local[:, idx, :], dtype=np.float32)
        track["carrier_id"] = int(carrier_ids[idx])
        track["persistent_tube_id"] = int(carrier_ids[idx])
        track["parent_tube_id"] = -1
        track["warmstart_source_chunk"] = -1
        track["warmstart_source_frame"] = -1
        track["is_warmstarted"] = False
    return tracks


def _warmstart_tracks_for_window_with_identity(
    *,
    previous_tracks: list[dict[str, Any]],
    previous_frame_ids: list[int],
    current_frame_ids: list[int],
) -> list[dict[str, Any]]:
    if not previous_tracks:
        return []
    prev_index = {int(frame_id): idx for idx, frame_id in enumerate(previous_frame_ids)}
    shared = [(cur_idx, prev_index[int(frame_id)], int(frame_id)) for cur_idx, frame_id in enumerate(current_frame_ids) if int(frame_id) in prev_index]
    if not shared:
        return []
    warmstart: list[dict[str, Any]] = []
    num_frames = len(current_frame_ids)
    for track in previous_tracks:
        uv_prev = np.asarray(track.get("uv_norm", []), dtype=np.float32)
        if uv_prev.shape[0] != len(previous_frame_ids):
            continue
        visibility_prev = np.asarray(track.get("visibility", np.zeros((len(previous_frame_ids),), dtype=np.float32)), dtype=np.float32)
        confidence_prev = np.asarray(track.get("confidence", np.zeros((len(previous_frame_ids),), dtype=np.float32)), dtype=np.float32)
        valid_prev = np.asarray(track.get("valid", np.zeros((len(previous_frame_ids),), dtype=bool)), dtype=bool)
        uv = np.zeros((num_frames, 2), dtype=np.float32)
        visibility = np.zeros((num_frames,), dtype=np.float32)
        confidence = np.zeros((num_frames,), dtype=np.float32)
        valid = np.zeros((num_frames,), dtype=bool)
        source_frames: list[int] = []
        for cur_idx, prev_idx, frame_id in shared:
            uv[cur_idx] = uv_prev[prev_idx]
            visibility[cur_idx] = visibility_prev[prev_idx]
            confidence[cur_idx] = confidence_prev[prev_idx]
            valid[cur_idx] = valid_prev[prev_idx]
            if valid[cur_idx]:
                source_frames.append(int(frame_id))
        if np.count_nonzero(valid) == 0:
            continue
        persistent = int(track.get("persistent_tube_id", track.get("carrier_id", -1)))
        warmstart.append(
            {
                "uv_norm": uv,
                "visibility": visibility,
                "confidence": confidence,
                "valid": valid,
                "persistent_tube_id": persistent,
                "parent_tube_id": persistent,
                "warmstart_source_chunk": int(track.get("window_index", -1)),
                "warmstart_source_frame": int(source_frames[0]) if source_frames else -1,
                "is_warmstarted": True,
            }
        )
    return warmstart


def _assign_persistent_tube_ids(
    *,
    tubes: list[dict[str, Any]],
    previous_tracks: list[dict[str, Any]],
    previous_frame_ids: list[int],
    current_frame_ids: list[int],
    window_index: int,
    next_persistent_id: int,
    uv_radius: float = 0.01,
) -> tuple[int, dict[str, Any]]:
    prev_index = {int(frame_id): idx for idx, frame_id in enumerate(previous_frame_ids)}
    shared = [(cur_idx, prev_index[int(frame_id)], int(frame_id)) for cur_idx, frame_id in enumerate(current_frame_ids) if int(frame_id) in prev_index]
    candidate_stats: dict[tuple[int, int], list[float | int]] = {}

    def source_key(track: dict[str, Any]) -> tuple[int, int, int] | None:
        if "source_frame_global" not in track or "source_xy" not in track:
            return None
        xy = track.get("source_xy")
        if xy is None:
            return None
        try:
            return (int(track["source_frame_global"]), int(xy[0]), int(xy[1]))
        except (TypeError, ValueError, IndexError):
            return None

    def add_candidate(cur_idx: int, prev_idx: int, dist: float, frame_id: int) -> None:
        key = (int(cur_idx), int(prev_idx))
        if key not in candidate_stats:
            candidate_stats[key] = [0.0, 0, int(frame_id)]
        candidate_stats[key][0] = float(candidate_stats[key][0]) + float(dist)
        candidate_stats[key][1] = int(candidate_stats[key][1]) + 1

    prev_by_source: dict[tuple[int, int, int], int] = {}
    for prev_idx, prev in enumerate(previous_tracks):
        key = source_key(prev)
        if key is not None and key not in prev_by_source:
            prev_by_source[key] = int(prev_idx)
    for cur_idx, cur in enumerate(tubes):
        key = source_key(cur)
        if key is not None and key in prev_by_source:
            add_candidate(cur_idx, prev_by_source[key], -1.0, key[0])

    try:
        from scipy.spatial import cKDTree
    except Exception:  # pragma: no cover - scipy is present in the canonical env.
        cKDTree = None

    if shared:
        if cKDTree is not None:
            for cur_local, prev_local, frame_id in shared:
                prev_points: list[np.ndarray] = []
                prev_ids: list[int] = []
                for prev_idx, prev in enumerate(previous_tracks):
                    prev_uv = np.asarray(prev.get("uv_norm", []), dtype=np.float32)
                    prev_valid = np.asarray(prev.get("valid", np.zeros((len(previous_frame_ids),), dtype=bool)), dtype=bool)
                    if prev_uv.shape[0] != len(previous_frame_ids):
                        continue
                    if not bool(prev_valid[prev_local]):
                        continue
                    uv = prev_uv[prev_local]
                    if np.isfinite(uv).all():
                        prev_points.append(uv)
                        prev_ids.append(int(prev_idx))
                if not prev_points:
                    continue
                tree = cKDTree(np.asarray(prev_points, dtype=np.float32))
                for cur_idx, cur in enumerate(tubes):
                    cur_uv = np.asarray(cur.get("uv_norm", []), dtype=np.float32)
                    cur_valid = np.asarray(cur.get("valid", np.zeros((len(current_frame_ids),), dtype=bool)), dtype=bool)
                    if cur_uv.shape[0] != len(current_frame_ids) or not bool(cur_valid[cur_local]):
                        continue
                    uv = cur_uv[cur_local]
                    if not np.isfinite(uv).all():
                        continue
                    dist, pos = tree.query(uv, k=1, distance_upper_bound=float(uv_radius))
                    if int(pos) >= len(prev_ids) or not np.isfinite(dist):
                        continue
                    add_candidate(cur_idx, prev_ids[int(pos)], float(dist), int(frame_id))
        else:
            for cur_idx, cur in enumerate(tubes):
                cur_uv = np.asarray(cur.get("uv_norm", []), dtype=np.float32)
                cur_valid = np.asarray(cur.get("valid", np.zeros((len(current_frame_ids),), dtype=bool)), dtype=bool)
                if cur_uv.shape[0] != len(current_frame_ids):
                    continue
                for prev_idx, prev in enumerate(previous_tracks):
                    prev_uv = np.asarray(prev.get("uv_norm", []), dtype=np.float32)
                    prev_valid = np.asarray(prev.get("valid", np.zeros((len(previous_frame_ids),), dtype=bool)), dtype=bool)
                    if prev_uv.shape[0] != len(previous_frame_ids):
                        continue
                    for cur_local, prev_local, frame_id in shared:
                        if not (cur_valid[cur_local] and prev_valid[prev_local]):
                            continue
                        dist = float(np.linalg.norm(cur_uv[cur_local] - prev_uv[prev_local]))
                        if np.isfinite(dist) and dist <= float(uv_radius):
                            add_candidate(cur_idx, prev_idx, dist, frame_id)
    candidates: list[tuple[float, int, int, int]] = []
    for (cur_idx, prev_idx), stats in candidate_stats.items():
        dist_sum, count, frame_id = stats
        if int(count) <= 0:
            continue
        mean_dist = float(dist_sum) / float(count)
        candidates.append((mean_dist, int(cur_idx), int(prev_idx), int(frame_id)))
    assigned_cur: set[int] = set()
    assigned_prev: set[int] = set()
    retained = 0
    for _, cur_idx, prev_idx, frame_id in sorted(candidates, key=lambda item: item[0]):
        if cur_idx in assigned_cur or prev_idx in assigned_prev:
            continue
        prev = previous_tracks[prev_idx]
        persistent = int(prev.get("persistent_tube_id", prev.get("carrier_id", -1)))
        if persistent < 0:
            continue
        tubes[cur_idx]["persistent_tube_id"] = persistent
        tubes[cur_idx]["parent_tube_id"] = persistent
        tubes[cur_idx]["warmstart_source_chunk"] = int(prev.get("window_index", window_index - 1))
        tubes[cur_idx]["warmstart_source_frame"] = int(frame_id)
        tubes[cur_idx]["is_warmstarted"] = True
        tubes[cur_idx]["window_index"] = int(window_index)
        assigned_cur.add(cur_idx)
        assigned_prev.add(prev_idx)
        retained += 1
    for cur_idx, tube in enumerate(tubes):
        if cur_idx in assigned_cur:
            continue
        tube["persistent_tube_id"] = int(next_persistent_id)
        tube["parent_tube_id"] = -1
        tube["warmstart_source_chunk"] = -1
        tube["warmstart_source_frame"] = -1
        tube["is_warmstarted"] = False
        tube["window_index"] = int(window_index)
        next_persistent_id += 1
    diag = {
        "persistent_tube_count": int(len({int(tube.get("persistent_tube_id", -1)) for tube in tubes if int(tube.get("persistent_tube_id", -1)) >= 0})),
        "persistent_tube_retention_count": int(retained),
        "persistent_tube_retention_rate": float(retained / max(len(tubes), 1)),
        "warmstart_tube_acceptance_rate": float(retained / max(len(previous_tracks), 1)) if previous_tracks else 0.0,
    }
    return next_persistent_id, diag


def _export_d2(
    *,
    stream: ScanNetStream,
    frame_ids: list[int],
    adapter: D4RTAdapter,
    args: argparse.Namespace,
    targets: OccupancyCoverageTargets,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene_dir = Path(args.output_root) / args.seq_name
    window_ids = _windows(frame_ids, window_size=int(args.window_size), window_stride=int(args.window_stride))
    sampler = CarrierSampler(
        max_points_per_mask=int(args.mask_fixed_points_per_mask),
        min_points_per_mask=int(args.mask_fixed_min_points_per_mask),
        strategy="grid_inside_mask",
    )
    windows: list[dict[str, Any]] = []
    total_queries = 0
    total_time = 0.0
    total_saved = 0
    for window_index, cur_frame_ids in enumerate(window_ids):
        data = _load_rgb_sparse_mask_window(stream, cur_frame_ids)
        frames = np.asarray(data["rgb"])
        masks = np.asarray(data["mask"])
        sources = sampler.sample(masks=masks, frame_ids=cur_frame_ids)
        t0 = time.time()
        batch = adapter.infer_carriers(
            video_rgb_uint8=frames,
            src_uv_norm=sources.src_uv,
            src_frame_local=sources.src_frame,
            query_chunk_size=int(args.query_chunk_size),
            carrier_id=sources.carrier_id,
            src_frame_global=sources.src_frame_global,
            src_xy=sources.src_xy,
            src_mask_id=sources.src_mask_id,
        )
        if batch.persistent_tube_id is None:
            batch.persistent_tube_id = np.asarray(batch.carrier_id, dtype=np.int64).copy()
        if batch.parent_tube_id is None:
            batch.parent_tube_id = np.full_like(batch.persistent_tube_id, -1)
        if batch.warmstart_source_chunk is None:
            batch.warmstart_source_chunk = np.full_like(batch.persistent_tube_id, -1)
        if batch.warmstart_source_frame is None:
            batch.warmstart_source_frame = np.full_like(batch.persistent_tube_id, -1)
        if batch.is_warmstarted is None:
            batch.is_warmstarted = np.zeros_like(batch.persistent_tube_id, dtype=bool)
        elapsed = float(time.time() - t0)
        saved = _save_batch(
            batch=batch,
            out_dir=scene_dir,
            frame_ids=cur_frame_ids,
            variant="D2_fixed_mask_aware_radius4",
            scene=args.seq_name,
            window_index=window_index,
        )
        saved.update(
            {
                "actual_source_query_count": int(sources.src_uv.shape[0]),
                "total_d4rt_time_sec": elapsed,
                "mask_measurement_frame_count": int(len(data.get("sparse_mask_present_frame_ids", []))),
                "missing_mask_frame_count": int(len(data.get("sparse_mask_missing_frame_ids", []))),
            }
        )
        windows.append(saved)
        total_queries += int(sources.src_uv.shape[0])
        total_time += elapsed
        total_saved += int(sources.src_uv.shape[0])
    row = {
        "variant": "D2_fixed_mask_aware_radius4",
        "scene": args.seq_name,
        "status": "ok",
        "num_windows": int(len(windows)),
        "actual_source_query_count": int(total_queries),
        "total_d4rt_time_sec": float(total_time),
        "num_carriers_saved": int(total_saved),
        "mark_radius_px": int(targets.mark_radius_px),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "output_scene_dir": str(scene_dir),
    }
    return windows, row


def _export_d5(
    *,
    stream: ScanNetStream,
    frame_ids: list[int],
    adapter: D4RTAdapter,
    args: argparse.Namespace,
    targets: OccupancyCoverageTargets,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene_dir = Path(args.output_root) / args.seq_name
    window_ids = _windows(frame_ids, window_size=int(args.window_size), window_stride=int(args.window_stride))
    previous_tracks: list[dict[str, Any]] = []
    previous_frame_ids: list[int] = []
    window_summaries: list[dict[str, Any]] = []
    total_queries = 0
    total_time = 0.0
    total_warmstart = 0
    total_retained = 0
    next_persistent_id = 0
    for window_index, cur_frame_ids in enumerate(window_ids):
        data = _load_rgb_sparse_mask_window(stream, cur_frame_ids)
        frames = np.asarray(data["rgb"])
        masks = np.asarray(data["mask"])
        warmstart = _warmstart_tracks_for_window_with_identity(
            previous_tracks=previous_tracks,
            previous_frame_ids=previous_frame_ids,
            current_frame_ids=cur_frame_ids,
        )
        seen_source_ids: set[int] = set()

        def decode(source_points: np.ndarray) -> list[dict[str, Any]]:
            src_frame = source_points[:, 0].astype(np.int64)
            src_uv = source_points[:, 1:3].astype(np.float32)
            src_frame_global, src_xy, src_mask_id = _source_arrays_from_points(
                source_points=source_points,
                frame_ids=cur_frame_ids,
                image_width=int(frames.shape[2]),
                image_height=int(frames.shape[1]),
                masks=masks,
            )
            carrier_ids = np.asarray(
                [
                    stable_source_carrier_id(
                        int(src_frame_global[idx]),
                        int(src_xy[idx, 0]),
                        int(src_xy[idx, 1]),
                        int(frames.shape[2]),
                    )
                    for idx in range(int(source_points.shape[0]))
                ],
                dtype=np.int64,
            )
            keep = np.asarray(
                [idx for idx, carrier_id in enumerate(carrier_ids.tolist()) if int(carrier_id) not in seen_source_ids],
                dtype=np.int64,
            )
            if keep.size == 0:
                return []
            for carrier_id in carrier_ids[keep].tolist():
                seen_source_ids.add(int(carrier_id))
            source_points = source_points[keep]
            src_frame = src_frame[keep]
            src_uv = src_uv[keep]
            src_frame_global = src_frame_global[keep]
            src_xy = src_xy[keep]
            src_mask_id = src_mask_id[keep]
            carrier_ids = carrier_ids[keep]
            batch = adapter.infer_carriers(
                video_rgb_uint8=frames,
                src_uv_norm=src_uv,
                src_frame_local=src_frame,
                query_chunk_size=int(args.query_chunk_size),
                carrier_id=carrier_ids,
                src_frame_global=src_frame_global,
                src_xy=src_xy,
                src_mask_id=src_mask_id,
            )
            return _tracks_from_batch_with_source(
                batch,
                source_points,
                carrier_ids,
                frame_ids=cur_frame_ids,
                image_width=int(frames.shape[2]),
                image_height=int(frames.shape[1]),
                masks=masks,
            )

        tubes, diag = query_d4rt_tubes_with_spatiotemporal_occupancy(
            frames=frames,
            masks=masks,
            decode_source_points=decode,
            coverage_targets=targets,
            query_budget=QueryBudget(
                max_source_points=int(args.query_budget),
                source_points_per_round=int(args.source_points_per_round),
            ),
            warmstart_tracks=warmstart,
        )
        topup_query_count = 0
        topup_track_count = 0
        topup_time = 0.0
        if int(args.topup_fixed_points_per_mask) > 0:
            topup_sampler = CarrierSampler(
                max_points_per_mask=int(args.topup_fixed_points_per_mask),
                min_points_per_mask=int(args.topup_fixed_min_points_per_mask),
                strategy="grid_inside_mask",
                seed=int(args.topup_seed),
            )
            topup_sources = topup_sampler.sample(masks=masks, frame_ids=cur_frame_ids)
            if topup_sources.src_uv.shape[0] > 0:
                source_points = np.concatenate(
                    [
                        np.asarray(topup_sources.src_frame, dtype=np.float32).reshape(-1, 1),
                        np.asarray(topup_sources.src_uv, dtype=np.float32).reshape(-1, 2),
                    ],
                    axis=1,
                )
                t_topup = time.time()
                topup_tracks = decode(source_points)
                topup_time = float(time.time() - t_topup)
                for track in topup_tracks:
                    track["topup_source"] = True
                tubes.extend(topup_tracks)
                topup_query_count = int(len(topup_tracks))
                topup_track_count = int(len(topup_tracks))
        next_persistent_id, identity_diag = _assign_persistent_tube_ids(
            tubes=tubes,
            previous_tracks=previous_tracks,
            previous_frame_ids=previous_frame_ids,
            current_frame_ids=cur_frame_ids,
            window_index=window_index,
            next_persistent_id=next_persistent_id,
            uv_radius=float(args.persistent_uv_radius),
        )
        batch = _tracks_to_batch(tubes, cur_frame_ids)
        saved = _save_batch(
            batch=batch,
            out_dir=scene_dir,
            frame_ids=cur_frame_ids,
            variant="D5_occupancy_dense_overlap_warmstart",
            scene=args.seq_name,
            window_index=window_index,
        )
        saved.update(
            {
                "actual_source_query_count": int(diag.get("actual_source_query_count", 0)),
                "warmstart_track_count": int(diag.get("warmstart_track_count", 0)),
                "query_budget_hit": bool(diag.get("query_budget_hit", False)),
                "mask_interior_coverage_mean": diag.get("mask_interior_coverage_mean"),
                "mask_boundary_coverage_mean": diag.get("mask_boundary_coverage_mean"),
                "topup_query_count": int(topup_query_count),
                "topup_track_count": int(topup_track_count),
                "topup_time_sec": float(topup_time),
                "total_d4rt_time_sec": float(diag.get("total_d4rt_time_sec", 0.0) or 0.0) + float(topup_time),
                "mask_measurement_frame_count": int(len(data.get("sparse_mask_present_frame_ids", []))),
                "missing_mask_frame_count": int(len(data.get("sparse_mask_missing_frame_ids", []))),
                **identity_diag,
            }
        )
        window_summaries.append(saved)
        total_queries += int(diag.get("actual_source_query_count", 0)) + int(topup_query_count)
        total_time += float(diag.get("total_d4rt_time_sec", 0.0) or 0.0) + float(topup_time)
        total_warmstart += int(diag.get("warmstart_track_count", 0))
        total_retained += int(identity_diag.get("persistent_tube_retention_count", 0))
        previous_tracks = tubes
        previous_frame_ids = list(cur_frame_ids)
        if bool(args.empty_cuda_cache_between_windows):
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    row = {
        "variant": "D5_occupancy_dense_overlap_warmstart",
        "scene": args.seq_name,
        "status": "ok",
        "num_windows": int(len(window_summaries)),
        "actual_source_query_count": int(total_queries),
        "total_d4rt_time_sec": float(total_time),
        "warmstart_track_count": int(total_warmstart),
        "persistent_tube_retention_count": int(total_retained),
        "persistent_tube_retention_rate": float(total_retained / max(sum(row["num_carriers"] for row in window_summaries), 1)),
        "topup_query_count": int(sum(int(row.get("topup_query_count", 0) or 0) for row in window_summaries)),
        "topup_track_count": int(sum(int(row.get("topup_track_count", 0) or 0) for row in window_summaries)),
        "num_carriers_saved": int(sum(row["num_carriers"] for row in window_summaries)),
        "mark_radius_px": int(targets.mark_radius_px),
        "output_scene_dir": str(scene_dir),
    }
    return window_summaries, row


def _write_summary(output_root: Path, row: dict[str, Any], windows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    payload = {"row": row, "windows": windows}
    summary_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output_root / "summary.csv"
    keys = sorted(row.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerow(row)
    lines = [
        "# v21.3 Occupancy Carrier Cache Export",
        "",
        "| variant | scene | windows | queries | carriers saved | warmstart | time sec | output |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        "| {variant} | {scene} | {windows} | {queries} | {carriers} | {warmstart} | {time:.6f} | `{output}` |".format(
            variant=row.get("variant"),
            scene=row.get("scene"),
            windows=int(row.get("num_windows", 0)),
            queries=int(row.get("actual_source_query_count", 0)),
            carriers=int(row.get("num_carriers_saved", 0)),
            warmstart=int(row.get("warmstart_track_count", 0) or 0),
            time=float(row.get("total_d4rt_time_sec", 0.0) or 0.0),
            output=row.get("output_scene_dir"),
        ),
    ]
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export v21.3 occupancy D4RT tubes as GeometryProvider carrier cache.")
    parser.add_argument("--d4rt-root", required=True)
    parser.add_argument("--d4rt-config", required=True)
    parser.add_argument("--d4rt-ckpt", required=True)
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--variant", choices=["D2", "D5"], required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--mask-fixed-points-per-mask", type=int, default=32)
    parser.add_argument("--mask-fixed-min-points-per-mask", type=int, default=4)
    parser.add_argument("--query-budget", type=int, default=7168)
    parser.add_argument("--source-points-per-round", type=int, default=512)
    parser.add_argument("--topup-fixed-points-per-mask", type=int, default=0)
    parser.add_argument("--topup-fixed-min-points-per-mask", type=int, default=2)
    parser.add_argument("--topup-seed", type=int, default=2600)
    parser.add_argument("--empty-cuda-cache-between-windows", action="store_true")
    parser.add_argument("--persistent-uv-radius", type=float, default=0.01)
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--mark-radius-px", type=int, default=4)
    parser.add_argument("--pixel-coverage-target", type=float, default=0.02)
    parser.add_argument("--mask-interior-coverage-target", type=float, default=0.20)
    parser.add_argument("--boundary-coverage-target", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-root", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stream = ScanNetStream(seq_name=args.seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    frame_ids = _frame_ids(stream, stride=args.frame_stride, max_frames=args.max_frames)
    targets = OccupancyCoverageTargets(
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
        mark_radius_px=int(args.mark_radius_px),
        pixel_coverage_target=float(args.pixel_coverage_target),
        mask_interior_coverage_target=float(args.mask_interior_coverage_target),
        boundary_coverage_target=float(args.boundary_coverage_target),
    )
    adapter = D4RTAdapter(
        d4rt_root=args.d4rt_root,
        model_config=args.d4rt_config,
        ckpt_path=args.d4rt_ckpt,
        device=args.device,
    )
    if args.variant == "D2":
        windows, row = _export_d2(stream=stream, frame_ids=frame_ids, adapter=adapter, args=args, targets=targets)
    else:
        windows, row = _export_d5(stream=stream, frame_ids=frame_ids, adapter=adapter, args=args, targets=targets)
    _write_summary(Path(args.summary_root) / args.seq_name, row, windows)
    print(json.dumps(_json_safe(row), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
