#!/usr/bin/env python3
"""Serve a diagnostic Viser view of v97 D4RT geometry against ScanNet GT geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry_provider.common import backproject_xy_world, fit_transform  # noqa: E402
from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v65_visualization_export import _load_scene_mesh, _window_colors  # noqa: E402


DEFAULT_PHASE2 = ROOT / "outputs/audit/v97_phase2_d4rt_micro_tracks_full_D3_gpu7_clamp002"
DEFAULT_GEOMETRY = ROOT / "outputs/audit/v97_phase8_d4rt_geometry_quality_D3_2048"
DEFAULT_OUTPUT = ROOT / "outputs/audit/v97_d4rt_gt_geometry_viewer_scene0011_d3"
DEFAULT_SCANNET = ROOT / "data/scannet/processed"


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def _stable_sample_indices(count: int, cap: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if cap <= 0 or count <= cap:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, num=int(cap), dtype=np.int64)


def _random_sample_indices(count: int, cap: int, seed: str) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if cap <= 0 or count <= cap:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(_stable_seed(seed))
    return np.sort(rng.choice(count, size=int(cap), replace=False).astype(np.int64))


def _window_sort_key(path: Path) -> tuple[int, str]:
    text = path.stem
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else 0, text)


def _split_train_eval(count: int, holdout_mod: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(count, dtype=np.int64)
    if count < 8 or holdout_mod <= 1:
        return idx, idx
    holdout = (idx % int(holdout_mod)) == 0
    train = ~holdout
    if int(np.count_nonzero(train)) < 4 or int(np.count_nonzero(holdout)) < 4:
        return idx, idx
    return idx[train], idx[holdout]


def _apply_fit(points: np.ndarray, fit: dict[str, Any] | None) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if fit is None or pts.size == 0:
        return np.full_like(pts, np.nan, dtype=np.float32)
    scale = float(fit["scale"])
    rotation = np.asarray(fit["rotation"], dtype=np.float64)
    translation = np.asarray(fit["translation"], dtype=np.float64)
    return (scale * (pts @ rotation.T) + translation).astype(np.float32)


def _finite_filter(points: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    pts = np.asarray(points)
    finite = np.isfinite(pts).all(axis=1)
    out: list[np.ndarray] = [pts[finite]]
    out.extend(np.asarray(array)[finite] for array in arrays)
    return tuple(out)


def _metric(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _fit_scope(
    *,
    scope: str,
    scope_id: str,
    source: np.ndarray,
    target: np.ndarray,
    robust_trim_percentile: float,
    holdout_mod: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    src = np.asarray(source, dtype=np.float32)
    dst = np.asarray(target, dtype=np.float32)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[finite]
    dst = dst[finite]
    train_idx, eval_idx = _split_train_eval(src.shape[0], int(holdout_mod))
    fit = fit_transform(src[train_idx], dst[train_idx], robust_trim_percentile=float(robust_trim_percentile))
    row: dict[str, Any] = {
        "scope": scope,
        "scope_id": scope_id,
        "anchor_count_total": int(src.shape[0]),
        "anchor_count_train": int(train_idx.shape[0]),
        "anchor_count_holdout": int(eval_idx.shape[0]),
        "fit_alignment_type": "eval_only_sim3",
        "fit_alignment_source": "scannet_depth_pose_backprojection",
        "robust_trim_percentile": float(robust_trim_percentile),
        "uses_gt_for_prediction": False,
        "uses_gt_for_visual_alignment": True,
        "is_diagnostic_visualization": True,
    }
    if fit is None:
        row["fit_status"] = "insufficient_anchors"
        return None, row
    train_res = np.linalg.norm(_apply_fit(src[train_idx], fit) - dst[train_idx], axis=1)
    eval_res = np.linalg.norm(_apply_fit(src[eval_idx], fit) - dst[eval_idx], axis=1)
    row.update(
        {
            "fit_status": "ok",
            "sim3_scale": float(fit["scale"]),
            "sim3_rotation_det": float(fit.get("rotation_det", np.linalg.det(np.asarray(fit["rotation"], dtype=np.float64)))),
            "sim3_translation_norm": float(np.linalg.norm(np.asarray(fit["translation"], dtype=np.float64))),
            "robust_kept_anchors": int(fit.get("robust_kept_anchors", train_idx.shape[0])),
            "train_residual_m": _metric(train_res),
            "holdout_residual_m": _metric(eval_res),
        }
    )
    return fit, row


def _sample_rgb(stream: ScanNetStream, frame_id: int, uv: np.ndarray) -> np.ndarray:
    if uv.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    rgb = stream.load_rgb(int(frame_id))
    h, w = rgb.shape[:2]
    xy = np.rint(
        np.stack(
            [
                uv[:, 0] * float(max(w - 1, 1)),
                uv[:, 1] * float(max(h - 1, 1)),
            ],
            axis=1,
        )
    ).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, max(w - 1, 0))
    xy[:, 1] = np.clip(xy[:, 1], 0, max(h - 1, 0))
    return np.asarray(rgb[xy[:, 1], xy[:, 0], :3], dtype=np.uint8)


def _load_d4rt_for_scene(args: argparse.Namespace) -> dict[str, Any]:
    phase2_root = _project(args.phase2_root)
    batch_root = phase2_root / "carrier_batches" / args.decode_variant / args.scene
    paths = sorted(batch_root.glob("*.npz"), key=_window_sort_key)
    if not paths:
        raise FileNotFoundError(f"No v97 D4RT carrier batches under {batch_root}")

    stream = ScanNetStream(seq_name=args.scene, backbone=args.backbone, root=_project(args.scannet_root))
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))

    anchor_source_parts: list[np.ndarray] = []
    anchor_target_parts: list[np.ndarray] = []
    anchor_index_by_scene: list[int] = []
    anchor_index_by_window: dict[str, list[int]] = defaultdict(list)
    display_xyz_parts: list[np.ndarray] = []
    display_window_id_parts: list[np.ndarray] = []
    display_frame_id_parts: list[np.ndarray] = []
    display_rgb_parts: list[np.ndarray] = []
    window_name_parts: list[str] = []
    frame_rows: list[dict[str, Any]] = []
    total_query_slots = 0
    total_uv_in01 = 0
    total_accepted = 0
    total_depth_hits = 0

    for window_index, batch_path in enumerate(paths):
        with np.load(batch_path, allow_pickle=True) as data:
            frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
            uv = np.asarray(data["uv_pred"], dtype=np.float32)
            xyz = np.asarray(data["xyz_ref"], dtype=np.float32)
            visibility = np.asarray(data["visibility_prob"], dtype=np.float32)
            confidence = np.asarray(data["confidence_prob"], dtype=np.float32)
            valid = np.asarray(data.get("valid", np.ones(uv.shape[:2], dtype=bool)), dtype=bool)
        per_frame_display_cap = max(1, int(args.max_display_points_per_window) // max(1, int(frame_ids.shape[0])))
        for local_idx, frame_id in enumerate(frame_ids.tolist()):
            uv_frame = uv[local_idx]
            xyz_frame = xyz[local_idx]
            uv_in01 = (
                np.isfinite(uv_frame).all(axis=1)
                & (uv_frame[:, 0] >= 0.0)
                & (uv_frame[:, 0] <= 1.0)
                & (uv_frame[:, 1] >= 0.0)
                & (uv_frame[:, 1] <= 1.0)
            )
            accepted = (
                valid[local_idx]
                & uv_in01
                & np.isfinite(xyz_frame).all(axis=1)
                & (visibility[local_idx] >= float(args.min_visibility))
                & (confidence[local_idx] >= float(args.min_confidence))
            )
            accepted_indices = np.flatnonzero(accepted)
            fit_local = accepted_indices[
                _stable_sample_indices(int(accepted_indices.shape[0]), int(args.max_fit_points_per_frame))
            ]
            total_query_slots += int(uv_frame.shape[0])
            total_uv_in01 += int(np.count_nonzero(uv_in01))
            total_accepted += int(accepted_indices.shape[0])

            depth_hit_count = 0
            if fit_local.size:
                depth = stream.load_depth(int(frame_id))
                h, w = depth.shape[:2]
                xy = np.stack(
                    [
                        uv_frame[fit_local, 0] * float(max(w - 1, 1)),
                        uv_frame[fit_local, 1] * float(max(h - 1, 1)),
                    ],
                    axis=1,
                )
                world, world_ok = backproject_xy_world(stream, int(frame_id), xy)
                src = xyz_frame[fit_local][world_ok].astype(np.float32, copy=False)
                dst = world[world_ok].astype(np.float32, copy=False)
                depth_hit_count = int(src.shape[0])
                if depth_hit_count:
                    start = sum(part.shape[0] for part in anchor_source_parts)
                    anchor_source_parts.append(src)
                    anchor_target_parts.append(dst)
                    indices = list(range(start, start + depth_hit_count))
                    anchor_index_by_scene.extend(indices)
                    anchor_index_by_window[batch_path.stem].extend(indices)
            total_depth_hits += depth_hit_count

            display_local = accepted_indices[
                _random_sample_indices(
                    int(accepted_indices.shape[0]),
                    int(per_frame_display_cap),
                    f"{args.scene}:{batch_path.stem}:{int(frame_id)}:display",
                )
            ]
            if display_local.size:
                display_xyz_parts.append(xyz_frame[display_local].astype(np.float32, copy=False))
                display_window_id_parts.append(np.full((display_local.shape[0],), int(window_index), dtype=np.int64))
                display_frame_id_parts.append(np.full((display_local.shape[0],), int(frame_id), dtype=np.int64))
                display_rgb_parts.append(_sample_rgb(stream, int(frame_id), uv_frame[display_local]))
            frame_rows.append(
                {
                    "scene_id": args.scene,
                    "window_id": batch_path.stem,
                    "window_index": int(window_index),
                    "frame_id": int(frame_id),
                    "query_count": int(uv_frame.shape[0]),
                    "uv_in01_count": int(np.count_nonzero(uv_in01)),
                    "accepted_count": int(accepted_indices.shape[0]),
                    "fit_sample_count": int(fit_local.shape[0]),
                    "fit_depth_hit_count": int(depth_hit_count),
                    "display_sample_count": int(display_local.shape[0]),
                }
            )
        window_name_parts.append(batch_path.stem)

    source = np.concatenate(anchor_source_parts, axis=0) if anchor_source_parts else np.empty((0, 3), dtype=np.float32)
    target = np.concatenate(anchor_target_parts, axis=0) if anchor_target_parts else np.empty((0, 3), dtype=np.float32)
    display_xyz = np.concatenate(display_xyz_parts, axis=0) if display_xyz_parts else np.empty((0, 3), dtype=np.float32)
    display_window = (
        np.concatenate(display_window_id_parts, axis=0) if display_window_id_parts else np.empty((0,), dtype=np.int64)
    )
    display_frame = (
        np.concatenate(display_frame_id_parts, axis=0) if display_frame_id_parts else np.empty((0,), dtype=np.int64)
    )
    display_rgb = np.concatenate(display_rgb_parts, axis=0) if display_rgb_parts else np.empty((0, 3), dtype=np.uint8)

    scene_idx = np.asarray(anchor_index_by_scene, dtype=np.int64)
    scene_fit, scene_fit_row = _fit_scope(
        scope="scene",
        scope_id=args.scene,
        source=source[scene_idx],
        target=target[scene_idx],
        robust_trim_percentile=float(args.robust_trim_percentile),
        holdout_mod=int(args.holdout_mod),
    )
    window_fits: dict[int, dict[str, Any] | None] = {}
    window_fit_rows: list[dict[str, Any]] = []
    for window_index, window_name in enumerate(window_name_parts):
        idx = np.asarray(anchor_index_by_window.get(window_name, []), dtype=np.int64)
        fit, row = _fit_scope(
            scope="window",
            scope_id=f"{args.scene}:{window_name}",
            source=source[idx],
            target=target[idx],
            robust_trim_percentile=float(args.robust_trim_percentile),
            holdout_mod=int(args.holdout_mod),
        )
        row["window_index"] = int(window_index)
        window_fits[window_index] = fit
        window_fit_rows.append(row)

    scene_fit_points = _apply_fit(display_xyz, scene_fit)
    window_fit_points = np.full_like(scene_fit_points, np.nan, dtype=np.float32)
    for window_index, fit in window_fits.items():
        sel = display_window == int(window_index)
        if np.any(sel):
            window_fit_points[sel] = _apply_fit(display_xyz[sel], fit)
    window_colors = _window_colors(display_window)

    scene_fit_points, scene_fit_colors, scene_fit_rgb, scene_fit_window, scene_fit_frame = _finite_filter(
        scene_fit_points, window_colors, display_rgb, display_window, display_frame
    )
    window_fit_points, window_fit_colors, window_fit_rgb, window_fit_window, window_fit_frame = _finite_filter(
        window_fit_points, window_colors, display_rgb, display_window, display_frame
    )
    raw_points, raw_colors, raw_rgb, raw_window, raw_frame = _finite_filter(
        display_xyz, window_colors, display_rgb, display_window, display_frame
    )

    return {
        "scene_fit_points": scene_fit_points.astype(np.float32),
        "scene_fit_colors": scene_fit_colors.astype(np.uint8),
        "scene_fit_rgb": scene_fit_rgb.astype(np.uint8),
        "scene_fit_window": scene_fit_window.astype(np.int64),
        "scene_fit_frame": scene_fit_frame.astype(np.int64),
        "window_fit_points": window_fit_points.astype(np.float32),
        "window_fit_colors": window_fit_colors.astype(np.uint8),
        "window_fit_rgb": window_fit_rgb.astype(np.uint8),
        "window_fit_window": window_fit_window.astype(np.int64),
        "window_fit_frame": window_fit_frame.astype(np.int64),
        "raw_ref_points": raw_points.astype(np.float32),
        "raw_ref_colors": raw_colors.astype(np.uint8),
        "raw_ref_rgb": raw_rgb.astype(np.uint8),
        "raw_ref_window": raw_window.astype(np.int64),
        "raw_ref_frame": raw_frame.astype(np.int64),
        "frame_rows": frame_rows,
        "scene_fit_row": scene_fit_row,
        "window_fit_rows": window_fit_rows,
        "window_names": window_name_parts,
        "summary": {
            "scene_id": args.scene,
            "decode_variant": args.decode_variant,
            "batch_file_count": int(len(paths)),
            "window_names": window_name_parts,
            "total_query_slots": int(total_query_slots),
            "total_uv_in01": int(total_uv_in01),
            "total_accepted": int(total_accepted),
            "total_depth_hits": int(total_depth_hits),
            "uv_in01_rate": float(total_uv_in01 / max(1, total_query_slots)),
            "accepted_rate_of_query_slots": float(total_accepted / max(1, total_query_slots)),
            "depth_hit_rate_of_fit_samples": float(
                total_depth_hits / max(1, sum(int(row["fit_sample_count"]) for row in frame_rows))
            ),
            "display_raw_ref_point_count": int(raw_points.shape[0]),
            "display_scene_fit_point_count": int(scene_fit_points.shape[0]),
            "display_window_fit_point_count": int(window_fit_points.shape[0]),
            "fit_anchor_count": int(source.shape[0]),
            "max_fit_points_per_frame": int(args.max_fit_points_per_frame),
            "max_display_points_per_window": int(args.max_display_points_per_window),
            "min_visibility": float(args.min_visibility),
            "min_confidence": float(args.min_confidence),
        },
    }


def export_layers(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gt_points, gt_colors, mesh_path = _load_scene_mesh(args.scene)
    if int(args.max_gt_points) > 0 and gt_points.shape[0] > int(args.max_gt_points):
        idx = _random_sample_indices(gt_points.shape[0], int(args.max_gt_points), f"{args.scene}:gt")
        gt_points = gt_points[idx]
        gt_colors = gt_colors[idx]
    d4rt = _load_d4rt_for_scene(args)
    layer_path = output_root / "viewer_layers.npz"
    np.savez_compressed(
        layer_path,
        gt_geometry_points=gt_points.astype(np.float32),
        gt_geometry_colors=gt_colors.astype(np.uint8),
        d4rt_scene_fit_points=d4rt["scene_fit_points"],
        d4rt_scene_fit_colors=d4rt["scene_fit_colors"],
        d4rt_scene_fit_rgb=d4rt["scene_fit_rgb"],
        d4rt_scene_fit_window=d4rt["scene_fit_window"],
        d4rt_scene_fit_frame=d4rt["scene_fit_frame"],
        d4rt_window_fit_points=d4rt["window_fit_points"],
        d4rt_window_fit_colors=d4rt["window_fit_colors"],
        d4rt_window_fit_rgb=d4rt["window_fit_rgb"],
        d4rt_window_fit_window=d4rt["window_fit_window"],
        d4rt_window_fit_frame=d4rt["window_fit_frame"],
        d4rt_raw_ref_points=d4rt["raw_ref_points"],
        d4rt_raw_ref_colors=d4rt["raw_ref_colors"],
        d4rt_raw_ref_rgb=d4rt["raw_ref_rgb"],
        d4rt_raw_ref_window=d4rt["raw_ref_window"],
        d4rt_raw_ref_frame=d4rt["raw_ref_frame"],
    )

    geometry_summary_path = _project(args.geometry_root) / "summary.json" if args.geometry_root else Path("")
    phase8_summary = _read_json(geometry_summary_path) if args.geometry_root else {}
    index = {
        "phase": "v97_d4rt_gt_geometry_viewer_export",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scene": args.scene,
        "decode_variant": args.decode_variant,
        "phase2_root": _rel(args.phase2_root),
        "geometry_root": _rel(args.geometry_root) if args.geometry_root else "",
        "output_root": _rel(output_root),
        "viewer_layers_npz": _rel(layer_path),
        "viewer_layers_npz_sha256": _sha256_file(layer_path),
        "mesh_path": _rel(mesh_path),
        "mesh_path_sha256": _sha256_file(mesh_path),
        "layers": [
            "GT geometry ScanNet mesh RGB",
            "D4RT scene-fit geometry window colors",
            "D4RT window-fit geometry window colors",
            "D4RT window-fit geometry RGB",
            "D4RT raw xyz_ref window colors",
        ],
        "gt_geometry_point_count": int(gt_points.shape[0]),
        "d4rt_summary": d4rt["summary"],
        "scene_fit_row": d4rt["scene_fit_row"],
        "window_fit_rows": d4rt["window_fit_rows"],
        "phase8_summary_reference": {
            key: phase8_summary.get(key)
            for key in [
                "decision",
                "sampled_point_count",
                "scene_fit_frame_paired_residual_p50_m_mean",
                "window_fit_frame_paired_residual_p50_m_mean",
                "scene_fit_frame_symmetric_chamfer_mean_m_mean",
                "window_fit_frame_symmetric_chamfer_mean_m_mean",
                "scale_pair_outside_10pct_count",
                "ate_sim3_rmse_m",
            ]
            if key in phase8_summary
        },
        "diagnostic_contract": {
            "uses_gt_for_prediction": False,
            "uses_gt_for_visual_alignment": True,
            "uses_rgbd_pose_mesh_for_visual_alignment": True,
            "is_method_safe": False,
            "is_diagnostic_visualization": True,
            "note": "ScanNet depth/pose/mesh are used only to visualize and audit D4RT geometry, not to produce method predictions.",
        },
    }
    _write_json(output_root / "viewer_index.json", index)
    _write_json(
        output_root / "frame_sampling_rows.json",
        {
            "schema": "stream4d_v97_d4rt_gt_geometry_viewer_frame_sampling_v1",
            "rows": d4rt["frame_rows"],
        },
    )
    return index


def serve(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    index_path = output_root / "viewer_index.json"
    if bool(args.rebuild_layers) or not index_path.exists():
        index = export_layers(args)
    else:
        index = _read_json(index_path)

    layer_path = _project(index["viewer_layers_npz"])
    with np.load(layer_path) as payload:
        layers = {key: np.asarray(payload[key]) for key in payload.files}

    import viser  # type: ignore

    server = viser.ViserServer(host=args.host, port=int(args.port), verbose=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/v97_d4rt_gt_geometry/grid",
        width=8.0,
        height=8.0,
        plane="xy",
        cell_size=0.5,
        section_size=2.0,
        position=(0.0, 0.0, -0.02),
    )
    handles: dict[str, Any] = {}

    def add_layer(layer_key: str, path: str, points: np.ndarray, colors: np.ndarray, point_size: float, visible: bool) -> None:
        handle = server.scene.add_point_cloud(
            path,
            points=np.asarray(points, dtype=np.float32),
            colors=np.asarray(colors, dtype=np.uint8),
            point_size=float(point_size),
            point_shape="circle",
            precision="float32",
        )
        handle.visible = bool(visible)
        handles[layer_key] = handle

    add_layer(
        "GT geometry",
        "/v97_d4rt_gt_geometry/GT geometry ScanNet mesh RGB",
        layers["gt_geometry_points"],
        layers["gt_geometry_colors"],
        args.gt_point_size,
        True,
    )
    add_layer(
        "D4RT scene-fit",
        "/v97_d4rt_gt_geometry/D4RT scene-fit geometry window colors",
        layers["d4rt_scene_fit_points"],
        layers["d4rt_scene_fit_colors"],
        args.d4rt_scene_point_size,
        True,
    )
    add_layer(
        "D4RT window-fit",
        "/v97_d4rt_gt_geometry/D4RT window-fit geometry window colors",
        layers["d4rt_window_fit_points"],
        layers["d4rt_window_fit_colors"],
        args.d4rt_window_point_size,
        False,
    )
    add_layer(
        "D4RT window-fit RGB",
        "/v97_d4rt_gt_geometry/D4RT window-fit geometry RGB",
        layers["d4rt_window_fit_points"],
        layers["d4rt_window_fit_rgb"],
        args.d4rt_window_point_size,
        False,
    )
    add_layer(
        "D4RT raw xyz_ref",
        "/v97_d4rt_gt_geometry/D4RT raw xyz_ref window colors",
        layers["d4rt_raw_ref_points"],
        layers["d4rt_raw_ref_colors"],
        args.d4rt_raw_point_size,
        False,
    )

    def set_only(active: set[str]) -> None:
        for key, handle in handles.items():
            handle.visible = key in active
            if key in controls:
                controls[key].value = key in active

    controls: dict[str, Any] = {}
    with server.gui.add_folder("v97 D4RT / GT geometry"):
        for layer_key in handles:
            controls[layer_key] = server.gui.add_checkbox(layer_key, initial_value=bool(handles[layer_key].visible))
            controls[layer_key].on_update(
                lambda _event, layer_key=layer_key: setattr(handles[layer_key], "visible", bool(controls[layer_key].value))
            )
        gt_scene = server.gui.add_button("GT + scene-fit")
        gt_window = server.gui.add_button("GT + window-fit")
        show_all = server.gui.add_button("show all")
        hide_d4rt = server.gui.add_button("hide D4RT")

    @gt_scene.on_click
    def _(_: Any) -> None:
        set_only({"GT geometry", "D4RT scene-fit"})

    @gt_window.on_click
    def _(_: Any) -> None:
        set_only({"GT geometry", "D4RT window-fit"})

    @show_all.on_click
    def _(_: Any) -> None:
        set_only(set(handles))

    @hide_d4rt.on_click
    def _(_: Any) -> None:
        set_only({"GT geometry"})

    status = {
        "phase": "v97_d4rt_gt_geometry_live_viewer",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scene": args.scene,
        "host": args.host,
        "port": int(args.port),
        "url": f"http://localhost:{int(args.port)}",
        "viewer_index": _rel(index_path),
        "viewer_layers_npz": index.get("viewer_layers_npz"),
        "viewer_layers_npz_sha256": index.get("viewer_layers_npz_sha256"),
        "layers": list(handles.keys()),
        "default_visible_layers": [key for key, handle in handles.items() if bool(handle.visible)],
        "gt_geometry_point_count": int(layers["gt_geometry_points"].shape[0]),
        "d4rt_scene_fit_point_count": int(layers["d4rt_scene_fit_points"].shape[0]),
        "d4rt_window_fit_point_count": int(layers["d4rt_window_fit_points"].shape[0]),
        "d4rt_raw_ref_point_count": int(layers["d4rt_raw_ref_points"].shape[0]),
        "d4rt_summary": index.get("d4rt_summary", {}),
        "scene_fit_row": index.get("scene_fit_row", {}),
        "phase8_summary_reference": index.get("phase8_summary_reference", {}),
        "diagnostic_contract": index.get("diagnostic_contract", {}),
    }
    _write_json(output_root / "live_viewer_status.json", status)
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
    parser.add_argument("--phase2-root", default=str(DEFAULT_PHASE2))
    parser.add_argument("--geometry-root", default=str(DEFAULT_GEOMETRY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--decode-variant", default="D3_adaptive1024")
    parser.add_argument("--scannet-root", default=str(DEFAULT_SCANNET))
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-fit-points-per-frame", type=int, default=2048)
    parser.add_argument("--max-display-points-per-window", type=int, default=15000)
    parser.add_argument("--max-gt-points", type=int, default=0)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    parser.add_argument("--holdout-mod", type=int, default=5)
    parser.add_argument("--gt-point-size", type=float, default=0.006)
    parser.add_argument("--d4rt-scene-point-size", type=float, default=0.016)
    parser.add_argument("--d4rt-window-point-size", type=float, default=0.014)
    parser.add_argument("--d4rt-raw-point-size", type=float, default=0.01)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--rebuild-layers", action="store_true")
    args = parser.parse_args()
    if args.export_only:
        print(json.dumps(_jsonable(export_layers(args)), indent=2, sort_keys=True))
    else:
        serve(args)


if __name__ == "__main__":
    main()
