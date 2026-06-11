from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.carrier_store import CarrierBatch, CarrierSources
from stream4d.d4rt_adapter import D4RTAdapter
from stream4d.diagnostics import carrier_diagnostics, save_overlay, write_json
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
    return value


def _stable_surfel_id(frame_id: int, x: np.ndarray, y: np.ndarray, width: int) -> np.ndarray:
    return (
        np.int64(frame_id) * np.int64(10_000_000_000)
        + y.astype(np.int64) * np.int64(max(width, 1))
        + x.astype(np.int64)
    )


def _read_seq_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _windows(frame_ids: list[int], window_size: int, window_stride: int) -> list[list[int]]:
    if not frame_ids:
        return []
    out: list[list[int]] = []
    start = 0
    window_size = max(1, int(window_size))
    window_stride = max(1, int(window_stride))
    while start < len(frame_ids):
        cur = frame_ids[start : start + window_size]
        if cur:
            out.append(cur)
        if start + window_size >= len(frame_ids):
            break
        start += window_stride
    return out


def _base_grid_xy(height: int, width: int, grid_size: int, margin_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    grid_size = max(1, int(grid_size))
    margin_ratio = float(np.clip(float(margin_ratio), 0.0, 0.49))
    margin_x = margin_ratio * float(max(width - 1, 1))
    margin_y = margin_ratio * float(max(height - 1, 1))
    xs = np.rint(np.linspace(margin_x, float(max(width - 1, 0)) - margin_x, num=grid_size)).astype(np.int64)
    ys = np.rint(np.linspace(margin_y, float(max(height - 1, 0)) - margin_y, num=grid_size)).astype(np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x.reshape(-1), grid_y.reshape(-1)


def _append_points(
    *,
    carrier_ids: list[np.ndarray],
    src_frames: list[np.ndarray],
    src_globals: list[np.ndarray],
    src_xys: list[np.ndarray],
    src_uvs: list[np.ndarray],
    src_masks: list[np.ndarray],
    frame_id: int,
    local_idx: int,
    xs: np.ndarray,
    ys: np.ndarray,
    mask_values: np.ndarray,
    width: int,
    height: int,
) -> None:
    if xs.size == 0:
        return
    xs = xs.astype(np.int64, copy=False)
    ys = ys.astype(np.int64, copy=False)
    carrier_ids.append(_stable_surfel_id(frame_id, xs, ys, width))
    src_frames.append(np.full(xs.shape[0], int(local_idx), dtype=np.int64))
    src_globals.append(np.full(xs.shape[0], int(frame_id), dtype=np.int64))
    src_xys.append(np.stack([xs, ys], axis=1).astype(np.int64, copy=False))
    src_uvs.append(
        np.stack(
            [
                xs.astype(np.float32) / float(max(width - 1, 1)),
                ys.astype(np.float32) / float(max(height - 1, 1)),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
    )
    src_masks.append(mask_values.astype(np.int64, copy=False).reshape(-1))


def _grid_sources(
    masks: np.ndarray,
    frame_ids: list[int],
    grid_size: int,
    grid_margin_ratio: float,
    mask_aware_min_points_per_mask: int,
    min_mask_area: int,
) -> tuple[CarrierSources, dict[str, Any]]:
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape [T,H,W], got {masks.shape}")
    _, height, width = masks.shape
    base_x, base_y = _base_grid_xy(height, width, grid_size, grid_margin_ratio)
    carrier_ids: list[np.ndarray] = []
    src_frames: list[np.ndarray] = []
    src_globals: list[np.ndarray] = []
    src_xys: list[np.ndarray] = []
    src_uvs: list[np.ndarray] = []
    src_masks: list[np.ndarray] = []
    extra_points = 0
    extra_masks = 0

    for local_idx, frame_id in enumerate(frame_ids):
        mask = masks[local_idx]
        used: set[tuple[int, int]] = set((int(x), int(y)) for x, y in zip(base_x.tolist(), base_y.tolist()))
        _append_points(
            carrier_ids=carrier_ids,
            src_frames=src_frames,
            src_globals=src_globals,
            src_xys=src_xys,
            src_uvs=src_uvs,
            src_masks=src_masks,
            frame_id=int(frame_id),
            local_idx=int(local_idx),
            xs=base_x,
            ys=base_y,
            mask_values=mask[base_y, base_x],
            width=width,
            height=height,
        )
        if mask_aware_min_points_per_mask <= 0:
            continue
        for mask_id in np.unique(mask):
            if int(mask_id) <= 0:
                continue
            ys, xs = np.where(mask == mask_id)
            if ys.shape[0] < int(min_mask_area):
                continue
            inside_grid = int(sum(1 for x, y in used if mask[y, x] == mask_id))
            need = int(mask_aware_min_points_per_mask) - inside_grid
            if need <= 0:
                continue
            order = np.lexsort((xs, ys))
            added_x: list[int] = []
            added_y: list[int] = []
            for idx in np.linspace(0, order.shape[0] - 1, num=min(need * 3, order.shape[0]), dtype=np.int64):
                x = int(xs[order[idx]])
                y = int(ys[order[idx]])
                if (x, y) in used:
                    continue
                used.add((x, y))
                added_x.append(x)
                added_y.append(y)
                if len(added_x) >= need:
                    break
            if not added_x:
                continue
            extra_masks += 1
            extra_points += len(added_x)
            _append_points(
                carrier_ids=carrier_ids,
                src_frames=src_frames,
                src_globals=src_globals,
                src_xys=src_xys,
                src_uvs=src_uvs,
                src_masks=src_masks,
                frame_id=int(frame_id),
                local_idx=int(local_idx),
                xs=np.asarray(added_x, dtype=np.int64),
                ys=np.asarray(added_y, dtype=np.int64),
                mask_values=np.full(len(added_x), int(mask_id), dtype=np.int64),
                width=width,
                height=height,
            )

    if not carrier_ids:
        empty = CarrierSources(
            carrier_id=np.empty((0,), dtype=np.int64),
            src_frame=np.empty((0,), dtype=np.int64),
            src_frame_global=np.empty((0,), dtype=np.int64),
            src_xy=np.empty((0, 2), dtype=np.int64),
            src_uv=np.empty((0, 2), dtype=np.float32),
            src_mask_id=np.empty((0,), dtype=np.int64),
        )
        return empty, {
            "grid_points_per_frame": int(base_x.shape[0]),
            "grid_margin_ratio": float(grid_margin_ratio),
            "mask_aware_extra_points": 0,
            "mask_aware_extra_masks": 0,
        }

    return (
        CarrierSources(
            carrier_id=np.concatenate(carrier_ids, axis=0),
            src_frame=np.concatenate(src_frames, axis=0),
            src_frame_global=np.concatenate(src_globals, axis=0),
            src_xy=np.concatenate(src_xys, axis=0),
            src_uv=np.concatenate(src_uvs, axis=0),
            src_mask_id=np.concatenate(src_masks, axis=0),
        ),
        {
            "grid_points_per_frame": int(base_x.shape[0]),
            "grid_margin_ratio": float(grid_margin_ratio),
            "mask_aware_extra_points": int(extra_points),
            "mask_aware_extra_masks": int(extra_masks),
        },
    )


def _percentile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _track_metrics(
    batch: CarrierBatch,
    image_hw: tuple[int, int],
    min_visibility: float,
    min_confidence: float,
) -> dict[str, Any]:
    target_count, num_tracks = batch.valid.shape
    model_h, model_w = int(image_hw[0]), int(image_hw[1])
    uv = np.asarray(batch.uv_pred, dtype=np.float32)
    uv_finite = np.isfinite(uv).all(axis=-1)
    uv_in01 = uv_finite & (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    visible = (
        np.asarray(batch.valid, dtype=bool)
        & uv_in01
        & (np.asarray(batch.visibility_prob) >= float(min_visibility))
        & (np.asarray(batch.confidence_prob) >= float(min_confidence))
    )
    lengths = np.sum(visible, axis=0) if num_tracks else np.empty((0,), dtype=np.int64)
    coverage: list[float] = []
    for local_idx in range(target_count):
        pts = uv[local_idx, visible[local_idx]]
        if pts.size == 0:
            coverage.append(0.0)
            continue
        x = np.clip(np.rint(pts[:, 0] * float(max(model_w - 1, 1))).astype(np.int64), 0, max(model_w - 1, 0))
        y = np.clip(np.rint(pts[:, 1] * float(max(model_h - 1, 1))).astype(np.int64), 0, max(model_h - 1, 0))
        unique = np.unique(y * max(model_w, 1) + x)
        coverage.append(float(unique.shape[0]) / float(max(model_h * model_w, 1)))

    carrier_indices = np.arange(num_tracks, dtype=np.int64)
    src_local = np.asarray(batch.src_frame, dtype=np.int64)
    self_ok = (src_local >= 0) & (src_local < target_count) & (carrier_indices < num_tracks)
    self_err_px = np.empty((0,), dtype=np.float32)
    if np.any(self_ok):
        src_idx = carrier_indices[self_ok]
        local_idx = src_local[self_ok]
        self_uv = uv[local_idx, src_idx]
        scale = np.asarray([float(max(model_w - 1, 1)), float(max(model_h - 1, 1))], dtype=np.float32)
        self_err_px = np.linalg.norm((self_uv - batch.src_uv[self_ok]) * scale.reshape(1, 2), axis=1)
        self_err_px = self_err_px[np.isfinite(self_err_px)]

    duplicate_source_rate = 0.0
    if batch.carrier_id.size:
        duplicate_source_rate = 1.0 - float(np.unique(batch.carrier_id).shape[0]) / float(batch.carrier_id.shape[0])

    return {
        "num_surfel_tracks": int(num_tracks),
        "num_valid_tracks": int(np.sum(lengths > 0)) if lengths.size else 0,
        "num_visible_observations": int(np.sum(visible)),
        "surfel_coverage_2d_per_frame_mean": float(np.mean(coverage)) if coverage else 0.0,
        "surfel_coverage_2d_per_frame_min": float(np.min(coverage)) if coverage else 0.0,
        "surfel_coverage_2d_per_frame_max": float(np.max(coverage)) if coverage else 0.0,
        "hole_ratio_2d_mean": float(1.0 - np.mean(coverage)) if coverage else 1.0,
        "track_length_visible_mean": float(np.mean(lengths)) if lengths.size else 0.0,
        "track_length_visible_p10": _percentile(lengths, 10) if lengths.size else None,
        "track_length_visible_p50": _percentile(lengths, 50) if lengths.size else None,
        "self_uv_error_mean": float(np.mean(self_err_px)) if self_err_px.size else None,
        "self_uv_error_p90": _percentile(self_err_px, 90),
        "duplicate_track_rate": float(duplicate_source_rate),
        "visible_min_visibility": float(min_visibility),
        "visible_min_confidence": float(min_confidence),
    }


def _cycle_probe(
    adapter: D4RTAdapter,
    video_rgb_uint8: np.ndarray,
    batch: CarrierBatch,
    image_hw: tuple[int, int],
    source_local: int,
    target_local: int,
    max_tracks: int,
    query_chunk_size: int,
) -> dict[str, Any]:
    if max_tracks <= 0 or batch.carrier_id.size == 0:
        return {"cycle_probe_enabled": False}
    target_count, num_tracks = batch.valid.shape
    if target_local < 0:
        target_local = target_count + int(target_local)
    source_local = int(source_local)
    target_local = int(target_local)
    if source_local < 0 or source_local >= target_count or target_local < 0 or target_local >= target_count:
        return {
            "cycle_probe_enabled": True,
            "cycle_status": "skipped_invalid_source_or_target",
            "cycle_source_local": source_local,
            "cycle_target_local": target_local,
        }

    uv_tgt = batch.uv_pred[target_local]
    in01 = (
        np.isfinite(uv_tgt).all(axis=1)
        & (uv_tgt[:, 0] >= 0.0)
        & (uv_tgt[:, 0] <= 1.0)
        & (uv_tgt[:, 1] >= 0.0)
        & (uv_tgt[:, 1] <= 1.0)
    )
    candidates = np.flatnonzero((batch.src_frame == source_local) & batch.valid[target_local] & in01)
    if candidates.size == 0:
        return {
            "cycle_probe_enabled": True,
            "cycle_status": "skipped_no_candidates",
            "cycle_source_local": source_local,
            "cycle_target_local": target_local,
        }
    if candidates.size > max_tracks:
        keep = np.linspace(0, candidates.size - 1, num=int(max_tracks), dtype=np.int64)
        candidates = candidates[keep]

    back = adapter.infer_carriers(
        video_rgb_uint8=video_rgb_uint8,
        src_uv_norm=uv_tgt[candidates],
        src_frame_local=np.full(candidates.shape[0], target_local, dtype=np.int64),
        target_frames_local=np.asarray([source_local], dtype=np.int64),
        carrier_id=batch.carrier_id[candidates],
        src_frame_global=batch.src_frame_global[candidates] if batch.src_frame_global is not None else None,
        src_xy=batch.src_xy[candidates] if batch.src_xy is not None else None,
        src_mask_id=batch.src_mask_id[candidates] if batch.src_mask_id is not None else None,
        query_chunk_size=query_chunk_size,
    )
    model_h, model_w = int(image_hw[0]), int(image_hw[1])
    scale = np.asarray([float(max(model_w - 1, 1)), float(max(model_h - 1, 1))], dtype=np.float32)
    uv_back = back.uv_pred[0]
    uv_err = np.linalg.norm((uv_back - batch.src_uv[candidates]) * scale.reshape(1, 2), axis=1)
    xyz_back = back.xyz_ref[0]
    xyz_src = batch.xyz_ref[source_local, candidates]
    xyz_err = np.linalg.norm(xyz_back - xyz_src, axis=1)
    back_in01 = (
        np.isfinite(uv_back).all(axis=1)
        & (uv_back[:, 0] >= 0.0)
        & (uv_back[:, 0] <= 1.0)
        & (uv_back[:, 1] >= 0.0)
        & (uv_back[:, 1] <= 1.0)
    )
    return {
        "cycle_probe_enabled": True,
        "cycle_status": "ok",
        "cycle_source_local": int(source_local),
        "cycle_target_local": int(target_local),
        "cycle_num_tracks": int(candidates.shape[0]),
        "cycle_uv_error_mean": float(np.nanmean(uv_err)) if uv_err.size else None,
        "cycle_uv_error_p90": _percentile(uv_err, 90),
        "cycle_3d_error_mean": float(np.nanmean(xyz_err)) if xyz_err.size else None,
        "cycle_3d_error_p90": _percentile(xyz_err, 90),
        "forward_backward_visibility_consistency": float(np.mean(back_in01)) if back_in01.size else None,
        "cycle_extra_infer": getattr(adapter, "last_infer_diagnostics", {}),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "num_windows": int(len(rows)),
        "num_ok_windows": int(sum(1 for row in rows if row.get("status") == "ok")),
        "num_failed_windows": int(sum(1 for row in rows if row.get("status") != "ok")),
    }
    for key in (
        "num_source_queries",
        "num_surfel_tracks",
        "num_valid_tracks",
        "num_visible_observations",
        "uv_in01_rate",
        "visibility_prob_mean",
        "confidence_prob_mean",
        "rho_mean",
        "surfel_coverage_2d_per_frame_mean",
        "hole_ratio_2d_mean",
        "track_length_visible_mean",
        "track_length_visible_p10",
        "self_uv_error_mean",
        "self_uv_error_p90",
        "cycle_uv_error_mean",
        "cycle_uv_error_p90",
        "cycle_3d_error_mean",
        "cycle_3d_error_p90",
    ):
        values = [float(row[key]) for row in rows if row.get("status") == "ok" and row.get(key) is not None]
        if values:
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_min"] = float(np.min(values))
            out[f"{key}_max"] = float(np.max(values))
    return out


def _write_markdown(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# v8 D4RT Grid Surfel Field Sanity",
        "",
        "This is a method-free Lane 1 diagnostic. It does not read GT labels and does not report AP.",
        "",
        "## Command",
        "",
        "```text",
        " ".join(args.command_argv),
        "```",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Windows",
            "",
            "| Scene | Window | Status | Frames | Grid | Tracks | uv in01 | Visible len mean | Self p90 px | Cycle p90 px | Coverage mean | Seconds |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {scene} | {window} | {status} | {frames} | {grid} | {tracks} | {uv} | {length} | {selfp90} | {cyclep90} | {coverage} | {seconds} |".format(
                scene=row.get("scene", ""),
                window=row.get("window_index", ""),
                status=row.get("status", ""),
                frames=row.get("num_frames", ""),
                grid=row.get("grid_size", ""),
                tracks=row.get("num_surfel_tracks", ""),
                uv="" if row.get("uv_in01_rate") is None else f"{row['uv_in01_rate']:.6g}",
                length="" if row.get("track_length_visible_mean") is None else f"{row['track_length_visible_mean']:.6g}",
                selfp90="" if row.get("self_uv_error_p90") is None else f"{row['self_uv_error_p90']:.6g}",
                cyclep90="" if row.get("cycle_uv_error_p90") is None else f"{row['cycle_uv_error_p90']:.6g}",
                coverage="" if row.get("surfel_coverage_2d_per_frame_mean") is None else f"{row['surfel_coverage_2d_per_frame_mean']:.6g}",
                seconds="" if row.get("seconds") is None else f"{row['seconds']:.2f}",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _process_window(
    args: argparse.Namespace,
    adapter: D4RTAdapter,
    stream: ScanNetStream,
    scene_dir: Path,
    window_index: int,
    frame_window: list[int],
) -> dict[str, Any]:
    started = time.time()
    data = stream.load_window(frame_window, require_masks=not bool(args.allow_missing_masks))
    sources, source_diag = _grid_sources(
        masks=np.asarray(data["mask"]),
        frame_ids=frame_window,
        grid_size=int(args.grid_size),
        grid_margin_ratio=float(args.grid_margin_ratio),
        mask_aware_min_points_per_mask=int(args.mask_aware_min_points_per_mask),
        min_mask_area=int(args.min_mask_area),
    )
    sources.save_npz(scene_dir / f"carrier_sources_window{window_index:03d}.npz")
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
    batch.save_npz(scene_dir / f"carriers_window{window_index:03d}.npz")
    row: dict[str, Any] = {
        "status": "ok",
        "scene": stream.seq_name,
        "window_index": int(window_index),
        "frame_ids": [int(v) for v in frame_window],
        "num_frames": int(len(frame_window)),
        "grid_size": int(args.grid_size),
        "num_source_queries": int(sources.carrier_id.shape[0]),
        "seconds": None,
        **source_diag,
        **carrier_diagnostics(batch),
        **_track_metrics(
            batch=batch,
            image_hw=adapter.image_hw,
            min_visibility=float(args.visible_min_visibility),
            min_confidence=float(args.visible_min_confidence),
        ),
        "surfel_coverage_3d_after_export": None,
        "adapter_last_infer": getattr(adapter, "last_infer_diagnostics", {}),
    }
    if int(args.cycle_max_tracks) > 0:
        cycle = _cycle_probe(
            adapter=adapter,
            video_rgb_uint8=np.asarray(data["rgb"]),
            batch=batch,
            image_hw=adapter.image_hw,
            source_local=int(args.cycle_source_local),
            target_local=int(args.cycle_target_local),
            max_tracks=int(args.cycle_max_tracks),
            query_chunk_size=int(args.query_chunk_size),
        )
        row.update({key: value for key, value in cycle.items() if key != "cycle_extra_infer"})
        row["cycle_extra_infer"] = cycle.get("cycle_extra_infer", {})
    if args.save_overlays:
        for local_idx, frame_id in enumerate(frame_window[: min(4, len(frame_window))]):
            save_overlay(
                scene_dir / "overlays" / f"window{window_index:03d}_frame{int(frame_id)}.png",
                data["rgb"][local_idx],
                batch.uv_pred[local_idx],
                max_points=int(args.overlay_max_points),
            )
    row["seconds"] = float(time.time() - started)
    write_json(scene_dir / f"window{window_index:03d}_summary.json", row)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d4rt-root", required=True)
    parser.add_argument("--d4rt-config", required=True)
    parser.add_argument("--d4rt-ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-name", default="")
    parser.add_argument("--seq-list", default="")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--grid-margin-ratio", type=float, default=0.0)
    parser.add_argument("--mask-aware-min-points-per-mask", type=int, default=0)
    parser.add_argument("--min-mask-area", type=int, default=8)
    parser.add_argument("--visible-min-visibility", type=float, default=0.5)
    parser.add_argument("--visible-min-confidence", type=float, default=0.5)
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--cycle-max-tracks", type=int, default=256)
    parser.add_argument("--cycle-source-local", type=int, default=0)
    parser.add_argument("--cycle-target-local", type=int, default=-1)
    parser.add_argument("--output-root", default="outputs/v8_d4rt_grid_surfel_field")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-max-points", type=int, default=2000)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-missing-masks", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    import sys

    args.command_argv = [sys.executable, "-m", "tools.export_d4rt_grid_surfel_field_v8", *sys.argv[1:]]
    if args.seq_name:
        scenes = [args.seq_name]
    elif args.seq_list:
        scenes = _read_seq_list(Path(args.seq_list))
    else:
        raise ValueError("Provide --seq-name or --seq-list")

    output_root = Path(args.output_root) / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    adapter = D4RTAdapter(
        d4rt_root=args.d4rt_root,
        model_config=args.d4rt_config,
        ckpt_path=args.d4rt_ckpt,
        device=args.device,
    )
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        stream = ScanNetStream(seq_name=scene, backbone=args.backbone, root=args.scannet_root)
        errors = stream.validate(require_masks=not bool(args.allow_missing_masks))
        if errors:
            row = {
                "status": "failed",
                "scene": scene,
                "window_index": None,
                "failure_reason": "; ".join(errors),
            }
            rows.append(row)
            if not args.continue_on_error:
                raise RuntimeError(row["failure_reason"])
            continue
        all_frames = stream.frame_ids(stride=int(args.frame_stride), max_frames=int(args.max_frames))
        windows = _windows(all_frames, int(args.window_size), int(args.window_stride))
        scene_dir = output_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_rows: list[dict[str, Any]] = []
        for window_index, frame_window in enumerate(windows):
            print(
                f"[v8-grid-surfel] scene={scene} window={window_index + 1}/{len(windows)} "
                f"frames={frame_window[0]}..{frame_window[-1]} grid={args.grid_size}",
                flush=True,
            )
            try:
                row = _process_window(args, adapter, stream, scene_dir, window_index, frame_window)
            except Exception as exc:
                row = {
                    "status": "failed",
                    "scene": scene,
                    "window_index": int(window_index),
                    "frame_ids": [int(v) for v in frame_window],
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
                if not args.continue_on_error:
                    raise
            rows.append(row)
            scene_rows.append(row)
        write_json(scene_dir / "summary.json", {"scene": scene, "summary": _aggregate(scene_rows), "windows": scene_rows})

    summary = _aggregate(rows)
    payload = {"run_name": args.run_name, "summary": summary, "windows": rows}
    (output_root / "summary.json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys() if key not in {"adapter_last_infer", "cycle_extra_infer"}})
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})
    _write_markdown(output_root / "summary.md", args, rows, summary)
    print(f"[v8-grid-surfel] wrote {output_root / 'summary.json'}")
    print(f"[v8-grid-surfel] wrote {output_root / 'summary.md'}")
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
