from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.carrier_sampler import CarrierSampler
from stream4d.d4rt_adapter import D4RTAdapter
from stream4d.scannet_stream import ScanNetStream
from stream4d_native.occupancy_dense_tracker import (
    QueryBudget,
    filter_tracks_before_marking_occupancy,
    query_d4rt_tubes_with_spatiotemporal_occupancy,
)
from stream4d_native.occupancy_state import OccupancyCoverageTargets, SpatioTemporalOccupancyState


def _frame_ids(stream: ScanNetStream, *, stride: int, max_frames: int) -> list[int]:
    ids = stream.frame_ids(stride=stride, max_frames=max_frames)
    if len(ids) > max_frames:
        ids = ids[:max_frames]
    return ids


def _windows(frame_ids: list[int], *, window_size: int, window_stride: int) -> list[list[int]]:
    out: list[list[int]] = []
    start = 0
    while start < len(frame_ids):
        cur = frame_ids[start : start + int(window_size)]
        if cur:
            out.append(cur)
        if start + int(window_size) >= len(frame_ids):
            break
        start += max(1, int(window_stride))
    return out


def _load_rgb_mask_window(stream: ScanNetStream, frame_ids: list[int]) -> dict[str, np.ndarray]:
    return {
        "rgb": np.stack([stream.load_rgb(frame_id) for frame_id in frame_ids], axis=0),
        "mask": np.stack([stream.load_mask(frame_id) for frame_id in frame_ids], axis=0),
    }


def _full_grid_sources(num_frames: int, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(0.0, 1.0, num=max(1, int(grid_size)), dtype=np.float32)
    uu, vv = np.meshgrid(coords, coords)
    uv_one = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1).astype(np.float32)
    src_uv = np.tile(uv_one, (int(num_frames), 1))
    src_frame = np.repeat(np.arange(int(num_frames), dtype=np.int64), uv_one.shape[0])
    return src_uv, src_frame


def _tracks_from_batch(batch: Any) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    num_tracks = int(batch.uv_pred.shape[1])
    for idx in range(num_tracks):
        tracks.append(
            {
                "uv_norm": np.asarray(batch.uv_pred[:, idx, :], dtype=np.float32),
                "xyz": np.asarray(batch.xyz_ref[:, idx, :], dtype=np.float32),
                "visibility": np.asarray(batch.visibility_prob[:, idx], dtype=np.float32),
                "confidence": np.asarray(batch.confidence_prob[:, idx], dtype=np.float32),
                "valid": np.asarray(batch.valid[:, idx], dtype=bool),
            }
        )
    return tracks


def _batch_uv_in01(batch: Any) -> float:
    uv = np.asarray(batch.uv_pred, dtype=np.float32)
    valid = np.asarray(batch.valid, dtype=bool)
    in_bounds = (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    return float(np.mean(valid & in_bounds)) if valid.size else 0.0


def _batch_visible_track_length(batch: Any, *, min_visibility: float, min_confidence: float) -> float:
    valid = np.asarray(batch.valid, dtype=bool)
    visibility = np.asarray(batch.visibility_prob, dtype=np.float32)
    confidence = np.asarray(batch.confidence_prob, dtype=np.float32)
    ok = valid & (visibility >= float(min_visibility)) & (confidence >= float(min_confidence))
    if ok.size == 0 or ok.shape[1] == 0:
        return 0.0
    return float(np.mean(np.count_nonzero(ok, axis=0) / max(ok.shape[0], 1)))


def _tracks_uv_in01(tracks: list[dict[str, Any]]) -> float:
    if not tracks:
        return 0.0
    values: list[np.ndarray] = []
    for track in tracks:
        uv = np.asarray(track.get("uv_norm", []), dtype=np.float32)
        valid = np.asarray(track.get("valid", np.ones((uv.shape[0],), dtype=bool)), dtype=bool)
        if uv.size == 0:
            continue
        in_bounds = (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
        values.append(valid & in_bounds)
    if not values:
        return 0.0
    return float(np.mean(np.concatenate(values, axis=0)))


def _tracks_visible_track_length(
    tracks: list[dict[str, Any]],
    *,
    min_visibility: float,
    min_confidence: float,
) -> float:
    if not tracks:
        return 0.0
    lengths: list[float] = []
    for track in tracks:
        visibility = np.asarray(track.get("visibility", []), dtype=np.float32)
        confidence = np.asarray(track.get("confidence", []), dtype=np.float32)
        valid = np.asarray(track.get("valid", np.ones_like(visibility, dtype=bool)), dtype=bool)
        if visibility.size == 0:
            continue
        ok = valid & (visibility >= float(min_visibility)) & (confidence >= float(min_confidence))
        lengths.append(float(np.count_nonzero(ok) / max(ok.shape[0], 1)))
    return float(np.mean(lengths)) if lengths else 0.0


def _mark_tracks(
    *,
    tracks: list[dict[str, Any]],
    frames_shape: tuple[int, int, int, int],
    masks: np.ndarray | None,
    targets: OccupancyCoverageTargets,
) -> dict[str, Any]:
    state = SpatioTemporalOccupancyState(
        num_frames=int(frames_shape[0]),
        image_height=int(frames_shape[1]),
        image_width=int(frames_shape[2]),
        masks=masks,
    )
    filtered = filter_tracks_before_marking_occupancy(
        tracks,
        min_visibility=float(targets.min_visibility),
        min_confidence=float(targets.min_confidence),
    )
    for idx, track in enumerate(filtered):
        state.mark_visible_track_as_visited(track=track, tube_id=idx, mark_radius_px=int(targets.mark_radius_px))
    return state.summarize(query_budget_hit=False, total_time_sec=None)


def _mask_pixel_count(masks: np.ndarray | None) -> int | None:
    if masks is None:
        return None
    return int(np.count_nonzero(np.asarray(masks) > 0))


def _fixed_variant(
    *,
    name: str,
    label: str,
    adapter: D4RTAdapter,
    frames: np.ndarray,
    masks: np.ndarray | None,
    src_uv: np.ndarray,
    src_frame: np.ndarray,
    query_chunk_size: int,
    targets: OccupancyCoverageTargets,
    source_policy: str,
) -> dict[str, Any]:
    t0 = time.time()
    batch = adapter.infer_carriers(
        video_rgb_uint8=frames,
        src_uv_norm=src_uv,
        src_frame_local=src_frame,
        query_chunk_size=query_chunk_size,
    )
    runtime = float(time.time() - t0)
    coverage = _mark_tracks(tracks=_tracks_from_batch(batch), frames_shape=frames.shape, masks=masks, targets=targets)
    actual = int(src_uv.shape[0])
    naive = int(frames.shape[0] * frames.shape[1] * frames.shape[2])
    mask_pixels = _mask_pixel_count(masks)
    coverage_fields = {
        key: coverage.get(key)
        for key in [
            "pixel_occupancy_coverage_mean",
            "pixel_occupancy_coverage_p10",
            "mask_interior_coverage_mean",
            "mask_interior_coverage_p10",
            "mask_boundary_coverage_mean",
            "mask_boundary_coverage_p10",
            "overlap_anchor_coverage",
            "unvisited_large_mask_count",
            "unvisited_boundary_count",
            "duplicate_track_rate",
            "redundant_query_rate",
            "coverage_saturation_round",
        ]
    }
    out = {
        "variant": name,
        "label": label,
        "status": "ok",
        "source_policy": source_policy,
        "uses_spatiotemporal_occupancy": False,
        "actual_source_query_count": actual,
        "naive_source_query_count": naive,
        "adaptive_speedup_vs_naive": float(naive / max(actual, 1)),
        "semantic_adaptive_speedup": None if mask_pixels is None else float(mask_pixels / max(actual, 1)),
        "num_output_tubes": actual,
        "uv_in01_rate": _batch_uv_in01(batch),
        "visible_track_length_mean": _batch_visible_track_length(
            batch,
            min_visibility=float(targets.min_visibility),
            min_confidence=float(targets.min_confidence),
        ),
        "query_budget_hit": False,
        "total_d4rt_time_sec": runtime,
        **coverage_fields,
        "num_kept_tubes": int(coverage.get("num_output_tubes", 0)),
        **{f"adapter_{key}": value for key, value in getattr(adapter, "last_infer_diagnostics", {}).items()},
    }
    return out


def _occupancy_variant(
    *,
    name: str,
    label: str,
    adapter: D4RTAdapter,
    frames: np.ndarray,
    masks: np.ndarray | None,
    query_chunk_size: int,
    targets: OccupancyCoverageTargets,
    budget: QueryBudget,
    source_policy: str,
) -> dict[str, Any]:
    def decode(source_points: np.ndarray) -> list[dict[str, Any]]:
        src_frame = source_points[:, 0].astype(np.int64)
        src_uv = source_points[:, 1:3].astype(np.float32)
        batch = adapter.infer_carriers(
            video_rgb_uint8=frames,
            src_uv_norm=src_uv,
            src_frame_local=src_frame,
            query_chunk_size=query_chunk_size,
        )
        return _tracks_from_batch(batch)

    tubes, diag = query_d4rt_tubes_with_spatiotemporal_occupancy(
        frames=frames,
        masks=masks,
        decode_source_points=decode,
        coverage_targets=targets,
        query_budget=budget,
    )
    mask_pixels = _mask_pixel_count(masks)
    actual = int(diag.get("actual_source_query_count", 0))
    return {
        "variant": name,
        "label": label,
        "status": "ok",
        "source_policy": source_policy,
        **diag,
        "semantic_adaptive_speedup": None if mask_pixels is None else float(mask_pixels / max(actual, 1)),
        "num_kept_tubes": int(len(tubes)),
        "uv_in01_rate": _tracks_uv_in01(tubes),
        "visible_track_length_mean": _tracks_visible_track_length(
            tubes,
            min_visibility=float(targets.min_visibility),
            min_confidence=float(targets.min_confidence),
        ),
    }


def _warmstart_tracks_for_window(
    *,
    previous_tracks: list[dict[str, Any]],
    previous_frame_ids: list[int],
    current_frame_ids: list[int],
) -> list[dict[str, Any]]:
    if not previous_tracks:
        return []
    prev_index = {int(frame_id): idx for idx, frame_id in enumerate(previous_frame_ids)}
    shared = [(cur_idx, prev_index[int(frame_id)]) for cur_idx, frame_id in enumerate(current_frame_ids) if int(frame_id) in prev_index]
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
        for cur_idx, prev_idx in shared:
            uv[cur_idx] = uv_prev[prev_idx]
            visibility[cur_idx] = visibility_prev[prev_idx]
            confidence[cur_idx] = confidence_prev[prev_idx]
            valid[cur_idx] = valid_prev[prev_idx]
        if np.count_nonzero(valid) == 0:
            continue
        warmstart.append(
            {
                "uv_norm": uv,
                "visibility": visibility,
                "confidence": confidence,
                "valid": valid,
            }
        )
    return warmstart


def _d5_overlap_warmstart_variant(
    *,
    stream: ScanNetStream,
    frame_ids: list[int],
    adapter: D4RTAdapter,
    args: argparse.Namespace,
    targets: OccupancyCoverageTargets,
) -> dict[str, Any]:
    window_ids = _windows(frame_ids, window_size=int(args.window_size), window_stride=int(args.window_stride))
    previous_tracks: list[dict[str, Any]] = []
    previous_frame_ids: list[int] = []
    window_rows: list[dict[str, Any]] = []
    all_tubes: list[dict[str, Any]] = []
    for window_index, cur_frame_ids in enumerate(window_ids):
        data = _load_rgb_mask_window(stream, cur_frame_ids)
        frames = np.asarray(data["rgb"])
        masks = np.asarray(data["mask"])
        warmstart = _warmstart_tracks_for_window(
            previous_tracks=previous_tracks,
            previous_frame_ids=previous_frame_ids,
            current_frame_ids=cur_frame_ids,
        )

        def decode(source_points: np.ndarray) -> list[dict[str, Any]]:
            src_frame = source_points[:, 0].astype(np.int64)
            src_uv = source_points[:, 1:3].astype(np.float32)
            batch = adapter.infer_carriers(
                video_rgb_uint8=frames,
                src_uv_norm=src_uv,
                src_frame_local=src_frame,
                query_chunk_size=int(args.query_chunk_size),
            )
            return _tracks_from_batch(batch)

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
        diag["window_index"] = int(window_index)
        diag["frame_start"] = int(cur_frame_ids[0])
        diag["frame_end"] = int(cur_frame_ids[-1])
        window_rows.append(diag)
        all_tubes.extend(tubes)
        previous_tracks = tubes
        previous_frame_ids = list(cur_frame_ids)

    def sum_metric(key: str) -> float:
        return float(sum(float(row.get(key, 0.0) or 0.0) for row in window_rows))

    def mean_metric(key: str) -> float | None:
        values = [row.get(key) for row in window_rows if row.get(key) is not None]
        if not values:
            return None
        return float(sum(float(v) for v in values) / len(values))

    mask_pixels = 0
    for cur_frame_ids in window_ids:
        mask_pixels += _mask_pixel_count(_load_rgb_mask_window(stream, cur_frame_ids)["mask"]) or 0
    actual = int(sum_metric("actual_source_query_count"))
    return {
        "variant": "D5_occupancy_dense_overlap_warmstart",
        "label": "occupancy mask-aware overlap warmstart",
        "status": "ok",
        "source_policy": "occupancy_mask_aware_overlap_warmstart",
        "uses_spatiotemporal_occupancy": True,
        "actual_source_query_count": actual,
        "naive_source_query_count": int(sum_metric("naive_source_query_count")),
        "adaptive_speedup_vs_naive": float(sum_metric("naive_source_query_count") / max(actual, 1)),
        "semantic_adaptive_speedup": float(mask_pixels / max(actual, 1)),
        "num_output_tubes": int(sum_metric("num_output_tubes")),
        "num_kept_tubes": int(len(all_tubes)),
        "pixel_occupancy_coverage_mean": mean_metric("pixel_occupancy_coverage_mean"),
        "pixel_occupancy_coverage_p10": mean_metric("pixel_occupancy_coverage_p10"),
        "mask_interior_coverage_mean": mean_metric("mask_interior_coverage_mean"),
        "mask_interior_coverage_p10": mean_metric("mask_interior_coverage_p10"),
        "mask_boundary_coverage_mean": mean_metric("mask_boundary_coverage_mean"),
        "mask_boundary_coverage_p10": mean_metric("mask_boundary_coverage_p10"),
        "uv_in01_rate": _tracks_uv_in01(all_tubes),
        "visible_track_length_mean": _tracks_visible_track_length(
            all_tubes,
            min_visibility=float(targets.min_visibility),
            min_confidence=float(targets.min_confidence),
        ),
        "query_budget_hit": any(bool(row.get("query_budget_hit")) for row in window_rows),
        "total_d4rt_time_sec": sum_metric("total_d4rt_time_sec"),
        "warmstart_track_count": int(sum_metric("warmstart_track_count")),
        "num_windows": int(len(window_rows)),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "window_rows": window_rows,
    }


def _write_outputs(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    payload = {"rows": rows}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    lines = [
        "# v21.3 Native Occupancy Ablation Diagnostic",
        "",
        "| variant | queries | occupancy | pixel cov | mask interior | mask boundary | uv in01 | visible len | budget hit | time sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            value = row.get(key)
            if value is None:
                return "NA"
            if isinstance(value, bool):
                return str(value)
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.6f}"
            return str(value)

        lines.append(
            "| {variant} | {queries} | {occ} | {pixel} | {interior} | {boundary} | {uv} | {visible} | {budget} | {time} |".format(
                variant=fmt("variant"),
                queries=fmt("actual_source_query_count"),
                occ=fmt("uses_spatiotemporal_occupancy"),
                pixel=fmt("pixel_occupancy_coverage_mean"),
                interior=fmt("mask_interior_coverage_mean"),
                boundary=fmt("mask_boundary_coverage_mean"),
                uv=fmt("uv_in01_rate"),
                visible=fmt("visible_track_length_mean"),
                budget=fmt("query_budget_hit"),
                time=fmt("total_d4rt_time_sec"),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d4rt-root", required=True)
    parser.add_argument("--d4rt-config", required=True)
    parser.add_argument("--d4rt-ckpt", required=True)
    parser.add_argument("--seq-name", default="scene0050_00")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--variants", default="D0,D2,D3,D4")
    parser.add_argument("--fixed-grid-size", type=int, default=32)
    parser.add_argument("--mask-fixed-points-per-mask", type=int, default=32)
    parser.add_argument("--mask-fixed-min-points-per-mask", type=int, default=4)
    parser.add_argument("--query-budget", type=int, default=4096)
    parser.add_argument("--source-points-per-round", type=int, default=512)
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--mark-radius-px", type=int, default=2)
    parser.add_argument("--pixel-coverage-target", type=float, default=0.02)
    parser.add_argument("--mask-interior-coverage-target", type=float, default=0.20)
    parser.add_argument("--boundary-coverage-target", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-prefix", default="outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stream = ScanNetStream(seq_name=args.seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    frame_ids = _frame_ids(stream, stride=args.frame_stride, max_frames=args.max_frames)
    data = _load_rgb_mask_window(stream, frame_ids)
    frames = np.asarray(data["rgb"])
    masks = np.asarray(data["mask"])
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
    variants = [item.strip().upper() for item in args.variants.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"[occupancy_ablation] start variant={variant}", flush=True)
        if variant == "D0":
            src_uv, src_frame = _full_grid_sources(frames.shape[0], grid_size=args.fixed_grid_size)
            rows.append(
                _fixed_variant(
                    name="D0_fixed_full_grid32",
                    label="fixed full-image grid",
                    adapter=adapter,
                    frames=frames,
                    masks=None,
                    src_uv=src_uv,
                    src_frame=src_frame,
                    query_chunk_size=int(args.query_chunk_size),
                    targets=targets,
                    source_policy=f"full_grid_{args.fixed_grid_size}x{args.fixed_grid_size}_per_frame",
                )
            )
        elif variant == "D1":
            src_uv, src_frame = _full_grid_sources(frames.shape[0], grid_size=48)
            rows.append(
                _fixed_variant(
                    name="D1_fixed_full_grid48",
                    label="fixed full-image grid48",
                    adapter=adapter,
                    frames=frames,
                    masks=None,
                    src_uv=src_uv,
                    src_frame=src_frame,
                    query_chunk_size=int(args.query_chunk_size),
                    targets=targets,
                    source_policy="full_grid_48x48_per_frame",
                )
            )
        elif variant == "D2":
            sampler = CarrierSampler(
                max_points_per_mask=int(args.mask_fixed_points_per_mask),
                min_points_per_mask=int(args.mask_fixed_min_points_per_mask),
                strategy="grid_inside_mask",
            )
            sources = sampler.sample(masks=masks, frame_ids=frame_ids)
            rows.append(
                _fixed_variant(
                    name="D2_mask_aware_fixed32",
                    label="fixed mask-aware grid",
                    adapter=adapter,
                    frames=frames,
                    masks=masks,
                    src_uv=sources.src_uv,
                    src_frame=sources.src_frame,
                    query_chunk_size=int(args.query_chunk_size),
                    targets=targets,
                    source_policy=f"mask_grid_inside_mask_{args.mask_fixed_points_per_mask}_per_mask",
                )
            )
        elif variant == "D3":
            rows.append(
                _occupancy_variant(
                    name="D3_occupancy_dense_uniform",
                    label="occupancy uniform",
                    adapter=adapter,
                    frames=frames,
                    masks=None,
                    query_chunk_size=int(args.query_chunk_size),
                    targets=targets,
                    budget=QueryBudget(
                        max_source_points=int(args.query_budget),
                        source_points_per_round=int(args.source_points_per_round),
                    ),
                    source_policy="occupancy_uniform_unvisited",
                )
            )
        elif variant == "D4":
            rows.append(
                _occupancy_variant(
                    name="D4_occupancy_dense_mask_aware",
                    label="occupancy mask-aware",
                    adapter=adapter,
                    frames=frames,
                    masks=masks,
                    query_chunk_size=int(args.query_chunk_size),
                    targets=targets,
                    budget=QueryBudget(
                        max_source_points=int(args.query_budget),
                        source_points_per_round=int(args.source_points_per_round),
                    ),
                    source_policy="occupancy_mask_interior_then_boundary_then_uniform",
                )
            )
        elif variant == "D5":
            rows.append(
                _d5_overlap_warmstart_variant(
                    stream=stream,
                    frame_ids=frame_ids,
                    adapter=adapter,
                    args=args,
                    targets=targets,
                )
            )
        else:
            rows.append({"variant": variant, "status": "not_run", "error": "unsupported variant"})
        print(f"[occupancy_ablation] done variant={variant}", flush=True)
    for row in rows:
        row.setdefault("seq_name", args.seq_name)
        row.setdefault("frame_start", int(frame_ids[0]))
        row.setdefault("frame_end", int(frame_ids[-1]))
        row.setdefault("num_frames", int(len(frame_ids)))
        row.setdefault("image_height", int(frames.shape[1]))
        row.setdefault("image_width", int(frames.shape[2]))
        row.setdefault("is_diagnostic_only", True)
        row.setdefault("uses_gt_for_prediction", False)
        row.setdefault("uses_rgbd_for_prediction", False)
        row.setdefault("uses_pose_for_prediction", False)
        row.setdefault("uses_scannet_mesh_for_prediction", False)
    _write_outputs(rows, Path(args.output_prefix))


if __name__ == "__main__":
    main()
