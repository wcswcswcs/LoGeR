#!/usr/bin/env python3
"""Decode v96 D4RT micro-primitive tracks from Phase1 query plans."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.d4rt_adapter import D4RTAdapter  # noqa: E402
from stream4d.scannet_stream import ScanNetStream  # noqa: E402


PHASE_ID = "v96_phase2_d4rt_micro_tracks"
RUN_ID = "v96_phase2_d4rt_micro_tracks"
DEFAULT_QUERY_ROOT = ROOT / "outputs/audit/v96_phase1_query_planner"
DEFAULT_OUT = ROOT / "outputs/audit/v96_phase2_d4rt_micro_tracks"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_SCANNET = ROOT / "data/scannet/processed"
DEFAULT_D4RT_ROOT = REPO_ROOT / "Open-d4rt"
DEFAULT_D4RT_CONFIG = DEFAULT_D4RT_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
DEFAULT_D4RT_CKPT = DEFAULT_D4RT_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"
DEFAULT_VARIANT_MAP = {
    "D0_uniform256": "Q0_uniform256",
    "D1_uniform1024": "Q1_uniform1024",
    "D2_adaptive512": "Q2_adaptive512",
    "D3_adaptive1024": "Q3_adaptive1024",
    "D4_occupancy_adaptive1024": "Q5_occupancy_adaptive1024",
}

MICRO_QUERY_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "decode_variant",
    "query_variant",
    "scene_id",
    "split",
    "window_id",
    "frame_id",
    "query_id",
    "query_u",
    "query_v",
    "query_u_norm",
    "query_v_norm",
    "query_stratum",
    "query_priority",
    "source_mask_id_optional",
    "semantic_gradient_score",
    "mask_conflict_score",
    "occupancy_before",
    "query_u_original_px",
    "query_v_original_px",
    "d4rt_input_width",
    "d4rt_input_height",
    "d4rt_output_width",
    "d4rt_output_height",
    "coordinate_grid",
    "uses_gt_for_query_selection",
    "uses_gt_for_prediction",
    "uses_future",
]
MICRO_TRACK_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "decode_variant",
    "query_variant",
    "scene_id",
    "window_id",
    "query_id",
    "source_frame_id",
    "target_frame_id",
    "u_src",
    "v_src",
    "u_tgt",
    "v_tgt",
    "x_3d",
    "y_3d",
    "z_3d",
    "visibility",
    "confidence",
    "uv_in01",
    "query_stratum",
    "u_src_original_px",
    "v_src_original_px",
    "d4rt_input_width",
    "d4rt_input_height",
    "d4rt_output_width",
    "d4rt_output_height",
    "coordinate_grid",
    "track_status",
    "uses_gt_for_prediction",
    "uses_future",
]


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return _rel(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_variant_map(raw: str) -> dict[str, str]:
    if not raw:
        return dict(DEFAULT_VARIANT_MAP)
    out: dict[str, str] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"variant map entry must be D=Q, got {part!r}")
        left, right = part.split("=", 1)
        out[left.strip()] = right.strip()
    return out


def _load_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask label image: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _mask_path_lookup(source_rows: Path) -> dict[tuple[str, str, int], Path]:
    out: dict[tuple[str, str, int], Path] = {}
    for row in _read_csv(source_rows):
        key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))
        if key in out:
            continue
        raw = row.get("mask_path", "")
        if raw:
            out[key] = _project(raw)
    return out


def _load_rgb_window(
    stream: ScanNetStream,
    frame_ids: list[int],
    *,
    resize_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    frames = [stream.load_rgb(fid) for fid in frame_ids]
    if resize_hw is not None:
        height, width = int(resize_hw[0]), int(resize_hw[1])
        frames = [
            cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_AREA if frame.shape[0] >= height and frame.shape[1] >= width else cv2.INTER_LINEAR,
            )
            for frame in frames
        ]
    return np.stack(frames, axis=0)


def _grid_arg(value: Any) -> int:
    return max(0, int(_num(value, 0.0)))


def _query_px_on_grid(row: dict[str, str], width: int, height: int) -> tuple[float, float]:
    if width > 0 and height > 0:
        u_norm = _num(row.get("query_u_norm"), 0.0)
        v_norm = _num(row.get("query_v_norm"), 0.0)
        return (
            float(np.clip(u_norm, 0.0, 1.0) * float(max(1, width - 1))),
            float(np.clip(v_norm, 0.0, 1.0) * float(max(1, height - 1))),
        )
    return (_num(row.get("query_u", row.get("query_x")), 0.0), _num(row.get("query_v", row.get("query_y")), 0.0))


def _available_consecutive_frame_ids(stream: ScanNetStream, start: int, count: int) -> list[int]:
    frame_ids: list[int] = []
    for frame_id in range(int(start), int(start) + int(count)):
        if (stream.rgb_dir / f"{int(frame_id)}.jpg").exists():
            frame_ids.append(int(frame_id))
    if not frame_ids:
        raise FileNotFoundError(f"no RGB frames found for contiguous D4RT clip from frame {start}")
    return frame_ids


def _label_boundary(label: np.ndarray, band_px: int = 4) -> np.ndarray:
    positive = label > 0
    edge = np.zeros(label.shape, dtype=np.uint8)
    diff_x = label[:, 1:] != label[:, :-1]
    diff_y = label[1:, :] != label[:-1, :]
    edge[:, 1:] |= diff_x
    edge[:, :-1] |= diff_x
    edge[1:, :] |= diff_y
    edge[:-1, :] |= diff_y
    edge &= positive.astype(np.uint8)
    kernel = np.ones((max(1, int(band_px)) * 2 + 1, max(1, int(band_px)) * 2 + 1), dtype=np.uint8)
    return cv2.dilate(edge, kernel, iterations=1).astype(bool) & positive


def _mark_disk(canvas: np.ndarray, y: int, x: int, radius: int) -> None:
    h, w = canvas.shape
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(canvas, (int(x), int(y)), int(radius), 1, thickness=-1)


def _read_selected_queries(
    query_path: Path,
    variant_map: dict[str, str],
    *,
    scenes: set[str],
    window_ids: set[str],
    max_windows: int,
    max_queries_per_frame: int,
    frame_id_min: int,
    frame_id_max: int,
) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    q_to_d = {q: d for d, q in variant_map.items()}
    selected: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    seen_windows: list[tuple[str, str]] = []
    with query_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qv = row.get("query_variant", "")
            if qv not in q_to_d:
                continue
            scene = row.get("scene_id", "")
            window = row.get("window_id", "")
            frame_id = int(_num(row.get("frame_id")))
            if scenes and scene not in scenes:
                continue
            if window_ids and window not in window_ids:
                continue
            if frame_id_min >= 0 and frame_id < frame_id_min:
                continue
            if frame_id_max >= 0 and frame_id > frame_id_max:
                continue
            sw = (scene, window)
            if sw not in seen_windows:
                if max_windows > 0 and len(seen_windows) >= max_windows:
                    continue
                seen_windows.append(sw)
            selected[(q_to_d[qv], scene, window)].append(row)
    if max_queries_per_frame > 0:
        capped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        for group_key, rows in selected.items():
            by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                by_frame[int(_num(row.get("frame_id")))].append(row)
            for frame_id, frame_rows in sorted(by_frame.items()):
                capped[group_key].extend(_balanced_cap_frame_rows(frame_rows, int(max_queries_per_frame)))
        selected = capped
    return selected


def _balanced_cap_frame_rows(rows: list[dict[str, str]], cap: int) -> list[dict[str, str]]:
    if cap <= 0 or len(rows) <= cap:
        return list(rows)
    by_stratum: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_stratum[row.get("query_stratum", "")].append(row)
    total = len(rows)
    raw_alloc: dict[str, float] = {name: cap * (len(vals) / max(total, 1)) for name, vals in by_stratum.items()}
    alloc: dict[str, int] = {name: min(len(by_stratum[name]), int(math.floor(value))) for name, value in raw_alloc.items()}
    remaining = cap - sum(alloc.values())
    fractional = sorted(raw_alloc, key=lambda name: (raw_alloc[name] - math.floor(raw_alloc[name]), len(by_stratum[name])), reverse=True)
    while remaining > 0 and fractional:
        progressed = False
        for name in fractional:
            if remaining <= 0:
                break
            if alloc[name] < len(by_stratum[name]):
                alloc[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    out: list[dict[str, str]] = []
    for name in sorted(by_stratum, key=lambda s: ({"uniform": 0, "interior": 1, "boundary": 2, "conflict": 3, "semgrad": 4}.get(s, 9), s)):
        out.extend(by_stratum[name][: alloc.get(name, 0)])
    return out[:cap]


def _quality_for_batch(
    *,
    rows: list[dict[str, str]],
    frame_ids: list[int],
    model_frame_ids: list[int],
    uv: np.ndarray,
    xyz: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    stream: ScanNetStream,
    scene: str,
    window_id: str,
    mask_lookup: dict[tuple[str, str, int], Path],
    occupancy_radius_px: int,
    min_visibility: float,
    min_confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_count = len(frame_ids)
    valid = valid[:target_count]
    uv = uv[:target_count]
    xyz = xyz[:target_count]
    visibility = visibility[:target_count]
    confidence = confidence[:target_count]
    in01 = valid & (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    accepted = in01 & (visibility >= float(min_visibility)) & (confidence >= float(min_confidence))
    track_lengths = np.count_nonzero(accepted, axis=0) if accepted.size else np.zeros((len(rows),), dtype=np.int64)
    jitter_values: list[float] = []
    flip_values: list[float] = []
    occupancy_rows: list[dict[str, Any]] = []

    labels_by_frame: dict[int, np.ndarray] = {}
    boundary_by_frame: dict[int, np.ndarray] = {}
    for frame_id in frame_ids:
        mask_path = mask_lookup.get((scene, window_id, int(frame_id)))
        if mask_path is not None and mask_path.exists():
            label = _load_label(mask_path).astype(np.int64, copy=False)
        else:
            label = stream.load_mask(frame_id).astype(np.int64, copy=False)
        labels_by_frame[int(frame_id)] = label
        boundary_by_frame[int(frame_id)] = _label_boundary(label)

    support_total = 0
    support_hit = 0
    boundary_total = 0
    boundary_hit = 0
    conflict_total = 0
    conflict_hit = 0
    for local_idx, frame_id in enumerate(frame_ids):
        label = labels_by_frame[int(frame_id)]
        boundary = boundary_by_frame[int(frame_id)]
        h, w = label.shape
        cover = np.zeros((h, w), dtype=np.uint8)
        for qidx in np.flatnonzero(accepted[local_idx]):
            x = int(np.clip(round(float(uv[local_idx, qidx, 0]) * (w - 1)), 0, w - 1))
            y = int(np.clip(round(float(uv[local_idx, qidx, 1]) * (h - 1)), 0, h - 1))
            _mark_disk(cover, y, x, int(occupancy_radius_px))
        cover_bool = cover.astype(bool)
        foreground = label > 0
        support_total += int(np.count_nonzero(foreground))
        support_hit += int(np.count_nonzero(cover_bool & foreground))
        boundary_total += int(np.count_nonzero(boundary))
        boundary_hit += int(np.count_nonzero(cover_bool & boundary))
        conflict_total += int(np.count_nonzero(boundary))
        conflict_hit += int(np.count_nonzero(cover_bool & boundary))
        occupancy_rows.append(
            {
                "schema_version": "stream4d_v96_micro_occupancy_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "target_frame_id": int(frame_id),
                "covered_pixel_count": int(np.count_nonzero(cover_bool)),
                "source_foreground_pixel_count": int(np.count_nonzero(foreground)),
                "covered_source_pixel_count": int(np.count_nonzero(cover_bool & foreground)),
                "boundary_pixel_count": int(np.count_nonzero(boundary)),
                "covered_boundary_pixel_count": int(np.count_nonzero(cover_bool & boundary)),
                "occupancy_radius_px": int(occupancy_radius_px),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    for qidx in range(len(rows)):
        xy = uv[:, qidx, :]
        ok = accepted[:, qidx]
        if np.count_nonzero(ok) >= 2:
            diffs = np.diff(xy[ok], axis=0)
            jitter_values.append(float(np.mean(np.linalg.norm(diffs, axis=1))))
        labels: list[int] = []
        for local_idx, frame_id in enumerate(frame_ids):
            if not bool(in01[local_idx, qidx]):
                continue
            label = labels_by_frame[int(frame_id)]
            h, w = label.shape
            x = int(np.clip(round(float(uv[local_idx, qidx, 0]) * (w - 1)), 0, w - 1))
            y = int(np.clip(round(float(uv[local_idx, qidx, 1]) * (h - 1)), 0, h - 1))
            labels.append(int(label[y, x]))
        if len(labels) >= 2:
            flips = sum(1 for a, b in zip(labels[:-1], labels[1:]) if a != b)
            flip_values.append(float(flips / max(1, len(labels) - 1)))

    quality = {
        "query_count": int(len(rows)),
        "target_frame_count": int(target_count),
        "valid_track_ratio": float(np.mean(valid)) if valid.size else 0.0,
        "uv_in01_rate": float(np.mean(in01)) if in01.size else 0.0,
        "visibility_mean": float(np.mean(visibility[valid])) if np.any(valid) else 0.0,
        "confidence_mean": float(np.mean(confidence[valid])) if np.any(valid) else 0.0,
        "track_length_visible_mean": float(np.mean(track_lengths)) if track_lengths.size else 0.0,
        "track_length_visible_p10": float(np.percentile(track_lengths, 10)) if track_lengths.size else 0.0,
        "projection_jitter_mean": float(np.mean(jitter_values)) if jitter_values else 0.0,
        "projection_jitter_p90": float(np.percentile(jitter_values, 90)) if jitter_values else 0.0,
        "mask_membership_flip_rate": float(np.mean(flip_values)) if flip_values else 0.0,
        "source_support_area_ratio": float(support_hit / max(1, support_total)),
        "boundary_band_support_ratio": float(boundary_hit / max(1, boundary_total)),
        "competing_edge_support_ratio": float(conflict_hit / max(1, conflict_total)),
        "semantic_gradient_support_ratio": float(boundary_hit / max(1, boundary_total)),
        "model_frame_ids": ",".join(str(int(v)) for v in model_frame_ids),
        "frame_ids": ",".join(str(int(v)) for v in frame_ids),
    }
    return quality, occupancy_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    query_root = _project(args.query_root)
    query_path = query_root / "query_plan_rows.csv"
    mask_lookup = _mask_path_lookup(_project(args.source_rows))
    d4rt_input_width = _grid_arg(getattr(args, "d4rt_input_width", 0))
    d4rt_input_height = _grid_arg(getattr(args, "d4rt_input_height", 0))
    d4rt_output_width = _grid_arg(getattr(args, "d4rt_output_width", 0))
    d4rt_output_height = _grid_arg(getattr(args, "d4rt_output_height", 0))
    d4rt_input_hw = (d4rt_input_height, d4rt_input_width) if d4rt_input_width > 0 and d4rt_input_height > 0 else None
    fixed_output_grid = d4rt_output_width > 0 and d4rt_output_height > 0
    coordinate_grid = f"fixed_{d4rt_output_width}x{d4rt_output_height}" if fixed_output_grid else "source_mask_resolution"
    variant_map = _parse_variant_map(args.variant_map)
    selected_variants = {part.strip() for part in args.decode_variants.split(",") if part.strip()}
    if selected_variants:
        variant_map = {d: q for d, q in variant_map.items() if d in selected_variants}
    scenes = {part.strip() for part in args.scenes.split(",") if part.strip()}
    window_ids = {part.strip() for part in args.window_ids.split(",") if part.strip()}
    selected = _read_selected_queries(
        query_path,
        variant_map,
        scenes=scenes,
        window_ids=window_ids,
        max_windows=int(args.max_windows),
        max_queries_per_frame=int(args.max_queries_per_frame),
        frame_id_min=int(args.frame_id_min),
        frame_id_max=int(args.frame_id_max),
    )
    if not selected:
        raise RuntimeError("No Phase1 query rows selected for Phase2 decode.")

    micro_query_path = output_root / "micro_query_rows.csv"
    micro_track_path = output_root / "micro_track_rows.csv"
    query_handle = micro_query_path.open("w", newline="", encoding="utf-8")
    track_handle = micro_track_path.open("w", newline="", encoding="utf-8")
    query_writer = csv.DictWriter(query_handle, fieldnames=MICRO_QUERY_FIELDS)
    track_writer = csv.DictWriter(track_handle, fieldnames=MICRO_TRACK_FIELDS)
    query_writer.writeheader()
    track_writer.writeheader()

    quality_rows: list[dict[str, Any]] = []
    occupancy_rows_all: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    adapter = D4RTAdapter(
        d4rt_root=_project(args.d4rt_root),
        model_config=_project(args.d4rt_config),
        ckpt_path=_project(args.d4rt_ckpt),
        device=args.device,
    )
    stream_cache: dict[str, ScanNetStream] = {}

    try:
        for group_idx, ((decode_variant, scene, window), rows) in enumerate(sorted(selected.items())):
            group_started = time.time()
            query_variant = variant_map[decode_variant]
            stream = stream_cache.get(scene)
            if stream is None:
                stream = ScanNetStream(scene, root=_project(args.scannet_root))
                errs = stream.validate(require_masks=False)
                if errs:
                    raise RuntimeError("; ".join(errs))
                stream_cache[scene] = stream
            active_frame_ids = sorted({int(_num(row.get("frame_id"))) for row in rows})
            if args.model_frame_mode == "active_sparse":
                target_frame_ids = list(active_frame_ids)
                model_frame_ids = target_frame_ids if len(target_frame_ids) >= 2 else [target_frame_ids[0], target_frame_ids[0]]
                decode_rows = list(rows)
            elif args.model_frame_mode == "contiguous32_from_first":
                model_frame_ids = _available_consecutive_frame_ids(
                    stream,
                    active_frame_ids[0],
                    int(args.contiguous_frame_count),
                )
                model_frame_set = set(model_frame_ids)
                target_frame_ids = [frame_id for frame_id in active_frame_ids if frame_id in model_frame_set]
                decode_rows = [row for row in rows if int(_num(row.get("frame_id"))) in model_frame_set]
                if not target_frame_ids or not decode_rows:
                    errors.append(
                        {
                            "decode_variant": decode_variant,
                            "query_variant": query_variant,
                            "scene_id": scene,
                            "window_id": window,
                            "query_count": len(rows),
                            "error": (
                                "No active source/target frame remains inside contiguous D4RT clip; "
                                f"active_frame_ids={active_frame_ids} model_frame_ids={model_frame_ids}"
                            ),
                        }
                    )
                    continue
            else:
                raise ValueError(f"unknown model_frame_mode={args.model_frame_mode!r}")
            frames = _load_rgb_window(stream, model_frame_ids, resize_hw=d4rt_input_hw)
            frame_to_local: dict[int, int] = {}
            for idx, frame in enumerate(model_frame_ids):
                frame_to_local.setdefault(int(frame), idx)
            target_frames_local = np.asarray([frame_to_local[int(frame)] for frame in target_frame_ids], dtype=np.int64)
            src_uv = np.asarray([[_num(row.get("query_u_norm")), _num(row.get("query_v_norm"))] for row in decode_rows], dtype=np.float32)
            if float(args.source_uv_clamp_eps) > 0.0:
                eps = float(args.source_uv_clamp_eps)
                src_uv = np.clip(src_uv, eps, 1.0 - eps).astype(np.float32, copy=False)
            src_frame_local = np.asarray([frame_to_local[int(_num(row.get("frame_id")))] for row in decode_rows], dtype=np.int64)
            src_frame_global = np.asarray([int(_num(row.get("frame_id"))) for row in decode_rows], dtype=np.int64)
            src_xy_list = []
            for row in decode_rows:
                query_u_meta, query_v_meta = _query_px_on_grid(row, d4rt_output_width, d4rt_output_height)
                src_xy_list.append([int(round(query_u_meta)), int(round(query_v_meta))])
            src_xy = np.asarray(src_xy_list, dtype=np.int32)
            src_mask_id = np.asarray([int(_num(row.get("source_mask_id_optional"))) for row in decode_rows], dtype=np.int32)
            carrier_id = np.arange(len(decode_rows), dtype=np.int64)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            try:
                batch = adapter.infer_carriers(
                    video_rgb_uint8=frames,
                    src_uv_norm=src_uv,
                    src_frame_local=src_frame_local,
                    target_frames_local=target_frames_local,
                    carrier_id=carrier_id,
                    src_frame_global=src_frame_global,
                    src_xy=src_xy,
                    src_mask_id=src_mask_id,
                    query_chunk_size=int(args.query_chunk_size),
                )
                status = "ok"
                error = ""
            except Exception as exc:  # pragma: no cover - depends on D4RT/GPU
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                errors.append(
                    {
                        "decode_variant": decode_variant,
                        "query_variant": query_variant,
                        "scene_id": scene,
                        "window_id": window,
                        "query_count": len(decode_rows),
                        "error": error,
                    }
                )
                continue
            peak_mb = 0.0
            if torch is not None and torch.cuda.is_available():
                peak_mb = float(torch.cuda.max_memory_allocated() / (1024.0**2))
            group_dir = output_root / "carrier_batches" / decode_variant / scene
            group_dir.mkdir(parents=True, exist_ok=True)
            carrier_path = group_dir / f"{window}.npz"
            np.savez_compressed(
                carrier_path,
                frame_ids=np.asarray(target_frame_ids, dtype=np.int64),
                active_frame_ids=np.asarray(active_frame_ids, dtype=np.int64),
                model_frame_ids=np.asarray(model_frame_ids, dtype=np.int64),
                target_frames_local=target_frames_local,
                model_frame_mode=np.asarray([args.model_frame_mode]),
                query_id=np.asarray([row.get("query_id", "") for row in decode_rows]),
                carrier_id=np.asarray(batch.carrier_id),
                src_frame=np.asarray(batch.src_frame),
                src_frame_global=src_frame_global,
                src_uv=src_uv,
                src_xy=src_xy,
                src_mask_id=src_mask_id,
                uv_pred=np.asarray(batch.uv_pred, dtype=np.float32)[: len(target_frame_ids)],
                xyz_ref=np.asarray(batch.xyz_ref, dtype=np.float32)[: len(target_frame_ids)],
                xyz_local=np.asarray(batch.xyz_local, dtype=np.float32)[: len(target_frame_ids)] if batch.xyz_local is not None else np.zeros((len(target_frame_ids), len(decode_rows), 3), dtype=np.float32),
                visibility_prob=np.asarray(batch.visibility_prob, dtype=np.float32)[: len(target_frame_ids)],
                confidence_prob=np.asarray(batch.confidence_prob, dtype=np.float32)[: len(target_frame_ids)],
                valid=np.asarray(batch.valid, dtype=bool)[: len(target_frame_ids)],
            )
            uv = np.asarray(batch.uv_pred, dtype=np.float32)[: len(target_frame_ids)]
            xyz = np.asarray(batch.xyz_ref, dtype=np.float32)[: len(target_frame_ids)]
            visibility = np.asarray(batch.visibility_prob, dtype=np.float32)[: len(target_frame_ids)]
            confidence = np.asarray(batch.confidence_prob, dtype=np.float32)[: len(target_frame_ids)]
            valid = np.asarray(batch.valid, dtype=bool)[: len(target_frame_ids)]
            quality, occupancy_rows = _quality_for_batch(
                rows=decode_rows,
                frame_ids=target_frame_ids,
                model_frame_ids=model_frame_ids,
                uv=uv,
                xyz=xyz,
                valid=valid,
                visibility=visibility,
                confidence=confidence,
                stream=stream,
                scene=scene,
                window_id=window,
                mask_lookup=mask_lookup,
                occupancy_radius_px=int(args.occupancy_radius_px),
                min_visibility=float(args.min_visibility),
                min_confidence=float(args.min_confidence),
            )
            group_runtime = float(time.time() - group_started)
            quality_row = {
                "schema_version": "stream4d_v96_micro_track_quality_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "decode_variant": decode_variant,
                "query_variant": query_variant,
                "scene_id": scene,
                "split": "dev",
                "window_id": window,
                **quality,
                "model_frame_mode": args.model_frame_mode,
                "d4rt_input_width": int(d4rt_input_width),
                "d4rt_input_height": int(d4rt_input_height),
                "d4rt_output_width": int(d4rt_output_width),
                "d4rt_output_height": int(d4rt_output_height),
                "coordinate_grid": coordinate_grid,
                "active_frame_ids": ",".join(str(int(v)) for v in active_frame_ids),
                "target_frames_local": ",".join(str(int(v)) for v in target_frames_local.tolist()),
                "runtime_decode_sec": group_runtime,
                "GPU_memory_peak_MB": peak_mb,
                "carrier_batch_npz": _rel(carrier_path),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "device": args.device,
                "track_status": status,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            quality_rows.append(quality_row)
            for occ in occupancy_rows:
                occ.update({"decode_variant": decode_variant, "query_variant": query_variant, "scene_id": scene, "window_id": window})
            occupancy_rows_all.extend(occupancy_rows)
            group_rows.append(
                {
                    "schema_version": "stream4d_v96_phase2_group_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "decode_variant": decode_variant,
                    "query_variant": query_variant,
                    "scene_id": scene,
                    "window_id": window,
                    "query_count": len(decode_rows),
                    "active_query_count_before_model_filter": len(rows),
                    "target_frame_count": len(target_frame_ids),
                    "active_frame_count_before_model_filter": len(active_frame_ids),
                    "model_frame_count": len(model_frame_ids),
                    "active_frame_ids": ",".join(str(int(v)) for v in active_frame_ids),
                    "target_frame_ids": ",".join(str(int(v)) for v in target_frame_ids),
                    "model_frame_ids": ",".join(str(int(v)) for v in model_frame_ids),
                    "target_frames_local": ",".join(str(int(v)) for v in target_frames_local.tolist()),
                    "model_frame_mode": args.model_frame_mode,
                    "runtime_decode_sec": group_runtime,
                    "GPU_memory_peak_MB": peak_mb,
                    "carrier_batch_npz": _rel(carrier_path),
                    "status": status,
                    "error": error,
                    "source_uv_clamp_eps": float(args.source_uv_clamp_eps),
                    "d4rt_input_width": int(d4rt_input_width),
                    "d4rt_input_height": int(d4rt_input_height),
                    "d4rt_output_width": int(d4rt_output_width),
                    "d4rt_output_height": int(d4rt_output_height),
                    "coordinate_grid": coordinate_grid,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            for row in decode_rows:
                query_u_out, query_v_out = _query_px_on_grid(row, d4rt_output_width, d4rt_output_height)
                query_writer.writerow(
                    {
                        "schema_version": "stream4d_v96_micro_query_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "decode_variant": decode_variant,
                        "query_variant": query_variant,
                        "scene_id": scene,
                        "split": "dev",
                        "window_id": window,
                        "frame_id": int(_num(row.get("frame_id"))),
                        "query_id": row.get("query_id", ""),
                        "query_u": query_u_out if fixed_output_grid else row.get("query_u", ""),
                        "query_v": query_v_out if fixed_output_grid else row.get("query_v", ""),
                        "query_u_norm": row.get("query_u_norm", ""),
                        "query_v_norm": row.get("query_v_norm", ""),
                        "query_stratum": row.get("query_stratum", ""),
                        "query_priority": row.get("query_priority", ""),
                        "source_mask_id_optional": row.get("source_mask_id_optional", ""),
                        "semantic_gradient_score": row.get("semantic_gradient_score", ""),
                        "mask_conflict_score": row.get("mask_conflict_score", ""),
                        "occupancy_before": row.get("occupancy_before", ""),
                        "query_u_original_px": row.get("query_u", row.get("query_x", "")),
                        "query_v_original_px": row.get("query_v", row.get("query_y", "")),
                        "d4rt_input_width": int(d4rt_input_width),
                        "d4rt_input_height": int(d4rt_input_height),
                        "d4rt_output_width": int(d4rt_output_width),
                        "d4rt_output_height": int(d4rt_output_height),
                        "coordinate_grid": coordinate_grid,
                        "uses_gt_for_query_selection": False,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
            first_mask_path = mask_lookup.get((scene, window, int(target_frame_ids[0])))
            first_label = _load_label(first_mask_path) if first_mask_path is not None and first_mask_path.exists() else stream.load_mask(target_frame_ids[0])
            h = int(first_label.shape[0])
            w = int(first_label.shape[1])
            out_w = int(d4rt_output_width) if fixed_output_grid else w
            out_h = int(d4rt_output_height) if fixed_output_grid else h
            for local_idx, target_frame in enumerate(target_frame_ids):
                for qidx, row in enumerate(decode_rows):
                    query_u_out, query_v_out = _query_px_on_grid(row, d4rt_output_width, d4rt_output_height)
                    u = float(uv[local_idx, qidx, 0])
                    v = float(uv[local_idx, qidx, 1])
                    ok = bool(valid[local_idx, qidx] and 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0)
                    track_writer.writerow(
                        {
                            "schema_version": "stream4d_v96_micro_track_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "decode_variant": decode_variant,
                            "query_variant": query_variant,
                            "scene_id": scene,
                            "window_id": window,
                            "query_id": row.get("query_id", ""),
                            "source_frame_id": int(_num(row.get("frame_id"))),
                            "target_frame_id": int(target_frame),
                            "u_src": query_u_out if fixed_output_grid else row.get("query_u", ""),
                            "v_src": query_v_out if fixed_output_grid else row.get("query_v", ""),
                            "u_tgt": float(u * (out_w - 1)),
                            "v_tgt": float(v * (out_h - 1)),
                            "x_3d": float(xyz[local_idx, qidx, 0]),
                            "y_3d": float(xyz[local_idx, qidx, 1]),
                            "z_3d": float(xyz[local_idx, qidx, 2]),
                            "visibility": float(visibility[local_idx, qidx]),
                            "confidence": float(confidence[local_idx, qidx]),
                            "uv_in01": ok,
                            "query_stratum": row.get("query_stratum", ""),
                            "u_src_original_px": row.get("query_u", row.get("query_x", "")),
                            "v_src_original_px": row.get("query_v", row.get("query_y", "")),
                            "d4rt_input_width": int(d4rt_input_width),
                            "d4rt_input_height": int(d4rt_input_height),
                            "d4rt_output_width": int(d4rt_output_width),
                            "d4rt_output_height": int(d4rt_output_height),
                            "coordinate_grid": coordinate_grid,
                            "track_status": "valid_in01" if ok else "invalid_or_oob",
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                    )
            if args.progress_every_groups > 0 and (group_idx + 1) % int(args.progress_every_groups) == 0:
                print({"phase": PHASE_ID, "processed_groups": group_idx + 1, "total_groups": len(selected)}, flush=True)
    finally:
        query_handle.close()
        track_handle.close()

    _write_csv(output_root / "micro_track_quality_rows.csv", quality_rows)
    _write_csv(output_root / "micro_occupancy_rows.csv", occupancy_rows_all)
    _write_csv(output_root / "decode_group_rows.csv", group_rows)
    _write_csv(output_root / "decode_error_rows.csv", errors)

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in quality_rows:
        by_variant[row["decode_variant"]].append(row)
    variant_summaries: list[dict[str, Any]] = []
    for variant, rows in sorted(by_variant.items()):
        total_q = sum(int(row["query_count"]) for row in rows)
        weights = np.asarray([int(row["query_count"]) for row in rows], dtype=np.float64)
        def wmean(key: str) -> float:
            vals = np.asarray([_num(row.get(key)) for row in rows], dtype=np.float64)
            return float(np.average(vals, weights=weights)) if weights.sum() > 0 else 0.0
        variant_summaries.append(
            {
                "decode_variant": variant,
                "query_variant": rows[0].get("query_variant", ""),
                "group_count": len(rows),
                "query_count": int(total_q),
                "valid_track_ratio": wmean("valid_track_ratio"),
                "uv_in01_rate": wmean("uv_in01_rate"),
                "visibility_mean": wmean("visibility_mean"),
                "confidence_mean": wmean("confidence_mean"),
                "track_length_visible_mean": wmean("track_length_visible_mean"),
                "track_length_visible_p10_mean": wmean("track_length_visible_p10"),
                "projection_jitter_mean": wmean("projection_jitter_mean"),
                "projection_jitter_p90_mean": wmean("projection_jitter_p90"),
                "mask_membership_flip_rate": wmean("mask_membership_flip_rate"),
                "source_support_area_ratio": wmean("source_support_area_ratio"),
                "boundary_band_support_ratio": wmean("boundary_band_support_ratio"),
                "competing_edge_support_ratio": wmean("competing_edge_support_ratio"),
                "semantic_gradient_support_ratio": wmean("semantic_gradient_support_ratio"),
                "runtime_decode_sec": float(sum(_num(row.get("runtime_decode_sec")) for row in rows)),
                "GPU_memory_peak_MB": float(max((_num(row.get("GPU_memory_peak_MB")) for row in rows), default=0.0)),
            }
        )
    summary_by_variant = {row["decode_variant"]: row for row in variant_summaries}
    d1 = summary_by_variant.get("D1_uniform1024", {})
    d3 = summary_by_variant.get("D3_adaptive1024", {})
    gate_rows = [
        {
            "schema_version": "stream4d_v96_phase2_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "D3_boundary_support_ge_uniform1024_plus_0p05_or_abs_0p08",
            "pass": bool(
                d3
                and (
                    _num(d3.get("boundary_band_support_ratio")) >= _num(d1.get("boundary_band_support_ratio")) + 0.05
                    or _num(d3.get("boundary_band_support_ratio")) >= 0.08
                )
            ),
            "observed": d3.get("boundary_band_support_ratio", ""),
            "uniform1024": d1.get("boundary_band_support_ratio", ""),
            "required": "D3 >= D1 + 0.05 or D3 >= 0.08",
        },
        {
            "schema_version": "stream4d_v96_phase2_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "D3_source_support_ge_uniform1024_plus_0p05_or_abs_0p12",
            "pass": bool(
                d3
                and (
                    _num(d3.get("source_support_area_ratio")) >= _num(d1.get("source_support_area_ratio")) + 0.05
                    or _num(d3.get("source_support_area_ratio")) >= 0.12
                )
            ),
            "observed": d3.get("source_support_area_ratio", ""),
            "uniform1024": d1.get("source_support_area_ratio", ""),
            "required": "D3 >= D1 + 0.05 or D3 >= 0.12",
        },
        {
            "schema_version": "stream4d_v96_phase2_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "D3_uv_in01_ge_0p90",
            "pass": bool(d3 and _num(d3.get("uv_in01_rate")) >= 0.90),
            "observed": d3.get("uv_in01_rate", ""),
            "required": 0.90,
        },
        {
            "schema_version": "stream4d_v96_phase2_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "D3_valid_track_ratio_ge_0p50",
            "pass": bool(d3 and _num(d3.get("valid_track_ratio")) >= 0.50),
            "observed": d3.get("valid_track_ratio", ""),
            "required": 0.50,
        },
        {
            "schema_version": "stream4d_v96_phase2_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "runtime_recorded_no_oom",
            "pass": len(errors) == 0 and all(_num(row.get("runtime_decode_sec")) > 0 for row in variant_summaries),
            "observed": f"errors={len(errors)} variants={len(variant_summaries)}",
            "required": "errors=0 and runtime>0",
        },
    ]
    for row in gate_rows:
        row["uses_gt_for_prediction"] = False
        row["uses_future"] = False
    phase2_pass = all(bool(row.get("pass")) for row in gate_rows)
    summary = {
        "schema": "stream4d_v96_phase2_d4rt_micro_tracks_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "PASS_V96_PHASE2_D4RT_MICRO_TRACKS" if phase2_pass else "NO_GO_V96_PHASE2_D4RT_MICRO_TRACKS",
        "query_root": _rel(query_root),
        "source_rows": _rel(_project(args.source_rows)),
        "output_root": _rel(output_root),
        "selected_group_count": int(len(selected)),
        "decoded_group_count": int(len(group_rows)),
        "error_count": int(len(errors)),
        "variant_summaries": variant_summaries,
        "gate_rows": gate_rows,
        "micro_query_rows": _rel(micro_query_path),
        "micro_track_rows": _rel(micro_track_path),
        "micro_track_quality_rows": _rel(output_root / "micro_track_quality_rows.csv"),
        "micro_occupancy_rows": _rel(output_root / "micro_occupancy_rows.csv"),
        "decode_group_rows": _rel(output_root / "decode_group_rows.csv"),
        "carrier_batches_dir": _rel(output_root / "carrier_batches"),
        "runtime_total_sec": float(time.time() - started),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": args.device,
        "query_chunk_size": int(args.query_chunk_size),
        "source_uv_clamp_eps": float(args.source_uv_clamp_eps),
        "d4rt_input_width": int(d4rt_input_width),
        "d4rt_input_height": int(d4rt_input_height),
        "d4rt_output_width": int(d4rt_output_width),
        "d4rt_output_height": int(d4rt_output_height),
        "coordinate_grid": coordinate_grid,
        "coordinate_grid_note": (
            "D4RT model queries use normalized uv; fixed grid parameters control external RGB resize "
            "and u_src/v_src/u_tgt/v_tgt pixel coordinates. Mask quality still samples original masks by normalized uv."
        ),
        "model_frame_mode": args.model_frame_mode,
        "contiguous_frame_count": int(args.contiguous_frame_count),
        "frame_id_min": int(args.frame_id_min),
        "frame_id_max": int(args.frame_id_max),
        "max_windows": int(args.max_windows),
        "max_queries_per_frame": int(args.max_queries_per_frame),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_csv(output_root / "variant_summary_rows.csv", variant_summaries)
    _write_csv(output_root / "phase2_gate_rows.csv", gate_rows)
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "decoded_group_count": len(group_rows), "runtime_total_sec": summary["runtime_total_sec"]}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode v96 D4RT micro-tracks from query plans.")
    parser.add_argument("--query-root", default=str(DEFAULT_QUERY_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--variant-map", default=",".join(f"{d}={q}" for d, q in DEFAULT_VARIANT_MAP.items()))
    parser.add_argument("--decode-variants", default="")
    parser.add_argument("--scenes", default="")
    parser.add_argument("--window-ids", default="")
    parser.add_argument("--frame-id-min", type=int, default=-1)
    parser.add_argument("--frame-id-max", type=int, default=-1)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--max-queries-per-frame", type=int, default=0)
    parser.add_argument("--scannet-root", default=str(DEFAULT_SCANNET))
    parser.add_argument("--d4rt-root", default=str(DEFAULT_D4RT_ROOT))
    parser.add_argument("--d4rt-config", default=str(DEFAULT_D4RT_CONFIG))
    parser.add_argument("--d4rt-ckpt", default=str(DEFAULT_D4RT_CKPT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--source-uv-clamp-eps", type=float, default=0.0)
    parser.add_argument("--d4rt-input-width", type=int, default=0)
    parser.add_argument("--d4rt-input-height", type=int, default=0)
    parser.add_argument("--d4rt-output-width", type=int, default=0)
    parser.add_argument("--d4rt-output-height", type=int, default=0)
    parser.add_argument("--model-frame-mode", choices=("active_sparse", "contiguous32_from_first"), default="active_sparse")
    parser.add_argument("--contiguous-frame-count", type=int, default=32)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--occupancy-radius-px", type=int, default=3)
    parser.add_argument("--progress-every-groups", type=int, default=1)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
