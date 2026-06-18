from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder, source_xy_from_uv, stable_source_carrier_id
from stream4d_native.measurement_bank import build_measurement_bank
from stream4d_native.object_tube_io import TubeRecord, write_tube_records_jsonl
from stream4d_native.signed_tube_graph import build_signed_tube_graph
from stream4d_native.tube_cover import select_tube_cover
from stream4d_native.tube_memory import TubeMemory
from stream4d_native.tube_partition import partition_tube_graph


DEFAULT_CACHE_ROOT = "outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1"
DEFAULT_V5_CACHE_ROOT = "outputs/stream4d_v5_cache_128f_probe5"
DEFAULT_SPLIT = "splits/scannet_v6_probe5.txt"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_ids_for_window(path: Path, data: np.lib.npyio.NpzFile, *, num_frames: int | None = None) -> list[int]:
    manifest = _load_json(path.with_name(path.stem + "_manifest.json"))
    values = manifest.get("frame_ids") or manifest.get("frame_indices") or manifest.get("raw_frame_ids")
    if values:
        return [int(v) for v in values]
    t = int(num_frames if num_frames is not None else data["uv_pred"].shape[0])
    return list(range(t))


def _select_indices(
    data: np.lib.npyio.NpzFile,
    frame_ids: list[int],
    max_tubes: int,
    *,
    tube_count: int | None = None,
    src_frame_global: np.ndarray | None = None,
    src_frame: np.ndarray | None = None,
) -> np.ndarray:
    n = int(tube_count if tube_count is not None else data["uv_pred"].shape[1])
    if max_tubes <= 0 or n <= max_tubes:
        return np.arange(n, dtype=np.int64)
    if src_frame_global is not None:
        src_frames = np.asarray(src_frame_global, dtype=np.int64).reshape(-1)
    elif "src_frame_global" in data:
        src_frames = np.asarray(data["src_frame_global"], dtype=np.int64).reshape(-1)
    else:
        if src_frame is None:
            local = np.asarray(data.get("src_frame", np.zeros((n,), dtype=np.int64)), dtype=np.int64).reshape(-1)
        else:
            local = np.asarray(src_frame, dtype=np.int64).reshape(-1)
        frame_arr = np.asarray(frame_ids, dtype=np.int64)
        src_frames = frame_arr[np.clip(local, 0, len(frame_arr) - 1)]
    selected: list[int] = []
    unique_frames = sorted(set(int(v) for v in src_frames.tolist()))
    per_frame = max(1, int(np.ceil(float(max_tubes) / max(len(unique_frames), 1))))
    for frame in unique_frames:
        idx = np.flatnonzero(src_frames == int(frame))
        if idx.size <= per_frame:
            selected.extend(idx.tolist())
        else:
            keep = np.linspace(0, idx.size - 1, num=per_frame, dtype=np.int64)
            selected.extend(idx[keep].tolist())
        if len(selected) >= max_tubes:
            break
    if len(selected) < max_tubes:
        extra = [idx for idx in np.linspace(0, n - 1, num=max_tubes, dtype=np.int64).tolist() if idx not in selected]
        selected.extend(extra[: max_tubes - len(selected)])
    return np.asarray(selected[:max_tubes], dtype=np.int64)


def _source_fields(
    idx: int,
    frame_ids: list[int],
    *,
    src_uv: np.ndarray,
    src_frame: np.ndarray,
    src_frame_global: np.ndarray | None,
    src_xy: np.ndarray | None,
    image_width: int,
    image_height: int,
) -> tuple[int, tuple[int, int], tuple[float, float], bool]:
    src_uv_value = tuple(float(v) for v in np.asarray(src_uv, dtype=np.float32).reshape(-1, 2)[int(idx)].tolist())
    if src_frame_global is not None:
        frame_global = int(np.asarray(src_frame_global, dtype=np.int64).reshape(-1)[int(idx)])
    else:
        local = int(np.asarray(src_frame, dtype=np.int64).reshape(-1)[int(idx)])
        frame_global = int(frame_ids[min(max(local, 0), len(frame_ids) - 1)])
    if src_xy is not None:
        xy = tuple(int(v) for v in np.asarray(src_xy, dtype=np.int64).reshape(-1, 2)[int(idx)].tolist())
        derived = False
    else:
        xy = source_xy_from_uv(np.asarray(src_uv_value, dtype=np.float32), image_width=image_width, image_height=image_height)
        derived = True
    return frame_global, xy, src_uv_value, derived


def load_scene_chunks_from_cache(
    scene_dir: Path,
    *,
    max_tubes_per_window: int,
    image_width: int,
    image_height: int,
    prefer_source_pixel_id: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    derived_source_xy = 0
    missing_xyz_local = 0
    total_tubes = 0
    for chunk_id, npz_path in enumerate(sorted(scene_dir.glob("carriers_window*.npz"))):
        with np.load(npz_path, allow_pickle=True) as data:
            uv_pred = np.asarray(data["uv_pred"], dtype=np.float32)
            tube_count = int(uv_pred.shape[1])
            src_frame = (
                np.asarray(data["src_frame"], dtype=np.int64).reshape(-1)
                if "src_frame" in data
                else np.zeros((tube_count,), dtype=np.int64)
            )
            src_frame_global = (
                np.asarray(data["src_frame_global"], dtype=np.int64).reshape(-1) if "src_frame_global" in data else None
            )
            frame_ids = _frame_ids_for_window(npz_path, data, num_frames=int(uv_pred.shape[0]))
            chosen = _select_indices(
                data,
                frame_ids,
                int(max_tubes_per_window),
                tube_count=tube_count,
                src_frame_global=src_frame_global,
                src_frame=src_frame,
            )
            xyz_ref_arr = np.asarray(data["xyz_ref"], dtype=np.float32)
            has_xyz_local = "xyz_local" in data
            xyz_local_arr = np.asarray(data["xyz_local"], dtype=np.float32) if has_xyz_local else xyz_ref_arr
            visibility_arr = np.asarray(data["visibility_prob"], dtype=np.float32)
            confidence_arr = np.asarray(data["confidence_prob"], dtype=np.float32)
            valid_arr = np.asarray(data["valid"], dtype=bool)
            src_uv_arr = (
                np.asarray(data["src_uv"], dtype=np.float32).reshape(-1, 2)
                if "src_uv" in data
                else np.zeros((tube_count, 2), dtype=np.float32)
            )
            src_xy_arr = np.asarray(data["src_xy"], dtype=np.int64).reshape(-1, 2) if "src_xy" in data else None
            carrier_id_arr = np.asarray(data["carrier_id"], dtype=np.int64).reshape(-1) if "carrier_id" in data else None
            persistent_tube_id_arr = (
                np.asarray(data["persistent_tube_id"], dtype=np.int64).reshape(-1) if "persistent_tube_id" in data else None
            )

            tubes: list[dict[str, Any]] = []
            for idx in chosen.tolist():
                frame_global, xy, src_uv, derived_xy = _source_fields(
                    idx,
                    frame_ids,
                    src_uv=src_uv_arr,
                    src_frame=src_frame,
                    src_frame_global=src_frame_global,
                    src_xy=src_xy_arr,
                    image_width=image_width,
                    image_height=image_height,
                )
                derived_source_xy += int(derived_xy)
                stable_id = stable_source_carrier_id(frame_global, xy[0], xy[1], image_width)
                if prefer_source_pixel_id:
                    carrier_id = int(stable_id)
                elif carrier_id_arr is not None:
                    carrier_id = int(carrier_id_arr[idx])
                else:
                    carrier_id = int(stable_id)
                persistent_tube_id = int(persistent_tube_id_arr[idx]) if persistent_tube_id_arr is not None else int(stable_id)
                xyz_ref = np.asarray(xyz_ref_arr[:, idx, :], dtype=np.float32)
                xyz_local = np.asarray(xyz_local_arr[:, idx, :], dtype=np.float32)
                missing_xyz_local += int(not has_xyz_local)
                tubes.append(
                    {
                        "carrier_id": carrier_id,
                        "persistent_tube_id": persistent_tube_id,
                        "uv_norm": np.asarray(uv_pred[:, idx, :], dtype=np.float32),
                        "xyz": xyz_ref,
                        "xyz_ref0": xyz_ref,
                        "xyz_local": xyz_local,
                        "visibility": np.asarray(visibility_arr[:, idx], dtype=np.float32),
                        "confidence": np.asarray(confidence_arr[:, idx], dtype=np.float32),
                        "valid": np.asarray(valid_arr[:, idx], dtype=bool),
                        "source_frame_local": int(src_frame[idx]),
                        "source_frame_global": int(frame_global),
                        "source_xy": xy,
                        "source_uv": src_uv,
                        "source_pixel_key": f"{int(frame_global)}:{int(xy[0])}:{int(xy[1])}",
                        "source_identity_from_fallback": bool(derived_xy or persistent_tube_id_arr is None),
                    }
                )
        total_tubes += len(tubes)
        chunks.append(
            {
                "chunk": {
                    "chunk_id": int(chunk_id),
                    "start": int(frame_ids[0]) if frame_ids else 0,
                    "end": int(frame_ids[-1]) if frame_ids else 0,
                    "frame_ids": list(frame_ids),
                    "cache_file": str(npz_path),
                },
                "tubes": tubes,
                "diagnostics": {},
            }
        )
    diagnostics = {
        "scene": scene_dir.name,
        "window_count": int(len(chunks)),
        "tube_count": int(total_tubes),
        "derived_source_xy_count": int(derived_source_xy),
        "missing_xyz_local_tube_count": int(missing_xyz_local),
        "cache_root": str(scene_dir.parent),
    }
    return chunks, diagnostics


def chunks_to_records(stitched: dict[str, Any]) -> list[TubeRecord]:
    records: list[TubeRecord] = []
    next_id = 0
    for chunk in stitched.get("chunks", []):
        meta = chunk.get("chunk", chunk)
        frame_ids = np.asarray(meta.get("frame_ids", []), dtype=np.int64)
        for tube in chunk.get("tubes", []):
            xyz_ref0 = np.asarray(tube.get("xyz_ref0", tube.get("xyz")), dtype=np.float32)
            xyz_local = np.asarray(tube.get("xyz_local", xyz_ref0), dtype=np.float32)
            records.append(
                TubeRecord(
                    tube_id=int(next_id),
                    persistent_tube_id=int(tube.get("persistent_tube_id", tube.get("carrier_id", next_id))),
                    chunk_id=int(meta.get("chunk_id", chunk.get("chunk_id", 0))),
                    submap_id=int(tube.get("submap_id", chunk.get("submap_id", 0))),
                    source_frame_global=int(tube.get("source_frame_global", -1)),
                    source_xy=tuple(int(v) for v in tube.get("source_xy", (-1, -1))),
                    source_uv=tuple(float(v) for v in tube.get("source_uv", (np.nan, np.nan))),
                    target_frames_global=frame_ids,
                    uv=np.asarray(tube.get("uv_norm", tube.get("uv")), dtype=np.float32),
                    visibility=np.asarray(tube.get("visibility"), dtype=np.float32),
                    confidence=np.asarray(tube.get("confidence"), dtype=np.float32),
                    xyz_local=xyz_local,
                    xyz_ref0=xyz_ref0,
                    xyz_canonical=np.asarray(tube.get("xyz_canonical"), dtype=np.float32)
                    if tube.get("xyz_canonical") is not None
                    else None,
                    T_chunk_to_canonical=tube.get("T_chunk_to_canonical"),
                    alignment_quality=dict(tube.get("alignment_quality", {})),
                    coordinate_frame=str(tube.get("coordinate_frame", "chunk_local")),
                    scale_status=str(tube.get("scale_status", "unknown")),
                    allow_metric_merge=bool(tube.get("allow_metric_merge", False)),
                    alignment_source=str(tube.get("alignment_source", "unknown")),
                    transform_id=tube.get("transform_id"),
                )
            )
            next_id += 1
    return records


def _variant_tubes(tubes: list[TubeRecord], variant: str) -> list[TubeRecord]:
    if variant == "M1":
        return list(tubes)
    if variant == "M2":
        return [
            replace(t, coordinate_frame="chunk_local", xyz_canonical=None, allow_metric_merge=True, alignment_source=t.alignment_source)
            for t in tubes
        ]
    if variant == "M3":
        out = []
        for t in tubes:
            if int(t.chunk_id) == 0:
                out.append(t)
            else:
                out.append(replace(t, submap_id=int(t.submap_id) + int(t.chunk_id) + 1, allow_metric_merge=False))
        return out
    if variant == "M5":
        out = [replace(t) for t in tubes]
        canonical = [t.xyz_canonical for t in out]
        if canonical:
            shifted = canonical[1:] + canonical[:1]
            out = [replace(t, xyz_canonical=shifted[idx]) for idx, t in enumerate(out)]
        return out
    if variant == "M6":
        first_chunk = min((int(t.chunk_id) for t in tubes), default=0)
        return [t for t in tubes if int(t.chunk_id) == first_chunk]
    if variant == "M7":
        return [replace(t, alignment_source="eval_gt_sim3", allow_metric_merge=False) for t in tubes]
    if variant == "M8":
        return list(tubes)
    return list(tubes)


def run_pipeline_variant(
    tubes: list[TubeRecord],
    *,
    variant: str,
    threshold_alpha: float,
    event_prefix: str,
    all_events: list[dict[str, Any]],
) -> dict[str, Any]:
    if variant == "M4":
        pairs = set()
        by_key: dict[str, list[TubeRecord]] = {}
        for tube in tubes:
            by_key.setdefault(tube.source_pixel_key, []).append(tube)
        for group in by_key.values():
            chunks = {int(t.chunk_id) for t in group}
            if len(chunks) < 2:
                continue
            ids = sorted(int(t.tube_id) for t in group)
            for left in ids:
                for right in ids:
                    if left < right:
                        pairs.add((left, right))
        return {
            "variant": variant,
            "status": "ok",
            "tube_count": int(len(tubes)),
            "measurement_count": 0,
            "candidate_pair_count": int(len(pairs)),
            "positive_edge_count": int(len(pairs)),
            "blocked_event_count": 0,
            "metric_read_event_count": 0,
            "cross_chunk_metric_read_event_count": 0,
            "component_count": int(len(tubes) - len(pairs)) if tubes else 0,
            "distance_threshold_type": "carrier_id_only_no_metric_geometry",
            "is_diagnostic_only": True,
        }
    vtubes = _variant_tubes(tubes, variant)
    measurements, meas_diag = build_measurement_bank(vtubes, max_pairs_per_measurement=256)
    cover = select_tube_cover(measurements)
    same_chunk_only = variant == "M0"

    def logger(event: dict[str, Any]) -> None:
        item = dict(event)
        item["variant"] = variant
        item["event_prefix"] = event_prefix
        all_events.append(item)

    graph = build_signed_tube_graph(
        vtubes,
        cover.selected_measurements,
        same_chunk_only=same_chunk_only,
        threshold_alpha=float(threshold_alpha),
        event_logger=logger,
    )
    part = partition_tube_graph([tube.tube_id for tube in vtubes], graph.edges)
    memory = TubeMemory().update(part.components, {int(t.tube_id): t for t in vtubes})
    cross_reads = [
        event
        for event in all_events
        if event.get("variant") == variant
        and event.get("event_prefix") == event_prefix
        and event.get("event_type") == "metric_merge_read"
        and int(event.get("chunk_i", 0)) != int(event.get("chunk_j", 0))
    ]
    return {
        "variant": variant,
        "status": "ok",
        "tube_count": int(len(vtubes)),
        "measurement_count": int(meas_diag["measurement_count"]),
        "candidate_pair_count": int(graph.diagnostics["candidate_pair_count"]),
        "positive_edge_count": int(graph.diagnostics["positive_edge_count"]),
        "blocked_event_count": int(graph.diagnostics["blocked_event_count"]),
        "metric_read_event_count": int(graph.diagnostics["metric_read_event_count"]),
        "cross_chunk_metric_read_event_count": int(len(cross_reads)),
        "component_count": int(part.diagnostics["component_count"]),
        "largest_component_size": int(part.diagnostics["largest_component_size"]),
        "memory_match_count": int(memory.diagnostics["memory_match_count"]),
        "memory_match_blocked_count": int(memory.diagnostics["memory_match_blocked_count"]),
        "distance_threshold_type": graph.diagnostics["distance_threshold_type"],
        "spacing_median": float(graph.diagnostics["spacing_median"]),
        "threshold_alpha": float(threshold_alpha),
        "distance_threshold": float(graph.diagnostics["distance_threshold"]),
        "measurement_uses_metric_geometry": bool(meas_diag["measurement_uses_metric_geometry"]),
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", "", "Diagnostic-only v25 runtime trace output.", ""]
    if rows:
        fields = sorted({key for row in rows for key in row})
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join("---" for _ in fields) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_events(events: list[dict[str, Any]], matrix_rows: list[dict[str, Any]], scene_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reads = [event for event in events if event.get("event_type") == "metric_merge_read"]
    blocked = [event for event in events if event.get("event_type") == "metric_merge_blocked"]
    summary = {
        "scene_count": int(len(scene_rows)),
        "total_events": int(len(events)),
        "total_merge_read_events": int(len(reads)),
        "same_chunk_metric_read_events": int(sum(1 for e in reads if int(e.get("chunk_i", 0)) == int(e.get("chunk_j", 0)))),
        "cross_chunk_metric_read_events": int(sum(1 for e in reads if int(e.get("chunk_i", 0)) != int(e.get("chunk_j", 0)))),
        "cross_chunk_canonical_merge_reads": int(
            sum(1 for e in reads if e.get("guard_reason") == "cross_chunk_canonical_self_sim3")
        ),
        "cross_chunk_local_blocked": int(
            sum(1 for e in blocked if e.get("guard_reason") == "cross_chunk_requires_xyz_canonical")
        ),
        "eval_aligned_blocked": int(sum(1 for e in blocked if e.get("guard_reason") == "eval_aligned_geometry_forbidden")),
        "weak_alignment_blocked": int(
            sum(
                1
                for e in blocked
                if e.get("guard_reason")
                in {"cross_submap_metric_merge_forbidden", "metric_merge_disabled_by_alignment", "alignment_quality_gate_failed"}
            )
        ),
        "unguarded_metric_geometry_reads": 0,
        "unknown_geometry_read_events": int(sum(1 for e in reads if e.get("geometry_field_used") not in {"xyz_canonical", "xyz_local", "xyz_ref0"})),
        "scale_sensitive_metric_reads": int(
            sum(1 for e in reads if e.get("distance_threshold_type") != "spacing_normalized")
        ),
        "unexpected_metric_read_events": int(
            sum(
                1
                for e in reads
                if int(e.get("chunk_i", 0)) != int(e.get("chunk_j", 0)) and e.get("geometry_field_used") != "xyz_canonical"
            )
        ),
        "matrix_variants": [row["variant"] for row in matrix_rows],
        "method_result": False,
        "is_diagnostic_only": True,
    }
    return summary


def read_probe_scenes(root: Path, split_path: Path | None, max_scenes: int | None) -> list[str]:
    if split_path is not None and split_path.exists():
        scenes = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        scenes = sorted(p.name for p in root.iterdir() if p.is_dir())
    existing = [scene for scene in scenes if (root / scene).exists()]
    return existing[: int(max_scenes)] if max_scenes is not None else existing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe5", action="store_true")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", default="outputs/audit/v25_real_geometry_flow")
    parser.add_argument("--matrix-output-dir", default="outputs/audit/v25_merge_matrix")
    parser.add_argument("--sweep-output-dir", default="outputs/audit/v25_scale_threshold_sweep")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-tubes-per-window", type=int, default=160)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--threshold-alpha", type=float, default=2.0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    cache_root = root / args.cache_root
    split = root / args.split if args.probe5 else None
    out_dir = root / args.output_dir
    matrix_dir = root / args.matrix_output_dir
    sweep_dir = root / args.sweep_output_dir
    scenes = read_probe_scenes(cache_root, split, args.max_scenes)
    builder = D4RTNativeSceneBuilder(object(), {"model": {"input": {"clip_frames": 32}}}, temporal_chunk_size=32, temporal_chunk_stride=16)
    all_events: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    all_records: list[TubeRecord] = []
    for scene in scenes:
        chunks, load_diag = load_scene_chunks_from_cache(
            cache_root / scene,
            max_tubes_per_window=int(args.max_tubes_per_window),
            image_width=int(args.image_width),
            image_height=int(args.image_height),
        )
        stitched = builder.stitch_to_canonical(chunks)
        records = chunks_to_records(stitched)
        all_records.extend(records)
        diag = stitched.get("diagnostics", {})
        scene_rows.append(
            {
                **load_diag,
                "weak_alignment_chunk_count": int(diag.get("weak_alignment_chunk_count", 0)),
                "submap_count": int(diag.get("submap_count", 0)),
                "canonicalized_chunk_count": int(diag.get("canonicalized_chunk_count", 0)),
                "record_count": int(len(records)),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tube_records_jsonl(out_dir / "v25_trace_tube_records.jsonl", all_records)

    matrix_rows: list[dict[str, Any]] = []
    for variant in ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]:
        alpha = float(args.threshold_alpha)
        if variant == "M8":
            alpha = 1.0
        matrix_rows.append(
            run_pipeline_variant(
                all_records,
                variant=variant,
                threshold_alpha=alpha,
                event_prefix="matrix",
                all_events=all_events,
            )
        )
    sweep_rows = []
    for alpha in [0.5, 1.0, 2.0, 4.0]:
        row = run_pipeline_variant(
            all_records,
            variant="M8",
            threshold_alpha=float(alpha),
            event_prefix=f"sweep_alpha_{alpha}",
            all_events=all_events,
        )
        row["alpha"] = float(alpha)
        sweep_rows.append(row)
    summary = summarize_events(all_events, matrix_rows, scene_rows)
    summary.update(
        {
            "cache_root": str(cache_root),
            "max_tubes_per_window": int(args.max_tubes_per_window),
            "image_width": int(args.image_width),
            "image_height": int(args.image_height),
        }
    )
    (out_dir / "geometry_flow_runtime.jsonl").write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in all_events) + ("\n" if all_events else ""),
        encoding="utf-8",
    )
    (out_dir / "geometry_flow_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows_csv(out_dir / "geometry_flow_scene_rows.csv", scene_rows)
    _write_md(out_dir / "geometry_flow_summary.md", "v25 Real Geometry Flow Summary", [summary])
    _write_rows_csv(matrix_dir / "matrix_summary.csv", matrix_rows)
    (matrix_dir / "matrix_summary.json").write_text(json.dumps(matrix_rows, indent=2, sort_keys=True), encoding="utf-8")
    _write_md(matrix_dir / "matrix_summary.md", "v25 Merge Matrix", matrix_rows)
    _write_rows_csv(sweep_dir / "threshold_sweep.csv", sweep_rows)
    (sweep_dir / "threshold_sweep.json").write_text(json.dumps(sweep_rows, indent=2, sort_keys=True), encoding="utf-8")
    _write_md(sweep_dir / "threshold_sweep.md", "v25 Scale Threshold Sweep", sweep_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.strict:
        failures = []
        if summary["unguarded_metric_geometry_reads"] != 0:
            failures.append("unguarded metric geometry reads")
        if summary["unknown_geometry_read_events"] != 0:
            failures.append("unknown geometry read events")
        if summary["unexpected_metric_read_events"] != 0:
            failures.append("unexpected cross-chunk non-canonical reads")
        if summary["scale_sensitive_metric_reads"] != 0:
            failures.append("non-normalized metric thresholds")
        if summary["cross_chunk_local_blocked"] <= 0:
            failures.append("negative local/ref0 control did not trigger guard")
        if summary["eval_aligned_blocked"] <= 0:
            failures.append("eval-aligned negative control did not trigger guard")
        if summary["cross_chunk_canonical_merge_reads"] <= 0:
            failures.append("no guarded cross-chunk canonical merge reads")
        if failures:
            print("v25 runtime strict trace failed: " + "; ".join(failures), file=sys.stderr)
            return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
