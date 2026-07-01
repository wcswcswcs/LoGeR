#!/usr/bin/env python3
"""Build overlap chunk query/source views for v97 D4RT method runs.

The generated query plan is method-side only: it groups existing GT-free query
rows into overlapping D4RT chunks so Phase2 can decode windows that have shared
frames.  It does not apply final GT Sim3 or use ScanNet GT.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v97_phase1_overlap_chunk_query_views"
RUN_ID = "v97_phase1_overlap_chunk_query_views"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_OUT = ROOT / "outputs/audit/v97_phase1_query_planner_overlap48_Q3_source_preserve2048"


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "docs", "Open-d4rt"}:
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _input_roots(raw: str) -> list[Path]:
    roots = [_project(part.strip()) for part in raw.split(",") if part.strip()]
    if not roots:
        raise ValueError("--input-roots must contain at least one root")
    return roots


def _make_windows(frames: list[int], chunk_size: int, overlap_frames: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if overlap_frames < 0 or overlap_frames >= chunk_size:
        raise ValueError("--overlap-frames must satisfy 0 <= overlap < chunk_size")
    if not frames:
        return []
    step = chunk_size - overlap_frames
    windows: list[list[int]] = []
    start = 0
    while start < len(frames):
        window = frames[start : min(start + chunk_size, len(frames))]
        if window:
            windows.append([int(v) for v in window])
        if start + chunk_size >= len(frames):
            break
        start += step
    return windows


def _read_query_rows(
    roots: list[Path],
    *,
    variant_id: str,
    scene_id: str,
) -> tuple[list[str], dict[int, list[dict[str, str]]], dict[str, Any]]:
    fieldnames: list[str] = []
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    seen_query_ids: set[str] = set()
    duplicate_query_ids = 0
    root_rows: list[dict[str, Any]] = []
    for root in roots:
        path = root / "query_plan_rows.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        row_count = 0
        kept_count = 0
        dropped_scene_count = 0
        dropped_variant_count = 0
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not fieldnames:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                row_count += 1
                if scene_id and row.get("scene_id", "") != scene_id:
                    dropped_scene_count += 1
                    continue
                if variant_id and row.get("variant_id", "") != variant_id:
                    dropped_variant_count += 1
                    continue
                qid = row.get("query_id", "")
                if qid in seen_query_ids:
                    duplicate_query_ids += 1
                    continue
                seen_query_ids.add(qid)
                by_frame[int(float(row["frame_id"]))].append(row)
                kept_count += 1
        root_rows.append(
            {
                "root": _rel(root),
                "row_count": row_count,
                "kept_count": kept_count,
                "dropped_scene_count": dropped_scene_count,
                "dropped_variant_count": dropped_variant_count,
            }
        )
    if not fieldnames:
        raise RuntimeError("input query roots contained no fields")
    return fieldnames, by_frame, {
        "input_root_rows": root_rows,
        "duplicate_query_ids_dropped": duplicate_query_ids,
        "kept_query_id_count": len(seen_query_ids),
    }


def _read_source_by_frame(source_rows: Path) -> tuple[list[str], dict[tuple[str, int], list[dict[str, str]]]]:
    by_frame: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with source_rows.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            by_frame[(row.get("scene_id", ""), int(float(row.get("frame_id", 0))))].append(row)
    return fieldnames, by_frame


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    roots = _input_roots(args.input_roots)
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    query_fields, by_frame, query_diag = _read_query_rows(roots, variant_id=args.variant_id, scene_id=args.scene_id)
    frames = sorted(by_frame)
    windows = _make_windows(frames, int(args.chunk_size), int(args.overlap_frames))
    query_out = output_root / "query_plan_rows.csv"
    source_out = output_root / "overlap_source_container_rows.csv"
    source_fields, source_by_frame = _read_source_by_frame(_project(args.source_rows))

    extra_query_fields = ["overlap_source_window_id", "overlap_chunk_index", "overlap_chunk_frame_start", "overlap_chunk_frame_end"]
    out_query_fields = list(query_fields)
    for field in extra_query_fields:
        if field not in out_query_fields:
            out_query_fields.append(field)

    out_source_fields = list(source_fields)
    for field in ["overlap_source_window_id", "overlap_chunk_index", "overlap_chunk_frame_start", "overlap_chunk_frame_end"]:
        if field not in out_source_fields:
            out_source_fields.append(field)

    query_count = 0
    source_count = 0
    frame_duplication_count = 0
    window_rows: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    per_frame_query_counts: Counter[int] = Counter()
    with query_out.open("w", newline="", encoding="utf-8") as query_handle, source_out.open("w", newline="", encoding="utf-8") as source_handle:
        query_writer = csv.DictWriter(query_handle, fieldnames=out_query_fields)
        source_writer = csv.DictWriter(source_handle, fieldnames=out_source_fields)
        query_writer.writeheader()
        source_writer.writeheader()
        frame_seen_count: Counter[int] = Counter()
        for chunk_index, window_frames in enumerate(windows):
            window_id = f"ov{int(args.chunk_size):02d}_{chunk_index:04d}"
            frame_set = set(window_frames)
            chunk_query_count = 0
            chunk_source_count = 0
            for frame_id in window_frames:
                frame_seen_count[int(frame_id)] += 1
                for src in source_by_frame.get((args.scene_id, int(frame_id)), []):
                    out_src = dict(src)
                    out_src["physical_source_key"] = (
                        f"{out_src.get('scene_id', '')}|{window_id}|{out_src.get('frame_id', '')}|{out_src.get('source_mask_id', '')}"
                    )
                    out_src["window_id"] = window_id
                    out_src["overlap_source_window_id"] = src.get("window_id", "")
                    out_src["overlap_chunk_index"] = chunk_index
                    out_src["overlap_chunk_frame_start"] = window_frames[0]
                    out_src["overlap_chunk_frame_end"] = window_frames[-1]
                    source_writer.writerow({field: out_src.get(field, "") for field in out_source_fields})
                    source_count += 1
                    chunk_source_count += 1
            for frame_id in window_frames:
                for row in by_frame[int(frame_id)]:
                    out = dict(row)
                    original_window = row.get("window_id", "")
                    original_query_id = row.get("query_id", "")
                    out["window_id"] = window_id
                    out["chunk_id"] = window_id
                    out["query_id"] = f"{args.variant_id}:overlap{chunk_index:04d}:{original_query_id}"
                    out["overlap_source_window_id"] = original_window
                    out["overlap_chunk_index"] = chunk_index
                    out["overlap_chunk_frame_start"] = window_frames[0]
                    out["overlap_chunk_frame_end"] = window_frames[-1]
                    query_writer.writerow({field: out.get(field, "") for field in out_query_fields})
                    query_count += 1
                    chunk_query_count += 1
                    stratum_counts[str(out.get("query_stratum", ""))] += 1
                    per_frame_query_counts[int(frame_id)] += 1
            frame_duplication_count += sum(1 for frame_id in frame_set if frame_seen_count[int(frame_id)] > 1)
            window_rows.append(
                {
                    "schema_version": "stream4d_v97_overlap_chunk_window_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "window_id": window_id,
                    "chunk_index": chunk_index,
                    "frame_start": int(window_frames[0]),
                    "frame_end": int(window_frames[-1]),
                    "frame_count": int(len(window_frames)),
                    "query_row_count": int(chunk_query_count),
                    "source_row_count": int(chunk_source_count),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    window_rows_path = output_root / "overlap_window_rows.csv"
    with window_rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(window_rows[0].keys()) if window_rows else [])
        writer.writeheader()
        writer.writerows(window_rows)

    summary = {
        "schema": "stream4d_v97_phase1_overlap_chunk_query_views_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input_roots": [_rel(root) for root in roots],
        "variant_id": args.variant_id,
        "scene_id": args.scene_id,
        "chunk_size": int(args.chunk_size),
        "overlap_frames": int(args.overlap_frames),
        "chunk_count": int(len(windows)),
        "unique_input_frame_count": int(len(frames)),
        "frame_min": int(frames[0]) if frames else None,
        "frame_max": int(frames[-1]) if frames else None,
        "query_row_count": int(query_count),
        "source_row_count": int(source_count),
        "frame_duplication_count": int(frame_duplication_count),
        "per_frame_query_count_min": int(min(per_frame_query_counts.values())) if per_frame_query_counts else 0,
        "per_frame_query_count_max": int(max(per_frame_query_counts.values())) if per_frame_query_counts else 0,
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "query_plan_rows": _rel(query_out),
        "overlap_source_rows": _rel(source_out),
        "overlap_window_rows": _rel(window_rows_path),
        "method_geometry_policy": "D4RT overlap chunks for method self-stitch; no final GT Sim3.",
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_total_sec": float(time.time() - started),
        **query_diag,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"output_root": _rel(output_root), "chunk_count": len(windows), "query_row_count": query_count, "source_row_count": source_count}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--variant-id", default="Q3_source_preserve2048")
    parser.add_argument("--scene-id", default="scene0011_00")
    parser.add_argument("--chunk-size", type=int, default=48)
    parser.add_argument("--overlap-frames", type=int, default=3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
