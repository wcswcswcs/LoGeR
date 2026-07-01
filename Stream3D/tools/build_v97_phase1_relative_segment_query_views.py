#!/usr/bin/env python3
"""Create relative active-frame segment query-plan views for v97 Phase2 repair."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v97_phase1_relative_segment_query_views"
RUN_ID = "v97_phase1_relative_segment_query_views"
DEFAULT_QUERY_PLAN = ROOT / "outputs/audit/v97_phase1_query_planner/query_plan_rows.csv"
DEFAULT_OUT = ROOT / "outputs/audit"


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    query_plan = _project(args.query_plan)
    segment_size = int(args.segment_size)
    if segment_size <= 0:
        raise ValueError("--segment-size must be positive")
    frames_by_window: dict[tuple[str, str], set[int]] = defaultdict(set)
    with query_plan.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frames_by_window[(row["scene_id"], row["window_id"])].add(int(row["frame_id"]))
    ordinal_by_window: dict[tuple[str, str], dict[int, int]] = {
        key: {frame_id: idx for idx, frame_id in enumerate(sorted(frames))}
        for key, frames in frames_by_window.items()
    }
    max_ordinal = max((max(mapping.values()) for mapping in ordinal_by_window.values() if mapping), default=-1)
    segment_count = max_ordinal // segment_size + 1 if max_ordinal >= 0 else 0
    out_base = _project(args.output_base)
    roots = [out_base / f"v97_phase1_query_planner_relseg{idx:02d}_size{segment_size}" for idx in range(segment_count)]
    handles = []
    writers = []
    counts = [0 for _ in roots]
    variant_counts: list[dict[str, int]] = [defaultdict(int) for _ in roots]  # type: ignore[list-item]
    frame_sets: list[set[tuple[str, str, int]]] = [set() for _ in roots]
    fieldnames: list[str] | None = None
    try:
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)
            handle = (root / "query_plan_rows.csv").open("w", newline="", encoding="utf-8")
            handles.append(handle)
        with query_plan.open(newline="", encoding="utf-8") as source_handle:
            reader = csv.DictReader(source_handle)
            fieldnames = list(reader.fieldnames or [])
            for handle in handles:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writers.append(writer)
            for row in reader:
                key = (row["scene_id"], row["window_id"])
                ordinal = ordinal_by_window[key][int(row["frame_id"])]
                seg_idx = ordinal // segment_size
                writers[seg_idx].writerow(row)
                counts[seg_idx] += 1
                variant_counts[seg_idx][row["variant_id"]] += 1
                frame_sets[seg_idx].add((row["scene_id"], row["window_id"], int(row["frame_id"])))
    finally:
        for handle in handles:
            handle.close()
    segment_rows = []
    for idx, root in enumerate(roots):
        frames = sorted(frame_sets[idx])
        summary = {
            "schema": "stream4d_v97_phase1_relative_segment_query_view_summary_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "segment_index": idx,
            "segment_size": segment_size,
            "source_query_plan": _rel(query_plan),
            "query_plan_rows": _rel(root / "query_plan_rows.csv"),
            "query_row_count": counts[idx],
            "frame_key_count": len(frames),
            "variant_counts": dict(sorted(variant_counts[idx].items())),
            "ordinal_start_inclusive": idx * segment_size,
            "ordinal_end_exclusive": (idx + 1) * segment_size,
            "metric_scope": "relative_window_segment_query_view",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        _write_json(root / "summary.json", summary)
        segment_rows.append(summary)
    manifest = {
        "schema": "stream4d_v97_phase1_relative_segment_query_views_manifest_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_query_plan": _rel(query_plan),
        "segment_size": segment_size,
        "segment_count": segment_count,
        "segments": segment_rows,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    manifest_root = out_base / f"v97_phase1_query_planner_relseg_manifest_size{segment_size}"
    _write_json(manifest_root / "summary.json", manifest)
    print(json.dumps({"segment_count": segment_count, "manifest_root": _rel(manifest_root), "row_counts": counts}, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-plan", default=str(DEFAULT_QUERY_PLAN))
    parser.add_argument("--output-base", default=str(DEFAULT_OUT))
    parser.add_argument("--segment-size", type=int, default=7)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
