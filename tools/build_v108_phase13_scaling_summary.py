#!/usr/bin/env python3
"""Build a v108 Phase13 scaling summary from Phase12 online run summaries.

This report is diagnostic only. It aggregates runtime, memory, active-object,
transaction, and lifecycle fields so long-sequence behavior can be inspected
without using any metric as an automatic pass/fail gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = resolve_path(value)
        return path.parent.name, path
    name, path_text = value.split("=", 1)
    return name.strip(), resolve_path(path_text.strip())


def numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except Exception:
            continue
    return values


def summarize_run(name: str, summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    rolling_summary_path = resolve_path(summary["rolling_summary"])
    rolling = read_json(rolling_summary_path)
    records = rolling.get("records", [])
    visible_counts = numeric_values(records, "visible_id_count")
    object_counts = numeric_values(records, "object_id_count")
    frame_count = int(summary.get("frame_count") or rolling.get("frame_count") or len(records))
    runtime_sec = float(summary.get("runtime_sec") or 0.0)
    wrapper_wall = float(rolling.get("wrapper_total_with_v107_g3_wall_time_sec") or runtime_sec or 0.0)
    total_runtime = float(rolling.get("total_runtime_sec") or 0.0)
    frame_ids = rolling.get("frame_ids") or []
    oom_frame = ""
    status = str(summary.get("status", ""))
    if "OOM" in status.upper():
        oom_frame = str(frame_ids[-1] if frame_ids else "")
    video = summary.get("full_scene_video") or {}
    return {
        "run_name": name,
        "summary": rel(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "status": status,
        "scene_id": summary.get("scene_id", ""),
        "preset": summary.get("preset", ""),
        "frame_count": frame_count,
        "first_frame_id": frame_ids[0] if frame_ids else "",
        "last_frame_id": frame_ids[-1] if frame_ids else "",
        "full_online_run_executed": bool(summary.get("full_online_run_executed")),
        "runtime_sec": runtime_sec,
        "wrapper_wall_sec": wrapper_wall,
        "wall_sec_per_frame": wrapper_wall / frame_count if frame_count else "",
        "total_runtime_sec": total_runtime,
        "peak_cuda_memory_mb": rolling.get("peak_cuda_memory_mb", ""),
        "final_active_stream_object_count": rolling.get("final_active_stream_object_count", ""),
        "mean_visible_id_count": rolling.get("mean_visible_id_count", ""),
        "max_visible_id_count": max(visible_counts) if visible_counts else "",
        "max_object_id_count": max(object_counts) if object_counts else "",
        "total_object_id_count": rolling.get("total_object_id_count", ""),
        "stream_memory_prune_event_count": len(rolling.get("stream_memory_prune_events") or []),
        "stream_object_prune_event_count": len(rolling.get("stream_object_prune_events") or []),
        "output_transaction_count": summary.get("output_transaction_count", ""),
        "durable_memory_mutation_request_count": summary.get("durable_memory_mutation_request_count", ""),
        "visual_review_image_count": summary.get("visual_review_image_count", ""),
        "full_scene_video_exists": bool(summary.get("full_scene_video_exists")),
        "full_scene_video": video.get("path", ""),
        "full_scene_video_sha256": video.get("sha256", ""),
        "casebook_manifest": summary.get("phase12_casebook_manifest", ""),
        "casebook_manifest_sha256": summary.get("phase12_casebook_manifest_sha256", ""),
        "oom_frame": oom_frame,
        "metrics_are_diagnostic_only": True,
        "quality_decision_rule": "Only high-resolution visual confirmation can decide good or bad.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run", action="append", default=[], help="name=path/to/phase12_full_online_summary.json")
    args = parser.parse_args()
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(name, path) for name, path in (parse_run(item) for item in args.run)]
    csv_path = output_root / "phase13_scaling_rows.csv"
    json_path = output_root / "phase13_scaling_summary.json"
    md_path = output_root / "phase13_scaling_summary.md"
    write_csv(csv_path, rows)
    payload = {
        "schema_version": "stream4d_v108_phase13_scaling_summary_v1",
        "status": "PHASE13_SCALING_DIAGNOSTIC_ONLY",
        "rows": rows,
        "metrics_are_diagnostic_only": True,
        "quality_decision_rule": "Only high-resolution visual confirmation can decide good or bad.",
        "row_count": len(rows),
    }
    write_json(json_path, payload)
    lines = [
        "# Phase13 Scaling Summary",
        "",
        "status: PHASE13_SCALING_DIAGNOSTIC_ONLY",
        "metrics_are_diagnostic_only: true",
        "",
        "| run | frames | wall/frame | peak CUDA MB | final active | video |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['frame_count']} | {row['wall_sec_per_frame']} | "
            f"{row['peak_cuda_memory_mb']} | {row['final_active_stream_object_count']} | {row['full_scene_video_exists']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "row_count": len(rows),
                "csv": rel(csv_path),
                "summary": rel(json_path),
                "markdown": rel(md_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
