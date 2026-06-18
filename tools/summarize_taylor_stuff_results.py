#!/usr/bin/env python3
"""Summarize Taylor sparse stuff experiment metrics.

This is an audit helper. It scans result directories for metrics_summary.json
files, extracts coverage and stuff track statistics when present, and writes a
CSV/JSON table for comparing existing backend candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Taylor stuff result metrics.")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_csv", default="results/taylor_stuff_results_summary.csv")
    parser.add_argument("--output_json", default="results/taylor_stuff_results_summary.json")
    parser.add_argument("--name_contains", default="v2_taylor_")
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _coverage_block(data: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("coverage", "after", "after_ensemble", "after_exclusive"):
        value = data.get(key)
        if isinstance(value, dict) and "stuff_coverage_mean" in value:
            return value
    if "stuff_coverage_mean" in data:
        return data
    return {}


def _track_stats(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("track_stats", "track_stats_after"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _stuff_track_summary(tracks: Iterable[Dict[str, Any]]) -> Tuple[int, str, str]:
    stuff = [
        track
        for track in tracks
        if str(track.get("source_type", "")) == "stuff_static"
    ]
    labels: List[str] = []
    details: List[str] = []
    for track in stuff:
        label = str(track.get("label") or track.get("L_sem") or "")
        if label:
            labels.append(label)
        frames = track.get("frames")
        mean_area = track.get("mean_area_ratio")
        if label:
            details.append(f"{label}:{frames}:{mean_area}")
    return len(stuff), ",".join(sorted(set(labels))), ";".join(details)


def _infer_scope(path: Path) -> str:
    parts = path.parts
    text = "/".join(parts).lower()
    if "stuff_only_slice_120_299" in text or "120_299" in text:
        return "slice120_299"
    if "stuff_only_full" in text:
        return "stuff_only_full"
    if "stuff_only" in text:
        return "stuff_only"
    return "full_or_mixed"


def _safe_read_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics_summary.json")):
        rel = metrics_path.relative_to(root)
        if str(args.name_contains) not in str(rel):
            continue
        data = _safe_read_json(metrics_path)
        if data is None:
            continue
        cov = _coverage_block(data)
        if not cov:
            continue
        tracks = _track_stats(data)
        stuff_count, stuff_labels, stuff_details = _stuff_track_summary(tracks)
        result_dir = metrics_path.parent
        video = data.get("output_video") or data.get("video") or str(result_dir / "overlay_final.mp4")
        row = {
            "result": str(result_dir),
            "scope": _infer_scope(metrics_path),
            "metrics": str(metrics_path),
            "video": str(video),
            "coverage_mean": _as_float(cov.get("coverage_mean")),
            "coverage_p10": _as_float(cov.get("coverage_p10")),
            "coverage_p50": _as_float(cov.get("coverage_p50")),
            "coverage_p90": _as_float(cov.get("coverage_p90")),
            "stuff_coverage_mean": _as_float(cov.get("stuff_coverage_mean")),
            "stuff_coverage_p10": _as_float(cov.get("stuff_coverage_p10")),
            "stuff_coverage_p50": _as_float(cov.get("stuff_coverage_p50")),
            "stuff_coverage_p90": _as_float(cov.get("stuff_coverage_p90")),
            "num_tracks": data.get("num_tracks"),
            "elapsed_sec": _as_float(data.get("elapsed_sec")),
            "stuff_track_count": stuff_count,
            "stuff_labels": stuff_labels,
            "stuff_track_details": stuff_details,
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["scope"],
            -1.0 if row["stuff_coverage_mean"] is None else -float(row["stuff_coverage_mean"]),
            row["result"],
        )
    )
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "result",
        "scope",
        "coverage_mean",
        "coverage_p10",
        "coverage_p50",
        "coverage_p90",
        "stuff_coverage_mean",
        "stuff_coverage_p10",
        "stuff_coverage_p50",
        "stuff_coverage_p90",
        "stuff_track_count",
        "stuff_labels",
        "num_tracks",
        "elapsed_sec",
        "video",
        "metrics",
        "stuff_track_details",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output_csv": str(output_csv), "output_json": str(output_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
