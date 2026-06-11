#!/usr/bin/env python3
"""Summarize v52 rollout timing and enforce runtime gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


STAGE_FIELDS = (
    "pass1_probe_seconds",
    "stage_b_seconds",
    "stage_c_seconds",
    "stage_d_seconds",
    "pass2_control_seconds",
    "probe_ttt_write_seconds",
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_status(run_dir: Path) -> str:
    path = run_dir / "run_status.txt"
    if not path.is_file():
        return "missing_status"
    text = path.read_text(encoding="utf-8", errors="replace")
    if "DONE " in text:
        return "done"
    if "FAIL " in text:
        return "fail"
    return "running_or_partial"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summarize_run(run_dir: Path) -> Dict[str, Any]:
    timing = _load_json(run_dir / "timing_summary.json")
    wall = _load_json(run_dir / "wall_time_summary.json")
    chunks = list(timing.get("chunks") or [])
    row: Dict[str, Any] = {
        "run_name": run_dir.name,
        "status": _run_status(run_dir),
        "num_chunks": len(chunks),
        "wall_seconds": _safe_float(wall.get("wall_seconds"), 0.0),
        "total_runtime_seconds_after_model_load": _safe_float(
            timing.get("total_runtime_seconds_after_model_load"), 0.0
        ),
        "total_runtime_seconds_including_model_load": _safe_float(
            timing.get("total_runtime_seconds_including_model_load"), 0.0
        ),
        "model_load_seconds": _safe_float(timing.get("model_load_seconds"), 0.0),
        "save_outputs_seconds": _safe_float(timing.get("save_outputs_seconds"), 0.0),
        "probe_cache_mode": wall.get("probe_cache_mode", timing.get("probe_cache_mode", "")),
        "probe_cache_payload": wall.get("probe_cache_payload", ""),
        "empty_cuda_cache_each_chunk": wall.get("empty_cuda_cache_each_chunk", ""),
    }
    if chunks:
        totals = {field: sum(_safe_float(c.get(field), 0.0) for c in chunks) for field in STAGE_FIELDS}
        chunk_totals = [_safe_float(c.get("chunk_total_seconds"), 0.0) for c in chunks]
        row.update(
            {
                "chunk_total_sum": sum(chunk_totals),
                "chunk_total_mean": mean(chunk_totals),
                "chunk_total_max": max(chunk_totals),
                "probe_cache_hit_rate": mean(1.0 if c.get("probe_cache_hit") else 0.0 for c in chunks),
                "cue_cache_hit_rate": mean(1.0 if c.get("cue_cache_hit") else 0.0 for c in chunks),
            }
        )
        row.update({f"{field}_sum": value for field, value in totals.items()})
        row.update({f"{field}_mean": value / len(chunks) for field, value in totals.items()})
        accounted = sum(totals.values())
        row["chunk_unaccounted_sum"] = row["chunk_total_sum"] - accounted
        row["chunk_unaccounted_mean"] = row["chunk_unaccounted_sum"] / len(chunks)
    return row


def _iter_run_dirs(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    for path in paths:
        if path.is_dir() and (path / "timing_summary.json").is_file():
            out.append(path)
        elif path.is_dir():
            out.extend(sorted(p.parent for p in path.rglob("timing_summary.json")))
    seen = set()
    unique = []
    for path in out:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: List[Dict[str, Any]], path: Path, *, max_wall_seconds: Optional[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "run",
        "status",
        "wall min",
        "chunk mean",
        "pass1 mean",
        "stageB mean",
        "pass2 mean",
        "TTT mean",
        "cache hit",
        "gate",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        wall_seconds = _safe_float(row.get("wall_seconds"), 0.0)
        gate = "n/a"
        if max_wall_seconds is not None and wall_seconds > 0:
            gate = "pass" if wall_seconds <= max_wall_seconds else "fail"
        values = [
            row.get("run_name", ""),
            row.get("status", ""),
            f"{wall_seconds / 60.0:.2f}" if wall_seconds else "",
            f"{_safe_float(row.get('chunk_total_mean')):.2f}",
            f"{_safe_float(row.get('pass1_probe_seconds_mean')):.2f}",
            f"{_safe_float(row.get('stage_b_seconds_mean')):.2f}",
            f"{_safe_float(row.get('pass2_control_seconds_mean')):.2f}",
            f"{_safe_float(row.get('probe_ttt_write_seconds_mean')):.2f}",
            f"{_safe_float(row.get('probe_cache_hit_rate')):.2f}",
            gate,
        ]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Rollout directories or roots containing timing_summary.json files.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=None)
    args = parser.parse_args()

    run_dirs = _iter_run_dirs(Path(p) for p in args.paths)
    rows = [_summarize_run(path) for path in run_dirs]
    rows.sort(key=lambda row: str(row.get("run_name", "")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "num_runs": len(rows),
        "max_wall_seconds": args.max_wall_seconds,
        "num_wall_gate_fail": sum(
            1
            for row in rows
            if args.max_wall_seconds is not None
            and _safe_float(row.get("wall_seconds"), 0.0) > args.max_wall_seconds
        ),
        "runs": rows,
    }
    (out_dir / "v52_runtime_profile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(rows, out_dir / "v52_runtime_profile_summary.csv")
    _write_md(rows, out_dir / "v52_runtime_profile_summary.md", max_wall_seconds=args.max_wall_seconds)
    print(f"Wrote {len(rows)} runtime rows to {out_dir}")


if __name__ == "__main__":
    main()
