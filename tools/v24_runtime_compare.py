#!/usr/bin/env python3
"""Compare v24 rollout trajectory reproducibility and runtime summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def _read_traj(path: Path) -> List[Tuple[float, List[float]]]:
    rows: List[Tuple[float, List[float]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].startswith("#"):
                continue
            if len(parts) < 8:
                raise ValueError(f"Unexpected trajectory row in {path}: {line!r}")
            rows.append((float(parts[0]), [float(x) for x in parts[1:8]]))
    return rows


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _wall_seconds_from_status(path: Path) -> int | None:
    if not path.exists():
        return None
    start = None
    end = None
    pattern = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+(START|DONE|FAIL)\b")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        stamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        if match.group(2) == "START":
            start = stamp
        elif match.group(2) in {"DONE", "FAIL"}:
            end = stamp
    if start is None or end is None:
        return None
    return int((end - start).total_seconds())


def _compare_traj(a: Path, b: Path) -> Dict[str, object]:
    aa = _read_traj(a)
    bb = _read_traj(b)
    n = min(len(aa), len(bb))
    max_trans = 0.0
    max_quat = 0.0
    max_ts = 0.0
    for idx in range(n):
        ta, va = aa[idx]
        tb, vb = bb[idx]
        max_ts = max(max_ts, abs(ta - tb))
        trans = max(abs(va[j] - vb[j]) for j in range(3))
        quat = max(abs(va[j] - vb[j]) for j in range(3, 7))
        max_trans = max(max_trans, trans)
        max_quat = max(max_quat, quat)
    return {
        "rows_a": len(aa),
        "rows_b": len(bb),
        "matched_rows": n,
        "row_count_equal": len(aa) == len(bb),
        "max_timestamp_abs_diff": max_ts,
        "max_translation_abs_diff": max_trans,
        "max_quaternion_abs_diff": max_quat,
        "exact_repro_within_1e_9": (
            len(aa) == len(bb)
            and max_ts <= 1e-9
            and max_trans <= 1e-9
            and max_quat <= 1e-9
        ),
    }


def _runtime(run_dir: Path) -> Dict[str, object]:
    wall = _read_json(run_dir / "wall_time_summary.json")
    timing = _read_json(run_dir / "timing_summary.json")
    chunks = timing.get("chunks", []) if isinstance(timing, dict) else []
    pass1 = 0.0
    pass2 = 0.0
    chunk_total = 0.0
    cache_hits = 0
    if isinstance(chunks, list):
        for row in chunks:
            if not isinstance(row, dict):
                continue
            pass1 += float(row.get("pass1_probe_seconds", 0.0) or 0.0)
            pass2 += float(row.get("pass2_control_seconds", 0.0) or 0.0)
            chunk_total += float(row.get("chunk_total_seconds", 0.0) or 0.0)
            cache_hits += int(bool(row.get("probe_cache_hit", False)))
    return {
        "wall_seconds": wall.get("wall_seconds", _wall_seconds_from_status(run_dir / "run_status.txt")),
        "model_load_seconds": timing.get("model_load_seconds") if isinstance(timing, dict) else None,
        "chunks": len(chunks) if isinstance(chunks, list) else 0,
        "probe_cache_hits": cache_hits,
        "pass1_probe_seconds_sum": pass1,
        "pass2_control_seconds_sum": pass2,
        "chunk_total_seconds_sum": chunk_total,
        "total_runtime_seconds_including_model_load": (
            timing.get("total_runtime_seconds_including_model_load")
            if isinstance(timing, dict)
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--seq", default="01")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    baseline_traj = args.baseline_run / f"{args.seq}.txt"
    candidate_traj = args.candidate_run / f"{args.seq}.txt"
    report = {
        "baseline_run": str(args.baseline_run),
        "candidate_run": str(args.candidate_run),
        "trajectory_compare": _compare_traj(baseline_traj, candidate_traj),
        "baseline_runtime": _runtime(args.baseline_run),
        "candidate_runtime": _runtime(args.candidate_run),
    }
    b_wall = report["baseline_runtime"].get("wall_seconds")
    c_wall = report["candidate_runtime"].get("wall_seconds")
    if isinstance(b_wall, (int, float)) and isinstance(c_wall, (int, float)) and c_wall > 0:
        report["wall_speedup_vs_baseline"] = float(b_wall) / float(c_wall)
    else:
        report["wall_speedup_vs_baseline"] = math.nan
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
