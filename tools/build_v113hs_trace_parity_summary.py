#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HorizonStream no-trace and trace-only outputs.")
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--sequence", default="00/02")
    parser.add_argument("--threshold", type=float, default=1e-6)
    return parser.parse_args()


def read_numeric_txt(path: Path) -> np.ndarray:
    rows = []
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float64)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def compare_npy_dir(base_dir: Path, trace_dir: Path) -> tuple[int, float, list[str]]:
    base_files = sorted(base_dir.glob("*.npy"))
    trace_files = sorted(trace_dir.glob("*.npy"))
    base_names = [p.name for p in base_files]
    trace_names = [p.name for p in trace_files]
    missing = sorted(set(base_names).symmetric_difference(trace_names))
    max_diff = 0.0
    count = 0
    for name in sorted(set(base_names).intersection(trace_names)):
        a = np.load(base_dir / name)
        b = np.load(trace_dir / name)
        max_diff = max(max_diff, max_abs_diff(a, b))
        count += 1
    return count, max_diff, missing


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_root)
    trace_root = Path(args.trace_root)
    results_root = Path(args.results_root)
    diag_dir = results_root / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    seq = args.sequence.strip("/")
    base_seq = baseline_root / seq
    trace_seq = trace_root / seq
    rows = []

    for rel in ["poses/abs_pose.txt", "poses/intri.txt"]:
        a = read_numeric_txt(base_seq / rel)
        b = read_numeric_txt(trace_seq / rel)
        diff = max_abs_diff(a, b)
        rows.append(
            {
                "artifact": rel,
                "count": int(a.shape[0]),
                "max_abs_diff": diff,
                "missing": "",
                "pass": diff <= args.threshold,
            }
        )

    for rel in ["depth/dpt", "depth/conf"]:
        count, diff, missing = compare_npy_dir(base_seq / rel, trace_seq / rel)
        rows.append(
            {
                "artifact": rel,
                "count": count,
                "max_abs_diff": diff,
                "missing": ",".join(missing),
                "pass": diff <= args.threshold and not missing,
            }
        )

    out_csv = diag_dir / "hs_noop_trace_parity_rows.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["artifact", "count", "max_abs_diff", "missing", "pass"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "baseline_root": str(baseline_root),
        "trace_root": str(trace_root),
        "sequence": seq,
        "threshold": args.threshold,
        "rows": rows,
        "pass": all(bool(row["pass"]) for row in rows),
    }
    out_json = diag_dir / "hs_noop_trace_parity_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
