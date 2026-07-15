#!/usr/bin/env python3
"""Run ACL2 v107TF Stage1 cache-operation trace manifest rows."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention/stage1_cache_operation_instrumentation"
DEFAULT_MANIFEST = STAGE1 / "run_manifest.csv"
DEFAULT_RESULTS_CSV = STAGE1 / "run_results.csv"
DEFAULT_RESULTS_JSONL = STAGE1 / "run_results.jsonl"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            fieldnames = next(reader)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_list(raw: str | None) -> set[str]:
    if raw is None or raw.strip() == "":
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def run_row(row: dict[str, str]) -> dict[str, Any]:
    started = time.time()
    trace_file = row.get("trace_file", "")
    if row.get("phase") == "run_worker_trace" and trace_file:
        Path(trace_file).unlink(missing_ok=True)
    proc = subprocess.run(
        row["command"],
        shell=True,
        cwd=row["cwd"],
        text=True,
        capture_output=True,
    )
    return {
        "schema": "acl2_v107tf_stage1_operation_trace_run_result_v1",
        "target_id": row["target_id"],
        "target_kind": row["target_kind"],
        "run_name": row["run_name"],
        "phase": row["phase"],
        "seq": row["seq"],
        "dataset": row["dataset"],
        "method": row["method"],
        "gpu": row["gpu"],
        "trace_global_idxs": row["trace_global_idxs"],
        "trace_start_idx": row["trace_start_idx"],
        "trace_end_idx_exclusive": row["trace_end_idx_exclusive"],
        "window_index": row["window_index"],
        "cwd": row["cwd"],
        "config": row["config"],
        "trace_file": row["trace_file"],
        "command": row["command"],
        "returncode": proc.returncode,
        "duration_sec": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS_JSONL)
    parser.add_argument("--seqs", default="", help="Comma-separated sequences; empty means all.")
    parser.add_argument("--target-kinds", default="", help="Comma-separated target kinds; empty means all.")
    parser.add_argument("--target-ids", default="", help="Comma-separated target ids; empty means all.")
    parser.add_argument("--phases", default="", help="Comma-separated phases; empty means all.")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    seqs = parse_list(args.seqs)
    kinds = parse_list(args.target_kinds)
    target_ids = parse_list(args.target_ids)
    phases = parse_list(args.phases)
    rows = [
        row for row in read_csv(args.manifest)
        if (not seqs or row["seq"] in seqs)
        and (not kinds or row["target_kind"] in kinds)
        and (not target_ids or row["target_id"] in target_ids)
        and (not phases or row["phase"] in phases)
    ]
    print(f"selected_manifest_rows={len(rows)} max_workers={args.max_workers}")
    if not rows:
        return

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {executor.submit(run_row, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            result = future.result()
            append_csv(args.results_csv, result)
            args.results_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.results_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
            print(
                f"RETURN {result['returncode']} target={row['target_id']} "
                f"phase={row['phase']} duration_sec={result['duration_sec']}"
            )
            if result["returncode"] != 0:
                print(result["stdout_tail"][-2000:])
                print(result["stderr_tail"][-2000:])
                raise SystemExit(int(result["returncode"]))


if __name__ == "__main__":
    main()
