#!/usr/bin/env python3
"""Run ACL2 v106R Stage1 targeted trace manifest rows."""

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
STAGE1_TRACE = ROOT / "results/acl2_v106r_lingbot_memory_operation_discovery_semantic_aware_control/stage1_memory_operation_map/targeted_trace"
MANIFEST = STAGE1_TRACE / "run_manifest.csv"
RESULTS_CSV = STAGE1_TRACE / "run_results.csv"
RESULTS_JSONL = STAGE1_TRACE / "run_results.jsonl"


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
    proc = subprocess.run(
        row["command"],
        shell=True,
        cwd=row["cwd"],
        text=True,
        capture_output=True,
    )
    return {
        "schema": "acl2_v106r_stage1_targeted_trace_run_result_v1",
        "target_id": row["target_id"],
        "target_kind": row["target_kind"],
        "run_name": row["run_name"],
        "phase": row["phase"],
        "seq": row["seq"],
        "dataset": row["dataset"],
        "method": row["method"],
        "gpu": row["gpu"],
        "trace_global_idxs": row["trace_global_idxs"],
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
    parser.add_argument("--seqs", default="", help="Comma-separated sequences; empty means all.")
    parser.add_argument("--target-kinds", default="", help="Comma-separated target kinds; empty means all.")
    parser.add_argument("--phases", default="", help="Comma-separated phases; empty means all.")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    seqs = parse_list(args.seqs)
    kinds = parse_list(args.target_kinds)
    phases = parse_list(args.phases)
    rows = [
        row for row in read_csv(MANIFEST)
        if (not seqs or row["seq"] in seqs)
        and (not kinds or row["target_kind"] in kinds)
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
            append_csv(RESULTS_CSV, result)
            with RESULTS_JSONL.open("a", encoding="utf-8") as handle:
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
