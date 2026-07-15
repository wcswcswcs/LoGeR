#!/usr/bin/env python3
"""Run policy manifest rows with per-GPU serial queues."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
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


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_list(raw: str | None) -> set[str]:
    if raw is None or raw.strip() == "":
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def successful_keys(results_csv: Path) -> set[tuple[str, str]]:
    if not results_csv.exists():
        return set()
    out: set[tuple[str, str]] = set()
    for row in read_csv(results_csv):
        if str(row.get("returncode", "")).strip() == "0":
            out.add((row.get("run_name", ""), row.get("phase", "")))
    return out


def run_row(row: dict[str, str]) -> dict[str, Any]:
    started = time.time()
    if row.get("phase") == "run_worker":
        for key in ("trace_file", "action_file"):
            value = row.get(key, "")
            if value:
                Path(value).unlink(missing_ok=True)
    proc = subprocess.run(
        row["command"],
        shell=True,
        cwd=row["cwd"],
        text=True,
        capture_output=True,
    )
    return {
        "schema": "acl2_v108tf_gpu_serial_manifest_run_result_v1",
        "run_name": row["run_name"],
        "phase": row["phase"],
        "target_id": row["target_id"],
        "target_kind": row["target_kind"],
        "seq": row["seq"],
        "dataset": row["dataset"],
        "method": row["method"],
        "action_name": row.get("action_name", ""),
        "action_family": row.get("action_family", ""),
        "stage4_action_mode": row.get("stage4_action_mode", ""),
        "selector": row.get("selector", ""),
        "selected_count": row.get("selected_count", ""),
        "gpu": row["gpu"],
        "cwd": row["cwd"],
        "config": row["config"],
        "trace_file": row["trace_file"],
        "action_file": row.get("action_file", ""),
        "command": row["command"],
        "returncode": proc.returncode,
        "duration_sec": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_queue(
    queue_key: str,
    rows: list[dict[str, str]],
    results_csv: Path,
    results_jsonl: Path,
    csv_lock: threading.Lock,
    jsonl_lock: threading.Lock,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        result = run_row(row)
        append_csv(results_csv, result, csv_lock)
        append_jsonl(results_jsonl, result, jsonl_lock)
        print(
            f"RETURN {result['returncode']} queue={queue_key} target={row['target_id']} "
            f"action={row.get('action_name', '')} phase={row['phase']} duration_sec={result['duration_sec']}",
            flush=True,
        )
        if result["returncode"] != 0:
            print(result["stdout_tail"][-2000:], flush=True)
            print(result["stderr_tail"][-2000:], flush=True)
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--actions", default="")
    parser.add_argument("--seqs", default="")
    parser.add_argument("--target-kinds", default="")
    parser.add_argument("--target-ids", default="")
    parser.add_argument("--phases", default="")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--serial-key", default="gpu")
    parser.add_argument("--skip-existing-success", action="store_true")
    args = parser.parse_args()

    actions = parse_list(args.actions)
    seqs = parse_list(args.seqs)
    kinds = parse_list(args.target_kinds)
    target_ids = parse_list(args.target_ids)
    phases = parse_list(args.phases)
    successes = successful_keys(args.results_csv) if args.skip_existing_success else set()
    rows = [
        row for row in read_csv(args.manifest)
        if (not actions or row.get("action_name", "") in actions)
        and (not seqs or row["seq"] in seqs)
        and (not kinds or row["target_kind"] in kinds)
        and (not target_ids or row["target_id"] in target_ids)
        and (not phases or row["phase"] in phases)
        and (row["run_name"], row["phase"]) not in successes
    ]
    print(
        f"selected_manifest_rows={len(rows)} max_workers={args.max_workers} "
        f"serial_key={args.serial_key} skipped_success={len(successes)}",
        flush=True,
    )
    if not rows:
        return

    queues: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        queues.setdefault(row.get(args.serial_key, ""), []).append(row)
    csv_lock = threading.Lock()
    jsonl_lock = threading.Lock()
    all_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.max_workers, len(queues)))) as executor:
        futures = {
            executor.submit(run_queue, key, queue_rows, args.results_csv, args.results_jsonl, csv_lock, jsonl_lock): key
            for key, queue_rows in queues.items()
        }
        for future in as_completed(futures):
            all_results.extend(future.result())
    failures = [row for row in all_results if int(row["returncode"]) != 0]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
