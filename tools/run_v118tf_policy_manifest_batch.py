#!/usr/bin/env python3
"""Run selected ACL2 v118 policy manifest rows with bounded parallelism."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_csv_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    include_gpus = parse_csv_set(args.include_gpus) if args.include_gpus else set()
    include_methods = [item for item in args.method_contains if item]
    rows: list[dict[str, str]] = []
    for row in load_rows(args.manifest):
        if row.get("phase") != "run_worker":
            continue
        if include_gpus and row.get("gpu", "") not in include_gpus:
            continue
        if include_methods and not any(item in row.get("method", "") for item in include_methods):
            continue
        rows.append(row)
    return rows


def command_for_row(row: dict[str, str], override_gpu: str | None) -> str:
    command = row["command"]
    if override_gpu is not None:
        command = re.sub(r"CUDA_VISIBLE_DEVICES=\S+", f"CUDA_VISIBLE_DEVICES={override_gpu}", command, count=1)
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--include-gpus", default="")
    parser.add_argument("--method-contains", action="append", default=[])
    parser.add_argument("--override-gpu")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--unique-gpus", action="store_true")
    parser.add_argument("--status-json", type=Path, required=True)
    args = parser.parse_args()

    rows = selected_rows(args)
    if not rows:
        raise SystemExit("no run_worker rows selected")
    args.status_json.parent.mkdir(parents=True, exist_ok=True)

    pending = list(rows)
    running: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    started_at = time.time()

    def write_status() -> None:
        payload = {
            "schema": "acl2_v118tf_policy_manifest_batch_status_v1",
            "manifest": rel(args.manifest),
            "max_parallel": args.max_parallel,
            "include_gpus": args.include_gpus,
            "method_contains": args.method_contains,
            "override_gpu": args.override_gpu,
            "unique_gpus": bool(args.unique_gpus),
            "elapsed_s": time.time() - started_at,
            "pending_count": len(pending),
            "running": [
                {"seq": item["row"].get("seq"), "method": item["row"].get("method"), "gpu": item["gpu"]}
                for item in running
            ],
            "finished": finished,
        }
        args.status_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    while pending or running:
        launched = True
        while pending and len(running) < max(1, args.max_parallel) and launched:
            launched = False
            busy_gpus = {item["gpu"] for item in running if item["gpu"]}
            selected_index = 0
            if args.unique_gpus:
                selected_index = -1
                for idx, candidate in enumerate(pending):
                    candidate_gpu = args.override_gpu if args.override_gpu is not None else candidate.get("gpu", "")
                    if not candidate_gpu or candidate_gpu not in busy_gpus:
                        selected_index = idx
                        break
                if selected_index < 0:
                    break
            row = pending.pop(selected_index)
            gpu = args.override_gpu if args.override_gpu is not None else row.get("gpu", "")
            command = command_for_row(row, args.override_gpu)
            proc = subprocess.Popen(command, shell=True, cwd=row["cwd"])
            running.append({"row": row, "proc": proc, "command": command, "gpu": gpu, "start": time.time()})
            launched = True
            write_status()
        time.sleep(2.0)
        still_running: list[dict[str, Any]] = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still_running.append(item)
                continue
            row = item["row"]
            finished.append(
                {
                    "seq": row.get("seq"),
                    "dataset": row.get("dataset"),
                    "method": row.get("method"),
                    "gpu": item["gpu"],
                    "returncode": rc,
                    "elapsed_s": time.time() - item["start"],
                    "log": row.get("log"),
                    "action_trace": row.get("action_trace"),
                    "command": item["command"],
                }
            )
        running = still_running
        write_status()

    failures = [row for row in finished if int(row["returncode"]) != 0]
    print(json.dumps(json.loads(args.status_json.read_text(encoding="utf-8")), indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
