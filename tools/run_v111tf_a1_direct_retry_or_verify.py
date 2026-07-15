#!/usr/bin/env python3
"""Retry or verify ACL2 v111TF A1 manifest rows without `conda run`.

The generic manifest runner records `conda run` return codes.  On this A1
batch, `conda run` returned non-zero for some rows while the child
`run_worker.py` kept running and later wrote valid workspace outputs.  This
helper keeps that audit trail intact and appends a later row that is explicit
about whether it directly retried the command or only verified an existing
output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


ENV_PYTHON = Path("/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python")
CONDA_PYTHON = "/mnt/data/users/chengshun.wang/miniconda3/bin/conda run -n loger python"
RESULT_ROOT = Path("results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory")
A1_ROOT = RESULT_ROOT / "batch_a_a1_anchor_selection"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", encoding="utf-8", newline="") as handle:
            fieldnames = next(csv.reader(handle))
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def latest_successes(results_csv: Path) -> set[tuple[str, str]]:
    if not results_csv.exists():
        return set()
    successes: set[tuple[str, str]] = set()
    for row in read_csv(results_csv):
        key = (row.get("run_name", ""), row.get("phase", ""))
        if str(row.get("returncode", "")).strip() == "0":
            successes.add(key)
        else:
            successes.discard(key)
    return successes


def run_worker_output_path(row: dict[str, str]) -> Path:
    return A1_ROOT / "workspace" / row["dataset"] / row["seq"] / row["method"] / "traj.txt"


def intrinsics_output_path(row: dict[str, str]) -> Path:
    return A1_ROOT / "workspace" / row["dataset"] / row["seq"] / row["method"] / "intrinsics.txt"


def direct_command(row: dict[str, str], override_gpu: str = "") -> str:
    command = row["command"]
    if CONDA_PYTHON not in command:
        direct = command
    else:
        direct = command.replace(CONDA_PYTHON, shlex.quote(str(ENV_PYTHON)), 1)
    if override_gpu:
        if re.search(r"CUDA_VISIBLE_DEVICES=[^ ]+", direct):
            direct = re.sub(r"CUDA_VISIBLE_DEVICES=[^ ]+", f"CUDA_VISIBLE_DEVICES={override_gpu}", direct, count=1)
        else:
            direct = f"CUDA_VISIBLE_DEVICES={override_gpu} {direct}"
    return direct


def result_row(
    row: dict[str, str],
    command: str,
    returncode: int,
    duration_sec: float,
    stdout: str,
    stderr: str,
    gpu: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": "acl2_v108tf_gpu_serial_manifest_run_result_v1",
        "run_name": row["run_name"],
        "phase": row["phase"],
        "target_id": row["target_id"],
        "target_kind": row["target_kind"],
        "seq": row["seq"],
        "dataset": row["dataset"],
        "method": row["method"],
        "action_name": row["action_name"],
        "action_family": row["action_family"],
        "stage4_action_mode": row["stage4_action_mode"],
        "selector": row["selector"],
        "selected_count": row["selected_count"],
        "gpu": row["gpu"] if gpu is None else gpu,
        "cwd": row["cwd"],
        "config": row["config"],
        "trace_file": row["trace_file"],
        "action_file": row["action_file"],
        "command": command,
        "returncode": returncode,
        "duration_sec": round(duration_sec, 3),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def verify_existing(row: dict[str, str]) -> dict[str, Any] | None:
    if row["phase"] != "run_worker":
        return None
    traj = run_worker_output_path(row)
    intrinsics = intrinsics_output_path(row)
    if not (traj.exists() and intrinsics.exists() and traj.stat().st_size > 0 and intrinsics.stat().st_size > 0):
        return None
    message = (
        "verified existing A1 run_worker outputs after orphaned conda-run child; "
        f"traj={traj} bytes={traj.stat().st_size}; "
        f"intrinsics={intrinsics} bytes={intrinsics.stat().st_size}"
    )
    return result_row(row, f"VERIFY_EXISTING_OUTPUT {traj}", 0, 0.0, message, "")


def run_direct(row: dict[str, str], override_gpu: str = "") -> dict[str, Any]:
    started = time.time()
    if row.get("phase") == "run_worker":
        for key in ("trace_file", "action_file"):
            value = row.get(key, "")
            if value:
                Path(value).unlink(missing_ok=True)
    command = direct_command(row, override_gpu=override_gpu)
    proc = subprocess.run(command, shell=True, cwd=row["cwd"], text=True, capture_output=True)
    return result_row(
        row,
        command,
        proc.returncode,
        time.time() - started,
        proc.stdout,
        proc.stderr,
        gpu=override_gpu or None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--seqs", default="")
    parser.add_argument("--actions", default="")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--override-gpu", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--retry-missing", action="store_true")
    args = parser.parse_args()

    seqs = {item for item in args.seqs.split(",") if item}
    actions = {item for item in args.actions.split(",") if item}
    successes = set() if args.force else latest_successes(args.results_csv)
    rows = [
        row for row in read_csv(args.manifest)
        if row["phase"] == args.phase
        and (not seqs or row["seq"] in seqs)
        and (not actions or row["action_name"] in actions)
        and (row["run_name"], row["phase"]) not in successes
    ]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    print(f"selected_rows={len(rows)} phase={args.phase} verify_existing={args.verify_existing} retry_missing={args.retry_missing}")

    failures = 0
    for row in rows:
        result: dict[str, Any] | None = None
        if args.verify_existing:
            result = verify_existing(row)
        if result is None and args.retry_missing:
            result = run_direct(row, override_gpu=args.override_gpu)
        if result is None:
            print(f"SKIP {row['run_name']} no existing output and retry disabled", flush=True)
            continue
        append_csv(args.results_csv, result)
        append_jsonl(args.results_jsonl, result)
        print(
            f"RETURN {result['returncode']} run_name={result['run_name']} "
            f"phase={result['phase']} duration_sec={result['duration_sec']}",
            flush=True,
        )
        if int(result["returncode"]) != 0:
            failures += 1
            print(str(result.get("stdout_tail", ""))[-2000:], flush=True)
            print(str(result.get("stderr_tail", ""))[-2000:], flush=True)
            break
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
