#!/usr/bin/env python3
"""Run selected ACL2 v106 Stage4 LingBot runtime manifest rows."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE4 = ROOT / "results/acl2_v106tf_lingbot_semantic_aware_memory_role_control/stage4_local_preserve_reference_block"
MANIFEST = STAGE4 / "run_manifest.csv"
RESULTS_CSV = STAGE4 / "run_results.csv"
RESULTS_JSONL = STAGE4 / "run_results.jsonl"


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
    return {x.strip() for x in raw.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="", help="Comma-separated action names; empty means all.")
    parser.add_argument("--seqs", default="", help="Comma-separated sequences; empty means all.")
    parser.add_argument("--phases", default="", help="Comma-separated phases; empty means all.")
    args = parser.parse_args()

    actions = parse_list(args.actions)
    seqs = parse_list(args.seqs)
    phases = parse_list(args.phases)
    rows = read_csv(MANIFEST)
    selected = [
        row for row in rows
        if (not actions or row["action_name"] in actions)
        and (not seqs or row["seq"] in seqs)
        and (not phases or row["phase"] in phases)
    ]
    print(f"selected_manifest_rows={len(selected)}")
    for row in selected:
        cwd = row["cwd"]
        command = row["command"]
        started = time.time()
        print(f"RUN {row['run_name']} {row['phase']}")
        proc = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True)
        duration = time.time() - started
        result = {
            "schema": "acl2_v106tf_stage4_runtime_run_result_v1",
            "run_name": row["run_name"],
            "phase": row["phase"],
            "seq": row["seq"],
            "dataset": row["dataset"],
            "method": row["method"],
            "action_name": row["action_name"],
            "action_family": row["action_family"],
            "stage4_action_mode": row["stage4_action_mode"],
            "head_action_pair_count": row["head_action_pair_count"],
            "cwd": cwd,
            "command": command,
            "returncode": proc.returncode,
            "duration_sec": round(duration, 3),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
        append_csv(RESULTS_CSV, result)
        with RESULTS_JSONL.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        print(f"RETURN {proc.returncode} duration_sec={duration:.3f}")
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
