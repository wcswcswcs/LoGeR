#!/usr/bin/env python3
"""Run ACL2 v105-TF LingBot Stage4 head-local trace manifest rows."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE4_HEAD = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage4_lingbot_headlocal_trace"
MANIFEST = STAGE4_HEAD / "run_manifest.csv"
RESULTS_CSV = STAGE4_HEAD / "run_results.csv"
RESULTS_JSONL = STAGE4_HEAD / "run_results.jsonl"


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
    parser.add_argument("--seqs", default="", help="Comma-separated sequences; empty means all.")
    parser.add_argument("--phases", default="", help="Comma-separated phases; empty means all.")
    args = parser.parse_args()

    seqs = parse_list(args.seqs)
    phases = parse_list(args.phases)
    rows = read_csv(MANIFEST)
    selected = [
        row for row in rows
        if (not seqs or row["seq"] in seqs)
        and (not phases or row["phase"] in phases)
    ]
    print(f"selected_manifest_rows={len(selected)}")
    for row in selected:
        started = time.time()
        print(f"RUN {row['run_name']} {row['phase']}")
        proc = subprocess.run(row["command"], shell=True, cwd=row["cwd"], text=True, capture_output=True)
        duration = time.time() - started
        result = {
            "schema": "acl2_v105tf_lingbot_stage4_headlocal_run_result_v1",
            "run_name": row["run_name"],
            "phase": row["phase"],
            "seq": row["seq"],
            "dataset": row["dataset"],
            "method": row["method"],
            "cwd": row["cwd"],
            "command": row["command"],
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
