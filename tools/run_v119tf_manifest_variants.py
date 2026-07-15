#!/usr/bin/env python3
"""Run selected v119 manifest variants with per-variant worker/evaluate order."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def log_success(path: Path, markers: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(marker in text for marker in markers)


def method_root(run_root: Path, row: dict[str, str]) -> Path:
    return run_root / "workspace" / row["dataset"] / str(row["seq"]).zfill(2) / row["method"]


def clean_target(run_root: Path, worker: dict[str, str]) -> None:
    root = method_root(run_root, worker)
    if root.exists():
        shutil.rmtree(root)
    action_file = Path(worker.get("action_file", ""))
    if action_file.exists():
        action_file.write_text("", encoding="utf-8")


def run_shell(row: dict[str, str]) -> int:
    proc = subprocess.run(
        row["command"],
        cwd=row["cwd"],
        shell=True,
        executable="/bin/bash",
    )
    return int(proc.returncode)


def run_job(
    *,
    run_root: Path,
    seq: str,
    variant: str,
    worker: dict[str, str],
    evaluate: dict[str, str],
    clean: bool,
    skip_existing: bool,
    gpu_locks: dict[str, threading.Lock] | None = None,
) -> dict[str, Any]:
    if skip_existing and log_success(Path(evaluate["log"]), ["Total successful: 1", "Total failed: 0"]):
        return {
            "seq": seq,
            "variant": variant,
            "status": "skipped_existing_success",
            "worker_return": "",
            "evaluate_return": "",
            "evaluate_log_success": True,
            "worker_log": worker["log"],
            "evaluate_log": evaluate["log"],
        }
    gpu = str(worker.get("gpu", "")).strip()
    lock = gpu_locks.get(gpu) if gpu_locks and gpu else None
    lock_wait_started = time.time()
    with lock if lock is not None else nullcontext():
        gpu_lock_wait_sec = round(time.time() - lock_wait_started, 3) if lock is not None else 0.0
        if clean:
            clean_target(run_root, worker)
        started = time.time()
        worker_return = run_shell(worker)
        evaluate_return: int | str = ""
        evaluate_log_success = False
        if worker_return == 0:
            evaluate_return = run_shell(evaluate)
            evaluate_log_success = evaluate_return == 0 and log_success(
                Path(evaluate["log"]),
                ["Total successful: 1", "Total failed: 0"],
            )
    return {
        "seq": seq,
        "variant": variant,
        "status": "complete" if worker_return == 0 and evaluate_return == 0 and evaluate_log_success else "failed",
        "worker_return": worker_return,
        "evaluate_return": evaluate_return,
        "evaluate_log_success": evaluate_log_success,
        "elapsed_sec": round(time.time() - started, 3),
        "gpu": gpu,
        "gpu_lock_wait_sec": gpu_lock_wait_sec,
        "worker_log": worker["log"],
        "evaluate_log": evaluate["log"],
        "action_file": worker.get("action_file", ""),
        "method": worker.get("method", ""),
        "dataset": worker.get("dataset", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seqs", default="00,02")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--status-json", default="")
    args = parser.parse_args()

    manifest = read_manifest(Path(args.manifest))
    run_root = Path(args.run_root)
    seqs = [part.strip().zfill(2) for part in args.seqs.split(",") if part.strip()]
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in manifest:
        rows[(str(row["seq"]).zfill(2), row["variant"], row["phase"])] = row

    jobs: list[tuple[str, str, dict[str, str], dict[str, str]]] = []
    missing: list[str] = []
    for seq in seqs:
        for variant in variants:
            worker = rows.get((seq, variant, "run_worker"))
            evaluate = rows.get((seq, variant, "evaluate"))
            if not worker or not evaluate:
                missing.append(f"{seq}:{variant}")
                continue
            jobs.append((seq, variant, worker, evaluate))
    if missing:
        raise RuntimeError(f"missing manifest jobs: {missing}")

    results: list[dict[str, Any]] = []
    gpu_locks = {
        str(worker.get("gpu", "")).strip(): threading.Lock()
        for _, _, worker, _ in jobs
        if str(worker.get("gpu", "")).strip()
    }
    executor_workers = max(1, int(args.max_parallel))
    if gpu_locks:
        executor_workers = max(executor_workers, len(jobs))
    with ThreadPoolExecutor(max_workers=executor_workers) as pool:
        futures = [
            pool.submit(
                run_job,
                run_root=run_root,
                seq=seq,
                variant=variant,
                worker=worker,
                evaluate=evaluate,
                clean=bool(args.clean),
                skip_existing=not args.no_skip_existing,
                gpu_locks=gpu_locks,
            )
            for seq, variant, worker, evaluate in jobs
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "schema": "acl2_v119tf_manifest_variant_runner_summary_v1",
        "manifest": str(Path(args.manifest)),
        "run_root": str(run_root),
        "seqs": seqs,
        "variants": variants,
        "max_parallel": int(args.max_parallel),
        "executor_workers": executor_workers,
        "clean": bool(args.clean),
        "result_count": len(results),
        "failed_count": sum(1 for item in results if item["status"] == "failed"),
        "results": sorted(results, key=lambda item: (item["seq"], item["variant"])),
    }
    status_path = Path(args.status_json) if args.status_json else run_root / "manifest_variant_runner_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
