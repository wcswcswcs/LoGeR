#!/usr/bin/env python3
"""Run the v118 Stage4 HS-GLA write/retention pilot matrix on multiple GPUs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python")
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE = RESULT_ROOT / "stage4_r11_hs_gla_write_retention_pilot"
JOB_MANIFEST = STAGE / "matrix_job_manifest.csv"
RUN_RESULTS = STAGE / "matrix_run_results.csv"
RUN_RESULTS_JSONL = STAGE / "matrix_run_results.jsonl"

RUN_TYPES = [
    {
        "branch": "HGW",
        "run_type": "candidate",
        "variant": "HGW1",
        "action": "HS_GW1_internal_candidate_write_value_tiny_tight",
        "control": "",
        "prefix": "stage4_r11_hgw1_internal_candidate_write_value_tiny_tight",
        "intervention_form": "write_value_scaling",
    },
    {
        "branch": "HGW",
        "run_type": "candidate",
        "variant": "HGW2",
        "action": "HS_GW2_semantic_role_write_value_tiny_tight",
        "control": "",
        "prefix": "stage4_r11_hgw2_semantic_role_write_value_tiny_tight",
        "intervention_form": "write_value_scaling",
    },
    {
        "branch": "HGW",
        "run_type": "candidate",
        "variant": "HGW3",
        "action": "HS_GW3_internal_plus_semantic_write_value_tiny_tight",
        "control": "",
        "prefix": "stage4_r11_hgw3_internal_plus_semantic_write_value_tiny_tight",
        "intervention_form": "write_value_scaling",
    },
    {
        "branch": "HGW",
        "run_type": "candidate",
        "variant": "HGW4",
        "action": "HS_GW4_state_delta_gain_tiny_tight",
        "control": "",
        "prefix": "stage4_r11_hgw4_state_delta_gain_tiny_tight",
        "intervention_form": "state_delta_scaling",
    },
    {
        "branch": "HGW",
        "run_type": "candidate",
        "variant": "HGW5",
        "action": "HS_GW5_full_candidate_reliability_calibration_tiny_tight",
        "control": "",
        "prefix": "stage4_r11_hgw5_full_candidate_reliability_calibration_tiny_tight",
        "intervention_form": "state_delta_scaling",
    },
    {
        "branch": "HGW",
        "run_type": "matched_control",
        "variant": "HGW3_RANDOM_SIGN",
        "action": "HS_GW3_internal_plus_semantic_write_value_tiny_tight",
        "control": "same_magnitude_random_sign",
        "prefix": "stage4_r11_hgw3_internal_plus_semantic_write_value_tiny_tight_random_sign",
        "intervention_form": "write_value_scaling",
    },
    {
        "branch": "HGW",
        "run_type": "matched_control",
        "variant": "HGW5_RANDOM_SIGN",
        "action": "HS_GW5_full_candidate_reliability_calibration_tiny_tight",
        "control": "same_magnitude_random_sign",
        "prefix": "stage4_r11_hgw5_full_candidate_reliability_calibration_tiny_tight_random_sign",
        "intervention_form": "state_delta_scaling",
    },
    {
        "branch": "HGR",
        "run_type": "candidate",
        "variant": "HGR1",
        "action": "HS_GR1_fixed_reference_reliability_only_state_delta_proxy_tiny_tight",
        "control": "",
        "prefix": "stage4_r11_hgr1_fixed_reference_reliability_only_state_delta_proxy_tiny_tight",
        "intervention_form": "retention_proxy_state_delta_gain",
    },
    {
        "branch": "HGR",
        "run_type": "candidate",
        "variant": "HGR2",
        "action": "HS_GR2_candidate_plus_reliability_state_delta_proxy_tiny_tight",
        "control": "",
        "prefix": "stage4_r11_hgr2_candidate_plus_reliability_state_delta_proxy_tiny_tight",
        "intervention_form": "retention_proxy_state_delta_gain",
    },
    {
        "branch": "HGR",
        "run_type": "candidate",
        "variant": "HGR3",
        "action": "HS_GR3_semantic_lifetime_prior_plus_reliability_proxy_tiny_tight",
        "control": "",
        "prefix": "stage4_r11_hgr3_semantic_lifetime_prior_plus_reliability_proxy_tiny_tight",
        "intervention_form": "retention_proxy_state_delta_gain",
    },
    {
        "branch": "HGR",
        "run_type": "candidate",
        "variant": "HGR4",
        "action": "HS_GR4_full_three_way_calibrated_retention_proxy_tiny_tight",
        "control": "",
        "prefix": "stage4_r11_hgr4_full_three_way_calibrated_retention_proxy_tiny_tight",
        "intervention_form": "retention_proxy_state_delta_gain",
    },
    {
        "branch": "HGR",
        "run_type": "matched_control",
        "variant": "HGR4_RANDOM_SIGN",
        "action": "HS_GR4_full_three_way_calibrated_retention_proxy_tiny_tight",
        "control": "same_magnitude_random_sign",
        "prefix": "stage4_r11_hgr4_full_three_way_calibrated_retention_proxy_tiny_tight_random_sign",
        "intervention_form": "retention_proxy_state_delta_gain",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--seqs", default="00,02")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing-success", action="store_true")
    return parser.parse_args()


def command_for(job: dict[str, Any]) -> list[str]:
    cmd = [
        str(PYTHON),
        str(ROOT / "tools/run_v114tf_hs_lq_case.py"),
        "--action",
        job["action"],
        "--seq",
        job["seq"],
        "--gpu",
        str(job["gpu"]),
        "--output-prefix",
        job["prefix"],
        "--results-root",
        str(STAGE),
        "--trace-enable",
        "1",
        "--action-audit-enable",
        "1",
        "--trace-gla-enable",
        "0",
        "--conda-env",
        "loger",
    ]
    if job.get("control"):
        cmd += ["--control", job["control"]]
    if int(job.get("max_frames") or 0) > 0:
        cmd += ["--max-frames", str(int(job["max_frames"]))]
    return cmd


def case_name(prefix: str, seq: str, max_frames: int) -> str:
    suffix = f"max{max_frames}_kitti_{seq}" if max_frames > 0 else f"full_kitti_{seq}"
    return f"{prefix}_{suffix}"


def read_manifest_returncode(job: dict[str, Any]) -> int | None:
    manifest = STAGE / "diagnostics" / case_name(job["prefix"], job["seq"], int(job["max_frames"])) / "run_manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    rc = data.get("returncode")
    return int(rc) if rc is not None else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    cmd = command_for(job)
    stdout_path = STAGE / "scheduler_logs" / f"{job['job_id']}.stdout.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = STAGE / "diagnostics" / case_name(job["prefix"], job["seq"], int(job["max_frames"])) / "run_manifest.json"
    if job.get("skip_existing_success") and read_manifest_returncode(job) == 0:
        return {
            **job,
            "returncode": 0,
            "elapsed_sec": 0.0,
            "stdout_log": str(stdout_path),
            "child_manifest": str(manifest),
            "skipped_existing_success": True,
        }
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as handle:
        handle.write("# scheduler child command\n")
        handle.write(" ".join(cmd) + "\n\n")
        handle.write("# scheduler thread env\n")
        thread_count = str(int(job.get("omp_threads") or 8))
        for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            handle.write(f"{key}={thread_count}\n")
        handle.write("\n")
        handle.flush()
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": thread_count,
                "MKL_NUM_THREADS": thread_count,
                "OPENBLAS_NUM_THREADS": thread_count,
                "NUMEXPR_NUM_THREADS": thread_count,
            }
        )
        proc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
    return {
        **job,
        "returncode": int(proc.returncode),
        "elapsed_sec": round(time.time() - start, 3),
        "stdout_log": str(stdout_path),
        "child_manifest": str(manifest),
        "skipped_existing_success": False,
    }


def worker(gpu: str, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    del gpu
    return [run_one(job) for job in jobs]


def main() -> None:
    args = parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    seqs = [item.strip() for item in args.seqs.split(",") if item.strip()]
    STAGE.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    idx = 0
    for run_type in RUN_TYPES:
        for seq in seqs:
            gpu = gpus[idx % len(gpus)]
            job = {
                **run_type,
                "seq": seq,
                "gpu": gpu,
                "max_frames": int(args.max_frames),
                "omp_threads": int(args.omp_threads),
                "job_id": f"{idx:03d}_{run_type['variant']}_seq{seq}",
                "skip_existing_success": bool(args.skip_existing_success),
            }
            job["command"] = " ".join(command_for(job))
            jobs.append(job)
            idx += 1
    write_csv(JOB_MANIFEST, jobs)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "job_count": len(jobs), "job_manifest": str(JOB_MANIFEST)}, sort_keys=True))
        return

    queues: dict[str, list[dict[str, Any]]] = {gpu: [] for gpu in gpus}
    for job in jobs:
        queues[str(job["gpu"])].append(job)

    results: list[dict[str, Any]] = []
    RUN_RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool, RUN_RESULTS_JSONL.open("w", encoding="utf-8") as jsonl:
        futures = {pool.submit(worker, gpu, gpu_jobs): gpu for gpu, gpu_jobs in queues.items()}
        for fut in as_completed(futures):
            gpu_results = fut.result()
            for row in gpu_results:
                results.append(row)
                jsonl.write(json.dumps(row, sort_keys=True) + "\n")
                jsonl.flush()
    results.sort(key=lambda row: row["job_id"])
    write_csv(RUN_RESULTS, results)
    print(
        json.dumps(
            {
                "job_count": len(jobs),
                "returncodes": {str(code): sum(1 for row in results if row["returncode"] == code) for code in sorted({row["returncode"] for row in results})},
                "job_manifest": str(JOB_MANIFEST),
                "run_results": str(RUN_RESULTS),
                "run_results_jsonl": str(RUN_RESULTS_JSONL),
            },
            sort_keys=True,
        )
    )
    if any(row["returncode"] != 0 for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
