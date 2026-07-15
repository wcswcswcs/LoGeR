#!/usr/bin/env python3
"""Run v118 HS-GLA OOM repair: smoke audits plus reduced-memory full pilots."""

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
STAGE = RESULT_ROOT / "stage4_r12_hs_gla_oom_repair"
JOB_MANIFEST = STAGE / "repair_job_manifest.csv"
RUN_RESULTS = STAGE / "repair_run_results.csv"
RUN_RESULTS_JSONL = STAGE / "repair_run_results.jsonl"

SMOKE_RUNS = [
    ("HGW1", "HS_GW1_internal_candidate_write_value_tiny_tight", "", "write_value_scaling"),
    ("HGW2", "HS_GW2_semantic_role_write_value_tiny_tight", "", "write_value_scaling"),
    ("HGW3", "HS_GW3_internal_plus_semantic_write_value_tiny_tight", "", "write_value_scaling"),
    ("HGW4", "HS_GW4_state_delta_gain_tiny_tight", "", "state_delta_scaling"),
    ("HGW5", "HS_GW5_full_candidate_reliability_calibration_tiny_tight", "", "state_delta_scaling"),
    ("HGW3_RANDOM_SIGN", "HS_GW3_internal_plus_semantic_write_value_tiny_tight", "same_magnitude_random_sign", "write_value_scaling"),
    ("HGW5_RANDOM_SIGN", "HS_GW5_full_candidate_reliability_calibration_tiny_tight", "same_magnitude_random_sign", "state_delta_scaling"),
    ("HGR1", "HS_GR1_fixed_reference_reliability_only_state_delta_proxy_tiny_tight", "", "retention_proxy_state_delta_gain"),
    ("HGR2", "HS_GR2_candidate_plus_reliability_state_delta_proxy_tiny_tight", "", "retention_proxy_state_delta_gain"),
    ("HGR3", "HS_GR3_semantic_lifetime_prior_plus_reliability_proxy_tiny_tight", "", "retention_proxy_state_delta_gain"),
    ("HGR4", "HS_GR4_full_three_way_calibrated_retention_proxy_tiny_tight", "", "retention_proxy_state_delta_gain"),
    ("HGR4_RANDOM_SIGN", "HS_GR4_full_three_way_calibrated_retention_proxy_tiny_tight", "same_magnitude_random_sign", "retention_proxy_state_delta_gain"),
]

FULL_REDUCED_RUNS = [
    ("HGW3", "HS_GW3_internal_plus_semantic_write_value_tiny_tight", "", "write_value_scaling"),
    ("HGW5", "HS_GW5_full_candidate_reliability_calibration_tiny_tight", "", "state_delta_scaling"),
    ("HGR4", "HS_GR4_full_three_way_calibrated_retention_proxy_tiny_tight", "", "retention_proxy_state_delta_gain"),
    ("HGW3_RANDOM_SIGN", "HS_GW3_internal_plus_semantic_write_value_tiny_tight", "same_magnitude_random_sign", "write_value_scaling"),
    ("HGW5_RANDOM_SIGN", "HS_GW5_full_candidate_reliability_calibration_tiny_tight", "same_magnitude_random_sign", "state_delta_scaling"),
    ("HGR4_RANDOM_SIGN", "HS_GR4_full_three_way_calibrated_retention_proxy_tiny_tight", "same_magnitude_random_sign", "retention_proxy_state_delta_gain"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--seqs", default="00,02")
    parser.add_argument("--smoke-seq", default="00")
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--skip-existing-success", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
        str(job["trace_enable"]),
        "--action-audit-enable",
        str(job["action_audit_enable"]),
        "--trace-gla-enable",
        "0",
        "--conda-env",
        "loger",
    ]
    if job.get("control"):
        cmd += ["--control", job["control"]]
    if int(job.get("max_frames") or 0) > 0:
        cmd += ["--max-frames", str(int(job["max_frames"]))]
    if int(job.get("chunk_block_num") or 0) > 0:
        cmd += ["--chunk-block-num", str(int(job["chunk_block_num"]))]
    if job.get("gq_layer_filter"):
        cmd += ["--gq-layer-filter", str(job["gq_layer_filter"])]
    return cmd


def case_name(prefix: str, seq: str, max_frames: int) -> str:
    suffix = f"max{max_frames}_kitti_{seq}" if max_frames > 0 else f"full_kitti_{seq}"
    return f"{prefix}_{suffix}"


def read_manifest_returncode(job: dict[str, Any]) -> int | None:
    path = STAGE / "diagnostics" / case_name(job["prefix"], job["seq"], int(job["max_frames"])) / "run_manifest.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
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
        return {**job, "returncode": 0, "elapsed_sec": 0.0, "stdout_log": str(stdout_path), "child_manifest": str(manifest), "skipped_existing_success": True}
    thread_count = str(int(job.get("omp_threads") or 8))
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
        }
    )
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as handle:
        handle.write("# scheduler child command\n")
        handle.write(" ".join(cmd) + "\n\n")
        handle.write("# scheduler thread env\n")
        for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            handle.write(f"{key}={thread_count}\n")
        handle.write("\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
    return {**job, "returncode": int(proc.returncode), "elapsed_sec": round(time.time() - start, 3), "stdout_log": str(stdout_path), "child_manifest": str(manifest), "skipped_existing_success": False}


def worker(gpu: str, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    del gpu
    return [run_one(job) for job in jobs]


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    seqs = [item.strip() for item in args.seqs.split(",") if item.strip()]
    jobs: list[dict[str, Any]] = []
    idx = 0
    for variant, action, control, form in SMOKE_RUNS:
        gpu = gpus[idx % len(gpus)]
        job = {
            "phase": "smoke_max128_audit",
            "variant": variant,
            "action": action,
            "control": control,
            "intervention_form": form,
            "seq": args.smoke_seq,
            "gpu": gpu,
            "max_frames": 128,
            "chunk_block_num": 0,
            "gq_layer_filter": "",
            "trace_enable": "1",
            "action_audit_enable": "1",
            "omp_threads": int(args.omp_threads),
            "prefix": f"stage4_r12_smoke_{variant.lower()}",
            "job_id": f"{idx:03d}_smoke_{variant}_seq{args.smoke_seq}",
            "skip_existing_success": bool(args.skip_existing_success),
        }
        job["command"] = " ".join(command_for(job))
        jobs.append(job)
        idx += 1
    for variant, action, control, form in FULL_REDUCED_RUNS:
        for seq in seqs:
            gpu = gpus[idx % len(gpus)]
            job = {
                "phase": "full_chunkblock1_layer23_notrace",
                "variant": variant,
                "action": action,
                "control": control,
                "intervention_form": form,
                "seq": seq,
                "gpu": gpu,
                "max_frames": 0,
                "chunk_block_num": 1,
                "gq_layer_filter": "23",
                "trace_enable": "0",
                "action_audit_enable": "0",
                "omp_threads": int(args.omp_threads),
                "prefix": f"stage4_r12_full_cb1_l23_{variant.lower()}",
                "job_id": f"{idx:03d}_full_cb1_l23_{variant}_seq{seq}",
                "skip_existing_success": bool(args.skip_existing_success),
            }
            job["command"] = " ".join(command_for(job))
            jobs.append(job)
            idx += 1
    return jobs


def main() -> None:
    args = parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    STAGE.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
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
            for row in fut.result():
                results.append(row)
                jsonl.write(json.dumps(row, sort_keys=True) + "\n")
                jsonl.flush()
    results.sort(key=lambda row: row["job_id"])
    write_csv(RUN_RESULTS, results)
    print(json.dumps({"job_count": len(jobs), "returncodes": {str(code): sum(1 for row in results if row["returncode"] == code) for code in sorted({row["returncode"] for row in results})}, "job_manifest": str(JOB_MANIFEST), "run_results": str(RUN_RESULTS), "run_results_jsonl": str(RUN_RESULTS_JSONL)}, sort_keys=True))
    if any(row["returncode"] != 0 for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
