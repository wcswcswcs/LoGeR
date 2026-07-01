#!/usr/bin/env python3
"""Run trace-disabled READ_NO_ACTION baselines and compare pose output SHA.

This is a diagnostic-only parity helper for ACL2 v104.  It reuses the already
recorded job_summary commands from a trace run, removes SWA raw-trace and
geometry sidecar dump flags, and compares the resulting pose text files with
the trace-run pose text files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


TRACE_FLAGS_WITH_VALUES = {
    "--swa_raw_transport_trace_dir",
    "--swa_raw_transport_trace_layer_mode",
    "--swa_raw_transport_trace_single_layer",
    "--swa_raw_transport_trace_max_queries",
    "--swa_raw_transport_trace_topk",
    "--swa_raw_transport_trace_direct_match_only",
    "--swa_raw_transport_trace_query_block_size",
    "--enable_swa_prev_ttt_tracked_instance_query_soft_trace",
    "--swa_prev_ttt_tracked_instance_query_soft_rho",
    "--swa_prev_ttt_tracked_instance_query_soft_min_keep",
    "--swa_prev_ttt_tracked_instance_query_soft_query_head_frac_threshold",
    "--swa_prev_ttt_tracked_instance_query_soft_topk",
    "--swa_prev_ttt_tracked_instance_query_soft_query_block_size",
    "--swa_prev_ttt_tracked_instance_query_soft_layer_mode",
    "--swa_prev_ttt_tracked_instance_query_soft_single_layer",
    "--swa_prev_ttt_tracked_instance_query_soft_attention_mass_max_queries",
    "--swa_prev_ttt_tracked_instance_query_soft_min_direct_witness_seeds",
    "--swa_prev_ttt_tracked_instance_query_soft_direct_match_mode",
    "--enable_swa_prev_ttt_tracked_instance_query_soft_action",
    "--swa_prev_ttt_tracked_instance_query_soft_action_runtime_authorized",
    "--per_chunk_geometry_dir",
    "--per_chunk_pose_trace_jsonl",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def remove_flags(cmd: list[str], flags: set[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] in flags:
            i += 2
            continue
        out.append(cmd[i])
        i += 1
    return out


def replace_flag(cmd: list[str], flag: str, value: str) -> list[str]:
    out = list(cmd)
    try:
        idx = out.index(flag)
    except ValueError:
        out.extend([flag, value])
        return out
    if idx + 1 >= len(out):
        out.append(value)
    else:
        out[idx + 1] = value
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pose_txt(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    return rows


def pose_max_abs_diff(a: Path, b: Path) -> float | str:
    try:
        rows_a = load_pose_txt(a)
        rows_b = load_pose_txt(b)
    except Exception as exc:
        return f"parse_error:{exc}"
    if len(rows_a) != len(rows_b):
        return "row_count_mismatch"
    max_diff = 0.0
    for row_a, row_b in zip(rows_a, rows_b):
        if len(row_a) != len(row_b):
            return "column_count_mismatch"
        for va, vb in zip(row_a, row_b):
            max_diff = max(max_diff, abs(va - vb))
    return max_diff


def build_job(trace_job_summary: Path, output_root: Path) -> dict[str, Any]:
    source = read_json(trace_job_summary)
    case_id = source["case_id"]
    seq = source["seq"]
    out_dir = output_root / case_id / "READ_NO_ACTION"
    output_txt = out_dir / f"{seq}.txt"
    cmd = remove_flags(list(source["cmd"]), TRACE_FLAGS_WITH_VALUES)
    cmd = replace_flag(cmd, "--output_txt", str(output_txt))
    cmd = replace_flag(cmd, "--hybrid_debug_jsonl", str(out_dir / "hmc_state_hash.jsonl"))
    cmd = replace_flag(cmd, "--read_cue_patch_dump_dir", str(out_dir / "read_cue_patch_dumps"))
    return {
        "case_id": case_id,
        "seq": seq,
        "trace_job_summary": str(trace_job_summary),
        "trace_output_txt": source.get("output_txt", ""),
        "out_dir": str(out_dir),
        "output_txt": str(output_txt),
        "cmd": cmd,
    }


def failure_reason(run_log: Path) -> str:
    if not run_log.is_file():
        return ""
    text = run_log.read_text(encoding="utf-8", errors="replace")
    if "torch.OutOfMemoryError" in text or "CUDA out of memory" in text:
        return "cuda_out_of_memory"
    if "Traceback (most recent call last)" in text:
        return "python_traceback"
    if "ERROR conda.cli.main_run" in text:
        return "conda_run_error"
    if "Killed" in text:
        return "process_killed"
    return "nonzero_returncode"


def existing_result(job: dict[str, Any]) -> dict[str, Any] | None:
    output_txt = Path(job["output_txt"])
    summary_path = Path(job["out_dir"]) / "job_summary.json"
    if not output_txt.is_file() or not summary_path.is_file():
        return None
    result = read_json(summary_path)
    result.pop("cmd", None)
    result["reused_existing_success"] = int(result.get("returncode", 1)) == 0
    return result


def missing_existing_result(job: dict[str, Any]) -> dict[str, Any]:
    result = dict(job)
    result.pop("cmd", None)
    summary_path = Path(job["out_dir"]) / "job_summary.json"
    if summary_path.is_file():
        prior = read_json(summary_path)
        result.update({k: v for k, v in prior.items() if k != "cmd"})
    result.setdefault("returncode", 999)
    result.setdefault("duration_sec", "")
    result.setdefault("run_log", str(Path(job["out_dir"]) / "run.log"))
    result["reused_existing_success"] = False
    result["runtime_action_allowed"] = False
    return result


def external_baseline_result(job: dict[str, Any], baseline_root: Path) -> dict[str, Any]:
    case_id = str(job["case_id"])
    seq = str(job["seq"])
    baseline_dir = baseline_root / case_id / "READ_NO_ACTION"
    baseline_txt = baseline_dir / f"{seq}.txt"
    summary_path = baseline_dir / "job_summary.json"
    result = dict(job)
    result.pop("cmd", None)
    result.update(
        {
            "out_dir": str(baseline_dir),
            "output_txt": str(baseline_txt),
            "run_log": str(baseline_dir / "run.log"),
            "external_baseline_root": str(baseline_root),
            "runtime_action_allowed": False,
        }
    )
    if summary_path.is_file():
        prior = read_json(summary_path)
        result.update(
            {
                k: v
                for k, v in prior.items()
                if k
                not in {
                    "cmd",
                    "case_id",
                    "seq",
                    "trace_job_summary",
                    "trace_output_txt",
                    "out_dir",
                    "output_txt",
                }
            }
        )
    result.setdefault("returncode", 0 if baseline_txt.is_file() else 999)
    result.setdefault("duration_sec", "")
    result["reused_existing_success"] = False
    result["reused_external_baseline_success"] = int(result.get("returncode", 1)) == 0 and baseline_txt.is_file()
    return result


def run_job(job: dict[str, Any], gpu: str) -> dict[str, Any]:
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    start = time.time()
    with run_log.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            job["cmd"],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
            env=env,
            text=True,
        )
    result = dict(job)
    result.pop("cmd", None)
    result.update(
        {
            "gpu": gpu,
            "returncode": proc.returncode,
            "duration_sec": time.time() - start,
            "run_log": str(run_log),
            "runtime_action_allowed": False,
            "reused_existing_success": False,
        }
    )
    write_json(out_dir / "job_summary.json", result | {"cmd": job["cmd"]})
    return result


def run_jobs_for_gpu(gpu: str, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for job in jobs:
        results.append(run_job(job, gpu))
        print(json.dumps(results[-1], ensure_ascii=False, sort_keys=True), flush=True)
    return results


def compare_job(result: dict[str, Any]) -> dict[str, Any]:
    trace_txt = Path(result["trace_output_txt"])
    baseline_txt = Path(result["output_txt"])
    row = dict(result)
    row["trace_output_exists"] = trace_txt.is_file()
    row["baseline_output_exists"] = baseline_txt.is_file()
    row["trace_pose_sha256"] = sha256_file(trace_txt) if trace_txt.is_file() else ""
    row["baseline_pose_sha256"] = sha256_file(baseline_txt) if baseline_txt.is_file() else ""
    row["pose_sha_equal"] = bool(row["trace_pose_sha256"] and row["trace_pose_sha256"] == row["baseline_pose_sha256"])
    row["pose_max_abs_diff"] = pose_max_abs_diff(trace_txt, baseline_txt) if trace_txt.is_file() and baseline_txt.is_file() else ""
    row["pose_numeric_equal"] = row["pose_max_abs_diff"] == 0.0
    row["parity_pass"] = int(row.get("returncode", 1)) == 0 and row["pose_sha_equal"] and row["pose_numeric_equal"]
    row["failure_reason"] = "" if int(row.get("returncode", 1)) == 0 else failure_reason(Path(str(row.get("run_log", ""))))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="Reuse an already materialized no-trace baseline root for comparison only; do not launch jobs.",
    )
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--gpus", default="0")
    parser.add_argument(
        "--reuse-existing-success",
        action="store_true",
        help="Reuse existing successful baseline outputs in output-root and rerun only missing/failed jobs.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not launch jobs; rebuild pose_sha_parity_rows.csv and summary.json from existing job summaries/outputs.",
    )
    args = parser.parse_args()

    selected = [case.strip() for case in args.case_ids.split(",") if case.strip()]
    summaries = sorted(args.trace_root.glob("*/READ_NO_ACTION/job_summary.json"))
    if selected:
        summaries = [p for p in summaries if p.parents[1].name in selected]
    jobs = [build_job(path, args.output_root) for path in summaries]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "job_manifest.csv", [{k: v for k, v in job.items() if k != "cmd"} for job in jobs])

    results: list[dict[str, Any]] = []
    pending_jobs: list[dict[str, Any]] = []
    if args.baseline_root is not None:
        results = [external_baseline_result(job, args.baseline_root) for job in jobs]
    elif args.summarize_only:
        results = [existing_result(job) or missing_existing_result(job) for job in jobs]
    else:
        for job in jobs:
            reused = existing_result(job) if args.reuse_existing_success else None
            if reused and int(reused.get("returncode", 1)) == 0:
                results.append(reused)
                print(json.dumps(reused, ensure_ascii=False, sort_keys=True), flush=True)
            else:
                pending_jobs.append(job)

        grouped: dict[str, list[dict[str, Any]]] = {gpu: [] for gpu in gpus}
        for idx, job in enumerate(pending_jobs):
            grouped[gpus[idx % len(gpus)]].append(job)

        with ThreadPoolExecutor(max_workers=max(1, len(gpus))) as executor:
            future_to_gpu = {
                executor.submit(run_jobs_for_gpu, gpu, gpu_jobs): gpu
                for gpu, gpu_jobs in grouped.items()
                if gpu_jobs
            }
            for future in as_completed(future_to_gpu):
                results.extend(future.result())
    results = sorted(results, key=lambda row: row["case_id"])
    rows = [compare_job(row) for row in results]
    write_csv(args.output_root / "pose_sha_parity_rows.csv", rows)
    write_csv(args.output_root / "job_results.csv", results)

    pass_count = sum(1 for row in rows if row["parity_pass"])
    summary = {
        "schema": "acl2_v104_no_trace_pose_sha_parity_summary_v1",
        "trace_root": str(args.trace_root),
        "output_root": str(args.output_root),
        "selected_case_count": len(jobs),
        "completed_job_count": len(results),
        "failed_job_count": sum(1 for row in results if int(row.get("returncode", 1)) != 0),
        "pose_sha_equal_case_count": sum(1 for row in rows if row["pose_sha_equal"]),
        "pose_numeric_equal_case_count": sum(1 for row in rows if row["pose_numeric_equal"]),
        "parity_pass_case_count": pass_count,
        "parity_pass": pass_count == len(jobs) and len(jobs) > 0,
        "reused_existing_success_case_count": sum(1 for row in results if row.get("reused_existing_success")),
        "failure_reason_counts": dict(Counter(row["failure_reason"] for row in rows if row["failure_reason"])),
        "runtime_action_allowed": False,
        "rows": str(args.output_root / "pose_sha_parity_rows.csv"),
    }
    if args.baseline_root is not None:
        summary["external_baseline_root"] = str(args.baseline_root)
        summary["reused_external_baseline_success_case_count"] = sum(
            1 for row in results if row.get("reused_external_baseline_success")
        )
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
