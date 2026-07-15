#!/usr/bin/env python3
"""Run ACL2 v119 LB-LR LingBot local-read workers with a small GPU scheduler."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_RESULT_ROOT_ENV = os.environ.get("ACL2_V119_LBLR_RESULT_ROOT", "").strip()
if _RESULT_ROOT_ENV:
    _result_root_path = Path(_RESULT_ROOT_ENV).expanduser()
    RESULT_ROOT = _result_root_path if _result_root_path.is_absolute() else ROOT / _result_root_path
else:
    RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
DATASET = "kitti_v105_00_01_02_05"
LOGIT_STAGE_SLUG = os.environ.get("ACL2_V119_LBLR_LOGIT_STAGE_SLUG", "stage2_lblr_local_read_logit").strip() or "stage2_lblr_local_read_logit"
VALUE_STAGE_SLUG = os.environ.get("ACL2_V119_LBLR_VALUE_STAGE_SLUG", "stage2_lblr_local_read_value").strip() or "stage2_lblr_local_read_value"
RUNTIME_STAGE_SLUG = os.environ.get("ACL2_V119_LBLR_RUNTIME_STAGE_SLUG", "stage2_lblr_runtime_full_thread8").strip() or "stage2_lblr_runtime_full_thread8"
STAGE_BY_FORM = {
    "logit": RESULT_ROOT / LOGIT_STAGE_SLUG,
    "value": RESULT_ROOT / VALUE_STAGE_SLUG,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_one(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one match for {root / pattern}, got {len(matches)}")
    return matches[0]


def jobs_for_form(form: str) -> list[dict[str, Any]]:
    stage = STAGE_BY_FORM[form]
    manifest = find_one("stage4_*_lingbot_ar_anchor_read_manifest.json", stage / "summary")
    payload = read_json(manifest)
    config = find_one("kitti_lingbot_*local_read_full_reuse_v105gt.yaml", stage / "configs")
    jobs = []
    for method in payload.get("concrete_methods", []):
        match = re.search(r"_seq(\d\d)$", str(method))
        if not match:
            raise RuntimeError(f"method name lacks seq suffix: {method}")
        seq = match.group(1)
        jobs.append(
            {
                "form": form,
                "stage": stage,
                "config": config,
                "method": str(method),
                "seq": seq,
            }
        )
    return jobs


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_scheduler(args: argparse.Namespace) -> int:
    forms = [part.strip() for part in args.forms.split(",") if part.strip()]
    gpus = [part.strip() for part in args.gpus.split(",") if part.strip()]
    if not gpus:
        raise RuntimeError("at least one GPU id is required")
    jobs: list[dict[str, Any]] = []
    for form in forms:
        if form not in STAGE_BY_FORM:
            raise RuntimeError(f"unknown form {form!r}; expected one of {sorted(STAGE_BY_FORM)}")
        jobs.extend(jobs_for_form(form))

    max_workers = min(max(1, int(args.max_workers)), len(gpus))
    runtime_root = RESULT_ROOT / RUNTIME_STAGE_SLUG
    logs = runtime_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = logs / f"lblr_worker_batch_status_{args.run_label}.json"
    conda = Path(args.conda_bin)
    run_worker = ROOT / "third_party/lingbot-map/benchmark/run_worker.py"
    pythonpath = os.pathsep.join(
        [
            str(ROOT / "third_party/lingbot-map"),
            str(ROOT / "third_party/lingbot-map/benchmark"),
            os.environ.get("PYTHONPATH", ""),
        ]
    )

    pending = list(jobs)
    running: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    status: dict[str, Any] = {
        "run_label": args.run_label,
        "forms": forms,
        "gpus": gpus,
        "max_workers": max_workers,
        "job_count": len(jobs),
        "pending_count": len(pending),
        "running": [],
        "completed": [],
        "all_rc_zero": False,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_status(status_path, status)

    def launch(job: dict[str, Any], gpu: str) -> dict[str, Any]:
        stage: Path = job["stage"]
        stage_runtime = stage / "runtime_full_thread8"
        action_dir = stage_runtime / "action_traces"
        log_dir = stage_runtime / "logs"
        action_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        method = job["method"]
        seq = job["seq"]
        trace_file = action_dir / f"{method}_seq{seq}.jsonl"
        log_path = log_dir / f"run_{method}_seq{seq}_gpu{gpu}_{args.run_label}.log"
        rc_path = log_dir / f"run_{method}_seq{seq}_gpu{gpu}_{args.run_label}.rc"
        if args.fresh:
            for path in (trace_file, log_path, rc_path):
                if path.exists():
                    path.unlink()
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": pythonpath,
                "ACL2_V105_STAGE4_ACTION_FILE": str(trace_file),
                "ACL2_V105_STAGE4_ACTION_LABEL": method,
                "ACL2_V105_GCA_TRACE_DATASET": DATASET,
                "ACL2_V105_GCA_TRACE_SEQ": seq,
                "ACL2_V105_GCA_TRACE_METHOD": method,
                "ACL2_V112_A2_ACTION_GLOBAL_IDXS": args.action_global_idxs,
                "ACL2_V112_A2_ACTION_MAX_ROWS": str(args.action_max_rows),
                "ACL2_V108_STAGE4_TRAJECTORY_ONLY_OUTPUT": "1",
                "OMP_NUM_THREADS": str(args.cpu_threads),
                "MKL_NUM_THREADS": str(args.cpu_threads),
                "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
                "NUMEXPR_NUM_THREADS": str(args.cpu_threads),
                "VECLIB_MAXIMUM_THREADS": str(args.cpu_threads),
            }
        )
        cmd = [
            str(conda),
            "run",
            "-n",
            args.conda_env,
            "--no-capture-output",
            "python",
            str(run_worker),
            "--config",
            str(job["config"]),
            "--method",
            method,
            "--dataset",
            DATASET,
            "--scene",
            seq,
            "--force",
        ]
        handle = log_path.open("w", encoding="utf-8")
        handle.write("# command\n" + " ".join(cmd) + "\n")
        handle.write(
            "# action_env\n"
            f"ACL2_V105_STAGE4_ACTION_FILE={trace_file}\n"
            f"ACL2_V105_STAGE4_ACTION_LABEL={method}\n"
            f"ACL2_V112_A2_ACTION_GLOBAL_IDXS={args.action_global_idxs}\n"
            f"ACL2_V112_A2_ACTION_MAX_ROWS={args.action_max_rows}\n"
            "ACL2_V108_STAGE4_TRAJECTORY_ONLY_OUTPUT=1\n"
            f"OMP_NUM_THREADS={args.cpu_threads}\n"
            f"MKL_NUM_THREADS={args.cpu_threads}\n"
            f"OPENBLAS_NUM_THREADS={args.cpu_threads}\n"
            f"NUMEXPR_NUM_THREADS={args.cpu_threads}\n"
        )
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        return {
            **job,
            "gpu": gpu,
            "pid": proc.pid,
            "proc": proc,
            "log_handle": handle,
            "log": log_path,
            "rc": rc_path,
            "trace": trace_file,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    try:
        while pending or running:
            used = {item["gpu"] for item in running}
            free_gpus = [gpu for gpu in gpus if gpu not in used]
            while pending and free_gpus and len(running) < max_workers:
                running.append(launch(pending.pop(0), free_gpus.pop(0)))

            still_running = []
            for item in running:
                proc = item["proc"]
                rc = proc.poll()
                if rc is None:
                    still_running.append(item)
                    continue
                item["log_handle"].close()
                item["rc"].write_text(str(rc) + "\n", encoding="utf-8")
                trace_lines = 0
                if item["trace"].exists():
                    with item["trace"].open(encoding="utf-8", errors="replace") as handle:
                        trace_lines = sum(1 for _ in handle)
                completed.append(
                    {
                        "form": item["form"],
                        "method": item["method"],
                        "seq": item["seq"],
                        "gpu": item["gpu"],
                        "pid": item["pid"],
                        "returncode": rc,
                        "log": str(item["log"].relative_to(ROOT)),
                        "rc_file": str(item["rc"].relative_to(ROOT)),
                        "action_trace": str(item["trace"].relative_to(ROOT)),
                        "action_trace_lines": trace_lines,
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                )
            running = still_running
            status.update(
                {
                    "pending_count": len(pending),
                    "running": [
                        {
                            "form": item["form"],
                            "method": item["method"],
                            "seq": item["seq"],
                            "gpu": item["gpu"],
                            "pid": item["pid"],
                            "log": str(item["log"].relative_to(ROOT)),
                        }
                        for item in running
                    ],
                    "completed": completed,
                    "completed_count": len(completed),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            )
            write_status(status_path, status)
            if pending or running:
                time.sleep(args.poll_seconds)
    finally:
        for item in running:
            proc = item.get("proc")
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            try:
                item["log_handle"].close()
            except Exception:
                pass

    all_rc_zero = bool(completed) and len(completed) == len(jobs) and all(item["returncode"] == 0 for item in completed)
    status.update(
        {
            "all_rc_zero": all_rc_zero,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "complete" if all_rc_zero else "complete_with_failures",
        }
    )
    write_status(status_path, status)
    print(json.dumps({"status": status["status"], "status_path": str(status_path.relative_to(ROOT)), "all_rc_zero": all_rc_zero}, indent=2))
    return 0 if all_rc_zero else 1


def detach(args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--run"]
    for key in ("forms", "gpus", "max_workers", "run_label", "conda_bin", "conda_env", "action_global_idxs", "action_max_rows", "poll_seconds", "cpu_threads"):
        cmd.extend([f"--{key.replace('_', '-')}", str(getattr(args, key))])
    if args.fresh:
        cmd.append("--fresh")
    logs = RESULT_ROOT / RUNTIME_STAGE_SLUG / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = (logs / f"lblr_worker_batch_scheduler_{args.run_label}.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True, text=True)
    pid_manifest = logs / f"detached_lblr_worker_batch_{args.run_label}.json"
    payload = {
        "pid": proc.pid,
        "cmd": cmd,
        "forms": args.forms,
        "gpus": args.gpus,
        "max_workers": args.max_workers,
        "run_label": args.run_label,
        "scheduler_log": str((logs / f"lblr_worker_batch_scheduler_{args.run_label}.log").relative_to(ROOT)),
        "status_path": str((logs / f"lblr_worker_batch_status_{args.run_label}.json").relative_to(ROOT)),
    }
    pid_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pid_manifest": str(pid_manifest.relative_to(ROOT)), **payload}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forms", default="logit,value")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--run-label", default="rerun1")
    parser.add_argument("--conda-bin", default="/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--action-global-idxs", default="0")
    parser.add_argument("--action-max-rows", type=int, default=80000)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.detach:
        return detach(args)
    return run_scheduler(args)


if __name__ == "__main__":
    raise SystemExit(main())
