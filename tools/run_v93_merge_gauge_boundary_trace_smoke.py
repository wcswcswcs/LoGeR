#!/usr/bin/env python3
"""Replay v92 native no-op trace smokes into a v93 instrumented trace root."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v93_semantic_object_identity_utils import ROOT, V92_ROOT


DEFAULT_V92_NOOP = V92_ROOT / "phase2_boundary_trace_ledger/noop_trace_smoke"
DEFAULT_OUT = ROOT / "phase3_merge_gauge_trace_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v92-noop-root", type=Path, default=DEFAULT_V92_NOOP)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--limit", type=int, default=4)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def remap_cmd(cmd: list[str], old_root: Path, new_root: Path) -> list[str]:
    old = old_root.as_posix()
    new = new_root.as_posix()
    return [part.replace(old, new) for part in cmd]


def job_name(path: Path) -> str:
    for part in path.parts:
        if part.startswith("seq") and "_native_noop" in part:
            return part
    return path.parent.name


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifests = sorted(args.v92_noop_root.glob("seq*_native_noop/phase9_swa_cache_value_run_manifest.json"))
    if args.limit > 0:
        manifests = manifests[: args.limit]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    launched = []
    for idx, manifest_path in enumerate(manifests):
        manifest = load_json(manifest_path)
        jobs = manifest.get("jobs", [])
        if not jobs:
            continue
        job = jobs[0]
        old_job_root = manifest_path.parent
        new_job_root = args.out_root / old_job_root.name
        cmd = remap_cmd(list(job["cmd"]), old_job_root, new_job_root)
        out_dir = Path(str(job.get("out_dir", "")).replace(old_job_root.as_posix(), new_job_root.as_posix()))
        out_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        gpu = gpus[idx % len(gpus)] if gpus else ""
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(manifest.get("args", {}).get("cuda_alloc_conf", "expandable_segments:True"))
        log_path = out_dir / "v93_outer_run.log"
        start = time.time()
        handle = log_path.open("w", encoding="utf-8")
        handle.write(f"$ {' '.join(cmd)}\n")
        handle.write(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n")
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=Path.cwd(), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        launched.append(
            {
                "name": old_job_root.name,
                "gpu": gpu,
                "proc": proc,
                "handle": handle,
                "start": start,
                "cmd": cmd,
                "out_dir": out_dir,
                "log_path": log_path,
                "old_manifest": manifest_path,
            }
        )
    rows = []
    failed = []
    for item in launched:
        returncode = item["proc"].wait()
        duration = time.time() - item["start"]
        item["handle"].write(f"\nreturncode={returncode}\nduration_sec={duration:.3f}\n")
        item["handle"].close()
        row = {
            "name": item["name"],
            "gpu": item["gpu"],
            "returncode": returncode,
            "duration_sec": duration,
            "out_dir": str(item["out_dir"]),
            "log_path": str(item["log_path"]),
            "merge_state_trace": str(item["out_dir"] / "merge_state_trace.jsonl"),
            "hmc_state_hash": str(item["out_dir"] / "hmc_state_hash.jsonl"),
            "old_manifest": str(item["old_manifest"]),
            "cmd": item["cmd"],
        }
        rows.append(row)
        if returncode != 0:
            failed.append(row)
    summary = {
        "phase": "Phase3_v93_instrumented_native_noop_trace_smoke",
        "out_root": str(args.out_root),
        "job_count": len(rows),
        "completed_count": sum(1 for row in rows if row["returncode"] == 0),
        "failed_count": len(failed),
        "all_completed": len(failed) == 0 and len(rows) > 0,
        "jobs": rows,
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }
    (args.out_root / "v93_merge_gauge_trace_smoke_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"job_count={summary['job_count']}")
    print(f"completed_count={summary['completed_count']}")
    print(f"failed_count={summary['failed_count']}")
    print(f"out_root={args.out_root}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
