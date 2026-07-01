#!/usr/bin/env python3
"""Run v93 Phase7 SWA secondary query/pair route-control smokes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v93_semantic_object_identity_utils import ROOT  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT / "phase7_swa_secondary_carrier/route_audit"
DEFAULT_MASK_CSV = ROOT / "phase7_swa_secondary_carrier/route_masks/v93_swa_secondary_route_mask_positions.csv"
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DATA_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset")
SEQ_SPECS = {
    "00": {"chunk": 2},
    "01": {"chunk": 9},
    "02": {"chunk": 7},
    "05": {"chunk": 2},
}
VARIANTS = {
    "actual": "v93_object_policy_pair_mask",
    "object": "v93_object_shuffle_pair_mask",
    "component": "v93_component_shuffle_pair_mask",
    "semantic": "v93_semantic_shuffle_pair_mask",
    "regime": "v93_regime_shuffle_pair_mask",
    "random": "v93_same_count_random_pair_mask",
    "geometry": "v93_geometry_only_pair_mask",
}
ROUTES = {
    "query": "P9_52_ATTENTION_BIAS_V92_POLICY_QUERY_MASS_AUDIT_LAST",
    "pair": "P9_54_ATTENTION_BIAS_V92_POLICY_PAIR_MASS_AUDIT_LAST",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mask-csv", type=Path, default=DEFAULT_MASK_CSV)
    parser.add_argument("--variants", default="actual,object,component,semantic,regime,random,geometry")
    parser.add_argument("--routes", default="query,pair")
    parser.add_argument("--seqs", default="00,01,02,05")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parts(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _build_job(args: argparse.Namespace, *, seq: str, variant_key: str, route: str, gpu: int) -> dict[str, Any]:
    variant = VARIANTS[variant_key]
    chunk = int(SEQ_SPECS[seq]["chunk"])
    case = ROUTES[route]
    out_root = args.output_root / f"seq{seq}_chunk{chunk:02d}_v93_{variant_key}_{route}"
    cmd = [
        str(args.conda),
        "run",
        "-n",
        str(args.conda_env),
        "python",
        "tools/run_v78_phase9_swa_cache_value_carryover.py",
        "--output-root",
        str(out_root),
        "--chunks",
        str(chunk),
        "--cases",
        case,
        "--gpus",
        str(gpu),
        "--input",
        str(DATA_ROOT / "sequences" / seq / "image_2"),
        "--gt",
        str(DATA_ROOT / "poses" / f"{seq}.txt"),
        "--stage-c-cache-dir",
        f"results/kitti_preprocess/{seq}/stage_c_cache_semantic_chunks",
        "--swa-source-layer-mode",
        "last",
        "--swa-overlap-feature-dump-dtype",
        "float16",
        "--swa-overlap-external-mask-csv",
        str(args.mask_csv),
        "--swa-overlap-external-mask-variant",
        variant,
        "--swa-overlap-external-mask-seq",
        seq,
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    return {
        "seq": seq,
        "chunk": chunk,
        "variant_key": variant_key,
        "variant": variant,
        "route": route,
        "case": case,
        "gpu": int(gpu),
        "output_root": str(out_root),
        "cmd": cmd,
        "cmd_shell": shlex.join(cmd),
    }


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.time()
    proc = subprocess.run(
        job["cmd"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    end = time.time()
    out_root = Path(job["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "v93_outer_run.log"
    log_path.write_text(proc.stdout, encoding="utf-8")
    updated = dict(job)
    updated.update(
        {
            "returncode": int(proc.returncode),
            "duration_sec": float(end - start),
            "outer_run_log": str(log_path),
            "metrics_csv": str(out_root / "phase9_swa_cache_value_metrics.csv"),
            "metrics_json": str(out_root / "phase9_swa_cache_value_metrics.json"),
            "manifest_json": str(out_root / "phase9_swa_cache_value_run_manifest.json"),
        }
    )
    return updated


def _job_key(job: dict[str, Any]) -> tuple[str, str, str]:
    return (str(job.get("variant_key", "")), str(job.get("route", "")), str(job.get("seq", "")))


def main() -> None:
    args = parse_args()
    seqs = _parts(args.seqs)
    variant_keys = _parts(args.variants)
    routes = _parts(args.routes)
    unknown_seqs = [seq for seq in seqs if seq not in SEQ_SPECS]
    unknown_variants = [variant for variant in variant_keys if variant not in VARIANTS]
    unknown_routes = [route for route in routes if route not in ROUTES]
    if unknown_seqs:
        raise ValueError(f"unknown seqs: {unknown_seqs}")
    if unknown_variants:
        raise ValueError(f"unknown variants: {unknown_variants}")
    if unknown_routes:
        raise ValueError(f"unknown routes: {unknown_routes}")
    gpus = [int(x) for x in _parts(args.gpus)]
    if not gpus:
        raise ValueError("--gpus must be non-empty")

    jobs: list[dict[str, Any]] = []
    cursor = 0
    for variant_key in variant_keys:
        for route in routes:
            for seq in seqs:
                jobs.append(
                    _build_job(
                        args,
                        seq=seq,
                        variant_key=variant_key,
                        route=route,
                        gpu=gpus[cursor % len(gpus)],
                    )
                )
                cursor += 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "v93_swa_secondary_route_run_manifest.json"
    manifest: dict[str, Any] = {"args": vars(args), "jobs": jobs}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"planned_jobs={len(jobs)} manifest={manifest_path}")
    if args.dry_run:
        return

    results: list[dict[str, Any]] = []
    results_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    max_workers = max(1, min(int(args.max_parallel), len(jobs)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_job, job) for job in jobs]
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            results.append(result)
            results_by_key[_job_key(result)] = result
            print(
                f"[gpu{result['gpu']}] variant={result['variant_key']} route={result['route']} "
                f"seq={result['seq']} returncode={result['returncode']} duration={result['duration_sec']:.1f}s"
            )
            manifest["jobs"] = [results_by_key.get(_job_key(job), job) for job in jobs]
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    failed = [
        {"seq": row["seq"], "variant": row["variant_key"], "route": row["route"], "returncode": row["returncode"]}
        for row in results
        if int(row.get("returncode") or 0) != 0
    ]
    manifest["completed_count"] = int(len(results))
    manifest["failed_jobs"] = failed
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"completed={len(results)} failed={len(failed)} manifest={manifest_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
