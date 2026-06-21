#!/usr/bin/env python3
"""Run ACL2 v74-TF mid/tail semantic-anchor context boost smoke jobs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_CHUNKS = [30, 6, 31, 19]
DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v74tf_training_free_semantic_memory_control/"
    "phase4_extra_nA_context_anchor_boost_midtail_rho010_top4"
)
DEFAULT_INPUT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_STAGE_C_CACHE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"

CASES: Dict[str, Dict[str, Any]] = {
    "native_no_boost": {"enable_context_source_skip": "0", "semantic_anchor_mode": "semantic"},
    "candidate": {"enable_context_source_skip": "1", "semantic_anchor_mode": "semantic"},
    "random_same_mass": {"enable_context_source_skip": "1", "semantic_anchor_mode": "random_same_mass"},
    "shuffled_semantic": {"enable_context_source_skip": "1", "semantic_anchor_mode": "shuffled_semantic"},
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _window(args: argparse.Namespace, chunk: int) -> Dict[str, int]:
    stride = int(args.chunk_size) - int(args.chunk_overlap)
    start = int(chunk) * stride
    end = start + int(args.chunk_size)
    return {
        "start_frame": int(start),
        "end_frame": int(end),
        "target_start_frame": int(start),
        "target_end_frame": int(end),
    }


def _build_command(args: argparse.Namespace, *, chunk: int, case: str, out_dir: Path) -> List[str]:
    cfg = CASES[case]
    window = _window(args, int(chunk))
    return [
        str(args.conda),
        "run",
        "--no-capture-output",
        "-n",
        str(args.conda_env),
        "python",
        "run_pipeline_abc_v2.py",
        "--input",
        str(args.input),
        "--output_video",
        "",
        "--output_txt",
        str(out_dir / "01.txt"),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--chunk_size",
        str(args.chunk_size),
        "--chunk_overlap",
        str(args.chunk_overlap),
        "--start_frame",
        str(window["start_frame"]),
        "--end_frame",
        str(window["end_frame"]),
        "--global_chunk_offset",
        str(chunk),
        "--device",
        "cuda",
        "--hybrid_memory_mode",
        "read_path_only",
        "--hmc_commit_mode",
        "controlled",
        "--semantic_prior_mode",
        "spg_v2",
        "--semantic_memory_paths",
        str(args.semantic_memory_paths),
        "--stage_c_cache_mode",
        "read",
        "--stage_c_cache_dir",
        str(args.stage_c_cache_dir),
        "--stage_c_cache_require_hit",
        "1",
        "--enable_frame_read_control",
        "0",
        "--read_path",
        "none",
        "--read_cue_source",
        "dyn",
        "--read_overlap_frames",
        str(args.chunk_overlap),
        "--enable_context_source_skip",
        str(cfg["enable_context_source_skip"]),
        "--context_source_skip_impl",
        "bias_boost",
        "--context_source_skip_scope",
        "frame",
        "--context_source_skip_mode",
        "boost",
        "--context_source_skip_mask",
        "semantic_anchor",
        "--context_source_skip_frame_region",
        str(args.frame_region),
        "--context_source_skip_layer_mode",
        str(args.context_layer_mode),
        "--context_source_skip_soft_rho",
        str(args.boost_rho if case != "native_no_boost" else 0.0),
        "--semantic_anchor_mode",
        str(cfg["semantic_anchor_mode"]),
        "--semantic_anchor_target_ratio",
        str(args.semantic_anchor_target_ratio),
        "--semantic_anchor_min_ratio",
        str(args.semantic_anchor_min_ratio),
        "--semantic_anchor_max_ratio",
        str(args.semantic_anchor_max_ratio),
        "--semantic_anchor_min_score",
        str(args.semantic_anchor_min_score),
        "--semantic_anchor_missing_trust_policy",
        str(args.semantic_anchor_missing_trust_policy),
        "--semantic_anchor_value_fallback",
        str(args.semantic_anchor_value_fallback),
        "--semantic_anchor_grid_rows",
        str(args.semantic_anchor_grid_rows),
        "--semantic_anchor_grid_cols",
        str(args.semantic_anchor_grid_cols),
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
    ]


def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    if str(job.get("cuda_alloc_conf") or "").strip():
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(job["cuda_alloc_conf"]).strip()
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.log"
    start_t = time.time()
    with run_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            job["cmd"],
            cwd=job["workdir"],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    end_t = time.time()
    job.update(
        {
            "returncode": int(proc.returncode),
            "duration_sec": float(end_t - start_t),
            "run_log": str(run_log),
            "trajectory": str(out_dir / "01.txt"),
            "hmc_state_hash": str(out_dir / "hmc_state_hash.jsonl"),
        }
    )
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunks", default=",".join(str(c) for c in DEFAULT_CHUNKS))
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage-c-cache-dir", type=Path, default=DEFAULT_STAGE_C_CACHE)
    parser.add_argument("--semantic-memory-paths", default="frame,global")
    parser.add_argument("--cuda-alloc-conf", default=DEFAULT_CUDA_ALLOC_CONF)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--frame-region", choices=("all", "head", "mid_tail", "tail"), default="mid_tail")
    parser.add_argument("--context-layer-mode", choices=("all", "first", "last", "early", "single"), default="early")
    parser.add_argument("--boost-rho", type=float, default=0.10)
    parser.add_argument("--semantic-anchor-target-ratio", type=float, default=0.12)
    parser.add_argument("--semantic-anchor-min-ratio", type=float, default=0.03)
    parser.add_argument("--semantic-anchor-max-ratio", type=float, default=0.30)
    parser.add_argument("--semantic-anchor-min-score", type=float, default=0.02)
    parser.add_argument("--semantic-anchor-missing-trust-policy", choices=("zero", "neutral"), default="neutral")
    parser.add_argument("--semantic-anchor-value-fallback", choices=("off", "semantic_value"), default="semantic_value")
    parser.add_argument("--semantic-anchor-grid-rows", type=int, default=4)
    parser.add_argument("--semantic-anchor-grid-cols", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    cases = [case.strip() for case in str(args.cases).split(",") if case.strip()]
    unknown = [case for case in cases if case not in CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(CASES)}")

    jobs: List[Dict[str, Any]] = []
    gpu_cursor = 0
    for chunk in chunks:
        window = _window(args, int(chunk))
        for case in cases:
            out_dir = args.output_root / f"chunk{chunk:02d}" / case
            cmd = _build_command(args, chunk=int(chunk), case=case, out_dir=out_dir)
            skipped = bool(args.skip_existing and (out_dir / "01.txt").exists() and (out_dir / "hmc_state_hash.jsonl").exists())
            jobs.append(
                {
                    "chunk": int(chunk),
                    "case": case,
                    "gpu": int(gpus[gpu_cursor % len(gpus)]),
                    "out_dir": str(out_dir),
                    "cmd": cmd,
                    "cmd_shell": shlex.join(cmd),
                    "workdir": str(args.workdir),
                    "semantic_anchor_mode_effective": CASES[case]["semantic_anchor_mode"],
                    "context_source_skip_enabled_effective": CASES[case]["enable_context_source_skip"],
                    "frame_region_effective": str(args.frame_region),
                    "boost_rho_effective": float(args.boost_rho if case != "native_no_boost" else 0.0),
                    "start_frame": int(window["start_frame"]),
                    "end_frame": int(window["end_frame"]),
                    "target_start_frame": int(window["target_start_frame"]),
                    "target_end_frame": int(window["target_end_frame"]),
                    "cuda_alloc_conf": str(args.cuda_alloc_conf),
                    "skipped": skipped,
                    "returncode": 0 if skipped else None,
                }
            )
            gpu_cursor += 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "v74tf_context_anchor_boost_manifest.json"
    manifest: Dict[str, Any] = {"args": vars(args), "jobs": jobs}
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"planned_jobs={len(jobs)} manifest={manifest_path}")
    if args.dry_run:
        return

    run_jobs = [job for job in jobs if not job["skipped"]]
    completed: List[Dict[str, Any]] = [job for job in jobs if job["skipped"]]
    completed_by_key = {(int(job["chunk"]), str(job["case"])): job for job in completed}
    completed_lock = threading.Lock()
    jobs_by_gpu: Dict[int, List[Dict[str, Any]]] = {int(gpu): [] for gpu in gpus}
    for job in run_jobs:
        jobs_by_gpu[int(job["gpu"])].append(job)

    def _run_gpu_queue(gpu: int, queue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        gpu_results: List[Dict[str, Any]] = []
        for job in queue:
            result = _run_job(job)
            gpu_results.append(result)
            with completed_lock:
                completed.append(result)
                completed_by_key[(int(result["chunk"]), str(result["case"]))] = result
                manifest["jobs"] = [
                    completed_by_key.get((int(job["chunk"]), str(job["case"])), job)
                    for job in jobs
                ]
                manifest["completed_count"] = len(completed_by_key)
                manifest_path.write_text(
                    json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(
                "done chunk={chunk} case={case} gpu={gpu} returncode={returncode} duration_sec={duration_sec:.1f}".format(
                    **result
                ),
                flush=True,
            )
        return gpu_results

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(_run_gpu_queue, gpu, queue) for gpu, queue in jobs_by_gpu.items() if queue]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
