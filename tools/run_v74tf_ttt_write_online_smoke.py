#!/usr/bin/env python3
"""Run v74-TF RADIO TTT-write online local-window smoke jobs."""

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


DEFAULT_CHUNKS = [6, 19, 30, 31]
DEFAULT_INPUT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_STAGE_C_CACHE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_RADIO_SIDECAR = Path("results/kitti_preprocess/01/radio_sidecar_chunks_r5_overlap")
DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v74tf_training_free_semantic_memory_control/"
    "phase5_harmful_no_persistent_ttt_dynamic_lowstable_top4"
)
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


TTT_CASES: Dict[str, Dict[str, Any]] = {
    "native_no_ttt_radio": {
        "enable_v70_radio_ttt_write_prior": "0",
        "v70_radio_ttt_mode": "dynamic_lowstable_no_persistent",
        "v70_radio_ttt_control": "none",
    },
    "candidate": {
        "enable_v70_radio_ttt_write_prior": "1",
        "v70_radio_ttt_mode": "dynamic_lowstable_no_persistent",
        "v70_radio_ttt_control": "none",
    },
    "geometry_only": {
        "enable_v70_radio_ttt_write_prior": "1",
        "v70_radio_ttt_mode": "dynamic_lowstable_no_persistent",
        "v70_radio_ttt_control": "geometry_only",
    },
    "spatial_shuffle": {
        "enable_v70_radio_ttt_write_prior": "1",
        "v70_radio_ttt_mode": "dynamic_lowstable_no_persistent",
        "v70_radio_ttt_control": "spatial_shuffle",
    },
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


def _context_window(args: argparse.Namespace, chunk: int) -> Dict[str, int]:
    stride = int(args.chunk_size) - int(args.chunk_overlap)
    context_chunks = max(int(args.context_chunks), 1)
    first_chunk = max(int(chunk) - context_chunks + 1, 0)
    actual_chunks = int(chunk) - first_chunk + 1
    start = first_chunk * stride
    end = start + int(args.chunk_size) + (actual_chunks - 1) * stride
    target_start = int(chunk) * stride
    target_end = target_start + int(args.chunk_size)
    return {
        "context_start_chunk": int(first_chunk),
        "context_chunks": int(actual_chunks),
        "start_frame": int(start),
        "end_frame": int(end),
        "target_start_frame": int(target_start),
        "target_end_frame": int(target_end),
    }


def _build_command(args: argparse.Namespace, *, chunk: int, case: str, out_dir: Path) -> List[str]:
    case_cfg = TTT_CASES[case]
    window = _context_window(args, int(chunk))
    return [
        str(args.conda),
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
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
        str(window["context_start_chunk"]),
        "--device",
        "cuda",
        "--hybrid_memory_mode",
        "ttt_write_only",
        "--hmc_commit_mode",
        "controlled",
        "--semantic_prior_mode",
        "spg_v2",
        "--stage_c_cache_mode",
        "read",
        "--stage_c_cache_dir",
        str(args.stage_c_cache_dir),
        "--stage_c_cache_require_hit",
        "1",
        "--read_path",
        "none",
        "--v70_radio_sidecar_dir",
        str(args.radio_sidecar_dir),
        "--enable_v70_radio_ttt_write_prior",
        str(case_cfg["enable_v70_radio_ttt_write_prior"]),
        "--v70_radio_ttt_mode",
        str(case_cfg["v70_radio_ttt_mode"]),
        "--v70_radio_ttt_control",
        str(case_cfg["v70_radio_ttt_control"]),
        "--v70_radio_ttt_suppress",
        str(args.v70_radio_ttt_suppress),
        "--v70_radio_ttt_min_confidence",
        str(args.v70_radio_ttt_min_confidence),
        "--v70_radio_ttt_min_stability",
        str(args.v70_radio_ttt_min_stability),
        "--v70_radio_ttt_min_interior",
        str(args.v70_radio_ttt_min_interior),
        "--v70_radio_ttt_max_activity_risk",
        str(args.v70_radio_ttt_max_activity_risk),
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
    job.update(
        {
            "returncode": int(proc.returncode),
            "duration_sec": float(time.time() - start_t),
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
    parser.add_argument("--cases", default=",".join(TTT_CASES))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage-c-cache-dir", type=Path, default=DEFAULT_STAGE_C_CACHE)
    parser.add_argument("--radio-sidecar-dir", type=Path, default=DEFAULT_RADIO_SIDECAR)
    parser.add_argument("--cuda-alloc-conf", default=DEFAULT_CUDA_ALLOC_CONF)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--context-chunks", type=int, default=2)
    parser.add_argument("--v70-radio-ttt-suppress", type=float, default=0.50)
    parser.add_argument("--v70-radio-ttt-min-confidence", type=float, default=0.45)
    parser.add_argument("--v70-radio-ttt-min-stability", type=float, default=0.35)
    parser.add_argument("--v70-radio-ttt-min-interior", type=float, default=0.25)
    parser.add_argument("--v70-radio-ttt-max-activity-risk", type=float, default=0.85)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    cases = [case.strip() for case in str(args.cases).split(",") if case.strip()]
    unknown = [case for case in cases if case not in TTT_CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(TTT_CASES)}")

    jobs: List[Dict[str, Any]] = []
    gpu_cursor = 0
    for chunk in chunks:
        window = _context_window(args, int(chunk))
        for case in cases:
            out_dir = args.output_root / f"chunk{int(chunk):02d}" / case
            cmd = _build_command(args, chunk=int(chunk), case=case, out_dir=out_dir)
            skipped = bool(args.skip_existing and (out_dir / "01.txt").exists() and (out_dir / "hmc_state_hash.jsonl").exists())
            case_cfg = TTT_CASES[case]
            jobs.append(
                {
                    "chunk": int(chunk),
                    "context_start_chunk": int(window["context_start_chunk"]),
                    "context_chunks": int(window["context_chunks"]),
                    "start_frame": int(window["start_frame"]),
                    "end_frame": int(window["end_frame"]),
                    "target_start_frame": int(window["target_start_frame"]),
                    "target_end_frame": int(window["target_end_frame"]),
                    "case": case,
                    "gpu": int(gpus[gpu_cursor % len(gpus)]),
                    "out_dir": str(out_dir),
                    "cmd": cmd,
                    "cmd_shell": shlex.join(cmd),
                    "workdir": str(args.workdir),
                    "enable_v70_radio_ttt_write_prior_effective": case_cfg["enable_v70_radio_ttt_write_prior"],
                    "v70_radio_ttt_mode_effective": case_cfg["v70_radio_ttt_mode"],
                    "v70_radio_ttt_control_effective": case_cfg["v70_radio_ttt_control"],
                    "cuda_alloc_conf": str(args.cuda_alloc_conf),
                    "skipped": skipped,
                    "returncode": 0 if skipped else None,
                }
            )
            gpu_cursor += 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "v74tf_ttt_write_online_manifest.json"
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
                manifest["jobs"] = [completed_by_key.get((int(job["chunk"]), str(job["case"])), job) for job in jobs]
                manifest["completed_count"] = len(completed_by_key)
                manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(
                "done",
                f"chunk={result['chunk']}",
                f"case={result['case']}",
                f"gpu={result['gpu']}",
                f"returncode={result['returncode']}",
                f"duration_sec={result['duration_sec']:.1f}",
                flush=True,
            )
        return gpu_results

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(_run_gpu_queue, int(gpu), queue) for gpu, queue in jobs_by_gpu.items() if queue]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    by_key = {(job["chunk"], job["case"]): job for job in completed}
    ordered = [by_key.get((job["chunk"], job["case"]), job) for job in jobs]
    manifest["jobs"] = ordered
    manifest["completed_count"] = len(completed)
    manifest["failed_jobs"] = [job for job in ordered if int(job.get("returncode") or 0) != 0]
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if manifest["failed_jobs"]:
        raise SystemExit(f"failed_jobs={len(manifest['failed_jobs'])}; see {manifest_path}")


if __name__ == "__main__":
    main()
