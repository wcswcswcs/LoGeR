#!/usr/bin/env python3
"""Run ACL2 v70 RADIO READ online local-window smoke jobs with a GPU queue."""

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


DEFAULT_CHUNKS = [12, 20, 30, 32]
DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v70_geometry_first_semantic_trust/"
    "report_final/phaseR5_radio_read_online_smoke_r3"
)
DEFAULT_INPUT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_STAGE_C_CACHE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_RADIO_SIDECAR = Path("results/kitti_preprocess/01/radio_sidecar_chunks_r5_overlap")
DEFAULT_READ_CUE_SOURCE = "v70.radio.object_interior_floor"
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


READ_CASES: Dict[str, Dict[str, Any]] = {
    "native_no_read": {
        "enable_frame_read_control": "0",
        "read_path": "none",
        "read_cue_source": "dyn",
    },
    "candidate": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "read_cue_source": DEFAULT_READ_CUE_SOURCE,
    },
    "geometry_only": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "read_cue_source": f"{DEFAULT_READ_CUE_SOURCE}.geometry_only",
    },
    "radio_feature_shuffle": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "read_cue_source": f"{DEFAULT_READ_CUE_SOURCE}.radio_feature_shuffle",
    },
    "radio_risk_shuffle": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "read_cue_source": f"{DEFAULT_READ_CUE_SOURCE}.radio_risk_shuffle",
    },
    "same_cue_random": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "read_cue_source": f"{DEFAULT_READ_CUE_SOURCE}.same_cue_distribution_random",
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


def _read_cases_for_cue(read_cue_source: str) -> Dict[str, Dict[str, Any]]:
    cue = str(read_cue_source or DEFAULT_READ_CUE_SOURCE).strip()
    cases = {name: dict(cfg) for name, cfg in READ_CASES.items()}
    cases["candidate"]["read_cue_source"] = cue
    for case, suffix in {
        "geometry_only": ".geometry_only",
        "radio_feature_shuffle": ".radio_feature_shuffle",
        "radio_risk_shuffle": ".radio_risk_shuffle",
        "same_cue_random": ".same_cue_distribution_random",
    }.items():
        cases[case]["read_cue_source"] = f"{cue}{suffix}"
    return cases


def _build_command(args: argparse.Namespace, *, chunk: int, case: str, out_dir: Path) -> List[str]:
    case_cfg = _read_cases_for_cue(args.read_cue_source)[case]
    start = int(chunk) * (int(args.chunk_size) - int(args.chunk_overlap))
    end = start + int(args.chunk_size)
    cmd = [
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
        str(start),
        "--end_frame",
        str(end),
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
        "--stage_c_cache_mode",
        "read",
        "--stage_c_cache_dir",
        str(args.stage_c_cache_dir),
        "--stage_c_cache_require_hit",
        "1",
        "--enable_frame_read_control",
        str(case_cfg["enable_frame_read_control"]),
        "--read_path",
        str(case_cfg["read_path"]),
        "--read_cue_source",
        str(case_cfg["read_cue_source"]),
        "--v70_radio_sidecar_dir",
        str(args.radio_sidecar_dir),
        "--read_layer_mode",
        str(args.read_layer_mode),
    ]
    if case != "native_no_read":
        cmd.extend(
            [
                "--beta_frame",
                str(args.beta_frame),
                "--frame_bias_mode",
                str(args.frame_bias_mode),
                "--read_calib_mode",
                "per_frame_quantile",
                "--read_target_mass",
                str(args.read_target_mass),
                "--read_calib_tau",
                str(args.read_calib_tau),
                "--read_blend_lambda",
                str(args.read_blend_lambda),
                "--read_topk_frac",
                str(args.read_topk_frac),
            ]
        )
    cmd.extend(
        [
            "--fast_cue_eval",
            "1",
            "--empty_cuda_cache_each_chunk",
            "1",
            "--hybrid_debug_jsonl",
            str(out_dir / "hmc_state_hash.jsonl"),
        ]
    )
    return cmd


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
    parser.add_argument("--cases", default=",".join(READ_CASES))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage-c-cache-dir", type=Path, default=DEFAULT_STAGE_C_CACHE)
    parser.add_argument("--radio-sidecar-dir", type=Path, default=DEFAULT_RADIO_SIDECAR)
    parser.add_argument("--read-cue-source", default=DEFAULT_READ_CUE_SOURCE)
    parser.add_argument("--read-layer-mode", default="early")
    parser.add_argument("--frame-bias-mode", default="key")
    parser.add_argument("--cuda-alloc-conf", default=DEFAULT_CUDA_ALLOC_CONF)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--beta-frame", type=float, default=0.5)
    parser.add_argument("--read-target-mass", type=float, default=0.10)
    parser.add_argument("--read-calib-tau", type=float, default=0.05)
    parser.add_argument("--read-blend-lambda", type=float, default=0.50)
    parser.add_argument("--read-topk-frac", type=float, default=0.10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    read_cases = _read_cases_for_cue(args.read_cue_source)
    cases = [case.strip() for case in str(args.cases).split(",") if case.strip()]
    unknown = [case for case in cases if case not in read_cases]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(read_cases)}")

    jobs: List[Dict[str, Any]] = []
    gpu_cursor = 0
    for chunk in chunks:
        start = int(chunk) * (int(args.chunk_size) - int(args.chunk_overlap))
        end = start + int(args.chunk_size)
        for case in cases:
            out_dir = args.output_root / f"chunk{chunk:02d}" / case
            cmd = _build_command(args, chunk=chunk, case=case, out_dir=out_dir)
            skipped = bool(args.skip_existing and (out_dir / "01.txt").exists() and (out_dir / "hmc_state_hash.jsonl").exists())
            jobs.append({
                "chunk": int(chunk),
                "start_frame": int(start),
                "end_frame": int(end),
                "case": case,
                "gpu": int(gpus[gpu_cursor % len(gpus)]),
                "out_dir": str(out_dir),
                "cmd": cmd,
                "cmd_shell": shlex.join(cmd),
                "workdir": str(args.workdir),
                "read_cue_source_effective": read_cases[case]["read_cue_source"],
                "cuda_alloc_conf": str(args.cuda_alloc_conf),
                "skipped": skipped,
                "returncode": 0 if skipped else None,
            })
            gpu_cursor += 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "v70_radio_read_online_manifest.json"
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
