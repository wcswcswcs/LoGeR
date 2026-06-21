#!/usr/bin/env python3
"""Run ACL2 v78 Phase3 SWA handoff bridge smoke jobs with a GPU queue."""

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


DEFAULT_CHUNKS = [6]
DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/"
    "report_final/phase3_swa_handoff/smoke_chunk06_context2"
)
DEFAULT_INPUT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_STAGE_C_CACHE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


PHASE3_CASES: Dict[str, Dict[str, Any]] = {
    "S0_NATIVE": {
        "enable_swa_overlap_bias": "0",
        "read_cue_source": "dyn",
        "swa_overlap_bias_mode": "pair",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "S1_SWA_MASS_PRESERVING_L13_NEG_ROUTE": {
        "enable_swa_overlap_bias": "1",
        "read_cue_source": "v78.l07_l13.l13_only",
        "swa_overlap_bias_mode": "semantic_role_negative",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "S2_SWA_MASS_PRESERVING_L13_STABLE_ROUTE": {
        "enable_swa_overlap_bias": "1",
        "read_cue_source": "v78.l07_l13.l13_only",
        "swa_overlap_bias_mode": "semantic_role_stable",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "S3_SWA_L07_MASK_L13_ROUTE": {
        "enable_swa_overlap_bias": "1",
        "read_cue_source": "v78.l07_l13.l07_mask_l13_neg",
        "swa_overlap_bias_mode": "semantic_role_negative",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "S4_SWA_CONTEXT_FLOOR": {
        "enable_swa_overlap_bias": "1",
        "read_cue_source": "dyn",
        "swa_overlap_bias_mode": "semantic_role_protect",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "S5_SWA_STABLE_PROTECT_PLUS_HARM_DAMP": {
        "enable_swa_overlap_bias": "1",
        "read_cue_source": "v78.l07_l13.l07_mask_l13_neg_plus_stable",
        "swa_overlap_bias_mode": "semantic_role_stable_protect_minus_negative",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
    },
    "S6_SWA_GEOMETRY_ONLY_ROUTE": {
        "enable_swa_overlap_bias": "1",
        "read_cue_source": "dyn",
        "swa_overlap_bias_mode": "source",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "S7_SWA_SAME_ROLE_MASS_RANDOM_ROUTE": {
        "enable_swa_overlap_bias": "1",
        "read_cue_source": "v78.l07_l13.l13_only",
        "swa_overlap_bias_mode": "semantic_role_negative_random_same_mass",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "swa",
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
    case_cfg = PHASE3_CASES[case]
    window = _context_window(args, int(chunk))
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
        str(window["start_frame"]),
        "--end_frame",
        str(window["end_frame"]),
        "--global_chunk_offset",
        str(window["context_start_chunk"]),
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
        "0",
        "--read_path",
        "none",
        "--read_cue_source",
        str(case_cfg["read_cue_source"]),
        "--read_layer_mode",
        "single",
        "--read_single_layer",
        str(args.read_single_layer),
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
        "--semantic_role_policy",
        str(case_cfg["semantic_role_policy"]),
        "--semantic_memory_paths",
        str(case_cfg["semantic_memory_paths"]),
        "--semantic_role_highd_quantile",
        str(args.semantic_role_highd_quantile),
        "--semantic_role_low_trust",
        str(args.semantic_role_low_trust),
        "--enable_swa_overlap_bias",
        str(case_cfg["enable_swa_overlap_bias"]),
        "--swa_overlap_bias_beta",
        str(args.swa_overlap_bias_beta if case != "S0_NATIVE" else 0.0),
        "--swa_overlap_bias_min_keep",
        str(args.swa_overlap_bias_min_keep),
        "--swa_overlap_bias_mode",
        str(case_cfg["swa_overlap_bias_mode"]),
        "--swa_overlap_bias_layer_mode",
        str(args.swa_overlap_bias_layer_mode),
        "--swa_overlap_feature_dump_dir",
        str(out_dir / "swa_overlap_feature_maps"),
        "--swa_overlap_feature_dump_dtype",
        str(args.swa_overlap_feature_dump_dtype),
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
    ]
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
    parser.add_argument("--cases", default=",".join(PHASE3_CASES))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage-c-cache-dir", type=Path, default=DEFAULT_STAGE_C_CACHE)
    parser.add_argument("--cuda-alloc-conf", default=DEFAULT_CUDA_ALLOC_CONF)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--context-chunks", type=int, default=2)
    parser.add_argument("--read-single-layer", type=int, default=13)
    parser.add_argument("--read-target-mass", type=float, default=0.10)
    parser.add_argument("--read-calib-tau", type=float, default=0.05)
    parser.add_argument("--read-blend-lambda", type=float, default=0.50)
    parser.add_argument("--read-topk-frac", type=float, default=0.10)
    parser.add_argument("--semantic-role-highd-quantile", type=float, default=0.80)
    parser.add_argument("--semantic-role-low-trust", type=float, default=0.20)
    parser.add_argument("--swa-overlap-bias-beta", type=float, default=0.50)
    parser.add_argument("--swa-overlap-bias-min-keep", type=float, default=0.50)
    parser.add_argument("--swa-overlap-bias-layer-mode", default="last")
    parser.add_argument("--swa-overlap-feature-dump-dtype", default="float16")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    cases = [case.strip() for case in str(args.cases).split(",") if case.strip()]
    unknown = [case for case in cases if case not in PHASE3_CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(PHASE3_CASES)}")

    jobs: List[Dict[str, Any]] = []
    gpu_cursor = 0
    for chunk in chunks:
        window = _context_window(args, int(chunk))
        for case in cases:
            out_dir = args.output_root / f"chunk{chunk:02d}" / case
            cmd = _build_command(args, chunk=chunk, case=case, out_dir=out_dir)
            skipped = bool(args.skip_existing and (out_dir / "01.txt").exists() and (out_dir / "hmc_state_hash.jsonl").exists())
            jobs.append({
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
                "read_cue_source_effective": PHASE3_CASES[case]["read_cue_source"],
                "swa_overlap_bias_mode_effective": PHASE3_CASES[case]["swa_overlap_bias_mode"],
                "semantic_role_policy_effective": PHASE3_CASES[case]["semantic_role_policy"],
                "semantic_memory_paths_effective": PHASE3_CASES[case]["semantic_memory_paths"],
                "cuda_alloc_conf": str(args.cuda_alloc_conf),
                "skipped": skipped,
                "returncode": 0 if skipped else None,
            })
            gpu_cursor += 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "phase3_swa_handoff_run_manifest.json"
    manifest: Dict[str, Any] = {"args": vars(args), "jobs": jobs}
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"planned_jobs={len(jobs)} manifest={manifest_path}")
    if args.dry_run:
        return

    run_jobs = [job for job in jobs if not job["skipped"]]
    completed: List[Dict[str, Any]] = [job for job in jobs if job["skipped"]]
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
                manifest["jobs"] = completed + [item for item in jobs if item not in completed]
                manifest_path.write_text(
                    json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(
                f"[gpu{gpu}] chunk={result['chunk']} case={result['case']} "
                f"returncode={result['returncode']} duration={result['duration_sec']:.1f}s"
            )
        return gpu_results

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(_run_gpu_queue, gpu, queue)
            for gpu, queue in jobs_by_gpu.items()
            if queue
        ]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    completed_by_key = {(int(job["chunk"]), str(job["case"])): job for job in completed}
    ordered = [completed_by_key.get((int(job["chunk"]), str(job["case"])), job) for job in jobs]
    failed = [job for job in ordered if int(job.get("returncode") or 0) != 0]
    manifest["jobs"] = ordered
    manifest["completed_count"] = int(len([job for job in ordered if job.get("returncode") is not None]))
    manifest["failed_jobs"] = [
        {"chunk": int(job["chunk"]), "case": str(job["case"]), "returncode": int(job.get("returncode") or -1)}
        for job in failed
    ]
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"completed={manifest['completed_count']} failed={len(failed)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
