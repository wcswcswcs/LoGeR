#!/usr/bin/env python3
"""Run ACL2 v78 Phase 1 Global-V L13 read-control jobs.

The runner is intentionally thin: it launches fixed, training-free H35/v53
read-path runs and writes an auditable manifest. It does not decide success.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/"
    "report_final/phase1_l13_controls/rollouts"
)
DEFAULT_INPUT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_STAGE_C_CACHE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_READ_CUE_SOURCE = "v68.read.global_v.l13"
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


PHASE1_CASES: Dict[str, Dict[str, Any]] = {
    "L13_BASE": {
        "enable_frame_read_control": "0",
        "read_path": "none",
        "read_cue_source": "dyn",
    },
    "L13_NEG_DAMP_ACTUAL": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "suffix": "",
    },
    "L13_NEG_RANDOM_SAME_MASS": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "suffix": ".random_same_mass",
    },
    "L13_NEG_RANDOM_GROUP_STRATIFIED": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "suffix": ".group_stratified_random",
    },
    "L13_NEG_RANDOM_LOWSTUFF_WITHIN_GROUP": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "suffix": ".lowstuff_within_group_random",
    },
    "L13_NEG_RANDOM_HIGH_D_WITHIN_GROUP": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "suffix": ".high_d_within_group_random",
    },
    "L13_NEG_RANDOM_SAME_ATTENTION_MASS": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "suffix": ".same_attention_mass_random",
    },
    "L13_NEG_RANDOM_SAME_COMPOSITION_AND_DG": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "suffix": ".same_composition_and_dg_random",
    },
    "L13_NEG_LABEL_SHUFFLE": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "suffix": ".label_shuffled",
    },
    "L13_NEG_CONFIDENCE_SHUFFLE": {
        "enable_frame_read_control": "1",
        "read_path": "frame",
        "suffix": ".confidence_shuffled",
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
    return [int(part.strip()) for part in str(text or "").split(",") if part.strip()]


def _case_cfg(case: str, read_cue_source: str) -> Dict[str, str]:
    cfg = dict(PHASE1_CASES[case])
    if "suffix" in cfg:
        cfg["read_cue_source"] = f"{read_cue_source}{cfg.pop('suffix')}"
    return cfg


def _build_command(args: argparse.Namespace, *, chunk: int, case: str, out_dir: Path) -> List[str]:
    cfg = _case_cfg(case, args.read_cue_source)
    start = int(chunk) * (int(args.chunk_size) - int(args.chunk_overlap))
    end = start + int(args.chunk_size)
    cmd = [
        str(args.conda),
        "run",
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
        str(cfg["enable_frame_read_control"]),
        "--read_path",
        str(cfg["read_path"]),
        "--read_cue_source",
        str(cfg["read_cue_source"]),
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
    ]
    if case != "L13_BASE":
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
    parser.add_argument("--chunks", default="6")
    parser.add_argument("--cases", default=",".join(PHASE1_CASES))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage-c-cache-dir", type=Path, default=DEFAULT_STAGE_C_CACHE)
    parser.add_argument("--read-cue-source", default=DEFAULT_READ_CUE_SOURCE)
    parser.add_argument("--cuda-alloc-conf", default=DEFAULT_CUDA_ALLOC_CONF)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--beta-frame", type=float, default=0.5)
    parser.add_argument("--frame-bias-mode", choices=("pair", "protected_pair", "key", "query"), default="key")
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
    cases = [case.strip() for case in str(args.cases).split(",") if case.strip()]
    unknown = [case for case in cases if case not in PHASE1_CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(PHASE1_CASES)}")

    jobs: List[Dict[str, Any]] = []
    gpu_cursor = 0
    for chunk in chunks:
        start = int(chunk) * (int(args.chunk_size) - int(args.chunk_overlap))
        end = start + int(args.chunk_size)
        for case in cases:
            out_dir = args.output_root / f"chunk{int(chunk):02d}" / case
            cmd = _build_command(args, chunk=int(chunk), case=case, out_dir=out_dir)
            skipped = bool(
                args.skip_existing
                and (out_dir / "01.txt").exists()
                and (out_dir / "hmc_state_hash.jsonl").exists()
            )
            cfg = _case_cfg(case, args.read_cue_source)
            job = {
                "chunk": int(chunk),
                "start_frame": int(start),
                "end_frame": int(end),
                "case": case,
                "gpu": int(gpus[gpu_cursor % len(gpus)]),
                "out_dir": str(out_dir),
                "cmd": cmd,
                "cmd_shell": shlex.join(cmd),
                "workdir": str(args.workdir),
                "read_cue_source_effective": cfg["read_cue_source"],
                "cuda_alloc_conf": str(args.cuda_alloc_conf),
                "skipped": skipped,
                "returncode": 0 if skipped else None,
            }
            gpu_cursor += 1
            jobs.append(job)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "phase1_l13_run_manifest.json"
    manifest: Dict[str, Any] = {"args": _jsonable(vars(args)), "jobs": jobs}
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"planned_jobs={len(jobs)} manifest={manifest_path}")
    if args.dry_run:
        return

    run_jobs = [job for job in jobs if not job["skipped"]]
    completed: List[Dict[str, Any]] = [job for job in jobs if job["skipped"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(_run_job, job) for job in run_jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                "done",
                f"chunk={result['chunk']}",
                f"case={result['case']}",
                f"gpu={result['gpu']}",
                f"returncode={result['returncode']}",
                f"duration_sec={result['duration_sec']:.1f}",
                flush=True,
            )
            manifest["completed_count"] = len(completed)
            manifest["jobs"] = completed + [job for job in jobs if job not in completed and job not in run_jobs]
            manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
