#!/usr/bin/env python3
"""Run minimal multi-sequence LoGeR prefixes for v81S S1 geometry repair.

This is an artifact-repair runner, not a method experiment. It runs native
read-path LoGeR on a fixed prefix and saves per-chunk geometry so S1 can
materialize adjacent overlap pairs for multiple KITTI sequences.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS1_multiseq_swa_overlap_repair/geometry_prefix_runs"
)
DEFAULT_DATA_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


def _parse_csv(text: str) -> list[str]:
    return [part.strip().zfill(2) for part in str(text).split(",") if part.strip()]


def _parse_gpus(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _geometry_file_count(path: Path) -> int:
    return len(list(path.glob("chunk_*.pt"))) if path.is_dir() else 0


def _qkv_file_count(path: Path) -> int:
    return len(list(path.glob("chunk_*.pt"))) if path.is_dir() else 0


def _build_command(args: argparse.Namespace, seq: str, run_dir: Path) -> list[str]:
    cmd = [
        str(args.conda),
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "run_pipeline_abc_v2.py",
        "--input",
        str(args.data_root / seq / "image_2"),
        "--output_video",
        "",
        "--output_txt",
        str(run_dir / f"{seq}.txt"),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--chunk_size",
        str(args.chunk_size),
        "--chunk_overlap",
        str(args.chunk_overlap),
        "--start_frame",
        "0",
        "--end_frame",
        str(args.end_frame),
        "--global_chunk_offset",
        "0",
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
        str(args.preprocess_root / seq / "stage_c_cache_semantic_chunks"),
        "--stage_c_cache_require_hit",
        "1",
        "--read_path",
        "none",
        "--read_cue_source",
        "dyn",
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--per_chunk_geometry_dir",
        str(run_dir / "per_chunk_geometry"),
        "--per_chunk_pose_trace_jsonl",
        str(run_dir / "per_chunk_pose_trace.jsonl"),
        "--hybrid_debug_jsonl",
        str(run_dir / "hmc_state_hash.jsonl"),
    ]
    if args.v68_layer_pca_feature_subdir:
        cmd.extend(
            [
                "--v68_layer_pca_feature_dir",
                str(run_dir / str(args.v68_layer_pca_feature_subdir)),
                "--v68_pca_taps",
                str(args.v68_pca_taps),
                "--v68_pca_layers",
                str(args.v68_pca_layers),
                "--v68_pca_max_feature_dim",
                str(args.v68_pca_max_feature_dim),
            ]
        )
    return cmd


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    if job.get("cuda_alloc_conf"):
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(job["cuda_alloc_conf"])
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            job["cmd"],
            cwd=job["workdir"],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    job["returncode"] = int(proc.returncode)
    job["duration_sec"] = float(time.time() - start)
    job["run_log"] = str(log_path)
    job["geometry_file_count"] = _geometry_file_count(Path(job["geometry_dir"]))
    if job.get("qkv_feature_dir"):
        job["qkv_feature_file_count"] = _qkv_file_count(Path(job["qkv_feature_dir"]))
    job["trajectory_exists"] = Path(job["trajectory"]).is_file()
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqs", default="00,01,02,05")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--end-frame", type=int, default=641)
    parser.add_argument("--min-existing-chunks", type=int, default=21)
    parser.add_argument("--cuda-alloc-conf", default=DEFAULT_CUDA_ALLOC_CONF)
    parser.add_argument("--v68-layer-pca-feature-subdir", default="")
    parser.add_argument(
        "--v68-pca-taps",
        default="global_q_raw_patchvec_layers,global_k_raw_patchvec_layers,global_v_raw_patchvec_layers",
    )
    parser.add_argument("--v68-pca-layers", default="0,4,8,12,16,17")
    parser.add_argument("--v68-pca-max-feature-dim", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seqs = _parse_csv(args.seqs)
    gpus = _parse_gpus(args.gpus)
    if not gpus:
        raise ValueError("--gpus must not be empty")
    args.output_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for idx, seq in enumerate(seqs):
        run_dir = args.output_root / f"seq{seq}_native_prefix{int(args.end_frame)}"
        geometry_dir = run_dir / "per_chunk_geometry"
        qkv_feature_dir = run_dir / str(args.v68_layer_pca_feature_subdir) if args.v68_layer_pca_feature_subdir else None
        trajectory = run_dir / f"{seq}.txt"
        skipped = False
        if args.skip_existing and _geometry_file_count(geometry_dir) >= int(args.min_existing_chunks):
            skipped = True
        job = {
            "seq": seq,
            "gpu": int(gpus[idx % len(gpus)]),
            "run_dir": str(run_dir),
            "geometry_dir": str(geometry_dir),
            "qkv_feature_dir": str(qkv_feature_dir) if qkv_feature_dir is not None else "",
            "trajectory": str(trajectory),
            "cmd": _build_command(args, seq, run_dir),
            "workdir": str(args.workdir),
            "cuda_alloc_conf": str(args.cuda_alloc_conf or ""),
            "skipped_existing": skipped,
        }
        if not skipped:
            jobs.append(job)

    manifest_path = args.output_root / "v81s_geometry_prefix_manifest.json"
    manifest: dict[str, Any] = {
        "schema": "acl2_v81s_geometry_prefix_repair_v1",
        "seqs": seqs,
        "gpus": gpus,
        "end_frame": int(args.end_frame),
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "dry_run": bool(args.dry_run),
        "jobs": jobs,
    }
    if args.dry_run or not jobs:
        manifest_path.write_text(
            json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(_jsonable({k: v for k, v in manifest.items() if k != "jobs"}), ensure_ascii=False, indent=2, sort_keys=True))
        print(f"wrote_manifest={manifest_path}")
        return

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        future_to_job = {pool.submit(_run_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "seq": result["seq"],
                        "gpu": result["gpu"],
                        "returncode": result["returncode"],
                        "geometry_file_count": result["geometry_file_count"],
                        "qkv_feature_file_count": result.get("qkv_feature_file_count"),
                        "duration_sec": result["duration_sec"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    manifest["jobs"] = sorted(results, key=lambda row: str(row["seq"]))
    manifest["failed_jobs"] = [row for row in manifest["jobs"] if int(row.get("returncode", 1)) != 0]
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote_manifest={manifest_path}")


if __name__ == "__main__":
    main()
