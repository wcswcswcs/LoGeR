#!/usr/bin/env python3
"""Run ACL2 v80 READ/SWA direct-hook smoke dumps for selected chunks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_CHECKPOINT = REPO_ROOT / "ckpts/LoGeR/latest.pt"
DEFAULT_CONFIG = REPO_ROOT / "ckpts/LoGeR/original_config.yaml"
DEFAULT_DATA = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset")


def _parse_ints(text: str) -> List[int]:
    out: List[int] = []
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _chunk_window(chunk: int, chunk_size: int, chunk_overlap: int) -> Dict[str, int]:
    stride = int(chunk_size) - int(chunk_overlap)
    start = int(chunk) * stride
    end = start + int(chunk_size)
    return {"chunk": int(chunk), "start_frame": int(start), "end_frame": int(end)}


def _stage_c_masklet(stage_c_root: Path, chunk: int) -> Path:
    matches = sorted(stage_c_root.glob(f"chunk_{int(chunk):03d}_*/masklet.pt"))
    if not matches:
        raise FileNotFoundError(f"missing stage-c masklet for chunk {chunk}: {stage_c_root}")
    return matches[0]


def _build_cmd(args: argparse.Namespace, *, chunk: int, out_dir: Path) -> List[str]:
    window = _chunk_window(chunk, args.chunk_size, args.chunk_overlap)
    return [
        str(args.conda),
        "run",
        "--no-capture-output",
        "-n",
        str(args.conda_env),
        "python",
        "run_pipeline_abc_v2.py",
        "--input",
        str(args.input_dir),
        "--output_video",
        "",
        "--output_txt",
        str(out_dir / f"{args.seq}.txt"),
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
        "--stage_c_cache_mode",
        "read",
        "--stage_c_cache_dir",
        str(args.stage_c_cache_dir),
        "--stage_c_cache_require_hit",
        "1",
        "--enable_frame_read_control",
        "1",
        "--read_path",
        "frame",
        "--read_cue_source",
        str(args.read_cue_source),
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
        "--read_cue_patch_dump_dir",
        str(out_dir / "read_cue_patch_dumps"),
        "--read_cue_patch_dump_dtype",
        str(args.read_cue_patch_dump_dtype),
        "--v68_export_full_pca_debug",
        "1",
        "--v68_pca_max_feature_dim",
        str(args.v68_pca_max_feature_dim),
        "--v68_layer_pca_feature_dir",
        str(out_dir / "pca_features"),
        "--v68_pca_taps",
        str(args.v68_pca_taps),
        "--v68_pca_layers",
        str(args.v68_pca_layers),
        "--beta_frame",
        str(args.beta_frame),
        "--frame_bias_mode",
        "key",
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


def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    if str(job.get("cuda_alloc_conf") or "").strip():
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(job["cuda_alloc_conf"]).strip()
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            job["cmd"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    job["returncode"] = int(proc.returncode)
    job["duration_sec"] = float(time.time() - start)
    job["run_log"] = str(log_path)
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="01")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--stage-c-cache-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--read-cue-source", default="v78.l07_l13.l07_action_only")
    parser.add_argument("--read-cue-patch-dump-dtype", default="float16")
    parser.add_argument(
        "--v68-pca-taps",
        default="attn_global_k,attn_global_v,attn_frame_v,swa_q,swa_k,swa_v,swa_cache_k,swa_cache_v",
    )
    parser.add_argument("--v68-pca-layers", default="5,6,13,14,17,18")
    parser.add_argument("--v68-pca-max-feature-dim", type=int, default=8)
    parser.add_argument("--beta-frame", type=float, default=0.5)
    parser.add_argument("--read-target-mass", type=float, default=0.1)
    parser.add_argument("--read-calib-tau", type=float, default=0.05)
    parser.add_argument("--read-blend-lambda", type=float, default=0.5)
    parser.add_argument("--read-topk-frac", type=float, default=0.1)
    parser.add_argument("--cuda-alloc-conf", default="expandable_segments:True")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()

    args.seq = str(args.seq).zfill(2)
    if args.input_dir is None:
        args.input_dir = DEFAULT_DATA / "sequences" / args.seq / "image_2"
    if args.stage_c_cache_dir is None:
        args.stage_c_cache_dir = Path(f"results/kitti_preprocess/{args.seq}/stage_c_cache_semantic_chunks")
    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)
    if not args.stage_c_cache_dir.exists():
        raise FileNotFoundError(args.stage_c_cache_dir)

    chunks = _parse_ints(args.chunks)
    gpus = _parse_ints(args.gpus)
    if not chunks:
        raise ValueError("--chunks is empty")
    if not gpus:
        raise ValueError("--gpus is empty")

    args.out_root.mkdir(parents=True, exist_ok=True)
    jobs: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        _stage_c_masklet(args.stage_c_cache_dir, chunk)
        out_dir = args.out_root / f"chunk{chunk:03d}"
        expected = out_dir / "pca_features" / f"chunk_{chunk:03d}.pt"
        skipped = bool(args.skip_existing and expected.exists())
        cmd = _build_cmd(args, chunk=chunk, out_dir=out_dir)
        jobs.append(
            {
                "chunk": int(chunk),
                "gpu": int(gpus[idx % len(gpus)]),
                "out_dir": str(out_dir),
                "cmd": cmd,
                "cmd_shell": shlex.join(cmd),
                "expected": str(expected),
                "read_cue_expected": str(out_dir / "read_cue_patch_dumps" / f"chunk_{chunk:03d}_read_cue_patch.pt"),
                "skipped": skipped,
                "returncode": 0 if skipped else None,
                "cuda_alloc_conf": str(args.cuda_alloc_conf),
            }
        )

    manifest = {"args": vars(args), "jobs": jobs}
    manifest_path = args.out_root / "read_swa_direct_hook_smoke_manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"planned_jobs={len(jobs)} manifest={manifest_path}")
    if args.no_run:
        return

    completed: List[Dict[str, Any]] = [job for job in jobs if job["skipped"]]
    run_jobs = [job for job in jobs if not job["skipped"]]
    by_gpu: Dict[int, List[Dict[str, Any]]] = {int(g): [] for g in gpus}
    for job in run_jobs:
        by_gpu[int(job["gpu"])].append(job)

    def _run_queue(queue: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [_run_job(dict(job)) for job in queue]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(_run_queue, queue) for queue in by_gpu.values() if queue]
        for future in concurrent.futures.as_completed(futures):
            completed.extend(future.result())

    completed = sorted(completed, key=lambda row: int(row["chunk"]))
    ok = all(
        int(row.get("returncode") or 0) == 0
        and Path(str(row.get("expected"))).exists()
        and Path(str(row.get("read_cue_expected"))).exists()
        for row in completed
    )
    summary = {
        "all_pipeline_jobs_ok": bool(ok),
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "out_root": str(args.out_root),
        "jobs": completed,
    }
    summary_path = args.out_root / "read_swa_direct_hook_smoke_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable({k: summary[k] for k in ["all_pipeline_jobs_ok", "diagnostic_only", "method_gate_claimed", "out_root"]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
