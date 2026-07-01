#!/usr/bin/env python3
"""Run ACL2 v79 Phase2 semantic READ/global-attention control jobs.

This is an experiment wrapper only. It selects single-chunk targets from the
Phase1 ledger, maps v79 READ cases to existing HMC read-cue implementations,
and records an auditable manifest. It does not change model code or claim
success.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/"
    "report_final/phase2_semantic_read_global_control/rollouts"
)
DEFAULT_TARGET_CSV = Path(
    "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/"
    "report_final/phase1_current_bad_target_mining_with_semantic_diagnosis/"
    "single_chunk_semantic_read_targets.csv"
)
DEFAULT_INPUT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_STAGE_C_CACHE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


CASES: dict[str, dict[str, str]] = {
    "READ0_NATIVE": {
        "role": "baseline_native_no_read",
        "read_path": "none",
        "enable_frame_read_control": "0",
        "cue": "dyn",
        "semantic_contract": "no semantic read action; native baseline",
    },
    "READ1_L07_SEMANTIC_LAYOUT_SELECT": {
        "role": "candidate",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_action_only",
        "semantic_contract": "L07 layout/motion selects semantic-risk frame read mass",
    },
    "READ2_L13_SEMANTIC_VALUE_DAMP": {
        "role": "candidate",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l13_only",
        "semantic_contract": "L13 value-risk damp signal without L07 layout mask",
    },
    "READ3_L13_STABLE_PROTECT": {
        "role": "candidate",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_mask_l13_stable",
        "semantic_contract": "L07 layout masks L13 value-risk while protecting stable support",
    },
    "READ4_L07_TO_L13_SEMANTIC_CONTRAST": {
        "role": "candidate",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_mask_l13_contrast",
        "semantic_contract": "L07 layout and L13 value-risk disagreement drives contrastive read",
    },
    "READ5_FRAME_L18_SEMANTIC_TAIL_STABILIZE": {
        "role": "candidate",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v68.read.global_v.l18",
        "semantic_contract": "higher-layer global value semantic-risk read cue for tail stability",
    },
    "READ6_GEOMETRY_ONLY_CONTROL": {
        "role": "geometry_only_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v68.read.global_v.l13.geometry_only",
        "semantic_contract": "geometry-only read control; no semantic role should be credited",
    },
    "READ7_LABEL_SHUFFLE": {
        "role": "semantic_shuffle_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_mask_l13_contrast.label_shuffled",
        "semantic_contract": "same contrast construction with shuffled labels",
    },
    "READ8_CONFIDENCE_SHUFFLE": {
        "role": "semantic_shuffle_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_mask_l13_contrast.confidence_shuffled",
        "semantic_contract": "same contrast construction with shuffled confidence",
    },
    "READ9_SAME_READ_MASS_RANDOM": {
        "role": "same_read_mass_random_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_mask_l13_contrast.same_attention_mass_random",
        "semantic_contract": "per-frame same read-mass randomization control",
    },
    "READ10_GROUP_STRATIFIED_RANDOM": {
        "role": "group_stratified_random_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_mask_l13_contrast.group_stratified_random",
        "semantic_contract": "semantic-group stratified randomization control",
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


def _parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text or "").split(",") if part.strip()]


def _parse_csv_cases(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _select_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, str]] = []
    with args.target_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if args.target_scope and str(row.get("target_scope")) != str(args.target_scope):
                continue
            if args.sequence and str(row.get("sequence")) != str(args.sequence):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(
            f"no target rows selected from {args.target_csv} "
            f"scope={args.target_scope!r} sequence={args.sequence!r}"
        )
    selected: list[dict[str, Any]] = []
    seen_chunks: set[int] = set()
    for row in rows:
        chunk = int(row["chunk_id"])
        if chunk in seen_chunks:
            continue
        seen_chunks.add(chunk)
        target = dict(row)
        target["chunk_id"] = chunk
        target["frame_start"] = int(row["frame_start"])
        target["frame_end"] = int(row["frame_end"])
        selected.append(target)
        if len(selected) >= int(args.max_targets):
            break
    return selected


def _build_command(args: argparse.Namespace, *, target: dict[str, Any], case: str, out_dir: Path) -> list[str]:
    cfg = CASES[case]
    chunk = int(target["chunk_id"])
    start = int(chunk) * (int(args.chunk_size) - int(args.chunk_overlap))
    end = start + int(args.chunk_size)
    if start != int(target.get("frame_start", start)):
        # Keep the run aligned with the pipeline chunk convention; the mismatch
        # remains visible in the manifest target row for audit.
        pass
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
        str(cfg["cue"]),
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
    ]
    if case != "READ0_NATIVE":
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


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
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
        proc = subprocess.run(job["cmd"], cwd=job["workdir"], env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
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
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--target-scope", default="primary_kitti01_v53_h35")
    parser.add_argument("--sequence", default="01")
    parser.add_argument("--max-targets", type=int, default=1)
    parser.add_argument("--cases", default=",".join(CASES))
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
    parser.add_argument("--beta-frame", type=float, default=0.5)
    parser.add_argument("--frame-bias-mode", choices=("pair", "protected_pair", "key", "query"), default="key")
    parser.add_argument("--read-target-mass", type=float, default=0.10)
    parser.add_argument("--read-calib-tau", type=float, default=0.05)
    parser.add_argument("--read-blend-lambda", type=float, default=0.50)
    parser.add_argument("--read-topk-frac", type=float, default=0.10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = _parse_csv_ints(args.gpus)
    cases = _parse_csv_cases(args.cases)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    unknown = [case for case in cases if case not in CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(CASES)}")
    targets = _select_targets(args)

    jobs: list[dict[str, Any]] = []
    gpu_cursor = 0
    for target in targets:
        chunk = int(target["chunk_id"])
        start = int(chunk) * (int(args.chunk_size) - int(args.chunk_overlap))
        end = start + int(args.chunk_size)
        for case in cases:
            out_dir = args.output_root / f"chunk{chunk:02d}" / case
            cmd = _build_command(args, target=target, case=case, out_dir=out_dir)
            skipped = bool(args.skip_existing and (out_dir / "01.txt").exists() and (out_dir / "hmc_state_hash.jsonl").exists())
            job = {
                "chunk": int(chunk),
                "start_frame": int(start),
                "end_frame": int(end),
                "phase1_target": target,
                "case": case,
                "case_role": CASES[case]["role"],
                "semantic_contract": CASES[case]["semantic_contract"],
                "gpu": int(gpus[gpu_cursor % len(gpus)]),
                "out_dir": str(out_dir),
                "cmd": cmd,
                "cmd_shell": shlex.join(cmd),
                "workdir": str(args.workdir),
                "read_cue_source_effective": CASES[case]["cue"],
                "cuda_alloc_conf": str(args.cuda_alloc_conf),
                "skipped": skipped,
                "returncode": 0 if skipped else None,
            }
            gpu_cursor += 1
            jobs.append(job)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "phase2_semantic_read_run_manifest.json"
    manifest: dict[str, Any] = {
        "args": _jsonable(vars(args)),
        "case_definitions": _jsonable(CASES),
        "selected_targets": _jsonable(targets),
        "jobs": _jsonable(jobs),
        "method_gate_claimed": False,
        "note": "Phase2 run manifest only; evaluate gate separately.",
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"planned_jobs={len(jobs)} manifest={manifest_path}")
    if args.dry_run:
        return

    run_jobs = [job for job in jobs if not job["skipped"]]
    completed: list[dict[str, Any]] = [job for job in jobs if job["skipped"]]
    queued_by_gpu: dict[int, list[dict[str, Any]]] = {int(gpu): [] for gpu in gpus}
    for job in run_jobs:
        queued_by_gpu.setdefault(int(job["gpu"]), []).append(job)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queued_by_gpu)) as pool:
        futures: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
        for gpu, queue in queued_by_gpu.items():
            if queue:
                futures[pool.submit(_run_job, queue.pop(0))] = int(gpu)
        while futures:
            for future in concurrent.futures.as_completed(list(futures)):
                gpu = futures.pop(future)
                break
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
            manifest["jobs"] = _jsonable(completed + [job for job in jobs if job not in completed and job not in run_jobs])
            manifest_path.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if queued_by_gpu.get(gpu):
                futures[pool.submit(_run_job, queued_by_gpu[gpu].pop(0))] = gpu

    by_key = {(int(job["chunk"]), str(job["case"])): job for job in completed}
    ordered = [by_key.get((int(job["chunk"]), str(job["case"])), job) for job in jobs]
    manifest["jobs"] = _jsonable(ordered)
    manifest["completed_count"] = len(completed)
    manifest["failed_jobs"] = _jsonable([job for job in ordered if int(job.get("returncode") or 0) != 0])
    manifest_path.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if manifest["failed_jobs"]:
        raise SystemExit(f"failed_jobs={len(manifest['failed_jobs'])}; see {manifest_path}")


if __name__ == "__main__":
    main()
