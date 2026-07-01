#!/usr/bin/env python3
"""Run v98 Stage7c same-run TTT write + SWA raw transport probes.

The runner is diagnostic-only.  It reproduces the v97 F2 probe_ttt_write dump
shape, adds SWA raw transport trace in the same pipeline run, and keeps the
result scoped to F3 instrumentation.  It does not enable an E3 runtime action
or claim write-to-use identity.
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


ROOT = Path("results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control")
DEFAULT_OUT = ROOT / "stage7c_ttt_swa_same_run_probe"
F2_EXACT_ROWS = (
    Path("results/acl2_v97tf_semantic_scale_evidence_gauge_safe_memory_control")
    / "trackF2_ttt_stable_anchor_retention_missing_good_write/exact_retention_rows.csv"
)
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DATA_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            clean = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
            writer.writerow(clean)


def case_ids_from_f2(limit: int) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in read_rows(F2_EXACT_ROWS):
        case_id = str(row.get("case_id", ""))
        if case_id and case_id not in seen:
            seen.add(case_id)
            ordered.append(case_id)
        if len(ordered) >= limit:
            break
    return ordered


def parse_case_id(case_id: str) -> dict[str, Any]:
    parts = str(case_id).split("_")
    if len(parts) != 3:
        raise ValueError(f"expected case_id=SEQ_PREV_CURR, got {case_id}")
    seq = parts[0]
    prev_chunk = int(parts[1])
    curr_chunk = int(parts[2])
    stride = 32 - 3
    start_frame = int(prev_chunk * stride)
    end_frame = int(start_frame + 32 + stride)
    return {
        "case_id": case_id,
        "seq": seq,
        "prev_chunk": prev_chunk,
        "curr_chunk": curr_chunk,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "global_chunk_offset": prev_chunk,
    }


def build_cmd(args: argparse.Namespace, case: dict[str, Any], out_dir: Path) -> list[str]:
    seq = str(case["seq"])
    cmd = [
        str(args.conda),
        "run",
        "--no-capture-output",
        "-n",
        str(args.conda_env),
        "python",
        "run_pipeline_abc_v2.py",
        "--input",
        str(DATA_ROOT / seq / "image_2"),
        "--output_video",
        "",
        "--output_txt",
        str(out_dir / f"{seq}.txt"),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--device",
        "cuda",
        "--start_frame",
        str(case["start_frame"]),
        "--end_frame",
        str(case["end_frame"]),
        "--stride",
        "1",
        "--chunk_size",
        "32",
        "--chunk_overlap",
        "3",
        "--window_size",
        "32",
        "--overlap_size",
        "3",
        "--global_chunk_offset",
        str(case["global_chunk_offset"]),
        "--hybrid_memory_mode",
        "ttt_write_only",
        "--hmc_commit_mode",
        "probe_ttt_write",
        "--ttt_spatial_post_delta_map_dump_dir",
        str(out_dir / "ttt_spatial_post_delta_maps"),
        "--ttt_spatial_post_delta_map_dump_dtype",
        "float16",
        "--ttt_write_token_contribution_diagnostic",
        "1",
        "--ttt_write_scale_state_mode",
        "v19_scale_state",
        "--ttt_write_scale_state_alpha",
        "1.0",
        "--ttt_write_scale_state_proxy",
        "pose_step_ema",
        "--swa_raw_transport_trace_dir",
        str(out_dir / "swa_raw_transport_trace"),
        "--swa_raw_transport_trace_layer_mode",
        str(args.swa_raw_transport_trace_layer_mode),
        "--swa_raw_transport_trace_single_layer",
        str(args.swa_raw_transport_trace_single_layer),
        "--swa_raw_transport_trace_max_queries",
        str(args.swa_raw_transport_trace_max_queries),
        "--swa_raw_transport_trace_topk",
        str(args.swa_raw_transport_trace_topk),
        "--empty_cuda_cache_each_chunk",
        "1",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
    ]
    if bool(args.enable_per_chunk_geometry_sidecar):
        cmd += [
            "--per_chunk_geometry_dir",
            str(out_dir / "per_chunk_geometry"),
            "--per_chunk_pose_trace_jsonl",
            str(out_dir / "per_chunk_pose_trace.jsonl"),
        ]
    if bool(args.use_stage_c_cache):
        cmd += [
            "--stage_c_cache_mode",
            "read",
            "--stage_c_cache_dir",
            str(ROOT.parent / f"kitti_preprocess/{seq}/stage_c_cache_semantic_chunks"),
            "--stage_c_cache_require_hit",
            "1",
        ]
    if bool(args.enable_swa_prev_ttt_stable_anchor_gate):
        cmd += [
            "--enable_swa_prev_ttt_stable_anchor_gate",
            "1",
            "--swa_prev_ttt_stable_anchor_gate_rho",
            str(args.swa_prev_ttt_stable_anchor_gate_rho),
            "--swa_prev_ttt_stable_anchor_gate_min",
            str(args.swa_prev_ttt_stable_anchor_gate_min),
            "--swa_prev_ttt_stable_anchor_gate_target",
            str(args.swa_prev_ttt_stable_anchor_gate_target),
            "--swa_prev_ttt_stable_anchor_gate_layer_mode",
            str(args.swa_prev_ttt_stable_anchor_gate_layer_mode),
            "--swa_prev_ttt_stable_anchor_gate_single_layer",
            str(args.swa_prev_ttt_stable_anchor_gate_single_layer),
        ]
    if bool(args.enable_swa_prev_ttt_anchor_query_soft):
        cmd += [
            "--enable_swa_prev_ttt_anchor_query_soft",
            "1",
            "--swa_prev_ttt_anchor_query_soft_rho",
            str(args.swa_prev_ttt_anchor_query_soft_rho),
            "--swa_prev_ttt_anchor_query_soft_min_keep",
            str(args.swa_prev_ttt_anchor_query_soft_min_keep),
            "--swa_prev_ttt_anchor_query_soft_query_head_frac_threshold",
            str(args.swa_prev_ttt_anchor_query_soft_query_head_frac_threshold),
            "--swa_prev_ttt_anchor_query_soft_topk",
            str(args.swa_prev_ttt_anchor_query_soft_topk),
            "--swa_prev_ttt_anchor_query_soft_query_block_size",
            str(args.swa_prev_ttt_anchor_query_soft_query_block_size),
            "--swa_prev_ttt_anchor_query_soft_layer_mode",
            str(args.swa_prev_ttt_anchor_query_soft_layer_mode),
            "--swa_prev_ttt_anchor_query_soft_single_layer",
            str(args.swa_prev_ttt_anchor_query_soft_single_layer),
            "--swa_prev_ttt_anchor_query_soft_attention_mass_max_queries",
            str(args.swa_prev_ttt_anchor_query_soft_attention_mass_max_queries),
        ]
    return cmd


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    started = time.time()
    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(job["cmd"], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    job.update(
        {
            "returncode": int(proc.returncode),
            "duration_sec": float(time.time() - started),
            "run_log": str(log_path),
            "output_txt": str(out_dir / f"{job['seq']}.txt"),
            "hmc_jsonl": str(out_dir / "hmc_state_hash.jsonl"),
        }
    )
    write_json(out_dir / "job_summary.json", job)
    return job


def run_jobs(jobs: list[dict[str, Any]], gpus: list[int]) -> list[dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {int(job["job_index"]): dict(job) for job in jobs if job.get("skipped")}
    queues: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in gpus}
    for job in jobs:
        if not job.get("skipped"):
            queues.setdefault(int(job["gpu"]), []).append(dict(job))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(queues), 1)) as pool:
        futures: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
        for gpu, queue in queues.items():
            if queue:
                futures[pool.submit(run_job, queue.pop(0))] = int(gpu)
        while futures:
            for future in concurrent.futures.as_completed(list(futures)):
                gpu = futures.pop(future)
                break
            result = future.result()
            completed[int(result["job_index"])] = result
            print(
                "done",
                f"case={result['case_id']}",
                f"gpu={result['gpu']}",
                f"returncode={result['returncode']}",
                f"duration_sec={result['duration_sec']:.1f}",
                flush=True,
            )
            if queues.get(gpu):
                futures[pool.submit(run_job, queues[gpu].pop(0))] = gpu
    return [completed.get(int(job["job_index"]), job) for job in jobs]


def summarize(output_root: Path, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    ttt_maps = list(output_root.glob("*/TTT_SWA_SAME_RUN/ttt_spatial_post_delta_maps/*.pt"))
    swa_traces = list(output_root.glob("*/TTT_SWA_SAME_RUN/swa_raw_transport_trace/*.pt"))
    geometry_sidecars = list(output_root.glob("*/TTT_SWA_SAME_RUN/per_chunk_geometry/*.pt"))
    pose_traces = list(output_root.glob("*/TTT_SWA_SAME_RUN/per_chunk_pose_trace.jsonl"))
    ttt_cases = {path.parents[2].name for path in ttt_maps}
    swa_cases = {path.parents[2].name for path in swa_traces}
    geometry_cases = {path.parents[2].name for path in geometry_sidecars}
    failed_jobs = [job for job in jobs if int(job.get("returncode", 1)) != 0]
    summary = {
        "schema": "acl2_v98_stage7c_ttt_swa_same_run_probe_summary_v1",
        "status": "complete" if not failed_jobs else "partial_or_failed",
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "gate_pass": False,
        "selected_case_count": len(jobs),
        "completed_job_count": len([job for job in jobs if job.get("returncode") is not None]),
        "failed_job_count": len(failed_jobs),
        "ttt_spatial_map_file_count": len(ttt_maps),
        "swa_raw_transport_trace_file_count": len(swa_traces),
        "per_chunk_geometry_sidecar_file_count": len(geometry_sidecars),
        "per_chunk_pose_trace_file_count": len(pose_traces),
        "case_with_ttt_map_count": len(ttt_cases),
        "case_with_swa_trace_count": len(swa_cases),
        "case_with_geometry_sidecar_count": len(geometry_cases),
        "case_with_both_count": len(ttt_cases & swa_cases),
        "case_with_trace_and_geometry_count": len(swa_cases & geometry_cases),
        "true_anchor_identity_available": False,
        "write_to_use_chain_available": False,
        "claim_scope": "same_run_ttt_write_and_swa_trace_only",
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case-id", action="append", default=None, help="Case id SEQ_PREV_CURR; repeat to override F2 defaults.")
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--conda", type=Path, default=CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--swa-raw-transport-trace-layer-mode", default="all", choices=("all", "first", "last", "single"))
    parser.add_argument("--swa-raw-transport-trace-single-layer", type=int, default=-1)
    parser.add_argument("--swa-raw-transport-trace-max-queries", type=int, default=128)
    parser.add_argument("--swa-raw-transport-trace-topk", type=int, default=8)
    parser.add_argument(
        "--enable-per-chunk-geometry-sidecar",
        type=int,
        default=0,
        help=(
            "If nonzero, pass run_pipeline_abc_v2 --per_chunk_geometry_dir and "
            "--per_chunk_pose_trace_jsonl so each case writes local_points/points/"
            "camera_poses sidecars for ACL2 true L2 diagnostics."
        ),
    )
    parser.add_argument("--use-stage-c-cache", type=int, default=1)
    parser.add_argument("--enable-swa-prev-ttt-stable-anchor-gate", type=int, default=0)
    parser.add_argument("--swa-prev-ttt-stable-anchor-gate-rho", type=float, default=0.0)
    parser.add_argument("--swa-prev-ttt-stable-anchor-gate-min", type=float, default=0.85)
    parser.add_argument(
        "--swa-prev-ttt-stable-anchor-gate-target",
        choices=("v", "value", "k", "key", "kv", "both"),
        default="v",
    )
    parser.add_argument(
        "--swa-prev-ttt-stable-anchor-gate-layer-mode",
        choices=("all", "first", "last", "single"),
        default="last",
    )
    parser.add_argument("--swa-prev-ttt-stable-anchor-gate-single-layer", type=int, default=-1)
    parser.add_argument("--enable-swa-prev-ttt-anchor-query-soft", type=int, default=0)
    parser.add_argument("--swa-prev-ttt-anchor-query-soft-rho", type=float, default=0.0)
    parser.add_argument("--swa-prev-ttt-anchor-query-soft-min-keep", type=float, default=0.5)
    parser.add_argument("--swa-prev-ttt-anchor-query-soft-query-head-frac-threshold", type=float, default=0.75)
    parser.add_argument("--swa-prev-ttt-anchor-query-soft-topk", type=int, default=8)
    parser.add_argument("--swa-prev-ttt-anchor-query-soft-query-block-size", type=int, default=64)
    parser.add_argument(
        "--swa-prev-ttt-anchor-query-soft-layer-mode",
        choices=("all", "first", "last", "single"),
        default="last",
    )
    parser.add_argument("--swa-prev-ttt-anchor-query-soft-single-layer", type=int, default=-1)
    parser.add_argument("--swa-prev-ttt-anchor-query-soft-attention-mass-max-queries", type=int, default=64)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    gpus = [int(item) for item in str(args.gpus).split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must list at least one GPU")
    case_ids = args.case_id if args.case_id else case_ids_from_f2(max(int(args.max_cases), 1))
    cases = [parse_case_id(case_id) for case_id in case_ids[: int(args.max_cases)]]
    jobs: list[dict[str, Any]] = []
    for case in cases:
        out_dir = args.output_root / str(case["case_id"]) / "TTT_SWA_SAME_RUN"
        cmd = build_cmd(args, case, out_dir)
        skipped = bool(
            args.skip_existing
            and (out_dir / f"{case['seq']}.txt").exists()
            and (out_dir / "hmc_state_hash.jsonl").exists()
        )
        jobs.append(
            {
                "job_index": len(jobs),
                **case,
                "variant": "TTT_SWA_SAME_RUN",
                "diagnostic_only": True,
                "gpu": int(gpus[len(jobs) % len(gpus)]),
                "out_dir": str(out_dir),
                "cmd": cmd,
                "cmd_shell": shlex.join(cmd),
                "skipped": skipped,
                "returncode": 0 if skipped else None,
            }
        )
    manifest = {
        "schema": "acl2_v98_stage7c_ttt_swa_same_run_probe_manifest_v1",
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "method_gate_claimed": False,
        "source_case_bank": str(F2_EXACT_ROWS),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "jobs": [{key: value for key, value in job.items() if key != "cmd"} for job in jobs],
    }
    write_json(args.output_root / "same_run_probe_manifest.json", manifest)
    write_rows(args.output_root / "job_manifest.csv", [{key: value for key, value in job.items() if key != "cmd"} for job in jobs])
    if args.no_run:
        write_json(args.output_root / "summary.json", {"status": "planned_not_run", "planned_jobs": len(jobs), "gate_pass": False})
        return
    completed = jobs if args.summarize_only else run_jobs(jobs, gpus)
    write_rows(args.output_root / "job_results.csv", [{key: value for key, value in job.items() if key != "cmd"} for job in completed])
    summarize(args.output_root, completed)


if __name__ == "__main__":
    main()
