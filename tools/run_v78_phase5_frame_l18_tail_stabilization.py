#!/usr/bin/env python3
"""Run ACL2 v78 Phase5 frame-attention L18 tail-stabilization smokes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_v68_phaseD_read_smoke import (  # noqa: E402
    LOWER_IS_BETTER_KEYS,
    _eval_run,
    _finite,
    _load_kitti_gt,
    _safe_ratio_improvement,
    _write_csv,
)


DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/"
    "report_final/phase5_frame_l18_tail/smoke_chunk06_context2_v1"
)
DEFAULT_INPUT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_STAGE_C_CACHE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"

MECHANISM_KEYS = [
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]


PHASE5_CASES: Dict[str, Dict[str, Any]] = {
    "F0_NATIVE": {
        "enable_context_source_skip": 0,
        "context_source_skip_mask": "dg_q90",
        "context_source_skip_frame_region": "all",
        "context_source_skip_mode": "soft",
        "context_source_skip_soft_rho": 0.0,
        "context_source_skip_soft_min_keep": 1.0,
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
    },
    "F1_FRAME_L18_TAIL_STABLE_PROTECT": {
        "enable_context_source_skip": 1,
        "context_source_skip_mask": "semantic_role_negative",
        "context_source_skip_frame_region": "tail",
        "context_source_skip_mode": "soft",
        "context_source_skip_soft_rho": 0.35,
        "context_source_skip_soft_min_keep": 0.75,
        "semantic_role_policy": "fine_fg_lowstuff_highd_skip",
        "semantic_memory_paths": "frame",
    },
    "F2_FRAME_L18_TAIL_HARM_DAMP": {
        "enable_context_source_skip": 1,
        "context_source_skip_mask": "semantic_role_negative",
        "context_source_skip_frame_region": "tail",
        "context_source_skip_mode": "soft",
        "context_source_skip_soft_rho": 0.60,
        "context_source_skip_soft_min_keep": 0.50,
        "semantic_role_policy": "fine_fg_lowstuff_highd_skip",
        "semantic_memory_paths": "frame",
    },
    "F3_FRAME_L18_HEAD_TAIL_REBALANCE": {
        "enable_context_source_skip": 1,
        "context_source_skip_mask": "semantic_role_negative",
        "context_source_skip_frame_region": "mid_tail",
        "context_source_skip_mode": "soft",
        "context_source_skip_soft_rho": 0.45,
        "context_source_skip_soft_min_keep": 0.65,
        "semantic_role_policy": "fine_fg_lowstuff_highd_skip",
        "semantic_memory_paths": "frame",
    },
    "F4_FRAME_L18_RANDOM_SAME_MASS": {
        "enable_context_source_skip": 1,
        "context_source_skip_mask": "random_same_mass_semantic_role_negative",
        "context_source_skip_frame_region": "tail",
        "context_source_skip_mode": "soft",
        "context_source_skip_soft_rho": 0.60,
        "context_source_skip_soft_min_keep": 0.50,
        "semantic_role_policy": "fine_fg_lowstuff_highd_skip",
        "semantic_memory_paths": "frame",
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text or "").split(",") if part.strip()]


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


def _read_jsonl_all(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finite_mean(values: Iterable[Any]) -> Optional[float]:
    xs = [_finite(v) for v in values]
    xs = [v for v in xs if v is not None]
    return float(np.mean(xs)) if xs else None


def _finite_max(values: Iterable[Any]) -> Optional[float]:
    xs = [_finite(v) for v in values]
    xs = [v for v in xs if v is not None]
    return float(np.max(xs)) if xs else None


def _aggregate_phase5_frame_hmc(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl_all(run_dir / "hmc_state_hash.jsonl")
    if not rows:
        return {"phase5_hmc_rows": 0}
    frame_summaries = [
        row.get("control_trace", {}).get("hook_effect_summary", {}).get("frame_attention", {})
        for row in rows
        if isinstance(row.get("control_trace", {}).get("hook_effect_summary", {}).get("frame_attention", {}), dict)
    ]
    return {
        "phase5_hmc_rows": int(len(rows)),
        "phase5_frame_num_calls_sum": int(sum(int(row.get("num_calls", 0) or 0) for row in frame_summaries)),
        "phase5_frame_context_source_skip_applied_sum": int(
            sum(int(row.get("num_context_source_skip_applied", 0) or 0) for row in frame_summaries)
        ),
        "phase5_frame_max_context_source_skip_tokens": _finite_max(
            row.get("max_context_source_skip_tokens") for row in frame_summaries
        ),
        "phase5_frame_mean_context_source_keep_ratio": _finite_mean(
            row.get("mean_context_source_keep_ratio") for row in frame_summaries
        ),
        "phase5_frame_mean_attention_mass_removed_before": _finite_mean(
            row.get("mean_attention_mass_removed_before") for row in frame_summaries
        ),
        "phase5_frame_mean_attention_mass_removed_after": _finite_mean(
            row.get("mean_attention_mass_removed_after") for row in frame_summaries
        ),
        "phase5_frame_mean_attention_mass_actual_after": _finite_mean(
            row.get("mean_attention_mass_actual_after") for row in frame_summaries
        ),
        "phase5_frame_attention_mass_available_any": bool(
            any(row.get("attention_mass_available") for row in frame_summaries)
        ),
        "phase5_frame_attention_mass_metrics_seen": sorted(
            {
                str(metric)
                for row in frame_summaries
                for metric in (row.get("attention_mass_metrics_seen") or [])
            }
        ),
    }


def _build_command(args: argparse.Namespace, *, chunk: int, case: str, out_dir: Path) -> List[str]:
    cfg = PHASE5_CASES[case]
    window = _context_window(args, int(chunk))
    map_dump_dir = out_dir / "frame_attention_map_dump"
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
        "--read_path",
        "none",
        "--read_cue_source",
        "dyn",
        "--read_overlap_frames",
        str(args.chunk_overlap),
        "--semantic_role_policy",
        str(cfg["semantic_role_policy"]),
        "--semantic_memory_paths",
        str(cfg["semantic_memory_paths"]),
        "--semantic_role_highd_quantile",
        str(args.semantic_role_highd_quantile),
        "--semantic_role_low_trust",
        str(args.semantic_role_low_trust),
        "--enable_context_source_skip",
        str(cfg["enable_context_source_skip"]),
        "--context_source_skip_scope",
        "frame",
        "--context_source_skip_impl",
        "bias",
        "--context_source_skip_mode",
        str(cfg["context_source_skip_mode"]),
        "--context_source_skip_mask",
        str(cfg["context_source_skip_mask"]),
        "--context_source_skip_frame_region",
        str(cfg["context_source_skip_frame_region"]),
        "--context_source_skip_layer_mode",
        "single",
        "--context_source_skip_single_layer",
        str(args.frame_single_layer),
        "--context_source_skip_soft_rho",
        str(cfg["context_source_skip_soft_rho"]),
        "--context_source_skip_soft_min_keep",
        str(cfg["context_source_skip_soft_min_keep"]),
        "--context_source_skip_record_attention_mass",
        "1",
        "--context_source_skip_attention_mass_max_queries",
        str(args.attention_mass_max_queries),
        "--context_source_skip_attention_map_dump_dir",
        str(map_dump_dir),
        "--context_source_skip_attention_map_dump_max_queries",
        str(args.attention_map_dump_max_queries),
        "--context_source_skip_attention_map_dump_dtype",
        str(args.attention_map_dump_dtype),
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


def _best_control_value(rows_by_name: Dict[str, Dict[str, Any]], controls: Sequence[str], key: str) -> Optional[float]:
    vals = [_finite(rows_by_name.get(name, {}).get(key)) for name in controls]
    vals = [value for value in vals if value is not None]
    return min(vals) if vals else None


def _build_phase5_decision(
    rows: Sequence[Dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    controls: Sequence[str],
) -> Dict[str, Any]:
    rows_by_name = {str(row["run"]): row for row in rows}
    cand = rows_by_name.get(candidate)
    base = rows_by_name.get(baseline)
    if cand is None:
        return {"phase5_gate_pass": False, "reason": f"missing_candidate:{candidate}"}
    if base is None:
        return {"phase5_gate_pass": False, "reason": f"missing_baseline:{baseline}"}

    comparisons: Dict[str, Dict[str, Any]] = {}
    metric_passes: List[str] = []
    for key in LOWER_IS_BETTER_KEYS:
        cand_v = _finite(cand.get(key))
        base_v = _finite(base.get(key))
        best_control = _best_control_value(rows_by_name, controls, key)
        beats_controls = cand_v is not None and best_control is not None and cand_v < best_control
        ratio_improvement = _safe_ratio_improvement(base_v, cand_v)
        future_worse = False
        if key in {"head10_to_tail10_pose_sim3_rmse_m", "scale_cv_head_mid_tail_pose_sim3"}:
            future_ratio = _safe_ratio_improvement(
                base.get("overlap3_to_future_pose_sim3_rmse_m"),
                cand.get("overlap3_to_future_pose_sim3_rmse_m"),
            )
            future_worse = bool(future_ratio is not None and future_ratio < -0.01)
            key_pass = bool(
                beats_controls
                and ratio_improvement is not None
                and ratio_improvement >= 0.10
                and not future_worse
            )
        else:
            key_pass = False
        if key_pass:
            metric_passes.append(key)
        comparisons[key] = {
            "candidate": cand_v,
            "baseline": base_v,
            "best_control": best_control,
            "candidate_minus_baseline": (cand_v - base_v) if cand_v is not None and base_v is not None else None,
            "candidate_minus_best_control": (
                cand_v - best_control if cand_v is not None and best_control is not None else None
            ),
            "improvement_vs_baseline_ratio": ratio_improvement,
            "beats_controls": beats_controls,
            "future_worse_gt1pct": future_worse,
            "phase5_metric_key_pass": key_pass,
        }

    action_fidelity = bool(
        _finite(cand.get("phase5_frame_context_source_skip_applied_sum")) is not None
        and float(cand.get("phase5_frame_context_source_skip_applied_sum") or 0.0) > 0.0
        and _finite(cand.get("phase5_frame_max_context_source_skip_tokens")) is not None
        and float(cand.get("phase5_frame_max_context_source_skip_tokens") or 0.0) > 0.0
    )
    return {
        "phase5_gate_pass": bool(metric_passes and action_fidelity),
        "candidate": candidate,
        "baseline": baseline,
        "controls": list(controls),
        "metric_passes": metric_passes,
        "action_fidelity_pass": action_fidelity,
        "comparisons": comparisons,
        "rule": (
            "Phase5 requires action fidelity, head_tail or scale_cv >=10% improvement, "
            "future_after_overlap not worsening by >1%, and beating same-mass random controls."
        ),
    }


def _evaluate(args: argparse.Namespace, jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    _, gt_poses_all, gt_pos_all = _load_kitti_gt(args.gt)
    rows: List[Dict[str, Any]] = []
    for job in jobs:
        if int(job.get("returncode") or 0) != 0:
            continue
        run_dir = Path(job["out_dir"])
        row = _eval_run(str(job["case"]), run_dir, gt_poses_all, gt_pos_all)
        row.update(_aggregate_phase5_frame_hmc(run_dir))
        rows.append(row)

    candidates = [
        "F1_FRAME_L18_TAIL_STABLE_PROTECT",
        "F2_FRAME_L18_TAIL_HARM_DAMP",
        "F3_FRAME_L18_HEAD_TAIL_REBALANCE",
    ]
    controls = ["F4_FRAME_L18_RANDOM_SAME_MASS"]
    decisions = {
        candidate: _build_phase5_decision(rows, candidate=candidate, baseline="F0_NATIVE", controls=controls)
        for candidate in candidates
    }
    payload = {
        "schema": "acl2_v78_phase5_frame_l18_tail_v1",
        "output_root": str(args.output_root),
        "runs": rows,
        "baseline": "F0_NATIVE",
        "candidates": candidates,
        "controls": controls,
        "decisions": decisions,
        "phase5_any_gate_pass": bool(any(dec.get("phase5_gate_pass") for dec in decisions.values())),
        "actuator_boundary": (
            "Uses frame-attention context_source_skip/source-side bias at single layer 18; "
            "does not claim direct frame-V tensor replacement."
        ),
    }
    json_path = args.output_root / "phase5_frame_l18_tail_metrics.json"
    csv_path = args.output_root / "phase5_frame_l18_tail_metrics.csv"
    decision_path = args.output_root / "phase5_frame_l18_tail_decision.json"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, rows)
    decision_path.write_text(
        json.dumps(
            {
                "phase5_any_gate_pass": payload["phase5_any_gate_pass"],
                "decisions": decisions,
                "actuator_boundary": payload["actuator_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable({"phase5_any_gate_pass": payload["phase5_any_gate_pass"], "decisions": decisions}), indent=2, sort_keys=True))
    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_decision={decision_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunks", default="6")
    parser.add_argument("--cases", default=",".join(PHASE5_CASES))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage-c-cache-dir", type=Path, default=DEFAULT_STAGE_C_CACHE)
    parser.add_argument("--cuda-alloc-conf", default=DEFAULT_CUDA_ALLOC_CONF)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--context-chunks", type=int, default=2)
    parser.add_argument("--frame-single-layer", type=int, default=18)
    parser.add_argument("--semantic-role-highd-quantile", type=float, default=0.75)
    parser.add_argument("--semantic-role-low-trust", type=float, default=0.55)
    parser.add_argument("--attention-mass-max-queries", type=int, default=256)
    parser.add_argument("--attention-map-dump-max-queries", type=int, default=32)
    parser.add_argument("--attention-map-dump-dtype", default="float16")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    cases = [case.strip() for case in str(args.cases).split(",") if case.strip()]
    unknown = [case for case in cases if case not in PHASE5_CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(PHASE5_CASES)}")

    jobs: List[Dict[str, Any]] = []
    gpu_cursor = 0
    for chunk in chunks:
        window = _context_window(args, int(chunk))
        for case in cases:
            out_dir = args.output_root / f"chunk{chunk:02d}" / case
            cmd = _build_command(args, chunk=chunk, case=case, out_dir=out_dir)
            skipped = bool(args.skip_existing and (out_dir / "01.txt").exists() and (out_dir / "hmc_state_hash.jsonl").exists())
            cfg = PHASE5_CASES[case]
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
                    "frame_single_layer": int(args.frame_single_layer),
                    "actuator_boundary": "frame_attention context_source_skip source-side bias",
                    "context_source_skip_mask_effective": cfg["context_source_skip_mask"],
                    "context_source_skip_frame_region_effective": cfg["context_source_skip_frame_region"],
                    "semantic_role_policy_effective": cfg["semantic_role_policy"],
                    "semantic_memory_paths_effective": cfg["semantic_memory_paths"],
                    "cuda_alloc_conf": str(args.cuda_alloc_conf),
                    "skipped": skipped,
                    "returncode": 0 if skipped else None,
                }
            )
            gpu_cursor += 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "phase5_frame_l18_tail_run_manifest.json"
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
                manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(
                f"[gpu{gpu}] chunk={result['chunk']} case={result['case']} "
                f"returncode={result['returncode']} duration={result['duration_sec']:.1f}s"
            )
        return gpu_results

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(_run_gpu_queue, gpu, queue) for gpu, queue in jobs_by_gpu.items() if queue]
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
    if failed:
        return
    _evaluate(args, ordered)


if __name__ == "__main__":
    main()
