#!/usr/bin/env python3
"""Run ACL2 v68 Phase E MERGE/gauge multi-chunk jobs with a GPU queue."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_CHUNKS = [6, 7, 8, 10, 12, 19, 20, 29, 30, 31, 32]
DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v68_integrated_cueconstruction/"
    "phaseE_merge_multichunk/s5_scale_only_alpha06_log07"
)
DEFAULT_INPUT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_STAGE_C_CACHE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


MERGE_CASES: Dict[str, Dict[str, Any]] = {
    "native_no_swa": {
        "semantic_merge": False,
        "strategy": "",
        "ttt_tail_drop": False,
    },
    "candidate": {
        "semantic_merge": True,
        "strategy": "S5_SUPPRESS_DYNAMIC_SKY",
        "ttt_tail_drop": False,
    },
    "geometry_only": {
        "semantic_merge": True,
        "strategy": "S1_GEOMETRY_ONLY",
        "ttt_tail_drop": False,
    },
    "same_cue_random": {
        "semantic_merge": True,
        "strategy": "S5_SUPPRESS_DYNAMIC_SKY_RANDOM",
        "ttt_tail_drop": False,
    },
    "label_shuffled": {
        "semantic_merge": True,
        "strategy": "S5_SUPPRESS_DYNAMIC_SKY_SHUFFLED",
        "ttt_tail_drop": False,
    },
    "overlap_support": {
        "semantic_merge": True,
        "strategy": "V68_OVERLAP_SUPPORT_WEIGHT",
        "ttt_tail_drop": False,
    },
    "overlap_support_random": {
        "semantic_merge": True,
        "strategy": "V68_OVERLAP_SUPPORT_WEIGHT_RANDOM",
        "ttt_tail_drop": False,
    },
    "overlap_support_shuffled": {
        "semantic_merge": True,
        "strategy": "V68_OVERLAP_SUPPORT_WEIGHT_SHUFFLED",
        "ttt_tail_drop": False,
    },
    "robust_semoverlap": {
        "semantic_merge": True,
        "strategy": "V68_ROBUST_SEMOVERLAP_WEIGHT",
        "ttt_tail_drop": False,
    },
    "robust_semoverlap_random": {
        "semantic_merge": True,
        "strategy": "V68_ROBUST_SEMOVERLAP_WEIGHT_RANDOM",
        "ttt_tail_drop": False,
    },
    "robust_semoverlap_shuffled": {
        "semantic_merge": True,
        "strategy": "V68_ROBUST_SEMOVERLAP_WEIGHT_SHUFFLED",
        "ttt_tail_drop": False,
    },
    "overlap_outlier": {
        "semantic_merge": True,
        "strategy": "V80_OVERLAP_OUTLIER_DOWNWEIGHT",
        "ttt_tail_drop": False,
    },
    "overlap_outlier_random": {
        "semantic_merge": True,
        "strategy": "V80_OVERLAP_OUTLIER_DOWNWEIGHT_RANDOM",
        "ttt_tail_drop": False,
    },
    "overlap_outlier_shuffled": {
        "semantic_merge": True,
        "strategy": "V80_OVERLAP_OUTLIER_DOWNWEIGHT_SHUFFLED",
        "ttt_tail_drop": False,
    },
    "retrieval_static": {
        "semantic_merge": True,
        "strategy": "V81_RETRIEVAL_STATIC_OVERLAP",
        "ttt_tail_drop": False,
    },
    "retrieval_static_random": {
        "semantic_merge": True,
        "strategy": "V81_RETRIEVAL_STATIC_OVERLAP_RANDOM",
        "ttt_tail_drop": False,
    },
    "retrieval_static_shuffled": {
        "semantic_merge": True,
        "strategy": "V81_RETRIEVAL_STATIC_OVERLAP_SHUFFLED",
        "ttt_tail_drop": False,
    },
    "latent_kalman": {
        "semantic_merge": True,
        "strategy": "V81_LATENT_KALMAN_OVERLAP",
        "ttt_tail_drop": False,
    },
    "latent_kalman_random": {
        "semantic_merge": True,
        "strategy": "V81_LATENT_KALMAN_OVERLAP_RANDOM",
        "ttt_tail_drop": False,
    },
    "latent_kalman_shuffled": {
        "semantic_merge": True,
        "strategy": "V81_LATENT_KALMAN_OVERLAP_SHUFFLED",
        "ttt_tail_drop": False,
    },
    "radio_component": {
        "semantic_merge": True,
        "strategy": "V73_RADIO_COMPONENT_HANDOFF",
        "ttt_tail_drop": False,
    },
    "radio_component_random": {
        "semantic_merge": True,
        "strategy": "V73_RADIO_COMPONENT_HANDOFF_RANDOM",
        "ttt_tail_drop": False,
    },
    "radio_component_shuffled": {
        "semantic_merge": True,
        "strategy": "V73_RADIO_COMPONENT_HANDOFF_SHUFFLED",
        "ttt_tail_drop": False,
    },
    "radio_qscale": {
        "semantic_merge": True,
        "strategy": "V73_RADIO_QSCALE_HANDOFF",
        "ttt_tail_drop": False,
    },
    "radio_qscale_random": {
        "semantic_merge": True,
        "strategy": "V73_RADIO_QSCALE_HANDOFF_RANDOM",
        "ttt_tail_drop": False,
    },
    "radio_qscale_shuffled": {
        "semantic_merge": True,
        "strategy": "V73_RADIO_QSCALE_HANDOFF_SHUFFLED",
        "ttt_tail_drop": False,
    },
    "thingstuff_state": {
        "semantic_merge": True,
        "strategy": "V73_THINGSTUFF_STATE_HANDOFF",
        "ttt_tail_drop": False,
    },
    "thingstuff_state_random": {
        "semantic_merge": True,
        "strategy": "V73_THINGSTUFF_STATE_HANDOFF_RANDOM",
        "ttt_tail_drop": False,
    },
    "thingstuff_state_shuffled": {
        "semantic_merge": True,
        "strategy": "V73_THINGSTUFF_STATE_HANDOFF_SHUFFLED",
        "ttt_tail_drop": False,
    },
    "thingstuff_radio_qscale": {
        "semantic_merge": True,
        "strategy": "V73_THINGSTUFF_RADIO_QSCALE_HANDOFF",
        "ttt_tail_drop": False,
    },
    "thingstuff_radio_qscale_random": {
        "semantic_merge": True,
        "strategy": "V73_THINGSTUFF_RADIO_QSCALE_HANDOFF_RANDOM",
        "ttt_tail_drop": False,
    },
    "thingstuff_radio_qscale_shuffled": {
        "semantic_merge": True,
        "strategy": "V73_THINGSTUFF_RADIO_QSCALE_HANDOFF_SHUFFLED",
        "ttt_tail_drop": False,
    },
    "combo_candidate": {
        "semantic_merge": True,
        "strategy": "S5_SUPPRESS_DYNAMIC_SKY",
        "ttt_tail_drop": True,
    },
    "combo_geometry_only": {
        "semantic_merge": True,
        "strategy": "S1_GEOMETRY_ONLY",
        "ttt_tail_drop": True,
    },
    "combo_same_cue_random": {
        "semantic_merge": True,
        "strategy": "S5_SUPPRESS_DYNAMIC_SKY_RANDOM",
        "ttt_tail_drop": True,
    },
    "combo_label_shuffled": {
        "semantic_merge": True,
        "strategy": "S5_SUPPRESS_DYNAMIC_SKY_SHUFFLED",
        "ttt_tail_drop": True,
    },
}
DEFAULT_CASES = ["native_no_swa", "candidate", "geometry_only", "same_cue_random", "label_shuffled"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _build_command(args: argparse.Namespace, *, chunk: int, case: str, out_dir: Path) -> List[str]:
    case_cfg = MERGE_CASES[case]
    ttt_tail_drop = bool(case_cfg.get("ttt_tail_drop", False))
    stride = int(args.chunk_size) - int(args.chunk_overlap)
    start = max(0, (int(chunk) - 1) * stride)
    end = int(chunk) * stride + int(args.chunk_size)
    global_chunk_offset = max(0, int(chunk) - 1)
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
        str(global_chunk_offset),
        "--device",
        "cuda",
        "--hybrid_memory_mode",
        "hybrid" if ttt_tail_drop else "read_path_only",
        "--hmc_commit_mode",
        "probe_ttt_write" if ttt_tail_drop else "controlled",
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
        "dyn",
    ]
    if ttt_tail_drop:
        cmd.extend(
            [
                "--read_overlap_frames",
                "3",
                "--ttt_write_gradient_reversal_mode",
                "tri_replay",
                "--ttt_write_gradient_reversal_risk_source",
                "update_conflict_energy",
                "--ttt_write_gradient_reversal_gamma",
                "0.0",
                "--ttt_write_gradient_reversal_negative_frac",
                "0.0",
                "--ttt_write_gradient_reversal_branch_mask",
                "0",
                "--ttt_write_tri_replay_positive_frac",
                "0.0",
                "--ttt_write_tri_replay_negative_frac",
                "0.0",
                "--ttt_write_tri_replay_neutral_lambda",
                "0.0",
                "--ttt_write_tri_replay_role_mode",
                "adaptive_writer_fused",
                "--ttt_write_native_mix_scales",
                "1.00,1.00,1.00",
                "--ttt_write_commit_ema_alpha",
                "1.0",
                "--ttt_write_commit_ema_branch_mask",
                "all",
                "--ttt_write_token_scope",
                "tail_overlap_drop",
                "--ttt_write_token_scope_floor",
                "0.0",
            ]
        )
    if bool(case_cfg["semantic_merge"]):
        cmd.extend(
            [
                "--semantic_merge_mode",
                "current_world_to_aligned_previous",
                "--semantic_merge_strategy",
                str(case_cfg["strategy"]),
                "--semantic_merge_use_semantic_confidence",
                "1",
                "--semantic_merge_semantic_conf_min",
                str(args.semantic_conf_min),
                "--semantic_merge_blend_alpha",
                str(args.blend_alpha),
                "--semantic_merge_blend_components",
                "scale_only",
                "--semantic_merge_max_blend_log_scale_delta",
                str(args.max_blend_log_scale_delta),
                "--semantic_merge_max_blend_rotation_delta_deg",
                "0.0",
                "--semantic_merge_max_blend_translation_delta",
                "0.0",
                "--semantic_merge_max_points",
                str(args.semantic_merge_max_points),
            ]
        )
        if bool(args.clip_blend_to_limit):
            cmd.extend(["--semantic_merge_clip_blend_to_limit", "1"])
        overlap_support_dir = str(args.semantic_merge_overlap_support_dir or "").strip()
        if overlap_support_dir:
            cmd.extend(
                [
                    "--semantic_merge_overlap_support_dir",
                    overlap_support_dir,
                    "--semantic_merge_overlap_support_kind",
                    str(args.semantic_merge_overlap_support_kind),
                    "--semantic_merge_overlap_support_floor",
                    str(args.semantic_merge_overlap_support_floor),
                ]
            )
        radio_sidecar_dir = str(args.radio_sidecar_dir or "").strip()
        if radio_sidecar_dir:
            cmd.extend(["--v70_radio_sidecar_dir", radio_sidecar_dir])
        if bool(args.reject_worse_than_native_overlap):
            cmd.extend(
                [
                    "--semantic_merge_reject_worse_than_native_overlap",
                    "1",
                    "--semantic_merge_native_overlap_tolerance",
                    str(args.native_overlap_tolerance),
                ]
            )
            if bool(args.semantic_merge_residual_safe_projection):
                cmd.extend(
                    [
                        "--semantic_merge_residual_safe_projection",
                        "1",
                        "--semantic_merge_residual_safe_projection_steps",
                        str(args.semantic_merge_residual_safe_projection_steps),
                    ]
                )
        promotion_gate_policy = str(args.semantic_merge_promotion_gate_policy or "none").strip().lower()
        if promotion_gate_policy not in {"", "none", "off"}:
            cmd.extend(
                [
                    "--semantic_merge_promotion_gate_policy",
                    promotion_gate_policy,
                    "--semantic_merge_promotion_qscale_min",
                    str(args.semantic_merge_promotion_qscale_min),
                    "--semantic_merge_promotion_random_qscale_gap_min",
                    str(args.semantic_merge_promotion_random_qscale_gap_min),
                ]
            )
        if bool(args.qscale_hold_refresh):
            cmd.extend(
                [
                    "--semantic_merge_qscale_hold_refresh",
                    "1",
                    "--semantic_merge_qscale_reference",
                    str(args.qscale_reference),
                    "--semantic_merge_qscale_min_factor",
                    str(args.qscale_min_factor),
                    "--semantic_merge_qscale_condition_reference",
                    str(args.qscale_condition_reference),
                    "--semantic_merge_qscale_residual_reference",
                    str(args.qscale_residual_reference),
                ]
            )
    online_scale_state_mode = str(args.online_scale_state_mode or "none").strip().lower()
    if online_scale_state_mode not in {"", "none", "off"}:
        cmd.extend(
            [
                "--online_scale_state_mode",
                online_scale_state_mode,
                "--online_scale_state_min",
                str(args.online_scale_state_min),
                "--online_scale_state_max",
                str(args.online_scale_state_max),
            ]
        )
        if bool(args.online_scale_state_pre_guard):
            cmd.extend(["--online_scale_state_pre_guard", "1"])
        gate_policy = str(args.online_scale_state_gate_policy or "none").strip().lower()
        if gate_policy not in {"", "none", "off"}:
            cmd.extend(
                [
                    "--online_scale_state_gate_policy",
                    gate_policy,
                    "--online_scale_state_qscale_min",
                    str(args.online_scale_state_qscale_min),
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
    if bool(job.get("disable_ttt_compile", False)):
        env["LOGER_TTT_DISABLE_COMPILE"] = "1"
    if bool(job.get("ttt_tail_drop", False)):
        env["TTT_WRITE_POST_ZP_SUMMARY"] = "1"
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
            "merge_state_trace": str(out_dir / "merge_state_trace.jsonl"),
        }
    )
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunks", default=",".join(str(c) for c in DEFAULT_CHUNKS))
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
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
    parser.add_argument("--blend-alpha", type=float, default=0.60)
    parser.add_argument("--max-blend-log-scale-delta", type=float, default=0.07)
    parser.add_argument("--clip-blend-to-limit", action="store_true")
    parser.add_argument("--semantic-conf-min", type=float, default=0.05)
    parser.add_argument("--semantic-merge-max-points", type=int, default=12000)
    parser.add_argument("--semantic-merge-overlap-support-dir", default="")
    parser.add_argument("--semantic-merge-overlap-support-kind", default="source_gate")
    parser.add_argument("--semantic-merge-overlap-support-floor", type=float, default=0.25)
    parser.add_argument("--radio-sidecar-dir", default="results/kitti_preprocess/01/radio_sidecar_chunks_r5_overlap")
    parser.add_argument("--reject-worse-than-native-overlap", action="store_true")
    parser.add_argument("--native-overlap-tolerance", type=float, default=0.0)
    parser.add_argument("--semantic-merge-residual-safe-projection", action="store_true")
    parser.add_argument("--semantic-merge-residual-safe-projection-steps", type=int, default=16)
    parser.add_argument("--semantic-merge-promotion-gate-policy", default="none")
    parser.add_argument("--semantic-merge-promotion-qscale-min", type=float, default=0.0)
    parser.add_argument("--semantic-merge-promotion-random-qscale-gap-min", type=float, default=0.02)
    parser.add_argument("--qscale-hold-refresh", action="store_true")
    parser.add_argument("--qscale-reference", type=float, default=0.35)
    parser.add_argument("--qscale-min-factor", type=float, default=0.35)
    parser.add_argument("--qscale-condition-reference", type=float, default=0.0)
    parser.add_argument("--qscale-residual-reference", type=float, default=0.0)
    parser.add_argument("--online-scale-state-mode", default="none")
    parser.add_argument("--online-scale-state-min", type=float, default=0.80)
    parser.add_argument("--online-scale-state-max", type=float, default=1.25)
    parser.add_argument(
        "--online-scale-state-pre-guard",
        action="store_true",
        help="Pass --online_scale_state_pre_guard=1 so scale-state is applied before the native-overlap rejection guard.",
    )
    parser.add_argument("--online-scale-state-gate-policy", default="none")
    parser.add_argument("--online-scale-state-qscale-min", type=float, default=0.50)
    parser.add_argument(
        "--disable-ttt-compile",
        action="store_true",
        help="Set LOGER_TTT_DISABLE_COMPILE=1 for combo TTT-tail-drop cases to reduce CUDA memory spikes.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    unknown = [case for case in cases if case not in MERGE_CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(MERGE_CASES)}")

    jobs: List[Dict[str, Any]] = []
    gpu_cursor = 0
    stride = int(args.chunk_size) - int(args.chunk_overlap)
    for chunk in chunks:
        start = max(0, (int(chunk) - 1) * stride)
        end = int(chunk) * stride + int(args.chunk_size)
        for case in cases:
            out_dir = args.output_root / f"chunk{chunk:02d}" / case
            cmd = _build_command(args, chunk=chunk, case=case, out_dir=out_dir)
            if args.skip_existing and (out_dir / "01.txt").exists() and (out_dir / "run.log").exists():
                continue
            jobs.append(
                {
                    "chunk": int(chunk),
                    "window_start": int(start),
                    "window_end": int(end),
                    "global_chunk_offset": max(0, int(chunk) - 1),
                    "case": case,
                    "strategy": MERGE_CASES[case]["strategy"],
                    "ttt_tail_drop": bool(MERGE_CASES[case].get("ttt_tail_drop", False)),
                    "gpu": int(gpus[gpu_cursor % len(gpus)]),
                    "out_dir": str(out_dir),
                    "cmd": cmd,
                    "workdir": str(args.workdir),
                    "cuda_alloc_conf": args.cuda_alloc_conf,
                    "disable_ttt_compile": bool(args.disable_ttt_compile),
                }
            )
            gpu_cursor += 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "phaseE_merge_run_manifest.json"
    manifest: Dict[str, Any] = {
        "output_root": str(args.output_root),
        "chunks": chunks,
        "cases": cases,
        "gpus": gpus,
        "job_count": len(jobs),
        "jobs": jobs,
        "dry_run": bool(args.dry_run),
        "skip_existing": bool(args.skip_existing),
        "rule_note": "Each target chunk runs a two-window segment [chunk-1, chunk] so the target chunk has a previous overlap for MERGE.",
    }
    if args.dry_run or not jobs:
        manifest_path.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(_jsonable({k: v for k, v in manifest.items() if k != "jobs"}), ensure_ascii=False, indent=2, sort_keys=True))
        print(f"wrote_manifest={manifest_path}")
        return

    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        future_to_job = {pool.submit(_run_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "chunk": result["chunk"],
                        "case": result["case"],
                        "gpu": result["gpu"],
                        "returncode": result["returncode"],
                        "duration_sec": result["duration_sec"],
                        "out_dir": result["out_dir"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    results_sorted = sorted(results, key=lambda row: (int(row["chunk"]), str(row["case"])))
    manifest["jobs"] = results_sorted
    manifest["failed_jobs"] = [row for row in results_sorted if int(row.get("returncode", 1)) != 0]
    manifest["job_count"] = len(results_sorted)
    manifest_path.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote_manifest={manifest_path}")
    if manifest["failed_jobs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
