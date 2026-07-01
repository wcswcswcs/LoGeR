#!/usr/bin/env python3
"""Run ACL2 v80 Phase3 short READ action/control jobs.

This wrapper uses the existing LoGeR READ actuator family that is already
implemented in ``HybridMemoryController``: v78 L07/L13 semantic read cues plus
geometry/shuffle/random controls. It is intentionally named as an existing
actuator run, because the v80 plan's newer QK-pair case names are not yet a
separate implementation in the codebase.
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
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase3_short_read_existing_actuator_control/rollouts"
)
DEFAULT_TARGET_CSV = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase1_three_memory_case_bank/short_single_chunk_cases.csv"
)
DEFAULT_SEQUENCE_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_STAGE_C_ROOT = Path("results/kitti_preprocess")
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


CASES: dict[str, dict[str, str]] = {
    "READ0_NATIVE": {
        "role": "baseline_native_no_read",
        "read_path": "none",
        "enable_frame_read_control": "0",
        "cue": "dyn",
        "semantic_contract": "no semantic READ action; native read-path-isolation baseline",
    },
    "READ1_EXISTING_L07_LAYOUT_SELECT": {
        "role": "candidate",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_action_only",
        "semantic_contract": "existing v78 L07 semantic layout/motion cue selects frame READ mass",
    },
    "READ3_EXISTING_L13_STABLE_PROTECT": {
        "role": "candidate_stable_protect",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_mask_l13_stable",
        "semantic_contract": (
            "existing v79/v78 stable-structure protection cue; Track H repair candidate for "
            "READ1 good-control overreach, still using the existing frame READ actuator"
        ),
    },
    "READ2_EXISTING_QK_SEM_EXP_GG_QKVAR_L05": {
        "role": "qk_candidate_existing_cue",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "mix.exp_add_gg_qkvar_l05",
        "semantic_contract": (
            "existing READ cue that mixes explicit semantic evidence with global Q/K variance; "
            "used as a minimal proxy for v80 READ_QK_PAIR_BIAS, not a dedicated new actuator"
        ),
    },
    "READ12_GG_SMD_A1B1G1": {
        "role": "v95_internal_proxy_candidate",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "gg.smd.product.a1b1g1.robustq",
        "semantic_contract": (
            "v95 Track D internal proxy cue selected by full_v2 cue-source audit; "
            "uses existing frame READ actuator and does not claim raw READ Q/K compatibility"
        ),
    },
    "READ13_FA_KEY_ALL_RUNTIME": {
        "role": "v95_trackH_internal_runtime_candidate",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "fa.key.all.high.robustq",
        "semantic_contract": (
            "v95 Track H runtime internal feature cue motivated by cue-gated READ1 upper-bound audit; "
            "uses frame-attention key compatibility directly and does not use offline case-id gating"
        ),
    },
    "READ14_FA_KEY_MIDDLE_RUNTIME": {
        "role": "v95_trackH_internal_runtime_candidate",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "fa.key.middle.high.robustq",
        "semantic_contract": (
            "v95 Track H runtime internal feature cue paired with FA_KEY_ALL upper-bound sibling; "
            "uses middle-layer frame-attention key compatibility directly and does not use offline case-id gating"
        ),
    },
    "READ15_GATE_FA_KEY_ALL_Q60_THEN_L07": {
        "role": "v95_trackH_runtime_gated_candidate",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "gate.fa_key_all_mean_ge_q60.then_v78_l07",
        "semantic_contract": (
            "v95 Track H runtime gate: enable the existing v78 L07 READ action only when the internal "
            "FA-key-all cue mean crosses the q60 threshold found by the diagnostic cue bank; inactive "
            "chunks use zero frame READ bias, with no offline case-id gating at runtime"
        ),
    },
    "READ16_GATE_FA_KEY_MIDDLE_Q60_THEN_L07": {
        "role": "v95_trackH_runtime_gated_candidate",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "gate.fa_key_middle_mean_ge_q60.then_v78_l07",
        "semantic_contract": (
            "v95 Track H runtime gate: enable the existing v78 L07 READ action only when the internal "
            "FA-key-middle cue mean crosses the q60 threshold found by the diagnostic cue bank; inactive "
            "chunks use strict no-op frame READ control, with no offline case-id gating at runtime"
        ),
    },
    "READ17_GATE_GG_QK_SHALLOW_Q40_THEN_L07": {
        "role": "v95_trackH_runtime_gated_candidate",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "gate.gg_qk_shallow_mean_le_q40.then_v78_l07",
        "semantic_contract": (
            "v95 Track H runtime gate: enable the existing v78 L07 READ action only when the internal "
            "GG-QK-shallow cue mean is at or below the q40 threshold found by the diagnostic cue bank; inactive "
            "chunks use strict no-op frame READ control, with no offline case-id gating at runtime"
        ),
    },
    "READ18_GATE_GG_QK_SHALLOW_Q40_CHUNKGE6_THEN_L07": {
        "role": "v95_trackH_runtime_gated_candidate_warmup_repair",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "gate.gg_qk_shallow_mean_le_q40.chunk_ge_6.then_v78_l07",
        "semantic_contract": (
            "v95 Track H runtime gate repair after Stage7 READ17 early false positives: enable the existing "
            "v78 L07 READ action only when chunk_idx >= 6 and the internal GG-QK-shallow cue mean is at or "
            "below the q40 threshold; inactive chunks use strict no-op frame READ control"
        ),
    },
    "READ21_GATE_GG_QK_SHALLOW_Q40_CHUNKGE6_RTOK005_THEN_L07": {
        "role": "v95_trackH_runtime_gated_candidate_rtok_repair",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "gate.gg_qk_shallow_mean_le_q40.chunk_ge_6.rtok_ge_0p005.then_v78_l07",
        "semantic_contract": (
            "v95 Track H runtime gate repair after READ18/strength-sweep full No-Go: enable the existing "
            "v78 L07 READ action only when chunk_idx >= 6, GG-QK-shallow mean is at or below q40, and "
            "the runtime prior mean_R_tok support is at least 0.005; inactive chunks use strict no-op "
            "frame READ control"
        ),
    },
    "READ2_DEDICATED_QK_PAIR_BIAS": {
        "role": "qk_candidate_dedicated_pair_bias",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_action_only",
        "frame_bias_mode": "qk_pair_stable_harm",
        "semantic_contract": (
            "dedicated pair-level READ_QK_PAIR_BIAS smoke: risky query tokens boost stable keys "
            "and veto harmful keys using the selected semantic READ cue as the query/key risk carrier"
        ),
    },
    "READ2_QK_KEYSTABLE_PAIR_BIAS": {
        "role": "qk_candidate_keystable_pair_bias",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_action_only",
        "frame_bias_mode": "qk_pair_key_stability",
        "semantic_contract": (
            "pair-level READ_QK bias with semantic READ risk on queries and frame-attention key stability "
            "on keys; tests whether semantic error regions can explain harmful TTT/write targets"
        ),
    },
    "READ7_GEOMETRY_ONLY_CONTROL": {
        "role": "geometry_only_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_action_only.geometry_only",
        "semantic_contract": "same existing actuator family with semantic content removed by geometry-only control",
    },
    "READ7_EXISTING_QK_GEOM_QKVAR_CONTROL": {
        "role": "qk_geometry_only_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "gg.qk.middle.var.robustq",
        "semantic_contract": "pure existing global Q/K variance read cue; geometry/internal-attention control for READ2 QK proxy",
    },
    "READ7_DEDICATED_QK_PAIR_GEOMETRY_CONTROL": {
        "role": "qk_pair_geometry_only_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "gg.qk.middle.var.robustq",
        "frame_bias_mode": "qk_pair_stable_harm",
        "semantic_contract": "same pair-level QK actuator driven by pure global Q/K variance rather than semantic READ cue",
    },
    "READ7_QK_KEYSTABLE_GEOMETRY_CONTROL": {
        "role": "qk_pair_keystable_geometry_only_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "gg.qk.middle.var.robustq",
        "frame_bias_mode": "qk_pair_key_stability",
        "semantic_contract": "same key-stability pair actuator driven by pure global Q/K variance rather than semantic READ cue",
    },
    "READ8_LABEL_SHUFFLE": {
        "role": "semantic_shuffle_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_action_only.label_shuffled",
        "semantic_contract": "same existing actuator family with labels deterministically shuffled",
    },
    "READ9_CONFIDENCE_SHUFFLE": {
        "role": "semantic_shuffle_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_action_only.confidence_shuffled",
        "semantic_contract": "same existing actuator family with confidence deterministically shuffled",
    },
    "READ10_SAME_READ_MASS_RANDOM": {
        "role": "same_read_mass_random_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_action_only.same_attention_mass_random",
        "semantic_contract": "same read-mass randomized control for the existing actuator family",
    },
    "READ10_EXISTING_QK_RANDOM_SAME_MASS": {
        "role": "qk_same_mass_random_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "random",
        "semantic_contract": "random read cue under the same calibration/topk settings as the existing QK proxy smoke",
    },
    "READ10_DEDICATED_QK_PAIR_RANDOM_SAME_MASS": {
        "role": "qk_pair_same_mass_random_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_action_only",
        "frame_bias_mode": "qk_pair_random_same_mass",
        "semantic_contract": (
            "same semantic READ cue as READ2_DEDICATED_QK_PAIR_BIAS, but pair logits are deterministically "
            "shuffled per frame to preserve pair-score mass while breaking query-key correspondence"
        ),
    },
    "READ10_QK_KEYSTABLE_RANDOM_SAME_MASS": {
        "role": "qk_pair_keystable_same_mass_random_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_action_only",
        "frame_bias_mode": "qk_pair_key_stability_random_same_mass",
        "semantic_contract": (
            "same semantic READ query risk and key-stability scores as READ2_QK_KEYSTABLE_PAIR_BIAS, "
            "but deterministic pair-score shuffle breaks query-key correspondence"
        ),
    },
    "READ11_GROUP_STRATIFIED_RANDOM": {
        "role": "group_stratified_random_control",
        "read_path": "frame",
        "enable_frame_read_control": "1",
        "cue": "v78.l07_l13.l07_action_only.group_stratified_random",
        "semantic_contract": "semantic-group-stratified randomized control for the existing actuator family",
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


def _parse_csv_text(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text or "").split(",") if part.strip()]


def _selected_cases_include_dedicated_qk_pair(cases: list[str]) -> bool:
    return any("DEDICATED_QK_PAIR" in str(case) or "QK_KEYSTABLE" in str(case) for case in cases)


def _seq_norm(value: Any) -> str:
    return f"{int(str(value).strip()):02d}"


def _select_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    seqs = {_seq_norm(seq) for seq in _parse_csv_text(args.seqs)}
    case_types = {str(case_type).strip() for case_type in _parse_csv_text(args.case_types)}
    max_per_bucket = int(args.max_targets_per_case_type_per_seq)
    counts: dict[tuple[str, str], int] = {}
    selected: list[dict[str, Any]] = []
    with args.target_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq = _seq_norm(row.get("seq") or row.get("sequence"))
            case_type = str(row.get("case_type", "")).strip()
            if seqs and seq not in seqs:
                continue
            if case_types and case_type not in case_types:
                continue
            bucket = (seq, case_type)
            if max_per_bucket > 0 and counts.get(bucket, 0) >= max_per_bucket:
                continue
            target = dict(row)
            target["seq"] = seq
            target["case_type"] = case_type
            target["chunk_id"] = int(row["chunk_id"])
            target["frame_start"] = int(row["frame_start"])
            target["frame_end"] = int(row["frame_end"])
            selected.append(target)
            counts[bucket] = counts.get(bucket, 0) + 1
            if int(args.max_targets_total) > 0 and len(selected) >= int(args.max_targets_total):
                break
    if not selected:
        raise ValueError(
            f"no targets selected from {args.target_csv} seqs={sorted(seqs)} case_types={sorted(case_types)}"
        )
    return selected


def _build_command(args: argparse.Namespace, *, target: dict[str, Any], case: str, out_dir: Path) -> list[str]:
    cfg = CASES[case]
    seq = str(target["seq"])
    chunk = int(target["chunk_id"])
    start = int(target["frame_start"])
    end = int(target["frame_end"])
    if args.python_bin:
        cmd = [str(args.python_bin), "run_pipeline_abc_v2.py"]
    else:
        cmd = [
            str(args.conda),
            "run",
            "-n",
            args.conda_env,
            "python",
            "run_pipeline_abc_v2.py",
        ]
    cmd.extend(
        [
            "--input",
        str(args.sequence_root / seq / "image_2"),
        "--output_video",
        "",
        "--output_txt",
        str(out_dir / f"{seq}.txt"),
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
        str(args.stage_c_root / seq / "stage_c_cache_semantic_chunks"),
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
    )
    if case != "READ0_NATIVE":
        cmd.extend(
            [
                "--beta_frame",
                str(args.beta_frame),
                "--frame_bias_mode",
                str(cfg.get("frame_bias_mode", args.frame_bias_mode)),
                "--frame_attention_record_bias_mass",
                "1" if args.frame_attention_record_bias_mass else "0",
                "--frame_attention_bias_mass_max_queries",
                str(args.frame_attention_bias_mass_max_queries),
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
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
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
            "trajectory": str(out_dir / f"{job['seq']}.txt"),
            "hmc_state_hash": str(out_dir / "hmc_state_hash.jsonl"),
        }
    )
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--seqs", default="00,01,02,05")
    parser.add_argument("--case-types", default="bad,good")
    parser.add_argument("--max-targets-per-case-type-per-seq", type=int, default=0)
    parser.add_argument("--max-targets-total", type=int, default=0)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=None,
        help="Run pipeline directly with this Python interpreter instead of conda run.",
    )
    parser.add_argument("--sequence-root", type=Path, default=DEFAULT_SEQUENCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage-c-root", type=Path, default=DEFAULT_STAGE_C_ROOT)
    parser.add_argument("--cuda-alloc-conf", default=DEFAULT_CUDA_ALLOC_CONF)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--beta-frame", type=float, default=0.5)
    parser.add_argument(
        "--frame-bias-mode",
        choices=(
            "pair",
            "protected_pair",
            "key",
            "query",
            "qk_pair_stable_harm",
            "qk_pair_random_same_mass",
            "read_qk_pair_bias",
            "qk_pair_key_stability",
            "qk_pair_key_stability_random_same_mass",
        ),
        default="key",
    )
    parser.add_argument("--frame-attention-record-bias-mass", action="store_true")
    parser.add_argument("--frame-attention-bias-mass-max-queries", type=int, default=64)
    parser.add_argument("--read-target-mass", type=float, default=0.10)
    parser.add_argument("--read-calib-tau", type=float, default=0.05)
    parser.add_argument("--read-blend-lambda", type=float, default=0.50)
    parser.add_argument("--read-topk-frac", type=float, default=0.10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = _parse_csv_ints(args.gpus)
    cases = _parse_csv_text(args.cases)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    unknown = [case for case in cases if case not in CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(CASES)}")
    targets = _select_targets(args)

    jobs: list[dict[str, Any]] = []
    gpu_cursor = 0
    for target in targets:
        seq = str(target["seq"])
        chunk = int(target["chunk_id"])
        case_type = str(target["case_type"])
        for case in cases:
            out_dir = args.output_root / f"seq{seq}" / f"chunk{chunk:03d}_{case_type}" / case
            cmd = _build_command(args, target=target, case=case, out_dir=out_dir)
            skipped = bool(
                args.skip_existing
                and (out_dir / f"{seq}.txt").exists()
                and (out_dir / "hmc_state_hash.jsonl").exists()
            )
            job = {
                "job_index": len(jobs),
                "seq": seq,
                "chunk": int(chunk),
                "case_type": case_type,
                "frame_start": int(target["frame_start"]),
                "frame_end": int(target["frame_end"]),
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
                "trajectory": str(out_dir / f"{seq}.txt"),
                "hmc_state_hash": str(out_dir / "hmc_state_hash.jsonl"),
                "skipped": skipped,
                "returncode": 0 if skipped else None,
            }
            gpu_cursor += 1
            jobs.append(job)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "phase3_short_read_existing_actuator_run_manifest.json"
    manifest: dict[str, Any] = {
        "args": _jsonable(vars(args)),
        "case_definitions": _jsonable(CASES),
        "selected_targets": _jsonable(targets),
        "jobs": _jsonable(jobs),
        "planned_jobs": len(jobs),
        "method_gate_claimed": False,
        "qk_pair_actuator_claimed": bool(_selected_cases_include_dedicated_qk_pair(cases)),
        "note": (
            "Phase3 READ run manifest. Dedicated QK-pair cases use the runtime frame_bias_mode="
            "qk_pair_* attention-logit actuator; existing READ cases remain legacy/proxy actuators."
        ),
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"planned_jobs={len(jobs)} manifest={manifest_path}")
    if args.dry_run:
        print(json.dumps({"planned_jobs": len(jobs), "dry_run": True}, ensure_ascii=False, sort_keys=True))
        return

    run_jobs = [dict(job) for job in jobs if not job["skipped"]]
    completed_by_index: dict[int, dict[str, Any]] = {int(job["job_index"]): dict(job) for job in jobs if job["skipped"]}
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
            completed_by_index[int(result["job_index"])] = result
            print(
                "done",
                f"seq={result['seq']}",
                f"chunk={result['chunk']}",
                f"case_type={result['case_type']}",
                f"case={result['case']}",
                f"gpu={result['gpu']}",
                f"returncode={result['returncode']}",
                f"duration_sec={result['duration_sec']:.1f}",
                flush=True,
            )
            manifest["completed_count"] = len(completed_by_index)
            manifest["completed_jobs"] = _jsonable([completed_by_index[idx] for idx in sorted(completed_by_index)])
            manifest_path.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if queued_by_gpu.get(gpu):
                futures[pool.submit(_run_job, queued_by_gpu[gpu].pop(0))] = gpu

    ordered = [completed_by_index.get(int(job["job_index"]), job) for job in jobs]
    failed = [job for job in ordered if int(job.get("returncode") or 0) != 0]
    manifest["jobs"] = _jsonable(ordered)
    manifest["completed_count"] = len(completed_by_index)
    manifest["failed_jobs"] = _jsonable(failed)
    manifest_path.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable({"planned_jobs": len(jobs), "completed_count": len(completed_by_index), "failed_jobs": len(failed)}), ensure_ascii=False, sort_keys=True))
    if failed:
        raise SystemExit(f"failed_jobs={len(failed)}; see {manifest_path}")


if __name__ == "__main__":
    main()
