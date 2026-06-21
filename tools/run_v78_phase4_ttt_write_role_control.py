#!/usr/bin/env python3
"""Run ACL2 v78 Phase4 TTT write-role control smokes and summarize gates."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_v68_phaseD_read_smoke import (
    LOWER_IS_BETTER_KEYS,
    _eval_run,
    _finite,
    _load_kitti_gt,
    _safe_ratio_improvement,
    _write_csv,
)


DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/"
    "report_final/phase4_ttt_write_update/smoke_chunk06_context2_role_control_v1"
)
DEFAULT_INPUT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2")
DEFAULT_GT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_STAGE_C_CACHE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"
DEFAULT_VISUAL_AUDIT = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase4_ttt_write_update/visual_smoke_chunk06_output_separated_r3/visual_integrity_audit.json"
)

MECHANISM_KEYS = [
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]


PHASE4_CASES: Dict[str, Dict[str, Any]] = {
    "T0_NATIVE_READPATH": {
        "hybrid_memory_mode": "read_path_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.0,
        "neutral_scale": 1.0,
        "negative_scale": 1.0,
    },
    "T0B_TTT_UNITY_PROBE": {
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "noop",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.0,
        "neutral_scale": 1.0,
        "negative_scale": 1.0,
    },
    "T1_TTT_STABLE_WRITE_FLOOR": {
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 1.0,
    },
    "T2_TTT_HARM_WRITE_DAMP": {
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.0,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
    },
    "T3_TTT_STABLE_POS_PLUS_HARM_DAMP": {
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
    },
    "T4_TTT_RANDOM_SAME_MASS_T3": {
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
    },
    "T5_TTT_SEMANTIC_SHUFFLE_T3": {
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "semantic_shuffle",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
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
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: Iterable[Any]) -> Optional[float]:
    xs = [_finite(v) for v in values]
    xs = [v for v in xs if v is not None]
    return float(np.mean(xs)) if xs else None


def _max(values: Iterable[Any]) -> Optional[float]:
    xs = [_finite(v) for v in values]
    xs = [v for v in xs if v is not None]
    return float(np.max(xs)) if xs else None


def _unique_present(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for value in values:
        if value is None or value == "":
            continue
        key = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _aggregate_phase4_hmc(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl_all(run_dir / "hmc_state_hash.jsonl")
    if not rows:
        return {"phase4_hmc_rows": 0}
    spatial_paths = [
        str(row.get("ttt_spatial_post_delta_map_dump_path"))
        for row in rows
        if row.get("ttt_spatial_post_delta_map_dump_path")
    ]
    role_stats = [row.get("prior_ttt_semantic_write_role_stats") for row in rows if row.get("prior_ttt_semantic_write_role_stats")]
    hash_values = _unique_present(row.get("probe_ttt_write_state_hash") for row in rows)
    return {
        "phase4_hmc_rows": int(len(rows)),
        "phase4_prior_ttt_write_present_count": int(sum(1 for row in rows if bool(row.get("prior_ttt_write_present")))),
        "phase4_prior_ttt_write_mean_avg": _mean(row.get("prior_ttt_write_mean") for row in rows),
        "phase4_semantic_role_policy_set": _unique_present(row.get("prior_semantic_role_policy") for row in rows),
        "phase4_semantic_memory_paths_set": _unique_present(row.get("prior_semantic_memory_paths") for row in rows),
        "phase4_semantic_role_control_mode_set": _unique_present(row.get("prior_semantic_role_control_mode") for row in rows),
        "phase4_semantic_role_control_applied_count": int(
            sum(1 for row in rows if bool(row.get("prior_semantic_role_control_applied")))
        ),
        "phase4_semantic_role_control_changed_fraction_mean": _mean(
            row.get("prior_semantic_role_control_changed_fraction") for row in rows
        ),
        "phase4_R_ttt_role_counts_last": rows[-1].get("prior_R_ttt_role_counts"),
        "phase4_R_ttt_role_counts_before_control_last": rows[-1].get("prior_R_ttt_role_counts_before_control"),
        "phase4_R_ttt_role_counts_after_control_last": rows[-1].get("prior_R_ttt_role_counts_after_control"),
        "phase4_ttt_semantic_write_role_stats_last": role_stats[-1] if role_stats else None,
        "phase4_ttt_semantic_write_role_max_abs_rel_change_max": _max(
            row.get("prior_ttt_semantic_write_role_max_abs_rel_change") for row in rows
        ),
        "phase4_ttt_semantic_write_role_intended_change_ge20_any": bool(
            any(bool(row.get("prior_ttt_semantic_write_role_intended_change_ge20")) for row in rows)
        ),
        "phase4_probe_ttt_write_debug_available_count": int(
            sum(1 for row in rows if bool(row.get("probe_ttt_write_debug_available")))
        ),
        "phase4_probe_ttt_write_post_delta_norm_mean_avg": _mean(
            row.get("probe_ttt_write_post_delta_norm_mean") for row in rows
        ),
        "phase4_probe_ttt_write_action_delta_norm_mean_avg": _mean(
            row.get("probe_ttt_write_action_delta_norm_mean") for row in rows
        ),
        "phase4_probe_ttt_write_native_delta_norm_mean_avg": _mean(
            row.get("probe_ttt_write_native_delta_norm_mean") for row in rows
        ),
        "phase4_probe_ttt_write_state_hash_set": hash_values,
        "phase4_probe_ttt_write_state_hash_count": int(len(hash_values)),
        "phase4_hash_H_next_set": _unique_present(row.get("hash_H_next") for row in rows),
        "phase4_commit_source_state_hash_set": _unique_present(row.get("commit_source_state_hash") for row in rows),
        "phase4_spatial_post_delta_status_set": _unique_present(
            row.get("ttt_spatial_post_delta_map_dump_status") for row in rows
        ),
        "phase4_spatial_post_delta_map_paths": spatial_paths,
        "phase4_spatial_post_delta_map_count": int(len(spatial_paths)),
        "phase4_state_double_write_safe_all": bool(all(bool(row.get("state_double_write_safe", True)) for row in rows)),
        "phase4_probe_no_commit_hash_equal_all": bool(
            all(bool(row.get("probe_no_commit_hash_equal", True)) for row in rows)
        ),
    }


def _load_visual_gate(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"visual_audit_path": str(path), "visual_gate_pass": False, "visual_gate_reason": "missing_visual_audit"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "gate_pass": True,
        "operator_update_final_separated": True,
        "semantic_label_present": True,
        "D_geo_present": True,
        "post_zp_delta_present": True,
        "same_write_mass_random_present": True,
    }
    ok = all(bool(payload.get(key)) is bool(value) for key, value in required.items())
    return {
        "visual_audit_path": str(path),
        "visual_gate_pass": bool(ok),
        "visual_gate_payload": payload,
    }


def _best_control_value(rows_by_name: Dict[str, Dict[str, Any]], controls: Sequence[str], key: str) -> Optional[float]:
    vals = [_finite(rows_by_name.get(name, {}).get(key)) for name in controls]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def _build_phase4_decision(
    rows: Sequence[Dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    controls: Sequence[str],
    visual_gate: Dict[str, Any],
    min_mechanism_improvement: float,
) -> Dict[str, Any]:
    rows_by_name = {str(row["run"]): row for row in rows}
    cand = rows_by_name.get(candidate)
    base = rows_by_name.get(baseline)
    if cand is None:
        return {"phase4_gate_pass": False, "candidate": candidate, "reason": f"missing_candidate:{candidate}"}
    if base is None:
        return {"phase4_gate_pass": False, "candidate": candidate, "reason": f"missing_baseline:{baseline}"}

    comparisons: Dict[str, Dict[str, Any]] = {}
    metric_passes: List[str] = []
    for key in LOWER_IS_BETTER_KEYS:
        cand_v = _finite(cand.get(key))
        base_v = _finite(base.get(key))
        best_ctrl = _best_control_value(rows_by_name, controls, key)
        beats_controls = cand_v is not None and best_ctrl is not None and cand_v < best_ctrl
        ratio = _safe_ratio_improvement(base_v, cand_v)
        mechanism_key = key in MECHANISM_KEYS
        key_pass = bool(mechanism_key and beats_controls and ratio is not None and ratio >= min_mechanism_improvement)
        if key_pass:
            metric_passes.append(key)
        comparisons[key] = {
            "candidate": cand_v,
            "baseline": base_v,
            "best_control": best_ctrl,
            "candidate_minus_baseline": (cand_v - base_v) if cand_v is not None and base_v is not None else None,
            "candidate_minus_best_control": (cand_v - best_ctrl) if cand_v is not None and best_ctrl is not None else None,
            "improvement_vs_baseline_ratio": ratio,
            "beats_controls": beats_controls,
            "mechanism_key": mechanism_key,
            "phase4_metric_key_pass": key_pass,
        }

    base_hashes = set(str(x) for x in (base.get("phase4_probe_ttt_write_state_hash_set") or []))
    cand_hashes = set(str(x) for x in (cand.get("phase4_probe_ttt_write_state_hash_set") or []))
    hash_changes = bool(cand_hashes and cand_hashes != base_hashes)
    write_change = bool(cand.get("phase4_ttt_semantic_write_role_intended_change_ge20_any"))
    post_delta = (
        "saved" in {str(x) for x in (cand.get("phase4_spatial_post_delta_status_set") or [])}
        and _finite(cand.get("phase4_probe_ttt_write_action_delta_norm_mean_avg")) is not None
        and float(cand.get("phase4_probe_ttt_write_action_delta_norm_mean_avg")) > 0.0
    )
    visual_ok = bool(visual_gate.get("visual_gate_pass"))
    gate_pass = bool(visual_ok and write_change and post_delta and hash_changes and metric_passes)
    return {
        "phase4_gate_pass": gate_pass,
        "candidate": candidate,
        "baseline": baseline,
        "controls": list(controls),
        "metric_passes": metric_passes,
        "visual_gate_pass": visual_ok,
        "write_mass_intended_group_changes_ge20": write_change,
        "post_zp_delta_changes": post_delta,
        "next_probe_state_hash_changes": hash_changes,
        "comparisons": comparisons,
        "rule": (
            "Requires visual gate, operator/update/final separation, intended write-role change >=20%, "
            "post-zp delta/action change, next probe state hash change, and >=10% improvement in "
            "head_tail/future/scale_cv while beating same-write-mass random and semantic shuffle controls."
        ),
    }


def _build_command(args: argparse.Namespace, *, chunk: int, case: str, out_dir: Path) -> List[str]:
    cfg = PHASE4_CASES[case]
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
        str(cfg["hybrid_memory_mode"]),
        "--hmc_commit_mode",
        str(cfg["hmc_commit_mode"]),
        "--semantic_prior_mode",
        str(cfg["semantic_prior_mode"]),
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
        "--semantic_role_policy",
        str(cfg["semantic_role_policy"]),
        "--semantic_memory_paths",
        str(cfg["semantic_memory_paths"]),
        "--semantic_role_control_mode",
        str(cfg["semantic_role_control_mode"]),
        "--semantic_role_control_seed",
        str(args.semantic_role_control_seed),
        "--semantic_role_highd_quantile",
        str(args.semantic_role_highd_quantile),
        "--semantic_role_low_trust",
        str(args.semantic_role_low_trust),
        "--semantic_role_positive_scale",
        str(cfg["positive_scale"]),
        "--semantic_role_neutral_scale",
        str(cfg["neutral_scale"]),
        "--semantic_role_negative_scale",
        str(cfg["negative_scale"]),
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--ttt_spatial_post_delta_map_dump_dir",
        str(out_dir / "ttt_spatial_post_delta_maps"),
        "--ttt_spatial_post_delta_map_dump_dtype",
        "float16",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
    ]
    if bool(cfg.get("hmc_ignore_semantic_prior", False)):
        cmd.extend(["--hmc_ignore_semantic_prior", "1"])
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


def _evaluate(args: argparse.Namespace, ordered_jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    _, gt_poses_all, gt_pos_all = _load_kitti_gt(args.gt)
    rows: List[Dict[str, Any]] = []
    for job in ordered_jobs:
        if int(job.get("returncode") or 0) != 0:
            continue
        run_name = str(job["case"])
        run_dir = Path(job["out_dir"])
        row = _eval_run(run_name, run_dir, gt_poses_all, gt_pos_all)
        row.update(_aggregate_phase4_hmc(run_dir))
        rows.append(row)

    visual_gate = _load_visual_gate(args.visual_audit)
    controls = [case.strip() for case in str(args.controls).split(",") if case.strip()]
    candidates = [case.strip() for case in str(args.candidates).split(",") if case.strip()]
    decisions = {
        cand: _build_phase4_decision(
            rows,
            candidate=cand,
            baseline=str(args.baseline),
            controls=controls,
            visual_gate=visual_gate,
            min_mechanism_improvement=float(args.min_mechanism_improvement),
        )
        for cand in candidates
    }
    summary = {
        "schema": "acl2_v78_phase4_ttt_write_role_control_summary_v1",
        "output_root": str(args.output_root),
        "baseline": str(args.baseline),
        "controls": controls,
        "candidates": candidates,
        "visual_gate": visual_gate,
        "runs": rows,
        "decisions": decisions,
        "phase4_any_gate_pass": bool(any(bool(d.get("phase4_gate_pass")) for d in decisions.values())),
    }
    metrics_json = args.output_root / "phase4_ttt_write_role_metrics.json"
    metrics_csv = args.output_root / "phase4_ttt_write_role_metrics.csv"
    decision_json = args.output_root / "phase4_ttt_write_role_decision.json"
    metrics_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decision_json.write_text(
        json.dumps(_jsonable({"decisions": decisions, "phase4_any_gate_pass": summary["phase4_any_gate_pass"]}), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(metrics_csv, rows)
    print(json.dumps(_jsonable({"phase4_any_gate_pass": summary["phase4_any_gate_pass"], "decisions": decisions}), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={metrics_json}")
    print(f"wrote_csv={metrics_csv}")
    print(f"wrote_decision={decision_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunks", default="6")
    parser.add_argument("--cases", default=",".join(PHASE4_CASES))
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
    parser.add_argument("--semantic-role-highd-quantile", type=float, default=0.75)
    parser.add_argument("--semantic-role-low-trust", type=float, default=0.55)
    parser.add_argument("--semantic-role-control-seed", type=int, default=7804)
    parser.add_argument("--visual-audit", type=Path, default=DEFAULT_VISUAL_AUDIT)
    parser.add_argument("--baseline", default="T0B_TTT_UNITY_PROBE")
    parser.add_argument("--controls", default="T4_TTT_RANDOM_SAME_MASS_T3,T5_TTT_SEMANTIC_SHUFFLE_T3")
    parser.add_argument("--candidates", default="T1_TTT_STABLE_WRITE_FLOOR,T2_TTT_HARM_WRITE_DAMP,T3_TTT_STABLE_POS_PLUS_HARM_DAMP")
    parser.add_argument("--min-mechanism-improvement", type=float, default=0.10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    cases = [case.strip() for case in str(args.cases).split(",") if case.strip()]
    unknown = [case for case in cases if case not in PHASE4_CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(PHASE4_CASES)}")

    jobs: List[Dict[str, Any]] = []
    gpu_cursor = 0
    for chunk in chunks:
        window = _context_window(args, int(chunk))
        for case in cases:
            out_dir = args.output_root / f"chunk{int(chunk):02d}" / case
            cmd = _build_command(args, chunk=int(chunk), case=case, out_dir=out_dir)
            skipped = bool(args.skip_existing and (out_dir / "01.txt").exists() and (out_dir / "hmc_state_hash.jsonl").exists())
            cfg = PHASE4_CASES[case]
            jobs.append(
                {
                    "chunk": int(chunk),
                    **window,
                    "case": case,
                    "case_config": cfg,
                    "gpu": int(gpus[gpu_cursor % len(gpus)]),
                    "out_dir": str(out_dir),
                    "cmd": cmd,
                    "cmd_shell": shlex.join(cmd),
                    "workdir": str(args.workdir),
                    "cuda_alloc_conf": str(args.cuda_alloc_conf),
                    "skipped": skipped,
                    "returncode": 0 if skipped else None,
                }
            )
            gpu_cursor += 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "phase4_ttt_write_role_run_manifest.json"
    manifest: Dict[str, Any] = {"args": _jsonable(vars(args)), "jobs": jobs}
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

    def _run_gpu_queue(gpu: int, queue: List[Dict[str, Any]]) -> None:
        for job in queue:
            result = _run_job(job)
            with completed_lock:
                completed.append(result)
                manifest["jobs"] = completed + [item for item in jobs if item not in completed]
                manifest_path.write_text(
                    json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(
                f"[gpu{gpu}] chunk={result['chunk']} case={result['case']} "
                f"returncode={result['returncode']} duration={result['duration_sec']:.1f}s",
                flush=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(_run_gpu_queue, int(gpu), queue)
            for gpu, queue in jobs_by_gpu.items()
            if queue
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

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
    if not bool(args.no_evaluate):
        _evaluate(args, ordered)


if __name__ == "__main__":
    main()
