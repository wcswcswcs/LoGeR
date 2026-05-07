#!/usr/bin/env python3
"""Upload lightweight HMC v2 experiment artifacts to Weights & Biases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import wandb


def parse_kitti_benchmark(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text()
    metrics: Dict[str, Any] = {}
    rpe = re.search(r"RPE stats.*?\n01\s+([0-9.]+)\s+([0-9.]+)", text, re.S)
    if rpe:
        metrics["rpe_translation_percent"] = float(rpe.group(1))
        metrics["rpe_rotation_deg_per_100m"] = float(rpe.group(2))
    ate = re.search(r"ATE RMSE stats.*?\n01\s+([0-9.]+)\s+([0-9.]+)", text, re.S)
    if ate:
        metrics["ate_rmse_m"] = float(ate.group(1))
        metrics["ate_rotation_rmse_deg"] = float(ate.group(2))
    matched = re.search(r"matched poses:\s+(\d+)", text)
    if matched:
        metrics["matched_poses"] = int(matched.group(1))
    seq_len = re.search(r"sequence length \[m\]:\s+([0-9.]+)", text)
    if seq_len:
        metrics["sequence_length_m"] = float(seq_len.group(1))
    return metrics


def summarize_hmc_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return {"hmc/chunks": 0}
    metrics: Dict[str, Any] = {
        "hmc/chunks": len(rows),
        "hmc/probe_no_commit_true": sum(bool(r.get("probe_no_commit_hash_equal")) for r in rows),
        "hmc/state_double_write_safe_true": sum(bool(r.get("state_double_write_safe")) for r in rows),
    }
    for key in [
        "pass1_pass2_pose_translation_max_diff",
        "pass1_pass2_pose_matrix_max_abs_diff",
        "pass1_pass2_local_points_max_abs_diff",
        "pass1_pass2_world_points_max_abs_diff",
        "pass1_pass2_confidence_max_abs_diff",
    ]:
        metrics[f"hmc/{key}"] = max(float(r.get(key, 0) or 0) for r in rows)
    for hook in ["frame_attention", "swa_read", "ttt_apply", "chunk_attention"]:
        vals = [int((r.get("probe_hmc_hook_trace_counts") or {}).get(hook, 0)) for r in rows]
        if vals:
            metrics[f"hmc/hook_{hook}_min_count"] = min(vals)
            metrics[f"hmc/hook_{hook}_max_count"] = max(vals)
    commit_modes = [str(r.get("hmc_commit_mode", "")) for r in rows if r.get("hmc_commit_mode")]
    if commit_modes:
        metrics["hmc/commit_mode"] = commit_modes[0]
    for key in [
        "memory_ttt_mean_rel_diff",
        "memory_ttt_max_rel_diff",
        "memory_ttt_w0_mean_rel_diff",
        "memory_ttt_w1_mean_rel_diff",
        "memory_ttt_w2_mean_rel_diff",
    ]:
        vals = [float(r.get(key, 0.0) or 0.0) for r in rows if r.get(key) is not None]
        if vals:
            metrics[f"hmc/{key}_mean"] = sum(vals) / len(vals)
            metrics[f"hmc/{key}_max"] = max(vals)
    return metrics


def summarize_phase_b_dir(directory: Path) -> Dict[str, Any]:
    csv_path = directory / "phase_b_chunk_summary.csv"
    if not csv_path.exists():
        return {}
    tag = directory.name.replace("HMC_B_dashboard_segments_", "phase_b_")
    rows = list(csv.DictReader(csv_path.open()))
    metrics: Dict[str, Any] = {f"{tag}/chunks_summarized": len(rows)}
    for column in [
        "C_dyn_mean",
        "C_dyn_p90",
        "C_unc_mean",
        "C_anchor_mean",
        "G_write_mean",
        "attn_dyn_mean",
        "ttt_update_proxy_mean",
    ]:
        vals: List[float] = []
        for row in rows:
            raw = row.get(column, "")
            try:
                val = float(raw)
            except ValueError:
                continue
            if not math.isnan(val):
                vals.append(val)
        if vals:
            metrics[f"{tag}/{column}"] = sum(vals) / len(vals)
    availability = directory / "phase_b_trace_availability.json"
    if availability.exists():
        for key, value in json.loads(availability.read_text()).items():
            if isinstance(value, bool):
                metrics[f"{tag}/availability_{key}"] = int(value)
    return metrics


def add_artifact(run: wandb.sdk.wandb_run.Run, name: str, artifact_type: str, paths: Iterable[Path]) -> None:
    artifact = wandb.Artifact(name, type=artifact_type)
    for path in paths:
        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file() and file_path.suffix != ".pt":
                    artifact.add_file(str(file_path), name=str(Path(path.name) / file_path.relative_to(path)))
        elif path.is_file() and path.suffix != ".pt":
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="loger-kitti01-gslwc")
    parser.add_argument("--entity", default="edward20121127")
    parser.add_argument("--name", required=True)
    parser.add_argument("--job_type", default="hmc_v2_upload")
    parser.add_argument("--result_dir", action="append", default=[], help="HMC result dir with kitti_benchmark.log.")
    parser.add_argument("--phase_b_dir", action="append", default=[], help="Phase B dashboard dir.")
    parser.add_argument("--config_json", default=None, help="Optional extra config JSON file.")
    args = parser.parse_args()

    result_dirs = [Path(p) for p in args.result_dir]
    phase_b_dirs = [Path(p) for p in args.phase_b_dir]
    metrics: Dict[str, Any] = {}
    config: Dict[str, Any] = {
        "result_dirs": [str(p) for p in result_dirs],
        "phase_b_dirs": [str(p) for p in phase_b_dirs],
    }
    if args.config_json:
        config.update(json.loads(Path(args.config_json).read_text()))

    for directory in result_dirs:
        prefix = directory.name
        for key, value in parse_kitti_benchmark(directory / "kitti_benchmark.log").items():
            metrics[f"{prefix}/{key}"] = value
        for key, value in summarize_hmc_state(directory / "hmc_state_hash.jsonl").items():
            metrics[f"{prefix}/{key}"] = value
    for directory in phase_b_dirs:
        metrics.update(summarize_phase_b_dir(directory))

    run = wandb.init(project=args.project, entity=args.entity, name=args.name, job_type=args.job_type, config=config)
    if metrics:
        run.log(metrics)
    add_artifact(run, f"{args.name}_artifacts", "hmc-v2-result", [*result_dirs, *phase_b_dirs])
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
