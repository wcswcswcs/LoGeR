#!/usr/bin/env python3
"""Run and summarize v80 TTT selected-write downstream probes.

This diagnostic runner executes a two-chunk window for each requested case:
baseline TTT write-only, selected low-support token filter, and same-mass
control token filter.  It records exact commands and summarizes whether the
chunk-N write state creates a measurable chunk-(N+1) trajectory response.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final"
)
DEFAULT_ROWS = REPORT_ROOT / "phase10_selected_write_extra_insights_20260623_0041" / "selected_write_extra_insight_rows.csv"
DEFAULT_OUT = REPORT_ROOT / f"phase10_seq02_downstream_probe_batch_{datetime.now().strftime('%Y%m%d_%H%M')}"
DEFAULT_TARGETS = (
    "bad_candidate:02:62",
    "bad_candidate:02:68",
    "bad_candidate:02:70",
    "good_counterexample:02:26",
    "good_counterexample:02:44",
)


@dataclass(frozen=True)
class CaseSpec:
    group: str
    seq: str
    chunk: int
    start_frame: int
    next_start_frame: int
    end_frame_exclusive: int
    support_dir: Path
    insight_row: dict[str, str]

    @property
    def case_id(self) -> str:
        return f"{self.group}_seq{self.seq}_chunk{self.chunk:03d}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--input-root", type=Path, default=Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt"))
    parser.add_argument("--config", type=Path, default=Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml"))
    parser.add_argument("--conda-python", type=Path, default=Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda"))
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--stage-c-root", type=Path, default=Path("results/kitti_preprocess"))
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--support-threshold", type=float, default=0.50)
    parser.add_argument("--dry-run", type=int, default=0)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_clean(payload), ensure_ascii=False, sort_keys=True) + "\n")


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_clean(value), ensure_ascii=False, sort_keys=True)
    return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _parse_target(raw: str) -> tuple[str, str, int]:
    parts = str(raw).split(":")
    if len(parts) != 3:
        raise ValueError(f"target must be group:seq:chunk, got {raw!r}")
    return parts[0], parts[1].zfill(2), int(parts[2])


def _chunk_bounds(stage_c_dir: Path, chunk: int) -> tuple[int, int]:
    pattern = re.compile(rf"^chunk_{chunk:03d}_(\d+)_(\d+)$")
    for item in sorted(stage_c_dir.iterdir()):
        if not item.is_dir():
            continue
        match = pattern.match(item.name)
        if match:
            return int(match.group(1)), int(match.group(2))
    raise FileNotFoundError(f"missing stage-C chunk dir for chunk {chunk:03d} under {stage_c_dir}")


def _case_specs(args: argparse.Namespace) -> list[CaseSpec]:
    rows = _read_csv(args.rows)
    row_by_key = {(row["group"], str(row["seq"]).zfill(2), int(row["chunk"])): row for row in rows}
    specs: list[CaseSpec] = []
    for raw in args.targets:
        group, seq, chunk = _parse_target(raw)
        row = row_by_key.get((group, seq, chunk))
        if row is None:
            raise KeyError(f"target {raw!r} not found in {args.rows}")
        selected_summary = Path(row["selected_summary"])
        selected_payload = json.loads(selected_summary.read_text(encoding="utf-8"))
        support_path = Path(selected_payload["support_path"])
        if not support_path.is_file():
            raise FileNotFoundError(f"support map missing: {support_path}")
        stage_c_dir = args.stage_c_root / seq / "stage_c_cache_semantic_chunks"
        start_frame, _ = _chunk_bounds(stage_c_dir, chunk)
        next_start, end_frame = _chunk_bounds(stage_c_dir, chunk + 1)
        specs.append(
            CaseSpec(
                group=group,
                seq=seq,
                chunk=chunk,
                start_frame=start_frame,
                next_start_frame=next_start,
                end_frame_exclusive=end_frame,
                support_dir=support_path.parent,
                insight_row=row,
            )
        )
    return specs


def _base_pipeline_cmd(args: argparse.Namespace, spec: CaseSpec, output_dir: Path, hmc_jsonl: Path) -> list[str]:
    return [
        str(args.conda_python),
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "run_pipeline_abc_v2.py",
        "--input",
        str(args.input_root / spec.seq / "image_2"),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--output_video",
        "",
        "--output_txt",
        str(output_dir / f"{spec.seq}.txt"),
        "--chunk_size",
        str(args.chunk_size),
        "--chunk_overlap",
        str(args.chunk_overlap),
        "--start_frame",
        str(spec.start_frame),
        "--end_frame",
        str(spec.end_frame_exclusive),
        "--global_chunk_offset",
        str(spec.chunk),
        "--device",
        "cuda",
        "--hybrid_memory_mode",
        "ttt_write_only",
        "--hmc_commit_mode",
        "controlled",
        "--semantic_prior_mode",
        "spg_v2",
        "--stage_c_cache_mode",
        "read",
        "--stage_c_cache_dir",
        str(args.stage_c_root / spec.seq / "stage_c_cache_semantic_chunks"),
        "--stage_c_cache_require_hit",
        "1",
        "--enable_frame_read_control",
        "0",
        "--read_path",
        "none",
        "--read_cue_source",
        "dyn",
        "--semantic_role_policy",
        "fine_ttt_lowstuff_highd_short",
        "--semantic_memory_paths",
        "ttt",
        "--semantic_role_control_mode",
        "none",
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--hybrid_debug_jsonl",
        str(hmc_jsonl),
    ]


def _variant_cmd(args: argparse.Namespace, spec: CaseSpec, variant: str, output_dir: Path, hmc_jsonl: Path) -> list[str]:
    cmd = _base_pipeline_cmd(args, spec, output_dir, hmc_jsonl)
    if variant == "LW1_TTT_SEMANTIC_BASE":
        return cmd
    score_key = "score_overlap" if variant == "LW52_SELECTED" else "control_overlap"
    return cmd + [
        "--semantic_ttt_overlap_support_dir",
        str(spec.support_dir),
        "--semantic_ttt_overlap_support_scope",
        "head_overlap",
        "--semantic_ttt_overlap_support_floor",
        "0.0",
        "--semantic_ttt_overlap_support_score_key",
        score_key,
        "--ttt_write_replay_token_filter_mode",
        "scoped_dynamic_veto",
        "--ttt_write_replay_token_filter_scope",
        "all",
        "--ttt_write_replay_token_filter_threshold",
        f"{args.support_threshold:.2f}",
        "--ttt_write_replay_token_filter_branch_mask",
        "0",
        "--ttt_write_replay_token_filter_blend",
        "1.0",
        "--ttt_write_replay_token_filter_blend_mode",
        "ttl_dynamic",
        "--ttt_write_transient_delta_subtract_scale",
        "1.0",
        "--ttt_write_transient_delta_branch_mask",
        "0",
        "--ttt_write_transient_delta_ttl",
        "1",
    ]


def _build_jobs(args: argparse.Namespace, specs: list[CaseSpec]) -> list[dict[str, Any]]:
    variants = ("LW1_TTT_SEMANTIC_BASE", "LW52_SELECTED", "LW53_CONTROL")
    jobs: list[dict[str, Any]] = []
    for spec in specs:
        case_dir = args.out_dir / spec.case_id
        for variant in variants:
            variant_dir = case_dir / variant
            hmc_jsonl = variant_dir / "hmc_state_hash.jsonl"
            log_path = variant_dir / "run.log"
            variant_dir.mkdir(parents=True, exist_ok=True)
            cmd = _variant_cmd(args, spec, variant, variant_dir, hmc_jsonl)
            jobs.append(
                {
                    "case_id": spec.case_id,
                    "group": spec.group,
                    "seq": spec.seq,
                    "chunk": spec.chunk,
                    "variant": variant,
                    "output_dir": variant_dir,
                    "hmc_jsonl": hmc_jsonl,
                    "log_path": log_path,
                    "command": cmd,
                    "command_string": " ".join(cmd),
                }
            )
    return jobs


def _run_jobs(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gpu_ids = [item.strip() for item in str(args.gpus).split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must not be empty")
    max_parallel = max(1, min(int(args.max_parallel), len(gpu_ids)))
    commands_path = args.out_dir / "commands.jsonl"
    results: list[dict[str, Any]] = []
    running: list[dict[str, Any]] = []
    next_job = 0
    next_gpu = 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if commands_path.exists():
        commands_path.unlink()

    def launch(job: dict[str, Any]) -> None:
        nonlocal next_gpu
        gpu = gpu_ids[next_gpu % len(gpu_ids)]
        next_gpu += 1
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "LOGER_TTT_DISABLE_COMPILE": "1",
            }
        )
        command_record = {
            **{key: job[key] for key in ("case_id", "group", "seq", "chunk", "variant")},
            "gpu": gpu,
            "cwd": str(Path.cwd()),
            "env": {
                "CUDA_VISIBLE_DEVICES": gpu,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "LOGER_TTT_DISABLE_COMPILE": "1",
            },
            "command": job["command"],
            "command_string": job["command_string"],
            "log_path": job["log_path"],
            "output_dir": job["output_dir"],
        }
        _append_jsonl(commands_path, command_record)
        if int(args.dry_run):
            results.append({**command_record, "returncode": None, "duration_sec": 0.0, "status": "dry_run"})
            return
        log_handle = Path(job["log_path"]).open("w", encoding="utf-8")
        started = time.time()
        proc = subprocess.Popen(job["command"], stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        running.append({**command_record, "proc": proc, "log_handle": log_handle, "started": started})
        print(f"launched {job['case_id']} {job['variant']} gpu={gpu} pid={proc.pid}", flush=True)

    while next_job < len(jobs) or running:
        while next_job < len(jobs) and len(running) < max_parallel:
            launch(jobs[next_job])
            next_job += 1
        if not running:
            continue
        time.sleep(5)
        still_running: list[dict[str, Any]] = []
        for item in running:
            proc = item["proc"]
            returncode = proc.poll()
            if returncode is None:
                still_running.append(item)
                continue
            item["log_handle"].close()
            duration = time.time() - float(item["started"])
            result = {
                **{key: item[key] for key in ("case_id", "group", "seq", "chunk", "variant", "gpu", "cwd", "env", "command", "command_string", "log_path", "output_dir")},
                "returncode": int(returncode),
                "duration_sec": duration,
                "status": "ok" if returncode == 0 else "failed",
            }
            results.append(result)
            print(
                f"finished {item['case_id']} {item['variant']} rc={returncode} duration_sec={duration:.1f}",
                flush=True,
            )
        running = still_running
    _write_csv(args.out_dir / "run_status.csv", results)
    return results


def _read_hmc_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_trajectory(path: Path) -> list[tuple[float, list[float]]]:
    rows: list[tuple[float, list[float]]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(part) for part in line.split()]
            if len(parts) >= 8:
                rows.append((parts[0], parts[1:8]))
    return rows


def _diff_region(
    base: list[tuple[float, list[float]]],
    other: list[tuple[float, list[float]]],
    lo: float,
    hi: float,
) -> dict[str, Any]:
    base_by_t = {timestamp: values for timestamp, values in base if lo <= timestamp < hi}
    diffs: list[float] = []
    for timestamp, values in other:
        if timestamp < lo or timestamp >= hi or timestamp not in base_by_t:
            continue
        diffs.extend(abs(a - b) for a, b in zip(values, base_by_t[timestamp]))
    if not diffs:
        return {"n_pose_values": 0, "max_abs_pose_value_diff_vs_LW1": None, "mean_abs_pose_value_diff_vs_LW1": None}
    return {
        "n_pose_values": len(diffs),
        "max_abs_pose_value_diff_vs_LW1": max(diffs),
        "mean_abs_pose_value_diff_vs_LW1": sum(diffs) / len(diffs),
    }


def _summarize(args: argparse.Namespace, specs: list[CaseSpec], run_results: list[dict[str, Any]]) -> dict[str, Any]:
    run_by_key = {(row["case_id"], row["variant"]): row for row in run_results}
    state_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    variants = ("LW1_TTT_SEMANTIC_BASE", "LW52_SELECTED", "LW53_CONTROL")
    for spec in specs:
        case_id = spec.case_id
        trajectories = {}
        for variant in variants:
            output_dir = Path(run_by_key[(case_id, variant)]["output_dir"])
            hmc_rows = _read_hmc_rows(output_dir / "hmc_state_hash.jsonl")
            for row in hmc_rows:
                chunk_idx = int(row.get("chunk_idx", 0))
                state_rows.append(
                    {
                        "case_id": case_id,
                        "group": spec.group,
                        "seq": spec.seq,
                        "chunk": spec.chunk,
                        "variant": variant,
                        "chunk_idx": chunk_idx,
                        "global_chunk": spec.chunk + chunk_idx,
                        "start_frame_local": row.get("start_frame"),
                        "end_frame_local": row.get("end_frame"),
                        "support_applied": row.get("prior_semantic_ttt_overlap_support_applied"),
                        "support_reason": row.get("prior_semantic_ttt_overlap_support_reason"),
                        "support_score_key": row.get("prior_semantic_ttt_overlap_support_score_key"),
                        "support_active_patch_frac": row.get("prior_semantic_ttt_overlap_support_active_patch_frac"),
                        "support_prior_mean_before": row.get("prior_semantic_ttt_overlap_support_prior_mean_before"),
                        "support_prior_mean_after": row.get("prior_semantic_ttt_overlap_support_prior_mean_after"),
                        "token_filter_applied": row.get("ttt_replay_token_filter_applied"),
                        "tokens_before": row.get("ttt_replay_token_filter_tokens_before"),
                        "tokens_after": row.get("ttt_replay_token_filter_tokens_after"),
                        "keep_mass": row.get("ttt_replay_token_filter_keep_mass"),
                        "pass1_pass2_pose_t_mean": row.get("pass1_pass2_pose_t_mean"),
                        "pass1_pass2_pose_matrix_abs_max": row.get("pass1_pass2_pose_matrix_abs_max"),
                        "memory_ttt_mean_rel_diff": row.get("memory_ttt_mean_rel_diff"),
                        "hash_H_next": row.get("hash_H_next"),
                        "controlled_input_state_hash": row.get("controlled_input_state_hash"),
                        "controlled_output_state_hash": row.get("controlled_output_state_hash"),
                    }
                )
            trajectories[variant] = _read_trajectory(output_dir / f"{spec.seq}.txt")

        base = trajectories["LW1_TTT_SEMANTIC_BASE"]
        for variant in ("LW52_SELECTED", "LW53_CONTROL"):
            for region, lo, hi in (
                (f"chunk{spec.chunk:03d}", spec.start_frame, spec.next_start_frame),
                (f"chunk{spec.chunk + 1:03d}", spec.next_start_frame, spec.end_frame_exclusive),
                ("all", spec.start_frame, spec.end_frame_exclusive),
            ):
                diff = _diff_region(base, trajectories[variant], lo, hi)
                trajectory_rows.append(
                    {
                        "case_id": case_id,
                        "group": spec.group,
                        "seq": spec.seq,
                        "chunk": spec.chunk,
                        "variant": variant,
                        "region": region,
                        "lo_frame": lo,
                        "hi_frame": hi,
                        **diff,
                    }
                )

        def _downstream_max(variant: str) -> float:
            for row in trajectory_rows:
                if row["case_id"] == case_id and row["variant"] == variant and row["region"] == f"chunk{spec.chunk + 1:03d}":
                    return _float(row.get("max_abs_pose_value_diff_vs_LW1"))
            return 0.0

        selected_downstream = _downstream_max("LW52_SELECTED")
        control_downstream = _downstream_max("LW53_CONTROL")
        case_rows.append(
            {
                "case_id": case_id,
                "group": spec.group,
                "seq": spec.seq,
                "chunk": spec.chunk,
                "next_chunk": spec.chunk + 1,
                "start_frame": spec.start_frame,
                "next_start_frame": spec.next_start_frame,
                "end_frame_exclusive": spec.end_frame_exclusive,
                "case_types_phase2": spec.insight_row.get("case_types_phase2"),
                "baseline_abs_error_mean_m_phase2": _float(spec.insight_row.get("baseline_abs_error_mean_m_phase2")),
                "selected_low_support_given_selected_runtime": _float(
                    spec.insight_row.get("selected_low_support_given_selected_runtime")
                ),
                "selected_low_support_mass": int(_float(spec.insight_row.get("selected_low_support_mass"))),
                "selected_runtime_mass": int(_float(spec.insight_row.get("selected_runtime_mass"))),
                "runtime_low_support_mass": int(_float(spec.insight_row.get("runtime_low_support_mass"))),
                "selected_downstream_max_abs_pose_value_diff_vs_LW1": selected_downstream,
                "control_downstream_max_abs_pose_value_diff_vs_LW1": control_downstream,
                "selected_minus_control_downstream_max": selected_downstream - control_downstream,
                "selected_support_dir": spec.support_dir,
            }
        )

    _write_csv(args.out_dir / "downstream_probe_state_rows.csv", state_rows)
    _write_csv(args.out_dir / "downstream_probe_trajectory_diff_rows.csv", trajectory_rows)
    _write_csv(args.out_dir / "downstream_probe_case_rows.csv", case_rows)

    bad_rows = [row for row in case_rows if row["group"] == "bad_candidate"]
    good_rows = [row for row in case_rows if row["group"] == "good_counterexample"]
    summary = {
        "schema": "acl2_v80_downstream_probe_batch_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "out_dir": args.out_dir,
        "rows_path": args.rows,
        "target_count": len(specs),
        "run_count": len(run_results),
        "failed_runs": [row for row in run_results if row.get("returncode") not in (0, None)],
        "case_rows": case_rows,
        "aggregates": {
            "bad_selected_downstream_max_mean": _mean(
                [row["selected_downstream_max_abs_pose_value_diff_vs_LW1"] for row in bad_rows]
            ),
            "bad_control_downstream_max_mean": _mean(
                [row["control_downstream_max_abs_pose_value_diff_vs_LW1"] for row in bad_rows]
            ),
            "good_selected_downstream_max_mean": _mean(
                [row["selected_downstream_max_abs_pose_value_diff_vs_LW1"] for row in good_rows]
            ),
            "good_control_downstream_max_mean": _mean(
                [row["control_downstream_max_abs_pose_value_diff_vs_LW1"] for row in good_rows]
            ),
        },
        "outputs": {
            "commands_jsonl": args.out_dir / "commands.jsonl",
            "run_status_csv": args.out_dir / "run_status.csv",
            "state_rows_csv": args.out_dir / "downstream_probe_state_rows.csv",
            "trajectory_diff_rows_csv": args.out_dir / "downstream_probe_trajectory_diff_rows.csv",
            "case_rows_csv": args.out_dir / "downstream_probe_case_rows.csv",
            "summary_json": args.out_dir / "downstream_probe_summary.json",
        },
    }
    _write_json(args.out_dir / "downstream_probe_summary.json", summary)
    return summary


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return sum(clean) / len(clean)


def main() -> None:
    args = _parse_args()
    specs = _case_specs(args)
    jobs = _build_jobs(args, specs)
    run_results = _run_jobs(args, jobs)
    if int(args.dry_run):
        _write_json(args.out_dir / "dry_run_jobs.json", {"jobs": jobs, "run_results": run_results})
        print(json.dumps({"dry_run": True, "job_count": len(jobs), "out_dir": str(args.out_dir)}, indent=2))
        return
    summary = _summarize(args, specs, run_results)
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))
    if summary.get("failed_runs"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
