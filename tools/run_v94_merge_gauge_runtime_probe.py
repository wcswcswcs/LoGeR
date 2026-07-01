#!/usr/bin/env python3
"""Run v94 Phase3R merge/gauge runtime trajectory probes.

This script reuses v93 geometry-only two-window commands, adds merge pose
serialization, and applies fixed diagnostic merge/gauge variants. It does not
promote the result to a method success; the paired summarizer decides what the
measured trajectories support.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
PHASE1_ROWS = ROOT / "phase1_boundary_failure_atlas/boundary_failure_rows.csv"
V93_TRACE_ROOT = Path(
    "results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier/phase3_merge_gauge_trace_smoke"
)
DEFAULT_OUT = ROOT / "phase3r_runtime_merge_gauge_probe"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seq_text(value: Any) -> str:
    try:
        return f"{int(float(value)):02d}"
    except (TypeError, ValueError):
        return str(value).zfill(2)


def set_arg(cmd: list[str], flag: str, value: str) -> list[str]:
    out = list(cmd)
    if flag in out:
        idx = out.index(flag)
        if idx + 1 >= len(out):
            raise ValueError(f"Missing value after {flag}")
        out[idx + 1] = value
        return out
    out.extend([flag, value])
    return out


def remove_arg(cmd: list[str], flag: str) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(cmd):
        if cmd[idx] == flag:
            idx += 2
            continue
        out.append(cmd[idx])
        idx += 1
    return out


SEMANTIC_MERGE_FLAGS = [
    "--semantic_merge_mode",
    "--semantic_merge_strategy",
    "--semantic_merge_use_semantic_confidence",
    "--semantic_merge_semantic_conf_min",
    "--semantic_merge_blend_alpha",
    "--semantic_merge_blend_components",
    "--semantic_merge_max_blend_log_scale_delta",
    "--semantic_merge_max_blend_rotation_delta_deg",
    "--semantic_merge_max_blend_translation_delta",
    "--semantic_merge_qscale_hold_refresh",
    "--semantic_merge_qscale_reference",
    "--semantic_merge_qscale_min_factor",
    "--semantic_merge_qscale_condition_reference",
    "--semantic_merge_qscale_residual_reference",
]


def remove_semantic_merge_args(cmd: list[str]) -> list[str]:
    out = list(cmd)
    for flag in SEMANTIC_MERGE_FLAGS:
        out = remove_arg(out, flag)
    return out


def load_source_command(seq: str, curr_chunk: int) -> list[str]:
    manifest = V93_TRACE_ROOT / f"seq{seq}_geometry_only_alltargets/phaseE_merge_run_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing v93 manifest: {manifest}")
    data = read_json(manifest)
    for job in data.get("jobs", []):
        if int(job.get("chunk")) == int(curr_chunk):
            return list(job["cmd"])
    raise KeyError(f"No v93 geometry_only job for seq={seq} chunk={curr_chunk}")


def select_targets(rows: pd.DataFrame, max_targets: int) -> list[dict[str, Any]]:
    rows = rows.copy()
    rows["seq"] = rows["seq"].map(seq_text)
    rows["boundary_jump_num"] = pd.to_numeric(rows["boundary_jump"], errors="coerce")
    rows["J_handoff_num"] = pd.to_numeric(rows["J_handoff"], errors="coerce")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(frame: pd.DataFrame, tag: str, limit: int) -> None:
        nonlocal selected
        for _, item in frame.iterrows():
            pid = str(item["pair_id"])
            if pid in seen:
                continue
            seen.add(pid)
            row = item.to_dict()
            row["probe_selection_tag"] = tag
            selected.append(row)
            if len([r for r in selected if r["probe_selection_tag"] == tag]) >= limit:
                break

    gauge = rows[rows["failure_type_primary"].astype(str).eq("HANDOFF_GAUGE")].sort_values(
        "boundary_jump_num", ascending=False
    )
    add(gauge, "handoff_gauge_primary", 99)

    expanded = rows[
        rows["local_chunk_good_flag"].astype(bool)
        & ~rows["pair_id"].astype(str).isin(seen)
        & rows["boundary_jump_num"].notna()
    ].sort_values("boundary_jump_num", ascending=False)
    add(expanded, "gauge_candidate_expanded_boundary_top", max(0, 8 - len(selected)))

    good = rows[
        rows["case_label_offline_only"].astype(str).eq("good")
        & ~rows["pair_id"].astype(str).isin(seen)
        & rows["boundary_jump_num"].notna()
    ].sort_values("J_handoff_num", ascending=False)
    add(good, "good_safe_control", 4)

    scale = rows[
        rows["failure_type_primary"].astype(str).eq("HANDOFF_SCALE")
        & ~rows["pair_id"].astype(str).isin(seen)
    ].sort_values("J_handoff_num", ascending=False)
    add(scale, "handoff_scale_context", 4)

    return selected[:max_targets]


def select_manual_targets(rows: pd.DataFrame, pair_ids: list[str]) -> list[dict[str, Any]]:
    rows = rows.copy()
    rows["seq"] = rows["seq"].map(seq_text)
    by_pair = {str(row["pair_id"]): row.to_dict() for _, row in rows.iterrows()}
    selected: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        if pair_id not in by_pair:
            raise KeyError(f"Missing requested pair_id in phase1 rows: {pair_id}")
        row = dict(by_pair[pair_id])
        row["probe_selection_tag"] = "manual_focus_bad_good_control"
        selected.append(row)
    return selected


def log_has_success(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "returncode=0" in text


def variant_command(cmd: list[str], variant: str, out_dir: Path, seq: str) -> list[str]:
    out = list(cmd)
    out = set_arg(out, "--output_txt", str(out_dir / f"{seq}.txt"))
    out = set_arg(out, "--hybrid_debug_jsonl", str(out_dir / "hmc_state_hash.jsonl"))
    out = set_arg(out, "--merge_state_trace_jsonl", str(out_dir / "merge_state_trace.jsonl"))
    out = set_arg(out, "--save_merge_states", str(out_dir / "merge_states"))
    out = set_arg(out, "--save_premerge_local_output", str(out_dir / "premerge_local_pose.jsonl"))
    out = set_arg(out, "--save_postmerge_global_output", str(out_dir / "postmerge_global_pose.jsonl"))
    if "--swa_overlap_feature_dump_dir" in out:
        out = set_arg(out, "--swa_overlap_feature_dump_dir", str(out_dir / "swa_overlap_feature_maps"))

    if variant == "native_actual":
        return out
    if variant == "merge_boundary_hold":
        return set_arg(out, "--semantic_merge_blend_alpha", "0.0")
    if variant == "merge_no_refresh":
        return set_arg(out, "--semantic_merge_qscale_hold_refresh", "0")
    if variant == "merge_robust_native_only":
        out = set_arg(out, "--semantic_merge_use_semantic_confidence", "0")
        return out
    if variant == "merge_mode_off_keep_args":
        return set_arg(out, "--semantic_merge_mode", "none")
    alpha_delta_match = re.fullmatch(r"merge_alpha_([0-9]+)p([0-9]+)_delta_([0-9]+)p([0-9]+)", variant)
    if alpha_delta_match:
        alpha = f"{alpha_delta_match.group(1)}.{alpha_delta_match.group(2)}"
        delta = f"{alpha_delta_match.group(3)}.{alpha_delta_match.group(4)}"
        out = set_arg(out, "--semantic_merge_blend_alpha", alpha)
        return set_arg(out, "--semantic_merge_max_blend_log_scale_delta", delta)
    alpha_match = re.fullmatch(r"merge_alpha_([0-9]+)p([0-9]+)", variant)
    if alpha_match:
        alpha = f"{alpha_match.group(1)}.{alpha_match.group(2)}"
        return set_arg(out, "--semantic_merge_blend_alpha", alpha)
    maxpts_match = re.fullmatch(r"merge_maxpts_([0-9]+)", variant)
    if maxpts_match:
        return set_arg(out, "--semantic_merge_max_points", maxpts_match.group(1))
    if variant == "merge_low_delta_0p03":
        return set_arg(out, "--semantic_merge_max_blend_log_scale_delta", "0.03")
    if variant == "merge_low_delta_0p02":
        return set_arg(out, "--semantic_merge_max_blend_log_scale_delta", "0.02")
    if variant == "merge_low_delta_0p01":
        return set_arg(out, "--semantic_merge_max_blend_log_scale_delta", "0.01")
    if variant == "native_no_semantic_merge":
        out = remove_semantic_merge_args(out)
        return out
    source_replace_match = re.fullmatch(r"swa_replace_stable_v(_random)?_alpha0p35", variant)
    if source_replace_match:
        out = remove_semantic_merge_args(out)
        mode = "stable_agreement_topq80_random_same_mass" if source_replace_match.group(1) else "stable_agreement_topq80"
        out = set_arg(out, "--enable_swa_overlap_source_gate", "0")
        out = set_arg(out, "--enable_swa_overlap_source_replace", "1")
        out = set_arg(out, "--swa_overlap_source_replace_alpha", "0.35")
        out = set_arg(out, "--swa_overlap_source_replace_mode", mode)
        out = set_arg(out, "--swa_overlap_source_replace_target", "v")
        out = set_arg(out, "--swa_overlap_source_replace_layer_mode", "last")
        out = set_arg(out, "--swa_overlap_source_replace_single_layer", "-1")
        out = set_arg(out, "--swa_overlap_feature_dump_dir", str(out_dir / "swa_overlap_feature_maps"))
        return out
    source_gate_match = re.fullmatch(r"swa_gate_stable_v(_random)?_rho0p35", variant)
    if source_gate_match:
        out = remove_semantic_merge_args(out)
        mode = "stable_agreement_topq80_random_same_mass" if source_gate_match.group(1) else "stable_agreement_topq80"
        out = set_arg(out, "--enable_swa_overlap_source_replace", "0")
        out = set_arg(out, "--enable_swa_overlap_source_gate", "1")
        out = set_arg(out, "--swa_overlap_source_gate_rho", "0.35")
        out = set_arg(out, "--swa_overlap_source_gate_min", "0.85")
        out = set_arg(out, "--swa_overlap_source_gate_mode", mode)
        out = set_arg(out, "--swa_overlap_source_gate_target", "v")
        out = set_arg(out, "--swa_overlap_source_gate_layer_mode", "last")
        out = set_arg(out, "--swa_overlap_source_gate_single_layer", "-1")
        out = set_arg(out, "--swa_overlap_feature_dump_dir", str(out_dir / "swa_overlap_feature_maps"))
        return out
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-rows", type=Path, default=PHASE1_ROWS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--variants", default="native_actual,merge_boundary_hold,merge_no_refresh,merge_robust_native_only")
    parser.add_argument("--max-targets", type=int, default=12)
    parser.add_argument("--target-pairs", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.phase1_rows)
    manual_pair_ids = [item.strip() for item in args.target_pairs.split(",") if item.strip()]
    targets = select_manual_targets(rows, manual_pair_ids) if manual_pair_ids else select_targets(rows, args.max_targets)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]

    jobs: list[dict[str, Any]] = []
    for target in targets:
        seq = seq_text(target["seq"])
        curr = int(target["curr_chunk"])
        base_cmd = load_source_command(seq, curr)
        for variant in variants:
            out_dir = args.out_root / f"{target['pair_id']}" / variant
            cmd = variant_command(base_cmd, variant, out_dir, seq)
            jobs.append(
                {
                    "pair_id": target["pair_id"],
                    "seq": seq,
                    "prev_chunk": int(target["prev_chunk"]),
                    "curr_chunk": curr,
                    "case_label_offline_only": target.get("case_label_offline_only", ""),
                    "failure_type_primary": target.get("failure_type_primary", ""),
                    "failure_type_secondary": target.get("failure_type_secondary", ""),
                    "probe_selection_tag": target["probe_selection_tag"],
                    "variant": variant,
                    "out_dir": str(out_dir),
                    "cmd": cmd,
                }
            )

    if args.dry_run:
        write_json(
            args.out_root / "runtime_probe_manifest.json",
            {
                "phase": "Phase3R_runtime_merge_gauge_probe",
                "dry_run": True,
                "target_count": len(targets),
                "job_count": len(jobs),
                "variants": variants,
                "targets": targets,
                "jobs": jobs,
            },
        )
        print(f"dry_run=True target_count={len(targets)} job_count={len(jobs)}")
        return

    launched: list[dict[str, Any]] = []
    pending = list(jobs)
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    def launch(job: dict[str, Any], gpu: str) -> dict[str, Any]:
        out_dir = Path(job["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "runtime_probe.log"
        if args.skip_existing and (out_dir / "postmerge_global_pose.jsonl").exists() and log_has_success(log_path):
            done = dict(job)
            done.update({"gpu": gpu, "returncode": 0, "duration_sec": 0.0, "log_path": str(log_path), "skipped_existing": True})
            completed.append(done)
            return {}
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        handle = log_path.open("w", encoding="utf-8")
        handle.write(f"$ {' '.join(job['cmd'])}\n")
        handle.write(f"CUDA_VISIBLE_DEVICES={gpu}\n")
        handle.flush()
        proc = subprocess.Popen(job["cmd"], cwd=Path.cwd(), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        active = dict(job)
        active.update({"gpu": gpu, "proc": proc, "handle": handle, "start": time.time(), "log_path": str(log_path)})
        return active

    gpu_cursor = 0
    while pending or launched:
        while pending and len(launched) < max(1, len(gpus)):
            gpu = gpus[gpu_cursor % len(gpus)] if gpus else ""
            gpu_cursor += 1
            active = launch(pending.pop(0), gpu)
            if active:
                launched.append(active)
        still: list[dict[str, Any]] = []
        for active in launched:
            ret = active["proc"].poll()
            if ret is None:
                still.append(active)
                continue
            duration = time.time() - active["start"]
            active["handle"].write(f"\nreturncode={ret}\nduration_sec={duration:.3f}\n")
            active["handle"].close()
            row = {k: v for k, v in active.items() if k not in {"proc", "handle", "start"}}
            row.update({"returncode": int(ret), "duration_sec": duration, "skipped_existing": False})
            completed.append(row)
            if ret != 0:
                failed.append(row)
        launched = still
        if launched:
            time.sleep(5)

    summary = {
        "phase": "Phase3R_runtime_merge_gauge_probe",
        "dry_run": False,
        "out_root": str(args.out_root),
        "target_count": len(targets),
        "job_count": len(jobs),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "all_completed": len(failed) == 0 and len(completed) == len(jobs),
        "variants": variants,
        "targets": targets,
        "jobs": completed,
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_root / "runtime_probe_manifest.json", summary)
    print(f"target_count={summary['target_count']}")
    print(f"job_count={summary['job_count']}")
    print(f"completed_count={summary['completed_count']}")
    print(f"failed_count={summary['failed_count']}")
    print(f"all_completed={summary['all_completed']}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
