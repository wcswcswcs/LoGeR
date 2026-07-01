#!/usr/bin/env python3
"""Run v96 trace-only raw-QK smoke jobs.

The smoke is intentionally diagnostic.  For each selected case it runs:

1. a beta=0 frame-read baseline that still builds READ cue patch dumps;
2. a beta=0 + trace-only source run that asks the attention hook to dump raw-QK
   source-attention maps while returning to the original no-mask attention path.
3. optionally, a trace-only counterfactual early K-side action probe with a
   nonzero source bias.  It records before/after attention mass but still
   returns to the original no-mask attention path, so it is not a promoted
   runtime action.

The script compares pose TXT outputs and records whether this is a valid
no-action parity smoke.  It does not promote any READ/SWA/TTT action.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any


ROOT = Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control")
OUT_ROOT = ROOT / "trackJ_raw_qk_trace_smoke"
DUMP_HEAD_MARGINAL = False
DUMP_FULL_QUERY_MARGINAL = True
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DATA_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")

CASES = [
    {
        "case_id": "01_008_009",
        "seq": "01",
        "chunk": 9,
        "start_frame": 261,
        "end_frame": 293,
        "label": "READ_LOCAL_BAD",
    },
    {
        "case_id": "02_007_008",
        "seq": "02",
        "chunk": 7,
        "start_frame": 203,
        "end_frame": 235,
        "label": "GOOD_CONTROL",
    },
]


def load_cases_from_atlas(case_ids: list[str]) -> list[dict[str, Any]]:
    atlas_path = ROOT / "trackA_case_response_atlas" / "rows.csv"
    rows: dict[str, dict[str, str]] = {}
    if atlas_path.is_file():
        with atlas_path.open("r", encoding="utf-8", newline="") as handle:
            rows = {row.get("case_id", ""): row for row in csv.DictReader(handle)}
    cases: list[dict[str, Any]] = []
    for case_id in case_ids:
        row = rows.get(case_id)
        if not row:
            raise ValueError(f"case_id not found in {atlas_path}: {case_id}")
        seq = str(row.get("seq", ""))
        chunk = int(row.get("curr_chunk", "0"))
        labels = str(row.get("action_response_labels", ""))
        label = "GOOD_CONTROL" if "GOOD_PROTECTION" in labels else "READ_LOCAL_BAD" if "READ_LOCAL_BAD" in labels else "SUPPORT"
        start_frame = chunk * 29
        cases.append(
            {
                "case_id": case_id,
                "seq": seq,
                "chunk": chunk,
                "start_frame": start_frame,
                "end_frame": start_frame + 32,
                "label": label,
            }
        )
    return cases


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
            writer.writerow(row)


def sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cmd(case: dict[str, Any], variant: str, out_dir: Path) -> list[str]:
    seq = str(case["seq"])
    chunk = int(case["chunk"])
    cmd = [
        str(CONDA),
        "run",
        "--no-capture-output",
        "-n",
        "loger",
        "python",
        "run_pipeline_abc_v2.py",
        "--input",
        str(DATA_ROOT / seq / "image_2"),
        "--output_video",
        "",
        "--output_txt",
        str(out_dir / f"{seq}.txt"),
        "--checkpoint",
        str(CHECKPOINT),
        "--config",
        str(CONFIG),
        "--chunk_size",
        "32",
        "--chunk_overlap",
        "3",
        "--start_frame",
        str(int(case["start_frame"])),
        "--end_frame",
        str(int(case["end_frame"])),
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
        f"results/kitti_preprocess/{seq}/stage_c_cache_semantic_chunks",
        "--stage_c_cache_require_hit",
        "1",
        "--read_path",
        "frame",
        "--beta_frame",
        "0.0",
        "--frame_bias_mode",
        "key",
        "--read_cue_source",
        "v78.l07_l13.l07_action_only",
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
        "--read_cue_patch_dump_dir",
        str(out_dir / "read_cue_patch_dumps"),
        "--read_cue_patch_dump_dtype",
        "float16",
    ]
    if variant in {"trace_noop", "action_trace_probe"}:
        rho = "0.0" if variant == "trace_noop" else "0.1"
        min_keep = "1.0" if variant == "trace_noop" else "0.9"
        cmd += [
            "--enable_context_source_skip",
            "1",
            "--context_source_skip_impl",
            "trace_only",
            "--context_source_skip_scope",
            "frame",
            "--context_source_skip_mode",
            "soft",
            "--context_source_skip_mask",
            "dg_q90",
            "--context_source_skip_soft_rho",
            rho,
            "--context_source_skip_soft_min_keep",
            min_keep,
            "--context_source_skip_layer_mode",
            "early",
            "--context_source_skip_record_attention_mass",
            "1",
            "--context_source_skip_attention_mass_max_queries",
            "128",
            "--context_source_skip_attention_map_dump_dir",
            str(out_dir / "raw_qk_attention_dumps"),
            "--context_source_skip_attention_map_dump_max_queries",
            "128",
            "--context_source_skip_attention_map_dump_dtype",
            "float16",
            "--context_source_skip_attention_map_dump_full_query_marginal",
            "1" if DUMP_FULL_QUERY_MARGINAL else "0",
            "--context_source_skip_attention_map_dump_query_block",
            "32",
        ]
        if DUMP_HEAD_MARGINAL:
            cmd += [
                "--context_source_skip_attention_map_dump_head_marginal",
                "1",
            ]
    return cmd


def run_job(case: dict[str, Any], variant: str, gpu: int) -> dict[str, Any]:
    out_dir = OUT_ROOT / case["case_id"] / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(case, variant, out_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    started = time.time()
    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=Path.cwd(), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    duration = time.time() - started
    payload = {
        "case_id": case["case_id"],
        "seq": case["seq"],
        "chunk": case["chunk"],
        "label": case["label"],
        "variant": variant,
        "gpu": gpu,
        "returncode": proc.returncode,
        "duration_sec": duration,
        "cmd": cmd,
        "cmd_shell": " ".join(cmd),
        "out_dir": str(out_dir),
        "run_log": str(log_path),
        "output_txt": str(out_dir / f"{case['seq']}.txt"),
        "output_txt_sha256": sha256(out_dir / f"{case['seq']}.txt"),
        "read_cue_dump_count": len(list((out_dir / "read_cue_patch_dumps").glob("*.pt"))),
        "raw_qk_dump_count": len(list((out_dir / "raw_qk_attention_dumps").glob("*.pt"))),
    }
    write_json(out_dir / "job_summary.json", payload)
    return payload


def read_pose_txt(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(x) for x in line.split()])
    return rows


def compare_case(case: dict[str, Any]) -> dict[str, Any]:
    seq = str(case["seq"])
    base = OUT_ROOT / case["case_id"] / "baseline_noop" / f"{seq}.txt"
    trace = OUT_ROOT / case["case_id"] / "trace_noop" / f"{seq}.txt"
    a = read_pose_txt(base)
    b = read_pose_txt(trace)
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "seq": seq,
        "chunk": case["chunk"],
        "label": case["label"],
        "baseline_txt": str(base),
        "trace_txt": str(trace),
        "baseline_sha256": sha256(base),
        "trace_sha256": sha256(trace),
        "baseline_rows": len(a),
        "trace_rows": len(b),
        "shape_match": len(a) == len(b) and all(len(x) == len(y) for x, y in zip(a, b)),
        "max_abs_pose_diff": None,
        "strict_sha_parity": False,
        "numeric_parity_le_1e_6": False,
    }
    if row["shape_match"] and a:
        max_abs = 0.0
        for ra, rb in zip(a, b):
            for va, vb in zip(ra, rb):
                max_abs = max(max_abs, abs(va - vb))
        row["max_abs_pose_diff"] = max_abs
        row["numeric_parity_le_1e_6"] = max_abs <= 1e-6
    row["strict_sha_parity"] = bool(row["baseline_sha256"] and row["baseline_sha256"] == row["trace_sha256"])
    trace_dir = OUT_ROOT / case["case_id"] / "trace_noop"
    raw_dumps = sorted((trace_dir / "raw_qk_attention_dumps").glob("*.pt"))
    read_dumps = sorted((trace_dir / "read_cue_patch_dumps").glob("*.pt"))
    row["raw_qk_dump_count"] = len(raw_dumps)
    row["read_cue_dump_count"] = len(read_dumps)
    row["trace_smoke_pass"] = bool(
        row["numeric_parity_le_1e_6"]
        and row["raw_qk_dump_count"] > 0
        and row["read_cue_dump_count"] > 0
    )
    return row


def semantic_alignment_case(case: dict[str, Any]) -> dict[str, Any]:
    import torch

    seq = str(case["seq"])
    chunk = int(case["chunk"])
    trace_dir = OUT_ROOT / case["case_id"] / "trace_noop"
    raw_dumps = sorted((trace_dir / "raw_qk_attention_dumps").glob("*.pt"))
    read_dump = trace_dir / "read_cue_patch_dumps" / f"chunk_{chunk:03d}_read_cue_patch.pt"
    hmc_path = trace_dir / "hmc_state_hash.jsonl"
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "seq": seq,
        "chunk": chunk,
        "label": case["label"],
        "raw_dump_count": len(raw_dumps),
        "read_dump": str(read_dump),
        "hmc_jsonl": str(hmc_path),
        "semantic_alignment_smoke_pass": False,
    }
    if not raw_dumps or not read_dump.is_file() or not hmc_path.is_file():
        row["failure_reason"] = "missing_raw_or_read_or_hmc_artifact"
        return row
    try:
        raw = torch.load(raw_dumps[0], map_location="cpu")
        read = torch.load(read_dump, map_location="cpu")
        hmc_records = [json.loads(line) for line in hmc_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        hmc = hmc_records[-1] if hmc_records else {}
        frame_stats = (
            hmc.get("control_trace", {})
            .get("hook_effect_summary", {})
            .get("frame_attention", {})
        )
        patch_grid = [int(v) for v in read.get("patch_grid", [])]
        num_frames = int(read.get("num_frames", 0) or 0)
        patch_tokens = int(patch_grid[0] * patch_grid[1]) if len(patch_grid) == 2 else 0
        q_shape = raw.get("q_shape", [])
        k_shape = raw.get("k_shape", [])
        source_token_count = int(raw.get("source_token_count", 0) or 0)
        if source_token_count <= 0 and isinstance(k_shape, list) and len(k_shape) >= 3:
            source_token_count = int(k_shape[2])
        special_tokens_per_frame = source_token_count - patch_tokens
        affected = raw.get("affected_mask")
        read_patch = read.get("tensors", {}).get("read_patch_final") if isinstance(read, dict) else None
        affected_true_count = int(affected.sum().item()) if torch.is_tensor(affected) else -1
        hmc_control_tokens = int(frame_stats.get("max_context_source_control_tokens", -2) or 0)
        q_shape_ok = len(q_shape) >= 3 and int(q_shape[0]) == num_frames and int(q_shape[2]) == source_token_count
        k_shape_ok = len(k_shape) >= 3 and int(k_shape[0]) == num_frames and int(k_shape[2]) == source_token_count
        read_patch_shape_ok = torch.is_tensor(read_patch) and tuple(int(v) for v in read_patch.shape) == (
            num_frames,
            patch_grid[0],
            patch_grid[1],
        )
        affected_shape_ok = torch.is_tensor(affected) and tuple(int(v) for v in affected.shape) == (
            num_frames,
            source_token_count,
        )
        semantic_prior_present = bool(hmc.get("prior_semantic_prior_present"))
        semantic_prior_consumed = bool(hmc.get("prior_semantic_action_prior_consumed_for_action"))
        dense_projection_nonempty = bool(hmc.get("prior_dense_semantic_token_projection_nonempty"))
        row.update(
            {
                "raw_schema": raw.get("schema", ""),
                "read_schema": read.get("schema", ""),
                "raw_q_shape": "x".join(str(v) for v in q_shape),
                "raw_k_shape": "x".join(str(v) for v in k_shape),
                "raw_source_token_count": source_token_count,
                "read_num_frames": num_frames,
                "read_patch_grid": "x".join(str(v) for v in patch_grid),
                "read_patch_tokens_per_frame": patch_tokens,
                "special_tokens_per_frame": special_tokens_per_frame,
                "affected_mask_shape": "x".join(str(v) for v in affected.shape) if torch.is_tensor(affected) else "",
                "affected_true_count": affected_true_count,
                "hmc_context_source_control_tokens": hmc_control_tokens,
                "semantic_prior_present": semantic_prior_present,
                "semantic_prior_consumed_for_action": semantic_prior_consumed,
                "dense_semantic_token_projection_nonempty": dense_projection_nonempty,
                "dense_semantic_patch_nonvoid_ratio": hmc.get("prior_dense_semantic_patch_nonvoid_ratio"),
                "dense_semantic_patch_purity": hmc.get("prior_dense_semantic_patch_purity"),
                "q_shape_ok": q_shape_ok,
                "k_shape_ok": k_shape_ok,
                "read_patch_shape_ok": read_patch_shape_ok,
                "affected_shape_ok": affected_shape_ok,
                "affected_count_matches_hmc": affected_true_count == hmc_control_tokens,
            }
        )
        row["semantic_alignment_smoke_pass"] = bool(
            q_shape_ok
            and k_shape_ok
            and read_patch_shape_ok
            and affected_shape_ok
            and special_tokens_per_frame >= 0
            and affected_true_count == hmc_control_tokens
            and semantic_prior_present
            and semantic_prior_consumed
            and dense_projection_nonempty
        )
        if not row["semantic_alignment_smoke_pass"]:
            row["failure_reason"] = "one_or_more_alignment_checks_failed"
    except Exception as exc:  # noqa: BLE001
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return row


def action_trace_probe_case(case: dict[str, Any]) -> dict[str, Any]:
    import torch

    trace_dir = OUT_ROOT / case["case_id"] / "action_trace_probe"
    raw_dumps = sorted((trace_dir / "raw_qk_attention_dumps").glob("*.pt"))
    hmc_path = trace_dir / "hmc_state_hash.jsonl"
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "seq": case["seq"],
        "chunk": case["chunk"],
        "label": case["label"],
        "raw_dump_count": len(raw_dumps),
        "hmc_jsonl": str(hmc_path),
        "action_trace_probe_pass": False,
    }
    if not raw_dumps or not hmc_path.is_file():
        row["failure_reason"] = "missing_raw_or_hmc_artifact"
        return row
    try:
        raw = torch.load(raw_dumps[0], map_location="cpu")
        affected = raw.get("affected_mask")

        def affected_mass_mean(attn_key: str) -> float | None:
            value = raw.get(attn_key)
            if not torch.is_tensor(value) or not torch.is_tensor(affected):
                return None
            attn = value.float()
            mask = affected.float()
            if attn.ndim == 4:
                return float((attn * mask[:, None, None, :]).sum(dim=-1).mean().item())
            if attn.ndim == 2:
                return float((attn * mask).sum(dim=-1).mean().item())
            return None

        before_raw = raw.get("source_attention_before_affected_mass_mean")
        after_raw = raw.get("source_attention_after_affected_mass_mean")
        before = float(before_raw) if before_raw is not None else affected_mass_mean("attention_before_control")
        after = float(after_raw) if after_raw is not None else affected_mass_mean("attention_after_bias_control")
        hmc_records = [json.loads(line) for line in hmc_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        hmc = hmc_records[-1] if hmc_records else {}
        frame_stats = (
            hmc.get("control_trace", {})
            .get("hook_effect_summary", {})
            .get("frame_attention", {})
        )
        mean_before = frame_stats.get("mean_attention_mass_removed_before")
        mean_after = frame_stats.get("mean_attention_mass_removed_after")
        source_weight_min = frame_stats.get("mean_source_weight_min")
        row.update(
            {
                "raw_schema": raw.get("schema", ""),
                "sampled_pairwise_attention_matrix_stored": raw.get("sampled_pairwise_attention_matrix_stored"),
                "sampled_query_count": (
                    int(raw.get("query_indices").numel()) if torch.is_tensor(raw.get("query_indices")) else None
                ),
                "raw_before_affected_mass_mean": before,
                "raw_after_affected_mass_mean": after,
                "raw_attention_mass_delta": (after - before) if after is not None and before is not None else None,
                "hmc_mean_attention_mass_removed_before": mean_before,
                "hmc_mean_attention_mass_removed_after": mean_after,
                "hmc_mean_source_weight_min": source_weight_min,
                "hmc_mean_source_weight_mean": frame_stats.get("mean_source_weight_mean"),
                "hmc_context_source_control_tokens": frame_stats.get("max_context_source_control_tokens"),
            }
        )
        row["action_trace_probe_pass"] = bool(
            after is not None
            and before is not None
            and after < before
            and mean_after is not None
            and mean_before is not None
            and float(mean_after) < float(mean_before)
            and source_weight_min is not None
            and float(source_weight_min) < 1.0
        )
    except Exception as exc:  # noqa: BLE001
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--case-ids", default="", help="comma-separated case ids from trackA_case_response_atlas/rows.csv")
    parser.add_argument("--max-workers", type=int, default=0, help="parallel job limit; default is number of GPUs")
    parser.add_argument("--dump-head-marginal", action="store_true", help="also dump per-head full-query source marginals")
    parser.add_argument(
        "--sampled-pairwise-dump",
        action="store_true",
        help="dump sampled source-target attention maps instead of full-query source marginals",
    )
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--include-action-probe", action="store_true")
    parser.add_argument("--action-probe-only", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="reuse existing job_summary.json files and recompute aggregate parity without launching jobs",
    )
    return parser.parse_args()


def load_existing_job_results(jobs: list[tuple[dict[str, Any], str, int]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case, variant, gpu in jobs:
        out_dir = OUT_ROOT / case["case_id"] / variant
        summary_path = out_dir / "job_summary.json"
        if summary_path.is_file():
            row = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            row = {
                "case_id": case["case_id"],
                "seq": case["seq"],
                "chunk": case["chunk"],
                "label": case["label"],
                "variant": variant,
                "gpu": gpu,
                "returncode": None,
                "duration_sec": None,
                "cmd": build_cmd(case, variant, out_dir),
                "cmd_shell": " ".join(build_cmd(case, variant, out_dir)),
                "out_dir": str(out_dir),
                "run_log": str(out_dir / "run.log"),
                "output_txt": str(out_dir / f"{case['seq']}.txt"),
            }
        row["output_txt_sha256"] = sha256(out_dir / f"{case['seq']}.txt")
        row["read_cue_dump_count"] = len(list((out_dir / "read_cue_patch_dumps").glob("*.pt")))
        row["raw_qk_dump_count"] = len(list((out_dir / "raw_qk_attention_dumps").glob("*.pt")))
        results.append(row)
    return sorted(results, key=lambda item: (item["case_id"], item["variant"]))


def main() -> None:
    global OUT_ROOT, DUMP_HEAD_MARGINAL, DUMP_FULL_QUERY_MARGINAL
    args = parse_args()
    OUT_ROOT = args.output_root
    DUMP_HEAD_MARGINAL = bool(args.dump_head_marginal)
    DUMP_FULL_QUERY_MARGINAL = not bool(args.sampled_pairwise_dump)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    gpus = [int(x) for x in str(args.gpus).split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    case_ids = [case_id.strip() for case_id in str(args.case_ids).split(",") if case_id.strip()]
    cases = load_cases_from_atlas(case_ids) if case_ids else CASES
    jobs: list[tuple[dict[str, Any], str, int]] = []
    def next_gpu() -> int:
        return gpus[len(jobs) % len(gpus)]

    for case in cases:
        if not args.action_probe_only:
            jobs.append((case, "baseline_noop", next_gpu()))
            jobs.append((case, "trace_noop", next_gpu()))
        if args.include_action_probe or args.action_probe_only:
            jobs.append((case, "action_trace_probe", next_gpu()))
    manifest_rows = [
        {
            "case_id": case["case_id"],
            "seq": case["seq"],
            "chunk": case["chunk"],
            "label": case["label"],
            "variant": variant,
            "gpu": gpu,
            "out_dir": str(OUT_ROOT / case["case_id"] / variant),
            "cmd_shell": " ".join(build_cmd(case, variant, OUT_ROOT / case["case_id"] / variant)),
        }
        for case, variant, gpu in jobs
    ]
    write_csv(OUT_ROOT / "job_manifest.csv", manifest_rows)
    if args.no_run:
        planned = {"status": "planned_not_run", "jobs": len(jobs)}
        if args.action_probe_only:
            write_json(OUT_ROOT / "action_probe_plan.json", planned)
        else:
            write_json(OUT_ROOT / "summary.json", planned)
        return
    if args.summarize_only:
        results = load_existing_job_results(jobs)
    else:
        results: list[dict[str, Any]] = []
        max_workers = max(1, min(len(jobs), int(args.max_workers) if int(args.max_workers) > 0 else len(gpus)))
        gpu_locks = {gpu: Lock() for gpu in set(gpus)}

        def run_job_with_gpu_lock(case: dict[str, Any], variant: str, gpu: int) -> dict[str, Any]:
            with gpu_locks[gpu]:
                return run_job(case, variant, gpu)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(run_job_with_gpu_lock, case, variant, gpu): (case, variant, gpu)
                for case, variant, gpu in jobs
            }
            for fut in as_completed(future_map):
                results.append(fut.result())
        results = sorted(results, key=lambda row: (row["case_id"], row["variant"]))
    write_csv(OUT_ROOT / "job_results.csv", results)
    previous_summary = read_json(OUT_ROOT / "summary.json") if args.action_probe_only else {}
    if args.action_probe_only:
        comparisons = previous_summary.get("comparisons", [])
        semantic_alignments = previous_summary.get("semantic_alignments", [])
    else:
        comparisons = [compare_case(case) for case in cases]
        write_csv(OUT_ROOT / "parity_results.csv", comparisons)
        semantic_alignments = [semantic_alignment_case(case) for case in cases]
        write_csv(OUT_ROOT / "semantic_alignment_results.csv", semantic_alignments)
    action_probes = [action_trace_probe_case(case) for case in cases] if (args.include_action_probe or args.action_probe_only) else []
    write_csv(OUT_ROOT / "action_trace_probe_results.csv", action_probes)
    summary = {
        "status": "complete",
        "dump_head_marginal": bool(args.dump_head_marginal),
        "dump_full_query_marginal": bool(DUMP_FULL_QUERY_MARGINAL),
        "sampled_pairwise_dump": bool(args.sampled_pairwise_dump),
        "per_gpu_serial_lock": True,
        "case_count": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "jobs": len(results),
        "job_returncode_all_zero": all(int(row["returncode"]) == 0 for row in results),
        "trace_smoke_pass_all_cases": (
            previous_summary.get("trace_smoke_pass_all_cases")
            if args.action_probe_only
            else all(bool(row["trace_smoke_pass"]) for row in comparisons)
        ),
        "semantic_alignment_smoke_pass_all_cases": (
            previous_summary.get("semantic_alignment_smoke_pass_all_cases")
            if args.action_probe_only
            else
            all(bool(row["semantic_alignment_smoke_pass"]) for row in semantic_alignments)
        ),
        "action_trace_probe_pass_all_cases": (
            all(bool(row["action_trace_probe_pass"]) for row in action_probes)
            if action_probes else None
        ),
        "comparisons": comparisons,
        "semantic_alignments": semantic_alignments,
        "action_trace_probes": action_probes,
    }
    write_json(OUT_ROOT / "summary.json", summary)


if __name__ == "__main__":
    main()
