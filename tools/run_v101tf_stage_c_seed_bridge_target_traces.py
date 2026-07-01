#!/usr/bin/env python3
"""Run v101 target-universe no-action traces for Stage-C seed bridge auditing.

This runner reuses the existing v96/v98 READ_NO_ACTION pipeline command shape
and only enables SWA raw transport trace dumps.  It is diagnostic-only: beta is
zero, Stage-C cache is read-only, and no runtime action is authorized.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
DEFAULT_TARGET = ROOT / "trackT_drift_target_relabel/target_universe_v101.csv"
DEFAULT_OUT = ROOT / "stage_c_seed_bridge_target_traces"


def load_v96_runner() -> Any:
    path = REPO_ROOT / "tools/run_v96tf_j4_read_weak_context_skip_pilot.py"
    spec = importlib.util.spec_from_file_location("loger_local_run_v96tf_j4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load local runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_v96_runner()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_fail_forward_docs(output_root: Path, *, status: str, selected_case_count: int, planned_jobs: int) -> None:
    write_text(
        output_root / "failure_report.md",
        "# Stage-C Seed Trace Diagnostic\n\n"
        f"- status: {status}\n"
        f"- selected_case_count: {selected_case_count}\n"
        f"- planned_jobs: {planned_jobs}\n"
        "- diagnostic_only: true\n"
        "- runtime_action_allowed: false\n\n"
        "This directory contains no-action trace materialization only. It is not a runtime method success claim.",
    )
    write_text(
        output_root / "what_would_have_to_be_true_to_pass.md",
        "A trace directory can only support downstream action after strict current support, scale observability, Q2 true-stage, and M4 simulator gates pass.",
    )
    write_text(
        output_root / "control_gap_report.md",
        "The control gap is unchanged: beta_frame=0 and READ_NO_ACTION were used, so no L3/runtime action effect is evaluated here.",
    )
    write_text(
        output_root / "next_attempt_recommendation.md",
        "Use trace payloads as provenance/current-support diagnostics; do not run runtime action unless strict Track U/V/Q2/M4 prerequisites are satisfied.",
    )
    write_csv(
        output_root / "false_positive_false_negative_rows.csv",
        [
            {
                "cue_name": "stage_c_seed_trace_materialization",
                "row_kind": "not_selector_trace_only",
                "selected_case_count": selected_case_count,
                "planned_jobs": planned_jobs,
                "reason": "No FP/FN selector is defined by this no-action trace directory.",
                "claim_level": "diagnostic_no_action",
            }
        ],
    )


def parse_case_id(case_id: str) -> tuple[str, int, int]:
    parts = str(case_id).split("_")
    if len(parts) < 3:
        raise ValueError(f"cannot parse v101 case_id: {case_id}")
    return f"{int(parts[0]):02d}", int(parts[1]), int(parts[2])


def case_from_target(row: dict[str, str], *, boundary_context: str) -> dict[str, Any]:
    seq, prev_chunk, curr_chunk = parse_case_id(row.get("case_id", ""))
    atlas_row = dict(row)
    atlas_row["seq"] = seq
    atlas_row["prev_chunk"] = str(prev_chunk)
    atlas_row["curr_chunk"] = str(curr_chunk)
    atlas_row.setdefault("L1_local_sim3_ate", "")
    atlas_row.setdefault("L2_head_tail_proxy_error", "")
    atlas_row.setdefault("L3_J_handoff", "")
    atlas_row.setdefault("v96_recommended_next_track", "")
    atlas_row.setdefault("action_response_labels", "")
    bucket = f"V101_{row.get('target_taxonomy', 'UNKNOWN')}"
    case = base.case_from_atlas(atlas_row, bucket, boundary_context=boundary_context)
    case["target_taxonomy"] = row.get("target_taxonomy", "")
    case["target_reason"] = row.get("target_reason", "")
    case["case_label"] = row.get("case_label", "")
    case["failure_type"] = row.get("failure_type", "")
    return case


def args_for_base(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        skip_existing=args.skip_existing,
        frame_region="all",
        query_region="all",
        soft_rho=0.0,
        soft_min_keep=1.0,
        layer_mode="early",
        single_layer=-1,
        head_indices="",
        attention_mass_max_queries=128,
        semantic_anchor_mode="semantic",
        semantic_anchor_target_ratio=0.12,
        semantic_anchor_min_ratio=0.03,
        semantic_anchor_max_ratio=0.30,
        semantic_anchor_min_score=0.02,
        semantic_anchor_missing_trust_policy="zero",
        semantic_anchor_value_fallback="off",
        swa_raw_transport_trace_dir="swa_raw_transport_trace",
        swa_raw_transport_trace_layer_mode=args.swa_raw_transport_trace_layer_mode,
        swa_raw_transport_trace_single_layer=args.swa_raw_transport_trace_single_layer,
        swa_raw_transport_trace_max_queries=args.swa_raw_transport_trace_max_queries,
        swa_raw_transport_trace_topk=args.swa_raw_transport_trace_topk,
        swa_raw_transport_trace_direct_match_only=args.swa_raw_transport_trace_direct_match_only,
        swa_raw_transport_trace_query_block_size=args.swa_raw_transport_trace_query_block_size,
        v68_export_full_pca_debug=0,
        v68_layer_pca_feature_subdir="pca_features",
        v68_pca_taps="pca_attn_global_k_layers,pca_attn_global_v_layers,pca_attn_frame_v_layers",
        v68_pca_layers="5,13,17",
        v68_pca_max_feature_dim=8,
    )


def replace_flag_value(cmd: list[str], flag: str, value: str) -> None:
    try:
        idx = cmd.index(flag)
    except ValueError:
        cmd.extend([flag, value])
        return
    if idx + 1 >= len(cmd):
        cmd.append(value)
    else:
        cmd[idx + 1] = value


def apply_command_overrides(jobs: list[dict[str, Any]], args: argparse.Namespace) -> None:
    diag_value = args.ttt_write_token_contribution_diagnostic
    if diag_value is None:
        diag_value = 1 if str(args.hmc_commit_mode) == "probe_ttt_write" else 0
    for job in jobs:
        cmd = job.get("cmd")
        if not isinstance(cmd, list):
            continue
        replace_flag_value(cmd, "--hybrid_memory_mode", str(args.hybrid_memory_mode))
        replace_flag_value(cmd, "--hmc_commit_mode", str(args.hmc_commit_mode))
        replace_flag_value(cmd, "--ttt_write_token_contribution_diagnostic", str(int(diag_value)))
        replace_flag_value(cmd, "--enable_v102_state_machine_trace", str(int(args.enable_v102_state_machine_trace)))
        replace_flag_value(cmd, "--v102_state_machine_action", str(args.v102_state_machine_action))
        replace_flag_value(cmd, "--v102_state_machine_layer_mode", str(args.v102_state_machine_layer_mode))
        replace_flag_value(cmd, "--v102_state_machine_single_layer", str(int(args.v102_state_machine_single_layer)))
        replace_flag_value(cmd, "--v102_state_machine_strict_gate_pass", str(int(args.v102_state_machine_strict_gate_pass)))
        replace_flag_value(cmd, "--v102_state_machine_true_l3_gate_pass", str(int(args.v102_state_machine_true_l3_gate_pass)))
        replace_flag_value(cmd, "--enable_v102_state_machine_action_probe", str(int(args.enable_v102_state_machine_action_probe)))
        replace_flag_value(cmd, "--v102_state_machine_probe_impl", str(args.v102_state_machine_probe_impl))
        replace_flag_value(cmd, "--v102_state_machine_unreliable_d_min", str(float(args.v102_state_machine_unreliable_d_min)))
        replace_flag_value(cmd, "--v102_state_machine_unreliable_g_min", str(float(args.v102_state_machine_unreliable_g_min)))
        replace_flag_value(cmd, "--v102_state_machine_supported_d_max", str(float(args.v102_state_machine_supported_d_max)))
        replace_flag_value(cmd, "--v102_state_machine_supported_k_min", str(float(args.v102_state_machine_supported_k_min)))
        replace_flag_value(
            cmd,
            "--v102_state_machine_supported_require_static_semantic",
            str(int(args.v102_state_machine_supported_require_static_semantic)),
        )
        replace_flag_value(
            cmd,
            "--v102_state_machine_soft_unsupported_min_keep",
            str(float(args.v102_state_machine_soft_unsupported_min_keep)),
        )
        replace_flag_value(cmd, "--v102_state_machine_hold_prev_frames", str(int(args.v102_state_machine_hold_prev_frames)))
        replace_flag_value(cmd, "--v102_state_machine_hold_soft_min_keep", str(float(args.v102_state_machine_hold_soft_min_keep)))
        replace_flag_value(cmd, "--v102_state_machine_delay_current_soft_min_keep", str(float(args.v102_state_machine_delay_current_soft_min_keep)))
        replace_flag_value(cmd, "--v102_state_machine_context_soft_min_keep", str(float(args.v102_state_machine_context_soft_min_keep)))
        replace_flag_value(cmd, "--v102_state_machine_min_history_keep_frac", str(float(args.v102_state_machine_min_history_keep_frac)))
        replace_flag_value(cmd, "--v102_state_machine_attention_mass_max_queries", str(int(args.v102_state_machine_attention_mass_max_queries)))
        replace_flag_value(
            cmd,
            "--enable_swa_prev_ttt_tracked_instance_query_soft_trace",
            str(int(args.enable_swa_prev_ttt_tracked_instance_query_soft_trace)),
        )
        replace_flag_value(
            cmd,
            "--swa_prev_ttt_tracked_instance_query_soft_rho",
            str(float(args.swa_prev_ttt_tracked_instance_query_soft_rho)),
        )
        replace_flag_value(
            cmd,
            "--swa_prev_ttt_tracked_instance_query_soft_min_keep",
            str(float(args.swa_prev_ttt_tracked_instance_query_soft_min_keep)),
        )
        replace_flag_value(
            cmd,
            "--swa_prev_ttt_tracked_instance_query_soft_query_head_frac_threshold",
            str(float(args.swa_prev_ttt_tracked_instance_query_soft_query_head_frac_threshold)),
        )
        replace_flag_value(
            cmd,
            "--swa_prev_ttt_tracked_instance_query_soft_topk",
            str(int(args.swa_prev_ttt_tracked_instance_query_soft_topk)),
        )
        replace_flag_value(
            cmd,
            "--swa_prev_ttt_tracked_instance_query_soft_query_block_size",
            str(int(args.swa_prev_ttt_tracked_instance_query_soft_query_block_size)),
        )
        replace_flag_value(
            cmd,
            "--swa_prev_ttt_tracked_instance_query_soft_layer_mode",
            str(args.swa_prev_ttt_tracked_instance_query_soft_layer_mode),
        )
        replace_flag_value(
            cmd,
            "--swa_prev_ttt_tracked_instance_query_soft_single_layer",
            str(int(args.swa_prev_ttt_tracked_instance_query_soft_single_layer)),
        )
        replace_flag_value(
            cmd,
            "--swa_prev_ttt_tracked_instance_query_soft_attention_mass_max_queries",
            str(int(args.swa_prev_ttt_tracked_instance_query_soft_attention_mass_max_queries)),
        )
        replace_flag_value(
            cmd,
            "--swa_prev_ttt_tracked_instance_query_soft_min_direct_witness_seeds",
            str(int(args.swa_prev_ttt_tracked_instance_query_soft_min_direct_witness_seeds)),
        )
        replace_flag_value(
            cmd,
            "--swa_prev_ttt_tracked_instance_query_soft_direct_match_mode",
            str(args.swa_prev_ttt_tracked_instance_query_soft_direct_match_mode),
        )
        if int(args.enable_per_chunk_geometry_sidecar):
            out_dir = Path(str(job.get("out_dir", "")))
            replace_flag_value(cmd, "--per_chunk_geometry_dir", str(out_dir / "per_chunk_geometry"))
            replace_flag_value(cmd, "--per_chunk_pose_trace_jsonl", str(out_dir / "per_chunk_pose_trace.jsonl"))
        job["ttt_write_token_contribution_diagnostic"] = int(diag_value)
        job["per_chunk_geometry_sidecar_enabled"] = bool(int(args.enable_per_chunk_geometry_sidecar))
        job["cmd_shell"] = shlex.join(cmd)
        if str(args.hmc_commit_mode) != "controlled":
            job["semantic_contract"] = (
                f"{job.get('semantic_contract', '')}; diagnostic hmc_commit_mode={args.hmc_commit_mode}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--boundary-context", choices=("prev_curr", "current"), default="prev_curr")
    parser.add_argument("--taxonomy", default="all", help="Comma-separated target_taxonomy filter, or all.")
    parser.add_argument("--case-ids", default="", help="Comma-separated explicit case_id filter applied after taxonomy.")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all selected rows.")
    parser.add_argument("--hybrid-memory-mode", default="read_path_only")
    parser.add_argument("--hmc-commit-mode", default="controlled", choices=("controlled", "probe_native", "split_ttt_native", "probe_ttt_write"))
    parser.add_argument(
        "--ttt-write-token-contribution-diagnostic",
        dest="ttt_write_token_contribution_diagnostic",
        type=int,
        choices=(0, 1),
        default=None,
        help="Override --ttt_write_token_contribution_diagnostic. Defaults to 1 for probe_ttt_write and 0 otherwise.",
    )
    parser.add_argument("--swa-raw-transport-trace-layer-mode", default="first", choices=("all", "first", "last", "single"))
    parser.add_argument("--swa-raw-transport-trace-single-layer", type=int, default=-1)
    parser.add_argument("--swa-raw-transport-trace-max-queries", type=int, default=16)
    parser.add_argument("--swa-raw-transport-trace-topk", type=int, default=4)
    parser.add_argument("--swa-raw-transport-trace-direct-match-only", type=int, choices=(0, 1), default=0)
    parser.add_argument("--swa-raw-transport-trace-query-block-size", type=int, default=128)
    parser.add_argument("--enable-v102-state-machine-trace", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--v102-state-machine-action",
        choices=(
            "TRANSMIT_SUPPORTED_ANCHORS",
            "REJECT_UNRELIABLE_ANCHORS",
            "DELAY_UPDATE",
            "HOLD_PREV_REFERENCE",
            "CONTEXT_ONLY_DEMOTION",
            "WRITE_CONFIRMED_ANCHORS_ONLY",
            "EXPIRE_UNSUPPORTED_STALE_ANCHORS",
            "REFRESH_SUPPORTED_STALE_ANCHORS",
            "WRITE_CONTEXT_ONLY",
        ),
        default="TRANSMIT_SUPPORTED_ANCHORS",
    )
    parser.add_argument("--v102-state-machine-layer-mode", choices=("all", "first", "last", "single"), default="last")
    parser.add_argument("--v102-state-machine-single-layer", type=int, default=-1)
    parser.add_argument("--v102-state-machine-strict-gate-pass", type=int, choices=(0, 1), default=0)
    parser.add_argument("--v102-state-machine-true-l3-gate-pass", type=int, choices=(0, 1), default=0)
    parser.add_argument("--enable-v102-state-machine-action-probe", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--v102-state-machine-probe-impl",
        choices=(
            "compact_kv_reject_unreliable",
            "compact_kv_transmit_supported",
            "source_soft_transmit_supported",
            "compact_kv_hold_prev_reference",
            "source_soft_hold_prev_reference",
            "compact_kv_delay_update",
            "source_soft_delay_update",
            "source_soft_context_only_demotion",
        ),
        default="compact_kv_reject_unreliable",
    )
    parser.add_argument("--v102-state-machine-unreliable-d-min", type=float, default=0.50)
    parser.add_argument("--v102-state-machine-unreliable-g-min", type=float, default=0.50)
    parser.add_argument("--v102-state-machine-supported-d-max", type=float, default=0.25)
    parser.add_argument("--v102-state-machine-supported-k-min", type=float, default=0.0)
    parser.add_argument("--v102-state-machine-supported-require-static-semantic", type=int, choices=(0, 1), default=1)
    parser.add_argument("--v102-state-machine-soft-unsupported-min-keep", type=float, default=0.50)
    parser.add_argument("--v102-state-machine-hold-prev-frames", type=int, default=1)
    parser.add_argument("--v102-state-machine-hold-soft-min-keep", type=float, default=0.50)
    parser.add_argument("--v102-state-machine-delay-current-soft-min-keep", type=float, default=0.50)
    parser.add_argument("--v102-state-machine-context-soft-min-keep", type=float, default=0.50)
    parser.add_argument("--v102-state-machine-min-history-keep-frac", type=float, default=0.05)
    parser.add_argument("--v102-state-machine-attention-mass-max-queries", type=int, default=64)
    parser.add_argument("--enable-swa-prev-ttt-tracked-instance-query-soft-trace", type=int, choices=(0, 1), default=0)
    parser.add_argument("--swa-prev-ttt-tracked-instance-query-soft-rho", type=float, default=0.0)
    parser.add_argument("--swa-prev-ttt-tracked-instance-query-soft-min-keep", type=float, default=0.5)
    parser.add_argument("--swa-prev-ttt-tracked-instance-query-soft-query-head-frac-threshold", type=float, default=0.75)
    parser.add_argument("--swa-prev-ttt-tracked-instance-query-soft-topk", type=int, default=4)
    parser.add_argument("--swa-prev-ttt-tracked-instance-query-soft-query-block-size", type=int, default=64)
    parser.add_argument(
        "--swa-prev-ttt-tracked-instance-query-soft-layer-mode",
        choices=("all", "first", "last", "single"),
        default="first",
    )
    parser.add_argument("--swa-prev-ttt-tracked-instance-query-soft-single-layer", type=int, default=-1)
    parser.add_argument("--swa-prev-ttt-tracked-instance-query-soft-attention-mass-max-queries", type=int, default=64)
    parser.add_argument("--swa-prev-ttt-tracked-instance-query-soft-min-direct-witness-seeds", type=int, default=4)
    parser.add_argument(
        "--swa-prev-ttt-tracked-instance-query-soft-direct-match-mode",
        choices=("any", "same_seed", "same_masklet"),
        default="any",
    )
    parser.add_argument(
        "--enable-per-chunk-geometry-sidecar",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Diagnostic-only: append run_pipeline_abc_v2 --per_chunk_geometry_dir "
            "and --per_chunk_pose_trace_jsonl for lifecycle-aligned geometry smoke runs."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    base.OUT_ROOT = args.output_root
    gpus = [int(item) for item in str(args.gpus).split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must list at least one GPU")
    taxonomy_filter = {item.strip() for item in str(args.taxonomy).split(",") if item.strip()}
    rows = read_rows(args.target_csv)
    if taxonomy_filter and taxonomy_filter != {"all"}:
        rows = [row for row in rows if row.get("target_taxonomy", "") in taxonomy_filter]
    case_filter = {item.strip() for item in str(args.case_ids).split(",") if item.strip()}
    if case_filter:
        rows = [row for row in rows if row.get("case_id", "") in case_filter]
    if int(args.max_cases) > 0:
        rows = rows[: int(args.max_cases)]
    cases = [case_from_target(row, boundary_context=args.boundary_context) for row in rows]
    base_args = args_for_base(args)
    jobs = base.build_jobs(cases, ["READ_NO_ACTION"], gpus, base_args)
    apply_command_overrides(jobs, args)
    manifest = {
        "schema": "acl2_v101_stage_c_seed_bridge_target_trace_manifest_v1",
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "target_csv": str(args.target_csv),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "selected_case_count": len(cases),
        "selected_cases": cases,
        "jobs": [{key: value for key, value in job.items() if key != "cmd"} for job in jobs],
    }
    write_json(args.output_root / "target_trace_manifest.json", manifest)
    write_csv(args.output_root / "job_manifest.csv", [{key: value for key, value in job.items() if key != "cmd"} for job in jobs])
    if args.no_run:
        write_json(
            args.output_root / "summary.json",
            {
                "schema": "acl2_v101_stage_c_seed_bridge_target_trace_summary_v1",
                "status": "planned_not_run",
                "planned_jobs": len(jobs),
                "diagnostic_only": True,
                "runtime_action_allowed": False,
                "gate_pass": False,
            },
        )
        write_fail_forward_docs(
            args.output_root,
            status="planned_not_run",
            selected_case_count=len(cases),
            planned_jobs=len(jobs),
        )
        return
    if args.summarize_only:
        completed = []
        for job in jobs:
            path = Path(job["out_dir"]) / "job_summary.json"
            completed.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else job)
    else:
        completed = base.run_jobs(jobs, gpus)
    trace_rows = []
    for job in completed:
        stats = base.extract_trace_stats(Path(job["out_dir"]))
        row = {key: value for key, value in job.items() if key != "cmd"}
        row.update(stats)
        trace_rows.append(row)
    trace_files = list(args.output_root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt"))
    geometry_sidecars = list(args.output_root.glob("*/READ_NO_ACTION/per_chunk_geometry/chunk_*.pt"))
    write_csv(args.output_root / "job_results.csv", [{key: value for key, value in job.items() if key != "cmd"} for job in completed])
    write_csv(args.output_root / "trace_rows.csv", trace_rows)
    write_json(
        args.output_root / "summary.json",
        {
            "schema": "acl2_v101_stage_c_seed_bridge_target_trace_summary_v1",
            "status": "complete" if all(int(job.get("returncode", 1)) == 0 for job in completed) else "partial_or_failed",
            "diagnostic_only": True,
            "runtime_action_allowed": False,
            "selected_case_count": len(cases),
            "completed_job_count": len(completed),
            "failed_job_count": sum(1 for job in completed if int(job.get("returncode", 1)) != 0),
            "trace_payload_file_count": len(trace_files),
            "case_with_trace_count": len({path.parents[2].name for path in trace_files}),
            "per_chunk_geometry_sidecar_enabled": bool(int(args.enable_per_chunk_geometry_sidecar)),
            "per_chunk_geometry_sidecar_file_count": len(geometry_sidecars),
            "case_with_per_chunk_geometry_sidecar_count": len({path.parents[2].name for path in geometry_sidecars}),
            "gate_pass": False,
            "gate_pass_note": "Target traces only materialize Stage-C seed provenance; no action gate is evaluated.",
        },
    )
    write_fail_forward_docs(
        args.output_root,
        status="complete" if all(int(job.get("returncode", 1)) == 0 for job in completed) else "partial_or_failed",
        selected_case_count=len(cases),
        planned_jobs=len(jobs),
    )


if __name__ == "__main__":
    main()
