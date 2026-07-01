#!/usr/bin/env python3
"""Run v98 Stage1 SWA cache/top-k trace extension jobs.

This runner is diagnostic-only.  It reuses the existing v96/v97 READ_NO_ACTION
trace path and only enables SWA raw transport trace dumps; it does not enable
old Track E source-gate/source-replace actions and does not claim method
success.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control")
DEFAULT_OUT = ROOT / "stage1_k_swa_trace_extension"
V95_CASES = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control/trackA_base_case_bank/canonical_case_rows.csv")
V97_CORE = Path(
    "results/acl2_v97tf_semantic_scale_evidence_gauge_safe_memory_control/"
    "trackK_semantic_scale_evidence_eligibility/swa_strict_stable_fallback_audit_rows.csv"
)


def load_local_v96_runner() -> Any:
    path = REPO_ROOT / "tools/run_v96tf_j4_read_weak_context_skip_pilot.py"
    spec = importlib.util.spec_from_file_location("loger_local_run_v96tf_j4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load local runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_local_v96_runner()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def core_case_ids() -> set[str]:
    return {row.get("case_id", "") for row in read_rows(V97_CORE) if row.get("case_id")}


def is_good(row: dict[str, str]) -> bool:
    labels = str(row.get("action_response_labels", ""))
    return (
        "GOOD_PROTECTION" in labels
        or str(row.get("v95_case_bucket", "")).strip() == "GOOD_PROTECTION"
        or str(row.get("case_label_offline_only", "")).strip().lower() == "good"
    )


def is_handoff_candidate(row: dict[str, str]) -> bool:
    labels = str(row.get("action_response_labels", ""))
    failure = str(row.get("failure_type_primary", ""))
    return "SWA_HANDOFF_CANDIDATE" in labels or "HANDOFF" in failure


def round_robin_by_seq(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    by_seq: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_seq.setdefault(str(row.get("seq", "")), []).append(row)
    selected: list[dict[str, str]] = []
    while len(selected) < limit:
        progressed = False
        for seq in sorted(by_seq):
            if by_seq[seq] and len(selected) < limit:
                selected.append(by_seq[seq].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def select_extension_cases(max_bad: int, max_good: int, boundary_context: str, good_control_selection: str) -> list[dict[str, Any]]:
    core = core_case_ids()
    rows = [row for row in read_rows(V95_CASES) if row.get("case_id") and row.get("case_id") not in core]
    rows = [row for row in rows if finite(row.get("L3_handoff_transfer_penalty_proxy")) is not None]
    good_rows = [row for row in rows if is_good(row)]
    bad_rows = [row for row in rows if not is_good(row) and is_handoff_candidate(row)]
    if len(bad_rows) < max_bad:
        seen = {row.get("case_id", "") for row in bad_rows}
        bad_rows.extend(
            row for row in rows
            if row.get("case_id", "") not in seen
            and not is_good(row)
            and str(row.get("failure_type_primary", "")).strip() not in {"", "SAFE_OR_UNASSIGNED"}
        )
        seen = {row.get("case_id", "") for row in bad_rows}
        bad_rows.extend(
            row for row in rows
            if row.get("case_id", "") not in seen
            and not is_good(row)
            and str(row.get("case_label_offline_only", "")).strip().lower() != "good"
        )

    def bad_sort_key(row: dict[str, str]) -> tuple[str, float]:
        return (str(row.get("seq", "")), -(finite(row.get("L3_handoff_transfer_penalty_proxy")) or 0.0))

    def good_sort_key(row: dict[str, str]) -> tuple[str, float]:
        l3 = finite(row.get("L3_handoff_transfer_penalty_proxy"))
        if good_control_selection == "legacy_seq_high_l3":
            return (str(row.get("seq", "")), -(l3 or 0.0))
        return (str(row.get("seq", "")), l3 if l3 is not None else float("inf"))

    good_rows = sorted(good_rows, key=good_sort_key)
    bad_rows = sorted(bad_rows, key=bad_sort_key)
    selected_bad = round_robin_by_seq(bad_rows, max_bad)
    selected_good = round_robin_by_seq(good_rows, max_good)

    cases: list[dict[str, Any]] = []
    for row in selected_bad:
        cases.append(base.case_from_atlas(row, "SWA_HANDOFF_NON_GOOD_EXT", boundary_context=boundary_context))
    for row in selected_good:
        cases.append(base.case_from_atlas(row, "SWA_HANDOFF_GOOD_CONTROL_EXT", boundary_context=boundary_context))
    return cases


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
        v68_export_full_pca_debug=0,
        v68_layer_pca_feature_subdir="pca_features",
        v68_pca_taps="pca_attn_global_k_layers,pca_attn_global_v_layers,pca_attn_frame_v_layers",
        v68_pca_layers="5,13,17",
        v68_pca_max_feature_dim=8,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--max-bad-cases", type=int, default=8)
    parser.add_argument("--max-good-cases", type=int, default=4)
    parser.add_argument("--boundary-context", choices=("prev_curr", "current"), default="prev_curr")
    parser.add_argument(
        "--good-control-selection",
        choices=("hygiene_low_l3", "legacy_seq_high_l3"),
        default="hygiene_low_l3",
        help="Select extension good controls by low L3 hygiene, or reproduce the original high-L3 sequence sort.",
    )
    parser.add_argument("--swa-raw-transport-trace-layer-mode", default="all", choices=("all", "first", "last", "single"))
    parser.add_argument("--swa-raw-transport-trace-single-layer", type=int, default=-1)
    parser.add_argument("--swa-raw-transport-trace-max-queries", type=int, default=128)
    parser.add_argument("--swa-raw-transport-trace-topk", type=int, default=8)
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

    cases = select_extension_cases(args.max_bad_cases, args.max_good_cases, args.boundary_context, args.good_control_selection)
    base_args = args_for_base(args)
    jobs = base.build_jobs(cases, ["READ_NO_ACTION"], gpus, base_args)
    manifest = {
        "schema": "acl2_v98_stage1_swa_cache_topk_trace_extension_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "old_track_e_action_enabled": False,
        "source_case_bank": str(V95_CASES),
        "excluded_v97_core_case_count": len(core_case_ids()),
        "case_selection_policy": {
            "bad_controls": "seq_round_robin_high_L3_non_good_handoff_or_failure",
            "good_controls": args.good_control_selection,
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "selected_cases": cases,
        "jobs": [{key: value for key, value in job.items() if key != "cmd"} for job in jobs],
    }
    base.write_json(args.output_root / "trace_extension_manifest.json", manifest)
    base.write_csv(args.output_root / "job_manifest.csv", [{key: value for key, value in job.items() if key != "cmd"} for job in jobs])
    if args.no_run:
        base.write_json(args.output_root / "summary.json", {"status": "planned_not_run", "planned_jobs": len(jobs), "gate_pass": False})
        return
    if args.summarize_only:
        completed = []
        for job in jobs:
            path = Path(job["out_dir"]) / "job_summary.json"
            completed.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else job)
    else:
        completed = base.run_jobs(jobs, gpus)
    rows = []
    for job in completed:
        stats = base.extract_trace_stats(Path(job["out_dir"]))
        row = {key: value for key, value in job.items() if key != "cmd"}
        row.update(stats)
        rows.append(row)
    trace_files = list(args.output_root.glob("*/READ_NO_ACTION/swa_raw_transport_trace/*.pt"))
    summary = {
        "schema": "acl2_v98_stage1_swa_cache_topk_trace_extension_summary_v1",
        "status": "complete" if all(int(job.get("returncode", 1)) == 0 for job in completed) else "partial_or_failed",
        "diagnostic_only": True,
        "selected_case_count": len(cases),
        "completed_job_count": len(completed),
        "failed_job_count": sum(1 for job in completed if int(job.get("returncode", 1)) != 0),
        "trace_payload_file_count": len(trace_files),
        "case_with_trace_count": len({path.parents[2].name for path in trace_files}),
        "gate_pass": False,
        "gate_pass_note": "This runner only collects Stage1 traces; build_v98tf... evaluates the Stage1 gate.",
    }
    base.write_csv(args.output_root / "job_results.csv", [{key: value for key, value in job.items() if key != "cmd"} for job in completed])
    base.write_csv(args.output_root / "trace_rows.csv", rows)
    base.write_json(args.output_root / "summary.json", summary)


if __name__ == "__main__":
    main()
