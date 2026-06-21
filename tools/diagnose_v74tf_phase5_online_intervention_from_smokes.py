#!/usr/bin/env python3
"""Phase 5 online memory-intervention aggregation from real smoke traces.

This tool does not rerun inference and does not synthesize component evidence.
It closes the Phase 5 bookkeeping gap where only offline Phase 4 rows were
wrapped as "counterfactual" evidence. Inputs are real online smoke CSV files
that already contain native/candidate/control trajectories and hook traces.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from v73_semantic_memory_common import finite_mean, finite_median, read_csv, safe_float, utc_now, write_csv, write_json, write_text
from v74tf_common import V74TF_ROOT


DEFAULT_OUT = V74TF_ROOT / "report_final" / "phase5_online_memory_intervention_after_nA_repairs"
DEFAULT_CONTEXT = (
    V74TF_ROOT
    / "phase4_extra_nA_context_anchor_boost_midtail_rho010_top4_semprior_min001"
    / "context_anchor_boost_smoke_results.csv"
)
DEFAULT_CONTEXT_TAIL = (
    V74TF_ROOT
    / "phase4_extra_nA_context_anchor_boost_tail_rho010_top4_semprior_min001"
    / "context_anchor_boost_smoke_results.csv"
)
DEFAULT_CONTEXT_TAIL_ALL = (
    V74TF_ROOT
    / "phase4_extra_nA_context_anchor_boost_tail_alllayers_rho010_top4_semprior_min001"
    / "context_anchor_boost_smoke_results.csv"
)
DEFAULT_CONTEXT_MIDTAIL_ALL = (
    V74TF_ROOT
    / "phase4_extra_nA_context_anchor_boost_midtail_alllayers_rho010_top4_semprior_min001"
    / "context_anchor_boost_smoke_results.csv"
)
DEFAULT_SWA = (
    V74TF_ROOT
    / "phase4_extra_nA_swa_overlap_replace_semstruct_q90_alpha010_top4"
    / "radio_swa_online_smoke_results.csv"
)
DEFAULT_SWA_LEAVE_ONE_OUT = (
    V74TF_ROOT
    / "phase5_component_leave_one_out_swa_turnoff_top4"
    / "radio_swa_online_smoke_results.csv"
)
DEFAULT_SWA_GATE = (
    V74TF_ROOT
    / "phase4_extra_nA_swa_overlap_gate_semstruct_q90_rho010_min090_top4"
    / "radio_swa_online_smoke_results.csv"
)
DEFAULT_READ_PAIR = (
    V74TF_ROOT
    / "phase4_extra_nA_online_read_smoke_top4_lam010_all_beta010_pair"
    / "radio_read_online_smoke_results.csv"
)
DEFAULT_READ_QUERY = (
    V74TF_ROOT
    / "phase4_extra_nA_online_read_smoke_top4_lam010_all_beta010_query"
    / "radio_read_online_smoke_results.csv"
)
DEFAULT_TTT_NO_PERSISTENT = (
    V74TF_ROOT
    / "phase5_harmful_no_persistent_ttt_dynamic_lowstable_top4"
    / "ttt_write_online_smoke_results.csv"
)
DEFAULT_REFRESH_HOLD_FLIP = (
    V74TF_ROOT
    / "phase5_refresh_hold_flip_radio_qscale_holdalpha005_top4"
    / "refresh_hold_flip_online_smoke_results.csv"
)
DEFAULT_REFRESH_HOLD_FLIP_09 = (
    V74TF_ROOT
    / "phase5_refresh_hold_flip_09_radio_qscale_holdalpha005_top8"
    / "refresh_hold_flip_online_smoke_results.csv"
)


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _metric(row: dict[str, str]) -> float | None:
    for key in ("target_chunk_ATE", "ATE_horizon"):
        val = safe_float(row.get(key))
        if val is not None:
            return float(val)
    return None


def _chunk(row: dict[str, str]) -> int | None:
    try:
        return int(str(row.get("chunk", "")).strip())
    except ValueError:
        return None


def _finite_min(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in (safe_float(x) for x in values) if v is not None]
    return min(vals) if vals else None


def _hook_signal(row: dict[str, str], family: str) -> dict[str, Any]:
    if family.startswith("stable_anchor_floor"):
        return {
            "delta_attention_mass": row.get("context_source_boost_tokens_max"),
            "delta_SWA_mass": None,
            "delta_merge_weight": None,
            "delta_TTT_write": None,
            "hook_mass_field": "context_source_boost_tokens_max",
            "hook_mass_value": row.get("context_source_boost_tokens_max"),
        }
    if family == "component_boost_SWA":
        return {
            "delta_attention_mass": None,
            "delta_SWA_mass": row.get("swa_overlap_source_replace_alpha_mean_max"),
            "delta_merge_weight": None,
            "delta_TTT_write": None,
            "hook_mass_field": "swa_overlap_source_replace_alpha_mean_max",
            "hook_mass_value": row.get("swa_overlap_source_replace_alpha_mean_max"),
        }
    if family == "component_leave_one_out_SWA":
        return {
            "delta_attention_mass": None,
            "delta_SWA_mass": row.get("turn_off_swa_effective_manifest"),
            "delta_merge_weight": None,
            "delta_TTT_write": None,
            "hook_mass_field": "turn_off_swa_effective_manifest",
            "hook_mass_value": row.get("turn_off_swa_effective_manifest"),
        }
    if family == "component_veto_SWA":
        return {
            "delta_attention_mass": None,
            "delta_SWA_mass": row.get("swa_overlap_source_gate_mean_max"),
            "delta_merge_weight": None,
            "delta_TTT_write": None,
            "hook_mass_field": "swa_overlap_source_gate_mean_max",
            "hook_mass_value": row.get("swa_overlap_source_gate_mean_max"),
        }
    if family == "harmful_no_persistent_TTT_if_available":
        return {
            "delta_attention_mass": None,
            "delta_SWA_mass": None,
            "delta_merge_weight": None,
            "delta_TTT_write": row.get("prior_v70_radio_ttt_changed_patch_frac"),
            "hook_mass_field": "prior_v70_radio_ttt_changed_patch_frac",
            "hook_mass_value": row.get("prior_v70_radio_ttt_changed_patch_frac"),
        }
    if family == "refresh_hold_flip":
        return {
            "delta_attention_mass": None,
            "delta_SWA_mass": None,
            "delta_merge_weight": row.get("semantic_merge_qscale_effective_blend_alpha_median"),
            "delta_TTT_write": None,
            "hook_mass_field": "semantic_merge_qscale_effective_blend_alpha_median",
            "hook_mass_value": row.get("semantic_merge_qscale_effective_blend_alpha_median"),
        }
    return {
        "delta_attention_mass": row.get("frame_attention_mean_abs_bias_max"),
        "delta_SWA_mass": None,
        "delta_merge_weight": None,
        "delta_TTT_write": None,
        "hook_mass_field": "frame_attention_mean_abs_bias_max",
        "hook_mass_value": row.get("frame_attention_mean_abs_bias_max"),
    }


def _analyze_source(
    *,
    rows_path: Path,
    intervention_type: str,
    memory_path: str,
    native_case: str,
    candidate_case: str = "candidate",
    seq: str = "01",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = read_csv(rows_path)
    chunks = sorted({c for c in (_chunk(row) for row in raw_rows) if c is not None})
    out_rows: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_rows = [row for row in raw_rows if _chunk(row) == chunk]
        native = next((row for row in chunk_rows if row.get("case") == native_case), None)
        candidate = next((row for row in chunk_rows if row.get("case") == candidate_case), None)
        if native is None or candidate is None:
            out_rows.append(
                {
                    "seq": seq,
                    "chunk_id": chunk,
                    "component_id": "",
                    "semantic_role": "",
                    "thingstuff_state": "",
                    "radio_component_stability": "",
                    "memory_path": memory_path,
                    "intervention_type": intervention_type,
                    "source_csv": str(rows_path),
                    "native_case": native_case,
                    "candidate_case": candidate_case,
                    "metric_name": "target_chunk_ATE_or_ATE_horizon",
                    "native_metric": None,
                    "candidate_metric": None,
                    "min_control_metric": None,
                    "delta_future_after_overlap": None,
                    "delta_head_to_tail": None,
                    "delta_scale_cv": None,
                    "delta_J_scale": None,
                    "candidate_hook_active": False,
                    "candidate_beats_all_controls": False,
                    "matched_random_control_available": False,
                    "matched_random_control_delta_ge_candidate": False,
                    "causal_support_row": False,
                    "diagnostic_scope": "online_smoke_counterfactual_trace_aggregated_not_new_inference",
                    "blocked_reason": "missing native or candidate row",
                }
            )
            continue
        native_metric = _metric(native)
        candidate_metric = _metric(candidate)
        control_rows = [row for row in chunk_rows if row.get("case") not in {native_case, candidate_case}]
        control_metrics = [_metric(row) for row in control_rows]
        min_control = _finite_min(control_metrics)
        delta = None if native_metric is None or candidate_metric is None else float(native_metric - candidate_metric)
        control_deltas = [
            float(native_metric - cm)
            for cm in control_metrics
            if native_metric is not None and cm is not None
        ]
        best_control_delta = max(control_deltas) if control_deltas else None
        beats_controls = bool(candidate_metric is not None and min_control is not None and candidate_metric < min_control)
        hook_active = _boolish(candidate.get("hook_active"))
        support = bool(candidate.get("returncode") == "0" and hook_active and delta is not None and delta > 0.0 and beats_controls)
        matched_random = bool(best_control_delta is not None and delta is not None and best_control_delta >= delta)
        hook = _hook_signal(candidate, intervention_type)
        out_rows.append(
            {
                "seq": seq,
                "chunk_id": chunk,
                "component_id": "",
                "semantic_role": candidate.get("requested_semantic_anchor_mode")
                or candidate.get("requested_read_cue_source")
                or candidate.get("prior_read_cue_source")
                or candidate.get("read_cue_source_effective_manifest")
                or "",
                "thingstuff_state": "",
                "radio_component_stability": candidate.get("prior_v70_radio_read_reason") or candidate.get("prior_semantic_anchor_reason") or "",
                "memory_path": memory_path,
                "intervention_type": intervention_type,
                "source_csv": str(rows_path),
                "native_case": native_case,
                "candidate_case": candidate_case,
                "control_cases": ",".join(str(row.get("case", "")) for row in control_rows),
                "metric_name": "target_chunk_ATE" if safe_float(candidate.get("target_chunk_ATE")) is not None else "ATE_horizon",
                "native_metric": native_metric,
                "candidate_metric": candidate_metric,
                "min_control_metric": min_control,
                "best_control_delta_J_scale": best_control_delta,
                "candidate_returncode": candidate.get("returncode"),
                "candidate_hook_active": hook_active,
                "candidate_beats_all_controls": beats_controls,
                "delta_attention_mass": hook["delta_attention_mass"],
                "delta_SWA_mass": hook["delta_SWA_mass"],
                "delta_merge_weight": hook["delta_merge_weight"],
                "delta_TTT_write": hook["delta_TTT_write"],
                "hook_mass_field": hook["hook_mass_field"],
                "hook_mass_value": hook["hook_mass_value"],
                "delta_future_after_overlap": None,
                "delta_head_to_tail": None,
                "delta_scale_cv": None,
                "delta_J_scale": delta,
                "matched_random_control_available": bool(control_rows),
                "matched_random_control_delta_ge_candidate": matched_random,
                "causal_support_row": support,
                "diagnostic_scope": "online_smoke_counterfactual_trace_aggregated_not_new_inference",
                "blocked_reason": "" if support else "not positive_vs_native_and_beating_controls",
            }
        )
    support_chunks = sorted(int(row["chunk_id"]) for row in out_rows if row.get("causal_support_row"))
    positive_chunks = sorted(
        int(row["chunk_id"])
        for row in out_rows
        if (safe_float(row.get("delta_J_scale")) is not None and float(row["delta_J_scale"]) > 0.0)
    )
    hook_chunks = sorted(int(row["chunk_id"]) for row in out_rows if _boolish(row.get("candidate_hook_active")))
    deltas = [safe_float(row.get("delta_J_scale")) for row in out_rows]
    finite_deltas = [float(delta) for delta in deltas if delta is not None]
    regressions = [-delta for delta in finite_deltas if delta < 0.0]
    mode_specific_gate = len(support_chunks) >= 3
    all_target_gate = len(support_chunks) >= 4
    median_delta = finite_median(row.get("delta_J_scale") for row in out_rows)
    max_regression = max(regressions) if regressions else 0.0
    non_reversal_gate = bool(
        str(seq) == "09"
        and median_delta is not None
        and float(median_delta) >= 0.0
        and len(positive_chunks) >= 1
        and len(hook_chunks) >= 1
        and float(max_regression) <= 0.2
    )
    summary = {
        "seq": seq,
        "intervention_type": intervention_type,
        "memory_path": memory_path,
        "source_csv": str(rows_path),
        "row_count": len(out_rows),
        "chunks": ",".join(str(x) for x in chunks),
        "hook_active_chunks": ",".join(str(x) for x in hook_chunks),
        "positive_chunks": ",".join(str(x) for x in positive_chunks),
        "positive_chunk_count": len(positive_chunks),
        "support_chunks": ",".join(str(x) for x in support_chunks),
        "support_chunk_count": len(support_chunks),
        "median_delta_J_scale": median_delta,
        "mean_delta_J_scale": finite_mean(row.get("delta_J_scale") for row in out_rows),
        "max_regression_m": max_regression,
        "mode_specific_gate_pass": mode_specific_gate,
        "all_target_gate_pass": all_target_gate,
        "causal_support": bool(mode_specific_gate or all_target_gate),
        "non_reversal_gate_pass": non_reversal_gate,
        "blocked_reason": ""
        if (mode_specific_gate or all_target_gate)
        else "online hook may exist, but candidate does not produce enough positive chunks beating controls",
        "non_reversal_blocked_reason": ""
        if (str(seq) != "09" or non_reversal_gate)
        else "09 median delta_J_scale is negative or lacks same-sign positive/hook evidence",
    }
    return out_rows, summary


def _write_report(out_dir: Path, summary: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# v74-TF Phase 5 Online Memory Intervention From Smokes",
        "",
        "This is an aggregation of already-run online smoke traces. It does not rerun inference and does not claim component-level evidence absent from the source CSV files.",
        "",
        f"- rows: `{summary['rows']}`",
        f"- phase5_01_gate_pass: `{summary['phase5_01_gate_pass']}`",
        f"- phase5_09_gate_pass: `{summary['phase5_09_gate_pass']}`",
        f"- phase5_gate_pass: `{summary['phase5_gate_pass']}`",
        f"- blocked_reason: `{summary['blocked_reason']}`",
        "",
        "| intervention | hook chunks | positive chunks | support chunks | median delta J/ATE | gate |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {intervention_type} | {hook_active_chunks} | {positive_chunks} | {support_chunks} | {median_delta_J_scale} | {causal_support} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Gate rule used here:",
            "",
            "- candidate must be an online candidate row with `returncode=0` and active hook evidence.",
            "- lower ATE is better, so `delta_J_scale = native_metric - candidate_metric`.",
            "- row support requires positive delta and candidate beating all finite controls for the same chunk.",
            "- intervention support requires at least 4 support chunks, or at least 3 support chunks in this fixed N-A mode-specific smoke.",
            "",
        ]
    )
    write_text(out_dir / "online_intervention_report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--context-anchor-results", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--context-anchor-tail-results", type=Path, default=DEFAULT_CONTEXT_TAIL)
    parser.add_argument("--context-anchor-tail-all-results", type=Path, default=DEFAULT_CONTEXT_TAIL_ALL)
    parser.add_argument("--context-anchor-midtail-all-results", type=Path, default=DEFAULT_CONTEXT_MIDTAIL_ALL)
    parser.add_argument("--swa-results", type=Path, default=DEFAULT_SWA)
    parser.add_argument("--swa-leave-one-out-results", type=Path, default=DEFAULT_SWA_LEAVE_ONE_OUT)
    parser.add_argument("--swa-gate-results", type=Path, default=DEFAULT_SWA_GATE)
    parser.add_argument("--read-pair-results", type=Path, default=DEFAULT_READ_PAIR)
    parser.add_argument("--read-query-results", type=Path, default=DEFAULT_READ_QUERY)
    parser.add_argument("--ttt-no-persistent-results", type=Path, default=DEFAULT_TTT_NO_PERSISTENT)
    parser.add_argument("--refresh-hold-flip-results", type=Path, default=DEFAULT_REFRESH_HOLD_FLIP)
    parser.add_argument("--refresh-hold-flip-09-results", type=Path, default=DEFAULT_REFRESH_HOLD_FLIP_09)
    args = parser.parse_args()

    specs = [
        {
            "rows_path": args.context_anchor_results,
            "intervention_type": "stable_anchor_floor_short",
            "memory_path": "PI3_context_source_skip_semantic_anchor",
            "native_case": "native_no_boost",
        },
        {
            "rows_path": args.context_anchor_tail_results,
            "intervention_type": "stable_anchor_floor_tail_only",
            "memory_path": "PI3_context_source_skip_semantic_anchor_tail",
            "native_case": "native_no_boost",
        },
        {
            "rows_path": args.context_anchor_tail_all_results,
            "intervention_type": "stable_anchor_floor_tail_all_layers",
            "memory_path": "PI3_context_source_skip_semantic_anchor_tail_all_layers",
            "native_case": "native_no_boost",
        },
        {
            "rows_path": args.context_anchor_midtail_all_results,
            "intervention_type": "stable_anchor_floor_midtail_all_layers",
            "memory_path": "PI3_context_source_skip_semantic_anchor_midtail_all_layers",
            "native_case": "native_no_boost",
        },
        {
            "rows_path": args.swa_results,
            "intervention_type": "component_boost_SWA",
            "memory_path": "SWA_overlap_source_replace",
            "native_case": "native_no_swa",
        },
        {
            "rows_path": args.swa_leave_one_out_results,
            "intervention_type": "component_leave_one_out_SWA",
            "memory_path": "Pi3_native_SWA_update_gate_leave_one_out",
            "native_case": "native_no_swa",
        },
        {
            "rows_path": args.swa_gate_results,
            "intervention_type": "component_veto_SWA",
            "memory_path": "SWA_overlap_source_gate",
            "native_case": "native_no_swa",
        },
        {
            "rows_path": args.read_pair_results,
            "intervention_type": "route_swap_geometry_context_transient_pair",
            "memory_path": "frame_attention_read_pair",
            "native_case": "native_no_read",
        },
        {
            "rows_path": args.read_query_results,
            "intervention_type": "route_swap_geometry_context_transient_query",
            "memory_path": "frame_attention_read_query",
            "native_case": "native_no_read",
        },
        {
            "rows_path": args.ttt_no_persistent_results,
            "intervention_type": "harmful_no_persistent_TTT_if_available",
            "memory_path": "RADIO_TTT_write_dynamic_lowstable_no_persistent",
            "native_case": "native_no_ttt_radio",
        },
        {
            "rows_path": args.refresh_hold_flip_results,
            "intervention_type": "refresh_hold_flip",
            "memory_path": "semantic_merge_RADIO_qscale_hold_refresh",
            "native_case": "native_no_swa",
            "candidate_case": "radio_qscale",
        },
        {
            "rows_path": args.refresh_hold_flip_09_results,
            "intervention_type": "refresh_hold_flip",
            "memory_path": "semantic_merge_RADIO_qscale_hold_refresh",
            "native_case": "native_no_swa",
            "candidate_case": "radio_qscale",
            "seq": "09",
        },
    ]
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing_inputs: list[str] = []
    for spec in specs:
        rows_path = Path(spec["rows_path"])
        if not rows_path.exists():
            missing_inputs.append(str(rows_path))
            continue
        rows, summary = _analyze_source(**spec)
        all_rows.extend(rows)
        summaries.append(summary)

    phase5_01_gate_pass = any(row.get("causal_support") for row in summaries if str(row.get("seq", "01")) == "01")
    phase5_09_gate_pass = any(
        row.get("non_reversal_gate_pass")
        for row in summaries
        if str(row.get("seq", "")) == "09" and str(row.get("intervention_type", "")) == "refresh_hold_flip"
    )
    phase5_gate_pass = bool(phase5_01_gate_pass and phase5_09_gate_pass)
    if missing_inputs:
        blocked_reason = "missing input CSVs: " + ",".join(missing_inputs)
    elif phase5_gate_pass:
        blocked_reason = ""
    elif phase5_01_gate_pass:
        seq09_reasons = [
            str(row.get("non_reversal_blocked_reason") or row.get("blocked_reason") or "")
            for row in summaries
            if str(row.get("seq", "")) == "09"
        ]
        blocked_reason = (
            "KITTI01 online causal support exists, but Phase5 gate remains false because "
            "KITTI09 non-reversal is not passed."
            + (f" 09 reason: {'; '.join(x for x in seq09_reasons if x)}" if seq09_reasons else "")
        )
    else:
        blocked_reason = (
            "Online memory-path traces are now aggregated, but no intervention has enough "
            "positive chunks beating controls; 09 non-reversal not evaluated because 01 gate is false."
        )
    summary = {
        "schema": "acl2_v74tf_phase5_online_memory_intervention_from_smokes_v1",
        "created_at": utc_now(),
        "rows": len(all_rows),
        "missing_inputs": missing_inputs,
        "intervention_summary": summaries,
        "phase5_01_gate_pass": bool(phase5_01_gate_pass),
        "phase5_09_gate_pass": bool(phase5_09_gate_pass),
        "phase5_gate_pass": bool(phase5_gate_pass),
        "blocked_reason": blocked_reason,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "online_intervention_rows.csv", all_rows)
    write_csv(args.out_dir / "online_intervention_summary.csv", summaries)
    write_json(args.out_dir / "online_intervention_summary.json", summary)
    _write_report(args.out_dir, summary, summaries)
    print(
        {
            "out_dir": str(args.out_dir),
            "rows": summary["rows"],
            "phase5_01_gate_pass": summary["phase5_01_gate_pass"],
            "phase5_gate_pass": summary["phase5_gate_pass"],
            "blocked_reason": summary["blocked_reason"],
        }
    )


if __name__ == "__main__":
    main()
