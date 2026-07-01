#!/usr/bin/env python3
"""Audit v91 semantic regime carrier availability or write a blocked Phase7 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from v86_soft_latent_utils import read_json, write_csv, write_json
from v91_semantic_regime_utils import ROOT, nseries, stable_shuffle


DEFAULT_OUT = ROOT / "phase7_carrier_attribution_or_blocked"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--route-dump-root",
        type=Path,
        default=DEFAULT_OUT / "route_dump_smoke",
        help="Optional true route-dump smoke output root to scan for Phase7 carrier evidence.",
    )
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _balanced(bad_recall: float, good_fpr: float) -> float:
    return float(0.5 * (bad_recall + 1.0 - good_fpr))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _nested_dict(obj: dict[str, Any], *keys: str) -> dict[str, Any]:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(key, {})
    return cur if isinstance(cur, dict) else {}


def _scan_route_dump(root: Path) -> dict[str, Any]:
    manifests = sorted(root.glob("**/phase9_swa_cache_value_run_manifest.json")) if root.exists() else []
    hmc_paths = sorted(root.glob("**/hmc_state_hash.jsonl")) if root.exists() else []
    hook_paths = sorted(root.glob("**/hook_effect_summary.jsonl")) if root.exists() else []
    metric_paths = sorted(root.glob("**/phase9_swa_cache_value_metrics.csv")) if root.exists() else []
    feature_files = sorted(root.glob("**/swa_overlap_feature_maps/*")) if root.exists() else []
    manifest_jobs = 0
    successful_jobs = 0
    failed_jobs = 0
    route_dump_seqs: set[str] = set()
    external_mask_rows_matching_max = 0
    external_mask_source_tokens_selected_max = 0
    external_mask_reason_counts: dict[str, int] = {}
    selected_tokens_proxy_max = 0
    overlap_bias_applied_sum_max = 0
    attention_mass_available_frac_max = 0.0
    actual_selected_lift: float | None = None
    random_selected_lift: float | None = None
    actual_minus_random_selected_lift: float | None = None
    actual_headmax_lift: float | None = None
    random_headmax_lift: float | None = None
    actual_minus_random_headmax_lift: float | None = None
    selected_lift_deltas: list[float] = []
    headmax_lift_deltas: list[float] = []
    for manifest in manifests:
        try:
            payload = read_json(manifest)
        except Exception:  # noqa: BLE001
            continue
        jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
        if not isinstance(jobs, list):
            continue
        args_obj = payload.get("args", {}) if isinstance(payload, dict) else {}
        if isinstance(args_obj, dict):
            seq_arg = str(args_obj.get("swa_overlap_external_mask_seq", "") or "").strip()
            if seq_arg:
                route_dump_seqs.add(seq_arg.zfill(2))
        manifest_jobs += len(jobs)
        for job in jobs:
            if not isinstance(job, dict):
                continue
            rc = job.get("returncode")
            if rc is None:
                continue
            if int(rc) == 0 and not bool(job.get("skipped", False)):
                successful_jobs += 1
            elif int(rc) != 0:
                failed_jobs += 1
    hmc_rows = 0
    for hmc in hmc_paths:
        for row in _read_jsonl(hmc):
            hmc_rows += 1
            swa_read = _nested_dict(row, "control_trace", "hook_effect_summary", "swa_read")
            if not swa_read:
                continue
            for key, value in swa_read.items():
                if "external_mask_rows_matching" in str(key):
                    try:
                        external_mask_rows_matching_max = max(external_mask_rows_matching_max, int(float(value or 0)))
                    except (TypeError, ValueError):
                        pass
                if "external_mask_source_tokens_selected" in str(key):
                    try:
                        external_mask_source_tokens_selected_max = max(external_mask_source_tokens_selected_max, int(float(value or 0)))
                    except (TypeError, ValueError):
                        pass
                if "external_mask_reason" in str(key):
                    reason = str(value)
                    external_mask_reason_counts[reason] = external_mask_reason_counts.get(reason, 0) + 1
    hook_rows = 0
    for hook in hook_paths:
        for row in _read_jsonl(hook):
            hook_rows += 1
            swa_read = _nested_dict(row, "hook_effect_summary", "swa_read")
            if not swa_read:
                continue
            try:
                selected_tokens_proxy_max = max(
                    selected_tokens_proxy_max,
                    int(float(swa_read.get("mean_attention_mass_removed_tokens") or 0.0)),
                )
            except (TypeError, ValueError):
                pass
            try:
                overlap_bias_applied_sum_max = max(
                    overlap_bias_applied_sum_max,
                    int(float(swa_read.get("num_swa_overlap_bias_applied") or 0.0)),
                )
            except (TypeError, ValueError):
                pass
    metric_rows = 0
    for metric_path in metric_paths:
        try:
            metric_df = pd.read_csv(metric_path)
        except Exception:  # noqa: BLE001
            continue
        metric_rows += int(len(metric_df))
        if "phase9_swa_overlap_bias_applied_sum" in metric_df:
            overlap_bias_applied_sum_max = max(
                overlap_bias_applied_sum_max,
                int(pd.to_numeric(metric_df["phase9_swa_overlap_bias_applied_sum"], errors="coerce").fillna(0).max()),
            )
        if "phase9_swa_attention_mass_available_frac" in metric_df:
            attention_mass_available_frac_max = max(
                attention_mass_available_frac_max,
                float(pd.to_numeric(metric_df["phase9_swa_attention_mass_available_frac"], errors="coerce").fillna(0.0).max()),
            )
        if "run" in metric_df:
            by_run = {str(row["run"]): row for _, row in metric_df.iterrows()}
            actual = by_run.get("P9_48_ATTENTION_BIAS_V84_EXTERNAL_ANCHOR_MASS_AUDIT_LAST")
            random = by_run.get("P9_49_ATTENTION_BIAS_V84_EXTERNAL_ANCHOR_RANDOM_SAME_MASS_MASS_AUDIT_LAST")
            if actual is not None:
                actual_selected_lift = float(actual.get("phase9_swa_attention_mass_selected_lift", 0.0) or 0.0)
                actual_headmax_lift = float(actual.get("phase9_swa_attention_mass_selected_head_max_lift", 0.0) or 0.0)
            if random is not None:
                random_selected_lift = float(random.get("phase9_swa_attention_mass_selected_lift", 0.0) or 0.0)
                random_headmax_lift = float(random.get("phase9_swa_attention_mass_selected_head_max_lift", 0.0) or 0.0)
            if actual_selected_lift is not None and random_selected_lift is not None:
                actual_minus_random_selected_lift = float(actual_selected_lift - random_selected_lift)
                selected_lift_deltas.append(actual_minus_random_selected_lift)
            if actual_headmax_lift is not None and random_headmax_lift is not None:
                actual_minus_random_headmax_lift = float(actual_headmax_lift - random_headmax_lift)
                headmax_lift_deltas.append(actual_minus_random_headmax_lift)
    selected_lift_mean = float(sum(selected_lift_deltas) / len(selected_lift_deltas)) if selected_lift_deltas else None
    headmax_lift_mean = float(sum(headmax_lift_deltas) / len(headmax_lift_deltas)) if headmax_lift_deltas else None
    return {
        "route_dump_root": str(root),
        "route_dump_manifest_count": int(len(manifests)),
        "route_dump_seq_coverage": int(len(route_dump_seqs)),
        "route_dump_seqs": sorted(route_dump_seqs),
        "route_dump_manifest_jobs": int(manifest_jobs),
        "route_dump_successful_jobs": int(successful_jobs),
        "route_dump_failed_jobs": int(failed_jobs),
        "route_dump_hmc_files": int(len(hmc_paths)),
        "route_dump_hmc_rows": int(hmc_rows),
        "route_dump_hook_files": int(len(hook_paths)),
        "route_dump_hook_rows": int(hook_rows),
        "route_dump_metrics_files": int(len(metric_paths)),
        "route_dump_metrics_rows": int(metric_rows),
        "route_dump_swa_feature_files": int(len(feature_files)),
        "route_dump_external_mask_rows_matching_max": int(external_mask_rows_matching_max),
        "route_dump_external_mask_source_tokens_selected_max": int(external_mask_source_tokens_selected_max),
        "route_dump_selected_tokens_proxy_max": int(selected_tokens_proxy_max),
        "route_dump_overlap_bias_applied_sum_max": int(overlap_bias_applied_sum_max),
        "route_dump_attention_mass_available_frac_max": float(attention_mass_available_frac_max),
        "route_dump_actual_selected_lift": actual_selected_lift,
        "route_dump_random_selected_lift": random_selected_lift,
        "route_dump_actual_minus_random_selected_lift": actual_minus_random_selected_lift,
        "route_dump_actual_minus_random_selected_lift_mean": selected_lift_mean,
        "route_dump_actual_minus_random_selected_lift_min": float(min(selected_lift_deltas)) if selected_lift_deltas else None,
        "route_dump_actual_minus_random_selected_lift_max": float(max(selected_lift_deltas)) if selected_lift_deltas else None,
        "route_dump_actual_beats_random_selected_lift_count": int(sum(1 for value in selected_lift_deltas if value > 0.0)),
        "route_dump_actual_random_pair_count": int(len(selected_lift_deltas)),
        "route_dump_actual_headmax_lift": actual_headmax_lift,
        "route_dump_random_headmax_lift": random_headmax_lift,
        "route_dump_actual_minus_random_headmax_lift": actual_minus_random_headmax_lift,
        "route_dump_actual_minus_random_headmax_lift_mean": headmax_lift_mean,
        "route_dump_actual_minus_random_headmax_lift_min": float(min(headmax_lift_deltas)) if headmax_lift_deltas else None,
        "route_dump_actual_minus_random_headmax_lift_max": float(max(headmax_lift_deltas)) if headmax_lift_deltas else None,
        "route_dump_actual_beats_random_headmax_lift_count": int(sum(1 for value in headmax_lift_deltas if value > 0.0)),
        "route_dump_external_mask_reason_counts": external_mask_reason_counts,
        "route_dump_smoke_available": bool(successful_jobs > 0 and (hmc_rows > 0 or hook_rows > 0)),
        "route_dump_external_mask_hit": bool(overlap_bias_applied_sum_max > 0 and selected_tokens_proxy_max > 0),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    phase3 = _json(args.root / "phase3_regime_conditioned_semantic_relevance/regime_conditioned_relevance_summary.json")
    phase4 = _json(args.root / "phase4_tracklet_mode_disambiguation/tracklet_mode_disambiguation_summary.json")
    phase5 = _json(args.root / "phase5_memory_update_policy/policy_state_audit.json")
    phase6 = _json(args.root / "phase6_adaptive_memory_baseline/delayed_commit_audit.json")
    preconditions = {
        "phase3_regime_semantic_gate_pass": bool(phase3.get("phase3_regime_semantic_gate_pass")),
        "phase4_tracklet_mode_gate_pass": bool(phase4.get("phase4_tracklet_mode_gate_pass")),
        "phase5_memory_update_policy_gate_pass": bool(phase5.get("phase5_memory_update_policy_gate_pass")),
        "phase6_delayed_commit_gate_pass": bool(phase6.get("phase6_delayed_commit_gate_pass")),
    }
    entered = any(preconditions.values())
    metrics: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    true_trace_available = False
    repair_attempted = ""
    materialization = _json(args.out_dir / "v91_external_mask_materialization/materialization_summary.json")
    route_scan = _scan_route_dump(args.route_dump_root)
    if entered:
        policy_path = args.root / "phase5_memory_update_policy/policy_state_rows.csv"
        if policy_path.exists():
            df = pd.read_csv(policy_path)
            df["seq"] = df["seq"].astype(str).str.zfill(2)
            risk_score = nseries(df, "P_reject") + nseries(df, "P_reset_risk") + nseries(df, "P_delay")
            update_score = nseries(df, "P_update")
            boundary_mass = nseries(df, "boundary_mass")
            direct_rows = []
            for idx, row in df.iterrows():
                direct_rows.append(
                    {
                        "seq": str(row.get("seq", "")).zfill(2),
                        "prev_chunk": row.get("prev_chunk", ""),
                        "curr_chunk": row.get("curr_chunk", ""),
                        "pair_id": row.get("pair_id", ""),
                        "policy_state": row.get("policy_state", ""),
                        "boundary_update_eligible_mass": float(update_score.loc[idx] * (1.0 - boundary_mass.loc[idx])),
                        "invalid_rejected_mass": float(risk_score.loc[idx] * boundary_mass.loc[idx]),
                        "context_only_mass": float(nseries(df, "S_context").loc[idx] + nseries(df, "S_lowobs").loc[idx]),
                        "trace_source": "policy_proxy_not_runtime_trace",
                        "runtime_action_allowed": False,
                    }
                )
            repair_attempted = "direct_boundary_update_trace_proxy_from_policy_rows"
            write_csv(args.out_dir / "direct_boundary_update_trace_proxy.csv", direct_rows)
            actual_bal = _balanced(float(phase5.get("bad_recall", 0.0)), float(phase5.get("good_FPR", 1.0)))
            sem_bal = _balanced(float(phase5.get("bad_recall", 0.0)), float(phase5.get("good_FPR", 1.0)))
            if "semantic_shuffle_state" in df:
                shuffled = stable_shuffle(df["policy_state"], "v91_phase7_semantic_carrier_control")
                sem_bal = float((shuffled.astype(str).isin(["UPDATE"]).mean() + (1.0 - shuffled.astype(str).isin(["REJECT", "RESET_RISK", "DELAY"]).mean())) * 0.5)
            metrics.append(
                {
                    "carrier_family": "merge_gauge_boundary_update_proxy",
                    "true_route_or_trace_available": False,
                    "proxy_rows": len(direct_rows),
                    "balanced_safety_score_from_phase5": actual_bal,
                    "semantic_shuffle_proxy_score": sem_bal,
                    "semantic_shuffle_margin_proxy": actual_bal - sem_bal,
                    "carrier_gate_pass": False,
                    "blocker": "proxy_trace_not_promoted_to_true_carrier",
                }
            )
        else:
            metrics.append(
                {
                    "carrier_family": "merge_gauge_boundary_update_proxy",
                    "true_route_or_trace_available": False,
                    "proxy_rows": 0,
                    "carrier_gate_pass": False,
                    "blocker": "policy_state_rows_missing",
                }
            )
        metrics.append(
            {
                "carrier_family": "SWA_external_mask_route_smoke",
                "true_route_or_trace_available": bool(route_scan["route_dump_smoke_available"]),
                "proxy_rows": int(materialization.get("position_rows", 0) or 0),
                "route_dump_successful_jobs": int(route_scan["route_dump_successful_jobs"]),
                "route_dump_hmc_rows": int(route_scan["route_dump_hmc_rows"]),
                "route_dump_selected_tokens_proxy_max": int(route_scan["route_dump_selected_tokens_proxy_max"]),
                "route_dump_actual_minus_random_selected_lift": route_scan["route_dump_actual_minus_random_selected_lift"],
                "route_dump_external_mask_source_tokens_selected_max": int(
                    route_scan["route_dump_external_mask_source_tokens_selected_max"]
                ),
                "carrier_gate_pass": False,
                "blocker": (
                    "route_dump_smoke_controls_incomplete_not_promoted"
                    if route_scan["route_dump_smoke_available"]
                    else "true_route_dump_unavailable"
                ),
            }
        )
    for family in ["READ", "SWA", "TTT"]:
        metrics.append(
            {
                "carrier_family": family,
                "true_route_or_trace_available": False,
                "proxy_rows": 0,
                "carrier_gate_pass": False,
                "blocker": "true_route_dump_or_runtime_trace_unavailable",
            }
        )
    true_trace_available = bool(route_scan["route_dump_smoke_available"])
    summary = {
        "phase": "Phase7_semantic_regime_carrier_or_blocked",
        "entered": entered,
        "phase7_carrier_gate_pass": False,
        "preconditions": preconditions,
        "true_route_or_trace_available": true_trace_available,
        "external_mask_materialization_feasible": bool(materialization.get("materialization_feasible", False)),
        "external_mask_position_rows": int(materialization.get("position_rows", 0) or 0),
        **route_scan,
        "direct_boundary_update_trace_proxy_rows": len(direct_rows),
        "repair_attempted": repair_attempted,
        "runtime_action_allowed": False,
        "runtime_action_executed": False,
        "ttt_allowed": False,
    }
    if not entered:
        summary["blocker"] = "carrier_not_entered_preconditions_failed"
    elif true_trace_available:
        summary["blocker"] = "true_route_smoke_available_controls_incomplete_not_promoted"
    else:
        summary["blocker"] = "true_memory_carrier_trace_unavailable_proxy_not_promoted"
    write_csv(args.out_dir / "carrier_metrics.csv", metrics)
    write_json(args.out_dir / "phase7_carrier_summary.json", summary)
    print(f"phase7_entered={summary['entered']}")
    print(f"phase7_carrier_gate_pass={summary['phase7_carrier_gate_pass']}")
    print(f"true_route_or_trace_available={summary['true_route_or_trace_available']}")
    print(f"direct_boundary_update_trace_proxy_rows={summary['direct_boundary_update_trace_proxy_rows']}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
