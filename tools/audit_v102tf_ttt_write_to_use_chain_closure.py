#!/usr/bin/env python3
"""Audit v102 C3 TTT write-to-use chain closure without running a TTT action."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
DEFAULT_TARGET_CSV = ROOT / "stage4_memory_action_surface_oracle/v102_state_machine_scaffold_trace_targets.csv"
DEFAULT_STAGE7E_ROWS = Path(
    "results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control/"
    "stage7e_ttt_stable_anchor_id_hook/anchor_id_case_rows.csv"
)
DEFAULT_STAGE7E_SUMMARY = Path(
    "results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control/"
    "stage7e_ttt_stable_anchor_id_hook/summary.json"
)
DEFAULT_STAGE7H_SUMMARY = Path(
    "results/acl2_v98tf_semantic_guided_swa_cache_topk_gauge_safe_memory_control/"
    "stage7h_prev_ttt_anchor_query_soft_action_pilot/summary.json"
)
DEFAULT_F5_SUMMARY = Path(
    "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/"
    "trackF5_ttt_write_to_use_state_chain/F5_summary.json"
)
DEFAULT_F5_MATERIALIZATION = Path(
    "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/"
    "trackF5_ttt_write_to_use_state_chain/write_to_use_materialization_audit.csv"
)
DEFAULT_OUTPUT_PREFIX = ROOT / "stage4_memory_action_surface_oracle/ttt_write_to_use_chain_closure"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--stage7e-rows", type=Path, default=DEFAULT_STAGE7E_ROWS)
    parser.add_argument("--stage7e-summary", type=Path, default=DEFAULT_STAGE7E_SUMMARY)
    parser.add_argument("--stage7h-summary", type=Path, default=DEFAULT_STAGE7H_SUMMARY)
    parser.add_argument("--f5-summary", type=Path, default=DEFAULT_F5_SUMMARY)
    parser.add_argument("--f5-materialization", type=Path, default=DEFAULT_F5_MATERIALIZATION)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_rows = read_csv(args.target_csv)
    stage7e_rows = {row.get("case_id", ""): row for row in read_csv(args.stage7e_rows)}
    stage7e_summary = read_json(args.stage7e_summary)
    stage7h_summary = read_json(args.stage7h_summary)
    f5_summary = read_json(args.f5_summary)
    f5_materialization_rows = read_csv(args.f5_materialization)
    f5_materialization = f5_materialization_rows[0] if f5_materialization_rows else {}

    rows: list[dict[str, Any]] = []
    for row in target_rows:
        case_id = row.get("case_id", "")
        stage7e = stage7e_rows.get(case_id, {})
        in_stage7e = bool(stage7e)
        per_anchor_write_cache = as_float(f5_materialization.get("r_write_cache_nonempty"), 0.0) > 0.0
        per_anchor_cache_current = as_float(f5_materialization.get("r_cache_current_nonempty"), 0.0) > 0.0
        per_anchor_ref_current = as_float(f5_materialization.get("r_ref_current_nonempty"), 0.0) > 0.0
        later_swa_use = in_stage7e and as_bool(stage7e.get("write_to_swa_topk_chain_observed"))
        rows.append(
            {
                "case_id": case_id,
                "seq": row.get("seq", ""),
                "role": row.get("ambiguous_or_control_role", ""),
                "strict_clean_handoff_positive": row.get("strict_clean_handoff_positive", ""),
                "L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy", ""),
                "in_stage7e_anchor_id_rows": in_stage7e,
                "stage7e_write_to_swa_topk_chain_observed": later_swa_use,
                "stage7e_anchor_id_topk_hit_frac_mean": stage7e.get("anchor_id_topk_hit_frac_mean", ""),
                "stage7e_anchor_id_topk_query_hit_frac_mean": stage7e.get("anchor_id_topk_query_hit_frac_mean", ""),
                "stage7e_anchor_id_route_mass_mean": stage7e.get("anchor_id_route_mass_mean", ""),
                "stage7e_anchor_id_retention_mean": stage7e.get("anchor_id_retention_mean", ""),
                "per_anchor_write_cache_materialized": per_anchor_write_cache,
                "per_anchor_cache_current_materialized": per_anchor_cache_current,
                "per_anchor_ref_current_materialized": per_anchor_ref_current,
                "write_to_later_read_hit_materialized": False,
                "write_to_later_swa_hit_materialized": later_swa_use,
                "later_L3_handoff_available": bool(row.get("L3_handoff_transfer_penalty_proxy", "")),
                "c3_full_chain_materialized": (
                    per_anchor_write_cache
                    and per_anchor_cache_current
                    and later_swa_use
                    and bool(row.get("L3_handoff_transfer_penalty_proxy", ""))
                ),
                "claim_scope": (
                    "stage7e_id_to_swa_topk_only"
                    if in_stage7e else "missing_stage7e_for_v102_target_case"
                ),
            }
        )

    target_count = len(rows)
    target_with_stage7e = sum(1 for row in rows if row["in_stage7e_anchor_id_rows"])
    target_with_swa_use = sum(1 for row in rows if row["stage7e_write_to_swa_topk_chain_observed"])
    full_chain_count = sum(1 for row in rows if row["c3_full_chain_materialized"])
    strict_rows = [row for row in rows if as_bool(row["strict_clean_handoff_positive"])]
    strict_with_stage7e = sum(1 for row in strict_rows if row["in_stage7e_anchor_id_rows"])
    strict_full_chain = sum(1 for row in strict_rows if row["c3_full_chain_materialized"])
    target_chain_coverage = target_with_swa_use / max(1, target_count)
    full_chain_coverage = full_chain_count / max(1, target_count)
    chain_coverage_pass = target_chain_coverage >= 0.80
    full_chain_pass = full_chain_coverage >= 0.80
    per_anchor_chain_materialized = bool(
        as_float(f5_materialization.get("r_write_cache_nonempty"), 0.0) > 0.0
        and as_float(f5_materialization.get("r_cache_current_nonempty"), 0.0) > 0.0
    )
    stage7h_gate_pass = bool(stage7h_summary.get("gate_pass"))

    summary = {
        "schema": "acl2_v102_ttt_write_to_use_chain_closure_v1",
        "target_count": target_count,
        "target_with_stage7e_anchor_id_count": target_with_stage7e,
        "target_with_stage7e_swa_topk_use_count": target_with_swa_use,
        "target_stage7e_swa_topk_use_coverage": target_chain_coverage,
        "target_full_write_cache_current_swa_l3_chain_count": full_chain_count,
        "target_full_write_cache_current_swa_l3_chain_coverage": full_chain_coverage,
        "strict_clean_handoff_positive_count": len(strict_rows),
        "strict_clean_handoff_positive_with_stage7e_count": strict_with_stage7e,
        "strict_clean_handoff_positive_full_chain_count": strict_full_chain,
        "stage7e_write_to_use_chain_available": bool(stage7e_summary.get("write_to_use_chain_available")),
        "stage7e_gate_pass": bool(stage7e_summary.get("gate_pass")),
        "stage7e_case_count": stage7e_summary.get("case_count"),
        "stage7e_sequence_coverage": stage7e_summary.get("sequence_coverage"),
        "stage7h_query_soft_gate_pass": stage7h_gate_pass,
        "stage7h_runtime_action_pilot_run": bool(stage7h_summary.get("runtime_action_pilot_run")),
        "f5_per_anchor_write_chain_materialized": per_anchor_chain_materialized,
        "f5_r_write_cache_nonempty": as_float(f5_materialization.get("r_write_cache_nonempty"), 0.0),
        "f5_r_cache_current_nonempty": as_float(f5_materialization.get("r_cache_current_nonempty"), 0.0),
        "f5_r_ref_current_nonempty": as_float(f5_materialization.get("r_ref_current_nonempty"), 0.0),
        "f5_anchor_state_row_count": as_float(f5_materialization.get("anchor_state_row_count"), 0.0),
        "c3_chain_coverage_pass": chain_coverage_pass,
        "c3_full_chain_materialization_pass": full_chain_pass and per_anchor_chain_materialized,
        "c3_ttt_action_surface_pass": False,
        "runtime_action_allowed": False,
        "stage5_allowed": False,
        "stage6_runtime_pilot_allowed": False,
        "stage7_full_validation_allowed": False,
        "conclusion": (
            "C3 remains diagnostic-only: v98 Stage7e proves an id-to-SWA-top-k chain exists on some cases, "
            "but only a subset of v102 target cases has that evidence, the strict-positive target lacks the "
            "full per-anchor write/cache/current chain, v101 F5 reports zero materialized per-anchor residual "
            "links, and the old query-soft pilot failed. Do not run TTT write/expire/refresh action."
        ),
    }

    out_prefix = args.output_prefix
    write_csv(out_prefix.with_name(out_prefix.name + "_rows.csv"), rows)
    out_prefix.with_name(out_prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# v102 C3 TTT Write-to-Use Chain Closure",
        "",
        "This is a diagnostic-only audit. It does not run or authorize a TTT action.",
        "",
        f"- target_count: {target_count}",
        f"- target_with_stage7e_swa_topk_use_count: {target_with_swa_use}",
        f"- target_stage7e_swa_topk_use_coverage: {target_chain_coverage}",
        f"- target_full_write_cache_current_swa_l3_chain_count: {full_chain_count}",
        f"- f5_r_write_cache_nonempty: {summary['f5_r_write_cache_nonempty']}",
        f"- f5_r_cache_current_nonempty: {summary['f5_r_cache_current_nonempty']}",
        f"- stage7h_query_soft_gate_pass: {stage7h_gate_pass}",
        "",
        "Conclusion:",
        "",
        summary["conclusion"],
        "",
    ]
    out_prefix.with_name(out_prefix.name + "_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
