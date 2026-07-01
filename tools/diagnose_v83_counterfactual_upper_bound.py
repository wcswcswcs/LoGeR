#!/usr/bin/env python3
"""Diagnose ACL2 v83 Phase4 counterfactual upper bounds.

This is diagnostic-only. It reads existing merge/gauge fallback and oracle
artifacts and checks whether any counterfactual/oracle family has enough bad
improvement, good-case protection, and control beating to justify runtime work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_OUT_DIR = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase4_counterfactual_upper_bound"
)
DEFAULT_PHASE3_SUMMARY = Path(
    "results/acl2_v83tf_clue_sufficiency_vs_action_misuse/"
    "phase3_carrier_alignment/carrier_alignment_summary.json"
)
DEFAULT_PHASE8E_SUMMARY = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase8e_projection_tol001_steps64_continuation/merge_gauge_fallback_summary.json"
)
DEFAULT_ORACLE_SUMMARY = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase12_merge_controller_oracle_audit/merge_controller_oracle_summary.json"
)


METRIC_ALIASES = {
    "head10_to_tail10_pose_sim3_rmse_m": "head_tail",
    "overlap3_to_future_pose_sim3_rmse_m": "overlap_future",
    "local_sim3_ate_rmse_m": "local_ate",
    "scale_cv_head_mid_tail_pose_sim3": "scale_cv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--phase3-summary", type=Path, default=DEFAULT_PHASE3_SUMMARY)
    parser.add_argument("--phase8e-summary", type=Path, default=DEFAULT_PHASE8E_SUMMARY)
    parser.add_argument("--oracle-summary", type=Path, default=DEFAULT_ORACLE_SUMMARY)
    parser.add_argument("--bad-improvement-min", type=float, default=0.10)
    parser.add_argument("--good-worsen-max", type=float, default=0.02)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in keys})


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def gate_for_metric(
    *,
    bad_improvement: float | None,
    good_worsen: float | None,
    bad_control_beat_count: int,
    bad_rows: int,
    args: argparse.Namespace,
) -> dict[str, bool]:
    gate = {
        "bad_improvement_ge_10pct": bad_improvement is not None and bad_improvement >= args.bad_improvement_min,
        "good_worsen_le_2pct": good_worsen is not None and good_worsen <= args.good_worsen_max,
        "beats_all_controls_on_bad": bad_rows > 0 and bad_control_beat_count >= bad_rows,
    }
    gate["metric_counterfactual_gate_pass"] = all(gate.values())
    return gate


def phase8e_rows(payload: Mapping[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in payload.get("candidates", []) if isinstance(payload.get("candidates"), list) else []:
        if not isinstance(candidate, dict):
            continue
        bad_rows = len(candidate.get("bad_chunks", []) or [])
        good_rows = len(candidate.get("good_chunks", []) or [])
        for metric, stats in sorted((candidate.get("metric_summary") or {}).items()):
            if not isinstance(stats, dict):
                continue
            bad_improvement = safe_float(stats.get("bad_median_improvement_vs_baseline_ratio"))
            good_worsen = safe_float(stats.get("good_max_worsen_vs_baseline_ratio"))
            beat_count = int(safe_float(stats.get("bad_control_beat_count")) or 0)
            gate = gate_for_metric(
                bad_improvement=bad_improvement,
                good_worsen=good_worsen,
                bad_control_beat_count=beat_count,
                bad_rows=bad_rows,
                args=args,
            )
            rows.append(
                {
                    "family": "merge_gauge_boundary_counterfactual",
                    "source": "v82_phase8e_projection_tol001_steps64",
                    "candidate_or_controller": candidate.get("candidate", ""),
                    "metric": metric,
                    "metric_alias": METRIC_ALIASES.get(metric, metric),
                    "bad_rows": bad_rows,
                    "good_rows": good_rows,
                    "bad_median_improvement_vs_baseline_ratio": bad_improvement,
                    "good_max_worsen_vs_baseline_ratio": good_worsen,
                    "bad_control_beat_count": beat_count,
                    "controls": candidate.get("controls", []),
                    "invalid_as_runtime_method": False,
                    "invalid_reason": "",
                    "gate": gate,
                    "counterfactual_gate_pass": gate["metric_counterfactual_gate_pass"],
                    "source_summary": str(args.phase8e_summary),
                }
            )
    return rows


def oracle_rows(payload: Mapping[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bad_rows = len(payload.get("bad_chunks", []) or [])
    good_rows = len(payload.get("good_chunks", []) or [])
    for controller in payload.get("controllers", []) if isinstance(payload.get("controllers"), list) else []:
        if not isinstance(controller, dict):
            continue
        invalid = bool(controller.get("invalid_as_runtime_method"))
        for metric, stats in sorted((controller.get("metric_summary") or {}).items()):
            if not isinstance(stats, dict):
                continue
            bad_improvement = safe_float(stats.get("bad_median_improvement_vs_baseline_ratio"))
            good_worsen = safe_float(stats.get("good_max_worsen_vs_baseline_ratio"))
            beat_count = int(safe_float(stats.get("bad_control_beat_count")) or 0)
            gate = gate_for_metric(
                bad_improvement=bad_improvement,
                good_worsen=good_worsen,
                bad_control_beat_count=beat_count,
                bad_rows=bad_rows,
                args=args,
            )
            rows.append(
                {
                    "family": "merge_gauge_oracle_upper_bound",
                    "source": "v82_phase12_merge_controller_oracle",
                    "candidate_or_controller": controller.get("controller", ""),
                    "metric": metric,
                    "metric_alias": metric,
                    "bad_rows": bad_rows,
                    "good_rows": good_rows,
                    "bad_median_improvement_vs_baseline_ratio": bad_improvement,
                    "good_max_worsen_vs_baseline_ratio": good_worsen,
                    "bad_control_beat_count": beat_count,
                    "controls": "geometry/random/shuffled controls in phase8e",
                    "invalid_as_runtime_method": invalid,
                    "invalid_reason": controller.get("invalid_reason", ""),
                    "gate": gate,
                    "counterfactual_gate_pass": gate["metric_counterfactual_gate_pass"],
                    "source_summary": str(args.oracle_summary),
                }
            )
    return rows


def skipped_rows(phase3: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "family": "swa_qk_counterfactual",
            "source": "v83_phase3_carrier_alignment",
            "candidate_or_controller": "skipped",
            "metric": "",
            "metric_alias": "",
            "bad_rows": "",
            "good_rows": "",
            "bad_median_improvement_vs_baseline_ratio": "",
            "good_max_worsen_vs_baseline_ratio": "",
            "bad_control_beat_count": "",
            "controls": "",
            "invalid_as_runtime_method": False,
            "invalid_reason": "Phase3 SWA carrier gate did not pass; do not run SWA QK counterfactual as a route to runtime action.",
            "gate": {
                "bad_improvement_ge_10pct": False,
                "good_worsen_le_2pct": False,
                "beats_all_controls_on_bad": False,
                "metric_counterfactual_gate_pass": False,
            },
            "counterfactual_gate_pass": False,
            "source_summary": str(DEFAULT_PHASE3_SUMMARY),
        },
        {
            "family": "ttt_write_counterfactual",
            "source": "v83_phase3_carrier_alignment",
            "candidate_or_controller": "skipped",
            "metric": "",
            "metric_alias": "",
            "bad_rows": "",
            "good_rows": "",
            "bad_median_improvement_vs_baseline_ratio": "",
            "good_max_worsen_vs_baseline_ratio": "",
            "bad_control_beat_count": "",
            "controls": "",
            "invalid_as_runtime_method": False,
            "invalid_reason": "TTT counterfactual requires confirmed SWA or merge/gauge evidence; Phase3 had no passing carrier.",
            "gate": {
                "bad_improvement_ge_10pct": False,
                "good_worsen_le_2pct": False,
                "beats_all_controls_on_bad": False,
                "metric_counterfactual_gate_pass": False,
            },
            "counterfactual_gate_pass": False,
            "source_summary": str(DEFAULT_PHASE3_SUMMARY),
        },
    ]


def render_report(path: Path, rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# ACL2 v83 Phase4 Counterfactual Upper-Bound Report",
        "",
        f"phase4_gate_pass: `{summary['phase4_gate_pass']}`",
        f"decision: `{summary['decision']}`",
        f"runtime_action_allowed: `{summary['runtime_action_allowed']}`",
        "",
        "| Family | Candidate | Metric | Gate | Bad Improvement | Good Worsen | Beat Count | Invalid Runtime |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {family} | {candidate} | {metric} | {gate} | {bad} | {good} | {beat} | {invalid} |".format(
                family=row.get("family", ""),
                candidate=row.get("candidate_or_controller", ""),
                metric=row.get("metric_alias") or row.get("metric", ""),
                gate=row.get("counterfactual_gate_pass", ""),
                bad=format_float(row.get("bad_median_improvement_vs_baseline_ratio")),
                good=format_float(row.get("good_max_worsen_vs_baseline_ratio")),
                beat=row.get("bad_control_beat_count", ""),
                invalid=row.get("invalid_as_runtime_method", ""),
            )
        )
    lines.extend(
        [
            "",
            "Gate requires bad improvement >=10%, good worsen <=2%, and beating geometry/random/shuffled controls on all bad chunks.",
            "GT/oracle rows are diagnostic only and are never treated as runtime methods.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def main() -> None:
    args = parse_args()
    phase3 = read_json(args.phase3_summary)
    phase8e = read_json(args.phase8e_summary)
    oracle = read_json(args.oracle_summary)
    rows = phase8e_rows(phase8e, args) + oracle_rows(oracle, args) + skipped_rows(phase3)
    non_invalid_passing = [
        row for row in rows if row.get("counterfactual_gate_pass") and not row.get("invalid_as_runtime_method")
    ]
    oracle_passing = [
        row for row in rows if row.get("counterfactual_gate_pass") and row.get("invalid_as_runtime_method")
    ]
    finite_improvements = [
        row
        for row in rows
        if safe_float(row.get("bad_median_improvement_vs_baseline_ratio")) is not None
    ]
    best = max(
        finite_improvements,
        key=lambda row: safe_float(row.get("bad_median_improvement_vs_baseline_ratio")) or -1.0,
    ) if finite_improvements else {}
    phase4_gate_pass = bool(non_invalid_passing or oracle_passing)
    runtime_action_allowed = bool(non_invalid_passing and phase3.get("phase3_gate_pass"))
    if non_invalid_passing and not phase3.get("phase3_gate_pass"):
        decision = "counterfactual_signal_exists_but_phase3_carrier_gate_blocks_runtime"
    elif non_invalid_passing:
        decision = "counterfactual_upper_bound_supports_runtime_design"
    elif oracle_passing:
        decision = "oracle_upper_bound_only_not_runtime"
    else:
        decision = "counterfactual_upper_bound_failed_stop_route"
    summary = {
        "schema": "acl2_v83_phase4_counterfactual_upper_bound_summary_v1",
        "phase3_summary": str(args.phase3_summary),
        "phase8e_summary": str(args.phase8e_summary),
        "oracle_summary": str(args.oracle_summary),
        "bad_improvement_min": args.bad_improvement_min,
        "good_worsen_max": args.good_worsen_max,
        "row_count": len(rows),
        "non_invalid_passing_count": len(non_invalid_passing),
        "oracle_passing_count": len(oracle_passing),
        "phase3_gate_pass": bool(phase3.get("phase3_gate_pass")),
        "phase4_gate_pass": phase4_gate_pass,
        "runtime_action_allowed": runtime_action_allowed,
        "decision": decision,
        "best_bad_improvement": {
            "family": best.get("family", ""),
            "candidate_or_controller": best.get("candidate_or_controller", ""),
            "metric": best.get("metric_alias") or best.get("metric", ""),
            "bad_median_improvement_vs_baseline_ratio": best.get("bad_median_improvement_vs_baseline_ratio", ""),
            "good_max_worsen_vs_baseline_ratio": best.get("good_max_worsen_vs_baseline_ratio", ""),
            "bad_control_beat_count": best.get("bad_control_beat_count", ""),
            "invalid_as_runtime_method": best.get("invalid_as_runtime_method", ""),
        },
        "stop_reason": (
            "No counterfactual/oracle row met bad>=10%, good<=2%, and all-bad control beating; "
            "SWA and TTT counterfactuals remain blocked by Phase3 carrier gate."
            if not phase4_gate_pass
            else ""
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "counterfactual_upper_bound_rows.csv", rows)
    write_json(args.out_dir / "counterfactual_upper_bound_summary.json", summary)
    render_report(args.out_dir / "counterfactual_upper_bound_report.md", rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
