#!/usr/bin/env python3
"""Audit v82 Phase12 route-control evidence as a Phase5-style rule refinement."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/acl2_v82tf_swa_carrier_semantic_scale_handoff")
DEFAULT_OUT = DEFAULT_ROOT / "phase12_route_control_rule_refinement"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _norm_seq(seq: str) -> str:
    return str(seq).split("_")[0]


def _pair_key(row: dict[str, str]) -> tuple[str, str]:
    chunk = row.get("chunk", "")
    try:
        chunk = str(int(chunk))
    except ValueError:
        pass
    return _norm_seq(row.get("seq", "")), chunk


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * q)))
    return vals[idx]


def _join_rows(paired_rows: list[dict[str, str]], pair_bank: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_pair = {(row.get("seq", ""), row.get("curr_chunk", "")): row for row in pair_bank}
    joined: list[dict[str, Any]] = []
    for row in paired_rows:
        out: dict[str, Any] = dict(row)
        pair = by_pair.get(_pair_key(row), {})
        for key in [
            "case_type",
            "base_case_type",
            "quality_type",
            "quality_source",
            "future_after_overlap",
            "boundary_jump",
            "overlap_scale_residual",
            "semantic_confidence_mean",
            "stable_overlap_mass",
            "harm_overlap_mass",
            "context_overlap_mass",
            "artifact_quality_risk",
            "forbidden_as_stable_evidence",
        ]:
            out[key] = pair.get(key, "")
        out["seq_base"] = _norm_seq(str(out.get("seq", "")))
        joined.append(out)
    return joined


def _filter_specs(rows: list[dict[str, Any]]) -> list[tuple[str, str, Any]]:
    overlap_vals = [
        v for v in (_float(row.get("overlap_scale_residual")) for row in rows) if v is not None
    ]
    conf_vals = [
        v for v in (_float(row.get("semantic_confidence_mean")) for row in rows) if v is not None
    ]
    overlap_q50 = _quantile(overlap_vals, 0.50)
    conf_q50 = _quantile(conf_vals, 0.50)
    return [
        ("all_pairs", "no additional filter", lambda r: True),
        ("high_quality_only", "quality_type == high_quality", lambda r: r.get("quality_type") == "high_quality"),
        ("low_conf_stress_only", "quality_type == low_conf_stress", lambda r: r.get("quality_type") == "low_conf_stress"),
        (
            "artifact_quality_risk_only",
            "artifact_quality_risk truthy",
            lambda r: _truthy(r.get("artifact_quality_risk")),
        ),
        (
            "overlap_scale_residual_ge_median",
            f"overlap_scale_residual >= distribution median {overlap_q50}",
            lambda r: overlap_q50 is not None
            and (_float(r.get("overlap_scale_residual")) is not None)
            and _float(r.get("overlap_scale_residual")) >= overlap_q50,
        ),
        (
            "semantic_confidence_ge_median",
            f"semantic_confidence_mean >= distribution median {conf_q50}",
            lambda r: conf_q50 is not None
            and (_float(r.get("semantic_confidence_mean")) is not None)
            and _float(r.get("semantic_confidence_mean")) >= conf_q50,
        ),
        (
            "harm_mass_ge_stable_mass",
            "harm_overlap_mass >= stable_overlap_mass",
            lambda r: (_float(r.get("harm_overlap_mass")) is not None)
            and (_float(r.get("stable_overlap_mass")) is not None)
            and _float(r.get("harm_overlap_mass")) >= _float(r.get("stable_overlap_mass")),
        ),
        (
            "stable_mass_gt_harm_mass",
            "stable_overlap_mass > harm_overlap_mass",
            lambda r: (_float(r.get("harm_overlap_mass")) is not None)
            and (_float(r.get("stable_overlap_mass")) is not None)
            and _float(r.get("stable_overlap_mass")) > _float(r.get("harm_overlap_mass")),
        ),
        (
            "context_mass_ge_half",
            "context_overlap_mass >= 0.5",
            lambda r: (_float(r.get("context_overlap_mass")) is not None)
            and _float(r.get("context_overlap_mass")) >= 0.5,
        ),
    ]


SIGNALS = [
    (
        "selected_lift_positive",
        "actual_minus_control_mean_swa_overlap_attention_mass_selected_lift",
        "positive",
    ),
    (
        "source_lift_positive",
        "actual_minus_control_mean_swa_overlap_attention_mass_source_lift",
        "positive",
    ),
    (
        "head10_rmse_improves",
        "actual_minus_control_head10_to_tail10_pose_sim3_rmse_m",
        "negative",
    ),
    (
        "overlap3_future_rmse_improves",
        "actual_minus_control_overlap3_to_future_pose_sim3_rmse_m",
        "negative",
    ),
    (
        "scale_cv_improves",
        "actual_minus_control_scale_cv_head_mid_tail_pose_sim3",
        "negative",
    ),
]


def _flag(value: float | None, direction: str) -> bool:
    if value is None:
        return False
    if direction == "positive":
        return value > 0.0
    if direction == "negative":
        return value < 0.0
    raise ValueError(direction)


def _eval_rule(
    rows: list[dict[str, Any]],
    route_group: str,
    control_kind: str,
    filter_name: str,
    filter_desc: str,
    filter_fn: Any,
    signal_name: str,
    metric_key: str,
    direction: str,
) -> dict[str, Any]:
    subset = [
        row
        for row in rows
        if row.get("route_group") == route_group
        and row.get("control_kind") == control_kind
        and row.get("base_case_type") in {"bad", "good"}
        and filter_fn(row)
    ]
    bad = [row for row in subset if row.get("base_case_type") == "bad"]
    good = [row for row in subset if row.get("base_case_type") == "good"]
    flagged = [row for row in subset if _flag(_float(row.get(metric_key)), direction)]
    bad_flagged = [row for row in bad if _flag(_float(row.get(metric_key)), direction)]
    good_flagged = [row for row in good if _flag(_float(row.get(metric_key)), direction)]
    vals = [_float(row.get(metric_key)) for row in subset]
    vals = [v for v in vals if v is not None]
    bad_vals = [_float(row.get(metric_key)) for row in bad]
    bad_vals = [v for v in bad_vals if v is not None]
    good_vals = [_float(row.get(metric_key)) for row in good]
    good_vals = [v for v in good_vals if v is not None]
    bad_recall = len(bad_flagged) / len(bad) if bad else None
    good_fp = len(good_flagged) / len(good) if good else None
    seqs = sorted({row.get("seq_base", "") for row in flagged if row.get("seq_base")})
    gate = {
        "bad_recall_ge_0_60": bad_recall is not None and bad_recall >= 0.60,
        "good_false_positive_rate_le_0_25": good_fp is not None and good_fp <= 0.25,
        "seq_coverage_ge_3": len(seqs) >= 3,
        "pair_counts_nonzero": bool(bad and good),
        "actual_beats_named_control": bool(flagged),
    }
    gate["rule_gate_pass"] = all(gate.values())
    return {
        "route_group": route_group,
        "control_kind": control_kind,
        "filter_name": filter_name,
        "filter_description": filter_desc,
        "signal_name": signal_name,
        "metric_key": metric_key,
        "direction": direction,
        "rows": len(subset),
        "bad_rows": len(bad),
        "good_rows": len(good),
        "flagged_rows": len(flagged),
        "bad_flagged": len(bad_flagged),
        "good_flagged": len(good_flagged),
        "bad_recall": bad_recall,
        "good_false_positive_rate": good_fp,
        "flagged_seq_coverage": seqs,
        "metric_median": _median(vals),
        "bad_metric_median": _median(bad_vals),
        "good_metric_median": _median(good_vals),
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-controls",
        type=Path,
        default=DEFAULT_ROOT
        / "phase12_per_head_route_localization"
        / "per_head_route_paired_controls.csv",
    )
    parser.add_argument(
        "--pair-bank",
        type=Path,
        default=DEFAULT_ROOT / "phase2_swa_pair_bank_v2" / "swa_pair_bank_v2.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    joined = _join_rows(_read_csv(args.paired_controls), _read_csv(args.pair_bank))
    _write_csv(args.out_dir / "route_control_rule_rows_joined.csv", joined)

    route_groups = sorted({row.get("route_group", "") for row in joined if row.get("route_group")})
    control_kinds = sorted({row.get("control_kind", "") for row in joined if row.get("control_kind")})
    filters = _filter_specs(joined)
    rule_rows: list[dict[str, Any]] = []
    for route_group in route_groups:
        for control_kind in control_kinds:
            if not any(
                row.get("route_group") == route_group and row.get("control_kind") == control_kind
                for row in joined
            ):
                continue
            for filter_name, filter_desc, filter_fn in filters:
                for signal_name, metric_key, direction in SIGNALS:
                    rule_rows.append(
                        _eval_rule(
                            joined,
                            route_group,
                            control_kind,
                            filter_name,
                            filter_desc,
                            filter_fn,
                            signal_name,
                            metric_key,
                            direction,
                        )
                    )

    base_gate_by_key = {
        (
            row["route_group"],
            row["control_kind"],
            row["filter_name"],
            row["signal_name"],
            row["metric_key"],
            row["direction"],
        ): bool(row["gate"]["rule_gate_pass"])
        for row in rule_rows
    }
    for row in rule_rows:
        same_key = (
            row["route_group"],
            "same_mass_random",
            row["filter_name"],
            row["signal_name"],
            row["metric_key"],
            row["direction"],
        )
        shuffle_key = (
            row["route_group"],
            "shuffled_semantic",
            row["filter_name"],
            row["signal_name"],
            row["metric_key"],
            row["direction"],
        )
        same_mass_pass = base_gate_by_key.get(same_key, False)
        semantic_shuffle_pass = base_gate_by_key.get(shuffle_key, False)
        row["gate"]["same_mass_random_rule_gate_pass"] = same_mass_pass
        row["gate"]["semantic_shuffled_rule_gate_pass"] = semantic_shuffle_pass
        row["gate"]["phase5_required_controls_gate_pass"] = bool(same_mass_pass and semantic_shuffle_pass)
        row["gate"]["within_control_gate_pass"] = bool(row["gate"]["rule_gate_pass"])
        row["gate"]["rule_gate_pass"] = bool(
            row["gate"]["within_control_gate_pass"]
            and row["control_kind"] == "same_mass_random"
            and row["gate"]["phase5_required_controls_gate_pass"]
        )
    _write_csv(args.out_dir / "route_control_rule_audit.csv", rule_rows)

    passing = [row for row in rule_rows if row["gate"]["rule_gate_pass"]]
    best = sorted(
        rule_rows,
        key=lambda r: (
            r["bad_recall"] if r["bad_recall"] is not None else -1.0,
            -(r["good_false_positive_rate"] if r["good_false_positive_rate"] is not None else 1.0),
            len(r["flagged_seq_coverage"]),
        ),
        reverse=True,
    )[:20]
    summary = {
        "schema": "acl2_v82_phase12_route_control_rule_refinement_v1",
        "joined_rows": len(joined),
        "rule_rows": len(rule_rows),
        "passing_rule_count": len(passing),
        "phase5_rule_gate_pass": bool(passing),
        "passing_rules": passing,
        "best_rules_by_bad_recall": best,
        "decision": "pass_to_action_audit" if passing else "no_go_route_control_rule_not_selective",
        "blocker": ""
        if passing
        else (
            "No fixed route-control rule met Phase5 bad_recall>=0.60, "
            "good_false_positive_rate<=0.25, seq_coverage>=3, same-route-mass random, "
            "and semantic-shuffled control gates together."
        ),
    }
    _write_json(args.out_dir / "route_control_rule_refinement_summary.json", summary)
    report = [
        "# v82 Phase12 Route-Control Rule Refinement",
        "",
        f"joined_rows: {summary['joined_rows']}",
        f"rule_rows: {summary['rule_rows']}",
        f"passing_rule_count: {summary['passing_rule_count']}",
        f"decision: {summary['decision']}",
        f"blocker: {summary['blocker'] or 'none'}",
        "",
        "## Top Rules By Bad Recall",
    ]
    for row in best[:12]:
        report.extend(
            [
                "",
                (
                    f"- {row['route_group']} / {row['control_kind']} / "
                    f"{row['filter_name']} / {row['signal_name']}"
                ),
                (
                    f"  rows={row['rows']} bad_recall={row['bad_recall']} "
                    f"good_fp={row['good_false_positive_rate']} "
                    f"seqs={row['flagged_seq_coverage']} pass={row['gate']['rule_gate_pass']}"
                ),
                f"  metric_median={row['metric_median']} bad_median={row['bad_metric_median']} good_median={row['good_metric_median']}",
            ]
        )
    (args.out_dir / "route_control_rule_refinement_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
