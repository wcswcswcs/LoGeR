#!/usr/bin/env python3
"""Search contextual v82 Phase12 route-control rules without changing the model.

The search only uses pair-bank / artifact metadata as filters. It then tests
whether an actual-vs-control metric has the desired sign on bad pairs while
protecting good pairs and passing same-mass plus semantic-shuffled controls.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/acl2_v82tf_swa_carrier_semantic_scale_handoff")
DEFAULT_OUT = DEFAULT_ROOT / "phase12_contextual_route_rule_search"


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


FilterFn = Callable[[dict[str, Any]], bool]


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


def _atomic_filter_specs(rows: list[dict[str, Any]]) -> list[tuple[str, str, FilterFn]]:
    overlap_vals = [v for v in (_float(row.get("overlap_scale_residual")) for row in rows) if v is not None]
    conf_vals = [v for v in (_float(row.get("semantic_confidence_mean")) for row in rows) if v is not None]
    stable_vals = [v for v in (_float(row.get("stable_overlap_mass")) for row in rows) if v is not None]
    harm_vals = [v for v in (_float(row.get("harm_overlap_mass")) for row in rows) if v is not None]
    context_vals = [v for v in (_float(row.get("context_overlap_mass")) for row in rows) if v is not None]
    thresholds = {
        "overlap_q50": _quantile(overlap_vals, 0.50),
        "overlap_q75": _quantile(overlap_vals, 0.75),
        "conf_q25": _quantile(conf_vals, 0.25),
        "conf_q50": _quantile(conf_vals, 0.50),
        "conf_q75": _quantile(conf_vals, 0.75),
        "stable_q50": _quantile(stable_vals, 0.50),
        "harm_q50": _quantile(harm_vals, 0.50),
        "context_q50": _quantile(context_vals, 0.50),
    }

    def ge(field: str, threshold_key: str) -> FilterFn:
        def inner(row: dict[str, Any]) -> bool:
            value = _float(row.get(field))
            threshold = thresholds[threshold_key]
            return value is not None and threshold is not None and value >= threshold

        return inner

    def le(field: str, threshold_key: str) -> FilterFn:
        def inner(row: dict[str, Any]) -> bool:
            value = _float(row.get(field))
            threshold = thresholds[threshold_key]
            return value is not None and threshold is not None and value <= threshold

        return inner

    return [
        ("high_quality", "quality_type == high_quality", lambda r: r.get("quality_type") == "high_quality"),
        ("low_conf_stress", "quality_type == low_conf_stress", lambda r: r.get("quality_type") == "low_conf_stress"),
        ("artifact_risk", "artifact_quality_risk truthy", lambda r: _truthy(r.get("artifact_quality_risk"))),
        ("overlap_resid_ge_q50", f"overlap_scale_residual >= {thresholds['overlap_q50']}", ge("overlap_scale_residual", "overlap_q50")),
        ("overlap_resid_ge_q75", f"overlap_scale_residual >= {thresholds['overlap_q75']}", ge("overlap_scale_residual", "overlap_q75")),
        ("overlap_resid_le_q50", f"overlap_scale_residual <= {thresholds['overlap_q50']}", le("overlap_scale_residual", "overlap_q50")),
        ("sem_conf_ge_q25", f"semantic_confidence_mean >= {thresholds['conf_q25']}", ge("semantic_confidence_mean", "conf_q25")),
        ("sem_conf_ge_q50", f"semantic_confidence_mean >= {thresholds['conf_q50']}", ge("semantic_confidence_mean", "conf_q50")),
        ("sem_conf_ge_q75", f"semantic_confidence_mean >= {thresholds['conf_q75']}", ge("semantic_confidence_mean", "conf_q75")),
        ("sem_conf_le_q50", f"semantic_confidence_mean <= {thresholds['conf_q50']}", le("semantic_confidence_mean", "conf_q50")),
        (
            "stable_gt_harm",
            "stable_overlap_mass > harm_overlap_mass",
            lambda r: (_float(r.get("stable_overlap_mass")) is not None)
            and (_float(r.get("harm_overlap_mass")) is not None)
            and _float(r.get("stable_overlap_mass")) > _float(r.get("harm_overlap_mass")),
        ),
        (
            "harm_ge_stable",
            "harm_overlap_mass >= stable_overlap_mass",
            lambda r: (_float(r.get("stable_overlap_mass")) is not None)
            and (_float(r.get("harm_overlap_mass")) is not None)
            and _float(r.get("harm_overlap_mass")) >= _float(r.get("stable_overlap_mass")),
        ),
        ("stable_mass_ge_q50", f"stable_overlap_mass >= {thresholds['stable_q50']}", ge("stable_overlap_mass", "stable_q50")),
        ("harm_mass_ge_q50", f"harm_overlap_mass >= {thresholds['harm_q50']}", ge("harm_overlap_mass", "harm_q50")),
        ("context_mass_ge_q50", f"context_overlap_mass >= {thresholds['context_q50']}", ge("context_overlap_mass", "context_q50")),
        ("context_mass_ge_0_5", "context_overlap_mass >= 0.5", lambda r: (_float(r.get("context_overlap_mass")) or -1.0) >= 0.5),
    ]


def _filter_specs(rows: list[dict[str, Any]], max_terms: int) -> list[tuple[str, str, FilterFn]]:
    atoms = _atomic_filter_specs(rows)
    specs: list[tuple[str, str, FilterFn]] = [("all_pairs", "no additional filter", lambda r: True)]
    seen_rowsets: set[tuple[int, ...]] = set()
    for term_count in range(1, max_terms + 1):
        for combo in itertools.combinations(atoms, term_count):
            names = tuple(item[0] for item in combo)
            descs = tuple(item[1] for item in combo)
            fns = tuple(item[2] for item in combo)

            def make_filter(inner_fns: tuple[FilterFn, ...]) -> FilterFn:
                return lambda row: all(fn(row) for fn in inner_fns)

            fn = make_filter(fns)
            rowset = tuple(idx for idx, row in enumerate(rows) if fn(row))
            if not rowset or rowset in seen_rowsets:
                continue
            seen_rowsets.add(rowset)
            specs.append((" AND ".join(names), " AND ".join(descs), fn))
    return specs


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
    filter_fn: FilterFn,
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
    vals = [v for v in (_float(row.get(metric_key)) for row in subset) if v is not None]
    bad_vals = [v for v in (_float(row.get(metric_key)) for row in bad) if v is not None]
    good_vals = [v for v in (_float(row.get(metric_key)) for row in good) if v is not None]
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
    gate["within_control_gate_pass"] = all(gate.values())
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
        default=DEFAULT_ROOT / "phase12_per_head_route_localization" / "per_head_route_paired_controls.csv",
    )
    parser.add_argument(
        "--pair-bank",
        type=Path,
        default=DEFAULT_ROOT / "phase2_swa_pair_bank_v2" / "swa_pair_bank_v2.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-terms", type=int, default=3)
    args = parser.parse_args()

    joined = _join_rows(_read_csv(args.paired_controls), _read_csv(args.pair_bank))
    _write_csv(args.out_dir / "contextual_route_rule_rows_joined.csv", joined)
    filters = _filter_specs(joined, max(1, int(args.max_terms)))
    route_groups = sorted({row.get("route_group", "") for row in joined if row.get("route_group")})
    control_kinds = sorted({row.get("control_kind", "") for row in joined if row.get("control_kind")})
    has_shuffle = {
        route_group: any(
            row.get("route_group") == route_group and row.get("control_kind") == "shuffled_semantic"
            for row in joined
        )
        for route_group in route_groups
    }

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
        ): bool(row["gate"]["within_control_gate_pass"])
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
        row["gate"]["semantic_shuffled_available_for_route_group"] = bool(has_shuffle.get(row["route_group"], False))
        row["gate"]["phase5_required_controls_gate_pass"] = bool(
            same_mass_pass and semantic_shuffle_pass and has_shuffle.get(row["route_group"], False)
        )
        row["gate"]["rule_gate_pass"] = bool(
            row["control_kind"] == "same_mass_random"
            and row["gate"]["within_control_gate_pass"]
            and row["gate"]["phase5_required_controls_gate_pass"]
        )

    _write_csv(args.out_dir / "contextual_route_rule_audit.csv", rule_rows)
    passing = [row for row in rule_rows if row["gate"]["rule_gate_pass"]]
    best = sorted(
        rule_rows,
        key=lambda r: (
            r["bad_recall"] if r["bad_recall"] is not None else -1.0,
            -(r["good_false_positive_rate"] if r["good_false_positive_rate"] is not None else 1.0),
            len(r["flagged_seq_coverage"]),
            r["rows"],
        ),
        reverse=True,
    )[:30]
    near = [
        row
        for row in rule_rows
        if row["control_kind"] == "same_mass_random"
        and row["gate"]["within_control_gate_pass"]
        and row["gate"]["same_mass_random_rule_gate_pass"]
    ]
    near = sorted(
        near,
        key=lambda r: (
            bool(r["gate"]["semantic_shuffled_rule_gate_pass"]),
            len(r["flagged_seq_coverage"]),
            r["bad_recall"] if r["bad_recall"] is not None else -1.0,
            -(r["good_false_positive_rate"] if r["good_false_positive_rate"] is not None else 1.0),
        ),
        reverse=True,
    )[:30]
    summary = {
        "schema": "acl2_v82_phase12_contextual_route_rule_search_v1",
        "joined_rows": len(joined),
        "base_case_missing_rows": len([row for row in joined if row.get("base_case_type") not in {"bad", "good"}]),
        "filter_count": len(filters),
        "rule_rows": len(rule_rows),
        "passing_rule_count": len(passing),
        "phase5_contextual_rule_gate_pass": bool(passing),
        "passing_rules": passing,
        "near_miss_same_mass_rules": near,
        "best_rules_by_bad_recall": best,
        "decision": "pass_to_action_audit" if passing else "no_go_contextual_route_rule_not_selective",
        "blocker": ""
        if passing
        else (
            "No contextual route-control rule met bad_recall>=0.60, "
            "good_false_positive_rate<=0.25, seq_coverage>=3, same-mass random, "
            "and semantic-shuffled controls together."
        ),
    }
    _write_json(args.out_dir / "contextual_route_rule_search_summary.json", summary)
    report = [
        "# v82 Phase12 Contextual Route Rule Search",
        "",
        f"joined_rows: {summary['joined_rows']}",
        f"base_case_missing_rows: {summary['base_case_missing_rows']}",
        f"filter_count: {summary['filter_count']}",
        f"rule_rows: {summary['rule_rows']}",
        f"passing_rule_count: {summary['passing_rule_count']}",
        f"decision: {summary['decision']}",
        f"blocker: {summary['blocker'] or 'none'}",
        "",
        "## Near Miss Same-Mass Rules",
    ]
    for row in near[:15]:
        report.extend(
            [
                "",
                f"- {row['route_group']} / {row['filter_name']} / {row['signal_name']}",
                (
                    f"  rows={row['rows']} bad_recall={row['bad_recall']} "
                    f"good_fp={row['good_false_positive_rate']} seqs={row['flagged_seq_coverage']} "
                    f"shuffle_pass={row['gate']['semantic_shuffled_rule_gate_pass']}"
                ),
                f"  filter={row['filter_description']}",
            ]
        )
    (args.out_dir / "contextual_route_rule_search_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
