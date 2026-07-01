#!/usr/bin/env python3
"""Decompose why v82 Phase12 contextual route rules fail promotion gates."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/acl2_v82tf_swa_carrier_semantic_scale_handoff")
DEFAULT_AUDIT = DEFAULT_ROOT / "phase12_contextual_route_rule_search" / "contextual_route_rule_audit.csv"
DEFAULT_OUT = DEFAULT_ROOT / "phase12_route_gate_failure_decomp"


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


def _gate(row: dict[str, str]) -> dict[str, Any]:
    try:
        parsed = ast.literal_eval(row.get("gate", "{}"))
    except (SyntaxError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _seq_coverage(row: dict[str, str]) -> int:
    try:
        parsed = ast.literal_eval(row.get("flagged_seq_coverage", "[]"))
    except (SyntaxError, ValueError):
        parsed = []
    return len(parsed) if isinstance(parsed, list) else 0


def _is_true(gate: dict[str, Any], key: str) -> bool:
    return bool(gate.get(key, False))


def _rank_key(row: dict[str, Any]) -> tuple[float, float, int, int]:
    bad = _float(row.get("bad_recall"))
    good = _float(row.get("good_false_positive_rate"))
    return (
        bad if bad is not None else -1.0,
        -(good if good is not None else 1.0),
        int(row.get("flagged_seq_coverage_count", 0)),
        int(row.get("rows", 0) or 0),
    )


def _row_digest(row: dict[str, str], gate: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "route_group": row.get("route_group", ""),
        "control_kind": row.get("control_kind", ""),
        "filter_name": row.get("filter_name", ""),
        "signal_name": row.get("signal_name", ""),
        "rows": row.get("rows", ""),
        "bad_recall": row.get("bad_recall", ""),
        "good_false_positive_rate": row.get("good_false_positive_rate", ""),
        "flagged_seq_coverage": row.get("flagged_seq_coverage", ""),
        "flagged_seq_coverage_count": _seq_coverage(row),
        "same_mass_random_rule_gate_pass": _is_true(gate, "same_mass_random_rule_gate_pass"),
        "semantic_shuffled_available_for_route_group": _is_true(
            gate, "semantic_shuffled_available_for_route_group"
        ),
        "semantic_shuffled_rule_gate_pass": _is_true(gate, "semantic_shuffled_rule_gate_pass"),
        "within_control_gate_pass": _is_true(gate, "within_control_gate_pass"),
        "rule_gate_pass": _is_true(gate, "rule_gate_pass"),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = _read_csv(args.audit_csv)
    gate_rows = [(row, _gate(row)) for row in rows]
    same_mass = [(row, gate) for row, gate in gate_rows if row.get("control_kind") == "same_mass_random"]
    same_mass_near = [
        (row, gate)
        for row, gate in same_mass
        if _is_true(gate, "within_control_gate_pass")
        and _is_true(gate, "same_mass_random_rule_gate_pass")
        and not _is_true(gate, "rule_gate_pass")
    ]
    near_rows = [_row_digest(row, gate) for row, gate in same_mass_near]
    near_rows = sorted(near_rows, key=_rank_key, reverse=True)

    gate_counter: Counter[str] = Counter()
    for _row, gate in same_mass:
        for key in [
            "bad_recall_ge_0_60",
            "good_false_positive_rate_le_0_25",
            "seq_coverage_ge_3",
            "pair_counts_nonzero",
            "actual_beats_named_control",
            "within_control_gate_pass",
            "same_mass_random_rule_gate_pass",
            "semantic_shuffled_available_for_route_group",
            "semantic_shuffled_rule_gate_pass",
            "phase5_required_controls_gate_pass",
            "rule_gate_pass",
        ]:
            if not _is_true(gate, key):
                gate_counter[key] += 1

    route_summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "same_mass_rule_rows": 0,
            "same_mass_within_gate_rows": 0,
            "same_mass_near_without_full_gate_rows": 0,
            "semantic_shuffle_available_rows": 0,
            "semantic_shuffle_pass_rows": 0,
            "rule_gate_pass_rows": 0,
            "best_bad_recall": None,
            "best_good_false_positive_rate": None,
            "best_seq_coverage_count": 0,
        }
    )
    for row, gate in same_mass:
        route = row.get("route_group", "")
        summary = route_summary[route]
        summary["same_mass_rule_rows"] += 1
        if _is_true(gate, "within_control_gate_pass"):
            summary["same_mass_within_gate_rows"] += 1
        if _is_true(gate, "within_control_gate_pass") and not _is_true(gate, "rule_gate_pass"):
            summary["same_mass_near_without_full_gate_rows"] += 1
        if _is_true(gate, "semantic_shuffled_available_for_route_group"):
            summary["semantic_shuffle_available_rows"] += 1
        if _is_true(gate, "semantic_shuffled_rule_gate_pass"):
            summary["semantic_shuffle_pass_rows"] += 1
        if _is_true(gate, "rule_gate_pass"):
            summary["rule_gate_pass_rows"] += 1
        bad = _float(row.get("bad_recall"))
        good = _float(row.get("good_false_positive_rate"))
        seqs = _seq_coverage(row)
        current = (
            summary["best_bad_recall"] if summary["best_bad_recall"] is not None else -1.0,
            -(summary["best_good_false_positive_rate"] if summary["best_good_false_positive_rate"] is not None else 1.0),
            int(summary["best_seq_coverage_count"]),
        )
        candidate = (bad if bad is not None else -1.0, -(good if good is not None else 1.0), seqs)
        if candidate > current:
            summary["best_bad_recall"] = bad
            summary["best_good_false_positive_rate"] = good
            summary["best_seq_coverage_count"] = seqs

    route_rows = [
        {"route_group": route, **dict(values)}
        for route, values in sorted(route_summary.items())
    ]
    _write_csv(args.out_dir / "route_gate_failure_by_route_group.csv", route_rows)
    _write_csv(args.out_dir / "route_gate_near_misses.csv", near_rows[:200])

    near_with_shuffle_available = [
        row for row in near_rows if row["semantic_shuffled_available_for_route_group"]
    ]
    near_without_shuffle_available = [
        row for row in near_rows if not row["semantic_shuffled_available_for_route_group"]
    ]
    summary = {
        "schema": "acl2_v82_phase12_route_gate_failure_decomp_v1",
        "input_audit_csv": str(args.audit_csv),
        "total_rule_rows": len(rows),
        "same_mass_rule_rows": len(same_mass),
        "same_mass_near_without_full_gate_rows": len(near_rows),
        "near_with_semantic_shuffle_available": len(near_with_shuffle_available),
        "near_without_semantic_shuffle_available": len(near_without_shuffle_available),
        "fully_passing_rule_rows": sum(1 for _row, gate in gate_rows if _is_true(gate, "rule_gate_pass")),
        "same_mass_failure_counts": dict(gate_counter),
        "route_group_summary_csv": str(args.out_dir / "route_gate_failure_by_route_group.csv"),
        "near_miss_csv": str(args.out_dir / "route_gate_near_misses.csv"),
        "top_near_misses": near_rows[:20],
    }
    _write_json(args.out_dir / "route_gate_failure_decomp_summary.json", summary)

    lines = [
        "# v82 Phase12 Route Gate Failure Decomposition",
        "",
        f"total_rule_rows: {summary['total_rule_rows']}",
        f"same_mass_rule_rows: {summary['same_mass_rule_rows']}",
        f"fully_passing_rule_rows: {summary['fully_passing_rule_rows']}",
        f"same_mass_near_without_full_gate_rows: {summary['same_mass_near_without_full_gate_rows']}",
        f"near_with_semantic_shuffle_available: {summary['near_with_semantic_shuffle_available']}",
        f"near_without_semantic_shuffle_available: {summary['near_without_semantic_shuffle_available']}",
        "",
        "## Same-Mass Failure Counts",
    ]
    for key, count in sorted(gate_counter.items()):
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Route Group Summary"])
    for row in route_rows:
        lines.append(
            "- {route_group}: within={same_mass_within_gate_rows} near={same_mass_near_without_full_gate_rows} "
            "shuffle_available_rows={semantic_shuffle_available_rows} shuffle_pass_rows={semantic_shuffle_pass_rows} "
            "rule_pass={rule_gate_pass_rows} best_bad={best_bad_recall} best_good_fp={best_good_false_positive_rate} "
            "best_seq_cov={best_seq_coverage_count}".format(**row)
        )
    lines.extend(["", "## Top Near Misses"])
    for row in near_rows[:20]:
        lines.append(
            "- {route_group} / {filter_name} / {signal_name}: bad={bad_recall} good_fp={good_false_positive_rate} "
            "seq_cov={flagged_seq_coverage_count} shuffle_available={semantic_shuffled_available_for_route_group} "
            "shuffle_pass={semantic_shuffled_rule_gate_pass}".format(**row)
        )
    (args.out_dir / "route_gate_failure_decomp_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
