#!/usr/bin/env python3
"""Audit ACL2 v85 Phase1 anchor-pair universe sufficiency."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_IN_DIR = Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase1_anchor_pair_universe")
POSITIVE_CLASSES = {"A_STRONG_MATURE", "A_STRONG_BOOTSTRAP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--min-pairs-per-dim", type=int, default=3)
    parser.add_argument("--max-zero-conf-positive-ratio", type=float, default=0.10)
    parser.add_argument("--min-feature-availability", type=float, default=0.90)
    parser.add_argument("--support-sequence-target", type=int, default=4)
    parser.add_argument("--no-support-sequence-target", type=int, default=3)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def safe_int(value: Any) -> int:
    try:
        return int(float(str(value or "0")))
    except ValueError:
        return 0


def safe_float(value: Any) -> float:
    try:
        out = float(str(value or "0"))
    except ValueError:
        return 0.0
    return out if math.isfinite(out) else 0.0


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def is_labelled(case_label: str) -> bool:
    return case_label in {"bad", "good"}


def is_stress(row: Mapping[str, Any]) -> bool:
    return row.get("quality_label") == "low_conf_stress" or row.get("anchor_support_class") == "A_STRESS_SEQ01"


def pair_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row.get("seq", "")).zfill(2), str(row.get("prev_chunk", "")), str(row.get("curr_chunk", ""))


def main() -> None:
    args = parse_args()
    in_dir = args.in_dir
    rows = read_csv(in_dir / "anchor_pair_rows.csv")
    by_pair = read_csv(in_dir / "anchor_pair_by_seq_chunk.csv")

    support_available = any(row.get("case_label") == "unlabelled_support" for row in by_pair)
    sequence_target = args.support_sequence_target if support_available else args.no_support_sequence_target
    sequence_coverage = len({row.get("seq") for row in by_pair if row.get("seq")})
    labelled_pair_rows = [row for row in by_pair if is_labelled(str(row.get("case_label")))]
    nonstress_pair_rows = [row for row in by_pair if row.get("quality_label") != "low_conf_stress"]
    min_anchor_count = args.feature_dim * args.min_pairs_per_dim
    anchor_count_pass_rows = [
        row for row in nonstress_pair_rows if safe_int(row.get("anchor_pair_count")) >= min_anchor_count
    ]
    anchor_count_pass_ratio = len(anchor_count_pass_rows) / len(nonstress_pair_rows) if nonstress_pair_rows else 0.0

    bad_pairs_with_positive = {
        pair_key(row)
        for row in rows
        if row.get("case_label") == "bad"
        and row.get("quality_label") != "low_conf_stress"
        and row.get("anchor_support_class") in POSITIVE_CLASSES
    }
    positives = [row for row in rows if row.get("anchor_support_class") in POSITIVE_CLASSES]
    zero_conf_positive = [row for row in positives if parse_bool(row.get("zero_conf_flag"))]
    zero_conf_positive_ratio = len(zero_conf_positive) / len(positives) if positives else 0.0

    labelled_nonstress_rows = [
        row for row in rows if is_labelled(str(row.get("case_label"))) and not is_stress(row)
    ]
    q_available = [row for row in labelled_nonstress_rows if parse_bool(row.get("feature_q_available"))]
    k_available = [row for row in labelled_nonstress_rows if parse_bool(row.get("feature_k_available"))]
    q_ratio = len(q_available) / len(labelled_nonstress_rows) if labelled_nonstress_rows else 0.0
    k_ratio = len(k_available) / len(labelled_nonstress_rows) if labelled_nonstress_rows else 0.0

    checks = {
        "adjacent_labelled_rows_ge_24": len(labelled_pair_rows) >= 24,
        "sequence_coverage_pass": sequence_coverage >= sequence_target,
        "anchor_pair_count_pass_ratio_ge_60pct": anchor_count_pass_ratio >= 0.60,
        "strong_bad_pair_rows_ge_5": len(bad_pairs_with_positive) >= 5,
        "zero_conf_positive_ratio_le_10pct": zero_conf_positive_ratio <= args.max_zero_conf_positive_ratio,
        "feature_q_availability_ge_90pct": q_ratio >= args.min_feature_availability,
        "feature_k_availability_ge_90pct": k_ratio >= args.min_feature_availability,
    }
    fail_reasons = [key for key, passed in checks.items() if not passed]
    phase1_gate_pass = not fail_reasons

    class_counts = Counter(row.get("anchor_support_class") for row in rows)
    case_counts = Counter(row.get("case_label") for row in rows)
    quality_counts = Counter(row.get("quality_label") for row in rows)
    summary = {
        "phase": "Phase1_anchor_pair_universe",
        "phase1_gate_pass": phase1_gate_pass,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "feature_dim": args.feature_dim,
        "min_anchor_count_per_nonstress_pair": min_anchor_count,
        "checks": checks,
        "fail_reasons": fail_reasons,
        "row_count": len(rows),
        "pair_row_count": len(by_pair),
        "adjacent_labelled_rows": len(labelled_pair_rows),
        "sequence_coverage": sequence_coverage,
        "sequence_target": sequence_target,
        "support_available": support_available,
        "nonstress_pair_rows": len(nonstress_pair_rows),
        "anchor_pair_count_pass_rows": len(anchor_count_pass_rows),
        "anchor_pair_count_pass_ratio": anchor_count_pass_ratio,
        "strong_bad_pair_rows": len(bad_pairs_with_positive),
        "positive_anchor_rows": len(positives),
        "zero_conf_positive_rows": len(zero_conf_positive),
        "zero_conf_positive_ratio": zero_conf_positive_ratio,
        "labelled_nonstress_anchor_rows": len(labelled_nonstress_rows),
        "feature_q_available_rows": len(q_available),
        "feature_k_available_rows": len(k_available),
        "feature_q_availability_ratio": q_ratio,
        "feature_k_availability_ratio": k_ratio,
        "anchor_support_class_counts": dict(sorted(class_counts.items())),
        "case_label_counts": dict(sorted(case_counts.items())),
        "quality_label_counts": dict(sorted(quality_counts.items())),
        "next_action": (
            "Phase2_direct_QK_dump_repair_required"
            if (not checks["feature_q_availability_ge_90pct"] or not checks["feature_k_availability_ge_90pct"])
            else "repair_phase1_anchor_support"
            if fail_reasons
            else "Phase2_qk_feature_sanity"
        ),
        "notes": [
            "Phase1 does not allow runtime action or TTT.",
            "v84 QK compatibility proxies are not counted as true Q/K feature availability.",
            "A_STRONG_MATURE is not assigned without explicit historical maturity evidence.",
        ],
    }
    write_json(in_dir / "anchor_pair_sufficiency_summary.json", summary)

    audit_rows = [
        {"check": key, "pass": passed, "value": summary_value_for(key, summary)}
        for key, passed in checks.items()
    ]
    write_csv(in_dir / "anchor_pair_gate_checks.csv", audit_rows)
    write_report(in_dir / "anchor_pair_universe_audit_report.md", summary)

    print(f"phase1_gate_pass={str(phase1_gate_pass).lower()}")
    print(f"fail_reasons={','.join(fail_reasons) if fail_reasons else 'none'}")
    print(f"adjacent_labelled_rows={len(labelled_pair_rows)}")
    print(f"sequence_coverage={sequence_coverage}/{sequence_target}")
    print(f"anchor_pair_count_pass_ratio={anchor_count_pass_ratio:.6f}")
    print(f"strong_bad_pair_rows={len(bad_pairs_with_positive)}")
    print(f"feature_q_availability_ratio={q_ratio:.6f}")
    print(f"feature_k_availability_ratio={k_ratio:.6f}")


def summary_value_for(check: str, summary: Mapping[str, Any]) -> Any:
    mapping = {
        "adjacent_labelled_rows_ge_24": summary["adjacent_labelled_rows"],
        "sequence_coverage_pass": f"{summary['sequence_coverage']}/{summary['sequence_target']}",
        "anchor_pair_count_pass_ratio_ge_60pct": summary["anchor_pair_count_pass_ratio"],
        "strong_bad_pair_rows_ge_5": summary["strong_bad_pair_rows"],
        "zero_conf_positive_ratio_le_10pct": summary["zero_conf_positive_ratio"],
        "feature_q_availability_ge_90pct": summary["feature_q_availability_ratio"],
        "feature_k_availability_ge_90pct": summary["feature_k_availability_ratio"],
    }
    return mapping.get(check, "")


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Phase1 Anchor Pair Universe Audit",
        "",
        f"- Phase1 gate pass: `{summary['phase1_gate_pass']}`",
        f"- Runtime action allowed: `{summary['runtime_action_allowed']}`",
        f"- TTT allowed: `{summary['ttt_allowed']}`",
        f"- Next action: `{summary['next_action']}`",
        "",
        "## Gate Checks",
        "",
        "| check | pass | value |",
        "|---|---:|---:|",
    ]
    for key, passed in summary["checks"].items():
        lines.append(f"| {key} | {passed} | {summary_value_for(key, summary)} |")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- row_count: {summary['row_count']}",
            f"- pair_row_count: {summary['pair_row_count']}",
            f"- positive_anchor_rows: {summary['positive_anchor_rows']}",
            f"- labelled_nonstress_anchor_rows: {summary['labelled_nonstress_anchor_rows']}",
            "",
            "## Notes",
            "",
        ]
    )
    for note in summary["notes"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
