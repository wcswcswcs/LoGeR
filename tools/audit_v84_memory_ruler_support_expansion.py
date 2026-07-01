#!/usr/bin/env python3
"""Audit ACL2 v84 Phase10 support-expansion candidate outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_PHASE10 = Path("results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion")
DEFAULT_PHASE10_CANDIDATES = Path("results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion_candidates")
DEFAULT_OUT_DIR = Path("results/acl2_v84tf_memory_ruler_audit/phase10_support_expansion_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase10-dir", type=Path, default=DEFAULT_PHASE10)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_PHASE10_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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


def safe_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def pair_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row.get("seq", "")), str(row.get("prev_chunk", "")), str(row.get("curr_chunk", ""))


def counter_to_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda item: str(item[0]))}


def main() -> None:
    args = parse_args()
    plan_path = args.phase10_dir / "support_expansion_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    pair_bank = read_csv(args.phase10_dir / "support_expansion_pair_bank.csv")
    summaries = read_csv(args.candidate_dir / "ruler_candidate_pair_summary.csv")
    tokens = read_csv(args.candidate_dir / "ruler_candidate_tokens.csv")

    summary_by_key = {pair_key(row): row for row in summaries}
    rows: list[dict[str, Any]] = []
    for bank_row in pair_bank:
        summary = summary_by_key.get(pair_key(bank_row), {})
        anchor_count = safe_float(summary.get("ruler_anchor_count")) or 0.0
        risk_count = safe_float(summary.get("ruler_risk_count")) or 0.0
        deg_count = safe_float(summary.get("ruler_degenerate_count")) or 0.0
        token_count = safe_float(summary.get("token_rows")) or 0.0
        zero_conf = safe_float(summary.get("zero_conf_ratio")) or safe_float(bank_row.get("either_zero_ratio")) or 0.0
        distance_count = safe_float(summary.get("pairwise_distance_ratio_count")) or 0.0
        failure_flags: list[str] = []
        if anchor_count <= 0:
            failure_flags.append("ruler_anchor_absent")
        if token_count > 0 and risk_count / token_count >= 0.50:
            failure_flags.append("risk_dominant")
        if token_count > 0 and deg_count / token_count >= 0.50:
            failure_flags.append("degenerate_dominant")
        if zero_conf > 0:
            failure_flags.append("zero_conf_or_lowconf")
        if anchor_count >= 2 and distance_count > 0:
            failure_flags.append("contradiction_observable")
        rows.append(
            {
                "seq": bank_row.get("seq"),
                "prev_chunk": bank_row.get("prev_chunk"),
                "curr_chunk": bank_row.get("curr_chunk"),
                "case_type": bank_row.get("case_type"),
                "base_case_type": bank_row.get("base_case_type"),
                "support_expansion_label_scope": bank_row.get("support_expansion_label_scope"),
                "quality_source": bank_row.get("quality_source"),
                "quality_type": bank_row.get("quality_type"),
                "forbidden_as_stable_evidence": bank_row.get("forbidden_as_stable_evidence"),
                "read_usage_available": summary.get("read_usage_available"),
                "swa_usage_available": summary.get("swa_usage_available"),
                "token_rows": token_count,
                "ruler_anchor_count": anchor_count,
                "ruler_risk_count": risk_count,
                "ruler_degenerate_count": deg_count,
                "ruler_context_count": safe_float(summary.get("ruler_context_count")) or 0.0,
                "ruler_anchor_mass": safe_float(summary.get("ruler_anchor_mass")),
                "zero_conf_ratio": zero_conf,
                "pairwise_distance_ratio_count": distance_count,
                "failure_flags": ";".join(failure_flags),
            }
        )

    base_rows = [row for row in rows if row.get("support_expansion_label_scope") == "labelled_v82_main_pair"]
    support_rows = [row for row in rows if row.get("support_expansion_label_scope") != "labelled_v82_main_pair"]
    labelled_bad = [row for row in base_rows if row.get("base_case_type") == "bad"]
    labelled_good = [row for row in base_rows if row.get("base_case_type") != "bad"]
    labelled_good_positive = [row for row in labelled_good if (safe_float(row.get("ruler_anchor_count")) or 0.0) > 0]
    labelled_bad_positive = [row for row in labelled_bad if (safe_float(row.get("ruler_anchor_count")) or 0.0) > 0]
    default_hq_support = [
        row
        for row in support_rows
        if row.get("quality_source") == "default" and row.get("quality_type") == "high_quality"
    ]

    failure_counts: Counter[str] = Counter()
    for row in rows:
        for flag in str(row.get("failure_flags") or "").split(";"):
            if flag:
                failure_counts[flag] += 1

    role_counts = Counter(row.get("ruler_role", "") for row in tokens)
    seqs = sorted({str(row.get("seq")) for row in rows if row.get("seq")})
    expanded_pair_rows = len(rows)
    base_pair_rows = int(plan.get("base_pair_rows") or len(base_rows) or 0)
    rows_increase_ratio = expanded_pair_rows / max(base_pair_rows, 1)
    good_fpr_anchor_observed = len(labelled_good_positive) / max(len(labelled_good), 1)
    bad_anchor_recall_observed = len(labelled_bad_positive) / max(len(labelled_bad), 1)
    anchor_support_pairs = [row for row in rows if (safe_float(row.get("ruler_anchor_count")) or 0.0) > 0]
    default_hq_anchor_support_pairs = [
        row for row in default_hq_support if (safe_float(row.get("ruler_anchor_count")) or 0.0) > 0
    ]
    semantic_shuffle_specificity_available = False
    expansion_useful_gate_pass = (
        rows_increase_ratio >= 2.0
        and len(seqs) >= 4
        and len(default_hq_anchor_support_pairs) >= 5
        and semantic_shuffle_specificity_available
        and good_fpr_anchor_observed <= 0.25
    )
    summary = {
        "schema": "acl2_v84_phase10_support_expansion_audit_v1",
        "expansion_useful_gate_pass": expansion_useful_gate_pass,
        "base_pair_rows": base_pair_rows,
        "expanded_pair_rows": expanded_pair_rows,
        "rows_increase_ratio": rows_increase_ratio,
        "sequence_coverage": seqs,
        "sequence_coverage_count": len(seqs),
        "rows_by_scope": counter_to_dict(Counter(row.get("support_expansion_label_scope", "") for row in rows)),
        "rows_by_case_type": counter_to_dict(Counter(row.get("case_type", "") for row in rows)),
        "token_role_counts": counter_to_dict(role_counts),
        "anchor_support_pair_count": len(anchor_support_pairs),
        "default_high_quality_support_rows": len(default_hq_support),
        "default_high_quality_support_anchor_pair_count": len(default_hq_anchor_support_pairs),
        "labelled_bad_anchor_recall_observed": bad_anchor_recall_observed,
        "labelled_good_anchor_fpr_observed": good_fpr_anchor_observed,
        "failure_flag_counts": counter_to_dict(failure_counts),
        "semantic_shuffle_specificity_available": semantic_shuffle_specificity_available,
        "good_fpr_gate_for_labelled_anchor_observed": good_fpr_anchor_observed <= 0.25,
        "notes": [
            "This audit expands support and failure attribution only; it does not prove Phase3 sufficiency.",
            "Unlabelled support rows are not counted as good/false-positive labels.",
            "The final useful gate remains false while semantic-shuffle/same-mass controls are unavailable.",
        ],
    }
    lines = [
        "# Phase10 Memory Ruler Support Expansion Audit",
        "",
        f"- Expansion useful gate pass: `{summary['expansion_useful_gate_pass']}`",
        f"- Rows: {expanded_pair_rows} ({rows_increase_ratio:.3f}x base)",
        f"- Sequence coverage: {seqs}",
        f"- Anchor support pairs: {len(anchor_support_pairs)}",
        f"- Default high-quality support anchor pairs: {len(default_hq_anchor_support_pairs)} / {len(default_hq_support)}",
        f"- Labelled bad anchor recall observed: {bad_anchor_recall_observed:.6f}",
        f"- Labelled good anchor FPR observed: {good_fpr_anchor_observed:.6f}",
        f"- Semantic-shuffle specificity available: `{semantic_shuffle_specificity_available}`",
        "",
        "## Interpretation",
        "",
        "Phase10 can repair support volume only if additional observable rows reveal enough anchor/contradiction/risk evidence. "
        "Rows with no bad/good label remain support evidence and are not treated as success cases.",
        "",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "support_expansion_audit_by_pair.csv", rows)
    write_json(args.out_dir / "support_expansion_audit_summary.json", summary)
    (args.out_dir / "support_expansion_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "expansion_useful_gate_pass": expansion_useful_gate_pass,
                "expanded_pair_rows": expanded_pair_rows,
                "anchor_support_pair_count": len(anchor_support_pairs),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
