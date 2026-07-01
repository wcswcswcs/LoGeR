#!/usr/bin/env python3
"""Build ACL2 v84 Phase2 Memory Ruler ledger from Phase1 candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_PHASE1 = Path("results/acl2_v84tf_memory_ruler_audit/phase1_ruler_candidate_universe")
DEFAULT_OUT_DIR = Path("results/acl2_v84tf_memory_ruler_audit/phase2_memory_ruler_ledger")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--selection-quantile", type=float, default=0.60)
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
    if isinstance(value, (list, tuple, dict)):
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


def key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row.get("seq", "")), str(row.get("prev_chunk", "")), str(row.get("curr_chunk", ""))


def finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        f = safe_float(value)
        if f is not None:
            out.append(f)
    return out


def mean_or_none(values: Iterable[Any]) -> float | None:
    vals = finite(values)
    return sum(vals) / len(vals) if vals else None


def median_or_none(values: Iterable[Any]) -> float | None:
    vals = finite(values)
    return median(vals) if vals else None


def sum_or_zero(values: Iterable[Any]) -> float:
    return sum(finite(values))


def minmax_scores(rows: list[dict[str, Any]], fields: Sequence[str]) -> dict[tuple[str, str, str], float]:
    field_values: dict[str, list[float]] = {field: [] for field in fields}
    for row in rows:
        for field in fields:
            value = safe_float(row.get(field))
            if value is not None:
                field_values[field].append(value)
    bounds: dict[str, tuple[float, float]] = {}
    for field, values in field_values.items():
        if values:
            lo, hi = min(values), max(values)
            bounds[field] = (lo, hi)
    out: dict[tuple[str, str, str], float] = {}
    for row in rows:
        vals: list[float] = []
        for field in fields:
            value = safe_float(row.get(field))
            if value is None or field not in bounds:
                continue
            lo, hi = bounds[field]
            if math.isclose(lo, hi):
                continue
            vals.append((value - lo) / (hi - lo))
        if vals:
            out[key(row)] = sum(vals) / len(vals)
    return out


def quantile(values: Sequence[float], q: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[int(lo)]
    frac = pos - lo
    return vals[int(lo)] * (1.0 - frac) + vals[int(hi)] * frac


def build_rows(tokens: list[dict[str, str]], summaries: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in tokens:
        grouped[key(row)].append(row)
    jswa_scores = minmax_scores(
        [dict(row) for row in summaries],
        ["future_after_overlap", "boundary_jump", "overlap_scale_residual", "prev_to_curr_scale_jump"],
    )
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        group = grouped.get(key(summary), [])
        anchor = [row for row in group if row.get("ruler_role") == "RULER_ANCHOR"]
        risk = [row for row in group if row.get("ruler_role") == "RULER_RISK"]
        context = [row for row in group if row.get("ruler_role") == "RULER_CONTEXT"]
        degenerate = [row for row in group if row.get("ruler_role") == "RULER_DEGENERATE"]
        denominator_terms: list[float] = []
        for row in group:
            read = safe_float(row.get("READ_usage"))
            swa = safe_float(row.get("SWA_usage"))
            if read is not None and swa is not None:
                denominator_terms.append(math.sqrt(max(read, 0.0) * max(swa, 0.0)))
        denom = sum(denominator_terms)
        anchor_mass = sum_or_zero(row.get("ruler_anchor_score") for row in anchor)
        rpi = anchor_mass / denom if denom > 1e-12 else None
        swa_sum = sum_or_zero(row.get("SWA_usage") for row in group)
        rci = (
            sum((safe_float(row.get("risk_score")) or 0.0) * (safe_float(row.get("SWA_usage")) or 0.0) for row in group) / swa_sum
            if swa_sum > 1e-12
            else None
        )
        dr_med = safe_float(summary.get("pairwise_log_distance_ratio_median"))
        dr_mad = safe_float(summary.get("pairwise_log_distance_ratio_mad"))
        anchor_count = len(anchor)
        rcx = ((dr_mad or 0.0) + abs(dr_med or 0.0)) * math.log1p(anchor_count) if dr_med is not None and dr_mad is not None else None
        median_geo = median_or_none(row.get("geometry_leverage") for row in group)
        median_ov = median_or_none(row.get("overlap_consistency") for row in group)
        zero_conf = safe_float(summary.get("zero_conf_ratio")) or 0.0
        roi = math.log1p(anchor_count) * (median_geo or 0.0) * (median_ov or 0.0) * (1.0 - zero_conf) if median_geo is not None and median_ov is not None else None
        read_ruler_mass = sum_or_zero(row.get("READ_usage") for row in anchor)
        swa_ruler_mass = sum_or_zero(row.get("SWA_usage") for row in anchor)
        read_risk_mass = sum_or_zero(row.get("READ_usage") for row in risk)
        swa_risk_mass = sum_or_zero(row.get("SWA_usage") for row in risk)
        out = {
            "seq": summary.get("seq"),
            "prev_chunk": summary.get("prev_chunk"),
            "curr_chunk": summary.get("curr_chunk"),
            "case_type": summary.get("case_type"),
            "quality_source": summary.get("quality_source"),
            "bad_good_label_source": "v82_swa_pair_bank_v2",
            "base_case_type": summary.get("base_case_type"),
            "future_after_overlap": summary.get("future_after_overlap"),
            "boundary_jump": summary.get("boundary_jump"),
            "raw_overlap_residual": summary.get("raw_overlap_residual"),
            "overlap_scale_residual": summary.get("overlap_scale_residual"),
            "prev_to_curr_scale_jump": summary.get("prev_to_curr_scale_jump"),
            "scale_cv": "",
            "J_SWA": jswa_scores.get(key(summary)),
            "ruler_anchor_count": anchor_count,
            "ruler_anchor_mass": anchor_mass,
            "ruler_anchor_ratio": anchor_count / len(group) if group else None,
            "ruler_context_mass": sum_or_zero(row.get("ruler_anchor_score") for row in context),
            "ruler_risk_mass": sum_or_zero(row.get("risk_score") for row in risk),
            "ruler_degenerate_mass": sum_or_zero(row.get("geometry_leverage") for row in degenerate),
            "zero_conf_ratio": zero_conf,
            "low_observability_flag": bool((roi or 0.0) <= 1e-8),
            "READ_ruler_mass": read_ruler_mass,
            "READ_risk_mass": read_risk_mass,
            "SWA_ruler_mass": swa_ruler_mass,
            "SWA_risk_mass": swa_risk_mass,
            "per_head_ruler_mass_mean": "",
            "per_head_ruler_mass_max": "",
            "best_head_id": "",
            "best_layer_id": "",
            "RPI": rpi,
            "RCI": rci,
            "RCX": rcx,
            "ROI": roi,
            "pairwise_log_distance_ratio_median": dr_med,
            "pairwise_log_distance_ratio_mad": dr_mad,
            "local_shape_residual": mean_or_none(row.get("overlap_residual") for row in group),
            "ruler_contradiction_index": rcx,
            "ruler_observability_index": roi,
            "best_head_RPI": "",
            "best_head_RCI": "",
            "per_head_RPI_entropy": "",
            "semantic_shuffle_available": False,
            "same_mass_random_available": False,
            "same_head_random_available": False,
            "token_rows": len(group),
            "read_usage_available": summary.get("read_usage_available"),
            "swa_usage_available": summary.get("swa_usage_available"),
            "swa_usage_source": summary.get("swa_usage_source"),
        }
        rows.append(out)
    return rows


def threshold_support(rows: list[dict[str, Any]], field: str, high_is_positive: bool = True, q: float = 0.60) -> dict[str, Any]:
    values = finite(row.get(field) for row in rows)
    threshold = quantile(values, q if high_is_positive else 1.0 - q)
    if threshold is None:
        return {"score": field, "available": 0, "positive_count": 0, "threshold": None}
    positives = []
    for row in rows:
        value = safe_float(row.get(field))
        if value is None:
            continue
        positive = value >= threshold if high_is_positive else value <= threshold
        if positive:
            positives.append(row)
    return {
        "score": field,
        "direction": "high" if high_is_positive else "low",
        "available": len(values),
        "threshold": threshold,
        "positive_count": len(positives),
        "positive_bad_count": sum(1 for row in positives if row.get("base_case_type") == "bad"),
        "positive_good_count": sum(1 for row in positives if row.get("base_case_type") != "bad"),
    }


def make_report(summary: dict[str, Any], support_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase2 Memory Ruler Ledger Report",
        "",
        f"- Phase2 gate pass: `{summary['phase2_gate_pass']}`",
        f"- Rows: {summary['rows']}",
        f"- Bad rows: {summary['bad_rows']}",
        f"- Good/false-positive rows: {summary['good_or_false_positive_rows']}",
        f"- Sequence coverage: {summary['sequence_coverage']}",
        f"- RPI/RCI/RCX/ROI high-quality availability ratio: {summary['score_available_high_quality_ratio']}",
        "",
        "## Positive Support",
        "",
        "| score | direction | available | threshold | positive_count | positive_bad | positive_good |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in support_rows:
        lines.append(
            f"| {row.get('score')} | {row.get('direction')} | {row.get('available')} | {serialize(row.get('threshold'))} | "
            f"{row.get('positive_count')} | {row.get('positive_bad_count')} | {row.get('positive_good_count')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Phase2 is a support/ledger gate, not a sufficiency claim.",
            "- True per-head route mass and same-head random controls remain unavailable at this phase.",
            "- Sparse `RULER_ANCHOR` support should be interpreted cautiously in Phase3.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    phase1 = args.phase1_dir
    tokens = read_csv(phase1 / "ruler_candidate_tokens.csv")
    summaries = read_csv(phase1 / "ruler_candidate_pair_summary.csv")
    rows = build_rows(tokens, summaries)
    high_quality = [row for row in rows if str(row.get("case_type", "")).endswith("highconf") or str(row.get("quality_source", "")) == "default"]
    score_fields = ["RPI", "RCI", "RCX", "ROI"]
    available_hq = [
        row
        for row in high_quality
        if all(safe_float(row.get(field)) is not None for field in score_fields)
    ]
    support = [
        threshold_support(rows, "RPI", high_is_positive=False, q=args.selection_quantile),
        threshold_support(rows, "RCI", high_is_positive=True, q=args.selection_quantile),
        threshold_support(rows, "RCX", high_is_positive=True, q=args.selection_quantile),
        threshold_support(rows, "ROI", high_is_positive=False, q=args.selection_quantile),
    ]
    bad_rows = [row for row in rows if row.get("base_case_type") == "bad"]
    good_rows = [row for row in rows if row.get("base_case_type") != "bad"]
    seqs = sorted({str(row.get("seq")) for row in rows if row.get("seq")})
    min_positive = len(bad_rows) * 0.60
    support_gate = any((row.get("positive_count") or 0) >= min_positive for row in support)
    availability_ratio = len(available_hq) / max(len(high_quality), 1)
    phase2_gate_pass = (
        len(rows) >= 24
        and len(bad_rows) >= 12
        and len(good_rows) >= 12
        and len(seqs) >= 3
        and availability_ratio >= 0.80
        and support_gate
    )
    out_dir = args.out_dir
    write_csv(out_dir / "memory_ruler_ledger.csv", rows)
    write_csv(
        out_dir / "memory_ruler_score_by_pair.csv",
        [
            {
                "seq": row.get("seq"),
                "prev_chunk": row.get("prev_chunk"),
                "curr_chunk": row.get("curr_chunk"),
                "case_type": row.get("case_type"),
                "base_case_type": row.get("base_case_type"),
                "RPI": row.get("RPI"),
                "RCI": row.get("RCI"),
                "RCX": row.get("RCX"),
                "ROI": row.get("ROI"),
                "J_SWA": row.get("J_SWA"),
            }
            for row in rows
        ],
    )
    write_csv(out_dir / "memory_ruler_score_by_head.csv", [{"status": "unavailable", "reason": "per-head route mass not available in Phase1 PCA proxy"}])
    role_rows: list[dict[str, Any]] = []
    for role in sorted({row.get("ruler_role", "") for row in tokens}):
        role_tokens = [row for row in tokens if row.get("ruler_role") == role]
        role_rows.append(
            {
                "ruler_role": role,
                "token_rows": len(role_tokens),
                "mean_anchor_score": mean_or_none(row.get("ruler_anchor_score") for row in role_tokens),
                "mean_risk_score": mean_or_none(row.get("risk_score") for row in role_tokens),
                "mean_READ_usage": mean_or_none(row.get("READ_usage") for row in role_tokens),
                "mean_SWA_usage": mean_or_none(row.get("SWA_usage") for row in role_tokens),
            }
        )
    write_csv(out_dir / "memory_ruler_score_by_semantic_role.csv", role_rows)
    write_csv(
        out_dir / "ruler_distance_ratio_histograms.csv",
        [
            {
                "seq": row.get("seq"),
                "prev_chunk": row.get("prev_chunk"),
                "curr_chunk": row.get("curr_chunk"),
                "pairwise_log_distance_ratio_median": row.get("pairwise_log_distance_ratio_median"),
                "pairwise_log_distance_ratio_mad": row.get("pairwise_log_distance_ratio_mad"),
                "pairwise_distance_ratio_count": next(
                    (s.get("pairwise_distance_ratio_count") for s in summaries if key(s) == key(row)),
                    "",
                ),
            }
            for row in rows
        ],
    )
    summary = {
        "schema": "acl2_v84_phase2_ledger_summary_v1",
        "phase2_gate_pass": phase2_gate_pass,
        "rows": len(rows),
        "bad_rows": len(bad_rows),
        "good_or_false_positive_rows": len(good_rows),
        "sequence_coverage": seqs,
        "sequence_coverage_count": len(seqs),
        "score_available_high_quality_rows": len(available_hq),
        "high_quality_rows": len(high_quality),
        "score_available_high_quality_ratio": availability_ratio,
        "positive_support_min": min_positive,
        "positive_support_gate_pass": support_gate,
        "positive_support_rows": support,
        "ruler_anchor_pair_count": sum(1 for row in rows if (safe_float(row.get("ruler_anchor_count")) or 0.0) > 0),
        "ruler_anchor_token_count": sum(1 for row in tokens if row.get("ruler_role") == "RULER_ANCHOR"),
        "notes": [
            "Phase2 passes support if any score has >=60% of bad_count selected rows by fixed quantile support.",
            "RPI low, RCI high, RCX high, and ROI low are tested as possible positive support directions.",
        ],
    }
    write_json(out_dir / "phase2_ledger_summary.json", summary)
    (out_dir / "phase2_ledger_report.md").write_text(make_report(summary, support), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "phase2_gate_pass": phase2_gate_pass, "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

