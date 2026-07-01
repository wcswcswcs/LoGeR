#!/usr/bin/env python3
"""Oracle feasibility audit for the ACL2 v100 selector/L3 gate.

This script enumerates binary selectors over the existing 28-case bank.  It is
diagnostic-only: selectors are constructed from labels/L3 after the fact and
must never be treated as runtime rules.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
OUT = ROOT / "trackD4_read_current_support_provider"
PROVENANCE_ROWS = OUT / "label_l3_hygiene_provenance_rows.csv"
CASE_ROWS = OUT / "missed_positive_l3_case_rows.csv"

MIN_RECALL = 0.65
MAX_FPR = 0.25
MIN_CORR = 0.50
MIN_SEQ_COVERAGE = 4
MAX_SELECTED_POSITIVE_SEQ_FRAC = 0.60


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            clean = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
                for key, value in row.items()
            }
            writer.writerow(clean)


def pearson_selected(rows: list[dict[str, Any]], selected: set[str]) -> float:
    xs = [1.0 if row["case_id"] in selected else 0.0 for row in rows]
    ys = [f(row.get("L3_handoff_transfer_penalty_proxy")) for row in rows]
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-12 or vy <= 1.0e-12:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def metric_row(rows: list[dict[str, Any]], selected: set[str], view: str, enumerated_index: int) -> dict[str, Any]:
    positives = {row["case_id"] for row in rows if row.get("case_label") != "good"}
    negatives = {row["case_id"] for row in rows if row.get("case_label") == "good"}
    selected_pos = selected & positives
    selected_neg = selected & negatives
    missed = positives - selected
    recall = len(selected_pos) / len(positives) if positives else math.nan
    fpr = len(selected_neg) / len(negatives) if negatives else math.nan
    seq_counts = Counter(str(row.get("seq", "")) for row in rows if row["case_id"] in selected_pos)
    seq_max_frac = max(seq_counts.values()) / len(selected_pos) if selected_pos and seq_counts else math.nan
    corr = pearson_selected(rows, selected)
    hygiene_fp = [
        row["case_id"]
        for row in rows
        if row["case_id"] in selected_neg and b(row.get("v98_hygiene_excluded_good_control"))
    ]
    return {
        "view": view,
        "enumerated_index": enumerated_index,
        "case_count": len(rows),
        "selected_case_count": len(selected),
        "bad_recall": recall,
        "good_FPR": fpr,
        "balanced_accuracy": ((recall + (1.0 - fpr)) / 2.0) if math.isfinite(recall) and math.isfinite(fpr) else math.nan,
        "selector_corr_L3": corr,
        "selector_abs_corr_L3": abs(corr) if math.isfinite(corr) else math.nan,
        "selected_positive_sequence_coverage": len(seq_counts),
        "selected_positive_sequence_max_frac": seq_max_frac,
        "selected_positive_sequence_counts": dict(seq_counts),
        "false_positive_cases": ";".join(sorted(selected_neg)),
        "hygiene_excluded_false_positive_cases": ";".join(sorted(hygiene_fp)),
        "missed_positive_cases": ";".join(sorted(missed)),
        "selected_cases": ";".join(sorted(selected)),
        "corr_gate_like": math.isfinite(corr) and corr >= MIN_CORR,
    }


def enumerate_view(rows: list[dict[str, Any]], view: str, *, forbid_hygiene_excluded_good: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = sorted(rows, key=lambda row: row["case_id"])
    positives = [row["case_id"] for row in rows if row.get("case_label") != "good"]
    negatives = [row["case_id"] for row in rows if row.get("case_label") == "good"]
    seq_by_case = {row["case_id"]: str(row.get("seq", "")) for row in rows}
    hygiene_excluded_good = {
        row["case_id"]
        for row in rows
        if row.get("case_label") == "good" and b(row.get("v98_hygiene_excluded_good_control"))
    }
    min_pos = math.ceil(MIN_RECALL * len(positives))
    max_neg = math.floor(MAX_FPR * len(negatives))

    top: list[dict[str, Any]] = []
    enumerated = 0
    corr_gate_count = 0
    hygiene_unsafe_corr_gate_count = 0
    for pos_count in range(min_pos, len(positives) + 1):
        for pos_sel_tuple in itertools.combinations(positives, pos_count):
            pos_sel = set(pos_sel_tuple)
            seq_counts = Counter(seq_by_case[case] for case in pos_sel)
            if len(seq_counts) < MIN_SEQ_COVERAGE:
                continue
            if max(seq_counts.values()) / len(pos_sel) > MAX_SELECTED_POSITIVE_SEQ_FRAC:
                continue
            for neg_count in range(0, max_neg + 1):
                for neg_sel_tuple in itertools.combinations(negatives, neg_count):
                    neg_sel = set(neg_sel_tuple)
                    if forbid_hygiene_excluded_good and (neg_sel & hygiene_excluded_good):
                        continue
                    selected = pos_sel | neg_sel
                    row = metric_row(rows, selected, view, enumerated)
                    enumerated += 1
                    if b(row["corr_gate_like"]):
                        corr_gate_count += 1
                        if neg_sel & hygiene_excluded_good:
                            hygiene_unsafe_corr_gate_count += 1
                    top.append(row)
                    top.sort(key=lambda item: (f(item.get("selector_corr_L3")), f(item.get("balanced_accuracy"))), reverse=True)
                    if len(top) > 20:
                        top = top[:20]

    best = top[0] if top else {}
    summary = {
        "view": view,
        "case_count": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "min_recall": MIN_RECALL,
        "max_fpr": MAX_FPR,
        "min_corr": MIN_CORR,
        "min_seq_coverage": MIN_SEQ_COVERAGE,
        "max_selected_positive_seq_frac": MAX_SELECTED_POSITIVE_SEQ_FRAC,
        "forbid_hygiene_excluded_good": forbid_hygiene_excluded_good,
        "enumerated_selector_count": enumerated,
        "corr_gate_like_count": corr_gate_count,
        "hygiene_unsafe_corr_gate_like_count": hygiene_unsafe_corr_gate_count,
        "best": best,
    }
    return summary, top


def load_case_rows() -> list[dict[str, Any]]:
    rows = read_rows(PROVENANCE_ROWS)
    if rows:
        return rows
    base = read_rows(CASE_ROWS)
    for row in base:
        row.setdefault("v98_hygiene_excluded_good_control", False)
    return base


def main() -> None:
    rows = load_case_rows()
    if not rows:
        raise SystemExit(f"missing input rows: {PROVENANCE_ROWS} or {CASE_ROWS}")

    filtered_rows = [
        row
        for row in rows
        if not (row.get("case_label") == "good" and b(row.get("v98_hygiene_excluded_good_control")))
    ]
    summaries: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for view_rows, view, forbid in [
        (rows, "full_28_no_hygiene_safety_constraint", False),
        (rows, "full_28_forbid_v98_hygiene_excluded_good_fp", True),
        (filtered_rows, "diagnostic_v98_hygiene_filtered_universe", False),
    ]:
        summary, top = enumerate_view(view_rows, view, forbid_hygiene_excluded_good=forbid)
        summaries.append(summary)
        top_rows.extend(top)

    payload = {
        "schema": "acl2_v100_oracle_l3_gate_feasibility_v1",
        "status": "complete",
        "note": "Oracle binary selector enumeration using labels/L3 only; diagnostic-only and not runtime-action evidence.",
        "input_rows": str(PROVENANCE_ROWS if PROVENANCE_ROWS.is_file() else CASE_ROWS),
        "views": summaries,
        "conclusion": (
            "The full 28-case L3 corr gate is oracle-feasible only when v98 hygiene-excluded high-L3 good controls may be selected. "
            "Forbidding those false positives on the full 28-case universe yields zero corr>=0.50 oracle selectors. "
            "The 26-case hygiene-filtered diagnostic universe can reach corr>=0.50, but only by selecting remaining good-high-L3 controls and missing low-L3 non-good cases, so it is not a full-gate success."
        ),
    }
    report_lines = [
        "# Oracle L3 Gate Feasibility Audit",
        "",
        "This audit is diagnostic-only. It enumerates label/L3 oracle selectors and must not be used as a runtime rule.",
        "",
    ]
    for summary in summaries:
        best = summary.get("best", {})
        report_lines.extend(
            [
                f"## {summary['view']}",
                "",
                f"- case_count: {summary['case_count']}",
                f"- enumerated_selector_count: {summary['enumerated_selector_count']}",
                f"- corr_gate_like_count: {summary['corr_gate_like_count']}",
                f"- hygiene_unsafe_corr_gate_like_count: {summary['hygiene_unsafe_corr_gate_like_count']}",
                f"- best_corr: {best.get('selector_corr_L3', '')}",
                f"- best_BA: {best.get('balanced_accuracy', '')}",
                f"- best_recall: {best.get('bad_recall', '')}",
                f"- best_FPR: {best.get('good_FPR', '')}",
                f"- best_false_positive_cases: {best.get('false_positive_cases', '') or 'none'}",
                f"- best_hygiene_excluded_false_positive_cases: {best.get('hygiene_excluded_false_positive_cases', '') or 'none'}",
                f"- best_missed_positive_cases: {best.get('missed_positive_cases', '') or 'none'}",
                "",
            ]
        )
    report_lines.extend(["## Conclusion", "", payload["conclusion"], ""])

    write_json(OUT / "oracle_l3_gate_feasibility_summary.json", payload)
    write_rows(OUT / "oracle_l3_gate_feasibility_top_rows.csv", top_rows)
    write_text(OUT / "oracle_l3_gate_feasibility_report.md", "\n".join(report_lines))


if __name__ == "__main__":
    main()
