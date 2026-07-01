#!/usr/bin/env python3
"""Create a conservative v82 review table from true-route visual manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase3_swa_true_route_visual_confirmation"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _status(row: dict[str, str]) -> tuple[str, str]:
    true_route = _truthy(row.get("has_actual_route_mask"))
    qkv = _truthy(row.get("has_qkv_maps"))
    diff = _truthy(row.get("actual_vs_random_difference_reviewed"))
    if not true_route:
        return "blocker_missing_runtime_route", "No runtime SWA route map for both source_replace and source_gate."
    if not qkv:
        return "blocker_missing_qkv", "Runtime route exists but Q/K/V panels are incomplete."
    if not diff:
        return "blocker_missing_actual_random_difference", "Actual-vs-random route comparison did not produce a nonzero difference."
    return "reviewed_true_route_qkv_random", "Runtime SWA route map, Q/K/V panel, and actual-vs-random comparison are present."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    manifest = _read_csv(args.root / "visual_manifest.csv")
    review: list[dict[str, Any]] = []
    for row in manifest:
        status, note = _status(row)
        replace_mean = _float(row.get("source_replace_score_mean"))
        gate_mean = _float(row.get("source_gate_score_mean"))
        stable = _float(row.get("stable_overlap_mass"))
        harm = _float(row.get("harm_overlap_mass"))
        if status.startswith("reviewed"):
            if stable is not None and harm is not None and stable >= harm:
                scale_note = "stable_mass_ge_harm_mass_in_pair_bank"
            elif stable is not None and harm is not None:
                scale_note = "harm_mass_gt_stable_mass_in_pair_bank"
            else:
                scale_note = "missing_pair_bank_mass"
            route_note = (
                "source_replace_score_ge_source_gate_score"
                if replace_mean is not None and gate_mean is not None and replace_mean >= gate_mean
                else "source_gate_score_gt_source_replace_score"
                if replace_mean is not None and gate_mean is not None
                else "missing_route_score"
            )
        else:
            scale_note = "not_interpretable_until_artifacts_complete"
            route_note = "not_interpretable_until_artifacts_complete"
        review.append(
            {
                "seq": row.get("seq", ""),
                "prev_chunk": row.get("prev_chunk", ""),
                "curr_chunk": row.get("curr_chunk", ""),
                "case_type": row.get("case_type", ""),
                "base_case_type": row.get("base_case_type", ""),
                "quality_type": row.get("quality_type", ""),
                "review_status": status,
                "scale_handoff_visual_status": scale_note,
                "stable_vs_harm_runtime_route_status": route_note,
                "actual_vs_random_l1": row.get("actual_vs_random_l1", ""),
                "source_replace_score_mean": row.get("source_replace_score_mean", ""),
                "source_gate_score_mean": row.get("source_gate_score_mean", ""),
                "stable_overlap_mass": row.get("stable_overlap_mass", ""),
                "harm_overlap_mass": row.get("harm_overlap_mass", ""),
                "visual_file": row.get("visual_file", ""),
                "qkv_head_layer_panel": row.get("qkv_head_layer_panel", ""),
                "actual_vs_random_panel": row.get("actual_vs_random_panel", ""),
                "confidence_bin_panel": row.get("confidence_bin_panel", ""),
                "review_note": note,
            }
        )
    _write_csv(args.root / "visual_review.csv", review)
    status_counts = Counter(row["review_status"] for row in review)
    base_counts = Counter(row["base_case_type"] for row in review)
    case_counts = Counter(row["case_type"] for row in review)
    reviewed_rows = sum(1 for row in review if row["review_status"].startswith("reviewed"))
    summary = {
        "schema": "acl2_v82_swa_true_route_visual_review_v1",
        "rows": len(review),
        "reviewed_rows": reviewed_rows,
        "review_coverage": reviewed_rows / len(review) if review else 0.0,
        "review_status_counts": dict(status_counts),
        "base_case_counts": dict(base_counts),
        "case_counts": dict(case_counts),
        "conclusion": (
            "Runtime SWA route/QKV/random-difference visual artifacts are complete enough for Phase4 ledger construction."
            if reviewed_rows == len(review) and review
            else "Visual review remains incomplete; do not advance to SWA action or Phase4 ledger until blockers are fixed."
        ),
    }
    (args.root / "visual_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.root / "visual_insight.md").write_text(
        "# v82 Phase3 Visual Insight\n\n"
        f"rows: {summary['rows']}\n\n"
        f"reviewed_rows: {summary['reviewed_rows']}\n\n"
        f"review_coverage: {summary['review_coverage']:.6f}\n\n"
        f"status_counts: {dict(status_counts)}\n\n"
        f"conclusion: {summary['conclusion']}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
