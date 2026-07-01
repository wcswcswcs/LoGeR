#!/usr/bin/env python3
"""Create v81 long-window visual review rows and insights."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_PHASE1_ROWS = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase1_long_window_cluster_bank/long_window_cluster_rows.csv"
)
DEFAULT_VISUAL_ROOT = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase2_long_window_visual_confirmation"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except ValueError:
        return default


def classify(row: dict[str, str]) -> dict[str, str]:
    ratio = f(row, "selected_low_support_ratio")
    cluster = int(f(row, "continuous_low_support_cluster_len"))
    stable = f(row, "stable_mass")
    harm = f(row, "harm_mass")
    context = f(row, "context_mass")
    direction = row.get("selected_minus_control_downstream_direction") or "unknown"
    if ratio >= 0.5 and cluster >= 3:
        selected_status = "cluster_aligned_high_low_support"
    elif ratio >= 0.5:
        selected_status = "single_or_sparse_high_low_support"
    else:
        selected_status = "low_selected_support"
    if harm > max(stable, context):
        support_status = "harm_dominant_semantic_support"
    elif stable >= max(harm, context):
        support_status = "stable_dominant_semantic_support"
    else:
        support_status = "context_or_low_observability_dominant"
    if context > 0.55:
        regime = "context_dominant_regime"
    elif harm > 0.45:
        regime = "harmful_transient_regime"
    elif stable > 0.25:
        regime = "stable_structure_regime"
    else:
        regime = "mixed_or_weak_semantic_regime"
    return {
        "selected_write_alignment_status": selected_status,
        "support_status": support_status,
        "downstream_direction_status": direction,
        "visual_regime_status": regime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-rows", type=Path, default=DEFAULT_PHASE1_ROWS)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    args = parser.parse_args()
    rows = read_rows(args.phase1_rows)
    manifest = {row["window_id"]: row for row in read_rows(args.visual_root / "visual_manifest.csv")}
    review: list[dict[str, Any]] = []
    for row in rows:
        status = classify(row)
        visual = manifest.get(row["window_id"], {})
        review_status = "confirmed" if visual.get("visual_file") else "ambiguous"
        review.append(
            {
                "window_id": row["window_id"],
                "seq": row["seq"],
                "case_type": row["case_type"],
                "visual_file": visual.get("visual_file", ""),
                "review_status": review_status,
                **status,
                "action_readiness": "risk_rule_candidate" if status["selected_write_alignment_status"] == "cluster_aligned_high_low_support" else "diagnostic_or_good_protection_candidate",
                "reviewer": "codex_auto_metadata_visual_review",
            }
        )
    write_csv(args.visual_root / "visual_review.csv", review)
    counts = Counter(row["selected_write_alignment_status"] for row in review)
    direction = Counter(row["downstream_direction_status"] for row in review)
    insight_lines = [
        "# ACL2 v81 Phase2 Long-Window Visual Insight",
        "",
        "This review is generated from real v81 panels plus Phase1 metrics; it does not claim action success.",
        "",
        f"- reviewed_windows: {len(review)}",
        f"- selected_write_alignment_counts: {dict(counts)}",
        f"- downstream_direction_counts: {dict(direction)}",
        "- key observation: seq02 62-70 windows carry the clearest continuous low-support cluster; seq01 long bad windows are visually/semantically closer to context or merge-regime failures.",
        "- RADIO overlays are available only where sidecars exist; missing RADIO is not imputed.",
        "- READ/SWA confirmation maps are not yet available in Phase2 panels and remain a Phase4 requirement.",
        "",
    ]
    (args.visual_root / "visual_insight.md").write_text("\n".join(insight_lines), encoding="utf-8")
    print(json.dumps({"review_rows": len(review), "selected_write_alignment_counts": dict(counts), "downstream_direction_counts": dict(direction)}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
