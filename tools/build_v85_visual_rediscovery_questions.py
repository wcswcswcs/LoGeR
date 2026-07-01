#!/usr/bin/env python3
"""Build Phase12 visual rediscovery question scaffold for v85."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_ROOT = Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "phase12_visual_rediscovery")
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
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    by_pair = read_csv(args.root / "phase1_anchor_pair_universe/anchor_pair_by_seq_chunk.csv")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for subdir in [
        "anchor_pair_failure_panels",
        "alignment_failure_panels",
        "scale_relevance_failure_panels",
        "route_failure_panels",
        "merge_boundary_failure_panels",
    ]:
        (args.out_dir / subdir).mkdir(parents=True, exist_ok=True)

    question_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for row in by_pair:
        seq = row["seq"]
        prev_chunk = row["prev_chunk"]
        curr_chunk = row["curr_chunk"]
        case_label = row["case_label"]
        strong = int(float(row.get("strong_bootstrap_count") or 0)) + int(float(row.get("strong_mature_count") or 0))
        panel_id = f"seq{seq}_chunk{int(float(prev_chunk)):03d}_{int(float(curr_chunk)):03d}_{case_label}"
        if case_label == "bad" and strong == 0:
            question = "Did this labelled bad pair lack reliable anchor pairs, or were candidates broad/risk/context?"
            category = "anchor_pair_failure"
        elif case_label == "bad":
            question = "Is the single strong bootstrap anchor structural and repeatable enough to support Phase3?"
            category = "anchor_pair_failure"
        elif case_label == "good" and strong > 0:
            question = "Would positive anchors on this good pair create a false-positive alignment/action risk?"
            category = "good_case_protection"
        elif seq == "01" or row.get("stress_seq01_count") not in {"", "0"}:
            question = "Are seq01/minconf stress rows excluded from positive evidence?"
            category = "seq01_stress_exclusion"
        else:
            continue
        question_rows.append(
            {
                "seq": seq,
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "case_label": case_label,
                "quality_label": row.get("quality_label"),
                "question_category": category,
                "visual_question": question,
                "anchor_pair_count": row.get("anchor_pair_count"),
                "positive_anchor_count": row.get("positive_anchor_count"),
                "strong_bootstrap_count": row.get("strong_bootstrap_count"),
                "risk_count": row.get("risk_count"),
                "context_degenerate_count": row.get("context_degenerate_count"),
                "stress_seq01_count": row.get("stress_seq01_count"),
                "panel_id": panel_id,
                "panel_status": "not_generated",
            }
        )
        manifest_rows.append(
            {
                "panel_id": panel_id,
                "panel_group": category,
                "expected_path": f"anchor_pair_failure_panels/{panel_id}.png",
                "exists": False,
                "non_empty": False,
                "status": "pending_generation",
            }
        )
        review_rows.append(
            {
                "panel_id": panel_id,
                "review_status": "pending",
                "confirmed_or_rejected": "",
                "review_note": "visual panel not generated yet",
            }
        )

    integrity = {
        "phase": "Phase12_visual_rediscovery",
        "visual_audit_gate_pass": False,
        "question_rows": len(question_rows),
        "manifest_rows": len(manifest_rows),
        "visual_files_exist": False,
        "non_empty_image_check_pass": False,
        "manifest_rows_complete": False,
        "review_coverage": 0.0,
        "visual_insight_present": True,
        "blocker": "visual_panels_not_generated_yet",
        "note": "This scaffold is intentionally not a visual pass; it lists required visual questions for final No-Go closure.",
    }
    write_csv(args.out_dir / "failed_case_to_visual_question.csv", question_rows)
    write_csv(args.out_dir / "visual_manifest.csv", manifest_rows)
    write_csv(args.out_dir / "visual_review.csv", review_rows)
    write_json(args.out_dir / "visual_integrity_audit.json", integrity)
    (args.out_dir / "visual_insight.md").write_text(
        "\n".join(
            [
                "# v85 Phase12 Visual Rediscovery Insight",
                "",
                "Current state: Phase1 failed before alignment or route tests.",
                "",
                "Primary visual question: do labelled bad pairs truly lack reliable structural anchor pairs, or did the row builder miss them?",
                "",
                "Required next action: generate real image panels for the manifest rows and review them before any final No-Go claim.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (args.out_dir / "new_hypothesis_bank.md").write_text(
        "\n".join(
            [
                "# v85 New Hypothesis Bank",
                "",
                "1. Strong labelled-bad support may be intrinsically sparse in the current v82/v84 pair bank.",
                "2. Additional labelled-bad high-confidence pairs may be needed before latent C fitting is meaningful.",
                "3. Broad context/risk candidates dominate many bad pairs; forcing them positive would likely repeat v84 source-mask failure.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"question_rows={len(question_rows)}")
    print("visual_audit_gate_pass=false")
    print("panel_status=not_generated")


if __name__ == "__main__":
    main()
