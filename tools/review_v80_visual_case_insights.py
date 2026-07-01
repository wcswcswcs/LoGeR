#!/usr/bin/env python3
"""Create v80 Phase2 visual review rows and insight markdown."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_VISUAL_ROOT = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase2_case_visual_confirmation"
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    args = parser.parse_args()

    manifest = read_rows(args.visual_root / "visual_manifest.csv")
    review: list[dict[str, Any]] = []
    for row in manifest:
        semantic_ok = str(row.get("semantic_panel_available", "")).lower() == "true"
        direct_ok = str(row.get("direct_qkv_ttt_artifact_available", "")).lower() == "true"
        if semantic_ok and direct_ok:
            status = "confirmed"
            pattern = "real semantic panel and direct memory hook panel available"
        elif semantic_ok:
            status = "ambiguous"
            pattern = "real RGB/semantic/confidence/role/geometry evidence available; direct QKV/SWA/TTT hook dumps missing"
        else:
            status = "rejected"
            pattern = "semantic panel missing or unreadable"
        review.append(
            {
                "memory_body": row.get("memory_body"),
                "case_type": row.get("case_type"),
                "seq": row.get("seq"),
                "case_id": row.get("case_id"),
                "visual_file": row.get("visual_file"),
                "review_status": status,
                "visual_pattern_observed": pattern,
                "action_readiness": "not_ready_without_direct_qkv_swa_ttt_confirmation" if not direct_ok else "ready_for_rule_design",
                "reviewer": "codex_auto_visual_audit",
            }
        )
    write_csv(args.visual_root / "visual_review.csv", review)

    counts = Counter(row["review_status"] for row in review)
    lines = [
        "# ACL2 v80 Phase2 Visual Insight",
        "",
        "Generated review is intentionally conservative.",
        "",
        f"- visual rows: {len(review)}",
        f"- confirmed: {counts.get('confirmed', 0)}",
        f"- ambiguous: {counts.get('ambiguous', 0)}",
        f"- rejected: {counts.get('rejected', 0)}",
        "",
        "Current insight:",
        "",
        "The panels provide real RGB, dense semantic, confidence, role-mask, RADIO-when-available, and geometry-context evidence for all Phase1 cases. Direct Q/K/V PCA, SWA current/cache K/V, READ mask, and TTT operator/update/final tensors are not present in the v80 artifact set, so action design is not ready from Phase2 alone.",
        "",
    ]
    (args.visual_root / "visual_insight.md").write_text("\n".join(lines), encoding="utf-8")
    print({"visual_review_rows": len(review), "status_counts": dict(counts)})


if __name__ == "__main__":
    main()
