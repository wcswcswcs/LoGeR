#!/usr/bin/env python3
"""Create a conservative review table for v81S S3 visual panels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS3_swa_visual_confirmation"
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    with (args.root / "visual_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    review: list[dict[str, Any]] = []
    for row in manifest:
        true_route = str(row.get("has_actual_route_mask", "")).lower() == "true"
        review.append(
            {
                "seq": row.get("seq", ""),
                "prev_chunk": row.get("prev_chunk", ""),
                "curr_chunk": row.get("curr_chunk", ""),
                "case_type": row.get("case_type", ""),
                "visual_file": row.get("visual_file", ""),
                "scale_handoff_status": "ambiguous_without_true_swa_action_route" if not true_route else "needs_human_review",
                "stable_carry_status": "missing_true_swa_carry_evidence" if not true_route else "needs_human_review",
                "harm_carry_status": "missing_true_swa_carry_evidence" if not true_route else "needs_human_review",
                "artifact_quality_risk": row.get("artifact_quality_risk", ""),
                "review_note": "Q/K/V and overlap residual panels exist; route panel is overlap sample proxy, not runtime SWA carried/rejected mask.",
            }
        )
    _write_csv(args.root / "visual_review.csv", review)
    insight = {
        "schema": "acl2_v81s_swa_visual_review_v1",
        "rows": len(review),
        "scale_handoff_status_counts": {
            status: sum(1 for row in review if row["scale_handoff_status"] == status)
            for status in sorted({row["scale_handoff_status"] for row in review})
        },
        "conclusion": "Visual/QKV artifacts are present, but true runtime SWA action route masks are missing; S3 cannot confirm SWA carry behavior yet.",
    }
    (args.root / "visual_insight.md").write_text(
        "# v81S S3 Visual Insight\n\n"
        f"rows: {insight['rows']}\n\n"
        f"conclusion: {insight['conclusion']}\n",
        encoding="utf-8",
    )
    (args.root / "visual_review_summary.json").write_text(json.dumps(insight, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(insight, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
