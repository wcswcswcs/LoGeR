#!/usr/bin/env python3
"""Audit v85 Phase12 visual rediscovery artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_VISUAL_DIR = Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase12_visual_rediscovery")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-dir", type=Path, default=DEFAULT_VISUAL_DIR)
    parser.add_argument("--min-review-coverage", type=float, default=0.80)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def is_png(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    with path.open("rb") as handle:
        return handle.read(8) == b"\x89PNG\r\n\x1a\n"


def main() -> None:
    args = parse_args()
    visual_dir = args.visual_dir
    manifest = read_csv(visual_dir / "visual_manifest.csv")
    review = read_csv(visual_dir / "visual_review.csv")
    review_by_panel = {row["panel_id"]: row for row in review}
    row_checks: list[dict[str, Any]] = []
    for row in manifest:
        path = visual_dir / row["expected_path"]
        exists = path.exists()
        non_empty = exists and path.stat().st_size > 0
        png_ok = is_png(path)
        review_row = review_by_panel.get(row["panel_id"], {})
        reviewed = review_row.get("review_status") == "reviewed"
        confirmed_or_rejected = review_row.get("confirmed_or_rejected") in {"confirmed", "rejected"}
        row_checks.append(
            {
                "panel_id": row["panel_id"],
                "path": str(path),
                "exists": exists,
                "non_empty": non_empty,
                "png_header_ok": png_ok,
                "reviewed": reviewed,
                "confirmed_or_rejected": confirmed_or_rejected,
                "pass": exists and non_empty and png_ok and reviewed and confirmed_or_rejected,
            }
        )
    manifest_rows_complete = bool(row_checks) and all(row["exists"] for row in row_checks)
    non_empty_image_check_pass = bool(row_checks) and all(row["non_empty"] and row["png_header_ok"] for row in row_checks)
    reviewed_count = sum(1 for row in row_checks if row["reviewed"] and row["confirmed_or_rejected"])
    review_coverage = reviewed_count / len(row_checks) if row_checks else 0.0
    visual_insight_present = (visual_dir / "visual_insight.md").exists() and (visual_dir / "visual_insight.md").stat().st_size > 0
    visual_audit_gate_pass = (
        manifest_rows_complete
        and non_empty_image_check_pass
        and review_coverage >= args.min_review_coverage
        and visual_insight_present
    )
    payload = {
        "phase": "Phase12_visual_rediscovery",
        "visual_audit_gate_pass": visual_audit_gate_pass,
        "manifest_rows": len(row_checks),
        "visual_files_exist": manifest_rows_complete,
        "non_empty_image_check_pass": non_empty_image_check_pass,
        "manifest_rows_complete": manifest_rows_complete,
        "review_coverage": review_coverage,
        "reviewed_count": reviewed_count,
        "visual_insight_present": visual_insight_present,
        "blocker": "" if visual_audit_gate_pass else "visual_audit_incomplete",
        "note": "Visual panels are data-derived Phase12 panels, not Phase3/route success evidence.",
    }
    write_csv(visual_dir / "visual_integrity_rows.csv", row_checks)
    write_json(visual_dir / "visual_integrity_audit.json", payload)
    print(f"visual_audit_gate_pass={str(visual_audit_gate_pass).lower()}")
    print(f"manifest_rows={len(row_checks)}")
    print(f"non_empty_image_check_pass={str(non_empty_image_check_pass).lower()}")
    print(f"review_coverage={review_coverage:.6f}")


if __name__ == "__main__":
    main()
