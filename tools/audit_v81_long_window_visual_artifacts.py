#!/usr/bin/env python3
"""Audit v81 long-window visual artifacts and review coverage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


DEFAULT_VISUAL_ROOT = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v81tf_longwindow_semantic_three_memory_control/"
    "report_final/phase2_long_window_visual_confirmation"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def image_ok(path: Path) -> dict[str, Any]:
    out = {"exists": path.is_file(), "width": None, "height": None, "nonblank": False, "error": ""}
    if not path.is_file():
        out["error"] = "missing_file"
        return out
    try:
        img = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(img)
        out.update({"width": img.width, "height": img.height, "nonblank": max(stat.stddev) > 0.0})
    except Exception as exc:
        out["error"] = type(exc).__name__
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--min-review-coverage", type=float, default=0.80)
    args = parser.parse_args()
    manifest = read_rows(args.visual_root / "visual_manifest.csv")
    review = read_rows(args.visual_root / "visual_review.csv")
    review_by_window = {row.get("window_id"): row for row in review}
    failures = []
    for row in manifest:
        ok = image_ok(Path(row.get("visual_file", "")))
        if not ok["exists"] or not ok["nonblank"]:
            failures.append({"window_id": row.get("window_id"), "visual_file": row.get("visual_file"), **ok})
    reviewed = [
        row for row in review
        if row.get("window_id") in {m.get("window_id") for m in manifest}
        and row.get("review_status") in {"confirmed", "ambiguous", "rejected"}
        and row.get("selected_write_alignment_status")
        and row.get("support_status")
        and row.get("downstream_direction_status")
        and row.get("visual_regime_status")
    ]
    coverage = len(reviewed) / max(len(manifest), 1)
    bad_reviewed = sum(1 for row in reviewed if row.get("case_type") == "bad")
    good_reviewed = sum(1 for row in reviewed if row.get("case_type") in {"good", "false_positive"})
    counts = Counter(row.get("case_type") for row in manifest)
    gate = (
        len(manifest) > 0
        and not failures
        and coverage >= float(args.min_review_coverage)
        and bad_reviewed >= 12
        and good_reviewed >= 12
    )
    audit = {
        "schema": "acl2_v81_phase2_long_window_visual_integrity_v1",
        "visual_root": str(args.visual_root),
        "gate_pass": gate,
        "num_visual_files": len(manifest),
        "image_failure_count": len(failures),
        "image_failures": failures[:20],
        "num_review_rows": len(review),
        "valid_review_rows": len(reviewed),
        "review_coverage": coverage,
        "bad_windows_reviewed": bad_reviewed,
        "good_false_positive_windows_reviewed": good_reviewed,
        "case_type_counts": dict(counts),
        "support_map_rendered_count": sum(str(row.get("support_map_rendered")).lower() == "true" for row in manifest),
        "radio_panel_available_rows": sum(str(row.get("has_radio")).lower() == "true" for row in manifest),
        "missing_review_windows": sorted({m.get("window_id") for m in manifest} - set(review_by_window)),
    }
    out = args.visual_root / "visual_integrity_audit.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
