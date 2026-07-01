#!/usr/bin/env python3
"""Audit v80 Phase2 visual case panels and review coverage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


DEFAULT_VISUAL_ROOT = Path(
    "results/kitti01_hmc_v2/"
    "acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase2_case_visual_confirmation"
)


def read_rows(path: Path) -> list[dict[str, Any]]:
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
    except Exception as exc:  # pragma: no cover - diagnostic path
        out["error"] = type(exc).__name__
    return out


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-root", type=Path, default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--min-review-coverage", type=float, default=0.80)
    args = parser.parse_args()

    manifest = read_rows(args.visual_root / "visual_manifest.csv")
    review = read_rows(args.visual_root / "visual_review.csv")
    review_by_case = {row.get("case_id"): row for row in review}
    statuses = {"confirmed", "rejected", "ambiguous"}
    image_failures = []
    by_memory_case_type: dict[str, Counter[str]] = defaultdict(Counter)
    direct_available = 0
    for row in manifest:
        path = Path(str(row.get("visual_file", "")))
        ok = image_ok(path)
        if not ok["exists"] or not ok["nonblank"] or not ok["width"] or not ok["height"]:
            image_failures.append({"case_id": row.get("case_id"), "visual_file": str(path), **ok})
        key = str(row.get("memory_body"))
        by_memory_case_type[key][str(row.get("case_type"))] += 1
        if str(row.get("direct_qkv_ttt_artifact_available", "")).lower() == "true":
            direct_available += 1
    reviewed = [
        row for row in review
        if row.get("case_id") in {m.get("case_id") for m in manifest}
        and row.get("review_status") in statuses
    ]
    review_coverage = len(reviewed) / max(len(manifest), 1)
    per_memory_gate = {
        memory: counts.get("good", 0) >= 12 and counts.get("bad", 0) >= 12
        for memory, counts in by_memory_case_type.items()
    }
    gate_pass = (
        len(manifest) > 0
        and not image_failures
        and review_coverage >= float(args.min_review_coverage)
        and all(per_memory_gate.get(memory, False) for memory in ("short", "mid", "long"))
    )
    audit = {
        "schema": "acl2_v80tf_phase2_visual_integrity_audit_v1",
        "visual_root": str(args.visual_root),
        "num_visual_files": len(manifest),
        "num_review_rows": len(review),
        "valid_review_rows": len(reviewed),
        "review_coverage": review_coverage,
        "image_failure_count": len(image_failures),
        "image_failures": image_failures[:20],
        "panel_counts_by_memory": {memory: dict(counts) for memory, counts in by_memory_case_type.items()},
        "per_memory_good_bad_gate": per_memory_gate,
        "direct_qkv_ttt_artifact_available_count": direct_available,
        "action_ready_gate_pass": direct_available == len(manifest),
        "gate_pass": gate_pass,
        "review_status_counts": dict(Counter(row.get("review_status") for row in review)),
        "notes": [
            "gate_pass validates panel existence/nonblank/review coverage/good-bad counts.",
            "action_ready_gate_pass is false when direct QKV/SWA/TTT hook artifacts are missing.",
        ],
    }
    write_json(args.visual_root / "visual_integrity_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
