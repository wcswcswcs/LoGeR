#!/usr/bin/env python3
"""Audit v78 visual artifacts against the plan's provenance/review gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REQUIRED_OVERLAY_COLUMNS = [
    "RGB_overlay_present",
    "semantic_overlay_present",
    "confidence_overlay_present",
    "PCA_overlay_present",
    "D_geo_overlay_present",
    "future_overlay_present",
    "action_mask_overlay_present",
    "same_mass_random_overlay_present",
    "group_stratified_random_overlay_present",
]
ALLOWED_REVIEW_STATUS = {"confirmed", "rejected", "ambiguous", "needs_new_tap"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_stats(path: Path) -> tuple[int, int, float]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    return int(img.width), int(img.height), float(arr.std())


def main() -> None:
    args = parse_args()
    manifest_path = args.visual_root / "visual_artifact_manifest.csv"
    review_path = args.visual_root / "visual_review.csv"
    insight_path = args.visual_root / "visual_insight.md"
    manifest = _read_csv(manifest_path) if manifest_path.exists() else []
    reviews = _read_csv(review_path) if review_path.exists() else []

    invalid_files: list[dict[str, Any]] = []
    sha_values: list[str] = []
    files_exist = True
    sha_ok = True
    dim_ok = True
    nonempty_ok = True
    overlays_ok = True

    for row in manifest:
        path = Path(row.get("visual_file", ""))
        if not path.exists():
            files_exist = False
            invalid_files.append({"visual_file": str(path), "reason": "missing_file"})
            continue
        actual_sha = _sha256(path)
        sha_values.append(actual_sha)
        if row.get("sha256") != actual_sha:
            sha_ok = False
            invalid_files.append({"visual_file": str(path), "reason": "sha256_mismatch"})
        width, height, std = _image_stats(path)
        if width < 512 or height < 256:
            dim_ok = False
            invalid_files.append({"visual_file": str(path), "reason": "dimensions_too_small", "width": width, "height": height})
        if std <= 1.0:
            nonempty_ok = False
            invalid_files.append({"visual_file": str(path), "reason": "low_intensity_std", "std": std})
        for col in REQUIRED_OVERLAY_COLUMNS:
            if not _truthy(row.get(col, "")):
                overlays_ok = False
                invalid_files.append({"visual_file": str(path), "reason": f"missing_{col}"})

    duplicate_image_count = len(sha_values) - len(set(sha_values))
    review_statuses = [str(r.get("review_status", "")).strip() for r in reviews]
    invalid_review_rows = [
        {"visual_file": r.get("visual_file", ""), "review_status": r.get("review_status", "")}
        for r in reviews
        if str(r.get("review_status", "")).strip() not in ALLOWED_REVIEW_STATUS
    ]
    review_coverage = len(reviews) / max(1, len(manifest))
    gate_pass = bool(
        manifest
        and files_exist
        and sha_ok
        and dim_ok
        and nonempty_ok
        and duplicate_image_count == 0
        and overlays_ok
        and review_coverage >= 0.8
        and not invalid_review_rows
        and insight_path.exists()
        and insight_path.stat().st_size > 0
    )
    summary = {
        "schema": "acl2_v78_visual_integrity_audit_v1",
        "visual_root": str(args.visual_root),
        "num_visual_files": len(manifest),
        "num_manifest_rows": len(manifest),
        "num_review_rows": len(reviews),
        "review_coverage": review_coverage,
        "all_files_exist": files_exist,
        "all_sha256_present": bool(manifest) and all(bool(r.get("sha256")) for r in manifest),
        "all_dimensions_present": dim_ok,
        "all_nonempty": nonempty_ok,
        "duplicate_image_count": duplicate_image_count,
        "missing_overlay_count": 0 if overlays_ok else sum(
            1 for r in manifest for col in REQUIRED_OVERLAY_COLUMNS if not _truthy(r.get(col, ""))
        ),
        "invalid_visual_count": len(invalid_files),
        "visual_insight_present": insight_path.exists() and insight_path.stat().st_size > 0,
        "invalid_review_rows": invalid_review_rows[:50],
        "review_status_counts": {status: review_statuses.count(status) for status in sorted(set(review_statuses))},
        "invalid_files": invalid_files[:50],
        "gate_pass": gate_pass,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
