#!/usr/bin/env python3
"""Build an isolated Phase24 duplicate-review-record probe for v108 guards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase20-template", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template_path = resolve_path(args.phase20_template)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = copy.deepcopy(json.loads(template_path.read_text(encoding="utf-8")))
    manifest["schema_version"] = "stream4d_v108_phase24_pre_repair_duplicate_review_records_probe_manifest_v1"
    manifest["manifest_is_template"] = False
    manifest["codex_must_not_mark_accepted"] = False
    manifest["negative_control"] = False
    manifest["not_real_user_acceptance"] = False
    manifest["phase24_negative_probe"] = "duplicate_conflicting_review_records_same_key"

    original_records = copy.deepcopy(manifest["records"])
    duplicates = []
    for record in original_records:
        duplicate = copy.deepcopy(record)
        duplicate["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
        duplicate["reviewer"] = "user"
        duplicate["visual_note"] = (
            "PHASE24 NEGATIVE CONTROL ONLY: duplicate accepted record after pending record; "
            "not a real user review."
        )
        duplicates.append(duplicate)
    manifest["records"] = [*original_records, *duplicates]

    manifest_path = output_root / "pre_repair_duplicate_conflicting_records_manifest.json"
    write_json(manifest_path, manifest)
    unique_keys = {
        (str(row["scene_id"]), int(row["object_id"]), int(row["frame_id"])) for row in manifest["records"]
    }
    summary = {
        "manifest": rel(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "record_count": int(len(manifest["records"])),
        "unique_key_count": int(len(unique_keys)),
        "duplicate_key_count": int(len(manifest["records"]) - len(unique_keys)),
        "expected_pre_repair_risk": "old loader overwrites pending rows with later accepted duplicate rows",
    }
    summary_path = output_root / "pre_repair_duplicate_probe_setup_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
