#!/usr/bin/env python3
"""Build an isolated Phase25 unexpected-review-key probe for v108 guards."""

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
    manifest["schema_version"] = "stream4d_v108_phase25_pre_repair_unexpected_manifest_key_probe_v1"
    manifest["manifest_is_template"] = False
    manifest["codex_must_not_mark_accepted"] = False
    manifest["negative_control"] = False
    manifest["not_real_user_acceptance"] = False
    manifest["phase25_negative_probe"] = "unexpected_extra_accepted_review_key"

    ready_records = copy.deepcopy(manifest["records"])
    for record in ready_records:
        record["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
        record["reviewer"] = "user"
        record["visual_note"] = (
            "PHASE25 NEGATIVE CONTROL: ready row accepted to expose extra-key scope handling; "
            "not real user review."
        )
    extra = copy.deepcopy(ready_records[0])
    extra["object_id"] = 999999
    extra["frame_id"] = 999999
    extra["event_index"] = 999999
    extra["reference_obj_id"] = -1
    extra["visual_note"] = "PHASE25 NEGATIVE CONTROL: unexpected extra accepted key must make manifest invalid."
    manifest["records"] = [*ready_records, extra]

    manifest_path = output_root / "pre_repair_unexpected_extra_accepted_key_manifest.json"
    write_json(manifest_path, manifest)
    ready_keys = {
        (str(row["scene_id"]), int(row["object_id"]), int(row["frame_id"])) for row in ready_records
    }
    manifest_keys = {
        (str(row["scene_id"]), int(row["object_id"]), int(row["frame_id"])) for row in manifest["records"]
    }
    unexpected_keys = sorted(manifest_keys - ready_keys)
    summary = {
        "manifest": rel(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "record_count": int(len(manifest["records"])),
        "ready_key_count": int(len(ready_keys)),
        "manifest_key_count": int(len(manifest_keys)),
        "unexpected_key_count": int(len(unexpected_keys)),
        "unexpected_keys": [list(key) for key in unexpected_keys],
        "expected_pre_repair_risk": "old Phase20 ignores unexpected manifest keys while accepting all ready rows",
    }
    summary_path = output_root / "pre_repair_unexpected_key_probe_setup_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
