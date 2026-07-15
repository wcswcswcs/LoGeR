#!/usr/bin/env python3
"""Build isolated Phase26 partial-acceptance probes for v108 guards."""

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

    template = json.loads(template_path.read_text(encoding="utf-8"))
    records = copy.deepcopy(template["records"])
    accepted = copy.deepcopy(records[0])
    accepted["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
    accepted["reviewer"] = "user"
    accepted["visual_note"] = (
        "PHASE26 NEGATIVE CONTROL: only one ready row accepted; not a real user review."
    )

    missing_manifest = copy.deepcopy(template)
    missing_manifest["schema_version"] = "stream4d_v108_phase26_pre_repair_partial_accepted_missing_manifest_v1"
    missing_manifest["manifest_is_template"] = False
    missing_manifest["codex_must_not_mark_accepted"] = False
    missing_manifest["negative_control"] = False
    missing_manifest["not_real_user_acceptance"] = False
    missing_manifest["phase26_negative_probe"] = "partial_accepted_missing_second_ready_row"
    missing_manifest["records"] = [accepted]
    missing_path = output_root / "pre_repair_partial_accepted_missing_second_manifest.json"
    write_json(missing_path, missing_manifest)

    pending_manifest = copy.deepcopy(template)
    pending_manifest["schema_version"] = "stream4d_v108_phase26_pre_repair_partial_accepted_pending_manifest_v1"
    pending_manifest["manifest_is_template"] = False
    pending_manifest["codex_must_not_mark_accepted"] = False
    pending_manifest["negative_control"] = False
    pending_manifest["not_real_user_acceptance"] = False
    pending_manifest["phase26_negative_probe"] = "partial_accepted_second_ready_row_pending"
    pending = copy.deepcopy(records[1])
    pending["visual_review_status"] = "USER_REVIEW_PENDING"
    pending["reviewer"] = "user"
    pending["visual_note"] = (
        "PHASE26 NEGATIVE CONTROL: second ready row remains pending; not a real user review."
    )
    pending_manifest["records"] = [accepted, pending]
    pending_path = output_root / "pre_repair_partial_accepted_second_pending_manifest.json"
    write_json(pending_path, pending_manifest)

    summary = {
        "missing_manifest": rel(missing_path),
        "missing_manifest_sha256": sha256_file(missing_path),
        "pending_manifest": rel(pending_path),
        "pending_manifest_sha256": sha256_file(pending_path),
        "ready_row_count": int(len(records)),
        "accepted_key": [
            str(accepted["scene_id"]),
            int(accepted["object_id"]),
            int(accepted["frame_id"]),
        ],
        "missing_case_record_count": int(len(missing_manifest["records"])),
        "pending_case_record_count": int(len(pending_manifest["records"])),
        "expected_pre_repair_risk": (
            "old Phase20 emits partial guarded transaction request when accepted_ready_count "
            "is nonzero but not all ready rows are accepted"
        ),
    }
    summary_path = output_root / "pre_repair_partial_acceptance_probe_setup_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
