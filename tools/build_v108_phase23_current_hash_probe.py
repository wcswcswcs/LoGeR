#!/usr/bin/env python3
"""Build an isolated Phase23 stale-sha evidence probe for v108 review guards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
READY_STATUS = "READY_EXCEPT_USER_VISUAL_ACCEPTANCE"


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
    parser.add_argument("--phase19-root", required=True)
    parser.add_argument("--phase20-template", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase19_root = resolve_path(args.phase19_root)
    template_path = resolve_path(args.phase20_template)
    output_root = resolve_path(args.output_root)
    fake_phase19_root = output_root / "phase19_tampered_current_hash"
    fake_images = fake_phase19_root / "review_images"
    fake_images.mkdir(parents=True, exist_ok=True)

    payload = json.loads((phase19_root / "phase19_review_candidate_rows.json").read_text(encoding="utf-8"))
    ready_evidence: dict[tuple[str, int, int], tuple[str, str, str]] = {}
    for row in payload.get("records", []):
        if row.get("source_label") != "baseline" or row.get("preflight_status") != READY_STATUS:
            continue
        event_index = int(row.get("event_index", -1))
        contact = dict(row.get("review_contact_sheet") or {})
        stale_path = str(contact.get("path", ""))
        stale_sha = str(contact.get("sha256", ""))
        tampered_path = fake_images / f"tampered_event{event_index:03d}_review_contact_sheet.png"
        tampered_path.write_bytes(
            (
                f"PHASE23 NEGATIVE CONTROL: tampered current file for event {event_index}; "
                f"manifest intentionally keeps stale sha {stale_sha}.\n"
            ).encode("utf-8")
        )
        current_sha = sha256_file(tampered_path)
        contact["path"] = rel(tampered_path)
        contact["source_path"] = stale_path
        contact["sha256"] = stale_sha
        contact["phase23_current_file_sha256"] = current_sha
        contact["phase23_expected_result"] = (
            "old_phase20_should_not_trust_stale_candidate_sha_without_current_file_hash"
        )
        row["review_contact_sheet"] = contact
        key = (str(row.get("scene_id", "")), int(row.get("live_obj_id", -1)), int(row.get("frame_id", -1)))
        ready_evidence[key] = (rel(tampered_path), stale_sha, current_sha)

    fake_rows_path = fake_phase19_root / "phase19_review_candidate_rows.json"
    write_json(fake_rows_path, payload)

    manifest = copy.deepcopy(json.loads(template_path.read_text(encoding="utf-8")))
    manifest["schema_version"] = "stream4d_v108_phase23_pre_repair_current_evidence_hash_probe_manifest_v1"
    manifest["manifest_is_template"] = False
    manifest["codex_must_not_mark_accepted"] = False
    manifest["user_review_required"] = True
    manifest["phase23_negative_probe"] = "current_contact_sheet_file_hash_mismatch"
    manifest["not_real_user_acceptance"] = False
    manifest["negative_control"] = False
    for record in manifest.get("records", []):
        key = (str(record.get("scene_id", "")), int(record.get("object_id", -1)), int(record.get("frame_id", -1)))
        evidence_path, stale_sha, current_sha = ready_evidence[key]
        record["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
        record["reviewer"] = "user"
        record["visual_note"] = (
            "PHASE23 NEGATIVE CONTROL ONLY: accepted fields with stale evidence sha and "
            "tampered current file; not a real user review."
        )
        record["evidence_paths"] = [evidence_path]
        record["evidence_sha256"] = [stale_sha]
        record["contact_sheet"] = evidence_path
        record["contact_sheet_sha256"] = stale_sha
        record["phase23_current_file_sha256"] = current_sha

    manifest_path = output_root / "pre_repair_user_accepted_stale_sha_manifest.json"
    write_json(manifest_path, manifest)
    summary = {
        "fake_phase19_root": rel(fake_phase19_root),
        "fake_phase19_candidate_rows_json": rel(fake_rows_path),
        "fake_phase19_candidate_rows_json_sha256": sha256_file(fake_rows_path),
        "manifest": rel(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "ready_evidence": [
            {
                "key": list(key),
                "path": evidence_path,
                "stale_sha256": stale_sha,
                "current_file_sha256": current_sha,
                "hash_mismatch": stale_sha != current_sha,
            }
            for key, (evidence_path, stale_sha, current_sha) in sorted(ready_evidence.items())
        ],
    }
    summary_path = output_root / "pre_repair_probe_setup_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
