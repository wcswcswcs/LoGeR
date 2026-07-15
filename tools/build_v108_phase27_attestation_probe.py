#!/usr/bin/env python3
"""Build isolated Phase27 explicit-user-attestation probes for v108 guards."""

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


def accepted_records(template: dict[str, Any], *, note: str) -> list[dict[str, Any]]:
    records = copy.deepcopy(template["records"])
    for row in records:
        row["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
        row["reviewer"] = "user"
        row["visual_note"] = note
    return records


def main() -> int:
    args = parse_args()
    template_path = resolve_path(args.phase20_template)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    template = json.loads(template_path.read_text(encoding="utf-8"))

    missing_attestation = copy.deepcopy(template)
    missing_attestation["schema_version"] = "stream4d_v108_phase27_pre_repair_all_accepted_missing_attestation_v1"
    missing_attestation["manifest_is_template"] = False
    missing_attestation["codex_must_not_mark_accepted"] = False
    missing_attestation["negative_control"] = False
    missing_attestation["not_real_user_acceptance"] = False
    missing_attestation["phase27_probe"] = "all_ready_rows_accepted_without_explicit_user_attestation"
    missing_attestation["records"] = accepted_records(
        template,
        note=(
            "PHASE27 PRE-REPAIR PROBE: all ready rows accepted but no explicit top-level "
            "user visual-attestation object is present; not a real user review."
        ),
    )
    missing_path = output_root / "pre_repair_all_accepted_missing_attestation_manifest.json"
    write_json(missing_path, missing_attestation)

    schema_check = copy.deepcopy(missing_attestation)
    schema_check["schema_version"] = "stream4d_v108_phase27_pre_repair_all_accepted_schema_check_provenance_v1"
    schema_check["phase27_schema_check"] = "schema_check_only_not_real_user_review"
    schema_check["records"] = accepted_records(
        template,
        note=(
            "PHASE27 PRE-REPAIR PROBE: all ready rows accepted with schema-check provenance; "
            "not a real user review."
        ),
    )
    schema_path = output_root / "pre_repair_all_accepted_schema_check_provenance_manifest.json"
    write_json(schema_path, schema_check)

    summary = {
        "missing_attestation_manifest": rel(missing_path),
        "missing_attestation_manifest_sha256": sha256_file(missing_path),
        "schema_check_manifest": rel(schema_path),
        "schema_check_manifest_sha256": sha256_file(schema_path),
        "ready_row_count": int(len(template["records"])),
        "expected_pre_repair_risk": (
            "old Phase20 accepts all-ready user/reviewer rows without a separate explicit "
            "user visual-attestation object and emits guarded transaction preflight requests"
        ),
    }
    summary_path = output_root / "pre_repair_attestation_probe_setup_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
