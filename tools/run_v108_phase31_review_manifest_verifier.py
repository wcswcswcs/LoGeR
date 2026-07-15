#!/usr/bin/env python3
"""Verify a v108 user review manifest before Phase20 activation preflight.

This verifier is intentionally read-only with respect to SAM2 memory and the
Phase20 transaction preflight path. It reports whether a review manifest is
clean enough to pass into Phase20, but it does not construct transaction
requests and does not apply any SAM2 mutation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v108_phase20_user_review_activation_preflight import (  # noqa: E402
    VisualReviewStatus,
    contact_sheet_pair,
    evidence_matches,
    jsonable,
    load_phase19_candidates,
    load_visual_review_manifest,
    manifest_has_accepted_records,
    parse_bool_marker,
    parse_int,
    read_json,
    rel,
    required_acceptance_attestation,
    resolve_path,
    review_manifest_provenance_audit,
    row_key,
    sha256_file,
    write_json,
    write_template_manifest,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fieldnames})


def normalize_manifest_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        return {"records": list(raw)}
    raise ValueError({"invalid_review_manifest_payload_type": type(raw).__name__})


def count_manifest_records(payload: dict[str, Any]) -> int:
    records = payload.get("records", [])
    return len(records) if isinstance(records, list) else 0


def manifest_template_like(payload: dict[str, Any]) -> bool:
    return bool(
        parse_bool_marker(payload.get("manifest_is_template"))
        or parse_bool_marker(payload.get("codex_must_not_mark_accepted"))
        or parse_bool_marker(payload.get("negative_control"))
        or parse_bool_marker(payload.get("not_real_user_acceptance"))
    )


def empty_provenance_audit(ready_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "accepted_records_present": False,
        "required_attestation": required_acceptance_attestation(ready_rows),
        "provided_attestation": {},
        "user_attestation_verified": False,
        "user_attestation_guard_reasons": [],
        "accepted_probe_or_schema_check_keys": [],
    }


def key_as_dict(key: tuple[str, int, int]) -> dict[str, Any]:
    return {"scene_id": key[0], "object_id": int(key[1]), "frame_id": int(key[2])}


def build_review_rows(
    *,
    ready_rows: list[dict[str, Any]],
    manifest_by_key: dict[tuple[str, int, int], Any],
    final_ready_for_phase20: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = {
        "ready_row_count": int(len(ready_rows)),
        "accepted_ready_count": 0,
        "pending_ready_count": 0,
        "missing_review_count": 0,
        "diagnostic_or_rejected_count": 0,
        "evidence_mismatch_count": 0,
    }
    rows: list[dict[str, Any]] = []
    for ready in ready_rows:
        key = row_key(ready)
        record = manifest_by_key.get(key)
        contact_path, contact_sha = contact_sheet_pair(ready)
        base = {
            "scene_id": key[0],
            "object_id": key[1],
            "live_obj_id": key[1],
            "frame_id": key[2],
            "event_index": parse_int(ready.get("event_index"), -1),
            "reference_obj_id": parse_int(ready.get("reference_obj_id"), -1),
            "preflight_status": str(ready.get("preflight_status", "")),
            "contact_sheet": contact_path,
            "contact_sheet_sha256": contact_sha,
            "durable_memory_mutation_request_emitted": False,
            "sam2_memory_mutation_applied": False,
            "transaction_preflight_constructed": False,
        }
        if record is None:
            counts["missing_review_count"] += 1
            rows.append(
                {
                    **base,
                    "review_verification_status": "MISSING_REVIEW_RECORD",
                    "visual_review_status": VisualReviewStatus.USER_REVIEW_PENDING.value,
                    "reviewer": "",
                    "evidence_match": False,
                    "guard_reason": "no matching user review record",
                    "policy_user_attestation_verified": False,
                }
            )
            continue

        match, match_reason = evidence_matches(record, ready)
        if record.durable_review_accepted:
            if match:
                counts["accepted_ready_count"] += 1
                row_status = "ACCEPTED_READY_EVIDENCE_MATCHED"
            else:
                counts["evidence_mismatch_count"] += 1
                row_status = "EVIDENCE_HASH_MISMATCH_BLOCKED"
        elif record.visual_review_status == VisualReviewStatus.USER_REVIEW_PENDING.value:
            counts["pending_ready_count"] += 1
            row_status = "USER_REVIEW_PENDING"
        else:
            counts["diagnostic_or_rejected_count"] += 1
            row_status = "DIAGNOSTIC_OR_NON_DURABLE_REVIEW"

        rows.append(
            {
                **base,
                "review_verification_status": row_status,
                "visual_review_status": record.visual_review_status,
                "reviewer": record.reviewer,
                "evidence_match": bool(match),
                "guard_reason": match_reason,
                "policy_user_attestation_verified": bool(final_ready_for_phase20 and row_status == "ACCEPTED_READY_EVIDENCE_MATCHED"),
            }
        )

    if 0 < counts["accepted_ready_count"] < counts["ready_row_count"]:
        for row in rows:
            if row.get("review_verification_status") == "ACCEPTED_READY_EVIDENCE_MATCHED":
                row["review_verification_status"] = "ACCEPTED_PARTIAL_COVERAGE_BLOCKED"
                row["guard_reason"] = (
                    str(row.get("guard_reason", "")) + "; partial_acceptance_requires_all_current_ready_rows"
                )
                row["policy_user_attestation_verified"] = False
    return rows, counts


def derive_status(
    *,
    manifest_path: Path | None,
    payload: dict[str, Any],
    manifest_error: str,
    unexpected_key_count: int,
    provenance_audit: dict[str, Any],
    counts: dict[str, int],
) -> str:
    if manifest_path is None:
        return "REVIEW_MANIFEST_NOT_PROVIDED"
    if manifest_error:
        return "REVIEW_MANIFEST_INVALID"

    accepted_records_present = bool(provenance_audit.get("accepted_records_present"))
    provenance_reasons = list(provenance_audit.get("user_attestation_guard_reasons", []))
    if accepted_records_present and provenance_reasons:
        return "REVIEW_MANIFEST_INVALID"
    if unexpected_key_count:
        return "REVIEW_MANIFEST_INVALID"
    if counts["evidence_mismatch_count"]:
        return "REVIEW_MANIFEST_EVIDENCE_MISMATCH_BLOCKED"
    if 0 < counts["accepted_ready_count"] < counts["ready_row_count"]:
        return "REVIEW_MANIFEST_PARTIAL_ACCEPTANCE_BLOCKED"
    if counts["accepted_ready_count"] == counts["ready_row_count"] and counts["ready_row_count"] > 0:
        if provenance_audit.get("user_attestation_verified"):
            return "REVIEW_MANIFEST_READY_FOR_PHASE20_PREFLIGHT"
        return "REVIEW_MANIFEST_ACCEPTED_WITHOUT_VERIFIED_ATTESTATION_BLOCKED"
    if manifest_template_like(payload):
        return "REVIEW_MANIFEST_TEMPLATE_ONLY_OR_NOT_REAL_ACCEPTANCE"
    if counts["missing_review_count"]:
        return "REVIEW_MANIFEST_INCOMPLETE_NO_DURABLE_ACCEPTANCE"
    return "REVIEW_MANIFEST_NO_DURABLE_ACCEPTANCE_PRESENT"


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stream4D v108 Phase31 Review Manifest Verifier",
        "",
        f"status: {summary['status']}",
        f"phase20_preflight_may_be_run: {str(summary['phase20_preflight_may_be_run']).lower()}",
        "durable_memory_mutation_request_emitted: false",
        "transaction_preflight_constructed: false",
        "sam2_memory_mutation_applied: false",
        "metrics_are_diagnostic_only: true",
        "",
        "## Counts",
        "",
        f"- ready_row_count: {summary['ready_row_count']}",
        f"- manifest_record_count: {summary['manifest_record_count']}",
        f"- accepted_ready_count: {summary['accepted_ready_count']}",
        f"- pending_ready_count: {summary['pending_ready_count']}",
        f"- missing_review_count: {summary['missing_review_count']}",
        f"- diagnostic_or_rejected_count: {summary['diagnostic_or_rejected_count']}",
        f"- evidence_mismatch_count: {summary['evidence_mismatch_count']}",
        f"- unexpected_review_key_count: {summary['unexpected_review_key_count']}",
        "",
        "## Ready Rows",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### event {row['event_index']} frame {row['frame_id']} live {row['live_obj_id']}",
                "",
                f"- review_verification_status: {row['review_verification_status']}",
                f"- visual_review_status: {row['visual_review_status']}",
                f"- reviewer: {row['reviewer']}",
                f"- evidence_match: {row['evidence_match']}",
                f"- policy_user_attestation_verified: {row['policy_user_attestation_verified']}",
                f"- guard_reason: {row['guard_reason']}",
                f"- contact_sheet: {row['contact_sheet']}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase19-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--review-manifest", default="")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    phase19_root = resolve_path(args.phase19_root)
    manifest_path = resolve_path(args.review_manifest) if str(args.review_manifest).strip() else None
    ready_rows = load_phase19_candidates(phase19_root)

    template_path = output_root / "phase31_expected_phase20_review_manifest_template.json"
    template_payload = write_template_manifest(template_path, ready_rows)
    rows_csv = output_root / "phase31_review_manifest_verifier_rows.csv"
    rows_json = output_root / "phase31_review_manifest_verifier_rows.json"
    summary_path = output_root / "phase31_review_manifest_verifier_summary.json"
    markdown_path = output_root / "phase31_review_manifest_verifier.md"

    payload: dict[str, Any] = {}
    manifest_by_key: dict[tuple[str, int, int], Any] = {}
    manifest_error = ""
    unexpected_keys: list[tuple[str, int, int]] = []
    provenance_audit = empty_provenance_audit(ready_rows)
    if manifest_path is not None:
        try:
            payload = normalize_manifest_payload(read_json(manifest_path))
            provenance_audit = review_manifest_provenance_audit(payload, ready_rows)
            manifest_by_key = load_visual_review_manifest(manifest_path)
            ready_keys = {row_key(row) for row in ready_rows}
            unexpected_keys = sorted(set(manifest_by_key) - ready_keys)
        except Exception as exc:
            manifest_error = str(exc)
            manifest_by_key = {}
    else:
        payload = dict(template_payload)

    provisional_ready = bool(
        manifest_path is not None
        and not manifest_error
        and not unexpected_keys
        and provenance_audit.get("user_attestation_verified")
    )
    rows, counts = build_review_rows(
        ready_rows=ready_rows,
        manifest_by_key=manifest_by_key,
        final_ready_for_phase20=False,
    )
    status = derive_status(
        manifest_path=manifest_path,
        payload=payload,
        manifest_error=manifest_error,
        unexpected_key_count=len(unexpected_keys),
        provenance_audit=provenance_audit,
        counts=counts,
    )
    phase20_preflight_may_be_run = bool(status == "REVIEW_MANIFEST_READY_FOR_PHASE20_PREFLIGHT")
    if provisional_ready and phase20_preflight_may_be_run:
        for row in rows:
            row["policy_user_attestation_verified"] = bool(
                row.get("review_verification_status") == "ACCEPTED_READY_EVIDENCE_MATCHED"
            )

    write_csv(rows_csv, rows)
    write_json(rows_json, {"schema_version": "stream4d_v108_phase31_review_manifest_verifier_rows_v1", "records": rows})
    summary = {
        "schema_version": "stream4d_v108_phase31_review_manifest_verifier_summary_v1",
        "status": status,
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "phase19_root": rel(phase19_root),
        "phase19_candidate_rows_json": rel(phase19_root / "phase19_review_candidate_rows.json"),
        "phase19_candidate_rows_json_sha256": sha256_file(phase19_root / "phase19_review_candidate_rows.json"),
        "review_manifest": rel(manifest_path) if manifest_path else "",
        "review_manifest_sha256": sha256_file(manifest_path) if manifest_path and manifest_path.is_file() else "",
        "template_manifest": rel(template_path),
        "template_manifest_sha256": sha256_file(template_path),
        "template_record_count": int(len(template_payload.get("records", []))),
        "manifest_record_count": int(count_manifest_records(payload)) if manifest_path is not None else 0,
        "ready_row_count": int(counts["ready_row_count"]),
        "accepted_ready_count": int(counts["accepted_ready_count"]),
        "pending_ready_count": int(counts["pending_ready_count"]),
        "missing_review_count": int(counts["missing_review_count"]),
        "diagnostic_or_rejected_count": int(counts["diagnostic_or_rejected_count"]),
        "evidence_mismatch_count": int(counts["evidence_mismatch_count"]),
        "unexpected_review_key_count": int(len(unexpected_keys)),
        "unexpected_review_keys": [key_as_dict(key) for key in unexpected_keys],
        "manifest_error": manifest_error,
        "manifest_provenance_audit": provenance_audit,
        "accepted_records_present": bool(provenance_audit.get("accepted_records_present")),
        "user_attestation_verified": bool(provenance_audit.get("user_attestation_verified")),
        "user_attestation_guard_reasons": list(provenance_audit.get("user_attestation_guard_reasons", [])),
        "required_ready_evidence_fingerprint_sha256": provenance_audit["required_attestation"][
            "ready_evidence_fingerprint_sha256"
        ],
        "activation_attestation_ready_for_policy": bool(phase20_preflight_may_be_run),
        "phase20_preflight_may_be_run": bool(phase20_preflight_may_be_run),
        "policy_user_attestation_verified_count": int(
            sum(1 for row in rows if row.get("policy_user_attestation_verified"))
        ),
        "durable_memory_mutation_request_emitted": False,
        "guarded_transaction_request_count": 0,
        "transaction_preflight_constructed": False,
        "sam2_memory_mutation_applied": False,
        "metrics_are_diagnostic_only": True,
        "visual_acceptance_claimed_by_codex": False,
        "review_rows_csv": rel(rows_csv),
        "review_rows_csv_sha256": sha256_file(rows_csv),
        "review_rows_json": rel(rows_json),
        "review_rows_json_sha256": sha256_file(rows_json),
        "markdown": rel(markdown_path),
        "acceptance_rule": (
            "Only an exact-scope manifest with reviewer=user durable acceptance for every current "
            "ready row, current evidence path/hash/file match, and the exact top-level user visual "
            "attestation may be passed into Phase20. This verifier never emits transaction requests "
            "and never applies SAM2 memory."
        ),
    }
    write_markdown(markdown_path, summary, rows)
    summary["markdown_sha256"] = sha256_file(markdown_path)
    write_json(summary_path, summary)

    print(
        json.dumps(
            {
                "summary": rel(summary_path),
                "status": status,
                "ready_row_count": int(counts["ready_row_count"]),
                "accepted_ready_count": int(counts["accepted_ready_count"]),
                "phase20_preflight_may_be_run": bool(phase20_preflight_may_be_run),
                "durable_memory_mutation_request_emitted": False,
                "transaction_preflight_constructed": False,
                "sam2_memory_mutation_applied": False,
            },
            sort_keys=True,
        )
    )
    if args.fail_on_not_ready and not phase20_preflight_may_be_run:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
