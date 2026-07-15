#!/usr/bin/env python3
"""Run file-level guard controls for v108 user-review activation."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE20_RUNNER = ROOT / "tools" / "run_v108_phase20_user_review_activation_preflight.py"
READY_STATUS = "READY_EXCEPT_USER_VISUAL_ACCEPTANCE"
ATTESTATION_TYPE = "stream4d_v108_explicit_user_visual_acceptance_v1"
ATTESTATION_TEXT = (
    "I visually reviewed every current ready contact sheet and accept all listed ready rows "
    "for durable SAM2 memory preflight."
)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def acceptance_fingerprint_entries_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in records:
        entries.append(
            {
                "scene_id": str(row.get("scene_id", "")),
                "object_id": int(row.get("object_id", -1)),
                "frame_id": int(row.get("frame_id", -1)),
                "event_index": int(row.get("event_index", -1)),
                "contact_sheet": str(row.get("contact_sheet", "")),
                "contact_sheet_sha256": str(row.get("contact_sheet_sha256", "")),
            }
        )
    entries.sort(key=lambda item: (item["scene_id"], item["object_id"], item["frame_id"], item["contact_sheet"]))
    return entries


def acceptance_fingerprint_sha256_from_records(records: list[dict[str, Any]]) -> str:
    import hashlib

    payload = json.dumps(acceptance_fingerprint_entries_from_records(records), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def attach_valid_attestation(payload: dict[str, Any], records_override: list[dict[str, Any]] | None = None) -> None:
    records = list(records_override if records_override is not None else payload.get("records", []))
    payload["explicit_user_visual_acceptance_attestation"] = {
        "attestation_type": ATTESTATION_TYPE,
        "reviewer": "user",
        "accepted_all_current_ready_rows": True,
        "attestation_text": ATTESTATION_TEXT,
        "ready_row_count": int(len(records)),
        "ready_evidence_fingerprint_sha256": acceptance_fingerprint_sha256_from_records(records),
        "review_timestamp_utc": "1970-01-01T00:00:00Z",
    }


def candidate_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("scene_id", "")), int(row.get("live_obj_id", -1)), int(row.get("frame_id", -1)))


def parse_stdout_json(text: str) -> dict[str, Any]:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return dict(json.loads(line))
        except Exception:
            continue
    return {}


def build_manifest(template: dict[str, Any], *, control_name: str, mode: str) -> dict[str, Any]:
    payload = copy.deepcopy(template)
    payload["schema_version"] = "stream4d_v108_phase21_user_review_guard_control_manifest_v1"
    payload["phase21_control_name"] = control_name
    payload["manifest_is_template"] = False
    payload["codex_must_not_mark_accepted"] = False
    payload["negative_control"] = False
    payload["not_real_user_acceptance"] = False
    payload["control_mode"] = mode
    payload["expected_result"] = {
        "durable_memory_mutation_request_emitted": False,
        "sam2_memory_mutation_applied": False,
    }
    records = payload.get("records", [])
    ready_records = copy.deepcopy(records)
    if mode == "template_manifest_as_review":
        payload = copy.deepcopy(template)
        payload["schema_version"] = "stream4d_v108_phase21_user_review_guard_control_manifest_v1"
        payload["phase21_control_name"] = control_name
        payload["control_mode"] = mode
        payload["expected_result"] = {
            "durable_memory_mutation_request_emitted": False,
            "sam2_memory_mutation_applied": False,
        }
    elif mode == "not_real_user_accepted_flag":
        payload["negative_control"] = True
        payload["not_real_user_acceptance"] = True
        for row in records:
            row["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
            row["reviewer"] = "user"
            row["visual_note"] = (
                "NEGATIVE CONTROL ONLY: top-level not_real_user_acceptance must be rejected "
                "even when row fields mimic user acceptance."
            )
    elif mode == "user_accepted_missing_attestation":
        for row in records:
            row["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
            row["reviewer"] = "user"
            row["visual_note"] = (
                "NEGATIVE CONTROL ONLY: accepted rows without top-level explicit user attestation "
                "must be rejected before transaction preflight."
            )
        payload.pop("explicit_user_visual_acceptance_attestation", None)
    elif mode == "user_accepted_schema_check_provenance":
        for row in records:
            row["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
            row["reviewer"] = "user"
            row["visual_note"] = (
                "NEGATIVE CONTROL ONLY: schema-check provenance must be rejected even when "
                "accepted fields and attestation are syntactically present."
            )
        attach_valid_attestation(payload, ready_records)
        payload["phase27_schema_check"] = "schema_check_only_not_real_user_review"
    elif mode == "non_user_accepted":
        for row in records:
            row["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
            row["reviewer"] = "codex_diagnostic_visual_review"
            row["visual_note"] = (
                "NEGATIVE CONTROL ONLY: non-user durable acceptance must be rejected. "
                "This is not a user review and must not activate memory."
            )
        attach_valid_attestation(payload, ready_records)
    elif mode == "user_accepted_bad_hash":
        for row in records:
            row["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
            row["reviewer"] = "user"
            row["visual_note"] = (
                "NEGATIVE CONTROL ONLY: user accepted status with corrupted evidence hash must be blocked. "
                "This is not a real user review."
            )
            row["evidence_sha256"] = ["0" * 64 for _ in row.get("evidence_sha256", [])]
            row["contact_sheet_sha256"] = "0" * 64
        attach_valid_attestation(payload, ready_records)
    elif mode == "user_accepted_current_file_hash_mismatch":
        for row in records:
            row["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
            row["reviewer"] = "user"
            row["visual_note"] = (
                "NEGATIVE CONTROL ONLY: user accepted fields with a stale evidence sha must be blocked "
                "when the current contact-sheet file hash no longer matches."
            )
        attach_valid_attestation(payload, ready_records)
    elif mode == "duplicate_conflicting_records":
        duplicate_records = []
        for row in records:
            duplicate = copy.deepcopy(row)
            duplicate["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
            duplicate["reviewer"] = "user"
            duplicate["visual_note"] = (
                "NEGATIVE CONTROL ONLY: duplicate accepted record after a pending record for the same "
                "scene/object/frame key must be rejected, not used by last-write-wins semantics."
            )
            duplicate_records.append(duplicate)
        payload["records"] = [*records, *duplicate_records]
        attach_valid_attestation(payload, ready_records)
    elif mode == "unexpected_extra_accepted_key":
        for row in records:
            row["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
            row["reviewer"] = "user"
            row["visual_note"] = (
                "NEGATIVE CONTROL ONLY: ready-row accepted fields plus an unexpected extra key "
                "must be rejected as an invalid manifest."
            )
        extra = copy.deepcopy(records[0])
        extra["object_id"] = 999999
        extra["frame_id"] = 999999
        extra["event_index"] = 999999
        extra["reference_obj_id"] = -1
        extra["visual_note"] = (
            "NEGATIVE CONTROL ONLY: unexpected extra accepted key must make manifest invalid."
        )
        payload["records"] = [*records, extra]
        attach_valid_attestation(payload, ready_records)
    elif mode == "partial_accepted_missing_second":
        payload["records"] = records[:1]
        for row in payload["records"]:
            row["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
            row["reviewer"] = "user"
            row["visual_note"] = (
                "NEGATIVE CONTROL ONLY: partial accepted coverage with a missing ready row "
                "must not emit a transaction preflight."
            )
        attach_valid_attestation(payload, ready_records)
    elif mode == "partial_accepted_second_pending":
        for index, row in enumerate(records):
            if index == 0:
                row["visual_review_status"] = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
                row["reviewer"] = "user"
                row["visual_note"] = (
                    "NEGATIVE CONTROL ONLY: first ready row accepted while another remains pending; "
                    "must not emit a transaction preflight."
                )
            else:
                row["visual_review_status"] = "USER_REVIEW_PENDING"
                row["reviewer"] = "user"
                row["visual_note"] = (
                    "NEGATIVE CONTROL ONLY: second ready row remains pending, so accepted coverage is partial."
                )
        attach_valid_attestation(payload, ready_records)
    elif mode == "diagnostic_not_durable":
        for row in records:
            row["visual_review_status"] = "DIAGNOSTIC_VISUAL_NOT_DURABLE"
            row["reviewer"] = "user"
            row["visual_note"] = (
                "NEGATIVE CONTROL ONLY: explicit diagnostic-not-durable decision must not activate memory."
            )
    elif mode == "missing_second_record":
        payload["records"] = records[:1]
        for row in payload["records"]:
            row["visual_review_status"] = "USER_REVIEW_PENDING"
            row["reviewer"] = "user"
            row["visual_note"] = (
                "NEGATIVE CONTROL ONLY: incomplete pending manifest must leave missing review rows blocked."
            )
    else:
        raise ValueError(f"unknown control mode: {mode}")
    return payload


def prepare_current_file_hash_mismatch_control(
    *,
    phase19_root: Path,
    output_root: Path,
    control_name: str,
    manifest_payload: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    source_payload = copy.deepcopy(read_json(phase19_root / "phase19_review_candidate_rows.json"))
    fake_phase19_root = output_root / "control_phase19_roots" / control_name
    fake_images = fake_phase19_root / "review_images"
    fake_images.mkdir(parents=True, exist_ok=True)
    evidence_by_key: dict[tuple[str, int, int], tuple[str, str, str]] = {}
    for row in source_payload.get("records", []):
        if row.get("source_label") != "baseline" or row.get("preflight_status") != READY_STATUS:
            continue
        event_index = int(row.get("event_index", -1))
        contact = dict(row.get("review_contact_sheet") or {})
        stale_path = str(contact.get("path", ""))
        stale_sha = str(contact.get("sha256", ""))
        tampered_path = fake_images / f"tampered_event{event_index:03d}_review_contact_sheet.png"
        tampered_path.write_bytes(
            (
                f"PHASE23 NEGATIVE CONTROL: current contact sheet file was replaced for event {event_index}; "
                f"manifest/candidate rows intentionally keep stale sha {stale_sha}.\n"
            ).encode("utf-8")
        )
        current_sha = sha256_file(tampered_path)
        contact["path"] = rel(tampered_path)
        contact["source_path"] = stale_path
        contact["sha256"] = stale_sha
        contact["phase23_current_file_sha256"] = current_sha
        contact["phase23_expected_guard_reason"] = "current_contact_sheet_sha256_mismatch"
        row["review_contact_sheet"] = contact
        evidence_by_key[candidate_key(row)] = (rel(tampered_path), stale_sha, current_sha)
    write_json(fake_phase19_root / "phase19_review_candidate_rows.json", source_payload)

    manifest_payload = copy.deepcopy(manifest_payload)
    for record in manifest_payload.get("records", []):
        key = (str(record.get("scene_id", "")), int(record.get("object_id", -1)), int(record.get("frame_id", -1)))
        if key not in evidence_by_key:
            continue
        evidence_path, stale_sha, current_sha = evidence_by_key[key]
        record["evidence_paths"] = [evidence_path]
        record["evidence_sha256"] = [stale_sha]
        record["contact_sheet"] = evidence_path
        record["contact_sheet_sha256"] = stale_sha
        record["phase23_current_file_sha256"] = current_sha
    attach_valid_attestation(manifest_payload)
    manifest_payload["phase23_control_fake_phase19_root"] = rel(fake_phase19_root)
    manifest_payload["phase23_control_ready_record_count"] = int(len(evidence_by_key))
    return fake_phase19_root, manifest_payload


def run_phase20(
    *,
    phase19_root: Path,
    output_root: Path,
    manifest: Path,
    max_requests_per_batch: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(PHASE20_RUNNER),
        "--phase19-root",
        rel(phase19_root),
        "--output-root",
        rel(output_root),
        "--review-manifest",
        rel(manifest),
        "--max-requests-per-batch",
        str(int(max_requests_per_batch)),
        "--run-self-checks",
    ]
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    stdout_payload = parse_stdout_json(completed.stdout)
    summary_path = resolve_path(stdout_payload.get("summary", output_root / "phase20_guarded_activation_summary.json"))
    summary = read_json(summary_path) if summary_path.is_file() else {}
    return {
        "cmd": " ".join(cmd),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "summary_path": rel(summary_path),
        "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else "",
        "summary": summary,
    }


def evaluate(control_name: str, mode: str, run: dict[str, Any]) -> tuple[bool, list[str]]:
    summary = dict(run.get("summary") or {})
    reasons: list[str] = []
    ok = True

    def require(condition: bool, reason: str) -> None:
        nonlocal ok
        if not condition:
            ok = False
            reasons.append(reason)

    require(summary.get("durable_memory_mutation_request_emitted") is False, "durable mutation request was emitted")
    require(summary.get("sam2_memory_mutation_applied") is False, "SAM2 memory mutation was applied")
    require(int(summary.get("guarded_transaction_request_count", -1)) == 0, "guarded transaction request count was nonzero")
    if mode == "non_user_accepted":
        require(int(run.get("returncode", -1)) == 2, "non-user accepted manifest did not return invalid-manifest exit code 2")
        require(summary.get("status") == "USER_REVIEW_MANIFEST_INVALID", "non-user accepted status was not invalid")
        require("reserved for explicit user review" in str(summary.get("manifest_error", "")), "missing non-user guard error")
    elif mode == "template_manifest_as_review":
        require(int(run.get("returncode", -1)) == 2, "template manifest did not return invalid-manifest exit code 2")
        require(summary.get("status") == "USER_REVIEW_MANIFEST_INVALID", "template manifest status was not invalid")
        require("manifest_is_template" in str(summary.get("manifest_error", "")), "missing template-manifest guard error")
    elif mode == "not_real_user_accepted_flag":
        require(int(run.get("returncode", -1)) == 2, "not-real accepted manifest did not return invalid-manifest exit code 2")
        require(summary.get("status") == "USER_REVIEW_MANIFEST_INVALID", "not-real accepted status was not invalid")
        error_text = str(summary.get("manifest_error", ""))
        require("not_real_user_acceptance" in error_text, "missing not-real-user-acceptance guard error")
        require("negative_control_manifest" in error_text, "missing negative-control guard error")
    elif mode == "user_accepted_missing_attestation":
        require(int(run.get("returncode", -1)) == 2, "missing-attestation manifest did not return invalid-manifest exit code 2")
        require(summary.get("status") == "USER_REVIEW_MANIFEST_INVALID", "missing-attestation status was not invalid")
        require(
            "missing_explicit_user_visual_acceptance_attestation" in str(summary.get("manifest_error", "")),
            "missing explicit-user-attestation guard error",
        )
    elif mode == "user_accepted_schema_check_provenance":
        require(int(run.get("returncode", -1)) == 2, "schema-check provenance manifest did not return invalid-manifest exit code 2")
        require(summary.get("status") == "USER_REVIEW_MANIFEST_INVALID", "schema-check provenance status was not invalid")
        require(
            "accepted_manifest_has_probe_or_schema_check_provenance" in str(summary.get("manifest_error", "")),
            "missing schema-check provenance guard error",
        )
    elif mode == "user_accepted_bad_hash":
        require(int(run.get("returncode", -1)) == 0, "bad-hash manifest returned nonzero")
        require(summary.get("status") == "USER_REVIEW_EVIDENCE_MISMATCH_BLOCKED", "bad-hash status was not mismatch-blocked")
        require(int(summary.get("evidence_mismatch_count", 0)) == int(summary.get("ready_row_count", -1)), "bad-hash mismatch count did not cover all ready rows")
    elif mode == "user_accepted_current_file_hash_mismatch":
        require(int(run.get("returncode", -1)) == 0, "current-file-hash mismatch manifest returned nonzero")
        require(
            summary.get("status") == "USER_REVIEW_EVIDENCE_MISMATCH_BLOCKED",
            "current-file-hash mismatch status was not mismatch-blocked",
        )
        require(
            int(summary.get("evidence_mismatch_count", 0)) == int(summary.get("ready_row_count", -1)),
            "current-file-hash mismatch count did not cover all ready rows",
        )
        activation_path = resolve_path(str(summary.get("activation_rows_json", "")))
        activation_rows = read_json(activation_path).get("records", []) if activation_path.is_file() else []
        require(
            bool(activation_rows)
            and all("current_contact_sheet_sha256_mismatch" in str(row.get("guard_reason", "")) for row in activation_rows),
            "activation rows did not record current_contact_sheet_sha256_mismatch",
        )
    elif mode == "duplicate_conflicting_records":
        require(int(run.get("returncode", -1)) == 2, "duplicate-record manifest did not return invalid-manifest exit code 2")
        require(summary.get("status") == "USER_REVIEW_MANIFEST_INVALID", "duplicate-record status was not invalid")
        require(
            "duplicate_visual_review_key" in str(summary.get("manifest_error", "")),
            "missing duplicate visual review key guard error",
        )
    elif mode == "unexpected_extra_accepted_key":
        require(int(run.get("returncode", -1)) == 2, "unexpected-extra-key manifest did not return invalid-manifest exit code 2")
        require(summary.get("status") == "USER_REVIEW_MANIFEST_INVALID", "unexpected-extra-key status was not invalid")
        require(
            "unexpected_visual_review_keys" in str(summary.get("manifest_error", "")),
            "missing unexpected visual review key guard error",
        )
    elif mode == "partial_accepted_missing_second":
        require(int(run.get("returncode", -1)) == 0, "partial-accepted-missing manifest returned nonzero")
        require(
            summary.get("status") == "USER_REVIEW_PARTIAL_ACCEPTANCE_BLOCKED",
            "partial-accepted-missing status was not partial-acceptance-blocked",
        )
        require(int(summary.get("accepted_ready_count", 0)) == 1, "partial-accepted-missing accepted count was not 1")
        require(int(summary.get("missing_review_count", 0)) == 1, "partial-accepted-missing missing count was not 1")
    elif mode == "partial_accepted_second_pending":
        require(int(run.get("returncode", -1)) == 0, "partial-accepted-pending manifest returned nonzero")
        require(
            summary.get("status") == "USER_REVIEW_PARTIAL_ACCEPTANCE_BLOCKED",
            "partial-accepted-pending status was not partial-acceptance-blocked",
        )
        require(int(summary.get("accepted_ready_count", 0)) == 1, "partial-accepted-pending accepted count was not 1")
        require(int(summary.get("pending_ready_count", 0)) == 1, "partial-accepted-pending pending count was not 1")
    elif mode == "diagnostic_not_durable":
        require(int(run.get("returncode", -1)) == 0, "diagnostic-not-durable manifest returned nonzero")
        require(summary.get("status") == "NO_USER_DURABLE_ACCEPTANCE_PRESENT", "diagnostic-not-durable status was not no-acceptance")
        require(int(summary.get("rejected_or_diagnostic_count", 0)) == int(summary.get("ready_row_count", -1)), "diagnostic count did not cover all ready rows")
    elif mode == "missing_second_record":
        require(int(run.get("returncode", -1)) == 0, "missing-record manifest returned nonzero")
        require(summary.get("status") == "NO_USER_DURABLE_ACCEPTANCE_PRESENT", "missing-record status was not no-acceptance")
        require(int(summary.get("missing_review_count", -1)) == 1, "missing-review count was not 1")
    else:
        ok = False
        reasons.append(f"unknown mode {mode}")
    if ok:
        reasons.append("control behaved as expected")
    return ok, reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase19-root", required=True)
    parser.add_argument("--phase20-template-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-requests-per-batch", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    phase19_root = resolve_path(args.phase19_root)
    template_root = resolve_path(args.phase20_template_root)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    template_path = template_root / "phase20_user_review_manifest_template.json"
    template = read_json(template_path)
    controls = [
        ("control_template_manifest_as_review", "template_manifest_as_review"),
        ("control_not_real_user_accepted_flag", "not_real_user_accepted_flag"),
        ("control_user_accepted_missing_attestation", "user_accepted_missing_attestation"),
        ("control_user_accepted_schema_check_provenance", "user_accepted_schema_check_provenance"),
        ("control_non_user_accepted", "non_user_accepted"),
        ("control_user_accepted_bad_hash", "user_accepted_bad_hash"),
        ("control_user_accepted_current_file_hash_mismatch", "user_accepted_current_file_hash_mismatch"),
        ("control_duplicate_conflicting_records", "duplicate_conflicting_records"),
        ("control_unexpected_extra_accepted_key", "unexpected_extra_accepted_key"),
        ("control_partial_accepted_missing_second", "partial_accepted_missing_second"),
        ("control_partial_accepted_second_pending", "partial_accepted_second_pending"),
        ("control_diagnostic_not_durable", "diagnostic_not_durable"),
        ("control_missing_second_record", "missing_second_record"),
    ]

    rows: list[dict[str, Any]] = []
    manifest_dir = output_root / "control_manifests"
    run_dir = output_root / "phase20_control_runs"
    for control_name, mode in controls:
        manifest_path = manifest_dir / f"{control_name}.json"
        control_phase19_root = phase19_root
        manifest_payload = build_manifest(template, control_name=control_name, mode=mode)
        if mode == "user_accepted_current_file_hash_mismatch":
            control_phase19_root, manifest_payload = prepare_current_file_hash_mismatch_control(
                phase19_root=phase19_root,
                output_root=output_root,
                control_name=control_name,
                manifest_payload=manifest_payload,
            )
        write_json(manifest_path, manifest_payload)
        run = run_phase20(
            phase19_root=control_phase19_root,
            output_root=run_dir / control_name,
            manifest=manifest_path,
            max_requests_per_batch=int(args.max_requests_per_batch),
        )
        passed, reasons = evaluate(control_name, mode, run)
        summary = dict(run.get("summary") or {})
        rows.append(
            {
                "control_name": control_name,
                "mode": mode,
                "phase19_root": rel(control_phase19_root),
                "manifest": rel(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "phase20_returncode": int(run.get("returncode", -1)),
                "phase20_summary": run.get("summary_path", ""),
                "phase20_summary_sha256": run.get("summary_sha256", ""),
                "observed_status": summary.get("status", ""),
                "ready_row_count": int(summary.get("ready_row_count", 0)),
                "accepted_ready_count": int(summary.get("accepted_ready_count", 0)),
                "pending_ready_count": int(summary.get("pending_ready_count", 0)),
                "missing_review_count": int(summary.get("missing_review_count", 0)),
                "rejected_or_diagnostic_count": int(summary.get("rejected_or_diagnostic_count", 0)),
                "evidence_mismatch_count": int(summary.get("evidence_mismatch_count", 0)),
                "guarded_transaction_request_count": int(summary.get("guarded_transaction_request_count", 0)),
                "durable_memory_mutation_request_emitted": bool(summary.get("durable_memory_mutation_request_emitted")),
                "sam2_memory_mutation_applied": bool(summary.get("sam2_memory_mutation_applied")),
                "control_passed": bool(passed),
                "control_reasons": reasons,
            }
        )

    rows_csv = output_root / "phase21_guard_control_rows.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    import csv

    with rows_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    rows_json = output_root / "phase21_guard_control_rows.json"
    write_json(rows_json, {"schema_version": "stream4d_v108_phase21_guard_control_rows_v1", "records": rows})
    summary = {
        "schema_version": "stream4d_v108_phase21_user_review_guard_controls_summary_v1",
        "status": "PHASE21_GUARD_CONTROLS_PASS" if all(row["control_passed"] for row in rows) else "PHASE21_GUARD_CONTROLS_FAIL",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "phase19_root": rel(phase19_root),
        "phase20_template_root": rel(template_root),
        "phase20_template_manifest": rel(template_path),
        "phase20_template_manifest_sha256": sha256_file(template_path),
        "control_count": int(len(rows)),
        "control_pass_count": int(sum(1 for row in rows if row["control_passed"])),
        "durable_memory_mutation_request_emitted_any": any(row["durable_memory_mutation_request_emitted"] for row in rows),
        "sam2_memory_mutation_applied_any": any(row["sam2_memory_mutation_applied"] for row in rows),
        "guarded_transaction_request_count_total": int(sum(int(row["guarded_transaction_request_count"]) for row in rows)),
        "rows_csv": rel(rows_csv),
        "rows_csv_sha256": sha256_file(rows_csv),
        "rows_json": rel(rows_json),
        "rows_json_sha256": sha256_file(rows_json),
        "controls": rows,
        "note": (
            "All manifests in this suite are negative controls. They are not user visual acceptance "
            "records and must not be used to activate durable memory."
        ),
    }
    summary_path = output_root / "phase21_guard_control_summary.json"
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": rel(summary_path),
                "status": summary["status"],
                "control_count": int(summary["control_count"]),
                "control_pass_count": int(summary["control_pass_count"]),
                "durable_memory_mutation_request_emitted_any": bool(summary["durable_memory_mutation_request_emitted_any"]),
                "sam2_memory_mutation_applied_any": bool(summary["sam2_memory_mutation_applied_any"]),
                "guarded_transaction_request_count_total": int(summary["guarded_transaction_request_count_total"]),
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "PHASE21_GUARD_CONTROLS_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
