#!/usr/bin/env python3
"""Guard the v108 post-review durable activation path without applying SAM2 memory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "Stream3D") not in sys.path:
    sys.path.insert(1, str(ROOT / "Stream3D"))

from Stream3D.stream4d_v108.transaction_manager import (  # noqa: E402
    Sam2MemoryMutationRequest,
    SparseTransactionScheduler,
)
from Stream3D.stream4d_v108.visual_review import (  # noqa: E402
    VisualReviewRecord,
    VisualReviewStatus,
    load_visual_review_manifest,
)


READY_STATUS = "READY_EXCEPT_USER_VISUAL_ACCEPTANCE"
TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
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
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fieldnames})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool_marker(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in TRUE_STRINGS


def parse_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if not text:
            return int(default)
        return int(float(text))
    except Exception:
        return int(default)


def load_phase19_candidates(phase19_root: Path) -> list[dict[str, Any]]:
    payload = read_json(phase19_root / "phase19_review_candidate_rows.json")
    rows = [dict(row) for row in payload.get("records", [])]
    ready_rows = [
        row
        for row in rows
        if row.get("source_label") == "baseline" and row.get("preflight_status") == READY_STATUS
    ]
    ready_rows.sort(key=lambda row: (parse_int(row.get("frame_id"), 999999), parse_int(row.get("event_index"), 999999)))
    return ready_rows


def row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("scene_id", "")), parse_int(row.get("live_obj_id"), -1), parse_int(row.get("frame_id"), -1))


def contact_sheet_pair(row: dict[str, Any]) -> tuple[str, str]:
    item = dict(row.get("review_contact_sheet") or {})
    return str(item.get("path", "")), str(item.get("sha256", ""))


def acceptance_fingerprint_entries(ready_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in ready_rows:
        path, digest = contact_sheet_pair(row)
        entries.append(
            {
                "scene_id": str(row.get("scene_id", "")),
                "object_id": parse_int(row.get("live_obj_id", row.get("object_id", -1)), -1),
                "frame_id": parse_int(row.get("frame_id"), -1),
                "event_index": parse_int(row.get("event_index"), -1),
                "contact_sheet": path,
                "contact_sheet_sha256": digest,
            }
        )
    entries.sort(key=lambda item: (item["scene_id"], item["object_id"], item["frame_id"], item["contact_sheet"]))
    return entries


def acceptance_fingerprint_sha256(ready_rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(acceptance_fingerprint_entries(ready_rows), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def required_acceptance_attestation(ready_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "attestation_type": ATTESTATION_TYPE,
        "reviewer": "user",
        "accepted_all_current_ready_rows": True,
        "attestation_text": ATTESTATION_TEXT,
        "ready_row_count": int(len(ready_rows)),
        "ready_evidence_fingerprint_sha256": acceptance_fingerprint_sha256(ready_rows),
        "review_timestamp_utc_required": True,
    }


def build_template_record(row: dict[str, Any]) -> dict[str, Any]:
    path, digest = contact_sheet_pair(row)
    return {
        "scene_id": str(row.get("scene_id", "")),
        "object_id": parse_int(row.get("live_obj_id"), -1),
        "frame_id": parse_int(row.get("frame_id"), -1),
        "visual_review_status": VisualReviewStatus.USER_REVIEW_PENDING.value,
        "visual_note": (
            "PENDING: no explicit user durable-memory acceptance has been provided. "
            "Only the user may change this to VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY after visual inspection."
        ),
        "evidence_paths": [path],
        "evidence_sha256": [digest],
        "reviewer": "user",
        "event_index": parse_int(row.get("event_index"), -1),
        "reference_obj_id": parse_int(row.get("reference_obj_id"), -1),
        "contact_sheet": path,
        "contact_sheet_sha256": digest,
    }


def write_template_manifest(path: Path, ready_rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = [build_template_record(row) for row in ready_rows]
    required_attestation = required_acceptance_attestation(ready_rows)
    payload = {
        "schema_version": "stream4d_v108_phase20_user_review_manifest_template_v1",
        "manifest_is_template": True,
        "codex_must_not_mark_accepted": True,
        "user_review_required": True,
        "allowed_statuses": [
            VisualReviewStatus.USER_REVIEW_PENDING.value,
            VisualReviewStatus.VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY.value,
            VisualReviewStatus.DIAGNOSTIC_VISUAL_NOT_DURABLE.value,
        ],
        "acceptance_rule": (
            "A durable activation preflight may proceed only when a record has "
            "visual_review_status=VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY and reviewer=user. "
            "Codex-generated diagnostic visual notes are not durable acceptance. "
            "Accepted records must also include an explicit top-level user visual-attestation object."
        ),
        "acceptance_attestation_required": required_attestation,
        "explicit_user_visual_acceptance_attestation": {
            "attestation_type": required_attestation["attestation_type"],
            "reviewer": "user",
            "accepted_all_current_ready_rows": False,
            "attestation_text": "",
            "ready_row_count": required_attestation["ready_row_count"],
            "ready_evidence_fingerprint_sha256": required_attestation["ready_evidence_fingerprint_sha256"],
            "review_timestamp_utc": "",
        },
        "records": records,
    }
    write_json(path, payload)
    return payload


def manifest_level_block_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if parse_bool_marker(payload.get("manifest_is_template")):
        reasons.append("manifest_is_template")
    if parse_bool_marker(payload.get("negative_control")):
        reasons.append("negative_control_manifest")
    if parse_bool_marker(payload.get("not_real_user_acceptance")):
        reasons.append("not_real_user_acceptance")
    if parse_bool_marker(payload.get("codex_must_not_mark_accepted")):
        for row in payload.get("records", []):
            if str(dict(row).get("visual_review_status", "")) == VisualReviewStatus.VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY.value:
                reasons.append("codex_must_not_mark_accepted_with_accepted_record")
                break
    return sorted(set(reasons))


def manifest_has_accepted_records(payload: dict[str, Any]) -> bool:
    return any(
        str(dict(row).get("visual_review_status", "")) == VisualReviewStatus.VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY.value
        for row in payload.get("records", [])
    )


def review_manifest_provenance_audit(payload: dict[str, Any], ready_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = manifest_level_block_reasons(payload)
    expected = required_acceptance_attestation(ready_rows)
    audit: dict[str, Any] = {
        "accepted_records_present": bool(manifest_has_accepted_records(payload)),
        "required_attestation": expected,
        "provided_attestation": {},
        "user_attestation_verified": False,
        "user_attestation_guard_reasons": [],
        "accepted_probe_or_schema_check_keys": [],
    }
    if not manifest_has_accepted_records(payload):
        audit["user_attestation_verified"] = False
        audit["user_attestation_guard_reasons"] = sorted(set(reasons))
        return audit

    for key, value in payload.items():
        low = str(key).lower()
        if value and ("schema_check" in low or low.endswith("_probe") or "negative_probe" in low):
            reason = f"accepted_manifest_has_probe_or_schema_check_provenance:{key}"
            reasons.append(reason)
            audit["accepted_probe_or_schema_check_keys"].append(str(key))

    attestation = payload.get("explicit_user_visual_acceptance_attestation")
    if not isinstance(attestation, dict):
        reasons.append("missing_explicit_user_visual_acceptance_attestation")
        audit["user_attestation_guard_reasons"] = sorted(set(reasons))
        return audit
    audit["provided_attestation"] = {
        "attestation_type": attestation.get("attestation_type", ""),
        "reviewer": attestation.get("reviewer", ""),
        "accepted_all_current_ready_rows": attestation.get("accepted_all_current_ready_rows", ""),
        "attestation_text": attestation.get("attestation_text", ""),
        "ready_row_count": attestation.get("ready_row_count", ""),
        "ready_evidence_fingerprint_sha256": attestation.get("ready_evidence_fingerprint_sha256", ""),
        "review_timestamp_utc": attestation.get("review_timestamp_utc", ""),
    }

    exact_fields = [
        "attestation_type",
        "reviewer",
        "accepted_all_current_ready_rows",
        "attestation_text",
        "ready_row_count",
        "ready_evidence_fingerprint_sha256",
    ]
    for field in exact_fields:
        if attestation.get(field) != expected.get(field):
            reasons.append(f"invalid_explicit_user_visual_acceptance_attestation:{field}")
    timestamp = str(attestation.get("review_timestamp_utc", "")).strip()
    if not timestamp:
        reasons.append("missing_explicit_user_visual_acceptance_attestation:review_timestamp_utc")
    audit["user_attestation_guard_reasons"] = sorted(set(reasons))
    audit["user_attestation_verified"] = bool(not audit["user_attestation_guard_reasons"])
    return audit


def review_manifest_guard_reasons(payload: dict[str, Any], ready_rows: list[dict[str, Any]]) -> list[str]:
    return list(review_manifest_provenance_audit(payload, ready_rows)["user_attestation_guard_reasons"])


def evidence_matches(record: VisualReviewRecord, row: dict[str, Any]) -> tuple[bool, str]:
    expected_path, expected_sha = contact_sheet_pair(row)
    pairs = set(zip(record.evidence_paths, record.evidence_sha256))
    if (expected_path, expected_sha) not in pairs:
        return (
            False,
            f"missing_expected_contact_sheet_pair path={expected_path} sha256={expected_sha}",
        )
    current_path = resolve_path(expected_path)
    if not current_path.is_file():
        return False, f"contact_sheet_file_missing path={expected_path}"
    current_sha = sha256_file(current_path)
    if current_sha != expected_sha:
        return (
            False,
            "current_contact_sheet_sha256_mismatch "
            f"path={expected_path} expected_sha256={expected_sha} current_sha256={current_sha}",
        )
    return True, "contact_sheet_path_sha256_and_current_file_hash_match"


def run_guard_self_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        bad = VisualReviewRecord(
            scene_id="synthetic_scene",
            object_id=1,
            frame_id=2,
            visual_review_status=VisualReviewStatus.VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY.value,
            visual_note="synthetic non-user acceptance guard check",
            evidence_paths=("synthetic.png",),
            evidence_sha256=("0" * 64,),
            reviewer="codex_diagnostic_visual_review",
        )
        bad.validate()
        checks["non_user_durable_acceptance_rejected"] = False
        checks["non_user_durable_acceptance_error"] = ""
    except Exception as exc:
        checks["non_user_durable_acceptance_rejected"] = True
        checks["non_user_durable_acceptance_error"] = str(exc)

    pending = VisualReviewRecord(
        scene_id="synthetic_scene",
        object_id=1,
        frame_id=2,
        visual_review_status=VisualReviewStatus.USER_REVIEW_PENDING.value,
        visual_note="synthetic pending guard check",
        evidence_paths=("synthetic.png",),
        evidence_sha256=("0" * 64,),
        reviewer="user",
    )
    pending.validate()
    checks["pending_user_record_valid"] = True
    checks["pending_user_record_durable_accepted"] = bool(pending.durable_review_accepted)

    accepted = VisualReviewRecord(
        scene_id="synthetic_scene",
        object_id=1,
        frame_id=2,
        visual_review_status=VisualReviewStatus.VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY.value,
        visual_note="synthetic user acceptance schema check only; not a real review",
        evidence_paths=("synthetic.png",),
        evidence_sha256=("0" * 64,),
        reviewer="user",
    )
    accepted.validate()
    checks["synthetic_user_acceptance_schema_valid"] = True
    checks["synthetic_user_acceptance_used_for_activation"] = False
    return checks


def build_activation_rows(
    *,
    ready_rows: list[dict[str, Any]],
    manifest_path: Path | None,
    user_attestation_verified: bool = False,
) -> tuple[list[dict[str, Any]], list[Sam2MemoryMutationRequest], dict[str, Any]]:
    if manifest_path is None:
        manifest: dict[tuple[str, int, int], VisualReviewRecord] = {}
    else:
        manifest = load_visual_review_manifest(manifest_path)
        ready_keys = {row_key(row) for row in ready_rows}
        unexpected_keys = sorted(set(manifest) - ready_keys)
        if unexpected_keys:
            raise ValueError(
                {
                    "unexpected_visual_review_keys": [
                        {"scene_id": key[0], "object_id": key[1], "frame_id": key[2]}
                        for key in unexpected_keys
                    ]
                }
            )

    activation_rows: list[dict[str, Any]] = []
    requests: list[Sam2MemoryMutationRequest] = []
    counts = {
        "ready_row_count": int(len(ready_rows)),
        "accepted_ready_count": 0,
        "pending_ready_count": 0,
        "missing_review_count": 0,
        "rejected_or_diagnostic_count": 0,
        "evidence_mismatch_count": 0,
    }
    for row in ready_rows:
        key = row_key(row)
        record = manifest.get(key)
        path, digest = contact_sheet_pair(row)
        base = {
            "scene_id": key[0],
            "event_index": parse_int(row.get("event_index"), -1),
            "frame_id": key[2],
            "live_obj_id": key[1],
            "reference_obj_id": parse_int(row.get("reference_obj_id"), -1),
            "preflight_status": row.get("preflight_status", ""),
            "contact_sheet": path,
            "contact_sheet_sha256": digest,
            "user_attestation_verified": bool(user_attestation_verified),
            "durable_memory_mutation_request_emitted": False,
            "sam2_memory_mutation_applied": False,
        }
        if record is None:
            counts["missing_review_count"] += 1
            activation_rows.append(
                {
                    **base,
                    "activation_status": "WAITING_FOR_USER_VISUAL_ACCEPTANCE",
                    "visual_review_status": VisualReviewStatus.USER_REVIEW_PENDING.value,
                    "reviewer": "",
                    "evidence_match": False,
                    "guard_reason": "no matching user review record",
                }
            )
            continue
        match, match_reason = evidence_matches(record, row)
        if record.durable_review_accepted:
            if match:
                counts["accepted_ready_count"] += 1
                requests.append(
                    Sam2MemoryMutationRequest(
                        frame_id=key[2],
                        global_object_id=key[1],
                        sam2_runtime_object_id=key[1],
                        mutation="durable_admission_after_explicit_user_visual_acceptance",
                        prompt_count=parse_int(row.get("physical_projected_positive_count"), 0),
                        evidence_status=(
                            "USER_VISUALLY_ACCEPTED_AND_ATTESTED_PREFLIGHT_NOT_APPLIED"
                            if user_attestation_verified
                            else "USER_VISUALLY_ACCEPTED_WITHOUT_VERIFIED_ATTESTATION_BLOCKED"
                        ),
                    )
                )
                activation_rows.append(
                    {
                        **base,
                        "activation_status": "USER_ACCEPTED_TRANSACTION_PREFLIGHT_READY_NOT_APPLIED",
                        "visual_review_status": record.visual_review_status,
                        "reviewer": record.reviewer,
                        "evidence_match": True,
                        "guard_reason": match_reason,
                    }
                )
            else:
                counts["evidence_mismatch_count"] += 1
                activation_rows.append(
                    {
                        **base,
                        "activation_status": "EVIDENCE_HASH_MISMATCH_BLOCKED",
                        "visual_review_status": record.visual_review_status,
                        "reviewer": record.reviewer,
                        "evidence_match": False,
                        "guard_reason": match_reason,
                    }
                )
            continue
        if record.visual_review_status == VisualReviewStatus.USER_REVIEW_PENDING.value:
            counts["pending_ready_count"] += 1
            status = "WAITING_FOR_USER_VISUAL_ACCEPTANCE"
        else:
            counts["rejected_or_diagnostic_count"] += 1
            status = "USER_REJECTED_OR_DIAGNOSTIC_ONLY"
        activation_rows.append(
            {
                **base,
                "activation_status": status,
                "visual_review_status": record.visual_review_status,
                "reviewer": record.reviewer,
                "evidence_match": bool(match),
                "guard_reason": match_reason,
            }
        )
    if 0 < counts["accepted_ready_count"] < counts["ready_row_count"]:
        requests = []
        for row in activation_rows:
            if row.get("activation_status") == "USER_ACCEPTED_TRANSACTION_PREFLIGHT_READY_NOT_APPLIED":
                row["activation_status"] = "USER_ACCEPTED_PARTIAL_COVERAGE_BLOCKED"
                row["guard_reason"] = (
                    str(row.get("guard_reason", ""))
                    + "; partial_acceptance_requires_all_ready_rows"
                )
    for row in activation_rows:
        row["policy_user_attestation_verified"] = bool(
            user_attestation_verified
            and row.get("activation_status") == "USER_ACCEPTED_TRANSACTION_PREFLIGHT_READY_NOT_APPLIED"
        )
    return activation_rows, requests, counts


def status_from_counts(*, manifest_path: Path | None, counts: dict[str, Any]) -> str:
    if manifest_path is None:
        return "AWAITING_USER_REVIEW_MANIFEST_TEMPLATE_ONLY"
    if counts["evidence_mismatch_count"]:
        return "USER_REVIEW_EVIDENCE_MISMATCH_BLOCKED"
    if counts["accepted_ready_count"] == counts["ready_row_count"] and counts["ready_row_count"]:
        return "USER_ACCEPTED_ALL_READY_DURABLE_TRANSACTION_PREFLIGHT_READY_NOT_APPLIED"
    if counts["accepted_ready_count"]:
        return "USER_REVIEW_PARTIAL_ACCEPTANCE_BLOCKED"
    return "NO_USER_DURABLE_ACCEPTANCE_PRESENT"


def write_markdown(path: Path, summary: dict[str, Any], activation_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stream4D v108 Phase20 User Review Activation Preflight",
        "",
        f"status: {summary['status']}",
        "durable_memory_mutation_request_emitted: false",
        "sam2_memory_mutation_applied: false",
        "metrics_are_diagnostic_only: true",
        "acceptance_rule: durable activation requires reviewer=user and VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY",
        "",
        "## Counts",
        "",
        f"- ready_row_count: {summary['ready_row_count']}",
        f"- accepted_ready_count: {summary['accepted_ready_count']}",
        f"- pending_ready_count: {summary['pending_ready_count']}",
        f"- missing_review_count: {summary['missing_review_count']}",
        f"- evidence_mismatch_count: {summary['evidence_mismatch_count']}",
        f"- guarded_transaction_request_count: {summary['guarded_transaction_request_count']}",
        "",
        "## Ready Rows",
        "",
    ]
    for row in activation_rows:
        lines.extend(
            [
                f"### event {row['event_index']} frame {row['frame_id']} live {row['live_obj_id']}",
                "",
                f"- activation_status: {row['activation_status']}",
                f"- visual_review_status: {row['visual_review_status']}",
                f"- reviewer: {row['reviewer']}",
                f"- evidence_match: {row['evidence_match']}",
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
    parser.add_argument("--max-requests-per-batch", type=int, default=4)
    parser.add_argument("--run-self-checks", action="store_true")
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
    template_path = output_root / "phase20_user_review_manifest_template.json"
    template_payload = write_template_manifest(template_path, ready_rows)

    rows_csv = output_root / "phase20_guarded_activation_rows.csv"
    rows_json = output_root / "phase20_guarded_activation_rows.json"
    batch_csv = output_root / "phase20_guarded_transaction_preflight_rows.csv"
    batch_json = output_root / "phase20_guarded_transaction_preflight_rows.json"
    summary_path = output_root / "phase20_guarded_activation_summary.json"
    markdown_path = output_root / "phase20_guarded_activation_preflight.md"

    manifest_error = ""
    manifest_level_guard_reasons: list[str] = []
    manifest_provenance_audit: dict[str, Any] = {
        "accepted_records_present": False,
        "required_attestation": required_acceptance_attestation(ready_rows),
        "provided_attestation": {},
        "user_attestation_verified": False,
        "user_attestation_guard_reasons": [],
        "accepted_probe_or_schema_check_keys": [],
    }
    exit_code = 0
    try:
        if manifest_path is not None:
            manifest_payload = dict(read_json(manifest_path))
            manifest_provenance_audit = review_manifest_provenance_audit(manifest_payload, ready_rows)
            manifest_level_guard_reasons = list(manifest_provenance_audit["user_attestation_guard_reasons"])
            if manifest_level_guard_reasons:
                raise ValueError(
                    "review manifest failed user-review provenance guards: "
                    + ",".join(manifest_level_guard_reasons)
                )
        activation_rows, requests, counts = build_activation_rows(
            ready_rows=ready_rows,
            manifest_path=manifest_path,
            user_attestation_verified=bool(manifest_provenance_audit["user_attestation_verified"]),
        )
        status = status_from_counts(manifest_path=manifest_path, counts=counts)
    except Exception as exc:
        activation_rows = []
        requests = []
        counts = {
            "ready_row_count": int(len(ready_rows)),
            "accepted_ready_count": 0,
            "pending_ready_count": 0,
            "missing_review_count": 0,
            "rejected_or_diagnostic_count": 0,
            "evidence_mismatch_count": 0,
        }
        status = "USER_REVIEW_MANIFEST_INVALID"
        manifest_error = str(exc)
        exit_code = 2

    scheduler = SparseTransactionScheduler(max_requests_per_batch=int(args.max_requests_per_batch))
    batches = scheduler.build_batches(requests)
    batch_rows = [row for batch in batches for row in batch.as_rows()]
    for row in batch_rows:
        row["shadow_only"] = True
        row["reason"] = "guarded user-review activation preflight; no SAM2 memory mutation applied"
    if not batch_rows:
        batch_rows = [
            {
                "batch_id": "",
                "batch_request_index": -1,
                "mode": "event_driven",
                "request_count": 0,
                "frame_id_min": "",
                "frame_id_max": "",
                "global_object_ids": [],
                "frame_id": -1,
                "global_object_id": -1,
                "sam2_runtime_object_id": "",
                "mutation": "",
                "prompt_count": 0,
                "evidence_status": "",
                "shadow_only": True,
                "reason": "no explicit user durable visual acceptance; no SAM2 memory mutation applied",
            }
        ]

    write_csv(rows_csv, activation_rows)
    write_json(rows_json, {"schema_version": "stream4d_v108_phase20_guarded_activation_rows_v1", "records": activation_rows})
    write_csv(batch_csv, batch_rows)
    write_json(
        batch_json,
        {
            "schema_version": "stream4d_v108_phase20_guarded_transaction_preflight_rows_v1",
            "records": batch_rows,
            "shadow_only": True,
            "durable_memory_mutation_request_emitted": False,
            "sam2_memory_mutation_applied": False,
        },
    )

    self_checks = run_guard_self_checks() if args.run_self_checks else {}
    summary = {
        "schema_version": "stream4d_v108_phase20_user_review_activation_preflight_summary_v1",
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
        "template_record_count": int(len(template_payload["records"])),
        "ready_row_count": int(counts["ready_row_count"]),
        "accepted_ready_count": int(counts["accepted_ready_count"]),
        "pending_ready_count": int(counts["pending_ready_count"]),
        "missing_review_count": int(counts["missing_review_count"]),
        "rejected_or_diagnostic_count": int(counts["rejected_or_diagnostic_count"]),
        "evidence_mismatch_count": int(counts["evidence_mismatch_count"]),
        "guarded_transaction_batch_count": int(len(batches)),
        "guarded_transaction_request_count": int(len(requests)),
        "activation_attestation_ready_for_policy": bool(
            manifest_provenance_audit["user_attestation_verified"]
            and status == "USER_ACCEPTED_ALL_READY_DURABLE_TRANSACTION_PREFLIGHT_READY_NOT_APPLIED"
            and len(requests) == counts["ready_row_count"]
            and counts["ready_row_count"] > 0
        ),
        "policy_user_attestation_verified_count": int(
            sum(1 for row in activation_rows if row.get("policy_user_attestation_verified"))
        ),
        "durable_memory_mutation_request_emitted": False,
        "sam2_memory_mutation_applied": False,
        "metrics_are_diagnostic_only": True,
        "manifest_error": manifest_error,
        "manifest_level_guard_reasons": manifest_level_guard_reasons,
        "manifest_provenance_audit": manifest_provenance_audit,
        "accepted_records_present": bool(manifest_provenance_audit["accepted_records_present"]),
        "user_attestation_verified": bool(manifest_provenance_audit["user_attestation_verified"]),
        "user_attestation_guard_reasons": list(manifest_provenance_audit["user_attestation_guard_reasons"]),
        "required_ready_evidence_fingerprint_sha256": manifest_provenance_audit["required_attestation"][
            "ready_evidence_fingerprint_sha256"
        ],
        "guard_self_checks": self_checks,
        "activation_rows_csv": rel(rows_csv),
        "activation_rows_csv_sha256": sha256_file(rows_csv),
        "activation_rows_json": rel(rows_json),
        "activation_rows_json_sha256": sha256_file(rows_json),
        "guarded_transaction_rows_csv": rel(batch_csv),
        "guarded_transaction_rows_csv_sha256": sha256_file(batch_csv),
        "guarded_transaction_rows_json": rel(batch_json),
        "guarded_transaction_rows_json_sha256": sha256_file(batch_json),
        "markdown": rel(markdown_path),
        "acceptance_rule": (
            "Only an explicit user review record with reviewer=user and "
            "visual_review_status=VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY may unlock "
            "a guarded durable transaction preflight, and accepted manifests must include "
            "the exact top-level explicit user visual-attestation object for the current ready set; "
            "this tool still does not apply SAM2 memory."
        ),
    }
    write_markdown(markdown_path, summary, activation_rows)
    summary["markdown_sha256"] = sha256_file(markdown_path)
    write_json(summary_path, summary)

    print(
        json.dumps(
            {
                "summary": rel(summary_path),
                "status": status,
                "ready_row_count": int(counts["ready_row_count"]),
                "accepted_ready_count": int(counts["accepted_ready_count"]),
                "guarded_transaction_request_count": int(len(requests)),
                "durable_memory_mutation_request_emitted": False,
                "sam2_memory_mutation_applied": False,
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
