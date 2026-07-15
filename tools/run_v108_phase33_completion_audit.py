#!/usr/bin/env python3
"""Audit the current v108 completion boundary without activating memory.

This is a read-only completion audit. It joins the Phase19 ready rows, the
Phase32 handoff packet, the Phase31 manifest verifier, and the Phase30 policy
flow audit into one evidence chain. It never builds transaction requests and
never applies SAM2 memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
READY_STATUS = "READY_EXCEPT_USER_VISUAL_ACCEPTANCE"
PHASE31_READY_STATUS = "REVIEW_MANIFEST_READY_FOR_PHASE20_PREFLIGHT"
PHASE30_PASS_STATUS = "POLICY_FLAG_FLOW_AUDIT_PASS"


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_int(value: Any, default: int = -1) -> int:
    try:
        text = str(value).strip()
        if not text:
            return int(default)
        return int(float(text))
    except Exception:
        return int(default)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def contact_sheet_pair(row: dict[str, Any]) -> tuple[str, str]:
    item = dict(row.get("review_contact_sheet") or {})
    return str(item.get("path", "")), str(item.get("sha256", ""))


def load_phase19_ready_rows(phase19_root: Path) -> list[dict[str, Any]]:
    payload = read_json(phase19_root / "phase19_review_candidate_rows.json")
    rows = [dict(row) for row in payload.get("records", [])]
    ready_rows = [
        row
        for row in rows
        if row.get("source_label") == "baseline" and row.get("preflight_status") == READY_STATUS
    ]
    ready_rows.sort(key=lambda row: (parse_int(row.get("frame_id"), 999999), parse_int(row.get("event_index"), 999999)))
    return ready_rows


def row_identity(row: dict[str, Any]) -> dict[str, Any]:
    contact_path, contact_sha = contact_sheet_pair(row)
    return {
        "scene_id": str(row.get("scene_id", "")),
        "event_index": parse_int(row.get("event_index")),
        "frame_id": parse_int(row.get("frame_id")),
        "live_obj_id": parse_int(row.get("live_obj_id")),
        "reference_obj_id": parse_int(row.get("reference_obj_id")),
        "contact_sheet": contact_path,
        "contact_sheet_sha256": contact_sha,
    }


def add_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    detail: str,
    evidence: dict[str, Any] | None = None,
    status_if_false: str = "FAIL",
) -> None:
    checks.append(
        {
            "name": name,
            "status": "PASS" if passed else status_if_false,
            "detail": detail,
            "evidence": evidence or {},
        }
    )


def status_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        status = str(check.get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_markdown(path: Path, summary: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    lines = [
        "# Stream4D v108 Phase33 Completion Audit",
        "",
        f"completion_status: {summary['completion_status']}",
        f"goal_achieved: {str(summary['goal_achieved']).lower()}",
        f"phase20_preflight_may_be_run: {str(summary['phase20_preflight_may_be_run']).lower()}",
        "durable_memory_mutation_request_emitted: false",
        "transaction_preflight_constructed: false",
        "sam2_memory_mutation_applied: false",
        "metrics_are_diagnostic_only: true",
        "",
        "## Evidence Chain",
        "",
    ]
    for check in checks:
        lines.extend(
            [
                f"### {check['name']}",
                "",
                f"- status: {check['status']}",
                f"- detail: {check['detail']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "",
            summary["conclusion"],
            "",
            "## Next Safe Action",
            "",
            summary["next_safe_action"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase19-root",
        default="Stream3D/outputs/audit/v108_phase19_review_preflight_scene0050_full99_20260715_0250",
    )
    parser.add_argument(
        "--phase32-root",
        default="Stream3D/outputs/audit/v108_phase32_user_review_handoff_scene0050_full99_20260715_0525",
    )
    parser.add_argument(
        "--output-root",
        default="Stream3D/outputs/audit/v108_phase33_completion_audit_scene0050_full99_20260715_0520",
    )
    parser.add_argument(
        "--phase31-summary",
        default="",
        help="Optional fresh Phase31 summary; defaults to the Phase32 pending-manifest verification.",
    )
    parser.add_argument(
        "--phase30-summary",
        default="",
        help="Optional fresh Phase30 summary; defaults to the Phase32 policy-flow audit.",
    )
    parser.add_argument("--plan", default="docs/stream4d_v108_dualplane_lifecycle_physical_gap_plan.md")
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
    phase32_root = resolve_path(args.phase32_root)
    plan_path = resolve_path(args.plan)
    phase32_summary_path = phase32_root / "phase32_user_review_handoff_summary.json"
    phase32_rows_path = phase32_root / "phase32_user_review_handoff_rows.json"
    phase31_summary_path = (
        resolve_path(args.phase31_summary)
        if str(args.phase31_summary).strip()
        else phase32_root / "phase31_verify_pending_handoff_manifest" / "phase31_review_manifest_verifier_summary.json"
    )
    phase30_summary_path = (
        resolve_path(args.phase30_summary)
        if str(args.phase30_summary).strip()
        else phase32_root / "policy_flow_audit_after_phase32" / "phase30_policy_flag_flow_summary.json"
    )

    checks: list[dict[str, Any]] = []
    plan_exists = plan_path.is_file()
    add_check(
        checks,
        name="plan file present",
        passed=plan_exists,
        detail="v108 plan is available for the completion boundary audit",
        evidence={"plan": rel(plan_path), "sha256": sha256_file(plan_path) if plan_exists else ""},
    )

    ready_rows = load_phase19_ready_rows(phase19_root)
    ready_identities = [row_identity(row) for row in ready_rows]
    add_check(
        checks,
        name="Phase19 current ready rows loaded",
        passed=len(ready_rows) == 2,
        detail="Phase19 current baseline ready rows are the rows still requiring user visual review",
        evidence={"phase19_root": rel(phase19_root), "ready_row_count": len(ready_rows), "ready_rows": ready_identities},
    )

    phase32_summary = read_json(phase32_summary_path)
    phase32_rows_payload = read_json(phase32_rows_path)
    phase32_rows = [dict(row) for row in phase32_rows_payload.get("records", [])]
    phase31_summary = read_json(phase31_summary_path)
    phase30_summary = read_json(phase30_summary_path)

    phase32_ready_identity = [
        {
            "event_index": parse_int(row.get("event_index")),
            "frame_id": parse_int(row.get("frame_id")),
            "live_obj_id": parse_int(row.get("live_obj_id")),
            "reference_obj_id": parse_int(row.get("reference_obj_id")),
            "contact_sheet_sha256": str(row.get("contact_sheet_sha256", "")),
        }
        for row in phase32_rows
    ]
    expected_identity = [
        {
            "event_index": row["event_index"],
            "frame_id": row["frame_id"],
            "live_obj_id": row["live_obj_id"],
            "reference_obj_id": row["reference_obj_id"],
            "contact_sheet_sha256": row["contact_sheet_sha256"],
        }
        for row in ready_identities
    ]
    add_check(
        checks,
        name="Phase32 handoff matches Phase19 ready rows",
        passed=phase32_ready_identity == expected_identity,
        detail="The handoff rows are byte-hash bound to the same current ready contact sheets",
        evidence={
            "phase32_summary": rel(phase32_summary_path),
            "phase32_status": phase32_summary.get("status", ""),
            "phase32_ready_row_count": phase32_summary.get("ready_row_count"),
            "phase32_rows": phase32_ready_identity,
        },
    )

    asset_evidence: list[dict[str, Any]] = []
    all_asset_hashes_match = True
    for row in phase32_rows:
        asset = resolve_path(str(row.get("handoff_asset_path", "")))
        asset_exists = asset.is_file()
        asset_sha = sha256_file(asset) if asset_exists else ""
        expected_sha = str(row.get("contact_sheet_sha256", ""))
        match = bool(asset_exists and asset_sha == expected_sha and parse_bool(row.get("copy_sha256_matches_source")))
        all_asset_hashes_match = bool(all_asset_hashes_match and match)
        asset_evidence.append(
            {
                "event_index": parse_int(row.get("event_index")),
                "asset": rel(asset),
                "exists": asset_exists,
                "asset_sha256": asset_sha,
                "expected_contact_sheet_sha256": expected_sha,
                "copy_sha256_matches_source": parse_bool(row.get("copy_sha256_matches_source")),
                "match": match,
            }
        )
    add_check(
        checks,
        name="Phase32 copied visual assets hash-match sources",
        passed=all_asset_hashes_match and bool(phase32_rows),
        detail="The locally copied review PNGs match the source contact-sheet hashes",
        evidence={"assets": asset_evidence},
    )

    phase31_status = str(phase31_summary.get("status", ""))
    phase31_allows_phase20 = parse_bool(phase31_summary.get("phase20_preflight_may_be_run"))
    phase31_pending_only = bool(
        phase31_status == "REVIEW_MANIFEST_TEMPLATE_ONLY_OR_NOT_REAL_ACCEPTANCE"
        and not phase31_allows_phase20
        and parse_int(phase31_summary.get("accepted_ready_count"), 0) == 0
        and parse_int(phase31_summary.get("pending_ready_count"), 0) == len(ready_rows)
    )
    add_check(
        checks,
        name="Phase31 pending handoff manifest fails closed",
        passed=phase31_pending_only,
        detail="The pending/template handoff manifest is not accepted as real durable user review",
        evidence={
            "phase31_summary": rel(phase31_summary_path),
            "status": phase31_status,
            "accepted_ready_count": phase31_summary.get("accepted_ready_count"),
            "pending_ready_count": phase31_summary.get("pending_ready_count"),
            "phase20_preflight_may_be_run": phase31_summary.get("phase20_preflight_may_be_run"),
        },
    )

    phase30_pass = str(phase30_summary.get("status", "")) == PHASE30_PASS_STATUS
    add_check(
        checks,
        name="Phase30 policy flag flow audit remains clean",
        passed=phase30_pass,
        detail="Policy helper callsites did not expose unsafe raw attestation/constant-true flow",
        evidence={
            "phase30_summary": rel(phase30_summary_path),
            "status": phase30_summary.get("status", ""),
            "policy_callsite_count": phase30_summary.get("policy_callsite_count"),
            "unsafe_policy_flag_call_count": phase30_summary.get("unsafe_policy_flag_call_count"),
            "parse_error_count": phase30_summary.get("parse_error_count"),
        },
    )

    flags_false = all(
        not parse_bool(payload.get(field))
        for payload in (phase32_summary, phase31_summary)
        for field in (
            "durable_memory_mutation_request_emitted",
            "transaction_preflight_constructed",
            "sam2_memory_mutation_applied",
        )
    )
    add_check(
        checks,
        name="No durable memory side effects emitted by current handoff path",
        passed=flags_false,
        detail="Phase31 and Phase32 summaries both report no transaction construction or SAM2 memory mutation",
        evidence={
            "phase32_flags": {
                "durable_memory_mutation_request_emitted": phase32_summary.get("durable_memory_mutation_request_emitted"),
                "transaction_preflight_constructed": phase32_summary.get("transaction_preflight_constructed"),
                "sam2_memory_mutation_applied": phase32_summary.get("sam2_memory_mutation_applied"),
            },
            "phase31_flags": {
                "durable_memory_mutation_request_emitted": phase31_summary.get("durable_memory_mutation_request_emitted"),
                "transaction_preflight_constructed": phase31_summary.get("transaction_preflight_constructed"),
                "sam2_memory_mutation_applied": phase31_summary.get("sam2_memory_mutation_applied"),
            },
        },
    )

    user_acceptance_verified = bool(phase31_status == PHASE31_READY_STATUS and phase31_allows_phase20)
    add_check(
        checks,
        name="Explicit user visual acceptance present",
        passed=user_acceptance_verified,
        detail="No exact-scope user-supplied accepted manifest has passed Phase31 for the current ready rows",
        evidence={
            "required_ready_evidence_fingerprint_sha256": phase31_summary.get(
                "required_ready_evidence_fingerprint_sha256", ""
            ),
            "review_manifest": phase31_summary.get("review_manifest", ""),
            "review_manifest_sha256": phase31_summary.get("review_manifest_sha256", ""),
            "phase31_status": phase31_status,
        },
        status_if_false="BLOCKED",
    )

    counts = status_counts(checks)
    evidence_fail_count = counts.get("FAIL", 0)
    blocked_count = counts.get("BLOCKED", 0)
    if evidence_fail_count:
        completion_status = "COMPLETION_AUDIT_EVIDENCE_INCOMPLETE_OR_INCONSISTENT"
        conclusion = (
            "The completion audit found internal evidence gaps or inconsistencies. Do not run Phase20 "
            "until those FAIL rows are repaired and the audit is rerun."
        )
    elif blocked_count:
        completion_status = "NOT_ACHIEVED_USER_VISUAL_ACCEPTANCE_MISSING"
        conclusion = (
            "All current local handoff and guard evidence is coherent, but the decisive user visual "
            "acceptance gate is still absent. Codex visual inspection, hashes, metrics, templates, "
            "and policy audits are diagnostic only and cannot be promoted to VISUAL_PASS."
        )
    else:
        completion_status = "READY_FOR_PHASE20_PREFLIGHT_AFTER_USER_ACCEPTANCE"
        conclusion = (
            "A real accepted user visual-review manifest appears to have passed Phase31. The next "
            "step would be Phase20 preflight, still without claiming SAM2 mutation success."
        )

    phase20_preflight_may_be_run = bool(not evidence_fail_count and not blocked_count and user_acceptance_verified)
    evidence_path = output_root / "phase33_completion_audit_evidence_chain.json"
    markdown_path = output_root / "phase33_completion_audit.md"
    summary_path = output_root / "phase33_completion_audit_summary.json"
    summary: dict[str, Any] = {
        "schema_version": "stream4d_v108_phase33_completion_audit_summary_v1",
        "completion_status": completion_status,
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "goal_achieved": bool(completion_status == "READY_FOR_PHASE20_PREFLIGHT_AFTER_USER_ACCEPTANCE"),
        "phase20_preflight_may_be_run": phase20_preflight_may_be_run,
        "durable_memory_mutation_request_emitted": False,
        "transaction_preflight_constructed": False,
        "sam2_memory_mutation_applied": False,
        "metrics_are_diagnostic_only": True,
        "visual_acceptance_claimed_by_codex": False,
        "phase19_root": rel(phase19_root),
        "phase32_root": rel(phase32_root),
        "phase31_summary": rel(phase31_summary_path),
        "phase30_summary": rel(phase30_summary_path),
        "ready_row_count": len(ready_rows),
        "ready_rows": ready_identities,
        "check_status_counts": counts,
        "checks": checks,
        "conclusion": conclusion,
        "next_safe_action": (
            "Have the user visually review the two current contact sheets under the exact hashes in "
            "this audit. If and only if the user supplies an exact-scope accepted manifest, rerun "
            "Phase31 first and pass into Phase20 only when phase20_preflight_may_be_run=true."
        ),
    }
    write_json(evidence_path, {"schema_version": "stream4d_v108_phase33_completion_audit_evidence_chain_v1", "checks": checks})
    summary["evidence_chain"] = rel(evidence_path)
    summary["evidence_chain_sha256"] = sha256_file(evidence_path)
    write_markdown(markdown_path, summary, checks)
    summary["markdown"] = rel(markdown_path)
    summary["markdown_sha256"] = sha256_file(markdown_path)
    write_json(summary_path, summary)

    print(
        json.dumps(
            {
                "summary": rel(summary_path),
                "completion_status": completion_status,
                "goal_achieved": summary["goal_achieved"],
                "phase20_preflight_may_be_run": phase20_preflight_may_be_run,
                "check_status_counts": counts,
                "durable_memory_mutation_request_emitted": False,
                "transaction_preflight_constructed": False,
                "sam2_memory_mutation_applied": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
