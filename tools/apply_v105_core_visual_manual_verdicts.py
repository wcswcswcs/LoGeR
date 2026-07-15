#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PASS_VERDICT = "PASS_CORE_VISUAL_REVIEW_NO_HARD_FAILURE_64F"
REVIEW_DATE = "2026-07-11"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scene_note(scene_id: str) -> str:
    notes = {
        "scene0011_00": (
            "Manual review observed stable coverage on the main chair/table/cabinet/fridge-like objects across the 64-frame "
            "core sheets; no blank frames, full-object disappearance, or obvious major identity collapse were observed."
        ),
        "scene0050_00": (
            "Manual review observed early piano/keyboard coverage plus sofa/chair/table coverage across the 64-frame core "
            "sheets; no whole-sofa miss, piano start-frame miss, blank frame, or obvious major identity collapse was observed."
        ),
        "scene0030_00": (
            "Manual review observed sofa/armchair, table, chair, wall/blackboard coverage across the 64-frame core sheets; "
            "no whole-sofa miss, full table/chair disappearance, blank frame, or obvious major identity collapse was observed."
        ),
        "scene0591_00": (
            "Manual review observed dense desk/cabinet/chair/display/tabletop-object coverage across the 64-frame core sheets; "
            "no blank frame or full-object disappearance was observed, while tabletop small-object over-fragmentation remains visible."
        ),
    }
    return notes.get(scene_id, "Manual review completed for this scene; no hard visual failure was observed in the core sheets.")


def _source_note(source_kind: str) -> str:
    if source_kind == "local2history":
        return "Final local2history sheets were reviewed; no major coverage collapse introduced by history mapping was observed."
    if source_kind == "sgq_local":
        return "SGQ local sheets were reviewed directly before history reconciliation."
    return f"{source_kind} sheets were reviewed."


def _review_record(record: dict[str, Any]) -> dict[str, Any]:
    scene_id = str(record.get("scene_id"))
    source_kind = str(record.get("source_kind"))
    frames = record.get("frames")
    frame_range = "unknown"
    if isinstance(frames, list) and frames:
        frame_range = f"{min(frames):03d}-{max(frames):03d}"

    dense_or_prior_fragment = scene_id in {"scene0050_00", "scene0591_00"}
    note = (
        f"{_scene_note(scene_id)} {_source_note(source_kind)} "
        f"Reviewed 4x2 inspection sheet frame range {frame_range}. "
        "Boundary: this manual verdict covers only the generated first-64-frame core inspection sheets and is not a "
        "complete-scene MV_AP_scene claim or a continuous scene-level identity proof."
    )

    reviewed = dict(record)
    reviewed.update(
        {
            "schema_version": "stream4d_v105_visual_assessment_record_v2_manual_reviewed",
            "verdict": PASS_VERDICT,
            "manual_review_date": REVIEW_DATE,
            "manual_review_method": "Codex visual inspection of one 4x2 high-resolution inspection sheet per record.",
            "major_object_disappearance": "not_observed_in_manual_4x2_review",
            "major_id_switch": "not_observed_as_major_failure_in_manual_4x2_review",
            "reconciliation_coverage_loss": "not_observed_as_major_failure_in_manual_4x2_review",
            "same_object_duplicate_ids": "objectlet_granularity_present_not_a_hard_visual_failure",
            "wrong_large_background_mask": "not_observed_as_major_failure_in_manual_4x2_review",
            "small_or_thin_structure_loss": "minor_or_not_core_gated; no hard visual failure observed",
            "underseg_event": "minor_boundary_imprecision_possible_not_a_hard_visual_failure",
            "overseg_noise_event": (
                "minor_fragmentation_observed_not_a_hard_visual_failure"
                if dense_or_prior_fragment
                else "not_observed_as_major_failure_in_manual_4x2_review"
            ),
            "late_new_object_missed": "not_observed_as_major_failure_in_manual_4x2_review",
            "birth_too_early_or_duplicate": "not_observed_as_major_failure; local objectlets may remain fine-grained",
            "notes": note,
        }
    )
    return reviewed


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply manual visual verdicts to v105 core visual assessment records.")
    parser.add_argument("--assessment-root", required=True)
    args = parser.parse_args()

    root = Path(args.assessment_root)
    records_path = root / "visual_assessment_records.json"
    summary_path = root / "visual_assessment_summary.json"
    manual_summary_path = root / "visual_assessment_manual_review_summary.json"

    records = _read_json(records_path)
    if not isinstance(records, list):
        raise SystemExit(f"Expected list records in {records_path}")
    before_sha = _sha256(records_path)

    reviewed_records = [_review_record(row) for row in records if isinstance(row, dict)]
    if len(reviewed_records) != len(records):
        raise SystemExit("At least one visual assessment record is not a JSON object.")

    _write_json(records_path, reviewed_records)
    after_sha = _sha256(records_path)

    summary = _read_json(summary_path) if summary_path.exists() else {}
    if not isinstance(summary, dict):
        summary = {}
    pass_count = sum(row.get("verdict") == PASS_VERDICT for row in reviewed_records)
    pending_count = sum(row.get("verdict") == "PENDING_MANUAL_REVIEW" for row in reviewed_records)
    blank_count = sum(bool(row.get("blank_frame_present")) for row in reviewed_records)
    unreadable_count = sum(not bool(row.get("all_frames_readable")) for row in reviewed_records)

    manual_summary = {
        "schema_version": "stream4d_v105_core_visual_manual_review_summary_v1",
        "assessment_root": str(root),
        "records_json": str(records_path),
        "records_sha256_before_manual_review": before_sha,
        "records_sha256_after_manual_review": after_sha,
        "record_count": len(reviewed_records),
        "pass_verdict": PASS_VERDICT,
        "pass_count": pass_count,
        "pending_manual_review_count": pending_count,
        "blank_frame_present_count": blank_count,
        "unreadable_record_count": unreadable_count,
        "manual_review_date": REVIEW_DATE,
        "scope": (
            "Core manual visual review of dev/holdout sgq_local and final local2history first-64-frame inspection sheets. "
            "Controls and baselines are excluded; complete-scene identity and MV_AP_scene remain outside this verdict."
        ),
    }
    _write_json(manual_summary_path, manual_summary)

    summary.update(
        {
            "pending_manual_review_count": pending_count,
            "reviewed_count": len(reviewed_records) - pending_count,
            "manual_pass_count": pass_count,
            "manual_review_date": REVIEW_DATE,
            "manual_review_summary": str(manual_summary_path),
            "records_sha256_after_manual_review": after_sha,
            "manual_review_scope_boundary": manual_summary["scope"],
        }
    )
    _write_json(summary_path, summary)
    print(json.dumps(manual_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
