from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class VisualReviewStatus(str, Enum):
    USER_REVIEW_PENDING = "USER_REVIEW_PENDING"
    VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY = "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY"
    DIAGNOSTIC_VISUAL_STABLE_BUT_PENDING = "DIAGNOSTIC_VISUAL_STABLE_BUT_PENDING"
    DIAGNOSTIC_VISUAL_OUTPUT_ONLY = "DIAGNOSTIC_VISUAL_OUTPUT_ONLY"
    DIAGNOSTIC_VISUAL_NOT_DURABLE = "DIAGNOSTIC_VISUAL_NOT_DURABLE"


@dataclass(frozen=True)
class VisualReviewRecord:
    scene_id: str
    object_id: int
    frame_id: int
    visual_review_status: str
    visual_note: str
    evidence_paths: tuple[str, ...]
    evidence_sha256: tuple[str, ...]
    reviewer: str = "codex_diagnostic_visual_review"

    @property
    def durable_review_accepted(self) -> bool:
        return self.visual_review_status == VisualReviewStatus.VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY.value

    def validate(self) -> None:
        valid = {item.value for item in VisualReviewStatus}
        if self.visual_review_status not in valid:
            raise ValueError({"invalid_visual_review_status": self.visual_review_status, "valid": sorted(valid)})
        if not self.visual_note.strip():
            raise ValueError("visual_note is required")
        if len(self.evidence_paths) != len(self.evidence_sha256):
            raise ValueError("evidence_paths and evidence_sha256 length mismatch")
        if self.durable_review_accepted and self.reviewer != "user":
            raise ValueError("durable visual acceptance is reserved for explicit user review")

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "object_id": int(self.object_id),
            "frame_id": int(self.frame_id),
            "visual_review_status": self.visual_review_status,
            "visual_note": self.visual_note,
            "evidence_paths": list(self.evidence_paths),
            "evidence_sha256": list(self.evidence_sha256),
            "reviewer": self.reviewer,
            "durable_review_accepted": bool(self.durable_review_accepted),
        }


def _record_from_dict(row: dict[str, Any]) -> VisualReviewRecord:
    evidence_paths = row.get("evidence_paths", ())
    evidence_sha256 = row.get("evidence_sha256", ())
    if isinstance(evidence_paths, str):
        evidence_paths = [evidence_paths]
    if isinstance(evidence_sha256, str):
        evidence_sha256 = [evidence_sha256]
    record = VisualReviewRecord(
        scene_id=str(row["scene_id"]),
        object_id=int(row["object_id"]),
        frame_id=int(row["frame_id"]),
        visual_review_status=str(row.get("visual_review_status", VisualReviewStatus.USER_REVIEW_PENDING.value)),
        visual_note=str(row.get("visual_note", "")),
        evidence_paths=tuple(str(item) for item in evidence_paths),
        evidence_sha256=tuple(str(item) for item in evidence_sha256),
        reviewer=str(row.get("reviewer", "codex_diagnostic_visual_review")),
    )
    record.validate()
    return record


def load_visual_review_manifest(path: Path) -> dict[tuple[str, int, int], VisualReviewRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("records", payload if isinstance(payload, list) else [])
    out: dict[tuple[str, int, int], VisualReviewRecord] = {}
    for row in rows:
        record = _record_from_dict(dict(row))
        key = (record.scene_id, int(record.object_id), int(record.frame_id))
        if key in out:
            raise ValueError(
                {
                    "duplicate_visual_review_key": {
                        "scene_id": key[0],
                        "object_id": key[1],
                        "frame_id": key[2],
                    }
                }
            )
        out[key] = record
    return out


def default_pending_review(
    *,
    scene_id: str,
    object_id: int,
    frame_id: int,
    evidence_paths: tuple[str, ...] = (),
    evidence_sha256: tuple[str, ...] = (),
) -> VisualReviewRecord:
    record = VisualReviewRecord(
        scene_id=str(scene_id),
        object_id=int(object_id),
        frame_id=int(frame_id),
        visual_review_status=VisualReviewStatus.USER_REVIEW_PENDING.value,
        visual_note="No explicit visual review record was provided.",
        evidence_paths=tuple(evidence_paths),
        evidence_sha256=tuple(evidence_sha256),
    )
    record.validate()
    return record
