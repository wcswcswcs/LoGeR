from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from Stream3D.stream4d_v108.artifacts import ArtifactRecord, ArtifactWriter, now_utc, sha256_file
from Stream3D.stream4d_v108.diagnostics import ReviewStatus


EVENT_TYPES = (
    "new_gap_hypothesis",
    "repair_existing_suspicion",
    "active_growth_alert",
    "dormant_visibility_suspicion",
    "reactivation_candidate",
    "transaction_suggestion",
)


@dataclass(frozen=True)
class OnlineEventRow:
    schema_version: str
    case_name: str
    event_id: str
    frame_id: int
    chunk_frame_index: int
    event_type: str
    global_object_id: int | None
    sam2_runtime_object_id: int | None
    evidence_status: str
    review_status: str
    diagnostic_only: bool
    reference_label_used: bool
    output_mutation_allowed: bool
    memory_mutation_allowed: bool
    source: str
    reason: str
    score_name: str
    score_value: float | None
    evidence_json: str


def _event(
    *,
    case_name: str,
    index: int,
    frame_id: int,
    chunk_frame_index: int,
    event_type: str,
    source: str,
    reason: str,
    evidence: dict[str, Any],
    score_name: str = "",
    score_value: float | None = None,
    global_object_id: int | None = None,
    sam2_runtime_object_id: int | None = None,
) -> OnlineEventRow:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown online event type: {event_type}")
    return OnlineEventRow(
        schema_version="stream4d_v108_online_event_row_v1",
        case_name=case_name,
        event_id=f"{case_name}_event_{index:06d}",
        frame_id=int(frame_id),
        chunk_frame_index=int(chunk_frame_index),
        event_type=event_type,
        global_object_id=global_object_id,
        sam2_runtime_object_id=sam2_runtime_object_id,
        evidence_status="RECORDED_NOT_ACCEPTED",
        review_status=ReviewStatus.USER_REVIEW_PENDING.value,
        diagnostic_only=True,
        reference_label_used=False,
        output_mutation_allowed=False,
        memory_mutation_allowed=False,
        source=source,
        reason=reason,
        score_name=score_name,
        score_value=score_value,
        evidence_json=json.dumps(evidence, sort_keys=True),
    )


def _frame_diag_by_index(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["chunk_frame_index"]): row for row in summary.get("frame_diagnostics", []) if "chunk_frame_index" in row}


def _birth_rows_by_index(summary: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    rolling = summary.get("v106_sam2_rolling_state", {})
    for row in rolling.get("post_start_birth_filter_records", []):
        out.setdefault(int(row.get("frame_idx", -1)), []).append(row)
    return out


def emit_online_events(summary: dict[str, Any], case_name: str) -> list[OnlineEventRow]:
    rows: list[OnlineEventRow] = []
    frame_diags = _frame_diag_by_index(summary)
    birth_by_index = _birth_rows_by_index(summary)
    records = sorted(summary.get("records", []), key=lambda row: int(row.get("chunk_frame_index", 0)))
    previous: dict[str, Any] | None = None

    def add(**kwargs: Any) -> None:
        rows.append(_event(case_name=case_name, index=len(rows), **kwargs))

    for record in records:
        frame_id = int(record["frame_id"])
        chunk_idx = int(record.get("chunk_frame_index", -1))
        object_count = int(record.get("object_id_count", 0))
        visible_count = int(record.get("visible_id_count", 0))
        foreground_ratio = float(record.get("foreground_ratio", 0.0))
        diag = frame_diags.get(chunk_idx, {})

        gap_mask_count = int(diag.get("gap_mask_count", 0))
        if gap_mask_count > 0:
            add(
                frame_id=frame_id,
                chunk_frame_index=chunk_idx,
                event_type="new_gap_hypothesis",
                source="frame_diagnostics.gap_mask_count",
                reason="gap masks were proposed by online rolling state",
                score_name="gap_mask_count",
                score_value=float(gap_mask_count),
                evidence={
                    "gap_mask_count": gap_mask_count,
                    "uncovered_ratio_before_gap": diag.get("uncovered_ratio_before_gap"),
                    "gap_runtime_sec": diag.get("gap_runtime_sec"),
                    "reference_labels_used": False,
                },
            )

        stream_pruned = diag.get("stream_pruned_object_ids") or []
        oversized_events = diag.get("stream_oversized_prune_events") or []
        if stream_pruned or oversized_events:
            add(
                frame_id=frame_id,
                chunk_frame_index=chunk_idx,
                event_type="repair_existing_suspicion",
                source="frame_diagnostics.stream_prune",
                reason="rolling state pruned object ids or oversized masks",
                score_name="stream_pruned_object_count",
                score_value=float(len(stream_pruned)),
                evidence={
                    "stream_pruned_object_ids": stream_pruned,
                    "stream_oversized_prune_events": oversized_events,
                    "reference_labels_used": False,
                },
            )

        for birth_row in birth_by_index.get(chunk_idx, []):
            input_mask_count = int(birth_row.get("input_mask_count", 0))
            new_mask_count = int(birth_row.get("new_mask_count", 0))
            admitted = int(birth_row.get("admitted_mask_count", 0))
            skipped = int(birth_row.get("skipped_mask_count", 0))
            if input_mask_count > 0 or new_mask_count > 0:
                add(
                    frame_id=frame_id,
                    chunk_frame_index=chunk_idx,
                    event_type="transaction_suggestion",
                    source="v106_sam2_rolling_state.post_start_birth_filter_records",
                    reason="birth filter saw candidate masks; Phase2 records suggestion only",
                    score_name="new_mask_count",
                    score_value=float(new_mask_count),
                    evidence={
                        "input_mask_count": input_mask_count,
                        "new_mask_count": new_mask_count,
                        "admitted_mask_count": admitted,
                        "skipped_mask_count": skipped,
                        "admitted_obj_ids": birth_row.get("admitted_obj_ids", []),
                        "skipped_obj_ids": birth_row.get("skipped_obj_ids", []),
                        "input_areas": birth_row.get("input_areas", []),
                        "input_edge_touch_counts": birth_row.get("input_edge_touch_counts", []),
                        "input_core16_areas": birth_row.get("input_core16_areas", []),
                        "reference_labels_used": False,
                    },
                )

        if previous is not None:
            prev_visible = int(previous.get("visible_id_count", 0))
            prev_object = int(previous.get("object_id_count", 0))
            prev_fg = float(previous.get("foreground_ratio", 0.0))
            visible_delta = visible_count - prev_visible
            object_delta = object_count - prev_object
            fg_delta = foreground_ratio - prev_fg

            if fg_delta >= 0.25 or object_delta >= 4:
                add(
                    frame_id=frame_id,
                    chunk_frame_index=chunk_idx,
                    event_type="active_growth_alert",
                    source="records.foreground_and_object_count_delta",
                    reason="large foreground or object-count increase in online output",
                    score_name="foreground_ratio_delta",
                    score_value=float(fg_delta),
                    evidence={
                        "previous_frame_id": previous.get("frame_id"),
                        "previous_foreground_ratio": prev_fg,
                        "current_foreground_ratio": foreground_ratio,
                        "object_count_delta": object_delta,
                        "visible_count_delta": visible_delta,
                        "reference_labels_used": False,
                    },
                )

            if visible_delta <= -4:
                add(
                    frame_id=frame_id,
                    chunk_frame_index=chunk_idx,
                    event_type="dormant_visibility_suspicion",
                    source="records.visible_id_count_delta",
                    reason="visible object count dropped sharply",
                    score_name="visible_id_count_delta",
                    score_value=float(visible_delta),
                    evidence={
                        "previous_frame_id": previous.get("frame_id"),
                        "previous_visible_id_count": prev_visible,
                        "current_visible_id_count": visible_count,
                        "reference_labels_used": False,
                    },
                )

            if visible_delta >= 4:
                add(
                    frame_id=frame_id,
                    chunk_frame_index=chunk_idx,
                    event_type="reactivation_candidate",
                    source="records.visible_id_count_delta",
                    reason="visible object count recovered sharply",
                    score_name="visible_id_count_delta",
                    score_value=float(visible_delta),
                    evidence={
                        "previous_frame_id": previous.get("frame_id"),
                        "previous_visible_id_count": prev_visible,
                        "current_visible_id_count": visible_count,
                        "reference_labels_used": False,
                    },
                )

        previous = record
    return rows


def summarize_events(rows: Iterable[OnlineEventRow], *, case_name: str, source_summary_path: Path) -> dict[str, Any]:
    rows_list = list(rows)
    counts = {event_type: 0 for event_type in EVENT_TYPES}
    for row in rows_list:
        counts[row.event_type] += 1
    return {
        "schema_version": "stream4d_v108_online_event_summary_v1",
        "case_name": case_name,
        "source_summary_path": source_summary_path.as_posix(),
        "source_summary_sha256": sha256_file(source_summary_path),
        "event_count": len(rows_list),
        "event_type_counts": counts,
        "reference_label_used_count": sum(1 for row in rows_list if row.reference_label_used),
        "output_mutation_allowed_count": sum(1 for row in rows_list if row.output_mutation_allowed),
        "memory_mutation_allowed_count": sum(1 for row in rows_list if row.memory_mutation_allowed),
        "all_events_shadow_only": all(
            row.diagnostic_only
            and not row.reference_label_used
            and not row.output_mutation_allowed
            and not row.memory_mutation_allowed
            and row.review_status == ReviewStatus.USER_REVIEW_PENDING.value
            for row in rows_list
        ),
    }


def write_events_parquet(rows: Iterable[OnlineEventRow], path: Path) -> ArtifactRecord:
    import pandas as pd

    rows_list = [row.__dict__ for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_list).to_parquet(path, index=False)
    return ArtifactRecord(
        path=path.as_posix(),
        schema_version="stream4d_v108_online_event_rows_parquet_v1",
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        created_at_utc=now_utc(),
    )


def write_event_casebook(rows: list[OnlineEventRow], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    by_type: dict[str, list[OnlineEventRow]] = {event_type: [] for event_type in EVENT_TYPES}
    for row in rows:
        by_type[row.event_type].append(row)
    for event_type, event_rows in by_type.items():
        if not event_rows:
            continue
        selected = sorted(
            event_rows,
            key=lambda row: abs(row.score_value) if row.score_value is not None else 0.0,
            reverse=True,
        )[:5]
        payload = {
            "schema_version": "stream4d_v108_event_casebook_v1",
            "event_type": event_type,
            "review_status": ReviewStatus.USER_REVIEW_PENDING.value,
            "rows": [row.__dict__ for row in selected],
        }
        path = output_dir / f"{event_type}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path.as_posix())
    return paths


def reference_path_access_audit(*, source_summary_path: Path, accepted_reference_inputs: list[str]) -> dict[str, Any]:
    forbidden = [path for path in accepted_reference_inputs if path]
    return {
        "schema_version": "stream4d_v108_reference_path_access_audit_v1",
        "source_summary_path": source_summary_path.as_posix(),
        "accepted_reference_input_count": len(forbidden),
        "accepted_reference_inputs": forbidden,
        "online_event_generation_accepts_reference_paths": False,
        "reference_label_path_read_count": 0,
        "reference_label_used_for_event_count": 0,
        "status": "PASS_NO_REFERENCE_INPUTS" if not forbidden else "FAIL_REFERENCE_INPUTS_PROVIDED",
    }


def write_online_event_artifacts(
    *,
    case_name: str,
    source_summary_path: Path,
    output_root: Path,
    accepted_reference_inputs: list[str] | None = None,
) -> dict[str, Any]:
    source_summary_path = Path(source_summary_path)
    output_root = Path(output_root)
    summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    rows = emit_online_events(summary, case_name)
    writer = ArtifactWriter(output_root)

    parquet_record = write_events_parquet(rows, output_root / "online_event_rows.parquet")
    writer.records.append(parquet_record)
    writer.write_jsonl("online_event_rows.jsonl", rows, "stream4d_v108_online_event_rows_jsonl_v1")
    casebook_paths = write_event_casebook(rows, output_root / "event_casebook")
    for casebook_path in casebook_paths:
        path = Path(casebook_path)
        writer.records.append(
            ArtifactRecord(
                path=path.as_posix(),
                schema_version="stream4d_v108_event_casebook_v1",
                sha256=sha256_file(path),
                byte_size=path.stat().st_size,
                created_at_utc=now_utc(),
            )
        )
    audit = reference_path_access_audit(
        source_summary_path=source_summary_path,
        accepted_reference_inputs=accepted_reference_inputs or [],
    )
    writer.write_json("reference_path_access_audit.json", audit, "stream4d_v108_reference_path_access_audit_v1")
    event_summary = summarize_events(rows, case_name=case_name, source_summary_path=source_summary_path)
    event_summary["casebook_paths"] = casebook_paths
    event_summary["reference_path_access_audit"] = audit
    event_summary["artifact_manifest"] = writer.manifest()
    writer.write_json("online_event_summary.json", event_summary, "stream4d_v108_online_event_summary_v1")
    return event_summary
