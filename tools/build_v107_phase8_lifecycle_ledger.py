#!/usr/bin/env python3
"""Build an auditable v107 Phase8 lifecycle ledger from scheduler smoke records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(jsonable(row))


def opt_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    return int(float(value))


def opt_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    return float(value)


def opt_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-summary", required=True)
    parser.add_argument("--records-csv", default="")
    parser.add_argument("--rolling-summary", default="")
    parser.add_argument("--foreground-fidelity-summary", default="")
    parser.add_argument("--foreground-fidelity-rows", default="")
    parser.add_argument("--visual-casebook-manifest", default="")
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def indexed_records(rows: list[dict[str, str]]) -> dict[tuple[str, int, int], dict[str, str]]:
    out: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        record_type = str(row.get("record_type", ""))
        event_index = opt_int(row.get("event_index"), -1)
        frame_id = opt_int(row.get("frame_id"), -1)
        if event_index is None or frame_id is None:
            continue
        out[(record_type, int(event_index), int(frame_id))] = row
    return out


def build_ledger(
    *,
    scheduler_summary: dict[str, Any],
    records: list[dict[str, str]],
    rolling_summary: dict[str, Any],
    fidelity_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frame_ids = [int(v) for v in scheduler_summary.get("frame_ids", [])]
    by_key = indexed_records(records)
    mapping_rows = [row for row in records if row.get("record_type") == "source_identity_mapping"]
    events = sorted(int(row["event_index"]) for row in mapping_rows if row.get("event_index") not in {"", None})

    mapping_by_event: dict[int, dict[str, str]] = {
        int(row["event_index"]): row for row in mapping_rows if row.get("event_index") not in {"", None}
    }
    demotion_by_event: dict[int, dict[str, str]] = {}
    for row in records:
        if row.get("record_type") == "demotion" and row.get("event_index") not in {"", None}:
            demotion_by_event[int(row["event_index"])] = row

    attempt_by_event = {
        int(row["event_index"]): row for row in records if row.get("record_type") == "attempt" and row.get("event_index")
    }
    probation_by_event = {
        int(row["event_index"]): row
        for row in records
        if row.get("record_type") == "probation_attempt" and row.get("event_index")
    }
    confirm_by_event = {
        int(row["event_index"]): row for row in records if row.get("record_type") == "confirm" and row.get("event_index")
    }
    prompt_assignment_by_event = {
        int(row["event_index"]): row
        for row in records
        if row.get("record_type") == "prompt_new_object_assignment" and row.get("event_index")
    }

    lifecycle_rows: list[dict[str, Any]] = []
    state_counter: Counter[str] = Counter()
    avoided_slot_frames = 0
    shadow_output_frames = 0
    probation_output_frames = 0
    skipped_unmapped_frames = 0
    skipped_long_term_policy_frames = 0
    diagnostic_iou_values: dict[str, list[float]] = defaultdict(list)

    for event_index in events:
        mapping = mapping_by_event[event_index]
        ref_id = int(mapping["reference_global_id"])
        live_obj_id = opt_int(mapping.get("live_obj_id"))
        source_live_area = opt_int(mapping.get("source_live_area_px"), 0) or 0
        source_mapping_iou = opt_float(mapping.get("source_mapping_iou"), 0.0) or 0.0
        accepted = opt_bool(mapping.get("source_mapping_accepted"))
        long_term_admitted = opt_bool(mapping.get("long_term_memory_admitted"))
        long_term_skip_reason = str(mapping.get("long_term_admission_skip_reason", ""))
        prompt_assignment = prompt_assignment_by_event.get(event_index, {})
        prompt_only_enabled = bool(prompt_assignment)
        prompt_assignment_frame = opt_int(prompt_assignment.get("frame_id"), None)
        if live_obj_id is None and prompt_only_enabled:
            live_obj_id = opt_int(prompt_assignment.get("live_obj_id"), None)
        if not source_live_area and prompt_only_enabled:
            source_live_area = opt_int(prompt_assignment.get("target_source_area_px"), 0) or 0
        demotion = demotion_by_event.get(event_index, {})
        demotion_frame = opt_int(demotion.get("frame_id"), None)
        removed = opt_bool(demotion.get("removed"))
        attempt = attempt_by_event.get(event_index, {})
        probation = probation_by_event.get(event_index, {})
        confirm = confirm_by_event.get(event_index, {})
        attempt_frame = opt_int(attempt.get("frame_id"), opt_int(probation.get("frame_id"), None))
        confirm_frame = opt_int(confirm.get("frame_id"), None)
        source_lag = opt_int(
            attempt.get("source_lag"),
            opt_int(probation.get("source_lag"), opt_int(confirm.get("source_lag"), None)),
        )

        for frame_id in frame_ids:
            state = "ACTIVE_SAM_SOURCE"
            emits_output = True
            occupies_sam_slot = bool(accepted)
            output_source = "active_sam"
            skip_reason = ""
            metric_prefix = ""
            selected_variant = ""
            positive_support: float | None = None
            negative_overlap: float | None = None
            iou_diag: float | None = None
            runtime_sec: float | None = None

            if not accepted and prompt_only_enabled and prompt_assignment_frame is not None and frame_id >= int(
                prompt_assignment_frame
            ):
                state = "PROMPT_NEW_OBJECT_DORMANT"
                emits_output = False
                occupies_sam_slot = False
                output_source = "none"
                skip_reason = str(mapping.get("source_mapping_skip_reason", ""))
                shadow = by_key.get(("shadow_output", event_index, frame_id))
                if shadow is not None:
                    state = "PROMPT_SHADOW_2D" if opt_bool(shadow.get("output_mask")) else "PROMPT_SHADOW_SKIPPED"
                    emits_output = opt_bool(shadow.get("output_mask"))
                    occupies_sam_slot = False
                    output_source = "sam2_image_shadow_prompt_new_object" if emits_output else "none"
                    metric_prefix = "shadow"
                    selected_variant = str(shadow.get("selected_variant", ""))
                    skip_reason = str(shadow.get("skip_reason", ""))
                    iou_diag = opt_float(shadow.get("shadow_iou_to_reference"))
                    positive_support = opt_float(shadow.get("positive_point_support_rate"))
                    negative_overlap = opt_float(shadow.get("shadow_negative_sibling_overlap_rate"))
                    runtime_sec = opt_float(shadow.get("runtime_sec"))
                if probation and attempt_frame is not None and frame_id == int(attempt_frame):
                    row = probation
                    state = "PROMPT_PROBATION_2D" if opt_bool(row.get("output_mask")) else "PROMPT_PROBATION_SKIPPED"
                    emits_output = opt_bool(row.get("output_mask"))
                    occupies_sam_slot = False
                    output_source = "sam2_image_probation_prompt_new_object" if emits_output else "none"
                    metric_prefix = "probation"
                    selected_variant = str(row.get("selected_variant", ""))
                    skip_reason = str(row.get("skip_reason", ""))
                    iou_diag = opt_float(row.get("probation_iou_to_reference"))
                    positive_support = opt_float(row.get("positive_point_support_rate"))
                    negative_overlap = opt_float(row.get("probation_negative_sibling_overlap_rate"))
                    runtime_sec = opt_float(row.get("runtime_sec"))
                if confirm and confirm_frame is not None and frame_id == int(confirm_frame):
                    row = confirm
                    committed = opt_bool(row.get("reactivation_committed_to_sam2_video_state"))
                    if str(row.get("selected_variant", "")) == "SKIPPED_PROBATION_FAILED" or (
                        "reactivation_committed_to_sam2_video_state" in row and not committed
                    ):
                        state = "PROMPT_REACTIVATION_SKIPPED"
                        emits_output = False
                        occupies_sam_slot = False
                        output_source = "none"
                        skip_reason = str(row.get("skip_reason", ""))
                    else:
                        state = "PROMPT_REACTIVATING_CONFIRM"
                        emits_output = opt_bool(row.get("target_present")) or bool(row)
                        occupies_sam_slot = True
                        output_source = "sam2_video_confirm_prompt_new_object"
                    metric_prefix = "confirm"
                    selected_variant = str(row.get("selected_variant", ""))
                    iou_diag = opt_float(row.get("confirm_iou_to_reference"))
                    positive_support = opt_float(row.get("confirm_positive_point_support_rate"))
                    negative_overlap = opt_float(row.get("confirm_negative_sibling_overlap_rate"))
                    runtime_sec = opt_float(row.get("runtime_sec"))

                if state in {
                    "PROMPT_NEW_OBJECT_DORMANT",
                    "PROMPT_SHADOW_2D",
                    "PROMPT_SHADOW_SKIPPED",
                    "PROMPT_PROBATION_2D",
                    "PROMPT_PROBATION_SKIPPED",
                    "PROMPT_REACTIVATION_SKIPPED",
                }:
                    avoided_slot_frames += 1
                if state == "PROMPT_SHADOW_2D":
                    shadow_output_frames += 1
                if state == "PROMPT_PROBATION_2D":
                    probation_output_frames += 1
            elif not accepted:
                is_long_term_policy_skip = bool(long_term_skip_reason)
                state = "SKIPPED_LONG_TERM_POLICY" if is_long_term_policy_skip else "SKIPPED_UNMAPPED"
                emits_output = False
                occupies_sam_slot = False
                output_source = "none"
                skip_reason = long_term_skip_reason or str(mapping.get("source_mapping_skip_reason", ""))
                if is_long_term_policy_skip:
                    skipped_long_term_policy_frames += int(frame_id != frame_ids[0])
                else:
                    skipped_unmapped_frames += int(frame_id != frame_ids[0])
            elif demotion_frame is not None and frame_id >= int(demotion_frame):
                occupies_sam_slot = False
                emits_output = False
                output_source = "none"
                state = "DORMANT_NO_OUTPUT"
                if probation and attempt_frame is not None and frame_id == int(attempt_frame):
                    row = probation
                    state = "PROBATION_2D" if opt_bool(row.get("output_mask")) else "PROBATION_SKIPPED"
                    emits_output = opt_bool(row.get("output_mask"))
                    occupies_sam_slot = False
                    output_source = "sam2_image_probation_prompt" if emits_output else "none"
                    metric_prefix = "probation"
                    selected_variant = str(row.get("selected_variant", ""))
                    skip_reason = str(row.get("skip_reason", ""))
                    iou_diag = opt_float(row.get("probation_iou_to_reference"))
                    positive_support = opt_float(row.get("positive_point_support_rate"))
                    negative_overlap = opt_float(row.get("probation_negative_sibling_overlap_rate"))
                    runtime_sec = opt_float(row.get("runtime_sec"))
                elif attempt_frame is not None and frame_id == int(attempt_frame):
                    row = attempt
                    state = "REACTIVATING_ATTEMPT"
                    emits_output = True
                    occupies_sam_slot = True
                    output_source = "sam2_video_readd_prompt"
                    metric_prefix = "attempt"
                    selected_variant = str(row.get("selected_variant", ""))
                    iou_diag = opt_float(row.get("attempt_iou_to_reference"))
                    negative_overlap = opt_float(row.get("attempt_negative_sibling_overlap_rate"))
                    runtime_sec = opt_float(row.get("runtime_sec"))
                elif confirm_frame is not None and frame_id == int(confirm_frame):
                    row = confirm
                    committed = opt_bool(row.get("reactivation_committed_to_sam2_video_state"))
                    if str(row.get("selected_variant", "")) == "SKIPPED_PROBATION_FAILED" or (
                        "reactivation_committed_to_sam2_video_state" in row and not committed
                    ):
                        state = "REACTIVATION_SKIPPED"
                        emits_output = False
                        occupies_sam_slot = False
                        output_source = "none"
                        skip_reason = str(row.get("skip_reason", ""))
                    else:
                        state = "REACTIVATING_CONFIRM"
                        emits_output = opt_bool(row.get("target_present")) or bool(row)
                        occupies_sam_slot = True
                        output_source = "sam2_video_confirm_prompt"
                    metric_prefix = "confirm"
                    selected_variant = str(row.get("selected_variant", ""))
                    iou_diag = opt_float(row.get("confirm_iou_to_reference"))
                    positive_support = opt_float(row.get("confirm_positive_point_support_rate"))
                    negative_overlap = opt_float(row.get("confirm_negative_sibling_overlap_rate"))
                    runtime_sec = opt_float(row.get("runtime_sec"))
                else:
                    shadow = by_key.get(("shadow_output", event_index, frame_id))
                    if shadow is not None:
                        state = "SHADOW_2D" if opt_bool(shadow.get("output_mask")) else "SHADOW_SKIPPED"
                        emits_output = opt_bool(shadow.get("output_mask"))
                        occupies_sam_slot = False
                        output_source = "sam2_image_shadow_prompt" if emits_output else "none"
                        metric_prefix = "shadow"
                        selected_variant = str(shadow.get("selected_variant", ""))
                        skip_reason = str(shadow.get("skip_reason", ""))
                        iou_diag = opt_float(shadow.get("shadow_iou_to_reference"))
                        positive_support = opt_float(shadow.get("positive_point_support_rate"))
                        negative_overlap = opt_float(shadow.get("shadow_negative_sibling_overlap_rate"))
                        runtime_sec = opt_float(shadow.get("runtime_sec"))

                if state in {
                    "DORMANT_NO_OUTPUT",
                    "SHADOW_2D",
                    "SHADOW_SKIPPED",
                    "PROBATION_2D",
                    "PROBATION_SKIPPED",
                    "REACTIVATION_SKIPPED",
                }:
                    avoided_slot_frames += 1
                if state == "SHADOW_2D":
                    shadow_output_frames += 1
                if state == "PROBATION_2D":
                    probation_output_frames += 1

            row_out = {
                "event_index": int(event_index),
                "frame_id": int(frame_id),
                "reference_global_id": int(ref_id),
                "live_obj_id": live_obj_id,
                "source_lag": source_lag,
                "source_live_area_px": int(source_live_area),
                "source_mapping_iou": float(source_mapping_iou),
                "source_mapping_accepted": bool(accepted),
                "source_mapping_found": opt_bool(mapping.get("source_mapping_found")),
                "long_term_memory_admitted": bool(long_term_admitted),
                "long_term_admission_skip_reason": long_term_skip_reason,
                "prompt_only_unmapped_source_reactivation": bool(prompt_only_enabled),
                "prompt_new_object_assignment_frame_id": prompt_assignment_frame,
                "prompt_new_object_trigger": str(prompt_assignment.get("prompt_new_object_trigger", "")),
                "long_term_min_source_area": opt_int(mapping.get("long_term_min_source_area")),
                "long_term_min_positive_points": opt_int(mapping.get("long_term_min_positive_points")),
                "long_term_min_confirm_positive_points": opt_int(mapping.get("long_term_min_confirm_positive_points")),
                "long_term_max_events": opt_int(mapping.get("long_term_max_events")),
                "removed_from_sam_state": bool(removed),
                "lifecycle_state": state,
                "occupies_sam_slot_estimate": bool(occupies_sam_slot),
                "emits_output": bool(emits_output),
                "output_source": output_source,
                "selected_variant": selected_variant,
                "skip_reason": skip_reason,
                "metric_prefix": metric_prefix,
                "reference_iou_diagnostic": iou_diag,
                "positive_point_support_rate": positive_support,
                "negative_sibling_overlap_rate": negative_overlap,
                "runtime_sec": runtime_sec,
                "reference_metrics_are_diagnostic_only": True,
            }
            lifecycle_rows.append(row_out)
            state_counter[state] += 1
            if iou_diag is not None:
                diagnostic_iou_values[state].append(float(iou_diag))

    frame_diag = {int(row.get("frame_id")): row for row in rolling_summary.get("frame_diagnostics", []) if row.get("frame_id") is not None}
    fidelity_by_frame = {int(row["frame_id"]): row for row in fidelity_rows if row.get("frame_id")}
    frame_rows: list[dict[str, Any]] = []
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in lifecycle_rows:
        rows_by_frame[int(row["frame_id"])].append(row)
    for frame_id in frame_ids:
        rows_f = rows_by_frame.get(int(frame_id), [])
        diag = frame_diag.get(int(frame_id), {})
        fidelity = fidelity_by_frame.get(int(frame_id), {})
        frame_rows.append(
            {
                "frame_id": int(frame_id),
                "tracked_event_count": int(len(rows_f)),
                "shadow_output_event_count": int(
                    sum(1 for row in rows_f if row["lifecycle_state"] in {"SHADOW_2D", "PROMPT_SHADOW_2D"})
                ),
                "probation_output_event_count": int(
                    sum(
                        1
                        for row in rows_f
                        if row["lifecycle_state"] in {"PROBATION_2D", "PROMPT_PROBATION_2D"}
                    )
                ),
                "dormant_no_output_event_count": int(
                    sum(1 for row in rows_f if row["lifecycle_state"] == "DORMANT_NO_OUTPUT")
                ),
                "reactivating_event_count": int(
                    sum(
                        1
                        for row in rows_f
                        if str(row["lifecycle_state"]).startswith("REACTIVATING")
                        or str(row["lifecycle_state"]).startswith("PROMPT_REACTIVATING")
                    )
                ),
                "skipped_long_term_policy_event_count": int(
                    sum(1 for row in rows_f if row["lifecycle_state"] == "SKIPPED_LONG_TERM_POLICY")
                ),
                "skipped_unmapped_event_count": int(
                    sum(1 for row in rows_f if row["lifecycle_state"] == "SKIPPED_UNMAPPED")
                ),
                "sam_slot_occupied_event_count_estimate": int(
                    sum(1 for row in rows_f if bool(row["occupies_sam_slot_estimate"]))
                ),
                "sam_slot_avoided_event_count_estimate": int(
                    sum(
                        1
                        for row in rows_f
                        if row["lifecycle_state"]
                        in {
                            "DORMANT_NO_OUTPUT",
                            "SHADOW_2D",
                            "SHADOW_SKIPPED",
                            "PROBATION_2D",
                            "PROBATION_SKIPPED",
                            "REACTIVATION_SKIPPED",
                            "PROMPT_NEW_OBJECT_DORMANT",
                            "PROMPT_SHADOW_2D",
                            "PROMPT_SHADOW_SKIPPED",
                            "PROMPT_PROBATION_2D",
                            "PROMPT_PROBATION_SKIPPED",
                            "PROMPT_REACTIVATION_SKIPPED",
                        }
                    )
                ),
                "candidate_visible_id_count_diagnostic": opt_int(fidelity.get("candidate_visible_id_count"), 0),
                "reference_visible_id_count_diagnostic": opt_int(fidelity.get("reference_visible_id_count"), 0),
                "foreground_recall_diagnostic": opt_float(fidelity.get("foreground_recall")),
                "foreground_precision_diagnostic": opt_float(fidelity.get("foreground_precision")),
                "foreground_iou_diagnostic": opt_float(fidelity.get("foreground_iou")),
                "rolling_final_frame_mask_count": opt_int(diag.get("final_frame_mask_count"), 0),
                "rolling_propagated_post_disjoin_count": opt_int(diag.get("propagated_post_disjoin_count"), 0),
                "rolling_gap_mask_count": opt_int(diag.get("gap_mask_count"), 0),
                "rolling_stream_active_object_count_after_prune": opt_int(
                    diag.get("stream_active_object_count_after_prune"), 0
                ),
                "reference_metrics_are_diagnostic_only": True,
            }
        )

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    summary = {
        "schema_version": "stream4d_v107_phase8_lifecycle_ledger_summary_v1",
        "frame_ids": frame_ids,
        "event_indices": events,
        "event_count": int(len(events)),
        "prompt_only_unmapped_source_event_count": int(len(prompt_assignment_by_event)),
        "prompt_new_object_assignment_count": int(len(prompt_assignment_by_event)),
        "lifecycle_row_count": int(len(lifecycle_rows)),
        "frame_row_count": int(len(frame_rows)),
        "state_counts": dict(sorted(state_counter.items())),
        "sam_slot_avoided_event_frame_count_estimate": int(avoided_slot_frames),
        "shadow_output_event_frame_count": int(shadow_output_frames),
        "probation_output_event_frame_count": int(probation_output_frames),
        "skipped_unmapped_event_frame_count": int(skipped_unmapped_frames),
        "skipped_long_term_policy_event_frame_count": int(skipped_long_term_policy_frames),
        "reference_metrics_are_diagnostic_only": True,
        "acceptance_gate_uses_diagnostic_reference_metrics": False,
        "visual_review_status": "USER_VISUAL_REVIEW_PENDING",
        "small_objects_may_skip_long_term_memory": True,
        "diagnostic_iou_mean_by_state": {key: mean(vals) for key, vals in sorted(diagnostic_iou_values.items())},
        "audit_note": (
            "This ledger is reconstructed from scheduler records. It estimates per-event lifecycle state and SAM slot "
            "residency for audit; it is not a full memory tensor accounting model and does not make visual acceptance claims."
        ),
    }
    return lifecycle_rows, frame_rows, summary


def main() -> int:
    args = parse_args()
    scheduler_summary_path = resolve_path(str(args.scheduler_summary), REPO_ROOT)
    scheduler_summary = read_json(scheduler_summary_path)
    scheduler_root = scheduler_summary_path.parent

    records_path = resolve_path(
        str(args.records_csv or scheduler_summary.get("records_csv", "g3_scheduler_records.csv")),
        scheduler_root,
    )
    rolling_summary_path = resolve_path(
        str(args.rolling_summary or scheduler_summary.get("rolling_summary", "")),
        scheduler_root,
    )
    fidelity_summary_path = (
        resolve_path(str(args.foreground_fidelity_summary), scheduler_root)
        if str(args.foreground_fidelity_summary).strip()
        else Path("")
    )
    fidelity_rows_path = (
        resolve_path(str(args.foreground_fidelity_rows), scheduler_root)
        if str(args.foreground_fidelity_rows).strip()
        else Path("")
    )
    if not fidelity_rows_path or not fidelity_rows_path.exists():
        sibling = scheduler_root / "foreground_fidelity_audit" / "foreground_fidelity_rows.csv"
        fidelity_rows_path = sibling if sibling.exists() else Path("")
    if not fidelity_summary_path or not fidelity_summary_path.exists():
        sibling_summary = scheduler_root / "foreground_fidelity_audit" / "foreground_fidelity_summary.json"
        fidelity_summary_path = sibling_summary if sibling_summary.exists() else Path("")

    output_root = resolve_path(str(args.output_root), REPO_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    records = read_csv(records_path)
    rolling_summary = read_json(rolling_summary_path)
    fidelity_rows = read_csv(fidelity_rows_path) if fidelity_rows_path and fidelity_rows_path.exists() else []
    fidelity_summary = read_json(fidelity_summary_path) if fidelity_summary_path and fidelity_summary_path.exists() else {}

    lifecycle_rows, frame_rows, summary = build_ledger(
        scheduler_summary=scheduler_summary,
        records=records,
        rolling_summary=rolling_summary,
        fidelity_rows=fidelity_rows,
    )

    lifecycle_csv = output_root / "object_lifecycle_ledger.csv"
    lifecycle_jsonl = output_root / "object_lifecycle_ledger.jsonl"
    frame_csv = output_root / "frame_lifecycle_summary.csv"
    summary_path = output_root / "lifecycle_ledger_summary.json"

    write_csv(lifecycle_csv, lifecycle_rows)
    with lifecycle_jsonl.open("w", encoding="utf-8") as handle:
        for row in lifecycle_rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    write_csv(frame_csv, frame_rows)

    visual_casebook_path = (
        resolve_path(str(args.visual_casebook_manifest), scheduler_root)
        if str(args.visual_casebook_manifest).strip()
        else scheduler_root / "visual_casebook_manifest.json"
    )
    summary.update(
        {
            "scheduler_summary": rel(scheduler_summary_path),
            "scheduler_summary_sha256": sha256_file(scheduler_summary_path),
            "records_csv": rel(records_path),
            "records_csv_sha256": sha256_file(records_path),
            "rolling_summary": rel(rolling_summary_path),
            "rolling_summary_sha256": sha256_file(rolling_summary_path),
            "foreground_fidelity_summary": rel(fidelity_summary_path) if fidelity_summary_path else "",
            "foreground_fidelity_summary_sha256": sha256_file(fidelity_summary_path)
            if fidelity_summary_path and fidelity_summary_path.exists()
            else "",
            "foreground_fidelity_rows": rel(fidelity_rows_path) if fidelity_rows_path else "",
            "foreground_fidelity_rows_sha256": sha256_file(fidelity_rows_path)
            if fidelity_rows_path and fidelity_rows_path.exists()
            else "",
            "visual_casebook_manifest": rel(visual_casebook_path) if visual_casebook_path.exists() else "",
            "visual_casebook_manifest_sha256": sha256_file(visual_casebook_path)
            if visual_casebook_path.exists()
            else "",
            "scheduler_small_objects_may_skip_long_term_memory": scheduler_summary.get(
                "small_objects_may_skip_long_term_memory"
            ),
            "scheduler_long_term_min_source_area": scheduler_summary.get("long_term_min_source_area"),
            "scheduler_long_term_min_positive_points": scheduler_summary.get("long_term_min_positive_points"),
            "scheduler_long_term_min_confirm_positive_points": scheduler_summary.get(
                "long_term_min_confirm_positive_points"
            ),
            "scheduler_long_term_admission_skip_reasons": scheduler_summary.get(
                "long_term_admission_skip_reasons", {}
            ),
            "scheduler_reactivation_probation_mode": scheduler_summary.get("reactivation_probation_mode"),
            "scheduler_probation_output_mask_count": scheduler_summary.get("probation_output_mask_count"),
            "scheduler_probation_skip_reasons": scheduler_summary.get("probation_skip_reasons", {}),
            "object_lifecycle_ledger_csv": rel(lifecycle_csv),
            "object_lifecycle_ledger_csv_sha256": sha256_file(lifecycle_csv),
            "object_lifecycle_ledger_jsonl": rel(lifecycle_jsonl),
            "object_lifecycle_ledger_jsonl_sha256": sha256_file(lifecycle_jsonl),
            "frame_lifecycle_summary_csv": rel(frame_csv),
            "frame_lifecycle_summary_csv_sha256": sha256_file(frame_csv),
            "foreground_fidelity_summary_diagnostic": {
                key: fidelity_summary.get(key)
                for key in (
                    "foreground_recall_mean",
                    "foreground_precision_mean",
                    "foreground_iou_mean",
                    "candidate_visible_id_count_mean",
                    "reference_visible_id_count_mean",
                )
                if key in fidelity_summary
            },
        }
    )
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": rel(summary_path),
                "lifecycle_rows": len(lifecycle_rows),
                "state_counts": summary["state_counts"],
                "sam_slot_avoided_event_frame_count_estimate": summary[
                    "sam_slot_avoided_event_frame_count_estimate"
                ],
                "shadow_output_event_frame_count": summary["shadow_output_event_frame_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
