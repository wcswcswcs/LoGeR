#!/usr/bin/env python3
"""Build a visual-first Phase14 ablation summary.

The summary deliberately keeps numeric fields diagnostic-only. Quality remains
USER_REVIEW_PENDING unless the run is manually accepted from high-resolution
visual evidence outside this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_name_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected NAME=SUMMARY_JSON, got {value!r}")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError(f"Missing run name in {value!r}")
    p = Path(path)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"Summary JSON does not exist: {p}")
    return name, p


def first_record(summary: dict[str, Any]) -> dict[str, Any]:
    index_path = Path(str(summary.get("phase12_visual_review_index_json", "")))
    if not index_path.is_file():
        return {}
    index = load_json(index_path)
    records = index.get("records", [])
    if not records:
        return {}
    return records[0]


def load_v107_rolling_stats(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    v107_path_raw = str(summary.get("v107_summary", ""))
    if not v107_path_raw:
        return {}
    v107_path = Path(v107_path_raw)
    if not v107_path.is_absolute():
        v107_path = Path.cwd() / v107_path
    if not v107_path.is_file():
        candidate = summary_path.parent / v107_path_raw
        if candidate.is_file():
            v107_path = candidate
    if not v107_path.is_file():
        return {}
    v107 = load_json(v107_path)
    rolling = v107.get("rolling_stats", {})
    return rolling if isinstance(rolling, dict) else {}


def row_from_summary(run_name: str, summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    record = first_record(summary)
    controls = summary.get("phase14_control_fields", {})
    if not isinstance(controls, dict):
        controls = {}
    rolling_stats = load_v107_rolling_stats(summary, summary_path)
    return {
        "run_name": run_name,
        "ablation": summary.get("phase14_ablation", ""),
        "implemented": True,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "phase14_ablation_args": " ".join(str(x) for x in summary.get("phase14_ablation_args", [])),
        "cross_run_cache_read_count_observed": summary.get("phase14_cross_run_cache_read_count_observed", ""),
        "status": summary.get("status", ""),
        "full_online_run_executed": summary.get("full_online_run_executed", ""),
        "existing_v107_root_reused": summary.get("existing_v107_root_reused", ""),
        "full_scene_video_exists": summary.get("full_scene_video_exists", ""),
        "full_scene_video": summary.get("full_scene_video", ""),
        "visual_review_image_count": summary.get("visual_review_image_count", ""),
        "visual_confirmation_required": summary.get("visual_confirmation_required", True),
        "metrics_are_diagnostic_only": summary.get("metrics_are_diagnostic_only", True),
        "quality_decision_rule": summary.get(
            "quality_decision_rule", "Only high-resolution visual confirmation can decide good or bad."
        ),
        "control_reactivation_prompt_mode": controls.get("reactivation_prompt_mode", ""),
        "control_output_plane_enabled": controls.get("output_plane_enabled", ""),
        "control_disable_output_plane": controls.get("disable_output_plane", ""),
        "control_disable_gap_birth": controls.get("disable_gap_birth", ""),
        "control_random_geometry_prompts_enabled": controls.get("random_geometry_prompts_enabled", ""),
        "control_shadow_output_mode": controls.get("shadow_output_mode", ""),
        "control_birth_admission_appearance_enabled": controls.get("birth_admission_appearance_enabled", ""),
        "control_disable_birth_admission_appearance": controls.get("disable_birth_admission_appearance", ""),
        "control_gap_max_points": controls.get("gap_max_points", ""),
        "control_gap_max_points_per_component": controls.get("gap_max_points_per_component", ""),
        "rolling_gap_birth_disabled_count": rolling_stats.get("gap_birth_disabled_count", ""),
        "rolling_gap_birth_disabled_mask_count": rolling_stats.get("gap_birth_disabled_mask_count", ""),
        "rolling_stream_add_masks_input_mask_count": rolling_stats.get("stream_add_masks_input_mask_count", ""),
        "rolling_stream_add_masks_admitted_mask_count": rolling_stats.get("stream_add_masks_admitted_mask_count", ""),
        "rolling_post_start_birth_filter_call_count": rolling_stats.get("post_start_birth_filter_call_count", ""),
        "rolling_gap_output_filter_call_count": rolling_stats.get("gap_output_filter_call_count", ""),
        "durable_memory_mutation_request_count": summary.get("durable_memory_mutation_request_count", ""),
        "primary_visual_kind": record.get("primary_visual_kind", ""),
        "primary_visual_path": record.get("visual_path", ""),
        "primary_visual_sha256": record.get("visual_sha256", ""),
        "generated_final_label_panel_path": record.get("generated_final_label_panel_path", ""),
        "generated_final_label_panel_sha256": record.get("generated_final_label_panel_sha256", ""),
        "source_g3_selected_variant": record.get("source_g3_selected_variant", ""),
        "source_g3_record_skip_reason": record.get("source_g3_record_skip_reason", ""),
        "event_index": record.get("event_index", ""),
        "frame_id": record.get("frame_id", ""),
        "live_obj_id": record.get("live_obj_id", ""),
        "candidate_area_px": record.get("candidate_area_px", ""),
        "candidate_bbox_xyxy": json.dumps(record.get("candidate_bbox_xyxy", []), separators=(",", ":")),
        "visual_review_status": record.get("visual_review_status", "USER_REVIEW_PENDING"),
        "manual_visual_judgment": "USER_REVIEW_PENDING",
        "unsupported_reason": "",
    }


def unsupported_row(name_reason: str) -> dict[str, Any]:
    if "=" in name_reason:
        name, reason = name_reason.split("=", 1)
    else:
        name, reason = name_reason, "No implemented v108 runner control was available in this wrapper."
    return {
        "run_name": name,
        "ablation": name,
        "implemented": False,
        "summary_path": "",
        "summary_sha256": "",
        "phase14_ablation_args": "",
        "cross_run_cache_read_count_observed": "",
        "status": "NOT_IMPLEMENTED_IN_CURRENT_WRAPPER",
        "full_online_run_executed": False,
        "existing_v107_root_reused": "",
        "full_scene_video_exists": "",
        "full_scene_video": "",
        "visual_review_image_count": 0,
        "visual_confirmation_required": True,
        "metrics_are_diagnostic_only": True,
        "quality_decision_rule": "Only high-resolution visual confirmation can decide good or bad.",
        "control_reactivation_prompt_mode": "",
        "control_output_plane_enabled": "",
        "control_disable_output_plane": "",
        "control_disable_gap_birth": "",
        "control_random_geometry_prompts_enabled": "",
        "control_shadow_output_mode": "",
        "control_birth_admission_appearance_enabled": "",
        "control_disable_birth_admission_appearance": "",
        "control_gap_max_points": "",
        "control_gap_max_points_per_component": "",
        "rolling_gap_birth_disabled_count": "",
        "rolling_gap_birth_disabled_mask_count": "",
        "rolling_stream_add_masks_input_mask_count": "",
        "rolling_stream_add_masks_admitted_mask_count": "",
        "rolling_post_start_birth_filter_call_count": "",
        "rolling_gap_output_filter_call_count": "",
        "durable_memory_mutation_request_count": "",
        "primary_visual_kind": "",
        "primary_visual_path": "",
        "primary_visual_sha256": "",
        "generated_final_label_panel_path": "",
        "generated_final_label_panel_sha256": "",
        "source_g3_selected_variant": "",
        "source_g3_record_skip_reason": "",
        "event_index": "",
        "frame_id": "",
        "live_obj_id": "",
        "candidate_area_px": "",
        "candidate_bbox_xyxy": "",
        "visual_review_status": "NO_VISUAL_EVIDENCE",
        "manual_visual_judgment": "NO_VISUAL_EVIDENCE",
        "unsupported_reason": reason,
    }


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "run_name",
        "implemented",
        "status",
        "visual_review_status",
        "primary_visual_kind",
        "manual_visual_judgment",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Phase14 Ablation Summary\n\n")
        f.write("Metrics and counts are diagnostic only. Good/bad decisions require high-resolution visual confirmation.\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(h, "")).replace("|", "\\|") for h in headers) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run", action="append", default=[], type=split_name_path)
    parser.add_argument("--unsupported", action="append", default=[])
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)

    rows = [row_from_summary(name, path) for name, path in args.run]
    rows.extend(unsupported_row(x) for x in args.unsupported)

    csv_path = out / "phase14_ablation_rows.csv"
    json_path = out / "phase14_ablation_summary.json"
    md_path = out / "phase14_ablation_summary.md"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": "v108_phase14_ablation_summary_v1",
        "status": "PHASE14_ABLATION_DIAGNOSTIC_ONLY_USER_REVIEW_PENDING",
        "metrics_are_diagnostic_only": True,
        "visual_confirmation_required": True,
        "quality_decision_rule": "Only high-resolution visual confirmation can decide good or bad.",
        "row_count": len(rows),
        "implemented_row_count": sum(1 for r in rows if r["implemented"]),
        "not_implemented_row_count": sum(1 for r in rows if not r["implemented"]),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, rows)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(json_path),
                "csv": str(csv_path),
                "markdown": str(md_path),
                "row_count": len(rows),
                "implemented_row_count": summary["implemented_row_count"],
                "not_implemented_row_count": summary["not_implemented_row_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
