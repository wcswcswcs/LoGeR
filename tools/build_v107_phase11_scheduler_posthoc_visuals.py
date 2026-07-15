#!/usr/bin/env python3
"""Build high-resolution scheduler visuals from frozen v107 artifacts.

This is a post-hoc visual exporter: it reads the existing scheduler labels,
reference labels, scheduler records, and LingBot prompt points. It does not run
SAM2 or change the frozen holdout outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for item in (REPO_ROOT, REPO_ROOT / "Grounded-SAM-2"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from tools.run_v107_phase7_lingbot_sam2_prompt_benchmark import (  # noqa: E402
    jsonable,
    load_points,
    load_reference_records,
    mask_metrics,
    read_json,
    rel,
    sha256_file,
    write_json,
)
from tools.run_v107_phase8_sam2_live_state_reactivation_probe import (  # noqa: E402
    draw_zoom_overlay,
    infer_lingbot_hw,
    point_arrays,
    points_for_event,
    prompt_point_rates,
    rgb_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-root", required=True, type=Path)
    parser.add_argument("--reference-run-root", required=True, type=Path)
    parser.add_argument("--prompt-probe-root", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--events",
        default="auto_output",
        help="Comma-separated event indices, or auto_output for records with actual probation/confirm output.",
    )
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--stages", default="attempt,confirm")
    parser.add_argument("--visual-pad", type=int, default=180)
    parser.add_argument("--visual-scale", type=int, default=3)
    return parser.parse_args()


def as_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int32, copy=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(jsonable(row))


def parse_event_indices(text: str, records: list[dict[str, Any]], max_events: int) -> list[int]:
    if str(text).strip() != "auto_output":
        return [int(v) for v in str(text).split(",") if str(v).strip()]
    selected: list[int] = []
    for row in records:
        event_index = int(row.get("event_index", -1))
        if event_index < 0 or event_index in selected:
            continue
        record_type = str(row.get("record_type", ""))
        selected_variant = str(row.get("selected_variant", ""))
        output_mask = bool(row.get("output_mask", False))
        target_present = bool(row.get("target_present", False))
        if record_type == "probation_attempt" and output_mask and not selected_variant.startswith("SKIPPED"):
            selected.append(event_index)
        elif record_type == "confirm" and target_present and not selected_variant.startswith("SKIPPED"):
            selected.append(event_index)
        if int(max_events) > 0 and len(selected) >= int(max_events):
            break
    return selected


def prompt_summary_paths(prompt_root: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    rows_name = str(summary.get("selected_rows_csv") or "prompt_capsule_visibility_rows.csv")
    points_name = str(summary.get("selected_visible_point_records") or "prompt_capsule_visible_point_records.json")

    def resolve_summary_path(text: str) -> Path:
        path = Path(text)
        if path.is_absolute():
            return path
        prompt_candidate = prompt_root / path
        if prompt_candidate.exists():
            return prompt_candidate
        return REPO_ROOT / path

    return resolve_summary_path(rows_name), resolve_summary_path(points_name)


def best_scheduler_mask(label: np.ndarray, ref_mask: np.ndarray, reference_global_id: int) -> tuple[np.ndarray, int, float]:
    direct = label == int(reference_global_id)
    best_mask = direct
    best_label = int(reference_global_id)
    union = int(np.count_nonzero(direct | ref_mask))
    best_iou = float(np.count_nonzero(direct & ref_mask) / max(union, 1))
    for label_id in [int(v) for v in np.unique(label) if int(v) != 0]:
        cand = label == label_id
        inter = int(np.count_nonzero(cand & ref_mask))
        if inter <= 0:
            continue
        cand_iou = float(inter / max(int(np.count_nonzero(cand | ref_mask)), 1))
        if cand_iou > best_iou:
            best_iou = cand_iou
            best_mask = cand
            best_label = int(label_id)
    return best_mask, best_label, best_iou


def main() -> int:
    started = time.time()
    args = parse_args()
    scheduler_root = as_path(args.scheduler_root)
    reference_root = as_path(args.reference_run_root)
    prompt_root = as_path(args.prompt_probe_root)
    probe_root = as_path(args.probe_root)
    scene_root = as_path(args.scene_root)
    output_root = as_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    scheduler_summary_path = scheduler_root / "g3_scheduler_summary.json"
    records_path = scheduler_root / "g3_scheduler_records.jsonl"
    scheduler_summary = read_json(scheduler_summary_path)
    rolling_root = scheduler_root / "v107_phase8_g3_rolling_scheduler_smoke"
    scheduler_label_root = rolling_root / "labels"
    records = read_jsonl(records_path)
    selected_events = set(parse_event_indices(str(args.events), records, int(args.max_events)))
    stages = {part.strip() for part in str(args.stages).split(",") if part.strip()}

    prompt_summary = read_json(prompt_root / "prompt_capsule_visibility_probe_summary.json")
    rows_path, points_path = prompt_summary_paths(prompt_root, prompt_summary)
    points_by_case = load_points(points_path)
    lingbot_hw = infer_lingbot_hw(prompt_root, prompt_summary)
    reference_records = load_reference_records(reference_root)
    event_setup = read_json(probe_root / "live_state_reactivation_event_setup.json")
    setup_by_event = {int(row["event_index"]): row for row in event_setup.get("events", [])}
    selected_pose = str(prompt_summary.get("selected_pose_mode", "direct_as_c2w"))

    out_rows: list[dict[str, Any]] = []
    for row in records:
        record_type = str(row.get("record_type", ""))
        if record_type == "probation_attempt":
            stage = "attempt"
            frame_key = "attempt_frame_id"
        elif record_type == "confirm":
            stage = "confirm"
            frame_key = "confirm_frame_id"
        else:
            continue
        event_index = int(row.get("event_index", -1))
        if event_index not in selected_events or stage not in stages:
            continue
        if str(row.get("selected_variant", "")).startswith("SKIPPED"):
            continue
        if record_type == "probation_attempt" and not bool(row.get("output_mask", False)):
            continue
        if record_type == "confirm" and not bool(row.get("target_present", False)):
            continue
        if event_index not in setup_by_event:
            raise RuntimeError(f"event {event_index} missing from {probe_root}")
        event = setup_by_event[event_index]
        ref_id = int(row["reference_global_id"])
        frame_id = int(row["frame_id"])
        expected_frame_id = int(event[frame_key])
        if frame_id != expected_frame_id:
            raise RuntimeError(f"event {event_index} {stage} frame mismatch: record {frame_id}, setup {expected_frame_id}")

        ref_label = load_label(Path(reference_records[frame_id]["label_path"]))
        sched_label = load_label(scheduler_label_root / f"frame_{frame_id:06d}.png")
        ref_mask = ref_label == ref_id
        pred_mask, pred_label_id, pred_iou = best_scheduler_mask(sched_label, ref_mask, ref_id)
        point_records = [
            item
            for item in points_for_event(
                points_by_case,
                source_frame_index=None,
                source_frame_id=int(event["source_frame_id"]),
                target_frame_id=frame_id,
                target_obj_id=ref_id,
            )
            if str(item.get("pose_mode", selected_pose)) == selected_pose
        ]
        coords, labels, neg_ids = point_arrays(
            point_records,
            lingbot_hw=lingbot_hw,
            orig_hw=ref_label.shape[:2],
            include_negative=True,
        )
        rgb = rgb_frame(scene_root, frame_id)
        metrics = mask_metrics(pred_mask, ref_mask, ref_label, set(int(v) for v in neg_ids))
        rates = prompt_point_rates(pred_mask, coords, labels)
        variant = str(row.get("selected_variant", ""))
        out_path = (
            output_root
            / "highres_scheduler_posthoc_visuals"
            / f"event{event_index:03d}_{stage}_{variant}_f{frame_id}_ref{ref_id}_predlabel{pred_label_id}.jpg"
        )
        color = (255, 190, 40) if stage == "attempt" else (255, 70, 170)
        draw_zoom_overlay(
            rgb=rgb,
            ref_mask=ref_mask,
            pred_mask=pred_mask,
            points=coords,
            labels=labels,
            title=f"posthoc scheduler event{event_index:03d} {stage} {variant} f{frame_id} ref{ref_id}",
            output_path=out_path,
            pad=int(args.visual_pad),
            scale=int(args.visual_scale),
            color=color,
        )
        out_rows.append(
            {
                "schema_version": "stream4d_v107_phase11_scheduler_posthoc_visual_v1",
                "event_index": event_index,
                "stage": stage,
                "record_type": record_type,
                "frame_id": frame_id,
                "reference_global_id": ref_id,
                "scheduler_pred_label_id": int(pred_label_id),
                "selected_variant": variant,
                "source_frame_id": int(event["source_frame_id"]),
                "source_lag": int(event["source_lag"]),
                "visual_path": rel(out_path),
                "visual_sha256": sha256_file(out_path),
                "reference_mask_area_px": int(np.count_nonzero(ref_mask)),
                "pred_mask_area_px": int(np.count_nonzero(pred_mask)),
                "best_scheduler_label_iou_to_reference": float(pred_iou),
                "record_iou_to_reference": row.get("probation_iou_to_reference")
                if stage == "attempt"
                else row.get("confirm_iou_to_reference"),
                "positive_point_count": int(sum(int(v) == 1 for v in labels)),
                "negative_point_count": int(sum(int(v) == 0 for v in labels)),
                "positive_point_support_rate": float(rates["positive_point_support_rate"]),
                "candidate_negative_point_conflict_rate": float(rates["candidate_negative_point_conflict_rate"]),
                "negative_sibling_overlap_rate": float(metrics["negative_sibling_overlap_rate"]),
                "projection_geometry_source": scheduler_summary.get("projection_geometry_source"),
                "uses_scannet_pose_or_depth_for_projection": bool(
                    scheduler_summary.get("uses_scannet_pose_or_depth_for_projection")
                ),
                "posthoc_no_model_rerun": True,
                "visual_review_status": "USER_VISUAL_REVIEW_PENDING",
                "metric_role": "diagnostic_only_not_acceptance_gate",
            }
        )

    manifest = {
        "schema_version": "stream4d_v107_phase11_scheduler_posthoc_visual_manifest_v1",
        "scheduler_root": rel(scheduler_root),
        "reference_run_root": rel(reference_root),
        "prompt_probe_root": rel(prompt_root),
        "probe_root": rel(probe_root),
        "scene_root": rel(scene_root),
        "scheduler_summary": rel(scheduler_summary_path),
        "scheduler_summary_sha256": sha256_file(scheduler_summary_path),
        "records_jsonl": rel(records_path),
        "records_jsonl_sha256": sha256_file(records_path),
        "prompt_points_json": rel(points_path),
        "prompt_points_json_sha256": sha256_file(points_path),
        "selected_events": sorted(selected_events),
        "stages": sorted(stages),
        "visual_count": len(out_rows),
        "posthoc_no_model_rerun": True,
        "visual_review_status": "USER_VISUAL_REVIEW_PENDING",
        "runtime_sec": float(time.time() - started),
        "rows": out_rows,
    }
    write_json(output_root / "scheduler_posthoc_visual_manifest.json", manifest)
    write_csv(output_root / "scheduler_posthoc_visual_rows.csv", out_rows)
    print(
        json.dumps(
            {
                "out_root": rel(output_root),
                "selected_events": sorted(selected_events),
                "visual_count": len(out_rows),
                "manifest_sha256": sha256_file(output_root / "scheduler_posthoc_visual_manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
