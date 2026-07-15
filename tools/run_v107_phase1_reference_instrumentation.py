#!/usr/bin/env python3
"""Build v107 Phase1 label-parity reference instrumentation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {path}")
    path.mkdir(parents=True, exist_ok=False)


def copy_or_link(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def imread_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def imread_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return -1, -1, -1, -1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def rgb_descriptor(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    pix = rgb[mask]
    if pix.size == 0:
        return {
            "rgb_mean_r": None,
            "rgb_mean_g": None,
            "rgb_mean_b": None,
            "rgb_std_r": None,
            "rgb_std_g": None,
            "rgb_std_b": None,
        }
    mean = pix.mean(axis=0)
    std = pix.std(axis=0)
    return {
        "rgb_mean_r": float(mean[0]),
        "rgb_mean_g": float(mean[1]),
        "rgb_mean_b": float(mean[2]),
        "rgb_std_r": float(std[0]),
        "rgb_std_g": float(std[1]),
        "rgb_std_b": float(std[2]),
    }


def resolve_path(path_like: str, base: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return ROOT / path


def rgb_path_for(summary: dict[str, Any], frame_id: int, rgb_root: str | None) -> Path:
    if rgb_root:
        root = Path(rgb_root)
        if not root.is_absolute():
            root = ROOT / root
    else:
        root = ROOT / "Stream3D/data/scannet/processed"
    return root / str(summary["scene_id"]) / "color" / f"{int(frame_id)}.jpg"


def removal_events(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for event in summary.get("stream_object_prune_events", []) or []:
        ids = list(event.get("pruned_object_ids") or [])
        if not ids and isinstance(event.get("events"), list):
            ids = [row.get("object_id") for row in event["events"] if row.get("object_id") is not None]
        reason = event.get("reason") or "invisible_after_frames"
        for obj_id in ids:
            out[int(obj_id)] = {
                "remove_frame_index": event.get("chunk_frame_index"),
                "remove_frame_id": event.get("frame_id"),
                "remove_reason": reason,
                "active_stream_object_count_after_prune": event.get("active_stream_object_count_after_prune"),
            }
    return out


def write_video_reference(summary: dict[str, Any], run_root: Path, out_video_dir: Path) -> dict[str, Any]:
    video = summary.get("v106_visual_confirmation_video") or {}
    src_text = video.get("path") or summary.get("video_path") or ""
    if not src_text:
        return {"copied": False, "reason": "no_reference_video_path_in_summary"}
    src = resolve_path(str(src_text), run_root)
    if not src.exists():
        return {"copied": False, "reason": f"missing_reference_video:{src}"}
    dst = out_video_dir / src.name
    mode = copy_or_link(src, dst)
    return {"copied": True, "copy_mode": mode, "source": rel(src), "path": rel(dst), "sha256": sha256_file(dst)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rgb-root", default="")
    parser.add_argument("--phase0-gate", default="")
    args = parser.parse_args()

    reference_run_root = Path(args.reference_run_root)
    if not reference_run_root.is_absolute():
        reference_run_root = ROOT / reference_run_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    ensure_clean_dir(output_root)

    summary_path = reference_run_root / "summary.json"
    summary = read_json(summary_path)
    records = sorted(summary.get("records", []), key=lambda row: int(row["chunk_frame_index"]))
    if not records:
        raise ValueError(f"summary contains no records: {summary_path}")

    phase1 = output_root / "phase1"
    reference_labels_dir = phase1 / "reference_labels"
    instrumented_labels_dir = phase1 / "instrumented_labels"
    reference_videos_dir = phase1 / "reference_videos"
    reference_labels_dir.mkdir(parents=True, exist_ok=True)
    instrumented_labels_dir.mkdir(parents=True, exist_ok=True)
    reference_videos_dir.mkdir(parents=True, exist_ok=True)

    object_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    label_manifest: list[dict[str, Any]] = []
    pixel_mismatch_count = 0
    missing_labels: list[str] = []
    total_reference_foreground_px = 0
    total_instrumented_foreground_px = 0
    total_intersection_foreground_px = 0
    id_frames: dict[int, list[int]] = defaultdict(list)
    id_areas: dict[int, list[int]] = defaultdict(list)
    id_frame_ids: dict[int, list[int]] = defaultdict(list)

    for record in records:
        frame_index = int(record["chunk_frame_index"])
        frame_id = int(record["frame_id"])
        src_label = resolve_path(str(record["label_path"]), reference_run_root)
        if not src_label.exists():
            missing_labels.append(str(src_label))
            continue
        label = imread_label(src_label)
        ref_dst = reference_labels_dir / src_label.name
        ref_copy_mode = copy_or_link(src_label, ref_dst)
        instrumented = np.zeros_like(label)
        ids = [int(v) for v in np.unique(label).tolist() if int(v) > 0]
        rgb_path = rgb_path_for(summary, frame_id, args.rgb_root or None)
        rgb = imread_rgb(rgb_path)
        if rgb.shape[:2] != label.shape[:2]:
            rgb = cv2.resize(rgb, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)

        union = label > 0
        total_reference_foreground_px += int(np.count_nonzero(union))
        for obj_id in ids:
            mask = label == int(obj_id)
            area = int(np.count_nonzero(mask))
            instrumented[mask] = int(obj_id)
            x0, y0, x1, y1 = bbox_from_mask(mask)
            descriptor = rgb_descriptor(rgb, mask)
            id_frames[int(obj_id)].append(frame_index)
            id_frame_ids[int(obj_id)].append(frame_id)
            id_areas[int(obj_id)].append(area)
            object_rows.append(
                {
                    "schema_version": "stream4d_v107_object_frame_row_v1",
                    "scene_id": summary.get("scene_id"),
                    "frame_index": frame_index,
                    "frame_id": frame_id,
                    "runtime_id": int(obj_id),
                    "global_id": int(obj_id),
                    "id_mapping_kind": "v106_label_id_identity_mapping",
                    "mask_area": area,
                    "unique_foreground_contribution_px": area,
                    "overlap_with_other_masks_px": 0,
                    "bbox_x0": x0,
                    "bbox_y0": y0,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "visible": True,
                    "active_slot_residence": "unknown_from_existing_reference_label_replay",
                    "birth_source": "frame0_seed" if frame_index == 0 else "reference_visible_or_gap_birth",
                    "descriptor_kind": "rgb_mask_mean_std_not_sam2",
                    "raw_logit_mean": None,
                    "raw_logit_min": None,
                    "raw_logit_max": None,
                    "raw_logit_std": None,
                    **descriptor,
                    "label_path": rel(src_label),
                    "rgb_path": rel(rgb_path),
                }
            )
        instr_dst = instrumented_labels_dir / src_label.name
        if not cv2.imwrite(str(instr_dst), instrumented):
            raise RuntimeError(f"failed to write instrumented label: {instr_dst}")
        inst_label = imread_label(instr_dst)
        exact_equal = bool(np.array_equal(label, inst_label))
        if not exact_equal:
            pixel_mismatch_count += int(np.count_nonzero(label != inst_label))
        inst_union = inst_label > 0
        total_instrumented_foreground_px += int(np.count_nonzero(inst_union))
        total_intersection_foreground_px += int(np.count_nonzero(np.logical_and(union, inst_union)))
        frame_rows.append(
            {
                "frame_index": frame_index,
                "frame_id": frame_id,
                "reference_label_path": rel(src_label),
                "reference_label_copy": rel(ref_dst),
                "instrumented_label_path": rel(instr_dst),
                "visible_id_count": len(ids),
                "reference_foreground_px": int(np.count_nonzero(union)),
                "instrumented_foreground_px": int(np.count_nonzero(inst_union)),
                "pixel_exact_equal": exact_equal,
            }
        )
        label_manifest.append(
            {
                "frame_index": frame_index,
                "frame_id": frame_id,
                "source": rel(src_label),
                "reference_copy": rel(ref_dst),
                "reference_copy_mode": ref_copy_mode,
                "instrumented": rel(instr_dst),
                "source_sha256": sha256_file(src_label),
                "reference_copy_sha256": sha256_file(ref_dst),
                "instrumented_sha256": sha256_file(instr_dst),
                "pixel_exact_equal": exact_equal,
            }
        )

    remove_by_id = removal_events(summary)
    lifecycle_rows: list[dict[str, Any]] = []
    for obj_id in sorted(id_frames):
        frames = id_frames[obj_id]
        frame_ids = id_frame_ids[obj_id]
        areas = id_areas[obj_id]
        gaps = [b - a for a, b in zip(frames, frames[1:]) if (b - a) > 1]
        remove = remove_by_id.get(obj_id, {})
        lifecycle_rows.append(
            {
                "schema_version": "stream4d_v107_object_lifecycle_trace_v1",
                "scene_id": summary.get("scene_id"),
                "global_id": int(obj_id),
                "runtime_id_initial": int(obj_id),
                "first_frame_index": int(frames[0]),
                "first_frame_id": int(frame_ids[0]),
                "last_frame_index": int(frames[-1]),
                "last_frame_id": int(frame_ids[-1]),
                "visible_frame_count": int(len(frames)),
                "support_span_frames": int(frames[-1] - frames[0] + 1),
                "total_unique_coverage_px": int(sum(areas)),
                "mean_area": float(np.mean(areas)),
                "max_area": int(max(areas)),
                "min_area": int(min(areas)),
                "gap_count": int(len(gaps)),
                "reappearance_count": int(len(gaps)),
                "max_gap_frames": int(max(gaps) if gaps else 0),
                "removed_by_v106": bool(remove),
                "remove_frame_index": remove.get("remove_frame_index"),
                "remove_frame_id": remove.get("remove_frame_id"),
                "remove_reason": remove.get("remove_reason"),
                "active_stream_object_count_after_remove": remove.get("active_stream_object_count_after_prune"),
                "raw_logit_available": False,
                "sam2_pooled_descriptor_available": False,
                "active_slot_residence_available": False,
                "availability_note": (
                    "Derived from frozen v106 reference label rasters and summary events; raw SAM2 logits, "
                    "pooled SAM2 descriptors, and true per-object active-slot residence require live Phase1 instrumentation."
                ),
            }
        )

    lifecycle_by_id = {int(row["global_id"]): row for row in lifecycle_rows}
    for row in object_rows:
        life = lifecycle_by_id[int(row["global_id"])]
        row["future_lifespan_frames"] = int(life["last_frame_index"] - int(row["frame_index"]) + 1)
        row["future_visible_frame_count"] = int(
            sum(1 for frame in id_frames[int(row["global_id"])] if frame >= int(row["frame_index"]))
        )
        row["future_unique_coverage_px"] = int(
            sum(
                area
                for frame, area in zip(id_frames[int(row["global_id"])], id_areas[int(row["global_id"])], strict=True)
                if frame >= int(row["frame_index"])
            )
        )

    object_frame_path = phase1 / "object_frame_rows.parquet"
    lifecycle_path = phase1 / "object_lifecycle_trace.parquet"
    pd.DataFrame(object_rows).to_parquet(object_frame_path, index=False)
    pd.DataFrame(lifecycle_rows).to_parquet(lifecycle_path, index=False)
    pd.DataFrame(object_rows).to_csv(phase1 / "object_frame_rows.csv", index=False)
    pd.DataFrame(lifecycle_rows).to_csv(phase1 / "object_lifecycle_trace.csv", index=False)

    rolling = summary.get("v106_sam2_rolling_state", {})
    rolling_records = rolling.get("records", []) if isinstance(rolling, dict) else []
    reconsolidation_records = {
        "schema_version": "stream4d_v107_reconsolidation_records_v1",
        "source": rel(summary_path),
        "rows": [row for row in rolling_records if row.get("kind") in {"reconsolidate", "birth_recon_preprune"}],
    }
    active_state_records = {
        "schema_version": "stream4d_v107_active_state_memory_records_v1",
        "source": rel(summary_path),
        "rolling_record_count_by_kind": {
            kind: int(sum(1 for row in rolling_records if row.get("kind") == kind))
            for kind in sorted({row.get("kind") for row in rolling_records})
        },
        "rolling_records": rolling_records,
        "stream_memory_prune_events": summary.get("stream_memory_prune_events", []),
        "stream_object_prune_events": summary.get("stream_object_prune_events", []),
        "final_active_stream_object_count": summary.get("final_active_stream_object_count"),
        "final_noncond_stream_frame_count": summary.get("final_noncond_stream_frame_count"),
    }
    write_json(phase1 / "reconsolidation_records.json", reconsolidation_records)
    write_json(phase1 / "active_state_memory_records.json", active_state_records)
    write_json(phase1 / "reference_label_manifest.json", {"rows": label_manifest, "row_count": len(label_manifest)})
    video_reference = write_video_reference(summary, reference_run_root, reference_videos_dir)

    frame_count = len(records)
    exact_frame_count = int(sum(1 for row in frame_rows if row["pixel_exact_equal"]))
    fg_recall = (
        float(total_intersection_foreground_px) / float(total_reference_foreground_px)
        if total_reference_foreground_px
        else 1.0
    )
    fg_precision = (
        float(total_intersection_foreground_px) / float(total_instrumented_foreground_px)
        if total_instrumented_foreground_px
        else 1.0
    )
    full_scope_complete = False
    scene_token = str(summary.get("scene_id", "unknown_scene")).upper()
    metric_summary = {
        "schema_version": "stream4d_v107_phase1_reference_metric_summary_v1",
        "created_unix_time": time.time(),
        "reference_run_root": rel(reference_run_root),
        "reference_summary": {"path": rel(summary_path), "sha256": sha256_file(summary_path)},
        "phase0_gate": rel(Path(args.phase0_gate)) if args.phase0_gate else "",
        "scene_id": summary.get("scene_id"),
        "frame_count": frame_count,
        "label_exact_parity_pass": exact_frame_count == frame_count and pixel_mismatch_count == 0 and not missing_labels,
        "exact_frame_count": exact_frame_count,
        "pixel_mismatch_count": pixel_mismatch_count,
        "missing_labels": missing_labels,
        "reference_foreground_recall_self": fg_recall,
        "reference_foreground_precision_self": fg_precision,
        "object_frame_row_count": len(object_rows),
        "object_lifecycle_count": len(lifecycle_rows),
        "object_frame_rows_parquet": rel(object_frame_path),
        "object_lifecycle_trace_parquet": rel(lifecycle_path),
        "reconsolidation_records": rel(phase1 / "reconsolidation_records.json"),
        "active_state_memory_records": rel(phase1 / "active_state_memory_records.json"),
        "reference_labels_dir": rel(reference_labels_dir),
        "instrumented_labels_dir": rel(instrumented_labels_dir),
        "reference_video": video_reference,
        "unavailable_in_label_replay": [
            "raw SAM2 logits",
            "SAM2 pooled appearance descriptor",
            "true per-object active-slot residence",
            "scene0011 full90",
            "32/48 frame all-memory short oracle",
        ],
        "full_phase1_scope_complete": full_scope_complete,
        "decision": (
            f"PASS_PHASE1_{scene_token}_LABEL_PARITY__FULL_PHASE1_SCOPE_PENDING"
            if exact_frame_count == frame_count and pixel_mismatch_count == 0 and not missing_labels
            else "NO_GO_PHASE1_LABEL_PARITY_FAILED"
        ),
        "honesty_note": (
            "This run is reference-label instrumentation. It proves label parser/reconstruction exact parity and "
            "creates object-level trace tables from frozen v106 outputs, but it does not claim raw-logit or SAM2 "
            "descriptor instrumentation."
        ),
    }
    write_json(phase1 / "reference_metric_summary.json", metric_summary)
    write_json(
        output_root / "run_summary.json",
        {
            "schema_version": "stream4d_v107_phase1_reference_instrumentation_run_summary_v1",
            "output_root": rel(output_root),
            "reference_metric_summary": rel(phase1 / "reference_metric_summary.json"),
            "decision": metric_summary["decision"],
            "label_exact_parity_pass": metric_summary["label_exact_parity_pass"],
            "full_phase1_scope_complete": metric_summary["full_phase1_scope_complete"],
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "decision": metric_summary["decision"],
                "label_exact_parity_pass": metric_summary["label_exact_parity_pass"],
                "full_phase1_scope_complete": metric_summary["full_phase1_scope_complete"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if metric_summary["label_exact_parity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
