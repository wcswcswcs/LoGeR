#!/usr/bin/env python3
"""Run a repaired v108 Phase12 visual-first online pilot.

This runner executes the real v107/v106 rolling online stream, then adds a
v108-owned lifecycle ledger, visual review index, and transaction boundary.
It never treats metrics as acceptance gates and never admits durable SAM2
memory without explicit user visual acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "Stream3D") not in sys.path:
    sys.path.insert(1, str(ROOT / "Stream3D"))

from Stream3D.stream4d_v108.lifecycle import DelayedAdmissionPolicy  # noqa: E402
from Stream3D.stream4d_v108.transaction_manager import (  # noqa: E402
    Plane,
    TransactionManager,
)
from tools.audit_v105_4dpm_style_per_frame_segmentors import (  # noqa: E402
    PALETTE,
    read_rgb,
)


PALETTE_ARRAY = np.asarray(PALETTE, dtype=np.uint8)
EDGE_KERNEL = np.ones((3, 3), dtype=np.uint8)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def optional_existing_path(*values: Any) -> Path | None:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        path = resolve_path(text)
        if path.is_file():
            return path
    return None


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
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if not text:
            return int(default)
        return int(float(text))
    except Exception:
        return int(default)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text:
            return float(default)
        return float(text)
    except Exception:
        return float(default)


def read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.uint16, copy=False)


def object_color(obj_id: int) -> np.ndarray:
    if int(obj_id) <= 0:
        return np.array([80, 220, 120], dtype=np.uint8)
    return PALETTE_ARRAY[(int(obj_id) - 1) % len(PALETTE_ARRAY)]


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    mask_b = np.asarray(mask).astype(bool)
    out = rgb.copy()
    if not bool(np.any(mask_b)):
        return out
    blended = (
        rgb.astype(np.float32) * (1.0 - float(alpha))
        + color.reshape(1, 1, 3).astype(np.float32) * float(alpha)
    ).astype(np.uint8)
    out[mask_b] = blended[mask_b]
    edge = cv2.morphologyEx(mask_b.astype(np.uint8), cv2.MORPH_GRADIENT, EDGE_KERNEL) > 0
    out[edge] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def title_bar(image: np.ndarray, text: str) -> np.ndarray:
    pad = 44
    out = np.full((image.shape[0] + pad, image.shape[1], 3), 255, dtype=np.uint8)
    out[pad:] = image
    cv2.putText(out, text[:120], (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (10, 10, 10), 2, cv2.LINE_AA)
    return out


def component_stats(mask: np.ndarray) -> dict[str, Any]:
    mask_b = np.asarray(mask).astype(bool)
    h, w = mask_b.shape[:2]
    image_area = max(1, int(h) * int(w))
    area = int(np.count_nonzero(mask_b))
    if area <= 0:
        return {
            "area_px": 0,
            "area_frac": 0.0,
            "bbox_area_frac": 0.0,
            "bbox_extent": 0.0,
            "edge_touch_count": 0,
            "bbox_xyxy": [],
        }
    ys, xs = np.where(mask_b)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    touches = [x0 <= 0, y0 <= 0, x1 >= w - 1, y1 >= h - 1]
    return {
        "area_px": int(area),
        "area_frac": float(area / image_area),
        "bbox_area_frac": float((bw * bh) / image_area),
        "bbox_extent": float(area / max(1, bw * bh)),
        "edge_touch_count": int(sum(bool(v) for v in touches)),
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
    }


def crop_bounds(mask_a: np.ndarray, mask_b: np.ndarray | None, shape: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    union = np.asarray(mask_a).astype(bool).copy()
    if mask_b is not None:
        union |= np.asarray(mask_b).astype(bool)
    h, w = shape[:2]
    if not bool(np.any(union)):
        return 0, 0, int(w), int(h)
    ys, xs = np.where(union)
    x0 = max(0, int(xs.min()) - int(margin))
    x1 = min(w, int(xs.max()) + int(margin) + 1)
    y0 = max(0, int(ys.min()) - int(margin))
    y1 = min(h, int(ys.max()) + int(margin) + 1)
    return x0, y0, x1, y1


def resize_panel(image: np.ndarray, scale: int) -> np.ndarray:
    scale_i = max(1, int(scale))
    if scale_i == 1:
        return image
    return cv2.resize(image, (image.shape[1] * scale_i, image.shape[0] * scale_i), interpolation=cv2.INTER_NEAREST)


def record_by_frame(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["frame_id"]): row for row in summary.get("records", []) if "frame_id" in row}


def parse_reason_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text == "[]":
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(item) for item in payload if str(item)]
    except Exception:
        pass
    return [item for item in text.replace("|", ";").split(";") if item]


def visual_row_key(row: dict[str, Any]) -> tuple[int, str, int, int, int]:
    return (
        parse_int(row.get("event_index"), -1),
        str(row.get("record_type", "")),
        parse_int(row.get("frame_id"), -1),
        parse_int(row.get("live_obj_id"), parse_int(row.get("source_mapping_candidate_live_id"), -1)),
        parse_int(row.get("reference_global_id"), -1),
    )


def build_source_mapping_index(rows: list[dict[str, str]]) -> dict[tuple[int, int, int], dict[str, str]]:
    out: dict[tuple[int, int, int], dict[str, str]] = {}
    for row in rows:
        if str(row.get("record_type", "")) != "source_identity_mapping":
            continue
        key = (
            parse_int(row.get("event_index"), -1),
            parse_int(row.get("live_obj_id"), parse_int(row.get("source_mapping_candidate_live_id"), -1)),
            parse_int(row.get("reference_global_id"), -1),
        )
        if key[0] < 0 or key[1] < 0:
            continue
        if key not in out or parse_bool(row.get("source_mapping_accepted")):
            out[key] = row
    return out


def build_physical_support_stats(
    row: dict[str, str],
    source_mapping_index: dict[tuple[int, int, int], dict[str, str]],
) -> dict[str, Any] | None:
    key = (
        parse_int(row.get("event_index"), -1),
        parse_int(row.get("live_obj_id"), parse_int(row.get("source_mapping_candidate_live_id"), -1)),
        parse_int(row.get("reference_global_id"), -1),
    )
    mapping = source_mapping_index.get(key)
    if mapping is None:
        return {
            "source_mapping_found": False,
            "physical_anchor_ready": False,
            "physical_anchor_readiness_reasons": ["source_mapping_missing"],
            "geometry_available": False,
            "projected_positive_count": 0,
            "attempt_positive_after_conflict": 0,
            "confirm_positive_after_conflict": 0,
            "attempt_negative_after_stability": 0,
            "confirm_negative_after_stability": 0,
            "conflict_diagnostics": {
                "positive_negative_conflict_count": 0,
                "positive_cluster_outlier_count": 0,
            },
            "source_anchor_area_frac": 0.0,
            "source_anchor_bbox_area_frac": 0.0,
            "source_anchor_edge_touch_count": 0,
            "source_anchor_extent": 0.0,
        }
    ready = parse_bool(mapping.get("physical_anchor_ready"))
    geometry_text = str(mapping.get("physical_anchor_uses_lingbot_geometry", "")).strip()
    geometry_available = parse_bool(geometry_text) if geometry_text else bool(ready)
    attempt_positive = parse_int(mapping.get("physical_anchor_attempt_positive_after_conflict"), 0)
    confirm_positive = parse_int(mapping.get("physical_anchor_confirm_positive_after_conflict"), 0)
    positives = [value for value in (attempt_positive, confirm_positive) if value > 0]
    projected_positive_count = min(positives) if positives else 0
    conflict_count = max(
        parse_int(row.get("attempt_prompt_anchor_conflict_dropped_positive_negative_conflict_count"), 0),
        parse_int(row.get("confirm_prompt_anchor_conflict_dropped_positive_negative_conflict_count"), 0),
    )
    outlier_count = max(
        parse_int(row.get("attempt_prompt_anchor_conflict_dropped_positive_cluster_outlier_count"), 0),
        parse_int(row.get("confirm_prompt_anchor_conflict_dropped_positive_cluster_outlier_count"), 0),
    )
    return {
        "source_mapping_found": True,
        "physical_anchor_ready": bool(ready),
        "physical_anchor_readiness_reasons": parse_reason_list(mapping.get("physical_anchor_readiness_reasons")),
        "geometry_available": bool(geometry_available),
        "projected_positive_count": int(projected_positive_count),
        "attempt_positive_after_conflict": int(attempt_positive),
        "confirm_positive_after_conflict": int(confirm_positive),
        "attempt_negative_after_stability": parse_int(mapping.get("physical_anchor_attempt_negative_after_stability"), 0),
        "confirm_negative_after_stability": parse_int(mapping.get("physical_anchor_confirm_negative_after_stability"), 0),
        "conflict_diagnostics": {
            "positive_negative_conflict_count": int(conflict_count),
            "positive_cluster_outlier_count": int(outlier_count),
        },
        "source_anchor_area_frac": parse_float(mapping.get("source_anchor_area_frac"), 0.0),
        "source_anchor_bbox_area_frac": parse_float(mapping.get("source_anchor_bbox_area_frac"), 0.0),
        "source_anchor_edge_touch_count": parse_int(mapping.get("source_anchor_edge_touch_count"), 0),
        "source_anchor_extent": parse_float(mapping.get("source_anchor_extent"), 0.0),
    }


def build_temporal_support_index(
    *,
    rows: list[dict[str, str]],
    rolling_records: dict[int, dict[str, Any]],
) -> dict[tuple[int, str, int, int, int], dict[str, Any]]:
    index: dict[tuple[int, str, int, int, int], dict[str, Any]] = {}
    previous_by_object: dict[tuple[int, int], dict[str, Any]] = {}
    ordered = sorted(
        rows,
        key=lambda row: (
            parse_int(row.get("frame_id"), -1),
            parse_int(row.get("event_index"), -1),
            str(row.get("record_type", "")),
        ),
    )
    for row in ordered:
        if str(row.get("record_type", "")) not in {"shadow_output", "probation_attempt", "confirm"}:
            continue
        frame_id = parse_int(row.get("frame_id"), -1)
        live_obj_id = parse_int(row.get("live_obj_id"), -1)
        reference_obj_id = parse_int(row.get("reference_global_id"), -1)
        record = rolling_records.get(frame_id)
        if frame_id < 0 or live_obj_id < 0 or record is None:
            continue
        try:
            label = read_label(resolve_path(record["label_path"]))
        except Exception:
            continue
        mask = label == int(live_obj_id) + 1
        area_px = int(np.count_nonzero(mask))
        if area_px <= 0:
            continue
        object_key = (int(live_obj_id), int(reference_obj_id))
        prev = previous_by_object.get(object_key)
        last_iou = -1.0
        visible_frame_count = 1
        iou_sum = 0.0
        previous_frame_id = -1
        if prev is not None:
            previous_mask = prev["mask"]
            inter = int(np.count_nonzero(np.asarray(previous_mask, dtype=bool) & mask))
            union = int(np.count_nonzero(np.asarray(previous_mask, dtype=bool) | mask))
            last_iou = float(inter / union) if union > 0 else 0.0
            visible_frame_count = int(prev["visible_frame_count"]) + 1
            iou_sum = float(prev["iou_sum"]) + float(last_iou)
            previous_frame_id = int(prev["frame_id"])
        previous_by_object[object_key] = {
            "mask": mask,
            "visible_frame_count": int(visible_frame_count),
            "iou_sum": float(iou_sum),
            "frame_id": int(frame_id),
        }
        mean_iou = float(iou_sum / max(1, visible_frame_count - 1)) if visible_frame_count > 1 else -1.0
        index[visual_row_key(row)] = {
            "visible_frame_count": int(visible_frame_count),
            "mean_iou_to_previous_visible": float(mean_iou),
            "last_iou_to_previous_visible": float(last_iou),
            "previous_visible_frame_id": int(previous_frame_id),
            "current_area_px": int(area_px),
            "source": "rolling_label_past_only_temporal_support_proxy",
        }
    return index


def select_visual_rows(rows: list[dict[str, str]], max_visuals: int) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if str(row.get("record_type", "")) in {"probation_attempt", "confirm", "shadow_output"}
        and parse_int(row.get("frame_id"), -1) >= 0
        and parse_int(row.get("live_obj_id"), -1) >= 0
    ]
    if not candidates:
        candidates = [
            row
            for row in rows
            if str(row.get("record_type", "")) in {"source_identity_mapping", "demotion"}
            and parse_int(row.get("frame_id"), -1) >= 0
            and parse_int(row.get("live_obj_id"), parse_int(row.get("source_mapping_candidate_live_id"), -1)) >= 0
        ]

    def score(row: dict[str, str]) -> tuple[int, int, int, int, int]:
        record_type = str(row.get("record_type", ""))
        output = parse_bool(row.get("output_mask")) or parse_bool(row.get("target_present"))
        committed = parse_bool(row.get("reactivation_committed_to_sam2_video_state"))
        has_event_visual = optional_existing_path(row.get("visual_path"), row.get("all_prompt_visual_path")) is not None
        priority = {
            "probation_attempt": 0,
            "confirm": 1,
            "shadow_output": 2,
            "source_identity_mapping": 3,
            "demotion": 4,
        }.get(record_type, 9)
        return (
            0 if has_event_visual else 1,
            0 if output or committed else 1,
            priority,
            parse_int(row.get("event_index"), 999999),
            parse_int(row.get("frame_id"), 999999),
        )

    selected: list[dict[str, str]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for row in sorted(candidates, key=score):
        obj_id = parse_int(row.get("live_obj_id"), parse_int(row.get("source_mapping_candidate_live_id"), -1))
        event_index = parse_int(row.get("event_index"), -1)
        if event_index >= 0:
            key = ("event", event_index, obj_id, "")
        else:
            key = ("frame", parse_int(row.get("frame_id"), -1), obj_id, str(row.get("record_type", "")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= int(max_visuals):
            break
    return selected


def build_visual_panels(
    *,
    scene_id: str,
    output_root: Path,
    v107_root: Path,
    v107_summary: dict[str, Any],
    rolling_summary: dict[str, Any],
    all_record_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    reference_run_root: Path,
    max_visuals: int,
    visual_scale: int,
    crop_margin: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    visual_dir = output_root / "phase12_highres_visual_review"
    visual_dir.mkdir(parents=True, exist_ok=True)
    rolling_records = record_by_frame(rolling_summary)
    source_mapping_index = build_source_mapping_index(all_record_rows)
    temporal_support_index = build_temporal_support_index(rows=all_record_rows, rolling_records=rolling_records)
    lifecycle_policy = DelayedAdmissionPolicy()
    visual_rows: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    manager = TransactionManager()

    for index, row in enumerate(selected_rows[: int(max_visuals)]):
        frame_id = parse_int(row.get("frame_id"), -1)
        live_obj_id = parse_int(row.get("live_obj_id"), parse_int(row.get("source_mapping_candidate_live_id"), -1))
        reference_obj_id = parse_int(row.get("reference_global_id"), -1)
        record = rolling_records.get(frame_id)
        if record is None or live_obj_id < 0:
            continue
        rgb_text = str(record.get("rgb_path") or "").strip()
        rgb_path = resolve_path(rgb_text) if rgb_text else Path("")
        if not rgb_text or not rgb_path.is_file():
            rgb_root = resolve_path(rolling_summary.get("rgb_root", "Stream3D/data/scannet/processed"))
            rgb_path = rgb_root / scene_id / "color" / f"{frame_id}.jpg"
        label_path = resolve_path(record["label_path"])
        rgb = read_rgb(rgb_path)
        label = read_label(label_path)
        if label.shape[:2] != rgb.shape[:2]:
            label = cv2.resize(label, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        # v105/v106 label images reserve 0 for background and store object ids as obj_id + 1.
        candidate_label_value = int(live_obj_id) + 1
        candidate_mask = label == int(candidate_label_value)

        ref_mask = None
        ref_label_path = reference_run_root / "labels" / f"frame_{frame_id:06d}.png"
        if reference_obj_id > 0 and ref_label_path.exists():
            ref_label = read_label(ref_label_path)
            if ref_label.shape[:2] != rgb.shape[:2]:
                ref_label = cv2.resize(ref_label, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            ref_mask = ref_label == int(reference_obj_id)

        x0, y0, x1, y1 = crop_bounds(candidate_mask, ref_mask, rgb.shape[:2], int(crop_margin))
        rgb_crop = rgb[y0:y1, x0:x1]
        cand_crop = candidate_mask[y0:y1, x0:x1]
        ref_crop = ref_mask[y0:y1, x0:x1] if ref_mask is not None else None

        panels = [
            title_bar(rgb_crop, f"RGB scene={scene_id} frame={frame_id}"),
            title_bar(
                overlay_mask(rgb_crop, cand_crop, object_color(live_obj_id)),
                f"online output mask live_obj={live_obj_id} label={candidate_label_value}",
            ),
        ]
        if ref_crop is not None:
            panels.append(
                title_bar(
                    overlay_mask(rgb_crop, ref_crop, np.array([45, 190, 255], dtype=np.uint8)),
                    f"reference diagnostic only ref_obj={reference_obj_id}",
                )
            )
            diff = rgb_crop.copy()
            cand_only = cand_crop & ~ref_crop
            ref_only = ref_crop & ~cand_crop
            both = cand_crop & ref_crop
            diff[both] = np.array([180, 180, 180], dtype=np.uint8)
            diff[cand_only] = np.array([255, 60, 50], dtype=np.uint8)
            diff[ref_only] = np.array([40, 180, 255], dtype=np.uint8)
            panels.append(title_bar(diff, "diagnostic diff red=online only cyan=reference only"))

        max_h = max(panel.shape[0] for panel in panels)
        padded: list[np.ndarray] = []
        for panel in panels:
            if panel.shape[0] < max_h:
                pad = np.full((max_h - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
                panel = np.concatenate([panel, pad], axis=0)
            padded.append(panel)
        full = np.concatenate(padded, axis=1)
        full = resize_panel(full, int(visual_scale))
        out_path = visual_dir / (
            f"phase12_case_{index:02d}_event{parse_int(row.get('event_index'), -1):03d}_"
            f"f{frame_id:06d}_live{live_obj_id:04d}.png"
        )
        cv2.imwrite(str(out_path), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
        generated_panel_sha256 = sha256_file(out_path)
        source_event_visual_path = optional_existing_path(row.get("visual_path"), row.get("all_prompt_visual_path"))
        source_event_visual_sha256 = sha256_file(source_event_visual_path) if source_event_visual_path is not None else ""
        primary_visual_path = source_event_visual_path or out_path
        primary_visual_sha256 = source_event_visual_sha256 or generated_panel_sha256
        event_specific_visual_used = source_event_visual_path is not None

        stats = component_stats(candidate_mask)
        record_type = str(row.get("record_type", ""))
        if record_type == "probation_attempt":
            event_prompt_area_px = parse_int(row.get("probation_mask_area_px"), -1)
        elif record_type == "confirm":
            event_prompt_area_px = parse_int(row.get("confirm_mask_area_px"), -1)
        else:
            event_prompt_area_px = -1
        committed_label_area_px = int(stats["area_px"])
        if event_prompt_area_px > 0 or committed_label_area_px > 0:
            denom = max(event_prompt_area_px, committed_label_area_px, 1)
            event_prompt_vs_committed_area_delta_frac = abs(event_prompt_area_px - committed_label_area_px) / float(denom)
        else:
            event_prompt_vs_committed_area_delta_frac = -1.0
        review_status = "USER_REVIEW_PENDING"
        physical_support_stats = build_physical_support_stats(row, source_mapping_index)
        temporal_support_stats = temporal_support_index.get(
            visual_row_key(row),
            {
                "visible_frame_count": 0,
                "mean_iou_to_previous_visible": -1.0,
                "source": "missing_temporal_support_proxy",
            },
        )
        diagnostic = lifecycle_policy.evaluate(
            frame_id=frame_id,
            global_object_id=live_obj_id,
            component_stats=stats,
            watcher_stats=temporal_support_stats,
            physical_support_stats=physical_support_stats,
            visual_review_status=review_status,
        )
        block_reasons_except_user_review = [
            reason for reason in diagnostic.reasons if reason != "visual_review_not_accepted_for_durable_memory"
        ]
        tx = manager.propose(
            frame_id=frame_id,
            global_object_id=live_obj_id,
            plane=Plane.OUTPUT,
            action="show_online_mask_pending_visual_review",
            evidence={
                "visual_path": rel(primary_visual_path),
                "visual_sha256": primary_visual_sha256,
                "event_specific_visual_used": bool(event_specific_visual_used),
                "source_event_visual_path": rel(source_event_visual_path) if source_event_visual_path is not None else "",
                "source_event_visual_sha256": source_event_visual_sha256,
                "generated_final_label_panel_path": rel(out_path),
                "generated_final_label_panel_sha256": generated_panel_sha256,
                "source_record_type": row.get("record_type", ""),
            },
        )
        tx = manager.apply_output_only(tx, "online output is visible for review; durable memory remains blocked")

        visual_row = {
            "scene_id": scene_id,
            "case_index": int(index),
            "event_index": parse_int(row.get("event_index"), -1),
            "record_type": row.get("record_type", ""),
            "frame_id": int(frame_id),
            "live_obj_id": int(live_obj_id),
            "candidate_label_value": int(candidate_label_value),
            "reference_obj_id": int(reference_obj_id),
            "visual_path": rel(primary_visual_path),
            "visual_sha256": primary_visual_sha256,
            "primary_visual_kind": "event_specific_prompt_visual" if event_specific_visual_used else "generated_final_label_diagnostic_panel",
            "event_specific_visual_used": bool(event_specific_visual_used),
            "source_event_visual_path": rel(source_event_visual_path) if source_event_visual_path is not None else "",
            "source_event_visual_sha256": source_event_visual_sha256,
            "generated_final_label_panel_path": rel(out_path),
            "generated_final_label_panel_sha256": generated_panel_sha256,
            "rgb_path": rel(rgb_path),
            "candidate_label_path": rel(label_path),
            "reference_label_path": rel(ref_label_path) if ref_label_path.exists() else "",
            "crop_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "visual_review_status": review_status,
            "visual_review_required": True,
            "visual_decision_is_final_gate": True,
            "metrics_are_diagnostic_only": True,
            "quality_decision_rule": "Only high-resolution visual confirmation can decide good or bad.",
            "generated_final_label_panel_is_drift_diagnostic_only": True,
            "reference_overlay_is_diagnostic_only": bool(ref_mask is not None),
            "event_prompt_and_committed_label_must_be_compared_visually": bool(event_specific_visual_used),
            "visual_acceptance_requires_same_object_in_event_prompt_and_committed_label": True,
            "event_prompt_output_area_px_diagnostic_only": int(event_prompt_area_px),
            "committed_label_area_px_diagnostic_only": int(committed_label_area_px),
            "event_prompt_vs_committed_area_delta_frac_diagnostic_only": float(
                event_prompt_vs_committed_area_delta_frac
            ),
            "candidate_area_px": int(stats["area_px"]),
            "candidate_bbox_xyxy": stats["bbox_xyxy"],
            "candidate_edge_touch_count": int(stats["edge_touch_count"]),
            "source_g3_record_skip_reason": row.get("skip_reason", ""),
            "source_g3_selected_variant": row.get("selected_variant", ""),
            "temporal_support_source": temporal_support_stats.get("source", ""),
            "temporal_visible_frame_count_diagnostic_only": int(
                temporal_support_stats.get("visible_frame_count", 0)
            ),
            "temporal_mean_iou_to_previous_visible_diagnostic_only": float(
                temporal_support_stats.get("mean_iou_to_previous_visible", -1.0)
            ),
            "temporal_last_iou_to_previous_visible_diagnostic_only": float(
                temporal_support_stats.get("last_iou_to_previous_visible", -1.0)
            ),
            "temporal_previous_visible_frame_id_diagnostic_only": int(
                temporal_support_stats.get("previous_visible_frame_id", -1)
            ),
            "physical_support_source_mapping_found": (
                bool(physical_support_stats.get("source_mapping_found")) if physical_support_stats else False
            ),
            "physical_anchor_ready_diagnostic_only": (
                bool(physical_support_stats.get("physical_anchor_ready")) if physical_support_stats else False
            ),
            "physical_anchor_readiness_reasons_diagnostic_only": (
                list(physical_support_stats.get("physical_anchor_readiness_reasons", []))
                if physical_support_stats
                else []
            ),
            "physical_projected_positive_count_diagnostic_only": (
                int(physical_support_stats.get("projected_positive_count", 0)) if physical_support_stats else 0
            ),
            "physical_attempt_positive_after_conflict_diagnostic_only": (
                int(physical_support_stats.get("attempt_positive_after_conflict", 0)) if physical_support_stats else 0
            ),
            "physical_confirm_positive_after_conflict_diagnostic_only": (
                int(physical_support_stats.get("confirm_positive_after_conflict", 0)) if physical_support_stats else 0
            ),
            "physical_attempt_negative_after_stability_diagnostic_only": (
                int(physical_support_stats.get("attempt_negative_after_stability", 0)) if physical_support_stats else 0
            ),
            "physical_confirm_negative_after_stability_diagnostic_only": (
                int(physical_support_stats.get("confirm_negative_after_stability", 0)) if physical_support_stats else 0
            ),
            "durable_memory_allowed_except_user_review": len(block_reasons_except_user_review) == 0,
            "durable_memory_block_reasons_except_user_review": block_reasons_except_user_review,
        }
        visual_rows.append(visual_row)
        lifecycle_rows.append(
            {
                **visual_row,
                "output_state": diagnostic.output_state.value,
                "output_allowed": bool(diagnostic.output_allowed),
                "durable_memory_allowed": bool(diagnostic.durable_memory_allowed),
                "durable_memory_block_reasons": list(diagnostic.reasons),
                "transaction_id": tx.transaction_id,
                "transaction_status": tx.status.value,
                "transaction_plane": tx.plane.value,
                "transaction_reason": tx.reason,
            }
        )
    return visual_rows, lifecycle_rows


def collect_file_asset(path_value: Any) -> dict[str, Any]:
    text = str(path_value or "").strip()
    if not text:
        return {"path": "", "exists": False, "sha256": ""}
    path = resolve_path(text)
    if not path.is_file():
        return {"path": rel(path), "exists": False, "sha256": ""}
    return {"path": rel(path), "exists": True, "sha256": sha256_file(path)}


def collect_visual_video_asset(g3_summary: dict[str, Any]) -> dict[str, Any]:
    visual = dict(g3_summary.get("visual") or {})
    video_asset = collect_file_asset(visual.get("path", ""))
    panel_assets = [collect_file_asset(item) for item in visual.get("saved_panel_frame_paths", [])]
    return {
        "schema_version": visual.get("schema_version", ""),
        "path": video_asset["path"],
        "exists": bool(video_asset["exists"]),
        "sha256": video_asset["sha256"],
        "frame_count": visual.get("frame_count"),
        "saved_panel_frame_count": visual.get("saved_panel_frame_count"),
        "saved_panel_frames": panel_assets,
        "layout": visual.get("layout", ""),
        "source": visual.get("source", ""),
        "visual_review_status": g3_summary.get("visual_review_status", "USER_VISUAL_REVIEW_PENDING"),
    }


def visual_asset_rows(items: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return rows
    for item in items:
        if not isinstance(item, dict):
            continue
        asset = collect_file_asset(item.get("path", ""))
        rows.append(
            {
                "path": asset["path"],
                "exists": bool(asset["exists"]),
                "sha256": asset["sha256"] or str(item.get("sha256", "")),
                "source_sha256": str(item.get("sha256", "")),
            }
        )
    return rows


def write_phase12_casebook(
    *,
    output_root: Path,
    scene_id: str,
    preset: str,
    g3_summary: dict[str, Any],
    visual_rows: list[dict[str, Any]],
    lifecycle_rows: list[dict[str, Any]],
    transaction_rows: list[dict[str, Any]],
    visual_csv: Path,
    visual_json: Path,
    lifecycle_csv: Path,
    transaction_csv: Path,
) -> tuple[Path, Path]:
    casebook_dir = output_root / "phase12_casebook"
    casebook_dir.mkdir(parents=True, exist_ok=True)
    casebook_manifest = casebook_dir / "casebook_manifest.json"
    casebook_markdown = casebook_dir / "casebook.md"
    video = collect_visual_video_asset(g3_summary)
    event_visuals = visual_asset_rows(g3_summary.get("highres_visuals"))
    probation_visuals = visual_asset_rows(g3_summary.get("highres_probation_visuals"))
    shadow_visuals = visual_asset_rows(g3_summary.get("highres_shadow_visuals"))
    payload = {
        "schema_version": "stream4d_v108_phase12_casebook_v1",
        "scene_id": scene_id,
        "preset": preset,
        "review_status": "USER_REVIEW_PENDING",
        "metrics_are_diagnostic_only": True,
        "quality_decision_rule": "Only high-resolution visual confirmation can decide good or bad.",
        "full_scene_video": video,
        "event_specific_visuals": event_visuals,
        "probation_visuals": probation_visuals,
        "shadow_visuals": shadow_visuals,
        "phase12_visual_review_index_csv": rel(visual_csv),
        "phase12_visual_review_index_json": rel(visual_json),
        "phase12_lifecycle_admission_rows_csv": rel(lifecycle_csv),
        "phase12_transaction_boundary_rows_csv": rel(transaction_csv),
        "visual_review_records": visual_rows,
        "lifecycle_record_count": int(len(lifecycle_rows)),
        "transaction_record_count": int(len(transaction_rows)),
        "durable_memory_mutation_request_count": 0,
        "diagnostic_v107_counts": {
            "event_count": g3_summary.get("event_count"),
            "probation_output_mask_count": g3_summary.get("probation_output_mask_count"),
            "shadow_output_mask_count": g3_summary.get("shadow_output_mask_count"),
            "actual_video_readd_record_count": g3_summary.get("actual_video_readd_record_count"),
            "long_term_memory_admitted_count_in_v107_scheduler": g3_summary.get("long_term_memory_admitted_count"),
        },
    }
    write_json(casebook_manifest, payload)
    lines = [
        f"# Phase12 Casebook: {preset}",
        "",
        "review_status: USER_REVIEW_PENDING",
        "metrics_are_diagnostic_only: true",
        "quality_decision_rule: Only high-resolution visual confirmation can decide good or bad.",
        "",
        "## Full Scene Video",
        "",
        f"path: {video['path']}",
        f"exists: {video['exists']}",
        f"sha256: {video['sha256']}",
        f"frame_count: {video['frame_count']}",
        "",
        "## Primary Visual Review Records",
        "",
    ]
    for row in visual_rows:
        lines.extend(
            [
                f"- case {row.get('case_index')} event {row.get('event_index')} "
                f"frame {row.get('frame_id')} live_obj {row.get('live_obj_id')}",
                f"  primary_visual_kind: {row.get('primary_visual_kind')}",
                f"  visual_path: {row.get('visual_path')}",
                f"  visual_sha256: {row.get('visual_sha256')}",
                f"  source_event_visual_path: {row.get('source_event_visual_path')}",
                f"  generated_final_label_panel_path: {row.get('generated_final_label_panel_path')}",
                "  visual_acceptance_rule: compare the event prompt visual and the generated final label panel; "
                "if they show different objects, reject durable memory for this case.",
                f"  event_prompt_area_px_diagnostic_only: "
                f"{row.get('event_prompt_output_area_px_diagnostic_only')}",
                f"  committed_label_area_px_diagnostic_only: "
                f"{row.get('committed_label_area_px_diagnostic_only')}",
                f"  event_prompt_vs_committed_area_delta_frac_diagnostic_only: "
                f"{row.get('event_prompt_vs_committed_area_delta_frac_diagnostic_only')}",
                f"  temporal_visible_frame_count_diagnostic_only: "
                f"{row.get('temporal_visible_frame_count_diagnostic_only')}",
                f"  temporal_mean_iou_to_previous_visible_diagnostic_only: "
                f"{row.get('temporal_mean_iou_to_previous_visible_diagnostic_only')}",
                f"  physical_anchor_ready_diagnostic_only: "
                f"{row.get('physical_anchor_ready_diagnostic_only')}",
                f"  physical_projected_positive_count_diagnostic_only: "
                f"{row.get('physical_projected_positive_count_diagnostic_only')}",
                f"  durable_memory_allowed_except_user_review: "
                f"{row.get('durable_memory_allowed_except_user_review')}",
                f"  durable_memory_block_reasons_except_user_review: "
                f"{row.get('durable_memory_block_reasons_except_user_review')}",
                "",
            ]
        )
    casebook_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return casebook_manifest, casebook_markdown


def base_v107_args_for_preset(preset: str, v107_root: Path, gpu: str) -> tuple[list[str], dict[str, Any]]:
    common = [
        "--gpu",
        str(gpu),
        "--rgb-root",
        "Stream3D/data/scannet/processed",
        "--events",
        "auto",
        "--auto-max-events-per-object",
        "2",
        "--auto-min-target-source-area",
        "0",
        "--auto-min-positive-points",
        "1",
        "--auto-min-confirm-positive-points",
        "1",
        "--auto-max-events",
        "0",
        "--long-term-min-source-area",
        "20000",
        "--long-term-min-positive-points",
        "1",
        "--long-term-min-confirm-positive-points",
        "1",
        "--long-term-max-events",
        "0",
        "--long-term-anchor-max-area-frac",
        "0.08",
        "--long-term-anchor-max-bbox-frac",
        "0.25",
        "--long-term-anchor-max-edge-touch-count",
        "1",
        "--reactivation-prompt-mode",
        "lingbot_geometry",
        "--prompt-core-min-source-mask-distance-px",
        "6.0",
        "--prompt-source-core-supplement-positive-points",
        "8",
        "--prompt-source-core-supplement-trigger-max-positive-points",
        "6",
        "--prompt-source-core-supplement-min-distance-px",
        "8.0",
        "--prompt-source-core-supplement-depth-abs-tolerance",
        "0.12",
        "--prompt-source-core-supplement-depth-rel-tolerance",
        "0.08",
        "--prompt-source-core-supplement-min-depth-conf",
        "1.0",
        "--prompt-source-core-supplement-duplicate-radius-px",
        "2.0",
        "--prompt-source-core-supplement-negative-points",
        "4",
        "--prompt-source-core-supplement-negative-trigger-max-negative-points",
        "6",
        "--prompt-source-core-supplement-negative-min-distance-px",
        "8.0",
        "--prompt-source-core-supplement-negative-max-neighbor-bbox-distance-px",
        "96",
        "--prompt-source-core-supplement-negative-target-border-margin-px",
        "32",
        "--prompt-source-core-supplement-negative-min-area-px",
        "64",
        "--prompt-source-core-supplement-negative-max-objects",
        "4",
        "--prompt-target-stability-depth-radius-px",
        "4",
        "--prompt-target-stability-max-local-depth-range-m",
        "0.08",
        "--prompt-target-stability-max-depth-abs-error",
        "0.08",
        "--prompt-target-stability-min-depth-conf",
        "1.0",
        "--prompt-target-stability-min-valid-depth-count",
        "40",
        "--prompt-anchor-conflict-negative-radius-px",
        "16",
        "--prompt-anchor-conflict-positive-cluster-radius-px",
        "96",
        "--prompt-anchor-conflict-min-positive-points",
        "2",
        "--prompt-target-mask-core-min-distance-px",
        "16.0",
        "--prompt-target-mask-core-min-positive-points",
        "2",
        "--birth-admission-max-uncovered-ratio",
        "0.50",
        "--birth-admission-max-edge-touch-count",
        "0",
        "--birth-admission-shape-min-uncovered-ratio",
        "0.95",
        "--gap-output-min-input-mask-count",
        "2",
        "--stream-growth-prune-ratio",
        "4.0",
        "--stream-growth-prune-min-area",
        "100000",
        "--stream-growth-prune-history",
        "6",
        "--stream-growth-prune-warmup",
        "4",
        "--stream-growth-prune-max-history-median-area",
        "10000",
        "--reactivation-probation-mode",
        "shadow_attempt_confirm_commit",
        "--probation-output-mode",
        "image_g3_selector",
        "--probation-min-positive-support",
        "0.50",
        "--online-select-neg-conflict-threshold",
        "0.05",
        "--online-select-min-g2-positive-support",
        "0.50",
        "--image-g3-selector-g2-eval-policy",
        "always_if_negatives",
        "--image-g3-selector-g2-select-policy",
        "strict_improvement",
        "--image-g3-selector-g2-min-neg-conflict-improvement",
        "0.0",
        "--output-root",
        v107_root.as_posix(),
    ]

    if preset in {"scene0030_pilot", "scene0030_full90"}:
        meta = {
            "scene_id": "scene0030_00",
            "reference_run_root": "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_reference90_20260714_0305/v106_stateful_sam2_rolling_scene_stream",
            "default_frame_start": 0,
            "default_frame_count": 30 if preset == "scene0030_pilot" else 90,
            "default_visual_events": "3,5",
            "default_probation_visual_events": "3,5",
        }
        args = [
            "--scene-id",
            meta["scene_id"],
            "--scene-root",
            "Stream3D/data/scannet/processed/scene0030_00",
            "--prompt-probe-root",
            "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_prompt_capsule90_20260714_0307",
            "--probe-root",
            "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_live_state_probe_events0_5_20260714_0310",
            "--reference-run-root",
            meta["reference_run_root"],
            "--auto-source-lags",
            "1",
            "--unmapped-source-policy",
            "skip",
            "--shadow-output-mode",
            "none",
            "--gap-output-max-edge-touch-count",
            "0",
            "--gap-output-shape-min-uncovered-ratio",
            "0.95",
        ]
        return args + common, meta

    if preset in {"scene0011_full90"}:
        meta = {
            "scene_id": "scene0011_00",
            "reference_run_root": "Stream3D/outputs/audit/v107_phase1_scene0011_v106_currentbest_full90_20260713_182708/v106_stateful_sam2_rolling_scene_stream",
            "default_frame_start": 0,
            "default_frame_count": 90,
            "default_visual_events": "0",
            "default_probation_visual_events": "0",
        }
        args = [
            "--scene-id",
            meta["scene_id"],
            "--scene-root",
            "Stream3D/data/scannet/processed/scene0011_00",
            "--prompt-probe-root",
            "Stream3D/outputs/audit/v107_phase10_scene0011_prompt_capsule_full90_20260714_011434",
            "--probe-root",
            "Stream3D/outputs/audit/v107_phase10_scene0011_live_state_probe_events012_20260714_011921",
            "--reference-run-root",
            meta["reference_run_root"],
            "--auto-source-lags",
            "1",
            "--unmapped-source-policy",
            "prompt_new_object",
            "--shadow-output-mode",
            "image_g1",
            "--shadow-min-source-area",
            "20000",
            "--shadow-min-positive-support",
            "0.50",
            "--shadow-max-events-per-frame",
            "4",
            "--gap-min-image-edge-distance-px",
            "32",
            "--gap-output-max-bbox-frac",
            "0.25",
            "--gap-output-max-edge-touch-count",
            "-1",
            "--gap-output-shape-min-uncovered-ratio",
            "0.0",
        ]
        return args + common, meta

    if preset in {"scene0050_pilot", "scene0050_full90", "scene0050_full99"}:
        is_pilot = preset == "scene0050_pilot"
        is_full90 = preset == "scene0050_full90"
        meta = {
            "scene_id": "scene0050_00",
            "reference_run_root": (
                "Stream3D/outputs/audit/v106_stateful_sam2_rolling_scene0050_area20k_e1_preprune6_maxvis45_labelcompact_noempty_full90_gpu6_20260713_1505/v106_stateful_sam2_rolling_scene_stream"
                if is_full90
                else "Stream3D/outputs/audit/v106_stateful_sam2_rolling_scene0050_area20k_e1_preprune6_maxvis45_labelcompact_noempty_full99_gpu6_20260714_1145/v106_stateful_sam2_rolling_scene_stream"
            ),
            "default_frame_start": 4450 if is_pilot else 4160,
            "default_frame_count": 11 if is_pilot else (90 if is_full90 else 99),
            "default_visual_events": "6,8,12,13,18",
            "default_probation_visual_events": "6,8,12,13,18",
        }
        args = [
            "--scene-id",
            meta["scene_id"],
            "--scene-root",
            "Stream3D/data/scannet/processed/scene0050_00",
            "--prompt-probe-root",
            "Stream3D/outputs/audit/v107_phase8_lingbot_prompt_capsule_alllag_confirm_20260713_2308",
            "--probe-root",
            "Stream3D/outputs/audit/v107_phase8_sam2_live_state_reactivation_probe24_confirm_reprompt_g3selector_vis3_20260714_0046",
            "--reference-run-root",
            meta["reference_run_root"],
            "--unmapped-source-policy",
            "prompt_new_object",
            "--shadow-output-mode",
            "image_g1",
            "--shadow-min-source-area",
            "20000",
            "--shadow-min-positive-support",
            "0.50",
            "--shadow-max-events-per-frame",
            "4",
            "--gap-min-image-edge-distance-px",
            "32",
            "--gap-output-max-bbox-frac",
            "0.25",
            "--gap-output-max-edge-touch-count",
            "-1",
            "--gap-output-shape-min-uncovered-ratio",
            "0.0",
        ]
        return args + common, meta

    raise ValueError(f"unknown preset {preset}")


def build_v107_command(args: argparse.Namespace, v107_root: Path) -> tuple[list[str], dict[str, Any]]:
    base_args, meta = base_v107_args_for_preset(str(args.preset), v107_root, str(args.gpu))
    frame_start = int(args.frame_start) if args.frame_start is not None else int(meta["default_frame_start"])
    frame_count = int(args.frame_count) if args.frame_count is not None else int(meta["default_frame_count"])
    visual_events = str(args.v107_visual_events) if args.v107_visual_events is not None else str(meta["default_visual_events"])
    probation_visual_events = (
        str(args.v107_probation_visual_events)
        if args.v107_probation_visual_events is not None
        else str(meta["default_probation_visual_events"])
    )
    full_args = [
        *base_args,
        "--frame-start",
        str(frame_start),
        "--frame-stride",
        str(args.frame_stride),
        "--frame-count",
        str(frame_count),
        "--visual-events",
        visual_events,
        "--probation-visual-events",
        probation_visual_events,
        "--shadow-visual-events",
        str(args.v107_shadow_visual_events or ""),
        "--shadow-visual-frame-ids",
        str(args.v107_shadow_visual_frame_ids or ""),
    ]
    ablation_args: list[str] = []
    phase14_ablation = str(args.phase14_ablation or "")
    if phase14_ablation == "no_lingbot":
        ablation_args.extend(["--reactivation-prompt-mode", "no_geometry"])
    elif phase14_ablation == "appearance_only":
        ablation_args.extend(["--reactivation-prompt-mode", "appearance_only"])
    elif phase14_ablation == "no_output_plane":
        ablation_args.append("--disable-output-plane")
    elif phase14_ablation == "no_gap_graph":
        ablation_args.append("--disable-gap-birth")
    elif phase14_ablation == "no_appearance":
        ablation_args.append("--disable-birth-admission-appearance")
    elif phase14_ablation == "no_watcher":
        ablation_args.extend(["--shadow-output-mode", "none"])
    elif phase14_ablation == "random_geometry":
        ablation_args.extend(
            [
                "--reactivation-prompt-mode",
                "random_geometry",
                "--prompt-source-core-supplement-positive-points",
                "0",
                "--prompt-source-core-supplement-negative-points",
                "0",
                "--prompt-target-stability-min-valid-depth-count",
                "0",
                "--prompt-target-stability-max-local-depth-range-m",
                "0",
                "--prompt-target-stability-max-depth-abs-error",
                "0",
                "--prompt-target-stability-min-depth-conf",
                "0",
                "--prompt-anchor-conflict-negative-radius-px",
                "0",
            ]
        )
    elif phase14_ablation == "single_point_gap":
        ablation_args.extend(
            [
                "--gap-max-points",
                "1",
                "--gap-max-points-per-component",
                "1",
                "--gap-area-per-extra-point",
                "1000000000",
            ]
        )
    elif phase14_ablation == "no_transaction":
        ablation_args.append("--disable-birth-transaction")
    elif phase14_ablation == "no_growth_repair":
        ablation_args.extend(
            [
                "--stream-growth-prune-ratio",
                "0.0",
                "--stream-growth-prune-min-area",
                "0",
                "--stream-growth-prune-max-history-median-area",
                "0",
            ]
        )
    elif phase14_ablation == "immediate_admission":
        ablation_args.extend(
            [
                "--birth-admission-immediate-area",
                "1",
                "--birth-transaction-min-pending",
                "1",
                "--birth-transaction-max-delay-frames",
                "0",
            ]
        )
    full_args.extend(ablation_args)
    meta.update(
        {
            "frame_start": frame_start,
            "frame_stride": int(args.frame_stride),
            "frame_count": frame_count,
            "visual_events": visual_events,
            "probation_visual_events": probation_visual_events,
            "shadow_visual_events": str(args.v107_shadow_visual_events or ""),
            "phase14_ablation": phase14_ablation,
            "phase14_ablation_args": ablation_args,
        }
    )
    return [sys.executable, (ROOT / "tools/run_v107_phase8_g3_rolling_scheduler_smoke.py").as_posix(), *full_args], meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--preset",
        required=True,
        choices=["scene0050_pilot", "scene0050_full90", "scene0050_full99", "scene0030_pilot", "scene0030_full90", "scene0011_full90"],
    )
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--existing-v107-root", default="")
    parser.add_argument("--skip-v107-run", action="store_true", default=False)
    parser.add_argument("--max-visuals", type=int, default=4)
    parser.add_argument("--visual-scale", type=int, default=2)
    parser.add_argument("--crop-margin", type=int, default=120)
    parser.add_argument("--v107-visual-events", default=None)
    parser.add_argument("--v107-probation-visual-events", default=None)
    parser.add_argument("--v107-shadow-visual-events", default="")
    parser.add_argument("--v107-shadow-visual-frame-ids", default="")
    parser.add_argument(
        "--phase14-ablation",
        default="",
        choices=[
            "",
            "baseline_cold",
            "no_lingbot",
            "appearance_only",
            "no_output_plane",
            "no_gap_graph",
            "no_appearance",
            "no_watcher",
            "random_geometry",
            "single_point_gap",
            "no_transaction",
            "no_growth_repair",
            "immediate_admission",
        ],
        help="Optional Phase14 ablation mapped to real v107 CLI switches. Unsupported ablations are not faked.",
    )
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

    v107_root = resolve_path(args.existing_v107_root) if args.existing_v107_root else output_root / "v107_online_runner"
    command, meta = build_v107_command(args, v107_root)
    (output_root / "phase12_v107_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")

    run_exit_code = 0
    run_stdout = ""
    run_stderr = ""
    full_online_run_executed = False
    if not bool(args.skip_v107_run):
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        run_exit_code = int(completed.returncode)
        run_stdout = completed.stdout
        run_stderr = completed.stderr
        full_online_run_executed = run_exit_code == 0
        (output_root / "phase12_v107_stdout.txt").write_text(run_stdout, encoding="utf-8")
        (output_root / "phase12_v107_stderr.txt").write_text(run_stderr, encoding="utf-8")
        if run_exit_code != 0:
            summary = {
                "schema_version": "stream4d_v108_phase12_full_online_summary_v1",
                "status": "FAILED_V107_ONLINE_RUN",
                "full_online_run_executed": False,
                "v107_exit_code": run_exit_code,
                "v107_command": command,
                "stdout_path": rel(output_root / "phase12_v107_stdout.txt"),
                "stderr_path": rel(output_root / "phase12_v107_stderr.txt"),
                "metrics_are_diagnostic_only": True,
                "visual_confirmation_required": True,
            }
            summary_path = output_root / "phase12_full_online_summary.json"
            write_json(summary_path, summary)
            print(json.dumps({"status": summary["status"], "summary": rel(summary_path)}, sort_keys=True))
            return 1

    g3_summary_path = v107_root / "g3_scheduler_summary.json"
    if not g3_summary_path.exists():
        raise FileNotFoundError(g3_summary_path)
    g3_summary = read_json(g3_summary_path)
    rolling_summary_path = resolve_path(g3_summary["rolling_summary"])
    rolling_summary = read_json(rolling_summary_path)
    records_csv = resolve_path(g3_summary["records_csv"])
    record_rows = read_csv_rows(records_csv)
    selected_rows = select_visual_rows(record_rows, int(args.max_visuals))
    reference_run_root = resolve_path(str(meta["reference_run_root"]))
    visual_rows, lifecycle_rows = build_visual_panels(
        scene_id=str(meta["scene_id"]),
        output_root=output_root,
        v107_root=v107_root,
        v107_summary=g3_summary,
        rolling_summary=rolling_summary,
        all_record_rows=record_rows,
        selected_rows=selected_rows,
        reference_run_root=reference_run_root,
        max_visuals=int(args.max_visuals),
        visual_scale=int(args.visual_scale),
        crop_margin=int(args.crop_margin),
    )
    visual_csv = output_root / "phase12_visual_review_index.csv"
    visual_json = output_root / "phase12_visual_review_index.json"
    lifecycle_csv = output_root / "phase12_lifecycle_admission_rows.csv"
    write_csv(visual_csv, visual_rows)
    write_json(
        visual_json,
        {
            "schema_version": "stream4d_v108_phase12_visual_review_index_v1",
            "records": visual_rows,
            "manual_review_required": True,
            "metrics_are_diagnostic_only": True,
            "quality_decision_rule": "Only high-resolution visual confirmation can decide good or bad.",
        },
    )
    write_csv(lifecycle_csv, lifecycle_rows)

    transaction_rows = [
        {
            "scene_id": row["scene_id"],
            "frame_id": row["frame_id"],
            "global_object_id": row["live_obj_id"],
            "output_transaction_status": row["transaction_status"],
            "durable_memory_allowed": row["durable_memory_allowed"],
            "durable_memory_allowed_except_user_review": row.get(
                "durable_memory_allowed_except_user_review", False
            ),
            "durable_memory_block_reasons": row.get("durable_memory_block_reasons", []),
            "durable_memory_block_reasons_except_user_review": row.get(
                "durable_memory_block_reasons_except_user_review", []
            ),
            "durable_memory_mutation_request_emitted": False,
            "reason": "durable memory is blocked until explicit user visual acceptance",
            "metrics_are_diagnostic_only": True,
        }
        for row in lifecycle_rows
    ]
    transaction_csv = output_root / "phase12_transaction_boundary_rows.csv"
    write_csv(transaction_csv, transaction_rows)
    casebook_manifest, casebook_markdown = write_phase12_casebook(
        output_root=output_root,
        scene_id=str(meta["scene_id"]),
        preset=str(args.preset),
        g3_summary=g3_summary,
        visual_rows=visual_rows,
        lifecycle_rows=lifecycle_rows,
        transaction_rows=transaction_rows,
        visual_csv=visual_csv,
        visual_json=visual_json,
        lifecycle_csv=lifecycle_csv,
        transaction_csv=transaction_csv,
    )
    full_scene_video = collect_visual_video_asset(g3_summary)

    summary = {
        "schema_version": "stream4d_v108_phase12_full_online_summary_v1",
        "status": "PHASE12_REPAIRED_ONLINE_REQUIRES_VISUAL_REVIEW",
        "preset": str(args.preset),
        "phase14_ablation": str(args.phase14_ablation or ""),
        "phase14_ablation_args": list(meta.get("phase14_ablation_args", [])),
        "phase14_cross_run_cache_read_count_observed": 0 if (not bool(args.skip_v107_run) and not bool(args.existing_v107_root)) else "",
        "scene_id": str(meta["scene_id"]),
        "runtime_sec": float(time.time() - started),
        "full_online_run_executed": bool(full_online_run_executed),
        "skip_v107_run": bool(args.skip_v107_run),
        "existing_v107_root_reused": bool(args.skip_v107_run),
        "v107_exit_code": int(run_exit_code),
        "v107_command": command,
        "v107_command_file": rel(output_root / "phase12_v107_command.txt"),
        "v107_root": rel(v107_root),
        "v107_summary": rel(g3_summary_path),
        "v107_summary_sha256": sha256_file(g3_summary_path),
        "rolling_summary": rel(rolling_summary_path),
        "rolling_summary_sha256": sha256_file(rolling_summary_path),
        "records_csv": rel(records_csv),
        "records_csv_sha256": sha256_file(records_csv),
        "phase12_visual_review_index_csv": rel(visual_csv),
        "phase12_visual_review_index_csv_sha256": sha256_file(visual_csv),
        "phase12_visual_review_index_json": rel(visual_json),
        "phase12_visual_review_index_json_sha256": sha256_file(visual_json),
        "phase12_lifecycle_admission_rows_csv": rel(lifecycle_csv),
        "phase12_lifecycle_admission_rows_csv_sha256": sha256_file(lifecycle_csv),
        "phase12_transaction_boundary_rows_csv": rel(transaction_csv),
        "phase12_transaction_boundary_rows_csv_sha256": sha256_file(transaction_csv),
        "phase12_casebook_manifest": rel(casebook_manifest),
        "phase12_casebook_manifest_sha256": sha256_file(casebook_manifest),
        "phase12_casebook_markdown": rel(casebook_markdown),
        "phase12_casebook_markdown_sha256": sha256_file(casebook_markdown),
        "full_scene_video": full_scene_video,
        "full_scene_video_exists": bool(full_scene_video["exists"]),
        "visual_review_image_count": int(len(visual_rows)),
        "visual_review_images": [
            {"path": row["visual_path"], "sha256": row["visual_sha256"]} for row in visual_rows
        ],
        "durable_memory_allowed_count": int(sum(parse_bool(row.get("durable_memory_allowed")) for row in lifecycle_rows)),
        "durable_memory_allowed_except_user_review_count": int(
            sum(parse_bool(row.get("durable_memory_allowed_except_user_review")) for row in lifecycle_rows)
        ),
        "durable_memory_block_reason_histogram": {
            reason: sum(1 for row in lifecycle_rows for reason2 in row.get("durable_memory_block_reasons", []) if reason2 == reason)
            for reason in sorted({reason for row in lifecycle_rows for reason in row.get("durable_memory_block_reasons", [])})
        },
        "durable_memory_block_reason_except_user_review_histogram": {
            reason: sum(
                1
                for row in lifecycle_rows
                for reason2 in row.get("durable_memory_block_reasons_except_user_review", [])
                if reason2 == reason
            )
            for reason in sorted(
                {
                    reason
                    for row in lifecycle_rows
                    for reason in row.get("durable_memory_block_reasons_except_user_review", [])
                }
            )
        },
        "durable_memory_mutation_request_count": 0,
        "output_transaction_count": int(len(transaction_rows)),
        "metrics_are_diagnostic_only": True,
        "quality_decision_rule": "Only high-resolution visual confirmation can decide good or bad.",
        "visual_confirmation_required": True,
        "durable_memory_rule": "No durable SAM2 memory admission without explicit user visual acceptance.",
        "uses_lingbot_geometry": bool(
            str(g3_summary.get("reactivation_prompt_mode", "")) in {"lingbot_geometry", "appearance_geometry_filter"}
        ),
        "uses_scannet_pose_or_depth_for_projection": False,
        "phase14_control_fields": {
            "reactivation_prompt_mode": g3_summary.get("reactivation_prompt_mode", ""),
            "output_plane_enabled": g3_summary.get("output_plane_enabled", ""),
            "disable_output_plane": g3_summary.get("disable_output_plane", ""),
            "disable_gap_birth": g3_summary.get("disable_gap_birth", ""),
            "random_geometry_prompts_enabled": g3_summary.get("random_geometry_prompts_enabled", ""),
            "shadow_output_mode": g3_summary.get("shadow_output_mode", ""),
            "birth_admission_appearance_enabled": g3_summary.get("birth_admission_appearance_enabled", ""),
            "disable_birth_admission_appearance": g3_summary.get("disable_birth_admission_appearance", ""),
            "gap_max_points": g3_summary.get("gap_max_points", ""),
            "gap_max_points_per_component": g3_summary.get("gap_max_points_per_component", ""),
            "prompt_target_stability_max_depth_abs_error": g3_summary.get(
                "prompt_target_stability_max_depth_abs_error", ""
            ),
            "prompt_target_stability_min_depth_conf": g3_summary.get("prompt_target_stability_min_depth_conf", ""),
            "prompt_source_core_supplement_min_depth_conf": g3_summary.get(
                "prompt_source_core_supplement_min_depth_conf", ""
            ),
            "physical_anchor_readiness_gate_enabled": g3_summary.get("physical_anchor_readiness_gate_enabled", ""),
            "physical_anchor_ready_source_mapping_count": g3_summary.get(
                "physical_anchor_ready_source_mapping_count", ""
            ),
            "physical_anchor_readiness_rejects_random_geometry": g3_summary.get(
                "physical_anchor_readiness_rejects_random_geometry", ""
            ),
        },
        "prompt_policy": {
            "positive_prompts": "historical source mask interior/core points projected with LingBot Map geometry",
            "negative_prompts": "nearby co-visible non-target source mask core points projected with LingBot Map geometry",
            "occlusion_rule": "v107 prompt projection keeps visible/depth-consistent points only",
            "edge_rule": "source-core and target-mask-core distance filters avoid object boundaries",
            "outlier_rule": "anchor conflict filter removes positive points near negatives or outside positive consensus",
        },
        "diagnostic_v107_counts": {
            "event_count": g3_summary.get("event_count"),
            "probation_output_mask_count": g3_summary.get("probation_output_mask_count"),
            "shadow_output_mask_count": g3_summary.get("shadow_output_mask_count"),
            "actual_video_readd_record_count": g3_summary.get("actual_video_readd_record_count"),
            "long_term_memory_admitted_count_in_v107_scheduler": g3_summary.get("long_term_memory_admitted_count"),
        },
        "diagnostic_note": (
            "v107 scheduler counts and reference metrics are logged for debugging only; "
            "Phase12 v108 durable memory remains blocked pending visual review."
        ),
    }
    summary_path = output_root / "phase12_full_online_summary.json"
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": rel(summary_path),
                "status": summary["status"],
                "visual_review_image_count": int(len(visual_rows)),
                "durable_memory_mutation_request_count": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
