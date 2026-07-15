#!/usr/bin/env python3
"""Build a small high-resolution visual sanity set for v108 Phase2 events."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_v105_4dpm_style_per_frame_segmentors import PALETTE, read_rgb, sha256_file  # noqa: E402
from tools.run_v106_stateful_sam2_scene_stream import _resolve  # noqa: E402


PALETTE_ARRAY = np.asarray(PALETTE, dtype=np.uint8)
PALETTE_LUT = np.zeros((65536, 3), dtype=np.uint8)
PALETTE_LUT[1:] = PALETTE_ARRAY[(np.arange(1, 65536, dtype=np.int64) - 1) % int(len(PALETTE_ARRAY))]
EDGE_KERNEL = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class EventVisualCase:
    case_name: str
    event_type: str
    event_summary: Path
    source_summary: Path
    frame_id: int
    previous_frame_id: int | None
    note: str


def draw_title(image: np.ndarray, title: str) -> np.ndarray:
    pad = 44
    out = np.full((image.shape[0] + pad, image.shape[1], 3), 255, dtype=np.uint8)
    out[pad:] = image
    cv2.putText(out, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (15, 15, 15), 2, cv2.LINE_AA)
    return out


def read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.uint16, copy=False)


def overlay(rgb: np.ndarray, label: np.ndarray, alpha: float = 0.52) -> np.ndarray:
    if label.shape[:2] != rgb.shape[:2]:
        label = cv2.resize(label, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    fg = label > 0
    out = rgb.copy()
    if bool(np.any(fg)):
        color = PALETTE_LUT[label]
        blended = cv2.addWeighted(rgb, 1.0 - alpha, color, alpha, 0)
        out[fg] = blended[fg]
        edge = cv2.morphologyEx(label, cv2.MORPH_GRADIENT, EDGE_KERNEL)
        out[(edge > 0) & fg] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def frame_record(summary: dict, frame_id: int) -> dict:
    for row in summary.get("records", []):
        if int(row.get("frame_id", -1)) == int(frame_id):
            return row
    raise KeyError(f"frame {frame_id} not found")


def load_rgb_and_overlay(summary: dict, frame_id: int) -> tuple[np.ndarray, np.ndarray, dict]:
    record = frame_record(summary, frame_id)
    label = read_label(_resolve(record["label_path"]))
    rgb_root = _resolve(summary.get("rgb_root", "Stream3D/data/scannet/processed"))
    rgb_path = rgb_root / str(summary["scene_id"]) / "color" / f"{frame_id}.jpg"
    rgb = read_rgb(rgb_path)
    return rgb, overlay(rgb, label), record


def build_case(case: EventVisualCase, output_root: Path) -> dict:
    source_summary = json.loads(case.source_summary.read_text())
    event_summary = json.loads(case.event_summary.read_text())
    event_counts = event_summary.get("event_type_counts", {})

    panels = []
    records = []
    if case.previous_frame_id is not None:
        prev_rgb, prev_overlay, prev_record = load_rgb_and_overlay(source_summary, case.previous_frame_id)
        panels.append(draw_title(prev_rgb, f"prev RGB frame {case.previous_frame_id}"))
        panels.append(draw_title(prev_overlay, f"prev overlay frame {case.previous_frame_id}"))
        records.append(prev_record)
    rgb, cur_overlay, cur_record = load_rgb_and_overlay(source_summary, case.frame_id)
    panels.append(draw_title(rgb, f"current RGB frame {case.frame_id}"))
    panels.append(draw_title(cur_overlay, f"current overlay frame {case.frame_id}"))
    records.append(cur_record)

    panel = np.concatenate(panels, axis=1)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{case.case_name}_{case.event_type}_frame_{case.frame_id:06d}.png"
    cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return {
        "case_name": case.case_name,
        "event_type": case.event_type,
        "frame_id": int(case.frame_id),
        "previous_frame_id": case.previous_frame_id,
        "note": case.note,
        "event_type_count": event_counts.get(case.event_type),
        "visual_path": path.as_posix(),
        "visual_sha256": sha256_file(path),
        "frame_records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    cases = [
        EventVisualCase(
            case_name="scene0050_90f",
            event_type="new_gap_and_transaction",
            event_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase2_online_events_scene0050_90f_20260714_1508/online_event_summary.json",
            source_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0050_90f_20260714_1452/v106_stateful_sam2_rolling_scene_stream/summary.json",
            frame_id=4300,
            previous_frame_id=4295,
            note="top new_gap_hypothesis and transaction_suggestion casebook row",
        ),
        EventVisualCase(
            case_name="scene0050_90f",
            event_type="active_growth_alert",
            event_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase2_online_events_scene0050_90f_20260714_1508/online_event_summary.json",
            source_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0050_90f_20260714_1452/v106_stateful_sam2_rolling_scene_stream/summary.json",
            frame_id=4500,
            previous_frame_id=4495,
            note="top active_growth_alert casebook row",
        ),
        EventVisualCase(
            case_name="scene0011_30f",
            event_type="dormant_visibility_suspicion",
            event_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase2_online_events_scene0011_30f_20260714_1508/online_event_summary.json",
            source_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0011_30f_20260714_1454/v106_stateful_sam2_rolling_scene_stream/summary.json",
            frame_id=10,
            previous_frame_id=5,
            note="top dormant_visibility_suspicion casebook row",
        ),
    ]
    output_root = REPO_ROOT / args.output_root
    records = [build_case(case, output_root) for case in cases]
    manifest = {
        "schema_version": "stream4d_v108_phase2_event_visual_review_manifest_v1",
        "review_standard": "visual_sanity_review; event_metrics_are_diagnostic_only",
        "records": records,
    }
    manifest_path = output_root / "phase2_event_visual_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
