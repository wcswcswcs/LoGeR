#!/usr/bin/env python3
"""Build a small high-resolution visual review set for v108 Phase1 parity."""

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
class ReviewCase:
    case_name: str
    frame_id: int
    candidate_summary: Path
    reference_summary: Path
    candidate_label_dir: Path
    reference_label_dir: Path


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


def diff_panel(rgb: np.ndarray, candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    cand_fg = candidate > 0
    ref_fg = reference > 0
    panel = rgb.copy()
    common = cand_fg & ref_fg
    cand_only = cand_fg & ~ref_fg
    ref_only = ref_fg & ~cand_fg
    panel[common] = (
        panel[common].astype(np.float32) * 0.45
        + np.array([180, 180, 180], dtype=np.float32) * 0.55
    ).astype(np.uint8)
    panel[cand_only] = np.array([255, 40, 40], dtype=np.uint8)
    panel[ref_only] = np.array([40, 180, 255], dtype=np.uint8)
    return panel


def draw_title(image: np.ndarray, title: str) -> np.ndarray:
    pad = 44
    out = np.full((image.shape[0] + pad, image.shape[1], 3), 255, dtype=np.uint8)
    out[pad:] = image
    cv2.putText(out, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 2, cv2.LINE_AA)
    return out


def frame_record(summary: dict, frame_id: int) -> dict:
    for row in summary.get("records", []):
        if int(row.get("frame_id", -1)) == int(frame_id):
            return row
    raise KeyError(f"frame {frame_id} not found in summary")


def build_case(case: ReviewCase, output_dir: Path) -> dict:
    candidate_summary = json.loads(case.candidate_summary.read_text())
    reference_summary = json.loads(case.reference_summary.read_text())
    cand_row = frame_record(candidate_summary, case.frame_id)
    ref_row = frame_record(reference_summary, case.frame_id)
    rgb_path = _resolve(cand_row.get("rgb_path", ref_row.get("rgb_path", ""))) if cand_row.get("rgb_path") or ref_row.get("rgb_path") else None
    if rgb_path is None:
        rgb_root = _resolve(candidate_summary.get("rgb_root", "Stream3D/data/scannet/processed"))
        rgb_path = rgb_root / str(candidate_summary["scene_id"]) / "color" / f"{case.frame_id}.jpg"
    rgb = read_rgb(rgb_path)
    candidate = read_label(case.candidate_label_dir / f"frame_{case.frame_id:06d}.png")
    reference = read_label(case.reference_label_dir / f"frame_{case.frame_id:06d}.png")
    if candidate.shape[:2] != rgb.shape[:2]:
        candidate = cv2.resize(candidate, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    if reference.shape[:2] != rgb.shape[:2]:
        reference = cv2.resize(reference, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    panels = [
        draw_title(rgb, f"{case.case_name} frame {case.frame_id}: RGB"),
        draw_title(overlay(rgb, reference), "reference overlay"),
        draw_title(overlay(rgb, candidate), "candidate overlay"),
        draw_title(diff_panel(rgb, candidate, reference), "diff: red=candidate only, cyan=reference only"),
    ]
    full = np.concatenate(panels, axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / f"{case.case_name}_frame_{case.frame_id:06d}_visual_compare.png"
    cv2.imwrite(str(full_path), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))

    cand_fg = candidate > 0
    ref_fg = reference > 0
    mismatch = (candidate != reference) | (cand_fg != ref_fg)
    zoom_path = ""
    component_sheet_path = ""
    mismatch_bbox = None
    mismatch_component_count = 0
    if bool(np.any(mismatch)):
        ys, xs = np.where(mismatch)
        margin = 80
        y0 = max(0, int(ys.min()) - margin)
        y1 = min(rgb.shape[0], int(ys.max()) + margin + 1)
        x0 = max(0, int(xs.min()) - margin)
        x1 = min(rgb.shape[1], int(xs.max()) + margin + 1)
        crop_panels = [
            draw_title(rgb[y0:y1, x0:x1], "RGB crop"),
            draw_title(overlay(rgb, reference)[y0:y1, x0:x1], "reference crop"),
            draw_title(overlay(rgb, candidate)[y0:y1, x0:x1], "candidate crop"),
            draw_title(diff_panel(rgb, candidate, reference)[y0:y1, x0:x1], "diff crop"),
        ]
        zoom = np.concatenate(crop_panels, axis=1)
        zoom = cv2.resize(zoom, (zoom.shape[1] * 3, zoom.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
        zoom_path_obj = output_dir / f"{case.case_name}_frame_{case.frame_id:06d}_mismatch_zoom3x.png"
        cv2.imwrite(str(zoom_path_obj), cv2.cvtColor(zoom, cv2.COLOR_RGB2BGR))
        zoom_path = zoom_path_obj.as_posix()
        mismatch_bbox = [int(x0), int(y0), int(x1), int(y1)]

        component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
            mismatch.astype(np.uint8), connectivity=8
        )
        crops = []
        for component_id in range(1, component_count):
            x, y, w, h, area = stats[component_id]
            margin = 24
            cx0 = max(0, int(x) - margin)
            cy0 = max(0, int(y) - margin)
            cx1 = min(rgb.shape[1], int(x + w) + margin)
            cy1 = min(rgb.shape[0], int(y + h) + margin)
            component_mask = component_labels[cy0:cy1, cx0:cx1] == component_id
            crop = diff_panel(rgb, candidate, reference)[cy0:cy1, cx0:cx1].copy()
            crop[component_mask] = np.array([255, 0, 0], dtype=np.uint8)
            crop = draw_title(crop, f"component {component_id} area {int(area)} px")
            crop = cv2.resize(crop, (crop.shape[1] * 16, crop.shape[0] * 16), interpolation=cv2.INTER_NEAREST)
            crops.append(crop)
        mismatch_component_count = len(crops)
        if crops:
            max_h = max(crop.shape[0] for crop in crops)
            padded = []
            for crop in crops:
                if crop.shape[0] < max_h:
                    pad = np.full((max_h - crop.shape[0], crop.shape[1], 3), 255, dtype=np.uint8)
                    crop = np.concatenate([crop, pad], axis=0)
                padded.append(crop)
            sheet = np.concatenate(padded, axis=1)
            component_sheet = output_dir / f"{case.case_name}_frame_{case.frame_id:06d}_mismatch_components_16x.png"
            cv2.imwrite(str(component_sheet), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
            component_sheet_path = component_sheet.as_posix()

    intersection = int(np.logical_and(cand_fg, ref_fg).sum())
    union = int(np.logical_or(cand_fg, ref_fg).sum())
    return {
        "case_name": case.case_name,
        "frame_id": int(case.frame_id),
        "rgb_path": str(rgb_path),
        "candidate_label_path": str(case.candidate_label_dir / f"frame_{case.frame_id:06d}.png"),
        "reference_label_path": str(case.reference_label_dir / f"frame_{case.frame_id:06d}.png"),
        "visual_compare_path": full_path.as_posix(),
        "visual_compare_sha256": sha256_file(full_path),
        "mismatch_zoom_path": zoom_path,
        "mismatch_zoom_sha256": sha256_file(Path(zoom_path)) if zoom_path else "",
        "mismatch_component_count": int(mismatch_component_count),
        "mismatch_component_sheet_path": component_sheet_path,
        "mismatch_component_sheet_sha256": sha256_file(Path(component_sheet_path)) if component_sheet_path else "",
        "mismatch_bbox_xyxy": mismatch_bbox,
        "candidate_fg_area_px": int(cand_fg.sum()),
        "reference_fg_area_px": int(ref_fg.sum()),
        "intersection_px": intersection,
        "union_px": union,
        "foreground_iou": float(intersection / union) if union else 1.0,
        "pixel_label_equal": bool(np.array_equal(candidate, reference)),
        "foreground_equal": bool(np.array_equal(cand_fg, ref_fg)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    cases = [
        ReviewCase(
            case_name="scene0050_30f",
            frame_id=4305,
            candidate_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0050_30f_20260714_1452/v106_stateful_sam2_rolling_scene_stream/summary.json",
            reference_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_reference_scene0050_30f_labelonly_20260714_1442/v106_stateful_sam2_rolling_scene_stream/summary.json",
            candidate_label_dir=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0050_30f_20260714_1452/v106_stateful_sam2_rolling_scene_stream/labels",
            reference_label_dir=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_reference_scene0050_30f_labelonly_20260714_1442/v106_stateful_sam2_rolling_scene_stream/labels",
        ),
        ReviewCase(
            case_name="scene0050_90f_frame4500",
            frame_id=4500,
            candidate_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0050_90f_20260714_1452/v106_stateful_sam2_rolling_scene_stream/summary.json",
            reference_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_reference_scene0050_90f_labelonly_20260714_1442/v106_stateful_sam2_rolling_scene_stream/summary.json",
            candidate_label_dir=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0050_90f_20260714_1452/v106_stateful_sam2_rolling_scene_stream/labels",
            reference_label_dir=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_reference_scene0050_90f_labelonly_20260714_1442/v106_stateful_sam2_rolling_scene_stream/labels",
        ),
        ReviewCase(
            case_name="scene0050_90f_tail",
            frame_id=4605,
            candidate_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0050_90f_20260714_1452/v106_stateful_sam2_rolling_scene_stream/summary.json",
            reference_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_reference_scene0050_90f_labelonly_20260714_1442/v106_stateful_sam2_rolling_scene_stream/summary.json",
            candidate_label_dir=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0050_90f_20260714_1452/v106_stateful_sam2_rolling_scene_stream/labels",
            reference_label_dir=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_reference_scene0050_90f_labelonly_20260714_1442/v106_stateful_sam2_rolling_scene_stream/labels",
        ),
        ReviewCase(
            case_name="scene0011_30f_mismatch",
            frame_id=0,
            candidate_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0011_30f_20260714_1454/v106_stateful_sam2_rolling_scene_stream/summary.json",
            reference_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_reference_scene0011_30f_labelonly_20260714_1442/v106_stateful_sam2_rolling_scene_stream/summary.json",
            candidate_label_dir=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0011_30f_20260714_1454/v106_stateful_sam2_rolling_scene_stream/labels",
            reference_label_dir=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_reference_scene0011_30f_labelonly_20260714_1442/v106_stateful_sam2_rolling_scene_stream/labels",
        ),
        ReviewCase(
            case_name="scene0011_30f_tail",
            frame_id=145,
            candidate_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0011_30f_20260714_1454/v106_stateful_sam2_rolling_scene_stream/summary.json",
            reference_summary=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_reference_scene0011_30f_labelonly_20260714_1442/v106_stateful_sam2_rolling_scene_stream/summary.json",
            candidate_label_dir=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_candidate_scene0011_30f_20260714_1454/v106_stateful_sam2_rolling_scene_stream/labels",
            reference_label_dir=REPO_ROOT / "Stream3D/outputs/audit/v108_phase1_reference_scene0011_30f_labelonly_20260714_1442/v106_stateful_sam2_rolling_scene_stream/labels",
        ),
    ]
    output_root = REPO_ROOT / args.output_root
    records = [build_case(case, output_root) for case in cases]
    manifest_path = output_root / "visual_review_manifest.json"
    manifest = {
        "schema_version": "stream4d_v108_phase1_visual_review_manifest_v1",
        "review_standard": "visual_confirmation_required; metrics_are_diagnostic_only",
        "case_count": len(records),
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
