#!/usr/bin/env python3
"""Focused scene0050 couch/sofa audit for v105 Phase 1.

This helper is intentionally diagnostic-only. It reads the existing ScanNet
GT instance/semantic PNGs and the already-generated baseline-x label PNGs, then
writes pixel coverage records plus RGB/GT/prediction comparison sheets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_ID = "scene0050_00"
FRAME_IDS = list(range(0, 160, 5))
COUCH_INSTANCE_PNG_ID = 14
COUCH_RAW_SEMANTIC_ID = 6
KEY_FRAMES = [35, 40, 55, 80, 105, 130, 150, 155]

BASELINES = {
    "X0_baseline_x_sam2_twostage": REPO_ROOT
    / "Stream3D/outputs/audit/v105_specgap_phase1_x0_scene0050_20260711/baseline_x_sam2_twostage_sam2/labels",
    "X1_baseline_x_gapadaptive": REPO_ROOT
    / "Stream3D/outputs/audit/v105_specgap_phase1_x1_scene0050_20260711/baseline_x_gapadaptive_sam2/labels",
}

COLOR_GT = np.array([0, 255, 80], dtype=np.uint8)
COLOR_X0 = np.array([255, 0, 255], dtype=np.uint8)
COLOR_X1 = np.array([0, 210, 255], dtype=np.uint8)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_label(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    label = np.asarray(image, dtype=np.int64)
    if shape_hw is not None and label.shape[:2] != shape_hw:
        h, w = shape_hw
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    return label


def _alpha_overlay(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    out = rgb.copy()
    if np.any(mask):
        out[mask] = (out[mask].astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
    return out


def _draw_contour(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 3) -> np.ndarray:
    out = rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.drawContours(out_bgr, contours, -1, bgr, thickness)
        out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out


def _fit_panel(image: np.ndarray, width: int) -> Image.Image:
    pil = Image.fromarray(image)
    scale = float(width) / float(pil.width)
    height = max(1, int(round(float(pil.height) * scale)))
    return pil.resize((width, height), Image.Resampling.BILINEAR)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


def _caption_panel(image: np.ndarray, title: str, lines: list[str], width: int) -> Image.Image:
    panel = _fit_panel(image, width)
    pad_h = 76
    out = Image.new("RGB", (panel.width, panel.height + pad_h), (18, 20, 24))
    out.paste(panel, (0, pad_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 6), title, fill=(245, 245, 245), font=_font(17, True))
    y = 30
    for line in lines[:3]:
        draw.text((8, y), line, fill=(220, 224, 232), font=_font(12, False))
        y += 15
    return out


def _paste_grid(panels: list[Image.Image], cols: int, gap: int = 8) -> Image.Image:
    if not panels:
        raise ValueError("no panels")
    rows = int(np.ceil(len(panels) / float(cols)))
    cell_w = max(p.width for p in panels)
    cell_h = max(p.height for p in panels)
    out = Image.new("RGB", (cols * cell_w + (cols - 1) * gap, rows * cell_h + (rows - 1) * gap), (12, 13, 16))
    for idx, panel in enumerate(panels):
        r, c = divmod(idx, cols)
        x = c * (cell_w + gap)
        y = r * (cell_h + gap)
        out.paste(panel, (x, y))
    return out


def _prediction_record(pred: np.ndarray, gt_mask: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    gt_pixels = int(np.count_nonzero(gt_mask))
    positive = pred > 0
    ids = [int(v) for v in np.unique(pred[gt_mask]) if int(v) > 0]
    top_ids: list[dict[str, Any]] = []
    best = {
        "pred_id": 0,
        "iou": 0.0,
        "intersection_pixels": 0,
        "pred_area_pixels": 0,
        "gt_recall": 0.0,
        "pred_precision": 0.0,
    }
    for pred_id in ids:
        mask = pred == int(pred_id)
        inter = int(np.count_nonzero(mask & gt_mask))
        union = int(np.count_nonzero(mask | gt_mask))
        pred_area = int(np.count_nonzero(mask))
        iou = float(inter) / float(union) if union else 0.0
        recall = float(inter) / float(gt_pixels) if gt_pixels else 0.0
        precision = float(inter) / float(pred_area) if pred_area else 0.0
        row = {
            "pred_id": int(pred_id),
            "intersection_pixels": int(inter),
            "pred_area_pixels": int(pred_area),
            "iou": float(iou),
            "gt_recall": float(recall),
            "pred_precision": float(precision),
        }
        top_ids.append(row)
        if iou > float(best["iou"]):
            best = row
    top_ids.sort(key=lambda row: (float(row["intersection_pixels"]), float(row["iou"])), reverse=True)
    covered = positive & gt_mask
    record = {
        "gt_pixels": int(gt_pixels),
        "positive_overlap_pixels": int(np.count_nonzero(covered)),
        "positive_overlap_recall": float(np.count_nonzero(covered)) / float(gt_pixels) if gt_pixels else 0.0,
        "overlapping_pred_id_count": int(len(ids)),
        "best_pred": best,
        "top_pred_ids_by_gt_intersection": top_ids[:8],
        "foreground_ratio": float(np.count_nonzero(positive)) / float(pred.size),
        "visible_pred_id_count": int(len([v for v in np.unique(pred) if int(v) > 0])),
    }
    best_mask = pred == int(best["pred_id"]) if int(best["pred_id"]) > 0 else np.zeros(pred.shape, dtype=bool)
    return record, covered, best_mask


def _panel_rgb_gt(rgb: np.ndarray, gt_mask: np.ndarray, sem_mask: np.ndarray) -> np.ndarray:
    out = _alpha_overlay(rgb, sem_mask, np.array([30, 120, 255], dtype=np.uint8), 0.28)
    out = _alpha_overlay(out, gt_mask, COLOR_GT, 0.38)
    out = _draw_contour(out, sem_mask, (30, 120, 255), 2)
    out = _draw_contour(out, gt_mask, (0, 255, 80), 3)
    return out


def _panel_pred(rgb: np.ndarray, gt_mask: np.ndarray, best_mask: np.ndarray, covered_mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    out = _alpha_overlay(rgb, gt_mask, COLOR_GT, 0.22)
    out = _alpha_overlay(out, best_mask, color, 0.48)
    miss = gt_mask & ~covered_mask
    out = _alpha_overlay(out, miss, np.array([255, 20, 20], dtype=np.uint8), 0.68)
    out = _draw_contour(out, gt_mask, (0, 255, 80), 3)
    out = _draw_contour(out, best_mask, tuple(int(v) for v in color.tolist()), 3)
    return out


def _panel_pred_opaque(rgb: np.ndarray, gt_mask: np.ndarray, best_mask: np.ndarray, covered_mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    out = _alpha_overlay(rgb, gt_mask, COLOR_GT, 0.15)
    out = _alpha_overlay(out, best_mask, color, 0.82)
    miss = gt_mask & ~covered_mask
    out = _alpha_overlay(out, miss, np.array([255, 0, 0], dtype=np.uint8), 0.90)
    out = _draw_contour(out, gt_mask, (0, 255, 80), 4)
    out = _draw_contour(out, best_mask, tuple(int(v) for v in color.tolist()), 4)
    return out


def _panel_gt_mask_only(gt_mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    out[gt_mask] = COLOR_GT
    out = _draw_contour(out, gt_mask, (255, 255, 255), 3)
    return out


def _panel_any_positive_mask_only(gt_mask: np.ndarray, covered_any: np.ndarray) -> np.ndarray:
    out = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    miss = gt_mask & ~covered_any
    out[gt_mask] = np.array([20, 95, 40], dtype=np.uint8)
    out[covered_any] = np.array([255, 255, 255], dtype=np.uint8)
    out[miss] = np.array([255, 0, 0], dtype=np.uint8)
    out = _draw_contour(out, gt_mask, (0, 255, 80), 3)
    return out


def _panel_best_mask_only(gt_mask: np.ndarray, best_mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    out = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    miss = gt_mask & ~best_mask
    overlap = gt_mask & best_mask
    extra = best_mask & ~gt_mask
    out[extra] = color
    out[overlap] = np.array([255, 255, 255], dtype=np.uint8)
    out[miss] = np.array([255, 0, 0], dtype=np.uint8)
    out = _draw_contour(out, gt_mask, (0, 255, 80), 3)
    out = _draw_contour(out, best_mask, tuple(int(v) for v in color.tolist()), 3)
    return out


def build_audit(output_root: Path, *, frame_ids: list[int], key_frames: list[int], panel_width: int) -> dict[str, Any]:
    scene_root = REPO_ROOT / "Stream3D/data/scannet/processed" / SCENE_ID
    color_dir = scene_root / "color"
    sem_dir = scene_root / "label-filt"
    inst_dir = scene_root / "instance/instance"

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    key_panels: list[Image.Image] = []
    late_panels: list[Image.Image] = []
    sem_inst_panels: list[Image.Image] = []
    opaque_key_panels: list[Image.Image] = []
    opaque_late_panels: list[Image.Image] = []
    mask_only_key_panels: list[Image.Image] = []
    mask_only_late_panels: list[Image.Image] = []

    for frame_id in frame_ids:
        rgb = _read_rgb(color_dir / f"{int(frame_id)}.jpg")
        shape_hw = rgb.shape[:2]
        sem = _read_label(sem_dir / f"{int(frame_id)}.png", shape_hw)
        inst = _read_label(inst_dir / f"{int(frame_id)}.png", shape_hw)
        gt_mask = inst == COUCH_INSTANCE_PNG_ID
        sem_mask = sem == COUCH_RAW_SEMANTIC_ID
        row: dict[str, Any] = {
            "scene_id": SCENE_ID,
            "frame_id": int(frame_id),
            "couch_instance_png_id": int(COUCH_INSTANCE_PNG_ID),
            "couch_raw_semantic_id": int(COUCH_RAW_SEMANTIC_ID),
            "gt_couch_pixels": int(np.count_nonzero(gt_mask)),
            "semantic_couch_pixels": int(np.count_nonzero(sem_mask)),
            "semantic_vs_instance_iou": float(np.count_nonzero(sem_mask & gt_mask))
            / float(np.count_nonzero(sem_mask | gt_mask))
            if np.count_nonzero(sem_mask | gt_mask)
            else 0.0,
            "predictions": {},
        }

        pred_visuals: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        for baseline_name, label_dir in BASELINES.items():
            pred = _read_label(label_dir / f"frame_{int(frame_id):06d}.png", shape_hw)
            pred_record, covered, best_mask = _prediction_record(pred, gt_mask)
            row["predictions"][baseline_name] = pred_record
            color = COLOR_X0 if baseline_name.startswith("X0") else COLOR_X1
            pred_visuals[baseline_name] = (_panel_pred(rgb, gt_mask, best_mask, covered, color), pred_record)

        records.append(row)

        if frame_id in key_frames or frame_id >= 80:
            gt_panel = _caption_panel(
                _panel_rgb_gt(rgb, gt_mask, sem_mask),
                f"frame {frame_id} RGB + GT",
                [
                    f"inst14 px={row['gt_couch_pixels']} sem6 px={row['semantic_couch_pixels']}",
                    f"sem/inst IoU={row['semantic_vs_instance_iou']:.3f}",
                    "green=GT instance, blue=semantic id6",
                ],
                panel_width,
            )
            x0_img, x0_rec = pred_visuals["X0_baseline_x_sam2_twostage"]
            x1_img, x1_rec = pred_visuals["X1_baseline_x_gapadaptive"]
            x0_best = x0_rec["best_pred"]
            x1_best = x1_rec["best_pred"]
            x0_pred = _read_label(BASELINES["X0_baseline_x_sam2_twostage"] / f"frame_{int(frame_id):06d}.png", shape_hw)
            x1_pred = _read_label(BASELINES["X1_baseline_x_gapadaptive"] / f"frame_{int(frame_id):06d}.png", shape_hw)
            _, x0_covered, x0_best_mask = _prediction_record(x0_pred, gt_mask)
            _, x1_covered, x1_best_mask = _prediction_record(x1_pred, gt_mask)
            x0_panel = _caption_panel(
                x0_img,
                f"frame {frame_id} X0",
                [
                    f"best={x0_best['pred_id']} IoU={x0_best['iou']:.3f} recall={x0_best['gt_recall']:.3f}",
                    f"pos_recall={x0_rec['positive_overlap_recall']:.3f} overlap_ids={x0_rec['overlapping_pred_id_count']}",
                    "magenta=best pred, red=GT miss",
                ],
                panel_width,
            )
            x0_opaque_panel = _caption_panel(
                _panel_pred_opaque(rgb, gt_mask, x0_best_mask, x0_covered, COLOR_X0),
                f"frame {frame_id} X0 opaque",
                [
                    f"best={x0_best['pred_id']} IoU={x0_best['iou']:.3f} recall={x0_best['gt_recall']:.3f}",
                    f"pos_recall={x0_rec['positive_overlap_recall']:.3f} overlap_ids={x0_rec['overlapping_pred_id_count']}",
                    "solid magenta=best pred, red=GT miss",
                ],
                panel_width,
            )
            x1_opaque_panel = _caption_panel(
                _panel_pred_opaque(rgb, gt_mask, x1_best_mask, x1_covered, COLOR_X1),
                f"frame {frame_id} X1 opaque",
                [
                    f"best={x1_best['pred_id']} IoU={x1_best['iou']:.3f} recall={x1_best['gt_recall']:.3f}",
                    f"pos_recall={x1_rec['positive_overlap_recall']:.3f} overlap_ids={x1_rec['overlapping_pred_id_count']}",
                    "solid cyan=best pred, red=GT miss",
                ],
                panel_width,
            )
            mask_gt_panel = _caption_panel(
                _panel_gt_mask_only(gt_mask),
                f"frame {frame_id} GT mask",
                [
                    f"GT couch px={row['gt_couch_pixels']}",
                    "green=GT couch instance",
                    "black=not GT couch",
                ],
                panel_width,
            )
            x0_any_panel = _caption_panel(
                _panel_any_positive_mask_only(gt_mask, x0_covered),
                f"frame {frame_id} X0 any-mask",
                [
                    f"any pos recall={x0_rec['positive_overlap_recall']:.3f}",
                    "white=GT px covered by any pred",
                    "red=GT px with no pred mask",
                ],
                panel_width,
            )
            x0_best_panel = _caption_panel(
                _panel_best_mask_only(gt_mask, x0_best_mask, COLOR_X0),
                f"frame {frame_id} X0 best-id",
                [
                    f"id={x0_best['pred_id']} IoU={x0_best['iou']:.3f} recall={x0_best['gt_recall']:.3f}",
                    "white=GT & best, magenta=best outside",
                    "red=GT not owned by best",
                ],
                panel_width,
            )
            x1_any_panel = _caption_panel(
                _panel_any_positive_mask_only(gt_mask, x1_covered),
                f"frame {frame_id} X1 any-mask",
                [
                    f"any pos recall={x1_rec['positive_overlap_recall']:.3f}",
                    "white=GT px covered by any pred",
                    "red=GT px with no pred mask",
                ],
                panel_width,
            )
            x1_best_panel = _caption_panel(
                _panel_best_mask_only(gt_mask, x1_best_mask, COLOR_X1),
                f"frame {frame_id} X1 best-id",
                [
                    f"id={x1_best['pred_id']} IoU={x1_best['iou']:.3f} recall={x1_best['gt_recall']:.3f}",
                    "white=GT & best, cyan=best outside",
                    "red=GT not owned by best",
                ],
                panel_width,
            )
            x1_panel = _caption_panel(
                x1_img,
                f"frame {frame_id} X1",
                [
                    f"best={x1_best['pred_id']} IoU={x1_best['iou']:.3f} recall={x1_best['gt_recall']:.3f}",
                    f"pos_recall={x1_rec['positive_overlap_recall']:.3f} overlap_ids={x1_rec['overlapping_pred_id_count']}",
                    "cyan=best pred, red=GT miss",
                ],
                panel_width,
            )
            row_panels = [gt_panel, x0_panel, x1_panel]
            opaque_row_panels = [gt_panel, x0_opaque_panel, x1_opaque_panel]
            mask_row_panels = [mask_gt_panel, x0_any_panel, x0_best_panel, x1_any_panel, x1_best_panel]
            if frame_id in key_frames:
                key_panels.extend(row_panels)
                opaque_key_panels.extend(opaque_row_panels)
                mask_only_key_panels.extend(mask_row_panels)
                sem_inst_panels.append(gt_panel)
            if frame_id >= 80:
                late_panels.extend(row_panels)
                opaque_late_panels.extend(opaque_row_panels)
                mask_only_late_panels.extend(mask_row_panels)

    summary = summarize(records)
    _write_json(output_root / "scene0050_couch_records.json", {"schema_version": "v105_phase1_couch_audit_v1", "rows": records})
    _write_json(output_root / "scene0050_couch_summary.json", summary)
    if key_panels:
        _paste_grid(key_panels, cols=3).save(output_root / "scene0050_couch_gt_x0_x1_keyframes.jpg", quality=94)
    if late_panels:
        _paste_grid(late_panels, cols=3).save(output_root / "scene0050_couch_gt_x0_x1_late_frames_80_155.jpg", quality=94)
    if sem_inst_panels:
        _paste_grid(sem_inst_panels, cols=4).save(output_root / "scene0050_couch_semantic_instance_keyframes.jpg", quality=94)
    if opaque_key_panels:
        _paste_grid(opaque_key_panels, cols=3).save(output_root / "scene0050_couch_high_opacity_overlay_keyframes.jpg", quality=94)
    if opaque_late_panels:
        _paste_grid(opaque_late_panels, cols=3).save(output_root / "scene0050_couch_high_opacity_overlay_late_frames_80_155.jpg", quality=94)
    if mask_only_key_panels:
        _paste_grid(mask_only_key_panels, cols=5).save(output_root / "scene0050_couch_mask_only_keyframes_5col.jpg", quality=94)
    if mask_only_late_panels:
        _paste_grid(mask_only_late_panels, cols=5).save(output_root / "scene0050_couch_mask_only_late_frames_80_155_5col.jpg", quality=94)
    return summary


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    frames_with_gt = [row for row in records if int(row["gt_couch_pixels"]) > 0]
    summary: dict[str, Any] = {
        "scene_id": SCENE_ID,
        "status": "diagnostic_generated",
        "frame_count": int(len(records)),
        "frames_with_gt_couch_pixels": int(len(frames_with_gt)),
        "couch_instance_png_id": int(COUCH_INSTANCE_PNG_ID),
        "couch_raw_semantic_id": int(COUCH_RAW_SEMANTIC_ID),
        "aggregation_evidence": {
            "source_file": "Stream3D/data/scannet/processed/scene0050_00/scene0050_00.aggregation.json",
            "label": "couch",
            "aggregation_object_id": 13,
            "instance_png_id_observed": 14,
            "semantic_label_file": "Stream3D/data/scannet/scannetv2-labels.combined.tsv",
            "raw_category_id_6": "couch",
            "nyu40class": "sofa",
        },
        "baseline_summaries": {},
        "notes": [
            "positive_overlap_recall only checks any positive predicted ID on GT couch pixels; it does not prove object-level identity.",
            "best_pred.iou and best_pred.gt_recall are stronger evidence for whether one predicted tracked ID owns the couch.",
            "red pixels in sheets are GT couch pixels not covered by any positive prediction.",
            "mask_only sheets separate any-positive coverage from best-ID object ownership to avoid alpha-blending ambiguity.",
        ],
    }
    for baseline_name in BASELINES:
        rows = [row["predictions"][baseline_name] for row in frames_with_gt]
        best_ids = [int(row["best_pred"]["pred_id"]) for row in rows]
        nonzero_best_ids = [pid for pid in best_ids if pid > 0]
        summary["baseline_summaries"][baseline_name] = {
            "mean_positive_overlap_recall": float(np.mean([float(row["positive_overlap_recall"]) for row in rows])) if rows else 0.0,
            "min_positive_overlap_recall": float(np.min([float(row["positive_overlap_recall"]) for row in rows])) if rows else 0.0,
            "mean_best_iou": float(np.mean([float(row["best_pred"]["iou"]) for row in rows])) if rows else 0.0,
            "min_best_iou": float(np.min([float(row["best_pred"]["iou"]) for row in rows])) if rows else 0.0,
            "mean_best_gt_recall": float(np.mean([float(row["best_pred"]["gt_recall"]) for row in rows])) if rows else 0.0,
            "min_best_gt_recall": float(np.min([float(row["best_pred"]["gt_recall"]) for row in rows])) if rows else 0.0,
            "distinct_best_pred_ids": sorted(set(nonzero_best_ids)),
            "distinct_best_pred_id_count": int(len(set(nonzero_best_ids))),
            "best_pred_id_sequence": best_ids,
            "mean_overlapping_pred_id_count": float(np.mean([float(row["overlapping_pred_id_count"]) for row in rows])) if rows else 0.0,
            "max_overlapping_pred_id_count": int(np.max([int(row["overlapping_pred_id_count"]) for row in rows])) if rows else 0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "Stream3D/outputs/audit/v105_specgap_phase1_sofa_blocker_20260711",
    )
    parser.add_argument("--panel-width", type=int, default=380)
    args = parser.parse_args()
    summary = build_audit(args.output_root, frame_ids=FRAME_IDS, key_frames=KEY_FRAMES, panel_width=int(args.panel_width))
    print(json.dumps({"output_root": str(args.output_root), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
