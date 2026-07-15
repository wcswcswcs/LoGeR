#!/usr/bin/env python3
"""Diagnose v108 scene video label churn and output coverage.

The report is intentionally diagnostic-only. It reads an existing rolling
summary plus label PNGs, joins them with frame diagnostics, and writes compact
CSV/JSON/markdown evidence for visual review windows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    ROOT
    / "Stream3D/outputs/audit/v108_phase13_scene0011_fullsample475_real_20260714_2255/"
    "v107_online_runner/v107_phase8_g3_rolling_scheduler_smoke/summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "Stream3D/outputs/audit/v108_phase34_video_sequential_visual_audit_scene0011_475f_20260715_0600"
)

PALETTE = np.asarray(
    [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
        (0, 128, 128),
        (220, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 215, 180),
        (0, 0, 128),
        (128, 128, 128),
    ],
    dtype=np.uint8,
)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: str | Path) -> str:
    p = resolve(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.uint16, copy=False)


def read_rgb(path: Path, shape: tuple[int, int]) -> np.ndarray:
    rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb is None:
        return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != shape:
        rgb = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return rgb


def label_ids(label: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(label) if int(v) > 0]


def id_area_map(label: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(label, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids.tolist(), counts.tolist(), strict=False) if int(i) > 0}


def top_objects(area_by_id: dict[int, int], image_area: int, limit: int = 12) -> list[dict[str, Any]]:
    rows = [
        {"label_id": int(obj_id), "object_id": int(obj_id) - 1, "area_px": int(area), "area_frac": float(area / image_area)}
        for obj_id, area in area_by_id.items()
    ]
    rows.sort(key=lambda r: int(r["area_px"]), reverse=True)
    return rows[:limit]


def bbox_for_id(label: np.ndarray, obj_id: int) -> list[int]:
    ys, xs = np.where(label == int(obj_id))
    if xs.size == 0:
        return []
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def best_overlap(prev: np.ndarray, curr: np.ndarray, prev_id: int, curr_ids: list[int]) -> dict[str, Any]:
    mask_a = prev == int(prev_id)
    area_a = int(np.count_nonzero(mask_a))
    if area_a <= 0:
        return {"best_label_id": 0, "best_iou": 0.0, "best_intersection_px": 0, "same_iou": 0.0}
    best_id = 0
    best_iou = 0.0
    best_inter = 0
    same_iou = 0.0
    for cid in curr_ids:
        mask_b = curr == int(cid)
        inter = int(np.count_nonzero(mask_a & mask_b))
        if inter <= 0:
            continue
        union = int(area_a + np.count_nonzero(mask_b) - inter)
        iou = float(inter / max(1, union))
        if int(cid) == int(prev_id):
            same_iou = iou
        if iou > best_iou:
            best_id = int(cid)
            best_iou = iou
            best_inter = inter
    return {
        "best_label_id": int(best_id),
        "best_object_id": int(best_id) - 1 if best_id > 0 else -1,
        "best_iou": float(best_iou),
        "best_intersection_px": int(best_inter),
        "same_iou": float(same_iou),
    }


def diag_by_index(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in summary.get("frame_diagnostics", []):
        idx = int(row.get("chunk_frame_index", -1))
        if idx >= 0:
            out[idx] = row
    return out


def record_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(summary.get("records", [])):
        copied = dict(row)
        copied.setdefault("chunk_frame_index", idx)
        rows.append(copied)
    return rows


def row_diag_features(diag: dict[str, Any] | None) -> dict[str, Any]:
    if not diag:
        return {
            "propagated_pre_disjoin_count": "",
            "propagated_post_disjoin_count": "",
            "pre_to_post_drop_count": "",
            "gap_mask_count": "",
            "gap_before_filter_count": "",
            "gap_dropped_by_filter_count": "",
            "gap_dropped_by_bbox_count": "",
            "final_frame_mask_count": "",
            "uncovered_ratio_before_gap": "",
            "gap_reuse_event_count": "",
            "output_relabel_event_count": "",
            "oversized_event_count": "",
            "stream_pruned_object_count": "",
        }
    gap_stats = diag.get("gap_stats", {}) if isinstance(diag.get("gap_stats", {}), dict) else {}
    filt = gap_stats.get("gap_output_shape_filter", {}) if isinstance(gap_stats.get("gap_output_shape_filter", {}), dict) else {}
    pre = int(diag.get("propagated_pre_disjoin_count", 0))
    post = int(diag.get("propagated_post_disjoin_count", 0))
    return {
        "propagated_pre_disjoin_count": pre,
        "propagated_post_disjoin_count": post,
        "pre_to_post_drop_count": int(pre - post),
        "gap_mask_count": int(diag.get("gap_mask_count", 0)),
        "gap_before_filter_count": int(gap_stats.get("post_disjoint_mask_count_before_gap_output_filter", diag.get("gap_mask_count", 0))),
        "gap_dropped_by_filter_count": int(filt.get("dropped_mask_count", 0)),
        "gap_dropped_by_bbox_count": len(filt.get("dropped_by_bbox_frac_indices", []) or []),
        "final_frame_mask_count": int(diag.get("final_frame_mask_count", 0)),
        "uncovered_ratio_before_gap": float(diag.get("uncovered_ratio_before_gap", 0.0)),
        "gap_reuse_event_count": int(diag.get("gap_reuse_event_count", 0)),
        "output_relabel_event_count": int(diag.get("output_relabel_event_count", 0)),
        "oversized_event_count": len(diag.get("stream_oversized_prune_events", []) or []),
        "stream_pruned_object_count": len(diag.get("stream_pruned_object_ids", []) or []),
    }


def overlay_label(rgb: np.ndarray, label: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    out = rgb.copy()
    ids = label_ids(label)
    color = np.zeros_like(out)
    for obj_id in ids:
        color[label == obj_id] = PALETTE[(int(obj_id) - 1) % len(PALETTE)]
    mask = label > 0
    out[mask] = (rgb[mask].astype(np.float32) * (1.0 - alpha) + color[mask].astype(np.float32) * alpha).astype(np.uint8)
    edge = np.zeros(label.shape[:2], dtype=np.uint8)
    for obj_id in ids:
        m = (label == obj_id).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edge, contours, -1, 255, 1)
    out[edge > 0] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def annotate_panel(image: np.ndarray, lines: list[str], width: int) -> np.ndarray:
    if image.shape[1] != width:
        ratio = float(width) / float(max(image.shape[1], 1))
        image = cv2.resize(image, (width, int(round(image.shape[0] * ratio))), interpolation=cv2.INTER_AREA)
    pad = 68
    panel = np.full((image.shape[0] + pad, image.shape[1], 3), 245, dtype=np.uint8)
    panel[pad:] = image
    y = 22
    for idx, line in enumerate(lines[:3]):
        size = 0.52 if idx else 0.62
        thick = 1 if idx else 2
        cv2.putText(panel, line[:110], (8, y), cv2.FONT_HERSHEY_SIMPLEX, size, (20, 20, 20), thick, cv2.LINE_AA)
        y += 21
    return panel


def make_diagnostic_sheet(
    out_path: Path,
    records: list[dict[str, Any]],
    frame_stats: dict[int, dict[str, Any]],
    start_index: int,
    *,
    cell_width: int,
) -> None:
    panels: list[np.ndarray] = []
    end = min(len(records), int(start_index) + 4)
    for idx in range(int(start_index), end):
        rec = records[idx]
        label = read_label(resolve(rec["label_path"]))
        rgb_path = resolve(str(rec.get("rgb_path", "")))
        rgb = read_rgb(rgb_path, label.shape[:2])
        overlay = overlay_label(rgb, label)
        stat = frame_stats[int(idx)]
        top_ids = ",".join(str(row["object_id"]) for row in stat["top_objects"][:5])
        lines = [
            f"i={idx:03d} source_f={int(stat['frame_id']):04d} ids={stat['visible_id_count']} fg={stat['foreground_ratio']:.3f}",
            f"pre/post={stat.get('propagated_pre_disjoin_count','')}/{stat.get('propagated_post_disjoin_count','')} gap={stat.get('gap_mask_count','')} reuse={stat.get('gap_reuse_event_count','')} relabel={stat.get('output_relabel_event_count','')} dropFilt={stat.get('gap_dropped_by_filter_count','')} uncovered={stat.get('uncovered_ratio_before_gap','')}",
            f"top object ids={top_ids}",
        ]
        panels.append(annotate_panel(overlay, lines, width=cell_width))
    if not panels:
        return
    h = max(p.shape[0] for p in panels)
    canvas = np.full((h, int(cell_width) * len(panels), 3), 245, dtype=np.uint8)
    for col, panel in enumerate(panels):
        canvas[: panel.shape[0], col * int(cell_width) : (col + 1) * int(cell_width)] = panel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def summarize(summary: dict[str, Any], output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    records = record_rows(summary)
    diagnostics = diag_by_index(summary)
    frame_rows: list[dict[str, Any]] = []
    labels: list[np.ndarray] = []
    image_area = 0
    for idx, rec in enumerate(records):
        label = read_label(resolve(rec["label_path"]))
        labels.append(label)
        image_area = int(label.size)
        areas = id_area_map(label)
        diag_features = row_diag_features(diagnostics.get(idx))
        frame_row: dict[str, Any] = {
            "chunk_frame_index": int(idx),
            "frame_id": int(rec.get("frame_id", idx)),
            "label_path": rel(rec["label_path"]),
            "visible_id_count": int(len(areas)),
            "foreground_pixels": int(np.count_nonzero(label > 0)),
            "foreground_ratio": float(np.count_nonzero(label > 0) / max(1, label.size)),
            "medium_or_larger_id_count": int(sum(1 for v in areas.values() if v >= int(args.medium_area_px))),
            "large_id_count": int(sum(1 for v in areas.values() if v >= int(args.large_area_px))),
            "top_objects": top_objects(areas, label.size, limit=12),
        }
        frame_row.update(diag_features)
        frame_rows.append(frame_row)

    adjacent_rows: list[dict[str, Any]] = []
    id_switch_rows: list[dict[str, Any]] = []
    for idx in range(1, len(labels)):
        prev = labels[idx - 1]
        curr = labels[idx]
        prev_areas = id_area_map(prev)
        curr_areas = id_area_map(curr)
        prev_ids = set(prev_areas)
        curr_ids = set(curr_areas)
        prev_medium = {obj_id for obj_id, area in prev_areas.items() if area >= int(args.medium_area_px)}
        curr_medium = {obj_id for obj_id, area in curr_areas.items() if area >= int(args.medium_area_px)}
        lost_medium = sorted(prev_medium - curr_ids)
        gained_medium = sorted(curr_medium - prev_ids)
        both_fg = (prev > 0) & (curr > 0)
        ownership_change = both_fg & (prev != curr)
        fg_lost = (prev > 0) & (curr == 0)
        fg_gained = (prev == 0) & (curr > 0)
        diag = diagnostics.get(idx)
        prev_diag = diagnostics.get(idx - 1)
        row = {
            "prev_chunk_frame_index": int(idx - 1),
            "chunk_frame_index": int(idx),
            "prev_frame_id": int(frame_rows[idx - 1]["frame_id"]),
            "frame_id": int(frame_rows[idx]["frame_id"]),
            "prev_visible_id_count": int(frame_rows[idx - 1]["visible_id_count"]),
            "visible_id_count": int(frame_rows[idx]["visible_id_count"]),
            "visible_id_delta": int(frame_rows[idx]["visible_id_count"] - frame_rows[idx - 1]["visible_id_count"]),
            "prev_foreground_ratio": float(frame_rows[idx - 1]["foreground_ratio"]),
            "foreground_ratio": float(frame_rows[idx]["foreground_ratio"]),
            "foreground_ratio_delta": float(frame_rows[idx]["foreground_ratio"] - frame_rows[idx - 1]["foreground_ratio"]),
            "foreground_lost_ratio": float(np.count_nonzero(fg_lost) / max(1, image_area)),
            "foreground_gained_ratio": float(np.count_nonzero(fg_gained) / max(1, image_area)),
            "ownership_change_ratio": float(np.count_nonzero(ownership_change) / max(1, image_area)),
            "medium_lost_count": int(len(lost_medium)),
            "medium_gained_count": int(len(gained_medium)),
            "medium_lost_object_ids": ",".join(str(v - 1) for v in lost_medium),
            "medium_gained_object_ids": ",".join(str(v - 1) for v in gained_medium),
            "pre_to_post_drop_count": int(row_diag_features(diag).get("pre_to_post_drop_count") or 0),
            "gap_before_filter_count": int(row_diag_features(diag).get("gap_before_filter_count") or 0),
            "gap_dropped_by_filter_count": int(row_diag_features(diag).get("gap_dropped_by_filter_count") or 0),
            "gap_dropped_by_bbox_count": int(row_diag_features(diag).get("gap_dropped_by_bbox_count") or 0),
            "oversized_event_count": int(row_diag_features(diag).get("oversized_event_count") or 0),
            "stream_pruned_object_count": int(row_diag_features(diag).get("stream_pruned_object_count") or 0),
            "uncovered_ratio_before_gap": float((diag or {}).get("uncovered_ratio_before_gap", 0.0)),
            "uncovered_ratio_delta": float((diag or {}).get("uncovered_ratio_before_gap", 0.0))
            - float((prev_diag or {}).get("uncovered_ratio_before_gap", 0.0)),
        }
        risk = (
            max(0.0, -float(row["foreground_ratio_delta"])) * 3.0
            + float(row["ownership_change_ratio"]) * 2.0
            + float(row["foreground_lost_ratio"]) * 2.0
            + min(0.25, float(row["medium_lost_count"]) * 0.03)
            + min(0.25, float(row["pre_to_post_drop_count"]) * 0.01)
            + min(0.25, float(row["gap_dropped_by_filter_count"]) * 0.04)
            + min(0.25, float(row["oversized_event_count"]) * 0.10)
            + max(0.0, float(row["uncovered_ratio_delta"])) * 1.5
        )
        row["diagnostic_risk_score"] = float(risk)
        adjacent_rows.append(row)

        for lost_id in lost_medium:
            ov = best_overlap(prev, curr, lost_id, sorted(curr_ids))
            if float(ov.get("best_iou", 0.0)) >= float(args.id_switch_iou):
                out = {
                    "prev_chunk_frame_index": int(idx - 1),
                    "chunk_frame_index": int(idx),
                    "prev_frame_id": int(frame_rows[idx - 1]["frame_id"]),
                    "frame_id": int(frame_rows[idx]["frame_id"]),
                    "lost_label_id": int(lost_id),
                    "lost_object_id": int(lost_id) - 1,
                    "lost_area_px": int(prev_areas[lost_id]),
                    "best_current_label_id": int(ov["best_label_id"]),
                    "best_current_object_id": int(ov["best_object_id"]),
                    "best_iou": float(ov["best_iou"]),
                    "best_intersection_px": int(ov["best_intersection_px"]),
                    "same_iou": float(ov["same_iou"]),
                    "prev_bbox_xyxy": bbox_for_id(prev, lost_id),
                    "best_current_bbox_xyxy": bbox_for_id(curr, int(ov["best_label_id"])) if int(ov["best_label_id"]) > 0 else [],
                }
                id_switch_rows.append(out)

    frame_csv = output_root / "phase34_frame_label_stats.csv"
    churn_csv = output_root / "phase34_adjacent_churn_rows.csv"
    switch_csv = output_root / "phase34_possible_id_switch_rows.csv"
    write_csv(frame_csv, [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in row.items()} for row in frame_rows])
    write_csv(churn_csv, adjacent_rows)
    write_csv(switch_csv, [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in row.items()} for row in id_switch_rows])

    top_churn = sorted(adjacent_rows, key=lambda r: float(r["diagnostic_risk_score"]), reverse=True)[: int(args.top_k)]
    top_drop = sorted(adjacent_rows, key=lambda r: float(r["foreground_ratio_delta"]))[: int(args.top_k)]
    top_uncovered = sorted(adjacent_rows, key=lambda r: float(r["uncovered_ratio_delta"]), reverse=True)[: int(args.top_k)]
    top_filter = sorted(adjacent_rows, key=lambda r: int(r["gap_dropped_by_filter_count"]), reverse=True)[: int(args.top_k)]
    top_switch = sorted(id_switch_rows, key=lambda r: (float(r["best_iou"]), int(r["lost_area_px"])), reverse=True)[: int(args.top_k)]

    selected_starts = set()
    for row in top_churn[:8] + top_drop[:4] + top_uncovered[:4] + top_filter[:4]:
        selected_starts.add(max(0, int(row["chunk_frame_index"]) // 4 * 4))
    for row in top_switch[:8]:
        selected_starts.add(max(0, int(row["chunk_frame_index"]) // 4 * 4))
    for idx in [0, 16, 236, 240, 248, 260, 264, 268, 452, 456]:
        selected_starts.add(max(0, min(len(records) - 1, idx) // 4 * 4))
    sheet_dir = output_root / "diagnostic_label_sheets_4f"
    sheet_paths: list[str] = []
    frame_stat_by_idx = {int(row["chunk_frame_index"]): row for row in frame_rows}
    for start in sorted(selected_starts):
        path = sheet_dir / f"phase34_label_diag_sheet_n{start:03d}_to_{min(start + 3, len(records) - 1):03d}.jpg"
        make_diagnostic_sheet(path, records, frame_stat_by_idx, start, cell_width=int(args.sheet_cell_width))
        if path.exists():
            sheet_paths.append(rel(path))

    summary_payload = {
        "schema_version": "stream4d_v108_phase34_label_churn_diagnostic_v1",
        "summary_path": rel(args.summary),
        "summary_sha256": sha256_file(resolve(args.summary)),
        "frame_count": int(len(records)),
        "image_area": int(image_area),
        "medium_area_px": int(args.medium_area_px),
        "large_area_px": int(args.large_area_px),
        "id_switch_iou": float(args.id_switch_iou),
        "mean_foreground_ratio": float(np.mean([r["foreground_ratio"] for r in frame_rows])) if frame_rows else 0.0,
        "min_foreground_ratio": float(np.min([r["foreground_ratio"] for r in frame_rows])) if frame_rows else 0.0,
        "max_uncovered_ratio_before_gap": float(
            max(float(r.get("uncovered_ratio_before_gap") or 0.0) for r in frame_rows)
        )
        if frame_rows
        else 0.0,
        "frame_csv": rel(frame_csv),
        "adjacent_churn_csv": rel(churn_csv),
        "possible_id_switch_csv": rel(switch_csv),
        "diagnostic_sheet_paths": sheet_paths,
        "top_risk_adjacent_rows": top_churn,
        "top_foreground_drop_rows": top_drop,
        "top_uncovered_jump_rows": top_uncovered,
        "top_gap_filter_drop_rows": top_filter,
        "top_possible_id_switch_rows": top_switch,
        "diagnostic_only": True,
        "visual_confirmation_required": True,
    }
    summary_json = output_root / "phase34_label_churn_summary.json"
    write_json(summary_json, summary_payload)
    write_markdown(output_root / "phase34_label_churn_report.md", summary_payload)
    return summary_payload


def table(rows: list[dict[str, Any]], cols: list[str], limit: int = 8) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows[:limit]:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# v108 Phase34 scene0011 label churn diagnostic",
        "",
        "This report is diagnostic-only. Metrics here identify failure modes; visual review remains the authority.",
        "",
        "## Core interpretation",
        "",
        "- The current rolling video is label-backed, so sparse coverage is not only a video export issue.",
        "- High-risk frames combine active-mask disjoin loss, gap candidates dropped by output shape filters, oversized active-object events, and foreground loss.",
        "- The plan-directed repair target is output-plane ownership/lifecycle behavior, not another success claim from aggregate metrics.",
        "",
        "## Aggregate",
        "",
        f"- frame_count: {payload['frame_count']}",
        f"- mean_foreground_ratio: {payload['mean_foreground_ratio']:.4f}",
        f"- min_foreground_ratio: {payload['min_foreground_ratio']:.4f}",
        f"- max_uncovered_ratio_before_gap: {payload['max_uncovered_ratio_before_gap']:.4f}",
        f"- medium_area_px: {payload['medium_area_px']}",
        f"- id_switch_iou: {payload['id_switch_iou']}",
        "",
        "## Top Adjacent Risk Rows",
        "",
    ]
    lines += table(
        payload["top_risk_adjacent_rows"],
        [
            "chunk_frame_index",
            "frame_id",
            "diagnostic_risk_score",
            "foreground_ratio_delta",
            "ownership_change_ratio",
            "medium_lost_count",
            "pre_to_post_drop_count",
            "gap_dropped_by_filter_count",
            "oversized_event_count",
            "uncovered_ratio_delta",
        ],
    )
    lines += ["", "## Top Gap Filter Drops", ""]
    lines += table(
        payload["top_gap_filter_drop_rows"],
        [
            "chunk_frame_index",
            "frame_id",
            "gap_before_filter_count",
            "gap_dropped_by_filter_count",
            "gap_dropped_by_bbox_count",
            "uncovered_ratio_before_gap",
            "foreground_ratio",
        ],
    )
    lines += ["", "## Top Possible ID Switch Rows", ""]
    lines += table(
        payload["top_possible_id_switch_rows"],
        [
            "chunk_frame_index",
            "frame_id",
            "lost_object_id",
            "lost_area_px",
            "best_current_object_id",
            "best_iou",
            "same_iou",
        ],
    )
    lines += [
        "",
        "## Diagnostic Sheets",
        "",
    ]
    for sheet in payload["diagnostic_sheet_paths"][:24]:
        lines.append(f"- {sheet}")
    lines += [
        "",
        "## Files",
        "",
        f"- frame stats: {payload['frame_csv']}",
        f"- adjacent churn: {payload['adjacent_churn_csv']}",
        f"- possible ID switches: {payload['possible_id_switch_csv']}",
        f"- summary JSON: {rel(path.with_suffix('.json')) if False else 'phase34_label_churn_summary.json'}",
        "",
        "## Conclusion",
        "",
        "The current evidence supports a multi-cause diagnosis: active-mask ownership is unstable for medium objects, large-surface pruning can remove visible coverage, and the gap-output filter drops large candidates exactly when uncovered ratio jumps. A repair should preserve current-frame output separately from durable SAM2 memory and avoid direct deletion on oversized alerts.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--medium-area-px", type=int, default=10000)
    parser.add_argument("--large-area-px", type=int, default=50000)
    parser.add_argument("--id-switch-iou", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sheet-cell-width", type=int, default=520)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = read_json(resolve(args.summary))
    payload = summarize(summary, output_root, args)
    print(
        json.dumps(
            {
                "summary": rel(output_root / "phase34_label_churn_summary.json"),
                "frame_count": payload["frame_count"],
                "top_risk_frame": payload["top_risk_adjacent_rows"][0]["frame_id"]
                if payload["top_risk_adjacent_rows"]
                else None,
                "diagnostic_sheet_count": len(payload["diagnostic_sheet_paths"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
