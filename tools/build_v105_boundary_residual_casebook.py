#!/usr/bin/env python3
"""Build a casebook for residual v105 chunk-boundary ID changes.

This is a read-only diagnostic helper for the post-hoc boundary relabel branch.
It recomputes every changed best-IoU match at chunk boundaries, classifies the
overlap pattern, and optionally writes small crop sheets for visual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except Exception:
        return 10**15


def read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int32)


def read_rgb(color_dir: Path, frame_id: int) -> np.ndarray:
    rgb = cv2.imread(str(color_dir / f"{int(frame_id)}.jpg"), cv2.IMREAD_COLOR)
    if rgb is None:
        rgb = cv2.imread(str(color_dir / f"{int(frame_id)}.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(color_dir / f"{int(frame_id)}.jpg")
    return rgb


def label_areas(label: np.ndarray) -> dict[int, int]:
    return {int(v): int(np.count_nonzero(label == int(v))) for v in np.unique(label) if int(v) > 0}


def bbox_for_masks(*masks: np.ndarray, pad: int, height: int, width: int) -> tuple[int, int, int, int]:
    combined = np.zeros((height, width), dtype=bool)
    for mask in masks:
        if mask.shape[:2] != combined.shape:
            continue
        combined |= mask.astype(bool)
    ys, xs = np.where(combined)
    if ys.size == 0:
        return 0, 0, width, height
    x0 = max(0, int(xs.min()) - int(pad))
    y0 = max(0, int(ys.min()) - int(pad))
    x1 = min(width, int(xs.max()) + int(pad) + 1)
    y1 = min(height, int(ys.max()) + int(pad) + 1)
    return x0, y0, x1, y1


def overlay_single(rgb: np.ndarray, label: np.ndarray, object_id: int, *, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.copy()
    mask = label == int(object_id)
    if not np.any(mask):
        return out
    tinted = out.copy()
    tinted[mask] = (
        0.45 * tinted[mask].astype(np.float32)
        + 0.55 * np.array(color, dtype=np.float32)[None, :]
    ).clip(0, 255).astype(np.uint8)
    out[mask] = tinted[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out


def put_lines(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    h = 20 * len(lines) + 8
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), thickness=-1)
    for idx, line in enumerate(lines):
        cv2.putText(out, line[:120], (6, 18 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_sheet(
    *,
    record: dict[str, Any],
    prev_rgb: np.ndarray,
    next_rgb: np.ndarray,
    prev_label: np.ndarray,
    next_label: np.ndarray,
    output_path: Path,
    crop_pad: int,
) -> None:
    prev_id = int(record["prev_id"])
    next_id = int(record["best_next_id"])
    h, w = prev_label.shape[:2]
    bbox = bbox_for_masks(prev_label == prev_id, next_label == next_id, pad=int(crop_pad), height=h, width=w)
    x0, y0, x1, y1 = bbox
    prev_crop = overlay_single(prev_rgb[y0:y1, x0:x1], prev_label[y0:y1, x0:x1], prev_id, color=(0, 220, 0))
    next_crop = overlay_single(next_rgb[y0:y1, x0:x1], next_label[y0:y1, x0:x1], next_id, color=(0, 0, 240))
    target_h = max(prev_crop.shape[0], next_crop.shape[0], 120)
    target_w = max(prev_crop.shape[1], next_crop.shape[1], 160)

    def pad_to(img: np.ndarray) -> np.ndarray:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img
        return canvas

    prev_crop = put_lines(
        pad_to(prev_crop),
        [
            f"prev frame {record['prev_frame_id']} id {prev_id}",
            f"area={record['prev_area']} sig_next={record['prev_significant_next_count']}",
        ],
    )
    next_crop = put_lines(
        pad_to(next_crop),
        [
            f"next frame {record['next_frame_id']} id {next_id}",
            f"area={record['next_area']} sig_prev={record['next_significant_prev_count']}",
        ],
    )
    header = np.zeros((80, target_w * 2, 3), dtype=np.uint8)
    lines = [
        f"boundary {record['prev_chunk_index']}->{record['next_chunk_index']} pair_index={record['pair_index']}",
        f"changed {prev_id}->{next_id} iou={record['best_iou']:.4f} inter={record['intersection_pixels']} cause={record['primary_cause']}",
        f"categories={','.join(record['categories'])}",
    ]
    header = put_lines(header, lines)
    sheet = np.vstack([header, np.hstack([prev_crop, next_crop])])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def overlaps_from_mask(
    source_mask: np.ndarray,
    target_label: np.ndarray,
    source_area: int,
    target_areas: dict[int, int],
    *,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    ids, counts = np.unique(target_label[source_mask], return_counts=True)
    rows: list[dict[str, Any]] = []
    for target_raw, count_raw in zip(ids.tolist(), counts.tolist(), strict=False):
        target_id = int(target_raw)
        intersection = int(count_raw)
        if target_id <= 0:
            continue
        union = int(source_area) + int(target_areas.get(target_id, 0)) - intersection
        iou = float(intersection) / float(max(union, 1))
        rows.append(
            {
                "id": int(target_id),
                "intersection_pixels": int(intersection),
                "area": int(target_areas.get(target_id, 0)),
                "iou": float(iou),
                "passes_iou_threshold": bool(iou >= float(iou_threshold)),
            }
        )
    rows.sort(key=lambda item: float(item["iou"]), reverse=True)
    return rows


def classify_record(
    *,
    best_iou: float,
    best_next_group: list[int],
    prev_sig_next_ids: list[int],
    next_sig_prev_ids: list[int],
    prev_id_still_visible_next: bool,
    best_next_visible_prev: bool,
    low_iou_threshold: float,
) -> tuple[str, list[str]]:
    categories: list[str] = []
    if float(best_iou) < float(low_iou_threshold):
        categories.append("low_iou_weak_overlap")
    if len(best_next_group) > 1:
        categories.append("many_prev_to_one_next")
    if len(prev_sig_next_ids) > 1:
        categories.append("one_prev_to_many_next")
    if len(next_sig_prev_ids) > 1:
        categories.append("next_receives_multiple_prev_overlaps")
    if bool(prev_id_still_visible_next):
        categories.append("prev_id_still_visible_in_next_frame")
    if bool(best_next_visible_prev):
        categories.append("best_next_id_visible_in_prev_frame")
    if not categories:
        categories.append("ambiguous_single_overlap_id_change")

    if "many_prev_to_one_next" in categories and "one_prev_to_many_next" in categories:
        primary = "mixed_merge_split_overlap"
    elif "many_prev_to_one_next" in categories:
        primary = "many_prev_to_one_next_merge_or_prev_overfragment"
    elif "one_prev_to_many_next" in categories:
        primary = "one_prev_to_many_next_split_or_next_overfragment"
    elif "low_iou_weak_overlap" in categories:
        primary = "low_iou_weak_overlap"
    elif "prev_id_still_visible_in_next_frame" in categories or "best_next_id_visible_in_prev_frame" in categories:
        primary = "id_swap_or_crossing_ambiguous"
    else:
        primary = "ambiguous_single_overlap_id_change"
    return primary, categories


def build_casebook(args: argparse.Namespace) -> dict[str, Any]:
    mask_dir = Path(args.mask_dir)
    if not mask_dir.is_absolute():
        mask_dir = REPO_ROOT / mask_dir
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    color_dir = Path(args.rgb_root)
    if not color_dir.is_absolute():
        color_dir = REPO_ROOT / color_dir
    color_dir = color_dir / str(args.scene_id) / "color"
    output_root.mkdir(parents=True, exist_ok=True)

    frame_ids = sorted(numeric_stem(path) for path in mask_dir.glob("*.png") if numeric_stem(path) < 10**12)
    if not frame_ids:
        raise FileNotFoundError({"mask_dir": rel(mask_dir), "reason": "no numeric masks"})

    labels: dict[int, np.ndarray] = {}

    def label(frame_id: int) -> np.ndarray:
        if int(frame_id) not in labels:
            labels[int(frame_id)] = read_label(mask_dir / f"{int(frame_id)}.png")
        return labels[int(frame_id)]

    records: list[dict[str, Any]] = []
    boundary_records: list[dict[str, Any]] = []
    for pair_index in range(len(frame_ids) - 1):
        if (pair_index + 1) % int(args.chunk_size) != 0:
            continue
        prev_frame_id = int(frame_ids[pair_index])
        next_frame_id = int(frame_ids[pair_index + 1])
        prev_label = label(prev_frame_id)
        next_label = label(next_frame_id)
        prev_ids = [int(v) for v in np.unique(prev_label) if int(v) > 0]
        next_ids = [int(v) for v in np.unique(next_label) if int(v) > 0]
        prev_areas = label_areas(prev_label)
        next_areas = label_areas(next_label)

        best_rows: dict[int, dict[str, Any]] = {}
        prev_overlap_rows: dict[int, list[dict[str, Any]]] = {}
        next_sig_prev: dict[int, list[int]] = defaultdict(list)
        for prev_id in prev_ids:
            rows = overlaps_from_mask(
                prev_label == int(prev_id),
                next_label,
                int(prev_areas[prev_id]),
                next_areas,
                iou_threshold=float(args.iou_threshold),
            )
            prev_overlap_rows[prev_id] = rows
            passing = [row for row in rows if bool(row["passes_iou_threshold"])]
            if not passing:
                continue
            best = passing[0]
            best_rows[prev_id] = best
            for row in passing:
                next_sig_prev[int(row["id"])].append(int(prev_id))

        best_next_groups: dict[int, list[int]] = defaultdict(list)
        for prev_id, best in best_rows.items():
            best_next_groups[int(best["id"])].append(int(prev_id))

        changed_in_boundary: list[dict[str, Any]] = []
        for prev_id, best in sorted(best_rows.items()):
            best_next_id = int(best["id"])
            if best_next_id == int(prev_id):
                continue
            prev_sig_next_ids = [int(row["id"]) for row in prev_overlap_rows[prev_id] if bool(row["passes_iou_threshold"])]
            next_sig_prev_ids = sorted(next_sig_prev.get(best_next_id, []))
            primary, categories = classify_record(
                best_iou=float(best["iou"]),
                best_next_group=sorted(best_next_groups.get(best_next_id, [])),
                prev_sig_next_ids=prev_sig_next_ids,
                next_sig_prev_ids=next_sig_prev_ids,
                prev_id_still_visible_next=bool(prev_id in next_ids),
                best_next_visible_prev=bool(best_next_id in prev_ids),
                low_iou_threshold=float(args.low_iou_threshold),
            )
            row = {
                "pair_index": int(pair_index),
                "prev_frame_id": int(prev_frame_id),
                "next_frame_id": int(next_frame_id),
                "prev_chunk_index": int(pair_index // int(args.chunk_size)),
                "next_chunk_index": int((pair_index + 1) // int(args.chunk_size)),
                "prev_id": int(prev_id),
                "best_next_id": int(best_next_id),
                "best_iou": float(best["iou"]),
                "intersection_pixels": int(best["intersection_pixels"]),
                "prev_area": int(prev_areas[prev_id]),
                "next_area": int(next_areas.get(best_next_id, 0)),
                "prev_significant_next_ids": prev_sig_next_ids,
                "prev_significant_next_count": int(len(prev_sig_next_ids)),
                "next_significant_prev_ids": next_sig_prev_ids,
                "next_significant_prev_count": int(len(next_sig_prev_ids)),
                "best_next_group_prev_ids": sorted(best_next_groups.get(best_next_id, [])),
                "best_next_group_prev_count": int(len(best_next_groups.get(best_next_id, []))),
                "prev_id_still_visible_next": bool(prev_id in next_ids),
                "best_next_id_visible_prev": bool(best_next_id in prev_ids),
                "primary_cause": primary,
                "categories": categories,
            }
            records.append(row)
            changed_in_boundary.append(row)
        boundary_records.append(
            {
                "pair_index": int(pair_index),
                "prev_frame_id": int(prev_frame_id),
                "next_frame_id": int(next_frame_id),
                "prev_chunk_index": int(pair_index // int(args.chunk_size)),
                "next_chunk_index": int((pair_index + 1) // int(args.chunk_size)),
                "visible_prev": int(len(prev_ids)),
                "visible_next": int(len(next_ids)),
                "changed_count": int(len(changed_in_boundary)),
                "primary_cause_counts": dict(Counter(row["primary_cause"] for row in changed_in_boundary)),
                "changed_records_first20": changed_in_boundary[:20],
            }
        )

    records.sort(key=lambda item: (int(item["intersection_pixels"]), float(item["best_iou"])), reverse=True)
    sheets: list[dict[str, Any]] = []
    sheet_dir = output_root / "sheets"
    for idx, record in enumerate(records[: int(args.max_sheets)]):
        prev_frame_id = int(record["prev_frame_id"])
        next_frame_id = int(record["next_frame_id"])
        sheet_path = sheet_dir / (
            f"{idx:03d}_b{int(record['prev_chunk_index']):03d}_{int(record['next_chunk_index']):03d}_"
            f"f{prev_frame_id}_to_{next_frame_id}_id{int(record['prev_id'])}_to_{int(record['best_next_id'])}.jpg"
        )
        make_sheet(
            record=record,
            prev_rgb=read_rgb(color_dir, prev_frame_id),
            next_rgb=read_rgb(color_dir, next_frame_id),
            prev_label=label(prev_frame_id),
            next_label=label(next_frame_id),
            output_path=sheet_path,
            crop_pad=int(args.crop_pad),
        )
        sheets.append({**record, "sheet_path": rel(sheet_path), "sheet_sha256": sha256_file(sheet_path)})

    primary_counts = Counter(row["primary_cause"] for row in records)
    category_counts = Counter(category for row in records for category in row["categories"])
    boundary_nonzero = [row for row in boundary_records if int(row["changed_count"]) > 0]
    payload = {
        "schema_version": "stream4d_v105_boundary_residual_casebook_v1",
        "scene_id": str(args.scene_id),
        "mask_dir": rel(mask_dir),
        "output_root": rel(output_root),
        "frame_count": int(len(frame_ids)),
        "chunk_size": int(args.chunk_size),
        "boundary_pair_count": int(len(boundary_records)),
        "boundary_pair_with_residual_count": int(len(boundary_nonzero)),
        "residual_changed_match_count": int(len(records)),
        "iou_threshold": float(args.iou_threshold),
        "low_iou_threshold": float(args.low_iou_threshold),
        "primary_cause_counts": dict(primary_counts),
        "category_counts": dict(category_counts),
        "top_records_by_intersection_first50": records[:50],
        "boundary_records": boundary_records,
        "sheet_count": int(len(sheets)),
        "sheets": sheets,
        "claim_boundary": (
            "Machine overlap casebook for residual post-hoc relabel ID changes. "
            "Categories are evidence tags, not final visual identity labels."
        ),
    }
    records_path = output_root / "boundary_residual_records.json"
    summary_path = output_root / "boundary_residual_casebook_summary.json"
    records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    payload["records_json"] = rel(records_path)
    payload["records_sha256"] = sha256_file(records_path)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    payload["summary_json"] = rel(summary_path)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--rgb-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--iou-threshold", type=float, default=0.05)
    parser.add_argument("--low-iou-threshold", type=float, default=0.15)
    parser.add_argument("--max-sheets", type=int, default=16)
    parser.add_argument("--crop-pad", type=int, default=32)
    return parser


def main() -> None:
    payload = build_casebook(build_parser().parse_args())
    print(
        json.dumps(
            {
                "summary": payload["summary_json"],
                "records": payload["records_json"],
                "residual_changed_match_count": payload["residual_changed_match_count"],
                "sheet_count": payload["sheet_count"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
