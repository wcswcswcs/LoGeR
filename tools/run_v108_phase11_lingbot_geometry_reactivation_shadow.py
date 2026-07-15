#!/usr/bin/env python3
"""LingBot geometry reactivation shadow for v108 Phase11.

This controlled runner uses LingBot-Map depth/pose/intrinsics/depth_conf to
project historical positive and nearby co-visible negative anchors, then prompts
SAM2 image predictor on the target frame. It is visual-first and shadow-only:
no Stream4D output, SAM2 video memory, or lifecycle state is mutated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
STREAM3D_ROOT = ROOT / "Stream3D"
GSAM2_ROOT = Path(os.environ.get("GSAM2_ROOT", str(ROOT / "Grounded-SAM-2"))).resolve()
for item in (TOOLS_ROOT, ROOT, STREAM3D_ROOT, GSAM2_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from Stream3D.stream4d_v108.geometry_capsule import (  # noqa: E402
    bbox_distance,
    bbox_from_mask,
    mask_depth_support,
    point_conflict_diagnostics,
    sample_interior_points,
)
from run_v107_phase5_prompt_capsule_visibility_probe import (  # noqa: E402
    load_lingbot_geometry,
    resize_label_to_shape,
    visibility_project,
)
from run_v107_phase7_lingbot_sam2_prompt_benchmark import map_lingbot_xy_to_original  # noqa: E402
from run_v108_phase10_2d_reactivation_shadow import (  # noqa: E402
    autocast_kwargs,
    build_sam2_predictor,
    mask_stats,
    predict_variant,
    random_box_like,
)


DEFAULT_REFERENCE_ROOT = (
    ROOT
    / "Stream3D/outputs/audit/v107_phase35_scene0030_crossscene_p34mechanism_30f_20260714_1905"
    / "v107_phase8_g3_rolling_scheduler_smoke"
)
DEFAULT_LINGBOT_NPZ = (
    ROOT
    / "Stream3D/outputs/audit/v107_phase11_holdout_scene0030_prompt_capsule90_20260714_0307"
    / "lingbot_raw_geometry_outputs.npz"
)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def resolve_path(text: str, base: Path = ROOT) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key)) for key in fields})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_reference_records(reference_root: Path) -> dict[int, dict[str, Any]]:
    summary_path = reference_root / "summary.json"
    if not summary_path.exists():
        nested = reference_root / "v106_stateful_sam2_rolling_scene_stream" / "summary.json"
        if nested.exists():
            summary_path = nested
    summary = read_json(summary_path)
    records: dict[int, dict[str, Any]] = {}
    for row in summary.get("records", []):
        item = dict(row)
        item["label_path"] = resolve_path(str(row["label_path"]), summary_path.parent)
        records[int(row["frame_id"])] = item
    return records


def load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int32, copy=False)


def load_rgb(scene_root: Path, scene_id: str, frame_id: int) -> np.ndarray:
    path = scene_root / scene_id / "color" / f"{int(frame_id)}.jpg"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_ids(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.copy()
    mask_b = np.asarray(mask).astype(bool)
    if np.any(mask_b):
        c = np.asarray(color, dtype=np.float32)
        out[mask_b] = ((1.0 - alpha) * out[mask_b].astype(np.float32) + alpha * c).clip(0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2, lineType=cv2.LINE_AA)
    return out


def draw_point(image: np.ndarray, xy: tuple[float, float], *, color: tuple[int, int, int], label: str) -> None:
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    cv2.circle(image, (x, y), 7, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(image, (x, y), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(image, label[:8], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def draw_box(image: np.ndarray, box: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    header = 34
    out = np.zeros((image.shape[0] + header, image.shape[1], 3), dtype=np.uint8)
    out[:] = 12
    out[header:] = image
    cv2.putText(out, text[:170], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def padded_bounds(
    image_hw: tuple[int, int],
    *,
    masks: list[np.ndarray],
    points_xy: list[tuple[float, float]],
    boxes: list[tuple[float, float, float, float]],
    pad_px: int,
) -> tuple[int, int, int, int]:
    h, w = int(image_hw[0]), int(image_hw[1])
    xs: list[int] = []
    ys: list[int] = []
    for mask in masks:
        bbox = bbox_from_mask(np.asarray(mask).astype(bool))
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        xs.extend([int(x0), int(x1)])
        ys.extend([int(y0), int(y1)])
    for x, y in points_xy:
        xs.append(int(round(float(x))))
        ys.append(int(round(float(y))))
    for box in boxes:
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
        xs.extend([x0, x1])
        ys.extend([y0, y1])
    if not xs or not ys:
        return (0, 0, w - 1, h - 1)
    return (
        max(0, min(xs) - int(pad_px)),
        max(0, min(ys) - int(pad_px)),
        min(w - 1, max(xs) + int(pad_px)),
        min(h - 1, max(ys) + int(pad_px)),
    )


def crop_arr(arr: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop
    return arr[y0 : y1 + 1, x0 : x1 + 1].copy()


def shift_xy(points_xy: list[tuple[float, float]], crop: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    return [(float(x) - crop[0], float(y) - crop[1]) for x, y in points_xy]


def shift_box(box: tuple[float, float, float, float], crop: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    return (box[0] - crop[0], box[1] - crop[1], box[2] - crop[0], box[3] - crop[1])


def box_from_points(
    points_xy: list[tuple[float, float]],
    image_hw: tuple[int, int],
    *,
    pad_px: float,
    min_side_px: float,
) -> tuple[float, float, float, float] | None:
    if not points_xy:
        return None
    arr = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    x0, y0 = arr.min(axis=0).tolist()
    x1, y1 = arr.max(axis=0).tolist()
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side_x = max(float(min_side_px), float(x1 - x0 + 1.0) + 2.0 * float(pad_px))
    side_y = max(float(min_side_px), float(y1 - y0 + 1.0) + 2.0 * float(pad_px))
    h, w = int(image_hw[0]), int(image_hw[1])
    return (
        float(max(0.0, cx - 0.5 * side_x)),
        float(max(0.0, cy - 0.5 * side_y)),
        float(min(w - 1.0, cx + 0.5 * side_x)),
        float(min(h - 1.0, cy + 0.5 * side_y)),
    )


def project_one(
    *,
    geometry: dict[str, np.ndarray],
    pose_c2w: np.ndarray,
    source_index: int,
    target_index: int,
    yx: tuple[int, int],
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
    min_depth_conf: float,
) -> tuple[dict[str, Any] | None, str]:
    projected, status, _meta = visibility_project(
        source_xy=(int(yx[0]), int(yx[1])),
        source_index=int(source_index),
        target_index=int(target_index),
        geometry=geometry,
        pose_c2w=pose_c2w,
        depth_abs_tolerance=float(depth_abs_tolerance),
        depth_rel_tolerance=float(depth_rel_tolerance),
        min_depth_conf=float(min_depth_conf),
    )
    return projected, str(status)


def collect_projected_prompts(
    *,
    object_id: int,
    source_frame_id: int,
    target_frame_id: int,
    source_label_orig: np.ndarray,
    target_label_orig: np.ndarray,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    geometry: dict[str, np.ndarray],
    frame_to_index: dict[int, int],
    pose_mode: str,
    args: argparse.Namespace,
    case_index: int,
) -> dict[str, Any]:
    shape_hw = tuple(int(v) for v in geometry["depth"].shape[1:3])
    source_label = resize_label_to_shape(source_label_orig, shape_hw)
    target_label = resize_label_to_shape(target_label_orig, shape_hw)
    source_idx = int(frame_to_index[int(source_frame_id)])
    target_idx = int(frame_to_index[int(target_frame_id)])
    pose_c2w = geometry["poses_direct"] if str(pose_mode) == "direct_as_c2w" else geometry["poses_inverted"]
    source_mask = source_label == int(object_id)
    target_mask = target_label == int(object_id)
    source_bbox = bbox_from_mask(source_mask)

    pos_points, pos_sample_stats = sample_interior_points(
        source_mask,
        count=int(args.positive_points),
        min_distance_px=float(args.source_core_min_distance_px),
        seed=int(args.seed) + int(object_id) * 17 + int(case_index),
    )
    negative_candidates: list[dict[str, Any]] = []
    if source_bbox is not None:
        for other_id in sorted(int(v) for v in np.unique(source_label).tolist() if int(v) > 0 and int(v) != int(object_id)):
            other_mask = source_label == int(other_id)
            area = int(np.count_nonzero(other_mask))
            if area < int(args.negative_min_area_px):
                continue
            other_bbox = bbox_from_mask(other_mask)
            if other_bbox is None:
                continue
            neighbor_distance = bbox_distance(source_bbox, other_bbox)
            if neighbor_distance > float(args.negative_neighbor_radius_px):
                continue
            pts, stats = sample_interior_points(
                other_mask,
                count=max(1, int(math.ceil(int(args.negative_points) / max(1, int(args.negative_max_objects))))),
                min_distance_px=float(args.negative_core_min_distance_px),
                seed=int(args.seed) + int(other_id) * 23 + int(case_index),
            )
            if pts:
                negative_candidates.append(
                    {
                        "source_obj_id": int(other_id),
                        "bbox_distance_px": float(neighbor_distance),
                        "source_area_px": int(area),
                        "points": pts,
                        "sample_stats": stats,
                    }
                )
    negative_candidates.sort(key=lambda row: (float(row["bbox_distance_px"]), -int(row["source_area_px"]), int(row["source_obj_id"])))
    negative_candidates = negative_candidates[: int(args.negative_max_objects)]

    lingbot_hw = shape_hw
    source_orig_hw = source_rgb.shape[:2]
    target_orig_hw = target_rgb.shape[:2]
    records: list[dict[str, Any]] = []
    source_draw_points: list[dict[str, Any]] = []
    positive_xy: list[tuple[float, float]] = []
    negative_xy: list[tuple[float, float]] = []

    def add_records(role: str, source_obj_id: int, points: list[tuple[int, int, float]], neighbor_distance: float) -> None:
        for point_idx, (y, x, dist_px) in enumerate(points):
            sx, sy = map_lingbot_xy_to_original(float(x), float(y), lingbot_hw=lingbot_hw, orig_hw=source_orig_hw)
            source_draw_points.append(
                {
                    "role": role,
                    "source_obj_id": int(source_obj_id),
                    "source_x_original": float(sx),
                    "source_y_original": float(sy),
                }
            )
            projected, status = project_one(
                geometry=geometry,
                pose_c2w=pose_c2w,
                source_index=source_idx,
                target_index=target_idx,
                yx=(int(y), int(x)),
                depth_abs_tolerance=float(args.depth_abs_tolerance),
                depth_rel_tolerance=float(args.depth_rel_tolerance),
                min_depth_conf=float(args.min_depth_conf),
            )
            row: dict[str, Any] = {
                "case_index": int(case_index),
                "scene_id": str(args.scene_id),
                "role": role,
                "target_obj_id": int(object_id),
                "source_obj_id": int(source_obj_id),
                "source_frame_id": int(source_frame_id),
                "target_frame_id": int(target_frame_id),
                "source_frame_index": int(source_idx),
                "target_frame_index": int(target_idx),
                "point_index": int(point_idx),
                "source_x_lingbot": float(x),
                "source_y_lingbot": float(y),
                "source_x_original": float(sx),
                "source_y_original": float(sy),
                "source_distance_to_mask_edge_px": float(dist_px),
                "source_neighbor_bbox_distance_px": float(neighbor_distance),
                "projection_status": str(status),
                "projection_visible_unoccluded": bool(projected is not None),
            }
            if projected is not None:
                tx, ty = map_lingbot_xy_to_original(
                    float(projected["target_x"]),
                    float(projected["target_y"]),
                    lingbot_hw=lingbot_hw,
                    orig_hw=target_orig_hw,
                )
                row.update(
                    {
                        "target_x_lingbot": float(projected["target_x"]),
                        "target_y_lingbot": float(projected["target_y"]),
                        "target_x_original": float(tx),
                        "target_y_original": float(ty),
                        "projected_depth_m": float(projected.get("projected_depth", -1.0)),
                        "observed_depth_m": float(projected.get("observed_depth", -1.0)),
                        "depth_abs_error_m": float(projected.get("depth_abs_error", -1.0)),
                        "depth_tolerance_m": float(projected.get("depth_tolerance", -1.0)),
                        "target_depth_conf": float(projected.get("target_depth_conf", -1.0)),
                    }
                )
                if role == "positive":
                    positive_xy.append((float(tx), float(ty)))
                else:
                    negative_xy.append((float(tx), float(ty)))
            records.append(row)

    add_records("positive", int(object_id), pos_points, 0.0)
    remaining_neg = int(args.negative_points)
    for candidate in negative_candidates:
        if remaining_neg <= 0:
            break
        pts = list(candidate["points"])[:remaining_neg]
        add_records("negative", int(candidate["source_obj_id"]), pts, float(candidate["bbox_distance_px"]))
        remaining_neg -= len(pts)

    source_support = mask_depth_support(
        source_mask,
        depth=geometry["depth"][source_idx],
        depth_conf=geometry["depth_conf"][source_idx],
        min_depth_conf=float(args.min_depth_conf),
        core_min_distance_px=float(args.source_core_min_distance_px),
    )
    target_support = mask_depth_support(
        target_mask,
        depth=geometry["depth"][target_idx],
        depth_conf=geometry["depth_conf"][target_idx],
        min_depth_conf=float(args.min_depth_conf),
        core_min_distance_px=float(args.source_core_min_distance_px),
    )
    conflict = point_conflict_diagnostics(
        positive_xy,
        negative_xy,
        negative_radius_px=float(args.anchor_conflict_negative_radius_px),
        positive_cluster_radius_px=float(args.anchor_conflict_positive_cluster_radius_px),
        min_positive_points=int(args.anchor_conflict_min_positive_points),
    )
    return {
        "source_label_lingbot": source_label,
        "target_label_lingbot": target_label,
        "source_mask_orig": source_label_orig == int(object_id),
        "target_mask_orig": target_label_orig == int(object_id),
        "source_positive_sample_stats": pos_sample_stats,
        "negative_candidates": negative_candidates,
        "source_draw_points": source_draw_points,
        "positive_xy": positive_xy,
        "negative_xy": negative_xy,
        "records": records,
        "source_support": source_support,
        "target_support": target_support,
        "conflict_diagnostics": conflict,
        "projection_rejection_counts": {
            reason: int(sum(1 for row in records if str(row.get("projection_status")) == reason))
            for reason in sorted({str(row.get("projection_status")) for row in records})
            if reason != "ok"
        },
    }


def make_visual_panel(
    *,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    source_draw_points: list[dict[str, Any]],
    positive_xy: list[tuple[float, float]],
    negative_xy: list[tuple[float, float]],
    prompt_box: tuple[float, float, float, float] | None,
    candidate_masks: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> Path:
    source_points = [(float(p["source_x_original"]), float(p["source_y_original"])) for p in source_draw_points]
    target_points = [*positive_xy, *negative_xy]
    source_crop = padded_bounds(source_rgb.shape[:2], masks=[source_mask], points_xy=source_points, boxes=[], pad_px=48)
    target_crop = padded_bounds(
        target_rgb.shape[:2],
        masks=[target_mask, *candidate_masks.values()],
        points_xy=target_points,
        boxes=[prompt_box] if prompt_box is not None else [],
        pad_px=48,
    )
    source_panel = overlay_mask(crop_arr(source_rgb, source_crop), crop_arr(source_mask, source_crop), color=(40, 220, 255), alpha=0.36)
    for point in source_draw_points:
        x = float(point["source_x_original"]) - source_crop[0]
        y = float(point["source_y_original"]) - source_crop[1]
        if point["role"] == "positive":
            draw_point(source_panel, (x, y), color=(20, 245, 80), label="P")
        else:
            draw_point(source_panel, (x, y), color=(250, 60, 50), label=f"N{int(point['source_obj_id'])}")
    source_panel = add_header(source_panel, "source anchors: green target positives, red nearby negatives")

    target_ref = overlay_mask(crop_arr(target_rgb, target_crop), crop_arr(target_mask, target_crop), color=(40, 220, 255), alpha=0.30)
    if prompt_box is not None:
        draw_box(target_ref, shift_box(prompt_box, target_crop), (255, 220, 40))
    for x, y in shift_xy(positive_xy, target_crop):
        draw_point(target_ref, (x, y), color=(20, 245, 80), label="P")
    for x, y in shift_xy(negative_xy, target_crop):
        draw_point(target_ref, (x, y), color=(250, 60, 50), label="N")
    target_ref = add_header(target_ref, "target visible unoccluded projections + prompt box")

    panels = [source_panel, target_ref]
    for variant, mask in candidate_masks.items():
        panel = overlay_mask(crop_arr(target_rgb, target_crop), crop_arr(target_mask, target_crop), color=(40, 220, 255), alpha=0.18)
        panel = overlay_mask(panel, crop_arr(mask, target_crop), color=(255, 70, 190), alpha=0.43)
        if prompt_box is not None:
            draw_box(panel, shift_box(prompt_box, target_crop), (255, 220, 40))
        for x, y in shift_xy(positive_xy, target_crop):
            draw_point(panel, (x, y), color=(20, 245, 80), label="P")
        for x, y in shift_xy(negative_xy, target_crop):
            draw_point(panel, (x, y), color=(250, 60, 50), label="N")
        panels.append(add_header(panel, f"{variant}; visual review required"))

    height = max(panel.shape[0] for panel in panels)
    padded: list[np.ndarray] = []
    for panel in panels:
        if panel.shape[0] == height:
            padded.append(panel)
            continue
        out = np.zeros((height, panel.shape[1], 3), dtype=np.uint8)
        out[:] = 12
        out[: panel.shape[0], : panel.shape[1]] = panel
        padded.append(out)
    final = add_header(np.concatenate(padded, axis=1), title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    return out_path


def run_case(
    *,
    predictor: Any,
    args: argparse.Namespace,
    case_index: int,
    object_id: int,
    source_label: np.ndarray,
    target_label: np.ndarray,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    geometry: dict[str, np.ndarray],
    frame_to_index: dict[int, int],
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prompt = collect_projected_prompts(
        object_id=int(object_id),
        source_frame_id=int(args.source_frame_id),
        target_frame_id=int(args.target_frame_id),
        source_label_orig=source_label,
        target_label_orig=target_label,
        source_rgb=source_rgb,
        target_rgb=target_rgb,
        geometry=geometry,
        frame_to_index=frame_to_index,
        pose_mode=str(args.pose_mode),
        args=args,
        case_index=int(case_index),
    )
    positive_xy = list(prompt["positive_xy"])
    negative_xy = list(prompt["negative_xy"])
    target_mask = target_label == int(object_id)
    prompt_box = box_from_points(
        positive_xy,
        target_rgb.shape[:2],
        pad_px=float(args.projected_box_pad_px),
        min_side_px=float(args.projected_box_min_side_px),
    )
    point_pos = np.asarray(positive_xy, dtype=np.float32).reshape(-1, 2) if positive_xy else np.zeros((0, 2), dtype=np.float32)
    point_neg = np.asarray(negative_xy, dtype=np.float32).reshape(-1, 2) if negative_xy else np.zeros((0, 2), dtype=np.float32)
    pos_labels = np.ones((point_pos.shape[0],), dtype=np.int32)
    both_coords = np.concatenate([point_pos, point_neg], axis=0) if point_neg.size else point_pos
    both_labels = np.concatenate([pos_labels, np.zeros((point_neg.shape[0],), dtype=np.int32)], axis=0) if point_neg.size else pos_labels

    variants: dict[str, tuple[np.ndarray | None, np.ndarray | None, tuple[float, float, float, float] | None]] = {
        "G1_positives": (point_pos, pos_labels, None),
        "G2_positives_negatives": (both_coords, both_labels, None),
        "G3_positives_box": (point_pos, pos_labels, prompt_box),
        "G4_positives_negatives_box": (both_coords, both_labels, prompt_box),
    }
    if prompt_box is not None:
        variants["RANDOM_box_control"] = (None, None, random_box_like(prompt_box, target_rgb.shape[:2], int(args.seed) + int(object_id) * 997))

    candidate_rows: list[dict[str, Any]] = []
    candidate_masks: dict[str, np.ndarray] = {}
    for variant, (coords, labels, box) in variants.items():
        has_points = coords is not None and labels is not None and np.asarray(coords).size > 0
        has_box = box is not None
        if not has_points and not has_box:
            candidate_rows.append(
                {
                    "case_index": int(case_index),
                    "scene_id": str(args.scene_id),
                    "object_id": int(object_id),
                    "variant": str(variant),
                    "candidate_generated": False,
                    "skip_reason": "no_visible_unoccluded_positive_points_or_box",
                    "metrics_are_diagnostic_only": True,
                    "visual_review_required": True,
                }
            )
            continue
        candidate, select = predict_variant(
            predictor=predictor,
            args=args,
            point_coords=coords if has_points else None,
            point_labels=labels if has_points else None,
            box=box,
        )
        candidate_masks[str(variant)] = candidate
        row = {
            "case_index": int(case_index),
            "scene_id": str(args.scene_id),
            "object_id": int(object_id),
            "source_frame_id": int(args.source_frame_id),
            "target_frame_id": int(args.target_frame_id),
            "variant": str(variant),
            "candidate_generated": True,
            "positive_point_count": int(len(positive_xy)),
            "negative_point_count": int(len(negative_xy)),
            "box_xyxy": list(box) if box is not None else [],
            "uses_lingbot_geometry": True,
            "uses_scannet_pose_or_depth_for_projection": False,
            "occlusion_checked": True,
            "prompt_points_visible_unoccluded_only": True,
            **select,
            **mask_stats(candidate, target_mask, target_label, int(object_id)),
            "metrics_are_diagnostic_only": True,
            "visual_review_required": True,
        }
        candidate_rows.append(row)

    visual_keep = {name: candidate_masks[name] for name in ["G2_positives_negatives", "G4_positives_negatives_box", "RANDOM_box_control"] if name in candidate_masks}
    visual_path = output_root / "visual_checks" / (
        f"phase11_case_{case_index:02d}_{args.scene_id}_src{int(args.source_frame_id):06d}_"
        f"tgt{int(args.target_frame_id):06d}_obj{int(object_id):04d}.png"
    )
    make_visual_panel(
        source_rgb=source_rgb,
        target_rgb=target_rgb,
        source_mask=prompt["source_mask_orig"],
        target_mask=target_mask,
        source_draw_points=prompt["source_draw_points"],
        positive_xy=positive_xy,
        negative_xy=negative_xy,
        prompt_box=prompt_box,
        candidate_masks=visual_keep,
        out_path=visual_path,
        title=f"Phase11 LingBot geometry reactivation scene={args.scene_id} obj={object_id}; visual judgment primary",
    )
    variant_visuals: dict[str, dict[str, str]] = {}
    for variant_name, candidate_mask in candidate_masks.items():
        variant_path = output_root / "visual_checks" / "variants" / (
            f"phase11_case_{case_index:02d}_{args.scene_id}_src{int(args.source_frame_id):06d}_"
            f"tgt{int(args.target_frame_id):06d}_obj{int(object_id):04d}_{variant_name}.png"
        )
        make_visual_panel(
            source_rgb=source_rgb,
            target_rgb=target_rgb,
            source_mask=prompt["source_mask_orig"],
            target_mask=target_mask,
            source_draw_points=prompt["source_draw_points"],
            positive_xy=positive_xy,
            negative_xy=negative_xy,
            prompt_box=prompt_box,
            candidate_masks={variant_name: candidate_mask},
            out_path=variant_path,
            title=f"Phase11 single variant scene={args.scene_id} obj={object_id} variant={variant_name}; visual judgment primary",
        )
        variant_visuals[variant_name] = {
            "path": rel(variant_path),
            "sha256": sha256_file(variant_path),
        }
    case_summary = {
        "case_index": int(case_index),
        "scene_id": str(args.scene_id),
        "object_id": int(object_id),
        "source_frame_id": int(args.source_frame_id),
        "target_frame_id": int(args.target_frame_id),
        "projected_positive_count": int(len(positive_xy)),
        "projected_negative_count": int(len(negative_xy)),
        "projection_rejection_counts": prompt["projection_rejection_counts"],
        "source_positive_sample_stats": prompt["source_positive_sample_stats"],
        "negative_candidate_object_ids": [int(row["source_obj_id"]) for row in prompt["negative_candidates"]],
        "source_support": prompt["source_support"],
        "target_support": prompt["target_support"],
        "conflict_diagnostics": prompt["conflict_diagnostics"],
        "prompt_box_xyxy": list(prompt_box) if prompt_box is not None else [],
        "visual_path": rel(visual_path),
        "visual_sha256": sha256_file(visual_path),
        "variant_visuals": variant_visuals,
        "uses_lingbot_geometry": True,
        "uses_scannet_pose_or_depth_for_projection": False,
        "occlusion_checked": True,
        "visual_review_required": True,
        "acceptance_rule": "Metrics are diagnostic only; quality must be judged by high-resolution visual review.",
    }
    return case_summary, prompt["records"], candidate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="scene0030_00")
    parser.add_argument("--scene-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--reference-run-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--lingbot-geometry-npz", default=str(DEFAULT_LINGBOT_NPZ))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-frame-id", type=int, required=True)
    parser.add_argument("--target-frame-id", type=int, required=True)
    parser.add_argument("--object-ids", required=True)
    parser.add_argument("--pose-mode", default="direct_as_c2w", choices=["direct_as_c2w", "inverted_as_c2w"])
    parser.add_argument("--positive-points", type=int, default=6)
    parser.add_argument("--negative-points", type=int, default=6)
    parser.add_argument("--negative-max-objects", type=int, default=3)
    parser.add_argument("--negative-min-area-px", type=int, default=64)
    parser.add_argument("--negative-neighbor-radius-px", type=float, default=80.0)
    parser.add_argument("--source-core-min-distance-px", type=float, default=10.0)
    parser.add_argument("--negative-core-min-distance-px", type=float, default=8.0)
    parser.add_argument("--depth-abs-tolerance", type=float, default=0.08)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.04)
    parser.add_argument("--min-depth-conf", type=float, default=1.0)
    parser.add_argument("--anchor-conflict-negative-radius-px", type=float, default=24.0)
    parser.add_argument("--anchor-conflict-positive-cluster-radius-px", type=float, default=80.0)
    parser.add_argument("--anchor-conflict-min-positive-points", type=int, default=3)
    parser.add_argument("--projected-box-pad-px", type=float, default=44.0)
    parser.add_argument("--projected-box-min-side-px", type=float, default=96.0)
    parser.add_argument("--sam2-checkpoint", default="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bf16", choices=["float32", "bf16", "float16"])
    parser.add_argument("--multimask-output", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10811)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    scene_root = resolve_path(str(args.scene_root))
    reference_root = resolve_path(str(args.reference_run_root))
    npz_path = resolve_path(str(args.lingbot_geometry_npz))
    output_root = resolve_path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    geometry = load_lingbot_geometry(npz_path)
    frame_ids = [int(v) for v in np.asarray(geometry["frame_ids"]).tolist()]
    frame_to_index = {int(frame_id): int(idx) for idx, frame_id in enumerate(frame_ids)}
    missing_geometry = [int(v) for v in [args.source_frame_id, args.target_frame_id] if int(v) not in frame_to_index]
    if missing_geometry:
        summary_path = output_root / "phase11_lingbot_geometry_reactivation_summary.json"
        summary = {
            "schema_version": "stream4d_v108_phase11_lingbot_geometry_reactivation_shadow_v1",
            "created_unix_time": time.time(),
            "runtime_sec": float(time.time() - started),
            "scene_id": str(args.scene_id),
            "geometry_available": False,
            "missing_geometry_frame_ids": missing_geometry,
            "lingbot_geometry_npz": rel(npz_path),
            "lingbot_geometry_npz_sha256": sha256_file(npz_path),
            "lingbot_frame_id_first_last": [int(frame_ids[0]), int(frame_ids[-1])] if frame_ids else [],
            "uses_lingbot_geometry": True,
            "uses_scannet_pose_or_depth_for_projection": False,
            "acceptance_rule": "Metrics are diagnostic only; no quality judgment without visual confirmation.",
            "shadow_only": True,
        }
        write_json(summary_path, summary)
        print(json.dumps({"summary": rel(summary_path), "geometry_available": False}, sort_keys=True))
        return 0

    records = load_reference_records(reference_root)
    for frame_id in [int(args.source_frame_id), int(args.target_frame_id)]:
        if int(frame_id) not in records:
            raise RuntimeError({"missing_reference_label_for_frame_id": int(frame_id), "reference_root": rel(reference_root)})
    source_label = load_label(Path(records[int(args.source_frame_id)]["label_path"]))
    target_label = load_label(Path(records[int(args.target_frame_id)]["label_path"]))
    source_rgb = load_rgb(scene_root, str(args.scene_id), int(args.source_frame_id))
    target_rgb = load_rgb(scene_root, str(args.scene_id), int(args.target_frame_id))

    predictor, checkpoint = build_sam2_predictor(args)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast(**autocast_kwargs(args)):
        predictor.set_image(target_rgb)

    case_summaries: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for case_idx, object_id in enumerate(parse_ids(str(args.object_ids))):
        case_summary, prompt_rows, cand_rows = run_case(
            predictor=predictor,
            args=args,
            case_index=int(case_idx),
            object_id=int(object_id),
            source_label=source_label,
            target_label=target_label,
            source_rgb=source_rgb,
            target_rgb=target_rgb,
            geometry=geometry,
            frame_to_index=frame_to_index,
            output_root=output_root,
        )
        case_summaries.append(case_summary)
        projection_rows.extend(prompt_rows)
        candidate_rows.extend(cand_rows)

    projection_csv = output_root / "projection_point_records.csv"
    candidate_csv = output_root / "geometry_reactivation_candidate_rows.csv"
    case_json = output_root / "case_summaries.json"
    write_csv(projection_csv, projection_rows)
    write_csv(candidate_csv, candidate_rows)
    write_json(case_json, {"cases": case_summaries})
    peak_cuda_mb = 0.0
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        peak_cuda_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    summary_path = output_root / "phase11_lingbot_geometry_reactivation_summary.json"
    summary = {
        "schema_version": "stream4d_v108_phase11_lingbot_geometry_reactivation_shadow_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "scene_id": str(args.scene_id),
        "source_frame_id": int(args.source_frame_id),
        "target_frame_id": int(args.target_frame_id),
        "object_ids": parse_ids(str(args.object_ids)),
        "case_count": int(len(case_summaries)),
        "projection_row_count": int(len(projection_rows)),
        "candidate_row_count": int(len(candidate_rows)),
        "lingbot_geometry_npz": rel(npz_path),
        "lingbot_geometry_npz_sha256": sha256_file(npz_path),
        "lingbot_frame_id_first_last": [int(frame_ids[0]), int(frame_ids[-1])] if frame_ids else [],
        "geometry_source": "LingBot-Map decoded pose_enc + depth + depth_conf + intrinsics",
        "uses_lingbot_geometry": True,
        "uses_scannet_pose_or_depth_for_projection": False,
        "occlusion_checked": True,
        "sam2_checkpoint": rel(checkpoint),
        "sam2_checkpoint_sha256": sha256_file(checkpoint),
        "sam2_model_cfg": str(args.sam2_model_cfg),
        "device": str(args.device),
        "model_dtype": str(args.model_dtype),
        "peak_cuda_allocated_mb": float(peak_cuda_mb),
        "projection_csv": rel(projection_csv),
        "projection_csv_sha256": sha256_file(projection_csv),
        "candidate_csv": rel(candidate_csv),
        "candidate_csv_sha256": sha256_file(candidate_csv),
        "case_summaries": rel(case_json),
        "case_summaries_sha256": sha256_file(case_json),
        "visual_paths": [case["visual_path"] for case in case_summaries],
        "visual_sha256": {case["visual_path"]: case["visual_sha256"] for case in case_summaries},
        "variants": ["G1_positives", "G2_positives_negatives", "G3_positives_box", "G4_positives_negatives_box"],
        "controls": ["RANDOM_box_control"],
        "coarse_mask_status": "not implemented in this shadow; G4 uses positives+negatives+box",
        "acceptance_rule": "Metrics are diagnostic only; quality must be judged by high-resolution visual review.",
        "shadow_only": True,
    }
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), "case_count": len(case_summaries), "candidate_row_count": len(candidate_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
