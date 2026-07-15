#!/usr/bin/env python3
"""Build visual-first LingBot anchor diagnostics for v108 Phase4.

This is a shadow diagnostic. It does not update Stream4D outputs, SAM2 memory,
or lifecycle state. It uses LingBot-Map raw geometry only for projection:
decoded depth, depth confidence, intrinsics, and pose tensors from the LingBot
geometry npz.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
for item in (TOOLS_ROOT, ROOT, ROOT / "Stream3D"):
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


DEFAULT_REFERENCE_ROOT = (
    ROOT
    / "Stream3D/outputs/audit/v108_phase1_reference_scene0050_90f_labelonly_20260714_1442"
    / "v106_stateful_sam2_rolling_scene_stream"
)
DEFAULT_LINGBOT_NPZ = (
    ROOT
    / "Stream3D/outputs/audit/v107_phase6_lingbot_prompt_capsule_delta32_20260713_2145"
    / "lingbot_raw_geometry_outputs.npz"
)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key)) for key in fields})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return ROOT / path


def load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int32, copy=False)


def load_rgb(scene_root: Path, scene_id: str, frame_id: int) -> np.ndarray:
    path = scene_root / scene_id / "color" / f"{int(frame_id)}.jpg"
    rgb_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)


def load_reference_records(reference_root: Path) -> dict[int, dict[str, Any]]:
    summary_path = reference_root / "summary.json"
    if not summary_path.exists():
        nested = reference_root / "v106_stateful_sam2_rolling_scene_stream" / "summary.json"
        if nested.exists():
            summary_path = nested
            reference_root = nested.parent
    summary = read_json(summary_path)
    records: dict[int, dict[str, Any]] = {}
    for row in summary.get("records", []):
        item = dict(row)
        item["label_path"] = resolve_path(str(row["label_path"]), summary_path.parent)
        records[int(row["frame_id"])] = item
    return records


def parse_object_ids(text: str) -> list[int]:
    if not text.strip():
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def select_object_ids(source_label: np.ndarray, target_label: np.ndarray, *, max_objects: int) -> list[int]:
    ids = []
    for obj_id in sorted(int(v) for v in np.unique(target_label).tolist() if int(v) > 0):
        source_area = int(np.count_nonzero(source_label == int(obj_id)))
        target_area = int(np.count_nonzero(target_label == int(obj_id)))
        if source_area > 0 and target_area > 0:
            ids.append((target_area, int(obj_id)))
    ids.sort(reverse=True)
    return [obj_id for _area, obj_id in ids[: int(max_objects)]]


def color_for_obj(obj_id: int) -> tuple[int, int, int]:
    seed = (int(obj_id) * 2654435761) & 0xFFFFFFFF
    return (
        int(70 + ((seed >> 0) & 0x7F)),
        int(70 + ((seed >> 8) & 0x7F)),
        int(70 + ((seed >> 16) & 0x7F)),
    )


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.copy()
    mask_b = np.asarray(mask).astype(bool)
    if np.any(mask_b):
        c = np.asarray(color, dtype=np.float32)
        out[mask_b] = ((1.0 - float(alpha)) * out[mask_b].astype(np.float32) + float(alpha) * c).clip(0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2, lineType=cv2.LINE_AA)
    return out


def draw_point(
    image: np.ndarray,
    xy: tuple[float, float],
    *,
    color: tuple[int, int, int],
    label: str,
    radius: int = 7,
) -> None:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(image, (x, y), radius + 2, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(image, label[:8], (x + radius + 3, y - radius - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    header = 34
    out = np.zeros((image.shape[0] + header, image.shape[1], 3), dtype=np.uint8)
    out[:] = 12
    out[header:] = image
    cv2.putText(out, text[:180], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def project_sample(
    *,
    geometry: dict[str, np.ndarray],
    pose_c2w: np.ndarray,
    source_index: int,
    target_index: int,
    source_yx: tuple[int, int],
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
    min_depth_conf: float,
) -> tuple[dict[str, Any] | None, str]:
    projected, status, _meta = visibility_project(
        source_xy=(int(source_yx[0]), int(source_yx[1])),
        source_index=int(source_index),
        target_index=int(target_index),
        geometry=geometry,
        pose_c2w=pose_c2w,
        depth_abs_tolerance=float(depth_abs_tolerance),
        depth_rel_tolerance=float(depth_rel_tolerance),
        min_depth_conf=float(min_depth_conf),
    )
    return projected, status


def build_case(
    *,
    obj_id: int,
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
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    shape_hw = tuple(int(v) for v in geometry["depth"].shape[1:3])
    source_label = resize_label_to_shape(source_label_orig, shape_hw)
    target_label = resize_label_to_shape(target_label_orig, shape_hw)
    source_idx = int(frame_to_index[int(source_frame_id)])
    target_idx = int(frame_to_index[int(target_frame_id)])
    pose_c2w = geometry["poses_direct"] if str(pose_mode) == "direct_as_c2w" else geometry["poses_inverted"]
    source_mask = source_label == int(obj_id)
    target_mask = target_label == int(obj_id)
    source_bbox = bbox_from_mask(source_mask)
    pos_points, pos_sample_stats = sample_interior_points(
        source_mask,
        count=int(args.positive_points),
        min_distance_px=float(args.source_core_min_distance_px),
        seed=int(args.seed) + case_index * 10007 + int(obj_id),
    )

    negative_candidates: list[dict[str, Any]] = []
    if source_bbox is not None:
        for other_id in sorted(int(v) for v in np.unique(source_label).tolist() if int(v) > 0 and int(v) != int(obj_id)):
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
                seed=int(args.seed) + case_index * 17713 + int(other_id),
            )
            if not pts:
                continue
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

    records: list[dict[str, Any]] = []
    source_point_draws: list[tuple[str, int, tuple[int, int, float]]] = []
    target_positive_xy: list[tuple[float, float]] = []
    target_negative_xy: list[tuple[float, float]] = []
    lingbot_hw = shape_hw
    source_orig_hw = source_rgb.shape[:2]
    target_orig_hw = target_rgb.shape[:2]

    def add_records(role: str, source_obj_id: int, points: list[tuple[int, int, float]], neighbor_distance: float) -> None:
        for point_idx, (y, x, source_distance) in enumerate(points):
            source_point_draws.append((role, int(source_obj_id), (int(y), int(x), float(source_distance))))
            projected, status = project_sample(
                geometry=geometry,
                pose_c2w=pose_c2w,
                source_index=source_idx,
                target_index=target_idx,
                source_yx=(int(y), int(x)),
                depth_abs_tolerance=float(args.depth_abs_tolerance),
                depth_rel_tolerance=float(args.depth_rel_tolerance),
                min_depth_conf=float(args.min_depth_conf),
            )
            base = {
                "case_index": int(case_index),
                "role": role,
                "target_obj_id": int(obj_id),
                "source_obj_id": int(source_obj_id),
                "source_frame_id": int(source_frame_id),
                "target_frame_id": int(target_frame_id),
                "source_frame_index": int(source_idx),
                "target_frame_index": int(target_idx),
                "point_index": int(point_idx),
                "source_x_lingbot": float(x),
                "source_y_lingbot": float(y),
                "source_distance_to_mask_edge_px": float(source_distance),
                "source_neighbor_bbox_distance_px": float(neighbor_distance),
                "projection_status": str(status),
                "projection_visible_unoccluded": bool(projected is not None),
            }
            sx, sy = map_lingbot_xy_to_original(float(x), float(y), lingbot_hw=lingbot_hw, orig_hw=source_orig_hw)
            base.update({"source_x_original": float(sx), "source_y_original": float(sy)})
            if projected is not None:
                tx, ty = map_lingbot_xy_to_original(
                    float(projected["target_x"]),
                    float(projected["target_y"]),
                    lingbot_hw=lingbot_hw,
                    orig_hw=target_orig_hw,
                )
                base.update(
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
                    target_positive_xy.append((float(tx), float(ty)))
                else:
                    target_negative_xy.append((float(tx), float(ty)))
            records.append(base)

    add_records("positive", int(obj_id), pos_points, 0.0)
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
        target_positive_xy,
        target_negative_xy,
        negative_radius_px=float(args.anchor_conflict_negative_radius_px),
        positive_cluster_radius_px=float(args.anchor_conflict_positive_cluster_radius_px),
        min_positive_points=int(args.anchor_conflict_min_positive_points),
    )

    source_vis = source_rgb.copy()
    target_vis = target_rgb.copy()
    source_mask_orig = source_label_orig == int(obj_id)
    target_mask_orig = target_label_orig == int(obj_id)
    source_vis = overlay_mask(source_vis, source_mask_orig, color=(40, 220, 255), alpha=0.35)
    target_vis = overlay_mask(target_vis, target_mask_orig, color=(40, 220, 255), alpha=0.35)

    for candidate in negative_candidates:
        other_orig = source_label_orig == int(candidate["source_obj_id"])
        source_vis = overlay_mask(source_vis, other_orig, color=(240, 80, 80), alpha=0.18)

    for role, source_obj_id, (y, x, _dist) in source_point_draws:
        sx, sy = map_lingbot_xy_to_original(float(x), float(y), lingbot_hw=lingbot_hw, orig_hw=source_orig_hw)
        if role == "positive":
            draw_point(source_vis, (sx, sy), color=(20, 245, 80), label="P")
        else:
            draw_point(source_vis, (sx, sy), color=(250, 60, 50), label=f"N{source_obj_id}")
    for row in records:
        if not bool(row.get("projection_visible_unoccluded", False)):
            continue
        xy = (float(row["target_x_original"]), float(row["target_y_original"]))
        if str(row["role"]) == "positive":
            draw_point(target_vis, xy, color=(20, 245, 80), label="P")
        else:
            draw_point(target_vis, xy, color=(250, 60, 50), label=f"N{int(row['source_obj_id'])}")

    source_head = add_header(source_vis, f"source frame {source_frame_id} obj {obj_id}; green positive core, red nearby co-visible negative cores")
    target_head = add_header(target_vis, f"target frame {target_frame_id} obj {obj_id}; only LingBot-visible unoccluded projections drawn")
    if source_head.shape[0] != target_head.shape[0]:
        h = max(source_head.shape[0], target_head.shape[0])
        def pad_to(img: np.ndarray) -> np.ndarray:
            if img.shape[0] == h:
                return img
            out = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
            out[:] = 12
            out[: img.shape[0], : img.shape[1]] = img
            return out
        source_head = pad_to(source_head)
        target_head = pad_to(target_head)
    panel = np.concatenate([source_head, target_head], axis=1)
    vis_path = output_root / "visual_checks" / (
        f"case_{case_index:02d}_{args.scene_id}_src{source_frame_id:06d}_tgt{target_frame_id:06d}_obj{obj_id:04d}.png"
    )
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    summary = {
        "case_index": int(case_index),
        "scene_id": str(args.scene_id),
        "target_obj_id": int(obj_id),
        "source_frame_id": int(source_frame_id),
        "target_frame_id": int(target_frame_id),
        "geometry_source": "LingBot-Map decoded pose_enc + depth + depth_conf + intrinsics",
        "uses_scannet_pose_or_depth_for_projection": False,
        "pose_mode": str(pose_mode),
        "source_positive_sample_stats": pos_sample_stats,
        "negative_candidate_count": int(len(negative_candidates)),
        "negative_candidate_object_ids": [int(row["source_obj_id"]) for row in negative_candidates],
        "source_support": source_support,
        "target_support": target_support,
        "projected_positive_count": int(len(target_positive_xy)),
        "projected_negative_count": int(len(target_negative_xy)),
        "projection_rejection_counts": {
            reason: int(sum(1 for row in records if str(row.get("projection_status")) == reason))
            for reason in sorted({str(row.get("projection_status")) for row in records})
            if reason != "ok"
        },
        "conflict_diagnostics": conflict,
        "visual_path": rel(vis_path),
        "visual_sha256": sha256_file(vis_path),
        "visual_review_required": True,
        "acceptance_rule": "Metrics are diagnostic only; quality must be judged by high-resolution visual review.",
    }
    return summary, records, vis_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--scene-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--reference-run-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--lingbot-geometry-npz", default=str(DEFAULT_LINGBOT_NPZ))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-frame-id", type=int, default=4495)
    parser.add_argument("--target-frame-id", type=int, default=4500)
    parser.add_argument("--object-ids", default="")
    parser.add_argument("--max-objects", type=int, default=3)
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
    parser.add_argument("--seed", type=int, default=1084)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    scene_root = resolve_path(str(args.scene_root), ROOT)
    reference_root = resolve_path(str(args.reference_run_root), ROOT)
    npz_path = resolve_path(str(args.lingbot_geometry_npz), ROOT)
    output_root = resolve_path(str(args.output_root), ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    geometry = load_lingbot_geometry(npz_path)
    frame_ids = [int(v) for v in np.asarray(geometry["frame_ids"]).tolist()]
    frame_to_index = {int(frame_id): int(idx) for idx, frame_id in enumerate(frame_ids)}
    unavailable = []
    for frame_id in [int(args.source_frame_id), int(args.target_frame_id)]:
        if int(frame_id) not in frame_to_index:
            unavailable.append(int(frame_id))
    if unavailable:
        summary = {
            "schema_version": "stream4d_v108_phase4_lingbot_anchor_visual_diagnostic_v1",
            "created_unix_time": time.time(),
            "runtime_sec": float(time.time() - started),
            "scene_id": str(args.scene_id),
            "geometry_available": False,
            "unavailable_frame_ids": unavailable,
            "lingbot_geometry_npz": rel(npz_path),
            "lingbot_geometry_npz_sha256": sha256_file(npz_path),
            "lingbot_frame_id_first_last": [int(frame_ids[0]), int(frame_ids[-1])] if frame_ids else [],
            "uses_scannet_pose_or_depth_for_projection": False,
            "acceptance_rule": "Metrics are diagnostic only; no quality judgment without visual confirmation.",
        }
        write_json(output_root / "phase4_lingbot_anchor_visual_summary.json", summary)
        print(json.dumps({"summary": rel(output_root / "phase4_lingbot_anchor_visual_summary.json"), "geometry_available": False}, sort_keys=True))
        return 0

    records_by_frame = load_reference_records(reference_root)
    for frame_id in [int(args.source_frame_id), int(args.target_frame_id)]:
        if int(frame_id) not in records_by_frame:
            raise RuntimeError({"missing_reference_label_for_frame_id": int(frame_id), "reference_root": rel(reference_root)})
    source_label = load_label(Path(records_by_frame[int(args.source_frame_id)]["label_path"]))
    target_label = load_label(Path(records_by_frame[int(args.target_frame_id)]["label_path"]))
    source_rgb = load_rgb(scene_root, str(args.scene_id), int(args.source_frame_id))
    target_rgb = load_rgb(scene_root, str(args.scene_id), int(args.target_frame_id))

    object_ids = parse_object_ids(str(args.object_ids))
    if not object_ids:
        object_ids = select_object_ids(source_label, target_label, max_objects=int(args.max_objects))
    case_summaries: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    visual_paths: list[Path] = []
    for case_idx, obj_id in enumerate(object_ids):
        summary, rows, vis_path = build_case(
            obj_id=int(obj_id),
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
            case_index=int(case_idx),
            output_root=output_root,
        )
        case_summaries.append(summary)
        all_records.extend(rows)
        visual_paths.append(vis_path)

    records_csv = output_root / "projection_point_records.csv"
    records_jsonl = output_root / "projection_point_records.jsonl"
    write_csv(records_csv, all_records)
    records_jsonl.write_text(
        "".join(json.dumps(jsonable(row), sort_keys=True) + "\n" for row in all_records),
        encoding="utf-8",
    )
    case_summary_path = output_root / "case_summaries.json"
    write_json(case_summary_path, {"cases": case_summaries})
    summary_path = output_root / "phase4_lingbot_anchor_visual_summary.json"
    summary = {
        "schema_version": "stream4d_v108_phase4_lingbot_anchor_visual_diagnostic_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "scene_id": str(args.scene_id),
        "geometry_available": True,
        "geometry_source": "LingBot-Map decoded pose_enc + depth + depth_conf + intrinsics",
        "uses_scannet_pose_or_depth_for_projection": False,
        "lingbot_geometry_npz": rel(npz_path),
        "lingbot_geometry_npz_sha256": sha256_file(npz_path),
        "lingbot_frame_id_first_last": [int(frame_ids[0]), int(frame_ids[-1])] if frame_ids else [],
        "source_frame_id": int(args.source_frame_id),
        "target_frame_id": int(args.target_frame_id),
        "object_ids": [int(v) for v in object_ids],
        "case_count": int(len(case_summaries)),
        "projection_record_count": int(len(all_records)),
        "records_csv": rel(records_csv),
        "records_csv_sha256": sha256_file(records_csv),
        "records_jsonl": rel(records_jsonl),
        "records_jsonl_sha256": sha256_file(records_jsonl),
        "case_summaries": rel(case_summary_path),
        "case_summaries_sha256": sha256_file(case_summary_path),
        "visual_paths": [rel(path) for path in visual_paths],
        "visual_sha256": {rel(path): sha256_file(path) for path in visual_paths},
        "acceptance_rule": "Metrics are diagnostic only; quality must be judged by high-resolution visual review.",
        "shadow_only": True,
    }
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), "case_count": len(case_summaries), "visual_count": len(visual_paths)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
