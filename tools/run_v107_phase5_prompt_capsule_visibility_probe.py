#!/usr/bin/env python3
"""Probe LingBot-Map visibility-gated prompt capsules for v107.

This diagnostic follows the user-requested repair direction:

* positive prompts come from the object's historical mask points;
* negative prompts come from nearby co-visible sibling objects in that same
  historical view;
* both positive and negative prompts are accepted only when the LingBot-Map
  projection into the target view is visible by a depth/occlusion test.

The geometry source is LingBot-Map output only: decoded pose_enc, predicted
depth, depth_conf, intrinsics, and the model's preprocessed RGB images. It does
not use ScanNet pose/depth for prompt projection.
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    ROOT
    / "Stream3D/outputs/audit/"
    / "v106_stateful_sam2_rolling_scene0050_area20k_e1_preprune6_maxvis45_labelcompact_noempty_full90_gpu6_20260713_1505"
    / "v106_stateful_sam2_rolling_scene_stream"
)


def _ensure_repo_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def resize_label_to_shape(label: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if label.shape[:2] == (h, w):
        return label.astype(np.int32, copy=False)
    return cv2.resize(label.astype(np.int32, copy=False), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)


def parse_frame_ids(text: str, start: int, stride: int, count: int) -> list[int]:
    if text.strip():
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    return [int(start + i * stride) for i in range(int(count))]


def parse_source_lags(text: str) -> list[int]:
    if not text.strip():
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value > 0:
            out.append(value)
    return sorted(set(out))


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0, max(bx0 - ax1, ax0 - bx1))
    dy = max(0, max(by0 - ay1, ay0 - by1))
    return float((dx * dx + dy * dy) ** 0.5)


def eroded_core(mask: np.ndarray, iterations: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    if int(iterations) <= 0:
        return binary.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    core = cv2.erode(binary, kernel, iterations=int(iterations))
    if np.count_nonzero(core) < min(16, max(1, np.count_nonzero(binary) // 10)):
        return binary.astype(bool)
    return core.astype(bool)


def sample_mask_points_spread(mask: np.ndarray, *, count: int, seed: int, max_candidates: int = 2500) -> list[tuple[int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0 or int(count) <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    coords = np.stack([ys.astype(np.float32), xs.astype(np.float32)], axis=1)
    if coords.shape[0] > int(max_candidates):
        keep = rng.choice(coords.shape[0], size=int(max_candidates), replace=False)
        coords = coords[keep]
    if coords.shape[0] <= int(count):
        return [(int(y), int(x)) for y, x in coords.astype(np.int64).tolist()]

    # Deterministic farthest-point spread from a random valid seed.
    chosen: list[int] = [int(rng.integers(0, coords.shape[0]))]
    min_dist2 = np.sum((coords - coords[chosen[0]]) ** 2, axis=1)
    while len(chosen) < int(count):
        idx = int(np.argmax(min_dist2))
        chosen.append(idx)
        dist2 = np.sum((coords - coords[idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    out = coords[np.asarray(chosen, dtype=np.int64)].astype(np.int64)
    return [(int(y), int(x)) for y, x in out.tolist()]


def load_reference_records(reference_root: Path, frame_ids: list[int]) -> tuple[dict[int, dict[str, Any]], str]:
    summary_path = reference_root / "summary.json"
    summary = read_json(summary_path)
    by_frame: dict[int, dict[str, Any]] = {}
    for row in summary.get("records", []):
        frame_id = int(row["frame_id"])
        if frame_id in frame_ids:
            item = dict(row)
            item["label_path"] = resolve_path(str(row["label_path"]), reference_root)
            by_frame[frame_id] = item
    missing = [fid for fid in frame_ids if fid not in by_frame]
    if missing:
        raise RuntimeError({"missing_reference_records": missing[:10], "missing_count": len(missing)})
    return by_frame, sha256_file(summary_path)


def run_lingbot_geometry(args: argparse.Namespace, frame_ids: list[int], output_root: Path) -> tuple[Path, dict[str, Any]]:
    _ensure_repo_path()
    from tools import build_v105_phase3_lingbot_stream_contract as lingbot_contract

    rgb_root = Path(args.rgb_root)
    if not rgb_root.is_absolute():
        rgb_root = ROOT / rgb_root
    image_paths = lingbot_contract._load_image_paths(rgb_root, args.scene_id, frame_ids)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    load_start = time.time()
    images = lingbot_contract._load_images(image_paths, int(args.image_size), int(args.patch_size))
    image_load_runtime_sec = float(time.time() - load_start)
    image_shape = tuple(int(v) for v in images.shape[-2:])
    model, model_info = lingbot_contract._build_model(args, device)
    predictions, forward_runtime_sec, peak_memory_bytes, dtype = lingbot_contract._run_streaming(model, images, args, device)

    depth = predictions["depth"].float().cpu().numpy().squeeze(0)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth_conf_tensor = predictions.get("depth_conf")
    if isinstance(depth_conf_tensor, torch.Tensor):
        depth_conf = depth_conf_tensor.float().cpu().numpy().squeeze(0)
    else:
        depth_conf = np.ones_like(depth, dtype=np.float32)
    images_np = predictions["images"].float().cpu().numpy().squeeze(0)
    poses_direct, intrinsics = lingbot_contract._decode_pose(predictions, image_shape)
    poses_inverted = np.linalg.inv(poses_direct.astype(np.float64)).astype(np.float32)
    pose_delta = np.linalg.norm(poses_direct[:, :3, 3] - poses_direct[0, :3, 3], axis=1)

    output_root.mkdir(parents=True, exist_ok=True)
    npz_path = output_root / "lingbot_raw_geometry_outputs.npz"
    np.savez_compressed(
        npz_path,
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        depth=depth.astype(np.float32),
        depth_conf=depth_conf.astype(np.float32),
        images=images_np.astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        poses_direct=poses_direct.astype(np.float32),
        poses_inverted=poses_inverted.astype(np.float32),
    )
    summary = {
        "schema_version": "stream4d_v107_lingbot_raw_geometry_outputs_v1",
        "created_unix_time": time.time(),
        "geometry_source": "LingBot-Map",
        "scene_id": args.scene_id,
        "frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "dtype": str(dtype),
        "image_shape": list(image_shape),
        "image_size": int(args.image_size),
        "patch_size": int(args.patch_size),
        "image_load_runtime_sec": image_load_runtime_sec,
        "forward_runtime_sec": forward_runtime_sec,
        "peak_memory_bytes": int(peak_memory_bytes),
        "model_info": model_info,
        "output_shape_summary": lingbot_contract._tensor_shape_summary(predictions),
        "pose_translation_delta_from_first": {
            "min": float(np.min(pose_delta)) if pose_delta.size else 0.0,
            "max": float(np.max(pose_delta)) if pose_delta.size else 0.0,
            "last": float(pose_delta[-1]) if pose_delta.size else 0.0,
        },
        "npz_path": rel(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "note": "Raw depth/depth_conf/pose/intrinsics/images are decoded from LingBot-Map outputs; no ScanNet pose/depth is used for projection.",
    }
    write_json(output_root / "lingbot_raw_geometry_summary.json", summary)
    return npz_path, summary


def load_lingbot_geometry(npz_path: Path) -> dict[str, np.ndarray]:
    payload = np.load(npz_path)
    required = {"frame_ids", "depth", "depth_conf", "images", "intrinsics", "poses_direct", "poses_inverted"}
    missing = sorted(required - set(payload.files))
    if missing:
        raise RuntimeError({"missing_lingbot_npz_keys": missing, "npz": str(npz_path)})
    return {key: np.asarray(payload[key]) for key in payload.files}


def backproject_pixel(
    *,
    xy: tuple[int, int],
    depth: np.ndarray,
    depth_conf: np.ndarray,
    intrinsic: np.ndarray,
    pose_c2w: np.ndarray,
    min_depth_conf: float,
) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    y, x = int(xy[0]), int(xy[1])
    h, w = depth.shape[:2]
    meta: dict[str, Any] = {"source_x": x, "source_y": y}
    if x < 0 or y < 0 or x >= w or y >= h:
        return None, "source_offscreen", meta
    z = float(depth[y, x])
    conf = float(depth_conf[y, x]) if depth_conf.shape == depth.shape else 1.0
    meta.update({"source_depth": z, "source_depth_conf": conf})
    if not math.isfinite(z) or z <= 0.0:
        return None, "source_depth_invalid", meta
    if not math.isfinite(conf) or conf < float(min_depth_conf):
        return None, "source_depth_conf_low", meta
    fx, fy, cx, cy = float(intrinsic[0, 0]), float(intrinsic[1, 1]), float(intrinsic[0, 2]), float(intrinsic[1, 2])
    point_cam = np.asarray([(x - cx) * z / fx, (y - cy) * z / fy, z, 1.0], dtype=np.float64)
    point_world = pose_c2w.astype(np.float64) @ point_cam
    return point_world[:3].astype(np.float32), "ok", meta


def project_world(
    *,
    point_world: np.ndarray,
    target_depth: np.ndarray,
    target_conf: np.ndarray,
    intrinsic: np.ndarray,
    pose_c2w: np.ndarray,
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
    min_depth_conf: float,
) -> tuple[dict[str, Any] | None, str]:
    w2c = np.linalg.inv(pose_c2w.astype(np.float64))
    point_cam = w2c @ np.asarray([float(point_world[0]), float(point_world[1]), float(point_world[2]), 1.0])
    x_cam, y_cam, z_cam = float(point_cam[0]), float(point_cam[1]), float(point_cam[2])
    if not math.isfinite(z_cam) or z_cam <= 0.0:
        return None, "target_behind_camera"
    fx, fy, cx, cy = float(intrinsic[0, 0]), float(intrinsic[1, 1]), float(intrinsic[0, 2]), float(intrinsic[1, 2])
    x = fx * x_cam / z_cam + cx
    y = fy * y_cam / z_cam + cy
    h, w = target_depth.shape[:2]
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return None, "target_offscreen"
    observed = float(target_depth[yi, xi])
    conf = float(target_conf[yi, xi]) if target_conf.shape == target_depth.shape else 1.0
    if not math.isfinite(observed) or observed <= 0.0:
        return None, "target_depth_invalid"
    if not math.isfinite(conf) or conf < float(min_depth_conf):
        return None, "target_depth_conf_low"
    tolerance = max(float(depth_abs_tolerance), abs(float(z_cam)) * float(depth_rel_tolerance))
    err = abs(observed - z_cam)
    if err > tolerance:
        return None, "target_occluded_or_depth_mismatch"
    return (
        {
            "target_x": float(x),
            "target_y": float(y),
            "target_depth_x": int(xi),
            "target_depth_y": int(yi),
            "projected_depth": float(z_cam),
            "observed_depth": float(observed),
            "target_depth_conf": float(conf),
            "depth_abs_error": float(err),
            "depth_tolerance": float(tolerance),
        },
        "ok",
    )


def visibility_project(
    *,
    source_xy: tuple[int, int],
    source_index: int,
    target_index: int,
    geometry: dict[str, np.ndarray],
    pose_c2w: np.ndarray,
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
    min_depth_conf: float,
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    point_world, source_status, source_meta = backproject_pixel(
        xy=source_xy,
        depth=geometry["depth"][source_index],
        depth_conf=geometry["depth_conf"][source_index],
        intrinsic=geometry["intrinsics"][source_index],
        pose_c2w=pose_c2w[source_index],
        min_depth_conf=float(min_depth_conf),
    )
    if point_world is None:
        return None, source_status, source_meta
    projected, target_status = project_world(
        point_world=point_world,
        target_depth=geometry["depth"][target_index],
        target_conf=geometry["depth_conf"][target_index],
        intrinsic=geometry["intrinsics"][target_index],
        pose_c2w=pose_c2w[target_index],
        depth_abs_tolerance=float(depth_abs_tolerance),
        depth_rel_tolerance=float(depth_rel_tolerance),
        min_depth_conf=float(min_depth_conf),
    )
    if projected is None:
        return None, target_status, source_meta
    projected.update(source_meta)
    return projected, "ok", source_meta


def color_for_obj(obj_id: int) -> tuple[int, int, int]:
    seed = (int(obj_id) * 2654435761) & 0xFFFFFFFF
    return (
        int(70 + ((seed >> 0) & 0x7F)),
        int(70 + ((seed >> 8) & 0x7F)),
        int(70 + ((seed >> 16) & 0x7F)),
    )


def overlay_mask(rgb: np.ndarray, label: np.ndarray, obj_id: int) -> np.ndarray:
    out = rgb.copy()
    mask = label == int(obj_id)
    if np.count_nonzero(mask) == 0:
        return out
    color = np.asarray(color_for_obj(obj_id), dtype=np.float32)
    out[mask] = (0.55 * out[mask].astype(np.float32) + 0.45 * color).clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 0), 1, lineType=cv2.LINE_AA)
    return out


def draw_points(rgb: np.ndarray, points: list[dict[str, Any]]) -> np.ndarray:
    out = rgb.copy()
    for point in points:
        x = int(round(float(point["target_x"])))
        y = int(round(float(point["target_y"])))
        role = str(point.get("role", ""))
        if role == "positive":
            color = (30, 230, 60)
            marker = "+"
        else:
            color = (235, 55, 45)
            marker = "x"
        cv2.circle(out, (x, y), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(out, marker, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return out


def put_label(rgb: np.ndarray, text: str) -> np.ndarray:
    h, w = rgb.shape[:2]
    header = 26
    out = np.zeros((h + header, w, 3), dtype=np.uint8)
    out[:] = 12
    out[header:, :, :] = rgb
    cv2.putText(out, text[:120], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def contact_sheet(paths: list[Path], out_path: Path, cols: int = 3, pad: int = 6) -> Path | None:
    images = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not images:
        return None
    h, w = images[0].shape[:2]
    rows = int(math.ceil(len(images) / float(cols)))
    canvas = np.zeros((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, 3), dtype=np.uint8)
    canvas[:] = 24
    for idx, image in enumerate(images):
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        y = (idx // cols) * (h + pad)
        x = (idx % cols) * (w + pad)
        canvas[y : y + h, x : x + w] = image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return out_path


def compute_prompt_cases(
    *,
    args: argparse.Namespace,
    geometry: dict[str, np.ndarray],
    reference_records: dict[int, dict[str, Any]],
    pose_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, np.ndarray]]:
    frame_ids = [int(v) for v in geometry["frame_ids"].tolist()]
    shape_hw = tuple(int(v) for v in geometry["depth"].shape[1:3])
    pose_c2w = geometry["poses_direct"] if pose_mode == "direct_as_c2w" else geometry["poses_inverted"]
    requested_lags = parse_source_lags(str(args.source_lags))

    labels_by_index: dict[int, np.ndarray] = {}
    for idx, frame_id in enumerate(frame_ids):
        label = load_label(Path(reference_records[int(frame_id)]["label_path"]))
        labels_by_index[idx] = resize_label_to_shape(label, shape_hw)

    rows: list[dict[str, Any]] = []
    point_records: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    for target_idx in range(1, len(frame_ids)):
        target_label = labels_by_index[target_idx]
        target_ids = [int(v) for v in np.unique(target_label).tolist() if int(v) > 0]
        target_ids = sorted(target_ids, key=lambda obj_id: int(np.count_nonzero(target_label == obj_id)), reverse=True)
        for obj_id in target_ids[: int(args.max_objects_per_frame)]:
            source_indices: list[int] = []
            if requested_lags:
                for lag in requested_lags:
                    candidate_idx = target_idx - int(lag)
                    if candidate_idx >= 0 and int(np.count_nonzero(labels_by_index[candidate_idx] == int(obj_id))) > 0:
                        source_indices.append(int(candidate_idx))
            else:
                for candidate_idx in range(target_idx - 1, max(-1, target_idx - int(args.history_window) - 1), -1):
                    if int(np.count_nonzero(labels_by_index[candidate_idx] == int(obj_id))) > 0:
                        source_indices.append(int(candidate_idx))
                        break
            if not source_indices:
                continue

            for source_idx in source_indices:
                source_label = labels_by_index[source_idx]
                obj_mask = source_label == int(obj_id)
                obj_bbox = bbox_from_mask(obj_mask)
                if obj_bbox is None:
                    continue
                pos_source_mask = eroded_core(obj_mask, int(args.core_erode_iters))
                pos_points = sample_mask_points_spread(
                    pos_source_mask,
                    count=int(args.positive_points_per_object),
                    seed=int(args.seed) + target_idx * 1009 + source_idx * 37 + obj_id,
                )

                neighbor_items: list[tuple[float, int, np.ndarray]] = []
                for other_id in [int(v) for v in np.unique(source_label).tolist() if int(v) > 0 and int(v) != int(obj_id)]:
                    other_mask = source_label == int(other_id)
                    other_bbox = bbox_from_mask(other_mask)
                    if other_bbox is None:
                        continue
                    dist = bbox_distance(obj_bbox, other_bbox)
                    if dist <= float(args.neighbor_radius_px):
                        neighbor_items.append((float(dist), int(other_id), eroded_core(other_mask, int(args.core_erode_iters))))
                neighbor_items.sort(key=lambda item: (item[0], item[1]))

                neg_points: list[tuple[int, int, int, float]] = []
                if neighbor_items:
                    per_neighbor = max(1, int(math.ceil(int(args.negative_points_per_object) / len(neighbor_items))))
                    for dist, other_id, other_mask in neighbor_items:
                        sampled = sample_mask_points_spread(
                            other_mask,
                            count=per_neighbor,
                            seed=int(args.seed) + target_idx * 917 + source_idx * 41 + obj_id * 13 + other_id,
                        )
                        for y, x in sampled:
                            neg_points.append((int(y), int(x), int(other_id), float(dist)))
                            if len(neg_points) >= int(args.negative_points_per_object):
                                break
                        if len(neg_points) >= int(args.negative_points_per_object):
                            break

                visible_pos: list[dict[str, Any]] = []
                visible_neg: list[dict[str, Any]] = []
                local_rejections: dict[str, int] = {}
                for point_idx, source_xy in enumerate(pos_points):
                    projected, status, _meta = visibility_project(
                        source_xy=source_xy,
                        source_index=int(source_idx),
                        target_index=int(target_idx),
                        geometry=geometry,
                        pose_c2w=pose_c2w,
                        depth_abs_tolerance=float(args.depth_abs_tolerance),
                        depth_rel_tolerance=float(args.depth_rel_tolerance),
                        min_depth_conf=float(args.min_depth_conf),
                    )
                    if projected is None:
                        rejection_counts[status] = rejection_counts.get(status, 0) + 1
                        local_rejections[status] = local_rejections.get(status, 0) + 1
                        continue
                    tx = int(round(float(projected["target_x"])))
                    ty = int(round(float(projected["target_y"])))
                    target_label_at_point = int(target_label[ty, tx]) if 0 <= ty < target_label.shape[0] and 0 <= tx < target_label.shape[1] else 0
                    projected.update(
                        {
                            "role": "positive",
                            "pose_mode": pose_mode,
                            "source_lag": int(target_idx - source_idx),
                            "source_obj_id": int(obj_id),
                            "target_obj_id": int(obj_id),
                            "target_label_at_point": int(target_label_at_point),
                            "reference_hit_target_obj": bool(target_label_at_point == int(obj_id)),
                            "source_frame_index": int(source_idx),
                            "source_frame_id": int(frame_ids[source_idx]),
                            "target_frame_index": int(target_idx),
                            "target_frame_id": int(frame_ids[target_idx]),
                            "point_index": int(point_idx),
                        }
                    )
                    visible_pos.append(projected)

                for point_idx, (y, x, other_id, neighbor_dist) in enumerate(neg_points):
                    projected, status, _meta = visibility_project(
                        source_xy=(int(y), int(x)),
                        source_index=int(source_idx),
                        target_index=int(target_idx),
                        geometry=geometry,
                        pose_c2w=pose_c2w,
                        depth_abs_tolerance=float(args.depth_abs_tolerance),
                        depth_rel_tolerance=float(args.depth_rel_tolerance),
                        min_depth_conf=float(args.min_depth_conf),
                    )
                    if projected is None:
                        rejection_counts[status] = rejection_counts.get(status, 0) + 1
                        local_rejections[status] = local_rejections.get(status, 0) + 1
                        continue
                    tx = int(round(float(projected["target_x"])))
                    ty = int(round(float(projected["target_y"])))
                    target_label_at_point = int(target_label[ty, tx]) if 0 <= ty < target_label.shape[0] and 0 <= tx < target_label.shape[1] else 0
                    projected.update(
                        {
                            "role": "negative",
                            "pose_mode": pose_mode,
                            "source_lag": int(target_idx - source_idx),
                            "source_obj_id": int(other_id),
                            "target_obj_id": int(obj_id),
                            "target_label_at_point": int(target_label_at_point),
                            "reference_hits_target_obj": bool(target_label_at_point == int(obj_id)),
                            "reference_hits_source_sibling": bool(target_label_at_point == int(other_id)),
                            "source_frame_index": int(source_idx),
                            "source_frame_id": int(frame_ids[source_idx]),
                            "target_frame_index": int(target_idx),
                            "target_frame_id": int(frame_ids[target_idx]),
                            "point_index": int(point_idx),
                            "cannot_link_reason": "nearby_co_visible_sibling",
                            "source_bbox_distance_px": float(neighbor_dist),
                        }
                    )
                    visible_neg.append(projected)

                pos_hits = int(sum(1 for point in visible_pos if point["reference_hit_target_obj"]))
                neg_hits_target = int(sum(1 for point in visible_neg if point["reference_hits_target_obj"]))
                neg_hits_sibling = int(sum(1 for point in visible_neg if point["reference_hits_source_sibling"]))
                rows.append(
                    {
                        "pose_mode": pose_mode,
                        "source_lag": int(target_idx - source_idx),
                        "target_frame_index": int(target_idx),
                        "target_frame_id": int(frame_ids[target_idx]),
                        "target_obj_id": int(obj_id),
                        "source_frame_index": int(source_idx),
                        "source_frame_id": int(frame_ids[source_idx]),
                        "source_obj_area_px_lingbot_res": int(np.count_nonzero(obj_mask)),
                        "source_neighbor_count": int(len(neighbor_items)),
                        "positive_sampled_count": int(len(pos_points)),
                        "positive_visible_count": int(len(visible_pos)),
                        "positive_reference_hit_count": int(pos_hits),
                        "negative_sampled_count": int(len(neg_points)),
                        "negative_visible_count": int(len(visible_neg)),
                        "negative_reference_hits_target_obj_count": int(neg_hits_target),
                        "negative_reference_hits_source_sibling_count": int(neg_hits_sibling),
                        "usable_positive_negative_prompt": bool(visible_pos and visible_neg),
                        "local_rejection_counts": json.dumps(local_rejections, sort_keys=True),
                    }
                )
                point_records.extend(visible_pos[: int(args.max_points_recorded_per_case)])
                point_records.extend(visible_neg[: int(args.max_points_recorded_per_case)])

    return rows, point_records, rejection_counts, labels_by_index


def summarize_rows(rows: list[dict[str, Any]], rejection_counts: dict[str, int]) -> dict[str, Any]:
    case_count = len(rows)
    usable = int(sum(1 for row in rows if row["usable_positive_negative_prompt"]))
    pos_sampled = int(sum(int(row["positive_sampled_count"]) for row in rows))
    pos_visible = int(sum(int(row["positive_visible_count"]) for row in rows))
    pos_hits = int(sum(int(row["positive_reference_hit_count"]) for row in rows))
    neg_sampled = int(sum(int(row["negative_sampled_count"]) for row in rows))
    neg_visible = int(sum(int(row["negative_visible_count"]) for row in rows))
    neg_target_hits = int(sum(int(row["negative_reference_hits_target_obj_count"]) for row in rows))
    neg_sibling_hits = int(sum(int(row["negative_reference_hits_source_sibling_count"]) for row in rows))
    return {
        "case_count": int(case_count),
        "usable_positive_negative_prompt_case_count": int(usable),
        "usable_positive_negative_prompt_case_rate": float(usable / max(case_count, 1)),
        "positive_sampled_count": int(pos_sampled),
        "positive_visible_count": int(pos_visible),
        "positive_visible_rate": float(pos_visible / max(pos_sampled, 1)),
        "positive_reference_hit_count": int(pos_hits),
        "positive_reference_hit_rate_among_visible": float(pos_hits / max(pos_visible, 1)),
        "negative_sampled_count": int(neg_sampled),
        "negative_visible_count": int(neg_visible),
        "negative_visible_rate": float(neg_visible / max(neg_sampled, 1)),
        "negative_reference_hits_target_obj_count": int(neg_target_hits),
        "negative_reference_hits_target_obj_rate_among_visible": float(neg_target_hits / max(neg_visible, 1)),
        "negative_reference_hits_source_sibling_count": int(neg_sibling_hits),
        "negative_reference_hits_source_sibling_rate_among_visible": float(neg_sibling_hits / max(neg_visible, 1)),
        "rejection_counts": {str(k): int(v) for k, v in sorted(rejection_counts.items())},
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pose_mode",
        "source_lag",
        "target_frame_index",
        "target_frame_id",
        "target_obj_id",
        "source_frame_index",
        "source_frame_id",
        "source_obj_area_px_lingbot_res",
        "source_neighbor_count",
        "positive_sampled_count",
        "positive_visible_count",
        "positive_reference_hit_count",
        "negative_sampled_count",
        "negative_visible_count",
        "negative_reference_hits_target_obj_count",
        "negative_reference_hits_source_sibling_count",
        "usable_positive_negative_prompt",
        "local_rejection_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_visualizations(
    *,
    args: argparse.Namespace,
    geometry: dict[str, np.ndarray],
    labels_by_index: dict[int, np.ndarray],
    point_records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    output_root: Path,
    pose_mode: str,
) -> tuple[list[Path], Path | None]:
    frame_ids = [int(v) for v in geometry["frame_ids"].tolist()]
    vis_root = output_root / "visual_overlays" / pose_mode
    vis_root.mkdir(parents=True, exist_ok=True)
    by_case: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for point in point_records:
        key = (int(point["target_frame_index"]), int(point["target_obj_id"]), int(point["source_frame_index"]))
        by_case.setdefault(key, []).append(point)

    candidate_rows = [
        row
        for row in rows
        if int(row["positive_visible_count"]) > 0 and int(row["negative_visible_count"]) > 0
    ]
    candidate_rows.sort(
        key=lambda row: (
            -int(row["positive_reference_hit_count"]),
            int(row["negative_reference_hits_target_obj_count"]),
            -int(row["negative_visible_count"]),
        )
    )
    out_paths: list[Path] = []
    for row in candidate_rows[: int(args.max_visual_cases)]:
        target_idx = int(row["target_frame_index"])
        obj_id = int(row["target_obj_id"])
        source_idx = int(row["source_frame_index"])
        points = by_case.get((target_idx, obj_id, source_idx), [])
        rgb = (geometry["images"][target_idx].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        over = overlay_mask(rgb, labels_by_index[target_idx], obj_id)
        over = draw_points(over, points)
        title = (
            f"{pose_mode} src={frame_ids[source_idx]} tgt={frame_ids[target_idx]} "
            f"lag={int(row.get('source_lag', target_idx - source_idx))} "
            f"obj={obj_id} pos={row['positive_visible_count']} neg={row['negative_visible_count']}"
        )
        over = put_label(over, title)
        out_path = vis_root / f"target{target_idx:03d}_frame{frame_ids[target_idx]:06d}_obj{obj_id:04d}_src{source_idx:03d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
        out_paths.append(out_path)
    sheet_path = contact_sheet(out_paths, vis_root / "prompt_projection_contact_sheet.jpg", cols=3)
    return out_paths, sheet_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run-root", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--rgb-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--frame-start", type=int, default=4450)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--geometry-npz", default="")
    parser.add_argument("--checkpoint", default="third_party/lingbot-map/checkpoints/lingbot-map-long.pt")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--max-frame-num", type=int, default=64)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=32)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--dtype", choices=["auto", "bf16", "float16", "float32"], default="bf16")
    parser.add_argument("--use-sdpa", action="store_true", default=True)
    parser.add_argument("--disable-3d-rope", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--history-window", type=int, default=6)
    parser.add_argument("--source-lags", default="", help="Comma-separated fixed source lags, e.g. 1,2,4,8,16. Empty uses nearest history within --history-window.")
    parser.add_argument("--positive-points-per-object", type=int, default=8)
    parser.add_argument("--negative-points-per-object", type=int, default=8)
    parser.add_argument("--max-objects-per-frame", type=int, default=12)
    parser.add_argument("--neighbor-radius-px", type=float, default=52.0)
    parser.add_argument("--core-erode-iters", type=int, default=2)
    parser.add_argument("--depth-abs-tolerance", type=float, default=0.12)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.08)
    parser.add_argument("--min-depth-conf", type=float, default=0.0)
    parser.add_argument("--pose-mode", choices=["auto", "direct_as_c2w", "inverted_as_c2w"], default="auto")
    parser.add_argument("--max-points-recorded-per-case", type=int, default=8)
    parser.add_argument("--max-point-records", type=int, default=4096)
    parser.add_argument("--max-visual-cases", type=int, default=12)
    parser.add_argument("--seed", type=int, default=107)
    return parser


def main() -> int:
    started = time.time()
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    frame_ids = parse_frame_ids(args.frame_ids, int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    reference_root = Path(args.reference_run_root)
    if not reference_root.is_absolute():
        reference_root = ROOT / reference_root
    reference_records, reference_summary_sha = load_reference_records(reference_root, frame_ids)

    if args.geometry_npz.strip():
        npz_path = Path(args.geometry_npz)
        if not npz_path.is_absolute():
            npz_path = ROOT / npz_path
        raw_summary = {
            "geometry_source": "LingBot-Map",
            "npz_path": rel(npz_path),
            "npz_sha256": sha256_file(npz_path),
            "loaded_existing_geometry_npz": True,
        }
    else:
        npz_path, raw_summary = run_lingbot_geometry(args, frame_ids, output_root)

    geometry = load_lingbot_geometry(npz_path)
    npz_frame_ids = [int(v) for v in geometry["frame_ids"].tolist()]
    if npz_frame_ids != frame_ids:
        raise RuntimeError({"frame_id_mismatch": {"requested": frame_ids, "geometry_npz": npz_frame_ids}})

    pose_modes = ["direct_as_c2w", "inverted_as_c2w"] if args.pose_mode == "auto" else [str(args.pose_mode)]
    mode_outputs: dict[str, Any] = {}
    selected_mode = pose_modes[0]
    selected_score = -1.0
    selected_rows: list[dict[str, Any]] = []
    selected_points: list[dict[str, Any]] = []
    selected_labels: dict[str, np.ndarray] | dict[int, np.ndarray] = {}

    for pose_mode in pose_modes:
        rows, point_records, rejection_counts, labels_by_index = compute_prompt_cases(
            args=args,
            geometry=geometry,
            reference_records=reference_records,
            pose_mode=pose_mode,
        )
        if len(point_records) > int(args.max_point_records):
            point_records = point_records[: int(args.max_point_records)]
        stats = summarize_rows(rows, rejection_counts)
        rows_csv = output_root / f"prompt_capsule_visibility_rows_{pose_mode}.csv"
        points_json = output_root / f"prompt_capsule_visible_point_records_{pose_mode}.json"
        write_rows_csv(rows_csv, rows)
        write_json(points_json, {"row_count": len(point_records), "rows": point_records})
        visual_paths, sheet_path = write_visualizations(
            args=args,
            geometry=geometry,
            labels_by_index=labels_by_index,
            point_records=point_records,
            rows=rows,
            output_root=output_root,
            pose_mode=pose_mode,
        )
        score = float(stats["positive_reference_hit_rate_among_visible"]) - float(
            stats["negative_reference_hits_target_obj_rate_among_visible"]
        )
        mode_outputs[pose_mode] = {
            **stats,
            "selection_score_pos_hit_minus_neg_target_hit": score,
            "rows_csv": rel(rows_csv),
            "rows_csv_sha256": sha256_file(rows_csv),
            "visible_point_records": rel(points_json),
            "visible_point_records_sha256": sha256_file(points_json),
            "visual_overlay_count": len(visual_paths),
            "visual_overlays": [rel(path) for path in visual_paths],
            "visual_contact_sheet": rel(sheet_path) if sheet_path is not None else "",
            "visual_contact_sheet_sha256": sha256_file(sheet_path) if sheet_path is not None else "",
        }
        if score > selected_score:
            selected_score = score
            selected_mode = pose_mode
            selected_rows = rows
            selected_points = point_records
            selected_labels = labels_by_index

    selected_rows_csv = output_root / "prompt_capsule_visibility_rows.csv"
    selected_points_json = output_root / "prompt_capsule_visible_point_records.json"
    write_rows_csv(selected_rows_csv, selected_rows)
    write_json(selected_points_json, {"row_count": len(selected_points), "rows": selected_points})
    selected_visual_paths, selected_sheet = write_visualizations(
        args=args,
        geometry=geometry,
        labels_by_index=selected_labels,  # type: ignore[arg-type]
        point_records=selected_points,
        rows=selected_rows,
        output_root=output_root,
        pose_mode=f"selected_{selected_mode}",
    )

    summary = {
        "schema_version": "stream4d_v107_phase5_lingbot_prompt_capsule_visibility_probe_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "scene_id": args.scene_id,
        "frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "reference_run_root": rel(reference_root),
        "reference_summary_sha256": reference_summary_sha,
        "uses_reference_method_labels": True,
        "uses_gt_instance_labels": False,
        "projection_geometry_source": "LingBot-Map decoded pose_enc + depth + depth_conf + intrinsics",
        "uses_scannet_pose_or_depth_for_projection": False,
        "raw_lingbot_geometry": raw_summary,
        "config": {
            "history_window": int(args.history_window),
            "source_lags": parse_source_lags(str(args.source_lags)),
            "positive_points_per_object": int(args.positive_points_per_object),
            "negative_points_per_object": int(args.negative_points_per_object),
            "max_objects_per_frame": int(args.max_objects_per_frame),
            "neighbor_radius_px_lingbot_resolution": float(args.neighbor_radius_px),
            "core_erode_iters": int(args.core_erode_iters),
            "depth_abs_tolerance": float(args.depth_abs_tolerance),
            "depth_rel_tolerance": float(args.depth_rel_tolerance),
            "min_depth_conf": float(args.min_depth_conf),
            "seed": int(args.seed),
        },
        "pose_mode_requested": args.pose_mode,
        "selected_pose_mode": selected_mode,
        "pose_mode_outputs": mode_outputs,
        "selected_rows_csv": rel(selected_rows_csv),
        "selected_rows_csv_sha256": sha256_file(selected_rows_csv),
        "selected_visible_point_records": rel(selected_points_json),
        "selected_visible_point_records_sha256": sha256_file(selected_points_json),
        "selected_visual_overlay_count": len(selected_visual_paths),
        "selected_visual_overlays": [rel(path) for path in selected_visual_paths],
        "selected_visual_contact_sheet": rel(selected_sheet) if selected_sheet is not None else "",
        "selected_visual_contact_sheet_sha256": sha256_file(selected_sheet) if selected_sheet is not None else "",
        "audit_note": (
            "Rows measure whether LingBot-visible historical positive/negative prompt points can be constructed. "
            "Reference-hit fields use frozen v106 labels only as diagnostic evidence, not as exact parity gates."
        ),
    }
    summary_path = output_root / "prompt_capsule_visibility_probe_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), "selected_pose_mode": selected_mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
