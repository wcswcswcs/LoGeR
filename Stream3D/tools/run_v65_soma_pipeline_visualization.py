from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.v47_common import resolve_mask_dir
from stream4d_native.v65_visualization_export import _id_colors, _load_gt, _load_scene_mesh


REQUIRED_LAYERS = ["GT geo", "GT sem", "D4RT geo", "SOMA sem"]


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == ROOT.name:
        return REPO_ROOT / path_obj
    return ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        try:
            return str(path_obj.relative_to(REPO_ROOT))
        except ValueError:
            return str(path_obj)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_record_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    return []


def _read_support_records(json_path: Path) -> tuple[list[dict[str, Any]], str]:
    if json_path.exists():
        return _read_record_manifest(json_path), "json_manifest"
    raise RuntimeError(f"pipeline support records missing: {json_path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_records_json(path: Path, rows: list[dict[str, Any]], *, schema_version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": schema_version,
        "row_count": len(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_mask_observation_id(value: str) -> tuple[str, int, int] | None:
    parts = str(value or "").split(":")
    if len(parts) != 3:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _numeric_frame_ids(path: Path, suffix: str) -> list[int]:
    if not path.exists():
        return []
    out: list[int] = []
    for item in path.glob(f"*{suffix}"):
        if item.stem.isdigit():
            out.append(int(item.stem))
    return sorted(set(out))


def _expected_stride_frames(scene: str, stride: int) -> list[int]:
    color_ids = _numeric_frame_ids(ROOT / "data/scannet/processed" / scene / "color", ".jpg")
    color_set = set(color_ids)
    if not color_ids:
        return []
    return [frame for frame in range(min(color_ids), max(color_ids) + 1, int(stride)) if frame in color_set]


def _resolve_pipeline_mask_dir(
    *,
    scene: str,
    pipeline_summary: dict[str, Any],
    override_mask_root: str | Path | None = None,
) -> Path:
    if override_mask_root is not None and str(override_mask_root).strip():
        return resolve_mask_dir(override_mask_root, scene)
    coverage = pipeline_summary.get("mask_frame_coverage")
    if isinstance(coverage, dict):
        mask_dir = str(coverage.get("mask_dir") or "").strip()
        if mask_dir:
            return _project(mask_dir)
    mask_root = str(pipeline_summary.get("mask_root") or "").strip()
    if mask_root:
        return resolve_mask_dir(mask_root, scene)
    return resolve_mask_dir(None, scene)


def _best_objectlet_variant(pipeline_root: Path, explicit: str) -> str:
    if explicit and explicit != "best":
        return explicit
    summary = _read_json(pipeline_root / "local_objectlets" / "local_objectlet_summary.json")
    variant = str(summary.get("best_real_variant") or "").strip()
    if not variant:
        raise RuntimeError(f"best_real_variant missing in {pipeline_root / 'local_objectlets/local_objectlet_summary.json'}")
    return variant


def _load_pipeline_support(
    *,
    pipeline_root: Path,
    scene: str,
    objectlet_variant: str,
    success_only: bool,
) -> tuple[dict[int, list[tuple[int, int, str]]], dict[tuple[int, int], int], dict[str, Any]]:
    objectlet_json_path = pipeline_root / "local_objectlets" / "objectlet_records.json"
    ledger_json_path = pipeline_root / "reprojection_ledger" / "reprojection_ledger_records.json"
    objectlet_records, objectlet_record_format = _read_support_records(objectlet_json_path)
    ledger_records, ledger_record_format = _read_support_records(ledger_json_path)
    selected_by_candidate: dict[str, tuple[str, int]] = {}
    object_to_idx: dict[str, int] = {}
    selected_rows = 0
    for row in objectlet_records:
        if row.get("scene") != scene or row.get("variant") != objectlet_variant:
            continue
        object_id = str(row.get("objectlet_id") or "").strip()
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not object_id or not candidate_id:
            continue
        if object_id not in object_to_idx:
            object_to_idx[object_id] = len(object_to_idx) + 1
        selected_by_candidate[candidate_id] = (object_id, object_to_idx[object_id])
        selected_rows += 1

    by_frame: dict[int, set[tuple[int, int, str]]] = defaultdict(set)
    mask_to_object_idx: dict[tuple[int, int], int] = {}
    duplicate_frame_mask_conflicts = 0
    ledger_rows = 0
    used_ledger_rows = 0
    skipped_failed_rows = 0
    for row in ledger_records:
        ledger_rows += 1
        selected = selected_by_candidate.get(str(row.get("candidate_id") or ""))
        if not selected:
            continue
        if success_only and not _parse_bool(row.get("reprojection_success")):
            skipped_failed_rows += 1
            continue
        parsed = _parse_mask_observation_id(str(row.get("best_mask_observation_id") or ""))
        if parsed is None:
            continue
        row_scene, frame_id, mask_id = parsed
        if row_scene != scene or mask_id <= 0:
            continue
        object_id, object_idx = selected
        key = (frame_id, mask_id)
        if key in mask_to_object_idx and mask_to_object_idx[key] != object_idx:
            duplicate_frame_mask_conflicts += 1
            object_idx = min(mask_to_object_idx[key], object_idx)
        mask_to_object_idx[key] = object_idx
        by_frame[frame_id].add((object_idx, mask_id, object_id))
        used_ledger_rows += 1

    support_by_frame = {frame: sorted(items) for frame, items in by_frame.items()}
    diag = {
        "objectlet_variant": objectlet_variant,
        "objectlet_record_format": objectlet_record_format,
        "ledger_record_format": ledger_record_format,
        "objectlet_row_count": int(selected_rows),
        "object_count": int(len(object_to_idx)),
        "ledger_row_count": int(ledger_rows),
        "used_ledger_row_count": int(used_ledger_rows),
        "skipped_failed_ledger_row_count": int(skipped_failed_rows),
        "support_frame_count": int(len(support_by_frame)),
        "support_pair_count": int(sum(len(values) for values in support_by_frame.values())),
        "duplicate_frame_mask_conflicts": int(duplicate_frame_mask_conflicts),
        "support_contract": "pipeline local_objectlets selected candidate_id joined to same-root reprojection_ledger best_mask_observation_id",
    }
    return support_by_frame, mask_to_object_idx, diag


def _overlay(rgb: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    if labels.shape[:2] != rgb.shape[:2]:
        labels = cv2.resize(labels, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    positive = labels > 0
    colors = np.zeros((*labels.shape[:2], 3), dtype=np.uint8)
    ids = np.unique(labels[positive])
    for value in ids:
        colors[labels == int(value)] = _id_colors(np.asarray([int(value)], dtype=np.int64))[0]
    out = rgb.copy()
    out[positive] = (
        (1.0 - float(alpha)) * out[positive].astype(np.float32) + float(alpha) * colors[positive].astype(np.float32)
    ).astype(np.uint8)
    return out


def _soma_object_color_rows(support_by_frame: dict[int, list[tuple[int, int, str]]]) -> list[dict[str, Any]]:
    object_id_by_idx: dict[int, str] = {}
    support_pair_count: Counter[int] = Counter()
    frames_by_object: dict[int, set[int]] = defaultdict(set)
    for frame_id, items in support_by_frame.items():
        for object_idx, _mask_id, object_id in items:
            object_idx = int(object_idx)
            object_id_by_idx.setdefault(object_idx, str(object_id))
            support_pair_count[object_idx] += 1
            frames_by_object.setdefault(object_idx, set()).add(int(frame_id))
    rows: list[dict[str, Any]] = []
    for object_idx in sorted(object_id_by_idx):
        color = _id_colors(np.asarray([object_idx], dtype=np.int64))[0]
        rows.append(
            {
                "object_idx": int(object_idx),
                "object_id": object_id_by_idx[object_idx],
                "color_r": int(color[0]),
                "color_g": int(color[1]),
                "color_b": int(color[2]),
                "support_pair_count": int(support_pair_count[object_idx]),
                "support_frame_count": int(len(frames_by_object.get(object_idx, set()))),
                "color_policy": "stable _id_colors(object_idx); same object_idx has same RGB across all frames",
            }
        )
    return rows


def _label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _writer(path: Path, shape: tuple[int, int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(shape[1]), int(shape[0])))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {path}")
    return writer


def _imread_if_exists(path: Path, flags: int) -> np.ndarray | None:
    if not path.exists():
        return None
    return cv2.imread(str(path), flags)


def _resize(image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if image.shape[:2] == shape[:2]:
        return image
    return cv2.resize(image, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_AREA)


def export_2d_videos(
    *,
    scene: str,
    stride: int,
    mask_dir: Path,
    output_root: Path,
    support_by_frame: dict[int, list[tuple[int, int, str]]],
    alpha: float,
    fps: float,
    resize_width: int,
    max_video_frames: int,
) -> dict[str, Any]:
    scene_root = ROOT / "data/scannet/processed" / scene
    rgb_dir = scene_root / "color"
    crop_mask_dir = mask_dir
    gt_instance_dir = scene_root / "instance" / "instance"
    gt_sem_dir = scene_root / "label-filt"
    frames = _expected_stride_frames(scene, stride)
    if max_video_frames > 0:
        frames = frames[: int(max_video_frames)]
    if not frames:
        raise RuntimeError(f"no stride frames found for scene={scene}")

    first_rgb = cv2.imread(str(rgb_dir / f"{frames[0]}.jpg"), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise FileNotFoundError(rgb_dir / f"{frames[0]}.jpg")
    if resize_width > 0 and first_rgb.shape[1] != int(resize_width):
        scale = float(resize_width) / float(first_rgb.shape[1])
        frame_shape = (int(round(first_rgb.shape[0] * scale)), int(resize_width), 3)
    else:
        frame_shape = first_rgb.shape

    paths = {
        "soma_video": output_root / f"{scene}_pipeline_soma_objects_2d.mp4",
        "gt_video": output_root / f"{scene}_pipeline_gt_instances_2d.mp4",
        "gt_sem_video": output_root / f"{scene}_pipeline_gt_sem_2d.mp4",
        "compare_video": output_root / f"{scene}_pipeline_soma_vs_gt_2d.mp4",
        "compare_sem_video": output_root / f"{scene}_pipeline_soma_vs_gt_sem_2d.mp4",
    }
    writers = {
        "soma_video": _writer(paths["soma_video"], frame_shape, fps),
        "gt_video": _writer(paths["gt_video"], frame_shape, fps),
        "gt_sem_video": _writer(paths["gt_sem_video"], frame_shape, fps),
        "compare_video": _writer(paths["compare_video"], (frame_shape[0], frame_shape[1] * 2, 3), fps),
        "compare_sem_video": _writer(paths["compare_sem_video"], (frame_shape[0], frame_shape[1] * 2, 3), fps),
    }

    frame_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    try:
        for frame in frames:
            rgb_bgr = _imread_if_exists(rgb_dir / f"{frame}.jpg", cv2.IMREAD_COLOR)
            crop_mask = _imread_if_exists(crop_mask_dir / f"{frame}.png", cv2.IMREAD_UNCHANGED)
            gt_instance = _imread_if_exists(gt_instance_dir / f"{frame}.png", cv2.IMREAD_UNCHANGED)
            gt_sem = _imread_if_exists(gt_sem_dir / f"{frame}.png", cv2.IMREAD_UNCHANGED)
            if rgb_bgr is None:
                frame_rows.append({"frame_id": frame, "ok": False, "missing_rgb": True})
                counters["missing_rgb"] += 1
                continue
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            if crop_mask is not None and crop_mask.ndim == 3:
                crop_mask = crop_mask[..., 0]
            if gt_instance is not None and gt_instance.ndim == 3:
                gt_instance = gt_instance[..., 0]
            if gt_sem is not None and gt_sem.ndim == 3:
                gt_sem = gt_sem[..., 0]
            soma_labels = np.zeros(rgb.shape[:2], dtype=np.int32)
            support_items = support_by_frame.get(int(frame), [])
            if crop_mask is None:
                counters["missing_soma_mask"] += 1
            else:
                if crop_mask.shape[:2] != rgb.shape[:2]:
                    crop_mask = cv2.resize(crop_mask.astype(np.int32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                for object_idx, mask_id, _object_id in support_items:
                    soma_labels[crop_mask == int(mask_id)] = int(object_idx)
            if gt_instance is None:
                counters["missing_gt_instance"] += 1
                gt_instance = np.zeros(rgb.shape[:2], dtype=np.int32)
            if gt_sem is None:
                counters["missing_gt_sem"] += 1
                gt_sem = np.zeros(rgb.shape[:2], dtype=np.int32)
            soma_rgb = _label(_overlay(rgb, soma_labels, alpha), f"SOMA pipeline | {scene} frame {frame}")
            gt_rgb = _label(_overlay(rgb, gt_instance.astype(np.int32), alpha), f"GT instance | {scene} frame {frame}")
            gt_sem_rgb = _label(_overlay(rgb, gt_sem.astype(np.int32), alpha), f"GT semantic | {scene} frame {frame}")
            soma_rgb = _resize(soma_rgb, frame_shape)
            gt_rgb = _resize(gt_rgb, frame_shape)
            gt_sem_rgb = _resize(gt_sem_rgb, frame_shape)
            writers["soma_video"].write(cv2.cvtColor(soma_rgb, cv2.COLOR_RGB2BGR))
            writers["gt_video"].write(cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))
            writers["gt_sem_video"].write(cv2.cvtColor(gt_sem_rgb, cv2.COLOR_RGB2BGR))
            writers["compare_video"].write(cv2.cvtColor(np.concatenate([soma_rgb, gt_rgb], axis=1), cv2.COLOR_RGB2BGR))
            writers["compare_sem_video"].write(cv2.cvtColor(np.concatenate([soma_rgb, gt_sem_rgb], axis=1), cv2.COLOR_RGB2BGR))
            soma_pixels = int(np.count_nonzero(soma_labels))
            counters["written_frames"] += 1
            counters["frames_with_soma_overlay"] += int(soma_pixels > 0)
            counters["frames_with_support_rows"] += int(bool(support_items))
            counters["soma_overlay_pixels"] += soma_pixels
            frame_rows.append(
                {
                    "frame_id": int(frame),
                    "ok": True,
                    "has_soma_mask_file": crop_mask is not None,
                    "soma_support_pair_count": int(len(support_items)),
                    "soma_overlay_pixel_count": soma_pixels,
                    "gt_instance_positive_pixel_count": int(np.count_nonzero(gt_instance)),
                    "gt_sem_positive_pixel_count": int(np.count_nonzero(gt_sem)),
                }
            )
    finally:
        for writer in writers.values():
            writer.release()

    frame_records_json = output_root / "pipeline_2d_frame_records.json"
    _write_records_json(frame_records_json, frame_rows, schema_version="stream4d_v65_soma_pipeline_2d_frame_record_v1")
    color_rows = _soma_object_color_rows(support_by_frame)
    color_records_json = output_root / "pipeline_2d_soma_object_color_records.json"
    _write_records_json(color_records_json, color_rows, schema_version="stream4d_v65_soma_pipeline_2d_object_color_record_v1")
    color_tuples = [(int(row["color_r"]), int(row["color_g"]), int(row["color_b"])) for row in color_rows]
    unique_color_count = len(set(color_tuples))
    status = {
        "phase": "v65_soma_pipeline_2d_visualization",
        "scene": scene,
        "stride": int(stride),
        "requested_stride_frame_count": int(len(frames)),
        "written_frame_count": int(counters["written_frames"]),
        "frames_with_support_rows": int(counters["frames_with_support_rows"]),
        "frames_with_soma_overlay": int(counters["frames_with_soma_overlay"]),
        "missing_soma_mask_frame_count": int(counters["missing_soma_mask"]),
        "missing_gt_instance_frame_count": int(counters["missing_gt_instance"]),
        "missing_gt_sem_frame_count": int(counters["missing_gt_sem"]),
        "total_soma_overlay_pixels": int(counters["soma_overlay_pixels"]),
        "frame_records_json": _rel(frame_records_json),
        "soma_object_color_records_json": _rel(color_records_json),
        "soma_object_count": int(len(color_rows)),
        "soma_unique_color_count": int(unique_color_count),
        "soma_color_collision_count": int(len(color_rows) - unique_color_count),
        "soma_color_policy": "SOMA 2D object masks use label=object_idx and RGB=_id_colors(object_idx); different object_idx values get independently generated colors.",
        "cropformer_mask_dir": _rel(crop_mask_dir),
        "videos": {name: _rel(path) for name, path in paths.items()},
        "video_sha256": {name: _sha256(path) for name, path in paths.items()},
    }
    _write_json(output_root / "pipeline_2d_video_status.json", status)
    return status


def _stable_seed(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def _sample_indices(count: int, max_count: int, seed: str) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_count <= 0 or count <= max_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(_stable_seed(seed))
    return np.sort(rng.choice(count, size=int(max_count), replace=False).astype(np.int64))


def _sample_rgb_colors(scene: str, frame_ids: np.ndarray, uv: np.ndarray) -> np.ndarray:
    stream = ScanNetStream(seq_name=scene, root=ROOT / "data/scannet/processed")
    colors = np.zeros((frame_ids.shape[0], 3), dtype=np.uint8)
    for frame_id in sorted(set(np.asarray(frame_ids, dtype=np.int64).tolist())):
        sel = frame_ids == int(frame_id)
        rgb = stream.load_rgb(int(frame_id))
        h, w = rgb.shape[:2]
        xy = np.rint(
            np.stack([uv[sel, 0] * float(max(w - 1, 1)), uv[sel, 1] * float(max(h - 1, 1))], axis=1)
        ).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, max(w - 1, 0))
        xy[:, 1] = np.clip(xy[:, 1], 0, max(h - 1, 0))
        colors[sel] = rgb[xy[:, 1], xy[:, 0], :3]
    return colors


def _load_pipeline_d4rt_points(
    *,
    pipeline_root: Path,
    scene: str,
    confidence_threshold: float,
    visibility_threshold: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    scene_dir = pipeline_root / "carrier_cache" / scene
    paths = sorted(scene_dir.glob("carriers_window*.npz"))
    if not paths:
        raise FileNotFoundError(f"missing pipeline carrier cache: {scene_dir}")
    point_parts: list[np.ndarray] = []
    uv_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    raw_slot_count = 0
    kept_slot_count = 0
    frames: set[int] = set()
    for path in paths:
        manifest = _read_json(path.with_name(f"{path.stem}_manifest.json"))
        frame_ids = [int(value) for value in manifest.get("frame_ids", [])]
        with np.load(path) as payload:
            xyz = np.asarray(payload["xyz_ref"], dtype=np.float32)
            uv = np.asarray(payload["uv_pred"], dtype=np.float32)
            valid = np.asarray(payload["valid"], dtype=bool)
            confidence = np.asarray(payload["confidence_prob"], dtype=np.float32)
            visibility = np.asarray(payload["visibility_prob"], dtype=np.float32)
        if len(frame_ids) != xyz.shape[0]:
            raise ValueError(f"frame manifest length mismatch in {path}")
        for local_idx, frame_id in enumerate(frame_ids):
            ok = (
                valid[local_idx]
                & np.isfinite(xyz[local_idx]).all(axis=1)
                & np.isfinite(uv[local_idx]).all(axis=1)
                & (uv[local_idx, :, 0] >= 0.0)
                & (uv[local_idx, :, 0] <= 1.0)
                & (uv[local_idx, :, 1] >= 0.0)
                & (uv[local_idx, :, 1] <= 1.0)
                & (confidence[local_idx] >= float(confidence_threshold))
                & (visibility[local_idx] >= float(visibility_threshold))
            )
            raw_slot_count += int(ok.shape[0])
            kept_slot_count += int(np.count_nonzero(ok))
            frames.add(int(frame_id))
            if not np.any(ok):
                continue
            point_parts.append(np.asarray(xyz[local_idx, ok], dtype=np.float32))
            uv_parts.append(np.asarray(uv[local_idx, ok], dtype=np.float32))
            frame_parts.append(np.full((np.count_nonzero(ok),), int(frame_id), dtype=np.int64))
    points = np.concatenate(point_parts, axis=0) if point_parts else np.zeros((0, 3), dtype=np.float32)
    uv = np.concatenate(uv_parts, axis=0) if uv_parts else np.zeros((0, 2), dtype=np.float32)
    frame_ids = np.concatenate(frame_parts, axis=0) if frame_parts else np.zeros((0,), dtype=np.int64)
    idx = _sample_indices(points.shape[0], int(max_points), seed=f"{scene}:v65_soma_pipeline_visualization")
    points = points[idx]
    uv = uv[idx]
    frame_ids = frame_ids[idx]
    colors = _sample_rgb_colors(scene, frame_ids, uv) if points.shape[0] else np.zeros((0, 3), dtype=np.uint8)
    diag = {
        "source": "pipeline carrier_cache xyz_ref/uv_pred",
        "carrier_cache_window_count": int(len(paths)),
        "unique_frame_count": int(len(frames)),
        "frame_min": int(min(frames)) if frames else None,
        "frame_max": int(max(frames)) if frames else None,
        "raw_slot_count": int(raw_slot_count),
        "kept_slot_count_before_sampling": int(kept_slot_count),
        "returned_point_count": int(points.shape[0]),
        "sampled": bool(points.shape[0] < kept_slot_count),
        "confidence_threshold": float(confidence_threshold),
        "visibility_threshold": float(visibility_threshold),
    }
    return points, colors, frame_ids, uv, diag


def _color_points_by_pipeline_support(
    *,
    scene: str,
    mask_dir: Path,
    points: np.ndarray,
    frame_ids: np.ndarray,
    uv: np.ndarray,
    mask_to_object_idx: dict[tuple[int, int], int],
) -> tuple[np.ndarray, dict[str, Any]]:
    colors = np.full((points.shape[0], 3), 170, dtype=np.uint8)
    owner = np.zeros((points.shape[0],), dtype=np.int32)
    evaluated = 0
    missing_mask_frames: set[int] = set()
    for frame_id in sorted(set(np.asarray(frame_ids, dtype=np.int64).tolist())):
        sel = frame_ids == int(frame_id)
        mask_path = mask_dir / f"{int(frame_id)}.png"
        if not mask_path.exists():
            missing_mask_frames.add(int(frame_id))
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            missing_mask_frames.add(int(frame_id))
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]
        h, w = mask.shape[:2]
        xy = np.rint(
            np.stack([uv[sel, 0] * float(max(w - 1, 1)), uv[sel, 1] * float(max(h - 1, 1))], axis=1)
        ).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, max(w - 1, 0))
        xy[:, 1] = np.clip(xy[:, 1], 0, max(h - 1, 0))
        mask_ids = mask[xy[:, 1], xy[:, 0]]
        global_idx = np.flatnonzero(sel)
        evaluated += int(global_idx.shape[0])
        for local_i, mask_id in enumerate(mask_ids.tolist()):
            object_idx = mask_to_object_idx.get((int(frame_id), int(mask_id)), 0)
            if object_idx > 0:
                owner[int(global_idx[local_i])] = int(object_idx)
    assigned = owner > 0
    if np.any(assigned):
        colors[assigned] = _id_colors(owner[assigned])
    diag = {
        "soma_sem_contract": "pipeline D4RT point frame/uv -> Cropformer mask id -> same-root objectlet/reprojection support map",
        "cropformer_mask_dir": _rel(mask_dir),
        "soma_sem_point_count": int(points.shape[0]),
        "soma_sem_evaluated_point_count": int(evaluated),
        "soma_sem_assigned_point_count": int(np.count_nonzero(assigned)),
        "soma_sem_unassigned_point_count": int(points.shape[0] - np.count_nonzero(assigned)),
        "soma_sem_assigned_ratio": float(np.count_nonzero(assigned) / max(points.shape[0], 1)),
        "soma_sem_assigned_object_count": int(np.unique(owner[assigned]).shape[0]) if np.any(assigned) else 0,
        "soma_sem_missing_mask_frame_count_in_sample": int(len(missing_mask_frames)),
        "soma_sem_missing_mask_frames_in_sample_first20": sorted(missing_mask_frames)[:20],
        "soma_sem_unassigned_color": [170, 170, 170],
    }
    return colors, diag


def export_3d_layers(
    *,
    scene: str,
    pipeline_root: Path,
    mask_dir: Path,
    output_root: Path,
    confidence_threshold: float,
    visibility_threshold: float,
    max_d4rt_points: int,
    mask_to_object_idx: dict[tuple[int, int], int],
) -> dict[str, Any]:
    scene_points, scene_colors, mesh_path = _load_scene_mesh(scene)
    gt_labels = _load_gt(scene)
    gt_positive = gt_labels > 0
    d4rt_points, d4rt_colors, frame_ids, uv, d4rt_diag = _load_pipeline_d4rt_points(
        pipeline_root=pipeline_root,
        scene=scene,
        confidence_threshold=confidence_threshold,
        visibility_threshold=visibility_threshold,
        max_points=max_d4rt_points,
    )
    soma_colors, soma_diag = _color_points_by_pipeline_support(
        scene=scene,
        mask_dir=mask_dir,
        points=d4rt_points,
        frame_ids=frame_ids,
        uv=uv,
        mask_to_object_idx=mask_to_object_idx,
    )
    layer_npz = output_root / f"{scene}_pipeline_four_layers.npz"
    np.savez_compressed(
        layer_npz,
        gt_geo_points=scene_points.astype(np.float32),
        gt_geo_colors=scene_colors.astype(np.uint8),
        gt_sem_points=scene_points[gt_positive].astype(np.float32),
        gt_sem_colors=_id_colors(gt_labels[gt_positive]).astype(np.uint8),
        d4rt_geo_points=d4rt_points.astype(np.float32),
        d4rt_geo_colors=d4rt_colors.astype(np.uint8),
        soma_sem_points=d4rt_points.astype(np.float32),
        soma_sem_colors=soma_colors.astype(np.uint8),
    )
    status = {
        "phase": "v65_soma_pipeline_3d_visualization",
        "scene": scene,
        "required_layers": REQUIRED_LAYERS,
        "layer_controls_required": True,
        "layers_npz": _rel(layer_npz),
        "layers_npz_sha256": _sha256(layer_npz),
        "mesh_path": _rel(mesh_path),
        "mesh_path_sha256": _sha256(mesh_path),
        "gt_geo_point_count": int(scene_points.shape[0]),
        "gt_sem_point_count": int(np.count_nonzero(gt_positive)),
        "gt_sem_instance_count": int(np.unique(gt_labels[gt_labels > 0]).shape[0]),
        "d4rt_geo_point_count": int(d4rt_points.shape[0]),
        "soma_sem_point_count": int(d4rt_points.shape[0]),
        "d4rt_diag": d4rt_diag,
        **soma_diag,
    }
    _write_json(output_root / "pipeline_3d_export_status.json", status)
    return status


def run(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_root = _project(args.pipeline_root)
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline_summary = _read_json(pipeline_root / "pipeline_summary.json")
    scene = str(args.scene or pipeline_summary.get("scene"))
    stride = int(args.stride or pipeline_summary.get("input_stride") or 5)
    mask_dir = _resolve_pipeline_mask_dir(scene=scene, pipeline_summary=pipeline_summary, override_mask_root=args.mask_root)
    objectlet_variant = _best_objectlet_variant(pipeline_root, str(args.objectlet_variant))
    support_by_frame, mask_to_object_idx, support_diag = _load_pipeline_support(
        pipeline_root=pipeline_root,
        scene=scene,
        objectlet_variant=objectlet_variant,
        success_only=bool(args.success_only),
    )
    video_status = export_2d_videos(
        scene=scene,
        stride=stride,
        mask_dir=mask_dir,
        output_root=output_root,
        support_by_frame=support_by_frame,
        alpha=float(args.alpha),
        fps=float(args.fps),
        resize_width=int(args.resize_width),
        max_video_frames=int(args.max_video_frames),
    )
    if bool(args.skip_3d):
        three_d_status = {
            "skipped": True,
            "reason": "skip_3d requested; 2D videos only",
        }
    else:
        three_d_status = export_3d_layers(
            scene=scene,
            pipeline_root=pipeline_root,
            mask_dir=mask_dir,
            output_root=output_root,
            confidence_threshold=float(args.confidence_threshold if args.confidence_threshold is not None else pipeline_summary.get("confidence_threshold", 0.2)),
            visibility_threshold=float(args.visibility_threshold if args.visibility_threshold is not None else pipeline_summary.get("visibility_threshold", 0.0)),
            max_d4rt_points=int(args.max_d4rt_points),
            mask_to_object_idx=mask_to_object_idx,
        )
    status = {
        "phase": "v65_soma_pipeline_visualization",
        "scene": scene,
        "stride": stride,
        "pipeline_root": _rel(pipeline_root),
        "pipeline_summary": _rel(pipeline_root / "pipeline_summary.json"),
        "pipeline_summary_sha256": _sha256(pipeline_root / "pipeline_summary.json"),
        "resolved_mask_dir": _rel(mask_dir),
        "pipeline_gate": pipeline_summary.get("pipeline_gate"),
        "mask_frame_coverage": pipeline_summary.get("mask_frame_coverage"),
        "objectlet_variant": objectlet_variant,
        "support_diag": support_diag,
        "video_status": video_status,
        "three_d_status": three_d_status,
        "note": "Visualization is coverage-aware and tied to one pipeline root. If pipeline_gate.ap_ready is false, SOMA sem is diagnostic incomplete support, not AP-ready output.",
    }
    _write_json(output_root / "pipeline_visualization_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export coverage-aware 2D/3D visualizations from one v65 SOMA pipeline root.")
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene", default="")
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--objectlet-variant", default="best")
    parser.add_argument("--success-only", type=int, default=1)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--visibility-threshold", type=float, default=None)
    parser.add_argument("--max-d4rt-points", type=int, default=1000000)
    parser.add_argument("--max-video-frames", type=int, default=0)
    parser.add_argument("--skip-3d", type=int, default=0)
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
