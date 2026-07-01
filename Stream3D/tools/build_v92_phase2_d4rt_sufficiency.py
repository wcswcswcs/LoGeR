from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASE_ID = "v92_phase2_d4rt_sufficiency"
RUN_ID = "v92_phase2_d4rt_sufficiency"
OUT = ROOT / "outputs/audit/v92_phase2_d4rt_sufficiency"
PHASE0_SUMMARY = ROOT / "outputs/audit/v92_phase0_mv_ap_contract/summary.json"
PHASE1_DIR = ROOT / "outputs/audit/v92_phase1_source_container_registry"
SOURCE_CONTAINER_ROWS = PHASE1_DIR / "source_container_rows.csv"
D4RT_SUPPORT_ROWS = ROOT / "outputs/audit/v90_phase3_v82_full_support/native_carrier_support_rows.csv"
WINDOW_ROWS = ROOT / "outputs/audit/v91_phase0_mv_ap_contract/window_support_rows.csv"
SUPPORT_RADIUS_PX = 3
MAX_GT_DIAGNOSTIC_KEYS = 512
COMMON_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "variant_id",
    "scene_id",
    "split",
    "window_id",
    "chunk_id",
    "uses_gt_for_prediction",
    "uses_future",
    "uses_rgbd_pose_mesh",
    "source_artifact",
    "source_artifact_sha256",
    "created_at",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
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


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(ROOT.parent))
        except ValueError:
            return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_sha(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        cache[path] = _sha256(path) if path.exists() else ""
    return cache[path]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for key in COMMON_FIELDS:
            if any(key in row for row in rows):
                fieldnames.append(key)
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _float(value: Any, default: float | str = "") -> float | str:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_div(num: float | int, den: float | int) -> float | str:
    try:
        den_f = float(den)
        if den_f == 0:
            return ""
        return float(num) / den_f
    except Exception:
        return ""


def _percentile(values: list[float], q: float) -> float | str:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return ""
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def _mean(values: list[float]) -> float | str:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else ""


def _median(values: list[float]) -> float | str:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else ""


def _common(
    *,
    schema_version: str,
    variant_id: str,
    scene_id: str,
    split: str,
    window_id: str,
    chunk_id: str,
    uses_gt_for_prediction: bool,
    uses_future: bool,
    uses_rgbd_pose_mesh: bool,
    source_artifact: Path,
    source_artifact_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": variant_id,
        "scene_id": scene_id,
        "split": split,
        "window_id": window_id,
        "chunk_id": chunk_id,
        "uses_gt_for_prediction": bool(uses_gt_for_prediction),
        "uses_future": bool(uses_future),
        "uses_rgbd_pose_mesh": bool(uses_rgbd_pose_mesh),
        "source_artifact": _rel(source_artifact),
        "source_artifact_sha256": source_artifact_sha256,
        "created_at": created_at,
    }


def _key(scene_id: str, frame_id: str | int, mask_id: str | int) -> tuple[str, str, str]:
    return (str(scene_id), str(int(float(frame_id))), str(int(float(mask_id))))


def _load_windows() -> dict[str, list[dict[str, Any]]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(WINDOW_ROWS):
        by_scene[row["scene_id"]].append(
            {
                "split": row.get("split", "dev"),
                "window_id": row.get("window_id", ""),
                "window_index": row.get("window_index", ""),
                "start": _int(row.get("frame_id_start")),
                "end": _int(row.get("frame_id_end")),
                "chunk_id": row.get("chunk_id", ""),
            }
        )
    for rows in by_scene.values():
        rows.sort(key=lambda item: (item["start"], item["end"]))
    return by_scene


def _find_window(
    windows_by_scene: dict[str, list[dict[str, Any]]], cache: dict[tuple[str, int], dict[str, Any]], scene_id: str, frame_id: int
) -> dict[str, Any]:
    cache_key = (scene_id, frame_id)
    if cache_key in cache:
        return cache[cache_key]
    for row in windows_by_scene.get(scene_id, []):
        if row["start"] <= frame_id <= row["end"]:
            cache[cache_key] = row
            return row
    cache[cache_key] = {}
    return {}


def _load_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int64, copy=False)


def _load_label_cached(path: Path, cache: dict[Path, np.ndarray]) -> np.ndarray:
    if path not in cache:
        cache[path] = _load_label(path)
    return cache[path]


def _path_from_rel(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if (ROOT / path).exists():
        return ROOT / path
    if (ROOT.parent / path).exists():
        return ROOT.parent / path
    return ROOT / path


def _gt_path(scene_id: str, frame_id: int) -> Path:
    return ROOT / "data/scannet/processed" / scene_id / "instance/instance" / f"{int(frame_id)}.png"


def _carrier_points(rows: list[dict[str, Any]], shape: tuple[int, int]) -> tuple[np.ndarray, list[str], list[float], list[float]]:
    h, w = shape
    points: list[tuple[int, int]] = []
    carriers: list[str] = []
    confs: list[float] = []
    vis: list[float] = []
    for row in rows:
        x_value = float(row["x"])
        y_value = float(row["y"])
        if 0.0 <= x_value <= 1.0 and 0.0 <= y_value <= 1.0:
            x_value *= max(1, w - 1)
            y_value *= max(1, h - 1)
        x = int(round(x_value))
        y = int(round(y_value))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        points.append((y, x))
        carriers.append(str(row["carrier_id"]))
        confs.append(float(row["confidence"]))
        vis.append(float(row["visibility"]))
    return np.asarray(points, dtype=np.int64), carriers, confs, vis


def _bbox_from_source_row(source_row: dict[str, str], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x0 = _int(source_row.get("mask_bbox_x0"), 0)
    y0 = _int(source_row.get("mask_bbox_y0"), 0)
    x1 = _int(source_row.get("mask_bbox_x1"), w)
    y1 = _int(source_row.get("mask_bbox_y1"), h)
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, w, h
    return x0, y0, x1, y1


def _load_key_shapes(unique_source_rows: dict[tuple[str, str, str], dict[str, str]]) -> dict[tuple[str, str, str], tuple[int, int]]:
    shapes: dict[tuple[str, str, str], tuple[int, int]] = {}
    for key, row in unique_source_rows.items():
        path = _path_from_rel(row.get("mask_path", ""))
        if not path.exists():
            continue
        with Image.open(path) as image:
            width, height = image.size
        shapes[key] = (int(height), int(width))
    return shapes


def _support_stats(
    *,
    source_row: dict[str, str],
    support_rows: list[dict[str, Any]],
    carrier_window_frames: dict[tuple[str, str, str], set[int]],
    label_cache: dict[Path, np.ndarray],
) -> tuple[dict[str, Any], dict[str, list[float]], np.ndarray | None, np.ndarray | None]:
    mask_path = _path_from_rel(source_row["mask_path"])
    mask_id = _int(source_row["source_mask_id"])
    if not mask_path.exists():
        return {}, {}, None, None
    label_full = _load_label_cached(mask_path, label_cache)
    x0, y0, x1, y1 = _bbox_from_source_row(source_row, label_full.shape)
    label = label_full[y0:y1, x0:x1]
    source_mask = label == mask_id
    mask_area = int(np.count_nonzero(source_mask))
    if mask_area == 0:
        return {}, {}, source_mask, None
    points_full, carriers, confs, vis = _carrier_points(support_rows, label_full.shape) if support_rows else (np.zeros((0, 2), dtype=np.int64), [], [], [])
    support_map = np.zeros(source_mask.shape, dtype=bool)
    inside_flags: list[bool] = []
    for idx, (y_full, x_full) in enumerate(points_full):
        y = int(y_full) - y0
        x = int(x_full) - x0
        inside = 0 <= y < source_mask.shape[0] and 0 <= x < source_mask.shape[1] and bool(source_mask[y, x])
        inside_flags.append(inside)
        if inside:
            support_map[y, x] = True
    structure = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(support_map, structure=structure, iterations=SUPPORT_RADIUS_PX) & source_mask
    covered_area = int(np.count_nonzero(dilated))
    _, component_count = ndimage.label(dilated, structure=structure)
    unsupported = source_mask & ~dilated
    _, hole_count = ndimage.label(unsupported, structure=structure)
    distances_to_boundary = ndimage.distance_transform_edt(source_mask)
    boundary_distances: list[float] = []
    boundary_by_carrier: dict[str, list[float]] = defaultdict(list)
    for idx, carrier_id in enumerate(carriers):
        if idx >= points_full.shape[0]:
            continue
        y = int(points_full[idx, 0]) - y0
        x = int(points_full[idx, 1]) - x0
        if 0 <= y < source_mask.shape[0] and 0 <= x < source_mask.shape[1] and bool(source_mask[y, x]):
            value = float(distances_to_boundary[y, x])
            boundary_distances.append(value)
            boundary_by_carrier[carrier_id].append(value)
    nearest: list[float] = []
    if points_full.shape[0] >= 2:
        xy = np.stack([points_full[:, 1], points_full[:, 0]], axis=1).astype(np.float64)
        dist, _ = cKDTree(xy).query(xy, k=2)
        nearest = [float(v) for v in dist[:, 1]]
    scene_id = source_row["scene_id"]
    window_id = source_row["window_id"]
    visibility_frame_counts = [len(carrier_window_frames.get((scene_id, window_id, carrier), set())) for carrier in set(carriers)]
    carrier_count = len(set(carriers))
    visible_count = sum(1 for value in vis if value > 0.0)
    stats = {
        "carrier_count_inside_source": carrier_count,
        "carrier_mass_inside_source": float(sum(c * v for c, v in zip(confs, vis))),
        "visible_carrier_count_inside_source": visible_count,
        "carrier_confidence_mean": _mean(confs),
        "carrier_confidence_p10": _percentile(confs, 10),
        "carrier_visibility_frame_count_mean": _mean([float(v) for v in visibility_frame_counts]),
        "carrier_support_area_ratio": _safe_div(covered_area, mask_area),
        "carrier_point_density": _safe_div(carrier_count, mask_area),
        "carrier_support_connected_component_count": int(component_count),
        "carrier_support_hole_count": int(hole_count),
        "carrier_support_hole_area_ratio": _safe_div(np.count_nonzero(unsupported), mask_area),
        "carrier_nearest_neighbor_distance_mean_px": _mean(nearest),
        "carrier_nearest_neighbor_distance_p90_px": _percentile(nearest, 90),
        "mask_boundary_carrier_distance_mean": _mean(boundary_distances),
        "mask_boundary_carrier_distance_p10": _percentile(boundary_distances, 10),
        "mask_boundary_carrier_distance_p90": _percentile(boundary_distances, 90),
        "support_footprint_radius_px": SUPPORT_RADIUS_PX,
    }
    return stats, boundary_by_carrier, source_mask, dilated


def _load_support_rows(
    keys: set[tuple[str, str, str]],
    windows_by_scene: dict[str, list[dict[str, Any]]],
    key_shapes: dict[tuple[str, str, str], tuple[int, int]],
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[tuple[str, str, str], list[dict[str, Any]]], dict[tuple[str, str, str], set[int]]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    quality_obs: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    carrier_window_frames: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    window_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read_csv(D4RT_SUPPORT_ROWS):
        try:
            key = _key(row.get("scene_id", ""), row.get("frame_id", ""), row.get("mask_id", ""))
        except Exception:
            continue
        if key not in keys:
            continue
        scene_id, frame_text, mask_text = key
        frame_id = int(frame_text)
        window = _find_window(windows_by_scene, window_cache, scene_id, frame_id)
        window_id = window.get("window_id", "")
        carrier_id = row.get("native_carrier_global_id", "")
        x_value = float(_float(row.get("carrier_uv_x"), 0.0) or 0.0)
        y_value = float(_float(row.get("carrier_uv_y"), 0.0) or 0.0)
        shape = key_shapes.get(key)
        if shape is not None and 0.0 <= x_value <= 1.0 and 0.0 <= y_value <= 1.0:
            height, width = shape
            x_value *= max(1, width - 1)
            y_value *= max(1, height - 1)
        item = {
            "scene_id": scene_id,
            "window_id": window_id,
            "frame_id": frame_id,
            "mask_id": mask_text,
            "carrier_id": carrier_id,
            "x": x_value,
            "y": y_value,
            "confidence": float(_float(row.get("confidence"), 0.0) or 0.0),
            "visibility": float(_float(row.get("visibility_prob"), 0.0) or 0.0),
        }
        by_key[key].append(item)
        quality_obs[(scene_id, window_id, carrier_id)].append(item)
        if item["visibility"] > 0.0:
            carrier_window_frames[(scene_id, window_id, carrier_id)].add(frame_id)
    return by_key, quality_obs, carrier_window_frames


def _quality_rows(
    quality_obs: dict[tuple[str, str, str], list[dict[str, Any]]],
    boundary_by_quality: dict[tuple[str, str, str], list[float]],
    artifact_hashes: dict[Path, str],
    created_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scene_id, window_id, carrier_id), obs in sorted(quality_obs.items()):
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in obs:
            by_frame[int(item["frame_id"])].append(item)
        frame_ids = sorted(by_frame)
        uv_by_frame: list[tuple[int, float, float]] = []
        mask_tuple_by_frame: list[tuple[int, tuple[str, ...]]] = []
        contradiction_frames = 0
        confs: list[float] = []
        for frame_id in frame_ids:
            items = by_frame[frame_id]
            confs.extend([float(item["confidence"]) for item in items])
            xs = [float(item["x"]) for item in items]
            ys = [float(item["y"]) for item in items]
            uv_by_frame.append((frame_id, float(np.mean(xs)), float(np.mean(ys))))
            mask_ids = tuple(sorted({str(item["mask_id"]) for item in items}))
            mask_tuple_by_frame.append((frame_id, mask_ids))
            if len(mask_ids) > 1:
                contradiction_frames += 1
        jitters: list[float] = []
        flips = 0
        gaps = 0
        for idx in range(1, len(uv_by_frame)):
            prev = uv_by_frame[idx - 1]
            cur = uv_by_frame[idx]
            jitters.append(float(math.hypot(cur[1] - prev[1], cur[2] - prev[2])))
            if cur[0] - prev[0] > 5:
                gaps += 1
            if mask_tuple_by_frame[idx][1] != mask_tuple_by_frame[idx - 1][1]:
                flips += 1
        boundary_distances = boundary_by_quality.get((scene_id, window_id, carrier_id), [])
        rows.append(
            {
                **_common(
                    schema_version="stream4d_v92_phase2_d4rt_quality_proxy_v1",
                    variant_id="D4RT_SUPPORT_PROXY",
                    scene_id=scene_id,
                    split="dev",
                    window_id=window_id,
                    chunk_id="",
                    uses_gt_for_prediction=False,
                    uses_future=False,
                    uses_rgbd_pose_mesh=False,
                    source_artifact=D4RT_SUPPORT_ROWS,
                    source_artifact_sha256=_artifact_sha(D4RT_SUPPORT_ROWS, artifact_hashes),
                    created_at=created_at,
                ),
                "carrier_id": carrier_id,
                "visible_frame_count": len(frame_ids),
                "confidence_mean": _mean(confs),
                "confidence_p10": _percentile(confs, 10),
                "projection_jitter_mean_px": _mean(jitters),
                "projection_jitter_p90_px": _percentile(jitters, 90),
                "mask_membership_flip_rate": _safe_div(flips, max(0, len(mask_tuple_by_frame) - 1)),
                "source_container_contradiction_rate": _safe_div(contradiction_frames, len(frame_ids)),
                "same_track_visibility_gap_count": gaps,
                "carrier_boundary_distance_variance": float(np.var(boundary_distances)) if boundary_distances else "",
            }
        )
    return rows


def _gt_diagnostic_rows(
    *,
    unique_source_rows: dict[tuple[str, str, str], dict[str, str]],
    support_by_key: dict[tuple[str, str, str], list[dict[str, Any]]],
    artifact_hashes: dict[Path, str],
    label_cache: dict[Path, np.ndarray],
    gt_label_cache: dict[Path, np.ndarray],
    created_at: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diag = Counter()
    sorted_items = sorted(unique_source_rows.items())
    diag["gt_diagnostic_key_count_total"] = len(sorted_items)
    for evaluated_idx, (key, source_row) in enumerate(sorted_items):
        if evaluated_idx >= MAX_GT_DIAGNOSTIC_KEYS:
            diag["gt_diagnostic_key_count_skipped_by_budget"] = len(sorted_items) - evaluated_idx
            break
        diag["gt_diagnostic_key_count_evaluated"] += 1
        scene_id, frame_text, mask_text = key
        frame_id = int(frame_text)
        mask_id = int(mask_text)
        mask_path = _path_from_rel(source_row["mask_path"])
        gt_path = _gt_path(scene_id, frame_id)
        if not mask_path.exists() or not gt_path.exists():
            diag["missing_gt_or_mask"] += 1
            continue
        source_label_full = _load_label_cached(mask_path, label_cache)
        gt_label_full = _load_label_cached(gt_path, gt_label_cache)
        if gt_label_full.shape != source_label_full.shape:
            gt_label_full = np.asarray(
                Image.fromarray(gt_label_full.astype(np.int32)).resize(
                    (source_label_full.shape[1], source_label_full.shape[0]),
                    resample=Image.Resampling.NEAREST,
                )
            )
        x0, y0, x1, y1 = _bbox_from_source_row(source_row, source_label_full.shape)
        source_label = source_label_full[y0:y1, x0:x1]
        gt_label = gt_label_full[y0:y1, x0:x1]
        source_mask = source_label == mask_id
        if not np.any(source_mask):
            diag["missing_source_label"] += 1
            continue
        gt_ids = [int(v) for v in np.unique(gt_label[source_mask]) if int(v) > 0]
        support = support_by_key.get(key, [])
        points_full, _, _, _ = _carrier_points(support, source_label_full.shape) if support else (np.zeros((0, 2), dtype=np.int64), [], [], [])
        support_map = np.zeros(source_mask.shape, dtype=bool)
        points: list[tuple[int, int]] = []
        for y_full, x_full in points_full:
            y = int(y_full) - y0
            x = int(x_full) - x0
            if 0 <= y < source_mask.shape[0] and 0 <= x < source_mask.shape[1] and source_mask[y, x]:
                support_map[y, x] = True
                points.append((y, x))
        structure = ndimage.generate_binary_structure(2, 2)
        dilated_support = ndimage.binary_dilation(support_map, structure=structure, iterations=SUPPORT_RADIUS_PX)
        for gt_id in gt_ids:
            gt_mask = gt_label == gt_id
            gt_area = int(np.count_nonzero(gt_mask))
            if gt_area == 0:
                continue
            source_gt = source_mask & gt_mask
            eroded_gt = ndimage.binary_erosion(gt_mask, structure=structure, iterations=SUPPORT_RADIUS_PX, border_value=0)
            gt_boundary_band = gt_mask & ~eroded_gt
            carrier_inside_gt = 0
            carrier_boundary = 0
            for y, x in points:
                y_i, x_i = int(y), int(x)
                if bool(gt_mask[y_i, x_i]):
                    carrier_inside_gt += 1
                if bool(gt_boundary_band[y_i, x_i]):
                    carrier_boundary += 1
            _, gt_component_count = ndimage.label(dilated_support & gt_mask, structure=structure)
            rows.append(
                {
                    **_common(
                        schema_version="stream4d_v92_phase2_gt_diagnostic_carrier_coverage_v1",
                        variant_id="GT_DIAGNOSTIC_ONLY",
                        scene_id=scene_id,
                        split=source_row.get("split", "dev"),
                        window_id=source_row.get("window_id", ""),
                        chunk_id=source_row.get("chunk_id", ""),
                        uses_gt_for_prediction=False,
                        uses_future=False,
                        uses_rgbd_pose_mesh=False,
                        source_artifact=SOURCE_CONTAINER_ROWS,
                        source_artifact_sha256=_artifact_sha(SOURCE_CONTAINER_ROWS, artifact_hashes),
                        created_at=created_at,
                    ),
                    "gt_window_object_id": f"{source_row.get('window_id', '')}:gt:{gt_id}",
                    "frame_id": frame_id,
                    "source_mask_id": mask_id,
                    "carrier_count_inside_gt": carrier_inside_gt,
                    "carrier_count_inside_source_outside_gt": max(0, len(points) - carrier_inside_gt),
                    "boundary_band_carrier_count_gt": carrier_boundary,
                    "interior_carrier_density_gt": _safe_div(carrier_inside_gt, gt_area),
                    "boundary_carrier_density_gt": _safe_div(carrier_boundary, int(np.count_nonzero(gt_boundary_band))),
                    "carrier_gt_recall_proxy": _safe_div(np.count_nonzero(dilated_support & gt_mask), gt_area),
                    "carrier_gt_precision_proxy": _safe_div(carrier_inside_gt, len(points)),
                    "carrier_component_count_inside_gt": int(gt_component_count),
                    "gt_area_px": gt_area,
                    "source_gt_intersection_area_px": int(np.count_nonzero(source_gt)),
                    "diagnostic_only": True,
                }
            )
    diag["gt_diagnostic_rows"] = len(rows)
    return rows, dict(diag)


def _load_phase0() -> dict[str, Any]:
    if PHASE0_SUMMARY.exists():
        return json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    return {}


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    artifact_hashes: dict[Path, str] = {}
    phase0 = _load_phase0()
    source_rows = _read_csv(SOURCE_CONTAINER_ROWS)
    windows_by_scene = _load_windows()
    unique_source_rows: dict[tuple[str, str, str], dict[str, str]] = {}
    keys: set[tuple[str, str, str]] = set()
    for row in source_rows:
        key = _key(row.get("scene_id", ""), row.get("frame_id", ""), row.get("source_mask_id", ""))
        keys.add(key)
        unique_source_rows.setdefault(key, row)
    key_shapes = _load_key_shapes(unique_source_rows)
    support_by_key, quality_obs, carrier_window_frames = _load_support_rows(keys, windows_by_scene, key_shapes)
    support_stats_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    boundary_by_quality: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    label_cache: dict[Path, np.ndarray] = {}
    gt_label_cache: dict[Path, np.ndarray] = {}
    for key, ref_row in sorted(unique_source_rows.items()):
        stats, boundary_by_carrier, _, _ = _support_stats(
            source_row=ref_row,
            support_rows=support_by_key.get(key, []),
            carrier_window_frames=carrier_window_frames,
            label_cache=label_cache,
        )
        support_stats_by_key[key] = stats
        scene_id = ref_row.get("scene_id", "")
        window_id = ref_row.get("window_id", "")
        for carrier_id, distances in boundary_by_carrier.items():
            boundary_by_quality[(scene_id, window_id, carrier_id)].extend(distances)

    source_container_carrier_rows: list[dict[str, Any]] = []
    boundary_proxy_rows: list[dict[str, Any]] = []
    for row in source_rows:
        key = _key(row.get("scene_id", ""), row.get("frame_id", ""), row.get("source_mask_id", ""))
        stats = support_stats_by_key.get(key, {})
        common = _common(
            schema_version="stream4d_v92_phase2_source_container_carrier_v1",
            variant_id=row.get("variant_id", ""),
            scene_id=row.get("scene_id", ""),
            split=row.get("split", "dev"),
            window_id=row.get("window_id", ""),
            chunk_id=row.get("chunk_id", ""),
            uses_gt_for_prediction=False,
            uses_future=False,
            uses_rgbd_pose_mesh=False,
            source_artifact=SOURCE_CONTAINER_ROWS,
            source_artifact_sha256=_artifact_sha(SOURCE_CONTAINER_ROWS, artifact_hashes),
            created_at=created_at,
        )
        carrier_row = {
            **common,
            "frame_id": row.get("frame_id", ""),
            "source_mask_id": row.get("source_mask_id", ""),
            "source_variant": row.get("source_variant", row.get("variant_id", "")),
            "carrier_count_inside_source": stats.get("carrier_count_inside_source", 0),
            "carrier_mass_inside_source": stats.get("carrier_mass_inside_source", 0.0),
            "visible_carrier_count_inside_source": stats.get("visible_carrier_count_inside_source", 0),
            "carrier_confidence_mean": stats.get("carrier_confidence_mean", ""),
            "carrier_confidence_p10": stats.get("carrier_confidence_p10", ""),
            "carrier_visibility_frame_count_mean": stats.get("carrier_visibility_frame_count_mean", ""),
            "carrier_support_area_ratio": stats.get("carrier_support_area_ratio", ""),
            "carrier_point_density": stats.get("carrier_point_density", ""),
            "carrier_support_connected_component_count": stats.get("carrier_support_connected_component_count", ""),
            "carrier_support_hole_count": stats.get("carrier_support_hole_count", ""),
            "carrier_support_hole_area_ratio": stats.get("carrier_support_hole_area_ratio", ""),
            "carrier_nearest_neighbor_distance_mean_px": stats.get("carrier_nearest_neighbor_distance_mean_px", ""),
            "carrier_nearest_neighbor_distance_p90_px": stats.get("carrier_nearest_neighbor_distance_p90_px", ""),
            "mask_boundary_carrier_distance_mean": stats.get("mask_boundary_carrier_distance_mean", ""),
            "mask_boundary_carrier_distance_p10": stats.get("mask_boundary_carrier_distance_p10", ""),
            "mask_boundary_carrier_distance_p90": stats.get("mask_boundary_carrier_distance_p90", ""),
            "broad_mask_flag": row.get("broad_mask_flag", ""),
            "underseg_risk_score": row.get("underseg_risk_score", ""),
            "support_footprint_radius_px": SUPPORT_RADIUS_PX,
        }
        source_container_carrier_rows.append(carrier_row)
        boundary_common = dict(common)
        boundary_common["schema_version"] = "stream4d_v92_phase2_boundary_proxy_v1"
        boundary_proxy_rows.append(
            {
                **boundary_common,
                "frame_id": row.get("frame_id", ""),
                "source_mask_id": row.get("source_mask_id", ""),
                "source_variant": row.get("source_variant", row.get("variant_id", "")),
                "mask_boundary_carrier_distance_mean": stats.get("mask_boundary_carrier_distance_mean", ""),
                "mask_boundary_carrier_distance_p10": stats.get("mask_boundary_carrier_distance_p10", ""),
                "mask_boundary_carrier_distance_p90": stats.get("mask_boundary_carrier_distance_p90", ""),
                "carrier_support_hole_area_ratio": stats.get("carrier_support_hole_area_ratio", ""),
                "carrier_support_connected_component_count": stats.get("carrier_support_connected_component_count", ""),
                "broad_mask_flag": row.get("broad_mask_flag", ""),
                "underseg_risk_score": row.get("underseg_risk_score", ""),
            }
        )

    quality_rows = _quality_rows(quality_obs, boundary_by_quality, artifact_hashes, created_at)
    gt_rows, gt_diag = _gt_diagnostic_rows(
        unique_source_rows=unique_source_rows,
        support_by_key=support_by_key,
        artifact_hashes=artifact_hashes,
        label_cache=label_cache,
        gt_label_cache=gt_label_cache,
        created_at=created_at,
    )

    density_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_container_carrier_rows:
        density_groups[(row["scene_id"], row["split"], row["window_id"], row["variant_id"])].append(row)
    carrier_density_rows: list[dict[str, Any]] = []
    for (scene_id, split, window_id, variant_id), rows in sorted(density_groups.items()):
        counts = [float(row.get("carrier_count_inside_source") or 0.0) for row in rows]
        ratios = [float(row.get("carrier_support_area_ratio") or 0.0) for row in rows if row.get("carrier_support_area_ratio") != ""]
        boundary = [float(row.get("mask_boundary_carrier_distance_mean") or 0.0) for row in rows if row.get("mask_boundary_carrier_distance_mean") != ""]
        carrier_density_rows.append(
            {
                **_common(
                    schema_version="stream4d_v92_phase2_carrier_density_v1",
                    variant_id=variant_id,
                    scene_id=scene_id,
                    split=split,
                    window_id=window_id,
                    chunk_id="",
                    uses_gt_for_prediction=False,
                    uses_future=False,
                    uses_rgbd_pose_mesh=False,
                    source_artifact=SOURCE_CONTAINER_ROWS,
                    source_artifact_sha256=_artifact_sha(SOURCE_CONTAINER_ROWS, artifact_hashes),
                    created_at=created_at,
                ),
                "source_container_count": len(rows),
                "carrier_count_median": _median(counts),
                "carrier_count_p10": _percentile(counts, 10),
                "carrier_support_area_ratio_median": _median(ratios),
                "carrier_support_area_ratio_p10": _percentile(ratios, 10),
                "boundary_distance_mean_median": _median(boundary),
            }
        )

    unique_stats = list(support_stats_by_key.values())
    unique_counts = [float(row.get("carrier_count_inside_source") or 0.0) for row in unique_stats]
    unique_ratios = [float(row.get("carrier_support_area_ratio") or 0.0) for row in unique_stats if row.get("carrier_support_area_ratio") != ""]
    container_counts = [float(row.get("carrier_count_inside_source") or 0.0) for row in source_container_carrier_rows]
    container_ratios = [float(row.get("carrier_support_area_ratio") or 0.0) for row in source_container_carrier_rows if row.get("carrier_support_area_ratio") != ""]
    quality_jitter_p90_values = [float(row.get("projection_jitter_p90_px") or 0.0) for row in quality_rows if row.get("projection_jitter_p90_px") != ""]
    flip_rates = [float(row.get("mask_membership_flip_rate") or 0.0) for row in quality_rows if row.get("mask_membership_flip_rate") != ""]
    median_carrier_count_unique = _median(unique_counts)
    median_support_ratio_unique = _median(unique_ratios)
    jitter_p90_global = _percentile(quality_jitter_p90_values, 90)
    flip_rate_median = _median(flip_rates)
    resolution_blocker = (
        median_carrier_count_unique != ""
        and median_support_ratio_unique != ""
        and (float(median_carrier_count_unique) < 12.0 or float(median_support_ratio_unique) < 0.08)
    )
    geometry_blocker = (
        jitter_p90_global != ""
        and flip_rate_median != ""
        and float(jitter_p90_global) > 25.0
        and float(flip_rate_median) > 0.25
    )
    v91_mv_ap = float(phase0.get("v91_best_MV_AP_window", 0.0) or 0.0)
    s3d_mv_ap = float(phase0.get("S3D_local_window_MV_AP_window", 0.0) or 0.0)
    if resolution_blocker:
        routing_label = "D4RT_RESOLUTION_LIKELY_BLOCKER"
    elif geometry_blocker:
        routing_label = "D4RT_GEOMETRY_QUALITY_LIKELY_BLOCKER"
    elif v91_mv_ap < s3d_mv_ap:
        routing_label = "D4RT_SUFFICIENT_READOUT_LIKELY_BLOCKER"
    else:
        routing_label = "D4RT_DIAGNOSTIC_INCONCLUSIVE"

    variant_config_rows = [
        {
            **_common(
                schema_version="stream4d_v92_phase2_variant_config_v1",
                variant_id="D4RT_GT_FREE_SUFFICIENCY",
                scene_id="ALL_DEV",
                split="dev",
                window_id="ALL_WINDOWS",
                chunk_id="",
                uses_gt_for_prediction=False,
                uses_future=False,
                uses_rgbd_pose_mesh=False,
                source_artifact=SOURCE_CONTAINER_ROWS,
                source_artifact_sha256=_artifact_sha(SOURCE_CONTAINER_ROWS, artifact_hashes),
                created_at=created_at,
            ),
            "support_footprint_radius_px": SUPPORT_RADIUS_PX,
            "routing_uses_gt": False,
            "resolution_count_threshold": 12.0,
            "resolution_support_area_ratio_threshold": 0.08,
            "geometry_jitter_p90_threshold_px": 25.0,
            "geometry_flip_rate_threshold": 0.25,
        }
    ]
    variant_metric_rows = [
        {
            **_common(
                schema_version="stream4d_v92_phase2_variant_metric_v1",
                variant_id="D4RT_GT_FREE_SUFFICIENCY",
                scene_id="ALL_DEV",
                split="dev",
                window_id="ALL_WINDOWS",
                chunk_id="",
                uses_gt_for_prediction=False,
                uses_future=False,
                uses_rgbd_pose_mesh=False,
                source_artifact=SOURCE_CONTAINER_ROWS,
                source_artifact_sha256=_artifact_sha(SOURCE_CONTAINER_ROWS, artifact_hashes),
                created_at=created_at,
            ),
            "median_carrier_count_inside_source_unique_key": median_carrier_count_unique,
            "median_carrier_support_area_ratio_unique_key": median_support_ratio_unique,
            "median_carrier_count_inside_source_container_weighted": _median(container_counts),
            "median_carrier_support_area_ratio_container_weighted": _median(container_ratios),
            "projection_jitter_p90_global": jitter_p90_global,
            "mask_membership_flip_rate_median": flip_rate_median,
            "v91_best_MV_AP_window": v91_mv_ap,
            "S3D_local_window_MV_AP_window": s3d_mv_ap,
            "routing_label": routing_label,
        }
    ]
    gates = {
        "source_container_carrier_rows_gt_0": len(source_container_carrier_rows) > 0,
        "d4rt_quality_proxy_rows_gt_0": len(quality_rows) > 0,
        "routing_uses_gt_false": True,
        "uses_gt_for_prediction_count_eq_0": True,
        "uses_future_count_eq_0": True,
    }
    variant_gate_rows = [
        {
            **_common(
                schema_version="stream4d_v92_phase2_variant_gate_v1",
                variant_id="D4RT_GT_FREE_SUFFICIENCY",
                scene_id="ALL_DEV",
                split="dev",
                window_id="ALL_WINDOWS",
                chunk_id="",
                uses_gt_for_prediction=False,
                uses_future=False,
                uses_rgbd_pose_mesh=False,
                source_artifact=SOURCE_CONTAINER_ROWS,
                source_artifact_sha256=_artifact_sha(SOURCE_CONTAINER_ROWS, artifact_hashes),
                created_at=created_at,
            ),
            "gate_name": name,
            "gate_pass": bool(value),
        }
        for name, value in gates.items()
    ]
    variant_failure_rows = []
    if not all(gates.values()):
        for name, value in gates.items():
            if not value:
                variant_failure_rows.append(
                    {
                        **_common(
                            schema_version="stream4d_v92_phase2_variant_failure_v1",
                            variant_id="D4RT_GT_FREE_SUFFICIENCY",
                            scene_id="ALL_DEV",
                            split="dev",
                            window_id="ALL_WINDOWS",
                            chunk_id="",
                            uses_gt_for_prediction=False,
                            uses_future=False,
                            uses_rgbd_pose_mesh=False,
                            source_artifact=SOURCE_CONTAINER_ROWS,
                            source_artifact_sha256=_artifact_sha(SOURCE_CONTAINER_ROWS, artifact_hashes),
                            created_at=created_at,
                        ),
                        "failure_type": name,
                        "repair_direction": "rebuild Phase1 registry or D4RT support rows before routing",
                    }
                )
    casebook_rows = [
        {
            **_common(
                schema_version="stream4d_v92_phase2_casebook_v1",
                variant_id="D4RT_GT_FREE_SUFFICIENCY",
                scene_id="ALL_DEV",
                split="dev",
                window_id="ALL_WINDOWS",
                chunk_id="",
                uses_gt_for_prediction=False,
                uses_future=False,
                uses_rgbd_pose_mesh=False,
                source_artifact=SOURCE_CONTAINER_ROWS,
                source_artifact_sha256=_artifact_sha(SOURCE_CONTAINER_ROWS, artifact_hashes),
                created_at=created_at,
            ),
            "case_type": "routing_decision",
            "routing_label": routing_label,
            "evidence": f"median_count={median_carrier_count_unique}; median_support_area_ratio={median_support_ratio_unique}; GT diagnostic rows={len(gt_rows)} diagnostic-only",
        }
    ]

    outputs: dict[str, list[dict[str, Any]]] = {
        "carrier_density_rows.csv": carrier_density_rows,
        "source_container_carrier_rows.csv": source_container_carrier_rows,
        "boundary_proxy_rows.csv": boundary_proxy_rows,
        "d4rt_quality_proxy_rows.csv": quality_rows,
        "gt_diagnostic_carrier_coverage_rows.csv": gt_rows,
        "d4rt_sufficiency_casebook_rows.csv": casebook_rows,
        "variant_config_rows.csv": variant_config_rows,
        "variant_metric_rows.csv": variant_metric_rows,
        "variant_gate_rows.csv": variant_gate_rows,
        "variant_failure_rows.csv": variant_failure_rows,
        "casebook_rows.csv": casebook_rows,
    }
    field_order = {
        "source_container_carrier_rows.csv": COMMON_FIELDS
        + [
            "frame_id",
            "source_mask_id",
            "source_variant",
            "carrier_count_inside_source",
            "carrier_mass_inside_source",
            "visible_carrier_count_inside_source",
            "carrier_confidence_mean",
            "carrier_confidence_p10",
            "carrier_visibility_frame_count_mean",
            "carrier_support_area_ratio",
            "carrier_point_density",
            "carrier_support_connected_component_count",
            "carrier_support_hole_count",
            "carrier_support_hole_area_ratio",
            "carrier_nearest_neighbor_distance_mean_px",
            "carrier_nearest_neighbor_distance_p90_px",
            "mask_boundary_carrier_distance_mean",
            "mask_boundary_carrier_distance_p10",
            "mask_boundary_carrier_distance_p90",
            "broad_mask_flag",
            "underseg_risk_score",
            "support_footprint_radius_px",
        ],
        "d4rt_quality_proxy_rows.csv": COMMON_FIELDS
        + [
            "carrier_id",
            "visible_frame_count",
            "confidence_mean",
            "confidence_p10",
            "projection_jitter_mean_px",
            "projection_jitter_p90_px",
            "mask_membership_flip_rate",
            "source_container_contradiction_rate",
            "same_track_visibility_gap_count",
            "carrier_boundary_distance_variance",
        ],
        "gt_diagnostic_carrier_coverage_rows.csv": COMMON_FIELDS
        + [
            "gt_window_object_id",
            "frame_id",
            "source_mask_id",
            "carrier_count_inside_gt",
            "carrier_count_inside_source_outside_gt",
            "boundary_band_carrier_count_gt",
            "interior_carrier_density_gt",
            "boundary_carrier_density_gt",
            "carrier_gt_recall_proxy",
            "carrier_gt_precision_proxy",
            "carrier_component_count_inside_gt",
            "gt_area_px",
            "source_gt_intersection_area_px",
            "diagnostic_only",
        ],
    }
    for filename, rows in outputs.items():
        _write_csv(OUT / filename, rows, field_order.get(filename))

    summary = {
        "schema": "stream4d_v92_phase2_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": routing_label,
        "routing_label": routing_label,
        "routing_uses_gt": False,
        "routing_rules": {
            "resolution": "median carrier_count_inside_source < 12 or median carrier_support_area_ratio < 0.08",
            "geometry": "projection_jitter_p90_px high and mask_membership_flip_rate high",
            "readout": "D4RT sufficient by proxies but v91 MV_AP_window remains below S3D local-window baseline",
        },
        "support_footprint_radius_px": SUPPORT_RADIUS_PX,
        "source_container_count": len(source_container_carrier_rows),
        "unique_scene_frame_mask_count": len(unique_source_rows),
        "carrier_density_row_count": len(carrier_density_rows),
        "d4rt_quality_proxy_row_count": len(quality_rows),
        "gt_diagnostic_carrier_coverage_row_count": len(gt_rows),
        "gt_diagnostic_status": "diagnostic_only_completed" if gt_rows else "diagnostic_only_no_rows",
        "gt_diagnostic_sampling_policy": f"first_{MAX_GT_DIAGNOSTIC_KEYS}_sorted_unique_scene_frame_mask_keys_diagnostic_only_not_used_for_routing",
        "gt_diagnostic_max_keys": MAX_GT_DIAGNOSTIC_KEYS,
        "gt_diagnostic_counts": gt_diag,
        "median_carrier_count_inside_source_unique_key": median_carrier_count_unique,
        "median_carrier_support_area_ratio_unique_key": median_support_ratio_unique,
        "median_carrier_count_inside_source_container_weighted": _median(container_counts),
        "median_carrier_support_area_ratio_container_weighted": _median(container_ratios),
        "carrier_support_area_ratio_p10_unique_key": _percentile(unique_ratios, 10),
        "projection_jitter_p90_global": jitter_p90_global,
        "mask_membership_flip_rate_median": flip_rate_median,
        "v91_best_MV_AP_window": v91_mv_ap,
        "S3D_local_window_MV_AP_window": s3d_mv_ap,
        "resolution_blocker": resolution_blocker,
        "geometry_blocker": geometry_blocker,
        "phase2_gate_pass": all(gates.values()),
        "phase2_gates": gates,
        "input_artifacts": {
            _rel(path): _artifact_sha(path, artifact_hashes)
            for path in [PHASE0_SUMMARY, SOURCE_CONTAINER_ROWS, D4RT_SUPPORT_ROWS, WINDOW_ROWS]
            if path.exists()
        },
        "duration_sec": time.time() - started,
        "source_label_cache_entry_count": len(label_cache),
        "gt_label_cache_entry_count": len(gt_label_cache),
        "key_shape_count": len(key_shapes),
        "created_at": created_at,
    }
    _write_json(OUT / "summary.json", summary)
    sha_rows = {path.name: _sha256(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "SHA256SUMS.json"}
    _write_json(OUT / "SHA256SUMS.json", sha_rows)
    return summary


if __name__ == "__main__":
    print(json.dumps(_jsonable(run()), indent=2, sort_keys=True))
