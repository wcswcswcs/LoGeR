from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v89_recalc_point_projected_mv_ap as recalc  # noqa: E402
from tools import run_v90_mv_ap_window_resurrection as phase1  # noqa: E402


OUT = ROOT / "outputs/audit/v90_phase3_carrier_supported_carving"
PHASE0_WINDOWS = ROOT / "outputs/audit/v90_phase0_mv_ap_contract/window_support_rows.csv"
PHASE1_ROOT = ROOT / "outputs/audit/v90_phase1_variant_resurrection"
DEFAULT_NATIVE_SUPPORT_ROWS = ROOT / "outputs/audit/v85_phase7_renderable_materializer/native_carrier_support_rows.csv"

METHOD_SOURCE_VARIANT = "R10_v82_local_B0_object_slot_config"
CONTROL_SOURCE_VARIANT = "C0_semantic_only_control"


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _mask_dir_by_scene() -> dict[str, Path]:
    rows = _read_csv(PHASE0_WINDOWS)
    out: dict[str, Path] = {}
    for row in rows:
        scene = row.get("scene_id", "")
        mask_source = row.get("mask_source", "")
        if scene and mask_source:
            out[scene] = ROOT / mask_source
    return out


def _window_maps(window_rows: list[dict[str, str]]) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], str]]:
    frame_to_window_index: dict[tuple[str, int], int] = {}
    frame_to_window_id: dict[tuple[str, int], str] = {}
    for row in window_rows:
        scene = row.get("scene_id", "")
        window_index = _int(row.get("window_index"), -1)
        window_id = row.get("window_id") or f"w{window_index:04d}"
        start = _int(row.get("frame_id_start"), -1)
        end = _int(row.get("frame_id_end"), -1)
        if not scene or window_index < 0 or start < 0 or end < 0:
            continue
        for frame_id in range(start, end + 1, 5):
            frame_to_window_index[(scene, frame_id)] = window_index
            frame_to_window_id[(scene, frame_id)] = window_id
    return frame_to_window_index, frame_to_window_id


def _local_slot_from_row(row: dict[str, Any]) -> str:
    local_slot = str(row.get("local_slot_id", "") or "")
    if local_slot:
        return local_slot
    chunk = _int(row.get("chunk_id"), -1)
    cluster = str(row.get("cluster_id", "") or "")
    if cluster == "":
        obj = str(row.get("mv_object_id", ""))
        if ":cluster" in obj and ":c" in obj:
            try:
                chunk_part = obj.split(":c", 1)[1].split(":cluster", 1)[0]
                cluster_part = obj.split(":cluster", 1)[1].split(":", 1)[0]
                chunk = int(chunk_part)
                cluster = cluster_part
            except Exception:
                pass
    if chunk >= 0 and cluster != "":
        return f"V80_object:c{chunk}:cluster{cluster}"
    return ""


def _load_support_rows(
    support_rows_path: Path,
) -> tuple[dict[tuple[str, str, int, int], list[dict[str, Any]]], dict[tuple[str, str, int], dict[int, list[dict[str, Any]]]]]:
    by_exact: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    by_slot_frame: dict[tuple[str, str, int], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in _read_csv(support_rows_path):
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        if not _bool(row.get("native_support_allowed", "True")):
            continue
        scene = row.get("scene_id", "")
        slot = row.get("local_slot_id", "")
        frame_id = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("mask_id"), -1)
        if not scene or not slot or frame_id < 0 or mask_id <= 0:
            continue
        item = {
            "x": _num(row.get("carrier_uv_x")),
            "y": _num(row.get("carrier_uv_y")),
            "confidence": _num(row.get("confidence"), 1.0),
            "visibility_prob": _num(row.get("visibility_prob"), 1.0),
            "native_carrier_global_id": row.get("native_carrier_global_id", ""),
        }
        by_exact[(scene, slot, frame_id, mask_id)].append(item)
        by_slot_frame[(scene, slot, frame_id)][mask_id].append(item)
    return by_exact, by_slot_frame


def _paint_support(points: list[dict[str, Any]], shape: tuple[int, int], radius: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    heat = np.zeros((h, w), dtype=np.float32)
    support = np.zeros((h, w), dtype=np.uint8)
    for point in points:
        x = int(round(max(0.0, min(1.0, float(point["x"]))) * (w - 1)))
        y = int(round(max(0.0, min(1.0, float(point["y"]))) * (h - 1)))
        weight = float(point.get("confidence", 1.0)) * float(point.get("visibility_prob", 1.0))
        cv2.circle(heat, (x, y), int(max(1, radius)), float(weight), thickness=-1)
        cv2.circle(support, (x, y), int(max(1, radius)), 1, thickness=-1)
    return heat, support.astype(bool)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    k = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _connected_component_around_support(source_mask: np.ndarray, support: np.ndarray) -> np.ndarray:
    labels_count, labels = cv2.connectedComponents(source_mask.astype(np.uint8), connectivity=8)
    return _connected_component_from_labels(labels_count, labels, support, source_mask)


def _connected_component_from_labels(labels_count: int, labels: np.ndarray, support: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    if labels_count <= 1:
        return np.zeros_like(source_mask, dtype=bool)
    keep = np.zeros_like(source_mask, dtype=bool)
    hit_labels = sorted({int(v) for v in np.unique(labels[support & source_mask]) if int(v) > 0})
    for label in hit_labels:
        keep |= labels == label
    return keep


def _adaptive_radius(source_area: int, support_count: int, base_radius: int) -> int:
    if support_count <= 0:
        return int(base_radius)
    sparse_radius = int(round(math.sqrt(max(1, source_area) / max(1, support_count)) * 0.45))
    return int(max(base_radius, min(48, sparse_radius)))


def _source_rows() -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(PHASE1_ROOT / "adapter_input_frame_mask_rows.csv"):
        variant = row.get("variant", "")
        if variant not in {METHOD_SOURCE_VARIANT, CONTROL_SOURCE_VARIANT}:
            continue
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        slot = _local_slot_from_row(row)
        if not slot:
            continue
        rows.append({**row, "local_slot_id": slot})
    return rows


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {"suffix": "A0_whole_mask_adapter", "mode": "whole", "source": METHOD_SOURCE_VARIANT, "radius": 0},
        {"suffix": "A1_carrier_support_only", "mode": "support_only", "source": METHOD_SOURCE_VARIANT, "radius": 3},
        {"suffix": "A2_mask_intersect_dilated_support_r08", "mode": "intersect", "source": METHOD_SOURCE_VARIANT, "radius": 8},
        {"suffix": "A2_mask_intersect_dilated_support_r16", "mode": "intersect", "source": METHOD_SOURCE_VARIANT, "radius": 16},
        {"suffix": "A2_mask_intersect_dilated_support_r32", "mode": "intersect", "source": METHOD_SOURCE_VARIANT, "radius": 32},
        {"suffix": "A3_mask_connected_component_around_support_r16", "mode": "component", "source": METHOD_SOURCE_VARIANT, "radius": 16},
        {"suffix": "A3_mask_connected_component_around_support_r32", "mode": "component", "source": METHOD_SOURCE_VARIANT, "radius": 32},
        {"suffix": "A4_multi_mask_union_carved_by_support_r24", "mode": "multi_union", "source": METHOD_SOURCE_VARIANT, "radius": 24},
        {"suffix": "A5_soft_heatmap_ranked_carving_adaptive", "mode": "adaptive", "source": METHOD_SOURCE_VARIANT, "radius": 8},
        {"suffix": "C0_A0_whole_mask_control", "mode": "whole", "source": CONTROL_SOURCE_VARIANT, "radius": 0},
        {"suffix": "C0_A2_mask_intersect_dilated_support_r16", "mode": "intersect", "source": CONTROL_SOURCE_VARIANT, "radius": 16},
        {"suffix": "C0_A3_mask_connected_component_around_support_r32", "mode": "component", "source": CONTROL_SOURCE_VARIANT, "radius": 32},
    ]


def _original_mask_eval_variants() -> set[str]:
    return {str(spec["suffix"]) for spec in _variant_specs() if spec.get("mode") == "whole"}


def _mask_path(mask_dirs: dict[str, Path], scene: str, frame_id: int) -> Path:
    return mask_dirs[scene] / f"{int(frame_id)}.png"


def _read_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int64, copy=False)


def _row_base_id(row: dict[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('variant')}|{row.get('mv_object_id')}|{row.get('frame_id')}|{row.get('mask_id')}"


def _generate_masks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mask_dirs = _mask_dir_by_scene()
    window_rows = _read_csv(PHASE0_WINDOWS)
    frame_to_window_index, frame_to_window_id = _window_maps(window_rows)
    support_rows_path = Path(args.native_support_rows)
    if not support_rows_path.is_absolute():
        support_rows_path = ROOT / support_rows_path
    support_exact, support_by_slot_frame = _load_support_rows(support_rows_path)
    source_rows = _source_rows()
    specs = _variant_specs()
    generated_rows: list[dict[str, Any]] = []
    heatmap_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    tasks_by_frame: dict[tuple[str, str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for spec in specs:
        source_variant = spec["source"]
        variant = spec["suffix"]
        for row in source_rows:
            if row.get("variant") != source_variant:
                continue
            scene = row.get("scene_id", "")
            frame_id = _int(row.get("frame_id"), -1)
            source_mask_id = _int(row.get("mask_id"), -1)
            slot = row.get("local_slot_id", "")
            if not scene or frame_id < 0 or source_mask_id <= 0 or not slot:
                continue
            tasks_by_frame[(variant, scene, frame_id)].append((spec, row))

    generated_mask_root = OUT / "generated_masks"
    for (variant, scene, frame_id), tasks in sorted(tasks_by_frame.items()):
        label = _read_label(_mask_path(mask_dirs, scene, frame_id))
        shape = label.shape
        frame_items: list[dict[str, Any]] = []
        component_cache: dict[int, tuple[int, np.ndarray, np.ndarray, tuple[int, int, int, int]] | None] = {}
        for spec, row in tasks:
            mode = spec["mode"]
            base_radius = int(spec["radius"])
            source_mask_id = _int(row.get("mask_id"), -1)
            slot = row.get("local_slot_id", "")
            source_mask = label == int(source_mask_id)
            source_area = int(np.count_nonzero(source_mask))
            exact_points = support_exact.get((scene, slot, frame_id, source_mask_id), [])
            radius = int(base_radius)
            if mode == "adaptive":
                radius = _adaptive_radius(source_area, len(exact_points), base_radius)
            window_id = frame_to_window_id.get((scene, frame_id), "")
            window_index = frame_to_window_index.get((scene, frame_id), -1)
            if not exact_points and mode in {"support_only", "intersect", "component", "adaptive"}:
                heatmap_rows.append(
                    {
                        "variant_id": variant,
                        "source_variant": row.get("variant", ""),
                        "scene_id": scene,
                        "window_id": window_id,
                        "window_index": window_index,
                        "mv_object_id": row.get("mv_object_id", ""),
                        "local_slot_id": slot,
                        "frame_id": frame_id,
                        "source_mask_id": source_mask_id,
                        "carving_mode": f"{mode}_no_exact_carrier_support",
                        "carrier_support_count": 0,
                        "support_area_raw": 0,
                        "support_area": 0,
                        "heatmap_sum": 0.0,
                        "support_point_radius": int(args.support_point_radius),
                        "dilation_radius": radius,
                        "source_mask_area": source_area,
                        "support_inside_source_mask_ratio": 0.0,
                        "carrier_projection_coverage_rate": 0.0,
                        "source": _rel(support_rows_path),
                    }
                )
                continue
            heat, support = _paint_support(exact_points, shape, max(1, int(args.support_point_radius)))
            support_area_raw = int(np.count_nonzero(support))
            dilated_support = _dilate(support, radius)
            support_area = int(np.count_nonzero(dilated_support))
            if mode == "whole":
                generated = source_mask
                carving_mode = "whole_source_mask"
            elif mode == "support_only":
                generated = support
                carving_mode = "carrier_support_only"
            elif mode in {"intersect", "adaptive"}:
                generated = source_mask & dilated_support
                carving_mode = f"source_intersect_dilated_support_r{radius:02d}"
            elif mode == "component":
                if not np.any(dilated_support & source_mask):
                    generated = np.zeros_like(source_mask, dtype=bool)
                else:
                    cached = component_cache.get(source_mask_id)
                    if source_mask_id not in component_cache:
                        ys, xs = np.nonzero(source_mask)
                        if ys.size == 0:
                            cached = None
                        else:
                            y0 = max(0, int(np.min(ys)) - 1)
                            y1 = min(source_mask.shape[0], int(np.max(ys)) + 2)
                            x0 = max(0, int(np.min(xs)) - 1)
                            x1 = min(source_mask.shape[1], int(np.max(xs)) + 2)
                            source_crop = source_mask[y0:y1, x0:x1]
                            labels_count, labels = cv2.connectedComponents(source_crop.astype(np.uint8), connectivity=8)
                            cached = (labels_count, labels, source_crop, (y0, y1, x0, x1))
                        component_cache[source_mask_id] = cached
                    if cached is None:
                        generated = np.zeros_like(source_mask, dtype=bool)
                    else:
                        labels_count, labels, source_crop, (y0, y1, x0, x1) = cached
                        support_crop = dilated_support[y0:y1, x0:x1]
                        generated_crop = _connected_component_from_labels(labels_count, labels, support_crop, source_crop)
                        generated = np.zeros_like(source_mask, dtype=bool)
                        generated[y0:y1, x0:x1] = generated_crop
                carving_mode = f"connected_component_touching_support_r{radius:02d}"
            elif mode == "multi_union":
                generated = np.zeros(shape, dtype=bool)
                for other_mask_id, points in support_by_slot_frame.get((scene, slot, frame_id), {}).items():
                    if len(points) < int(args.min_support_points):
                        continue
                    _heat_other, support_other = _paint_support(points, shape, max(1, int(args.support_point_radius)))
                    generated |= (label == int(other_mask_id)) & _dilate(support_other, radius)
                carving_mode = f"multi_mask_union_intersect_dilated_support_r{radius:02d}"
            else:
                raise ValueError(f"unsupported carving mode {mode}")
            generated_area = int(np.count_nonzero(generated))
            support_inside = int(np.count_nonzero(support & source_mask))
            support_inside_ratio = float(support_inside / max(1, support_area_raw))
            carrier_projection_coverage_rate = float(support_inside / max(1, source_area))
            broad_risk = bool(generated_area > source_area * float(args.broad_risk_area_ratio)) if source_area else False
            heatmap_rows.append(
                {
                    "variant_id": variant,
                    "source_variant": row.get("variant", ""),
                    "scene_id": scene,
                    "window_id": window_id,
                    "window_index": window_index,
                    "mv_object_id": row.get("mv_object_id", ""),
                    "local_slot_id": slot,
                    "frame_id": frame_id,
                    "source_mask_id": source_mask_id,
                    "carving_mode": carving_mode,
                    "carrier_support_count": len(exact_points),
                    "support_area_raw": support_area_raw,
                    "support_area": support_area,
                    "heatmap_sum": float(np.sum(heat)),
                    "support_point_radius": int(args.support_point_radius),
                    "dilation_radius": radius,
                    "source_mask_area": source_area,
                    "support_inside_source_mask_ratio": support_inside_ratio,
                    "carrier_projection_coverage_rate": carrier_projection_coverage_rate,
                    "source": _rel(support_rows_path),
                }
            )
            if generated_area <= 0:
                continue
            new_mask_id = len(frame_items) + 1
            frame_items.append(
                {
                    "new_mask_id": new_mask_id,
                    "mask": generated,
                    "score": _num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)),
                    "row": row,
                    "carving_mode": carving_mode,
                    "source_area": source_area,
                    "support_area": support_area,
                    "support_count": len(exact_points),
                    "generated_area": generated_area,
                    "support_inside_ratio": support_inside_ratio,
                    "coverage_rate": carrier_projection_coverage_rate,
                    "broad_risk": broad_risk,
                    "window_id": window_id,
                    "window_index": window_index,
                    "source_mask_id": source_mask_id,
                    "radius": radius,
                    "mode": mode,
                }
            )
        if not frame_items:
            continue
        label_out = np.zeros(shape, dtype=np.uint16)
        for item in frame_items:
            write_mask = item["mask"] & (label_out == 0)
            label_out[write_mask] = int(item["new_mask_id"])
        out_dir = generated_mask_root / variant / scene / "mask"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{int(frame_id)}.png"
        if not cv2.imwrite(str(out_path), label_out):
            raise RuntimeError(f"failed to write generated mask {out_path}")
        for item in frame_items:
            row = item["row"]
            final_area = int(np.count_nonzero(label_out == int(item["new_mask_id"])))
            if final_area <= 0:
                continue
            generated_rows.append(
                {
                    "variant_id": variant,
                    "source_variant": row.get("variant", ""),
                    "scene_id": scene,
                    "window_id": item["window_id"],
                    "window_index": item["window_index"],
                    "mv_object_id": row.get("mv_object_id", ""),
                    "local_slot_id": row.get("local_slot_id", ""),
                    "frame_id": int(frame_id),
                    "source_mask_id": item["source_mask_id"],
                    "new_mask_id": int(item["new_mask_id"]),
                    "generated_mask_path": _rel(out_path),
                    "carving_mode": item["carving_mode"],
                    "carrier_support_count": int(item["support_count"]),
                    "support_area": int(item["support_area"]),
                    "source_mask_area": int(item["source_area"]),
                    "generated_mask_area": final_area,
                    "pre_wta_generated_mask_area": int(item["generated_area"]),
                    "support_inside_source_mask_ratio": item["support_inside_ratio"],
                    "carrier_projection_coverage_rate": item["coverage_rate"],
                    "broad_risk": bool(item["broad_risk"]),
                    "object_score": item["score"],
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                }
            )
            if item["mode"] != "whole":
                eval_rows.append(
                    {
                        "split": "dev",
                        "scene_id": scene,
                        "source_variant": variant,
                        "variant": variant,
                        "mv_object_id": row.get("mv_object_id", ""),
                        "frame_id": int(frame_id),
                        "mask_id": int(item["new_mask_id"]),
                        "frame_mask_score": item["score"],
                        "object_score": item["score"],
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                        "uses_rgbd_pose_mesh": False,
                        "materializable": True,
                    }
                )
    for spec in specs:
        if spec.get("mode") != "whole":
            continue
        variant = str(spec["suffix"])
        source_variant = str(spec["source"])
        for row in source_rows:
            if row.get("variant") != source_variant:
                continue
            eval_rows.append(
                {
                    "split": "dev",
                    "scene_id": row.get("scene_id", ""),
                    "source_variant": variant,
                    "variant": variant,
                    "mv_object_id": row.get("mv_object_id", ""),
                    "frame_id": _int(row.get("frame_id"), -1),
                    "mask_id": _int(row.get("mask_id"), -1),
                    "frame_mask_score": _num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)),
                    "object_score": _num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "uses_rgbd_pose_mesh": False,
                    "materializable": True,
                    "eval_mask_source": "original_cropformer_mask_dir_for_phase1_parity",
                }
            )
    manifest = {
        "generated_mask_root": _rel(generated_mask_root),
        "variant_count": len({row["variant_id"] for row in generated_rows}),
        "generated_mask_rows": len(generated_rows),
        "eval_rows": len(eval_rows),
        "heatmap_rows": len(heatmap_rows),
        "source_rows": len(source_rows),
        "native_support_rows": _rel(support_rows_path),
        "method_source_variant": METHOD_SOURCE_VARIANT,
        "control_source_variant": CONTROL_SOURCE_VARIANT,
        "support_point_radius": int(args.support_point_radius),
        "min_support_points": int(args.min_support_points),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    return heatmap_rows, generated_rows, eval_rows, manifest


def _all_iou_rows(
    iou: np.ndarray,
    pred_ids: list[int],
    gt_ids: list[int],
    top_k: int = 100,
    **_kwargs: Any,
) -> list[dict[str, Any]]:
    rows = []
    if iou.size == 0:
        return rows
    for pidx, pred_id in enumerate(pred_ids):
        for gidx, gt_id in enumerate(gt_ids):
            rows.append({"pred_id": int(pred_id), "gt_id": int(gt_id), "iou": float(iou[pidx, gidx])})
    return rows


def _evaluate(eval_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    original_top = recalc._top_iou_rows
    original_mask_dir = recalc._mask_dir
    original_mask_dirs = _mask_dir_by_scene()
    original_mask_variants = _original_mask_eval_variants()
    recalc._top_iou_rows = _all_iou_rows
    metric_rows: list[dict[str, Any]] = []
    iou_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    scope = recalc._frame_scope()
    local_export_root = ROOT / "outputs/audit/v89_recalc_point_projected_mv_ap"
    variants = sorted({str(row.get("variant", "")) for row in eval_rows if row.get("variant")})
    try:
        for variant in variants:
            if variant in original_mask_variants:
                recalc._mask_dir = lambda scene: original_mask_dirs[scene]
            else:
                recalc._mask_dir = lambda scene, _variant=variant: OUT / "generated_masks" / _variant / scene / "mask"
            for scene in ["scene0011_00", "scene0050_00"]:
                rows = [row for row in eval_rows if row.get("variant") == variant and row.get("scene_id") == scene]
                if not rows:
                    continue
                frame_ids = scope.get(("dev", scene))
                metric, cases, tops, _window_rows = recalc._evaluate_frame_mask_variant_local_window(
                    scene=scene,
                    split="dev",
                    variant=variant,
                    frame_ids=frame_ids,
                    rows=rows,
                    score_mode="input",
                    local_export_root=local_export_root,
                    window_source_step="S3D_L1_local_merged_masks",
                )
                sf50_f1 = phase1._f1(metric.get("SF50_precision"), metric.get("SF50_recall"))
                metric = {
                    **metric,
                    "variant_id": variant,
                    "MV_AP_window": metric.get("MV_AP"),
                    "MV_AP50_window": metric.get("MV_AP50"),
                    "MV_AP25_window": metric.get("MV_AP25"),
                    "score_free_Match50_window": sf50_f1,
                    "score_free_Match50_precision_window": metric.get("SF50_precision"),
                    "score_free_Match50_recall_window": metric.get("SF50_recall"),
                    "same_frame_collision_count": 0,
                    "metric_scope": "local_window_gt_projection",
                }
                metric_rows.append(metric)
                case_rows.extend(cases)
                for row in tops:
                    iou_rows.append(
                        {
                            **row,
                            "variant_id": variant,
                            "mv_iou": row.get("iou", ""),
                            "matrix_scope": "phase3_full_pred_gt_iou_matrix_local_window_support",
                            "full_zero_pairs_omitted": False,
                        }
                    )
    finally:
        recalc._top_iou_rows = original_top
        recalc._mask_dir = original_mask_dir
    return metric_rows, iou_rows, case_rows


def _aggregate(metric_rows: list[dict[str, Any]], generated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gen_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generated_rows:
        gen_by_variant[str(row.get("variant_id", ""))].append(row)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row.get("variant_id", ""))].append(row)
    out = []
    for variant, rows in sorted(grouped.items()):
        gen = gen_by_variant.get(variant, [])
        source_areas = [_num(row.get("source_mask_area")) for row in gen]
        generated_areas = [_num(row.get("generated_mask_area")) for row in gen]
        out.append(
            {
                "variant_id": variant,
                "scene_count": len(rows),
                "mean_MV_AP_window": _mean([_num(row.get("MV_AP_window")) for row in rows]),
                "mean_MV_AP50_window": _mean([_num(row.get("MV_AP50_window")) for row in rows]),
                "mean_MV_AP25_window": _mean([_num(row.get("MV_AP25_window")) for row in rows]),
                "mean_score_free_Match50_window": _mean([_num(row.get("score_free_Match50_window")) for row in rows]),
                "mean_GT_best_IoU_window": _mean([_num(row.get("gt_best_iou_mean")) for row in rows]),
                "source_mask_area_mean": _mean(source_areas),
                "generated_mask_area_mean": _mean(generated_areas),
                "area_shrink_ratio": float(_mean(generated_areas) / max(1.0, _mean(source_areas))),
                "carrier_projection_coverage_rate": _mean([_num(row.get("carrier_projection_coverage_rate")) for row in gen]),
                "support_inside_source_mask_ratio": _mean([_num(row.get("support_inside_source_mask_ratio")) for row in gen]),
                "missing_frame_support_rate": _mean([1.0 if _int(row.get("carrier_support_count")) <= 0 else 0.0 for row in gen]),
                "same_frame_collision_count": int(sum(_int(row.get("same_frame_collision_count")) for row in rows)),
                "uses_gt_for_prediction": any(_bool(row.get("uses_gt_for_prediction")) for row in gen),
                "uses_future": any(_bool(row.get("uses_future")) for row in gen),
                "generated_mask_rows": len(gen),
                "broad_risk_count": int(sum(1 for row in gen if _bool(row.get("broad_risk")))),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reuse_generated:
        heatmap_rows = _read_csv(OUT / "carrier_support_heatmap_rows.csv")
        generated_rows = _read_csv(OUT / "generated_mask_rows.csv")
        eval_rows = _read_csv(OUT / "eval_frame_mask_rows.csv")
        manifest_path = OUT / "generated_mask_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if not eval_rows or not generated_rows:
            raise RuntimeError("--reuse-generated requested but generated/eval CSVs are missing or empty")
    else:
        heatmap_rows, generated_rows, eval_rows, manifest = _generate_masks(args)
        _write_csv(OUT / "carrier_support_heatmap_rows.csv", heatmap_rows)
        _write_csv(OUT / "generated_mask_rows.csv", generated_rows)
        _write_json(OUT / "generated_mask_manifest.json", manifest)
        _write_csv(OUT / "eval_frame_mask_rows.csv", eval_rows)
    metric_rows, iou_rows, case_rows = _evaluate(eval_rows)
    aggregate_rows = _aggregate(metric_rows, generated_rows)
    _write_csv(OUT / "mv_metric_rows.csv", metric_rows)
    _write_csv(OUT / "mv_iou_matrix_rows.csv", iou_rows)
    _write_csv(OUT / "carving_casebook_rows.csv", case_rows)
    _write_csv(OUT / "mv_metric_aggregate_rows.csv", aggregate_rows)

    a0 = next((row for row in aggregate_rows if row.get("variant_id") == "A0_whole_mask_adapter"), {})
    method_candidates = [row for row in aggregate_rows if str(row.get("variant_id", "")).startswith("A") and row.get("variant_id") != "A0_whole_mask_adapter"]
    control_candidates = [row for row in aggregate_rows if str(row.get("variant_id", "")).startswith("C0_")]
    best_carved = max(method_candidates, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    best_control = max(control_candidates, key=lambda row: _num(row.get("mean_MV_AP_window")), default={})
    progress_gate = bool(best_carved) and (
        _num(best_carved.get("mean_MV_AP_window")) >= _num(a0.get("mean_MV_AP_window")) + 0.008
        and _num(best_carved.get("mean_MV_AP50_window")) >= _num(a0.get("mean_MV_AP50_window")) + 0.015
        and _num(best_carved.get("mean_MV_AP_window")) >= _num(best_control.get("mean_MV_AP_window")) + 0.003
        and _int(best_carved.get("same_frame_collision_count")) == 0
        and not _bool(best_carved.get("uses_gt_for_prediction"))
        and not _bool(best_carved.get("uses_future"))
    )
    summary = {
        "phase": "v90_phase3_carrier_supported_carving",
        "schema": "stream4d_v90_phase3_carrier_supported_carving_v1",
        "phase3_pass": bool(metric_rows),
        "runtime_sec": time.time() - t0,
        "gpu_usage_note": "No model forward was run; native carrier support rows already contain UV projections, so Phase3 raster/carving/eval is CPU mask processing.",
        "inputs": {
            "phase1_adapter_rows": _rel(PHASE1_ROOT / "adapter_input_frame_mask_rows.csv"),
            "native_carrier_support_rows": _rel(Path(args.native_support_rows) if Path(args.native_support_rows).is_absolute() else ROOT / args.native_support_rows),
            "window_support_rows": _rel(PHASE0_WINDOWS),
        },
        "row_counts": {
            "carrier_support_heatmap_rows": len(heatmap_rows),
            "generated_mask_rows": len(generated_rows),
            "eval_frame_mask_rows": len(eval_rows),
            "mv_metric_rows": len(metric_rows),
            "mv_iou_matrix_rows": len(iou_rows),
            "carving_casebook_rows": len(case_rows),
        },
        "A0_whole_mask_adapter": a0,
        "best_carved_variant": best_carved.get("variant_id", ""),
        "best_carved_metrics": best_carved,
        "best_control_variant_under_same_carving_protocol": best_control.get("variant_id", ""),
        "best_control_metrics_under_same_carving_protocol": best_control,
        "progress_gate": progress_gate,
        "progress_gate_criteria": {
            "best_carved_MV_AP_window": ">= A0 + 0.008",
            "best_carved_MV_AP50_window": ">= A0 + 0.015",
            "best_carved_MV_AP_window_vs_control": ">= best_control + 0.003",
            "same_frame_collision_count": 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        "support_sparsity": {
            "method_generated_rows": int(sum(1 for row in generated_rows if str(row.get("variant_id", "")).startswith("A"))),
            "method_zero_support_heatmap_rows": int(sum(1 for row in heatmap_rows if str(row.get("variant_id", "")).startswith("A") and _int(row.get("carrier_support_count")) <= 0)),
            "method_heatmap_rows": int(sum(1 for row in heatmap_rows if str(row.get("variant_id", "")).startswith("A"))),
        },
        "outputs": {
            "carrier_support_heatmap_rows": _rel(OUT / "carrier_support_heatmap_rows.csv"),
            "generated_mask_rows": _rel(OUT / "generated_mask_rows.csv"),
            "generated_mask_manifest": _rel(OUT / "generated_mask_manifest.json"),
            "mv_metric_rows": _rel(OUT / "mv_metric_rows.csv"),
            "mv_iou_matrix_rows": _rel(OUT / "mv_iou_matrix_rows.csv"),
            "carving_casebook_rows": _rel(OUT / "carving_casebook_rows.csv"),
            "mv_metric_aggregate_rows": _rel(OUT / "mv_metric_aggregate_rows.csv"),
        },
    }
    _write_json(OUT / "summary.json", summary)
    sha_paths = [
        OUT / "carrier_support_heatmap_rows.csv",
        OUT / "generated_mask_rows.csv",
        OUT / "generated_mask_manifest.json",
        OUT / "eval_frame_mask_rows.csv",
        OUT / "mv_metric_rows.csv",
        OUT / "mv_iou_matrix_rows.csv",
        OUT / "carving_casebook_rows.csv",
        OUT / "mv_metric_aggregate_rows.csv",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in sha_paths if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v90 Phase3 carrier-supported 2D mask carving and MV_AP_window eval.")
    parser.add_argument("--support-point-radius", type=int, default=3)
    parser.add_argument("--min-support-points", type=int, default=1)
    parser.add_argument("--broad-risk-area-ratio", type=float, default=1.25)
    parser.add_argument("--reuse-generated", action="store_true", help="Reuse generated Phase3 masks/adapter CSVs and rerun evaluation only.")
    parser.add_argument(
        "--native-support-rows",
        default=str(DEFAULT_NATIVE_SUPPORT_ROWS.relative_to(ROOT)),
        help="CSV with scene/local_slot/frame/mask carrier UV support rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
