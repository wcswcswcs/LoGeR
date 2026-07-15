#!/usr/bin/env python3
"""Phase5 frozen-birth replay audit for Stream4D v105.

This runner consumes a birth mask bank dumped from the baseline-x gap-adaptive
reference run. It does not run SAM gap segmentation. Each saved birth mask is
replayed at its original anchor frame, propagated forward with the same SAM2
tracker, then per-frame labels are rebuilt and compared against the frozen
reference labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v105" / "baseline_chunk_table" / "baseline_x_gapadaptive_sam2.generated.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

from tools.audit_v105_baseline_x_sam2_twostage_tracking import (  # noqa: E402
    add_masks_to_stream_state,
    infer_stream_frame,
    load_config,
    make_args,
    propagate_new_masks_chunked,
    setup_models,
)
from tools.audit_v105_4dpm_largest_tracking_baseline import (  # noqa: E402
    disjoin_keep_order,
    label_from_id_masks,
    make_numeric_frame_dir,
    make_sheet_grid,
    write_video,
)
from tools.audit_v105_4dpm_style_per_frame_segmentors import (  # noqa: E402
    annotate_frame,
    mask_stats,
    overlay_label,
    parse_frame_ids,
    read_rgb,
    sha256_file,
)
from sgq_v105.sam2_feature_bank import (  # noqa: E402
    Sam2FrameFeatureBank,
    _move_tree,
    _tree_nbytes,
)


class ForwardCounter:
    def __init__(self, label: str, fn: Any) -> None:
        self.label = str(label)
        self.fn = fn
        self.count = 0
        self.total_runtime_sec = 0.0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.time()
        try:
            return self.fn(*args, **kwargs)
        finally:
            self.count += 1
            self.total_runtime_sec += time.time() - t0

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": int(self.count),
            "total_runtime_sec": float(self.total_runtime_sec),
        }


def resolve_path(path_text: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def make_baseline_args(config_path: Path, cli: argparse.Namespace) -> SimpleNamespace:
    config = load_config(config_path)
    baseline_cli = SimpleNamespace(
        config=str(config_path),
        scene_id=cli.scene_id,
        rgb_root=cli.rgb_root,
        frame_start=cli.frame_start,
        frame_stride=cli.frame_stride,
        frame_count=cli.frame_count,
        frame_ids=cli.frame_ids,
        output_root=cli.output_root,
        seed=cli.seed,
        birth_dump_dir="",
    )
    args = make_args(config, baseline_cli)
    args.output_root = str(cli.output_root)
    return args


def load_birth_rows(path: Path, frame_ids: list[int], scene_id: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"birth rows must be a list: {path}")
    payload_frame_ids = [int(v) for v in payload.get("frame_ids", [])]
    if payload_frame_ids and payload_frame_ids != [int(v) for v in frame_ids]:
        raise ValueError(
            f"birth bank frame_ids mismatch: birth={payload_frame_ids[:8]}... run={frame_ids[:8]}..."
        )
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if str(row.get("scene_id")) != str(scene_id):
            raise ValueError(f"birth row {idx} scene mismatch: {row.get('scene_id')} != {scene_id}")
        chunk_idx = int(row["chunk_frame_index"])
        if chunk_idx < 0 or chunk_idx >= len(frame_ids):
            raise ValueError(f"birth row {idx} chunk_frame_index out of range: {chunk_idx}")
        frame_id = int(row["frame_id"])
        if frame_id != int(frame_ids[chunk_idx]):
            raise ValueError(
                f"birth row {idx} frame_id mismatch: row={frame_id} expected={frame_ids[chunk_idx]}"
            )
        mask_path = resolve_path(str(row["mask_path"]))
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        copied = dict(row)
        copied["chunk_frame_index"] = chunk_idx
        copied["frame_id"] = frame_id
        copied["obj_id"] = int(row["obj_id"])
        copied["mask_path"] = str(mask_path)
        out.append(copied)
    out.sort(key=lambda item: (int(item["chunk_frame_index"]), int(item["obj_id"])))
    return out


def _row_mask_area(row: dict[str, Any], h: int, w: int) -> int:
    if row.get("mask_area") is not None:
        return int(row["mask_area"])
    return int(np.count_nonzero(load_mask(Path(row["mask_path"]), h, w)))


def filter_birth_rows_by_anchor_area(
    rows: list[dict[str, Any]],
    *,
    h: int,
    w: int,
    min_area: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_area = max(0, int(min_area))
    if min_area <= 0:
        return rows, {
            "enabled": False,
            "min_birth_mask_area": 0,
            "input_birth_record_count": int(len(rows)),
            "kept_birth_record_count": int(len(rows)),
            "dropped_birth_record_count": 0,
            "dropped_birth_mask_area_sum": 0,
            "dropped_records": [],
        }

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    dropped_area = 0
    for row in rows:
        area = _row_mask_area(row, h, w)
        if area < min_area:
            dropped_area += int(area)
            dropped.append(
                {
                    "chunk_frame_index": int(row["chunk_frame_index"]),
                    "frame_id": int(row["frame_id"]),
                    "obj_id": int(row["obj_id"]),
                    "mask_area": int(area),
                    "source": str(row.get("source", "")),
                    "phase5_role": str(row.get("phase5_role", "")),
                    "reason": "anchor_mask_area_below_min_birth_mask_area",
                }
            )
            continue
        kept.append(row)

    return kept, {
        "enabled": True,
        "min_birth_mask_area": int(min_area),
        "input_birth_record_count": int(len(rows)),
        "kept_birth_record_count": int(len(kept)),
        "dropped_birth_record_count": int(len(dropped)),
        "dropped_birth_mask_area_sum": int(dropped_area),
        "dropped_records": dropped[:256],
        "dropped_records_truncated": bool(len(dropped) > 256),
    }


def filter_current_rows_by_area(
    rows: list[tuple[int, np.ndarray]],
    *,
    min_area: int,
    min_component_area: int,
) -> tuple[list[tuple[int, np.ndarray]], dict[str, Any]]:
    min_area = max(0, int(min_area))
    min_component_area = max(0, int(min_component_area))
    if min_area <= 0 and min_component_area <= 0:
        return rows, {
            "enabled": False,
            "min_output_mask_area": 0,
            "min_output_component_area": 0,
            "input_mask_count": int(len(rows)),
            "kept_mask_count": int(len(rows)),
            "dropped_mask_count": 0,
            "dropped_mask_area_sum": 0,
            "removed_small_component_count": 0,
            "removed_small_component_pixel_sum": 0,
            "dropped_obj_ids": [],
        }

    kept: list[tuple[int, np.ndarray]] = []
    dropped: list[dict[str, Any]] = []
    dropped_area = 0
    removed_small_component_count = 0
    removed_small_component_pixels = 0
    component_cleaned_records: list[dict[str, Any]] = []
    for obj_id, mask in rows:
        mask_bool = mask.astype(bool)
        area_before = int(np.count_nonzero(mask_bool))
        if min_area > 0 and area_before < min_area:
            dropped_area += int(area_before)
            dropped.append(
                {
                    "obj_id": int(obj_id),
                    "mask_area": int(area_before),
                    "post_component_area": int(area_before),
                    "reason": "mask_area_below_min_output_mask_area",
                }
            )
            continue
        if min_component_area > 0 and area_before > 0:
            num_labels, comp, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
            keep_labels: list[int] = []
            dropped_component_count = 0
            dropped_component_pixels = 0
            for label_idx in range(1, int(num_labels)):
                comp_area = int(stats[label_idx, cv2.CC_STAT_AREA])
                if comp_area >= min_component_area:
                    keep_labels.append(int(label_idx))
                else:
                    dropped_component_count += 1
                    dropped_component_pixels += int(comp_area)
            if dropped_component_count > 0:
                removed_small_component_count += int(dropped_component_count)
                removed_small_component_pixels += int(dropped_component_pixels)
                component_cleaned_records.append(
                    {
                        "obj_id": int(obj_id),
                        "component_count": int(max(0, num_labels - 1)),
                        "kept_component_count": int(len(keep_labels)),
                        "dropped_component_count": int(dropped_component_count),
                        "dropped_component_pixels": int(dropped_component_pixels),
                        "mask_area_before": int(area_before),
                    }
                )
            if keep_labels:
                mask_bool = np.isin(comp, np.asarray(keep_labels, dtype=comp.dtype))
            else:
                mask_bool = np.zeros_like(mask_bool, dtype=bool)
        area_after = int(np.count_nonzero(mask_bool))
        if min_area > 0 and area_after < min_area:
            dropped_area += int(area_after)
            dropped.append(
                {
                    "obj_id": int(obj_id),
                    "mask_area": int(area_before),
                    "post_component_area": int(area_after),
                    "reason": "post_component_area_below_min_output_mask_area",
                }
            )
            continue
        kept.append((int(obj_id), mask_bool.astype(bool)))
    return kept, {
        "enabled": True,
        "min_output_mask_area": int(min_area),
        "min_output_component_area": int(min_component_area),
        "input_mask_count": int(len(rows)),
        "kept_mask_count": int(len(kept)),
        "dropped_mask_count": int(len(dropped)),
        "dropped_mask_area_sum": int(dropped_area),
        "removed_small_component_count": int(removed_small_component_count),
        "removed_small_component_pixel_sum": int(removed_small_component_pixels),
        "component_cleaned_records": component_cleaned_records[:128],
        "component_cleaned_records_truncated": bool(len(component_cleaned_records) > 128),
        "dropped_obj_ids": [int(row["obj_id"]) for row in dropped],
        "dropped_records": dropped[:128],
        "dropped_records_truncated": bool(len(dropped) > 128),
    }


def load_mask(path: Path, h: int, w: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    mask = img > 0
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask.astype(bool)


def order_rows_for_ownership(
    rows: list[tuple[int, np.ndarray]],
    *,
    priority: str,
) -> list[tuple[int, np.ndarray]]:
    if str(priority) == "current":
        return rows
    if str(priority) == "area_ascending":
        return sorted(rows, key=lambda item: (int(np.count_nonzero(item[1])), int(item[0])))
    if str(priority) == "area_descending":
        return sorted(rows, key=lambda item: (-int(np.count_nonzero(item[1])), int(item[0])))
    raise ValueError(f"unsupported ownership priority: {priority}")


def load_reference_labels(summary_path: Path) -> dict[int, Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    labels: dict[int, Path] = {}
    for row in summary.get("records", []):
        frame_id = int(row["frame_id"])
        label_path = resolve_path(str(row["label_path"]))
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        labels[frame_id] = label_path
    return labels


def _role_set_for_row(row: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    explicit = row.get("phase5_role")
    if explicit:
        roles.add(str(explicit))
    for item in row.get("phase5_merged_roles", []) or []:
        roles.add(str(item))
    source = str(row.get("source", ""))
    if "birth_new" in source:
        roles.add("birth_new")
    if "repair_existing" in source:
        roles.add("repair_existing")
    if "handoff" in source or "phase3" in source:
        roles.add("inherited")
    if not roles and source in {"gap_birth", "frame0_seed"}:
        roles.add("birth_new")
    return roles


def _birth_role_index(birth_rows: list[dict[str, Any]]) -> dict[str, Any]:
    roles_by_obj: dict[int, set[str]] = defaultdict(set)
    anchors_by_obj: dict[int, int] = {}
    for row in birth_rows:
        obj_id = int(row["obj_id"])
        roles_by_obj[obj_id].update(_role_set_for_row(row))
        anchors_by_obj[obj_id] = min(
            int(row["chunk_frame_index"]),
            int(anchors_by_obj.get(obj_id, int(row["chunk_frame_index"]))),
        )
    new_birth_ids = sorted(obj_id for obj_id, roles in roles_by_obj.items() if "birth_new" in roles)
    inherited_ids = sorted(
        obj_id
        for obj_id, roles in roles_by_obj.items()
        if ("inherited" in roles or "repair_existing" in roles) and "birth_new" not in roles
    )
    return {
        "roles_by_obj": {int(k): sorted(v) for k, v in roles_by_obj.items()},
        "anchors_by_obj": {int(k): int(v) for k, v in anchors_by_obj.items()},
        "new_birth_ids": [int(v) for v in new_birth_ids],
        "inherited_or_repaired_ids": [int(v) for v in inherited_ids],
    }


def _mask_overlap_coeff(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    area_a = int(np.count_nonzero(a))
    area_b = int(np.count_nonzero(b))
    inter = int(np.count_nonzero(a & b))
    denom = max(1, min(area_a, area_b))
    return {
        "intersection_pixels": int(inter),
        "a_area": int(area_a),
        "b_area": int(area_b),
        "overlap_coefficient": float(inter / denom),
    }


def duplicate_rebirth_audit_from_pre_disjoin(
    *,
    per_frame_rows: list[list[tuple[int, np.ndarray]]],
    role_index: dict[str, Any],
    duplicate_window_frames: int,
    duplicate_overlap_threshold: float,
) -> dict[str, Any]:
    new_birth_ids = {int(v) for v in role_index.get("new_birth_ids", [])}
    inherited_ids = {int(v) for v in role_index.get("inherited_or_repaired_ids", [])}
    anchors_by_obj = {int(k): int(v) for k, v in role_index.get("anchors_by_obj", {}).items()}
    best_by_new: dict[int, dict[str, Any]] = {}
    volume_by_new: dict[int, int] = defaultdict(int)

    for frame_idx, rows in enumerate(per_frame_rows):
        by_obj: dict[int, list[np.ndarray]] = defaultdict(list)
        for obj_id, mask in rows:
            by_obj[int(obj_id)].append(mask.astype(bool))
        inherited_masks: list[tuple[int, np.ndarray]] = []
        for obj_id in sorted(inherited_ids):
            masks = by_obj.get(int(obj_id), [])
            if masks:
                merged = np.logical_or.reduce(np.stack(masks, axis=0))
                inherited_masks.append((int(obj_id), merged.astype(bool)))
        if not inherited_masks:
            continue
        for new_id in sorted(new_birth_ids):
            anchor = int(anchors_by_obj.get(int(new_id), 0))
            if frame_idx < anchor or frame_idx > anchor + int(duplicate_window_frames):
                continue
            masks = by_obj.get(int(new_id), [])
            if not masks:
                continue
            new_mask = np.logical_or.reduce(np.stack(masks, axis=0)).astype(bool)
            new_area = int(np.count_nonzero(new_mask))
            volume_by_new[int(new_id)] += int(new_area)
            for inherited_id, inherited_mask in inherited_masks:
                overlap = _mask_overlap_coeff(new_mask, inherited_mask)
                record = {
                    "new_birth_obj_id": int(new_id),
                    "inherited_obj_id": int(inherited_id),
                    "frame_index": int(frame_idx),
                    **overlap,
                }
                previous = best_by_new.get(int(new_id))
                if previous is None or float(record["overlap_coefficient"]) > float(previous["overlap_coefficient"]):
                    best_by_new[int(new_id)] = record

    duplicate_records = []
    duplicate_ids: set[int] = set()
    duplicate_volume = 0
    total_volume = 0
    for new_id in sorted(new_birth_ids):
        volume = int(volume_by_new.get(int(new_id), 0))
        total_volume += volume
        best = best_by_new.get(int(new_id), {
            "new_birth_obj_id": int(new_id),
            "inherited_obj_id": None,
            "frame_index": None,
            "intersection_pixels": 0,
            "a_area": 0,
            "b_area": 0,
            "overlap_coefficient": 0.0,
        })
        is_duplicate = float(best.get("overlap_coefficient", 0.0)) >= float(duplicate_overlap_threshold)
        if is_duplicate:
            duplicate_ids.add(int(new_id))
            duplicate_volume += volume
        duplicate_records.append(
            {
                **best,
                "new_birth_volume_pixels_in_window": int(volume),
                "is_duplicate_rebirth": bool(is_duplicate),
            }
        )

    return {
        "schema_version": "stream4d_phase5_pre_disjoin_duplicate_rebirth_audit_v1",
        "duplicate_window_frames": int(duplicate_window_frames),
        "duplicate_overlap_threshold": float(duplicate_overlap_threshold),
        "feature_similarity_available": False,
        "feature_similarity_used": False,
        "new_birth_id_count": int(len(new_birth_ids)),
        "inherited_or_repaired_id_count": int(len(inherited_ids)),
        "duplicate_birth_id_count": int(len(duplicate_ids)),
        "duplicate_birth_ids": [int(v) for v in sorted(duplicate_ids)],
        "drr_count": float(len(duplicate_ids) / len(new_birth_ids)) if new_birth_ids else 0.0,
        "new_birth_volume_pixels_in_window": int(total_volume),
        "duplicate_birth_volume_pixels_in_window": int(duplicate_volume),
        "drr_area": float(duplicate_volume / total_volume) if total_volume > 0 else 0.0,
        "records": duplicate_records,
    }


def visible_ids(label: np.ndarray) -> set[int]:
    ids = np.unique(label.astype(np.int64))
    return {int(v) - 1 for v in ids.tolist() if int(v) > 0}


def compare_labels(pred: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    if pred.shape != ref.shape:
        raise ValueError(f"label shape mismatch: pred={pred.shape} ref={ref.shape}")
    pred_fg = pred > 0
    ref_fg = ref > 0
    union = pred_fg | ref_fg
    inter = pred_fg & ref_fg
    fg_iou = 1.0 if not np.any(union) else float(np.count_nonzero(inter)) / float(np.count_nonzero(union))
    pred_ids = visible_ids(pred)
    ref_ids = visible_ids(ref)
    id_union = pred_ids | ref_ids
    id_inter = pred_ids & ref_ids
    per_id_ious = []
    for obj_id in sorted(id_union):
        pred_obj = pred == int(obj_id) + 1
        ref_obj = ref == int(obj_id) + 1
        obj_union = pred_obj | ref_obj
        if np.any(obj_union):
            per_id_ious.append(float(np.count_nonzero(pred_obj & ref_obj)) / float(np.count_nonzero(obj_union)))
    return {
        "exact_equal": bool(np.array_equal(pred, ref)),
        "pixel_equal_ratio": float(np.mean(pred == ref)),
        "foreground_iou": float(fg_iou),
        "pred_visible_id_count": int(len(pred_ids)),
        "ref_visible_id_count": int(len(ref_ids)),
        "visible_id_jaccard": float(len(id_inter) / len(id_union)) if id_union else 1.0,
        "missing_ref_ids": [int(v) for v in sorted(ref_ids - pred_ids)],
        "extra_pred_ids": [int(v) for v in sorted(pred_ids - ref_ids)],
        "min_per_id_iou": float(min(per_id_ious)) if per_id_ious else 1.0,
        "mean_per_id_iou": float(np.mean(per_id_ious)) if per_id_ious else 1.0,
    }


def empty_feature_bank_summary() -> dict[str, Any]:
    return {
        "enabled": False,
        "storage_device": "",
        "video_gpu_hot_window": 0,
        "build_runtime_sec": 0.0,
        "video_bank_summary": None,
        "bind_stats": {},
        "forward_counter": None,
    }


def install_video_feature_bank_patch(
    tracker_model: Any,
    *,
    frame_ids: list[int],
    frame_paths: list[Path],
    storage_device: str,
    video_gpu_hot_window: int,
) -> dict[str, Any]:
    import torch

    video_gpu_hot_window = max(int(video_gpu_hot_window), 0)
    video_bank = Sam2FrameFeatureBank(storage_device=str(storage_device), clone_tensors=True)
    build_t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        video_bank.build_for_video_paths(tracker_model, frame_ids=frame_ids, frame_paths=frame_paths)
    build_runtime_sec = time.time() - build_t0

    original_get_image_feature = tracker_model._get_image_feature
    forward_counter = ForwardCounter(
        "phase5_video_feature_bank_video_predictor.forward_image",
        tracker_model.forward_image,
    )
    tracker_model.forward_image = forward_counter  # type: ignore[method-assign]

    bind_stats = {
        "video_feature_inject_count": 0,
        "video_feature_inject_miss_count": 0,
        "video_feature_cache_hit_count": 0,
        "video_feature_cache_evict_count": 0,
        "video_feature_h2d_bytes": 0,
        "video_gpu_hot_window": int(video_gpu_hot_window),
    }

    def bank_get_image_feature(inference_state: dict[str, Any], frame_idx: int, batch_size: int) -> Any:
        local_index = int(frame_idx)
        frame_offset = int(inference_state.get("_v105_feature_bank_frame_offset", 0) or 0)
        chunk_index = int(local_index + frame_offset)
        if chunk_index in video_bank.records_by_chunk_index:
            device = inference_state.get("device", video_bank.storage_device)
            cache = inference_state.setdefault("cached_features", {})
            hot_order = inference_state.setdefault("_v105_phase5_feature_bank_hot_order", OrderedDict())
            if not isinstance(hot_order, OrderedDict):
                hot_order = OrderedDict((int(k), None) for k in cache.keys())
                inference_state["_v105_phase5_feature_bank_hot_order"] = hot_order
            cache_key = int(local_index)
            if cache_key in cache:
                bind_stats["video_feature_cache_hit_count"] += 1
                hot_order[cache_key] = None
                hot_order.move_to_end(cache_key)
            else:
                record = video_bank.get_chunk_features(chunk_index)
                image = record.input_image.to(device, non_blocking=True)
                backbone_out = _move_tree(record.backbone_out, device)
                moved_bytes = _tree_nbytes(image) + _tree_nbytes(backbone_out)
                record_device = str(record.input_image.device)
                target_device = str(device)
                if not record_device.startswith(target_device):
                    video_bank.feature_bank_h2d_bytes += int(moved_bytes)
                    bind_stats["video_feature_h2d_bytes"] += int(moved_bytes)
                cache[cache_key] = (image, backbone_out)
                hot_order[cache_key] = None
                hot_order.move_to_end(cache_key)
                bind_stats["video_feature_inject_count"] += 1
            if video_gpu_hot_window > 0:
                while len(hot_order) > video_gpu_hot_window:
                    evict_idx, _ = hot_order.popitem(last=False)
                    if int(evict_idx) == cache_key:
                        hot_order[int(evict_idx)] = None
                        hot_order.move_to_end(int(evict_idx))
                        break
                    if int(evict_idx) in cache:
                        cache.pop(int(evict_idx), None)
                        bind_stats["video_feature_cache_evict_count"] += 1
        else:
            bind_stats["video_feature_inject_miss_count"] += 1
        return original_get_image_feature(inference_state, int(frame_idx), int(batch_size))

    tracker_model._get_image_feature = bank_get_image_feature  # type: ignore[method-assign]
    return {
        "enabled": True,
        "storage_device": str(storage_device),
        "video_gpu_hot_window": int(video_gpu_hot_window),
        "build_runtime_sec": float(build_runtime_sec),
        "video_bank": video_bank,
        "bind_stats": bind_stats,
        "forward_counter": forward_counter,
    }


def serializable_feature_bank_summary(feature_bank_info: dict[str, Any]) -> dict[str, Any]:
    if not bool(feature_bank_info.get("enabled", False)):
        return empty_feature_bank_summary()
    video_bank = feature_bank_info["video_bank"]
    forward_counter = feature_bank_info["forward_counter"]
    return {
        "enabled": True,
        "storage_device": str(feature_bank_info.get("storage_device", "")),
        "video_gpu_hot_window": int(feature_bank_info.get("video_gpu_hot_window", 0) or 0),
        "build_runtime_sec": float(feature_bank_info.get("build_runtime_sec", 0.0) or 0.0),
        "video_bank_summary": video_bank.summary(),
        "bind_stats": dict(feature_bank_info.get("bind_stats", {})),
        "forward_counter": forward_counter.summary(),
    }


def pad_tensor_dim0(tensor: Any, target: int, fill_value: float) -> Any:
    import torch

    old = int(tensor.shape[0])
    if old >= int(target):
        return tensor
    pad_shape = (int(target) - old, *tensor.shape[1:])
    pad = torch.full(
        pad_shape,
        fill_value=float(fill_value),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=0)


def reconcile_stream_state_object_count(
    predictor: Any,
    state: dict[str, Any],
    *,
    repair_mode: str,
) -> list[dict[str, Any]]:
    """Pad prior SAM2 state outputs after adding new objects mid-stream."""
    predictor.propagate_in_video_preflight(state)
    batch_size = int(predictor._get_obj_num(state))
    repaired: list[dict[str, Any]] = []
    for storage_key, is_cond in (("cond_frame_outputs", True), ("non_cond_frame_outputs", False)):
        output_dict = state["output_dict"][storage_key]
        for frame_idx, out in list(output_dict.items()):
            obj_ptr = out.get("obj_ptr")
            if obj_ptr is None or int(obj_ptr.shape[0]) == batch_size:
                continue
            old_count = int(obj_ptr.shape[0])
            if str(repair_mode) == "reconsolidate":
                consolidated = predictor._consolidate_temp_output_across_obj(
                    state,
                    int(frame_idx),
                    is_cond=bool(is_cond),
                    run_mem_encoder=True,
                )
                output_dict[int(frame_idx)] = consolidated
                predictor._add_output_per_object(state, int(frame_idx), consolidated, storage_key)
            elif str(repair_mode) == "pad":
                if "pred_masks" in out and out["pred_masks"] is not None:
                    out["pred_masks"] = pad_tensor_dim0(out["pred_masks"], batch_size, -1024.0)
                if "pred_masks_video_res" in out and out["pred_masks_video_res"] is not None:
                    out["pred_masks_video_res"] = pad_tensor_dim0(out["pred_masks_video_res"], batch_size, -1024.0)
                if out.get("maskmem_features") is not None:
                    out["maskmem_features"] = pad_tensor_dim0(out["maskmem_features"], batch_size, 0.0)
                if out.get("maskmem_pos_enc") is not None:
                    out["maskmem_pos_enc"] = [
                        pad_tensor_dim0(pos, batch_size, 0.0)
                        for pos in out["maskmem_pos_enc"]
                    ]
                out["obj_ptr"] = pad_tensor_dim0(out["obj_ptr"], batch_size, 0.0)
                if out.get("object_score_logits") is not None:
                    out["object_score_logits"] = pad_tensor_dim0(out["object_score_logits"], batch_size, -10.0)
                predictor._add_output_per_object(state, int(frame_idx), out, storage_key)
            else:
                raise ValueError(f"unsupported stream state repair mode: {repair_mode}")
            repaired.append(
                {
                    "storage_key": str(storage_key),
                    "frame_idx": int(frame_idx),
                    "old_obj_count": int(old_count),
                    "new_obj_count": int(batch_size),
                    "repair_mode": str(repair_mode),
                }
            )
    return repaired


def run(cli: argparse.Namespace) -> None:
    import torch

    config_path = resolve_path(cli.config)
    args = make_baseline_args(config_path, cli)
    args.scene_id = str(cli.scene_id or args.scene_id)
    args.output_root = str(cli.output_root)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    frame_ids = parse_frame_ids(str(args.frame_ids), int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    rgb_root = resolve_path(args.rgb_root) / args.scene_id / "color"
    frame_paths = [rgb_root / f"{frame_id}.jpg" for frame_id in frame_ids]
    missing = [str(path) for path in frame_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[:5])

    output_root = resolve_path(cli.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    label_dir = output_root / "labels"
    overlay_dir = output_root / "overlays"
    sheet_dir = output_root / "sheets"
    video_dir_out = output_root / "videos"
    for directory in (label_dir, overlay_dir, sheet_dir, video_dir_out):
        directory.mkdir(parents=True, exist_ok=True)

    rgbs = [read_rgb(path) for path in frame_paths]
    h, w = rgbs[0].shape[:2]
    video_dir = make_numeric_frame_dir(frame_paths, output_root)
    birth_records_path = resolve_path(cli.birth_records)
    reference_summary_path = resolve_path(cli.reference_summary)
    birth_rows = load_birth_rows(birth_records_path, frame_ids, args.scene_id)
    input_birth_record_count = int(len(birth_rows))
    birth_rows, birth_row_area_filter = filter_birth_rows_by_anchor_area(
        birth_rows,
        h=h,
        w=w,
        min_area=int(cli.min_birth_mask_area),
    )
    reference_labels = load_reference_labels(reference_summary_path)
    missing_ref = [frame_id for frame_id in frame_ids if int(frame_id) not in reference_labels]
    if missing_ref:
        raise FileNotFoundError(f"missing reference labels for frames: {missing_ref[:8]}")

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in birth_rows:
        groups[int(row["chunk_frame_index"])].append(row)
    birth_role_index = _birth_role_index(birth_rows)

    raw_by_frame: list[list[tuple[int, np.ndarray]]] = [[] for _ in frame_ids]
    propagated_by_frame: list[list[tuple[int, np.ndarray]]] = [[] for _ in frame_ids]

    t_setup = time.time()
    models = setup_models(args)
    setup_sec = time.time() - t_setup
    tracker_model = models["tracker_model"]
    feature_bank_info = empty_feature_bank_summary()
    video_state_template: dict[str, Any] | None = None
    video_state_template_init_sec = 0.0
    state_template_clone_count = 0

    total_tracking_sec = 0.0
    group_records: list[dict[str, Any]] = []
    scheduler_mode = str(cli.scheduler_mode)
    total_t0 = time.time()
    if bool(cli.use_video_feature_bank):
        feature_bank_info = install_video_feature_bank_patch(
            tracker_model,
            frame_ids=frame_ids,
            frame_paths=frame_paths,
            storage_device=str(cli.video_feature_bank_storage_device),
            video_gpu_hot_window=int(cli.video_gpu_hot_window),
        )
    if bool(cli.reuse_video_state_template):
        if scheduler_mode != "independent_anchor":
            raise ValueError("--reuse-video-state-template is only implemented for --scheduler-mode independent_anchor")
        import torch

        template_t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            video_state_template = tracker_model.init_state(
                video_path=str(video_dir),
                offload_video_to_cpu=bool(args.offload_video_to_cpu),
                offload_state_to_cpu=bool(args.offload_state_to_cpu),
                async_loading_frames=False,
            )
        video_state_template["cached_features"] = {}
        video_state_template_init_sec = time.time() - template_t0
        total_tracking_sec += float(video_state_template_init_sec)
    if scheduler_mode == "independent_anchor":
        for anchor_idx in sorted(groups):
            rows = sorted(groups[anchor_idx], key=lambda item: int(item["obj_id"]))
            obj_ids = np.asarray([int(row["obj_id"]) for row in rows], dtype=np.int64)
            masks = np.stack([load_mask(Path(row["mask_path"]), h, w) for row in rows], axis=0).astype(bool)
            for obj_id, mask in zip(obj_ids.tolist(), masks, strict=False):
                raw_by_frame[anchor_idx].append((int(obj_id), mask.astype(bool)))

            prop_t0 = time.time()
            propagated = {}
            chunk_runtime_records: list[dict[str, Any]] = []
            if int(anchor_idx) < len(frame_ids) - 1:
                propagated = propagate_new_masks_chunked(
                    tracker_model,
                    tracker=str(args.tracker_backend),
                    video_dir=video_dir,
                    seed_frame=int(anchor_idx),
                    obj_ids=obj_ids,
                    masks=masks,
                    total_frames=len(frame_ids),
                    offload_video_to_cpu=bool(args.offload_video_to_cpu),
                    offload_state_to_cpu=bool(args.offload_state_to_cpu),
                    chunk_size=int(args.propagation_chunk_size),
                    video_state_template=video_state_template,
                    chunk_runtime_records=chunk_runtime_records,
                )
            prop_sec = time.time() - prop_t0
            total_tracking_sec += prop_sec
            if video_state_template is not None:
                state_template_clone_count += int(len(chunk_runtime_records))

            propagated_future_rows = 0
            for frame_idx, frame_outputs in propagated.items():
                if int(frame_idx) <= int(anchor_idx):
                    continue
                if int(frame_idx) >= len(frame_ids):
                    continue
                for obj_id, mask in frame_outputs.items():
                    propagated_by_frame[int(frame_idx)].append((int(obj_id), mask.astype(bool)))
                    propagated_future_rows += 1

            group_record = {
                "anchor_chunk_frame_index": int(anchor_idx),
                "anchor_frame_id": int(frame_ids[anchor_idx]),
                "birth_count": int(len(rows)),
                "obj_id_min": int(obj_ids.min()) if obj_ids.size else None,
                "obj_id_max": int(obj_ids.max()) if obj_ids.size else None,
                "propagation_runtime_sec": float(prop_sec),
                "future_output_mask_count": int(propagated_future_rows),
                "reuse_video_state_template": bool(video_state_template is not None),
                "state_template_chunk_runtime_records": chunk_runtime_records,
            }
            group_records.append(group_record)
            print(json.dumps(group_record, ensure_ascii=True), flush=True)
    elif scheduler_mode == "streaming_state":
        import torch

        for anchor_idx in sorted(groups):
            rows = sorted(groups[anchor_idx], key=lambda item: int(item["obj_id"]))
            for row in rows:
                mask = load_mask(Path(row["mask_path"]), h, w)
                raw_by_frame[int(anchor_idx)].append((int(row["obj_id"]), mask.astype(bool)))

        stream_t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            stream_state = tracker_model.init_state(
                video_path=str(video_dir),
                offload_video_to_cpu=bool(args.offload_video_to_cpu),
                offload_state_to_cpu=bool(args.offload_state_to_cpu),
                async_loading_frames=False,
            )
        init_sec = time.time() - stream_t0
        total_tracking_sec += init_sec
        group_records.append(
            {
                "scheduler_mode": "streaming_state",
                "stream_state_init_runtime_sec": float(init_sec),
                "anchor_group_count": int(len(groups)),
            }
        )
        for chunk_idx in range(len(frame_ids)):
            infer_sec = 0.0
            if chunk_idx > 0:
                infer_t0 = time.time()
                current_ids, current_masks = infer_stream_frame(
                    tracker_model,
                    stream_state,
                    frame_idx=int(chunk_idx),
                )
                infer_sec = time.time() - infer_t0
                total_tracking_sec += infer_sec
                for obj_id, mask in zip(current_ids.tolist(), current_masks.astype(bool), strict=False):
                    propagated_by_frame[chunk_idx].append((int(obj_id), mask.astype(bool)))

            raw_rows = sorted(raw_by_frame[chunk_idx], key=lambda item: int(item[0]))
            if raw_rows:
                obj_ids = np.asarray([int(item[0]) for item in raw_rows], dtype=np.int64)
                masks = np.stack([item[1].astype(bool) for item in raw_rows], axis=0)
                add_t0 = time.time()
                add_masks_to_stream_state(
                    tracker_model,
                    stream_state,
                    tracker=str(args.tracker_backend),
                    frame_idx=int(chunk_idx),
                    obj_ids=obj_ids,
                    masks=masks,
                )
                repaired_shapes = reconcile_stream_state_object_count(
                    tracker_model,
                    stream_state,
                    repair_mode=str(cli.stream_state_repair_mode),
                )
                add_sec = time.time() - add_t0
                total_tracking_sec += add_sec
            else:
                add_sec = 0.0
                obj_ids = np.zeros((0,), dtype=np.int64)
                repaired_shapes = []
            frame_record = {
                "scheduler_mode": "streaming_state",
                "chunk_frame_index": int(chunk_idx),
                "frame_id": int(frame_ids[chunk_idx]),
                "raw_birth_count": int(len(raw_rows)),
                "raw_obj_id_min": int(obj_ids.min()) if obj_ids.size else None,
                "raw_obj_id_max": int(obj_ids.max()) if obj_ids.size else None,
                "infer_runtime_sec": float(infer_sec),
                "add_birth_runtime_sec": float(add_sec),
                "state_shape_repair_count": int(len(repaired_shapes)),
                "state_shape_repairs": repaired_shapes[:16],
            }
            group_records.append(frame_record)
            print(json.dumps(frame_record, ensure_ascii=True), flush=True)
        try:
            tracker_model.reset_state(stream_state)
        except Exception:
            pass
        try:
            stream_state.clear()
        except Exception:
            pass
    else:
        raise ValueError(f"unsupported scheduler_mode={scheduler_mode}")

    overlay_paths: list[Path] = []
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    pre_disjoin_rows_for_audit: list[list[tuple[int, np.ndarray]]] = []
    for chunk_idx, (frame_id, rgb) in enumerate(zip(frame_ids, rgbs, strict=True)):
        propagated_rows = sorted(propagated_by_frame[chunk_idx], key=lambda item: int(item[0]))
        raw_rows = sorted(raw_by_frame[chunk_idx], key=lambda item: int(item[0]))
        if bool(cli.anchor_birth_priority) and raw_rows:
            rows = raw_rows + propagated_rows
        else:
            rows = propagated_rows + raw_rows
            rows.sort(key=lambda item: int(item[0]))
        rows = order_rows_for_ownership(rows, priority=str(cli.ownership_priority))
        rows, output_mask_area_filter = filter_current_rows_by_area(
            rows,
            min_area=int(cli.min_output_mask_area),
            min_component_area=int(cli.min_output_component_area),
        )
        pre_disjoin_rows_for_audit.append([(int(obj_id), mask.astype(bool)) for obj_id, mask in rows])
        if rows:
            obj_ids = np.asarray([int(item[0]) for item in rows], dtype=np.int64)
            masks_pre = np.stack([item[1].astype(bool) for item in rows], axis=0)
            masks_all, keep = disjoin_keep_order(masks_pre, h, w, empty_ratio=float(args.empty_ratio))
            masks = masks_all[keep]
            obj_ids = obj_ids[keep]
        else:
            obj_ids = np.zeros((0,), dtype=np.int64)
            masks_pre = np.zeros((0, h, w), dtype=bool)
            masks = np.zeros((0, h, w), dtype=bool)

        label = label_from_id_masks(obj_ids, masks, h, w)
        label_path = label_dir / f"frame_{int(frame_id):06d}.png"
        cv2.imwrite(str(label_path), label)
        ref_label = cv2.imread(str(reference_labels[int(frame_id)]), cv2.IMREAD_UNCHANGED)
        if ref_label is None:
            raise FileNotFoundError(reference_labels[int(frame_id)])
        if ref_label.ndim == 3:
            ref_label = ref_label[:, :, 0]
        compare = compare_labels(label.astype(np.uint16), ref_label.astype(np.uint16))
        compare.update({"chunk_frame_index": int(chunk_idx), "frame_id": int(frame_id)})
        comparisons.append(compare)

        overlay = overlay_label(rgb, label)
        stats = mask_stats(label)
        annotated = annotate_frame(
            overlay,
            f"phase5 frozen-birth replay frame {chunk_idx:02d} / id {int(frame_id)}",
            [
                f"masks={stats['visible_id_count']} fg={stats['foreground_ratio']:.3f} ids={int(obj_ids.size)}",
                f"eq={compare['exact_equal']} fg_iou={compare['foreground_iou']:.4f}",
            ],
        )
        overlay_path = overlay_dir / f"frame_{chunk_idx:02d}_id_{int(frame_id):06d}.jpg"
        annotated.save(overlay_path, quality=95)
        overlay_paths.append(overlay_path)
        records.append(
            {
                "chunk_frame_index": int(chunk_idx),
                "frame_id": int(frame_id),
                "label_path": str(label_path),
                "reference_label_path": str(reference_labels[int(frame_id)]),
                "overlay_path": str(overlay_path),
                "object_id_count": int(obj_ids.size),
                "pre_disjoin_mask_count": int(masks_pre.shape[0]),
                "visible_id_count": int(stats["visible_id_count"]),
                "foreground_ratio": float(stats["foreground_ratio"]),
                "output_mask_area_filter": output_mask_area_filter,
            }
        )

    sheet_paths: list[str] = []
    for start in range(0, len(overlay_paths), 8):
        part = overlay_paths[start : start + 8]
        end = start + len(part) - 1
        sheet_path = sheet_dir / f"phase5_frozen_birth_replay_{args.scene_id}_frames_{start:02d}_{end:02d}_4x2.jpg"
        make_sheet_grid(part, sheet_path, int(args.sheet_cell_width), cols=4)
        sheet_paths.append(str(sheet_path))
    video_path = video_dir_out / f"phase5_frozen_birth_replay_{args.scene_id}_chunk0.mp4"
    write_video(overlay_paths, video_path, fps=float(args.fps))

    reference_summary = json.loads(reference_summary_path.read_text(encoding="utf-8"))
    reference_runtime = reference_summary.get("total_runtime_sec")
    total_sec = time.time() - total_t0
    exact_frame_count = sum(1 for row in comparisons if row["exact_equal"])
    fg_ious = [float(row["foreground_iou"]) for row in comparisons]
    per_id_mins = [float(row["min_per_id_iou"]) for row in comparisons]
    pixel_equal = [float(row["pixel_equal_ratio"]) for row in comparisons]
    duplicate_rebirth_audit = duplicate_rebirth_audit_from_pre_disjoin(
        per_frame_rows=pre_disjoin_rows_for_audit,
        role_index=birth_role_index,
        duplicate_window_frames=int(cli.duplicate_window_frames),
        duplicate_overlap_threshold=float(cli.duplicate_overlap_threshold),
    )
    peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    summary = {
        "schema_version": "stream4d_v105_phase5_frozen_birth_replay_summary_v2",
        "scene_id": str(args.scene_id),
        "config_path": str(config_path),
        "birth_records_path": str(birth_records_path),
        "birth_records_sha256": sha256_file(birth_records_path),
        "reference_summary_path": str(reference_summary_path),
        "reference_summary_sha256": sha256_file(reference_summary_path),
        "frame_ids": [int(v) for v in frame_ids],
        "frame_count": int(len(frame_ids)),
        "input_birth_record_count": int(input_birth_record_count),
        "birth_record_count": int(len(birth_rows)),
        "birth_row_area_filter": birth_row_area_filter,
        "min_output_mask_area": int(cli.min_output_mask_area),
        "min_output_component_area": int(cli.min_output_component_area),
        "anchor_group_count": int(len(groups)),
        "setup_sec": float(setup_sec),
        "total_runtime_sec": float(total_sec),
        "total_tracking_runtime_sec": float(total_tracking_sec),
        "reference_total_runtime_sec": float(reference_runtime) if reference_runtime is not None else None,
        "runtime_ratio_vs_reference": (
            float(total_sec) / float(reference_runtime)
            if reference_runtime is not None and float(reference_runtime) > 0.0
            else None
        ),
        "all_labels_exact_equal": bool(exact_frame_count == len(comparisons)),
        "exact_label_frame_count": int(exact_frame_count),
        "min_pixel_equal_ratio": float(min(pixel_equal)) if pixel_equal else 1.0,
        "mean_pixel_equal_ratio": float(np.mean(pixel_equal)) if pixel_equal else 1.0,
        "min_foreground_iou": float(min(fg_ious)) if fg_ious else 1.0,
        "mean_foreground_iou": float(np.mean(fg_ious)) if fg_ious else 1.0,
        "min_per_id_iou": float(min(per_id_mins)) if per_id_mins else 1.0,
        "peak_cuda_memory_mb": float(peak_mb),
        "offload_video_to_cpu": bool(args.offload_video_to_cpu),
        "offload_state_to_cpu": bool(args.offload_state_to_cpu),
        "propagation_chunk_size": int(args.propagation_chunk_size),
        "anchor_birth_priority": bool(cli.anchor_birth_priority),
        "ownership_priority": str(cli.ownership_priority),
        "scheduler_mode": scheduler_mode,
        "stream_state_repair_mode": str(cli.stream_state_repair_mode),
        "birth_role_index": birth_role_index,
        "duplicate_rebirth_audit": duplicate_rebirth_audit,
        "reuse_video_state_template": bool(video_state_template is not None),
        "video_state_template_init_sec": float(video_state_template_init_sec),
        "state_template_clone_count": int(state_template_clone_count),
        "video_feature_bank": serializable_feature_bank_summary(feature_bank_info),
        "group_records": group_records,
        "records": records,
        "comparisons": comparisons,
        "video_path": str(video_path),
        "sheet_paths": sheet_paths,
    }
    summary_path = output_root / "phase5_frozen_birth_replay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "video": str(video_path), "sheets": sheet_paths}, ensure_ascii=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--birth-records", required=True)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rgb-root", default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--frame-ids", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--anchor-birth-priority", action="store_true")
    parser.add_argument(
        "--ownership-priority",
        choices=["current", "area_ascending", "area_descending"],
        default="current",
        help="Experimental per-frame disjoin ordering; current preserves legacy behavior.",
    )
    parser.add_argument("--scheduler-mode", choices=["independent_anchor", "streaming_state"], default="independent_anchor")
    parser.add_argument("--stream-state-repair-mode", choices=["reconsolidate", "pad"], default="reconsolidate")
    parser.add_argument("--reuse-video-state-template", action="store_true", default=False)
    parser.add_argument("--use-video-feature-bank", action="store_true")
    parser.add_argument("--video-feature-bank-storage-device", default="cuda")
    parser.add_argument("--video-gpu-hot-window", type=int, default=8)
    parser.add_argument("--duplicate-window-frames", type=int, default=3)
    parser.add_argument("--duplicate-overlap-threshold", type=float, default=0.55)
    parser.add_argument(
        "--min-birth-mask-area",
        type=int,
        default=0,
        help="Drop anchor birth rows whose saved mask area is below this value before replay.",
    )
    parser.add_argument(
        "--min-output-mask-area",
        type=int,
        default=0,
        help="Drop per-frame masks below this area before disjoin/label/video export.",
    )
    parser.add_argument(
        "--min-output-component-area",
        type=int,
        default=0,
        help="Remove connected components below this area inside each per-frame mask before disjoin/label/video export.",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
