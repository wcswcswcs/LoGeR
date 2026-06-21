from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .v47_common import (
    ROOT,
    bbox_from_mask,
    color_feature,
    dominant_gt,
    load_gt_label,
    load_mask_label,
    read_json,
    safe_mean,
    safe_quantile,
    utc_now,
)


def _window_sort_key(path: Path) -> int:
    text = path.stem.replace("carriers_window", "")
    try:
        return int(text)
    except ValueError:
        return 0


def _manifest_for(npz_path: Path) -> dict[str, Any]:
    path = npz_path.with_name(f"{npz_path.stem}_manifest.json")
    if not path.exists():
        return {}
    return read_json(path)


def _sample_labels(
    label: np.ndarray | None,
    uv: np.ndarray,
    visible: np.ndarray,
    allowed_labels: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.zeros((uv.shape[0],), dtype=np.int32)
    valid_uv = visible & (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
    if label is None or not np.any(valid_uv):
        return observed, valid_uv
    height, width = label.shape[:2]
    xs = np.rint(uv[valid_uv, 0] * float(width - 1)).astype(np.int64)
    ys = np.rint(uv[valid_uv, 1] * float(height - 1)).astype(np.int64)
    sampled = label[ys, xs].astype(np.int32, copy=False)
    if allowed_labels:
        sampled = np.asarray([int(value) if int(value) in allowed_labels else 0 for value in sampled], dtype=np.int32)
    else:
        sampled = np.zeros_like(sampled, dtype=np.int32)
    observed[np.flatnonzero(valid_uv)] = sampled
    return observed, valid_uv


def build_observation_tables(
    *,
    carrier_cache_root: Path,
    scenes: list[str],
    visibility_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    min_mask_area: int = 64,
    feature_backend: str = "colorhist_fallback",
) -> dict[str, Any]:
    carrier_rows: list[dict[str, Any]] = []
    mask_rows_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    support_counter: Counter[tuple[str, int, int]] = Counter()
    visible_support_counter: Counter[tuple[str, int, int]] = Counter()
    source_manifests: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    missing_scene_dirs: list[str] = []
    mask_node_serial = 0

    for scene in scenes:
        scene_dir = carrier_cache_root / scene
        if not scene_dir.exists():
            missing_scene_dirs.append(scene)
            continue
        for window_serial, npz_path in enumerate(sorted(scene_dir.glob("carriers_window*.npz"), key=_window_sort_key)):
            manifest = _manifest_for(npz_path)
            frame_ids = [int(value) for value in manifest.get("frame_ids", [])]
            data = np.load(npz_path)
            uv_pred = np.asarray(data["uv_pred"], dtype=np.float32)
            visibility_prob = np.asarray(data["visibility_prob"], dtype=np.float32)
            confidence_prob = np.asarray(data["confidence_prob"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=bool)
            carrier_ids = np.asarray(data["carrier_id"], dtype=np.int64) if "carrier_id" in data.files else np.arange(uv_pred.shape[1])
            if not frame_ids:
                frame_ids = list(range(int(uv_pred.shape[0])))
            source_manifests.append(manifest)
            frame_count_with_masks = 0
            visible_count = 0
            valid_uv_count = 0
            inside_count = 0
            for local_index, frame_id in enumerate(frame_ids[: uv_pred.shape[0]]):
                label = load_mask_label(scene, int(frame_id))
                allowed_labels: set[int] = set()
                mask_areas: dict[int, int] = {}
                if label is not None:
                    frame_count_with_masks += 1
                    values, counts = np.unique(label, return_counts=True)
                    gt = load_gt_label(scene, int(frame_id))
                    for value, count in zip(values, counts):
                        mask_id = int(value)
                        area = int(count)
                        if mask_id <= 0 or area < int(min_mask_area):
                            continue
                        allowed_labels.add(mask_id)
                        mask_areas[mask_id] = area
                        key = (scene, int(frame_id), mask_id)
                        if key not in mask_rows_by_key:
                            mask = label == mask_id
                            bbox = bbox_from_mask(mask)
                            diag_gt, diag_purity = dominant_gt(mask, gt)
                            feature, feature_ok = color_feature(scene, int(frame_id), mask) if feature_backend else ([], False)
                            mask_rows_by_key[key] = {
                                "node_id": mask_node_serial,
                                "mask_observation_id": f"{scene}:{int(frame_id)}:{mask_id}",
                                "scene": scene,
                                "scene_id": scene,
                                "frame_id": int(frame_id),
                                "mask_id": mask_id,
                                "mask_area": area,
                                "bbox_x0": bbox[0],
                                "bbox_y0": bbox[1],
                                "bbox_x1": bbox[2],
                                "bbox_y1": bbox[3],
                                "carrier_count": 0,
                                "visible_carrier_count": 0,
                                "support_density": 0.0,
                                "core_feature": feature,
                                "prototype_features": [feature] if feature else [],
                                "context_feature": [],
                                "boundary_feature": [],
                                "feature_backend": feature_backend or "disabled",
                                "core_feature_valid": bool(feature_ok),
                                "prototype_count": 1 if feature else 0,
                                "diagnostic_gt_instance": diag_gt,
                                "diagnostic_gt_purity": diag_purity,
                                "uses_gt_for_prediction": False,
                                "uses_gt_for_diagnostic_labels": True,
                            }
                            mask_node_serial += 1
                frame_visible = (
                    valid[local_index]
                    & (visibility_prob[local_index] >= float(visibility_threshold))
                    & (confidence_prob[local_index] >= float(confidence_threshold))
                )
                observed, valid_uv = _sample_labels(label, uv_pred[local_index], frame_visible, allowed_labels)
                visible_count += int(frame_visible.sum())
                valid_uv_count += int(valid_uv.sum())
                inside_count += int((observed > 0).sum())
                for carrier_index in range(int(uv_pred.shape[1])):
                    mask_id = int(observed[carrier_index])
                    key = (scene, int(frame_id), mask_id)
                    if mask_id > 0:
                        support_counter[key] += 1
                        if bool(frame_visible[carrier_index]):
                            visible_support_counter[key] += 1
                    carrier_rows.append(
                        {
                            "scene": scene,
                            "carrier_id": int(carrier_ids[carrier_index]),
                            "carrier_global_id": f"{scene}:{int(carrier_ids[carrier_index])}",
                            "frame_id": int(frame_id),
                            "chunk_id": int(window_serial),
                            "submap_id": int(window_serial),
                            "window_index": int(window_serial),
                            "carrier_index": int(carrier_index),
                            "uv_x": float(uv_pred[local_index, carrier_index, 0]),
                            "uv_y": float(uv_pred[local_index, carrier_index, 1]),
                            "visible": bool(frame_visible[carrier_index]),
                            "confidence": float(confidence_prob[local_index, carrier_index]),
                            "visibility_prob": float(visibility_prob[local_index, carrier_index]),
                            "valid": bool(valid[local_index, carrier_index]),
                            "valid_uv": bool(valid_uv[carrier_index]),
                            "mask_label_available": label is not None,
                            "observed_mask_id": mask_id if mask_id > 0 else None,
                            "observed_mask_area": int(mask_areas.get(mask_id, 0)) if mask_id > 0 else None,
                            "observed_mask_support_density": None,
                            "inside_prepared_mask": bool(mask_id > 0),
                            "is_boundary_region": False,
                            "scale_guard_pass": True,
                            "allow_metric_relation": True,
                            "uses_gt_for_prediction": False,
                        }
                    )
            window_rows.append(
                {
                    "scene": scene,
                    "window_index": int(window_serial),
                    "carrier_npz": str(npz_path.relative_to(ROOT) if npz_path.is_relative_to(ROOT) else npz_path),
                    "manifest": str(npz_path.with_name(f"{npz_path.stem}_manifest.json").relative_to(ROOT))
                    if npz_path.with_name(f"{npz_path.stem}_manifest.json").exists()
                    else "",
                    "frame_count": len(frame_ids[: uv_pred.shape[0]]),
                    "frame_count_with_masks": frame_count_with_masks,
                    "carrier_count": int(carrier_ids.shape[0]),
                    "visible_observation_count": visible_count,
                    "valid_uv_observation_count": valid_uv_count,
                    "inside_mask_observation_count": inside_count,
                    "uv_in01_rate": float(valid_uv_count / max(visible_count, 1)),
                    "carrier_inside_any_mask_ratio": None,
                    "variant": manifest.get("variant"),
                    "uses_gt_for_prediction": False,
                    "uses_pose_for_prediction": bool(manifest.get("uses_pose_for_prediction", False)),
                    "uses_rgbd_for_prediction": bool(manifest.get("uses_rgbd_for_prediction", False)),
                    "uses_scannet_mesh_for_prediction": bool(manifest.get("uses_scannet_mesh_for_prediction", False)),
                }
            )

    for key, row in mask_rows_by_key.items():
        row["carrier_count"] = int(support_counter[key])
        row["visible_carrier_count"] = int(visible_support_counter[key])
        row["support_density"] = float(support_counter[key] / max(int(row["mask_area"]), 1))

    support_density_by_key = {key: float(mask_rows_by_key[key]["support_density"]) for key in mask_rows_by_key}
    area_by_key = {key: int(mask_rows_by_key[key]["mask_area"]) for key in mask_rows_by_key}
    for row in carrier_rows:
        mask_id = row.get("observed_mask_id")
        if mask_id is None:
            continue
        key = (str(row["scene"]), int(row["frame_id"]), int(mask_id))
        row["observed_mask_area"] = area_by_key.get(key)
        row["observed_mask_support_density"] = support_density_by_key.get(key)

    mask_rows = list(mask_rows_by_key.values())
    carrier_visible = [row for row in carrier_rows if row["visible"]]
    carrier_valid_uv = [row for row in carrier_rows if row["visible"] and row["valid_uv"]]
    carrier_mask_eval = [row for row in carrier_valid_uv if row["mask_label_available"]]
    for row in window_rows:
        scene = str(row["scene"])
        window_index = int(row["window_index"])
        eval_rows = [
            carrier_row
            for carrier_row in carrier_rows
            if str(carrier_row["scene"]) == scene
            and int(carrier_row["window_index"]) == window_index
            and carrier_row["visible"]
            and carrier_row["valid_uv"]
            and carrier_row["mask_label_available"]
        ]
        row["mask_frame_valid_uv_observation_count"] = len(eval_rows)
        row["carrier_inside_any_mask_ratio"] = float(
            sum(1 for carrier_row in eval_rows if carrier_row["observed_mask_id"] is not None) / max(len(eval_rows), 1)
        )
    frames_per_carrier: Counter[str] = Counter()
    for row in carrier_rows:
        if row["visible"]:
            frames_per_carrier[str(row["carrier_global_id"])] += 1
    mask_carrier_counts = [int(row["carrier_count"]) for row in mask_rows]
    feature_success = [1.0 if row["core_feature_valid"] else 0.0 for row in mask_rows]
    frame_diffs: list[int] = []
    for manifest in source_manifests:
        frames = [int(value) for value in manifest.get("frame_ids", [])]
        frame_diffs.extend([b - a for a, b in zip(frames, frames[1:])])
    summary = {
        "phase": "v47_observation_tables",
        "created_at": utc_now(),
        "carrier_cache_root": str(carrier_cache_root.relative_to(ROOT) if carrier_cache_root.is_relative_to(ROOT) else carrier_cache_root),
        "scenes": scenes,
        "missing_scene_dirs": missing_scene_dirs,
        "visibility_threshold": float(visibility_threshold),
        "confidence_threshold": float(confidence_threshold),
        "min_mask_area": int(min_mask_area),
        "carrier_row_count": len(carrier_rows),
        "unique_carrier_count": len({row["carrier_global_id"] for row in carrier_rows}),
        "unique_frame_count": len({(row["scene"], row["frame_id"]) for row in carrier_rows}),
        "uv_in01_rate": float(len(carrier_valid_uv) / max(len(carrier_visible), 1)),
        "visible_rate": float(len(carrier_visible) / max(len(carrier_rows), 1)),
        "confidence_mean": safe_mean(row["confidence"] for row in carrier_rows),
        "observed_mask_hit_rate": float(sum(1 for row in carrier_mask_eval if row["observed_mask_id"] is not None) / max(len(carrier_mask_eval), 1)),
        "observed_mask_null_rate": float(sum(1 for row in carrier_mask_eval if row["observed_mask_id"] is None) / max(len(carrier_mask_eval), 1)),
        "visible_valid_uv_with_mask_label_count": len(carrier_mask_eval),
        "visible_valid_uv_without_mask_label_count": len(carrier_valid_uv) - len(carrier_mask_eval),
        "frames_per_carrier_mean": safe_mean(frames_per_carrier.values()),
        "frames_per_carrier_p10": safe_quantile(frames_per_carrier.values(), 0.10),
        "scale_weak_row_count": 0,
        "allow_metric_relation_ratio": 1.0 if carrier_rows else 0.0,
        "mask_count": len(mask_rows),
        "mask_with_ge1_carrier_ratio": float(sum(value >= 1 for value in mask_carrier_counts) / max(len(mask_carrier_counts), 1)),
        "mask_with_ge5_carrier_ratio": float(sum(value >= 5 for value in mask_carrier_counts) / max(len(mask_carrier_counts), 1)),
        "mask_with_ge16_carrier_ratio": float(sum(value >= 16 for value in mask_carrier_counts) / max(len(mask_carrier_counts), 1)),
        "carrier_inside_any_mask_ratio": float(sum(1 for row in carrier_mask_eval if row["observed_mask_id"] is not None) / max(len(carrier_mask_eval), 1)),
        "mean_carriers_per_mask": safe_mean(mask_carrier_counts),
        "support_density_mean": safe_mean(row["support_density"] for row in mask_rows),
        "support_density_p10": safe_quantile((row["support_density"] for row in mask_rows), 0.10),
        "feature_backend": feature_backend or "disabled",
        "feature_success_rate": safe_mean(feature_success),
        "core_feature_valid_rate": safe_mean(feature_success),
        "prototype_count_mean": safe_mean(row["prototype_count"] for row in mask_rows),
        "D4RT_encoder_stride": 1 if frame_diffs and all(diff == 1 for diff in frame_diffs) else (frame_diffs[0] if frame_diffs else None),
        "frame_stride_all_eq_1": bool(frame_diffs and all(diff == 1 for diff in frame_diffs)),
        "temporal_chunk_size": max((len(manifest.get("frame_ids", [])) for manifest in source_manifests), default=0),
        "checkpoint_clip_frames": 32,
        "carrier_observation_table_exists": True,
        "mask_observation_table_exists": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    summary["gate"] = {
        "carrier_observation_table_exists": True,
        "mask_observation_table_exists": True,
        "mask_with_ge5_carrier_ratio_pass": bool(summary["mask_with_ge5_carrier_ratio"] >= 0.70),
        "mask_with_ge16_carrier_ratio_pass": bool(summary["mask_with_ge16_carrier_ratio"] >= 0.40),
        "carrier_inside_any_mask_ratio_pass": bool(summary["carrier_inside_any_mask_ratio"] >= 0.65),
        "feature_success_rate_pass": bool((summary["feature_success_rate"] or 0.0) >= 0.95)
        if feature_backend
        else True,
        "frame_stride_all_eq_1": bool(summary["frame_stride_all_eq_1"]),
        "missing_scene_dirs": missing_scene_dirs,
    }
    summary["gate"]["pass"] = bool(
        summary["gate"]["mask_with_ge5_carrier_ratio_pass"]
        and summary["gate"]["mask_with_ge16_carrier_ratio_pass"]
        and summary["gate"]["carrier_inside_any_mask_ratio_pass"]
        and summary["gate"]["feature_success_rate_pass"]
        and summary["gate"]["frame_stride_all_eq_1"]
        and not missing_scene_dirs
    )
    return {"carrier_rows": carrier_rows, "mask_rows": mask_rows, "window_rows": window_rows, "summary": summary}
