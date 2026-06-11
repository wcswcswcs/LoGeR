from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

STREAM3D_ROOT = Path(__file__).resolve().parents[1]
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from geometry_provider.d4rt_carrier_provider import D4RTCarrierProjectionProvider
from stream4d.scannet_stream import ScanNetStream
from tools.diagnose_v22_ref0_scale_convention import (
    _mean_positive,
    _median_positive,
    _safe_ratio,
    _sample_depth_uv,
)
from tools.diagnose_v22_ref0_trajectory_scale import _finite_values, _relative_ref_to_target
from tools.run_v22_direct_reconstruction_benchmark import (
    _fit_ref0_pose_scale,
    _fit_rigid_no_scale,
    _json_safe,
    _read_seq_list,
    _sample_indices,
)


LABEL_KEYS = [
    "target_depth_over_local_z_median",
    "eval_ref0_depth_scale",
]

D4RT_INTERNAL_FEATURE_KEYS = [
    "anchor_count",
    "visibility_mean",
    "visibility_median",
    "confidence_mean",
    "confidence_median",
    "uv_x_mean",
    "uv_y_mean",
    "uv_x_std",
    "uv_y_std",
    "uv_bbox_area",
    "pred_local_abs_z_median",
    "pred_local_abs_z_mean",
    "pred_local_abs_z_iqr_over_median",
    "pred_ref_abs_z_median",
    "pred_ref_abs_z_mean",
    "pred_ref_abs_z_iqr_over_median",
    "pred_local_norm_median",
    "pred_local_norm_mean",
    "pred_ref_norm_median",
    "pred_ref_norm_mean",
    "local_over_ref_z_median",
    "local_over_ref_norm_median",
    "source_frame_unique_count",
    "source_frame_span",
    "d4rt_translation_norm",
    "rigid_residual_median",
    "rigid_residual_p90",
]

POSE_DIAGNOSTIC_FEATURE_KEYS = [
    *D4RT_INTERNAL_FEATURE_KEYS,
    "pose_translation_norm",
    "trajectory_scale_ratio",
]


def _positive_stats(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 1e-8)]
    if arr.size == 0:
        return {"median": None, "mean": None, "q25": None, "q75": None, "iqr_over_median": None}
    median = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return {
        "median": median,
        "mean": float(np.mean(arr)),
        "q25": q25,
        "q75": q75,
        "iqr_over_median": _safe_ratio(q75 - q25, median),
    }


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _row_value(row: dict[str, Any], key: str) -> float | None:
    value = _finite_float(row.get(key))
    return value


def _safe_absrel_summary(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    ok = np.isfinite(pred_arr) & np.isfinite(target_arr) & (target_arr > 1e-8)
    if np.count_nonzero(ok) == 0:
        return {"count": 0, "mean_absrel": None, "median_absrel": None, "max_absrel": None}
    err = np.abs(pred_arr[ok] - target_arr[ok]) / target_arr[ok]
    return {
        "count": int(err.size),
        "mean_absrel": float(np.mean(err)),
        "median_absrel": float(np.median(err)),
        "max_absrel": float(np.max(err)),
    }


def _fit_linear_predictor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    x_train = np.asarray(train_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64).reshape(-1)
    x_test = np.asarray(test_x, dtype=np.float64)
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("train_x/test_x must be 2D arrays")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("train_x/train_y row count mismatch")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("train_x/test_x feature count mismatch")
    finite_train = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
    finite_test = np.isfinite(x_test).all(axis=1)
    pred = np.full((x_test.shape[0],), np.nan, dtype=np.float64)
    if np.count_nonzero(finite_train) == 0 or np.count_nonzero(finite_test) == 0:
        return pred
    x_train = x_train[finite_train]
    y_train = y_train[finite_train]
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_std = np.where(x_std > 1e-8, x_std, 1.0)
    design = np.concatenate(
        [np.ones((x_train.shape[0], 1), dtype=np.float64), (x_train - x_mean) / x_std],
        axis=1,
    )
    coef, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    x_eval = x_test[finite_test]
    test_design = np.concatenate(
        [np.ones((x_eval.shape[0], 1), dtype=np.float64), (x_eval - x_mean) / x_std],
        axis=1,
    )
    pred[finite_test] = np.maximum(test_design @ coef, 1e-8)
    return pred


def _matrix_from_rows(rows: list[dict[str, Any]], feature_keys: list[str]) -> np.ndarray:
    matrix = np.full((len(rows), len(feature_keys)), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for col_idx, key in enumerate(feature_keys):
            value = _row_value(row, key)
            if value is not None:
                matrix[row_idx, col_idx] = value
    return matrix


def _leave_one_scene_out_predictions(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    feature_keys: list[str],
) -> list[dict[str, Any]]:
    scenes = sorted({str(row.get("scene")) for row in rows if row.get("scene") is not None})
    out: list[dict[str, Any]] = []
    for scene in scenes:
        train_rows = [row for row in rows if str(row.get("scene")) != scene]
        test_rows = [row for row in rows if str(row.get("scene")) == scene]
        train_y = np.asarray([_row_value(row, label_key) if _row_value(row, label_key) is not None else np.nan for row in train_rows], dtype=np.float64)
        test_y = np.asarray([_row_value(row, label_key) if _row_value(row, label_key) is not None else np.nan for row in test_rows], dtype=np.float64)
        train_x = _matrix_from_rows(train_rows, feature_keys)
        test_x = _matrix_from_rows(test_rows, feature_keys)
        pred = _fit_linear_predictor(train_x, train_y, test_x)
        for row, target, predicted in zip(test_rows, test_y, pred):
            if not np.isfinite(target) or target <= 1e-8:
                continue
            out.append(
                {
                    "scene": str(row.get("scene")),
                    "window_index": int(row.get("window_index", -1)),
                    "frame_id": int(row.get("frame_id", -1)),
                    "local_idx": int(row.get("local_idx", -1)),
                    "label_key": label_key,
                    "target_scale": float(target),
                    "predicted_scale": float(predicted) if np.isfinite(predicted) else None,
                    "absrel": float(abs(float(predicted) - target) / target) if np.isfinite(predicted) else None,
                }
            )
    return out


def _loo_global_median_predictions(rows: list[dict[str, Any]], label_key: str) -> list[dict[str, Any]]:
    scenes = sorted({str(row.get("scene")) for row in rows if row.get("scene") is not None})
    out: list[dict[str, Any]] = []
    for scene in scenes:
        train_values = np.asarray(
            [
                value
                for row in rows
                if str(row.get("scene")) != scene
                for value in [_row_value(row, label_key)]
                if value is not None and value > 1e-8
            ],
            dtype=np.float64,
        )
        if train_values.size == 0:
            continue
        pred = float(np.median(train_values))
        for row in rows:
            if str(row.get("scene")) != scene:
                continue
            target = _row_value(row, label_key)
            if target is None or target <= 1e-8:
                continue
            out.append(
                {
                    "scene": scene,
                    "window_index": int(row.get("window_index", -1)),
                    "frame_id": int(row.get("frame_id", -1)),
                    "local_idx": int(row.get("local_idx", -1)),
                    "label_key": label_key,
                    "target_scale": float(target),
                    "predicted_scale": pred,
                    "absrel": float(abs(pred - target) / target),
                }
            )
    return out


def _oracle_scene_median_predictions(rows: list[dict[str, Any]], label_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    scenes = sorted({str(row.get("scene")) for row in rows if row.get("scene") is not None})
    for scene in scenes:
        values = np.asarray(
            [
                value
                for row in rows
                if str(row.get("scene")) == scene
                for value in [_row_value(row, label_key)]
                if value is not None and value > 1e-8
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        pred = float(np.median(values))
        for row in rows:
            if str(row.get("scene")) != scene:
                continue
            target = _row_value(row, label_key)
            if target is None or target <= 1e-8:
                continue
            out.append(
                {
                    "scene": scene,
                    "window_index": int(row.get("window_index", -1)),
                    "frame_id": int(row.get("frame_id", -1)),
                    "local_idx": int(row.get("local_idx", -1)),
                    "label_key": label_key,
                    "target_scale": float(target),
                    "predicted_scale": pred,
                    "absrel": float(abs(pred - target) / target),
                }
            )
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=np.float64)
    sorted_arr = arr[order]
    start = 0
    while start < sorted_arr.size:
        end = start + 1
        while end < sorted_arr.size and sorted_arr[end] == sorted_arr[start]:
            end += 1
        avg_rank = 0.5 * float(start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[ok]
    yv = yv[ok]
    if xv.size < 3 or float(np.std(xv)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
        return None
    return float(np.corrcoef(xv, yv)[0, 1])


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    ok = np.isfinite(xv) & np.isfinite(yv)
    if np.count_nonzero(ok) < 3:
        return None
    return _pearson_corr(_rankdata(xv[ok]), _rankdata(yv[ok]))


def _correlation_rows(rows: list[dict[str, Any]], label_key: str, feature_keys: list[str]) -> list[dict[str, Any]]:
    y = np.asarray([_row_value(row, label_key) if _row_value(row, label_key) is not None else np.nan for row in rows], dtype=np.float64)
    out: list[dict[str, Any]] = []
    for key in feature_keys:
        x = np.asarray([_row_value(row, key) if _row_value(row, key) is not None else np.nan for row in rows], dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(ok) < 3:
            continue
        out.append(
            {
                "label_key": label_key,
                "feature_key": key,
                "count": int(np.count_nonzero(ok)),
                "pearson": _pearson_corr(x[ok], y[ok]),
                "spearman": _spearman_corr(x[ok], y[ok]),
                "uses_scannet_pose_for_feature": key in {"pose_translation_norm", "trajectory_scale_ratio"},
                "uses_scannet_depth_for_label": label_key == "target_depth_over_local_z_median",
                "uses_eval_ref0_depth_scale_label": label_key == "eval_ref0_depth_scale",
            }
        )
    out.sort(key=lambda row: abs(float(row["spearman"] or 0.0)), reverse=True)
    return out


def _prediction_summary(name: str, prediction_rows: list[dict[str, Any]], *, feature_count: int, uses_pose: bool) -> dict[str, Any]:
    pred = np.asarray(
        [row.get("predicted_scale") if row.get("predicted_scale") is not None else np.nan for row in prediction_rows],
        dtype=np.float64,
    )
    target = np.asarray([row.get("target_scale") for row in prediction_rows], dtype=np.float64)
    summary = _safe_absrel_summary(pred, target)
    finite = np.isfinite(pred) & np.isfinite(target) & (target > 1e-8)
    label_key = prediction_rows[0].get("label_key") if prediction_rows else None
    return {
        "predictor": name,
        "label_key": label_key,
        "feature_count": int(feature_count),
        "uses_scannet_pose_for_features": bool(uses_pose),
        "uses_scannet_depth_for_label": label_key == "target_depth_over_local_z_median",
        **summary,
        "target_scale_median": float(np.median(target[finite])) if np.any(finite) else None,
        "predicted_scale_median": float(np.median(pred[finite])) if np.any(finite) else None,
    }


def _summarize_scene_predictions(
    rows: list[dict[str, Any]],
    prediction_sets: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    scenes = sorted({str(row.get("scene")) for row in rows if row.get("scene") is not None})
    out: list[dict[str, Any]] = []
    for scene in scenes:
        scene_rows = [row for row in rows if str(row.get("scene")) == scene]
        summary: dict[str, Any] = {
            "scene": scene,
            "frame_count": int(len(scene_rows)),
        }
        for label in LABEL_KEYS:
            values = np.asarray(
                [
                    value
                    for row in scene_rows
                    for value in [_row_value(row, label)]
                    if value is not None and value > 1e-8
                ],
                dtype=np.float64,
            )
            summary[f"{label}_median"] = float(np.median(values)) if values.size else None
            summary[f"{label}_std"] = float(np.std(values)) if values.size else None
        for name, pred_rows in prediction_sets.items():
            scene_pred_rows = [row for row in pred_rows if str(row.get("scene")) == scene]
            pred = np.asarray(
                [row.get("predicted_scale") if row.get("predicted_scale") is not None else np.nan for row in scene_pred_rows],
                dtype=np.float64,
            )
            target = np.asarray([row.get("target_scale") for row in scene_pred_rows], dtype=np.float64)
            absrel = _safe_absrel_summary(pred, target)
            summary[f"{name}_mean_absrel"] = absrel["mean_absrel"]
            summary[f"{name}_median_absrel"] = absrel["median_absrel"]
            summary[f"{name}_predicted_median"] = float(np.nanmedian(pred)) if np.any(np.isfinite(pred)) else None
        out.append(summary)
    return out


def _extract_frame_rows(
    scene: str,
    stream: ScanNetStream,
    provider: D4RTCarrierProjectionProvider,
    *,
    max_windows_per_scene: int | None,
    max_anchors: int,
    min_visibility: float,
    min_confidence: float,
    robust_trim_percentile: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache = provider._load_scene(scene)
    windows = list(cache["windows"])
    if max_windows_per_scene is not None:
        windows = windows[: int(max_windows_per_scene)]
    frame_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    depth_cache: dict[int, np.ndarray] = {}
    for window_index, window in enumerate(windows):
        _, eval_diag = _fit_ref0_pose_scale(
            stream,
            window,
            robust_trim_percentile=float(robust_trim_percentile),
            max_anchors=int(max_anchors),
        )
        eval_scale = eval_diag.get("ref0_pose_scale")
        window_rows.append(
            {
                "scene": scene,
                "window_index": int(window_index),
                "window_path": str(window.path),
                "ref_frame": int(window.frame_ids[0]) if window.frame_ids else -1,
                "window_frame_count": int(len(window.frame_ids)),
                "eval_ref0_depth_scale": float(eval_scale) if eval_scale is not None and np.isfinite(float(eval_scale)) else None,
                **eval_diag,
            }
        )
        if not window.frame_ids:
            continue
        pose0 = stream.load_pose(int(window.frame_ids[0]))
        with np.load(window.path) as data:
            if "xyz_local" not in data.files or "xyz_ref" not in data.files:
                continue
            xyz_local = np.asarray(data["xyz_local"], dtype=np.float64)
            xyz_ref = np.asarray(data["xyz_ref"], dtype=np.float64)
            src_frame = np.asarray(data.get("src_frame", np.zeros((xyz_ref.shape[1],), dtype=np.int64)), dtype=np.int64)
        if xyz_local.shape != np.asarray(window.xyz).shape or xyz_ref.shape != xyz_local.shape:
            continue
        per_frame_cap = max(16, int(max_anchors) // max(len(window.frame_ids) - 1, 1))
        for local_idx, frame_id in enumerate(window.frame_ids):
            if local_idx == 0:
                continue
            uv = np.asarray(window.uv[local_idx], dtype=np.float64)
            valid = (
                np.asarray(window.valid[local_idx], dtype=bool)
                & np.isfinite(xyz_local[local_idx]).all(axis=1)
                & np.isfinite(xyz_ref[local_idx]).all(axis=1)
                & (np.asarray(window.visibility[local_idx], dtype=np.float64) >= float(min_visibility))
                & (np.asarray(window.confidence[local_idx], dtype=np.float64) >= float(min_confidence))
                & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] <= 1.0)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] <= 1.0)
            )
            indices = np.flatnonzero(valid)
            if indices.shape[0] < 8:
                continue
            indices = _sample_indices(indices, per_frame_cap)
            target_depth = _sample_depth_uv(
                stream.load_depth(int(frame_id)) if int(frame_id) not in depth_cache else depth_cache[int(frame_id)],
                uv[indices],
            )
            if int(frame_id) not in depth_cache:
                depth_cache[int(frame_id)] = stream.load_depth(int(frame_id))
                target_depth = _sample_depth_uv(depth_cache[int(frame_id)], uv[indices])
            local_points = xyz_local[local_idx, indices]
            ref_points = xyz_ref[local_idx, indices]
            local_z = np.abs(local_points[:, 2])
            ref_z = np.abs(ref_points[:, 2])
            local_norm = np.linalg.norm(local_points, axis=1)
            ref_norm = np.linalg.norm(ref_points, axis=1)
            local_z_stats = _positive_stats(local_z)
            ref_z_stats = _positive_stats(ref_z)
            local_norm_stats = _positive_stats(local_norm)
            ref_norm_stats = _positive_stats(ref_norm)
            target_depth_median = _median_positive(target_depth)
            target_depth_mean = _mean_positive(target_depth)
            d4rt_translation_norm = None
            pose_translation_norm = None
            trajectory_scale_ratio = None
            rigid_residual_median = None
            rigid_residual_p90 = None
            if np.isfinite(pose0).all():
                pose = stream.load_pose(int(frame_id))
                if np.isfinite(pose).all():
                    try:
                        _, trans, residual = _fit_rigid_no_scale(ref_points, local_points)
                        d4rt_translation_norm = float(np.linalg.norm(trans))
                        rel_ref_to_target = _relative_ref_to_target(pose0, pose)
                        pose_translation_norm = float(np.linalg.norm(rel_ref_to_target[:3, 3]))
                        if d4rt_translation_norm > 1e-8 and pose_translation_norm > 1e-8:
                            trajectory_scale_ratio = float(pose_translation_norm / d4rt_translation_norm)
                        rigid_residual_median = float(np.median(residual))
                        rigid_residual_p90 = float(np.percentile(residual, 90))
                    except Exception:
                        pass
            src_subset = src_frame[indices] if src_frame.shape[0] >= int(np.max(indices)) + 1 else np.zeros(indices.shape, dtype=np.int64)
            uv_subset = uv[indices]
            row: dict[str, Any] = {
                "scene": scene,
                "window_index": int(window_index),
                "window_path": str(window.path),
                "ref_frame": int(window.frame_ids[0]),
                "frame_id": int(frame_id),
                "local_idx": int(local_idx),
                "anchor_count": int(indices.shape[0]),
                "eval_ref0_depth_scale": float(eval_scale) if eval_scale is not None and np.isfinite(float(eval_scale)) else None,
                "target_depth_valid": int(np.count_nonzero(np.isfinite(target_depth) & (target_depth > 0.0))),
                "target_depth_median": target_depth_median,
                "target_depth_mean": target_depth_mean,
                "target_depth_over_local_z_median": _safe_ratio(target_depth_median, local_z_stats["median"]),
                "target_depth_over_ref_z_median": _safe_ratio(target_depth_median, ref_z_stats["median"]),
                "target_depth_over_local_z_mean": _safe_ratio(target_depth_mean, local_z_stats["mean"]),
                "visibility_mean": float(np.mean(window.visibility[local_idx, indices])),
                "visibility_median": float(np.median(window.visibility[local_idx, indices])),
                "confidence_mean": float(np.mean(window.confidence[local_idx, indices])),
                "confidence_median": float(np.median(window.confidence[local_idx, indices])),
                "uv_x_mean": float(np.mean(uv_subset[:, 0])),
                "uv_y_mean": float(np.mean(uv_subset[:, 1])),
                "uv_x_std": float(np.std(uv_subset[:, 0])),
                "uv_y_std": float(np.std(uv_subset[:, 1])),
                "uv_bbox_area": float(
                    max(float(np.max(uv_subset[:, 0]) - np.min(uv_subset[:, 0])), 0.0)
                    * max(float(np.max(uv_subset[:, 1]) - np.min(uv_subset[:, 1])), 0.0)
                ),
                "pred_local_abs_z_median": local_z_stats["median"],
                "pred_local_abs_z_mean": local_z_stats["mean"],
                "pred_local_abs_z_iqr_over_median": local_z_stats["iqr_over_median"],
                "pred_ref_abs_z_median": ref_z_stats["median"],
                "pred_ref_abs_z_mean": ref_z_stats["mean"],
                "pred_ref_abs_z_iqr_over_median": ref_z_stats["iqr_over_median"],
                "pred_local_norm_median": local_norm_stats["median"],
                "pred_local_norm_mean": local_norm_stats["mean"],
                "pred_ref_norm_median": ref_norm_stats["median"],
                "pred_ref_norm_mean": ref_norm_stats["mean"],
                "local_over_ref_z_median": _safe_ratio(local_z_stats["median"], ref_z_stats["median"]),
                "local_over_ref_norm_median": _safe_ratio(local_norm_stats["median"], ref_norm_stats["median"]),
                "source_frame_unique_count": int(np.unique(src_subset).shape[0]) if src_subset.size else 0,
                "source_frame_span": int(np.max(src_subset) - np.min(src_subset)) if src_subset.size else 0,
                "d4rt_translation_norm": d4rt_translation_norm,
                "pose_translation_norm": pose_translation_norm,
                "trajectory_scale_ratio": trajectory_scale_ratio,
                "rigid_residual_median": rigid_residual_median,
                "rigid_residual_p90": rigid_residual_p90,
            }
            frame_rows.append(row)
    return frame_rows, window_rows


def _predictor_rows(frame_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    summaries: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    named_predictions: dict[str, list[dict[str, Any]]] = {}
    for label_key in LABEL_KEYS:
        predictors = [
            ("global_median_loo", _loo_global_median_predictions(frame_rows, label_key), 0, False),
            ("oracle_scene_median", _oracle_scene_median_predictions(frame_rows, label_key), 0, False),
            (
                "linear_loo_d4rt_internal",
                _leave_one_scene_out_predictions(
                    frame_rows,
                    label_key=label_key,
                    feature_keys=D4RT_INTERNAL_FEATURE_KEYS,
                ),
                len(D4RT_INTERNAL_FEATURE_KEYS),
                False,
            ),
            (
                "linear_loo_d4rt_plus_pose_diagnostic",
                _leave_one_scene_out_predictions(
                    frame_rows,
                    label_key=label_key,
                    feature_keys=POSE_DIAGNOSTIC_FEATURE_KEYS,
                ),
                len(POSE_DIAGNOSTIC_FEATURE_KEYS),
                True,
            ),
        ]
        for base_name, rows, feature_count, uses_pose in predictors:
            predictor_name = f"{base_name}__{label_key}"
            tagged_rows: list[dict[str, Any]] = []
            for row in rows:
                tagged = {"predictor": predictor_name, **row}
                tagged_rows.append(tagged)
                all_predictions.append(tagged)
            named_predictions[predictor_name] = tagged_rows
            summaries.append(_prediction_summary(predictor_name, tagged_rows, feature_count=feature_count, uses_pose=uses_pose))
    summaries.sort(key=lambda row: (str(row.get("label_key")), float(row.get("mean_absrel") or 9999.0)))
    return summaries, all_predictions, named_predictions


def _univariate_predictor_summaries(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label_key in LABEL_KEYS:
        for feature_key in POSE_DIAGNOSTIC_FEATURE_KEYS:
            predictions = _leave_one_scene_out_predictions(
                frame_rows,
                label_key=label_key,
                feature_keys=[feature_key],
            )
            if not predictions:
                continue
            summary = _prediction_summary(
                f"linear_loo_univariate__{feature_key}__{label_key}",
                predictions,
                feature_count=1,
                uses_pose=feature_key in {"pose_translation_norm", "trajectory_scale_ratio"},
            )
            summary["feature_key"] = feature_key
            rows.append(summary)
    rows.sort(
        key=lambda row: (
            str(row.get("label_key")),
            bool(row.get("uses_scannet_pose_for_features")),
            float(row.get("mean_absrel") or 9999.0),
        )
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.6f}"
    if isinstance(value, bool):
        return str(value)
    return str(value)


def _write_md(
    path: Path,
    metadata: dict[str, Any],
    predictor_summaries: list[dict[str, Any]],
    univariate_summaries: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    scene_summaries: list[dict[str, Any]],
) -> None:
    lines: list[str] = [
        "# v22.16 target-scale observability diagnostic",
        "",
        "Diagnostic-only: treats ScanNet `target_depth / D4RT local_z` and R23 eval-only scale as labels, then tests whether D4RT-internal frame statistics can predict them under leave-one-scene-out evaluation.",
        "",
        "Rows using ScanNet pose as features are marked diagnostic controls; rows using target depth or R23 scale as labels are not method results.",
        "",
        "## Metadata",
        "",
        "| item | value |",
        "|---|---:|",
    ]
    for key in sorted(metadata.keys()):
        lines.append(f"| {key} | {_fmt(metadata[key])} |")
    lines.extend(
        [
            "",
            "## Predictor Summary",
            "",
            "| predictor | label | count | mean absrel | median absrel | max absrel | pred median | target median | uses pose feature |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in predictor_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("predictor")),
                    _fmt(row.get("label_key")),
                    _fmt(row.get("count")),
                    _fmt(row.get("mean_absrel")),
                    _fmt(row.get("median_absrel")),
                    _fmt(row.get("max_absrel")),
                    _fmt(row.get("predicted_scale_median")),
                    _fmt(row.get("target_scale_median")),
                    _fmt(row.get("uses_scannet_pose_for_features")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Best Univariate LOO",
            "",
            "| label | feature | count | mean absrel | median absrel | max absrel | uses pose feature |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    non_pose_seen: dict[str, int] = {}
    pose_seen: dict[str, int] = {}
    for row in univariate_summaries:
        label = str(row.get("label_key"))
        uses_pose = bool(row.get("uses_scannet_pose_for_features"))
        seen = pose_seen if uses_pose else non_pose_seen
        if seen.get(label, 0) >= 5:
            continue
        seen[label] = seen.get(label, 0) + 1
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("label_key")),
                    _fmt(row.get("feature_key")),
                    _fmt(row.get("count")),
                    _fmt(row.get("mean_absrel")),
                    _fmt(row.get("median_absrel")),
                    _fmt(row.get("max_absrel")),
                    _fmt(row.get("uses_scannet_pose_for_features")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Strongest Feature Correlations",
            "",
            "| label | feature | count | pearson | spearman | uses pose feature |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in correlation_rows[:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("label_key")),
                    _fmt(row.get("feature_key")),
                    _fmt(row.get("count")),
                    _fmt(row.get("pearson")),
                    _fmt(row.get("spearman")),
                    _fmt(row.get("uses_scannet_pose_for_feature")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-scene",
            "",
            "| scene | frames | target/local scale med | target/local scale std | eval scale med | D4RT LOO target absrel | pose-control target absrel | D4RT LOO eval absrel |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scene_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scene")),
                    _fmt(row.get("frame_count")),
                    _fmt(row.get("target_depth_over_local_z_median_median")),
                    _fmt(row.get("target_depth_over_local_z_median_std")),
                    _fmt(row.get("eval_ref0_depth_scale_median")),
                    _fmt(row.get("linear_loo_d4rt_internal__target_depth_over_local_z_median_mean_absrel")),
                    _fmt(row.get("linear_loo_d4rt_plus_pose_diagnostic__target_depth_over_local_z_median_mean_absrel")),
                    _fmt(row.get("linear_loo_d4rt_internal__eval_ref0_depth_scale_mean_absrel")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose whether v22 target/local depth scale is observable from D4RT-internal statistics.")
    parser.add_argument("--seq-list", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v22_local_xyz_probe5_r1")
    parser.add_argument("--audit-root", default="outputs/audit/v22_16_target_scale_observability_probe5")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-windows-per-scene", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--robust-trim-percentile", type=float, default=90.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    scenes = _read_seq_list(Path(args.seq_list))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    provider = D4RTCarrierProjectionProvider(
        debug_root=args.cache_root,
        mode="raw",
        max_anchors=int(args.max_anchors),
    )

    all_frame_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    for scene in scenes:
        if not (Path(args.cache_root) / scene).exists():
            continue
        stream = ScanNetStream(seq_name=scene, backbone=args.backbone)
        frame_rows, window_rows = _extract_frame_rows(
            scene,
            stream,
            provider,
            max_windows_per_scene=args.max_windows_per_scene,
            max_anchors=int(args.max_anchors),
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            robust_trim_percentile=float(args.robust_trim_percentile),
        )
        all_frame_rows.extend(frame_rows)
        all_window_rows.extend(window_rows)

    predictor_summaries, prediction_rows, named_predictions = _predictor_rows(all_frame_rows)
    univariate_summaries = _univariate_predictor_summaries(all_frame_rows)
    correlation_rows: list[dict[str, Any]] = []
    for label_key in LABEL_KEYS:
        correlation_rows.extend(_correlation_rows(all_frame_rows, label_key, POSE_DIAGNOSTIC_FEATURE_KEYS))
    correlation_rows.sort(key=lambda row: abs(float(row.get("spearman") or 0.0)), reverse=True)
    scene_summaries = _summarize_scene_predictions(all_frame_rows, named_predictions)
    metadata = {
        "scene_count": len(scenes),
        "frame_row_count": len(all_frame_rows),
        "window_row_count": len(all_window_rows),
        "d4rt_internal_feature_count": len(D4RT_INTERNAL_FEATURE_KEYS),
        "pose_diagnostic_feature_count": len(POSE_DIAGNOSTIC_FEATURE_KEYS),
        "uses_scannet_depth_for_target_label": True,
        "uses_scannet_pose_for_pose_control_features": True,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
    }
    for label_key in LABEL_KEYS:
        non_pose = [
            row
            for row in univariate_summaries
            if row.get("label_key") == label_key and not bool(row.get("uses_scannet_pose_for_features"))
        ]
        pose = [
            row
            for row in univariate_summaries
            if row.get("label_key") == label_key and bool(row.get("uses_scannet_pose_for_features"))
        ]
        if non_pose:
            metadata[f"best_univariate_d4rt_feature_for_{label_key}"] = non_pose[0].get("feature_key")
            metadata[f"best_univariate_d4rt_mean_absrel_for_{label_key}"] = non_pose[0].get("mean_absrel")
        if pose:
            metadata[f"best_univariate_pose_feature_for_{label_key}"] = pose[0].get("feature_key")
            metadata[f"best_univariate_pose_mean_absrel_for_{label_key}"] = pose[0].get("mean_absrel")

    _write_csv(audit_root / "target_scale_observability_frame_rows.csv", all_frame_rows)
    _write_csv(audit_root / "target_scale_observability_window_rows.csv", all_window_rows)
    _write_csv(audit_root / "target_scale_observability_predictor_summary.csv", predictor_summaries)
    _write_csv(audit_root / "target_scale_observability_univariate_summary.csv", univariate_summaries)
    _write_csv(audit_root / "target_scale_observability_predictions.csv", prediction_rows)
    _write_csv(audit_root / "target_scale_observability_feature_correlations.csv", correlation_rows)
    _write_csv(audit_root / "target_scale_observability_scene_summary.csv", scene_summaries)
    (audit_root / "target_scale_observability_frame_rows.json").write_text(json.dumps(_json_safe(all_frame_rows), indent=2), encoding="utf-8")
    (audit_root / "target_scale_observability_window_rows.json").write_text(json.dumps(_json_safe(all_window_rows), indent=2), encoding="utf-8")
    (audit_root / "target_scale_observability_predictor_summary.json").write_text(json.dumps(_json_safe(predictor_summaries), indent=2), encoding="utf-8")
    (audit_root / "target_scale_observability_univariate_summary.json").write_text(json.dumps(_json_safe(univariate_summaries), indent=2), encoding="utf-8")
    (audit_root / "target_scale_observability_predictions.json").write_text(json.dumps(_json_safe(prediction_rows), indent=2), encoding="utf-8")
    (audit_root / "target_scale_observability_feature_correlations.json").write_text(json.dumps(_json_safe(correlation_rows), indent=2), encoding="utf-8")
    (audit_root / "target_scale_observability_scene_summary.json").write_text(json.dumps(_json_safe(scene_summaries), indent=2), encoding="utf-8")
    (audit_root / "target_scale_observability_metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    _write_md(audit_root / "target_scale_observability.md", metadata, predictor_summaries, univariate_summaries, correlation_rows, scene_summaries)
    print(f"Wrote v22.16 target-scale observability diagnostic to {audit_root}")


if __name__ == "__main__":
    main()
