from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v38_oracle_attribution import (
    _candidate_oracle_stats,
    _class_agnostic_gt,
    _parse_metric_file,
    _quality_stats,
)


MIN_REGION_SIZE = 100
RAW_CONFIG = "v38_i4_sparse_export_trace_probe5"
CLASS_AGNOSTIC_ID = 3


VARIANTS: dict[str, dict[str, Any]] = {
    "D1_const_min100": {
        "config": "v38_d1_const_min100_probe5",
        "mode": "const_min100",
        "description": "area >= 100 and constant score; non-GT reproduction of v37 best postprocess",
    },
    "D2_wta_quality": {
        "config": "v38_d2_wta_quality_probe5",
        "mode": "wta",
        "score_mode": "quality",
        "min_area_after": 100,
        "margin": 0.0,
        "description": "one-vertex-one-owner by non-GT object quality; drop masks under 100 vertices after ownership",
    },
    "D3_wta_quality_margin005": {
        "config": "v38_d3_wta_quality_margin005_probe5",
        "mode": "wta",
        "score_mode": "quality",
        "min_area_after": 100,
        "margin": 0.05,
        "description": "D2 plus unknown assignment for low-margin conflict vertices",
    },
    "D4_nms_quality_min_ioc50_wta": {
        "config": "v38_d4_nms_quality_min_ioc50_wta_probe5",
        "mode": "nms_wta",
        "score_mode": "quality",
        "min_area": 100,
        "min_area_after": 100,
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.50,
        "max_instances": 300,
        "description": "quality-ordered min-IoC NMS, max 300 objects, then WTA ownership",
    },
    "D5_nms_quality_min_ioc30_wta": {
        "config": "v38_d5_nms_quality_min_ioc30_wta_probe5",
        "mode": "nms_wta",
        "score_mode": "quality",
        "min_area": 100,
        "min_area_after": 100,
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.30,
        "max_instances": 300,
        "description": "stricter quality-ordered min-IoC NMS, max 300 objects, then WTA ownership",
    },
    "D6_top300_quality_wta": {
        "config": "v38_d6_top300_quality_wta_probe5",
        "mode": "topk_wta",
        "score_mode": "quality",
        "min_area": 100,
        "min_area_after": 100,
        "max_instances": 300,
        "description": "top 300 objects by non-GT object quality, then WTA ownership",
    },
    "D7_shuffled_quality_control": {
        "config": "v38_d7_shuffled_quality_control_probe5",
        "mode": "nms_wta",
        "score_mode": "shuffled_quality",
        "min_area": 100,
        "min_area_after": 100,
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.50,
        "max_instances": 300,
        "description": "shuffled-priority control for D4 selection and WTA",
    },
    "D8_no_temporal_quality_control": {
        "config": "v38_d8_no_temporal_quality_control_probe5",
        "mode": "nms_wta",
        "score_mode": "no_temporal_quality",
        "min_area": 100,
        "min_area_after": 100,
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.50,
        "max_instances": 300,
        "description": "D4-style selection with the temporal-span term removed from quality",
    },
    "D9_nms_quality_min_ioc50_keepmask": {
        "config": "v38_d9_nms_quality_min_ioc50_keepmask_probe5",
        "mode": "nms_keepmask",
        "score_mode": "quality",
        "min_area": 100,
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.50,
        "max_instances": 300,
        "description": "quality-ordered min-IoC NMS, max 300 objects, keep original masks without WTA",
    },
    "D10_nms_no_temporal_min_ioc50_keepmask": {
        "config": "v38_d10_nms_no_temporal_min_ioc50_keepmask_probe5",
        "mode": "nms_keepmask",
        "score_mode": "no_temporal_quality",
        "min_area": 100,
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.50,
        "max_instances": 300,
        "description": "D9 with temporal-span feature removed",
    },
    "D11_nms_inverse_area_min_ioc50_keepmask": {
        "config": "v38_d11_nms_inverse_area_min_ioc50_keepmask_probe5",
        "mode": "nms_keepmask",
        "score_mode": "inverse_area",
        "min_area": 100,
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.50,
        "max_instances": 300,
        "description": "small-object-priority min-IoC NMS control, keep original masks without WTA",
    },
    "D12_nms_mid_area_min_ioc50_keepmask": {
        "config": "v38_d12_nms_mid_area_min_ioc50_keepmask_probe5",
        "mode": "nms_keepmask",
        "score_mode": "mid_area",
        "min_area": 100,
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.50,
        "max_instances": 300,
        "description": "mid-size-priority min-IoC NMS control, keep original masks without WTA",
    },
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def _load_trace_rows(root: Path, trace_root: Path, scene: str, expected: int) -> list[dict[str, Any]]:
    path = root / trace_root / "scenes" / scene / "prediction_trace_rows.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda row: int(row["prediction_index"]))
    if len(rows) != expected:
        raise ValueError(f"{scene}: trace row count {len(rows)} != prediction count {expected}")
    return rows


def _quality_scores(
    scene: str,
    masks: np.ndarray,
    trace_rows: list[dict[str, Any]],
    mode: str,
    seed: int,
) -> np.ndarray:
    areas = masks.sum(axis=0).astype(np.float64)
    temporal_span = np.asarray([float(row.get("temporal_span") or 0.0) for row in trace_rows], dtype=np.float64)
    tube_count = np.asarray([float(row.get("object_tube_count") or 0.0) for row in trace_rows], dtype=np.float64)
    overlap = np.asarray([float(row.get("overlap_with_other_predictions") or 0.0) for row in trace_rows], dtype=np.float64)
    log_area = _normalize(np.log1p(np.maximum(areas, 0.0)))
    temporal = _normalize(np.log1p(np.maximum(temporal_span, 0.0)))
    tubes = _normalize(np.log1p(np.maximum(tube_count, 0.0)))
    anti_overlap = np.clip(1.0 - overlap, 0.0, 1.0)
    if mode == "quality":
        score = 0.45 * log_area + 0.25 * temporal + 0.15 * tubes + 0.15 * anti_overlap
    elif mode == "no_temporal_quality":
        score = 0.60 * log_area + 0.20 * tubes + 0.20 * anti_overlap
    elif mode == "shuffled_quality":
        base = 0.45 * log_area + 0.25 * temporal + 0.15 * tubes + 0.15 * anti_overlap
        rng = np.random.default_rng(int(seed) + sum(ord(ch) for ch in scene))
        score = base.copy()
        rng.shuffle(score)
    elif mode == "area":
        score = log_area
    elif mode == "inverse_area":
        score = 1.0 - log_area
    elif mode == "mid_area":
        score = 1.0 - np.minimum(np.abs(log_area - 0.5) * 2.0, 1.0)
    else:
        raise ValueError(f"Unsupported score mode: {mode}")
    return score.astype(np.float32, copy=False)


def _apply_wta(masks: np.ndarray, priority: np.ndarray, margin: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    rows, cols = np.nonzero(masks)
    out = np.zeros_like(masks, dtype=bool)
    if rows.size == 0:
        return out, {
            "conflict_points_before": 0,
            "conflict_points_after": 0,
            "unknown_conflict_points": 0,
            "removed_assignments": 0,
        }
    owner_counts = masks.sum(axis=1).astype(np.int32)
    conflict_points_before = int(np.count_nonzero(owner_counts > 1))
    order = np.lexsort((cols, -priority[cols], rows))
    rows_o = rows[order]
    cols_o = cols[order]
    kept_assignments = 0
    unknown_conflict_points = 0
    start = 0
    while start < rows_o.shape[0]:
        end = start + 1
        while end < rows_o.shape[0] and rows_o[end] == rows_o[start]:
            end += 1
        group_cols = cols_o[start:end]
        if group_cols.shape[0] == 1:
            out[int(rows_o[start]), int(group_cols[0])] = True
            kept_assignments += 1
        else:
            group_priority = priority[group_cols]
            if float(margin) > 0.0:
                best = float(group_priority[0])
                second = float(group_priority[1])
                if best - second < float(margin):
                    unknown_conflict_points += 1
                    start = end
                    continue
            out[int(rows_o[start]), int(group_cols[0])] = True
            kept_assignments += 1
        start = end
    after_counts = out.sum(axis=1).astype(np.int32)
    return out, {
        "conflict_points_before": conflict_points_before,
        "conflict_points_after": int(np.count_nonzero(after_counts > 1)),
        "unknown_conflict_points": unknown_conflict_points,
        "removed_assignments": int(rows.size - kept_assignments),
    }


def _order_indices(priority: np.ndarray, areas: np.ndarray) -> list[int]:
    return sorted(range(priority.shape[0]), key=lambda idx: (-float(priority[idx]), -float(areas[idx]), int(idx)))


def _sparse_nms(
    masks: np.ndarray,
    priority: np.ndarray,
    min_area: int,
    max_instances: int,
    threshold: float,
    mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    areas = masks.sum(axis=0).astype(np.float64)
    kept: list[int] = []
    suppressed: list[dict[str, Any]] = []
    owners_by_point: dict[int, list[int]] = {}
    point_cache: dict[int, np.ndarray] = {}
    for idx in _order_indices(priority, areas):
        area = float(areas[idx])
        if int(min_area) > 0 and area < float(min_area):
            suppressed.append({"idx": int(idx), "reason": "min_area", "area": area, "priority": float(priority[idx])})
            continue
        points = point_cache.get(idx)
        if points is None:
            points = np.flatnonzero(masks[:, idx]).astype(np.int64)
            point_cache[idx] = points
        intersections: dict[int, int] = defaultdict(int)
        for point in points.tolist():
            for kept_idx in owners_by_point.get(int(point), ()):
                intersections[int(kept_idx)] += 1
        max_overlap = 0.0
        if intersections:
            for kept_idx, inter in intersections.items():
                kept_area = float(areas[kept_idx])
                if mode == "min_ioc":
                    denom = min(kept_area, area)
                elif mode == "candidate_ioc":
                    denom = area
                elif mode == "iou":
                    denom = kept_area + area - float(inter)
                else:
                    raise ValueError(f"Unsupported overlap mode: {mode}")
                max_overlap = max(max_overlap, float(inter) / max(denom, 1.0))
        if max_overlap >= float(threshold):
            suppressed.append(
                {
                    "idx": int(idx),
                    "reason": "overlap",
                    "overlap": float(max_overlap),
                    "area": area,
                    "priority": float(priority[idx]),
                }
            )
            continue
        kept.append(int(idx))
        for point in points.tolist():
            owners_by_point.setdefault(int(point), []).append(int(idx))
        if int(max_instances) > 0 and len(kept) >= int(max_instances):
            break
    return np.asarray(kept, dtype=np.int64), suppressed


def _filter_min_area(
    masks: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    areas = masks.sum(axis=0).astype(np.int64)
    keep = areas >= int(min_area)
    return (
        masks[:, keep],
        scores[keep],
        classes[keep],
        {
            "dropped_by_min_area_after": int(np.count_nonzero(~keep)),
            "num_predictions_after_min_area": int(np.count_nonzero(keep)),
        },
    )


def _variant_outputs(
    scene: str,
    variant: str,
    meta: dict[str, Any],
    masks: np.ndarray,
    classes: np.ndarray,
    trace_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mode = str(meta["mode"])
    raw_quality = _quality_scores(scene, masks, trace_rows, str(meta.get("score_mode", "quality")), seed)
    if mode == "const_min100":
        keep = masks.sum(axis=0) >= MIN_REGION_SIZE
        return (
            masks[:, keep],
            np.ones((int(np.count_nonzero(keep)),), dtype=np.float32),
            classes[keep],
            {"kept_indices": np.flatnonzero(keep).astype(np.int64).tolist(), "dropped_by_min_area": int(np.count_nonzero(~keep))},
        )
    if mode == "wta":
        owned, wta_diag = _apply_wta(masks, raw_quality, margin=float(meta.get("margin", 0.0)))
        out_masks, out_scores, out_classes, filter_diag = _filter_min_area(
            owned,
            raw_quality,
            classes,
            int(meta.get("min_area_after", 0)),
        )
        return out_masks, out_scores, out_classes, {**wta_diag, **filter_diag}
    if mode == "nms_wta":
        keep, suppressed = _sparse_nms(
            masks=masks,
            priority=raw_quality,
            min_area=int(meta.get("min_area", 0)),
            max_instances=int(meta.get("max_instances", 0)),
            threshold=float(meta.get("overlap_threshold", 1.0)),
            mode=str(meta.get("overlap_mode", "min_ioc")),
        )
        selected_masks = masks[:, keep]
        selected_scores = raw_quality[keep]
        selected_classes = classes[keep]
        owned, wta_diag = _apply_wta(selected_masks, selected_scores, margin=0.0)
        out_masks, out_scores, out_classes, filter_diag = _filter_min_area(
            owned,
            selected_scores,
            selected_classes,
            int(meta.get("min_area_after", 0)),
        )
        return (
            out_masks,
            out_scores,
            out_classes,
            {
                "kept_before_wta": int(keep.shape[0]),
                "suppressed_count": int(len(suppressed)),
                "suppressed_preview": suppressed[:20],
                **wta_diag,
                **filter_diag,
            },
        )
    if mode == "nms_keepmask":
        keep, suppressed = _sparse_nms(
            masks=masks,
            priority=raw_quality,
            min_area=int(meta.get("min_area", 0)),
            max_instances=int(meta.get("max_instances", 0)),
            threshold=float(meta.get("overlap_threshold", 1.0)),
            mode=str(meta.get("overlap_mode", "min_ioc")),
        )
        return (
            masks[:, keep],
            raw_quality[keep],
            classes[keep],
            {
                "kept_after_nms": int(keep.shape[0]),
                "suppressed_count": int(len(suppressed)),
                "suppressed_preview": suppressed[:20],
            },
        )
    if mode == "topk_wta":
        areas = masks.sum(axis=0).astype(np.int64)
        eligible = np.flatnonzero(areas >= int(meta.get("min_area", 0)))
        order = sorted(eligible.tolist(), key=lambda idx: (-float(raw_quality[idx]), -float(areas[idx]), int(idx)))
        keep = np.asarray(order[: int(meta.get("max_instances", 300))], dtype=np.int64)
        selected_masks = masks[:, keep]
        selected_scores = raw_quality[keep]
        selected_classes = classes[keep]
        owned, wta_diag = _apply_wta(selected_masks, selected_scores, margin=0.0)
        out_masks, out_scores, out_classes, filter_diag = _filter_min_area(
            owned,
            selected_scores,
            selected_classes,
            int(meta.get("min_area_after", 0)),
        )
        return (
            out_masks,
            out_scores,
            out_classes,
            {
                "eligible_count": int(eligible.shape[0]),
                "kept_before_wta": int(keep.shape[0]),
                **wta_diag,
                **filter_diag,
            },
        )
    raise ValueError(f"Unsupported materializer mode: {mode}")


def _write_prediction(root: Path, config: str, scene: str, masks: np.ndarray, scores: np.ndarray, classes: np.ndarray, raw_pre_points: np.ndarray) -> None:
    pred_dir = root / "data/prediction" / f"{config}_class_agnostic"
    tmp_dir = root / "data/TMP" / config
    pred_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{scene}.npz",
        pred_masks=masks.astype(bool, copy=False),
        pred_score=scores.astype(np.float32, copy=False),
        pred_classes=classes.astype(np.int32, copy=False),
    )
    np.save(tmp_dir / f"{scene}_pre_points.npy", raw_pre_points.astype(np.int64, copy=False))


def _write_manifest(root: Path, variant: str, meta: dict[str, Any], source_config: str) -> None:
    config = str(meta["config"])
    manifest = build_prediction_manifest(
        root=root,
        output_config=config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[source_config, "outputs/audit/v38_export_trace/export_trace_summary.json"],
        pre_points_policy=f"copied_raw_support:{source_config}",
        support_policy=f"v38_phaseD_object_materializer:{variant}:{meta['mode']}",
        notes=f"v38 Phase D non-GT materializer variant {variant}: {meta['description']}. Diagnostic-only because ScanNet mesh export bridge is used; forbidden for method table.",
        extra={
            "phase": "v38_phaseD_object_materializer",
            "variant": variant,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "forbidden_for_method_table": True,
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "alignment_source": "scannet_rgbd_pose_mesh_export_bridge",
            "alignment_used_for_prediction": True,
        },
    )
    write_prediction_manifest(config, manifest, root=root, pred_suffix="class_agnostic")


def _copy_raw_manifest_alias(root: Path, source_config: str) -> None:
    manifest_path = root / "data/prediction" / f"{source_config}_class_agnostic" / "config_manifest.json"
    if manifest_path.exists():
        return
    manifest = build_prediction_manifest(
        root=root,
        output_config=source_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=["outputs/audit/v38_export_trace/export_trace_summary.json"],
        pre_points_policy="copied_from_v38_export_trace",
        support_policy="v38_export_trace",
        notes="v38 Phase D raw trace alias manifest.",
        extra={
            "phase": "v38_phaseD_D0_raw_trace",
            "forbidden_for_method_table": True,
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "alignment_source": "scannet_rgbd_pose_mesh_export_bridge",
            "alignment_used_for_prediction": True,
        },
    )
    write_prediction_manifest(source_config, manifest, root=root, pred_suffix="class_agnostic")


def _generate_variants(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    scenes = _read_split(root / args.split)
    _copy_raw_manifest_alias(root, args.input_config)
    for variant, meta in VARIANTS.items():
        _write_manifest(root, variant, meta, args.input_config)

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        pred_path = root / "data/prediction" / f"{args.input_config}_class_agnostic" / f"{scene}.npz"
        pre_path = root / "data/TMP" / args.input_config / f"{scene}_pre_points.npy"
        gt_path = root / args.gt_path / f"{scene}.txt"
        with np.load(pred_path) as pred:
            masks = np.asarray(pred["pred_masks"], dtype=bool)
            scores = np.asarray(pred["pred_score"], dtype=np.float32)
            classes = np.asarray(pred["pred_classes"], dtype=np.int32)
        raw_pre_points = np.load(pre_path).astype(np.int64)
        trace_rows = _load_trace_rows(root, Path(args.trace_root), scene, masks.shape[1])
        gt_eval = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
        raw_stats = _candidate_oracle_stats(masks, scores, gt_eval, raw_pre_points)
        raw_row = _quality_stats(scene, "D0_raw_trace", masks, scores, gt_eval, raw_pre_points, raw_stats)
        raw_row.update({"config": args.input_config, "num_objects_in": int(masks.shape[1]), "num_predictions_out": int(masks.shape[1])})
        rows.append(raw_row)
        for variant, meta in VARIANTS.items():
            out_masks, out_scores, out_classes, diag = _variant_outputs(
                scene=scene,
                variant=variant,
                meta=meta,
                masks=masks,
                classes=classes,
                trace_rows=trace_rows,
                seed=int(args.seed),
            )
            config = str(meta["config"])
            _write_prediction(root, config, scene, out_masks, out_scores, out_classes, raw_pre_points)
            stats = _candidate_oracle_stats(out_masks, out_scores, gt_eval, raw_pre_points)
            row = _quality_stats(scene, variant, out_masks, out_scores, gt_eval, raw_pre_points, stats)
            row.update(
                {
                    "config": config,
                    "num_objects_in": int(masks.shape[1]),
                    "num_predictions_out": int(out_masks.shape[1]),
                    "prediction_per_object_mean": float(out_masks.shape[1] / max(masks.shape[1], 1)),
                    "owned_vertex_ratio": float(np.count_nonzero(np.any(out_masks, axis=1)) / max(out_masks.shape[0], 1)),
                    "unknown_vertex_ratio": float(1.0 - np.count_nonzero(np.any(out_masks, axis=1)) / max(out_masks.shape[0], 1)),
                    **diag,
                }
            )
            rows.append(row)
            del out_masks, out_scores, out_classes
        del masks, scores, classes
    return rows


def _run_eval(args: argparse.Namespace, root: Path, config: str, output_root: Path) -> dict[str, Any]:
    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metric_file = eval_dir / f"{config}_class_agnostic.txt"
    log_path = eval_dir / f"{config}_evaluate.log"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(root / "data/prediction" / f"{config}_class_agnostic"),
        "--gt_path",
        str(root / args.gt_path),
        "--dataset",
        "scannet",
        "--no_class",
        "--tmp_root",
        str(root / "data/TMP"),
        "--tmp_config",
        config,
        "--output_file",
        str(metric_file),
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT)
    metrics = _parse_metric_file(metric_file) if metric_file.exists() else {}
    return {
        "config": config,
        "command": " ".join(cmd),
        "exit_code": int(proc.returncode),
        "metric_file": str(metric_file),
        "log_path": str(log_path),
        "metrics": metrics,
    }


def _aggregate_rows(rows: list[dict[str, Any]], evals: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    variants = ["D0_raw_trace", *VARIANTS.keys()]
    out: list[dict[str, Any]] = []
    for variant in variants:
        items = [row for row in rows if str(row["variant"]) == variant]
        if not items:
            continue
        numeric_keys = sorted(
            key for row in items for key, value in row.items() if isinstance(value, (int, float)) and key != "scene"
        )
        agg: dict[str, Any] = {"variant": variant, "scene_count": len(items)}
        for key in numeric_keys:
            vals = [float(row[key]) for row in items if isinstance(row.get(key), (int, float))]
            if vals:
                agg[key] = float(np.mean(vals))
        eval_row = evals.get(variant)
        if eval_row is not None:
            agg.update(
                {
                    "AP": eval_row.get("metrics", {}).get("AP"),
                    "AP50": eval_row.get("metrics", {}).get("AP50"),
                    "AP25": eval_row.get("metrics", {}).get("AP25"),
                    "eval_exit_code": eval_row.get("exit_code"),
                    "metric_file": eval_row.get("metric_file"),
                }
            )
        out.append(agg)
    return out


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stream4D v38 Phase D Object Materializer",
        "",
        "| variant | AP | AP50 | AP25 | #pred | conflict | dup | best IoU mean | owned | control |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["matrix"]:
        control = "yes" if str(row.get("variant")) in {"D7_shuffled_quality_control", "D8_no_temporal_quality_control"} else "no"
        lines.append(
            "| {variant} | {AP} | {AP50} | {AP25} | {pred} | {conflict} | {dup} | {iou} | {owned} | {control} |".format(
                variant=row.get("variant"),
                AP=row.get("AP"),
                AP50=row.get("AP50"),
                AP25=row.get("AP25"),
                pred=row.get("num_predictions"),
                conflict=row.get("mean_export_conflict_rate"),
                dup=row.get("duplicate_prediction_rate"),
                iou=row.get("per_GT_best_IoU_mean"),
                owned=row.get("owned_vertex_ratio"),
                control=control,
            )
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "Phase D variants do not use GT for prediction. GT is read only by the evaluator/diagnostic metrics. Outputs are diagnostic-only because the ScanNet mesh/RGB-D/pose export bridge is used.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric(row: dict[str, Any] | None, key: str, default: float) -> float:
    if row is None:
        return float(default)
    value = row.get(key)
    if value is None:
        return float(default)
    return float(value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    if Path.cwd().resolve() != root:
        os.chdir(root)
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _generate_variants(args, root)
    evals: dict[str, dict[str, Any]] = {"D0_raw_trace": _run_eval(args, root, args.input_config, output_root)}
    for variant, meta in VARIANTS.items():
        evals[variant] = _run_eval(args, root, str(meta["config"]), output_root)
    matrix = _aggregate_rows(rows, evals)
    best = max(
        (row for row in matrix if str(row.get("variant")) not in {"D7_shuffled_quality_control", "D8_no_temporal_quality_control"}),
        key=lambda row: float(row.get("AP") or -1.0),
    )
    d4 = next((row for row in matrix if row.get("variant") == "D4_nms_quality_min_ioc50_wta"), None)
    shuffled = next((row for row in matrix if row.get("variant") == "D7_shuffled_quality_control"), None)
    no_temporal = next((row for row in matrix if row.get("variant") == "D8_no_temporal_quality_control"), None)
    summary = {
        "phase": "v38_phaseD_object_materializer",
        "input_config": args.input_config,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "forbidden_for_method_table": True,
        "matrix": matrix,
        "scene_rows": rows,
        "evals": evals,
        "best_non_control_variant": best,
        "phaseD_gate": {
            "num_predictions_out_le_300": bool(_metric(best, "num_predictions_out", 1e9) <= 300.0),
            "duplicate_prediction_rate_le_010": bool(_metric(best, "duplicate_prediction_rate", 1.0) <= 0.10),
            "mean_export_conflict_rate_le_010": bool(_metric(best, "mean_export_conflict_rate", 1.0) <= 0.10),
            "AP_ge_012": bool(_metric(best, "AP", 0.0) >= 0.12),
            "AP25_ge_050": bool(_metric(best, "AP25", 0.0) >= 0.50),
            "real_beats_shuffled_control": (
                None
                if d4 is None or shuffled is None
                else bool(_metric(d4, "AP", 0.0) > _metric(shuffled, "AP", 0.0))
            ),
            "real_beats_no_temporal_control": (
                None
                if d4 is None or no_temporal is None
                else bool(_metric(d4, "AP", 0.0) > _metric(no_temporal, "AP", 0.0))
            ),
        },
        "notes": [
            "No GT is used to construct Phase D prediction variants.",
            "GT is used after prediction generation for AP and attribution metrics only.",
            "Outputs remain diagnostic-only and forbidden for method tables because they use the ScanNet mesh/RGB-D/pose export bridge inherited from Phase B.",
        ],
    }
    _write_json(output_root / "object_materializer_summary.json", summary)
    _write_csv(output_root / "object_materializer_matrix.csv", matrix)
    _write_csv(output_root / "object_materializer_scene_rows.csv", rows)
    _write_markdown(output_root / "object_materializer_summary.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v38 Phase D non-GT object-level materializer variants.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--input-config", default=RAW_CONFIG)
    parser.add_argument("--trace-root", default="outputs/audit/v38_export_trace")
    parser.add_argument("--output-root", default="outputs/audit/v38_object_materializer")
    parser.add_argument("--gt-path", default="data/scannet/gt")
    parser.add_argument("--seed", type=int, default=3801)
    parser.add_argument("--cuda-visible-devices", default="6")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
