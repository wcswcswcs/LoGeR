#!/usr/bin/env python3
"""Compute v98.1 MV_AP_scene after local holdout unlocks scene evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_v98_1_canonical_holdout_metrics import (  # noqa: E402
    BASE_VARIANT,
    RUN_ID as HOLDOUT_RUN_ID,
    _apply_score_policy,
    _bool,
    _f1,
    _int,
    _load_selected_rows,
    _load_source_scope,
    _num,
    _project,
    _read_label,
    _rel,
    _score_array,
    _top_iou_rows,
    _write_csv,
    _write_json,
)
from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


RUN_ID = "v98_1_phase14_canonical_mv_ap_scene"
SCORE_POLICY = "frame_count"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _mean(values: list[Any]) -> float:
    vals = [_num(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


def _load_eval_rows(input_rows: Path, *, base_variant: str, score_policy: str, variant_family: str) -> list[dict[str, Any]]:
    base_rows = _load_selected_rows(input_rows, allowed_variants={base_variant})
    scored = _apply_score_policy(base_rows, base_variant=base_variant, policy=score_policy)
    return [{**row, "variant_family": variant_family} for row in scored]


def _load_control_rows(input_rows: Path) -> list[dict[str, Any]]:
    rows = _load_selected_rows(input_rows)
    return [{**row, "variant_family": "scene_control"} for row in rows]


def _evaluate_scene_variant(
    *,
    variant: str,
    rows: list[dict[str, Any]],
    scope: dict[str, Any],
    scene_object_mode: str,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, int]:
    by_frame_mask: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene = str(row["scene_id"])
        frame = int(row["frame_id"])
        if (scene, frame) not in scope["mask_path_by_frame"]:
            continue
        by_frame_mask[(scene, frame, int(row["mask_id"]))].append(row)

    metric_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    total_duplicate_conflicts = 0
    total_missing_masks = 0
    for scene in sorted(scope["frames_by_scene"]):
        object_ids: set[str] = set()
        object_scores: dict[str, float] = {}
        mask_to_obj: dict[tuple[int, int], str] = {}
        duplicate_conflicts = 0
        for (row_scene, frame, mask_id), vals in sorted(by_frame_mask.items()):
            if row_scene != scene:
                continue
            vals_sorted = sorted(vals, key=lambda r: (_num(r.get("score")), str(r.get("mv_object_id"))), reverse=True)
            chosen = vals_sorted[0]
            if len({str(v.get("mv_object_id", "")) for v in vals_sorted}) > 1:
                duplicate_conflicts += len(vals_sorted) - 1
            raw_oid = str(chosen["mv_object_id"])
            if scene_object_mode == "window_fragmented":
                window = scope["frame_to_window"].get((scene, int(frame)), "")
                scene_oid = f"{window}|{raw_oid}"
            elif scene_object_mode == "scene_birth_id_no_extra_stitching":
                scene_oid = raw_oid
            else:
                raise ValueError(f"unknown scene_object_mode {scene_object_mode}")
            object_ids.add(scene_oid)
            object_scores[scene_oid] = max(float(object_scores.get(scene_oid, 0.0)), _num(chosen.get("score"), 1.0))
            mask_to_obj[(int(frame), int(mask_id))] = scene_oid

        object_to_idx = {oid: idx + 1 for idx, oid in enumerate(sorted(object_ids))}
        idx_to_obj = {idx: oid for oid, idx in object_to_idx.items()}
        acc = SparseSceneIoU()
        missing_masks = 0
        for frame in scope["frames_by_scene"].get(scene, []):
            mask_path = scope["mask_path_by_frame"].get((scene, int(frame)))
            label = None
            if mask_path is not None and mask_path.exists():
                label = _read_label(mask_path)
                shape_hw = tuple(int(v) for v in label.shape[:2])
            else:
                missing_masks += 1
                shape_hw = (968, 1296)
            gt = _load_gt_2d(scene, int(frame), shape_hw)
            if label is None:
                label = np.zeros(shape_hw, dtype=np.int64)
            pred = np.zeros(shape_hw, dtype=np.int64)
            for mask_id in np.unique(label):
                mask_id = int(mask_id)
                if mask_id <= 0:
                    continue
                scene_oid = mask_to_obj.get((int(frame), mask_id), "")
                pred_id = object_to_idx.get(scene_oid, 0)
                if pred_id > 0:
                    pred[label == mask_id] = pred_id
            acc.add(pred, gt)
            frame_rows.append(
                {
                    "schema_version": "stream4d_v98_1_canonical_mv_ap_scene_frame_v1",
                    "phase_id": "v98_phase14_mv_ap_scene",
                    "run_id": RUN_ID,
                    "split": split_name,
                    "variant_id": variant,
                    "scene_id": scene,
                    "frame_id": int(frame),
                    "scene_object_mode": scene_object_mode,
                    "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                    "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                    "mask_path": _rel(mask_path) if mask_path is not None else "",
                    "mask_exists": bool(mask_path is not None and mask_path.exists()),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": True,
                    "uses_future": False,
                }
            )
        summary, iou, pred_ids, gt_ids = _summarize_iou(
            accumulator=acc,
            min_pred_pixels=1,
            min_gt_pixels=1,
            score_mode="input",
            input_scores=_score_array(object_to_idx, object_scores),
        )
        sf50 = summary.get("score_free_match_at_050") or {}
        sf25 = summary.get("score_free_match_at_025") or {}
        metric_rows.append(
            {
                "schema_version": "stream4d_v98_1_canonical_mv_ap_scene_scene_row_v1",
                "phase_id": "v98_phase14_mv_ap_scene",
                "run_id": RUN_ID,
                "split": split_name,
                "variant_id": variant,
                "scene_id": scene,
                "scene_object_mode": scene_object_mode,
                "MV_AP_scene": summary.get("ap"),
                "MV_AP50_scene": summary.get("ap50"),
                "MV_AP25_scene": summary.get("ap25"),
                "ScoreFreeMatch50_scene": _f1(sf50.get("precision"), sf50.get("recall")),
                "ScoreFreeMatch25_scene": _f1(sf25.get("precision"), sf25.get("recall")),
                "frame_count": summary.get("frame_count"),
                "gt_object_count": summary.get("evaluated_gt_count"),
                "pred_object_count": summary.get("evaluated_pred_count"),
                "same_frame_collision_count": int(duplicate_conflicts),
                "missing_mask_raster_count": int(missing_masks),
                "metric_scope": "scene_level_raw_gt_no_window_split",
                "canonical_metric_source": "run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )
        for top in _top_iou_rows(iou, pred_ids, gt_ids):
            top_rows.append(
                {
                    "schema_version": "stream4d_v98_1_canonical_mv_ap_scene_top_iou_v1",
                    "phase_id": "v98_phase14_mv_ap_scene",
                    "run_id": RUN_ID,
                    "split": split_name,
                    "variant_id": variant,
                    "scene_id": scene,
                    "scene_object_mode": scene_object_mode,
                    "pred_id": top["pred_id"],
                    "mv_object_id": idx_to_obj.get(int(top["pred_id"]), ""),
                    "gt_id": top["gt_id"],
                    "iou": top["iou"],
                    "matrix_scope": "scene_level_raw_gt_no_window_split",
                }
            )
        total_duplicate_conflicts += duplicate_conflicts
        total_missing_masks += missing_masks
    return metric_rows, frame_rows, top_rows, total_duplicate_conflicts, total_missing_masks


def _aggregate(rows: list[dict[str, Any]], *, variant_family: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("split", "")), str(row.get("variant_id", "")), str(row.get("scene_object_mode", "")))].append(row)
    out: list[dict[str, Any]] = []
    for (split, variant, mode), vals in sorted(grouped.items()):
        out.append(
            {
                "schema_version": "stream4d_v98_1_canonical_mv_ap_scene_aggregate_v1",
                "phase_id": "v98_phase14_mv_ap_scene",
                "run_id": RUN_ID,
                "split": split,
                "variant_id": variant,
                "variant_family": variant_family,
                "scene_object_mode": mode,
                "scene_count": len(vals),
                "mean_MV_AP_scene": _mean([row.get("MV_AP_scene") for row in vals]),
                "mean_MV_AP50_scene": _mean([row.get("MV_AP50_scene") for row in vals]),
                "mean_MV_AP25_scene": _mean([row.get("MV_AP25_scene") for row in vals]),
                "mean_score_free_Match50_scene": _mean([row.get("ScoreFreeMatch50_scene") for row in vals]),
                "same_frame_collision_count": int(sum(_int(row.get("same_frame_collision_count")) for row in vals)),
                "missing_mask_raster_count": int(sum(_int(row.get("missing_mask_raster_count")) for row in vals)),
                "metric_scope": "scene_level_raw_gt_no_window_split",
                "canonical_metric_source": "run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--source-rows", required=True)
    parser.add_argument("--real-input-rows", required=True)
    parser.add_argument("--control-input-rows", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-variant", default=BASE_VARIANT)
    parser.add_argument("--score-policy", default=SCORE_POLICY, choices=["frame_count"])
    args = parser.parse_args()

    started = time.time()
    source_rows = _project(args.source_rows)
    real_input_rows = _project(args.real_input_rows)
    output_root = _project(args.output_root)
    scope = _load_source_scope(source_rows)
    eval_rows = _load_eval_rows(real_input_rows, base_variant=args.base_variant, score_policy=args.score_policy, variant_family="frozen_real_scene")
    if args.control_input_rows:
        eval_rows.extend(_load_control_rows(_project(args.control_input_rows)))

    metric_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant_id"]) for row in eval_rows}):
        rows = [row for row in eval_rows if row.get("variant_id") == variant]
        for mode in ("scene_birth_id_no_extra_stitching", "window_fragmented"):
            m, f, t, _dups, _missing = _evaluate_scene_variant(
                variant=variant,
                rows=rows,
                scope=scope,
                scene_object_mode=mode,
                split_name=args.split_name,
            )
            metric_rows.extend(m)
            frame_rows.extend(f)
            top_rows.extend(t)

    aggregate_rows: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant_id"]) for row in metric_rows}):
        family = "frozen_real_scene" if variant.startswith(args.base_variant) else "scene_control"
        aggregate_rows.extend(_aggregate([row for row in metric_rows if row.get("variant_id") == variant], variant_family=family))

    target_variant = f"{args.base_variant}__score_{args.score_policy}"
    target = next(
        (
            row
            for row in aggregate_rows
            if row.get("variant_id") == target_variant and row.get("scene_object_mode") == "scene_birth_id_no_extra_stitching"
        ),
        {},
    )
    fragmented = next(
        (
            row
            for row in aggregate_rows
            if row.get("variant_id") == target_variant and row.get("scene_object_mode") == "window_fragmented"
        ),
        {},
    )
    control_rows = [
        row
        for row in aggregate_rows
        if row.get("variant_family") == "scene_control" and row.get("scene_object_mode") == "scene_birth_id_no_extra_stitching"
    ]
    best_control = max(control_rows, key=lambda r: (_num(r.get("mean_MV_AP_scene"), -1.0), _num(r.get("mean_MV_AP50_scene"), -1.0)), default={})
    scene_gain = _num(target.get("mean_MV_AP_scene"), float("nan")) - _num(fragmented.get("mean_MV_AP_scene"), float("nan"))
    gates = {
        "MV_AP_scene_computed": bool(target),
        "same_frame_collision_count_eq_0": _int(target.get("same_frame_collision_count"), 1) == 0,
        "missing_mask_raster_count_eq_0": _int(target.get("missing_mask_raster_count"), 1) == 0,
        "uses_gt_for_prediction_false": not any(_bool(row.get("uses_gt_for_prediction")) for row in eval_rows) and not scope["uses_gt_for_prediction"],
        "uses_future_false": not any(_bool(row.get("uses_future")) for row in eval_rows) and not scope["uses_future"],
    }
    _write_csv(output_root / f"{args.split_name}_mv_scene_metric_rows.csv", metric_rows)
    _write_csv(output_root / f"{args.split_name}_mv_scene_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(output_root / f"{args.split_name}_mv_scene_frame_rows.csv", frame_rows)
    _write_csv(output_root / f"{args.split_name}_mv_scene_top_iou_rows.csv", top_rows)
    summary = {
        "schema": "stream4d_v98_1_phase14_canonical_mv_ap_scene_summary_v1",
        "phase_id": "v98_phase14_mv_ap_scene",
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "split": args.split_name,
        "target_variant": target_variant,
        "scene_object_mode": "scene_birth_id_no_extra_stitching",
        "MV_AP_scene": target.get("mean_MV_AP_scene", ""),
        "MV_AP50_scene": target.get("mean_MV_AP50_scene", ""),
        "MV_AP25_scene": target.get("mean_MV_AP25_scene", ""),
        "window_fragmented_MV_AP_scene": fragmented.get("mean_MV_AP_scene", ""),
        "scene_minus_window_fragmented_MV_AP_scene": scene_gain if math.isfinite(scene_gain) else "",
        "best_scene_control_variant": best_control.get("variant_id", ""),
        "best_scene_control_MV_AP_scene": best_control.get("mean_MV_AP_scene", ""),
        "best_scene_control_MV_AP50_scene": best_control.get("mean_MV_AP50_scene", ""),
        "gate_results": gates,
        "metric_source_scene": "run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "adapter_contract": "MV_AP_scene: prediction ids are scene-level object ids; GT ids are raw scene instance ids; no window split",
        "source_rows": source_rows,
        "real_input_rows": real_input_rows,
        "control_input_rows": _project(args.control_input_rows) if args.control_input_rows else "",
        "runtime_sec": float(time.time() - started),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "parent_holdout_run": HOLDOUT_RUN_ID,
    }
    _write_json(output_root / f"{args.split_name}_mv_scene_summary.json", summary)
    print(json.dumps(summary, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
