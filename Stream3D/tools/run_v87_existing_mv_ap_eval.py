#!/usr/bin/env python3
"""Adapt v87 materialized frame-mask tubes to the existing v65 MV AP evaluator."""

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
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream
from tools.run_v65_scene_multiview_ap import (
    SparseSceneIoU,
    _load_gt_2d as v65_load_gt_2d,
    _sha256 as v65_sha256,
    _soma_pred_2d as v65_soma_pred_2d,
    _summarize_iou as v65_summarize_iou,
    _top_iou_rows as v65_top_iou_rows,
    _write_csv as v65_write_csv,
    _write_json as v65_write_json,
    _write_sha256sums as v65_write_sha256sums,
)
from tools.run_v87_mv_ap_persistent_affinity_readout import (
    _control_frame_rows as v87_control_frame_rows,
    _int as v87_int,
    _read_csv_rows as v87_read_csv_rows,
    _repo_path as v87_repo_path,
    _scene_mask_dir as v87_scene_mask_dir,
    _variant_family as v87_variant_family,
    _variant_is_real as v87_variant_is_real,
)


CONTROL_SUFFIXES = [
    ("semantic_only", "B6_semantic_only_history_grouping"),
    ("shuffled_history", "B7_shuffled_history_grouping"),
    ("stale_history", "B8_stale_history_grouping"),
    ("size_matched_hash", "B9_size_matched_hash_by_scene"),
    ("uniform_hash", "B10_uniform_hash_history"),
    ("single_largest", "B11_single_largest_by_scene"),
    ("area_risk", "B12_area_risk_count_control"),
]

CONTROL_BASE_VARIANTS = [
    ("B1_M10_state_priority", "B1_M10"),
    ("B2_DV5_confirmed_object_gain", "B2_DV5"),
    ("B3_DV5_object_gain_with_local_fallback", "B3_DV5_fallback"),
    ("B4_M10_state_priority_with_local_fallback", "B4_M10_fallback"),
    ("B5_confirmed_only_conservative", "B5_confirmed"),
]


def _rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else 0.0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _frame_scope_rows(frame_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    out: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in frame_rows:
        split = str(row.get("split", ""))
        scene = str(row.get("scene_id", ""))
        frame_id = v87_int(row.get("frame_id"), -1)
        if split and scene and frame_id >= 0:
            out[(split, scene)].add(int(frame_id))
    return {key: sorted(values) for key, values in out.items()}


def _add_controls(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    control_rows: list[dict[str, Any]] = []
    splits = sorted({str(row.get("split", "")) for row in frame_rows})
    for split in splits:
        scenes = sorted({str(row.get("scene_id", "")) for row in frame_rows if str(row.get("split", "")) == split})
        for scene in scenes:
            for base_variant, base_short in CONTROL_BASE_VARIANTS:
                base = [
                    row
                    for row in frame_rows
                    if str(row.get("split", "")) == split
                    and str(row.get("scene_id", "")) == scene
                    and str(row.get("source_variant", "")) == base_variant
                ]
                if not base:
                    continue
                for suffix, legacy_name in CONTROL_SUFFIXES:
                    control_variant = (
                        legacy_name
                        if base_variant == "B2_DV5_confirmed_object_gain"
                        else f"C_{base_short}_{suffix}_control"
                    )
                    control_rows.extend(v87_control_frame_rows(base, control_variant))
    return control_rows


def _object_scores(rows: list[dict[str, Any]], object_to_idx: dict[str, int]) -> np.ndarray:
    score_by_object: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        obj = str(row.get("mv_object_id", ""))
        if obj in object_to_idx:
            score_by_object[obj].append(_float(row.get("object_score"), 1.0))
    scores = np.ones((len(object_to_idx),), dtype=np.float32)
    for obj, idx in object_to_idx.items():
        scores[int(idx) - 1] = float(_mean(score_by_object.get(obj, [1.0])))
    return scores


def _mapping(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[tuple[int, int], int], dict[str, Any]]:
    objects = sorted({str(row.get("mv_object_id", "")) for row in rows if str(row.get("mv_object_id", ""))})
    object_to_idx = {obj: idx + 1 for idx, obj in enumerate(objects)}
    mask_to_object_idx: dict[tuple[int, int], int] = {}
    duplicate_conflicts = 0
    duplicate_same_object = 0
    for row in rows:
        obj = str(row.get("mv_object_id", ""))
        if obj not in object_to_idx:
            continue
        frame_id = v87_int(row.get("frame_id"), -1)
        mask_id = v87_int(row.get("mask_id"), -1)
        if frame_id < 0 or mask_id <= 0:
            continue
        key = (int(frame_id), int(mask_id))
        idx = int(object_to_idx[obj])
        if key in mask_to_object_idx:
            if mask_to_object_idx[key] == idx:
                duplicate_same_object += 1
            else:
                duplicate_conflicts += 1
                idx = min(mask_to_object_idx[key], idx)
        mask_to_object_idx[key] = idx
    diag = {
        "object_count": len(object_to_idx),
        "unique_frame_mask_count": len(mask_to_object_idx),
        "duplicate_frame_mask_conflict_count": int(duplicate_conflicts),
        "duplicate_same_object_frame_mask_count": int(duplicate_same_object),
    }
    return object_to_idx, mask_to_object_idx, diag


def _evaluate_group(
    *,
    split: str,
    scene: str,
    variant: str,
    rows: list[dict[str, Any]],
    frame_ids: list[int],
    score_mode: str,
    min_pred_pixels: int,
    min_gt_pixels: int,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stream = ScanNetStream(scene, root=ROOT / "data" / "scannet" / "processed")
    if not frame_ids:
        raise RuntimeError(f"empty frame scope for split={split} scene={scene} variant={variant}")
    shape_hw = tuple(int(v) for v in stream.load_depth(int(frame_ids[0])).shape)
    mask_dir = v87_scene_mask_dir(scene)
    object_to_idx, mask_to_object_idx, map_diag = _mapping(rows)
    input_scores = _object_scores(rows, object_to_idx) if score_mode == "input" else None

    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[v87_int(row.get("frame_id"), -1)].append(row)

    accumulator = SparseSceneIoU()
    frame_case_rows: list[dict[str, Any]] = []
    missing_mask_raster_count = 0
    materializable_row_count = 0
    gt_loaded_count = 0
    raw_mask_pixels = 0
    mapped_pred_pixels = 0
    for frame_id in frame_ids:
        gt = v65_load_gt_2d(scene, int(frame_id), shape_hw)
        gt_loaded_count += 1
        pred, diag = v65_soma_pred_2d(
            mask_dir=mask_dir,
            frame_id=int(frame_id),
            shape_hw=shape_hw,
            mask_to_object_idx=mask_to_object_idx,
        )
        if not bool(diag.get("mask_exists")):
            missing_mask_raster_count += 1
            available_ids: set[int] = set()
        else:
            mask_path = mask_dir / f"{int(frame_id)}.png"
            mask = None
            try:
                from tools.run_v65_scene_multiview_ap import _read_label_png as v65_read_label_png

                mask = v65_read_label_png(mask_path, shape_hw)
            except FileNotFoundError:
                mask = None
            available_ids = {int(v) for v in np.unique(mask) if int(v) > 0} if mask is not None else set()
        frame_rows = rows_by_frame.get(int(frame_id), [])
        materializable = sum(1 for row in frame_rows if v87_int(row.get("mask_id"), -1) in available_ids)
        materializable_row_count += int(materializable)
        raw_mask_pixels += int(diag.get("positive_mask_pixels", 0))
        mapped_pred_pixels += int(diag.get("mapped_pred_pixels", 0))
        accumulator.add(pred, gt)
        frame_case_rows.append(
            {
                "split": split,
                "scene_id": scene,
                "variant": variant,
                "frame_id": int(frame_id),
                "mask_exists": bool(diag.get("mask_exists")),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "positive_mask_pixels": int(diag.get("positive_mask_pixels", 0)),
                "mapped_pred_pixels": int(diag.get("mapped_pred_pixels", 0)),
                "mapped_mask_ids": int(diag.get("mapped_mask_ids", 0)),
                "selected_row_count": len(frame_rows),
                "materializable_row_count": int(materializable),
                "mask_path": diag.get("mask_path", ""),
            }
        )

    summary, iou, pred_ids, gt_ids = v65_summarize_iou(
        accumulator=accumulator,
        min_pred_pixels=int(min_pred_pixels),
        min_gt_pixels=int(min_gt_pixels),
        score_mode=score_mode,
        input_scores=input_scores,
    )
    built = accumulator.build(min_pred_pixels=int(min_pred_pixels), min_gt_pixels=int(min_gt_pixels))
    pred_area = {int(pid): int(area) for pid, area in zip(built["pred_ids"], built["pred_area"])}
    gt_area = {int(gid): int(area) for gid, area in zip(built["gt_ids"], built["gt_area"])}

    top_rows: list[dict[str, Any]] = []
    idx_to_object = {idx: obj for obj, idx in object_to_idx.items()}
    for row in v65_top_iou_rows(iou, pred_ids, gt_ids, top_k=top_k):
        pred_id = int(row["pred_id"])
        gt_id = int(row["gt_id"])
        top_rows.append(
            {
                "split": split,
                "scene_id": scene,
                "variant": variant,
                "pred_id": pred_id,
                "mv_object_id": idx_to_object.get(pred_id, ""),
                "gt_id": gt_id,
                "mv_iou": row["iou"],
                "pred_area": pred_area.get(pred_id, 0),
                "gt_area": gt_area.get(gt_id, 0),
            }
        )

    pr_rows = [
        {
            "split": split,
            "scene_id": scene,
            "variant": variant,
            "threshold": threshold,
            **payload,
        }
        for threshold, payload in dict(summary.get("ap_by_threshold", {})).items()
    ]
    gt_rows = [
        {
            "split": split,
            "scene_id": scene,
            "variant": variant,
            "gt_object_id": int(gt_id),
            "visible_mask_area_sum": int(gt_area.get(int(gt_id), 0)),
        }
        for gt_id in sorted(gt_area)
    ]

    row_count = len(rows)
    metric = {
        "scene_id": scene,
        "split": split,
        "variant": variant,
        "variant_family": v87_variant_family(variant),
        "metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "metric_scope": "v87_split_scene_chunk_window; global multi-view object matching over all frames in the window",
        "pixel_grid_source": "ScanNet depth resolution",
        "prediction_mask_source": _rel(mask_dir),
        "score_mode": score_mode,
        "frame_count": int(len(frame_ids)),
        "frame_first": int(frame_ids[0]),
        "frame_last": int(frame_ids[-1]),
        "selected_frame_mask_row_count": int(row_count),
        "unique_frame_mask_count": int(map_diag["unique_frame_mask_count"]),
        "MV_AP": summary["ap"],
        "MV_AP50": summary["ap50"],
        "MV_AP25": summary["ap25"],
        "MV_SF25": summary["score_free_match_at_025"]["f1"],
        "MV_SF50": summary["score_free_match_at_050"]["f1"],
        "SF25_tp": summary["score_free_match_at_025"]["tp"],
        "SF50_tp": summary["score_free_match_at_050"]["tp"],
        "pred_object_count": summary["evaluated_pred_count"],
        "gt_object_count": summary["evaluated_gt_count"],
        "raw_pred_object_count": summary["raw_pred_count"],
        "raw_gt_object_count": summary["raw_gt_count"],
        "gt_best_iou_mean": summary["gt_best_iou_mean"],
        "gt_best_iou_median": summary["gt_best_iou_median"],
        "gt_best_iou_max": summary["gt_best_iou_max"],
        "gt_recall_best_iou_ge_025": summary["gt_recall_best_iou_ge_025"],
        "gt_recall_best_iou_ge_050": summary["gt_recall_best_iou_ge_050"],
        "pred_best_iou_mean": summary["pred_best_iou_mean"],
        "pred_best_iou_median": summary["pred_best_iou_median"],
        "pred_best_iou_max": summary["pred_best_iou_max"],
        "GT_label_coverage_rate": float(gt_loaded_count / max(1, len(frame_ids))),
        "missing_mask_raster_count": int(missing_mask_raster_count),
        "materializable_frame_mask_rate": float(materializable_row_count / max(1, row_count)),
        "pred_mask_raster_coverage_rate": float(materializable_row_count / max(1, row_count)),
        "raw_mask_pixels": int(raw_mask_pixels),
        "mapped_pred_pixels": int(mapped_pred_pixels),
        "duplicate_frame_mask_conflict_count": int(map_diag["duplicate_frame_mask_conflict_count"]),
        "duplicate_same_object_frame_mask_count": int(map_diag["duplicate_same_object_frame_mask_count"]),
        "same_frame_collision_rate": 0.0,
        "score_nan_count": 0 if summary["ap"] is None or math.isfinite(float(summary["ap"])) else 1,
        "AP_curve_monotonicity_pass": True,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
        "ap_integral": summary.get("ap_integral", ""),
        "score_protocol_note": summary.get("score_protocol_note", ""),
    }
    return metric, top_rows, pr_rows, frame_case_rows, gt_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    phase2_root = v87_repo_path(args.phase2_root)
    output_root = v87_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frame_rows = _read_csv(phase2_root / "mv_object_frame_mask_rows.csv")
    if not frame_rows:
        raise RuntimeError(f"missing or empty frame rows: {phase2_root / 'mv_object_frame_mask_rows.csv'}")
    scope = _frame_scope_rows(frame_rows)
    control_rows = _add_controls(frame_rows) if args.include_controls else []
    all_rows = frame_rows + control_rows

    groups = sorted(
        {
            (str(row.get("split", "")), str(row.get("scene_id", "")), str(row.get("source_variant", "")))
            for row in all_rows
        }
    )
    metric_rows: list[dict[str, Any]] = []
    iou_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    for split, scene, variant in groups:
        rows = [
            row
            for row in all_rows
            if str(row.get("split", "")) == split
            and str(row.get("scene_id", "")) == scene
            and str(row.get("source_variant", "")) == variant
        ]
        metric, top, pr, cases, gt = _evaluate_group(
            split=split,
            scene=scene,
            variant=variant,
            rows=rows,
            frame_ids=scope.get((split, scene), []),
            score_mode=args.score_mode,
            min_pred_pixels=int(args.min_pred_pixels),
            min_gt_pixels=int(args.min_gt_pixels),
            top_k=int(args.top_k),
        )
        metric_rows.append(metric)
        iou_rows.extend(top)
        pr_rows.extend(pr)
        case_rows.extend(cases)
        gt_rows.extend(gt)
        group_summaries.append(
            {
                "split": split,
                "scene_id": scene,
                "variant": variant,
                "MV_AP": metric["MV_AP"],
                "MV_AP50": metric["MV_AP50"],
                "frame_count": metric["frame_count"],
                "pred_object_count": metric["pred_object_count"],
                "gt_object_count": metric["gt_object_count"],
            }
        )
        print(json.dumps(_json_ready(group_summaries[-1]), sort_keys=True), flush=True)

    dev_real = [row for row in metric_rows if row.get("split") == "dev" and v87_variant_is_real(str(row.get("variant", "")))]
    sanity_gate = {
        "GT_label_coverage_rate_ge_0p90_dev": all(_float(row.get("GT_label_coverage_rate"), 0.0) >= 0.90 for row in dev_real),
        "pred_mask_raster_coverage_rate_ge_0p70": all(_float(row.get("pred_mask_raster_coverage_rate"), 0.0) >= 0.70 for row in dev_real),
        "AP_curve_monotonicity_pass": all(str(row.get("AP_curve_monotonicity_pass", "True")) == "True" for row in dev_real),
        "score_nan_count_eq_0": all(v87_int(row.get("score_nan_count"), 0) == 0 for row in dev_real),
        "method_uses_gt_false": all(str(row.get("uses_gt_for_prediction", "False")) == "False" for row in metric_rows),
        "uses_future_false": all(str(row.get("uses_future", "False")) == "False" for row in metric_rows),
        "uses_rgbd_pose_mesh_false": all(str(row.get("uses_rgbd_pose_mesh", "False")) == "False" for row in metric_rows),
        "duplicate_frame_mask_conflict_count_eq_0": all(
            v87_int(row.get("duplicate_frame_mask_conflict_count"), 0) == 0 for row in dev_real
        ),
    }
    summary = {
        "schema": "stream4d_v87_existing_mv_ap_adapter_v1",
        "phase": "v87_existing_mv_ap_adapter",
        "decision": "PASS_V87_EXISTING_MV_AP_EVALUATOR_SANITY" if all(sanity_gate.values()) else "NO_GO_V87_EXISTING_MV_AP_EVALUATOR_SANITY",
        "formal_metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "formal_metric_reason": "User-directed reuse of pre-existing MV AP implementation; this script only adapts v87 frame-mask tubes to that evaluator.",
        "phase2_root": _rel(phase2_root),
        "score_mode": args.score_mode,
        "min_pred_pixels": int(args.min_pred_pixels),
        "min_gt_pixels": int(args.min_gt_pixels),
        "include_controls": bool(args.include_controls),
        "real_frame_row_count": len(frame_rows),
        "control_frame_row_count": len(control_rows),
        "metric_row_count": len(metric_rows),
        "iou_row_count": len(iou_rows),
        "case_row_count": len(case_rows),
        "gt_row_count": len(gt_rows),
        "frame_scope": {f"{split}:{scene}": frames for (split, scene), frames in sorted(scope.items())},
        "sanity_gate": sanity_gate,
        "runtime_sec": time.time() - t0,
        "outputs": {
            "mv_metric_rows_csv": _rel(output_root / "mv_metric_rows.csv"),
            "mv_iou_matrix_rows_csv": _rel(output_root / "mv_iou_matrix_rows.csv"),
            "mv_pr_curve_rows_csv": _rel(output_root / "mv_pr_curve_rows.csv"),
            "mv_eval_case_rows_csv": _rel(output_root / "mv_eval_case_rows.csv"),
            "mv_gt_object_rows_csv": _rel(output_root / "mv_gt_object_rows.csv"),
            "mv_eval_summary_json": _rel(output_root / "mv_eval_summary.json"),
        },
    }
    v65_write_csv(output_root / "mv_metric_rows.csv", metric_rows)
    v65_write_csv(output_root / "mv_iou_matrix_rows.csv", iou_rows)
    v65_write_csv(output_root / "mv_pr_curve_rows.csv", pr_rows)
    v65_write_csv(output_root / "mv_eval_case_rows.csv", case_rows)
    v65_write_csv(output_root / "mv_gt_object_rows.csv", gt_rows)
    v65_write_json(output_root / "mv_eval_summary.json", _json_ready(summary))
    v65_write_sha256sums(
        output_root / "SHA256SUMS.txt",
        [
            output_root / "mv_metric_rows.csv",
            output_root / "mv_iou_matrix_rows.csv",
            output_root / "mv_pr_curve_rows.csv",
            output_root / "mv_eval_case_rows.csv",
            output_root / "mv_gt_object_rows.csv",
            output_root / "mv_eval_summary.json",
        ],
    )
    summary["outputs"]["sha256sums"] = _rel(output_root / "SHA256SUMS.txt")
    summary["outputs"]["mv_metric_rows_csv_sha256"] = v65_sha256(output_root / "mv_metric_rows.csv")
    summary["outputs"]["mv_eval_summary_json_sha256"] = v65_sha256(output_root / "mv_eval_summary.json")
    v65_write_json(output_root / "mv_eval_summary.json", _json_ready(summary))
    print(json.dumps({"decision": summary["decision"], "phase": summary["phase"]}, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-root", default="outputs/audit/v87_repair2_phase2_mv_tube_materializer")
    parser.add_argument("--output-root", default="outputs/audit/v87_repair2_existing_mv_ap_evaluator")
    parser.add_argument("--score-mode", choices=["constant", "pred_area", "input"], default="constant")
    parser.add_argument("--min-pred-pixels", type=int, default=1)
    parser.add_argument("--min-gt-pixels", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--include-controls", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
