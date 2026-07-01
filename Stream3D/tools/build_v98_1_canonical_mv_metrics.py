#!/usr/bin/env python3
"""Recompute v98.1 MV_AP_window with the v90 local-window contract."""

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

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v89_recalc_point_projected_mv_ap as recalc  # noqa: E402
from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


OUT_BASE = ROOT / "outputs/audit"
PHASE9 = OUT_BASE / "v98_phase9_render_snap"
PHASE10 = OUT_BASE / "v98_phase10_controls"
PHASE11 = OUT_BASE / "v98_phase11_failure_decomposition"
PHASE12 = OUT_BASE / "v98_phase12_dev_decision"

INPUT_ROWS = PHASE9 / "mv_object_frame_mask_rows.csv"
LOCAL_EXPORT_ROOT = OUT_BASE / "v89_recalc_point_projected_mv_ap"
WINDOW_SOURCE_STEP = "S3D_L1_local_merged_masks"

RUN_ID = "v98_1_canonical_mv_metrics"

B0_MV_AP_WINDOW = 0.023169647579624655
B0_MV_AP50_WINDOW = 0.07720796704691124
BEST_LOCKED_CONTROL_VARIANT = "P3_C0_area_semantic_hybrid_score"
BEST_LOCKED_CONTROL_MV_AP_WINDOW = 0.05775790465217242
BEST_LOCKED_CONTROL_MV_AP50_WINDOW = 0.17695961955544454
V91_BEST_MV_AP_WINDOW = 0.06799544580104074
V91_BEST_MV_AP50_WINDOW = 0.18017992227130697


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return _rel(value)
    return value


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _read_csv(path: Path) -> list[dict[str, str]]:
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
        writer = csv.DictWriter(handle, fieldnames=keys or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[Any]) -> float:
    vals = [_num(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


def _f1(precision: Any, recall: Any) -> float:
    p = _num(precision)
    r = _num(recall)
    return float(2.0 * p * r / max(1e-12, p + r))


def _load_eval_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _read_csv(path):
        variant = row.get("variant_id", "")
        scene = row.get("scene_id", "")
        frame = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("selected_mask_id", row.get("mask_id_or_generated_id")), -1)
        object_id = row.get("mv_object_id") or row.get("object_id", "")
        if not variant or not scene or frame < 0 or mask_id <= 0 or not object_id:
            continue
        out.append(
            {
                "split": "dev",
                "scene_id": scene,
                "source_variant": variant,
                "variant": variant,
                "mv_object_id": object_id,
                "object_id": object_id,
                "frame_id": frame,
                "mask_id": mask_id,
                "frame_mask_score": _num(row.get("score"), 1.0),
                "object_score": _num(row.get("score"), 1.0),
                "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                "uses_future": _bool(row.get("uses_future")),
                "uses_rgbd_pose_mesh": False,
                "materializable": True,
                "selection_reason": "v98_1_phase9_snap_to_mask_canonical_metric_input",
            }
        )
    return out


def _evaluate_window(eval_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scope = recalc._frame_scope()
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    variants = sorted({str(row["variant"]) for row in eval_rows})
    for variant in variants:
        for scene in ("scene0011_00", "scene0050_00"):
            rows = [row for row in eval_rows if row.get("variant") == variant and row.get("scene_id") == scene]
            if not rows:
                continue
            metric, cases, tops, window_rows = recalc._evaluate_frame_mask_variant_local_window(
                scene=scene,
                split="dev",
                variant=variant,
                frame_ids=scope.get(("dev", scene), []),
                rows=rows,
                score_mode="input",
                local_export_root=LOCAL_EXPORT_ROOT,
                window_source_step=WINDOW_SOURCE_STEP,
            )
            metric_rows.append(
                {
                    **metric,
                    "schema_version": "stream4d_v98_1_canonical_mv_ap_window_scene_row_v1",
                    "phase_id": "v98_phase9_render_snap",
                    "run_id": RUN_ID,
                    "variant_id": variant,
                    "MV_AP_window": metric.get("MV_AP"),
                    "MV_AP50_window": metric.get("MV_AP50"),
                    "MV_AP25_window": metric.get("MV_AP25"),
                    "ScoreFreeMatch50_window": _f1(metric.get("SF50_precision"), metric.get("SF50_recall")),
                    "ScoreFreeMatch50_precision_window": metric.get("SF50_precision"),
                    "ScoreFreeMatch50_recall_window": metric.get("SF50_recall"),
                    "same_frame_collision_count": int(_int(metric.get("duplicate_frame_mask_conflict_count"), 0)),
                    "metric_scope": "local_window_gt_projection",
                    "canonical_metric_source": "run_v89_recalc_point_projected_mv_ap._evaluate_frame_mask_variant_local_window",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            for case in cases:
                case_rows.append({**case, "variant_id": variant})
            for top in tops:
                top_rows.append({**top, "variant_id": variant, "mv_iou": top.get("iou", ""), "matrix_scope": "canonical_local_window_support"})
            for win in window_rows:
                top_rows.append(
                    {
                        "variant_id": variant,
                        "scene_id": win.get("scene_id", ""),
                        "window_index": win.get("window_index", ""),
                        "window_metric_MV_AP": win.get("MV_AP", ""),
                        "window_metric_MV_AP50": win.get("MV_AP50", ""),
                        "matrix_scope": "canonical_window_metric_diagnostic",
                    }
                )
    return metric_rows, case_rows, top_rows


def _read_label(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask label: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    if image.shape[:2] != shape_hw:
        image = cv2.resize(np.asarray(image), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(image, dtype=np.int64)


def _evaluate_scene(eval_rows: list[dict[str, Any]], *, min_pred_pixels: int, min_gt_pixels: int) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
    scope = recalc._frame_scope()
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        by_frame[(row["scene_id"], int(row["frame_id"]))].append(row)
    eval_keys = {(scene, frame) for (split, scene), frames in scope.items() if split == "dev" for frame in frames}
    scene_offsets = {scene: (idx + 1) * 1_000_000 for idx, scene in enumerate(sorted({key[0] for key in eval_keys}))}
    object_index: dict[str, int] = {}
    object_scores: dict[str, float] = {}
    acc = SparseSceneIoU()
    frame_rows: list[dict[str, Any]] = []
    pixel_collision_count = 0
    missing_mask_raster_count = 0
    for scene, frame in sorted(eval_keys):
        mask_dir = recalc._mask_dir(scene)
        mask_path = mask_dir / f"{int(frame)}.png"
        gt_probe = _load_gt_2d(scene, int(frame), (968, 1296))
        shape_hw = tuple(int(v) for v in gt_probe.shape)
        pred = np.zeros(shape_hw, dtype=np.int64)
        if mask_path.exists():
            label = _read_label(mask_path, shape_hw)
        else:
            missing_mask_raster_count += 1
            label = np.zeros(shape_hw, dtype=np.int64)
        for row in sorted(by_frame.get((scene, frame), []), key=lambda r: (-_num(r.get("object_score")), str(r.get("mv_object_id", "")))):
            oid = str(row.get("mv_object_id", ""))
            if not oid:
                continue
            if oid not in object_index:
                object_index[oid] = len(object_index) + 1
            object_scores[oid] = max(float(object_scores.get(oid, 0.0)), _num(row.get("object_score"), 1.0))
            mask = label == int(row.get("mask_id", 0))
            pixel_collision_count += int(np.count_nonzero((pred > 0) & mask))
            pred[(pred == 0) & mask] = object_index[oid]
        gt = np.where(gt_probe > 0, gt_probe + int(scene_offsets.get(scene, 0)), 0).astype(np.int64, copy=False)
        acc.add(pred, gt)
        frame_rows.append(
            {
                "schema_version": "stream4d_v98_1_canonical_mv_ap_scene_frame_v1",
                "phase_id": "v98_phase9_render_snap",
                "run_id": RUN_ID,
                "metric_name": "MV_AP_scene",
                "scene_id": scene,
                "frame_id": int(frame),
                "status": "evaluated_scene_frame_dedup",
                "emitted_object_count": len(by_frame.get((scene, frame), [])),
                "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )
    scores = np.ones((len(object_index),), dtype=np.float32)
    for oid, idx in object_index.items():
        scores[idx - 1] = float(object_scores.get(oid, 1.0))
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=int(min_pred_pixels),
        min_gt_pixels=int(min_gt_pixels),
        score_mode="input",
        input_scores=scores,
    )
    return summary, frame_rows, pixel_collision_count, missing_mask_raster_count


def _aggregate_window(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row.get("variant_id", ""))].append(row)
    out: list[dict[str, Any]] = []
    for variant, rows in sorted(grouped.items()):
        out.append(
            {
                "schema_version": "stream4d_v98_1_canonical_mv_metric_aggregate_v1",
                "phase_id": "v98_phase9_render_snap",
                "run_id": RUN_ID,
                "variant_id": variant,
                "scene_count": len(rows),
                "mean_MV_AP_window": _mean([row.get("MV_AP_window") for row in rows]),
                "mean_MV_AP50_window": _mean([row.get("MV_AP50_window") for row in rows]),
                "mean_MV_AP25_window": _mean([row.get("MV_AP25_window") for row in rows]),
                "mean_score_free_Match50_window": _mean([row.get("ScoreFreeMatch50_window") for row in rows]),
                "mean_gt_object_count": _mean([row.get("gt_object_count") for row in rows]),
                "mean_pred_object_count": _mean([row.get("pred_object_count") for row in rows]),
                "same_frame_collision_count": int(sum(_int(row.get("same_frame_collision_count")) for row in rows)),
                "missing_mask_raster_count": int(sum(_int(row.get("missing_mask_raster_count")) for row in rows)),
                "metric_scope": "local_window_gt_projection",
                "canonical_metric_source": "v89_v90_local_window_support",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-rows", default=str(INPUT_ROWS))
    parser.add_argument("--compute-scene-diagnostic", action="store_true", help="Optional non-v90 diagnostic; default off because v90 leaves MV_AP_scene/local2history pending.")
    parser.add_argument("--min-pred-pixels-scene", type=int, default=64)
    parser.add_argument("--min-gt-pixels-scene", type=int, default=64)
    args = parser.parse_args()

    started = time.time()
    eval_rows = _load_eval_rows(Path(args.input_rows))
    window_metric_rows, window_case_rows, local_top_rows = _evaluate_window(eval_rows)
    aggregate_rows = _aggregate_window(window_metric_rows)
    scene_metric_rows: list[dict[str, Any]] = []
    scene_frame_rows: list[dict[str, Any]] = []
    if args.compute_scene_diagnostic:
        for variant in sorted({row["variant"] for row in eval_rows}):
            rows = [row for row in eval_rows if row["variant"] == variant]
            summary, frames, pixel_collision_count, missing_count = _evaluate_scene(
                rows,
                min_pred_pixels=int(args.min_pred_pixels_scene),
                min_gt_pixels=int(args.min_gt_pixels_scene),
            )
            scene_metric_rows.append(
                {
                    "schema_version": "stream4d_v98_1_canonical_mv_ap_scene_metric_v1",
                    "phase_id": "v98_phase9_render_snap",
                    "run_id": RUN_ID,
                    "variant_id": variant,
                    "metric_scope": "scene_plain_diagnostic_not_v90_gate",
                    "MV_AP_scene": summary.get("ap"),
                    "MV_AP50_scene": summary.get("ap50"),
                    "MV_AP25_scene": summary.get("ap25"),
                    "ScoreFreeMatch50_scene": (summary.get("score_free_match_at_050") or {}).get("recall"),
                    "ScoreFreeMatch25_scene": (summary.get("score_free_match_at_025") or {}).get("recall"),
                    "pred_object_count_scene": summary.get("evaluated_pred_count"),
                    "gt_object_count_scene": summary.get("evaluated_gt_count"),
                    "pixel_collision_count_scene": int(pixel_collision_count),
                    "missing_mask_raster_count_scene": int(missing_count),
                    "min_pred_pixels": int(args.min_pred_pixels_scene),
                    "min_gt_pixels": int(args.min_gt_pixels_scene),
                    "canonical_metric_source": "optional_scene_plain_diagnostic_not_used_for_v90_local_gate",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": True,
                    "uses_future": False,
                }
            )
            scene_frame_rows.extend({**row, "variant_id": variant} for row in frames)
    else:
        for variant in sorted({row["variant"] for row in eval_rows}):
            scene_metric_rows.append(
                {
                    "schema_version": "stream4d_v98_1_canonical_mv_ap_scene_metric_v1",
                    "phase_id": "v98_phase9_render_snap",
                    "run_id": RUN_ID,
                    "variant_id": variant,
                    "metric_scope": "v90_phase10_local2history_pending",
                    "MV_AP_scene": "",
                    "MV_AP50_scene": "",
                    "MV_AP25_scene": "",
                    "ScoreFreeMatch50_scene": "",
                    "scene_metric_status": "not_computed_by_default_v90_contract_leaves_MV_AP_scene_local2history_pending",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": False,
                    "uses_future": False,
                }
            )
    scene_by_variant = {row["variant_id"]: row for row in scene_metric_rows}
    for row in aggregate_rows:
        scene = scene_by_variant.get(row["variant_id"], {})
        row.update(
            {
                "MV_AP_scene": scene.get("MV_AP_scene", ""),
                "MV_AP50_scene": scene.get("MV_AP50_scene", ""),
                "MV_AP25_scene": scene.get("MV_AP25_scene", ""),
                "ScoreFreeMatch50_scene": scene.get("ScoreFreeMatch50_scene", ""),
                "scene_metric_scope": scene.get("metric_scope", ""),
                "scene_metric_status": scene.get("scene_metric_status", ""),
            }
        )

    best_real = max(aggregate_rows, key=lambda row: (_num(row.get("mean_MV_AP_window"), -1), _num(row.get("mean_MV_AP50_window"), -1)), default={})
    best_real_ap = _num(best_real.get("mean_MV_AP_window"), -1)
    best_real_ap50 = _num(best_real.get("mean_MV_AP50_window"), -1)
    gates = {
        "best_real_MV_AP_window_ge_B0_plus_0p010": best_real_ap >= B0_MV_AP_WINDOW + 0.010,
        "best_real_MV_AP50_window_ge_B0_plus_0p020": best_real_ap50 >= B0_MV_AP50_WINDOW + 0.020,
        "best_real_MV_AP_window_ge_best_control_plus_0p005": best_real_ap >= BEST_LOCKED_CONTROL_MV_AP_WINDOW + 0.005,
        "best_real_MV_AP50_window_ge_best_control_plus_0p010": best_real_ap50 >= BEST_LOCKED_CONTROL_MV_AP50_WINDOW + 0.010,
        "best_real_MV_AP_window_ge_v91_plus_0p002": best_real_ap >= V91_BEST_MV_AP_WINDOW + 0.002,
        "same_frame_collision_count_eq_0": _int(best_real.get("same_frame_collision_count"), 1) == 0,
        "missing_mask_raster_count_eq_0": _int(best_real.get("missing_mask_raster_count"), 1) == 0,
    }
    dev_gate_pass = all(gates.values())
    blocker = "NONE" if dev_gate_pass else "CONTROL_OR_RANKING_BLOCKER"
    if not gates["best_real_MV_AP_window_ge_best_control_plus_0p005"] or not gates["best_real_MV_AP50_window_ge_best_control_plus_0p010"]:
        blocker = "CONTROL_BIAS_BLOCKER"
    if not gates["best_real_MV_AP_window_ge_v91_plus_0p002"]:
        blocker = "RANKING_BLOCKER" if blocker == "NONE" else blocker
    decision = "GO_V98_DEV_LOCAL_MV_AP_WINDOW" if dev_gate_pass else "NO_GO_CONTROL_BIAS"

    PHASE9.mkdir(parents=True, exist_ok=True)
    _write_csv(PHASE9 / "canonical_mv_metric_rows.csv", window_metric_rows)
    _write_csv(PHASE9 / "canonical_mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(PHASE9 / "canonical_mv_scene_metric_rows.csv", scene_metric_rows)
    _write_csv(PHASE9 / "canonical_mv_scene_frame_rows.csv", scene_frame_rows)
    _write_csv(PHASE9 / "canonical_mv_case_rows.csv", window_case_rows)
    _write_csv(PHASE9 / "canonical_mv_top_iou_rows.csv", local_top_rows)

    PHASE10.mkdir(parents=True, exist_ok=True)
    _write_csv(
        PHASE10 / "canonical_control_reference_rows.csv",
        [
            {
                "variant_id": BEST_LOCKED_CONTROL_VARIANT,
                "control_type": "locked_reference_from_v98_plan",
                "mean_MV_AP_window": BEST_LOCKED_CONTROL_MV_AP_WINDOW,
                "mean_MV_AP50_window": BEST_LOCKED_CONTROL_MV_AP50_WINDOW,
                "metric_scope": "local_window_gt_projection",
                "metric_source": "plan_locked_reference_not_rerun",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        ],
    )

    failure_rows = []
    if not dev_gate_pass:
        failure_rows.append(
            {
                "blocker": blocker,
                "best_real_variant": best_real.get("variant_id", ""),
                "best_real_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
                "best_real_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
                "best_real_MV_AP_scene": best_real.get("MV_AP_scene", ""),
                "best_real_MV_AP50_scene": best_real.get("MV_AP50_scene", ""),
                "MV_AP_scene_status": best_real.get("scene_metric_status", ""),
                "best_control_variant": BEST_LOCKED_CONTROL_VARIANT,
                "best_control_MV_AP_window": BEST_LOCKED_CONTROL_MV_AP_WINDOW,
                "best_control_MV_AP50_window": BEST_LOCKED_CONTROL_MV_AP50_WINDOW,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    PHASE11.mkdir(parents=True, exist_ok=True)
    _write_csv(PHASE11 / "canonical_failure_decomposition_rows.csv", failure_rows)
    _write_json(
        PHASE11 / "canonical_summary.json",
        {
            "schema": "stream4d_v98_1_canonical_failure_summary_v1",
            "phase_id": "v98_phase11_failure_decomposition",
            "run_id": RUN_ID,
            "created_at": _created_at(),
            "primary_blocker": blocker,
            "best_real_variant": best_real.get("variant_id", ""),
            "best_real_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
            "best_real_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
            "best_real_MV_AP_scene": best_real.get("MV_AP_scene", ""),
            "best_real_MV_AP50_scene": best_real.get("MV_AP50_scene", ""),
            "MV_AP_scene_status": best_real.get("scene_metric_status", ""),
            "gate_results": gates,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    )
    decision_payload = {
        "schema": "stream4d_v98_1_canonical_phase12_dev_decision_v1",
        "phase_id": "v98_phase12_dev_decision",
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": decision,
        "best_real_variant": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
        "best_real_MV_AP_scene": best_real.get("MV_AP_scene", ""),
        "best_real_MV_AP50_scene": best_real.get("MV_AP50_scene", ""),
        "best_control_variant": BEST_LOCKED_CONTROL_VARIANT,
        "best_control_MV_AP_window": BEST_LOCKED_CONTROL_MV_AP_WINDOW,
        "best_control_MV_AP50_window": BEST_LOCKED_CONTROL_MV_AP50_WINDOW,
        "dev_gate_pass": dev_gate_pass,
        "holdout_allowed": bool(dev_gate_pass),
        "local2history_allowed": False,
        "primary_blocker": blocker,
        "gate_results": gates,
        "metric_source_window": "run_v89_recalc_point_projected_mv_ap._evaluate_frame_mask_variant_local_window",
        "metric_source_scene": "not_computed_by_default; v90 treats MV_AP_scene/local2history as pending after local-window gate",
        "runtime_sec": float(time.time() - started),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    PHASE12.mkdir(parents=True, exist_ok=True)
    _write_json(PHASE12 / "canonical_final_dev_decision.json", decision_payload)
    print(json.dumps(decision_payload, sort_keys=True))


if __name__ == "__main__":
    main()
