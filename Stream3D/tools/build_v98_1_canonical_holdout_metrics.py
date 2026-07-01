#!/usr/bin/env python3
"""Compute v98.1 frozen holdout MV_AP_window with the canonical v65 AP core."""

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

from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


OUT_BASE = ROOT / "outputs/audit"
PHASE13 = OUT_BASE / "v98_phase13_holdout"
DEFAULT_SOURCE_ROWS = PHASE13 / "source_container_rows.csv"
DEFAULT_REAL_ROWS = OUT_BASE / "v98_phase13_holdout_phase9_render_snap/mv_object_frame_mask_rows.csv"
DEFAULT_CONTROL_ROWS = OUT_BASE / "v98_phase13_holdout_phase10_controls/control_mv_object_frame_mask_rows.csv"
DEFAULT_OUTPUT_ROOT = OUT_BASE / "v98_phase13_holdout_canonical_metrics"

BASE_VARIANT = "F2_mask_centered_plus_semantic_residual_proxy"
SCORE_POLICY = "frame_count"
RUN_ID = "v98_1_phase13_canonical_holdout_metrics"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        return int(round(_num(value, default)))
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


def _read_label(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask label: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    label = np.asarray(image, dtype=np.int64)
    if shape_hw is not None and label.shape[:2] != shape_hw:
        label = cv2.resize(label, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(label, dtype=np.int64)


def _window_scoped_gt(gt: np.ndarray, window_key: str, gt_id_map: dict[tuple[str, int], int]) -> np.ndarray:
    out = np.zeros_like(gt, dtype=np.int64)
    for raw in np.unique(gt):
        raw_i = int(raw)
        if raw_i <= 0:
            continue
        key = (window_key, raw_i)
        if key not in gt_id_map:
            gt_id_map[key] = len(gt_id_map) + 1
        out[gt == raw_i] = gt_id_map[key]
    return out


def _score_array(object_to_idx: dict[str, int], object_scores: dict[str, float]) -> np.ndarray:
    scores = np.ones((len(object_to_idx),), dtype=np.float32)
    for oid, idx in object_to_idx.items():
        scores[idx - 1] = float(object_scores.get(oid, 1.0))
    return scores


def _top_iou_rows(iou: np.ndarray, pred_ids: np.ndarray, gt_ids: np.ndarray, *, top_k: int = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if iou.size == 0:
        return rows
    pred_ids = np.asarray(pred_ids)
    gt_ids = np.asarray(gt_ids)
    flat = np.argsort(iou.reshape(-1))[::-1]
    width = iou.shape[1]
    for flat_idx in flat[:top_k]:
        pi = int(flat_idx // width)
        gi = int(flat_idx % width)
        val = float(iou[pi, gi])
        if val <= 0:
            break
        rows.append({"pred_id": int(pred_ids[pi]), "gt_id": int(gt_ids[gi]), "iou": val})
    return rows


def _load_source_scope(source_rows: Path) -> dict[str, Any]:
    frames_by_scene_window: dict[tuple[str, str], set[int]] = defaultdict(set)
    mask_path_by_frame: dict[tuple[str, int], Path] = {}
    rows_by_frame: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    windows_by_scene: dict[str, set[str]] = defaultdict(set)
    uses_future = False
    uses_gt_for_prediction = False
    for row in _read_csv(source_rows):
        scene = row.get("scene_id", "")
        window = row.get("window_id", "")
        frame = _int(row.get("frame_id"), -1)
        if not scene or not window or frame < 0:
            continue
        frames_by_scene_window[(scene, window)].add(frame)
        windows_by_scene[scene].add(window)
        rows_by_frame[(scene, frame)].append(row)
        if row.get("mask_path"):
            mask_path_by_frame.setdefault((scene, frame), _project(row["mask_path"]))
        uses_future = uses_future or _bool(row.get("uses_future"))
        uses_gt_for_prediction = uses_gt_for_prediction or _bool(row.get("uses_gt_for_prediction"))
    frames_by_scene = {
        scene: sorted({frame for (s, _w), frames in frames_by_scene_window.items() if s == scene for frame in frames})
        for scene in sorted(windows_by_scene)
    }
    frame_to_window: dict[tuple[str, int], str] = {}
    for (scene, window), frames in frames_by_scene_window.items():
        for frame in frames:
            frame_to_window[(scene, frame)] = window
    return {
        "frames_by_scene_window": {key: sorted(vals) for key, vals in frames_by_scene_window.items()},
        "frames_by_scene": frames_by_scene,
        "windows_by_scene": {scene: sorted(windows) for scene, windows in windows_by_scene.items()},
        "frame_to_window": frame_to_window,
        "mask_path_by_frame": mask_path_by_frame,
        "rows_by_frame": rows_by_frame,
        "uses_future": uses_future,
        "uses_gt_for_prediction": uses_gt_for_prediction,
    }


def _load_selected_rows(path: Path, *, allowed_variants: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for row in _read_csv(path):
        variant = row.get("variant_id", "")
        if allowed_variants is not None and variant not in allowed_variants:
            continue
        scene = row.get("scene_id", "")
        frame = _int(row.get("frame_id"), -1)
        mask_id = _int(row.get("selected_mask_id", row.get("mask_id_or_generated_id")), -1)
        oid = row.get("mv_object_id") or row.get("object_id", "")
        if not variant or not scene or frame < 0 or mask_id <= 0 or not oid:
            continue
        rows.append(
            {
                "split": "holdout",
                "variant_id": variant,
                "variant": variant,
                "scene_id": scene,
                "frame_id": frame,
                "mask_id": mask_id,
                "mv_object_id": oid,
                "object_id": oid,
                "score": _num(row.get("score"), _num(row.get("object_score"), 1.0)),
                "support_iou": _num(row.get("support_iou")),
                "mask_precision": _num(row.get("mask_precision")),
                "support_area": _num(row.get("support_area")),
                "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                "uses_future": _bool(row.get("uses_future")),
            }
        )
    return rows


def _apply_score_policy(rows: list[dict[str, Any]], *, base_variant: str, policy: str) -> list[dict[str, Any]]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("variant_id") == base_variant:
            by_object[str(row.get("mv_object_id", ""))].append(row)
    max_frame_count = 1.0
    stats: dict[str, dict[str, float]] = {}
    for oid, vals in by_object.items():
        frames = {(row["scene_id"], int(row["frame_id"])) for row in vals}
        frame_count = float(len(frames))
        max_frame_count = max(max_frame_count, frame_count)
        stats[oid] = {"frame_count": frame_count}
    scored: list[dict[str, Any]] = []
    variant_id = f"{base_variant}__score_{policy}"
    for oid, vals in by_object.items():
        st = stats.get(oid, {})
        if policy != "frame_count":
            raise ValueError(f"holdout frozen policy must be frame_count, got {policy}")
        score = st.get("frame_count", 0.0) / max_frame_count
        for row in vals:
            scored.append({**row, "variant_id": variant_id, "variant": variant_id, "score": float(score), "score_policy": policy})
    return scored


def _metric_row(variant: str, scene: str, summary: dict[str, Any], *, duplicate_conflicts: int, missing_masks: int) -> dict[str, Any]:
    sf50 = summary.get("score_free_match_at_050") or {}
    sf25 = summary.get("score_free_match_at_025") or {}
    return {
        "schema_version": "stream4d_v98_1_holdout_canonical_mv_ap_window_scene_row_v1",
        "phase_id": "v98_phase13_holdout",
        "run_id": RUN_ID,
        "variant_id": variant,
        "scene_id": scene,
        "MV_AP_window": summary.get("ap"),
        "MV_AP50_window": summary.get("ap50"),
        "MV_AP25_window": summary.get("ap25"),
        "ScoreFreeMatch50_window": _f1(sf50.get("precision"), sf50.get("recall")),
        "ScoreFreeMatch50_precision_window": sf50.get("precision"),
        "ScoreFreeMatch50_recall_window": sf50.get("recall"),
        "ScoreFreeMatch25_window": _f1(sf25.get("precision"), sf25.get("recall")),
        "frame_count": summary.get("frame_count"),
        "gt_object_count": summary.get("evaluated_gt_count"),
        "pred_object_count": summary.get("evaluated_pred_count"),
        "same_frame_collision_count": int(duplicate_conflicts),
        "missing_mask_raster_count": int(missing_masks),
        "metric_scope": "same_scene_temporal_holdout_local_window_gt_projection",
        "canonical_metric_source": "run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "adapter_contract": "v90_local_window_scope_from_holdout_source_rows",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
    }


def _evaluate_variant(variant: str, rows: list[dict[str, Any]], scope: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    rows_by_frame_mask: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene = str(row["scene_id"])
        frame = int(row["frame_id"])
        window = scope["frame_to_window"].get((scene, frame), "")
        if not window:
            continue
        rows_by_frame_mask[(scene, window, frame, int(row["mask_id"]))].append(row)
    for scene in sorted(scope["frames_by_scene"]):
        object_ids = set()
        object_scores: dict[str, float] = {}
        mask_to_obj: dict[tuple[str, int, int], str] = {}
        duplicate_conflicts = 0
        for (row_scene, window, frame, mask_id), vals in sorted(rows_by_frame_mask.items()):
            if row_scene != scene:
                continue
            vals_sorted = sorted(vals, key=lambda r: (_num(r.get("score")), str(r.get("mv_object_id"))), reverse=True)
            chosen = vals_sorted[0]
            if len({str(v.get("mv_object_id", "")) for v in vals_sorted}) > 1:
                duplicate_conflicts += len(vals_sorted) - 1
            scoped_oid = f"{window}|{chosen['mv_object_id']}"
            object_ids.add(scoped_oid)
            object_scores[scoped_oid] = max(float(object_scores.get(scoped_oid, 0.0)), _num(chosen.get("score"), 1.0))
            mask_to_obj[(window, frame, mask_id)] = scoped_oid
        object_to_idx = {oid: idx + 1 for idx, oid in enumerate(sorted(object_ids))}
        idx_to_obj = {idx: oid for oid, idx in object_to_idx.items()}
        acc = SparseSceneIoU()
        gt_id_map: dict[tuple[str, int], int] = {}
        missing_masks = 0
        for window in scope["windows_by_scene"].get(scene, []):
            for frame in scope["frames_by_scene_window"].get((scene, window), []):
                mask_path = scope["mask_path_by_frame"].get((scene, int(frame)))
                label: np.ndarray | None = None
                if mask_path is not None and mask_path.exists():
                    label = _read_label(mask_path)
                    shape_hw = tuple(int(v) for v in label.shape[:2])
                else:
                    missing_masks += 1
                    shape_hw = (968, 1296)
                gt = _load_gt_2d(scene, int(frame), shape_hw)
                if label is None:
                    label = np.zeros(shape_hw, dtype=np.int64)
                elif label.shape[:2] != shape_hw:
                    label = _read_label(mask_path, shape_hw) if mask_path is not None else np.zeros(shape_hw, dtype=np.int64)
                pred = np.zeros(shape_hw, dtype=np.int64)
                for mask_id in np.unique(label):
                    mask_id = int(mask_id)
                    if mask_id <= 0:
                        continue
                    scoped_oid = mask_to_obj.get((window, int(frame), mask_id), "")
                    pred_id = object_to_idx.get(scoped_oid, 0)
                    if pred_id > 0:
                        pred[label == mask_id] = pred_id
                gt_window = _window_scoped_gt(gt, window, gt_id_map)
                acc.add(pred, gt_window)
                case_rows.append(
                    {
                        "schema_version": "stream4d_v98_1_holdout_canonical_case_v1",
                        "phase_id": "v98_phase13_holdout",
                        "run_id": RUN_ID,
                        "variant_id": variant,
                        "scene_id": scene,
                        "window_id": window,
                        "frame_id": int(frame),
                        "gt_positive_pixels": int(np.count_nonzero(gt > 0)),
                        "pred_positive_pixels": int(np.count_nonzero(pred > 0)),
                        "mask_path": _rel(mask_path) if mask_path is not None else "",
                        "mask_exists": bool(mask_path is not None and mask_path.exists()),
                        "support_policy": "holdout_local_window_gt_projection",
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
        metric_rows.append(_metric_row(variant, scene, summary, duplicate_conflicts=duplicate_conflicts, missing_masks=missing_masks))
        for top in _top_iou_rows(iou, pred_ids, gt_ids):
            top_rows.append(
                {
                    "schema_version": "stream4d_v98_1_holdout_canonical_top_iou_v1",
                    "phase_id": "v98_phase13_holdout",
                    "run_id": RUN_ID,
                    "variant_id": variant,
                    "scene_id": scene,
                    "pred_id": top["pred_id"],
                    "mv_object_id": idx_to_obj.get(int(top["pred_id"]), ""),
                    "gt_id": top["gt_id"],
                    "iou": top["iou"],
                    "matrix_scope": "holdout_local_window_support",
                }
            )
    return metric_rows, case_rows, top_rows


def _aggregate(metric_rows: list[dict[str, Any]], *, family: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row.get("variant_id", ""))].append(row)
    out: list[dict[str, Any]] = []
    for variant, rows in sorted(grouped.items()):
        out.append(
            {
                "schema_version": "stream4d_v98_1_holdout_canonical_mv_ap_window_aggregate_v1",
                "phase_id": "v98_phase13_holdout",
                "run_id": RUN_ID,
                "variant_id": variant,
                "variant_family": family,
                "scene_count": len(rows),
                "mean_MV_AP_window": _mean([row.get("MV_AP_window") for row in rows]),
                "mean_MV_AP50_window": _mean([row.get("MV_AP50_window") for row in rows]),
                "mean_MV_AP25_window": _mean([row.get("MV_AP25_window") for row in rows]),
                "mean_score_free_Match50_window": _mean([row.get("ScoreFreeMatch50_window") for row in rows]),
                "same_frame_collision_count": int(sum(_int(row.get("same_frame_collision_count")) for row in rows)),
                "missing_mask_raster_count": int(sum(_int(row.get("missing_mask_raster_count")) for row in rows)),
                "metric_scope": "same_scene_temporal_holdout_local_window_gt_projection",
                "canonical_metric_source": "run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--real-input-rows", default=str(DEFAULT_REAL_ROWS))
    parser.add_argument("--control-input-rows", default=str(DEFAULT_CONTROL_ROWS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--base-variant", default=BASE_VARIANT)
    parser.add_argument("--score-policy", default=SCORE_POLICY, choices=["frame_count"])
    args = parser.parse_args()

    started = time.time()
    source_rows = _project(args.source_rows)
    real_input_rows = _project(args.real_input_rows)
    control_input_rows = _project(args.control_input_rows)
    output_root = _project(args.output_root)

    scope = _load_source_scope(source_rows)
    base_rows = _load_selected_rows(real_input_rows, allowed_variants={args.base_variant})
    frozen_real_rows = _apply_score_policy(base_rows, base_variant=args.base_variant, policy=args.score_policy)
    control_rows = _load_selected_rows(control_input_rows)
    all_eval_rows = frozen_real_rows + control_rows

    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant_id"]) for row in all_eval_rows}):
        rows = [row for row in all_eval_rows if row.get("variant_id") == variant]
        m, c, t = _evaluate_variant(variant, rows, scope)
        metric_rows.extend(m)
        case_rows.extend(c)
        top_rows.extend(t)

    real_metric_rows = [row for row in metric_rows if row.get("variant_id") == f"{args.base_variant}__score_{args.score_policy}"]
    control_metric_rows = [row for row in metric_rows if row.get("variant_id") != f"{args.base_variant}__score_{args.score_policy}"]
    real_agg = _aggregate(real_metric_rows, family="frozen_real")
    control_agg = _aggregate(control_metric_rows, family="holdout_control")
    aggregate_rows = real_agg + control_agg
    frozen = real_agg[0] if real_agg else {}
    c0 = next((row for row in control_agg if row.get("variant_id") == "C0_mask_only_frame_masks"), {})
    best_control = max(control_agg, key=lambda r: (_num(r.get("mean_MV_AP_window"), -1), _num(r.get("mean_MV_AP50_window"), -1)), default={})
    frozen_ap = _num(frozen.get("mean_MV_AP_window"), -1.0)
    frozen_ap50 = _num(frozen.get("mean_MV_AP50_window"), -1.0)
    b0_ap = _num(c0.get("mean_MV_AP_window"), float("nan"))
    b0_ap50 = _num(c0.get("mean_MV_AP50_window"), float("nan"))
    best_control_ap = _num(best_control.get("mean_MV_AP_window"), float("nan"))
    best_control_ap50 = _num(best_control.get("mean_MV_AP50_window"), float("nan"))
    gates = {
        "holdout_MV_AP_window_ge_holdout_B0_plus_0p008": math.isfinite(b0_ap) and frozen_ap >= b0_ap + 0.008,
        "holdout_MV_AP50_window_ge_holdout_B0_plus_0p015": math.isfinite(b0_ap50) and frozen_ap50 >= b0_ap50 + 0.015,
        "holdout_MV_AP_window_ge_holdout_best_control_plus_0p005": math.isfinite(best_control_ap) and frozen_ap >= best_control_ap + 0.005,
        "holdout_MV_AP50_window_ge_holdout_best_control_plus_0p010_diagnostic": math.isfinite(best_control_ap50) and frozen_ap50 >= best_control_ap50 + 0.010,
        "same_frame_collision_count_eq_0": _int(frozen.get("same_frame_collision_count"), 1) == 0,
        "missing_mask_raster_count_eq_0": _int(frozen.get("missing_mask_raster_count"), 1) == 0,
        "uses_gt_for_prediction_false": not any(_bool(row.get("uses_gt_for_prediction")) for row in all_eval_rows) and not scope["uses_gt_for_prediction"],
        "uses_future_false": not any(_bool(row.get("uses_future")) for row in all_eval_rows) and not scope["uses_future"],
    }
    required_gate_names = [
        "holdout_MV_AP_window_ge_holdout_B0_plus_0p008",
        "holdout_MV_AP50_window_ge_holdout_B0_plus_0p015",
        "holdout_MV_AP_window_ge_holdout_best_control_plus_0p005",
        "same_frame_collision_count_eq_0",
        "missing_mask_raster_count_eq_0",
        "uses_gt_for_prediction_false",
        "uses_future_false",
    ]
    holdout_gate_pass = all(bool(gates[name]) for name in required_gate_names)
    decision = "GO_V98_HOLDOUT_LOCAL_MV_AP_WINDOW" if holdout_gate_pass else "NO_GO_V98_HOLDOUT_LOCAL_MV_AP_WINDOW"

    _write_csv(output_root / "canonical_holdout_mv_metric_rows.csv", metric_rows)
    _write_csv(output_root / "canonical_holdout_mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(output_root / "canonical_holdout_case_rows.csv", case_rows)
    _write_csv(output_root / "canonical_holdout_top_iou_rows.csv", top_rows)
    summary = {
        "schema": "stream4d_v98_1_phase13_canonical_holdout_summary_v1",
        "phase_id": "v98_phase13_holdout",
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": decision,
        "holdout_gate_pass": holdout_gate_pass,
        "frozen_real_variant": frozen.get("variant_id", ""),
        "frozen_real_MV_AP_window": frozen.get("mean_MV_AP_window", ""),
        "frozen_real_MV_AP50_window": frozen.get("mean_MV_AP50_window", ""),
        "holdout_B0_variant": c0.get("variant_id", ""),
        "holdout_B0_MV_AP_window": c0.get("mean_MV_AP_window", ""),
        "holdout_B0_MV_AP50_window": c0.get("mean_MV_AP50_window", ""),
        "holdout_best_control_variant": best_control.get("variant_id", ""),
        "holdout_best_control_MV_AP_window": best_control.get("mean_MV_AP_window", ""),
        "holdout_best_control_MV_AP50_window": best_control.get("mean_MV_AP50_window", ""),
        "gate_results": gates,
        "required_gate_names": required_gate_names,
        "MV_AP_scene_status": "not_computed_v90_phase10_local2history_not_started",
        "local2history_allowed": bool(holdout_gate_pass),
        "metric_source_window": "v65 SparseSceneIoU/_summarize_iou via holdout local-window adapter",
        "metric_source_scene": "not_computed; requires scene/history object ids after holdout pass",
        "source_rows": source_rows,
        "real_input_rows": real_input_rows,
        "control_input_rows": control_input_rows,
        "runtime_sec": float(time.time() - started),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(output_root / "canonical_holdout_summary.json", summary)
    print(json.dumps(_jsonable(summary), sort_keys=True))


if __name__ == "__main__":
    main()
