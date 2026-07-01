#!/usr/bin/env python3
"""GT-score oracle upper bound for v99 holdout candidate rows.

Diagnostic only: this uses GT IoU as prediction score and must not be reported
as a method result. It answers whether score-only repair has enough headroom for
the existing holdout object/mask universe.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v98_1_canonical_holdout_metrics as holdout  # noqa: E402
from tools.run_v65_scene_multiview_ap import SparseSceneIoU, _load_gt_2d, _summarize_iou  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10c_holdout_oracle_upper_bound"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
HOLDOUT_SOURCE_ROWS = AUDIT_ROOT / "v98_phase13_holdout/source_container_rows.csv"
HOLDOUT_FIXED_ROWS = AUDIT_ROOT / "v99_phase2_f2_strengthening/holdout_mv_object_frame_mask_rows.csv"
ORACLE_VARIANT = "V99P10C_GT_ORACLE_BEST_IOU_SCORE_UPPER_BOUND"


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        try:
            return q.resolve().relative_to(STREAM3D_ROOT).as_posix()
        except ValueError:
            return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


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


def _f1(precision: Any, recall: Any) -> float:
    p = _num(precision)
    r = _num(recall)
    return float(2.0 * p * r / max(1e-12, p + r))


def _metric_row(variant: str, scene: str, summary: dict[str, Any], *, duplicate_conflicts: int, missing_masks: int) -> dict[str, Any]:
    sf50 = summary.get("score_free_match_at_050") or {}
    sf25 = summary.get("score_free_match_at_025") or {}
    return {
        "schema_version": "stream4d_v99_phase10c_oracle_metric_scene_v1",
        "phase_id": "v99_phase10c_holdout_oracle_upper_bound",
        "variant_id": variant,
        "scene_id": scene,
        "MV_AP_window": summary.get("ap"),
        "MV_AP50_window": summary.get("ap50"),
        "MV_AP25_window": summary.get("ap25"),
        "ScoreFreeMatch50_window": _f1(sf50.get("precision"), sf50.get("recall")),
        "ScoreFreeMatch50_precision_window": sf50.get("precision"),
        "ScoreFreeMatch50_recall_window": sf50.get("recall"),
        "ScoreFreeMatch25_window": _f1(sf25.get("precision"), sf25.get("recall")),
        "frame_count": summary.get("frame_count", ""),
        "gt_object_count": summary.get("evaluated_gt_count"),
        "pred_object_count": summary.get("evaluated_pred_count"),
        "same_frame_collision_count": int(duplicate_conflicts),
        "missing_mask_raster_count": int(missing_masks),
        "metric_scope": "same_scene_temporal_holdout_local_window_gt_projection",
        "canonical_metric_source": "run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "adapter_contract": "v90_local_window_scope_from_holdout_source_rows",
        "uses_gt_for_prediction": True,
        "uses_gt_for_eval": True,
        "uses_future": False,
    }


def _load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(HOLDOUT_FIXED_ROWS):
        rows.append(
            {
                **row,
                "mask_id": int(row["mask_id"]),
                "frame_id": int(row["frame_id"]),
                "score": _num(row.get("score")),
            }
        )
    return rows


def _evaluate_oracle(rows: list[dict[str, Any]], scope: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    oracle_score_rows: list[dict[str, Any]] = []
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
        original_scores: dict[str, float] = {}
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
            original_scores[scoped_oid] = max(float(original_scores.get(scoped_oid, 0.0)), _num(chosen.get("score"), 1.0))
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
        _base_summary, iou, pred_ids, gt_ids = _summarize_iou(
            accumulator=acc,
            min_pred_pixels=1,
            min_gt_pixels=1,
            score_mode="input",
            input_scores=_score_array(object_to_idx, original_scores),
        )
        oracle_scores: dict[str, float] = {}
        for pred_index, pred_id in enumerate(pred_ids):
            oid = idx_to_obj.get(int(pred_id), "")
            if not oid:
                continue
            best_iou = float(np.max(iou[pred_index, :])) if iou.shape[1] else 0.0
            oracle_scores[oid] = best_iou
            oracle_score_rows.append(
                {
                    "schema_version": "stream4d_v99_phase10c_oracle_score_v1",
                    "phase_id": "v99_phase10c_holdout_oracle_upper_bound",
                    "variant_id": ORACLE_VARIANT,
                    "scene_id": scene,
                    "pred_id": int(pred_id),
                    "mv_object_id": oid,
                    "oracle_best_iou_score": best_iou,
                    "original_score": original_scores.get(oid, ""),
                    "uses_gt_for_prediction": True,
                    "uses_future": False,
                }
            )
        oracle_summary, oracle_iou, oracle_pred_ids, oracle_gt_ids = _summarize_iou(
            accumulator=acc,
            min_pred_pixels=1,
            min_gt_pixels=1,
            score_mode="input",
            input_scores=_score_array(object_to_idx, oracle_scores),
        )
        metric_rows.append(_metric_row(ORACLE_VARIANT, scene, oracle_summary, duplicate_conflicts=duplicate_conflicts, missing_masks=missing_masks))
        if oracle_iou.size:
            flat = np.argsort(oracle_iou.reshape(-1))[::-1]
            width = oracle_iou.shape[1]
            for flat_idx in flat[:100]:
                pi = int(flat_idx // width)
                gi = int(flat_idx % width)
                val = float(oracle_iou[pi, gi])
                if val <= 0:
                    break
                pred_id = int(oracle_pred_ids[pi])
                top_rows.append(
                    {
                        "schema_version": "stream4d_v99_phase10c_oracle_top_iou_v1",
                        "phase_id": "v99_phase10c_holdout_oracle_upper_bound",
                        "variant_id": ORACLE_VARIANT,
                        "scene_id": scene,
                        "pred_id": pred_id,
                        "mv_object_id": idx_to_obj.get(pred_id, ""),
                        "gt_id": int(oracle_gt_ids[gi]),
                        "iou": val,
                        "uses_gt_for_prediction": True,
                        "uses_future": False,
                    }
                )
    return metric_rows, oracle_score_rows, top_rows


def _aggregate(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v99_phase10c_oracle_metric_aggregate_v1",
        "phase_id": "v99_phase10c_holdout_oracle_upper_bound",
        "variant_id": ORACLE_VARIANT,
        "scene_count": len(metric_rows),
        "mean_MV_AP_window": float(np.mean([_num(row.get("MV_AP_window")) for row in metric_rows])) if metric_rows else 0.0,
        "mean_MV_AP50_window": float(np.mean([_num(row.get("MV_AP50_window")) for row in metric_rows])) if metric_rows else 0.0,
        "mean_MV_AP25_window": float(np.mean([_num(row.get("MV_AP25_window")) for row in metric_rows])) if metric_rows else 0.0,
        "same_frame_collision_count": int(sum(int(_num(row.get("same_frame_collision_count"))) for row in metric_rows)),
        "missing_mask_raster_count": int(sum(int(_num(row.get("missing_mask_raster_count"))) for row in metric_rows)),
        "metric_scope": "same_scene_temporal_holdout_local_window_gt_projection",
        "uses_gt_for_prediction": True,
        "uses_future": False,
    }


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    scope = holdout._load_source_scope(HOLDOUT_SOURCE_ROWS)
    rows = _load_rows()
    metric_rows, oracle_score_rows, top_rows = _evaluate_oracle(rows, scope)
    aggregate = _aggregate(metric_rows)
    f2_holdout_window = float(phase0["F2_base_holdout_MV_AP_window"])
    f2_holdout_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"])
    strict_gate = bool(
        _num(aggregate["mean_MV_AP_window"]) >= f2_holdout_window + 0.005
        and _num(aggregate["mean_MV_AP50_window"]) >= f2_holdout_ap50 + 0.010
    )
    summary = {
        "schema_version": "stream4d_v99_phase10c_holdout_oracle_upper_bound_summary_v1",
        "phase_id": "v99_phase10c_holdout_oracle_upper_bound",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "DIAGNOSTIC_ORACLE_UPPER_BOUND_SCORE_CAN_PASS_GATE" if strict_gate else "DIAGNOSTIC_ORACLE_UPPER_BOUND_SCORE_STILL_FAILS_GATE",
        "formal_claim_allowed": False,
        "uses_gt_for_prediction": True,
        "uses_future": False,
        "oracle_MV_AP_window": aggregate["mean_MV_AP_window"],
        "oracle_MV_AP50_window": aggregate["mean_MV_AP50_window"],
        "oracle_MV_AP25_window": aggregate["mean_MV_AP25_window"],
        "F2_base_holdout_MV_AP_window": f2_holdout_window,
        "F2_base_holdout_MV_AP50_window": f2_holdout_ap50,
        "oracle_delta_vs_F2_base_holdout_window": _num(aggregate["mean_MV_AP_window"]) - f2_holdout_window,
        "strict_holdout_gate_pass_under_oracle_score": strict_gate,
        "row_count": len(rows),
        "pred_object_score_count": len(oracle_score_rows),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "oracle_metric_rows": _rel(OUT_DIR / "oracle_metric_rows.csv"),
            "oracle_metric_aggregate_rows": _rel(OUT_DIR / "oracle_metric_aggregate_rows.csv"),
            "oracle_score_rows": _rel(OUT_DIR / "oracle_score_rows.csv"),
            "oracle_top_iou_rows": _rel(OUT_DIR / "oracle_top_iou_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "oracle_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "oracle_metric_aggregate_rows.csv", [aggregate])
    _write_csv(OUT_DIR / "oracle_score_rows.csv", oracle_score_rows)
    _write_csv(OUT_DIR / "oracle_top_iou_rows.csv", top_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if strict_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
