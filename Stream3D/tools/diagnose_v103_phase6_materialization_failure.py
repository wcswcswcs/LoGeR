#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.build_v103_phase6_mask_clustering_local_object_birth import (  # noqa: E402
    DEFAULT_PHASE2_SCENE0011,
    DEFAULT_PHASE2_SCENE0050,
    _jsonable,
    _load_gt_2d,
    _load_label_png,
    _project,
    _read_json,
    _rel,
    _write_csv,
    _write_json,
)


PHASE_ID = "v103_phase6_materialization_failure_diagnostic"
DEFAULT_PHASE6_ROOT = STREAM3D_ROOT / "outputs/audit/v103_phase6_positive_core_clustering_q5c_repair5_r1"
DEFAULT_OUT = STREAM3D_ROOT / "outputs/audit/v103_phase6_materialization_failure_diag_positive_core_r1"


def _as_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def _finite_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _object_frame_stats(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if rows.empty:
        return out
    object_like = _as_bool(rows["selected_mask_is_object_like"])
    broad = _as_bool(rows["selected_mask_is_broad"])
    work = rows.copy()
    work["_selected_mask_is_object_like_bool"] = object_like
    work["_selected_mask_is_broad_bool"] = broad
    for oid, group in work.groupby("mv_object_id", sort=False):
        out[str(oid)] = {
            "selected_frame_count": int(group["frame_id"].nunique()),
            "selected_row_count": int(len(group)),
            "selected_object_like_rate": float(group["_selected_mask_is_object_like_bool"].mean()),
            "selected_broad_rate": float(group["_selected_mask_is_broad_bool"].mean()),
            "selected_area_mean": float(group.get("selected_mask_area", pd.Series(dtype=float)).mean())
            if "selected_mask_area" in group
            else 0.0,
            "selected_support_mean": float(group.get("support_count", pd.Series(dtype=float)).mean())
            if "support_count" in group
            else 0.0,
            "object_score": float(group.get("object_score", pd.Series([0.0])).max()),
        }
    return out


def _accumulate_scene(
    *,
    scene: str,
    rows: pd.DataFrame,
    phase2_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frame_ids = [int(v) for v in phase2_summary["frame_ids"]]
    mask_root = _project(phase2_summary["mask_root"])
    object_ids = sorted(rows["mv_object_id"].astype(str).unique().tolist())
    object_to_idx = {oid: idx + 1 for idx, oid in enumerate(object_ids)}
    idx_to_object = {idx: oid for oid, idx in object_to_idx.items()}
    frame_stats = _object_frame_stats(rows)

    pred_area: Counter[int] = Counter()
    gt_area: Counter[int] = Counter()
    intersection: Counter[tuple[int, int]] = Counter()
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows.to_dict("records"):
        by_frame[int(row["frame_id"])].append(row)

    frame_emitted_counts: list[int] = []
    missing_mask_rows = 0
    wta_dropped_pixels = 0
    for frame_id in frame_ids:
        mask_path = mask_root / f"{int(frame_id)}.png"
        if not mask_path.exists():
            gt = _load_gt_2d(scene, frame_id, (968, 1296))
            for label, count in zip(*np.unique(gt[gt > 0], return_counts=True)):
                gt_area[int(label)] += int(count)
            continue
        label_img = _load_label_png(mask_path)
        pred = np.zeros(label_img.shape, dtype=np.int64)
        emitted = 0
        ordered = sorted(
            by_frame.get(int(frame_id), []),
            key=lambda r: (-float(r.get("object_score", 0.0)), str(r.get("mv_object_id", ""))),
        )
        for row in ordered:
            oid = str(row["mv_object_id"])
            mask_id = int(row["selected_mask_id"])
            pixels = label_img == mask_id
            if not np.any(pixels):
                missing_mask_rows += 1
                continue
            assign = pixels & (pred == 0)
            wta_dropped_pixels += int(np.count_nonzero(pixels & (pred > 0)))
            pred[assign] = int(object_to_idx[oid])
            emitted += 1
        frame_emitted_counts.append(emitted)
        gt = _load_gt_2d(scene, frame_id, label_img.shape)
        if np.any(pred > 0):
            labels, counts = np.unique(pred[pred > 0], return_counts=True)
            for pred_idx, count in zip(labels, counts):
                pred_area[int(pred_idx)] += int(count)
        if np.any(gt > 0):
            labels, counts = np.unique(gt[gt > 0], return_counts=True)
            for gt_idx, count in zip(labels, counts):
                gt_area[int(gt_idx)] += int(count)
        both = (pred > 0) & (gt > 0)
        if np.any(both):
            pred_vals = pred[both].astype(np.int64, copy=False)
            gt_vals = gt[both].astype(np.int64, copy=False)
            max_gt = int(gt_vals.max()) + 1
            encoded = pred_vals * max_gt + gt_vals
            keys, counts = np.unique(encoded, return_counts=True)
            for key, count in zip(keys.tolist(), counts.tolist()):
                pred_idx = int(key // max_gt)
                gt_idx = int(key % max_gt)
                intersection[(pred_idx, gt_idx)] += int(count)

    best_for_pred: dict[int, tuple[int, float, int]] = {}
    best_for_gt: dict[int, tuple[int, float, int]] = {}
    for (pred_idx, gt_idx), inter in intersection.items():
        union = int(pred_area[pred_idx]) + int(gt_area[gt_idx]) - int(inter)
        iou = float(inter / union) if union > 0 else 0.0
        prev_pred = best_for_pred.get(pred_idx, (0, -1.0, 0))
        if iou > prev_pred[1]:
            best_for_pred[pred_idx] = (gt_idx, iou, int(inter))
        prev_gt = best_for_gt.get(gt_idx, (0, -1.0, 0))
        if iou > prev_gt[1]:
            best_for_gt[gt_idx] = (pred_idx, iou, int(inter))

    pred_rows: list[dict[str, Any]] = []
    for pred_idx, oid in sorted(idx_to_object.items()):
        best_gt, best_iou, inter = best_for_pred.get(pred_idx, (0, 0.0, 0))
        stats = frame_stats.get(oid, {})
        pred_rows.append(
            {
                "schema_version": "stream4d_v103_phase6_pred_object_gt_diag_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "mv_object_id": oid,
                "pred_area_pixels": int(pred_area.get(pred_idx, 0)),
                "best_gt_id": int(best_gt),
                "best_gt_iou": float(max(0.0, best_iou)),
                "best_gt_intersection_pixels": int(inter),
                "selected_frame_count": int(stats.get("selected_frame_count", 0)),
                "selected_object_like_rate": float(stats.get("selected_object_like_rate", 0.0)),
                "selected_broad_rate": float(stats.get("selected_broad_rate", 0.0)),
                "selected_area_mean": float(stats.get("selected_area_mean", 0.0)),
                "selected_support_mean": float(stats.get("selected_support_mean", 0.0)),
                "object_score": float(stats.get("object_score", 0.0)),
                "diagnostic_only_uses_gt": True,
            }
        )

    gt_rows: list[dict[str, Any]] = []
    for gt_idx in sorted(gt_area):
        best_pred, best_iou, inter = best_for_gt.get(gt_idx, (0, 0.0, 0))
        gt_rows.append(
            {
                "schema_version": "stream4d_v103_phase6_gt_object_pred_diag_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "gt_id": int(gt_idx),
                "gt_area_pixels": int(gt_area[gt_idx]),
                "best_pred_object_id": idx_to_object.get(int(best_pred), ""),
                "best_pred_iou": float(max(0.0, best_iou)),
                "best_pred_intersection_pixels": int(inter),
                "diagnostic_only_uses_gt": True,
            }
        )

    pred_ious = [float(row["best_gt_iou"]) for row in pred_rows]
    gt_ious = [float(row["best_pred_iou"]) for row in gt_rows]
    object_like_rate = float(_as_bool(rows["selected_mask_is_object_like"]).mean()) if not rows.empty else 0.0
    broad_rate = float(_as_bool(rows["selected_mask_is_broad"]).mean()) if not rows.empty else 0.0
    summary = {
        "schema_version": "stream4d_v103_phase6_materialization_scene_diag_summary_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "pred_object_count": int(len(pred_rows)),
        "gt_object_count": int(len(gt_rows)),
        "pred_to_gt_count_ratio": float(len(pred_rows) / max(1, len(gt_rows))),
        "selected_row_count": int(len(rows)),
        "selected_object_like_rate": object_like_rate,
        "selected_broad_rate": broad_rate,
        "frame_emitted_mean": float(np.mean(frame_emitted_counts)) if frame_emitted_counts else 0.0,
        "frame_emitted_min": int(np.min(frame_emitted_counts)) if frame_emitted_counts else 0,
        "frame_emitted_max": int(np.max(frame_emitted_counts)) if frame_emitted_counts else 0,
        "pred_best_gt_iou": _finite_stats(pred_ious),
        "gt_best_pred_iou": _finite_stats(gt_ious),
        "gt_iou_ge_0p25_rate": float(np.mean(np.asarray(gt_ious) >= 0.25)) if gt_ious else 0.0,
        "gt_iou_ge_0p50_rate": float(np.mean(np.asarray(gt_ious) >= 0.50)) if gt_ious else 0.0,
        "pred_iou_ge_0p25_rate": float(np.mean(np.asarray(pred_ious) >= 0.25)) if pred_ious else 0.0,
        "pred_iou_ge_0p50_rate": float(np.mean(np.asarray(pred_ious) >= 0.50)) if pred_ious else 0.0,
        "missing_mask_rows": int(missing_mask_rows),
        "wta_dropped_pixels": int(wta_dropped_pixels),
        "diagnostic_only_uses_gt": True,
    }
    return pred_rows, gt_rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose v103 Phase6 local object materialization failure with GT-only attribution.")
    parser.add_argument("--phase6-root", default=str(DEFAULT_PHASE6_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--variant-id", default="", help="Default: best_variant_id from phase6 summary.json")
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_PHASE2_SCENE0050))
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    phase6_root = _project(args.phase6_root)
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    summary = _read_json(phase6_root / "summary.json")
    variant_id = str(args.variant_id or summary.get("best_variant_id", ""))
    if not variant_id:
        raise RuntimeError("variant_id is empty and phase6 summary has no best_variant_id")
    frame_rows_path = phase6_root / "local_object_frame_mask_rows.csv"
    if not frame_rows_path.exists():
        raise FileNotFoundError(frame_rows_path)
    rows = pd.read_csv(frame_rows_path)
    rows = rows[rows["variant_id"].astype(str) == variant_id].copy()
    if rows.empty:
        raise RuntimeError(f"no selected local object frame rows for variant_id={variant_id}")
    phase2_summaries = {
        "scene0011_00": _read_json(_project(args.scene0011_phase2_root) / "summary.json"),
        "scene0050_00": _read_json(_project(args.scene0050_phase2_root) / "summary.json"),
    }

    all_pred_rows: list[dict[str, Any]] = []
    all_gt_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    for scene, group in rows.groupby("scene_id", sort=True):
        if str(scene) not in phase2_summaries:
            continue
        pred_rows, gt_rows, scene_summary = _accumulate_scene(
            scene=str(scene),
            rows=group,
            phase2_summary=phase2_summaries[str(scene)],
        )
        all_pred_rows.extend(pred_rows)
        all_gt_rows.extend(gt_rows)
        scene_summaries.append(scene_summary)

    _write_csv(out / "pred_object_gt_iou_rows.csv", all_pred_rows)
    _write_csv(out / "gt_object_pred_iou_rows.csv", all_gt_rows)
    _write_csv(out / "scene_summary_rows.csv", scene_summaries)
    payload = {
        "schema_version": "stream4d_v103_phase6_materialization_failure_diagnostic_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "phase6_root": _rel(phase6_root),
        "variant_id": variant_id,
        "scene_count": len(scene_summaries),
        "diagnostic_only_uses_gt": True,
        "uses_gt_for_prediction": False,
        "truthfulness_note": "This diagnostic reconstructs the Phase6 WTA prediction from saved rows and uses GT only to explain materialization failures.",
        "scene_summaries": scene_summaries,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "scene_summary_rows": _rel(out / "scene_summary_rows.csv"),
            "pred_object_gt_iou_rows": _rel(out / "pred_object_gt_iou_rows.csv"),
            "gt_object_pred_iou_rows": _rel(out / "gt_object_pred_iou_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", payload)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
