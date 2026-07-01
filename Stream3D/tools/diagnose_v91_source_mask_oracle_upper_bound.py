from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from tools import run_v89_recalc_point_projected_mv_ap as recalc  # noqa: E402
from tools import run_v91_phase4_adaptive_uncertainty_materialization as adaptive  # noqa: E402
from tools.run_v65_scene_multiview_ap import _load_gt_2d, _read_label_png  # noqa: E402


OUT = ROOT / "outputs/audit/v91_source_mask_oracle_upper_bound"
LOCAL_EXPORT_ROOT = ROOT / "outputs/audit/v89_recalc_point_projected_mv_ap"
WINDOW_SOURCE_STEP = "S3D_L1_local_merged_masks"
SCENES = ["scene0011_00", "scene0050_00"]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(adaptive._jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adaptive._jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gt_mask_pair_stats(gt: np.ndarray, source: np.ndarray) -> tuple[dict[int, int], dict[int, int], dict[tuple[int, int], int]]:
    gt_pos = gt > 0
    src_pos = source > 0
    gt_area: dict[int, int] = {}
    src_area: dict[int, int] = {}
    inter: dict[tuple[int, int], int] = {}
    if np.any(gt_pos):
        ids, counts = np.unique(gt[gt_pos], return_counts=True)
        gt_area = {int(idx): int(count) for idx, count in zip(ids, counts) if int(idx) > 0}
    if np.any(src_pos):
        ids, counts = np.unique(source[src_pos], return_counts=True)
        src_area = {int(idx): int(count) for idx, count in zip(ids, counts) if int(idx) > 0}
    both = gt_pos & src_pos
    if np.any(both):
        gt_vals = gt[both].astype(np.int64, copy=False)
        src_vals = source[both].astype(np.int64, copy=False)
        base = int(np.max(src_vals)) + 1
        encoded = gt_vals * base + src_vals
        ids, counts = np.unique(encoded, return_counts=True)
        for value, count in zip(ids, counts):
            gt_id = int(value // base)
            src_id = int(value % base)
            if gt_id > 0 and src_id > 0:
                inter[(gt_id, src_id)] = int(count)
    return gt_area, src_area, inter


def _best_source_by_gt(gt: np.ndarray, source: np.ndarray) -> dict[int, dict[str, Any]]:
    gt_area, src_area, inter = _gt_mask_pair_stats(gt, source)
    best: dict[int, dict[str, Any]] = {}
    for (gt_id, src_id), intersection in inter.items():
        union = int(gt_area.get(gt_id, 0)) + int(src_area.get(src_id, 0)) - int(intersection)
        iou = float(intersection / union) if union > 0 else 0.0
        coverage = float(intersection / max(1, int(gt_area.get(gt_id, 0))))
        precision = float(intersection / max(1, int(src_area.get(src_id, 0))))
        prev = best.get(gt_id)
        if prev is None or iou > float(prev["source_gt_iou"]):
            best[gt_id] = {
                "gt_id": int(gt_id),
                "source_mask_id": int(src_id),
                "source_gt_iou": iou,
                "source_gt_coverage": coverage,
                "source_precision": precision,
                "intersection_pixels": int(intersection),
                "gt_pixels": int(gt_area.get(gt_id, 0)),
                "source_pixels": int(src_area.get(src_id, 0)),
            }
    return best


def _save_label_png(path: Path, label: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_value = int(np.max(label)) if label.size else 0
    if max_value > np.iinfo(np.uint16).max:
        raise ValueError(f"label id {max_value} exceeds uint16 PNG range for {path}")
    ok = cv2.imwrite(str(path), label.astype(np.uint16, copy=False))
    if not ok:
        raise RuntimeError(f"failed to write label png: {path}")


def _window_maps(scene: str, frame_ids: list[int]) -> tuple[dict[int, list[int]], dict[int, int]]:
    _raw_dir, local_rows = recalc._local_object_rows(LOCAL_EXPORT_ROOT, scene, WINDOW_SOURCE_STEP)
    window_indices = sorted(
        {
            recalc._int(row.get("window_index"), -1)
            for row in local_rows
            if recalc._int(row.get("window_index"), -1) >= 0
        }
    )
    window_scope = recalc._local_window_frame_scope(frame_ids, window_indices)
    frame_to_window = {
        int(frame_id): int(window_index)
        for window_index, support_frames in window_scope.items()
        for frame_id in support_frames
    }
    return window_scope, frame_to_window


def _variant_configs() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "OR0_gt_window_projection_sanity",
            "mode": "gt_full",
            "description": "Exact GT projection materialized as prediction; evaluator/support sanity only.",
        },
        {
            "variant_id": "OR1_source_whole_best_mask_per_gt",
            "mode": "source_whole_best_unique",
            "min_iou": 0.0,
            "description": "For each GT frame instance, choose its best source CropFormer mask; each source mask can serve one GT per frame.",
        },
        {
            "variant_id": "OR2_source_whole_best_mask_iou25",
            "mode": "source_whole_best_unique",
            "min_iou": 0.25,
            "description": "Same as OR1, but drop best-source matches below IoU 0.25 to avoid low-overlap diagnostic FPs.",
        },
        {
            "variant_id": "OR3_source_gt_intersection_upper",
            "mode": "source_gt_intersection",
            "min_iou": 0.0,
            "description": "GT-clipped best source mask; tests source coverage ceiling, not a legal method prediction.",
        },
    ]


def _build_gt_full_scene(
    *,
    scene: str,
    frame_ids: list[int],
    frame_to_window: dict[int, int],
    shape_hw: tuple[int, int],
    variant_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = OUT / "generated_masks" / variant_id / scene / "mask"
    frame_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    frame_summary_rows: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        window_index = frame_to_window.get(int(frame_id))
        gt = _load_gt_2d(scene, int(frame_id), shape_hw)
        generated = np.zeros(shape_hw, dtype=np.int64)
        next_label = 1
        gt_ids = [int(value) for value in np.unique(gt) if int(value) > 0]
        for gt_id in gt_ids:
            generated[gt == gt_id] = int(next_label)
            frame_rows.append(
                {
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "frame_id": int(frame_id),
                    "mask_id": int(next_label),
                    "mv_object_id": f"gt{int(gt_id)}",
                    "object_score": 1.0,
                    "oracle_source": "gt_exact_projection",
                    "uses_gt_for_prediction": True,
                    "uses_future": False,
                }
            )
            assignment_rows.append(
                {
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "window_index": "" if window_index is None else int(window_index),
                    "frame_id": int(frame_id),
                    "gt_id": int(gt_id),
                    "generated_mask_id": int(next_label),
                    "source_mask_id": "",
                    "source_gt_iou": 1.0,
                    "source_gt_coverage": 1.0,
                    "source_precision": 1.0,
                    "prediction_region": "gt_exact_projection",
                    "diagnostic_only": True,
                    "uses_gt_for_prediction": True,
                    "uses_future": False,
                }
            )
            next_label += 1
        _save_label_png(out_dir / f"{int(frame_id)}.png", generated)
        frame_summary_rows.append(
            {
                "variant_id": variant_id,
                "scene_id": scene,
                "frame_id": int(frame_id),
                "window_index": "" if window_index is None else int(window_index),
                "generated_instance_count": int(next_label - 1),
                "gt_instance_count": int(len(gt_ids)),
                "source_mask_count": "",
            }
        )
    return frame_rows, assignment_rows, frame_summary_rows


def _build_source_oracle_scene(
    *,
    scene: str,
    frame_ids: list[int],
    frame_to_window: dict[int, int],
    shape_hw: tuple[int, int],
    variant_id: str,
    mode: str,
    min_iou: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_dir = recalc._mask_dir(scene)
    out_dir = OUT / "generated_masks" / variant_id / scene / "mask"
    frame_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    frame_summary_rows: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        window_index = frame_to_window.get(int(frame_id))
        gt = _load_gt_2d(scene, int(frame_id), shape_hw)
        source_path = source_dir / f"{int(frame_id)}.png"
        if not source_path.exists():
            source = np.zeros(shape_hw, dtype=np.int64)
        else:
            source = _read_label_png(source_path, shape_hw)
        best_by_gt = _best_source_by_gt(gt, source)
        chosen: list[dict[str, Any]] = []
        if mode == "source_whole_best_unique":
            by_source: dict[int, dict[str, Any]] = {}
            for row in best_by_gt.values():
                if float(row["source_gt_iou"]) < float(min_iou):
                    continue
                src_id = int(row["source_mask_id"])
                prev = by_source.get(src_id)
                if prev is None or float(row["source_gt_iou"]) > float(prev["source_gt_iou"]):
                    by_source[src_id] = row
            chosen = sorted(by_source.values(), key=lambda row: (int(row["source_mask_id"]), int(row["gt_id"])))
        elif mode == "source_gt_intersection":
            chosen = [
                row
                for row in sorted(best_by_gt.values(), key=lambda item: int(item["gt_id"]))
                if float(row["source_gt_iou"]) >= float(min_iou)
            ]
        else:
            raise ValueError(f"unsupported oracle mode: {mode}")

        generated = np.zeros(shape_hw, dtype=np.int64)
        next_label = 1
        for row in chosen:
            gt_id = int(row["gt_id"])
            src_id = int(row["source_mask_id"])
            if mode == "source_whole_best_unique":
                region = source == src_id
                prediction_region = "whole_source_mask"
            else:
                region = (source == src_id) & (gt == gt_id)
                prediction_region = "source_mask_intersect_gt"
            if not np.any(region):
                continue
            generated[region] = int(next_label)
            score = float(row["source_gt_iou"])
            frame_rows.append(
                {
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "frame_id": int(frame_id),
                    "mask_id": int(next_label),
                    "mv_object_id": f"gt{gt_id}",
                    "object_score": score,
                    "source_mask_id": int(src_id),
                    "source_gt_iou": score,
                    "source_gt_coverage": float(row["source_gt_coverage"]),
                    "source_precision": float(row["source_precision"]),
                    "oracle_source": prediction_region,
                    "uses_gt_for_prediction": True,
                    "uses_future": False,
                }
            )
            assignment_rows.append(
                {
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "window_index": "" if window_index is None else int(window_index),
                    "frame_id": int(frame_id),
                    "gt_id": gt_id,
                    "generated_mask_id": int(next_label),
                    "source_mask_id": int(src_id),
                    "source_gt_iou": score,
                    "source_gt_coverage": float(row["source_gt_coverage"]),
                    "source_precision": float(row["source_precision"]),
                    "intersection_pixels": int(row["intersection_pixels"]),
                    "gt_pixels": int(row["gt_pixels"]),
                    "source_pixels": int(row["source_pixels"]),
                    "prediction_region": prediction_region,
                    "diagnostic_only": True,
                    "uses_gt_for_prediction": True,
                    "uses_future": False,
                }
            )
            next_label += 1
        _save_label_png(out_dir / f"{int(frame_id)}.png", generated)
        frame_summary_rows.append(
            {
                "variant_id": variant_id,
                "scene_id": scene,
                "frame_id": int(frame_id),
                "window_index": "" if window_index is None else int(window_index),
                "generated_instance_count": int(next_label - 1),
                "gt_instance_count": int(len([value for value in np.unique(gt) if int(value) > 0])),
                "source_mask_count": int(len([value for value in np.unique(source) if int(value) > 0])),
                "source_mask_dir": adaptive._rel(source_dir),
                "source_mask_exists": bool(source_path.exists()),
            }
        )
    return frame_rows, assignment_rows, frame_summary_rows


def _score_free_f1(row: dict[str, Any]) -> float:
    return adaptive._num(row.get("SF50_tp")) * 2.0 / max(
        1e-12,
        adaptive._num(row.get("pred_object_count")) + adaptive._num(row.get("gt_object_count")),
    )


def _evaluate_variant_scene(
    *,
    scene: str,
    frame_ids: list[int],
    variant_id: str,
    rows: list[dict[str, Any]],
    score_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    original_mask_dir = recalc._mask_dir
    try:
        recalc._mask_dir = lambda _scene, _variant=variant_id: OUT / "generated_masks" / _variant / _scene / "mask"
        return recalc._evaluate_frame_mask_variant_local_window(
            scene=scene,
            split="dev",
            variant=variant_id,
            frame_ids=frame_ids,
            rows=rows,
            score_mode=score_mode,
            local_export_root=LOCAL_EXPORT_ROOT,
            window_source_step=WINDOW_SOURCE_STEP,
        )
    finally:
        recalc._mask_dir = original_mask_dir


def run(_args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    scope = recalc._frame_scope()

    config_rows: list[dict[str, Any]] = []
    frame_mask_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    frame_summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for config in _variant_configs():
        variant_id = str(config["variant_id"])
        config_rows.append(
            {
                "variant_id": variant_id,
                "mode": config["mode"],
                "min_iou": config.get("min_iou", ""),
                "description": config["description"],
                "diagnostic_only": True,
                "uses_gt_for_prediction": True,
                "uses_future": False,
                "metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
                "support_policy": "local_window_gt_projection",
            }
        )
        for scene in SCENES:
            frame_ids = [int(v) for v in scope.get(("dev", scene), [])]
            if not frame_ids:
                raise RuntimeError(f"missing dev frame scope for {scene}")
            stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
            shape_hw = tuple(int(v) for v in stream.load_depth(frame_ids[0]).shape)
            _window_scope, frame_to_window = _window_maps(scene, frame_ids)
            if config["mode"] == "gt_full":
                rows, assignments, frame_summaries = _build_gt_full_scene(
                    scene=scene,
                    frame_ids=frame_ids,
                    frame_to_window=frame_to_window,
                    shape_hw=shape_hw,
                    variant_id=variant_id,
                )
            else:
                rows, assignments, frame_summaries = _build_source_oracle_scene(
                    scene=scene,
                    frame_ids=frame_ids,
                    frame_to_window=frame_to_window,
                    shape_hw=shape_hw,
                    variant_id=variant_id,
                    mode=str(config["mode"]),
                    min_iou=float(config.get("min_iou", 0.0)),
                )
            frame_mask_rows.extend(rows)
            assignment_rows.extend(assignments)
            frame_summary_rows.extend(frame_summaries)
            for score_mode in ["input", "constant"]:
                metric, cases, tops, windows = _evaluate_variant_scene(
                    scene=scene,
                    frame_ids=frame_ids,
                    variant_id=variant_id,
                    rows=rows,
                    score_mode=score_mode,
                )
                metric_rows.append(
                    {
                        **metric,
                        "variant_id": variant_id,
                        "diagnostic_only": True,
                        "uses_gt_for_prediction": True,
                        "uses_future": False,
                    }
                )
                case_rows.extend(
                    {
                        **row,
                        "variant_id": variant_id,
                        "diagnostic_only": True,
                        "uses_gt_for_prediction": True,
                        "uses_future": False,
                    }
                    for row in cases
                )
                top_rows.extend(
                    {
                        **row,
                        "variant_id": variant_id,
                        "diagnostic_only": True,
                        "uses_gt_for_prediction": True,
                        "uses_future": False,
                    }
                    for row in tops
                )
                window_rows.extend(
                    {
                        **row,
                        "variant_id": variant_id,
                        "diagnostic_only": True,
                        "uses_gt_for_prediction": True,
                        "uses_future": False,
                    }
                    for row in windows
                )

    aggregate_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row.get("variant", row.get("variant_id", ""))), str(row.get("score_mode", "")))].append(row)
    for (variant_id, score_mode), rows in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "variant_id": variant_id,
                "score_mode": score_mode,
                "scene_count": len(rows),
                "mean_MV_AP_window": adaptive._mean([adaptive._num(row.get("MV_AP")) for row in rows]),
                "mean_MV_AP50_window": adaptive._mean([adaptive._num(row.get("MV_AP50")) for row in rows]),
                "mean_MV_AP25_window": adaptive._mean([adaptive._num(row.get("MV_AP25")) for row in rows]),
                "mean_score_free_Match50_window": adaptive._mean([_score_free_f1(row) for row in rows]),
                "mean_gt_recall_best_iou_ge_050": adaptive._mean([adaptive._num(row.get("gt_recall_best_iou_ge_050")) for row in rows]),
                "mean_gt_best_iou_mean": adaptive._mean([adaptive._num(row.get("gt_best_iou_mean")) for row in rows]),
                "mean_pred_object_count": adaptive._mean([adaptive._num(row.get("pred_object_count")) for row in rows]),
                "mean_gt_object_count": adaptive._mean([adaptive._num(row.get("gt_object_count")) for row in rows]),
                "same_frame_collision_count": int(sum(adaptive._int(row.get("duplicate_frame_mask_conflict_count")) for row in rows)),
                "missing_mask_raster_count": int(sum(adaptive._int(row.get("missing_mask_raster_count")) for row in rows)),
                "diagnostic_only": True,
                "uses_gt_for_prediction": True,
                "uses_future": False,
            }
        )

    best_source_constant = max(
        [row for row in aggregate_rows if row["score_mode"] == "constant" and row["variant_id"] != "OR0_gt_window_projection_sanity"],
        key=lambda row: adaptive._num(row.get("mean_MV_AP50_window")),
        default={},
    )
    sanity = next(
        (row for row in aggregate_rows if row["variant_id"] == "OR0_gt_window_projection_sanity" and row["score_mode"] == "constant"),
        {},
    )
    summary = {
        "phase": "v91_source_mask_oracle_upper_bound",
        "schema": "stream4d_v91_source_mask_oracle_upper_bound_v1",
        "purpose": "diagnostic-only oracle to separate evaluator/support sanity, source mask universe coverage, and legal method readout failure",
        "variant_count": len(config_rows),
        "score_modes": ["input", "constant"],
        "support_policy": "local_window_gt_projection",
        "metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "diagnostic_only": True,
        "uses_gt_for_prediction": True,
        "uses_future": False,
        "gt_usage": "oracle source-mask selection and OR0/OR3 diagnostic GT materialization; never a legal candidate",
        "sanity_gt_projection_constant": sanity,
        "best_source_oracle_constant": best_source_constant,
        "row_counts": {
            "config_rows": len(config_rows),
            "frame_mask_rows": len(frame_mask_rows),
            "oracle_assignment_rows": len(assignment_rows),
            "frame_summary_rows": len(frame_summary_rows),
            "metric_rows": len(metric_rows),
            "aggregate_rows": len(aggregate_rows),
            "casebook_rows": len(case_rows),
            "top_iou_rows": len(top_rows),
            "window_metric_rows": len(window_rows),
        },
        "runtime_sec": time.time() - started,
    }

    _write_csv(OUT / "variant_config_rows.csv", config_rows)
    _write_csv(OUT / "mv_object_frame_mask_rows.csv", frame_mask_rows)
    _write_csv(OUT / "oracle_assignment_rows.csv", assignment_rows)
    _write_csv(OUT / "frame_summary_rows.csv", frame_summary_rows)
    _write_csv(OUT / "mv_metric_rows.csv", metric_rows)
    _write_csv(OUT / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(OUT / "casebook_rows.csv", case_rows)
    _write_csv(OUT / "mv_top_iou_rows.csv", top_rows)
    _write_csv(OUT / "window_metric_rows.csv", window_rows)
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "variant_config_rows.csv",
        OUT / "mv_object_frame_mask_rows.csv",
        OUT / "oracle_assignment_rows.csv",
        OUT / "frame_summary_rows.csv",
        OUT / "mv_metric_rows.csv",
        OUT / "mv_metric_aggregate_rows.csv",
        OUT / "casebook_rows.csv",
        OUT / "mv_top_iou_rows.csv",
        OUT / "window_metric_rows.csv",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {adaptive._rel(path): adaptive._sha256(path) for path in outputs if path.exists()})
    print(json.dumps(adaptive._jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic-only v91 source mask oracle upper-bound audit.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
