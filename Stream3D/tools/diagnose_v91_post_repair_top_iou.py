from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v89_recalc_point_projected_mv_ap as recalc  # noqa: E402
from tools import run_v91_phase4_adaptive_uncertainty_materialization as adaptive  # noqa: E402


OUT = ROOT / "outputs/audit/v91_post_repair_top_iou_diagnostic"
LOCAL_EXPORT_ROOT = ROOT / "outputs/audit/v89_recalc_point_projected_mv_ap"
WINDOW_SOURCE_STEP = "S3D_L1_local_merged_masks"


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "label": "AD4_best_before_continuation",
            "root": ROOT / "outputs/audit/v91_phase4_adaptive_uncertainty_materialization",
            "variant_id": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
            "family": "adaptive_uncertainty",
        },
        {
            "label": "SR2_scene_risk_best",
            "root": ROOT / "outputs/audit/v91_phase4_scene_risk_materialization",
            "variant_id": "V91_SR2_highrisk_broad_top4_r16_drop5",
            "family": "scene_risk_materialization",
        },
        {
            "label": "SW1_support_wta_best",
            "root": ROOT / "outputs/audit/v91_phase4_support_wta_repair",
            "variant_id": "V91_SW1_highrisk_support_count_wta_r16",
            "family": "support_wta",
        },
        {
            "label": "BC3_broad_core_best",
            "root": ROOT / "outputs/audit/v91_phase4_broad_core_precision_repair",
            "variant_id": "V91_BC3_highrisk_broad_core_r20_spr3",
            "family": "broad_core_precision",
        },
        {
            "label": "M1_multimask_reference",
            "root": ROOT / "outputs/audit/v91_phase4_witness_cover_multimask_materialization",
            "variant_id": "V91_M1_W8a_top2_r16_drop5_sceneorig",
            "family": "multimask_reference",
        },
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(adaptive._jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adaptive._jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _evaluate_variant(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(spec["root"])
    variant = str(spec["variant_id"])
    rows = [row for row in _read_csv(root / "mv_object_frame_mask_rows.csv") if row.get("variant_id", row.get("variant", "")) == variant]
    if not rows:
        raise RuntimeError(f"missing frame-mask rows for {variant} under {root}")
    scope = recalc._frame_scope()
    original_mask_dir = recalc._mask_dir
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    try:
        recalc._mask_dir = lambda scene, _root=root, _variant=variant: _root / "generated_masks" / _variant / scene / "mask"
        for scene in ["scene0011_00", "scene0050_00"]:
            scene_rows = [row for row in rows if row.get("scene_id") == scene]
            if not scene_rows:
                continue
            metric, cases, tops, windows = recalc._evaluate_frame_mask_variant_local_window(
                scene=scene,
                split="dev",
                variant=variant,
                frame_ids=scope.get(("dev", scene)),
                rows=scene_rows,
                score_mode="input",
                local_export_root=LOCAL_EXPORT_ROOT,
                window_source_step=WINDOW_SOURCE_STEP,
            )
            metric_rows.append({**metric, "diagnostic_label": spec["label"], "diagnostic_family": spec["family"], "variant_id": variant})
            case_rows.extend({**row, "diagnostic_label": spec["label"], "diagnostic_family": spec["family"], "variant_id": variant} for row in cases)
            top_rows.extend({**row, "diagnostic_label": spec["label"], "diagnostic_family": spec["family"], "variant_id": variant} for row in tops)
            window_rows.extend({**row, "diagnostic_label": spec["label"], "diagnostic_family": spec["family"], "variant_id": variant} for row in windows)
    finally:
        recalc._mask_dir = original_mask_dir
    return metric_rows, case_rows, top_rows, window_rows


def run(_args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    config_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for spec in _variant_specs():
        config_rows.append(
            {
                "diagnostic_label": spec["label"],
                "diagnostic_family": spec["family"],
                "variant_id": spec["variant_id"],
                "root": adaptive._rel(Path(spec["root"])),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "gt_usage": "diagnostic_evaluator_only",
            }
        )
        metrics, cases, tops, windows = _evaluate_variant(spec)
        metric_rows.extend(metrics)
        case_rows.extend(cases)
        top_rows.extend(tops)
        window_rows.extend(windows)

    summary_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        summary_rows.append(
            {
                "diagnostic_label": row.get("diagnostic_label", ""),
                "diagnostic_family": row.get("diagnostic_family", ""),
                "variant_id": row.get("variant_id", ""),
                "scene_id": row.get("scene_id", ""),
                "MV_AP_window": adaptive._num(row.get("MV_AP")),
                "MV_AP50_window": adaptive._num(row.get("MV_AP50")),
                "MV_AP25_window": adaptive._num(row.get("MV_AP25")),
                "score_free_Match50_window": adaptive._num(row.get("SF50_tp"))
                * 2.0
                / max(1e-12, adaptive._num(row.get("pred_object_count")) + adaptive._num(row.get("gt_object_count"))),
                "gt_best_iou_mean": adaptive._num(row.get("gt_best_iou_mean")),
                "gt_best_iou_median": adaptive._num(row.get("gt_best_iou_median")),
                "gt_recall_best_iou_ge_050": adaptive._num(row.get("gt_recall_best_iou_ge_050")),
                "pred_best_iou_mean": adaptive._num(row.get("pred_best_iou_mean")),
                "pred_object_count": adaptive._num(row.get("pred_object_count")),
                "gt_object_count": adaptive._num(row.get("gt_object_count")),
                "missing_mask_raster_count": adaptive._int(row.get("missing_mask_raster_count")),
                "same_frame_collision_count": adaptive._int(row.get("duplicate_frame_mask_conflict_count")),
            }
        )

    scene0011_rows = [row for row in summary_rows if row.get("scene_id") == "scene0011_00"]
    best_scene0011_ap50 = max(scene0011_rows, key=lambda row: adaptive._num(row.get("MV_AP50_window")), default={})
    best_scene0011_gt_recall = max(scene0011_rows, key=lambda row: adaptive._num(row.get("gt_recall_best_iou_ge_050")), default={})
    summary = {
        "phase": "v91_post_repair_top_iou_diagnostic",
        "schema": "stream4d_v91_post_repair_top_iou_diagnostic_v1",
        "variant_count": len(config_rows),
        "best_scene0011_ap50": best_scene0011_ap50,
        "best_scene0011_gt_recall_best_iou_ge_050": best_scene0011_gt_recall,
        "scene0011_ap50_values": {
            row["variant_id"]: row["MV_AP50_window"] for row in scene0011_rows
        },
        "scene0011_gt_recall50_values": {
            row["variant_id"]: row["gt_recall_best_iou_ge_050"] for row in scene0011_rows
        },
        "row_counts": {
            "config_rows": len(config_rows),
            "metric_rows": len(metric_rows),
            "summary_rows": len(summary_rows),
            "casebook_rows": len(case_rows),
            "top_iou_rows": len(top_rows),
            "window_metric_rows": len(window_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "gt_usage": "diagnostic_evaluator_only",
        "runtime_sec": time.time() - started,
    }

    _write_csv(OUT / "variant_config_rows.csv", config_rows)
    _write_csv(OUT / "mv_metric_rows.csv", metric_rows)
    _write_csv(OUT / "scene_summary_rows.csv", summary_rows)
    _write_csv(OUT / "casebook_rows.csv", case_rows)
    _write_csv(OUT / "mv_top_iou_rows.csv", top_rows)
    _write_csv(OUT / "window_metric_rows.csv", window_rows)
    _write_json(OUT / "summary.json", summary)
    outputs = [
        OUT / "variant_config_rows.csv",
        OUT / "mv_metric_rows.csv",
        OUT / "scene_summary_rows.csv",
        OUT / "casebook_rows.csv",
        OUT / "mv_top_iou_rows.csv",
        OUT / "window_metric_rows.csv",
        OUT / "summary.json",
    ]
    _write_json(OUT / "SHA256SUMS.json", {adaptive._rel(path): adaptive._sha256(path) for path in outputs if path.exists()})
    print(json.dumps(adaptive._jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose v91 post-repair local-window top IoU using existing generated masks.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
