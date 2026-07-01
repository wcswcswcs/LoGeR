from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v89_mv_ap_stream3d_local_baseline as v89


OUT = v89.V89_ROOT / "v89_mv_ap_formula_investigation"


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
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_window_id(mv_object_id: str) -> str:
    match = re.search(r"\|w(\d+)\|", mv_object_id)
    return match.group(1) if match else ""


def _run_pred_area_probe(rows: list[dict[str, str]]) -> dict[str, Any]:
    scope = v89._frame_scope_from_v88()
    metric_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    iou_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for scene in ["scene0011_00", "scene0050_00"]:
        frame_ids = scope[("dev", scene)]
        group = [
            row
            for row in rows
            if row.get("split") == "dev"
            and row.get("scene_id") == scene
            and row.get("variant") == "S3D_L1_local_merged_masks"
        ]
        metric, _cases, ious, prs, _gts = v89._evaluate_s3d_group(
            split="dev",
            scene=scene,
            variant="S3D_L1_local_merged_masks",
            rows=group,
            frame_ids=frame_ids,
            score_mode="pred_area",
            min_pred_pixels=1,
            min_gt_pixels=1,
            top_k=50,
        )
        metric_rows.append(metric)
        pr_rows.extend(prs)
        iou_rows.extend(ious)
    _write_csv(OUT / "s3d_l1_pred_area_metric_rows.csv", metric_rows)
    _write_csv(OUT / "s3d_l1_pred_area_pr_curve_rows.csv", pr_rows)
    _write_csv(OUT / "s3d_l1_pred_area_iou_top_rows.csv", iou_rows)
    summary = {
        "phase": "v89_mv_ap_formula_investigation",
        "question": "why S3D_L1 local MV_AP is low and whether v65 MV_AP formula is wrong",
        "score_mode": "pred_area",
        "formal_metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "scene_metrics": metric_rows,
        "aggregate_MV_AP_mean_over_scenes": sum(float(row["MV_AP"]) for row in metric_rows) / len(metric_rows),
        "aggregate_MV_AP50_mean_over_scenes": sum(float(row["MV_AP50"]) for row in metric_rows) / len(metric_rows),
        "runtime_sec": time.time() - t0,
    }
    _write_json(OUT / "summary.json", summary)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phase2_rows = _read_csv(v89.PHASE2 / "mv_object_frame_mask_rows.csv")
    metrics = _read_csv(v89.PHASE3 / "mv_metric_rows.csv")
    pr_rows = _read_csv(v89.PHASE3 / "mv_pr_curve_rows.csv")
    object_rows = _read_csv(v89.PHASE2 / "mv_object_rows.csv")
    iou_rows = _read_csv(v89.PHASE3 / "mv_iou_matrix_rows.csv")
    v66_rows = _read_csv(v89.V89_ROOT / "v66_scene_mv_ap_probe5_full" / "mv_ap_rows.csv")

    pred_area_summary = _run_pred_area_probe(phase2_rows)
    summary: dict[str, Any] = {
        "phase": "v89_mv_ap_formula_investigation",
        "answer": (
            "The v65 AP arithmetic replays correctly. The suspicious low S3D_L1 number is caused by "
            "evaluating window-local Stream3D masks as scene-level object tubes, plus AP averaging over "
            "high IoU thresholds and many false positives; it is not evidence that SparseSceneIoU/_summarize_iou "
            "arithmetic is broken."
        ),
        "formal_metric_source": "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou",
        "s3d_l1_input_scene_replay": {},
        "s3d_l1_window_scope": {},
        "s3d_l1_top_iou_evidence": {},
        "s3d_l1_pred_area_probe": {
            "aggregate_MV_AP_mean_over_scenes": pred_area_summary["aggregate_MV_AP_mean_over_scenes"],
            "aggregate_MV_AP50_mean_over_scenes": pred_area_summary["aggregate_MV_AP50_mean_over_scenes"],
            "artifact": str(OUT / "summary.json"),
        },
        "v66_final_stream3d_comparison": {},
    }

    for scene in ["scene0011_00", "scene0050_00"]:
        metric = next(
            row
            for row in metrics
            if row["scene_id"] == scene
            and row["variant"] == "S3D_L1_local_merged_masks"
            and row["score_mode"] == "input"
        )
        scene_pr = [
            row
            for row in pr_rows
            if row["scene_id"] == scene
            and row["variant"] == "S3D_L1_local_merged_masks"
            and row["score_mode"] == "input"
        ]
        ap_values = [_num(row["ap"]) for row in scene_pr]
        replay = sum(ap_values) / len(ap_values)
        p50 = next(row for row in scene_pr if row["threshold"] == "0.50")
        scene_objects = [
            row
            for row in object_rows
            if row["scene_id"] == scene and row["variant"] == "S3D_L1_local_merged_masks"
        ]
        window_counts: dict[str, int] = {}
        for row in scene_objects:
            window_id = _parse_window_id(row["mv_object_id"])
            window_counts[window_id] = window_counts.get(window_id, 0) + 1
        top_iou = [
            row
            for row in iou_rows
            if row["scene_id"] == scene
            and row["variant"] == "S3D_L1_local_merged_masks"
            and row["score_mode"] == "input"
        ][:10]
        summary["s3d_l1_input_scene_replay"][scene] = {
            "metric_row_MV_AP": _num(metric["MV_AP"]),
            "replayed_mean_ap_from_pr_threshold_rows": replay,
            "absolute_delta": abs(_num(metric["MV_AP"]) - replay),
            "threshold_count": len(scene_pr),
            "threshold_aps": {row["threshold"]: _num(row["ap"]) for row in scene_pr},
            "AP50_tp": int(_num(p50["tp"])),
            "AP50_fp_at_best_threshold": int(_num(p50["fp"])),
            "AP50_gt_count": int(_num(p50["gt_count"])),
            "AP50_precision": _num(p50["precision"]),
            "AP50_recall": _num(p50["recall"]),
            "AP50_score_threshold_count": int(_num(p50["score_threshold_count"])),
            "pred_object_count": int(_num(metric["pred_object_count"])),
            "gt_object_count": int(_num(metric["gt_object_count"])),
            "gt_recall_best_iou_ge_050": _num(metric["gt_recall_best_iou_ge_050"]),
            "pred_best_iou_median": _num(metric["pred_best_iou_median"]),
        }
        summary["s3d_l1_window_scope"][scene] = {
            "object_count": len(scene_objects),
            "window_count_detected_from_object_ids": len([key for key in window_counts if key]),
            "window_ids": sorted([key for key in window_counts if key]),
            "objects_per_window_min": min(window_counts.values()) if window_counts else 0,
            "objects_per_window_max": max(window_counts.values()) if window_counts else 0,
            "objects_per_window_mean": sum(window_counts.values()) / len(window_counts) if window_counts else 0,
            "object_id_example": scene_objects[0]["mv_object_id"] if scene_objects else "",
            "scope_issue": (
                "mv_object_id contains local window id wXXXX, so these are local-window objects rather "
                "than globally fused Stream3D final objects."
            ),
        }
        summary["s3d_l1_top_iou_evidence"][scene] = [
            {
                "mv_object_id": row["mv_object_id"],
                "gt_id": row["gt_id"],
                "mv_iou": _num(row["mv_iou"]),
                "pred_area": int(_num(row["pred_area"])),
                "gt_area": int(_num(row["gt_area"])),
            }
            for row in top_iou
        ]

    for scene in ["scene0011_00", "scene0050_00"]:
        vals = [
            row
            for row in v66_rows
            if row["scene_id"] == scene and row["method"] in {"Stream3D_constant", "Stream3D_pred_area"}
        ]
        summary["v66_final_stream3d_comparison"][scene] = {
            row["method"]: {
                "AP": _num(row["AP"]),
                "AP50": _num(row["AP50"]),
                "AP25": _num(row["AP25"]),
                "score_free_match50_recall": _num(row["score_free_match50_recall"]),
                "pred_count": int(_num(row["pred_count"])),
                "gt_count": int(_num(row["gt_count"])),
                "frame_count": int(_num(row["frame_count"])),
                "scope": row["matching_scope"],
                "source": row["source"],
            }
            for row in vals
            if row["score_mode"] in {"constant", "pred_area"}
        }

    scene_mvaps = [
        summary["s3d_l1_input_scene_replay"][scene]["metric_row_MV_AP"]
        for scene in ["scene0011_00", "scene0050_00"]
    ]
    scene_ap50 = [
        _num(
            next(
                row
                for row in metrics
                if row["scene_id"] == scene
                and row["variant"] == "S3D_L1_local_merged_masks"
                and row["score_mode"] == "input"
            )["MV_AP50"]
        )
        for scene in ["scene0011_00", "scene0050_00"]
    ]
    summary["s3d_l1_input_aggregate"] = {
        "mean_scene_MV_AP": sum(scene_mvaps) / len(scene_mvaps),
        "mean_scene_MV_AP50": sum(scene_ap50) / len(scene_ap50),
        "aggregation_rule_observed": "simple mean over dev scenes in v89 phase4 summary",
    }
    _write_json(OUT / "s3d_l1_mv_ap_root_cause_summary.json", summary)
    print(json.dumps(summary["s3d_l1_input_aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
