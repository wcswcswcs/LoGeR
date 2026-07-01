from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.constants import SCANNET_IDS
from tools.v64r2_visible_support_utils import json_safe, read_seq_list


MIN_REGION_SIZE = 100


def _class_agnostic_gt(gt_ids: np.ndarray) -> np.ndarray:
    return gt_ids % 1000 + int(SCANNET_IDS[0]) * 1000


def _gt_masks(gt_ids: np.ndarray) -> np.ndarray:
    masks: list[np.ndarray] = []
    for instance_id in np.unique(gt_ids):
        instance_id_int = int(instance_id)
        if instance_id_int < 1000:
            continue
        mask = gt_ids == instance_id_int
        if int(mask.sum()) >= MIN_REGION_SIZE:
            masks.append(mask)
    if not masks:
        return np.zeros((0, gt_ids.shape[0]), dtype=bool)
    return np.stack(masks, axis=0).astype(bool, copy=False)


def _iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if gt.shape[0] == 0 or pred.shape[1] == 0:
        return np.zeros((gt.shape[0], pred.shape[1]), dtype=np.float64)
    gt_i = gt.astype(np.int64, copy=False)
    pred_i = pred.astype(np.int64, copy=False)
    inter = gt_i @ pred_i
    gt_area = gt_i.sum(axis=1, keepdims=True)
    pred_area = pred_i.sum(axis=0, keepdims=True)
    union = gt_area + pred_area - inter
    return inter / np.maximum(union, 1)


def _summarize_values(prefix: str, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p90": None,
            f"{prefix}_ge_025": 0,
            f"{prefix}_ge_050": 0,
        }
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_ge_025": int(np.count_nonzero(values >= 0.25)),
        f"{prefix}_ge_050": int(np.count_nonzero(values >= 0.50)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose class-agnostic AP candidate-vs-GT IoU distributions.")
    parser.add_argument("--configs", required=True, help="Comma-separated output configs without _class_agnostic suffix.")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--audit-root", default="outputs/audit/v64r2_ap_iou_distribution")
    args = parser.parse_args()

    root = Path(".").resolve()
    configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    scenes = read_seq_list(Path(args.seq_list))
    rows: list[dict[str, Any]] = []
    for config in configs:
        for scene in scenes:
            pred_path = root / "data" / "prediction" / f"{config}_class_agnostic" / f"{scene}.npz"
            pre_path = root / "data" / "TMP" / config / f"{scene}_pre_points.npy"
            gt_path = root / "data" / "scannet" / "gt" / f"{scene}.txt"
            row: dict[str, Any] = {"config": config, "scene": scene}
            try:
                with np.load(pred_path) as pred:
                    masks = np.asarray(pred["pred_masks"], dtype=bool)
                    scores = np.asarray(pred["pred_score"], dtype=np.float32)
                pre_points = np.load(pre_path).astype(np.int64)
                cropped_pred = masks[pre_points, :]
                pred_area = cropped_pred.sum(axis=0).astype(np.int64)
                keep = pred_area >= MIN_REGION_SIZE
                cropped_pred = cropped_pred[:, keep]
                kept_scores = scores[keep] if scores.shape[0] == keep.shape[0] else np.ones((int(np.count_nonzero(keep)),), dtype=np.float32)
                gt = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))[pre_points]
                gt_mask = _gt_masks(gt)
                iou = _iou_matrix(gt_mask, cropped_pred)
                pred_best = iou.max(axis=0) if iou.size else np.zeros((cropped_pred.shape[1],), dtype=np.float64)
                gt_best = iou.max(axis=1) if iou.size else np.zeros((gt_mask.shape[0],), dtype=np.float64)
                row.update(
                    {
                        "status": "ok",
                        "pre_points_count": int(pre_points.shape[0]),
                        "raw_pred_count": int(masks.shape[1]),
                        "kept_pred_count": int(cropped_pred.shape[1]),
                        "dropped_pred_lt100": int(np.count_nonzero(~keep)),
                        "gt_instance_count": int(gt_mask.shape[0]),
                        "pred_area_mean": float(np.mean(pred_area[keep])) if np.any(keep) else None,
                        "pred_area_median": float(np.median(pred_area[keep])) if np.any(keep) else None,
                        "score_mean": float(np.mean(kept_scores)) if kept_scores.size else None,
                    }
                )
                row.update(_summarize_values("pred_best_iou", pred_best))
                row.update(_summarize_values("gt_best_iou", gt_best))
            except Exception as exc:
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    for config in configs:
        subset = [row for row in rows if row.get("config") == config and row.get("status") == "ok"]
        summary: dict[str, Any] = {"config": config, "status": "ok" if subset else "missing_or_failed", "num_scenes": int(len(subset))}
        for key in sorted({key for row in subset for key, value in row.items() if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool)}):
            vals = [float(row[key]) for row in subset if row.get(key) is not None and np.isfinite(float(row[key]))]
            if vals:
                summary[key] = float(np.mean(vals))
        summary_rows.append(summary)

    audit_root = Path(args.audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "summary": summary_rows, "min_region_size": MIN_REGION_SIZE}
    (audit_root / "ap_iou_distribution_summary.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(audit_root / "ap_iou_distribution_scene_rows.csv", rows)
    _write_csv(audit_root / "ap_iou_distribution_summary.csv", summary_rows)
    print(json.dumps(json_safe({"summary": summary_rows}), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
