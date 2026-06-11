from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _prediction_dir(root: Path, config: str, suffix: str) -> Path:
    return root / "data" / "prediction" / (config if config.endswith(suffix) else f"{config}{suffix}")


def _tmp_path(root: Path, config: str, scene_id: str) -> Path:
    candidates = [
        root / "data" / "TMP" / config / f"{scene_id}_pre_points.npy",
        root / "TMP" / config / f"{scene_id}_pre_points.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_full_masks(root: Path, config: str, pred_suffix: str, scene_id: str, gt_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    pred_path = _prediction_dir(root, config, pred_suffix) / f"{scene_id}.npz"
    tmp_path = _tmp_path(root, config, scene_id)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not tmp_path.exists():
        raise FileNotFoundError(tmp_path)
    with np.load(pred_path) as pred:
        pred_masks = np.asarray(pred["pred_masks"], dtype=bool)
        pred_score = np.asarray(pred["pred_score"], dtype=np.float32)
        pred_classes = np.asarray(pred["pred_classes"], dtype=np.int32)
    pre_points = np.load(tmp_path).astype(np.int64)
    scene_vertices = int(gt_ids.shape[0])
    if pred_masks.shape[0] == scene_vertices:
        return pred_masks, pred_score, pred_classes, "full_scene"
    if pred_masks.shape[0] == pre_points.shape[0]:
        full = np.zeros((scene_vertices, pred_masks.shape[1]), dtype=bool)
        full[pre_points, :] = pred_masks
        return full, pred_score, pred_classes, "tmp_cropped_expanded"
    raise ValueError(
        f"{scene_id}: pred mask first dimension {pred_masks.shape[0]} does not match "
        f"scene vertices {scene_vertices} or pre_points {pre_points.shape[0]}"
    )


def _prediction_overlap_stats(masks: np.ndarray) -> dict[str, Any]:
    num_pred = int(masks.shape[1])
    areas = masks.sum(axis=0).astype(np.int64)
    union = np.any(masks, axis=1) if num_pred else np.zeros((masks.shape[0],), dtype=bool)
    owner_counts = masks.sum(axis=1).astype(np.int16) if num_pred else np.zeros((masks.shape[0],), dtype=np.int16)
    conflict_points = int(np.count_nonzero(owner_counts > 1))
    union_count = int(np.count_nonzero(union))

    overlap_iou: list[float] = []
    overlap_ioc: list[float] = []
    duplicate_pairs = 0
    if num_pred >= 2:
        for left in range(num_pred):
            left_mask = masks[:, left]
            if areas[left] <= 0:
                continue
            intersections = masks[:, left + 1 :][left_mask].sum(axis=0).astype(np.int64)
            right_areas = areas[left + 1 :]
            nonzero = np.flatnonzero(intersections > 0)
            for idx in nonzero.tolist():
                inter = int(intersections[idx])
                right_area = int(right_areas[idx])
                union_pair = int(areas[left] + right_area - inter)
                iou = float(inter / max(union_pair, 1))
                ioc = float(inter / max(min(int(areas[left]), right_area), 1))
                overlap_iou.append(iou)
                overlap_ioc.append(ioc)
                duplicate_pairs += int(iou >= 0.5)

    area_values = [int(v) for v in areas.tolist()]
    return {
        "num_pred_instances": num_pred,
        "num_prediction_union": union_count,
        "prediction_union_ratio": float(union_count / max(masks.shape[0], 1)),
        "point_ownership_conflict_points": conflict_points,
        "support_conflict_rate": float(conflict_points / max(union_count, 1)),
        "point_assignments": int(np.count_nonzero(masks)),
        "pred_area_mean": float(mean(area_values)) if area_values else 0.0,
        "pred_area_median": float(median(area_values)) if area_values else 0.0,
        "pred_area_min": int(min(area_values)) if area_values else 0,
        "pred_area_max": int(max(area_values)) if area_values else 0,
        "tiny_mask_ratio_area_lt_100": float(sum(1 for v in area_values if v < 100) / max(num_pred, 1)),
        "large_mask_ratio_area_gt_5000": float(sum(1 for v in area_values if v > 5000) / max(num_pred, 1)),
        "overlap_pair_count": int(len(overlap_iou)),
        "overlap_iou_mean_nonzero": float(mean(overlap_iou)) if overlap_iou else 0.0,
        "overlap_ioc_mean_nonzero": float(mean(overlap_ioc)) if overlap_ioc else 0.0,
        "overlap_iou_p90_nonzero": float(np.percentile(overlap_iou, 90)) if overlap_iou else 0.0,
        "duplicate_prediction_pairs_iou_ge_0p50": int(duplicate_pairs),
        "duplicate_prediction_rate": float(duplicate_pairs / max(num_pred, 1)),
    }


def _gt_diagnostics(masks: np.ndarray, gt_ids: np.ndarray) -> dict[str, Any]:
    pred_areas = masks.sum(axis=0).astype(np.int64)
    union = np.any(masks, axis=1) if masks.shape[1] else np.zeros((masks.shape[0],), dtype=bool)
    gt_instance_ids = [int(v) for v in np.unique(gt_ids[gt_ids >= 1000]).tolist()]
    best_ious: list[float] = []
    duplicates: list[int] = []
    gt_coverage: list[float] = []
    for instance_id in gt_instance_ids:
        gt_mask = gt_ids == int(instance_id)
        gt_area = int(np.count_nonzero(gt_mask))
        if gt_area <= 0:
            continue
        intersections = masks[gt_mask].sum(axis=0).astype(np.int64) if masks.shape[1] else np.zeros((0,), dtype=np.int64)
        unions = gt_area + pred_areas - intersections
        ious = intersections / np.maximum(unions, 1)
        best = float(np.max(ious)) if ious.size else 0.0
        best_ious.append(best)
        duplicates.append(int(np.count_nonzero(ious >= 0.25)))
        gt_coverage.append(float(np.count_nonzero(union & gt_mask) / max(gt_area, 1)))
    return {
        "gt_diagnostic_uses_gt": True,
        "gt_instance_count": int(len(best_ious)),
        "per_gt_best_iou_mean": float(mean(best_ious)) if best_ious else 0.0,
        "per_gt_best_iou_ge_0p25": float(sum(v >= 0.25 for v in best_ious) / max(len(best_ious), 1)),
        "per_gt_best_iou_ge_0p50": float(sum(v >= 0.50 for v in best_ious) / max(len(best_ious), 1)),
        "per_gt_best_iou_ge_0p75": float(sum(v >= 0.75 for v in best_ious) / max(len(best_ious), 1)),
        "duplicate_predictions_per_gt_mean_iou_ge_0p25": float(mean(duplicates)) if duplicates else 0.0,
        "missed_gt_count_iou_lt_0p25": int(sum(v < 0.25 for v in best_ious)),
        "gt_coverage_by_prediction_union_mean": float(mean(gt_coverage)) if gt_coverage else 0.0,
    }


def _process_scene(args: argparse.Namespace, root: Path, config: str, scene_id: str) -> dict[str, Any]:
    gt_path = root / args.gt_root / f"{scene_id}.txt"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    gt_ids = np.loadtxt(gt_path).astype(np.int64)
    masks, scores, classes, mask_shape_mode = _load_full_masks(root, config, args.pred_suffix, scene_id, gt_ids)
    row = {
        "config": config,
        "scene_id": scene_id,
        "ok": True,
        "mask_shape_mode": mask_shape_mode,
        "num_scores": int(scores.shape[0]),
        "num_classes": int(classes.shape[0]),
        "num_scene_vertices": int(gt_ids.shape[0]),
        **_prediction_overlap_stats(masks),
    }
    if args.include_gt_diagnostics:
        row.update(_gt_diagnostics(masks, gt_ids))
    return row


def _aggregate(config: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    numeric_keys = sorted(
        key
        for row in ok_rows
        for key, value in row.items()
        if isinstance(value, (int, float)) and key not in {"gt_diagnostic_uses_gt"}
    )
    out: dict[str, Any] = {"config": config, "scenes": len(rows), "ok_scenes": len(ok_rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in ok_rows if isinstance(row.get(key), (int, float))]
        if values:
            out[f"{key}_mean"] = float(mean(values))
    return out


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Prediction Quality Diagnostic",
        "",
        f"- include_gt_diagnostics: `{payload['args']['include_gt_diagnostics']}`",
        f"- diagnostic_only: `True`",
        "",
        "## Config Summary",
        "",
        "| Config | #pred | union % | conflict % | area mean/median | dup rate | per-GT IoU mean | GT IoU >=25/50/75 | missed GT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["config"],
                    f"{row.get('num_pred_instances_mean', 0.0):.2f}",
                    f"{row.get('prediction_union_ratio_mean', 0.0) * 100.0:.4f}",
                    f"{row.get('support_conflict_rate_mean', 0.0) * 100.0:.4f}",
                    f"{row.get('pred_area_mean_mean', 0.0):.2f}/{row.get('pred_area_median_mean', 0.0):.2f}",
                    f"{row.get('duplicate_prediction_rate_mean', 0.0):.4f}",
                    f"{row.get('per_gt_best_iou_mean_mean', 0.0):.4f}",
                    f"{row.get('per_gt_best_iou_ge_0p25_mean', 0.0):.4f}/{row.get('per_gt_best_iou_ge_0p50_mean', 0.0):.4f}/{row.get('per_gt_best_iou_ge_0p75_mean', 0.0):.4f}",
                    f"{row.get('missed_gt_count_iou_lt_0p25_mean', 0.0):.2f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Evidence Boundary", ""])
    lines.append(
        "GT diagnostics are for error analysis only. They are diagnostic-only and must not be used as model-selection method results."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--gt-root", default="data/scannet/gt")
    parser.add_argument("--include-gt-diagnostics", action="store_true")
    parser.add_argument("--output-prefix", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    scene_ids = _read_seq_list(root / args.seq_list)
    configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for config in configs:
        config_rows = [_process_scene(args, root, config, scene_id) for scene_id in scene_ids]
        rows.extend(config_rows)
        aggregates.append(_aggregate(config, config_rows))

    payload = {"args": vars(args), "aggregates": aggregates, "rows": rows}
    prefix = root / args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))
    _write_markdown(md_path, payload)
    print(f"[prediction-quality] wrote {json_path}")
    print(f"[prediction-quality] wrote {csv_path}")
    print(f"[prediction-quality] wrote {md_path}")


if __name__ == "__main__":
    main()
