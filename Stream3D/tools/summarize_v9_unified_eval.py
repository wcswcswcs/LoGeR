from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, variance
from typing import Any

import numpy as np

from tools.prediction_manifest import load_prediction_manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_metric_file(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"ap": None, "ap50": None, "ap25": None}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {"ap": None, "ap50": None, "ap25": None}
    parts = lines[-1].replace("\t", ",").split(",")
    if len(parts) < 3:
        return {"ap": None, "ap50": None, "ap25": None}
    try:
        return {"ap": float(parts[0]), "ap50": float(parts[1]), "ap25": float(parts[2])}
    except ValueError:
        return {"ap": None, "ap50": None, "ap25": None}


def _prediction_dir(root: Path, config: str, suffix: str) -> Path:
    if config.endswith(suffix):
        return root / "data" / "prediction" / config
    return root / "data" / "prediction" / f"{config}{suffix}"


def _tmp_path(root: Path, config: str, scene_id: str) -> Path:
    return root / "data" / "TMP" / config / f"{scene_id}_pre_points.npy"


def _metric_path(root: Path, dataset: str, config: str, suffix: str) -> Path:
    name = config if config.endswith(suffix) else f"{config}{suffix}"
    return root / "data" / "evaluation" / dataset / f"{name}.txt"


def _load_prediction_full(root: Path, config: str, suffix: str, scene_id: str, scene_vertices: int) -> np.ndarray:
    pred_path = _prediction_dir(root, config, suffix) / f"{scene_id}.npz"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    with np.load(pred_path) as pred:
        masks = np.asarray(pred["pred_masks"], dtype=bool)
    if masks.shape[0] == scene_vertices:
        return masks
    pre_points = np.load(_tmp_path(root, config, scene_id)).astype(np.int64)
    if masks.shape[0] != pre_points.shape[0]:
        raise ValueError(
            f"{scene_id}: {config} pred first dim {masks.shape[0]} is neither full scene "
            f"{scene_vertices} nor its own pre_points {pre_points.shape[0]}"
        )
    out = np.zeros((scene_vertices, masks.shape[1]), dtype=bool)
    out[pre_points, :] = masks
    return out


def _load_scores(root: Path, config: str, suffix: str, scene_id: str) -> np.ndarray:
    pred_path = _prediction_dir(root, config, suffix) / f"{scene_id}.npz"
    with np.load(pred_path) as pred:
        return np.asarray(pred["pred_score"], dtype=np.float64)


def _count_gt_instances(gt_ids: np.ndarray, min_region_size: int = 1) -> int:
    ids, counts = np.unique(gt_ids[gt_ids >= 1000].astype(np.int64), return_counts=True)
    return int(np.count_nonzero(counts >= int(min_region_size))) if ids.size else 0


def _gt_masks(gt_ids: np.ndarray, min_region_size: int) -> tuple[list[int], list[np.ndarray]]:
    ids, counts = np.unique(gt_ids[gt_ids >= 1000].astype(np.int64), return_counts=True)
    out_ids: list[int] = []
    masks: list[np.ndarray] = []
    for gt_id, count in zip(ids.tolist(), counts.tolist()):
        if int(count) < int(min_region_size):
            continue
        out_ids.append(int(gt_id))
        masks.append(gt_ids == int(gt_id))
    return out_ids, masks


def _iou_matrix(gt_masks: list[np.ndarray], pred_support: np.ndarray) -> np.ndarray:
    if not gt_masks or pred_support.shape[1] == 0:
        return np.zeros((len(gt_masks), pred_support.shape[1]), dtype=np.float64)
    pred_areas = pred_support.sum(axis=0).astype(np.float64)
    out = np.zeros((len(gt_masks), pred_support.shape[1]), dtype=np.float64)
    for row_idx, gt_mask in enumerate(gt_masks):
        gt_area = float(np.count_nonzero(gt_mask))
        inter = pred_support[gt_mask, :].sum(axis=0).astype(np.float64)
        union = gt_area + pred_areas - inter
        valid = union > 0.0
        out[row_idx, valid] = inter[valid] / union[valid]
    return out


def _average_precision(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    if num_gt <= 0:
        return float("nan")
    if tp.size == 0:
        return 0.0
    tp_cum = np.cumsum(tp.astype(np.float64))
    fp_cum = np.cumsum(fp.astype(np.float64))
    recall = tp_cum / float(num_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def _diagnostic_ap(iou: np.ndarray, pred_scores: np.ndarray, pred_areas: np.ndarray, threshold: float) -> float:
    num_gt = int(iou.shape[0])
    valid_pred = np.flatnonzero(pred_areas >= 100)
    if num_gt == 0:
        return float("nan")
    if valid_pred.size == 0:
        return 0.0
    order = sorted(valid_pred.tolist(), key=lambda idx: (-float(pred_scores[idx]), int(idx)))
    gt_used = np.zeros((num_gt,), dtype=bool)
    tp = np.zeros((len(order),), dtype=bool)
    fp = np.zeros((len(order),), dtype=bool)
    for rank, pred_idx in enumerate(order):
        if num_gt == 0:
            fp[rank] = True
            continue
        gt_idx = int(np.argmax(iou[:, pred_idx]))
        best = float(iou[gt_idx, pred_idx])
        if best > float(threshold) and not gt_used[gt_idx]:
            tp[rank] = True
            gt_used[gt_idx] = True
        else:
            fp[rank] = True
    return _average_precision(tp, fp, num_gt)


def _scene_diagnostics(
    *,
    root: Path,
    config: str,
    suffix: str,
    scene_id: str,
    min_region_size: int,
) -> dict[str, Any]:
    gt_path = root / "data" / "scannet" / "gt" / f"{scene_id}.txt"
    gt_ids_full = np.loadtxt(gt_path).astype(np.int64)
    scene_vertices = int(gt_ids_full.shape[0])
    pre_points = np.load(_tmp_path(root, config, scene_id)).astype(np.int64)
    masks = _load_prediction_full(root, config, suffix, scene_id, scene_vertices)
    scores = _load_scores(root, config, suffix, scene_id)
    if scores.shape[0] != masks.shape[1]:
        scores = np.ones((masks.shape[1],), dtype=np.float64)

    target_mask = masks[pre_points, :] if pre_points.size else np.zeros((0, masks.shape[1]), dtype=bool)
    union = np.any(masks, axis=1) if masks.shape[1] else np.zeros((scene_vertices,), dtype=bool)
    target_union = np.any(target_mask, axis=1) if target_mask.shape[1] else np.zeros((pre_points.shape[0],), dtype=bool)
    owner_counts = masks.sum(axis=1).astype(np.int64) if masks.shape[1] else np.zeros((scene_vertices,), dtype=np.int64)
    conflict_points = int(np.count_nonzero(owner_counts[union] > 1)) if np.any(union) else 0
    pred_areas_full = masks.sum(axis=0).astype(np.int64) if masks.shape[1] else np.zeros((0,), dtype=np.int64)
    pred_areas_target = target_mask.sum(axis=0).astype(np.int64) if target_mask.shape[1] else np.zeros((0,), dtype=np.int64)

    gt_ids_target = gt_ids_full[pre_points]
    gt_eval_ids, gt_eval_masks = _gt_masks(gt_ids_target, min_region_size=min_region_size)
    iou = _iou_matrix(gt_eval_masks, target_mask)
    if iou.shape[0] == 0:
        best_iou = np.zeros((0,), dtype=np.float64)
    elif iou.shape[1] == 0:
        best_iou = np.zeros((iou.shape[0],), dtype=np.float64)
    else:
        best_iou = iou.max(axis=1)
    duplicate_counts = (iou >= 0.25).sum(axis=1) if iou.shape[0] else np.zeros((0,), dtype=np.int64)
    ap_thresholds = [round(v, 2) for v in np.arange(0.5, 0.95, 0.05).tolist()]
    diagnostic_aps = [_diagnostic_ap(iou, scores, pred_areas_target, threshold) for threshold in ap_thresholds]

    return {
        "scene_id": scene_id,
        "num_scene_vertices": scene_vertices,
        "num_pre_points": int(pre_points.shape[0]),
        "pre_points_ratio": float(pre_points.shape[0] / max(scene_vertices, 1)),
        "num_prediction_union": int(np.count_nonzero(union)),
        "prediction_union_ratio": float(np.count_nonzero(union) / max(scene_vertices, 1)),
        "prediction_union_in_target_count": int(np.count_nonzero(target_union)),
        "prediction_union_in_target_ratio_of_scene": float(np.count_nonzero(target_union) / max(scene_vertices, 1)),
        "prediction_union_in_target_ratio_of_target": float(
            np.count_nonzero(target_union) / max(int(pre_points.shape[0]), 1)
        ),
        "num_pred_instances": int(masks.shape[1]),
        "mean_points_per_pred": float(np.mean(pred_areas_full)) if pred_areas_full.size else 0.0,
        "tiny_mask_ratio_lt100_vertices": float(np.mean(pred_areas_full < 100)) if pred_areas_full.size else 0.0,
        "large_mask_ratio_gt1000_vertices": float(np.mean(pred_areas_full > 1000)) if pred_areas_full.size else 0.0,
        "conflict_rate": float(conflict_points / max(int(np.count_nonzero(union)), 1)),
        "num_gt_instances_in_pre_points": _count_gt_instances(gt_ids_target),
        "num_gt_instances_fullmesh": _count_gt_instances(gt_ids_full),
        "num_eval_gt_instances_in_pre_points": int(len(gt_eval_ids)),
        "per_gt_best_iou_mean": float(np.mean(best_iou)) if best_iou.size else None,
        "gt_iou_ge_025_count": int(np.count_nonzero(best_iou >= 0.25)),
        "gt_iou_ge_050_count": int(np.count_nonzero(best_iou >= 0.50)),
        "missed_eval_gt_count_iou_lt_025": int(np.count_nonzero(best_iou < 0.25)),
        "duplicate_predictions_per_gt_mean_at_025": float(np.mean(np.maximum(duplicate_counts - 1, 0)))
        if duplicate_counts.size
        else None,
        "diagnostic_ap": float(np.nanmean(diagnostic_aps)) if diagnostic_aps else None,
        "diagnostic_ap50": _diagnostic_ap(iou, scores, pred_areas_target, 0.50),
        "diagnostic_ap25": _diagnostic_ap(iou, scores, pred_areas_target, 0.25),
        "diagnostic_ap_note": "Internal class-agnostic support diagnostic; official AP columns come from evaluation.evaluate output.",
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(mean(values)) if values else None


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(row[key]) for row in rows if row.get(key) is not None))


def _aggregate_row(
    *,
    root: Path,
    dataset: str,
    suffix: str,
    scene_ids: list[str],
    row_spec: dict[str, Any],
    min_region_size: int,
) -> dict[str, Any]:
    config = str(row_spec["output_config"])
    manifest, manifest_path = load_prediction_manifest(root, config, suffix.lstrip("_"))
    metrics = _parse_metric_file(_metric_path(root, dataset, config, suffix))
    scene_rows = [
        _scene_diagnostics(
            root=root,
            config=config,
            suffix=suffix,
            scene_id=scene_id,
            min_region_size=min_region_size,
        )
        for scene_id in scene_ids
    ]
    diagnostic_scene_ap = [
        float(row["diagnostic_ap"])
        for row in scene_rows
        if row.get("diagnostic_ap") is not None and np.isfinite(float(row["diagnostic_ap"]))
    ]
    aggregate = {
        "method": str(row_spec.get("method", config)),
        "prediction_config": str(row_spec.get("prediction_config", manifest.get("prediction_config", "")) if manifest else row_spec.get("prediction_config", "")),
        "pre_points_config": str(row_spec.get("pre_points_config", manifest.get("pre_points_config", "")) if manifest else row_spec.get("pre_points_config", "")),
        "output_config": config,
        "eval_policy": str(row_spec.get("eval_policy", manifest.get("eval_policy", "")) if manifest else row_spec.get("eval_policy", "")),
        "split": str(row_spec.get("split", "")),
        "num_scenes": int(len(scene_rows)),
        "ap": metrics["ap"],
        "ap50": metrics["ap50"],
        "ap25": metrics["ap25"],
        "pre_points_ratio": _mean(scene_rows, "pre_points_ratio"),
        "prediction_union_ratio": _mean(scene_rows, "prediction_union_ratio"),
        "union_in_target_ratio_of_scene": _mean(scene_rows, "prediction_union_in_target_ratio_of_scene"),
        "union_in_target_ratio_of_target": _mean(scene_rows, "prediction_union_in_target_ratio_of_target"),
        "gt_crop": _mean(scene_rows, "num_gt_instances_in_pre_points"),
        "gt_full": _mean(scene_rows, "num_gt_instances_fullmesh"),
        "num_pred_per_scene": _mean(scene_rows, "num_pred_instances"),
        "objects_per_scene": _mean(scene_rows, "num_pred_instances"),
        "points_per_scene": _mean(scene_rows, "num_prediction_union"),
        "tiny_mask_ratio_lt100_vertices": _mean(scene_rows, "tiny_mask_ratio_lt100_vertices"),
        "large_mask_ratio_gt1000_vertices": _mean(scene_rows, "large_mask_ratio_gt1000_vertices"),
        "conflict_rate": _mean(scene_rows, "conflict_rate"),
        "per_scene_diagnostic_ap_variance": float(variance(diagnostic_scene_ap))
        if len(diagnostic_scene_ap) >= 2
        else None,
        "per_gt_best_iou_mean": _mean(scene_rows, "per_gt_best_iou_mean"),
        "gt_iou_ge_025_count": _sum(scene_rows, "gt_iou_ge_025_count"),
        "gt_iou_ge_050_count": _sum(scene_rows, "gt_iou_ge_050_count"),
        "missed_eval_gt_count_iou_lt_025": _sum(scene_rows, "missed_eval_gt_count_iou_lt_025"),
        "duplicate_predictions_per_gt_mean_at_025": _mean(scene_rows, "duplicate_predictions_per_gt_mean_at_025"),
        "uses_gt": bool(manifest.get("uses_gt", False)) if manifest else None,
        "is_diagnostic_only": bool(manifest.get("is_diagnostic_only", False)) if manifest else None,
        "is_method_result": bool(manifest.get("is_method_result", False)) if manifest else None,
        "manifest_path": str(manifest_path) if manifest_path else "",
        "metric_integrity_pass": row_spec.get("metric_integrity_pass"),
        "scene_rows": scene_rows,
    }
    return aggregate


def _attach_same_support_gaps(rows: list[dict[str, Any]], stream3d_config: str) -> None:
    baselines: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["prediction_config"] == stream3d_config:
            baselines[str(row["pre_points_config"])] = row
    for row in rows:
        baseline = baselines.get(str(row["pre_points_config"]))
        if baseline is None:
            row["stream3d_same_support_ap"] = None
            row["stream3d_same_support_ap50"] = None
            row["stream3d_same_support_ap25"] = None
            row["same_support_gap_ap"] = None
            row["same_support_gap_ap50"] = None
            row["same_support_gap_ap25"] = None
            continue
        row["stream3d_same_support_ap"] = baseline.get("ap")
        row["stream3d_same_support_ap50"] = baseline.get("ap50")
        row["stream3d_same_support_ap25"] = baseline.get("ap25")
        row["same_support_gap_ap"] = (
            float(row["ap"]) - float(baseline["ap"])
            if row.get("ap") is not None and baseline.get("ap") is not None
            else None
        )
        row["same_support_gap_ap50"] = (
            float(row["ap50"]) - float(baseline["ap50"])
            if row.get("ap50") is not None and baseline.get("ap50") is not None
            else None
        )
        row["same_support_gap_ap25"] = (
            float(row["ap25"]) - float(baseline["ap25"])
            if row.get("ap25") is not None and baseline.get("ap25") is not None
            else None
        )


def _fmt(value: Any, scale: float = 1.0, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value * scale:.{digits}f}"
    return str(value)


def _write_markdown(output: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Stream4D v9 Unified Evaluation Matrix",
        "",
        "Official AP columns are parsed from `evaluation.evaluate` output. Diagnostic IoU/support columns are computed from the same prediction and TMP support files.",
        "",
        "| method | prediction | pre_points | policy | AP | AP50 | AP25 | pre% | union% | GT crop/full | #pred | conflict | Stream3D same-support AP/AP50/AP25 | gap |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["prediction_config"]),
                    str(row["pre_points_config"]),
                    str(row["eval_policy"]),
                    _fmt(row.get("ap"), 100.0),
                    _fmt(row.get("ap50"), 100.0),
                    _fmt(row.get("ap25"), 100.0),
                    _fmt(row.get("pre_points_ratio"), 100.0),
                    _fmt(row.get("prediction_union_ratio"), 100.0),
                    f"{_fmt(row.get('gt_crop'), 1.0, 2)}/{_fmt(row.get('gt_full'), 1.0, 2)}",
                    _fmt(row.get("num_pred_per_scene"), 1.0, 2),
                    _fmt(row.get("conflict_rate"), 100.0),
                    f"{_fmt(row.get('stream3d_same_support_ap'), 100.0)}/{_fmt(row.get('stream3d_same_support_ap50'), 100.0)}/{_fmt(row.get('stream3d_same_support_ap25'), 100.0)}",
                    f"{_fmt(row.get('same_support_gap_ap'), 100.0)}/{_fmt(row.get('same_support_gap_ap50'), 100.0)}/{_fmt(row.get('same_support_gap_ap25'), 100.0)}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| method | union in target scene/target | tiny <100 | large >1000 | best IoU mean | GT IoU>=.25 | GT IoU>=.50 | missed GT | dup/GT | diag AP var | manifest |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    f"{_fmt(row.get('union_in_target_ratio_of_scene'), 100.0)}/{_fmt(row.get('union_in_target_ratio_of_target'), 100.0)}",
                    _fmt(row.get("tiny_mask_ratio_lt100_vertices"), 100.0),
                    _fmt(row.get("large_mask_ratio_gt1000_vertices"), 100.0),
                    _fmt(row.get("per_gt_best_iou_mean"), 1.0),
                    str(row.get("gt_iou_ge_025_count")),
                    str(row.get("gt_iou_ge_050_count")),
                    str(row.get("missed_eval_gt_count_iou_lt_025")),
                    _fmt(row.get("duplicate_predictions_per_gt_mean_at_025"), 1.0),
                    _fmt(row.get("per_scene_diagnostic_ap_variance"), 1.0, 6),
                    str(row.get("manifest_path", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- matrix: `{Path(args.matrix_json).resolve()}`",
            f"- seq_list: `{Path(args.seq_list).resolve()}`",
            f"- root: `{Path(args.root).resolve()}`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--matrix-json", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--dataset", default="scannet")
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--stream3d-config", default="scannet")
    parser.add_argument("--min-region-size", type=int, default=100)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scene_ids = _read_seq_list((root / args.seq_list).resolve())
    matrix_path = Path(args.matrix_json)
    if not matrix_path.is_absolute():
        matrix_path = root / matrix_path
    specs = json.loads(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(specs, list):
        raise ValueError("--matrix-json must contain a list of row specs")

    rows = [
        _aggregate_row(
            root=root,
            dataset=args.dataset,
            suffix=args.pred_suffix,
            scene_ids=scene_ids,
            row_spec=spec,
            min_region_size=int(args.min_region_size),
        )
        for spec in specs
    ]
    _attach_same_support_gaps(rows, stream3d_config=args.stream3d_config)

    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = root / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "rows": rows}
    output_prefix.with_suffix(".json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    flat_rows = [{key: value for key, value in row.items() if key != "scene_rows"} for row in rows]
    fieldnames = list(flat_rows[0].keys()) if flat_rows else ["method"]
    with output_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_safe(flat_rows))
    _write_markdown(output_prefix.with_suffix(".md"), rows, args)
    print(f"[v9-unified-eval] wrote {output_prefix.with_suffix('.json')}")
    print(f"[v9-unified-eval] wrote {output_prefix.with_suffix('.csv')}")
    print(f"[v9-unified-eval] wrote {output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
