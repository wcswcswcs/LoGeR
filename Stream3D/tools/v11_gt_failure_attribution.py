from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.constants import SCANNET_IDS


MIN_REGION_SIZE = 100


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _split(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _prediction_dir(root: Path, config: str, suffix: str) -> Path:
    suffix_norm = suffix[1:] if suffix.startswith("_") else suffix
    if config.endswith(suffix_norm):
        return root / "data" / "prediction" / config
    return root / "data" / "prediction" / f"{config}_{suffix_norm}"


def _tmp_path(root: Path, config: str, scene: str) -> Path:
    return root / "data" / "TMP" / config / f"{scene}_pre_points.npy"


def _load_prediction_full(root: Path, config: str, suffix: str, scene: str, scene_vertices: int) -> np.ndarray:
    path = _prediction_dir(root, config, suffix) / f"{scene}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        masks = np.asarray(data["pred_masks"], dtype=bool)
    if masks.shape[0] == scene_vertices:
        return masks
    pre_points = np.load(_tmp_path(root, config, scene)).astype(np.int64)
    if masks.shape[0] != pre_points.shape[0]:
        raise ValueError(
            f"{scene}: {config} mask dim {masks.shape[0]} is neither full scene {scene_vertices} "
            f"nor its own pre_points {pre_points.shape[0]}"
        )
    full = np.zeros((scene_vertices, masks.shape[1]), dtype=bool)
    full[pre_points, :] = masks
    return full


def _class_agnostic_gt(gt_ids: np.ndarray) -> np.ndarray:
    return gt_ids % 1000 + int(SCANNET_IDS[0]) * 1000


def _gt_masks(gt_ids: np.ndarray) -> tuple[list[int], list[int], list[np.ndarray]]:
    ids: list[int] = []
    counts: list[int] = []
    masks: list[np.ndarray] = []
    for gt_id, count in zip(*np.unique(gt_ids[gt_ids >= 1000], return_counts=True)):
        if int(count) < MIN_REGION_SIZE:
            continue
        ids.append(int(gt_id))
        counts.append(int(count))
        masks.append(gt_ids == int(gt_id))
    return ids, counts, masks


def _best_iou(gt_mask: np.ndarray, pred_masks: np.ndarray) -> tuple[float, int]:
    if pred_masks.shape[1] == 0:
        return 0.0, -1
    pred_areas = pred_masks.sum(axis=0).astype(np.float64)
    gt_area = float(np.count_nonzero(gt_mask))
    inter = pred_masks[gt_mask, :].sum(axis=0).astype(np.float64)
    union = gt_area + pred_areas - inter
    iou = np.zeros((pred_masks.shape[1],), dtype=np.float64)
    valid = union > 0.0
    iou[valid] = inter[valid] / union[valid]
    idx = int(np.argmax(iou)) if iou.size else -1
    return (float(iou[idx]) if idx >= 0 else 0.0), idx


def _load_pool(root: Path, configs: list[str], suffix: str, scene: str, scene_vertices: int) -> np.ndarray:
    parts = [_load_prediction_full(root, config, suffix, scene, scene_vertices) for config in configs]
    return np.concatenate(parts, axis=1) if parts else np.zeros((scene_vertices, 0), dtype=bool)


def _classify(pool_iou: float, method_iou: float, *, low: float, high: float) -> str:
    if pool_iou < low:
        return "no_candidate"
    if method_iou < low:
        return "filtered_candidate"
    if pool_iou >= high and method_iou < high:
        return "wrong_assignment_or_fragmentation"
    if method_iou >= low and method_iou < high:
        return "boundary_bad"
    return "matched"


def _process_scene(args: argparse.Namespace, scene: str) -> list[dict[str, Any]]:
    root = Path(args.root)
    gt_path = root / "data" / "scannet" / "gt" / f"{scene}.txt"
    gt_ids = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
    scene_vertices = int(gt_ids.shape[0])
    gt_ids_list, gt_counts, gt_masks = _gt_masks(gt_ids)
    pool = _load_pool(root, _split(args.pool_configs), args.pred_suffix, scene, scene_vertices)
    method = _load_prediction_full(root, args.method_config, args.pred_suffix, scene, scene_vertices)
    rows: list[dict[str, Any]] = []
    for gt_id, gt_count, gt_mask in zip(gt_ids_list, gt_counts, gt_masks):
        pool_iou, pool_idx = _best_iou(gt_mask, pool)
        method_iou, method_idx = _best_iou(gt_mask, method)
        rows.append(
            {
                "scene": scene,
                "gt_id": int(gt_id),
                "gt_vertices": int(gt_count),
                "method_config": args.method_config,
                "pool_name": args.pool_name,
                "pool_best_iou": float(pool_iou),
                "pool_best_index": int(pool_idx),
                "method_best_iou": float(method_iou),
                "method_best_index": int(method_idx),
                "failure_class": _classify(
                    pool_iou,
                    method_iou,
                    low=float(args.low_iou),
                    high=float(args.high_iou),
                ),
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    classes = sorted({str(row["failure_class"]) for row in rows})
    counts = {name: int(sum(1 for row in rows if row["failure_class"] == name)) for name in classes}
    total = max(len(rows), 1)
    return {
        "diagnostic_only": True,
        "uses_gt": True,
        "is_method_result": False,
        "method_config": args.method_config,
        "pool_name": args.pool_name,
        "pool_configs": _split(args.pool_configs),
        "num_gt": int(len(rows)),
        "low_iou": float(args.low_iou),
        "high_iou": float(args.high_iou),
        "mean_pool_best_iou": float(np.mean([float(row["pool_best_iou"]) for row in rows])) if rows else 0.0,
        "mean_method_best_iou": float(np.mean([float(row["method_best_iou"]) for row in rows])) if rows else 0.0,
        "count_by_failure_class": counts,
        "ratio_by_failure_class": {name: float(value / total) for name, value in counts.items()},
    }


def _write_outputs(prefix: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(_json_safe({"summary": summary, "rows": rows}), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if rows:
        with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        f"# v11 GT Failure Attribution: {summary['method_config']}",
        "",
        "GT is used only for diagnostic attribution. This is not a method result.",
        "",
        "## Summary",
        "",
    ]
    for key in ("num_gt", "mean_pool_best_iou", "mean_method_best_iou"):
        lines.append(f"- {key}: `{summary.get(key)}`")
    lines.extend(["", "## Failure Classes", "", "| class | count | ratio |", "|---|---:|---:|"])
    for name, count in summary["count_by_failure_class"].items():
        ratio = summary["ratio_by_failure_class"].get(name, 0.0)
        lines.append(f"| {name} | {count} | {ratio:.6f} |")
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--pool-name", required=True)
    parser.add_argument("--pool-configs", required=True)
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--pred-suffix", default="class_agnostic")
    parser.add_argument("--low-iou", type=float, default=0.25)
    parser.add_argument("--high-iou", type=float, default=0.50)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for scene in _read_seq_list(Path(args.seq_list)):
        rows.extend(_process_scene(args, scene))
    summary = _aggregate(rows, args)
    _write_outputs(Path(args.output_prefix), rows, summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
