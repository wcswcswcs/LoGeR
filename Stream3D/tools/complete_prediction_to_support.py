from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.strip().startswith("#")]


def _tmp_path(root: Path, config: str, scene_id: str) -> Path:
    candidates = [
        root / "data" / "TMP" / config / f"{scene_id}_pre_points.npy",
        root / "TMP" / config / f"{scene_id}_pre_points.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _pred_path(root: Path, config: str, scene_id: str, suffix: str) -> Path:
    dirname = config if config.endswith(suffix) else f"{config}{suffix}"
    return root / "data" / "prediction" / dirname / f"{scene_id}.npz"


def _scene_points(root: Path, scene_id: str) -> np.ndarray:
    path = root / "data" / "scannet" / "processed" / scene_id / f"{scene_id}_vh_clean_2.ply"
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"Failed to read scene points: {path}")
    return points


def _load_full_masks(root: Path, config: str, scene_id: str, suffix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = _pred_path(root, config, scene_id, suffix)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        masks = data["pred_masks"].astype(bool, copy=False)
        scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)
    scene_vertices = _scene_points(root, scene_id).shape[0]
    if masks.shape[0] == scene_vertices:
        return masks, scores, classes
    pre_points = np.load(_tmp_path(root, config, scene_id)).astype(np.int64)
    if masks.shape[0] != pre_points.shape[0]:
        raise ValueError(f"{scene_id}: mask rows={masks.shape[0]} cannot expand with pre_points={pre_points.shape[0]}")
    full = np.zeros((scene_vertices, masks.shape[1]), dtype=bool)
    full[pre_points, :] = masks
    return full, scores, classes


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


def process_scene(args: argparse.Namespace, root: Path, scene_id: str) -> dict[str, Any]:
    points = _scene_points(root, scene_id)
    masks, scores, classes = _load_full_masks(root, args.input_config, scene_id, args.pred_suffix)
    target_ids = np.load(_tmp_path(root, args.target_support_config, scene_id)).astype(np.int64)
    target_ids = target_ids[(target_ids >= 0) & (target_ids < points.shape[0])]

    core_ids_by_object: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    all_core_ids: list[np.ndarray] = []
    for obj_idx in range(masks.shape[1]):
        ids = np.flatnonzero(masks[:, obj_idx]).astype(np.int64)
        if ids.shape[0] < int(args.min_core_points):
            core_ids_by_object.append(np.zeros((0,), dtype=np.int64))
            continue
        core_ids_by_object.append(ids)
        all_core_ids.append(ids)
        labels.append(np.full((ids.shape[0],), obj_idx, dtype=np.int64))

    out_masks = np.zeros((points.shape[0], masks.shape[1]), dtype=bool)
    if all_core_ids and target_ids.size:
        core_ids = np.concatenate(all_core_ids, axis=0)
        core_labels = np.concatenate(labels, axis=0)
        tree = cKDTree(points[core_ids])
        distances, nn = tree.query(points[target_ids], k=1, workers=-1)
        keep = np.isfinite(distances)
        if float(args.max_radius) >= 0.0:
            keep &= distances <= float(args.max_radius)
        kept_target = target_ids[keep]
        kept_labels = core_labels[nn[keep]]
        out_masks[kept_target, kept_labels] = True
        if bool(args.keep_core_points):
            for obj_idx, ids in enumerate(core_ids_by_object):
                if ids.size:
                    out_masks[ids, obj_idx] = True
    elif bool(args.keep_core_points):
        out_masks = masks.copy()

    keep_objects = np.flatnonzero(out_masks.sum(axis=0) >= int(args.min_points_per_object)).astype(np.int64)
    out_masks = out_masks[:, keep_objects]
    out_scores = scores[keep_objects] if scores.shape[0] == masks.shape[1] else np.ones((keep_objects.shape[0],), dtype=np.float32)
    out_classes = classes[keep_objects] if classes.shape[0] == masks.shape[1] else np.zeros((keep_objects.shape[0],), dtype=np.int32)

    pred_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_dir / f"{scene_id}.npz", pred_masks=out_masks, pred_score=out_scores, pred_classes=out_classes)
    pre_points = np.flatnonzero(out_masks.any(axis=1)).astype(np.int64) if out_masks.shape[1] else np.zeros((0,), dtype=np.int64)
    tmp_dir = root / "data" / "TMP" / args.output_config
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.save(tmp_dir / f"{scene_id}_pre_points.npy", pre_points)

    conflict = int(np.count_nonzero(out_masks.sum(axis=1) > 1)) if out_masks.shape[1] else 0
    return {
        "scene_id": scene_id,
        "input_config": args.input_config,
        "target_support_config": args.target_support_config,
        "output_config": args.output_config,
        "num_scene_points": int(points.shape[0]),
        "num_target_support_points": int(target_ids.shape[0]),
        "num_input_objects": int(masks.shape[1]),
        "num_output_objects": int(out_masks.shape[1]),
        "num_output_points": int(pre_points.shape[0]),
        "output_point_ratio": float(pre_points.shape[0] / max(points.shape[0], 1)),
        "target_support_fill_ratio": float(pre_points.shape[0] / max(target_ids.shape[0], 1)),
        "conflict_rate": float(conflict / max(pre_points.shape[0], 1)),
    }


def _aggregate(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    numeric_keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float, np.generic))})
    return {
        "algorithm": "complete_prediction_to_support",
        "uses_gt": False,
        "is_method_result": True,
        "input_config": args.input_config,
        "target_support_config": args.target_support_config,
        "output_config": args.output_config,
        "num_scenes": len(rows),
        **{
            f"{key}_mean": float(np.mean([float(row[key]) for row in rows if key in row]))
            for key in numeric_keys
            if any(key in row for row in rows)
        },
    }


def _write_summary(root: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _aggregate(rows, args)
    payload = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    prefix = out_dir / f"{args.output_config}_summary"
    prefix.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = ["scene_id"] + sorted({k for row in rows for k in row if k != "scene_id"})
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    lines = [
        f"# {args.output_config}",
        "",
        "| scene | objects | points | target fill | point ratio | conflict |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene_id']} | {row['num_output_objects']} | {row['num_output_points']} | "
            f"{row['target_support_fill_ratio']:.6f} | {row['output_point_ratio']:.6f} | {row['conflict_rate']:.6f} |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.input_config, args.target_support_config],
        pre_points_policy="recompute",
        support_policy=f"support_completion:{args.target_support_config}:radius={args.max_radius}",
        notes="Support completion diagnostic/method prototype: assigns target support points to nearest input object core without GT.",
        extra={
            "algorithm": "complete_prediction_to_support",
            "eval_policy": args.eval_policy,
            "input_config": args.input_config,
            "target_support_config": args.target_support_config,
            "max_radius": float(args.max_radius),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=root, pred_suffix=args.pred_suffix.lstrip("_"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete sparse object masks onto a target support with nearest-core WTA.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--target-support-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--max-radius", type=float, default=0.10, help="Negative value assigns every target support point.")
    parser.add_argument("--min-core-points", type=int, default=20)
    parser.add_argument("--min-points-per-object", type=int, default=20)
    parser.add_argument("--keep-core-points", action="store_true")
    parser.add_argument("--summary-root", default="outputs/v9_support_completion")
    parser.add_argument("--eval-policy", default="own_recompute_support_completion")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    rows = [process_scene(args, root, scene_id) for scene_id in _read_seq_list(root / args.seq_list)]
    _write_summary(root, args, rows)
    print(f"[complete-prediction-to-support] wrote {root / args.summary_root / (args.output_config + '_summary.md')}")


if __name__ == "__main__":
    main()
