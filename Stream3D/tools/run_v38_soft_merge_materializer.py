from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v38_oracle_attribution import (
    _candidate_oracle_stats,
    _class_agnostic_gt,
    _parse_metric_file,
    _quality_stats,
)


INPUT_CONFIG = "v38_d1_const_min100_probe5"
MIN_AREA = 100


VARIANTS: dict[str, dict[str, Any]] = {
    "M1_iou050_soft_merge": {
        "config": "v38_m1_iou050_soft_merge_probe5",
        "overlap_mode": "iou",
        "overlap_threshold": 0.50,
        "max_instances": 300,
        "merge_after_cap_below_threshold": True,
        "description": "IoU 0.50 duplicate grouping; merge suppressed masks into kept masks instead of dropping them",
    },
    "M2_min_ioc050_soft_merge": {
        "config": "v38_m2_min_ioc050_soft_merge_probe5",
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.50,
        "max_instances": 300,
        "merge_after_cap_below_threshold": True,
        "description": "min-IoC 0.50 duplicate grouping; merge suppressed masks into kept masks instead of dropping them",
    },
    "M3_min_ioc030_soft_merge": {
        "config": "v38_m3_min_ioc030_soft_merge_probe5",
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.30,
        "max_instances": 300,
        "merge_after_cap_below_threshold": True,
        "description": "min-IoC 0.30 duplicate grouping; merge suppressed masks into kept masks instead of dropping them",
    },
    "M4_iou050_strict_cap_soft_merge": {
        "config": "v38_m4_iou050_strict_cap_soft_merge_probe5",
        "overlap_mode": "iou",
        "overlap_threshold": 0.50,
        "max_instances": 300,
        "merge_after_cap_below_threshold": False,
        "description": "IoU 0.50 duplicate grouping; merge only threshold-matched duplicates after max-instance cap, drop unmatched overflow",
    },
    "M5_min_ioc050_strict_cap_soft_merge": {
        "config": "v38_m5_min_ioc050_strict_cap_soft_merge_probe5",
        "overlap_mode": "min_ioc",
        "overlap_threshold": 0.50,
        "max_instances": 300,
        "merge_after_cap_below_threshold": False,
        "description": "min-IoC 0.50 duplicate grouping; merge only threshold-matched duplicates after max-instance cap, drop unmatched overflow",
    },
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _order_indices(areas: np.ndarray) -> list[int]:
    return sorted(range(areas.shape[0]), key=lambda idx: (-float(areas[idx]), int(idx)))


def _overlap(inter: int, kept_area: float, area: float, mode: str) -> float:
    if mode == "iou":
        denom = kept_area + area - float(inter)
    elif mode == "min_ioc":
        denom = min(kept_area, area)
    elif mode == "candidate_ioc":
        denom = area
    else:
        raise ValueError(f"Unsupported overlap mode: {mode}")
    return float(inter) / max(float(denom), 1.0)


def _soft_merge(
    masks: np.ndarray,
    max_instances: int,
    threshold: float,
    mode: str,
    merge_after_cap_below_threshold: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    areas = masks.sum(axis=0).astype(np.float64)
    kept_masks: list[np.ndarray] = []
    kept_source_indices: list[int] = []
    group_sizes: list[int] = []
    owners_by_point: dict[int, list[int]] = defaultdict(list)
    point_cache: dict[int, np.ndarray] = {}
    dropped_low_area = 0
    dropped_unmatched_after_cap = 0
    merged_count = 0

    for idx in _order_indices(areas):
        area = float(areas[idx])
        if area < float(MIN_AREA):
            dropped_low_area += 1
            continue
        points = point_cache.get(idx)
        if points is None:
            points = np.flatnonzero(masks[:, idx]).astype(np.int64)
            point_cache[idx] = points
        intersections: dict[int, int] = defaultdict(int)
        for point in points.tolist():
            for kept_idx in owners_by_point.get(int(point), ()):
                intersections[int(kept_idx)] += 1
        best_kept = -1
        best_overlap = 0.0
        for kept_idx, inter in intersections.items():
            ov = _overlap(inter, float(np.count_nonzero(kept_masks[kept_idx])), area, mode)
            if ov > best_overlap:
                best_overlap = ov
                best_kept = int(kept_idx)
        if best_kept >= 0 and best_overlap >= float(threshold):
            before = kept_masks[best_kept].copy()
            kept_masks[best_kept][points] = True
            new_points = np.flatnonzero(kept_masks[best_kept] & ~before).astype(np.int64)
            for point in new_points.tolist():
                owners_by_point.setdefault(int(point), []).append(best_kept)
            group_sizes[best_kept] += 1
            merged_count += 1
            continue
        if len(kept_masks) >= int(max_instances):
            if best_kept >= 0 and (bool(merge_after_cap_below_threshold) or best_overlap >= float(threshold)):
                before = kept_masks[best_kept].copy()
                kept_masks[best_kept][points] = True
                new_points = np.flatnonzero(kept_masks[best_kept] & ~before).astype(np.int64)
                for point in new_points.tolist():
                    owners_by_point.setdefault(int(point), []).append(best_kept)
                group_sizes[best_kept] += 1
                merged_count += 1
            else:
                dropped_unmatched_after_cap += 1
            continue
        new_mask = np.zeros((masks.shape[0],), dtype=bool)
        new_mask[points] = True
        kept_idx = len(kept_masks)
        kept_masks.append(new_mask)
        kept_source_indices.append(int(idx))
        group_sizes.append(1)
        for point in points.tolist():
            owners_by_point.setdefault(int(point), []).append(kept_idx)

    if kept_masks:
        out = np.stack(kept_masks, axis=1)
        source = np.asarray(kept_source_indices, dtype=np.int64)
    else:
        out = masks[:, :0]
        source = np.zeros((0,), dtype=np.int64)
    diag = {
        "kept_groups": int(out.shape[1]),
        "merged_count": int(merged_count),
        "dropped_low_area": int(dropped_low_area),
        "dropped_unmatched_after_cap": int(dropped_unmatched_after_cap),
        "mean_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "max_group_size": int(max(group_sizes)) if group_sizes else 0,
    }
    return out, source, diag


def _write_prediction(root: Path, config: str, scene: str, masks: np.ndarray, scores: np.ndarray, classes: np.ndarray, raw_pre_points: np.ndarray) -> None:
    pred_dir = root / "data/prediction" / f"{config}_class_agnostic"
    tmp_dir = root / "data/TMP" / config
    pred_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{scene}.npz",
        pred_masks=masks.astype(bool, copy=False),
        pred_score=scores.astype(np.float32, copy=False),
        pred_classes=classes.astype(np.int32, copy=False),
    )
    np.save(tmp_dir / f"{scene}_pre_points.npy", raw_pre_points.astype(np.int64, copy=False))


def _write_manifest(root: Path, variant: str, meta: dict[str, Any], input_config: str) -> None:
    config = str(meta["config"])
    manifest = build_prediction_manifest(
        root=root,
        output_config=config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[input_config],
        pre_points_policy=f"copied_raw_support:{input_config}",
        support_policy=f"v38_phaseF_soft_merge:{variant}:{meta['overlap_mode']}@{meta['overlap_threshold']}",
        notes=f"v38 Phase F soft merge {variant}: {meta['description']}. Diagnostic-only because ScanNet mesh export bridge is used.",
        extra={
            "phase": "v38_phaseF_soft_merge",
            "variant": variant,
            "forbidden_for_method_table": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "alignment_source": "scannet_rgbd_pose_mesh_export_bridge",
            "alignment_used_for_prediction": True,
        },
    )
    write_prediction_manifest(config, manifest, root=root, pred_suffix="class_agnostic")


def _generate(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    scenes = _read_split(root / args.split)
    for variant, meta in VARIANTS.items():
        _write_manifest(root, variant, meta, args.input_config)
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        pred_path = root / "data/prediction" / f"{args.input_config}_class_agnostic" / f"{scene}.npz"
        pre_path = root / "data/TMP" / args.input_config / f"{scene}_pre_points.npy"
        gt_path = root / args.gt_path / f"{scene}.txt"
        with np.load(pred_path) as pred:
            masks = np.asarray(pred["pred_masks"], dtype=bool)
            classes = np.asarray(pred["pred_classes"], dtype=np.int32)
        raw_pre_points = np.load(pre_path).astype(np.int64)
        gt_eval = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
        for variant, meta in VARIANTS.items():
            out_masks, source_indices, diag = _soft_merge(
                masks,
                max_instances=int(meta["max_instances"]),
                threshold=float(meta["overlap_threshold"]),
                mode=str(meta["overlap_mode"]),
                merge_after_cap_below_threshold=bool(meta.get("merge_after_cap_below_threshold", True)),
            )
            scores = np.ones((out_masks.shape[1],), dtype=np.float32)
            out_classes = classes[source_indices] if source_indices.size else classes[:0]
            config = str(meta["config"])
            _write_prediction(root, config, scene, out_masks, scores, out_classes, raw_pre_points)
            stats = _candidate_oracle_stats(out_masks, scores, gt_eval, raw_pre_points)
            row = _quality_stats(scene, variant, out_masks, scores, gt_eval, raw_pre_points, stats)
            row.update({"config": config, "num_predictions_before": int(masks.shape[1]), **diag})
            rows.append(row)
    return rows


def _run_eval(args: argparse.Namespace, root: Path, config: str, output_root: Path) -> dict[str, Any]:
    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metric_file = eval_dir / f"{config}_class_agnostic.txt"
    log_path = eval_dir / f"{config}_evaluate.log"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(root / "data/prediction" / f"{config}_class_agnostic"),
        "--gt_path",
        str(root / args.gt_path),
        "--dataset",
        "scannet",
        "--no_class",
        "--tmp_root",
        str(root / "data/TMP"),
        "--tmp_config",
        config,
        "--output_file",
        str(metric_file),
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT)
    return {
        "config": config,
        "command": " ".join(cmd),
        "exit_code": int(proc.returncode),
        "metric_file": str(metric_file),
        "log_path": str(log_path),
        "metrics": _parse_metric_file(metric_file) if metric_file.exists() else {},
    }


def _aggregate(rows: list[dict[str, Any]], evals: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for variant in VARIANTS:
        items = [row for row in rows if row["variant"] == variant]
        agg: dict[str, Any] = {"variant": variant, "scene_count": len(items)}
        keys = sorted(k for row in items for k, v in row.items() if isinstance(v, (int, float)) and k != "scene")
        for key in keys:
            vals = [float(row[key]) for row in items if isinstance(row.get(key), (int, float))]
            if vals:
                agg[key] = float(np.mean(vals))
        ev = evals[variant]
        agg.update(
            {
                "AP": ev["metrics"].get("AP"),
                "AP50": ev["metrics"].get("AP50"),
                "AP25": ev["metrics"].get("AP25"),
                "eval_exit_code": ev["exit_code"],
                "metric_file": ev["metric_file"],
            }
        )
        matrix.append(agg)
    return matrix


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    if Path.cwd().resolve() != root:
        os.chdir(root)
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _generate(args, root)
    evals = {variant: _run_eval(args, root, str(meta["config"]), output_root) for variant, meta in VARIANTS.items()}
    matrix = _aggregate(rows, evals)
    f0_metric = _parse_metric_file(root / "outputs/audit/v38_object_materializer/eval/v38_d1_const_min100_probe5_class_agnostic.txt")
    summary = {
        "phase": "v38_phaseF_soft_merge",
        "input_config": args.input_config,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "matrix": matrix,
        "scene_rows": rows,
        "evals": evals,
        "phaseF_soft_merge_gate": {
            row["variant"]: {
                "num_predictions_after_le_300": bool(float(row.get("num_predictions", 1e9)) <= 300.0),
                "duplicate_rate_after_le_010": bool(float(row.get("duplicate_prediction_rate", 1.0)) <= 0.10),
                "conflict_rate_after_le_010": bool(float(row.get("mean_export_conflict_rate", 1.0)) <= 0.10),
                "AP_not_worse_than_F0_by_gt_001": bool(float(row.get("AP") or 0.0) >= float(f0_metric["AP"]) - 0.01),
                "AP25_improves_or_stable": bool(float(row.get("AP25") or 0.0) >= float(f0_metric["AP25"]) - 1e-12),
            }
            for row in matrix
        },
    }
    _write_json(output_root / "soft_merge_summary.json", summary)
    _write_csv(output_root / "soft_merge_matrix.csv", matrix)
    _write_csv(output_root / "soft_merge_scene_rows.csv", rows)
    lines = [
        "# Stream4D v38 Phase F Soft Merge",
        "",
        "| variant | AP | AP50 | AP25 | #pred | conflict | dup | best IoU mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in matrix:
        lines.append(
            "| {variant} | {AP} | {AP50} | {AP25} | {pred} | {conflict} | {dup} | {iou} |".format(
                variant=row["variant"],
                AP=row.get("AP"),
                AP50=row.get("AP50"),
                AP25=row.get("AP25"),
                pred=row.get("num_predictions"),
                conflict=row.get("mean_export_conflict_rate"),
                dup=row.get("duplicate_prediction_rate"),
                iou=row.get("per_GT_best_IoU_mean"),
            )
        )
    (output_root / "soft_merge_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v38 Phase F soft merge materializer.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--input-config", default=INPUT_CONFIG)
    parser.add_argument("--gt-path", default="data/scannet/gt")
    parser.add_argument("--output-root", default="outputs/audit/v38_soft_merge")
    parser.add_argument("--cuda-visible-devices", default="6")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
