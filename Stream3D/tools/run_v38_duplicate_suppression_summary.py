from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v38_oracle_attribution import (
    _candidate_oracle_stats,
    _class_agnostic_gt,
    _parse_metric_file,
    _quality_stats,
)


VARIANTS = {
    "F0_const_min100": "v38_d1_const_min100_probe5",
    "F2_iou_nms050": "v38_f2_iou_nms050_probe5",
    "F3_iou_nms070": "v38_f3_iou_nms070_probe5",
    "F6_min_ioc_nms050": "v38_f6_min_ioc_nms050_probe5",
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


def _metric_for_variant(root: Path, output_root: Path, variant: str, config: str) -> tuple[dict[str, float], str, int]:
    if variant == "F0_const_min100":
        metric_file = root / "outputs/audit/v38_object_materializer/eval/v38_d1_const_min100_probe5_class_agnostic.txt"
        return _parse_metric_file(metric_file), str(metric_file), 0
    metric_file = output_root / "eval" / f"{config}_class_agnostic.txt"
    exit_path = output_root / "eval" / f"{config}_evaluate.exit_code"
    return _parse_metric_file(metric_file), str(metric_file), int(exit_path.read_text(encoding="utf-8").strip())


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    if Path.cwd().resolve() != root:
        os.chdir(root)
    output_root = root / args.output_root
    scenes = _read_split(root / args.split)
    rows: list[dict[str, Any]] = []
    for variant, config in VARIANTS.items():
        for scene in scenes:
            pred_path = root / "data/prediction" / f"{config}_class_agnostic" / f"{scene}.npz"
            pre_path = root / "data/TMP" / config / f"{scene}_pre_points.npy"
            gt_path = root / args.gt_path / f"{scene}.txt"
            with np.load(pred_path) as pred:
                masks = np.asarray(pred["pred_masks"], dtype=bool)
                scores = np.asarray(pred["pred_score"], dtype=np.float32)
            raw_pre_points = np.load(pre_path).astype(np.int64)
            gt_eval = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
            stats = _candidate_oracle_stats(masks, scores, gt_eval, raw_pre_points)
            row = _quality_stats(scene, variant, masks, scores, gt_eval, raw_pre_points, stats)
            row["config"] = config
            rows.append(row)

    matrix: list[dict[str, Any]] = []
    for variant, config in VARIANTS.items():
        items = [row for row in rows if row["variant"] == variant]
        agg: dict[str, Any] = {"variant": variant, "config": config, "scene_count": len(items)}
        keys = sorted(
            key for row in items for key, value in row.items() if isinstance(value, (int, float)) and key != "scene"
        )
        for key in keys:
            vals = [float(row[key]) for row in items if isinstance(row.get(key), (int, float))]
            if vals:
                agg[key] = float(np.mean(vals))
        metrics, metric_file, exit_code = _metric_for_variant(root, output_root, variant, config)
        agg.update(
            {
                "AP": metrics["AP"],
                "AP50": metrics["AP50"],
                "AP25": metrics["AP25"],
                "eval_exit_code": exit_code,
                "metric_file": metric_file,
            }
        )
        matrix.append(agg)

    f0 = matrix[0]
    gate = {}
    for row in matrix[1:]:
        gate[row["variant"]] = {
            "num_predictions_after_le_300": bool(float(row.get("num_predictions", 1e9)) <= 300.0),
            "duplicate_rate_after_le_010": bool(float(row.get("duplicate_prediction_rate", 1.0)) <= 0.10),
            "conflict_rate_after_le_010": bool(float(row.get("mean_export_conflict_rate", 1.0)) <= 0.10),
            "AP_not_worse_than_F0_by_gt_001": bool(float(row["AP"]) >= float(f0["AP"]) - 0.01),
            "AP25_improves_or_stable": bool(float(row["AP25"]) >= float(f0["AP25"]) - 1e-12),
        }
    summary = {
        "phase": "v38_phaseF_duplicate_suppression",
        "input_config": "v38_d1_const_min100_probe5",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "variants": VARIANTS,
        "matrix": matrix,
        "scene_rows": rows,
        "phaseF_gate": gate,
    }
    _write_json(output_root / "duplicate_suppression_summary.json", summary)
    _write_csv(output_root / "duplicate_suppression_matrix.csv", matrix)
    _write_csv(output_root / "duplicate_suppression_scene_rows.csv", rows)
    lines = [
        "# Stream4D v38 Phase F Duplicate Suppression",
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
    (output_root / "duplicate_suppression_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize v38 Phase F duplicate suppression outputs.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--gt-path", default="data/scannet/gt")
    parser.add_argument("--output-root", default="outputs/audit/v38_duplicate_suppression")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
