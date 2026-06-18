from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v38_object_materializer import _load_trace_rows, _quality_scores
from tools.run_v38_oracle_attribution import (
    _candidate_oracle_stats,
    _class_agnostic_gt,
    _parse_metric_file,
    _quality_stats,
)


MIN_REGION_SIZE = 100
RAW_CONFIG = "v38_i4_sparse_export_trace_probe5"


VARIANTS: dict[str, dict[str, Any]] = {
    "E0_const_min100": {
        "config": "v38_e0_const_min100_probe5",
        "score_mode": "constant",
        "uses_gt": False,
        "description": "fixed min100 candidate set, constant score",
    },
    "E1_area_score_min100": {
        "config": "v38_e1_area_score_min100_probe5",
        "score_mode": "area",
        "uses_gt": False,
        "description": "fixed min100 candidate set, score by raw mask area",
    },
    "E2_log_area_score_min100": {
        "config": "v38_e2_log_area_score_min100_probe5",
        "score_mode": "log_area",
        "uses_gt": False,
        "description": "fixed min100 candidate set, score by normalized log area",
    },
    "E3_inverse_log_area_score_min100": {
        "config": "v38_e3_inverse_log_area_score_min100_probe5",
        "score_mode": "inverse_log_area",
        "uses_gt": False,
        "description": "fixed min100 candidate set, smaller masks score higher",
    },
    "E4_quality_score_min100": {
        "config": "v38_e4_quality_score_min100_probe5",
        "score_mode": "quality",
        "uses_gt": False,
        "description": "fixed min100 candidate set, non-GT object quality score",
    },
    "E5_no_temporal_quality_score_min100": {
        "config": "v38_e5_no_temporal_quality_score_min100_probe5",
        "score_mode": "no_temporal_quality",
        "uses_gt": False,
        "description": "fixed min100 candidate set, quality score without temporal-span term",
    },
    "E6_shuffled_quality_control_min100": {
        "config": "v38_e6_shuffled_quality_control_min100_probe5",
        "score_mode": "shuffled_quality",
        "uses_gt": False,
        "description": "fixed min100 candidate set, shuffled quality-score control",
    },
    "E7_oracle_iou_score_min100": {
        "config": "v38_e7_oracle_iou_score_min100_probe5",
        "score_mode": "oracle_iou",
        "uses_gt": True,
        "description": "fixed min100 candidate set, oracle score=max GT IoU; diagnostic only",
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
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def _score_values(
    mode: str,
    scene: str,
    masks: np.ndarray,
    trace_rows: list[dict[str, Any]],
    oracle_stats: dict[str, Any],
    seed: int,
) -> np.ndarray:
    areas = masks.sum(axis=0).astype(np.float64)
    if mode == "constant":
        return np.ones((masks.shape[1],), dtype=np.float32)
    if mode == "area":
        return areas.astype(np.float32, copy=False)
    if mode == "log_area":
        return _normalize(np.log1p(np.maximum(areas, 0.0))).astype(np.float32, copy=False)
    if mode == "inverse_log_area":
        return (1.0 - _normalize(np.log1p(np.maximum(areas, 0.0)))).astype(np.float32, copy=False)
    if mode in {"quality", "no_temporal_quality", "shuffled_quality"}:
        return _quality_scores(scene, masks, trace_rows, mode, seed)
    if mode == "oracle_iou":
        return np.asarray(oracle_stats["max_iou"], dtype=np.float32)
    raise ValueError(f"Unsupported score mode: {mode}")


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


def _write_manifest(root: Path, variant: str, meta: dict[str, Any], source_config: str) -> None:
    config = str(meta["config"])
    uses_gt = bool(meta["uses_gt"])
    manifest = build_prediction_manifest(
        root=root,
        output_config=config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=uses_gt,
        gt_usage="oracle_iou_score_diagnostic" if uses_gt else "none",
        source_configs=[source_config],
        pre_points_policy=f"copied_raw_support:{source_config}",
        support_policy=f"v38_phaseE_score_diagnostics:{variant}:{meta['score_mode']}",
        notes=f"v38 Phase E score diagnostic {variant}: {meta['description']}. Diagnostic-only; forbidden for method table.",
        extra={
            "phase": "v38_phaseE_score_diagnostics",
            "variant": variant,
            "uses_gt_for_prediction": uses_gt,
            "uses_gt_for_diagnostic": uses_gt,
            "forbidden_for_method_table": True,
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "alignment_source": "scannet_rgbd_pose_mesh_export_bridge",
            "alignment_used_for_prediction": True,
        },
    )
    write_prediction_manifest(config, manifest, root=root, pred_suffix="class_agnostic")


def _generate_variants(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    scenes = _read_split(root / args.split)
    for variant, meta in VARIANTS.items():
        _write_manifest(root, variant, meta, args.input_config)

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        pred_path = root / "data/prediction" / f"{args.input_config}_class_agnostic" / f"{scene}.npz"
        pre_path = root / "data/TMP" / args.input_config / f"{scene}_pre_points.npy"
        gt_path = root / args.gt_path / f"{scene}.txt"
        with np.load(pred_path) as pred:
            raw_masks = np.asarray(pred["pred_masks"], dtype=bool)
            raw_classes = np.asarray(pred["pred_classes"], dtype=np.int32)
        raw_pre_points = np.load(pre_path).astype(np.int64)
        trace_rows_all = _load_trace_rows(root, Path(args.trace_root), scene, raw_masks.shape[1])
        areas = raw_masks.sum(axis=0).astype(np.int64)
        keep = areas >= MIN_REGION_SIZE
        masks = raw_masks[:, keep]
        classes = raw_classes[keep]
        trace_rows = [row for idx, row in enumerate(trace_rows_all) if bool(keep[idx])]
        gt_eval = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
        oracle_stats = _candidate_oracle_stats(masks, np.ones((masks.shape[1],), dtype=np.float32), gt_eval, raw_pre_points)
        for variant, meta in VARIANTS.items():
            scores = _score_values(str(meta["score_mode"]), scene, masks, trace_rows, oracle_stats, int(args.seed))
            config = str(meta["config"])
            _write_prediction(root, config, scene, masks, scores, classes, raw_pre_points)
            stats = _candidate_oracle_stats(masks, scores, gt_eval, raw_pre_points)
            row = _quality_stats(scene, variant, masks, scores, gt_eval, raw_pre_points, stats)
            row.update(
                {
                    "config": config,
                    "num_candidates": int(masks.shape[1]),
                    "score_mode": str(meta["score_mode"]),
                    "uses_gt_for_prediction": bool(meta["uses_gt"]),
                    "dropped_by_min_area": int(np.count_nonzero(~keep)),
                }
            )
            rows.append(row)
        del raw_masks, raw_classes, masks, classes
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
    metrics = _parse_metric_file(metric_file) if metric_file.exists() else {}
    return {
        "config": config,
        "command": " ".join(cmd),
        "exit_code": int(proc.returncode),
        "metric_file": str(metric_file),
        "log_path": str(log_path),
        "metrics": metrics,
    }


def _aggregate_rows(rows: list[dict[str, Any]], evals: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in VARIANTS:
        items = [row for row in rows if str(row["variant"]) == variant]
        numeric_keys = sorted(
            key for row in items for key, value in row.items() if isinstance(value, (int, float)) and key != "scene"
        )
        agg: dict[str, Any] = {"variant": variant, "scene_count": len(items)}
        for key in numeric_keys:
            vals = [float(row[key]) for row in items if isinstance(row.get(key), (int, float))]
            if vals:
                agg[key] = float(np.mean(vals))
        eval_row = evals.get(variant)
        if eval_row is not None:
            agg.update(
                {
                    "AP": eval_row.get("metrics", {}).get("AP"),
                    "AP50": eval_row.get("metrics", {}).get("AP50"),
                    "AP25": eval_row.get("metrics", {}).get("AP25"),
                    "eval_exit_code": eval_row.get("exit_code"),
                    "metric_file": eval_row.get("metric_file"),
                }
            )
        out.append(agg)
    return out


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stream4D v38 Phase E Score Diagnostics",
        "",
        "| variant | AP | AP50 | AP25 | #cand | score rho | score pearson | oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["matrix"]:
        lines.append(
            "| {variant} | {AP} | {AP50} | {AP25} | {cand} | {rho} | {pearson} | {oracle} |".format(
                variant=row.get("variant"),
                AP=row.get("AP"),
                AP50=row.get("AP50"),
                AP25=row.get("AP25"),
                cand=row.get("num_candidates"),
                rho=row.get("score_IoU_spearman"),
                pearson=row.get("score_IoU_pearson"),
                oracle=row.get("uses_gt_for_prediction"),
            )
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "E7 uses GT IoU as an oracle diagnostic score and is forbidden for method tables. E0-E6 do not use GT for prediction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    if Path.cwd().resolve() != root:
        os.chdir(root)
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _generate_variants(args, root)
    evals = {variant: _run_eval(args, root, str(meta["config"]), output_root) for variant, meta in VARIANTS.items()}
    matrix = _aggregate_rows(rows, evals)
    non_oracle = [row for row in matrix if not bool(row.get("uses_gt_for_prediction", False))]
    best_non_oracle = max(non_oracle, key=lambda row: float(row.get("AP") or -1.0))
    oracle = next(row for row in matrix if row["variant"] == "E7_oracle_iou_score_min100")
    constant = next(row for row in matrix if row["variant"] == "E0_const_min100")
    summary = {
        "phase": "v38_phaseE_score_diagnostics",
        "input_config": args.input_config,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": True,
        "matrix": matrix,
        "scene_rows": rows,
        "evals": evals,
        "best_non_oracle_variant": best_non_oracle,
        "oracle_variant": oracle,
        "constant_variant": constant,
        "phaseE_gate": {
            "best_non_oracle_beats_constant": bool(float(best_non_oracle.get("AP") or 0.0) > float(constant.get("AP") or 0.0)),
            "oracle_score_AP_ge_012": bool(float(oracle.get("AP") or 0.0) >= 0.12),
            "best_non_oracle_AP_ge_012": bool(float(best_non_oracle.get("AP") or 0.0) >= 0.12),
        },
        "notes": [
            "All variants share the same min100 candidate masks; only scores change.",
            "E7 uses GT IoU score for diagnostic upper-bound only.",
            "E0-E6 do not use GT for prediction.",
        ],
    }
    _write_json(output_root / "score_diagnostics_summary.json", summary)
    _write_csv(output_root / "score_diagnostics_matrix.csv", matrix)
    _write_csv(output_root / "score_diagnostics_scene_rows.csv", rows)
    _write_markdown(output_root / "score_diagnostics_summary.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v38 Phase E score diagnostics on fixed min100 candidates.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--input-config", default=RAW_CONFIG)
    parser.add_argument("--trace-root", default="outputs/audit/v38_export_trace")
    parser.add_argument("--output-root", default="outputs/audit/v38_score_diagnostics")
    parser.add_argument("--gt-path", default="data/scannet/gt")
    parser.add_argument("--seed", type=int, default=3802)
    parser.add_argument("--cuda-visible-devices", default="6")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
