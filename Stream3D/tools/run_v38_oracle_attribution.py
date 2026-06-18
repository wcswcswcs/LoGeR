from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.constants import SCANNET_IDS
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split


MIN_REGION_SIZE = 100
THRESHOLDS = (0.25, 0.50, 0.75)
RAW_CONFIG = "v38_i4_sparse_export_trace_probe5"


VARIANTS = {
    "C1_const_min100": {
        "config": "v38_c1_const_min100_probe5",
        "uses_gt": False,
        "description": "same masks as C0, area >= 100, constant score",
    },
    "C2_oracle_score": {
        "config": "v38_oracle_c2_score_probe5",
        "uses_gt": True,
        "description": "same masks as C0, score = max GT IoU",
    },
    "C3_oracle_best_per_gt": {
        "config": "v38_oracle_c3_best_per_gt_probe5",
        "uses_gt": True,
        "description": "keep one best current candidate per GT instance",
    },
    "C5_oracle_vertex_owner": {
        "config": "v38_oracle_c5_vertex_owner_probe5",
        "uses_gt": True,
        "description": "resolve overlapping vertices with GT-aware ownership among current objects",
    },
    "C6_oracle_gt_mask_by_object": {
        "config": "v38_oracle_c6_gt_mask_by_object_probe5",
        "uses_gt": True,
        "description": "replace each object mask by its best GT instance mask inside raw support",
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


def _parse_metric_file(path: Path) -> dict[str, float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty metric file: {path}")
    parts = lines[-1].split(",")
    if len(parts) != 3:
        raise ValueError(f"Could not parse final AP row from {path}: {lines[-1]}")
    return {"AP": float(parts[0]), "AP50": float(parts[1]), "AP25": float(parts[2])}


def _class_agnostic_gt(gt_ids: np.ndarray) -> np.ndarray:
    return gt_ids.astype(np.int64) % 1000 + int(SCANNET_IDS[0]) * 1000


def _valid_gt_areas(gt_eval: np.ndarray, support: np.ndarray) -> dict[int, int]:
    support_gt = gt_eval[support]
    vals, counts = np.unique(support_gt, return_counts=True)
    out: dict[int, int] = {}
    for value, count in zip(vals.tolist(), counts.tolist()):
        value_i = int(value)
        count_i = int(count)
        if value_i >= 1000 and count_i >= MIN_REGION_SIZE:
            out[value_i] = count_i
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.shape[0] < 2 or y.shape[0] < 2:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x.astype(np.float64), y.astype(np.float64))[0, 1])


def _candidate_oracle_stats(
    masks: np.ndarray,
    scores: np.ndarray,
    gt_eval: np.ndarray,
    support: np.ndarray,
) -> dict[str, Any]:
    gt_areas = _valid_gt_areas(gt_eval, support)
    valid_gt_ids = sorted(gt_areas)
    per_gt_best = {gt_id: 0.0 for gt_id in valid_gt_ids}
    per_gt_best_pred = {gt_id: -1 for gt_id in valid_gt_ids}
    best_gt = np.full((masks.shape[1],), -1, dtype=np.int64)
    max_iou = np.zeros((masks.shape[1],), dtype=np.float32)
    areas = masks.sum(axis=0).astype(np.int64)

    for pred_idx in range(masks.shape[1]):
        ids = np.flatnonzero(masks[:, pred_idx])
        area = int(ids.shape[0])
        if area <= 0:
            continue
        vals, counts = np.unique(gt_eval[ids], return_counts=True)
        for value, inter in zip(vals.tolist(), counts.tolist()):
            gt_id = int(value)
            if gt_id not in gt_areas:
                continue
            union = area + int(gt_areas[gt_id]) - int(inter)
            iou = float(inter / max(union, 1))
            if iou > float(max_iou[pred_idx]):
                max_iou[pred_idx] = iou
                best_gt[pred_idx] = gt_id
            if iou > per_gt_best[gt_id]:
                per_gt_best[gt_id] = iou
                per_gt_best_pred[gt_id] = int(pred_idx)

    best_values = np.asarray([per_gt_best[gt_id] for gt_id in valid_gt_ids], dtype=np.float64)
    valid_score_mask = areas >= MIN_REGION_SIZE
    score_iou_pearson = _corr(scores[valid_score_mask].astype(np.float64), max_iou[valid_score_mask].astype(np.float64))
    score_iou_spearman = _corr(
        _rankdata(scores[valid_score_mask].astype(np.float64)),
        _rankdata(max_iou[valid_score_mask].astype(np.float64)),
    ) if int(np.count_nonzero(valid_score_mask)) >= 2 else None
    return {
        "gt_areas": gt_areas,
        "best_gt": best_gt,
        "max_iou": max_iou,
        "per_gt_best": per_gt_best,
        "per_gt_best_pred": per_gt_best_pred,
        "score_IoU_pearson": score_iou_pearson,
        "score_IoU_spearman": score_iou_spearman,
        "per_GT_best_IoU_mean": float(np.mean(best_values)) if best_values.size else 0.0,
        "per_GT_best_IoU_p50": float(np.median(best_values)) if best_values.size else 0.0,
        "per_GT_best_IoU_ge_25": float(np.mean(best_values >= 0.25)) if best_values.size else 0.0,
        "per_GT_best_IoU_ge_50": float(np.mean(best_values >= 0.50)) if best_values.size else 0.0,
        "per_GT_best_IoU_ge_75": float(np.mean(best_values >= 0.75)) if best_values.size else 0.0,
    }


def _exact_duplicate_rate(masks: np.ndarray) -> float:
    seen: dict[bytes, int] = {}
    duplicate = 0
    for idx in range(masks.shape[1]):
        ids = np.flatnonzero(masks[:, idx]).astype(np.int64)
        key = ids.tobytes()
        if key in seen:
            duplicate += 1
        else:
            seen[key] = idx
    return float(duplicate / max(masks.shape[1], 1))


def _quality_stats(
    scene: str,
    variant: str,
    masks: np.ndarray,
    scores: np.ndarray,
    gt_eval: np.ndarray,
    support: np.ndarray,
    oracle_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if oracle_stats is None:
        oracle_stats = _candidate_oracle_stats(masks, scores, gt_eval, support)
    areas = masks.sum(axis=0).astype(np.int64)
    union = np.any(masks, axis=1) if masks.shape[1] else np.zeros((masks.shape[0],), dtype=bool)
    union_count = int(np.count_nonzero(union))
    owner_counts = masks.sum(axis=1).astype(np.int32) if masks.shape[1] else np.zeros((masks.shape[0],), dtype=np.int32)
    conflict = int(np.count_nonzero(owner_counts > 1))
    gt_ids_all = set(int(v) for v in np.unique(gt_eval) if int(v) >= 1000)
    gt_ids_covered = set(int(v) for v in np.unique(gt_eval[union]) if int(v) >= 1000) if union_count else set()
    return {
        "scene": scene,
        "variant": variant,
        "num_predictions": int(masks.shape[1]),
        "pre_percent": float(union_count / max(masks.shape[0], 1)),
        "union_percent": float(union_count / max(masks.shape[0], 1)),
        "mesh_coverage": float(union_count / max(masks.shape[0], 1)),
        "covered_GT_instance_ratio": float(len(gt_ids_covered) / max(len(gt_ids_all), 1)),
        "mean_num_predictions": int(masks.shape[1]),
        "duplicate_prediction_rate": _exact_duplicate_rate(masks),
        "mean_export_conflict_rate": float(conflict / max(union_count, 1)),
        "mean_vertices_per_prediction": float(np.mean(areas)) if areas.size else 0.0,
        "score_IoU_spearman": oracle_stats.get("score_IoU_spearman"),
        "score_IoU_pearson": oracle_stats.get("score_IoU_pearson"),
        "per_GT_best_IoU_mean": oracle_stats.get("per_GT_best_IoU_mean"),
        "per_GT_best_IoU_p50": oracle_stats.get("per_GT_best_IoU_p50"),
        "per_GT_best_IoU_ge_25": oracle_stats.get("per_GT_best_IoU_ge_25"),
        "per_GT_best_IoU_ge_50": oracle_stats.get("per_GT_best_IoU_ge_50"),
        "per_GT_best_IoU_ge_75": oracle_stats.get("per_GT_best_IoU_ge_75"),
    }


def _write_prediction(
    root: Path,
    config: str,
    scene: str,
    masks: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    raw_pre_points: np.ndarray,
) -> None:
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


def _write_manifest(root: Path, variant: str, config: str, source_config: str) -> None:
    meta = VARIANTS[variant]
    uses_gt = bool(meta["uses_gt"])
    manifest = build_prediction_manifest(
        root=root,
        output_config=config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=uses_gt,
        gt_usage=("oracle_attribution_diagnostic" if uses_gt else "none"),
        source_configs=[source_config],
        pre_points_policy=f"copied_raw_support:{source_config}",
        support_policy=f"v38_oracle_attribution:{variant}",
        notes=f"v38 Phase C {variant}: {meta['description']}. Diagnostic-only; forbidden for method table.",
        extra={
            "phase": "v38_phaseC_oracle_attribution",
            "variant": variant,
            "uses_gt_for_prediction": uses_gt,
            "uses_gt_for_diagnostic": uses_gt,
            "forbidden_for_method_table": True,
            "is_oracle_attribution": uses_gt,
        },
    )
    write_prediction_manifest(config, manifest, root=root, pred_suffix="class_agnostic")


def _copy_raw_manifest_alias(root: Path, source_config: str) -> None:
    manifest_path = root / "data/prediction" / f"{source_config}_class_agnostic" / "config_manifest.json"
    if manifest_path.exists():
        return
    manifest = build_prediction_manifest(
        root=root,
        output_config=source_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=["outputs/audit/v38_export_trace/export_trace_summary.json"],
        pre_points_policy="copied_from_v38_export_trace",
        support_policy="v38_export_trace",
        notes="v38 Phase C raw trace alias manifest.",
        extra={
            "phase": "v38_phaseC_C0_raw_trace",
            "forbidden_for_method_table": True,
        },
    )
    write_prediction_manifest(source_config, manifest, root=root, pred_suffix="class_agnostic")


def _variant_masks(
    variant: str,
    masks: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    gt_eval: np.ndarray,
    support_mask: np.ndarray,
    raw_stats: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    areas = masks.sum(axis=0).astype(np.int64)
    best_gt = np.asarray(raw_stats["best_gt"], dtype=np.int64)
    max_iou = np.asarray(raw_stats["max_iou"], dtype=np.float32)
    if variant == "C1_const_min100":
        keep = areas >= MIN_REGION_SIZE
        return masks[:, keep], np.ones((int(np.count_nonzero(keep)),), dtype=np.float32), classes[keep]
    if variant == "C2_oracle_score":
        return masks.copy(), max_iou.astype(np.float32, copy=True), classes.copy()
    if variant == "C3_oracle_best_per_gt":
        keep_indices = sorted(set(int(v) for v in raw_stats["per_gt_best_pred"].values() if int(v) >= 0))
        keep = np.asarray(keep_indices, dtype=np.int64)
        if keep.size == 0:
            return masks[:, :0], np.zeros((0,), dtype=np.float32), classes[:0]
        return masks[:, keep], max_iou[keep].astype(np.float32), classes[keep]
    if variant == "C5_oracle_vertex_owner":
        rows, cols = np.nonzero(masks)
        if rows.size == 0:
            return masks.copy(), max_iou.astype(np.float32, copy=True), classes.copy()
        match = ((gt_eval[rows] == best_gt[cols]) & (gt_eval[rows] >= 1000)).astype(np.int8)
        order = np.lexsort((cols, -scores[cols], -max_iou[cols], -match, rows))
        rows_o = rows[order]
        cols_o = cols[order]
        first = np.ones((rows_o.shape[0],), dtype=bool)
        first[1:] = rows_o[1:] != rows_o[:-1]
        out = np.zeros_like(masks, dtype=bool)
        out[rows_o[first], cols_o[first]] = True
        return out, max_iou.astype(np.float32, copy=True), classes.copy()
    if variant == "C6_oracle_gt_mask_by_object":
        out = np.zeros_like(masks, dtype=bool)
        valid = np.flatnonzero((best_gt >= 1000) & (max_iou > 0.0))
        for pred_idx in valid.tolist():
            out[:, int(pred_idx)] = (gt_eval == int(best_gt[int(pred_idx)])) & support_mask
        return out, max_iou.astype(np.float32, copy=True), classes.copy()
    raise ValueError(f"Unsupported variant: {variant}")


def _generate_variants(args: argparse.Namespace, root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    scenes = _read_split(root / args.split)
    source_config = args.input_config
    _copy_raw_manifest_alias(root, source_config)
    for variant, meta in VARIANTS.items():
        _write_manifest(root, variant, str(meta["config"]), source_config)

    rows: list[dict[str, Any]] = []
    variant_scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene in scenes:
        pred_path = root / "data/prediction" / f"{source_config}_class_agnostic" / f"{scene}.npz"
        pre_path = root / "data/TMP" / source_config / f"{scene}_pre_points.npy"
        gt_path = root / args.gt_path / f"{scene}.txt"
        with np.load(pred_path) as pred:
            masks = np.asarray(pred["pred_masks"], dtype=bool)
            scores = np.asarray(pred["pred_score"], dtype=np.float32)
            classes = np.asarray(pred["pred_classes"], dtype=np.int32)
        raw_pre_points = np.load(pre_path).astype(np.int64)
        gt_eval = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
        support_mask = np.zeros((gt_eval.shape[0],), dtype=bool)
        support_mask[raw_pre_points] = True
        raw_stats = _candidate_oracle_stats(masks, scores, gt_eval, raw_pre_points)
        raw_row = _quality_stats(scene, "C0_raw_trace", masks, scores, gt_eval, raw_pre_points, raw_stats)
        rows.append(raw_row)
        variant_scene_rows["C0_raw_trace"].append(raw_row)
        c4_row = {
            **raw_row,
            "variant": "C4_one_object_one_main_mask",
            "equivalent_to": "C0_raw_trace",
            "equivalence_reason": "Phase B trace showed current exporter already emits one kept prediction mask per Stream4D object candidate.",
        }
        rows.append(c4_row)
        variant_scene_rows["C4_one_object_one_main_mask"].append(c4_row)
        for variant, meta in VARIANTS.items():
            out_masks, out_scores, out_classes = _variant_masks(variant, masks, scores, classes, gt_eval, support_mask, raw_stats)
            config = str(meta["config"])
            _write_prediction(root, config, scene, out_masks, out_scores, out_classes, raw_pre_points)
            stats = _candidate_oracle_stats(out_masks, out_scores, gt_eval, raw_pre_points)
            row = _quality_stats(scene, variant, out_masks, out_scores, gt_eval, raw_pre_points, stats)
            rows.append(row)
            variant_scene_rows[variant].append(row)
            del out_masks, out_scores, out_classes
        del masks, scores, classes
    return rows, variant_scene_rows


def _run_eval(args: argparse.Namespace, root: Path, config: str, output_root: Path, allow_oracle: bool) -> dict[str, Any]:
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
    ]
    if allow_oracle:
        cmd.append("--allow-oracle-eval")
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


def _same_support_row(root: Path) -> dict[str, Any]:
    metric_file = root / "data/evaluation/scannet/scannet_on_stream4d_32f_probe5_class_agnostic.txt"
    metrics = _parse_metric_file(metric_file)
    return {
        "config": "scannet_on_stream4d_32f_probe5",
        "variant": "C7_same_support_stream3d",
        "exit_code": 0,
        "metric_file": str(metric_file),
        "metrics": metrics,
        "source": "existing v37 same-support Stream3D metric file",
    }


def _aggregate_rows(rows: list[dict[str, Any]], evals: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    variants = sorted(set(str(row["variant"]) for row in rows))
    out: list[dict[str, Any]] = []
    for variant in variants:
        items = [row for row in rows if str(row["variant"]) == variant]
        numeric_keys = sorted(
            key for row in items for key, value in row.items() if isinstance(value, (int, float)) and key != "scene"
        )
        agg: dict[str, Any] = {
            "variant": variant,
            "scene_count": len(items),
        }
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
        "# Stream4D v38 Phase C Oracle Attribution Matrix",
        "",
        "| variant | AP | AP50 | AP25 | pre% | conflict | #pred | dup | best IoU mean | GT>=25/50/75 | score rho |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["matrix"]:
        lines.append(
            "| {variant} | {AP} | {AP50} | {AP25} | {pre} | {conflict} | {pred} | {dup} | {iou} | {ge25}/{ge50}/{ge75} | {rho} |".format(
                variant=row.get("variant"),
                AP=row.get("AP"),
                AP50=row.get("AP50"),
                AP25=row.get("AP25"),
                pre=row.get("pre_percent"),
                conflict=row.get("mean_export_conflict_rate"),
                pred=row.get("num_predictions"),
                dup=row.get("duplicate_prediction_rate"),
                iou=row.get("per_GT_best_IoU_mean"),
                ge25=row.get("per_GT_best_IoU_ge_25"),
                ge50=row.get("per_GT_best_IoU_ge_50"),
                ge75=row.get("per_GT_best_IoU_ge_75"),
                rho=row.get("score_IoU_spearman"),
            )
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "Rows containing `oracle` use GT for diagnostic attribution only. They are not method results and are forbidden for method tables.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    if Path.cwd().resolve() != root:
        os.chdir(root)
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows, _variant_scene_rows = _generate_variants(args, root)
    evals: dict[str, dict[str, Any]] = {
        "C0_raw_trace": _run_eval(args, root, args.input_config, output_root, allow_oracle=True),
    }
    evals["C4_one_object_one_main_mask"] = {
        **evals["C0_raw_trace"],
        "variant": "C4_one_object_one_main_mask",
        "equivalent_to": "C0_raw_trace",
        "equivalence_reason": "Phase B trace showed current exporter already emits one kept prediction mask per Stream4D object candidate.",
    }
    for variant, meta in VARIANTS.items():
        evals[variant] = _run_eval(args, root, str(meta["config"]), output_root, allow_oracle=True)
    evals["C7_same_support_stream3d"] = _same_support_row(root)
    matrix = _aggregate_rows(rows, evals)
    matrix.append(
        {
            "variant": "C7_same_support_stream3d",
            "scene_count": len(_read_split(root / args.split)),
            "AP": evals["C7_same_support_stream3d"]["metrics"].get("AP"),
            "AP50": evals["C7_same_support_stream3d"]["metrics"].get("AP50"),
            "AP25": evals["C7_same_support_stream3d"]["metrics"].get("AP25"),
            "eval_exit_code": 0,
            "metric_file": evals["C7_same_support_stream3d"]["metric_file"],
        }
    )
    summary = {
        "phase": "v38_phaseC_oracle_attribution",
        "input_config": args.input_config,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "matrix": matrix,
        "scene_rows": rows,
        "evals": evals,
        "notes": [
            "Oracle rows use GT for diagnostic attribution only.",
            "All generated variants copy the raw C0 pre_points support for fair same-support comparison.",
            "C4 one-object-one-main-mask is recorded as an explicit C0-equivalent row because Phase B showed one kept candidate per Stream4D object.",
        ],
    }
    _write_json(output_root / "oracle_attribution_summary.json", summary)
    _write_csv(output_root / "oracle_attribution_matrix.csv", matrix)
    _write_csv(output_root / "oracle_attribution_scene_rows.csv", rows)
    _write_markdown(output_root / "oracle_attribution_summary.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v38 Phase C oracle attribution matrix for traced AP export.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--input-config", default=RAW_CONFIG)
    parser.add_argument("--output-root", default="outputs/audit/v38_oracle_attribution")
    parser.add_argument("--gt-path", default="data/scannet/gt")
    parser.add_argument("--cuda-visible-devices", default="6")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
