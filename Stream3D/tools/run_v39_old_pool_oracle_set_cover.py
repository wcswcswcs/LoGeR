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
    MIN_REGION_SIZE,
    _candidate_oracle_stats,
    _class_agnostic_gt,
    _parse_metric_file,
    _quality_stats,
    _valid_gt_areas,
)


RAW_CONFIG = "v38_i4_sparse_export_trace_probe5"
SCENE0081 = "scene0081_01"


VARIANTS: dict[str, dict[str, Any]] = {
    "B0_raw_all_candidates": {
        "config": "v39_b0_raw_all_candidates_probe5",
        "description": "copy the full v38 traced old candidate pool",
    },
    "B1_per_gt_best_candidate": {
        "config": "v39_b1_per_gt_best_candidate_probe5",
        "description": "GT oracle: keep one highest-IoU old-pool candidate per GT instance",
    },
    "B2_per_gt_top2_candidates": {
        "config": "v39_b2_per_gt_top2_candidates_probe5",
        "description": "GT oracle: keep top-2 old-pool candidates per GT instance",
    },
    "B3_min_duplicate_set_cover": {
        "config": "v39_b3_min_duplicate_set_cover_probe5",
        "description": "GT oracle greedy set cover over IoU>=0.25 candidates, minimizing duplicate GT coverage",
    },
    "B4_ap_aware_set_cover": {
        "config": "v39_b4_ap_aware_set_cover_probe5",
        "description": "GT oracle AP-aware cover: one high-IoU candidate per GT with IoU scores",
    },
    "B5_conflict_aware_set_cover": {
        "config": "v39_b5_conflict_aware_set_cover_probe5",
        "description": "GT oracle conflict-aware cover: trade candidate IoU against raw overlap/conflict trace penalty",
    },
    "B6_object_count_regularized": {
        "config": "v39_b6_object_count_regularized_probe5",
        "description": "GT oracle object-count-regularized cover: compact one-candidate-per-GT support with low-IoU pruning",
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
        writer.writerows([{key: _csv_value(row.get(key)) for key in keys} for row in rows])


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _load_trace_penalties(root: Path, trace_root: Path, scene: str, count: int) -> np.ndarray:
    path = root / trace_root / "scenes" / scene / "prediction_trace_rows.csv"
    penalties = np.zeros((count,), dtype=np.float32)
    if not path.exists():
        return penalties
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            idx = int(row["prediction_index"])
            if 0 <= idx < count:
                overlap = float(row.get("overlap_with_other_predictions") or 0.0)
                conflict = float(row.get("conflict_vertex_count") or 0.0)
                vertices = max(float(row.get("vertex_count") or 0.0), 1.0)
                penalties[idx] = float(0.5 * overlap + 0.5 * min(conflict / vertices, 1.0))
    return penalties


def _candidate_gt_iou(
    masks: np.ndarray,
    gt_eval: np.ndarray,
    support: np.ndarray,
) -> tuple[dict[int, list[tuple[float, int]]], np.ndarray, np.ndarray]:
    gt_areas = _valid_gt_areas(gt_eval, support)
    per_gt: dict[int, list[tuple[float, int]]] = {gt_id: [] for gt_id in gt_areas}
    max_iou = np.zeros((masks.shape[1],), dtype=np.float32)
    best_gt = np.full((masks.shape[1],), -1, dtype=np.int64)
    for pred_idx in range(masks.shape[1]):
        ids = np.flatnonzero(masks[:, pred_idx])
        area = int(ids.shape[0])
        if area < MIN_REGION_SIZE:
            continue
        vals, counts = np.unique(gt_eval[ids], return_counts=True)
        for value, inter in zip(vals.tolist(), counts.tolist()):
            gt_id = int(value)
            if gt_id not in gt_areas:
                continue
            union = area + int(gt_areas[gt_id]) - int(inter)
            iou = float(inter / max(union, 1))
            if iou > max_iou[pred_idx]:
                max_iou[pred_idx] = iou
                best_gt[pred_idx] = gt_id
            if iou > 0.0:
                per_gt[gt_id].append((iou, int(pred_idx)))
    for gt_id in per_gt:
        per_gt[gt_id].sort(key=lambda item: (-item[0], item[1]))
    return per_gt, max_iou, best_gt


def _unique(indices: list[int]) -> np.ndarray:
    return np.asarray(sorted(set(int(idx) for idx in indices)), dtype=np.int64)


def _greedy_cover(per_gt: dict[int, list[tuple[float, int]]], threshold: float = 0.25) -> np.ndarray:
    candidate_to_gts: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for gt_id, pairs in per_gt.items():
        for iou, pred_idx in pairs:
            if iou >= threshold:
                candidate_to_gts[pred_idx].append((gt_id, iou))
    uncovered = set(per_gt)
    selected: list[int] = []
    while uncovered:
        best_idx = -1
        best_gain = -1.0
        for pred_idx, covers in candidate_to_gts.items():
            gain = sum(iou for gt_id, iou in covers if gt_id in uncovered)
            if gain > best_gain or (gain == best_gain and pred_idx < best_idx):
                best_gain = gain
                best_idx = int(pred_idx)
        if best_idx < 0 or best_gain <= 0.0:
            break
        selected.append(best_idx)
        for gt_id, _iou in candidate_to_gts[best_idx]:
            uncovered.discard(gt_id)
        candidate_to_gts.pop(best_idx, None)
    for gt_id in sorted(uncovered):
        if per_gt[gt_id]:
            selected.append(int(per_gt[gt_id][0][1]))
    return _unique(selected)


def _select_indices(
    variant: str,
    masks: np.ndarray,
    per_gt: dict[int, list[tuple[float, int]]],
    max_iou: np.ndarray,
    trace_penalty: np.ndarray,
) -> np.ndarray:
    if variant == "B0_raw_all_candidates":
        return np.arange(masks.shape[1], dtype=np.int64)
    if variant in {"B1_per_gt_best_candidate", "B4_ap_aware_set_cover"}:
        return _unique([pairs[0][1] for pairs in per_gt.values() if pairs])
    if variant == "B2_per_gt_top2_candidates":
        out: list[int] = []
        for pairs in per_gt.values():
            out.extend(pred_idx for _iou, pred_idx in pairs[:2])
        return _unique(out)
    if variant == "B3_min_duplicate_set_cover":
        return _greedy_cover(per_gt, threshold=0.25)
    if variant == "B5_conflict_aware_set_cover":
        out = []
        for pairs in per_gt.values():
            if not pairs:
                continue
            best = max(pairs[:25], key=lambda item: (item[0] - 0.20 * float(trace_penalty[item[1]]), item[0], -item[1]))
            out.append(int(best[1]))
        return _unique(out)
    if variant == "B6_object_count_regularized":
        out = []
        for pairs in per_gt.values():
            if not pairs:
                continue
            best_iou, best_idx = pairs[0]
            if best_iou >= 0.10:
                out.append(int(best_idx))
        return _unique(out)
    raise ValueError(f"Unsupported variant: {variant}")


def _multiplicity_metrics(masks: np.ndarray, gt_eval: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    per_gt, _max_iou, _best_gt = _candidate_gt_iou(masks, gt_eval, support)
    counts = np.asarray([sum(1 for iou, _idx in pairs if iou >= 0.25) for pairs in per_gt.values()], dtype=np.float64)
    if counts.size == 0:
        return {
            "candidate_multiplicity_per_GT_mean": 0.0,
            "candidate_multiplicity_per_GT_p90": 0.0,
            "gt_count": 0,
        }
    return {
        "candidate_multiplicity_per_GT_mean": float(np.mean(counts)),
        "candidate_multiplicity_per_GT_p90": float(np.percentile(counts, 90)),
        "gt_count": int(counts.size),
    }


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


def _write_manifest(root: Path, variant: str, config: str, input_config: str, scene_only: bool = False) -> None:
    meta = VARIANTS[variant]
    manifest = build_prediction_manifest(
        root=root,
        output_config=config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=True,
        gt_usage="v39_old_pool_oracle_set_cover",
        source_configs=[input_config, "outputs/audit/v38_export_trace/export_trace_summary.json"],
        pre_points_policy=f"copied_raw_support:{input_config}",
        support_policy=f"v39_phaseB_old_pool_oracle_set_cover:{variant}",
        notes=f"v39 Phase B {variant}: {meta['description']}. Diagnostic-only oracle; forbidden for method table.",
        extra={
            "phase": "v39_phaseB_old_pool_oracle_set_cover",
            "variant": variant,
            "scene_only": bool(scene_only),
            "forbidden_for_method_table": True,
            "uses_gt_for_prediction": True,
            "uses_gt_for_diagnostic": True,
            "uses_gt_for_diagnostic_labels": True,
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "uses_eval_sim3_for_prediction": False,
            "uses_d4rt_self_sim3": False,
            "uses_frozen_visual_backbone": False,
            "visual_backbone_name": None,
            "mask_source": "v38_old_candidate_pool",
            "object_birth_source": "old_candidate_pool_oracle_diagnostic",
            "d4rt_role": "old_v37_v38_candidate_source_diagnostic",
            "geometry_field": "scannet_mesh_eval_bridge",
            "coordinate_frame": "scannet_eval_mesh",
            "alignment_source": "scannet_rgbd_pose_mesh_export_bridge",
            "alignment_used_for_prediction": True,
        },
    )
    write_prediction_manifest(config, manifest, root=root, pred_suffix="class_agnostic")


def _generate(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    scenes = _read_split(root / args.split)
    rows: list[dict[str, Any]] = []
    for variant, meta in VARIANTS.items():
        _write_manifest(root, variant, str(meta["config"]), args.input_config, scene_only=False)
        _write_manifest(root, variant, f"{meta['config']}_scene0081", args.input_config, scene_only=True)
    for scene in scenes:
        pred_path = root / "data/prediction" / f"{args.input_config}_class_agnostic" / f"{scene}.npz"
        pre_path = root / "data/TMP" / args.input_config / f"{scene}_pre_points.npy"
        gt_path = root / args.gt_path / f"{scene}.txt"
        with np.load(pred_path) as pred:
            masks = np.asarray(pred["pred_masks"], dtype=bool)
            scores = np.asarray(pred["pred_score"], dtype=np.float32)
            classes = np.asarray(pred["pred_classes"], dtype=np.int32)
        raw_pre_points = np.load(pre_path).astype(np.int64)
        gt_eval = _class_agnostic_gt(np.loadtxt(gt_path, dtype=np.int64))
        per_gt, max_iou, _best_gt = _candidate_gt_iou(masks, gt_eval, raw_pre_points)
        trace_penalty = _load_trace_penalties(root, Path(args.trace_root), scene, masks.shape[1])
        for variant, meta in VARIANTS.items():
            keep = _select_indices(variant, masks, per_gt, max_iou, trace_penalty)
            out_masks = masks[:, keep]
            out_scores = max_iou[keep].astype(np.float32, copy=True)
            if variant == "B0_raw_all_candidates":
                out_scores = scores[keep].astype(np.float32, copy=True)
            out_classes = classes[keep]
            config = str(meta["config"])
            _write_prediction(root, config, scene, out_masks, out_scores, out_classes, raw_pre_points)
            if scene == args.scene0081:
                _write_prediction(root, f"{config}_scene0081", scene, out_masks, out_scores, out_classes, raw_pre_points)
            stats = _candidate_oracle_stats(out_masks, out_scores, gt_eval, raw_pre_points)
            row = _quality_stats(scene, variant, out_masks, out_scores, gt_eval, raw_pre_points, stats)
            row.update(
                {
                    "config": config,
                    "num_predictions_before": int(masks.shape[1]),
                    **_multiplicity_metrics(out_masks, gt_eval, raw_pre_points),
                }
            )
            rows.append(row)
            del out_masks, out_scores, out_classes
        del masks, scores, classes
    return rows


def _run_eval(args: argparse.Namespace, root: Path, config: str, output_root: Path, scene_only: bool = False) -> dict[str, Any]:
    eval_dir = output_root / ("eval_scene0081" if scene_only else "eval")
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


def _aggregate(rows: list[dict[str, Any]], evals: dict[str, dict[str, Any]], scene_evals: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for variant, meta in VARIANTS.items():
        items = [row for row in rows if row["variant"] == variant]
        agg: dict[str, Any] = {"variant": variant, "config": meta["config"], "scene_count": len(items)}
        keys = sorted(key for row in items for key, value in row.items() if isinstance(value, (int, float)) and key != "scene")
        for key in keys:
            vals = [float(row[key]) for row in items if isinstance(row.get(key), (int, float))]
            if vals:
                agg[key] = float(np.mean(vals))
        ev = evals[variant]
        scene_ev = scene_evals[variant]
        agg.update(
            {
                "AP": ev["metrics"].get("AP"),
                "AP50": ev["metrics"].get("AP50"),
                "AP25": ev["metrics"].get("AP25"),
                "eval_exit_code": ev["exit_code"],
                "metric_file": ev["metric_file"],
                "scene0081_AP": scene_ev["metrics"].get("AP"),
                "scene0081_AP50": scene_ev["metrics"].get("AP50"),
                "scene0081_AP25": scene_ev["metrics"].get("AP25"),
                "scene0081_eval_exit_code": scene_ev["exit_code"],
                "scene0081_metric_file": scene_ev["metric_file"],
            }
        )
        matrix.append(agg)
    return matrix


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stream4D v39 Phase B Old Pool Oracle Set Cover",
        "",
        "| variant | AP | AP50 | AP25 | predictions | conflict | multiplicity mean/p90 | scene0081 AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["matrix"]:
        lines.append(
            "| {variant} | {AP} | {AP50} | {AP25} | {pred} | {conflict} | {mult}/{p90} | {s81} |".format(
                variant=row["variant"],
                AP=row.get("AP"),
                AP50=row.get("AP50"),
                AP25=row.get("AP25"),
                pred=row.get("num_predictions"),
                conflict=row.get("mean_export_conflict_rate"),
                mult=row.get("candidate_multiplicity_per_GT_mean"),
                p90=row.get("candidate_multiplicity_per_GT_p90"),
                s81=row.get("scene0081_AP"),
            )
        )
    gate = summary["phaseB_gate"]
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"`best_oracle_variant={gate['best_oracle_variant']}`",
            "",
            f"`old_pool_oracle_compact_set_pass={gate['old_pool_oracle_compact_set_pass']}`",
            "",
            f"`stop1_old_pool_oracle_low={gate['stop1_old_pool_oracle_low']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    if Path.cwd().resolve() != root:
        os.chdir(root)
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _generate(args, root)
    evals = {variant: _run_eval(args, root, str(meta["config"]), output_root) for variant, meta in VARIANTS.items()}
    scene_evals = {
        variant: _run_eval(args, root, f"{meta['config']}_scene0081", output_root, scene_only=True)
        for variant, meta in VARIANTS.items()
    }
    matrix = _aggregate(rows, evals, scene_evals)
    compact = [row for row in matrix if row["variant"] in {"B4_ap_aware_set_cover", "B6_object_count_regularized"}]
    best = max(compact, key=lambda row: float(row.get("AP") or -1.0))
    gate = {
        "best_oracle_variant": best["variant"],
        "best_oracle_AP": best.get("AP"),
        "best_oracle_AP50": best.get("AP50"),
        "best_oracle_AP25": best.get("AP25"),
        "best_oracle_mean_predictions_per_scene": best.get("num_predictions"),
        "best_oracle_conflict": best.get("mean_export_conflict_rate"),
        "old_pool_oracle_compact_set_pass": bool(
            float(best.get("AP") or 0.0) >= 0.35
            and float(best.get("AP50") or 0.0) >= 0.50
            and float(best.get("AP25") or 0.0) >= 0.70
            and float(best.get("num_predictions") or 1e9) <= 200.0
            and float(best.get("mean_export_conflict_rate") or 1.0) <= 0.10
        ),
    }
    gate["stop1_old_pool_oracle_low"] = bool(float(best.get("AP") or 0.0) < 0.35)
    summary = {
        "phase": "v39_phaseB_old_pool_oracle_set_cover",
        "input_config": args.input_config,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": True,
        "matrix": matrix,
        "scene_rows": rows,
        "evals": evals,
        "scene0081_evals": scene_evals,
        "phaseB_gate": gate,
        "notes": [
            "All B1-B6 rows are GT/oracle diagnostic rows and are forbidden for method tables.",
            "B0 copies the old v38 candidate pool under the v39 Phase B manifest boundary.",
            "scene0081_AP values come from scene0081-only prediction directories and evaluator runs.",
        ],
    }
    _write_json(output_root / "old_pool_oracle_set_cover_summary.json", summary)
    _write_csv(output_root / "old_pool_oracle_set_cover_matrix.csv", matrix)
    _write_csv(output_root / "old_pool_oracle_set_cover_scene_rows.csv", rows)
    _write_markdown(output_root / "old_pool_oracle_set_cover_summary.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v39 Phase B old candidate pool oracle set-cover diagnostic.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--input-config", default=RAW_CONFIG)
    parser.add_argument("--output-root", default="outputs/audit/v39_old_pool_oracle_set_cover")
    parser.add_argument("--gt-path", default="data/scannet/gt")
    parser.add_argument("--trace-root", default="outputs/audit/v38_export_trace")
    parser.add_argument("--scene0081", default=SCENE0081)
    parser.add_argument("--cuda-visible-devices", default="6")
    return parser


def main() -> None:
    print(json.dumps(_json_safe(run(build_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
