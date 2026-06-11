"""v15 GT-read-only multi-measurement union oracle diagnostic.

This tool answers whether a measurement pool could explain objects if several
measurements were allowed to form one latent object. It reads GT only for
oracle selection and metric attribution. Any output prediction is forbidden for
method tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tools.oracle_candidate_upper_bound import _class_agnostic_gt, _gt_instance_masks
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


THRESHOLDS = (0.10, 0.25, 0.50, 0.75)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prediction_path(root: Path, config: str, scene: str, suffix: str) -> Path:
    return root / "data" / "prediction" / f"{config}_{suffix}" / f"{scene}.npz"


def _tmp_path(root: Path, config: str, scene: str) -> Path:
    return root / "data" / "TMP" / config / f"{scene}_pre_points.npy"


def _gt_path(root: Path, scene: str) -> Path:
    return root / "data" / "scannet" / "gt" / f"{scene}.txt"


def _metric_path(root: Path, config: str) -> Path:
    return root / "data" / "evaluation" / "scannet" / f"{config}_class_agnostic.txt"


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


def _iou_bool(lhs: np.ndarray, rhs: np.ndarray) -> float:
    inter = int(np.count_nonzero(lhs & rhs))
    union = int(np.count_nonzero(lhs | rhs))
    return float(inter / max(union, 1))


def _individual_iou(gt_mask: np.ndarray, pred_masks: np.ndarray) -> np.ndarray:
    if pred_masks.shape[1] == 0:
        return np.zeros((0,), dtype=np.float64)
    gt = gt_mask.astype(bool, copy=False)
    pred = pred_masks.astype(bool, copy=False)
    inter = np.count_nonzero(pred & gt[:, None], axis=0).astype(np.float64)
    pred_area = np.count_nonzero(pred, axis=0).astype(np.float64)
    gt_area = float(np.count_nonzero(gt))
    union = gt_area + pred_area - inter
    return inter / np.maximum(union, 1.0)


def _greedy_union(
    gt_mask: np.ndarray,
    pred_masks: np.ndarray,
    *,
    max_k: int,
    max_candidates: int,
    min_improvement: float,
) -> tuple[list[int], list[float]]:
    individual = _individual_iou(gt_mask, pred_masks)
    if individual.size == 0 or float(np.max(individual)) <= 0.0:
        return [], [0.0 for _ in range(max_k)]
    candidate_order = np.argsort(-individual, kind="mergesort")
    candidate_order = candidate_order[: int(max_candidates)]
    selected: list[int] = []
    selected_set: set[int] = set()
    union_mask = np.zeros((gt_mask.shape[0],), dtype=bool)
    best_iou = 0.0
    curve: list[float] = []
    for _ in range(int(max_k)):
        best_candidate = -1
        best_candidate_iou = best_iou
        for pred_idx in candidate_order.tolist():
            pred_idx = int(pred_idx)
            if pred_idx in selected_set:
                continue
            next_iou = _iou_bool(gt_mask, union_mask | pred_masks[:, pred_idx])
            if next_iou > best_candidate_iou:
                best_candidate = pred_idx
                best_candidate_iou = float(next_iou)
        if best_candidate < 0 or best_candidate_iou <= best_iou + float(min_improvement):
            curve.append(float(best_iou))
            continue
        selected.append(int(best_candidate))
        selected_set.add(int(best_candidate))
        union_mask |= pred_masks[:, best_candidate]
        best_iou = float(best_candidate_iou)
        curve.append(float(best_iou))
    if len(curve) < int(max_k):
        curve.extend([float(best_iou)] * (int(max_k) - len(curve)))
    return selected, curve


def _scene_union_oracle(
    *,
    root: Path,
    scene: str,
    pred_config: str,
    pre_points_config: str,
    suffix: str,
    k_values: list[int],
    min_region_size: int,
    max_candidates_per_gt: int,
    min_improvement: float,
    min_output_iou: float,
) -> tuple[dict[str, Any], dict[int, list[np.ndarray]], dict[int, list[float]]]:
    pred_path = _prediction_path(root, pred_config, scene, suffix)
    support_path = _tmp_path(root, pre_points_config, scene)
    gt_path = _gt_path(root, scene)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not support_path.exists():
        raise FileNotFoundError(support_path)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    with np.load(pred_path) as pred_npz:
        pred_masks_full = np.asarray(pred_npz["pred_masks"], dtype=bool)
        if "pred_score" in pred_npz.files:
            pred_scores = np.asarray(pred_npz["pred_score"], dtype=np.float32)
        else:
            pred_scores = np.ones((pred_masks_full.shape[1],), dtype=np.float32)
    support = np.load(support_path).astype(np.int64)
    gt_ids_full = np.loadtxt(gt_path, dtype=np.int64)
    gt_ids_crop = _class_agnostic_gt(gt_ids_full.astype(np.int64))[support]
    gt_masks, gt_instance_ids, gt_counts = _gt_instance_masks(gt_ids_crop)
    pred_masks_crop = pred_masks_full[support]
    pred_areas = np.count_nonzero(pred_masks_crop, axis=0)
    valid_pred = np.flatnonzero(pred_areas >= int(min_region_size))
    valid_pred_masks_crop = pred_masks_crop[:, valid_pred]

    max_k = max(k_values)
    per_gt: list[dict[str, Any]] = []
    output_masks_by_k: dict[int, list[np.ndarray]] = {int(k): [] for k in k_values}
    output_scores_by_k: dict[int, list[float]] = {int(k): [] for k in k_values}
    selected_union_points_by_k: dict[int, set[int]] = {int(k): set() for k in k_values}

    for gt_idx in range(gt_masks.shape[0]):
        selected_local, curve = _greedy_union(
            gt_masks[gt_idx],
            valid_pred_masks_crop,
            max_k=max_k,
            max_candidates=int(max_candidates_per_gt),
            min_improvement=float(min_improvement),
        )
        selected_pred_full = [int(valid_pred[idx]) for idx in selected_local]
        row: dict[str, Any] = {
            "gt_index": int(gt_idx),
            "gt_instance_id": int(gt_instance_ids[gt_idx]),
            "gt_vertex_count_in_support": int(gt_counts[gt_idx]),
            "selected_pred_indices": selected_pred_full,
            "selected_count": int(len(selected_pred_full)),
        }
        for k in k_values:
            k_int = int(k)
            use_local = selected_local[:k_int]
            best_iou = float(curve[k_int - 1]) if k_int - 1 < len(curve) else 0.0
            row[f"best_iou_k{k_int}"] = best_iou
            row[f"selected_count_k{k_int}"] = int(len(use_local))
            if use_local and best_iou >= float(min_output_iou):
                pred_indices = [int(valid_pred[idx]) for idx in use_local]
                union_full = np.any(pred_masks_full[:, pred_indices], axis=1)
                output_masks_by_k[k_int].append(union_full)
                output_scores_by_k[k_int].append(float(best_iou + 1e-4 * np.mean(pred_scores[pred_indices])))
                selected_union_points_by_k[k_int].update(int(v) for v in np.flatnonzero(union_full).tolist())
        per_gt.append(row)

    scene_summary: dict[str, Any] = {
        "scene": scene,
        "pred_config": pred_config,
        "pre_points_config": pre_points_config,
        "num_gt_instances": int(gt_masks.shape[0]),
        "num_pred_instances": int(pred_masks_full.shape[1]),
        "num_valid_pred_instances_in_support": int(valid_pred.shape[0]),
        "candidate_support_points": int(support.shape[0]),
        "scene_gt_points": int(gt_ids_full.shape[0]),
        "candidate_support_pre_ratio": float(support.shape[0] / max(gt_ids_full.shape[0], 1)),
        "per_gt": per_gt,
    }
    for k in k_values:
        key = f"k{int(k)}"
        best_vals = np.asarray([float(row[f"best_iou_k{int(k)}"]) for row in per_gt], dtype=np.float64)
        scene_summary[f"mean_best_iou_{key}"] = float(np.mean(best_vals)) if best_vals.size else 0.0
        scene_summary[f"median_best_iou_{key}"] = float(np.median(best_vals)) if best_vals.size else 0.0
        scene_summary[f"mean_selected_count_{key}"] = (
            float(np.mean([int(row[f"selected_count_k{int(k)}"]) for row in per_gt])) if per_gt else 0.0
        )
        scene_summary[f"selected_union_pre_ratio_{key}"] = float(
            len(selected_union_points_by_k[int(k)]) / max(gt_ids_full.shape[0], 1)
        )
        for th in THRESHOLDS:
            th_key = str(th).replace(".", "p")
            scene_summary[f"gt_best_iou_ge_{th_key}_{key}"] = int(np.count_nonzero(best_vals >= th))
    return scene_summary, output_masks_by_k, output_scores_by_k


def _write_outputs(
    *,
    root: Path,
    scene_outputs: dict[str, tuple[dict[int, list[np.ndarray]], dict[int, list[float]]]],
    output_config_prefix: str,
    pred_suffix: str,
    pre_points_config: str,
    source_config: str,
    k_values: list[int],
    eval_support: str,
    algorithm_name: str,
) -> list[str]:
    if "oracle" not in output_config_prefix.lower():
        raise ValueError("--output-config-prefix for union oracle must contain 'oracle'")
    output_configs: list[str] = []
    for k in k_values:
        k_int = int(k)
        output_config = f"{output_config_prefix}_k{k_int}"
        output_configs.append(output_config)
        pred_dir = root / "data" / "prediction" / f"{output_config}_{pred_suffix}"
        tmp_dir = root / "data" / "TMP" / output_config
        pred_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_prediction_manifest(
            root=root,
            output_config=output_config,
            is_method_result=False,
            is_diagnostic_only=True,
            uses_gt=True,
            gt_usage=f"{algorithm_name}_multi_measurement_union_oracle_selection",
            source_configs=[source_config, pre_points_config],
            pre_points_policy=f"oracle_{eval_support}_support",
            support_policy=algorithm_name,
            notes="GT-selected multi-measurement union oracle diagnostic. Forbidden for method tables.",
            extra={
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
                "gt_selected_output": True,
                "forbidden_for_method_table": True,
                "alignment_source": "none",
                "alignment_used_for_prediction": False,
                "alignment_used_for_diagnostic": False,
                "oracle_k": k_int,
            },
        )
        write_prediction_manifest(output_config, manifest, root=root, pred_suffix=pred_suffix)

        for scene, (masks_by_k, scores_by_k) in scene_outputs.items():
            masks = masks_by_k[k_int]
            scores = scores_by_k[k_int]
            if masks:
                pred_masks = np.stack(masks, axis=1).astype(bool, copy=False)
                pred_score = np.asarray(scores, dtype=np.float32)
                pre_points = np.flatnonzero(np.any(pred_masks, axis=1)).astype(np.int64)
            else:
                gt_path = _gt_path(root, scene)
                n_points = int(np.loadtxt(gt_path, dtype=np.int64).shape[0])
                pred_masks = np.zeros((n_points, 0), dtype=bool)
                pred_score = np.zeros((0,), dtype=np.float32)
                pre_points = np.zeros((0,), dtype=np.int64)
            np.savez_compressed(
                pred_dir / f"{scene}.npz",
                pred_masks=pred_masks,
                pred_score=pred_score,
                pred_classes=np.zeros((pred_masks.shape[1],), dtype=np.int32),
            )
            if eval_support == "candidate":
                shutil.copyfile(_tmp_path(root, pre_points_config, scene), tmp_dir / f"{scene}_pre_points.npy")
            elif eval_support == "selected_union":
                np.save(tmp_dir / f"{scene}_pre_points.npy", pre_points)
            else:
                raise ValueError(f"Unsupported eval support: {eval_support}")
    return output_configs


def _evaluate(root: Path, output_config: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(Path("data/prediction") / f"{output_config}_class_agnostic"),
        "--gt_path",
        "data/scannet/gt",
        "--dataset",
        "scannet",
        "--output_file",
        str(Path("data/evaluation/scannet") / f"{output_config}_class_agnostic.txt"),
        "--tmp_root",
        "data/TMP",
        "--tmp_config",
        output_config,
        "--no_class",
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    subprocess.run(cmd, cwd=str(root), check=True)


def _mean_scene_key(scenes: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in scenes if row.get(key) is not None]
    return float(np.mean(vals)) if vals else 0.0


def _write_bundle(prefix: Path, payload: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    rows = payload["rows"]
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "# Stream4D v15 Union Oracle Diagnostic",
        "",
        "GT is used only for oracle union selection and diagnostic attribution. These rows are forbidden for method tables.",
        "",
        "| K | AP/AP50/AP25 | candidate pre% | selected union pre% | best IoU | IoU>=0.25 | IoU>=0.50 | selected/object | gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["k"]),
                    f"{_fmt(row.get('ap'))}/{_fmt(row.get('ap50'))}/{_fmt(row.get('ap25'))}",
                    _fmt(row.get("candidate_support_pre_ratio"), scale=100.0),
                    _fmt(row.get("selected_union_pre_ratio"), scale=100.0),
                    _fmt(row.get("mean_best_iou")),
                    _fmt(row.get("mean_gt_best_iou_ge_0p25"), digits=2),
                    _fmt(row.get("mean_gt_best_iou_ge_0p50"), digits=2),
                    _fmt(row.get("mean_selected_count"), digits=2),
                    str(row.get("gate_pass")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- any_gate_pass: `{payload['summary'].get('any_gate_pass')}`",
            f"- best_k_by_ap50: `{payload['summary'].get('best_k_by_ap50')}`",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, *, scale: float = 1.0, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value) * float(scale):.{int(digits)}f}"
    except Exception:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--pred-config", required=True)
    parser.add_argument("--pre-points-config", default="", help="Defaults to --pred-config.")
    parser.add_argument("--pred-suffix", default="class_agnostic")
    parser.add_argument("--k", type=int, action="append", default=None)
    parser.add_argument("--output-config-prefix", required=True)
    parser.add_argument("--summary-prefix", default="outputs/audit/v15_phase1/union_oracle_probe5")
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--max-candidates-per-gt", type=int, default=256)
    parser.add_argument("--min-improvement", type=float, default=1e-6)
    parser.add_argument("--min-output-iou", type=float, default=0.25)
    parser.add_argument("--eval-support", choices=["candidate", "selected_union"], default="candidate")
    parser.add_argument("--algorithm-name", default="v15_union_oracle")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    pre_points_config = args.pre_points_config or args.pred_config
    raw_k_values = args.k if args.k is not None else [1, 2, 4, 8, 16, 32]
    k_values = sorted({int(k) for k in raw_k_values if int(k) > 0})
    scenes = _read_seq_list(root / args.seq_list)
    scene_summaries: list[dict[str, Any]] = []
    scene_outputs: dict[str, tuple[dict[int, list[np.ndarray]], dict[int, list[float]]]] = {}
    for scene in scenes:
        scene_summary, masks_by_k, scores_by_k = _scene_union_oracle(
            root=root,
            scene=scene,
            pred_config=args.pred_config,
            pre_points_config=pre_points_config,
            suffix=args.pred_suffix,
            k_values=k_values,
            min_region_size=int(args.min_region_size),
            max_candidates_per_gt=int(args.max_candidates_per_gt),
            min_improvement=float(args.min_improvement),
            min_output_iou=float(args.min_output_iou),
        )
        scene_summaries.append(scene_summary)
        scene_outputs[scene] = (masks_by_k, scores_by_k)

    output_configs = _write_outputs(
        root=root,
        scene_outputs=scene_outputs,
        output_config_prefix=args.output_config_prefix,
        pred_suffix=args.pred_suffix,
        pre_points_config=pre_points_config,
        source_config=args.pred_config,
        k_values=k_values,
        eval_support=args.eval_support,
        algorithm_name=str(args.algorithm_name),
    )
    if not args.skip_eval:
        for output_config in output_configs:
            _evaluate(root, output_config)

    rows: list[dict[str, Any]] = []
    for k, output_config in zip(k_values, output_configs):
        key = f"k{int(k)}"
        metric = _parse_metric_file(_metric_path(root, output_config))
        mean_best = _mean_scene_key(scene_summaries, f"mean_best_iou_{key}")
        selected_union_pre = _mean_scene_key(scene_summaries, f"selected_union_pre_ratio_{key}")
        candidate_pre = _mean_scene_key(scene_summaries, "candidate_support_pre_ratio")
        row = {
            "k": int(k),
            "output_config": output_config,
            "ap": metric["ap"],
            "ap50": metric["ap50"],
            "ap25": metric["ap25"],
            "candidate_support_pre_ratio": candidate_pre,
            "selected_union_pre_ratio": selected_union_pre,
            "mean_best_iou": mean_best,
            "median_best_iou": _mean_scene_key(scene_summaries, f"median_best_iou_{key}"),
            "mean_selected_count": _mean_scene_key(scene_summaries, f"mean_selected_count_{key}"),
            "mean_gt_best_iou_ge_0p10": _mean_scene_key(scene_summaries, f"gt_best_iou_ge_0p1_{key}"),
            "mean_gt_best_iou_ge_0p25": _mean_scene_key(scene_summaries, f"gt_best_iou_ge_0p25_{key}"),
            "mean_gt_best_iou_ge_0p50": _mean_scene_key(scene_summaries, f"gt_best_iou_ge_0p5_{key}"),
            "mean_gt_best_iou_ge_0p75": _mean_scene_key(scene_summaries, f"gt_best_iou_ge_0p75_{key}"),
        }
        row["gate_pass"] = bool(
            row["ap50"] is not None
            and row["ap25"] is not None
            and float(row["ap50"]) >= 0.60
            and float(row["ap25"]) >= 0.78
            and float(candidate_pre) >= 0.25
        )
        rows.append(row)

    best = max(rows, key=lambda row: float(row.get("ap50") or 0.0)) if rows else {}
    payload = {
        "summary": {
            "algorithm": str(args.algorithm_name),
            "pred_config": args.pred_config,
            "pre_points_config": pre_points_config,
            "output_config_prefix": args.output_config_prefix,
            "eval_support": args.eval_support,
            "num_scenes": int(len(scene_summaries)),
            "k_values": k_values,
            "best_k_by_ap50": best.get("k"),
            "best_ap50": best.get("ap50"),
            "any_gate_pass": bool(any(row.get("gate_pass") for row in rows)),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": True,
            "gt_selected_output": True,
            "forbidden_for_method_table": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
        },
        "rows": rows,
        "scenes": scene_summaries,
    }
    _write_bundle(root / args.summary_prefix, payload)
    print(json.dumps(_json_safe(payload["summary"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
