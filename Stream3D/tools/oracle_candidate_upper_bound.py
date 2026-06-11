"""GT-read-only candidate-pool upper-bound diagnostic.

This tool must not be used to produce a method result. It reads GT only to
answer whether a prediction pool contains enough high-IoU candidates in the
first place. The optional output prediction is an oracle diagnostic artifact.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from evaluation.constants import SCANNET_IDS
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


MIN_REGION_SIZE = 100
THRESHOLDS = (0.25, 0.5, 0.75, 0.8, 0.9)


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _prediction_path(root: Path, config: str, scene: str, suffix: str) -> Path:
    return root / "data" / "prediction" / f"{config}_{suffix}" / f"{scene}.npz"


def _tmp_path(root: Path, config: str, scene: str) -> Path:
    return root / "data" / "TMP" / config / f"{scene}_pre_points.npy"


def _gt_path(root: Path, scene: str) -> Path:
    return root / "data" / "scannet" / "gt" / f"{scene}.txt"


def _class_agnostic_gt(gt_ids: np.ndarray) -> np.ndarray:
    return gt_ids % 1000 + int(SCANNET_IDS[0]) * 1000


def _gt_instance_masks(gt_ids_crop: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    instance_ids: list[int] = []
    counts: list[int] = []
    masks: list[np.ndarray] = []
    for instance_id in np.unique(gt_ids_crop):
        instance_id_int = int(instance_id)
        if instance_id_int < 1000:
            continue
        mask = gt_ids_crop == instance_id
        count = int(mask.sum())
        if count < MIN_REGION_SIZE:
            continue
        instance_ids.append(instance_id_int)
        counts.append(count)
        masks.append(mask)
    if not masks:
        return np.zeros((0, gt_ids_crop.shape[0]), dtype=bool), instance_ids, counts
    return np.stack(masks, axis=0), instance_ids, counts


def _iou_matrix(gt_masks: np.ndarray, pred_masks: np.ndarray) -> np.ndarray:
    if gt_masks.size == 0 or pred_masks.size == 0:
        return np.zeros((gt_masks.shape[0], pred_masks.shape[1]), dtype=np.float64)
    # Use int64 here. uint8 matmul silently overflows for ScanNet vertex counts.
    gt_int = gt_masks.astype(np.int64)
    pred_int = pred_masks.astype(np.int64)
    intersections = gt_int @ pred_int
    gt_area = gt_int.sum(axis=1, keepdims=True)
    pred_area = pred_int.sum(axis=0, keepdims=True)
    unions = gt_area + pred_area - intersections
    with np.errstate(divide="ignore", invalid="ignore"):
        ious = intersections / np.maximum(unions, 1)
    return ious.astype(np.float64)


def _greedy_one_to_one(ious: np.ndarray, min_select_iou: float) -> list[tuple[int, int, float]]:
    pairs: list[tuple[float, int, int]] = []
    for gt_idx, pred_idx in zip(*np.nonzero(ious >= min_select_iou)):
        pairs.append((float(ious[gt_idx, pred_idx]), int(gt_idx), int(pred_idx)))
    pairs.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    selected: list[tuple[int, int, float]] = []
    for iou, gt_idx, pred_idx in pairs:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        selected.append((gt_idx, pred_idx, iou))
    selected.sort(key=lambda row: row[2], reverse=True)
    return selected


def _scene_summary(
    scene: str,
    pred_npz: np.lib.npyio.NpzFile,
    support: np.ndarray,
    gt_ids_full: np.ndarray,
    min_select_iou: float,
) -> tuple[dict[str, object], list[int], list[float]]:
    gt_ids_crop = _class_agnostic_gt(gt_ids_full.astype(np.int64))[support]
    gt_masks, gt_instance_ids, gt_counts = _gt_instance_masks(gt_ids_crop)

    pred_masks_full = pred_npz["pred_masks"].astype(bool)
    pred_masks_crop = pred_masks_full[support]
    pred_areas = pred_masks_crop.sum(axis=0)
    valid_pred = np.flatnonzero(pred_areas >= MIN_REGION_SIZE)
    valid_pred_masks_crop = pred_masks_crop[:, valid_pred]

    ious = _iou_matrix(gt_masks, valid_pred_masks_crop)
    if ious.shape[1]:
        best_iou_per_gt = ious.max(axis=1)
        best_pred_local = ious.argmax(axis=1)
    else:
        best_iou_per_gt = np.zeros((gt_masks.shape[0],), dtype=np.float64)
        best_pred_local = np.full((gt_masks.shape[0],), -1, dtype=np.int64)

    selected_local = _greedy_one_to_one(ious, min_select_iou)
    selected_pred_indices = [int(valid_pred[pred_local]) for _, pred_local, _ in selected_local]
    selected_scores = [float(iou) for _, _, iou in selected_local]

    threshold_counts = {
        f"gt_best_iou_ge_{str(th).replace('.', 'p')}": int((best_iou_per_gt >= th).sum())
        for th in THRESHOLDS
    }
    selected_threshold_counts = {
        f"oracle_selected_iou_ge_{str(th).replace('.', 'p')}": int(
            sum(iou >= th for _, _, iou in selected_local)
        )
        for th in THRESHOLDS
    }

    summary: dict[str, object] = {
        "scene": scene,
        "num_gt_instances": int(gt_masks.shape[0]),
        "num_pred_instances": int(pred_masks_full.shape[1]),
        "num_valid_pred_instances_in_support": int(valid_pred.shape[0]),
        "num_oracle_selected": int(len(selected_local)),
        "min_select_iou": float(min_select_iou),
        "mean_best_iou_per_gt": float(best_iou_per_gt.mean()) if best_iou_per_gt.size else 0.0,
        "median_best_iou_per_gt": float(np.median(best_iou_per_gt)) if best_iou_per_gt.size else 0.0,
        "max_best_iou_per_gt": float(best_iou_per_gt.max()) if best_iou_per_gt.size else 0.0,
        "gt_instance_ids": gt_instance_ids,
        "gt_instance_vertex_counts": gt_counts,
        "best_iou_per_gt": [float(x) for x in best_iou_per_gt.tolist()],
        "best_pred_index_per_gt": [
            int(valid_pred[idx]) if idx >= 0 and len(valid_pred) else -1 for idx in best_pred_local.tolist()
        ],
        "oracle_selected_pred_indices": selected_pred_indices,
        "oracle_selected_scores": selected_scores,
    }
    summary.update(threshold_counts)
    summary.update(selected_threshold_counts)
    return summary, selected_pred_indices, selected_scores


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    seq_list = _read_seq_list(Path(args.seq_list))
    summary_root = root / args.summary_root
    summary_root.mkdir(parents=True, exist_ok=True)

    output_pred_dir = None
    output_tmp_dir = None
    if args.output_config:
        if "oracle" not in args.output_config.lower():
            raise ValueError(
                "--output-config for oracle_candidate_upper_bound must contain 'oracle'. "
                "This tool reads GT and any output prediction is diagnostic-only."
            )
        output_pred_dir = root / "data" / "prediction" / f"{args.output_config}_{args.pred_suffix}"
        output_tmp_dir = root / "data" / "TMP" / args.output_config
        if "oracle" not in output_pred_dir.name.lower() or "oracle" not in output_tmp_dir.name.lower():
            raise ValueError("Oracle output prediction and TMP directories must contain 'oracle'")
        output_pred_dir.mkdir(parents=True, exist_ok=True)
        output_tmp_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_prediction_manifest(
            root=root,
            output_config=args.output_config,
            is_method_result=False,
            is_diagnostic_only=True,
            uses_gt=True,
            gt_usage="oracle_gt_candidate_selection",
            source_configs=[args.pred_config, args.pre_points_config],
            pre_points_policy="fixed_path",
            support_policy="oracle",
            notes=(
                "GT-read-only candidate-pool upper-bound diagnostic. "
                "This output must not be reported as a method result."
            ),
            extra={
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
                "gt_selected_output": True,
                "forbidden_for_method_table": True,
                "alignment_source": "none",
                "alignment_used_for_prediction": False,
                "alignment_used_for_diagnostic": False,
            },
        )
        write_prediction_manifest(args.output_config, manifest, root=root, pred_suffix=args.pred_suffix)

    scene_summaries: list[dict[str, object]] = []
    for scene in seq_list:
        pred_path = _prediction_path(root, args.pred_config, scene, args.pred_suffix)
        support_path = _tmp_path(root, args.pre_points_config, scene)
        gt_path = _gt_path(root, scene)
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        if not support_path.exists():
            raise FileNotFoundError(support_path)
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)

        pred_npz = np.load(pred_path)
        support = np.load(support_path).astype(np.int64)
        gt_ids_full = np.loadtxt(gt_path, dtype=np.int64)
        scene_summary, selected_indices, selected_scores = _scene_summary(
            scene,
            pred_npz,
            support,
            gt_ids_full,
            args.min_select_iou,
        )
        scene_summaries.append(scene_summary)

        if output_pred_dir is not None and output_tmp_dir is not None:
            pred_masks = pred_npz["pred_masks"][:, selected_indices]
            pred_classes = pred_npz["pred_classes"][selected_indices]
            pred_scores = np.asarray(selected_scores, dtype=np.float32)
            np.savez_compressed(
                output_pred_dir / f"{scene}.npz",
                pred_masks=pred_masks,
                pred_score=pred_scores,
                pred_classes=pred_classes,
            )
            shutil.copyfile(support_path, output_tmp_dir / f"{scene}_pre_points.npy")

    def _mean(key: str) -> float:
        vals = [float(row[key]) for row in scene_summaries if key in row]
        return float(np.mean(vals)) if vals else 0.0

    aggregate: dict[str, object] = {
        "pred_config": args.pred_config,
        "pre_points_config": args.pre_points_config,
        "output_config": args.output_config,
        "num_scenes": len(scene_summaries),
        "min_select_iou": float(args.min_select_iou),
        "mean_num_gt_instances": _mean("num_gt_instances"),
        "mean_num_pred_instances": _mean("num_pred_instances"),
        "mean_num_valid_pred_instances_in_support": _mean("num_valid_pred_instances_in_support"),
        "mean_num_oracle_selected": _mean("num_oracle_selected"),
        "mean_best_iou_per_gt": _mean("mean_best_iou_per_gt"),
        "median_best_iou_per_gt_mean": _mean("median_best_iou_per_gt"),
    }
    for th in THRESHOLDS:
        key = f"gt_best_iou_ge_{str(th).replace('.', 'p')}"
        sel_key = f"oracle_selected_iou_ge_{str(th).replace('.', 'p')}"
        aggregate[f"mean_{key}"] = _mean(key)
        aggregate[f"mean_{sel_key}"] = _mean(sel_key)

    payload = {"aggregate": aggregate, "scenes": scene_summaries}
    out_path = summary_root / f"{args.output_config or args.pred_config}_oracle_upper_bound_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[oracle-candidate-upper-bound] wrote {out_path}")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--pred-config", required=True)
    parser.add_argument("--pre-points-config", required=True)
    parser.add_argument("--output-config", default="")
    parser.add_argument("--pred-suffix", default="class_agnostic")
    parser.add_argument("--min-select-iou", type=float, default=0.25)
    parser.add_argument("--summary-root", default="outputs/oracle_candidate_upper_bound")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
