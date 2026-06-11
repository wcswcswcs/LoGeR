from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.constants import SCANNET_IDS
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


MIN_REGION_SIZE = 100
THRESHOLDS = (0.25, 0.5, 0.75)


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


def _split_configs(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _prediction_path(root: Path, config: str, suffix: str, scene: str) -> Path:
    suffix_norm = suffix[1:] if suffix.startswith("_") else suffix
    dirname = config if config.endswith(suffix_norm) else f"{config}_{suffix_norm}"
    return root / "data" / "prediction" / dirname / f"{scene}.npz"


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
    gt_int = gt_masks.astype(np.int64, copy=False)
    pred_int = pred_masks.astype(np.int64, copy=False)
    intersections = gt_int @ pred_int
    gt_area = gt_int.sum(axis=1, keepdims=True)
    pred_area = pred_int.sum(axis=0, keepdims=True)
    unions = gt_area + pred_area - intersections
    return (intersections / np.maximum(unions, 1)).astype(np.float64)


def _load_prediction(root: Path, config: str, suffix: str, scene: str) -> dict[str, np.ndarray]:
    path = _prediction_path(root, config, suffix, scene)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        masks = np.asarray(data["pred_masks"], dtype=bool)
        scores = np.asarray(data["pred_score"], dtype=np.float32)
        classes = np.asarray(data["pred_classes"], dtype=np.int32)
    if scores.shape[0] != masks.shape[1]:
        scores = np.ones((masks.shape[1],), dtype=np.float32)
    if classes.shape[0] != masks.shape[1]:
        classes = np.zeros((masks.shape[1],), dtype=np.int32)
    return {"pred_masks": masks, "pred_score": scores, "pred_classes": classes}


def _candidate_overlap(candidate: np.ndarray, kept: np.ndarray, kept_areas: np.ndarray, mode: str) -> float:
    if kept.shape[1] == 0:
        return 0.0
    cand_area = float(np.count_nonzero(candidate))
    if cand_area <= 0.0:
        return 0.0
    inter = np.logical_and(kept, candidate[:, None]).sum(axis=0).astype(np.float64)
    if mode == "iou":
        overlap = inter / np.maximum(kept_areas + cand_area - inter, 1.0)
    elif mode == "min_ioc":
        overlap = inter / np.maximum(np.minimum(kept_areas, cand_area), 1.0)
    elif mode == "candidate_ioc":
        overlap = inter / max(cand_area, 1.0)
    else:
        raise ValueError(f"Unsupported dedup overlap mode: {mode}")
    return float(np.max(overlap)) if overlap.size else 0.0


def _dedup_candidates(
    masks: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    sources: list[str],
    *,
    support: np.ndarray,
    min_area: int,
    threshold: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if masks.shape[1] == 0:
        return masks, scores[:0], classes[:0], [], {
            "num_candidates_before_area_filter": 0,
            "num_candidates_after_area_filter": 0,
            "num_candidates_after_dedup": 0,
            "num_candidates_dedup_removed": 0,
        }
    crop = masks[support, :]
    areas = crop.sum(axis=0).astype(np.int64)
    valid = np.flatnonzero(areas >= int(min_area))
    if valid.size == 0:
        return masks[:, :0], scores[:0], classes[:0], [], {
            "num_candidates_before_area_filter": int(masks.shape[1]),
            "num_candidates_after_area_filter": 0,
            "num_candidates_after_dedup": 0,
            "num_candidates_dedup_removed": 0,
        }
    order = sorted(
        valid.tolist(),
        key=lambda idx: (float(scores[idx]), int(areas[idx]), -int(idx)),
        reverse=True,
    )
    kept_indices: list[int] = []
    kept_crop_parts: list[np.ndarray] = []
    kept_areas: list[float] = []
    max_overlaps: list[float] = []
    for idx in order:
        candidate = crop[:, idx]
        if kept_crop_parts:
            kept_crop = np.stack(kept_crop_parts, axis=1)
            overlap = _candidate_overlap(
                candidate,
                kept_crop,
                np.asarray(kept_areas, dtype=np.float64),
                mode,
            )
        else:
            overlap = 0.0
        max_overlaps.append(overlap)
        if overlap >= float(threshold):
            continue
        kept_indices.append(int(idx))
        kept_crop_parts.append(candidate.copy())
        kept_areas.append(float(areas[idx]))

    kept = np.asarray(kept_indices, dtype=np.int64)
    return (
        masks[:, kept],
        scores[kept],
        classes[kept],
        [sources[int(idx)] for idx in kept.tolist()],
        {
            "num_candidates_before_area_filter": int(masks.shape[1]),
            "num_candidates_after_area_filter": int(valid.shape[0]),
            "num_candidates_after_dedup": int(kept.shape[0]),
            "num_candidates_dedup_removed": int(valid.shape[0] - kept.shape[0]),
            "dedup_max_overlap_mean": float(np.mean(max_overlaps)) if max_overlaps else 0.0,
            "dedup_threshold": float(threshold),
            "dedup_overlap_mode": mode,
        },
    )


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
    selected.sort(key=lambda item: item[2], reverse=True)
    return selected


def _conflict_rate(masks_crop: np.ndarray) -> float:
    if masks_crop.shape[1] == 0:
        return 0.0
    union = np.any(masks_crop, axis=1)
    if not np.any(union):
        return 0.0
    owners = masks_crop.sum(axis=1)
    return float(np.count_nonzero(owners[union] > 1) / max(int(np.count_nonzero(union)), 1))


def _scene_summary(
    *,
    scene: str,
    pool_masks_full: np.ndarray,
    pool_scores: np.ndarray,
    pool_classes: np.ndarray,
    pool_sources: list[str],
    support: np.ndarray,
    gt_ids_full: np.ndarray,
    min_select_iou: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    gt_ids_crop = _class_agnostic_gt(gt_ids_full.astype(np.int64))[support]
    gt_masks, gt_instance_ids, gt_counts = _gt_instance_masks(gt_ids_crop)
    pred_masks_crop = pool_masks_full[support, :]
    ious = _iou_matrix(gt_masks, pred_masks_crop)
    if ious.shape[1] > 0:
        best_iou_per_gt = ious.max(axis=1)
        best_pred_idx = ious.argmax(axis=1)
    else:
        best_iou_per_gt = np.zeros((gt_masks.shape[0],), dtype=np.float64)
        best_pred_idx = np.full((gt_masks.shape[0],), -1, dtype=np.int64)
    selected = _greedy_one_to_one(ious, min_select_iou)
    selected_pred_indices = np.asarray([pred_idx for _, pred_idx, _ in selected], dtype=np.int64)
    selected_scores = np.asarray([iou for _, _, iou in selected], dtype=np.float32)

    area = pred_masks_crop.sum(axis=0).astype(np.float64)
    area_quantiles = {
        f"candidate_area_q{q:02d}": float(np.percentile(area, q)) if area.size else 0.0
        for q in (10, 25, 50, 75, 90)
    }
    source_counts: dict[str, int] = {}
    for source in pool_sources:
        source_counts[source] = source_counts.get(source, 0) + 1

    duplicate_counts = (ious >= 0.25).sum(axis=1) if ious.size else np.zeros((gt_masks.shape[0],), dtype=np.int64)
    summary: dict[str, Any] = {
        "scene": scene,
        "num_scene_vertices": int(gt_ids_full.shape[0]),
        "support_points": int(support.shape[0]),
        "candidate_pool_union_points": int(np.count_nonzero(np.any(pool_masks_full, axis=1)))
        if pool_masks_full.shape[1]
        else 0,
        "candidate_pool_union_ratio": float(
            np.count_nonzero(np.any(pool_masks_full, axis=1)) / max(int(gt_ids_full.shape[0]), 1)
        )
        if pool_masks_full.shape[1]
        else 0.0,
        "num_gt_instances": int(gt_masks.shape[0]),
        "gt_instance_ids": gt_instance_ids,
        "gt_instance_vertex_counts": gt_counts,
        "num_candidates": int(pool_masks_full.shape[1]),
        "num_oracle_selected": int(selected_pred_indices.shape[0]),
        "mean_best_iou_per_gt": float(best_iou_per_gt.mean()) if best_iou_per_gt.size else 0.0,
        "median_best_iou_per_gt": float(np.median(best_iou_per_gt)) if best_iou_per_gt.size else 0.0,
        "max_best_iou_per_gt": float(best_iou_per_gt.max()) if best_iou_per_gt.size else 0.0,
        "candidate_conflict_rate": _conflict_rate(pred_masks_crop),
        "duplicate_predictions_per_gt_mean_at_025": float(duplicate_counts.mean())
        if duplicate_counts.size
        else 0.0,
        "duplicate_predictions_per_gt_median_at_025": float(np.median(duplicate_counts))
        if duplicate_counts.size
        else 0.0,
        "best_iou_per_gt": [float(v) for v in best_iou_per_gt.tolist()],
        "best_pred_index_per_gt": [int(v) for v in best_pred_idx.tolist()],
        "oracle_selected_pred_indices": [int(v) for v in selected_pred_indices.tolist()],
        "oracle_selected_scores": [float(v) for v in selected_scores.tolist()],
        "oracle_selected_sources": [pool_sources[int(v)] for v in selected_pred_indices.tolist()],
        "source_counts_after_dedup": source_counts,
        **area_quantiles,
    }
    for th in THRESHOLDS:
        key = str(th).replace(".", "p")
        summary[f"gt_best_iou_ge_{key}"] = int(np.count_nonzero(best_iou_per_gt >= th))
        summary[f"oracle_selected_iou_ge_{key}"] = int(np.count_nonzero(selected_scores >= th))

    selected_masks = pool_masks_full[:, selected_pred_indices] if selected_pred_indices.size else pool_masks_full[:, :0]
    selected_classes = (
        pool_classes[selected_pred_indices] if selected_pred_indices.size else np.zeros((0,), dtype=np.int32)
    )
    return summary, selected_masks, selected_scores, selected_classes


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return float(np.mean(vals)) if vals else 0.0


def _write_tabular(out_prefix: Path, payload: dict[str, Any]) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    rows = payload["scenes"]
    if rows:
        fieldnames = [
            "scene",
            "support_points",
            "candidate_pool_union_ratio",
            "num_gt_instances",
            "num_candidates",
            "num_oracle_selected",
            "mean_best_iou_per_gt",
            "median_best_iou_per_gt",
            "gt_best_iou_ge_0p25",
            "gt_best_iou_ge_0p5",
            "gt_best_iou_ge_0p75",
            "oracle_selected_iou_ge_0p25",
            "oracle_selected_iou_ge_0p5",
            "oracle_selected_iou_ge_0p75",
            "candidate_conflict_rate",
            "duplicate_predictions_per_gt_mean_at_025",
        ]
        with out_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
    agg = payload["aggregate"]
    lines = [
        f"# {agg['pool_name']}",
        "",
        "GT is read only for oracle upper-bound diagnostics. These rows are not method results.",
        "",
        "## Aggregate",
        "",
    ]
    for key in (
        "num_scenes",
        "mean_candidate_count",
        "mean_candidate_pool_union_ratio",
        "mean_best_iou_per_gt",
        "median_best_iou_per_gt_mean",
        "mean_gt_best_iou_ge_0p25",
        "mean_gt_best_iou_ge_0p5",
        "mean_gt_best_iou_ge_0p75",
        "mean_oracle_selected",
        "mean_oracle_selected_iou_ge_0p25",
        "mean_oracle_selected_iou_ge_0p5",
        "mean_oracle_selected_iou_ge_0p75",
        "mean_candidate_conflict_rate",
        "mean_duplicate_predictions_per_gt_at_025",
    ):
        lines.append(f"- {key}: `{agg.get(key)}`")
    lines.extend(
        [
            "",
            "## Scenes",
            "",
            "| scene | candidates | union% | GT | best IoU mean | GT>=.25 | GT>=.50 | oracle selected | conflict | dup/GT |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scene"]),
                    str(row["num_candidates"]),
                    f"{float(row['candidate_pool_union_ratio']) * 100.0:.4f}",
                    str(row["num_gt_instances"]),
                    f"{float(row['mean_best_iou_per_gt']):.6f}",
                    str(row["gt_best_iou_ge_0p25"]),
                    str(row["gt_best_iou_ge_0p5"]),
                    str(row["num_oracle_selected"]),
                    f"{float(row['candidate_conflict_rate']) * 100.0:.4f}",
                    f"{float(row['duplicate_predictions_per_gt_mean_at_025']):.4f}",
                ]
            )
            + " |"
        )
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    configs = _split_configs(args.pool_configs)
    if not configs:
        raise ValueError("--pool-configs must contain at least one config")
    scenes = _read_seq_list((root / args.seq_list).resolve())
    summary_root = root / args.summary_root
    summary_root.mkdir(parents=True, exist_ok=True)

    output_pred_dir = root / "data" / "prediction" / f"{args.output_config}_class_agnostic"
    output_tmp_dir = root / "data" / "TMP" / args.output_config
    if "oracle" not in args.output_config.lower():
        raise ValueError("--output-config must contain 'oracle' for GT-read diagnostic outputs")
    output_pred_dir.mkdir(parents=True, exist_ok=True)
    output_tmp_dir.mkdir(parents=True, exist_ok=True)

    scene_summaries: list[dict[str, Any]] = []
    dedup_summaries: list[dict[str, Any]] = []
    for scene in scenes:
        loaded = [_load_prediction(root, config, args.pred_suffix, scene) for config in configs]
        vertex_counts = {int(item["pred_masks"].shape[0]) for item in loaded}
        if len(vertex_counts) != 1:
            raise ValueError(f"{scene}: prediction vertex count mismatch: {sorted(vertex_counts)}")
        all_masks = np.concatenate([item["pred_masks"] for item in loaded], axis=1)
        all_scores = np.concatenate([item["pred_score"] for item in loaded], axis=0)
        all_classes = np.concatenate([item["pred_classes"] for item in loaded], axis=0)
        all_sources: list[str] = []
        for config, item in zip(configs, loaded):
            all_sources.extend([config] * int(item["pred_masks"].shape[1]))

        if args.support_mode == "union":
            raw_union = np.flatnonzero(np.any(all_masks, axis=1)).astype(np.int64)
            support_for_dedup = raw_union
        elif args.support_mode == "fixed":
            support_path = _tmp_path(root, args.pre_points_config, scene)
            if not support_path.exists():
                raise FileNotFoundError(support_path)
            support_for_dedup = np.load(support_path).astype(np.int64)
        else:
            raise ValueError(f"Unsupported support mode: {args.support_mode}")

        pool_masks, pool_scores, pool_classes, pool_sources, dedup_diag = _dedup_candidates(
            all_masks,
            all_scores,
            all_classes,
            all_sources,
            support=support_for_dedup,
            min_area=int(args.min_candidate_points),
            threshold=float(args.dedup_threshold),
            mode=args.dedup_overlap_mode,
        )
        if args.support_mode == "union":
            support = np.flatnonzero(np.any(pool_masks, axis=1)).astype(np.int64)
        else:
            support = support_for_dedup
        gt_path = _gt_path(root, scene)
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        gt_ids_full = np.loadtxt(gt_path, dtype=np.int64)
        scene_summary, selected_masks, selected_scores, selected_classes = _scene_summary(
            scene=scene,
            pool_masks_full=pool_masks,
            pool_scores=pool_scores,
            pool_classes=pool_classes,
            pool_sources=pool_sources,
            support=support,
            gt_ids_full=gt_ids_full,
            min_select_iou=float(args.min_select_iou),
        )
        scene_summary.update(dedup_diag)
        scene_summaries.append(scene_summary)
        dedup_summaries.append(dedup_diag)

        np.savez_compressed(
            output_pred_dir / f"{scene}.npz",
            pred_masks=selected_masks,
            pred_score=selected_scores,
            pred_classes=selected_classes,
        )
        np.save(output_tmp_dir / f"{scene}_pre_points.npy", support)

    aggregate: dict[str, Any] = {
        "pool_name": args.pool_name or args.output_config,
        "output_config": args.output_config,
        "pool_configs": configs,
        "support_mode": args.support_mode,
        "pre_points_config": args.pre_points_config,
        "num_scenes": int(len(scene_summaries)),
        "min_select_iou": float(args.min_select_iou),
        "dedup_threshold": float(args.dedup_threshold),
        "dedup_overlap_mode": args.dedup_overlap_mode,
        "mean_candidate_count": _mean(scene_summaries, "num_candidates"),
        "mean_candidate_pool_union_ratio": _mean(scene_summaries, "candidate_pool_union_ratio"),
        "mean_best_iou_per_gt": _mean(scene_summaries, "mean_best_iou_per_gt"),
        "median_best_iou_per_gt_mean": _mean(scene_summaries, "median_best_iou_per_gt"),
        "mean_oracle_selected": _mean(scene_summaries, "num_oracle_selected"),
        "mean_candidate_conflict_rate": _mean(scene_summaries, "candidate_conflict_rate"),
        "mean_duplicate_predictions_per_gt_at_025": _mean(
            scene_summaries,
            "duplicate_predictions_per_gt_mean_at_025",
        ),
        "is_diagnostic_only": True,
        "uses_gt_for_prediction": True,
        "is_method_result": False,
    }
    for th in THRESHOLDS:
        key = str(th).replace(".", "p")
        aggregate[f"mean_gt_best_iou_ge_{key}"] = _mean(scene_summaries, f"gt_best_iou_ge_{key}")
        aggregate[f"mean_oracle_selected_iou_ge_{key}"] = _mean(
            scene_summaries,
            f"oracle_selected_iou_ge_{key}",
        )

    payload = {"aggregate": aggregate, "scenes": scene_summaries}
    out_prefix = summary_root / f"{args.output_config}_upper_bound"
    _write_tabular(out_prefix, payload)

    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=True,
        gt_usage="oracle_gt_candidate_pool_selection",
        source_configs=configs,
        pre_points_policy=f"v11_oracle_support:{args.support_mode}",
        support_policy="candidate_pool_oracle_upper_bound",
        notes=(
            "GT-read-only candidate-pool oracle upper-bound diagnostic. "
            "The output prediction is selected with GT and must not enter a method table."
        ),
        extra={
            "algorithm": "v11_candidate_pool_oracle",
            "eval_policy": "oracle_candidate_upper_bound",
            "prediction_config": args.output_config,
            "pre_points_config": args.output_config,
            "support_source": args.support_mode,
            "geometry_source": "mixed_existing_predictions",
            "uses_gt_for_prediction": True,
            "uses_gt_for_diagnostic": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "summary_path": str(out_prefix.with_suffix(".json")),
            "seq_list": args.seq_list,
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=root, pred_suffix=args.pred_suffix)

    if args.copy_fixed_support and args.support_mode == "fixed":
        for scene in scenes:
            shutil.copyfile(_tmp_path(root, args.pre_points_config, scene), output_tmp_dir / f"{scene}_pre_points.npy")

    print(json.dumps(_json_safe({"aggregate": aggregate, "summary": str(out_prefix.with_suffix(".json"))}), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--pool-name", default="")
    parser.add_argument("--pool-configs", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--pred-suffix", default="class_agnostic")
    parser.add_argument("--support-mode", choices=["union", "fixed"], default="union")
    parser.add_argument("--pre-points-config", default="")
    parser.add_argument("--min-candidate-points", type=int, default=100)
    parser.add_argument("--min-select-iou", type=float, default=0.25)
    parser.add_argument("--dedup-threshold", type=float, default=0.95)
    parser.add_argument("--dedup-overlap-mode", choices=["iou", "min_ioc", "candidate_ioc"], default="min_ioc")
    parser.add_argument("--summary-root", default="outputs/audit/v11_candidate_oracle")
    parser.add_argument("--copy-fixed-support", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
