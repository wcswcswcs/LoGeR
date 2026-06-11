from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.self_discovered_boundary_refine import ProjectionCache, _json_safe, _sample_ids


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _prediction_path(root: Path, config: str, suffix: str, seq_name: str) -> Path:
    return root / "data" / "prediction" / f"{config}{suffix}" / f"{seq_name}.npz"


def _tmp_path(root: Path, config: str, seq_name: str) -> Path:
    return root / "data" / "TMP" / config / f"{seq_name}_pre_points.npy"


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _discover_mask(
    cache: ProjectionCache,
    frame_id: int,
    point_ids: np.ndarray,
    args: argparse.Namespace,
) -> tuple[int, dict[str, float]] | None:
    xy, z, _ = cache.project(frame_id, point_ids)
    if xy.size == 0:
        return None
    depth = cache.depth(frame_id)
    mask = cache.mask(frame_id)
    obs_depth = depth[xy[:, 1], xy[:, 0]]
    projected = np.isfinite(obs_depth) & (obs_depth > 0.0)
    visible = projected & (np.abs(obs_depth - z) <= float(args.depth_tolerance))
    visible_count = int(np.count_nonzero(visible))
    if visible_count < int(args.min_visible_points):
        return None
    labels, counts = np.unique(mask[xy[visible, 1], xy[visible, 0]], return_counts=True)
    keep = labels > 0
    labels = labels[keep]
    counts = counts[keep]
    if labels.size == 0:
        return None
    best_pos = int(np.argmax(counts))
    best_label = int(labels[best_pos])
    best_count = int(counts[best_pos])
    dominant_ratio = best_count / max(float(visible_count), 1.0)
    if best_count < int(args.min_dominant_points) or dominant_ratio < float(args.min_dominant_ratio):
        return None
    return best_label, {
        "visible": float(visible_count),
        "dominant_count": float(best_count),
        "dominant_ratio": float(dominant_ratio),
    }


def _score_instance(
    cache: ProjectionCache,
    frame_ids: list[int],
    point_ids: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    if point_ids.size == 0:
        return {
            "self_silhouette_quality": 0.0,
            "inside_visible_ratio": 0.0,
            "interior_ratio": 0.0,
            "used_observations": 0.0,
            "visible_points": 0.0,
            "dominant_ratio_mean": 0.0,
        }
    sampled_for_discovery = _sample_ids(point_ids, int(args.discovery_max_points))
    sampled_for_scoring = _sample_ids(point_ids, int(args.score_max_points))

    used_observations = 0
    visible_total = 0
    inside_total = 0
    interior_total = 0
    dominant_ratios: list[float] = []
    for frame_id in frame_ids:
        discovered = _discover_mask(cache, frame_id, sampled_for_discovery, args)
        if discovered is None:
            continue
        mask_id, discover_diag = discovered
        xy, z, _ = cache.project(frame_id, sampled_for_scoring)
        if xy.size == 0:
            continue
        depth = cache.depth(frame_id)
        mask = cache.mask(frame_id)
        obs_depth = depth[xy[:, 1], xy[:, 0]]
        projected = np.isfinite(obs_depth) & (obs_depth > 0.0)
        visible = projected & (np.abs(obs_depth - z) <= float(args.depth_tolerance))
        visible_count = int(np.count_nonzero(visible))
        if visible_count < int(args.min_visible_points):
            continue
        inside = visible & (mask[xy[:, 1], xy[:, 0]] == int(mask_id))
        inside_count = int(np.count_nonzero(inside))
        interior_count = 0
        if inside_count:
            distance = cache.distance_inside_mask(frame_id, int(mask_id))
            margins = distance[xy[inside, 1], xy[inside, 0]]
            interior_count = int(np.count_nonzero(margins >= float(args.boundary_margin_px)))
        used_observations += 1
        visible_total += visible_count
        inside_total += inside_count
        interior_total += interior_count
        dominant_ratios.append(float(discover_diag["dominant_ratio"]))
        if int(args.max_observations) > 0 and used_observations >= int(args.max_observations):
            break

    inside_visible_ratio = inside_total / max(float(visible_total), 1.0)
    interior_ratio = interior_total / max(float(inside_total), 1.0)
    observation_weight = min(1.0, used_observations / max(float(args.observation_saturation), 1.0))
    visible_weight = min(1.0, visible_total / max(float(args.visible_saturation), 1.0))
    dominant_quality = float(np.mean(dominant_ratios)) if dominant_ratios else 0.0
    quality = (
        inside_visible_ratio
        * (0.5 + 0.5 * interior_ratio)
        * (0.5 + 0.5 * observation_weight)
        * (0.5 + 0.5 * visible_weight)
        * (0.5 + 0.5 * dominant_quality)
    )
    return {
        "self_silhouette_quality": float(quality),
        "inside_visible_ratio": float(inside_visible_ratio),
        "interior_ratio": float(interior_ratio),
        "used_observations": float(used_observations),
        "visible_points": float(visible_total),
        "dominant_ratio_mean": float(dominant_quality),
    }


def _compose_scores(
    source_scores: np.ndarray,
    areas: np.ndarray,
    quality: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    score_norm = _normalize(source_scores)
    area_norm = _normalize(np.log1p(np.maximum(areas.astype(np.float64), 0.0)))
    if args.quality_mode == "self_silhouette":
        scores = quality
    elif args.quality_mode == "score_self_silhouette":
        scores = float(args.score_weight) * score_norm + (1.0 - float(args.score_weight)) * quality
    elif args.quality_mode == "score_self_silhouette_area":
        remain = max(0.0, 1.0 - float(args.score_weight) - float(args.silhouette_weight))
        scores = float(args.score_weight) * score_norm + float(args.silhouette_weight) * quality + remain * area_norm
    elif args.quality_mode == "self_silhouette_area":
        scores = float(args.silhouette_weight) * quality + (1.0 - float(args.silhouette_weight)) * area_norm
    else:
        raise ValueError(f"Unsupported quality mode: {args.quality_mode}")
    return scores.astype(np.float32)


def process_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, Any]:
    root = Path(args.root)
    pred_path = _prediction_path(root, args.input_config, args.pred_suffix, seq_name)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction: {pred_path}")
    with np.load(pred_path) as data:
        masks = data["pred_masks"].astype(bool, copy=False)
        source_scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)
    if masks.ndim != 2:
        raise ValueError(f"{seq_name}: pred_masks must be 2D")
    if source_scores.shape[0] != masks.shape[1] or classes.shape[0] != masks.shape[1]:
        raise ValueError(f"{seq_name}: inconsistent prediction arrays")

    stream = ScanNetStream(seq_name=seq_name, backbone=args.backbone, root=root / args.scannet_root)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    scene_points = np.asarray(o3d.io.read_point_cloud(str(stream.mesh_path)).points, dtype=np.float32)
    if scene_points.shape[0] != masks.shape[0]:
        raise RuntimeError(f"{seq_name}: mesh points {scene_points.shape[0]} != prediction rows {masks.shape[0]}")
    cache = ProjectionCache(stream, scene_points)
    frame_ids = stream.frame_ids(stride=int(args.frame_stride), max_frames=int(args.max_frames))

    support_ids: np.ndarray | None = None
    if args.score_support_config:
        support_path = _tmp_path(root, args.score_support_config, seq_name)
        if not support_path.exists():
            raise FileNotFoundError(f"Missing score support pre_points: {support_path}")
        support_ids = np.load(support_path).astype(np.int64)
        support_ids = support_ids[(support_ids >= 0) & (support_ids < masks.shape[0])]

    records: list[dict[str, float]] = []
    point_areas: list[float] = []
    for idx in range(masks.shape[1]):
        point_ids = np.flatnonzero(masks[:, idx]).astype(np.int64)
        if support_ids is not None:
            point_ids = np.intersect1d(point_ids, support_ids, assume_unique=False)
        point_areas.append(float(point_ids.shape[0]))
        records.append(_score_instance(cache, frame_ids, point_ids, args))

    quality = np.asarray([record["self_silhouette_quality"] for record in records], dtype=np.float32)
    areas = np.asarray(point_areas, dtype=np.float32)
    scores = _compose_scores(source_scores, areas, quality, args)

    keep = np.ones((masks.shape[1],), dtype=bool)
    if float(args.min_self_silhouette_quality) > 0.0:
        keep &= quality >= float(args.min_self_silhouette_quality)
    if int(args.min_visible_points_total) > 0:
        visible = np.asarray([record["visible_points"] for record in records], dtype=np.float32)
        keep &= visible >= int(args.min_visible_points_total)
    if int(args.min_support_area) > 0:
        keep &= areas >= int(args.min_support_area)
    if int(args.max_instances) > 0 and int(np.count_nonzero(keep)) > int(args.max_instances):
        keep_indices = np.flatnonzero(keep)
        order = keep_indices[np.argsort(-scores[keep_indices], kind="stable")[: int(args.max_instances)]]
        new_keep = np.zeros_like(keep)
        new_keep[order] = True
        keep = new_keep

    masks_out = masks[:, keep]
    scores_out = scores[keep]
    classes_out = classes[keep]
    order = np.argsort(-scores_out, kind="stable")
    masks_out = masks_out[:, order]
    scores_out = scores_out[order]
    classes_out = classes_out[order]

    pred_out_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_out_dir / f"{seq_name}.npz",
        pred_masks=masks_out,
        pred_score=scores_out.astype(np.float32, copy=False),
        pred_classes=classes_out.astype(np.int32, copy=False),
    )

    tmp_in = _tmp_path(root, args.input_config, seq_name)
    tmp_out = _tmp_path(root, args.output_config, seq_name)
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    if tmp_in.exists():
        shutil.copy2(tmp_in, tmp_out)
    else:
        np.save(tmp_out, np.flatnonzero(np.any(masks_out, axis=1)).astype(np.int64))

    return {
        "seq_name": seq_name,
        "input_config": args.input_config,
        "output_config": args.output_config,
        "num_instances_before": int(masks.shape[1]),
        "num_instances_after": int(masks_out.shape[1]),
        "num_removed": int(masks.shape[1] - masks_out.shape[1]),
        "self_silhouette_quality_min": float(np.min(quality)) if quality.size else 0.0,
        "self_silhouette_quality_mean": float(np.mean(quality)) if quality.size else 0.0,
        "self_silhouette_quality_max": float(np.max(quality)) if quality.size else 0.0,
        "inside_visible_ratio_mean": float(np.mean([item["inside_visible_ratio"] for item in records])) if records else 0.0,
        "interior_ratio_mean": float(np.mean([item["interior_ratio"] for item in records])) if records else 0.0,
        "used_observations_mean": float(np.mean([item["used_observations"] for item in records])) if records else 0.0,
        "visible_points_mean": float(np.mean([item["visible_points"] for item in records])) if records else 0.0,
        "dominant_ratio_mean": float(np.mean([item["dominant_ratio_mean"] for item in records])) if records else 0.0,
        "score_min": float(np.min(scores_out)) if scores_out.size else 0.0,
        "score_mean": float(np.mean(scores_out)) if scores_out.size else 0.0,
        "score_max": float(np.max(scores_out)) if scores_out.size else 0.0,
    }


def aggregate(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, float | str]:
    numeric_keys = sorted(
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float))
    )
    means: dict[str, float] = {}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if vals:
            means[f"mean_{key}"] = float(np.mean(vals))
    return {
        "input_config": args.input_config,
        "output_config": args.output_config,
        "quality_mode": args.quality_mode,
        "score_weight": float(args.score_weight),
        "silhouette_weight": float(args.silhouette_weight),
        "min_self_silhouette_quality": float(args.min_self_silhouette_quality),
        "score_support_config": args.score_support_config,
        "scenes": len(rows),
        **means,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score final predictions by self-discovered 2D silhouette agreement.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--score-support-config", default="")
    parser.add_argument(
        "--quality-mode",
        default="score_self_silhouette",
        choices=["self_silhouette", "score_self_silhouette", "self_silhouette_area", "score_self_silhouette_area"],
    )
    parser.add_argument("--score-weight", type=float, default=0.90)
    parser.add_argument("--silhouette-weight", type=float, default=0.08)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--max-observations", type=int, default=8)
    parser.add_argument("--discovery-max-points", type=int, default=800)
    parser.add_argument("--score-max-points", type=int, default=1600)
    parser.add_argument("--depth-tolerance", type=float, default=0.08)
    parser.add_argument("--boundary-margin-px", type=float, default=2.0)
    parser.add_argument("--min-visible-points", type=int, default=5)
    parser.add_argument("--min-dominant-points", type=int, default=5)
    parser.add_argument("--min-dominant-ratio", type=float, default=0.35)
    parser.add_argument("--visible-saturation", type=float, default=200.0)
    parser.add_argument("--observation-saturation", type=float, default=4.0)
    parser.add_argument("--min-self-silhouette-quality", type=float, default=0.0)
    parser.add_argument("--min-visible-points-total", type=int, default=0)
    parser.add_argument("--min-support-area", type=int, default=0)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--summary-root", default="outputs/self_discovered_silhouette_score_v4_1")
    parser.add_argument("--eval-policy", default="self_discovered_silhouette_score")
    parser.add_argument("--diagnostic-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    rows = [process_sequence(args, seq_name) for seq_name in _read_seq_list(root / args.seq_list)]
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"args": vars(args), "aggregate": aggregate(rows, args), "rows": rows}
    out_path = out_dir / f"{args.output_config}_summary.json"
    out_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    source_configs = [args.input_config]
    if args.score_support_config:
        source_configs.append(args.score_support_config)
    manifest = build_prediction_manifest(
        root=args.root,
        output_config=args.output_config,
        is_method_result=not bool(args.diagnostic_only),
        is_diagnostic_only=bool(args.diagnostic_only),
        uses_gt=False,
        gt_usage="none",
        source_configs=source_configs,
        pre_points_policy="input_tmp_copy",
        support_policy=(
            f"self_discovered_silhouette:{args.quality_mode}:"
            f"frame_stride={args.frame_stride}:max_frames={args.max_frames}"
        ),
        notes=(
            "Scores predicted object masks by agreement with self-discovered non-GT 2D "
            "mask silhouettes and RGB-D visibility; no GT is read."
        ),
        extra={
            "algorithm": "self_discovered_silhouette_score",
            "eval_policy": args.eval_policy,
            "input_config": args.input_config,
            "score_support_config": args.score_support_config,
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=args.root, pred_suffix=args.pred_suffix.lstrip("_"))
    print(f"[self-discovered-silhouette-score] wrote {out_path}")


if __name__ == "__main__":
    main()
