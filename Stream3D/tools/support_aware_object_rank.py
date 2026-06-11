from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _tmp_path(root: Path, config: str, seq_name: str) -> Path:
    return root / "data" / "TMP" / config / f"{seq_name}_pre_points.npy"


def _prediction_path(root: Path, config: str, suffix: str, seq_name: str) -> Path:
    return root / "data" / "prediction" / f"{config}{suffix}" / f"{seq_name}.npz"


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32, copy=False)


def _support_features(masks: np.ndarray, support_ids: np.ndarray) -> dict[str, np.ndarray]:
    support_ids = support_ids.astype(np.int64, copy=False)
    if support_ids.size == 0:
        support_masks = np.zeros((0, masks.shape[1]), dtype=bool)
    else:
        if int(support_ids.min()) < 0 or int(support_ids.max()) >= masks.shape[0]:
            raise ValueError(
                f"support pre_points out of prediction vertex range: "
                f"min={int(support_ids.min())}, max={int(support_ids.max())}, vertices={masks.shape[0]}"
            )
        support_masks = masks[support_ids, :]
    support_area = support_masks.sum(axis=0).astype(np.float64)
    full_area = masks.sum(axis=0).astype(np.float64)
    owner_counts = support_masks.sum(axis=1) if support_masks.shape[0] else np.zeros((0,), dtype=np.int64)
    conflict_area = np.zeros((masks.shape[1],), dtype=np.float64)
    unique_area = np.zeros((masks.shape[1],), dtype=np.float64)
    if support_masks.shape[0]:
        conflict_points = owner_counts > 1
        unique_points = owner_counts == 1
        conflict_area = support_masks[conflict_points, :].sum(axis=0).astype(np.float64)
        unique_area = support_masks[unique_points, :].sum(axis=0).astype(np.float64)
    support_area_safe = np.maximum(support_area, 1.0)
    return {
        "support_area": support_area,
        "full_area": full_area,
        "support_fraction": support_area / np.maximum(full_area, 1.0),
        "conflict_ratio": conflict_area / support_area_safe,
        "unique_ratio": unique_area / support_area_safe,
    }


def _quality(scores: np.ndarray, features: dict[str, np.ndarray], mode: str, score_weight: float) -> np.ndarray:
    support_area = features["support_area"]
    support_area_norm = _normalize(np.log1p(np.maximum(support_area, 0.0)))
    full_area_norm = _normalize(np.log1p(np.maximum(features["full_area"], 0.0)))
    unique_ratio = features["unique_ratio"].astype(np.float32, copy=False)
    conflict_ratio = features["conflict_ratio"].astype(np.float32, copy=False)
    support_fraction = features["support_fraction"].astype(np.float32, copy=False)
    score_norm = _normalize(scores.astype(np.float64, copy=False))

    if mode == "support_area":
        quality = support_area_norm
    elif mode == "support_area_unique":
        quality = support_area_norm * (0.25 + unique_ratio)
    elif mode == "support_area_conflict_penalty":
        quality = support_area_norm * (1.0 - 0.5 * conflict_ratio)
    elif mode == "support_fraction":
        quality = support_fraction
    elif mode == "support_area_fraction":
        quality = support_area_norm * (0.25 + support_fraction)
    elif mode == "score_support_area":
        quality = float(score_weight) * score_norm + (1.0 - float(score_weight)) * support_area_norm
    elif mode == "score_support_area_conflict_penalty":
        area_quality = support_area_norm * (1.0 - 0.5 * conflict_ratio)
        quality = float(score_weight) * score_norm + (1.0 - float(score_weight)) * area_quality
    elif mode == "full_area":
        quality = full_area_norm
    else:
        raise ValueError(f"Unsupported quality mode: {mode}")
    return np.asarray(quality, dtype=np.float32)


def _overlap_with_kept(
    kept_masks: np.ndarray,
    kept_counts: np.ndarray,
    mask: np.ndarray,
    mask_count: float,
    mode: str,
) -> float:
    if kept_masks.shape[1] == 0 or mask_count <= 0:
        return 0.0
    intersections = np.logical_and(kept_masks, mask[:, None]).sum(axis=0).astype(np.float64)
    if mode == "iou":
        denom = kept_counts + mask_count - intersections
    elif mode == "candidate_ioc":
        denom = np.full_like(intersections, mask_count, dtype=np.float64)
    elif mode == "min_ioc":
        denom = np.minimum(kept_counts, mask_count)
    else:
        raise ValueError(f"Unsupported overlap mode: {mode}")
    overlaps = intersections / np.maximum(denom, 1.0)
    return float(np.max(overlaps)) if overlaps.size else 0.0


def _apply_competition(
    masks: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    support_area: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    keep_area = support_area >= float(args.min_support_area)
    candidate_indices = np.flatnonzero(keep_area)
    order = sorted(candidate_indices.tolist(), key=lambda idx: (-float(scores[idx]), -float(support_area[idx]), idx))
    kept: list[int] = []
    suppressed_by_overlap = 0
    suppressed_by_area = int(np.count_nonzero(~keep_area))
    counts = masks.sum(axis=0).astype(np.float64)
    for idx in order:
        if kept and float(args.overlap_threshold) > 0.0:
            kept_arr = np.asarray(kept, dtype=np.int64)
            overlap = _overlap_with_kept(
                masks[:, kept_arr],
                counts[kept_arr],
                masks[:, idx],
                float(counts[idx]),
                args.overlap_mode,
            )
            if overlap >= float(args.overlap_threshold):
                suppressed_by_overlap += 1
                continue
        kept.append(idx)
        if int(args.max_instances) > 0 and len(kept) >= int(args.max_instances):
            break

    kept_arr = np.asarray(kept, dtype=np.int64)
    return masks[:, kept_arr], scores[kept_arr], classes[kept_arr], {
        "num_suppressed_by_min_support_area": float(suppressed_by_area),
        "num_suppressed_by_overlap": float(suppressed_by_overlap),
        "num_instances_after_competition": float(kept_arr.shape[0]),
    }


def process_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, float | str]:
    root = Path(args.root)
    pred_in = _prediction_path(root, args.input_config, args.pred_suffix, seq_name)
    if not pred_in.exists():
        raise FileNotFoundError(f"Missing prediction: {pred_in}")
    with np.load(pred_in) as data:
        masks = data["pred_masks"].astype(bool, copy=False)
        source_scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)
    if masks.ndim != 2:
        raise ValueError(f"{seq_name}: pred_masks must be 2D, got {masks.shape}")
    if source_scores.shape[0] != masks.shape[1] or classes.shape[0] != masks.shape[1]:
        raise ValueError(f"{seq_name}: prediction arrays have inconsistent instance counts")

    support_path = _tmp_path(root, args.score_pre_points_config, seq_name)
    if not support_path.exists():
        raise FileNotFoundError(f"Missing score support pre_points: {support_path}")
    support_ids = np.load(support_path).astype(np.int64)
    features = _support_features(masks, support_ids)
    quality = _quality(source_scores, features, args.quality_mode, args.score_weight)
    scores = quality.astype(np.float32, copy=False)
    if args.preserve_empty_scores:
        scores = scores.copy()
        scores[features["support_area"] <= 0] = np.minimum(scores[features["support_area"] <= 0], -1.0)

    diag = {
        "num_instances_before": float(masks.shape[1]),
        "num_score_support_points": float(support_ids.shape[0]),
        "support_area_min": float(np.min(features["support_area"])) if features["support_area"].size else 0.0,
        "support_area_mean": float(np.mean(features["support_area"])) if features["support_area"].size else 0.0,
        "support_area_max": float(np.max(features["support_area"])) if features["support_area"].size else 0.0,
        "quality_min": float(np.min(scores)) if scores.size else 0.0,
        "quality_mean": float(np.mean(scores)) if scores.size else 0.0,
        "quality_max": float(np.max(scores)) if scores.size else 0.0,
        "mean_conflict_ratio": float(np.mean(features["conflict_ratio"])) if features["conflict_ratio"].size else 0.0,
        "mean_unique_ratio": float(np.mean(features["unique_ratio"])) if features["unique_ratio"].size else 0.0,
    }

    if float(args.overlap_threshold) > 0.0 or int(args.min_support_area) > 0 or int(args.max_instances) > 0:
        masks, scores, classes, comp_diag = _apply_competition(
            masks=masks,
            scores=scores,
            classes=classes,
            support_area=features["support_area"],
            args=args,
        )
        diag.update(comp_diag)
    else:
        diag.update(
            {
                "num_suppressed_by_min_support_area": 0.0,
                "num_suppressed_by_overlap": 0.0,
                "num_instances_after_competition": float(masks.shape[1]),
            }
        )

    pred_out_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_out_dir / f"{seq_name}.npz",
        pred_masks=masks,
        pred_score=scores,
        pred_classes=classes,
    )

    tmp_out_dir = root / "data" / "TMP" / args.output_config
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = tmp_out_dir / f"{seq_name}_pre_points.npy"
    tmp_in = _tmp_path(root, args.input_config, seq_name)
    if args.tmp_policy == "input" and tmp_in.exists():
        shutil.copy2(tmp_in, tmp_out)
        tmp_mode = "copied_input_tmp"
    else:
        np.save(tmp_out, np.flatnonzero(np.any(masks, axis=1)).astype(np.int64))
        tmp_mode = "recomputed"

    return {
        "seq_name": seq_name,
        "input_config": args.input_config,
        "output_config": args.output_config,
        "score_pre_points_config": args.score_pre_points_config,
        "quality_mode": args.quality_mode,
        "score_weight": float(args.score_weight),
        "overlap_threshold": float(args.overlap_threshold),
        "overlap_mode": args.overlap_mode,
        "min_support_area": int(args.min_support_area),
        "max_instances": int(args.max_instances),
        "tmp_mode": tmp_mode,
        "output_union_count": float(np.count_nonzero(np.any(masks, axis=1))),
        **diag,
    }


def aggregate(rows: list[dict[str, float | str]], args: argparse.Namespace) -> dict[str, float | str]:
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
        "score_pre_points_config": args.score_pre_points_config,
        "quality_mode": args.quality_mode,
        "score_weight": float(args.score_weight),
        "overlap_threshold": float(args.overlap_threshold),
        "overlap_mode": args.overlap_mode,
        "scenes": len(rows),
        **means,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--score-pre-points-config", required=True)
    parser.add_argument(
        "--quality-mode",
        default="support_area",
        choices=[
            "support_area",
            "support_area_unique",
            "support_area_conflict_penalty",
            "support_fraction",
            "support_area_fraction",
            "score_support_area",
            "score_support_area_conflict_penalty",
            "full_area",
        ],
    )
    parser.add_argument("--score-weight", type=float, default=0.25)
    parser.add_argument("--preserve-empty-scores", action="store_true")
    parser.add_argument("--overlap-threshold", type=float, default=0.0)
    parser.add_argument("--overlap-mode", default="min_ioc", choices=["iou", "candidate_ioc", "min_ioc"])
    parser.add_argument("--min-support-area", type=int, default=0)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--summary-root", default="outputs/stream4d_support_aware_rank_v4_1")
    parser.add_argument("--tmp-policy", default="input", choices=["input", "recompute"])
    parser.add_argument("--eval-policy", default="support_aware_object_rank")
    parser.add_argument("--diagnostic-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = [process_sequence(args, seq_name) for seq_name in _read_seq_list(Path(args.seq_list))]
    root = Path(args.root)
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"args": vars(args), "aggregate": aggregate(rows, args), "rows": rows}
    out_path = out_dir / f"{args.output_config}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = build_prediction_manifest(
        root=args.root,
        output_config=args.output_config,
        is_method_result=not bool(args.diagnostic_only),
        is_diagnostic_only=bool(args.diagnostic_only),
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.input_config, args.score_pre_points_config],
        pre_points_policy=args.tmp_policy,
        support_policy=f"support_aware_rank:{args.quality_mode}:overlap={args.overlap_mode}:{args.overlap_threshold}",
        notes="Ranks and suppresses predicted object masks using only prediction/support overlap statistics; no GT is read.",
        extra={
            "algorithm": "support_aware_object_rank",
            "eval_policy": args.eval_policy,
            "input_config": args.input_config,
            "score_pre_points_config": args.score_pre_points_config,
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=args.root, pred_suffix=args.pred_suffix.lstrip("_"))
    print(f"[support-aware-object-rank] wrote {out_path}")


if __name__ == "__main__":
    main()
