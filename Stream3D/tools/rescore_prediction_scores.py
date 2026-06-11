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


def _normalize_feature(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    max_value = float(np.max(values)) if values.size else 0.0
    min_value = float(np.min(values)) if values.size else 0.0
    if max_value <= min_value:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32, copy=False)


def _score_feature(areas: np.ndarray, scores: np.ndarray, mode: str) -> np.ndarray:
    values = areas.astype(np.float32, copy=False)
    if mode == "none":
        return np.zeros_like(values, dtype=np.float32)
    if mode == "area":
        feature = values
    elif mode == "sqrt_area":
        feature = np.sqrt(np.maximum(values, 0.0))
    elif mode == "log_area":
        feature = np.log1p(np.maximum(values, 0.0))
    elif mode == "source_score":
        feature = scores.astype(np.float32, copy=False)
    elif mode == "inverse_area":
        feature = -values
    elif mode == "inverse_sqrt_area":
        feature = -np.sqrt(np.maximum(values, 0.0))
    elif mode == "inverse_log_area":
        feature = -np.log1p(np.maximum(values, 0.0))
    else:
        raise ValueError(f"Unsupported score feature mode: {mode}")
    return _normalize_feature(feature)


def _tmp_path(root: Path, config: str, seq_name: str) -> Path:
    return root / "data" / "TMP" / config / f"{seq_name}_pre_points.npy"


def rescore_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, float | str]:
    root = Path(args.root)
    pred_in = root / "data" / "prediction" / f"{args.input_config}{args.pred_suffix}" / f"{seq_name}.npz"
    if not pred_in.exists():
        raise FileNotFoundError(f"Missing prediction: {pred_in}")
    with np.load(pred_in) as data:
        masks = data["pred_masks"].astype(bool, copy=False)
        scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)

    if masks.ndim != 2:
        raise ValueError(f"{seq_name}: pred_masks must be 2D, got {masks.shape}")
    if scores.shape[0] != masks.shape[1]:
        raise ValueError(f"{seq_name}: pred_score length does not match masks: {scores.shape} vs {masks.shape}")

    areas = masks.sum(axis=0).astype(np.float32)
    keep = np.ones((masks.shape[1],), dtype=bool)
    if int(args.min_area) > 0:
        keep &= areas >= float(args.min_area)
    if int(args.max_area) > 0:
        keep &= areas <= float(args.max_area)
    if not np.all(keep):
        masks = masks[:, keep]
        scores = scores[keep]
        classes = classes[keep]
        areas = areas[keep]

    feature = _score_feature(areas, scores, args.score_feature)
    if args.base_score_mode == "preserve":
        base_scores = scores.copy()
    elif args.base_score_mode == "constant":
        base_scores = np.full_like(scores, float(args.constant_score), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported base score mode: {args.base_score_mode}")
    new_scores = (base_scores + float(args.tiebreaker_weight) * feature).astype(np.float32, copy=False)

    pred_out_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_out_dir / f"{seq_name}.npz",
        pred_masks=masks,
        pred_score=new_scores,
        pred_classes=classes,
    )

    tmp_in = _tmp_path(root, args.input_config, seq_name)
    tmp_out_dir = root / "data" / "TMP" / args.output_config
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = tmp_out_dir / f"{seq_name}_pre_points.npy"
    if tmp_in.exists():
        shutil.copy2(tmp_in, tmp_out)
        tmp_mode = "copied_input_tmp"
    else:
        pre_points = np.flatnonzero(np.any(masks, axis=1)).astype(np.int64)
        np.save(tmp_out, pre_points)
        tmp_mode = "recomputed_missing_input_tmp"

    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.input_config],
        pre_points_policy=tmp_mode,
        support_policy=(
            f"rescore_prediction_scores:base={args.base_score_mode}:"
            f"feature={args.score_feature}:weight={args.tiebreaker_weight}:"
            f"area_filter={args.min_area}-{args.max_area}"
        ),
        notes="Prediction score calibration from prediction mask statistics only; no GT used.",
        extra={
            "input_config": args.input_config,
            "eval_policy": args.eval_policy,
            "score_feature": args.score_feature,
            "base_score_mode": args.base_score_mode,
            "constant_score": float(args.constant_score),
            "tiebreaker_weight": float(args.tiebreaker_weight),
            "min_area": int(args.min_area),
            "max_area": int(args.max_area),
        },
    )
    write_prediction_manifest(
        args.output_config,
        manifest,
        root=root,
        pred_suffix=args.pred_suffix.lstrip("_"),
    )

    return {
        "seq_name": seq_name,
        "input_config": args.input_config,
        "output_config": args.output_config,
        "num_instances": float(masks.shape[1]),
        "num_instances_before_filter": float(keep.shape[0]),
        "num_instances_removed_by_area": float(keep.shape[0] - np.count_nonzero(keep)),
        "min_area_filter": int(args.min_area),
        "max_area_filter": int(args.max_area),
        "score_feature": args.score_feature,
        "base_score_mode": args.base_score_mode,
        "tiebreaker_weight": float(args.tiebreaker_weight),
        "tmp_mode": tmp_mode,
        "area_min": float(np.min(areas)) if areas.size else 0.0,
        "area_mean": float(np.mean(areas)) if areas.size else 0.0,
        "area_max": float(np.max(areas)) if areas.size else 0.0,
        "score_min_before": float(np.min(scores)) if scores.size else 0.0,
        "score_max_before": float(np.max(scores)) if scores.size else 0.0,
        "score_min_after": float(np.min(new_scores)) if new_scores.size else 0.0,
        "score_max_after": float(np.max(new_scores)) if new_scores.size else 0.0,
    }


def aggregate(rows: list[dict[str, float | str]], args: argparse.Namespace) -> dict[str, float | str]:
    numeric_keys = sorted(
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float))
    )
    means = {}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if vals:
            means[f"mean_{key}"] = float(np.mean(vals))
    return {
        "input_config": args.input_config,
        "output_config": args.output_config,
        "score_feature": args.score_feature,
        "base_score_mode": args.base_score_mode,
        "tiebreaker_weight": float(args.tiebreaker_weight),
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
    parser.add_argument(
        "--score-feature",
        default="log_area",
        choices=[
            "none",
            "area",
            "sqrt_area",
            "log_area",
            "source_score",
            "inverse_area",
            "inverse_sqrt_area",
            "inverse_log_area",
        ],
    )
    parser.add_argument("--base-score-mode", default="preserve", choices=["preserve", "constant"])
    parser.add_argument("--constant-score", type=float, default=1.0)
    parser.add_argument("--tiebreaker-weight", type=float, default=0.01)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--max-area", type=int, default=0)
    parser.add_argument("--eval-policy", default="own_recompute_score_calibration")
    parser.add_argument("--summary-root", default="outputs/stream4d_score_calibration_v4_1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    rows = [rescore_sequence(args, seq_name) for seq_name in _read_seq_list(Path(args.seq_list))]
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"args": vars(args), "aggregate": aggregate(rows, args), "rows": rows}
    out_path = out_dir / f"{args.output_config}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[rescore-prediction-scores] wrote {out_path}")


if __name__ == "__main__":
    main()
