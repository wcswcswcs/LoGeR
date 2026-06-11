from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stream4d.diagnostics import write_json
from stream4d.rescore_scannet import (
    _score_object,
    _seq_names,
    verify_object_dict_prediction_alignment,
)
from stream4d.scannet_stream import ScanNetStream


def _point_ids(value: dict, num_points: int) -> np.ndarray:
    ids = np.asarray(value.get("point_ids", []), dtype=np.int64).reshape(-1)
    ids = ids[(ids >= 0) & (ids < int(num_points))]
    if ids.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.unique(ids)


def _overlap(a: np.ndarray, b: np.ndarray, mode: str) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = int(np.intersect1d(a, b, assume_unique=True).shape[0])
    if mode == "ioc":
        denom = max(1, min(int(a.size), int(b.size)))
    elif mode == "iou":
        denom = max(1, int(a.size + b.size - inter))
    else:
        raise ValueError(f"Unsupported overlap mode: {mode}")
    return float(inter / denom)


def _nms(
    object_items: list[tuple[int, dict]],
    num_points: int,
    scores: np.ndarray,
    areas: np.ndarray,
    min_points: int,
    threshold: float,
    mode: str,
    max_instances: int,
) -> tuple[list[int], list[dict]]:
    point_sets = [_point_ids(value, num_points=num_points) for _, value in object_items]
    candidates = [idx for idx, area in enumerate(areas.tolist()) if float(area) >= float(min_points)]
    order = sorted(candidates, key=lambda idx: (float(scores[idx]), int(areas[idx]), -idx), reverse=True)
    kept: list[int] = []
    suppressed_records: list[dict] = []
    for idx in order:
        best_overlap = 0.0
        best_kept = None
        for kept_idx in kept:
            overlap = _overlap(point_sets[idx], point_sets[kept_idx], mode=mode)
            if overlap > best_overlap:
                best_overlap = overlap
                best_kept = kept_idx
        if best_overlap >= float(threshold):
            suppressed_records.append(
                {
                    "suppressed_object_id": int(object_items[idx][0]),
                    "kept_object_id": int(object_items[best_kept][0]) if best_kept is not None else None,
                    "overlap": float(best_overlap),
                    "score": float(scores[idx]),
                    "area": float(areas[idx]),
                }
            )
            continue
        kept.append(idx)
        if max_instances > 0 and len(kept) >= int(max_instances):
            break
    return sorted(kept), suppressed_records


def _pre_points_input_path(args: argparse.Namespace, seq_name: str) -> Path:
    if args.pre_points_policy == "fixed_path":
        return Path(args.fixed_pre_points_root) / args.fixed_pre_points_config / f"{seq_name}_pre_points.npy"
    return Path("data/TMP") / args.input_config / f"{seq_name}_pre_points.npy"


def _write_pre_points(
    args: argparse.Namespace,
    seq_name: str,
    input_pre_points_path: Path,
    tmp_out_dir: Path,
    pred_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    input_pre_points = np.load(input_pre_points_path).astype(np.int64)
    prediction_union = np.flatnonzero(pred_masks.any(axis=1)).astype(np.int64)
    if args.pre_points_policy == "recompute":
        output_pre_points = prediction_union
    elif args.pre_points_policy in {"inherit", "fixed_path"}:
        output_pre_points = input_pre_points
    else:
        raise ValueError(f"Unsupported pre_points_policy: {args.pre_points_policy}")
    np.save(tmp_out_dir / f"{seq_name}_pre_points.npy", output_pre_points)
    return output_pre_points, prediction_union, int(input_pre_points.shape[0])


def _process_sequence(args: argparse.Namespace, seq_name: str) -> dict:
    stream = ScanNetStream(seq_name=seq_name, backbone=args.backbone)
    pred_in = Path("data/prediction") / f"{args.input_config}_class_agnostic" / f"{seq_name}.npz"
    object_in = stream.object_dir / args.input_config / "object_dict.npy"
    tmp_in = _pre_points_input_path(args, seq_name)
    if not pred_in.exists():
        raise FileNotFoundError(f"Missing prediction: {pred_in}")
    if not object_in.exists():
        raise FileNotFoundError(f"Missing object_dict: {object_in}")
    if not tmp_in.exists():
        raise FileNotFoundError(f"Missing pre_points: {tmp_in}")

    pred = np.load(pred_in)
    pred_masks_in = pred["pred_masks"]
    object_dict = np.load(object_in, allow_pickle=True).item()
    object_items = [(int(k), v) for k, v in sorted(object_dict.items(), key=lambda item: int(item[0]))]
    if pred_masks_in.ndim != 2:
        raise ValueError(f"pred_masks must be 2D, got {pred_masks_in.shape}")
    if pred_masks_in.shape[1] != len(object_items):
        raise RuntimeError(
            f"{seq_name}: pred columns {pred_masks_in.shape[1]} do not match object count {len(object_items)}"
        )
    alignment = verify_object_dict_prediction_alignment(
        pred_masks_in,
        object_items,
        threshold=float(args.alignment_iou_threshold),
    )
    if alignment["alignment_checked"] and alignment["alignment_failed_instances"] > 0:
        raise RuntimeError(f"{seq_name}: object_dict/pred alignment failed: {alignment}")

    scores = np.asarray([_score_object(value, args.select_mode) for _, value in object_items], dtype=np.float32)
    areas = np.asarray([_score_object(value, "area") for _, value in object_items], dtype=np.float32)
    kept_indices, suppressed_records = _nms(
        object_items=object_items,
        num_points=int(pred_masks_in.shape[0]),
        scores=scores,
        areas=areas,
        min_points=int(args.filter_min_points_per_object),
        threshold=float(args.point_overlap_threshold),
        mode=args.point_overlap_mode,
        max_instances=int(args.filter_max_instances),
    )
    kept_indices_arr = np.asarray(kept_indices, dtype=np.int64)
    pred_masks = pred_masks_in[:, kept_indices_arr]
    if args.output_score_mode == "one":
        pred_scores = np.ones((kept_indices_arr.shape[0],), dtype=np.float32)
    elif args.output_score_mode == "select":
        pred_scores = scores[kept_indices_arr].astype(np.float32)
    else:
        raise ValueError(f"Unsupported output score mode: {args.output_score_mode}")

    pred_out_dir = Path("data/prediction") / f"{args.output_config}_class_agnostic"
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_out_dir / f"{seq_name}.npz",
        pred_masks=pred_masks,
        pred_score=pred_scores,
        pred_classes=pred["pred_classes"][kept_indices_arr],
    )

    tmp_out_dir = Path("data/TMP") / args.output_config
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
    pre_points, prediction_union, input_pre_points_count = _write_pre_points(
        args=args,
        seq_name=seq_name,
        input_pre_points_path=tmp_in,
        tmp_out_dir=tmp_out_dir,
        pred_masks=pred_masks,
    )

    object_out_dir = stream.object_dir / args.output_config
    object_out_dir.mkdir(parents=True, exist_ok=True)
    kept_object_dict = {object_items[int(idx)][0]: object_items[int(idx)][1] for idx in kept_indices}
    np.save(object_out_dir / "object_dict.npy", kept_object_dict, allow_pickle=True)

    pre_points_set = set(pre_points.tolist())
    prediction_union_set = set(prediction_union.tolist())
    return {
        "seq_name": seq_name,
        "input_config": args.input_config,
        "output_config": args.output_config,
        "num_instances": int(pred_masks_in.shape[1]),
        "num_kept_instances": int(kept_indices_arr.shape[0]),
        "num_suppressed_instances": int(len(suppressed_records)),
        "point_overlap_threshold": float(args.point_overlap_threshold),
        "point_overlap_mode": args.point_overlap_mode,
        "select_mode": args.select_mode,
        "output_score_mode": args.output_score_mode,
        "filter_min_points_per_object": int(args.filter_min_points_per_object),
        "filter_max_instances": int(args.filter_max_instances),
        "pre_points_policy": args.pre_points_policy,
        "input_pre_points_count": int(input_pre_points_count),
        "output_pre_points_count": int(pre_points.shape[0]),
        "prediction_union_count": int(prediction_union.shape[0]),
        "pre_points_equals_prediction_union": bool(pre_points_set == prediction_union_set),
        "prediction_union_subset_of_pre_points": bool(prediction_union_set.issubset(pre_points_set)),
        "alignment_checked": bool(alignment["alignment_checked"]),
        "alignment_failed_instances": int(alignment["alignment_failed_instances"]),
        "suppressed_preview": suppressed_records[:50],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-name", default="")
    parser.add_argument("--seq-list", default="")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--select-mode", default="mask_count")
    parser.add_argument("--output-score-mode", default="one", choices=["one", "select"])
    parser.add_argument("--point-overlap-threshold", type=float, default=0.75)
    parser.add_argument("--point-overlap-mode", default="ioc", choices=["ioc", "iou"])
    parser.add_argument("--filter-min-points-per-object", type=int, default=0)
    parser.add_argument("--filter-max-instances", type=int, default=0)
    parser.add_argument("--pre-points-policy", default="recompute", choices=["recompute", "inherit", "fixed_path"])
    parser.add_argument("--fixed-pre-points-root", default="data/TMP")
    parser.add_argument("--fixed-pre-points-config", default="")
    parser.add_argument("--alignment-iou-threshold", type=float, default=0.99)
    parser.add_argument("--debug-root", default="outputs/stream4d_point_nms")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.pre_points_policy == "fixed_path" and not args.fixed_pre_points_config:
        raise ValueError("--fixed-pre-points-config is required for fixed_path")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    summaries = []
    errors = []
    for seq_name in _seq_names(args):
        try:
            summary = _process_sequence(args, seq_name)
            summaries.append(summary)
            print(
                f"[point-nms] seq={seq_name} instances={summary['num_instances']} "
                f"kept={summary['num_kept_instances']} suppressed={summary['num_suppressed_instances']} "
                f"points={summary['output_pre_points_count']} union={summary['prediction_union_count']} "
                f"thr={summary['point_overlap_threshold']:.3f} mode={summary['point_overlap_mode']}",
                flush=True,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            message = f"{type(exc).__name__}: {exc}"
            print(f"[point-nms][ERROR] seq={seq_name} {message}", flush=True)
            errors.append({"seq_name": seq_name, "error": message})
    out_dir = Path(args.debug_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{args.output_config}_summary.json", {"summaries": summaries, "errors": errors})


if __name__ == "__main__":
    main()
