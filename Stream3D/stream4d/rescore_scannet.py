from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .diagnostics import write_json
from .scannet_stream import ScanNetStream


def _seq_names(args: argparse.Namespace) -> list[str]:
    if args.seq_name:
        return [args.seq_name]
    if args.seq_list:
        with Path(args.seq_list).open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    raise ValueError("Provide --seq-name or --seq-list")


def _unique_mask_observations(mask_list: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    best: dict[tuple[int, int], float] = {}
    for frame_id, mask_id, coverage in mask_list:
        key = (int(frame_id), int(mask_id))
        best[key] = max(best.get(key, 0.0), float(coverage))
    return [(frame_id, mask_id, coverage) for (frame_id, mask_id), coverage in best.items()]


def _score_object(value: dict, mode: str) -> float:
    point_ids = np.asarray(value.get("point_ids", []), dtype=np.int64)
    area = float(point_ids.shape[0])
    carrier_count = float(np.asarray(value.get("carrier_ids", []), dtype=np.int64).shape[0])
    masks = _unique_mask_observations(list(value.get("mask_list", [])))
    coverages = np.asarray([float(item[2]) for item in masks], dtype=np.float32)
    mask_count = float(len(masks))
    coverage_sum = float(coverages.sum()) if coverages.size else 0.0
    coverage_max = float(coverages.max()) if coverages.size else 0.0
    coverage_mean = float(coverages.mean()) if coverages.size else 0.0

    if mode == "one":
        return 1.0
    if mode == "area":
        return area
    if mode == "inverse_area":
        return 1.0 / (area + 1.0)
    if mode == "carrier_count":
        return carrier_count
    if mode == "mask_count":
        return mask_count
    if mode == "coverage_sum":
        return coverage_sum
    if mode == "coverage_max":
        return coverage_max
    if mode == "coverage_mean":
        return coverage_mean
    if mode == "coverage_area_sqrt":
        return coverage_sum * np.sqrt(max(area, 1.0))
    if mode == "mask_area_sqrt":
        return mask_count * np.sqrt(max(area, 1.0))
    if mode == "carrier_density":
        return carrier_count / (area + 1.0)
    if mode == "mask_density":
        return mask_count / (area + 1.0)
    raise ValueError(f"Unsupported score mode: {mode}")


def _rank_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32)
    finite = np.isfinite(scores)
    if not np.any(finite):
        return np.ones_like(scores, dtype=np.float32)
    safe = scores.astype(np.float64, copy=True)
    min_finite = float(np.min(safe[finite]))
    safe[~finite] = min_finite
    order = np.argsort(safe, kind="mergesort")
    ranks = np.empty_like(safe, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=safe.shape[0], dtype=np.float32)
    return ranks


def verify_object_dict_prediction_alignment(
    pred_masks: np.ndarray,
    object_items: list[tuple[int, dict]],
    threshold: float = 0.99,
    include_records: bool = False,
) -> dict:
    """Check that object_dict point_ids and prediction columns describe the same objects."""
    if pred_masks.ndim != 2:
        raise ValueError(f"pred_masks must be 2D, got shape={pred_masks.shape}")
    if pred_masks.shape[1] != len(object_items):
        raise ValueError(
            f"pred columns {pred_masks.shape[1]} do not match object items {len(object_items)}"
        )

    records: list[dict] = []
    checked_ious: list[float] = []
    failed = 0
    cannot_verify = False
    num_points = int(pred_masks.shape[0])

    for column_idx, (object_id, value) in enumerate(object_items):
        if "point_ids" not in value:
            cannot_verify = True
            continue
        raw_object_ids = np.asarray(value.get("point_ids", []), dtype=np.int64).reshape(-1)
        in_range = (raw_object_ids >= 0) & (raw_object_ids < num_points)
        object_ids = np.unique(raw_object_ids[in_range])
        pred_ids = np.flatnonzero(pred_masks[:, column_idx]).astype(np.int64)
        intersection = int(np.intersect1d(object_ids, pred_ids, assume_unique=True).shape[0])
        object_area = int(object_ids.shape[0])
        pred_area = int(pred_ids.shape[0])
        union = object_area + pred_area - intersection
        if union == 0:
            point_iou = 1.0
            point_ioc = 1.0
        else:
            point_iou = float(intersection / union)
            point_ioc = float(intersection / max(min(object_area, pred_area), 1))
        out_of_range = int(raw_object_ids.shape[0] - object_ids.shape[0])
        is_failed = point_iou < float(threshold) or out_of_range > 0
        failed += int(is_failed)
        checked_ious.append(point_iou)
        if include_records or is_failed:
            records.append(
                {
                    "object_id": int(object_id),
                    "column_idx": int(column_idx),
                    "point_iou": point_iou,
                    "point_ioc": point_ioc,
                    "area_object_dict": object_area,
                    "area_pred_column": pred_area,
                    "intersection": intersection,
                    "out_of_range_point_ids": out_of_range,
                    "failed": bool(is_failed),
                }
            )

    checked = len(checked_ious) > 0
    mean_iou = float(np.mean(checked_ious)) if checked else None
    min_iou = float(np.min(checked_ious)) if checked else None
    return {
        "alignment_checked": bool(checked),
        "cannot_verify_alignment": bool(cannot_verify or not checked),
        "alignment_num_checked": int(len(checked_ious)),
        "alignment_mean_iou": mean_iou,
        "alignment_min_iou": min_iou,
        "alignment_failed_instances": int(failed),
        "alignment_threshold": float(threshold),
        "alignment_records": records,
    }


def _pre_points_input_path(args: argparse.Namespace, seq_name: str) -> Path:
    if args.pre_points_policy == "fixed_path":
        return Path(args.fixed_pre_points_root) / args.fixed_pre_points_config / f"{seq_name}_pre_points.npy"
    return Path("data/TMP") / args.input_config / f"{seq_name}_pre_points.npy"


def _write_pre_points(
    args: argparse.Namespace,
    seq_name: str,
    tmp_in: Path,
    tmp_out_dir: Path,
    pred_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    input_pre_points = np.load(tmp_in).astype(np.int64)
    prediction_union = np.flatnonzero(pred_masks.any(axis=1)).astype(np.int64)
    if args.pre_points_policy == "recompute":
        output_pre_points = prediction_union
    elif args.pre_points_policy in {"inherit", "fixed_path"}:
        output_pre_points = input_pre_points
    else:
        raise ValueError(f"Unsupported pre_points_policy: {args.pre_points_policy}")
    np.save(tmp_out_dir / f"{seq_name}_pre_points.npy", output_pre_points)
    return output_pre_points, prediction_union, int(input_pre_points.shape[0]), int(prediction_union.shape[0])


def _process_sequence(args: argparse.Namespace, seq_name: str) -> dict:
    stream = ScanNetStream(seq_name=seq_name, backbone=args.backbone)
    pred_in = Path("data/prediction") / f"{args.input_config}_class_agnostic" / f"{seq_name}.npz"
    tmp_in = _pre_points_input_path(args, seq_name)
    object_in = stream.object_dir / args.input_config / "object_dict.npy"
    if not pred_in.exists():
        raise FileNotFoundError(f"Missing prediction: {pred_in}")
    if not tmp_in.exists():
        raise FileNotFoundError(f"Missing TMP pre-points: {tmp_in}")
    if not object_in.exists():
        raise FileNotFoundError(f"Missing object_dict: {object_in}")

    pred = np.load(pred_in)
    object_dict = np.load(object_in, allow_pickle=True).item()
    object_items = [(int(k), v) for k, v in sorted(object_dict.items(), key=lambda item: int(item[0]))]
    num_instances = int(pred["pred_masks"].shape[1])
    if len(object_items) != num_instances:
        raise RuntimeError(
            f"{seq_name}: object count {len(object_items)} does not match pred columns {num_instances}"
        )
    alignment = verify_object_dict_prediction_alignment(
        pred["pred_masks"],
        object_items,
        threshold=float(args.alignment_iou_threshold),
    )
    if alignment["alignment_checked"] and alignment["alignment_failed_instances"] > 0:
        worst = sorted(
            alignment["alignment_records"],
            key=lambda item: (float(item["point_iou"]), int(item["out_of_range_point_ids"])),
        )[:5]
        raise RuntimeError(
            f"{seq_name}: object_dict/pred alignment failed for "
            f"{alignment['alignment_failed_instances']} instances; worst={worst}"
        )

    score_values = np.asarray([_score_object(value, args.score_mode) for _, value in object_items], dtype=np.float32)
    select_values = np.asarray([_score_object(value, args.select_mode) for _, value in object_items], dtype=np.float32)
    areas = np.asarray([_score_object(value, "area") for _, value in object_items], dtype=np.float32)
    keep = areas >= float(args.filter_min_points_per_object)
    max_instances = int(args.filter_max_instances)
    if args.filter_max_instances_ratio > 0.0:
        max_instances = int(round(float(num_instances) * float(args.filter_max_instances_ratio)))
        max_instances = max(int(args.filter_min_instances), max_instances)
        if args.filter_max_instances > 0:
            max_instances = min(int(args.filter_max_instances), max_instances)
    if max_instances > 0 and int(np.count_nonzero(keep)) > max_instances:
        candidate_ids = np.flatnonzero(keep)
        order = np.argsort(select_values[candidate_ids], kind="mergesort")
        selected = candidate_ids[order[-max_instances:]]
        top_keep = np.zeros_like(keep, dtype=bool)
        top_keep[selected] = True
        keep = top_keep
    kept_indices = np.flatnonzero(keep)
    scores = score_values[kept_indices]
    if args.rank_scores:
        scores = _rank_scores(scores)

    pred_out_dir = Path("data/prediction") / f"{args.output_config}_class_agnostic"
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    pred_masks = pred["pred_masks"][:, kept_indices]
    np.savez_compressed(
        pred_out_dir / f"{seq_name}.npz",
        pred_masks=pred_masks,
        pred_score=scores.astype(np.float32),
        pred_classes=pred["pred_classes"][kept_indices],
    )

    tmp_out_dir = Path("data/TMP") / args.output_config
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
    pre_points, prediction_union, input_pre_points_count, prediction_union_count = _write_pre_points(
        args=args,
        seq_name=seq_name,
        tmp_in=tmp_in,
        tmp_out_dir=tmp_out_dir,
        pred_masks=pred_masks,
    )
    pre_points_set = set(pre_points.tolist())
    prediction_union_set = set(prediction_union.tolist())

    object_out_dir = stream.object_dir / args.output_config
    object_out_dir.mkdir(parents=True, exist_ok=True)
    kept_object_dict = {object_items[int(idx)][0]: object_items[int(idx)][1] for idx in kept_indices.tolist()}
    np.save(object_out_dir / "object_dict.npy", kept_object_dict, allow_pickle=True)

    return {
        "seq_name": seq_name,
        "num_instances": num_instances,
        "num_kept_instances": int(kept_indices.shape[0]),
        "num_pre_points": int(pre_points.shape[0]),
        "pre_points_policy": args.pre_points_policy,
        "input_pre_points_path": str(tmp_in),
        "input_pre_points_count": input_pre_points_count,
        "output_pre_points_count": int(pre_points.shape[0]),
        "prediction_union_count": prediction_union_count,
        "pre_points_equals_prediction_union": bool(pre_points_set == prediction_union_set),
        "prediction_union_subset_of_pre_points": bool(prediction_union_set.issubset(pre_points_set)),
        "score_mode": args.score_mode,
        "select_mode": args.select_mode,
        "rank_scores": bool(args.rank_scores),
        "filter_min_points_per_object": int(args.filter_min_points_per_object),
        "filter_max_instances": int(max_instances),
        "filter_max_instances_arg": int(args.filter_max_instances),
        "filter_min_instances": int(args.filter_min_instances),
        "filter_max_instances_ratio": float(args.filter_max_instances_ratio),
        "score_min": float(scores.min()) if scores.size else 0.0,
        "score_max": float(scores.max()) if scores.size else 0.0,
        "score_mean": float(scores.mean()) if scores.size else 0.0,
        "alignment_checked": bool(alignment["alignment_checked"]),
        "cannot_verify_alignment": bool(alignment["cannot_verify_alignment"]),
        "alignment_num_checked": int(alignment["alignment_num_checked"]),
        "alignment_mean_iou": alignment["alignment_mean_iou"],
        "alignment_min_iou": alignment["alignment_min_iou"],
        "alignment_failed_instances": int(alignment["alignment_failed_instances"]),
        "alignment_iou_threshold": float(args.alignment_iou_threshold),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-name", default="")
    parser.add_argument("--seq-list", default="")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument(
        "--score-mode",
        default="one",
        choices=[
            "one",
            "area",
            "inverse_area",
            "carrier_count",
            "mask_count",
            "coverage_sum",
            "coverage_max",
            "coverage_mean",
            "coverage_area_sqrt",
            "mask_area_sqrt",
            "carrier_density",
            "mask_density",
        ],
    )
    parser.add_argument(
        "--select-mode",
        default="area",
        choices=[
            "one",
            "area",
            "inverse_area",
            "carrier_count",
            "mask_count",
            "coverage_sum",
            "coverage_max",
            "coverage_mean",
            "coverage_area_sqrt",
            "mask_area_sqrt",
            "carrier_density",
            "mask_density",
        ],
    )
    parser.add_argument("--rank-scores", action="store_true")
    parser.add_argument("--filter-min-points-per-object", type=int, default=0)
    parser.add_argument("--filter-max-instances", type=int, default=0)
    parser.add_argument("--filter-min-instances", type=int, default=0)
    parser.add_argument("--filter-max-instances-ratio", type=float, default=0.0)
    parser.add_argument("--pre-points-policy", default="recompute", choices=["recompute", "inherit", "fixed_path"])
    parser.add_argument("--fixed-pre-points-root", default="data/TMP")
    parser.add_argument("--fixed-pre-points-config", default="")
    parser.add_argument("--alignment-iou-threshold", type=float, default=0.99)
    parser.add_argument("--debug-root", default="outputs/stream4d_rescore")
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
                f"[stream4d-rescore] seq={seq_name} instances={summary['num_instances']} "
                f"kept={summary['num_kept_instances']} points={summary['num_pre_points']} "
                f"policy={summary['pre_points_policy']} union={summary['prediction_union_count']} "
                f"score_min={summary['score_min']:.4f} score_max={summary['score_max']:.4f}",
                flush=True,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            message = f"{type(exc).__name__}: {exc}"
            print(f"[stream4d-rescore][ERROR] seq={seq_name} {message}", flush=True)
            errors.append({"seq_name": seq_name, "error": message})
    out_dir = Path(args.debug_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{args.output_config}_summary.json", {"summaries": summaries, "errors": errors})


if __name__ == "__main__":
    main()
