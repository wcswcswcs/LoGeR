from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _external_support_signal(config: str | None) -> bool:
    if not config:
        return False
    name = str(config).lower()
    return name == "scannet" or name.startswith("scannet_") or "stream3d" in name


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_prediction(root: Path, config: str, suffix: str, seq_name: str) -> dict[str, np.ndarray]:
    path = root / "data" / "prediction" / f"{config}{suffix}" / f"{seq_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction: {path}")
    with np.load(path) as data:
        return {
            "pred_masks": data["pred_masks"].astype(bool, copy=False),
            "pred_score": data["pred_score"].astype(np.float32, copy=False),
            "pred_classes": data["pred_classes"].astype(np.int32, copy=False),
        }


def _tmp_path(root: Path, config: str, seq_name: str) -> Path:
    return root / "data" / "TMP" / config / f"{seq_name}_pre_points.npy"


def _score_array(count: int, value: float) -> np.ndarray:
    return np.full((int(count),), float(value), dtype=np.float32)


def _score_array_or_preserve(count: int, value: float, source_scores: np.ndarray) -> np.ndarray:
    if float(value) >= 0.0:
        return _score_array(count, value)
    if source_scores.shape[0] == count:
        return source_scores.astype(np.float32, copy=False)
    return _score_array(count, 1.0)


def _classes_for(count: int, classes: np.ndarray) -> np.ndarray:
    if classes.shape[0] == count:
        return classes.astype(np.int32, copy=False)
    return np.zeros((int(count),), dtype=np.int32)


def _filter_secondary_by_iou(
    primary_masks: np.ndarray,
    secondary_masks: np.ndarray,
    threshold: float,
    overlap_mode: str,
    support_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if threshold <= 0.0 or primary_masks.shape[1] == 0 or secondary_masks.shape[1] == 0:
        return np.ones((secondary_masks.shape[1],), dtype=bool), {
            "secondary_skipped_by_iou": 0.0,
            "secondary_max_overlap_mean": 0.0,
            "secondary_overlap_support_points": float(0 if support_ids is None else support_ids.shape[0]),
        }

    if support_ids is not None:
        support_ids = support_ids.astype(np.int64, copy=False)
        if support_ids.size:
            if int(support_ids.min()) < 0 or int(support_ids.max()) >= primary_masks.shape[0]:
                raise ValueError(
                    f"drop-overlap pre_points outside prediction range: "
                    f"min={int(support_ids.min())}, max={int(support_ids.max())}, vertices={primary_masks.shape[0]}"
                )
            primary_masks = primary_masks[support_ids, :]
            secondary_masks = secondary_masks[support_ids, :]
        else:
            primary_masks = np.zeros((0, primary_masks.shape[1]), dtype=bool)
            secondary_masks = np.zeros((0, secondary_masks.shape[1]), dtype=bool)

    primary_counts = primary_masks.sum(axis=0).astype(np.float64)
    keep = np.ones((secondary_masks.shape[1],), dtype=bool)
    max_overlaps = []
    for idx in range(secondary_masks.shape[1]):
        mask = secondary_masks[:, idx]
        secondary_count = float(mask.sum())
        intersections = np.logical_and(primary_masks, mask[:, None]).sum(axis=0).astype(np.float64)
        if overlap_mode == "iou":
            unions = primary_counts + secondary_count - intersections
            overlaps = intersections / np.maximum(unions, 1.0)
        elif overlap_mode == "secondary_ioc":
            overlaps = intersections / max(secondary_count, 1.0)
        elif overlap_mode == "min_ioc":
            overlaps = intersections / np.maximum(np.minimum(primary_counts, secondary_count), 1.0)
        else:
            raise ValueError(f"Unsupported overlap mode: {overlap_mode}")
        max_overlap = float(np.max(overlaps)) if overlaps.size else 0.0
        max_overlaps.append(max_overlap)
        keep[idx] = max_overlap < threshold
    return keep, {
        "secondary_skipped_by_iou": float(np.count_nonzero(~keep)),
        "secondary_max_overlap_mean": float(np.mean(max_overlaps)) if max_overlaps else 0.0,
        "secondary_overlap_mode": overlap_mode,
        "secondary_overlap_support_points": float(0 if support_ids is None else support_ids.shape[0]),
    }


def _select_variant_masks(
    primary_masks: np.ndarray,
    primary_scores: np.ndarray,
    secondary_masks: np.ndarray,
    secondary_scores: np.ndarray,
    min_primary_ioc: float,
    max_expansion: float,
    add_unmatched_secondary: bool,
    primary_score: float,
    secondary_score: float,
    unmatched_min_secondary_score: float,
    unmatched_min_area: int,
    unmatched_max_area: int,
    unmatched_top_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if primary_masks.shape[1] == 0:
        return secondary_masks.copy(), _score_array_or_preserve(secondary_masks.shape[1], secondary_score, secondary_scores), {
            "variant_primary_replaced": 0.0,
            "variant_unmatched_secondary_added": float(secondary_masks.shape[1]),
            "variant_matched_secondary": 0.0,
        }
    if secondary_masks.shape[1] == 0:
        return primary_masks.copy(), _score_array_or_preserve(primary_masks.shape[1], primary_score, primary_scores), {
            "variant_primary_replaced": 0.0,
            "variant_unmatched_secondary_added": 0.0,
            "variant_matched_secondary": 0.0,
        }

    primary_counts = primary_masks.sum(axis=0).astype(np.float64)
    selected = [primary_masks[:, idx].copy() for idx in range(primary_masks.shape[1])]
    primary_score_values = _score_array_or_preserve(primary_masks.shape[1], primary_score, primary_scores)
    selected_scores = [float(value) for value in primary_score_values.tolist()]
    used_secondary: set[int] = set()
    replaced = 0
    matched_secondary = 0

    for sec_idx in range(secondary_masks.shape[1]):
        sec_mask = secondary_masks[:, sec_idx]
        sec_count = float(np.count_nonzero(sec_mask))
        if sec_count <= 0.0:
            continue
        intersections = np.logical_and(primary_masks, sec_mask[:, None]).sum(axis=0).astype(np.float64)
        primary_ioc = intersections / np.maximum(primary_counts, 1.0)
        best_idx = int(np.argmax(primary_ioc)) if primary_ioc.size else -1
        if best_idx < 0:
            continue
        best_ioc = float(primary_ioc[best_idx])
        if best_ioc < float(min_primary_ioc):
            continue
        matched_secondary += 1
        expansion = sec_count / max(float(primary_counts[best_idx]), 1.0)
        if expansion <= float(max_expansion):
            current_count = float(np.count_nonzero(selected[best_idx]))
            current_expansion = current_count / max(float(primary_counts[best_idx]), 1.0)
            if expansion > current_expansion:
                selected[best_idx] = sec_mask.copy()
                used_secondary.add(sec_idx)
                replaced += 1

    unmatched_added = 0
    unmatched_candidates: list[tuple[float, int, np.ndarray]] = []
    if add_unmatched_secondary:
        for sec_idx in range(secondary_masks.shape[1]):
            if sec_idx in used_secondary:
                continue
            sec_mask = secondary_masks[:, sec_idx]
            sec_area = int(np.count_nonzero(sec_mask))
            if sec_area <= 0:
                continue
            if secondary_scores.size and float(secondary_scores[sec_idx]) < float(unmatched_min_secondary_score):
                continue
            if int(unmatched_min_area) > 0 and sec_area < int(unmatched_min_area):
                continue
            if int(unmatched_max_area) > 0 and sec_area > int(unmatched_max_area):
                continue
            intersections = np.logical_and(primary_masks, sec_mask[:, None]).sum(axis=0).astype(np.float64)
            best_ioc = float(np.max(intersections / np.maximum(primary_counts, 1.0))) if intersections.size else 0.0
            if best_ioc < float(min_primary_ioc):
                score = float(secondary_scores[sec_idx]) if secondary_scores.size else float(secondary_score)
                unmatched_candidates.append((score, sec_idx, sec_mask.copy()))

        unmatched_candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        if int(unmatched_top_k) > 0:
            unmatched_candidates = unmatched_candidates[: int(unmatched_top_k)]
        for _, sec_idx, sec_mask in unmatched_candidates:
            selected.append(sec_mask)
            if secondary_score < 0.0 and secondary_scores.shape[0] > sec_idx:
                selected_scores.append(float(secondary_scores[sec_idx]))
            else:
                selected_scores.append(float(secondary_score))
            unmatched_added += 1

    return np.stack(selected, axis=1).astype(bool, copy=False), np.asarray(selected_scores, dtype=np.float32), {
        "variant_primary_replaced": float(replaced),
        "variant_unmatched_secondary_added": float(unmatched_added),
        "variant_unmatched_secondary_candidates": float(len(unmatched_candidates)),
        "variant_matched_secondary": float(matched_secondary),
    }


def fuse_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, float | str]:
    root = Path(args.root)
    primary = _load_prediction(root, args.primary_config, args.pred_suffix, seq_name)
    secondary = _load_prediction(root, args.secondary_config, args.pred_suffix, seq_name)
    primary_masks = primary["pred_masks"]
    secondary_masks = secondary["pred_masks"]
    if primary_masks.shape[0] != secondary_masks.shape[0]:
        raise ValueError(
            f"{seq_name}: vertex count mismatch primary={primary_masks.shape[0]} "
            f"secondary={secondary_masks.shape[0]}"
        )
    support_ids = None
    if args.drop_overlap_pre_points_config:
        support_path = _tmp_path(root, args.drop_overlap_pre_points_config, seq_name)
        if not support_path.exists():
            raise FileNotFoundError(f"Missing drop-overlap pre_points: {support_path}")
        support_ids = np.load(support_path).astype(np.int64)

    variant_diag: dict[str, float] = {}
    if args.fusion_mode == "concatenate":
        secondary_keep, secondary_filter_diag = _filter_secondary_by_iou(
            primary_masks,
            secondary_masks,
            float(args.drop_secondary_iou_threshold),
            args.drop_secondary_overlap_mode,
            support_ids=support_ids,
        )
        secondary_masks = secondary_masks[:, secondary_keep]
        secondary_classes = _classes_for(secondary["pred_classes"].shape[0], secondary["pred_classes"])[secondary_keep]
        masks = np.concatenate([primary_masks, secondary_masks], axis=1)
        scores = np.concatenate(
            [
                _score_array_or_preserve(primary_masks.shape[1], args.primary_score, primary["pred_score"]),
                _score_array_or_preserve(secondary_masks.shape[1], args.secondary_score, secondary["pred_score"][secondary_keep]),
            ],
            axis=0,
        )
        classes = np.concatenate(
            [
                _classes_for(primary_masks.shape[1], primary["pred_classes"]),
                secondary_classes,
            ],
            axis=0,
        ).astype(np.int32, copy=False)
    elif args.fusion_mode == "select_secondary":
        masks, scores, variant_diag = _select_variant_masks(
            primary_masks,
            primary["pred_score"],
            secondary_masks,
            secondary["pred_score"],
            min_primary_ioc=float(args.select_min_primary_ioc),
            max_expansion=float(args.select_max_expansion),
            add_unmatched_secondary=bool(args.add_unmatched_secondary),
            primary_score=float(args.primary_score),
            secondary_score=float(args.secondary_score),
            unmatched_min_secondary_score=float(args.unmatched_min_secondary_score),
            unmatched_min_area=int(args.unmatched_min_area),
            unmatched_max_area=int(args.unmatched_max_area),
            unmatched_top_k=int(args.unmatched_top_k),
        )
        secondary_filter_diag = {
            "secondary_skipped_by_iou": 0.0,
            "secondary_max_iou_mean": 0.0,
        }
        classes = np.zeros((masks.shape[1],), dtype=np.int32)
    else:
        raise ValueError(f"Unsupported fusion mode: {args.fusion_mode}")

    pred_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{seq_name}.npz",
        pred_masks=masks,
        pred_score=scores,
        pred_classes=classes,
    )

    pre_points = np.flatnonzero(np.any(masks, axis=1)).astype(np.int64)
    tmp_dir = root / "data" / "TMP" / args.output_config
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.save(tmp_dir / f"{seq_name}_pre_points.npy", pre_points)

    primary_union = np.flatnonzero(np.any(primary_masks, axis=1))
    secondary_union = np.flatnonzero(np.any(secondary_masks, axis=1)) if secondary_masks.shape[1] else np.empty((0,), dtype=np.int64)
    return {
        "seq_name": seq_name,
        "num_scene_vertices": float(primary_masks.shape[0]),
        "num_primary_instances": float(primary_masks.shape[1]),
        "num_secondary_instances": float(secondary_masks.shape[1]),
        "num_output_instances": float(masks.shape[1]),
        "primary_union_count": float(primary_union.shape[0]),
        "secondary_union_count": float(secondary_union.shape[0]),
        "output_union_count": float(pre_points.shape[0]),
        "primary_union_ratio": float(primary_union.shape[0] / max(primary_masks.shape[0], 1)),
        "secondary_union_ratio": float(secondary_union.shape[0] / max(primary_masks.shape[0], 1)),
        "output_union_ratio": float(pre_points.shape[0] / max(primary_masks.shape[0], 1)),
        **secondary_filter_diag,
        **variant_diag,
    }


def aggregate(rows: list[dict[str, float | str]], args: argparse.Namespace) -> dict:
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
        "output_config": args.output_config,
        "primary_config": args.primary_config,
        "secondary_config": args.secondary_config,
        "primary_score": float(args.primary_score),
        "secondary_score": float(args.secondary_score),
        "drop_secondary_iou_threshold": float(args.drop_secondary_iou_threshold),
        "drop_overlap_pre_points_config": args.drop_overlap_pre_points_config,
        "fusion_mode": args.fusion_mode,
        "select_min_primary_ioc": float(args.select_min_primary_ioc),
        "select_max_expansion": float(args.select_max_expansion),
        "add_unmatched_secondary": bool(args.add_unmatched_secondary),
        "unmatched_min_secondary_score": float(args.unmatched_min_secondary_score),
        "unmatched_min_area": int(args.unmatched_min_area),
        "unmatched_max_area": int(args.unmatched_max_area),
        "unmatched_top_k": int(args.unmatched_top_k),
        "scenes": len(rows),
        **means,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--primary-config", required=True)
    parser.add_argument("--secondary-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--primary-score", type=float, default=1.0)
    parser.add_argument("--secondary-score", type=float, default=0.2)
    parser.add_argument(
        "--preserve-primary-score",
        action="store_true",
        help="keep primary prediction scores instead of replacing them with --primary-score",
    )
    parser.add_argument(
        "--preserve-secondary-score",
        action="store_true",
        help="keep secondary prediction scores instead of replacing them with --secondary-score",
    )
    parser.add_argument("--drop-secondary-iou-threshold", type=float, default=0.0)
    parser.add_argument(
        "--drop-overlap-pre-points-config",
        default="",
        help="if set, compute primary/secondary suppression overlap only on this TMP pre_points support",
    )
    parser.add_argument(
        "--drop-secondary-overlap-mode",
        default="iou",
        choices=["iou", "secondary_ioc", "min_ioc"],
        help=(
            "overlap metric for suppressing secondary masks; secondary_ioc is useful "
            "when low-score recall masks are mostly contained in high-score masks"
        ),
    )
    parser.add_argument(
        "--fusion-mode",
        default="concatenate",
        choices=["concatenate", "select_secondary"],
    )
    parser.add_argument("--select-min-primary-ioc", type=float, default=0.7)
    parser.add_argument("--select-max-expansion", type=float, default=2.0)
    parser.add_argument("--add-unmatched-secondary", action="store_true")
    parser.add_argument("--unmatched-min-secondary-score", type=float, default=0.0)
    parser.add_argument("--unmatched-min-area", type=int, default=0)
    parser.add_argument("--unmatched-max-area", type=int, default=0)
    parser.add_argument("--unmatched-top-k", type=int, default=0)
    parser.add_argument("--summary-root", default="outputs/stream4d_fusion_v4_1")
    parser.add_argument("--eval-policy", default="fuse_prediction_configs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.preserve_primary_score:
        args.primary_score = -1.0
    if args.preserve_secondary_score:
        args.secondary_score = -1.0
    root = Path(args.root)
    rows = [fuse_sequence(args, seq_name) for seq_name in _read_seq_list(Path(args.seq_list))]
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"args": vars(args), "aggregate": aggregate(rows, args), "rows": rows}
    out_path = out_dir / f"{args.output_config}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    external_support = _external_support_signal(args.drop_overlap_pre_points_config)
    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=not external_support,
        is_diagnostic_only=external_support,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.primary_config, args.secondary_config],
        pre_points_policy="recompute",
        support_policy=f"fusion:{args.fusion_mode}",
        notes=(
            "Generated by tools.fuse_prediction_configs without GT. "
            "Marked diagnostic-only when overlap suppression uses external Stream3D/scannet support."
        ),
        extra={
            "eval_policy": args.eval_policy,
            "drop_overlap_pre_points_config": args.drop_overlap_pre_points_config,
            "drop_overlap_pre_points_external_support": external_support,
            "drop_secondary_iou_threshold": float(args.drop_secondary_iou_threshold),
            "drop_secondary_overlap_mode": args.drop_secondary_overlap_mode,
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=root)
    print(f"[fuse-prediction-configs] wrote {out_path}")


if __name__ == "__main__":
    main()
