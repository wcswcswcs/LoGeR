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


def _priority(scores: np.ndarray, areas: np.ndarray, mode: str) -> np.ndarray:
    if mode == "score":
        return scores.astype(np.float64, copy=False)
    max_area = max(float(np.max(areas)) if areas.size else 0.0, 1.0)
    area_norm = areas.astype(np.float64, copy=False) / max_area
    if mode == "score_area_desc":
        return scores.astype(np.float64, copy=False) + 1e-3 * area_norm
    if mode == "score_area_asc":
        return scores.astype(np.float64, copy=False) - 1e-3 * area_norm
    if mode == "score_log_area_desc":
        feature = np.log1p(np.maximum(areas.astype(np.float64), 0.0))
        feature = feature / max(float(np.max(feature)) if feature.size else 0.0, 1.0)
        return scores.astype(np.float64, copy=False) + 1e-3 * feature
    if mode == "score_log_area_asc":
        feature = np.log1p(np.maximum(areas.astype(np.float64), 0.0))
        feature = feature / max(float(np.max(feature)) if feature.size else 0.0, 1.0)
        return scores.astype(np.float64, copy=False) - 1e-3 * feature
    raise ValueError(f"Unsupported priority mode: {mode}")


def _apply_wta(
    masks: np.ndarray,
    priority: np.ndarray,
    min_conflict_owners: int,
    min_priority_margin: float,
) -> tuple[np.ndarray, dict[str, float]]:
    refined = masks.copy()
    owner_counts = refined.sum(axis=1)
    conflict_points = np.flatnonzero(owner_counts >= int(min_conflict_owners))
    removed_assignments = 0
    for point_id in conflict_points.tolist():
        owners = np.flatnonzero(refined[point_id])
        if owners.size <= 1:
            continue
        owner_priority = priority[owners]
        # Stable tie-break: higher priority wins; if equal, lower column index wins.
        best_local = int(np.lexsort((owners, -owner_priority))[0])
        winner = int(owners[best_local])
        if float(min_priority_margin) > 0.0:
            sorted_priority = np.sort(owner_priority)
            second_best = float(sorted_priority[-2]) if sorted_priority.size >= 2 else -np.inf
            if float(owner_priority[best_local]) - second_best < float(min_priority_margin):
                continue
        losers = owners[owners != winner]
        refined[point_id, losers] = False
        removed_assignments += int(losers.size)
    after_counts = refined.sum(axis=1)
    empty_instances = int(np.count_nonzero(refined.sum(axis=0) == 0))
    return refined, {
        "num_conflict_points_before": float(conflict_points.shape[0]),
        "num_conflict_points_after": float(np.count_nonzero(after_counts >= int(min_conflict_owners))),
        "removed_point_assignments": float(removed_assignments),
        "empty_instances_after_wta": float(empty_instances),
    }


def process_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, float | str]:
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
        raise ValueError(f"{seq_name}: scores length does not match masks: {scores.shape} vs {masks.shape}")
    areas_before = masks.sum(axis=0).astype(np.float64)
    refined, diag = _apply_wta(
        masks=masks,
        priority=_priority(scores, areas_before, args.priority_mode),
        min_conflict_owners=args.min_conflict_owners,
        min_priority_margin=args.min_priority_margin,
    )

    keep = np.ones((refined.shape[1],), dtype=bool)
    if args.drop_empty:
        keep &= refined.sum(axis=0) > 0
    if int(args.min_area_after) > 0:
        keep &= refined.sum(axis=0) >= int(args.min_area_after)
    refined = refined[:, keep]
    scores_out = scores[keep]
    classes_out = classes[keep]

    pred_out_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_out_dir / f"{seq_name}.npz",
        pred_masks=refined,
        pred_score=scores_out,
        pred_classes=classes_out,
    )

    tmp_in = _tmp_path(root, args.input_config, seq_name)
    tmp_out_dir = root / "data" / "TMP" / args.output_config
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = tmp_out_dir / f"{seq_name}_pre_points.npy"
    if tmp_in.exists():
        shutil.copy2(tmp_in, tmp_out)
        tmp_mode = "copied_input_tmp"
    else:
        np.save(tmp_out, np.flatnonzero(np.any(refined, axis=1)).astype(np.int64))
        tmp_mode = "recomputed_missing_input_tmp"

    return {
        "seq_name": seq_name,
        "input_config": args.input_config,
        "output_config": args.output_config,
        "priority_mode": args.priority_mode,
        "min_conflict_owners": int(args.min_conflict_owners),
        "min_priority_margin": float(args.min_priority_margin),
        "drop_empty": bool(args.drop_empty),
        "min_area_after": int(args.min_area_after),
        "tmp_mode": tmp_mode,
        "num_instances_before": float(masks.shape[1]),
        "num_instances_after": float(refined.shape[1]),
        "union_count_before": float(np.count_nonzero(np.any(masks, axis=1))),
        "union_count_after": float(np.count_nonzero(np.any(refined, axis=1))),
        "point_assignments_before": float(np.count_nonzero(masks)),
        "point_assignments_after": float(np.count_nonzero(refined)),
        **diag,
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
        "priority_mode": args.priority_mode,
        "min_conflict_owners": int(args.min_conflict_owners),
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
        "--priority-mode",
        default="score",
        choices=[
            "score",
            "score_area_desc",
            "score_area_asc",
            "score_log_area_desc",
            "score_log_area_asc",
        ],
    )
    parser.add_argument("--min-conflict-owners", type=int, default=2)
    parser.add_argument("--min-priority-margin", type=float, default=0.0)
    parser.add_argument("--drop-empty", action="store_true")
    parser.add_argument("--min-area-after", type=int, default=0)
    parser.add_argument("--summary-root", default="outputs/stream4d_point_wta_v4_1")
    parser.add_argument("--eval-policy", default="prediction_point_wta")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--forbidden-for-method-table", action="store_true")
    parser.add_argument("--uses-rgbd-for-prediction", action="store_true")
    parser.add_argument("--uses-pose-for-prediction", action="store_true")
    parser.add_argument("--uses-scannet-mesh-for-prediction", action="store_true")
    parser.add_argument("--alignment-source", default="none")
    parser.add_argument("--alignment-used-for-prediction", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    rows = [process_sequence(args, seq_name) for seq_name in _read_seq_list(Path(args.seq_list))]
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
        source_configs=[args.input_config],
        pre_points_policy="input_tmp_copy_or_union_recompute",
        support_policy=(
            f"prediction_point_wta:{args.priority_mode}:"
            f"min_conflict_owners={args.min_conflict_owners}:"
            f"min_priority_margin={args.min_priority_margin}:"
            f"drop_empty={bool(args.drop_empty)}:min_area_after={args.min_area_after}"
        ),
        notes="Applies winner-take-all deconfliction to predicted point masks using prediction scores/areas only; no GT is read.",
        extra={
            "algorithm": "wta_prediction_points",
            "eval_policy": args.eval_policy,
            "input_config": args.input_config,
            "priority_mode": args.priority_mode,
            "min_conflict_owners": int(args.min_conflict_owners),
            "min_priority_margin": float(args.min_priority_margin),
            "drop_empty": bool(args.drop_empty),
            "min_area_after": int(args.min_area_after),
            "forbidden_for_method_table": bool(args.forbidden_for_method_table),
            "uses_rgbd_for_prediction": bool(args.uses_rgbd_for_prediction),
            "uses_pose_for_prediction": bool(args.uses_pose_for_prediction),
            "uses_scannet_mesh_for_prediction": bool(args.uses_scannet_mesh_for_prediction),
            "alignment_source": args.alignment_source,
            "alignment_used_for_prediction": bool(args.alignment_used_for_prediction),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=args.root, pred_suffix=args.pred_suffix.lstrip("_"))
    print(f"[wta-prediction-points] wrote {out_path}")


if __name__ == "__main__":
    main()
