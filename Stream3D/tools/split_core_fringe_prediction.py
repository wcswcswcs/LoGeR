from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _tmp_path(root: Path, config: str, seq_name: str) -> Path:
    return root / "data" / "TMP" / config / f"{seq_name}_pre_points.npy"


def _prediction_path(root: Path, config: str, suffix: str, seq_name: str) -> Path:
    return root / "data" / "prediction" / f"{config}{suffix}" / f"{seq_name}.npz"


def _scene_points_path(root: Path, seq_name: str, backbone: str) -> Path:
    del backbone
    return root / "data" / "scannet" / "processed" / seq_name / f"{seq_name}_vh_clean_2.ply"


def _load_scene_points(root: Path, seq_name: str, backbone: str) -> np.ndarray:
    path = _scene_points_path(root, seq_name, backbone)
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"Failed to load scene points from {path}")
    return points


def _score_value(source_score: float, override: float) -> float:
    if float(override) < 0.0:
        return float(source_score)
    return float(override)


def _build_wta_assignments(
    support_masks: np.ndarray,
    scores: np.ndarray,
    priority_mode: str,
) -> np.ndarray:
    assignments = np.full((support_masks.shape[0],), -1, dtype=np.int64)
    if support_masks.size == 0 or support_masks.shape[1] == 0:
        return assignments

    support_areas = support_masks.sum(axis=0).astype(np.float64)
    safe_areas = np.maximum(support_areas, 1.0)
    score_values = scores.astype(np.float64, copy=False)
    if priority_mode == "score":
        priorities = score_values
    elif priority_mode == "small_area":
        priorities = -safe_areas
    elif priority_mode == "large_area":
        priorities = safe_areas
    elif priority_mode == "score_over_sqrt_area":
        priorities = score_values / np.sqrt(safe_areas)
    else:
        raise ValueError(f"Unsupported WTA priority mode: {priority_mode}")

    for row_idx in range(support_masks.shape[0]):
        owners = np.flatnonzero(support_masks[row_idx])
        if owners.size == 0:
            continue
        best_local = int(np.argmax(priorities[owners]))
        assignments[row_idx] = int(owners[best_local])
    return assignments


def _grow_support_by_radius(
    *,
    core_support: np.ndarray,
    support_mask: np.ndarray,
    support_ids: np.ndarray,
    full_mask: np.ndarray,
    owner_counts: np.ndarray,
    scene_points: np.ndarray,
    radius: float,
    candidate_mode: str,
    max_owners: int,
) -> tuple[np.ndarray, dict[str, int]]:
    core_ids = support_ids[core_support]
    if core_ids.size == 0:
        return core_support, {
            "growth_candidate_points": int(np.count_nonzero(support_mask)),
            "growth_kept_points": 0,
            "growth_added_points": 0,
        }
    if candidate_mode == "support":
        candidate_support = support_mask.copy()
        if int(max_owners) > 0:
            candidate_support &= owner_counts <= int(max_owners)
        candidate_ids = support_ids[candidate_support]
        candidate_lookup = np.flatnonzero(candidate_support)
    elif candidate_mode == "full":
        candidate_ids = np.flatnonzero(full_mask)
        if int(max_owners) > 0:
            owner_by_vertex = np.zeros((full_mask.shape[0],), dtype=np.int32)
            owner_by_vertex[support_ids] = owner_counts.astype(np.int32, copy=False)
            candidate_ids = candidate_ids[owner_by_vertex[candidate_ids] <= int(max_owners)]
        candidate_lookup = None
    elif candidate_mode == "scene":
        candidate_ids = np.arange(full_mask.shape[0], dtype=np.int64)
        if int(max_owners) > 0:
            owner_by_vertex = np.zeros((full_mask.shape[0],), dtype=np.int32)
            owner_by_vertex[support_ids] = owner_counts.astype(np.int32, copy=False)
            candidate_ids = candidate_ids[owner_by_vertex[candidate_ids] <= int(max_owners)]
        candidate_lookup = None
    else:
        raise ValueError(f"Unsupported growth candidate mode: {candidate_mode}")

    if candidate_ids.size == 0:
        return core_support, {
            "growth_candidate_points": 0,
            "growth_kept_points": int(core_ids.size),
            "growth_added_points": 0,
        }

    tree = cKDTree(scene_points[core_ids])
    distances, _ = tree.query(scene_points[candidate_ids], k=1, workers=-1)
    keep_candidate = np.isfinite(distances) & (distances <= float(radius))

    grown_support = core_support.copy()
    if candidate_mode == "support":
        grown_support[candidate_lookup[keep_candidate]] = True
        kept_points = int(np.count_nonzero(grown_support))
    else:
        # Full-mode growth is represented later by absolute vertex ids; this function
        # returns a synthetic support mask for diagnostics and uses caller-side ids.
        grown_support = np.zeros_like(core_support)
        support_member = np.isin(support_ids, candidate_ids[keep_candidate], assume_unique=False)
        grown_support[support_member] = True
        grown_support |= core_support
        kept_points = int(np.count_nonzero(keep_candidate))

    return grown_support, {
        "growth_candidate_points": int(candidate_ids.size),
        "growth_kept_points": kept_points,
        "growth_added_points": int(max(kept_points - core_ids.size, 0)),
    }


def _copy_or_link_tmp(args: argparse.Namespace, root: Path, seq_name: str, masks: np.ndarray) -> str:
    tmp_out_dir = root / "data" / "TMP" / args.output_config
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = tmp_out_dir / f"{seq_name}_pre_points.npy"

    if args.tmp_policy == "support":
        tmp_in = _tmp_path(root, args.support_config, seq_name)
    elif args.tmp_policy == "input":
        tmp_in = _tmp_path(root, args.input_config, seq_name)
    elif args.tmp_policy == "recompute":
        np.save(tmp_out, np.flatnonzero(np.any(masks, axis=1)).astype(np.int64))
        return "recompute_output_union"
    else:
        raise ValueError(f"Unsupported tmp policy: {args.tmp_policy}")

    if not tmp_in.exists():
        raise FileNotFoundError(f"Missing TMP pre_points for tmp policy {args.tmp_policy}: {tmp_in}")
    shutil.copy2(tmp_in, tmp_out)
    return f"copied_{args.tmp_policy}_tmp"


def process_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, float | str]:
    root = Path(args.root)
    scene_points = None
    if args.growth_mode == "radius":
        scene_points = _load_scene_points(root, seq_name, args.backbone)
    pred_path = _prediction_path(root, args.input_config, args.pred_suffix, seq_name)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction: {pred_path}")
    with np.load(pred_path) as data:
        masks = data["pred_masks"].astype(bool, copy=False)
        scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)
    if masks.ndim != 2:
        raise ValueError(f"{seq_name}: pred_masks must be 2D, got {masks.shape}")
    if scores.shape[0] != masks.shape[1] or classes.shape[0] != masks.shape[1]:
        raise ValueError(f"{seq_name}: scores/classes length does not match masks: {masks.shape}")

    support_path = _tmp_path(root, args.support_config, seq_name)
    if not support_path.exists():
        raise FileNotFoundError(f"Missing support pre_points: {support_path}")
    support_ids = np.load(support_path).astype(np.int64)
    if support_ids.size:
        if int(support_ids.min()) < 0 or int(support_ids.max()) >= masks.shape[0]:
            raise ValueError(
                f"{seq_name}: support ids outside prediction range: "
                f"min={int(support_ids.min())}, max={int(support_ids.max())}, vertices={masks.shape[0]}"
            )
        support_masks = masks[support_ids, :]
    else:
        support_masks = np.zeros((0, masks.shape[1]), dtype=bool)

    owner_counts = support_masks.sum(axis=1) if support_masks.size else np.zeros((support_masks.shape[0],), dtype=np.int64)
    low_conflict_points = owner_counts <= int(args.max_core_owners)
    high_conflict_points = owner_counts > int(args.max_core_owners)
    wta_assignments = None
    if args.assignment_mode == "wta":
        wta_assignments = _build_wta_assignments(support_masks, scores, args.wta_priority)

    output_masks: list[np.ndarray] = []
    output_scores: list[float] = []
    output_classes: list[int] = []
    core_ratios: list[float] = []
    conflict_ratios: list[float] = []
    num_core = 0
    num_low = 0
    num_skipped_core = 0
    num_skipped_low = 0
    growth_candidate_points = 0
    growth_kept_points = 0
    growth_added_points = 0

    for idx in range(masks.shape[1]):
        support_mask = support_masks[:, idx] if support_masks.shape[0] else np.zeros((0,), dtype=bool)
        support_area = int(np.count_nonzero(support_mask))
        full_area = int(np.count_nonzero(masks[:, idx]))
        if support_area < int(args.min_support_area):
            num_skipped_core += 1
            num_skipped_low += 1
            continue

        if args.assignment_mode == "low_conflict":
            core_support = support_mask & low_conflict_points
        elif args.assignment_mode == "wta":
            assert wta_assignments is not None
            core_support = wta_assignments == idx
        else:
            raise ValueError(f"Unsupported assignment mode: {args.assignment_mode}")
        core_area = int(np.count_nonzero(core_support))
        if args.assignment_mode == "low_conflict":
            conflict_area = int(np.count_nonzero(support_mask & high_conflict_points))
        else:
            conflict_area = int(np.count_nonzero(support_mask & (wta_assignments != idx)))
        core_ratio = core_area / max(float(support_area), 1.0)
        conflict_ratio = conflict_area / max(float(support_area), 1.0)
        core_ratios.append(float(core_ratio))
        conflict_ratios.append(float(conflict_ratio))

        if core_area >= int(args.min_core_points) and core_ratio >= float(args.min_core_ratio):
            selected_support = core_support
            growth_stats = {
                "growth_candidate_points": int(np.count_nonzero(support_mask)),
                "growth_kept_points": int(core_area),
                "growth_added_points": 0,
            }
            if args.growth_mode == "radius":
                assert scene_points is not None
                selected_support, growth_stats = _grow_support_by_radius(
                    core_support=core_support,
                    support_mask=support_mask,
                    support_ids=support_ids,
                    full_mask=masks[:, idx],
                    owner_counts=owner_counts,
                    scene_points=scene_points,
                    radius=float(args.growth_radius),
                    candidate_mode=args.growth_candidate_mode,
                    max_owners=int(args.growth_max_owners),
                )
            core_mask = np.zeros((masks.shape[0],), dtype=bool)
            if args.growth_mode == "radius" and args.growth_candidate_mode in {"full", "scene"}:
                core_ids = support_ids[core_support]
                if args.growth_candidate_mode == "full":
                    candidate_ids = np.flatnonzero(masks[:, idx])
                else:
                    candidate_ids = np.arange(masks.shape[0], dtype=np.int64)
                if int(args.growth_max_owners) > 0:
                    owner_by_vertex = np.zeros((masks.shape[0],), dtype=np.int32)
                    owner_by_vertex[support_ids] = owner_counts.astype(np.int32, copy=False)
                    candidate_ids = candidate_ids[owner_by_vertex[candidate_ids] <= int(args.growth_max_owners)]
                tree = cKDTree(scene_points[core_ids])
                distances, _ = tree.query(scene_points[candidate_ids], k=1, workers=-1)
                keep_candidate = np.isfinite(distances) & (distances <= float(args.growth_radius))
                core_mask[candidate_ids[keep_candidate]] = True
                core_mask[core_ids] = True
            else:
                core_mask[support_ids[selected_support]] = True
            growth_candidate_points += int(growth_stats["growth_candidate_points"])
            growth_kept_points += int(np.count_nonzero(core_mask))
            growth_added_points += int(max(np.count_nonzero(core_mask) - core_area, 0))
            output_masks.append(core_mask)
            output_scores.append(_score_value(float(scores[idx]), float(args.core_score)))
            output_classes.append(int(classes[idx]))
            num_core += 1
        else:
            num_skipped_core += 1

        if args.low_mode == "none":
            continue
        if args.assignment_mode == "wta":
            raise ValueError("low-mode variants are only defined for assignment-mode=low_conflict")
        if args.low_mode == "full":
            low_mask = masks[:, idx].copy()
        elif args.low_mode == "support_full":
            low_mask = np.zeros((masks.shape[0],), dtype=bool)
            low_mask[support_ids[support_mask]] = True
        elif args.low_mode == "fringe":
            low_support = support_mask & high_conflict_points
            low_mask = np.zeros((masks.shape[0],), dtype=bool)
            low_mask[support_ids[low_support]] = True
        elif args.low_mode == "fringe_plus_core":
            low_support = support_mask
            low_mask = np.zeros((masks.shape[0],), dtype=bool)
            low_mask[support_ids[low_support]] = True
        else:
            raise ValueError(f"Unsupported low mode: {args.low_mode}")
        low_area = int(np.count_nonzero(low_mask))
        if low_area >= int(args.min_low_points):
            output_masks.append(low_mask)
            output_scores.append(float(args.low_score))
            output_classes.append(int(classes[idx]))
            num_low += 1
        else:
            num_skipped_low += 1

    if output_masks:
        out_masks = np.stack(output_masks, axis=1).astype(bool, copy=False)
        out_scores = np.asarray(output_scores, dtype=np.float32)
        out_classes = np.asarray(output_classes, dtype=np.int32)
    else:
        out_masks = np.zeros((masks.shape[0], 0), dtype=bool)
        out_scores = np.zeros((0,), dtype=np.float32)
        out_classes = np.zeros((0,), dtype=np.int32)

    pred_out_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_out_dir / f"{seq_name}.npz",
        pred_masks=out_masks,
        pred_score=out_scores,
        pred_classes=out_classes,
    )
    tmp_mode = _copy_or_link_tmp(args, root, seq_name, out_masks)

    support_union = int(np.count_nonzero(np.any(support_masks, axis=1))) if support_masks.shape[1] else 0
    if support_ids.size and out_masks.shape[1]:
        out_support_masks = out_masks[support_ids, :]
        output_support_union = int(np.count_nonzero(np.any(out_support_masks, axis=1)))
        output_support_assignments = int(np.count_nonzero(out_support_masks))
        output_support_conflict_points = int(np.count_nonzero(out_support_masks.sum(axis=1) > 1))
    else:
        output_support_union = 0
        output_support_assignments = 0
        output_support_conflict_points = 0

    return {
        "seq_name": seq_name,
        "input_config": args.input_config,
        "output_config": args.output_config,
        "support_config": args.support_config,
        "low_mode": args.low_mode,
        "assignment_mode": args.assignment_mode,
        "wta_priority": args.wta_priority,
        "tmp_mode": tmp_mode,
        "num_scene_vertices": float(masks.shape[0]),
        "num_support_points": float(support_ids.shape[0]),
        "num_instances_before": float(masks.shape[1]),
        "num_instances_after": float(out_masks.shape[1]),
        "num_core_instances": float(num_core),
        "num_low_instances": float(num_low),
        "num_skipped_core": float(num_skipped_core),
        "num_skipped_low": float(num_skipped_low),
        "growth_candidate_points": float(growth_candidate_points),
        "growth_kept_points": float(growth_kept_points),
        "growth_added_points": float(growth_added_points),
        "input_support_union": float(support_union),
        "output_union": float(np.count_nonzero(np.any(out_masks, axis=1))),
        "output_support_union": float(output_support_union),
        "output_support_assignments": float(output_support_assignments),
        "output_support_conflict_points": float(output_support_conflict_points),
        "output_support_conflict_ratio": float(output_support_conflict_points / max(output_support_union, 1)),
        "mean_core_ratio": float(np.mean(core_ratios)) if core_ratios else 0.0,
        "mean_conflict_ratio": float(np.mean(conflict_ratios)) if conflict_ratios else 0.0,
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
        "support_config": args.support_config,
        "low_mode": args.low_mode,
        "assignment_mode": args.assignment_mode,
        "wta_priority": args.wta_priority,
        "growth_mode": args.growth_mode,
        "growth_candidate_mode": args.growth_candidate_mode,
        "growth_radius": float(args.growth_radius),
        "growth_max_owners": int(args.growth_max_owners),
        "core_score": float(args.core_score),
        "low_score": float(args.low_score),
        "max_core_owners": int(args.max_core_owners),
        "min_core_points": int(args.min_core_points),
        "min_core_ratio": float(args.min_core_ratio),
        "min_low_points": int(args.min_low_points),
        "min_support_area": int(args.min_support_area),
        "scenes": len(rows),
        **means,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--support-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--max-core-owners", type=int, default=1)
    parser.add_argument(
        "--assignment-mode",
        default="low_conflict",
        choices=["low_conflict", "wta"],
        help="low_conflict drops points with too many owners; wta assigns each support point to one owner.",
    )
    parser.add_argument(
        "--wta-priority",
        default="score_over_sqrt_area",
        choices=["score", "small_area", "large_area", "score_over_sqrt_area"],
        help="Owner priority used when assignment-mode=wta.",
    )
    parser.add_argument("--min-core-points", type=int, default=10)
    parser.add_argument("--min-core-ratio", type=float, default=0.05)
    parser.add_argument("--min-low-points", type=int, default=10)
    parser.add_argument("--min-support-area", type=int, default=10)
    parser.add_argument("--growth-mode", default="none", choices=["none", "radius"])
    parser.add_argument("--growth-candidate-mode", default="support", choices=["support", "full", "scene"])
    parser.add_argument("--growth-radius", type=float, default=0.05)
    parser.add_argument(
        "--growth-max-owners",
        type=int,
        default=0,
        help="If >0, radius growth only considers candidate points with at most this many owners.",
    )
    parser.add_argument(
        "--core-score",
        type=float,
        default=-1.0,
        help="score for core layer; negative means preserve the source prediction score",
    )
    parser.add_argument("--low-score", type=float, default=0.01)
    parser.add_argument(
        "--low-mode",
        default="full",
        choices=["none", "full", "support_full", "fringe", "fringe_plus_core"],
    )
    parser.add_argument("--tmp-policy", default="support", choices=["support", "input", "recompute"])
    parser.add_argument(
        "--eval-policy",
        default="own_recompute_core_fringe_postprocess",
        help="Audit label describing how this method result should be evaluated.",
    )
    parser.add_argument("--summary-root", default="outputs/stream4d_core_fringe_v4_1")
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
        root=str(root),
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.input_config, args.support_config],
        pre_points_policy=args.tmp_policy,
        support_policy=(
            f"split_core_fringe:{args.low_mode}:assignment={args.assignment_mode}:"
            f"wta={args.wta_priority}:growth={args.growth_mode}:"
            f"candidate={args.growth_candidate_mode}:r={args.growth_radius}:support={args.support_config}"
        ),
        notes="Core/fringe support postprocess from prediction ownership conflicts; no GT used.",
        extra={
            "input_config": args.input_config,
            "support_config": args.support_config,
            "max_core_owners": int(args.max_core_owners),
            "assignment_mode": args.assignment_mode,
            "wta_priority": args.wta_priority,
            "growth_mode": args.growth_mode,
            "growth_candidate_mode": args.growth_candidate_mode,
            "growth_radius": float(args.growth_radius),
            "growth_max_owners": int(args.growth_max_owners),
            "min_core_points": int(args.min_core_points),
            "min_core_ratio": float(args.min_core_ratio),
            "low_mode": args.low_mode,
            "tmp_policy": args.tmp_policy,
            "eval_policy": args.eval_policy,
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=str(root), pred_suffix=args.pred_suffix.lstrip("_"))
    print(f"[split-core-fringe-prediction] wrote {out_path}")


if __name__ == "__main__":
    main()
