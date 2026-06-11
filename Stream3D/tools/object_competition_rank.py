from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _prediction_path(root: Path, config: str, suffix: str, seq_name: str) -> Path:
    dirname = config if config.endswith(suffix) else f"{config}{suffix}"
    return root / "data" / "prediction" / dirname / f"{seq_name}.npz"


def _tmp_path(root: Path, config: str, seq_name: str) -> Path:
    return root / "data" / "TMP" / config / f"{seq_name}_pre_points.npy"


def _mesh_points(root: Path, seq_name: str, scannet_root: str) -> np.ndarray:
    path = root / scannet_root / seq_name / f"{seq_name}_vh_clean_2.ply"
    if not path.exists():
        raise FileNotFoundError(path)
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"Invalid mesh points: {path}")
    return points


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return values.astype(np.float32)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def _support_features(
    masks: np.ndarray,
    support_ids: np.ndarray,
    points: np.ndarray | None,
) -> dict[str, np.ndarray]:
    support_masks = masks[support_ids, :] if support_ids.size else np.zeros((0, masks.shape[1]), dtype=bool)
    support_area = support_masks.sum(axis=0).astype(np.float64)
    full_area = masks.sum(axis=0).astype(np.float64)
    support_fraction = support_area / np.maximum(full_area, 1.0)

    owner_counts = support_masks.sum(axis=1) if support_masks.shape[0] else np.zeros((0,), dtype=np.int32)
    conflict_area = np.zeros((masks.shape[1],), dtype=np.float64)
    unique_area = np.zeros((masks.shape[1],), dtype=np.float64)
    if support_masks.shape[0]:
        conflict_area = support_masks[owner_counts > 1, :].sum(axis=0).astype(np.float64)
        unique_area = support_masks[owner_counts == 1, :].sum(axis=0).astype(np.float64)

    support_area_safe = np.maximum(support_area, 1.0)
    conflict_ratio = conflict_area / support_area_safe
    unique_ratio = unique_area / support_area_safe

    compact_density = np.zeros((masks.shape[1],), dtype=np.float64)
    bbox_diag = np.zeros((masks.shape[1],), dtype=np.float64)
    if points is not None:
        for idx in range(masks.shape[1]):
            ids = support_ids[support_masks[:, idx]] if support_ids.size else np.flatnonzero(masks[:, idx])
            if ids.shape[0] < 3:
                continue
            xyz = points[ids]
            span = np.maximum(xyz.max(axis=0) - xyz.min(axis=0), 1.0e-4)
            bbox_diag[idx] = float(np.linalg.norm(span))
            volume = float(np.prod(span))
            compact_density[idx] = float(ids.shape[0]) / max(volume, 1.0e-6)

    return {
        "support_area": support_area,
        "full_area": full_area,
        "support_fraction": support_fraction,
        "conflict_ratio": conflict_ratio,
        "unique_ratio": unique_ratio,
        "area_norm": _normalize(np.log1p(support_area)),
        "full_area_norm": _normalize(np.log1p(full_area)),
        "compactness_norm": _normalize(np.log1p(compact_density)),
        "bbox_tightness_norm": 1.0 - _normalize(bbox_diag),
    }


def _quality(scores: np.ndarray, features: dict[str, np.ndarray], mode: str) -> np.ndarray:
    score_norm = _normalize(scores)
    area = features["area_norm"]
    unique = features["unique_ratio"]
    conflict = features["conflict_ratio"]
    compact = features["compactness_norm"]
    tight = features["bbox_tightness_norm"]
    support_fraction = features["support_fraction"]

    if mode == "score_unique_compact":
        quality = 0.55 * score_norm + 0.20 * unique + 0.15 * compact + 0.10 * tight - 0.20 * conflict
    elif mode == "unique_compact_area":
        quality = 0.30 * unique + 0.30 * compact + 0.20 * area + 0.10 * score_norm + 0.10 * support_fraction - 0.25 * conflict
    elif mode == "score_compact":
        quality = 0.70 * score_norm + 0.20 * compact + 0.10 * unique - 0.10 * conflict
    elif mode == "area_unique":
        quality = 0.45 * area + 0.35 * unique + 0.10 * score_norm + 0.10 * support_fraction - 0.25 * conflict
    elif mode == "compact_only":
        quality = 0.60 * compact + 0.20 * unique + 0.20 * tight - 0.20 * conflict
    else:
        raise ValueError(f"Unsupported quality mode: {mode}")
    return np.asarray(quality, dtype=np.float32)


def _group_candidates(
    support_masks: np.ndarray,
    support_area: np.ndarray,
    threshold: float,
    overlap_mode: str,
) -> list[list[int]]:
    num = support_masks.shape[1]
    if num == 0:
        return []
    ints = support_masks.astype(np.int32).T @ support_masks.astype(np.int32)
    area_i = support_area.reshape(-1, 1)
    area_j = support_area.reshape(1, -1)
    if overlap_mode == "iou":
        denom = area_i + area_j - ints
    elif overlap_mode == "min_ioc":
        denom = np.minimum(area_i, area_j)
    elif overlap_mode == "candidate_ioc":
        denom = area_j
    else:
        raise ValueError(f"Unsupported overlap mode: {overlap_mode}")
    overlap = ints / np.maximum(denom, 1.0)

    uf = UnionFind(num)
    rows, cols = np.nonzero(np.triu(overlap >= float(threshold), k=1))
    for row, col in zip(rows.tolist(), cols.tolist()):
        uf.union(row, col)
    groups_by_root: dict[int, list[int]] = {}
    for idx in range(num):
        groups_by_root.setdefault(uf.find(idx), []).append(idx)
    return list(groups_by_root.values())


def _max_overlap(
    kept_masks: np.ndarray,
    kept_counts: np.ndarray,
    mask: np.ndarray,
    mask_count: float,
    mode: str,
) -> float:
    if kept_masks.shape[1] == 0 or mask_count <= 0.0:
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
    overlap = intersections / np.maximum(denom, 1.0)
    return float(np.max(overlap)) if overlap.size else 0.0


def _support_novel_points(
    support_masks: np.ndarray,
    selected: list[int],
    candidate_idx: int,
) -> int:
    candidate = support_masks[:, candidate_idx]
    if not selected:
        return int(np.count_nonzero(candidate))
    kept_union = np.any(support_masks[:, np.asarray(selected, dtype=np.int64)], axis=1)
    return int(np.count_nonzero(candidate & ~kept_union))


def _select_representatives(
    masks: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    support_ids: np.ndarray,
    points: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    support_masks_all = masks[support_ids, :] if support_ids.size else np.zeros((0, masks.shape[1]), dtype=bool)
    features = _support_features(masks, support_ids, points)
    valid = features["support_area"] >= float(args.min_support_area)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        empty = np.zeros((masks.shape[0], 0), dtype=bool)
        return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32), {
            "num_instances_before": int(masks.shape[1]),
            "num_valid_candidates": 0,
            "num_groups": 0,
            "num_selected": 0,
            "group_size_mean": 0.0,
            "group_size_max": 0,
        }

    valid_support_masks = support_masks_all[:, valid_indices]
    valid_support_area = features["support_area"][valid_indices]
    valid_scores = scores[valid_indices]
    quality_all = _quality(scores, features, args.quality_mode)
    quality = quality_all[valid_indices]

    groups_local = _group_candidates(
        valid_support_masks,
        valid_support_area,
        threshold=float(args.group_overlap_threshold),
        overlap_mode=args.group_overlap_mode,
    )
    selected: list[int] = []
    group_records: list[dict[str, Any]] = []
    for group in groups_local:
        group_arr = np.asarray(group, dtype=np.int64)
        local_best = sorted(
            group_arr.tolist(),
            key=lambda idx: (
                -float(quality[idx]),
                -float(valid_scores[idx]),
                -float(valid_support_area[idx]),
                int(valid_indices[idx]),
            ),
        )[0]
        global_best = int(valid_indices[local_best])
        selected.append(global_best)
        group_records.append(
            {
                "size": int(len(group)),
                "selected_index": global_best,
                "selected_quality": float(quality[local_best]),
                "selected_score": float(scores[global_best]),
                "selected_support_area": float(features["support_area"][global_best]),
            }
        )

    selected_full = sorted(
        selected,
        key=lambda idx: (-float(quality_all[idx]), -float(scores[idx]), -float(features["support_area"][idx]), idx),
    )
    selected = selected_full
    if int(args.max_instances) > 0:
        base_limit = int(args.max_instances)
        if int(args.small_rescue_reserve) > 0:
            base_limit = max(0, int(args.max_instances) - int(args.small_rescue_reserve))
        selected = selected_full[:base_limit]

    rescue_selected: list[int] = []
    rescue_rejected_overlap = 0
    rescue_rejected_novelty = 0
    rescue_rejected_area = 0
    if int(args.small_rescue_reserve) > 0 and int(args.max_instances) > 0:
        max_rescue_area = float(args.small_rescue_max_support_area)
        remaining_budget = max(0, int(args.max_instances) - len(selected))
        rescue_budget = min(int(args.small_rescue_reserve), remaining_budget)
        selected_set = set(selected)
        selected_masks = masks[:, np.asarray(selected, dtype=np.int64)] if selected else np.zeros((masks.shape[0], 0), dtype=bool)
        selected_counts = selected_masks.sum(axis=0).astype(np.float64)
        rescue_candidates = sorted(
            [idx for idx in valid_indices.tolist() if idx not in selected_set],
            key=lambda idx: (
                float(features["support_area"][idx]),
                -float(quality_all[idx]),
                -float(scores[idx]),
                idx,
            ),
        )
        for idx in rescue_candidates:
            support_area = float(features["support_area"][idx])
            if support_area < float(args.small_rescue_min_support_area) or (
                max_rescue_area > 0.0 and support_area > max_rescue_area
            ):
                rescue_rejected_area += 1
                continue
            novel_points = _support_novel_points(support_masks_all, selected + rescue_selected, int(idx))
            if novel_points < int(args.small_rescue_min_novel_points):
                rescue_rejected_novelty += 1
                continue
            overlap = _max_overlap(
                kept_masks=selected_masks,
                kept_counts=selected_counts,
                mask=masks[:, idx],
                mask_count=float(masks[:, idx].sum()),
                mode=args.small_rescue_overlap_mode,
            )
            if overlap >= float(args.small_rescue_overlap_threshold):
                rescue_rejected_overlap += 1
                continue
            rescue_selected.append(int(idx))
            selected_masks = np.concatenate([selected_masks, masks[:, idx : idx + 1]], axis=1)
            selected_counts = np.concatenate([selected_counts, np.asarray([float(masks[:, idx].sum())])])
            if len(rescue_selected) >= rescue_budget:
                break
        selected.extend(rescue_selected)
    selected_arr = np.asarray(selected, dtype=np.int64)

    out_scores = (0.5 + 0.5 * _normalize(quality_all[selected_arr])).astype(np.float32)
    if args.preserve_original_score:
        out_scores = scores[selected_arr].astype(np.float32)

    group_sizes = [record["size"] for record in group_records]
    return masks[:, selected_arr], out_scores, classes[selected_arr], {
        "num_instances_before": int(masks.shape[1]),
        "num_valid_candidates": int(valid_indices.shape[0]),
        "num_groups": int(len(groups_local)),
        "num_selected": int(selected_arr.shape[0]),
        "num_selected_base": int(selected_arr.shape[0] - len(rescue_selected)),
        "num_selected_small_rescue": int(len(rescue_selected)),
        "small_rescue_rejected_area": int(rescue_rejected_area),
        "small_rescue_rejected_novelty": int(rescue_rejected_novelty),
        "small_rescue_rejected_overlap": int(rescue_rejected_overlap),
        "group_size_mean": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "group_size_max": int(max(group_sizes)) if group_sizes else 0,
        "quality_mode": args.quality_mode,
        "group_overlap_mode": args.group_overlap_mode,
        "group_overlap_threshold": float(args.group_overlap_threshold),
        "min_support_area": int(args.min_support_area),
        "mean_selected_support_area": float(np.mean(features["support_area"][selected_arr])) if selected_arr.size else 0.0,
        "mean_selected_conflict_ratio": float(np.mean(features["conflict_ratio"][selected_arr])) if selected_arr.size else 0.0,
        "mean_selected_unique_ratio": float(np.mean(features["unique_ratio"][selected_arr])) if selected_arr.size else 0.0,
        "selected_preview": group_records[:50],
    }


def _process_one(args: argparse.Namespace, root: Path, seq_name: str) -> dict[str, Any]:
    pred_path = _prediction_path(root, args.input_config, args.pred_suffix, seq_name)
    support_path = _tmp_path(root, args.score_pre_points_config, seq_name)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not support_path.exists():
        raise FileNotFoundError(support_path)

    with np.load(pred_path) as data:
        masks = data["pred_masks"].astype(bool, copy=False)
        scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)
    support_ids = np.load(support_path).astype(np.int64)
    support_ids = support_ids[(support_ids >= 0) & (support_ids < masks.shape[0])]

    points = None
    if not args.no_geometry:
        points = _mesh_points(root, seq_name, args.scannet_root)
        if points.shape[0] != masks.shape[0]:
            raise RuntimeError(f"{seq_name}: mesh points {points.shape[0]} != prediction rows {masks.shape[0]}")

    out_masks, out_scores, out_classes, diag = _select_representatives(
        masks=masks,
        scores=scores,
        classes=classes,
        support_ids=support_ids,
        points=points,
        args=args,
    )

    out_pred_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_pred_dir / f"{seq_name}.npz",
        pred_masks=out_masks,
        pred_score=out_scores,
        pred_classes=out_classes,
    )

    out_tmp_dir = root / "data" / "TMP" / args.output_config
    out_tmp_dir.mkdir(parents=True, exist_ok=True)
    input_tmp = _tmp_path(root, args.input_config, seq_name)
    if input_tmp.exists() and args.tmp_policy == "inherit":
        shutil.copy2(input_tmp, out_tmp_dir / f"{seq_name}_pre_points.npy")
        tmp_mode = "inherit_input"
    elif args.tmp_policy == "score_support":
        shutil.copy2(support_path, out_tmp_dir / f"{seq_name}_pre_points.npy")
        tmp_mode = "score_support"
    else:
        np.save(out_tmp_dir / f"{seq_name}_pre_points.npy", np.flatnonzero(out_masks.any(axis=1)).astype(np.int64))
        tmp_mode = "recompute_union"

    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.input_config, args.score_pre_points_config],
        pre_points_policy=args.tmp_policy,
        support_policy=(
            f"object_competition_rank:quality={args.quality_mode}:"
            f"group={args.group_overlap_mode}@{args.group_overlap_threshold}:"
            f"small_rescue={args.small_rescue_reserve}"
        ),
        notes="Object-level competition/ranking postprocess from predictions and support only; no GT used.",
        extra={
            "input_config": args.input_config,
            "score_pre_points_config": args.score_pre_points_config,
            "eval_policy": args.eval_policy,
            "quality_mode": args.quality_mode,
            "group_overlap_mode": args.group_overlap_mode,
            "group_overlap_threshold": float(args.group_overlap_threshold),
            "min_support_area": int(args.min_support_area),
            "max_instances": int(args.max_instances),
            "small_rescue_reserve": int(args.small_rescue_reserve),
            "small_rescue_min_support_area": int(args.small_rescue_min_support_area),
            "small_rescue_max_support_area": int(args.small_rescue_max_support_area),
            "small_rescue_min_novel_points": int(args.small_rescue_min_novel_points),
            "small_rescue_overlap_threshold": float(args.small_rescue_overlap_threshold),
            "small_rescue_overlap_mode": args.small_rescue_overlap_mode,
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=root, pred_suffix=args.pred_suffix.lstrip("_"))

    return {
        "seq_name": seq_name,
        "input_config": args.input_config,
        "output_config": args.output_config,
        "score_pre_points_config": args.score_pre_points_config,
        "tmp_mode": tmp_mode,
        "output_union_count": int(np.count_nonzero(out_masks.any(axis=1))) if out_masks.size else 0,
        **diag,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--score-pre-points-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--quality-mode", default="score_unique_compact", choices=[
        "score_unique_compact",
        "unique_compact_area",
        "score_compact",
        "area_unique",
        "compact_only",
    ])
    parser.add_argument("--group-overlap-mode", default="min_ioc", choices=["iou", "min_ioc", "candidate_ioc"])
    parser.add_argument("--group-overlap-threshold", type=float, default=0.5)
    parser.add_argument("--min-support-area", type=int, default=1)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--preserve-original-score", action="store_true")
    parser.add_argument("--no-geometry", action="store_true")
    parser.add_argument("--tmp-policy", default="inherit", choices=["inherit", "score_support", "recompute"])
    parser.add_argument("--eval-policy", default="own_recompute_object_competition")
    parser.add_argument("--small-rescue-reserve", type=int, default=0)
    parser.add_argument("--small-rescue-min-support-area", type=int, default=1)
    parser.add_argument("--small-rescue-max-support-area", type=int, default=0)
    parser.add_argument("--small-rescue-min-novel-points", type=int, default=20)
    parser.add_argument("--small-rescue-overlap-threshold", type=float, default=0.50)
    parser.add_argument("--small-rescue-overlap-mode", default="min_ioc", choices=["iou", "candidate_ioc", "min_ioc"])
    parser.add_argument("--summary-root", default="outputs/stream4d_object_competition_v4_1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    rows = [_process_one(args, root, seq_name) for seq_name in _read_seq_list(root / args.seq_list)]
    numeric_keys = sorted(
        key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.generic))
    )
    aggregate = {f"mean_{key}": float(np.mean([float(row[key]) for row in rows])) for key in numeric_keys}
    payload = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_config}_summary.json"
    out_path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[object-competition-rank] wrote {out_path}")


if __name__ == "__main__":
    main()
