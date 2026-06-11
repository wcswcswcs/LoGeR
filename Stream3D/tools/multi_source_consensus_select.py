from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


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


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_weights(text: str | None, count: int) -> np.ndarray:
    if not text:
        return np.ones((count,), dtype=np.float32)
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != count:
        raise ValueError(f"Expected {count} source weights, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


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


def _load_prediction(root: Path, config: str, suffix: str, seq_name: str) -> dict[str, np.ndarray]:
    path = _prediction_path(root, config, suffix, seq_name)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {
            "pred_masks": data["pred_masks"].astype(bool, copy=False),
            "pred_score": data["pred_score"].astype(np.float32, copy=False),
            "pred_classes": data["pred_classes"].astype(np.int32, copy=False),
        }


def _overlap_matrix(support_masks: np.ndarray, areas: np.ndarray, mode: str) -> np.ndarray:
    if support_masks.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    intersections = support_masks.astype(np.int32).T @ support_masks.astype(np.int32)
    intersections = intersections.astype(np.float64)
    area_i = areas.reshape(-1, 1)
    area_j = areas.reshape(1, -1)
    if mode == "iou":
        denom = area_i + area_j - intersections
    elif mode == "min_ioc":
        denom = np.minimum(area_i, area_j)
    elif mode == "candidate_ioc":
        denom = area_j
    else:
        raise ValueError(f"Unsupported overlap mode: {mode}")
    return (intersections / np.maximum(denom, 1.0)).astype(np.float32)


def _groups_from_overlap(overlap: np.ndarray, threshold: float) -> list[list[int]]:
    num = overlap.shape[0]
    uf = UnionFind(num)
    rows, cols = np.nonzero(np.triu(overlap >= float(threshold), k=1))
    for row, col in zip(rows.tolist(), cols.tolist()):
        uf.union(int(row), int(col))
    groups: dict[int, list[int]] = {}
    for idx in range(num):
        groups.setdefault(uf.find(idx), []).append(idx)
    return list(groups.values())


def _max_overlap(
    kept_masks: np.ndarray,
    kept_areas: np.ndarray,
    mask: np.ndarray,
    area: float,
    mode: str,
) -> float:
    if kept_masks.shape[1] == 0 or area <= 0.0:
        return 0.0
    intersections = np.logical_and(kept_masks, mask[:, None]).sum(axis=0).astype(np.float64)
    if mode == "iou":
        denom = kept_areas + area - intersections
    elif mode == "min_ioc":
        denom = np.minimum(kept_areas, area)
    elif mode == "candidate_ioc":
        denom = np.full_like(intersections, area, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported overlap mode: {mode}")
    values = intersections / np.maximum(denom, 1.0)
    return float(np.max(values)) if values.size else 0.0


def _process_one(args: argparse.Namespace, root: Path, seq_name: str) -> dict[str, Any]:
    source_configs = _parse_csv(args.source_configs)
    low_configs = _parse_csv(args.low_recall_configs or "")
    if not source_configs:
        raise ValueError("--source-configs must contain at least one config")
    source_weights = _parse_weights(args.source_weights, len(source_configs))
    support_path = _tmp_path(root, args.score_pre_points_config, seq_name)
    if not support_path.exists():
        raise FileNotFoundError(support_path)
    support_ids = np.load(support_path).astype(np.int64)

    masks_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    classes_all: list[np.ndarray] = []
    source_ids_all: list[np.ndarray] = []
    local_ids_all: list[np.ndarray] = []
    source_names: list[str] = []
    for source_id, config in enumerate(source_configs):
        pred = _load_prediction(root, config, args.pred_suffix, seq_name)
        if masks_all and pred["pred_masks"].shape[0] != masks_all[0].shape[0]:
            raise ValueError(f"{seq_name}: source {config} has incompatible vertex count")
        n = pred["pred_masks"].shape[1]
        masks_all.append(pred["pred_masks"])
        scores_all.append(pred["pred_score"])
        classes_all.append(pred["pred_classes"])
        source_ids_all.append(np.full((n,), source_id, dtype=np.int32))
        local_ids_all.append(np.arange(n, dtype=np.int32))
        source_names.append(config)

    masks = np.concatenate(masks_all, axis=1)
    scores = np.concatenate(scores_all, axis=0)
    classes = np.concatenate(classes_all, axis=0)
    source_ids = np.concatenate(source_ids_all, axis=0)
    local_ids = np.concatenate(local_ids_all, axis=0)
    support_ids = support_ids[(support_ids >= 0) & (support_ids < masks.shape[0])]
    support_masks_all = masks[support_ids, :] if support_ids.size else np.zeros((0, masks.shape[1]), dtype=bool)
    support_area_all = support_masks_all.sum(axis=0).astype(np.float64)
    valid = support_area_all >= float(args.min_support_area)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        support_masks = support_masks_all[:, valid_indices]
        support_area = support_area_all[valid_indices]
        valid_scores = scores[valid_indices]
        valid_source_ids = source_ids[valid_indices]
        valid_local_ids = local_ids[valid_indices]
    else:
        support_masks = np.zeros((support_masks_all.shape[0], 0), dtype=bool)
        support_area = np.zeros((0,), dtype=np.float64)
        valid_scores = np.zeros((0,), dtype=np.float32)
        valid_source_ids = np.zeros((0,), dtype=np.int32)
        valid_local_ids = np.zeros((0,), dtype=np.int32)

    owner_counts = support_masks.sum(axis=1) if support_masks.shape[0] else np.zeros((0,), dtype=np.int32)
    conflict_area = (
        support_masks[owner_counts > 1, :].sum(axis=0).astype(np.float64)
        if support_masks.shape[0]
        else np.zeros((support_masks.shape[1],), dtype=np.float64)
    )
    conflict_ratio = conflict_area / np.maximum(support_area, 1.0)
    area_norm = _normalize(np.log1p(support_area))
    score_norm = _normalize(valid_scores)
    source_weight_values = source_weights[valid_source_ids] if valid_source_ids.size else np.zeros((0,), dtype=np.float32)

    overlap = _overlap_matrix(support_masks, support_area, args.group_overlap_mode)
    groups = _groups_from_overlap(overlap, args.group_overlap_threshold)

    selected_valid_indices: list[int] = []
    selected_groups: list[dict[str, Any]] = []
    for group in groups:
        group_arr = np.asarray(group, dtype=np.int64)
        group_source_count = int(np.unique(valid_source_ids[group_arr]).shape[0])
        if args.require_multi_source and group_source_count < 2:
            continue
        consensus_bonus = float(args.consensus_weight) * float(group_source_count - 1)
        quality = (
            float(args.source_weight_scale) * source_weight_values[group_arr]
            + float(args.score_weight) * score_norm[group_arr]
            + float(args.area_weight) * area_norm[group_arr]
            - float(args.conflict_weight) * conflict_ratio[group_arr]
            + consensus_bonus
        )
        best_pos = int(np.lexsort((valid_indices[group_arr], -quality))[0])
        best_local = int(group_arr[best_pos])
        selected_valid_indices.append(best_local)
        selected_groups.append(
            {
                "group_size": int(group_arr.shape[0]),
                "group_source_count": group_source_count,
                "selected_global_index": int(valid_indices[best_local]),
                "selected_source": source_names[int(valid_source_ids[best_local])],
                "selected_local_index": int(valid_local_ids[best_local]),
                "selected_quality": float(quality[best_pos]),
                "selected_support_area": float(support_area[best_local]),
                "selected_conflict_ratio": float(conflict_ratio[best_local]),
            }
        )

    selected_valid_indices = sorted(
        selected_valid_indices,
        key=lambda idx: (
            -len({int(valid_source_ids[j]) for j in groups[next(k for k, g in enumerate(groups) if idx in g)]}),
            -float(source_weight_values[idx]),
            -float(score_norm[idx]),
            -float(support_area[idx]),
            int(valid_indices[idx]),
        ),
    )
    selected_global = valid_indices[np.asarray(selected_valid_indices, dtype=np.int64)] if selected_valid_indices else np.zeros((0,), dtype=np.int64)

    out_masks: list[np.ndarray] = []
    out_classes: list[int] = []
    out_scores: list[float] = []
    for rank, valid_idx in enumerate(selected_valid_indices):
        global_idx = int(valid_indices[valid_idx])
        out_masks.append(masks[:, global_idx].copy())
        out_classes.append(int(classes[global_idx]))
        out_scores.append(float(args.high_score_base) - float(args.high_score_decay) * rank)

    low_added = 0
    if low_configs:
        high_support_masks = (
            np.stack([mask[support_ids] for mask in out_masks], axis=1)
            if out_masks and support_ids.size
            else np.zeros((support_ids.shape[0], 0), dtype=bool)
        )
        high_support_areas = high_support_masks.sum(axis=0).astype(np.float64)
        for config in low_configs:
            pred = _load_prediction(root, config, args.pred_suffix, seq_name)
            low_support_masks = (
                pred["pred_masks"][support_ids, :]
                if support_ids.size
                else np.zeros((0, pred["pred_masks"].shape[1]), dtype=bool)
            )
            low_areas = low_support_masks.sum(axis=0).astype(np.float64)
            order = np.argsort(-pred["pred_score"], kind="stable")
            for local_idx in order.tolist():
                if low_areas[local_idx] < float(args.low_min_support_area):
                    continue
                overlap_value = _max_overlap(
                    high_support_masks,
                    high_support_areas,
                    low_support_masks[:, local_idx],
                    float(low_areas[local_idx]),
                    args.low_drop_overlap_mode,
                )
                if overlap_value >= float(args.low_drop_overlap_threshold):
                    continue
                out_masks.append(pred["pred_masks"][:, local_idx].copy())
                out_classes.append(int(pred["pred_classes"][local_idx]))
                out_scores.append(float(args.low_score))
                low_added += 1
                if int(args.low_top_k) > 0 and low_added >= int(args.low_top_k):
                    break

    if out_masks:
        masks_out = np.stack(out_masks, axis=1).astype(bool, copy=False)
        scores_out = np.asarray(out_scores, dtype=np.float32)
        classes_out = np.asarray(out_classes, dtype=np.int32)
    else:
        masks_out = np.zeros((masks.shape[0], 0), dtype=bool)
        scores_out = np.zeros((0,), dtype=np.float32)
        classes_out = np.zeros((0,), dtype=np.int32)

    pred_out = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_out / f"{seq_name}.npz",
        pred_masks=masks_out,
        pred_score=scores_out,
        pred_classes=classes_out,
    )
    tmp_out = _tmp_path(root, args.output_config, seq_name)
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    if args.tmp_policy == "score_support":
        shutil.copy2(support_path, tmp_out)
        tmp_mode = "score_support"
    else:
        np.save(tmp_out, np.flatnonzero(masks_out.any(axis=1)).astype(np.int64))
        tmp_mode = "recompute_union"

    selected_source_counts = {
        source_names[source_id]: int(np.count_nonzero(source_ids[selected_global] == source_id))
        for source_id in range(len(source_names))
    }
    return {
        "seq_name": seq_name,
        "source_configs": source_configs,
        "low_recall_configs": low_configs,
        "output_config": args.output_config,
        "tmp_mode": tmp_mode,
        "num_input_instances": int(masks.shape[1]),
        "num_valid_instances": int(valid_indices.shape[0]),
        "num_groups": int(len(groups)),
        "num_selected_high": int(selected_global.shape[0]),
        "num_low_added": int(low_added),
        "num_output_instances": int(masks_out.shape[1]),
        "output_union": int(masks_out.any(axis=1).sum()) if masks_out.shape[1] else 0,
        "support_union": int(masks_out[support_ids, :].any(axis=1).sum()) if masks_out.shape[1] and support_ids.size else 0,
        "selected_source_counts": selected_source_counts,
        "group_source_count_mean": float(np.mean([r["group_source_count"] for r in selected_groups])) if selected_groups else 0.0,
        "group_size_mean": float(np.mean([r["group_size"] for r in selected_groups])) if selected_groups else 0.0,
        "selected_support_area_mean": float(np.mean([r["selected_support_area"] for r in selected_groups])) if selected_groups else 0.0,
        "selected_conflict_ratio_mean": float(np.mean([r["selected_conflict_ratio"] for r in selected_groups])) if selected_groups else 0.0,
        "selected_preview": selected_groups[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select prediction objects by cross-source consensus without GT.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--source-configs", required=True)
    parser.add_argument("--source-weights", default=None)
    parser.add_argument("--low-recall-configs", default="")
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--score-pre-points-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--min-support-area", type=float, default=1.0)
    parser.add_argument("--group-overlap-mode", default="min_ioc", choices=["iou", "min_ioc", "candidate_ioc"])
    parser.add_argument("--group-overlap-threshold", type=float, default=0.70)
    parser.add_argument("--require-multi-source", action="store_true")
    parser.add_argument("--source-weight-scale", type=float, default=1.0)
    parser.add_argument("--score-weight", type=float, default=0.20)
    parser.add_argument("--area-weight", type=float, default=0.20)
    parser.add_argument("--conflict-weight", type=float, default=0.30)
    parser.add_argument("--consensus-weight", type=float, default=0.25)
    parser.add_argument("--high-score-base", type=float, default=2.0)
    parser.add_argument("--high-score-decay", type=float, default=0.0001)
    parser.add_argument("--low-score", type=float, default=0.01)
    parser.add_argument("--low-top-k", type=int, default=0)
    parser.add_argument("--low-min-support-area", type=float, default=1.0)
    parser.add_argument("--low-drop-overlap-mode", default="candidate_ioc", choices=["iou", "min_ioc", "candidate_ioc"])
    parser.add_argument("--low-drop-overlap-threshold", type=float, default=0.85)
    parser.add_argument("--tmp-policy", default="score_support", choices=["score_support", "recompute"])
    parser.add_argument("--summary-root", default="outputs/multi_source_consensus_select")
    args = parser.parse_args()

    root = Path(args.root)
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
    print(f"[multi-source-consensus-select] wrote {out_path}")


if __name__ == "__main__":
    main()
