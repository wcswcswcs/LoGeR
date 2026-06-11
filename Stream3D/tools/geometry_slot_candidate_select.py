from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


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


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_weights(text: str | None, count: int) -> np.ndarray:
    if not text:
        return np.ones((count,), dtype=np.float32)
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != count:
        raise ValueError(f"Expected {count} source weights, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


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


def _support_components(
    points: np.ndarray,
    radius: float,
    min_points: int,
    max_slots: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    tree = cKDTree(points)
    pairs = tree.query_pairs(float(radius), output_type="ndarray")
    uf = UnionFind(points.shape[0])
    for left, right in pairs:
        uf.union(int(left), int(right))
    by_root: dict[int, list[int]] = {}
    for idx in range(points.shape[0]):
        by_root.setdefault(uf.find(idx), []).append(idx)
    components = [
        np.asarray(indices, dtype=np.int64)
        for indices in by_root.values()
        if len(indices) >= int(min_points)
    ]
    components.sort(key=lambda item: item.shape[0], reverse=True)
    if int(max_slots) > 0:
        components = components[: int(max_slots)]
    sizes = [int(item.shape[0]) for item in components]
    return components, {
        "geometry_pairs": int(pairs.shape[0]),
        "geometry_raw_components": int(len(by_root)),
        "geometry_kept_slots": int(len(components)),
        "geometry_slot_size_mean": float(np.mean(sizes)) if sizes else 0.0,
        "geometry_slot_size_max": int(max(sizes)) if sizes else 0,
        "geometry_slot_size_min": int(min(sizes)) if sizes else 0,
    }


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


def _candidate_conflict_ratio(candidate_support: np.ndarray) -> np.ndarray:
    if candidate_support.size == 0:
        return np.zeros((candidate_support.shape[1],), dtype=np.float32)
    owner_counts = candidate_support.sum(axis=1)
    conflict_area = candidate_support[owner_counts > 1, :].sum(axis=0).astype(np.float64)
    area = np.maximum(candidate_support.sum(axis=0).astype(np.float64), 1.0)
    return (conflict_area / area).astype(np.float32)


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
    elif mode == "candidate_ioc":
        denom = np.full_like(intersections, area, dtype=np.float64)
    elif mode == "min_ioc":
        denom = np.minimum(kept_areas, area)
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
    points = _mesh_points(root, seq_name, args.scannet_root)
    support_ids = support_ids[(support_ids >= 0) & (support_ids < points.shape[0])]
    support_points = points[support_ids]
    slots, slot_diag = _support_components(
        support_points,
        radius=float(args.slot_radius),
        min_points=int(args.min_slot_points),
        max_slots=int(args.max_slots),
    )

    masks_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    classes_all: list[np.ndarray] = []
    source_ids_all: list[np.ndarray] = []
    for source_id, config in enumerate(source_configs):
        pred = _load_prediction(root, config, args.pred_suffix, seq_name)
        if pred["pred_masks"].shape[0] != points.shape[0]:
            raise ValueError(
                f"{seq_name}: {config} vertices={pred['pred_masks'].shape[0]} mesh={points.shape[0]}"
            )
        count = pred["pred_masks"].shape[1]
        masks_all.append(pred["pred_masks"])
        scores_all.append(pred["pred_score"])
        classes_all.append(pred["pred_classes"])
        source_ids_all.append(np.full((count,), source_id, dtype=np.int32))

    masks = np.concatenate(masks_all, axis=1)
    scores = np.concatenate(scores_all, axis=0)
    classes = np.concatenate(classes_all, axis=0)
    source_ids = np.concatenate(source_ids_all, axis=0)
    candidate_support = masks[support_ids, :] if support_ids.size else np.zeros((0, masks.shape[1]), dtype=bool)
    candidate_area = candidate_support.sum(axis=0).astype(np.float64)
    valid_candidates = candidate_area >= float(args.min_candidate_area)
    conflict_ratio = _candidate_conflict_ratio(candidate_support)
    score_norm = _normalize(scores)
    area_norm = _normalize(np.log1p(candidate_area))
    source_weight = source_weights[source_ids]

    slot_masks = np.zeros((support_ids.shape[0], len(slots)), dtype=bool)
    for slot_idx, local_ids in enumerate(slots):
        slot_masks[local_ids, slot_idx] = True
    slot_area = slot_masks.sum(axis=0).astype(np.float64)
    intersections = (
        slot_masks.astype(np.int32).T @ candidate_support.astype(np.int32)
        if slot_masks.shape[1] and candidate_support.shape[1]
        else np.zeros((slot_masks.shape[1], candidate_support.shape[1]), dtype=np.float64)
    )
    intersections = intersections.astype(np.float64, copy=False)
    slot_ioc = intersections / np.maximum(slot_area.reshape(-1, 1), 1.0)
    candidate_ioc = intersections / np.maximum(candidate_area.reshape(1, -1), 1.0)
    union = slot_area.reshape(-1, 1) + candidate_area.reshape(1, -1) - intersections
    iou = intersections / np.maximum(union, 1.0)
    area_ratio = candidate_area.reshape(1, -1) / np.maximum(slot_area.reshape(-1, 1), 1.0)
    ratio_error = np.abs(np.log(np.maximum(area_ratio, 1.0e-6)))
    area_match = np.clip(1.0 - ratio_error / max(float(args.max_log_area_error), 1.0e-6), 0.0, 1.0)
    quality = (
        float(args.slot_ioc_weight) * slot_ioc
        + float(args.candidate_ioc_weight) * candidate_ioc
        + float(args.iou_weight) * iou
        + float(args.score_weight) * score_norm.reshape(1, -1)
        + float(args.area_weight) * area_norm.reshape(1, -1)
        + float(args.source_weight_scale) * source_weight.reshape(1, -1)
        + float(args.area_match_weight) * area_match
        - float(args.conflict_weight) * conflict_ratio.reshape(1, -1)
    )
    valid_edges = (
        valid_candidates.reshape(1, -1)
        & (slot_ioc >= float(args.min_slot_ioc))
        & (candidate_ioc >= float(args.min_candidate_ioc))
        & (iou >= float(args.min_iou))
        & (area_ratio >= float(args.min_area_ratio))
        & (area_ratio <= float(args.max_area_ratio))
        & (quality >= float(args.min_quality))
    )
    rows, cols = np.nonzero(valid_edges)
    edge_quality = quality[rows, cols] if rows.size else np.zeros((0,), dtype=np.float64)
    order = np.lexsort((cols, rows, -edge_quality))
    slot_taken = np.zeros((slot_masks.shape[1],), dtype=bool)
    cand_taken = np.zeros((masks.shape[1],), dtype=bool)
    selected: list[int] = []
    selected_rows: list[int] = []
    for edge_idx in order.tolist():
        slot_idx = int(rows[edge_idx])
        cand_idx = int(cols[edge_idx])
        if slot_taken[slot_idx] or cand_taken[cand_idx]:
            continue
        slot_taken[slot_idx] = True
        cand_taken[cand_idx] = True
        selected.append(cand_idx)
        selected_rows.append(slot_idx)

    out_masks: list[np.ndarray] = []
    out_classes: list[int] = []
    out_scores: list[float] = []
    for rank, cand_idx in enumerate(selected):
        out_masks.append(masks[:, cand_idx].copy())
        out_classes.append(int(classes[cand_idx]))
        out_scores.append(float(args.selected_score_base) - float(args.selected_score_decay) * rank)

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
            low_support = (
                pred["pred_masks"][support_ids, :]
                if support_ids.size
                else np.zeros((0, pred["pred_masks"].shape[1]), dtype=bool)
            )
            low_area = low_support.sum(axis=0).astype(np.float64)
            for cand_idx in np.argsort(-pred["pred_score"], kind="stable").tolist():
                if low_area[cand_idx] < float(args.low_min_support_area):
                    continue
                overlap = _max_overlap(
                    high_support_masks,
                    high_support_areas,
                    low_support[:, cand_idx],
                    float(low_area[cand_idx]),
                    args.low_drop_overlap_mode,
                )
                if overlap >= float(args.low_drop_overlap_threshold):
                    continue
                out_masks.append(pred["pred_masks"][:, cand_idx].copy())
                out_classes.append(int(pred["pred_classes"][cand_idx]))
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

    selected_quality = quality[selected_rows, selected] if selected else np.zeros((0,), dtype=np.float64)
    selected_slot_ioc = slot_ioc[selected_rows, selected] if selected else np.zeros((0,), dtype=np.float64)
    selected_candidate_ioc = candidate_ioc[selected_rows, selected] if selected else np.zeros((0,), dtype=np.float64)
    return {
        "seq_name": seq_name,
        "output_config": args.output_config,
        "tmp_mode": tmp_mode,
        "source_configs": source_configs,
        "low_recall_configs": low_configs,
        "num_input_candidates": int(masks.shape[1]),
        "num_valid_candidates": int(np.count_nonzero(valid_candidates)),
        "num_slots": int(slot_masks.shape[1]),
        "num_candidate_edges": int(rows.shape[0]),
        "num_selected_high": int(len(selected)),
        "num_low_added": int(low_added),
        "num_output_instances": int(masks_out.shape[1]),
        "output_union": int(masks_out.any(axis=1).sum()) if masks_out.shape[1] else 0,
        "support_union": int(masks_out[support_ids, :].any(axis=1).sum()) if masks_out.shape[1] else 0,
        "selected_quality_mean": float(np.mean(selected_quality)) if selected_quality.size else 0.0,
        "selected_slot_ioc_mean": float(np.mean(selected_slot_ioc)) if selected_slot_ioc.size else 0.0,
        "selected_candidate_ioc_mean": float(np.mean(selected_candidate_ioc)) if selected_candidate_ioc.size else 0.0,
        **slot_diag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign candidates to GT-free 3D connected-component slots.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--source-configs", required=True)
    parser.add_argument("--source-weights", default=None)
    parser.add_argument("--low-recall-configs", default="")
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--score-pre-points-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--slot-radius", type=float, default=0.05)
    parser.add_argument("--min-slot-points", type=int, default=20)
    parser.add_argument("--max-slots", type=int, default=0)
    parser.add_argument("--min-candidate-area", type=float, default=1.0)
    parser.add_argument("--min-slot-ioc", type=float, default=0.20)
    parser.add_argument("--min-candidate-ioc", type=float, default=0.20)
    parser.add_argument("--min-iou", type=float, default=0.02)
    parser.add_argument("--min-area-ratio", type=float, default=0.10)
    parser.add_argument("--max-area-ratio", type=float, default=5.0)
    parser.add_argument("--max-log-area-error", type=float, default=2.0)
    parser.add_argument("--min-quality", type=float, default=-999.0)
    parser.add_argument("--slot-ioc-weight", type=float, default=0.25)
    parser.add_argument("--candidate-ioc-weight", type=float, default=0.30)
    parser.add_argument("--iou-weight", type=float, default=0.15)
    parser.add_argument("--score-weight", type=float, default=0.15)
    parser.add_argument("--area-weight", type=float, default=0.05)
    parser.add_argument("--source-weight-scale", type=float, default=0.10)
    parser.add_argument("--area-match-weight", type=float, default=0.10)
    parser.add_argument("--conflict-weight", type=float, default=0.20)
    parser.add_argument("--selected-score-base", type=float, default=2.0)
    parser.add_argument("--selected-score-decay", type=float, default=0.0001)
    parser.add_argument("--low-score", type=float, default=0.01)
    parser.add_argument("--low-top-k", type=int, default=0)
    parser.add_argument("--low-min-support-area", type=float, default=1.0)
    parser.add_argument("--low-drop-overlap-mode", default="candidate_ioc", choices=["iou", "min_ioc", "candidate_ioc"])
    parser.add_argument("--low-drop-overlap-threshold", type=float, default=0.85)
    parser.add_argument("--tmp-policy", default="score_support", choices=["score_support", "recompute"])
    parser.add_argument("--summary-root", default="outputs/geometry_slot_candidate_select")
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
    print(f"[geometry-slot-candidate-select] wrote {out_path}")


if __name__ == "__main__":
    main()
