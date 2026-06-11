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
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
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


def _mesh_points(root: Path, scannet_root: str, seq_name: str) -> np.ndarray:
    path = root / scannet_root / seq_name / f"{seq_name}_vh_clean_2.ply"
    if not path.exists():
        raise FileNotFoundError(path)
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"{seq_name}: invalid mesh points shape {points.shape}")
    return points


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros((size,), dtype=np.int8)

    def find(self, item: int) -> int:
        item = int(item)
        while int(self.parent[item]) != item:
            self.parent[item] = self.parent[int(self.parent[item])]
            item = int(self.parent[item])
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


def _component_labels(points: np.ndarray, radius: float) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if points.shape[0] == 1:
        return np.zeros((1,), dtype=np.int64)
    tree = cKDTree(points)
    pairs = tree.query_pairs(float(radius), output_type="ndarray")
    uf = UnionFind(points.shape[0])
    if pairs.size:
        for left, right in pairs:
            uf.union(int(left), int(right))
    roots = np.asarray([uf.find(idx) for idx in range(points.shape[0])], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64, copy=False)


def _select_component_members(
    labels: np.ndarray,
    max_components: int,
    min_component_points: int,
    min_component_ratio: float,
) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0,), dtype=bool)
    counts = np.bincount(labels)
    order = np.argsort(-counts, kind="stable")
    total = int(labels.shape[0])
    keep_labels: list[int] = []
    for label in order.tolist():
        count = int(counts[label])
        if count < int(min_component_points):
            continue
        if count / max(total, 1) < float(min_component_ratio):
            continue
        keep_labels.append(int(label))
        if int(max_components) > 0 and len(keep_labels) >= int(max_components):
            break
    if not keep_labels:
        keep_labels = [int(order[0])]
    return np.isin(labels, np.asarray(keep_labels, dtype=np.int64))


def refine_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, Any]:
    root = Path(args.root)
    pred_path = _prediction_path(root, args.input_config, args.pred_suffix, seq_name)
    support_path = _tmp_path(root, args.support_config, seq_name)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not support_path.exists():
        raise FileNotFoundError(support_path)

    with np.load(pred_path) as data:
        masks = data["pred_masks"].astype(bool, copy=False)
        scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)
    if masks.ndim != 2:
        raise ValueError(f"{seq_name}: pred_masks must be 2D, got {masks.shape}")
    if scores.shape[0] != masks.shape[1] or classes.shape[0] != masks.shape[1]:
        raise ValueError(f"{seq_name}: inconsistent prediction shapes")

    support_ids = np.load(support_path).astype(np.int64)
    support_ids = support_ids[(support_ids >= 0) & (support_ids < masks.shape[0])]
    points = _mesh_points(root, args.scannet_root, seq_name)
    if points.shape[0] != masks.shape[0]:
        raise RuntimeError(f"{seq_name}: mesh points {points.shape[0]} != prediction rows {masks.shape[0]}")

    refined = masks.copy() if args.outside_support == "keep" else np.zeros_like(masks, dtype=bool)
    support_masks = masks[support_ids, :] if support_ids.size else np.zeros((0, masks.shape[1]), dtype=bool)
    component_counts: list[int] = []
    kept_component_counts: list[int] = []
    support_area_before: list[int] = []
    support_area_after: list[int] = []
    changed_instances = 0

    for idx in range(masks.shape[1]):
        local_support = support_ids[support_masks[:, idx]]
        before = int(local_support.shape[0])
        support_area_before.append(before)
        if before < int(args.min_support_area):
            if args.drop_small:
                refined[support_ids, idx] = False
                after = 0
            else:
                refined[support_ids, idx] = support_masks[:, idx]
                after = before
            component_counts.append(0)
            kept_component_counts.append(0)
            support_area_after.append(after)
            if after != before:
                changed_instances += 1
            continue

        labels = _component_labels(points[local_support], float(args.radius))
        counts = np.bincount(labels) if labels.size else np.zeros((0,), dtype=np.int64)
        keep_local = _select_component_members(
            labels,
            max_components=int(args.max_components_per_instance),
            min_component_points=int(args.min_component_points),
            min_component_ratio=float(args.min_component_ratio),
        )
        kept_ids = local_support[keep_local]
        refined[support_ids, idx] = False
        refined[kept_ids, idx] = True
        after = int(kept_ids.shape[0])
        component_counts.append(int(counts.shape[0]))
        kept_component_counts.append(int(np.unique(labels[keep_local]).shape[0]) if keep_local.size else 0)
        support_area_after.append(after)
        if after != before:
            changed_instances += 1

    keep_instance = np.ones((refined.shape[1],), dtype=bool)
    if args.drop_empty:
        keep_instance &= refined.any(axis=0)
    refined = refined[:, keep_instance]
    scores_out = scores[keep_instance]
    classes_out = classes[keep_instance]

    out_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{seq_name}.npz",
        pred_masks=refined,
        pred_score=scores_out,
        pred_classes=classes_out,
    )

    tmp_out = _tmp_path(root, args.output_config, seq_name)
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    if args.tmp_policy == "support":
        shutil.copy2(support_path, tmp_out)
        tmp_mode = "support"
    elif args.tmp_policy == "input":
        input_tmp = _tmp_path(root, args.input_config, seq_name)
        if input_tmp.exists():
            shutil.copy2(input_tmp, tmp_out)
            tmp_mode = "input"
        else:
            np.save(tmp_out, np.flatnonzero(refined.any(axis=1)).astype(np.int64))
            tmp_mode = "recompute_missing_input"
    else:
        np.save(tmp_out, np.flatnonzero(refined.any(axis=1)).astype(np.int64))
        tmp_mode = "recompute"

    before_arr = np.asarray(support_area_before, dtype=np.float64)
    after_arr = np.asarray(support_area_after, dtype=np.float64)
    comp_arr = np.asarray(component_counts, dtype=np.float64)
    kept_comp_arr = np.asarray(kept_component_counts, dtype=np.float64)
    return {
        "seq_name": seq_name,
        "input_config": args.input_config,
        "output_config": args.output_config,
        "support_config": args.support_config,
        "tmp_mode": tmp_mode,
        "num_instances_before": int(masks.shape[1]),
        "num_instances_after": int(refined.shape[1]),
        "num_changed_instances": int(changed_instances),
        "num_support_points": int(support_ids.shape[0]),
        "support_union_before": int(np.count_nonzero(support_masks.any(axis=1))) if support_masks.size else 0,
        "support_union_after": int(np.count_nonzero(refined[support_ids, :].any(axis=1))) if support_ids.size else 0,
        "support_area_before_mean": float(np.mean(before_arr)) if before_arr.size else 0.0,
        "support_area_after_mean": float(np.mean(after_arr)) if after_arr.size else 0.0,
        "support_area_keep_ratio_mean": float(np.mean(after_arr / np.maximum(before_arr, 1.0))) if before_arr.size else 0.0,
        "component_count_mean": float(np.mean(comp_arr)) if comp_arr.size else 0.0,
        "component_count_max": int(np.max(comp_arr)) if comp_arr.size else 0,
        "kept_component_count_mean": float(np.mean(kept_comp_arr)) if kept_comp_arr.size else 0.0,
        "radius": float(args.radius),
        "max_components_per_instance": int(args.max_components_per_instance),
        "min_component_points": int(args.min_component_points),
        "min_component_ratio": float(args.min_component_ratio),
        "outside_support": args.outside_support,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine prediction masks by 3D connected components inside a support set.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--support-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--radius", type=float, default=0.06)
    parser.add_argument("--max-components-per-instance", type=int, default=1)
    parser.add_argument("--min-component-points", type=int, default=20)
    parser.add_argument("--min-component-ratio", type=float, default=0.05)
    parser.add_argument("--min-support-area", type=int, default=20)
    parser.add_argument("--drop-small", action="store_true")
    parser.add_argument("--drop-empty", action="store_true")
    parser.add_argument("--outside-support", default="drop", choices=["drop", "keep"])
    parser.add_argument("--tmp-policy", default="support", choices=["support", "input", "recompute"])
    parser.add_argument("--summary-root", default="outputs/support_component_refine_v4_1")
    args = parser.parse_args()

    root = Path(args.root)
    rows = [refine_sequence(args, seq_name) for seq_name in _read_seq_list(root / args.seq_list)]
    numeric_keys = sorted(
        key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.generic))
    )
    aggregate = {}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, np.generic))]
        if vals:
            aggregate[f"mean_{key}"] = float(np.mean(vals))
    payload = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_config}_summary.json"
    out_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[support-component-refine] wrote {out_path}")


if __name__ == "__main__":
    main()
