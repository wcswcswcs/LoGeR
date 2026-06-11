from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prediction_path(root: Path, config: str, suffix: str, seq_name: str) -> Path:
    dirname = config if config.endswith(suffix) else f"{config}{suffix}"
    return root / "data" / "prediction" / dirname / f"{seq_name}.npz"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _pair_intersections(masks: np.ndarray) -> dict[tuple[int, int], int]:
    point_ids, obj_ids = np.nonzero(masks)
    if point_ids.size == 0:
        return {}
    order = np.lexsort((obj_ids, point_ids))
    point_ids = point_ids[order]
    obj_ids = obj_ids[order]
    pairs: Counter[tuple[int, int]] = Counter()
    start = 0
    while start < point_ids.size:
        end = start + 1
        while end < point_ids.size and point_ids[end] == point_ids[start]:
            end += 1
        owners = np.unique(obj_ids[start:end])
        if owners.size > 1:
            owners_list = owners.tolist()
            for i, left in enumerate(owners_list[:-1]):
                for right in owners_list[i + 1 :]:
                    pairs[(int(left), int(right))] += 1
        start = end
    return dict(pairs)


def process_sequence(args: argparse.Namespace, seq_name: str) -> dict[str, Any]:
    root = Path(args.root)
    pred_path = _prediction_path(root, args.input_config, args.pred_suffix, seq_name)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    with np.load(pred_path) as data:
        masks = data["pred_masks"].astype(bool, copy=False)
        scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)

    num_objects = int(masks.shape[1])
    areas = masks.sum(axis=0).astype(np.int64)
    valid = areas >= int(args.min_area)
    dsu = DSU(num_objects)
    pair_intersections = _pair_intersections(masks[:, valid]) if np.any(valid) else {}
    valid_indices = np.flatnonzero(valid).astype(np.int64)
    merge_edges = 0
    for (left_local, right_local), intersection in pair_intersections.items():
        left = int(valid_indices[left_local])
        right = int(valid_indices[right_local])
        denom = max(1, min(int(areas[left]), int(areas[right])))
        overlap = float(intersection) / float(denom)
        if overlap >= float(args.min_ioc):
            dsu.union(left, right)
            merge_edges += 1

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(num_objects):
        if not valid[idx]:
            continue
        groups[dsu.find(idx)].append(idx)

    out_masks: list[np.ndarray] = []
    out_scores: list[float] = []
    out_classes: list[int] = []
    merged_groups = 0
    for members in groups.values():
        if len(members) > 1:
            merged_groups += 1
        member_arr = np.asarray(members, dtype=np.int64)
        merged = np.any(masks[:, member_arr], axis=1)
        if int(np.count_nonzero(merged)) < int(args.min_output_area):
            continue
        out_masks.append(merged)
        if scores.shape[0] == num_objects:
            out_scores.append(float(np.max(scores[member_arr])))
        else:
            out_scores.append(1.0)
        if classes.shape[0] == num_objects:
            cls_vals = classes[member_arr].astype(int).tolist()
            out_classes.append(int(Counter(cls_vals).most_common(1)[0][0]))
        else:
            out_classes.append(0)

    if out_masks:
        masks_out = np.stack(out_masks, axis=1).astype(bool, copy=False)
        scores_out = np.asarray(out_scores, dtype=np.float32)
        classes_out = np.asarray(out_classes, dtype=np.int32)
        order = np.argsort(-scores_out, kind="stable")
        masks_out = masks_out[:, order]
        scores_out = scores_out[order]
        classes_out = classes_out[order]
    else:
        masks_out = np.zeros((masks.shape[0], 0), dtype=bool)
        scores_out = np.zeros((0,), dtype=np.float32)
        classes_out = np.zeros((0,), dtype=np.int32)

    pred_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{seq_name}.npz",
        pred_masks=masks_out,
        pred_score=scores_out,
        pred_classes=classes_out,
    )
    pre_points = np.flatnonzero(np.any(masks_out, axis=1)).astype(np.int64)
    tmp_dir = root / "data" / "TMP" / args.output_config
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.save(tmp_dir / f"{seq_name}_pre_points.npy", pre_points)
    owner_counts = masks_out.sum(axis=1) if masks_out.shape[1] else np.zeros((masks_out.shape[0],), dtype=np.int64)
    conflict = int(np.count_nonzero(owner_counts > 1))
    return {
        "seq_name": seq_name,
        "num_input_objects": num_objects,
        "num_valid_input_objects": int(np.count_nonzero(valid)),
        "num_output_objects": int(masks_out.shape[1]),
        "num_merge_edges": int(merge_edges),
        "num_merged_groups": int(merged_groups),
        "num_output_points": int(pre_points.shape[0]),
        "output_point_ratio": float(pre_points.shape[0] / max(masks.shape[0], 1)),
        "conflict_rate": float(conflict / max(pre_points.shape[0], 1)),
        "min_ioc": float(args.min_ioc),
    }


def aggregate(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    numeric_keys = sorted(k for row in rows for k, v in row.items() if isinstance(v, (int, float)))
    return {
        "algorithm": "merge_overlapping_prediction_masks",
        "input_config": args.input_config,
        "output_config": args.output_config,
        "min_ioc": float(args.min_ioc),
        "num_scenes": len(rows),
        **{
            f"{key}_mean": float(np.mean([float(row[key]) for row in rows if key in row]))
            for key in numeric_keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--min-ioc", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--min-output-area", type=int, default=20)
    parser.add_argument("--summary-root", default="outputs/v9_mask_merge")
    parser.add_argument("--eval-policy", default="own_recompute_overlap_merge")
    args = parser.parse_args()

    rows = [process_sequence(args, seq_name) for seq_name in _read_seq_list(Path(args.seq_list))]
    out_dir = Path(args.root) / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"args": vars(args), "aggregate": aggregate(rows, args), "rows": rows}
    out_path = out_dir / f"{args.output_config}_summary.json"
    out_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    manifest = build_prediction_manifest(
        root=args.root,
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.input_config],
        pre_points_policy="recompute",
        support_policy=f"overlap_merge:min_ioc={args.min_ioc}",
        notes="Merges overlapping predicted masks into scene-level object candidates using only prediction mask overlap; no GT is read.",
        extra={
            "algorithm": "merge_overlapping_prediction_masks",
            "eval_policy": args.eval_policy,
            "input_config": args.input_config,
            "min_ioc": float(args.min_ioc),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=args.root, pred_suffix=args.pred_suffix.lstrip("_"))
    print(f"[merge-overlapping-prediction-masks] wrote {out_path}")


if __name__ == "__main__":
    main()
