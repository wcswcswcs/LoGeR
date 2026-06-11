from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


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
    candidates = [
        root / "data" / "TMP" / config / f"{seq_name}_pre_points.npy",
        root / "TMP" / config / f"{seq_name}_pre_points.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _mesh_points(root: Path, seq_name: str, scannet_root: str) -> np.ndarray:
    path = root / scannet_root / seq_name / f"{seq_name}_vh_clean_2.ply"
    if not path.exists():
        raise FileNotFoundError(f"Missing scene mesh: {path}")
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"Invalid scene mesh points: {path}")
    return points


def _complete_one(args: argparse.Namespace, root: Path, seq_name: str) -> dict[str, Any]:
    pred_path = _prediction_path(root, args.pred_config, args.pred_suffix, seq_name)
    support_path = _tmp_path(root, args.support_config, seq_name)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction: {pred_path}")
    if not support_path.exists():
        raise FileNotFoundError(f"Missing support pre_points: {support_path}")

    with np.load(pred_path) as data:
        masks = data["pred_masks"].astype(bool, copy=True)
        scores = data["pred_score"].astype(np.float32, copy=False)
        classes = data["pred_classes"].astype(np.int32, copy=False)

    points = _mesh_points(root, seq_name, args.scannet_root)
    if masks.shape[0] != points.shape[0]:
        raise RuntimeError(
            f"{seq_name}: prediction must be full-scene for support completion; "
            f"pred rows={masks.shape[0]}, mesh vertices={points.shape[0]}"
        )

    support_ids = np.load(support_path).astype(np.int64)
    support_ids = support_ids[(support_ids >= 0) & (support_ids < points.shape[0])]
    support_ids = np.unique(support_ids)

    original_union = np.flatnonzero(masks.any(axis=1)).astype(np.int64)
    candidate_ids = support_ids
    if args.only_missing:
        covered = np.zeros((points.shape[0],), dtype=bool)
        covered[original_union] = True
        candidate_ids = candidate_ids[~covered[candidate_ids]]

    source_ids: list[np.ndarray] = []
    source_owner: list[np.ndarray] = []
    source_counts = masks.sum(axis=0).astype(np.int64)
    for object_idx in range(masks.shape[1]):
        ids = np.flatnonzero(masks[:, object_idx]).astype(np.int64)
        if ids.shape[0] < int(args.min_source_points):
            continue
        source_ids.append(ids)
        source_owner.append(np.full((ids.shape[0],), object_idx, dtype=np.int64))

    assigned_count = 0
    if candidate_ids.size and source_ids:
        all_source_ids = np.concatenate(source_ids, axis=0)
        all_source_owner = np.concatenate(source_owner, axis=0)
        tree = cKDTree(points[all_source_ids])
        dist, nn = tree.query(points[candidate_ids], k=1, distance_upper_bound=float(args.radius))
        valid = np.isfinite(dist) & (nn < all_source_ids.shape[0])
        valid_candidates = candidate_ids[valid]
        valid_dist = dist[valid].astype(np.float32)
        valid_owner = all_source_owner[nn[valid]]

        additions_by_owner: dict[int, list[tuple[float, int]]] = {}
        for point_id, owner, distance in zip(valid_candidates.tolist(), valid_owner.tolist(), valid_dist.tolist()):
            additions_by_owner.setdefault(int(owner), []).append((float(distance), int(point_id)))

        for owner, additions in additions_by_owner.items():
            additions.sort(key=lambda item: (item[0], item[1]))
            if float(args.max_added_ratio) > 0.0:
                limit = int(np.ceil(float(source_counts[owner]) * float(args.max_added_ratio)))
                additions = additions[: max(0, limit)]
            if int(args.max_added_points_per_object) > 0:
                additions = additions[: int(args.max_added_points_per_object)]
            if not additions:
                continue
            add_ids = np.asarray([point_id for _, point_id in additions], dtype=np.int64)
            masks[add_ids, owner] = True
            assigned_count += int(add_ids.shape[0])

    output_pred_dir = root / "data" / "prediction" / f"{args.output_config}{args.pred_suffix}"
    output_pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_pred_dir / f"{seq_name}.npz",
        pred_masks=masks,
        pred_score=scores,
        pred_classes=classes,
    )

    output_union = np.flatnonzero(masks.any(axis=1)).astype(np.int64)
    output_tmp_dir = root / "data" / "TMP" / args.output_config
    output_tmp_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_tmp_dir / f"{seq_name}_pre_points.npy", output_union)

    output_union_in_support = int(np.count_nonzero(masks[support_ids, :].any(axis=1)))
    return {
        "seq_name": seq_name,
        "pred_config": args.pred_config,
        "support_config": args.support_config,
        "output_config": args.output_config,
        "num_scene_vertices": int(points.shape[0]),
        "num_instances": int(masks.shape[1]),
        "num_support_points": int(support_ids.shape[0]),
        "support_ratio": float(support_ids.shape[0] / max(points.shape[0], 1)),
        "num_original_union": int(original_union.shape[0]),
        "original_union_ratio": float(original_union.shape[0] / max(points.shape[0], 1)),
        "num_candidate_points": int(candidate_ids.shape[0]),
        "num_assigned_points": int(assigned_count),
        "num_output_union": int(output_union.shape[0]),
        "output_union_ratio": float(output_union.shape[0] / max(points.shape[0], 1)),
        "num_output_union_in_support": output_union_in_support,
        "output_union_in_support_ratio_of_support": float(output_union_in_support / max(support_ids.shape[0], 1)),
        "radius": float(args.radius),
        "only_missing": bool(args.only_missing),
        "max_added_ratio": float(args.max_added_ratio),
        "max_added_points_per_object": int(args.max_added_points_per_object),
        "min_source_points": int(args.min_source_points),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--pred-config", required=True)
    parser.add_argument("--support-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--pred-suffix", default="_class_agnostic")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--radius", type=float, default=0.03)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--max-added-ratio", type=float, default=0.0)
    parser.add_argument("--max-added-points-per-object", type=int, default=0)
    parser.add_argument("--min-source-points", type=int, default=20)
    parser.add_argument("--summary-root", default="outputs/stream4d_support_completion_v4_1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    rows = [_complete_one(args, root, seq_name) for seq_name in _read_seq_list(root / args.seq_list)]
    numeric_keys = sorted(
        key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.generic))
    )
    aggregate = {f"mean_{key}": float(np.mean([float(row[key]) for row in rows])) for key in numeric_keys}
    summary = {"args": vars(args), "aggregate": aggregate, "rows": rows}
    out_dir = root / args.summary_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_config}_summary.json"
    out_path.write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[support-completion] wrote {out_path}")


if __name__ == "__main__":
    main()
