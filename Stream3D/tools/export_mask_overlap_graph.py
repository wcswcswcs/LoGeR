from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream


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


class ComponentIndex:
    def __init__(self, frames: list[int], masks: list[int]) -> None:
        self.parent = list(range(len(frames)))
        self.members: dict[int, set[int]] = {idx: {idx} for idx in range(len(frames))}
        self.frame_masks: dict[int, dict[int, int]] = {
            idx: {int(frames[idx]): int(masks[idx])} for idx in range(len(frames))
        }

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def can_union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        frames_left = self.frame_masks[root_left]
        frames_right = self.frame_masks[root_right]
        for frame_id, mask_id in frames_left.items():
            other = frames_right.get(frame_id)
            if other is not None and int(other) != int(mask_id):
                return False
        return True

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right or not self.can_union(root_left, root_right):
            return False
        if len(self.members[root_left]) < len(self.members[root_right]):
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.members[root_left].update(self.members.pop(root_right))
        self.frame_masks[root_left].update(self.frame_masks.pop(root_right))
        return True

    def components(self) -> list[list[int]]:
        return [sorted(self.members[root]) for root in sorted(self.members)]


def _load_observations(debug_root: Path, seq_name: str, min_coverage: float) -> list[tuple[int, int, float]]:
    seq_dir = debug_root / seq_name
    if not seq_dir.exists():
        raise FileNotFoundError(seq_dir)
    best: dict[tuple[int, int], float] = {}
    for window_path in sorted(seq_dir.glob("local_props_window*.json")):
        with window_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for prop in payload.get("proposals", []):
            for item in prop.get("mask_observations", []):
                coverage = float(item.get("coverage", 0.0))
                if coverage < float(min_coverage):
                    continue
                key = (int(item["frame_id"]), int(item["mask_id"]))
                best[key] = max(coverage, best.get(key, 0.0))
    observations = [(frame_id, mask_id, coverage) for (frame_id, mask_id), coverage in best.items()]
    observations.sort(key=lambda item: item[2], reverse=True)
    return observations


def _backproject_observations(
    exporter: ScanNetExporter,
    observations: list[tuple[int, int, float]],
    min_points: int,
    nn_radius: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    total_queries = 0
    total_hits = 0
    dropped_small = 0
    for obs_id, (frame_id, mask_id, coverage) in enumerate(observations):
        point_ids, query_count = exporter._backproject_mask(  # diagnostic/experimental tool.
            int(frame_id),
            int(mask_id),
            nn_radius=float(nn_radius),
        )
        total_queries += int(query_count)
        total_hits += int(point_ids.shape[0])
        point_ids = np.unique(point_ids.astype(np.int64))
        if point_ids.shape[0] < int(min_points):
            dropped_small += 1
            continue
        rows.append(
            {
                "obs_id": int(obs_id),
                "frame_id": int(frame_id),
                "mask_id": int(mask_id),
                "coverage": float(coverage),
                "point_ids": point_ids,
            }
        )
    return rows, {
        "backproject_queries": float(total_queries),
        "backproject_hits": float(total_hits),
        "backproject_hit_rate": float(total_hits / max(total_queries, 1)),
        "dropped_small_observations": float(dropped_small),
    }


def _edge_candidates(rows: list[dict[str, Any]], overlap_mode: str) -> list[tuple[float, int, int, int]]:
    point_to_obs: dict[int, list[int]] = {}
    for idx, row in enumerate(rows):
        for point_id in row["point_ids"].tolist():
            point_to_obs.setdefault(int(point_id), []).append(idx)
    intersections: Counter[tuple[int, int]] = Counter()
    for owners in point_to_obs.values():
        if len(owners) < 2:
            continue
        owners = sorted(set(owners))
        for pos, left in enumerate(owners):
            for right in owners[pos + 1 :]:
                if int(rows[left]["frame_id"]) == int(rows[right]["frame_id"]):
                    continue
                intersections[(left, right)] += 1

    sizes = np.asarray([len(row["point_ids"]) for row in rows], dtype=np.float64)
    edges: list[tuple[float, int, int, int]] = []
    for (left, right), shared in intersections.items():
        if overlap_mode == "min_ioc":
            denom = min(float(sizes[left]), float(sizes[right]))
        elif overlap_mode == "iou":
            denom = float(sizes[left] + sizes[right] - shared)
        elif overlap_mode == "left_ioc":
            denom = float(sizes[left])
        else:
            raise ValueError(f"Unsupported overlap mode: {overlap_mode}")
        score = float(shared / max(denom, 1.0))
        edges.append((score, int(shared), int(left), int(right)))
    edges.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return edges


def _build_components(
    rows: list[dict[str, Any]],
    min_shared_points: int,
    min_overlap: float,
    overlap_mode: str,
) -> tuple[list[list[int]], dict[str, float]]:
    frames = [int(row["frame_id"]) for row in rows]
    masks = [int(row["mask_id"]) for row in rows]
    components = ComponentIndex(frames, masks)
    edges = _edge_candidates(rows, overlap_mode)
    accepted = 0
    rejected_weak = 0
    rejected_conflict = 0
    for score, shared, left, right in edges:
        if int(shared) < int(min_shared_points) or float(score) < float(min_overlap):
            rejected_weak += 1
            continue
        if not components.can_union(left, right):
            rejected_conflict += 1
            continue
        if components.union(left, right):
            accepted += 1
    return components.components(), {
        "edge_candidates": float(len(edges)),
        "accepted_edges": float(accepted),
        "rejected_weak_edges": float(rejected_weak),
        "rejected_conflict_edges": float(rejected_conflict),
    }


def _component_score(rows: list[dict[str, Any]], member_ids: list[int], mode: str) -> float:
    coverages = np.asarray([float(rows[idx]["coverage"]) for idx in member_ids], dtype=np.float64)
    point_count = len(set().union(*(set(rows[idx]["point_ids"].tolist()) for idx in member_ids)))
    if mode == "mean_coverage":
        return float(np.mean(coverages))
    if mode == "sum_coverage":
        return float(np.sum(coverages))
    if mode == "observations":
        return float(len(member_ids))
    if mode == "points":
        return float(point_count)
    if mode == "coverage_points":
        return float(np.mean(coverages) * np.sqrt(max(point_count, 1)))
    raise ValueError(f"Unsupported score mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3D overlap graph from unique 2D mask observations.")
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--min-coverage", type=float, default=0.005)
    parser.add_argument("--min-points-per-mask", type=int, default=100)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--overlap-mode", default="min_ioc", choices=["min_ioc", "iou", "left_ioc"])
    parser.add_argument("--min-shared-points", type=int, default=25)
    parser.add_argument("--min-overlap", type=float, default=0.25)
    parser.add_argument("--min-component-observations", type=int, default=2)
    parser.add_argument("--min-component-frames", type=int, default=2)
    parser.add_argument("--score-mode", default="coverage_points", choices=["mean_coverage", "sum_coverage", "observations", "points", "coverage_points"])
    parser.add_argument("--summary-root", default="outputs/mask_overlap_graph")
    args = parser.parse_args()

    stream = ScanNetStream(seq_name=args.seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(stream, output_config=args.output_config, export_nn_radius=args.export_nn_radius)
    observations = _load_observations(Path(args.debug_root), args.seq_name, args.min_coverage)
    rows, backproject_diag = _backproject_observations(
        exporter,
        observations,
        min_points=args.min_points_per_mask,
        nn_radius=args.export_nn_radius,
    )
    components, graph_diag = _build_components(
        rows,
        min_shared_points=args.min_shared_points,
        min_overlap=args.min_overlap,
        overlap_mode=args.overlap_mode,
    )

    masks: list[np.ndarray] = []
    scores: list[float] = []
    kept_components: list[dict[str, Any]] = []
    dropped_components = 0
    for component in components:
        frames = {int(rows[idx]["frame_id"]) for idx in component}
        if len(component) < int(args.min_component_observations) or len(frames) < int(args.min_component_frames):
            dropped_components += 1
            continue
        point_ids = sorted(set().union(*(set(rows[idx]["point_ids"].tolist()) for idx in component)))
        if not point_ids:
            dropped_components += 1
            continue
        mask = np.zeros((exporter.scene_points.shape[0],), dtype=bool)
        mask[np.asarray(point_ids, dtype=np.int64)] = True
        masks.append(mask)
        score = _component_score(rows, component, args.score_mode)
        scores.append(score)
        kept_components.append(
            {
                "num_observations": int(len(component)),
                "num_frames": int(len(frames)),
                "num_points": int(len(point_ids)),
                "score": float(score),
            }
        )

    if masks:
        pred_masks = np.stack(masks, axis=1).astype(bool, copy=False)
        pred_score = np.asarray(scores, dtype=np.float32)
    else:
        pred_masks = np.zeros((exporter.scene_points.shape[0], 0), dtype=bool)
        pred_score = np.zeros((0,), dtype=np.float32)
    pred_classes = np.zeros((pred_score.shape[0],), dtype=np.int32)
    pred_dir = Path("data/prediction") / f"{args.output_config}_class_agnostic"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{args.seq_name}.npz",
        pred_masks=pred_masks,
        pred_score=pred_score,
        pred_classes=pred_classes,
    )
    tmp_dir = Path("data/TMP") / args.output_config
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pre_points = np.flatnonzero(pred_masks.any(axis=1)).astype(np.int64)
    np.save(tmp_dir / f"{args.seq_name}_pre_points.npy", pre_points)

    summary = {
        "args": vars(args),
        "loaded_observations": len(observations),
        "backprojected_observations": len(rows),
        "raw_components": len(components),
        "kept_components": int(pred_score.shape[0]),
        "dropped_components": int(dropped_components),
        "union_points": int(pre_points.shape[0]),
        "score_min": float(np.min(pred_score)) if pred_score.size else 0.0,
        "score_mean": float(np.mean(pred_score)) if pred_score.size else 0.0,
        "score_max": float(np.max(pred_score)) if pred_score.size else 0.0,
        **backproject_diag,
        **graph_diag,
        "components_preview": kept_components[:50],
    }
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{args.output_config}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, ensure_ascii=False)
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
