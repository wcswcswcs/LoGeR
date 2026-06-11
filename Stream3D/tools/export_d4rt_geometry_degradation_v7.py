from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from tools.d4rt_geometry_diagnostic import _backproject_xy_world, fit_sim3_umeyama
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fit_window_sim3(stream: ScanNetStream, carrier_path: Path, max_anchors: int) -> dict[str, Any]:
    with np.load(carrier_path) as data:
        src_frame = np.asarray(data["src_frame"], dtype=np.int64)
        src_frame_global = np.asarray(data.get("src_frame_global", src_frame), dtype=np.int64)
        src_xy = np.asarray(data["src_xy"], dtype=np.float32)
        xyz_ref = np.asarray(data["xyz_ref"], dtype=np.float32)
        valid = np.asarray(data.get("valid", np.ones(xyz_ref.shape[:2], dtype=bool)), dtype=bool)
    indices = np.arange(src_frame.shape[0], dtype=np.int64)
    local_ok = (src_frame >= 0) & (src_frame < xyz_ref.shape[0])
    indices = indices[local_ok]
    if indices.size > int(max_anchors):
        pick = np.linspace(0, indices.size - 1, num=int(max_anchors), dtype=np.int64)
        indices = indices[pick]
    local = src_frame[indices]
    d4rt = xyz_ref[local, indices]
    ok = np.isfinite(d4rt).all(axis=1) & valid[local, indices]
    indices = indices[ok]
    local = src_frame[indices]
    d4rt = xyz_ref[local, indices]
    rgbd_world = np.full((indices.shape[0], 3), np.nan, dtype=np.float32)
    rgbd_valid = np.zeros((indices.shape[0],), dtype=bool)
    for frame_id in sorted(set(int(v) for v in src_frame_global[indices].tolist())):
        frame_mask = src_frame_global[indices] == int(frame_id)
        world, valid_mask = _backproject_xy_world(stream, int(frame_id), src_xy[indices][frame_mask])
        rgbd_world[frame_mask] = world
        rgbd_valid[frame_mask] = valid_mask
    ok = rgbd_valid & np.isfinite(d4rt).all(axis=1) & np.isfinite(rgbd_world).all(axis=1)
    if np.count_nonzero(ok) < 4:
        raise RuntimeError(f"{carrier_path}: too few finite Sim3 anchors ({int(np.count_nonzero(ok))})")
    fit = fit_sim3_umeyama(d4rt[ok], rgbd_world[ok])
    residual = fit["residual"]
    return {
        "scale": float(fit["scale"]),
        "rotation": np.asarray(fit["rotation"], dtype=np.float64),
        "translation": np.asarray(fit["translation"], dtype=np.float64),
        "anchor_count": int(fit["anchor_count"]),
        "residual_mean": float(np.mean(residual)),
        "residual_median": float(np.median(residual)),
        "residual_p90": float(np.percentile(residual, 90)),
        "residual_p95": float(np.percentile(residual, 95)),
    }


def _transform(points: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    return float(fit["scale"]) * (points @ np.asarray(fit["rotation"]).T) + np.asarray(fit["translation"])


def _group_window_masks(carrier_path: Path, fit: dict[str, Any], exporter, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float]]:
    with np.load(carrier_path) as data:
        src_frame = np.asarray(data["src_frame"], dtype=np.int64)
        src_frame_global = np.asarray(data.get("src_frame_global", src_frame), dtype=np.int64)
        src_mask_id = np.asarray(data.get("src_mask_id", np.zeros_like(src_frame)), dtype=np.int64)
        xyz_ref = np.asarray(data["xyz_ref"], dtype=np.float32)
        valid = np.asarray(data.get("valid", np.ones(xyz_ref.shape[:2], dtype=bool)), dtype=bool)
    groups: dict[tuple[int, int], list[int]] = {}
    for idx, (frame_id, mask_id) in enumerate(zip(src_frame_global.tolist(), src_mask_id.tolist())):
        if int(mask_id) <= 0:
            continue
        groups.setdefault((int(frame_id), int(mask_id)), []).append(int(idx))

    objects: list[dict[str, Any]] = []
    diag = Counter()
    for (frame_id, mask_id), carrier_indices in sorted(groups.items()):
        if len(carrier_indices) < int(args.min_carriers_per_mask):
            diag["dropped_small_carrier_mask"] += 1
            continue
        idx = np.asarray(carrier_indices, dtype=np.int64)
        local = src_frame[idx]
        ok_local = (local >= 0) & (local < xyz_ref.shape[0])
        idx = idx[ok_local]
        local = local[ok_local]
        points = xyz_ref[local, idx]
        ok = np.isfinite(points).all(axis=1) & valid[local, idx]
        points = points[ok]
        if points.shape[0] < int(args.min_carriers_per_mask):
            diag["dropped_invalid_d4rt_points"] += 1
            continue
        world = _transform(points, fit)
        dist, point_ids = exporter.tree.query(world, k=1, distance_upper_bound=float(args.mesh_nn_radius))
        hit = np.isfinite(dist) & (point_ids < exporter.scene_points.shape[0])
        point_set = set(int(v) for v in point_ids[hit].tolist())
        if len(point_set) < int(args.min_points_per_object):
            diag["dropped_small_mesh_mask"] += 1
            continue
        objects.append(
            {
                "frame_id": int(frame_id),
                "mask_id": int(mask_id),
                "carrier_count": int(len(carrier_indices)),
                "point_ids": point_set,
                "centroid": np.mean(world[hit], axis=0) if np.any(hit) else np.zeros((3,), dtype=np.float32),
                "score": float(len(point_set)),
            }
        )
        diag["d4rt_mask_groups_kept"] += 1
        diag["d4rt_points_hit_mesh"] += int(np.count_nonzero(hit))
        diag["d4rt_points_total"] += int(points.shape[0])
    return objects, {f"window_{key}": float(value) for key, value in diag.items()}


class _ObjectDSU:
    def __init__(self, objects: list[dict[str, Any]]) -> None:
        self.parent = list(range(len(objects)))
        self.members = {idx: {idx} for idx in range(len(objects))}
        self.frames = {idx: {int(obj["frame_id"])} for idx, obj in enumerate(objects)}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        rl = self.find(left)
        rr = self.find(right)
        if rl == rr or self.frames[rl] & self.frames[rr]:
            return False
        if len(self.members[rl]) < len(self.members[rr]):
            rl, rr = rr, rl
        self.parent[rr] = rl
        self.members[rl].update(self.members.pop(rr))
        self.frames[rl].update(self.frames.pop(rr))
        return True

    def components(self) -> list[list[int]]:
        return [sorted(v) for _, v in sorted(self.members.items())]


def _merge_normalized(objects: list[dict[str, Any]], alpha: float) -> tuple[list[list[int]], dict[str, float]]:
    if not objects:
        return [], {"normalized_merge_edges": 0.0, "normalized_merge_accepted": 0.0}
    centroids = np.stack([np.asarray(obj["centroid"], dtype=np.float32) for obj in objects], axis=0)
    distances: list[float] = []
    for idx in range(centroids.shape[0]):
        diff = centroids - centroids[idx]
        dist = np.linalg.norm(diff, axis=1)
        dist[idx] = np.inf
        nearest = float(np.min(dist))
        if np.isfinite(nearest):
            distances.append(nearest)
    scale = float(np.median(distances)) if distances else 0.05
    threshold = max(float(alpha) * max(scale, 1e-6), 1e-6)
    dsu = _ObjectDSU(objects)
    edges = 0
    accepted = 0
    for left in range(len(objects)):
        for right in range(left + 1, len(objects)):
            if int(objects[left]["frame_id"]) == int(objects[right]["frame_id"]):
                continue
            dist = float(np.linalg.norm(centroids[left] - centroids[right]))
            if dist <= threshold:
                edges += 1
                accepted += int(dsu.union(left, right))
    return dsu.components(), {
        "normalized_merge_scale": scale,
        "normalized_merge_threshold": threshold,
        "normalized_merge_edges": float(edges),
        "normalized_merge_accepted": float(accepted),
    }


def _write_outputs(objects: list[dict[str, Any]], components: list[list[int]], exporter, args: argparse.Namespace) -> dict[str, float]:
    records: list[set[int]] = []
    scores: list[float] = []
    object_dict: dict[int, dict[str, Any]] = {}
    for component in components:
        point_ids: set[int] = set()
        mask_list: list[tuple[int, int, float]] = []
        carrier_count = 0
        for idx in component:
            obj = objects[int(idx)]
            point_ids.update(obj["point_ids"])
            mask_list.append((int(obj["frame_id"]), int(obj["mask_id"]), float(obj["carrier_count"])))
            carrier_count += int(obj["carrier_count"])
        if len(point_ids) < int(args.min_points_per_object):
            continue
        out_idx = len(records)
        records.append(point_ids)
        scores.append(float(carrier_count * np.sqrt(max(len(point_ids), 1))))
        object_dict[out_idx] = {
            "point_ids": np.asarray(sorted(point_ids), dtype=np.int64),
            "mask_list": sorted(mask_list, key=lambda item: (item[0], item[1])),
            "repre_mask_list": sorted(mask_list, key=lambda item: item[2], reverse=True)[:5],
            "carrier_ids": np.empty((0,), dtype=np.int64),
        }
    masks = np.zeros((exporter.scene_points.shape[0], len(records)), dtype=bool)
    for idx, point_ids in enumerate(records):
        masks[np.asarray(sorted(point_ids), dtype=np.int64), idx] = True
    pred_dir = Path("data/prediction") / f"{args.output_config}_class_agnostic"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_dir / f"{args.seq_name}.npz",
        pred_masks=masks,
        pred_score=np.asarray(scores, dtype=np.float32),
        pred_classes=np.zeros((len(records),), dtype=np.int32),
    )
    tmp_dir = Path("data/TMP") / args.output_config
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pre_points = np.flatnonzero(masks.any(axis=1)).astype(np.int64)
    np.save(tmp_dir / f"{args.seq_name}_pre_points.npy", pre_points)
    object_dir = exporter.stream.object_dir / args.output_config
    object_dir.mkdir(parents=True, exist_ok=True)
    np.save(object_dir / "object_dict.npy", object_dict, allow_pickle=True)
    return {"num_exported_objects": float(len(records)), "num_exported_points": float(pre_points.shape[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description="D4RT geometry replacement diagnostic for ScanNet probe scenes.")
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--max-anchors-per-window", type=int, default=2000)
    parser.add_argument("--min-carriers-per-mask", type=int, default=8)
    parser.add_argument("--min-points-per-object", type=int, default=30)
    parser.add_argument("--mesh-nn-radius", type=float, default=0.10)
    parser.add_argument("--merge-normalized", action="store_true")
    parser.add_argument("--normalized-alpha", type=float, default=1.0)
    parser.add_argument("--summary-root", default="outputs/v7_d4rt_geometry_degradation")
    args = parser.parse_args()

    stream = ScanNetStream(seq_name=args.seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))
    from stream4d.export_scannet import ScanNetExporter

    exporter = ScanNetExporter(stream, output_config=args.output_config, export_nn_radius=float(args.mesh_nn_radius))
    scene_dir = Path(args.debug_root) / args.seq_name
    carrier_paths = sorted(scene_dir.glob("carriers_window*.npz"))
    if not carrier_paths:
        raise FileNotFoundError(f"no carriers_window*.npz under {scene_dir}")

    all_objects: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    diag = Counter()
    for carrier_path in carrier_paths:
        fit = _fit_window_sim3(stream, carrier_path, max_anchors=int(args.max_anchors_per_window))
        objects, object_diag = _group_window_masks(carrier_path, fit, exporter, args)
        all_objects.extend(objects)
        for key, value in object_diag.items():
            diag[key] += float(value)
        summary_rows.append(
            {
                "carrier_file": str(carrier_path),
                "objects": int(len(objects)),
                **{key: value for key, value in fit.items() if key not in {"rotation", "translation"}},
                **object_diag,
            }
        )

    if args.merge_normalized:
        components, merge_diag = _merge_normalized(all_objects, alpha=float(args.normalized_alpha))
    else:
        components = [[idx] for idx in range(len(all_objects))]
        merge_diag = {"normalized_merge_edges": 0.0, "normalized_merge_accepted": 0.0}
    export_diag = _write_outputs(all_objects, components, exporter, args)
    summary = {
        "args": vars(args),
        "geometry_path": "G3_D4RT_geometry_normalized_merge" if args.merge_normalized else "G1_D4RT_shared_reference_point_map",
        "is_diagnostic_only": True,
        "uses_gt": False,
        "sim3_usage": "export/eval adapter only; no GT labels used",
        "windows": summary_rows,
        "num_raw_objects": int(len(all_objects)),
        "num_components": int(len(components)),
        **{key: float(value) for key, value in diag.items()},
        **merge_diag,
        **export_diag,
    }
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.output_config}_{args.seq_name}_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = build_prediction_manifest(
        root=".",
        output_config=args.output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(args.debug_root)],
        pre_points_policy="recompute",
        support_policy=summary["geometry_path"],
        notes="D4RT geometry degradation diagnostic; Sim3 is used only as export/evaluation adapter.",
        extra={
            "geometry_path": summary["geometry_path"],
            "sim3_usage": summary["sim3_usage"],
            "summary_path": str(summary_path),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=".")
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
