from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d_native.d4rt_scene_builder import source_xy_from_uv
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v36_external_downstream_assignment import RegionNode, _load_masks, _load_tubes
from tools.run_v37_temporal_curriculum import (
    _aggregate_seed_rows,
    _region_diagnostics,
    _seed_metrics,
    _write_csv,
    _write_json,
)


SOURCE_NAME = "v37_tube_split"


def _visible(tube: Any, local_idx: int, *, min_visibility: float, min_confidence: float) -> bool:
    uv = np.asarray(tube.uv[local_idx], dtype=np.float32)
    return bool(
        np.isfinite(uv).all()
        and 0.0 <= float(uv[0]) <= 1.0
        and 0.0 <= float(uv[1]) <= 1.0
        and float(tube.visibility[local_idx]) >= float(min_visibility)
        and float(tube.confidence[local_idx]) >= float(min_confidence)
    )


def _tube_points_by_frame(tubes: list[Any], shape_by_frame: dict[int, tuple[int, int]], args: argparse.Namespace) -> dict[int, list[tuple[int, int, int]]]:
    points: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for tube in tubes:
        for local_idx, frame_id in enumerate(np.asarray(tube.target_frames_global, dtype=np.int64).tolist()):
            frame = int(frame_id)
            shape = shape_by_frame.get(frame)
            if shape is None or not _visible(tube, local_idx, min_visibility=float(args.min_visibility), min_confidence=float(args.min_confidence)):
                continue
            x, y = source_xy_from_uv(tube.uv[local_idx], image_width=int(shape[1]), image_height=int(shape[0]))
            points[frame].append((int(x), int(y), int(tube.tube_id)))
    return points


def _farthest_init(points_xy: np.ndarray, k: int) -> np.ndarray:
    centers = [points_xy[0].astype(np.float32)]
    while len(centers) < k:
        stacked = np.stack(centers, axis=0)
        d2 = np.sum((points_xy[:, None, :] - stacked[None, :, :]) ** 2, axis=2)
        centers.append(points_xy[int(np.argmax(np.min(d2, axis=1)))].astype(np.float32))
    return np.stack(centers, axis=0)


def _kmeans(points_xy: np.ndarray, k: int, *, iterations: int = 8) -> np.ndarray:
    centers = _farthest_init(points_xy, int(k))
    for _ in range(int(iterations)):
        d2 = np.sum((points_xy[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        new_centers = centers.copy()
        for idx in range(int(k)):
            members = points_xy[labels == idx]
            if members.size:
                new_centers[idx] = np.mean(members, axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers


def _component_masks(label_map: np.ndarray, mask: np.ndarray, label_value: int, min_area: int) -> list[np.ndarray]:
    binary = np.asarray((label_map == int(label_value)) & mask, dtype=np.uint8)
    count, cc = cv2.connectedComponents(binary, connectivity=8)
    out = []
    for comp_id in range(1, int(count)):
        part = cc == comp_id
        if int(part.sum()) >= int(min_area):
            out.append(part)
    return out


def _split_mask(mask: np.ndarray, support_points: list[tuple[int, int, int]], args: argparse.Namespace) -> tuple[list[np.ndarray], dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    points = []
    seen_xy = set()
    for x, y, tube_id in support_points:
        if 0 <= int(y) < mask.shape[0] and 0 <= int(x) < mask.shape[1] and bool(mask[int(y), int(x)]):
            key = (int(x), int(y))
            if key not in seen_xy:
                seen_xy.add(key)
                points.append((int(x), int(y), int(tube_id)))
    if len(points) < int(args.min_support_points):
        return [mask], {"split": False, "reason": "insufficient_support_points", "support_point_count": len(points)}
    max_k = min(int(args.max_splits), len(points))
    if max_k < 2:
        return [mask], {"split": False, "reason": "max_k_lt_2", "support_point_count": len(points)}
    k = min(max_k, max(2, int(np.ceil(len(points) / max(int(args.points_per_split), 1)))))
    xy = np.asarray([[x, y] for x, y, _ in points], dtype=np.float32)
    centers = _kmeans(xy, k)
    if centers.shape[0] >= 2:
        d2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        d2[d2 == 0.0] = np.inf
        if float(np.sqrt(np.min(d2))) < float(args.min_center_distance):
            return [mask], {
                "split": False,
                "reason": "centers_too_close",
                "support_point_count": len(points),
                "min_center_distance": float(np.sqrt(np.min(d2))),
            }
    yy, xx = np.nonzero(mask)
    coords = np.stack([xx.astype(np.float32), yy.astype(np.float32)], axis=1)
    d2 = np.sum((coords[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1).astype(np.int32)
    label_map = np.full(mask.shape, -1, dtype=np.int32)
    label_map[yy, xx] = labels
    parts: list[np.ndarray] = []
    for label_value in range(int(k)):
        parts.extend(_component_masks(label_map, mask, label_value, int(args.min_child_area)))
    if len(parts) < 2:
        return [mask], {"split": False, "reason": "less_than_two_valid_children", "support_point_count": len(points)}
    covered = int(sum(int(part.sum()) for part in parts))
    return parts, {
        "split": True,
        "reason": "ok",
        "support_point_count": len(points),
        "child_count": int(len(parts)),
        "parent_area": area,
        "covered_area": covered,
    }


def _raw_frame_paths(mask_root: Path, scene: str, source: str, mode: str) -> list[Path]:
    root = mask_root / scene / source / mode
    return sorted(root.glob(f"{source}_frame*_masks.npz"))


def _frame_from_path(path: Path) -> int:
    return int(path.stem.split("_frame", 1)[1].split("_", 1)[0])


def _save_masks(output_dir: Path, frame_id: int, masks: list[np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if masks:
        arr = np.stack([np.asarray(mask, dtype=bool) for mask in masks], axis=0)
    else:
        arr = np.zeros((0, 1, 1), dtype=bool)
    np.savez_compressed(output_dir / f"{SOURCE_NAME}_frame{int(frame_id):06d}_masks.npz", masks=arr, scores=np.ones((arr.shape[0],), dtype=np.float32))


def _variant_keep(variant: str, *, support_count: int, raw_area: int, area_p50: float, area_p75: float, was_split: bool) -> bool:
    if was_split:
        return True
    if variant == "tube_voronoi_keep_all":
        return True
    if variant == "tube_voronoi_unknown_large":
        return bool(raw_area <= area_p75 or support_count >= 1)
    if variant == "tube_voronoi_supported_or_small":
        return bool(support_count >= 1 or raw_area <= area_p50)
    if variant == "tube_voronoi_supported_only":
        return bool(support_count >= 1)
    raise ValueError(variant)


def _nodes_from_output(mask_root: Path, scene: str, variant: str, min_region_area: int) -> tuple[list[RegionNode], dict[int, np.ndarray], dict[str, Any]]:
    setattr(_load_masks, "max_regions_per_scene", 0)
    return _load_masks(mask_root, scene, SOURCE_NAME, variant, min_region_area)


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _read_split(Path(args.split))
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    out_root = Path(args.output_root)
    source_root = out_root / "sources"
    summary_rows = []
    scene_rows = []
    manifests: dict[str, Any] = {}
    for scene in scenes:
        raw_paths = _raw_frame_paths(Path(args.mask_root), scene, args.source, args.mode)
        if not raw_paths:
            raise FileNotFoundError(f"no raw masks for {scene}: {args.mask_root}/{scene}/{args.source}/{args.mode}")
        shape_by_frame: dict[int, tuple[int, int]] = {}
        raw_by_frame: dict[int, list[np.ndarray]] = {}
        raw_areas = []
        for path in raw_paths:
            frame = _frame_from_path(path)
            data = np.load(path)
            masks = [np.asarray(mask, dtype=bool) for mask in np.asarray(data["masks"], dtype=bool)]
            raw_by_frame[frame] = masks
            if masks:
                shape_by_frame[frame] = masks[0].shape
                raw_areas.extend([int(mask.sum()) for mask in masks])
        area_p50 = float(np.quantile(raw_areas, 0.50)) if raw_areas else 0.0
        area_p75 = float(np.quantile(raw_areas, 0.75)) if raw_areas else 0.0
        tubes = _load_tubes(scene, args)
        points_by_frame = _tube_points_by_frame(tubes, shape_by_frame, args)
        split_stats = Counter()
        variant_masks: dict[str, dict[int, list[np.ndarray]]] = {
            variant: {frame: [] for frame in raw_by_frame} for variant in variants
        }
        for frame, masks in raw_by_frame.items():
            frame_points = points_by_frame.get(frame, [])
            for mask in masks:
                support = [(x, y, tid) for x, y, tid in frame_points if bool(mask[int(y), int(x)])]
                parts, info = _split_mask(mask, support, args)
                split_stats[f"split_{info['reason']}"] += 1
                split_stats["parent_masks"] += 1
                if info.get("split"):
                    split_stats["split_parent_masks"] += 1
                    split_stats["split_child_masks"] += len(parts)
                for variant in variants:
                    keep = _variant_keep(
                        variant,
                        support_count=len(support),
                        raw_area=int(mask.sum()),
                        area_p50=area_p50,
                        area_p75=area_p75,
                        was_split=bool(info.get("split")),
                    )
                    if keep:
                        variant_masks[variant][frame].extend(parts if bool(info.get("split")) else [mask])
        for variant in variants:
            for frame, masks in variant_masks[variant].items():
                _save_masks(source_root / scene / SOURCE_NAME / variant, frame, masks)
            nodes, labels_by_frame, mask_manifest = _nodes_from_output(source_root, scene, variant, int(args.min_region_area))
            diagnostics, gt_area = _region_diagnostics(scene, nodes, labels_by_frame, compute_rgb=False)
            active = {int(node.node_id) for node in nodes}
            row = _seed_metrics(
                scene,
                nodes,
                labels_by_frame,
                diagnostics,
                gt_area,
                active,
                variant=f"C10_{variant}",
                status="tube_seeded_split_source",
            )
            row["source_root"] = str(source_root)
            row["source"] = SOURCE_NAME
            row["mode"] = variant
            scene_rows.append(row)
        manifests[scene] = {
            "raw_frame_count": int(len(raw_by_frame)),
            "raw_mask_count": int(sum(len(v) for v in raw_by_frame.values())),
            "area_p50": area_p50,
            "area_p75": area_p75,
            "tube_point_frames": int(len(points_by_frame)),
            "tube_point_count": int(sum(len(v) for v in points_by_frame.values())),
            "split_stats": dict(split_stats),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
            "method_inputs": "watershed masks + D4RT tube UV/visibility/confidence",
        }
    summary_rows = _aggregate_seed_rows(scene_rows)
    _write_csv(out_root / "tube_seeded_split_scene_rows.csv", scene_rows)
    _write_csv(out_root / "tube_seeded_split_summary.csv", summary_rows)
    _write_json(out_root / "tube_seeded_split_summary.json", summary_rows)
    payload = {
        "plan": "docs/stream4d_v37_temporal_curriculum_masklet_plan.md",
        "phase": "Phase C tube-seeded split repair",
        "source_root": str(source_root),
        "source": SOURCE_NAME,
        "summary": summary_rows,
        "manifests": manifests,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    _write_json(out_root / "tube_seeded_split_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/evaluate v37 D4RT tube-seeded same-frame split source.")
    parser.add_argument("--mask-root", default="outputs/audit/v36_external_mask_source_all")
    parser.add_argument("--source", default="watershed")
    parser.add_argument("--mode", default="all_masks")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--output-root", default="outputs/audit/v37_same_frame_objectlets/tube_seeded_split")
    parser.add_argument("--variants", default="tube_voronoi_keep_all,tube_voronoi_unknown_large,tube_voronoi_supported_or_small,tube_voronoi_supported_only")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--min-child-area", type=int, default=64)
    parser.add_argument("--min-support-points", type=int, default=2)
    parser.add_argument("--max-splits", type=int, default=4)
    parser.add_argument("--points-per-split", type=int, default=3)
    parser.add_argument("--min-center-distance", type=float, default=32.0)
    args = parser.parse_args()
    print(json.dumps(_json_safe(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

