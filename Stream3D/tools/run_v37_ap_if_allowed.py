from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split
from tools.run_v37_4d_if_allowed import (
    _build_scene_state,
    _merge_components_rgb_temporal_topk,
)
from tools.run_v37_temporal_curriculum import _load_masks


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _component_from_variant(state, variant: str) -> tuple[list[list[int]], dict[str, Any]]:
    if variant == "I4_sparse_rgb_temporal_gap1_rgb099_top1":
        return _merge_components_rgb_temporal_topk(
            state,
            state.components,
            min_rgb_similarity=0.99,
            max_frame_gap=1,
            max_rgb_fallback_per_component=1,
        )
    if variant == "I4_sparse_rgb_temporal_gap2_rgb099_top1":
        return _merge_components_rgb_temporal_topk(
            state,
            state.components,
            min_rgb_similarity=0.99,
            max_frame_gap=2,
            max_rgb_fallback_per_component=1,
        )
    if variant == "I4_sparse_rgb_temporal_gap2_rgb0995_top1":
        return _merge_components_rgb_temporal_topk(
            state,
            state.components,
            min_rgb_similarity=0.995,
            max_frame_gap=2,
            max_rgb_fallback_per_component=1,
        )
    raise ValueError(f"Unsupported AP export variant: {variant}")


def _component_mask_xy(
    labels_by_frame: dict[int, np.ndarray],
    frame_id: int,
    node_id: int,
    depth_shape: tuple[int, int],
) -> np.ndarray:
    mask = labels_by_frame[int(frame_id)] == int(node_id) + 1
    if mask.shape != depth_shape:
        mask = cv2.resize(mask.astype(np.uint8), (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack([xs.astype(np.int64), ys.astype(np.int64)], axis=1)


def _gt_coverage_ratio(gt_path: Path, point_union: np.ndarray) -> tuple[float | None, int, int]:
    if not gt_path.exists():
        return None, 0, 0
    gt = np.loadtxt(gt_path, dtype=np.int64)
    valid_gt = sorted(int(v) for v in np.unique(gt) if int(v) > 0)
    if not valid_gt:
        return None, 0, 0
    covered = set(int(v) for v in np.unique(gt[point_union]) if int(v) > 0) if point_union.size else set()
    return float(len(covered) / len(valid_gt)), int(len(covered)), int(len(valid_gt))


def _export_scene(args: argparse.Namespace, scene: str, state, components: list[list[int]]) -> dict[str, Any]:
    stream = ScanNetStream(seq_name=scene)
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_nn_radius=float(args.export_nn_radius),
        export_support_mode="mask_backproject",
        export_mask_sample_stride=int(args.export_mask_sample_stride),
        export_mask_max_pixels=int(args.export_mask_max_pixels),
        export_min_points_per_object=int(args.export_min_points_per_object),
        export_score_mode="area",
    )
    nodes_by_id = {int(node.node_id): node for node in state.nodes}
    _, labels_by_frame, _manifest = _load_masks(Path(args.mask_root), scene, args.source, args.mode, int(args.min_region_area))
    depth_shape_by_frame: dict[int, tuple[int, int]] = {}
    object_records = []
    object_dict = {}
    mask_observation_count = 0
    backproject_pixel_count = 0
    backproject_hit_count = 0
    for object_id, component in enumerate(components):
        point_ids: set[int] = set()
        mask_list = []
        for node_id in component:
            node = nodes_by_id[int(node_id)]
            frame_id = int(node.frame_id)
            if frame_id not in depth_shape_by_frame:
                depth_shape_by_frame[frame_id] = exporter.stream.load_depth(frame_id).shape
            xy = _component_mask_xy(labels_by_frame, frame_id, int(node.node_id), depth_shape_by_frame[frame_id])
            if xy.size == 0:
                continue
            hit_ids, _dist = exporter._backproject_xy(frame_id, xy, nn_radius=float(args.export_nn_radius))
            backproject_pixel_count += int(xy.shape[0])
            backproject_hit_count += int(hit_ids.shape[0])
            point_ids.update(int(v) for v in hit_ids.tolist())
            mask_list.append((int(frame_id), int(node.mask_index), float(node.area)))
            mask_observation_count += 1
        sorted_points = sorted(point_ids)
        object_dict[int(object_id)] = {
            "point_ids": np.asarray(sorted_points, dtype=np.int64),
            "mask_list": mask_list,
            "repre_mask_list": mask_list[: min(len(mask_list), 8)],
            "score": float(len(sorted_points)),
            "area_score": float(len(sorted_points)),
            "source_variant": args.variant,
        }
        object_records.append({
            "object_id": int(object_id),
            "point_ids": set(sorted_points),
            "score": float(len(sorted_points)),
            "area_score": float(len(sorted_points)),
        })
    diag = exporter._write_outputs(object_records, object_dict, np.zeros((exporter.scene_points.shape[0],), dtype=np.uint16))
    pred_path = Path("data/prediction") / f"{args.output_config}_class_agnostic" / f"{scene}.npz"
    with np.load(pred_path) as pred:
        pred_masks = np.asarray(pred["pred_masks"], dtype=bool)
        point_union = np.flatnonzero(pred_masks.any(axis=1)).astype(np.int64)
    gt_ratio, gt_covered, gt_total = _gt_coverage_ratio(Path(args.gt_path) / f"{scene}.txt", point_union)
    diag.update({
        "scene": scene,
        "variant": args.variant,
        "num_components": int(len(components)),
        "mask_observation_count": int(mask_observation_count),
        "backproject_pixel_count": int(backproject_pixel_count),
        "backproject_hit_count": int(backproject_hit_count),
        "backproject_hit_rate": float(backproject_hit_count / max(backproject_pixel_count, 1)),
        "pre_percent": float(diag["num_exported_points"] / max(diag["num_scene_points"], 1.0)),
        "union_percent": float(point_union.shape[0] / max(pred_masks.shape[0], 1)),
        "mesh_coverage": float(point_union.shape[0] / max(pred_masks.shape[0], 1)),
        "covered_GT_instance_ratio": gt_ratio,
        "covered_GT_instances": int(gt_covered),
        "total_GT_instances": int(gt_total),
        "prediction_path": str(pred_path),
    })
    return diag


def run(args: argparse.Namespace) -> dict[str, Any]:
    decision = json.loads(Path(args.memory_decision).read_text(encoding="utf-8"))
    if decision.get("final_status") != "GO_4D_MEMORY":
        raise RuntimeError(f"4D memory decision is not eligible for AP export: {args.memory_decision}")
    if args.variant == "":
        args.variant = str(decision.get("best_variant"))
    setattr(_load_masks, "max_regions_per_scene", int(args.max_regions_per_scene))
    scenes = _read_split(Path(args.split))
    rows = []
    pair_row_count = 0
    for scene in scenes:
        state, pair_row_count = _build_scene_state(scene, args, pair_row_count)
        components, _info = _component_from_variant(state, args.variant)
        rows.append(_export_scene(args, scene, state, components))
    summary = {
        "plan": "docs/stream4d_v37_temporal_curriculum_masklet_plan.md",
        "phase": "J_ap_if_allowed_export",
        "output_config": args.output_config,
        "variant": args.variant,
        "memory_decision": args.memory_decision,
        "is_diagnostic_only": True,
        "is_method_result": False,
        "uses_rgbd_for_prediction": True,
        "uses_pose_for_prediction": True,
        "uses_scannet_mesh_for_prediction": True,
        "scene_rows": rows,
        "mean_pre_percent": float(np.mean([float(row["pre_percent"]) for row in rows])) if rows else None,
        "mean_union_percent": float(np.mean([float(row["union_percent"]) for row in rows])) if rows else None,
        "mean_mesh_coverage": float(np.mean([float(row["mesh_coverage"]) for row in rows])) if rows else None,
        "mean_covered_GT_instance_ratio": float(
            np.mean([float(row["covered_GT_instance_ratio"]) for row in rows if row["covered_GT_instance_ratio"] is not None])
        ) if any(row["covered_GT_instance_ratio"] is not None for row in rows) else None,
    }
    manifest = build_prediction_manifest(
        output_config=args.output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.memory_decision],
        pre_points_policy="recompute",
        support_policy="external_mask_backproject",
        notes="v37 Phase J evaluation-only AP export from 4D memory components; uses ScanNet RGB-D/pose/mesh as materialization bridge.",
        extra={
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
            "forbidden_for_method_table": True,
            "phase": "J_ap_if_allowed_export",
            "temporal_stage": args.variant,
            "mask_source": f"{args.source}:{args.mode}",
            "alignment_source": "scannet_rgbd_pose_mesh_export_bridge",
            "alignment_used_for_prediction": True,
        },
    )
    write_prediction_manifest(args.output_config, manifest, pred_suffix="class_agnostic")
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "ap_export_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v37 Phase J evaluation-only AP predictions after 4D memory gate.")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--mask-root", default="outputs/audit/v37_dino_compact_filter_sources")
    parser.add_argument("--source", default="v37_dino_compact_filter")
    parser.add_argument("--mode", default="dino_k2_compact060_filter")
    parser.add_argument("--memory-decision", default="outputs/audit/v37_4d_if_allowed_i4_sparse/4d_memory_decision.json")
    parser.add_argument("--variant", default="")
    parser.add_argument("--output-config", default="v37_i4_sparse_ap_eval_probe5")
    parser.add_argument("--output-root", default="outputs/audit/v37_ap_if_allowed_i4_sparse")
    parser.add_argument("--gt-path", default="data/scannet/gt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--max-regions-per-scene", type=int, default=0)
    parser.add_argument("--max-support-pairs-per-tube", type=int, default=20000)
    parser.add_argument("--max-same-frame-pairs-per-frame", type=int, default=4000)
    parser.add_argument("--max-allpair-samples-per-scene", type=int, default=30000)
    parser.add_argument("--max-shuffled-pair-rows-per-scene", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=3701)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-mask-sample-stride", type=int, default=2)
    parser.add_argument("--export-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--export-min-points-per-object", type=int, default=1)
    summary = run(parser.parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
