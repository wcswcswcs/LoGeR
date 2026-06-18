from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.object_field import ObjectFieldCandidate
from stream4d_native.object_field_native_export import (
    NativeObjectFieldExportConfig,
    export_object_fields_to_native_points,
)
from stream4d_native.semantic_material_inference import (
    SemanticMaterialInferenceConfig,
    TubeAttachmentScore,
    run_semantic_material_inference,
)
from tools.run_v26_object_quality_diagnostics import _json_safe
from tools.run_v36_external_downstream_assignment import RegionNode, _collect_observations, _load_masks, _load_tubes
from tools.run_v37_same_frame_oracle_rgb_split import _boundary_split
from tools.run_v37_temporal_curriculum import (
    _components_chain_then_closure,
    _drop_rgb_incoherent_components,
    _filter_edges_by_rgb,
    _filtered_edges,
    _frame_rank_map,
    _isolate_rgb_outlier_nodes,
    _safe_div,
    _split_components_by_rgb,
    _support_pair_counts,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _write_points_npz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        np.savez_compressed(
            path,
            object_id=np.zeros((0,), dtype=np.int64),
            primary_field_id=np.zeros((0,), dtype=np.int64),
            tube_id=np.zeros((0,), dtype=np.int64),
            frame_id=np.zeros((0,), dtype=np.int64),
            local_point_index=np.zeros((0,), dtype=np.int64),
            xyz=np.zeros((0, 3), dtype=np.float32),
            uv=np.zeros((0, 2), dtype=np.float32),
            visibility=np.zeros((0,), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
        )
        return
    np.savez_compressed(
        path,
        object_id=np.asarray([row["object_id"] for row in rows], dtype=np.int64),
        primary_field_id=np.asarray([row["primary_field_id"] for row in rows], dtype=np.int64),
        tube_id=np.asarray([row["tube_id"] for row in rows], dtype=np.int64),
        frame_id=np.asarray([row["frame_id"] for row in rows], dtype=np.int64),
        local_point_index=np.asarray([row["local_point_index"] for row in rows], dtype=np.int64),
        xyz=np.asarray([[row["x"], row["y"], row["z"]] for row in rows], dtype=np.float32),
        uv=np.asarray([[row["u"], row["v"]] for row in rows], dtype=np.float32),
        visibility=np.asarray([row["visibility"] for row in rows], dtype=np.float32),
        confidence=np.asarray([row["confidence"] for row in rows], dtype=np.float32),
    )


def _node_mask(labels_by_frame: dict[int, np.ndarray], node: Any) -> np.ndarray:
    return np.asarray(labels_by_frame[int(node.frame_id)] == int(node.node_id) + 1, dtype=bool)


def _rgb_diagnostics_no_gt(scene: str, nodes: list[Any], labels_by_frame: dict[int, np.ndarray]) -> dict[int, dict[str, Any]]:
    from stream4d.scannet_stream import ScanNetStream
    import cv2

    stream = ScanNetStream(seq_name=scene)
    rgb_cache: dict[int, np.ndarray] = {}
    out: dict[int, dict[str, Any]] = {}
    for node in nodes:
        frame = int(node.frame_id)
        if frame not in rgb_cache:
            rgb_cache[frame] = stream.load_rgb(frame)
        rgb = rgb_cache[frame]
        mask = _node_mask(labels_by_frame, node)
        if rgb.shape[:2] != mask.shape:
            rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
        pix = rgb[mask]
        rgb_mean = [float(v) for v in np.mean(pix.astype(np.float32), axis=0).tolist()] if pix.size else None
        out[int(node.node_id)] = {"rgb_mean": rgb_mean}
    return out


def _component_mask_xy(
    labels_by_frame: dict[int, np.ndarray],
    frame_id: int,
    node_id: int,
    depth_shape: tuple[int, int],
) -> np.ndarray:
    mask = labels_by_frame[int(frame_id)] == int(node_id) + 1
    if mask.shape != depth_shape:
        import cv2

        mask = cv2.resize(mask.astype(np.uint8), (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack([xs.astype(np.int64), ys.astype(np.int64)], axis=1)


def _run_diagnostic_ap_bridge(args: argparse.Namespace, state: dict[str, Any], object_fields: list[Any]) -> dict[str, Any]:
    if not str(args.diagnostic_ap_output_config):
        return {
            "diagnostic_ap_bridge_requested": False,
            "diagnostic_ap_bridge_status": "not_requested",
        }
    from stream4d.export_scannet import ScanNetExporter
    from stream4d.scannet_stream import ScanNetStream
    from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest

    stream = ScanNetStream(seq_name=args.scene)
    errors = stream.validate(require_masks=False)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=args.diagnostic_ap_output_config,
        export_nn_radius=float(args.diagnostic_export_nn_radius),
        export_support_mode="mask_backproject",
        export_mask_sample_stride=int(args.diagnostic_export_mask_sample_stride),
        export_mask_max_pixels=int(args.diagnostic_export_mask_max_pixels),
        export_min_points_per_object=int(args.diagnostic_export_min_points_per_object),
        export_score_mode="area",
    )
    nodes_by_id = {int(node.node_id): node for node in state["nodes"]}
    depth_shape_by_frame: dict[int, tuple[int, int]] = {}
    object_records = []
    object_dict = {}
    mask_observation_count = 0
    backproject_pixel_count = 0
    backproject_hit_count = 0
    for field in object_fields:
        point_ids: set[int] = set()
        mask_list = []
        for node_id_raw in field.semantic_masklet_ids:
            node_id = int(node_id_raw)
            node = nodes_by_id.get(node_id)
            if node is None:
                continue
            frame_id = int(node.frame_id)
            if frame_id not in depth_shape_by_frame:
                depth_shape_by_frame[frame_id] = exporter.stream.load_depth(frame_id).shape
            xy = _component_mask_xy(state["labels_by_frame"], frame_id, node_id, depth_shape_by_frame[frame_id])
            if xy.size == 0:
                continue
            hit_ids, _dist = exporter._backproject_xy(frame_id, xy, nn_radius=float(args.diagnostic_export_nn_radius))
            backproject_pixel_count += int(xy.shape[0])
            backproject_hit_count += int(hit_ids.shape[0])
            point_ids.update(int(v) for v in hit_ids.tolist())
            mask_list.append((int(frame_id), int(node.mask_index), float(node.area)))
            mask_observation_count += 1
        sorted_points = sorted(point_ids)
        object_dict[int(field.object_id)] = {
            "point_ids": np.asarray(sorted_points, dtype=np.int64),
            "mask_list": mask_list,
            "repre_mask_list": mask_list[: min(len(mask_list), 8)],
            "score": float(len(sorted_points)),
            "area_score": float(len(sorted_points)),
            "source_variant": "v41_1_object_field_diagnostic_mask_backproject",
        }
        object_records.append(
            {
                "object_id": int(field.object_id),
                "point_ids": set(sorted_points),
                "score": float(len(sorted_points)),
                "area_score": float(len(sorted_points)),
            }
        )
    output_diag = exporter._write_outputs(
        object_records,
        object_dict,
        np.zeros((exporter.scene_points.shape[0],), dtype=np.uint16),
    )
    manifest = build_prediction_manifest(
        output_config=args.diagnostic_ap_output_config,
        is_method_result=False,
        is_diagnostic_only=True,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(Path(args.output_root) / "native_object_field_export_summary.json")],
        pre_points_policy="diagnostic_recompute",
        support_policy="object_field_mask_backproject",
        notes=(
            "Diagnostic-only v41.1 ObjectField AP bridge. This consumes semantic-material "
            "ObjectFields but uses ScanNet RGB-D/pose/mesh to materialize mesh-vertex "
            "prediction masks, so it must not enter method tables."
        ),
        extra={
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "uses_gt_for_prediction": False,
            "uses_gt_sim3_for_prediction": False,
            "forbidden_for_method_table": True,
            "geometry_source": "diagnostic_scannet_rgbd_pose_mesh_bridge",
            "alignment_source": "scannet_pose_depth_mesh_bridge",
            "eval_policy": "diagnostic_object_field_mask_backproject_only",
        },
    )
    write_prediction_manifest(args.diagnostic_ap_output_config, manifest, pred_suffix="class_agnostic")
    pred_path = Path("data/prediction") / f"{args.diagnostic_ap_output_config}_class_agnostic" / f"{args.scene}.npz"
    diag = {
        "diagnostic_ap_bridge_requested": True,
        "diagnostic_ap_bridge_status": "wrote_prediction_npz",
        "diagnostic_ap_output_config": args.diagnostic_ap_output_config,
        "diagnostic_prediction_path": str(pred_path),
        "diagnostic_is_method_result": False,
        "diagnostic_is_diagnostic_only": True,
        "diagnostic_forbidden_for_method_table": True,
        "diagnostic_uses_rgbd_for_prediction": True,
        "diagnostic_uses_pose_for_prediction": True,
        "diagnostic_uses_scannet_mesh_for_prediction": True,
        "diagnostic_mask_observation_count": int(mask_observation_count),
        "diagnostic_backproject_pixel_count": int(backproject_pixel_count),
        "diagnostic_backproject_hit_count": int(backproject_hit_count),
        "diagnostic_backproject_hit_rate": float(backproject_hit_count / max(backproject_pixel_count, 1)),
    }
    diag.update({f"diagnostic_{key}": value for key, value in output_diag.items()})
    return diag


def _component_counter(component: list[int], support_by_region: dict[int, Counter[int]]) -> Counter[int]:
    counter: Counter[int] = Counter()
    for node_id in component:
        counter.update(support_by_region.get(int(node_id), Counter()))
    return counter


def _boundary_refine_labels_no_gt(
    scene: str,
    nodes: list[RegionNode],
    labels_by_frame: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[RegionNode], dict[int, np.ndarray], dict[str, Any]]:
    if not bool(getattr(args, "enable_boundary_split", False)):
        return nodes, labels_by_frame, {
            "boundary_split_used": False,
            "boundary_split_variant": "",
            "boundary_parent_region_count": int(len(nodes)),
            "boundary_output_region_count": int(len(nodes)),
            "boundary_split_parent_count": 0,
            "boundary_added_region_count": 0,
            "boundary_dropped_pixel_count": 0,
        }

    from stream4d.scannet_stream import ScanNetStream
    import cv2

    stream = ScanNetStream(seq_name=scene)
    rgb_cache: dict[int, np.ndarray] = {}
    new_nodes: list[RegionNode] = []
    new_labels_by_frame: dict[int, np.ndarray] = {
        int(frame_id): np.zeros_like(label, dtype=np.int32) for frame_id, label in labels_by_frame.items()
    }
    split_parent_count = 0
    dropped_pixel_count = 0
    for node in nodes:
        frame_id = int(node.frame_id)
        label = labels_by_frame.get(frame_id)
        if label is None:
            continue
        mask = np.asarray(label == int(node.node_id) + 1, dtype=bool)
        parent_area = int(mask.sum())
        if parent_area <= 0:
            continue
        if frame_id not in rgb_cache:
            rgb_cache[frame_id] = stream.load_rgb(frame_id)
        rgb = rgb_cache[frame_id]
        if rgb.shape[:2] != mask.shape:
            rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
        parts = _boundary_split(mask, rgb, args, variant=str(args.boundary_variant))
        if len(parts) > 1:
            split_parent_count += 1
        assigned_area = 0
        for part_idx, part in enumerate(parts):
            part_mask = np.asarray(part, dtype=bool)
            area = int(part_mask.sum())
            if area < int(args.min_region_area):
                continue
            new_node = RegionNode(
                node_id=len(new_nodes),
                scene=str(node.scene),
                source=str(node.source),
                mode=str(node.mode),
                frame_id=frame_id,
                mask_index=int(node.mask_index) * 1000 + int(part_idx),
                area=area,
            )
            new_nodes.append(new_node)
            new_labels_by_frame[frame_id][part_mask] = int(new_node.node_id) + 1
            assigned_area += area
        dropped_pixel_count += max(0, parent_area - assigned_area)
    return new_nodes, new_labels_by_frame, {
        "boundary_split_used": True,
        "boundary_split_variant": str(args.boundary_variant),
        "boundary_parent_region_count": int(len(nodes)),
        "boundary_output_region_count": int(len(new_nodes)),
        "boundary_split_parent_count": int(split_parent_count),
        "boundary_added_region_count": int(max(0, len(new_nodes) - len(nodes))),
        "boundary_dropped_pixel_count": int(dropped_pixel_count),
    }


def _make_candidates_and_scores(
    components: list[list[int]],
    support_by_region: dict[int, Counter[int]],
    observation_count_by_tube: dict[int, int],
    *,
    max_candidates: int,
    max_tubes_per_candidate: int,
    include_forbidden_birth_probe: bool,
) -> tuple[list[ObjectFieldCandidate], list[TubeAttachmentScore], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    for component in components:
        support_counter = _component_counter(component, support_by_region)
        if not support_counter:
            continue
        support_mass = int(sum(support_counter.values()))
        raw_rows.append(
            {
                "component": [int(v) for v in component],
                "support_counter": support_counter,
                "score": float(support_mass + 0.01 * len(component) + 0.001 * len(support_counter)),
            }
        )
    raw_rows = sorted(raw_rows, key=lambda row: (-float(row["score"]), int(row["component"][0])))[: int(max_candidates)]

    candidates: list[ObjectFieldCandidate] = []
    scores: list[TubeAttachmentScore] = []
    candidate_rows: list[dict[str, Any]] = []
    for candidate_id, row in enumerate(raw_rows):
        counter: Counter[int] = row["support_counter"]
        top_tubes = [int(tube_id) for tube_id, _count in counter.most_common(int(max_tubes_per_candidate))]
        component = [int(v) for v in row["component"]]
        candidates.append(
            ObjectFieldCandidate(
                candidate_id=int(candidate_id),
                semantic_masklet_ids=tuple(component),
                material_tube_ids=tuple(top_tubes),
                score=float(row["score"]),
                birth_source="semantic_masklet",
            )
        )
        candidate_rows.append(
            {
                "candidate_id": int(candidate_id),
                "semantic_masklet_count": int(len(component)),
                "material_tube_count": int(len(top_tubes)),
                "support_observation_count": int(sum(counter.values())),
                "score": float(row["score"]),
            }
        )
        for tube_id, count in counter.items():
            obs = int(observation_count_by_tube.get(int(tube_id), 0))
            scores.append(
                TubeAttachmentScore(
                    tube_id=int(tube_id),
                    object_id=int(candidate_id),
                    score=float(_safe_div(float(count), float(obs))),
                )
            )
    if include_forbidden_birth_probe:
        candidates.append(
            ObjectFieldCandidate(
                candidate_id=int(len(candidates)),
                semantic_masklet_ids=(),
                material_tube_ids=tuple(),
                score=999999.0,
                birth_source="d4rt_tube",
            )
        )
    return candidates, scores, candidate_rows


def _build_no_gt_components(scene: str, args: argparse.Namespace) -> dict[str, Any]:
    setattr(_load_masks, "max_regions_per_scene", int(args.max_regions_per_scene))
    nodes, labels_by_frame, mask_manifest = _load_masks(
        Path(args.mask_root),
        scene,
        args.source,
        args.mode,
        int(args.min_region_area),
    )
    nodes, labels_by_frame, boundary_info = _boundary_refine_labels_no_gt(scene, nodes, labels_by_frame, args)
    mask_manifest = {**mask_manifest, **boundary_info, "region_count_after_boundary_refine": int(len(nodes))}
    frame_rank = _frame_rank_map(labels_by_frame)
    tubes = _load_tubes(scene, args)
    support_by_region, support_by_tube, observation_count_by_tube = _collect_observations(nodes, labels_by_frame, tubes, args)
    support_sets = {idx: set(counter.keys()) for idx, counter in support_by_region.items()}
    pair_counts = _support_pair_counts(
        nodes,
        support_by_region,
        max_pairs_per_tube=int(args.max_support_pairs_per_tube),
        seed=int(args.seed),
        frame_rank=frame_rank,
    )
    short_edges = _filtered_edges(
        nodes,
        support_sets,
        pair_counts,
        frame_rank,
        max_delta=int(args.short_max_delta),
        min_shared=int(args.short_min_shared),
        min_jaccard=float(args.short_min_jaccard),
    )
    closure_edges: list[tuple[float, int, float, int, int, int]] = []
    if not bool(args.disable_closure):
        closure_edges = _filtered_edges(
            nodes,
            support_sets,
            pair_counts,
            frame_rank,
            min_delta=int(args.closure_min_delta),
            max_delta=None,
            min_shared=int(args.closure_min_shared),
            min_jaccard=float(args.closure_min_jaccard),
        )
    diagnostics = None
    rejected_rgb = 0
    if bool(args.compute_rgb_filter):
        diagnostics = _rgb_diagnostics_no_gt(scene, nodes, labels_by_frame)
        closure_edges, rejected_rgb = _filter_edges_by_rgb(
            closure_edges,
            diagnostics,
            min_rgb_similarity=float(args.rgb_min_similarity),
        )
    components, edge_info = _components_chain_then_closure(nodes, short_edges, closure_edges, frame_rank=frame_rank)
    split_info = {"rgb_split_components": 0, "rgb_split_new_components": 0}
    if bool(args.compute_rgb_filter):
        components, split_info = _split_components_by_rgb(
            nodes,
            components,
            diagnostics,
            min_rgb_similarity=float(args.rgb_split_min_similarity),
        )
        edge_info.update(split_info)
        edge_info["rejected_rgb"] = int(rejected_rgb)
        if bool(args.rgb_isolate_outliers):
            components, isolate_info = _isolate_rgb_outlier_nodes(
                nodes,
                components,
                diagnostics,
                min_center_similarity=float(args.rgb_outlier_min_center_similarity),
                max_component_nodes=int(args.rgb_outlier_max_component_nodes),
            )
            edge_info.update(isolate_info)
        if bool(args.rgb_drop_incoherent):
            components, drop_info = _drop_rgb_incoherent_components(
                nodes,
                components,
                diagnostics,
                min_pairwise_similarity=float(args.rgb_drop_min_pairwise_similarity),
                max_component_nodes=int(args.rgb_drop_max_component_nodes),
            )
            edge_info.update(drop_info)
    return {
        "nodes": nodes,
        "labels_by_frame": labels_by_frame,
        "mask_manifest": mask_manifest,
        "tubes": tubes,
        "support_by_region": support_by_region,
        "support_by_tube": support_by_tube,
        "observation_count_by_tube": observation_count_by_tube,
        "components": components,
        "edge_info": edge_info,
        "pair_count": int(len(pair_counts)),
        "short_edge_count": int(len(short_edges)),
        "closure_edge_count": int(len(closure_edges)),
        "rgb_filter_used": bool(args.compute_rgb_filter),
        "rgb_rejected_edge_count": int(rejected_rgb),
        "rgb_split_components": int(split_info["rgb_split_components"]),
        "rgb_split_new_components": int(split_info["rgb_split_new_components"]),
        "rgb_unknown_components": int(edge_info.get("rgb_unknown_components", 0)),
        "rgb_unknown_nodes": int(edge_info.get("rgb_unknown_nodes", 0)),
        "rgb_outlier_components": int(edge_info.get("rgb_outlier_components", 0)),
        "rgb_outlier_nodes": int(edge_info.get("rgb_outlier_nodes", 0)),
        **boundary_info,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = _build_no_gt_components(args.scene, args)
    candidates, attachment_scores, candidate_rows = _make_candidates_and_scores(
        state["components"],
        state["support_by_region"],
        state["observation_count_by_tube"],
        max_candidates=int(args.max_candidates),
        max_tubes_per_candidate=int(args.max_tubes_per_candidate),
        include_forbidden_birth_probe=bool(args.include_forbidden_birth_probe),
    )
    inference = run_semantic_material_inference(
        candidates,
        attachment_scores,
        config=SemanticMaterialInferenceConfig(
            attach_threshold=float(args.attach_threshold),
            attach_margin=float(args.attach_margin),
            max_fields=int(args.max_fields),
            duplicate_support_jaccard=float(args.duplicate_support_jaccard),
            duplicate_material_jaccard=float(args.duplicate_material_jaccard),
            adaptive_attach_threshold=float(args.adaptive_attach_threshold),
            adaptive_attach_score_quantile=float(args.adaptive_attach_score_quantile),
            adaptive_attach_quantile_min=float(args.adaptive_attach_quantile_min),
        ),
        diagnostic_metrics={"AP_bridge": None},
    )
    export = export_object_fields_to_native_points(
        inference.object_fields,
        state["tubes"],
        config=NativeObjectFieldExportConfig(
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            require_semantic_birth=True,
            require_canonical=True,
            require_method_safe_alignment=True,
        ),
    )
    diagnostic_ap = _run_diagnostic_ap_bridge(args, state, inference.object_fields)
    method_safe_tube_count = sum(
        1
        for tube in state["tubes"]
        if str(tube.coordinate_frame) == "d4rt_canonical"
        and str(tube.alignment_source) in {"same_chunk_identity", "d4rt_self_sim3"}
        and bool(tube.allow_metric_merge)
        and bool(dict(tube.alignment_quality or {}).get("pass_gate", False))
    )
    summary = {
        **export.summary,
        "scene": args.scene,
        "status": (
            "PARTIAL_NATIVE_SUPPORT_EXPORT_PASS_AP_NOT_EVALUATED"
            if bool(export.summary.get("native_export_smoke_pass"))
            else "NO_GO_NATIVE_SUPPORT_EXPORT_EMPTY_AP_NOT_EVALUATED"
        ),
        "no_gt_component_builder": True,
        "rgb_filter_used": bool(state["rgb_filter_used"]),
        "rgb_rejected_edge_count": int(state["rgb_rejected_edge_count"]),
        "rgb_split_components": int(state["rgb_split_components"]),
        "rgb_split_new_components": int(state["rgb_split_new_components"]),
        "rgb_unknown_components": int(state["rgb_unknown_components"]),
        "rgb_unknown_nodes": int(state["rgb_unknown_nodes"]),
        "rgb_outlier_components": int(state["rgb_outlier_components"]),
        "rgb_outlier_nodes": int(state["rgb_outlier_nodes"]),
        "gt_labels_loaded": False,
        "scanNet_ap_backprojection_used": False,
        "mask_manifest": state["mask_manifest"],
        "region_count": int(len(state["nodes"])),
        "tube_count": int(len(state["tubes"])),
        "method_safe_tube_count": int(method_safe_tube_count),
        "support_region_count": int(len(state["support_by_region"])),
        "support_tube_count": int(len(state["support_by_tube"])),
        "component_count": int(len(state["components"])),
        "pair_count": int(state["pair_count"]),
        "short_edge_count": int(state["short_edge_count"]),
        "closure_edge_count": int(state["closure_edge_count"]),
        "boundary_split_used": bool(state["boundary_split_used"]),
        "boundary_split_variant": str(state["boundary_split_variant"]),
        "boundary_parent_region_count": int(state["boundary_parent_region_count"]),
        "boundary_output_region_count": int(state["boundary_output_region_count"]),
        "boundary_split_parent_count": int(state["boundary_split_parent_count"]),
        "boundary_added_region_count": int(state["boundary_added_region_count"]),
        "boundary_dropped_pixel_count": int(state["boundary_dropped_pixel_count"]),
        "edge_info": state["edge_info"],
        "candidate_count": int(len(candidates)),
        "candidate_count_including_forbidden_birth_probe": int(len(candidates)),
        "attachment_score_count": int(len(attachment_scores)),
        "selected_object_field_count": int(len(inference.object_fields)),
        "inference_constraint_audit": inference.constraint_audit,
        "inference_metrics": inference.metrics,
        "diagnostic_ap_bridge": diagnostic_ap,
        "repair_result": "native_object_field_support_adapter_added",
        "remaining_blocker": "method_compatible_scannet_ap_bridge_not_implemented",
    }
    _write_json(out_dir / "native_object_field_export_summary.json", summary)
    _write_csv(out_dir / "native_object_field_candidates.csv", candidate_rows)
    _write_csv(out_dir / "native_object_field_rows.csv", export.object_rows)
    _write_csv(out_dir / "native_object_point_rows.csv", export.point_rows)
    _write_points_npz(out_dir / "native_object_points.npz", export.point_rows)
    _write_json(
        out_dir / "native_object_fields.json",
        [
            {
                "object_id": int(field.object_id),
                "primary_field_id": int(field.primary_field_id),
                "semantic_masklet_ids": [int(v) for v in field.semantic_masklet_ids],
                "attached_tube_ids": [int(v) for v in field.attached_tube_ids],
                "confidence": float(field.confidence),
                "birth_state": str(field.birth_state),
            }
            for field in inference.object_fields
        ],
    )
    answer = "\n".join(
        [
            "# v41.1 Native ObjectField Export Smoke",
            "",
            f"status: `{summary['status']}`",
            f"scene: `{args.scene}`",
            f"native_export_smoke_pass: `{summary['native_export_smoke_pass']}`",
            f"selected_object_field_count: `{summary['selected_object_field_count']}`",
            f"exported_object_count: `{summary['exported_object_count']}`",
            f"exported_tube_count: `{summary['exported_tube_count']}`",
            f"native_point_count: `{summary['native_point_count']}`",
            f"method_safe_tube_count: `{summary['method_safe_tube_count']}`",
            "",
            "This is a D4RT-native support export smoke, not a ScanNet AP result.",
            "AP_bridge_status: `not_evaluated_native_support_not_scannet_ap`",
            "remaining_blocker: `method_compatible_scannet_ap_bridge_not_implemented`",
        ]
    )
    (out_dir / "native_object_field_export_answer.md").write_text(answer + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="scene0050_00")
    parser.add_argument("--mask-root", default="outputs/audit/v37_dino_compact_filter_sources")
    parser.add_argument("--source", default="v37_dino_compact_filter")
    parser.add_argument("--mode", default="dino_k2_compact060_filter")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--output-root", default="outputs/audit/v41_1_native_object_field_export_smoke")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--max-regions-per-scene", type=int, default=0)
    parser.add_argument("--max-support-pairs-per-tube", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=41101)
    parser.add_argument("--short-max-delta", type=int, default=1)
    parser.add_argument("--short-min-shared", type=int, default=2)
    parser.add_argument("--short-min-jaccard", type=float, default=0.0)
    parser.add_argument("--disable-closure", action="store_true", default=True)
    parser.add_argument("--enable-closure", dest="disable_closure", action="store_false")
    parser.add_argument("--closure-min-delta", type=int, default=9)
    parser.add_argument("--closure-min-shared", type=int, default=2)
    parser.add_argument("--closure-min-jaccard", type=float, default=0.01)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--max-fields", type=int, default=64)
    parser.add_argument("--max-tubes-per-candidate", type=int, default=128)
    parser.add_argument("--attach-threshold", type=float, default=0.25)
    parser.add_argument("--attach-margin", type=float, default=0.05)
    parser.add_argument("--duplicate-support-jaccard", type=float, default=0.90)
    parser.add_argument("--duplicate-material-jaccard", type=float, default=1.01)
    parser.add_argument("--adaptive-attach-threshold", type=float, default=0.0)
    parser.add_argument("--adaptive-attach-score-quantile", type=float, default=0.25)
    parser.add_argument("--adaptive-attach-quantile-min", type=float, default=1.01)
    parser.add_argument("--include-forbidden-birth-probe", action="store_true", default=True)
    parser.add_argument("--no-forbidden-birth-probe", dest="include_forbidden_birth_probe", action="store_false")
    parser.add_argument("--compute-rgb-filter", action="store_true", default=False)
    parser.add_argument("--rgb-min-similarity", type=float, default=0.90)
    parser.add_argument("--rgb-split-min-similarity", type=float, default=0.90)
    parser.add_argument("--rgb-drop-incoherent", action="store_true", default=False)
    parser.add_argument("--rgb-drop-min-pairwise-similarity", type=float, default=0.85)
    parser.add_argument("--rgb-drop-max-component-nodes", type=int, default=0)
    parser.add_argument("--rgb-isolate-outliers", action="store_true", default=False)
    parser.add_argument("--rgb-outlier-min-center-similarity", type=float, default=0.85)
    parser.add_argument("--rgb-outlier-max-component-nodes", type=int, default=0)
    parser.add_argument("--enable-boundary-split", action="store_true", default=False)
    parser.add_argument("--boundary-variant", default="boundary_watershed_q85_split")
    parser.add_argument("--min-child-area", type=int, default=64)
    parser.add_argument("--boundary-min-split-area", type=int, default=1024)
    parser.add_argument("--boundary-gradient-quantile", type=float, default=0.90)
    parser.add_argument("--boundary-min-gradient", type=float, default=0.08)
    parser.add_argument("--boundary-edge-dilate", type=int, default=1)
    parser.add_argument("--boundary-min-child-fraction", type=float, default=0.05)
    parser.add_argument("--boundary-min-core-coverage", type=float, default=0.35)
    parser.add_argument("--boundary-max-child-count", type=int, default=6)
    parser.add_argument("--diagnostic-ap-output-config", default="")
    parser.add_argument("--diagnostic-export-nn-radius", type=float, default=0.05)
    parser.add_argument("--diagnostic-export-mask-sample-stride", type=int, default=2)
    parser.add_argument("--diagnostic-export-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--diagnostic-export-min-points-per-object", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
