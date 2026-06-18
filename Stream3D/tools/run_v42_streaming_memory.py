from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.object_field import ObjectField
from stream4d_native.object_field_memory_v42 import (
    V42MemoryExpansionConfig,
    expand_v42_object_fields_with_token_support,
)
from stream4d_native.object_field_native_export import (
    NativeObjectFieldExportConfig,
    export_object_fields_to_native_points,
)
from stream4d_native.semantic_material_part_graph import build_token_material_support
from tools.diagnose_v42_material_query_reason import (
    _build_fast_material_measurements,
    _label_maps_from_masks,
    _load_tokens,
    _source_masks,
)
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v36_external_downstream_assignment import _load_gt, _load_tubes
from tools.run_v42_native_support_bridge import _labels_from_tube_assignments, _mean, _offset_labels, load_v42_object_fields
from tools.run_v42_part_gated_alignment import _load_role_rows
from tools.run_v42_semantic_part_audit import _load_d4rt_records
from tools.run_v42_tube_role_real import _audit_material_cache_stride


ROOT = Path(__file__).resolve().parents[1]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key, "")) for key in keys})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_frame_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _object_field_rows(
    fields: list[ObjectField],
    *,
    scene: str,
    variant: str,
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in fields:
        rows.append(
            {
                "scene": scene,
                "variant": variant,
                "source": source,
                "object_id": int(field.object_id),
                "primary_field_id": int(field.primary_field_id),
                "semantic_masklet_ids": [int(v) for v in field.semantic_masklet_ids],
                "attached_tube_ids": [int(v) for v in field.attached_tube_ids],
                "semantic_masklet_count": int(len(field.semantic_masklet_ids)),
                "attached_tube_count": int(len(field.attached_tube_ids)),
                "confidence": float(field.confidence),
                "birth_source": "semantic_part_graph",
                "memory_expanded": True,
            }
        )
    return rows


def _namespace_for_native(args: argparse.Namespace, cache_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        cache_root=str(cache_root),
        max_tubes_per_window=int(args.max_tubes_per_window),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
    )


def _scene_metric_proxy_pass(row: dict[str, Any]) -> bool:
    if row.get("tube_4D_ARI") is None or row.get("tube_purity") is None or row.get("tube_completeness") is None:
        return False
    return bool(float(row["tube_purity"]) >= 0.86 and float(row["tube_completeness"]) >= 0.52)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v42 stride-1 streaming memory support expansion.")
    parser.add_argument("--object-field-root", required=True)
    parser.add_argument("--part-graph-root", required=True)
    parser.add_argument("--material-cache-root", required=True)
    parser.add_argument("--role-root", required=True)
    parser.add_argument("--external-source-root", default="outputs/audit/v42_source_audit_external_stride1_smoke")
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--variant", default="Q5")
    parser.add_argument("--sources", default="dinov2_maskcut")
    parser.add_argument("--frame-ids", default="0,10,20,30")
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--max-tubes-per-window", type=int, default=1024)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--material-image-width", type=int, default=1296)
    parser.add_argument("--material-image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.50)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--backfill-overlap-iou", type=float, default=0.10)
    parser.add_argument("--backfill-max-masks-per-frame", type=int, default=8)
    parser.add_argument("--material-backfill-min-tubes", type=int, default=1)
    parser.add_argument("--material-backfill-max-candidate-area-fraction", type=float, default=1.0)
    parser.add_argument("--require-material-frame-stride", type=int, default=1)
    parser.add_argument("--min-inside-count", type=int, default=1)
    parser.add_argument("--max-outside-visible-count", type=int, default=-1)
    parser.add_argument("--include-unknown-role", action="store_true")
    parser.add_argument("--disable-scene-role", action="store_true")
    parser.add_argument("--scene-min-weight", type=float, default=0.35)
    parser.add_argument("--scene-max-unknown-weight", type=float, default=0.65)
    parser.add_argument("--scene-max-residual", type=float, default=0.008)
    parser.add_argument("--outside-count-penalty", type=float, default=0.01)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    sources = [item.strip() for item in str(args.sources).split(",") if item.strip()]
    frame_ids = _parse_frame_ids(str(args.frame_ids))
    variant = str(args.variant)
    object_field_root = ROOT / str(args.object_field_root)
    part_graph_root = ROOT / str(args.part_graph_root)
    material_cache_root = ROOT / str(args.material_cache_root)
    role_root = ROOT / str(args.role_root)
    external_root = ROOT / str(args.external_source_root)
    native_cache_root = ROOT / str(args.cache_root) if str(args.cache_root).strip() else material_cache_root / variant
    output_root = ROOT / str(args.output_root)
    memory_config = V42MemoryExpansionConfig(
        min_inside_count=int(args.min_inside_count),
        max_outside_visible_count=None
        if int(args.max_outside_visible_count) < 0
        else int(args.max_outside_visible_count),
        include_unknown_role=bool(args.include_unknown_role),
        include_scene_role=not bool(args.disable_scene_role),
        scene_min_weight=float(args.scene_min_weight),
        scene_max_unknown_weight=float(args.scene_max_unknown_weight),
        scene_max_residual=float(args.scene_max_residual),
        outside_count_penalty=float(args.outside_count_penalty),
    )

    expansion_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    native_object_rows: list[dict[str, Any]] = []
    native_point_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    aggregate_pred: dict[int, int] = {}
    aggregate_gt: dict[int, int] = {}

    for scene_index, scene in enumerate(scenes):
        stride_diag = _audit_material_cache_stride(
            cache_root=material_cache_root / variant,
            scene=scene,
            required_stride=int(args.require_material_frame_stride),
        )
        records, d4rt_diag = _load_d4rt_records(
            cache_root=material_cache_root / variant,
            scene=scene,
            max_tubes_per_window=int(args.max_tubes_per_window),
            image_width=int(args.material_image_width),
            image_height=int(args.material_image_height),
        )
        d4rt_diag["material_cache_stride_audit"] = stride_diag
        stream = ScanNetStream(seq_name=scene)
        native_args = _namespace_for_native(args, native_cache_root)
        tubes = _load_tubes(scene, native_args)
        gt_labels = _load_gt(scene, tubes, native_args)
        for source in sources:
            fields = load_v42_object_fields(
                object_field_root,
                scene=scene,
                variant=variant,
                source=source,
            )
            tokens = _load_tokens(part_graph_root, variant, scene, source)
            role_by_tube = _load_role_rows(role_root, scene=scene, variant=variant, source=source)
            masks_by_frame, mask_diag = _source_masks(
                source=source,
                stream=stream,
                scene=scene,
                frame_ids=frame_ids,
                min_area=int(args.min_area),
                sample_frames=int(args.sample_frames),
                external_root=external_root,
                d4rt_records=records,
                backfill_overlap_iou=float(args.backfill_overlap_iou),
                backfill_max_masks_per_frame=int(args.backfill_max_masks_per_frame),
                material_backfill_min_tubes=int(args.material_backfill_min_tubes),
                material_backfill_max_candidate_area_fraction=float(
                    args.material_backfill_max_candidate_area_fraction
                ),
                material_min_visibility=float(args.min_visibility),
                material_min_confidence=float(args.min_confidence),
            )
            label_maps = _label_maps_from_masks(masks_by_frame)
            measurements, measurement_diag = _build_fast_material_measurements(
                records,
                masks_by_frame=label_maps,
                min_visibility=float(args.min_visibility),
                min_confidence=float(args.min_confidence),
            )
            support_by_token = build_token_material_support(tokens, measurements)
            expanded = expand_v42_object_fields_with_token_support(
                fields,
                support_by_token,
                role_by_tube,
                config=memory_config,
            )
            current_object_rows = _object_field_rows(
                expanded.object_fields,
                scene=scene,
                variant=variant,
                source=source,
            )
            object_rows.extend(current_object_rows)
            for row in expanded.expansion_rows:
                expansion_rows.append({"scene": scene, "variant": variant, "source": source, **row})
            native = export_object_fields_to_native_points(
                expanded.object_fields,
                tubes,
                config=NativeObjectFieldExportConfig(
                    min_visibility=float(args.min_visibility),
                    min_confidence=float(args.min_confidence),
                    require_semantic_birth=True,
                    require_canonical=True,
                    require_method_safe_alignment=True,
                ),
            )
            labels_pred, unknown_ratio, label_info = _labels_from_tube_assignments(
                expanded.object_fields,
                gt_labels,
                unknown_label_base=(scene_index + 1) * 1_000_000,
            )
            gt_labeled = {
                int(tube_id): int(gt)
                for tube_id, gt in gt_labels.items()
                if int(gt) > 0 and int(tube_id) in labels_pred
            }
            metrics = _cluster_metrics(labels_pred, gt_labeled) if labels_pred else {
                "ari": None,
                "purity": None,
                "completeness": None,
                "overmerge": None,
                "oversplit": None,
                "labeled_tube_count": 0,
            }
            global_pred, global_gt = _offset_labels(scene_index, labels_pred, gt_labeled)
            aggregate_pred.update(global_pred)
            aggregate_gt.update(global_gt)
            for row in native.object_rows:
                native_object_rows.append({"scene": scene, "variant": variant, "source": source, **row})
            for row in native.point_rows:
                native_point_rows.append({"scene": scene, "variant": variant, "source": source, **row})
            scene_row = {
                "scene": scene,
                "variant": variant,
                "source": source,
                "input_object_field_count": int(len(fields)),
                "expanded_object_field_count": int(len(expanded.object_fields)),
                "token_count": int(len(tokens)),
                "role_row_count": int(len(role_by_tube)),
                "measurement_count": int(len(measurements)),
                **{f"memory_{key}": value for key, value in expanded.summary.items() if key != "config"},
                "cache_tube_count": int(len(tubes)),
                "native_export_smoke_pass": bool(native.summary["native_export_smoke_pass"]),
                "exported_object_count": int(native.summary["exported_object_count"]),
                "exported_tube_count": int(native.summary["exported_tube_count"]),
                "native_point_count": int(native.summary["native_point_count"]),
                "missing_tube_count": int(native.summary["missing_tube_count"]),
                "rejected_tube_count": int(native.summary["rejected_tube_count"]),
                "tube_4D_ARI": metrics.get("ari"),
                "tube_purity": metrics.get("purity"),
                "tube_completeness": metrics.get("completeness"),
                "tube_overmerge": metrics.get("overmerge"),
                "tube_oversplit": metrics.get("oversplit"),
                "unknown_labeled_tube_ratio": float(unknown_ratio),
                **label_info,
                "uses_gt_for_prediction": False,
                "uses_gt_for_scoring": True,
                "is_method_ap_result": False,
                "AP_bridge_status": native.summary["AP_bridge_status"],
            }
            scene_row["native_support_metric_proxy_pass"] = _scene_metric_proxy_pass(scene_row)
            scene_rows.append(scene_row)
            manifests.append(
                {
                    "scene": scene,
                    "variant": variant,
                    "source": source,
                    "object_field_root": str(object_field_root),
                    "part_graph_root": str(part_graph_root),
                    "material_cache_root": str(material_cache_root),
                    "native_cache_root": str(native_cache_root),
                    "role_root": str(role_root),
                    "external_source_root": str(external_root),
                    "frame_ids": frame_ids,
                    "d4rt_diag": d4rt_diag,
                    "mask_diag": mask_diag,
                    "measurement_diag": measurement_diag,
                    "memory_summary": expanded.summary,
                    "native_summary": native.summary,
                }
            )

    aggregate_metrics = _cluster_metrics(aggregate_pred, aggregate_gt) if aggregate_pred else {
        "ari": None,
        "purity": None,
        "completeness": None,
        "overmerge": None,
        "oversplit": None,
        "labeled_tube_count": 0,
    }
    summary = {
        "phase": "v42_streaming_memory_stride1_support",
        "object_field_root": str(object_field_root),
        "part_graph_root": str(part_graph_root),
        "material_cache_root": str(material_cache_root),
        "native_cache_root": str(native_cache_root),
        "role_root": str(role_root),
        "external_source_root": str(external_root),
        "scene_count": int(len(scene_rows)),
        "input_object_field_count": int(sum(int(row["input_object_field_count"]) for row in scene_rows)),
        "expanded_object_field_count": int(sum(int(row["expanded_object_field_count"]) for row in scene_rows)),
        "added_tube_count": int(sum(int(row["memory_added_tube_count"]) for row in scene_rows)),
        "expanded_attached_tube_count": int(sum(int(row["memory_expanded_attached_tube_count"]) for row in scene_rows)),
        "exported_object_count": int(sum(int(row["exported_object_count"]) for row in scene_rows)),
        "exported_tube_count": int(sum(int(row["exported_tube_count"]) for row in scene_rows)),
        "native_point_count": int(sum(int(row["native_point_count"]) for row in scene_rows)),
        "native_export_smoke_pass": bool(scene_rows and all(bool(row["native_export_smoke_pass"]) for row in scene_rows)),
        "native_support_metric_proxy_pass": bool(scene_rows and all(bool(row["native_support_metric_proxy_pass"]) for row in scene_rows)),
        "aggregate_tube_4D_ARI": aggregate_metrics.get("ari"),
        "aggregate_tube_purity": aggregate_metrics.get("purity"),
        "aggregate_tube_completeness": aggregate_metrics.get("completeness"),
        "aggregate_tube_overmerge": aggregate_metrics.get("overmerge"),
        "aggregate_tube_oversplit": aggregate_metrics.get("oversplit"),
        "mean_unknown_labeled_tube_ratio": _mean([row["unknown_labeled_tube_ratio"] for row in scene_rows]),
        "memory_config": memory_config.__dict__,
        "material_cache_stride_required": int(args.require_material_frame_stride),
        "uses_gt_for_prediction": False,
        "uses_gt_for_scoring": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "is_method_ap_result": False,
        "AP_bridge_status": "not_evaluated_native_support_not_scannet_ap",
        "phase8_gate_pass": False,
        "phase8_gate_blocker": "streaming memory native support is not ScanNet AP/native method AP; AP materialization remains unrun",
        "manifests": manifests,
    }
    _write_csv(output_root / "memory_object_field_rows.csv", object_rows)
    _write_csv(output_root / "memory_expansion_rows.csv", expansion_rows)
    _write_csv(output_root / "memory_scene_rows.csv", scene_rows)
    _write_csv(output_root / "memory_native_object_rows.csv", native_object_rows)
    _write_csv(output_root / "memory_native_point_rows.csv", native_point_rows)
    _write_json(output_root / "memory_summary.json", summary)
    print(
        json.dumps(
            _json_safe(
                {
                    "output_root": str(output_root),
                    "scene_rows": len(scene_rows),
                    "added_tube_count": summary["added_tube_count"],
                    "aggregate_tube_4D_ARI": summary["aggregate_tube_4D_ARI"],
                    "aggregate_tube_purity": summary["aggregate_tube_purity"],
                    "aggregate_tube_completeness": summary["aggregate_tube_completeness"],
                    "native_support_metric_proxy_pass": summary["native_support_metric_proxy_pass"],
                    "phase8_gate_pass": summary["phase8_gate_pass"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
