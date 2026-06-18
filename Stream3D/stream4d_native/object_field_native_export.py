from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .object_field import ObjectField
from .object_tube_io import TubeRecord


ALLOWED_METHOD_ALIGNMENT_SOURCES = {"same_chunk_identity", "d4rt_self_sim3"}


@dataclass(frozen=True)
class NativeObjectFieldExportConfig:
    min_visibility: float = 0.50
    min_confidence: float = 0.50
    require_semantic_birth: bool = True
    require_canonical: bool = True
    require_method_safe_alignment: bool = True


@dataclass
class NativeObjectFieldExportResult:
    point_rows: list[dict[str, Any]]
    object_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _json_counter(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _field_invalid_reason(field: ObjectField, config: NativeObjectFieldExportConfig) -> str | None:
    if config.require_semantic_birth and not field.semantic_masklet_ids:
        return "missing_semantic_masklet_birth"
    return None


def _tube_reject_reason(tube: TubeRecord, config: NativeObjectFieldExportConfig) -> str | None:
    if str(tube.alignment_source) == "eval_gt_sim3":
        return "eval_aligned_geometry_forbidden"
    if config.require_canonical:
        if str(tube.coordinate_frame) != "d4rt_canonical":
            return "noncanonical_coordinate_frame"
        if tube.xyz_canonical is None:
            return "missing_xyz_canonical"
    if config.require_method_safe_alignment:
        if not bool(tube.allow_metric_merge):
            return "metric_merge_disabled_by_alignment"
        if str(tube.alignment_source) not in ALLOWED_METHOD_ALIGNMENT_SOURCES:
            return "unsupported_alignment_source"
        if not bool(dict(tube.alignment_quality or {}).get("pass_gate", False)):
            return "alignment_quality_gate_failed"
    return None


def _valid_point_mask(tube: TubeRecord, config: NativeObjectFieldExportConfig) -> np.ndarray:
    xyz = np.asarray(tube.xyz_canonical if tube.xyz_canonical is not None else tube.xyz_ref0, dtype=np.float32)
    uv = np.asarray(tube.uv, dtype=np.float32)
    visibility = np.asarray(tube.visibility, dtype=np.float32)
    confidence = np.asarray(tube.confidence, dtype=np.float32)
    finite_xyz = np.isfinite(xyz).all(axis=1)
    finite_uv = np.isfinite(uv).all(axis=1)
    in_image = (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
    visible = visibility >= float(config.min_visibility)
    confident = confidence >= float(config.min_confidence)
    return finite_xyz & finite_uv & in_image & visible & confident


def export_object_fields_to_native_points(
    object_fields: list[ObjectField],
    tubes: list[TubeRecord] | dict[int, TubeRecord],
    *,
    config: NativeObjectFieldExportConfig | None = None,
) -> NativeObjectFieldExportResult:
    """Export semantic-born ObjectFields to method-safe D4RT canonical support points.

    This adapter intentionally does not produce ScanNet AP masks. It only creates
    a native D4RT support artifact whose provenance can be audited before a
    separate evaluation bridge is implemented.
    """

    cfg = config or NativeObjectFieldExportConfig()
    tube_by_id = {int(tube_id): tube for tube_id, tube in tubes.items()} if isinstance(tubes, dict) else {int(t.tube_id): t for t in tubes}
    point_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    invalid_field_reasons: Counter[str] = Counter()
    rejected_tube_reasons: Counter[str] = Counter()
    missing_tube_count = 0
    low_quality_point_count = 0
    exported_tube_ids: set[int] = set()
    alignment_sources_used: Counter[str] = Counter()
    coordinate_frames_used: Counter[str] = Counter()
    points_by_object: Counter[int] = Counter()
    tubes_by_object: defaultdict[int, set[int]] = defaultdict(set)

    for field in object_fields:
        field_reason = _field_invalid_reason(field, cfg)
        if field_reason is not None:
            invalid_field_reasons[field_reason] += 1
            continue
        field_point_count = 0
        field_tube_count = 0
        field_rejected_tubes: Counter[str] = Counter()
        for tube_id_raw in field.attached_tube_ids:
            tube_id = int(tube_id_raw)
            tube = tube_by_id.get(tube_id)
            if tube is None:
                missing_tube_count += 1
                field_rejected_tubes["missing_tube_record"] += 1
                continue
            reason = _tube_reject_reason(tube, cfg)
            if reason is not None:
                rejected_tube_reasons[reason] += 1
                field_rejected_tubes[reason] += 1
                continue
            valid = _valid_point_mask(tube, cfg)
            xyz = np.asarray(tube.xyz_canonical, dtype=np.float32)
            uv = np.asarray(tube.uv, dtype=np.float32)
            frames = np.asarray(tube.target_frames_global, dtype=np.int64)
            visibility = np.asarray(tube.visibility, dtype=np.float32)
            confidence = np.asarray(tube.confidence, dtype=np.float32)
            rejected_points = int(valid.size - int(valid.sum()))
            low_quality_point_count += rejected_points
            if not bool(valid.any()):
                field_rejected_tubes["no_visible_confident_points"] += 1
                continue
            field_tube_count += 1
            exported_tube_ids.add(tube_id)
            alignment_sources_used[str(tube.alignment_source)] += 1
            coordinate_frames_used[str(tube.coordinate_frame)] += 1
            tubes_by_object[int(field.object_id)].add(tube_id)
            for local_idx in np.flatnonzero(valid).tolist():
                row = {
                    "object_id": int(field.object_id),
                    "primary_field_id": int(field.primary_field_id),
                    "semantic_masklet_count": int(len(field.semantic_masklet_ids)),
                    "tube_id": tube_id,
                    "frame_id": int(frames[int(local_idx)]),
                    "local_point_index": int(local_idx),
                    "x": float(xyz[int(local_idx), 0]),
                    "y": float(xyz[int(local_idx), 1]),
                    "z": float(xyz[int(local_idx), 2]),
                    "u": float(uv[int(local_idx), 0]),
                    "v": float(uv[int(local_idx), 1]),
                    "visibility": float(visibility[int(local_idx)]),
                    "confidence": float(confidence[int(local_idx)]),
                }
                point_rows.append(row)
                field_point_count += 1
                points_by_object[int(field.object_id)] += 1
        object_rows.append(
            {
                "object_id": int(field.object_id),
                "primary_field_id": int(field.primary_field_id),
                "semantic_masklet_count": int(len(field.semantic_masklet_ids)),
                "input_attached_tube_count": int(len(field.attached_tube_ids)),
                "exported_tube_count": int(field_tube_count),
                "exported_point_count": int(field_point_count),
                "confidence": float(field.confidence),
                "rejected_tube_count_by_reason": _json_counter(field_rejected_tubes),
            }
        )

    forbidden_usage = {
        "uses_gt_for_prediction": False,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
    }
    exported_objects = int(sum(1 for row in object_rows if int(row["exported_point_count"]) > 0))
    summary = {
        "status": "PASS_NATIVE_D4RT_OBJECT_FIELD_SUPPORT_EXPORT" if exported_objects > 0 and point_rows else "NO_GO_NATIVE_D4RT_OBJECT_FIELD_SUPPORT_EXPORT_EMPTY",
        "native_export_smoke_pass": bool(exported_objects > 0 and point_rows),
        "artifact_kind": "d4rt_native_object_field_support_points",
        "is_ap_result": False,
        "is_method_ap_result": False,
        "AP_bridge_status": "not_evaluated_native_support_not_scannet_ap",
        "real_method_ap_status": "not_run",
        "input_object_field_count": int(len(object_fields)),
        "invalid_field_count": int(sum(invalid_field_reasons.values())),
        "invalid_field_count_by_reason": _json_counter(invalid_field_reasons),
        "exported_object_count": exported_objects,
        "exported_tube_count": int(len(exported_tube_ids)),
        "native_point_count": int(len(point_rows)),
        "missing_tube_count": int(missing_tube_count),
        "rejected_tube_count": int(sum(rejected_tube_reasons.values())),
        "rejected_tube_count_by_reason": _json_counter(rejected_tube_reasons),
        "low_quality_or_offscreen_point_count": int(low_quality_point_count),
        "points_by_object": {str(k): int(v) for k, v in sorted(points_by_object.items())},
        "tubes_by_object_count": {str(k): int(len(v)) for k, v in sorted(tubes_by_object.items())},
        "alignment_sources_used": _json_counter(alignment_sources_used),
        "coordinate_frames_used": _json_counter(coordinate_frames_used),
        "allowed_method_alignment_sources": sorted(ALLOWED_METHOD_ALIGNMENT_SOURCES),
        "require_semantic_birth": bool(cfg.require_semantic_birth),
        "require_canonical": bool(cfg.require_canonical),
        "require_method_safe_alignment": bool(cfg.require_method_safe_alignment),
        "min_visibility": float(cfg.min_visibility),
        "min_confidence": float(cfg.min_confidence),
        **forbidden_usage,
    }
    return NativeObjectFieldExportResult(point_rows=point_rows, object_rows=object_rows, summary=summary)

