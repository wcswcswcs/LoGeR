from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.material_tube_roles import MaterialTubeEvidence, infer_tube_roles, summarize_tube_roles
from stream4d_native.object_aware_self_stitch import evaluate_role_aware_stitch_variants
from tools.diagnose_v42_material_query_reason import (
    _label_maps_from_masks,
    _load_d4rt_records,
    _load_query_rows,
    _source_masks,
)


ROOT = Path(__file__).resolve().parents[1]


STATIC_REASON_PRIOR = {
    "fixed_grid": 0.50,
    "exploration": 0.90,
    "exploration_fill": 0.90,
    "overlap_anchor": 0.70,
    "mask_boundary": 0.35,
    "mask_interior": 0.25,
    "disagreement_proxy": 0.25,
    "unmatched": 0.35,
}

OBJECT_REASON_PRIOR = {
    "fixed_grid": 0.35,
    "exploration": 0.10,
    "exploration_fill": 0.10,
    "overlap_anchor": 0.55,
    "mask_boundary": 0.80,
    "mask_interior": 0.90,
    "disagreement_proxy": 0.75,
    "unmatched": 0.35,
}


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


def _audit_material_cache_stride(
    *,
    cache_root: Path,
    scene: str,
    required_stride: int,
) -> dict[str, Any]:
    scene_root = Path(cache_root) / str(scene)
    manifests = sorted(scene_root.glob("carriers_window*_manifest.json"))
    window_rows: list[dict[str, Any]] = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame_ids = [int(v) for v in payload.get("frame_ids", payload.get("raw_frame_ids", []))]
        deltas = [int(b - a) for a, b in zip(frame_ids[:-1], frame_ids[1:])]
        unique_deltas = sorted(set(deltas))
        pass_required = bool(deltas) and unique_deltas == [int(required_stride)]
        window_rows.append(
            {
                "manifest": str(path),
                "window_index": int(payload.get("window_index", len(window_rows))),
                "num_frames": int(len(frame_ids)),
                "first_frame_id": int(frame_ids[0]) if frame_ids else None,
                "last_frame_id": int(frame_ids[-1]) if frame_ids else None,
                "frame_delta_unique": unique_deltas,
                "required_stride": int(required_stride),
                "uses_required_stride": bool(pass_required),
            }
        )
    all_pass = bool(window_rows) and all(bool(row["uses_required_stride"]) for row in window_rows)
    diag = {
        "cache_root": str(cache_root),
        "scene": str(scene),
        "required_stride": int(required_stride),
        "manifest_count": int(len(window_rows)),
        "uses_required_stride": bool(all_pass),
        "windows": window_rows,
    }
    if int(required_stride) > 0 and not all_pass:
        raise RuntimeError(
            "material cache stride audit failed: "
            f"scene={scene} cache_root={cache_root} required_stride={required_stride} "
            f"diag={json.dumps(_json_safe(diag), sort_keys=True)}"
        )
    return diag


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q)))


def _visible_valid_stats(record: Any, *, min_visibility: float, min_confidence: float) -> dict[str, Any]:
    visibility = np.asarray(record.get_geometry_for_measurement(field="visibility"), dtype=np.float32).reshape(-1)
    confidence = np.asarray(record.get_geometry_for_measurement(field="confidence"), dtype=np.float32).reshape(-1)
    uv = np.asarray(record.get_geometry_for_measurement(field="uv"), dtype=np.float32)
    in_bounds = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= 1.0)
    )
    accepted = in_bounds & (visibility >= float(min_visibility)) & (confidence >= float(min_confidence))
    accepted_uv = uv[accepted]
    if accepted_uv.shape[0] >= 2:
        motion = float(np.mean(np.linalg.norm(np.diff(accepted_uv, axis=0), axis=1)))
    else:
        motion = 0.0
    accepted_ratio = float(np.mean(accepted)) if accepted.size else 0.0
    quality = 0.40 * float(np.mean(visibility)) + 0.40 * float(np.mean(confidence)) + 0.20 * accepted_ratio
    residual_proxy = _clip01((1.0 - quality) * 0.15)
    return {
        "visibility_mean": float(np.mean(visibility)) if visibility.size else 0.0,
        "confidence_mean": float(np.mean(confidence)) if confidence.size else 0.0,
        "accepted_track_ratio": accepted_ratio,
        "accepted_track_length": int(np.count_nonzero(accepted)),
        "uv_in_bounds_rate": float(np.mean(in_bounds)) if in_bounds.size else 0.0,
        "uv_motion_magnitude": motion,
        "self_stitch_residual_proxy": residual_proxy,
        "scale_proxy": float(1.0 + 0.20 * _clip01(motion / 0.20)),
    }


def _mask_projection_stats(
    record: Any,
    *,
    label_maps: dict[int, np.ndarray],
    min_visibility: float,
    min_confidence: float,
) -> dict[str, Any]:
    frames = np.asarray(record.target_frames_global, dtype=np.int64).reshape(-1)
    uv_all = np.asarray(record.get_geometry_for_measurement(field="uv"), dtype=np.float32)
    visibility = np.asarray(record.get_geometry_for_measurement(field="visibility"), dtype=np.float32).reshape(-1)
    confidence = np.asarray(record.get_geometry_for_measurement(field="confidence"), dtype=np.float32).reshape(-1)
    labels: list[int] = []
    for local_idx, frame_value in enumerate(frames.tolist()):
        frame_id = int(frame_value)
        mask = label_maps.get(frame_id)
        if mask is None:
            continue
        if visibility[local_idx] < float(min_visibility) or confidence[local_idx] < float(min_confidence):
            continue
        uv = uv_all[local_idx]
        if not (np.isfinite(uv).all() and 0.0 <= float(uv[0]) <= 1.0 and 0.0 <= float(uv[1]) <= 1.0):
            continue
        height, width = mask.shape[:2]
        x = int(np.clip(np.rint(float(uv[0]) * (width - 1)), 0, width - 1))
        y = int(np.clip(np.rint(float(uv[1]) * (height - 1)), 0, height - 1))
        labels.append(int(mask[y, x]))
    if not labels:
        return {
            "mask_projection_count": 0,
            "inside_mask_ratio": 0.0,
            "background_projection_ratio": 0.0,
            "dominant_mask_ratio": 0.0,
            "dominant_mask_id": 0,
            "object_masklet_consistency_raw": 0.0,
        }
    label_values, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    count_by_label = {int(label): int(count) for label, count in zip(label_values.tolist(), counts.tolist())}
    foreground_counts = {label: count for label, count in count_by_label.items() if int(label) > 0}
    inside = int(sum(foreground_counts.values()))
    background = int(count_by_label.get(0, 0))
    if foreground_counts:
        dominant_mask_id, dominant_count = max(foreground_counts.items(), key=lambda item: item[1])
    else:
        dominant_mask_id, dominant_count = 0, background
    inside_ratio = float(inside / max(len(labels), 1))
    background_ratio = float(background / max(len(labels), 1))
    dominant_ratio = float(dominant_count / max(len(labels), 1))
    object_consistency_raw = float(inside_ratio * dominant_ratio)
    return {
        "mask_projection_count": int(len(labels)),
        "inside_mask_ratio": inside_ratio,
        "background_projection_ratio": background_ratio,
        "dominant_mask_ratio": dominant_ratio,
        "dominant_mask_id": int(dominant_mask_id),
        "object_masklet_consistency_raw": object_consistency_raw,
    }


def _evidence_from_record(
    record: Any,
    *,
    query_info: dict[str, Any],
    label_maps: dict[int, np.ndarray],
    min_visibility: float,
    min_confidence: float,
) -> tuple[MaterialTubeEvidence, dict[str, Any]]:
    valid_stats = _visible_valid_stats(record, min_visibility=min_visibility, min_confidence=min_confidence)
    mask_stats = _mask_projection_stats(
        record,
        label_maps=label_maps,
        min_visibility=min_visibility,
        min_confidence=min_confidence,
    )
    reason = str(query_info.get("reason", "unmatched"))
    static_prior = float(STATIC_REASON_PRIOR.get(reason, STATIC_REASON_PRIOR["unmatched"]))
    object_prior = float(OBJECT_REASON_PRIOR.get(reason, OBJECT_REASON_PRIOR["unmatched"]))
    semantic_stability_raw = max(float(mask_stats["background_projection_ratio"]), float(mask_stats["dominant_mask_ratio"]))
    semantic_stability = _clip01(0.65 * semantic_stability_raw + 0.35 * static_prior)
    object_consistency = _clip01(float(mask_stats["object_masklet_consistency_raw"]) * (0.50 + 0.50 * object_prior))
    evidence = MaterialTubeEvidence(
        tube_id=int(record.tube_id),
        visibility=float(valid_stats["visibility_mean"]),
        confidence=float(valid_stats["confidence_mean"]),
        self_stitch_residual=float(valid_stats["self_stitch_residual_proxy"]),
        semantic_stability=float(semantic_stability),
        object_masklet_consistency=float(object_consistency),
        motion_magnitude=float(valid_stats["uv_motion_magnitude"]),
        scale_proxy=float(valid_stats["scale_proxy"]),
    )
    row = {
        "tube_id": int(record.tube_id),
        "persistent_tube_id": int(record.persistent_tube_id),
        "source_pixel_key": record.source_pixel_key,
        "source_frame_global": int(record.source_frame_global),
        "source_x": int(record.source_xy[0]),
        "source_y": int(record.source_xy[1]),
        "query_reason": reason,
        "query_score": float(query_info.get("score", 0.0)),
        "query_index": int(query_info.get("query_index", -1)),
        "semantic_static_prior": static_prior,
        "object_prior": object_prior,
        "semantic_stability_raw": float(semantic_stability_raw),
        **valid_stats,
        **mask_stats,
        "semantic_stability": float(evidence.semantic_stability),
        "object_masklet_consistency": float(evidence.object_masklet_consistency),
        "dynamic_proxy_label": bool(
            evidence.object_masklet_consistency >= 0.25
            and reason in {"mask_boundary", "mask_interior", "disagreement_proxy", "overlap_anchor"}
        ),
    }
    return evidence, row


def _role_summary(
    *,
    scene: str,
    variant: str,
    source: str,
    evidences: list[MaterialTubeEvidence],
    role_rows: list[dict[str, Any]],
    role_summary: dict[str, Any],
) -> dict[str, Any]:
    residuals = [float(e.self_stitch_residual) for e in evidences]
    scene_residuals = [float(row["self_stitch_residual_proxy"]) for row in role_rows if row["role"] == "scene"]
    object_or_part = [row for row in role_rows if row["role"] in {"object", "part"}]
    rejected = [row for row in role_rows if row["role"] in {"scene", "unknown"}]
    baseline_dynamic = float(sum(1 for row in role_rows if row["dynamic_proxy_label"]) / max(len(role_rows), 1))
    scene_dynamic = [
        row for row in role_rows if row["role"] == "scene" and bool(row["dynamic_proxy_label"])
    ]
    scene_count = int(sum(1 for row in role_rows if row["role"] == "scene"))
    dynamic_leakage_ratio = float(len(scene_dynamic) / max(scene_count, 1))
    if baseline_dynamic > 0.0:
        dynamic_reduction = float((baseline_dynamic - dynamic_leakage_ratio) / baseline_dynamic)
    else:
        dynamic_reduction = None
    object_support_mean = _safe_mean([float(row["object_masklet_consistency"]) for row in object_or_part])
    rejected_support_mean = _safe_mean([float(row["object_masklet_consistency"]) for row in rejected])
    unknown_ratio = float(role_summary.get("unknown_role_ratio", 0.0))
    residual_gate = (
        scene_residuals
        and residuals
        and float(np.median(np.asarray(scene_residuals, dtype=np.float64)))
        < float(np.median(np.asarray(residuals, dtype=np.float64)))
    )
    unknown_gate = 0.05 <= unknown_ratio <= 0.45
    dynamic_gate = dynamic_reduction is not None and float(dynamic_reduction) >= 0.30
    object_gate = (
        object_support_mean is not None
        and rejected_support_mean is not None
        and float(object_support_mean) > float(rejected_support_mean)
    )
    return {
        "scene": scene,
        "variant": variant,
        "source": source,
        **role_summary,
        "scene_anchor_count": int(scene_count),
        "dynamic_support_count": int(sum(1 for row in role_rows if row["role"] in {"object", "part"})),
        "static_anchor_residual_proxy_median": _safe_quantile(scene_residuals, 0.50),
        "static_anchor_residual_proxy_p90": _safe_quantile(scene_residuals, 0.90),
        "all_tube_residual_proxy_median": _safe_quantile(residuals, 0.50),
        "all_tube_residual_proxy_p90": _safe_quantile(residuals, 0.90),
        "baseline_dynamic_proxy_ratio": baseline_dynamic,
        "dynamic_leakage_ratio": dynamic_leakage_ratio,
        "dynamic_leakage_reduction_vs_all_tubes": dynamic_reduction,
        "object_support_consistency_object_or_part_mean": object_support_mean,
        "object_support_consistency_rejected_mean": rejected_support_mean,
        "residual_proxy_gate_pass": bool(residual_gate),
        "unknown_ratio_gate_pass": bool(unknown_gate),
        "dynamic_leakage_proxy_gate_pass": bool(dynamic_gate),
        "object_support_consistency_gate_pass": bool(object_gate),
        "phase5_proxy_gate_pass": bool(residual_gate and unknown_gate and dynamic_gate and object_gate),
        "phase5_gate_pass": False,
        "phase5_gate_blocker": "single_window_proxy_residual_and_dynamic_proxy_only",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
        "measurement_uses_metric_geometry": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v42 real role-conditioned material tube diagnostics.")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--variant", default="Q5")
    parser.add_argument("--sources", default="dinov2_maskcut")
    parser.add_argument("--frame-ids", default="0,10,20,30")
    parser.add_argument("--query-root", required=True)
    parser.add_argument("--material-cache-root", required=True)
    parser.add_argument("--external-source-root", default="outputs/audit/v42_source_audit_external_stride1_smoke")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--material-max-tubes-per-window", type=int, default=1024)
    parser.add_argument("--material-image-width", type=int, default=1296)
    parser.add_argument("--material-image-height", type=int, default=968)
    parser.add_argument("--material-min-visibility", type=float, default=0.5)
    parser.add_argument("--material-min-confidence", type=float, default=0.5)
    parser.add_argument("--backfill-overlap-iou", type=float, default=0.10)
    parser.add_argument("--backfill-max-masks-per-frame", type=int, default=8)
    parser.add_argument("--material-backfill-min-tubes", type=int, default=1)
    parser.add_argument("--material-backfill-max-candidate-area-fraction", type=float, default=1.0)
    parser.add_argument("--role-residual-scale", type=float, default=0.10)
    parser.add_argument("--role-motion-scale", type=float, default=0.08)
    parser.add_argument("--role-threshold", type=float, default=0.25)
    parser.add_argument("--role-margin", type=float, default=0.05)
    parser.add_argument("--role-scene-dynamic-penalty", type=float, default=0.50)
    parser.add_argument("--require-material-frame-stride", type=int, default=1)
    args = parser.parse_args()

    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    sources = [item.strip() for item in str(args.sources).split(",") if item.strip()]
    frame_ids = _parse_frame_ids(str(args.frame_ids))
    variant = str(args.variant)
    query_root = ROOT / str(args.query_root)
    material_cache_root = ROOT / str(args.material_cache_root)
    external_root = ROOT / str(args.external_source_root)
    output_root = ROOT / str(args.output_root)

    all_role_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for scene in scenes:
        stride_diag = _audit_material_cache_stride(
            cache_root=material_cache_root / variant,
            scene=scene,
            required_stride=int(args.require_material_frame_stride),
        )
        records, d4rt_diag = _load_d4rt_records(
            cache_root=material_cache_root / variant,
            scene=scene,
            max_tubes_per_window=int(args.material_max_tubes_per_window),
            image_width=int(args.material_image_width),
            image_height=int(args.material_image_height),
        )
        d4rt_diag["material_cache_stride_audit"] = stride_diag
        query_rows = _load_query_rows(query_root, scene, variant)
        stream = ScanNetStream(seq_name=scene)
        for source in sources:
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
                material_min_visibility=float(args.material_min_visibility),
                material_min_confidence=float(args.material_min_confidence),
            )
            label_maps = _label_maps_from_masks(masks_by_frame)
            evidences: list[MaterialTubeEvidence] = []
            base_rows: list[dict[str, Any]] = []
            for record in records:
                query_info = query_rows.get(record.source_pixel_key, {"reason": "unmatched", "score": 0.0, "query_index": -1})
                evidence, row = _evidence_from_record(
                    record,
                    query_info=query_info,
                    label_maps=label_maps,
                    min_visibility=float(args.material_min_visibility),
                    min_confidence=float(args.material_min_confidence),
                )
                evidences.append(evidence)
                base_rows.append(row)
            roles = infer_tube_roles(
                evidences,
                residual_scale=float(args.role_residual_scale),
                motion_scale=float(args.role_motion_scale),
                threshold=float(args.role_threshold),
                margin=float(args.role_margin),
                scene_dynamic_penalty=float(args.role_scene_dynamic_penalty),
            )
            role_by_id = {int(role.tube_id): role for role in roles}
            role_rows: list[dict[str, Any]] = []
            for row in base_rows:
                role = role_by_id[int(row["tube_id"])]
                role_rows.append(
                    {
                        "scene": scene,
                        "variant": variant,
                        "source": source,
                        **row,
                        "scene_role_weight": float(role.scene_role_weight),
                        "object_role_weight": float(role.object_role_weight),
                        "part_role_weight": float(role.part_role_weight),
                        "unknown_role_weight": float(role.unknown_role_weight),
                        "role": role.role,
                    }
                )
            summary = _role_summary(
                scene=scene,
                variant=variant,
                source=source,
                evidences=evidences,
                role_rows=role_rows,
                role_summary=summarize_tube_roles(evidences, roles),
            )
            stitch_rows = evaluate_role_aware_stitch_variants(evidences, roles)
            for row in stitch_rows:
                variant_rows.append({"scene": scene, "variant": variant, "source": source, **row})
            all_role_rows.extend(role_rows)
            summary_rows.append(summary)
            manifests.append(
                {
                    "scene": scene,
                    "variant": variant,
                    "source": source,
                    "d4rt_diag": d4rt_diag,
                    "mask_diag": mask_diag,
                    "query_row_count": int(len(query_rows)),
                    "role_params": {
                        "role_residual_scale": float(args.role_residual_scale),
                        "role_motion_scale": float(args.role_motion_scale),
                        "role_threshold": float(args.role_threshold),
                        "role_margin": float(args.role_margin),
                        "role_scene_dynamic_penalty": float(args.role_scene_dynamic_penalty),
                    },
                }
            )

    _write_csv(output_root / "tube_role_rows.csv", all_role_rows)
    _write_csv(output_root / "role_summary_rows.csv", summary_rows)
    _write_csv(output_root / "role_variant_rows.csv", variant_rows)
    _write_json(
        output_root / "role_summary.json",
        {
            "phase": "v42_tube_roles_real",
            "scenes": scenes,
            "variant": variant,
            "sources": sources,
            "frame_ids": frame_ids,
            "query_root": str(query_root),
            "material_cache_root": str(material_cache_root),
            "external_source_root": str(external_root),
            "tube_role_rows_csv": str(output_root / "tube_role_rows.csv"),
            "role_summary_rows_csv": str(output_root / "role_summary_rows.csv"),
            "role_variant_rows_csv": str(output_root / "role_variant_rows.csv"),
            "summary_rows": summary_rows,
            "manifests": manifests,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": False,
            "measurement_uses_metric_geometry": False,
            "note": "Single-window role diagnostic. self_stitch_residual and dynamic labels are proxies, so phase5_gate_pass remains false.",
        },
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "output_root": str(output_root),
                    "tube_role_rows": len(all_role_rows),
                    "summary_rows": len(summary_rows),
                    "variant_rows": len(variant_rows),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
