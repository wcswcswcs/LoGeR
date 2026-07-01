from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_float, read_json, utc_now, write_csv, write_json
from .v53_ap_diagnostic import build_v53_ap_diagnostic


BRIDGE_VARIANT = "AP7_v53_local_objectlet_wta_conflict_suppression"
AP4_VARIANT = "AP4_v64r2_rgbd_pose_mesh_bridge_diagnostic"
AP5_VARIANT = "AP5_v64r2_d4rt_chunk_scale_first_eval_sim3_pending"
AP_EVAL_SCOPE = "ap_eval_pre_points_no_class"
FULL_SCENE_SCOPE = "full_scene_all_gt_instances"
SCANNET_CLASS_AGNOSTIC_INSTANCE_OFFSET = 2000
D4RT_SCENE_SIM3_METRIC_PATH = (
    "data/evaluation/scannet/stream4d_v10_g2_d4rt_scene_sim3_probe5_class_agnostic.txt"
)
D4RT_SCENE_SIM3_SUMMARY_PATH = "outputs/v10_d4rt_geometry/stream4d_v10_g2_d4rt_scene_sim3_probe5_summary.json"
D4RT_SCENE_SIM3_MANIFEST_PATH = (
    "data/prediction/stream4d_v10_g2_d4rt_scene_sim3_probe5_class_agnostic/config_manifest.json"
)


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _load_dict(path: str | Path) -> dict[str, Any]:
    path_obj = _project(path)
    if not path_obj.exists():
        return {}
    payload = read_json(path_obj)
    return payload if isinstance(payload, dict) else {}


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _parse_ap_metric_file(path: str | Path) -> dict[str, float]:
    path_obj = _project(path)
    if not path_obj.exists():
        return {}
    lines = [line.strip() for line in path_obj.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return {}
    parts = lines[-1].split(",")
    if len(parts) != 3:
        return {}
    try:
        return {"AP": float(parts[0]), "AP50": float(parts[1]), "AP25": float(parts[2])}
    except ValueError:
        return {}


def _load_d4rt_chunk_scale_first_ap5() -> dict[str, Any]:
    metrics = _parse_ap_metric_file(D4RT_SCENE_SIM3_METRIC_PATH)
    summary = _load_dict(D4RT_SCENE_SIM3_SUMMARY_PATH)
    manifest = _load_dict(D4RT_SCENE_SIM3_MANIFEST_PATH)
    numeric_mean = summary.get("numeric_mean", {}) if isinstance(summary.get("numeric_mean"), dict) else {}
    return {
        "status": "blocked_chunk_scale_first_d4rt_ap_not_materialized",
        "AP": None,
        "AP50": None,
        "AP25": None,
        "legacy_scene_sim3_AP": metrics.get("AP"),
        "legacy_scene_sim3_AP50": metrics.get("AP50"),
        "legacy_scene_sim3_AP25": metrics.get("AP25"),
        "prediction_count": numeric_mean.get("num_exported_objects"),
        "mean_predictions_per_scene": numeric_mean.get("num_exported_objects"),
        "prediction_union_ratio": None,
        "pre_points_percent": None,
        "metric_file": _rel(D4RT_SCENE_SIM3_METRIC_PATH),
        "summary_path": _rel(D4RT_SCENE_SIM3_SUMMARY_PATH),
        "manifest_path": _rel(D4RT_SCENE_SIM3_MANIFEST_PATH),
        "source_output_config": manifest.get("output_config"),
        "source_command": manifest.get("command"),
        "support_policy": manifest.get("support_policy"),
        "pre_points_policy": manifest.get("pre_points_policy"),
        "geometry_source": manifest.get("geometry_source"),
        "gt_usage": manifest.get("gt_usage"),
        "source_config": (manifest.get("source_configs") or [""])[0] if isinstance(manifest.get("source_configs"), list) else "",
        "mode": summary.get("mode"),
        "algorithm": summary.get("algorithm"),
        "num_scenes": summary.get("num_scenes"),
        "anchor_count_mean": numeric_mean.get("anchor_count"),
        "anchor_valid_mean": numeric_mean.get("anchor_valid"),
        "median_residual_mean": numeric_mean.get("median_residual"),
        "p90_residual_mean": numeric_mean.get("p90_residual"),
        "num_3d_masks_after_projection_mean": numeric_mean.get("num_3d_masks_after_projection"),
        "num_exported_points_mean": numeric_mean.get("num_exported_points"),
        "empty_projected_mask_ratio_mean": numeric_mean.get("empty_projected_mask_ratio"),
        "uses_gt_for_prediction": bool(manifest.get("uses_gt_for_prediction", False)),
        "uses_gt_for_evaluation_alignment": False,
        "uses_gt_for_diagnostic": bool(manifest.get("uses_gt_for_diagnostic", True)),
        "is_method_result": bool(manifest.get("is_method_result", False)),
        "is_diagnostic_only": bool(manifest.get("is_diagnostic_only", True)),
        "failure_reason": (
            "not run: legacy v10 scene_sim3 AP did not satisfy the current chunk-scale-first requirement; "
            "v44 recheck still has one adjacent pair outside 10% (scene0081_01 window 1-2 scale_next_over_prev=0.8872272553594305)."
        ),
    }


def _metric_row(
    *,
    variant: str,
    status: str,
    export_policy: str,
    forbidden_for_method_table: bool,
    uses_rgbd_pose_mesh_for_export: bool,
    is_method_result: bool,
    source_row: dict[str, Any] | None = None,
    failure_reason: str = "",
) -> dict[str, Any]:
    source_row = source_row or {}
    return {
        "variant": variant,
        "status": status,
        "AP": source_row.get("AP"),
        "AP50": source_row.get("AP50"),
        "AP25": source_row.get("AP25"),
        "prediction_count": None,
        "mean_predictions_per_scene": source_row.get("predictions_per_scene"),
        "prediction_union_ratio": source_row.get("mean_union_percent"),
        "pre_points_percent": source_row.get("mean_pre_percent"),
        "gt_coverage_recall@0.25": None,
        "gt_coverage_recall@0.50": None,
        "per_GT_best_IoU_mean": None,
        "duplicate_predictions_per_GT": None,
        "gt_coverage_scope": None,
        "ap_eval_gt_count": None,
        "full_scene_gt_coverage_recall@0.25": None,
        "full_scene_gt_coverage_recall@0.50": None,
        "full_scene_per_GT_best_IoU_mean": None,
        "full_scene_duplicate_predictions_per_GT": None,
        "full_scene_coverage_scope": None,
        "full_scene_gt_count": None,
        "conflict_rate": source_row.get("mean_export_conflict_rate"),
        "empty_prediction_rate": source_row.get("empty_prediction_rate"),
        "tiny_prediction_rate": None,
        "runtime_sec": None,
        "export_policy": export_policy,
        "forbidden_for_method_table": forbidden_for_method_table,
        "uses_gt_for_prediction": False,
        "uses_gt_for_evaluation_alignment": False,
        "uses_rgbd_pose_mesh_for_export": uses_rgbd_pose_mesh_for_export,
        "is_method_result": is_method_result,
        "is_diagnostic_only": not is_method_result,
        "failure_reason": failure_reason,
    }


def _load_gt_ids(scene: str) -> np.ndarray | None:
    path = ROOT / "data/scannet/gt" / f"{scene}.txt"
    if not path.exists():
        return None
    try:
        return np.loadtxt(path, dtype=np.int64)
    except Exception:
        return None


def _load_pre_points(config: str, scene: str) -> np.ndarray | None:
    path = ROOT / "data/TMP" / config / f"{scene}_pre_points.npy"
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _empty_iou_row(scene: str, iou_scope: str, failure_reason: str) -> dict[str, Any]:
    return {
        "variant": AP4_VARIANT,
        "scene_id": scene,
        "iou_scope": iou_scope,
        "gt_instance_id": "",
        "gt_area": "",
        "best_prediction_id": "",
        "best_iou": None,
        "best_prediction_score": None,
        "duplicate_predictions_at_0p25": None,
        "duplicate_predictions_at_0p50": None,
        "failure_reason": failure_reason,
    }


def _scope_prediction_and_gt(
    *,
    config: str,
    scene: str,
    iou_scope: str,
    gt_ids: np.ndarray,
    pred_masks: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any], str]:
    metadata: dict[str, Any] = {
        "total_points": int(gt_ids.shape[0]),
        "prediction_points": int(pred_masks.shape[0]) if pred_masks.ndim >= 1 else None,
    }
    if pred_masks.ndim != 2:
        return None, None, metadata, f"prediction_mask_rank_{pred_masks.ndim}_not_2"
    if gt_ids.shape[0] != pred_masks.shape[0]:
        return None, None, metadata, f"gt_pred_length_mismatch_gt={gt_ids.shape[0]}_pred={pred_masks.shape[0]}"
    if iou_scope == FULL_SCENE_SCOPE:
        metadata["no_class_transform"] = False
        return pred_masks, gt_ids, metadata, ""
    if iou_scope != AP_EVAL_SCOPE:
        return None, None, metadata, f"unknown_iou_scope={iou_scope}"

    pre_points = _load_pre_points(config, scene)
    if pre_points is None:
        return None, None, metadata, "missing_or_unreadable_tmp_pre_points"
    pre_points = np.asarray(pre_points, dtype=np.int64)
    metadata["pre_points_count"] = int(pre_points.shape[0])
    metadata["no_class_transform"] = True
    if pre_points.size and (int(pre_points.min()) < 0 or int(pre_points.max()) >= gt_ids.shape[0]):
        return None, None, metadata, "tmp_pre_points_index_out_of_bounds"
    scoped_pred_masks = pred_masks[pre_points, :]
    scoped_gt_ids = gt_ids[pre_points] % 1000 + SCANNET_CLASS_AGNOSTIC_INSTANCE_OFFSET
    return scoped_pred_masks, scoped_gt_ids, metadata, ""


def _per_gt_iou_rows_for_bridge(
    config: str,
    scene_rows: list[dict[str, Any]],
    *,
    iou_scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_root = ROOT / "data/prediction" / f"{config}_class_agnostic"
    for scene_row in scene_rows:
        scene = str(scene_row.get("scene") or "")
        pred_path = pred_root / f"{scene}.npz"
        gt_ids = _load_gt_ids(scene)
        if gt_ids is None or not pred_path.exists():
            rows.append(_empty_iou_row(scene, iou_scope, "missing_gt_or_prediction_file"))
            continue
        with np.load(pred_path) as payload:
            pred_masks = np.asarray(payload["pred_masks"], dtype=bool)
            pred_scores = np.asarray(payload["pred_score"], dtype=np.float64)
        scoped_pred_masks, scoped_gt_ids, metadata, failure_reason = _scope_prediction_and_gt(
            config=config,
            scene=scene,
            iou_scope=iou_scope,
            gt_ids=gt_ids,
            pred_masks=pred_masks,
        )
        if scoped_pred_masks is None or scoped_gt_ids is None:
            row = _empty_iou_row(scene, iou_scope, failure_reason)
            row.update(metadata)
            rows.append(row)
            continue
        pred_areas = scoped_pred_masks.sum(axis=0).astype(np.float64)
        valid_gt_ids = [int(value) for value in np.unique(scoped_gt_ids) if int(value) >= 1000]
        for gt_id in valid_gt_ids:
            gt_mask = scoped_gt_ids == int(gt_id)
            gt_area = float(np.count_nonzero(gt_mask))
            if gt_area < 100.0 or scoped_pred_masks.shape[1] == 0:
                continue
            intersections = scoped_pred_masks[gt_mask, :].sum(axis=0).astype(np.float64)
            unions = gt_area + pred_areas - intersections
            ious = np.divide(intersections, np.maximum(unions, 1.0))
            best_idx = int(np.argmax(ious)) if ious.size else -1
            best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0
            row = {
                "variant": AP4_VARIANT,
                "scene_id": scene,
                "iou_scope": iou_scope,
                "gt_instance_id": gt_id,
                "gt_area": gt_area,
                "best_prediction_id": best_idx,
                "best_iou": best_iou,
                "best_prediction_score": float(pred_scores[best_idx]) if best_idx >= 0 and best_idx < len(pred_scores) else None,
                "duplicate_predictions_at_0p25": int(np.count_nonzero(ious >= 0.25)),
                "duplicate_predictions_at_0p50": int(np.count_nonzero(ious >= 0.50)),
                "failure_reason": "" if best_iou > 0.0 else "no_overlap_with_any_prediction",
            }
            row.update(metadata)
            rows.append(row)
    return rows


def _prediction_trace_rows(config: str, scene_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_root = ROOT / "data/prediction" / f"{config}_class_agnostic"
    for scene_row in scene_rows:
        scene = str(scene_row.get("scene") or "")
        pred_path = pred_root / f"{scene}.npz"
        if not pred_path.exists():
            continue
        with np.load(pred_path) as payload:
            pred_masks = np.asarray(payload["pred_masks"], dtype=bool)
            pred_scores = np.asarray(payload["pred_score"], dtype=np.float64)
        for idx in range(pred_masks.shape[1]):
            rows.append(
                {
                    "variant": AP4_VARIANT,
                    "scene_id": scene,
                    "prediction_id": idx,
                    "prediction_point_count": int(np.count_nonzero(pred_masks[:, idx])),
                    "prediction_score": float(pred_scores[idx]) if idx < len(pred_scores) else None,
                    "trace_policy": "v53_objectlet_source_mask_backproject_wta",
                    "history_id": "",
                    "material_ids": [],
                }
            )
    return rows


def _iou_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if _finite(row.get("best_iou"))]
    if not valid:
        return {"count": 0}
    best_ious = [float(row["best_iou"]) for row in valid]
    duplicate_counts = [float(row.get("duplicate_predictions_at_0p25") or 0.0) for row in valid]
    return {
        "count": len(valid),
        "recall25": sum(value >= 0.25 for value in best_ious) / max(len(best_ious), 1),
        "recall50": sum(value >= 0.50 for value in best_ious) / max(len(best_ious), 1),
        "mean_iou": float(np.mean(best_ious)),
        "mean_duplicates": float(np.mean(duplicate_counts)),
    }


def _apply_iou_summary(
    ap_rows: list[dict[str, Any]],
    per_gt_rows: list[dict[str, Any]],
    full_scene_rows: list[dict[str, Any]],
) -> None:
    ap_eval_stats = _iou_stats(per_gt_rows)
    full_scene_stats = _iou_stats(full_scene_rows)
    for row in ap_rows:
        if row["variant"] == AP4_VARIANT:
            if ap_eval_stats["count"]:
                row["gt_coverage_recall@0.25"] = ap_eval_stats["recall25"]
                row["gt_coverage_recall@0.50"] = ap_eval_stats["recall50"]
                row["per_GT_best_IoU_mean"] = ap_eval_stats["mean_iou"]
                row["duplicate_predictions_per_GT"] = ap_eval_stats["mean_duplicates"]
                row["gt_coverage_scope"] = AP_EVAL_SCOPE
                row["ap_eval_gt_count"] = ap_eval_stats["count"]
            if full_scene_stats["count"]:
                row["full_scene_gt_coverage_recall@0.25"] = full_scene_stats["recall25"]
                row["full_scene_gt_coverage_recall@0.50"] = full_scene_stats["recall50"]
                row["full_scene_per_GT_best_IoU_mean"] = full_scene_stats["mean_iou"]
                row["full_scene_duplicate_predictions_per_GT"] = full_scene_stats["mean_duplicates"]
                row["full_scene_coverage_scope"] = FULL_SCENE_SCOPE
                row["full_scene_gt_count"] = full_scene_stats["count"]
            row["prediction_count"] = sum(1 for trace in row.get("_trace_rows", []))
            row.pop("_trace_rows", None)


def build_v64r2_scannet_ap_probe5(
    *,
    output_root: str | Path = "outputs/audit/v64r2_scannet_ap_probe5",
    output_config_prefix: str = "v64r2_probe5_v53_bridge",
    export_mask_sample_stride: int = 4,
    export_mask_max_pixels: int = 30000,
    export_nn_radius: float = 0.05,
) -> dict[str, Any]:
    bridge_payload = build_v53_ap_diagnostic(
        output_root=output_root,
        output_config_prefix=output_config_prefix,
        export_mask_sample_stride=export_mask_sample_stride,
        export_mask_max_pixels=export_mask_max_pixels,
        export_nn_radius=export_nn_radius,
    )
    bridge_rows = {str(row.get("variant")): row for row in bridge_payload.get("ap_rows", [])}
    bridge_details = bridge_payload.get("bridge_details", {})
    best_bridge = bridge_rows.get(BRIDGE_VARIANT, {})
    bridge_detail = bridge_details.get(BRIDGE_VARIANT, {}) if isinstance(bridge_details, dict) else {}
    bridge_config = str(bridge_detail.get("config") or f"{output_config_prefix}_wta")
    scene_rows = list(bridge_detail.get("scene_rows", []) if isinstance(bridge_detail, dict) else [])
    per_gt_rows = _per_gt_iou_rows_for_bridge(bridge_config, scene_rows, iou_scope=AP_EVAL_SCOPE)
    full_scene_rows = _per_gt_iou_rows_for_bridge(bridge_config, scene_rows, iou_scope=FULL_SCENE_SCOPE)
    trace_rows = _prediction_trace_rows(bridge_config, scene_rows)
    ap_rows = [
        _metric_row(
            variant="AP0_v62_native_component_field",
            status="native_component_field_available_no_scannet_mask",
            export_policy="native_component_field_no_mesh_mask",
            forbidden_for_method_table=False,
            uses_rgbd_pose_mesh_for_export=False,
            is_method_result=False,
            failure_reason="component-level ownership field has no ScanNet point/mesh mask adapter",
        ),
        _metric_row(
            variant="AP1_v64r2_confirmed_core_only",
            status="blocked_method_safe_materializer",
            export_policy="confirmed_core_only_native_method_safe",
            forbidden_for_method_table=False,
            uses_rgbd_pose_mesh_for_export=False,
            is_method_result=False,
            failure_reason="missing native component/carrier to ScanNet point/mesh calibration",
        ),
        _metric_row(
            variant="AP2_v64r2_confirmed_plus_tentative",
            status="blocked_method_safe_materializer",
            export_policy="confirmed_plus_tentative_native_method_safe",
            forbidden_for_method_table=False,
            uses_rgbd_pose_mesh_for_export=False,
            is_method_result=False,
            failure_reason="missing native component/carrier to ScanNet point/mesh calibration",
        ),
        _metric_row(
            variant="AP3_v64r2_confirmed_plus_shared_excluded",
            status="blocked_method_safe_materializer",
            export_policy="confirmed_plus_shared_excluded_native_method_safe",
            forbidden_for_method_table=False,
            uses_rgbd_pose_mesh_for_export=False,
            is_method_result=False,
            failure_reason="missing native component/carrier to ScanNet point/mesh calibration",
        ),
        _metric_row(
            variant=AP4_VARIANT,
            status="ran" if best_bridge.get("status") == "ran" else str(best_bridge.get("status") or "failed"),
            export_policy="rgbd_pose_mesh_bridge_wta_diagnostic",
            forbidden_for_method_table=True,
            uses_rgbd_pose_mesh_for_export=True,
            is_method_result=False,
            source_row=best_bridge,
            failure_reason="" if best_bridge.get("status") == "ran" else "diagnostic bridge did not complete",
        ),
        _metric_row(
            variant=AP5_VARIANT,
            status="blocked_chunk_scale_first_d4rt_ap_not_materialized",
            export_policy="d4rt_chunk_scale_first_eval_sim3_point_mesh_materialization_pending",
            forbidden_for_method_table=True,
            uses_rgbd_pose_mesh_for_export=False,
            is_method_result=False,
            failure_reason="D4RT chunk-scale-first aligned point/mesh AP export has not been materialized",
        ),
        _metric_row(
            variant="AP6_stream3d_cropformer_baseline_if_available",
            status="not_available_in_current_artifacts",
            export_policy="stream3d_cropformer_baseline",
            forbidden_for_method_table=True,
            uses_rgbd_pose_mesh_for_export=False,
            is_method_result=False,
            failure_reason="no locked same-split Stream3D-CropFormer baseline artifact found for v64r2 probe5",
        ),
    ]
    for row in ap_rows:
        if row["variant"] == AP4_VARIANT:
            row["_trace_rows"] = trace_rows
        if row["variant"] == AP5_VARIANT:
            ap5_source = _load_d4rt_chunk_scale_first_ap5()
            row.update(
                {
                    "status": ap5_source.get("status"),
                    "AP": ap5_source.get("AP"),
                    "AP50": ap5_source.get("AP50"),
                    "AP25": ap5_source.get("AP25"),
                    "prediction_count": ap5_source.get("prediction_count"),
                    "mean_predictions_per_scene": ap5_source.get("mean_predictions_per_scene"),
                    "pre_points_percent": ap5_source.get("pre_points_percent"),
                    "prediction_union_ratio": ap5_source.get("prediction_union_ratio"),
                    "uses_gt_for_prediction": ap5_source.get("uses_gt_for_prediction", False),
                    "uses_gt_for_evaluation_alignment": ap5_source.get("uses_gt_for_evaluation_alignment", False),
                    "uses_gt_for_diagnostic": ap5_source.get("uses_gt_for_diagnostic", True),
                    "uses_rgbd_pose_mesh_for_export": False,
                    "is_method_result": False,
                    "is_diagnostic_only": True,
                    "failure_reason": ap5_source.get("failure_reason", ""),
                    "source_metric_file": ap5_source.get("metric_file"),
                    "source_summary_path": ap5_source.get("summary_path"),
                    "source_manifest_path": ap5_source.get("manifest_path"),
                    "source_output_config": ap5_source.get("source_output_config"),
                    "source_command": ap5_source.get("source_command"),
                    "source_support_policy": ap5_source.get("support_policy"),
                    "source_pre_points_policy": ap5_source.get("pre_points_policy"),
                    "source_geometry_source": ap5_source.get("geometry_source"),
                    "source_gt_usage": ap5_source.get("gt_usage"),
                    "source_debug_root": ap5_source.get("source_config"),
                    "d4rt_sim3_mode": ap5_source.get("mode"),
                    "d4rt_sim3_algorithm": ap5_source.get("algorithm"),
                    "d4rt_scene_count": ap5_source.get("num_scenes"),
                    "d4rt_anchor_count_mean": ap5_source.get("anchor_count_mean"),
                    "d4rt_anchor_valid_mean": ap5_source.get("anchor_valid_mean"),
                    "d4rt_median_residual_mean": ap5_source.get("median_residual_mean"),
                    "d4rt_p90_residual_mean": ap5_source.get("p90_residual_mean"),
                    "d4rt_3d_masks_after_projection_mean": ap5_source.get("num_3d_masks_after_projection_mean"),
                    "d4rt_exported_points_mean": ap5_source.get("num_exported_points_mean"),
                    "d4rt_empty_projected_mask_ratio_mean": ap5_source.get("empty_projected_mask_ratio_mean"),
                    "legacy_scene_sim3_AP": ap5_source.get("legacy_scene_sim3_AP"),
                    "legacy_scene_sim3_AP50": ap5_source.get("legacy_scene_sim3_AP50"),
                    "legacy_scene_sim3_AP25": ap5_source.get("legacy_scene_sim3_AP25"),
                }
            )
    _apply_iou_summary(ap_rows, per_gt_rows, full_scene_rows)
    ap4 = next(row for row in ap_rows if row["variant"] == AP4_VARIANT)
    ap5 = next(row for row in ap_rows if row["variant"] == AP5_VARIANT)
    evaluator_runs = best_bridge.get("status") == "ran"
    ap_finite = _finite(ap4.get("AP"))
    diagnostic_useful = bool(
        (ap_finite and float(ap4["AP"]) > 0.0)
        or (_finite(ap4.get("AP25")) and float(ap4["AP25"]) > 0.0)
        or (_finite(ap4.get("per_GT_best_IoU_mean")) and float(ap4["per_GT_best_IoU_mean"]) > 0.0)
    )
    method_candidates = [
        row
        for row in ap_rows
        if _finite(row.get("AP")) and not bool(row.get("forbidden_for_method_table")) and not bool(row.get("uses_rgbd_pose_mesh_for_export"))
    ]
    diagnostic_candidates = [
        row
        for row in ap_rows
        if _finite(row.get("AP")) and bool(row.get("forbidden_for_method_table"))
    ]
    best_diagnostic = max(diagnostic_candidates, key=lambda row: float(row["AP"])) if diagnostic_candidates else ap4
    gate = {
        "evaluator_runs": bool(evaluator_runs),
        "AP_finite_or_failure_explicit": bool(ap_finite or ap4.get("failure_reason")),
        "prediction_count_gt_0": bool(trace_rows),
        "mean_predictions_per_scene_le_300": parse_float(ap4.get("mean_predictions_per_scene"), 9999.0) <= 300.0,
        "policy_rows_complete": all("forbidden_for_method_table" in row for row in ap_rows),
        "ap_smoke_pass": False,
        "ap_diagnostic_useful": diagnostic_useful,
        "method_ap_candidate_available": bool(method_candidates),
    }
    gate["ap_smoke_pass"] = bool(
        gate["evaluator_runs"]
        and gate["AP_finite_or_failure_explicit"]
        and gate["prediction_count_gt_0"]
        and gate["mean_predictions_per_scene_le_300"]
        and gate["policy_rows_complete"]
    )
    summary = {
        "phase": "v64r2_scannet_ap_probe5",
        "created_at": utc_now(),
        "split": "probe5",
        "gate": gate,
        "best_diagnostic_AP": best_diagnostic.get("AP"),
        "best_diagnostic_AP50": best_diagnostic.get("AP50"),
        "best_diagnostic_AP25": best_diagnostic.get("AP25"),
        "best_diagnostic_variant": best_diagnostic["variant"],
        "ap4_rgbd_pose_mesh_bridge_AP": ap4.get("AP"),
        "ap4_rgbd_pose_mesh_bridge_AP50": ap4.get("AP50"),
        "ap4_rgbd_pose_mesh_bridge_AP25": ap4.get("AP25"),
        "ap5_d4rt_chunk_scale_first_AP": ap5.get("AP"),
        "ap5_d4rt_chunk_scale_first_AP50": ap5.get("AP50"),
        "ap5_d4rt_chunk_scale_first_AP25": ap5.get("AP25"),
        "ap5_d4rt_chunk_scale_first_status": ap5.get("status"),
        "ap5_legacy_scene_sim3_reference_AP": ap5.get("legacy_scene_sim3_AP"),
        "ap5_legacy_scene_sim3_reference_AP50": ap5.get("legacy_scene_sim3_AP50"),
        "ap5_legacy_scene_sim3_reference_AP25": ap5.get("legacy_scene_sim3_AP25"),
        "method_safe_AP_available": bool(method_candidates),
        "diagnostic_AP_available": bool(diagnostic_candidates),
        "scannet_ap_status": "PARTIAL_SCANNET_AP_DIAGNOSTIC" if ap_finite and not method_candidates else (
            "GO_SCANNET_AP_METHOD" if method_candidates else "NO_GO_SCANNET_MATERIALIZATION"
        ),
        "bridge_config": bridge_config,
        "bridge_metric_scope": bridge_payload.get("summary", {}).get("metric_scope"),
        "ap_evaluation_scope": AP_EVAL_SCOPE,
        "ap_scope_gt_count": ap4.get("ap_eval_gt_count"),
        "full_scene_coverage_scope": FULL_SCENE_SCOPE,
        "full_scene_gt_count": ap4.get("full_scene_gt_count"),
        "scope_note": "AP row IoU metrics use evaluator pre_points with no_class GT transform; full-scene coverage metrics are diagnostic-only and reported separately.",
        "raw_bridge_payload": bridge_payload,
    }
    return {
        "summary": summary,
        "ap_metric_rows": ap_rows,
        "per_gt_iou_rows": per_gt_rows,
        "full_scene_coverage_rows": full_scene_rows,
        "prediction_trace_rows": trace_rows,
        "ap_scene_rows": scene_rows,
    }


def write_v64r2_scannet_ap_probe5(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "ap_smoke_summary.json", payload["summary"])
    write_csv(out / "ap_metric_rows.csv", payload["ap_metric_rows"])
    write_csv(out / "per_gt_iou_rows.csv", payload["per_gt_iou_rows"])
    write_csv(out / "full_scene_coverage_rows.csv", payload["full_scene_coverage_rows"])
    write_csv(out / "prediction_trace_rows.csv", payload["prediction_trace_rows"])
    write_csv(out / "ap_scene_rows.csv", payload["ap_scene_rows"])


def build_v64r2_scannet_ap_locked_split(
    *,
    split: str,
    probe_summary_path: str | Path = "outputs/audit/v64r2_scannet_ap_probe5/ap_smoke_summary.json",
    probe_metric_rows_path: str | Path = "outputs/audit/v64r2_scannet_ap_probe5/ap_metric_rows.csv",
) -> dict[str, Any]:
    from .v47_common import read_csv

    probe_summary = _load_dict(probe_summary_path)
    probe_rows = read_csv(_project(probe_metric_rows_path)) if _project(probe_metric_rows_path).exists() else []
    method_safe_available = bool(probe_summary.get("method_safe_AP_available"))
    diagnostic_available = bool(probe_summary.get("diagnostic_AP_available"))
    rows: list[dict[str, Any]] = []
    for row in probe_rows:
        rows.append(
            {
                "split": split,
                "variant": row.get("variant"),
                "AP": None,
                "AP50": None,
                "AP25": None,
                "prediction_union_ratio": None,
                "gt_coverage_recall@0.25": None,
                "gt_coverage_recall@0.50": None,
                "num_instances": None,
                "mean_predictions_per_scene": None,
                "runtime_per_frame": None,
                "export_policy": row.get("export_policy"),
                "method_safety_policy": "method_safe_required_for_tune_final",
                "status": "not_run_method_safe_ap_unavailable" if not method_safe_available else "not_run_locked_split_not_launched",
                "failure_reason": (
                    "probe5 only has diagnostic RGB-D/pose/mesh AP; method-safe materializer is blocked"
                    if not method_safe_available
                    else "locked split runner not launched in this v64r2 pass"
                ),
            }
        )
    summary_name = "ap_tune_summary" if split == "tune30" else "ap_final_summary"
    summary = {
        "phase": f"v64r2_scannet_ap_{split}",
        "created_at": utc_now(),
        "split": split,
        "status": "not_run_method_safe_ap_unavailable" if not method_safe_available else "not_run_locked_split_not_launched",
        "method_safe_AP_available": method_safe_available,
        "diagnostic_AP_available": diagnostic_available,
        "probe5_scannet_ap_status": probe_summary.get("scannet_ap_status"),
        "blocked_reason": "Do not tune or final-evaluate diagnostic RGB-D/pose/mesh bridge as method AP."
        if not method_safe_available
        else "Locked split was not launched.",
        "gate": {
            "method_safe_AP_available": method_safe_available,
            "locked_hyperparameters": True,
            "final_or_tune_metrics_finite": False,
            "pass": False,
        },
        "output_summary_name": summary_name,
    }
    return {"summary": summary, "ap_metric_rows": rows}


def write_v64r2_scannet_ap_locked_split(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    summary_name = payload["summary"]["output_summary_name"]
    write_json(out / f"{summary_name}.json", payload["summary"])
    write_csv(out / "ap_metric_rows.csv", payload["ap_metric_rows"])
