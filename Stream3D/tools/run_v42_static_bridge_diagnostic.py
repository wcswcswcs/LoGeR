from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _safe_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _source_has(path: Path, text: str) -> bool:
    return path.exists() and text in path.read_text(encoding="utf-8")


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(np.asarray(clean, dtype=np.float64))) if clean else None


def _import_reference_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "table4_label": row.get("table4_label", ""),
                "source_row_id": row.get("source_row_id", ""),
                "status": row.get("status", ""),
                "AP": _safe_float(row.get("AP", "")),
                "AP50": _safe_float(row.get("AP50", "")),
                "AP25": _safe_float(row.get("AP25", "")),
                "predictions_per_scene": _safe_float(row.get("mean_predictions_per_scene", "")),
                "conflict": "",
                "multiplicity": "",
                "per_GT_best_IoU@50": "",
                "D4RT_hit_rate": "",
                "native_support_4D_ARI": "",
                "native_support_purity": "",
                "native_support_completeness": "",
                "unknown_labeled_tube_ratio": "",
                "is_method_result": row.get("is_method_result", ""),
                "is_diagnostic_only": row.get("is_diagnostic_only", ""),
                "forbidden_for_method_table": row.get("forbidden_for_method_table", ""),
                "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
                "uses_rgbd_for_prediction": row.get("uses_rgbd_for_prediction", ""),
                "uses_pose_for_prediction": row.get("uses_pose_for_prediction", ""),
                "uses_scannet_mesh_for_prediction": row.get("uses_scannet_mesh_for_prediction", ""),
                "uses_eval_sim3_for_prediction": row.get("uses_eval_sim3_for_prediction", ""),
                "geometry_backend": row.get("geometry_backend", ""),
                "object_algorithm": row.get("object_algorithm", ""),
                "materializer": row.get("materializer", ""),
                "source_artifact": row.get("source_artifact", ""),
                "method_compatibility_manifest": {
                    "imported_from_reference_table": True,
                    "training_free": row.get("training_free", ""),
                },
                "note": row.get("note", ""),
            }
        )
    return out


def _v42_native_support_row(memory_summary: dict[str, Any], scene_rows: list[dict[str, str]]) -> dict[str, Any]:
    scene_count = int(memory_summary.get("scene_count", len(scene_rows)) or len(scene_rows) or 1)
    hit_rates = []
    for row in scene_rows:
        exported = _safe_float(row.get("exported_tube_count", ""))
        cache = _safe_float(row.get("cache_tube_count", ""))
        if exported is not None and cache is not None and cache > 0:
            hit_rates.append(float(exported / cache))
    return {
        "table4_label": "v42 ObjectField + D4RT native support",
        "source_row_id": "V42-O-D4RT-native-support-memory",
        "status": "ok_native_support_not_ap",
        "AP": "",
        "AP50": "",
        "AP25": "",
        "predictions_per_scene": float(memory_summary.get("expanded_object_field_count", 0) / max(scene_count, 1)),
        "conflict": "",
        "multiplicity": "",
        "per_GT_best_IoU@50": "",
        "D4RT_hit_rate": _mean(hit_rates),
        "native_support_4D_ARI": memory_summary.get("aggregate_tube_4D_ARI"),
        "native_support_purity": memory_summary.get("aggregate_tube_purity"),
        "native_support_completeness": memory_summary.get("aggregate_tube_completeness"),
        "unknown_labeled_tube_ratio": memory_summary.get("mean_unknown_labeled_tube_ratio"),
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": False,
        "uses_gt_for_prediction": False,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "geometry_backend": "D4RT native support points",
        "object_algorithm": "v42 semantic-material factor graph + stride-1 streaming memory",
        "materializer": "native support export only, not ScanNet AP masks",
        "source_artifact": str(memory_summary.get("object_field_root", "")),
        "method_compatibility_manifest": {
            "uses_gt_for_prediction": False,
            "uses_rgbd_for_prediction": False,
            "uses_pose_for_prediction": False,
            "uses_scannet_mesh_for_prediction": False,
            "is_method_ap_result": False,
            "AP_bridge_status": memory_summary.get("AP_bridge_status"),
            "native_support_metric_proxy_pass": memory_summary.get("native_support_metric_proxy_pass"),
            "phase8_gate_pass": memory_summary.get("phase8_gate_pass"),
        },
        "note": "Native support proxy improved after stride-1 memory; still not AP.",
    }


def _v42_blocked_rows(memory_summary: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "AP": "",
        "AP50": "",
        "AP25": "",
        "conflict": "",
        "multiplicity": "",
        "per_GT_best_IoU@50": "",
        "D4RT_hit_rate": "",
        "native_support_4D_ARI": "",
        "native_support_purity": "",
        "native_support_completeness": "",
        "unknown_labeled_tube_ratio": "",
        "uses_gt_for_prediction": False,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "object_algorithm": "v42 semantic-material factor graph + stride-1 streaming memory",
        "source_artifact": str(memory_summary.get("object_field_root", "")),
    }
    return [
        {
            **common,
            "table4_label": "v42 ObjectField + diagnostic GT/RGB-D geometry",
            "source_row_id": "V42-O-GTGeo-diagnostic",
            "status": "not_run_materializer_missing",
            "predictions_per_scene": "",
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "geometry_backend": "ScanNet RGB-D/pose/mesh diagnostic bridge",
            "materializer": "not implemented for v42 memory object fields",
            "method_compatibility_manifest": {
                "not_run_reason": "No v42 ObjectField to ScanNet mesh diagnostic materializer has been implemented in this run.",
                "would_be_diagnostic_only": True,
            },
            "note": "Needed to distinguish representation vs D4RT bridge, but not run yet.",
        },
        {
            **common,
            "table4_label": "v42 ObjectField + method-compatible AP bridge",
            "source_row_id": "V42-O-method-AP",
            "status": "blocked_native_ap_exporter_missing",
            "predictions_per_scene": "",
            "is_method_result": False,
            "is_diagnostic_only": False,
            "forbidden_for_method_table": False,
            "geometry_backend": "D4RT native to ScanNet AP calibration",
            "materializer": "missing",
            "method_compatibility_manifest": {
                "blocked_reason": "ScanNet AP needs mesh-vertex prediction masks; current native support points have no mesh vertex ids and export_d4rt_nn remains unimplemented.",
                "AP_bridge_status": memory_summary.get("AP_bridge_status"),
            },
            "note": "This is the current remaining blocker after native support proxy improved.",
        },
    ]


def _v42_gtgeo_row_from_summary(gtgeo_summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = dict(gtgeo_summary.get("aggregate", {}))
    scene_count = int(aggregate.get("scene_count", 0) or 0)
    return {
        "table4_label": "v42 ObjectField + diagnostic GT/RGB-D geometry",
        "source_row_id": "V42-O-GTGeo-diagnostic",
        "status": "ok_diagnostic_gtgeo_not_method",
        "AP": aggregate.get("AP"),
        "AP50": aggregate.get("AP50"),
        "AP25": aggregate.get("AP25"),
        "predictions_per_scene": aggregate.get("mean_num_predictions"),
        "conflict": aggregate.get("mean_export_conflict_rate"),
        "multiplicity": "",
        "per_GT_best_IoU@50": aggregate.get("per_GT_best_IoU_ge_50"),
        "D4RT_hit_rate": "",
        "native_support_4D_ARI": "",
        "native_support_purity": "",
        "native_support_completeness": "",
        "unknown_labeled_tube_ratio": "",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_rgbd_for_prediction": True,
        "uses_pose_for_prediction": True,
        "uses_scannet_mesh_for_prediction": True,
        "uses_eval_sim3_for_prediction": False,
        "geometry_backend": "ScanNet RGB-D/pose/mesh diagnostic bridge",
        "object_algorithm": "v42 semantic-material factor graph + stride-1 streaming memory",
        "materializer": "run_v42_diagnostic_gtgeo_materializer",
        "source_artifact": str(gtgeo_summary.get("memory_object_rows", "")),
        "method_compatibility_manifest": {
            "scene_count": scene_count,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "uses_gt_for_prediction": False,
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "forbidden_for_method_table": True,
            "phase8_gate_pass": gtgeo_summary.get("phase8_gate_pass"),
            "phase8_gate_blocker": gtgeo_summary.get("phase8_gate_blocker"),
        },
        "note": "Diagnostic-only GT/RGB-D bridge; measures v42 object representation under forbidden mesh materialization.",
    }


def _finite_or_blank(value: Any) -> float | str:
    try:
        current = float(value)
    except (TypeError, ValueError):
        return ""
    return current if np.isfinite(current) else ""


def _v42_native_nn_row_from_summary(native_nn_summary: dict[str, Any]) -> dict[str, Any]:
    best = dict(native_nn_summary.get("best_by_AP", {}))
    return {
        "table4_label": "v42 ObjectField + D4RT native xyz NN diagnostic",
        "source_row_id": "V42-O-D4RT-native-NN",
        "status": native_nn_summary.get("status", "unknown"),
        "AP": _finite_or_blank(best.get("AP")),
        "AP50": _finite_or_blank(best.get("AP50")),
        "AP25": _finite_or_blank(best.get("AP25")),
        "predictions_per_scene": best.get("num_predictions", ""),
        "conflict": best.get("mean_export_conflict_rate", ""),
        "multiplicity": "",
        "per_GT_best_IoU@50": best.get("per_GT_best_IoU_ge_50", ""),
        "D4RT_hit_rate": best.get("native_point_hit_rate", ""),
        "native_support_4D_ARI": "",
        "native_support_purity": "",
        "native_support_completeness": "",
        "unknown_labeled_tube_ratio": "",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": True,
        "uses_eval_sim3_for_prediction": False,
        "geometry_backend": "D4RT canonical xyz nearest-neighbor to ScanNet mesh",
        "object_algorithm": "v42 semantic-material factor graph + stride-1 streaming memory",
        "materializer": "run_v42_native_nn_ap_bridge",
        "source_artifact": str(native_nn_summary.get("native_point_rows", "")),
        "method_compatibility_manifest": {
            "is_method_result": False,
            "is_diagnostic_only": True,
            "uses_gt_for_prediction": False,
            "uses_rgbd_for_prediction": False,
            "uses_pose_for_prediction": False,
            "uses_scannet_mesh_for_prediction": True,
            "forbidden_for_method_table": True,
            "phase8_gate_pass": native_nn_summary.get("phase8_gate_pass"),
            "phase8_gate_blocker": native_nn_summary.get("phase8_gate_blocker"),
            "best_radius": best.get("nn_radius"),
            "best_native_point_hit_rate": best.get("native_point_hit_rate"),
        },
        "note": "Native xyz to mesh NN diagnostic produced no valid AP; indicates missing native calibration/alignment to ScanNet mesh.",
    }


def _v42_calibrated_native_row_from_summary(calibrated_summary: dict[str, Any]) -> dict[str, Any]:
    best = dict(calibrated_summary.get("best_by_AP", {}))
    calibration_rows = list(calibrated_summary.get("calibration_rows", []))
    return {
        "table4_label": "v42 ObjectField + RGB-D/pose calibrated D4RT native xyz diagnostic",
        "source_row_id": "V42-O-D4RT-native-Calibrated-diagnostic",
        "status": calibrated_summary.get("status", "unknown"),
        "AP": _finite_or_blank(best.get("AP")),
        "AP50": _finite_or_blank(best.get("AP50")),
        "AP25": _finite_or_blank(best.get("AP25")),
        "predictions_per_scene": best.get("num_predictions", ""),
        "conflict": best.get("mean_export_conflict_rate", ""),
        "multiplicity": "",
        "per_GT_best_IoU@50": best.get("per_GT_best_IoU_ge_50", ""),
        "D4RT_hit_rate": best.get("calibrated_native_hit_rate", ""),
        "native_support_4D_ARI": "",
        "native_support_purity": "",
        "native_support_completeness": "",
        "unknown_labeled_tube_ratio": "",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_rgbd_for_prediction": True,
        "uses_pose_for_prediction": True,
        "uses_scannet_mesh_for_prediction": True,
        "uses_eval_sim3_for_prediction": False,
        "geometry_backend": "D4RT native xyz calibrated to ScanNet mesh with RGB-D/pose Sim3",
        "object_algorithm": "v42 semantic-material factor graph + stride-1 all-frame streaming memory",
        "materializer": "run_v42_calibrated_native_ap_bridge",
        "source_artifact": str(calibrated_summary.get("native_point_rows", "")),
        "method_compatibility_manifest": {
            "is_method_result": False,
            "is_diagnostic_only": True,
            "uses_gt_for_prediction": False,
            "uses_rgbd_for_prediction": True,
            "uses_pose_for_prediction": True,
            "uses_scannet_mesh_for_prediction": True,
            "forbidden_for_method_table": True,
            "phase8_gate_pass": calibrated_summary.get("phase8_gate_pass"),
            "phase8_gate_blocker": calibrated_summary.get("phase8_gate_blocker"),
            "best_radius": best.get("nn_radius"),
            "best_calibrated_native_hit_rate": best.get("calibrated_native_hit_rate"),
            "calibration_scene_rows": calibration_rows,
        },
        "note": "Diagnostic-only calibration row; useful for root cause but forbidden as method AP.",
    }


def _v42_native_projection_row_from_summary(projection_summary: dict[str, Any]) -> dict[str, Any]:
    rows = list(projection_summary.get("rows", []))
    p90_values = [
        _safe_float(row.get("projection_error_p90"))
        for row in rows
        if row.get("projection_error_p90") not in {None, ""}
    ]
    p90_values = [float(value) for value in p90_values if value is not None and np.isfinite(float(value))]
    within_values = [
        _safe_float(row.get("projection_within_0p02"))
        for row in rows
        if row.get("projection_within_0p02") not in {None, ""}
    ]
    within_values = [float(value) for value in within_values if value is not None and np.isfinite(float(value))]
    return {
        "table4_label": "v42 D4RT native projection/materializer readiness audit",
        "source_row_id": "V42-O-D4RT-native-projection-audit",
        "status": projection_summary.get("status", "unknown"),
        "AP": "",
        "AP50": "",
        "AP25": "",
        "predictions_per_scene": "",
        "conflict": "",
        "multiplicity": "",
        "per_GT_best_IoU@50": "",
        "D4RT_hit_rate": "",
        "native_support_4D_ARI": "",
        "native_support_purity": "",
        "native_support_completeness": "",
        "unknown_labeled_tube_ratio": "",
        "native_projection_error_p90_mean": _mean(p90_values),
        "native_projection_within_0p02_mean": _mean(within_values),
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": False,
        "uses_gt_for_prediction": False,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "geometry_backend": "D4RT native xyz/uv projection consistency",
        "object_algorithm": "v42 stride-1 Q5 native cache audit",
        "materializer": "run_v42_native_projection_consistency",
        "source_artifact": str(projection_summary.get("cache_root", "")),
        "method_compatibility_manifest": {
            "method_ap_materializer_ready": projection_summary.get("method_ap_materializer_ready"),
            "projection_all_scenes_gate_pass": projection_summary.get("projection_all_scenes_gate_pass"),
            "uses_gt_for_prediction": False,
            "uses_rgbd_for_prediction": False,
            "uses_pose_for_prediction": False,
            "uses_scannet_mesh_for_prediction": False,
            "forbidden_for_method_table": False,
            "scene_summaries": projection_summary.get("scene_summaries", []),
        },
        "note": "No-RGBD/no-pose audit: native cache lacks scene transform/mesh ids and xyz->uv projection gate fails.",
    }


def _v42_native_tube_ap_row_from_summary(
    native_tube_summary: dict[str, Any],
    *,
    source_row_id: str = "V42-O-D4RT-native-tube-AP",
    table4_label: str = "v42 ObjectField + D4RT native tube-space AP",
    repair_attempt: bool = False,
) -> dict[str, Any]:
    return {
        "table4_label": table4_label,
        "source_row_id": source_row_id,
        "status": native_tube_summary.get("status", "unknown"),
        "AP": "",
        "AP50": "",
        "AP25": "",
        "predictions_per_scene": (_safe_float(native_tube_summary.get("prediction_count")) or 0.0) / max(
            len(native_tube_summary.get("scenes", []) or []),
            1,
        ),
        "conflict": "",
        "multiplicity": "",
        "per_GT_best_IoU@50": native_tube_summary.get("per_gt_best_tube_iou_ge_50", ""),
        "D4RT_hit_rate": native_tube_summary.get("labeled_tube_coverage", ""),
        "native_support_4D_ARI": "",
        "native_support_purity": "",
        "native_support_completeness": "",
        "unknown_labeled_tube_ratio": "",
        "native_tube_AP": native_tube_summary.get("native_tube_AP"),
        "native_tube_AP50": native_tube_summary.get("native_tube_AP50"),
        "native_tube_AP25": native_tube_summary.get("native_tube_AP25"),
        "native_tube_per_gt_best_iou_mean": native_tube_summary.get("per_gt_best_tube_iou_mean"),
        "metric_scope": native_tube_summary.get("metric_scope", "d4rt_native_tube_space"),
        "native_tube_score_mode": native_tube_summary.get("score_mode", ""),
        "native_tube_min_pred_tube_count": native_tube_summary.get("min_pred_tube_count", ""),
        "native_tube_max_pred_tube_count": native_tube_summary.get("max_pred_tube_count", ""),
        "repair_attempt": bool(repair_attempt),
        "is_scannet_ap_result": False,
        "is_method_result": True,
        "is_diagnostic_only": False,
        "forbidden_for_method_table": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_scoring": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "geometry_backend": "D4RT native tube set",
        "object_algorithm": "v42 semantic-material factor graph + stride-1 all-frame streaming memory",
        "materializer": "run_v42_native_tube_ap_metric, not ScanNet mesh masks",
        "source_artifact": str(native_tube_summary.get("memory_object_rows", "")),
        "method_compatibility_manifest": {
            "metric_scope": native_tube_summary.get("metric_scope", "d4rt_native_tube_space"),
            "score_mode": native_tube_summary.get("score_mode"),
            "min_pred_tube_count": native_tube_summary.get("min_pred_tube_count"),
            "max_pred_tube_count": native_tube_summary.get("max_pred_tube_count"),
            "repair_attempt": bool(repair_attempt),
            "is_scannet_ap_result": False,
            "is_method_result": True,
            "uses_gt_for_prediction": False,
            "uses_gt_for_scoring": True,
            "uses_rgbd_for_prediction": False,
            "uses_pose_for_prediction": False,
            "uses_scannet_mesh_for_prediction": False,
            "phase8_gate_pass": native_tube_summary.get("phase8_gate_pass"),
            "phase8_gate_blocker": native_tube_summary.get("phase8_gate_blocker"),
        },
        "note": "Prediction-only native tube-space score repair attempt; not a ScanNet AP result."
        if repair_attempt
        else "Method-compatible native tube-space metric; not a ScanNet AP result.",
    }


def _ap_blocker_checks() -> list[dict[str, Any]]:
    export_scannet = ROOT / "stream4d/export_scannet.py"
    evaluate = ROOT / "evaluation/evaluate.py"
    v37_ap_tool = ROOT / "tools/run_v37_ap_if_allowed.py"
    return [
        {
            "check": "ScanNet evaluator requires mesh-vertex pred_masks",
            "pass": _source_has(evaluate, "wrong number of lines")
            and _source_has(evaluate, "vs #mesh vertices")
            and _source_has(evaluate, "pred_masks"),
            "evidence": str(evaluate.relative_to(ROOT)),
        },
        {
            "check": "existing RGB-D AP bridge is diagnostic-only",
            "pass": _source_has(export_scannet, "Diagnostic-only RGB-D bridge export")
            and _source_has(export_scannet, "\"forbidden_for_method_table\": True"),
            "evidence": str(export_scannet.relative_to(ROOT)),
        },
        {
            "check": "D4RT native AP export remains unimplemented",
            "pass": _source_has(export_scannet, "def export_d4rt_nn")
            and _source_has(export_scannet, "raise NotImplementedError")
            and _source_has(export_scannet, "scene-coordinate calibration path"),
            "evidence": str(export_scannet.relative_to(ROOT)),
        },
        {
            "check": "legacy AP tool uses ScanNetExporter backprojection",
            "pass": _source_has(v37_ap_tool, "ScanNetExporter")
            and _source_has(v37_ap_tool, "export_support_mode=\"mask_backproject\"")
            and _source_has(v37_ap_tool, "uses_rgbd_for_prediction"),
            "evidence": str(v37_ap_tool.relative_to(ROOT)),
        },
    ]


def build_static_bridge(
    *,
    memory_summary_path: Path,
    memory_scene_rows_path: Path,
    reference_table_path: Path,
    gtgeo_summary_path: Path | None = None,
    native_nn_summary_path: Path | None = None,
    calibrated_native_summary_path: Path | None = None,
    native_projection_summary_path: Path | None = None,
    native_tube_ap_summary_path: Path | None = None,
    native_tube_score_repair_summary_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    memory_summary = _read_json(memory_summary_path)
    memory_scene_rows = _read_csv(memory_scene_rows_path)
    reference_rows = _import_reference_rows(_read_csv(reference_table_path))
    blocked_rows = _v42_blocked_rows(memory_summary)
    gtgeo_status = "not_run"
    if gtgeo_summary_path is not None and gtgeo_summary_path.exists():
        gtgeo_summary = _read_json(gtgeo_summary_path)
        if str(gtgeo_summary.get("status", "")).startswith("OK_DIAGNOSTIC_GTGEO_AP"):
            blocked_rows = [
                _v42_gtgeo_row_from_summary(gtgeo_summary)
                if row.get("source_row_id") == "V42-O-GTGeo-diagnostic"
                else row
                for row in blocked_rows
            ]
            gtgeo_status = "ok_diagnostic_gtgeo_not_method"
        else:
            gtgeo_status = str(gtgeo_summary.get("status", "failed_or_inconclusive"))
    rows = reference_rows + [_v42_native_support_row(memory_summary, memory_scene_rows)] + blocked_rows
    native_nn_status = "not_run"
    if native_nn_summary_path is not None and native_nn_summary_path.exists():
        native_nn_summary = _read_json(native_nn_summary_path)
        native_nn_status = str(native_nn_summary.get("status", "unknown"))
        rows.append(_v42_native_nn_row_from_summary(native_nn_summary))
    calibrated_native_status = "not_run"
    if calibrated_native_summary_path is not None and calibrated_native_summary_path.exists():
        calibrated_native_summary = _read_json(calibrated_native_summary_path)
        calibrated_native_status = str(calibrated_native_summary.get("status", "unknown"))
        rows.append(_v42_calibrated_native_row_from_summary(calibrated_native_summary))
    native_projection_status = "not_run"
    if native_projection_summary_path is not None and native_projection_summary_path.exists():
        native_projection_summary = _read_json(native_projection_summary_path)
        native_projection_status = str(native_projection_summary.get("status", "unknown"))
        rows.append(_v42_native_projection_row_from_summary(native_projection_summary))
    native_tube_ap_status = "not_run"
    native_tube_ap = None
    native_tube_ap50 = None
    native_tube_ap25 = None
    if native_tube_ap_summary_path is not None and native_tube_ap_summary_path.exists():
        native_tube_summary = _read_json(native_tube_ap_summary_path)
        native_tube_ap_status = str(native_tube_summary.get("status", "unknown"))
        native_tube_ap = native_tube_summary.get("native_tube_AP")
        native_tube_ap50 = native_tube_summary.get("native_tube_AP50")
        native_tube_ap25 = native_tube_summary.get("native_tube_AP25")
        rows.append(_v42_native_tube_ap_row_from_summary(native_tube_summary))
    native_tube_score_repair_status = "not_run"
    native_tube_score_repair_ap = None
    native_tube_score_repair_ap50 = None
    native_tube_score_repair_ap25 = None
    if native_tube_score_repair_summary_path is not None and native_tube_score_repair_summary_path.exists():
        native_tube_score_summary = _read_json(native_tube_score_repair_summary_path)
        native_tube_score_repair_status = str(native_tube_score_summary.get("status", "unknown"))
        native_tube_score_repair_ap = native_tube_score_summary.get("native_tube_AP")
        native_tube_score_repair_ap50 = native_tube_score_summary.get("native_tube_AP50")
        native_tube_score_repair_ap25 = native_tube_score_summary.get("native_tube_AP25")
        rows.append(
            _v42_native_tube_ap_row_from_summary(
                native_tube_score_summary,
                source_row_id="V42-O-D4RT-native-tube-AP-score-repair",
                table4_label="v42 ObjectField + D4RT native tube-space AP score repair",
                repair_attempt=True,
            )
        )
    checks = _ap_blocker_checks()
    ap_bridge_blocked = all(bool(row["pass"]) for row in checks)
    native_support_improved = bool(memory_summary.get("native_support_metric_proxy_pass") is True)
    summary = {
        "phase": "v42_static_bridge_root_cause_diagnostic",
        "memory_summary": str(memory_summary_path),
        "memory_scene_rows": str(memory_scene_rows_path),
        "reference_table": str(reference_table_path),
        "gtgeo_summary": str(gtgeo_summary_path) if gtgeo_summary_path is not None else None,
        "native_nn_summary": str(native_nn_summary_path) if native_nn_summary_path is not None else None,
        "calibrated_native_summary": str(calibrated_native_summary_path) if calibrated_native_summary_path is not None else None,
        "native_projection_summary": str(native_projection_summary_path) if native_projection_summary_path is not None else None,
        "native_tube_ap_summary": str(native_tube_ap_summary_path) if native_tube_ap_summary_path is not None else None,
        "native_tube_score_repair_summary": str(native_tube_score_repair_summary_path)
        if native_tube_score_repair_summary_path is not None
        else None,
        "row_count": int(len(rows)),
        "v42_native_support_metric_proxy_pass": native_support_improved,
        "v42_native_support_4D_ARI": memory_summary.get("aggregate_tube_4D_ARI"),
        "v42_native_support_purity": memory_summary.get("aggregate_tube_purity"),
        "v42_native_support_completeness": memory_summary.get("aggregate_tube_completeness"),
        "v42_phase8_gate_pass": bool(memory_summary.get("phase8_gate_pass")),
        "method_ap_goal_reached": False,
        "v42_gtgeo_diagnostic_status": gtgeo_status,
        "v42_native_nn_status": native_nn_status,
        "v42_calibrated_native_status": calibrated_native_status,
        "v42_native_projection_status": native_projection_status,
        "v42_native_tube_ap_status": native_tube_ap_status,
        "v42_native_tube_AP": native_tube_ap,
        "v42_native_tube_AP50": native_tube_ap50,
        "v42_native_tube_AP25": native_tube_ap25,
        "v42_native_tube_score_repair_status": native_tube_score_repair_status,
        "v42_native_tube_score_repair_AP": native_tube_score_repair_ap,
        "v42_native_tube_score_repair_AP50": native_tube_score_repair_ap50,
        "v42_native_tube_score_repair_AP25": native_tube_score_repair_ap25,
        "ap_bridge_blocked_by_missing_native_exporter": bool(ap_bridge_blocked),
        "status": "PARTIAL_NATIVE_SUPPORT_IMPROVED_AP_BRIDGE_BLOCKED"
        if native_support_improved and ap_bridge_blocked
        else "STATIC_BRIDGE_DIAGNOSTIC_INCONCLUSIVE",
        "interpretation": [
            "v42 native support improved under stride-1 memory.",
            "Native tube-space AP can be reported only as a D4RT-native metric, not ScanNet AP.",
            "Existing AP materialization routes remain diagnostic-only or unimplemented for native D4RT support.",
        ],
        "checks": checks,
        "uses_gt_for_prediction": False,
        "uses_gt_for_scoring": True,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v42 static bridge/root-cause diagnostic table.")
    parser.add_argument(
        "--memory-summary",
        default="outputs/audit/v42_streaming_memory_unioncap320_allframe_r1/memory_summary.json",
    )
    parser.add_argument(
        "--memory-scene-rows",
        default="outputs/audit/v42_streaming_memory_unioncap320_allframe_r1/memory_scene_rows.csv",
    )
    parser.add_argument(
        "--reference-table",
        default="outputs/audit/v41_1_stream3d_first_comparison/table4_static_bridge_stream3d_first.csv",
    )
    parser.add_argument(
        "--gtgeo-summary",
        default="outputs/audit/v42_gtgeo_materializer_allframe_r1/gtgeo_materializer_summary.json",
    )
    parser.add_argument(
        "--native-nn-summary",
        default="outputs/audit/v42_native_nn_ap_bridge_allframe_r1/native_nn_ap_bridge_summary.json",
    )
    parser.add_argument(
        "--calibrated-native-summary",
        default="outputs/audit/v42_calibrated_native_ap_bridge_allframe_r1/calibrated_native_ap_bridge_summary.json",
    )
    parser.add_argument(
        "--native-projection-summary",
        default="outputs/audit/v42_native_projection_consistency_allframe_r1/native_projection_consistency_summary.json",
    )
    parser.add_argument(
        "--native-tube-ap-summary",
        default="outputs/audit/v42_native_tube_ap_metric_allframe_r1/native_tube_ap_summary.json",
    )
    parser.add_argument(
        "--native-tube-score-repair-summary",
        default="outputs/audit/v42_native_tube_ap_metric_allframe_score_logtube_r1/native_tube_ap_summary.json",
    )
    parser.add_argument("--output-root", default="outputs/audit/v42_static_bridge_allframe_r1")
    args = parser.parse_args()

    output_root = ROOT / str(args.output_root)
    rows, summary = build_static_bridge(
        memory_summary_path=ROOT / str(args.memory_summary),
        memory_scene_rows_path=ROOT / str(args.memory_scene_rows),
        reference_table_path=ROOT / str(args.reference_table),
        gtgeo_summary_path=ROOT / str(args.gtgeo_summary) if str(args.gtgeo_summary).strip() else None,
        native_nn_summary_path=ROOT / str(args.native_nn_summary) if str(args.native_nn_summary).strip() else None,
        calibrated_native_summary_path=ROOT / str(args.calibrated_native_summary)
        if str(args.calibrated_native_summary).strip()
        else None,
        native_projection_summary_path=ROOT / str(args.native_projection_summary)
        if str(args.native_projection_summary).strip()
        else None,
        native_tube_ap_summary_path=ROOT / str(args.native_tube_ap_summary)
        if str(args.native_tube_ap_summary).strip()
        else None,
        native_tube_score_repair_summary_path=ROOT / str(args.native_tube_score_repair_summary)
        if str(args.native_tube_score_repair_summary).strip()
        else None,
    )
    _write_csv(output_root / "static_bridge_summary.csv", rows)
    _write_json(output_root / "static_bridge_summary.json", summary)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "row_count": summary["row_count"],
                "status": summary["status"],
                "method_ap_goal_reached": summary["method_ap_goal_reached"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
