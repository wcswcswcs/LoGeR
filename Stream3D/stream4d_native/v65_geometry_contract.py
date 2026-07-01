from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .v65_common import (
    float_or_none,
    load_dict,
    load_rows,
    mean_by_key,
    project,
    rel,
    sha256_file,
    write_standard_outputs,
)


GEOM_ROOT = "outputs/audit/v65_geometry_contract"

SOURCE_PATHS = {
    "d4rt_geometry_rows": "outputs/audit/v64r2_d4rt_chunk_scale_first_ap_probe5/D4RT_geometry_replacement_stream3d_probe5.csv",
    "chunk_scale_rows": "outputs/audit/v64r2_chunk_scale_diagnostic_probe5_recheck/chunk_scale_ratio_rows.csv",
    "chunk_scale_summary": "outputs/audit/v64r2_chunk_scale_diagnostic_probe5_recheck/chunk_scale_diagnostic_summary.json",
    "scene_rows": "outputs/audit/v64r2_scene_level_d4rt_geometry_probe5/scene_level_d4rt_geometry_scene_rows.csv",
    "scene_summary": "outputs/audit/v64r2_scene_level_d4rt_geometry_probe5/scene_level_d4rt_geometry_summary.json",
    "ate_rows": "outputs/audit/v64r2_d4rt_ate_probe5/d4rt_ate_scene_summary.csv",
    "ate_summary": "outputs/audit/v64r2_d4rt_ate_probe5/d4rt_ate_summary.json",
    "fragmentation_rows": "outputs/audit/v65_ap_failure_decomp/fragmentation_rows.csv",
    "ap_contract_rows": "outputs/audit/v65_ap_contract/ap_contract_rows.csv",
    "failure_summary": "outputs/audit/v65_ap_failure_decomp/failure_summary.json",
}


def build_v65_geometry_contract() -> dict[str, Any]:
    d4rt_rows = load_rows(SOURCE_PATHS["d4rt_geometry_rows"])
    chunk_rows = load_rows(SOURCE_PATHS["chunk_scale_rows"])
    scene_rows = load_rows(SOURCE_PATHS["scene_rows"])
    ate_rows = load_rows(SOURCE_PATHS["ate_rows"])
    frag_rows = load_rows(SOURCE_PATHS["fragmentation_rows"])
    ap_rows = load_rows(SOURCE_PATHS["ap_contract_rows"])
    metric_rows: list[dict[str, Any]] = []
    metric_rows.extend(_g0_image_space_rows(d4rt_rows))
    metric_rows.extend(_g1_single_window_rows(d4rt_rows))
    metric_rows.extend(_g2_chunk_metric_rows(chunk_rows))
    metric_rows.extend(_g3_scene_rows(scene_rows, ate_rows))
    object_rows = _g4_object_rows(frag_rows)
    metric_rows.extend(object_rows)
    metric_rows.extend(_g5_materialization_rows(ap_rows, frag_rows))
    normalized_chunk_rows = _chunk_scale_rows(chunk_rows)
    sources = _source_rows()
    levels_present = sorted({row["metric_level"] for row in metric_rows})
    required_levels = [f"G{i}" for i in range(6)]
    missing_levels = [level for level in required_levels if level not in levels_present]
    by_level: dict[str, dict[str, Any]] = {}
    for level in required_levels:
        rows = [row for row in metric_rows if row["metric_level"] == level]
        pass_values = [row["pass_gate"] for row in rows if row.get("pass_gate") != ""]
        by_level[level] = {
            "row_count": len(rows),
            "pass_count": sum(1 for value in pass_values if str(value) == "True"),
            "fail_count": sum(1 for value in pass_values if str(value) == "False"),
            "diagnostic_row_count": sum(1 for row in rows if str(row.get("is_diagnostic_metric")) == "True"),
            "method_safe_row_count": sum(1 for row in rows if str(row.get("is_method_safe_metric")) == "True"),
        }
    chunk_pair_count = len(normalized_chunk_rows)
    chunk_within_10 = sum(1 for row in normalized_chunk_rows if row.get("pass_v65_10pct_gate") is True)
    summary = {
        "phase": "v65_geometry_contract",
        "source_files": sources,
        "geometry_metric_row_count": len(metric_rows),
        "chunk_scale_row_count": len(normalized_chunk_rows),
        "object_geometry_row_count": len(object_rows),
        "levels_present": levels_present,
        "missing_levels": missing_levels,
        "rows_by_level": by_level,
        "chunk_scale": {
            "adjacent_pair_count": chunk_pair_count,
            "within_10pct_count": chunk_within_10,
            "within_10pct_rate": (chunk_within_10 / chunk_pair_count) if chunk_pair_count else None,
            "all_adjacent_within_10pct": chunk_pair_count > 0 and chunk_within_10 == chunk_pair_count,
        },
        "scene_geometry_status": _scene_status(scene_rows, ate_rows),
        "object_geometry_status": _object_status(object_rows),
        "contract_note": (
            "v65 geometry contract separates image/window/chunk/scene/object/materialization levels; "
            "rows using ScanNet mesh/depth/pose/GT/eval-Sim3 are diagnostic-only."
        ),
        "gate": {
            "all_geometry_rows_have_metric_level": all(bool(row.get("metric_level")) for row in metric_rows),
            "G0_to_G5_present": not missing_levels,
            "G2_chunk_scale_reported": chunk_pair_count > 0,
            "G4_object_support_reported": len(object_rows) > 0,
            "diagnostic_metrics_separated": all(
                row.get("is_diagnostic_metric") in {True, False, "True", "False"} for row in metric_rows
            ),
            "chunk_level_not_promoted_to_scene_claim": True,
        },
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    summary["geometry_status"] = (
        "GO_GEOMETRY_CONTRACT_LOCKED" if summary["gate"]["pass"] else "NO_GO_GEOMETRY_CONTRACT"
    )
    return {
        "summary": summary,
        "geometry_metric_rows": metric_rows,
        "chunk_scale_rows": normalized_chunk_rows,
        "object_geometry_rows": object_rows,
        "source_artifact_rows": sources,
    }


def write_v65_geometry_contract(output_root: str | Path, payload: dict[str, Any]) -> None:
    write_standard_outputs(
        output_root,
        {
            "geometry_contract_summary.json": payload["summary"],
            "geometry_metric_rows.csv": payload["geometry_metric_rows"],
            "chunk_scale_rows.csv": payload["chunk_scale_rows"],
            "object_geometry_rows.csv": payload["object_geometry_rows"],
            "source_artifact_rows.csv": payload["source_artifact_rows"],
        },
    )


def _base_row(
    *,
    metric_level: str,
    metric_family: str,
    metric_name: str,
    value: float | None,
    source: str,
    scene_id: str = "",
    window_id: str = "",
    chunk_id: str = "",
    history_id: str = "",
    variant: str = "",
    threshold: str = "",
    pass_gate: bool | str = "",
    uses_gt_for_metric: bool = False,
    uses_rgbd_pose_mesh_for_metric: bool = False,
    is_method_safe_metric: bool = False,
    is_diagnostic_metric: bool = True,
    coordinate_frame: str = "",
    alignment_type: str = "",
    alignment_source: str = "",
    scale_source: str = "",
    support_scope: str = "",
    note: str = "",
) -> dict[str, Any]:
    return {
        "metric_level": metric_level,
        "metric_family": metric_family,
        "metric_name": metric_name,
        "value": value if value is not None else "",
        "threshold": threshold,
        "pass_gate": pass_gate,
        "scene_id": scene_id,
        "window_id": window_id,
        "chunk_id": chunk_id,
        "history_id": history_id,
        "variant": variant,
        "source_artifact": source,
        "source_artifact_hash": sha256_file(source),
        "uses_gt_for_metric": uses_gt_for_metric,
        "uses_rgbd_pose_mesh_for_metric": uses_rgbd_pose_mesh_for_metric,
        "is_method_safe_metric": is_method_safe_metric,
        "is_diagnostic_metric": is_diagnostic_metric,
        "coordinate_frame": coordinate_frame,
        "alignment_type": alignment_type,
        "alignment_source": alignment_source,
        "scale_source": scale_source,
        "support_scope": support_scope,
        "note": note,
    }


def _g0_image_space_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = SOURCE_PATHS["d4rt_geometry_rows"]
    metrics = {
        "self_stitch_mutual_uv_match_ratio_mean_mean": "mutual UV match ratio",
        "self_stitch_cycle_consistency_pass_ratio_mean_mean": "cycle consistency pass ratio",
        "self_stitch_same_source_pixel_match_ratio_mean_mean": "same source pixel match ratio",
        "self_stitch_stable_id_match_ratio_mean_mean": "stable id match ratio",
    }
    out: list[dict[str, Any]] = []
    for key, name in metrics.items():
        out.append(
            _base_row(
                metric_level="G0",
                metric_family="image_space_d4rt_correspondence",
                metric_name=key,
                value=mean_by_key(rows, key),
                source=source,
                variant="D4RT_G11_G12_probe5_mean",
                pass_gate="",
                uses_gt_for_metric=False,
                uses_rgbd_pose_mesh_for_metric=False,
                is_method_safe_metric=True,
                is_diagnostic_metric=True,
                coordinate_frame="image_uv",
                alignment_type="none",
                alignment_source="D4RT self-stitch diagnostics",
                scale_source="none",
                note=name,
            )
        )
    return out


def _g1_single_window_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = SOURCE_PATHS["d4rt_geometry_rows"]
    return [
        _base_row(
            metric_level="G1",
            metric_family="single_window_metric_geometry",
            metric_name="sim3_residual_median_mean",
            value=mean_by_key(rows, "sim3_residual_median_mean"),
            source=source,
            variant="D4RT_G11_G12_probe5_mean",
            threshold="<=0.05m diagnostic target",
            pass_gate=_le(mean_by_key(rows, "sim3_residual_median_mean"), 0.05),
            uses_gt_for_metric=False,
            uses_rgbd_pose_mesh_for_metric=False,
            is_method_safe_metric=True,
            is_diagnostic_metric=True,
            coordinate_frame="local_metric",
            alignment_type="D4RT self-Sim3",
            alignment_source="D4RT provider diagnostics",
            scale_source="self",
        ),
        _base_row(
            metric_level="G1",
            metric_family="single_window_metric_geometry",
            metric_name="sim3_residual_p90_mean",
            value=mean_by_key(rows, "sim3_residual_p90_mean"),
            source=source,
            variant="D4RT_G11_G12_probe5_mean",
            threshold="<=0.15m diagnostic target",
            pass_gate=_le(mean_by_key(rows, "sim3_residual_p90_mean"), 0.15),
            uses_gt_for_metric=False,
            uses_rgbd_pose_mesh_for_metric=False,
            is_method_safe_metric=True,
            is_diagnostic_metric=True,
            coordinate_frame="local_metric",
            alignment_type="D4RT self-Sim3",
            alignment_source="D4RT provider diagnostics",
            scale_source="self",
        ),
    ]


def _g2_chunk_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = SOURCE_PATHS["chunk_scale_rows"]
    out: list[dict[str, Any]] = []
    for row in rows:
        value = float_or_none(row.get("scale_next_over_prev"))
        abs_log = float_or_none(row.get("abs_log_scale_ratio"))
        out.append(
            _base_row(
                metric_level="G2",
                metric_family="chunk_stitch_scale_drift",
                metric_name="scale_next_over_prev",
                value=value,
                source=source,
                scene_id=str(row.get("scene") or ""),
                chunk_id=str(row.get("window_pair") or ""),
                threshold="abs(log(scale_next_over_prev))<=log(1.10)",
                pass_gate=_le(abs_log, math.log(1.10)),
                uses_gt_for_metric=False,
                uses_rgbd_pose_mesh_for_metric=False,
                is_method_safe_metric=True,
                is_diagnostic_metric=True,
                coordinate_frame="chunk_local_metric",
                alignment_type="self-stitch adjacent chunk scale ratio",
                alignment_source="D4RT self-stitch overlap",
                scale_source="D4RT self-Sim3",
            )
        )
    return out


def _g3_scene_rows(scene_rows: list[dict[str, Any]], ate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scene_source = SOURCE_PATHS["scene_rows"]
    ate_source = SOURCE_PATHS["ate_rows"]
    out: list[dict[str, Any]] = []
    for row in scene_rows:
        scene = str(row.get("scene") or "")
        variant = str(row.get("variant") or "")
        for metric, threshold in [
            ("scene_fit_sim3_residual_median", "<=0.05m diagnostic target"),
            ("scene_fit_sim3_residual_p90", "<=0.15m diagnostic target"),
            ("used_frame_support_chamfer_l1", ""),
            ("used_frame_support_fscore@10cm", ""),
            ("used_frame_support_fscore@20cm", ""),
            ("full_mesh_chamfer_l1", ""),
            ("full_mesh_fscore@10cm", ""),
            ("full_mesh_fscore@20cm", ""),
        ]:
            value = float_or_none(row.get(metric))
            pass_gate: bool | str = ""
            if metric == "scene_fit_sim3_residual_median":
                pass_gate = _le(value, 0.05)
            elif metric == "scene_fit_sim3_residual_p90":
                pass_gate = _le(value, 0.15)
            out.append(
                _base_row(
                    metric_level="G3",
                    metric_family="scene_level_alignment",
                    metric_name=metric,
                    value=value,
                    source=scene_source,
                    scene_id=scene,
                    variant=variant,
                    threshold=threshold,
                    pass_gate=pass_gate,
                    uses_gt_for_metric=True,
                    uses_rgbd_pose_mesh_for_metric=True,
                    is_method_safe_metric=False,
                    is_diagnostic_metric=True,
                    coordinate_frame="ScanNet mesh / used-frame visible support",
                    alignment_type=str(row.get("provider_mode") or "eval_sim3"),
                    alignment_source="ScanNet mesh/depth/pose diagnostic alignment",
                    scale_source=str(row.get("chunk_scale_policy") or ""),
                    support_scope="FULLMESH_OR_USED_FRAME_VISIBLE_SUPPORT",
                    note="Scene-level diagnostic; not a method-safe geometry claim.",
                )
            )
    for row in ate_rows:
        scene = str(row.get("scene") or "")
        for metric in ["ate_sim3_rmse_m", "ate_sim3_median_m", "ate_sim3_p90_m"]:
            out.append(
                _base_row(
                    metric_level="G3",
                    metric_family="scene_level_trajectory_alignment",
                    metric_name=metric,
                    value=float_or_none(row.get(metric)),
                    source=ate_source,
                    scene_id=scene,
                    threshold="diagnostic only",
                    pass_gate="",
                    uses_gt_for_metric=True,
                    uses_rgbd_pose_mesh_for_metric=True,
                    is_method_safe_metric=False,
                    is_diagnostic_metric=True,
                    coordinate_frame="ScanNet camera trajectory",
                    alignment_type="diagnostic Sim3 aligned ATE",
                    alignment_source="ScanNet pose centers",
                    scale_source="eval Sim3",
                    note="Uses ScanNet pose for evaluation; forbidden as method prediction evidence.",
                )
            )
    return out


def _g4_object_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = SOURCE_PATHS["fragmentation_rows"]
    out: list[dict[str, Any]] = []
    for row in rows:
        scene = str(row.get("scene_id") or "")
        variant = str(row.get("variant_row_id") or "")
        for metric in [
            "raw_pred_count",
            "kept_pred_count",
            "dropped_pred_lt100",
            "tiny_fragment_ratio",
            "pred_best_iou_median",
            "gt_best_iou_median",
            "gt_best_iou_ge_050_mean",
            "fragment_count_per_history",
        ]:
            out.append(
                _base_row(
                    metric_level="G4",
                    metric_family="object_support_geometry",
                    metric_name=metric,
                    value=float_or_none(row.get(metric)),
                    source=source,
                    scene_id=scene,
                    variant=variant,
                    history_id=str(row.get("history_id") or ""),
                    uses_gt_for_metric=metric in {"pred_best_iou_median", "gt_best_iou_median", "gt_best_iou_ge_050_mean"},
                    uses_rgbd_pose_mesh_for_metric=False,
                    is_method_safe_metric=metric
                    not in {"pred_best_iou_median", "gt_best_iou_median", "gt_best_iou_ge_050_mean"},
                    is_diagnostic_metric=True,
                    coordinate_frame="AP support points",
                    alignment_type="prediction-to-GT support IoU diagnostic",
                    alignment_source="v65 failure decomposition",
                    support_scope=str(row.get("support_scope") or ""),
                )
            )
    return out


def _g5_materialization_rows(ap_rows: list[dict[str, Any]], frag_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = SOURCE_PATHS["ap_contract_rows"]
    out: list[dict[str, Any]] = []
    frag_by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in frag_rows:
        frag_by_variant.setdefault(str(row.get("variant_row_id") or ""), []).append(row)
    for row in ap_rows:
        row_id = str(row.get("row_id") or "")
        if row_id not in {"A3", "A4", "A5", "A6", "A7", "A8", "A9"}:
            continue
        dropped_mean = mean_by_key(frag_by_variant.get(row_id, []), "dropped_pred_lt100")
        raw_mean = mean_by_key(frag_by_variant.get(row_id, []), "raw_pred_count")
        min_region_drop_rate = (dropped_mean / raw_mean) if raw_mean not in (None, 0) and dropped_mean is not None else None
        for metric, value in [
            ("AP", float_or_none(row.get("AP"))),
            ("AP50", float_or_none(row.get("AP50"))),
            ("AP25", float_or_none(row.get("AP25"))),
            ("prediction_union_ratio", float_or_none(row.get("prediction_union_ratio"))),
            ("prediction_union_inside_support_ratio", float_or_none(row.get("prediction_union_inside_support_ratio"))),
            ("min_region_drop_rate", min_region_drop_rate),
        ]:
            out.append(
                _base_row(
                    metric_level="G5",
                    metric_family="ap_materialization_geometry",
                    metric_name=metric,
                    value=value,
                    source=source,
                    variant=row_id,
                    support_scope=str(row.get("support_scope") or ""),
                    uses_gt_for_metric=metric in {"AP", "AP50", "AP25"},
                    uses_rgbd_pose_mesh_for_metric=str(row.get("uses_rgbd_pose_mesh_for_export")) == "True",
                    is_method_safe_metric=str(row.get("forbidden_for_method_table")) != "True",
                    is_diagnostic_metric=True,
                    coordinate_frame="AP evaluator support",
                    alignment_type="materialized prediction masks to evaluator support",
                    alignment_source=str(row.get("support_owner") or ""),
                    scale_source="AP support policy",
                    note=str(row.get("note") or ""),
                )
            )
    return out


def _chunk_scale_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source = SOURCE_PATHS["chunk_scale_rows"]
    out: list[dict[str, Any]] = []
    for row in rows:
        abs_log = float_or_none(row.get("abs_log_scale_ratio"))
        out.append(
            {
                "scene_id": row.get("scene") or "",
                "window_pair": row.get("window_pair") or "",
                "scale_prev": row.get("scale_prev") or "",
                "scale_next": row.get("scale_next") or "",
                "scale_next_over_prev": row.get("scale_next_over_prev") or "",
                "abs_log_scale_ratio": row.get("abs_log_scale_ratio") or "",
                "scale_aligned_within_5pct": row.get("scale_aligned_within_5pct") or "",
                "scale_aligned_within_10pct": row.get("scale_aligned_within_10pct") or "",
                "pass_v65_10pct_gate": _le(abs_log, math.log(1.10)),
                "source_artifact": source,
                "source_artifact_hash": sha256_file(source),
            }
        )
    return out


def _source_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, path in SOURCE_PATHS.items():
        path_obj = project(path)
        rows.append(
            {
                "source_label": label,
                "source_artifact": rel(path),
                "exists": path_obj.exists(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _scene_status(scene_rows: list[dict[str, Any]], ate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    used_f10 = mean_by_key(scene_rows, "used_frame_support_fscore@10cm")
    full_f10 = mean_by_key(scene_rows, "full_mesh_fscore@10cm")
    ate_rmse = mean_by_key(ate_rows, "ate_sim3_rmse_m")
    sim3_med = mean_by_key(scene_rows, "scene_fit_sim3_residual_median")
    sim3_p90 = mean_by_key(scene_rows, "scene_fit_sim3_residual_p90")
    return {
        "scene_row_count": len(scene_rows),
        "ate_scene_count": len(ate_rows),
        "scene_fit_sim3_residual_median_mean": sim3_med,
        "scene_fit_sim3_residual_p90_mean": sim3_p90,
        "used_frame_support_fscore_10cm_mean": used_f10,
        "full_mesh_fscore_10cm_mean": full_f10,
        "ate_sim3_rmse_m_mean": ate_rmse,
        "claim_status": "diagnostic_only_uses_scannet_mesh_pose_eval_sim3",
    }


def _object_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tiny_rows = [row for row in rows if row.get("metric_name") == "tiny_fragment_ratio"]
    pred_iou = [row for row in rows if row.get("metric_name") == "pred_best_iou_median"]
    gt_iou = [row for row in rows if row.get("metric_name") == "gt_best_iou_ge_050_mean"]
    return {
        "object_metric_row_count": len(rows),
        "tiny_fragment_ratio_mean": _mean_values(tiny_rows),
        "pred_best_iou_median_mean": _mean_values(pred_iou),
        "gt_best_iou_ge_050_mean": _mean_values(gt_iou),
        "claim_status": "object_support_diagnostic_available_not_method_AP",
    }


def _mean_values(rows: list[dict[str, Any]]) -> float | None:
    vals = [float_or_none(row.get("value")) for row in rows]
    vals = [val for val in vals if val is not None]
    return sum(vals) / len(vals) if vals else None


def _le(value: float | None, threshold: float) -> bool | str:
    if value is None:
        return ""
    return value <= threshold
