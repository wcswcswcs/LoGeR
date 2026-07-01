from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from typing import Any

import numpy as np

from .v65_common import (
    bool_value,
    float_or_none,
    load_dict,
    parse_eval_metric_file,
    project,
    rel,
    sha256_file,
    support_stats_from_summary,
    write_standard_outputs,
)
from .v65_a9_method_safe_materializer import A9_ROOT
from .soma_inference_policy import (
    gt_geometry_inference_reasons,
    normalize_reportability,
    policy_violation_reasons,
)

PROBE5_SPLIT = "splits/scannet_v6_probe5.txt"
FULLMESH_TMP_CONFIG = "v65_fullmesh_probe5"

DIRECT_SOMA_ROWS = [
    {
        "row_id": "A3",
        "method_name": "SOMA v64r2 bridge",
        "variant": "v65_A3_soma_v64r2_bridge_prediction_union",
        "source_config": "v64r2_probe5_v53_bridge_wta",
        "tmp_config": "v64r2_probe5_v53_bridge_wta",
        "support_scope": "PREDICTION_UNION_ISLAND",
        "support_policy": "v53 bridge prediction-union pre_points",
        "support_owner": "SOMA v64r2 bridge prediction union",
        "input_frame_policy": "v53_bridge_objectlets_mixed_or_dense_frames",
        "input_frame_policy_source": "outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000/objectlet_rows.csv",
        "stream3d_stride10_aligned": False,
        "frame_policy_comparison_note": "v53 bridge target_frame_ids include non-stride-10 frames; not fair-comparable to Stream3D stride-10 input.",
    },
    {
        "row_id": "A4",
        "method_name": "SOMA v64r2 bridge same masks",
        "variant": "v65_A4_soma_v64r2_bridge_used_frame_support",
        "source_config": "v64r2_probe5_v53_bridge_wta_used_support",
        "tmp_config": "v64r2_probe5_v53_bridge_wta_used_support",
        "support_scope": "USED_FRAME_VISIBLE_SUPPORT",
        "support_policy": "used_frame_depth_pose_visible_mask_support",
        "support_owner": "ScanNet RGB-D/pose used-frame support diagnostic",
        "support_summary_path": "outputs/audit/v64r2_bridge_wta_used_frame_support_check/v64r2_probe5_v53_bridge_wta_used_support_used_frame_support_summary.json",
        "input_frame_policy": "v53_bridge_objectlets_mixed_or_dense_frames",
        "input_frame_policy_source": "outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000/objectlet_rows.csv",
        "stream3d_stride10_aligned": False,
        "frame_policy_comparison_note": "used-frame support is derived from v53 bridge objectlets whose target_frame_ids are not stride-10 aligned.",
    },
    {
        "row_id": "A5",
        "method_name": "SOMA D4RT G11",
        "variant": "v65_A5_soma_d4rt_g11_prediction_union",
        "source_config": "v64r2_d4rt_chunk_scale_first_ap_probe5_g11",
        "tmp_config": "v64r2_d4rt_chunk_scale_first_ap_probe5_g11",
        "support_scope": "PREDICTION_UNION_ISLAND",
        "support_policy": "D4RT G11 prediction-union pre_points",
        "support_owner": "SOMA D4RT G11 prediction union",
        "input_frame_policy": "D4RT debug frames stride-10, window_size=32, window_stride=16",
        "input_frame_policy_source": "outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1/*/carriers_window*_manifest.json",
        "stream3d_stride10_aligned": True,
        "frame_policy_comparison_note": "D4RT carrier manifests list frame_ids [0,10,20,...]; input cadence matches Stream3D stride-10, but support remains SOMA-owned diagnostic.",
    },
    {
        "row_id": "A6",
        "method_name": "SOMA D4RT G11",
        "variant": "v65_A6_soma_d4rt_g11_used_frame_support",
        "source_config": "v64r2_d4rt_chunk_scale_first_ap_probe5_g11_used_support",
        "tmp_config": "v64r2_d4rt_chunk_scale_first_ap_probe5_g11_used_support",
        "support_scope": "USED_FRAME_VISIBLE_SUPPORT",
        "support_policy": "used_frame_depth_pose_visible_mask_support",
        "support_owner": "ScanNet RGB-D/pose used-frame support diagnostic",
        "support_summary_path": "outputs/audit/v64r2_used_frame_support_ap_probe5/v64r2_d4rt_chunk_scale_first_ap_probe5_g11_used_support_used_frame_support_summary.json",
        "input_frame_policy": "D4RT debug frames stride-10, window_size=32, window_stride=16",
        "input_frame_policy_source": "outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1/*/carriers_window*_manifest.json",
        "stream3d_stride10_aligned": True,
        "frame_policy_comparison_note": "D4RT carrier manifests list frame_ids [0,10,20,...]; input cadence matches Stream3D stride-10, support uses RGB-D/pose diagnostic visible set.",
    },
    {
        "row_id": "A7",
        "method_name": "SOMA D4RT G12",
        "variant": "v65_A7_soma_d4rt_g12_prediction_union",
        "source_config": "v64r2_d4rt_chunk_scale_first_ap_probe5_g12",
        "tmp_config": "v64r2_d4rt_chunk_scale_first_ap_probe5_g12",
        "support_scope": "PREDICTION_UNION_ISLAND",
        "support_policy": "D4RT G12 prediction-union pre_points",
        "support_owner": "SOMA D4RT G12 prediction union",
        "input_frame_policy": "D4RT debug frames stride-10, window_size=32, window_stride=16",
        "input_frame_policy_source": "outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1/*/carriers_window*_manifest.json",
        "stream3d_stride10_aligned": True,
        "frame_policy_comparison_note": "D4RT carrier manifests list frame_ids [0,10,20,...]; input cadence matches Stream3D stride-10, but support remains SOMA-owned diagnostic.",
    },
    {
        "row_id": "A8",
        "method_name": "SOMA D4RT G12",
        "variant": "v65_A8_soma_d4rt_g12_used_frame_support",
        "source_config": "v64r2_d4rt_chunk_scale_first_ap_probe5_g12_used_support",
        "tmp_config": "v64r2_d4rt_chunk_scale_first_ap_probe5_g12_used_support",
        "support_scope": "USED_FRAME_VISIBLE_SUPPORT",
        "support_policy": "used_frame_depth_pose_visible_mask_support",
        "support_owner": "ScanNet RGB-D/pose used-frame support diagnostic",
        "support_summary_path": "outputs/audit/v64r2_used_frame_support_ap_probe5/v64r2_d4rt_chunk_scale_first_ap_probe5_g12_used_support_used_frame_support_summary.json",
        "input_frame_policy": "D4RT debug frames stride-10, window_size=32, window_stride=16",
        "input_frame_policy_source": "outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1/*/carriers_window*_manifest.json",
        "stream3d_stride10_aligned": True,
        "frame_policy_comparison_note": "D4RT carrier manifests list frame_ids [0,10,20,...]; input cadence matches Stream3D stride-10, support uses RGB-D/pose diagnostic visible set.",
    },
]

STREAM3D_ROWS = [
    {
        "row_id": "A0",
        "method_name": "Stream3D-CropFormer",
        "variant": "v65_A0_stream3d_fullmesh_probe5",
        "tmp_config": FULLMESH_TMP_CONFIG,
        "support_scope": "FULLMESH",
        "support_policy": "fullmesh_all_scene_vertices",
        "support_owner": "v65 evaluator fullmesh support",
        "input_frame_policy": "Stream3D ScanNet/CropFormer input stride=10",
        "input_frame_policy_source": "Stream3D/dataset/scannet.py:get_frame_list(stride), Stream3D/stream4d/scannet_stream.py:frame_ids(stride=10)",
        "stream3d_stride10_aligned": True,
        "frame_policy_comparison_note": "Baseline row follows Stream3D stride-10 convention; no SOMA method-safe fullmesh row exists in v65.",
    },
    {
        "row_id": "A1",
        "method_name": "Stream3D-CropFormer",
        "variant": "v65_A1_stream3d_used_frame_support",
        "tmp_config": "v64r2_probe5_v53_bridge_wta_used_support",
        "support_scope": "USED_FRAME_VISIBLE_SUPPORT",
        "support_policy": "same used-frame visible support as A4",
        "support_owner": "SOMA v64r2 used-frame support diagnostic",
        "support_uses_rgbd_pose_mesh": True,
        "input_frame_policy": "Stream3D ScanNet/CropFormer input stride=10",
        "input_frame_policy_source": "Stream3D/dataset/scannet.py:get_frame_list(stride), Stream3D/stream4d/scannet_stream.py:frame_ids(stride=10)",
        "stream3d_stride10_aligned": True,
        "frame_policy_comparison_note": "Stream3D side is stride-10; paired SOMA bridge A4 is not stride-10 aligned, so frame policy blocks fair comparison.",
    },
    {
        "row_id": "A2",
        "method_name": "Stream3D-CropFormer",
        "variant": "v65_A2_stream3d_on_soma_bridge_prediction_union",
        "tmp_config": "v64r2_probe5_v53_bridge_wta",
        "support_scope": "SAME_SUPPORT_ON_SOMA_SUPPORT_DIAGNOSTIC",
        "support_policy": "same support as A3 SOMA bridge prediction-union",
        "support_owner": "SOMA v64r2 bridge prediction union",
        "support_uses_rgbd_pose_mesh": True,
        "input_frame_policy": "Stream3D ScanNet/CropFormer input stride=10",
        "input_frame_policy_source": "Stream3D/dataset/scannet.py:get_frame_list(stride), Stream3D/stream4d/scannet_stream.py:frame_ids(stride=10)",
        "stream3d_stride10_aligned": True,
        "frame_policy_comparison_note": "Stream3D side is stride-10; paired SOMA bridge A3 is not stride-10 aligned, so frame policy blocks fair comparison.",
    },
]


def run_v65_ap_recompute(*, audit_root: str | Path = "outputs/audit/v65_ap_contract") -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    commands.append(_run_a9_materializer_attempt())
    _build_fullmesh_tmp_config(PROBE5_SPLIT)
    for spec in DIRECT_SOMA_ROWS:
        commands.append(_run_direct_eval(spec, audit_root=audit_root))
    for spec in STREAM3D_ROWS:
        commands.append(_run_stream3d_cross_support(spec, audit_root=audit_root))
    return commands


def build_v65_ap_contract(*, command_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = [
        _row_from_spec(spec) for spec in [*STREAM3D_ROWS, *DIRECT_SOMA_ROWS]
    ]
    rows.append(_native_blocked_row())
    matrix = _comparability_matrix(rows)
    selfcheck = _selfcheck(rows)
    summary = {
        "phase": "v65_ap_contract",
        "row_count": len(rows),
        "support_scopes": sorted({row["support_scope"] for row in rows}),
        "all_rows_have_support_scope": all(bool(row.get("support_scope")) for row in rows),
        "all_rows_have_manifest": all(bool(row.get("source_manifest_path")) for row in rows),
        "all_evaluated_rows_have_hash": all(
            bool(row.get("evaluator_output_hash")) for row in rows if row.get("AP") not in (None, "")
        ),
        "method_safe_rows_with_AP": [
            row["row_id"]
            for row in rows
            if row.get("AP") not in (None, "") and row.get("is_method_result") and not row.get("forbidden_for_method_table")
        ],
        "comparison_allowed_pairs": sum(1 for row in matrix if row["comparison_status"] == "comparable"),
        "old_diagnostic_ap_support_scope": "PREDICTION_UNION_ISLAND",
        "old_diagnostic_ap_can_compare_to_stream3d": False,
        "ap_values_source": "v65_current_run_evaluator_outputs_only",
        "command_count": len(command_rows or []),
        "failed_command_count": sum(1 for row in command_rows or [] if int(row.get("returncode", 0)) != 0),
        "stream3d_rows_recomputed": all(
            row.get("status") == "ran" for row in rows if str(row.get("method_name", "")).startswith("Stream3D")
        ),
        "input_frame_policy_locked": all(bool(row.get("input_frame_policy")) for row in rows),
        "frame_policy_blocks_bridge_stream3d_pairs": any(
            row["comparison_status"] == "not_comparable_input_frame_policy" for row in matrix
        ),
    }
    summary["gate"] = {
        "all_AP_rows_have_support_scope": summary["all_rows_have_support_scope"],
        "all_rows_have_manifest": summary["all_rows_have_manifest"],
        "all_evaluated_rows_have_hash": summary["all_evaluated_rows_have_hash"],
        "diagnostic_rgbd_rows_forbidden": selfcheck["diagnostic_rgbd_rows_forbidden"],
        "method_rows_not_forbidden": selfcheck["method_rows_not_forbidden"],
        "soma_inference_policy_clean": selfcheck["soma_inference_policy_clean"],
        "stream3d_comparison_rows_same_scope_only": True,
        "all_recompute_commands_pass": summary["failed_command_count"] == 0,
        "stream3d_rows_recomputed": summary["stream3d_rows_recomputed"],
        "all_rows_have_input_frame_policy": summary["input_frame_policy_locked"],
        "comparability_checks_input_frame_policy": True,
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return {
        "summary": summary,
        "ap_contract_rows": rows,
        "ap_comparability_matrix": matrix,
        "evaluator_selfcheck_summary": selfcheck,
        "ap_recompute_commands": command_rows or [],
    }


def write_v65_ap_contract(output_root: str | Path, payload: dict[str, Any]) -> None:
    write_standard_outputs(
        output_root,
        {
            "ap_contract_summary.json": payload["summary"],
            "ap_contract_rows.csv": payload["ap_contract_rows"],
            "ap_comparability_matrix.csv": payload["ap_comparability_matrix"],
            "evaluator_selfcheck_summary.json": payload["evaluator_selfcheck_summary"],
            "ap_recompute_commands.csv": payload.get("ap_recompute_commands", []),
        },
    )


def _row_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    variant = str(spec["variant"])
    metric_path = f"data/evaluation/scannet/{variant}_class_agnostic.txt"
    source_config = str(spec.get("source_config") or variant)
    manifest_path = f"data/prediction/{source_config}_class_agnostic/config_manifest.json"
    if spec["method_name"].startswith("Stream3D"):
        manifest_path = f"data/prediction/{variant}_class_agnostic/config_manifest.json"
    manifest = load_dict(manifest_path)
    metric = parse_eval_metric_file(metric_path)
    stats = support_stats_from_summary(spec.get("support_summary_path", "")) if spec.get("support_summary_path") else {}
    return _complete_row(
        row_id=spec["row_id"],
        method_name=spec["method_name"],
        variant=variant,
        status="ran" if metric["AP"] is not None else "missing_or_not_run",
        AP=metric["AP"],
        AP50=metric["AP50"],
        AP25=metric["AP25"],
        evaluator_command=manifest.get("command"),
        evaluator_output_file=rel(metric_path),
        evaluator_output_hash=sha256_file(metric_path),
        source_manifest_path=rel(manifest_path),
        support_scope=spec["support_scope"],
        support_policy=spec["support_policy"],
        support_policy_hash=_support_policy_hash(spec, manifest_path),
        support_owner=spec["support_owner"],
        input_frame_policy=spec.get("input_frame_policy") or _infer_input_frame_policy(manifest),
        input_frame_policy_source=spec.get("input_frame_policy_source") or "",
        stream3d_stride10_aligned=spec.get("stream3d_stride10_aligned", False),
        frame_policy_comparison_note=spec.get("frame_policy_comparison_note") or "",
        uses_gt_for_prediction=bool_value(manifest.get("uses_gt_for_prediction")) or bool_value(manifest.get("uses_gt_sim3_for_prediction")),
        uses_gt_for_evaluation=bool_value(manifest.get("uses_gt_for_diagnostic")) or bool_value(manifest.get("uses_rgbd_for_evaluation_support")),
        uses_rgbd_pose_mesh_for_export=bool_value(manifest.get("uses_rgbd_for_evaluation_support"))
        or bool_value(manifest.get("uses_rgbd_pose_mesh_for_export"))
        or bool_value(manifest.get("forbidden_for_method_table"))
        or bool(spec.get("support_uses_rgbd_pose_mesh")),
        is_method_result=bool_value(manifest.get("is_method_result")),
        is_diagnostic_only=bool_value(manifest.get("is_diagnostic_only")),
        note=spec.get("note") or "AP parsed from v65 current-run evaluator output.",
        **stats,
    )


def _native_blocked_row() -> dict[str, Any]:
    a9_summary_path = Path(A9_ROOT) / "a9_materializer_summary.json"
    a9 = load_dict(a9_summary_path)
    if not a9:
        a9 = {
            "status": "not_run_a9_materializer_attempt",
            "method_safe_native_support_available": False,
            "scan_ap_join_key_available": False,
            "method_safe_ap_available": False,
            "blocker": "A9 materializer attempt has not been run in v65.",
            "support_scope": "NATIVE_CARRIER_SUPPORT_NO_SCANNET_AP_MASK",
            "support_policy": "A9 materializer attempt missing",
            "input_frame_policy": "UNKNOWN_INPUT_FRAME_POLICY",
            "native_carrier_attempt_summary": "",
        }
    return _complete_row(
        row_id="A9",
        method_name="SOMA v62/v64r2 native component field",
        variant="v65_A9_native_component_field_method_safe_attempt",
        status=str(a9.get("status") or "blocked_method_safe_materializer"),
        AP=None,
        AP50=None,
        AP25=None,
        evaluator_command=(
            f"{sys.executable} tools/run_v65_a9_method_safe_materializer.py --output-root {A9_ROOT}"
        ),
        evaluator_output_file="",
        evaluator_output_hash="",
        source_manifest_path=rel(a9_summary_path),
        support_scope=str(a9.get("support_scope") or "NATIVE_CARRIER_SUPPORT_NO_SCANNET_AP_MASK"),
        support_policy=str(a9.get("support_policy") or "native component field has no ScanNet AP mask materializer"),
        support_policy_hash=sha256_file(a9_summary_path),
        support_owner="SOMA",
        input_frame_policy=str(a9.get("input_frame_policy") or "UNKNOWN_INPUT_FRAME_POLICY"),
        input_frame_policy_source=str(a9.get("native_carrier_attempt_summary") or ""),
        stream3d_stride10_aligned=False,
        frame_policy_comparison_note=(
            "A9 attempt produces native carrier/component support, but no ScanNet AP row; "
            "therefore it cannot be compared to Stream3D stride-10 AP."
        ),
        uses_gt_for_prediction=False,
        uses_gt_for_evaluation=False,
        uses_rgbd_pose_mesh_for_export=False,
        is_method_result=False,
        is_diagnostic_only=True,
        note=str(a9.get("blocker") or "Method-safe ownership field exists, but AP materializer is unavailable."),
    )


def _complete_row(**kwargs: Any) -> dict[str, Any]:
    row = {
        "row_id": kwargs.get("row_id"),
        "method_name": kwargs.get("method_name"),
        "variant": kwargs.get("variant"),
        "split": "scannet_v6_probe5",
        "scene_count": 5,
        "evaluator_name": "evaluation.evaluate class_agnostic",
        "evaluator_command": kwargs.get("evaluator_command") or "",
        "evaluator_output_file": kwargs.get("evaluator_output_file") or "",
        "evaluator_output_hash": kwargs.get("evaluator_output_hash") or "",
        "source_manifest_path": kwargs.get("source_manifest_path") or "",
        "support_scope": kwargs.get("support_scope") or "UNKNOWN_SUPPORT",
        "support_policy": kwargs.get("support_policy") or "",
        "support_policy_hash": kwargs.get("support_policy_hash") or "",
        "support_owner": kwargs.get("support_owner") or "",
        "input_frame_policy": kwargs.get("input_frame_policy") or "UNKNOWN_INPUT_FRAME_POLICY",
        "input_frame_policy_source": kwargs.get("input_frame_policy_source") or "",
        "stream3d_stride10_aligned": bool(kwargs.get("stream3d_stride10_aligned")),
        "frame_policy_comparison_note": kwargs.get("frame_policy_comparison_note") or "",
        "pre_points_count_mean": kwargs.get("pre_points_count_mean"),
        "pre_points_count_min": kwargs.get("pre_points_count_min"),
        "pre_points_count_max": kwargs.get("pre_points_count_max"),
        "gt_instance_count_mean": kwargs.get("gt_instance_count_mean"),
        "full_scene_gt_instance_count_mean": kwargs.get("full_scene_gt_instance_count_mean"),
        "prediction_count": kwargs.get("prediction_count"),
        "mean_predictions_per_scene": kwargs.get("mean_predictions_per_scene"),
        "prediction_union_ratio": kwargs.get("prediction_union_ratio"),
        "prediction_union_inside_support_ratio": kwargs.get("prediction_union_inside_support_ratio"),
        "AP": kwargs.get("AP"),
        "AP50": kwargs.get("AP50"),
        "AP25": kwargs.get("AP25"),
        "per_GT_best_IoU_mean": kwargs.get("per_GT_best_IoU_mean"),
        "pred_best_IoU_median": kwargs.get("pred_best_IoU_median"),
        "gt_best_IoU_median": kwargs.get("gt_best_IoU_median"),
        "duplicate_predictions_per_GT": kwargs.get("duplicate_predictions_per_GT"),
        "conflict_rate": kwargs.get("conflict_rate"),
        "uses_gt_for_prediction": bool(kwargs.get("uses_gt_for_prediction")),
        "uses_gt_for_evaluation": bool(kwargs.get("uses_gt_for_evaluation")),
        "uses_rgbd_pose_mesh_for_export": bool(kwargs.get("uses_rgbd_pose_mesh_for_export")),
        "is_method_result": bool(kwargs.get("is_method_result")),
        "is_diagnostic_only": bool(kwargs.get("is_diagnostic_only")),
        "status": kwargs.get("status") or "",
        "note": kwargs.get("note") or "",
    }
    row["gt_geometry_inference_reasons"] = ";".join(gt_geometry_inference_reasons(row))
    row = normalize_reportability(row, context=f"v65 AP row {row['row_id']}")
    return row


def _comparability_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for left in rows:
        for right in rows:
            if left["row_id"] >= right["row_id"]:
                continue
            same = {
                "support_scope_same": left["support_scope"] == right["support_scope"] and left["support_scope"] != "UNKNOWN_SUPPORT",
                "support_policy_hash_same": bool(
                    left.get("support_policy_hash")
                    and right.get("support_policy_hash")
                    and left.get("support_policy_hash") == right.get("support_policy_hash")
                ),
                "evaluator_same": left["evaluator_name"] == right["evaluator_name"],
                "split_same": left["split"] == right["split"],
                "class_setting_same": True,
                "score_policy_documented": bool(left.get("support_policy") and right.get("support_policy")),
                "input_frame_policy_same": left.get("input_frame_policy") == right.get("input_frame_policy"),
                "stream3d_stride10_aligned": bool(
                    left.get("stream3d_stride10_aligned") and right.get("stream3d_stride10_aligned")
                ),
            }
            comparable = bool(all(same.values()))
            if not comparable and all(
                same[key]
                for key in [
                    "support_scope_same",
                    "support_policy_hash_same",
                    "evaluator_same",
                    "split_same",
                    "class_setting_same",
                    "score_policy_documented",
                ]
            ):
                status = "not_comparable_input_frame_policy"
                reason = "input frame policy is not aligned to Stream3D stride-10 on both sides"
            else:
                status = "comparable" if comparable else "not_comparable"
                reason = "" if comparable else "support-hash/evaluator/split/class/score/frame-policy contract not all satisfied"
            matrix.append(
                {
                    "left_row_id": left["row_id"],
                    "right_row_id": right["row_id"],
                    **same,
                    "comparison_status": status,
                    "reason": reason,
                }
            )
    return matrix


def _selfcheck(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostic_rgbd_ok = all(
        row["forbidden_for_method_table"]
        for row in rows
        if row.get("uses_rgbd_pose_mesh_for_export") or row.get("uses_gt_for_evaluation")
    )
    method_rows_ok = all(
        not row["forbidden_for_method_table"] for row in rows if row.get("is_method_result") and row.get("AP") not in (None, "")
    )
    policy_violations = [
        {"row_id": row["row_id"], "violations": policy_violation_reasons(row)}
        for row in rows
        if policy_violation_reasons(row)
    ]
    return {
        "diagnostic_rgbd_rows_forbidden": diagnostic_rgbd_ok,
        "method_rows_not_forbidden": method_rows_ok,
        "soma_inference_policy_clean": not policy_violations,
        "soma_inference_policy_violations": policy_violations,
        "rows_missing_support_scope": [row["row_id"] for row in rows if not row.get("support_scope")],
        "rows_missing_input_frame_policy": [row["row_id"] for row in rows if not row.get("input_frame_policy")],
        "rows_missing_hash": [row["row_id"] for row in rows if row.get("AP") not in (None, "") and not row.get("evaluator_output_hash")],
        "pass": bool(diagnostic_rgbd_ok and method_rows_ok and not policy_violations),
    }


def _infer_input_frame_policy(manifest: dict[str, Any]) -> str:
    text = " ".join(str(manifest.get(key, "")) for key in ["command", "source_configs", "support_policy", "chunking_policy"])
    if "stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1" in text:
        return "D4RT debug frames stride-10, window_size=32, window_stride=16"
    if "scannet" in text.lower():
        return "Stream3D ScanNet/CropFormer input stride=10"
    return "UNKNOWN_INPUT_FRAME_POLICY"


def _support_policy_hash(spec: dict[str, Any], manifest_path: str) -> str:
    tmp_config = spec.get("tmp_config")
    if tmp_config:
        tmp_manifest = f"data/TMP/{tmp_config}/config_manifest.json"
        digest = sha256_file(tmp_manifest)
        if digest:
            return digest
    if spec.get("support_summary_path"):
        digest = sha256_file(spec["support_summary_path"])
        if digest:
            return digest
    return sha256_file(manifest_path)


def _build_fullmesh_tmp_config(split_path: str) -> None:
    split = project(split_path)
    scenes = [line.strip() for line in split.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = project("data/TMP") / FULLMESH_TMP_CONFIG
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in scenes:
        gt_path = project("data/scannet/gt") / f"{scene}.txt"
        gt = np.loadtxt(gt_path, dtype=np.int64)
        pre_points = np.arange(gt.shape[0], dtype=np.int64)
        np.save(out / f"{scene}_pre_points.npy", pre_points)
        rows.append({"scene": scene, "pre_points_count": int(pre_points.shape[0])})
    manifest = {
        "schema_version": "stream4d_prediction_manifest_v1",
        "output_config": FULLMESH_TMP_CONFIG,
        "support_policy": "fullmesh_all_scene_vertices",
        "pre_points_policy": "all_scene_vertices_from_gt_length_for_evaluation_adapter",
        "is_method_result": False,
        "is_diagnostic_only": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": False,
        "uses_rgbd_for_evaluation_support": False,
        "notes": "v65 AP evaluation support: all scene vertices for probe5 scenes. Uses GT file length only to create evaluator support indices; does not alter predictions.",
        "rows": rows,
    }
    from .v47_common import write_json

    write_json(out / "config_manifest.json", manifest)


def _run_direct_eval(spec: dict[str, Any], *, audit_root: str | Path) -> dict[str, Any]:
    variant = str(spec["variant"])
    source_config = str(spec["source_config"])
    output_file = f"data/evaluation/scannet/{variant}_class_agnostic.txt"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        f"data/prediction/{source_config}_class_agnostic",
        "--gt_path",
        "data/scannet/gt",
        "--dataset",
        "scannet",
        "--output_file",
        output_file,
        "--tmp_root",
        "data/TMP",
        "--tmp_config",
        str(spec["tmp_config"]),
        "--no_class",
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    return _run_cmd(spec["row_id"], cmd, cwd=project("."))


def _run_stream3d_cross_support(spec: dict[str, Any], *, audit_root: str | Path) -> dict[str, Any]:
    variant = str(spec["variant"])
    cmd = [
        sys.executable,
        "tools/evaluate_cross_prepoints.py",
        "--root",
        ".",
        "--seq-list",
        PROBE5_SPLIT,
        "--pred-config",
        "scannet",
        "--source-pre-points-config",
        "scannet_self_inherit_probe5",
        "--pre-points-config",
        str(spec["tmp_config"]),
        "--output-config",
        variant,
        "--dataset",
        "scannet",
        "--no-class",
        "--output-file",
        f"data/evaluation/scannet/{variant}_class_agnostic.txt",
        "--audit-root",
        str(audit_root),
        "--require-manifest",
        "--allow-diagnostic-eval",
        "--eval-policy",
        "v65_same_support_or_fullmesh_contract",
    ]
    return _run_cmd(spec["row_id"], cmd, cwd=project("."))


def _run_a9_materializer_attempt() -> dict[str, Any]:
    cmd = [
        sys.executable,
        "tools/run_v65_a9_method_safe_materializer.py",
        "--output-root",
        A9_ROOT,
    ]
    return _run_cmd("A9", cmd, cwd=project("."))


def _run_cmd(row_id: str, cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env)
    return {
        "row_id": row_id,
        "command": " ".join(cmd),
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }
