from __future__ import annotations

from pathlib import Path
from typing import Any

from .v47_common import read_csv
from .v65_ap_contract import _run_stream3d_cross_support
from .v65_common import float_or_none, load_dict, parse_eval_metric_file, project, rel, sha256_file, write_standard_outputs


EXTRA_PARITY_SPECS = [
    {
        "row_id": "S3D3",
        "method_name": "Stream3D-CropFormer",
        "variant": "v65_S3D3_stream3d_on_d4rt_g11_prediction_union",
        "tmp_config": "v64r2_d4rt_chunk_scale_first_ap_probe5_g11",
        "support_scope": "SAME_SUPPORT_ON_SOMA_SUPPORT_DIAGNOSTIC",
        "support_policy": "same support as SOMA D4RT G11 prediction-union",
        "support_owner": "SOMA D4RT G11 prediction union",
        "support_owner_bias_note": "support is defined by SOMA D4RT G11 prediction-union; diagnostic only.",
        "support_uses_rgbd_pose_mesh": True,
        "input_frame_policy": "Stream3D ScanNet/CropFormer input stride=10; support owner D4RT debug frames stride-10",
        "input_frame_policy_source": "Stream3D/dataset/scannet.py; outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1/*/carriers_window*_manifest.json",
        "stream3d_stride10_aligned": True,
    },
    {
        "row_id": "S3D4",
        "method_name": "Stream3D-CropFormer",
        "variant": "v65_S3D4_stream3d_on_d4rt_g12_prediction_union",
        "tmp_config": "v64r2_d4rt_chunk_scale_first_ap_probe5_g12",
        "support_scope": "SAME_SUPPORT_ON_SOMA_SUPPORT_DIAGNOSTIC",
        "support_policy": "same support as SOMA D4RT G12 prediction-union",
        "support_owner": "SOMA D4RT G12 prediction union",
        "support_owner_bias_note": "support is defined by SOMA D4RT G12 prediction-union; diagnostic only.",
        "support_uses_rgbd_pose_mesh": True,
        "input_frame_policy": "Stream3D ScanNet/CropFormer input stride=10; support owner D4RT debug frames stride-10",
        "input_frame_policy_source": "Stream3D/dataset/scannet.py; outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1/*/carriers_window*_manifest.json",
        "stream3d_stride10_aligned": True,
    },
]

CONTRACT_TO_PARITY = [
    ("S3D0", "A0", "Stream3D-CropFormer fullmesh", "FULLMESH", "v65 evaluator fullmesh support", ""),
    (
        "S3D1",
        "A1",
        "Stream3D-CropFormer used-frame visible support",
        "USED_FRAME_VISIBLE_SUPPORT",
        "SOMA v64r2 used-frame support diagnostic",
        "support generated from used RGB-D/pose visible support; diagnostic/forbidden.",
    ),
    (
        "S3D2",
        "A2",
        "Stream3D-CropFormer on SOMA bridge prediction-union support",
        "SAME_SUPPORT_ON_SOMA_SUPPORT_DIAGNOSTIC",
        "SOMA v64r2 bridge prediction union",
        "support is owned by SOMA bridge prediction-union; diagnostic only.",
    ),
]


def run_v65_stream3d_parity(*, audit_root: str | Path = "outputs/audit/v65_stream3d_parity") -> list[dict[str, Any]]:
    return [_run_stream3d_cross_support(spec, audit_root=audit_root) for spec in EXTRA_PARITY_SPECS]


def build_v65_stream3d_parity(*, command_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    contract_rows = _contract_rows()
    stream3d_rows: list[dict[str, Any]] = []
    support_bias_rows: list[dict[str, Any]] = []
    for row_id, contract_id, label, support_scope, owner, bias in CONTRACT_TO_PARITY:
        source = contract_rows.get(contract_id, {})
        aggregate = _aggregate_for_variant(str(source.get("variant") or ""))
        row = {
            "baseline_variant": row_id,
            "source_contract_row": contract_id,
            "label": label,
            "support_scope": support_scope,
            "AP": float_or_none(source.get("AP")),
            "AP50": float_or_none(source.get("AP50")),
            "AP25": float_or_none(source.get("AP25")),
            "pre_points_count_mean": _mean_target_pre_points(aggregate),
            "gt_instance_count_mean": aggregate.get("mean_gt_instances_in_target_pre_points")
            if aggregate
            else float_or_none(source.get("gt_instance_count_mean")),
            "prediction_count": float_or_none(source.get("prediction_count")),
            "mean_predictions_per_scene": aggregate.get("mean_num_pred_instances")
            if aggregate
            else float_or_none(source.get("mean_predictions_per_scene")),
            "prediction_union_ratio": aggregate.get("mean_prediction_union_ratio")
            if aggregate
            else float_or_none(source.get("prediction_union_ratio")),
            "support_owner": owner,
            "support_owner_bias_note": bias,
            "input_frame_policy": source.get("input_frame_policy"),
            "input_frame_policy_source": source.get("input_frame_policy_source"),
            "stream3d_stride10_aligned": source.get("stream3d_stride10_aligned"),
            "frame_policy_comparison_note": source.get("frame_policy_comparison_note"),
            "evaluator_output_file": source.get("evaluator_output_file"),
            "evaluator_output_hash": source.get("evaluator_output_hash"),
            "forbidden_for_method_table": source.get("forbidden_for_method_table"),
        }
        stream3d_rows.append(row)
        support_bias_rows.append(_bias_row(row))
    for spec in EXTRA_PARITY_SPECS:
        row = _row_from_extra(spec)
        stream3d_rows.append(row)
        support_bias_rows.append(_bias_row(row))

    s3d0_available = any(row["baseline_variant"] == "S3D0" and row.get("AP") is not None for row in stream3d_rows)
    same_support_available = any(
        str(row.get("support_scope")) == "SAME_SUPPORT_ON_SOMA_SUPPORT_DIAGNOSTIC" and row.get("AP") is not None
        for row in stream3d_rows
    )
    failed = [row for row in command_rows or [] if int(row.get("returncode", 0)) != 0]
    summary = {
        "phase": "v65_stream3d_parity",
        "stream3d_official_status": "available_v65_fullmesh_probe5" if s3d0_available else "blocked",
        "S3D0_or_reference_available": s3d0_available,
        "same_support_row_available": same_support_available,
        "row_count": len(stream3d_rows),
        "failed_command_count": len(failed),
        "stride10_aligned_row_count": sum(
            1 for row in stream3d_rows if str(row.get("stream3d_stride10_aligned")).lower() == "true"
        ),
        "gate": {
            "S3D0_or_accepted_reference_baseline_available": s3d0_available,
            "S3D_rows_have_schema": all("support_scope" in row and "AP" in row for row in stream3d_rows),
            "at_least_one_same_support_row_available": same_support_available,
            "all_extra_commands_pass": len(failed) == 0,
            "input_frame_policy_recorded": all(bool(row.get("input_frame_policy")) for row in stream3d_rows),
        },
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return {
        "summary": summary,
        "stream3d_ap_rows": stream3d_rows,
        "support_bias_rows": support_bias_rows,
        "stream3d_parity_commands": command_rows or [],
    }


def write_v65_stream3d_parity(output_root: str | Path, payload: dict[str, Any]) -> None:
    write_standard_outputs(
        output_root,
        {
            "stream3d_parity_summary.json": payload["summary"],
            "stream3d_ap_rows.csv": payload["stream3d_ap_rows"],
            "support_bias_rows.csv": payload["support_bias_rows"],
            "stream3d_parity_commands.csv": payload.get("stream3d_parity_commands", []),
        },
    )


def _contract_rows() -> dict[str, dict[str, str]]:
    path = project("outputs/audit/v65_ap_contract/ap_contract_rows.csv")
    if not path.exists():
        return {}
    return {row["row_id"]: row for row in read_csv(path)}


def _row_from_extra(spec: dict[str, Any]) -> dict[str, Any]:
    metric_path = f"data/evaluation/scannet/{spec['variant']}_class_agnostic.txt"
    metric = parse_eval_metric_file(metric_path)
    summary = load_dict(f"outputs/audit/v65_stream3d_parity/cross_prepoints/{spec['variant']}_summary.json")
    aggregate = summary.get("aggregate") if isinstance(summary.get("aggregate"), dict) else {}
    return {
        "baseline_variant": spec["row_id"],
        "source_contract_row": "",
        "label": spec["variant"],
        "support_scope": spec["support_scope"],
        "AP": metric.get("AP"),
        "AP50": metric.get("AP50"),
        "AP25": metric.get("AP25"),
        "pre_points_count_mean": _mean_target_pre_points(aggregate),
        "gt_instance_count_mean": aggregate.get("mean_gt_instances_in_target_pre_points"),
        "prediction_count": None,
        "mean_predictions_per_scene": aggregate.get("mean_num_pred_instances"),
        "prediction_union_ratio": aggregate.get("mean_prediction_union_ratio"),
        "support_owner": spec["support_owner"],
        "support_owner_bias_note": spec["support_owner_bias_note"],
        "input_frame_policy": spec.get("input_frame_policy"),
        "input_frame_policy_source": spec.get("input_frame_policy_source"),
        "stream3d_stride10_aligned": spec.get("stream3d_stride10_aligned", False),
        "frame_policy_comparison_note": "Stream3D prediction cadence and D4RT support owner cadence are both stride-10; still diagnostic because support is SOMA-owned.",
        "evaluator_output_file": rel(metric_path),
        "evaluator_output_hash": sha256_file(metric_path),
        "forbidden_for_method_table": True,
    }


def _bias_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_variant": row.get("baseline_variant"),
        "support_scope": row.get("support_scope"),
        "support_owner": row.get("support_owner"),
        "support_owner_bias_note": row.get("support_owner_bias_note"),
        "input_frame_policy": row.get("input_frame_policy"),
        "stream3d_stride10_aligned": row.get("stream3d_stride10_aligned"),
        "forbidden_for_method_table": row.get("forbidden_for_method_table"),
        "can_use_for_official_win_loss": bool(
            row.get("baseline_variant") == "S3D0"
            and row.get("support_scope") == "FULLMESH"
            and str(row.get("forbidden_for_method_table")).lower() == "false"
            and str(row.get("stream3d_stride10_aligned")).lower() == "true"
        ),
    }


def _aggregate_for_variant(variant: str) -> dict[str, Any]:
    for root in ["outputs/audit/v65_ap_contract", "outputs/audit/v65_stream3d_parity"]:
        summary = load_dict(f"{root}/cross_prepoints/{variant}_summary.json")
        aggregate = summary.get("aggregate") if isinstance(summary.get("aggregate"), dict) else {}
        if aggregate:
            return aggregate
    return {}


def _mean_target_pre_points(aggregate: dict[str, Any]) -> float | None:
    scenes = float_or_none(aggregate.get("ok_scenes") or aggregate.get("scenes"))
    total = float_or_none(aggregate.get("sum_target_pre_points"))
    if scenes and total is not None:
        return float(total / scenes)
    return None
