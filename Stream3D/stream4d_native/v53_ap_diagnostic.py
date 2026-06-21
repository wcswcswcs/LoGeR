from __future__ import annotations

import subprocess
import sys
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_float, parse_int, read_csv, read_json, utc_now, write_csv, write_json


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _load_json(path: str | Path) -> dict[str, Any]:
    path_obj = _project(path)
    if not path_obj.exists():
        return {}
    payload = read_json(path_obj)
    return payload if isinstance(payload, dict) else {}


def _csv_fieldnames(path: str | Path) -> list[str]:
    path_obj = _project(path)
    if not path_obj.exists():
        return []
    with path_obj.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def native_method_export_repair_audit(
    *,
    carrier_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv",
    objectlet_rows_path: str | Path = "outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000/objectlet_rows.csv",
    chunk_component_rows_path: str | Path = "outputs/audit/v53_chunk_universe/chunk_component_rows.csv",
    v53_native_carrier_summary_path: str | Path = "outputs/audit/v53_native_carrier_materialization/native_carrier_summary.json",
    v42_native_summary_path: str | Path = "outputs/audit/v42_streaming_memory_unioncap320_r1/memory_summary.json",
    native_exporter_path: str | Path = "stream4d_native/object_field_native_export.py",
) -> dict[str, Any]:
    native_point_fields = {
        "native_point_id",
        "native_point_ids",
        "point_id",
        "point_ids",
        "scene_point_id",
        "scene_point_ids",
        "mesh_vertex_id",
        "mesh_vertex_ids",
        "vertex_id",
        "vertex_ids",
    }
    object_field_link_fields = {
        "tube_id",
        "tube_ids",
        "object_field_id",
        "object_field_ids",
        "primary_field_id",
        "attached_tube_ids",
        "native_object_id",
    }
    carrier_fields = _csv_fieldnames(carrier_table_path)
    objectlet_fields = _csv_fieldnames(objectlet_rows_path)
    chunk_fields = _csv_fieldnames(chunk_component_rows_path)
    carrier_field_set = set(carrier_fields)
    objectlet_field_set = set(objectlet_fields)
    chunk_field_set = set(chunk_fields)
    v53_native_carrier_summary = _load_json(v53_native_carrier_summary_path)
    v42_summary = _load_json(v42_native_summary_path)
    native_exporter_exists = _project(native_exporter_path).exists()
    has_carrier_native_mapping = bool(carrier_field_set & native_point_fields)
    has_objectlet_native_mapping = bool(objectlet_field_set & (native_point_fields | object_field_link_fields))
    has_chunk_native_mapping = bool(chunk_field_set & (native_point_fields | object_field_link_fields))
    v53_native_carrier_support_available = (
        bool(v53_native_carrier_summary.get("native_carrier_materialization_pass"))
        and bool(v53_native_carrier_summary.get("method_safe_native_support_available"))
        and not bool(v53_native_carrier_summary.get("uses_gt_for_prediction"))
        and not bool(v53_native_carrier_summary.get("uses_gt_for_diagnostic_labels"))
        and not bool(v53_native_carrier_summary.get("uses_rgbd_pose_mesh_for_export"))
        and not bool(v53_native_carrier_summary.get("is_scannet_ap_export"))
    )
    can_reuse_v42_as_ap = (
        bool(v42_summary.get("native_export_smoke_pass"))
        and str(v42_summary.get("AP_bridge_status")) != "not_evaluated_native_support_not_scannet_ap"
        and not bool(v42_summary.get("uses_gt_for_scoring"))
    )
    current_artifacts_have_method_safe_ap_inputs = bool(
        has_carrier_native_mapping or has_objectlet_native_mapping or has_chunk_native_mapping or can_reuse_v42_as_ap
    )
    if current_artifacts_have_method_safe_ap_inputs:
        repair_result = "method_safe_native_ap_inputs_available"
    elif v53_native_carrier_support_available:
        repair_result = "native_d4rt_carrier_support_available_scannet_ap_still_blocked"
    else:
        repair_result = "blocked_current_v53_artifacts_missing_native_point_or_mesh_vertex_mapping"
    blocked_reason = (
        "v53 selected objectlets are now materialized to D4RT carrier_global_id support observations, but that support "
        "does not include ScanNet scene point ids or mesh vertex ids, so it cannot satisfy the current ScanNet AP evaluator "
        "mask contract without a native-carrier evaluator or an audited carrier-to-scene calibration."
        if v53_native_carrier_support_available and not current_artifacts_have_method_safe_ap_inputs
        else (
            "v53 selected objectlets are keyed by component/source_mask_observation_id and the available carrier/chunk/objectlet tables "
            "do not carry native point ids, ScanNet scene point ids, mesh vertex ids, or ObjectField tube links. "
            "The v42 native exporter only emits native D4RT support points and records AP_bridge_status=not_evaluated_native_support_not_scannet_ap."
        )
    )
    required_future_change = (
        "connect audited v53 D4RT carrier ids to ScanNet scene/mesh point ids or add a native-carrier AP-style evaluator; "
        "do not reuse RGB-D/pose/mesh backprojection for a method table row."
        if v53_native_carrier_support_available and not current_artifacts_have_method_safe_ap_inputs
        else (
            "materialize an audited v53 component/objectlet to native D4RT point or ScanNet scene-point mapping, "
            "then implement a method-safe native-to-AP exporter without RGB-D/pose/mesh backprojection leakage."
        )
    )
    return {
        "repair_attempted": True,
        "repair_result": repair_result,
        "method_safe_native_ap_export_available": bool(current_artifacts_have_method_safe_ap_inputs),
        "method_safe_native_support_available": bool(v53_native_carrier_support_available),
        "carrier_table_path": _rel(carrier_table_path),
        "carrier_table_exists": _project(carrier_table_path).exists(),
        "carrier_fields": carrier_fields,
        "carrier_has_native_point_or_mesh_vertex_mapping": has_carrier_native_mapping,
        "objectlet_rows_path": _rel(objectlet_rows_path),
        "objectlet_fields": objectlet_fields,
        "objectlet_has_native_or_object_field_link": has_objectlet_native_mapping,
        "chunk_component_rows_path": _rel(chunk_component_rows_path),
        "chunk_component_fields": chunk_fields,
        "chunk_component_has_native_or_object_field_link": has_chunk_native_mapping,
        "v53_native_carrier_summary_path": _rel(v53_native_carrier_summary_path),
        "v53_native_carrier_summary_exists": bool(v53_native_carrier_summary),
        "v53_native_carrier_materialization_pass": v53_native_carrier_summary.get("native_carrier_materialization_pass"),
        "v53_native_carrier_support_available": bool(v53_native_carrier_support_available),
        "v53_native_carrier_support_kind": v53_native_carrier_summary.get("native_support_kind"),
        "v53_native_carrier_observation_row_count": v53_native_carrier_summary.get("native_observation_row_count"),
        "v53_native_unique_carrier_count": v53_native_carrier_summary.get("native_unique_carrier_count"),
        "v53_native_objectlet_count_with_carriers": v53_native_carrier_summary.get("objectlet_count_with_native_carriers"),
        "v53_native_AP_bridge_status": v53_native_carrier_summary.get("AP_bridge_status"),
        "v53_native_real_method_ap_status": v53_native_carrier_summary.get("real_method_ap_status"),
        "v42_native_summary_path": _rel(v42_native_summary_path),
        "v42_native_summary_exists": bool(v42_summary),
        "v42_native_export_smoke_pass": v42_summary.get("native_export_smoke_pass"),
        "v42_native_point_count": v42_summary.get("native_point_count"),
        "v42_AP_bridge_status": v42_summary.get("AP_bridge_status"),
        "v42_uses_gt_for_prediction": v42_summary.get("uses_gt_for_prediction"),
        "v42_uses_gt_for_scoring": v42_summary.get("uses_gt_for_scoring"),
        "v42_can_be_reused_as_method_safe_scannet_ap": can_reuse_v42_as_ap,
        "native_exporter_path": _rel(native_exporter_path),
        "native_exporter_exists": native_exporter_exists,
        "attempted_repair_paths": [
            "reuse_v42_native_support_bridge",
            "map_v53_objectlets_through_carrier_table",
            "map_v53_objectlets_through_chunk_component_rows",
            "materialize_v53_objectlets_to_d4rt_carrier_support",
            "rgbd_pose_mesh_backprojection_bridge_as_diagnostic_only",
        ],
        "blocked_reason": blocked_reason,
        "required_future_change": required_future_change,
    }


def _parse_ap_metric_file(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return None
    parts = lines[-1].split(",")
    if len(parts) != 3:
        return None
    try:
        return {"AP": float(parts[0]), "AP50": float(parts[1]), "AP25": float(parts[2])}
    except ValueError:
        return None


def ap_smoke_gate(best: dict[str, Any], selected_object_count: int, exporter_input_rows_available: bool) -> dict[str, bool]:
    gate = {
        "uses_gt_for_prediction_eq_false": not bool(best.get("uses_gt_for_prediction", False)),
        "selected_object_count_gt_0": int(selected_object_count) > 0,
        "maskless_object_count_eq_0": parse_int(best.get("maskless_object_count")) == 0,
        "birth_from_d4rt_tube_count_eq_0": parse_int(best.get("birth_from_d4rt_tube_count")) == 0,
        "mean_predictions_per_scene_le_350": parse_float(best.get("mean_predictions_per_scene"), 9999.0) <= 350.0,
        "exporter_input_rows_available": bool(exporter_input_rows_available),
    }
    gate["pass"] = bool(all(gate.values()))
    return gate


def ap_diagnostic_identity_gate(best: dict[str, Any]) -> dict[str, bool]:
    gate = {
        "ARI_ge_0.43": parse_float(best.get("4D_ARI"), -9999.0) >= 0.43,
        "purity_ge_0.82": parse_float(best.get("4D_purity"), -9999.0) >= 0.82,
        "completeness_ge_0.45": parse_float(best.get("4D_completeness"), -9999.0) >= 0.45,
        "mean_predictions_per_scene_le_280": parse_float(best.get("mean_predictions_per_scene"), 9999.0) <= 280.0,
        "conflict_rate_le_0.25": parse_float(best.get("conflict_rate"), 9999.0) <= 0.25,
        "no_GT_prediction_leakage": not bool(best.get("uses_gt_for_prediction", False)),
    }
    gate["pass"] = bool(all(gate.values()))
    return gate


def _selected_objectlet_rows(objectlet_rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    return [row for row in objectlet_rows if str(row.get("variant")) == str(variant)]


def _mask_lookup(mask_table_path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(mask_table_path)
    return {str(row.get("mask_observation_id")): row for row in rows}


def _run_bridge_export(
    *,
    selected_rows: list[dict[str, str]],
    mask_rows_by_id: dict[str, dict[str, str]],
    output_root: Path,
    config: str,
    ap_row: str,
    export_mask_sample_stride: int,
    export_mask_max_pixels: int,
    export_nn_radius: float,
    export_score_mode: str,
    score_policy: str,
    wta_policy: str,
    export_min_points_per_object: int,
    export_enable_wta: bool,
) -> dict[str, Any]:
    from stream4d.export_scannet import ScanNetExporter
    from stream4d.reliable_densifier import apply_wta_to_records
    from stream4d.scannet_stream import ScanNetStream
    from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest

    scene_objects: dict[str, list[tuple[int, dict[str, str], dict[str, str]]]] = defaultdict(list)
    missing_mask_rows: list[dict[str, Any]] = []
    for object_index, objectlet in enumerate(selected_rows):
        mask_id = str(objectlet.get("source_mask_observation_id") or "")
        mask_row = mask_rows_by_id.get(mask_id)
        if not mask_row:
            missing_mask_rows.append(
                {
                    "variant": ap_row,
                    "scene": objectlet.get("scene"),
                    "objectlet_id": objectlet.get("objectlet_id"),
                    "failure_stage": "input_lookup",
                    "error": f"missing mask_observation_id {mask_id}",
                }
            )
            continue
        scene = str(mask_row.get("scene") or mask_row.get("scene_id") or objectlet.get("scene") or "")
        scene_objects[scene].append((object_index, objectlet, mask_row))

    scene_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = list(missing_mask_rows)
    for scene, objects in sorted(scene_objects.items()):
        try:
            stream = ScanNetStream(seq_name=scene)
            errors = stream.validate(require_masks=False)
            if errors:
                raise RuntimeError("; ".join(errors))
            exporter = ScanNetExporter(
                stream,
                output_config=config,
                export_nn_radius=export_nn_radius,
                export_support_mode="mask_backproject",
                export_mask_sample_stride=export_mask_sample_stride,
                export_mask_max_pixels=export_mask_max_pixels,
                export_min_points_per_object=export_min_points_per_object,
                export_score_mode=export_score_mode,
            )
            object_records: list[dict[str, Any]] = []
            object_dict: dict[int, dict[str, Any]] = {}
            backproject_queries = 0
            backproject_hits = 0
            for object_id, objectlet, mask in objects:
                frame_id = parse_int(mask.get("frame_id"))
                mask_label_id = parse_int(mask.get("mask_id"))
                hit_ids, query_count = exporter._backproject_mask(frame_id, mask_label_id, nn_radius=export_nn_radius)
                backproject_queries += int(query_count)
                backproject_hits += int(hit_ids.shape[0])
                point_ids = set(int(value) for value in hit_ids.tolist())
                record = {
                    "object_id": int(object_id),
                    "point_ids": point_ids,
                    "score": float(len(point_ids)),
                    "area_score": float(len(point_ids)),
                    "reliability": parse_float(objectlet.get("candidate_success_rate"), 0.0) * max(float(len(point_ids)), 1.0),
                    "observations": 1.0,
                }
                object_records.append(record)
                object_dict[int(object_id)] = {
                    "point_ids": np.asarray(sorted(point_ids), dtype=np.int64),
                    "mask_list": [(frame_id, mask_label_id, parse_float(mask.get("mask_area")))],
                    "repre_mask_list": [(frame_id, mask_label_id, parse_float(mask.get("mask_area")))],
                    "score": float(len(point_ids)),
                    "area_score": float(len(point_ids)),
                    "source_variant": "v53_local_objectlet_rgbd_pose_mesh_bridge_diagnostic",
                    "source_objectlet_id": objectlet.get("objectlet_id"),
                }
            wta_diag: dict[str, float] = {}
            if export_enable_wta:
                object_records, wta_diag = apply_wta_to_records(object_records)
                for record in object_records:
                    object_id = int(record["object_id"])
                    if object_id in object_dict:
                        point_ids = sorted(record["point_ids"])
                        object_dict[object_id]["point_ids"] = np.asarray(point_ids, dtype=np.int64)
                        object_dict[object_id]["score"] = float(record.get("score", len(point_ids)))
                        object_dict[object_id]["area_score"] = float(record.get("area_score", len(point_ids)))
            diag = exporter._write_outputs(
                object_records,
                object_dict,
                np.zeros((exporter.scene_points.shape[0],), dtype=np.uint16),
            )
            diag.update(wta_diag)
            manifest = build_prediction_manifest(
                output_config=config,
                root=ROOT,
                is_method_result=False,
                is_diagnostic_only=True,
                uses_gt=False,
                gt_usage="none",
                source_configs=[
                    "outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000/objectlet_rows.csv",
                    "outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv",
                ],
                pre_points_policy="rgbd_pose_mesh_bridge_recompute",
                support_policy="v53_best_legal_objectlet_source_mask_backproject",
                notes="v53 AP RGB-D/pose/mesh bridge diagnostic. Forbidden for method table; local identity controls failed.",
                extra={
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_evaluation_alignment": False,
                    "uses_rgbd_pose_mesh_for_export": True,
                    "uses_rgbd_for_prediction": True,
                    "uses_pose_for_prediction": True,
                    "uses_scannet_mesh_for_prediction": True,
                    "forbidden_for_method_table": True,
                    "phase": "v53_ap_diagnostic",
                    "ap_row": ap_row,
                    "score_policy": score_policy,
                    "wta_policy": wta_policy,
                    "export_score_mode": export_score_mode,
                    "export_min_points_per_object": export_min_points_per_object,
                },
            )
            write_prediction_manifest(config, manifest, root=ROOT, pred_suffix="class_agnostic")
            pred_path = ROOT / "data/prediction" / f"{config}_class_agnostic" / f"{scene}.npz"
            with np.load(pred_path) as pred:
                pred_masks = np.asarray(pred["pred_masks"], dtype=bool)
                pre_points = np.flatnonzero(pred_masks.any(axis=1)).astype(np.int64)
            scene_rows.append(
                {
                    "variant": ap_row,
                    "scene": scene,
                    "exporter_exit_code": 0,
                    "candidate_object_count": len(objects),
                    "num_exported_objects": diag.get("num_exported_objects"),
                    "num_scene_points": diag.get("num_scene_points"),
                    "num_exported_points": diag.get("num_exported_points"),
                    "prediction_file_exists": pred_path.exists(),
                    "prediction_path": _rel(pred_path),
                    "pre_points_path": _rel(ROOT / "data/TMP" / config / f"{scene}_pre_points.npy"),
                    "pre_percent": float(diag.get("num_exported_points", 0.0) / max(float(diag.get("num_scene_points", 1.0)), 1.0)),
                    "union_percent": float(pre_points.shape[0] / max(pred_masks.shape[0], 1)),
                    "export_conflict_rate": diag.get("export_conflict_rate"),
                    "backproject_queries": backproject_queries,
                    "backproject_hits": backproject_hits,
                    "backproject_hit_rate": backproject_hits / max(backproject_queries, 1),
                    "score_policy": score_policy,
                    "wta_policy": wta_policy,
                    "export_score_mode": export_score_mode,
                    "export_min_points_per_object": export_min_points_per_object,
                    "wta_pre_conflict_rate": diag.get("densify_wta_pre_conflict_rate"),
                    "wta_removed_assignment_rate": diag.get("densify_wta_removed_assignment_rate"),
                    "error": "",
                }
            )
        except Exception as exc:
            scene_rows.append(
                {
                    "variant": ap_row,
                    "scene": scene,
                    "exporter_exit_code": 1,
                    "candidate_object_count": len(objects),
                    "num_exported_objects": 0,
                    "num_scene_points": None,
                    "num_exported_points": 0,
                    "prediction_file_exists": False,
                    "prediction_path": "",
                    "pre_points_path": "",
                    "pre_percent": 0.0,
                    "union_percent": 0.0,
                    "export_conflict_rate": None,
                    "backproject_queries": 0,
                    "backproject_hits": 0,
                    "backproject_hit_rate": 0.0,
                    "score_policy": score_policy,
                    "wta_policy": wta_policy,
                    "export_score_mode": export_score_mode,
                    "export_min_points_per_object": export_min_points_per_object,
                    "wta_pre_conflict_rate": None,
                    "wta_removed_assignment_rate": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failure_rows.append(
                {
                    "variant": ap_row,
                    "scene": scene,
                    "failure_stage": "exporter",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metric_file = eval_dir / f"{config}_class_agnostic.txt"
    log_path = eval_dir / f"{config}_evaluate.log"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(ROOT / "data/prediction" / f"{config}_class_agnostic"),
        "--gt_path",
        str(ROOT / "data/scannet/gt"),
        "--dataset",
        "scannet",
        "--no_class",
        "--tmp_root",
        str(ROOT / "data/TMP"),
        "--tmp_config",
        config,
        "--output_file",
        str(metric_file),
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    evaluator_exit_code = None
    metrics = None
    if any(row.get("prediction_file_exists") for row in scene_rows):
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=handle, stderr=subprocess.STDOUT)
        evaluator_exit_code = int(proc.returncode)
        metrics = _parse_ap_metric_file(metric_file)
        if metrics is None:
            fallback = eval_dir / f"{config}_class_agnostic_class_agnostic.txt"
            metrics = _parse_ap_metric_file(fallback)
    else:
        failure_rows.append(
            {
                "variant": ap_row,
                "scene": "ALL",
                "failure_stage": "evaluator",
                "error": "not_run_no_prediction_files",
            }
        )

    return {
        "config": config,
        "ap_row": ap_row,
        "score_policy": score_policy,
        "wta_policy": wta_policy,
        "export_score_mode": export_score_mode,
        "export_min_points_per_object": export_min_points_per_object,
        "scene_rows": scene_rows,
        "failure_rows": failure_rows,
        "evaluator_command": " ".join(cmd),
        "evaluator_exit_code": evaluator_exit_code,
        "metric_file": _rel(metric_file),
        "evaluator_log": _rel(log_path),
        "metrics": metrics or {},
    }


def build_v53_ap_diagnostic(
    *,
    objectlet_summary_path: str | Path = "outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000/local_objectlet_summary.json",
    objectlet_rows_path: str | Path = "outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000/objectlet_rows.csv",
    mask_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv",
    v53_native_carrier_summary_path: str | Path = "outputs/audit/v53_native_carrier_materialization/native_carrier_summary.json",
    output_root: str | Path = "outputs/audit/v53_ap_diagnostic",
    output_config_prefix: str = "v53_local_objectlets_l6_best_legal",
    export_mask_sample_stride: int = 4,
    export_mask_max_pixels: int = 30000,
    export_nn_radius: float = 0.05,
) -> dict[str, Any]:
    output_root_path = _project(output_root)
    summary = _load_json(objectlet_summary_path)
    best = summary.get("best_real_row", {}) if isinstance(summary.get("best_real_row"), dict) else {}
    best_variant = str(summary.get("best_real_variant") or best.get("variant") or "L6_coverage_first_minnew025")
    objectlet_rows = read_csv(_project(objectlet_rows_path)) if _project(objectlet_rows_path).exists() else []
    selected_rows = _selected_objectlet_rows(objectlet_rows, best_variant)
    mask_rows_by_id = _mask_lookup(_project(mask_table_path)) if _project(mask_table_path).exists() else {}
    smoke_gate = ap_smoke_gate(best, len(selected_rows), bool(selected_rows and mask_rows_by_id))
    identity_gate = ap_diagnostic_identity_gate(best)
    native_repair_audit = native_method_export_repair_audit(
        objectlet_rows_path=objectlet_rows_path,
        v53_native_carrier_summary_path=v53_native_carrier_summary_path,
    )

    bridge_specs = [
        {
            "variant": "AP5_v53_local_objectlet_rgbd_pose_mesh_bridge_diagnostic",
            "config": f"{output_config_prefix}_bridge",
            "score_policy": "area_num_backprojected_points",
            "export_score_mode": "area",
            "wta_policy": "none",
            "export_min_points_per_object": 1,
            "export_enable_wta": False,
        },
        {
            "variant": "AP6_v53_local_objectlet_constant_score_min_region_sweep",
            "config": f"{output_config_prefix}_constant_score_min100",
            "score_policy": "constant_score_one_min_region_100",
            "export_score_mode": "one",
            "wta_policy": "none",
            "export_min_points_per_object": 100,
            "export_enable_wta": False,
        },
        {
            "variant": "AP7_v53_local_objectlet_wta_conflict_suppression",
            "config": f"{output_config_prefix}_wta",
            "score_policy": "area_num_backprojected_points",
            "export_score_mode": "area",
            "wta_policy": "point_wta_by_area_reliability",
            "export_min_points_per_object": 1,
            "export_enable_wta": True,
        },
    ]
    bridges: dict[str, dict[str, Any]] = {}
    if smoke_gate["pass"] and identity_gate["pass"]:
        for spec in bridge_specs:
            bridges[spec["variant"]] = _run_bridge_export(
                selected_rows=selected_rows,
                mask_rows_by_id=mask_rows_by_id,
                output_root=output_root_path,
                config=spec["config"],
                ap_row=spec["variant"],
                export_mask_sample_stride=export_mask_sample_stride,
                export_mask_max_pixels=export_mask_max_pixels,
                export_nn_radius=export_nn_radius,
                export_score_mode=spec["export_score_mode"],
                score_policy=spec["score_policy"],
                wta_policy=spec["wta_policy"],
                export_min_points_per_object=spec["export_min_points_per_object"],
                export_enable_wta=spec["export_enable_wta"],
            )

    def bridge_stats(bridge: dict[str, Any]) -> dict[str, Any]:
        scene_rows = bridge.get("scene_rows", [])
        exporter_ok = bool(scene_rows) and all(parse_int(row.get("exporter_exit_code"), 1) == 0 for row in scene_rows)
        evaluator_ok = bridge.get("evaluator_exit_code") == 0
        return {
            "metrics": bridge.get("metrics", {}),
            "scene_rows": scene_rows,
            "mean_pre": float(np.mean([parse_float(row.get("pre_percent")) for row in scene_rows])) if scene_rows else None,
            "mean_union": float(np.mean([parse_float(row.get("union_percent")) for row in scene_rows])) if scene_rows else None,
            "mean_conflict": float(np.mean([parse_float(row.get("export_conflict_rate")) for row in scene_rows])) if scene_rows else None,
            "empty_prediction_rate": sum(1 for row in scene_rows if parse_float(row.get("num_exported_objects")) <= 0.0)
            / max(len(scene_rows), 1),
            "exporter_ok": exporter_ok,
            "evaluator_ok": evaluator_ok,
            "ran": bool(exporter_ok and evaluator_ok),
        }

    stats = {variant: bridge_stats(bridge) for variant, bridge in bridges.items()}

    def bridge_row(spec: dict[str, Any]) -> dict[str, Any]:
        variant = spec["variant"]
        item = stats.get(variant, {})
        metrics = item.get("metrics", {})
        ran = bool(item.get("ran"))
        return {
            "variant": variant,
            "status": "ran" if ran else ("not_run_ap_smoke_or_identity_gate_failed" if not bridges else "failed"),
            "AP": metrics.get("AP"),
            "AP50": metrics.get("AP50"),
            "AP25": metrics.get("AP25"),
            "predictions_per_scene": best.get("mean_predictions_per_scene"),
            "mean_pre_percent": item.get("mean_pre"),
            "mean_union_percent": item.get("mean_union"),
            "mean_export_conflict_rate": item.get("mean_conflict"),
            "empty_prediction_rate": item.get("empty_prediction_rate"),
            "duplicate_prediction_rate": None,
            "min_region_size": spec["export_min_points_per_object"],
            "score_policy": spec["score_policy"],
            "wta_policy": spec["wta_policy"],
            "alignment_policy": "RGB-D_pose_mesh_bridge_diagnostic",
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "exporter_exit_code": 0 if item.get("exporter_ok") else (None if not bridges else 1),
            "evaluator_exit_code": bridges.get(variant, {}).get("evaluator_exit_code"),
        }

    ap3_native_support_available = bool(native_repair_audit.get("v53_native_carrier_support_available"))
    ap3_status = (
        "native_d4rt_carrier_support_materialized_scannet_ap_blocked"
        if ap3_native_support_available
        else "blocked_missing_native_component_to_mesh_materializer"
    )
    ap3_alignment_policy = (
        "v53_component_objectlets_to_d4rt_carrier_support_not_scannet_ap"
        if ap3_native_support_available
        else "native_method_safe_export_not_available_for_v53_component_objectlets"
    )
    ap_rows = [
        {
            "variant": "AP3_v53_local_objectlet_native_export",
            "status": ap3_status,
            "AP": None,
            "AP50": None,
            "AP25": None,
            "predictions_per_scene": best.get("mean_predictions_per_scene"),
            "mean_pre_percent": None,
            "mean_union_percent": None,
            "mean_export_conflict_rate": None,
            "empty_prediction_rate": None,
            "duplicate_prediction_rate": None,
            "min_region_size": 100,
            "score_policy": "local_objectlet_evidence",
            "wta_policy": "none",
            "alignment_policy": ap3_alignment_policy,
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": False,
            "exporter_exit_code": 2,
            "evaluator_exit_code": None,
            "repair_result": native_repair_audit["repair_result"],
            "method_safe_native_support_available": native_repair_audit.get("method_safe_native_support_available"),
            "v53_native_carrier_summary_path": native_repair_audit.get("v53_native_carrier_summary_path"),
            "v53_native_carrier_observation_row_count": native_repair_audit.get("v53_native_carrier_observation_row_count"),
            "v53_native_unique_carrier_count": native_repair_audit.get("v53_native_unique_carrier_count"),
            "v53_native_AP_bridge_status": native_repair_audit.get("v53_native_AP_bridge_status"),
            "required_future_change": native_repair_audit["required_future_change"],
        },
        {
            "variant": "AP4_v53_history_native_export",
            "status": "not_run_history_not_promoted_after_local_control_failure",
            "AP": None,
            "AP50": None,
            "AP25": None,
            "predictions_per_scene": None,
            "mean_pre_percent": None,
            "mean_union_percent": None,
            "mean_export_conflict_rate": None,
            "empty_prediction_rate": None,
            "duplicate_prediction_rate": None,
            "min_region_size": 100,
            "score_policy": "history_objectlet_evidence",
            "wta_policy": "none",
            "alignment_policy": "history_not_run",
            "uses_gt_for_prediction": False,
            "uses_gt_for_evaluation_alignment": False,
            "uses_rgbd_pose_mesh_for_export": False,
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": False,
            "exporter_exit_code": None,
            "evaluator_exit_code": None,
        },
    ] + [bridge_row(spec) for spec in bridge_specs]

    ap_policy_rows = [
        {
            "variant": row["variant"],
            "is_method_result": row["is_method_result"],
            "is_diagnostic_only": row["is_diagnostic_only"],
            "uses_gt_for_prediction": row["uses_gt_for_prediction"],
            "uses_gt_for_evaluation_alignment": row["uses_gt_for_evaluation_alignment"],
            "uses_rgbd_pose_mesh_for_export": row["uses_rgbd_pose_mesh_for_export"],
            "forbidden_for_method_table": row["forbidden_for_method_table"],
            "policy_clean": (not row["uses_gt_for_prediction"]) and (not row["is_method_result"] or not row["forbidden_for_method_table"]),
        }
        for row in ap_rows
    ]
    ap_values = [row for row in ap_rows if row.get("AP") is not None]
    best_ap = max(ap_values, key=lambda row: parse_float(row.get("AP")), default={})
    ap_failure_casebook: list[dict[str, Any]] = []
    ap_scene_rows: list[dict[str, Any]] = []
    for bridge in bridges.values():
        ap_failure_casebook.extend(bridge.get("failure_rows", []))
        ap_scene_rows.extend(bridge.get("scene_rows", []))
    ap_failure_casebook.append(
        {
            "variant": "AP3_v53_local_objectlet_native_export",
            "scene": "ALL",
            "failure_stage": "native_materialization",
            "error": (
                "v53 D4RT carrier support materialized, but ScanNet AP mesh/scene-point masks are still unavailable"
                if ap3_native_support_available
                else "method-safe native component/objectlet-to-mesh materializer unavailable"
            ),
            "repair_result": native_repair_audit["repair_result"],
            "blocked_reason": native_repair_audit["blocked_reason"],
            "required_future_change": native_repair_audit["required_future_change"],
        }
    )
    ap_failure_casebook.append(
        {
            "variant": "AP4_v53_history_native_export",
            "scene": "ALL",
            "failure_stage": "history_not_promoted",
            "error": "history AP not run because Phase 6 local success gate failed controls",
        }
    )
    scene_export_counts = Counter(str(row.get("scene")) for row in selected_rows)
    gate = {
        "ap_smoke_pass": bool(smoke_gate["pass"] and any(stats.get(spec["variant"], {}).get("ran") for spec in bridge_specs)),
        "ap_diagnostic_identity_gate_pass": bool(identity_gate["pass"]),
        "ap_diagnostic_useful": bool(ap_values or ap_failure_casebook),
        "method_safe_ap_available": any(row.get("AP") is not None and not row.get("forbidden_for_method_table") for row in ap_rows),
        "method_safe_native_support_available": bool(native_repair_audit.get("method_safe_native_support_available")),
        "v53_native_carrier_support_available": bool(native_repair_audit.get("v53_native_carrier_support_available")),
        "native_method_export_repair_attempted": bool(native_repair_audit.get("repair_attempted")),
        "native_method_export_repair_result": native_repair_audit["repair_result"],
        "rgbd_bridge_ap_ran": bool(stats.get("AP5_v53_local_objectlet_rgbd_pose_mesh_bridge_diagnostic", {}).get("ran")),
        "ap6_constant_score_min_region_ran": bool(stats.get("AP6_v53_local_objectlet_constant_score_min_region_sweep", {}).get("ran")),
        "ap7_wta_conflict_suppression_ran": bool(stats.get("AP7_v53_local_objectlet_wta_conflict_suppression", {}).get("ran")),
        "required_bridge_rows_ran": bool(bridges) and all(stats.get(spec["variant"], {}).get("ran") for spec in bridge_specs),
        "pass": bool(bridges) and all(stats.get(spec["variant"], {}).get("ran") for spec in bridge_specs),
    }
    return {
        "phase": "v53_ap_diagnostic",
        "created_at": utc_now(),
        "source_objectlet_summary": _rel(objectlet_summary_path),
        "source_objectlet_rows": _rel(objectlet_rows_path),
        "source_mask_table": _rel(mask_table_path),
        "best_local_variant": best_variant,
        "selected_object_count": len(selected_rows),
        "selected_object_count_by_scene": dict(scene_export_counts),
        "ap_smoke_gate": smoke_gate,
        "ap_diagnostic_identity_gate": identity_gate,
        "summary": {
            "gate": gate,
            "best_AP": best_ap.get("AP"),
            "best_AP50": best_ap.get("AP50"),
            "best_AP25": best_ap.get("AP25"),
            "best_AP_variant": best_ap.get("variant"),
            "ap_row_count": len(ap_rows),
            "ap_metric_row_count": len(ap_values),
            "metric_scope": (
                "AP5/AP6/AP7 are RGB-D/pose/mesh bridge diagnostics; AP3 has method-safe D4RT carrier support but no ScanNet AP mask"
                if ap3_native_support_available
                else "AP5/AP6/AP7 are RGB-D/pose/mesh bridge diagnostics; AP3 method-safe native export is blocked"
            ),
            "native_method_export_repair_result": native_repair_audit["repair_result"],
            "method_safe_native_support_available": native_repair_audit.get("method_safe_native_support_available"),
            "v53_native_carrier_support_available": native_repair_audit.get("v53_native_carrier_support_available"),
            "v53_native_carrier_observation_row_count": native_repair_audit.get("v53_native_carrier_observation_row_count"),
            "v53_native_unique_carrier_count": native_repair_audit.get("v53_native_unique_carrier_count"),
            "native_method_export_required_future_change": native_repair_audit["required_future_change"],
        },
        "gate": gate,
        "ap_rows": ap_rows,
        "ap_policy_rows": ap_policy_rows,
        "ap_failure_casebook": ap_failure_casebook,
        "ap_scene_rows": ap_scene_rows,
        "native_method_export_repair_audit": native_repair_audit,
        "bridge_details": bridges,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def write_v53_ap_diagnostic(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "ap_export_summary.json", payload)
    write_csv(out / "ap_metric_rows.csv", payload["ap_rows"])
    write_csv(out / "ap_policy_rows.csv", payload["ap_policy_rows"])
    write_csv(out / "ap_failure_casebook.csv", payload["ap_failure_casebook"])
    write_csv(out / "ap_scene_rows.csv", payload["ap_scene_rows"])
