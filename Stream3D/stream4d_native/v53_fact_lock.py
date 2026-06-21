from __future__ import annotations

from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _exists(path: str | Path) -> bool:
    return _project(path).exists()


def _path_row(key: str, path: str | Path, note: str = "", required: bool = True) -> dict[str, Any]:
    exists = _exists(path)
    return {
        "key": key,
        "value": bool(exists),
        "available": bool(exists),
        "required": bool(required),
        "source": _rel(path),
        "note": note,
    }


def _value_row(
    key: str,
    value: Any,
    source: str | Path,
    note: str = "",
    required: bool = True,
) -> dict[str, Any]:
    return {
        "key": key,
        "value": value,
        "available": value not in (None, ""),
        "required": bool(required),
        "source": _rel(source),
        "note": note,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    path_obj = _project(path)
    if not path_obj.exists():
        return {}
    payload = read_json(path_obj)
    return payload if isinstance(payload, dict) else {}


def _scene_mask_count(scene: str, source: str) -> int:
    mask_root = ROOT / "data/scannet/processed" / scene / source / "mask"
    if not mask_root.exists():
        return 0
    return sum(1 for _path in mask_root.glob("*.png"))


def build_v53_fact_lock(
    scenes: list[str] | None = None,
    carrier_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv",
    mask_table_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/mask_observation_table.csv",
    observation_summary_path: str | Path = "outputs/audit/v47_observation_tables_metricfix/observation_table_summary.json",
    component_summary_path: str | Path = "outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix/carrier_supertrack_summary.json",
) -> dict[str, Any]:
    scenes = scenes or ["scene0011_00", "scene0030_00", "scene0050_00", "scene0081_01", "scene0591_00"]
    observation_summary = _load_json(observation_summary_path)
    component_summary = _load_json(component_summary_path)

    cropformer_counts = {scene: _scene_mask_count(scene, "output_Cropformer") for scene in scenes}
    sam2_counts = {scene: _scene_mask_count(scene, "output_SAM2") for scene in scenes}
    sam_counts = {scene: _scene_mask_count(scene, "output_SAM") for scene in scenes}
    cropformer_total = sum(cropformer_counts.values())
    sam2_total = sum(sam2_counts.values())
    sam_total = sum(sam_counts.values())

    fact_rows: list[dict[str, Any]] = [
        _path_row("carrier_observation_table_available", carrier_table_path, "v47 carrier observation table"),
        _path_row("mask_observation_table_available", mask_table_path, "v47 CropFormer mask observation table"),
        _path_row("U32_components_available", component_summary_path, "v47 union_32_fine_metricfix component summary"),
        _value_row(
            "D4RT_encoder_stride_eq_1",
            observation_summary.get("D4RT_encoder_stride") == 1,
            observation_summary_path,
            f"D4RT_encoder_stride={observation_summary.get('D4RT_encoder_stride')}",
        ),
        _value_row(
            "scale_guard_pass",
            int(observation_summary.get("scale_weak_row_count", -1)) == 0
            and float(observation_summary.get("allow_metric_relation_ratio", 0.0)) >= 0.99,
            observation_summary_path,
            "scale_weak_row_count=0 and allow_metric_relation_ratio>=0.99 required for metric relations",
        ),
        _value_row(
            "cropformer_mask_source_available",
            cropformer_total > 0,
            ROOT / "data/scannet/processed",
            f"selected scene CropFormer PNG count={cropformer_total}",
        ),
        _path_row("ap_exporter_available", "stream4d/export_scannet.py", "generic ScanNet exporter source"),
        _path_row("ap_evaluator_available", "evaluation/evaluate.py", "ScanNet evaluator source"),
        _value_row(
            "stream3d_current_mask_source",
            "output_Cropformer/mask",
            "dataset/scannet.py",
            "Stream3D run.py enables CropFormer and comments SAM2; main.py hardcodes Cropformer",
        ),
        _value_row(
            "selected_scene_sam2_png_count",
            sam2_total,
            ROOT / "data/scannet/processed",
            "SAM2 is supported by code path but absent in selected ScanNet processed scenes",
            required=False,
        ),
        _value_row(
            "selected_scene_sam_png_count",
            sam_total,
            ROOT / "data/scannet/processed",
            "SAM is absent in selected ScanNet processed scenes",
            required=False,
        ),
    ]

    gate = {
        row["key"]: bool(row["value"]) if row["required"] else True
        for row in fact_rows
        if row["key"]
        in {
            "carrier_observation_table_available",
            "mask_observation_table_available",
            "U32_components_available",
            "D4RT_encoder_stride_eq_1",
            "scale_guard_pass",
            "cropformer_mask_source_available",
            "ap_exporter_available",
            "ap_evaluator_available",
        }
    }
    gate["pass"] = bool(all(gate.values()))

    ap_contract = {
        "phase": "v53_ap_contract",
        "method_claim_requires_strict_identity_gate": True,
        "ap_smoke_can_run_before_strict_gate": True,
        "diagnostic_rows_must_set_forbidden_for_method_table": True,
        "uses_gt_for_prediction_allowed": False,
        "uses_rgbd_pose_mesh_for_method_prediction_allowed": False,
        "created_at": utc_now(),
    }
    visualization_manifest = {
        "phase": "v53_visualization_manifest",
        "root": "outputs/audit/v53_visualizations",
        "directories": [
            "outputs/audit/v53_visualizations/local_objectlets",
            "outputs/audit/v53_visualizations/reprojection",
            "outputs/audit/v53_visualizations/history",
            "outputs/audit/v53_visualizations/semantic",
            "outputs/audit/v53_visualizations/ap",
            "outputs/audit/v53_visualizations/casebook",
        ],
        "created_at": utc_now(),
    }
    return {
        "summary": {
            "phase": "v53_fact_lock",
            "created_at": utc_now(),
            "scenes": scenes,
            "cropformer_png_counts": cropformer_counts,
            "sam2_png_counts": sam2_counts,
            "sam_png_counts": sam_counts,
            "cropformer_png_count": cropformer_total,
            "sam2_png_count": sam2_total,
            "sam_png_count": sam_total,
            "component_count_from_v47_summary": component_summary.get("component_count"),
            "carrier_row_count": observation_summary.get("carrier_row_count"),
            "mask_count": observation_summary.get("mask_count"),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
        "fact_rows": fact_rows,
        "gate": gate,
        "ap_contract": ap_contract,
        "visualization_manifest": visualization_manifest,
    }


def write_v53_fact_lock(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    for directory in payload["visualization_manifest"]["directories"]:
        _project(directory).mkdir(parents=True, exist_ok=True)
    write_json(out / "fact_lock.json", {**payload["summary"], "gate": payload["gate"]})
    write_csv(out / "fact_lock_rows.csv", payload["fact_rows"])
    write_json(out / "ap_contract.json", payload["ap_contract"])
    write_json(ROOT / "outputs/audit/v53_visualizations/visualization_manifest.json", payload["visualization_manifest"])


__all__ = ["build_v53_fact_lock", "write_v53_fact_lock"]
