from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_by_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    matches = [row for row in rows if str(row.get("variant")) == variant]
    if len(matches) != 1:
        raise ValueError(f"expected one row for variant={variant!r}, found {len(matches)}")
    return matches[0]


def _float_or_none(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _metric_row(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"variant": row.get("variant")}
    for field in fields:
        out[field] = _float_or_none(row, field)
    if "scene_count" in row:
        out["scene_count"] = int(float(row["scene_count"]))
    return out


def _source_record(root: Path, rel_path: str) -> dict[str, str]:
    path = root / rel_path
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": rel_path, "sha256": _sha256(path)}


def build_lock(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve()
    sources = {
        "v26_full_alpha3_summary": "Stream3D/outputs/audit/v26_object_quality_d5_topup20_patch2_alpha3/d5_topup20_patch2_alpha3_summary.json",
        "v26_window0_alpha3_summary": "Stream3D/outputs/audit/v26_object_quality_d5_topup20_patch2_window0_alpha3/d5_topup20_patch2_window0_alpha3_summary.json",
        "v27_partition_summary": "Stream3D/outputs/audit/v27_signed_pair_partition_r4/v27_signed_pair_partition_r4_summary.json",
        "v27_local_memory_summary": "Stream3D/outputs/audit/v27_local_core_memory_r5/v27_local_core_memory_r5_summary.json",
        "v27_highbudget_partition_summary": "Stream3D/outputs/audit/v27_signed_pair_partition_scene0050_highbudget_r3/v27_signed_pair_partition_scene0050_highbudget_r3_summary.json",
        "v27_highbudget_local_memory_summary": "Stream3D/outputs/audit/v27_local_core_memory_scene0050_highbudget_r2/v27_local_core_memory_scene0050_highbudget_r2_summary.json",
        "v27_measurement_manifest": "Stream3D/outputs/audit/v27_measurements_by_scene_probe5/v27_measurements_by_scene_probe5_manifest.json",
    }

    v26_full = _read_json(root / sources["v26_full_alpha3_summary"])
    v26_window0 = _read_json(root / sources["v26_window0_alpha3_summary"])
    v27_partition = _read_json(root / sources["v27_partition_summary"])
    v27_local_memory = _read_json(root / sources["v27_local_memory_summary"])
    v27_highbudget_partition = _read_json(root / sources["v27_highbudget_partition_summary"])
    v27_highbudget_local_memory = _read_json(root / sources["v27_highbudget_local_memory_summary"])
    v27_measurement_manifest = _read_json(root / sources["v27_measurement_manifest"])

    partition_fields = [
        "ari_mean",
        "purity_mean",
        "false_merge_rate_mean",
        "largest_component_ratio_mean",
        "kept_edge_count_mean",
        "overmerge_count_mean",
        "oversplit_count_mean",
    ]
    local_memory_fields = [
        "ari_mean",
        "purity_mean",
        "completeness_mean",
        "false_merge_rate_mean",
        "object_chunk_span_mean",
        "memory_merge_count_mean",
        "overmerge_count_mean",
        "oversplit_count_mean",
    ]

    lock: dict[str, Any] = {
        "schema_version": 1,
        "plan": "docs/stream4d_v28_mask_region_proposal_ownership_plan.md",
        "created_by": "tools.build_v28_phaseA_baseline_lock",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": True,
        "geometry_field": "D4RT canonical tube diagnostics from v26/v27 artifacts",
        "coordinate_frame": "diagnostic summary rows; no method output",
        "alignment_source": "v26/v27 diagnostic artifact provenance",
        "phaseA_validation": {
            "current_source_py_compile_pass": bool(args.current_source_py_compile_pass),
            "current_source_unittest_pass": bool(args.current_source_unittest_pass),
            "clean_zip_py_compile_pass": bool(args.clean_zip_py_compile_pass),
            "clean_zip_unittest_pass": bool(args.clean_zip_unittest_pass),
            "missing_module_count": int(args.missing_module_count),
            "test_count": int(args.test_count),
            "v27_baseline_metrics_loaded": True,
        },
        "source_files": {name: _source_record(root, rel) for name, rel in sources.items()},
        "v26_d5_patch2_full_alpha3": {
            "ari_mean": float(v26_full["real_partition_ari_mean"]),
            "purity_mean": float(v26_full["real_partition_purity_mean"]),
            "completeness_mean": None,
            "false_merge_rate_mean": float(v26_full["real_false_merge_rate_mean"]),
            "edge_auc_mean": float(v26_full["real_edge_auc_mean"]),
            "shuffle_edge_auc_mean": float(v26_full["shuffle_edge_auc_mean"]),
            "scene_count": int(v26_full["scene_count"]),
            "missing_metrics": {
                "completeness_mean": "not present in v26 object-quality summary or partition rows; not inferred",
            },
        },
        "v26_d5_patch2_window0_alpha3": {
            "ari_mean": float(v26_window0["real_partition_ari_mean"]),
            "purity_mean": float(v26_window0["real_partition_purity_mean"]),
            "completeness_mean": None,
            "false_merge_rate_mean": float(v26_window0["real_false_merge_rate_mean"]),
            "edge_auc_mean": float(v26_window0["real_edge_auc_mean"]),
            "shuffle_edge_auc_mean": float(v26_window0["shuffle_edge_auc_mean"]),
            "scene_count": int(v26_window0["scene_count"]),
            "missing_metrics": {
                "completeness_mean": "not present in v26 object-quality summary or partition rows; not inferred",
            },
        },
        "v27_pair_attribution": {
            "dominant_false_merge_source": v27_measurement_manifest.get("dominant_false_merge_source"),
            "pair_row_count": v27_measurement_manifest.get("pair_row_count"),
            "scene_output_count": v27_measurement_manifest.get("scene_output_count"),
            "phase_b_complete": v27_measurement_manifest.get("phase_b_complete"),
            "phase_c_boundary_negative_fields_complete": v27_measurement_manifest.get(
                "phase_c_boundary_negative_fields_complete"
            ),
            "phase_c_appearance_motion_fields_complete": v27_measurement_manifest.get(
                "phase_c_appearance_motion_fields_complete"
            ),
        },
        "v27_sampled_partition": {
            variant: _metric_row(_row_by_variant(v27_partition, variant), partition_fields)
            for variant in [
                "S0_positive_only",
                "S4_soft_signed_score",
                "S5_window0_positive_only",
            ]
        },
        "v27_local_core_memory": {
            variant: _metric_row(_row_by_variant(v27_local_memory, variant), local_memory_fields)
            for variant in [
                "L0_full_positive_cc",
                "L0_window0_positive_cc",
                "L2_local_strict_cores",
                "H2_object_memory_signed",
                "H5_memory_appearance_motion",
            ]
        },
        "v27_scene0050_highbudget_partition": {
            variant: _metric_row(_row_by_variant(v27_highbudget_partition, variant), partition_fields)
            for variant in [
                "S0_positive_only",
                "S3_negative_majority_veto",
                "S5_window0_positive_only",
            ]
        },
        "v27_scene0050_highbudget_local_core_memory": {
            variant: _metric_row(_row_by_variant(v27_highbudget_local_memory, variant), local_memory_fields)
            for variant in [
                "L0_full_positive_cc",
                "L0_window0_positive_cc",
                "L2_local_strict_cores",
                "H2_object_memory_signed",
                "H5_memory_appearance_motion",
            ]
        },
    }

    payload = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode("utf-8")
    lock["baseline_hash"] = hashlib.sha256(payload).hexdigest()
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--current-source-py-compile-pass", action="store_true")
    parser.add_argument("--current-source-unittest-pass", action="store_true")
    parser.add_argument("--clean-zip-py-compile-pass", action="store_true")
    parser.add_argument("--clean-zip-unittest-pass", action="store_true")
    parser.add_argument("--missing-module-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=0)
    args = parser.parse_args()

    lock = build_lock(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(output), "baseline_hash": lock["baseline_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
