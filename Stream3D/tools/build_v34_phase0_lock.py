from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLAN_PATH = "docs/stream4d_v34_3d_object_identity_first_plan.md"

SOURCE_FILES = {
    "v23_pointcloud": "outputs/audit/v23_pointcloud/pointcloud_summary.json",
    "v25_guard": "outputs/audit/v25_real_geometry_flow_d5/geometry_flow_summary.json",
    "v26_coverage": "outputs/audit/v26_phaseC_coverage_d5_probe5_topup20_patch2_window0/d5_probe5_topup20_patch2_window0_summary.json",
    "v26_object_quality": "outputs/audit/v26_object_quality_d5_topup20_patch2_window0/d5_topup20_patch2_window0_summary.json",
    "v26_object_quality_alpha3": "outputs/audit/v26_object_quality_d5_topup20_patch2_window0_alpha3/d5_topup20_patch2_window0_alpha3_summary.json",
    "v26_object_quality_negfix": "outputs/audit/v26_object_quality_d5_topup20_patch2_window0_negfix_alpha3/d5_topup20_patch2_window0_negfix_alpha3_summary.json",
    "v27_pair_attribution": "outputs/audit/v27_pair_attribution/v27_pair_attribution_category_summary.json",
    "v27_partition": "outputs/audit/v27_signed_pair_partition_r4/v27_signed_pair_partition_r4_summary.json",
    "v27_local_core_memory": "outputs/audit/v27_local_core_memory_r5/v27_local_core_memory_r5_summary.json",
    "v30_phaseA": "outputs/audit/v30_phaseA_lock/phaseA_lock_manifest.json",
    "v31_decomposition": "outputs/audit/v31_decomposition_oracle/decomposition_summary.csv",
    "v31_solver_summary": "outputs/audit/v31_slot_ownership/solver_summary.csv",
    "v31_negative_prune": "outputs/audit/v31_slot_ownership/r11_e8_negative_prune_real_summary.csv",
    "v31_graph_split": "outputs/audit/v31_slot_ownership/r13_e8_graph_split_real_summary.csv",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _find_json_row(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get(key)) == value:
            return row
    raise ValueError(f"missing row {key}={value}")


def _safe_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in ("", None):
        return None
    return float(value)


def _parse_unittest_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    ran_match = re.search(r"Ran\s+(\d+)\s+tests?", text)
    skip_match = re.search(r"OK\s+\(skipped=(\d+)\)", text)
    return {
        "py_compile_pass": "## py_compile" in text and "Traceback" not in text,
        "unittest_pass": re.search(r"\nOK(?:\s+\(skipped=\d+\))?\n", text) is not None,
        "unittest_test_count": int(ran_match.group(1)) if ran_match else None,
        "unittest_skipped_count": int(skip_match.group(1)) if skip_match else 0,
        "log_path": str(path),
    }


def _metric_pack(row: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for out_key, in_key in mapping.items():
        out[out_key] = _safe_float(row, in_key)
    return out


def _aggregate_rows(rows: list[dict[str, str]], *, exclude_gt_oracle: bool = True) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("scene") in ("", "ALL", None):
            continue
        if exclude_gt_oracle and row.get("variant") == "D6_GT_full_oracle_forbidden":
            continue
        control_kind = row.get("control_kind") or "real"
        if control_kind != "real":
            continue
        if not row.get("ARI"):
            continue
        name = row.get("solver") or row.get("variant") or "unknown"
        grouped[(name, control_kind, row.get("source_file", ""))].append(row)
    out: list[dict[str, Any]] = []
    for (name, control_kind, source_file), items in grouped.items():
        if len(items) < 5:
            continue
        def mean(key: str) -> float | None:
            vals: list[float] = []
            for item in items:
                value = item.get(key)
                if value in ("", None):
                    continue
                vals.append(float(value))
            return float(sum(vals) / len(vals)) if vals else None

        scene0081 = next((item for item in items if item.get("scene") == "scene0081_01"), None)
        out.append(
            {
                "name": name,
                "control_kind": control_kind,
                "source_file": source_file,
                "scene_count": len(items),
                "ARI": mean("ARI"),
                "purity": mean("purity"),
                "completeness": mean("completeness"),
                "unknown_tube_ratio": mean("unknown_tube_ratio"),
                "scene0081_ARI": float(scene0081["ARI"]) if scene0081 is not None else None,
            }
        )
    return sorted(out, key=lambda row: float(row.get("ARI") or -1.0), reverse=True)


def _load_v31_best(source_paths: list[Path]) -> dict[str, Any]:
    all_rows: list[dict[str, str]] = []
    for path in source_paths:
        for row in _read_csv(path):
            row = dict(row)
            row["source_file"] = str(path)
            all_rows.append(row)
    aggregated = _aggregate_rows(all_rows)
    if not aggregated:
        raise ValueError("missing v31 aggregated rows")
    best = aggregated[0]
    best_non_oracle = next(
        row for row in aggregated if "decomposition_oracle" not in row["source_file"]
    )
    return {
        "best_overall_diagnostic": best,
        "best_non_gt_solver": best_non_oracle,
        "top_rows": aggregated[:10],
    }


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_lock(stream3d_root: Path, out_dir: Path, package_log: Path) -> dict[str, Any]:
    sources = {name: stream3d_root / rel for name, rel in SOURCE_FILES.items()}
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required baseline files: " + ", ".join(missing))

    point_rows = _read_json(sources["v23_pointcloud"])
    p5 = _find_json_row(point_rows, "variant", "P5")
    p6 = _find_json_row(point_rows, "variant", "P6")
    guard = _read_json(sources["v25_guard"])
    coverage = _read_json(sources["v26_coverage"])
    oq = _read_json(sources["v26_object_quality"])
    oq_alpha3 = _read_json(sources["v26_object_quality_alpha3"])
    oq_negfix = _read_json(sources["v26_object_quality_negfix"])

    pair_rows = _read_json(sources["v27_pair_attribution"])
    same_mask = next(row for row in pair_rows if row.get("scene") == "ALL" and row.get("category") == "B0_same_frame_same_mask")

    partition_rows = _read_json(sources["v27_partition"])
    partition_by_variant = {row["variant"]: row for row in partition_rows}
    memory_rows = _read_json(sources["v27_local_core_memory"])
    memory_by_variant = {row["variant"]: row for row in memory_rows}

    v30 = _read_json(sources["v30_phaseA"])
    v31 = _load_v31_best(
        [
            sources["v31_decomposition"],
            sources["v31_solver_summary"],
            sources["v31_negative_prune"],
            sources["v31_graph_split"],
        ]
    )
    validation = _parse_unittest_log(package_log)

    lock = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": PLAN_PATH,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "phase0_validation": {
            **validation,
            "original_python_command_pass": False,
            "original_python_command_failure": "python command not found in current environment",
            "environment_repairs": [
                "Installed user-site numpy scipy networkx opencv-python-headless scikit-learn.",
                "Installed user-site matplotlib.",
            ],
            "code_repairs": [
                "Made stream4d/d4rt_adapter.py tolerate missing torch at module import; D4RTAdapter construction still requires torch.",
                "Made tools/diagnose_v22_loger_scale_proxy.py tolerate missing torch at module import; loading .pt outputs still requires torch.",
            ],
            "optional_runtime_unavailable": {
                "torch": True,
                "open3d": True,
                "cuda_for_python_env": True,
            },
        },
        "source_files": {name: str(path) for name, path in sources.items()},
        "source_hashes": {name: _sha256(path) for name, path in sources.items()},
        "current_state": {
            "v23_geometry_upper_bound": {
                "P5_xyz_ref0_ref0_pose_eval_scale": _metric_pack(p5, {"F@10cm": "F@10cm", "F@20cm": "F@20cm"}),
                "P6_full_eval_sim3": _metric_pack(p6, {"F@10cm": "F@10cm", "F@20cm": "F@20cm"}),
                "interpretation": "D4RT geometry has a useful evaluation upper bound; not treated as the v34 main blocker.",
            },
            "v25_geometry_guard": {
                "cross_chunk_canonical_merge_reads": guard["cross_chunk_canonical_merge_reads"],
                "unguarded_metric_geometry_reads": guard["unguarded_metric_geometry_reads"],
                "unexpected_metric_read_events": guard["unexpected_metric_read_events"],
                "scale_sensitive_metric_reads": guard["scale_sensitive_metric_reads"],
                "weak_alignment_blocked": guard["weak_alignment_blocked"],
            },
            "v26_coverage": {
                "phase_c_pass": coverage["phase_c_pass"],
                "mask_interior_coverage_mean_mean": coverage["mask_interior_coverage_mean_mean"],
                "large_masks_with_ge8_boundary_tubes_ratio_mean": coverage["large_masks_with_ge8_boundary_tubes_ratio_mean"],
                "node_gt_label_coverage_mean": coverage["node_gt_label_coverage_mean"],
                "covered_GT_instance_ratio_mean": coverage["covered_GT_instance_ratio_mean"],
            },
            "v26_object_quality": {
                "window0": _metric_pack(
                    oq,
                    {
                        "real_partition_ARI": "real_partition_ari_mean",
                        "real_partition_purity": "real_partition_purity_mean",
                        "real_false_merge_rate": "real_false_merge_rate_mean",
                        "real_edge_AUC": "real_edge_auc_mean",
                    },
                ),
                "window0_alpha3": _metric_pack(
                    oq_alpha3,
                    {
                        "real_partition_ARI": "real_partition_ari_mean",
                        "real_partition_purity": "real_partition_purity_mean",
                        "real_false_merge_rate": "real_false_merge_rate_mean",
                        "real_edge_AUC": "real_edge_auc_mean",
                    },
                ),
                "window0_negfix_alpha3": _metric_pack(
                    oq_negfix,
                    {
                        "negative_majority_partition_ARI": "negative_majority_partition_ari_mean",
                        "negative_majority_partition_purity": "negative_majority_partition_purity_mean",
                        "negative_majority_false_merge_rate": "negative_majority_false_merge_rate_mean",
                    },
                ),
            },
            "v27_same_mask_mixed_signal": {
                "same_GT_ratio": same_mask["same_GT_ratio"],
                "different_GT_ratio": same_mask["different_GT_ratio"],
                "false_merge_rate": same_mask["false_merge_rate"],
                "pair_count": same_mask["pair_count"],
            },
            "v27_partition_and_memory": {
                "S5_window0_positive_only": _metric_pack(
                    partition_by_variant["S5_window0_positive_only"],
                    {"ARI": "ari_mean", "purity": "purity_mean", "false_merge_rate": "false_merge_rate_mean"},
                ),
                "S6_merge_scope_positive_only": _metric_pack(
                    partition_by_variant["S6_merge_scope_positive_only"],
                    {"ARI": "ari_mean", "purity": "purity_mean", "false_merge_rate": "false_merge_rate_mean"},
                ),
                "best_local_core_memory_H2": _metric_pack(
                    memory_by_variant["H2_object_memory_signed"],
                    {
                        "ARI": "ari_mean",
                        "purity": "purity_mean",
                        "completeness": "completeness_mean",
                        "false_merge_rate": "false_merge_rate_mean",
                    },
                ),
            },
            "v30_oracle_and_solver": {
                "real_O5_oracle": {
                    "ARI": v30["metrics"]["real_O5_ARI"],
                    "purity": v30["metrics"]["real_O5_purity"],
                    "completeness": v30["metrics"]["real_O5_completeness"],
                    "scene0081_ARI": v30["metrics"]["real_O5_scene0081_ARI"],
                },
                "best_non_gt_continuation": {
                    "solver": v30["metrics"]["best_continuation_solver"],
                    "ARI": v30["metrics"]["best_continuation_ARI"],
                    "purity": v30["metrics"]["best_continuation_purity"],
                    "completeness": v30["metrics"]["best_continuation_completeness"],
                    "scene0081_ARI": v30["metrics"]["best_continuation_scene0081_ARI"],
                },
                "controls": {
                    "shuffled_best_ARI": v30["metrics"]["shuffled_best_ARI"],
                    "no_temporal_best_ARI": v30["metrics"]["no_temporal_best_ARI"],
                    "mask_only_best_ARI": v30["metrics"]["mask_only_best_ARI"],
                },
            },
            "v31_oracle_and_solver": v31,
        },
    }

    gates = {
        "py_compile_pass": bool(lock["phase0_validation"]["py_compile_pass"]),
        "unittest_pass": bool(lock["phase0_validation"]["unittest_pass"]),
        "required_baseline_files_present": not missing,
        "v23_geometry_loaded": p5["F@10cm"] is not None and p6["F@20cm"] is not None,
        "v25_guard_loaded": guard["unguarded_metric_geometry_reads"] == 0,
        "v26_coverage_loaded": bool(coverage["phase_c_pass"]),
        "v27_mixed_signal_loaded": same_mask["different_GT_ratio"] is not None,
        "v30_v31_solver_failure_loaded": v30["metrics"]["best_continuation_ARI"] is not None
        and v31["best_non_gt_solver"]["ARI"] is not None,
    }
    lock["phase0_gates"] = gates
    lock["all_required_baseline_metrics_loaded"] = all(gates.values())

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "current_state_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")

    table_rows = [
        {"item": "py_compile_pass", "value": gates["py_compile_pass"], "source": str(package_log)},
        {"item": "unittest_pass", "value": gates["unittest_pass"], "source": str(package_log)},
        {"item": "unittest_test_count", "value": validation["unittest_test_count"], "source": str(package_log)},
        {"item": "v23_P5_F@10cm", "value": p5["F@10cm"], "source": str(sources["v23_pointcloud"])},
        {"item": "v23_P5_F@20cm", "value": p5["F@20cm"], "source": str(sources["v23_pointcloud"])},
        {"item": "v23_P6_F@10cm", "value": p6["F@10cm"], "source": str(sources["v23_pointcloud"])},
        {"item": "v23_P6_F@20cm", "value": p6["F@20cm"], "source": str(sources["v23_pointcloud"])},
        {"item": "v25_unguarded_metric_geometry_reads", "value": guard["unguarded_metric_geometry_reads"], "source": str(sources["v25_guard"])},
        {"item": "v25_scale_sensitive_metric_reads", "value": guard["scale_sensitive_metric_reads"], "source": str(sources["v25_guard"])},
        {"item": "v26_interior_coverage", "value": coverage["mask_interior_coverage_mean_mean"], "source": str(sources["v26_coverage"])},
        {"item": "v26_boundary_ge8_ratio", "value": coverage["large_masks_with_ge8_boundary_tubes_ratio_mean"], "source": str(sources["v26_coverage"])},
        {"item": "v26_node_gt_coverage", "value": coverage["node_gt_label_coverage_mean"], "source": str(sources["v26_coverage"])},
        {"item": "v27_same_mask_same_GT_ratio", "value": same_mask["same_GT_ratio"], "source": str(sources["v27_pair_attribution"])},
        {"item": "v27_same_mask_diff_GT_ratio", "value": same_mask["different_GT_ratio"], "source": str(sources["v27_pair_attribution"])},
        {"item": "v27_window0_positive_only_ARI", "value": partition_by_variant["S5_window0_positive_only"]["ari_mean"], "source": str(sources["v27_partition"])},
        {"item": "v30_real_O5_oracle_ARI", "value": v30["metrics"]["real_O5_ARI"], "source": str(sources["v30_phaseA"])},
        {"item": "v30_best_non_gt_ARI", "value": v30["metrics"]["best_continuation_ARI"], "source": str(sources["v30_phaseA"])},
        {"item": "v30_best_non_gt_purity", "value": v30["metrics"]["best_continuation_purity"], "source": str(sources["v30_phaseA"])},
        {"item": "v31_best_diagnostic_ARI", "value": v31["best_overall_diagnostic"]["ARI"], "source": v31["best_overall_diagnostic"]["source_file"]},
        {"item": "v31_best_non_gt_solver_ARI", "value": v31["best_non_gt_solver"]["ARI"], "source": v31["best_non_gt_solver"]["source_file"]},
    ]
    _write_table(out_dir / "current_state_table.md", table_rows)
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stream4D v34 Phase 0 current-state lock.")
    parser.add_argument("--stream3d-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/audit/v34_phase0"))
    parser.add_argument(
        "--package-log",
        type=Path,
        default=Path("outputs/audit/v34_phase0/package_validation_python3_after_optional_torch_patches.log"),
    )
    args = parser.parse_args()
    lock = build_lock(args.stream3d_root.resolve(), args.out_dir, args.package_log)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "all_required_baseline_metrics_loaded": lock["all_required_baseline_metrics_loaded"],
                "phase0_gates": lock["phase0_gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
