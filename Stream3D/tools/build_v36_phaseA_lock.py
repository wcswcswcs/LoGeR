from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLAN_PATH = "docs/stream4d_v36_masklet_first_object_identity_plan.md"

LOCAL_GATE = {
    "ARI": 0.40,
    "purity": 0.85,
    "completeness": 0.50,
    "unknown_tube_ratio_max": 0.40,
    "scene0081_ARI": 0.20,
}


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_from_file(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        return int(text.splitlines()[-1].strip())
    except ValueError:
        return None


def _log_tail(path: Path, max_chars: int = 4000) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _parse_unittest_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    ran_match = re.search(r"Ran\s+(\d+)\s+tests?", text)
    failed = bool(re.search(r"\nFAILED\s+\(", text))
    errored = "Traceback (most recent call last)" in text or "\nERROR:" in text
    ok = re.search(r"\nOK(?:\s+\([^)]*\))?\s*$", text) is not None
    skipped_match = re.search(r"skipped=(\d+)", text)
    return {
        "unittest_log_exists": path.exists(),
        "unittest_test_count": int(ran_match.group(1)) if ran_match else None,
        "unittest_skipped_count": int(skipped_match.group(1)) if skipped_match else None,
        "unittest_log_ok_marker": bool(ok),
        "unittest_log_failed_marker": bool(failed),
        "unittest_log_error_marker": bool(errored),
    }


def _validation_status(out_dir: Path) -> dict[str, Any]:
    py_log = out_dir / "py_compile.log"
    py_exit = _int_from_file(out_dir / "py_compile.exit_code")
    unit_log = out_dir / "unittest.log"
    unit_exit = _int_from_file(out_dir / "unittest.exit_code")
    unittest_info = _parse_unittest_log(unit_log)
    py_text = py_log.read_text(encoding="utf-8", errors="replace") if py_log.exists() else ""
    py_has_error = bool(py_text.strip())
    return {
        "py_compile": {
            "log_path": str(py_log),
            "exit_code_path": str(out_dir / "py_compile.exit_code"),
            "log_exists": py_log.exists(),
            "exit_code": py_exit,
            "log_bytes": py_log.stat().st_size if py_log.exists() else None,
            "pass": py_exit == 0 and not py_has_error,
            "tail": _log_tail(py_log),
        },
        "unittest": {
            "log_path": str(unit_log),
            "exit_code_path": str(out_dir / "unittest.exit_code"),
            "exit_code": unit_exit,
            "pass": unit_exit == 0 and bool(unittest_info["unittest_log_ok_marker"]),
            "tail": _log_tail(unit_log),
            **unittest_info,
        },
    }


def _find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str] | None:
    for row in rows:
        ok = True
        for key, value in criteria.items():
            if str(row.get(key)) != str(value):
                ok = False
                break
        if ok:
            return row
    return None


def _metric_row(row: dict[str, Any] | None, mapping: dict[str, str]) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: _float(row.get(col)) for key, col in mapping.items()}


def _best_row(rows: list[dict[str, str]], metric: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(metric) not in ("", None)]
    if not candidates:
        return None
    row = max(candidates, key=lambda item: float(item[metric]))
    out: dict[str, Any] = {}
    for key, value in row.items():
        out[key] = _float(value)
        if out[key] is None:
            out[key] = value
    return out


def _report_key_values(path: Path, keys: list[str]) -> dict[str, str | None]:
    if not path.exists():
        return {key: None for key in keys}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str | None] = {}
    for key in keys:
        match = re.search(rf"\|\s*{re.escape(key)}\s*\|\s*([^|]+?)\s*\|", text)
        out[key] = match.group(1).strip() if match else None
    return out


def _report_assignments(path: Path, keys: list[str]) -> dict[str, str | None]:
    if not path.exists():
        return {key: None for key in keys}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str | None] = {}
    for key in keys:
        match = re.search(rf"\b{re.escape(key)}\s*=\s*([^\s`]+)", text)
        out[key] = match.group(1).strip() if match else None
    return out


def _gate_status(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    ari = _float(metrics.get("ARI"))
    purity = _float(metrics.get("purity"))
    completeness = _float(metrics.get("completeness"))
    unknown = _float(metrics.get("unknown_tube_ratio") or metrics.get("unknown"))
    scene0081 = _float(metrics.get("scene0081_ARI"))
    checks = {
        "ari_pass": ari is not None and ari >= LOCAL_GATE["ARI"],
        "purity_pass": purity is not None and purity >= LOCAL_GATE["purity"],
        "completeness_pass": completeness is not None and completeness >= LOCAL_GATE["completeness"],
        "unknown_pass": unknown is not None and unknown <= LOCAL_GATE["unknown_tube_ratio_max"],
        "scene0081_pass": scene0081 is not None and scene0081 >= LOCAL_GATE["scene0081_ARI"],
    }
    return {**checks, "pass_3D_gate": bool(all(checks.values()))}


def _source_status(stream3d_root: Path, paths: dict[str, str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, rel in paths.items():
        path = stream3d_root / rel
        out[name] = {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            "sha256": _sha256(path),
        }
    return out


def build_lock(stream3d_root: Path, output: Path, package_log: Path) -> dict[str, Any]:
    audit = stream3d_root / "outputs" / "audit"
    repo_root = stream3d_root.parent
    out_dir = output.parent

    source_paths = {
        "v35_final_decision_json": "outputs/audit/v35_final_decision/decision_summary.json",
        "v35_continuation_decision_json": "outputs/audit/v35_final_decision/continuation_decision_summary.json",
        "v35_decision_table_csv": "outputs/audit/v35_final_decision/decision_table.csv",
        "v35_mask_source_audit_json": "outputs/audit/v35_mask_source_audit/mask_source_audit.json",
        "v35_oracle_summary_csv": "outputs/audit/v35_mask_source_audit/proposal_rebuild_conda/v35_mask_source_rebuild_conda_oracle_summary.csv",
        "v35_routeA_summary_csv": "outputs/audit/v35_routeA_region_first/routeA_summary.csv",
        "v35_routeB_feature_metrics_csv": "outputs/audit/v35_routeB_visual_embedding_conda/routeB_feature_metrics.csv",
        "v35_routeB_repair_object_metrics_csv": "outputs/audit/v35_routeB_visual_embedding_repair_high_threshold_conda/routeB_object_metrics.csv",
        "v35_routeD_pair_graph_summary_csv": "outputs/audit/v35_routeD_learned_diagnostic_conda/routeD_pair_graph_summary.csv",
        "v34_report_md": "../docs/stream4d_v34_3d_object_identity_first_report.md",
        "v35_report_md": "../docs/stream4d_v35_break_glass_3d_identity_report.md",
        "v31_report_md": "../docs/stream4d_v31_seed_anchor_lowtail_report.md",
        "v26_report_md": "../docs/stream4d_v26_object_quality_report.md",
        "v23_report_md": "../docs/stream4d_v23_d4rt_reconstruction_quality_report.md",
    }
    status = _source_status(stream3d_root, source_paths)

    v35_final = _read_json(audit / "v35_final_decision" / "decision_summary.json")
    v35_cont = _read_json(audit / "v35_final_decision" / "continuation_decision_summary.json")
    v35_mask_source_audit = _read_json(audit / "v35_mask_source_audit" / "mask_source_audit.json")
    decision_rows = _read_csv(audit / "v35_final_decision" / "decision_table.csv")
    oracle_rows = _read_csv(audit / "v35_mask_source_audit" / "proposal_rebuild_conda" / "v35_mask_source_rebuild_conda_oracle_summary.csv")
    route_a_rows = _read_csv(audit / "v35_routeA_region_first" / "routeA_summary.csv")
    route_b_features = _read_csv(audit / "v35_routeB_visual_embedding_conda" / "routeB_feature_metrics.csv")
    route_b_repair_rows = _read_csv(
        audit / "v35_routeB_visual_embedding_repair_high_threshold_conda" / "routeB_object_metrics.csv"
    )
    route_d_rows = _read_csv(audit / "v35_routeD_learned_diagnostic_conda" / "routeD_pair_graph_summary.csv")

    route_a_a2 = _find_row(route_a_rows, scene="ALL", variant="A2_boundary_hard")
    route_b_b4 = _find_row(route_b_repair_rows, scene="ALL", variant="B4_embedding_d4rt_boundary_unknown")
    route_d_d8 = _find_row(route_d_rows, scene="ALL", variant="D8_pair_graph_rf_no_negative_ablation")
    route_b_feature_all = _find_row(route_b_features, scene="ALL")
    oracle_o5 = _find_row(oracle_rows, scene="ALL", pool="O5_hybrid")
    decision_best = _best_row(decision_rows, "ARI")

    v34_report = repo_root / "docs" / "stream4d_v34_3d_object_identity_first_report.md"
    v35_report = repo_root / "docs" / "stream4d_v35_break_glass_3d_identity_report.md"
    report_values = {
        "v34_report_assignments": _report_assignments(
            v34_report,
            [
                "final_status",
                "method_pass_count",
                "window0_baseline_ari",
            ],
        ),
        "v34_report_locked_facts_table": _report_key_values(
            v34_report,
            [
                "v23 P5 F@10cm",
                "v23 P5 F@20cm",
                "v25 scale_sensitive_metric_reads",
                "v26 interior coverage mean",
                "v30 O5 oracle ARI",
                "v30 best non-GT ARI",
            ],
        ),
        "v35_report_locked_facts_table": _report_key_values(
            v35_report,
            [
                "v34_final_status",
                "v34_method_pass_count",
                "v34_window0_baseline_ari",
                "v34_best_method_route",
                "v34_best_method_ARI",
                "v34_best_method_purity",
                "v34_best_method_completeness",
                "v34_D8_best_AUC",
                "v34_D8_best_ARI",
            ],
        ),
        "v35_report_assignments": _report_assignments(v35_report, ["final_status"]),
    }

    o5_metrics = _metric_row(
        oracle_o5,
        {
            "oracle_ARI": "oracle_ARI",
            "oracle_purity": "oracle_purity",
            "oracle_completeness": "oracle_completeness",
            "scene0081_oracle_ARI": "scene0081_oracle_ARI",
            "GT_with_best_IoU_ge_025": "GT_with_best_IoU_ge_025",
            "GT_with_best_IoU_ge_050": "GT_with_best_IoU_ge_050",
            "oracle_per_GT_best_IoU_mean": "oracle_per_GT_best_IoU_mean",
            "proposal_count": "proposal_count",
        },
    )
    route_a_metrics = _metric_row(
        route_a_a2,
        {
            "ARI": "local_ARI",
            "purity": "local_purity",
            "completeness": "local_completeness",
            "unknown_tube_ratio": "unknown_tube_ratio",
            "scene0081_ARI": "scene0081_local_ARI",
        },
    )
    route_b_metrics = _metric_row(
        route_b_b4,
        {
            "ARI": "local_ARI",
            "purity": "local_purity",
            "completeness": "local_completeness",
            "unknown_tube_ratio": "unknown_tube_ratio",
            "scene0081_ARI": "scene0081_local_ARI",
        },
    )
    route_d_metrics = _metric_row(
        route_d_d8,
        {
            "AUC": "diagnostic_auc",
            "ARI": "local_ARI",
            "purity": "local_purity",
            "completeness": "local_completeness",
            "unknown_tube_ratio": "unknown_tube_ratio",
            "scene0081_ARI": "scene0081_local_ARI",
        },
    )
    route_b_feature_metrics = _metric_row(
        route_b_feature_all,
        {
            "same_GT_pair_AUC": "same_GT_pair_AUC",
            "mixed_region_AUC": "mixed_region_AUC",
            "scene0081_feature_AUC": "scene0081_feature_AUC",
        },
    )

    continuation_best = None
    if isinstance(v35_cont, dict):
        continuation_best = v35_cont.get("best_continuation_method_metrics") or v35_cont.get("best_continuation")

    package_validation = _validation_status(out_dir)
    py_pass = bool(package_validation["py_compile"]["pass"])
    unittest_pass = bool(package_validation["unittest"]["pass"])
    v35_final_status_loaded = isinstance(v35_final, dict) or isinstance(v35_cont, dict)

    lock = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v36_phaseA",
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
        "package_validation": package_validation,
        "phaseA_success_criteria": {
            "py_compile_pass": py_pass,
            "unittest_pass": unittest_pass,
            "v35_final_status_loaded": v35_final_status_loaded,
            "phaseA_pass": bool(py_pass and unittest_pass and v35_final_status_loaded),
        },
        "source_files": status,
        "current_state_lock": {
            "v35_final_status": (
                v35_cont.get("continuation_status")
                if isinstance(v35_cont, dict)
                else (v35_final.get("final_status") if isinstance(v35_final, dict) else None)
            ),
            "v35_allowed_4d": v35_cont.get("allowed_4d") if isinstance(v35_cont, dict) else None,
            "v35_allowed_ap": v35_cont.get("allowed_ap") if isinstance(v35_cont, dict) else None,
            "v35_mask_source_audit": {
                "source_rows": v35_mask_source_audit.get("source_rows") if isinstance(v35_mask_source_audit, dict) else None,
                "external_source_availability": (
                    v35_mask_source_audit.get("external_source_availability")
                    if isinstance(v35_mask_source_audit, dict)
                    else None
                ),
                "success_criteria": v35_mask_source_audit.get("success_criteria") if isinstance(v35_mask_source_audit, dict) else None,
                "proposal_rows_are_current_v35_run": (
                    v35_mask_source_audit.get("proposal_rows_are_current_v35_run")
                    if isinstance(v35_mask_source_audit, dict)
                    else None
                ),
            },
            "v34_and_older_report_values": report_values,
            "v35_decision_table_best_by_ARI": decision_best,
            "v35_O5_hybrid_oracle": o5_metrics,
            "v35_routeA_A2_boundary_hard": {
                "metrics": route_a_metrics,
                "gate": _gate_status(route_a_metrics),
            },
            "v35_routeB_DINOv2_feature_metrics": route_b_feature_metrics,
            "v35_routeB_B4_embedding_d4rt_boundary_unknown": {
                "metrics": route_b_metrics,
                "gate": _gate_status(route_b_metrics),
            },
            "v35_routeD_D8_pair_graph_rf_no_negative_ablation": {
                "metrics": route_d_metrics,
                "gate": _gate_status(route_d_metrics),
                "is_learned_diagnostic": True,
                "uses_gt_for_prediction": True,
            },
            "v35_continuation_best": continuation_best,
            "v36_interpretation_from_plan": {
                "old_proposal_row_reruns_are_invalid_v36": True,
                "must_generate_real_2d_regions_from_rgb_or_masks": True,
                "must_attempt_external_mask_source_if_needed": True,
                "do_not_run_4d_or_ap_unless_3d_gate_passes": True,
            },
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"created_at={lock['created_at']}",
        f"py_compile_pass={str(py_pass).lower()}",
        f"py_compile_exit_code={package_validation['py_compile']['exit_code']}",
        f"unittest_pass={str(unittest_pass).lower()}",
        f"unittest_exit_code={package_validation['unittest']['exit_code']}",
        f"unittest_test_count={package_validation['unittest']['unittest_test_count']}",
        f"v35_final_status_loaded={str(v35_final_status_loaded).lower()}",
        f"phaseA_pass={str(lock['phaseA_success_criteria']['phaseA_pass']).lower()}",
        f"current_state_lock={output}",
    ]
    package_log.parent.mkdir(parents=True, exist_ok=True)
    package_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/v36_phaseA/current_state_lock.json"))
    parser.add_argument("--package-log", type=Path, default=Path("outputs/audit/v36_phaseA/package_validation.log"))
    args = parser.parse_args()

    stream3d_root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else stream3d_root / args.output
    package_log = args.package_log if args.package_log.is_absolute() else stream3d_root / args.package_log
    lock = build_lock(stream3d_root, output, package_log)
    print(json.dumps(lock["phaseA_success_criteria"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
