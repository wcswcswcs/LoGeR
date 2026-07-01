from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


DEFAULT_V59_ZIP = "code_audit_pack/stream4d_v59_soma_manifold_audit_20260621_182000.zip"
DEFAULT_V59_PHASE0 = "outputs/audit/v59_phase0_fact_lock/fact_lock.json"
DEFAULT_V59_PHASE1 = "outputs/audit/v59_phase1_graph/graph_summary.json"
DEFAULT_V59_PHASE2 = "outputs/audit/v59_phase2_paths_repair_margin070_noexcl_semcat/path_summary.json"
DEFAULT_V59_FINAL = "outputs/audit/v59_final_decision/final_decision.json"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    stream3d_path = ROOT / path_obj
    if stream3d_path.exists() or str(path_obj).startswith("outputs/"):
        return stream3d_path
    return ROOT.parent / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def wilson_upper_95(false_count: int, pair_count: int) -> float | None:
    if pair_count <= 0:
        return None
    z = 1.959963984540054
    phat = float(false_count) / float(pair_count)
    denom = 1.0 + z * z / pair_count
    center = phat + z * z / (2.0 * pair_count)
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * pair_count)) / pair_count)
    return float((center + radius) / denom)


def calibrated_same_category_gate(
    *,
    method_false_count: int,
    method_pair_count: int,
    baseline_false_rate: float | None,
    baseline_low_threshold: float = 0.05,
) -> dict[str, Any]:
    method_rate = None if method_pair_count <= 0 else float(method_false_count) / float(method_pair_count)
    upper = wilson_upper_95(method_false_count, method_pair_count)
    if baseline_false_rate is None:
        pass_gate = False
        mode = "missing_baseline"
        required = None
    elif baseline_false_rate < baseline_low_threshold:
        zero_false_pass = method_false_count == 0 and method_pair_count >= 50
        wilson_pass = upper is not None and upper <= 0.05 and method_pair_count >= 50
        pass_gate = bool(zero_false_pass or wilson_pass)
        mode = "low_baseline_exact_or_wilson"
        required = 0.05
    else:
        margin = min(0.05, 0.5 * float(baseline_false_rate))
        required = float(baseline_false_rate) - margin
        pass_gate = method_rate is not None and method_rate <= required
        mode = "baseline_relative_improvement"
    return {
        "mode": mode,
        "method_false_count": int(method_false_count),
        "method_pair_count": int(method_pair_count),
        "method_false_rate": method_rate,
        "method_wilson_upper95": upper,
        "baseline_false_rate": baseline_false_rate,
        "required_max_rate": required,
        "pass": pass_gate,
    }


def build_v60_fact_lock(
    *,
    v59_zip_path: str | Path = DEFAULT_V59_ZIP,
    v59_phase0_path: str | Path = DEFAULT_V59_PHASE0,
    v59_phase1_path: str | Path = DEFAULT_V59_PHASE1,
    v59_phase2_path: str | Path = DEFAULT_V59_PHASE2,
    v59_final_path: str | Path = DEFAULT_V59_FINAL,
) -> dict[str, Any]:
    zip_info = _inspect_v59_zip(v59_zip_path)
    local_info = _inspect_local_v59_artifacts(v59_phase0_path, v59_phase1_path, v59_phase2_path, v59_final_path)
    phase0 = read_json(_project(v59_phase0_path))
    phase1 = read_json(_project(v59_phase1_path))
    phase2 = read_json(_project(v59_phase2_path))
    final = read_json(_project(v59_final_path))

    method_false_count = int(phase2.get("same_category_method_false_count") or 0)
    method_pair_count = int(phase2.get("same_category_method_pair_count") or 0)
    baseline_rate = phase2.get("same_category_baseline_false_path_rate")
    baseline_rate = None if baseline_rate is None else float(baseline_rate)
    calibrated = calibrated_same_category_gate(
        method_false_count=method_false_count,
        method_pair_count=method_pair_count,
        baseline_false_rate=baseline_rate,
    )
    gate = {
        "zip_or_local_v59_phase0_outputs_available": bool(
            zip_info["zip_contains_v59_phase0_outputs"] or local_info["local_v59_phase0_outputs_present"]
        ),
        "zip_or_local_v59_phase1_outputs_available": bool(
            zip_info["zip_contains_v59_phase1_outputs"] or local_info["local_v59_phase1_outputs_present"]
        ),
        "zip_or_local_v59_phase2_outputs_available": bool(
            zip_info["zip_contains_v59_phase2_outputs"] or local_info["local_v59_phase2_outputs_present"]
        ),
        "zip_or_local_final_decision_available": bool(
            zip_info["zip_contains_final_decision_json"] or local_info["local_v59_final_decision_present"]
        ),
        "source_code_present": bool(zip_info["source_code_present"] or local_info["local_v59_source_code_present"]),
        "logs_present": bool(zip_info["logs_present"] or local_info["local_v59_logs_present"]),
        "recap_present": bool(zip_info["recap_present"] or local_info["local_v59_recap_present"]),
        "same_category_calibrated_gate_pass": bool(calibrated["pass"]),
    }
    gate["pass"] = bool(all(gate.values()))
    fact_lock = {
        "phase": "v60_phase0_fact_lock",
        "created_at": utc_now(),
        "method_note": (
            "Phase0 recalibrates the v59 same-category gate and checks artifact integrity. "
            "It does not change v59 results and does not create a v60 prediction."
        ),
        "v59_zip_path": _rel(v59_zip_path),
        "v59_zip_exists": _project(v59_zip_path).exists(),
        **zip_info,
        **local_info,
        "artifact_source_for_v60": (
            "zip_and_local_outputs"
            if _project(v59_zip_path).exists()
            else "local_v59_outputs_zip_missing"
        ),
        "v59_final_label": final.get("final_label"),
        "v59_partial_label": final.get("partial_label"),
        "v59_goal_achieved": final.get("goal_achieved"),
        "phase0_fact_lock_pass": bool((phase0.get("gate") or {}).get("pass")),
        "phase1_graph_gate_pass": bool((phase1.get("gate") or {}).get("pass")),
        "phase2_original_gate_pass": bool((phase2.get("gate") or {}).get("pass")),
        "phase2_path_precision": phase2.get("path_precision_diagnostic"),
        "phase2_part_to_core_precision": phase2.get("part_to_core_path_precision"),
        "phase2_shortcut_quarantine_precision": phase2.get("shortcut_quarantine_precision"),
        "phase2_same_category_method_pair_count": method_pair_count,
        "phase2_same_category_method_false_count": method_false_count,
        "phase2_same_category_method_false_rate": phase2.get("same_category_false_path_rate"),
        "phase2_same_category_baseline_false_rate": baseline_rate,
        "phase2_same_category_required_max_rate_old": phase2.get("same_category_required_max_rate"),
        "phase2_same_category_calibrated": calibrated,
        "phase2_same_category_gate_calibrated_pass": bool(calibrated["pass"]),
        "same_category_history_label_coverage": phase2.get("same_category_history_label_coverage"),
        "same_category_history_label_ambiguity": phase2.get("same_category_history_label_ambiguity"),
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "diagnostic_label_sources": [
            "v59 Phase2 same-category diagnostic metrics",
            "v59 artifact zip file listing for integrity only",
        ],
        "input_paths": {
            "v59_phase0": _rel(v59_phase0_path),
            "v59_phase1": _rel(v59_phase1_path),
            "v59_phase2": _rel(v59_phase2_path),
            "v59_final": _rel(v59_final_path),
        },
    }
    gate_rows = [
        {
            "row": "v59_old_gate",
            "method_false_rate": phase2.get("same_category_false_path_rate"),
            "baseline_false_rate": baseline_rate,
            "required_max_rate": phase2.get("same_category_required_max_rate"),
            "pass": bool((phase2.get("gate") or {}).get("same_category_false_path_rate_le_semantic_pairwise_baseline_minus_0_05")),
            "note": "v59 negative-threshold hard gate",
        },
        {
            "row": "v60_calibrated_gate",
            "method_false_rate": calibrated["method_false_rate"],
            "method_false_count": calibrated["method_false_count"],
            "method_pair_count": calibrated["method_pair_count"],
            "method_wilson_upper95": calibrated["method_wilson_upper95"],
            "baseline_false_rate": calibrated["baseline_false_rate"],
            "required_max_rate": calibrated["required_max_rate"],
            "pass": calibrated["pass"],
            "note": calibrated["mode"],
        },
    ]
    return {"fact_lock": fact_lock, "same_category_gate_rows": gate_rows, "artifact_tree_rows": zip_info["artifact_tree_rows"]}


def write_v60_fact_lock(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "fact_lock": root / "fact_lock.json",
        "same_category_gate_rows": root / "same_category_gate_rows.csv",
        "artifact_tree_rows": root / "artifact_tree_rows.csv",
    }
    write_json(paths["fact_lock"], result["fact_lock"])
    write_csv(paths["same_category_gate_rows"], result["same_category_gate_rows"])
    write_csv(paths["artifact_tree_rows"], result["artifact_tree_rows"])
    tree_path = root / "artifact_integrity_tree.txt"
    tree_lines = [row["path"] for row in result["artifact_tree_rows"]]
    tree_path.write_text("\n".join(tree_lines) + "\n", encoding="utf-8")
    paths["artifact_integrity_tree"] = tree_path
    return {name: _rel(path) for name, path in paths.items()}


def write_v60_phase0_visualizations(result: dict[str, Any], visualization_root: str | Path) -> dict[str, str]:
    root = _project(visualization_root)
    root.mkdir(parents=True, exist_ok=True)
    fact = result["fact_lock"]
    calibrated = fact["phase2_same_category_calibrated"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        status_path = root / "v60_phase0_v59_status_dashboard.png"
        labels = ["phase0", "phase1", "v59 phase2", "v60 calibrated"]
        values = [
            1.0 if fact["phase0_fact_lock_pass"] else 0.0,
            1.0 if fact["phase1_graph_gate_pass"] else 0.0,
            1.0 if fact["phase2_original_gate_pass"] else 0.0,
            1.0 if fact["phase2_same_category_gate_calibrated_pass"] else 0.0,
        ]
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.bar(labels, values, color=["#52796F", "#52796F", "#B56576", "#2A9D8F"])
        ax.set_ylim(0.0, 1.05)
        ax.set_title("v60 Phase0 v59 status")
        fig.tight_layout()
        fig.savefig(status_path, dpi=160)
        plt.close(fig)

        calib_path = root / "same_category_gate_calibration_plot.png"
        fig, ax = plt.subplots(figsize=(7.4, 4.0))
        old_required = fact["phase2_same_category_required_max_rate_old"]
        values = [
            fact["phase2_same_category_method_false_rate"],
            fact["phase2_same_category_baseline_false_rate"],
            old_required,
            calibrated["required_max_rate"],
            calibrated["method_wilson_upper95"],
        ]
        labels = ["method", "baseline", "old required", "v60 required", "wilson95"]
        ax.bar(labels, [0.0 if value is None else float(value) for value in values], color="#457B9D")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("same-category gate calibration")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(calib_path, dpi=160)
        plt.close(fig)
        return {
            "status_dashboard": _rel(status_path),
            "same_category_calibration_plot": _rel(calib_path),
            "visualization_status": "created",
        }
    except Exception as exc:  # pragma: no cover
        error_path = root / "v60_phase0_visualization_error.txt"
        error_path.write_text(f"visualization unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"visualization_error": _rel(error_path), "visualization_status": "unavailable"}


def _inspect_v59_zip(v59_zip_path: str | Path) -> dict[str, Any]:
    path = _project(v59_zip_path)
    rows: list[dict[str, Any]] = []
    names: list[str] = []
    if path.exists():
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    for name in names:
        if name.endswith("/"):
            continue
        rows.append({"path": name})

    def has(fragment: str) -> bool:
        return any(fragment in name for name in names)

    return {
        "zip_file_count": len(rows),
        "zip_contains_v59_phase0_outputs": has("outputs/audit/v59_phase0_fact_lock/fact_lock.json"),
        "zip_contains_v59_phase1_outputs": has("outputs/audit/v59_phase1_graph/graph_summary.json"),
        "zip_contains_v59_phase2_outputs": has("outputs/audit/v59_phase2_paths_repair_margin070_noexcl_semcat/path_summary.json"),
        "zip_contains_final_decision_json": has("outputs/audit/v59_final_decision/final_decision.json"),
        "source_code_present": has("Stream3D/stream4d_native/v59_") and has("Stream3D/tools/run_v59_"),
        "logs_present": has("docs/stream4d_v59_执行日志.md"),
        "recap_present": has("docs/stream4d_v59_实验结果复盘.md"),
        "artifact_tree_rows": rows,
    }


def _inspect_local_v59_artifacts(
    v59_phase0_path: str | Path,
    v59_phase1_path: str | Path,
    v59_phase2_path: str | Path,
    v59_final_path: str | Path,
) -> dict[str, Any]:
    return {
        "local_v59_phase0_outputs_present": _project(v59_phase0_path).exists(),
        "local_v59_phase1_outputs_present": _project(v59_phase1_path).exists(),
        "local_v59_phase2_outputs_present": _project(v59_phase2_path).exists(),
        "local_v59_final_decision_present": _project(v59_final_path).exists(),
        "local_v59_source_code_present": all(
            (_project(path).exists())
            for path in [
                "stream4d_native/v59_fact_lock.py",
                "stream4d_native/v59_graph_builder.py",
                "stream4d_native/v59_manifold_paths.py",
                "stream4d_native/v59_final_decision.py",
            ]
        ),
        "local_v59_logs_present": _project("../docs/stream4d_v59_执行日志.md").exists()
        or (ROOT.parent / "docs/stream4d_v59_执行日志.md").exists(),
        "local_v59_recap_present": _project("../docs/stream4d_v59_实验结果复盘.md").exists()
        or (ROOT.parent / "docs/stream4d_v59_实验结果复盘.md").exists(),
    }
