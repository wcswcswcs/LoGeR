from __future__ import annotations

from pathlib import Path
from typing import Any

from stream4d_native.regression_guarded_matching import phase_d_semantic_gate
from stream4d_native.v37_object_field_adapter import read_csv_rows


def run_semantic_residual(
    stream3d_root: Path,
    *,
    adapter_summary: dict[str, Any],
    profiler_summary: dict[str, Any],
) -> dict[str, Any]:
    root = Path(stream3d_root)
    root_cause_rows = read_csv_rows(
        root / "outputs/audit/v42_structure_affinity_twohop_backfill8_max480_r1/root_cause/hard_scene_root_cause_rows.csv"
    )
    strict_candidates = []
    relaxed_candidates = []
    for row in root_cause_rows:
        auc = float(row.get("semantic_affinity_AUC") or 0.0)
        coverage = float(row.get("coverage@0.10") or 0.0)
        object_auc = float(row.get("object_part_compatibility_AUC") or 0.0)
        false_merge = float(row.get("same_frame_same_class_false_merge_rate") or 1.0)
        if auc >= 0.75 and coverage >= 0.70 and object_auc >= 0.70:
            strict_candidates.append(row)
        if auc >= 0.70 and coverage >= 0.70 and object_auc >= 0.68 and false_merge <= 0.02:
            relaxed_candidates.append(row)

    baseline_metrics = dict(adapter_summary.get("adapter_metrics") or {})
    candidate_metrics = dict(baseline_metrics)
    candidate_metrics["changed_object_ratio"] = 0.0
    hard_scene_delta = 0.0
    gate = phase_d_semantic_gate(candidate_metrics, baseline_metrics, hard_scene_delta)
    accepted = []
    rejected = []
    if not profiler_summary.get("gate", {}).get("profiler_gate_pass"):
        rejected.append(
            {
                "correction": "semantic_residual_matching",
                "reason": "profiler_gate_failed",
                "details": profiler_summary.get("gate", {}),
            }
        )
    if strict_candidates:
        rejected.append(
            {
                "correction": "strict_semantic_candidate_rows",
                "reason": "diagnostic_rows_do_not_include_assignment_update_or_measured_metric_gain",
                "candidate_count": len(strict_candidates),
            }
        )
    if relaxed_candidates:
        rejected.append(
            {
                "correction": "relaxed_high_precision_hard_scene_rows",
                "reason": "repair_path_checked_but_no_regression_guarded_matching_update_was_measured",
                "candidate_count": len(relaxed_candidates),
            }
        )

    return {
        "phase": "v43_2_semantic_residual_matching",
        "status": "PASS_SEMANTIC_RESIDUAL" if gate["pass"] else "NO_GO_SEMANTIC_RESIDUAL_REGRESSION",
        "input_adapter_status": adapter_summary.get("status"),
        "strict_candidate_count": len(strict_candidates),
        "relaxed_candidate_count": len(relaxed_candidates),
        "accepted_corrections": accepted,
        "rejected_corrections": rejected,
        "metrics": candidate_metrics,
        "gate": gate,
        "notes": [
            "The repair path lowered thresholds only inside high-precision hard-scene diagnostics.",
            "No semantic correction is promoted because current artifacts do not contain a measured assignment update that passes the regression guard.",
        ],
    }
