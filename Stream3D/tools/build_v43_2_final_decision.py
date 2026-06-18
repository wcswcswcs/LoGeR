from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v37_object_field_adapter import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fact-lock", default="outputs/audit/v43_2_fact_lock/fact_lock.json")
    parser.add_argument("--adapter-summary", default="outputs/audit/v43_2_v37_parity_adapter/adapter_parity_summary.json")
    parser.add_argument("--profiler-summary", default="outputs/audit/v43_2_matching_error_profiler/matching_error_profiler_summary.json")
    parser.add_argument("--semantic-summary", default="outputs/audit/v43_2_semantic_residual_matching/semantic_residual_summary.json")
    parser.add_argument("--material-summary", default="outputs/audit/v43_2_material_residual_matching/material_residual_summary.json")
    parser.add_argument("--stage1-summary", default="outputs/audit/v43_2_full_matching_significance/full_matching_significance_summary.json")
    parser.add_argument("--geometry-summary", default="outputs/audit/v43_2_geometry_optimization_diagnostic/geometry_optimization_diagnostic_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v43_2_final_decision")
    args = parser.parse_args()
    fact = read_json(ROOT / args.fact_lock) or {}
    adapter = read_json(ROOT / args.adapter_summary) or {}
    profiler = read_json(ROOT / args.profiler_summary) or {}
    semantic = read_json(ROOT / args.semantic_summary) or {}
    material = read_json(ROOT / args.material_summary) or {}
    stage1 = read_json(ROOT / args.stage1_summary) or {}
    geometry = read_json(ROOT / args.geometry_summary) or {}

    if not adapter.get("gate", {}).get("metric_parity_pass"):
        final_status = "NO_GO_ADAPTER_REGRESSION"
        failure_axis = "adapter_metric_parity"
    elif not profiler.get("gate", {}).get("profiler_gate_pass"):
        final_status = "NO_GO_NO_REPAIRABLE_ERRORS"
        failure_axis = "profiler"
    elif str(semantic.get("status")) == "NO_GO_SEMANTIC_RESIDUAL_REGRESSION":
        final_status = "NO_GO_SEMANTIC_RESIDUAL_REGRESSION"
        failure_axis = "semantic_residual"
    elif str(material.get("status")) == "NO_GO_MATERIAL_NOT_DISCRIMINATIVE":
        final_status = "NO_GO_MATERIAL_NOT_DISCRIMINATIVE"
        failure_axis = "material_residual"
    else:
        final_status = str(stage1.get("status") or "NO_GO_MATCHING_NOT_SIGNIFICANT")
        failure_axis = "matching_significance"

    payload = {
        "phase": "v43_2_final_decision",
        "final_status": final_status,
        "failure_axis": failure_axis,
        "stage2_allowed": bool(stage1.get("stage2_allowed")),
        "ap_bridge_status": "AP_BRIDGE_BLOCKED_OR_NOT_METHOD_SUCCESS",
        "first_page_answers": {
            "v37_to_v43_adapter_preserves_v37": adapter.get("gate", {}).get("metric_parity_pass"),
            "stage1_matching_significantly_exceeds_v37_v41_1": stage1.get("status", "").startswith("GO_STAGE1"),
            "minimum_significant_gate": stage1.get("minimum_significant_gate", {}).get("pass"),
            "strong_conference_gate": stage1.get("strong_conference_gate", {}).get("pass"),
            "bootstrap_lower_bound_supports_significance": stage1.get("significance", {}).get("checks", {}).get("pass"),
            "residual_corrections_accepted": {
                "semantic": semantic.get("accepted_corrections", []),
                "material": material.get("accepted_corrections", []),
            },
            "d4rt_material_beats_controls": material.get("gate", {}).get("pass"),
            "d4rt_tubes_birth_object": False,
            "stage2_allowed": bool(stage1.get("stage2_allowed")),
            "ap_bridge_blocker": True,
            "failure_axis": failure_axis,
        },
        "source_summaries": {
            "fact_lock": args.fact_lock,
            "adapter": args.adapter_summary,
            "profiler": args.profiler_summary,
            "semantic": args.semantic_summary,
            "material": args.material_summary,
            "stage1": args.stage1_summary,
            "geometry": args.geometry_summary,
        },
        "key_metrics": {
            "v37": fact.get("v37_4d_best_metrics"),
            "adapter": adapter.get("adapter_metrics"),
            "f6": stage1.get("f6_metrics"),
            "minimum_gate": stage1.get("minimum_significant_gate"),
            "compactness_gate": stage1.get("compactness_gate"),
            "control_gate": stage1.get("control_gate"),
            "significance": stage1.get("significance", {}).get("checks"),
        },
        "geometry": geometry,
    }
    out = ROOT / args.output_root
    write_json(out / "final_decision.json", payload)
    print(json.dumps({"final_decision": str(out / "final_decision.json"), "final_status": final_status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
