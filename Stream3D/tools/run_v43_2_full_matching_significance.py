from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stream4d_native.matching_significance import (
    compactness_gate,
    control_gate,
    minimum_significant_gate,
    significance_summary,
    stage1_decision,
    strong_conference_gate,
)
from stream4d_native.v37_object_field_adapter import read_json, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def _variant(label: str, status: str, metrics: dict[str, Any], source: str) -> dict[str, Any]:
    row = dict(metrics)
    row.update({"variant": label, "status": status, "source": source})
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fact-lock", default="outputs/audit/v43_2_fact_lock/fact_lock.json")
    parser.add_argument("--adapter-summary", default="outputs/audit/v43_2_v37_parity_adapter/adapter_parity_summary.json")
    parser.add_argument("--semantic-summary", default="outputs/audit/v43_2_semantic_residual_matching/semantic_residual_summary.json")
    parser.add_argument("--material-summary", default="outputs/audit/v43_2_material_residual_matching/material_residual_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v43_2_full_matching_significance")
    args = parser.parse_args()
    fact = read_json(ROOT / args.fact_lock) or {}
    adapter = read_json(ROOT / args.adapter_summary) or {}
    semantic = read_json(ROOT / args.semantic_summary) or {}
    material = read_json(ROOT / args.material_summary) or {}
    v37 = dict(fact.get("v37_4d_best_metrics") or adapter.get("v37_best_metrics") or {})
    v41 = dict(fact.get("v41_1_native_support_metrics") or {})
    adapter_metrics = dict(adapter.get("adapter_metrics") or {})
    semantic_metrics = dict(semantic.get("metrics") or adapter_metrics)
    material_metrics = dict(material.get("metrics") or semantic_metrics)
    f6_metrics = dict(material_metrics)
    scene_rows = adapter.get("scene_rows", [])
    sig = significance_summary(scene_rows, scene_rows)
    variants = [
        _variant("F0_v37_original", "imported_prior", v37, "v37_4d_memory_decision"),
        _variant("F1_v37_to_v43_adapter", adapter.get("status", ""), adapter_metrics, args.adapter_summary),
        _variant("F2_v41_1_corrected_native_support", "imported_prior", v41, args.fact_lock),
        _variant("F3_v42_best_available_proxy", "proxy_partial_not_matching_success", {}, args.fact_lock),
        _variant("F4_semantic_residual_only", semantic.get("status", ""), semantic_metrics, args.semantic_summary),
        _variant("F5_semantic_plus_material_residual", material.get("status", ""), material_metrics, args.material_summary),
        _variant("F6_full_v43_2_stage1_matching", "no_accepted_residual_corrections", f6_metrics, args.material_summary),
        _variant("F7_tube_birth_negative_control", "not_run_for_method_row", {}, "negative_control_declared_only"),
        _variant("F8_global_all_pair_semantic_graph_negative_control", "forbidden_by_plan_not_run", {}, "negative_control_declared_only"),
        _variant("F9_material_only_matching_negative_control", "forbidden_as_method_not_run", {}, "negative_control_declared_only"),
    ]
    min_gate = minimum_significant_gate(f6_metrics)
    strong_gate = strong_conference_gate(f6_metrics)
    compact = compactness_gate(f6_metrics)
    controls = control_gate(f6_metrics)
    decision = stage1_decision(f6_metrics, sig)
    payload = {
        "phase": "v43_2_full_matching_significance",
        "status": decision.label,
        "final_label": decision.label,
        "reason": decision.reason,
        "f6_metrics": f6_metrics,
        "minimum_significant_gate": min_gate,
        "strong_conference_gate": strong_gate,
        "compactness_gate": compact,
        "control_gate": controls,
        "significance": sig,
        "variants": variants,
        "stage2_allowed": decision.label.startswith("GO_STAGE1"),
    }
    out = ROOT / args.output_root
    write_json(out / "full_matching_significance_summary.json", payload)
    write_csv(out / "stage1_variant_rows.csv", variants)
    write_csv(out / "ari_scene_delta_rows.csv", sig.get("ari_scene_delta_rows", []))
    print(json.dumps({"summary": str(out / "full_matching_significance_summary.json"), "status": payload["status"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
