from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stream4d_native.matching_significance import V37_BASELINE, V43_MINIMUM_GATE, V43_STRONG_GATE
from stream4d_native.v37_object_field_adapter import read_json, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def _row(fact: str, value: Any, source: Path, status: str) -> dict[str, Any]:
    return {"fact": fact, "value": value, "source": str(source), "status": status}


def build_fact_lock(output_root: Path) -> dict[str, Any]:
    v37_4d_path = ROOT / "outputs/audit/v37_4d_if_allowed_i4_sparse/4d_memory_decision.json"
    v37_3d_path = ROOT / "outputs/audit/v37_adaptive_density_final_probe5/v37_final_decision/decision_summary.json"
    v37_scene_rows = ROOT / "outputs/audit/v37_4d_if_allowed_i4_sparse/4d_memory_scene_rows.csv"
    v41_native_path = (
        ROOT
        / "outputs/audit/v41_1_native_support_metrics_probe5_sweep/offsetfix2_closure_rgb090_t035_m010_birthgate/native_support_metrics_summary.json"
    )
    v42_fact_path = ROOT / "outputs/audit/v42_fact_lock/fact_lock.json"
    v42_radio_path = ROOT / "outputs/audit/v42_source_audit/radio_vipe_availability.json"
    v42_source_path = ROOT / "outputs/audit/v42_source_audit/source_availability.json"
    v42_report = ROOT.parent / "docs/stream4d_v42_semantic_part_material_factor_graph_report.md"

    v37_4d = read_json(v37_4d_path) or {}
    v37_3d = read_json(v37_3d_path) or {}
    v41_native = read_json(v41_native_path) or {}
    v42_fact = read_json(v42_fact_path) or {}
    v42_radio = read_json(v42_radio_path) or {}
    v42_source = read_json(v42_source_path) or {}
    v37_best = dict(v37_4d.get("best_metrics") or {})
    v37_3d_best = dict(v37_3d.get("best_metrics") or {})
    v37_controls = dict(v37_3d.get("controls") or {})
    v41_metrics = dict(v41_native.get("aggregate_metrics") or {})

    rows = [
        _row("v37_4D_ARI", v37_best.get("4D_ARI"), v37_4d_path, "imported_prior" if v37_best else "missing"),
        _row("v37_4D_purity", v37_best.get("4D_purity"), v37_4d_path, "imported_prior" if v37_best else "missing"),
        _row(
            "v37_4D_completeness",
            v37_best.get("4D_completeness"),
            v37_4d_path,
            "imported_prior" if v37_best else "missing",
        ),
        _row(
            "v37_temporal_span_mean",
            v37_best.get("temporal_span_mean"),
            v37_4d_path,
            "imported_prior" if v37_best else "missing",
        ),
        _row("v37_scene0081_ARI", v37_best.get("scene0081_ARI"), v37_4d_path, "imported_prior" if v37_best else "missing"),
        _row("v37_3D_ARI", v37_3d_best.get("ARI"), v37_3d_path, "imported_prior" if v37_3d_best else "missing"),
        _row("v37_3D_purity", v37_3d_best.get("purity"), v37_3d_path, "imported_prior" if v37_3d_best else "missing"),
        _row(
            "v37_3D_completeness",
            v37_3d_best.get("completeness"),
            v37_3d_path,
            "imported_prior" if v37_3d_best else "missing",
        ),
        _row("v37_real_minus_shuffled", v37_controls.get("real_minus_shuffled"), v37_3d_path, "imported_prior"),
        _row("v37_real_minus_no_temporal", v37_controls.get("real_minus_no_temporal"), v37_3d_path, "imported_prior"),
        _row("v37_real_minus_mask_only", v37_controls.get("real_minus_mask_only"), v37_3d_path, "imported_prior"),
        _row("v41_1_native_support_4D_ARI", v41_metrics.get("4D_ARI"), v41_native_path, "imported_prior" if v41_metrics else "missing"),
        _row(
            "v41_1_native_support_purity",
            v41_metrics.get("4D_purity"),
            v41_native_path,
            "imported_prior" if v41_metrics else "missing",
        ),
        _row(
            "v41_1_native_support_completeness",
            v41_metrics.get("4D_completeness"),
            v41_native_path,
            "imported_prior" if v41_metrics else "missing",
        ),
        _row(
            "v41_1_birth_from_d4rt_tube_count_sum",
            v41_metrics.get("birth_from_d4rt_tube_count_sum"),
            v41_native_path,
            "imported_prior" if v41_metrics else "missing",
        ),
        _row("v42_radio_available", v42_radio.get("radio_available"), v42_radio_path, "imported_prior" if v42_radio else "missing"),
        _row("v42_source_availability_loaded", bool(v42_source.get("sources")), v42_source_path, "imported_prior" if v42_source else "missing"),
        _row("v42_proxy_status", "partial_not_matching_success", v42_report, "imported_prior"),
        _row("v42_fact_lock_loaded", bool(v42_fact), v42_fact_path, "imported_prior" if v42_fact else "missing"),
    ]
    payload = {
        "phase": "v43_2_fact_lock",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "training_free": True,
            "prior_evidence_is_marked_imported": True,
            "v42_proxy_not_stage1_success": True,
            "ap_bridge_not_stage1_success": True,
        },
        "baseline": {
            "v37_plan_baseline": V37_BASELINE,
            "minimum_significant_gate": V43_MINIMUM_GATE,
            "strong_conference_gate": V43_STRONG_GATE,
        },
        "artifacts": {
            "v37_4d_decision": str(v37_4d_path),
            "v37_4d_scene_rows": str(v37_scene_rows),
            "v37_3d_decision": str(v37_3d_path),
            "v41_1_native_support": str(v41_native_path),
            "v42_fact_lock": str(v42_fact_path),
            "v42_radio_availability": str(v42_radio_path),
            "v42_source_availability": str(v42_source_path),
        },
        "v37_4d_best_metrics": v37_best,
        "v37_3d_best_metrics": v37_3d_best,
        "v37_3d_controls": v37_controls,
        "v41_1_native_support_metrics": v41_metrics,
        "v42_radio_availability": v42_radio,
        "rows": rows,
        "gate": {
            "v37_loaded_as_primary_matching_baseline": bool(v37_best),
            "v41_1_loaded_as_structural_invariant_baseline": bool(v41_metrics),
            "v42_marked_proxy_partial_not_matching_success": True,
            "phaseA_pass": bool(v37_best and v41_metrics),
        },
    }
    write_json(output_root / "fact_lock.json", payload)
    write_csv(output_root / "fact_lock_rows.csv", rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/audit/v43_2_fact_lock")
    args = parser.parse_args()
    payload = build_fact_lock(ROOT / args.output_root)
    print(json.dumps({"fact_lock": str(ROOT / args.output_root / "fact_lock.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
