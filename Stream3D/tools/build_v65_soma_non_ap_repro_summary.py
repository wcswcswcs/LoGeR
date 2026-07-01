#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_common import load_dict, load_rows, sha256_file, write_standard_outputs
from stream4d_native.v47_common import utc_now


DEFAULT_OUTPUT_ROOT = "outputs/audit/v65_soma_non_ap_repro"

INPUT_PATHS = {
    "solver_summary": "outputs/audit/v62_solver_v2/solver_summary.json",
    "solver_energy_rows": "outputs/audit/v62_solver_v2/energy_rows.csv",
    "native_summary": "outputs/audit/v62_native_field/native_field_summary.json",
    "native_component_rows": "outputs/audit/v62_native_field/native_component_state_rows.csv",
    "v62_final": "outputs/audit/v62_final/final_decision.json",
    "v62_final_metric_rows": "outputs/audit/v62_final/final_metric_rows.csv",
    "main_fact_lock": "outputs/audit/v64r2_phaseA0_main_fact_lock/main_fact_lock_summary.json",
    "main_metric_rows": "outputs/audit/v64r2_phaseA0_main_fact_lock/main_metric_rows.csv",
    "native_contract": "outputs/audit/v64r2_native_contract/native_contract_summary.json",
    "object_field_rows": "outputs/audit/v64r2_native_contract/object_field_rows.csv",
    "material_state_rows": "outputs/audit/v64r2_native_contract/material_state_rows.csv",
    "v64r2_final": "outputs/audit/v64r2_final/final_decision.json",
    "v64r2_final_metric_rows": "outputs/audit/v64r2_final/final_metric_rows.csv",
    "soma_object_bank": "outputs/audit/v65_soma_object_bank/soma_object_bank_summary.json",
    "soma_object_bank_rows": "outputs/audit/v65_soma_object_bank/soma_object_bank_rows.csv",
    "soma_object_material_rows": "outputs/audit/v65_soma_object_bank/soma_object_material_rows.csv",
    "soma_object_support_rows": "outputs/audit/v65_soma_object_bank/soma_object_support_rows.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize reproduced SOMA non-AP metrics.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    payload = build_summary()
    write_standard_outputs(
        args.output_root,
        {
            "soma_non_ap_repro_summary.json": payload["summary"],
            "soma_non_ap_metric_rows.csv": payload["metric_rows"],
            "soma_non_ap_artifact_hashes.csv": payload["hash_rows"],
        },
    )
    print(
        {
            "summary": f"{args.output_root}/soma_non_ap_repro_summary.json",
            "metric_rows": f"{args.output_root}/soma_non_ap_metric_rows.csv",
            "hash_rows": f"{args.output_root}/soma_non_ap_artifact_hashes.csv",
            "main_ownership_status": payload["summary"]["main_ownership_status"],
            "non_ap_repro_gate": payload["summary"]["gate"],
        }
    )


def build_summary() -> dict[str, Any]:
    solver = load_dict(INPUT_PATHS["solver_summary"])
    native = load_dict(INPUT_PATHS["native_summary"])
    final62 = load_dict(INPUT_PATHS["v62_final"])
    main = load_dict(INPUT_PATHS["main_fact_lock"])
    contract = load_dict(INPUT_PATHS["native_contract"])
    final64 = load_dict(INPUT_PATHS["v64r2_final"])
    object_bank = load_dict(INPUT_PATHS["soma_object_bank"])
    native_component_rows = load_rows(INPUT_PATHS["native_component_rows"])
    state_counts = _state_counts(native_component_rows)
    main_summary = main.get("summary") if isinstance(main.get("summary"), dict) else {}
    main_gate = main.get("gate") if isinstance(main.get("gate"), dict) else {}
    solver_gate = solver.get("gate") if isinstance(solver.get("gate"), dict) else {}
    native_gate = native.get("gate") if isinstance(native.get("gate"), dict) else {}
    contract_gate = contract.get("gate") if isinstance(contract.get("gate"), dict) else {}
    object_bank_gate = object_bank.get("gate") if isinstance(object_bank.get("gate"), dict) else {}
    no_gt_inference = not any(
        bool(payload.get("uses_gt_for_prediction"))
        for payload in [solver, native, contract, object_bank]
        if isinstance(payload, dict)
    )
    no_rgbd_pose_mesh_export = not any(
        bool(payload.get("uses_rgbd_pose_mesh_for_export"))
        for payload in [native, contract, object_bank]
        if isinstance(payload, dict)
    )
    gate = {
        "solver_gate_pass": bool(solver_gate.get("pass")),
        "native_field_gate_pass": bool(native_gate.get("pass")),
        "v62_decision_go": final62.get("decision_label") == "GO_V62_VERIFIED_OWNERSHIP_FIELD",
        "main_ownership_go": main_summary.get("main_ownership_status") == "GO_MAIN_OWNERSHIP_FIELD",
        "native_contract_gate_pass": bool(contract_gate.get("pass")),
        "object_bank_available": bool(object_bank.get("object_bank_available")),
        "no_gt_inference": no_gt_inference,
        "no_rgbd_pose_mesh_export": no_rgbd_pose_mesh_export,
        "ap_excluded_from_this_repro": True,
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v65_soma_non_ap_repro",
        "created_at": utc_now(),
        "scope": "SOMA non-AP ownership/native/object-bank metric reproduction only",
        "ap_excluded": True,
        "main_ownership_status": main_summary.get("main_ownership_status"),
        "v62_decision_label": final62.get("decision_label"),
        "v64r2_decision_label": final64.get("decision_label"),
        "core_purity": main_summary.get("core_purity"),
        "core_completeness": main_summary.get("core_completeness"),
        "state_coverage_rate": main_summary.get("state_coverage_rate"),
        "real_minus_shuffled_ARI": main_summary.get("real_minus_shuffled_ARI"),
        "real_minus_no_temporal_ARI": main_summary.get("real_minus_no_temporal_ARI"),
        "component_state_counts": state_counts,
        "native_observation_row_count": native.get("native_observation_row_count"),
        "component_level_available": native.get("component_level_field_available"),
        "carrier_level_available": native.get("carrier_level_field_available"),
        "component_to_carrier_mapping_available": native.get("component_to_carrier_mapping_available"),
        "object_count": contract.get("object_count"),
        "material_count": contract.get("material_count"),
        "confirmed_material_count": contract.get("confirmed_material_count"),
        "support_mask_available": contract.get("support_mask_available"),
        "soma_object_bank": {
            "object_count": object_bank.get("object_count"),
            "material_assignment_count": object_bank.get("material_assignment_count"),
            "object_support_row_count": object_bank.get("object_support_row_count"),
            "objects_with_any_support_count": object_bank.get("objects_with_any_support_count"),
            "objects_with_view_mask_support_count": object_bank.get("objects_with_view_mask_support_count"),
            "objects_with_native_point_or_carrier_support_count": object_bank.get(
                "objects_with_native_point_or_carrier_support_count"
            ),
            "native_carrier_support_row_count": object_bank.get("native_carrier_support_row_count"),
            "native_support_join_available": object_bank.get("native_support_join_available"),
            "object_support_coverage_ratio": object_bank.get("object_support_coverage_ratio"),
            "object_native_support_coverage_ratio": object_bank.get("object_native_support_coverage_ratio"),
            "support_kind_counts": object_bank.get("support_kind_counts"),
            "support_scene_counts": object_bank.get("support_scene_counts"),
            "gate": object_bank_gate,
            "blockers": object_bank.get("blockers"),
        },
        "ground_truth_usage": {
            "uses_gt_for_prediction": not no_gt_inference,
            "uses_gt_for_diagnostic_labels": bool(solver.get("uses_gt_for_diagnostic_labels")),
            "uses_rgbd_pose_mesh_for_export": not no_rgbd_pose_mesh_export,
        },
        "known_limitations": [
            "true_carrier_level_native_field_claim_blocked",
            "native_AP_or_mesh_materialization_claim_blocked",
            "verified_object_to_native_carrier_mapping_missing",
        ],
        "input_paths": INPUT_PATHS,
        "gate": gate,
    }
    return {
        "summary": summary,
        "metric_rows": _metric_rows(summary, solver, native, final62, main, contract, final64, object_bank, state_counts),
        "hash_rows": _hash_rows(),
    }


def _state_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {"confirmed": 0, "tentative": 0, "shared": 0, "quarantine": 0, "unknown": 0}
    for row in rows:
        state = str(row.get("state") or "unknown")
        if state not in counts:
            state = "unknown"
        counts[state] += 1
    return counts


def _metric_rows(
    summary: dict[str, Any],
    solver: dict[str, Any],
    native: dict[str, Any],
    final62: dict[str, Any],
    main: dict[str, Any],
    contract: dict[str, Any],
    final64: dict[str, Any],
    object_bank: dict[str, Any],
    state_counts: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(group: str, name: str, value: Any, source: str) -> None:
        rows.append({"metric_group": group, "metric": name, "value": value, "source": source})

    add("solver", "full_solver_core_purity", solver.get("full_solver_core_purity"), INPUT_PATHS["solver_summary"])
    add("solver", "full_solver_core_completeness", solver.get("full_solver_core_completeness"), INPUT_PATHS["solver_summary"])
    add("solver", "full_solver_real_minus_shuffled_ARI", solver.get("full_solver_real_minus_shuffled_ARI"), INPUT_PATHS["solver_summary"])
    add(
        "solver",
        "full_solver_real_minus_no_temporal_ARI",
        solver.get("full_solver_real_minus_no_temporal_ARI"),
        INPUT_PATHS["solver_summary"],
    )
    add("solver", "S1_minus_S6_core_ARI", solver.get("S1_minus_S6_core_ARI"), INPUT_PATHS["solver_summary"])
    add("v62_final", "decision_label", final62.get("decision_label"), INPUT_PATHS["v62_final"])
    add("v62_final", "blocked_claims", final62.get("blocked_claims"), INPUT_PATHS["v62_final"])
    for key, value in state_counts.items():
        add("native_field", f"{key}_component_count", value, INPUT_PATHS["native_component_rows"])
    add("native_field", "native_observation_row_count", native.get("native_observation_row_count"), INPUT_PATHS["native_summary"])
    add("native_field", "carrier_level_available", native.get("carrier_level_field_available"), INPUT_PATHS["native_summary"])
    add("main_fact_lock", "main_ownership_status", summary.get("main_ownership_status"), INPUT_PATHS["main_fact_lock"])
    add("main_fact_lock", "core_purity", summary.get("core_purity"), INPUT_PATHS["main_fact_lock"])
    add("main_fact_lock", "core_completeness", summary.get("core_completeness"), INPUT_PATHS["main_fact_lock"])
    add("main_fact_lock", "state_coverage_rate", summary.get("state_coverage_rate"), INPUT_PATHS["main_fact_lock"])
    add("main_fact_lock", "real_minus_shuffled_ARI", summary.get("real_minus_shuffled_ARI"), INPUT_PATHS["main_fact_lock"])
    add("main_fact_lock", "real_minus_no_temporal_ARI", summary.get("real_minus_no_temporal_ARI"), INPUT_PATHS["main_fact_lock"])
    add("native_contract", "object_count", contract.get("object_count"), INPUT_PATHS["native_contract"])
    add("native_contract", "material_count", contract.get("material_count"), INPUT_PATHS["native_contract"])
    add("native_contract", "confirmed_material_count", contract.get("confirmed_material_count"), INPUT_PATHS["native_contract"])
    add("v64r2_final", "decision_label", final64.get("decision_label"), INPUT_PATHS["v64r2_final"])
    add("v64r2_final", "blocked_claims", final64.get("blocked_claims"), INPUT_PATHS["v64r2_final"])
    add("object_bank", "object_count", object_bank.get("object_count"), INPUT_PATHS["soma_object_bank"])
    add("object_bank", "material_assignment_count", object_bank.get("material_assignment_count"), INPUT_PATHS["soma_object_bank"])
    add("object_bank", "object_support_row_count", object_bank.get("object_support_row_count"), INPUT_PATHS["soma_object_bank"])
    add("object_bank", "objects_with_any_support_count", object_bank.get("objects_with_any_support_count"), INPUT_PATHS["soma_object_bank"])
    add(
        "object_bank",
        "objects_with_native_point_or_carrier_support_count",
        object_bank.get("objects_with_native_point_or_carrier_support_count"),
        INPUT_PATHS["soma_object_bank"],
    )
    add("object_bank", "native_support_join_available", object_bank.get("native_support_join_available"), INPUT_PATHS["soma_object_bank"])
    add("object_bank", "blockers", object_bank.get("blockers"), INPUT_PATHS["soma_object_bank"])
    add("repro_gate", "pass", summary.get("gate", {}).get("pass"), "outputs/audit/v65_soma_non_ap_repro/soma_non_ap_repro_summary.json")
    return rows


def _hash_rows() -> list[dict[str, str]]:
    return [{"path": path, "sha256": sha256_file(path)} for path in INPUT_PATHS.values()]


if __name__ == "__main__":
    main()
