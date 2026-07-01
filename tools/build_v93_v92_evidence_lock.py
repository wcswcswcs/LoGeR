#!/usr/bin/env python3
"""Build v93 Phase0 evidence lock from measured v92 artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v86_soft_latent_utils import read_json, write_csv, write_json  # noqa: E402
from tools.v93_semantic_object_identity_utils import ROOT, V92_ROOT, sha256_file  # noqa: E402


REQUIRED_INPUTS = [
    "report_final/final_decision.json",
    "report_final/final_report.md",
    "phase1_semantic_policy_row_bank/phase1_gate_summary.json",
    "phase1_semantic_policy_row_bank/semantic_policy_rows.csv",
    "phase2_boundary_trace_ledger/phase2_gate_summary.json",
    "phase2_boundary_trace_ledger/boundary_trace_rows.csv",
    "phase2_boundary_trace_ledger/hidden_boundary_field_audit.json",
    "phase2_boundary_trace_ledger/noop_trace_smoke_summary.json",
    "phase4_swa_semantic_policy_route_audit/phase4_swa_carrier_audit/phase4_swa_carrier_summary.json",
    "phase7_data_source_expansion/semantic_source_expansion_candidate_summary.json",
    "phase7_data_source_expansion/radio_tracklet_sidecar_summary.json",
    "phase7_data_source_expansion/expanded_semantic_policy_summary.json",
    "phase7_data_source_expansion/semantic_source_expansion_candidate_rows.csv",
    "phase9_visual_rediscovery_or_blocked/visual_rediscovery_summary.json",
    "phase9_visual_rediscovery_or_blocked/visual_requirement_matrix.csv",
]

FORBIDDEN_REPEATS = [
    "compact_component_threshold_sweep",
    "source_side_semantic_mask_route_boost",
    "swa_query_pair_alpha_sweep",
    "same_policy_row_bank_cutoff_retuning",
    "treat_boundary_proxy_as_true_merge_gauge_carrier",
    "direct_ttt_without_confirmed_carrier_runtime_action",
    "component_proxy_success_claim_without_object_identity_or_radio_join",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v92-root", type=Path, default=V92_ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase0_v92_evidence_lock")
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _section_has_field(hidden: dict[str, Any], key: str) -> bool:
    sections = hidden.get("sections", {})
    if not isinstance(sections, dict):
        return False
    return any(bool(sec.get(key)) for sec in sections.values() if isinstance(sec, dict))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    v92_root = args.v92_root

    final = _json(v92_root / "report_final/final_decision.json")
    phase1 = _json(v92_root / "phase1_semantic_policy_row_bank/phase1_gate_summary.json")
    phase2 = _json(v92_root / "phase2_boundary_trace_ledger/phase2_gate_summary.json")
    hidden = _json(v92_root / "phase2_boundary_trace_ledger/hidden_boundary_field_audit.json")
    noop = _json(v92_root / "phase2_boundary_trace_ledger/noop_trace_smoke_summary.json")
    phase4 = _json(v92_root / "phase4_swa_semantic_policy_route_audit/phase4_swa_carrier_audit/phase4_swa_carrier_summary.json")
    source = _json(v92_root / "phase7_data_source_expansion/semantic_source_expansion_candidate_summary.json")
    sidecar = _json(v92_root / "phase7_data_source_expansion/radio_tracklet_sidecar_summary.json")
    expanded = _json(v92_root / "phase7_data_source_expansion/expanded_semantic_policy_summary.json")

    required_rows = []
    missing = []
    for rel in REQUIRED_INPUTS:
        path = v92_root / rel
        exists = path.exists()
        if not exists:
            missing.append(rel)
        required_rows.append(
            {
                "path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists and path.is_file() else "",
                "sha256": sha256_file(path) if exists and path.is_file() else "",
            }
        )

    query = phase4.get("query", {}) if isinstance(phase4.get("query"), dict) else {}
    pair = phase4.get("pair", {}) if isinstance(phase4.get("pair"), dict) else {}
    facts = {
        "v92_final_status": final.get("final_status"),
        "v92_blocker": final.get("blocker"),
        "phase1_gate_pass": bool(phase1.get("phase1_semantic_policy_row_bank_gate_pass")),
        "phase2_true_trace_ratio": phase2.get("true_trace_ratio"),
        "phase2_boundary_update_norm_available_ratio": phase2.get("boundary_update_norm_available_ratio"),
        "phase2_merge_residual_delta_available_ratio": phase2.get("merge_residual_delta_available_ratio"),
        "hidden_boundary_update_norm_found": _section_has_field(hidden, "has_boundary_update_norm"),
        "hidden_merge_residual_delta_found": _section_has_field(hidden, "has_merge_residual_delta"),
        "noop_all_completed": bool(noop.get("all_completed")),
        "noop_non_identity_transform_rows": noop.get("non_identity_transform_rows"),
        "noop_residual_field_rows": noop.get("residual_field_rows"),
        "phase4_query_action_fidelity_all": query.get("action_fidelity_all"),
        "phase4_query_actual_minus_random_p95": query.get("actual_minus_random_p95"),
        "phase4_pair_action_fidelity_all": pair.get("action_fidelity_all"),
        "phase4_pair_actual_minus_random_p95": pair.get("actual_minus_random_p95"),
        "phase4_pair_semantic_shuffle_margin": pair.get("semantic_shuffle_margin"),
        "phase4_pair_component_shuffle_margin": pair.get("component_shuffle_margin"),
        "phase4_pair_regime_shuffle_margin": pair.get("regime_shuffle_margin"),
        "phase7_object_identity_available_ratio": source.get("object_identity_available_ratio"),
        "phase7_has_radio_ratio": source.get("has_radio_ratio"),
        "phase7_component_tracklet_available_ratio": source.get("component_tracklet_available_ratio"),
        "phase7_expanded_policy_gate_pass": bool(expanded.get("phase7_expanded_policy_gate_pass")),
        "phase7_data_source_expansion_useful": bool(expanded.get("phase7_data_source_expansion_useful")),
        "runtime_action_allowed": bool(final.get("runtime_action_allowed")),
        "counterfactual_executed": bool(final.get("counterfactual_executed")),
        "ttt_allowed": bool(final.get("ttt_allowed")),
        "radio_sidecar_phase": sidecar.get("phase"),
    }

    checks = {
        "required_inputs_exist": len(missing) == 0,
        "final_status_locked": facts["v92_final_status"] == "NO_GO_SEMANTIC_SOURCE_SPECIFICITY_INSUFFICIENT",
        "phase1_pass_locked": facts["phase1_gate_pass"] is True,
        "phase2_true_trace_only_4_of_49": abs(float(facts["phase2_true_trace_ratio"] or 0.0) - (4.0 / 49.0)) < 1e-9,
        "phase2_merge_residual_delta_zero": float(facts["phase2_merge_residual_delta_available_ratio"] or 0.0) == 0.0,
        "hidden_fields_absent": not facts["hidden_boundary_update_norm_found"] and not facts["hidden_merge_residual_delta_found"],
        "noop_identity_only": facts["noop_all_completed"] is True
        and int(facts["noop_non_identity_transform_rows"] or 0) == 0
        and int(facts["noop_residual_field_rows"] or 0) == 0,
        "phase4_swa_tiny_or_not_specific": bool(query.get("action_fidelity_all"))
        and bool(pair.get("action_fidelity_all"))
        and float(abs(query.get("actual_minus_random_p95") or 0.0)) < 0.01
        and float(abs(pair.get("actual_minus_random_p95") or 0.0)) < 0.01
        and min(
            float(pair.get("semantic_shuffle_margin") or 0.0),
            float(pair.get("component_shuffle_margin") or 0.0),
            float(pair.get("regime_shuffle_margin") or 0.0),
        )
        < 0.05,
        "phase7_source_specificity_insufficient": float(source.get("object_identity_available_ratio") or 0.0) == 0.0
        and float(source.get("has_radio_ratio") or 0.0) < 0.30
        and bool(expanded.get("phase7_data_source_expansion_useful")) is False,
        "runtime_counterfactual_ttt_blocked": not facts["runtime_action_allowed"]
        and not facts["counterfactual_executed"]
        and not facts["ttt_allowed"],
    }
    gate_pass = all(checks.values())
    summary = {
        "phase": "Phase0_v92_evidence_lock_for_v93",
        "phase0_gate_pass": gate_pass,
        "blocker": "" if gate_pass else "phase0_v92_evidence_lock_failed",
        "required_input_missing": missing,
        "checks": checks,
        "facts": facts,
        "forbidden_repeats": FORBIDDEN_REPEATS,
        "v92_root": str(v92_root),
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }

    write_csv(args.out_dir / "required_inputs.csv", required_rows)
    write_json(args.out_dir / "evidence_lock.json", {"facts": facts, "checks": checks, "summary": summary})
    write_json(args.out_dir / "phase0_gate_summary.json", summary)
    write_csv(
        args.out_dir / "v93_hypothesis_matrix.csv",
        [
            {
                "hypothesis": "H1_object_identity_source_specificity",
                "phase": "Phase1_Phase2",
                "current_status": "requires_v93_row_level_join",
                "locked_v92_evidence": "phase7 object_identity_available_ratio=0.0",
            },
            {
                "hypothesis": "H2_merge_gauge_boundary_carrier",
                "phase": "Phase3_Phase4",
                "current_status": "requires_true_trace_instrumentation",
                "locked_v92_evidence": "phase2 true_trace_ratio=4/49 and merge_residual_delta_available_ratio=0.0",
            },
            {
                "hypothesis": "H3_object_topology_explains_geometry_conflict",
                "phase": "Phase2",
                "current_status": "requires_object_or_radio_source",
                "locked_v92_evidence": "component proxy alone did not pass expanded policy gate",
            },
            {
                "hypothesis": "H4_counterfactual_before_runtime",
                "phase": "Phase5_Phase6",
                "current_status": "blocked_until_carrier_pass",
                "locked_v92_evidence": "counterfactual_executed=false runtime_action_allowed=false",
            },
            {
                "hypothesis": "H5_ttt_only_after_confirmed_runtime_carrier",
                "phase": "Phase8",
                "current_status": "blocked",
                "locked_v92_evidence": "ttt_allowed=false",
            },
        ],
    )
    write_csv(args.out_dir / "forbidden_repeats.csv", [{"forbidden_repeat": item} for item in FORBIDDEN_REPEATS])
    (args.out_dir / "forbidden_repeats.md").write_text(
        "\n".join(["# v93 Forbidden Repeats", "", *[f"- {item}" for item in FORBIDDEN_REPEATS]]) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "v92_no_go_boundary.md").write_text(
        "\n".join(
            [
                "# v92 No-Go Boundary Locked for v93",
                "",
                f"- final_status: `{facts['v92_final_status']}`",
                f"- blocker: `{facts['v92_blocker']}`",
                f"- Phase1 policy row-bank pass: `{facts['phase1_gate_pass']}`",
                f"- true boundary trace ratio: `{facts['phase2_true_trace_ratio']}` (= 4/49 required lock)",
                f"- merge_residual_delta_available_ratio: `{facts['phase2_merge_residual_delta_available_ratio']}`",
                f"- hidden boundary_update_norm found: `{facts['hidden_boundary_update_norm_found']}`",
                f"- hidden merge_residual_delta found: `{facts['hidden_merge_residual_delta_found']}`",
                f"- no-op smoke completed: `{facts['noop_all_completed']}`",
                f"- no-op non_identity_transform_rows: `{facts['noop_non_identity_transform_rows']}`",
                f"- Phase4 query actual_minus_random_p95: `{facts['phase4_query_actual_minus_random_p95']}`",
                f"- Phase4 pair actual_minus_random_p95: `{facts['phase4_pair_actual_minus_random_p95']}`",
                f"- Phase7 object_identity_available_ratio: `{facts['phase7_object_identity_available_ratio']}`",
                f"- Phase7 has_radio_ratio: `{facts['phase7_has_radio_ratio']}`",
                f"- Phase7 component_tracklet_available_ratio: `{facts['phase7_component_tracklet_available_ratio']}`",
                f"- runtime_action_allowed: `{facts['runtime_action_allowed']}`",
                f"- counterfactual_executed: `{facts['counterfactual_executed']}`",
                f"- ttt_allowed: `{facts['ttt_allowed']}`",
                "",
                "Conclusion: v93 may not repeat compact component / SWA route-lift sweeps. It must first try row-level object/RADIO joins and true merge/gauge trace instrumentation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"phase0_gate_pass={gate_pass}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
