#!/usr/bin/env python3
"""Build ACL2 v94 Phase0 restart evidence lock from measured prior artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")

PRIOR_FINAL_DECISIONS = {
    "v83": Path("results/acl2_v83tf_clue_sufficiency_vs_action_misuse/phase10_decision_matrix/final_decision.json"),
    "v84": Path("results/acl2_v84tf_memory_ruler_audit/phase12_decision_matrix/final_decision.json"),
    "v85": Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/report_final/final_decision.json"),
    "v86": Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/report_final/final_decision.json"),
    "v87": Path("results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/report_final/final_decision.json"),
    "v88": Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/report_final/final_decision.json"),
    "v89": Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control/report_final/final_decision.json"),
    "v90": Path("results/acl2_v90tf_semantic_object_topology_scale_mode_memory_control/report_final/final_decision.json"),
    "v91": Path("results/acl2_v91tf_semantic_topology_regime_adaptive_memory_control/report_final/final_decision.json"),
    "v92": Path("results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery/report_final/final_decision.json"),
}

OPTIONAL_LATEST_PRIORS = {
    "v93": Path("results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier/report_final/final_decision.json"),
}

V92_ROOT = Path("results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery")
V93_ROOT = Path("results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase0_restart_evidence_lock")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit output should expose malformed inputs.
        return {"read_error": str(exc), "path": str(path)}
    return data if isinstance(data, dict) else {"value": data}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def b(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def first_present(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return ""


def matrix_row(version: str, path: Path, data: dict[str, Any], required: bool) -> dict[str, Any]:
    return {
        "version": version,
        "required_by_v94_plan": required,
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
        "final_status": first_present(data, ["final_status", "overall_status", "decision", "status"]),
        "blocker": data.get("blocker", ""),
        "runtime_action_allowed": data.get("runtime_action_allowed", ""),
        "runtime_action_executed": data.get("runtime_action_executed", ""),
        "counterfactual_executed": data.get("counterfactual_executed", ""),
        "ttt_allowed": data.get("ttt_allowed", ""),
        "phase0_gate_pass": data.get("phase0_gate_pass", ""),
        "decision_labels": ";".join(str(x) for x in (data.get("decision_labels") or data.get("labels") or [])),
        "read_error": data.get("read_error", ""),
    }


def main() -> None:
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    required_rows: list[dict[str, Any]] = []
    final_payloads: dict[str, dict[str, Any]] = {}
    for version, path in PRIOR_FINAL_DECISIONS.items():
        payload = read_json(path)
        final_payloads[version] = payload
        required_rows.append(matrix_row(version, path, payload, required=True))
    optional_rows: list[dict[str, Any]] = []
    for version, path in OPTIONAL_LATEST_PRIORS.items():
        payload = read_json(path)
        final_payloads[version] = payload
        optional_rows.append(matrix_row(version, path, payload, required=False))

    v92_phase1 = read_json(V92_ROOT / "phase1_semantic_policy_row_bank/phase1_gate_summary.json")
    v92_phase2 = read_json(V92_ROOT / "phase2_boundary_trace_ledger/phase2_gate_summary.json")
    v92_phase4 = read_json(
        V92_ROOT / "phase4_swa_semantic_policy_route_audit/phase4_swa_carrier_audit/phase4_swa_carrier_summary.json"
    )
    v92_phase7 = read_json(V92_ROOT / "phase7_data_source_expansion/semantic_source_expansion_candidate_summary.json")
    v93_phase3 = read_json(V93_ROOT / "phase3_merge_gauge_trace_audit/phase3_trace_availability_summary.json")
    v93_phase5 = read_json(V93_ROOT / "phase5_merge_gauge_counterfactual_upper_bound/counterfactual_upper_bound_summary.json")
    v93_phase7 = read_json(V93_ROOT / "phase7_swa_secondary_carrier/carrier_audit/phase7_swa_secondary_summary.json")

    missing_required = [row["version"] for row in required_rows if not row["exists"]]
    runtime_ttt_blocked = all(
        row["exists"] and b(row.get("runtime_action_allowed")) is False and b(row.get("ttt_allowed")) is False
        for row in required_rows
    )
    v92_policy_pass = b(v92_phase1.get("phase1_semantic_policy_row_bank_gate_pass"))
    v92_carrier_failure_locked = (
        b(final_payloads["v92"].get("runtime_action_allowed")) is False
        and b(final_payloads["v92"].get("ttt_allowed")) is False
        and (
            b(v92_phase2.get("phase2_boundary_trace_availability_gate_pass")) is False
            or b(v92_phase4.get("phase4_swa_carrier_gate_pass")) is False
            or "CARRIER" in str(final_payloads["v92"].get("decision_labels", "")).upper()
        )
    )

    forbidden_repeats = [
        "repeat_semantic_score_threshold_sweep_without_true_carrier_trace",
        "repeat_swa_source_query_pair_mask_alpha_sweep_from_v92_v93",
        "treat_boundary_proxy_as_true_merge_gauge_carrier",
        "claim_component_or_topology_proxy_as_object_identity_without_row_join",
        "promote_feature_match_valid_ratio_as_scale_ruler",
        "run_runtime_action_before_counterfactual_upper_bound_pass",
        "run_ttt_or_write_strength_sweep_without_confirmed_carrier_runtime_pass",
        "claim_trace_level_counterfactual_as runtime trajectory improvement",
    ]
    hypotheses = [
        {
            "hypothesis": "H1_boundary_gauge_update_error",
            "v94_question": "Are most long-sequence errors local chunk failures or handoff scale/gauge failures?",
            "required_phase": "Phase1_boundary_failure_atlas",
            "entry_status": "open",
        },
        {
            "hypothesis": "H2_true_memory_carrier_localizable",
            "v94_question": "Which READ/SWA/merge-gauge/TTT body is causally sensitive to boundary geometry?",
            "required_phase": "Phase2_true_trace_then_Phase3_sensitivity",
            "entry_status": "open",
        },
        {
            "hypothesis": "H3_semantics_explain_evidence_trust",
            "v94_question": "Can semantic taxonomy explain unreliable carrier updates rather than directly predict scale?",
            "required_phase": "Phase4_taxonomy_then_Phase5_alignment",
            "entry_status": "open",
        },
        {
            "hypothesis": "H4_proxy_trace_cannot_unlock_action",
            "v94_question": "True carrier trace must cover merge/gauge required fields before action phases.",
            "required_phase": "Phase2_gate",
            "entry_status": "locked_constraint",
        },
        {
            "hypothesis": "H5_ttt_last_not_rescue",
            "v94_question": "TTT remains diagnostic until confirmed runtime carrier evidence exists.",
            "required_phase": "Phase8_only_after_runtime_pass",
            "entry_status": "locked_constraint",
        },
    ]
    still_open_questions = [
        "What fraction of boundary rows are LOCAL_BAD vs HANDOFF_SCALE/HANDOFF_GAUGE?",
        "Does merge/gauge true trace now cover boundary_update_norm and merge_residual_delta in the v94 boundary atlas?",
        "Which memory body changes geometry under neutral intervention without hurting good rows?",
        "Can semantic evidence categories beat geometry-only and shuffle controls within conflict rows?",
        "Does any semantic-carrier counterfactual upper bound pass good-protection controls?",
        "Is runtime action legal after Phase6, and is TTT still blocked?",
    ]

    v93_latest = final_payloads.get("v93", {})
    facts = {
        "v83_v92_required_final_decisions_found": len(missing_required) == 0,
        "missing_required_versions": missing_required,
        "v92_semantic_policy_pass_locked": v92_policy_pass,
        "v92_phase1_bad_recall": v92_phase1.get("bad_recall"),
        "v92_phase1_good_FPR": v92_phase1.get("good_FPR"),
        "v92_phase1_semantic_shuffle_margin": v92_phase1.get("semantic_shuffle_margin"),
        "v92_phase2_trace_gate_pass": v92_phase2.get("phase2_boundary_trace_availability_gate_pass"),
        "v92_true_trace_ratio": v92_phase2.get("true_trace_ratio"),
        "v92_boundary_update_norm_available_ratio": v92_phase2.get("boundary_update_norm_available_ratio"),
        "v92_merge_residual_delta_available_ratio": v92_phase2.get("merge_residual_delta_available_ratio"),
        "v92_swa_carrier_gate_pass": v92_phase4.get("phase4_swa_carrier_gate_pass"),
        "v92_swa_blocker": v92_phase4.get("blocker"),
        "v92_object_identity_available_ratio": v92_phase7.get("object_identity_available_ratio"),
        "v92_has_radio_ratio": v92_phase7.get("has_radio_ratio"),
        "v92_carrier_failure_locked": v92_carrier_failure_locked,
        "runtime_and_ttt_blocked_for_required_priors": runtime_ttt_blocked,
        "optional_v93_final_status": v93_latest.get("final_status"),
        "optional_v93_blocker": v93_latest.get("blocker"),
        "optional_v93_phase3_trace_gate_pass": v93_phase3.get("phase3_trace_availability_gate_pass"),
        "optional_v93_boundary_update_norm_available_ratio": v93_phase3.get("boundary_update_norm_available_ratio"),
        "optional_v93_merge_residual_delta_available_ratio": v93_phase3.get("merge_residual_delta_available_ratio"),
        "optional_v93_counterfactual_gate_pass": v93_phase5.get("phase5_counterfactual_gate_pass"),
        "optional_v93_counterfactual_blocker": v93_phase5.get("blocker"),
        "optional_v93_swa_secondary_gate_pass": v93_phase7.get("phase7_swa_secondary_carrier_gate_pass"),
        "optional_v93_swa_secondary_blocker": v93_phase7.get("blocker"),
    }
    checks = {
        "all_required_prior_final_decisions_exist": len(missing_required) == 0,
        "v92_semantic_policy_pass_locked": v92_policy_pass,
        "v92_carrier_failure_locked": v92_carrier_failure_locked,
        "forbidden_repeats_non_empty": len(forbidden_repeats) > 0,
        "hypotheses_h1_h5_written": len(hypotheses) == 5,
        "runtime_and_ttt_blocked": runtime_ttt_blocked,
    }
    phase0_gate_pass = all(checks.values())
    summary = {
        "phase": "Phase0_restart_evidence_lock",
        "phase0_gate_pass": phase0_gate_pass,
        "blocker": "" if phase0_gate_pass else "phase0_restart_evidence_lock_failed",
        "checks": checks,
        "facts": facts,
        "required_prior_versions": list(PRIOR_FINAL_DECISIONS),
        "optional_latest_prior_versions_recorded": [row["version"] for row in optional_rows if row["exists"]],
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }
    restart_boundary = {
        "restart_reason": "v94 changes from semantic score or route-mask sweep to causal failure localization.",
        "training_free": True,
        "no_gt_runtime_feature": True,
        "no_explicit_runtime_scale_correction": True,
        "plan_required_prior_scope": "v83-v92",
        "optional_latest_v93_included_as_context": bool(optional_rows and optional_rows[0]["exists"]),
        "v83_v92_no_go_matrix": required_rows,
        "latest_v93_context": optional_rows,
        "facts": facts,
        "forbidden_repeats": forbidden_repeats,
        "still_open_questions": still_open_questions,
    }

    write_json(out / "restart_boundary.json", restart_boundary)
    write_csv(out / "prior_no_go_matrix.csv", required_rows + optional_rows)
    (out / "forbidden_repeats.md").write_text(
        "\n".join(["# v94 Forbidden Repeats", "", *[f"- {item}" for item in forbidden_repeats]]) + "\n",
        encoding="utf-8",
    )
    (out / "still_open_questions.md").write_text(
        "\n".join(["# v94 Still Open Questions", "", *[f"- {item}" for item in still_open_questions]]) + "\n",
        encoding="utf-8",
    )
    write_csv(out / "v94_hypothesis_matrix.csv", hypotheses)
    write_json(out / "phase0_gate_summary.json", summary)
    write_json(
        out / "phase0_source_payload_digest.json",
        {
            "required_final_decisions": {version: matrix_row(version, path, final_payloads[version], True) for version, path in PRIOR_FINAL_DECISIONS.items()},
            "optional_latest_priors": {version: matrix_row(version, path, final_payloads[version], False) for version, path in OPTIONAL_LATEST_PRIORS.items()},
            "v92_phase1": v92_phase1,
            "v92_phase2": v92_phase2,
            "v92_phase4": v92_phase4,
            "v92_phase7": v92_phase7,
            "v93_phase3": v93_phase3,
            "v93_phase5": v93_phase5,
            "v93_phase7": v93_phase7,
        },
    )

    print(f"phase0_gate_pass={phase0_gate_pass}")
    print(f"missing_required_versions={','.join(missing_required) if missing_required else 'none'}")
    print(f"v92_semantic_policy_pass_locked={v92_policy_pass}")
    print(f"v92_carrier_failure_locked={v92_carrier_failure_locked}")
    print(f"runtime_and_ttt_blocked={runtime_ttt_blocked}")
    print(f"optional_v93_final_status={v93_latest.get('final_status', '')}")
    if summary["blocker"]:
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
