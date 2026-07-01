from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


SUMMARY_PATHS = {
    "phase0_fact_lock": "outputs/audit/v63_phase0_fact_lock/fact_lock_summary.json",
    "phase1_query_candidates": "outputs/audit/v63_query_candidates/query_candidate_summary.json",
    "phase2_query_policy": "outputs/audit/v63_query_policy/query_policy_summary.json",
    "phase3_d4rt_query": "outputs/audit/v63_d4rt_query/query_execution_summary.json",
    "phase4_action_outcome": "outputs/audit/v63_action_outcome/action_outcome_summary.json",
}


@dataclass(frozen=True)
class V63FullEvalConfig:
    output_root: str | Path = "outputs/audit/v63_final"
    visualization_root: str | Path = "outputs/audit/v63_visualizations"


def build_v63_final_decision(config: V63FullEvalConfig | None = None) -> dict[str, Any]:
    cfg = config or V63FullEvalConfig()
    summaries = {key: _read(path) for key, path in SUMMARY_PATHS.items()}
    gates = {key: bool((summary.get("gate") or {}).get("pass")) for key, summary in summaries.items()}
    action = summaries["phase4_action_outcome"]
    action_gate = action.get("gate") or {}
    phase4_pass = gates["phase4_action_outcome"]
    blocked_claims = []
    if not phase4_pass:
        blocked_claims.append("active_query_method_contribution_claim")
    blocked_claims.extend(
        [
            "interventional_ownership_update_claim",
            "heldout_future_decoy_validation_claim",
            "native_query_field_extension_claim",
            "native_AP_or_mesh_materialization_claim",
        ]
    )
    decision_label = "NO_GO_V63_ACTIVE_QUERY_UTILITY" if not phase4_pass else "GO_V63_ACTIVE_QUERY_UTILITY"
    final_decision = {
        "phase": "v63_final",
        "created_at": utc_now(),
        "decision_label": decision_label,
        "plan_path": "docs/stream4d_v63_soma_query_interventional_material_evidence_plan.md",
        "claim_table": {
            "Claim A/B/D carryover": {
                "label": "UNCHANGED_FROM_V62_VERIFIED_OWNERSHIP_FIELD",
                "pass": True,
                "evidence": "v63 Phase 0 fact-lock preserved v62 final state; no v63 update was applied to ownership field.",
            },
            "Claim C active query": {
                "label": "NO_GO_ACTIVE_QUERY_METHOD_CONTRIBUTION" if not phase4_pass else "GO_ACTIVE_QUERY_METHOD_CONTRIBUTION",
                "pass": phase4_pass,
                "evidence": "Phase 4 action utility must beat fixed/no-temporal controls; current gate failed.",
            },
            "Interventional update": {
                "label": "NOT_RUN_DUE_PHASE4_NO_GO",
                "pass": False,
                "evidence": "Plan stop rule blocks state update when action outcomes do not beat controls.",
            },
            "Heldout/future/decoy validation": {
                "label": "NOT_RUN_DUE_PHASE4_NO_GO",
                "pass": False,
                "evidence": "Would be required for a method contribution, but Phase 4 gate failed first.",
            },
            "Native AP diagnostic": {
                "label": "NOT_RUN",
                "pass": False,
                "evidence": "AP is diagnostic only and was not promoted or run in v63.",
            },
        },
        "phase_gates": gates,
        "blocked_claims": blocked_claims,
        "stop_rule_triggered": {
            "rule": "If action outcomes do not beat fixed/random controls in utility, block Claim C.",
            "triggered": not phase4_pass,
            "evidence": "outputs/audit/v63_action_outcome/action_outcome_summary.json",
        },
        "key_metrics": {
            "R0_valid_material_evidence_rate": (action.get("r0_metrics") or {}).get("valid_material_evidence_rate"),
            "R0_confirm_or_quarantine_rate": (action.get("r0_metrics") or {}).get("confirm_or_quarantine_rate"),
            "R0_mean_action_utility": (action.get("r0_metrics") or {}).get("mean_action_utility"),
            "best_fixed_control_confirm_or_quarantine_rate": action.get("best_fixed_control_confirm_or_quarantine_rate"),
            "best_fixed_control_mean_action_utility": action.get("best_fixed_control_mean_action_utility"),
            "real_minus_best_fixed_utility": action.get("real_minus_best_fixed_utility"),
            "real_minus_shuffled_query_utility": action.get("real_minus_shuffled_query_utility"),
            "real_minus_no_temporal_query_utility": action.get("real_minus_no_temporal_query_utility"),
            "false_confirm_rate": (action.get("r0_metrics") or {}).get("false_confirm_rate"),
            "ap_status": "not_run",
            "native_query_ledger_available": False,
        },
        "final_gate": {
            "query_valid_material_evidence_rate_ge_0_50": action_gate.get("query_valid_material_evidence_rate_ge_0_50"),
            "query_to_confirm_or_quarantine_rate_ge_best_fixed_plus_0_15": action_gate.get("query_to_confirm_or_quarantine_rate_ge_best_fixed_plus_0_15"),
            "false_confirm_rate_le_0_05": action_gate.get("false_confirm_rate_le_0_05"),
            "real_minus_shuffled_query_utility_ge_0_10": action_gate.get("real_minus_shuffled_query_utility_ge_0_10"),
            "real_minus_no_temporal_query_utility_ge_0_05": action_gate.get("real_minus_no_temporal_query_utility_ge_0_05"),
            "real_minus_best_fixed_utility_positive": action_gate.get("real_minus_best_fixed_utility_positive"),
            "native_query_ledger_available": False,
            "pass": False,
        },
        "phase5_to_phase8_status": "not_run_due_phase4_action_utility_no_go",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
        "input_paths": SUMMARY_PATHS,
    }
    return {
        "final_decision": final_decision,
        "final_metric_rows": _final_metric_rows(summaries, final_decision),
    }


def write_v63_final_decision(result: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    root = _project(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "final_decision": root / "final_decision.json",
        "final_metric_rows": root / "final_metric_rows.csv",
    }
    write_json(paths["final_decision"], result["final_decision"])
    write_csv(paths["final_metric_rows"], result["final_metric_rows"])
    return {key: _rel(path) for key, path in paths.items()}


def build_v63_visual_dashboard(
    final_decision_path: str | Path = "outputs/audit/v63_final/final_decision.json",
    output_path: str | Path = "outputs/audit/v63_visualizations/v63_dashboard.html",
) -> dict[str, str]:
    final = read_json(_project(final_decision_path))
    output = _project(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    claim_rows = "".join(
        f"<tr><td>{name}</td><td>{payload.get('label')}</td><td>{'PASS' if payload.get('pass') else 'NO-GO'}</td><td>{payload.get('evidence')}</td></tr>"
        for name, payload in final.get("claim_table", {}).items()
    )
    metric_rows = "".join(f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in final.get("key_metrics", {}).items())
    gate_rows = "".join(f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in final.get("final_gate", {}).items())
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stream4D v63 Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #20242a; }}
    h1 {{ font-size: 26px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d7dce2; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f6; }}
    code {{ background: #f4f6f8; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Stream4D v63 SOMA-Query</h1>
  <p>Decision: <code>{final.get('decision_label')}</code></p>
  <h2>Claims</h2>
  <table><thead><tr><th>Claim</th><th>Label</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{claim_rows}</tbody></table>
  <h2>Key Metrics</h2>
  <table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{metric_rows}</tbody></table>
  <h2>Final Gate</h2>
  <table><thead><tr><th>Gate</th><th>Value</th></tr></thead><tbody>{gate_rows}</tbody></table>
  <h2>Stop Rule</h2>
  <p><code>{final.get('stop_rule_triggered', {}).get('rule')}</code></p>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return {"dashboard": _rel(output)}


def _final_metric_rows(summaries: dict[str, dict[str, Any]], final_decision: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, summary in summaries.items():
        gate = summary.get("gate") or {}
        rows.append(
            {
                "row_id": phase,
                "phase": phase,
                "method_status": summary.get("method_status", ""),
                "gate_pass": gate.get("pass"),
                "uses_gt_for_prediction": summary.get("uses_gt_for_prediction", False),
                "uses_gt_for_diagnostic_labels": summary.get("uses_gt_for_diagnostic_labels", False),
            }
        )
    for key, value in final_decision.get("key_metrics", {}).items():
        rows.append(
            {
                "row_id": f"metric_{key}",
                "phase": "v63_final",
                "metric_name": key,
                "metric_value": value,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": False,
            }
        )
    return rows


def _read(path: str) -> dict[str, Any]:
    full = ROOT / path
    return read_json(full) if full.exists() else {}


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)
