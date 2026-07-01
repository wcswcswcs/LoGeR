from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_bool, parse_float, read_csv, utc_now, write_csv, write_json


DEFAULT_ACTION_ROWS = "outputs/audit/v63_action_outcome/action_outcome_rows.csv"
DEFAULT_UTILITY_ROWS = "outputs/audit/v63_action_outcome/action_utility_rows.csv"


@dataclass(frozen=True)
class V63ActionUtilityRepairAuditConfig:
    action_outcome_rows: str | Path = DEFAULT_ACTION_ROWS
    action_utility_rows: str | Path = DEFAULT_UTILITY_ROWS
    output_root: str | Path = "outputs/audit/v63_action_utility_repair_audit"
    cq_margin: float = 0.15


def build_v63_action_utility_repair_audit(config: V63ActionUtilityRepairAuditConfig | None = None) -> dict[str, Any]:
    cfg = config or V63ActionUtilityRepairAuditConfig()
    action_rows = read_csv(_project(cfg.action_outcome_rows))
    utility_rows = read_csv(_project(cfg.action_utility_rows))
    utility_by_policy = {row.get("policy_id", ""): row for row in utility_rows}
    r0 = utility_by_policy.get("R0_real_policy", {})
    fixed = [row for row in utility_rows if row.get("policy_id", "").startswith("C")]
    best_fixed_cq = max((parse_float(row.get("confirm_or_quarantine_rate")) for row in fixed), default=0.0)
    best_fixed_cq_policy = _best_policy(fixed, "confirm_or_quarantine_rate")
    best_fixed_utility = max((parse_float(row.get("mean_action_utility")) for row in fixed), default=0.0)
    best_fixed_utility_policy = _best_policy(fixed, "mean_action_utility")
    r0_cq = parse_float(r0.get("confirm_or_quarantine_rate"))
    r0_utility = parse_float(r0.get("mean_action_utility"))
    cq_threshold = best_fixed_cq + float(cfg.cq_margin)
    all_defer_controls = _all_defer_control_rows(utility_rows)
    no_temporal_probe = _no_temporal_probe(action_rows)
    temporal_rows = _temporal_evidence_rows(action_rows)
    protocol_findings = {
        "raw_cq_gate_threshold": cq_threshold,
        "raw_cq_gate_threshold_exceeds_one": cq_threshold > 1.0,
        "raw_cq_best_fixed_policy": best_fixed_cq_policy,
        "raw_cq_best_fixed_rate": best_fixed_cq,
        "raw_cq_r0_rate": r0_cq,
        "all_defer_controls_with_high_utility": all_defer_controls,
        "best_fixed_utility_policy": best_fixed_utility_policy,
        "best_fixed_utility": best_fixed_utility,
        "r0_utility": r0_utility,
        "no_temporal_source_only_probe": no_temporal_probe,
    }
    summary = {
        "phase": "v63_action_utility_repair_audit",
        "created_at": utc_now(),
        "method_status": "post_no_go_protocol_repair_audit_not_promoted",
        "input_paths": {
            "action_outcome_rows": _rel(cfg.action_outcome_rows),
            "action_utility_rows": _rel(cfg.action_utility_rows),
        },
        "protocol_findings": protocol_findings,
        "repair_assessment": _repair_assessment(protocol_findings),
        "gate": {
            "raw_cq_gate_feasible": cq_threshold <= 1.0,
            "all_defer_controls_do_not_dominate_utility": not bool(all_defer_controls),
            "no_temporal_control_requires_temporal_evidence": bool(no_temporal_probe.get("temporal_required_control_would_be_lower")),
            "canonical_promotion_allowed": False,
            "pass": False,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }
    return {
        "summary": summary,
        "protocol_finding_rows": _finding_rows(protocol_findings),
        "temporal_evidence_rows": temporal_rows,
    }


def write_v63_action_utility_repair_audit(
    result: dict[str, Any],
    config: V63ActionUtilityRepairAuditConfig | None = None,
) -> dict[str, str]:
    cfg = config or V63ActionUtilityRepairAuditConfig()
    output_root = _project(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "repair_audit_summary.json"
    finding_rows_path = output_root / "protocol_finding_rows.csv"
    temporal_rows_path = output_root / "temporal_evidence_rows.csv"
    write_json(summary_path, result["summary"])
    write_csv(finding_rows_path, result["protocol_finding_rows"])
    write_csv(temporal_rows_path, result["temporal_evidence_rows"])
    return {
        "repair_audit_summary": _rel(summary_path),
        "protocol_finding_rows": _rel(finding_rows_path),
        "temporal_evidence_rows": _rel(temporal_rows_path),
    }


def _best_policy(rows: list[dict[str, str]], metric: str) -> str:
    if not rows:
        return ""
    return max(rows, key=lambda row: parse_float(row.get(metric))).get("policy_id", "")


def _all_defer_control_rows(utility_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in utility_rows:
        policy = row.get("policy_id", "")
        if not policy.startswith("C"):
            continue
        outcomes = _counter_from_string(row.get("action_outcome_counts", ""))
        actions = _counter_from_string(row.get("planned_action_counts", ""))
        query_count = int(parse_float(row.get("query_count")))
        mean_utility = parse_float(row.get("mean_action_utility"))
        if (
            mean_utility > 1.0
            and outcomes == {"defer": query_count}
            and (actions.get("defer", 0) == query_count or actions.get("control_semantic_only", 0) == query_count)
        ):
            out.append(
                {
                    "policy_id": policy,
                    "query_count": query_count,
                    "mean_action_utility": mean_utility,
                    "confirm_or_quarantine_rate": parse_float(row.get("confirm_or_quarantine_rate")),
                    "interpretation": "control receives high utility from no-op/all-defer behavior",
                }
            )
    return out


def _no_temporal_probe(action_rows: list[dict[str, str]]) -> dict[str, Any]:
    r2 = [row for row in action_rows if row.get("policy_id") == "R2_no_temporal_source_frame_only"]
    if not r2:
        return {}
    current_utilities = [parse_float(row.get("action_utility")) for row in r2]
    temporal_required_utilities = [_temporal_required_no_temporal_utility(row) for row in r2]
    current_cq = _mean(row.get("action_outcome") in {"confirm", "quarantine"} for row in r2)
    counterfactual_cq = _mean(_temporal_required_no_temporal_outcome(row) in {"confirm", "quarantine"} for row in r2)
    control_would_be_lower = (
        float(np.mean(temporal_required_utilities)) < float(np.mean(current_utilities))
        if current_utilities and temporal_required_utilities
        else False
    )
    interpretation = (
        "Current R2 still lets source-frame evidence drive confirm/quarantine; require temporal evidence for these actions."
        if control_would_be_lower
        else "Current R2 already behaves like a temporal-required no-temporal control for confirm/quarantine."
    )
    return {
        "r2_query_count": len(r2),
        "current_mean_action_utility": float(np.mean(current_utilities)) if current_utilities else None,
        "temporal_required_counterfactual_mean_action_utility": (
            float(np.mean(temporal_required_utilities)) if temporal_required_utilities else None
        ),
        "current_confirm_or_quarantine_rate": current_cq,
        "temporal_required_counterfactual_confirm_or_quarantine_rate": counterfactual_cq,
        "temporal_required_control_would_be_lower": control_would_be_lower,
        "interpretation": interpretation,
    }


def _temporal_required_no_temporal_outcome(row: dict[str, str]) -> str:
    planned_action = row.get("planned_action", "")
    if planned_action == "defer":
        return "defer"
    if planned_action == "reject_decoy":
        return "defer"
    return "defer"


def _temporal_required_no_temporal_utility(row: dict[str, str]) -> float:
    planned_action = row.get("planned_action", "")
    outcome = _temporal_required_no_temporal_outcome(row)
    valid_material_evidence = False
    success = False
    safe_defer = False
    noise_failure = False
    if planned_action == "defer" and outcome == "defer":
        success = True
        safe_defer = True
    elif planned_action == "reject_decoy" and outcome != "confirm":
        success = True
        noise_failure = True
    elif planned_action in {"confirm", "quarantine"}:
        noise_failure = True
    score = 0.0
    if success:
        score += 1.0
    if outcome in {"confirm", "quarantine"} and valid_material_evidence:
        score += 0.25
    if safe_defer:
        score += 0.35
    if noise_failure:
        score -= 0.45
    if planned_action == "reject_decoy" and outcome != "confirm":
        score += 0.25
    score -= 0.05
    return float(score)


def _temporal_evidence_rows(action_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in action_rows:
        policy = row.get("policy_id", "")
        if policy.startswith("R1") or policy.startswith("R2"):
            continue
        grouped.setdefault(policy, []).append(row)
    for policy, group in sorted(grouped.items()):
        non_source_counts = [_non_source_accepted_count(row) for row in group]
        non_source_valid = [count >= 2 and parse_float(row.get("in_bounds_valid_ratio")) >= 0.80 for row, count in zip(group, non_source_counts)]
        rows.append(
            {
                "policy_id": policy,
                "query_count": len(group),
                "original_valid_material_evidence_rate": _mean(parse_bool(row.get("valid_material_evidence")) for row in group),
                "non_source_temporal_valid_rate": _mean(non_source_valid),
                "mean_non_source_accepted_count": float(np.mean(non_source_counts)) if non_source_counts else None,
                "min_non_source_accepted_count": int(np.min(non_source_counts)) if non_source_counts else None,
                "max_non_source_accepted_count": int(np.max(non_source_counts)) if non_source_counts else None,
            }
        )
    return rows


def _non_source_accepted_count(row: dict[str, str]) -> int:
    carrier_path = _project(row.get("carrier_batch_npz", ""))
    if not carrier_path.exists():
        return 0
    query_index = int(parse_float(row.get("d4rt_query_index")))
    support_frame = int(parse_float(row.get("support_frame_id")))
    carrier = np.load(carrier_path)
    frame_ids = [int(value) for value in np.asarray(carrier["frame_ids"]).tolist()]
    valid = np.asarray(carrier["valid"], dtype=bool)[:, query_index]
    visibility = np.asarray(carrier["visibility_prob"], dtype=np.float32)[:, query_index]
    confidence = np.asarray(carrier["confidence_prob"], dtype=np.float32)[:, query_index]
    accepted = valid & (visibility >= 0.5) & (confidence >= 0.5)
    mask = np.asarray([frame_id != support_frame for frame_id in frame_ids], dtype=bool)
    return int(np.count_nonzero(accepted[mask]))


def _finding_rows(findings: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in findings.items():
        if isinstance(value, list):
            for item in value:
                rows.append({"finding": key, **item})
        elif isinstance(value, dict):
            rows.append({"finding": key, "value_json": value})
        else:
            rows.append({"finding": key, "value": value})
    return rows


def _repair_assessment(findings: dict[str, Any]) -> dict[str, Any]:
    threshold_exceeds_one = bool(findings.get("raw_cq_gate_threshold_exceeds_one"))
    all_defer = bool(findings.get("all_defer_controls_with_high_utility"))
    no_temporal = findings.get("no_temporal_source_only_probe", {})
    return {
        "canonical_v63_decision_should_remain_no_go": True,
        "reason": (
            "The audit finds protocol/metric issues worth repairing, but it does not by itself produce a new canonical "
            "Phase 4 pass. A repaired scorer/control protocol must be rerun and then downstream Phase 5-8 gates must pass."
        ),
        "identified_repair_targets": {
            "raw_cq_gate_infeasible": threshold_exceeds_one,
            "no_op_controls_over_rewarded": all_defer,
            "no_temporal_control_too_permissive": bool(no_temporal.get("temporal_required_control_would_be_lower")),
        },
    }


def _counter_from_string(value: str) -> dict[str, int]:
    if not value:
        return {}
    try:
        import json

        payload = json.loads(value)
        return {str(key): int(val) for key, val in payload.items()}
    except Exception:
        return {}


def _mean(values: Any) -> float | None:
    vals = [bool(value) for value in values]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)
