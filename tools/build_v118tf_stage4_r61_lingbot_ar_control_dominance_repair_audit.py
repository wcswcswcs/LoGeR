#!/usr/bin/env python3
"""Audit the final LB-AR control-dominance repair feasibility after R59.

This script does not run geometry and does not invent new metrics. It reads the
R59 forced-action matrix and asks whether the recommended decision-theoretic
repair can promote a non-oracle, semantic-specific policy.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
R59_SUMMARY_ROOT = RESULT_ROOT / "stage4_r59_lingbot_ar_forced_action_diagnostic/summary"
OUT_ROOT = RESULT_ROOT / "stage4_r61_lingbot_ar_control_dominance_repair_audit/summary"
SUMMARY_PATH = OUT_ROOT / "stage4_r61_control_dominance_repair_summary.json"
ROW_PATH = OUT_ROOT / "stage4_r61_control_dominance_rows.csv"
REPORT_PATH = OUT_ROOT / "STAGE4_R61_CONTROL_DOMINANCE_REPAIR_REPORT.md"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
BOUNDARY_REPORT = RESULT_ROOT / "V118_METHOD_AND_NO_GO_BOUNDARIES.md"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def row_key(stage: str, seq: str, action: str) -> tuple[str, str, str]:
    return (stage, seq, action)


def add_registry_row(row: dict[str, Any]) -> None:
    if REGISTRY.exists():
        rows = read_csv(REGISTRY)
    else:
        rows = []
    fields: list[str] = []
    for old in rows:
        for key in old:
            if key not in fields:
                fields.append(key)
    for key in row:
        if key not in fields:
            fields.append(key)
    kept = [
        old
        for old in rows
        if not (
            old.get("stage") == row.get("stage")
            and old.get("surface_or_branch") == row.get("surface_or_branch")
            and old.get("artifact") == row.get("artifact")
        )
    ]
    kept.append({key: row.get(key, "") for key in fields})
    with REGISTRY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def selected_status(selected_action: str, control_dominance_pass: bool, rel_vs_default: float) -> str:
    if selected_action == "abstain":
        if control_dominance_pass:
            return "SAFETY_ABSTAIN_ONLY_NOT_GEOMETRY_PROMOTION"
        return "ABSTAIN_REJECTED_BY_CONTROL_DOMINANCE"
    if not control_dominance_pass:
        return "REJECTED_BY_CONTROL_DOMINANCE"
    if rel_vs_default < 0.03:
        return "CONTROL_DOMINANT_BUT_GEOMETRY_GATE_FAIL"
    return "LOCAL_SEMANTIC_ACTION_PASS"


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    action_rows = read_csv(R59_SUMMARY_ROOT / "stage4_r59_action_matrix_rows.csv")
    policy_rows = read_csv(R59_SUMMARY_ROOT / "stage4_r59_policy_rows.csv")
    winner_rows = read_csv(R59_SUMMARY_ROOT / "stage4_r59_sequence_winner_rows.csv")

    by_stage_seq_action: dict[tuple[str, str, str], dict[str, str]] = {}
    by_stage_seq: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in action_rows:
        key = row_key(row["stage"], row["seq"], row["action"])
        by_stage_seq_action[key] = row
        by_stage_seq.setdefault((row["stage"], row["seq"]), []).append(row)

    winner_by_seq = {row["seq"]: row for row in winner_rows}
    latest_policy_rows = [row for row in policy_rows if row["policy_rule"] == "stable_guarded_risk"]
    if not latest_policy_rows:
        raise RuntimeError("No stable_guarded_risk policy rows found in R59 policy CSV")

    audit_rows: list[dict[str, Any]] = []
    oracle_best_rels: list[float] = []
    control_or_random_winners: list[str] = []
    for prow in latest_policy_rows:
        stage = prow["stage"]
        seq = prow["seq"]
        selected_action = prow["policy_action"]
        selected = by_stage_seq_action.get(row_key(stage, seq, selected_action))
        if selected is None:
            raise RuntimeError(f"Missing selected action row for stage={stage} seq={seq} action={selected_action}")
        controls = [row for row in by_stage_seq[(stage, seq)] if row["action"] != selected_action]
        selected_ate = as_float(selected, "ate")
        selected_rel = as_float(selected, "rel_vs_default")
        control_ates = [as_float(row, "ate") for row in controls]
        best_control_ate = min(control_ates) if control_ates else float("nan")
        control_dominance_pass = all(selected_ate <= ate for ate in control_ates)
        winner = winner_by_seq[seq]
        best_action = winner["best_ate_action"]
        best_rel = float(winner["best_rel_vs_baseline"])
        oracle_best_rels.append(best_rel)
        best_role = by_stage_seq_action[row_key(winner["best_ate_stage"], seq, best_action)]["role"]
        winner_is_control_or_random = best_action == "random" or "control" in best_role
        if winner_is_control_or_random:
            control_or_random_winners.append(seq)
        status = selected_status(selected_action, control_dominance_pass, selected_rel)
        audit_rows.append(
            {
                "schema": "acl2_v118tf_stage4_r61_control_dominance_row_v1",
                "stage": stage,
                "seq": seq,
                "policy_rule": prow["policy_rule"],
                "selected_action": selected_action,
                "selection_reason": prow["selection_reason"],
                "selected_role": selected["role"],
                "selected_ate": selected_ate,
                "selected_rel_vs_default": selected_rel,
                "best_control_ate": best_control_ate,
                "selected_minus_best_control_ate": selected_ate - best_control_ate,
                "control_dominance_pass": control_dominance_pass,
                "best_ate_action": best_action,
                "best_ate_role": best_role,
                "best_rel_vs_baseline": best_rel,
                "winner_is_control_or_random": winner_is_control_or_random,
                "semantic_claim_status": status,
                "semantic_claim_pass": status == "LOCAL_SEMANTIC_ACTION_PASS" and not winner_is_control_or_random,
            }
        )

    summary_stats = {
        "latest_policy_sequence_count": len(audit_rows),
        "control_dominance_pass_count": sum(1 for row in audit_rows if row["control_dominance_pass"]),
        "semantic_claim_pass_count": sum(1 for row in audit_rows if row["semantic_claim_pass"]),
        "rejected_sequences": [row["seq"] for row in audit_rows if not row["semantic_claim_pass"]],
        "control_or_random_winner_sequences": control_or_random_winners,
        "oracle_best_action_median_rel_vs_baseline": median(oracle_best_rels),
    }
    return audit_rows, summary_stats


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ACL2 v118 Stage4-R61 LB-AR Control-Dominance Repair Audit",
        "",
        f"- decision: `{summary['stage4_r61_decision']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- semantic_claim_pass_count: `{summary['semantic_claim_pass_count']}`",
        f"- control_or_random_winner_sequences: `{','.join(summary['control_or_random_winner_sequences'])}`",
        "",
        "## Rows",
        "",
        "| seq | selected | selected rel | control dominance | winner | winner role | status |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seq']} | `{row['selected_action']}` | {row['selected_rel_vs_default']} | "
            f"{row['control_dominance_pass']} | `{row['best_ate_action']}` | `{row['best_ate_role']}` | "
            f"`{row['semantic_claim_status']}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The control-dominance repair is useful as a rejection gate, but it does not rescue LB-AR. "
        "It would reject the R58 selected actions on 08/09 and cannot promote random/control winners "
        "as semantic-specific carrier causality.",
        "",
        "The oracle best-action median relation is reported only as an action-surface upper bound from "
        "already observed controls. It is not a valid deployable policy because it uses per-sequence "
        "outcomes and includes control/random winners.",
        "",
        "## Outputs",
        "",
        f"- summary: `{summary['outputs']['summary']}`",
        f"- rows: `{summary['outputs']['rows']}`",
        f"- report: `{summary['outputs']['report']}`",
    ]
    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def update_boundary_report(summary: dict[str, Any]) -> None:
    if not BOUNDARY_REPORT.exists():
        return
    text = BOUNDARY_REPORT.read_text(encoding="utf-8")
    marker = "\n## R61 Control-Dominance Repair Feasibility\n"
    base = text.split(marker)[0].rstrip()
    addendum = [
        "",
        "## R61 Control-Dominance Repair Feasibility",
        "",
        f"- decision: `{summary['stage4_r61_decision']}`",
        "- R61 attempted the R59-recommended decision-theoretic repair audit: require the selected action to beat same-schedule controls before any semantic claim.",
        "- Result: the gate rejects R58 seq08/seq09; seq08 is won by random and seq09 is won by a control/risk action, so the repair cannot establish semantic-specific causality.",
        "- No new geometry was run and no metrics were backfilled; this is a feasibility/no-claim audit over the completed R59 matrix.",
        f"- artifact: `{summary['outputs']['summary']}`",
    ]
    BOUNDARY_REPORT.write_text(base + marker + "\n".join(addendum[2:]).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    r59_summary = read_json(R59_SUMMARY_ROOT / "stage4_r59_forced_action_diagnostic_summary.json")
    rows, stats = build_rows()
    all_rows_rejected = stats["semantic_claim_pass_count"] == 0
    summary = {
        "schema": "acl2_v118tf_stage4_r61_control_dominance_repair_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage4_r61_decision": "NO_GO_LB_AR_R61_CONTROL_DOMINANCE_REPAIR_NOT_PROMOTABLE",
        "global_goal_achieved": False,
        "source_decision": r59_summary.get("stage4_r59_decision", ""),
        "repair_attempt": "control_dominance_semantic_claim_gate",
        "fresh_extension_complete": bool(r59_summary.get("fresh_extension_complete")),
        "all_rows_rejected_for_semantic_claim": all_rows_rejected,
        "latest_policy_sequence_count": stats["latest_policy_sequence_count"],
        "control_dominance_pass_count": stats["control_dominance_pass_count"],
        "semantic_claim_pass_count": stats["semantic_claim_pass_count"],
        "rejected_sequences": stats["rejected_sequences"],
        "control_or_random_winner_sequences": stats["control_or_random_winner_sequences"],
        "oracle_best_action_median_rel_vs_baseline": stats["oracle_best_action_median_rel_vs_baseline"],
        "oracle_boundary": (
            "The oracle best-action upper bound is not a deployable or semantic-specific policy; "
            "it uses observed per-sequence outcomes and includes control/random winners."
        ),
        "interpretation": (
            "The R59-recommended control-dominance gate can prevent false promotion, but it does not "
            "repair LB-AR into a semantic-specific geometry improvement. R58 remains the decisive "
            "fresh blocker."
        ),
        "outputs": {
            "summary": rel(SUMMARY_PATH),
            "rows": rel(ROW_PATH),
            "report": rel(REPORT_PATH),
            "boundary_report": rel(BOUNDARY_REPORT),
        },
    }
    write_csv(ROW_PATH, rows)
    write_json(SUMMARY_PATH, summary)
    write_report(summary, rows)
    update_boundary_report(summary)
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R61-LB-AR-ControlDominanceRepair",
            "surface_or_branch": "LB-AR",
            "status": summary["stage4_r61_decision"],
            "artifact": rel(SUMMARY_PATH),
            "notes": "Decision-theoretic repair feasibility audit over R59 action matrix; no new geometry; global goal remains false.",
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
