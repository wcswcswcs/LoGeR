#!/usr/bin/env python3
"""Aggregate ACL2 v118 LingBot AR fresh forced-action diagnostics.

R59 is not a new geometry run. It consolidates R51/R52/R55/R58 evidence to
decide whether the current stable-guarded action rule has a semantic-specific
claim or is blocked by generic action sensitivity / control wins.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
OUT = RESULT_ROOT / "stage4_r59_lingbot_ar_forced_action_diagnostic"
SUMMARY_DIR = OUT / "summary"

STAGES = [
    {
        "stage": "R51",
        "slug": "stage4_r51_lingbot_ar_fresh_r47_policy_validation",
        "summary_key": "stage4_r51_decision",
        "policy_rule": "r47",
        "validation_boundary": "fresh_03_04",
    },
    {
        "stage": "R52",
        "slug": "stage4_r52_lingbot_ar_fresh_r47_policy_validation_06_07",
        "summary_key": "stage4_r52_decision",
        "policy_rule": "r47",
        "validation_boundary": "fresh_06_07",
    },
    {
        "stage": "R55",
        "slug": "stage4_r55_lingbot_ar_fresh_guarded_risk_policy_validation_06_07_gpu",
        "summary_key": "stage4_r55_decision",
        "policy_rule": "stable_guarded_risk",
        "validation_boundary": "fresh_06_07",
    },
    {
        "stage": "R58",
        "slug": "stage4_r58_lingbot_ar_fresh_guarded_risk_policy_validation_08_09",
        "summary_key": "stage4_r58_decision",
        "policy_rule": "stable_guarded_risk",
        "validation_boundary": "fresh_08_09",
    },
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def action_from_row(row: dict[str, str]) -> str:
    role = row.get("role", "")
    action = row.get("policy_action", "")
    if role in {"token_random_control", "forced_random_control"}:
        return "random"
    if action:
        return action
    if "risk" in role:
        return "risk"
    if "reverse" in role:
        return "reverse"
    return role or "unknown"


def stage_paths(slug: str) -> dict[str, Path]:
    base = RESULT_ROOT / slug / "summary"
    return {
        "summary": base / "stage4_r51_fresh_policy_summary.json",
        "runtime_rows": base / "stage4_r51_fresh_policy_runtime_rows.csv",
        "comparison_rows": base / "stage4_r51_fresh_policy_comparison_rows.csv",
        "policy_rows": base / "stage4_r51_fresh_policy_rows.csv",
    }


def build_stage_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    policy_rows_out: list[dict[str, Any]] = []
    seen_baseline: set[tuple[str, str]] = set()

    for spec in STAGES:
        paths = stage_paths(spec["slug"])
        summary = read_json(paths["summary"])
        runtime_rows = read_csv(paths["runtime_rows"])
        policy_rows = read_csv(paths["policy_rows"])
        decision = summary.get(spec["summary_key"], "")
        stage_rows.append(
            {
                "schema": "acl2_v118tf_stage4_r59_stage_summary_row_v1",
                "stage": spec["stage"],
                "slug": spec["slug"],
                "policy_rule": spec["policy_rule"],
                "validation_boundary": spec["validation_boundary"],
                "summary_exists": bool(summary),
                "runtime_rows": len(runtime_rows),
                "decision": decision,
                "complete": summary.get("complete", ""),
                "action_fidelity": summary.get("action_fidelity", ""),
                "policy_gate": summary.get("policy_gate", ""),
                "baseline_gate": summary.get("baseline_gate", ""),
                "candidate_better_all_controls": summary.get("candidate_better_all_controls", ""),
                "all_sequences_nonharm": summary.get("all_sequences_nonharm", ""),
                "median_rel_vs_default": summary.get("median_rel_vs_default", ""),
                "max_harm": summary.get("max_harm", ""),
                "summary_path": rel(paths["summary"]),
            }
        )
        for prow in policy_rows:
            policy_rows_out.append(
                {
                    "schema": "acl2_v118tf_stage4_r59_policy_row_v1",
                    "stage": spec["stage"],
                    "policy_rule": spec["policy_rule"],
                    "seq": str(prow.get("seq", "")).zfill(2),
                    "policy_action": prow.get("policy_action", ""),
                    "selection_reason": prow.get("selection_reason", ""),
                    "internal_semantic_corr": prow.get("internal_semantic_corr", ""),
                    "stable_to_weak_lowtrust": prow.get("stable_to_weak_lowtrust", ""),
                    "dynamic_plus_lowtrust_mean": prow.get("dynamic_plus_lowtrust_mean", ""),
                }
            )
        for row in runtime_rows:
            seq = str(row.get("seq", "")).zfill(2)
            if not seq:
                continue
            baseline_key = (spec["stage"], seq)
            if baseline_key not in seen_baseline:
                baseline_ate = fnum(row.get("baseline_ate"))
                if baseline_ate is not None:
                    action_rows.append(
                        {
                            "schema": "acl2_v118tf_stage4_r59_action_matrix_row_v1",
                            "stage": spec["stage"],
                            "seq": seq,
                            "role": "baseline_selected_by_abstention",
                            "action": "abstain",
                            "method": row.get("baseline_method", ""),
                            "ate": baseline_ate,
                            "rpe_rot": row.get("baseline_rpe_rot", ""),
                            "rpe_trans": row.get("baseline_rpe_trans", ""),
                            "rel_vs_default": 0.0,
                            "action_fidelity_pass": True,
                        }
                    )
                    seen_baseline.add(baseline_key)
            action_rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r59_action_matrix_row_v1",
                    "stage": spec["stage"],
                    "seq": seq,
                    "role": row.get("role", ""),
                    "action": action_from_row(row),
                    "method": row.get("method", ""),
                    "ate": row.get("ate", ""),
                    "rpe_rot": row.get("rpe_rot", ""),
                    "rpe_trans": row.get("rpe_trans", ""),
                    "rel_vs_default": row.get("ate_rel_improvement_vs_default", ""),
                    "action_fidelity_pass": row.get("action_fidelity_pass", ""),
                }
            )
    return stage_rows, policy_rows_out, action_rows


def build_winner_rows(action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in action_rows:
        seq = str(row["seq"]).zfill(2)
        grouped.setdefault(seq, []).append(row)

    out: list[dict[str, Any]] = []
    for seq, rows in sorted(grouped.items()):
        # For duplicate stages/actions, keep the best ATE observed but preserve source.
        best_by_action: dict[str, dict[str, Any]] = {}
        for row in rows:
            ate = fnum(row.get("ate"))
            if ate is None:
                continue
            action = str(row.get("action", ""))
            current = best_by_action.get(action)
            if current is None or ate < float(current["ate"]):
                best_by_action[action] = row | {"ate": ate}
        if not best_by_action:
            continue
        best_ate_action, best_ate_row = min(best_by_action.items(), key=lambda item: float(item[1]["ate"]))
        rpe_candidates = [(action, row) for action, row in best_by_action.items() if fnum(row.get("rpe_trans")) is not None]
        best_rpe_action, best_rpe_row = min(rpe_candidates, key=lambda item: float(item[1]["rpe_trans"])) if rpe_candidates else ("", {})
        baseline = best_by_action.get("abstain", {})
        baseline_ate = fnum(baseline.get("ate"))
        best_ate = fnum(best_ate_row.get("ate"))
        random_ate = fnum(best_by_action.get("random", {}).get("ate"))
        risk_ate = fnum(best_by_action.get("risk", {}).get("ate"))
        reverse_ate = fnum(best_by_action.get("reverse", {}).get("ate"))
        out.append(
            {
                "schema": "acl2_v118tf_stage4_r59_seq_winner_row_v1",
                "seq": seq,
                "best_ate_action": best_ate_action,
                "best_ate": best_ate,
                "best_ate_stage": best_ate_row.get("stage", ""),
                "best_rpe_trans_action": best_rpe_action,
                "best_rpe_trans": best_rpe_row.get("rpe_trans", ""),
                "best_rpe_trans_stage": best_rpe_row.get("stage", ""),
                "baseline_ate": baseline_ate,
                "random_ate": random_ate,
                "risk_ate": risk_ate,
                "reverse_ate": reverse_ate,
                "best_rel_vs_baseline": (baseline_ate - best_ate) / baseline_ate if baseline_ate not in (None, 0.0) and best_ate is not None else "",
                "random_beats_baseline": random_ate is not None and baseline_ate is not None and random_ate < baseline_ate,
                "risk_beats_baseline": risk_ate is not None and baseline_ate is not None and risk_ate < baseline_ate,
                "reverse_beats_baseline": reverse_ate is not None and baseline_ate is not None and reverse_ate < baseline_ate,
                "semantic_specificity_warning": best_ate_action == "random" or (
                    random_ate is not None and best_ate is not None and random_ate <= best_ate + 1e-12
                ),
            }
        )
    return out


def latest_stable_policy_actions(policy_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the latest stable_guarded_risk selection for each sequence."""
    order = {spec["stage"]: idx for idx, spec in enumerate(STAGES)}
    selected: dict[str, dict[str, Any]] = {}
    for row in policy_rows:
        if row.get("policy_rule") != "stable_guarded_risk":
            continue
        seq = str(row.get("seq", "")).zfill(2)
        if not seq:
            continue
        current = selected.get(seq)
        if current is None or order.get(str(row.get("stage")), -1) >= order.get(str(current.get("stage")), -1):
            selected[seq] = row
    return selected


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    stage_rows, policy_rows, action_rows = build_stage_rows()
    winner_rows = build_winner_rows(action_rows)
    stable_policy_by_seq = latest_stable_policy_actions(policy_rows)

    complete_stages = [row for row in stage_rows if row["complete"] is True or str(row["complete"]).lower() == "true"]
    fresh_extension_complete = len(complete_stages) == len(STAGES)
    pass_stages = [row["stage"] for row in stage_rows if str(row.get("policy_gate", "")).lower() == "true"]
    no_go_stages = [row["stage"] for row in stage_rows if row.get("decision") and str(row.get("policy_gate", "")).lower() != "true"]
    semantic_warnings = [row for row in winner_rows if row["semantic_specificity_warning"]]
    stable_rule_no_go = any(row["stage"] == "R58" and str(row.get("policy_gate", "")).lower() != "true" for row in stage_rows)
    winner_actions = {row["seq"]: row["best_ate_action"] for row in winner_rows}
    selected_mismatch_rows: list[dict[str, Any]] = []
    for row in winner_rows:
        seq = row["seq"]
        selected = stable_policy_by_seq.get(seq)
        if not selected:
            continue
        selected_action = selected.get("policy_action", "")
        if selected_action != row["best_ate_action"]:
            selected_mismatch_rows.append(
                {
                    "seq": seq,
                    "selected_stage": selected.get("stage", ""),
                    "selected_action": selected_action,
                    "best_ate_action": row["best_ate_action"],
                    "best_ate": row["best_ate"],
                    "selection_reason": selected.get("selection_reason", ""),
                }
            )
    selected_rels = [fnum(row.get("median_rel_vs_default")) for row in stage_rows if fnum(row.get("median_rel_vs_default")) is not None]

    decision = (
        "NO_GO_LB_AR_STABLE_GUARDED_RULE_BLOCKED_BY_FRESH_08_09_ACTION_SURFACE"
        if stable_rule_no_go and semantic_warnings
        else "PARTIAL_DIAGNOSTIC_INCOMPLETE"
    )
    summary = {
        "schema": "acl2_v118tf_stage4_r59_lingbot_ar_forced_action_diagnostic_summary_v1",
        "stage4_r59_decision": decision,
        "global_goal_achieved": False,
        "fresh_extension_complete": fresh_extension_complete,
        "pass_stages": pass_stages,
        "no_go_stages": no_go_stages,
        "sequence_winner_actions": winner_actions,
        "stable_guarded_selected_action_mismatch_count": len(selected_mismatch_rows),
        "stable_guarded_selected_action_mismatch_rows": selected_mismatch_rows,
        "sequence_count": len(winner_rows),
        "semantic_specificity_warning_count": len(semantic_warnings) + len(selected_mismatch_rows),
        "semantic_specificity_warning_sequences": sorted({row["seq"] for row in semantic_warnings} | {row["seq"] for row in selected_mismatch_rows}),
        "median_of_stage_median_rels": median(selected_rels) if selected_rels else "",
        "interpretation": (
            "Fresh forced-action evidence shows the current sign/ratio polarity rule is not a general semantic carrier. "
            "R58 has complete runtime/eval/action-fidelity but fails controls: random/risk can beat selected semantic actions."
        ),
        "recommended_next": (
            "Do not promote LB-AR stable_guarded_risk. Future repair must first separate generic action sensitivity "
            "from semantic-specific carrier causality; a new rule needs a fresh holdout after being specified from diagnostics."
        ),
        "outputs": {
            "stage_rows": rel(SUMMARY_DIR / "stage4_r59_stage_rows.csv"),
            "policy_rows": rel(SUMMARY_DIR / "stage4_r59_policy_rows.csv"),
            "action_matrix": rel(SUMMARY_DIR / "stage4_r59_action_matrix_rows.csv"),
            "winner_rows": rel(SUMMARY_DIR / "stage4_r59_sequence_winner_rows.csv"),
            "summary": rel(SUMMARY_DIR / "stage4_r59_forced_action_diagnostic_summary.json"),
            "report": rel(SUMMARY_DIR / "STAGE4_R59_FORCED_ACTION_DIAGNOSTIC_REPORT.md"),
        },
    }

    write_csv(SUMMARY_DIR / "stage4_r59_stage_rows.csv", stage_rows)
    write_csv(SUMMARY_DIR / "stage4_r59_policy_rows.csv", policy_rows)
    write_csv(SUMMARY_DIR / "stage4_r59_action_matrix_rows.csv", action_rows)
    write_csv(SUMMARY_DIR / "stage4_r59_sequence_winner_rows.csv", winner_rows)
    write_json(SUMMARY_DIR / "stage4_r59_forced_action_diagnostic_summary.json", summary)

    lines = [
        "# ACL2 v118 Stage4-R59 LingBot AR Forced-Action Diagnostic",
        "",
        f"- decision: `{summary['stage4_r59_decision']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- semantic_specificity_warning_sequences: `{','.join(summary['semantic_specificity_warning_sequences'])}`",
        f"- stable_guarded_selected_action_mismatch_count: `{summary['stable_guarded_selected_action_mismatch_count']}`",
        "",
        "| seq | best ATE action | best ATE | best RPE-trans action | best RPE trans | random warning |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in winner_rows:
        lines.append(
            f"| {row['seq']} | {row['best_ate_action']} | {row['best_ate']} | "
            f"{row['best_rpe_trans_action']} | {row['best_rpe_trans']} | {row['semantic_specificity_warning']} |"
        )
    lines += ["", "## Interpretation", "", summary["interpretation"], "", "## Recommended Next", "", summary["recommended_next"]]
    (SUMMARY_DIR / "STAGE4_R59_FORCED_ACTION_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
