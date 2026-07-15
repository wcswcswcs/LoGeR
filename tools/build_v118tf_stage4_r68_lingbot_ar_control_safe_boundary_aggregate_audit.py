#!/usr/bin/env python3
"""Aggregate the R68 LB-AR control-safe boundary repair evidence.

R68 is not a geometry runner. It combines the completed R59 fresh action matrix
with the R67 seq10 control-safe-boundary validation, then separates actionable
semantic passes from safety abstains and generic-action danger zones.
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
R59_SUMMARY = RESULT_ROOT / "stage4_r59_lingbot_ar_forced_action_diagnostic/summary"
R67_SUMMARY = RESULT_ROOT / "stage4_r67_lingbot_ar_fresh_control_safe_boundary_policy_validation_10/summary"
OUT = RESULT_ROOT / "stage4_r68_lingbot_ar_control_safe_boundary_aggregate_audit/summary"
ROWS_PATH = OUT / "stage4_r68_control_safe_boundary_rows.csv"
SUMMARY_PATH = OUT / "stage4_r68_control_safe_boundary_summary.json"
REPORT_PATH = OUT / "STAGE4_R68_CONTROL_SAFE_BOUNDARY_AGGREGATE_REPORT.md"

NEGATIVE_RATIO_LOW = 0.05
NEGATIVE_RATIO_HIGH = 0.20
RISK_MIN_CORR = 0.75
RISK_MIN_RATIO = 0.20
RISK_MIN_DYNAMIC = 0.20
MIN_REL = 0.03


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
        return {str(key): clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_json(val) for val in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def choose_action(corr: float, ratio: float, dynamic: float) -> tuple[str, str]:
    if corr <= 0.0:
        if ratio < NEGATIVE_RATIO_LOW:
            return "reverse", "negative_corr_low_ratio_reverse_control_safe"
        if ratio >= NEGATIVE_RATIO_HIGH:
            return "reverse", "negative_corr_high_ratio_reverse_control_probe"
        return "abstain", "negative_corr_mid_ratio_control_danger_abstain"
    if corr >= RISK_MIN_CORR and ratio >= RISK_MIN_RATIO and dynamic >= RISK_MIN_DYNAMIC:
        return "risk", "positive_corr_high_ratio_dynamic_risk_control_safe"
    return "abstain", "positive_corr_or_ratio_not_control_safe_abstain"


def load_features() -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for row in read_csv(R59_SUMMARY / "stage4_r59_policy_rows.csv"):
        seq = str(row["seq"]).zfill(2)
        if seq not in features:
            features[seq] = {
                "seq": seq,
                "feature_source": rel(R59_SUMMARY / "stage4_r59_policy_rows.csv"),
                "internal_semantic_corr": fnum(row["internal_semantic_corr"]),
                "stable_to_weak_lowtrust": fnum(row["stable_to_weak_lowtrust"]),
                "dynamic_plus_lowtrust_mean": fnum(row["dynamic_plus_lowtrust_mean"]),
            }
    for row in read_csv(R67_SUMMARY / "stage4_r51_fresh_policy_rows.csv"):
        seq = str(row["seq"]).zfill(2)
        features[seq] = {
            "seq": seq,
            "feature_source": rel(R67_SUMMARY / "stage4_r51_fresh_policy_rows.csv"),
            "internal_semantic_corr": fnum(row["internal_semantic_corr"]),
            "stable_to_weak_lowtrust": fnum(row["stable_to_weak_lowtrust"]),
            "dynamic_plus_lowtrust_mean": fnum(row["dynamic_plus_lowtrust_mean"]),
        }
    return features


def load_action_ates() -> dict[str, dict[str, dict[str, Any]]]:
    actions: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_csv(R59_SUMMARY / "stage4_r59_sequence_winner_rows.csv"):
        seq = str(row["seq"]).zfill(2)
        actions[seq] = {
            "abstain": {
                "ate": fnum(row["baseline_ate"]),
                "source": rel(R59_SUMMARY / "stage4_r59_sequence_winner_rows.csv"),
                "role": "baseline_selected_by_abstention",
            },
            "risk": {
                "ate": fnum(row["risk_ate"]),
                "source": rel(R59_SUMMARY / "stage4_r59_sequence_winner_rows.csv"),
                "role": "risk_or_control",
            },
            "reverse": {
                "ate": fnum(row["reverse_ate"]),
                "source": rel(R59_SUMMARY / "stage4_r59_sequence_winner_rows.csv"),
                "role": "reverse_or_control",
            },
            "random": {
                "ate": fnum(row["random_ate"]),
                "source": rel(R59_SUMMARY / "stage4_r59_sequence_winner_rows.csv"),
                "role": "random_control",
            },
        }
    runtime_rows = read_csv(R67_SUMMARY / "stage4_r51_fresh_policy_runtime_rows.csv")
    for row in runtime_rows:
        seq = str(row["seq"]).zfill(2)
        bucket = actions.setdefault(seq, {})
        if "abstain" not in bucket:
            bucket["abstain"] = {
                "ate": fnum(row["baseline_ate"]),
                "source": rel(R67_SUMMARY / "stage4_r51_fresh_policy_runtime_rows.csv"),
                "role": "baseline_selected_by_abstention",
            }
        action = str(row.get("policy_action", ""))
        if action:
            bucket[action] = {
                "ate": fnum(row["ate"]),
                "source": rel(R67_SUMMARY / "stage4_r51_fresh_policy_runtime_rows.csv"),
                "role": row.get("role", ""),
            }
    return actions


def rel_vs_baseline(baseline_ate: float | None, action_ate: float | None) -> float | None:
    if baseline_ate in (None, 0.0) or action_ate is None:
        return None
    return (baseline_ate - action_ate) / baseline_ate


def row_classification(selected_action: str, selected_rel: float | None, selected_better_controls: bool, best_action: str) -> str:
    if selected_action == "abstain":
        if best_action == "abstain":
            return "SAFE_ABSTAIN_BASELINE_BEST"
        return "ABSTAIN_GENERIC_ACTION_DANGER"
    if not selected_better_controls:
        return "ACTION_REJECTED_BY_CONTROL_DOMINANCE"
    if selected_rel is None or selected_rel < MIN_REL:
        return "ACTION_REJECTED_BY_GEOMETRY_GATE"
    return "ACTIONABLE_SEMANTIC_CONTROL_PASS"


def build_rows() -> list[dict[str, Any]]:
    features = load_features()
    actions = load_action_ates()
    rows: list[dict[str, Any]] = []
    for seq in sorted(features):
        feat = features[seq]
        corr = feat["internal_semantic_corr"]
        ratio = feat["stable_to_weak_lowtrust"]
        dynamic = feat["dynamic_plus_lowtrust_mean"]
        if corr is None or ratio is None or dynamic is None:
            raise RuntimeError(f"missing features for seq {seq}")
        selected_action, reason = choose_action(corr, ratio, dynamic)
        action_map = actions.get(seq, {})
        baseline_ate = action_map.get("abstain", {}).get("ate")
        selected_ate = action_map.get(selected_action, {}).get("ate") if selected_action != "abstain" else baseline_ate
        selected_rel = rel_vs_baseline(baseline_ate, selected_ate)
        valid_actions = {
            action: payload
            for action, payload in action_map.items()
            if action != "abstain" and payload.get("ate") is not None
        }
        if selected_action == "abstain":
            controls = valid_actions
        else:
            controls = {action: payload for action, payload in valid_actions.items() if action != selected_action}
        control_ates = [float(payload["ate"]) for payload in controls.values()]
        best_control_action = ""
        best_control_ate = None
        if controls:
            best_control_action, best_control_payload = min(controls.items(), key=lambda item: float(item[1]["ate"]))
            best_control_ate = float(best_control_payload["ate"])
        selected_better_controls = bool(control_ates) and selected_ate is not None and all(float(selected_ate) < ate for ate in control_ates)
        if selected_action == "abstain":
            selected_better_controls = bool(control_ates) and baseline_ate is not None and all(float(baseline_ate) <= ate for ate in control_ates)
        best_action, best_payload = min(
            ((action, payload) for action, payload in action_map.items() if payload.get("ate") is not None),
            key=lambda item: float(item[1]["ate"]),
        )
        best_ate = float(best_payload["ate"])
        best_rel = rel_vs_baseline(baseline_ate, best_ate)
        classification = row_classification(selected_action, selected_rel, selected_better_controls, best_action)
        rows.append(
            {
                "schema": "acl2_v118tf_stage4_r68_control_safe_boundary_row_v1",
                "seq": seq,
                "internal_semantic_corr": corr,
                "stable_to_weak_lowtrust": ratio,
                "dynamic_plus_lowtrust_mean": dynamic,
                "selected_action": selected_action,
                "selection_reason": reason,
                "baseline_ate": baseline_ate,
                "selected_ate": selected_ate,
                "selected_rel_vs_baseline": selected_rel,
                "available_actions": ";".join(sorted(valid_actions)),
                "best_control_action": best_control_action,
                "best_control_ate": best_control_ate,
                "selected_minus_best_control_ate": (
                    float(selected_ate) - best_control_ate
                    if selected_ate is not None and best_control_ate is not None
                    else None
                ),
                "selected_better_controls": selected_better_controls,
                "best_action": best_action,
                "best_ate": best_ate,
                "best_rel_vs_baseline": best_rel,
                "winner_is_generic_or_unselected_control": best_action in {"random"} or best_action != selected_action,
                "classification": classification,
                "feature_source": feat["feature_source"],
                "selected_metric_source": action_map.get(selected_action, action_map.get("abstain", {})).get("source", ""),
            }
        )
    return rows


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ACL2 v118 Stage4-R68 LB-AR Control-Safe Boundary Aggregate Audit",
        "",
        f"- decision: `{summary['stage4_r68_decision']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- fresh_holdout_after_rule: `{summary['fresh_holdout_after_rule']}`",
        f"- actionable_pass_sequences: `{','.join(summary['actionable_pass_sequences'])}`",
        f"- safe_abstain_sequences: `{','.join(summary['safe_abstain_sequences'])}`",
        f"- generic_danger_abstain_sequences: `{','.join(summary['generic_danger_abstain_sequences'])}`",
        "",
        "## Rule",
        "",
        "```text",
        *summary["rule"]["logic"],
        "```",
        "",
        "## Rows",
        "",
        "| seq | selected | rel | best control | best action | classification |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seq']} | `{row['selected_action']}` | {row['selected_rel_vs_baseline']} | "
            f"`{row['best_control_action']}` {row['best_control_ate']} | `{row['best_action']}` {row['best_ate']} | "
            f"`{row['classification']}` |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        summary["boundary"],
        "",
        "## Outputs",
        "",
        f"- summary: `{summary['outputs']['summary']}`",
        f"- rows: `{summary['outputs']['rows']}`",
        f"- report: `{summary['outputs']['report']}`",
    ]
    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    rows = build_rows()
    actionable = [row for row in rows if row["selected_action"] != "abstain"]
    actionable_pass = [row for row in rows if row["classification"] == "ACTIONABLE_SEMANTIC_CONTROL_PASS"]
    safe_abstain = [row for row in rows if row["classification"] == "SAFE_ABSTAIN_BASELINE_BEST"]
    danger_abstain = [row for row in rows if row["classification"] == "ABSTAIN_GENERIC_ACTION_DANGER"]
    rejected_action = [row for row in rows if row["classification"].startswith("ACTION_REJECTED")]
    actionable_rels = [float(row["selected_rel_vs_baseline"]) for row in actionable_pass if row["selected_rel_vs_baseline"] is not None]
    selected_rels = [
        float(row["selected_rel_vs_baseline"])
        for row in rows
        if row["selected_rel_vs_baseline"] is not None
    ]
    actionable_all_pass = bool(actionable) and len(actionable) == len(actionable_pass)
    decision = (
        "PARTIAL_REPAIR_ACTIONABLE_SUBSET_PASS_GENERIC_DANGER_REMAINS_GLOBAL_FALSE"
        if actionable_all_pass and danger_abstain and not rejected_action
        else "CONTROL_SAFE_BOUNDARY_REPAIR_NOT_PROMOTABLE"
    )
    summary = {
        "schema": "acl2_v118tf_stage4_r68_control_safe_boundary_summary_v1",
        "stage4_r68_decision": decision,
        "global_goal_achieved": False,
        "fresh_holdout_after_rule": False,
        "claim_level": "exploratory_repair_hypothesis_with_first_class_seq10_runtime_not_global_success",
        "sequence_count": len(rows),
        "actionable_sequence_count": len(actionable),
        "actionable_pass_count": len(actionable_pass),
        "safe_abstain_count": len(safe_abstain),
        "generic_danger_abstain_count": len(danger_abstain),
        "rejected_action_count": len(rejected_action),
        "actionable_pass_sequences": [row["seq"] for row in actionable_pass],
        "safe_abstain_sequences": [row["seq"] for row in safe_abstain],
        "generic_danger_abstain_sequences": [row["seq"] for row in danger_abstain],
        "rejected_action_sequences": [row["seq"] for row in rejected_action],
        "actionable_median_rel_vs_baseline": median(actionable_rels) if actionable_rels else "",
        "selected_all_sequence_median_rel_vs_baseline": median(selected_rels) if selected_rels else "",
        "rule": {
            "name": "control_safe_boundary",
            "logic": [
                f"if corr <= 0 and stable_to_weak_lowtrust < {NEGATIVE_RATIO_LOW:.2f}: reverse",
                f"elif corr <= 0 and stable_to_weak_lowtrust >= {NEGATIVE_RATIO_HIGH:.2f}: reverse",
                "elif corr <= 0: abstain",
                (
                    f"elif corr >= {RISK_MIN_CORR:.2f} "
                    f"and stable_to_weak_lowtrust >= {RISK_MIN_RATIO:.2f} "
                    f"and dynamic_plus_lowtrust_mean >= {RISK_MIN_DYNAMIC:.2f}: risk"
                ),
                "else: abstain",
            ],
        },
        "outputs": {
            "rows": rel(ROWS_PATH),
            "summary": rel(SUMMARY_PATH),
            "report": rel(REPORT_PATH),
        },
        "boundary": (
            "R68 is a repair audit, not a final v118 success. It shows the control-safe boundary rule "
            "can isolate an actionable fresh subset whose selected actions beat baseline and matched controls "
            "on 03/04/07/10, while 06 is a safe abstain and 08/09 remain generic-action danger zones. "
            "Because the high-ratio negative-correlation branch was introduced after inspecting R65/R66, "
            "fresh_holdout_after_rule=false; the rule must be frozen and validated on a new split before any "
            "promoted semantic-carrier claim."
        ),
    }
    write_csv(ROWS_PATH, rows)
    write_json(SUMMARY_PATH, summary)
    write_report(summary, rows)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
