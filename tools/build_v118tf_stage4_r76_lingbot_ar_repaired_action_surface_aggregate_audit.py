#!/usr/bin/env python3
"""Aggregate the repaired LB-AR fresh action surface after R70/R75."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
R68_ROWS = (
    RESULT_ROOT
    / "stage4_r68_lingbot_ar_control_safe_boundary_aggregate_audit/summary/stage4_r68_control_safe_boundary_rows.csv"
)
R70_RUNTIME = (
    RESULT_ROOT
    / "stage4_r70_lingbot_ar_fresh_control_safe_boundary_v2_policy_validation_08_09/summary/stage4_r51_fresh_policy_runtime_rows.csv"
)
R75_SUMMARY = (
    RESULT_ROOT
    / "stage4_r75_lingbot_ar_fresh_exclude67_matched_controls_08/summary/stage4_r75_exclude67_control_summary.json"
)
OUT = RESULT_ROOT / "stage4_r76_lingbot_ar_repaired_action_surface_aggregate_audit/summary"
ROWS_PATH = OUT / "stage4_r76_repaired_action_surface_rows.csv"
SUMMARY_PATH = OUT / "stage4_r76_repaired_action_surface_summary.json"
REPORT_PATH = OUT / "STAGE4_R76_REPAIRED_ACTION_SURFACE_AGGREGATE_REPORT.md"
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


def rel_vs_baseline(baseline_ate: float | None, selected_ate: float | None) -> float | None:
    if baseline_ate in (None, 0.0) or selected_ate is None:
        return None
    return (baseline_ate - selected_ate) / baseline_ate


def row_classification(selected_action: str, selected_rel: float | None, selected_better_controls: bool) -> str:
    if selected_action == "abstain":
        return "SAFE_ABSTAIN_BASELINE_BEST" if selected_better_controls else "ABSTAIN_GENERIC_ACTION_DANGER"
    if not selected_better_controls:
        return "ACTION_REJECTED_BY_CONTROL_DOMINANCE"
    if selected_rel is None or selected_rel < MIN_REL:
        return "ACTION_REJECTED_BY_GEOMETRY_GATE"
    return "ACTIONABLE_SEMANTIC_CONTROL_PASS"


def base_rows_from_r68() -> list[dict[str, Any]]:
    keep = {"03", "04", "06", "07", "10"}
    rows: list[dict[str, Any]] = []
    for row in read_csv(R68_ROWS):
        seq = str(row["seq"]).zfill(2)
        if seq not in keep:
            continue
        rows.append(
            {
                "schema": "acl2_v118tf_stage4_r76_repaired_action_surface_row_v1",
                "seq": seq,
                "selected_action": row["selected_action"],
                "selection_reason": row["selection_reason"],
                "baseline_ate": fnum(row["baseline_ate"]),
                "selected_ate": fnum(row["selected_ate"]),
                "selected_rel_vs_baseline": fnum(row["selected_rel_vs_baseline"]),
                "available_controls": row["available_actions"],
                "best_control_action": row["best_control_action"],
                "best_control_ate": fnum(row["best_control_ate"]),
                "selected_minus_best_control_ate": fnum(row["selected_minus_best_control_ate"]),
                "selected_better_controls": row["selected_better_controls"] == "True",
                "classification": row["classification"],
                "metric_source": row["selected_metric_source"],
                "repair_source": "R68_retained_local_pass_or_safe_abstain",
            }
        )
    return rows


def seq09_from_r70() -> dict[str, Any]:
    runtime_rows = read_csv(R70_RUNTIME)
    seq_rows = [row for row in runtime_rows if str(row["seq"]).zfill(2) == "09"]
    candidate = next(row for row in seq_rows if row["role"] == "candidate")
    controls = [row for row in seq_rows if row["role"] != "candidate"]
    baseline_ate = fnum(candidate["baseline_ate"])
    selected_ate = fnum(candidate["ate"])
    selected_rel = rel_vs_baseline(baseline_ate, selected_ate)
    control_ates = [(row["policy_action"], fnum(row["ate"])) for row in controls]
    complete = selected_ate is not None and all(ate is not None for _, ate in control_ates)
    best_control_action, best_control_ate = min(control_ates, key=lambda item: float(item[1] if item[1] is not None else "inf"))
    selected_better_controls = complete and all(float(selected_ate) < float(ate) for _, ate in control_ates if ate is not None)
    return {
        "schema": "acl2_v118tf_stage4_r76_repaired_action_surface_row_v1",
        "seq": "09",
        "selected_action": "risk",
        "selection_reason": "R70_fresh_seq09_risk_candidate_beats_baseline_and_controls",
        "baseline_ate": baseline_ate,
        "selected_ate": selected_ate,
        "selected_rel_vs_baseline": selected_rel,
        "available_controls": ";".join(row["policy_action"] for row in controls),
        "best_control_action": best_control_action,
        "best_control_ate": best_control_ate,
        "selected_minus_best_control_ate": (
            float(selected_ate) - float(best_control_ate)
            if selected_ate is not None and best_control_ate is not None
            else None
        ),
        "selected_better_controls": selected_better_controls,
        "classification": row_classification("risk", selected_rel, selected_better_controls),
        "metric_source": rel(R70_RUNTIME),
        "repair_source": "R70_seq09_fresh_control_safe_boundary_v2",
    }


def seq08_from_r75() -> dict[str, Any]:
    if not R75_SUMMARY.is_file():
        return {
            "schema": "acl2_v118tf_stage4_r76_repaired_action_surface_row_v1",
            "seq": "08",
            "selected_action": "risk_exclude67",
            "classification": "R75_SUMMARY_MISSING_PENDING",
            "metric_source": rel(R75_SUMMARY),
            "repair_source": "R75_pending",
        }
    summary = json.loads(R75_SUMMARY.read_text(encoding="utf-8"))
    candidate = summary.get("r74_candidate", {})
    baseline = summary.get("baseline", {})
    prior_random = summary.get("r71_original_random_control", {})
    controls = list(summary.get("result_rows", []))
    baseline_ate = fnum(baseline.get("ate"))
    selected_ate = fnum(candidate.get("ate"))
    selected_rel = rel_vs_baseline(baseline_ate, selected_ate)
    control_items = [(str(row.get("role")), fnum(row.get("ate"))) for row in controls]
    if prior_random.get("eval_exists"):
        control_items.append(("r71_original_random_control", fnum(prior_random.get("ate"))))
    valid_controls = [(name, ate) for name, ate in control_items if ate is not None]
    best_control_action = ""
    best_control_ate = None
    if valid_controls:
        best_control_action, best_control_ate = min(valid_controls, key=lambda item: float(item[1]))
    selected_better_controls = (
        selected_ate is not None and bool(valid_controls) and all(float(selected_ate) < float(ate) for _, ate in valid_controls)
    )
    complete = bool(summary.get("complete")) and selected_ate is not None and baseline_ate is not None
    classification = (
        row_classification("risk_exclude67", selected_rel, selected_better_controls)
        if complete
        else "R75_INCOMPLETE_PENDING"
    )
    return {
        "schema": "acl2_v118tf_stage4_r76_repaired_action_surface_row_v1",
        "seq": "08",
        "selected_action": "risk_exclude67",
        "selection_reason": "R74_exclude67_source_frame_repair_validated_by_R75_matched_controls",
        "baseline_ate": baseline_ate,
        "selected_ate": selected_ate,
        "selected_rel_vs_baseline": selected_rel,
        "available_controls": ";".join(name for name, _ in valid_controls),
        "best_control_action": best_control_action,
        "best_control_ate": best_control_ate,
        "selected_minus_best_control_ate": (
            float(selected_ate) - float(best_control_ate)
            if selected_ate is not None and best_control_ate is not None
            else None
        ),
        "selected_better_controls": selected_better_controls,
        "classification": classification,
        "r75_decision": summary.get("stage4_r75_decision"),
        "metric_source": rel(R75_SUMMARY),
        "repair_source": "R74_source_subset_plus_R75_matched_controls",
    }


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ACL2 v118 Stage4-R76 LB-AR Repaired Action Surface Aggregate Audit",
        "",
        f"- decision: `{summary['stage4_r76_decision']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- fresh_holdout_after_rule: `{summary['fresh_holdout_after_rule']}`",
        f"- actionable_pass_sequences: `{','.join(summary['actionable_pass_sequences'])}`",
        f"- safe_abstain_sequences: `{','.join(summary['safe_abstain_sequences'])}`",
        f"- rejected_or_pending_sequences: `{','.join(summary['rejected_or_pending_sequences'])}`",
        "",
        "## Rows",
        "",
        "| seq | selected | rel | best control | classification | source |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seq']} | `{row.get('selected_action', '')}` | {row.get('selected_rel_vs_baseline', '')} | "
            f"`{row.get('best_control_action', '')}` {row.get('best_control_ate', '')} | "
            f"`{row.get('classification', '')}` | `{row.get('repair_source', '')}` |"
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
    rows = base_rows_from_r68()
    rows.append(seq08_from_r75())
    rows.append(seq09_from_r70())
    rows = sorted(rows, key=lambda row: str(row["seq"]))
    actionable_pass = [row for row in rows if row.get("classification") == "ACTIONABLE_SEMANTIC_CONTROL_PASS"]
    safe_abstain = [row for row in rows if row.get("classification") == "SAFE_ABSTAIN_BASELINE_BEST"]
    rejected_or_pending = [
        row
        for row in rows
        if row.get("classification") not in {"ACTIONABLE_SEMANTIC_CONTROL_PASS", "SAFE_ABSTAIN_BASELINE_BEST"}
    ]
    selected_rels = [
        float(row["selected_rel_vs_baseline"])
        for row in rows
        if row.get("selected_rel_vs_baseline") is not None
    ]
    actionable_rels = [
        float(row["selected_rel_vs_baseline"])
        for row in actionable_pass
        if row.get("selected_rel_vs_baseline") is not None
    ]
    if not rejected_or_pending and len(actionable_pass) == 6 and len(safe_abstain) == 1:
        decision = "R76_LB_AR_FRESH_ACTION_SURFACE_REPAIRED_LOCAL_AGGREGATE_PASS_GLOBAL_FALSE"
    else:
        decision = "R76_LB_AR_REPAIRED_ACTION_SURFACE_NOT_PROMOTABLE_OR_PENDING"
    summary = {
        "schema": "acl2_v118tf_stage4_r76_repaired_action_surface_summary_v1",
        "stage4_r76_decision": decision,
        "global_goal_achieved": False,
        "fresh_holdout_after_rule": False,
        "claim_level": "postfinal_exploratory_lbar_action_surface_repair_not_blind_holdout",
        "sequence_count": len(rows),
        "actionable_pass_count": len(actionable_pass),
        "safe_abstain_count": len(safe_abstain),
        "rejected_or_pending_count": len(rejected_or_pending),
        "actionable_pass_sequences": [row["seq"] for row in actionable_pass],
        "safe_abstain_sequences": [row["seq"] for row in safe_abstain],
        "rejected_or_pending_sequences": [row["seq"] for row in rejected_or_pending],
        "actionable_median_rel_vs_baseline": median(actionable_rels) if actionable_rels else "",
        "selected_all_sequence_median_rel_vs_baseline": median(selected_rels) if selected_rels else "",
        "outputs": {
            "rows": rel(ROWS_PATH),
            "summary": rel(SUMMARY_PATH),
            "report": rel(REPORT_PATH),
        },
        "boundary": (
            "R76 is a repaired LB-AR action-surface aggregate, not a global v118 success. "
            "It reuses post-final exploratory evidence and therefore keeps fresh_holdout_after_rule=false. "
            "Promotion would require freezing the R76 rule family and validating it on a genuinely new split."
        ),
        "result_rows": rows,
    }
    write_csv(ROWS_PATH, rows)
    write_json(SUMMARY_PATH, summary)
    write_report(summary, rows)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
