#!/usr/bin/env python3
"""Refresh LB-AI branch closure after R23-R27 runtime and holdout evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
BRANCH = RESULT_ROOT / "branches/LB-AI"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"

ARTIFACTS = {
    "r23": RESULT_ROOT / "stage4_r23_lingbot_ai_anchor_initialization/summary/stage4_r23_lingbot_ai_anchor_initialization_summary.json",
    "r24": RESULT_ROOT / "stage4_r24_lingbot_ai_bin_anchor/summary/stage4_r24_lingbot_ai_bin_anchor_summary.json",
    "r25": RESULT_ROOT / "stage4_r25_lingbot_ai_internal_read_anchor/summary/stage4_r25_lingbot_ai_internal_read_anchor_summary.json",
    "r26": RESULT_ROOT / "stage4_r26_lingbot_ai_calibrated_polarity_anchor/summary/stage4_r26_lingbot_ai_calibrated_polarity_anchor_summary.json",
    "r27_cue": RESULT_ROOT / "stage4_r27_holdout_cue_prep/summary/stage4_r27_holdout_cue_prep_summary.json",
    "r27": RESULT_ROOT / "stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor/summary/stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_summary.json",
}
R27_ROWS = RESULT_ROOT / "stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor/summary/stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor_rows.csv"
R27_COMBINED_ROWS = RESULT_ROOT / "stage4_r27_lingbot_ai_holdout_calibrated_polarity_anchor/summary/stage4_r27_combined_r26_r27_ate_control_rows.csv"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def add_registry_row(row: dict[str, Any]) -> None:
    rows = read_csv(REGISTRY)
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
    with REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def main() -> None:
    summaries = {key: read_json(path) for key, path in ARTIFACTS.items()}
    r27 = summaries["r27"]
    if not r27:
        raise FileNotFoundError(ARTIFACTS["r27"])
    status = "COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS"
    run_rows = [
        {
            "schema": "acl2_v118tf_branch_run_manifest_row_v2",
            "branch": "LB-AI",
            "stage": "Stage4-R23",
            "policy_family": "semantic_anchor_first32",
            "status": summaries["r23"].get("stage4_r23_decision", ""),
            "artifact": rel(ARTIFACTS["r23"]),
        },
        {
            "schema": "acl2_v118tf_branch_run_manifest_row_v2",
            "branch": "LB-AI",
            "stage": "Stage4-R24",
            "policy_family": "semantic_bin_balanced_anchor_first32",
            "status": summaries["r24"].get("stage4_r24_decision", ""),
            "artifact": rel(ARTIFACTS["r24"]),
        },
        {
            "schema": "acl2_v118tf_branch_run_manifest_row_v2",
            "branch": "LB-AI",
            "stage": "Stage4-R25",
            "policy_family": "internal_qk_nondefault_anchor_first32",
            "status": summaries["r25"].get("stage4_r25_decision", ""),
            "artifact": rel(ARTIFACTS["r25"]),
        },
        {
            "schema": "acl2_v118tf_branch_run_manifest_row_v2",
            "branch": "LB-AI",
            "stage": "Stage4-R26",
            "policy_family": "calibrated_polarity_dev_00_02",
            "status": summaries["r26"].get("stage4_r26_decision", ""),
            "artifact": rel(ARTIFACTS["r26"]),
        },
        {
            "schema": "acl2_v118tf_branch_run_manifest_row_v2",
            "branch": "LB-AI",
            "stage": "Stage4-R27-CuePrep",
            "policy_family": "fresh_holdout_cue_prep_01_05",
            "status": summaries["r27_cue"].get("stage4_r27_cue_prep_decision", ""),
            "artifact": rel(ARTIFACTS["r27_cue"]),
        },
        {
            "schema": "acl2_v118tf_branch_run_manifest_row_v2",
            "branch": "LB-AI",
            "stage": "Stage4-R27",
            "policy_family": "frozen_calibrated_polarity_holdout_01_05",
            "status": r27.get("stage4_r27_decision", ""),
            "artifact": rel(ARTIFACTS["r27"]),
        },
    ]
    geometry_rows = []
    for row in read_csv(R27_COMBINED_ROWS):
        if row.get("role") == "candidate":
            geometry_rows.append(
                {
                    "schema": "acl2_v118tf_branch_geometry_metric_row_v2",
                    "branch": "LB-AI",
                    "source_stage": row.get("source_stage", ""),
                    "seq": row.get("seq", ""),
                    "policy": row.get("policy", ""),
                    "ate_full": row.get("ate", ""),
                    "baseline_ate": row.get("baseline_ate", ""),
                    "ate_rel_improvement_vs_default": row.get("ate_rel_improvement_vs_default", ""),
                    "rpe_rot": row.get("rpe_rot", ""),
                    "rpe_trans": row.get("rpe_trans", ""),
                    "status": "CANDIDATE_METRIC_RECORDED",
                    "blocker": "",
                }
            )
    control_rows = []
    for role, payload in r27.get("holdout_comparisons", {}).items():
        control_rows.append(
            {
                "schema": "acl2_v118tf_branch_control_comparison_row_v2",
                "branch": "LB-AI",
                "policy": "AI4_HOLDOUT_CALIBRATED_POLARITY_ANCHOR_FIRST32",
                "control": role,
                "comparison_metric": "ATE",
                "status": (
                    "candidate_better_all_holdout"
                    if payload.get("metrics", {}).get("ate", {}).get("all_candidate_better_than_control")
                    else "control_catches_or_beats_candidate"
                ),
                "median_candidate_minus_control": payload.get("metrics", {}).get("ate", {}).get("median_candidate_minus_control", ""),
                "all_rel_margins_gt_0p01": payload.get("rel_improvement_margin_vs_control", {}).get("all_margins_gt_0p01", ""),
            }
        )
    action_rows = [
        {
            "schema": "acl2_v118tf_branch_action_fidelity_row_v2",
            "branch": "LB-AI",
            "stage": "Stage4-R27",
            "complete": r27.get("complete"),
            "action_fidelity": r27.get("action_fidelity"),
            "row_count": r27.get("row_count"),
            "artifact": rel(ARTIFACTS["r27"]),
        }
    ]
    fail_forward_lines = [
        "# LB-AI Fail-Forward Log",
        "",
        "- R23 semantic first32 anchor: baseline gate passed but candidate failed controls on seq00.",
        "- R24 bin-balanced semantic anchor: temporal/bin coverage did not fix attribution; baseline gate failed.",
        "- R25 internal-QK anchor: improved default on 00/02 but failed controls on seq00.",
        "- R26 calibrated polarity: first dev candidate beating default/opposite/random on 00/02, but required fresh holdout because rule was post-R25.",
        "- R27 fresh 01/05 holdout: seq01 improved, seq05 harmed default by 5.14% and lost opposite-polarity control; four-sequence max-harm gate failed.",
        "",
        "No same-version retuning from 01/05 is allowed by the plan.",
    ]
    summary = {
        "schema": "acl2_v118tf_branch_decision_summary_v2",
        "branch": "LB-AI",
        "model": "LingBot",
        "surface": "LB-Anchor",
        "operation": "Anchor initialization",
        "status": status,
        "runtime_launched": True,
        "gpu_runtime_launched": True,
        "primary_blocker": "R27 fresh 01/05 holdout failed four-sequence ATE max-harm and control gates",
        "repair_attempt_count": len(run_rows),
        "stage4_runtime_stages": [row["stage"] for row in run_rows],
        "latest_decision": r27.get("stage4_r27_decision"),
        "global_goal_achieved": False,
        "four_sequence_ate": r27.get("four_sequence_ate", {}),
        "holdout_candidate_better_all_controls": r27.get("holdout_candidate_better_all_controls"),
        "four_sequence_ate_gate": r27.get("four_sequence_ate_gate"),
        "boundary": "LB-AI is no longer accurately described as Stage3 structural-blocked; it ran R23-R27 runtime variants and closes as No-Go after required variants plus failed fresh holdout.",
        "outputs": {
            "run_manifest": rel(BRANCH / "LB-AI_RUN_MANIFEST.csv"),
            "action_fidelity": rel(BRANCH / "LB-AI_ACTION_FIDELITY.csv"),
            "geometry_metrics": rel(BRANCH / "LB-AI_GEOMETRY_METRICS.csv"),
            "control_comparison": rel(BRANCH / "LB-AI_CONTROL_COMPARISON.csv"),
            "fail_forward_log": rel(BRANCH / "LB-AI_FAIL_FORWARD_LOG.md"),
            "summary": rel(BRANCH / "LB-AI_DECISION_SUMMARY.json"),
            "report": rel(BRANCH / "LB-AI_REPORT.md"),
        },
    }
    BRANCH.mkdir(parents=True, exist_ok=True)
    write_csv(BRANCH / "LB-AI_RUN_MANIFEST.csv", run_rows)
    write_csv(BRANCH / "LB-AI_ACTION_FIDELITY.csv", action_rows)
    write_csv(BRANCH / "LB-AI_GEOMETRY_METRICS.csv", geometry_rows)
    write_csv(BRANCH / "LB-AI_CONTROL_COMPARISON.csv", control_rows)
    (BRANCH / "LB-AI_FAIL_FORWARD_LOG.md").write_text("\n".join(fail_forward_lines) + "\n", encoding="utf-8")
    write_json(BRANCH / "LB-AI_DECISION_SUMMARY.json", summary)
    report = [
        "# ACL2 v118-TF LB-AI Report",
        "",
        f"- status: `{status}`",
        f"- latest_decision: `{r27.get('stage4_r27_decision')}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- primary_blocker: `{summary['primary_blocker']}`",
        "",
        "## Four-Sequence ATE",
        "",
        "```json",
        json.dumps(r27.get("four_sequence_ate", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Boundary",
        "",
        summary["boundary"],
    ]
    (BRANCH / "LB-AI_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R27-LBAI-Closure",
            "surface_or_branch": "LB-AI",
            "status": status,
            "artifact": rel(BRANCH / "LB-AI_DECISION_SUMMARY.json"),
            "notes": "LB-AI branch artifact refreshed from stale structural-blocked to runtime No-Go after R23-R27 and failed fresh 01/05 holdout",
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
