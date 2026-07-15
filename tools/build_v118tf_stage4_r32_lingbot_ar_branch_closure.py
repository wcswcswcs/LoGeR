#!/usr/bin/env python3
"""Refresh LB-AR branch closure after R28-R32 runtime evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
BRANCH = RESULT_ROOT / "branches/LB-AR"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"

ARTIFACTS = {
    "r28": RESULT_ROOT / "stage4_r28_lingbot_ar_anchor_read/summary/stage4_r28_lingbot_ar_anchor_read_summary.json",
    "r29": RESULT_ROOT / "stage4_r29_lingbot_ar_anchor_read_sdpa_repair/summary/stage4_r29_lingbot_ar_anchor_read_summary.json",
    "r30": RESULT_ROOT / "stage4_r30_lingbot_ar_source_value_scaling/summary/stage4_r30_lingbot_ar_anchor_read_summary.json",
    "r31": RESULT_ROOT / "stage4_r31_lingbot_ar_source_value_cue_ablation/summary/stage4_r31_lingbot_ar_source_value_cue_ablation_summary.json",
    "r32": RESULT_ROOT / "stage4_r32_lingbot_ar_component_shuffle/summary/stage4_r32_lingbot_ar_source_value_cue_ablation_summary.json",
}
ROW_ARTIFACTS = {
    "r29": RESULT_ROOT / "stage4_r29_lingbot_ar_anchor_read_sdpa_repair/summary/stage4_r29_lingbot_ar_anchor_read_rows.csv",
    "r30": RESULT_ROOT / "stage4_r30_lingbot_ar_source_value_scaling/summary/stage4_r30_lingbot_ar_anchor_read_rows.csv",
    "r31": RESULT_ROOT / "stage4_r31_lingbot_ar_source_value_cue_ablation/summary/stage4_r31_lingbot_ar_source_value_cue_ablation_rows.csv",
    "r32": RESULT_ROOT / "stage4_r32_lingbot_ar_component_shuffle/summary/stage4_r32_lingbot_ar_source_value_cue_ablation_rows.csv",
}


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


def stage_decision(stage: str, summary: dict[str, Any]) -> str:
    for key in (f"stage4_{stage}_decision", f"stage4_{stage}_cue_prep_decision"):
        if key in summary:
            return str(summary[key])
    return ""


def main() -> None:
    summaries = {key: read_json(path) for key, path in ARTIFACTS.items()}
    if not summaries["r32"]:
        raise FileNotFoundError(ARTIFACTS["r32"])
    status = "COMPLETE_NO_GO_AFTER_REQUIRED_VARIANTS"
    run_rows = [
        {
            "schema": "acl2_v118tf_branch_run_manifest_row_v2",
            "branch": "LB-AR",
            "stage": "Stage4-R28",
            "policy_family": "selected_query_attention_weight_flashinfer_attempt",
            "status": stage_decision("r28", summaries["r28"]),
            "artifact": rel(ARTIFACTS["r28"]),
        },
        {
            "schema": "acl2_v118tf_branch_run_manifest_row_v2",
            "branch": "LB-AR",
            "stage": "Stage4-R29",
            "policy_family": "selected_query_attention_weight_sdpa_repair",
            "status": stage_decision("r29", summaries["r29"]),
            "artifact": rel(ARTIFACTS["r29"]),
        },
        {
            "schema": "acl2_v118tf_branch_run_manifest_row_v2",
            "branch": "LB-AR",
            "stage": "Stage4-R30",
            "policy_family": "source_value_scaling_full_plus_reverse_shuffle_controls",
            "status": stage_decision("r30", summaries["r30"]),
            "artifact": rel(ARTIFACTS["r30"]),
        },
        {
            "schema": "acl2_v118tf_branch_run_manifest_row_v2",
            "branch": "LB-AR",
            "stage": "Stage4-R31",
            "policy_family": "source_value_cue_ablation_matrix",
            "status": stage_decision("r31", summaries["r31"]),
            "artifact": rel(ARTIFACTS["r31"]),
        },
        {
            "schema": "acl2_v118tf_branch_run_manifest_row_v2",
            "branch": "LB-AR",
            "stage": "Stage4-R32",
            "policy_family": "full_cue_component_shuffle_mechanism_controls",
            "status": stage_decision("r32", summaries["r32"]),
            "artifact": rel(ARTIFACTS["r32"]),
        },
    ]
    geometry_rows: list[dict[str, Any]] = []
    for stage, path in ROW_ARTIFACTS.items():
        for row in read_csv(path):
            role = row.get("role", "")
            if not role.startswith("candidate"):
                continue
            geometry_rows.append(
                {
                    "schema": "acl2_v118tf_branch_geometry_metric_row_v2",
                    "branch": "LB-AR",
                    "source_stage": f"Stage4-{stage.upper()}",
                    "seq": row.get("seq", ""),
                    "policy": row.get("policy", ""),
                    "role": role,
                    "ate_full": row.get("ate", ""),
                    "baseline_ate": row.get("baseline_ate", ""),
                    "ate_rel_improvement_vs_default": row.get("ate_rel_improvement_vs_default", ""),
                    "rpe_rot": row.get("rpe_rot", ""),
                    "rpe_trans": row.get("rpe_trans", ""),
                    "status": "CANDIDATE_METRIC_RECORDED",
                    "blocker": "",
                }
            )
    action_rows = [
        {
            "schema": "acl2_v118tf_branch_action_fidelity_row_v2",
            "branch": "LB-AR",
            "stage": f"Stage4-{stage.upper()}",
            "complete": summary.get("complete"),
            "action_fidelity": summary.get("action_fidelity"),
            "row_count": summary.get("row_count"),
            "artifact": rel(ARTIFACTS[stage]),
        }
        for stage, summary in summaries.items()
        if summary
    ]
    control_rows: list[dict[str, Any]] = []
    for stage in ("r30", "r31", "r32"):
        summary = summaries[stage]
        comps = summary.get("comparisons") or summary.get("control_comparisons") or {}
        for control, payload in comps.items():
            if control in {"baseline"}:
                continue
            if stage == "r30":
                ate_payload = payload.get("metrics", {}).get("ate", {})
                control_rows.append(
                    {
                        "schema": "acl2_v118tf_branch_control_comparison_row_v2",
                        "branch": "LB-AR",
                        "stage": f"Stage4-{stage.upper()}",
                        "variant": "full_three_way",
                        "control": control,
                        "comparison_metric": "ATE",
                        "status": "candidate_better_all_controls" if ate_payload.get("all_candidate_better_than_control") else "control_catches_or_beats_candidate",
                        "median_candidate_minus_control": ate_payload.get("median_candidate_minus_control", ""),
                    }
                )
            else:
                for variant, variant_payload in comps.items():
                    for control, ctrl_payload in variant_payload.items():
                        control_rows.append(
                            {
                                "schema": "acl2_v118tf_branch_control_comparison_row_v2",
                                "branch": "LB-AR",
                                "stage": f"Stage4-{stage.upper()}",
                                "variant": variant,
                                "control": control,
                                "comparison_metric": "ATE",
                                "status": "variant_better_all_controls" if ctrl_payload.get("all_variant_better_than_control") else "control_catches_or_beats_variant",
                                "median_candidate_minus_control": ctrl_payload.get("median_variant_minus_control", ""),
                            }
                        )
                break
    fail_forward_lines = [
        "# LB-AR Fail-Forward Log",
        "",
        "- R28 selected-query anchor-read on FlashInfer produced no valid action-fidelity rows for the planned row type.",
        "- R29 switched to SDPA and fixed action fidelity, but selected-query attention weighting failed baseline/control gates.",
        "- R30 added source-value scaling. The intervention was real and large, but seq00 improvement was paired with seq02 harm; reverse/shuffle controls matched or beat the candidate.",
        "- R31 ran the cue ablation matrix: internal-only, semantic-only, internal+semantic, internal+reliability, semantic+reliability, and full three-way. No variant improved both 00/02 or beat R30 controls.",
        "- R32 shuffled internal, semantic, and reliability components inside the full cue. The best median variant still harmed seq02 and failed controls.",
        "",
        "LB-AR is therefore closed as No-Go after required variants and mechanism controls, not as a global success.",
    ]
    r31_best = summaries["r31"].get("best_variants", {})
    r32_best = summaries["r32"].get("best_variants", {})
    summary = {
        "schema": "acl2_v118tf_branch_decision_summary_v2",
        "branch": "LB-AR",
        "model": "LingBot",
        "surface": "LB-Anchor",
        "operation": "Anchor read",
        "status": status,
        "runtime_launched": True,
        "gpu_runtime_launched": True,
        "primary_blocker": "R28-R32 anchor-read interventions failed stability and control gates; seq02 harm persisted and R30 controls matched or beat candidates",
        "repair_attempt_count": len(run_rows),
        "stage4_runtime_stages": [row["stage"] for row in run_rows],
        "latest_decision": stage_decision("r32", summaries["r32"]),
        "global_goal_achieved": False,
        "r31_best_variants": r31_best,
        "r32_best_variants": r32_best,
        "boundary": "LB-AR is no longer accurately described as Stage3 structural-blocked; it ran R28-R32 runtime variants and closes as No-Go after required cue ablations and component-shuffle mechanism controls.",
        "outputs": {
            "run_manifest": rel(BRANCH / "LB-AR_RUN_MANIFEST.csv"),
            "action_fidelity": rel(BRANCH / "LB-AR_ACTION_FIDELITY.csv"),
            "geometry_metrics": rel(BRANCH / "LB-AR_GEOMETRY_METRICS.csv"),
            "control_comparison": rel(BRANCH / "LB-AR_CONTROL_COMPARISON.csv"),
            "fail_forward_log": rel(BRANCH / "LB-AR_FAIL_FORWARD_LOG.md"),
            "summary": rel(BRANCH / "LB-AR_DECISION_SUMMARY.json"),
            "report": rel(BRANCH / "LB-AR_REPORT.md"),
        },
    }
    BRANCH.mkdir(parents=True, exist_ok=True)
    write_csv(BRANCH / "LB-AR_RUN_MANIFEST.csv", run_rows)
    write_csv(BRANCH / "LB-AR_ACTION_FIDELITY.csv", action_rows)
    write_csv(BRANCH / "LB-AR_GEOMETRY_METRICS.csv", geometry_rows)
    write_csv(BRANCH / "LB-AR_CONTROL_COMPARISON.csv", control_rows)
    (BRANCH / "LB-AR_FAIL_FORWARD_LOG.md").write_text("\n".join(fail_forward_lines) + "\n", encoding="utf-8")
    write_json(BRANCH / "LB-AR_DECISION_SUMMARY.json", summary)
    report = [
        "# ACL2 v118-TF LB-AR Report",
        "",
        f"- status: `{status}`",
        f"- latest_decision: `{summary['latest_decision']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"- primary_blocker: `{summary['primary_blocker']}`",
        "",
        "## R31 Best Variant",
        "",
        "```json",
        json.dumps(r31_best, indent=2, sort_keys=True),
        "```",
        "",
        "## R32 Best Variant",
        "",
        "```json",
        json.dumps(r32_best, indent=2, sort_keys=True),
        "```",
        "",
        "## Boundary",
        "",
        summary["boundary"],
    ]
    (BRANCH / "LB-AR_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R32-LBAR-Closure",
            "surface_or_branch": "LB-AR",
            "status": status,
            "artifact": rel(BRANCH / "LB-AR_DECISION_SUMMARY.json"),
            "notes": "LB-AR branch artifact refreshed from stale structural-blocked to runtime No-Go after R28-R32 attention/source-value/cue-ablation/component-shuffle controls",
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
