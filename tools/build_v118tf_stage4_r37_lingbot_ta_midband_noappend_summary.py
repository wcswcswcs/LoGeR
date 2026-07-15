#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R37 LingBot TA midband no-append repair."""

from __future__ import annotations

import json
from typing import Any

import build_v118tf_stage4_r34_lingbot_ta_guarded_noappend_summary as r34s


ROOT = r34s.ROOT
RESULT_ROOT = r34s.RESULT_ROOT
STAGE = RESULT_ROOT / "stage4_r37_lingbot_ta_midband_noappend"
DATASET = r34s.DATASET
BASELINE_METHOD = r34s.BASELINE_METHOD
MANIFEST = STAGE / "summary/stage4_r37_lingbot_ta_midband_noappend_manifest.json"


def selected_set(manifest: dict[str, Any], method: str, seq: str) -> set[int]:
    return {
        int(value)
        for value in manifest.get("force_non_keyframe_indices_by_method", {})
        .get(method, {})
        .get(seq, [])
    }


def summarize_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    action_dir = STAGE / "runtime_full_thread8/action_traces"
    trace_dir = STAGE / "runtime_full_thread8/traces"
    methods = manifest.get("methods", {})
    manifest_rows = {
        (row.get("method", ""), row.get("seq", "")): row
        for row in r34s.read_csv(STAGE / "summary/stage4_r37_lingbot_ta_midband_noappend_manifest.csv")
    }
    for method, meta in methods.items():
        for seq in ("00", "02"):
            selected = selected_set(manifest, method, seq)
            metrics = r34s.read_json(r34s.metric_path(seq, method))
            baseline = r34s.read_json(r34s.metric_path(seq, BASELINE_METHOD))
            ate = r34s.fnum(metrics.get("ate"))
            baseline_ate = r34s.fnum(baseline.get("ate"))
            action_path = action_dir / f"{method}_seq{seq}.jsonl"
            fi_path = trace_dir / f"{method}_seq{seq}.jsonl"
            manifest_row = manifest_rows.get((method, seq), {})
            rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r37_lingbot_ta_midband_noappend_row_v1",
                    "dataset": DATASET,
                    "seq": seq,
                    "method": method,
                    "branch": "LB-TA",
                    "policy": meta.get("policy", ""),
                    "role": meta.get("role", ""),
                    "selection_mode": manifest_row.get("selection_mode", ""),
                    "risk_q50": manifest_row.get("risk_q50", ""),
                    "forced_non_keyframe_count": len(selected),
                    "min_frame_gap": manifest_row.get("min_frame_gap", ""),
                    "eval_exists": r34s.metric_path(seq, method).exists(),
                    "complete_marker_exists": r34s.complete_path(seq, method).exists(),
                    "ate": metrics.get("ate", ""),
                    "rpe_rot": metrics.get("rpe_rot", ""),
                    "rpe_trans": metrics.get("rpe_trans", ""),
                    "baseline_method": BASELINE_METHOD,
                    "baseline_ate": baseline.get("ate", ""),
                    "baseline_rpe_rot": baseline.get("rpe_rot", ""),
                    "baseline_rpe_trans": baseline.get("rpe_trans", ""),
                    "ate_rel_improvement_vs_default": (
                        (baseline_ate - ate) / baseline_ate
                        if ate is not None and baseline_ate not in (None, 0.0)
                        else ""
                    ),
                    "guarded_risk_median": manifest_row.get("guarded_risk_median", ""),
                    "guarded_risk_min": manifest_row.get("guarded_risk_min", ""),
                    "guarded_risk_max": manifest_row.get("guarded_risk_max", ""),
                    "eval_json": (
                        r34s.rel(r34s.metric_path(seq, method))
                        if r34s.metric_path(seq, method).exists()
                        else str(r34s.metric_path(seq, method))
                    ),
                    "action_trace": r34s.rel(action_path) if action_path.exists() else str(action_path),
                    "fi_trace": r34s.rel(fi_path) if fi_path.exists() else str(fi_path),
                    **r34s.action_trace_stats(action_path, selected),
                    **r34s.fi_trace_stats(fi_path),
                }
            )
    return rows


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ACL2 v118-TF Stage4-R37 LB-TA Midband No-Append Report",
        "",
        f"- decision: `{summary['decision']}`",
        f"- complete: `{summary['complete']}`",
        f"- action_fidelity: `{summary['action_fidelity']}`",
        f"- baseline_gate: `{summary['baseline_gate']}`",
        f"- candidate_better_all_controls: `{summary['candidate_better_all_controls']}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        "",
        "## Rows",
        "",
        "| seq | role | ATE | rel vs default | action fidelity |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['seq']}` | `{row['role']}` | {row['ate']} | "
            f"{row['ate_rel_improvement_vs_default']} | {row['selected_action_fidelity']} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "R37 is a post-R36 calibration repair that tests de-clustered median-risk semantic-support selection on both seq00 and seq02. It keeps fixed count, fixed hard no-append runtime surface, and matched temporal-random control.",
    ]
    r34s.write_text(STAGE / "summary/STAGE4_R37_LINGBOT_TA_MIDBAND_NOAPPEND_REPORT.md", "\n".join(lines))


def main() -> None:
    manifest = r34s.read_json(MANIFEST)
    rows = summarize_rows(manifest)
    rows_csv = STAGE / "summary/stage4_r37_lingbot_ta_midband_noappend_rows.csv"
    r34s.write_csv(rows_csv, rows)
    comparisons = r34s.compare(rows)
    complete = bool(rows) and all(row["eval_exists"] and row["complete_marker_exists"] for row in rows)
    action_fidelity = bool(rows) and all(r34s.fnum(row.get("selected_action_fidelity")) == 1.0 for row in rows)
    candidate_better_all_controls = all(
        comparisons.get(role, {}).get("metrics", {}).get("ate", {}).get("all_candidate_better_than_control") is True
        for role in ("reverse_control", "matched_temporal_random_control")
    )
    baseline_gate = bool(
        comparisons.get("baseline", {})
        .get("ate_rel_improvement_vs_default", {})
        .get("pilot_gate", False)
    )
    global_goal = complete and action_fidelity and candidate_better_all_controls and baseline_gate
    if global_goal:
        decision = "TA_MIDBAND_NOAPPEND_PASS"
    elif candidate_better_all_controls and not baseline_gate:
        decision = "TA_MIDBAND_NOAPPEND_CONTROL_PASS_BASELINE_GATE_FAIL"
    else:
        decision = "TA_MIDBAND_NOAPPEND_CONTROL_OR_BASELINE_NO_GO"
    summary = {
        "schema": "acl2_v118tf_stage4_r37_lingbot_ta_midband_noappend_summary_v1",
        "stage": r34s.rel(STAGE),
        "branch": "LB-TA",
        "decision": decision,
        "complete": complete,
        "action_fidelity": action_fidelity,
        "candidate_better_all_controls": candidate_better_all_controls,
        "baseline_gate": baseline_gate,
        "global_goal_achieved": global_goal,
        "row_count": len(rows),
        "comparisons": comparisons,
        "outputs": {
            "manifest": r34s.rel(MANIFEST),
            "rows_csv": r34s.rel(rows_csv),
            "report": r34s.rel(STAGE / "summary/STAGE4_R37_LINGBOT_TA_MIDBAND_NOAPPEND_REPORT.md"),
            "runtime": r34s.rel(STAGE / "runtime_full_thread8"),
        },
        "boundary": (
            "Post-R36 all-sequence midband semantic-support repair. Success requires beating "
            "high-risk reverse and temporal-random controls plus default baseline gate."
        ),
    }
    r34s.write_json(STAGE / "summary/stage4_r37_lingbot_ta_midband_noappend_summary.json", summary)
    build_report(summary, rows)
    r34s.add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": "Stage4-R37",
            "surface_or_branch": "LB-TA",
            "status": decision,
            "artifact": r34s.rel(STAGE / "summary/stage4_r37_lingbot_ta_midband_noappend_summary.json"),
            "notes": "Post-R36 all-sequence midband de-clustered semantic-support hard no-append with high-risk reverse and temporal-random controls",
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
