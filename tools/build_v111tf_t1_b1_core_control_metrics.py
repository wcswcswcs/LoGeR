#!/usr/bin/env python3
"""Summarize ACL2 v111TF T1 B1 core-control metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v109tf_stage2_f_core_ablation_metrics as stage2m  # noqa: E402
import build_v110r_stage3_pilot_metrics as stage3m  # noqa: E402
import build_v110r_stage4_full_validation_metrics as stage4m  # noqa: E402
import build_v110r_stage8_b1_full_control_metrics as m  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"
T1 = RESULT_ROOT / "batch_t_t1_b1_core_controls"
V110_STAGE4 = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality/stage4_full_00_01_02_05_validation"
CONFIG_ROWS = T1 / "action_config_rows.csv"
RUN_RESULTS = T1 / "run_results.csv"
WORKSPACE = T1 / "workspace"
SEQUENCES = ("00", "01", "02", "05")
CAUSAL_MARGIN = 0.03


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stage3m.base.clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    stage3m.OUT = T1
    stage3m.CONFIG_ROWS = CONFIG_ROWS
    stage3m.RUN_RESULTS = RUN_RESULTS
    stage3m.WORKSPACE = WORKSPACE
    stage3m.SEQUENCES = SEQUENCES
    stage3m.install_stage3_overrides()

    m.STAGE4 = V110_STAGE4
    m.STAGE8 = T1
    m.CONFIG_ROWS = CONFIG_ROWS
    m.RUN_RESULTS = RUN_RESULTS
    m.WORKSPACE = WORKSPACE
    m.SEQUENCES = SEQUENCES
    m.CAUSAL_MARGIN = CAUSAL_MARGIN

    config_rows = m.read_csv(CONFIG_ROWS)
    latest = stage2m.latest_run_results(m.read_csv(RUN_RESULTS))
    full_rows, rolling_rows, local_rows, fidelity_rows = stage2m.metric_rows(config_rows, latest)
    for rows in (full_rows, rolling_rows, local_rows, fidelity_rows):
        stage3m.add_candidate_metadata(rows)
        for row in rows:
            row["schema"] = str(row.get("schema", "")).replace("acl2_v110r_stage3", "acl2_v111tf_t1_core")

    f19_control = stage4m.f19_rows()
    control_policy_rows = stage4m.policy_summary_rows(full_rows, rolling_rows, fidelity_rows, f19_control)
    for row in control_policy_rows:
        row["schema"] = "acl2_v111tf_t1_core_control_policy_summary_row_v1"
        row["source_stage"] = "v111_t1_b1_core_controls"
    semantic_rows = m.stage4_b1_semantic_rows()
    for row in semantic_rows:
        row["schema"] = "acl2_v111tf_t1_core_frozen_b1_semantic_policy_summary_row_v1"
        row["source_stage"] = "frozen_v110_stage4_b1_semantic_rows"
    combined_rows = semantic_rows + control_policy_rows
    stage4_full_rows = m.read_csv(V110_STAGE4 / "full_metric_rows.csv")
    per_seq_rows = m.per_sequence_summary(full_rows, stage4_full_rows)

    metric_complete = len(full_rows) == len(config_rows) and all(m.bool_value(row.get("metric_available")) for row in full_rows)
    all_action = len(fidelity_rows) == len(config_rows) and all(m.bool_value(row.get("action_fidelity_pass")) for row in fidelity_rows)
    decision = m.semantic_decision_row(combined_rows, metric_complete, all_action)
    decision["schema"] = "acl2_v111tf_t1_core_b1_semantic_decision_row_v1"
    decision["causal_margin_required"] = CAUSAL_MARGIN

    observed_counts: dict[str, int] = {}
    for row in latest.values():
        if stage2m.safe_rc(row) == 0:
            phase = str(row.get("phase", ""))
            observed_counts[phase] = observed_counts.get(phase, 0) + 1

    summary = {
        "schema": "acl2_v111tf_t1_b1_core_control_summary_v1",
        "metric_complete": metric_complete,
        "all_action_fidelity": all_action,
        "observed_prepare_count": observed_counts.get("prepare", 0),
        "observed_run_worker_count": observed_counts.get("run_worker", 0),
        "observed_evaluate_count": observed_counts.get("evaluate", 0),
        "observed_report_count": observed_counts.get("report", 0),
        "expected_run_worker_count": len(config_rows),
        "full_metric_row_count": len(full_rows),
        "rolling_metric_row_count": len(rolling_rows),
        "local_handoff_metric_row_count": len(local_rows),
        "action_fidelity_row_count": len(fidelity_rows),
        "control_policy_summary_row_count": len(control_policy_rows),
        "combined_policy_summary_row_count": len(combined_rows),
        "semantic_causality_pass": decision["semantic_causality_pass"],
        "semantic_causality_pass_core_subset": decision["semantic_causality_pass"],
        "taxonomy": decision["taxonomy"],
        "blocker": decision["blocker"],
        "core_subset_only": True,
        "missing_v111_t1_required_controls": [
            "semantic_shuffle_seed0..9",
            "same_count_random_seed0..50",
            "same_bucket_random_seed0..50",
            "schedule_only_matched_seed0..20",
            "role_rotation_dynamic_to_stable",
            "role_rotation_dynamic_to_weak",
            "high_risk_without_semantic_trust",
            "dynamic_only",
            "boundary_only",
            "weak_context_only",
        ],
        "outputs": {
            "full_metric_rows": rel(T1 / "full_metric_rows.csv"),
            "rolling_metric_rows": rel(T1 / "rolling_metric_rows.csv"),
            "local_handoff_metric_rows": rel(T1 / "local_handoff_metric_rows.csv"),
            "action_fidelity_rows": rel(T1 / "action_fidelity_rows.csv"),
            "control_policy_summary_rows": rel(T1 / "control_policy_summary_rows.csv"),
            "combined_policy_summary_rows": rel(T1 / "combined_policy_summary_rows.csv"),
            "per_sequence_control_rows": rel(T1 / "per_sequence_control_rows.csv"),
            "semantic_decision_rows": rel(T1 / "semantic_decision_rows.csv"),
            "report": rel(T1 / "T1_B1_CORE_CONTROL_SEMANTIC_CAUSALITY_REPORT.md"),
            "summary": rel(T1 / "t1_core_summary.json"),
        },
    }

    m.write_csv(T1 / "full_metric_rows.csv", full_rows)
    m.write_csv(T1 / "rolling_metric_rows.csv", rolling_rows)
    m.write_csv(T1 / "local_handoff_metric_rows.csv", local_rows)
    m.write_csv(T1 / "action_fidelity_rows.csv", fidelity_rows)
    m.write_csv(T1 / "control_policy_summary_rows.csv", control_policy_rows)
    m.write_csv(T1 / "combined_policy_summary_rows.csv", combined_rows)
    m.write_csv(T1 / "per_sequence_control_rows.csv", per_seq_rows)
    m.write_csv(T1 / "semantic_decision_rows.csv", [decision])
    write_json(T1 / "t1_core_summary.json", summary)
    report = m.build_report(summary, decision, combined_rows).replace(
        "# ACL2 v110R Stage8 B1 Full Control Report",
        "# ACL2 v111TF T1 B1 Core Control Report",
    )
    report += (
        "\n\n## v111 Boundary\n\n"
        "This is only the v111 T1 core-control subset. It does not satisfy the full T1 stronger-control requirement "
        "until multi-seed shuffle/random, same-bucket, schedule-only, role-rotation, and role-only controls are added.\n"
    )
    m.write_text(T1 / "T1_B1_CORE_CONTROL_SEMANTIC_CAUSALITY_REPORT.md", report)
    print(json.dumps(stage3m.base.clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
