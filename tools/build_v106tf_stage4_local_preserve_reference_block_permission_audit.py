#!/usr/bin/env python3
"""Build ACL2 v106 Stage4 permission audit without fabricating action metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V106 = ROOT / "results/acl2_v106tf_lingbot_semantic_aware_memory_role_control"
STAGE2 = V106 / "stage2_moge_metric_verifier"
STAGE3 = V106 / "stage3_memory_role_disambiguation"
OUT = V106 / "stage4_local_preserve_reference_block"


ACTION_ROWS = [
    ("no_action", "control", "required_control"),
    ("anchor_reference_block", "action_a", "stage4_action"),
    ("trajectory_write_block", "action_b", "stage4_action"),
    ("reference_trajectory_block", "action_c", "stage4_action"),
    ("context_only_with_local_preserve", "action_d", "stage4_action"),
    ("geometry_only_role", "control", "required_control"),
    ("semantic_only_role", "control", "required_control"),
    ("same_count_random_role", "control", "required_control"),
    ("semantic_label_shuffle_role", "control", "required_control"),
    ("context_role_rotation", "control", "required_control"),
    ("head_random_same_count", "control", "required_control"),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    stage2 = read_json(STAGE2 / "stage2_summary.json")
    stage3 = read_json(STAGE3 / "stage3_summary.json")

    stage3_pass = bool(stage3.get("stage3_disambiguation_pass", False))
    moge_available = bool(stage2.get("moge_available", False))
    moge_proxy_or_missing = bool(stage2.get("moge_proxy_or_missing", True))
    moge_based_promotion_allowed = bool(stage2.get("stage4_moge_based_action_promotion_allowed", False))
    action_allowed = stage3_pass and moge_available and not moge_proxy_or_missing and moge_based_promotion_allowed
    blocker = "" if action_allowed else "proxy_only_moge_missing_blocks_stage4_action_promotion"
    if not stage3_pass:
        blocker = "stage3_disambiguation_not_passed"

    action_config_rows: list[dict[str, Any]] = []
    action_trace_rows: list[dict[str, Any]] = []
    action_metric_rows: list[dict[str, Any]] = []
    for action_name, action_family, requirement_type in ACTION_ROWS:
        run_status = "eligible_not_run" if action_allowed else "not_run_blocked"
        action_config_rows.append(
            {
                "schema": "acl2_v106tf_stage4_action_config_row_v1",
                "action_name": action_name,
                "action_family": action_family,
                "requirement_type": requirement_type,
                "stage3_disambiguation_pass": stage3_pass,
                "moge_available": moge_available,
                "moge_proxy_or_missing": moge_proxy_or_missing,
                "stage4_moge_based_action_promotion_allowed": moge_based_promotion_allowed,
                "run_status": run_status,
                "blocker": blocker,
                "note": (
                    "No LingBot runtime action was launched; no action metric is claimed."
                    if not action_allowed
                    else "Stage4 action is eligible, but this permission audit does not launch runtime jobs."
                ),
            }
        )
        action_trace_rows.append(
            {
                "schema": "acl2_v106tf_stage4_action_trace_row_v1",
                "action_name": action_name,
                "trace_available": False,
                "blocked_anchor_context_token_count": "",
                "local_window_preserved_token_count": "",
                "anchor_attention_mass_before": "",
                "anchor_attention_mass_after": "",
                "local_window_attention_mass_before": "",
                "local_window_attention_mass_after": "",
                "trajectory_write_blocked_count": "",
                "trajectory_write_allowed_count": "",
                "run_status": run_status,
                "blocker": blocker,
            }
        )
        action_metric_rows.append(
            {
                "schema": "acl2_v106tf_stage4_action_metric_row_v1",
                "action_name": action_name,
                "metric_available": False,
                "bad_L3_median_improvement": "",
                "good_median_harm": "",
                "good_max_harm": "",
                "rolling_worse_fraction_delta": "",
                "trace_fidelity_pass": False,
                "local_window_attention_preserved": False,
                "run_status": run_status,
                "blocker": blocker,
            }
        )

    write_csv(OUT / "action_config_rows.csv", action_config_rows)
    write_csv(OUT / "action_trace_rows.csv", action_trace_rows)
    write_csv(OUT / "action_metric_rows.csv", action_metric_rows)

    summary = {
        "schema": "acl2_v106tf_stage4_permission_audit_summary_v1",
        "stage3_disambiguation_pass": stage3_pass,
        "stage3_rule_profile": stage3.get("rule_profile", ""),
        "moge_available": moge_available,
        "moge_proxy_or_missing": moge_proxy_or_missing,
        "stage4_moge_based_action_promotion_allowed": moge_based_promotion_allowed,
        "stage4_runtime_action_allowed": action_allowed,
        "stage4_runtime_action_run": False,
        "stage4_action_pass": False,
        "stage4_status": "NOT_RUN_PROXY_ONLY_MOGE_MISSING" if not action_allowed else "ELIGIBLE_NOT_RUN",
        "blocker": blocker,
        "no_fabricated_metrics": True,
        "candidate_action_count": len([row for row in ACTION_ROWS if row[2] == "stage4_action"]),
        "required_control_count": len([row for row in ACTION_ROWS if row[2] == "required_control"]),
        "outputs": {
            "action_config_rows": (OUT / "action_config_rows.csv").relative_to(ROOT).as_posix(),
            "action_trace_rows": (OUT / "action_trace_rows.csv").relative_to(ROOT).as_posix(),
            "action_metric_rows": (OUT / "action_metric_rows.csv").relative_to(ROOT).as_posix(),
            "good_harm_attribution": (OUT / "good_harm_attribution.md").relative_to(ROOT).as_posix(),
            "trace_fidelity_report": (OUT / "trace_fidelity_report.md").relative_to(ROOT).as_posix(),
            "stage4_summary": (OUT / "stage4_summary.json").relative_to(ROOT).as_posix(),
        },
    }
    (OUT / "stage4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trace_report = f"""# Stage4 Trace Fidelity Report

Stage4 runtime action was not launched.

Reason:
- stage3_disambiguation_pass: `{stage3_pass}`
- moge_available: `{moge_available}`
- moge_proxy_or_missing: `{moge_proxy_or_missing}`
- stage4_moge_based_action_promotion_allowed: `{moge_based_promotion_allowed}`
- blocker: `{blocker}`

No trace fidelity pass is claimed because there is no runtime action trace.
"""
    (OUT / "trace_fidelity_report.md").write_text(trace_report, encoding="utf-8")
    harm_report = f"""# Stage4 Good Harm Attribution

No Stage4 action was run, so good-harm attribution is unavailable.

The required attribution axes remain:
- local window harmed?
- anchor context harmed?
- trajectory memory harmed?
- decoder output harmed?

No fabricated harm, improvement, or attention-preservation metric is reported.
"""
    (OUT / "good_harm_attribution.md").write_text(harm_report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
