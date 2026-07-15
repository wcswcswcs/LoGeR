#!/usr/bin/env python3
"""Build v109TF Stage2 F19 hard-negative safety candidate metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v109tf_stage2_role_specific_full_candidate_metrics as fullm  # noqa: E402
import build_v109tf_stage2_role_specific_pilot_metrics as rolem  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
OUT = RESULT_ROOT / "stage2_role_specific_safety_candidates"


def configure() -> None:
    fullm.configure()
    rolem.OUT = OUT
    rolem.CONFIG_ROWS = OUT / "action_config_rows.csv"
    rolem.RUN_RESULTS = OUT / "run_results.csv"
    rolem.WORKSPACE = OUT / "workspace"
    rolem.SUMMARY_ROW_SCHEMA = "acl2_v109tf_stage2_role_specific_safety_candidate_summary_row_v1"
    rolem.SUMMARY_SCHEMA = "acl2_v109tf_stage2_role_specific_safety_candidate_summary_v1"
    rolem.SUMMARY_JSON = "role_specific_safety_candidate_summary.json"
    rolem.REPORT_MD = "role_specific_safety_candidate_report.md"
    rolem.REPORT_TITLE = "# ACL2 v109TF Stage2 F19 Safety Candidate Report"
    rolem.SCOPE_NOTE = "full KITTI 00/01/02/05 F19 hard-negative safety candidate"
    rolem.GATE_SCOPE = "KITTI 00/01/02/05 F19 safety candidate"


def build() -> dict:
    configure()
    summary = rolem.build()
    summary["safety_candidate_pass"] = summary.get("role_pilot_pass", False)
    summary["safety_candidate_pre_gate_any_pass"] = summary.get("role_pilot_pre_gate_any_pass", False)
    summary["candidate_scope"] = "full KITTI 00/01/02/05"
    summary["hard_negative_attribution"] = rolem.rel(
        RESULT_ROOT
        / "stage2_role_specific_full_candidates/F_seq01_seq05_hard_negative_attribution.md"
    )
    summary["outputs"]["role_specific_safety_candidate_summary"] = rolem.rel(OUT / rolem.SUMMARY_JSON)
    summary["outputs"]["role_specific_safety_candidate_report"] = rolem.rel(OUT / rolem.REPORT_MD)
    rolem.write_json(OUT / rolem.SUMMARY_JSON, summary)
    return summary


def main() -> None:
    print(json.dumps(rolem.base.clean_json(build()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
