#!/usr/bin/env python3
"""Summarize v107R Stage7B keyframe-aware full-sequence policy ATE."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V107R = ROOT / "results/acl2_v107r_lingbot_semantic_memory_decision_cue_operation_control"
OUT = V107R / "stage7b_full_sequence_keyframe_aware_policy"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v107r_stage7_full_sequence_selected_policy_summary as base  # noqa: E402


def main() -> None:
    base.OUT = OUT
    base.CONFIG_ROWS = OUT / "action_config_rows.csv"
    base.RUN_RESULTS = OUT / "run_results.csv"
    base.WORKSPACE = OUT / "workspace"
    summary = base.build()
    summary["schema"] = "acl2_v107r_stage7b_keyframe_aware_full_sequence_summary_v1"
    summary["policy_boundary"] = (
        "Stage7B snaps Stage6 semantic-risk selected frames to nearby full-sequence "
        "base keyframes before forcing non-keyframe cache behavior."
    )
    base.write_json(OUT / "stage7b_summary.json", summary)
    base.write_json(base.STAGE7_COMPAT / "stage7_summary.json", summary)
    print(json.dumps(base.clean_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
