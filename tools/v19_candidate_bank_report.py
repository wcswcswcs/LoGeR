#!/usr/bin/env python3
"""Aggregate ACL2 v19 trajectory-state short-rollout candidate metrics.

This reuses the v18 horizon reporter's audited same-frame-intersection
metric code, but registers the v19 scale-state candidate families and
renames the report artifacts for v19.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import v18_true_action_report as impl


impl.FAMILY_BY_CANDIDATE.update(
    {
        "K1_H9": "baseline",
        "SCALETTT_01_SPECIAL_TOKEN_W0_A005": "scale_state_ttt_replay",
        "SCALETTT_02_STRUCTURE_LOWDG_W0_A005": "scale_state_ttt_replay",
        "SCALETTT_03_OVERLAP_STATIC_W0W2_A005": "scale_state_ttt_replay",
        "SCALETTT_04_STRUCTURE_LOWDG_W0_A010": "scale_state_ttt_replay",
        "SCALETTT_05_OVERLAP_STATIC_W0W2_A010": "scale_state_ttt_replay",
        "SCALECOMMIT_01_PZBASIS_HARM_W0_G025": "scale_state_commit_modulation",
        "SCALECOMMIT_02_AUXGEO_OVERLAP_W0_G025": "scale_state_commit_modulation",
        "SCALECOMMIT_03_HIST_DELTA_W0_G025": "scale_state_commit_modulation",
    }
)


def _arg_value(name: str) -> str | None:
    if name not in sys.argv:
        return None
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        return None
    return sys.argv[idx + 1]


def _postprocess_outputs(out_dir: Path) -> None:
    md_old = out_dir / "acl2_v18_true_action_report.md"
    md_new = out_dir / "acl2_v19_candidate_bank_report.md"
    if md_old.exists():
        text = md_old.read_text(encoding="utf-8")
        text = text.replace(
            "# ACL2 v18 True Action Candidate Report",
            "# ACL2 v19 Trajectory-State Candidate Bank Report",
        )
        text = text.replace("v18 true-action", "v19 trajectory-state")
        md_new.write_text(text, encoding="utf-8")

    summary_path = out_dir / "true_action_gate_summary.json"
    if summary_path.exists():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in rows:
            row["phase"] = "v19 trajectory-state candidate bank"
        summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    out_dir_text = _arg_value("--out-dir")
    impl.main()
    if out_dir_text:
        _postprocess_outputs(Path(out_dir_text))


if __name__ == "__main__":
    main()
