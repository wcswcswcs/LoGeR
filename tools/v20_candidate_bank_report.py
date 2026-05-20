#!/usr/bin/env python3
"""Aggregate ACL2 v20 short-rollout candidate metrics."""

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
        "S1_00_C23_PAST": "support_variant",
        "S1_01_C23_FULL_CHUNK": "support_variant",
        "S1_02_C23_FULL_CHUNK_NO_OVERLAP": "support_variant",
        "S1_03_C23_PAST_PLUS_FUTURE_LIGHT": "support_variant",
        "S1_04_C23_NEAR24": "support_variant",
        "SCALECOMMIT_01_PZBASIS_HARM_W0_G025": "scale_state_anchor",
        "KVS_00_C23_PAST_PAIR": "context_skip_baseline",
        "KVS_01_FRAME_EARLY_DG_Q80_HARD": "context_skip",
        "KVS_02_FRAME_EARLY_DG_Q90_HARD": "context_skip",
        "KVS_03_FRAME_EARLY_LOWSTUFF_HIGHD_HARD": "context_skip_semantic_value",
        "KVS_06_FRAME_EARLY_DG_Q90_HARD_PLUS_PAIR": "context_skip_plus_pair_bias",
        "KVS_07_CHUNK_EARLY_DG_Q90_HARD": "context_skip_chunk",
        "KVS_09_FRAME_EARLY_DG_Q90_SOFT_R025": "context_skip_soft_fallback",
        "KVS_09_FRAME_EARLY_DG_Q90_SOFT": "context_skip_soft_fallback",
        "TTTSS_03_SCALECOMMIT_DGQ90_HARD": "scale_state_context_skip_combo",
        "TTTSS_03B_SCALECOMMIT_DGQ80_HARD": "scale_state_context_skip_combo",
        "TTTSS_03C_SCALECOMMIT_DGQ80_HARD_PLUS_PAIR": "scale_state_context_skip_pair_combo",
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
    md_new = out_dir / "acl2_v20_candidate_bank_report.md"
    if md_old.exists():
        text = md_old.read_text(encoding="utf-8")
        text = text.replace(
            "# ACL2 v18 True Action Candidate Report",
            "# ACL2 v20 ContextSkip / SemanticMemory Candidate Report",
        )
        text = text.replace("v18 true-action", "v20 short-rollout")
        md_new.write_text(text, encoding="utf-8")

    summary_path = out_dir / "true_action_gate_summary.json"
    if summary_path.exists():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in rows:
            row["phase"] = "v20 contextskip semantic-memory candidate bank"
        summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    out_dir_text = _arg_value("--out-dir")
    impl.main()
    if out_dir_text:
        _postprocess_outputs(Path(out_dir_text))


if __name__ == "__main__":
    main()
