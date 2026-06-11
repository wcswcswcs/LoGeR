#!/usr/bin/env python3
"""Aggregate ACL2 v23 semantic all-memory short-rollout metrics."""

from __future__ import annotations

import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import v22_candidate_bank_report as impl  # noqa: E402
from tools import v18_true_action_report as base_impl  # noqa: E402


base_impl.FAMILY_BY_CANDIDATE.update(
    {
        "P0_01_SEMANTIC_ROLE_NOOP_IGNORED": "v23_phase0_noop",
        "P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED": "v23_phase0_noop",
        "P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY": "v23_phase0_debug",
        "FRAME_SEM_01_STRUCTURE_KEEP": "v23_single_path_frame",
        "FRAME_SEM_02_LOWSTUFF_HIGHD_SKIP": "v23_single_path_frame",
        "GLOBAL_SEM_01_STRUCTURE_KEEP": "v23_single_path_global",
        "SWA_SEM_01_STRUCTURE_LONG_KEEP": "v23_single_path_swa",
        "TTT_SEM_01_STRUCTURE_POSITIVE": "v23_single_path_ttt",
        "TTT_SEM_02_LOWSTUFF_HIGHD_SHORT_NEG": "v23_single_path_ttt",
        "ALLSEM_01_FRAME_GLOBAL_STRUCTURE_KEEP": "v23_all_memory",
        "ALLSEM_02_FRAME_GLOBAL_LOWSTUFF_HIGHD_SKIP": "v23_all_memory",
        "ALLSEM_03_FRAME_GLOBAL_SWA_STRUCTURE_LONG_KEEP": "v23_all_memory",
        "ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG": "v23_all_memory",
        "ALLSEM_05_FRAME_GLOBAL_SWA_TTT_ALL_ROLE": "v23_all_memory",
        "ALLSEM_06_ALL_ROLE_LONG_SHORT": "v23_all_memory_lifecycle",
    }
)


if __name__ == "__main__":
    raise SystemExit(impl.main())
