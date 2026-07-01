#!/usr/bin/env python3
"""Run accelerated v80 long-case TTT semantic TTL/control smokes.

This wraps the v78 single-window TTT runner into a multi-window, multi-GPU
queue over the v80 long bad/good case bank.  It is still diagnostic-only:
the script can report a representative smoke signal, but it never claims the
full v80 method gate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import shlex
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_v78_bad_window_tables import _evaluate_run  # noqa: E402
from tools.run_v78_ttt_long_window_regime_action_smoke import (  # noqa: E402
    CASES,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONDA,
    DEFAULT_CONFIG,
    DEFAULT_CUDA_ALLOC_CONF,
    DEFAULT_DATA_ROOT,
    _aggregate_hmc,
    _build_command,
    _jsonable,
    _parse_ints,
    _run_job,
    _window_bounds,
    _write_csv,
)


DEFAULT_CASE_BANK = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase1_three_memory_case_bank/long_five_chunk_cases.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase5_ttt_long_case_accelerated_smoke"
)
DEFAULT_CASES = ",".join(
    [
        "LW0_READPATH_NATIVE",
        "LW1_TTT_SEMANTIC_BASE",
        "LW23_TTT_OVERLAP_ROLECAL_TTL_B0_PRIOR100",
        "LW24_TTT_OVERLAP_ROLECAL_TTL_RANDOM_ROLE_B0_PRIOR100",
        "LW25_TTT_OVERLAP_ROLECAL_TTL_B0_PRIOR108",
        "LW26_TTT_OVERLAP_ROLECAL_TTL_RANDOM_ROLE_B0_PRIOR108",
    ]
)
PAIRED_RANDOM_CONTROLS = {
    "LW7_TTT_NATIVE_ORTHO_SUPPRESS_B0": "LW48_TTT_NATIVE_ORTHO_SUPPRESS_RANDOM_ROLE_B0",
    "LW8_TTT_KV_STATIC_PLUS_ORTHO_B0": "LW49_TTT_KV_STATIC_PLUS_ORTHO_RANDOM_ROLE_B0",
    "LW23_TTT_OVERLAP_ROLECAL_TTL_B0_PRIOR100": "LW24_TTT_OVERLAP_ROLECAL_TTL_RANDOM_ROLE_B0_PRIOR100",
    "LW25_TTT_OVERLAP_ROLECAL_TTL_B0_PRIOR108": "LW26_TTT_OVERLAP_ROLECAL_TTL_RANDOM_ROLE_B0_PRIOR108",
    "LW36_TTT_READHARM_LOCAL_VETO_B0": "LW37_TTT_READHARM_LOCAL_VETO_RANDOM_ROLE_B0",
    "LW38_TTT_READHARM_NEGATIVE_LOCAL_VETO_B0": "LW39_TTT_READHARM_NEGATIVE_LOCAL_VETO_RANDOM_ROLE_B0",
    "LW40_TTT_V80_HEAD_SUPPORT_VETO_B0": "LW41_TTT_V80_HEAD_SUPPORT_RANDOM_VETO_B0",
    "LW42_TTT_V80_SELECTED_WRITE_SUPPORT_VETO_B0": "LW43_TTT_V80_SELECTED_WRITE_SUPPORT_RANDOM_VETO_B0",
    "LW44_TTT_V80_FRAME279_SELECTED_WRITE_FULL_VETO_B0": (
        "LW45_TTT_V80_FRAME279_SELECTED_WRITE_FULL_RANDOM_VETO_B0"
    ),
    "LW46_TTT_V80_CHUNK009_CONTROL_DELTA_FULL_VETO_B0": (
        "LW47_TTT_V80_CHUNK009_CONTROL_DELTA_FULL_RANDOM_VETO_B0"
    ),
    "LW50_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_VETO_B0": (
        "LW51_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_RANDOM_VETO_B0"
    ),
    "LW52_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_HIGHCTRL_VETO_B0": (
        "LW53_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_HIGHCTRL_RANDOM_VETO_B0"
    ),
    "LW54_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_HARD_HIGHCTRL_VETO_B0": (
        "LW55_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_HARD_HIGHCTRL_RANDOM_VETO_B0"
    ),
    "LW56_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_HARD_ALL_HIGHCTRL_VETO": (
        "LW57_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_HARD_ALL_HIGHCTRL_RANDOM_VETO"
    ),
    "LW58_TTT_V80_SEQ05_ABSERR_SOURCE_SUPPORT_HARD_ALL_VETO": (
        "LW59_TTT_V80_SEQ05_ABSERR_SOURCE_SUPPORT_HARD_ALL_RANDOM_VETO"
    ),
}
CASES.update(
    {
        "LW48_TTT_NATIVE_ORTHO_SUPPRESS_RANDOM_ROLE_B0": {
            "description": "random same-mass role control for LW7 branch0 native-delta orthogonal suppression",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "random_same_mass",
            "diagnostic_control": True,
            "extra": [
                "--semantic_role_control_seed",
                "79048",
                "--ttt_write_native_delta_gate_mode",
                "orthogonal_suppress",
                "--ttt_write_native_delta_gate_fallback",
                "0.25",
                "--ttt_write_native_delta_gate_branch_mask",
                "0",
            ],
        },
        "LW49_TTT_KV_STATIC_PLUS_ORTHO_RANDOM_ROLE_B0": {
            "description": "random same-mass role control for LW8 branch0 K/V static gate plus native orthogonal suppression",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "random_same_mass",
            "diagnostic_control": True,
            "extra": [
                "--semantic_role_control_seed",
                "79049",
                "--ttt_write_replay_feature_gate_mode",
                "kv_frame_static_center",
                "--ttt_write_replay_feature_gate_rho",
                "0.35",
                "--ttt_write_replay_feature_gate_min",
                "0.65",
                "--ttt_write_replay_feature_gate_branch_mask",
                "0",
                "--ttt_write_native_delta_gate_mode",
                "orthogonal_suppress",
                "--ttt_write_native_delta_gate_fallback",
                "0.25",
                "--ttt_write_native_delta_gate_branch_mask",
                "0",
            ],
        },
        "LW44_TTT_V80_FRAME279_SELECTED_WRITE_FULL_VETO_B0": {
            "description": "v80 frame279 full-frame selected-write low-support TTT replay veto on branch0",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase9_seq01_ref055_v80_selected_write_support_maps_"
                    "chunk009_frame279_control_delta_full32"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "score_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "0",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "ttl_dynamic",
                "--ttt_write_transient_delta_subtract_scale",
                "1.0",
                "--ttt_write_transient_delta_branch_mask",
                "0",
                "--ttt_write_transient_delta_ttl",
                "1",
            ],
        },
        "LW45_TTT_V80_FRAME279_SELECTED_WRITE_FULL_RANDOM_VETO_B0": {
            "description": "same-mass random control for LW44 frame279 full-frame selected-write veto",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "diagnostic_control": True,
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase9_seq01_ref055_v80_selected_write_support_maps_"
                    "chunk009_frame279_control_delta_full32"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "control_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "0",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "ttl_dynamic",
                "--ttt_write_transient_delta_subtract_scale",
                "1.0",
                "--ttt_write_transient_delta_branch_mask",
                "0",
                "--ttt_write_transient_delta_ttl",
                "1",
            ],
        },
        "LW46_TTT_V80_CHUNK009_CONTROL_DELTA_FULL_VETO_B0": {
            "description": "v80 chunk009 full-frame geometry-control-delta support TTT replay veto on branch0",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase9_seq01_ref055_v80_error_semantic_support_maps_"
                    "chunk009_control_delta_full32"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "score_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "0",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "ttl_dynamic",
                "--ttt_write_transient_delta_subtract_scale",
                "1.0",
                "--ttt_write_transient_delta_branch_mask",
                "0",
                "--ttt_write_transient_delta_ttl",
                "1",
            ],
        },
        "LW47_TTT_V80_CHUNK009_CONTROL_DELTA_FULL_RANDOM_VETO_B0": {
            "description": "random support-map control for LW46 chunk009 control-delta full-frame veto",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "diagnostic_control": True,
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase9_seq01_ref055_v80_error_semantic_support_maps_"
                    "chunk009_control_delta_full32"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "control_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "0",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "ttl_dynamic",
                "--ttt_write_transient_delta_subtract_scale",
                "1.0",
                "--ttt_write_transient_delta_branch_mask",
                "0",
                "--ttt_write_transient_delta_ttl",
                "1",
            ],
        },
        "LW50_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_VETO_B0": {
            "description": "seq05 diagnostic absolute-error selected-write low-support TTT replay veto on branch0",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase10_seq05_abs_error_selected_write_support_maps_20260622_2242"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "score_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "0",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "ttl_dynamic",
                "--ttt_write_transient_delta_subtract_scale",
                "1.0",
                "--ttt_write_transient_delta_branch_mask",
                "0",
                "--ttt_write_transient_delta_ttl",
                "1",
            ],
        },
        "LW51_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_RANDOM_VETO_B0": {
            "description": "same-mass random control for LW50 seq05 diagnostic absolute-error selected-write veto",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "diagnostic_control": True,
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase10_seq05_abs_error_selected_write_support_maps_20260622_2242"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "control_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "0",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "ttl_dynamic",
                "--ttt_write_transient_delta_subtract_scale",
                "1.0",
                "--ttt_write_transient_delta_branch_mask",
                "0",
                "--ttt_write_transient_delta_ttl",
                "1",
            ],
        },
        "LW52_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_HIGHCTRL_VETO_B0": {
            "description": "seq05 absolute-error selected-write low-support TTT replay veto with high-support random control map",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase10_seq05_abs_error_selected_write_support_maps_highctrl_20260622_2322"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "score_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "0",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "ttl_dynamic",
                "--ttt_write_transient_delta_subtract_scale",
                "1.0",
                "--ttt_write_transient_delta_branch_mask",
                "0",
                "--ttt_write_transient_delta_ttl",
                "1",
            ],
        },
        "LW53_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_HIGHCTRL_RANDOM_VETO_B0": {
            "description": "same-mass high-support random control for LW52 seq05 selected-write low-support veto",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "diagnostic_control": True,
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase10_seq05_abs_error_selected_write_support_maps_highctrl_20260622_2322"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "control_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "0",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "ttl_dynamic",
                "--ttt_write_transient_delta_subtract_scale",
                "1.0",
                "--ttt_write_transient_delta_branch_mask",
                "0",
                "--ttt_write_transient_delta_ttl",
                "1",
            ],
        },
        "LW54_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_HARD_HIGHCTRL_VETO_B0": {
            "description": "hard branch0 filtered replay for seq05 selected-write low-support veto with high-support random control",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase10_seq05_abs_error_selected_write_support_maps_highctrl_20260622_2322"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "score_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "0",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "linear",
            ],
        },
        "LW55_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_HARD_HIGHCTRL_RANDOM_VETO_B0": {
            "description": "hard branch0 filtered replay high-support random control for LW54",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "diagnostic_control": True,
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase10_seq05_abs_error_selected_write_support_maps_highctrl_20260622_2322"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "control_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "0",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "linear",
            ],
        },
        "LW56_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_HARD_ALL_HIGHCTRL_VETO": {
            "description": "hard all-branch filtered replay for seq05 selected-write low-support veto with high-support random control",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase10_seq05_abs_error_selected_write_support_maps_highctrl_20260622_2322"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "score_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "all",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "linear",
            ],
        },
        "LW57_TTT_V80_SEQ05_ABSERR_SELECTED_WRITE_HARD_ALL_HIGHCTRL_RANDOM_VETO": {
            "description": "hard all-branch filtered replay high-support random control for LW56",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "diagnostic_control": True,
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase10_seq05_abs_error_selected_write_support_maps_highctrl_20260622_2322"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "control_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "all",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "linear",
            ],
        },
        "LW58_TTT_V80_SEQ05_ABSERR_SOURCE_SUPPORT_HARD_ALL_VETO": {
            "description": "hard all-branch filtered replay using full seq05 chunk83 absolute-error semantic source-support map",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase10_seq05_chunk83_abs_error_semantic_support_20260622_2236"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "score_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "all",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "linear",
            ],
        },
        "LW59_TTT_V80_SEQ05_ABSERR_SOURCE_SUPPORT_HARD_ALL_RANDOM_VETO": {
            "description": "hard all-branch filtered replay using random control from full seq05 chunk83 source-support map",
            "hybrid_memory_mode": "ttt_write_only",
            "hmc_commit_mode": "controlled",
            "semantic_prior_mode": "spg_v2",
            "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
            "semantic_memory_paths": "ttt",
            "semantic_role_control_mode": "none",
            "diagnostic_control": True,
            "extra": [
                "--semantic_ttt_overlap_support_dir",
                (
                    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
                    "report_final/phase10_seq05_chunk83_abs_error_semantic_support_20260622_2236"
                ),
                "--semantic_ttt_overlap_support_score_key",
                "control_overlap",
                "--semantic_ttt_overlap_support_scope",
                "head_overlap",
                "--semantic_ttt_overlap_support_floor",
                "0.0",
                "--ttt_write_replay_token_filter_mode",
                "scoped_dynamic_veto",
                "--ttt_write_replay_token_filter_scope",
                "all",
                "--ttt_write_replay_token_filter_threshold",
                "0.50",
                "--ttt_write_replay_token_filter_branch_mask",
                "all",
                "--ttt_write_replay_token_filter_blend",
                "1.0",
                "--ttt_write_replay_token_filter_blend_mode",
                "linear",
            ],
        },
    }
)
METRIC_KEYS = (
    "window5_joint_sim3_rmse_m",
    "window5_subchunk_scale_cv",
    "downstream_future_consistency_proxy_m",
)


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[Any]) -> float | None:
    vals = [_finite(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _median(values: list[Any]) -> float | None:
    vals = [_finite(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def _safe_improvement(base: Any, cand: Any) -> float | None:
    b = _finite(base)
    c = _finite(cand)
    if b is None or c is None or abs(b) < 1e-12:
        return None
    return float((b - c) / abs(b))


def _metric_ratio(cand: Any, base: Any) -> float | None:
    c = _finite(cand)
    b = _finite(base)
    if c is None or b is None or abs(b) < 1e-12:
        return None
    return float(c / abs(b))


def _parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _read_case_bank(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row = dict(row)
            row["seq"] = str(row.get("seq", "")).zfill(2)
            row["chunk_start"] = int(row["chunk_start"])
            row["chunk_end"] = int(row["chunk_end"])
            row["case_rank"] = int(float(row.get("case_rank") or 999999))
            row["J_long"] = _finite(row.get("J_long"))
            row["window_chunks"] = "-".join(str(i) for i in range(row["chunk_start"], row["chunk_end"] + 1))
            row["window_id"] = (
                f"seq{row['seq']}_chunks{row['chunk_start']:03d}_{row['chunk_end']:03d}_"
                f"{row.get('case_type', 'case')}_rank{row['case_rank']:02d}"
            )
            rows.append(row)
    return rows


def _select_windows(args: argparse.Namespace) -> list[dict[str, Any]]:
    seqs = {seq.zfill(2) for seq in _parse_csv_list(args.seqs)}
    case_types = set(_parse_csv_list(args.case_types))
    selected: list[dict[str, Any]] = []
    rows = [
        row
        for row in _read_case_bank(args.case_bank)
        if row["seq"] in seqs and str(row.get("case_type")) in case_types
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["seq"], str(row.get("case_type"))), []).append(row)
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: (int(r["case_rank"]), int(r["chunk_start"])))
        selected.extend(group[: int(args.max_targets_per_case_type_per_seq)])
    return selected


def _stage_c_cache_dir(args: argparse.Namespace, seq: str) -> Path:
    if args.stage_c_cache_root is None:
        return Path(f"results/kitti_preprocess/{seq}/stage_c_cache_semantic_chunks")
    root = Path(args.stage_c_cache_root)
    if root.name == "stage_c_cache_semantic_chunks":
        return root
    return root / seq / "stage_c_cache_semantic_chunks"


def _build_job_args(args: argparse.Namespace, seq: str, stage_c_cache_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        seq=seq,
        conda=args.conda,
        conda_env=args.conda_env,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        config=args.config,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        stage_c_cache_dir=stage_c_cache_dir,
    )


def _window_pair_future_proxy(pairs: list[dict[str, Any]], start_chunk: int, end_chunk: int) -> float | None:
    vals: list[Any] = []
    for row in pairs:
        left = row.get("start_chunk_id")
        right = row.get("end_chunk_id")
        if left is None or right is None:
            continue
        if int(left) >= int(start_chunk) and int(right) <= int(end_chunk):
            vals.append(row.get("tail3_to_future_from_boundary_sim3_rmse_m"))
    return _mean(vals)


def _evaluate_jobs(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    for job in jobs:
        seq = str(job["seq"]).zfill(2)
        case_name = str(job["case"])
        out_dir = Path(job["out_dir"])
        row: dict[str, Any] = {
            "job_id": job["job_id"],
            "window_id": job["window_id"],
            "seq": seq,
            "case_type": job["case_type"],
            "source_case_rank": job.get("source_case_rank"),
            "source_J_long": job.get("source_J_long"),
            "target_window_chunks": job["target_window_chunks"],
            "case": case_name,
            "description": CASES[case_name].get("description"),
            "diagnostic_control": bool(CASES[case_name].get("diagnostic_control", False)),
            "returncode": int(job.get("returncode") if job.get("returncode") is not None else -999),
            "duration_sec": job.get("duration_sec"),
            "trajectory": str(out_dir / f"{seq}.txt"),
            "run_log": str(out_dir / "run.log"),
            "hmc_state_hash": str(out_dir / "hmc_state_hash.jsonl"),
        }
        if int(row["returncode"]) == 0 and Path(row["trajectory"]).exists():
            single, pairs, window5, summary = _evaluate_run(
                name=case_name,
                seq=seq,
                path=Path(row["trajectory"]),
                gt_root=args.data_root / "poses",
                chunk_size=int(args.chunk_size),
                overlap=int(args.chunk_overlap),
                min_coverage=float(args.min_coverage),
            )
            target = None
            for item in window5:
                if str(item.get("window_chunks")) == str(job["target_window_chunks"]):
                    target = item
                    break
            if target is None and window5:
                target = window5[0]
                row["target_window_fallback_used"] = True
            if target is not None:
                row.update(target)
            row["single_chunk_eval_rows"] = len(single)
            row["pair_eval_rows"] = len(pairs)
            row["window5_eval_rows"] = len(window5)
            row["trajectory_frame_count"] = summary.get("frame_count")
            row["downstream_future_consistency_proxy_m"] = _window_pair_future_proxy(
                pairs,
                int(job["chunk_start"]),
                int(job["chunk_end"]),
            )
            row.update(_aggregate_hmc(out_dir))
        metric_rows.append(row)

    by_window_case = {(row["window_id"], row["case"]): row for row in metric_rows}
    for row in metric_rows:
        base = by_window_case.get((row["window_id"], args.baseline))
        native = by_window_case.get((row["window_id"], args.native_baseline))
        control = by_window_case.get((row["window_id"], PAIRED_RANDOM_CONTROLS.get(str(row["case"]), "")))
        if base:
            proxy_parts = []
            for key in METRIC_KEYS:
                imp = _safe_improvement(base.get(key), row.get(key))
                ratio = _metric_ratio(row.get(key), base.get(key))
                row[f"{key}_improvement_vs_ttt_baseline_ratio"] = imp
                row[f"{key}_ratio_vs_ttt_baseline"] = ratio
                if ratio is not None:
                    proxy_parts.append(ratio)
            row["J_proxy_ratio_vs_ttt_baseline"] = _mean(proxy_parts)
            if row.get("J_proxy_ratio_vs_ttt_baseline") is not None:
                row["J_proxy_improvement_vs_ttt_baseline_ratio"] = 1.0 - float(
                    row["J_proxy_ratio_vs_ttt_baseline"]
                )
        if native:
            row["window5_joint_sim3_rmse_improvement_vs_native_ratio"] = _safe_improvement(
                native.get("window5_joint_sim3_rmse_m"),
                row.get("window5_joint_sim3_rmse_m"),
            )
        if control:
            control_parts = []
            for key in METRIC_KEYS:
                imp = _safe_improvement(control.get(key), row.get(key))
                ratio = _metric_ratio(row.get(key), control.get(key))
                row[f"{key}_improvement_vs_paired_random_control_ratio"] = imp
                if ratio is not None:
                    control_parts.append(ratio)
            row["J_proxy_ratio_vs_paired_random_control"] = _mean(control_parts)
            if row.get("J_proxy_ratio_vs_paired_random_control") is not None:
                row["J_proxy_improvement_vs_paired_random_control_ratio"] = 1.0 - float(
                    row["J_proxy_ratio_vs_paired_random_control"]
                )

    decisions: dict[str, Any] = {}
    candidate_cases = [
        case
        for case in _parse_csv_list(args.cases)
        if case not in {args.baseline, args.native_baseline} and case in PAIRED_RANDOM_CONTROLS
    ]
    for case_name in candidate_cases:
        rows = [row for row in metric_rows if str(row.get("case")) == case_name and int(row.get("returncode") or 0) == 0]
        bad_rows = [row for row in rows if str(row.get("case_type")) == "bad"]
        good_rows = [row for row in rows if str(row.get("case_type")) == "good"]
        bad_improvements = [row.get("J_proxy_improvement_vs_ttt_baseline_ratio") for row in bad_rows]
        good_improvements = [row.get("J_proxy_improvement_vs_ttt_baseline_ratio") for row in good_rows]
        control_improvements = [row.get("J_proxy_improvement_vs_paired_random_control_ratio") for row in bad_rows]
        bad_median = _median(bad_improvements)
        good_worst = min([float(v) for v in good_improvements if _finite(v) is not None], default=None)
        control_median = _median(control_improvements)
        decisions[case_name] = {
            "candidate": case_name,
            "paired_random_control": PAIRED_RANDOM_CONTROLS.get(case_name),
            "bad_eval_n": len(bad_rows),
            "good_eval_n": len(good_rows),
            "bad_median_J_proxy_improvement_vs_ttt_baseline_ratio": bad_median,
            "good_worst_J_proxy_improvement_vs_ttt_baseline_ratio": good_worst,
            "bad_median_J_proxy_improvement_vs_paired_random_control_ratio": control_median,
            "bad_improves_ge_min": bool(bad_median is not None and bad_median >= float(args.min_bad_improvement)),
            "good_not_worse_than_limit": bool(
                good_worst is not None and good_worst >= -float(args.max_good_worsen)
            ),
            "beats_paired_random_control_on_bad": bool(control_median is not None and control_median > 0.0),
            "representative_smoke_signal_pass": bool(
                bad_median is not None
                and bad_median >= float(args.min_bad_improvement)
                and good_worst is not None
                and good_worst >= -float(args.max_good_worsen)
                and control_median is not None
                and control_median > 0.0
            ),
            "method_gate_claimed": False,
            "full_v80_gate_claimed": False,
        }

    summary = {
        "schema": "acl2_v80_ttt_long_case_accelerated_smoke_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "full_v80_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "runtime_promotion_blocker": (
            "representative accelerated smoke only; full v80 Phase5 requires all planned "
            "long cases and plan controls before promotion"
        ),
        "args": _jsonable(vars(args)),
        "metric_keys": METRIC_KEYS,
        "baseline": args.baseline,
        "native_baseline": args.native_baseline,
        "paired_random_controls": PAIRED_RANDOM_CONTROLS,
        "metrics": metric_rows,
        "decisions": decisions,
        "any_representative_smoke_signal_pass": bool(
            any(bool(item.get("representative_smoke_signal_pass")) for item in decisions.values())
        ),
        "all_jobs_ok": bool(all(int(job.get("returncode") or 0) == 0 for job in jobs)),
    }
    metrics_json = args.output_root / "ttt_long_case_accelerated_metrics.json"
    metrics_csv = args.output_root / "ttt_long_case_accelerated_metrics.csv"
    decision_json = args.output_root / "ttt_long_case_accelerated_decision.json"
    metrics_json.write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(metrics_csv, metric_rows)
    decision_json.write_text(
        json.dumps(
            _jsonable(
                {
                    "diagnostic_only": True,
                    "method_gate_claimed": False,
                    "full_v80_gate_claimed": False,
                    "runtime_promotion_allowed": False,
                    "decisions": decisions,
                    "any_representative_smoke_signal_pass": summary["any_representative_smoke_signal_pass"],
                    "all_jobs_ok": summary["all_jobs_ok"],
                    "metrics_json": str(metrics_json),
                    "metrics_csv": str(metrics_csv),
                }
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            _jsonable(
                {
                    "all_jobs_ok": summary["all_jobs_ok"],
                    "any_representative_smoke_signal_pass": summary["any_representative_smoke_signal_pass"],
                    "decision_json": decision_json,
                    "metrics_json": metrics_json,
                }
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-bank", type=Path, default=DEFAULT_CASE_BANK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seqs", default="00,01,02,05")
    parser.add_argument("--case-types", default="bad,good")
    parser.add_argument("--max-targets-per-case-type-per-seq", type=int, default=1)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stage-c-cache-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--cuda-alloc-conf", default=DEFAULT_CUDA_ALLOC_CONF)
    parser.add_argument(
        "--disable-ttt-compile",
        type=int,
        default=1,
        help="Set LOGER_TTT_DISABLE_COMPILE=1 for child runs to avoid long inductor compile stalls.",
    )
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--baseline", default="LW1_TTT_SEMANTIC_BASE")
    parser.add_argument("--native-baseline", default="LW0_READPATH_NATIVE")
    parser.add_argument("--min-bad-improvement", type=float, default=0.05)
    parser.add_argument("--max-good-worsen", type=float, default=0.02)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rollouts_root = args.output_root / "rollouts"
    gpus = _parse_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    cases = _parse_csv_list(args.cases)
    unknown = [case for case in cases if case not in CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(CASES)}")
    for required in (args.baseline, args.native_baseline):
        if required not in cases:
            raise ValueError(f"required baseline {required!r} is not included in --cases")
    selected_windows = _select_windows(args)
    if not selected_windows:
        raise ValueError("no windows selected from case bank")

    jobs: list[dict[str, Any]] = []
    for window in selected_windows:
        seq = str(window["seq"]).zfill(2)
        chunks = list(range(int(window["chunk_start"]), int(window["chunk_end"]) + 1))
        bounds = _window_bounds(chunks, int(args.chunk_size), int(args.chunk_overlap))
        stage_dir = _stage_c_cache_dir(args, seq)
        job_args = _build_job_args(args, seq, stage_dir)
        for case_name in cases:
            out_dir = rollouts_root / str(window["window_id"]) / case_name
            cmd = _build_command(job_args, case_name, out_dir, bounds)
            traj = out_dir / f"{seq}.txt"
            hmc = out_dir / "hmc_state_hash.jsonl"
            skipped = bool(args.skip_existing and traj.exists() and hmc.exists())
            job_id = f"{window['window_id']}/{case_name}"
            idx = len(jobs)
            jobs.append(
                {
                    "job_id": job_id,
                    "seq": seq,
                    "case": case_name,
                    "case_config": CASES[case_name],
                    "case_type": str(window.get("case_type")),
                    "source_case_rank": int(window.get("case_rank") or 0),
                    "source_J_long": window.get("J_long"),
                    "window_id": str(window["window_id"]),
                    "target_window_chunks": str(window["window_chunks"]),
                    "chunk_start": int(window["chunk_start"]),
                    "chunk_end": int(window["chunk_end"]),
                    "window_bounds": bounds,
                    "stage_c_cache_dir": str(stage_dir),
                    "gpu": int(gpus[idx % len(gpus)]),
                    "out_dir": str(out_dir),
                    "cmd": cmd,
                    "cmd_shell": shlex.join(cmd),
                    "workdir": str(args.workdir),
                    "cuda_alloc_conf": str(args.cuda_alloc_conf),
                    "disable_ttt_compile": bool(args.disable_ttt_compile),
                    "skipped": skipped,
                    "returncode": 0 if skipped else None,
                }
            )

    manifest = {
        "schema": "acl2_v80_ttt_long_case_accelerated_smoke_manifest_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "full_v80_gate_claimed": False,
        "args": _jsonable(vars(args)),
        "selected_windows": selected_windows,
        "cases": {name: CASES[name] for name in cases},
        "jobs": jobs,
    }
    manifest_path = args.output_root / "ttt_long_case_accelerated_run_manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            _jsonable(
                {
                    "selected_windows": len(selected_windows),
                    "planned_jobs": len(jobs),
                    "manifest": manifest_path,
                    "dry_run": bool(args.dry_run),
                }
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if args.dry_run:
        return

    run_jobs = [job for job in jobs if not bool(job.get("skipped"))]
    completed: list[dict[str, Any]] = [job for job in jobs if bool(job.get("skipped"))]
    completed_lock = threading.Lock()
    jobs_by_gpu: dict[int, list[dict[str, Any]]] = {int(gpu): [] for gpu in gpus}
    for job in run_jobs:
        jobs_by_gpu[int(job["gpu"])].append(job)

    def run_gpu_queue(queue: list[dict[str, Any]]) -> None:
        for job in queue:
            result = _run_job(job)
            with completed_lock:
                completed.append(result)
            print(
                f"finished job={result['job_id']} gpu={result['gpu']} "
                f"returncode={result['returncode']} duration_sec={result['duration_sec']:.1f}"
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(gpus))) as pool:
        futures = [pool.submit(run_gpu_queue, queue) for queue in jobs_by_gpu.values() if queue]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    completed_by_id = {str(job["job_id"]): job for job in completed}
    ordered_completed = [completed_by_id.get(str(job["job_id"]), job) for job in jobs]
    run_summary = {
        "schema": "acl2_v80_ttt_long_case_accelerated_smoke_run_summary_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "full_v80_gate_claimed": False,
        "jobs": ordered_completed,
        "all_jobs_ok": bool(all(int(job.get("returncode") or 0) == 0 for job in ordered_completed)),
    }
    run_summary_path = args.output_root / "ttt_long_case_accelerated_run_summary.json"
    run_summary_path.write_text(
        json.dumps(_jsonable(run_summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote_run_summary={run_summary_path} all_jobs_ok={run_summary['all_jobs_ok']}")
    if not args.no_evaluate:
        _evaluate_jobs(args, ordered_completed)


if __name__ == "__main__":
    main()
